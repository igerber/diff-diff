#!/usr/bin/env python3
"""Coverage Monte Carlo study for SDID variance methods.

Generates null-panel Monte Carlo samples (no treatment effect) across
four representative DGPs (``balanced``, ``unbalanced``, ``aer63``,
``stratified_survey``), fits SyntheticDiD under each of the three
variance methods (placebo, bootstrap, jackknife), and records rejection
rates at α ∈ {0.01, 0.05, 0.10} plus the ratio of mean estimated SE to
the empirical sampling SD of τ̂.

The ``stratified_survey`` DGP runs bootstrap and jackknife; placebo is
skipped because its cohort packs all treated units into stratum 1,
which has 0 never-treated units, so the stratified-permutation
allocator is structurally infeasible on this DGP (raises Case B —
treated-containing stratum with zero controls — at fit-time).
Jackknife is reported
with a documented anti-conservatism caveat — with only 2 PSUs per
stratum, the stratified PSU-level jackknife formula has 1 effective DoF
per stratum, a known limitation (see REGISTRY §SyntheticDiD "Note
(survey + jackknife composition)"). The harness skips unsupported
method × DGP cells via the per-DGP ``survey_design_factory`` in the
``DGPSpec`` registry.

The output JSON underwrites the calibration table in
``docs/methodology/REGISTRY.md`` §SyntheticDiD, including the
stratified-survey bootstrap calibration gate [0.02, 0.10] that validates
the hybrid pairs-bootstrap + Rao-Wu weighted-FW composition.

Usage:
    # Full run (~15–40 min on M-series Mac with Rust backend; AER §6.3 refit
    # is the long tail at ~37 min. Matches the wall-clock wording in
    # REGISTRY.md §SyntheticDiD coverage MC note.)
    python benchmarks/python/coverage_sdid.py \\
        --n-seeds 500 --n-bootstrap 200 \\
        --output benchmarks/data/sdid_coverage.json

    # Single DGP (chunkable across sessions)
    python benchmarks/python/coverage_sdid.py \\
        --dgps balanced --n-seeds 500 \\
        --output benchmarks/data/sdid_coverage_balanced.json

    # Quick smoke-check (~2 minutes at these settings)
    python benchmarks/python/coverage_sdid.py \\
        --n-seeds 20 --n-bootstrap 40 \\
        --output /tmp/sdid_coverage_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _get_backend_from_args() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", default="auto", choices=["auto", "python", "rust"])
    args, _ = parser.parse_known_args()
    return args.backend


_requested_backend = _get_backend_from_args()
if _requested_backend in ("python", "rust"):
    os.environ["DIFF_DIFF_BACKEND"] = _requested_backend


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import diff_diff  # noqa: E402 — imports after env var gate the backend
from diff_diff import HAS_RUST_BACKEND, SyntheticDiD  # noqa: E402

ALL_METHODS = ("placebo", "bootstrap", "jackknife")
ALL_DGPS = ("balanced", "unbalanced", "aer63", "stratified_survey")
ALPHAS = (0.01, 0.05, 0.10)


@dataclass
class DGPSpec:
    name: str
    description: str
    # generator returns (DataFrame, post_periods)
    generator: Callable[[int], Tuple[pd.DataFrame, List[int]]]
    # Optional factory: returns (SurveyDesign, methods_supported_set) given a
    # DataFrame. None for non-survey DGPs. The methods_supported set lets the
    # harness skip the methods that raise NotImplementedError on this design
    # (e.g., placebo/jackknife under strata/PSU/FPC). For non-survey DGPs all
    # methods are supported.
    survey_design_factory: Optional[Callable[[pd.DataFrame], Tuple[Any, Tuple[str, ...]]]] = None


def _balanced_dgp(seed: int) -> Tuple[pd.DataFrame, List[int]]:
    """Balanced / exchangeable panel with null effect.

    Small enough to run quickly; controls are exchangeable by construction
    (iid unit FE draws), supporting tight placebo-tracks-refit assertions.
    """
    rng = np.random.default_rng(seed)
    n_control, n_treated = 20, 3
    n_pre, n_post = 8, 4
    rows = []
    for unit in range(n_control + n_treated):
        unit_fe = rng.normal(0, 1.5)
        for t in range(n_pre + n_post):
            y = 10.0 + unit_fe + 0.25 * t + rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": unit,
                    "period": t,
                    "treated": int(unit >= n_control),
                    "outcome": y,
                }
            )
    return pd.DataFrame(rows), list(range(n_pre, n_pre + n_post))


def _unbalanced_dgp(seed: int) -> Tuple[pd.DataFrame, List[int]]:
    """Unbalanced panel: more treated, uneven unit-FE variance.

    Exercises a regime where placebo's ``n_control > n_treated`` constraint
    is tight but still satisfied, and where refit's re-estimation has more
    signal to distinguish from the fixed-weight shortcut.
    """
    rng = np.random.default_rng(seed)
    n_control, n_treated = 30, 8
    n_pre, n_post = 10, 6
    rows = []
    for unit in range(n_control + n_treated):
        # Treated units have higher FE variance (non-exchangeable cohorts)
        scale = 2.5 if unit >= n_control else 1.0
        unit_fe = rng.normal(0, scale)
        for t in range(n_pre + n_post):
            y = 10.0 + unit_fe + 0.20 * t + rng.normal(0, 0.6)
            rows.append(
                {
                    "unit": unit,
                    "period": t,
                    "treated": int(unit >= n_control),
                    "outcome": y,
                }
            )
    return pd.DataFrame(rows), list(range(n_pre, n_pre + n_post))


def _aer63_dgp(seed: int) -> Tuple[pd.DataFrame, List[int]]:
    """Arkhangelsky et al. (2021) AER §6.3 non-exchangeable DGP.

    N=100, N1=20, T=120, T1=5, rank=2 factor structure, iid N(0, σ²)
    errors with σ=2. Set to τ=0 here (H0 calibration). The paper reports
    95% jackknife coverage at 98% on iid errors and 93% on AR(1) ρ=0.7;
    we reproduce the iid setting.
    """
    rng = np.random.default_rng(seed)
    n_control, n_treated = 80, 20
    n_pre, n_post = 115, 5  # T=120, T1=5 (post)
    n_factors = 2
    sigma = 2.0

    # Factor loadings per unit (rank-2). Treated units have shifted loadings,
    # which is what makes the DGP non-exchangeable.
    loadings_control = rng.normal(0, 1.0, size=(n_control, n_factors))
    loadings_treated = rng.normal(0.5, 1.0, size=(n_treated, n_factors))
    loadings = np.vstack([loadings_control, loadings_treated])
    factors = rng.normal(0, 1.0, size=(n_pre + n_post, n_factors))

    unit_fe = rng.normal(0, 1.0, size=(n_control + n_treated,))
    time_fe = rng.normal(0, 0.3, size=(n_pre + n_post,))

    rows = []
    for unit in range(n_control + n_treated):
        for t in range(n_pre + n_post):
            iif = float(loadings[unit] @ factors[t])
            y = unit_fe[unit] + time_fe[t] + iif + rng.normal(0, sigma)
            rows.append(
                {
                    "unit": unit,
                    "period": t,
                    "treated": int(unit >= n_control),
                    "outcome": y,
                }
            )
    return pd.DataFrame(rows), list(range(n_pre, n_pre + n_post))


def _stratified_survey_dgp(seed: int) -> Tuple[pd.DataFrame, List[int]]:
    """BRFSS/ACS-style stratified survey panel, null treatment (PR #352).

    N=40 (10 per PSU × 4 PSUs across 2 strata), T=12 (6 pre, 6 post),
    moderate weight variation (Kish DEFF ≈ 1.4), psu_re_sd=1.5 (modest
    ICC). Each unit is a respondent with constant per-unit survey
    weight, stratum, and PSU columns. Used to validate the SDID
    survey-bootstrap calibration: the bootstrap row should land near
    nominal at α=0.05 (PR #352 §3c calibration gate, [0.02, 0.10]).
    """
    from diff_diff.prep_dgp import generate_survey_did_data
    df = generate_survey_did_data(
        n_units=40,
        n_periods=12,
        cohort_periods=[7],
        never_treated_frac=0.2,
        treatment_effect=0.0,  # null for coverage MC
        n_strata=2,
        psu_per_stratum=2,
        fpc_per_stratum=200.0,
        weight_variation="moderate",
        psu_re_sd=1.5,
        psu_period_factor=0.5,
        seed=seed,
    )
    # generate_survey_did_data emits per-observation 'treated' (post-only
    # for treated units); SDID requires a unit-level ever-treated indicator
    # (constant across time). Derive from 'first_treat' (cohort, 0 for
    # never-treated). Periods are 1-indexed (prep_dgp.py L1211-L1212), so
    # cohort 7 with n_periods=12 → post = [7, 8, 9, 10, 11, 12] (6 post
    # periods). Derive from df["period"].max() so any change to n_periods
    # propagates (PR #355 R13 P1 — the hard-coded range(7, 12) dropped
    # period 12 into the pre window, contaminating calibration).
    df = df.copy()
    df["treated"] = (df["first_treat"] > 0).astype(int)
    cohort_onset = 7
    period_max = int(df["period"].max())
    post_periods = list(range(cohort_onset, period_max + 1))
    return df, post_periods


def _stratified_survey_design(df: pd.DataFrame) -> Tuple[Any, Tuple[str, ...]]:
    """Build the SurveyDesign for the stratified_survey DGP.

    Methods supported on this DGP:
    * **bootstrap** — weighted-FW + Rao-Wu (PR #355). Calibration
      validated here.
    * **jackknife** — PSU-level LOO with stratum aggregation (Rust &
      Rao 1996). Reported here with a known anti-conservatism caveat:
      with ``psu_per_stratum=2``, within-stratum jackknife has only
      ``n_h - 1 = 1`` effective DoF per stratum, which is a well-
      documented limitation of the stratified jackknife formula when
      PSU counts are low. The reported ``se_over_truesd`` is expected
      to land below 1; this is not a bug — users needing tight SE
      calibration with few PSUs should prefer ``bootstrap``.
    * **placebo** — NOT supported on this DGP: the treated cohort packs
      into stratum 1 (which has 0 never-treated units by construction),
      so the stratified-permutation allocator raises Case B (zero
      controls in a treated-containing stratum) at fit-time. This is a
      property of the DGP, not of the placebo allocator; the placebo
      survey method is exercised by
      ``tests/test_survey_phase5.py::TestSDIDSurveyPlaceboFullDesign``.
    """
    from diff_diff import SurveyDesign
    return (
        SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc"),
        ("bootstrap", "jackknife"),
    )


DGPS: Dict[str, DGPSpec] = {
    "balanced": DGPSpec(
        "balanced",
        "Balanced / exchangeable: N_co=20, N_tr=3, T_pre=8, T_post=4",
        _balanced_dgp,
    ),
    "unbalanced": DGPSpec(
        "unbalanced",
        "Unbalanced: N_co=30, N_tr=8, heterogeneous unit-FE variance",
        _unbalanced_dgp,
    ),
    "aer63": DGPSpec(
        "aer63",
        "Arkhangelsky et al. (2021) AER §6.3: N=100, N1=20, T=120, T1=5, rank=2, σ=2",
        _aer63_dgp,
    ),
    "stratified_survey": DGPSpec(
        "stratified_survey",
        "BRFSS-style: N=40, strata=2, PSU=2/stratum, psu_re_sd=1.5 (PR #352)",
        _stratified_survey_dgp,
        survey_design_factory=_stratified_survey_design,
    ),
}


def _fit_one(
    method: str,
    df: pd.DataFrame,
    post_periods: List[int],
    n_bootstrap: int,
    seed: int,
    survey_design: Optional[Any] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Fit SDID and return (att, se, p_value); (None, None, None) on failure.

    For survey DGPs the harness passes a SurveyDesign via ``survey_design``;
    ``fit()`` routes strata/PSU/FPC designs through the method-specific
    survey variance path — bootstrap (PR #355 weighted-FW + Rao-Wu),
    placebo (stratified permutation + weighted-FW), or jackknife (PSU-
    level LOO with stratum aggregation). The DGP's
    ``survey_design_factory`` declares which methods are supported on
    that specific DGP, so the caller skips unsupported methods entirely
    rather than catching the resulting NotImplementedError / Case B-D
    ValueError here.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_kwargs = dict(
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=post_periods,
            )
            if survey_design is not None:
                fit_kwargs["survey_design"] = survey_design
            r = SyntheticDiD(
                variance_method=method,
                n_bootstrap=n_bootstrap,
                seed=seed,
            ).fit(df, **fit_kwargs)
        att = float(r.att) if np.isfinite(r.att) else None
        se = float(r.se) if np.isfinite(r.se) else None
        p_value = float(r.p_value) if np.isfinite(r.p_value) else None
        return att, se, p_value
    except (ValueError, RuntimeError, NotImplementedError):
        return None, None, None


def _summarize(
    atts: np.ndarray,
    ses: np.ndarray,
    p_values: np.ndarray,
) -> Dict[str, Any]:
    """Reduce per-seed (att, se, p_value) arrays to summary stats.

    Unsupported / all-failed cells serialize as ``null`` rather than
    ``float('nan')`` so the output is strict JSON (PR #355 R11 P3).
    """
    finite = np.isfinite(atts) & np.isfinite(ses) & np.isfinite(p_values)
    n_successful = int(finite.sum())
    if n_successful == 0:
        return {
            "n_successful_fits": 0,
            "rejection_rate": {f"{a:.2f}": None for a in ALPHAS},
            "mean_se": None,
            "true_sd_tau_hat": None,
            "se_over_truesd": None,
        }
    atts_f = atts[finite]
    ses_f = ses[finite]
    ps_f = p_values[finite]
    rejection = {f"{a:.2f}": float(np.mean(ps_f < a)) for a in ALPHAS}
    mean_se = float(ses_f.mean())
    true_sd = float(atts_f.std(ddof=1)) if n_successful > 1 else None
    if true_sd is not None and np.isfinite(true_sd) and true_sd > 0:
        ratio: Optional[float] = float(mean_se / true_sd)
    else:
        ratio = None
    return {
        "n_successful_fits": n_successful,
        "rejection_rate": rejection,
        "mean_se": mean_se,
        "true_sd_tau_hat": true_sd,
        "se_over_truesd": ratio,
    }


def _run_dgp(
    name: str,
    spec: DGPSpec,
    n_seeds: int,
    n_bootstrap: int,
    methods: Tuple[str, ...],
) -> Dict[str, Any]:
    """Run all methods × n_seeds for one DGP. Returns summary dict.

    For survey DGPs (``spec.survey_design_factory is not None``) the harness
    constructs the SurveyDesign once per seed (it depends only on the column
    names, not the DataFrame contents) and skips methods not in
    ``supported_methods`` — those rows in the artifact have
    ``n_successful_fits=0``.
    """
    print(f"\n=== DGP: {name} ({spec.description}) ===", flush=True)

    # Preallocate per-method arrays
    atts = {m: np.full(n_seeds, np.nan) for m in methods}
    ses = {m: np.full(n_seeds, np.nan) for m in methods}
    pvs = {m: np.full(n_seeds, np.nan) for m in methods}

    start = time.time()
    for seed in range(n_seeds):
        df, post = spec.generator(seed)
        if spec.survey_design_factory is not None:
            survey_design, supported_methods = spec.survey_design_factory(df)
        else:
            survey_design = None
            supported_methods = methods
        for method in methods:
            if method not in supported_methods:
                # Method-specific guard fires (e.g., placebo + strata).
                # Leave NaN; the summary will report n_successful_fits=0.
                continue
            att, se, p = _fit_one(method, df, post, n_bootstrap, seed, survey_design)
            if att is not None:
                atts[method][seed] = att
            if se is not None:
                ses[method][seed] = se
            if p is not None:
                pvs[method][seed] = p
        if (seed + 1) % 10 == 0 or seed + 1 == n_seeds:
            elapsed = time.time() - start
            pace = (seed + 1) / max(elapsed, 1e-9)
            eta = (n_seeds - seed - 1) / max(pace, 1e-9)
            print(
                f"  seed {seed + 1}/{n_seeds}  " f"(elapsed {elapsed:.1f}s, eta {eta:.1f}s)",
                flush=True,
            )

    summaries: Dict[str, Any] = {m: _summarize(atts[m], ses[m], pvs[m]) for m in methods}
    summaries["_elapsed_sec"] = round(time.time() - start, 2)
    return summaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "SDID coverage Monte Carlo study — rejection rates across "
            "placebo / bootstrap / jackknife on representative DGPs."
        )
    )
    p.add_argument("--n-seeds", type=int, default=500, help="MC replications per DGP (default 500)")
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Bootstrap replications per fit (default 200)",
    )
    p.add_argument(
        "--dgps",
        type=str,
        default=",".join(ALL_DGPS),
        help=f"Comma-separated DGPs to run (default all: {','.join(ALL_DGPS)})",
    )
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_METHODS),
        help=f"Comma-separated variance methods (default all: {','.join(ALL_METHODS)})",
    )
    p.add_argument("--output", type=str, required=True, help="Output JSON path")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "python", "rust"],
        help="Backend selection (parsed before diff_diff import)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dgps_requested = tuple(d.strip() for d in args.dgps.split(",") if d.strip())
    methods_requested = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    unknown_dgps = [d for d in dgps_requested if d not in DGPS]
    if unknown_dgps:
        sys.exit(f"Unknown DGP(s): {unknown_dgps}. Known: {list(DGPS)}")
    unknown_methods = [m for m in methods_requested if m not in ALL_METHODS]
    if unknown_methods:
        sys.exit(f"Unknown method(s): {unknown_methods}. Known: {ALL_METHODS}")

    actual_backend = "rust" if HAS_RUST_BACKEND else "python"
    lib_version = getattr(diff_diff, "__version__", "unknown")

    print(
        f"SDID coverage MC: n_seeds={args.n_seeds}, n_bootstrap={args.n_bootstrap}, "
        f"backend={actual_backend}, version={lib_version}",
        flush=True,
    )
    print(f"DGPs: {list(dgps_requested)}", flush=True)
    print(f"Methods: {list(methods_requested)}", flush=True)

    overall_start = time.time()
    per_dgp = {}
    for dgp_name in dgps_requested:
        per_dgp[dgp_name] = _run_dgp(
            dgp_name,
            DGPS[dgp_name],
            args.n_seeds,
            args.n_bootstrap,
            methods_requested,
        )

    output = {
        "metadata": {
            "n_seeds": args.n_seeds,
            "n_bootstrap": args.n_bootstrap,
            "library_version": lib_version,
            "backend": actual_backend,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_elapsed_sec": round(time.time() - overall_start, 2),
            "methods": list(methods_requested),
            "alphas": list(ALPHAS),
        },
        "dgps": {name: DGPS[name].description for name in dgps_requested},
        "per_dgp": per_dgp,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # ``allow_nan=False`` so any stray float('nan') / inf fails loudly
        # instead of serializing as bare ``NaN`` / ``Infinity`` tokens
        # (non-strict JSON that strict parsers reject; PR #355 R11 P3).
        # ``_summarize`` returns ``None`` for unsupported / all-failed
        # cells precisely so this gate doesn't trip.
        json.dump(output, f, indent=2, allow_nan=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
