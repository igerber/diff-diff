#!/usr/bin/env python3
"""Monte-Carlo coverage study for the LP-DiD regression-adjustment (RA) SE.

The canonical RA standard error is Stata ``teffects ra ... atet vce(cluster)``
only - no R package computes it (``alexCardazzi`` uses direct covariate
inclusion, not RA), so the R-parity harness
(``tests/test_methodology_lpdid.py``) can anchor the RA *point* estimate but only
*pins* the library influence-function SE as a documented regression value. This
study validates that SE the way coverage is the ultimate test of a standard
error: simulate panels with a KNOWN dynamic treatment effect, fit the RA path
(``reweight=True`` + covariate), and check that the reported 95% CI covers the
true effect at ~95%.

The RA IF variance is asymptotic (no finite-sample factor - the ``teffects``
convention); the study sweeps the cluster count G and checks that empirical
coverage holds near nominal. In practice it is well-calibrated (~0.95) even at
modest G because the reported CI uses a ``t(G-1)`` reference distribution, which
widens to compensate at small G. Coverage at/near ~0.95 across G is the
validation. Mirrors
``benchmarks/python/coverage_sdid.py``: it lives under ``benchmarks/`` (never run
in gated CI - CI's isolated-install copies only ``tests/``) and writes an
artifact that underwrites ``docs/methodology/REGISTRY.md`` ## LPDiD Deviation 2.

Usage::

    python benchmarks/python/coverage_lpdid_ra.py            # default sweep
    python benchmarks/python/coverage_lpdid_ra.py --reps 200 # quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diff_diff.lpdid import LPDiD  # noqa: E402

PRE, POST = 2, 3
# Homogeneous dynamic effect (so the true RA-ATT at horizon h is exactly TAU[h]).
TAU = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5}
BETA_X = 0.7
# True pooled-post effect: mean long-difference over [0, POST] == mean of TAU (the
# post window [0, POST] spans exactly TAU's horizons), so the pooled-row RA estimand is:
TRUE_POOLED = sum(TAU.values()) / len(TAU)


def simulate_panel(n_units: int, rng: np.random.Generator, n_periods: int = 10) -> pd.DataFrame:
    """Staggered absorbing panel, ~40% never-treated, homogeneous effects + AR(1) noise."""
    cohorts = rng.choice([4, 6, 0], size=n_units, p=[0.3, 0.3, 0.4])
    unit_fe = rng.normal(0, 2.0, n_units)
    time_fe = rng.normal(0, 1.0, n_periods + 1)
    rows = []
    for u in range(n_units):
        g = int(cohorts[u])
        eps_prev = 0.0
        for t in range(1, n_periods + 1):
            eps = 0.3 * eps_prev + rng.normal(0, 1.0)
            eps_prev = eps
            treated = int(g > 0 and t >= g)
            eff = TAU.get(t - g, 0.0) if treated else 0.0
            x = rng.normal(0, 1.0)
            y = unit_fe[u] + time_fe[t] + eff + BETA_X * x + eps
            rows.append((u + 1, t, treated, y, x))
    return pd.DataFrame(rows, columns=["unit", "time", "treat", "y", "x"])


def run_coverage(n_units: int, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    hits = {h: 0 for h in TAU}
    valid = {h: 0 for h in TAU}
    pooled_hits = 0
    pooled_valid = 0
    n_fit_errors = 0
    for _ in range(reps):
        panel = simulate_panel(n_units, rng)
        try:
            res = LPDiD(pre_window=PRE, post_window=POST, reweight=True, cluster="unit").fit(
                panel,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                covariates=["x"],
            )
        except Exception:
            n_fit_errors += 1
            continue
        es = res.event_study.set_index("horizon")
        for h, tau in TAU.items():
            if h not in es.index:
                continue
            lo, hi = es.loc[h, "conf_low"], es.loc[h, "conf_high"]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            valid[h] += 1
            if lo <= tau <= hi:
                hits[h] += 1
        # Headline pooled-row RA CI coverage of the true pooled effect - validates the
        # SE that backs results.att/results.se (not just the event-study horizons).
        plo, phi = res.conf_int
        if np.isfinite(plo) and np.isfinite(phi):
            pooled_valid += 1
            if plo <= TRUE_POOLED <= phi:
                pooled_hits += 1
    coverage = {h: (hits[h] / valid[h] if valid[h] else float("nan")) for h in TAU}
    mean_event_coverage = float(np.nanmean(list(coverage.values())))
    pooled_att_coverage = (pooled_hits / pooled_valid) if pooled_valid else float("nan")
    min_valid_share = (min(min(valid.values()), pooled_valid) / reps) if reps else 0.0
    return {
        "n_units": n_units,
        "reps": reps,
        "per_horizon": coverage,
        "mean_event_coverage": mean_event_coverage,
        "pooled_att_coverage": pooled_att_coverage,
        "n_valid": valid,
        "n_pooled_valid": pooled_valid,
        "n_fit_errors": n_fit_errors,
        "min_valid_share": min_valid_share,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=500, help="Monte-Carlo reps per G (default 500)")
    ap.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[30, 100, 300],
        help="cluster counts G to sweep (default 30 100 300)",
    )
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument(
        "--out",
        type=str,
        default=str(_REPO_ROOT / "benchmarks" / "data" / "lpdid_ra_coverage.json"),
    )
    args = ap.parse_args()

    results = []
    print(f"LP-DiD RA-SE Monte-Carlo coverage (nominal 0.95), reps={args.reps}")
    print(f"{'G':>5} {'mean_ev':>8} {'pooled':>8}  " + "  ".join(f"h={h}" for h in TAU) + "  errs")
    for i, g in enumerate(args.clusters):
        r = run_coverage(g, args.reps, args.seed + i)
        results.append(r)
        ph = "  ".join(f"{r['per_horizon'][h]:.3f}" for h in TAU)
        print(
            f"{g:>5} {r['mean_event_coverage']:>8.3f} {r['pooled_att_coverage']:>8.3f}  "
            f"{ph}  {r['n_fit_errors']}"
        )

    # Surface a broken or miscalibrated regeneration loudly rather than writing a
    # misleading artifact (it underwrites REGISTRY ## LPDiD Deviation 2). Mass fit
    # failures or NaN coverage are always hard errors. Out-of-band coverage is a hard
    # error for substantive runs (reps >= 200, incl. the committed default 500), a warning
    # for small noisy diagnostic runs. The aggregates (headline pooled-row RA coverage and
    # the event-horizon mean) use a tight band; individual horizons are noisier, so a
    # per-horizon miscalibration is caught with a slightly wider band (so a bad horizon
    # cannot hide behind a good average).
    SUBSTANTIVE_REPS = 200
    AGG_BAND = (0.90, 0.98)
    PER_HORIZON_BAND = (0.88, 0.99)

    def _check(r: dict, name: str, cov: float, band: tuple) -> None:
        if not np.isfinite(cov):
            raise RuntimeError(f"G={r['n_units']}: {name} coverage is NaN")
        if band[0] <= cov <= band[1]:
            return
        msg = f"G={r['n_units']} {name} coverage {cov:.3f} outside {list(band)}"
        if args.reps >= SUBSTANTIVE_REPS:
            raise RuntimeError(
                msg + f" (reps={args.reps}) - the committed artifact underwrites a REGISTRY "
                "claim, so a substantive run must not write an out-of-band result"
            )
        print(f"  WARNING: {msg} (reps={args.reps} < {SUBSTANTIVE_REPS}; noisy)")

    for r in results:
        if r["min_valid_share"] < 0.5:
            raise RuntimeError(
                f"G={r['n_units']}: only {r['min_valid_share']:.0%} of reps produced a valid CI "
                f"({r['n_fit_errors']} fit errors) - regeneration looks broken"
            )
        _check(r, "pooled-att", r["pooled_att_coverage"], AGG_BAND)
        _check(r, "mean-event", r["mean_event_coverage"], AGG_BAND)
        for h, cov in r["per_horizon"].items():
            _check(r, f"h={h}", cov, PER_HORIZON_BAND)

    artifact = {
        "study": "LPDiD RA influence-function SE coverage",
        "nominal": 0.95,
        "tau": {str(k): v for k, v in TAU.items()},
        "note": (
            "RA IF variance is asymptotic (teffects convention, no finite-sample "
            "factor); empirical coverage of the true effect holds near nominal across G "
            "for both the event-study horizons and the headline pooled-row RA CI - the "
            "t(G-1) reference keeps it well-calibrated even at modest G. Underwrites "
            "REGISTRY ## LPDiD Deviation 2."
        ),
        "sweep": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
