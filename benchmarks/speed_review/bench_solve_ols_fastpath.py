"""solve_ols opt-in Cholesky fast-path A/B benchmark (DIFF_DIFF_SOLVE_OLS_FASTPATH).

Each scenario runs in a fresh subprocess (the speed_review noise protocol);
within a scenario the knob-off and knob-on arms share that process — the
knob is resolved per call, so same-process A/B is exact. Scenarios are the
solver-bound shapes from the 2026-07 attribution:

- ``county_policy`` / ``firm_churn``  - SunAbraham saturated fits (fe_absorption shapes)
- ``scanner_twfe``                    - TwoWayFixedEffects (demean-bound control arm)
- ``cs_cov40``                        - CallawaySantAnna dr, 40 covariates, 2M rows
                                        (per-cell solver floor)
- ``survey_absorb``                   - DiD absorb + BRR replicates (weighted lane ->
                                        numpy twin coverage)
- ``skiprank_micro``                  - direct solve_ols(skip_rank_check=True) on a
                                        firm-shape matrix (self-certification lane)

The knob-off arm doubles as the BYTE-IDENTITY gate: run this script once on
the pristine base tree (``PYTHONPATH=<base-tree> --arms off --out before.json``)
and once on the branch (``--arms off,on``), then ``--check-identity before.json``
asserts the knob-off ATT/SE are exactly equal to the base tree's (the default
path must be untouched) and reports the knob-on deltas.

Usage::

    python benchmarks/speed_review/bench_solve_ols_fastpath.py --repeats 3 \
        --out benchmarks/speed_review/baselines/solve_ols_fastpath_after.json \
        --check-identity benchmarks/speed_review/baselines/solve_ols_fastpath_before.json
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fe_absorption_datagen as datagen  # noqa: E402

KNOB = "DIFF_DIFF_SOLVE_OLS_FASTPATH"

# (scenario, timed fits per arm)
SCENARIOS = {
    "county_policy": 3,
    "firm_churn": 2,
    "scanner_twfe": 3,
    "cs_cov40": 3,
    "survey_absorb": 2,
    "skiprank_micro": 2,
}


def make_cs_panel(n_units=100_000, n_periods=20, n_cov=40, seed=0):
    """Staggered panel, 5 treated cohorts + never-treated, covariate-selected
    cohorts (same DGP as the 2026-07 attribution profiling)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_units, n_cov))
    score = 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] + rng.standard_normal(n_units)
    qs = np.quantile(score, [0.35, 0.48, 0.61, 0.74, 0.87])
    cohort_vals = np.array([0, 4, 7, 10, 13, 16])
    g_unit = cohort_vals[np.searchsorted(qs, score)]
    unit = np.repeat(np.arange(n_units, dtype=np.int64), n_periods)
    t = np.tile(np.arange(n_periods, dtype=np.int64), n_units)
    g = np.repeat(g_unit, n_periods)
    alpha = np.repeat(rng.normal(0, 1, n_units), n_periods)
    beta = rng.normal(0.1, 0.05, n_cov)
    xb = np.repeat(X @ beta, n_periods)
    treated = (g > 0) & (t >= g)
    y = alpha + 0.05 * t + xb + 2.0 * treated + rng.normal(0, 1, n_units * n_periods)
    cols = {"unit": unit, "time": t, "first_treat": g, "y": y}
    for j in range(n_cov):
        cols[f"x{j}"] = np.repeat(X[:, j], n_periods)
    return pd.DataFrame(cols)


def make_skiprank_matrix(n=2_400_000, k=130, seed=1):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    y = X @ rng.normal(0, 0.5, k) + rng.standard_normal(n)
    return X, y


def build(scenario):
    """Return (payload, fit) where fit(payload) -> (att-like, se-like)."""
    if scenario in ("county_policy", "firm_churn"):
        df, _ = datagen.build(scenario)
        from diff_diff import SunAbraham

        def fit(d):
            res = SunAbraham().fit(
                d, outcome="y", unit="unit", time="time", first_treat="first_treat"
            )
            return float(res.att), float(res.se)

        return df, fit
    if scenario == "scanner_twfe":
        df, _ = datagen.build(scenario)
        from diff_diff import TwoWayFixedEffects

        def fit(d):
            res = TwoWayFixedEffects().fit(
                d, outcome="y", treatment="treated", time="post", unit="unit"
            )
            return float(res.att), float(res.se)

        return df, fit
    if scenario == "cs_cov40":
        df = make_cs_panel()
        from diff_diff import CallawaySantAnna

        covs = [f"x{j}" for j in range(40)]

        def fit(d):
            res = CallawaySantAnna(estimation_method="dr").fit(
                d,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=covs,
                aggregate="all",
            )
            return float(res.overall_att), float(res.overall_se)

        return df, fit
    if scenario == "survey_absorb":
        df, _ = datagen.build(scenario)
        from diff_diff import DifferenceInDifferences
        from diff_diff.survey import SurveyDesign

        rep_cols = [c for c in df.columns if c.startswith("rw")]

        def fit(d):
            design = SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="BRR")
            res = DifferenceInDifferences().fit(
                d,
                outcome="y",
                treatment="treated",
                time="post",
                absorb=["state", "month"],
                survey_design=design,
            )
            return float(res.att), float(res.se)

        return df, fit
    if scenario == "skiprank_micro":
        X, y = make_skiprank_matrix()
        from diff_diff.linalg import solve_ols

        def fit(payload):
            Xm, ym = payload
            coef, _, vcov = solve_ols(Xm, ym, skip_rank_check=True)
            return float(np.sum(coef)), float(np.sqrt(vcov[1, 1]))

        return (X, y), fit
    raise ValueError(scenario)


def run_scenario(scenario, arms, repeats_scale):
    import warnings

    warnings.filterwarnings("ignore")
    payload, fit = build(scenario)
    n_fits = max(1, round(SCENARIOS[scenario] * repeats_scale))

    # Warmup once through the same front door (imports, BLAS/rayon init).
    os.environ.pop(KNOB, None)
    fit(payload)

    out = {}
    for arm in arms:
        if arm == "on":
            os.environ[KNOB] = "1"
        else:
            os.environ.pop(KNOB, None)
        times = []
        att = se = None
        for _ in range(n_fits):
            t0 = time.perf_counter()
            att, se = fit(payload)
            times.append(time.perf_counter() - t0)
        out[arm] = {
            "times": times,
            "median": statistics.median(times),
            "att": att,
            "se": se,
        }
        os.environ.pop(KNOB, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="off,on", help="comma list: off,on")
    ap.add_argument("--only", default=None, help="comma list of scenario ids")
    ap.add_argument("--repeats", type=float, default=1.0, help="scale timed-fit counts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-identity", default=None, metavar="BEFORE_JSON")
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="run scenarios in this process (default: one fresh subprocess per "
        "scenario, the speed_review noise protocol — in-process runs showed "
        "cross-scenario state contaminating arm timings by ~25%%)",
    )
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ids = list(SCENARIOS) if args.only is None else args.only.split(",")

    results = {"metadata": {"platform": platform.platform(), "arms": arms}}
    if not args.in_process and len(ids) > 1:
        # Fresh subprocess per scenario, strictly sequential (matches
        # bench_fe_absorption's noise protocol).
        import subprocess
        import tempfile

        for scenario in ids:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                tmp = tf.name
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--only",
                scenario,
                "--arms",
                args.arms,
                "--repeats",
                str(args.repeats),
                "--out",
                tmp,
                "--in-process",
            ]
            subprocess.run(cmd, check=True)
            with open(tmp) as f:
                results[scenario] = json.load(f)[scenario]
            os.unlink(tmp)
    else:
        for scenario in ids:
            print(f"=== {scenario}", flush=True)
            results[scenario] = run_scenario(scenario, arms, args.repeats)
            for arm, r in results[scenario].items():
                print(
                    f"  {arm:>3}: median {r['median']:.3f}s  att={r['att']:.6f}  se={r['se']:.6g}",
                    flush=True,
                )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")

    if args.check_identity:
        with open(args.check_identity) as f:
            before = json.load(f)
        failures = []
        for scenario in ids:
            if scenario not in before or scenario not in results:
                continue
            b = before[scenario]["off"]
            a_off = results[scenario]["off"]
            # Byte-identity gate: knob-off on the branch == the base tree.
            if not (b["att"] == a_off["att"] and b["se"] == a_off["se"]):
                failures.append(
                    f"{scenario}: knob-off (att={a_off['att']!r}, se={a_off['se']!r}) "
                    f"!= base tree (att={b['att']!r}, se={b['se']!r})"
                )
            if "on" in results[scenario]:
                a_on = results[scenario]["on"]
                d_att = abs(a_on["att"] - a_off["att"])
                d_se = abs(a_on["se"] - a_off["se"]) / max(abs(a_off["se"]), 1e-300)
                speedup = a_off["median"] / a_on["median"]
                print(
                    f"{scenario}: speedup {speedup:.2f}x  |dATT|={d_att:.2e}  "
                    f"|dSE|/SE={d_se:.2e}"
                )
        if failures:
            print("IDENTITY GATE FAILED:")
            for f_ in failures:
                print("  " + f_)
            sys.exit(1)
        print("identity gate: knob-off byte-identical to base tree ✓")


if __name__ == "__main__":
    main()
