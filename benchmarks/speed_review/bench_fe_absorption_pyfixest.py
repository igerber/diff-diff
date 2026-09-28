"""Optional pyfixest yardstick for the FE-absorption suite.

Runs the same seeded scenarios as ``bench_fe_absorption.py`` through
``pyfixest.feols`` so the committed diff-diff baselines can be read against
the state of the art, not only against our own past. pyfixest is NOT a
dependency of diff-diff or of the test suite - this script exits cleanly if
it is not installed (mirror of the guarded R lanes in ``benchmarks/R/``).

Honesty checks per scenario:

- ``scanner_twfe`` / ``geo_experiment`` / ``tail_stress`` / ``guard_small``
  estimate the identical regression (``y ~ d | fe1 + fe2``); the yardstick
  asserts coefficient agreement < 1e-6 against the diff-diff baseline JSON.
- ``county_policy`` / ``firm_churn`` are Sun-Abraham event studies; pyfixest
  0.60 has no ``sunab()``, so the yardstick runs the saturated
  ``i(rel_time)`` event-study regression - the same demeaning load (outcome +
  many interaction columns through unit+time FE) but a different estimand
  weighting, so it is TIMING-ONLY (no coefficient assertion).
- ``survey_absorb`` is skipped: pyfixest has no BRR replicate-weight
  refit path, so there is no comparable computation.

Usage::

    pip install pyfixest   # in any environment alongside diff-diff
    python benchmarks/speed_review/bench_fe_absorption_pyfixest.py \
        [--quick] [--repeats 3] [--only geo_experiment] \
        [--baseline benchmarks/speed_review/baselines/fe_absorption_before.json]
"""

import argparse
import json
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fe_absorption_datagen as datagen  # noqa: E402

try:
    import pyfixest as pf
except ImportError:
    print(
        "pyfixest is not installed - the yardstick lane is optional; "
        "run `pip install pyfixest` to enable it."
    )
    sys.exit(0)

COEF_PARITY_TOL = 1e-6
PARITY_SCENARIOS = ("scanner_twfe", "geo_experiment", "tail_stress", "guard_small")
TIMING_ONLY = ("county_policy", "firm_churn")
SKIPPED = ("survey_absorb",)


def _formula_and_frame(scenario, df):
    """Return (formula, frame, coef_name) for the pyfixest equivalent."""
    if scenario in ("scanner_twfe", "guard_small"):
        return "y ~ d | unit + post", df, "d"
    if scenario in ("geo_experiment", "tail_stress"):
        return "y ~ d | store + week", df, "d"
    if scenario in TIMING_ONLY:
        # Saturated event study: same demeaning load as Sun-Abraham
        # (outcome + interaction block through unit+time FE), timing-only.
        frame = df.copy()
        rel = frame["time"] - frame["first_treat"]
        # never-treated (first_treat == 0) pinned to the reference bin
        frame["rel_time"] = np.where(frame["first_treat"] > 0, rel, -1)
        return "y ~ i(rel_time, ref=-1) | unit + time", frame, None
    raise ValueError(f"no pyfixest mapping for {scenario!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", choices=datagen.SCENARIO_IDS)
    ap.add_argument(
        "--baseline",
        default="benchmarks/speed_review/baselines/fe_absorption_before.json",
        help="diff-diff baseline JSON for the coefficient-parity check",
    )
    ap.add_argument(
        "--out", default="benchmarks/speed_review/baselines/fe_absorption_pyfixest.json"
    )
    args = ap.parse_args()

    if args.repeats < 1:
        ap.error("--repeats must be >= 1")

    baseline = {}
    bp = Path(args.baseline)
    if bp.exists():
        baseline = {r["scenario"]: r for r in json.loads(bp.read_text())["results"] if "att" in r}
    else:
        print(f"(no diff-diff baseline at {args.baseline}; parity checks skipped)")

    scenarios = [args.only] if args.only else [s for s in datagen.SCENARIO_IDS if s not in SKIPPED]
    print(
        f"pyfixest yardstick {pf.__version__} | repeats={args.repeats} | " f"{platform.platform()}"
    )

    results, failures = [], 0
    for scen in scenarios:
        if scen in SKIPPED:
            print(f"  {scen:16s} SKIP (no BRR replicate path in pyfixest)")
            continue
        df, meta = datagen.build(scen, quick=args.quick)
        fml, frame, coef_name = _formula_and_frame(scen, df)

        # JIT/compile warmup on the quick-size frame through the same formula
        wdf, _ = datagen.build(scen, quick=True, seed=1)
        wfml, wframe, _ = _formula_and_frame(scen, wdf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf.feols(wfml, data=wframe, vcov="hetero")

        times, coef = [], None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = pf.feols(fml, data=frame, vcov="hetero")
            times.append(time.perf_counter() - t0)
            if coef_name is not None:
                coef = float(m.coef()[coef_name])
        med = statistics.median(times)
        cv = (statistics.stdev(times) / statistics.mean(times)) if len(times) > 1 else 0.0

        parity = None
        if coef_name is not None and scen in baseline:
            delta = abs(coef - baseline[scen]["att"])
            parity = delta <= COEF_PARITY_TOL
            if scen in PARITY_SCENARIOS and not parity:
                failures += 1
            print(
                f"  {scen:16s} n={meta['n_obs']:>10,} median={med:8.3f}s cv={cv:.1%} "
                f"|d coef vs diff-diff|={delta:.2e} {'ok' if parity else 'FAIL'}"
            )
        else:
            print(
                f"  {scen:16s} n={meta['n_obs']:>10,} median={med:8.3f}s cv={cv:.1%} "
                f"(timing-only)"
            )
        results.append(
            {
                "scenario": scen,
                "n_obs": meta["n_obs"],
                "checksum": meta["checksum"],
                "formula": fml,
                "coef": coef,
                "fit_median_s": round(med, 4),
                "fit_cv": round(cv, 4),
                "parity_ok": parity,
            }
        )

    payload = {
        "suite": "fe_absorption_pyfixest",
        "pyfixest": pf.__version__,
        "platform": platform.platform(),
        "repeats": args.repeats,
        "quick": args.quick,
        "results": results,
    }
    try:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")
    except OSError as e:
        print(f"\n(could not write {args.out}: {e})")
    if failures:
        print(f"COEFFICIENT PARITY FAILED for {failures} scenario(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
