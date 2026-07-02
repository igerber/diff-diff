"""FE-absorption (MAP demeaning) wall-clock + identity benchmark suite.

Measures the fixed-effects absorption hot path across seven realistic
practitioner workloads (see ``fe_absorption_datagen.py`` and
docs/performance-scenarios.md for shape provenance). This is the before/after
surface for the demean_by_groups optimization work motivated by the 2026-07
pyfixest gap analysis: the ``_before``/``_after`` baseline JSONs committed
under ``baselines/`` carry both the timings and the ATT/SE identity fields.

Noise protocol
--------------
Every (scenario, repeat) runs in a fresh subprocess, strictly sequential.
Small scenarios time a warmup fit then the median of several in-process fits;
large scenarios time one fit per subprocess. The driver pools all timed fits
across subprocesses and reports median / min / max / CV, flagging CV > 10%
as too noisy to trust.

Identity gate
-------------
``--check-estimates baselines/fe_absorption_before.json`` compares each
scenario's ATT/SE against a previously committed baseline at the unified
tolerances (ATT atol=1e-9; SE rtol=1e-7; survey_absorb SE rtol=1e-6 because
the ~5.8e-11 accumulation-order drift stacks across 80 replicate re-demeans).
``tail_stress`` is excluded from the gate: its estimate is EXPECTED to move
when the MAP iteration cap is raised (it currently stops at max_iter without
converging) - the driver prints the delta instead.

Usage::

    python benchmarks/speed_review/bench_fe_absorption.py                # full sweep
    python benchmarks/speed_review/bench_fe_absorption.py --quick        # smoke
    python benchmarks/speed_review/bench_fe_absorption.py --only geo_experiment
    python benchmarks/speed_review/bench_fe_absorption.py --repeats 3
    python benchmarks/speed_review/bench_fe_absorption.py \
        --out benchmarks/speed_review/baselines/fe_absorption_after.json \
        --check-estimates benchmarks/speed_review/baselines/fe_absorption_before.json
"""

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fe_absorption_datagen as datagen  # noqa: E402

# Unified identity tolerances (single source for PR-A capture and PR-B gate).
ATT_ATOL = 1e-9
SE_RTOL = 1e-7
SE_RTOL_SURVEY = 1e-6  # drift stacks across ~80 replicate re-demeans
GATE_EXEMPT = ("tail_stress",)  # estimate expected to move when the cap rises
CV_FLAG = 0.10

# scenario -> number of timed in-process fits (after one warmup fit).
# Sub-second scenarios repeat in-process so the subprocess median is stable.
INPROC_FITS = {
    "county_policy": 3,
    "firm_churn": 1,
    "scanner_twfe": 1,
    "geo_experiment": 1,
    "survey_absorb": 1,
    "tail_stress": 1,
    "guard_small": 7,
}


def _peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS ru_maxrss is bytes; Linux is KiB.
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


# ------------------------------------------------------------------- fitting
def _fit(scenario, df):
    """Run the scenario's estimator front door once; return (att, se, extras)."""
    if scenario in ("county_policy", "firm_churn"):
        from diff_diff import SunAbraham

        res = SunAbraham().fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
    elif scenario in ("scanner_twfe", "guard_small"):
        from diff_diff import TwoWayFixedEffects

        res = TwoWayFixedEffects().fit(
            df, outcome="y", treatment="treated", time="post", unit="unit"
        )
    elif scenario in ("geo_experiment", "tail_stress"):
        from diff_diff import DifferenceInDifferences

        res = DifferenceInDifferences().fit(
            df, outcome="y", treatment="treated", time="post", absorb=["store", "week"]
        )
    elif scenario == "survey_absorb":
        from diff_diff import DifferenceInDifferences
        from diff_diff.survey import SurveyDesign

        rep_cols = [c for c in df.columns if c.startswith("rw")]
        design = SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="BRR")
        res = DifferenceInDifferences().fit(
            df,
            outcome="y",
            treatment="treated",
            time="post",
            absorb=["state", "month"],
            survey_design=design,
        )
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return float(res.att), float(res.se)


def _warmup(scenario):
    """Tiny fit through the same front door (imports, caches, BLAS init)."""
    df, _ = datagen.build(scenario, quick=True, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit(scenario, df)


# ------------------------------------------------------------------ worker IO
def _worker(cfg_json):
    cfg = json.loads(cfg_json)
    scenario, quick = cfg["scenario"], cfg["quick"]

    t0 = time.perf_counter()
    df, meta = datagen.build(scenario, quick=quick)
    t_datagen = time.perf_counter() - t0

    _warmup(scenario)

    fit_times, warn_msgs = [], set()
    att = se = None
    for _ in range(INPROC_FITS[scenario] if not quick else 1):
        t1 = time.perf_counter()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            att, se = _fit(scenario, df)
        fit_times.append(time.perf_counter() - t1)
        warn_msgs.update(str(w.message)[:110] for w in wl)

    try:
        from diff_diff._backend import HAS_RUST_BACKEND
    except Exception:
        HAS_RUST_BACKEND = False
    env_b = os.environ.get("DIFF_DIFF_BACKEND", "auto")
    resolved = "python" if (env_b == "python" or not HAS_RUST_BACKEND) else "rust"

    print(
        "FE_RESULT "
        + json.dumps(
            {
                "fit_times": fit_times,
                "t_datagen": t_datagen,
                "att": att,
                "se": se,
                "n_obs": meta["n_obs"],
                "checksum": meta["checksum"],
                "peak_rss_mb": _peak_rss_mb(),
                "backend_resolved": resolved,
                "warnings": sorted(warn_msgs),
            }
        )
    )


def _measure(scenario, repeats, backend, quick):
    env = {**os.environ, "DIFF_DIFF_BACKEND": backend}
    cfg = json.dumps({"scenario": scenario, "quick": quick})
    payloads = []
    for _ in range(repeats):
        p = subprocess.run(
            [sys.executable, __file__, "--worker", cfg],
            capture_output=True,
            text=True,
            env=env,
        )
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("FE_RESULT ")), None)
        if line is None:
            return None, p.stderr.strip().splitlines()[-3:] or ["(no output)"]
        payloads.append(json.loads(line[len("FE_RESULT ") :]))
    return payloads, None


# --------------------------------------------------------------- identity gate
def check_estimates(results, before_path):
    """Compare ATT/SE per scenario against a prior baseline JSON. Returns the
    number of failures (0 = pass). tail_stress reports its delta, ungated.
    """
    before = {r["scenario"]: r for r in json.loads(Path(before_path).read_text())["results"]}
    failures = 0
    for r in results:
        scen = r["scenario"]
        b = before.get(scen)
        if b is None or r.get("att") is None or b.get("att") is None:
            print(f"  identity {scen:16s} SKIP (missing in one side)")
            continue
        d_att = abs(r["att"] - b["att"])
        rel_se = abs(r["se"] - b["se"]) / max(abs(b["se"]), 1e-300)
        se_rtol = SE_RTOL_SURVEY if scen == "survey_absorb" else SE_RTOL
        if scen in GATE_EXEMPT:
            print(
                f"  identity {scen:16s} EXEMPT (expected shift): "
                f"|d att|={d_att:.3e} rel d se={rel_se:.3e}"
            )
            continue
        ok = d_att <= ATT_ATOL and rel_se <= se_rtol
        status = "ok" if ok else "FAIL"
        print(
            f"  identity {scen:16s} {status}: |d att|={d_att:.3e} (atol {ATT_ATOL}) "
            f"rel d se={rel_se:.3e} (rtol {se_rtol})"
        )
        failures += 0 if ok else 1
    return failures


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3, help="subprocess repeats")
    ap.add_argument("--only", choices=datagen.SCENARIO_IDS)
    ap.add_argument(
        "--backend",
        choices=("python", "rust", "auto"),
        default="python",
        help="DIFF_DIFF_BACKEND for the fits (default python, matching the committed baselines)",
    )
    ap.add_argument("--out", default="benchmarks/speed_review/baselines/fe_absorption.json")
    ap.add_argument(
        "--check-estimates",
        metavar="BEFORE_JSON",
        help="compare ATT/SE against a prior baseline at the unified tolerances",
    )
    args = ap.parse_args()

    if args.worker:
        _worker(args.worker)
        return

    if args.repeats < 1:
        ap.error("--repeats must be >= 1")
    scenarios = [args.only] if args.only else list(datagen.SCENARIO_IDS)

    print(
        f"FE-absorption suite | repeats={args.repeats} | backend={args.backend} | "
        f"{platform.platform()} | py{sys.version.split()[0]}"
    )
    results = []
    for scen in scenarios:
        payloads, err = _measure(scen, args.repeats, args.backend, args.quick)
        if payloads is None:
            print(f"  {scen:16s} -> ERROR: {err}")
            results.append({"scenario": scen, "error": err})
            continue
        pooled = [t for p in payloads for t in p["fit_times"]]
        med = statistics.median(pooled)
        cv = (statistics.stdev(pooled) / statistics.mean(pooled)) if len(pooled) > 1 else 0.0
        checksums = {p["checksum"] for p in payloads}
        assert len(checksums) == 1, f"{scen}: data not deterministic across runs"
        atts = {p["att"] for p in payloads}
        assert len(atts) == 1, f"{scen}: estimate not deterministic across runs"
        last = payloads[-1]
        noisy = cv > CV_FLAG
        results.append(
            {
                "scenario": scen,
                "n_obs": last["n_obs"],
                "checksum": last["checksum"],
                "att": last["att"],
                "se": last["se"],
                "fit_median_s": round(med, 4),
                "fit_min_s": round(min(pooled), 4),
                "fit_max_s": round(max(pooled), 4),
                "fit_cv": round(cv, 4),
                "noisy": noisy,
                "n_timed_fits": len(pooled),
                "datagen_s": round(last["t_datagen"], 3),
                "peak_rss_mb": round(max(p["peak_rss_mb"] for p in payloads), 1),
                "backend_resolved": last["backend_resolved"],
                "warnings": last["warnings"],
            }
        )
        flag = "  ** CV>10%, rerun **" if noisy else ""
        print(
            f"  {scen:16s} n={last['n_obs']:>10,} median={med:8.3f}s "
            f"[{min(pooled):.3f}, {max(pooled):.3f}] cv={cv:.1%} "
            f"rss={last['peak_rss_mb']:.0f}MB{flag}"
        )
        if last["warnings"]:
            for w in last["warnings"]:
                print(f"      warn: {w}")

    exit_code = 0
    if args.check_estimates:
        print("\nidentity gate vs", args.check_estimates)
        failures = check_estimates([r for r in results if "error" not in r], args.check_estimates)
        if failures:
            print(f"IDENTITY GATE FAILED: {failures} scenario(s) moved")
            exit_code = 1

    try:
        import diff_diff

        versions = {
            "diff_diff": diff_diff.__version__,
            "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__,
        }
    except Exception:
        versions = {}
    payload = {
        "suite": "fe_absorption",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "versions": versions,
        "backend_requested": args.backend,
        "repeats": args.repeats,
        "quick": args.quick,
        "tolerances": {
            "att_atol": ATT_ATOL,
            "se_rtol": SE_RTOL,
            "se_rtol_survey": SE_RTOL_SURVEY,
            "gate_exempt": list(GATE_EXEMPT),
        },
        "results": results,
    }
    try:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")
    except OSError as e:
        print(f"\n(could not write {args.out}: {e})")

    if any("error" in r for r in results):
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
