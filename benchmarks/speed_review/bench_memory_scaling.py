"""Peak-RSS memory benchmark for the memory-scaling work (B1 #561 / B2 #563 / C #567).

Distinct from the wall-clock ``bench_*.py`` scenarios: this measures **peak resident
memory** at the millions-of-units tail, where the dense ``(n_bootstrap x n_units)``
multiplier-bootstrap weight matrix (CallawaySantAnna / EfficientDiD / HAD) and the
TWFE within-transform working set dominate.

Each config is fit in an **isolated subprocess** and peak memory is
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (bytes on macOS, KiB on Linux). Each is
repeated and the **median** is reported — single-run ``ru_maxrss`` on these
transient-heavy paths is allocator-dependent (±1-2 GB at the 10 GB scale).

This measures the CURRENT tree (the "after"). For an honest before/after, run the same
script from a pre-#561 checkout (e.g. a throwaway worktree at the parent of the #561
merge) — all three wins are post-#561, so that tree is the un-optimized baseline. Do
NOT monkeypatch the chunking off on the current tree: that baseline is contaminated by
the other same-line optimizations (P2 identity-skip, #567 within_transform).

Usage::

    python benchmarks/speed_review/bench_memory_scaling.py            # full sweep
    python benchmarks/speed_review/bench_memory_scaling.py --quick    # tiny smoke
    python benchmarks/speed_review/bench_memory_scaling.py --repeats 3
    python benchmarks/speed_review/bench_memory_scaling.py --only cs  # one estimator
"""

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def _peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS ru_maxrss is bytes; Linux is KiB.
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


# ---------------------------------------------------------------- panel builders
def _staggered_panel(n_units, n_periods=6, seed=7):
    """Balanced staggered panel: unit, time, outcome, first_treat (0 = never)."""
    rng = np.random.default_rng(seed)
    first_treat = rng.choice([0, n_periods // 2, n_periods // 2 + 1], size=n_units)
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)
    ft = np.repeat(first_treat, n_periods)
    post = (time >= ft) & (ft > 0)
    y = (
        1.0
        + 0.1 * (unit / n_units)
        + 0.2 * time
        + 1.5 * post
        + rng.standard_normal(n_units * n_periods)
    )
    return pd.DataFrame({"unit": unit, "time": time, "outcome": y, "first_treat": ft})


def _dose_panel(n_units, n_periods=4, seed=11):
    """Continuous-dose panel for HAD: unit, period, dose, outcome, w (unit-constant)."""
    rng = np.random.default_rng(seed)
    d_post = rng.uniform(0.0, 1.0, n_units)
    period = np.tile(np.arange(n_periods), n_units)
    unit = np.repeat(np.arange(n_units), n_periods)
    dose = np.where(period == n_periods - 1, np.repeat(d_post, n_periods), 0.0)
    y = 0.2 * period + np.where(period == n_periods - 1, 2.0 * dose, 0.0)
    y = y + 0.5 * rng.standard_normal(n_units * n_periods)
    w_unit = 1.0 + 0.3 * np.abs(rng.standard_normal(n_units))
    return pd.DataFrame(
        {
            "unit": unit,
            "period": period,
            "dose": dose,
            "outcome": y,
            "w": np.repeat(w_unit, n_periods),
        }
    )


def _twfe_panel(n_units, ncov=6, seed=0):
    """Two-period TWFE panel: unit, time(0/1), treated, post, outcome, x0..x{ncov-1}."""
    rng = np.random.default_rng(seed)
    n = n_units * 2
    unit = np.repeat(np.arange(n_units), 2)
    time = np.tile([0, 1], n_units)
    treated = np.repeat(rng.integers(0, 2, n_units), 2)
    d = {"unit": unit, "time": time, "treated": treated, "post": time}
    for j in range(ncov):
        d[f"x{j}"] = rng.standard_normal(n)
    d["outcome"] = (
        1.0 + 0.1 * (unit / n_units) + 0.3 * time + 1.5 * (treated * time) + rng.standard_normal(n)
    )
    return pd.DataFrame(d)


# ----------------------------------------------------------------- per-config fit
def _run_one(cfg):
    est = cfg["estimator"]
    nu, nb = cfg["n_units"], cfg.get("n_bootstrap", 0)
    if est == "cs":
        from diff_diff import CallawaySantAnna

        CallawaySantAnna(estimation_method="dr", cluster="unit", n_bootstrap=nb, seed=42).fit(
            _staggered_panel(nu), "outcome", "unit", "time", "first_treat"
        )
    elif est == "effdid":
        from diff_diff import EfficientDiD

        EfficientDiD(n_bootstrap=nb, seed=42).fit(
            _staggered_panel(nu), "outcome", "unit", "time", "first_treat"
        )
    elif est == "had":
        from diff_diff import HeterogeneousAdoptionDiD
        from diff_diff.survey import SurveyDesign

        HeterogeneousAdoptionDiD(design="continuous_at_zero", n_bootstrap=nb, seed=42).fit(
            _dose_panel(nu),
            "outcome",
            "dose",
            "period",
            "unit",
            aggregate="event_study",
            survey_design=SurveyDesign(weights="w"),
        )
    elif est == "twfe":
        from diff_diff.twfe import TwoWayFixedEffects

        ncov = cfg.get("ncov", 6)
        TwoWayFixedEffects(robust=True).fit(
            _twfe_panel(nu, ncov=ncov),
            outcome="outcome",
            treatment="treated",
            post="post",
            unit="unit",
            covariates=[f"x{j}" for j in range(ncov)],
        )
    else:
        raise ValueError(f"unknown estimator {est!r}")
    return _peak_rss_mb()


# ------------------------------------------------------------------ subprocess IO
def _worker(cfg_json):
    warnings.simplefilter("ignore")
    rss = _run_one(json.loads(cfg_json))
    try:
        from diff_diff._backend import HAS_RUST_BACKEND
    except Exception:
        HAS_RUST_BACKEND = False
    # Report the actually-active backend so the driver records what was measured,
    # not just what was requested (DIFF_DIFF_BACKEND=rust silently falls back to
    # python when the extension is not built).
    env_b = os.environ.get("DIFF_DIFF_BACKEND", "auto")
    resolved = "python" if (env_b == "python" or not HAS_RUST_BACKEND) else "rust"
    print("RSS_RESULT " + json.dumps({"rss_mb": rss, "backend_resolved": resolved}))


def _measure(cfg, repeats, backend):
    env = {**os.environ, "DIFF_DIFF_BACKEND": backend}
    vals, resolved = [], None
    for _ in range(repeats):
        p = subprocess.run(
            [sys.executable, __file__, "--worker", json.dumps(cfg)],
            capture_output=True,
            text=True,
            env=env,
        )
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RSS_RESULT ")), None)
        if line is None:
            return None, p.stderr.strip().splitlines()[-1:] or ["(no output)"]
        payload = json.loads(line[len("RSS_RESULT ") :])
        vals.append(payload["rss_mb"])
        resolved = payload.get("backend_resolved")
    return statistics.median(vals), {"runs": vals, "backend_resolved": resolved}


# ------------------------------------------------------------------------- config
# Heavy estimators (EfficientDiD sieve-DR, HAD local-linear) are capped at 500k to
# keep a single run feasible; CS / TWFE also probe 1M. n_bootstrap=999 is the
# production default and sizes the weight matrix.
CONFIGS = [
    {
        "label": "CS bootstrap (cluster=unit, 999)",
        "estimator": "cs",
        "n_units": 500_000,
        "n_bootstrap": 999,
    },
    {
        "label": "CS bootstrap (cluster=unit, 999)",
        "estimator": "cs",
        "n_units": 1_000_000,
        "n_bootstrap": 999,
    },
    {
        "label": "EfficientDiD bootstrap (999)",
        "estimator": "effdid",
        "n_units": 500_000,
        "n_bootstrap": 999,
    },
    {
        "label": "HAD event-study cband (999)",
        "estimator": "had",
        "n_units": 500_000,
        "n_bootstrap": 999,
    },
    {
        "label": "TWFE hc1 within-transform (6 cov)",
        "estimator": "twfe",
        "n_units": 500_000,
        "ncov": 6,
    },
    {
        "label": "TWFE hc1 within-transform (6 cov)",
        "estimator": "twfe",
        "n_units": 1_000_000,
        "ncov": 6,
    },
]

QUICK_CONFIGS = [
    {"label": "CS bootstrap (smoke)", "estimator": "cs", "n_units": 2_000, "n_bootstrap": 99},
    {"label": "EfficientDiD (smoke)", "estimator": "effdid", "n_units": 2_000, "n_bootstrap": 99},
    {"label": "HAD cband (smoke)", "estimator": "had", "n_units": 2_000, "n_bootstrap": 99},
    {"label": "TWFE (smoke)", "estimator": "twfe", "n_units": 2_000, "ncov": 6},
]


_ESTIMATORS = ("cs", "effdid", "had", "twfe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--only", choices=_ESTIMATORS, help="run only this estimator")
    ap.add_argument(
        "--backend",
        choices=("python", "rust", "auto"),
        default="python",
        help="DIFF_DIFF_BACKEND for the fits (default python, matching the committed baselines)",
    )
    ap.add_argument("--out", default="benchmarks/speed_review/baselines/memory_scaling.json")
    args = ap.parse_args()

    if args.worker:
        _worker(args.worker)
        return

    if args.repeats < 1:
        ap.error("--repeats must be >= 1")

    configs = QUICK_CONFIGS if args.quick else CONFIGS
    if args.only:
        configs = [c for c in configs if c["estimator"] == args.only]
    if not configs:
        ap.error("no configs selected")

    print(
        f"peak RSS (median of {args.repeats}) | backend={args.backend} | "
        f"{platform.platform()} | py{sys.version.split()[0]}"
    )
    results = []
    resolved_seen = set()
    for cfg in configs:
        med, detail = _measure(cfg, args.repeats, args.backend)
        if med is None:
            print(f"  {cfg['label']:36s} n={cfg['n_units']:>9,} -> ERROR: {detail}")
            results.append({**cfg, "peak_rss_mb": None, "error": detail})
        else:
            runs = detail["runs"]
            resolved_seen.add(detail["backend_resolved"])
            print(
                f"  {cfg['label']:36s} n={cfg['n_units']:>9,} -> {med:8.0f} MB   "
                f"runs={['%.0f' % v for v in runs]}"
            )
            results.append(
                {**cfg, "peak_rss_mb": round(med, 1), "runs": [round(v, 1) for v in runs]}
            )

    resolved = sorted(b for b in resolved_seen if b) or [args.backend]
    if args.backend != "python" or resolved != ["python"]:
        print(f"  NOTE: requested backend={args.backend}, resolved={resolved}")

    try:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "platform": platform.platform(),
                    "backend_requested": args.backend,
                    "backend_resolved": resolved,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.out}")
    except OSError as e:
        print(f"\n(could not write {args.out}: {e})")


if __name__ == "__main__":
    main()
