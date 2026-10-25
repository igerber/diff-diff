#!/usr/bin/env python3
"""
Benchmark: TwoWayFixedEffects event study (``event_study=True, spec="within"``).

Live successor of the frozen 3.x ``benchmark_multiperiod.py`` arm: the same
unit-FE event-study design (``outcome ~ treated x period | unit``, clustered
by unit), fit through ``TwoWayFixedEffects(cluster=...).fit(...,
event_study=True, spec="within")``, which returns the unified
``EventStudyResults`` surface instead of the removed ``MultiPeriodDiDResults``
container. The CLI and the output JSON keys are identical to the frozen arm so
``run_benchmarks.py`` and ``compare_results.py`` consume it unchanged, and the
R side stays ``benchmarks/R/benchmark_multiperiod.R`` (the model is the same).

Two conversions the frozen arm also carried:

* The surface's ``event_time`` holds CALENDAR period labels; the R script and
  ``compare_results.compare_event_study`` join on reference-RELATIVE
  ``event_time``, so each row is emitted as ``{"period": label, "event_time":
  label - reference_period, ...}``.
* The surface has no post-period average. The headline ``att``/``se`` are the
  former post-average arithmetic, inlined below exactly as
  ``tests/helpers/event_study_average.py`` (benchmarks cannot import
  ``tests/``): equal-weight mean of the post-period ``att`` rows and
  ``sqrt(sum(post vcov block) / n_post**2)`` with the Cartesian ``np.ix_``
  block (plain ``vcov[cols, cols]`` would keep only the diagonal and drop the
  cross-period covariances). Student-t p-value / 95% CI use the surface's
  shared post-row ``df``; R uses ``nobs - k`` without the absorbed-FE
  subtraction - a pre-existing difference, and ``compare_estimates`` gates
  ``att``/``se`` only.

Usage:
    python benchmark_twfe_event_study.py --data path/to/data.csv \
        --output path/to/results.json --n-pre 4 --n-post 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


# IMPORTANT: Parse --backend and set environment variable BEFORE importing diff_diff
# This ensures the backend configuration is respected by all modules
def _get_backend_from_args():
    """Parse --backend argument without importing diff_diff."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", default="auto", choices=["auto", "python", "rust"])
    args, _ = parser.parse_known_args()
    return args.backend


_requested_backend = _get_backend_from_args()
if _requested_backend in ("python", "rust"):
    os.environ["DIFF_DIFF_BACKEND"] = _requested_backend

# NOW import diff_diff and other dependencies (will see the env var)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Add repo root to path for benchmarks.python.utils. When
# DIFF_DIFF_BENCH_USE_INSTALLED=1 (isolated-venv refresh runs), APPEND so the
# venv's installed diff-diff wheel wins over the dev tree; otherwise PREPEND
# (historical behavior: benchmark the working tree).
_REPO_ROOT = str(Path(__file__).parent.parent.parent)
if os.environ.get("DIFF_DIFF_BENCH_USE_INSTALLED") == "1":
    sys.path.append(_REPO_ROOT)
else:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.python.utils import Timer, collect_provenance  # noqa: E402
from diff_diff import HAS_RUST_BACKEND, TwoWayFixedEffects  # noqa: E402
from diff_diff.utils import safe_inference  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TwoWayFixedEffects event-study mode (spec='within')"
    )
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument("--cluster", default="unit", help="Column to cluster standard errors on")
    parser.add_argument("--n-pre", type=int, required=True, help="Number of pre-treatment periods")
    parser.add_argument(
        "--n-post", type=int, required=True, help="Number of post-treatment periods"
    )
    parser.add_argument(
        "--reference-period",
        type=int,
        default=None,
        help="Reference period (default: last pre-period = n_pre)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "python", "rust"],
        help="Backend to use: auto (default), python (pure Python), rust (Rust backend)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one untimed fit before the timed fit (JIT/cache warm-up)",
    )
    return parser.parse_args()


def get_actual_backend() -> str:
    """Return the actual backend being used based on HAS_RUST_BACKEND."""
    return "rust" if HAS_RUST_BACKEND else "python"


def post_period_average(es) -> Dict[str, float]:
    """Headline post-period average of an ``EventStudyResults`` surface.

    Reproduces the former post-average arithmetic and NaN contract exactly as
    ``tests/helpers/event_study_average.py``: ``att = mean(post att rows)``;
    ``se = sqrt(sum(post vcov block) / n_post**2)`` over the Cartesian
    ``np.ix_`` block; any NaN post ``att`` voids all five fields; a non-finite
    or negative summed variance voids ``se`` (and hence p-value / CI); a zero
    variance gives ``se = 0.0`` with an undefined (NaN) p-value / CI. The
    p-value and CI come from the library's own ``safe_inference`` (all-NaN on
    a non-positive or non-finite SE - never a finite p-value for undefined
    inference) with the surface's shared post-row ``df`` (Student t; normal
    when the df is undefined), matching the R script's
    ``is.finite(avg_se) && avg_se > 0`` guard.
    """
    nan5 = {
        "att": float("nan"),
        "se": float("nan"),
        "pvalue": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
    }
    if es.post_periods is None:
        raise ValueError("post_period_average requires a surface with post_periods provenance")
    post_rows = np.isin(es.event_time, list(es.post_periods))
    post_att = np.asarray(es.att, dtype=float)[post_rows]
    if post_att.size == 0:
        raise ValueError("post_period_average: no post-period rows on the surface")
    if np.any(np.isnan(post_att)):
        return nan5
    att_bar = float(np.mean(post_att))
    out = dict(nan5)
    out["att"] = att_bar
    if es.vcov is None or es.vcov_index is None:
        return out
    post_cols = np.isin(es.vcov_index, list(es.post_periods))
    n_post = int(post_cols.sum())
    if n_post == 0:
        return out
    avg_var = float(np.sum(np.asarray(es.vcov, dtype=float)[np.ix_(post_cols, post_cols)])) / (
        n_post**2
    )
    if not np.isfinite(avg_var) or avg_var < 0:
        return out
    se_bar = float(np.sqrt(avg_var))
    out["se"] = se_bar
    # Shared post-row df of the surface (the fit's resolved inference df);
    # undefined df -> None (normal reference inside safe_inference).
    df = float(np.asarray(es.df, dtype=float)[post_rows][0])
    df_arg = df if np.isfinite(df) and df > 0 else None
    _t, p_value, (ci_lo, ci_hi) = safe_inference(att_bar, se_bar, alpha=0.05, df=df_arg)
    out["pvalue"] = float(p_value)
    out["ci_lower"] = float(ci_lo)
    out["ci_upper"] = float(ci_hi)
    return out


def main():
    args = parse_args()

    # Get actual backend (already configured via env var before imports)
    actual_backend = get_actual_backend()
    print(f"Using backend: {actual_backend}")

    # Load data
    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data)

    # Compute post_periods and reference_period from args
    all_periods = sorted(data["time"].unique())
    n_pre = args.n_pre
    post_periods = [p for p in all_periods if p > n_pre]
    ref_period = args.reference_period if args.reference_period is not None else n_pre

    print(f"All periods: {all_periods}")
    print(f"Post periods: {post_periods}")
    print(f"Reference period: {ref_period}")

    fit_kwargs: Dict[str, Any] = dict(
        outcome="outcome",
        treatment="treated",
        unit="unit",
        time="time",
        event_study=True,
        spec="within",
        post_periods=post_periods,
        reference_period=ref_period,
    )

    # Run benchmark
    print("Running TwoWayFixedEffects event-study estimation (spec='within')...")

    if args.warmup:
        print("Warm-up fit (untimed)...")
        TwoWayFixedEffects(cluster=args.cluster).fit(data, **fit_kwargs)

    twfe = TwoWayFixedEffects(cluster=args.cluster)

    with Timer() as timer:
        results = twfe.fit(data, **fit_kwargs)

    total_time = timer.elapsed

    # Extract period effects (excluding reference period). The surface's
    # event_time holds CALENDAR labels; R and compare_event_study join on
    # reference-relative event_time, so convert here (as the frozen arm did).
    period_effects = []
    for label, att, se, is_ref in zip(
        results.event_time, results.att, results.se, results.is_reference
    ):
        if is_ref:
            continue
        period_effects.append(
            {
                "period": int(label),
                "event_time": int(label - ref_period),
                "att": float(att),
                "se": float(se),
            }
        )
    period_effects.sort(key=lambda row: row["period"])

    headline = post_period_average(results)

    # Build output
    output = {
        "estimator": "diff_diff.TwoWayFixedEffects[event_study, spec=within]",
        "backend": actual_backend,
        "cluster": args.cluster,
        # Average treatment effect (across post-periods)
        "att": headline["att"],
        "se": headline["se"],
        "pvalue": headline["pvalue"],
        "ci_lower": headline["ci_lower"],
        "ci_upper": headline["ci_upper"],
        # Reference period
        "reference_period": int(ref_period),
        # Period-level effects
        "period_effects": period_effects,
        # Timing
        "timing": {
            "estimation_seconds": total_time,
            "total_seconds": total_time,
        },
        # Metadata
        "metadata": {
            "n_units": int(data["unit"].nunique()),
            "n_periods": int(data["time"].nunique()),
            "n_obs": len(data),
            "n_pre": n_pre,
            "n_post": len(post_periods),
            "warmup": args.warmup,
        },
        # Wheel/backend provenance (refresh runs hard-fail on mismatch)
        "provenance": collect_provenance(),
    }

    # Write output
    print(f"Writing results to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"ATT: {headline['att']:.6f}")
    print(f"SE:  {headline['se']:.6f}")
    print(f"Completed in {total_time:.3f} seconds")
    return output


if __name__ == "__main__":
    main()
