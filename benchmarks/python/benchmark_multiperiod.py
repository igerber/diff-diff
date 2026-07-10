#!/usr/bin/env python3
"""
Benchmark: MultiPeriodDiD event study (diff-diff MultiPeriodDiD).

Usage:
    python benchmark_multiperiod.py --data path/to/data.csv --output path/to/results.json \
        --n-pre 4 --n-post 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

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
import pandas as pd

# Add repo root to path for benchmarks.python.utils. When
# DIFF_DIFF_BENCH_USE_INSTALLED=1 (isolated-venv refresh runs), APPEND so the
# venv's installed diff-diff wheel wins over the dev tree; otherwise PREPEND
# (historical behavior: benchmark the working tree).
_REPO_ROOT = str(Path(__file__).parent.parent.parent)
if os.environ.get("DIFF_DIFF_BENCH_USE_INSTALLED") == "1":
    sys.path.append(_REPO_ROOT)
else:
    sys.path.insert(0, _REPO_ROOT)

from diff_diff import MultiPeriodDiD, HAS_RUST_BACKEND
from benchmarks.python.utils import Timer, collect_provenance


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MultiPeriodDiD estimator")
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument(
        "--cluster", default="unit", help="Column to cluster standard errors on"
    )
    parser.add_argument(
        "--n-pre", type=int, required=True, help="Number of pre-treatment periods"
    )
    parser.add_argument(
        "--n-post", type=int, required=True, help="Number of post-treatment periods"
    )
    parser.add_argument(
        "--reference-period", type=int, default=None,
        help="Reference period (default: last pre-period = n_pre)"
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "python", "rust"],
        help="Backend to use: auto (default), python (pure Python), rust (Rust backend)"
    )
    parser.add_argument(
        "--warmup", action="store_true",
        help="Run one untimed fit before the timed fit (JIT/cache warm-up)"
    )
    return parser.parse_args()


def get_actual_backend() -> str:
    """Return the actual backend being used based on HAS_RUST_BACKEND."""
    return "rust" if HAS_RUST_BACKEND else "python"


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

    # Run benchmark
    print("Running MultiPeriodDiD estimation...")

    if args.warmup:
        print("Warm-up fit (untimed)...")
        MultiPeriodDiD(robust=True, cluster=args.cluster).fit(
            data,
            outcome="outcome",
            treatment="treated",
            time="time",
            post_periods=post_periods,
            reference_period=ref_period,
            absorb=["unit"],
        )

    did = MultiPeriodDiD(robust=True, cluster=args.cluster)

    with Timer() as timer:
        results = did.fit(
            data,
            outcome="outcome",
            treatment="treated",
            time="time",
            post_periods=post_periods,
            reference_period=ref_period,
            absorb=["unit"],
        )

    total_time = timer.elapsed

    # Extract period effects (excluding reference period)
    period_effects = []
    for period, pe in sorted(results.period_effects.items()):
        event_time = period - ref_period
        period_effects.append({
            "period": int(period),
            "event_time": int(event_time),
            "att": float(pe.effect),
            "se": float(pe.se),
        })

    # Build output
    output = {
        "estimator": "diff_diff.MultiPeriodDiD",
        "backend": actual_backend,
        "cluster": args.cluster,
        # Average treatment effect (across post-periods)
        "att": float(results.avg_att),
        "se": float(results.avg_se),
        "pvalue": float(results.avg_p_value),
        "ci_lower": float(results.avg_conf_int[0]),
        "ci_upper": float(results.avg_conf_int[1]),
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

    print(f"ATT: {results.avg_att:.6f}")
    print(f"SE:  {results.avg_se:.6f}")
    print(f"Completed in {total_time:.3f} seconds")
    return output


if __name__ == "__main__":
    main()
