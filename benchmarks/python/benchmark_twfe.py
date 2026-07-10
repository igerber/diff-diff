#!/usr/bin/env python3
"""
Benchmark: TwoWayFixedEffects (diff-diff TwoWayFixedEffects class).

This benchmarks the actual TwoWayFixedEffects class with within-transformation,
as opposed to benchmark_basic.py which uses DifferenceInDifferences with formula.

Usage:
    python benchmark_twfe.py --data path/to/data.csv --output path/to/results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# IMPORTANT: Parse --backend and set environment variable BEFORE importing diff_diff
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

from diff_diff import TwoWayFixedEffects, HAS_RUST_BACKEND
from benchmarks.python.utils import Timer, collect_provenance


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TwoWayFixedEffects estimator")
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
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

    actual_backend = get_actual_backend()
    print(f"Using backend: {actual_backend}")

    # Load data
    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data)

    # Run benchmark using TwoWayFixedEffects (within-transformation approach)
    print("Running TWFE estimation...")

    if args.warmup:
        print("Warm-up fit (untimed)...")
        TwoWayFixedEffects(robust=True).fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

    twfe = TwoWayFixedEffects(robust=True)  # auto-clusters at unit level

    with Timer() as timer:
        results = twfe.fit(
            data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
        )

    att = results.att
    se = results.se
    pvalue = results.p_value
    ci = results.conf_int

    total_time = timer.elapsed

    # Build output
    output = {
        "estimator": "diff_diff.TwoWayFixedEffects",
        "backend": actual_backend,
        "cluster": "unit",
        # Treatment effect
        "att": float(att),
        "se": float(se),
        "pvalue": float(pvalue),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        # Model statistics
        "model_stats": {
            "n_obs": len(data),
            "n_units": len(data["unit"].unique()),
            "n_periods": len(data["post"].unique()),
        },
        # Timing
        "timing": {
            "estimation_seconds": total_time,
            "total_seconds": total_time,
        },
        # Metadata
        "metadata": {
            "n_units": len(data["unit"].unique()),
            "n_periods": len(data["post"].unique()),
            "n_obs": len(data),
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

    print(f"Completed in {total_time:.3f} seconds")
    return output


if __name__ == "__main__":
    main()
