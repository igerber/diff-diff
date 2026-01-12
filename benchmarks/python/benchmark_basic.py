#!/usr/bin/env python3
"""
Benchmark: Basic DiD / TWFE (diff-diff DifferenceInDifferences / TwoWayFixedEffects).

Usage:
    python benchmark_basic.py --data path/to/data.csv --output path/to/results.json
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
import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diff_diff import DifferenceInDifferences, HAS_RUST_BACKEND
from benchmarks.python.utils import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark basic DiD estimator")
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument(
        "--cluster", default="unit", help="Column to cluster standard errors on"
    )
    parser.add_argument(
        "--type", default="twfe", choices=["basic", "twfe"],
        help="Estimator type (basic or twfe, default: twfe)"
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "python", "rust"],
        help="Backend to use: auto (default), python (pure Python), rust (Rust backend)"
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

    # Run benchmark
    print("Running DiD estimation...")

    # Use DifferenceInDifferences with formula to match R's fixest::feols
    did = DifferenceInDifferences(robust=True, cluster=args.cluster)

    with Timer() as timer:
        results = did.fit(
            data,
            formula="outcome ~ treated * post",
        )

    att = results.att
    se = results.se
    pvalue = results.p_value
    ci = results.conf_int

    total_time = timer.elapsed

    # Build output
    output = {
        "estimator": "diff_diff.DifferenceInDifferences",
        "backend": actual_backend,
        "cluster": args.cluster,
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
            "n_periods": len(data["time"].unique()),
        },
        # Timing
        "timing": {
            "estimation_seconds": total_time,
            "total_seconds": total_time,
        },
        # Metadata
        "metadata": {
            "n_units": len(data["unit"].unique()),
            "n_periods": len(data["time"].unique()),
            "n_obs": len(data),
        },
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
