#!/usr/bin/env python3
"""
Benchmark: Synthetic DiD (diff-diff SyntheticDiD).

Usage:
    python benchmark_synthdid.py --data path/to/data.csv --output path/to/results.json
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

from diff_diff import SyntheticDiD, HAS_RUST_BACKEND
from benchmarks.python.utils import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Synthetic DiD estimator")
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument(
        "--n-bootstrap", type=int, default=200, help="Number of bootstrap iterations"
    )
    parser.add_argument(
        "--variance-method", type=str, default="placebo",
        choices=["bootstrap", "placebo"],
        help="Variance estimation method (default: placebo to match R)"
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

    # Determine post periods from data
    # Assumes 'post' column exists with 0/1 indicators
    post_periods = sorted(data[data["post"] == 1]["time"].unique().tolist())

    # Run benchmark
    print("Running Synthetic DiD estimation...")
    sdid = SyntheticDiD(
        n_bootstrap=args.n_bootstrap,
        variance_method=args.variance_method,
        seed=42
    )

    with Timer() as timer:
        results = sdid.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=post_periods,
        )

    total_time = timer.elapsed

    # Get weights
    unit_weights_df = results.get_unit_weights_df()
    time_weights_df = results.get_time_weights_df()

    # Build output
    output = {
        "estimator": "diff_diff.SyntheticDiD",
        "backend": actual_backend,
        # Point estimate and SE
        "att": float(results.att),
        "se": float(results.se),
        # Weights
        "unit_weights": unit_weights_df["weight"].tolist(),
        "time_weights": time_weights_df["weight"].tolist(),
        # Timing
        "timing": {
            "estimation_seconds": total_time,
            "total_seconds": total_time,
        },
        # Metadata
        "metadata": {
            "n_control": len(data[data["treated"] == 0]["unit"].unique()),
            "n_treated": len(data[data["treated"] == 1]["unit"].unique()),
            "n_pre_periods": len(data[data["post"] == 0]["time"].unique()),
            "n_post_periods": len(post_periods),
            "n_bootstrap": args.n_bootstrap,
            "variance_method": args.variance_method,
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
