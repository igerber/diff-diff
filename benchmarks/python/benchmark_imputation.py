#!/usr/bin/env python3
"""
Benchmark: Imputation DiD Estimator (diff-diff ImputationDiD).

Compares against R's didimputation::did_imputation (Borusyak et al. 2024).

Usage:
    python benchmark_imputation.py --data path/to/data.csv --output path/to/results.json
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diff_diff import ImputationDiD, HAS_RUST_BACKEND
from benchmarks.python.utils import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Imputation DiD estimator")
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "python", "rust"],
        help="Backend to use: auto (default), python (pure Python), rust (Rust backend)",
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
    df = pd.read_csv(args.data)

    # Run benchmark
    print("Running ImputationDiD estimation...")
    est = ImputationDiD(
        n_bootstrap=0,  # Analytical SE (conservative variance, Theorem 3)
    )

    with Timer() as estimation_timer:
        results = est.fit(
            df,
            outcome="outcome",
            time="time",
            unit="unit",
            first_treat="first_treat",
        )
        # Post-fit event study stays inside the timed region (the 3.x
        # fit(aggregate="event_study") route, removed at 4.0).
        es = results.aggregate("event_study")

    estimation_time = estimation_timer.elapsed
    total_time = estimation_time

    # Store data info before looping (avoid shadowing)
    n_units = len(df["unit"].unique())
    n_periods = len(df["time"].unique())
    n_obs = len(df)

    # Format event study effects from the post-fit container, skipping the
    # reference-period marker row exactly as the legacy reader did.
    es_effects = []
    for row in es.to_dataframe().sort_values("event_time").itertuples(index=False):
        if bool(row.is_reference):
            continue
        es_effects.append(
            {
                "event_time": int(row.event_time),
                "att": float(row.att),
                "se": float(row.se),
            }
        )

    # Build output
    output = {
        "estimator": "diff_diff.ImputationDiD",
        "backend": actual_backend,
        # Overall ATT
        "overall_att": float(results.att),
        "overall_se": float(results.se),
        # Event study
        "event_study": es_effects,
        # Timing
        "timing": {
            "estimation_seconds": estimation_time,
            "total_seconds": total_time,
        },
        # Metadata
        "metadata": {
            "n_units": n_units,
            "n_periods": n_periods,
            "n_obs": n_obs,
            "n_treated_obs": results.n_treated_obs,
            "n_untreated_obs": results.n_untreated_obs,
        },
    }

    # Write output
    print(f"Writing results to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Overall ATT: {results.att:.6f} (SE: {results.se:.6f})")
    print(f"Completed in {total_time:.3f} seconds")
    return output


if __name__ == "__main__":
    main()
