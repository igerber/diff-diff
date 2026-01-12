#!/usr/bin/env python3
"""
Benchmark: Callaway-Sant'Anna Estimator (diff-diff CallawaySantAnna).

Usage:
    python benchmark_callaway.py --data path/to/data.csv --output path/to/results.json
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

from diff_diff import CallawaySantAnna, HAS_RUST_BACKEND
from benchmarks.python.utils import BenchmarkResult, Timer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Callaway-Sant'Anna estimator"
    )
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument(
        "--method",
        default="dr",
        choices=["dr", "ipw", "reg"],
        help="Estimation method",
    )
    parser.add_argument(
        "--control-group",
        default="never_treated",
        choices=["never_treated", "not_yet_treated"],
        help="Control group definition",
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
    df = pd.read_csv(args.data)

    # Run benchmark
    print("Running Callaway-Sant'Anna estimation...")
    # Use analytical SE (n_bootstrap=0) - matches R's did package after
    # influence function aggregation fix (accounts for covariance)
    cs = CallawaySantAnna(
        estimation_method=args.method,
        control_group=args.control_group,
        n_bootstrap=0,  # Analytical SE now correct with influence functions
        seed=42,
    )

    with Timer() as estimation_timer:
        results = cs.fit(
            df,
            outcome="outcome",
            time="time",
            unit="unit",
            first_treat="first_treat",
            aggregate="all",  # Get event study and group aggregations
        )

    estimation_time = estimation_timer.elapsed
    total_time = estimation_time

    # Store data info before looping (avoid shadowing)
    n_units = len(df["unit"].unique())
    n_periods = len(df["time"].unique())
    n_obs = len(df)

    # Format group-time effects from results dict
    gt_effects = []
    for (g, t), effect_data in results.group_time_effects.items():
        gt_effects.append(
            {
                "group": int(g),
                "time": int(t),
                "att": float(effect_data["effect"]),
                "se": float(effect_data["se"]),
            }
        )

    # Format event study effects (if available)
    es_effects = []
    if results.event_study_effects:
        for rel_t, effect_data in sorted(results.event_study_effects.items()):
            es_effects.append({
                "event_time": int(rel_t),
                "att": float(effect_data["effect"]),
                "se": float(effect_data["se"]),
            })

    # Format group effects (if available)
    grp_effects = []
    if results.group_effects:
        for g, effect_data in sorted(results.group_effects.items()):
            grp_effects.append({
                "group": int(g),
                "att": float(effect_data["effect"]),
                "se": float(effect_data["se"]),
            })

    # Build output
    output = {
        "estimator": "diff_diff.CallawaySantAnna",
        "backend": actual_backend,
        "method": args.method,
        "control_group": args.control_group,
        # Overall ATT
        "overall_att": float(results.overall_att),
        "overall_se": float(results.overall_se),
        # Group-time effects
        "group_time_effects": gt_effects,
        # Event study
        "event_study": es_effects,
        # Group effects
        "group_effects": grp_effects,
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
