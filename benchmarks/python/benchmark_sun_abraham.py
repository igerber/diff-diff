#!/usr/bin/env python3
"""
Benchmark: SunAbraham interaction-weighted estimator (diff-diff SunAbraham class).

This benchmarks the SunAbraham estimator with cluster-robust SEs,
matching R's fixest::sunab() approach.

Usage:
    python benchmark_sun_abraham.py --data path/to/data.csv --output path/to/results.json
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
import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diff_diff import SunAbraham, HAS_RUST_BACKEND
from benchmarks.python.utils import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SunAbraham estimator")
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

    actual_backend = get_actual_backend()
    print(f"Using backend: {actual_backend}")

    # Load data
    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data)

    # Run benchmark using SunAbraham (analytical SEs, no bootstrap)
    print("Running Sun-Abraham estimation...")

    sa = SunAbraham(control_group="never_treated", n_bootstrap=0)

    with Timer() as timer:
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

    overall_att = results.att
    overall_se = results.se
    overall_pvalue = results.p_value

    # Extract event study effects
    event_study = []
    for e in sorted(results.event_study_effects.keys()):
        eff = results.event_study_effects[e]
        event_study.append(
            {
                "event_time": int(e),
                "att": float(eff["effect"]),
                "se": float(eff["se"]),
            }
        )

    total_time = timer.elapsed

    # Build output
    output = {
        "estimator": "diff_diff.SunAbraham",
        "backend": actual_backend,
        "cluster": "unit",
        # Overall ATT
        "overall_att": float(overall_att),
        "overall_se": float(overall_se),
        "overall_pvalue": float(overall_pvalue),
        # Event study effects
        "event_study": event_study,
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
            "n_groups": len(results.groups),
            "n_event_study_effects": len(event_study),
        },
    }

    # Write output
    print(f"Writing results to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Overall ATT: {overall_att:.6f} (SE: {overall_se:.6f})")
    print(f"Completed in {total_time:.3f} seconds")
    return output


if __name__ == "__main__":
    main()
