#!/usr/bin/env python3
"""
Benchmark: WooldridgeDiD (ETWFE) Estimator (diff-diff WooldridgeDiD).

Validates OLS ETWFE ATT(g,t) against Callaway-Sant'Anna on mpdta data
(Proposition 3.1 equivalence), and measures estimation timing.

Usage:
    python benchmark_wooldridge.py --data path/to/mpdta.csv --output path/to/results.json
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diff_diff import WooldridgeDiD, HAS_RUST_BACKEND
from benchmarks.python.utils import Timer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark WooldridgeDiD (ETWFE) estimator"
    )
    parser.add_argument("--data", required=True, help="Path to input CSV data (mpdta format)")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
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

    actual_backend = get_actual_backend()
    print(f"Using backend: {actual_backend}")

    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)

    # Run OLS ETWFE estimation
    print("Running WooldridgeDiD (OLS ETWFE) estimation...")
    est = WooldridgeDiD(method="ols", control_group="not_yet_treated")

    with Timer() as estimation_timer:
        results = est.fit(
            df,
            outcome="lemp",
            unit="countyreal",
            time="year",
            cohort="first_treat",
        )

    estimation_time = estimation_timer.elapsed

    # Compute event study aggregation
    results.aggregate("event")
    total_time = estimation_timer.elapsed

    # Store data info
    n_units = len(df["countyreal"].unique())
    n_periods = len(df["year"].unique())
    n_obs = len(df)

    # Format ATT(g,t) effects
    gt_effects_out = []
    for (g, t), cell in sorted(results.group_time_effects.items()):
        gt_effects_out.append({
            "cohort": int(g),
            "time": int(t),
            "att": float(cell["att"]),
            "se": float(cell["se"]),
        })

    # Format event study effects
    es_effects = []
    if results.event_study_effects:
        for rel_t, effect_data in sorted(results.event_study_effects.items()):
            es_effects.append({
                "event_time": int(rel_t),
                "att": float(effect_data["att"]),
                "se": float(effect_data["se"]),
            })

    output = {
        "estimator": "diff_diff.WooldridgeDiD",
        "method": "ols",
        "control_group": "not_yet_treated",
        "backend": actual_backend,
        # Overall ATT
        "overall_att": float(results.overall_att),
        "overall_se": float(results.overall_se),
        # Group-time ATT(g,t)
        "group_time_effects": gt_effects_out,
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
            "n_cohorts": len(results.groups),
        },
    }

    print(f"Writing results to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Overall ATT: {results.overall_att:.6f} (SE: {results.overall_se:.6f})")
    print(f"Completed in {total_time:.3f} seconds")
    return output


if __name__ == "__main__":
    main()
