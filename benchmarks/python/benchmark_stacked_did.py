#!/usr/bin/env python3
"""
Benchmark: Stacked DiD Estimator (diff-diff StackedDiD).

Compares against R's stacked-did-weights reference implementation
(Wing, Freedman & Hollingsworth 2024).

Usage:
    python benchmark_stacked_did.py --data path/to/data.csv --output path/to/results.json
    python benchmark_stacked_did.py --data path/to/data.csv --output path/to/results.json --kappa-pre 2 --kappa-post 2
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

# NOW import diff_diff and other dependencies
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diff_diff import StackedDiD, HAS_RUST_BACKEND
from benchmarks.python.utils import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Stacked DiD estimator")
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "python", "rust"],
        help="Backend to use: auto (default), python (pure Python), rust (Rust backend)",
    )
    parser.add_argument(
        "--kappa-pre",
        type=int,
        default=2,
        help="Number of pre-treatment event-study periods (default: 2)",
    )
    parser.add_argument(
        "--kappa-post",
        type=int,
        default=2,
        help="Number of post-treatment event-study periods (default: 2)",
    )
    return parser.parse_args()


def get_actual_backend() -> str:
    """Return the actual backend being used based on HAS_RUST_BACKEND."""
    return "rust" if HAS_RUST_BACKEND else "python"


def main():
    args = parse_args()

    # Get actual backend
    actual_backend = get_actual_backend()
    print(f"Using backend: {actual_backend}")

    # Load data
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)

    kappa_pre = args.kappa_pre
    kappa_post = args.kappa_post
    print(f"kappa_pre={kappa_pre}, kappa_post={kappa_post}")

    # Run benchmark
    print("Running StackedDiD estimation...")
    est = StackedDiD(
        kappa_pre=kappa_pre,
        kappa_post=kappa_post,
        weighting="aggregate",
        clean_control="not_yet_treated",
        cluster="unit",
    )

    with Timer() as estimation_timer:
        results = est.fit(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

    estimation_time = estimation_timer.elapsed
    total_time = estimation_time

    # Store data info
    n_units = len(df["unit"].unique())
    n_periods = len(df["time"].unique())
    n_obs = len(df)

    # Format event study effects
    es_effects = []
    if results.event_study_effects:
        for rel_t, effect_data in sorted(results.event_study_effects.items()):
            # Skip reference period marker (n_obs == 0)
            if effect_data.get("n_obs", 1) == 0:
                continue
            es_effects.append(
                {
                    "event_time": int(rel_t),
                    "att": float(effect_data["effect"]),
                    "se": float(effect_data["se"]),
                }
            )

    # Build output
    output = {
        "estimator": "diff_diff.StackedDiD",
        "backend": actual_backend,
        # Overall ATT
        "overall_att": float(results.overall_att),
        "overall_se": float(results.overall_se),
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
            "n_stacked_obs": results.n_stacked_obs,
            "n_sub_experiments": results.n_sub_experiments,
            "kappa_pre": kappa_pre,
            "kappa_post": kappa_post,
            "weighting": "aggregate",
            "clean_control": "not_yet_treated",
        },
    }

    # Write output
    print(f"Writing results to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Overall ATT: {results.overall_att:.6f} (SE: {results.overall_se:.6f})")
    if results.event_study_effects:
        for h, eff in sorted(results.event_study_effects.items()):
            if eff.get("n_obs", 1) > 0:
                print(f"  h={h}: ATT={eff['effect']:.6f} (SE={eff['se']:.6f})")
    print(f"Completed in {total_time:.3f} seconds")
    return output


if __name__ == "__main__":
    main()
