#!/usr/bin/env python3
"""
Benchmark: HonestDiD Sensitivity Analysis (diff-diff HonestDiD).

Usage:
    python benchmark_honest.py --beta path/to/beta.json --sigma path/to/sigma.csv --output path/to/results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diff_diff import HonestDiD, DeltaRM, DeltaSD
from benchmarks.python.utils import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark HonestDiD estimator")
    parser.add_argument("--beta", required=True, help="Path to beta JSON array")
    parser.add_argument("--sigma", required=True, help="Path to sigma CSV matrix")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument("--num-pre", type=int, default=4, help="Number of pre-periods")
    parser.add_argument(
        "--num-post", type=int, default=1, help="Number of post-periods"
    )
    parser.add_argument(
        "--m-grid",
        default="0,0.5,1,1.5,2",
        help="Comma-separated M values for sensitivity grid",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load inputs
    print(f"Loading beta from: {args.beta}")
    with open(args.beta) as f:
        beta = np.array(json.load(f))

    print(f"Loading sigma from: {args.sigma}")
    sigma = pd.read_csv(args.sigma, header=None).values

    # Parse M grid
    m_grid = [float(x) for x in args.m_grid.split(",")]

    print(f"Running HonestDiD with {args.num_pre} pre-periods, {args.num_post} post-periods")
    print(f"M grid: {m_grid}")

    # Create a mock results object with the beta and sigma
    # HonestDiD expects event study results with coefficients and covariance
    class MockEventStudyResults:
        def __init__(self, beta, sigma, num_pre, num_post):
            self.coefficients = beta
            self.vcov = sigma
            self.n_pre_periods = num_pre
            self.n_post_periods = num_post
            # Create period effects for compatibility
            self.period_effects = []
            for i, b in enumerate(beta):
                event_time = i - num_pre + 1 if i < num_pre else i - num_pre + 1
                self.period_effects.append(type('PeriodEffect', (), {
                    'period': event_time,
                    'coefficient': b,
                    'se': np.sqrt(sigma[i, i]) if i < len(sigma) else 0.1
                })())

    mock_results = MockEventStudyResults(beta, sigma, args.num_pre, args.num_post)

    # Run Delta RM sensitivity analysis
    print("Running Delta RM sensitivity analysis...")
    rm_results_list = []
    with Timer() as rm_timer:
        for m_bar in m_grid:
            honest = HonestDiD(delta=DeltaRM(Mbar=m_bar))
            try:
                result = honest.fit(
                    mock_results,
                    target_post_period=1,  # First post period
                )
                rm_results_list.append({
                    "Mbar": m_bar,
                    "lb": float(result.lb),
                    "ub": float(result.ub),
                })
            except Exception as e:
                print(f"  Warning: DeltaRM with Mbar={m_bar} failed: {e}")
                rm_results_list.append({
                    "Mbar": m_bar,
                    "lb": None,
                    "ub": None,
                })
    rm_time = rm_timer.elapsed

    # Run Delta SD sensitivity analysis
    print("Running Delta SD sensitivity analysis...")
    sd_m_grid = [m / 10 for m in m_grid]  # Scale down for smoothness
    sd_results_list = []
    with Timer() as sd_timer:
        for m in sd_m_grid:
            honest = HonestDiD(delta=DeltaSD(M=m))
            try:
                result = honest.fit(
                    mock_results,
                    target_post_period=1,
                )
                sd_results_list.append({
                    "M": m,
                    "lb": float(result.lb),
                    "ub": float(result.ub),
                })
            except Exception as e:
                print(f"  Warning: DeltaSD with M={m} failed: {e}")
                sd_results_list.append({
                    "M": m,
                    "lb": None,
                    "ub": None,
                })
    sd_time = sd_timer.elapsed

    total_time = rm_time + sd_time

    # Build output
    output = {
        "estimator": "diff_diff.HonestDiD",
        # Delta RM results
        "delta_rm": {
            "M_grid": m_grid,
            "results": rm_results_list,
        },
        # Delta SD results
        "delta_sd": {
            "M_grid": sd_m_grid,
            "results": sd_results_list,
        },
        # Configuration
        "config": {
            "num_pre_periods": args.num_pre,
            "num_post_periods": args.num_post,
            "beta": beta.tolist(),
            "sigma_diag": np.diag(sigma).tolist(),
        },
        # Timing
        "timing": {
            "delta_rm_seconds": rm_time,
            "delta_sd_seconds": sd_time,
            "total_seconds": total_time,
        },
        # Metadata
        "metadata": {},
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
