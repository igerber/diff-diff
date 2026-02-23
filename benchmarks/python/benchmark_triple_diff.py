#!/usr/bin/env python3
"""
Benchmark: TripleDifference (diff-diff TripleDifference class).

Usage:
    python benchmark_triple_diff.py --data path/to/data.csv --output path/to/results.json \
        [--method dr|reg|ipw] [--covariates]
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

from diff_diff import TripleDifference
from benchmarks.python.utils import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TripleDifference estimator")
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument(
        "--method", default="dr", choices=["dr", "reg", "ipw"],
        help="Estimation method: dr (default), reg, ipw"
    )
    parser.add_argument(
        "--covariates", action="store_true",
        help="Include covariates (columns starting with 'cov')"
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "python", "rust"],
        help="Backend to use: auto (default), python (pure Python), rust (Rust backend)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data)

    # Map R column names to Python convention
    column_map = {"y": "outcome", "state": "group", "id": "unit_id"}
    data = data.rename(columns=column_map)

    # Map R time encoding {1, 2} to Python {0, 1}
    if data["time"].min() == 1:
        data["time"] = data["time"] - 1

    # Detect covariate columns
    cov_cols = [c for c in data.columns if c.startswith("cov")]
    covariates = cov_cols if args.covariates and cov_cols else None

    print(f"Running DDD estimation (method={args.method}, "
          f"covariates={covariates is not None})...")

    ddd = TripleDifference(estimation_method=args.method)

    with Timer() as timer:
        results = ddd.fit(
            data,
            outcome="outcome",
            group="group",
            partition="partition",
            time="time",
            covariates=covariates,
        )

    # Compute CI bounds
    ci_lower, ci_upper = results.conf_int

    output = {
        "ATT": results.att,
        "se": results.se,
        "lci": ci_lower,
        "uci": ci_upper,
        "method": args.method,
        "covariates": args.covariates,
        "n_obs": results.n_obs,
        "elapsed_seconds": timer.elapsed,
    }

    print(f"Writing results to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("Done.")
    print(f"  ATT = {results.att:.6f}")
    print(f"  SE  = {results.se:.6f}")
    print(f"  Time: {timer.elapsed:.3f}s")


if __name__ == "__main__":
    main()
