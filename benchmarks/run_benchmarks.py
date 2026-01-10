#!/usr/bin/env python3
"""
Main benchmark runner for diff-diff vs R packages.

This script orchestrates benchmarks across Python and R, generates synthetic
datasets, runs both implementations, and compares results.

Usage:
    python run_benchmarks.py --all
    python run_benchmarks.py --estimator callaway
    python run_benchmarks.py --estimator synthdid
    python run_benchmarks.py --generate-data-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Setup paths
BENCHMARK_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARK_DIR.parent
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_DIR = BENCHMARK_DIR / "results"

sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.python.utils import (
    generate_staggered_data,
    generate_basic_did_data,
    generate_sdid_data,
    save_benchmark_data,
)
from benchmarks.compare_results import (
    compare_estimates,
    generate_comparison_report,
    load_results,
)


def check_r_installation() -> bool:
    """Check if R is installed and accessible."""
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_r_benchmark(
    script_name: str,
    data_path: Path,
    output_path: Path,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute R benchmark script and return results.

    Parameters
    ----------
    script_name : str
        Name of R script in benchmarks/R directory.
    data_path : Path
        Path to input data CSV.
    output_path : Path
        Path for output JSON.
    extra_args : list, optional
        Additional command line arguments.

    Returns
    -------
    dict
        Parsed JSON results from R script.
    """
    r_script = BENCHMARK_DIR / "R" / script_name

    cmd = [
        "Rscript",
        str(r_script),
        "--data", str(data_path),
        "--output", str(output_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  R script failed:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        raise RuntimeError(f"R script {script_name} failed")

    with open(output_path) as f:
        return json.load(f)


def run_python_benchmark(
    script_name: str,
    data_path: Path,
    output_path: Path,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute Python benchmark script and return results.

    Parameters
    ----------
    script_name : str
        Name of Python script in benchmarks/python directory.
    data_path : Path
        Path to input data CSV.
    output_path : Path
        Path for output JSON.
    extra_args : list, optional
        Additional command line arguments.

    Returns
    -------
    dict
        Parsed JSON results from Python script.
    """
    py_script = BENCHMARK_DIR / "python" / script_name

    cmd = [
        sys.executable,
        str(py_script),
        "--data", str(data_path),
        "--output", str(output_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Python script failed:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        raise RuntimeError(f"Python script {script_name} failed")

    with open(output_path) as f:
        return json.load(f)


def generate_synthetic_datasets(seed: int = 42) -> Dict[str, Path]:
    """
    Generate all synthetic datasets for benchmarking.

    Returns
    -------
    dict
        Mapping of dataset name to file path.
    """
    print("Generating synthetic datasets...")
    datasets = {}

    # Staggered data for Callaway-Sant'Anna
    print("  - staggered_simple (200 units, 8 periods)")
    staggered_data = generate_staggered_data(
        n_units=200, n_periods=8, n_cohorts=3, treatment_effect=2.0, seed=seed
    )
    staggered_path = DATA_DIR / "synthetic" / "staggered_simple.csv"
    save_benchmark_data(staggered_data, staggered_path)
    datasets["staggered_simple"] = staggered_path

    # Large staggered for performance testing
    print("  - staggered_large (500 units, 15 periods)")
    staggered_large = generate_staggered_data(
        n_units=500, n_periods=15, n_cohorts=5, treatment_effect=2.0, seed=seed
    )
    staggered_large_path = DATA_DIR / "synthetic" / "staggered_large.csv"
    save_benchmark_data(staggered_large, staggered_large_path)
    datasets["staggered_large"] = staggered_large_path

    # Basic 2x2 DiD data
    print("  - basic_did (100 units, 4 periods)")
    basic_data = generate_basic_did_data(
        n_units=100, n_periods=4, treatment_effect=5.0, seed=seed
    )
    basic_path = DATA_DIR / "synthetic" / "basic_did.csv"
    save_benchmark_data(basic_data, basic_path)
    datasets["basic_did"] = basic_path

    # Synthetic DiD data
    print("  - sdid_data (50 units, 20 periods)")
    sdid_data = generate_sdid_data(
        n_control=40, n_treated=10, n_pre=15, n_post=5, treatment_effect=4.0, seed=seed
    )
    sdid_path = DATA_DIR / "synthetic" / "sdid_data.csv"
    save_benchmark_data(sdid_data, sdid_path)
    datasets["sdid_data"] = sdid_path

    print(f"Generated {len(datasets)} datasets")
    return datasets


def run_callaway_benchmark(
    data_path: Path,
    name: str = "callaway",
) -> Dict[str, Any]:
    """Run Callaway-Sant'Anna benchmarks (Python and R)."""
    print(f"\n{'='*60}")
    print(f"CALLAWAY-SANT'ANNA BENCHMARK")
    print(f"{'='*60}")

    results = {"name": name, "python": None, "r": None, "comparison": None}

    # Python benchmark
    print("\nRunning Python (diff_diff.CallawaySantAnna)...")
    py_output = RESULTS_DIR / "accuracy" / f"python_{name}.json"
    py_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        results["python"] = run_python_benchmark(
            "benchmark_callaway.py", data_path, py_output
        )
        print(f"  ATT: {results['python']['overall_att']:.4f}")
        print(f"  SE:  {results['python']['overall_se']:.4f}")
        print(f"  Time: {results['python']['timing']['total_seconds']:.3f}s")
    except Exception as e:
        print(f"  Failed: {e}")

    # R benchmark
    print("\nRunning R (did::att_gt)...")
    r_output = RESULTS_DIR / "accuracy" / f"r_{name}.json"
    try:
        results["r"] = run_r_benchmark("benchmark_did.R", data_path, r_output)
        print(f"  ATT: {results['r']['overall_att']:.4f}")
        print(f"  SE:  {results['r']['overall_se']:.4f}")
        print(f"  Time: {results['r']['timing']['total_seconds']:.3f}s")
    except Exception as e:
        print(f"  Failed: {e}")

    # Compare results
    if results["python"] and results["r"]:
        print("\nComparison:")
        comparison = compare_estimates(
            results["python"], results["r"], "CallawaySantAnna"
        )
        results["comparison"] = comparison
        print(f"  ATT diff: {comparison.att_diff:.2e}")
        print(f"  SE rel diff: {comparison.se_rel_diff:.1%}")
        print(f"  Status: {'PASS' if comparison.passed else 'FAIL'}")
        print(f"  Speed: Python is {1/comparison.time_ratio:.1f}x {'faster' if comparison.time_ratio < 1 else 'slower'}")

    return results


def run_synthdid_benchmark(
    data_path: Path,
    name: str = "synthdid",
) -> Dict[str, Any]:
    """Run Synthetic DiD benchmarks (Python and R)."""
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DID BENCHMARK")
    print(f"{'='*60}")

    results = {"name": name, "python": None, "r": None, "comparison": None}

    # Python benchmark
    print("\nRunning Python (diff_diff.SyntheticDiD)...")
    py_output = RESULTS_DIR / "accuracy" / f"python_{name}.json"
    py_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        results["python"] = run_python_benchmark(
            "benchmark_synthdid.py",
            data_path,
            py_output,
            extra_args=["--n-bootstrap", "50"],  # Fewer for speed
        )
        print(f"  ATT: {results['python']['att']:.4f}")
        print(f"  SE:  {results['python']['se']:.4f}")
        print(f"  Time: {results['python']['timing']['total_seconds']:.3f}s")
    except Exception as e:
        print(f"  Failed: {e}")

    # R benchmark
    print("\nRunning R (synthdid::synthdid_estimate)...")
    r_output = RESULTS_DIR / "accuracy" / f"r_{name}.json"
    try:
        results["r"] = run_r_benchmark("benchmark_synthdid.R", data_path, r_output)
        print(f"  ATT: {results['r']['att']:.4f}")
        print(f"  SE:  {results['r']['se']:.4f}")
        print(f"  Time: {results['r']['timing']['total_seconds']:.3f}s")
    except Exception as e:
        print(f"  Failed: {e}")

    # Compare results
    if results["python"] and results["r"]:
        print("\nComparison:")
        comparison = compare_estimates(results["python"], results["r"], "SyntheticDiD")
        results["comparison"] = comparison
        print(f"  ATT diff: {comparison.att_diff:.2e}")
        print(f"  SE rel diff: {comparison.se_rel_diff:.1%}")
        print(f"  Status: {'PASS' if comparison.passed else 'FAIL'}")

    return results


def run_basic_did_benchmark(
    data_path: Path,
    name: str = "basic",
) -> Dict[str, Any]:
    """Run basic DiD / TWFE benchmarks (Python and R)."""
    print(f"\n{'='*60}")
    print(f"BASIC DID / TWFE BENCHMARK")
    print(f"{'='*60}")

    results = {"name": name, "python": None, "r": None, "comparison": None}

    # Python benchmark
    print("\nRunning Python (diff_diff.TwoWayFixedEffects)...")
    py_output = RESULTS_DIR / "accuracy" / f"python_{name}.json"
    py_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        results["python"] = run_python_benchmark(
            "benchmark_basic.py", data_path, py_output, extra_args=["--type", "twfe"]
        )
        print(f"  ATT: {results['python']['att']:.4f}")
        print(f"  SE:  {results['python']['se']:.4f}")
        print(f"  Time: {results['python']['timing']['total_seconds']:.3f}s")
    except Exception as e:
        print(f"  Failed: {e}")

    # R benchmark
    print("\nRunning R (fixest::feols)...")
    r_output = RESULTS_DIR / "accuracy" / f"r_{name}.json"
    try:
        results["r"] = run_r_benchmark(
            "benchmark_fixest.R", data_path, r_output, extra_args=["--type", "twfe"]
        )
        print(f"  ATT: {results['r']['att']:.4f}")
        print(f"  SE:  {results['r']['se']:.4f}")
        print(f"  Time: {results['r']['timing']['total_seconds']:.3f}s")
    except Exception as e:
        print(f"  Failed: {e}")

    # Compare results
    if results["python"] and results["r"]:
        print("\nComparison:")
        comparison = compare_estimates(results["python"], results["r"], "BasicDiD/TWFE")
        results["comparison"] = comparison
        print(f"  ATT diff: {comparison.att_diff:.2e}")
        print(f"  SE rel diff: {comparison.se_rel_diff:.1%}")
        print(f"  Status: {'PASS' if comparison.passed else 'FAIL'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run diff-diff benchmarks against R packages"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--estimator",
        choices=["callaway", "synthdid", "basic"],
        help="Run specific estimator benchmark",
    )
    parser.add_argument(
        "--generate-data-only",
        action="store_true",
        help="Only generate synthetic datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation",
    )
    args = parser.parse_args()

    # Check R installation
    if not check_r_installation():
        print("WARNING: R is not installed or not accessible.")
        print("R benchmarks will be skipped.")
        print("Install R with: brew install r")
        print("Then install packages: Rscript benchmarks/R/requirements.R")

    # Generate synthetic datasets
    datasets = generate_synthetic_datasets(seed=args.seed)

    if args.generate_data_only:
        print("\nData generation complete. Datasets saved to:")
        for name, path in datasets.items():
            print(f"  {name}: {path}")
        return

    # Run benchmarks
    all_results = []

    if args.all or args.estimator == "callaway":
        results = run_callaway_benchmark(datasets["staggered_simple"])
        all_results.append(results)

    if args.all or args.estimator == "synthdid":
        results = run_synthdid_benchmark(datasets["sdid_data"])
        all_results.append(results)

    if args.all or args.estimator == "basic":
        results = run_basic_did_benchmark(datasets["basic_did"])
        all_results.append(results)

    # Generate summary report
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        comparisons = [r["comparison"] for r in all_results if r.get("comparison")]
        if comparisons:
            report = generate_comparison_report(
                comparisons, RESULTS_DIR / "comparison_report.txt"
            )
            print(report)
        else:
            print("No comparisons available.")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
