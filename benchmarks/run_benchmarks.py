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
    compute_timing_stats,
)
from benchmarks.compare_results import (
    compare_estimates,
    generate_comparison_report,
    load_results,
)

# Dataset scale configurations
SCALE_CONFIGS = {
    "small": {
        "staggered": {"n_units": 200, "n_periods": 8, "n_cohorts": 3},
        "basic": {"n_units": 100, "n_periods": 4},
        "sdid": {"n_control": 40, "n_treated": 10, "n_pre": 15, "n_post": 5},
    },
    "1k": {
        "staggered": {"n_units": 1000, "n_periods": 10, "n_cohorts": 4},
        "basic": {"n_units": 1000, "n_periods": 6},
        "sdid": {"n_control": 800, "n_treated": 200, "n_pre": 20, "n_post": 10},
    },
    "5k": {
        "staggered": {"n_units": 5000, "n_periods": 12, "n_cohorts": 5},
        "basic": {"n_units": 5000, "n_periods": 8},
        "sdid": {"n_control": 4000, "n_treated": 1000, "n_pre": 25, "n_post": 15},
    },
    "10k": {
        "staggered": {"n_units": 10000, "n_periods": 15, "n_cohorts": 6},
        "basic": {"n_units": 10000, "n_periods": 10},
        "sdid": {"n_control": 8000, "n_treated": 2000, "n_pre": 30, "n_post": 20},
    },
    "20k": {
        "staggered": {"n_units": 20000, "n_periods": 18, "n_cohorts": 7},
        "basic": {"n_units": 20000, "n_periods": 12},
        "sdid": {"n_control": 16000, "n_treated": 4000, "n_pre": 35, "n_post": 25},
    },
}

# Timeout configurations (seconds) by scale
TIMEOUT_CONFIGS = {
    "small": {"python": 60, "r": 300},
    "1k": {"python": 300, "r": 1800},
    "5k": {"python": 600, "r": 3600},
    "10k": {"python": 1200, "r": 7200},
    "20k": {"python": 2400, "r": 14400},
}


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
    timeout: Optional[int] = None,
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
    timeout : int, optional
        Timeout in seconds.

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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  R script timed out after {timeout}s")
        raise RuntimeError(f"R script {script_name} timed out after {timeout}s")

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
    timeout: Optional[int] = None,
    backend: str = "auto",
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
    timeout : int, optional
        Timeout in seconds.
    backend : str
        Backend to use: 'auto', 'python', or 'rust'.

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
        "--backend", backend,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd[:4])}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  Python script timed out after {timeout}s")
        raise RuntimeError(f"Python script {script_name} timed out after {timeout}s")

    if result.returncode != 0:
        print(f"  Python script failed:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        raise RuntimeError(f"Python script {script_name} failed")

    with open(output_path) as f:
        return json.load(f)


def generate_synthetic_datasets(
    seed: int = 42,
    scales: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Generate all synthetic datasets for benchmarking.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    scales : list of str, optional
        Scales to generate. If None, generates all scales.
        Options: 'small', '1k', '5k', '10k'

    Returns
    -------
    dict
        Mapping of dataset name to file path.
    """
    if scales is None:
        scales = ["small"]  # Default to small for backward compatibility

    print("Generating synthetic datasets...")
    datasets = {}

    for scale in scales:
        if scale not in SCALE_CONFIGS:
            print(f"  WARNING: Unknown scale '{scale}', skipping")
            continue

        config = SCALE_CONFIGS[scale]
        print(f"\n  Scale: {scale}")

        # Staggered data for Callaway-Sant'Anna
        stag_cfg = config["staggered"]
        n_obs = stag_cfg["n_units"] * stag_cfg["n_periods"]
        print(f"    - staggered_{scale} ({stag_cfg['n_units']} units, {stag_cfg['n_periods']} periods, {n_obs:,} obs)")
        staggered_data = generate_staggered_data(
            n_units=stag_cfg["n_units"],
            n_periods=stag_cfg["n_periods"],
            n_cohorts=stag_cfg["n_cohorts"],
            treatment_effect=2.0,
            seed=seed,
        )
        staggered_path = DATA_DIR / "synthetic" / f"staggered_{scale}.csv"
        save_benchmark_data(staggered_data, staggered_path)
        datasets[f"staggered_{scale}"] = staggered_path

        # Basic 2x2 DiD data
        basic_cfg = config["basic"]
        n_obs = basic_cfg["n_units"] * basic_cfg["n_periods"]
        print(f"    - basic_{scale} ({basic_cfg['n_units']} units, {basic_cfg['n_periods']} periods, {n_obs:,} obs)")
        basic_data = generate_basic_did_data(
            n_units=basic_cfg["n_units"],
            n_periods=basic_cfg["n_periods"],
            treatment_effect=5.0,
            seed=seed,
        )
        basic_path = DATA_DIR / "synthetic" / f"basic_{scale}.csv"
        save_benchmark_data(basic_data, basic_path)
        datasets[f"basic_{scale}"] = basic_path

        # Synthetic DiD data
        sdid_cfg = config["sdid"]
        n_units = sdid_cfg["n_control"] + sdid_cfg["n_treated"]
        n_periods = sdid_cfg["n_pre"] + sdid_cfg["n_post"]
        n_obs = n_units * n_periods
        print(f"    - sdid_{scale} ({n_units} units, {n_periods} periods, {n_obs:,} obs)")
        sdid_data = generate_sdid_data(
            n_control=sdid_cfg["n_control"],
            n_treated=sdid_cfg["n_treated"],
            n_pre=sdid_cfg["n_pre"],
            n_post=sdid_cfg["n_post"],
            treatment_effect=4.0,
            seed=seed,
        )
        sdid_path = DATA_DIR / "synthetic" / f"sdid_{scale}.csv"
        save_benchmark_data(sdid_data, sdid_path)
        datasets[f"sdid_{scale}"] = sdid_path

    print(f"\nGenerated {len(datasets)} datasets")
    return datasets


def run_callaway_benchmark(
    data_path: Path,
    name: str = "callaway",
    scale: str = "small",
    n_replications: int = 1,
    backends: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run Callaway-Sant'Anna benchmarks (Python and R) with replications."""
    print(f"\n{'='*60}")
    print(f"CALLAWAY-SANT'ANNA BENCHMARK ({scale})")
    print(f"{'='*60}")

    if backends is None:
        backends = ["python", "rust"]

    timeouts = TIMEOUT_CONFIGS.get(scale, TIMEOUT_CONFIGS["small"])
    results = {
        "name": name,
        "scale": scale,
        "n_replications": n_replications,
        "python_pure": None,
        "python_rust": None,
        "r": None,
        "comparison": None,
    }

    # Run Python benchmark for each backend
    for backend in backends:
        # Map backend name to label (python -> pure, rust -> rust)
        backend_label = f"python_{'pure' if backend == 'python' else backend}"
        print(f"\nRunning Python (diff_diff.CallawaySantAnna, backend={backend}) - {n_replications} replications...")
        py_output = RESULTS_DIR / "accuracy" / f"{backend_label}_{name}_{scale}.json"
        py_output.parent.mkdir(parents=True, exist_ok=True)

        py_timings = []
        py_result = None
        for rep in range(n_replications):
            try:
                py_result = run_python_benchmark(
                    "benchmark_callaway.py", data_path, py_output,
                    timeout=timeouts["python"],
                    backend=backend,
                )
                py_timings.append(py_result["timing"]["total_seconds"])
                if rep == 0:
                    print(f"  ATT: {py_result['overall_att']:.4f}")
                    print(f"  SE:  {py_result['overall_se']:.4f}")
                print(f"  Rep {rep+1}/{n_replications}: {py_timings[-1]:.3f}s")
            except Exception as e:
                print(f"  Rep {rep+1} failed: {e}")

        if py_result and py_timings:
            timing_stats = compute_timing_stats(py_timings)
            py_result["timing"] = timing_stats
            results[backend_label] = py_result
            print(f"  Mean time: {timing_stats['stats']['mean']:.3f}s ± {timing_stats['stats']['std']:.3f}s")

    # For backward compatibility, also store as "python" (use rust if available)
    if results.get("python_rust"):
        results["python"] = results["python_rust"]
    elif results.get("python_pure"):
        results["python"] = results["python_pure"]

    # R benchmark with replications
    print(f"\nRunning R (did::att_gt) - {n_replications} replications...")
    r_output = RESULTS_DIR / "accuracy" / f"r_{name}_{scale}.json"

    r_timings = []
    r_result = None
    for rep in range(n_replications):
        try:
            r_result = run_r_benchmark(
                "benchmark_did.R", data_path, r_output,
                timeout=timeouts["r"]
            )
            r_timings.append(r_result["timing"]["total_seconds"])
            if rep == 0:
                print(f"  ATT: {r_result['overall_att']:.4f}")
                print(f"  SE:  {r_result['overall_se']:.4f}")
            print(f"  Rep {rep+1}/{n_replications}: {r_timings[-1]:.3f}s")
        except Exception as e:
            print(f"  Rep {rep+1} failed: {e}")

    if r_result and r_timings:
        timing_stats = compute_timing_stats(r_timings)
        r_result["timing"] = timing_stats
        results["r"] = r_result
        print(f"  Mean time: {timing_stats['stats']['mean']:.3f}s ± {timing_stats['stats']['std']:.3f}s")

    # Compare results
    if results["python"] and results["r"]:
        print("\nComparison (Python vs R):")
        comparison = compare_estimates(
            results["python"], results["r"], "CallawaySantAnna", scale=scale,
            python_pure_results=results.get("python_pure"),
            python_rust_results=results.get("python_rust"),
        )
        results["comparison"] = comparison
        print(f"  ATT diff: {comparison.att_diff:.2e}")
        print(f"  SE rel diff: {comparison.se_rel_diff:.1%}")
        print(f"  Status: {'PASS' if comparison.passed else 'FAIL'}")

    # Print timing comparison table
    print("\nTiming Comparison:")
    print(f"  {'Backend':<15} {'Time (s)':<12} {'vs R':<12} {'vs Pure Python':<15}")
    print(f"  {'-'*54}")

    r_mean = results["r"]["timing"]["stats"]["mean"] if results["r"] else None
    pure_mean = results["python_pure"]["timing"]["stats"]["mean"] if results.get("python_pure") else None
    rust_mean = results["python_rust"]["timing"]["stats"]["mean"] if results.get("python_rust") else None

    if r_mean:
        print(f"  {'R':<15} {r_mean:<12.3f} {'1.00x':<12} {'-':<15}")
    if pure_mean:
        r_speedup = f"{r_mean/pure_mean:.2f}x" if r_mean else "-"
        print(f"  {'Python (pure)':<15} {pure_mean:<12.3f} {r_speedup:<12} {'1.00x':<15}")
    if rust_mean:
        r_speedup = f"{r_mean/rust_mean:.2f}x" if r_mean else "-"
        pure_speedup = f"{pure_mean/rust_mean:.2f}x" if pure_mean else "-"
        print(f"  {'Python (rust)':<15} {rust_mean:<12.3f} {r_speedup:<12} {pure_speedup:<15}")

    return results


def run_synthdid_benchmark(
    data_path: Path,
    name: str = "synthdid",
    scale: str = "small",
    n_replications: int = 1,
) -> Dict[str, Any]:
    """Run Synthetic DiD benchmarks (Python and R) with replications."""
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DID BENCHMARK ({scale})")
    print(f"{'='*60}")

    timeouts = TIMEOUT_CONFIGS.get(scale, TIMEOUT_CONFIGS["small"])
    results = {
        "name": name,
        "scale": scale,
        "n_replications": n_replications,
        "python": None,
        "r": None,
        "comparison": None,
    }

    # Python benchmark with replications
    print(f"\nRunning Python (diff_diff.SyntheticDiD) - {n_replications} replications...")
    py_output = RESULTS_DIR / "accuracy" / f"python_{name}_{scale}.json"
    py_output.parent.mkdir(parents=True, exist_ok=True)

    py_timings = []
    py_result = None
    for rep in range(n_replications):
        try:
            py_result = run_python_benchmark(
                "benchmark_synthdid.py",
                data_path,
                py_output,
                extra_args=["--n-bootstrap", "50"],
                timeout=timeouts["python"]
            )
            py_timings.append(py_result["timing"]["total_seconds"])
            if rep == 0:
                print(f"  ATT: {py_result['att']:.4f}")
                print(f"  SE:  {py_result['se']:.4f}")
            print(f"  Rep {rep+1}/{n_replications}: {py_timings[-1]:.3f}s")
        except Exception as e:
            print(f"  Rep {rep+1} failed: {e}")

    if py_result and py_timings:
        timing_stats = compute_timing_stats(py_timings)
        py_result["timing"] = timing_stats
        results["python"] = py_result
        print(f"  Mean time: {timing_stats['stats']['mean']:.3f}s ± {timing_stats['stats']['std']:.3f}s")

    # R benchmark with replications
    print(f"\nRunning R (synthdid::synthdid_estimate) - {n_replications} replications...")
    r_output = RESULTS_DIR / "accuracy" / f"r_{name}_{scale}.json"

    r_timings = []
    r_result = None
    for rep in range(n_replications):
        try:
            r_result = run_r_benchmark(
                "benchmark_synthdid.R", data_path, r_output,
                timeout=timeouts["r"]
            )
            r_timings.append(r_result["timing"]["total_seconds"])
            if rep == 0:
                print(f"  ATT: {r_result['att']:.4f}")
                print(f"  SE:  {r_result['se']:.4f}")
            print(f"  Rep {rep+1}/{n_replications}: {r_timings[-1]:.3f}s")
        except Exception as e:
            print(f"  Rep {rep+1} failed: {e}")

    if r_result and r_timings:
        timing_stats = compute_timing_stats(r_timings)
        r_result["timing"] = timing_stats
        results["r"] = r_result
        print(f"  Mean time: {timing_stats['stats']['mean']:.3f}s ± {timing_stats['stats']['std']:.3f}s")

    # Compare results
    if results["python"] and results["r"]:
        print("\nComparison:")
        comparison = compare_estimates(results["python"], results["r"], "SyntheticDiD", scale=scale)
        results["comparison"] = comparison
        print(f"  ATT diff: {comparison.att_diff:.2e}")
        print(f"  SE rel diff: {comparison.se_rel_diff:.1%}")
        print(f"  Status: {'PASS' if comparison.passed else 'FAIL'}")

        # Compute speedup from timing stats
        py_mean = results["python"]["timing"]["stats"]["mean"]
        r_mean = results["r"]["timing"]["stats"]["mean"]
        speedup = r_mean / py_mean if py_mean > 0 else float('inf')
        print(f"  Speed: Python is {speedup:.1f}x faster")

    return results


def run_basic_did_benchmark(
    data_path: Path,
    name: str = "basic",
    scale: str = "small",
    n_replications: int = 1,
    backends: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run basic DiD / TWFE benchmarks (Python and R) with replications."""
    print(f"\n{'='*60}")
    print(f"BASIC DID / TWFE BENCHMARK ({scale})")
    print(f"{'='*60}")

    if backends is None:
        backends = ["python", "rust"]

    timeouts = TIMEOUT_CONFIGS.get(scale, TIMEOUT_CONFIGS["small"])
    results = {
        "name": name,
        "scale": scale,
        "n_replications": n_replications,
        "python_pure": None,
        "python_rust": None,
        "r": None,
        "comparison": None,
    }

    # Run Python benchmark for each backend
    for backend in backends:
        # Map backend name to label (python -> pure, rust -> rust)
        backend_label = f"python_{'pure' if backend == 'python' else backend}"
        print(f"\nRunning Python (diff_diff.DifferenceInDifferences, backend={backend}) - {n_replications} replications...")
        py_output = RESULTS_DIR / "accuracy" / f"{backend_label}_{name}_{scale}.json"
        py_output.parent.mkdir(parents=True, exist_ok=True)

        py_timings = []
        py_result = None
        for rep in range(n_replications):
            try:
                py_result = run_python_benchmark(
                    "benchmark_basic.py", data_path, py_output,
                    extra_args=["--type", "twfe"],
                    timeout=timeouts["python"],
                    backend=backend,
                )
                py_timings.append(py_result["timing"]["total_seconds"])
                if rep == 0:
                    print(f"  ATT: {py_result['att']:.4f}")
                    print(f"  SE:  {py_result['se']:.4f}")
                print(f"  Rep {rep+1}/{n_replications}: {py_timings[-1]:.3f}s")
            except Exception as e:
                print(f"  Rep {rep+1} failed: {e}")

        if py_result and py_timings:
            timing_stats = compute_timing_stats(py_timings)
            py_result["timing"] = timing_stats
            results[backend_label] = py_result
            print(f"  Mean time: {timing_stats['stats']['mean']:.3f}s ± {timing_stats['stats']['std']:.3f}s")

    # For backward compatibility, also store as "python" (use rust if available)
    if results.get("python_rust"):
        results["python"] = results["python_rust"]
    elif results.get("python_pure"):
        results["python"] = results["python_pure"]

    # R benchmark with replications
    print(f"\nRunning R (fixest::feols) - {n_replications} replications...")
    r_output = RESULTS_DIR / "accuracy" / f"r_{name}_{scale}.json"

    r_timings = []
    r_result = None
    for rep in range(n_replications):
        try:
            r_result = run_r_benchmark(
                "benchmark_fixest.R", data_path, r_output,
                extra_args=["--type", "twfe"],
                timeout=timeouts["r"]
            )
            r_timings.append(r_result["timing"]["total_seconds"])
            if rep == 0:
                print(f"  ATT: {r_result['att']:.4f}")
                print(f"  SE:  {r_result['se']:.4f}")
            print(f"  Rep {rep+1}/{n_replications}: {r_timings[-1]:.3f}s")
        except Exception as e:
            print(f"  Rep {rep+1} failed: {e}")

    if r_result and r_timings:
        timing_stats = compute_timing_stats(r_timings)
        r_result["timing"] = timing_stats
        results["r"] = r_result
        print(f"  Mean time: {timing_stats['stats']['mean']:.3f}s ± {timing_stats['stats']['std']:.3f}s")

    # Compare results
    if results["python"] and results["r"]:
        print("\nComparison (Python vs R):")
        comparison = compare_estimates(
            results["python"], results["r"], "BasicDiD/TWFE", scale=scale,
            python_pure_results=results.get("python_pure"),
            python_rust_results=results.get("python_rust"),
        )
        results["comparison"] = comparison
        print(f"  ATT diff: {comparison.att_diff:.2e}")
        print(f"  SE rel diff: {comparison.se_rel_diff:.1%}")
        print(f"  Status: {'PASS' if comparison.passed else 'FAIL'}")

    # Print timing comparison table
    print("\nTiming Comparison:")
    print(f"  {'Backend':<15} {'Time (s)':<12} {'vs R':<12} {'vs Pure Python':<15}")
    print(f"  {'-'*54}")

    r_mean = results["r"]["timing"]["stats"]["mean"] if results["r"] else None
    pure_mean = results["python_pure"]["timing"]["stats"]["mean"] if results.get("python_pure") else None
    rust_mean = results["python_rust"]["timing"]["stats"]["mean"] if results.get("python_rust") else None

    if r_mean:
        print(f"  {'R':<15} {r_mean:<12.3f} {'1.00x':<12} {'-':<15}")
    if pure_mean:
        r_speedup = f"{r_mean/pure_mean:.2f}x" if r_mean else "-"
        print(f"  {'Python (pure)':<15} {pure_mean:<12.3f} {r_speedup:<12} {'1.00x':<15}")
    if rust_mean:
        r_speedup = f"{r_mean/rust_mean:.2f}x" if r_mean else "-"
        pure_speedup = f"{pure_mean/rust_mean:.2f}x" if pure_mean else "-"
        print(f"  {'Python (rust)':<15} {rust_mean:<12.3f} {r_speedup:<12} {pure_speedup:<15}")

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
    parser.add_argument(
        "--replications",
        type=int,
        default=1,
        help="Number of replications for timing measurements (default: 1)",
    )
    parser.add_argument(
        "--scale",
        choices=["small", "1k", "5k", "10k", "20k", "all"],
        default="small",
        help="Dataset scale to use (default: small). Use 'all' for all scales.",
    )
    args = parser.parse_args()

    # Determine which scales to run
    if args.scale == "all":
        scales = ["small", "1k", "5k", "10k", "20k"]
    else:
        scales = [args.scale]

    # Check R installation
    if not check_r_installation():
        print("WARNING: R is not installed or not accessible.")
        print("R benchmarks will be skipped.")
        print("Install R with: brew install r")
        print("Then install packages: Rscript benchmarks/R/requirements.R")

    # Generate synthetic datasets for requested scales
    datasets = generate_synthetic_datasets(seed=args.seed, scales=scales)

    if args.generate_data_only:
        print("\nData generation complete. Datasets saved to:")
        for name, path in datasets.items():
            print(f"  {name}: {path}")
        return

    # Run benchmarks for each scale
    all_results = []

    for scale in scales:
        print(f"\n{'#'*60}")
        print(f"# SCALE: {scale.upper()}")
        print(f"{'#'*60}")

        if args.all or args.estimator == "callaway":
            stag_key = f"staggered_{scale}"
            if stag_key in datasets:
                results = run_callaway_benchmark(
                    datasets[stag_key],
                    scale=scale,
                    n_replications=args.replications,
                )
                all_results.append(results)

        if args.all or args.estimator == "synthdid":
            sdid_key = f"sdid_{scale}"
            if sdid_key in datasets:
                results = run_synthdid_benchmark(
                    datasets[sdid_key],
                    scale=scale,
                    n_replications=args.replications,
                )
                all_results.append(results)

        if args.all or args.estimator == "basic":
            basic_key = f"basic_{scale}"
            if basic_key in datasets:
                results = run_basic_did_benchmark(
                    datasets[basic_key],
                    scale=scale,
                    n_replications=args.replications,
                )
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
