#!/usr/bin/env python3
"""
Compare benchmark results between Python (diff-diff) and R packages.

This module validates numerical accuracy and reports performance differences.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ComparisonResult:
    """Results from comparing Python and R estimates."""

    estimator: str
    python_att: float
    r_att: float
    att_diff: float
    att_rel_diff: float
    python_se: float
    r_se: float
    se_rel_diff: float
    python_time: float
    r_time: float
    time_ratio: float  # Python time / R time
    ci_overlap: bool
    passed: bool
    notes: str = ""
    # Optional timing stats for multi-replication benchmarks
    python_time_std: float = 0.0
    r_time_std: float = 0.0
    n_replications: int = 1
    scale: str = "small"

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"

        # Format time string with optional std
        if self.n_replications > 1:
            py_time_str = f"{self.python_time:.3f}s ± {self.python_time_std:.3f}s"
            r_time_str = f"{self.r_time:.3f}s ± {self.r_time_std:.3f}s"
        else:
            py_time_str = f"{self.python_time:.3f}s"
            r_time_str = f"{self.r_time:.3f}s"

        return (
            f"{self.estimator} ({self.scale}): [{status}]\n"
            f"  ATT:  Python={self.python_att:.6f}, R={self.r_att:.6f}, "
            f"diff={self.att_diff:.2e}\n"
            f"  SE:   Python={self.python_se:.6f}, R={self.r_se:.6f}, "
            f"rel_diff={self.se_rel_diff:.1%}\n"
            f"  Time: Python={py_time_str}, R={r_time_str}, "
            f"speedup={1/self.time_ratio:.1f}x\n"
            f"  CI overlap: {self.ci_overlap}"
            + (f" (n={self.n_replications})" if self.n_replications > 1 else "")
        )


def compare_estimates(
    python_results: Dict[str, Any],
    r_results: Dict[str, Any],
    estimator: str,
    atol: float = 1e-4,
    se_rtol: float = 0.10,
    scale: str = "small",
) -> ComparisonResult:
    """
    Compare Python and R estimates for numerical equivalence.

    Parameters
    ----------
    python_results : dict
        Results from Python estimator.
    r_results : dict
        Results from R package.
    estimator : str
        Name of the estimator being compared.
    atol : float
        Absolute tolerance for ATT comparison.
    se_rtol : float
        Relative tolerance for SE comparison (default 10%).
    scale : str
        Dataset scale used for this comparison.

    Returns
    -------
    ComparisonResult
        Detailed comparison results.
    """
    # Extract ATT and SE
    py_att = python_results.get("overall_att", python_results.get("att", 0))
    r_att = r_results.get("overall_att", r_results.get("att", 0))

    py_se = python_results.get("overall_se", python_results.get("se", 0))
    r_se = r_results.get("overall_se", r_results.get("se", 0))

    # Extract timing - handle both old format (total_seconds) and new format (stats.mean)
    py_timing = python_results.get("timing", {})
    r_timing = r_results.get("timing", {})

    # New format with stats
    py_stats = py_timing.get("stats", {})
    r_stats = r_timing.get("stats", {})

    # Get mean timing (fall back to total_seconds for backward compatibility)
    py_time = py_stats.get("mean", py_timing.get("total_seconds", 0))
    r_time = r_stats.get("mean", r_timing.get("total_seconds", 0))

    # Get std (0 if not available)
    py_time_std = py_stats.get("std", 0)
    r_time_std = r_stats.get("std", 0)

    # Get number of replications
    n_reps = py_timing.get("n_reps", 1)

    # Compute differences
    att_diff = abs(py_att - r_att)
    att_rel_diff = att_diff / (abs(r_att) + 1e-10)
    se_rel_diff = abs(py_se - r_se) / (r_se + 1e-10) if r_se > 0 else 0

    # Check CI overlap
    py_ci = (py_att - 1.96 * py_se, py_att + 1.96 * py_se)
    r_ci = (r_att - 1.96 * r_se, r_att + 1.96 * r_se)
    ci_overlap = (py_ci[0] <= r_att <= py_ci[1]) or (r_ci[0] <= py_att <= r_ci[1])

    # Time ratio (< 1 means Python is faster)
    time_ratio = py_time / r_time if r_time > 0 else float("inf")

    # Determine pass/fail
    # ATT must be close, and either SE must be close OR confidence intervals must overlap
    # (CI overlap indicates same statistical conclusions despite different SE methods)
    att_ok = att_diff < atol or att_rel_diff < 0.01
    se_ok = se_rel_diff < se_rtol
    passed = att_ok and (se_ok or ci_overlap)

    notes = []
    if not att_ok:
        notes.append(f"ATT diff too large: {att_diff:.2e}")
    if not se_ok and not ci_overlap:
        notes.append(f"SE rel diff too large: {se_rel_diff:.1%}")
    elif not se_ok and ci_overlap:
        notes.append(f"SE differs ({se_rel_diff:.1%}) but CI overlap - methodological difference")

    return ComparisonResult(
        estimator=estimator,
        python_att=py_att,
        r_att=r_att,
        att_diff=att_diff,
        att_rel_diff=att_rel_diff,
        python_se=py_se,
        r_se=r_se,
        se_rel_diff=se_rel_diff,
        python_time=py_time,
        r_time=r_time,
        time_ratio=time_ratio,
        ci_overlap=ci_overlap,
        passed=passed,
        notes="; ".join(notes),
        python_time_std=py_time_std,
        r_time_std=r_time_std,
        n_replications=n_reps,
        scale=scale,
    )


def compare_group_time_effects(
    python_gt: List[Dict],
    r_gt: List[Dict],
    atol: float = 1e-4,
) -> Tuple[float, float, bool]:
    """
    Compare group-time effects between Python and R.

    Returns
    -------
    correlation : float
        Correlation between ATT estimates.
    max_diff : float
        Maximum absolute difference.
    all_close : bool
        Whether all estimates are within tolerance.
    """
    # Create dictionaries keyed by (group, time)
    py_dict = {(e["group"], e["time"]): e["att"] for e in python_gt}
    r_dict = {(e["group"], e["time"]): e["att"] for e in r_gt}

    # Find common keys
    common_keys = set(py_dict.keys()) & set(r_dict.keys())
    if not common_keys:
        return 0.0, float("inf"), False

    py_vals = np.array([py_dict[k] for k in sorted(common_keys)])
    r_vals = np.array([r_dict[k] for k in sorted(common_keys)])

    correlation = np.corrcoef(py_vals, r_vals)[0, 1]
    max_diff = np.max(np.abs(py_vals - r_vals))
    all_close = np.allclose(py_vals, r_vals, atol=atol, rtol=0.01)

    return correlation, max_diff, all_close


def compare_event_study(
    python_es: List[Dict],
    r_es: List[Dict],
    atol: float = 1e-4,
) -> Tuple[float, float, bool]:
    """
    Compare event study effects between Python and R.

    Returns
    -------
    correlation : float
        Correlation between ATT estimates.
    max_diff : float
        Maximum absolute difference.
    all_close : bool
        Whether all estimates are within tolerance.
    """
    # Create dictionaries keyed by event_time
    py_dict = {e["event_time"]: e["att"] for e in python_es}
    r_dict = {e["event_time"]: e["att"] for e in r_es}

    common_keys = set(py_dict.keys()) & set(r_dict.keys())
    if not common_keys:
        return 0.0, float("inf"), False

    py_vals = np.array([py_dict[k] for k in sorted(common_keys)])
    r_vals = np.array([r_dict[k] for k in sorted(common_keys)])

    correlation = np.corrcoef(py_vals, r_vals)[0, 1] if len(py_vals) > 1 else 1.0
    max_diff = np.max(np.abs(py_vals - r_vals))
    all_close = np.allclose(py_vals, r_vals, atol=atol, rtol=0.01)

    return correlation, max_diff, all_close


def load_results(path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def generate_comparison_report(
    comparisons: List[ComparisonResult],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a formatted comparison report.

    Parameters
    ----------
    comparisons : list of ComparisonResult
        Comparison results to report.
    output_path : Path, optional
        If provided, write report to this file.

    Returns
    -------
    str
        Formatted report.
    """
    lines = [
        "=" * 70,
        "BENCHMARK COMPARISON REPORT: diff-diff vs R Packages",
        "=" * 70,
        "",
    ]

    # Summary
    n_passed = sum(c.passed for c in comparisons)
    n_total = len(comparisons)
    lines.append(f"Summary: {n_passed}/{n_total} comparisons passed\n")

    # Detailed results
    for comp in comparisons:
        lines.append("-" * 70)
        lines.append(str(comp))
        lines.append("")

    # Performance summary
    lines.append("=" * 70)
    lines.append("PERFORMANCE SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Check if we have multi-replication data
    has_std = any(comp.n_replications > 1 for comp in comparisons)

    if has_std:
        lines.append(f"{'Estimator':<20} {'Scale':<8} {'Python (s)':<18} {'R (s)':<18} {'Speedup':<10}")
        lines.append("-" * 80)
        for comp in comparisons:
            factor = 1 / comp.time_ratio if comp.time_ratio > 0 else float('inf')
            if comp.n_replications > 1:
                py_str = f"{comp.python_time:.3f} ± {comp.python_time_std:.3f}"
                r_str = f"{comp.r_time:.3f} ± {comp.r_time_std:.3f}"
            else:
                py_str = f"{comp.python_time:.3f}"
                r_str = f"{comp.r_time:.3f}"
            lines.append(
                f"{comp.estimator:<20} {comp.scale:<8} {py_str:<18} {r_str:<18} "
                f"{factor:.1f}x"
            )
    else:
        lines.append(f"{'Estimator':<25} {'Python (s)':<12} {'R (s)':<12} {'Ratio':<10}")
        lines.append("-" * 60)
        for comp in comparisons:
            faster = "Python" if comp.time_ratio < 1 else "R"
            factor = 1 / comp.time_ratio if comp.time_ratio < 1 else comp.time_ratio
            lines.append(
                f"{comp.estimator:<25} {comp.python_time:<12.3f} {comp.r_time:<12.3f} "
                f"{factor:.1f}x ({faster} faster)"
            )

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


def main():
    """Run comparison on all available benchmark results."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for comparison report",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    comparisons = []

    # Compare each estimator if results exist
    estimators = [
        ("callaway", "CallawaySantAnna"),
        ("synthdid", "SyntheticDiD"),
        ("basic", "BasicDiD"),
    ]

    for key, name in estimators:
        py_path = results_dir / "accuracy" / f"python_{key}.json"
        r_path = results_dir / "accuracy" / f"r_{key}.json"

        if py_path.exists() and r_path.exists():
            py_results = load_results(py_path)
            r_results = load_results(r_path)
            comparison = compare_estimates(py_results, r_results, name)
            comparisons.append(comparison)
            print(f"Compared {name}")
        else:
            print(f"Skipping {name}: results not found")

    if comparisons:
        report = generate_comparison_report(comparisons, args.output)
        print("\n" + report)
    else:
        print("No benchmark results found to compare.")


if __name__ == "__main__":
    main()
