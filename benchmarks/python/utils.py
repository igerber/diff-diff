"""
Common utilities for Python benchmarks.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_timing_stats(
    timings: List[float],
    exclude_first: bool = True,
) -> Dict[str, Any]:
    """
    Compute timing statistics from a list of timing measurements.

    Parameters
    ----------
    timings : list of float
        List of timing measurements in seconds.
    exclude_first : bool
        Whether to exclude the first measurement from statistics
        (may be affected by warm-up effects).

    Returns
    -------
    dict
        Dictionary containing:
        - raw_timings: All raw timing values
        - n_reps: Number of replications (after exclusion)
        - first_run_excluded: Whether first run was excluded
        - first_run_seconds: The first timing value
        - stats: Dictionary with mean, std, median, min, max
    """
    if not timings:
        return {
            "raw_timings": [],
            "n_reps": 0,
            "first_run_excluded": False,
            "first_run_seconds": None,
            "stats": {},
        }

    first_run = timings[0]
    analysis_timings = timings[1:] if exclude_first and len(timings) > 1 else timings

    if len(analysis_timings) == 0:
        analysis_timings = timings  # Fall back if only one measurement

    return {
        "raw_timings": timings,
        "n_reps": len(analysis_timings),
        "first_run_excluded": exclude_first and len(timings) > 1,
        "first_run_seconds": first_run,
        "stats": {
            "mean": float(np.mean(analysis_timings)),
            "std": float(np.std(analysis_timings, ddof=1)) if len(analysis_timings) > 1 else 0.0,
            "median": float(np.median(analysis_timings)),
            "min": float(np.min(analysis_timings)),
            "max": float(np.max(analysis_timings)),
        },
    }


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    estimator: str
    att: float
    se: float
    timing_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "estimator": self.estimator,
            "att": self.att,
            "se": self.se,
            "timing": {"total_seconds": self.timing_seconds},
            "metadata": self.metadata,
            **self.extra,
        }

    def to_json(self, path: Path) -> None:
        """Write results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_serializer)


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


def generate_staggered_data(
    n_units: int = 200,
    n_periods: int = 8,
    n_cohorts: int = 3,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic staggered adoption data for benchmarking.

    Parameters
    ----------
    n_units : int
        Number of units.
    n_periods : int
        Number of time periods.
    n_cohorts : int
        Number of treatment cohorts (excluding never-treated).
    treatment_effect : float
        Base treatment effect.
    never_treated_frac : float
        Fraction of never-treated units.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: unit, time, outcome, first_treat, treated.
    """
    rng = np.random.default_rng(seed)

    # Assign units to cohorts
    n_never_treated = int(n_units * never_treated_frac)
    n_treated = n_units - n_never_treated
    units_per_cohort = n_treated // n_cohorts

    # Generate first treatment times (starting from period 3)
    first_treat_times = []
    for i in range(n_units):
        if i < n_never_treated:
            first_treat_times.append(0)  # Never treated (coded as 0)
        else:
            cohort = (i - n_never_treated) // units_per_cohort
            cohort = min(cohort, n_cohorts - 1)
            first_treat_times.append(3 + cohort)

    # Generate panel data
    data = []
    for unit in range(n_units):
        unit_fe = rng.normal(0, 2)
        first_treat = first_treat_times[unit]

        for t in range(1, n_periods + 1):
            time_fe = t * 0.5  # Linear time trend

            # Treatment indicator
            if first_treat == 0:
                treated = 0
                post = 0
            else:
                treated = 1
                post = 1 if t >= first_treat else 0

            # Dynamic treatment effect
            if post == 1:
                relative_time = t - first_treat
                effect = treatment_effect * (1 + 0.1 * relative_time)
            else:
                effect = 0

            outcome = unit_fe + time_fe + effect + rng.normal(0, 0.5)

            data.append(
                {
                    "unit": unit,
                    "time": t,
                    "outcome": outcome,
                    "first_treat": first_treat,
                    "treated": treated,
                    "post": post,
                }
            )

    return pd.DataFrame(data)


def generate_basic_did_data(
    n_units: int = 100,
    n_periods: int = 4,
    treatment_effect: float = 5.0,
    treatment_period: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate basic 2x2 DiD data for benchmarking.

    Parameters
    ----------
    n_units : int
        Number of units.
    n_periods : int
        Number of time periods.
    treatment_effect : float
        True treatment effect.
    treatment_period : int
        First post-treatment period.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: unit, time, outcome, treated, post.
    """
    rng = np.random.default_rng(seed)

    n_treated = n_units // 2
    data = []

    for unit in range(n_units):
        unit_fe = rng.normal(0, 2)
        treated = 1 if unit < n_treated else 0

        for t in range(1, n_periods + 1):
            time_fe = t * 0.5
            post = 1 if t >= treatment_period else 0

            effect = treatment_effect if (treated and post) else 0
            outcome = unit_fe + time_fe + effect + rng.normal(0, 1)

            data.append(
                {
                    "unit": unit,
                    "time": t,
                    "outcome": outcome,
                    "treated": treated,
                    "post": post,
                }
            )

    return pd.DataFrame(data)


def generate_sdid_data(
    n_control: int = 40,
    n_treated: int = 10,
    n_pre: int = 15,
    n_post: int = 5,
    treatment_effect: float = 4.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate data suitable for Synthetic DiD benchmarking.

    Parameters
    ----------
    n_control : int
        Number of control units.
    n_treated : int
        Number of treated units.
    n_pre : int
        Number of pre-treatment periods.
    n_post : int
        Number of post-treatment periods.
    treatment_effect : float
        True treatment effect.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: unit, time, outcome, treated, post.
    """
    rng = np.random.default_rng(seed)

    n_units = n_control + n_treated
    n_periods = n_pre + n_post
    data = []

    for unit in range(n_units):
        unit_fe = rng.normal(0, 2)
        treated = 1 if unit >= n_control else 0

        for t in range(1, n_periods + 1):
            time_fe = np.sin(t * 0.3) + t * 0.1  # Non-linear time trend
            post = 1 if t > n_pre else 0

            effect = treatment_effect if (treated and post) else 0
            outcome = unit_fe + time_fe + effect + rng.normal(0, 0.5)

            data.append(
                {
                    "unit": unit,
                    "time": t,
                    "outcome": outcome,
                    "treated": treated,
                    "post": post,
                }
            )

    return pd.DataFrame(data)


def load_benchmark_data(path: Path) -> pd.DataFrame:
    """Load benchmark data from CSV."""
    return pd.read_csv(path)


def save_benchmark_data(data: pd.DataFrame, path: Path) -> None:
    """Save benchmark data to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
