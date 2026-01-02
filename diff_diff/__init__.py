"""
diff-diff: A library for Difference-in-Differences analysis.

This library provides sklearn-like estimators for causal inference
using the difference-in-differences methodology.
"""

from diff_diff.estimators import DifferenceInDifferences, MultiPeriodDiD
from diff_diff.results import DiDResults, MultiPeriodDiDResults, PeriodEffect

__version__ = "0.1.0"
__all__ = [
    "DifferenceInDifferences",
    "MultiPeriodDiD",
    "DiDResults",
    "MultiPeriodDiDResults",
    "PeriodEffect",
]
