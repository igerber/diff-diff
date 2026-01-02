"""
diff-diff: A library for Difference-in-Differences analysis.

This library provides sklearn-like estimators for causal inference
using the difference-in-differences methodology.
"""

from diff_diff.estimators import DifferenceInDifferences, MultiPeriodDiD, SyntheticDiD
from diff_diff.results import DiDResults, MultiPeriodDiDResults, PeriodEffect, SyntheticDiDResults

__version__ = "0.3.0"
__all__ = [
    "DifferenceInDifferences",
    "MultiPeriodDiD",
    "SyntheticDiD",
    "DiDResults",
    "MultiPeriodDiDResults",
    "SyntheticDiDResults",
    "PeriodEffect",
]
