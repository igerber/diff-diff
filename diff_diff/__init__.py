"""
diff-diff: A library for Difference-in-Differences analysis.

This library provides sklearn-like estimators for causal inference
using the difference-in-differences methodology.
"""

from diff_diff.estimators import DifferenceInDifferences, MultiPeriodDiD, SyntheticDiD
from diff_diff.results import DiDResults, MultiPeriodDiDResults, PeriodEffect, SyntheticDiDResults
from diff_diff.prep import (
    make_treatment_indicator,
    make_post_indicator,
    wide_to_long,
    balance_panel,
    validate_did_data,
    summarize_did_data,
    generate_did_data,
    create_event_time,
    aggregate_to_cohorts,
)

__version__ = "0.3.0"
__all__ = [
    # Estimators
    "DifferenceInDifferences",
    "MultiPeriodDiD",
    "SyntheticDiD",
    # Results
    "DiDResults",
    "MultiPeriodDiDResults",
    "SyntheticDiDResults",
    "PeriodEffect",
    # Data preparation utilities
    "make_treatment_indicator",
    "make_post_indicator",
    "wide_to_long",
    "balance_panel",
    "validate_did_data",
    "summarize_did_data",
    "generate_did_data",
    "create_event_time",
    "aggregate_to_cohorts",
]
