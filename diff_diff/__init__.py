"""
diff-diff: A library for Difference-in-Differences analysis.

This library provides sklearn-like estimators for causal inference
using the difference-in-differences methodology.
"""

from diff_diff.estimators import (
    DifferenceInDifferences,
    TwoWayFixedEffects,
    MultiPeriodDiD,
    SyntheticDiD,
)
from diff_diff.staggered import (
    CallawaySantAnna,
    CallawaySantAnnaResults,
    GroupTimeEffect,
)
from diff_diff.results import (
    DiDResults,
    MultiPeriodDiDResults,
    PeriodEffect,
    SyntheticDiDResults,
)
from diff_diff.visualization import (
    plot_event_study,
    plot_group_effects,
)
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
    rank_control_units,
)
from diff_diff.utils import (
    check_parallel_trends,
    check_parallel_trends_robust,
    equivalence_test_trends,
    WildBootstrapResults,
    wild_bootstrap_se,
)
from diff_diff.diagnostics import (
    PlaceboTestResults,
    run_placebo_test,
    placebo_timing_test,
    placebo_group_test,
    permutation_test,
    leave_one_out_test,
    run_all_placebo_tests,
)

__version__ = "0.5.0"
__all__ = [
    # Estimators
    "DifferenceInDifferences",
    "TwoWayFixedEffects",
    "MultiPeriodDiD",
    "SyntheticDiD",
    "CallawaySantAnna",
    # Results
    "DiDResults",
    "MultiPeriodDiDResults",
    "SyntheticDiDResults",
    "PeriodEffect",
    "CallawaySantAnnaResults",
    "GroupTimeEffect",
    # Visualization
    "plot_event_study",
    "plot_group_effects",
    # Parallel trends testing
    "check_parallel_trends",
    "check_parallel_trends_robust",
    "equivalence_test_trends",
    # Wild cluster bootstrap
    "WildBootstrapResults",
    "wild_bootstrap_se",
    # Placebo tests / diagnostics
    "PlaceboTestResults",
    "run_placebo_test",
    "placebo_timing_test",
    "placebo_group_test",
    "permutation_test",
    "leave_one_out_test",
    "run_all_placebo_tests",
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
    "rank_control_units",
]
