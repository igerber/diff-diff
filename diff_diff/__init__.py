"""
diff-diff: A library for Difference-in-Differences analysis.

This library provides sklearn-like estimators for causal inference
using the difference-in-differences methodology.
"""

# Import backend detection from dedicated module (avoids circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_bootstrap_weights,
    _rust_compute_robust_vcov,
    _rust_project_simplex,
    _rust_solve_ols,
    _rust_synthetic_weights,
)

from diff_diff.bacon import (
    BaconDecomposition,
    BaconDecompositionResults,
    Comparison2x2,
    bacon_decompose,
)
from diff_diff.diagnostics import (
    PlaceboTestResults,
    leave_one_out_test,
    permutation_test,
    placebo_group_test,
    placebo_timing_test,
    run_all_placebo_tests,
    run_placebo_test,
)
from diff_diff.linalg import (
    InferenceResult,
    LinearRegression,
)
from diff_diff.estimators import (
    DifferenceInDifferences,
    MultiPeriodDiD,
    SyntheticDiD,
    TwoWayFixedEffects,
)
from diff_diff.honest_did import (
    DeltaRM,
    DeltaSD,
    DeltaSDRM,
    HonestDiD,
    HonestDiDResults,
    SensitivityResults,
    compute_honest_did,
    sensitivity_plot,
)
from diff_diff.power import (
    PowerAnalysis,
    PowerResults,
    SimulationPowerResults,
    compute_mde,
    compute_power,
    compute_sample_size,
    simulate_power,
)
from diff_diff.pretrends import (
    PreTrendsPower,
    PreTrendsPowerCurve,
    PreTrendsPowerResults,
    compute_mdv,
    compute_pretrends_power,
)
from diff_diff.prep import (
    aggregate_to_cohorts,
    balance_panel,
    create_event_time,
    generate_did_data,
    make_post_indicator,
    make_treatment_indicator,
    rank_control_units,
    summarize_did_data,
    validate_did_data,
    wide_to_long,
)
from diff_diff.results import (
    DiDResults,
    MultiPeriodDiDResults,
    PeriodEffect,
    SyntheticDiDResults,
)
from diff_diff.staggered import (
    CallawaySantAnna,
    CallawaySantAnnaResults,
    CSBootstrapResults,
    GroupTimeEffect,
)
from diff_diff.sun_abraham import (
    SABootstrapResults,
    SunAbraham,
    SunAbrahamResults,
)
from diff_diff.triple_diff import (
    TripleDifference,
    TripleDifferenceResults,
    triple_difference,
)
from diff_diff.utils import (
    WildBootstrapResults,
    check_parallel_trends,
    check_parallel_trends_robust,
    equivalence_test_trends,
    wild_bootstrap_se,
)
from diff_diff.visualization import (
    plot_bacon,
    plot_event_study,
    plot_group_effects,
    plot_honest_event_study,
    plot_power_curve,
    plot_pretrends_power,
    plot_sensitivity,
)

__version__ = "2.0.3"
__all__ = [
    # Estimators
    "DifferenceInDifferences",
    "TwoWayFixedEffects",
    "MultiPeriodDiD",
    "SyntheticDiD",
    "CallawaySantAnna",
    "SunAbraham",
    "TripleDifference",
    # Bacon Decomposition
    "BaconDecomposition",
    "BaconDecompositionResults",
    "Comparison2x2",
    "bacon_decompose",
    "plot_bacon",
    # Results
    "DiDResults",
    "MultiPeriodDiDResults",
    "SyntheticDiDResults",
    "PeriodEffect",
    "CallawaySantAnnaResults",
    "CSBootstrapResults",
    "GroupTimeEffect",
    "SunAbrahamResults",
    "SABootstrapResults",
    "TripleDifferenceResults",
    "triple_difference",
    # Visualization
    "plot_event_study",
    "plot_group_effects",
    "plot_sensitivity",
    "plot_honest_event_study",
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
    # Honest DiD sensitivity analysis
    "HonestDiD",
    "HonestDiDResults",
    "SensitivityResults",
    "DeltaSD",
    "DeltaRM",
    "DeltaSDRM",
    "compute_honest_did",
    "sensitivity_plot",
    # Power analysis
    "PowerAnalysis",
    "PowerResults",
    "SimulationPowerResults",
    "compute_mde",
    "compute_power",
    "compute_sample_size",
    "simulate_power",
    "plot_power_curve",
    # Pre-trends power analysis
    "PreTrendsPower",
    "PreTrendsPowerResults",
    "PreTrendsPowerCurve",
    "compute_pretrends_power",
    "compute_mdv",
    "plot_pretrends_power",
    # Rust backend
    "HAS_RUST_BACKEND",
    # Linear algebra helpers
    "LinearRegression",
    "InferenceResult",
]
