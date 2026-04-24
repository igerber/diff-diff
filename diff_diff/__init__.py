"""
diff-diff: A library for Difference-in-Differences analysis.

This library provides sklearn-like estimators for causal inference
using the difference-in-differences methodology.

For rigorous analysis, follow the 8-step practitioner workflow based
on Baker et al. (2025). After estimation, call
``practitioner_next_steps(results)`` for context-aware guidance on
remaining diagnostic steps.

AI agents: call ``diff_diff.get_llm_guide()`` for a complete API reference.
Use ``get_llm_guide("practitioner")`` for the 8-step workflow or
``get_llm_guide("full")`` for comprehensive documentation.
"""

# Import backend detection from dedicated module (avoids circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_compute_robust_vcov,
    _rust_project_simplex,
    _rust_solve_ols,
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
from diff_diff.local_linear import (
    KERNELS,
    BandwidthResult,
    BiasCorrectedFit,
    LocalLinearFit,
    bias_corrected_local_linear,
    epanechnikov_kernel,
    kernel_moments,
    local_linear_fit,
    mse_optimal_bandwidth,
    triangular_kernel,
    uniform_kernel,
)
from diff_diff.had import (
    HeterogeneousAdoptionDiD,
    HeterogeneousAdoptionDiDEventStudyResults,
    HeterogeneousAdoptionDiDResults,
)
from diff_diff.had_pretests import (
    HADPretestReport,
    QUGTestResults,
    StuteJointResult,
    StuteTestResults,
    YatchewTestResults,
    did_had_pretest_workflow,
    joint_homogeneity_test,
    joint_pretrends_test,
    qug_test,
    stute_joint_pretest,
    stute_test,
    yatchew_hr_test,
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
    SimulationMDEResults,
    SimulationPowerResults,
    SimulationSampleSizeResults,
    SurveyPowerConfig,
    compute_mde,
    compute_power,
    compute_sample_size,
    simulate_mde,
    simulate_power,
    simulate_sample_size,
)
from diff_diff.pretrends import (
    PreTrendsPower,
    PreTrendsPowerCurve,
    PreTrendsPowerResults,
    compute_mdv,
    compute_pretrends_power,
)
from diff_diff.prep import (
    aggregate_survey,
    aggregate_to_cohorts,
    balance_panel,
    create_event_time,
    generate_continuous_did_data,
    generate_did_data,
    generate_ddd_data,
    generate_event_study_data,
    generate_factor_data,
    generate_panel_data,
    generate_reversible_did_data,
    generate_staggered_data,
    generate_staggered_ddd_data,
    generate_survey_did_data,
    make_post_indicator,
    make_treatment_indicator,
    rank_control_units,
    summarize_did_data,
    trim_weights,
    validate_did_data,
    wide_to_long,
)
from diff_diff.results import (
    DiDResults,
    MultiPeriodDiDResults,
    PeriodEffect,
    SyntheticDiDResults,
)
from diff_diff.survey import (
    DEFFDiagnostics,
    SurveyDesign,
    SurveyMetadata,
    compute_deff_diagnostics,
)
from diff_diff.staggered import (
    CallawaySantAnna,
    CallawaySantAnnaResults,
    CSBootstrapResults,
    GroupTimeEffect,
)
from diff_diff.imputation import (
    ImputationBootstrapResults,
    ImputationDiD,
    ImputationDiDResults,
    imputation_did,
)
from diff_diff.two_stage import (
    TwoStageBootstrapResults,
    TwoStageDiD,
    TwoStageDiDResults,
    two_stage_did,
)
from diff_diff.stacked_did import (
    StackedDiD,
    StackedDiDResults,
    stacked_did,
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
from diff_diff.staggered_triple_diff import (
    StaggeredTripleDifference,
)
from diff_diff.staggered_triple_diff_results import (
    StaggeredTripleDiffResults,
)
from diff_diff.continuous_did import (
    ContinuousDiD,
    ContinuousDiDResults,
    DoseResponseCurve,
)
from diff_diff.efficient_did import (
    EfficientDiD,
    EfficientDiDResults,
    EDiDBootstrapResults,
)
from diff_diff.chaisemartin_dhaultfoeuille import (
    ChaisemartinDHaultfoeuille,
    TWFEWeightsResult,
    chaisemartin_dhaultfoeuille,
    twowayfeweights,
)
from diff_diff.chaisemartin_dhaultfoeuille_results import (
    ChaisemartinDHaultfoeuilleResults,
    DCDHBootstrapResults,
)
from diff_diff.trop import (
    TROP,
    TROPResults,
    trop,
)
from diff_diff.wooldridge import WooldridgeDiD
from diff_diff.wooldridge_results import WooldridgeDiDResults
from diff_diff.utils import (
    WildBootstrapResults,
    check_parallel_trends,
    check_parallel_trends_robust,
    equivalence_test_trends,
    wild_bootstrap_se,
)
from diff_diff.visualization import (
    plot_bacon,
    plot_dose_response,
    plot_event_study,
    plot_group_effects,
    plot_group_time_heatmap,
    plot_honest_event_study,
    plot_power_curve,
    plot_pretrends_power,
    plot_sensitivity,
    plot_staircase,
    plot_synth_weights,
)
from diff_diff.practitioner import practitioner_next_steps
from diff_diff.business_report import (
    BUSINESS_REPORT_SCHEMA_VERSION,
    BusinessContext,
    BusinessReport,
)
from diff_diff.diagnostic_report import (
    DIAGNOSTIC_REPORT_SCHEMA_VERSION,
    DiagnosticReport,
    DiagnosticReportResults,
)
from diff_diff._guides_api import get_llm_guide
from diff_diff.datasets import (
    clear_cache,
    list_datasets,
    load_card_krueger,
    load_castle_doctrine,
    load_dataset,
    load_divorce_laws,
    load_mpdta,
)

# Estimator aliases — short names for convenience
DiD = DifferenceInDifferences
TWFE = TwoWayFixedEffects
EventStudy = MultiPeriodDiD
SDiD = SyntheticDiD
CS = CallawaySantAnna
CDiD = ContinuousDiD
SA = SunAbraham
BJS = ImputationDiD
Gardner = TwoStageDiD
DDD = TripleDifference
SDDD = StaggeredTripleDifference
Stacked = StackedDiD
Bacon = BaconDecomposition
EDiD = EfficientDiD
ETWFE = WooldridgeDiD
DCDH = ChaisemartinDHaultfoeuille
HAD = HeterogeneousAdoptionDiD

__version__ = "3.2.0"
__all__ = [
    # Estimators
    "DifferenceInDifferences",
    "TwoWayFixedEffects",
    "MultiPeriodDiD",
    "SyntheticDiD",
    "CallawaySantAnna",
    "ChaisemartinDHaultfoeuille",
    "ContinuousDiD",
    "SunAbraham",
    "ImputationDiD",
    "TwoStageDiD",
    "TripleDifference",
    "TROP",
    "StackedDiD",
    # Estimator aliases (short names)
    "DiD",
    "TWFE",
    "EventStudy",
    "SDiD",
    "CS",
    "CDiD",
    "DCDH",
    "SA",
    "BJS",
    "Gardner",
    "DDD",
    "SDDD",
    "Stacked",
    "Bacon",
    # Bacon Decomposition
    "BaconDecomposition",
    "BaconDecompositionResults",
    "Comparison2x2",
    "bacon_decompose",
    # Results
    "DiDResults",
    "MultiPeriodDiDResults",
    "SyntheticDiDResults",
    "PeriodEffect",
    "CallawaySantAnnaResults",
    "CSBootstrapResults",
    "GroupTimeEffect",
    "ContinuousDiDResults",
    "DoseResponseCurve",
    "SunAbrahamResults",
    "SABootstrapResults",
    "ImputationDiDResults",
    "ImputationBootstrapResults",
    "imputation_did",
    "TwoStageDiDResults",
    "TwoStageBootstrapResults",
    "two_stage_did",
    "TripleDifferenceResults",
    "triple_difference",
    "StaggeredTripleDifference",
    "StaggeredTripleDiffResults",
    "TROPResults",
    "trop",
    "StackedDiDResults",
    "stacked_did",
    # EfficientDiD
    "EfficientDiD",
    "EfficientDiDResults",
    "EDiDBootstrapResults",
    "EDiD",
    # ChaisemartinDHaultfoeuille (dCDH)
    "ChaisemartinDHaultfoeuilleResults",
    "DCDHBootstrapResults",
    "TWFEWeightsResult",
    "chaisemartin_dhaultfoeuille",
    "twowayfeweights",
    # WooldridgeDiD (ETWFE)
    "WooldridgeDiD",
    "WooldridgeDiDResults",
    "ETWFE",
    # Visualization
    "plot_bacon",
    "plot_event_study",
    "plot_group_effects",
    "plot_sensitivity",
    "plot_honest_event_study",
    "plot_power_curve",
    "plot_pretrends_power",
    "plot_synth_weights",
    "plot_staircase",
    "plot_dose_response",
    "plot_group_time_heatmap",
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
    "trim_weights",
    "validate_did_data",
    "summarize_did_data",
    "generate_did_data",
    "generate_staggered_data",
    "generate_factor_data",
    "generate_ddd_data",
    "generate_panel_data",
    "generate_event_study_data",
    "generate_staggered_ddd_data",
    "generate_survey_did_data",
    "generate_continuous_did_data",
    "generate_reversible_did_data",
    "create_event_time",
    "aggregate_survey",
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
    "SimulationMDEResults",
    "SimulationPowerResults",
    "SimulationSampleSizeResults",
    "SurveyPowerConfig",
    "compute_mde",
    "compute_power",
    "compute_sample_size",
    "simulate_mde",
    "simulate_power",
    "simulate_sample_size",
    # Pre-trends power analysis
    "PreTrendsPower",
    "PreTrendsPowerResults",
    "PreTrendsPowerCurve",
    "compute_pretrends_power",
    "compute_mdv",
    # Survey support
    "SurveyDesign",
    "SurveyMetadata",
    "DEFFDiagnostics",
    "compute_deff_diagnostics",
    # Rust backend
    "HAS_RUST_BACKEND",
    # Linear algebra helpers
    "LinearRegression",
    "InferenceResult",
    # Local-linear regression infrastructure (Phase 1a for HeterogeneousAdoptionDiD)
    "KERNELS",
    "LocalLinearFit",
    "epanechnikov_kernel",
    "kernel_moments",
    "local_linear_fit",
    "triangular_kernel",
    "uniform_kernel",
    # MSE-optimal bandwidth selector (Phase 1b for HeterogeneousAdoptionDiD)
    "BandwidthResult",
    "mse_optimal_bandwidth",
    # Bias-corrected local-linear (Phase 1c for HeterogeneousAdoptionDiD)
    "BiasCorrectedFit",
    "bias_corrected_local_linear",
    # HeterogeneousAdoptionDiD (Phase 2a single-period, Phase 2b event study)
    "HeterogeneousAdoptionDiD",
    "HeterogeneousAdoptionDiDResults",
    "HeterogeneousAdoptionDiDEventStudyResults",
    "HAD",
    # HeterogeneousAdoptionDiD pre-test diagnostics (Phase 3)
    "qug_test",
    "stute_test",
    "yatchew_hr_test",
    "did_had_pretest_workflow",
    "QUGTestResults",
    "StuteTestResults",
    "YatchewTestResults",
    "HADPretestReport",
    # HAD joint pre-tests (Phase 3 follow-up) — multi-period event-study
    # workflow dispatch via did_had_pretest_workflow(aggregate="event_study")
    "stute_joint_pretest",
    "joint_pretrends_test",
    "joint_homogeneity_test",
    "StuteJointResult",
    # Datasets
    "load_card_krueger",
    "load_castle_doctrine",
    "load_divorce_laws",
    "load_mpdta",
    "load_dataset",
    "list_datasets",
    "clear_cache",
    # Practitioner guidance
    "practitioner_next_steps",
    "BusinessReport",
    "BusinessContext",
    "BUSINESS_REPORT_SCHEMA_VERSION",
    "DiagnosticReport",
    "DiagnosticReportResults",
    "DIAGNOSTIC_REPORT_SCHEMA_VERSION",
    # LLM guide accessor
    "get_llm_guide",
]
