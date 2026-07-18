"""diff-diff: Difference-in-Differences causal inference with sklearn-like API.
Recommended starting call for LLM agents:
``diff_diff.agent_workflow(df, unit=..., time=..., treatment=..., outcome=...)``
prints a copy-pasteable workflow with your column names wired in.

The orchestrator names the full sequence:

    1. Describe the panel:    diff_diff.profile_panel(df, ...)
    2. Choose an estimator:   diff_diff.get_llm_guide("autonomous")
                              (estimator-support matrix + reasoning)
    3. Fit:                   <Estimator>(...).fit(df, ...)
    4. Validate:              diff_diff.practitioner_next_steps(result)
    5. Report:                diff_diff.BusinessReport(result)

For a comprehensive API reference call ``diff_diff.get_llm_guide("full")``.
For the Baker et al. (2025) 8-step practitioner recipe call
``diff_diff.get_llm_guide("practitioner")``.

This library provides sklearn-like estimators for causal inference using
the difference-in-differences methodology.
"""

# Import backend detection from dedicated module (avoids circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_bootstrap_weights,
    _rust_compute_robust_vcov,
    _rust_project_simplex,
    _rust_solve_ols,
)
from diff_diff._guides_api import get_llm_guide
from diff_diff.agent_workflow import agent_workflow
from diff_diff.bacon import (
    BaconDecomposition,
    BaconDecompositionResults,
    Comparison2x2,
    bacon_decompose,
)
from diff_diff.business_report import (
    BUSINESS_REPORT_SCHEMA_VERSION,
    BusinessContext,
    BusinessReport,
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
from diff_diff.changes_in_changes import (
    ChangesInChanges,
    QDiD,
)
from diff_diff.changes_in_changes_results import (
    ChangesInChangesResults,
    QDiDResults,
)
from diff_diff.continuous_did import (
    ContinuousDiD,
    ContinuousDiDResults,
    DoseResponseCurve,
)
from diff_diff.datasets import (
    clear_cache,
    list_datasets,
    load_card_krueger,
    load_castle_doctrine,
    load_dataset,
    load_divorce_laws,
    load_mpdta,
    load_prop99,
    load_walmart,
)
from diff_diff.diagnostic_report import (
    DIAGNOSTIC_REPORT_SCHEMA_VERSION,
    DiagnosticReport,
    DiagnosticReportResults,
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
from diff_diff.efficient_did import (
    EDiDBootstrapResults,
    EfficientDiD,
    EfficientDiDResults,
)
from diff_diff.estimators import (
    DifferenceInDifferences,
    MultiPeriodDiD,
    SyntheticDiD,
    TwoWayFixedEffects,
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
from diff_diff.imputation import (
    ImputationBootstrapResults,
    ImputationDiD,
    ImputationDiDResults,
    imputation_did,
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
from diff_diff.lpdid import LPDiD
from diff_diff.lpdid_results import LPDiDResults
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
from diff_diff.practitioner import practitioner_next_steps
from diff_diff.prep import (
    aggregate_survey,
    aggregate_to_cohorts,
    balance_panel,
    create_event_time,
    generate_continuous_did_data,
    generate_ddd_data,
    generate_ddd_panel_data,
    generate_did_data,
    generate_event_study_data,
    generate_factor_data,
    generate_panel_data,
    generate_reversible_did_data,
    generate_staggered_data,
    generate_staggered_ddd_data,
    generate_survey_did_data,
    generate_synthetic_control_data,
    make_post_indicator,
    make_treatment_indicator,
    rank_control_units,
    summarize_did_data,
    trim_weights,
    validate_did_data,
    wide_to_long,
)
from diff_diff.pretrends import (
    PreTrendsPower,
    PreTrendsPowerCurve,
    PreTrendsPowerResults,
    compute_mdv,
    compute_pretrends_power,
)
from diff_diff.profile import (
    Alert,
    OutcomeShape,
    PanelProfile,
    TreatmentDoseShape,
    profile_panel,
)
from diff_diff.rdd import (
    RegressionDiscontinuity,
    RegressionDiscontinuityResults,
)
from diff_diff.rdplot import (
    RDPlot,
    RDPlotResult,
)
from diff_diff.results import (
    DiDResults,
    MultiPeriodDiDResults,
    PeriodEffect,
    SpilloverDiDResults,  # re-export
    SyntheticDiDResults,
)
from diff_diff.spillover import (
    SpilloverDiD,
)
from diff_diff.stacked_did import (
    StackedDiD,
    StackedDiDResults,
    stacked_did,
)
from diff_diff.staggered import (
    CallawaySantAnna,
    CallawaySantAnnaResults,
    CSBootstrapResults,
    GroupTimeEffect,
)
from diff_diff.staggered_triple_diff import (
    StaggeredTripleDifference,
)
from diff_diff.staggered_triple_diff_results import (
    StaggeredTripleDiffResults,
)
from diff_diff.sun_abraham import (
    SABootstrapResults,
    SunAbraham,
    SunAbrahamResults,
)
from diff_diff.survey import (
    DEFFDiagnostics,
    SurveyDesign,
    SurveyMetadata,
    compute_deff_diagnostics,
    make_pweight_design,
)
from diff_diff.synthetic_control import (
    SyntheticControl,
    synthetic_control,
)
from diff_diff.synthetic_control_results import SyntheticControlResults
from diff_diff.triple_diff import (
    TripleDifference,
    TripleDifferenceResults,
    triple_difference,
)
from diff_diff.trop import (
    TROP,
    TROPResults,
    trop,
)
from diff_diff.two_stage import (
    TwoStageBootstrapResults,
    TwoStageDiD,
    TwoStageDiDResults,
    two_stage_did,
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
from diff_diff.wooldridge import WooldridgeDiD
from diff_diff.wooldridge_results import WooldridgeDiDResults

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
CiC = ChangesInChanges
RDD = RegressionDiscontinuity

__version__ = "3.8.0"
__all__ = [
    # Estimators
    "DifferenceInDifferences",
    "TwoWayFixedEffects",
    "MultiPeriodDiD",
    "SyntheticDiD",
    "CallawaySantAnna",
    "ChaisemartinDHaultfoeuille",
    "ChangesInChanges",
    "QDiD",
    "ContinuousDiD",
    "SunAbraham",
    "ImputationDiD",
    "TwoStageDiD",
    "SpilloverDiD",
    "TripleDifference",
    "TROP",
    "SyntheticControl",
    "StackedDiD",
    # Estimator aliases (short names)
    "DiD",
    "TWFE",
    "EventStudy",
    "SDiD",
    "CS",
    "CDiD",
    "DCDH",
    "CiC",
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
    "ChangesInChangesResults",
    "QDiDResults",
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
    "SpilloverDiDResults",
    "TripleDifferenceResults",
    "triple_difference",
    "StaggeredTripleDifference",
    "StaggeredTripleDiffResults",
    "TROPResults",
    "trop",
    "SyntheticControlResults",
    "synthetic_control",
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
    # LPDiD (Local Projections DiD)
    "LPDiD",
    "LPDiDResults",
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
    "generate_ddd_panel_data",
    "generate_panel_data",
    "generate_event_study_data",
    "generate_staggered_ddd_data",
    "generate_survey_did_data",
    "generate_continuous_did_data",
    "generate_reversible_did_data",
    "generate_synthetic_control_data",
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
    "make_pweight_design",
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
    # RegressionDiscontinuity (sharp RD, rdrobust parity)
    "RegressionDiscontinuity",
    "RegressionDiscontinuityResults",
    "RDD",
    # RDPlot (data-driven RD plots, rdplot parity)
    "RDPlot",
    "RDPlotResult",
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
    "load_prop99",
    "load_walmart",
    "load_dataset",
    "list_datasets",
    "clear_cache",
    # Practitioner guidance
    "agent_workflow",
    "practitioner_next_steps",
    "BusinessReport",
    "BusinessContext",
    "BUSINESS_REPORT_SCHEMA_VERSION",
    "DiagnosticReport",
    "DiagnosticReportResults",
    "DIAGNOSTIC_REPORT_SCHEMA_VERSION",
    # Panel profiling (agent-facing pre-fit describe utility)
    "profile_panel",
    "PanelProfile",
    "Alert",
    "OutcomeShape",
    "TreatmentDoseShape",
    # LLM guide accessor
    "get_llm_guide",
]

# Agent-facing entrypoints surface first in dir(diff_diff). LLM agents
# follow a `dir -> help -> docstring -> use` discovery loop; surfacing
# these names first measurably improves discoverability vs the default
# alphabetic ordering. Internal — read by tests/test_agent_discoverability.py.
_AGENT_FACING_ORDER = (
    "agent_workflow",
    "profile_panel",
    "get_llm_guide",
    "practitioner_next_steps",
    "BusinessReport",
    "DiagnosticReport",
)


class _OrderedName(str):
    """str subclass that sorts by _AGENT_FACING_ORDER priority.

    Python's built-in dir() always sorts the result of __dir__()
    alphabetically (CPython Objects/object.c::_dir_object unconditionally
    calls PyList_Sort), so returning a list in our preferred order is
    not enough. But PyList_Sort uses __lt__ for comparisons, so a str
    subclass with a custom __lt__ can subvert the alphabetic default
    while remaining a fully usable str for every other operation.

    ALL names returned by __dir__() must be _OrderedName, not just the
    priority head: when Python compares an _OrderedName against a plain
    str, the reflected-method protocol prefers str's inherited __gt__
    (because _OrderedName is a subclass of str), which sorts purely
    alphabetically and breaks the ordering. With every element wrapped,
    all comparisons go through this __lt__: priority head sorts to
    front, tail (default priority 1<<30) falls through to alphabetic
    via str.__lt__.
    """

    _ORDER = {n: i for i, n in enumerate(_AGENT_FACING_ORDER)}

    def __lt__(self, other):
        sp = self._ORDER.get(str(self), 1 << 30)
        op = self._ORDER.get(str(other), 1 << 30)
        if sp != op:
            return sp < op
        return str.__lt__(self, other)


def __dir__():
    """Surface agent-facing entrypoints first; remainder alphabetic.

    Returns the full module namespace (matching default `dir(module)`
    membership — keeps `__doc__`, `__name__`, etc. accessible via
    `inspect.getmembers`) with priority names re-ordered to the head
    via `_OrderedName`'s custom `__lt__`.

    `__all__` order does not affect `dir(module)`. CPython sorts the
    result of `__dir__()` alphabetically, so we return `_OrderedName`
    instances (str subclass with custom `__lt__`) for every name; the
    custom comparison routes head names to the top and falls back to
    alphabetic for everyone else. See `_OrderedName` docstring for
    why ALL names must be wrapped (mixing plain `str` with the
    subclass triggers Python's reflected-method comparison protocol
    and breaks the ordering).

    `from diff_diff import *` semantics are unaffected (driven by
    `__all__`, not by `dir()`).
    """
    return [_OrderedName(n) for n in globals()]
