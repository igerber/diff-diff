"""Diagnostic-marker roster + consumer-propagation contract (ledger M-091).

This file is the ``test_ref`` for ledger row M-091 (spec section 3.5). It
enforces three things:

1. Every class-backed diagnostic RESULT container subclasses
   :class:`diff_diff.Diagnostic`, exposes ``summary()`` / ``to_dataframe()``,
   and is exempt from the estimator quintet by type.
2. Estimator results are NOT marked; ``TWFEWeightsResult`` (a docs-family
   member, narrowed out of the type contract) is neither.
3. The three consumers route by the marker: BusinessReport and
   DiagnosticReport reject marked non-Bacon primaries by type,
   ``practitioner_next_steps`` routes marked diagnostics through
   diagnostic-specific handling, and Bacon retains its existing read-out.
"""

import numpy as np
import pandas as pd
import pytest
from results_foundation import make_constructed_diagnostics

import diff_diff
from diff_diff import BaconDecomposition, Diagnostic
from diff_diff._reporting_helpers import describe_target_parameter
from diff_diff.business_report import BusinessReport
from diff_diff.diagnostic_report import DiagnosticReport, DiagnosticReportResults
from diff_diff.practitioner import practitioner_next_steps

# The class-backed diagnostic roster (spec section 3.5). Bacon is handled
# separately: it is produced by a real fit below (its container is only
# meaningful when populated by the decomposition).
DIAGNOSTIC_ROSTER = [
    "BaconDecompositionResults",
    "RDPlotResult",
    "HonestDiDResults",
    "SensitivityResults",
    "PreTrendsPowerResults",
    "PreTrendsPowerCurve",
    "PowerResults",
    "SimulationPowerResults",
    "SimulationMDEResults",
    "SimulationSampleSizeResults",
    "PlaceboTestResults",
    "QUGTestResults",
    "StuteTestResults",
    "YatchewTestResults",
    "StuteJointResult",
    "HADPretestReport",
    "DiagnosticReportResults",
]

# Representative ESTIMATOR results: marked with BaseResults, never Diagnostic.
ESTIMATOR_SAMPLE = [
    "DiDResults",
    "CallawaySantAnnaResults",
    "LPDiDResults",
    "HeterogeneousAdoptionDiDResults",
    "HeterogeneousAdoptionDiDEventStudyResults",
    "WooldridgeDiDResults",
]


def _staggered_panel(seed=7):
    """Small staggered panel that BaconDecomposition accepts."""
    rng = np.random.RandomState(seed)
    n_units, n_periods = 12, 6
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    first_treat = np.repeat(np.array([0, 0, 0, 0, 3, 3, 3, 3, 4, 4, 4, 4]), n_periods)
    post = (times >= first_treat) & (first_treat > 0)
    unit_fe = np.repeat(rng.randn(n_units) * 2, n_periods)
    time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
    outcome = unit_fe + time_fe + 1.5 * post + rng.randn(n_units * n_periods) * 0.5
    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcome,
            "first_treat": first_treat.astype(int),
            "treated": post.astype(int),
        }
    )


@pytest.fixture(scope="module")
def bacon_result():
    data = _staggered_panel()
    return BaconDecomposition().fit(
        data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
    )


@pytest.fixture(scope="module")
def constructed():
    return make_constructed_diagnostics()


def _roster_instance(name, constructed, bacon_result):
    if name == "BaconDecompositionResults":
        return bacon_result
    return constructed[name]


# ---------------------------------------------------------------------------
# 1. Roster contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", DIAGNOSTIC_ROSTER)
def test_roster_member_is_marked_diagnostic(name, constructed, bacon_result):
    obj = _roster_instance(name, constructed, bacon_result)
    assert isinstance(obj, Diagnostic), f"{name} must subclass Diagnostic"


@pytest.mark.parametrize("name", DIAGNOSTIC_ROSTER)
def test_roster_member_exposes_serialization_pair(name, constructed, bacon_result):
    obj = _roster_instance(name, constructed, bacon_result)
    summary = obj.summary()
    assert isinstance(summary, str) and summary, f"{name}.summary() must be a str"
    frame = obj.to_dataframe()
    assert isinstance(frame, pd.DataFrame), f"{name}.to_dataframe() must be a DataFrame"


@pytest.mark.parametrize("name", DIAGNOSTIC_ROSTER)
def test_roster_member_has_no_quintet(name, constructed, bacon_result):
    # Diagnostics are exempt from the estimator quintet BY TYPE. A diagnostic
    # exposing all five canonical inference names would blur the boundary the
    # marker exists to enforce (PlaceboTestResults carries se/t_stat/p_value
    # but its estimate is `placebo_effect`, not `att` - so no full quintet).
    obj = _roster_instance(name, constructed, bacon_result)
    quintet = ("att", "se", "t_stat", "p_value", "conf_int")
    assert not all(
        hasattr(obj, q) for q in quintet
    ), f"{name} exposes the full estimator quintet; diagnostics must not."


def test_roster_matches_marked_classes_in_namespace():
    # Guard against a roster member silently losing its marker: every name in
    # DIAGNOSTIC_ROSTER resolves to a Diagnostic subclass.
    import diff_diff.diagnostic_report as drm

    for name in DIAGNOSTIC_ROSTER:
        cls = getattr(diff_diff, name, None) or getattr(drm, name)
        assert issubclass(cls, Diagnostic), name


# ---------------------------------------------------------------------------
# 2. Negative roster: estimators are not diagnostics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ESTIMATOR_SAMPLE)
def test_estimator_results_not_marked(name):
    cls = getattr(diff_diff, name)
    assert issubclass(cls, diff_diff.BaseResults), f"{name} must be BaseResults"
    assert not issubclass(cls, Diagnostic), f"{name} must NOT be Diagnostic"


def test_twfe_weights_result_is_neither():
    # Narrowed out of the type contract (docs-family only): a plain __slots__
    # class, not a marked container.
    cls = diff_diff.TWFEWeightsResult
    assert not issubclass(cls, Diagnostic)
    assert not issubclass(cls, diff_diff.BaseResults)


def test_frozen_diagnostic_constructs(constructed):
    # DiagnosticReportResults is the only frozen roster member; pin that
    # frozen-dataclass + non-dataclass marker inheritance is safe.
    drr = constructed["DiagnosticReportResults"]
    assert isinstance(drr, DiagnosticReportResults)
    assert isinstance(drr, Diagnostic)
    with pytest.raises((AttributeError, TypeError)):
        drr.schema = {}  # frozen


# ---------------------------------------------------------------------------
# 3. Consumer propagation
# ---------------------------------------------------------------------------
def test_business_report_rejects_bacon_by_type(bacon_result):
    with pytest.raises(TypeError, match="diagnostic"):
        BusinessReport(bacon_result)


def test_business_report_rejects_non_bacon_marked(constructed):
    with pytest.raises(TypeError, match="diagnostic"):
        BusinessReport(constructed["HonestDiDResults"])


def test_diagnostic_report_rejects_non_bacon_marked(constructed):
    with pytest.raises(TypeError, match="diagnostic"):
        DiagnosticReport(constructed["PreTrendsPowerCurve"])


def test_diagnostic_report_retains_bacon_readout(bacon_result):
    # Bacon is NOT rejected: it keeps its dedicated read-out.
    report = DiagnosticReport(bacon_result)
    schema = report.to_dict()
    assert "bacon" in schema
    assert schema["bacon"].get("status") is not None


def test_practitioner_routes_marked_through_diagnostic_handler(constructed):
    out = practitioner_next_steps(constructed["PreTrendsPowerCurve"], verbose=False)
    assert out["estimator"].endswith("(diagnostic result)")
    labels = [s["label"] for s in out["next_steps"]]
    # Estimator framing (Steps 1-2) is skipped for a diagnostic input.
    assert "Define target parameter" not in labels
    assert "estimation" not in out["completed"]


def test_practitioner_bacon_keeps_name_keyed_handler(bacon_result):
    # Bacon stays on _handle_bacon with its estimator-selection framing.
    out = practitioner_next_steps(bacon_result, verbose=False)
    labels = [s["label"] for s in out["next_steps"]]
    assert any("heterogeneity-robust estimator" in lbl for lbl in labels)


def test_describe_target_parameter_marks_diagnostic_kind(constructed):
    # Defensive-depth branch for direct callers.
    block = describe_target_parameter(constructed["PreTrendsPowerResults"])
    assert block["aggregation"] == "diagnostic"
    assert block["headline_attribute"] == ""


def test_describe_target_parameter_bacon_still_twfe(bacon_result):
    # The named Bacon branch wins over the generic diagnostic branch.
    block = describe_target_parameter(bacon_result)
    assert block["aggregation"] == "twfe"
    assert block["headline_attribute"] == "twfe_estimate"
