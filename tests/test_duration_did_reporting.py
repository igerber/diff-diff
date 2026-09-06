"""Native hazard reporting without incompatible generic diagnostics."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from diff_diff import (
    BusinessReport,
    DiagnosticReport,
    DifferenceInDifferences,
    DurationDiD,
    practitioner_next_steps,
)
from diff_diff._reporting_helpers import describe_target_parameter
from tests.test_duration_did import duration_panel, fit_duration


@pytest.mark.parametrize("method", ["common_dynamics", "proportional_hazards"])
def test_native_payload_and_assumptions(method):
    r = fit_duration(method=method, n_bootstrap=19, alpha=0.025, seed=7)
    schema = DiagnosticReport(r).to_dict()
    native = schema["estimator_native_diagnostics"]
    assert native["status"] == "ran"
    assert native["estimator"] == "DurationDiD"
    assert native["pretrend_test"] == r.pretrend_test().to_dict()
    assert native["inference_status"] == r.inference_status
    br = BusinessReport(r)
    body = br.to_dict()
    assert body["robustness"]["estimator_native"]["pretrend_test"] == native["pretrend_test"]
    for text in (br.summary(), br.full_report()):
        assert "hazard pretest" in text.lower()
        assert "97.5%" in text
        assert "does not establish identification" in text
        assert "large-cluster" not in text
    assert method in body["assumption"]["description"]
    assert "parallel trends" not in body["assumption"]["description"].lower()
    assert not any(x["role"] == "sensitivity" for x in body["references"])
    guidance = practitioner_next_steps(r, verbose=False)
    encoded = json.dumps(guidance)
    assert "hazard" in encoded and "pretrend_test()" in encoded
    assert "Which PT variant" not in encoded and "compute_honest_did" not in encoded
    assert describe_target_parameter(r)["aggregation"] == "uniform_post_periods"


@pytest.mark.parametrize("method", ["common_dynamics", "proportional_hazards"])
@pytest.mark.parametrize("state", ["rejecting", "nonrejecting", "unavailable"])
def test_diagnostic_report_renders_stored_hazard_views(method, state):
    if state == "rejecting":
        data = duration_panel(control=(100, 80, 60, 40, 20, 10), treated=(100, 50, 40, 25, 10, 2))
    elif state == "unavailable":
        data = duration_panel(control=(3, 2, 1, 1, 1, 1), treated=(3, 2, 1, 1, 1, 1), n=3, scale=1)
    else:
        data = duration_panel()
    r = fit_duration(data, method=method, n_bootstrap=39, seed=19, alpha=0.025)
    pretest = r.pretrend_test()
    stored = r.to_dict()
    dr = DiagnosticReport(r, alpha=0.1)
    returned = dr.run_all()
    expected_decision = {"rejecting": True, "nonrejecting": False, "unavailable": None}[state]
    assert pretest.reject is expected_decision
    expected_phrase = {
        "rejecting": "hazard pretest rejects",
        "nonrejecting": "hazard pretest does not reject",
        "unavailable": "Hazard pretest unavailable",
    }[state]
    # Test the live report, returned container, and the full report's native section.
    native_section = (
        dr.full_report()
        .split("## Estimator-native diagnostics", 1)[1]
        .split("## Placebo battery", 1)[0]
    )
    for text in (dr.summary(), returned.summary(), dr.full_report(), native_section):
        assert expected_phrase in text
        assert "97.5% simultaneous confidence level" in text
        if expected_decision is None:
            assert "No rejection decision is available" in text
            assert "does not reject" not in text
            assert all(reason in text for reason in pretest.reasons)
        else:
            assert f"p={pretest.p_value:.4g}" in text
            assert "Non-rejection does not establish identification or adequate power" in text
    pd.testing.assert_frame_equal(dr.to_dataframe(), returned.to_dataframe())
    for table in (dr.to_dataframe(), returned.to_dataframe()):
        row = table.set_index("check").loc["estimator_native"]
        assert row["status"] == "ran"  # Extraction status is distinct from test availability.
        if expected_decision is None:
            assert pd.isna(row["headline"])
            assert all(reason in row["reason"] for reason in pretest.reasons)
        else:
            assert row["headline"] == pretest.p_value
            assert pd.isna(row["reason"])
    assert r.to_dict() == stored  # Rendering never refits or alters stored inference.
    assert returned.schema["estimator_native_diagnostics"]["pretrend_test"] == pretest.to_dict()


def test_diagnostic_report_invalid_counterfactual_guidance():
    data = duration_panel(control=(100, 70, 40, 20, 20, 20), treated=(100, 99, 98, 97, 96, 95))
    r = fit_duration(data, n_bootstrap=15, seed=2)
    assert r.estimation_status == "invalid_counterfactual"
    dr = DiagnosticReport(r)
    for text in (dr.summary(), dr.run_all().summary(), dr.full_report()):
        assert "counterfactual survival path is invalid" in text
        assert "canonical causal effects are unavailable" in text
        assert all(term in text for term in ("survival_curve", "raw extrapolations", "fit_periods"))
        assert "rank deficiency" not in text
        assert "survey-design collapse" not in text


@pytest.mark.parametrize("key", ["parallel_trends", "sensitivity", "pretrends_power", "bacon"])
@pytest.mark.parametrize("automatic", [True, False])
def test_reject_overrides(key, automatic):
    r = fit_duration(n_bootstrap=2, seed=1)
    with pytest.raises(ValueError, match="precomputed"):
        DiagnosticReport(r, precomputed={key: {}})
    with pytest.raises(ValueError, match="precomputed"):
        BusinessReport(r, precomputed={key: {}}, auto_diagnostics=automatic)
    with pytest.raises(ValueError, match="honest_did_results") as error:
        BusinessReport(r, honest_did_results={}, auto_diagnostics=automatic)
    assert all(word in str(error.value) for word in ("pretrend_test()", "method", "fit_periods"))
    assert "Synthetic" not in str(error.value)


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
def test_supplied_foreign_diagnostics_rejected(detached, automatic):
    data = duration_panel()
    data["post"] = (data["date"] >= 4).astype(int)
    foreign = DifferenceInDifferences().fit(data, "absorbed", "group", "post")
    dr = DiagnosticReport(foreign)
    supplied = dr.run_all() if detached else dr
    r = fit_duration(n_bootstrap=2, seed=1)
    with pytest.raises(ValueError, match="requires a DurationDiDResults report"):
        BusinessReport(r, diagnostics=supplied, auto_diagnostics=automatic).to_dict()


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_supplied_same_fit_diagnostics_accepted(detached, available):
    data = (
        None
        if available
        else duration_panel(control=(3, 2, 1, 1, 1, 1), treated=(3, 2, 1, 1, 1, 1), n=3, scale=1)
    )
    r = fit_duration(data, n_bootstrap=13, seed=2, alpha=0.025)
    dr = DiagnosticReport(r, alpha=0.1)
    if detached:
        supplied = dr.run_all()
        # A detached container may carry a strict-JSON round-tripped schema.
        supplied.schema.clear()
        supplied.schema.update(json.loads(json.dumps(DiagnosticReport(r).to_dict())))
    else:
        supplied = dr
    body = BusinessReport(r, diagnostics=supplied, auto_diagnostics=False).to_dict()
    assert body["robustness"]["estimator_native"]["pretrend_test"] == r.pretrend_test().to_dict()


@pytest.mark.parametrize("detached", [False, True])
def test_supplied_different_duration_fit_rejected(detached):
    r = fit_duration(n_bootstrap=13, seed=2)
    dr = DiagnosticReport(fit_duration(n_bootstrap=13, seed=3))
    supplied = dr.run_all() if detached else dr
    with pytest.raises(ValueError, match="must match this fit's stored native hazard diagnostic"):
        BusinessReport(r, diagnostics=supplied).to_dict()


def test_validated_native_payload_is_owned_by_business_report():
    r = fit_duration(n_bootstrap=13, seed=2)
    supplied = DiagnosticReport(r).run_all()
    br = BusinessReport(r, diagnostics=supplied)
    original = deepcopy(br.to_dict()["robustness"]["estimator_native"])
    supplied.schema["estimator_native_diagnostics"]["pretrend_test"]["reject"] = True
    assert br.to_dict()["robustness"]["estimator_native"] == original


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("section", ["parallel_trends", "sensitivity", "pretrends_power", "bacon"])
def test_supplied_generic_sections_rejected_even_with_duration_provenance(detached, section):
    r = fit_duration(n_bootstrap=13, seed=2)
    dr = DiagnosticReport(r)
    report = dr.run_all()
    report.schema[section] = {"status": "ran", "p_value": 0.99}
    with pytest.raises(ValueError, match=f"computed generic {section}"):
        BusinessReport(r, diagnostics=report if detached else dr).to_dict()


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("change", ["missing", "estimator", "method", "reject", "alpha", "counts"])
def test_supplied_native_payload_mismatch_rejected(detached, change):
    r = fit_duration(n_bootstrap=13, seed=2)
    dr = DiagnosticReport(r)
    report = dr.run_all()
    native = report.schema["estimator_native_diagnostics"]
    if change == "missing":
        report.schema.pop("estimator_native_diagnostics")
    elif change in {"estimator", "method"}:
        native[change] = "foreign"
    elif change == "counts":
        native["n_bootstrap_valid"] = 0
    else:
        native["pretrend_test"][change] = True if change == "reject" else 0.1
    with pytest.raises(ValueError, match="must match this fit's stored native hazard diagnostic"):
        BusinessReport(r, diagnostics=report if detached else dr).to_dict()


@pytest.mark.parametrize("automatic", [False, True])
def test_long_horizon_summary_bounds_support_prose_and_preserves_details(automatic):
    lengths = []
    for n_dates in (6, 60):
        counts = (3, 2) + (1,) * (n_dates - 2)
        data = duration_panel(control=counts, treated=counts, n=3, scale=1)
        r = DurationDiD(n_bootstrap=15, seed=2).fit(
            data, "absorbed", "group", "id", "date", post_periods=list(range(4, n_dates))
        )
        br = BusinessReport(r, auto_diagnostics=automatic)
        schema = deepcopy(br.to_dict())
        summary, full = br.summary(), br.full_report()
        assert f"{len(r.support_warnings)} survivor/exit support warnings" in summary
        assert "Pooled individual bootstrap" in summary
        assert "Hazard pretest unavailable" in summary
        assert summary.count("Caveat:") <= 2
        assert "See full_report()" in summary
        assert all(warning in full for warning in r.support_warnings)
        assert [
            c["message"] for c in schema["caveats"] if c["topic"] == "duration_support"
        ] == r.support_warnings
        assert br.to_dict() == schema  # Compact rendering never truncates the structured data.
        lengths.append(len(summary))
    assert lengths[1] < lengths[0] + 150


@pytest.mark.parametrize("automatic", [True, False])
def test_few_treated_and_failed_families_without_auto_diagnostics(automatic):
    data = duration_panel(control=(3, 2, 1, 1, 1, 1), treated=(3, 2, 1, 1, 1, 1), n=3, scale=1)
    r = fit_duration(data, n_bootstrap=15, seed=2)
    br = BusinessReport(r, auto_diagnostics=automatic)
    for text in (br.summary(), br.full_report()):
        assert "3 treated individuals" in text
        assert "Pooled individual bootstrap" in text
        assert "Hazard pretest unavailable" in text
        assert "large-cluster asymptotics" not in text
        assert "Consider SyntheticDiD" not in text
        assert "exact-permutation" not in text


def test_fit_level_confidence_preserved():
    r = fit_duration(n_bootstrap=13, alpha=0.025, seed=8)
    h = BusinessReport(r, alpha=0.1).to_dict()["headline"]
    assert h["ci_lower"] == r.conf_int[0]
    assert h["ci_level"] == 97.5
    assert not h["alpha_was_honored"]


def test_invalid_curve_visible_without_auto_diagnostics():
    data = duration_panel(control=(100, 70, 40, 20, 20, 20), treated=(100, 99, 98, 97, 96, 95))
    r = fit_duration(data, n_bootstrap=10, seed=1)
    report = BusinessReport(r, auto_diagnostics=False)
    assert "Invalid counterfactual" in report.summary()
    assert "Invalid counterfactual" in report.full_report()


def test_rejected_hazard_pretest_is_rendered_without_identification_claim():
    data = duration_panel(control=(100, 80, 60, 40, 20, 10), treated=(100, 50, 40, 25, 10, 2))
    r = fit_duration(data, n_bootstrap=39, seed=19)
    assert r.pretrend_results.status == "available"
    assert r.pretrend_results.reject is True
    for text in (BusinessReport(r).summary(), BusinessReport(r).full_report()):
        assert "hazard pretest rejects" in text
        assert "does not establish identification" in text
