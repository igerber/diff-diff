"""Owned Duration results, serialization and v4 aggregation contracts."""

import json
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    BaseResults,
    Diagnostic,
    EventStudyResults,
    compute_honest_did,
    compute_pretrends_power,
    plot_event_study,
)
from diff_diff.aggregation import AGGREGATION_SCHEMA
from diff_diff.duration_did_results import (
    _CONTRAST_COLUMNS,
    _EFFECT_COLUMNS,
    _FAILURE_COLUMNS,
    _SURVIVAL_COLUMNS,
)
from diff_diff.results_base import EVENT_STUDY_SCHEMA
from tests.test_duration_did import duration_panel, fit_duration


@pytest.fixture
def result():
    return fit_duration(n_bootstrap=13, seed=15, alpha=0.025)


def test_schemas_and_reference(result):
    assert list(result.effects) == _EFFECT_COLUMNS
    assert list(result.survival_curve) == _SURVIVAL_COLUMNS
    assert list(result.bootstrap_failures) == _FAILURE_COLUMNS
    assert list(result.pretrend_results.contrasts) == _CONTRAST_COLUMNS
    assert tuple(result.to_dataframe()) == EVENT_STUDY_SCHEMA
    assert tuple(result.to_dataframe("simple")) == AGGREGATION_SCHEMA
    es = result.aggregate("event_study")
    assert isinstance(es, EventStudyResults)
    assert es.source == "DurationDiDResults"
    np.testing.assert_array_equal(es.event_time, [-1, 0, 1])
    assert es.att[0] == 0 and es.is_reference.tolist() == [True, False, False]
    assert np.isnan(es.se[0]) and np.isnan(es.n[0])
    np.testing.assert_array_equal(es.vcov_index, [0, 1])
    np.testing.assert_array_equal(es.vcov, result.vcov)
    assert es.n_kind == "units"
    assert np.isnan(es.df).all()
    simple = result.aggregate("simple")
    assert simple.att[0] == result.att
    assert simple.conf_int_lower[0] == result.conf_int[0]
    assert simple.weight[0] == 1
    assert isinstance(result, BaseResults) and not isinstance(result, Diagnostic)
    assert isinstance(result.pretrend_test(), Diagnostic)
    assert not any(f.name.startswith("_AGGREGATE") for f in fields(result))


def test_copy_ownership(result):
    clone = replace(result)
    clone.effects.loc[0, "att"] = 100
    clone.bootstrap_effects[0, 0] = 100
    clone.time_weights[1] = 100
    clone.pretrend_results.contrasts.loc[0, "contrast"] = 100
    assert result.effects.att.iloc[0] != 100
    assert result.bootstrap_effects[0, 0] != 100
    assert result.time_weights[1] != 100
    assert result.pretrend_results.contrasts.contrast.iloc[0] != 100
    diagnostic_copy = result.pretrend_test()
    diagnostic_copy.contrasts.loc[0, "contrast"] = 100
    survival_copy = result.to_dataframe("survival")
    survival_copy.loc[0, "treated_survival"] = 100
    result.aggregate("event_study").vcov[0, 0] = 100
    assert result.vcov[0, 0] != 100
    assert result.survival_curve.treated_survival.iloc[0] == 0.9


def test_serialization_confidence_and_metadata(result):
    payload = result.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["att"] == result.att
    assert payload["conf_int_lower"] == result.conf_int[0]
    assert payload["pretrend_results"]["alpha"] == 0.025
    assert payload["inference_method"] == "bootstrap"
    assert payload["raw_att"] == pytest.approx(result.att)
    for r in (result, result.pretrend_test()):
        assert "97.5%" in r.summary()
        with pytest.raises(ValueError, match="never recomputes"):
            r.summary(alpha=0.05)
    assert "normalized observation interval" in result.summary()
    ph = fit_duration(method="proportional_hazards", n_bootstrap=13, seed=15)
    assert "dimensionless hazard ratio" in ph.summary()


@pytest.mark.parametrize(
    "clock", [pd.date_range("2020", periods=6), pd.timedelta_range("0 days", periods=6)]
)
def test_temporal_json_labels(clock):
    from diff_diff import DurationDiD

    data = duration_panel()
    data["date"] = data.date.map(dict(enumerate(clock)))
    r = DurationDiD(n_bootstrap=2, seed=1).fit(
        data, "absorbed", "group", "id", "date", post_periods=list(clock[4:])
    )
    decoded = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    assert decoded["periods"][0] == clock[0].isoformat()
    assert decoded["time_weights"][0]["period"] == clock[1].isoformat()
    assert isinstance(decoded["time_step"], str)


@pytest.mark.parametrize("level", ["group", "calendar", "total", "nonsense"])
def test_rejected_aggregation(result, level):
    with pytest.raises(ValueError):
        result.aggregate(level)
    with pytest.raises(ValueError):
        result.to_dataframe(level)


def test_no_custom_aggregation_or_generic_sensitivity(result):
    with pytest.raises(ValueError):
        result.aggregate("simple", weights="equal")
    with pytest.raises(ValueError):
        result.aggregate("event_study", balance_e=1)
    for consumer in (compute_honest_did, compute_pretrends_power):
        for value in (result, result.aggregate("event_study")):
            with pytest.raises((TypeError, ValueError, NotImplementedError)):
                consumer(value)


def test_plotting(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = plot_event_study(result.aggregate("event_study"), alpha=result.alpha, show=False)
    assert ax is not None
    plt.close("all")
