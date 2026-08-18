"""Tests for JSON serialization of LWDiDResults.to_dict().

Regression tests for the shawcharles review finding that ``to_dict()``
leaked numpy scalar types and arrays into nested dicts, so
``json.dumps(result.to_dict())`` raised TypeError.
"""

import json

import numpy as np
import pandas as pd
import pytest

from diff_diff import LWDiD, generate_staggered_data
from diff_diff.lwdid_results import _json_native_key, _to_json_native


def _make_common_timing_panel(n_treated=20, n_control=30, n_pre=4, n_post=3, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treated + n_control):
        is_treated = i < n_treated
        unit_fe = rng.normal(0, 1)
        for t in range(1, n_pre + n_post + 1):
            post = t > n_pre
            treat = 1 if (is_treated and post) else 0
            y = unit_fe + 0.3 * t + rng.normal(0, 0.5) + 2.0 * treat
            rows.append({"unit": i, "time": t, "y": y, "treat": treat})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def staggered_data():
    return generate_staggered_data(n_units=120, n_periods=8, seed=3)


class TestToDictJsonSerializable:
    """json.dumps(result.to_dict()) must succeed for every result flavor."""

    def test_common_timing_roundtrip(self):
        data = _make_common_timing_panel()
        result = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1").fit(
            data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        payload = result.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        assert roundtrip["att"] == pytest.approx(result.att)
        assert roundtrip["se"] == pytest.approx(result.se)
        assert roundtrip["n_obs"] == result.n_obs

    def test_staggered_roundtrip(self, staggered_data):
        result = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1").fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
            first_treat="first_treat",
        )
        payload = result.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        assert roundtrip["att"] == pytest.approx(result.att)
        # Nested cohort dicts must contain only native types
        for key, info in roundtrip["cohort_effects"].items():
            assert isinstance(key, str)
            assert info["att"] == pytest.approx(result.cohort_effects[int(key)]["att"])
        assert set(roundtrip["cohort_time_effects"]) == {
            f"{g},{t}" for (g, t) in result.cohort_time_effects
        }

    def test_event_study_roundtrip(self, staggered_data):
        result = LWDiD(rolling="demean", estimation_method="reg", n_bootstrap=99, seed=5).fit(
            staggered_data,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
            first_treat="first_treat",
        )
        payload = result.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        assert "event_study_effects" in roundtrip
        for key, info in roundtrip["event_study_effects"].items():
            expected = result.event_study_effects[int(key)]
            assert info["effect"] == pytest.approx(expected["effect"])
            assert isinstance(info["conf_int"], list)
        assert roundtrip["reference_periods"] == list(result.reference_periods)


def _relabel_staggered_datetime(data):
    """Relabel an integer staggered panel with quarterly Timestamps."""
    date_map = {
        t: pd.Timestamp("2000-01-01") + pd.DateOffset(months=3 * (int(t) - 1))
        for t in sorted(data["period"].unique())
    }
    panel = data.copy()
    panel["date"] = panel["period"].map(date_map)
    panel["adopt"] = panel["first_treat"].map(lambda g: date_map[g] if g > 0 else pd.NaT)
    return panel


class TestDatetimeLabelsJsonSerializable:
    """Datetime/Period cohort and time labels must serialize to JSON strings.

    Regression tests: after ``_relabel_staggered_results`` restores datetime
    labels, nested ``info["cohort"]``/``info["time"]`` entries were
    pd.Timestamp/pd.Period objects and ``json.dumps(result.to_dict())``
    raised TypeError.
    """

    def test_datetime_staggered_roundtrip(self, staggered_data):
        panel = _relabel_staggered_datetime(staggered_data)
        result = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1").fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="date",
            treatment="treated",
            first_treat="adopt",
        )
        payload = result.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        assert roundtrip["att"] == pytest.approx(result.att)
        # Nested cohort/time labels must be ISO-8601 strings
        for key, info in roundtrip["cohort_effects"].items():
            assert isinstance(key, str)
            assert isinstance(info["cohort"], str)
            assert pd.Timestamp(info["cohort"]) in result.cohort_effects
        for info in roundtrip["cohort_time_effects"].values():
            assert isinstance(info["cohort"], str)
            assert isinstance(info["time"], str)
            pd.Timestamp(info["time"])  # parses back without error

    def test_period_staggered_roundtrip(self, staggered_data):
        panel = _relabel_staggered_datetime(staggered_data)
        panel["date"] = panel["date"].dt.to_period("Q")
        panel["adopt"] = pd.PeriodIndex(panel["adopt"], freq="Q")
        result = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1").fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="date",
            treatment="treated",
            first_treat="adopt",
        )
        payload = result.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        assert roundtrip["att"] == pytest.approx(result.att)
        # Period labels keep their frequency semantics, e.g. "2000Q1"
        for info in roundtrip["cohort_effects"].values():
            assert isinstance(info["cohort"], str)
            assert pd.Period(info["cohort"], freq="Q") in result.cohort_effects


class TestNaTSerializationContract:
    """NaT values must map to None so the payload stays json.dumps-able.

    Direct unit coverage for the NaT branches of the private helpers:
    the branch is unreachable through ``to_dict()`` in the current design
    (never-treated cohorts are dropped before relabeling), so the contract
    is pinned here explicitly.
    """

    def test_nat_maps_to_none(self):
        assert _to_json_native(pd.NaT) is None
        assert _to_json_native(np.datetime64("NaT")) is None
        assert _json_native_key(pd.NaT) is None
        # A nested dict containing NaT values must be json.dumps-able
        payload = {
            "cohorts": {
                pd.Timestamp("2000-01-01"): {"adopt": pd.NaT},
                "never": [pd.NaT, np.datetime64("NaT")],
            }
        }
        roundtrip = json.loads(json.dumps(_to_json_native(payload)))
        assert roundtrip["cohorts"]["2000-01-01T00:00:00"]["adopt"] is None
        assert roundtrip["cohorts"]["never"] == [None, None]
