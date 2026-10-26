"""PR 4c-i / 4c-ii (4.0 cut): the TWFE event-study surface is built from the
core's payload - and nothing moved.

One layer of oracle remains: the COMMITTED oracle
``tests/fixtures/v4_twfe_event_study_oracles.json``, captured on the
PRE-REFACTOR tree (``provenance.captured_at_head`` = 68e383bd; the two
clustered hc2_bm lanes at d45d5316 under ``provenance.lane_additions``) by
``tests/_capture_v4_merge_mpd_oracles.py`` for every lane of
``tests/helpers/twfe_es_oracle_lanes.py`` - every ``EventStudyResults`` field
of the TWFE surface. The live-twin and payload-identity layers that PR 4c-i
carried (``MultiPeriodDiD``, ``_from_mpd``, ``_mpd_results_from_payload``)
went with the class in PR 4c-ii; the committed capture is what pins the
surface now, so a re-capture on the current tree is a re-baseline, not an
oracle.

Tolerances follow ``tests/test_aggregate_contract.py``: analytic arithmetic at
rtol 1e-12 / atol 1e-13; confidence-interval bounds at rtol 1e-9 (scipy's
``t.ppf`` moved between 1.13 and 1.14+). Committed-oracle ``t_stat`` columns
are checked against the bound those input tolerances imply for the ratio
``att / se`` (``_assert_t_stat_within_propagated_bound``), because a flat atol
is tighter than the inputs' own contract whenever ``se < 1``.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import warnings
from typing import Any, Dict

import numpy as np
import pytest
from event_study_average import post_period_average
from event_study_surface_builders import calendar_surface
from twfe_es_oracle_lanes import LANES, PANELS, Lane

import diff_diff.twfe
from diff_diff import TwoWayFixedEffects
from diff_diff.results_base import (
    EventStudyResults,
    _calendar_surface_from_arrays,
    _EventStudyCorePayload,
    _from_event_study_payload,
    _resolve_scalar_df_survey,
    _scalar_df_survey_from_metadata,
)
from diff_diff.survey import SurveyMetadata
from diff_diff.utils import safe_inference

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "v4_twfe_event_study_oracles.json")

ANALYTIC_TOL = dict(rtol=1e-12, atol=1e-13)
T_STAT_RTOL_FACTOR = 2.0
CI_RTOL = 1e-9
CI_FIELDS = {"conf_int_lower", "conf_int_upper"}
LABEL_FIELDS = {"event_time", "vcov_index", "is_reference"}
TUPLE_FIELDS = {"post_periods", "reference_event_times"}


def _load_oracle() -> Dict[str, Any]:
    if not os.path.exists(FIXTURE):
        pytest.skip("v4_twfe_event_study_oracles.json not present (see tests/fixtures/README.md)")
    with open(FIXTURE) as fh:
        return json.load(fh)


def _decode_array(stored: Dict[str, Any]) -> np.ndarray:
    def _none_to_nan(obj: Any) -> Any:
        if isinstance(obj, list):
            return [_none_to_nan(v) for v in obj]
        return np.nan if obj is None else obj

    return np.asarray(_none_to_nan(stored["values"]), dtype=np.dtype(stored["dtype"]))


def _fit_lane(lane: Lane):
    df = PANELS[lane.panel]()
    sd = lane.survey()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        twfe = TwoWayFixedEffects(**lane.twfe_ctor).fit(
            df, event_study=True, survey_design=sd, **lane.twfe_fit
        )
    assert isinstance(twfe, EventStudyResults)
    return df, twfe


def _tol_for(field: str) -> Dict[str, float]:
    if field in CI_FIELDS:
        return dict(rtol=CI_RTOL, atol=ANALYTIC_TOL["atol"])
    return dict(ANALYTIC_TOL)


def _assert_t_stat_within_propagated_bound(
    live_t: np.ndarray, stored_t: np.ndarray, stored_att: np.ndarray, stored_se: np.ndarray
) -> None:
    """``t_stat = att / se`` checked against the bound its inputs' tolerances imply.

    With ``|d att| <= atol + rtol |att|`` and ``|d se| <= atol + rtol |se|`` (the
    ``ANALYTIC_TOL`` contract that ``att`` and ``se`` pass individually), first-order
    propagation gives ``|d t| <= atol (1 + |t|) / se + 2 rtol |t|``. A flat
    ``atol=1e-13`` on ``t`` is tighter than that whenever ``se < 1``: on the
    rank-deficient ``pooled_nan_post_period`` lane the near-zero period-0 effect
    (``att`` -3.8e-3, ``se`` 0.29, ``t`` -0.013) drifts 1.6e-13 (Linux OpenBLAS) and
    2.4e-13 (Windows) against the macOS Accelerate capture while ``att`` and ``se``
    both pass, exactly the ``atol / se`` amplification this bound allows for.
    """
    mask = ~np.isnan(stored_t)
    np.testing.assert_array_equal(np.isnan(live_t), ~mask, err_msg="t_stat")
    atol, rtol = ANALYTIC_TOL["atol"], ANALYTIC_TOL["rtol"]
    t, se = stored_t[mask], stored_se[mask]
    bound = atol * (1.0 + np.abs(t)) / se + T_STAT_RTOL_FACTOR * rtol * np.abs(t)
    diff = np.abs(live_t[mask] - t)
    assert np.all(diff <= bound), (
        "t_stat outside the propagated att/se bound",
        dict(diff=diff, bound=bound, live=live_t[mask], stored=t, att=stored_att[mask], se=se),
    )


def _assert_surface_matches_record(live: EventStudyResults, stored: Dict[str, Any]) -> None:
    """Every ``dataclasses.fields(EventStudyResults)`` value against its record."""
    for f in dataclasses.fields(live):
        name = f.name
        value = getattr(live, name)
        rec = stored[name]
        if isinstance(rec, dict) and "dtype" in rec:
            assert value is not None, name
            expected = _decode_array(rec)
            assert np.asarray(value).dtype.str == rec["dtype"], (name, np.asarray(value).dtype)
            assert np.asarray(value).shape == expected.shape, name
            if name == "t_stat":
                _assert_t_stat_within_propagated_bound(
                    np.asarray(value, dtype=float),
                    expected,
                    _decode_array(stored["att"]),
                    _decode_array(stored["se"]),
                )
            elif expected.dtype.kind == "f":
                np.testing.assert_array_equal(np.isnan(value), np.isnan(expected), err_msg=name)
                mask = ~np.isnan(expected)
                np.testing.assert_allclose(
                    np.asarray(value)[mask], expected[mask], err_msg=name, **_tol_for(name)
                )
            else:
                np.testing.assert_array_equal(np.asarray(value), expected, err_msg=name)
        elif name in TUPLE_FIELDS:
            assert (tuple(rec) if rec is not None else None) == value, name
        elif isinstance(rec, float) and isinstance(value, float):
            np.testing.assert_allclose(value, rec, err_msg=name, **_tol_for(name))
        else:
            assert rec == value, (name, rec, value)


# --------------------------------------------------------------------------- #
# Committed oracle (captured before the refactor)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(LANES))
def test_lane_echo_matches_committed_oracle(name):
    """Lane drift is reported as lane drift, not as a numeric regression."""
    oracle = _load_oracle()
    assert oracle["lanes"][name]["echo"] == json.loads(json.dumps(LANES[name].echo()))


@pytest.mark.parametrize("name", sorted(LANES))
def test_surface_matches_committed_oracle(name):
    """The refactored TWFE event-study surface reproduces the pre-refactor capture."""
    oracle = _load_oracle()
    _, twfe = _fit_lane(LANES[name])
    _assert_surface_matches_record(twfe, oracle["lanes"][name]["twfe"])


# --------------------------------------------------------------------------- #
# The core's payload and the shared surface builder
# --------------------------------------------------------------------------- #


def test_core_returns_payload():
    df = PANELS["main"]()
    payload = TwoWayFixedEffects()._fit_event_study_core(
        df,
        "outcome",
        "treated",
        "period",
        [4, 5, 6, 7],
        None,
        None,
        None,
        3,
        None,
        None,
        "analytical",
        include_treatment_main=True,
        cluster_override=None,
        estimator_name="TwoWayFixedEffects",
    )
    assert isinstance(payload, _EventStudyCorePayload)
    assert payload.periods == sorted(payload.periods)
    assert 3 not in payload.periods
    for arr in (payload.att, payload.se, payload.t_stat, payload.p_value, payload.ci_lower):
        assert arr.shape == (len(payload.periods),)
    assert set(payload.df_by_period) == set(payload.periods)
    surface = _from_event_study_payload(
        payload, source="TwoWayFixedEffects", estimation_spec="pooled"
    )
    assert isinstance(surface, EventStudyResults)
    assert surface.source == "TwoWayFixedEffects"
    assert surface.estimation_spec == "pooled"
    assert surface.event_time.tolist() == sorted(payload.periods + [3])
    assert surface.is_reference.sum() == 1
    src = inspect.getsource(diff_diff.twfe)
    assert "_from_mpd" not in src
    assert "MultiPeriodDiDResults" not in src
    assert "period_effects" not in src


def test_calendar_surface_builder_rejects_unsorted_periods():
    """The sorted-label contract of the shared builder: ``periods`` arrive
    index-aligned with the per-period arrays, so an unsorted list must raise
    instead of being silently re-sorted (which would pair labels with the wrong
    values). Formerly guarded through the removed dict-keyed adapter."""
    with pytest.raises(ValueError, match="sorted order"):
        _calendar_surface_from_arrays(
            periods=[5, 6, 1, 2],
            ref=3,
            att=[0.5, 0.6, 0.1, 0.2],
            se=[0.5, 0.6, 0.1, 0.2],
            t_stat=[1.0, 1.0, 1.0, 1.0],
            p_value=[0.5, 0.5, 0.5, 0.5],
            ci_lo=[-0.5, -0.4, -0.9, -0.8],
            ci_hi=[1.5, 1.6, 1.1, 1.2],
            full_vcov=None,
            interaction_indices=None,
            df_map=None,
            post_periods=[5, 6],
            alpha=0.05,
            source="TwoWayFixedEffects",
            df_survey=None,
        )
    # The same arrays in sorted order build cleanly.
    s = _calendar_surface_from_arrays(
        periods=[1, 2, 5, 6],
        ref=3,
        att=[0.1, 0.2, 0.5, 0.6],
        se=[0.1, 0.2, 0.5, 0.6],
        t_stat=[1.0, 1.0, 1.0, 1.0],
        p_value=[0.5, 0.5, 0.5, 0.5],
        ci_lo=[-0.9, -0.8, -0.5, -0.4],
        ci_hi=[1.1, 1.2, 1.5, 1.6],
        full_vcov=None,
        interaction_indices=None,
        df_map=None,
        post_periods=[5, 6],
        alpha=0.05,
        source="TwoWayFixedEffects",
        df_survey=None,
    )
    np.testing.assert_array_equal(s.event_time, [1, 2, 3, 5, 6])
    ref_row = int(np.where(s.event_time == 3)[0][0])
    # the synthesized reference row: att 0.0 (the normalization), NaN inference
    assert s.is_reference[ref_row] and s.att[ref_row] == 0.0 and np.isnan(s.se[ref_row])
    np.testing.assert_array_equal(s.att[~s.is_reference], [0.1, 0.2, 0.5, 0.6])


def test_df_survey_metadata_split_equals_resolver():
    def _sm(**kw):
        return SurveyMetadata(
            weight_type="pweight", effective_n=10.0, design_effect=1.0, sum_weights=10.0, **kw
        )

    class _Res:
        def __init__(self, sm):
            self.survey_metadata = sm

    tsl = _sm(df_survey=56)
    rep_defined = _sm(df_survey=14, replicate_method="JK1")
    rep_undefined = _sm(df_survey=None, replicate_method="JK1")
    for sm, expected in ((tsl, 56.0), (rep_defined, 14.0), (rep_undefined, 0.0), (None, None)):
        assert _scalar_df_survey_from_metadata(sm) == expected
        assert _resolve_scalar_df_survey(_Res(sm)) == expected
    assert isinstance(_scalar_df_survey_from_metadata(tsl), float)


# --------------------------------------------------------------------------- #
# The post-period average helper (the former post-average block, now test-side)
# --------------------------------------------------------------------------- #


def _surface(att, se, vcov, post, ref):
    """Hand-built calendar surface for the helper's NaN lanes: per-row inference
    comes from ``safe_inference`` (all-NaN on the reference row and on any
    undefined-SE row), never from an inline ratio."""
    att = np.asarray(att, dtype=float).copy()
    se = np.asarray(se, dtype=float).copy()
    labels = np.arange(len(att))
    is_ref = labels == ref
    att[is_ref] = np.nan
    se[is_ref] = np.nan
    t = np.full(len(att), np.nan)
    p = np.full(len(att), np.nan)
    lo = np.full(len(att), np.nan)
    hi = np.full(len(att), np.nan)
    for i in np.where(~is_ref)[0]:
        t[i], p[i], (lo[i], hi[i]) = safe_inference(float(att[i]), float(se[i]), alpha=0.05)
    return EventStudyResults(
        event_time=labels,
        att=att,
        se=se,
        t_stat=t,
        p_value=p,
        conf_int_lower=lo,
        conf_int_upper=hi,
        is_reference=is_ref,
        n=np.full(len(att), np.nan),
        time_scale="calendar",
        source="TwoWayFixedEffects",
        post_periods=tuple(post),
        vcov=np.asarray(vcov, dtype=float),
        vcov_index=labels[~is_ref],
    )


def test_post_period_average_nan_contract():
    # (d) healthy: hand arithmetic
    s = _surface([0.1, 0.0, 1.0, 2.0], [0.1, 0.1, 0.2, 0.3], np.diag([0.01, 0.04, 0.09]), [2, 3], 1)
    att_bar, se_bar = post_period_average(s)
    assert att_bar == 1.5
    assert se_bar == pytest.approx(np.sqrt((0.04 + 0.09) / 4))
    # (a) one post att NaN -> (nan, nan)
    s = _surface(
        [0.1, 0.0, np.nan, 2.0], [0.1, 0.1, 0.2, 0.3], np.diag([0.01, 0.04, 0.09]), [2, 3], 1
    )
    att_bar, se_bar = post_period_average(s)
    assert np.isnan(att_bar) and np.isnan(se_bar)
    # (b) finite att but a NaN entry in the post vcov block -> finite att_bar, NaN se_bar
    v = np.diag([0.01, 0.04, 0.09]).astype(float)
    v[2, 2] = np.nan
    s = _surface([0.1, 0.0, 1.0, 2.0], [0.1, 0.1, 0.2, 0.3], v, [2, 3], 1)
    att_bar, se_bar = post_period_average(s)
    assert att_bar == 1.5 and np.isnan(se_bar)
    # (c) negative summed variance -> NaN se_bar, no RuntimeWarning
    v = np.array([[0.01, 0.0, 0.0], [0.0, 0.04, -0.5], [0.0, -0.5, 0.09]])
    s = _surface([0.1, 0.0, 1.0, 2.0], [0.1, 0.1, 0.2, 0.3], v, [2, 3], 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        att_bar, se_bar = post_period_average(s)
    assert att_bar == 1.5 and np.isnan(se_bar)
    # (e) a REAL fit: the rank-deficient NaN post period on the TWFE pooled lane
    lane = LANES["pooled_nan_post_period"]
    _, twfe = _fit_lane(lane)
    row3 = int(np.where(twfe.event_time == 3)[0][0])
    assert np.isnan(twfe.att[row3])
    att_bar, se_bar = post_period_average(twfe)
    assert np.isnan(att_bar)
    assert np.isnan(se_bar)


def test_calendar_surface_builder_uses_safe_inference():
    """The hand-built-surface helper derives 2-tuple rows through safe_inference:
    alpha drives the interval level and undefined SEs fail closed to all-NaN."""
    from scipy import stats

    s90 = calendar_surface(
        {1: (1.0, 0.5), 3: (2.0, 0.25)}, reference=2, post_periods=[3], alpha=0.10
    )
    s95 = calendar_surface({1: (1.0, 0.5), 3: (2.0, 0.25)}, reference=2, post_periods=[3])
    row = int(np.where(s90.event_time == 3)[0][0])
    z90, z95 = stats.norm.ppf(0.95), stats.norm.ppf(0.975)
    np.testing.assert_allclose(s90.conf_int_upper[row] - s90.att[row], z90 * 0.25, rtol=1e-12)
    np.testing.assert_allclose(s95.conf_int_upper[row] - s95.att[row], z95 * 0.25, rtol=1e-12)
    assert s90.conf_int_upper[row] < s95.conf_int_upper[row]
    for bad_se in (0.0, -1.0, np.nan, np.inf):
        s = calendar_surface({1: (1.0, bad_se), 3: (2.0, 0.25)}, reference=2, post_periods=[3])
        r = int(np.where(s.event_time == 1)[0][0])
        assert np.isnan(s.t_stat[r]) and np.isnan(s.p_value[r])
        assert np.isnan(s.conf_int_lower[r]) and np.isnan(s.conf_int_upper[r])
    ref = int(np.where(s95.event_time == 2)[0][0])
    assert s95.is_reference[ref] and np.isnan(s95.se[ref]) and np.isnan(s95.p_value[ref])
