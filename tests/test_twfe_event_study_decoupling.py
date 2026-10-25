"""PR 4c-i (4.0 cut): the TWFE event-study surface is built from the core's
payload, not from a MultiPeriodDiD container - and nothing moved.

Three layers of oracle:

* the COMMITTED oracle ``tests/fixtures/v4_twfe_event_study_oracles.json``,
  captured on the unmodified tree (``provenance.captured_at_head``) by
  ``tests/_capture_v4_merge_mpd_oracles.py`` for every lane of
  ``tests/helpers/twfe_es_oracle_lanes.py`` - the TWFE surface AND its
  MultiPeriodDiD twin, every ``EventStudyResults`` field;
* the LIVE MultiPeriodDiD twin (``_from_mpd(MultiPeriodDiD(...).fit(...))``),
  which exists until PR 4c-ii deletes the class;
* the payload itself: ``_from_event_study_payload`` and
  ``_from_mpd(_mpd_results_from_payload(...))`` must be bit-identical on the
  same payload (same inputs, same shared builder).

Tolerances follow ``tests/test_aggregate_contract.py``: analytic arithmetic at
rtol 1e-12 / atol 1e-13; confidence-interval bounds at rtol 1e-9 (scipy's
``t.ppf`` moved between 1.13 and 1.14+). Committed-oracle ``t_stat`` columns
are checked against the bound those input tolerances imply for the ratio
``att / se`` (``_assert_t_stat_within_propagated_bound``), because a flat atol
is tighter than the inputs' own contract whenever ``se < 1``; the live
TWFE-vs-MPD within-lane comparison keeps the base analytic tolerance.
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
from twfe_es_oracle_lanes import FINITE_POOLED_LANES, LANES, PANELS, WITHIN_LANES, Lane

import diff_diff.twfe
from diff_diff import MultiPeriodDiD, TwoWayFixedEffects
from diff_diff.estimators import _mpd_results_from_payload
from diff_diff.results import MultiPeriodDiDResults, PeriodEffect
from diff_diff.results_base import (
    EventStudyResults,
    _EventStudyCorePayload,
    _from_event_study_payload,
    _from_mpd,
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
        mpd_native = MultiPeriodDiD(**lane.mpd_ctor).fit(df, survey_design=sd, **lane.mpd_fit)
    assert isinstance(twfe, EventStudyResults)
    return df, twfe, mpd_native


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


def _assert_surfaces_equal(a: EventStudyResults, b: EventStudyResults, *, exact: bool, skip=()):
    for f in dataclasses.fields(a):
        name = f.name
        if name in skip:
            continue
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            assert va is not None and vb is not None, name
            va, vb = np.asarray(va), np.asarray(vb)
            assert va.dtype == vb.dtype, name
            assert va.shape == vb.shape, name
            if va.dtype.kind == "f":
                np.testing.assert_array_equal(np.isnan(va), np.isnan(vb), err_msg=name)
                m = ~np.isnan(va)
                if exact:
                    np.testing.assert_array_equal(va[m], vb[m], err_msg=name)
                else:
                    np.testing.assert_allclose(va[m], vb[m], err_msg=name, **_tol_for(name))
            else:
                np.testing.assert_array_equal(va, vb, err_msg=name)
        else:
            assert va == vb, (name, va, vb)


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
    _, twfe, _ = _fit_lane(LANES[name])
    _assert_surface_matches_record(twfe, oracle["lanes"][name]["twfe"])


@pytest.mark.parametrize("name", sorted(LANES))
def test_mpd_twin_still_matches_committed_oracle(name):
    """The refactor left MultiPeriodDiD's own surface untouched too."""
    oracle = _load_oracle()
    _, _, mpd_native = _fit_lane(LANES[name])
    _assert_surface_matches_record(_from_mpd(mpd_native), oracle["lanes"][name]["mpd_twin"])


# --------------------------------------------------------------------------- #
# Live oracles (MultiPeriodDiD exists until PR 4c-ii)
# --------------------------------------------------------------------------- #


def _payload_for(lane: Lane) -> _EventStudyCorePayload:
    df = PANELS[lane.panel]()
    fit = dict(lane.mpd_fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = MultiPeriodDiD(**lane.mpd_ctor)
        payload = est._fit_event_study_core(
            df,
            fit["outcome"],
            fit["treatment"],
            fit["time"],
            fit.get("post_periods"),
            fit.get("covariates"),
            fit.get("fixed_effects"),
            fit.get("absorb"),
            fit.get("reference_period"),
            fit.get("unit"),
            lane.survey(),
            "analytical",
            _frame_offset=1,
        )
    assert isinstance(payload, _EventStudyCorePayload)
    return payload


@pytest.mark.parametrize("name", sorted(LANES))
def test_payload_surface_bit_identical_to_from_mpd(name):
    """Same payload, same shared builder: the two routes must be byte-equal."""
    payload = _payload_for(LANES[name])
    a = _from_event_study_payload(payload, source="MultiPeriodDiDResults")
    b = _from_mpd(_mpd_results_from_payload(payload))
    _assert_surfaces_equal(a, b, exact=True)


@pytest.mark.parametrize("name", sorted(LANES))
def test_twfe_surface_full_container_matches_mpd_twin(name):
    """TWFE fit vs the live MultiPeriodDiD twin over EVERY field but the provenance
    labels: exact on the pooled lanes (same design), 1e-12 on the within lanes
    (the within spec absorbs the unit FE where MPD snaps it)."""
    lane = LANES[name]
    _, twfe, mpd_native = _fit_lane(lane)
    twin = _from_mpd(mpd_native)
    _assert_surfaces_equal(
        twfe, twin, exact=lane.twfe_fit["spec"] == "pooled", skip=("source", "estimation_spec")
    )
    assert twfe.source == "TwoWayFixedEffects"
    assert twfe.estimation_spec == lane.twfe_fit["spec"]
    assert twin.source == "MultiPeriodDiDResults"


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
        warn_legacy_reference_default=False,
        estimator_name="TwoWayFixedEffects",
    )
    assert isinstance(payload, _EventStudyCorePayload)
    assert payload.periods == sorted(payload.periods)
    assert 3 not in payload.periods
    for arr in (payload.att, payload.se, payload.t_stat, payload.p_value, payload.ci_lower):
        assert arr.shape == (len(payload.periods),)
    assert set(payload.df_by_period) == set(payload.periods)
    src = inspect.getsource(diff_diff.twfe)
    assert "_from_mpd" not in src
    assert "MultiPeriodDiDResults" not in src
    assert "period_effects" not in src


def test_from_mpd_unsorted_dict_keeps_label_value_pairing():
    """The sorted-zip contract: an insertion order of {5, 6, 1, 2} must still pair
    every label with ITS PeriodEffect and sort the vcov sub-block with it."""
    effects = {5: 0.5, 6: 0.6, 1: 0.1, 2: 0.2}
    idx = {5: 2, 6: 3, 1: 0, 2: 1}
    period_effects = {
        p: PeriodEffect(
            period=p,
            effect=e,
            se=0.1 * p,
            t_stat=1.0,  # deliberate literal: the test pins label/value pairing, not inference
            p_value=0.5,
            conf_int=(e - 1, e + 1),
        )
        for p, e in effects.items()  # insertion order 5, 6, 1, 2
    }
    full_vcov = np.diag([0.01, 0.04, 0.25, 0.36])
    res = MultiPeriodDiDResults(
        period_effects=period_effects,
        avg_att=0.55,
        avg_se=0.1,
        avg_t_stat=5.5,
        avg_p_value=0.0,
        avg_conf_int=(0.3, 0.8),
        n_obs=10,
        n_treated=5,
        n_control=5,
        pre_periods=[1, 2, 3],
        post_periods=[5, 6],
        reference_period=3,
        vcov=full_vcov,
        interaction_indices=idx,
    )
    s = _from_mpd(res)
    np.testing.assert_array_equal(s.event_time, [1, 2, 3, 5, 6])
    for row, label in enumerate(s.event_time.tolist()):
        if label == 3:
            assert s.is_reference[row]
            continue
        assert s.att[row] == effects[label]
        assert s.se[row] == 0.1 * label
        assert (s.conf_int_lower[row], s.conf_int_upper[row]) == (
            effects[label] - 1,
            effects[label] + 1,
        )
    np.testing.assert_array_equal(s.vcov_index, [1, 2, 5, 6])
    np.testing.assert_array_equal(np.diag(s.vcov), [0.01, 0.04, 0.25, 0.36])


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
# The post-period average helper (replaces avg_att / avg_se in ported tests)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(FINITE_POOLED_LANES))
def test_post_average_identity_matches_mpd_avg(name):
    """`post_period_average` reproduces MultiPeriodDiD's avg_att / avg_se EXACTLY
    on the pooled twin (the live pin that licenses the ported sites' rewrite)."""
    _, twfe, mpd_native = _fit_lane(LANES[name])
    att_bar, se_bar = post_period_average(twfe)
    assert att_bar == mpd_native.avg_att
    assert se_bar == mpd_native.avg_se


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
    # (e) a REAL fit: the rank-deficient NaN post period, TWFE pooled vs MPD twin
    lane = LANES["pooled_nan_post_period"]
    _, twfe, mpd_native = _fit_lane(lane)
    row3 = int(np.where(twfe.event_time == 3)[0][0])
    assert np.isnan(twfe.att[row3])
    att_bar, se_bar = post_period_average(twfe)
    assert np.isnan(att_bar) and np.isnan(mpd_native.avg_att)
    assert np.isnan(se_bar) and np.isnan(mpd_native.avg_se)


def test_within_lanes_are_the_absorb_twins():
    """Guards the lane table itself: every within lane pairs spec='within' + unit
    with an MPD absorb=['unit'] fit (the 1e-12 twin the merge tests pin)."""
    for name in WITHIN_LANES:
        lane = LANES[name]
        assert lane.twfe_fit["unit"] == "unit"
        assert lane.mpd_fit["absorb"] == ["unit"]


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
