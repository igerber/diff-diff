"""Item 6 of the 4.0 cut - the DDD container unification (rows M-014 / M-140 / M-141).

``TripleDifference.fit`` returns ONE results shape for both of its designs:
``TripleDifferenceResults`` with a ``design`` discriminator (``"2x2x2"`` /
``"staggered"``), the canonical quintet stored natively, the group-time table
empty in the 2x2x2 design, and the staggered design's event-study / cohort
surfaces produced POST-FIT by ``aggregate()`` (bootstrapped fits replay the
fit-time multiplier bootstrap from the retained kit). The 3.x
``StaggeredTripleDiffResults`` container and the fit-time ``aggregate=`` /
``balance_e=`` keywords are gone; the paper's Eq. 4.14 overall (the former
``overall_att_es`` field) is ``aggregate('simple', weights='event_time')``.

Numeric pins are literals captured on the UNMODIFIED tree under
``DIFF_DIFF_BACKEND=python`` (the pre-change fit-time ``aggregate="all"`` /
``balance_e=1`` surfaces on the ``generate_staggered_ddd_data(n_units=96,
n_periods=6, cohort_periods=[3, 4], seed=42)`` panel, the ``seed=11`` per-unit
pweight lane of tests/test_v4_merge_ddd.py, and the ``n_bootstrap=49, seed=7,
cband=True`` lane), so they are tolerance-based AND python-backend-guarded like
the merge gates.
"""

from __future__ import annotations

import inspect
import os
import pickle
import warnings
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import diff_diff
from diff_diff import SurveyDesign, TripleDifference, TripleDifferenceResults
from diff_diff.aggregation import AggregationResult
from diff_diff.prep_dgp import generate_ddd_data, generate_staggered_ddd_data
from diff_diff.results_base import EventStudyResults, build_event_study_surface

ORACLE_RTOL = 1e-9
ORACLE_ATOL = 1e-12
_BACKEND_IS_PYTHON = os.environ.get("DIFF_DIFF_BACKEND", "").lower() == "python"
requires_python_backend = pytest.mark.skipif(
    not _BACKEND_IS_PYTHON,
    reason="literal oracles were captured under DIFF_DIFF_BACKEND=python",
)

STAG_KW = dict(n_units=96, n_periods=6, cohort_periods=[3, 4], seed=42)
CROSS_KW = dict(
    n_per_cell=200,
    treatment_effect=2.0,
    group_effect=1.0,
    partition_effect=0.5,
    time_effect=0.7,
    noise_sd=1.0,
    add_covariates=True,
    seed=42,
)
S = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
C = dict(outcome="outcome", group="group", partition="partition", post="time")

# ---------------------------------------------------------------------------
# Oracles (pre-change fit-time surfaces; see the module docstring)
# ---------------------------------------------------------------------------

# aggregate="all" on the plain analytical fit
ES_ORACLE = {  # event time -> (effect, se)
    -2.0: (-0.0027171020705352477, 0.32095504907992906),
    -1.0: (-0.2859337822535215, 0.21658182913951848),
    0.0: (3.160855577888972, 0.24096993187145188),
    1.0: (3.2718809379369556, 0.26314749853218433),
    2.0: (3.3843545714882985, 0.2749885435697253),
    3.0: (3.426518594109598, 0.3635995672681256),
}
GROUP_ORACLE = {  # cohort -> (effect, se)
    3.0: (3.0395757421191214, 0.26041596101583653),
    4.0: (3.634132600087188, 0.3032102909862472),
}
ES_OVERALL_ORACLE = dict(  # overall_att_es and siblings, analytical
    att=3.3109024203559563,
    se=0.23391014483341283,
    t_stat=14.15459095506067,
    p_value=1.7495276788677547e-45,
    ci=(2.8524469608639196, 3.769357879847993),
)
# aggregate="event_study", balance_e=1 (both cohorts are observed through e=1,
# so the balanced surface coincides with the unbalanced one on this panel - the
# pin is that balance_e threads through without disturbing anything)
ES_OVERALL_BALANCE1_ORACLE = ES_OVERALL_ORACLE
# aggregate="event_study", balance_e=3: cohort 3 is observed through e=3 (t=6)
# and cohort 4 only through e=2, so balancing on e=3 DROPS cohort 4 (and with
# it the e=-2 horizon) - a genuinely different surface from the unbalanced one
# (CI review on PR #878 asked for a partial-cohort-removal oracle). (effect, se)
BALANCE3_ES_ORACLE = {
    -1.0: (-0.027719840746928036, 0.27963313073357926),
    0.0: (2.762999666832223, 0.28233821941873094),
    1.0: (2.946059062614397, 0.39516152783198905),
    2.0: (3.0227256449202673, 0.3155992489776067),
    3.0: (3.426518594109598, 0.3635995672681256),
}
BALANCE3_ES_OVERALL_ORACLE = dict(
    att=3.0395757421191214,
    se=0.26041596101583653,
    t_stat=11.672002477353058,
    p_value=1.7720240558757052e-31,
    ci=(2.529169837528695, 3.549981646709548),
)
# n_bootstrap=49, seed=7, cband=True, aggregate="event_study", balance_e=3
BOOT_BALANCE3_CBAND_ORACLE = 2.49333299518289
BOOT_BALANCE3_ES_SE_ORACLE = {
    -1.0: 0.2722181669173392,
    0.0: 0.28424292019694336,
    1.0: 0.3731732550535591,
    2.0: 0.31109084929419584,
    3.0: 0.3383619126874234,
}
BOOT_BALANCE3_ES_OVERALL_ORACLE = dict(
    att=3.0395757421191214,
    se=0.25314100819156565,
    p_value=0.02,
    ci=(2.5589358849283936, 3.5484478904824353),
)
# n_bootstrap=49, seed=7, cband=True, aggregate="all"
BOOT_CBAND_ORACLE = 2.7121511569356116
BOOT_ES_SE_ORACLE = {
    -2.0: 0.31692583310553385,
    -1.0: 0.22336582093958765,
    0.0: 0.22103913589369725,
    1.0: 0.28671846315219596,
    2.0: 0.3034921174110529,
    3.0: 0.3383619126874234,
}
BOOT_GROUP_SE_ORACLE = {3.0: 0.2531410081915656, 4.0: 0.3379035199522487}
BOOT_ES_OVERALL_ORACLE = dict(
    att=3.3109024203559563,
    se=0.23970805380084317,
    p_value=0.02,
    ci=(3.00823738849901, 3.8702221169961026),
)
# survey_design=SurveyDesign(weights="w"), aggregate="all"
SURVEY_ES_OVERALL_ORACLE = dict(
    att=3.28298050259862,
    se=0.25007641854869433,
    t_stat=13.127909147336757,
    p_value=4.581613702172688e-23,
    ci=(2.7865160417217636, 3.779444963475476),
    df=95.0,
)
SURVEY_GROUP_ORACLE = {
    3.0: (2.9067119210278998, 0.28522911091462966),
    4.0: (3.6833713597964355, 0.3124802376382699),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stag():
    return generate_staggered_ddd_data(**STAG_KW)


@pytest.fixture(scope="module")
def cross():
    return generate_ddd_data(**CROSS_KW)


@pytest.fixture(scope="module")
def stag_survey(stag):
    df = stag.copy()
    rng = np.random.default_rng(11)
    units = sorted(df["unit"].unique())
    df["w"] = df["unit"].map(dict(zip(units, rng.uniform(0.5, 2.0, size=len(units)))))
    return df


def _fit_stag(df, ctor=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TripleDifference(**(ctor or {})).fit(df, partition="eligibility", **S, **kw)


@pytest.fixture(scope="module")
def stag_fit(stag):
    return _fit_stag(stag)


@pytest.fixture(scope="module")
def boot_fit(stag):
    return _fit_stag(stag, ctor=dict(n_bootstrap=49, seed=7, cband=True))


@pytest.fixture(scope="module")
def survey_fit(stag_survey):
    return _fit_stag(stag_survey, survey_design=SurveyDesign(weights="w"))


@pytest.fixture(scope="module")
def cross_fit(cross):
    return TripleDifference().fit(cross, **C)


@pytest.fixture(scope="module")
def half_step_fit():
    # A half-step calendar: periods 0.5..4.0 and cohorts 2.0 / 2.5, so the
    # event-study horizons e = t - g are fractional (-1.0, -0.5, 0.0, 0.5, ...).
    df = generate_staggered_ddd_data(n_units=96, n_periods=8, cohort_periods=[4, 5], seed=42)
    df["period"] = df["period"] / 2.0
    df["first_treat"] = df["first_treat"] / 2.0
    return _fit_stag(df)


def _row(es, e):
    idx = [i for i, t in enumerate(es.event_time) if float(t) == e]
    assert len(idx) == 1, (e, es.event_time)
    return idx[0]


# ---------------------------------------------------------------------------
# (1) Removal pins - the class, its module, the fit-time keywords
# ---------------------------------------------------------------------------


class TestRemoval:
    def test_staggered_container_is_gone(self):
        import importlib

        assert not hasattr(diff_diff, "StaggeredTripleDiffResults")
        assert "StaggeredTripleDiffResults" not in diff_diff.__all__
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("diff_diff.staggered_triple_diff_results")
        assert "TripleDifferenceResults" in diff_diff.__all__

    def test_results_module_moved_and_re_exported(self):
        from diff_diff import triple_diff, triple_diff_results

        assert TripleDifferenceResults is triple_diff_results.TripleDifferenceResults
        # The 3.x class path still resolves (pickles and existing readers).
        assert triple_diff.TripleDifferenceResults is TripleDifferenceResults

    @pytest.mark.parametrize("kw", [{"aggregate": "event_study"}, {"balance_e": 1}])
    def test_fit_time_keywords_raise_type_error_in_both_designs(self, stag, cross, kw):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            TripleDifference().fit(cross, **C, **kw)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            TripleDifference().fit(stag, partition="eligibility", **S, **kw)

    def test_fit_signature_lacks_the_keywords(self):
        params = inspect.signature(TripleDifference.fit).parameters
        assert "aggregate" not in params and "balance_e" not in params
        assert "balance_e" in inspect.signature(TripleDifferenceResults.aggregate).parameters


# ---------------------------------------------------------------------------
# (2) Shape - one class, two designs, canonical storage
# ---------------------------------------------------------------------------


class TestShape:
    def test_both_designs_return_the_unified_class(self, stag_fit, cross_fit):
        assert type(stag_fit) is type(cross_fit) is TripleDifferenceResults
        assert stag_fit.design == "staggered"
        assert cross_fit.design == "2x2x2"

    def test_degenerate_2x2x2_view(self, cross_fit):
        assert cross_fit.group_time_effects == {}
        assert cross_fit.groups == [] and cross_fit.time_periods == []
        assert cross_fit.n_treated_units is None and cross_fit.control_group is None
        assert cross_fit.n_treated_eligible is not None
        assert cross_fit.vcov_type == "hc1"

    def test_staggered_fields(self, stag_fit):
        assert set(stag_fit.groups) == {3.0, 4.0}
        assert len(stag_fit.time_periods) == 6
        assert stag_fit.n_treated_eligible is None and stag_fit.vcov_type is None
        assert stag_fit.control_group == "not_yet_treated"
        assert stag_fit.anticipation == 0 and stag_fit.pscore_trim == 0.01
        assert stag_fit.inference_method == "analytical" and stag_fit.n_bootstrap is None
        # The former fit-time surfaces are carrier fields, None on every fit.
        assert stag_fit.event_study_effects is None
        assert stag_fit.group_effects is None
        assert stag_fit.cband_crit_value is None

    def test_canonical_quintet_native_and_no_overall_storage(self):
        names = {f.name for f in fields(TripleDifferenceResults)}
        assert {"att", "se", "t_stat", "p_value", "conf_int", "design", "df_inference"} <= names
        assert not [n for n in names if n.startswith("overall_")]
        for legacy in ("overall_att", "overall_se", "overall_att_es", "overall_se_es"):
            assert not hasattr(TripleDifferenceResults, legacy)

    def test_2x2x2_positional_layout_is_byte_stable(self):
        res = TripleDifferenceResults(1.0, 0.5, 2.0, 0.05, (0.0, 2.0), 100, 25, 25, 25, 25, "dr")
        assert res.design == "2x2x2" and res.estimation_method == "dr"
        assert res.n_treated_eligible == 25 and res.alpha == 0.05

    def test_post_init_guards(self):
        with pytest.raises(ValueError, match="design must be one of"):
            TripleDifferenceResults(
                1.0, 0.5, 2.0, 0.05, (0.0, 2.0), 100, 25, 25, 25, 25, design="x"
            )
        with pytest.raises(ValueError, match="requires non-empty group_time_effects"):
            TripleDifferenceResults(1.0, 0.5, 2.0, 0.05, (0.0, 2.0), 100, design="staggered")
        with pytest.raises(ValueError, match="requires the four cell counts"):
            TripleDifferenceResults(1.0, 0.5, 2.0, 0.05, (0.0, 2.0), 100)

    def test_df_inference_provenance(self, stag_fit, cross_fit, boot_fit, survey_fit):
        # 2x2x2 non-survey: the engine's t df (n_obs - 8 - len(covariates))
        assert cross_fit.df_inference == cross_fit.n_obs - 8
        # staggered: the overall uses the normal reference on plain fits
        assert stag_fit.df_inference is None and boot_fit.df_inference is None
        assert survey_fit.df_inference is None and survey_fit.survey_metadata.df_survey == 95


# ---------------------------------------------------------------------------
# (3) Post-fit aggregate()
# ---------------------------------------------------------------------------


class TestAggregateVocabulary:
    def test_simple_relays_the_stored_quintet(self, stag_fit, cross_fit):
        for res, n_kind, n in (
            (stag_fit, "units", 96.0),
            (cross_fit, "obs", float(cross_fit.n_obs)),
        ):
            simple = res.aggregate("simple")
            assert isinstance(simple, AggregationResult)
            assert simple.level == "simple" and list(simple.label) == ["overall"]
            assert simple.att[0] == res.att and simple.se[0] == res.se
            assert simple.t_stat[0] == res.t_stat and simple.p_value[0] == res.p_value
            assert (simple.conf_int_lower[0], simple.conf_int_upper[0]) == tuple(res.conf_int)
            assert simple.n_kind == n_kind and simple.n[0] == n
            assert simple.estimator == "TripleDifference"
        assert np.isnan(stag_fit.aggregate("simple").df[0])  # normal reference at fit
        assert cross_fit.aggregate("simple").df[0] == cross_fit.n_obs - 8

    @pytest.mark.parametrize("level", ["event_study", "group"])
    def test_2x2x2_recompute_levels_name_the_staggered_design(self, cross_fit, level):
        with pytest.raises(ValueError, match="fit the staggered design"):
            cross_fit.aggregate(level)

    def test_2x2x2_rejects_weights_and_balance_e(self, cross_fit):
        with pytest.raises(ValueError, match="fit the staggered design"):
            cross_fit.aggregate("simple", weights="event_time")
        with pytest.raises(ValueError, match="fit the staggered design"):
            cross_fit.aggregate("simple", balance_e=1)

    def test_total_is_not_offered(self, stag_fit):
        with pytest.raises(ValueError, match="Unsupported aggregation type 'total'"):
            stag_fit.aggregate("total")

    def test_weights_selector_contract(self, stag_fit):
        with pytest.raises(ValueError, match="weights='event_time'"):
            stag_fit.aggregate("simple", weights="bogus")
        for level in ("event_study", "group"):
            with pytest.raises(ValueError, match="applies to aggregate\\('simple'\\) only"):
                stag_fit.aggregate(level, weights="event_time")

    def test_balance_e_on_plain_simple_is_rejected(self, stag_fit):
        with pytest.raises(ValueError, match="balance_e applies to"):
            stag_fit.aggregate("simple", balance_e=1)


@requires_python_backend
class TestAggregateReproducesFitTimeSurfaces:
    def test_event_study(self, stag_fit):
        es = stag_fit.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        assert sorted(float(t) for t in es.event_time) == sorted(ES_ORACLE)
        for e, (eff, se) in ES_ORACLE.items():
            i = _row(es, e)
            assert_allclose(es.att[i], eff, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
            assert_allclose(es.se[i], se, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert es.cband_crit_value is None
        assert es.vcov is not None and es.vcov.shape == (len(ES_ORACLE), len(ES_ORACLE))

    def test_group(self, stag_fit):
        grp = stag_fit.aggregate("group")
        assert isinstance(grp, AggregationResult) and grp.level == "group"
        got = dict(zip((float(g) for g in grp.label), zip(grp.att, grp.se)))
        assert set(got) == set(GROUP_ORACLE)
        for g, (eff, se) in GROUP_ORACLE.items():
            assert_allclose(got[g][0], eff, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
            assert_allclose(got[g][1], se, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)

    def test_event_time_overall_is_the_eq_414_field(self, stag_fit):
        o = stag_fit.aggregate("simple", weights="event_time")
        assert o.level == "simple" and list(o.label) == ["overall"] and list(o.target) == ["att"]
        assert_allclose(o.att[0], ES_OVERALL_ORACLE["att"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(o.se[0], ES_OVERALL_ORACLE["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(
            o.t_stat[0], ES_OVERALL_ORACLE["t_stat"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL
        )
        assert_allclose(
            o.p_value[0], ES_OVERALL_ORACLE["p_value"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL
        )
        assert_allclose(o.conf_int_lower[0], ES_OVERALL_ORACLE["ci"][0], rtol=ORACLE_RTOL)
        assert_allclose(o.conf_int_upper[0], ES_OVERALL_ORACLE["ci"][1], rtol=ORACLE_RTOL)
        assert np.isnan(o.df[0])  # plain fit: normal reference, no df governed it
        assert o.n_kind == "units" and o.n[0] == 96.0 and o.weight[0] == 1.0
        # distinct from the cohort-weighted headline
        assert o.att[0] != stag_fit.att

    def test_balance_e_threads_into_the_eq_414_overall(self, stag_fit):
        o = stag_fit.aggregate("simple", weights="event_time", balance_e=1)
        assert_allclose(o.att[0], ES_OVERALL_BALANCE1_ORACLE["att"], rtol=ORACLE_RTOL)
        assert_allclose(o.se[0], ES_OVERALL_BALANCE1_ORACLE["se"], rtol=ORACLE_RTOL)

    def test_balance_e_3_drops_cohort_4_analytical(self, stag_fit):
        # Partial cohort removal: cohort 4 is unobserved at e=3, so the balanced
        # surface loses it (and the e=-2 horizon it alone contributed).
        es = stag_fit.aggregate("event_study", balance_e=3)
        assert [float(t) for t in es.event_time] == sorted(BALANCE3_ES_ORACLE)
        for e, (eff, se) in BALANCE3_ES_ORACLE.items():
            i = _row(es, e)
            assert_allclose(es.att[i], eff, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
            assert_allclose(es.se[i], se, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        unbalanced = stag_fit.aggregate("event_study")
        assert len(unbalanced.event_time) == len(es.event_time) + 1
        assert es.att[_row(es, 0.0)] != unbalanced.att[_row(unbalanced, 0.0)]
        o = stag_fit.aggregate("simple", weights="event_time", balance_e=3)
        assert_allclose(o.att[0], BALANCE3_ES_OVERALL_ORACLE["att"], rtol=ORACLE_RTOL)
        assert_allclose(o.se[0], BALANCE3_ES_OVERALL_ORACLE["se"], rtol=ORACLE_RTOL)
        assert_allclose(o.t_stat[0], BALANCE3_ES_OVERALL_ORACLE["t_stat"], rtol=ORACLE_RTOL)
        assert_allclose(o.p_value[0], BALANCE3_ES_OVERALL_ORACLE["p_value"], rtol=ORACLE_RTOL)
        assert_allclose(o.conf_int_lower[0], BALANCE3_ES_OVERALL_ORACLE["ci"][0], rtol=ORACLE_RTOL)
        assert_allclose(o.conf_int_upper[0], BALANCE3_ES_OVERALL_ORACLE["ci"][1], rtol=ORACLE_RTOL)
        assert o.att[0] != ES_OVERALL_ORACLE["att"]

    def test_balance_e_3_bootstrap_replay(self, boot_fit):
        es = boot_fit.aggregate("event_study", balance_e=3)
        assert [float(t) for t in es.event_time] == sorted(BOOT_BALANCE3_ES_SE_ORACLE)
        assert_allclose(
            es.cband_crit_value, BOOT_BALANCE3_CBAND_ORACLE, rtol=ORACLE_RTOL, atol=ORACLE_ATOL
        )
        assert es.cband_crit_value != BOOT_CBAND_ORACLE  # the replay re-ran on the balanced set
        for e, se in BOOT_BALANCE3_ES_SE_ORACLE.items():
            i = _row(es, e)
            assert_allclose(es.se[i], se, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
            assert_allclose(es.att[i], BALANCE3_ES_ORACLE[e][0], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        o = boot_fit.aggregate("simple", weights="event_time", balance_e=3)
        assert_allclose(o.att[0], BOOT_BALANCE3_ES_OVERALL_ORACLE["att"], rtol=ORACLE_RTOL)
        assert_allclose(o.se[0], BOOT_BALANCE3_ES_OVERALL_ORACLE["se"], rtol=ORACLE_RTOL)
        assert_allclose(o.p_value[0], BOOT_BALANCE3_ES_OVERALL_ORACLE["p_value"], rtol=ORACLE_RTOL)
        assert_allclose(
            o.conf_int_lower[0], BOOT_BALANCE3_ES_OVERALL_ORACLE["ci"][0], rtol=ORACLE_RTOL
        )
        assert_allclose(
            o.conf_int_upper[0], BOOT_BALANCE3_ES_OVERALL_ORACLE["ci"][1], rtol=ORACLE_RTOL
        )
        assert np.isnan(o.df[0])

    def test_bootstrap_replay(self, boot_fit):
        assert boot_fit.inference_method == "bootstrap" and boot_fit.n_bootstrap == 49
        es = boot_fit.aggregate("event_study")
        assert_allclose(es.cband_crit_value, BOOT_CBAND_ORACLE, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        for e, se in BOOT_ES_SE_ORACLE.items():
            assert_allclose(es.se[_row(es, e)], se, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert es.cband_lower is not None and np.all(es.cband_lower <= es.att)
        grp = boot_fit.aggregate("group")
        got = dict(zip((float(g) for g in grp.label), grp.se))
        for g, se in BOOT_GROUP_SE_ORACLE.items():
            assert_allclose(got[g], se, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert np.isnan(grp.df).all()  # percentile inference: no df governs it
        o = boot_fit.aggregate("simple", weights="event_time")
        assert_allclose(o.att[0], BOOT_ES_OVERALL_ORACLE["att"], rtol=ORACLE_RTOL)
        assert_allclose(o.se[0], BOOT_ES_OVERALL_ORACLE["se"], rtol=ORACLE_RTOL)
        assert_allclose(o.p_value[0], BOOT_ES_OVERALL_ORACLE["p_value"], rtol=ORACLE_RTOL)
        assert_allclose(o.conf_int_lower[0], BOOT_ES_OVERALL_ORACLE["ci"][0], rtol=ORACLE_RTOL)
        assert np.isnan(o.df[0])
        simple = boot_fit.aggregate("simple")
        assert simple.se[0] == boot_fit.se and np.isnan(simple.df[0])

    def test_survey_pweight(self, survey_fit):
        grp = survey_fit.aggregate("group")
        got = dict(zip((float(g) for g in grp.label), zip(grp.att, grp.se)))
        for g, (eff, se) in SURVEY_GROUP_ORACLE.items():
            assert_allclose(got[g][0], eff, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
            assert_allclose(got[g][1], se, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        o = survey_fit.aggregate("simple", weights="event_time")
        assert_allclose(o.att[0], SURVEY_ES_OVERALL_ORACLE["att"], rtol=ORACLE_RTOL)
        assert_allclose(o.se[0], SURVEY_ES_OVERALL_ORACLE["se"], rtol=ORACLE_RTOL)
        assert_allclose(o.p_value[0], SURVEY_ES_OVERALL_ORACLE["p_value"], rtol=ORACLE_RTOL)
        # the survey df governed the t-reference, and the relay reports it too
        assert o.df[0] == SURVEY_ES_OVERALL_ORACLE["df"]
        assert survey_fit.aggregate("simple").df[0] == SURVEY_ES_OVERALL_ORACLE["df"]


class TestEq414Warnings:
    def test_no_post_horizon_fails_soft(self, stag_fit):
        with pytest.warns(UserWarning, match="no post-treatment horizons are available"):
            o = stag_fit.aggregate("simple", weights="event_time", balance_e=100)
        assert np.isnan(o.att[0]) and np.isnan(o.se[0]) and np.isnan(o.p_value[0])
        assert np.isfinite(stag_fit.att)  # the headline is untouched

    def test_single_psu_bootstrap_se_is_nan_with_cause_accurate_warning(self, stag):
        df = stag.copy()
        df["w"] = 1.0
        df["psu"] = 0  # one PSU -> the bootstrap variance is unidentified
        df["fpc_col"] = 10**6
        design = SurveyDesign(weights="w", psu="psu", fpc="fpc_col")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = TripleDifference(estimation_method="reg", n_bootstrap=20, seed=7).fit(
                df, partition="eligibility", survey_design=design, **S
            )
            o = res.aggregate("simple", weights="event_time")
        assert np.isfinite(o.att[0]) and np.isnan(o.se[0]) and np.isnan(o.p_value[0])
        msgs = [str(w.message) for w in caught if "weights='event_time'))" in str(w.message)]
        assert msgs and all("unidentified" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# (4) Read surfaces (no FutureWarning anywhere; the duck-typed consumers)
# ---------------------------------------------------------------------------


class TestReadSurfaces:
    @pytest.fixture(params=["stag_fit", "cross_fit", "boot_fit"])
    def res(self, request):
        return request.getfixturevalue(request.param)

    def test_surfaces_are_warning_free(self, res):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            d = res.to_dict()
            assert d["att"] == res.att and d["design"] == res.design
            assert "df_inference" in d and not [k for k in d if k.startswith("overall_")]
            assert "ATT" in res.summary() and "TripleDifferenceResults" in repr(res)
            frame = res.to_dataframe("overall")
            assert len(frame) == 1 and frame["att"].iloc[0] == res.att
            assert isinstance(res.coef_var, float) and isinstance(res.is_significant, bool)
            assert isinstance(res.epv_summary(), pd.DataFrame)

    def test_to_dataframe_levels(self, stag_fit, cross_fit):
        assert len(stag_fit.to_dataframe()) == len(stag_fit.group_time_effects)
        assert list(stag_fit.to_dataframe().columns[:2]) == ["group", "time"]
        assert len(cross_fit.to_dataframe()) == 1
        for cols_expected in ("design", "pscore_trim", "alpha", "df_inference", "att"):
            assert cols_expected in cross_fit.to_dataframe().columns
        with pytest.raises(ValueError, match="fit the staggered design"):
            cross_fit.to_dataframe("group_time")
        # 4.0 (M-093 8b): level="event_study" relays the aggregate() container;
        # level="group" still raises; the 2x2x2 design has no event study.
        pd.testing.assert_frame_equal(
            stag_fit.to_dataframe("event_study"),
            stag_fit.aggregate("event_study").to_dataframe(),
        )
        with pytest.raises(ValueError, match="not stored on 4.0 fits"):
            stag_fit.to_dataframe("group")
        with pytest.raises(ValueError, match="2x2x2 TripleDifference design"):
            cross_fit.to_dataframe("event_study")
        with pytest.raises(ValueError, match="Unknown level"):
            stag_fit.to_dataframe("bogus")

    def test_to_dict_blocks(self, stag_fit, cross_fit, boot_fit):
        d = stag_fit.to_dict()
        assert d["n_groups"] == 2 and d["control_group"] == "not_yet_treated"
        assert "n_treated_eligible" not in d and "n_bootstrap" not in d
        assert boot_fit.to_dict()["n_bootstrap"] == 49
        c = cross_fit.to_dict()
        assert c["n_treated_eligible"] is not None and "n_groups" not in c
        assert c["vcov_type"] == "hc1"

    def test_summary_text_by_design(self, stag_fit, cross_fit):
        s = stag_fit.summary()
        assert "Staggered Triple Difference" in s and "results.aggregate('event_study')" in s
        assert "Eq. 4.14" not in s
        assert "Sample Composition by Cell" in cross_fit.summary()
        assert "design='staggered'" in repr(stag_fit) and "design=" not in repr(cross_fit)

    def test_epv_summary_columns(self, stag_fit, cross_fit):
        assert list(stag_fit.epv_summary().columns[:2]) == ["group", "time"]
        assert list(cross_fit.epv_summary().columns[:1]) == ["subgroup"]

    def test_duck_typed_consumers_on_2x2x2(self, cross_fit):
        from diff_diff import plot_event_study, plot_group_effects

        with pytest.raises(ValueError, match="staggered design only"):
            build_event_study_surface(cross_fit)
        with pytest.raises(ValueError, match="no group-time cells to plot"):
            plot_group_effects(cross_fit, show=False)
        with pytest.raises(TypeError):
            plot_event_study(cross_fit, show=False)


# ---------------------------------------------------------------------------
# (5) Pickles
# ---------------------------------------------------------------------------


class TestPickle:
    def test_round_trip_both_designs(self, stag_fit, cross_fit):
        back = pickle.loads(pickle.dumps(stag_fit))
        assert back.design == "staggered" and back.att == stag_fit.att
        es = back.aggregate("event_study")  # the kit survives
        assert len(es.event_time) == len(stag_fit.aggregate("event_study").event_time)
        back_c = pickle.loads(pickle.dumps(cross_fit))
        assert back_c.design == "2x2x2" and back_c.to_dict() == cross_fit.to_dict()

    def test_simulated_3x_2x2x2_state_loads_with_defaults(self, cross_fit):
        state = dict(cross_fit.__dict__)
        new_keys = {f.name for f in fields(TripleDifferenceResults)} - {
            "att",
            "se",
            "t_stat",
            "p_value",
            "conf_int",
            "n_obs",
            "n_treated_eligible",
            "n_treated_ineligible",
            "n_control_eligible",
            "n_control_ineligible",
            "estimation_method",
            "alpha",
            "group_means",
            "pscore_stats",
            "r_squared",
            "covariate_balance",
            "inference_method",
            "vcov_type",
            "cluster_name",
            "n_bootstrap",
            "n_clusters",
            "survey_metadata",
            "epv_diagnostics",
            "epv_threshold",
            "pscore_fallback",
        }
        for key in new_keys | {"_aggregation_kit"}:
            state.pop(key, None)
        obj = TripleDifferenceResults.__new__(TripleDifferenceResults)
        obj.__setstate__(state)
        assert obj.design == "2x2x2" and obj.group_time_effects == {} and obj.groups == []
        assert obj.df_inference is None and obj.pscore_trim == 0.01
        assert obj.att == cross_fit.att and obj.aggregate("simple").att[0] == cross_fit.att

    def test_old_class_path_unpickles(self, cross_fit):
        # Protocol 0 spells the class path in clear text, so a 3.x pickle (class
        # path diff_diff.triple_diff.TripleDifferenceResults) is simulated by
        # rewriting the module name; the re-export is what makes it resolve.
        raw = pickle.dumps(cross_fit, protocol=0)
        assert b"diff_diff.triple_diff_results\nTripleDifferenceResults" in raw
        legacy = raw.replace(
            b"diff_diff.triple_diff_results\nTripleDifferenceResults",
            b"diff_diff.triple_diff\nTripleDifferenceResults",
        )
        back = pickle.loads(legacy)
        assert type(back) is TripleDifferenceResults and back.att == cross_fit.att


# ---------------------------------------------------------------------------
# (6) Report surface
# ---------------------------------------------------------------------------


class TestReportSurface:
    def test_target_parameter_headline_is_att_in_both_designs(self, stag_fit, cross_fit):
        from diff_diff._reporting_helpers import describe_target_parameter

        tp_s = describe_target_parameter(stag_fit)
        tp_c = describe_target_parameter(cross_fit)
        assert tp_s["aggregation"] == "staggered_ddd" and tp_s["headline_attribute"] == "att"
        assert tp_c["aggregation"] == "ddd" and tp_c["headline_attribute"] == "att"

    def test_diagnostic_report_staggered_derives_pt_post_fit(self, stag_fit):
        from diff_diff.diagnostic_report import DIAGNOSTIC_REPORT_SCHEMA_VERSION, DiagnosticReport

        assert DIAGNOSTIC_REPORT_SCHEMA_VERSION == "3.0"
        dr = DiagnosticReport(stag_fit).run_all()
        assert "parallel_trends" in dr.applicable_checks
        reason = dr.skipped_checks.get("parallel_trends", "")
        assert "Re-fit" not in reason and "aggregate='event_study'" not in reason
        d = dr.to_dict()
        assert d["design"] == "staggered" and d["estimator"] == "TripleDifferenceResults"
        assert d["headline_metric"]["name"] == "att"
        assert d["target_parameter"]["headline_attribute"] == "att"

    def test_diagnostic_report_keeps_fractional_pre_horizons(self, half_step_fit):
        # CI review on PR #878 (P0): the DR adapter used to key the derived
        # surface by ``int(k)``, so on a half-step calendar ``-0.5`` collapsed
        # into the treatment horizon ``0`` and the pre-period vanished from
        # the PT family without a warning. Every negative horizon on the
        # surface must reach the PT test, labelled at its real value and
        # aligned with the covariance index (joint Wald, not Bonferroni).
        from diff_diff.diagnostic_report import DiagnosticReport

        es = half_step_fit.aggregate("event_study")
        horizons = [float(t) for t in es.event_time]
        assert any(not float(t).is_integer() for t in horizons)
        pre = [t for t, ref in zip(horizons, es.is_reference) if t < 0 and not ref]
        assert -0.5 in pre and len(pre) >= 2
        pt = DiagnosticReport(half_step_fit).to_dict()["parallel_trends"]
        assert pt["status"] == "ran" and pt["method"] == "joint_wald_event_study"
        assert pt["n_pre_periods"] == len(pre)
        assert [row["period"] for row in pt["per_period"]] == sorted(pre)
        assert np.isfinite(pt["joint_p_value"])

    def test_diagnostic_report_2x2x2_has_no_pt_row(self, cross_fit):
        from diff_diff.diagnostic_report import DiagnosticReport

        dr = DiagnosticReport(cross_fit).run_all()
        assert "parallel_trends" not in dr.applicable_checks
        assert "parallel_trends" not in dr.skipped_checks
        assert dr.to_dict()["design"] == "2x2x2"

    def test_diagnostic_report_design_is_none_elsewhere(self):
        from diff_diff import CallawaySantAnna
        from diff_diff.diagnostic_report import DiagnosticReport
        from diff_diff.prep_dgp import generate_staggered_data

        cs = CallawaySantAnna().fit(
            generate_staggered_data(n_units=60, n_periods=5, cohort_periods=[3], seed=1),
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert DiagnosticReport(cs).run_all().to_dict()["design"] is None

    @pytest.mark.parametrize("control_group", ["not_yet_treated", "never_treated"])
    def test_business_report_staggered(self, stag, control_group):
        from diff_diff.business_report import BUSINESS_REPORT_SCHEMA_VERSION, BusinessReport

        assert BUSINESS_REPORT_SCHEMA_VERSION == "4.0"
        res = _fit_stag(stag, ctor=dict(control_group=control_group))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = BusinessReport(res, auto_diagnostics=False).to_dict()
        assert d["estimator"]["class_name"] == "TripleDifferenceResults"
        assert d["estimator"]["design"] == "staggered"
        assert d["sample"]["n_never_enabled"] > 0 and d["sample"]["n_control"] is None
        assert d["target_parameter"]["headline_attribute"] == "att"
        assert d["assumption"]["parallel_trends_variant"] == "triple_difference_cancellation"

    def test_business_report_2x2x2(self, cross_fit):
        from diff_diff.business_report import BusinessReport

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = BusinessReport(cross_fit, auto_diagnostics=False).to_dict()
        assert d["estimator"]["design"] == "2x2x2"
        assert d["assumption"]["parallel_trends_variant"] == "triple_difference_cancellation"


# ---------------------------------------------------------------------------
# (7) Practitioner routing
# ---------------------------------------------------------------------------


class TestPractitioner:
    def test_staggered_fit_gets_the_post_fit_steps(self, stag_fit, cross_fit):
        from diff_diff.practitioner import practitioner_next_steps

        staggered = practitioner_next_steps(stag_fit, verbose=False)
        code = "\n".join(step["code"] for step in staggered["next_steps"])
        assert "results.aggregate('event_study')" in code
        assert "weights='event_time'" in code
        assert "aggregate=" not in code  # no fit-time keyword is taught anywhere
        two_by_two = practitioner_next_steps(cross_fit, verbose=False)
        labels = [step["label"] for step in two_by_two["next_steps"]]
        assert "Assess DDD identifying assumption" in labels
        assert "aggregate('event_study')" not in "\n".join(
            s["code"] for s in two_by_two["next_steps"]
        )
