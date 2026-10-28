"""Tests for the staggered-DDD engine behind TripleDifference's staggered mode
(``fit(..., first_treat=, partition=)``; formerly the ``StaggeredTripleDifference``
class, removed at the 4.0 cut, PR 4a).

Since PR 6 of the 4.0 cut the staggered fit returns the unified
``TripleDifferenceResults`` (``design == "staggered"``); the event-study, cohort and
Eq. 4.14 surfaces are post-fit ``results.aggregate(...)`` calls, not fit-time fields.
"""

import numpy as np
import pytest

from diff_diff import (
    AggregationResult,
    EventStudyResults,
    TripleDifference,
    TripleDifferenceResults,
    generate_staggered_ddd_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_data():
    """Staggered DDD data with known treatment effect, no covariates."""
    return generate_staggered_ddd_data(n_units=300, treatment_effect=3.0, seed=42)


@pytest.fixture(scope="module")
def data_with_covariates():
    """Staggered DDD data with covariates."""
    return generate_staggered_ddd_data(
        n_units=300, treatment_effect=3.0, add_covariates=True, seed=123
    )


@pytest.fixture(scope="module")
def null_effect_data():
    """Staggered DDD data with zero treatment effect."""
    return generate_staggered_ddd_data(n_units=300, treatment_effect=0.0, seed=99)


@pytest.fixture(scope="module")
def dynamic_data():
    """Staggered DDD data with dynamic treatment effects."""
    return generate_staggered_ddd_data(
        n_units=300,
        treatment_effect=3.0,
        dynamic_effects=True,
        effect_growth=0.2,
        seed=55,
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffInit:
    def test_default_params(self):
        est = TripleDifference()
        params = est.get_params()
        assert params["estimation_method"] == "dr"
        assert params["alpha"] == 0.05
        assert params["anticipation"] == 0
        assert params["base_period"] == "varying"
        assert params["n_bootstrap"] == 0
        assert params["pscore_trim"] == 0.01

    def test_set_params(self):
        est = TripleDifference()
        est.set_params(estimation_method="ipw", alpha=0.10)
        assert est.estimation_method == "ipw"
        assert est.alpha == 0.10

    def test_set_params_updates_bootstrap_weights(self):
        est = TripleDifference()
        est.set_params(bootstrap_weights="mammen")
        assert est.bootstrap_weights == "mammen"

    def test_invalid_estimation_method(self):
        with pytest.raises(ValueError, match="estimation_method"):
            TripleDifference(estimation_method="ols")

    def test_invalid_pscore_trim(self):
        with pytest.raises(ValueError, match="pscore_trim"):
            TripleDifference(pscore_trim=0.6)

    def test_invalid_base_period(self):
        with pytest.raises(ValueError, match="base_period"):
            TripleDifference(base_period="fixed")

    def test_set_params_unknown(self):
        est = TripleDifference()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_param=42)


# ---------------------------------------------------------------------------
# Basic fit tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffBasic:
    def test_fit_returns_results(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert isinstance(res, TripleDifferenceResults)
        assert res.design == "staggered"
        assert est.is_fitted_

    def test_results_structure(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert isinstance(res.att, float)
        assert isinstance(res.se, float)
        assert res.se > 0
        assert isinstance(res.conf_int, tuple)
        assert len(res.conf_int) == 2
        assert res.conf_int[0] < res.att < res.conf_int[1]
        assert res.t_stat == pytest.approx(res.att / res.se)
        # The 3.x fit-time surfaces are not stored on 4.0 fits (post-fit aggregate()).
        assert res.event_study_effects is None
        assert res.group_effects is None
        assert res.cband_crit_value is None
        assert len(res.groups) > 0
        assert len(res.time_periods) > 0
        assert res.n_obs > 0
        assert res.n_treated_units > 0
        assert res.n_control_units > 0
        assert res.n_eligible > 0
        assert res.n_ineligible > 0

    def test_group_time_effects_populated(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert len(res.group_time_effects) > 0
        for (g, t), eff in res.group_time_effects.items():
            assert "effect" in eff
            assert "se" in eff
            assert "t_stat" in eff
            assert "p_value" in eff
            assert "conf_int" in eff
            assert "n_treated" in eff
            assert "n_control" in eff

    def test_summary_runs(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        summary = res.summary()
        assert "Staggered Triple Difference" in summary
        assert "ATT" in summary
        # 4.0: no fit-time event-study / cohort / Eq. 4.14 blocks; a pointer line instead.
        assert "results.aggregate('event_study')" in summary
        assert "results.aggregate('group')" in summary
        assert "Eq. 4.14" not in summary

    def test_event_time_overall_aggregate_surface(self, simple_data):
        """The paper's Eq. 4.14 overall (the 3.x fit-time ``overall_att_es`` field) is
        a post-fit surface: ``aggregate("simple", weights="event_time")`` returns a
        one-row AggregationResult that is the unweighted mean of the post-treatment
        event-study effects. The fit-time fields are gone from the attribute, summary()
        and to_dict() surfaces."""
        res = TripleDifference().fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        o = res.aggregate("simple", weights="event_time")
        assert isinstance(o, AggregationResult)
        assert len(o.att) == 1
        assert np.isfinite(o.att[0]) and np.isfinite(o.se[0]) and o.se[0] > 0
        assert o.t_stat[0] == pytest.approx(o.att[0] / o.se[0])
        assert 0.0 <= o.p_value[0] <= 1.0
        assert o.conf_int_lower[0] < o.att[0] < o.conf_int_upper[0]
        # Eq. 4.14: the unweighted mean of the finite post-treatment ES(e) effects.
        es = res.aggregate("event_study")
        post = es.att[(es.event_time >= 0) & ~es.is_reference & np.isfinite(es.att)]
        assert len(post) > 0
        assert o.att[0] == pytest.approx(post.mean(), abs=1e-10)
        # The Eq. 4.14 overall differs from the stored cohort-size-weighted headline.
        assert o.att[0] != res.att
        # The 3.x fit-time fields and their summary()/to_dict() entries are gone.
        for f in (
            "overall_att_es",
            "overall_se_es",
            "overall_t_stat_es",
            "overall_p_value_es",
            "overall_conf_int_es",
        ):
            assert not hasattr(res, f)
        assert not any(k.startswith("overall_") for k in res.to_dict())
        assert "Eq. 4.14" not in res.summary()

    def test_event_time_overall_unperturbed_by_other_aggregations(self, simple_data):
        """The Eq. 4.14 overall is deterministic and independent of which other
        post-fit aggregations ran on the same result (the 4.0 form of the 3.x
        guard that aggregate='all' matched aggregate='event_study' bit-for-bit).

        aggregate('event_study') / aggregate('group') memoize into a scratch copy of
        the kit and must neither drop nor perturb the event-study-average overall.
        """
        res = TripleDifference().fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        o_first = res.aggregate("simple", weights="event_time")
        # Run the other recompute surfaces in between (the former aggregate='all').
        res.aggregate("group")
        res.aggregate("event_study")
        o_again = res.aggregate("simple", weights="event_time")
        assert np.isfinite(o_first.att[0]) and np.isfinite(o_first.se[0])
        # Identical point estimate AND inference: same data, same analytical
        # influence-function SE (no bootstrap), so bit-for-bit close.
        assert o_again.att[0] == pytest.approx(o_first.att[0], abs=1e-10)
        assert o_again.se[0] == pytest.approx(o_first.se[0], abs=1e-10)
        assert o_again.t_stat[0] == pytest.approx(o_first.t_stat[0], abs=1e-10)
        assert o_again.p_value[0] == pytest.approx(o_first.p_value[0], abs=1e-10)
        assert o_again.conf_int_lower[0] == pytest.approx(o_first.conf_int_lower[0], abs=1e-10)
        assert o_again.conf_int_upper[0] == pytest.approx(o_first.conf_int_upper[0], abs=1e-10)
        # The stored headline is a relay, untouched by the recomputes.
        assert res.aggregate("simple").att[0] == res.att
        assert res.aggregate("simple").se[0] == res.se

    def test_to_dataframe_group_time(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        df = res.to_dataframe("group_time")
        assert "group" in df.columns
        assert "effect" in df.columns
        assert len(df) == len(res.group_time_effects)

    def test_repr(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        r = repr(res)
        assert "TripleDifferenceResults" in r
        assert "design='staggered'" in r

    def test_significance_properties(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert isinstance(res.is_significant, bool)
        assert isinstance(res.significance_stars, str)


# ---------------------------------------------------------------------------
# Recovery tests (known DGP)
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffRecovery:
    def test_att_recovery(self, simple_data):
        """ATT should be within 2 SE of true effect."""
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert abs(res.att - 3.0) < 2 * res.se

    def test_null_effect(self, null_effect_data):
        """ATT should not be significant when true effect is 0."""
        est = TripleDifference()
        res = est.fit(
            null_effect_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert abs(res.att) < 2 * res.se

    def test_att_with_covariates(self, data_with_covariates):
        """ATT recovery with covariates."""
        est = TripleDifference(estimation_method="dr")
        res = est.fit(
            data_with_covariates,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
            covariates=["x1", "x2"],
        )
        assert abs(res.att - 3.0) < 2 * res.se


# ---------------------------------------------------------------------------
# Estimation method tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffMethods:
    @pytest.mark.parametrize("method", ["dr", "ipw", "reg"])
    def test_method_produces_finite_results(self, simple_data, method):
        est = TripleDifference(estimation_method=method)
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert res.se > 0


# ---------------------------------------------------------------------------
# Event study tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffEventStudy:
    def test_event_study_aggregation(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        # Not stored at fit time; produced post-fit by aggregate().
        assert res.event_study_effects is None
        es = res.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        n_horizons = len(es.event_time)
        assert n_horizons > 0
        for arr in (es.att, es.se, es.t_stat, es.p_value, es.conf_int_lower, es.conf_int_upper):
            assert len(arr) == n_horizons
        estimated = ~es.is_reference
        assert np.all(np.isfinite(es.att[estimated]))
        assert np.all(es.se[estimated] > 0)
        assert es.cband_crit_value is None  # analytical fit: no simultaneous band

    def test_pretreatment_near_zero(self, simple_data):
        """Pre-treatment event study effects should be near zero."""
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        es = res.aggregate("event_study")
        pre = (es.event_time < 0) & ~es.is_reference
        assert pre.any()
        for e, att, se in zip(es.event_time[pre], es.att[pre], es.se[pre]):
            # Pre-treatment effects within 3 SE of zero
            assert abs(att) < 3 * se, f"Pre-treatment effect at e={e} is {att:.3f} (SE={se:.3f})"

    def test_posttreatment_positive(self, simple_data):
        """Post-treatment event study effects should be positive."""
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        es = res.aggregate("event_study")
        post = (es.event_time >= 0) & ~es.is_reference
        assert post.any()
        assert np.all(es.att[post] > 0)

    def test_event_study_dataframe(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        es = res.aggregate("event_study")
        df = es.to_dataframe()
        for col in ("event_time", "att", "se", "conf_int_lower", "conf_int_upper", "is_reference"):
            assert col in df.columns
        assert len(df) == len(es.event_time)
        np.testing.assert_allclose(df["att"].to_numpy(), es.att, rtol=0, atol=0)
        # The fit-time event-study table is gone from the results container.
        with pytest.raises(ValueError, match="not stored on 4.0 fits"):
            res.to_dataframe("event_study")

    def test_event_study_and_group_both_available(self, simple_data):
        # The former aggregate='all' fit: both surfaces come from the same result.
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        es = res.aggregate("event_study")
        g = res.aggregate("group")
        assert len(es.event_time) > 0
        assert len(g.att) == len(res.groups)
        # Neither recompute populates the (None) fit-time carrier fields.
        assert res.event_study_effects is None
        assert res.group_effects is None

    def test_aggregate_group(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert res.group_effects is None  # not stored at fit time
        g = res.aggregate("group")
        assert isinstance(g, AggregationResult)
        assert len(g.att) > 0
        assert set(g.label) == set(res.groups)
        assert np.all(np.isfinite(g.att))
        assert np.all(g.se > 0)
        assert np.all(g.conf_int_lower < g.att) and np.all(g.att < g.conf_int_upper)
        df = g.to_dataframe()
        assert "label" in df.columns and "att" in df.columns
        assert len(df) == len(res.groups)
        with pytest.raises(ValueError, match="not stored on 4.0 fits"):
            res.to_dataframe("group")


# ---------------------------------------------------------------------------
# GMM combination tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffGMM:
    def test_gmm_weights_sum_to_one(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        for (g, t), weights in res.gmm_weights.items():
            w_sum = sum(weights.values())
            assert abs(w_sum - 1.0) < 1e-10, f"GMM weights for (g={g}, t={t}) sum to {w_sum}"

    def test_comparison_group_counts(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        for (g, t), k in res.comparison_group_counts.items():
            assert k >= 1

    def test_single_comparison_group_weight_is_one(self):
        """With only one valid comparison group, GMM weight should be 1."""
        data = generate_staggered_ddd_data(
            n_units=100,
            cohort_periods=[3],
            never_enabled_frac=0.3,
            seed=77,
        )
        est = TripleDifference()
        res = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        for (g, t), weights in res.gmm_weights.items():
            if len(weights) == 1:
                w = list(weights.values())[0]
                assert abs(w - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Bootstrap tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffBootstrap:
    def test_bootstrap_runs(self, simple_data, ci_params):
        n_boot = ci_params.bootstrap(199)
        est = TripleDifference(n_bootstrap=n_boot, seed=42)
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert res.bootstrap_results is not None
        assert res.bootstrap_results.n_bootstrap == n_boot
        assert res.inference_method == "bootstrap"
        assert res.n_bootstrap == n_boot
        assert res.to_dict()["n_bootstrap"] == n_boot

    def test_bootstrap_with_event_study(self, simple_data, ci_params):
        n_boot = ci_params.bootstrap(199)
        est = TripleDifference(n_bootstrap=n_boot, seed=42)
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert res.bootstrap_results is not None
        assert res.cband_crit_value is None  # no fit-time event-study surface
        # aggregate('event_study') replays the fit-time multiplier bootstrap.
        es = res.aggregate("event_study")
        assert es.cband_crit_value is not None
        assert es.cband_crit_value > 0
        assert es.cband_lower is not None and es.cband_upper is not None
        estimated = ~es.is_reference
        assert np.all(np.isfinite(es.cband_lower[estimated]))
        assert np.all(es.cband_lower[estimated] <= es.cband_upper[estimated])
        # The replay is deterministic from the retained RNG state.
        es_again = res.aggregate("event_study")
        assert es_again.cband_crit_value == es.cband_crit_value
        np.testing.assert_allclose(es_again.se, es.se, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffEdgeCases:
    def test_nonbinary_eligibility_raises(self, simple_data):
        bad_data = simple_data.copy()
        bad_data.loc[0, "eligibility"] = 2
        est = TripleDifference()
        with pytest.raises(ValueError, match="binary"):
            est.fit(
                bad_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
            )

    def test_varying_eligibility_raises(self):
        data = generate_staggered_ddd_data(n_units=50, seed=1)
        # Make eligibility vary within a unit
        data.loc[data["unit"] == 0, "eligibility"] = [0, 1, 0, 1, 0, 1, 0, 1]
        est = TripleDifference()
        with pytest.raises(ValueError, match="time-invariant"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
            )

    def test_missing_column_raises(self, simple_data):
        est = TripleDifference()
        with pytest.raises(ValueError, match="Missing columns"):
            est.fit(
                simple_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="nonexistent",
                partition="eligibility",
            )

    def test_inf_first_treat_works(self):
        """Never-enabled units encoded as inf should be recoded to 0, and the
        recoding must surface a UserWarning with the affected row count
        (axis-E silent coercion, mirroring the StaggeredDiD behavior)."""
        data = generate_staggered_ddd_data(n_units=100, seed=33)
        data["first_treat"] = data["first_treat"].astype(float)
        inf_mask = data["first_treat"] == 0
        n_inf_rows = int(inf_mask.sum())
        data.loc[inf_mask, "first_treat"] = np.inf
        est = TripleDifference()

        with pytest.warns(
            UserWarning,
            match=rf"{n_inf_rows} row\(s\) have first_treat=inf; recoding to 0",
        ):
            res = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
            )
        assert np.isfinite(res.att)

    def test_survey_design_invalid_type_raises(self, simple_data):
        est = TripleDifference()
        with pytest.raises(TypeError, match="SurveyDesign"):
            est.fit(
                simple_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
                survey_design="something",
            )

    def test_invalid_aggregate_raises(self, simple_data):
        est = TripleDifference()
        # 4.0: fit() no longer takes aggregate= (or balance_e=) at all.
        with pytest.raises(TypeError, match="aggregate"):
            est.fit(
                simple_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
                aggregate="invalid",
            )
        with pytest.raises(TypeError, match="balance_e"):
            est.fit(
                simple_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
                balance_e=1,
            )
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            res.aggregate("invalid")

    def test_base_period_universal(self, simple_data):
        est = TripleDifference(base_period="universal")
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        assert np.isfinite(res.att)
        assert abs(res.att - 3.0) < 2 * res.se

    def test_to_dict(self, simple_data):
        est = TripleDifference()
        res = est.fit(
            simple_data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
        )
        d = res.to_dict()
        for key in (
            "att",
            "se",
            "t_stat",
            "p_value",
            "conf_int_lower",
            "conf_int_upper",
            "n_obs",
            "estimation_method",
            "design",
            "n_groups",
            "n_periods",
            "groups",
            "time_periods",
        ):
            assert key in d
        assert d["att"] == res.att
        assert d["se"] == res.se
        assert d["design"] == "staggered"
        assert d["n_groups"] == len(res.groups)
        # Canonical names only: no 3.x overall_* keys, no fit-time surface keys.
        assert not any(k.startswith("overall_") for k in d)
        assert "event_study_effects" not in d
        assert "group_effects" not in d
        assert "n_bootstrap" not in d  # analytical fit


# ---------------------------------------------------------------------------
# Regression tests for specific bug fixes
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffRegressions:
    def test_base_period_outside_panel_warns(self):
        """Cohort with base period before observed panel should warn, not crash."""
        # Cohort g=2 with anticipation=1 needs base_period = g-1-1 = 0,
        # but periods start at 1. Should warn and skip that cell.
        data = generate_staggered_ddd_data(
            n_units=100,
            n_periods=4,
            cohort_periods=[2, 4],
            seed=77,
        )
        est = TripleDifference(anticipation=1)
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            res = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
            )
        base_period_warnings = [w for w in caught if "outside the observed panel" in str(w.message)]
        assert len(base_period_warnings) > 0, "Expected warning about base period"
        assert np.isfinite(res.att)

    def test_empty_subgroup_warns(self):
        """Data where one (S,Q) cell is empty should warn, not crash."""
        data = generate_staggered_ddd_data(
            n_units=100,
            cohort_periods=[4, 6],
            seed=88,
        )
        # Remove all ineligible units from cohort 6 to make (S=6,Q=0) empty
        mask = ~((data["first_treat"] == 6) & (data["eligibility"] == 0))
        data = data[mask].reset_index(drop=True)
        est = TripleDifference()
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            res = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
            )
        subgroup_warnings = [w for w in caught if "Empty subgroup" in str(w.message)]
        assert len(subgroup_warnings) > 0, "Expected warning about empty subgroup"
        assert np.isfinite(res.att)

    def test_collinear_covariates_cached_ps_finite(self):
        """Collinear covariates with PS cache reuse should produce finite results."""
        data = generate_staggered_ddd_data(
            n_units=200,
            treatment_effect=3.0,
            add_covariates=True,
            seed=55,
        )
        # Add a perfectly collinear covariate (x3 = 2*x1)
        data["x3"] = 2.0 * data["x1"]
        est = TripleDifference(
            estimation_method="dr",
            rank_deficient_action="warn",
        )
        import warnings as _w

        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            res = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
                covariates=["x1", "x2", "x3"],
            )
        # All group-time effects should be finite despite collinearity
        for (g, t), eff in res.group_time_effects.items():
            assert np.isfinite(eff["effect"]), f"Non-finite ATT at (g={g},t={t})"
            assert np.isfinite(eff["se"]), f"Non-finite SE at (g={g},t={t})"


# ---------------------------------------------------------------------------
# Silent-failure audit PR #9: finding #17 — OR influence-function solve
# fell back to np.linalg.lstsq silently when X'WX was rank deficient. Now
# tracked at _compute_did_panel() and aggregated into one fit-level warning.
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffORSolveFallback:
    def test_collinear_covariates_emit_lstsq_fallback_warning(self):
        """Perfectly collinear covariates should trigger the aggregate OR
        rank-guard warning (default rank_deficient_action='warn'; suppressed
        under 'silent')."""
        data = generate_staggered_ddd_data(
            n_units=200,
            treatment_effect=3.0,
            add_covariates=True,
            seed=55,
        )
        # x3 ≡ 2·x1 makes covX = [intercept, x1, x2, x3] rank-deficient
        # per pair, so XpX in _compute_did_panel() is near-singular.
        data["x3"] = 2.0 * data["x1"]
        est = TripleDifference(estimation_method="dr")  # default "warn"
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            res = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
                covariates=["x1", "x2", "x3"],
            )
        or_warnings = [
            w for w in caught if "outcome-regression influence-function step" in str(w.message)
        ]
        assert len(or_warnings) == 1, (
            f"Expected exactly one aggregate OR rank-guard warning, " f"got {len(or_warnings)}."
        )
        msg = str(or_warnings[0].message)
        assert "(g, g_c, t) pair(s)" in msg
        assert "rank-guarded inverse" in msg
        # Point estimates should still be finite (rank-guard truncation).
        assert np.isfinite(res.att)

    def test_collinear_covariates_emit_ps_hessian_warning(self):
        """Collinear propensity-score covariates should trigger the aggregate
        PS-Hessian rank-guard warning under IPW/DR inference (default
        rank_deficient_action='warn'; suppressed under 'silent'). Sibling of the
        OR-side finding; surfaced by PR #334 CI review."""
        data = generate_staggered_ddd_data(
            n_units=200,
            treatment_effect=3.0,
            add_covariates=True,
            seed=55,
        )
        data["x3"] = 2.0 * data["x1"]
        est = TripleDifference(
            estimation_method="ipw",  # exercises PS path, skips OR projection
        )
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
                covariates=["x1", "x2", "x3"],
            )
        ps_warnings = [w for w in caught if "propensity-score Hessian" in str(w.message)]
        assert len(ps_warnings) == 1, (
            f"Expected exactly one aggregate PS-Hessian rank-guard "
            f"warning under IPW, got {len(ps_warnings)}: "
            f"{[str(w.message) for w in ps_warnings]}"
        )
        msg = str(ps_warnings[0].message)
        assert "(g, g_c, t) pair(s)" in msg
        assert "rank-guarded inverse" in msg
        assert "IPW/DR" in msg

    def test_well_conditioned_covariates_emit_no_lstsq_warning(self):
        """Clean, well-conditioned covariates should NOT trigger either
        OR or PS-Hessian aggregate warning — regression-safety for the
        happy path."""
        data = generate_staggered_ddd_data(
            n_units=300,
            treatment_effect=3.0,
            add_covariates=True,
            seed=42,
        )
        est = TripleDifference(estimation_method="dr")
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
                covariates=["x1", "x2"],
            )
        lstsq_warnings = [w for w in caught if "Rank-deficient X'WX" in str(w.message)]
        assert lstsq_warnings == [], (
            f"Unexpected lstsq-fallback warning on clean covariates: "
            f"{[str(w.message) for w in lstsq_warnings]}"
        )

    def test_no_covariates_no_warning(self, simple_data):
        """Without covariates both OR and PS paths are skipped, so no
        aggregate warning should be emitted."""
        est = TripleDifference(estimation_method="reg")
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            est.fit(
                simple_data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                partition="eligibility",
            )
        lstsq_warnings = [w for w in caught if "Rank-deficient X'WX" in str(w.message)]
        assert lstsq_warnings == []


@pytest.fixture(scope="module")
def alpha_fitted():
    data = generate_staggered_ddd_data(n_units=300, treatment_effect=3.0, seed=42)
    return TripleDifference().fit(
        data,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        partition="eligibility",
    )


class TestSummaryAlphaContract:
    """summary(alpha=...) never recomputes stored inference.

    Family-wide guard (results_base._require_fit_alpha) on the unified
    TripleDifferenceResults container's staggered summary.
    """

    @pytest.mark.parametrize("bad_alpha", [0.10, 0.0])
    def test_summary_rejects_non_fit_alpha(self, alpha_fitted, bad_alpha):
        with pytest.raises(ValueError, match="never recomputes"):
            alpha_fitted.summary(alpha=bad_alpha)

    def test_summary_accepts_fit_alpha(self, alpha_fitted):
        assert alpha_fitted.summary(alpha=alpha_fitted.alpha) == alpha_fitted.summary()
