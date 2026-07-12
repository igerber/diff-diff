"""Tests for StaggeredTripleDifference estimator."""

import numpy as np
import pytest

from diff_diff import (
    SDDD,
    StaggeredTripleDifference,
    StaggeredTripleDiffResults,
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
        est = StaggeredTripleDifference()
        params = est.get_params()
        assert params["estimation_method"] == "dr"
        assert params["alpha"] == 0.05
        assert params["anticipation"] == 0
        assert params["base_period"] == "varying"
        assert params["n_bootstrap"] == 0
        assert params["pscore_trim"] == 0.01

    def test_alias(self):
        assert SDDD is StaggeredTripleDifference

    def test_set_params(self):
        est = StaggeredTripleDifference()
        est.set_params(estimation_method="ipw", alpha=0.10)
        assert est.estimation_method == "ipw"
        assert est.alpha == 0.10

    def test_set_params_updates_bootstrap_weights(self):
        est = StaggeredTripleDifference()
        est.set_params(bootstrap_weights="mammen")
        assert est.bootstrap_weights == "mammen"

    def test_invalid_estimation_method(self):
        with pytest.raises(ValueError, match="estimation_method"):
            StaggeredTripleDifference(estimation_method="ols")

    def test_invalid_pscore_trim(self):
        with pytest.raises(ValueError, match="pscore_trim"):
            StaggeredTripleDifference(pscore_trim=0.6)

    def test_invalid_base_period(self):
        with pytest.raises(ValueError, match="base_period"):
            StaggeredTripleDifference(base_period="fixed")

    def test_set_params_unknown(self):
        est = StaggeredTripleDifference()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_param=42)


# ---------------------------------------------------------------------------
# Basic fit tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffBasic:
    def test_fit_returns_results(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert isinstance(res, StaggeredTripleDiffResults)
        assert est.is_fitted_

    def test_results_structure(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert isinstance(res.overall_att, float)
        assert isinstance(res.overall_se, float)
        assert res.overall_se > 0
        assert isinstance(res.overall_conf_int, tuple)
        assert len(res.overall_conf_int) == 2
        assert len(res.groups) > 0
        assert len(res.time_periods) > 0
        assert res.n_obs > 0
        assert res.n_treated_units > 0
        assert res.n_control_units > 0
        assert res.n_eligible > 0
        assert res.n_ineligible > 0

    def test_group_time_effects_populated(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
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
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        summary = res.summary()
        assert "Staggered Triple Difference" in summary
        assert "ATT" in summary

    def test_overall_att_es_result_surface(self, simple_data):
        """Opt-in Eq. 4.14 overall (overall_att_es) is exposed on the result surface
        (attribute, summary(), to_dict()) under aggregate='event_study', and is None
        for the default (no event-study) fit."""
        res = StaggeredTripleDifference().fit(
            simple_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        es_fields = (
            "overall_att_es",
            "overall_se_es",
            "overall_t_stat_es",
            "overall_p_value_es",
            "overall_conf_int_es",
        )
        assert res.overall_att_es is not None and np.isfinite(res.overall_att_es)
        for f in es_fields:
            assert getattr(res, f) is not None
        assert "Eq. 4.14" in res.summary()
        d = res.to_dict()
        for f in es_fields:
            assert f in d
        # Default fit (no event-study aggregation): fields are None and absent from to_dict.
        res_default = StaggeredTripleDifference().fit(
            simple_data, "outcome", "unit", "period", "first_treat", "eligibility"
        )
        assert res_default.overall_att_es is None
        assert "overall_att_es" not in res_default.to_dict()

    def test_overall_att_es_aggregate_all_matches_event_study(self, simple_data):
        """Eq. 4.14 overall (overall_att_es) is populated under aggregate='all' (not only
        'event_study') and matches the 'event_study' path bit-for-bit on the same data.

        Regression guard for the documented public contract that BOTH surfaces expose
        overall_att_es: aggregate='all' additionally computes the per-cohort group
        aggregation, and must neither drop nor perturb the event-study-average overall.
        """
        res_all = StaggeredTripleDifference().fit(
            simple_data, "outcome", "unit", "period", "first_treat", "eligibility", aggregate="all"
        )
        res_es = StaggeredTripleDifference().fit(
            simple_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        # Populated (not None / finite) on the 'all' surface.
        assert res_all.overall_att_es is not None and np.isfinite(res_all.overall_att_es)
        assert res_all.overall_se_es is not None and np.isfinite(res_all.overall_se_es)
        # Identical point estimate AND inference vs the event_study path: same data, same
        # analytical influence-function SE (no bootstrap), so bit-for-bit close.
        assert res_all.overall_att_es == pytest.approx(res_es.overall_att_es, abs=1e-10)
        assert res_all.overall_se_es == pytest.approx(res_es.overall_se_es, abs=1e-10)
        assert res_all.overall_t_stat_es == pytest.approx(res_es.overall_t_stat_es, abs=1e-10)
        assert res_all.overall_p_value_es == pytest.approx(res_es.overall_p_value_es, abs=1e-10)
        ci_all, ci_es = res_all.overall_conf_int_es, res_es.overall_conf_int_es
        assert ci_all is not None and ci_es is not None
        assert ci_all[0] == pytest.approx(ci_es[0], abs=1e-10)
        assert ci_all[1] == pytest.approx(ci_es[1], abs=1e-10)

    def test_to_dataframe_group_time(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        df = res.to_dataframe("group_time")
        assert "group" in df.columns
        assert "effect" in df.columns
        assert len(df) == len(res.group_time_effects)

    def test_repr(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        r = repr(res)
        assert "StaggeredTripleDiffResults" in r

    def test_significance_properties(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert isinstance(res.is_significant, bool)
        assert isinstance(res.significance_stars, str)


# ---------------------------------------------------------------------------
# Recovery tests (known DGP)
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffRecovery:
    def test_att_recovery(self, simple_data):
        """ATT should be within 2 SE of true effect."""
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert abs(res.overall_att - 3.0) < 2 * res.overall_se

    def test_null_effect(self, null_effect_data):
        """ATT should not be significant when true effect is 0."""
        est = StaggeredTripleDifference()
        res = est.fit(null_effect_data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert abs(res.overall_att) < 2 * res.overall_se

    def test_att_with_covariates(self, data_with_covariates):
        """ATT recovery with covariates."""
        est = StaggeredTripleDifference(estimation_method="dr")
        res = est.fit(
            data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            covariates=["x1", "x2"],
        )
        assert abs(res.overall_att - 3.0) < 2 * res.overall_se


# ---------------------------------------------------------------------------
# Estimation method tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffMethods:
    @pytest.mark.parametrize("method", ["dr", "ipw", "reg"])
    def test_method_produces_finite_results(self, simple_data, method):
        est = StaggeredTripleDifference(estimation_method=method)
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert res.overall_se > 0


# ---------------------------------------------------------------------------
# Event study tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffEventStudy:
    def test_event_study_aggregation(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(
            simple_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        assert len(res.event_study_effects) > 0

    def test_pretreatment_near_zero(self, simple_data):
        """Pre-treatment event study effects should be near zero."""
        est = StaggeredTripleDifference()
        res = est.fit(
            simple_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        for e, eff in res.event_study_effects.items():
            if e < 0:
                # Pre-treatment effects within 3 SE of zero
                assert abs(eff["effect"]) < 3 * eff["se"], (
                    f"Pre-treatment effect at e={e} is {eff['effect']:.3f} " f"(SE={eff['se']:.3f})"
                )

    def test_posttreatment_positive(self, simple_data):
        """Post-treatment event study effects should be positive."""
        est = StaggeredTripleDifference()
        res = est.fit(
            simple_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        for e, eff in res.event_study_effects.items():
            if e >= 0:
                assert eff["effect"] > 0

    def test_event_study_dataframe(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(
            simple_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        df = res.to_dataframe("event_study")
        assert "relative_period" in df.columns
        assert "effect" in df.columns

    def test_aggregate_all(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(
            simple_data, "outcome", "unit", "period", "first_treat", "eligibility", aggregate="all"
        )
        assert res.event_study_effects is not None
        assert res.group_effects is not None

    def test_aggregate_group(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(
            simple_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="group",
        )
        assert res.group_effects is not None
        assert len(res.group_effects) > 0


# ---------------------------------------------------------------------------
# GMM combination tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffGMM:
    def test_gmm_weights_sum_to_one(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        for (g, t), weights in res.gmm_weights.items():
            w_sum = sum(weights.values())
            assert abs(w_sum - 1.0) < 1e-10, f"GMM weights for (g={g}, t={t}) sum to {w_sum}"

    def test_comparison_group_counts(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
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
        est = StaggeredTripleDifference()
        res = est.fit(data, "outcome", "unit", "period", "first_treat", "eligibility")
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
        est = StaggeredTripleDifference(n_bootstrap=n_boot, seed=42)
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert res.bootstrap_results is not None
        assert res.bootstrap_results.n_bootstrap == n_boot

    def test_bootstrap_with_event_study(self, simple_data, ci_params):
        n_boot = ci_params.bootstrap(199)
        est = StaggeredTripleDifference(n_bootstrap=n_boot, seed=42)
        res = est.fit(
            simple_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        assert res.bootstrap_results is not None
        if res.cband_crit_value is not None:
            assert res.cband_crit_value > 0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestStaggeredTripleDiffEdgeCases:
    def test_nonbinary_eligibility_raises(self, simple_data):
        bad_data = simple_data.copy()
        bad_data.loc[0, "eligibility"] = 2
        est = StaggeredTripleDifference()
        with pytest.raises(ValueError, match="binary"):
            est.fit(bad_data, "outcome", "unit", "period", "first_treat", "eligibility")

    def test_varying_eligibility_raises(self):
        data = generate_staggered_ddd_data(n_units=50, seed=1)
        # Make eligibility vary within a unit
        data.loc[data["unit"] == 0, "eligibility"] = [0, 1, 0, 1, 0, 1, 0, 1]
        est = StaggeredTripleDifference()
        with pytest.raises(ValueError, match="time-invariant"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "eligibility")

    def test_missing_column_raises(self, simple_data):
        est = StaggeredTripleDifference()
        with pytest.raises(ValueError, match="Missing columns"):
            est.fit(simple_data, "outcome", "unit", "period", "nonexistent", "eligibility")

    def test_inf_first_treat_works(self):
        """Never-enabled units encoded as inf should be recoded to 0, and the
        recoding must surface a UserWarning with the affected row count
        (axis-E silent coercion, mirroring the StaggeredDiD behavior)."""
        data = generate_staggered_ddd_data(n_units=100, seed=33)
        data["first_treat"] = data["first_treat"].astype(float)
        inf_mask = data["first_treat"] == 0
        n_inf_rows = int(inf_mask.sum())
        data.loc[inf_mask, "first_treat"] = np.inf
        est = StaggeredTripleDifference()

        with pytest.warns(
            UserWarning,
            match=rf"{n_inf_rows} row\(s\) have first_treat=inf; recoding to 0",
        ):
            res = est.fit(data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert np.isfinite(res.overall_att)

    def test_survey_design_invalid_type_raises(self, simple_data):
        est = StaggeredTripleDifference()
        with pytest.raises(TypeError, match="SurveyDesign"):
            est.fit(
                simple_data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                survey_design="something",
            )

    def test_invalid_aggregate_raises(self, simple_data):
        est = StaggeredTripleDifference()
        with pytest.raises(ValueError, match="aggregate"):
            est.fit(
                simple_data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                aggregate="invalid",
            )

    def test_base_period_universal(self, simple_data):
        est = StaggeredTripleDifference(base_period="universal")
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        assert np.isfinite(res.overall_att)
        assert abs(res.overall_att - 3.0) < 2 * res.overall_se

    def test_to_dict(self, simple_data):
        est = StaggeredTripleDifference()
        res = est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        d = res.to_dict()
        assert "overall_att" in d
        assert "n_obs" in d
        assert "estimation_method" in d


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
        est = StaggeredTripleDifference(anticipation=1)
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            res = est.fit(data, "outcome", "unit", "period", "first_treat", "eligibility")
        base_period_warnings = [w for w in caught if "outside the observed panel" in str(w.message)]
        assert len(base_period_warnings) > 0, "Expected warning about base period"
        assert np.isfinite(res.overall_att)

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
        est = StaggeredTripleDifference()
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            res = est.fit(data, "outcome", "unit", "period", "first_treat", "eligibility")
        subgroup_warnings = [w for w in caught if "Empty subgroup" in str(w.message)]
        assert len(subgroup_warnings) > 0, "Expected warning about empty subgroup"
        assert np.isfinite(res.overall_att)

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
        est = StaggeredTripleDifference(
            estimation_method="dr",
            rank_deficient_action="warn",
        )
        import warnings as _w

        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            res = est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
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
        est = StaggeredTripleDifference(estimation_method="dr")  # default "warn"
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            res = est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
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
        assert np.isfinite(res.overall_att)

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
        est = StaggeredTripleDifference(
            estimation_method="ipw",  # exercises PS path, skips OR projection
        )
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
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
        est = StaggeredTripleDifference(estimation_method="dr")
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
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
        est = StaggeredTripleDifference(estimation_method="reg")
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            est.fit(simple_data, "outcome", "unit", "period", "first_treat", "eligibility")
        lstsq_warnings = [w for w in caught if "Rank-deficient X'WX" in str(w.message)]
        assert lstsq_warnings == []
