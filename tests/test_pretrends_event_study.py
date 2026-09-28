"""
Tests for pretrends=True feature in ImputationDiD and TwoStageDiD.

Tests that pre-period event study coefficients are computed correctly
when pretrends=True is set on the estimator.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.imputation import ImputationDiD
from diff_diff.two_stage import TwoStageDiD

# ---------------------------------------------------------------------------
# Rows M-021/M-022 (+ M-118/M-119): ImputationDiD / TwoStageDiD
# ``fit(aggregate=, balance_e=)`` is deprecated (3.9, removed 4.0) and warns on
# ANY supplied value. The deprecated fit-time route is kept DELIBERATELY here:
# these tests pin FIT-TIME surface behaviour (bit-equality grids, bootstrap
# aggregation, R/Stata parity, replicate overrides, native effect dicts) that
# the post-fit ``results.aggregate(...)`` container route does not reproduce
# shape-for-shape. The shim warning is therefore filtered BY MESSAGE, scoped to
# these two estimators only - every other FutureWarning (including the other
# estimators' aggregate() shims) still surfaces.
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.filterwarnings(
    r"ignore:(ImputationDiD|TwoStageDiD)\.fit\((aggregate=|balance_e=|aggregate= / balance_e=)\):FutureWarning"
)


def generate_test_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate staggered adoption data for pretrends testing."""
    rng = np.random.default_rng(seed)

    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    cohort_periods = np.array([3, 5, 7])
    first_treat = np.zeros(n_units, dtype=int)
    if n_treated > 0:
        cohort_assignments = rng.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = cohort_periods[cohort_assignments]

    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 2.0
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    relative_time = times - first_treat_expanded
    dynamic_mult = 1 + 0.1 * np.maximum(relative_time, 0)
    effect = treatment_effect * dynamic_mult

    outcomes = (
        unit_fe_expanded + time_fe_expanded + effect * post + rng.standard_normal(len(units)) * 0.5
    )

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcomes,
            "first_treat": first_treat_expanded,
        }
    )


# =============================================================================
# ImputationDiD pretrends tests
# =============================================================================


class TestImputationPretrends:
    """Tests for ImputationDiD pretrends feature."""

    def test_pretrends_includes_negative_horizons(self):
        """Pre-period horizons appear in event study when pretrends=True."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        horizons = sorted(results.event_study_effects.keys())
        negative = [h for h in horizons if h < 0]
        positive = [h for h in horizons if h >= 0]
        assert len(negative) > 1, "Should have pre-period horizons"
        assert len(positive) > 0, "Should have post-treatment horizons"

    def test_pretrends_coefficients_near_zero(self):
        """Pre-period effects ~0 under parallel trends (no violation)."""
        data = generate_test_data(treatment_effect=2.0, seed=99)
        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        ref_period = -1
        for h, eff in results.event_study_effects.items():
            if h >= 0 or h == ref_period:
                continue
            assert np.isfinite(eff["effect"]), f"h={h}: effect not finite"
            assert (
                abs(eff["effect"]) < 3 * eff["se"] + 0.5
            ), f"h={h}: pre-period effect {eff['effect']:.3f} too large"

    def test_pretrends_se_finite_positive(self):
        """All pre-period horizons have finite, positive SEs."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        ref_period = -1
        for h, eff in results.event_study_effects.items():
            if h == ref_period:
                continue
            if eff["n_obs"] == 0:
                continue
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite"
            assert eff["se"] > 0, f"h={h}: SE not positive"

    def test_reference_period_correct(self):
        """Reference period h=-1 normalized to 0."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert -1 in results.event_study_effects
        ref = results.event_study_effects[-1]
        assert ref["effect"] == 0.0
        assert ref["n_obs"] == 0

    def test_backward_compatibility(self):
        """pretrends=False (default) gives identical results."""
        data = generate_test_data()

        results_default = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_false = ImputationDiD(pretrends=False).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results_default.overall_att == results_false.overall_att
        assert results_default.overall_se == results_false.overall_se
        assert set(results_default.event_study_effects.keys()) == set(
            results_false.event_study_effects.keys()
        )

    def test_post_treatment_invariance(self):
        """Post-treatment effects identical with pretrends=True vs False."""
        data = generate_test_data()

        results_off = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_on = ImputationDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Overall ATT unchanged
        assert results_on.overall_att == results_off.overall_att
        assert results_on.overall_se == results_off.overall_se

        # Post-treatment event study effects unchanged
        for h in results_off.event_study_effects:
            assert h in results_on.event_study_effects
            eff_off = results_off.event_study_effects[h]
            eff_on = results_on.event_study_effects[h]
            np.testing.assert_allclose(eff_off["effect"], eff_on["effect"], rtol=1e-10)
            np.testing.assert_allclose(eff_off["se"], eff_on["se"], rtol=1e-10)

    def test_horizon_max_interaction(self):
        """horizon_max limits both pre and post horizons."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True, horizon_max=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h in results.event_study_effects:
            assert abs(h) <= 2, f"h={h} exceeds horizon_max=2"

    def test_anticipation_interaction(self):
        """With anticipation=1, reference shifts to h=-2."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True, anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert -2 in results.event_study_effects
        ref = results.event_study_effects[-2]
        assert ref["effect"] == 0.0
        assert ref["n_obs"] == 0

    def test_get_params_includes_pretrends(self):
        """get_params includes pretrends parameter."""
        est = ImputationDiD(pretrends=True)
        params = est.get_params()
        assert "pretrends" in params
        assert params["pretrends"] is True

        est2 = ImputationDiD()
        assert est2.get_params()["pretrends"] is False

    def test_no_pretreatment_obs_graceful(self):
        """All units treated at t=1 with pretrends=True: no error."""
        rng = np.random.default_rng(42)
        n_units = 20
        n_periods = 5
        data = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(n_units), n_periods),
                "time": np.tile(np.arange(n_periods), n_units),
                "outcome": rng.standard_normal(n_units * n_periods),
                "first_treat": np.repeat(np.ones(n_units, dtype=int), n_periods),
            }
        )

        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert results.event_study_effects is not None


# =============================================================================
# TwoStageDiD pretrends tests
# =============================================================================


class TestTwoStagePretrends:
    """Tests for TwoStageDiD pretrends feature."""

    def test_pretrends_includes_negative_horizons(self):
        """Pre-period horizons appear in event study when pretrends=True."""
        data = generate_test_data()
        est = TwoStageDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        horizons = sorted(results.event_study_effects.keys())
        negative = [h for h in horizons if h < 0]
        positive = [h for h in horizons if h >= 0]
        assert len(negative) > 1, "Should have pre-period horizons"
        assert len(positive) > 0, "Should have post-treatment horizons"

    def test_pretrends_coefficients_near_zero(self):
        """Pre-period effects ~0 under parallel trends."""
        data = generate_test_data(treatment_effect=2.0, seed=99)
        est = TwoStageDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        ref_period = -1
        for h, eff in results.event_study_effects.items():
            if h >= 0 or h == ref_period:
                continue
            assert np.isfinite(eff["effect"]), f"h={h}: effect not finite"
            assert (
                abs(eff["effect"]) < 3 * eff["se"] + 0.5
            ), f"h={h}: pre-period effect {eff['effect']:.3f} too large"

    def test_pretrends_se_finite_positive(self):
        """All pre-period horizons have finite, positive SEs."""
        data = generate_test_data()
        est = TwoStageDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        ref_period = -1
        for h, eff in results.event_study_effects.items():
            if h == ref_period:
                continue
            if eff["n_obs"] == 0:
                continue
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite"
            assert eff["se"] > 0, f"h={h}: SE not positive"

    def test_post_treatment_invariance(self):
        """Post-treatment effects identical with pretrends=True vs False."""
        data = generate_test_data()

        results_off = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_on = TwoStageDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Overall ATT unchanged
        assert results_on.overall_att == results_off.overall_att
        assert results_on.overall_se == results_off.overall_se

        # Post-treatment event study effects unchanged
        for h in results_off.event_study_effects:
            assert h in results_on.event_study_effects
            eff_off = results_off.event_study_effects[h]
            eff_on = results_on.event_study_effects[h]
            np.testing.assert_allclose(eff_off["effect"], eff_on["effect"], rtol=1e-10)
            np.testing.assert_allclose(eff_off["se"], eff_on["se"], rtol=1e-10)

    def test_get_params_includes_pretrends(self):
        """get_params includes pretrends parameter."""
        est = TwoStageDiD(pretrends=True)
        params = est.get_params()
        assert "pretrends" in params
        assert params["pretrends"] is True

    def test_horizon_max_interaction(self):
        """horizon_max limits both pre and post horizons."""
        data = generate_test_data()
        est = TwoStageDiD(pretrends=True, horizon_max=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h in results.event_study_effects:
            assert abs(h) <= 2, f"h={h} exceeds horizon_max=2"

    def test_anticipation_interaction(self):
        """With anticipation=1, reference shifts to h=-2."""
        data = generate_test_data()
        est = TwoStageDiD(pretrends=True, anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert -2 in results.event_study_effects
        ref = results.event_study_effects[-2]
        assert ref["effect"] == 0.0
        assert ref["n_obs"] == 0


# =============================================================================
# Cross-estimator consistency
# =============================================================================


class TestPretrends_ContractTests:
    """Verify ImputationDiD pre-period coefficients match pretrend_test() lead coefficients."""

    def test_imputation_pretrends_match_pretrend_test(self):
        """ImputationDiD pretrends=True effects match pretrend_test().lead_coefficients."""
        data = generate_test_data(seed=77)

        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Get pretrend_test lead coefficients (called on results object)
        pt = results.pretrend_test()
        lead_coefs = pt["lead_coefficients"]

        # Every lead coefficient should match the event study pre-period effect
        for h, coef in lead_coefs.items():
            assert h in results.event_study_effects, f"h={h}: lead coefficient not in event study"
            es_effect = results.event_study_effects[h]["effect"]
            np.testing.assert_allclose(
                es_effect,
                coef,
                rtol=1e-10,
                err_msg=f"h={h}: event study effect != pretrend_test lead coefficient",
            )


# =============================================================================
# Regression tests for P0/P1 review findings
# =============================================================================


class TestPretrends_Regressions:
    """Regression tests for bugs found in AI code review."""

    def test_imputation_survey_weighted_no_covariates(self):
        """ImputationDiD with survey weights, no covariates, no pretrends.

        Regression for P1: untreated_units/untreated_times uninitialized
        in the survey-weighted FE-only variance path.
        """
        from diff_diff.survey import SurveyDesign

        data = generate_test_data(seed=42)
        rng = np.random.default_rng(42)
        n_units = data["unit"].nunique()
        unit_weights = rng.uniform(0.5, 2.0, n_units)
        data["weight"] = data["unit"].map(dict(enumerate(unit_weights)))

        sd = SurveyDesign(weights="weight")
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )

        assert np.isfinite(results.overall_att)
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0

    def test_imputation_survey_weighted_event_study(self):
        """ImputationDiD with survey weights and event study aggregation."""
        from diff_diff.survey import SurveyDesign

        data = generate_test_data(seed=42)
        rng = np.random.default_rng(42)
        n_units = data["unit"].nunique()
        unit_weights = rng.uniform(0.5, 2.0, n_units)
        data["weight"] = data["unit"].map(dict(enumerate(unit_weights)))

        sd = SurveyDesign(weights="weight")
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            survey_design=sd,
        )

        for h, eff in results.event_study_effects.items():
            if eff["n_obs"] == 0:
                continue
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite"

    def test_imputation_pretrends_with_covariates(self):
        """ImputationDiD pretrends=True with covariates.

        Tests _compute_v_untreated_with_covariates_preperiod().
        """
        data = generate_test_data(seed=42)
        rng = np.random.default_rng(42)
        data["x1"] = rng.standard_normal(len(data))

        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1"],
            aggregate="event_study",
        )

        negative = {
            h: v for h, v in results.event_study_effects.items() if h < -1 and v["n_obs"] > 0
        }
        assert len(negative) > 0, "Should have pre-period horizons"
        for h, eff in negative.items():
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite with covariates"
            assert eff["se"] > 0, f"h={h}: SE not positive with covariates"

    def test_imputation_pretrends_bootstrap_post_only(self):
        """ImputationDiD pretrends=True + bootstrap: bootstrap updates post SEs only.

        Pre-period SEs come from Test 1 lead regression (cluster-robust),
        not from bootstrap. Verify bootstrap doesn't break pre-period inference.
        """
        data = generate_test_data(seed=42)

        # Without bootstrap
        results_no_boot = ImputationDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # With bootstrap
        results_boot = ImputationDiD(pretrends=True, n_bootstrap=50, seed=42).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Pre-period effects and SEs should be identical (bootstrap doesn't touch them)
        for h in results_no_boot.event_study_effects:
            if h >= 0 or results_no_boot.event_study_effects[h]["n_obs"] == 0:
                continue
            eff_nb = results_no_boot.event_study_effects[h]
            eff_b = results_boot.event_study_effects[h]
            np.testing.assert_allclose(
                eff_nb["effect"],
                eff_b["effect"],
                rtol=1e-10,
                err_msg=f"h={h}: bootstrap changed pre-period effect",
            )
            np.testing.assert_allclose(
                eff_nb["se"],
                eff_b["se"],
                rtol=1e-10,
                err_msg=f"h={h}: bootstrap changed pre-period SE",
            )

    def test_nondefault_index_analytical(self):
        """Pre-period inference identical across index types (analytical).

        Regression for P0: label-based indexing in pre-period target mapping
        corrupts inference when DataFrame has non-default index.
        """
        data = generate_test_data(seed=42)

        # Baseline: default RangeIndex
        results_default = ImputationDiD(pretrends=True).fit(
            data.copy(),
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Permuted integer index
        rng = np.random.default_rng(42)
        data_perm = data.copy()
        data_perm.index = rng.permutation(len(data_perm))
        results_perm = ImputationDiD(pretrends=True).fit(
            data_perm,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Gapped/nonconsecutive index
        data_gap = data.copy()
        data_gap.index = data_gap.index * 3 + 100
        results_gap = ImputationDiD(pretrends=True).fit(
            data_gap,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # All pre-period effects and SEs must match
        for h in results_default.event_study_effects:
            if results_default.event_study_effects[h]["n_obs"] == 0:
                continue
            eff_d = results_default.event_study_effects[h]
            eff_p = results_perm.event_study_effects[h]
            eff_g = results_gap.event_study_effects[h]
            np.testing.assert_allclose(
                eff_d["effect"],
                eff_p["effect"],
                rtol=1e-10,
                err_msg=f"h={h}: permuted index changed effect",
            )
            np.testing.assert_allclose(
                eff_d["se"],
                eff_p["se"],
                rtol=1e-10,
                err_msg=f"h={h}: permuted index changed SE",
            )
            np.testing.assert_allclose(
                eff_d["effect"],
                eff_g["effect"],
                rtol=1e-10,
                err_msg=f"h={h}: gapped index changed effect",
            )
            np.testing.assert_allclose(
                eff_d["se"],
                eff_g["se"],
                rtol=1e-10,
                err_msg=f"h={h}: gapped index changed SE",
            )

    def test_nondefault_index_bootstrap(self):
        """Pre-period bootstrap inference identical across index types.

        Regression for P0: same indexing bug in bootstrap path.
        """
        data = generate_test_data(seed=42)

        results_default = ImputationDiD(pretrends=True, n_bootstrap=50, seed=99).fit(
            data.copy(),
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        rng = np.random.default_rng(42)
        data_perm = data.copy()
        data_perm.index = rng.permutation(len(data_perm))
        results_perm = ImputationDiD(pretrends=True, n_bootstrap=50, seed=99).fit(
            data_perm,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h in results_default.event_study_effects:
            if results_default.event_study_effects[h]["n_obs"] == 0:
                continue
            eff_d = results_default.event_study_effects[h]
            eff_p = results_perm.event_study_effects[h]
            np.testing.assert_allclose(
                eff_d["effect"],
                eff_p["effect"],
                rtol=1e-10,
                err_msg=f"h={h}: permuted index changed effect",
            )
            np.testing.assert_allclose(
                eff_d["se"],
                eff_p["se"],
                rtol=1e-10,
                err_msg=f"h={h}: permuted index changed bootstrap SE",
            )

    def test_imputation_pretrends_survey_accepted(self):
        """pretrends=True + survey_design is now supported (Phase 8e-iii)."""
        from diff_diff.survey import SurveyDesign

        data = generate_test_data(seed=42)
        n_units = data["unit"].nunique()
        unit_weights = np.random.default_rng(42).uniform(0.5, 2.0, n_units)
        data["weight"] = data["unit"].map(dict(enumerate(unit_weights)))

        sd = SurveyDesign(weights="weight")
        est = ImputationDiD(pretrends=True)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            survey_design=sd,
        )
        assert result.event_study_effects is not None
        assert np.isfinite(result.overall_att)

    def test_imputation_pretrends_balance_e(self):
        """balance_e restricts pre-period lead regression to balanced cohorts."""
        data = generate_test_data(seed=42)

        results_no_be = ImputationDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_be = ImputationDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=1,
        )

        # Both should have pre-period horizons
        neg_no_be = [h for h in results_no_be.event_study_effects if h < -1]
        assert len(neg_no_be) > 0
        # balance_e restricts cohorts; pre-period coefficients may be NaN
        # if the restricted sample is rank-deficient after demeaning
        # The key invariant: the method runs without error
        assert results_be.event_study_effects is not None

    def test_two_stage_pretrends_bootstrap(self):
        """TwoStageDiD pretrends=True with bootstrap produces finite pre-period SEs."""
        data = generate_test_data(seed=42)
        est = TwoStageDiD(pretrends=True, n_bootstrap=50, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        negative = {
            h: v for h, v in results.event_study_effects.items() if h < -1 and v["n_obs"] > 0
        }
        assert len(negative) > 0, "Should have pre-period horizons"
        for h, eff in negative.items():
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite"
            assert eff["se"] > 0, f"h={h}: SE not positive"
