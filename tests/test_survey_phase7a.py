"""Tests for Phase 7a: CallawaySantAnna IPW/DR + covariates + survey support.

Covers: DR nuisance IF corrections (PS + OR), IPW unblocking,
scale invariance, uniform-weight equivalence, aggregation, bootstrap,
and edge cases.
"""

import numpy as np
import pytest

from diff_diff import CallawaySantAnna, SurveyDesign, generate_staggered_data

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def staggered_data_with_covariates():
    """Staggered panel with a covariate and survey columns."""
    data = generate_staggered_data(n_units=200, n_periods=6, seed=42)
    rng = np.random.default_rng(42)

    unit_ids = data["unit"].unique()
    n_units = len(unit_ids)
    unit_map = {uid: i for i, uid in enumerate(unit_ids)}
    idx = data["unit"].map(unit_map).values

    # Covariate: unit-level (constant within unit, varies across units)
    unit_x = rng.normal(0, 1, n_units)
    data["x1"] = unit_x[idx]

    # Survey design columns (unit-level)
    data["weight"] = (1.0 + 0.5 * (np.arange(n_units) % 5))[idx]
    data["stratum"] = (np.arange(n_units) // 40)[idx]
    data["psu"] = (np.arange(n_units) // 10)[idx]

    return data


@pytest.fixture(scope="module")
def survey_weights_only():
    return SurveyDesign(weights="weight")


@pytest.fixture(scope="module")
def survey_full_design():
    return SurveyDesign(weights="weight", strata="stratum", psu="psu")


# =============================================================================
# Smoke Tests: IPW and DR with covariates + survey produce finite results
# =============================================================================


class TestSmokeIPWDRSurvey:
    """Basic smoke tests that IPW/DR + covariates + survey runs and returns
    finite results."""

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_finite_results(self, staggered_data_with_covariates, survey_weights_only, method):
        result = CallawaySantAnna(estimation_method=method).fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_weights_only,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_event_study(self, staggered_data_with_covariates, survey_weights_only, method):
        result = CallawaySantAnna(estimation_method=method).fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_weights_only,
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        for e, info in result.event_study_effects.items():
            if info["n_groups"] > 0:
                assert np.isfinite(info["effect"])
                assert np.isfinite(info["se"])

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_full_design(self, staggered_data_with_covariates, survey_full_design, method):
        """Strata/PSU design with covariates + IPW/DR."""
        result = CallawaySantAnna(estimation_method=method).fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_full_design,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0


# =============================================================================
# Scale Invariance: doubling all weights doesn't change ATT
# =============================================================================


class TestScaleInvariance:
    """Multiplying all survey weights by a constant should not change ATT."""

    @pytest.mark.parametrize("method", ["ipw", "dr", "reg"])
    def test_double_weights_same_att(self, staggered_data_with_covariates, method):
        data = staggered_data_with_covariates.copy()

        # Fit with original weights
        sd1 = SurveyDesign(weights="weight")
        r1 = CallawaySantAnna(estimation_method=method).fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=sd1,
        )

        # Fit with doubled weights
        data["weight2"] = data["weight"] * 2
        sd2 = SurveyDesign(weights="weight2")
        r2 = CallawaySantAnna(estimation_method=method).fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=sd2,
        )

        np.testing.assert_allclose(r1.overall_att, r2.overall_att, atol=1e-10)


# =============================================================================
# Uniform Weights Match Unweighted
# =============================================================================


class TestUniformWeightsMatchUnweighted:
    """Survey weights = 1.0 for all should match the no-survey path."""

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_uniform_weights(self, method):
        data = generate_staggered_data(n_units=100, n_periods=5, seed=123)
        rng = np.random.default_rng(123)
        data["x1"] = rng.normal(0, 1, len(data))
        data["weight_ones"] = 1.0

        # No survey
        r_no_survey = CallawaySantAnna(estimation_method=method).fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )

        # Survey with uniform weights
        sd = SurveyDesign(weights="weight_ones")
        r_survey = CallawaySantAnna(estimation_method=method).fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=sd,
        )

        # ATTs should be very close (not exact due to weight normalization)
        np.testing.assert_allclose(r_no_survey.overall_att, r_survey.overall_att, rtol=1e-6)


# =============================================================================
# IF Correction Non-Zero: corrected IF differs from plug-in
# =============================================================================


class TestIFCorrectionNonZero:
    """The DR nuisance IF corrections should make a difference (non-trivial)."""

    def test_dr_se_differs_from_reg(self, staggered_data_with_covariates, survey_weights_only):
        """DR and reg methods should give different SEs (DR has PS correction)."""
        r_reg = CallawaySantAnna(estimation_method="reg").fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_weights_only,
        )
        r_dr = CallawaySantAnna(estimation_method="dr").fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_weights_only,
        )
        # SEs should differ because DR has additional IF corrections
        assert r_reg.overall_se != r_dr.overall_se


# =============================================================================
# All 3 Methods Agree Without Covariates
# =============================================================================


class TestMethodsAgreeNoCovariates:
    """Without covariates, reg/ipw/dr should give identical ATTs under survey."""

    def test_no_covariate_methods_agree(self, staggered_data_with_covariates, survey_weights_only):
        results = {}
        for method in ["reg", "ipw", "dr"]:
            r = CallawaySantAnna(estimation_method=method).fit(
                staggered_data_with_covariates,
                "outcome",
                "unit",
                "period",
                "first_treat",
                survey_design=survey_weights_only,
            )
            results[method] = r.overall_att

        np.testing.assert_allclose(results["reg"], results["ipw"], atol=1e-10)
        np.testing.assert_allclose(results["reg"], results["dr"], atol=1e-10)


# =============================================================================
# Aggregation
# =============================================================================


class TestAggregation:
    """Aggregation types work with DR + covariates + survey."""

    @pytest.mark.parametrize("agg", ["simple", "event_study", "group"])
    def test_aggregation(self, staggered_data_with_covariates, survey_weights_only, agg):
        result = CallawaySantAnna(estimation_method="dr").fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_weights_only,
            aggregate=agg,
        )
        assert np.isfinite(result.overall_att)
        if agg == "event_study":
            assert result.event_study_effects is not None
        if agg == "group":
            assert result.group_effects is not None


# =============================================================================
# Bootstrap
# =============================================================================


class TestBootstrap:
    """Bootstrap with IPW/DR + covariates + survey."""

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_bootstrap_runs(self, staggered_data_with_covariates, survey_weights_only, method):
        result = CallawaySantAnna(estimation_method=method, n_bootstrap=49, seed=42).fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_weights_only,
        )
        assert np.isfinite(result.overall_att)
        assert result.bootstrap_results is not None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases for IPW/DR + survey + covariates."""

    def test_single_covariate(self):
        """Works with a single binary covariate."""
        data = generate_staggered_data(n_units=100, seed=99)
        rng = np.random.default_rng(99)
        data["binary_x"] = rng.choice([0, 1], len(data))
        data["weight"] = rng.uniform(0.5, 2.0, len(data)).round(1)
        # Make weights constant within unit
        unit_w = data.groupby("unit")["weight"].first()
        data["weight"] = data["unit"].map(unit_w)

        sd = SurveyDesign(weights="weight")
        for method in ["ipw", "dr"]:
            result = CallawaySantAnna(estimation_method=method).fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                covariates=["binary_x"],
                survey_design=sd,
            )
            assert np.isfinite(result.overall_att)

    def test_not_yet_treated_control(self, staggered_data_with_covariates, survey_weights_only):
        """control_group='not_yet_treated' with DR + covariates + survey."""
        result = CallawaySantAnna(estimation_method="dr", control_group="not_yet_treated").fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_weights_only,
        )
        assert np.isfinite(result.overall_att)

    def test_universal_base_period(self, staggered_data_with_covariates, survey_weights_only):
        """base_period='universal' with DR + covariates + survey."""
        result = CallawaySantAnna(estimation_method="dr", base_period="universal").fit(
            staggered_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_weights_only,
        )
        assert np.isfinite(result.overall_att)


# =============================================================================
# Non-Survey DR IF Corrections
# =============================================================================


class TestNonSurveyDRIFCorrections:
    """Verify that non-survey DR path also has IF corrections."""

    def test_dr_se_differs_from_reg_no_survey(self):
        """DR and reg should give different SEs without survey (IF corrections)."""
        data = generate_staggered_data(n_units=150, n_periods=6, seed=42)
        rng = np.random.default_rng(42)
        data["x1"] = rng.normal(0, 1, len(data))

        r_reg = CallawaySantAnna(estimation_method="reg").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )
        r_dr = CallawaySantAnna(estimation_method="dr").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )
        # SEs should differ (DR has nuisance IF corrections)
        assert r_reg.overall_se != r_dr.overall_se
        # But ATTs should be similar (both consistent under correct specification)
        assert abs(r_reg.overall_att - r_dr.overall_att) < 1.0
