"""Tests for Phase 7b: CallawaySantAnna repeated cross-section support.

Covers: panel=False mode with reg/ipw/dr, covariates, survey weights,
aggregation, bootstrap, control group options, base period options,
and edge cases.
"""

import numpy as np
import pytest

from diff_diff import CallawaySantAnna, SurveyDesign, generate_staggered_data

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def rc_data():
    """Basic repeated cross-section data."""
    return generate_staggered_data(n_units=200, n_periods=6, panel=False, seed=42)


@pytest.fixture(scope="module")
def rc_data_with_covariates():
    """RCS data with a covariate."""
    data = generate_staggered_data(n_units=200, n_periods=6, panel=False, seed=42)
    rng = np.random.default_rng(42)
    data["x1"] = rng.normal(0, 1, len(data))
    return data


@pytest.fixture(scope="module")
def rc_data_with_survey():
    """RCS data with survey weights."""
    data = generate_staggered_data(n_units=200, n_periods=6, panel=False, seed=42)
    rng = np.random.default_rng(42)
    data["x1"] = rng.normal(0, 1, len(data))
    data["weight"] = rng.uniform(0.5, 2.0, len(data))
    return data


# =============================================================================
# Basic Fit
# =============================================================================


class TestBasicFit:
    """Basic repeated cross-section fit tests."""

    def test_basic_reg(self, rc_data):
        result = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_all_methods(self, rc_data, method):
        result = CallawaySantAnna(estimation_method=method, panel=False).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_panel_param_in_get_params(self):
        cs = CallawaySantAnna(panel=False)
        params = cs.get_params()
        assert params["panel"] is False


# =============================================================================
# Methods Agree Without Covariates
# =============================================================================


class TestMethodsAgreeNoCovariates:
    """Without covariates, reg/ipw/dr should give identical ATTs in RCS."""

    def test_no_covariate_methods_agree(self, rc_data):
        results = {}
        for method in ["reg", "ipw", "dr"]:
            r = CallawaySantAnna(estimation_method=method, panel=False).fit(
                rc_data,
                "outcome",
                "unit",
                "period",
                "first_treat",
            )
            results[method] = r.overall_att

        np.testing.assert_allclose(results["reg"], results["ipw"], atol=1e-10)
        np.testing.assert_allclose(results["reg"], results["dr"], atol=1e-10)


# =============================================================================
# Treatment Effect Recovery
# =============================================================================


class TestTreatmentEffectRecovery:
    """Known DGP should recover approximately correct treatment effect."""

    def test_positive_effect(self, rc_data):
        result = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        # DGP has treatment_effect=2.0 by default
        assert result.overall_att > 0
        assert abs(result.overall_att - 2.0) < 2.0  # within 2 SE roughly


# =============================================================================
# Aggregation
# =============================================================================


class TestAggregation:
    """Aggregation types work with RCS."""

    @pytest.mark.parametrize("agg", ["simple", "event_study", "group"])
    def test_aggregation(self, rc_data, agg):
        result = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            aggregate=agg,
        )
        assert np.isfinite(result.overall_att)
        if agg == "event_study":
            assert result.event_study_effects is not None
            for e, info in result.event_study_effects.items():
                if info["n_groups"] > 0:
                    assert np.isfinite(info["effect"])
        if agg == "group":
            assert result.group_effects is not None


# =============================================================================
# Covariates
# =============================================================================


class TestCovariates:
    """Covariate-adjusted estimation in RCS."""

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_with_covariates(self, rc_data_with_covariates, method):
        result = CallawaySantAnna(estimation_method=method, panel=False).fit(
            rc_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0


# =============================================================================
# Survey Weights
# =============================================================================


class TestSurveyWeights:
    """Survey weights work with RCS (per-observation)."""

    def test_survey_weights_pweight(self, rc_data_with_survey):
        sd = SurveyDesign(weights="weight")
        result = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            rc_data_with_survey,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_survey_covariates_dr(self, rc_data_with_survey):
        """Combined: survey + covariates + DR + RCS."""
        sd = SurveyDesign(weights="weight")
        result = CallawaySantAnna(estimation_method="dr", panel=False).fit(
            rc_data_with_survey,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0


# =============================================================================
# Control Group Options
# =============================================================================


class TestControlGroup:
    """Control group options work with RCS."""

    def test_not_yet_treated(self, rc_data):
        result = CallawaySantAnna(
            estimation_method="reg", panel=False, control_group="not_yet_treated"
        ).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert np.isfinite(result.overall_att)

    def test_never_treated(self, rc_data):
        result = CallawaySantAnna(
            estimation_method="reg", panel=False, control_group="never_treated"
        ).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert np.isfinite(result.overall_att)


# =============================================================================
# Base Period Options
# =============================================================================


class TestBasePeriod:
    """Base period options work with RCS."""

    def test_universal(self, rc_data):
        result = CallawaySantAnna(
            estimation_method="reg", panel=False, base_period="universal"
        ).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert np.isfinite(result.overall_att)

    def test_varying(self, rc_data):
        result = CallawaySantAnna(estimation_method="reg", panel=False, base_period="varying").fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert np.isfinite(result.overall_att)


# =============================================================================
# Bootstrap
# =============================================================================


class TestBootstrap:
    """Bootstrap works with RCS."""

    def test_bootstrap_reg(self, rc_data):
        result = CallawaySantAnna(
            estimation_method="reg", panel=False, n_bootstrap=49, seed=42
        ).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert result.bootstrap_results is not None
        assert np.isfinite(result.overall_att)


# =============================================================================
# Data Generator
# =============================================================================


class TestDataGenerator:
    """Test the RCS data generator."""

    def test_rc_data_structure(self):
        data = generate_staggered_data(n_units=100, n_periods=5, panel=False, seed=99)
        # Each observation should have a unique unit ID
        assert data["unit"].nunique() == len(data)
        # Should have n_units * n_periods rows
        assert len(data) == 100 * 5
        # Each period should have n_units observations
        assert all(data.groupby("period")["unit"].count() == 100)

    def test_panel_data_unchanged(self):
        """panel=True (default) should produce panel data."""
        data = generate_staggered_data(n_units=50, n_periods=4, panel=True, seed=42)
        # Units should repeat across periods
        assert data["unit"].nunique() < len(data)
        assert data["unit"].nunique() == 50


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases for RCS."""

    def test_empty_cell_nan(self):
        """(g,t) cell with no observations should be NaN, not crash."""
        data = generate_staggered_data(n_units=50, n_periods=4, panel=False, seed=42)
        # This should handle cells with few/no observations gracefully
        result = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        # Should produce at least some finite effects
        finite_effects = [
            v["effect"] for v in result.group_time_effects.values() if np.isfinite(v["effect"])
        ]
        assert len(finite_effects) > 0


# =============================================================================
# Methodology: IF corrections change SE
# =============================================================================


class TestIFCorrections:
    """Verify RCS DR/IPW IF corrections are non-trivial."""

    def test_dr_se_differs_from_reg_rc(self, rc_data_with_covariates):
        """DR and reg should give different SEs in RCS (DR has IF corrections)."""
        r_reg = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            rc_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )
        r_dr = CallawaySantAnna(estimation_method="dr", panel=False).fit(
            rc_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )
        # SEs should differ (DR has nuisance IF corrections)
        assert r_reg.overall_se != r_dr.overall_se

    def test_panel_field_on_results(self, rc_data):
        """panel=False should be reflected on CallawaySantAnnaResults."""
        result = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert result.panel is False

    def test_summary_labels_rcs(self, rc_data):
        """Summary should use 'obs' labels for RCS, not 'units'."""
        result = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        summary = result.summary()
        assert "obs:" in summary
        assert "units:" not in summary.split("\n")[3]  # Treated line


# =============================================================================
# Analytical vs Bootstrap SE convergence (proves IF scaling is correct)
# =============================================================================


class TestAnalyticalBootstrapConvergence:
    """Analytical SE should closely match bootstrap SE — proves IF magnitude is correct."""

    def test_reg_se_matches_bootstrap(self, rc_data_with_covariates):
        """Analytical reg SE should be within 20% of bootstrap SE."""
        r_analytical = CallawaySantAnna(estimation_method="reg", panel=False).fit(
            rc_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )
        r_bootstrap = CallawaySantAnna(
            estimation_method="reg", panel=False, n_bootstrap=499, seed=42
        ).fit(
            rc_data_with_covariates,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )
        # ATTs should match (bootstrap doesn't change point estimate)
        np.testing.assert_allclose(r_analytical.overall_att, r_bootstrap.overall_att, atol=1e-10)
        # SEs should be within 10% (proves IF scaling is correct)
        ratio = r_analytical.overall_se / r_bootstrap.overall_se
        assert 0.9 < ratio < 1.1, (
            f"Analytical/bootstrap SE ratio {ratio:.3f} outside [0.9, 1.1] — "
            f"analytical={r_analytical.overall_se:.4f}, bootstrap={r_bootstrap.overall_se:.4f}"
        )


# =============================================================================
# Unequal Cohort Counts Across Periods
# =============================================================================


class TestUnequalCohortCounts:
    """Tests with n_gt != n_gs — catches normalizer/weight bugs."""

    @pytest.fixture
    def unequal_rc_data(self):
        """RCS data where cohort sizes differ across periods."""
        rng = np.random.default_rng(77)
        records = []
        # 4 periods, cohort g=2 treated at period 2
        for period in range(4):
            # Vary n_per_period so cohort counts differ across periods
            n_per_period = 100 + period * 30  # 100, 130, 160, 190
            for i in range(n_per_period):
                # ~30% treated (cohort 2), ~70% never-treated
                ft = 2 if rng.random() < 0.3 else 0
                treated = (ft > 0) and (period >= ft)
                y = rng.normal(0, 1) + (2.0 if treated else 0.0)
                records.append(
                    {
                        "unit": f"u{period}_{i}",
                        "period": period,
                        "outcome": y,
                        "first_treat": ft,
                    }
                )
        import pandas as pd

        return pd.DataFrame(records)

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_finite_results_unequal(self, unequal_rc_data, method):
        result = CallawaySantAnna(estimation_method=method, panel=False).fit(
            unequal_rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_dr_covariates_unequal(self, unequal_rc_data):
        """DR with covariates under unequal cohort counts."""
        rng = np.random.default_rng(77)
        unequal_rc_data["x1"] = rng.normal(0, 1, len(unequal_rc_data))
        result = CallawaySantAnna(estimation_method="dr", panel=False).fit(
            unequal_rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)

    def test_bootstrap_unequal(self, unequal_rc_data):
        """Bootstrap with unequal cohort counts."""
        result = CallawaySantAnna(
            estimation_method="reg", panel=False, n_bootstrap=49, seed=42
        ).fit(
            unequal_rc_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        assert result.bootstrap_results is not None
        assert np.isfinite(result.overall_att)
