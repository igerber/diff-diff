"""Tests for Phase 3 survey support: OLS-based standalone estimators.

Covers: StackedDiD, SunAbraham, BaconDecomposition, TripleDifference,
ContinuousDiD, EfficientDiD.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import SurveyDesign

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def staggered_survey_data():
    """Staggered treatment panel with survey design columns.

    60 units, 8 periods, 2 treatment cohorts (t=4, t=6), 20 never-treated.
    5 strata, 12 PSUs, FPC, and sampling weights included.
    """
    np.random.seed(42)
    n_units = 60
    n_periods = 8
    rows = []
    for unit in range(n_units):
        if unit < 20:
            ft = 4  # Early cohort
        elif unit < 40:
            ft = 6  # Late cohort
        else:
            ft = 0  # Never treated

        stratum = unit // 12  # 5 strata
        psu = unit // 5  # 12 PSUs
        fpc_val = 120.0  # Population per stratum
        wt = 1.0 + 0.3 * stratum

        for t in range(1, n_periods + 1):
            y = 10.0 + unit * 0.05 + t * 0.2
            if ft > 0 and t >= ft:
                y += 2.0  # Treatment effect
            y += np.random.normal(0, 0.5)

            rows.append(
                {
                    "unit": unit,
                    "time": t,
                    "first_treat": ft,
                    "outcome": y,
                    "weight": wt,
                    "stratum": stratum,
                    "psu": psu,
                    "fpc": fpc_val,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def ddd_survey_data():
    """Cross-sectional DDD data with survey columns."""
    np.random.seed(42)
    n = 400
    data = pd.DataFrame(
        {
            "outcome": np.random.randn(n) + 0.5,
            "group": np.random.choice([0, 1], n),
            "partition": np.random.choice([0, 1], n),
            "time": np.random.choice([0, 1], n),
            "weight": np.random.uniform(0.5, 2.0, n),
            "stratum": np.random.choice([1, 2, 3], n),
        }
    )
    # Add treatment effect for treated+eligible+post
    mask = (data["group"] == 1) & (data["partition"] == 1) & (data["time"] == 1)
    data.loc[mask, "outcome"] += 1.5
    return data


@pytest.fixture
def continuous_survey_data():
    """Panel data for continuous DiD with survey columns."""
    np.random.seed(42)
    n_u, n_t = 80, 4
    units = np.repeat(range(n_u), n_t)
    times = np.tile(range(1, n_t + 1), n_u)
    ft = np.repeat(np.where(np.arange(n_u) < 40, 3, 0), n_t)
    dose_per_unit = np.where(np.arange(n_u) < 40, np.random.uniform(0.5, 2.0, n_u), 0.0)
    dose = np.repeat(dose_per_unit, n_t)
    y = np.random.randn(len(units)) + 0.5 * dose * (times >= ft) * (ft > 0)
    w = np.repeat(np.random.uniform(0.5, 2.0, n_u), n_t)  # constant within unit
    strata = np.repeat(np.where(np.arange(n_u) < 40, 1, 2), n_t)

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "first_treat": ft,
            "dose": dose,
            "outcome": y,
            "weight": w,
            "stratum": strata,
        }
    )


# =============================================================================
# SunAbraham
# =============================================================================


class TestSunAbrahamSurvey:
    """Survey design support for SunAbraham."""

    def test_smoke_weights_only(self, staggered_survey_data):
        """SunAbraham runs with weights-only survey design."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight")
        est = SunAbraham()
        result = est.fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_uniform_weights_match_unweighted(self, staggered_survey_data):
        """Uniform survey weights should match unweighted result."""
        from diff_diff import SunAbraham

        staggered_survey_data["uniform_w"] = 1.0
        sd = SurveyDesign(weights="uniform_w")

        r_unw = SunAbraham().fit(staggered_survey_data, "outcome", "unit", "time", "first_treat")
        r_w = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert abs(r_unw.overall_att - r_w.overall_att) < 1e-10

    def test_survey_metadata_fields(self, staggered_survey_data):
        """survey_metadata has correct fields with full design."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc", nest=True)
        result = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0
        assert sm.n_strata is not None
        assert sm.n_psu is not None

    def test_se_differs_with_design(self, staggered_survey_data):
        """SEs should differ between weights-only and full design."""
        from diff_diff import SunAbraham

        sd_w = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc", nest=True)
        r_w = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd_w,
        )
        r_full = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd_full,
        )
        # ATTs should be the same (same weights)
        assert abs(r_w.overall_att - r_full.overall_att) < 1e-10
        # SEs should differ due to different variance estimators
        assert r_w.overall_se != r_full.overall_se

    def test_bootstrap_weights_only_uses_pairs(self, staggered_survey_data):
        """Bootstrap + weights-only survey uses pairs bootstrap (no Rao-Wu)."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight")
        result = SunAbraham(n_bootstrap=99, seed=42).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert result.bootstrap_results is not None
        assert result.bootstrap_results.weight_type == "pairs"
        assert result.bootstrap_results.n_bootstrap == 99
        assert np.isfinite(result.overall_se)
        assert np.isfinite(result.overall_att)

    def test_bootstrap_survey_strata_uses_rao_wu(self, staggered_survey_data):
        """Bootstrap + survey with strata/PSU uses Rao-Wu rescaled bootstrap."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu", nest=True)
        result = SunAbraham(n_bootstrap=99, seed=42).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert result.bootstrap_results is not None
        assert result.bootstrap_results.weight_type == "rao_wu"
        assert result.bootstrap_results.n_bootstrap == 99
        assert np.isfinite(result.overall_se)
        assert np.isfinite(result.overall_att)
        # Event study effects should also have finite bootstrap SEs
        for e, eff in result.event_study_effects.items():
            assert np.isfinite(eff["se"]), f"Event study e={e} has non-finite SE"

    def test_summary_includes_survey(self, staggered_survey_data):
        """Summary output should include survey design section."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        summary = result.summary()
        assert "Survey Design" in summary
        assert "pweight" in summary

    def test_no_survey_metadata_is_none(self, staggered_survey_data):
        """Without survey, survey_metadata should be None."""
        from diff_diff import SunAbraham

        result = SunAbraham().fit(staggered_survey_data, "outcome", "unit", "time", "first_treat")
        assert result.survey_metadata is None


# =============================================================================
# StackedDiD
# =============================================================================


class TestStackedDiDSurvey:
    """Survey design support for StackedDiD."""

    def test_smoke_weights_only(self, staggered_survey_data):
        """StackedDiD runs with weights-only survey design."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(weights="weight")
        result = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_survey_metadata_present(self, staggered_survey_data):
        """survey_metadata populated with full design."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert result.survey_metadata is not None
        assert result.survey_metadata.weight_type == "pweight"

    def test_q_weight_composition(self, staggered_survey_data):
        """Survey weights should change results vs unweighted."""
        from diff_diff import StackedDiD

        r_unw = StackedDiD().fit(staggered_survey_data, "outcome", "unit", "time", "first_treat")
        sd = SurveyDesign(weights="weight")
        r_w = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        # ATT should differ (non-uniform weights)
        assert r_unw.overall_att != r_w.overall_att

    def test_convenience_function(self, staggered_survey_data):
        """stacked_did() convenience function threads survey_design."""
        from diff_diff.stacked_did import stacked_did

        sd = SurveyDesign(weights="weight")
        result = stacked_did(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert result.survey_metadata is not None

    def test_summary_includes_survey(self, staggered_survey_data):
        """Summary includes survey design section."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()

    def test_fweight_rejected(self, staggered_survey_data):
        """StackedDiD should reject fweight since Q-weight composition breaks it."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(weights="weight", weight_type="fweight")
        with pytest.raises(ValueError, match="fweight"):
            StackedDiD().fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )

    def test_aweight_rejected(self, staggered_survey_data):
        """StackedDiD should reject aweight since Q-weight composition breaks it."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(weights="weight", weight_type="aweight")
        with pytest.raises(ValueError, match="aweight"):
            StackedDiD().fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )


# =============================================================================
# BaconDecomposition
# =============================================================================


class TestBaconDecompositionSurvey:
    """Survey design support for BaconDecomposition."""

    def test_smoke_weights_only(self, staggered_survey_data):
        """BaconDecomposition runs with weights-only survey design."""
        from diff_diff import BaconDecomposition

        sd = SurveyDesign(weights="weight")
        result = BaconDecomposition().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.twfe_estimate)
        assert len(result.comparisons) > 0
        assert result.survey_metadata is not None

    def test_weighted_changes_twfe(self, staggered_survey_data):
        """Survey weights should change TWFE estimate."""
        from diff_diff import BaconDecomposition

        r_unw = BaconDecomposition().fit(
            staggered_survey_data, "outcome", "unit", "time", "first_treat"
        )
        sd = SurveyDesign(weights="weight")
        r_w = BaconDecomposition().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert r_unw.twfe_estimate != r_w.twfe_estimate

    def test_summary_includes_survey(self, staggered_survey_data):
        """Summary includes survey design section."""
        from diff_diff import BaconDecomposition

        sd = SurveyDesign(weights="weight")
        result = BaconDecomposition().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()

    def test_exact_weights_survey_weighted(self, staggered_survey_data):
        """BaconDecomposition exact weights should use survey-weighted shares."""
        from diff_diff import BaconDecomposition

        sd = SurveyDesign(weights="weight")
        r = BaconDecomposition(weights="exact").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(r.twfe_estimate)
        # Exact weights should still produce valid comparisons
        assert len(r.comparisons) > 0
        for comp in r.comparisons:
            assert np.isfinite(comp.weight)
        # With non-uniform weights, exact weights should differ from
        # approximate weights (approximate uses n_k*(1-n_k)*Var(D))
        r_approx = BaconDecomposition(weights="approximate").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        # At least one comparison weight should differ
        exact_weights = {(c.treated_group, c.control_group): c.weight for c in r.comparisons}
        approx_weights = {
            (c.treated_group, c.control_group): c.weight for c in r_approx.comparisons
        }
        common_keys = set(exact_weights) & set(approx_weights)
        assert len(common_keys) > 0
        diffs = [abs(exact_weights[k] - approx_weights[k]) for k in common_keys]
        assert max(diffs) > 1e-10, "Exact and approximate weights should differ"

    def test_convenience_function(self, staggered_survey_data):
        """bacon_decompose() convenience function threads survey_design."""
        from diff_diff.bacon import bacon_decompose

        sd = SurveyDesign(weights="weight")
        result = bacon_decompose(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert result.survey_metadata is not None


# =============================================================================
# TripleDifference
# =============================================================================


class TestTripleDifferenceSurvey:
    """Survey design support for TripleDifference (reg method only)."""

    def test_smoke_reg_method(self, ddd_survey_data):
        """TripleDifference reg method runs with survey design."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_ipw_survey_works(self, ddd_survey_data):
        """IPW + survey now works (unblocked by weighted solve_logit in Phase 4)."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight")
        result = TripleDifference(estimation_method="ipw").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)

    def test_dr_survey_works(self, ddd_survey_data):
        """DR + survey now works (unblocked by weighted solve_logit in Phase 4)."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight")
        result = TripleDifference(estimation_method="dr").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)

    def test_weighted_changes_att(self, ddd_survey_data):
        """Survey weights should change ATT."""
        from diff_diff import TripleDifference

        r_unw = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data, "outcome", "group", "partition", "time"
        )
        sd = SurveyDesign(weights="weight")
        r_w = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert r_unw.att != r_w.att

    def test_survey_metadata_in_to_dict(self, ddd_survey_data):
        """to_dict() includes survey metadata fields."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        d = result.to_dict()
        assert "weight_type" in d
        assert "effective_n" in d
        assert "design_effect" in d

    def test_summary_includes_survey(self, ddd_survey_data):
        """Summary includes survey design section."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()

    def test_reg_with_covariates_survey(self, ddd_survey_data):
        """TripleDifference reg method works with covariates + survey."""
        from diff_diff import TripleDifference

        # Add a covariate that affects the outcome
        ddd_survey_data["x1"] = np.random.randn(len(ddd_survey_data)) * 0.5
        ddd_survey_data["outcome"] += 0.3 * ddd_survey_data["x1"]
        sd = SurveyDesign(weights="weight", strata="stratum")
        result = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            covariates=["x1"],
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None
        # Survey df should be used for inference
        assert result.survey_metadata.df_survey is not None

    def test_convenience_function(self, ddd_survey_data):
        """triple_difference() convenience function threads survey_design."""
        from diff_diff.triple_diff import triple_difference

        sd = SurveyDesign(weights="weight")
        result = triple_difference(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            estimation_method="reg",
            survey_design=sd,
        )
        assert result.survey_metadata is not None


# =============================================================================
# ContinuousDiD
# =============================================================================


class TestContinuousDiDSurvey:
    """Survey design support for ContinuousDiD."""

    def test_smoke_weights_only(self, continuous_survey_data):
        """ContinuousDiD runs with survey design (analytical SEs)."""
        from diff_diff import ContinuousDiD

        sd = SurveyDesign(weights="weight")
        result = ContinuousDiD(n_bootstrap=0).fit(
            continuous_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert result.survey_metadata is not None

    def test_bootstrap_survey_supported(self, continuous_survey_data):
        """Bootstrap + survey now works via PSU-level multiplier bootstrap."""
        from diff_diff import ContinuousDiD

        sd = SurveyDesign(weights="weight")
        result = ContinuousDiD(n_bootstrap=30, seed=42).fit(
            continuous_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_att_se)

    def test_summary_includes_survey(self, continuous_survey_data):
        """Summary includes survey design section."""
        from diff_diff import ContinuousDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = ContinuousDiD(n_bootstrap=0).fit(
            continuous_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()


# =============================================================================
# EfficientDiD
# =============================================================================


class TestEfficientDiDSurvey:
    """Survey design support for EfficientDiD."""

    def test_smoke_weights_only(self, staggered_survey_data):
        """EfficientDiD runs with weights-only survey design."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_bootstrap_survey_supported(self, staggered_survey_data):
        """Bootstrap + survey now works via PSU-level multiplier bootstrap."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        result = EfficientDiD(n_bootstrap=30, seed=42).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)

    def test_covariates_survey_raises(self, staggered_survey_data):
        """Covariates + survey should raise NotImplementedError."""
        from diff_diff import EfficientDiD

        # Add a covariate column
        staggered_survey_data["x1"] = np.random.randn(len(staggered_survey_data))
        sd = SurveyDesign(weights="weight")
        with pytest.raises(NotImplementedError, match="covariates"):
            EfficientDiD(n_bootstrap=0).fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1"],
                survey_design=sd,
            )

    def test_survey_metadata_fields(self, staggered_survey_data):
        """survey_metadata has correct fields."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0

    def test_summary_includes_survey(self, staggered_survey_data):
        """Summary includes survey design section."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()

    def test_no_survey_metadata_is_none(self, staggered_survey_data):
        """Without survey, survey_metadata should be None."""
        from diff_diff import EfficientDiD

        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data, "outcome", "unit", "time", "first_treat"
        )
        assert result.survey_metadata is None

    def test_survey_event_study_aggregation(self, staggered_survey_data):
        """EfficientDiD survey with aggregate='event_study' produces finite results."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            aggregate="event_study",
            survey_design=sd,
        )
        assert result.event_study_effects is not None
        for e, eff in result.event_study_effects.items():
            assert np.isfinite(eff["effect"])
            assert np.isfinite(eff["se"])
            assert eff["se"] > 0

    def test_survey_group_aggregation(self, staggered_survey_data):
        """EfficientDiD survey with aggregate='group' produces finite results."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            aggregate="group",
            survey_design=sd,
        )
        assert result.group_effects is not None
        for g, eff in result.group_effects.items():
            assert np.isfinite(eff["effect"])
            assert np.isfinite(eff["se"])

    def test_survey_all_aggregation(self, staggered_survey_data):
        """EfficientDiD survey with aggregate='all' produces finite results."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            aggregate="all",
            survey_design=sd,
        )
        assert result.event_study_effects is not None
        assert result.group_effects is not None
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)


# =============================================================================
# Scale Invariance (applies to all estimators)
# =============================================================================


class TestScaleInvariance:
    """Multiplying all survey weights by a constant should not change ATT or SE."""

    def test_sun_abraham_scale_invariance(self, staggered_survey_data):
        from diff_diff import SunAbraham

        sd1 = SurveyDesign(weights="weight")
        r1 = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd1,
        )

        staggered_survey_data["weight_x10"] = staggered_survey_data["weight"] * 10.0
        sd2 = SurveyDesign(weights="weight_x10")
        r2 = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd2,
        )

        assert abs(r1.overall_att - r2.overall_att) < 1e-10
        assert abs(r1.overall_se - r2.overall_se) < 1e-8

    def test_efficient_did_scale_invariance(self, staggered_survey_data):
        from diff_diff import EfficientDiD

        sd1 = SurveyDesign(weights="weight")
        r1 = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd1,
        )

        staggered_survey_data["weight_x10"] = staggered_survey_data["weight"] * 10.0
        sd2 = SurveyDesign(weights="weight_x10")
        r2 = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd2,
        )

        assert abs(r1.overall_att - r2.overall_att) < 1e-10
        assert abs(r1.overall_se - r2.overall_se) < 1e-8

    def test_triple_diff_scale_invariance(self, ddd_survey_data):
        from diff_diff import TripleDifference

        sd1 = SurveyDesign(weights="weight")
        r1 = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd1,
        )
        ddd_survey_data["weight_x10"] = ddd_survey_data["weight"] * 10.0
        sd2 = SurveyDesign(weights="weight_x10")
        r2 = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd2,
        )
        assert abs(r1.att - r2.att) < 1e-10
        assert abs(r1.se - r2.se) < 1e-8

    def test_sun_abraham_sub_unit_weight_scale_invariance(self, staggered_survey_data):
        """SunAbraham overall ATT should be invariant to sub-1 weight rescaling."""
        from diff_diff import SunAbraham

        sd1 = SurveyDesign(weights="weight")
        r1 = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd1,
        )

        # Scale weights to be < 1 (e.g., 0.01x)
        staggered_survey_data["weight_tiny"] = staggered_survey_data["weight"] * 0.01
        sd2 = SurveyDesign(weights="weight_tiny")
        r2 = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd2,
        )

        assert abs(r1.overall_att - r2.overall_att) < 1e-10
        assert abs(r1.overall_se - r2.overall_se) < 1e-8


# =============================================================================
# Regression Tests (PR #226 review feedback)
# =============================================================================


class TestReviewRegressions:
    """Targeted tests for issues found in PR #226 review."""

    def test_stacked_did_no_weight_survey(self, staggered_survey_data):
        """StackedDiD should handle SurveyDesign without weights column."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(strata="stratum")  # No weights
        result = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert result.survey_metadata is not None

    def test_triple_diff_survey_df(self, ddd_survey_data):
        """TripleDifference should use survey df for p-values when survey active."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight", strata="stratum")
        r_survey = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        r_nosurv = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data, "outcome", "group", "partition", "time"
        )
        # P-values should differ (different df)
        if np.isfinite(r_survey.p_value) and np.isfinite(r_nosurv.p_value):
            assert r_survey.p_value != r_nosurv.p_value

    def test_efficient_did_weights_only_se(self, staggered_survey_data):
        """EfficientDiD weights-only survey SE should be reasonable (not tiny)."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        r = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0
        assert r.overall_se > 0.01  # Not artificially tiny

    def test_continuous_did_eventstudy_survey(self, continuous_survey_data):
        """ContinuousDiD aggregate=eventstudy should work with survey design."""
        from diff_diff import ContinuousDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = ContinuousDiD(n_bootstrap=0).fit(
            continuous_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            aggregate="eventstudy",
            survey_design=sd,
        )
        assert result.event_study_effects is not None
        assert result.survey_metadata is not None
        for e, eff in result.event_study_effects.items():
            if np.isfinite(eff["effect"]):
                assert np.isfinite(eff["se"]), f"Non-finite SE for e={e}"

    def test_within_unit_varying_weights_rejected(self):
        """Time-varying survey weights within units should be rejected."""
        from diff_diff import ContinuousDiD

        np.random.seed(42)
        n_u, n_t = 20, 4
        data = pd.DataFrame(
            {
                "unit": np.repeat(range(n_u), n_t),
                "time": np.tile(range(1, n_t + 1), n_u),
                "first_treat": np.repeat(np.where(np.arange(n_u) < 10, 3, 0), n_t),
                "dose": np.repeat(np.where(np.arange(n_u) < 10, 1.0, 0.0), n_t),
                "outcome": np.random.randn(n_u * n_t),
                "w": np.random.uniform(0.5, 2.0, n_u * n_t),
            }
        )
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="varies within units"):
            ContinuousDiD(n_bootstrap=0).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                "dose",
                survey_design=sd,
            )

    def test_within_unit_varying_strata_rejected(self):
        """Time-varying strata within units should be rejected."""
        from diff_diff import EfficientDiD

        np.random.seed(42)
        n_u, n_t = 20, 4
        data = pd.DataFrame(
            {
                "unit": np.repeat(range(n_u), n_t),
                "time": np.tile(range(1, n_t + 1), n_u),
                "first_treat": np.repeat(np.where(np.arange(n_u) < 10, 3, 0), n_t),
                "outcome": np.random.randn(n_u * n_t),
                "w": np.repeat(np.ones(n_u), n_t),
                "strat": np.tile([1, 2, 1, 2], n_u),
            }
        )
        sd = SurveyDesign(weights="w", strata="strat")
        with pytest.raises(ValueError, match="varies within units"):
            EfficientDiD(n_bootstrap=0).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )

    def test_bacon_exact_varying_weights_rejected(self):
        """BaconDecomposition exact weights should reject time-varying survey weights."""
        from diff_diff import BaconDecomposition

        np.random.seed(42)
        n_u, n_t = 20, 4
        data = pd.DataFrame(
            {
                "unit": np.repeat(range(n_u), n_t),
                "time": np.tile(range(1, n_t + 1), n_u),
                "first_treat": np.repeat(np.where(np.arange(n_u) < 10, 3, 0), n_t),
                "outcome": np.random.randn(n_u * n_t),
                # Time-varying weights (different per period)
                "w": np.random.uniform(0.5, 2.0, n_u * n_t),
            }
        )
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="varies within units"):
            BaconDecomposition(weights="exact").fit(
                data, "outcome", "unit", "time", "first_treat", survey_design=sd
            )

    def test_sun_abraham_survey_df_regression(self, staggered_survey_data):
        """SunAbraham survey inference should use survey df, not normal approx."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc", nest=True)
        result = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.df_survey is not None
        # Overall p-value should use t-distribution (survey df), not normal
        # Recompute with normal approx and verify they differ
        from diff_diff.utils import safe_inference

        _, p_normal, _ = safe_inference(result.overall_att, result.overall_se, alpha=0.05, df=None)
        _, p_survey, _ = safe_inference(
            result.overall_att, result.overall_se, alpha=0.05, df=sm.df_survey
        )
        # With small survey df, t-dist p-value > normal p-value
        if np.isfinite(result.overall_p_value) and np.isfinite(p_normal):
            assert result.overall_p_value == pytest.approx(p_survey, rel=1e-10)

    def test_continuous_did_dose_response_survey_pvalue(self, continuous_survey_data):
        """DoseResponseCurve.to_dataframe() p-values should use survey df."""
        from diff_diff import ContinuousDiD
        from diff_diff.utils import safe_inference

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = ContinuousDiD(n_bootstrap=0).fit(
            continuous_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=sd,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.df_survey is not None
        # Check that dose-response curve carries survey df
        assert result.dose_response_att.df_survey == sm.df_survey
        # Check exported p-values use survey df, not normal approx
        att_df = result.dose_response_att.to_dataframe()
        for i in range(min(3, len(att_df))):
            row = att_df.iloc[i]
            if np.isfinite(row["effect"]) and np.isfinite(row["se"]) and row["se"] > 0:
                _, expected_p, _ = safe_inference(row["effect"], row["se"], df=sm.df_survey)
                assert row["p_value"] == pytest.approx(expected_p, rel=1e-10)


# =============================================================================
# Survey Edge Case Coverage Gaps
# =============================================================================


class TestSurveyEdgeCases:
    """Tests for FPC census, single-PSU NaN, and full-design bootstrap."""

    def test_fpc_census_zero_variance_stacked(self, staggered_survey_data):
        """When FPC == n_PSU (full census), sampling variance should be ~0."""
        from diff_diff import StackedDiD

        data = staggered_survey_data.copy()
        # Set FPC = n_PSU per stratum (census)
        for h in data["stratum"].unique():
            mask = data["stratum"] == h
            n_psu_h = data.loc[mask, "psu"].nunique()
            data.loc[mask, "fpc"] = float(n_psu_h)

        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", fpc="fpc",
            nest=True,
        )
        result = StackedDiD().fit(
            data, "outcome", "unit", "time", "first_treat",
            survey_design=sd,
        )
        assert result.overall_se == pytest.approx(0.0, abs=1e-10)

    def test_single_psu_lonely_remove_nan_se(self, staggered_survey_data):
        """All strata with 1 PSU + lonely_psu='remove' -> NaN SE."""
        from diff_diff import SunAbraham

        data = staggered_survey_data.copy()
        # Force all units into a single PSU per stratum
        data["single_psu"] = data["stratum"]

        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="single_psu",
            lonely_psu="remove",
        )
        with pytest.warns(UserWarning, match="only 1 PSU"):
            result = SunAbraham().fit(
                data, "outcome", "unit", "time", "first_treat",
                survey_design=sd,
            )
        # All strata removed -> NaN SE
        assert np.isnan(result.overall_se)

    def test_full_design_bootstrap_continuous_did(self, staggered_survey_data):
        """Bootstrap with strata survey design works for ContinuousDiD."""
        from diff_diff import ContinuousDiD

        data = staggered_survey_data.copy()
        # Add dose column
        data["dose"] = np.where(data["first_treat"] > 0, 1.0 + 0.5 * data["unit"] % 3, 0.0)

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = ContinuousDiD(n_bootstrap=30, seed=42).fit(
            data, "outcome", "unit", "time", "first_treat", "dose",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_att_se)
        assert result.survey_metadata is not None

    def test_full_design_bootstrap_efficient_did(self, staggered_survey_data):
        """Bootstrap with strata survey design works for EfficientDiD."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = EfficientDiD(n_bootstrap=30, seed=42).fit(
            staggered_survey_data, "outcome", "unit", "time", "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None
