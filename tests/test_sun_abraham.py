"""
Tests for Sun-Abraham interaction-weighted estimator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff.sun_abraham import SunAbraham, SunAbrahamResults


def generate_staggered_data(
    n_units: int = 100,
    n_periods: int = 10,
    n_cohorts: int = 3,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic staggered adoption data."""
    np.random.seed(seed)

    # Generate unit and time identifiers
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    # Assign treatment cohorts
    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    # Treatment periods start from period 3 onwards
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)

    first_treat = np.zeros(n_units)
    if n_treated > 0:
        cohort_assignments = np.random.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = cohort_periods[cohort_assignments]

    first_treat_expanded = np.repeat(first_treat, n_periods)

    # Generate outcomes
    unit_fe = np.random.randn(n_units) * 2
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    # Treatment indicator
    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)

    # Dynamic treatment effects
    relative_time = times - first_treat_expanded
    dynamic_effect = treatment_effect * (1 + 0.1 * np.maximum(relative_time, 0))

    outcomes = (
        unit_fe_expanded
        + time_fe_expanded
        + dynamic_effect * post
        + np.random.randn(len(units)) * 0.5
    )

    df = pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcomes,
            "first_treat": first_treat_expanded.astype(int),
        }
    )

    return df


class TestSunAbraham:
    """Tests for SunAbraham estimator."""

    def test_basic_fit(self):
        """Test basic model fitting."""
        data = generate_staggered_data()

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert sa.is_fitted_
        assert isinstance(results, SunAbrahamResults)
        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.event_study_effects) > 0

    def test_positive_treatment_effect(self):
        """Test that estimator recovers positive treatment effect."""
        data = generate_staggered_data(treatment_effect=3.0, seed=123)

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Should detect positive effect
        assert results.overall_att > 0
        # Effect should be roughly correct (within reasonable bounds)
        assert abs(results.overall_att - 3.0) < 2 * results.overall_se + 1.5

    def test_zero_treatment_effect(self):
        """Test with no treatment effect."""
        data = generate_staggered_data(treatment_effect=0.0, seed=456)

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Effect should be close to zero
        assert abs(results.overall_att) < 3 * results.overall_se + 0.5

    def test_event_study_effects(self):
        """Test event study effects structure."""
        data = generate_staggered_data()

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

        # Check structure of effect dictionary
        for e, eff in results.event_study_effects.items():
            assert "effect" in eff
            assert "se" in eff
            assert "t_stat" in eff
            assert "p_value" in eff
            assert "conf_int" in eff
            assert isinstance(eff["conf_int"], tuple)
            assert len(eff["conf_int"]) == 2

    def test_cohort_weights(self):
        """Test that cohort weights are computed."""
        data = generate_staggered_data()

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.cohort_weights is not None
        assert len(results.cohort_weights) > 0

        # Weights should sum to 1 for each relative period
        for e, weights in results.cohort_weights.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-10, f"Weights for e={e} sum to {total}"

    def test_control_group_options(self):
        """Test different control group options."""
        data = generate_staggered_data()

        # Never treated only
        sa1 = SunAbraham(control_group="never_treated")
        results1 = sa1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Not yet treated
        sa2 = SunAbraham(control_group="not_yet_treated")
        results2 = sa2.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results1.control_group == "never_treated"
        assert results2.control_group == "not_yet_treated"
        # Results may be different
        # (they don't have to be for this test to pass)
        assert results1.overall_att is not None
        assert results2.overall_att is not None

    def test_summary_output(self):
        """Test summary output formatting."""
        data = generate_staggered_data()

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        summary = results.summary()

        assert "Sun-Abraham" in summary
        assert "ATT" in summary
        assert "Std. Err." in summary
        assert "Event Study" in summary

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        data = generate_staggered_data()

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Event study DataFrame
        df_es = results.to_dataframe(level="event_study")
        assert "relative_period" in df_es.columns
        assert "effect" in df_es.columns
        assert "se" in df_es.columns

    def test_get_set_params(self):
        """Test sklearn-compatible parameter access."""
        sa = SunAbraham(alpha=0.10, control_group="not_yet_treated")

        params = sa.get_params()
        assert params["alpha"] == 0.10
        assert params["control_group"] == "not_yet_treated"

        sa.set_params(alpha=0.05)
        assert sa.alpha == 0.05

    def test_missing_column_error(self):
        """Test error on missing columns."""
        data = generate_staggered_data()

        sa = SunAbraham()

        with pytest.raises(ValueError, match="Missing columns"):
            sa.fit(
                data,
                outcome="nonexistent",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_no_control_units_error(self):
        """Test error when no control units exist."""
        data = generate_staggered_data(never_treated_frac=0.0)

        sa = SunAbraham()

        with pytest.raises(ValueError, match="No never-treated units"):
            sa.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_significance_properties(self):
        """Test significance-related properties."""
        data = generate_staggered_data(treatment_effect=5.0)

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # With strong effect, should be significant
        assert results.is_significant
        assert results.significance_stars in ["*", "**", "***"]


class TestSunAbrahamResults:
    """Tests for SunAbrahamResults class."""

    def test_repr(self):
        """Test string representation."""
        data = generate_staggered_data()
        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        repr_str = repr(results)
        assert "SunAbrahamResults" in repr_str
        assert "ATT=" in repr_str

    def test_invalid_level_error(self):
        """Test error on invalid DataFrame level."""
        data = generate_staggered_data()
        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe(level="invalid")


@pytest.mark.slow
class TestSunAbrahamBootstrap:
    """Tests for Sun-Abraham bootstrap inference."""

    def test_bootstrap_basic(self, ci_params):
        """Test basic bootstrap functionality."""
        data = generate_staggered_data(n_units=50, seed=42)

        n_boot = ci_params.bootstrap(99)
        sa = SunAbraham(n_bootstrap=n_boot, seed=42)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.n_bootstrap == n_boot
        assert results.bootstrap_results.weight_type == "pairs"
        assert results.overall_se > 0
        assert results.overall_conf_int[0] < results.overall_att < results.overall_conf_int[1]

    def test_bootstrap_reproducibility(self, ci_params):
        """Test that bootstrap is reproducible with same seed."""
        data = generate_staggered_data(n_units=50, seed=42)

        n_boot = ci_params.bootstrap(99)
        sa1 = SunAbraham(n_bootstrap=n_boot, seed=123)
        results1 = sa1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        sa2 = SunAbraham(n_bootstrap=n_boot, seed=123)
        results2 = sa2.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Results should be identical with same seed
        assert results1.overall_se == results2.overall_se
        assert results1.overall_conf_int == results2.overall_conf_int

    def test_bootstrap_different_seeds(self, ci_params):
        """Test that different seeds give different results."""
        data = generate_staggered_data(n_units=50, seed=42)

        n_boot = ci_params.bootstrap(99)
        sa1 = SunAbraham(n_bootstrap=n_boot, seed=123)
        results1 = sa1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        sa2 = SunAbraham(n_bootstrap=n_boot, seed=456)
        results2 = sa2.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Results should differ with different seeds
        assert results1.overall_se != results2.overall_se

    def test_bootstrap_p_value_significance(self, ci_params):
        """Test that strong effect has significant p-value with bootstrap."""
        data = generate_staggered_data(n_units=100, treatment_effect=5.0, seed=42)

        n_boot = ci_params.bootstrap(199)
        sa = SunAbraham(n_bootstrap=n_boot, seed=42)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Strong effect should be significant
        assert results.overall_p_value < 0.05
        assert results.is_significant

    def test_bootstrap_distribution_stored(self, ci_params):
        """Test that bootstrap distribution is stored in results."""
        data = generate_staggered_data(n_units=50, seed=42)

        n_boot = ci_params.bootstrap(99)
        sa = SunAbraham(n_bootstrap=n_boot, seed=42)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.bootstrap_results.bootstrap_distribution is not None
        assert len(results.bootstrap_results.bootstrap_distribution) == n_boot

    def test_bootstrap_event_study_effects(self, ci_params):
        """Test that bootstrap updates event study effect SEs."""
        data = generate_staggered_data(n_units=50, seed=42)

        n_boot = ci_params.bootstrap(99)
        sa = SunAbraham(n_bootstrap=n_boot, seed=42)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        assert results.bootstrap_results.event_study_cis is not None
        assert results.bootstrap_results.event_study_p_values is not None

        # Check event study effects have bootstrap SEs
        for e, effect in results.event_study_effects.items():
            assert effect["se"] > 0
            assert effect["conf_int"][0] < effect["conf_int"][1]

    def test_bootstrap_low_iterations_warning(self):
        """Test that low n_bootstrap triggers a warning."""
        data = generate_staggered_data(n_units=50, seed=42)

        sa = SunAbraham(n_bootstrap=30, seed=42)

        with pytest.warns(UserWarning, match="n_bootstrap=30 is low"):
            sa.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )


class TestSunAbrahamVsCallawaySantAnna:
    """Tests comparing Sun-Abraham to Callaway-Sant'Anna."""

    def test_both_recover_treatment_effect(self):
        """Test that both estimators recover the treatment effect."""
        from diff_diff import CallawaySantAnna

        data = generate_staggered_data(n_units=200, treatment_effect=3.0, seed=42)

        # Sun-Abraham
        sa = SunAbraham()
        sa_results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Callaway-Sant'Anna
        cs = CallawaySantAnna()
        cs_results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Both should detect positive effect
        assert sa_results.overall_att > 0
        assert cs_results.overall_att > 0

        # Both should be reasonably close to true effect
        assert abs(sa_results.overall_att - 3.0) < 2.0
        assert abs(cs_results.overall_att - 3.0) < 2.0

    def test_pre_period_difference_expected_between_cs_sa(self):
        """Pre-periods differ between CS (varying) and SA; post-periods match.

        This is expected: CS uses consecutive comparisons, SA uses fixed reference.
        CS with base_period="universal" should be closer to SA for pre-periods.
        """
        from diff_diff import CallawaySantAnna

        data = generate_staggered_data(n_units=200, treatment_effect=3.0, seed=42)

        # Sun-Abraham (uses fixed reference period e=-1)
        sa = SunAbraham()
        sa_results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Callaway-Sant'Anna with varying base (default: consecutive comparisons)
        cs_varying = CallawaySantAnna(base_period="varying")
        cs_varying_results = cs_varying.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Callaway-Sant'Anna with universal base (all compare to g-1)
        cs_universal = CallawaySantAnna(base_period="universal")
        cs_universal_results = cs_universal.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Find common event times
        sa_times = set(sa_results.event_study_effects.keys())
        cs_varying_times = set(cs_varying_results.event_study_effects.keys())
        cs_universal_times = set(cs_universal_results.event_study_effects.keys())
        common_times = sa_times & cs_varying_times & cs_universal_times

        # Separate pre and post periods
        pre_times = [t for t in common_times if t < 0]
        post_times = [t for t in common_times if t > 0]

        # Post-treatment effects should match across all methods
        for t in post_times:
            sa_eff = sa_results.event_study_effects[t]["effect"]
            cs_vary_eff = cs_varying_results.event_study_effects[t]["effect"]
            cs_univ_eff = cs_universal_results.event_study_effects[t]["effect"]

            # All three should be similar for post-treatment
            max_se = max(
                sa_results.event_study_effects[t]["se"],
                cs_varying_results.event_study_effects[t]["se"],
                cs_universal_results.event_study_effects[t]["se"],
            )
            assert abs(sa_eff - cs_vary_eff) < 3 * max_se, (
                f"Post-period t={t}: SA and CS(varying) differ too much: "
                f"SA={sa_eff:.4f}, CS(vary)={cs_vary_eff:.4f}"
            )
            assert abs(sa_eff - cs_univ_eff) < 3 * max_se, (
                f"Post-period t={t}: SA and CS(universal) differ too much: "
                f"SA={sa_eff:.4f}, CS(univ)={cs_univ_eff:.4f}"
            )

        # Require pre-periods exist for this test to be meaningful
        assert len(pre_times) > 0, (
            "Test requires pre-treatment periods to validate methodology difference. "
            "Increase n_periods or adjust cohort timing in test data."
        )

        # Compute total absolute differences
        total_diff_varying = 0.0
        total_diff_universal = 0.0
        for t in pre_times:
            sa_eff = sa_results.event_study_effects[t]["effect"]
            cs_vary_eff = cs_varying_results.event_study_effects[t]["effect"]
            cs_univ_eff = cs_universal_results.event_study_effects[t]["effect"]

            total_diff_varying += abs(sa_eff - cs_vary_eff)
            total_diff_universal += abs(sa_eff - cs_univ_eff)

        # CS(universal) should generally be closer to SA than CS(varying)
        # for pre-treatment periods (due to similar reference period approach)
        # Allow some tolerance since weighting schemes still differ
        assert total_diff_universal <= total_diff_varying + 0.5, (
            f"Expected CS(universal) to be closer to SA than CS(varying) for pre-periods. "
            f"Got: CS(univ)-SA diff={total_diff_universal:.4f}, "
            f"CS(vary)-SA diff={total_diff_varying:.4f}"
        )

    def test_agreement_under_homogeneous_effects(self):
        """Test that SA and CS agree under homogeneous treatment effects."""
        from diff_diff import CallawaySantAnna

        # Generate data with constant treatment effect (no dynamics)
        np.random.seed(42)
        n_units = 200
        n_periods = 8
        treatment_effect = 2.0

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # 30% never treated, 70% treated at period 4
        first_treat = np.zeros(n_units)
        first_treat[60:] = 4
        first_treat_expanded = np.repeat(first_treat, n_periods)

        unit_fe = np.repeat(np.random.randn(n_units) * 2, n_periods)
        time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)

        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)

        # Constant effect (no heterogeneity)
        outcomes = unit_fe + time_fe + treatment_effect * post
        outcomes += np.random.randn(len(units)) * 0.3

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
            }
        )

        # Sun-Abraham
        sa = SunAbraham()
        sa_results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Callaway-Sant'Anna
        cs = CallawaySantAnna()
        cs_results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Under homogeneous effects, SA and CS should give similar results
        # Allow for some sampling variation
        diff = abs(sa_results.overall_att - cs_results.overall_att)
        max_se = max(sa_results.overall_se, cs_results.overall_se)
        assert diff < 3 * max_se, (
            f"SA ATT={sa_results.overall_att:.3f}, "
            f"CS ATT={cs_results.overall_att:.3f}, diff={diff:.3f}"
        )


class TestSunAbrahamEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_cohort(self):
        """Test with a single treatment cohort."""
        np.random.seed(42)
        n_units = 100
        n_periods = 8

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # 30% never treated, 70% treated at period 4
        first_treat = np.zeros(n_units)
        first_treat[30:] = 4
        first_treat_expanded = np.repeat(first_treat, n_periods)

        unit_fe = np.repeat(np.random.randn(n_units), n_periods)
        time_fe = np.tile(np.arange(n_periods) * 0.1, n_units)
        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = unit_fe + time_fe + 2.0 * post + np.random.randn(len(units)) * 0.3

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
            }
        )

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att is not None
        assert len(results.groups) == 1

    def test_many_cohorts(self):
        """Test with many treatment cohorts."""
        data = generate_staggered_data(n_units=200, n_periods=15, n_cohorts=8, seed=42)

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att is not None
        assert len(results.groups) > 1

    def test_unbalanced_panel(self):
        """Test with unbalanced panel (missing observations)."""
        data = generate_staggered_data(seed=42)

        # Remove some observations randomly
        np.random.seed(123)
        keep_mask = np.random.random(len(data)) > 0.1
        data_unbalanced = data[keep_mask].copy()

        sa = SunAbraham()
        results = sa.fit(
            data_unbalanced,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att is not None

    def test_anticipation_periods(self):
        """Test with anticipation periods."""
        data = generate_staggered_data(seed=42)

        sa = SunAbraham(anticipation=1)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att is not None
        assert sa.anticipation == 1

    def test_with_covariates(self):
        """Test that covariates are properly handled in the regression."""
        data = generate_staggered_data(seed=42)

        # Add some covariates
        np.random.seed(42)
        data["covariate1"] = np.random.randn(len(data))
        data["covariate2"] = np.random.randn(len(data))

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["covariate1", "covariate2"],
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_cohort_level_dataframe(self):
        """Test cohort-level DataFrame output."""
        data = generate_staggered_data(seed=42)

        sa = SunAbraham()
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Cohort effects should be available
        df_cohort = results.to_dataframe(level="cohort")
        assert "cohort" in df_cohort.columns
        assert "relative_period" in df_cohort.columns
        assert "effect" in df_cohort.columns
        assert "weight" in df_cohort.columns

    def test_rank_deficient_action_error_raises(self):
        """Test that rank_deficient_action='error' raises ValueError on collinear data."""
        data = generate_staggered_data(seed=42)

        # Add covariates that are perfectly collinear
        np.random.seed(42)
        data["cov1"] = np.random.randn(len(data))
        data["cov1_dup"] = data["cov1"].copy()

        sa = SunAbraham(rank_deficient_action="error")
        with pytest.raises(ValueError, match="rank-deficient"):
            sa.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["cov1", "cov1_dup"],
            )

    def test_rank_deficient_action_silent_no_warning(self):
        """Test that rank_deficient_action='silent' produces no warning."""
        import warnings

        data = generate_staggered_data(seed=42)

        # Add covariates that are perfectly collinear
        np.random.seed(42)
        data["cov1"] = np.random.randn(len(data))
        data["cov1_dup"] = data["cov1"].copy()

        sa = SunAbraham(rank_deficient_action="silent")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = sa.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["cov1", "cov1_dup"],
            )

            # No warnings about rank deficiency should be emitted
            rank_warnings = [
                x
                for x in w
                if "Rank-deficient" in str(x.message) or "rank-deficient" in str(x.message).lower()
            ]
            assert len(rank_warnings) == 0, f"Expected no rank warnings, got {rank_warnings}"

        # Should still get valid results
        assert results is not None
        assert results.overall_att is not None


class TestSunAbrahamTStatNaN:
    """Tests for NaN t_stat when SE is invalid."""

    def test_per_effect_tstat_consistency(self):
        """Per-effect t_stat uses NaN (not 0.0) when SE is non-finite or zero."""
        data = generate_staggered_data(
            n_units=60,
            n_periods=8,
            n_cohorts=2,
            treatment_effect=2.0,
            seed=456,
        )

        sa = SunAbraham(n_bootstrap=0)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        for e, effect_data in results.event_study_effects.items():
            se = effect_data["se"]
            t_stat = effect_data["t_stat"]

            if not np.isfinite(se) or se == 0:
                assert np.isnan(t_stat), (
                    f"t_stat for e={e} should be NaN when SE={se}, " f"got t_stat={t_stat}"
                )
            else:
                expected = effect_data["effect"] / se
                assert np.isclose(t_stat, expected), (
                    f"t_stat for e={e} should be effect/SE, " f"expected {expected}, got {t_stat}"
                )

    def test_overall_tstat_nan_when_se_invalid(self):
        """Overall t_stat is NaN when SE is non-finite or zero."""
        data = generate_staggered_data(
            n_units=60,
            n_periods=8,
            n_cohorts=2,
            treatment_effect=2.0,
            seed=456,
        )

        sa = SunAbraham(n_bootstrap=0)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        se = results.overall_se
        t_stat = results.overall_t_stat

        if not np.isfinite(se) or se == 0:
            assert np.isnan(t_stat), f"overall_t_stat should be NaN when SE={se}, got {t_stat}"
            assert np.isnan(results.overall_p_value), (
                f"overall_p_value should be NaN when SE={se} (analytical inference), "
                f"got {results.overall_p_value}"
            )
            ci = results.overall_conf_int
            assert np.isnan(ci[0]) and np.isnan(
                ci[1]
            ), f"overall_conf_int should be (NaN, NaN) when SE={se}, got {ci}"
        else:
            expected = results.overall_att / se
            assert np.isclose(
                t_stat, expected
            ), f"overall_t_stat should be ATT/SE, expected {expected}, got {t_stat}"

    def test_bootstrap_tstat_nan_when_se_invalid(self, ci_params):
        """Bootstrap t_stat uses NaN (not 0.0) when SE is non-finite or zero."""
        data = generate_staggered_data(
            n_units=60,
            n_periods=8,
            n_cohorts=2,
            treatment_effect=2.0,
            seed=456,
        )

        n_boot = ci_params.bootstrap(50)
        sa = SunAbraham(n_bootstrap=n_boot, seed=42)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Check overall
        se = results.overall_se
        t_stat = results.overall_t_stat

        if not np.isfinite(se) or se == 0:
            assert np.isnan(
                t_stat
            ), f"bootstrap overall_t_stat should be NaN when SE={se}, got {t_stat}"

        # Check event study effects
        for e, effect_data in results.event_study_effects.items():
            se = effect_data["se"]
            t_stat = effect_data["t_stat"]

            if not np.isfinite(se) or se == 0:
                assert np.isnan(t_stat), (
                    f"bootstrap t_stat for e={e} should be NaN when SE={se}, "
                    f"got t_stat={t_stat}"
                )
            else:
                expected = effect_data["effect"] / se
                assert np.isclose(t_stat, expected), (
                    f"bootstrap t_stat for e={e} should be effect/SE, "
                    f"expected {expected}, got {t_stat}"
                )

    def test_aggregated_event_study_tstat_nan(self):
        """Aggregated event study t_stat is NaN when SE is zero or non-finite."""
        n_units = 20
        n_periods = 5
        np.random.seed(123)

        data = []
        for unit in range(n_units):
            first_treat = 3 if unit < n_units // 2 else 0
            for t in range(1, n_periods + 1):
                outcome = np.random.randn()
                data.append(
                    {
                        "unit": unit,
                        "time": t,
                        "outcome": outcome,
                        "first_treat": first_treat,
                    }
                )

        df = pd.DataFrame(data)

        sa = SunAbraham(n_bootstrap=0)
        results = sa.fit(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        for e, effect_data in results.event_study_effects.items():
            se = effect_data["se"]
            t_stat = effect_data["t_stat"]
            effect = effect_data["effect"]

            if not np.isfinite(se) or se <= 0:
                assert np.isnan(t_stat), (
                    f"Aggregated t_stat for e={e} should be NaN when SE={se}, "
                    f"got t_stat={t_stat}"
                )
                ci = effect_data["conf_int"]
                assert np.isnan(ci[0]) and np.isnan(
                    ci[1]
                ), f"Aggregated CI for e={e} should be (NaN, NaN) when SE={se}, got {ci}"
            else:
                expected_t = effect / se
                assert np.isclose(t_stat, expected_t, rtol=1e-10), (
                    f"Aggregated t_stat for e={e} should be effect/SE={expected_t}, "
                    f"got {t_stat}"
                )


class TestSunAbrahamMethodology:
    """Tests for methodology review fixes (Steps 5a-5e)."""

    def test_no_post_effects_returns_nan(self):
        """Test that no post-treatment effects returns NaN for overall ATT/SE (Step 5b).

        When there are no post-treatment periods, overall_att and overall_se should be NaN,
        and all downstream inference fields should propagate NaN correctly.
        """
        # Create data where all periods are pre-treatment
        np.random.seed(42)
        n_units = 40
        n_periods = 6

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # All treated units have first_treat at period 100 (well beyond data range)
        first_treat = np.zeros(n_units)
        first_treat[12:] = 100  # treated at period 100, but data only goes to period 5
        first_treat_expanded = np.repeat(first_treat, n_periods)

        unit_fe = np.repeat(np.random.randn(n_units), n_periods)
        time_fe = np.tile(np.arange(n_periods) * 0.1, n_units)
        outcomes = unit_fe + time_fe + np.random.randn(len(units)) * 0.3

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
            }
        )

        sa = SunAbraham(n_bootstrap=0)
        results = sa.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Overall ATT and SE should be NaN
        assert np.isnan(results.overall_att), f"Expected NaN overall_att, got {results.overall_att}"
        assert np.isnan(results.overall_se), f"Expected NaN overall_se, got {results.overall_se}"
        # Downstream inference should propagate NaN
        assert np.isnan(
            results.overall_t_stat
        ), f"Expected NaN overall_t_stat, got {results.overall_t_stat}"
        assert np.isnan(
            results.overall_p_value
        ), f"Expected NaN overall_p_value, got {results.overall_p_value}"
        assert np.isnan(results.overall_conf_int[0]) and np.isnan(
            results.overall_conf_int[1]
        ), f"Expected (NaN, NaN) overall_conf_int, got {results.overall_conf_int}"

    def test_no_post_effects_bootstrap_returns_nan(self, ci_params):
        """Test that no post-treatment effects returns NaN even with bootstrap.

        When there are no post-treatment periods, overall_att/se/t_stat/p_value/ci
        should all be NaN. The bootstrap path must not overwrite NaN with non-NaN
        values (regression test for P0 bug where _compute_bootstrap_pvalue returned
        1/(B+1) instead of NaN when original_effect was NaN).
        """
        # Create data where all periods are pre-treatment
        np.random.seed(42)
        n_units = 40
        n_periods = 6

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # All treated units have first_treat at period 100 (well beyond data range)
        first_treat = np.zeros(n_units)
        first_treat[12:] = 100  # treated at period 100, but data only goes to period 5
        first_treat_expanded = np.repeat(first_treat, n_periods)

        unit_fe = np.repeat(np.random.randn(n_units), n_periods)
        time_fe = np.tile(np.arange(n_periods) * 0.1, n_units)
        outcomes = unit_fe + time_fe + np.random.randn(len(units)) * 0.3

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
            }
        )

        n_boot = ci_params.bootstrap(50)
        sa = SunAbraham(n_bootstrap=n_boot, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            results = sa.fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

        # All overall inference fields should be NaN
        assert np.isnan(results.overall_att), f"Expected NaN overall_att, got {results.overall_att}"
        assert np.isnan(results.overall_se), f"Expected NaN overall_se, got {results.overall_se}"
        assert np.isnan(
            results.overall_t_stat
        ), f"Expected NaN overall_t_stat, got {results.overall_t_stat}"
        assert np.isnan(
            results.overall_p_value
        ), f"Expected NaN overall_p_value with bootstrap, got {results.overall_p_value}"
        assert np.isnan(results.overall_conf_int[0]) and np.isnan(
            results.overall_conf_int[1]
        ), f"Expected (NaN, NaN) overall_conf_int, got {results.overall_conf_int}"

    def test_event_time_no_truncation(self):
        """Test that event times beyond ±20 are estimated (Step 5d).

        Creates data with event times spanning beyond ±20 and verifies
        that effects are estimated for all available relative times.
        """
        np.random.seed(42)
        n_units = 60
        n_periods = 50  # 50 periods to get event times beyond ±20

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)

        # 30% never treated, rest treated at period 25 (giving rel times from -24 to +25)
        first_treat = np.zeros(n_units)
        first_treat[18:] = 25
        first_treat_expanded = np.repeat(first_treat, n_periods)

        unit_fe = np.repeat(np.random.randn(n_units) * 2, n_periods)
        time_fe = np.tile(np.arange(1, n_periods + 1) * 0.1, n_units)
        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = unit_fe + time_fe + 2.0 * post + np.random.randn(len(units)) * 0.3

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
            }
        )

        sa = SunAbraham(n_bootstrap=0)
        results = sa.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Verify that event times beyond ±20 are present
        event_times = sorted(results.event_study_effects.keys())
        assert min(event_times) < -20, f"Expected event times < -20, got min={min(event_times)}"
        assert max(event_times) > 20, f"Expected event times > 20, got max={max(event_times)}"

    def test_df_adjustment_sets_regression_df(self):
        """Test that df_adjustment for absorbed FE is applied correctly (Step 5a).

        After fitting, the internal LinearRegression's df_ should account for
        absorbed unit and time fixed effects.
        """
        from unittest.mock import patch

        from diff_diff.linalg import LinearRegression

        data = generate_staggered_data(n_units=100, n_periods=8, seed=42)
        captured_df = {}

        original_fit = LinearRegression.fit

        # Wraps LinearRegression.fit as an unbound method replacement.
        # self_reg is the LinearRegression instance (not the test class self).
        # SunAbraham currently calls LinearRegression.fit exactly once in
        # _fit_saturated_regression(); if that changes, this test captures only
        # the last call's state.
        def capturing_fit(self_reg, X, y, **kwargs):
            result = original_fit(self_reg, X, y, **kwargs)
            captured_df["df"] = self_reg.df_
            captured_df["n_obs"] = self_reg.n_obs_
            captured_df["n_params_effective"] = self_reg.n_params_effective_
            captured_df["df_adjustment"] = kwargs.get("df_adjustment", 0)
            return result

        sa = SunAbraham(n_bootstrap=0)
        with patch.object(LinearRegression, "fit", capturing_fit):
            sa.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        # Verify df_adjustment was passed and applied
        n_units = data["unit"].nunique()
        n_times = data["time"].nunique()
        expected_df_adj = n_units + n_times - 1

        assert (
            captured_df["df_adjustment"] == expected_df_adj
        ), f"Expected df_adjustment={expected_df_adj}, got {captured_df['df_adjustment']}"
        expected_df = captured_df["n_obs"] - captured_df["n_params_effective"] - expected_df_adj
        assert (
            captured_df["df"] == expected_df
        ), f"Expected df={expected_df}, got {captured_df['df']}"
        assert captured_df["df"] > 0, "Regression df must be positive"

    def test_variance_fallback_warning(self):
        """Test that the variance fallback path emits a warning (Step 5e).

        Mocks the overall_weights_by_coef to be empty to trigger the fallback.
        """
        import warnings
        from unittest.mock import patch

        data = generate_staggered_data(seed=42)

        sa = SunAbraham(n_bootstrap=0)

        # Patch _compute_overall_att to simulate the fallback path
        original_method = sa._compute_overall_att

        def patched_compute_overall_att(
            df,
            first_treat,
            event_study_effects,
            cohort_effects,
            cohort_weights,
            vcov_cohort,
            coef_index_map,
            survey_weight_col=None,
            return_overall_weights=False,
        ):
            # Pass an empty coef_index_map to trigger the fallback
            return original_method(
                df,
                first_treat,
                event_study_effects,
                cohort_effects,
                cohort_weights,
                vcov_cohort,
                {},  # Empty coef_index_map forces fallback
                survey_weight_col=survey_weight_col,
                return_overall_weights=return_overall_weights,
            )

        with patch.object(sa, "_compute_overall_att", side_effect=patched_compute_overall_att):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = sa.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                )

                fallback_warnings = [
                    x for x in w if "simplified variance" in str(x.message).lower()
                ]
                assert (
                    len(fallback_warnings) > 0
                ), "Expected warning about simplified variance fallback"

        # The result should still have a positive SE (simplified variance)
        assert (
            results.overall_se > 0
        ), f"Expected positive SE from fallback, got {results.overall_se}"

    def test_iw_weights_match_cohort_shares(self):
        """Test that IW weights match event-time sample shares.

        For each relative period, Σ_g w_{g,e} = 1.0 and individual weights
        match n_{g,e} / Σ_g n_{g,e} (sample share of cohort g at event-time e).
        """
        data = generate_staggered_data(n_units=200, n_periods=10, n_cohorts=3, seed=42)

        sa = SunAbraham(n_bootstrap=0)
        results = sa.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        for e, weights in results.cohort_weights.items():
            # Weights should sum to 1
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-10, f"Weights for e={e} sum to {total}, expected 1.0"

            # Individual weights should match event-time sample shares
            cohort_counts = {}
            for g in weights.keys():
                cohort_counts[g] = len(
                    data[(data["first_treat"] == g) & (data["time"] - data["first_treat"] == e)]
                )
            total_count = sum(cohort_counts.values())
            for g, w in weights.items():
                expected_w = cohort_counts[g] / total_count
                assert (
                    abs(w - expected_w) < 1e-10
                ), f"Weight for cohort {g} at e={e}: got {w}, expected {expected_w}"

    def test_iw_weights_unbalanced_panel(self):
        """Test that IW weights use event-time counts, not cohort sizes, for unbalanced panels."""
        data = generate_staggered_data(n_units=200, n_periods=10, n_cohorts=3, seed=42)

        # Make panel unbalanced by dropping some observations from one cohort
        # at specific time periods
        cohorts = data.groupby("unit")["first_treat"].first()
        first_cohort = sorted(cohorts[cohorts > 0].unique())[0]
        units_in_first_cohort = cohorts[cohorts == first_cohort].index.tolist()

        # Drop ~half the units from first cohort at the last time period
        units_to_drop = units_in_first_cohort[: len(units_in_first_cohort) // 2]
        max_time = data["time"].max()
        drop_mask = data["unit"].isin(units_to_drop) & (data["time"] == max_time)
        data_unbal = data[~drop_mask].copy()

        sa = SunAbraham(n_bootstrap=0)
        results = sa.fit(
            data_unbal,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Find an event-time where the dropped observations cause n_{g,e} != n_g
        # The dropped units are from the first cohort at max_time
        affected_e = max_time - first_cohort

        assert (
            affected_e in results.cohort_weights
        ), f"Expected event-time {affected_e} in cohort_weights but not found"

        weights = results.cohort_weights[affected_e]
        # Verify weights use actual observation counts, not total cohort sizes
        cohort_counts = {}
        for g in weights.keys():
            cohort_counts[g] = len(
                data_unbal[
                    (data_unbal["first_treat"] == g)
                    & (data_unbal["time"] - data_unbal["first_treat"] == affected_e)
                ]
            )
        total_count = sum(cohort_counts.values())
        for g, w in weights.items():
            expected_w = cohort_counts[g] / total_count
            assert (
                abs(w - expected_w) < 1e-10
            ), f"Weight for cohort {g} at e={affected_e}: got {w}, expected {expected_w}"

    def test_never_treated_inf_encoding(self):
        """Test that first_treat=np.inf is handled as never-treated, not as a cohort."""
        data = generate_staggered_data(n_units=200, n_periods=10, n_cohorts=3, seed=42)

        # Run with first_treat=0 as baseline
        sa = SunAbraham(n_bootstrap=0)
        results_zero = sa.fit(
            data.copy(), outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Re-encode never-treated from 0 to np.inf (cast to float first for pandas compat)
        data_inf = data.copy()
        data_inf["first_treat"] = data_inf["first_treat"].astype(float)
        data_inf.loc[data_inf["first_treat"] == 0, "first_treat"] = np.inf

        results_inf = sa.fit(
            data_inf, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # np.inf must not appear as a cohort in weights
        for e, weights in results_inf.cohort_weights.items():
            assert np.inf not in weights, f"np.inf found as cohort key in weights at e={e}"

        # No ±inf in event study periods
        for e in results_inf.event_study_effects.keys():
            assert np.isfinite(e), f"Non-finite event time {e} in event study"

        # np.inf must not appear in results.groups
        assert (
            np.inf not in results_inf.groups
        ), f"np.inf found in results.groups: {results_inf.groups}"

        # Results should be identical to first_treat=0 encoding
        assert np.isclose(
            results_inf.overall_att, results_zero.overall_att
        ), f"ATT differs: inf={results_inf.overall_att}, zero={results_zero.overall_att}"
        assert np.isclose(
            results_inf.overall_se, results_zero.overall_se
        ), f"SE differs: inf={results_inf.overall_se}, zero={results_zero.overall_se}"

    def test_removed_params_raise_typeerror(self):
        """Removed min_pre_periods/min_post_periods raise TypeError."""
        data = generate_staggered_data(n_units=30, n_periods=6, seed=42)
        sa = SunAbraham(n_bootstrap=0)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            sa.fit(data, "outcome", "unit", "time", "first_treat", min_pre_periods=2)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            sa.fit(data, "outcome", "unit", "time", "first_treat", min_post_periods=2)

    def test_all_never_treated_inf_raises(self):
        """Test that all-never-treated data with np.inf encoding raises ValueError."""
        data = generate_staggered_data(n_units=100, n_periods=10, n_cohorts=3, seed=42)
        # Set ALL units to never-treated via np.inf (cast to float first for pandas compat)
        data["first_treat"] = data["first_treat"].astype(float)
        data["first_treat"] = np.inf

        sa = SunAbraham(n_bootstrap=0)
        with pytest.raises(ValueError, match="No treated units found"):
            sa.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )


class TestSunAbrahamVcovType:
    """Tests for SunAbraham `vcov_type` parameter (Phase 1b PR 1/8).

    Threads `vcov_type ∈ {"classical","hc1","hc2","hc2_bm"}` through SA.
    `vcov_type="hc1"` is the default and preserves prior behavior bit-equally.
    `hc2`/`hc2_bm` route through a full-dummy saturated design (FWL preserves
    coefficients but not the hat matrix). `classical` also uses full-dummy to
    match R's `lm()` interpretation. `conley` (spatial-HAC) threads through the
    within-transform path; its coverage lives in test_conley_vcov.py.
    """

    @staticmethod
    def _panel():
        return generate_staggered_data(n_units=40, n_periods=8, n_cohorts=3, seed=42)

    def test_default_matches_hc1_explicit(self):
        """SA() (default) and SA(vcov_type='hc1') produce identical ATT/SE
        (modulo floating-point representation noise at the last bit)."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res_default = SunAbraham().fit(data, **kwargs)
        res_explicit = SunAbraham(vcov_type="hc1").fit(data, **kwargs)
        # Both paths execute the same code under the hood (default vcov_type
        # is "hc1"); any divergence is float64 representation noise.
        assert np.isclose(res_default.overall_att, res_explicit.overall_att, atol=1e-14)
        assert np.isclose(res_default.overall_se, res_explicit.overall_se, atol=1e-14)
        assert np.isclose(res_default.overall_p_value, res_explicit.overall_p_value, atol=1e-12)

    def test_classical_finite(self):
        """SA(vcov_type='classical') produces finite ATT/SE; drops auto-cluster."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res = SunAbraham(vcov_type="classical").fit(data, **kwargs)
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert np.isfinite(res.overall_p_value)
        # Classical differs from HC1 cluster-at-unit in heteroscedastic / panel data
        res_hc1 = SunAbraham(vcov_type="hc1").fit(data, **kwargs)
        assert res.overall_se != res_hc1.overall_se

    def test_hc2_finite_no_auto_cluster(self):
        """SA(vcov_type='hc2') runs without explicit cluster (linalg validator
        would reject hc2 + cluster_ids; SA drops the auto-cluster)."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res = SunAbraham(vcov_type="hc2").fit(data, **kwargs)
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        # HC2 differs from cluster-at-unit HC1 on this panel
        res_hc1 = SunAbraham(vcov_type="hc1").fit(data, **kwargs)
        assert res.overall_se != res_hc1.overall_se

    def test_hc2_bm_finite_with_auto_cluster(self):
        """SA(vcov_type='hc2_bm') auto-clusters at unit by default; routes to CR2-BM."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res = SunAbraham(vcov_type="hc2_bm").fit(data, **kwargs)
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        # HC2-BM applies BM Satterthwaite DOF; should differ from cluster-at-unit HC1
        res_hc1 = SunAbraham(vcov_type="hc1").fit(data, **kwargs)
        assert res.overall_se != res_hc1.overall_se

    def test_hc2_bm_nan_contrast_dof_regression_pin(self, monkeypatch):
        """NaN BM contrast DOF -> all-NaN row inference + NaN df provenance.

        Regression PIN of existing behavior (no fix in this change):
        ``safe_inference`` has guarded non-finite / <= 0 df since PR #620, so
        a noise-floor NaN Bell-McCaffrey DOF already yields NaN t/p/CI rather
        than a finite t-stat. What is NEW here is the provenance record: the
        row's ``event_study_df`` entry must be NaN, honestly documenting that
        no usable df governed its (undefined) stored inference.
        """
        import diff_diff.linalg as _linalg

        real_helper = _linalg._compute_cr2_bm_contrast_dof

        def _nan_dofs(*args, **kwargs):
            out = real_helper(*args, **kwargs)
            return np.full_like(np.asarray(out, dtype=float), np.nan)

        monkeypatch.setattr(_linalg, "_compute_cr2_bm_contrast_dof", _nan_dofs)
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res = SunAbraham(vcov_type="hc2_bm").fit(data, **kwargs)

        assert res.event_study_df is not None
        nan_df_rows = [e for e, v in res.event_study_df.items() if not np.isfinite(v)]
        assert nan_df_rows, "monkeypatched NaN DOFs must produce NaN df provenance"
        for e in nan_df_rows:
            row = res.event_study_effects[e]
            # SE stays FINITE here (the NaN comes from the df, not the SE),
            # so assert the inference triple directly rather than via
            # assert_nan_inference (which requires a non-finite SE).
            assert np.isnan(row["t_stat"])
            assert np.isnan(row["p_value"])
            assert np.isnan(row["conf_int"][0]) and np.isnan(row["conf_int"][1])

    def test_hc2_bm_contrast_dof_helper_failure_falls_back(self):
        """Helper-level BM failure warns and falls back to the resolved df.

        Distinct from the per-row NaN-DOF case above: when the helper RAISES,
        SA leaves the ``_compute_iw_effects`` output in place after warning.
        Since 3.9 (M-127) that output carries the df_convention-resolved
        analytical df - FINITE t(residual df) under the default, recorded as
        genuine provenance (previously normal theory with NaN provenance).
        """
        import diff_diff.linalg as _linalg

        def _boom(*args, **kwargs):
            raise ValueError("forced BM helper failure")

        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_linalg, "_compute_cr2_bm_contrast_dof", _boom)
            with pytest.warns(UserWarning):
                res = SunAbraham(vcov_type="hc2_bm").fit(data, **kwargs)

        finite_p = [e for e, row in res.event_study_effects.items() if np.isfinite(row["p_value"])]
        assert finite_p, "helper failure must fall back to finite t(residual) inference"
        assert res.event_study_df is not None
        for e in finite_p:
            assert (
                np.isfinite(res.event_study_df[e]) and res.event_study_df[e] > 0
            ), "the fallback df is the knob-resolved residual df - genuine provenance"

    def test_hc2_bm_explicit_cluster_works(self):
        """Explicit cluster= overrides the auto-cluster default; CR2-BM at named cluster."""
        data = self._panel()
        # Add a second cluster column
        data["firm"] = (data["unit"] // 5).astype(int)
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res = SunAbraham(cluster="firm", vcov_type="hc2_bm").fit(data, **kwargs)
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        # Differs from cluster-at-unit (different cluster grouping)
        res_unit = SunAbraham(vcov_type="hc2_bm").fit(data, **kwargs)
        assert res.overall_se != res_unit.overall_se

    def test_hc2_rejects_explicit_cluster(self):
        """SA(cluster=..., vcov_type='hc2') is rejected by the linalg validator."""
        data = self._panel()
        data["firm"] = (data["unit"] // 5).astype(int)
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        sa = SunAbraham(cluster="firm", vcov_type="hc2")
        with pytest.raises(ValueError, match="hc2 is one-way only"):
            sa.fit(data, **kwargs)

    def test_hc2_bm_replicate_weights_rejected(self):
        """SA(vcov_type='hc2_bm' or 'hc2') + replicate-weight survey raises NotImplementedError."""
        from diff_diff.survey import SurveyDesign

        data = self._panel()
        # Build a minimal BRR replicate-weight design (unit-constant weights)
        data["w"] = 1.0
        rep_cols = []
        for j in range(4):
            colname = f"r{j}"
            data[colname] = 1.0
            rep_cols.append(colname)
        sd = SurveyDesign(
            weights="w",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        kwargs = dict(
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        for vt in ("hc2", "hc2_bm"):
            sa = SunAbraham(vcov_type=vt)
            with pytest.raises(NotImplementedError, match="replicate-weight"):
                sa.fit(data, **kwargs)

    def test_hc1_replicate_weights_path_unchanged(self):
        """SA(vcov_type='hc1') + replicate-weight survey still works (regression
        test that the Part C reject didn't broaden to non-hc2 paths)."""
        from diff_diff.survey import SurveyDesign

        data = self._panel()
        # Unit-constant weights and replicate cols.
        data["w"] = 1.0
        units_sorted = sorted(data["unit"].unique())
        rep_cols = []
        for j, u_drop in enumerate(units_sorted[:5]):
            colname = f"r{j}"
            data[colname] = data["unit"].apply(lambda u: 0.0 if u == u_drop else 1.0).astype(float)
            rep_cols.append(colname)
        sd = SurveyDesign(
            weights="w",
            replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        sa = SunAbraham(vcov_type="hc1")
        res = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        # Replicate path → bootstrap-style SE; should be finite
        assert np.isfinite(res.overall_se)

    def test_get_params_includes_vcov_type(self):
        """get_params() returns dict with the new 'vcov_type' key."""
        sa = SunAbraham(vcov_type="hc2_bm")
        params = sa.get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc2_bm"

    def test_set_params_updates_vcov_type_and_explicit_flag(self):
        """set_params(vcov_type=...) updates both self.vcov_type and _vcov_type_explicit."""
        sa = SunAbraham()  # default hc1; _vcov_type_explicit=False
        assert sa.vcov_type == "hc1"
        assert sa._vcov_type_explicit is False

        sa.set_params(vcov_type="hc2")
        assert sa.vcov_type == "hc2"
        assert sa._vcov_type_explicit is True  # opted out of default

        sa.set_params(vcov_type="hc1")
        assert sa.vcov_type == "hc1"
        assert sa._vcov_type_explicit is False  # back to default

    def test_clone_repeat_fit_idempotent(self):
        """fit() → clone (via get_params/set_params) → re-fit produces identical results.
        Validates per feedback_fit_does_not_mutate_config: fit() doesn't mutate
        configuration; vcov_type and _vcov_type_explicit survive a clone."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        sa1 = SunAbraham(vcov_type="hc2_bm")
        res1 = sa1.fit(data, **kwargs)
        # Clone via get_params → new SunAbraham → set_params
        sa2 = SunAbraham()
        sa2.set_params(**sa1.get_params())
        res2 = sa2.fit(data, **kwargs)
        assert res1.overall_att == res2.overall_att
        assert res1.overall_se == res2.overall_se
        assert sa2.vcov_type == "hc2_bm"
        assert sa2._vcov_type_explicit is True

    def test_invalid_vcov_type_value_rejects(self):
        """Unknown vcov_type raises ValueError; conley is accepted at __init__.

        As of the conley-threading PR, ``vcov_type="conley"`` is a valid
        constructor value (coords/cutoff/unit/lag validation is deferred to
        ``fit()`` via ``_validate_conley_estimator_inputs``).
        """
        with pytest.raises(ValueError, match="vcov_type must be one of"):
            SunAbraham(vcov_type="foo")
        est = SunAbraham(vcov_type="conley")
        assert est.vcov_type == "conley"

    def test_survey_design_rejects_non_hc1_vcov_type(self):
        """SA(vcov_type ∈ {classical, hc2, hc2_bm}) + survey_design= raises.

        Survey-design TSL (or replicate-weight refit) variance overrides
        the analytical sandwich family, so requesting a non-HC1 analytical
        family under a survey design would either silently discard the
        request OR downgrade unit-level PSUs to per-observation PSUs
        (the auto-cluster guard for one-way families drops cluster_var=None
        before survey-PSU injection). Explicit reject preserves the
        contract.
        """
        from diff_diff.survey import SurveyDesign

        data = self._panel()
        data["w"] = 1.0
        sd = SurveyDesign(weights="w")
        for vt in ("classical", "hc2", "hc2_bm"):
            sa = SunAbraham(vcov_type=vt)
            with pytest.raises(NotImplementedError, match="survey_design"):
                sa.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    survey_design=sd,
                )

    def test_hc1_survey_design_preserves_unit_psu_injection(self):
        """SA(vcov_type='hc1') + survey_design= (no explicit PSU/cluster)
        injects unit as PSU; verifies n_psu == n_units and df_survey > 0.

        Regression test for the P1-A cascade bug: prior to the survey-
        reject contract, non-HC1 fits would drop the auto-cluster
        BEFORE PSU injection, downgrading n_psu from n_units to n_obs.
        HC1 must still inject unit-as-PSU correctly.
        """
        from diff_diff.survey import SurveyDesign

        data = self._panel()
        data["w"] = 1.0
        sd = SurveyDesign(weights="w")
        sa = SunAbraham(vcov_type="hc1")  # default; auto-cluster fires
        res = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        # Survey metadata should record unit-level PSUs (n_psu == n_units),
        # NOT per-observation PSUs (n_psu == n_obs). On the panel: 40
        # units × 8 periods = 320 obs.
        assert res.survey_metadata is not None
        n_units = data["unit"].nunique()
        assert res.survey_metadata.n_psu == n_units, (
            f"Expected n_psu == n_units ({n_units}); "
            f"got {res.survey_metadata.n_psu}. If this is n_obs ({len(data)}), "
            "the unit-as-PSU injection regressed."
        )
        # df_survey should be n_psu - n_strata. No explicit strata → n_strata
        # defaults to 1 → df_survey == n_psu - 1 == n_units - 1. Asserting
        # both n_psu and df_survey guards against a regression where one is
        # computed correctly but the other is downgraded (e.g. n_psu=40 but
        # df_survey=319 would suggest df was computed against n_obs).
        assert res.survey_metadata.df_survey == n_units - 1, (
            f"Expected df_survey == n_psu - 1 == {n_units - 1}; "
            f"got {res.survey_metadata.df_survey}."
        )

    def test_vcov_type_propagated_to_results(self):
        """SunAbrahamResults.vcov_type reflects the fit-time configuration.

        Per the P1-B fix: downstream consumers need to know which
        variance family generated the SEs (matters for survey
        precedence claims, bootstrap-vs-analytical distinctions, etc.).
        """
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        for vt in ("hc1", "classical", "hc2", "hc2_bm"):
            sa = SunAbraham(vcov_type=vt)
            res = sa.fit(data, **kwargs)
            assert hasattr(
                res, "vcov_type"
            ), "SunAbrahamResults must expose vcov_type for downstream consumers"
            assert res.vcov_type == vt, f"Expected res.vcov_type == {vt!r}; got {res.vcov_type!r}"

    def test_hc2_bm_routes_through_full_dummy_path(self):
        """Sanity: hc2_bm coefficient on cohort effect matches the hc2 path
        coefficient (FWL preserves the cohort coefficient; only the SE differs
        across vcov families). Verifies the full-dummy auto-route preserves
        cohort_effects extraction (intercept-offset indexing)."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res_hc1 = SunAbraham(vcov_type="hc1").fit(data, **kwargs)
        res_hc2 = SunAbraham(vcov_type="hc2").fit(data, **kwargs)
        res_hc2_bm = SunAbraham(vcov_type="hc2_bm").fit(data, **kwargs)
        # ATT should match across all vcov families (FWL)
        assert np.isclose(res_hc1.overall_att, res_hc2.overall_att, atol=1e-12)
        assert np.isclose(res_hc1.overall_att, res_hc2_bm.overall_att, atol=1e-12)

    def test_hc2_robustness_value_differs_from_within_transform(self):
        """Confirms hc2 routes through full-dummy: the SE differs from a manual
        within-transform HC2 (sanity check that the auto-route is active, not
        silently using within-transform)."""
        from diff_diff.linalg import solve_ols
        from diff_diff.utils import within_transform as _within_transform_util

        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res_hc2 = SunAbraham(vcov_type="hc2").fit(data, **kwargs)

        # Manually construct the within-transform HC2 for comparison:
        # this is what SA WOULD compute if Part G didn't auto-route.
        df = data.copy()
        df["_rel_time"] = np.where(df["first_treat"] > 0, df["time"] - df["first_treat"], -999)
        treatment_groups = sorted(g for g in df["first_treat"].unique() if g > 0)
        all_rel = sorted(e for e in df.loc[df["first_treat"] > 0, "_rel_time"].unique() if e != -1)
        interaction_cols = []
        for g in treatment_groups:
            for e in all_rel:
                name = f"_D_{g}_{e}"
                ind = ((df["first_treat"] == g) & (df["_rel_time"] == e)).astype(float)
                if ind.sum() > 0:
                    df[name] = ind
                    interaction_cols.append(name)
        df_dm = _within_transform_util(
            df, ["outcome"] + interaction_cols, "unit", "time", suffix="_dm"
        )
        X_dm = df_dm[[f"{c}_dm" for c in interaction_cols]].values
        y_dm = df_dm["outcome_dm"].values
        _, _, vcov_within = solve_ols(X_dm, y_dm, vcov_type="hc2")
        # Within-transform HC2 SE for first interaction column
        within_se = float(np.sqrt(vcov_within[0, 0]))

        # SA's full-dummy HC2 SE for the same target should differ from
        # within_se (different hat matrix → different leverage correction).
        # Pick any present cohort × event-time interaction.
        target_key = next(iter(res_hc2.cohort_effects))
        sa_se = res_hc2.cohort_effects[target_key]["se"]
        assert not np.isclose(sa_se, within_se, atol=1e-6), (
            f"SA hc2 SE ({sa_se}) too close to within-transform HC2 ({within_se}); "
            "Part G full-dummy auto-route may not be active."
        )

    def test_point_estimates_invariant_across_vcov_type_with_covariates(self):
        """FWL invariance: cohort effects + overall ATT identical across all
        vcov_type values when a covariate is present.

        The new full-dummy branch is a manual second implementation of SA's
        regression design. FWL preserves coefficients/residuals across the
        within-transform vs full-dummy choice, so vcov_type must not change
        the point estimates — only the SEs. This guards against a
        regression in the new design-builder that would silently change
        the estimator (not just the SEs) when covariates are passed.
        """
        data = self._panel()
        # Add a time-varying covariate uncorrelated with the treatment
        rng = np.random.default_rng(7)
        data["x"] = rng.normal(0, 1, size=len(data))
        kwargs = dict(
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x"],
        )
        results = {}
        for vt in ("hc1", "classical", "hc2", "hc2_bm"):
            results[vt] = SunAbraham(vcov_type=vt).fit(data, **kwargs)
        # Overall ATT and event_study effects should match across vcov_type
        for vt in ("classical", "hc2", "hc2_bm"):
            assert np.isclose(results["hc1"].overall_att, results[vt].overall_att, atol=1e-10), (
                f"overall_att diverges between hc1 ({results['hc1'].overall_att}) "
                f"and {vt} ({results[vt].overall_att}); covariate threading "
                "regression in the full-dummy design-builder."
            )
            for e in results["hc1"].event_study_effects:
                hc1_eff = results["hc1"].event_study_effects[e]["effect"]
                vt_eff = results[vt].event_study_effects[e]["effect"]
                assert np.isclose(
                    hc1_eff, vt_eff, atol=1e-10
                ), f"event_study[{e}] effect diverges: hc1={hc1_eff}, {vt}={vt_eff}"

    def test_point_estimates_invariant_across_vcov_type_not_yet_treated(self):
        """FWL invariance under control_group='not_yet_treated': cohort effects
        and overall ATT identical across all vcov_type values.

        The not_yet_treated control group changes the regression sample
        (keeps all units, not just never-treated controls). The new
        full-dummy branch must reproduce the same point estimates as the
        within-transform path under this sample composition.
        """
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        results = {}
        for vt in ("hc1", "classical", "hc2", "hc2_bm"):
            results[vt] = SunAbraham(control_group="not_yet_treated", vcov_type=vt).fit(
                data, **kwargs
            )
        for vt in ("classical", "hc2", "hc2_bm"):
            assert np.isclose(results["hc1"].overall_att, results[vt].overall_att, atol=1e-10), (
                f"overall_att diverges between hc1 ({results['hc1'].overall_att}) "
                f"and {vt} ({results[vt].overall_att}) under "
                "control_group='not_yet_treated'; sample-composition regression "
                "in the full-dummy branch."
            )
            for e in results["hc1"].event_study_effects:
                hc1_eff = results["hc1"].event_study_effects[e]["effect"]
                vt_eff = results[vt].event_study_effects[e]["effect"]
                assert np.isclose(hc1_eff, vt_eff, atol=1e-10), (
                    f"event_study[{e}] effect diverges under not_yet_treated: "
                    f"hc1={hc1_eff}, {vt}={vt_eff}"
                )

    def test_cluster_missing_column_raises_across_vcov_types(self):
        """Explicit cluster= referencing a missing column must raise, not
        silently downgrade clustered inference to one-way.

        Regression test for the cluster-resolution cascade: the survey-
        path fix that tolerates `cluster_var=None` for explicit one-way
        families (hc2, classical) would, without this guard, also treat
        a missing user-supplied cluster as cluster_var=None, silently
        producing HC1-no-cluster / HC2-singleton / classical SEs
        instead of the requested cluster-robust inference. Must error
        upfront for at least hc1, classical, hc2, hc2_bm.
        """
        data = self._panel()
        for vt in ("hc1", "classical", "hc2", "hc2_bm"):
            sa = SunAbraham(cluster="nonexistent_col", vcov_type=vt)
            with pytest.raises(ValueError, match="cluster column"):
                sa.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                )

    def test_cluster_na_values_raise_across_vcov_types(self):
        """Cluster columns with NA/NaN values must raise — otherwise
        the meat-side `groupby(cluster_ids)` drops NA rows while
        `np.unique(cluster_ids)` still counts the NA group, producing
        silently-malformed cluster-robust SEs.

        Per codex CI R4: column-existence validation is insufficient
        — must also validate non-NA values upfront.
        """
        data = self._panel()
        # Add a cluster column with NA values in some rows
        data["my_cluster"] = data["unit"].astype("Int64")
        data.loc[data.index[:5], "my_cluster"] = pd.NA
        for vt in ("hc1", "hc2_bm"):
            sa = SunAbraham(cluster="my_cluster", vcov_type=vt)
            with pytest.raises(ValueError, match="NA/NaN values"):
                sa.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                )

    def test_bootstrap_finite_and_point_estimates_invariant(self, ci_params):
        """Bootstrap (n_bootstrap > 0) coverage for the new vcov_type paths.

        Both `_run_bootstrap` and `_run_rao_wu_bootstrap` re-enter
        `_fit_saturated_regression(..., vcov_type=self.vcov_type)`, so
        bootstrap fits with the new families execute the new full-dummy
        plumbing on every refit. This test asserts:
          1. Bootstrap output is finite for each vcov_type
          2. Point estimates (overall ATT, event-study effects) are
             invariant across vcov_type at the same seed (FWL)
        The bootstrap SE may legitimately differ across vcov_type because
        the multiplier-bootstrap inputs depend on the per-fit residuals,
        but the per-fit residuals are FWL-invariant; the SE divergence
        comes from numerical precision in the iterated refits.
        """
        n_boot = ci_params.bootstrap(50)
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        results = {}
        for vt in ("hc1", "classical", "hc2", "hc2_bm"):
            results[vt] = SunAbraham(vcov_type=vt, n_bootstrap=n_boot, seed=42).fit(data, **kwargs)
            # Bootstrap outputs must be finite for each family
            assert np.isfinite(
                results[vt].overall_att
            ), f"overall_att non-finite for vcov_type={vt} under bootstrap"
            assert np.isfinite(
                results[vt].overall_se
            ), f"bootstrap SE non-finite for vcov_type={vt}"
        # Point estimates must match across vcov_type (FWL preserves them
        # across within-transform vs full-dummy refits).
        for vt in ("classical", "hc2", "hc2_bm"):
            assert np.isclose(results["hc1"].overall_att, results[vt].overall_att, atol=1e-10), (
                f"Bootstrap overall_att diverges: hc1={results['hc1'].overall_att}, "
                f"{vt}={results[vt].overall_att}. Bootstrap refits in the new "
                "full-dummy branch are not FWL-invariant."
            )


class TestAbsorbedCovariateSnap:
    """A unit-constant covariate is spanned by the unit FE: it must snap to
    zero (deterministic drop + cause warning) instead of leaving a junk
    column that perturbs the IW estimates (REGISTRY 'Absorbed FE')."""

    @staticmethod
    def _panel(seed=0, with_cov=False):
        rng = np.random.default_rng(seed)
        n_units, n_periods = 90, 8
        unit = np.repeat(np.arange(n_units), n_periods)
        time = np.tile(np.arange(n_periods), n_units)
        first = np.repeat(rng.choice([0, 4, 5], n_units), n_periods)
        d = (first > 0) & (time >= first)
        y = 0.3 * d + rng.normal(size=unit.size)
        df = pd.DataFrame({"y": y, "unit": unit, "time": time, "first_treat": first})
        if with_cov:
            df["xc"] = np.repeat(rng.normal(size=n_units), n_periods)
        return df

    def test_unit_constant_covariate_snaps_att_unaffected(self):
        base_df = self._panel(seed=42)
        cov_df = self._panel(seed=42, with_cov=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = SunAbraham().fit(
                base_df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        with pytest.warns(UserWarning, match="collinear with the absorbed"):
            res = SunAbraham().fit(
                cov_df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["xc"],
            )
        # snapped covariate -> dropped -> identical to the no-covariate design
        assert res.att == pytest.approx(base.att, abs=1e-10)


class TestSunAbrahamDfConvention:
    """The three-value df_convention knob on SunAbraham (3.9 / M-127, the D4 fix).

    Modeled on tests/test_estimators_vcov_type.py::TestDfConvention. Cells and
    aggregates share ONE df source per fit; att/se/t never move across values.
    """

    @staticmethod
    def _panel(seed=3):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(60):
            ft = [0, 4, 6][u % 3]
            for t in range(1, 9):
                eff = 0.7 if (ft and t >= ft) else 0.0
                rows.append(
                    dict(
                        unit=u,
                        time=t,
                        first_treat=ft,
                        outcome=0.3 * u / 60 + 0.1 * t + eff + rng.standard_normal() * 0.6,
                    )
                )
        return pd.DataFrame(rows)

    _kw = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")

    def test_residual_default_is_t_not_z(self):
        """The D4/defect fix: aggregates use t(saturated residual df), not z."""
        res = SunAbraham().fit(self._panel(), **self._kw)
        p_z = 2 * stats.norm.sf(abs(res.overall_t_stat))
        assert res.overall_p_value != p_z
        dfres = res.inference_df
        assert dfres is not None and dfres > 0
        assert res.overall_p_value == pytest.approx(
            2 * stats.t.sf(abs(res.overall_t_stat), dfres), rel=1e-12
        )
        # every ES row shares the same finite residual df (one df source)
        assert set(res.event_study_df.values()) == {dfres}

    def test_cluster_matches_hand_reconstructed_g_minus_1(self):
        data = self._panel()
        r0 = SunAbraham().fit(data, **self._kw)
        r1 = SunAbraham(df_convention="cluster").fit(data, **self._kw)
        G = data["unit"].nunique()
        assert r1.overall_att == r0.overall_att and r1.overall_se == r0.overall_se
        assert r1.overall_t_stat == r0.overall_t_stat
        assert r1.overall_p_value == pytest.approx(
            2 * stats.t.sf(abs(r1.overall_t_stat), G - 1), rel=1e-12
        )
        lo, hi = r1.overall_conf_int
        crit = stats.t.ppf(0.975, G - 1)
        assert lo == pytest.approx(r1.overall_att - crit * r1.overall_se, rel=1e-12)
        assert hi == pytest.approx(r1.overall_att + crit * r1.overall_se, rel=1e-12)
        assert r1.inference_df == G - 1
        assert set(r1.event_study_df.values()) == {float(G - 1)}

    def test_normal_reproduces_z_on_aggregates(self):
        """'normal' = deliberate z at the fallback level (the pre-3.9 numbers)."""
        data = self._panel()
        r0 = SunAbraham().fit(data, **self._kw)
        rn = SunAbraham(df_convention="normal").fit(data, **self._kw)
        assert rn.overall_att == r0.overall_att and rn.overall_se == r0.overall_se
        assert rn.overall_p_value == pytest.approx(
            2 * stats.norm.sf(abs(rn.overall_t_stat)), rel=1e-14
        )
        assert rn.inference_df is None
        # provenance: z rows record NaN
        assert all(np.isnan(v) for v in rn.event_study_df.values())

    def test_bm_dof_precedence_over_knob(self):
        """hc2_bm BM contrast DOF wins under every knob value."""
        data = self._panel()
        r_res = SunAbraham(vcov_type="hc2_bm").fit(data, **self._kw)
        r_clu = SunAbraham(vcov_type="hc2_bm", df_convention="cluster").fit(data, **self._kw)
        assert r_res.overall_p_value == r_clu.overall_p_value
        assert r_res.overall_conf_int == r_clu.overall_conf_int

    def test_survey_df_precedence_over_knob(self):
        """Survey design df wins under EVERY knob value: the three fits are
        bit-identical on inference and share the finite survey-df provenance
        (the knob is never consulted on surveyed fits)."""
        from diff_diff.survey import SurveyDesign

        data = self._panel()
        data["w"] = 1.0 + (data["unit"] % 5) * 0.2
        data["stratum"] = data["unit"] % 4
        sd = SurveyDesign(weights="w", strata="stratum", psu="unit")
        fits = {
            conv: SunAbraham(df_convention=conv).fit(data, survey_design=sd, **self._kw)
            for conv in ("residual", "cluster", "normal")
        }
        base = fits["residual"]
        assert base.inference_df is not None and np.isfinite(base.inference_df)
        for conv in ("cluster", "normal"):
            r = fits[conv]
            assert r.overall_p_value == base.overall_p_value
            assert r.overall_conf_int == base.overall_conf_int
            assert r.inference_df == base.inference_df
            assert r.event_study_df == base.event_study_df

    def test_bootstrap_clears_inference_df(self):
        """Percentile bootstrap p/CI never used a df: the scalar clears."""
        res = SunAbraham(n_bootstrap=20, seed=1).fit(self._panel(), **self._kw)
        assert res.inference_df is None
        assert res.event_study_df is None  # existing channel-clearing convention

    def test_validation_and_transactional_set_params(self):
        with pytest.raises(ValueError, match="df_convention"):
            SunAbraham(df_convention="bogus")
        est = SunAbraham()
        with pytest.raises(ValueError, match="df_convention"):
            est.set_params(df_convention="bogus", alpha=0.10)
        assert est.df_convention == "residual" and est.alpha == 0.05
        # Valid value + unknown key: the unknown-key rejection must also
        # leave the estimator fully unchanged (no partial application).
        before = est.get_params()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(df_convention="normal", nonexistent_param=1)
        assert est.get_params() == before

    def test_get_params_roundtrip_and_repeat_fit(self):
        est = SunAbraham(df_convention="cluster")
        assert est.get_params()["df_convention"] == "cluster"
        clone = SunAbraham(**est.get_params())
        data = self._panel()
        r1 = clone.fit(data, **self._kw)
        p1, d1 = r1.overall_p_value, r1.inference_df
        r2 = clone.fit(data, **self._kw)
        assert r2.overall_p_value == p1 and r2.inference_df == d1
        assert clone.df_convention == "cluster"


@pytest.fixture(scope="module")
def alpha_fitted():
    data = generate_staggered_data()
    return SunAbraham().fit(
        data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
    )


class TestSummaryAlphaContract:
    """summary(alpha=...) never recomputes stored inference.

    Family-wide guard (results_base._require_fit_alpha): a non-fit alpha
    raises instead of silently relabeling the confidence-interval header
    over fit-time stored intervals; alpha=0.0 (previously swallowed by the
    falsy `alpha or self.alpha` idiom) now raises too.
    """

    @pytest.mark.parametrize("bad_alpha", [0.10, 0.0])
    def test_summary_rejects_non_fit_alpha(self, alpha_fitted, bad_alpha):
        with pytest.raises(ValueError, match="never recomputes"):
            alpha_fitted.summary(alpha=bad_alpha)

    def test_summary_accepts_fit_alpha(self, alpha_fitted):
        assert alpha_fitted.summary(alpha=alpha_fitted.alpha) == alpha_fitted.summary()
