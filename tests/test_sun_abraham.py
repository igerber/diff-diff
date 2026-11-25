"""
Tests for Sun-Abraham interaction-weighted estimator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from legacy_surface import es_dict
from scipy import stats

from diff_diff.results_base import EVENT_STUDY_SCHEMA
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
        assert results.att is not None
        assert results.se > 0
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
        assert results.att > 0
        # Effect should be roughly correct (within reasonable bounds)
        assert abs(results.att - 3.0) < 2 * results.se + 1.5

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
        assert abs(results.att) < 3 * results.se + 0.5

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
        assert results1.att is not None
        assert results2.att is not None

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
        # 4.0 (M-093 8b): the unified EVENT_STUDY_SCHEMA frame; the omitted
        # reference period appears as the single is_reference row.
        assert list(df_es.columns) == list(EVENT_STUDY_SCHEMA)
        assert int(df_es["is_reference"].sum()) == 1

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
        assert results.se > 0
        assert results.conf_int[0] < results.att < results.conf_int[1]

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
        assert results1.se == results2.se
        assert results1.conf_int == results2.conf_int

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
        assert results1.se != results2.se

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
        assert results.p_value < 0.05
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
        assert sa_results.att > 0
        assert cs_results.att > 0

        # Both should be reasonably close to true effect
        assert abs(sa_results.att - 3.0) < 2.0
        assert abs(cs_results.att - 3.0) < 2.0

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
        )
        cs_varying_es = es_dict(cs_varying_results)

        # Callaway-Sant'Anna with universal base (all compare to g-1)
        cs_universal = CallawaySantAnna(base_period="universal")
        cs_universal_results = cs_universal.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        cs_universal_es = es_dict(cs_universal_results)

        # Find common event times
        sa_times = set(sa_results.event_study_effects.keys())
        cs_varying_times = set(cs_varying_es.keys())
        cs_universal_times = set(cs_universal_es.keys())
        common_times = sa_times & cs_varying_times & cs_universal_times

        # Separate pre and post periods
        pre_times = [t for t in common_times if t < 0]
        post_times = [t for t in common_times if t > 0]

        # Post-treatment effects should match across all methods
        for t in post_times:
            sa_eff = sa_results.event_study_effects[t]["effect"]
            cs_vary_eff = cs_varying_es[t]["effect"]
            cs_univ_eff = cs_universal_es[t]["effect"]

            # All three should be similar for post-treatment
            max_se = max(
                sa_results.event_study_effects[t]["se"],
                cs_varying_es[t]["se"],
                cs_universal_es[t]["se"],
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
            cs_vary_eff = cs_varying_es[t]["effect"]
            cs_univ_eff = cs_universal_es[t]["effect"]

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
        diff = abs(sa_results.att - cs_results.att)
        max_se = max(sa_results.se, cs_results.se)
        assert diff < 3 * max_se, (
            f"SA ATT={sa_results.att:.3f}, " f"CS ATT={cs_results.att:.3f}, diff={diff:.3f}"
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

        assert results.att is not None
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

        assert results.att is not None
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

        assert results.att is not None

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

        assert results.att is not None
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

        assert results.att is not None
        assert results.se > 0

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
        assert results.att is not None


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

        se = results.se
        t_stat = results.t_stat

        if not np.isfinite(se) or se == 0:
            assert np.isnan(t_stat), f"overall_t_stat should be NaN when SE={se}, got {t_stat}"
            assert np.isnan(results.p_value), (
                f"overall_p_value should be NaN when SE={se} (analytical inference), "
                f"got {results.p_value}"
            )
            ci = results.conf_int
            assert np.isnan(ci[0]) and np.isnan(
                ci[1]
            ), f"overall_conf_int should be (NaN, NaN) when SE={se}, got {ci}"
        else:
            expected = results.att / se
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
        se = results.se
        t_stat = results.t_stat

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
        assert np.isnan(results.att), f"Expected NaN overall_att, got {results.att}"
        assert np.isnan(results.se), f"Expected NaN overall_se, got {results.se}"
        # Downstream inference should propagate NaN
        assert np.isnan(results.t_stat), f"Expected NaN overall_t_stat, got {results.t_stat}"
        assert np.isnan(results.p_value), f"Expected NaN overall_p_value, got {results.p_value}"
        assert np.isnan(results.conf_int[0]) and np.isnan(
            results.conf_int[1]
        ), f"Expected (NaN, NaN) overall_conf_int, got {results.conf_int}"

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
        assert np.isnan(results.att), f"Expected NaN overall_att, got {results.att}"
        assert np.isnan(results.se), f"Expected NaN overall_se, got {results.se}"
        assert np.isnan(results.t_stat), f"Expected NaN overall_t_stat, got {results.t_stat}"
        assert np.isnan(
            results.p_value
        ), f"Expected NaN overall_p_value with bootstrap, got {results.p_value}"
        assert np.isnan(results.conf_int[0]) and np.isnan(
            results.conf_int[1]
        ), f"Expected (NaN, NaN) overall_conf_int, got {results.conf_int}"

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
        assert results.se > 0, f"Expected positive SE from fallback, got {results.se}"

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
            results_inf.att, results_zero.att
        ), f"ATT differs: inf={results_inf.att}, zero={results_zero.att}"
        assert np.isclose(
            results_inf.se, results_zero.se
        ), f"SE differs: inf={results_inf.se}, zero={results_zero.se}"

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
        assert np.isclose(res_default.att, res_explicit.att, atol=1e-14)
        assert np.isclose(res_default.se, res_explicit.se, atol=1e-14)
        assert np.isclose(res_default.p_value, res_explicit.p_value, atol=1e-12)

    def test_classical_finite(self):
        """SA(vcov_type='classical') produces finite ATT/SE; drops auto-cluster."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res = SunAbraham(vcov_type="classical").fit(data, **kwargs)
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert np.isfinite(res.p_value)
        # Classical differs from HC1 cluster-at-unit in heteroscedastic / panel data
        res_hc1 = SunAbraham(vcov_type="hc1").fit(data, **kwargs)
        assert res.se != res_hc1.se

    def test_hc2_finite_no_auto_cluster(self):
        """SA(vcov_type='hc2') runs without explicit cluster (linalg validator
        would reject hc2 + cluster_ids; SA drops the auto-cluster)."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res = SunAbraham(vcov_type="hc2").fit(data, **kwargs)
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        # HC2 differs from cluster-at-unit HC1 on this panel
        res_hc1 = SunAbraham(vcov_type="hc1").fit(data, **kwargs)
        assert res.se != res_hc1.se

    def test_hc2_bm_finite_with_auto_cluster(self):
        """SA(vcov_type='hc2_bm') auto-clusters at unit by default; routes to CR2-BM."""
        data = self._panel()
        kwargs = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        res = SunAbraham(vcov_type="hc2_bm").fit(data, **kwargs)
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        # HC2-BM applies BM Satterthwaite DOF; should differ from cluster-at-unit HC1
        res_hc1 = SunAbraham(vcov_type="hc1").fit(data, **kwargs)
        assert res.se != res_hc1.se

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
        analytical df - FINITE t(G-1) under the 4.0 default, recorded as
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
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        # Differs from cluster-at-unit (different cluster grouping)
        res_unit = SunAbraham(vcov_type="hc2_bm").fit(data, **kwargs)
        assert res.se != res_unit.se

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
        assert np.isfinite(res.att)
        # Replicate path → bootstrap-style SE; should be finite
        assert np.isfinite(res.se)

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
        assert res1.att == res2.att
        assert res1.se == res2.se
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
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
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
        assert np.isclose(res_hc1.att, res_hc2.att, atol=1e-12)
        assert np.isclose(res_hc1.att, res_hc2_bm.att, atol=1e-12)

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
            assert np.isclose(results["hc1"].att, results[vt].att, atol=1e-10), (
                f"overall_att diverges between hc1 ({results['hc1'].att}) "
                f"and {vt} ({results[vt].att}); covariate threading "
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
            assert np.isclose(results["hc1"].att, results[vt].att, atol=1e-10), (
                f"overall_att diverges between hc1 ({results['hc1'].att}) "
                f"and {vt} ({results[vt].att}) under "
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
                results[vt].att
            ), f"overall_att non-finite for vcov_type={vt} under bootstrap"
            assert np.isfinite(results[vt].se), f"bootstrap SE non-finite for vcov_type={vt}"
        # Point estimates must match across vcov_type (FWL preserves them
        # across within-transform vs full-dummy refits).
        for vt in ("classical", "hc2", "hc2_bm"):
            assert np.isclose(results["hc1"].att, results[vt].att, atol=1e-10), (
                f"Bootstrap overall_att diverges: hc1={results['hc1'].att}, "
                f"{vt}={results[vt].att}. Bootstrap refits in the new "
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

    def test_default_is_t_g_minus_1_not_z(self):
        """The 4.0 default: aggregates use t(G-1) (M-128), not z and not the
        residual df; explicit "residual" keeps the D4-fix t(saturated
        residual df)."""
        data = self._panel()
        res = SunAbraham().fit(data, **self._kw)
        p_z = 2 * stats.norm.sf(abs(res.t_stat))
        assert res.p_value != p_z
        G = data["unit"].nunique()
        assert res.inference_df == G - 1
        assert res.p_value == pytest.approx(2 * stats.t.sf(abs(res.t_stat), G - 1), rel=1e-12)
        # every ES row shares the same G-1 df (one df source)
        assert set(res.event_study_df.values()) == {float(G - 1)}
        rr = SunAbraham(df_convention="residual").fit(data, **self._kw)
        dfres = rr.inference_df
        assert dfres is not None and dfres > G - 1
        assert rr.p_value == pytest.approx(2 * stats.t.sf(abs(rr.t_stat), dfres), rel=1e-12)
        assert set(rr.event_study_df.values()) == {dfres}

    def test_cluster_matches_hand_reconstructed_g_minus_1(self):
        data = self._panel()
        r0 = SunAbraham(df_convention="residual").fit(data, **self._kw)
        r1 = SunAbraham(df_convention="cluster").fit(data, **self._kw)
        G = data["unit"].nunique()
        assert r1.att == r0.att and r1.se == r0.se
        assert r1.t_stat == r0.t_stat
        assert r1.p_value == pytest.approx(2 * stats.t.sf(abs(r1.t_stat), G - 1), rel=1e-12)
        lo, hi = r1.conf_int
        crit = stats.t.ppf(0.975, G - 1)
        assert lo == pytest.approx(r1.att - crit * r1.se, rel=1e-12)
        assert hi == pytest.approx(r1.att + crit * r1.se, rel=1e-12)
        assert r1.inference_df == G - 1
        assert set(r1.event_study_df.values()) == {float(G - 1)}

    def test_normal_reproduces_z_on_aggregates(self):
        """'normal' = deliberate z at the fallback level (the pre-3.9 numbers)."""
        data = self._panel()
        r0 = SunAbraham().fit(data, **self._kw)
        rn = SunAbraham(df_convention="normal").fit(data, **self._kw)
        assert rn.att == r0.att and rn.se == r0.se
        assert rn.p_value == pytest.approx(2 * stats.norm.sf(abs(rn.t_stat)), rel=1e-14)
        assert rn.inference_df is None
        # provenance: z rows record NaN
        assert all(np.isnan(v) for v in rn.event_study_df.values())

    def test_bm_dof_precedence_over_knob(self):
        """hc2_bm BM contrast DOF wins under every knob value."""
        data = self._panel()
        r_res = SunAbraham(vcov_type="hc2_bm", df_convention="residual").fit(data, **self._kw)
        r_clu = SunAbraham(vcov_type="hc2_bm", df_convention="cluster").fit(data, **self._kw)
        assert r_res.p_value == r_clu.p_value
        assert r_res.conf_int == r_clu.conf_int

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
            assert r.p_value == base.p_value
            assert r.conf_int == base.conf_int
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
        assert est.df_convention == "cluster" and est.alpha == 0.05
        # Valid value + unknown key: the unknown-key rejection must also
        # leave the estimator fully unchanged (no partial application).
        before = est.get_params()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(df_convention="normal", nonexistent_param=1)
        assert est.get_params() == before

    def test_get_params_roundtrip_and_repeat_fit(self):
        est = SunAbraham(df_convention="residual")
        assert est.get_params()["df_convention"] == "residual"
        clone = SunAbraham(**est.get_params())
        data = self._panel()
        r1 = clone.fit(data, **self._kw)
        p1, d1 = r1.p_value, r1.inference_df
        r2 = clone.fit(data, **self._kw)
        assert r2.p_value == p1 and r2.inference_df == d1
        assert clone.df_convention == "residual"


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


# ---------------------------------------------------------------------------
# Per-cohort reference, cohorts without a pre-period, identification guard
# ---------------------------------------------------------------------------

_ID_KW = dict(outcome="y", unit="unit", time="time", first_treat="g")


def _tau(e):
    """True effect at relative period ``e`` (zero before treatment)."""
    return 2.0 + 0.5 * e if e >= 0 else 0.0


def _ident_panel(noise=0.0, seed=0, n_units=90, cohorts=(0, 3, 5), n_periods=8):
    """Unit effect + linear time trend + effect ``2 + 0.5 e``; noise sd ``noise``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = cohorts[i % len(cohorts)]
        alpha = rng.normal()
        for t in range(1, n_periods + 1):
            y = alpha + 0.3 * t + rng.normal(scale=noise) if noise else alpha + 0.3 * t
            rows.append((i, t, g, y + (_tau(t - g) if g and t >= g else 0.0)))
    return pd.DataFrame(rows, columns=["unit", "time", "g", "y"])


def _fit_id(data, est=None, **fit_kwargs):
    """Fit and return (results, list of UserWarning texts)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = SunAbraham(**(est or {})).fit(data, **_ID_KW, **fit_kwargs)
    texts = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return res, texts


def _oracle_refs(data, anticipation=0):
    """Independent re-derivation of the per-cohort reference rule."""
    comparison = set(data.loc[data["g"] == 0, "time"])
    e_ref = -1 - anticipation
    refs = {}
    for g in sorted(x for x in data["g"].unique() if x > 0):
        rel = set((data.loc[data["g"] == g, "time"] - g).tolist())
        pre = sorted((e for e in rel if e < -anticipation), reverse=True)
        candidates = ([e_ref] if e_ref in rel else []) + [e for e in pre if e != e_ref]
        usable = [e for e in candidates if g + e in comparison]
        refs[g] = (usable or candidates or [None])[0]
    return refs


def _oracle_unidentified(data, refs):
    """Cells with a nonzero entry in the null space of the explicit-dummy design."""
    d = data.reset_index(drop=True)
    rel = d["time"] - d["g"]
    # A cell is a (cohort, relative period) other than the cohort's reference
    # and other than e = -1, which is never an estimated period.
    cells = sorted(
        {(int(g), int(e)) for g, e in zip(d["g"], rel) if g > 0 and e != refs[g] and e != -1}
    )
    cell_cols = [((d["g"] == g) & (rel == e)).astype(float).values for g, e in cells]
    X = np.column_stack(
        cell_cols
        + [pd.get_dummies(d["unit"], dtype=float).values]
        + [pd.get_dummies(d["time"], dtype=float).values]
    )
    # full_matrices=True: a sparse panel has fewer rows than columns, and the
    # economy SVD would silently omit null directions.
    _, s, vt = np.linalg.svd(X, full_matrices=True)
    rank = int((s > 1e-9 * s.max()).sum())
    null = vt[rank:].T
    return {c for c, row in zip(cells, null[: len(cells)]) if np.abs(row).max() > 1e-7}, cells


def _nan_cells(res):
    return {
        (int(g), int(e)) for (g, e), v in res.cohort_effects.items() if not np.isfinite(v["effect"])
    }


def _true_att(data, res):
    """Truth for the overall ATT: post rows weighted by treated observations."""
    rel = (data["time"] - data["g"])[data["g"] > 0]
    post = [e for e in res.event_study_effects if e >= 0]
    n = np.array([(rel == e).sum() for e in post], dtype=float)
    return float(np.sum(n * np.array([_tau(e) for e in post])) / n.sum())


def _gap_panel(noise=0.0, seed=0):
    d = _ident_panel(noise=noise, seed=seed)
    return d[~((d["g"] == 5) & (d["time"] == 4))]  # cohort 5 unobserved at e = -1


def _nt_absent_panel(noise=0.0, seed=0, period=6):
    d = _ident_panel(noise=noise, seed=seed)
    return d[~((d["g"] == 0) & (d["time"] == period))]


def _split_cohort_panel(noise=0.0):
    """Cohort 5's units are observed at two disjoint sets of periods."""
    d = _ident_panel(noise=noise)
    even = d["unit"] % 2 == 0
    drop = (d["g"] == 5) & ((even & ~d["time"].isin([1, 3])) | (~even & d["time"].isin([1, 3, 4])))
    return d[~drop]


class TestSunAbrahamCohortReference:
    """A cohort not observed at e = -1 - anticipation; cohorts with no pre-period."""

    @pytest.mark.parametrize(
        "case",
        [
            "gap",
            "treated_at_first_period",
            "first_observed_at_zero",
            "anticipation",
            "no_comparison_at_reference",
        ],
    )
    def test_exact_recovery_noiseless(self, case):
        d = _ident_panel()
        est = {}
        if case == "gap":
            data, fitted = _gap_panel(), _gap_panel()
        elif case == "treated_at_first_period":
            data = pd.concat([d[d["g"] != 3], d[d["g"] == 3].assign(g=1)])
            # recompute outcomes for the relabelled cohort so the truth holds
            data = data.assign(
                y=lambda x: x["y"]
                - np.where((x["g"] == 1) & (x["time"] >= 3), [_tau(t - 3) for t in x["time"]], 0.0)
                + np.where(x["g"] == 1, [_tau(t - 1) for t in x["time"]], 0.0)
            )
            fitted = data[data["g"] != 1]
        elif case == "first_observed_at_zero":
            data = d[~((d["g"] == 3) & (d["time"] < 3))]
            fitted = data[data["g"] != 3]
        elif case == "no_comparison_at_reference":
            # Never-treated units are absent at t = 4, cohort 5's e = -1: the
            # cohort is normalized at e = -2 (t = 3) instead, and everything
            # stays identified.
            data = fitted = _nt_absent_panel(period=4)
        else:
            est = {"anticipation": 1}
            data = d[~((d["g"] == 3) & (d["time"] == 1))]  # no row at e <= -2
            fitted = data[data["g"] != 3]
        res, texts = _fit_id(data, est=est)
        assert not any("rank" in t.lower() or "not identified by" in t for t in texts)
        if case == "no_comparison_at_reference":
            assert any("cohort 5: -2" in t for t in texts)
            assert res.reference_event_times == (-2, -1)
        assert np.isfinite(res.att)
        for (g, e), v in res.cohort_effects.items():
            assert abs(v["effect"] - _tau(e)) < 1e-9, (g, e, v["effect"])
        for e, v in res.event_study_effects.items():
            assert abs(v["effect"] - _tau(e)) < 1e-9, (e, v["effect"])
        assert abs(res.att - _true_att(fitted, res)) < 1e-9

    @pytest.mark.parametrize("n_bootstrap", [0, 40])
    @pytest.mark.parametrize("control_group", ["never_treated", "not_yet_treated"])
    def test_exclusion_equals_the_reduced_fit(self, n_bootstrap, control_group):
        d = _ident_panel(noise=0.5)
        full = d[~((d["g"] == 3) & (d["time"] < 3))]  # cohort 3 has no pre-period
        est = {"n_bootstrap": n_bootstrap, "seed": 11, "control_group": control_group}
        res, texts = _fit_id(full, est=est)
        ref, ref_texts = _fit_id(full[full["g"] != 3], est=est)
        assert any("EXCLUDED" in t and "Cohort(s) 3" in t for t in texts)
        assert not any("EXCLUDED" in t for t in ref_texts)
        assert res.groups == ref.groups == [5]
        assert res.n_obs == ref.n_obs and res.n_treated_units == ref.n_treated_units
        np.testing.assert_allclose([res.att, res.se], [ref.att, ref.se], rtol=1e-12)
        assert set(res.event_study_effects) == set(ref.event_study_effects)
        for e, v in res.event_study_effects.items():
            w = ref.event_study_effects[e]
            np.testing.assert_allclose([v["effect"], v["se"]], [w["effect"], w["se"]], rtol=1e-12)
        assert set(res.cohort_effects) == set(ref.cohort_effects)
        for k, v in res.cohort_effects.items():
            np.testing.assert_allclose(v["effect"], ref.cohort_effects[k]["effect"], rtol=1e-12)

    def test_fallback_reference_matches_an_independent_regression(self):
        data = _gap_panel(noise=0.5)
        res, texts = _fit_id(data)
        assert any("cohort 5: -2" in t and "mix normalization bases" in t for t in texts)
        refs = {3: -1, 5: -2}
        d = data.reset_index(drop=True)
        rel = d["time"] - d["g"]
        cells = sorted({(int(g), int(e)) for g, e in zip(d["g"], rel) if g > 0 and e != refs[g]})
        X = np.column_stack(
            [((d["g"] == g) & (rel == e)).astype(float).values for g, e in cells]
            + [pd.get_dummies(d["unit"], dtype=float).values]
            + [pd.get_dummies(d["time"], dtype=float).values[:, 1:]]
        )
        beta = np.linalg.lstsq(X, d["y"].values, rcond=None)[0][: len(cells)]
        assert {(int(g), int(e)) for g, e in res.cohort_effects} == set(cells)
        for (g, e), b in zip(cells, beta):
            np.testing.assert_allclose(
                res.cohort_effects[(g, e)]["effect"], b, rtol=1e-8, atol=1e-10
            )
        # the pooled e = -2 row holds cohort 3 only: cohort 5's cell is its reference
        assert set(res.cohort_weights[-2]) == {3}

    def test_fractional_pre_period_is_a_pre_period(self):
        # Cohort treated at 3, observed at 2.5 but not at 2: e = -0.5 is before
        # treatment, so the cohort is kept and normalized there.
        d = _ident_panel(n_units=60, cohorts=(0, 3))
        half = d[d["time"] == 2].assign(time=2.5)
        half = half.assign(y=half["y"] + 0.3 * 0.5)
        data = pd.concat([d[~((d["g"] == 3) & (d["time"] == 2))], half[half["g"] == 3]])
        # never-treated units need the half period too, or its cells are unidentified
        data = pd.concat([data, half[half["g"] == 0]]).sort_values(["unit", "time"])
        res, texts = _fit_id(data)
        assert any("cohort 3: -0.5" in t for t in texts)
        assert res.reference_event_times == (-0.5,)
        assert res.reference_period == -0.5 and res.reference_observed
        for (g, e), v in res.cohort_effects.items():
            assert abs(v["effect"] - _tau(e)) < 1e-9

    def test_every_cohort_without_a_pre_period_raises(self):
        d = _ident_panel()
        data = d[~((d["g"] > 0) & (d["time"] < d["g"]))]
        with pytest.raises(ValueError, match="only\\s+treated cohorts"):
            SunAbraham().fit(data, **_ID_KW)

    def test_no_never_treated_error_keeps_precedence(self):
        d = _ident_panel()
        data = d[(d["g"] > 0) & (d["time"] >= d["g"])]  # no controls AND no pre-period
        with pytest.raises(ValueError, match="No never-treated units"):
            SunAbraham().fit(data, **_ID_KW)

    @pytest.mark.parametrize("vcov_type", ["hc1", "hc2_bm", "classical"])
    def test_variance_families_agree_on_point_estimates(self, vcov_type):
        base, _ = _fit_id(_gap_panel(noise=0.5))
        res, _ = _fit_id(_gap_panel(noise=0.5), est={"vcov_type": vcov_type})
        np.testing.assert_allclose(res.att, base.att, rtol=1e-9)
        assert np.isfinite(res.se) and res.se > 0
        for e, v in res.event_study_effects.items():
            np.testing.assert_allclose(
                v["effect"], base.event_study_effects[e]["effect"], atol=1e-8
            )
            assert np.isfinite(v["se"]) and v["se"] > 0

    def test_conley_reads_coordinates_from_the_reduced_frame(self):
        d = _ident_panel(noise=0.5)
        rng = np.random.default_rng(4)
        loc = {u: (rng.uniform(30, 45), rng.uniform(-100, -80)) for u in d["unit"].unique()}
        d = d.assign(lat=d["unit"].map(lambda u: loc[u][0]), lon=d["unit"].map(lambda u: loc[u][1]))
        full = d[~((d["g"] == 3) & (d["time"] < 3))]
        est = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=500.0,
            conley_lag_cutoff=1,
        )
        res, texts = _fit_id(full, est=est)
        ref, _ = _fit_id(full[full["g"] != 3], est=est)
        assert any("EXCLUDED" in t for t in texts)
        np.testing.assert_allclose([res.att, res.se], [ref.att, ref.se], rtol=1e-12)


class TestSunAbrahamReferenceFields:
    """reference_event_times / reference_period and the pre-trends consumer."""

    def test_balanced_fit_records_the_single_reference(self):
        res, texts = _fit_id(_ident_panel(noise=0.5))
        assert res.reference_event_times == (-1,)
        assert res.reference_period == -1 and res.reference_observed
        assert not any("normalized at their latest" in t for t in texts)
        frame = res.to_dataframe()
        ref_rows = frame[frame["is_reference"]]
        assert ref_rows["event_time"].tolist() == [-1] and ref_rows["att"].tolist() == [0.0]

    def test_mixed_references_are_recorded_and_refused_by_pretrends(self):
        from diff_diff import DiagnosticReport, compute_pretrends_power

        res, _ = _fit_id(_gap_panel(noise=0.5))
        assert res.reference_event_times == (-2, -1)
        assert res.reference_period == -1 and res.reference_observed
        # e = -2 is an ESTIMATED row here (cohort 3), not a reference row
        frame = res.to_dataframe()
        assert frame.loc[frame["is_reference"], "event_time"].tolist() == [-1]
        with pytest.raises(ValueError, match="DISTINCT normalization\\s+bases"):
            compute_pretrends_power(res)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            section = DiagnosticReport(res).to_dict()["pretrends_power"]
        assert section["status"] == "error"
        assert "DISTINCT normalization" in section["reason"]

    def test_balanced_pretrends_is_unchanged_by_the_new_field(self):
        import dataclasses

        from diff_diff import compute_pretrends_power

        res, _ = _fit_id(_ident_panel(noise=0.5))
        legacy = dataclasses.replace(res, reference_event_times=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new, old = compute_pretrends_power(res), compute_pretrends_power(legacy)
        # The deterministic inputs are identical; the MDV itself comes from a
        # simulated box probability and is compared to its Monte Carlo noise.
        np.testing.assert_array_equal(new.violation_weights, old.violation_weights)
        np.testing.assert_array_equal(new.vcov, old.vcov)
        np.testing.assert_array_equal(new.pre_period_effects, old.pre_period_effects)
        assert new.covariance_source == old.covariance_source == "full_pre_period_vcov"
        assert new.mdv == pytest.approx(old.mdv, rel=1e-3)

    @pytest.mark.parametrize("layout", ["single_cohort_gap", "biennial"])
    def test_shared_fallback_reference_becomes_the_anchor(self, layout):
        from diff_diff import compute_pretrends_power

        if layout == "single_cohort_gap":
            d = _ident_panel(noise=0.5, n_units=60, cohorts=(0, 5))
            data = d[~((d["g"] == 5) & (d["time"] == 4))]
        else:  # odd periods only: no cohort is observed at e = -1
            d = _ident_panel(noise=0.5, n_units=90, cohorts=(0, 5, 7), n_periods=11)
            data = d[d["time"] % 2 == 1]
        res, texts = _fit_id(data)
        assert res.reference_event_times == (-2,)
        assert res.reference_period == -2 and res.reference_observed
        assert any("normalized at their latest" in t for t in texts)
        assert not any("mix normalization bases" in t for t in texts)
        assert -2 not in res.event_study_effects and -1 not in res.event_study_effects
        frame = res.to_dataframe()
        assert frame.loc[frame["is_reference"], "event_time"].tolist() == [-2]
        # Roth's linear violation is anchored at the true base, -2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp = compute_pretrends_power(res)
        pre = sorted(e for e in res.event_study_effects if e < 0)
        np.testing.assert_allclose(np.abs(pp.violation_weights), [abs(e + 2) for e in pre])


class TestSunAbrahamIdentificationGuard:
    """Coefficients the observed rows do not identify are NaN, never a wrong number."""

    @pytest.mark.parametrize("builder", [_nt_absent_panel, _split_cohort_panel])
    def test_structured_unidentified_cells_are_nan(self, builder):
        data = builder()
        res, texts = _fit_id(data)
        expected, cells = _oracle_unidentified(data, _oracle_refs(data))
        assert expected  # the panel does exercise the guard
        assert _nan_cells(res) == expected
        assert {(int(g), int(e)) for g, e in res.cohort_effects} == set(cells)
        warning = [t for t in texts if "not identified by the observed rows" in t]
        assert len(warning) == 1
        g, e = sorted(expected)[0]
        assert f"({g}, {e})" in warning[0]
        for (g, e), v in res.cohort_effects.items():
            if (int(g), int(e)) in expected:
                assert np.isnan(v["effect"]) and np.isnan(v["se"])
            else:
                assert abs(v["effect"] - _tau(e)) < 1e-9
        pooled_nan = {e for (_, e) in expected}
        for e, v in res.event_study_effects.items():
            if e in pooled_nan:
                assert np.isnan(v["effect"]) and np.isnan(v["se"])
                assert np.isnan(v["p_value"]) and np.all(np.isnan(v["conf_int"]))
            else:
                assert abs(v["effect"] - _tau(e)) < 1e-9
        if any(e >= 0 for e in pooled_nan):
            assert np.isnan(res.att) and np.isnan(res.se)

    @staticmethod
    def _random_panels():
        base = _ident_panel()
        for seed in range(40):
            rng = np.random.default_rng(seed)
            yield base[rng.random(len(base)) > rng.choice([0.3, 0.6, 0.85])]
        for seed in range(10):  # every unit observed at two adjacent periods
            rng = np.random.default_rng(100 + seed)
            start = rng.integers(1, 8, size=42)
            rows = []
            for i in range(42):
                g = (0, 3, 4)[i % 3]
                alpha = rng.normal()
                for t in (start[i], start[i] + 1):
                    rows.append((i, t, g, alpha + 0.3 * t + (_tau(t - g) if g and t >= g else 0.0)))
            yield pd.DataFrame(rows, columns=["unit", "time", "g", "y"])

    def test_no_wrong_finite_number_on_random_missingness(self):
        n_checked = n_flagged = 0
        for data in self._random_panels():
            refs = _oracle_refs(data)
            data = data[~data["g"].isin([g for g, r in refs.items() if r is None])]
            refs = {g: r for g, r in refs.items() if r is not None}
            if not refs or not (data["g"] == 0).any():
                continue
            res, _ = _fit_id(data)
            expected, cells = _oracle_unidentified(data, refs)
            assert _nan_cells(res) == expected
            for (g, e), v in res.cohort_effects.items():
                if (int(g), int(e)) not in expected:
                    assert abs(v["effect"] - _tau(e)) < 1e-6, (g, e, v["effect"])
            for e, v in res.event_study_effects.items():
                if np.isfinite(v["effect"]):
                    assert abs(v["effect"] - _tau(e)) < 1e-6
            n_checked += 1
            n_flagged += bool(expected)
        assert n_checked >= 40 and n_flagged >= 10

    def test_full_dummy_path_flags_the_same_cells(self):
        for period in (1, 6):
            data = _nt_absent_panel(period=period)
            within, _ = _fit_id(data)
            for vcov_type in ("hc2_bm", "classical"):
                res, _ = _fit_id(data, est={"vcov_type": vcov_type})
                assert _nan_cells(res) == _nan_cells(within) != set()
                for k, v in res.cohort_effects.items():
                    if np.isfinite(v["effect"]):
                        assert abs(v["effect"] - _tau(k[1])) < 1e-9

    def test_rank_deficient_action(self):
        data = _nt_absent_panel()
        with pytest.raises(ValueError, match="rank-deficient.*\\(3, 3\\)"):
            SunAbraham(rank_deficient_action="error").fit(data, **_ID_KW)
        silent, texts = _fit_id(data, est={"rank_deficient_action": "silent"})
        assert not any("not identified" in t or "ank-deficient" in t for t in texts)
        warned, texts = _fit_id(data)
        assert sum("not identified by the observed rows" in t for t in texts) == 1
        assert _nan_cells(silent) == _nan_cells(warned) != set()

    def test_resampling_drops_draws_that_lose_the_reference(self):
        # Cohort 5's reference cell is held by ONE unit: a draw without that
        # unit cannot normalize the cohort. Such draws must be dropped, not
        # entered into the distribution under an arbitrary baseline (which
        # gives SEs of 0.5-0.8 on this nearly noiseless panel).
        d = _ident_panel(noise=1e-6)
        keeper = d.loc[d["g"] == 5, "unit"].min()
        data = d[~((d["g"] == 5) & (d["time"] < 5) & (d["unit"] != keeper))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = SunAbraham(n_bootstrap=199, seed=7).fit(data, **_ID_KW)
        assert any("non-finite bootstrap samples" in str(w.message) for w in caught)
        assert np.isfinite(res.se)
        for e, v in res.event_study_effects.items():
            assert np.isfinite(v["se"]) and v["se"] < 1e-4, (e, v["se"])

    def test_resampling_never_substitutes_the_original_estimate(self):
        # `rank_deficient_action` governs the FULL-SAMPLE fit. A draw that
        # cannot identify a cell must come back NaN and be dropped under every
        # setting; raising inside the draw would hand it to the bootstrap's
        # exception fallback, which substitutes the original estimate and
        # shrinks the SE (0.0751 instead of 0.0959 on this panel).
        d = _ident_panel(noise=0.5)
        keeper = d.loc[d["g"] == 5, "unit"].min()
        data = d[~((d["g"] == 5) & (d["time"] < 5) & (d["unit"] != keeper))]
        ses = {}
        for action in ("warn", "silent", "error"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = SunAbraham(n_bootstrap=199, seed=7, rank_deficient_action=action).fit(
                    data, **_ID_KW
                )
            texts = [str(w.message) for w in caught]
            assert not any("Using original estimate" in t for t in texts), action
            assert any("non-finite bootstrap samples" in t for t in texts), action
            ses[action] = (res.se, tuple(v["se"] for v in res.event_study_effects.values()))
        assert ses["warn"] == ses["silent"] == ses["error"]
        assert np.isfinite(ses["warn"][0]) and ses["warn"][0] > 0

    @pytest.mark.parametrize("vcov_type", ["hc1", "hc2_bm"])
    def test_every_cell_unidentified_reports_nan(self, vcov_type):
        # One treated cohort (g = 2) and never-treated units observed only at
        # t = 1: the single cell (2, 0) has no comparison row, so no design
        # column is left. That is reported, not raised as "no interactions".
        data = pd.DataFrame(
            {
                "unit": [0, 1, 1, 2, 2],
                "time": [1, 1, 2, 1, 2],
                "g": [0, 2, 2, 2, 2],
                "y": [0.0, 0.5, 3.0, 0.2, 2.9],
            }
        )
        res, texts = _fit_id(data, est={"vcov_type": vcov_type})
        assert sum("not identified by the observed rows" in t and "(2, 0)" in t for t in texts) == 1
        assert _nan_cells(res) == {(2, 0)}
        assert np.isnan(res.cohort_effects[(2, 0)]["se"])
        row = res.event_study_effects[0]
        assert np.isnan(row["effect"]) and np.isnan(row["se"]) and np.isnan(row["p_value"])
        assert np.isnan(res.att) and np.isnan(res.se) and np.isnan(res.p_value)
        assert np.all(np.isnan(res.event_study_vcov))
        silent, texts = _fit_id(
            data, est={"vcov_type": vcov_type, "rank_deficient_action": "silent"}
        )
        assert not any("not identified" in t for t in texts) and np.isnan(silent.att)
        # every inference field is undefined together, under both settings
        from tests.conftest import assert_nan_inference

        for fitted in (res, silent):
            assert_nan_inference(fitted.event_study_effects[0])
            assert_nan_inference(
                {
                    "se": fitted.se,
                    "t_stat": fitted.t_stat,
                    "p_value": fitted.p_value,
                    "conf_int": fitted.conf_int,
                }
            )
        with pytest.raises(ValueError, match="rank-deficient.*\\(2, 0\\)"):
            SunAbraham(vcov_type=vcov_type, rank_deficient_action="error").fit(data, **_ID_KW)

    def test_event_study_covariance_keeps_the_identified_block(self):
        from diff_diff import compute_pretrends_power

        data = _nt_absent_panel(noise=0.5, period=1)  # flags (3, -2) and (5, -4)
        res, _ = _fit_id(data)
        nan_rows = {e for e, v in res.event_study_effects.items() if np.isnan(v["effect"])}
        assert nan_rows == {-4, -2}
        index = list(res.event_study_vcov_index)
        bad = [index.index(e) for e in sorted(nan_rows)]
        good = [i for i in range(len(index)) if i not in bad]
        vcov = res.event_study_vcov
        assert np.all(np.isnan(vcov[bad, :])) and np.all(np.isnan(vcov[:, bad]))
        block = vcov[np.ix_(good, good)]
        assert np.all(np.isfinite(block))
        ses = np.array([res.event_study_effects[index[i]]["se"] for i in good])
        np.testing.assert_allclose(np.sqrt(np.diag(block)), ses, rtol=1e-10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp = compute_pretrends_power(res, pre_periods=[-3])
        assert pp.covariance_source == "full_pre_period_vcov"
        assert np.isfinite(pp.mdv)

    def test_undefined_variance_is_not_reported_as_zero(self):
        # Full rank, zero residual df: the covariance is undefined (NaN) with
        # no unidentified cell, and must stay NaN in the event-study block.
        data = pd.DataFrame(
            {
                "unit": [0, 0, 1, 1],
                "time": [1, 2, 1, 2],
                "g": [0, 0, 2, 2],
                "y": [0.0, 1.0, 0.5, 3.0],
            }
        )
        res, _ = _fit_id(data, est={"cluster": False})
        assert res.att == pytest.approx(1.5)
        assert np.all(np.isnan(res.event_study_vcov))


class TestStructuralIdentificationHelper:
    """`_structural_identification` against the null space of the design."""

    @staticmethod
    def _call(data, refs, weights=None):
        from diff_diff.sun_abraham import _structural_identification

        rel = sorted({e for g, e in zip(data["g"], data["time"] - data["g"]) if g > 0} - {-1})
        return _structural_identification(
            data["unit"].values, data["time"].values, data["g"].values, rel, refs, -1, weights
        )

    def test_identified_design(self):
        assert self._call(_ident_panel(), {3: -1, 5: -1}) == ([], [])
        assert self._call(_gap_panel(), {3: -1, 5: -2}) == ([], [])

    @pytest.mark.parametrize("builder", [_nt_absent_panel, _split_cohort_panel])
    def test_unidentified_cells_and_minimal_omission(self, builder):
        data = builder()
        refs = _oracle_refs(data)
        omit, unidentified = self._call(data, refs)
        expected, cells = _oracle_unidentified(data, refs)
        assert {(int(g), int(e)) for g, e in unidentified} == expected
        assert len(omit) == 1 and set(omit) <= set(unidentified)
        # leaving `omit` out makes the cell block full rank given the fixed effects
        d = data.reset_index(drop=True)
        rel = d["time"] - d["g"]
        kept = [c for c in cells if c not in {(int(g), int(e)) for g, e in omit}]
        fe = np.column_stack(
            [
                pd.get_dummies(d["unit"], dtype=float).values,
                pd.get_dummies(d["time"], dtype=float).values,
            ]
        )
        X = np.column_stack(
            [((d["g"] == g) & (rel == e)).astype(float).values for g, e in kept] + [fe]
        )
        assert np.linalg.matrix_rank(X) == np.linalg.matrix_rank(fe) + len(kept)

    def test_zero_weight_rows_are_not_observed(self):
        data = _ident_panel()
        weights = np.where(data["g"] == 0, np.where(data["time"] == 4, 0.0, 1.0), 1.0)
        dropped = data[weights > 0]
        refs = {3: -1, 5: -1}
        assert self._call(data, refs, weights) == self._call(dropped, refs)
        assert self._call(data, refs, weights)[1] != []

    def test_matches_the_null_space_on_random_panels(self):
        n_flagged = 0
        for seed in range(200):
            rng = np.random.default_rng(seed)
            cohorts = [(0, 3, 5), (0, 2, 4, 6), (0, 5)][seed % 3]
            base = _ident_panel(n_units=60, cohorts=cohorts)
            data = base[rng.random(len(base)) > rng.choice([0.2, 0.5, 0.8, 0.9, 0.95])]
            refs = _oracle_refs(data)
            data = data[~data["g"].isin([g for g, r in refs.items() if r is None])]
            refs = {g: r for g, r in refs.items() if r is not None}
            if not refs or not (data["g"] == 0).any():
                continue
            _, unidentified = self._call(data, refs)
            expected, _ = _oracle_unidentified(data, refs)
            assert {(int(g), int(e)) for g, e in unidentified} == expected, seed
            n_flagged += bool(expected)
        assert n_flagged >= 40


class TestSunAbrahamIdentificationSurvey:
    """Survey designs: refusal, zero weights, replicate weights."""

    @staticmethod
    def _with_design(data):
        data = data.copy()
        data["w"] = 1.0 + (data["unit"] % 5) * 0.2
        data["stratum"] = data["unit"] % 4
        return data

    @staticmethod
    def _add_replicates(data, n_rep=12, seed=3):
        rng = np.random.default_rng(seed)
        data = data.copy()
        cols = []
        for j in range(n_rep):
            jitter = {u: rng.uniform(0.5, 1.5) for u in data["unit"].unique()}
            data[f"rw{j}"] = data["unit"].map(jitter) * data["w"]
            cols.append(f"rw{j}")
        return data, cols

    def test_cohort_without_a_pre_period_is_refused(self):
        from diff_diff.survey import SurveyDesign

        d = self._with_design(_ident_panel(noise=0.5))
        data = d[~((d["g"] == 3) & (d["time"] < 3))]
        with pytest.raises(NotImplementedError, match="Cohort\\(s\\) 3 .*domain estimation"):
            SunAbraham().fit(data, survey_design=SurveyDesign(weights="w"), **_ID_KW)

    def test_zero_weight_pre_periods_count_as_unobserved(self):
        from diff_diff.survey import SurveyDesign

        # Cohort 3's units observed before treatment all have weight 0; the
        # positive-weight units of the cohort are observed from e = 0 only.
        d = self._with_design(_ident_panel(noise=0.5))
        late = (d["g"] == 3) & (d["unit"] % 2 == 0)
        data = d[~(late & (d["time"] < 3))].copy()
        data.loc[(data["g"] == 3) & (data["unit"] % 2 == 1), "w"] = 0.0
        with pytest.raises(NotImplementedError, match="Cohort\\(s\\) 3 "):
            SunAbraham().fit(data, survey_design=SurveyDesign(weights="w"), **_ID_KW)

    def test_fallback_reference_under_tsl_and_replicate_designs(self):
        from diff_diff.survey import SurveyDesign

        data = self._with_design(_gap_panel(noise=0.5))
        tsl, texts = _fit_id(
            data, survey_design=SurveyDesign(weights="w", strata="stratum", psu="unit")
        )
        assert any("cohort 5: -2" in t for t in texts)
        assert np.isfinite(tsl.se) and tsl.se > 0
        data, cols = self._add_replicates(data)
        rep, _ = _fit_id(
            data,
            survey_design=SurveyDesign(weights="w", replicate_weights=cols, replicate_method="JK1"),
        )
        np.testing.assert_allclose(rep.att, tsl.att, rtol=1e-10)
        assert np.isfinite(rep.se) and rep.se > 0
        assert all(np.isfinite(v["se"]) and v["se"] > 0 for v in rep.event_study_effects.values())

    def test_single_cohort_fallback_under_replicate_weights(self):
        from diff_diff.survey import SurveyDesign

        # The reference period (-2) holds no estimated cell: it must not sit in
        # the replicate vector as a structural NaN.
        d = self._with_design(_ident_panel(noise=0.5, n_units=60, cohorts=(0, 5)))
        data, cols = self._add_replicates(d[~((d["g"] == 5) & (d["time"] == 4))])
        res, _ = _fit_id(
            data,
            survey_design=SurveyDesign(weights="w", replicate_weights=cols, replicate_method="JK1"),
        )
        assert -2 not in res.event_study_effects
        assert np.isfinite(res.se) and res.se > 0
        assert all(np.isfinite(v["se"]) and v["se"] > 0 for v in res.event_study_effects.values())

    def test_zero_weight_cohort_does_not_poison_pooled_rows(self):
        from diff_diff.survey import SurveyDesign

        d = self._with_design(_ident_panel(noise=0.5))
        zeroed = d.copy()
        zeroed.loc[zeroed["g"] == 5, "w"] = 0.0
        res, _ = _fit_id(zeroed, survey_design=SurveyDesign(weights="w"))
        ref, _ = _fit_id(d[d["g"] != 5], survey_design=SurveyDesign(weights="w"))
        np.testing.assert_allclose(res.att, ref.att, rtol=1e-10)
        assert set(res.event_study_effects) == set(ref.event_study_effects)
        for e, v in res.event_study_effects.items():
            np.testing.assert_allclose(v["effect"], ref.event_study_effects[e]["effect"], rtol=1e-9)
            # SEs are not compared: the zero-weight units stay in the design.
            assert np.isfinite(v["se"]) and v["se"] > 0
        assert all(g == 3 for g, _ in res.cohort_effects)

    def test_zero_weight_cell_beside_an_observed_cohort(self):
        from diff_diff.survey import SurveyDesign

        # The only cohort-5 units observed at t = 8 (e = 3) are zero-weight
        # units; cohort 3 is observed there (e = 5 for cohort 3, so e = 3 is
        # pooled from cohort 3's t = 6 cell alone).
        d = self._with_design(_ident_panel(noise=0.5))
        odd5 = (d["g"] == 5) & (d["unit"] % 2 == 1)
        data = d[~((d["g"] == 5) & ~odd5 & (d["time"] == 8))].copy()
        data.loc[(data["g"] == 5) & (data["unit"] % 2 == 1), "w"] = 0.0
        res, _ = _fit_id(data, survey_design=SurveyDesign(weights="w"))
        assert (5, 3) not in res.cohort_effects
        row = res.event_study_effects[3]
        assert np.isfinite(row["effect"]) and np.isfinite(row["se"])
        np.testing.assert_allclose(row["effect"], res.cohort_effects[(3, 3)]["effect"], rtol=1e-12)
        assert np.isfinite(res.att)

    def test_guard_under_replicate_weights_keeps_identified_inference(self):
        from diff_diff.survey import SurveyDesign

        d = self._with_design(_nt_absent_panel(noise=0.5, period=1))
        data, cols = self._add_replicates(d)
        res, _ = _fit_id(
            data,
            survey_design=SurveyDesign(weights="w", replicate_weights=cols, replicate_method="JK1"),
        )
        assert _nan_cells(res) == {(3, -2), (5, -4)}
        for e, v in res.event_study_effects.items():
            if e in (-4, -2):
                assert np.isnan(v["effect"]) and np.isnan(v["se"])
            else:
                assert np.isfinite(v["se"]) and v["se"] > 0, e
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        for k, v in res.cohort_effects.items():
            assert np.isfinite(v["se"]) == np.isfinite(v["effect"])


class TestSunAbrahamBootstrapRowPositions:
    """Both bootstrap paths index rows by POSITION, whatever the frame's index."""

    def _relabelled(self, data):
        shifted = data.copy()
        shifted.index = shifted.index + 1000
        reversed_ = data.copy()
        reversed_.index = np.arange(len(data))[::-1].copy()
        return shifted, reversed_

    def test_pairs_bootstrap_ignores_index_labels(self):
        data = _ident_panel(noise=0.5)
        base, _ = _fit_id(data, est={"n_bootstrap": 40, "seed": 3})
        for other in self._relabelled(data):
            res, _ = _fit_id(other, est={"n_bootstrap": 40, "seed": 3})
            assert res.se == base.se
            for e, v in res.event_study_effects.items():
                assert v["se"] == base.event_study_effects[e]["se"]

    def test_rao_wu_bootstrap_ignores_index_labels(self):
        from diff_diff.survey import SurveyDesign

        data = _ident_panel(noise=0.5)
        data["w"] = 1.0 + (data["unit"] % 5) * 0.2
        data["stratum"] = data["unit"] % 4
        sd = SurveyDesign(weights="w", strata="stratum", psu="unit")
        base, _ = _fit_id(data, est={"n_bootstrap": 40, "seed": 3}, survey_design=sd)
        for other in self._relabelled(data):
            res, _ = _fit_id(other, est={"n_bootstrap": 40, "seed": 3}, survey_design=sd)
            assert res.se == base.se

    def test_bootstrap_with_a_filtered_out_unit(self):
        # A unit with a negative first_treat is neither treated nor a control;
        # under the default control group its rows are filtered out, leaving a
        # frame whose index is no longer 0..n-1.
        data = _ident_panel(noise=0.5)
        data.loc[data["unit"] == 4, "g"] = -1
        res, _ = _fit_id(data, est={"n_bootstrap": 40, "seed": 3})
        assert np.isfinite(res.se) and res.se > 0
