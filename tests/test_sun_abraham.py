"""
Tests for Sun-Abraham interaction-weighted estimator.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.sun_abraham import SunAbraham, SunAbrahamResults, SABootstrapResults


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


class TestSunAbrahamBootstrap:
    """Tests for Sun-Abraham bootstrap inference."""

    def test_bootstrap_basic(self):
        """Test basic bootstrap functionality."""
        data = generate_staggered_data(n_units=50, seed=42)

        sa = SunAbraham(n_bootstrap=99, seed=42)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.n_bootstrap == 99
        assert results.bootstrap_results.weight_type == "rademacher"
        assert results.overall_se > 0
        assert (
            results.overall_conf_int[0]
            < results.overall_att
            < results.overall_conf_int[1]
        )

    def test_bootstrap_weight_types(self):
        """Test different bootstrap weight types."""
        data = generate_staggered_data(n_units=50, seed=42)

        weight_types = ["rademacher", "mammen", "webb"]

        for wt in weight_types:
            sa = SunAbraham(n_bootstrap=49, bootstrap_weights=wt, seed=42)
            results = sa.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

            assert results.bootstrap_results is not None
            assert results.bootstrap_results.weight_type == wt
            assert results.overall_se > 0

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with same seed."""
        data = generate_staggered_data(n_units=50, seed=42)

        sa1 = SunAbraham(n_bootstrap=99, seed=123)
        results1 = sa1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        sa2 = SunAbraham(n_bootstrap=99, seed=123)
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

    def test_bootstrap_different_seeds(self):
        """Test that different seeds give different results."""
        data = generate_staggered_data(n_units=50, seed=42)

        sa1 = SunAbraham(n_bootstrap=99, seed=123)
        results1 = sa1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        sa2 = SunAbraham(n_bootstrap=99, seed=456)
        results2 = sa2.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Results should differ with different seeds
        assert results1.overall_se != results2.overall_se

    def test_bootstrap_p_value_significance(self):
        """Test that strong effect has significant p-value with bootstrap."""
        data = generate_staggered_data(n_units=100, treatment_effect=5.0, seed=42)

        sa = SunAbraham(n_bootstrap=199, seed=42)
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

    def test_bootstrap_distribution_stored(self):
        """Test that bootstrap distribution is stored in results."""
        data = generate_staggered_data(n_units=50, seed=42)

        sa = SunAbraham(n_bootstrap=99, seed=42)
        results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.bootstrap_results.bootstrap_distribution is not None
        assert len(results.bootstrap_results.bootstrap_distribution) == 99

    def test_bootstrap_event_study_effects(self):
        """Test that bootstrap updates event study effect SEs."""
        data = generate_staggered_data(n_units=50, seed=42)

        sa = SunAbraham(n_bootstrap=99, seed=42)
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

    def test_bootstrap_invalid_weight_type(self):
        """Test that invalid weight type raises error."""
        with pytest.raises(ValueError, match="bootstrap_weights"):
            SunAbraham(bootstrap_weights="invalid")

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

        data = generate_staggered_data(
            n_units=200, treatment_effect=3.0, seed=42
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

        # Both should detect positive effect
        assert sa_results.overall_att > 0
        assert cs_results.overall_att > 0

        # Both should be reasonably close to true effect
        assert abs(sa_results.overall_att - 3.0) < 2.0
        assert abs(cs_results.overall_att - 3.0) < 2.0

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
        data = generate_staggered_data(
            n_units=200, n_periods=15, n_cohorts=8, seed=42
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
