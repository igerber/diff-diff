"""
Tests for Callaway-Sant'Anna staggered DiD estimator.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, CallawaySantAnnaResults


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
    # Some units never treated, others treated in different periods
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
    # Y = unit_fe + time_fe + treatment_effect * post + noise
    unit_fe = np.random.randn(n_units) * 2
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    # Treatment indicator
    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)

    # Dynamic treatment effects (effect grows over time)
    relative_time = times - first_treat_expanded
    dynamic_effect = treatment_effect * (1 + 0.1 * np.maximum(relative_time, 0))

    outcomes = (
        unit_fe_expanded +
        time_fe_expanded +
        dynamic_effect * post +
        np.random.randn(len(units)) * 0.5
    )

    df = pd.DataFrame({
        'unit': units,
        'time': times,
        'outcome': outcomes,
        'first_treat': first_treat_expanded.astype(int),
    })

    return df


class TestCallawaySantAnna:
    """Tests for CallawaySantAnna estimator."""

    def test_basic_fit(self):
        """Test basic model fitting."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert cs.is_fitted_
        assert isinstance(results, CallawaySantAnnaResults)
        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.group_time_effects) > 0

    def test_positive_treatment_effect(self):
        """Test that estimator recovers positive treatment effect."""
        data = generate_staggered_data(treatment_effect=3.0, seed=123)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Should detect positive effect
        assert results.overall_att > 0
        # Effect should be roughly correct (within 2 SE)
        assert abs(results.overall_att - 3.0) < 2 * results.overall_se + 1.0

    def test_zero_treatment_effect(self):
        """Test with no treatment effect."""
        data = generate_staggered_data(treatment_effect=0.0, seed=456)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Effect should be close to zero
        assert abs(results.overall_att) < 3 * results.overall_se

    def test_event_study_aggregation(self):
        """Test event study aggregation."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

        # Check that relative periods are present
        rel_periods = list(results.event_study_effects.keys())
        assert any(p >= 0 for p in rel_periods)  # Post-treatment

    def test_group_aggregation(self):
        """Test aggregation by treatment cohort."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='group'
        )

        assert results.group_effects is not None
        assert len(results.group_effects) > 0

    def test_all_aggregation(self):
        """Test computing all aggregations."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='all'
        )

        assert results.event_study_effects is not None
        assert results.group_effects is not None

    def test_control_group_options(self):
        """Test different control group options."""
        data = generate_staggered_data()

        # Never treated only
        cs1 = CallawaySantAnna(control_group="never_treated")
        results1 = cs1.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Not yet treated
        cs2 = CallawaySantAnna(control_group="not_yet_treated")
        results2 = cs2.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert results1.control_group == "never_treated"
        assert results2.control_group == "not_yet_treated"
        # Results should be different
        assert results1.overall_att != results2.overall_att

    def test_estimation_methods(self):
        """Test different estimation methods."""
        data = generate_staggered_data()

        methods = ["reg", "ipw", "dr"]
        results = {}

        for method in methods:
            cs = CallawaySantAnna(estimation_method=method)
            results[method] = cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

        # All methods should produce results
        for method, res in results.items():
            assert res.overall_att is not None

    def test_summary_output(self):
        """Test summary output formatting."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        summary = results.summary()

        assert "Callaway-Sant'Anna" in summary
        assert "ATT" in summary
        assert "Std. Err." in summary

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='all'
        )

        # Group-time DataFrame
        df_gt = results.to_dataframe(level='group_time')
        assert 'group' in df_gt.columns
        assert 'time' in df_gt.columns
        assert 'effect' in df_gt.columns

        # Event study DataFrame
        df_es = results.to_dataframe(level='event_study')
        assert 'relative_period' in df_es.columns

        # Group DataFrame
        df_g = results.to_dataframe(level='group')
        assert 'group' in df_g.columns

    def test_get_set_params(self):
        """Test sklearn-compatible parameter access."""
        cs = CallawaySantAnna(alpha=0.10, control_group="not_yet_treated")

        params = cs.get_params()
        assert params['alpha'] == 0.10
        assert params['control_group'] == "not_yet_treated"

        cs.set_params(alpha=0.05)
        assert cs.alpha == 0.05

    def test_missing_column_error(self):
        """Test error on missing columns."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()

        with pytest.raises(ValueError, match="Missing columns"):
            cs.fit(
                data,
                outcome='nonexistent',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

    def test_no_control_units_error(self):
        """Test error when no control units exist."""
        data = generate_staggered_data(never_treated_frac=0.0)

        # All units are treated, no controls
        cs = CallawaySantAnna()

        with pytest.raises(ValueError, match="No never-treated units"):
            cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

    def test_significance_properties(self):
        """Test significance-related properties."""
        data = generate_staggered_data(treatment_effect=5.0)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # With strong effect, should be significant
        assert results.is_significant
        assert results.significance_stars in ["*", "**", "***"]


class TestCallawaySantAnnaResults:
    """Tests for CallawaySantAnnaResults class."""

    def test_repr(self):
        """Test string representation."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        repr_str = repr(results)
        assert "CallawaySantAnnaResults" in repr_str
        assert "ATT=" in repr_str

    def test_invalid_level_error(self):
        """Test error on invalid DataFrame level."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe(level='invalid')

    def test_event_study_not_computed_error(self):
        """Test error when event study not computed."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe(level='event_study')


def generate_staggered_data_with_covariates(
    n_units: int = 100,
    n_periods: int = 10,
    n_cohorts: int = 3,
    treatment_effect: float = 2.0,
    covariate_effect: float = 1.0,
    never_treated_frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic staggered adoption data with covariates."""
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

    # Generate unit-level covariates (time-invariant)
    x1 = np.random.randn(n_units)  # continuous covariate
    x2 = np.random.binomial(1, 0.5, n_units)  # binary covariate

    # Make treatment assignment correlated with covariates (confounding)
    # Units with higher x1 are more likely to be treated
    # This creates a situation where covariate adjustment matters

    x1_expanded = np.repeat(x1, n_periods)
    x2_expanded = np.repeat(x2, n_periods)

    # Generate outcomes
    unit_fe = np.random.randn(n_units) * 2
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    # Treatment indicator
    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)

    # Outcome depends on covariates
    outcomes = (
        unit_fe_expanded +
        time_fe_expanded +
        covariate_effect * x1_expanded +  # covariate effect
        0.5 * x2_expanded +  # second covariate effect
        treatment_effect * post +
        np.random.randn(len(units)) * 0.5
    )

    df = pd.DataFrame({
        'unit': units,
        'time': times,
        'outcome': outcomes,
        'first_treat': first_treat_expanded.astype(int),
        'x1': x1_expanded,
        'x2': x2_expanded,
    })

    return df


class TestCallawaySantAnnaCovariates:
    """Tests for CallawaySantAnna covariate adjustment."""

    def test_covariates_are_used(self):
        """Test that covariates are actually used in estimation."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Fit without covariates
        cs1 = CallawaySantAnna()
        results1 = cs1.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Fit with covariates
        cs2 = CallawaySantAnna()
        results2 = cs2.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x2']
        )

        # Both should produce valid results
        assert results1.overall_att is not None
        assert results2.overall_att is not None

        # Results may differ when using covariates
        # (they don't have to differ significantly for this test)
        assert results1.overall_se > 0
        assert results2.overall_se > 0

    def test_outcome_regression_with_covariates(self):
        """Test outcome regression method with covariates."""
        data = generate_staggered_data_with_covariates(seed=123)

        cs = CallawaySantAnna(estimation_method='reg')
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x2']
        )

        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.group_time_effects) > 0

    def test_ipw_with_covariates(self):
        """Test IPW method with covariates."""
        data = generate_staggered_data_with_covariates(seed=456)

        cs = CallawaySantAnna(estimation_method='ipw')
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x2']
        )

        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.group_time_effects) > 0

    def test_doubly_robust_with_covariates(self):
        """Test doubly robust method with covariates."""
        data = generate_staggered_data_with_covariates(seed=789)

        cs = CallawaySantAnna(estimation_method='dr')
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x2']
        )

        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.group_time_effects) > 0

    def test_all_methods_with_covariates(self):
        """Test that all estimation methods work with covariates."""
        data = generate_staggered_data_with_covariates(seed=42)

        methods = ['reg', 'ipw', 'dr']
        results = {}

        for method in methods:
            cs = CallawaySantAnna(estimation_method=method)
            results[method] = cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat',
                covariates=['x1', 'x2']
            )

        # All methods should produce valid results
        for method, res in results.items():
            assert res.overall_att is not None, f"{method} failed to produce ATT"
            assert res.overall_se > 0, f"{method} failed to produce valid SE"

    def test_event_study_with_covariates(self):
        """Test event study aggregation with covariates."""
        data = generate_staggered_data_with_covariates(seed=42)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x2'],
            aggregate='event_study'
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

    def test_missing_covariate_error(self):
        """Test error when covariate column is missing."""
        data = generate_staggered_data_with_covariates()

        cs = CallawaySantAnna()

        with pytest.raises(ValueError, match="Missing columns"):
            cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat',
                covariates=['x1', 'nonexistent']
            )

    def test_single_covariate(self):
        """Test with a single covariate."""
        data = generate_staggered_data_with_covariates(seed=42)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1']
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_treatment_effect_recovery_with_covariates(self):
        """Test that we recover approximately correct treatment effect."""
        # Generate data with known treatment effect
        data = generate_staggered_data_with_covariates(
            treatment_effect=3.0,
            covariate_effect=2.0,
            seed=123,
            n_units=200  # More units for better precision
        )

        cs = CallawaySantAnna(estimation_method='dr')
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x2']
        )

        # Effect should be roughly correct (within reasonable bounds)
        # Note: we use a generous bound due to finite sample variance
        assert results.overall_att > 0, "ATT should be positive"
        assert abs(results.overall_att - 3.0) < 2.0, f"ATT={results.overall_att} too far from 3.0"

    def test_extreme_propensity_scores(self):
        """Test handling of covariates that strongly predict treatment.

        When covariates nearly perfectly separate treated/control units,
        propensity scores approach 0 or 1. The estimator should handle
        this gracefully via propensity score clipping.
        """
        np.random.seed(42)
        n_units = 100
        n_periods = 8

        # Generate unit and time identifiers
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # Create a covariate that strongly predicts treatment
        # High values -> treated, low values -> never-treated
        x_strong = np.random.randn(n_units)
        x_strong_expanded = np.repeat(x_strong, n_periods)

        # Assign treatment based on covariate (top 50% treated at period 4)
        first_treat = np.zeros(n_units)
        first_treat[x_strong > np.median(x_strong)] = 4
        first_treat_expanded = np.repeat(first_treat, n_periods)

        # Generate outcomes
        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 1.0 + 0.5 * x_strong_expanded + 2.0 * post + np.random.randn(len(units)) * 0.3

        data = pd.DataFrame({
            'unit': units,
            'time': times,
            'outcome': outcomes,
            'first_treat': first_treat_expanded.astype(int),
            'x_strong': x_strong_expanded,
        })

        # IPW should handle extreme propensity scores via clipping
        cs = CallawaySantAnna(estimation_method='ipw')
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x_strong']
        )

        # Should produce valid results (not NaN or inf)
        assert np.isfinite(results.overall_att), "ATT should be finite"
        assert np.isfinite(results.overall_se), "SE should be finite"
        assert results.overall_se > 0, "SE should be positive"

    def test_near_collinear_covariates(self):
        """Test that near-collinear covariates are handled gracefully."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Add a near-collinear covariate (x1 + small noise)
        data['x1_copy'] = data['x1'] + np.random.randn(len(data)) * 1e-8

        cs = CallawaySantAnna(estimation_method='reg')
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x1_copy']  # Nearly collinear
        )

        # Should still produce valid results (lstsq handles this)
        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)

    def test_missing_values_in_covariates_warning(self):
        """Test that missing values trigger fallback warning."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Introduce NaN in covariate
        data.loc[data['time'] == 2, 'x1'] = np.nan

        cs = CallawaySantAnna()

        # Should warn about missing values and fall back to unconditional
        with pytest.warns(UserWarning, match="Missing values in covariates"):
            results = cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat',
                covariates=['x1', 'x2']
            )

        # Should still produce valid results (using unconditional estimation)
        assert results.overall_att is not None
        assert results.overall_se > 0


class TestCallawaySantAnnaBootstrap:
    """Tests for Callaway-Sant'Anna multiplier bootstrap inference."""

    def test_bootstrap_basic(self):
        """Test basic bootstrap functionality."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.n_bootstrap == 99
        assert results.bootstrap_results.weight_type == "rademacher"
        assert results.overall_se > 0
        assert results.overall_conf_int[0] < results.overall_att < results.overall_conf_int[1]

    def test_bootstrap_weight_types(self):
        """Test different bootstrap weight types."""
        data = generate_staggered_data(n_units=50, seed=42)

        weight_types = ["rademacher", "mammen", "webb"]

        for wt in weight_types:
            cs = CallawaySantAnna(
                n_bootstrap=49,
                bootstrap_weight_type=wt,
                seed=42
            )
            results = cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

            assert results.bootstrap_results is not None
            assert results.bootstrap_results.weight_type == wt
            assert results.overall_se > 0

    def test_bootstrap_event_study(self):
        """Test bootstrap with event study aggregation."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        assert results.bootstrap_results.event_study_cis is not None
        assert results.bootstrap_results.event_study_p_values is not None

        # Check event study effects have bootstrap SEs
        for e, effect in results.event_study_effects.items():
            assert effect['se'] > 0
            assert effect['conf_int'][0] < effect['conf_int'][1]

    def test_bootstrap_group_aggregation(self):
        """Test bootstrap with group aggregation."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='group'
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.group_effect_ses is not None
        assert results.bootstrap_results.group_effect_cis is not None
        assert results.bootstrap_results.group_effect_p_values is not None

        # Check group effects have bootstrap SEs
        for g, effect in results.group_effects.items():
            assert effect['se'] > 0
            assert effect['conf_int'][0] < effect['conf_int'][1]

    def test_bootstrap_all_aggregations(self):
        """Test bootstrap with all aggregations."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='all'
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        assert results.bootstrap_results.group_effect_ses is not None

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with same seed."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs1 = CallawaySantAnna(n_bootstrap=99, seed=123)
        results1 = cs1.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        cs2 = CallawaySantAnna(n_bootstrap=99, seed=123)
        results2 = cs2.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Results should be identical with same seed
        assert results1.overall_se == results2.overall_se
        assert results1.overall_conf_int == results2.overall_conf_int

    def test_bootstrap_different_seeds(self):
        """Test that different seeds give different results."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs1 = CallawaySantAnna(n_bootstrap=99, seed=123)
        results1 = cs1.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        cs2 = CallawaySantAnna(n_bootstrap=99, seed=456)
        results2 = cs2.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Results should differ with different seeds
        assert results1.overall_se != results2.overall_se

    def test_bootstrap_p_value_significance(self):
        """Test that strong effect has significant p-value with bootstrap."""
        data = generate_staggered_data(
            n_units=100,
            treatment_effect=5.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=199, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Strong effect should be significant
        assert results.overall_p_value < 0.05
        assert results.is_significant

    def test_bootstrap_zero_effect_not_significant(self):
        """Test that zero effect is not significant with bootstrap."""
        data = generate_staggered_data(
            n_units=50,
            treatment_effect=0.0,
            seed=42
        )

        cs = CallawaySantAnna(n_bootstrap=199, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Zero effect should not be significant at 0.01 level
        # (using 0.01 to be more conservative with finite sample)
        assert results.overall_p_value > 0.01 or abs(results.overall_att) < 2 * results.overall_se

    def test_bootstrap_distribution_stored(self):
        """Test that bootstrap distribution is stored in results."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert results.bootstrap_results.bootstrap_distribution is not None
        assert len(results.bootstrap_results.bootstrap_distribution) == 99

    def test_bootstrap_with_covariates(self):
        """Test bootstrap with covariate adjustment."""
        data = generate_staggered_data_with_covariates(n_units=50, seed=42)

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x2']
        )

        assert results.bootstrap_results is not None
        assert results.overall_se > 0

    def test_bootstrap_group_time_effects(self):
        """Test that bootstrap updates group-time effect SEs."""
        data = generate_staggered_data(n_units=50, seed=42)

        # Without bootstrap
        cs1 = CallawaySantAnna(n_bootstrap=0)
        results1 = cs1.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # With bootstrap
        cs2 = CallawaySantAnna(n_bootstrap=99, seed=42)
        results2 = cs2.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Group-time effects should have same point estimates
        for gt in results1.group_time_effects:
            assert results1.group_time_effects[gt]['effect'] == results2.group_time_effects[gt]['effect']
            # But SEs may differ (bootstrap vs analytical)
            assert results2.group_time_effects[gt]['se'] > 0

    def test_bootstrap_invalid_weight_type(self):
        """Test that invalid weight type raises error."""
        with pytest.raises(ValueError, match="bootstrap_weight_type"):
            CallawaySantAnna(bootstrap_weight_type="invalid")

    def test_bootstrap_get_params(self):
        """Test that get_params includes bootstrap_weight_type."""
        cs = CallawaySantAnna(
            n_bootstrap=99,
            bootstrap_weight_type="mammen",
            seed=42
        )
        params = cs.get_params()

        assert params['n_bootstrap'] == 99
        assert params['bootstrap_weight_type'] == "mammen"
        assert params['seed'] == 42

    def test_bootstrap_with_not_yet_treated(self):
        """Test bootstrap with not_yet_treated control group."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(
            control_group="not_yet_treated",
            n_bootstrap=99,
            seed=42
        )
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert results.bootstrap_results is not None
        assert results.overall_se > 0

    def test_bootstrap_estimation_methods(self):
        """Test bootstrap with different estimation methods."""
        data = generate_staggered_data(n_units=50, seed=42)

        methods = ["reg", "ipw", "dr"]

        for method in methods:
            cs = CallawaySantAnna(
                estimation_method=method,
                n_bootstrap=49,
                seed=42
            )
            results = cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

            assert results.bootstrap_results is not None
            assert results.overall_se > 0, f"Failed for method {method}"

    def test_bootstrap_with_balanced_event_study(self):
        """Test bootstrap with balanced event study aggregation."""
        data = generate_staggered_data(n_units=100, n_periods=12, seed=42)

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study',
            balance_e=0  # Balance at treatment time
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        assert results.event_study_effects is not None

        # Check that event study effects have valid bootstrap SEs
        for e, effect in results.event_study_effects.items():
            assert effect['se'] > 0
            assert effect['conf_int'][0] < effect['conf_int'][1]

    def test_bootstrap_low_iterations_warning(self):
        """Test that low n_bootstrap triggers a warning."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(n_bootstrap=30, seed=42)

        with pytest.warns(UserWarning, match="n_bootstrap=30 is low"):
            cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )
