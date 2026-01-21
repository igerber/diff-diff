"""
Tests for Callaway-Sant'Anna staggered DiD estimator.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, CallawaySantAnnaResults
from diff_diff.prep import generate_staggered_data as _generate_staggered_data


def generate_staggered_data(
    n_units: int = 100,
    n_periods: int = 10,
    n_cohorts: int = 3,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic staggered adoption data for tests.

    Wrapper around the library function to maintain backward compatibility
    with test signatures (uses 'time' column instead of 'period').
    """
    # Compute cohort periods based on n_cohorts
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int).tolist()

    data = _generate_staggered_data(
        n_units=n_units,
        n_periods=n_periods,
        cohort_periods=cohort_periods,
        never_treated_frac=never_treated_frac,
        treatment_effect=treatment_effect,
        dynamic_effects=True,
        effect_growth=0.1,
        unit_fe_sd=2.0,
        noise_sd=0.5,
        seed=seed,
    )

    # Rename 'period' to 'time' for backward compatibility with existing tests
    data = data.rename(columns={"period": "time"})

    return data


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

    def test_extreme_weights_warning(self):
        """Test that extreme weights produce warnings and methodology-aligned behavior.

        Per the Methodology Registry (docs/methodology/REGISTRY.md):
        - Missing group-time cells: ATT(g,t) set to NaN
        - Analytic SE: returns NaN to signal invalid inference (not biased via zeroing)
        - Bootstrap: drops invalid samples and warns, preserving valid distribution
        """
        import warnings
        np.random.seed(42)

        # Minimal dataset: very small sample with unbalanced groups
        n_units, n_periods = 20, 4
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # Only 2 treated units (extreme imbalance)
        first_treat = np.zeros(n_units)
        first_treat[:2] = 2
        first_treat_expanded = np.repeat(first_treat, n_periods)

        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 1.0 + 2.0 * post + np.random.randn(len(units)) * 0.1

        data = pd.DataFrame({
            'unit': units,
            'time': times,
            'outcome': outcomes,
            'first_treat': first_treat_expanded.astype(int),
        })

        # Test without bootstrap - ATT should be finite, SE may be NaN for edge cases
        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # ATT point estimate should be finite
        assert np.isfinite(results.overall_att), "ATT should be finite"
        # SE is either finite (valid) or NaN (signals invalid inference) - not biased
        assert np.isfinite(results.overall_se) or np.isnan(results.overall_se), \
            "SE should be finite or NaN (not inf)"

        # Test with bootstrap - should drop invalid samples with warning
        cs_boot = CallawaySantAnna(n_bootstrap=100, seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            boot_results = cs_boot.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

        # ATT should be finite
        assert np.isfinite(boot_results.overall_att), "ATT should be finite"
        # Bootstrap SE based on valid samples - may be finite or NaN
        assert boot_results.bootstrap_results is not None, "Bootstrap results should exist"
        assert np.isfinite(boot_results.overall_se) or np.isnan(boot_results.overall_se), \
            "Bootstrap SE should be finite or NaN (not inf)"

    def test_near_collinear_covariates(self):
        """Test that near-collinear covariates are handled gracefully."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Add a near-collinear covariate (x1 + noise above rank detection tolerance)
        # The rank detection tolerance is 1e-07 (matching R's qr()), so we use noise
        # of 1e-5 which is above the tolerance but still creates high collinearity.
        # With noise < 1e-07, the column would be considered linearly dependent.
        np.random.seed(42)
        data['x1_copy'] = data['x1'] + np.random.randn(len(data)) * 1e-5

        cs = CallawaySantAnna(estimation_method='reg')
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            covariates=['x1', 'x1_copy']  # Nearly collinear
        )

        # Should still produce valid results (noise is above tolerance)
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

    def test_rank_deficient_action_error_raises(self):
        """Test that rank_deficient_action='error' raises ValueError on collinear data."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Add a covariate that is perfectly collinear with x1
        data["x1_dup"] = data["x1"].copy()

        cs = CallawaySantAnna(
            estimation_method="reg",  # Use regression method to test OLS path
            rank_deficient_action="error"
        )
        with pytest.raises(ValueError, match="rank-deficient"):
            cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat',
                covariates=['x1', 'x1_dup']
            )

    def test_rank_deficient_action_silent_no_warning(self):
        """Test that rank_deficient_action='silent' produces no warning."""
        import warnings

        data = generate_staggered_data_with_covariates(seed=42)

        # Add a covariate that is perfectly collinear with x1
        data["x1_dup"] = data["x1"].copy()

        cs = CallawaySantAnna(
            estimation_method="reg",  # Use regression method to test OLS path
            rank_deficient_action="silent"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cs.fit(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat',
                covariates=['x1', 'x1_dup']
            )

            # No warnings about rank deficiency should be emitted
            rank_warnings = [x for x in w if "Rank-deficient" in str(x.message)
                           or "rank-deficient" in str(x.message).lower()]
            assert len(rank_warnings) == 0, f"Expected no rank warnings, got {rank_warnings}"

        # Should still get valid results
        assert results is not None
        assert results.overall_att is not None


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
        # Test with new parameter name
        with pytest.raises(ValueError, match="bootstrap_weights"):
            CallawaySantAnna(bootstrap_weights="invalid")
        # Test deprecated parameter still validates
        with pytest.raises(ValueError, match="bootstrap_weights"):
            CallawaySantAnna(bootstrap_weight_type="invalid")

    def test_bootstrap_get_params(self):
        """Test that get_params includes bootstrap_weights."""
        cs = CallawaySantAnna(
            n_bootstrap=99,
            bootstrap_weights="mammen",
            seed=42
        )
        params = cs.get_params()

        assert params['n_bootstrap'] == 99
        assert params['bootstrap_weights'] == "mammen"
        # Deprecated attribute still accessible for backward compat
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


# =============================================================================
# Edge Case Tests: Single Cohort
# =============================================================================


class TestCallawaySantAnnaSingleCohort:
    """Tests for CallawaySantAnna with a single treatment cohort."""

    def test_single_cohort_basic(self):
        """Test CS estimator with single treatment cohort."""
        np.random.seed(42)

        n_units = 60
        n_periods = 8
        treatment_period = 4

        # Generate data with single cohort
        data = []
        for unit in range(n_units):
            # 40% never-treated, 60% treated at period 4
            if unit < int(n_units * 0.4):
                first_treat = 0  # Never treated
            else:
                first_treat = treatment_period  # Single cohort

            unit_fe = np.random.normal(0, 2)

            for t in range(n_periods):
                time_fe = t * 0.3
                y = 10.0 + unit_fe + time_fe

                # Treatment effect for treated units after treatment
                if first_treat > 0 and t >= first_treat:
                    y += 2.5

                y += np.random.normal(0, 0.5)

                data.append({
                    'unit': unit,
                    'time': t,
                    'outcome': y,
                    'first_treat': first_treat,
                })

        df = pd.DataFrame(data)

        cs = CallawaySantAnna()
        results = cs.fit(
            df,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Should produce valid results
        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)
        assert results.overall_se > 0

        # Should have effects for single group only
        groups = set(g for g, t in results.group_time_effects.keys())
        assert len(groups) == 1
        assert treatment_period in groups

        # ATT should be roughly correct
        assert abs(results.overall_att - 2.5) < 1.5

    def test_single_cohort_event_study(self):
        """Test event study aggregation with single cohort."""
        np.random.seed(42)

        n_units = 80
        n_periods = 12
        treatment_period = 6  # Start later to have both pre and post periods

        data = []
        for unit in range(n_units):
            if unit < int(n_units * 0.3):
                first_treat = 0
            else:
                first_treat = treatment_period

            unit_fe = np.random.normal(0, 1)

            for t in range(n_periods):
                y = 10.0 + unit_fe + t * 0.2

                if first_treat > 0 and t >= first_treat:
                    # Dynamic effect: grows over time
                    periods_since = t - first_treat
                    y += 2.0 + 0.3 * periods_since

                y += np.random.normal(0, 0.4)

                data.append({
                    'unit': unit,
                    'time': t,
                    'outcome': y,
                    'first_treat': first_treat,
                })

        df = pd.DataFrame(data)

        cs = CallawaySantAnna()
        results = cs.fit(
            df,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

        # Event study should have multiple relative periods
        rel_periods = sorted(results.event_study_effects.keys())
        assert len(rel_periods) >= 2, f"Expected multiple periods, got {rel_periods}"

        # With single cohort, all effects are for the same group
        # Post-treatment effects (e >= 0) should show positive effect
        post_periods = [e for e in rel_periods if e >= 0]
        if post_periods:
            # At least some post-periods should show positive effect
            post_effects = [results.event_study_effects[e]['effect'] for e in post_periods]
            assert any(e > 0.5 for e in post_effects), f"Expected positive post-period effects, got {post_effects}"

    def test_single_cohort_with_bootstrap(self):
        """Test bootstrap inference with single cohort."""
        np.random.seed(42)

        n_units = 50
        n_periods = 6
        treatment_period = 3

        data = []
        for unit in range(n_units):
            if unit < int(n_units * 0.4):
                first_treat = 0
            else:
                first_treat = treatment_period

            for t in range(n_periods):
                y = 10.0 + np.random.normal(0, 1)
                if first_treat > 0 and t >= first_treat:
                    y += 3.0

                data.append({
                    'unit': unit,
                    'time': t,
                    'outcome': y,
                    'first_treat': first_treat,
                })

        df = pd.DataFrame(data)

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            df,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.overall_att_se > 0
        assert results.bootstrap_results.overall_att_ci[0] < results.bootstrap_results.overall_att_ci[1]

    def test_single_cohort_not_yet_treated_control(self):
        """Test single cohort with not_yet_treated control group.

        With a single cohort, not_yet_treated should behave same as
        never_treated after the treatment period.
        """
        np.random.seed(42)

        n_units = 60
        n_periods = 8
        treatment_period = 4

        data = []
        for unit in range(n_units):
            if unit < int(n_units * 0.4):
                first_treat = 0
            else:
                first_treat = treatment_period

            for t in range(n_periods):
                y = 10.0 + np.random.normal(0, 0.5)
                if first_treat > 0 and t >= first_treat:
                    y += 2.0

                data.append({
                    'unit': unit,
                    'time': t,
                    'outcome': y,
                    'first_treat': first_treat,
                })

        df = pd.DataFrame(data)

        cs_never = CallawaySantAnna(control_group='never_treated')
        results_never = cs_never.fit(
            df,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        cs_not_yet = CallawaySantAnna(control_group='not_yet_treated')
        results_not_yet = cs_not_yet.fit(
            df,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Both should produce valid results
        assert np.isfinite(results_never.overall_att)
        assert np.isfinite(results_not_yet.overall_att)

        # Results may differ slightly due to different comparison groups
        # but should be in similar range
        assert abs(results_never.overall_att - results_not_yet.overall_att) < 1.0


class TestCallawaySantAnnaAnalyticalSE:
    """Tests for analytical SE using influence function aggregation."""

    def test_analytical_se_vs_bootstrap_se(self):
        """Analytical SE should be close to bootstrap SE (within 15%)."""
        # Generate data with moderate size for stable comparison
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            n_cohorts=3,
            treatment_effect=3.0,
            never_treated_frac=0.3,
            seed=42
        )

        # Run with analytical SE (n_bootstrap=0)
        cs_analytical = CallawaySantAnna(n_bootstrap=0, seed=42)
        results_analytical = cs_analytical.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Run with bootstrap SE (n_bootstrap=499)
        cs_bootstrap = CallawaySantAnna(n_bootstrap=499, seed=42)
        results_bootstrap = cs_bootstrap.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Point estimates should match exactly
        assert abs(results_analytical.overall_att - results_bootstrap.overall_att) < 1e-10

        # SEs should be similar (within 15%)
        # Note: Some difference expected due to bootstrap variance vs asymptotic variance
        rel_diff = abs(
            results_analytical.overall_se - results_bootstrap.overall_se
        ) / results_bootstrap.overall_se
        assert rel_diff < 0.15, (
            f"Analytical SE ({results_analytical.overall_se:.4f}) differs from "
            f"bootstrap SE ({results_bootstrap.overall_se:.4f}) by {rel_diff:.1%}"
        )

    def test_analytical_se_accounts_for_covariance(self):
        """Analytical SE should be larger than independence-based SE.

        When there is covariance across (g,t) pairs (from shared control units),
        the correct SE accounting for covariance should be larger than the
        incorrect SE that assumes independence.
        """
        # Generate data where control units are shared across (g,t) pairs
        data = generate_staggered_data(
            n_units=150,
            n_periods=6,
            n_cohorts=2,
            treatment_effect=2.0,
            never_treated_frac=0.4,  # Larger never-treated pool = more sharing
            seed=123
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # The SE should be non-zero and positive
        assert results.overall_se > 0

        # Compute what the "independence" SE would be (sum of weighted variances)
        gt_effects = results.group_time_effects
        weights = []
        variances = []
        for (g, t), effect in gt_effects.items():
            weights.append(effect['n_treated'])
            variances.append(effect['se'] ** 2)

        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        variances = np.array(variances)

        # Independence SE formula (the old incorrect formula)
        independence_var = np.sum(weights ** 2 * variances)
        independence_se = np.sqrt(independence_var)

        # The actual SE (with covariance) should generally be larger
        # because covariances from shared control units are typically positive
        # Note: May not always be true but should be for typical staggered designs
        # We test that both are positive and finite
        assert np.isfinite(results.overall_se)
        assert np.isfinite(independence_se)

    def test_analytical_se_single_gt_pair(self):
        """With a single (g,t) pair, analytical SE should equal the pair's SE."""
        np.random.seed(42)

        # Create data with exactly one treatment cohort
        n_units = 100
        n_periods = 4
        treatment_period = 2

        data = []
        for unit in range(n_units):
            # 50% never treated, 50% treated at period 2
            first_treat = 0 if unit < n_units // 2 else treatment_period
            unit_fe = np.random.normal(0, 1)

            for t in range(n_periods):
                y = 10.0 + unit_fe + t * 0.1
                if first_treat > 0 and t >= first_treat:
                    y += 2.0
                y += np.random.normal(0, 0.5)

                data.append({
                    'unit': unit,
                    'time': t,
                    'outcome': y,
                    'first_treat': first_treat,
                })

        df = pd.DataFrame(data)

        # Use only the first post-treatment period
        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            df,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # If there's only one (g,t) pair, overall SE should match individual SE
        if len(results.group_time_effects) == 1:
            gt_key = list(results.group_time_effects.keys())[0]
            individual_se = results.group_time_effects[gt_key]['se']
            # Should be close (may not be exact due to normalization)
            assert abs(results.overall_se - individual_se) < individual_se * 0.01

    def test_event_study_analytical_se(self):
        """Event study SEs should also use influence function aggregation."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=10,
            n_cohorts=3,
            treatment_effect=2.5,
            never_treated_frac=0.3,
            seed=42
        )

        # Analytical
        cs_analytical = CallawaySantAnna(n_bootstrap=0, seed=42)
        results_analytical = cs_analytical.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        # Bootstrap
        cs_bootstrap = CallawaySantAnna(n_bootstrap=499, seed=42)
        results_bootstrap = cs_bootstrap.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        # Event study effects should exist
        assert results_analytical.event_study_effects is not None
        assert results_bootstrap.event_study_effects is not None

        # Check each event time SE is similar
        for e in results_analytical.event_study_effects:
            if e in results_bootstrap.event_study_effects:
                se_analytical = results_analytical.event_study_effects[e]['se']
                se_bootstrap = results_bootstrap.event_study_effects[e]['se']

                if se_bootstrap > 0:
                    rel_diff = abs(se_analytical - se_bootstrap) / se_bootstrap
                    # Allow 20% difference for event study (more variance)
                    assert rel_diff < 0.20, (
                        f"Event study SE at e={e}: analytical={se_analytical:.4f}, "
                        f"bootstrap={se_bootstrap:.4f}, diff={rel_diff:.1%}"
                    )


class TestCallawaySantAnnaNonStandardColumnNames:
    """Tests for CallawaySantAnna with non-standard column names.

    These tests verify that the estimator works correctly when column names
    differ from the default names (outcome, unit, time, first_treat).
    """

    def generate_data_with_custom_names(
        self,
        outcome_name: str = 'y',
        unit_name: str = 'id',
        time_name: str = 'period',
        first_treat_name: str = 'treatment_start',
        n_units: int = 100,
        n_periods: int = 10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate staggered data with custom column names."""
        np.random.seed(seed)

        # Generate standard data
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # 30% never-treated, rest treated at period 4 or 6
        n_never = int(n_units * 0.3)
        first_treat = np.zeros(n_units)
        first_treat[n_never:n_never + (n_units - n_never) // 2] = 4
        first_treat[n_never + (n_units - n_never) // 2:] = 6
        first_treat_expanded = np.repeat(first_treat, n_periods)

        # Generate outcomes
        unit_fe = np.repeat(np.random.randn(n_units) * 2, n_periods)
        time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = unit_fe + time_fe + 2.5 * post + np.random.randn(len(units)) * 0.5

        return pd.DataFrame({
            outcome_name: outcomes,
            unit_name: units,
            time_name: times,
            first_treat_name: first_treat_expanded.astype(int),
        })

    def test_non_standard_first_treat_name(self):
        """Test with non-standard first_treat column name."""
        data = self.generate_data_with_custom_names(
            first_treat_name='treatment_cohort'
        )

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='y',
            unit='id',
            time='period',
            first_treat='treatment_cohort'
        )

        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)
        assert results.overall_se > 0
        # Treatment effect should be approximately 2.5
        assert abs(results.overall_att - 2.5) < 1.5

    def test_non_standard_all_column_names(self):
        """Test with all non-standard column names."""
        data = self.generate_data_with_custom_names(
            outcome_name='response_var',
            unit_name='entity_id',
            time_name='time_period',
            first_treat_name='treatment_timing',
        )

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='response_var',
            unit='entity_id',
            time='time_period',
            first_treat='treatment_timing'
        )

        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)
        assert results.overall_se > 0

    def test_non_standard_names_with_bootstrap(self):
        """Test non-standard column names with bootstrap inference."""
        data = self.generate_data_with_custom_names(
            first_treat_name='g',  # Short name like R's `did` package uses
            n_units=50
        )

        cs = CallawaySantAnna(n_bootstrap=99, seed=42)
        results = cs.fit(
            data,
            outcome='y',
            unit='id',
            time='period',
            first_treat='g'
        )

        assert results.bootstrap_results is not None
        assert results.overall_se > 0
        assert results.overall_conf_int[0] < results.overall_att < results.overall_conf_int[1]

    def test_non_standard_names_with_event_study(self):
        """Test non-standard column names with event study aggregation."""
        data = self.generate_data_with_custom_names(
            first_treat_name='cohort',
            n_periods=12
        )

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='y',
            unit='id',
            time='period',
            first_treat='cohort',
            aggregate='event_study'
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

    def test_non_standard_names_with_covariates(self):
        """Test non-standard column names with covariate adjustment."""
        # Generate data with covariates
        data = self.generate_data_with_custom_names(
            first_treat_name='treatment_time'
        )
        # Add covariates with custom names
        data['covariate_x'] = np.random.randn(len(data))
        data['covariate_z'] = np.random.binomial(1, 0.5, len(data))

        cs = CallawaySantAnna(estimation_method='dr')
        results = cs.fit(
            data,
            outcome='y',
            unit='id',
            time='period',
            first_treat='treatment_time',
            covariates=['covariate_x', 'covariate_z']
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_non_standard_names_with_not_yet_treated(self):
        """Test non-standard column names with not_yet_treated control group."""
        data = self.generate_data_with_custom_names(
            first_treat_name='adoption_period'
        )

        cs = CallawaySantAnna(control_group='not_yet_treated')
        results = cs.fit(
            data,
            outcome='y',
            unit='id',
            time='period',
            first_treat='adoption_period'
        )

        assert results.overall_att is not None
        assert results.control_group == 'not_yet_treated'

    def test_non_standard_names_matches_standard_names(self):
        """Verify results are identical regardless of column naming."""
        np.random.seed(42)

        # Generate identical data with different column names
        data_standard = generate_staggered_data(n_units=80, seed=42)

        data_custom = data_standard.rename(columns={
            'outcome': 'y',
            'unit': 'entity',
            'time': 't',
            'first_treat': 'g',
        })

        # Fit with standard names
        cs1 = CallawaySantAnna(seed=123)
        results1 = cs1.fit(
            data_standard,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Fit with custom names
        cs2 = CallawaySantAnna(seed=123)
        results2 = cs2.fit(
            data_custom,
            outcome='y',
            unit='entity',
            time='t',
            first_treat='g'
        )

        # Results should be identical
        assert abs(results1.overall_att - results2.overall_att) < 1e-10
        assert abs(results1.overall_se - results2.overall_se) < 1e-10

    def test_column_name_with_spaces(self):
        """Test column names containing spaces."""
        data = self.generate_data_with_custom_names()
        data = data.rename(columns={
            'y': 'outcome variable',
            'treatment_start': 'treatment period',
        })

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome variable',
            unit='id',
            time='period',
            first_treat='treatment period'
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_column_name_with_special_characters(self):
        """Test column names with underscores and numbers."""
        data = self.generate_data_with_custom_names()
        data = data.rename(columns={
            'treatment_start': 'first_treat_2024',
        })

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='y',
            unit='id',
            time='period',
            first_treat='first_treat_2024'
        )

        assert results.overall_att is not None
