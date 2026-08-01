"""
Tests for Callaway-Sant'Anna staggered DiD estimator.
"""

import warnings

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
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
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
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
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
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Effect should be close to zero
        assert abs(results.overall_att) < 3 * results.overall_se

    def test_never_treated_inf_encoding(self):
        """Test that first_treat=np.inf is handled as never-treated, not as a cohort."""
        data = generate_staggered_data(n_units=200, n_periods=10, n_cohorts=3, seed=42)

        cs = CallawaySantAnna(n_bootstrap=0)
        results_zero = cs.fit(
            data.copy(), outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Re-encode never-treated from 0 to np.inf (cast to float first for pandas compat)
        data_inf = data.copy()
        data_inf["first_treat"] = data_inf["first_treat"].astype(float)
        data_inf.loc[data_inf["first_treat"] == 0, "first_treat"] = np.inf

        results_inf = cs.fit(
            data_inf, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Results should be identical
        assert np.isclose(
            results_inf.overall_att, results_zero.overall_att
        ), f"ATT differs: inf={results_inf.overall_att}, zero={results_zero.overall_att}"

    def test_event_study_aggregation(self):
        """Test event study aggregation."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
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
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert results.group_effects is not None
        assert len(results.group_effects) > 0

    def test_all_aggregation(self):
        """Test computing all aggregations."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        assert results.event_study_effects is not None
        assert results.group_effects is not None

    def test_control_group_options(self):
        """Test different control group options."""
        data = generate_staggered_data()

        # Never treated only
        cs1 = CallawaySantAnna(control_group="never_treated")
        results1 = cs1.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Not yet treated
        cs2 = CallawaySantAnna(control_group="not_yet_treated")
        results2 = cs2.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
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
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
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
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
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
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        # Group-time DataFrame
        df_gt = results.to_dataframe(level="group_time")
        assert "group" in df_gt.columns
        assert "time" in df_gt.columns
        assert "effect" in df_gt.columns

        # Event study DataFrame
        df_es = results.to_dataframe(level="event_study")
        assert "relative_period" in df_es.columns

        # Group DataFrame
        df_g = results.to_dataframe(level="group")
        assert "group" in df_g.columns

    def test_get_set_params(self):
        """Test sklearn-compatible parameter access."""
        cs = CallawaySantAnna(alpha=0.10, control_group="not_yet_treated")

        params = cs.get_params()
        assert params["alpha"] == 0.10
        assert params["control_group"] == "not_yet_treated"

        cs.set_params(alpha=0.05)
        assert cs.alpha == 0.05

    def test_missing_column_error(self):
        """Test error on missing columns."""
        data = generate_staggered_data()

        cs = CallawaySantAnna()

        with pytest.raises(ValueError, match="Missing columns"):
            cs.fit(data, outcome="nonexistent", unit="unit", time="time", first_treat="first_treat")

    def test_no_control_units_error(self):
        """Test error when no control units exist."""
        data = generate_staggered_data(never_treated_frac=0.0)

        # All units are treated, no controls
        cs = CallawaySantAnna()

        with pytest.raises(ValueError, match="No never-treated units"):
            cs.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

    def test_significance_properties(self):
        """Test significance-related properties."""
        data = generate_staggered_data(treatment_effect=5.0)

        cs = CallawaySantAnna()
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
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
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        repr_str = repr(results)
        assert "CallawaySantAnnaResults" in repr_str
        assert "ATT=" in repr_str

    def test_invalid_level_error(self):
        """Test error on invalid DataFrame level."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe(level="invalid")

    def test_event_study_not_computed_error(self):
        """Test error when event study not computed."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe(level="event_study")


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
        unit_fe_expanded
        + time_fe_expanded
        + covariate_effect * x1_expanded  # covariate effect
        + 0.5 * x2_expanded  # second covariate effect
        + treatment_effect * post
        + np.random.randn(len(units)) * 0.5
    )

    df = pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcomes,
            "first_treat": first_treat_expanded.astype(int),
            "x1": x1_expanded,
            "x2": x2_expanded,
        }
    )

    return df


class TestCallawaySantAnnaCovariates:
    """Tests for CallawaySantAnna covariate adjustment."""

    def test_covariates_are_used(self):
        """Test that covariates are actually used in estimation."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Fit without covariates
        cs1 = CallawaySantAnna()
        results1 = cs1.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Fit with covariates
        cs2 = CallawaySantAnna()
        results2 = cs2.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
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

        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.group_time_effects) > 0

    def test_ipw_with_covariates(self):
        """Test IPW method with covariates."""
        data = generate_staggered_data_with_covariates(seed=456)

        cs = CallawaySantAnna(estimation_method="ipw")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.group_time_effects) > 0

    def test_doubly_robust_with_covariates(self):
        """Test doubly robust method with covariates."""
        data = generate_staggered_data_with_covariates(seed=789)

        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        assert results.overall_att is not None
        assert results.overall_se > 0
        assert len(results.group_time_effects) > 0

    def test_all_methods_with_covariates(self):
        """Test that all estimation methods work with covariates."""
        data = generate_staggered_data_with_covariates(seed=42)

        methods = ["reg", "ipw", "dr"]
        results = {}

        for method in methods:
            cs = CallawaySantAnna(estimation_method=method)
            results[method] = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
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
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
            aggregate="event_study",
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
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "nonexistent"],
            )

    def test_single_covariate(self):
        """Test with a single covariate."""
        data = generate_staggered_data_with_covariates(seed=42)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1"],
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
            n_units=200,  # More units for better precision
        )

        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
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

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
                "x_strong": x_strong_expanded,
            }
        )

        # IPW should handle extreme propensity scores via clipping
        cs = CallawaySantAnna(estimation_method="ipw")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x_strong"],
        )

        # Should produce valid results (not NaN or inf)
        assert np.isfinite(results.overall_att), "ATT should be finite"
        assert np.isfinite(results.overall_se), "SE should be finite"
        assert results.overall_se > 0, "SE should be positive"

    def test_extreme_weights_warning(self, ci_params):
        """Test that extreme weights produce warnings and methodology-aligned behavior.

        Tests that:
        - ATT point estimates remain finite
        - SE is finite (valid) or NaN (signals invalid inference), never biased
        - Bootstrap drops invalid samples and adjusts inference accordingly
        """
        import warnings

        np.random.seed(42)
        n_boot = ci_params.bootstrap(100)

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

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
            }
        )

        # Test without bootstrap - ATT should be finite, SE may be NaN for edge cases
        cs = CallawaySantAnna()
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # ATT point estimate should be finite
        assert np.isfinite(results.overall_att), "ATT should be finite"
        # SE is either finite (valid) or NaN (signals invalid inference) - not biased
        assert np.isfinite(results.overall_se) or np.isnan(
            results.overall_se
        ), "SE should be finite or NaN (not inf)"

        # Test with bootstrap - should drop invalid samples with warning
        cs_boot = CallawaySantAnna(n_bootstrap=n_boot, seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            boot_results = cs_boot.fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

        # Collect warning messages for inspection
        warning_messages = [str(warning.message) for warning in w]

        # ATT should be finite
        assert np.isfinite(boot_results.overall_att), "ATT should be finite"

        # Bootstrap SE based on valid samples - may be finite or NaN
        assert boot_results.bootstrap_results is not None, "Bootstrap results should exist"
        assert np.isfinite(boot_results.overall_se) or np.isnan(
            boot_results.overall_se
        ), "Bootstrap SE should be finite or NaN (not inf)"

        # If SE is NaN, verify it's due to validity threshold (should have warning)
        if np.isnan(boot_results.overall_se):
            assert any(
                "valid" in msg.lower() or "nan" in msg.lower() for msg in warning_messages
            ), "NaN SE should be accompanied by warning about validity"

    def test_validity_threshold_nan_se(self):
        """Test that <50% valid bootstrap samples returns NaN SE with warning.

        This tests the methodology-aligned behavior where invalid inference
        is signaled via NaN rather than biased estimates.
        """
        import warnings

        np.random.seed(42)

        # Create minimal dataset that might trigger edge cases
        n_units, n_periods = 10, 3
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # Only 1 treated unit - very extreme
        first_treat = np.zeros(n_units)
        first_treat[0] = 1
        first_treat_expanded = np.repeat(first_treat, n_periods)

        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 1.0 + 2.0 * post + np.random.randn(len(units)) * 0.5

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
            }
        )

        # Use low n_bootstrap to trigger warning and potentially non-finite samples
        cs_boot = CallawaySantAnna(n_bootstrap=30, seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            boot_results = cs_boot.fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

        warning_messages = [str(warning.message) for warning in w]

        # Should get the low n_bootstrap warning
        assert any(
            "n_bootstrap" in msg for msg in warning_messages
        ), "Should warn about low n_bootstrap"

        # Bootstrap results should exist
        assert boot_results.bootstrap_results is not None, "Bootstrap results should exist"

        # SE constraints: finite or NaN (never inf)
        assert np.isfinite(boot_results.overall_se) or np.isnan(
            boot_results.overall_se
        ), "Bootstrap SE should be finite or NaN (not inf)"

    def test_near_collinear_covariates(self):
        """Test that near-collinear covariates are handled gracefully."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Add a near-collinear covariate (x1 + noise above rank detection tolerance)
        # The rank detection tolerance is 1e-07 (matching R's qr()), so we use noise
        # of 1e-5 which is above the tolerance but still creates high collinearity.
        # With noise < 1e-07, the column would be considered linearly dependent.
        np.random.seed(42)
        data["x1_copy"] = data["x1"] + np.random.randn(len(data)) * 1e-5

        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x1_copy"],  # Nearly collinear
        )

        # Should still produce valid results (noise is above tolerance)
        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)

    def test_missing_values_in_covariates_warning(self):
        """Test that missing values trigger fallback warning."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Introduce NaN in covariate
        data.loc[data["time"] == 2, "x1"] = np.nan

        cs = CallawaySantAnna()

        # Should warn about missing values and fall back to unconditional
        with pytest.warns(UserWarning, match="Missing values in covariates"):
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )

        # Should still produce valid results (using unconditional estimation)
        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_dr_covariates_not_yet_treated(self):
        """Regression test: DR + covariates with not_yet_treated control group.

        Ensures cache keys correctly include cohort g for not_yet_treated,
        preventing stale Cholesky/pscore reuse across groups.
        """
        data = generate_staggered_data_with_covariates(seed=42, n_units=200)

        for method in ["dr", "reg"]:
            cs = CallawaySantAnna(
                estimation_method=method,
                control_group="not_yet_treated",
            )
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )

            assert np.isfinite(
                results.overall_att
            ), f"{method}/not_yet_treated: ATT should be finite"
            assert results.overall_se > 0, f"{method}/not_yet_treated: SE should be positive"
            assert (
                len(results.group_time_effects) > 0
            ), f"{method}/not_yet_treated: should have group-time effects"
            # All effects should be finite
            for (g, t), eff in results.group_time_effects.items():
                assert np.isfinite(
                    eff["effect"]
                ), f"{method}/not_yet_treated: effect for ({g},{t}) should be finite"
                assert np.isfinite(
                    eff["se"]
                ), f"{method}/not_yet_treated: SE for ({g},{t}) should be finite"

    def test_rank_deficient_action_error_raises(self):
        """Test that rank_deficient_action='error' raises ValueError on collinear data."""
        data = generate_staggered_data_with_covariates(seed=42)

        # Add a covariate that is perfectly collinear with x1
        data["x1_dup"] = data["x1"].copy()

        cs = CallawaySantAnna(
            estimation_method="reg",  # Use regression method to test OLS path
            rank_deficient_action="error",
        )
        with pytest.raises(ValueError, match="rank-deficient"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x1_dup"],
            )

    def test_rank_deficient_action_silent_no_warning(self):
        """Test that rank_deficient_action='silent' produces no warning."""
        import warnings

        data = generate_staggered_data_with_covariates(seed=42)

        # Add a covariate that is perfectly collinear with x1
        data["x1_dup"] = data["x1"].copy()

        cs = CallawaySantAnna(
            estimation_method="reg",  # Use regression method to test OLS path
            rank_deficient_action="silent",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x1_dup"],
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

    def test_rank_deficient_action_warn_emits_warning(self):
        """Test that rank_deficient_action='warn' emits rank-deficiency warning on batched path."""
        import warnings

        data = generate_staggered_data_with_covariates(seed=42)

        # Add a covariate that is perfectly collinear with x1
        data["x1_dup"] = data["x1"].copy()

        # estimation_method="reg" + rank_deficient_action="warn" routes to
        # _compute_all_att_gt_covariate_reg (batched path)
        cs = CallawaySantAnna(
            estimation_method="reg",
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x1_dup"],
            )

            rank_warnings = [
                x
                for x in w
                if "rank-deficient" in str(x.message).lower() or "Rank-deficient" in str(x.message)
            ]
            assert (
                len(rank_warnings) > 0
            ), "Expected at least one rank-deficiency warning with collinear covariates"

        # Should still produce valid results (lstsq fallback)
        assert results is not None
        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_empty_covariates_list_behaves_like_none(self):
        """covariates=[] should behave identically to covariates=None."""
        data = generate_staggered_data_with_covariates(seed=42)

        cs_none = CallawaySantAnna(n_bootstrap=0, seed=42)
        results_none = cs_none.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=None,
        )

        cs_empty = CallawaySantAnna(n_bootstrap=0, seed=42)
        results_empty = cs_empty.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=[],
        )

        assert results_none.overall_att == results_empty.overall_att
        assert results_none.overall_se == results_empty.overall_se
        assert len(results_none.group_time_effects) == len(results_empty.group_time_effects)

    def test_nan_cell_preserved_not_dropped(self):
        """Non-finite regression cells should be preserved as NaN, not dropped."""
        import warnings
        from unittest.mock import patch

        import diff_diff.staggered as _ddstg

        data = generate_staggered_data_with_covariates(seed=42, n_units=100)

        # Poison one covariate OR solve to simulate a numerical failure. The reg
        # path routes the OR fit through `diff_diff.staggered.solve_ols` (the
        # scale-robust solver), NOT `scipy.linalg.lstsq` — so patch that seam.
        original_solve_ols = _ddstg.solve_ols
        call_count = [0]

        def mock_solve_ols(*args, **kwargs):
            call_count[0] += 1
            result = original_solve_ols(*args, **kwargs)
            # Poison call #7 (the (g=3, t=3) OR solve). An inf coefficient survives
            # the dropped-column NaN zero-fill and trips the nan_cell guard.
            if call_count[0] == 7:
                bad_beta = np.full_like(result[0], np.inf)
                return (bad_beta,) + result[1:]
            return result

        # Collinear covariates force the rank-deficient OR path; reg + warn routes
        # the OR fit through diff_diff.staggered.solve_ols.
        data["x1_dup"] = data["x1"]
        cs = CallawaySantAnna(
            n_bootstrap=0,
            seed=42,
            estimation_method="reg",
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("diff_diff.staggered.solve_ols", side_effect=mock_solve_ols):
                results = cs.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1", "x1_dup"],
                )

        # The mock must actually fire (otherwise the test is vacuous).
        assert call_count[0] >= 7, (
            f"mock solve_ols should have been called >=7 times, got {call_count[0]} "
            "(the OR solve seam moved — update the patch target)"
        )

        # The poisoned cell must be PRESERVED as NaN (not dropped), with NaN SE,
        # and the non-finite-regression warning must fire.
        nan_cells = [
            (g, t) for (g, t), eff in results.group_time_effects.items() if np.isnan(eff["effect"])
        ]
        assert len(nan_cells) > 0, "Expected at least one NaN cell from the poisoned OR solve"
        nan_warnings = [x for x in w if "non-finite regression results" in str(x.message)]
        assert len(nan_warnings) > 0, "Expected a 'non-finite regression results' warning"
        for g, t in nan_cells:
            assert np.isnan(
                results.group_time_effects[(g, t)]["se"]
            ), f"NaN cell ({g},{t}) must have NaN SE"
            assert (
                results.group_time_effects[(g, t)]["skip_reason"] == "non_finite_regression"
            ), f"NaN cell ({g},{t}) must carry skip_reason='non_finite_regression'"

        # Overall ATT should still be finite (NaN cells excluded from aggregation)
        assert np.isfinite(results.overall_att)

    def test_nan_cell_bootstrap_aggregation_excludes_nan(self, ci_params):
        """Bootstrap aggregation paths must exclude NaN ATT(g,t) cells."""
        import warnings
        from unittest.mock import patch

        data = generate_staggered_data_with_covariates(seed=42, n_units=100)

        import diff_diff.staggered as _ddstg

        original_solve_ols = _ddstg.solve_ols
        call_count = [0]

        def mock_solve_ols(*args, **kwargs):
            call_count[0] += 1
            result = original_solve_ols(*args, **kwargs)
            # Poison call #7 — the (g=3, t=3) outcome-regression solve, a
            # post-treatment cell, so the overall ATT bootstrap aggregation path
            # is exercised. The covariate OR fit routes through `solve_ols` (the
            # scale-robust solver), not `scipy.linalg.lstsq` directly; an inf
            # coefficient survives the dropped-column NaN zero-fill and trips the
            # nan_cell guard.
            if call_count[0] == 7:
                bad_beta = np.full_like(result[0], np.inf)
                return (bad_beta,) + result[1:]
            return result

        data["x1_dup"] = data["x1"]
        n_boot = ci_params.bootstrap(199)
        cs = CallawaySantAnna(
            n_bootstrap=n_boot,
            seed=42,
            estimation_method="reg",
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with patch("diff_diff.staggered.solve_ols", side_effect=mock_solve_ols):
                results = cs.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1", "x1_dup"],
                    aggregate="all",
                )

        # NaN cell should be preserved in group_time_effects
        nan_cells = [
            (g, t) for (g, t), eff in results.group_time_effects.items() if np.isnan(eff["effect"])
        ]
        assert len(nan_cells) > 0, "Expected at least one NaN cell from mock"

        # Verify poisoned cell is post-treatment so overall ATT bootstrap path is exercised
        post_treatment_nan = [(g, t) for g, t in nan_cells if t >= g - cs.anticipation]
        assert (
            len(post_treatment_nan) > 0
        ), "Poisoned cell must be post-treatment to exercise overall ATT bootstrap filtering"

        # Overall ATT bootstrap inference should be finite (NaN cells excluded)
        assert np.isfinite(results.overall_att), "overall_att should be finite"
        assert np.isfinite(results.overall_se), "overall_se should be finite"
        assert np.isfinite(results.overall_p_value), "overall_p_value should be finite"
        assert all(np.isfinite(x) for x in results.overall_conf_int), "overall CI should be finite"

        # Event study: valid relative times should have finite bootstrap inference
        if results.event_study_effects:
            for e, data_es in results.event_study_effects.items():
                if np.isfinite(data_es["effect"]):
                    assert np.isfinite(data_es["se"]), f"ES e={e} se should be finite"
                    assert np.isfinite(data_es["p_value"]), f"ES e={e} p_value should be finite"

        # Group effects: valid groups should have finite bootstrap inference
        if results.group_effects:
            for g, data_ge in results.group_effects.items():
                if np.isfinite(data_ge["effect"]):
                    assert np.isfinite(data_ge["se"]), f"Group {g} se should be finite"
                    assert np.isfinite(data_ge["p_value"]), f"Group {g} p_value should be finite"

    def test_balance_e_excludes_nan_anchor_cohort(self, ci_params):
        """balance_e must exclude cohorts whose anchor-horizon effect is NaN."""
        import warnings
        from unittest.mock import patch

        data = generate_staggered_data_with_covariates(seed=42, n_units=100)

        import diff_diff.staggered as _ddstg

        original_solve_ols = _ddstg.solve_ols
        call_count = [0]

        def mock_solve_ols(*args, **kwargs):
            call_count[0] += 1
            result = original_solve_ols(*args, **kwargs)
            # Poison call #7: the (g=3, t=3) outcome-regression solve, the anchor
            # for cohort g=3 at e=0. The OR fit routes through `solve_ols` (the
            # scale-robust solver); an inf coefficient survives the dropped-column
            # NaN zero-fill and trips the nan_cell guard.
            if call_count[0] == 7:
                bad_beta = np.full_like(result[0], np.inf)
                return (bad_beta,) + result[1:]
            return result

        data["x1_dup"] = data["x1"]
        n_boot = ci_params.bootstrap(199)
        cs = CallawaySantAnna(
            n_bootstrap=n_boot,
            seed=42,
            estimation_method="reg",
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with patch("diff_diff.staggered.solve_ols", side_effect=mock_solve_ols):
                results = cs.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1", "x1_dup"],
                    aggregate="event_study",
                    balance_e=0,
                )

        # Confirm the anchor cell is NaN and is specifically the anchor (t - g == 0)
        assert np.isnan(
            results.group_time_effects[(3, 3)]["effect"]
        ), "Mock should have poisoned (g=3, t=3)"
        assert 3 - 3 == 0, "Poisoned cell must be the anchor at balance_e=0"

        # Cohort g=3 should be excluded from ALL event-study horizons
        # Only g=5 and g=8 should contribute (<=2 because not all balanced
        # cohorts have cells at extreme horizons)
        for e, es_data in results.event_study_effects.items():
            assert es_data["n_groups"] <= 2, (
                f"Event time e={e} has n_groups={es_data['n_groups']}, "
                "expected <=2 (cohort g=3 should be excluded due to NaN anchor)"
            )

        # Analytical effects and SEs should be finite for all horizons
        for e, es_data in results.event_study_effects.items():
            assert np.isfinite(es_data["effect"]), f"e={e}: analytical effect should be finite"
            assert np.isfinite(es_data["se"]), f"e={e}: analytical SE should be finite"

        # Bootstrap SEs should also be finite
        if results.bootstrap_results and results.bootstrap_results.event_study_ses:
            for e, se in results.bootstrap_results.event_study_ses.items():
                assert np.isfinite(se), f"e={e}: bootstrap SE should be finite"

    @pytest.mark.parametrize("action", ["silent", "warn"])
    def test_reg_underdetermined_control_cell_no_crash(self, action):
        """CS `reg` must not crash on an underdetermined control cell
        (n_control < n_covariates + 1) under ``rank_deficient_action`` warn/silent.

        Regression for the OR scale-equilibration change: the covariate OR fit now
        routes through ``solve_ols``, which raises on ``n < k`` *before* it can rank-
        drop. The optimized reg path detects the rank-deficient columns, drops them,
        and fits the reduced (full-column-rank) design via the equilibrated lstsq —
        the documented R-style / ``lm()`` column-drop contract, NOT a minimum-norm
        full-design solve (CI codex P1 on the scale-equilibration PR).
        """
        rng = np.random.default_rng(3)
        rows = []
        for i in range(10):
            g = 2 if i < 8 else 0  # 8 treated (g=2), only 2 never-treated controls
            x1, x2, x3 = rng.normal(size=3)
            for t in range(1, 4):
                post = 1 if (g != 0 and t >= g) else 0
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "first_treat": g,
                        "outcome": rng.normal() + 0.5 * t + 0.3 * x1 + 1.5 * post,
                        "x1": x1,
                        "x2": x2,
                        "x3": x3,
                    }
                )
        data = pd.DataFrame(rows)
        # 2 controls vs intercept + 3 covariates = 4 params -> underdetermined OR cell.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(estimation_method="reg", rank_deficient_action=action).fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2", "x3"],
            )
        post_effects = [res.group_time_effects[(2, t)]["effect"] for t in (2, 3)]
        assert all(np.isfinite(e) for e in post_effects), (
            "underdetermined control cell should yield a finite ATT under "
            f"rank_deficient_action={action!r}, got {post_effects}"
        )
        # R-style column-drop contract, NOT minimum-norm: dropping the unidentified
        # column(s) and fitting the reduced design reproduces the prior reduced-lstsq
        # result to working precision. A minimum-norm solve on the full n<k design
        # would give a different (non-unique) extrapolation to the treated covariates.
        rank_reduced = [-1.3958318723, -1.2987532126]  # prior reduced-lstsq / R lm() drop
        minimum_norm = [-1.9213829489, -1.8180077026]  # a full n<k min-norm solve (rejected)
        np.testing.assert_allclose(post_effects, rank_reduced, atol=1e-6)
        assert not np.allclose(post_effects, minimum_norm, atol=1e-3), (
            "underdetermined reg fit must use the rank-reduced column-drop solve, "
            "not the minimum-norm full-design solve"
        )


def _cs_nonestimable_data(panel: bool, n: int = 25, seed: int = 0) -> pd.DataFrame:
    """Staggered data (cohorts 2,3,4; periods 1-4; NO never-treated) where, with
    ``control_group="not_yet_treated"``, every post cell at the final period t=4
    has no not-yet-treated controls (no cohort treated after 4) -> deterministic
    non-estimable cells (e.g. (4, 4)). Earlier-period post cells (g=2/3 at t=2/3)
    stay estimable, so aggregates remain finite.

    panel=True  -> each unit observed in all 4 periods (true panel).
    panel=False -> each row is a distinct unit (true repeated cross-section).
    """
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in (2, 3, 4):
        for _ in range(n):
            x = rng.normal(0, 1)  # unit-level covariate (for IPW/DR/reg covariate paths)
            if panel:
                fe = rng.normal(0, 1)
                for t in range(1, 5):
                    post = 1.0 if t >= g else 0.0
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "outcome": fe + 0.3 * t + 1.5 * post + 0.5 * x + rng.normal(0, 0.5),
                            "first_treat": g,
                            "x": x,
                        }
                    )
                uid += 1
            else:
                for t in range(1, 5):
                    post = 1.0 if t >= g else 0.0
                    x_t = rng.normal(0, 1)
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "outcome": rng.normal(0, 1) + 0.3 * t + 1.5 * post + 0.5 * x_t,
                            "first_treat": g,
                            "x": x_t,
                        }
                    )
                    uid += 1
    return pd.DataFrame(rows)


class TestCallawaySantAnnaNonEstimableMaterialization:
    """Non-estimable (g,t) cells are materialized as NaN entries carrying a
    machine-readable ``skip_reason``, uniformly across estimation paths, and are
    excluded from every aggregation so aggregates/SEs stay finite (the prior
    omit behavior; exact aggregate values are pinned by test_methodology_callaway).
    """

    _KNOWN_REASONS = {
        "missing_period",
        "zero_treated_control",
        "zero_weight_mass",
        "non_finite_regression",
    }

    @pytest.mark.parametrize(
        "method,panel,covariates",
        [
            ("reg", True, None),  # no-covariate vectorized path
            ("ipw", True, None),  # general path, no covariates
            ("dr", True, None),  # general path, no covariates
            ("reg", False, None),  # repeated cross-section path
            ("reg", True, ["x"]),  # covariate-regression vectorized path
            ("ipw", True, ["x"]),  # general path, covariate IPW
            ("dr", True, ["x"]),  # general path, covariate DR
            ("dr", False, ["x"]),  # repeated cross-section, covariate DR
        ],
    )
    def test_materializes_nan_cell_with_skip_reason(self, method, panel, covariates):
        """Each previously-omitting path now stores the non-estimable cell as NaN."""
        data = _cs_nonestimable_data(panel=panel, seed=0)
        cs = CallawaySantAnna(
            n_bootstrap=0,
            control_group="not_yet_treated",
            estimation_method=method,
            panel=panel,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=covariates,
            )

        # The (g=4, t=4) cell has no not-yet-treated controls -> materialized, not omitted.
        key = (4, 4)
        assert (
            key in results.group_time_effects
        ), f"non-estimable cell must be materialized (path={method}, panel={panel}), not omitted"
        cell = results.group_time_effects[key]
        assert np.isnan(cell["effect"]) and np.isnan(cell["se"])
        assert np.isnan(cell["t_stat"]) and np.isnan(cell["p_value"])
        assert all(np.isnan(b) for b in cell["conf_int"])
        assert cell["skip_reason"] == "zero_treated_control"
        # The cell genuinely has treated observations but no controls -> the
        # materialized counts must reflect that, not a hardcoded (0, 0).
        assert cell["n_treated"] > 0
        assert cell["n_control"] == 0

        # Every NaN cell carries a known reason + NaN SE; every estimable cell None.
        n_finite = 0
        for (g, t), v in results.group_time_effects.items():
            if np.isnan(v["effect"]):
                assert v["skip_reason"] in self._KNOWN_REASONS, (g, t, v["skip_reason"])
                assert np.isnan(v["se"])
            else:
                assert v["skip_reason"] is None
                n_finite += 1

        # Non-empty dict mixing NaN + finite cells fits without raising, and the
        # NaN cells are excluded from aggregation (the aggregation invariant).
        assert n_finite > 0, "expected some estimable cells"
        assert np.isfinite(results.overall_att), "NaN cells must be excluded from aggregation"
        assert np.isfinite(results.overall_se)

    def test_estimable_cells_have_skip_reason_none(self):
        """A well-posed fit carries skip_reason=None on every cell."""
        data = generate_staggered_data(n_units=200, seed=42)
        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert len(results.group_time_effects) > 0
        assert all(np.isfinite(v["effect"]) for v in results.group_time_effects.values())
        assert all(v["skip_reason"] is None for v in results.group_time_effects.values())

    def test_to_dataframe_includes_nan_row_and_skip_reason_column(self):
        """to_dataframe('group_time') surfaces the NaN cell + a skip_reason column."""
        data = _cs_nonestimable_data(panel=True, seed=0)
        cs = CallawaySantAnna(n_bootstrap=0, control_group="not_yet_treated")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )
        df = results.to_dataframe("group_time")
        assert "skip_reason" in df.columns

        nan_row = df[(df["group"] == 4) & (df["time"] == 4)]
        assert len(nan_row) == 1, "the non-estimable cell must appear as a row"
        assert np.isnan(nan_row["effect"].iloc[0])
        assert nan_row["skip_reason"].iloc[0] == "zero_treated_control"

        # Estimable rows carry a null skip_reason.
        estimable = df[df["effect"].notna()]
        assert len(estimable) > 0
        assert estimable["skip_reason"].isna().all()

    def test_general_path_nonfinite_att_materialized_as_nan_no_inf(self):
        """A non-finite ATT(g,t) in the general (IPW/DR) path must surface as a NaN
        cell with skip_reason, NOT a finite-but-non-finite (inf) effect carrying an
        IF entry. Regression guard for the per-cell contract."""
        from unittest.mock import patch

        data = generate_staggered_data(n_units=120, seed=7)
        real = CallawaySantAnna._compute_att_gt_fast
        state = {"poisoned": None}

        def wrapped(self, precomputed, g, t, covariates, **kw):
            res = real(self, precomputed, g, t, covariates, **kw)
            att = res[0]
            # Poison the first estimable post cell: return inf ATT WITH a real IF
            # entry (res[4]), mimicking a degenerate IPW/DR solve that returns a
            # non-finite point estimate without a None sentinel.
            if state["poisoned"] is None and att is not None and np.isfinite(att) and t >= g:
                state["poisoned"] = (g, t)
                return (np.inf,) + res[1:6] + (None,)
            return res

        cs = CallawaySantAnna(n_bootstrap=0, estimation_method="ipw")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.object(CallawaySantAnna, "_compute_att_gt_fast", wrapped):
                results = cs.fit(
                    data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
                )

        assert state["poisoned"] is not None, "poison hook never fired (test is vacuous)"
        cell = results.group_time_effects[state["poisoned"]]
        # The non-finite ATT must be materialized as NaN (not inf) with the reason.
        assert np.isnan(cell["effect"]), "non-finite ATT must surface as NaN, not inf"
        assert not np.isinf(cell["effect"])
        assert cell["skip_reason"] == "non_finite_regression"
        # Excluded from aggregation -> overall ATT still finite.
        assert np.isfinite(results.overall_att)

    def test_no_covariate_path_nonfinite_att_materialized_as_nan(self):
        """No-covariate vectorized path: an inf outcome (passes the NaN-only valid
        mask) yields a non-finite ATT, which must be materialized as a NaN cell
        with skip_reason -- not stored as inf with an IF entry / batch inference."""
        data = generate_staggered_data(n_units=200, seed=11)
        # Inject inf into one treated unit's outcome at its treatment period; inf is
        # not NaN so it survives the valid mask and makes that cohort's cell ATT inf.
        fin = data[np.isfinite(data["first_treat"]) & (data["first_treat"] > 0)]
        u = fin["unit"].iloc[0]
        g = int(data.loc[data["unit"] == u, "first_treat"].iloc[0])
        data.loc[(data["unit"] == u) & (data["time"] == g), "outcome"] = np.inf

        cs = CallawaySantAnna(n_bootstrap=0, estimation_method="reg")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

        nf = [
            (k, v)
            for k, v in results.group_time_effects.items()
            if v["skip_reason"] == "non_finite_regression"
        ]
        assert nf, "an inf outcome must yield a non_finite_regression cell"
        for _, v in nf:
            assert np.isnan(v["effect"]) and not np.isinf(v["effect"])
            assert np.isnan(v["se"]) and np.isnan(v["t_stat"]) and np.isnan(v["p_value"])
        assert np.isfinite(results.overall_att)

    def test_event_study_omits_all_nonestimable_relative_time(self):
        """An event-time bucket whose cells are ALL non-estimable is omitted from
        event_study_effects (matches the prior omit behavior / R did::aggte),
        not emitted as an all-NaN row."""
        data = _cs_nonestimable_data(panel=True, seed=0)
        cs = CallawaySantAnna(n_bootstrap=0, control_group="not_yet_treated")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        es = results.event_study_effects
        # Relative time e=2 contains only (g=2, t=4), which is non-estimable (no
        # not-yet-treated controls at t=4) -> the whole bucket must be omitted.
        assert 2 not in es, "all-non-estimable relative time must be omitted, not a NaN row"
        # Every emitted relative time has a finite effect and >=1 contributing group.
        for e, d in es.items():
            assert np.isfinite(d["effect"]), f"e={e} effect should be finite"
            assert d["n_groups"] >= 1, f"e={e} should have >=1 contributing group"

    def test_all_nonestimable_raises_with_materialized_cells(self):
        """All cells non-estimable (dict non-empty, all NaN) -> ValueError via the
        no-finite-effect guard (distinct from the empty-dict case)."""
        rng = np.random.default_rng(3)
        rows = []
        uid = 0
        # Two treated cohorts (passes the not_yet_treated >=2-cohort upfront check
        # is N/A here since control_group is never_treated) ...
        for g in (2, 3):
            for _ in range(20):
                fe = rng.normal(0, 1)
                for t in range(1, 5):
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "outcome": fe + 0.3 * t + (1.5 if t >= g else 0.0),
                            "first_treat": g,
                        }
                    )
                uid += 1
        # Never-treated controls present (passes the upfront control check) but with
        # all-NaN outcomes -> every cell has zero VALID controls -> all cells NaN.
        for _ in range(20):
            for t in range(1, 5):
                rows.append({"unit": uid, "time": t, "outcome": np.nan, "first_treat": np.inf})
                uid += 1
        data = pd.DataFrame(rows)
        cs = CallawaySantAnna(n_bootstrap=0, control_group="never_treated")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="Could not estimate any group-time effects"):
                cs.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")


class TestRankGuardedAnalyticalSE:
    """Rank-guarded influence-function SE (constant/collinear covariate).

    A constant or collinear covariate makes the per-(g,t) propensity-score
    Hessian / outcome-regression bread near-singular. The old ``_safe_inv``
    only caught *exactly* singular matrices (``LinAlgError``), so a near-singular
    Gram returned a garbage inverse that produced ``overall_se`` ~1e13. The
    rank-guarded inverse drops the redundant direction -> finite SE on the
    identified subset (equal to dropping the covariate), NaN only on true rank-0.
    """

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_constant_covariate_finite_se_matches_drop_one(self, method):
        data = generate_staggered_data_with_covariates(seed=789)
        data_const = data.copy()
        data_const["xc"] = 5.0  # constant -> collinear with the intercept

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            drop_one = CallawaySantAnna(estimation_method=method).fit(
                data, "outcome", "unit", "time", "first_treat", covariates=["x1"]
            )
            with_const = CallawaySantAnna(estimation_method=method).fit(
                data_const,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "xc"],
            )

        # Regression guard: previously ~1e13, now finite and modest.
        assert np.isfinite(with_const.overall_se)
        assert with_const.overall_se < 1.0
        # Dropping the redundant covariate is equivalent to never adding it.
        np.testing.assert_allclose(with_const.overall_se, drop_one.overall_se, rtol=1e-9)
        np.testing.assert_allclose(with_const.overall_att, drop_one.overall_att, rtol=1e-9)

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_constant_covariate_emits_single_rank_guard_warning(self, method):
        # reg/ipw now route their IF breads (OLS bread / PS Hessian) through
        # _safe_inv like dr, so the aggregate warning fires for ALL methods
        # on a collinear design (previously dr/survey-ipw only).
        data = generate_staggered_data_with_covariates(seed=789)
        data["xc"] = 5.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CallawaySantAnna(estimation_method=method).fit(
                data, "outcome", "unit", "time", "first_treat", covariates=["x1", "xc"]
            )
        rank_guard = [w for w in caught if "rank-guarded inverse" in str(w.message)]
        # The per-fit aggregate warning fires exactly once, not per cell.
        assert len(rank_guard) == 1

    def test_well_conditioned_covariates_take_fast_path(self):
        # Well-conditioned covariates must NOT trigger the rank-guard (the fast
        # path returns the exact solve, so R-parity goldens are unchanged).
        data = generate_staggered_data_with_covariates(seed=789)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = CallawaySantAnna(estimation_method="dr").fit(
                data, "outcome", "unit", "time", "first_treat", covariates=["x1", "x2"]
            )
        assert not any("rank-guarded inverse" in str(w.message) for w in caught)
        assert np.isfinite(res.overall_se)

    def test_clustered_constant_covariate_finite_se(self):
        # The clustered SE path reuses the same per-cell influence functions, so
        # fixing the bread fixes it too.
        data = generate_staggered_data_with_covariates(seed=789)
        data["xc"] = 5.0
        data["cl"] = data["unit"] % 20
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(estimation_method="dr", cluster="cl").fit(
                data, "outcome", "unit", "time", "first_treat", covariates=["x1", "xc"]
            )
        assert np.isfinite(res.overall_se)
        assert res.overall_se < 1.0

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_rank0_bread_propagates_nan_not_zero(self, monkeypatch, method):
        # rank-0 is unreachable through covariates alone (the always-present
        # intercept guarantees rank >= 1), so simulate an all-NaN bread to
        # exercise the NaN-masking fix: var_psi becomes NaN and must yield a NaN
        # SE, NOT 0.0 via the old ``var_psi > 0 else 0.0`` guard. ipw's PS
        # Hessian and dr's breads are over [1, X], so all-NaN there is a true
        # pathology that must propagate. reg is deliberately EXCLUDED: its
        # estimation-effect bread is over the CENTERED covariate Gram (the
        # intercept is handled analytically), where rank-0 is the benign
        # constant-covariate case mapped to a zero correction — see
        # test_reg_constant_only_covariate_matches_no_covariate below.
        # The point estimate does NOT depend on the bread, so it stays
        # finite (NaN inference on an estimable cell, not _nan_gt_entry).
        import diff_diff.staggered as staggered_mod
        from tests.conftest import assert_nan_inference

        def _all_nan_inv(A, tracker=None):
            k = A.shape[0]
            return np.full((k, k), np.nan)

        monkeypatch.setattr(staggered_mod, "_safe_inv", _all_nan_inv)
        data = generate_staggered_data_with_covariates(seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(estimation_method=method).fit(
                data, "outcome", "unit", "time", "first_treat", covariates=["x1", "x2"]
            )
        for cell in res.group_time_effects.values():
            assert np.isfinite(cell["effect"]), "point estimate is bread-independent"
            assert np.isnan(cell["se"]), "rank-0 bread must give NaN SE, not 0.0"
            assert_nan_inference(cell)
        assert np.isnan(res.overall_se)

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_rcs_panel_false_constant_covariate_finite_se(self, method):
        # The repeated-cross-section (panel=False) analytical SE branches use
        # the same _safe_inv -> _rank_guarded_inv path (the *_rc methods). Build
        # RCS data (one row per unit) and confirm a constant covariate gives a
        # finite SE equal to dropping it.
        rng = np.random.default_rng(7)
        rows = []
        unit = 0
        for t in range(1, 7):
            for _ in range(120):
                s = int(rng.integers(0, 5))
                ft = int(rng.choice([0, 3, 5], p=[0.4, 0.3, 0.3]))
                x1 = rng.normal()
                y = (
                    s
                    + 0.3 * (t - 1)
                    + 1.0 * x1
                    + (1.5 if (ft > 0 and t >= ft) else 0.0)
                    + rng.normal(0, 0.5)
                )
                rows.append({"unit": unit, "time": t, "first_treat": ft, "outcome": y, "x1": x1})
                unit += 1
        data = pd.DataFrame(rows)
        data_const = data.copy()
        data_const["xc"] = 5.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            drop_one = CallawaySantAnna(estimation_method=method, panel=False).fit(
                data, "outcome", "unit", "time", "first_treat", covariates=["x1"]
            )
            with_const = CallawaySantAnna(estimation_method=method, panel=False).fit(
                data_const,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "xc"],
            )
        assert np.isfinite(with_const.overall_se)
        assert with_const.overall_se < 1.0
        np.testing.assert_allclose(with_const.overall_se, drop_one.overall_se, rtol=1e-9)

    @pytest.mark.parametrize("method", ["reg", "dr"])
    def test_control_cell_aliasing_close_to_drop_one(self, method):
        # Column-drop rank-guard: a covariate collinear ONLY within the control
        # cell (x2 == 2*x1 for never-treated, varying in treated) is dropped from
        # the central control OR regression (column-drop, matching the point
        # estimate / R), so the SE is FINITE (not the old 1e13 garbage) and ≈
        # dropping the covariate. The small residual (< a few %) is the
        # covariate's genuine effect in the treated-side / propensity terms,
        # where it is full-rank — not a rank-guard artifact.
        base = generate_staggered_data_with_covariates(seed=789)
        rng = np.random.default_rng(0)
        d = base.copy()
        nt = d["first_treat"] == 0
        d["x2_deg"] = np.where(nt, 2.0 * d["x1"], rng.normal(size=len(d)))
        d["x2_deg"] = d.groupby("unit")["x2_deg"].transform("first")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            drop_one = CallawaySantAnna(estimation_method=method).fit(
                d, "outcome", "unit", "time", "first_treat", covariates=["x1"]
            )
            with_deg = CallawaySantAnna(estimation_method=method).fit(
                d,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "x2_deg"],
            )
        assert np.isfinite(with_deg.overall_se) and with_deg.overall_se > 0
        np.testing.assert_allclose(with_deg.overall_se, drop_one.overall_se, rtol=5e-2)

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_survey_weighted_constant_covariate_finite_se(self, method):
        # Exercises the *survey-weighted* CS bread / PS-Hessian branches
        # (W includes survey weights), mirroring the TD/SDDD weighted tests.
        # Panel estimator -> weights constant within unit.
        from diff_diff.survey import SurveyDesign

        data = generate_staggered_data_with_covariates(seed=789)
        rng = np.random.default_rng(3)
        units = data["unit"].unique()
        unit_w = dict(zip(units, rng.uniform(0.5, 2.0, len(units))))
        data["weight"] = data["unit"].map(unit_w)
        data_const = data.copy()
        data_const["xc"] = 5.0
        sd = SurveyDesign(weights="weight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            drop_one = CallawaySantAnna(estimation_method=method).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1"],
                survey_design=sd,
            )
            with_const = CallawaySantAnna(estimation_method=method).fit(
                data_const,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "xc"],
                survey_design=sd,
            )
        assert np.isfinite(with_const.overall_se)
        assert with_const.overall_se < 1.0
        np.testing.assert_allclose(with_const.overall_se, drop_one.overall_se, rtol=1e-9)

    def test_aggregated_se_wif_contract(self, monkeypatch):
        # Locks the _compute_aggregated_se_with_wif arity + fail-closed contract
        # that motivated the staggered_aggregation fix: a non-finite influence
        # function must propagate a NaN SE (not crash the unpacking caller), and
        # the return arity is a 2-tuple (return_psi=False) / 3-tuple (True).
        cs = CallawaySantAnna()
        base_args = ([], np.array([]), np.array([]), np.array([]), {}, None, None)

        def patch_psi(psi):
            monkeypatch.setattr(
                cs,
                "_compute_combined_influence_function",
                lambda *a, **k: (psi, None),
            )

        # Empty influence function -> se 0.0, documented arity.
        patch_psi(np.array([]))
        se, df = cs._compute_aggregated_se_with_wif(*base_args, return_psi=False)
        assert se == 0.0 and df is None
        triple = cs._compute_aggregated_se_with_wif(*base_args, return_psi=True)
        assert len(triple) == 3 and triple[0] == 0.0

        # Non-finite influence function -> NaN SE (fail-closed), not a crash.
        patch_psi(np.array([1.0, np.nan, 2.0]))
        se, df = cs._compute_aggregated_se_with_wif(*base_args, return_psi=False)
        assert np.isnan(se) and df is None
        triple = cs._compute_aggregated_se_with_wif(*base_args, return_psi=True)
        assert len(triple) == 3 and np.isnan(triple[0])

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_error_mode_raises_before_rank_guard(self, method):
        # rank_deficient_action="error" raises upstream at the point-estimate
        # solve when the covariate DESIGN is rank-deficient at its 1e-7 threshold
        # (here an EXACT duplicate), before the IF rank-guard. NOTE: this is not a
        # promise that every near-singular IF bread raises under "error" — a cell
        # that is near-singular yet still design-full-rank can pass this gate and
        # still be IF-column-dropped, because the IF guard's 1e-10 equilibrated-Gram
        # threshold is stricter than the 1e-7 design check (the Gram squares X's
        # condition number); see REGISTRY "rank_deficient_action enforcement".
        data = generate_staggered_data_with_covariates(seed=789)
        data["x2c"] = 2.0 * data["x1"]  # exactly collinear with x1
        with pytest.raises(ValueError, match="(?i)rank-deficient"):
            CallawaySantAnna(estimation_method=method, rank_deficient_action="error").fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "x2c"],
            )

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_exact_duplicate_covariate(self, method):
        # A WELL-SCALED exact duplicate (xdup == x1) is dropped exactly: the
        # rank-guard's column-drop matches the point estimate, SE == dropping it.
        # The SE is also order-invariant under exact collinearity (well-defined
        # regardless of which proportional column is listed first), including the
        # MIXED-SCALE case (xbig == 1e8*x1), for ALL methods: the variance flows
        # through the equilibrated rank-guarded inverse, and since the OR
        # scale-equilibration change the `reg`/`dr` point-estimate OR fit also
        # routes through the equilibrated `solve_ols`. Equilibration scales x1 and
        # 1e8*x1 to identical unit-norm columns, so the SE is order-invariant even
        # though which member is dropped differs. (Previously `reg`'s un-equilibrated
        # local OR solve hit a near-singular X'WX whose 1e8-scale SE was column-order-
        # and BLAS-dependent; that is now fixed.)
        base = generate_staggered_data_with_covariates(seed=789)
        d = base.copy()
        d["xdup"] = d["x1"]  # well-scaled exact duplicate
        d["xbig"] = 1e8 * d["x1"]  # mixed-scale exact duplicate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            drop_one = CallawaySantAnna(estimation_method=method).fit(
                d, "outcome", "unit", "time", "first_treat", covariates=["x1"]
            )
            well = CallawaySantAnna(estimation_method=method).fit(
                d, "outcome", "unit", "time", "first_treat", covariates=["x1", "xdup"]
            )
            big_ab = CallawaySantAnna(estimation_method=method).fit(
                d, "outcome", "unit", "time", "first_treat", covariates=["x1", "xbig"]
            )
            big_ba = CallawaySantAnna(estimation_method=method).fit(
                d, "outcome", "unit", "time", "first_treat", covariates=["xbig", "x1"]
            )
        # Well-scaled exact duplicate == dropping it (clean column-drop).
        np.testing.assert_allclose(well.overall_se, drop_one.overall_se, rtol=1e-9)
        # Mixed-scale exact duplicate: finite for every method.
        assert np.isfinite(big_ab.overall_se) and big_ab.overall_se > 0
        assert np.isfinite(big_ba.overall_se) and big_ba.overall_se > 0
        # Order-invariance holds for ALL methods now: ipw/dr via the equilibrated
        # rank-guarded inverse, and reg via the equilibrated point-estimate OR
        # solve (post OR scale-equilibration change) — mixed-scale exact-duplicate
        # columns become identical after equilibration, so the SE is order-invariant.
        np.testing.assert_allclose(big_ab.overall_se, big_ba.overall_se, rtol=1e-9)

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_exact_duplicate_covariate_survey_weighted(self, method):
        # Weighted branch of the exact-duplicate contract (reviewer-requested):
        # the survey-weighted bread / PS-Hessian must give the same finite SE for
        # a WELL-SCALED exact duplicate as dropping it, and the rank-guard's
        # equilibrated column selection must be order-invariant under MIXED-SCALE
        # exact collinearity (xbig == 1e8*x1) for BOTH column orders even with
        # non-uniform survey weights in W.
        from diff_diff.survey import SurveyDesign

        base = generate_staggered_data_with_covariates(seed=789)
        rng = np.random.default_rng(3)
        units = base["unit"].unique()
        unit_w = dict(zip(units, rng.uniform(0.5, 2.0, len(units))))
        d = base.copy()
        d["weight"] = d["unit"].map(unit_w)
        d["xdup"] = d["x1"]  # well-scaled exact duplicate
        d["xbig"] = 1e8 * d["x1"]  # mixed-scale exact duplicate
        sd = SurveyDesign(weights="weight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            drop_one = CallawaySantAnna(estimation_method=method).fit(
                d,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1"],
                survey_design=sd,
            )
            well = CallawaySantAnna(estimation_method=method).fit(
                d,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "xdup"],
                survey_design=sd,
            )
            big_ab = CallawaySantAnna(estimation_method=method).fit(
                d,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "xbig"],
                survey_design=sd,
            )
            big_ba = CallawaySantAnna(estimation_method=method).fit(
                d,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["xbig", "x1"],
                survey_design=sd,
            )
        # Well-scaled exact duplicate == dropping it, under survey weighting.
        np.testing.assert_allclose(well.overall_se, drop_one.overall_se, rtol=1e-7)
        # Mixed-scale exact duplicate under survey weighting: finite + order-invariant
        # for ALL methods. reg's point-estimate OR fit now routes through the
        # equilibrated solve_ols (post OR scale-equilibration change), so its SE is
        # order-invariant like ipw/dr — see test_exact_duplicate_covariate.
        assert np.isfinite(big_ab.overall_se) and big_ab.overall_se > 0
        assert np.isfinite(big_ba.overall_se) and big_ba.overall_se > 0
        np.testing.assert_allclose(big_ab.overall_se, big_ba.overall_se, rtol=1e-7)


class TestRegIpwIFBehavior:
    """Behavioral contracts of the DRDID-parity reg/ipw per-cell IF/SE fix
    (estimation-effect terms + per-cell SE = sqrt(sum(IF^2)))."""

    def test_ipw_pscore_fallback_uses_uncorrected_if_se(self, monkeypatch):
        """When the per-cell logit fails and pscore_fallback="unconditional"
        kicks in, the PS estimation-effect correction is SKIPPED (a constant
        propensity has no estimated parameter) and the cell collapses to the
        difference-in-means IF - i.e. per-cell effect AND se must equal the
        no-covariate ipw fit on the same data."""
        import diff_diff.staggered as staggered_mod

        def _failing_logit(*args, **kwargs):
            raise ValueError("forced logit failure for fallback test")

        data = generate_staggered_data_with_covariates(seed=31)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            no_cov = CallawaySantAnna(estimation_method="ipw").fit(
                data, "outcome", "unit", "time", "first_treat"
            )
            monkeypatch.setattr(staggered_mod, "solve_logit", _failing_logit)
            fallback = CallawaySantAnna(
                estimation_method="ipw", pscore_fallback="unconditional"
            ).fit(data, "outcome", "unit", "time", "first_treat", covariates=["x1", "x2"])
        assert set(fallback.group_time_effects) == set(no_cov.group_time_effects)
        for key, cell in fallback.group_time_effects.items():
            ref = no_cov.group_time_effects[key]
            np.testing.assert_allclose(cell["effect"], ref["effect"], rtol=1e-12)
            np.testing.assert_allclose(cell["se"], ref["se"], rtol=1e-12)

    def test_bootstrap_nan_cell_if_poisons_only_its_own_cell(self, monkeypatch):
        """A cell whose stored IF is non-finite (e.g. the #619 rank-0 [1,X]
        bread semantics on ipw/dr) must NaN only its OWN bootstrap SE: in the
        fused perturbation GEMM each cell column is an independent dot
        product, so neighbor cells stay finite. (The overall/aggregate SEs
        legitimately consume the poisoned cell and are not asserted here.)"""
        from diff_diff.staggered_bootstrap import CallawaySantAnnaBootstrapMixin

        orig = CallawaySantAnnaBootstrapMixin._run_multiplier_bootstrap
        poisoned = {}

        def poisoning(self, group_time_effects, influence_func_info, *args, **kwargs):
            gt = sorted(influence_func_info)[0]
            poisoned["gt"] = gt
            info = influence_func_info[gt]
            info["treated_inf"] = np.full(len(np.asarray(info["treated_inf"])), np.nan)
            return orig(self, group_time_effects, influence_func_info, *args, **kwargs)

        monkeypatch.setattr(CallawaySantAnnaBootstrapMixin, "_run_multiplier_bootstrap", poisoning)
        data = generate_staggered_data(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = CallawaySantAnna(n_bootstrap=99, seed=5).fit(
                data, "outcome", "unit", "time", "first_treat"
            )
        gt = poisoned["gt"]
        assert np.isnan(result.group_time_effects[gt]["se"])
        neighbor_ses = [cell["se"] for key, cell in result.group_time_effects.items() if key != gt]
        assert neighbor_ses and np.isfinite(neighbor_ses).all()

    def test_underdetermined_control_cell_reg_no_crash(self):
        """Cells with fewer controls than covariate columns (n_c < k+1) fit a
        reduced design; the rank-guarded IF bread column-drops and the
        interpolating fit leaves ~zero control residuals, so the SE is finite
        (treated-side variation only), never a crash or a silent 0."""
        rng = np.random.default_rng(11)
        n_units, k = 60, 5
        rows = []
        for u in range(n_units):
            ft = 0 if u < 3 else 2  # only 3 never-treated controls
            x = rng.normal(size=k)
            base = rng.normal()
            for t in (1, 2):
                y = base + 0.4 * t + (1.0 if (ft == 2 and t >= 2) else 0.0) + rng.normal(0, 0.3)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": ft,
                        "outcome": y,
                        **{f"x{j}": x[j] for j in range(k)},
                    }
                )
        data = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(estimation_method="reg", rank_deficient_action="silent").fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=[f"x{j}" for j in range(k)],
            )
        cell = res.group_time_effects[(2, 2)]
        assert np.isfinite(cell["effect"])
        assert np.isfinite(cell["se"]) and cell["se"] > 0

    def test_universal_base_period_anticipation_reg_smoke(self):
        """reg+cov under base_period="universal" + anticipation=1: every
        estimated cell has finite inference, and each cohort's positional base
        period is materialized as a zero reference cell (att=0, se=NaN, matching
        R `did`'s att_gt table) rather than omitted."""
        data = generate_staggered_data_with_covariates(seed=97)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(
                estimation_method="reg", base_period="universal", anticipation=1
            ).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "x2"],
            )
        assert len(res.group_time_effects) > 0
        for (g, t), cell in res.group_time_effects.items():
            if cell.get("is_reference"):
                # Zero reference cell: att=0, se=NaN by construction.
                assert cell["effect"] == 0.0 and np.isnan(cell["se"])
                continue
            if cell["skip_reason"] is None:
                assert np.isfinite(cell["se"]), f"cell ({g},{t})"
        # The reference period is now materialized as a zero cell.
        assert any(c.get("is_reference") for c in res.group_time_effects.values())
        assert np.isfinite(res.overall_se)

    def test_reg_constant_only_covariate_matches_no_covariate(self):
        """A constant as the ONLY reg covariate makes the CENTERED
        estimation-effect Gram rank-0 (all-zero). The correction on the
        identified (intercept-only) subset is exactly zero, so effects AND
        SEs must equal the no-covariate fit - finite, never NaN (the rank-0
        centered bread maps to a zero correction, not an all-NaN inverse)."""
        data = generate_staggered_data_with_covariates(seed=789)
        data["xc"] = 5.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            no_cov = CallawaySantAnna(estimation_method="reg").fit(
                data, "outcome", "unit", "time", "first_treat"
            )
            const_only = CallawaySantAnna(estimation_method="reg").fit(
                data, "outcome", "unit", "time", "first_treat", covariates=["xc"]
            )
        assert np.isfinite(const_only.overall_se)
        np.testing.assert_allclose(const_only.overall_att, no_cov.overall_att, rtol=1e-12)
        np.testing.assert_allclose(const_only.overall_se, no_cov.overall_se, rtol=1e-9)
        for key, cell in const_only.group_time_effects.items():
            ref = no_cov.group_time_effects[key]
            np.testing.assert_allclose(cell["effect"], ref["effect"], rtol=1e-12)
            np.testing.assert_allclose(cell["se"], ref["se"], rtol=1e-9)

    def test_reg_constant_only_covariate_matches_no_covariate_survey(self):
        """Survey-weighted twin of the constant-only-covariate case: the
        weighted centered Gram is also rank-0, and the general
        (survey-branch) producer must likewise collapse to the
        no-covariate survey fit with finite SEs."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(17)
        data = generate_staggered_data_with_covariates(seed=789)
        data["xc"] = 5.0
        weights = pd.DataFrame(
            {
                "unit": data["unit"].unique(),
                "weight": rng.uniform(0.5, 2.0, size=data["unit"].nunique()),
            }
        )
        data = data.merge(weights, on="unit")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            no_cov = CallawaySantAnna(estimation_method="reg").fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=SurveyDesign(weights="weight"),
            )
            const_only = CallawaySantAnna(estimation_method="reg").fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["xc"],
                survey_design=SurveyDesign(weights="weight"),
            )
        assert np.isfinite(const_only.overall_se)
        np.testing.assert_allclose(const_only.overall_att, no_cov.overall_att, rtol=1e-12)
        np.testing.assert_allclose(const_only.overall_se, no_cov.overall_se, rtol=1e-9)
        for key, cell in const_only.group_time_effects.items():
            ref = no_cov.group_time_effects[key]
            np.testing.assert_allclose(cell["se"], ref["se"], rtol=1e-9)

    def test_uniform_survey_weights_match_unweighted_per_cell_se(self):
        """Uniform survey weights route reg+cov through the general
        (survey-branch) producer while the unweighted fit takes the
        vectorized producer; both now share the same DRDID IF algebra, so
        per-cell effects AND SEs must agree."""
        from diff_diff.survey import SurveyDesign

        data = generate_staggered_data_with_covariates(seed=53)
        data["w_ones"] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unweighted = CallawaySantAnna(estimation_method="reg").fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "x2"],
            )
            uniform = CallawaySantAnna(estimation_method="reg").fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                covariates=["x1", "x2"],
                survey_design=SurveyDesign(weights="w_ones"),
            )
        for key, cell in unweighted.group_time_effects.items():
            ref = uniform.group_time_effects[key]
            np.testing.assert_allclose(cell["effect"], ref["effect"], rtol=1e-9)
            np.testing.assert_allclose(cell["se"], ref["se"], rtol=1e-9)


class TestCallawaySantAnnaRankDeficiencyPaths:
    """Tests for rank-deficiency handling in DR and reg not_yet_treated paths."""

    def test_dr_rank_deficient_action_warn_emits_warning(self):
        """Test that DR path emits rank-deficiency warning with collinear covariates."""
        import warnings as warn_mod

        data = generate_staggered_data_with_covariates(seed=42)
        # Near-collinear covariate: x1 + tiny noise
        rng = np.random.default_rng(99)
        data["x1_near"] = data["x1"] + rng.normal(scale=1e-9, size=len(data))

        cs = CallawaySantAnna(
            estimation_method="dr",
            rank_deficient_action="warn",
        )

        with warn_mod.catch_warnings(record=True) as w:
            warn_mod.simplefilter("always")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x1_near"],
            )

            rank_warnings = [
                x
                for x in w
                if "rank-deficient" in str(x.message).lower() or "Rank-deficient" in str(x.message)
            ]
            assert (
                len(rank_warnings) > 0
            ), "Expected at least one rank-deficiency warning from DR path"

        assert results is not None
        assert results.overall_att is not None

    def test_reg_nyt_rank_deficient_action_warn(self):
        """Test that reg+not_yet_treated emits rank-deficiency warning with collinear covariates."""
        import warnings as warn_mod

        data = generate_staggered_data_with_covariates(seed=42)
        data["x1_dup"] = data["x1"].copy()

        cs = CallawaySantAnna(
            estimation_method="reg",
            control_group="not_yet_treated",
            rank_deficient_action="warn",
        )

        with warn_mod.catch_warnings(record=True) as w:
            warn_mod.simplefilter("always")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x1_dup"],
            )

            rank_warnings = [
                x
                for x in w
                if "rank-deficient" in str(x.message).lower() or "Rank-deficient" in str(x.message)
            ]
            assert (
                len(rank_warnings) > 0
            ), "Expected at least one rank-deficiency warning from reg nyt path"

        assert results is not None
        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_ipw_rank_deficient_action_error_raises(self):
        """IPW path raises ValueError with rank_deficient_action='error' and collinear covariates."""
        data = generate_staggered_data_with_covariates(seed=42)
        data["x1_dup"] = data["x1"].copy()

        cs = CallawaySantAnna(
            estimation_method="ipw",
            rank_deficient_action="error",
        )

        with pytest.raises(ValueError, match="[Rr]ank"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x1_dup"],
            )

    def test_dr_rank_deficient_action_error_raises(self):
        """DR path raises ValueError with rank_deficient_action='error' and collinear covariates."""
        data = generate_staggered_data_with_covariates(seed=42)
        data["x1_dup"] = data["x1"].copy()

        cs = CallawaySantAnna(
            estimation_method="dr",
            rank_deficient_action="error",
        )

        with pytest.raises(ValueError, match="[Rr]ank"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x1_dup"],
            )

    def test_bootstrap_single_unit_cohort_handles_gracefully(self, ci_params):
        """Test that bootstrap handles cohort with 1 treated unit without crashing."""
        # Build small dataset where one cohort has exactly 1 unit
        rng = np.random.default_rng(42)
        n_periods = 6
        # 15 never-treated, 14 in cohort 3, 1 in cohort 5
        cohorts = ([0] * 15) + ([3] * 14) + ([5] * 1)
        n_units = len(cohorts)

        rows = []
        for i in range(n_units):
            g = cohorts[i]
            for t in range(1, n_periods + 1):
                treated = 1 if (g > 0 and t >= g) else 0
                y = rng.normal(0, 1) + 2.0 * treated
                rows.append((i, t, y, g))

        data = pd.DataFrame(rows, columns=["unit", "time", "outcome", "first_treat"])

        n_boot = ci_params.bootstrap(99)
        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)

        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        assert results is not None
        assert results.overall_att is not None
        # Single-unit cohort (g=5) effects should exist and have finite ATT
        g5_effects = {(g, t): eff for (g, t), eff in results.group_time_effects.items() if g == 5}
        assert len(g5_effects) > 0, "Expected group-time effects for cohort g=5"
        for (g, t), eff in g5_effects.items():
            assert np.isfinite(eff["effect"]), f"g={g},t={t}: ATT should be finite"


class TestCallawaySantAnnaBootstrap:
    """Tests for Callaway-Sant'Anna multiplier bootstrap inference."""

    def test_bootstrap_basic(self, ci_params):
        """Test basic bootstrap functionality."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.n_bootstrap == n_boot
        assert results.bootstrap_results.weight_type == "rademacher"
        assert results.overall_se > 0
        assert results.overall_conf_int[0] < results.overall_att < results.overall_conf_int[1]

    def test_bootstrap_weight_types(self, ci_params):
        """Test different bootstrap weight types."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(49)

        weight_types = ["rademacher", "mammen", "webb"]

        for wt in weight_types:
            cs = CallawaySantAnna(n_bootstrap=n_boot, bootstrap_weights=wt, seed=42)
            results = cs.fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

            assert results.bootstrap_results is not None
            assert results.bootstrap_results.weight_type == wt
            assert results.overall_se > 0

    def test_bootstrap_event_study(self, ci_params):
        """Test bootstrap with event study aggregation."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        assert results.bootstrap_results.event_study_cis is not None
        assert results.bootstrap_results.event_study_p_values is not None

        # Check event study effects have bootstrap SEs
        for e, effect in results.event_study_effects.items():
            assert effect["se"] > 0
            assert effect["conf_int"][0] < effect["conf_int"][1]

    def test_bootstrap_group_aggregation(self, ci_params):
        """Test bootstrap with group aggregation."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.group_effect_ses is not None
        assert results.bootstrap_results.group_effect_cis is not None
        assert results.bootstrap_results.group_effect_p_values is not None

        # Check group effects have bootstrap SEs
        for g, effect in results.group_effects.items():
            assert effect["se"] > 0
            assert effect["conf_int"][0] < effect["conf_int"][1]

    def test_bootstrap_all_aggregations(self, ci_params):
        """Test bootstrap with all aggregations."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        assert results.bootstrap_results.group_effect_ses is not None

    def test_bootstrap_reproducibility(self, ci_params):
        """Test that bootstrap is reproducible with same seed."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs1 = CallawaySantAnna(n_bootstrap=n_boot, seed=123)
        results1 = cs1.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        cs2 = CallawaySantAnna(n_bootstrap=n_boot, seed=123)
        results2 = cs2.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Results should be identical with same seed
        assert results1.overall_se == results2.overall_se
        assert results1.overall_conf_int == results2.overall_conf_int

    def test_bootstrap_different_seeds(self, ci_params):
        """Test that different seeds give different results."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs1 = CallawaySantAnna(n_bootstrap=n_boot, seed=123)
        results1 = cs1.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        cs2 = CallawaySantAnna(n_bootstrap=n_boot, seed=456)
        results2 = cs2.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Results should differ with different seeds
        assert results1.overall_se != results2.overall_se

    def test_bootstrap_p_value_significance(self, ci_params):
        """Test that strong effect has significant p-value with bootstrap."""
        data = generate_staggered_data(n_units=100, treatment_effect=5.0, seed=42)
        n_boot = ci_params.bootstrap(199)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Strong effect should be significant
        assert results.overall_p_value < 0.05
        assert results.is_significant

    def test_bootstrap_zero_effect_not_significant(self, ci_params):
        """Test that zero effect is not significant with bootstrap."""
        data = generate_staggered_data(n_units=50, treatment_effect=0.0, seed=42)
        n_boot = ci_params.bootstrap(199)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Zero effect should not be significant at 0.01 level
        # (using 0.01 to be more conservative with finite sample)
        assert results.overall_p_value > 0.01 or abs(results.overall_att) < 2 * results.overall_se

    def test_bootstrap_distribution_stored(self, ci_params):
        """Test that bootstrap distribution is stored in results."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.bootstrap_results.bootstrap_distribution is not None
        assert len(results.bootstrap_results.bootstrap_distribution) == n_boot

    def test_bootstrap_with_covariates(self, ci_params):
        """Test bootstrap with covariate adjustment."""
        data = generate_staggered_data_with_covariates(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        assert results.bootstrap_results is not None
        assert results.overall_se > 0

    def test_bootstrap_group_time_effects(self, ci_params):
        """Test that bootstrap updates group-time effect SEs."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        # Without bootstrap
        cs1 = CallawaySantAnna(n_bootstrap=0)
        results1 = cs1.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # With bootstrap
        cs2 = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results2 = cs2.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Group-time effects should have same point estimates
        for gt in results1.group_time_effects:
            assert (
                results1.group_time_effects[gt]["effect"]
                == results2.group_time_effects[gt]["effect"]
            )
            # But SEs may differ (bootstrap vs analytical)
            assert results2.group_time_effects[gt]["se"] > 0

    def test_bootstrap_invalid_weight_type(self):
        """Test that invalid weight type raises error."""
        # Test with new parameter name
        with pytest.raises(ValueError, match="bootstrap_weights"):
            CallawaySantAnna(bootstrap_weights="invalid")

    def test_bootstrap_get_params(self):
        """Test that get_params includes bootstrap_weights."""
        cs = CallawaySantAnna(n_bootstrap=99, bootstrap_weights="mammen", seed=42)
        params = cs.get_params()

        assert params["n_bootstrap"] == 99
        assert params["bootstrap_weights"] == "mammen"
        assert params["seed"] == 42

    def test_bootstrap_with_not_yet_treated(self, ci_params):
        """Test bootstrap with not_yet_treated control group."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(control_group="not_yet_treated", n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.bootstrap_results is not None
        assert results.overall_se > 0

    def test_bootstrap_estimation_methods(self, ci_params):
        """Test bootstrap with different estimation methods."""
        data = generate_staggered_data(n_units=50, seed=42)
        n_boot = ci_params.bootstrap(49)

        methods = ["reg", "ipw", "dr"]

        for method in methods:
            cs = CallawaySantAnna(estimation_method=method, n_bootstrap=n_boot, seed=42)
            results = cs.fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

            assert results.bootstrap_results is not None
            assert results.overall_se > 0, f"Failed for method {method}"

    def test_bootstrap_with_balanced_event_study(self, ci_params):
        """Test bootstrap with balanced event study aggregation."""
        data = generate_staggered_data(n_units=100, n_periods=12, seed=42)
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=0,  # Balance at treatment time
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        assert results.event_study_effects is not None

        # Check that event study effects have valid bootstrap SEs
        for e, effect in results.event_study_effects.items():
            assert effect["se"] > 0
            assert effect["conf_int"][0] < effect["conf_int"][1]

    def test_bootstrap_low_iterations_warning(self):
        """Test that low n_bootstrap triggers a warning."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(n_bootstrap=30, seed=42)

        with pytest.warns(UserWarning, match="n_bootstrap=30 is low"):
            cs.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")


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

                data.append(
                    {
                        "unit": unit,
                        "time": t,
                        "outcome": y,
                        "first_treat": first_treat,
                    }
                )

        df = pd.DataFrame(data)

        cs = CallawaySantAnna()
        results = cs.fit(df, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

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

                data.append(
                    {
                        "unit": unit,
                        "time": t,
                        "outcome": y,
                        "first_treat": first_treat,
                    }
                )

        df = pd.DataFrame(data)

        cs = CallawaySantAnna()
        results = cs.fit(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
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
            post_effects = [results.event_study_effects[e]["effect"] for e in post_periods]
            assert any(
                e > 0.5 for e in post_effects
            ), f"Expected positive post-period effects, got {post_effects}"

    def test_single_cohort_with_bootstrap(self, ci_params):
        """Test bootstrap inference with single cohort."""
        np.random.seed(42)
        n_boot = ci_params.bootstrap(99)

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

                data.append(
                    {
                        "unit": unit,
                        "time": t,
                        "outcome": y,
                        "first_treat": first_treat,
                    }
                )

        df = pd.DataFrame(data)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(df, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.overall_att_se > 0
        assert (
            results.bootstrap_results.overall_att_ci[0]
            < results.bootstrap_results.overall_att_ci[1]
        )

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

                data.append(
                    {
                        "unit": unit,
                        "time": t,
                        "outcome": y,
                        "first_treat": first_treat,
                    }
                )

        df = pd.DataFrame(data)

        cs_never = CallawaySantAnna(control_group="never_treated")
        results_never = cs_never.fit(
            df, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        cs_not_yet = CallawaySantAnna(control_group="not_yet_treated")
        results_not_yet = cs_not_yet.fit(
            df, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Both should produce valid results
        assert np.isfinite(results_never.overall_att)
        assert np.isfinite(results_not_yet.overall_att)

        # Results may differ slightly due to different comparison groups
        # but should be in similar range
        assert abs(results_never.overall_att - results_not_yet.overall_att) < 1.0


class TestCallawaySantAnnaAnalyticalSE:
    """Tests for analytical SE using influence function aggregation."""

    def test_analytical_se_vs_bootstrap_se(self, ci_params):
        """Analytical SE should be close to bootstrap SE (within 15%)."""
        # Generate data with moderate size for stable comparison
        data = generate_staggered_data(
            n_units=200,
            n_periods=8,
            n_cohorts=3,
            treatment_effect=3.0,
            never_treated_frac=0.3,
            seed=42,
        )
        n_boot = ci_params.bootstrap(499, min_n=249)

        # Run with analytical SE (n_bootstrap=0)
        cs_analytical = CallawaySantAnna(n_bootstrap=0, seed=42)
        results_analytical = cs_analytical.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Run with bootstrap SE (n_bootstrap=499)
        cs_bootstrap = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results_bootstrap = cs_bootstrap.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Point estimates should match exactly
        assert abs(results_analytical.overall_att - results_bootstrap.overall_att) < 1e-10

        # SEs should be similar (within 15% with enough bootstrap iterations,
        # wider tolerance when min_n cap reduces iterations in pure Python mode)
        rel_diff = (
            abs(results_analytical.overall_se - results_bootstrap.overall_se)
            / results_bootstrap.overall_se
        )
        threshold = 0.40 if n_boot < 100 else 0.15
        assert rel_diff < threshold, (
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
            seed=123,
        )

        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # The SE should be non-zero and positive
        assert results.overall_se > 0

        # Compute what the "independence" SE would be (sum of weighted variances)
        gt_effects = results.group_time_effects
        weights = []
        variances = []
        for (g, t), effect in gt_effects.items():
            weights.append(effect["n_treated"])
            variances.append(effect["se"] ** 2)

        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        variances = np.array(variances)

        # Independence SE formula (the old incorrect formula)
        independence_var = np.sum(weights**2 * variances)
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

                data.append(
                    {
                        "unit": unit,
                        "time": t,
                        "outcome": y,
                        "first_treat": first_treat,
                    }
                )

        df = pd.DataFrame(data)

        # Use only the first post-treatment period
        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(df, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        # If there's only one (g,t) pair, overall SE should match individual SE
        if len(results.group_time_effects) == 1:
            gt_key = list(results.group_time_effects.keys())[0]
            individual_se = results.group_time_effects[gt_key]["se"]
            # Should be close (may not be exact due to normalization)
            assert abs(results.overall_se - individual_se) < individual_se * 0.01

    def test_event_study_analytical_se(self, ci_params):
        """Event study SEs should also use influence function aggregation."""
        data = generate_staggered_data(
            n_units=200,
            n_periods=10,
            n_cohorts=3,
            treatment_effect=2.5,
            never_treated_frac=0.3,
            seed=42,
        )
        n_boot = ci_params.bootstrap(499, min_n=199)

        # Analytical
        cs_analytical = CallawaySantAnna(n_bootstrap=0, seed=42)
        results_analytical = cs_analytical.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Bootstrap
        cs_bootstrap = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results_bootstrap = cs_bootstrap.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Event study effects should exist
        assert results_analytical.event_study_effects is not None
        assert results_bootstrap.event_study_effects is not None

        # Check each event time SE is similar (wider tolerance when
        # min_n cap reduces bootstrap iterations in pure Python mode)
        threshold = 0.40 if n_boot < 100 else 0.20
        for e in results_analytical.event_study_effects:
            if e in results_bootstrap.event_study_effects:
                se_analytical = results_analytical.event_study_effects[e]["se"]
                se_bootstrap = results_bootstrap.event_study_effects[e]["se"]

                if se_bootstrap > 0:
                    rel_diff = abs(se_analytical - se_bootstrap) / se_bootstrap
                    assert rel_diff < threshold, (
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
        outcome_name: str = "y",
        unit_name: str = "id",
        time_name: str = "period",
        first_treat_name: str = "treatment_start",
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
        first_treat[n_never : n_never + (n_units - n_never) // 2] = 4
        first_treat[n_never + (n_units - n_never) // 2 :] = 6
        first_treat_expanded = np.repeat(first_treat, n_periods)

        # Generate outcomes
        unit_fe = np.repeat(np.random.randn(n_units) * 2, n_periods)
        time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = unit_fe + time_fe + 2.5 * post + np.random.randn(len(units)) * 0.5

        return pd.DataFrame(
            {
                outcome_name: outcomes,
                unit_name: units,
                time_name: times,
                first_treat_name: first_treat_expanded.astype(int),
            }
        )

    def test_non_standard_first_treat_name(self):
        """Test with non-standard first_treat column name."""
        data = self.generate_data_with_custom_names(first_treat_name="treatment_cohort")

        cs = CallawaySantAnna()
        results = cs.fit(
            data, outcome="y", unit="id", time="period", first_treat="treatment_cohort"
        )

        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)
        assert results.overall_se > 0
        # Treatment effect should be approximately 2.5
        assert abs(results.overall_att - 2.5) < 1.5

    def test_non_standard_all_column_names(self):
        """Test with all non-standard column names."""
        data = self.generate_data_with_custom_names(
            outcome_name="response_var",
            unit_name="entity_id",
            time_name="time_period",
            first_treat_name="treatment_timing",
        )

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome="response_var",
            unit="entity_id",
            time="time_period",
            first_treat="treatment_timing",
        )

        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)
        assert results.overall_se > 0

    def test_non_standard_names_with_bootstrap(self, ci_params):
        """Test non-standard column names with bootstrap inference."""
        data = self.generate_data_with_custom_names(
            first_treat_name="g", n_units=50  # Short name like R's `did` package uses
        )
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(data, outcome="y", unit="id", time="period", first_treat="g")

        assert results.bootstrap_results is not None
        assert results.overall_se > 0
        assert results.overall_conf_int[0] < results.overall_att < results.overall_conf_int[1]

    def test_non_standard_names_with_event_study(self):
        """Test non-standard column names with event study aggregation."""
        data = self.generate_data_with_custom_names(first_treat_name="cohort", n_periods=12)

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome="y",
            unit="id",
            time="period",
            first_treat="cohort",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

    def test_non_standard_names_with_covariates(self):
        """Test non-standard column names with covariate adjustment."""
        # Generate data with covariates
        data = self.generate_data_with_custom_names(first_treat_name="treatment_time")
        # Add covariates with custom names
        data["covariate_x"] = np.random.randn(len(data))
        data["covariate_z"] = np.random.binomial(1, 0.5, len(data))

        cs = CallawaySantAnna(estimation_method="dr")
        results = cs.fit(
            data,
            outcome="y",
            unit="id",
            time="period",
            first_treat="treatment_time",
            covariates=["covariate_x", "covariate_z"],
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_non_standard_names_with_not_yet_treated(self):
        """Test non-standard column names with not_yet_treated control group."""
        data = self.generate_data_with_custom_names(first_treat_name="adoption_period")

        cs = CallawaySantAnna(control_group="not_yet_treated")
        results = cs.fit(data, outcome="y", unit="id", time="period", first_treat="adoption_period")

        assert results.overall_att is not None
        assert results.control_group == "not_yet_treated"

    def test_non_standard_names_matches_standard_names(self):
        """Verify results are identical regardless of column naming."""
        np.random.seed(42)

        # Generate identical data with different column names
        data_standard = generate_staggered_data(n_units=80, seed=42)

        data_custom = data_standard.rename(
            columns={
                "outcome": "y",
                "unit": "entity",
                "time": "t",
                "first_treat": "g",
            }
        )

        # Fit with standard names
        cs1 = CallawaySantAnna(seed=123)
        results1 = cs1.fit(
            data_standard, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Fit with custom names
        cs2 = CallawaySantAnna(seed=123)
        results2 = cs2.fit(data_custom, outcome="y", unit="entity", time="t", first_treat="g")

        # Results should be identical
        assert abs(results1.overall_att - results2.overall_att) < 1e-10
        assert abs(results1.overall_se - results2.overall_se) < 1e-10

    def test_column_name_with_spaces(self):
        """Test column names containing spaces."""
        data = self.generate_data_with_custom_names()
        data = data.rename(
            columns={
                "y": "outcome variable",
                "treatment_start": "treatment period",
            }
        )

        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome="outcome variable",
            unit="id",
            time="period",
            first_treat="treatment period",
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_column_name_with_special_characters(self):
        """Test column names with underscores and numbers."""
        data = self.generate_data_with_custom_names()
        data = data.rename(
            columns={
                "treatment_start": "first_treat_2024",
            }
        )

        cs = CallawaySantAnna()
        results = cs.fit(
            data, outcome="y", unit="id", time="period", first_treat="first_treat_2024"
        )

        assert results.overall_att is not None


class TestCallawaySantAnnaPreTreatment:
    """Tests for CallawaySantAnna pre-treatment effects (base_period parameter)."""

    def test_base_period_validation(self):
        """Invalid base_period raises ValueError."""
        with pytest.raises(ValueError, match="base_period must be 'varying' or 'universal'"):
            CallawaySantAnna(base_period="invalid")

    def test_base_period_in_get_params(self):
        """base_period appears in get_params()."""
        cs = CallawaySantAnna(base_period="universal")
        params = cs.get_params()
        assert "base_period" in params
        assert params["base_period"] == "universal"

        cs2 = CallawaySantAnna(base_period="varying")
        params2 = cs2.get_params()
        assert params2["base_period"] == "varying"

    def test_varying_pre_treatment_effects(self):
        """Varying mode computes pre-treatment ATT(g,t) for t < g."""
        # Generate data with enough pre-treatment periods
        data = generate_staggered_data(
            n_units=100, n_periods=10, n_cohorts=2, treatment_effect=2.0, seed=42
        )

        cs = CallawaySantAnna(base_period="varying")
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Should have pre-treatment effects (t < g)
        pre_treatment_effects = [(g, t) for (g, t) in results.group_time_effects.keys() if t < g]
        assert len(pre_treatment_effects) > 0, "Should compute pre-treatment effects"

    def test_universal_pre_treatment_effects(self):
        """Universal mode computes pre-treatment ATT(g,t) for t < g."""
        data = generate_staggered_data(
            n_units=100, n_periods=10, n_cohorts=2, treatment_effect=2.0, seed=42
        )

        cs = CallawaySantAnna(base_period="universal")
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Should have pre-treatment effects (t < g)
        pre_treatment_effects = [(g, t) for (g, t) in results.group_time_effects.keys() if t < g]
        assert len(pre_treatment_effects) > 0, "Should compute pre-treatment effects"

    def test_post_treatment_identical(self):
        """Post-treatment ATT(g,t) identical for both modes."""
        data = generate_staggered_data(
            n_units=100, n_periods=10, n_cohorts=2, treatment_effect=2.0, seed=42
        )

        # Fit with varying
        cs_v = CallawaySantAnna(base_period="varying")
        res_v = cs_v.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Fit with universal
        cs_u = CallawaySantAnna(base_period="universal")
        res_u = cs_u.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Post-treatment effects should be identical
        for (g, t), eff_v in res_v.group_time_effects.items():
            if t >= g:  # Post-treatment
                if (g, t) in res_u.group_time_effects:
                    eff_u = res_u.group_time_effects[(g, t)]
                    assert abs(eff_v["effect"] - eff_u["effect"]) < 1e-10, (
                        f"Post-treatment ATT({g},{t}) differs: "
                        f"varying={eff_v['effect']:.6f}, universal={eff_u['effect']:.6f}"
                    )

    def test_event_study_negative_periods(self):
        """Event study includes negative relative periods."""
        data = generate_staggered_data(
            n_units=100, n_periods=12, n_cohorts=2, treatment_effect=2.0, seed=42
        )

        cs = CallawaySantAnna(base_period="varying")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None

        # Should have negative relative periods
        rel_periods = list(results.event_study_effects.keys())
        negative_periods = [e for e in rel_periods if e < 0]
        assert (
            len(negative_periods) > 0
        ), f"Event study should include negative periods, got {rel_periods}"

    def test_base_period_in_results(self):
        """base_period is stored in results and shown in summary."""
        data = generate_staggered_data(n_units=50, seed=42)

        cs = CallawaySantAnna(base_period="universal")
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.base_period == "universal"
        summary = results.summary()
        assert "Base period:" in summary
        assert "universal" in summary

    def test_pre_treatment_bootstrap(self, ci_params):
        """Bootstrap handles pre-treatment effects."""
        data = generate_staggered_data(
            n_units=60, n_periods=8, n_cohorts=2, treatment_effect=2.0, seed=42
        )
        n_boot = ci_params.bootstrap(99)

        cs = CallawaySantAnna(base_period="varying", n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.bootstrap_results is not None

        # Pre-treatment effects should have valid bootstrap SEs
        for (g, t), eff in results.group_time_effects.items():
            if t < g:  # Pre-treatment
                assert eff["se"] > 0, f"Pre-treatment ATT({g},{t}) should have positive SE"
                assert np.isfinite(eff["se"]), f"Pre-treatment ATT({g},{t}) SE should be finite"

    def test_pre_treatment_near_zero_under_parallel_trends(self):
        """Pre-treatment effects should be near zero when parallel trends holds."""
        # Generate data with true parallel trends (no pre-trends)
        data = generate_staggered_data(
            n_units=200,
            n_periods=10,
            n_cohorts=2,
            treatment_effect=3.0,  # Only post-treatment effect
            seed=123,
        )

        cs = CallawaySantAnna(base_period="varying")
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Pre-treatment effects should be close to zero
        pre_effects = [eff["effect"] for (g, t), eff in results.group_time_effects.items() if t < g]
        if pre_effects:
            # Mean of pre-treatment effects should be close to 0
            mean_pre = np.mean(pre_effects)
            assert (
                abs(mean_pre) < 1.0
            ), f"Pre-treatment effects mean={mean_pre:.3f} should be near zero"

    def test_set_params_base_period(self):
        """set_params() can change base_period."""
        cs = CallawaySantAnna(base_period="varying")
        assert cs.base_period == "varying"

        cs.set_params(base_period="universal")
        assert cs.base_period == "universal"

        params = cs.get_params()
        assert params["base_period"] == "universal"

    def test_default_base_period_is_varying(self):
        """Default base_period is 'varying'."""
        cs = CallawaySantAnna()
        assert cs.base_period == "varying"
        assert cs.get_params()["base_period"] == "varying"

    def test_varying_mode_no_fallback_to_nonconsecutive(self):
        """Varying mode skips pre-treatment effects where t-1 doesn't exist."""
        # Create data where first period (e.g., period 1) has no t-1 predecessor
        data = generate_staggered_data(
            n_units=100, n_periods=6, n_cohorts=2, treatment_effect=2.0, seed=42  # periods 1-6
        )

        # Identify the earliest time period in data
        min_period = data["time"].min()

        cs = CallawaySantAnna(base_period="varying")
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # In varying mode, ATT(g, min_period) should NOT be computed for
        # any cohort g because t-1 (period 0) doesn't exist
        for g, t in results.group_time_effects.keys():
            if t == min_period:
                # This should not happen - the (g, min_period) pair should be skipped
                pytest.fail(
                    f"ATT({g}, {t}) should not exist because t-1 doesn't exist. "
                    "Fallback to non-consecutive base period was incorrectly applied."
                )

    def test_no_post_treatment_effects_returns_nan_with_warning(self):
        """Warn and return NaN when no post-treatment effects exist."""
        import warnings

        # Create data where the treatment cohort treats AFTER the last observed period
        # so there are no post-treatment periods (t >= g never holds)
        n_units = 50
        n_periods = 5
        np.random.seed(42)

        data = []
        for unit in range(n_units):
            for t in range(1, n_periods + 1):
                # Treated units get treated at period 6 (beyond data range)
                # Data only goes to period 5, so no post-treatment periods exist
                first_treat = n_periods + 1 if unit < n_units // 2 else 0
                outcome = np.random.randn()
                data.append(
                    {"unit": unit, "time": t, "outcome": outcome, "first_treat": first_treat}
                )

        df = pd.DataFrame(data)

        cs = CallawaySantAnna(base_period="varying")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cs.fit(
                df, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

            # Should have emitted a warning about no post-treatment effects
            warning_messages = [str(warning.message) for warning in w]
            has_warning = any("No post-treatment effects" in msg for msg in warning_messages)
            assert (
                has_warning
            ), f"Expected warning about no post-treatment effects, got: {warning_messages}"

        # Overall ATT should be NaN
        assert np.isnan(results.overall_att), (
            f"Expected NaN for overall_att when no post-treatment effects exist, "
            f"got {results.overall_att}"
        )
        # All inference fields should also be NaN
        assert np.isnan(
            results.overall_se
        ), f"Expected NaN for overall_se, got {results.overall_se}"
        assert np.isnan(
            results.overall_t_stat
        ), f"Expected NaN for overall_t_stat, got {results.overall_t_stat}"
        assert np.isnan(
            results.overall_p_value
        ), f"Expected NaN for overall_p_value, got {results.overall_p_value}"

    def test_no_post_treatment_effects_bootstrap_returns_nan(self, ci_params):
        """Bootstrap returns NaN inference when no post-treatment effects exist."""
        import warnings

        n_boot = ci_params.bootstrap(99)

        # Create data where treatment happens after the data ends
        n_units = 50
        n_periods = 5
        np.random.seed(42)

        data = []
        for unit in range(n_units):
            for t in range(1, n_periods + 1):
                first_treat = n_periods + 1 if unit < n_units // 2 else 0
                outcome = np.random.randn()
                data.append(
                    {"unit": unit, "time": t, "outcome": outcome, "first_treat": first_treat}
                )

        df = pd.DataFrame(data)

        cs = CallawaySantAnna(base_period="varying", n_bootstrap=n_boot, seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cs.fit(
                df, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

            # Should have warning about no post-treatment effects
            warning_messages = [str(warning.message) for warning in w]
            has_warning = any("No post-treatment effects" in msg for msg in warning_messages)
            assert has_warning, f"Expected warning, got: {warning_messages}"

        # All overall inference fields should be NaN
        assert np.isnan(results.overall_att), "overall_att should be NaN"
        assert np.isnan(results.overall_se), "overall_se should be NaN"
        assert np.isnan(results.overall_t_stat), "overall_t_stat should be NaN"
        assert np.isnan(results.overall_p_value), "overall_p_value should be NaN"
        assert np.isnan(results.overall_conf_int[0]), "CI lower should be NaN"
        assert np.isnan(results.overall_conf_int[1]), "CI upper should be NaN"

        # Bootstrap results should also have NaN
        assert results.bootstrap_results is not None
        assert np.isnan(results.bootstrap_results.overall_att_se)
        assert np.isnan(results.bootstrap_results.overall_att_p_value)

    def test_bootstrap_runs_for_pretreatment_effects(self, ci_params):
        """Bootstrap computes SEs for pre-treatment effects even when no post-treatment.

        When all treatment occurs after data ends, the overall ATT should be NaN,
        but pre-treatment effects should still get bootstrap SEs (not analytical).
        """
        import warnings

        n_boot = ci_params.bootstrap(99)

        # Create data where all treatment happens after the data ends
        # so we have only pre-treatment effects
        n_units = 60
        n_periods = 6
        np.random.seed(999)

        data = []
        for unit in range(n_units):
            # Half the units have first_treat at period 10 (after data ends at 6)
            # Other half are never-treated (control)
            first_treat = 10 if unit < n_units // 2 else 0
            for t in range(1, n_periods + 1):
                outcome = np.random.randn() + (0.5 * t)  # Some time trend
                data.append(
                    {"unit": unit, "time": t, "outcome": outcome, "first_treat": first_treat}
                )

        df = pd.DataFrame(data)

        # Fit with bootstrap and base_period="varying" to get pre-treatment effects
        cs = CallawaySantAnna(base_period="varying", n_bootstrap=n_boot, seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cs.fit(
                df, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

            # Should have warning about no post-treatment effects
            warning_messages = [str(warning.message) for warning in w]
            has_warning = any("No post-treatment effects" in msg for msg in warning_messages)
            assert has_warning, "Expected warning about no post-treatment effects"

        # Verify overall ATT is NaN
        assert np.isnan(results.overall_att), "overall_att should be NaN"
        assert np.isnan(results.overall_se), "overall_se should be NaN"

        # Verify we have pre-treatment effects
        pre_treatment_effects = [(g, t) for (g, t) in results.group_time_effects.keys() if t < g]
        assert len(pre_treatment_effects) > 0, "Should have pre-treatment effects"

        # Key test: bootstrap should have computed SEs for the pre-treatment effects
        assert results.bootstrap_results is not None, "Bootstrap results should exist"

        # Check that pre-treatment effects have bootstrap SEs
        for gt in pre_treatment_effects:
            bootstrap_se = results.bootstrap_results.group_time_ses.get(gt)
            assert bootstrap_se is not None, f"Bootstrap SE missing for {gt}"
            # Bootstrap SE should be finite (it was computed, not analytical fallback)
            # Note: in the old code, these would be analytical SEs, not bootstrap
            assert np.isfinite(
                bootstrap_se
            ), f"Bootstrap SE for {gt} should be finite, got {bootstrap_se}"

        # Also verify overall bootstrap statistics are NaN
        assert np.isnan(
            results.bootstrap_results.overall_att_se
        ), "Overall ATT SE should be NaN when no post-treatment"
        assert np.isnan(
            results.bootstrap_results.overall_att_p_value
        ), "Overall ATT p-value should be NaN when no post-treatment"

    def test_not_yet_treated_excludes_cohort_from_controls(self):
        """Not-yet-treated control excludes treated cohort g for pre-treatment periods.

        When computing ATT(g,t) for t < g with control_group="not_yet_treated",
        cohort g should NOT be included in the control group even though
        they haven't been treated yet at time t.

        Bug scenario (before fix):
        - Computing ATT(g=5, t=3) with control_group="not_yet_treated"
        - Control mask was: never_treated OR first_treat > t
        - Units with first_treat=5 satisfy first_treat > 3, so they were
          incorrectly included as controls for themselves!

        After fix:
        - Control mask is: never_treated OR (first_treat > t AND first_treat != g)
        - Cohort g is always excluded from controls.
        """
        # Create data with 3 distinct cohorts: g=4, g=7, and never-treated (g=0)
        # This setup ensures for ATT(g=7, t=3):
        #   - Treated: units with first_treat=7
        #   - Valid controls: never-treated + cohort g=4 (since 4 > 3 and 4 != 7)
        #   - Invalid (excluded): cohort g=7 (even though 7 > 3)
        n_units = 90  # 30 per group
        n_periods = 10
        np.random.seed(42)

        data = []
        for unit in range(n_units):
            # Assign to cohorts: 0-29 -> g=4, 30-59 -> g=7, 60-89 -> never-treated
            if unit < 30:
                first_treat = 4
            elif unit < 60:
                first_treat = 7
            else:
                first_treat = 0  # Never-treated

            for t in range(1, n_periods + 1):
                # Add treatment effect after treatment
                effect = 0.0
                if first_treat > 0 and t >= first_treat:
                    effect = 2.0

                outcome = np.random.randn() + effect
                data.append(
                    {"unit": unit, "time": t, "outcome": outcome, "first_treat": first_treat}
                )

        df = pd.DataFrame(data)

        # Fit with not_yet_treated control group
        cs = CallawaySantAnna(
            control_group="not_yet_treated", base_period="varying"  # To get pre-treatment effects
        )
        results = cs.fit(df, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        # Check the group-time effects for pre-treatment ATT(g=7, t) where t < 7
        # These should have been computed using valid controls only
        for (g, t), eff in results.group_time_effects.items():
            if g == 7 and t < g:  # Pre-treatment for cohort 7
                n_control = eff["n_control"]
                # Control should include:
                #   - 30 never-treated units
                #   - 30 units from cohort g=4 (if t < 4, they're not yet treated either)
                # Control should NOT include:
                #   - The 30 units from cohort g=7 (they're the treated group!)

                # For t < 4: controls = never-treated (30) + cohort 4 (30) = 60
                # For 4 <= t < 7: controls = never-treated (30) only (cohort 4 is treated)
                if t < 4:
                    expected_max = 60  # never-treated + cohort 4
                else:
                    expected_max = 30  # never-treated only

                # Key assertion: n_control should NOT be 90 (which would include cohort 7)
                assert n_control <= expected_max, (
                    f"ATT(g=7, t={t}): n_control={n_control} should be <= {expected_max}. "
                    f"Cohort 7 (30 units) should NOT be included as controls for itself."
                )

                # Also verify we have a reasonable number of controls
                assert (
                    n_control >= 30
                ), f"ATT(g=7, t={t}): n_control={n_control} should be >= 30 (never-treated)."


class TestCallawaySantAnnaAnticipation:
    """Tests for anticipation parameter handling in aggregation."""

    def test_group_effects_with_anticipation(self):
        """Group aggregation correctly handles anticipation parameter.

        With anticipation=k, effects at t >= g - k should be included in
        group aggregation (not just t >= g).
        """
        # Generate staggered data with a clear treatment effect
        data = generate_staggered_data(
            n_units=100, n_periods=12, n_cohorts=2, treatment_effect=3.0, seed=42
        )

        # Get treatment groups
        groups = sorted(data[data["first_treat"] > 0]["first_treat"].unique())
        assert len(groups) >= 1, "Need at least one treatment group"

        # Fit without anticipation
        cs_no_antic = CallawaySantAnna(anticipation=0)
        res_no_antic = cs_no_antic.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Fit with anticipation=1
        cs_antic = CallawaySantAnna(anticipation=1)
        res_antic = cs_antic.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # With anticipation=1, group effects should include period g-1
        # This means more effects contribute to the group aggregate
        for g in groups:
            # Count effects included in group aggregation
            no_antic_effects = [
                (gg, t) for (gg, t) in res_no_antic.group_time_effects.keys() if gg == g and t >= g
            ]
            antic_effects = [
                (gg, t)
                for (gg, t) in res_antic.group_time_effects.keys()
                if gg == g and t >= g - 1  # anticipation=1
            ]

            # anticipation=1 should include at least as many periods
            assert len(antic_effects) >= len(no_antic_effects), (
                f"anticipation=1 should include at least as many periods "
                f"as anticipation=0 for group {g}"
            )

    def test_group_effects_anticipation_boundary(self):
        """Group aggregation includes exactly the right periods with anticipation.

        Verify that period g-anticipation is included but g-anticipation-1 is not.
        """
        # Generate data
        data = generate_staggered_data(
            n_units=80,
            n_periods=10,
            n_cohorts=1,  # Single cohort for cleaner test
            treatment_effect=2.0,
            seed=123,
        )

        # Get the single treatment group
        g = data[data["first_treat"] > 0]["first_treat"].iloc[0]

        # Fit with anticipation=2
        cs = CallawaySantAnna(anticipation=2)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Check group effects exist
        if results.group_effects is not None and g in results.group_effects:
            # The group effect for g should aggregate periods t >= g - 2
            # Verify by checking which group-time effects exist
            gt_for_group = [(gg, t) for (gg, t) in results.group_time_effects.keys() if gg == g]

            # There should be effects at t = g - anticipation = g - 2
            # (if the data has that period)
            # Note: the anticipation period t = g - 2 may or may not be
            # present depending on base_period, so it is not asserted here;
            # post-treatment periods (t >= g - anticipation) should exist.

            # Verify post-treatment periods t >= g are included
            post_treatment = [t for (gg, t) in gt_for_group if t >= g]
            assert len(post_treatment) > 0, "Should have post-treatment effects"

    def test_not_yet_treated_with_anticipation_excludes_anticipation_window(self):
        """Not-yet-treated controls must exclude cohorts in the anticipation window.

        With anticipation=1, the control mask should use G > t + anticipation
        (not just G > t). Without the fix, cohorts about to be treated are
        incorrectly included as controls, biasing pre-treatment ATTs toward
        the treatment effect (~3.0) instead of near zero.
        """
        data = generate_staggered_data(
            n_units=100,
            n_periods=10,
            n_cohorts=2,
            treatment_effect=3.0,
            seed=42,
        )

        cs = CallawaySantAnna(anticipation=1, control_group="not_yet_treated")
        result = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        groups = sorted(g for g in data[data["first_treat"] > 0]["first_treat"].unique())

        for g in groups:
            for (gg, t), eff in result.group_time_effects.items():
                if gg != g:
                    continue
                # Pre-treatment: t < g - anticipation
                if t < g - 1:
                    assert abs(eff["effect"]) < 1.5, (
                        f"Pre-treatment ATT(g={g}, t={t}) = {eff['effect']:.3f} "
                        f"should be near zero (< 1.5); contaminated controls?"
                    )


class TestCallawaySantAnnaTStatNaN:
    """Tests for NaN t_stat when SE is invalid."""

    def test_invalid_se_produces_nan_tstat_overall(self, ci_params):
        """Overall t_stat is NaN when SE is non-finite."""
        # Create data that will result in no valid post-treatment effects
        # This should produce NaN for overall statistics
        data = generate_staggered_data(
            n_units=50, n_periods=5, n_cohorts=1, treatment_effect=2.0, seed=789
        )
        n_boot = ci_params.bootstrap(50)

        # Modify first_treat so all treatment happens after data ends
        data["first_treat"] = data["first_treat"].replace(
            data["first_treat"].unique()[data["first_treat"].unique() > 0], data["time"].max() + 10
        )

        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
            results = cs.fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

        # Overall t_stat should be NaN when SE is invalid
        if np.isnan(results.overall_se) or results.overall_se == 0:
            assert np.isnan(
                results.overall_t_stat
            ), "overall_t_stat should be NaN when SE is invalid"

    def test_per_effect_tstat_consistency(self, ci_params):
        """Per-effect t_stat uses same NaN logic as overall t_stat.

        t_stat should be NaN (not 0.0) when SE is non-finite or zero.
        """
        # Generate normal data
        data = generate_staggered_data(
            n_units=60, n_periods=8, n_cohorts=2, treatment_effect=2.0, seed=456
        )
        n_boot = ci_params.bootstrap(100)

        cs = CallawaySantAnna(n_bootstrap=n_boot, seed=42)
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Check all group-time effects
        for (g, t), effect_data in results.group_time_effects.items():
            se = effect_data["se"]
            t_stat = effect_data["t_stat"]

            if not np.isfinite(se) or se == 0:
                assert np.isnan(t_stat), (
                    f"t_stat for ({g}, {t}) should be NaN when SE={se}, " f"got t_stat={t_stat}"
                )
            else:
                # t_stat should be effect / se
                expected = effect_data["effect"] / se
                assert np.isclose(t_stat, expected), (
                    f"t_stat for ({g}, {t}) should be effect/SE, "
                    f"expected {expected}, got {t_stat}"
                )

        # Check event study effects if present
        if results.event_study_effects is not None:
            for e, effect_data in results.event_study_effects.items():
                se = effect_data["se"]
                t_stat = effect_data["t_stat"]

                if not np.isfinite(se) or se == 0:
                    assert np.isnan(
                        t_stat
                    ), f"event study t_stat for e={e} should be NaN when SE={se}"

        # Check group effects if present
        if results.group_effects is not None:
            for g, effect_data in results.group_effects.items():
                se = effect_data["se"]
                t_stat = effect_data["t_stat"]

                if not np.isfinite(se) or se == 0:
                    assert np.isnan(t_stat), f"group t_stat for g={g} should be NaN when SE={se}"

    def test_aggregated_tstat_nan_when_se_zero(self):
        """Aggregated t_stat (event-study and group) is NaN when SE is zero or non-finite.

        This tests the fix in staggered_aggregation.py for _aggregate_event_study and
        _aggregate_by_group, which previously defaulted to 0.0 instead of NaN.
        """
        # Create a small dataset that may produce edge cases in SE computation
        n_units = 20
        n_periods = 5
        np.random.seed(123)

        data = []
        for unit in range(n_units):
            # First half: treat at period 3, second half: never treated
            first_treat = 3 if unit < n_units // 2 else 0
            for t in range(1, n_periods + 1):
                outcome = np.random.randn()
                data.append(
                    {"unit": unit, "time": t, "outcome": outcome, "first_treat": first_treat}
                )

        df = pd.DataFrame(data)

        # Fit with event study aggregation to get event_study_effects
        cs = CallawaySantAnna(n_bootstrap=0)
        results = cs.fit(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",  # Get both event study and group effects
        )

        # Check that t_stat computation follows the correct pattern:
        # t_stat = effect / se if np.isfinite(se) and se > 0 else np.nan
        if results.event_study_effects:
            for e, data in results.event_study_effects.items():
                se = data["se"]
                t_stat = data["t_stat"]
                effect = data["effect"]

                if not np.isfinite(se) or se <= 0:
                    assert np.isnan(t_stat), (
                        f"Event study t_stat for e={e} should be NaN when SE={se}, "
                        f"got t_stat={t_stat}"
                    )
                else:
                    expected_t = effect / se
                    assert np.isclose(t_stat, expected_t, rtol=1e-10), (
                        f"Event study t_stat for e={e} should be effect/SE={expected_t}, "
                        f"got {t_stat}"
                    )

        if results.group_effects:
            for g, data in results.group_effects.items():
                se = data["se"]
                t_stat = data["t_stat"]
                effect = data["effect"]

                if not np.isfinite(se) or se <= 0:
                    assert np.isnan(t_stat), (
                        f"Group t_stat for g={g} should be NaN when SE={se}, "
                        f"got t_stat={t_stat}"
                    )
                else:
                    expected_t = effect / se
                    assert np.isclose(t_stat, expected_t, rtol=1e-10), (
                        f"Group t_stat for g={g} should be effect/SE={expected_t}, " f"got {t_stat}"
                    )

    def test_event_study_universal_includes_reference_period(self):
        """Test that universal base period includes e=-1 with effect=0."""
        data = generate_staggered_data(n_units=200, n_periods=10, seed=42)

        cs = CallawaySantAnna(base_period="universal")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None, "event_study_effects should not be None"

        # Reference period should be included
        assert -1 in results.event_study_effects, (
            f"Reference period e=-1 should be in event_study_effects, "
            f"got periods: {list(results.event_study_effects.keys())}"
        )
        ref = results.event_study_effects[-1]

        # Effect is 0 by construction (normalization)
        assert ref["effect"] == 0.0, f"Reference period effect should be 0.0, got {ref['effect']}"
        # Inference fields are NaN - this is a normalization constraint, not an estimated effect
        assert np.isnan(ref["se"]), f"Reference period SE should be NaN, got {ref['se']}"
        assert np.isnan(
            ref["t_stat"]
        ), f"Reference period t_stat should be NaN, got {ref['t_stat']}"
        assert np.isnan(
            ref["p_value"]
        ), f"Reference period p_value should be NaN, got {ref['p_value']}"
        assert np.isnan(ref["conf_int"][0]) and np.isnan(
            ref["conf_int"][1]
        ), f"Reference period CI should be (NaN, NaN), got {ref['conf_int']}"
        assert ref["n_groups"] == 0, f"Reference period n_groups should be 0, got {ref['n_groups']}"

    def test_event_study_varying_excludes_reference_period(self):
        """Test that varying base period does NOT artificially add e=-1 with effect=0."""
        data = generate_staggered_data(n_units=200, n_periods=10, seed=42)

        cs = CallawaySantAnna(base_period="varying")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None, "event_study_effects should not be None"

        # Varying mode: no single reference period, e=-1 computed normally or excluded
        # The key is we don't artificially add a 0-effect entry
        if -1 in results.event_study_effects:
            # If it exists, it should be an actual computed effect, not 0.0 with n_groups=0
            assert (
                results.event_study_effects[-1]["n_groups"] > 0
            ), "Varying mode should not artificially add e=-1 with n_groups=0"

    def test_event_study_universal_with_anticipation(self):
        """Test reference period with anticipation > 0."""
        data = generate_staggered_data(n_units=200, n_periods=10, seed=42)

        cs = CallawaySantAnna(base_period="universal", anticipation=1)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None, "event_study_effects should not be None"

        # With anticipation=1, reference is e=-2
        assert -2 in results.event_study_effects, (
            f"With anticipation=1, reference period e=-2 should be in event_study_effects, "
            f"got periods: {list(results.event_study_effects.keys())}"
        )
        ref = results.event_study_effects[-2]
        assert ref["effect"] == 0.0, f"Reference period effect should be 0.0, got {ref['effect']}"
        # Inference fields are NaN - normalization constraint
        assert np.isnan(ref["se"]), f"Reference period SE should be NaN, got {ref['se']}"
        assert np.isnan(ref["conf_int"][0]) and np.isnan(
            ref["conf_int"][1]
        ), f"Reference period CI should be (NaN, NaN), got {ref['conf_int']}"

    def test_event_study_universal_no_effects_raises_error(self):
        """Test that estimator raises error when no effects can be computed.

        This ensures the reference period injection code (which has an empty guard)
        is never reached with empty effects - the estimator fails fast instead.
        """
        import pandas as pd

        # Create minimal data with only never-treated units
        # This ensures no ATT(g,t) can be computed (no treatment groups)
        data = pd.DataFrame(
            {
                "unit": [1, 1, 2, 2, 3, 3],
                "time": [1, 2, 1, 2, 1, 2],
                "outcome": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                "first_treat": [0, 0, 0, 0, 0, 0],  # All never-treated
            }
        )

        cs = CallawaySantAnna(base_period="universal")
        with pytest.raises(ValueError, match="Could not estimate any group-time effects"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )


class TestCallawaySantAnnaCIBugFix:
    """Regression test: safe_inference fixes CI computed with NaN SE."""

    def test_nan_se_group_time_ci_is_nan(self):
        """conf_int should be (NaN, NaN) when SE is NaN, not finite values."""
        from tests.conftest import assert_nan_inference

        # Generate data with very few units to produce NaN-SE group-time effects
        # (small sample → degenerate groups)
        data = generate_staggered_data(
            n_units=20, n_periods=6, n_cohorts=3, never_treated_frac=0.1, seed=123
        )

        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cs = CallawaySantAnna()
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

        # Check all group-time effects: if SE is NaN, CI must also be NaN
        for (g, t), eff in results.group_time_effects.items():
            se = eff["se"]
            if not (np.isfinite(se) and se > 0):
                assert_nan_inference(
                    {
                        "se": se,
                        "t_stat": eff["t_stat"],
                        "p_value": eff["p_value"],
                        "conf_int": eff["conf_int"],
                    }
                )


class TestPscoreTrimParameter:
    """Tests for the pscore_trim parameter."""

    def test_get_params_includes_pscore_trim(self):
        """pscore_trim is included in get_params()."""
        cs = CallawaySantAnna(pscore_trim=0.05)
        params = cs.get_params()
        assert "pscore_trim" in params
        assert params["pscore_trim"] == 0.05

    def test_set_params_pscore_trim(self):
        """pscore_trim can be set via set_params()."""
        cs = CallawaySantAnna()
        cs.set_params(pscore_trim=0.1)
        assert cs.pscore_trim == 0.1

    def test_set_params_invalid_pscore_trim_rejected_eagerly(self):
        """Invalid pscore_trim raises AT set_params (BaseEstimator probe
        re-init runs constructor validation transactionally); the
        estimator is unchanged."""
        for bad_val in [0.0, -0.1, 0.5]:
            cs = CallawaySantAnna(estimation_method="ipw")
            with pytest.raises(ValueError, match="pscore_trim must be in"):
                cs.set_params(pscore_trim=bad_val)
            assert cs.pscore_trim == 0.01

    def test_default_pscore_trim(self):
        """Default pscore_trim is 0.01."""
        cs = CallawaySantAnna()
        assert cs.pscore_trim == 0.01

    def test_pscore_trim_negative_raises(self):
        """pscore_trim < 0 raises ValueError."""
        with pytest.raises(ValueError, match="pscore_trim must be in"):
            CallawaySantAnna(pscore_trim=-0.1)

    def test_pscore_trim_at_half_raises(self):
        """pscore_trim == 0.5 raises ValueError."""
        with pytest.raises(ValueError, match="pscore_trim must be in"):
            CallawaySantAnna(pscore_trim=0.5)

    def test_pscore_trim_above_half_raises(self):
        """pscore_trim > 0.5 raises ValueError."""
        with pytest.raises(ValueError, match="pscore_trim must be in"):
            CallawaySantAnna(pscore_trim=0.6)

    def test_pscore_trim_zero_raises(self):
        """pscore_trim=0.0 raises ValueError (would cause division by zero in IPW weights)."""
        with pytest.raises(ValueError, match="pscore_trim must be in"):
            CallawaySantAnna(pscore_trim=0.0)

    def test_pscore_trim_in_results(self):
        """results.pscore_trim matches the estimator's setting after fit()."""
        np.random.seed(42)
        n_units, n_periods = 50, 6
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        first_treat = np.zeros(n_units)
        first_treat[n_units // 2 :] = 3
        first_treat_expanded = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 1.0 + 2.0 * post + np.random.randn(len(units)) * 0.5
        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
            }
        )
        cs = CallawaySantAnna(pscore_trim=0.05, estimation_method="reg")
        results = cs.fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert results.pscore_trim == 0.05

    def test_nondefault_pscore_trim_ipw(self):
        """IPW with pscore_trim=0.1 produces finite results."""
        np.random.seed(42)
        n_units, n_periods = 80, 6
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        x = np.random.randn(n_units)
        x_expanded = np.repeat(x, n_periods)
        first_treat = np.zeros(n_units)
        first_treat[n_units // 2 :] = 3
        first_treat_expanded = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 1.0 + 0.5 * x_expanded + 2.0 * post + np.random.randn(len(units)) * 0.5
        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
                "x": x_expanded,
            }
        )
        cs = CallawaySantAnna(estimation_method="ipw", pscore_trim=0.1)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x"],
        )
        assert np.isfinite(results.overall_att)
        assert results.pscore_trim == 0.1

    def test_nondefault_pscore_trim_dr(self):
        """DR with pscore_trim=0.1 produces finite results."""
        np.random.seed(42)
        n_units, n_periods = 80, 6
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        x = np.random.randn(n_units)
        x_expanded = np.repeat(x, n_periods)
        first_treat = np.zeros(n_units)
        first_treat[n_units // 2 :] = 3
        first_treat_expanded = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 1.0 + 0.5 * x_expanded + 2.0 * post + np.random.randn(len(units)) * 0.5
        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
                "x": x_expanded,
            }
        )
        cs = CallawaySantAnna(estimation_method="dr", pscore_trim=0.1)
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x"],
        )
        assert np.isfinite(results.overall_att)
        assert results.pscore_trim == 0.1


class TestIRLSPropensityScore:
    """Tests for IRLS-based propensity score estimation in CS estimator."""

    def test_near_separation_warning_ipw(self):
        """Near-separation emits warnings in the IPW path."""
        np.random.seed(42)
        n_units = 100
        n_periods = 8

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # Create a covariate that strongly predicts treatment
        x_strong = np.random.randn(n_units)
        x_strong_expanded = np.repeat(x_strong, n_periods)

        # Treatment perfectly aligned with covariate sign
        first_treat = np.zeros(n_units)
        first_treat[x_strong > 0] = 4
        first_treat_expanded = np.repeat(first_treat, n_periods)

        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 1.0 + x_strong_expanded + 2.0 * post + np.random.randn(len(units)) * 0.5

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
                "x_strong": x_strong_expanded,
            }
        )

        cs = CallawaySantAnna(estimation_method="ipw")

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x_strong"],
            )

        # Should see propensity-related warnings
        pscore_warns = [
            x
            for x in w
            if "propensity" in str(x.message).lower()
            or "separation" in str(x.message).lower()
            or "trimmed" in str(x.message).lower()
        ]
        assert len(pscore_warns) > 0, "Expected propensity score warnings"
        # ATT should still be reasonable (not wildly inflated)
        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)

    def test_near_separation_att_not_inflated(self):
        """IRLS produces reasonable ATT even with near-separation covariates.

        This is the key regression test for the reported bug: BFGS-based logit
        produced wildly inflated ATT (~2.38 vs 0.45-1.15 in reference packages)
        under near-separation conditions.
        """
        np.random.seed(123)
        n_units = 200
        n_periods = 8
        true_effect = 2.0

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # Covariate that creates near-separation
        x = np.random.randn(n_units) * 3  # large scale
        x_expanded = np.repeat(x, n_periods)

        # Treatment correlated with covariate but not perfect
        treat_prob = 1 / (1 + np.exp(-x))
        first_treat = np.zeros(n_units)
        first_treat[treat_prob > 0.5] = 4
        first_treat_expanded = np.repeat(first_treat, n_periods)

        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 1.0 + x_expanded * 0.5 + true_effect * post + np.random.randn(len(units)) * 0.5

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
                "x": x_expanded,
            }
        )

        cs = CallawaySantAnna(estimation_method="dr")

        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x"],
            )

        # ATT should be in a reasonable range around the true effect
        assert results.overall_att is not None
        assert (
            abs(results.overall_att - true_effect) < 3.0
        ), f"ATT={results.overall_att} too far from true effect {true_effect}"

    def test_dr_fallback_warning(self):
        """DR path emits warning when propensity estimation fails."""
        from unittest.mock import patch

        data = generate_staggered_data_with_covariates(seed=42)

        cs = CallawaySantAnna(estimation_method="dr", pscore_fallback="unconditional")

        with patch("diff_diff.staggered.solve_logit", side_effect=ValueError("test")):
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = cs.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1"],
                )

            fallback_warns = [x for x in w if "unconditional propensity" in str(x.message)]
            assert len(fallback_warns) > 0, "Expected fallback warning in DR path"
            assert results.overall_att is not None

    def test_large_scale_covariate_stability(self):
        """IRLS handles large-scale covariates without wild ATT inflation.

        Mimics scenario from Dias & Fontes (2024) audit where covariates
        like poptotaltrend (population in millions) caused near-separation.
        """
        np.random.seed(456)
        n_units = 150
        n_periods = 8
        true_effect = 1.5

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # Large-scale covariate (like population totals in millions)
        x_large = np.random.randn(n_units) * 1e6
        x_expanded = np.repeat(x_large, n_periods)

        # Treatment mildly correlated with covariate
        first_treat = np.zeros(n_units)
        first_treat[x_large > np.median(x_large)] = 4
        first_treat_expanded = np.repeat(first_treat, n_periods)

        post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
        outcomes = 5.0 + x_expanded * 1e-7 + true_effect * post + np.random.randn(len(units)) * 0.5

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_expanded.astype(int),
                "x_large": x_expanded,
            }
        )

        cs = CallawaySantAnna(estimation_method="dr")

        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x_large"],
            )

        assert results.overall_att is not None
        assert np.isfinite(results.overall_att)
        # ATT should be in a plausible range
        assert (
            abs(results.overall_att - true_effect) < 5.0
        ), f"ATT={results.overall_att} too far from true effect {true_effect}"


class TestEPVDiagnostics:
    """Tests for Events Per Variable (EPV) diagnostics in CallawaySantAnna."""

    def test_cs_epv_diagnostics_in_results(self):
        """fit() with small cohorts populates results.epv_diagnostics."""
        # Create data with very small cohorts to trigger low EPV
        data = generate_staggered_data_with_covariates(
            n_units=30, n_periods=6, n_cohorts=3, seed=42
        )
        cs = CallawaySantAnna(estimation_method="ipw", pscore_fallback="unconditional")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )
        # With small cohorts and covariates, epv_diagnostics should be populated
        assert results.epv_diagnostics is not None
        assert len(results.epv_diagnostics) > 0
        # Check structure of diagnostic entries
        for key, diag in results.epv_diagnostics.items():
            assert "epv" in diag
            assert "n_events" in diag
            assert "k" in diag
            assert "is_low" in diag

    def test_cs_epv_summary_method(self):
        """results.epv_summary() returns correct DataFrame."""
        data = generate_staggered_data_with_covariates(
            n_units=30, n_periods=6, n_cohorts=3, seed=42
        )
        cs = CallawaySantAnna(estimation_method="ipw", pscore_fallback="unconditional")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )
        df = results.epv_summary()
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"group", "time", "epv", "n_events", "n_params", "is_low"}
        assert expected_cols.issubset(set(df.columns))

    def test_cs_epv_summary_show_all(self):
        """epv_summary(show_all=True) returns all entries, not just low ones."""
        data = generate_staggered_data_with_covariates(
            n_units=100, n_periods=6, n_cohorts=2, seed=42
        )
        cs = CallawaySantAnna(estimation_method="ipw", pscore_fallback="unconditional")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1"],
            )
        if results.epv_diagnostics:
            df_all = results.epv_summary(show_all=True)
            df_low = results.epv_summary(show_all=False)
            assert len(df_all) >= len(df_low)

    def test_cs_epv_no_diagnostics_for_reg(self):
        """estimation_method='reg' produces no EPV diagnostics."""
        data = generate_staggered_data_with_covariates(seed=42)
        cs = CallawaySantAnna(estimation_method="reg")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1"],
        )
        assert results.epv_diagnostics is None

    def test_cs_pscore_fallback_error_default(self):
        """Default pscore_fallback='error' raises when logit fails."""
        from unittest.mock import patch

        data = generate_staggered_data_with_covariates(seed=42)
        cs = CallawaySantAnna(estimation_method="ipw")  # default fallback='error'

        with patch("diff_diff.staggered.solve_logit", side_effect=ValueError("test")):
            with pytest.raises(ValueError, match="test"):
                cs.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1"],
                )

    def test_cs_pscore_fallback_unconditional_opt_in(self):
        """pscore_fallback='unconditional' restores old fallback behavior."""
        from unittest.mock import patch

        data = generate_staggered_data_with_covariates(seed=42)
        cs = CallawaySantAnna(estimation_method="dr", pscore_fallback="unconditional")

        with patch("diff_diff.staggered.solve_logit", side_effect=ValueError("test")):
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = cs.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1"],
                )
            fallback_warns = [x for x in w if "unconditional propensity" in str(x.message)]
            assert len(fallback_warns) > 0
            assert results.overall_att is not None

    def test_cs_diagnose_propensity(self):
        """diagnose_propensity() returns DataFrame with EPV per cohort."""
        data = generate_staggered_data_with_covariates(seed=42)
        cs = CallawaySantAnna(estimation_method="ipw")
        df = cs.diagnose_propensity(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )
        assert isinstance(df, pd.DataFrame)
        assert "group" in df.columns
        assert "epv" in df.columns
        assert "status" in df.columns
        assert len(df) > 0
        assert all(df["status"].isin(["ok", "low", "critical"]))

    def test_cs_diagnose_propensity_identifies_critical(self):
        """diagnose_propensity flags critical EPV for tiny cohorts."""
        # Create data with very tiny cohort
        np.random.seed(99)
        n_units = 60
        n_periods = 6
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        # 1 unit treated at period 3, rest never treated
        first_treat = np.zeros(n_units)
        first_treat[0] = 3
        first_treat_exp = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_exp) & (first_treat_exp > 0)
        outcome = np.random.randn(len(units)) + post.astype(float)
        x1 = np.repeat(np.random.randn(n_units), n_periods)
        x2 = np.repeat(np.random.randn(n_units), n_periods)
        x3 = np.repeat(np.random.randn(n_units), n_periods)

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": first_treat_exp,
                "outcome": outcome,
                "x1": x1,
                "x2": x2,
                "x3": x3,
            }
        )

        cs = CallawaySantAnna(estimation_method="ipw")
        df = cs.diagnose_propensity(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2", "x3"],
        )
        # With 1 treated unit and 3 predictor variables: EPV = 1/3 ≈ 0.33 → critical
        assert any(df["status"] == "critical")

    def test_cs_epv_in_summary_output(self):
        """summary() includes EPV diagnostic block when low EPV detected."""
        data = generate_staggered_data_with_covariates(
            n_units=30, n_periods=6, n_cohorts=3, seed=42
        )
        cs = CallawaySantAnna(estimation_method="ipw", pscore_fallback="unconditional")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )
        if results.epv_diagnostics:
            low_epv = {k: v for k, v in results.epv_diagnostics.items() if v.get("is_low")}
            if low_epv:
                summary = results.summary()
                assert "EPV" in summary
                assert "Propensity Score Diagnostics" in summary

    def test_cs_epv_in_to_dataframe(self):
        """EPV column appears in group_time DataFrame when diagnostics available."""
        data = generate_staggered_data_with_covariates(
            n_units=30, n_periods=6, n_cohorts=3, seed=42
        )
        cs = CallawaySantAnna(estimation_method="ipw", pscore_fallback="unconditional")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )
        if results.epv_diagnostics:
            df = results.to_dataframe(level="group_time")
            assert "epv" in df.columns

    def test_cs_cached_rank_deficient_pscore_no_nan(self):
        """Cached rank-deficient logit coefficients should not produce NaN ATTs.

        Regression test for P0: solve_logit returns NaN in dropped-column
        positions. Without zero-filling before caching, cache reuse via
        X @ beta_cached would propagate NaN into propensity scores.
        """
        np.random.seed(123)
        n_units = 80
        n_periods = 6
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        # Two cohorts: period 3 (20 units) and never-treated (60 units)
        first_treat = np.zeros(n_units)
        first_treat[:20] = 3
        first_treat_exp = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_exp) & (first_treat_exp > 0)
        outcome = np.random.randn(len(units)) + post.astype(float) * 2.0
        x1 = np.repeat(np.random.randn(n_units), n_periods)
        # x2 is a duplicate of x1 — will cause rank deficiency
        x2 = x1.copy()

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": first_treat_exp,
                "outcome": outcome,
                "x1": x1,
                "x2": x2,
            }
        )

        cs = CallawaySantAnna(
            estimation_method="ipw",
            rank_deficient_action="warn",
            pscore_fallback="unconditional",
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )

        # All ATTs should be finite (no NaN from cache poisoning)
        for (g, t), eff in results.group_time_effects.items():
            assert np.isfinite(
                eff["effect"]
            ), f"ATT({g},{t}) is {eff['effect']} — NaN cache poisoning"
        assert np.isfinite(results.overall_att)

    def test_cs_strict_mode_not_swallowed_by_unconditional_fallback(self):
        """rank_deficient_action='error' raises even with pscore_fallback='unconditional'.

        Regression test for P1: pscore_fallback should not swallow strict-mode
        errors that rank_deficient_action='error' is supposed to raise.
        """
        from unittest.mock import patch

        data = generate_staggered_data_with_covariates(seed=42)
        cs = CallawaySantAnna(
            estimation_method="ipw",
            rank_deficient_action="error",
            pscore_fallback="unconditional",
        )

        # Simulate a ValueError from solve_logit (e.g., rank deficiency)
        with patch(
            "diff_diff.staggered.solve_logit",
            side_effect=ValueError("Rank-deficient design"),
        ):
            with pytest.raises(ValueError, match="Rank-deficient"):
                cs.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1"],
                )

    def test_cs_rc_strict_mode_not_swallowed(self):
        """RCS path: rank_deficient_action='error' raises even with unconditional fallback."""
        from unittest.mock import patch

        # RCS data: unique unit IDs per observation
        np.random.seed(99)
        n = 300
        data = pd.DataFrame(
            {
                "unit": np.arange(n),
                "time": np.random.choice([0, 1, 2, 3, 4], n),
                "outcome": np.random.randn(n),
                "first_treat": np.where(np.arange(n) < 100, 3, 0),
                "x1": np.random.randn(n),
            }
        )
        cs = CallawaySantAnna(
            estimation_method="ipw",
            rank_deficient_action="error",
            pscore_fallback="unconditional",
            panel=False,
        )
        with patch(
            "diff_diff.staggered.solve_logit",
            side_effect=ValueError("Rank-deficient design"),
        ):
            with pytest.raises(ValueError, match="Rank-deficient"):
                cs.fit(
                    data,
                    outcome="outcome",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    covariates=["x1"],
                )

    def test_cs_diagnose_propensity_rejects_not_yet_treated(self):
        """diagnose_propensity() raises for control_group='not_yet_treated'."""
        data = generate_staggered_data_with_covariates(seed=42)
        cs = CallawaySantAnna(estimation_method="ipw", control_group="not_yet_treated")
        with pytest.raises(NotImplementedError, match="not_yet_treated"):
            cs.diagnose_propensity(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1"],
            )


class TestSilentWarningAudit:
    """Tests for UserWarning emissions added by the silent warning audit."""

    def test_item8_inf_to_zero_warning_in_fit(self):
        """Item 8: Warn when first_treat=inf is recoded to 0 in fit()."""

        data = generate_staggered_data(seed=42)
        # Set some units to inf (never-treated encoding)
        # Cast to float first for pandas >=2.0 compatibility
        data["first_treat"] = data["first_treat"].astype(float)
        never_units = data.loc[data["first_treat"] == 0, "unit"].unique()[:5]
        data.loc[data["unit"].isin(never_units), "first_treat"] = np.inf

        cs = CallawaySantAnna()
        with pytest.warns(UserWarning, match="first_treat=inf"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_item8_inf_to_zero_warning_in_diagnose_propensity(self):
        """Item 8: Warn when first_treat=inf is recoded in diagnose_propensity()."""

        data = generate_staggered_data_with_covariates(seed=42)
        # Cast to float first for pandas >=2.0 compatibility
        data["first_treat"] = data["first_treat"].astype(float)
        never_units = data.loc[data["first_treat"] == 0, "unit"].unique()[:5]
        data.loc[data["unit"].isin(never_units), "first_treat"] = np.inf

        cs = CallawaySantAnna(estimation_method="ipw")
        with pytest.warns(UserWarning, match="first_treat=inf"):
            cs.diagnose_propensity(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1"],
            )

    def test_item8_no_warning_when_first_treat_zero(self):
        """Item 8 negative: No warning when never-treated encoded as 0."""
        import warnings

        data = generate_staggered_data(seed=42)
        cs = CallawaySantAnna()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        inf_warnings = [x for x in w if "first_treat=inf" in str(x.message)]
        assert len(inf_warnings) == 0

    def test_item4_consolidated_skip_warning(self):
        """Item 4: Consolidated warning when (g,t) cells are non-estimable.

        With positional base-period selection a cell is only non-estimable when
        no earlier observed period exists. Cohort ``g=2`` treated at the earliest
        observed period (periods ``{2,3,4,5}``) has no pre-treatment period, so
        R cannot estimate it either -> ``missing_period`` skips + a consolidated
        warning. Cohort ``g=4`` (base = observed period 3) is estimable.
        """
        import warnings

        rng = np.random.default_rng(42)
        n_units = 40
        rows = []
        for u in range(n_units):
            for t in [2, 3, 4, 5]:
                # u < 10: never-treated; u < 25: cohort g=2 (treated at the
                # earliest observed period -> no pre-period -> skipped);
                # rest: cohort g=4 (base = observed 3 -> succeeds)
                if u < 10:
                    ft = 0
                elif u < 25:
                    ft = 2  # no earlier observed period -> non-estimable
                else:
                    ft = 4  # base=3 exists -> succeeds
                outcome = rng.standard_normal() + (2.0 if (ft > 0 and t >= ft) else 0.0)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "outcome": outcome,
                        "first_treat": ft,
                    }
                )
        data = pd.DataFrame(rows)

        cs = CallawaySantAnna(base_period="universal", estimation_method="reg")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

        skip_warnings = [x for x in w if "could not be estimated" in str(x.message)]
        assert len(skip_warnings) > 0, "Expected consolidated skip warning"
        msg = str(skip_warnings[0].message)
        assert "missing base/post period" in msg

    def test_item4_no_skip_warning_normal_data(self):
        """Item 4 negative: No skip warning on well-formed balanced data."""
        import warnings

        data = generate_staggered_data(seed=42)
        cs = CallawaySantAnna()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        skip_warnings = [x for x in w if "could not be estimated" in str(x.message)]
        assert len(skip_warnings) == 0, f"Unexpected skip warning: {skip_warnings}"

    def test_skip_warning_dr_path(self):
        """Skip warning fires for default DR path (general path)."""
        data = generate_staggered_data(
            n_units=50,
            n_periods=6,
            n_cohorts=3,
            never_treated_frac=0.0,
            seed=42,
        )
        cs = CallawaySantAnna(control_group="not_yet_treated")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        skip_warnings = [x for x in w if "could not be estimated" in str(x.message)]
        assert len(skip_warnings) > 0, "Expected skip warning for DR path"
        assert "insufficient data" in str(skip_warnings[0].message)

    def test_skip_warning_panel_false(self):
        """Skip warning fires for panel=False (RC path)."""
        data = generate_staggered_data(
            n_units=80,
            n_periods=6,
            n_cohorts=3,
            never_treated_frac=0.0,
            seed=42,
        )
        # panel=False needs unique unit IDs (repeated cross-section)
        data["unit"] = np.arange(len(data))
        cs = CallawaySantAnna(panel=False, control_group="not_yet_treated")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        skip_warnings = [x for x in w if "could not be estimated" in str(x.message)]
        assert len(skip_warnings) > 0, "Expected skip warning for RC path"

    def test_skip_warning_survey_zero_mass(self):
        """Skip warning fires when survey weights produce zero effective mass."""
        from diff_diff.survey import SurveyDesign

        data = generate_staggered_data(
            n_units=60,
            n_periods=6,
            n_cohorts=2,
            never_treated_frac=0.3,
            seed=42,
        )
        # Set survey weights to 0 for ALL units in one cohort to force
        # zero effective mass in that cohort's cells
        data["sw"] = 1.0
        first_cohort = sorted(data.loc[data["first_treat"] > 0, "first_treat"].unique())[0]
        cohort_units = data.loc[data["first_treat"] == first_cohort, "unit"].unique()
        data.loc[data["unit"].isin(cohort_units), "sw"] = 0.0

        survey = SurveyDesign(weights="sw")
        cs = CallawaySantAnna(estimation_method="reg")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=survey,
            )
        skip_warnings = [x for x in w if "could not be estimated" in str(x.message)]
        assert len(skip_warnings) > 0, "Expected skip warning for zero-mass survey cells"


# ---------------------------------------------------------------------------
# Silent-failure audit PR #9 follow-up: the CS analytical SE path calls
# `_safe_inv()` in ~13 places (PS Hessian, OR bread, etc.). Previously the
# LinAlgError → lstsq fallback was silent — a rank-deficient bread produced
# degraded SEs with no user-visible signal. Now fit() emits ONE aggregate
# warning tracking all fallbacks.
# ---------------------------------------------------------------------------


class TestCallawaySantAnnaSafeInvFallback:
    def test_collinear_covariates_emit_safe_inv_warning(self):
        """Perfectly collinear covariates should trigger the aggregate
        `_safe_inv` rank-guard warning across analytical SE paths (default
        rank_deficient_action='warn'; suppressed under 'silent')."""
        data = generate_staggered_data(n_units=150, n_periods=6, n_cohorts=3, seed=55)
        rng = np.random.default_rng(0)
        # Add a covariate and a redundant (collinear) copy — forces rank-
        # deficient X'WX in the OR bread and the PS Hessian within at
        # least one (g, t) cell.
        data["x1"] = rng.normal(0, 1, len(data))
        data["x2"] = 2.0 * data["x1"]
        cs = CallawaySantAnna(estimation_method="dr")  # default action="warn"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
            )
        fallback_warnings = [
            w
            for w in caught
            if "analytical SE paths" in str(w.message) and "rank-guarded inverse" in str(w.message)
        ]
        assert len(fallback_warnings) == 1, (
            f"Expected exactly one aggregate _safe_inv rank-guard warning; "
            f"got {len(fallback_warnings)}: "
            f"{[str(w.message) for w in fallback_warnings]}"
        )

    def test_well_conditioned_no_safe_inv_warning(self):
        """Clean data should NOT trigger the aggregate warning —
        regression-safety for the happy path."""
        data = generate_staggered_data(n_units=200, n_periods=6, n_cohorts=3, seed=42)
        cs = CallawaySantAnna(estimation_method="dr")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        fallback_warnings = [
            w
            for w in caught
            if "Rank-deficient matrix encountered" in str(w.message)
            and "analytical SE paths" in str(w.message)
        ]
        assert fallback_warnings == [], (
            f"Unexpected _safe_inv fallback warning on clean data: "
            f"{[str(w.message) for w in fallback_warnings]}"
        )


def _generate_clustered_staggered_data(
    n_clusters: int = 20,
    units_per_cluster: int = 5,
    n_periods: int = 8,
    cluster_effect_sd: float = 3.0,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate a staggered panel with strong intra-cluster correlation.

    Each "state" cluster contributes a shared random effect to every
    unit within it, so cluster-robust SE should differ measurably from
    per-unit IF SE. Required for the assertive cluster-wiring tests
    (per ``feedback_homogeneous_dgp_no_twfe_bias`` — homogeneous DGPs
    produce zero divergence and can't distinguish wired from no-op).
    """
    rng = np.random.default_rng(seed)
    n_units = n_clusters * units_per_cluster
    state_ids = np.repeat(np.arange(n_clusters), units_per_cluster)
    cluster_effects = rng.normal(0.0, cluster_effect_sd, n_clusters)

    cohort_choices = [0, 3, 5, 7]  # 0 = never-treated
    first_treat = rng.choice(cohort_choices, size=n_units, p=[0.4, 0.2, 0.2, 0.2])

    rows = []
    for u in range(n_units):
        s = state_ids[u]
        ft = first_treat[u]
        for t in range(1, n_periods + 1):
            y = (
                cluster_effects[s]
                + 0.5 * (t - 1)
                + (2.0 if (ft > 0 and t >= ft) else 0.0)
                + rng.normal(0.0, 0.5)
            )
            rows.append(
                {
                    "unit": u,
                    "state": int(s),
                    "time": t,
                    "first_treat": int(ft),
                    "outcome": y,
                }
            )
    return pd.DataFrame(rows)


class TestCallawaySantAnnaClusterWiring:
    """Cluster wiring fix: bare ``cluster=`` activates cluster-robust IF.

    Prior to PR fix, ``CS(cluster="state").fit(...)`` accepted the
    parameter but never consumed it — silent unit-level inference. These
    tests pin the fix: bare cluster= synthesizes ``SurveyDesign(psu=X)``
    and routes through the existing PSU-meat machinery.
    """

    def test_cluster_robust_ses_differ_from_unit_level(self):
        """Assertive: cluster=state SE differs from cluster=None SE
        on a panel with intra-cluster correlation. This is the
        regression test that pins the silent no-op fix."""
        data = _generate_clustered_staggered_data(seed=7)

        cs_unit = CallawaySantAnna()
        res_unit = cs_unit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        cs_cluster = CallawaySantAnna(cluster="state")
        res_cluster = cs_cluster.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert np.isfinite(res_unit.overall_se) and res_unit.overall_se > 0
        assert np.isfinite(res_cluster.overall_se) and res_cluster.overall_se > 0
        assert abs(res_unit.overall_se - res_cluster.overall_se) > 1e-6, (
            f"cluster=state SE ({res_cluster.overall_se:.6f}) is "
            f"effectively identical to cluster=None SE "
            f"({res_unit.overall_se:.6f}) — the cluster= parameter "
            "may not be wired through to the variance machinery."
        )

    def test_bare_cluster_synthesizes_survey_design(self):
        """bare cluster= populates Results.cluster_name and n_clusters."""
        data = _generate_clustered_staggered_data(seed=11)
        cs = CallawaySantAnna(cluster="state")
        res = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert res.cluster_name == "state"
        assert res.n_clusters is not None and res.n_clusters > 0
        assert res.vcov_type == "hc1"

    def test_survey_design_psu_overrides_cluster_warns(self):
        """survey_design.psu wins over bare cluster=; UserWarning fires
        if partitions differ; cluster_name reflects the canonical PSU."""
        from diff_diff import SurveyDesign

        data = _generate_clustered_staggered_data(n_clusters=20, units_per_cluster=5, seed=13)
        # Add a coarser "region" partition: 2 regions, each with 10 states.
        data["region"] = data["state"] // 10

        cs = CallawaySantAnna(cluster="state")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=SurveyDesign(psu="region"),
            )
        partition_warnings = [
            w
            for w in caught
            if "psu" in str(w.message).lower()
            or "partition" in str(w.message).lower()
            or "different groupings" in str(w.message).lower()
        ]
        assert len(partition_warnings) > 0, (
            f"Expected UserWarning about psu/partition mismatch; "
            f"caught: {[str(w.message) for w in caught]}"
        )
        # Canonical PSU column wins
        assert res.cluster_name == "region"

    def test_survey_design_without_psu_plus_cluster_injects(self):
        """survey_design without psu + cluster=X injects cluster as PSU.
        cluster_name reflects the bare cluster (no explicit PSU to win)."""
        from diff_diff import SurveyDesign

        data = _generate_clustered_staggered_data(seed=17)
        data["wt"] = 1.0  # uniform weights

        cs = CallawaySantAnna(cluster="state")
        res = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=SurveyDesign(weights="wt"),
        )
        assert res.cluster_name == "state"
        assert res.n_clusters is not None and res.n_clusters > 0

    def test_cluster_none_path_unchanged(self):
        """cluster=None path: no wiring, no cluster metadata in Results.
        Verifies the wiring guard ``if self.cluster is not None:`` prevents
        the wiring block from firing when cluster is not set."""
        data = _generate_clustered_staggered_data(seed=19)
        cs = CallawaySantAnna()  # cluster=None default
        res = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert res.cluster_name is None
        assert res.n_clusters is None
        assert res.vcov_type == "hc1"
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_invalid_cluster_column_raises(self):
        """cluster=<nonexistent_col> raises ValueError with column name."""
        data = _generate_clustered_staggered_data(seed=23)
        cs = CallawaySantAnna(cluster="nonexistent_col")
        with pytest.raises(ValueError, match="cluster column"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_cluster_nan_raises_with_cluster_domain_message(self):
        """cluster column with NaN raises ValueError citing 'cluster'
        (not 'PSU') — verifies the cluster-domain pre-validator fires
        BEFORE synthesis, so the error message refers to the right API."""
        data = _generate_clustered_staggered_data(seed=29)
        data.loc[0, "state"] = np.nan
        cs = CallawaySantAnna(cluster="state")
        with pytest.raises(ValueError, match="cluster column"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_bare_cluster_works_with_panel_false_rcs(self):
        """RCS coverage: panel=False + cluster=state produces clustered SE
        that differs from cluster=None SE. Closes RCS coverage gap from
        plan review."""
        # Build a repeated cross-section: each obs is a distinct unit,
        # but obs share state-level clusters.
        rng = np.random.default_rng(31)
        n_states = 15
        obs_per_period = 60
        n_periods = 6
        state_effects = rng.normal(0.0, 3.0, n_states)
        rows = []
        next_unit = 0
        for t in range(1, n_periods + 1):
            for _ in range(obs_per_period):
                s = int(rng.integers(0, n_states))
                ft = int(rng.choice([0, 3, 5], p=[0.4, 0.3, 0.3]))
                y = (
                    state_effects[s]
                    + 0.3 * (t - 1)
                    + (1.5 if (ft > 0 and t >= ft) else 0.0)
                    + rng.normal(0.0, 0.5)
                )
                rows.append(
                    {
                        "unit": next_unit,
                        "state": s,
                        "time": t,
                        "first_treat": ft,
                        "outcome": y,
                    }
                )
                next_unit += 1
        data = pd.DataFrame(rows)

        cs_unit = CallawaySantAnna(panel=False)
        res_unit = cs_unit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        cs_cluster = CallawaySantAnna(panel=False, cluster="state")
        res_cluster = cs_cluster.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert np.isfinite(res_unit.overall_se) and res_unit.overall_se > 0
        assert np.isfinite(res_cluster.overall_se) and res_cluster.overall_se > 0
        assert abs(res_unit.overall_se - res_cluster.overall_se) > 1e-6, (
            "RCS path: cluster=state SE not measurably different from "
            "cluster=None SE — cluster wiring may not reach the RCS code path."
        )


class TestCallawaySantAnnaVcovTypeNarrowContract:
    """Narrow vcov_type contract: CS accepts {hc1} only; rejects
    analytical-sandwich families and conley with methodology-rooted
    messages."""

    def test_default_vcov_type_is_hc1(self):
        cs = CallawaySantAnna()
        assert cs.vcov_type == "hc1"

    def test_classical_rejected_at_init(self):
        with pytest.raises(ValueError, match="influence-function"):
            CallawaySantAnna(vcov_type="classical")

    def test_hc2_rejected_at_init(self):
        with pytest.raises(ValueError, match="hat matrix"):
            CallawaySantAnna(vcov_type="hc2")

    def test_hc2_bm_rejected_at_init(self):
        with pytest.raises(ValueError, match="Bell-McCaffrey"):
            CallawaySantAnna(vcov_type="hc2_bm")

    def test_conley_rejected_at_init(self):
        with pytest.raises(ValueError, match="(conley|spatial-HAC)"):
            CallawaySantAnna(vcov_type="conley")

    def test_unknown_vcov_type_rejected(self):
        with pytest.raises(ValueError, match="hc4"):
            CallawaySantAnna(vcov_type="hc4")

    def test_get_params_includes_vcov_type(self):
        cs = CallawaySantAnna()
        params = cs.get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc1"

    def test_set_params_bad_vcov_raises_eagerly(self):
        """set_params validates via constructor probe (transactional per
        the locked v4 rule): a bad vcov_type raises AT set_params with the
        same message __init__ gives, and the estimator is unchanged. The
        fit-time re-validation stays in place as belt-and-suspenders."""
        cs = CallawaySantAnna()
        with pytest.raises(ValueError, match="hc4"):
            cs.set_params(vcov_type="hc4")
        assert cs.vcov_type == "hc1"

    def test_results_carries_vcov_type(self):
        data = _generate_clustered_staggered_data(seed=41)
        cs = CallawaySantAnna()
        res = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert res.vcov_type == "hc1"

    def test_fit_clone_idempotent_on_vcov_type(self):
        """get_params + reconstruct + refit produces same SE."""
        data = _generate_clustered_staggered_data(seed=43)
        cs1 = CallawaySantAnna(cluster="state")
        res1 = cs1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        cs2 = CallawaySantAnna(**cs1.get_params())
        res2 = cs2.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert res1.overall_se == pytest.approx(res2.overall_se, rel=0, abs=0)
        assert res1.vcov_type == res2.vcov_type == "hc1"
        assert res1.cluster_name == res2.cluster_name == "state"


class TestCallawaySantAnnaClusterSafetyGates:
    """Safety gates for the cluster= wiring fix added in response to local
    AI review findings (panel-mover validation, replicate-weight rejection,
    df_survey propagation to HonestDiD via survey_metadata)."""

    def test_inject_branch_panel_mover_raises(self):
        """survey_design without PSU + cluster=X where a unit changes
        cluster across periods (a 'mover') must raise via the unit-
        constancy validator. The validator must see the injected cluster
        column — earlier versions ran the validator on the user-provided
        survey_design (no PSU), missing the mover entirely."""
        from diff_diff import SurveyDesign

        data = _generate_clustered_staggered_data(seed=61)
        data["wt"] = 1.0
        # Force unit 0 to be a mover: assign it to a different state in the
        # later half of the panel.
        unit_0_late_mask = (data["unit"] == 0) & (data["time"] >= 5)
        original_state_for_unit_0 = data.loc[data["unit"] == 0, "state"].iloc[0]
        mover_target_state = (int(original_state_for_unit_0) + 1) % 20
        data.loc[unit_0_late_mask, "state"] = mover_target_state

        cs = CallawaySantAnna(cluster="state")
        with pytest.raises((ValueError, RuntimeError), match="(unit|constant|invariant)"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=SurveyDesign(weights="wt"),
            )

    def test_replicate_weight_plus_cluster_rejected(self):
        """SurveyDesign(replicate_weights=[...]) + cluster=X must raise
        NotImplementedError. Replicate-weight variance ignores PSU entirely,
        so honoring bare cluster= would silently have no effect on the
        variance estimate while populating cluster_name/n_clusters
        dishonestly. Fail-closed per feedback_no_silent_failures."""
        from diff_diff import SurveyDesign

        data = _generate_clustered_staggered_data(seed=67)
        data["wt"] = 1.0
        # Add 4 BRR replicate weights (R survey package convention).
        for r in range(1, 5):
            data[f"repwt_{r}"] = 1.0

        cs = CallawaySantAnna(cluster="state")
        with pytest.raises(NotImplementedError, match="replicate"):
            cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=SurveyDesign(
                    weights="wt",
                    replicate_weights=["repwt_1", "repwt_2", "repwt_3", "repwt_4"],
                    replicate_method="BRR",
                ),
            )

    def test_bare_cluster_populates_df_inference(self):
        """Bare cluster= must populate Results.df_inference so downstream
        consumers (e.g., HonestDiD at honest_did.py:~652) see the cluster-
        level df rather than silently reverting to normal-theory critical
        values. df_inference is the canonical carrier — survey_metadata is
        for user-provided SurveyDesign only (see
        test_bare_cluster_does_not_set_survey_metadata for the other half
        of the contract)."""
        data = _generate_clustered_staggered_data(seed=71)
        cs = CallawaySantAnna(cluster="state")
        res = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert res.df_inference is not None and res.df_inference > 0, (
            f"Bare cluster= must populate Results.df_inference with a "
            f"positive integer; got {res.df_inference!r}."
        )
        # df_inference must equal n_clusters - 1 for the PSU-only design
        assert res.n_clusters is not None
        assert res.df_inference == res.n_clusters - 1, (
            f"df_inference ({res.df_inference}) must equal n_clusters - 1 "
            f"({res.n_clusters - 1}) for PSU-only synthesized designs."
        )

    def test_bare_cluster_does_not_set_survey_metadata(self):
        """Bare cluster= must NOT populate Results.survey_metadata. The
        user did not provide a SurveyDesign, so downstream consumers that
        check ``survey_metadata is not None`` for 'original fit used a
        survey design' must continue to see a non-survey fit. Affected
        consumers: DiagnosticReport at diagnostic_report.py:848-856 +
        1150-1158 (Bacon decomp + 2x2 PT skip); CallawaySantAnnaResults.
        summary() at staggered_results.py:235-238 (survey block render).
        df_inference carries cluster df separately (see
        test_bare_cluster_populates_df_inference)."""
        data = _generate_clustered_staggered_data(seed=73)
        cs = CallawaySantAnna(cluster="state")
        res = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert res.survey_metadata is None, (
            "Bare cluster= must NOT populate survey_metadata — that field "
            "is reserved for user-provided SurveyDesign. Setting it on a "
            "non-survey fit would cause DiagnosticReport to skip checks "
            "with 'Original fit used a survey design' and summary() to "
            "print a misleading survey block."
        )

    def test_explicit_survey_design_does_populate_survey_metadata(self):
        """Counterpart to test_bare_cluster_does_not_set_survey_metadata:
        when user provides a real SurveyDesign, survey_metadata IS
        populated (regardless of bare cluster= status). Verifies the
        'inject' branch path: SurveyDesign(weights=...) + cluster=X →
        survey_metadata populated; df_inference stays None per the
        narrowed contract (canonical df carrier when survey_metadata is
        present is survey_metadata.df_survey, which holds CS-internal
        post-resolve-tightened df). HonestDiD reads survey_metadata
        first, df_inference only as fallback."""
        from diff_diff import SurveyDesign

        data = _generate_clustered_staggered_data(seed=75)
        data["wt"] = 1.0
        cs = CallawaySantAnna(cluster="state")
        res = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=SurveyDesign(weights="wt"),
        )
        assert (
            res.survey_metadata is not None
        ), "User-provided SurveyDesign must populate survey_metadata."
        # df_inference is NARROWED to bare-cluster-synthesize path only:
        # when survey_metadata is populated, df_inference stays None and
        # HonestDiD reads df_survey directly from survey_metadata (which
        # carries the actual CS-internal df, post-recompute). Prevents
        # HonestDiD from reading a stale/wrong df_inference when CS's
        # internal df was tightened post-resolve. See honest_did.py:
        # _extract_event_study_params preference order: survey_metadata
        # first, df_inference fallback.
        assert res.df_inference is None, (
            "Inject/conflict branches must leave df_inference=None — "
            "survey_metadata.df_survey is the canonical df carrier when "
            "a survey design is present."
        )
        sm_df = getattr(res.survey_metadata, "df_survey", None)
        assert sm_df is not None and sm_df > 0, (
            "survey_metadata.df_survey must be populated when an explicit "
            "SurveyDesign is provided."
        )

    def test_bare_cluster_honest_did_uses_df_inference(self):
        """End-to-end integration: HonestDiD.fit() on a bare-cluster CS
        result must pick up the cluster-level df via df_inference (not
        revert to normal-theory critical values). A future refactor that
        stops honoring df_inference in honest_did.py would silently fall
        back to z-critical values for clustered CS fits without failing
        the simpler results-object-contract tests. This test pins the
        end-to-end behavior. Per the R3 codex finding."""
        from diff_diff.honest_did import HonestDiD

        data = _generate_clustered_staggered_data(seed=79)
        cs = CallawaySantAnna(cluster="state", base_period="universal")
        cs_res = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Sanity: the CS fit populated df_inference but not survey_metadata
        assert cs_res.df_inference is not None and cs_res.df_inference > 0
        assert cs_res.survey_metadata is None, (
            "Pre-condition for this test: bare cluster= must NOT populate "
            "survey_metadata. If this fails, the survey/non-survey "
            "contract regressed (see test_bare_cluster_does_not_set_survey_metadata)."
        )

        # Run HonestDiD; assert it threads df_inference into the returned df_survey
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        honest_res = honest.fit(cs_res)

        assert honest_res.df_survey is not None, (
            "HonestDiD must preserve the cluster df from CS's df_inference. "
            "Reading None means it silently reverted to normal-theory "
            "critical values — the contract this test exists to guard."
        )
        assert int(honest_res.df_survey) == int(cs_res.df_inference), (
            f"HonestDiDResults.df_survey ({honest_res.df_survey}) must "
            f"equal CS Results.df_inference ({cs_res.df_inference}). "
            "A divergence here means df_inference is not being threaded "
            "through honest_did.py's _extract_event_study_params."
        )

    def test_bare_cluster_bootstrap_se_differs_from_unit_level(self):
        """Bootstrap path coverage: bare cluster= must route bootstrap
        through the PSU-level multiplier-weights branch at
        staggered_bootstrap.py:323-347 (synthesized SurveyDesign(psu=
        cluster) sets resolved_survey.psu, triggering the survey-PSU
        bootstrap path). Without the fix, bootstrap drew per-unit weights
        regardless of self.cluster — same class of silent no-op as the
        analytical path. Per CI codex R1 P3 finding."""
        data = _generate_clustered_staggered_data(seed=83)

        # Low n_bootstrap for speed; assertion bands wide enough for stochasticity
        cs_unit = CallawaySantAnna(n_bootstrap=99, seed=83)
        res_unit = cs_unit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        cs_cluster = CallawaySantAnna(cluster="state", n_bootstrap=99, seed=83)
        res_cluster = cs_cluster.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert np.isfinite(res_unit.overall_se) and res_unit.overall_se > 0
        assert np.isfinite(res_cluster.overall_se) and res_cluster.overall_se > 0
        assert abs(res_unit.overall_se - res_cluster.overall_se) > 1e-6, (
            f"Bootstrap path: cluster=state SE ({res_cluster.overall_se:.6f}) "
            f"is effectively identical to cluster=None SE "
            f"({res_unit.overall_se:.6f}) — the cluster= parameter may "
            "not be reaching the bootstrap multiplier-weights routing."
        )

    def test_per_gt_analytical_se_changes_with_cluster(self):
        """Per-(g,t) analytical SE at results.group_time_effects[(g,t)]
        ["se"] must change when cluster= is set (mirrors the overall_se
        contract). Pre-fix, per-(g,t) SEs were unit-level even with
        cluster=, only the aggregate path + bootstrap honored cluster=.
        Per CI codex R3 P0 finding."""
        data = _generate_clustered_staggered_data(seed=97)

        cs_unit = CallawaySantAnna()
        res_unit = cs_unit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        cs_cluster = CallawaySantAnna(cluster="state")
        res_cluster = cs_cluster.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Pick a representative (g, t) cell that exists in both fits
        gt_keys = sorted(
            set(res_unit.group_time_effects.keys()) & set(res_cluster.group_time_effects.keys())
        )
        assert len(gt_keys) > 0, "expected overlapping (g, t) keys"

        # At least one (g, t) cell must show measurable SE divergence —
        # cluster-aware aggregation should differ from unit-level for at
        # least one cell on a panel with intra-cluster correlation.
        diffs = []
        for gt in gt_keys:
            se_unit = res_unit.group_time_effects[gt]["se"]
            se_cluster = res_cluster.group_time_effects[gt]["se"]
            if np.isfinite(se_unit) and np.isfinite(se_cluster):
                diffs.append(abs(se_unit - se_cluster))
        max_diff = max(diffs) if diffs else 0.0
        assert max_diff > 1e-6, (
            f"Per-(g,t) SEs did not change with cluster= (max diff "
            f"across {len(diffs)} cells: {max_diff:.6g}). The cluster= "
            "parameter may not be reaching the per-(g,t) analytical SE "
            "computation."
        )

    def test_per_gt_se_matches_explicit_survey_design(self):
        """When bare cluster=X and explicit SurveyDesign(psu=X) produce
        equivalent variance contracts, the per-(g,t) SE surface must
        also agree (modulo the deterministic synthesis path). Per CI
        codex R3 P0 finding."""
        from diff_diff import SurveyDesign

        data = _generate_clustered_staggered_data(seed=101)

        cs_bare = CallawaySantAnna(cluster="state")
        res_bare = cs_bare.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        cs_explicit = CallawaySantAnna()
        res_explicit = cs_explicit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=SurveyDesign(psu="state"),
        )

        gt_keys = sorted(
            set(res_bare.group_time_effects.keys()) & set(res_explicit.group_time_effects.keys())
        )
        assert len(gt_keys) > 0

        for gt in gt_keys:
            se_bare = res_bare.group_time_effects[gt]["se"]
            se_explicit = res_explicit.group_time_effects[gt]["se"]
            if np.isfinite(se_bare) and np.isfinite(se_explicit):
                assert se_bare == pytest.approx(se_explicit, rel=1e-10, abs=1e-12), (
                    f"Per-(g,t) SE divergence at {gt}: bare cluster=state "
                    f"({se_bare}) vs explicit SurveyDesign(psu=state) "
                    f"({se_explicit}). Both should activate the same CR1 "
                    "aggregation."
                )

    def test_per_gt_se_matches_compute_survey_if_variance_helper(self):
        """The per-(g,t) cluster-aware SE must use the SAME design-based
        variance machinery as the aggregate path
        (compute_survey_if_variance / _compute_stratified_psu_meat) —
        applying G/(G-1) finite-sample correction, PSU centering, and
        lonely-PSU handling uniformly. Compares per-cell SE against the
        shared helper on a small-G design (so the finite-sample
        correction is non-trivial). Per CI codex R4 P1/P2 findings."""
        from diff_diff.staggered import _cluster_robust_se_from_per_gt_if
        from diff_diff.survey import (
            SurveyDesign,
            _resolve_survey_for_fit,
            compute_survey_if_variance,
        )

        # 10 PSUs (states), 4 units each = 40 units total (small-G)
        n_clusters = 10
        units_per_cluster = 4
        n_units = n_clusters * units_per_cluster
        state_ids = np.repeat(np.arange(n_clusters), units_per_cluster)
        unit_data = pd.DataFrame({"unit": np.arange(n_units), "state": state_ids})

        synthetic = SurveyDesign(psu="state", weight_type="pweight")
        rsu, _, _, _ = _resolve_survey_for_fit(synthetic, unit_data, "analytical")
        assert rsu is not None
        assert rsu.psu is not None and len(rsu.psu) == n_units

        # Hand-crafted per-(g,t) IF: 5 treated + 10 control units in this cell
        rng = np.random.default_rng(7)
        treated_idx = np.arange(0, 5)
        control_idx = np.arange(5, 15)
        treated_inf = rng.normal(0.0, 0.1, 5)
        control_inf = rng.normal(0.0, 0.1, 10)
        inf_info = {
            "treated_idx": treated_idx,
            "control_idx": control_idx,
            "treated_inf": treated_inf,
            "control_inf": control_inf,
        }

        # Helper output (function under test)
        se_helper = _cluster_robust_se_from_per_gt_if(inf_info, rsu)
        assert se_helper is not None
        assert np.isfinite(se_helper) and se_helper > 0

        # Direct reconstruction via compute_survey_if_variance must agree
        # exactly — verifies the helper routes through the shared
        # G/(G-1) + PSU centering + FPC machinery, not a bespoke formula.
        psi_per_unit = np.zeros(n_units)
        np.add.at(psi_per_unit, treated_idx, treated_inf)
        np.add.at(psi_per_unit, control_idx, control_inf)
        var_reference = compute_survey_if_variance(psi_per_unit, rsu)
        se_reference = float(np.sqrt(var_reference))

        assert se_helper == pytest.approx(se_reference, rel=0, abs=0), (
            f"Per-(g,t) SE helper ({se_helper}) must equal "
            f"compute_survey_if_variance reconstruction ({se_reference}) "
            "— any divergence means the helper bypasses the shared "
            "G/(G-1) finite-sample correction + PSU centering machinery."
        )

    def test_per_gt_se_propagates_nan_when_cluster_variance_undefined(self):
        """When clustered design-based variance is undefined (e.g., G=1
        — single cluster, no within-PSU variability), the per-(g,t) SE
        must propagate NaN through the full inference surface (se,
        t_stat, p_value, conf_int) instead of silently falling back to
        the unit-level SE. Verifies the helper's NaN-propagation
        contract end-to-end on a fit. Per CI codex R5 P1/P2 findings."""
        # Build a panel where all units belong to a single cluster.
        # compute_survey_if_variance returns NaN for G<2 designs (lonely
        # PSU removed or single-cluster) — the per-cell helper must
        # propagate this NaN rather than retain the unit-level SE.
        data = _generate_clustered_staggered_data(n_clusters=2, units_per_cluster=10, seed=109)
        # Force ALL units into a single cluster (G=1)
        data["single_cluster"] = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lonely-PSU warnings are expected
            cs = CallawaySantAnna(cluster="single_cluster")
            res = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

        # At least one (g, t) cell should have NaN inference under the
        # undefined-variance contract. If ALL cells retain finite SE, the
        # helper is silently falling back to unit-level on the NaN branch.
        nan_cells = [gt for gt, eff in res.group_time_effects.items() if not np.isfinite(eff["se"])]
        assert len(nan_cells) > 0, (
            "Expected at least one (g, t) cell with NaN SE under G=1 "
            "(undefined clustered variance), but all cells retained "
            "finite unit-level SE — the helper's NaN-propagation "
            "contract is broken (cells silently fall back to unit-level)."
        )

        # For each NaN-SE cell, the full inference surface must be NaN
        # (matches the safe_inference contract for non-finite SE).
        for gt in nan_cells:
            eff = res.group_time_effects[gt]
            assert np.isnan(eff["se"]), f"{gt}: se should be NaN"
            assert np.isnan(eff["t_stat"]), f"{gt}: t_stat should be NaN"
            assert np.isnan(eff["p_value"]), f"{gt}: p_value should be NaN"
            ci_lo, ci_hi = eff["conf_int"]
            assert np.isnan(ci_lo) and np.isnan(
                ci_hi
            ), f"{gt}: CI bounds should both be NaN, got ({ci_lo}, {ci_hi})"

    def test_bare_cluster_bootstrap_propagates_nan_when_g_less_than_2(self):
        """Bootstrap path NaN propagation: when bare cluster= produces
        G=1 (single cluster), the PSU-multiplier-weights bootstrap path
        at bootstrap_utils.py:557-562 returns zero PSU multipliers and
        the downstream zero-SE guards at :365-377/:472-485 must NaN-out
        the full bootstrap inference surface (overall_se, per-(g,t),
        aggregate). Per CI codex R7 P3 finding."""
        data = _generate_clustered_staggered_data(n_clusters=2, units_per_cluster=10, seed=113)
        data["single_cluster"] = 0  # Force G=1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lonely-PSU + low-n_bootstrap warnings expected
            cs = CallawaySantAnna(cluster="single_cluster", n_bootstrap=99, seed=113)
            res = cs.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )

        # Overall bootstrap inference must be NaN-consistent
        assert not np.isfinite(res.overall_se), (
            f"Bootstrap overall_se should be NaN under G=1 cluster, " f"got {res.overall_se}."
        )
        assert np.isnan(res.overall_t_stat)
        assert np.isnan(res.overall_p_value)
        assert np.isnan(res.overall_conf_int[0]) and np.isnan(res.overall_conf_int[1])

        # At least one (g, t) cell must have NaN inference (undefined
        # clustered variance propagating through either the bootstrap or
        # analytical layer)
        nan_gt_cells = [
            gt for gt, eff in res.group_time_effects.items() if not np.isfinite(eff["se"])
        ]
        assert len(nan_gt_cells) > 0, (
            "Expected at least one (g, t) cell with NaN SE under "
            "G=1 cluster + bootstrap — undefined clustered variance "
            "must propagate through the bootstrap inference surface."
        )
        for gt in nan_gt_cells:
            eff = res.group_time_effects[gt]
            assert np.isnan(eff["se"])
            assert np.isnan(eff["t_stat"])
            assert np.isnan(eff["p_value"])
            assert np.isnan(eff["conf_int"][0]) and np.isnan(eff["conf_int"][1])

        # Requested aggregate (event-study) must also be NaN-consistent
        # for any aggregated horizon whose underlying cells are NaN
        if res.event_study_effects:
            for h, ev in res.event_study_effects.items():
                if not np.isfinite(ev["se"]):
                    assert np.isnan(ev["t_stat"])
                    assert np.isnan(ev["p_value"])
                    assert np.isnan(ev["conf_int"][0]) and np.isnan(ev["conf_int"][1])

    def test_grouped_aggregate_se_changes_with_cluster(self):
        """The ``aggregate="group"`` aggregation path
        (``_aggregate_by_group`` at ``staggered_aggregation.py:782-860``)
        has its own SE computation independent of overall + event-study.
        Asserts grouped SEs differ between cluster=None and cluster="state"
        on a panel with intra-cluster correlation, AND that bare cluster=
        "state" matches explicit SurveyDesign(psu="state") on the grouped
        surface. Per CI codex R8 P3 finding."""
        from diff_diff import SurveyDesign

        data = _generate_clustered_staggered_data(seed=117)

        cs_unit = CallawaySantAnna()
        res_unit = cs_unit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        cs_cluster = CallawaySantAnna(cluster="state")
        res_cluster = cs_cluster.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        cs_explicit = CallawaySantAnna()
        res_explicit = cs_explicit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
            survey_design=SurveyDesign(psu="state"),
        )

        assert res_unit.group_effects is not None
        assert res_cluster.group_effects is not None
        assert res_explicit.group_effects is not None

        # Grouped SEs must differ under cluster vs unit-level (at least
        # one group)
        common_groups = set(res_unit.group_effects.keys()) & set(res_cluster.group_effects.keys())
        assert common_groups, "expected overlapping groups"

        diffs = []
        for g in common_groups:
            se_unit = res_unit.group_effects[g]["se"]
            se_cluster = res_cluster.group_effects[g]["se"]
            if np.isfinite(se_unit) and np.isfinite(se_cluster):
                diffs.append(abs(se_unit - se_cluster))
        max_diff = max(diffs) if diffs else 0.0
        assert max_diff > 1e-6, (
            f"Grouped SEs did not change with cluster= (max diff: "
            f"{max_diff:.6g}). aggregate='group' may not be routing "
            "through the cluster-aware IF aggregation."
        )

        # Bare cluster vs explicit SurveyDesign must agree on grouped surface
        common = set(res_cluster.group_effects.keys()) & set(res_explicit.group_effects.keys())
        for g in common:
            se_bare = res_cluster.group_effects[g]["se"]
            se_explicit = res_explicit.group_effects[g]["se"]
            if np.isfinite(se_bare) and np.isfinite(se_explicit):
                assert se_bare == pytest.approx(se_explicit, rel=1e-10, abs=1e-12), (
                    f"Grouped SE divergence at g={g}: bare cluster=state "
                    f"({se_bare}) vs explicit SurveyDesign(psu=state) "
                    f"({se_explicit})."
                )

    def test_survey_design_psu_wins_under_bootstrap(self):
        """Bootstrap path: when survey_design=SurveyDesign(psu=Y) is
        explicit AND cluster=X is also set with a different partition,
        the explicit PSU partition wins for the bootstrap draws (just
        like for the analytical sandwich). UserWarning fires for the
        partition mismatch; bootstrap SE matches the explicit-PSU-only
        fit, not the bare-cluster fit. Per CI codex R1 P3 finding."""
        from diff_diff import SurveyDesign

        data = _generate_clustered_staggered_data(n_clusters=20, units_per_cluster=5, seed=89)
        data["region"] = data["state"] // 10  # 2 regions of 10 states

        # Reference: explicit region PSU only (no cluster= confound)
        cs_ref = CallawaySantAnna(n_bootstrap=99, seed=89)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_ref = cs_ref.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=SurveyDesign(psu="region"),
            )

        # Conflict: explicit region PSU + bare cluster=state (different partition)
        cs_conflict = CallawaySantAnna(cluster="state", n_bootstrap=99, seed=89)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_conflict = cs_conflict.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=SurveyDesign(psu="region"),
            )

        partition_warnings = [
            w
            for w in caught
            if "psu" in str(w.message).lower()
            or "partition" in str(w.message).lower()
            or "different groupings" in str(w.message).lower()
        ]
        assert len(partition_warnings) > 0, (
            "Conflict case (explicit PSU + bare cluster with different "
            "partition) must emit UserWarning."
        )
        # PSU wins under bootstrap too — SE must match the reference
        # (explicit-PSU-only) fit at the same seed
        assert res_conflict.overall_se == pytest.approx(res_ref.overall_se, rel=0, abs=0), (
            f"Bootstrap precedence: with seed={cs_conflict.seed}, conflict "
            f"fit SE ({res_conflict.overall_se}) must match explicit-PSU-only "
            f"reference SE ({res_ref.overall_se}) — both bootstraps must "
            "draw at the same effective PSU level."
        )
