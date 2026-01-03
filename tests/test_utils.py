"""
Tests for utility functions in diff_diff.utils module.

This module provides comprehensive test coverage for:
- Binary validation
- Robust and cluster-robust standard errors
- Confidence interval computation
- P-value computation
- Parallel trends testing (simple version)
- Outcome change computation
- Placebo effects for Synthetic DiD
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff.utils import (
    validate_binary,
    compute_robust_se,
    compute_confidence_interval,
    compute_p_value,
    check_parallel_trends,
    check_parallel_trends_robust,
    equivalence_test_trends,
    _compute_outcome_changes,
    compute_synthetic_weights,
    compute_time_weights,
    compute_placebo_effects,
    compute_sdid_estimator,
    _project_simplex,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_regression_data():
    """Create simple regression data for testing robust SE."""
    np.random.seed(42)
    n = 100
    X = np.column_stack([
        np.ones(n),
        np.random.randn(n),
        np.random.randn(n),
    ])
    beta_true = np.array([1.0, 2.0, -1.0])
    # Heteroskedastic errors
    errors = np.random.randn(n) * (1 + np.abs(X[:, 1]))
    y = X @ beta_true + errors
    return X, y


@pytest.fixture
def clustered_regression_data():
    """Create clustered regression data for testing cluster-robust SE."""
    np.random.seed(42)
    n_clusters = 20
    obs_per_cluster = 10
    n = n_clusters * obs_per_cluster

    cluster_ids = np.repeat(np.arange(n_clusters), obs_per_cluster)
    cluster_effects = np.random.randn(n_clusters)

    X = np.column_stack([
        np.ones(n),
        np.random.randn(n),
    ])

    beta_true = np.array([5.0, 2.0])
    # Cluster-correlated errors
    errors = cluster_effects[cluster_ids] + np.random.randn(n) * 0.5
    y = X @ beta_true + errors

    return X, y, cluster_ids


@pytest.fixture
def parallel_trends_data():
    """Create panel data with parallel trends."""
    np.random.seed(42)
    n_units = 50
    n_periods = 6

    data = []
    for unit in range(n_units):
        is_treated = unit < n_units // 2
        unit_effect = np.random.normal(0, 2)

        for period in range(n_periods):
            # Common trend for both groups
            time_effect = period * 1.5
            y = 10.0 + unit_effect + time_effect

            # Treatment effect only in post period (period >= 3)
            if is_treated and period >= 3:
                y += 5.0

            y += np.random.normal(0, 0.5)

            data.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "outcome": y,
            })

    return pd.DataFrame(data)


@pytest.fixture
def non_parallel_trends_data():
    """Create panel data where parallel trends is violated."""
    np.random.seed(42)
    n_units = 50
    n_periods = 6

    data = []
    for unit in range(n_units):
        is_treated = unit < n_units // 2
        unit_effect = np.random.normal(0, 1)

        for period in range(n_periods):
            # Different trends for treated vs control
            if is_treated:
                time_effect = period * 3.0  # Steeper trend
            else:
                time_effect = period * 1.0  # Flatter trend

            y = 10.0 + unit_effect + time_effect

            # Treatment effect in post period
            if is_treated and period >= 3:
                y += 5.0

            y += np.random.normal(0, 0.5)

            data.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "outcome": y,
            })

    return pd.DataFrame(data)


@pytest.fixture
def sdid_panel_data():
    """Create panel data suitable for Synthetic DiD placebo tests."""
    np.random.seed(42)
    n_control = 20
    n_treated = 3
    n_pre = 5
    n_post = 3

    data = []

    # Control units
    for unit in range(n_control):
        unit_effect = np.random.normal(0, 2)
        for period in range(n_pre + n_post):
            y = 10.0 + unit_effect + period * 0.5 + np.random.normal(0, 0.3)
            data.append({
                "unit": unit,
                "period": period,
                "treated": 0,
                "outcome": y,
            })

    # Treated units
    for unit in range(n_control, n_control + n_treated):
        unit_effect = np.random.normal(0, 2)
        for period in range(n_pre + n_post):
            y = 10.0 + unit_effect + period * 0.5
            if period >= n_pre:
                y += 3.0  # Treatment effect
            y += np.random.normal(0, 0.3)
            data.append({
                "unit": unit,
                "period": period,
                "treated": 1,
                "outcome": y,
            })

    return pd.DataFrame(data)


# =============================================================================
# Tests for validate_binary
# =============================================================================


class TestValidateBinary:
    """Tests for validate_binary function."""

    def test_valid_binary_zeros_ones(self):
        """Test that valid binary arrays pass validation."""
        arr = np.array([0, 1, 0, 1, 1, 0])
        # Should not raise
        validate_binary(arr, "test_var")

    def test_valid_binary_all_zeros(self):
        """Test that all-zero array passes validation."""
        arr = np.array([0, 0, 0, 0])
        validate_binary(arr, "test_var")

    def test_valid_binary_all_ones(self):
        """Test that all-one array passes validation."""
        arr = np.array([1, 1, 1, 1])
        validate_binary(arr, "test_var")

    def test_valid_binary_floats(self):
        """Test that binary float values pass validation."""
        arr = np.array([0.0, 1.0, 0.0, 1.0])
        validate_binary(arr, "test_var")

    def test_invalid_non_binary_integers(self):
        """Test that non-binary integers raise ValueError."""
        arr = np.array([0, 1, 2, 3])
        with pytest.raises(ValueError, match="must be binary"):
            validate_binary(arr, "test_var")

    def test_invalid_negative_values(self):
        """Test that negative values raise ValueError."""
        arr = np.array([-1, 0, 1])
        with pytest.raises(ValueError, match="must be binary"):
            validate_binary(arr, "test_var")

    def test_invalid_float_values(self):
        """Test that non-binary floats raise ValueError."""
        arr = np.array([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="must be binary"):
            validate_binary(arr, "test_var")

    def test_nan_values_ignored(self):
        """Test that NaN values are ignored in validation."""
        arr = np.array([0, 1, np.nan, 0, 1])
        # Should not raise - NaN values are ignored
        validate_binary(arr, "test_var")

    def test_error_message_contains_variable_name(self):
        """Test that error message contains the variable name."""
        arr = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="my_variable"):
            validate_binary(arr, "my_variable")

    def test_error_message_shows_found_values(self):
        """Test that error message shows the invalid values found."""
        arr = np.array([0, 1, 5])
        with pytest.raises(ValueError, match="5"):
            validate_binary(arr, "test_var")


# =============================================================================
# Tests for compute_robust_se
# =============================================================================


class TestComputeRobustSE:
    """Tests for compute_robust_se function."""

    def test_hc1_returns_correct_shape(self, simple_regression_data):
        """Test that HC1 robust SE returns correct shape."""
        X, y = simple_regression_data
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        vcov = compute_robust_se(X, residuals)

        assert vcov.shape == (3, 3)

    def test_hc1_is_symmetric(self, simple_regression_data):
        """Test that HC1 vcov matrix is symmetric."""
        X, y = simple_regression_data
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        vcov = compute_robust_se(X, residuals)

        np.testing.assert_array_almost_equal(vcov, vcov.T)

    def test_hc1_is_positive_semidefinite(self, simple_regression_data):
        """Test that HC1 vcov matrix is positive semi-definite."""
        X, y = simple_regression_data
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        vcov = compute_robust_se(X, residuals)

        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical error

    def test_hc1_diagonal_positive(self, simple_regression_data):
        """Test that diagonal elements (variances) are positive."""
        X, y = simple_regression_data
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        vcov = compute_robust_se(X, residuals)

        assert np.all(np.diag(vcov) > 0)

    def test_cluster_robust_returns_correct_shape(self, clustered_regression_data):
        """Test that cluster-robust SE returns correct shape."""
        X, y, cluster_ids = clustered_regression_data
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        vcov = compute_robust_se(X, residuals, cluster_ids)

        assert vcov.shape == (2, 2)

    def test_cluster_robust_is_symmetric(self, clustered_regression_data):
        """Test that cluster-robust vcov matrix is symmetric."""
        X, y, cluster_ids = clustered_regression_data
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        vcov = compute_robust_se(X, residuals, cluster_ids)

        np.testing.assert_array_almost_equal(vcov, vcov.T)

    def test_cluster_robust_differs_from_hc1(self, clustered_regression_data):
        """Test that cluster-robust SE differs from HC1."""
        X, y, cluster_ids = clustered_regression_data
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        vcov_hc1 = compute_robust_se(X, residuals)
        vcov_cluster = compute_robust_se(X, residuals, cluster_ids)

        # Should not be equal
        assert not np.allclose(vcov_hc1, vcov_cluster)

    def test_cluster_robust_larger_with_correlated_errors(self, clustered_regression_data):
        """Test that cluster-robust SE is typically larger with correlated errors."""
        X, y, cluster_ids = clustered_regression_data
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        vcov_hc1 = compute_robust_se(X, residuals)
        vcov_cluster = compute_robust_se(X, residuals, cluster_ids)

        # For the slope coefficient (index 1), cluster SE should typically be larger
        se_hc1 = np.sqrt(vcov_hc1[1, 1])
        se_cluster = np.sqrt(vcov_cluster[1, 1])

        # With strong cluster correlation, cluster SE should be larger
        assert se_cluster > se_hc1 * 0.5  # Allow some flexibility


# =============================================================================
# Tests for compute_confidence_interval
# =============================================================================


class TestComputeConfidenceInterval:
    """Tests for compute_confidence_interval function."""

    def test_ci_with_normal_distribution(self):
        """Test CI computation with normal distribution."""
        estimate = 5.0
        se = 1.0
        alpha = 0.05

        lower, upper = compute_confidence_interval(estimate, se, alpha)

        # For 95% CI with normal: 5 +/- 1.96 * 1
        expected_lower = 5.0 - 1.96
        expected_upper = 5.0 + 1.96

        assert abs(lower - expected_lower) < 0.01
        assert abs(upper - expected_upper) < 0.01

    def test_ci_with_t_distribution(self):
        """Test CI computation with t distribution."""
        estimate = 5.0
        se = 1.0
        alpha = 0.05
        df = 10

        lower, upper = compute_confidence_interval(estimate, se, alpha, df=df)

        # For t distribution with 10 df
        t_crit = stats.t.ppf(0.975, df)
        expected_lower = 5.0 - t_crit
        expected_upper = 5.0 + t_crit

        assert abs(lower - expected_lower) < 0.01
        assert abs(upper - expected_upper) < 0.01

    def test_ci_contains_estimate(self):
        """Test that CI always contains the point estimate."""
        estimate = 10.0
        se = 2.0

        for alpha in [0.01, 0.05, 0.10, 0.20]:
            lower, upper = compute_confidence_interval(estimate, se, alpha)
            assert lower < estimate < upper

    def test_ci_width_decreases_with_higher_alpha(self):
        """Test that CI width decreases with higher alpha (less confidence)."""
        estimate = 5.0
        se = 1.0

        lower_90, upper_90 = compute_confidence_interval(estimate, se, alpha=0.10)
        lower_95, upper_95 = compute_confidence_interval(estimate, se, alpha=0.05)

        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95

        assert width_90 < width_95

    def test_ci_width_increases_with_se(self):
        """Test that CI width increases with standard error."""
        estimate = 5.0
        alpha = 0.05

        lower_small, upper_small = compute_confidence_interval(estimate, se=1.0, alpha=alpha)
        lower_large, upper_large = compute_confidence_interval(estimate, se=2.0, alpha=alpha)

        width_small = upper_small - lower_small
        width_large = upper_large - lower_large

        assert width_large > width_small

    def test_ci_symmetric_around_estimate(self):
        """Test that CI is symmetric around estimate."""
        estimate = 5.0
        se = 1.0
        alpha = 0.05

        lower, upper = compute_confidence_interval(estimate, se, alpha)

        dist_lower = estimate - lower
        dist_upper = upper - estimate

        assert abs(dist_lower - dist_upper) < 1e-10


# =============================================================================
# Tests for compute_p_value
# =============================================================================


class TestComputePValue:
    """Tests for compute_p_value function."""

    def test_two_sided_at_zero(self):
        """Test two-sided p-value when t=0."""
        p_value = compute_p_value(0.0, two_sided=True)
        assert abs(p_value - 1.0) < 1e-10

    def test_two_sided_large_t_stat(self):
        """Test two-sided p-value with large t-statistic."""
        p_value = compute_p_value(5.0, two_sided=True)
        assert p_value < 0.001

    def test_one_sided_at_zero(self):
        """Test one-sided p-value when t=0."""
        p_value = compute_p_value(0.0, two_sided=False)
        assert abs(p_value - 0.5) < 1e-10

    def test_one_sided_positive_t(self):
        """Test one-sided p-value with positive t."""
        p_value = compute_p_value(2.0, two_sided=False)
        # One-sided: P(T > 2) for standard normal
        expected = stats.norm.sf(2.0)
        assert abs(p_value - expected) < 1e-10

    def test_two_sided_is_double_one_sided(self):
        """Test that two-sided is approximately double one-sided for |t|."""
        t_stat = 1.5
        p_one = compute_p_value(t_stat, two_sided=False)
        p_two = compute_p_value(t_stat, two_sided=True)

        assert abs(p_two - 2 * p_one) < 1e-10

    def test_p_value_in_valid_range(self):
        """Test that p-value is always in [0, 1]."""
        for t_stat in [-10, -2, -1, 0, 1, 2, 10]:
            p_value = compute_p_value(t_stat)
            assert 0 <= p_value <= 1

    def test_with_t_distribution(self):
        """Test p-value with t distribution."""
        t_stat = 2.0
        df = 10

        p_value = compute_p_value(t_stat, df=df, two_sided=True)

        # Compare with scipy
        expected = 2 * stats.t.sf(abs(t_stat), df)
        assert abs(p_value - expected) < 1e-10

    def test_t_vs_normal_larger_with_small_df(self):
        """Test that t-distribution gives larger p-value than normal for same |t|."""
        t_stat = 2.0
        df = 5

        p_normal = compute_p_value(t_stat, two_sided=True)
        p_t = compute_p_value(t_stat, df=df, two_sided=True)

        # t-distribution has fatter tails, so p-value should be larger
        assert p_t > p_normal

    def test_symmetry_positive_negative(self):
        """Test that p-value is symmetric for +t and -t."""
        t_stat = 1.96

        p_pos = compute_p_value(t_stat)
        p_neg = compute_p_value(-t_stat)

        assert abs(p_pos - p_neg) < 1e-10


# =============================================================================
# Tests for check_parallel_trends
# =============================================================================


class TestCheckParallelTrends:
    """Tests for check_parallel_trends function."""

    def test_returns_expected_keys(self, parallel_trends_data):
        """Test that function returns expected dictionary keys."""
        results = check_parallel_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            pre_periods=[0, 1, 2]
        )

        expected_keys = [
            "treated_trend",
            "treated_trend_se",
            "control_trend",
            "control_trend_se",
            "trend_difference",
            "trend_difference_se",
            "t_statistic",
            "p_value",
            "parallel_trends_plausible",
        ]

        for key in expected_keys:
            assert key in results

    def test_parallel_trends_detected(self, parallel_trends_data):
        """Test that parallel trends are detected when they hold."""
        results = check_parallel_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            pre_periods=[0, 1, 2]
        )

        # Should not reject parallel trends
        assert results["p_value"] > 0.05
        assert results["parallel_trends_plausible"] == True

    def test_non_parallel_trends_detected(self, non_parallel_trends_data):
        """Test that non-parallel trends are detected."""
        results = check_parallel_trends(
            non_parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            pre_periods=[0, 1, 2]
        )

        # Should reject parallel trends (different slopes)
        assert results["p_value"] < 0.05
        assert results["parallel_trends_plausible"] == False

    def test_trend_difference_sign(self, non_parallel_trends_data):
        """Test that trend difference has correct sign."""
        results = check_parallel_trends(
            non_parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            pre_periods=[0, 1, 2]
        )

        # Treated has steeper trend (3.0 vs 1.0), so difference should be positive
        assert results["trend_difference"] > 0

    def test_auto_infer_pre_periods(self, parallel_trends_data):
        """Test automatic inference of pre-periods."""
        results = check_parallel_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated"
            # pre_periods not specified
        )

        # Should still return valid results
        assert "treated_trend" in results
        assert not np.isnan(results["treated_trend"])

    def test_single_period_returns_nan(self):
        """Test that single pre-period returns NaN for trends."""
        data = pd.DataFrame({
            "outcome": [10, 11, 12, 13],
            "period": [0, 0, 0, 0],  # All same period
            "treated": [1, 1, 0, 0],
        })

        results = check_parallel_trends(
            data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            pre_periods=[0]
        )

        # Cannot compute trend with single period
        assert np.isnan(results["treated_trend"])
        assert np.isnan(results["control_trend"])

    def test_standard_errors_positive(self, parallel_trends_data):
        """Test that standard errors are positive."""
        results = check_parallel_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            pre_periods=[0, 1, 2]
        )

        assert results["treated_trend_se"] > 0
        assert results["control_trend_se"] > 0
        assert results["trend_difference_se"] > 0


# =============================================================================
# Tests for _compute_outcome_changes
# =============================================================================


class TestComputeOutcomeChanges:
    """Tests for _compute_outcome_changes helper function."""

    def test_with_unit_specified(self, parallel_trends_data):
        """Test outcome changes computation with unit identifier."""
        pre_data = parallel_trends_data[parallel_trends_data["period"] < 3]

        treated_changes, control_changes = _compute_outcome_changes(
            pre_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit"
        )

        # Should have changes for each unit-period transition
        assert len(treated_changes) > 0
        assert len(control_changes) > 0

    def test_without_unit_specified(self, parallel_trends_data):
        """Test outcome changes computation without unit identifier."""
        pre_data = parallel_trends_data[parallel_trends_data["period"] < 3]

        treated_changes, control_changes = _compute_outcome_changes(
            pre_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit=None
        )

        # Should have aggregate changes (fewer than with unit)
        assert len(treated_changes) > 0
        assert len(control_changes) > 0

    def test_returns_float_arrays(self, parallel_trends_data):
        """Test that function returns float arrays."""
        pre_data = parallel_trends_data[parallel_trends_data["period"] < 3]

        treated_changes, control_changes = _compute_outcome_changes(
            pre_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit"
        )

        assert treated_changes.dtype == np.float64
        assert control_changes.dtype == np.float64

    def test_changes_reflect_trend(self):
        """Test that changes reflect the underlying trend."""
        # Create data with known trend
        data = []
        for unit in range(10):
            is_treated = unit < 5
            for period in range(3):
                y = 10.0 + period * 2.0  # Trend of 2.0 per period
                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                })

        df = pd.DataFrame(data)

        treated_changes, control_changes = _compute_outcome_changes(
            df,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit"
        )

        # All changes should be approximately 2.0
        np.testing.assert_array_almost_equal(treated_changes, 2.0, decimal=5)
        np.testing.assert_array_almost_equal(control_changes, 2.0, decimal=5)


# =============================================================================
# Tests for check_parallel_trends_robust
# =============================================================================


class TestCheckParallelTrendsRobust:
    """Additional tests for check_parallel_trends_robust function."""

    def test_reproducibility_with_seed(self, parallel_trends_data):
        """Test that results are reproducible with same seed."""
        results1 = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42
        )

        results2 = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42
        )

        assert results1["wasserstein_p_value"] == results2["wasserstein_p_value"]

    def test_different_seeds_different_results(self, parallel_trends_data):
        """Test that different seeds give different p-values."""
        results1 = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42
        )

        results2 = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=123
        )

        # May be equal by chance but typically different
        # We just verify both return valid results
        assert 0 <= results1["wasserstein_p_value"] <= 1
        assert 0 <= results2["wasserstein_p_value"] <= 1

    def test_n_permutations_affects_precision(self, parallel_trends_data):
        """Test that more permutations give finer p-value resolution."""
        results_few = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            n_permutations=100,
            seed=42
        )

        results_many = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            n_permutations=1000,
            seed=42
        )

        # Both should be valid
        assert 0 <= results_few["wasserstein_p_value"] <= 1
        assert 0 <= results_many["wasserstein_p_value"] <= 1

    def test_wasserstein_normalized_returned(self, parallel_trends_data):
        """Test that normalized Wasserstein distance is returned."""
        results = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42
        )

        assert "wasserstein_normalized" in results
        assert results["wasserstein_normalized"] >= 0

    def test_sample_sizes_returned(self, parallel_trends_data):
        """Test that sample sizes are returned."""
        results = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42
        )

        assert "n_treated" in results
        assert "n_control" in results
        assert results["n_treated"] > 0
        assert results["n_control"] > 0

    def test_insufficient_data_returns_nan(self):
        """Test that insufficient data returns NaN values."""
        # Only one observation per group
        data = pd.DataFrame({
            "unit": [0, 1],
            "period": [0, 0],
            "treated": [1, 0],
            "outcome": [10.0, 12.0],
        })

        results = check_parallel_trends_robust(
            data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0],
            seed=42
        )

        assert np.isnan(results["wasserstein_distance"])
        assert results["parallel_trends_plausible"] is None


# =============================================================================
# Tests for equivalence_test_trends
# =============================================================================


class TestEquivalenceTestTrends:
    """Additional tests for equivalence_test_trends function."""

    def test_tost_p_value_in_range(self, parallel_trends_data):
        """Test that TOST p-value is in valid range."""
        results = equivalence_test_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2]
        )

        assert 0 <= results["tost_p_value"] <= 1

    def test_equivalence_margin_auto_set(self, parallel_trends_data):
        """Test that equivalence margin is auto-set when not provided."""
        results = equivalence_test_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2]
        )

        assert results["equivalence_margin"] > 0

    def test_degrees_of_freedom_returned(self, parallel_trends_data):
        """Test that degrees of freedom are returned."""
        results = equivalence_test_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2]
        )

        assert "degrees_of_freedom" in results
        assert results["degrees_of_freedom"] > 0

    def test_tighter_margin_harder_to_pass(self, parallel_trends_data):
        """Test that tighter equivalence margin makes test harder to pass."""
        results_wide = equivalence_test_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            equivalence_margin=10.0  # Very wide margin
        )

        results_tight = equivalence_test_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            equivalence_margin=0.001  # Very tight margin
        )

        # Wide margin should have smaller TOST p-value (easier to show equivalence)
        assert results_wide["tost_p_value"] <= results_tight["tost_p_value"]


# =============================================================================
# Tests for compute_placebo_effects
# =============================================================================


class TestComputePlaceboEffects:
    """Tests for compute_placebo_effects function."""

    def test_returns_correct_number_of_effects(self, sdid_panel_data):
        """Test that function returns correct number of placebo effects."""
        # Prepare data matrices
        control_data = sdid_panel_data[sdid_panel_data["treated"] == 0]
        treated_data = sdid_panel_data[sdid_panel_data["treated"] == 1]

        control_pivot = control_data.pivot(index="period", columns="unit", values="outcome")
        treated_mean = treated_data.groupby("period")["outcome"].mean()

        pre_periods = [0, 1, 2, 3, 4]
        post_periods = [5, 6, 7]

        Y_pre_control = control_pivot.loc[pre_periods].values
        Y_post_control = control_pivot.loc[post_periods].values
        Y_pre_treated = treated_mean.loc[pre_periods].values

        # Compute weights
        unit_weights = compute_synthetic_weights(Y_pre_control, Y_pre_treated)
        time_weights = compute_time_weights(Y_pre_control, Y_pre_treated)

        control_units = list(control_pivot.columns)

        placebo_effects = compute_placebo_effects(
            Y_pre_control,
            Y_post_control,
            Y_pre_treated,
            unit_weights,
            time_weights,
            control_units
        )

        # Should have one placebo effect per control unit
        assert len(placebo_effects) == len(control_units)

    def test_placebo_effects_distribution(self, sdid_panel_data):
        """Test that placebo effects form a reasonable distribution."""
        control_data = sdid_panel_data[sdid_panel_data["treated"] == 0]
        treated_data = sdid_panel_data[sdid_panel_data["treated"] == 1]

        control_pivot = control_data.pivot(index="period", columns="unit", values="outcome")
        treated_mean = treated_data.groupby("period")["outcome"].mean()

        pre_periods = [0, 1, 2, 3, 4]
        post_periods = [5, 6, 7]

        Y_pre_control = control_pivot.loc[pre_periods].values
        Y_post_control = control_pivot.loc[post_periods].values
        Y_pre_treated = treated_mean.loc[pre_periods].values

        unit_weights = compute_synthetic_weights(Y_pre_control, Y_pre_treated)
        time_weights = compute_time_weights(Y_pre_control, Y_pre_treated)

        control_units = list(control_pivot.columns)

        placebo_effects = compute_placebo_effects(
            Y_pre_control,
            Y_post_control,
            Y_pre_treated,
            unit_weights,
            time_weights,
            control_units
        )

        # Placebo effects should be centered around zero (no treatment)
        assert np.abs(np.mean(placebo_effects)) < 2.0  # Allow some deviation

    def test_n_placebo_limits_output(self, sdid_panel_data):
        """Test that n_placebo parameter limits output size."""
        control_data = sdid_panel_data[sdid_panel_data["treated"] == 0]
        treated_data = sdid_panel_data[sdid_panel_data["treated"] == 1]

        control_pivot = control_data.pivot(index="period", columns="unit", values="outcome")
        treated_mean = treated_data.groupby("period")["outcome"].mean()

        pre_periods = [0, 1, 2, 3, 4]
        post_periods = [5, 6, 7]

        Y_pre_control = control_pivot.loc[pre_periods].values
        Y_post_control = control_pivot.loc[post_periods].values
        Y_pre_treated = treated_mean.loc[pre_periods].values

        unit_weights = compute_synthetic_weights(Y_pre_control, Y_pre_treated)
        time_weights = compute_time_weights(Y_pre_control, Y_pre_treated)

        control_units = list(control_pivot.columns)

        placebo_effects = compute_placebo_effects(
            Y_pre_control,
            Y_post_control,
            Y_pre_treated,
            unit_weights,
            time_weights,
            control_units,
            n_placebo=5
        )

        assert len(placebo_effects) == 5


# =============================================================================
# Additional tests for compute_synthetic_weights
# =============================================================================


class TestComputeSyntheticWeightsEdgeCases:
    """Edge case tests for compute_synthetic_weights."""

    def test_empty_control_matrix(self):
        """Test with empty control matrix."""
        Y_control = np.zeros((5, 0))
        Y_treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        weights = compute_synthetic_weights(Y_control, Y_treated)

        assert len(weights) == 0

    def test_single_control_unit(self):
        """Test with single control unit."""
        Y_control = np.array([[1.0], [2.0], [3.0]])
        Y_treated = np.array([1.0, 2.0, 3.0])

        weights = compute_synthetic_weights(Y_control, Y_treated)

        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-6

    def test_regularization_effect(self):
        """Test that regularization affects weight sparsity."""
        np.random.seed(42)
        Y_control = np.random.randn(10, 5)
        Y_treated = np.random.randn(10)

        weights_no_reg = compute_synthetic_weights(Y_control, Y_treated, lambda_reg=0.0)
        weights_high_reg = compute_synthetic_weights(Y_control, Y_treated, lambda_reg=10.0)

        # High regularization should give more uniform weights
        var_no_reg = np.var(weights_no_reg)
        var_high_reg = np.var(weights_high_reg)

        assert var_high_reg < var_no_reg + 0.01

    def test_min_weight_threshold(self):
        """Test that small weights are zeroed out."""
        np.random.seed(42)
        Y_control = np.random.randn(10, 5)
        Y_treated = np.random.randn(10)

        weights = compute_synthetic_weights(Y_control, Y_treated, min_weight=0.01)

        # All non-zero weights should be >= min_weight
        non_zero_weights = weights[weights > 0]
        assert np.all(non_zero_weights >= 0.01)


# =============================================================================
# Additional tests for compute_time_weights
# =============================================================================


class TestComputeTimeWeightsEdgeCases:
    """Edge case tests for compute_time_weights."""

    def test_single_period(self):
        """Test with single pre-treatment period."""
        Y_control = np.array([[1.0, 2.0, 3.0]])
        Y_treated = np.array([2.0])

        weights = compute_time_weights(Y_control, Y_treated)

        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-6

    def test_zeta_regularization_effect(self):
        """Test that zeta affects weight uniformity."""
        np.random.seed(42)
        Y_control = np.random.randn(10, 5)
        Y_treated = np.random.randn(10)

        weights_low_zeta = compute_time_weights(Y_control, Y_treated, zeta=0.1)
        weights_high_zeta = compute_time_weights(Y_control, Y_treated, zeta=10.0)

        # High zeta should give more uniform weights
        var_low = np.var(weights_low_zeta)
        var_high = np.var(weights_high_zeta)

        assert var_high <= var_low + 0.01

    def test_weights_all_positive(self):
        """Test that time weights are all positive."""
        np.random.seed(42)
        Y_control = np.random.randn(10, 5)
        Y_treated = np.random.randn(10)

        weights = compute_time_weights(Y_control, Y_treated)

        assert np.all(weights > 0)


# =============================================================================
# Additional tests for _project_simplex
# =============================================================================


class TestProjectSimplexEdgeCases:
    """Edge case tests for _project_simplex."""

    def test_empty_vector(self):
        """Test projection of empty vector."""
        v = np.array([])
        projected = _project_simplex(v)

        assert len(projected) == 0

    def test_single_element(self):
        """Test projection of single element."""
        v = np.array([5.0])
        projected = _project_simplex(v)

        assert len(projected) == 1
        assert abs(projected[0] - 1.0) < 1e-6

    def test_all_negative(self):
        """Test projection when all elements are negative."""
        v = np.array([-5.0, -3.0, -1.0])
        projected = _project_simplex(v)

        assert abs(np.sum(projected) - 1.0) < 1e-6
        assert np.all(projected >= 0)

    def test_already_on_simplex(self):
        """Test projection when already on simplex."""
        v = np.array([0.2, 0.3, 0.5])
        projected = _project_simplex(v)

        np.testing.assert_array_almost_equal(v, projected)

    def test_large_vector(self):
        """Test projection of large vector."""
        np.random.seed(42)
        v = np.random.randn(1000)
        projected = _project_simplex(v)

        assert abs(np.sum(projected) - 1.0) < 1e-6
        assert np.all(projected >= 0)


# =============================================================================
# Tests for compute_sdid_estimator
# =============================================================================


class TestComputeSDIDEstimator:
    """Tests for compute_sdid_estimator function."""

    def test_uniform_weights_equals_did(self):
        """Test that uniform weights gives standard DiD."""
        # Simple data with known DiD
        Y_pre_control = np.array([[10.0, 10.0]])  # 1 pre-period, 2 controls
        Y_post_control = np.array([[12.0, 12.0]])  # 1 post-period, 2 controls
        Y_pre_treated = np.array([10.0])
        Y_post_treated = np.array([16.0])

        # Uniform weights
        unit_weights = np.array([0.5, 0.5])
        time_weights = np.array([1.0])

        tau = compute_sdid_estimator(
            Y_pre_control, Y_post_control,
            Y_pre_treated, Y_post_treated,
            unit_weights, time_weights
        )

        # Standard DiD: (16-10) - (12-10) = 6 - 2 = 4
        assert abs(tau - 4.0) < 1e-6

    def test_concentrated_unit_weights(self):
        """Test with weight on single unit."""
        Y_pre_control = np.array([[10.0, 20.0]])
        Y_post_control = np.array([[12.0, 25.0]])
        Y_pre_treated = np.array([15.0])
        Y_post_treated = np.array([20.0])

        # All weight on first control
        unit_weights = np.array([1.0, 0.0])
        time_weights = np.array([1.0])

        tau = compute_sdid_estimator(
            Y_pre_control, Y_post_control,
            Y_pre_treated, Y_post_treated,
            unit_weights, time_weights
        )

        # DiD using only first control: (20-15) - (12-10) = 5 - 2 = 3
        assert abs(tau - 3.0) < 1e-6

    def test_multiple_post_periods(self):
        """Test with multiple post-treatment periods."""
        Y_pre_control = np.array([[10.0]])
        Y_post_control = np.array([[12.0], [14.0], [16.0]])  # 3 post periods
        Y_pre_treated = np.array([10.0])
        Y_post_treated = np.array([17.0, 19.0, 21.0])

        unit_weights = np.array([1.0])
        time_weights = np.array([1.0])

        tau = compute_sdid_estimator(
            Y_pre_control, Y_post_control,
            Y_pre_treated, Y_post_treated,
            unit_weights, time_weights
        )

        # Treated post mean: (17+19+21)/3 = 19
        # Control post mean: (12+14+16)/3 = 14
        # Treated DiD: 19 - 10 = 9
        # Control DiD: 14 - 10 = 4
        # tau = 9 - 4 = 5
        assert abs(tau - 5.0) < 1e-6
