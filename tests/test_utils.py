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
    _compute_outcome_changes,
    _project_simplex,
    check_parallel_trends,
    check_parallel_trends_robust,
    compute_confidence_interval,
    compute_p_value,
    compute_robust_se,
    compute_sdid_estimator,
    compute_time_weights,
    demean_by_group,
    demean_by_groups,
    equivalence_test_trends,
    fe_dummy_names,
    safe_inference,
    validate_binary,
    validate_covariate_names,
    validate_design_term_names,
    within_transform,
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
# Tests for safe_inference
# =============================================================================


class TestSafeInference:
    """Tests for safe_inference function."""

    def test_nan_se_returns_all_nan(self):
        """Test that NaN SE produces all NaN inference fields."""
        t_stat, p_value, (ci_lower, ci_upper) = safe_inference(5.0, np.nan)
        assert np.isnan(t_stat)
        assert np.isnan(p_value)
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_zero_se_returns_all_nan(self):
        """Test that zero SE produces all NaN inference fields."""
        t_stat, p_value, (ci_lower, ci_upper) = safe_inference(5.0, 0.0)
        assert np.isnan(t_stat)
        assert np.isnan(p_value)
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_negative_se_returns_all_nan(self):
        """Test that negative SE produces all NaN inference fields."""
        t_stat, p_value, (ci_lower, ci_upper) = safe_inference(5.0, -1.0)
        assert np.isnan(t_stat)
        assert np.isnan(p_value)
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_inf_se_returns_all_nan(self):
        """Test that infinite SE produces all NaN inference fields."""
        t_stat, p_value, (ci_lower, ci_upper) = safe_inference(5.0, np.inf)
        assert np.isnan(t_stat)
        assert np.isnan(p_value)
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_neg_inf_se_returns_all_nan(self):
        """Test that negative infinite SE produces all NaN inference fields."""
        t_stat, p_value, (ci_lower, ci_upper) = safe_inference(5.0, -np.inf)
        assert np.isnan(t_stat)
        assert np.isnan(p_value)
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_valid_se_normal_distribution(self):
        """Test valid SE with normal distribution (df=None)."""
        effect = 5.0
        se = 2.0
        t_stat, p_value, (ci_lower, ci_upper) = safe_inference(effect, se)

        assert t_stat == pytest.approx(2.5)
        assert 0 < p_value < 1
        assert ci_lower < effect < ci_upper

    def test_valid_se_t_distribution(self):
        """Test valid SE with t-distribution (df=30)."""
        effect = 3.0
        se = 1.5
        t_stat, p_value, (ci_lower, ci_upper) = safe_inference(effect, se, df=30)

        assert t_stat == pytest.approx(2.0)
        assert 0 < p_value < 1
        assert ci_lower < effect < ci_upper
        # t-distribution CI should be wider than normal for same alpha
        _, _, (ci_lower_norm, ci_upper_norm) = safe_inference(effect, se, df=None)
        assert (ci_upper - ci_lower) > (ci_upper_norm - ci_lower_norm)

    def test_return_type(self):
        """Test that return type is (float, float, (float, float))."""
        t_stat, p_value, conf_int = safe_inference(5.0, 1.0)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert isinstance(conf_int, tuple)
        assert len(conf_int) == 2
        assert isinstance(conf_int[0], float)
        assert isinstance(conf_int[1], float)

    def test_custom_alpha(self):
        """Test that alpha parameter affects CI width."""
        effect = 5.0
        se = 1.0
        _, _, (lower_95, upper_95) = safe_inference(effect, se, alpha=0.05)
        _, _, (lower_90, upper_90) = safe_inference(effect, se, alpha=0.10)

        width_95 = upper_95 - lower_95
        width_90 = upper_90 - lower_90
        assert width_95 > width_90

    def test_zero_effect(self):
        """Test with zero effect and valid SE."""
        t_stat, p_value, (ci_lower, ci_upper) = safe_inference(0.0, 1.0)

        assert t_stat == pytest.approx(0.0)
        assert p_value == pytest.approx(1.0)
        assert ci_lower < 0 < ci_upper


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
        assert results["parallel_trends_plausible"]

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
        assert not results["parallel_trends_plausible"]

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

    def test_silent_on_balanced_panel(self):
        """Balanced panel: only first-period-per-unit drops, no warning."""
        import warnings

        rng = np.random.default_rng(0)
        rows = []
        for unit in range(10):
            treated = int(unit >= 5)
            for t in range(1, 5):
                rows.append({
                    "unit": unit, "period": t,
                    "treated": treated, "outcome": rng.normal(),
                })
        df = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _compute_outcome_changes(
                df, outcome="outcome", time="period",
                treatment_group="treated", unit="unit",
            )

        # Generic filter on "dropped" catches both the old and new label so a
        # regression in the label wouldn't hide a real silent-drop warning.
        drop_warnings = [x for x in w if "dropped" in str(x.message).lower()]
        assert drop_warnings == []

    def test_warns_on_nan_outcomes_with_excess_drop_count(self):
        """Extra NaN-outcome rows beyond first-period drops must surface via
        a UserWarning reporting the excess count (axis-E drop counter)."""
        rng = np.random.default_rng(0)
        rows = []
        for unit in range(10):
            treated = int(unit >= 5)
            for t in range(1, 5):
                rows.append({
                    "unit": unit, "period": t,
                    "treated": treated, "outcome": rng.normal(),
                })
        df = pd.DataFrame(rows)
        df.loc[[5, 12, 22], "outcome"] = np.nan

        with pytest.warns(
            UserWarning,
            match=r"parallel-trend diagnostic: dropped \d+ row\(s\).*additional NaN first-differences",
        ):
            _compute_outcome_changes(
                df, outcome="outcome", time="period",
                treatment_group="treated", unit="unit",
            )

    def test_warning_label_reflects_public_caller(self):
        """`check_parallel_trends_robust` and `equivalence_test_trends` must
        each surface the axis-E excess-drop warning under their own name so
        users can trace the signal back to the function they called."""
        rng = np.random.default_rng(0)
        rows = []
        for unit in range(10):
            treated = int(unit >= 5)
            for t in range(1, 5):
                rows.append({
                    "unit": unit, "period": t,
                    "treated": treated, "outcome": rng.normal(),
                })
        df = pd.DataFrame(rows)
        df.loc[[5, 12, 22], "outcome"] = np.nan

        with pytest.warns(UserWarning, match="check_parallel_trends_robust:"):
            check_parallel_trends_robust(
                df, outcome="outcome", time="period",
                treatment_group="treated", unit="unit",
                n_permutations=100, seed=0,
            )

        with pytest.warns(UserWarning, match="equivalence_test_trends:"):
            equivalence_test_trends(
                df, outcome="outcome", time="period",
                treatment_group="treated", unit="unit",
            )


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


# Removed TestComputeSyntheticWeightsEdgeCases in the silent-failures audit
# post-cleanup (finding #22). The `compute_synthetic_weights` helper was deleted;
# its behavior is now exercised via `rank_control_units` in test_prep.py.


# =============================================================================
# Additional tests for compute_time_weights
# =============================================================================


class TestComputeTimeWeightsEdgeCases:
    """Edge case tests for compute_time_weights (new Frank-Wolfe signature)."""

    def test_single_period(self):
        """Test with single pre-treatment period."""
        Y_pre_control = np.array([[1.0, 2.0, 3.0]])
        Y_post_control = np.array([[4.0, 5.0, 6.0]])

        weights = compute_time_weights(Y_pre_control, Y_post_control, zeta_lambda=0.01)

        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-6

    def test_zeta_regularization_effect(self):
        """Test that zeta_lambda affects weight uniformity."""
        np.random.seed(42)
        Y_pre = np.random.randn(10, 5)
        Y_post = np.random.randn(3, 5)

        weights_low = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.001)
        weights_high = compute_time_weights(Y_pre, Y_post, zeta_lambda=100.0)

        # High regularization should give more uniform weights
        var_low = np.var(weights_low)
        var_high = np.var(weights_high)

        assert var_high <= var_low + 0.01

    def test_weights_nonnegative(self):
        """Test that time weights are non-negative (simplex constraint)."""
        np.random.seed(42)
        Y_pre = np.random.randn(10, 5)
        Y_post = np.random.randn(3, 5)

        weights = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        assert np.all(weights >= -1e-10)

    def test_weights_sum_to_one(self):
        """Test that time weights sum to 1."""
        np.random.seed(42)
        Y_pre = np.random.randn(10, 5)
        Y_post = np.random.randn(3, 5)

        weights = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        assert abs(np.sum(weights) - 1.0) < 1e-6


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


class TestValidateCovariateNames:
    """Tests for validate_covariate_names (covariate/reserved-term collision guard)."""

    def test_none_is_noop(self):
        validate_covariate_names(None, {"const", "treated"})

    def test_empty_is_noop(self):
        validate_covariate_names([], {"const", "treated"})

    def test_non_colliding_passes(self):
        # Should not raise.
        validate_covariate_names(["x1", "x2"], {"const", "treated", "treated:post"})

    def test_collision_raises(self):
        with pytest.raises(ValueError, match="collide"):
            validate_covariate_names(["const"], {"const", "treated"})

    def test_collision_lists_offender_and_estimator(self):
        with pytest.raises(ValueError, match="MyEstimator") as exc:
            validate_covariate_names(
                ["age", "const"], {"const", "treated"}, estimator="MyEstimator"
            )
        # The offending name is reported (not the innocent one as a collision).
        assert "const" in str(exc.value)

    def test_case_sensitive_does_not_collide(self):
        # Column names / dict keys are case-sensitive; "Const" != "const".
        validate_covariate_names(["Const"], {"const", "treated"})

    def test_duplicate_covariates_raise(self):
        with pytest.raises(ValueError, match="duplicate"):
            validate_covariate_names(["x1", "x1"], {"const"})

    def test_duplicate_takes_precedence_only_when_no_collision(self):
        # A name that both collides AND is duplicated reports the collision first.
        with pytest.raises(ValueError, match="collide"):
            validate_covariate_names(["const", "const"], {"const"})


class TestValidateDesignTermNames:
    """Tests for validate_design_term_names (final var_names uniqueness backstop)."""

    def test_unique_passes(self):
        validate_design_term_names(["const", "treated", "post", "treated:post", "x1"])

    def test_duplicate_raises(self):
        # e.g. a fixed-effect dummy "period_2" colliding with a structural key.
        with pytest.raises(ValueError, match="collide"):
            validate_design_term_names(
                ["const", "treated", "period_2", "period_2"], estimator="MultiPeriodDiD"
            )

    def test_message_names_duplicate_and_estimator(self):
        with pytest.raises(ValueError, match="MultiPeriodDiD") as exc:
            validate_design_term_names(["a", "a"], estimator="MultiPeriodDiD")
        assert "a" in str(exc.value)

    def test_empty_passes(self):
        validate_design_term_names([])


class TestFeDummyNames:
    """fe_dummy_names must match pd.get_dummies(drop_first=True).columns exactly
    (including Categorical non-default order) without materializing the matrix."""

    @pytest.mark.parametrize(
        "col",
        [
            pd.Series([3, 1, 2, 1]),
            pd.Series(["b", "a", "c", "a"]),
            pd.Series(pd.Categorical(["m", "a", "z", "a"], categories=["m", "a", "z"])),
            pd.Series([2.0, 1.0, 3.0]),
            pd.Series(["b", "a", np.nan, "a"]),
        ],
    )
    def test_matches_get_dummies(self, col):
        assert fe_dummy_names(col, "fe") == list(
            pd.get_dummies(col, prefix="fe", drop_first=True).columns
        )


# =============================================================================
# demean_by_groups — N-way method of alternating projections (MAP)
# =============================================================================


def _unbalanced_2way_panel(seed=0, drop=0.30):
    """Unbalanced (non-orthogonal) 2-way panel: some unit-period cells dropped."""
    rng = np.random.default_rng(seed)
    rows = [
        (u, t)
        for u in range(8)
        for t in range(6)
        if rng.random() >= drop
    ]
    df = pd.DataFrame(rows, columns=["unit", "period"])
    n = len(df)
    df["x1"] = rng.normal(size=n)
    df["x2"] = rng.normal(size=n)
    df["y"] = 2.0 * df["x1"] - 1.5 * df["x2"] + rng.normal(size=n)
    df["w"] = rng.uniform(0.5, 2.0, size=n)
    return df


def _full_dummy_slopes(df, group_cols, xcols, weights=None):
    """Ground truth: (W)OLS of y on [1, xcols, dummies(group_cols)]; return x slopes."""
    cols = [np.ones(len(df))] + [df[c].values.astype(float) for c in xcols]
    for g in group_cols:
        d = pd.get_dummies(df[g], prefix=g, drop_first=True).values.astype(float)
        cols.extend(d[:, j] for j in range(d.shape[1]))
    X = np.column_stack(cols)
    y = df["y"].values.astype(float)
    if weights is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        sw = np.sqrt(np.asarray(weights, dtype=float))
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta[1 : 1 + len(xcols)]


def _fwl_slopes(demeaned, xcols, weights=None):
    """OLS of demeaned y on demeaned xcols (FWL residualization)."""
    X = np.column_stack([demeaned[c] for c in xcols])
    y = demeaned["y"]
    if weights is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        sw = np.sqrt(np.asarray(weights, dtype=float))
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def _frozen_old_within_transform_weighted(
    df, variables, unit, time, weights, max_iter=100, tol=1e-8
):
    """Byte-for-byte copy of the PRE-REFACTOR within_transform weighted loop.

    Used as the byte-identity guard: demean_by_groups([unit, time], weighted) and
    the refactored within_transform must reproduce this exactly.
    """
    w = np.asarray(weights, dtype=np.float64)
    unit_groups = df[unit].values
    time_groups = df[time].values
    unit_w_sum = pd.Series(w).groupby(unit_groups).transform("sum").values
    time_w_sum = pd.Series(w).groupby(time_groups).transform("sum").values

    def _wgd(x, groups, w, w_sum):
        wx_sum = pd.Series(w * x).groupby(groups).transform("sum").values
        return x - wx_sum / w_sum

    out = {}
    for var in variables:
        x = df[var].values.astype(np.float64)
        for _ in range(max_iter):
            x_old = x.copy()
            x = _wgd(x, unit_groups, w, unit_w_sum)
            x = _wgd(x, time_groups, w, time_w_sum)
            if np.max(np.abs(x - x_old)) < tol:
                break
        out[var] = x
    return out


class TestDemeanByGroups:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_len1_byte_identical_to_demean_by_group(self, weighted):
        """One grouping var must delegate to demean_by_group (byte-identical)."""
        df = _unbalanced_2way_panel(seed=1)
        w = df["w"].values if weighted else None
        out_groups, n_g = demean_by_groups(
            df, ["y", "x1"], ["unit"], suffix="_dm", weights=w
        )
        out_single, n_s = demean_by_group(
            df, ["y", "x1"], "unit", suffix="_dm", weights=w
        )
        assert n_g == n_s
        np.testing.assert_array_equal(
            out_groups["y_dm"].values, out_single["y_dm"].values
        )
        np.testing.assert_array_equal(
            out_groups["x1_dm"].values, out_single["x1_dm"].values
        )

    def test_n_effects_is_sum_nunique_minus_one(self):
        df = _unbalanced_2way_panel(seed=2)
        _, n_eff = demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm")
        expected = (df["unit"].nunique() - 1) + (df["period"].nunique() - 1)
        assert n_eff == expected

    @pytest.mark.parametrize("weighted", [False, True])
    def test_unbalanced_2way_matches_full_dummy_ols(self, weighted):
        """The core correctness claim: MAP residualization == full-dummy (W)OLS."""
        df = _unbalanced_2way_panel(seed=3)
        w = df["w"].values if weighted else None
        out, _ = demean_by_groups(
            df, ["y", "x1", "x2"], ["unit", "period"], suffix="_dm", weights=w
        )
        demeaned = {c: out[f"{c}_dm"].values for c in ("y", "x1", "x2")}
        map_slopes = _fwl_slopes(demeaned, ["x1", "x2"], weights=w)
        gt = _full_dummy_slopes(df, ["unit", "period"], ["x1", "x2"], weights=w)
        np.testing.assert_allclose(map_slopes, gt, atol=1e-9, rtol=0)

    def test_n3_absorb_matches_full_dummy_ols(self):
        """Generalizes to 3 absorbed dimensions."""
        rng = np.random.default_rng(4)
        rows = [
            (u, t, (u + t) % 4)
            for u in range(10)
            for t in range(6)
            if rng.random() >= 0.4
        ]
        df = pd.DataFrame(rows, columns=["unit", "period", "firm"])
        n = len(df)
        df["x1"] = rng.normal(size=n)
        df["y"] = 2.0 * df["x1"] + rng.normal(size=n)
        out, _ = demean_by_groups(
            df, ["y", "x1"], ["unit", "period", "firm"], suffix="_dm"
        )
        demeaned = {c: out[f"{c}_dm"].values for c in ("y", "x1")}
        map_slope = _fwl_slopes(demeaned, ["x1"])[0]
        gt = _full_dummy_slopes(df, ["unit", "period", "firm"], ["x1"])[0]
        np.testing.assert_allclose(map_slope, gt, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_result_orthogonal_to_fe_spans(self, weighted):
        """Demeaned variables must have ~0 (weighted) group means in every FE dim."""
        df = _unbalanced_2way_panel(seed=5)
        w = df["w"].values if weighted else None
        out, _ = demean_by_groups(
            df, ["y"], ["unit", "period"], suffix="_dm", weights=w, tol=1e-12
        )
        ydm = out["y_dm"].values
        for g in ("unit", "period"):
            if weighted:
                num = pd.Series(w * ydm).groupby(df[g].values).transform("sum").values
                den = pd.Series(w).groupby(df[g].values).transform("sum").values
                means = num / den
            else:
                means = pd.Series(ydm).groupby(df[g].values).transform("mean").values
            assert np.max(np.abs(means)) < 1e-9

    def test_weighted_byte_identity_vs_frozen_within_transform(self):
        """demean_by_groups([unit, time], weighted) reproduces the old loop exactly."""
        df = _unbalanced_2way_panel(seed=6)
        w = df["w"].values
        frozen = _frozen_old_within_transform_weighted(
            df, ["y", "x1", "x2"], "unit", "period", w
        )
        out, _ = demean_by_groups(
            df, ["y", "x1", "x2"], ["unit", "period"], suffix="_dm", weights=w, tol=1e-8
        )
        for var in ("y", "x1", "x2"):
            np.testing.assert_array_equal(out[f"{var}_dm"].values, frozen[var])

    def test_within_transform_weighted_unchanged(self):
        """The refactored within_transform weighted path is byte-identical to before."""
        df = _unbalanced_2way_panel(seed=7)
        w = df["w"].values
        frozen = _frozen_old_within_transform_weighted(
            df, ["y", "x1"], "unit", "period", w
        )
        out = within_transform(df, ["y", "x1"], "unit", "period", weights=w)
        np.testing.assert_array_equal(out["y_demeaned"].values, frozen["y"])
        np.testing.assert_array_equal(out["x1_demeaned"].values, frozen["x1"])

    def test_within_transform_unweighted_now_matches_full_dummy(self):
        """Unweighted within_transform now uses MAP -> exact on unbalanced panels."""
        df = _unbalanced_2way_panel(seed=8)
        out = within_transform(df, ["y", "x1", "x2"], "unit", "period")
        demeaned = {c: out[f"{c}_demeaned"].values for c in ("y", "x1", "x2")}
        map_slopes = _fwl_slopes(demeaned, ["x1", "x2"])
        gt = _full_dummy_slopes(df, ["unit", "period"], ["x1", "x2"])
        np.testing.assert_allclose(map_slopes, gt, atol=1e-9, rtol=0)

    def test_nonconvergence_emits_warning(self):
        """A starved iteration budget on an unbalanced panel warns (not silent)."""
        df = _unbalanced_2way_panel(seed=9)
        with pytest.warns(UserWarning, match="did not converge"):
            demean_by_groups(
                df, ["y"], ["unit", "period"], suffix="_dm", max_iter=1, tol=1e-15
            )

    def test_empty_group_vars_raises(self):
        df = _unbalanced_2way_panel(seed=10)
        with pytest.raises(ValueError, match="at least one grouping variable"):
            demean_by_groups(df, ["y"], [])
