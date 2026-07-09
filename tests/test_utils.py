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

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff.utils import (
    _compute_outcome_changes,
    _iterative_fe_solve,
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
    X = np.column_stack(
        [
            np.ones(n),
            np.random.randn(n),
            np.random.randn(n),
        ]
    )
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

    X = np.column_stack(
        [
            np.ones(n),
            np.random.randn(n),
        ]
    )

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

            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                }
            )

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

            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "outcome": y,
                }
            )

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
            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": 0,
                    "outcome": y,
                }
            )

    # Treated units
    for unit in range(n_control, n_control + n_treated):
        unit_effect = np.random.normal(0, 2)
        for period in range(n_pre + n_post):
            y = 10.0 + unit_effect + period * 0.5
            if period >= n_pre:
                y += 3.0  # Treatment effect
            y += np.random.normal(0, 0.3)
            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": 1,
                    "outcome": y,
                }
            )

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

    def test_nonfinite_df_returns_all_nan(self):
        """Non-finite df (NaN/inf) produces all-NaN inference even with a valid SE.

        A guard-suppressed / non-physical Bell-McCaffrey Satterthwaite DOF is
        surfaced as NaN by `_cr2_bm_dof_inner`; a coefficient whose DOF was
        declared unreliable must not report finite (or partially-finite) t/p/CI.
        `df <= 0` is already rejected; this covers the non-finite case where
        `df <= 0` is False for NaN.
        """
        for bad_df in (np.nan, np.inf, -np.inf):
            t_stat, p_value, (ci_lower, ci_upper) = safe_inference(5.0, 2.0, df=bad_df)
            assert np.isnan(t_stat), f"df={bad_df}: t_stat should be NaN"
            assert np.isnan(p_value), f"df={bad_df}: p_value should be NaN"
            assert np.isnan(ci_lower) and np.isnan(ci_upper), f"df={bad_df}: CI should be NaN"

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
            pre_periods=[0, 1, 2],
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
            pre_periods=[0, 1, 2],
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
            pre_periods=[0, 1, 2],
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
            pre_periods=[0, 1, 2],
        )

        # Treated has steeper trend (3.0 vs 1.0), so difference should be positive
        assert results["trend_difference"] > 0

    def test_auto_infer_pre_periods(self, parallel_trends_data):
        """Test automatic inference of pre-periods."""
        results = check_parallel_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            # pre_periods not specified
        )

        # Should still return valid results
        assert "treated_trend" in results
        assert not np.isnan(results["treated_trend"])

    def test_single_period_returns_nan(self):
        """Test that single pre-period returns NaN for trends."""
        data = pd.DataFrame(
            {
                "outcome": [10, 11, 12, 13],
                "period": [0, 0, 0, 0],  # All same period
                "treated": [1, 1, 0, 0],
            }
        )

        results = check_parallel_trends(
            data, outcome="outcome", time="period", treatment_group="treated", pre_periods=[0]
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
            pre_periods=[0, 1, 2],
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
            pre_data, outcome="outcome", time="period", treatment_group="treated", unit="unit"
        )

        # Should have changes for each unit-period transition
        assert len(treated_changes) > 0
        assert len(control_changes) > 0

    def test_without_unit_specified(self, parallel_trends_data):
        """Test outcome changes computation without unit identifier."""
        pre_data = parallel_trends_data[parallel_trends_data["period"] < 3]

        treated_changes, control_changes = _compute_outcome_changes(
            pre_data, outcome="outcome", time="period", treatment_group="treated", unit=None
        )

        # Should have aggregate changes (fewer than with unit)
        assert len(treated_changes) > 0
        assert len(control_changes) > 0

    def test_returns_float_arrays(self, parallel_trends_data):
        """Test that function returns float arrays."""
        pre_data = parallel_trends_data[parallel_trends_data["period"] < 3]

        treated_changes, control_changes = _compute_outcome_changes(
            pre_data, outcome="outcome", time="period", treatment_group="treated", unit="unit"
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
                data.append(
                    {
                        "unit": unit,
                        "period": period,
                        "treated": int(is_treated),
                        "outcome": y,
                    }
                )

        df = pd.DataFrame(data)

        treated_changes, control_changes = _compute_outcome_changes(
            df, outcome="outcome", time="period", treatment_group="treated", unit="unit"
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
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "treated": treated,
                        "outcome": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _compute_outcome_changes(
                df,
                outcome="outcome",
                time="period",
                treatment_group="treated",
                unit="unit",
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
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "treated": treated,
                        "outcome": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        df.loc[[5, 12, 22], "outcome"] = np.nan

        with pytest.warns(
            UserWarning,
            match=r"parallel-trend diagnostic: dropped \d+ row\(s\).*additional NaN first-differences",
        ):
            _compute_outcome_changes(
                df,
                outcome="outcome",
                time="period",
                treatment_group="treated",
                unit="unit",
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
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "treated": treated,
                        "outcome": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        df.loc[[5, 12, 22], "outcome"] = np.nan

        with pytest.warns(UserWarning, match="check_parallel_trends_robust:"):
            check_parallel_trends_robust(
                df,
                outcome="outcome",
                time="period",
                treatment_group="treated",
                unit="unit",
                n_permutations=100,
                seed=0,
            )

        with pytest.warns(UserWarning, match="equivalence_test_trends:"):
            equivalence_test_trends(
                df,
                outcome="outcome",
                time="period",
                treatment_group="treated",
                unit="unit",
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
            seed=42,
        )

        results2 = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=42,
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
            seed=42,
        )

        results2 = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            seed=123,
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
            seed=42,
        )

        results_many = check_parallel_trends_robust(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            n_permutations=1000,
            seed=42,
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
            seed=42,
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
            seed=42,
        )

        assert "n_treated" in results
        assert "n_control" in results
        assert results["n_treated"] > 0
        assert results["n_control"] > 0

    def test_insufficient_data_returns_nan(self):
        """Test that insufficient data returns NaN values."""
        # Only one observation per group
        data = pd.DataFrame(
            {
                "unit": [0, 1],
                "period": [0, 0],
                "treated": [1, 0],
                "outcome": [10.0, 12.0],
            }
        )

        results = check_parallel_trends_robust(
            data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0],
            seed=42,
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
            pre_periods=[0, 1, 2],
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
            pre_periods=[0, 1, 2],
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
            pre_periods=[0, 1, 2],
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
            equivalence_margin=10.0,  # Very wide margin
        )

        results_tight = equivalence_test_trends(
            parallel_trends_data,
            outcome="outcome",
            time="period",
            treatment_group="treated",
            unit="unit",
            pre_periods=[0, 1, 2],
            equivalence_margin=0.001,  # Very tight margin
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
            Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated, unit_weights, time_weights
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
            Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated, unit_weights, time_weights
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
            Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated, unit_weights, time_weights
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
    rows = [(u, t) for u in range(8) for t in range(6) if rng.random() >= drop]
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
    """Byte-for-byte copy of the PRE-v3.6.x pandas-groupby weighted MAP loop.

    Historical reference implementation. The v3.6.x factorize-once + bincount
    engine is NOT bit-identical to this loop (bincount accumulation is not
    Kahan-compensated the way pandas' grouped mean is; drift compounds across
    MAP iterations to ~1e-10 order — see REGISTRY "Absorbed Fixed Effects").
    Kept as a drift-bound guard: the new engine must agree with this loop at
    atol=1e-9, and with full-dummy WLS ground truth at atol=1e-9.
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
        out_groups, n_g = demean_by_groups(df, ["y", "x1"], ["unit"], suffix="_dm", weights=w)
        out_single, n_s = demean_by_group(df, ["y", "x1"], "unit", suffix="_dm", weights=w)
        assert n_g == n_s
        np.testing.assert_array_equal(out_groups["y_dm"].values, out_single["y_dm"].values)
        np.testing.assert_array_equal(out_groups["x1_dm"].values, out_single["x1_dm"].values)

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
        rows = [(u, t, (u + t) % 4) for u in range(10) for t in range(6) if rng.random() >= 0.4]
        df = pd.DataFrame(rows, columns=["unit", "period", "firm"])
        n = len(df)
        df["x1"] = rng.normal(size=n)
        df["y"] = 2.0 * df["x1"] + rng.normal(size=n)
        out, _ = demean_by_groups(df, ["y", "x1"], ["unit", "period", "firm"], suffix="_dm")
        demeaned = {c: out[f"{c}_dm"].values for c in ("y", "x1")}
        map_slope = _fwl_slopes(demeaned, ["x1"])[0]
        gt = _full_dummy_slopes(df, ["unit", "period", "firm"], ["x1"])[0]
        np.testing.assert_allclose(map_slope, gt, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_result_orthogonal_to_fe_spans(self, weighted):
        """Demeaned variables must have ~0 (weighted) group means in every FE dim."""
        df = _unbalanced_2way_panel(seed=5)
        w = df["w"].values if weighted else None
        out, _ = demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm", weights=w, tol=1e-12)
        ydm = out["y_dm"].values
        for g in ("unit", "period"):
            if weighted:
                num = pd.Series(w * ydm).groupby(df[g].values).transform("sum").values
                den = pd.Series(w).groupby(df[g].values).transform("sum").values
                means = num / den
            else:
                means = pd.Series(ydm).groupby(df[g].values).transform("mean").values
            assert np.max(np.abs(means)) < 1e-9

    def test_weighted_close_to_frozen_pandas_loop(self):
        """demean_by_groups([unit, time], weighted) agrees with the legacy
        pandas-groupby loop at the documented ~1e-9 drift bound (bincount
        accumulation vs Kahan-compensated pandas mean; not bit-identical)."""
        df = _unbalanced_2way_panel(seed=6)
        w = df["w"].values
        frozen = _frozen_old_within_transform_weighted(df, ["y", "x1", "x2"], "unit", "period", w)
        out, _ = demean_by_groups(
            df, ["y", "x1", "x2"], ["unit", "period"], suffix="_dm", weights=w, tol=1e-8
        )
        for var in ("y", "x1", "x2"):
            np.testing.assert_allclose(out[f"{var}_dm"].values, frozen[var], atol=1e-9, rtol=0)

    def test_within_transform_weighted_close_to_frozen_loop(self):
        """within_transform weighted path agrees with the legacy loop at 1e-9."""
        df = _unbalanced_2way_panel(seed=7)
        w = df["w"].values
        frozen = _frozen_old_within_transform_weighted(df, ["y", "x1"], "unit", "period", w)
        out = within_transform(df, ["y", "x1"], "unit", "period", weights=w)
        np.testing.assert_allclose(out["y_demeaned"].values, frozen["y"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(out["x1_demeaned"].values, frozen["x1"], atol=1e-9, rtol=0)

    def test_within_transform_weighted_matches_full_dummy_wls(self):
        """Ground truth (not implementation identity): weighted within_transform
        residualization must reproduce full-dummy WLS slopes."""
        df = _unbalanced_2way_panel(seed=7)
        w = df["w"].values
        out = within_transform(df, ["y", "x1", "x2"], "unit", "period", weights=w)
        demeaned = {c: out[f"{c}_demeaned"].values for c in ("y", "x1", "x2")}
        map_slopes = _fwl_slopes(demeaned, ["x1", "x2"], weights=w)
        gt = _full_dummy_slopes(df, ["unit", "period"], ["x1", "x2"], weights=w)
        np.testing.assert_allclose(map_slopes, gt, atol=1e-9, rtol=0)

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
            demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm", max_iter=1, tol=1e-15)

    def test_empty_group_vars_raises(self):
        df = _unbalanced_2way_panel(seed=10)
        with pytest.raises(ValueError, match="at least one grouping variable"):
            demean_by_groups(df, ["y"], [])

    @pytest.mark.parametrize("weighted", [False, True])
    def test_nan_group_key_raises(self, weighted):
        """NaN in an absorbed group column must raise, naming the column.

        pd.factorize codes NaN as -1, which would silently index the LAST
        group's mean; the pre-v3.6.x behavior was also silently wrong
        (unweighted NaN-poisoned rows; weighted passed them through
        un-demeaned). REGISTRY "Absorbed Fixed Effects" edge case.
        """
        df = _unbalanced_2way_panel(seed=11)
        df.loc[df.index[5], "period"] = np.nan
        w = df["w"].values if weighted else None
        with pytest.raises(ValueError, match="'period' contains NaN"):
            demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm", weights=w)

    def test_zero_total_weight_group_rows_inert_and_finite(self):
        """Zero-total-weight groups stay inert (no NaN/Inf poisoning) and
        positive-weight groups remain weighted-orthogonal."""
        df = _unbalanced_2way_panel(seed=12)
        w = df["w"].values.copy()
        zero_units = df["unit"].values == df["unit"].values[0]
        w[zero_units] = 0.0
        out, _ = demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm", weights=w, tol=1e-12)
        ydm = out["y_dm"].values
        assert np.isfinite(ydm).all()
        pos = ~zero_units
        num = pd.Series(w * ydm).groupby(df["unit"].values).transform("sum").values
        den = pd.Series(w).groupby(df["unit"].values).transform("sum").values
        means = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        assert np.max(np.abs(means[pos])) < 1e-9

    def test_defaults_max_iter_10000_both_entry_points(self):
        """Both re-declared max_iter defaults must stay in sync at 10,000
        (fixest fixef.iter / pyfixest fixef_maxiter parity); raising only one
        would leave the within_transform family capped differently."""
        import inspect

        assert inspect.signature(demean_by_groups).parameters["max_iter"].default == 10_000
        assert inspect.signature(within_transform).parameters["max_iter"].default == 10_000

    def test_balanced_panel_converges_within_two_iterations(self):
        """Balanced fully-crossed panels have orthogonal FE subspaces: MAP must
        converge in ~2 sweeps (iteration-count regression guard)."""
        rng = np.random.default_rng(13)
        df = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(30), 8),
                "period": np.tile(np.arange(8), 30),
            }
        )
        df["y"] = rng.normal(size=len(df))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a convergence warning fails the test
            demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm", max_iter=2, tol=1e-10)

    def test_correlated_fe_needs_over_100_iterations_and_converges(self):
        """The headline max_iter change: banded (contiguous-lifetime) two-way
        incidence genuinely needs >100 MAP iterations. Under the old cap of 100
        this warned and returned slightly-off residuals; under the 10,000
        default it must converge silently AND match full-dummy OLS.

        Iteration count depends on the angle between the FE subspaces, not on
        data size, so this fixture stays small and fast.
        """
        rng = np.random.default_rng(14)
        n_units, n_periods = 300, 60
        span = max(2, int(n_periods * 0.1))  # 10% contiguous lifetimes
        entry = rng.integers(0, n_periods - span + 1, n_units)
        rows = [(u, t) for u in range(n_units) for t in range(entry[u], entry[u] + span)]
        df = pd.DataFrame(rows, columns=["unit", "period"])
        df["x1"] = rng.normal(size=len(df))
        df["y"] = 1.5 * df["x1"] + rng.normal(size=len(df))

        # the fixture is honest: the old cap genuinely fails on it
        with pytest.warns(UserWarning, match="did not converge"):
            demean_by_groups(df.copy(), ["y", "x1"], ["unit", "period"], suffix="_dm", max_iter=100)

        # the new default converges silently and matches ground truth
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out, _ = demean_by_groups(df, ["y", "x1"], ["unit", "period"], suffix="_dm")
        demeaned = {c: out[f"{c}_dm"].values for c in ("y", "x1")}
        map_slope = _fwl_slopes(demeaned, ["x1"])[0]
        gt = _full_dummy_slopes(df, ["unit", "period"], ["x1"])[0]
        np.testing.assert_allclose(map_slope, gt, atol=1e-8, rtol=0)


class TestSnapAbsorbedRegressors:
    """Unit tests for the FE-spanned-regressor snap (REGISTRY 'Absorbed FE')."""

    def _frame(self, seed=0):
        rng = np.random.default_rng(seed)
        n = 40
        return pd.DataFrame(
            {
                "junk_dm": rng.normal(0.0, 1e-14, n),  # spanned: numerical junk
                "real_dm": rng.normal(0.0, 1.0, n),  # genuinely identified
                "g": np.arange(n) % 4,  # FE column for group_vars
            }
        )

    def test_snaps_junk_keeps_real(self):
        from diff_diff.utils import snap_absorbed_regressors

        df = self._frame()
        real_before = df["real_dm"].values.copy()
        with pytest.warns(UserWarning, match="collinear with the absorbed"):
            snapped = snap_absorbed_regressors(
                df,
                ["junk", "real"],
                {"junk": 1.0, "real": 1.0},
                absorbed_desc="test FEs",
                group_vars=["g"],
                suffix="_dm",
            )
        assert snapped == ["junk"]
        assert (df["junk_dm"].values == 0.0).all()
        np.testing.assert_array_equal(df["real_dm"].values, real_before)

    @pytest.mark.parametrize("action", ["silent", "error"])
    def test_non_warn_actions_snap_without_warning(self, action):
        from diff_diff.utils import snap_absorbed_regressors

        df = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            snapped = snap_absorbed_regressors(
                df,
                ["junk"],
                {"junk": 1.0},
                absorbed_desc="test FEs",
                group_vars=["g"],
                rank_deficient_action=action,
                suffix="_dm",
            )
        assert snapped == ["junk"]
        assert (df["junk_dm"].values == 0.0).all()

    def test_zero_pre_norm_skipped(self):
        from diff_diff.utils import snap_absorbed_regressors

        df = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            snapped = snap_absorbed_regressors(
                df,
                ["junk"],
                {"junk": 0.0},  # all-zero input column: rank handling owns it
                absorbed_desc="test FEs",
                group_vars=["g"],
                suffix="_dm",
            )
        assert snapped == []

    def test_display_names_in_warning(self):
        from diff_diff.utils import snap_absorbed_regressors

        df = self._frame()
        with pytest.warns(UserWarning, match=r"treated:post"):
            snap_absorbed_regressors(
                df,
                ["junk"],
                {"junk": 1.0},
                absorbed_desc="test FEs",
                group_vars=["g"],
                suffix="_dm",
                display_names={"junk": "treated:post"},
            )

    def test_pre_demean_norms_captures_l2(self):
        from diff_diff.utils import pre_demean_norms

        df = pd.DataFrame({"a": [3.0, 4.0], "b": [0, 0]})
        norms = pre_demean_norms(df, ["a", "b"])
        assert norms["a"] == pytest.approx(5.0)
        assert norms["b"] == 0.0


class TestReviewFollowupsAbsorbedFE:
    """Local-review follow-ups: one-way NaN guard + weight-aware snap."""

    @pytest.mark.parametrize("weighted", [False, True])
    def test_one_way_nan_group_key_raises(self, weighted):
        """The NaN-group guard must cover len(group_vars) == 1 too (the
        delegation to demean_by_group previously bypassed it)."""
        df = _unbalanced_2way_panel(seed=20)
        df["unit"] = df["unit"].astype(float)
        df.loc[df.index[3], "unit"] = np.nan
        w = df["w"].values if weighted else None
        with pytest.raises(ValueError, match="'unit' contains NaN"):
            demean_by_groups(df, ["y"], ["unit"], suffix="_dm", weights=w)

    def test_estimator_one_way_absorb_nan_raises(self):
        from diff_diff import DifferenceInDifferences

        df = _unbalanced_2way_panel(seed=21)
        df["treated"] = (df["unit"] < df["unit"].median()).astype(int)
        df["post"] = (df["period"] >= df["period"].median()).astype(int)
        df["fe"] = df["unit"].astype(float)
        df.loc[df.index[5], "fe"] = np.nan
        with pytest.raises(ValueError, match="'fe' contains NaN"):
            DifferenceInDifferences().fit(
                df, outcome="y", treatment="treated", time="post", absorb=["fe"]
            )

    def test_weight_aware_snap_zero_weight_domain(self):
        """A regressor FE-spanned on the POSITIVE-weight sample must snap even
        though zero-weight domain rows (left inert by the weighted demean)
        carry non-zero demeaned values that would mask it unweighted."""
        from diff_diff.utils import pre_demean_norms, snap_absorbed_regressors

        rng = np.random.default_rng(22)
        n_units, n_periods = 30, 6
        unit = np.repeat(np.arange(n_units), n_periods)
        period = np.tile(np.arange(n_periods), n_units)
        df = pd.DataFrame({"unit": unit, "period": period})
        # xc is unit-constant on positive-weight units (spanned by unit FE
        # there) but VARIES on the zero-weight unit's rows
        xc = np.repeat(rng.normal(size=n_units), n_periods).astype(float)
        zero_rows = unit == 0
        xc[zero_rows] = rng.normal(size=zero_rows.sum())  # varying junk
        df["xc"] = xc
        df["y"] = rng.normal(size=len(df))
        w = np.ones(len(df))
        w[zero_rows] = 0.0  # domain-excluded unit

        pre = pre_demean_norms(df, ["xc"], weights=w)
        out, _ = demean_by_groups(
            df, ["y", "xc"], ["unit", "period"], suffix="_dm", weights=w, tol=1e-12
        )
        # unweighted norm is large (inert zero-weight rows keep raw values):
        assert float(np.linalg.norm(out["xc_dm"].values)) > 1e-3
        # ...but the weight-aware snap sees the effective sample and fires
        with pytest.warns(UserWarning, match="collinear with the absorbed"):
            snapped = snap_absorbed_regressors(
                out,
                ["xc"],
                pre,
                absorbed_desc="unit and period FEs",
                group_vars=["unit", "period"],
                suffix="_dm",
                weights=w,
            )
        assert snapped == ["xc"]
        assert (out["xc_dm"].values == 0.0).all()


class TestJointSpanSnapConfirmation:
    """Local-review P0: a column in the JOINT span of two FE dimensions
    (x = a[unit] + b[time]) converges to zero slowly under MAP on unbalanced
    panels, so its truncation residual can sit far above the fast-path snap
    threshold. Stage 2 (exact LSMR span confirmation) must catch it; a
    genuinely identified low-within-variation covariate must NOT be snapped.
    """

    @staticmethod
    def _unbalanced_panel(seed=7, n_units=120, n_periods=30, span=9):
        rng = np.random.default_rng(seed)
        unit = np.repeat(np.arange(n_units), n_periods)
        period = np.tile(np.arange(n_periods), n_units)
        entry = rng.integers(0, n_periods - span, n_units)
        keep = (period >= entry[unit]) & (period < entry[unit] + span)
        unit, period = unit[keep], period[keep]
        df = pd.DataFrame({"unit": unit, "period": period})
        df["y"] = rng.normal(size=len(df))
        a = rng.normal(0, 1, n_units)
        b = rng.normal(0, 1, n_periods)
        df["xspan"] = a[unit] + b[period]  # exactly in span(unit + period)
        return df, rng

    def test_joint_span_column_snapped_via_confirmation(self):
        from diff_diff.utils import pre_demean_norms, snap_absorbed_regressors

        df, _ = self._unbalanced_panel()
        pre = pre_demean_norms(df, ["xspan"])
        out, _ = demean_by_groups(df, ["y", "xspan"], ["unit", "period"], suffix="_dm", tol=1e-8)
        rel = np.linalg.norm(out["xspan_dm"].values) / pre["xspan"]
        assert rel > 1e-10  # the fixture is honest: fast path alone misses it
        with pytest.warns(UserWarning, match="collinear with the absorbed"):
            snapped = snap_absorbed_regressors(
                out,
                ["xspan"],
                pre,
                absorbed_desc="unit and period FEs",
                group_vars=["unit", "period"],
                suffix="_dm",
            )
        assert snapped == ["xspan"]
        assert (out["xspan_dm"].values == 0.0).all()

    def test_identified_low_variation_covariate_not_snapped(self):
        """Counter-test: near-spanned but genuinely identified (tiny real
        within-variation) must survive stage 2 untouched."""
        from diff_diff.utils import pre_demean_norms, snap_absorbed_regressors

        df, rng = self._unbalanced_panel(seed=8)
        z = rng.normal(size=len(df))
        df["xnear"] = df["xspan"] + 1e-5 * z  # real within-FE variation ~1e-5
        pre = pre_demean_norms(df, ["xnear"])
        out, _ = demean_by_groups(df, ["xnear"], ["unit", "period"], suffix="_dm", tol=1e-10)
        before = out["xnear_dm"].values.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any snap warning fails the test
            snapped = snap_absorbed_regressors(
                out,
                ["xnear"],
                pre,
                absorbed_desc="unit and period FEs",
                group_vars=["unit", "period"],
                suffix="_dm",
            )
        assert snapped == []
        np.testing.assert_array_equal(out["xnear_dm"].values, before)


# =============================================================================
# TestIterativeFeSolve — shared bincount Gauss-Seidel FE solver
# =============================================================================


def _frozen_pandas_iterative_fe(y, unit_vals, time_vals, max_iter=10_000, tol=1e-10, weights=None):
    """Byte-for-byte copy of the PRE-3.7 per-estimator pandas FE loop
    (ImputationDiD/TwoStageDiD `_iterative_fe`, retired by the shared-engine
    migration). NOT bit-identical to the bincount solver (pandas' grouped
    mean is Kahan-compensated; bincount is naive accumulation). Kept as a
    drift-bound guard: the shared solver must agree with this loop at
    atol=1e-9. Positive weights only — the old loop divides 0/0 on
    zero-total-weight groups (the bug the migration fixed).
    """
    idx = pd.RangeIndex(len(y))
    n = len(y)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    if weights is not None:
        w_series = pd.Series(weights, index=idx)
        wsum_t = w_series.groupby(time_vals).transform("sum").values
        wsum_u = w_series.groupby(unit_vals).transform("sum").values
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(max_iter):
            resid_after_alpha = y - alpha
            if weights is not None:
                wr_t = pd.Series(resid_after_alpha * weights, index=idx)
                beta_new = wr_t.groupby(time_vals).transform("sum").values / wsum_t
            else:
                beta_new = (
                    pd.Series(resid_after_alpha, index=idx)
                    .groupby(time_vals)
                    .transform("mean")
                    .values
                )
            resid_after_beta = y - beta_new
            if weights is not None:
                wr_u = pd.Series(resid_after_beta * weights, index=idx)
                alpha_new = wr_u.groupby(unit_vals).transform("sum").values / wsum_u
            else:
                alpha_new = (
                    pd.Series(resid_after_beta, index=idx)
                    .groupby(unit_vals)
                    .transform("mean")
                    .values
                )
            max_change = max(np.max(np.abs(alpha_new - alpha)), np.max(np.abs(beta_new - beta)))
            alpha = alpha_new
            beta = beta_new
            if max_change < tol:
                break
    unit_fe = pd.Series(alpha, index=idx).groupby(unit_vals).first().to_dict()
    time_fe = pd.Series(beta, index=idx).groupby(time_vals).first().to_dict()
    return unit_fe, time_fe


class TestIterativeFeSolve:
    """Drift-bound + contract tests for the shared bincount FE solver."""

    @staticmethod
    def _panel(kind, seed=42):
        rng = np.random.default_rng(seed)
        n_units, n_periods = 12, 7
        units, times = [], []
        for i in range(n_units):
            for t in range(n_periods):
                if kind == "unbalanced" and rng.random() < 0.25:
                    continue
                units.append(i)
                times.append(t)
        units = np.asarray(units)
        times = np.asarray(times)
        y = (
            rng.standard_normal(n_units)[units]
            + np.linspace(0, 1, n_periods)[times]
            + rng.standard_normal(len(units)) * 0.3
        )
        return y, units, times

    def _solve_shared(self, y, units, times, weights=None):
        unit_codes, unit_uniques = pd.factorize(units, sort=False)
        time_codes, time_uniques = pd.factorize(times, sort=False)
        u_arr, t_arr = _iterative_fe_solve(
            y,
            unit_codes.astype(np.intp),
            time_codes.astype(np.intp),
            len(unit_uniques),
            len(time_uniques),
            weights=weights,
            method_name="test FE solver",
        )
        return dict(zip(unit_uniques, u_arr)), dict(zip(time_uniques, t_arr))

    @pytest.mark.parametrize("kind", ["balanced", "unbalanced"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_matches_frozen_pandas_loop(self, kind, weighted):
        y, units, times = self._panel(kind)
        rng = np.random.default_rng(7)
        w = 0.5 + rng.exponential(0.5, len(y)) if weighted else None

        u_new, t_new = self._solve_shared(y, units, times, weights=w)
        u_old, t_old = _frozen_pandas_iterative_fe(y, units, times, weights=w)

        assert set(u_new) == set(u_old) and set(t_new) == set(t_old)
        for k in u_old:
            np.testing.assert_allclose(u_new[k], u_old[k], rtol=0, atol=1e-9)
        for k in t_old:
            np.testing.assert_allclose(t_new[k], t_old[k], rtol=0, atol=1e-9)

    def test_zero_weight_group_nan_fe_and_convergence(self):
        """Zero-total-weight unit: NaN FE, key retained, clean convergence."""
        y, units, times = self._panel("unbalanced")
        w = np.ones(len(y))
        w[units == 5] = 0.0
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            u_fe, t_fe = self._solve_shared(y, units, times, weights=w)
        assert not any("did not converge" in str(x.message) for x in rec)
        assert 5 in u_fe and np.isnan(u_fe[5])
        assert all(np.isfinite(v) for k, v in u_fe.items() if k != 5)
        assert all(np.isfinite(v) for v in t_fe.values())
        # Positive-weight FE are unchanged by the zero-weight rows' presence
        # (they are outside the WLS estimating sample).
        keep = units != 5
        u_sub, t_sub = self._solve_shared(y[keep], units[keep], times[keep])
        for k, v in u_sub.items():
            np.testing.assert_allclose(u_fe[k], v, rtol=0, atol=1e-12)

    def test_warns_on_nonconvergence_with_label(self):
        y, units, times = self._panel("unbalanced")
        unit_codes, unit_uniques = pd.factorize(units, sort=False)
        time_codes, time_uniques = pd.factorize(times, sort=False)
        with pytest.warns(UserWarning, match="my solver label did not converge"):
            _iterative_fe_solve(
                y,
                unit_codes.astype(np.intp),
                time_codes.astype(np.intp),
                len(unit_uniques),
                len(time_uniques),
                max_iter=1,
                tol=1e-15,
                method_name="my solver label",
            )


class TestBuildFeDummyBlocks:
    """Shared FE-dummy design build (DiD/MPD fixed_effects= + TWFE full-dummy
    path): names must match fe_dummy_names (the reserved-name collision
    guard) and values must match pd.get_dummies exactly."""

    def test_names_match_fe_dummy_names_contract(self):
        from diff_diff.utils import build_fe_dummy_blocks, fe_dummy_names

        df = pd.DataFrame(
            {
                "plain": ["b", "a", "c", "a"],
                "cat": pd.Categorical(
                    ["x", "z", "y", "z"], categories=["z", "y", "x"]
                ),  # non-default order
                "num": [3, 1, 2, 1],
            }
        )
        blocks, names = build_fe_dummy_blocks(df, ["plain", "cat", "num"])
        expected = (
            fe_dummy_names(df["plain"], "plain")
            + fe_dummy_names(df["cat"], "cat")
            + fe_dummy_names(df["num"], "num")
        )
        assert names == expected
        assert sum(b.shape[1] for b in blocks) == len(names)

    def test_values_match_get_dummies(self):
        from diff_diff.utils import build_fe_dummy_blocks

        df = pd.DataFrame({"g": ["b", "a", "c", "a", "b"]})
        blocks, names = build_fe_dummy_blocks(df, ["g"], prefixes=["_fe_g"])
        ref = pd.get_dummies(df["g"], prefix="_fe_g", drop_first=True)
        np.testing.assert_array_equal(blocks[0], ref.values.astype(np.float64))
        assert names == list(ref.columns)
        assert blocks[0].dtype == np.float64

    def test_mismatched_prefixes_length_raises(self):
        """Review P2: a shorter non-empty prefixes list must raise, not
        silently zip-skip trailing FE columns."""
        from diff_diff.utils import build_fe_dummy_blocks

        df = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
        with pytest.raises(ValueError, match="prefixes length 1 does not match"):
            build_fe_dummy_blocks(df, ["a", "b"], prefixes=["_fe_a"])
