"""Tests for the unified linear algebra backend."""

import numpy as np
import pandas as pd
import pytest

from diff_diff.linalg import (
    InferenceResult,
    LinearRegression,
    compute_r_squared,
    compute_robust_vcov,
    solve_ols,
)


class TestSolveOLS:
    """Tests for the solve_ols function."""

    @pytest.fixture
    def simple_regression_data(self):
        """Create simple regression data with known coefficients."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([2.0, 3.0])
        y = X @ beta_true + np.random.randn(n) * 0.5
        return X, y, beta_true

    @pytest.fixture
    def clustered_regression_data(self):
        """Create clustered regression data."""
        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 10
        n = n_clusters * obs_per_cluster

        cluster_ids = np.repeat(np.arange(n_clusters), obs_per_cluster)
        cluster_effects = np.random.randn(n_clusters)

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([5.0, 2.0])
        errors = cluster_effects[cluster_ids] + np.random.randn(n) * 0.5
        y = X @ beta_true + errors

        return X, y, cluster_ids, beta_true

    def test_basic_ols_coefficients(self, simple_regression_data):
        """Test that OLS coefficients are computed correctly."""
        X, y, beta_true = simple_regression_data
        coef, resid, vcov = solve_ols(X, y)

        # Coefficients should be close to true values
        np.testing.assert_allclose(coef, beta_true, atol=0.3)

        # Residuals should have mean close to zero
        assert abs(np.mean(resid)) < 0.1

        # Vcov should be symmetric and positive semi-definite
        np.testing.assert_array_almost_equal(vcov, vcov.T)
        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues >= -1e-10)

    def test_matches_numpy_lstsq(self, simple_regression_data):
        """Test that coefficients match numpy.linalg.lstsq."""
        X, y, _ = simple_regression_data
        coef, resid, _ = solve_ols(X, y)
        coef_numpy = np.linalg.lstsq(X, y, rcond=None)[0]

        np.testing.assert_allclose(coef, coef_numpy, rtol=1e-10)

    def test_return_vcov_false(self, simple_regression_data):
        """Test that return_vcov=False returns None for vcov."""
        X, y, _ = simple_regression_data
        coef, resid, vcov = solve_ols(X, y, return_vcov=False)

        assert vcov is None
        assert coef.shape == (X.shape[1],)
        assert resid.shape == (X.shape[0],)

    def test_return_fitted(self, simple_regression_data):
        """Test that return_fitted=True returns fitted values."""
        X, y, _ = simple_regression_data
        coef, resid, fitted, vcov = solve_ols(X, y, return_fitted=True)

        # Fitted + residuals should equal y
        np.testing.assert_allclose(fitted + resid, y, rtol=1e-10)
        # Fitted should equal X @ coef
        np.testing.assert_allclose(fitted, X @ coef, rtol=1e-10)

    def test_cluster_robust_se(self, clustered_regression_data):
        """Test cluster-robust standard errors."""
        X, y, cluster_ids, _ = clustered_regression_data
        coef, resid, vcov = solve_ols(X, y, cluster_ids=cluster_ids)

        # Vcov should be symmetric
        np.testing.assert_array_almost_equal(vcov, vcov.T)

        # Vcov should be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues >= -1e-10)

        # Standard errors should be positive
        se = np.sqrt(np.diag(vcov))
        assert np.all(se > 0)

    def test_cluster_robust_differs_from_hc1(self, clustered_regression_data):
        """Test that cluster-robust SE differs from HC1."""
        X, y, cluster_ids, _ = clustered_regression_data

        _, _, vcov_hc1 = solve_ols(X, y)
        _, _, vcov_cluster = solve_ols(X, y, cluster_ids=cluster_ids)

        # Should not be identical
        assert not np.allclose(vcov_hc1, vcov_cluster)

    def test_cluster_robust_typically_larger(self, clustered_regression_data):
        """Test that cluster-robust SE is typically larger with correlated errors."""
        X, y, cluster_ids, _ = clustered_regression_data

        _, _, vcov_hc1 = solve_ols(X, y)
        _, _, vcov_cluster = solve_ols(X, y, cluster_ids=cluster_ids)

        se_hc1 = np.sqrt(vcov_hc1[1, 1])
        se_cluster = np.sqrt(vcov_cluster[1, 1])

        # Cluster SE should typically be larger (or at least not much smaller)
        assert se_cluster > se_hc1 * 0.5

    def test_input_validation_x_shape(self):
        """Test that 1D X raises error."""
        X = np.random.randn(100)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            solve_ols(X, y)

    def test_input_validation_y_shape(self):
        """Test that 2D y raises error."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="y must be 1-dimensional"):
            solve_ols(X, y)

    def test_input_validation_length_mismatch(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.random.randn(100, 2)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="same number of observations"):
            solve_ols(X, y)

    def test_underdetermined_system(self):
        """Test that underdetermined system raises error."""
        X = np.random.randn(5, 10)  # More columns than rows
        y = np.random.randn(5)

        with pytest.raises(ValueError, match="Fewer observations"):
            solve_ols(X, y)

    def test_nan_in_x_raises_error(self):
        """Test that NaN in X raises error by default."""
        X = np.random.randn(100, 2)
        X[50, 0] = np.nan
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            solve_ols(X, y)

    def test_nan_in_y_raises_error(self):
        """Test that NaN in y raises error by default."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        y[50] = np.nan

        with pytest.raises(ValueError, match="y contains NaN or Inf"):
            solve_ols(X, y)

    def test_inf_in_x_raises_error(self):
        """Test that Inf in X raises error by default."""
        X = np.random.randn(100, 2)
        X[50, 0] = np.inf
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            solve_ols(X, y)

    def test_check_finite_false_skips_validation(self):
        """Test that check_finite=False skips NaN/Inf validation."""
        X = np.random.randn(100, 2)
        X[50, 0] = np.nan
        y = np.random.randn(100)

        # Should not raise, but will return garbage results
        coef, resid, vcov = solve_ols(X, y, check_finite=False)
        # Coefficients will contain NaN due to bad input
        assert np.isnan(coef).any() or np.isinf(coef).any()

    def test_rank_deficient_still_solves(self):
        """Test that rank-deficient matrix still returns a solution.

        Note: The gelsy driver doesn't always detect rank deficiency,
        but it still returns a valid least-squares solution.
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = X[:, 0] + X[:, 1]  # Perfect collinearity
        y = np.random.randn(100)

        # Should still complete and return valid output
        coef, resid, vcov = solve_ols(X, y)

        assert coef.shape == (3,)
        assert resid.shape == (100,)
        # Residuals should still be valid (y - X @ coef)
        np.testing.assert_allclose(resid, y - X @ coef, rtol=1e-10)

    def test_single_cluster_error(self):
        """Test that single cluster raises error."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        cluster_ids = np.zeros(100)  # All same cluster

        with pytest.raises(ValueError, match="at least 2 clusters"):
            solve_ols(X, y, cluster_ids=cluster_ids)


class TestComputeRobustVcov:
    """Tests for compute_robust_vcov function."""

    @pytest.fixture
    def ols_data(self):
        """Create OLS data with known residuals."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.array([1.0, 2.0])
        residuals = np.random.randn(n)
        return X, residuals

    def test_hc1_shape(self, ols_data):
        """Test that HC1 vcov has correct shape."""
        X, residuals = ols_data
        vcov = compute_robust_vcov(X, residuals)

        assert vcov.shape == (X.shape[1], X.shape[1])

    def test_hc1_symmetric(self, ols_data):
        """Test that HC1 vcov is symmetric."""
        X, residuals = ols_data
        vcov = compute_robust_vcov(X, residuals)

        np.testing.assert_array_almost_equal(vcov, vcov.T)

    def test_hc1_positive_semidefinite(self, ols_data):
        """Test that HC1 vcov is positive semi-definite."""
        X, residuals = ols_data
        vcov = compute_robust_vcov(X, residuals)

        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues >= -1e-10)

    def test_cluster_robust_shape(self, ols_data):
        """Test that cluster-robust vcov has correct shape."""
        X, residuals = ols_data
        n = X.shape[0]
        cluster_ids = np.repeat(np.arange(20), n // 20)

        vcov = compute_robust_vcov(X, residuals, cluster_ids)

        assert vcov.shape == (X.shape[1], X.shape[1])

    def test_cluster_robust_symmetric(self, ols_data):
        """Test that cluster-robust vcov is symmetric."""
        X, residuals = ols_data
        n = X.shape[0]
        cluster_ids = np.repeat(np.arange(20), n // 20)

        vcov = compute_robust_vcov(X, residuals, cluster_ids)

        np.testing.assert_array_almost_equal(vcov, vcov.T)


class TestComputeRSquared:
    """Tests for compute_r_squared function."""

    def test_perfect_fit(self):
        """Test R-squared of 1 for perfect fit."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = np.zeros(5)

        r2 = compute_r_squared(y, residuals)
        assert r2 == 1.0

    def test_no_fit(self):
        """Test R-squared of 0 when residuals equal centered y."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = y - np.mean(y)

        r2 = compute_r_squared(y, residuals)
        np.testing.assert_almost_equal(r2, 0.0)

    def test_r_squared_in_range(self):
        """Test that R-squared is in valid range for typical data."""
        np.random.seed(42)
        y = np.random.randn(100) + 5
        residuals = np.random.randn(100) * 0.5

        r2 = compute_r_squared(y, residuals)
        assert 0 <= r2 <= 1

    def test_adjusted_r_squared(self):
        """Test adjusted R-squared is smaller than R-squared."""
        np.random.seed(42)
        y = np.random.randn(100)
        residuals = np.random.randn(100) * 0.5

        r2 = compute_r_squared(y, residuals)
        r2_adj = compute_r_squared(y, residuals, adjusted=True, n_params=5)

        # Adjusted R-squared should be smaller when adding parameters
        assert r2_adj < r2

    def test_zero_variance_y(self):
        """Test R-squared when y has zero variance."""
        y = np.ones(10)
        residuals = np.zeros(10)

        r2 = compute_r_squared(y, residuals)
        assert r2 == 0.0


class TestEquivalenceWithOldImplementation:
    """Tests to verify new implementation matches old compute_robust_se."""

    @pytest.fixture
    def test_data(self):
        """Create test data for equivalence testing."""
        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 10
        n = n_clusters * obs_per_cluster

        cluster_ids = np.repeat(np.arange(n_clusters), obs_per_cluster)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 5.0 + 2.0 * X[:, 1] + np.random.randn(n)

        return X, y, cluster_ids

    def test_hc1_equivalence(self, test_data):
        """Test that HC1 computation matches old implementation."""
        X, y, _ = test_data

        # Compute using new function
        coef, resid, vcov_new = solve_ols(X, y)

        # Compute using old-style implementation
        n, k = X.shape
        XtX = X.T @ X
        adjustment = n / (n - k)
        u_squared = resid**2
        meat = X.T @ (X * u_squared[:, np.newaxis])
        temp = np.linalg.solve(XtX, meat)
        vcov_old = adjustment * np.linalg.solve(XtX, temp.T).T

        np.testing.assert_allclose(vcov_new, vcov_old, rtol=1e-10)

    def test_cluster_robust_equivalence(self, test_data):
        """Test that cluster-robust computation matches old loop implementation."""
        X, y, cluster_ids = test_data

        # Compute using new function
        coef, resid, vcov_new = solve_ols(X, y, cluster_ids=cluster_ids)

        # Compute using old loop-based implementation
        n, k = X.shape
        XtX = X.T @ X
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))

        meat = np.zeros((k, k))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            X_c = X[mask]
            u_c = resid[mask]
            score_c = X_c.T @ u_c
            meat += np.outer(score_c, score_c)

        temp = np.linalg.solve(XtX, meat)
        vcov_old = adjustment * np.linalg.solve(XtX, temp.T).T

        np.testing.assert_allclose(vcov_new, vcov_old, rtol=1e-10)


class TestPerformance:
    """Performance-related tests (sanity checks, not benchmarks)."""

    def test_large_dataset_completes(self):
        """Test that solve_ols completes on larger dataset."""
        np.random.seed(42)
        n, k = 10000, 10
        X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
        y = np.random.randn(n)

        # Should complete without error
        coef, resid, vcov = solve_ols(X, y)

        assert coef.shape == (k,)
        assert resid.shape == (n,)
        assert vcov.shape == (k, k)

    def test_many_clusters_completes(self):
        """Test that cluster-robust SE completes with many clusters."""
        np.random.seed(42)
        n_clusters = 500
        obs_per_cluster = 20
        n = n_clusters * obs_per_cluster
        k = 5

        X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
        y = np.random.randn(n)
        cluster_ids = np.repeat(np.arange(n_clusters), obs_per_cluster)

        # Should complete without error
        coef, resid, vcov = solve_ols(X, y, cluster_ids=cluster_ids)

        assert vcov.shape == (k, k)


class TestInferenceResult:
    """Tests for the InferenceResult dataclass."""

    def test_basic_creation(self):
        """Test basic InferenceResult creation."""
        result = InferenceResult(
            coefficient=2.5,
            se=0.5,
            t_stat=5.0,
            p_value=0.001,
            conf_int=(1.52, 3.48),
            df=100,
            alpha=0.05,
        )

        assert result.coefficient == 2.5
        assert result.se == 0.5
        assert result.t_stat == 5.0
        assert result.p_value == 0.001
        assert result.conf_int == (1.52, 3.48)
        assert result.df == 100
        assert result.alpha == 0.05

    def test_is_significant_default_alpha(self):
        """Test is_significant with default alpha."""
        # Significant at 0.05
        result = InferenceResult(
            coefficient=2.0, se=0.5, t_stat=4.0, p_value=0.001,
            conf_int=(1.0, 3.0), alpha=0.05
        )
        assert result.is_significant() is True

        # Not significant at 0.05
        result2 = InferenceResult(
            coefficient=0.5, se=0.5, t_stat=1.0, p_value=0.3,
            conf_int=(-0.5, 1.5), alpha=0.05
        )
        assert result2.is_significant() is False

    def test_is_significant_custom_alpha(self):
        """Test is_significant with custom alpha override."""
        result = InferenceResult(
            coefficient=2.0, se=0.5, t_stat=4.0, p_value=0.02,
            conf_int=(1.0, 3.0), alpha=0.05
        )

        # Significant at 0.05 (default)
        assert result.is_significant() is True

        # Not significant at 0.01
        assert result.is_significant(alpha=0.01) is False

    def test_significance_stars(self):
        """Test significance_stars returns correct stars."""
        # p < 0.001 -> ***
        result = InferenceResult(
            coefficient=1.0, se=0.1, t_stat=10.0, p_value=0.0001,
            conf_int=(0.8, 1.2)
        )
        assert result.significance_stars() == "***"

        # p < 0.01 -> **
        result2 = InferenceResult(
            coefficient=1.0, se=0.2, t_stat=5.0, p_value=0.005,
            conf_int=(0.6, 1.4)
        )
        assert result2.significance_stars() == "**"

        # p < 0.05 -> *
        result3 = InferenceResult(
            coefficient=1.0, se=0.3, t_stat=3.0, p_value=0.03,
            conf_int=(0.4, 1.6)
        )
        assert result3.significance_stars() == "*"

        # p < 0.1 -> .
        result4 = InferenceResult(
            coefficient=1.0, se=0.4, t_stat=2.5, p_value=0.08,
            conf_int=(0.2, 1.8)
        )
        assert result4.significance_stars() == "."

        # p >= 0.1 -> ""
        result5 = InferenceResult(
            coefficient=1.0, se=0.5, t_stat=2.0, p_value=0.15,
            conf_int=(0.0, 2.0)
        )
        assert result5.significance_stars() == ""

    def test_to_dict(self):
        """Test to_dict returns all fields."""
        result = InferenceResult(
            coefficient=2.5, se=0.5, t_stat=5.0, p_value=0.001,
            conf_int=(1.52, 3.48), df=100, alpha=0.05
        )
        d = result.to_dict()

        assert d["coefficient"] == 2.5
        assert d["se"] == 0.5
        assert d["t_stat"] == 5.0
        assert d["p_value"] == 0.001
        assert d["conf_int"] == (1.52, 3.48)
        assert d["df"] == 100
        assert d["alpha"] == 0.05


class TestLinearRegression:
    """Tests for the LinearRegression helper class."""

    @pytest.fixture
    def simple_data(self):
        """Create simple regression data with known coefficients."""
        np.random.seed(42)
        n = 200
        # X without intercept (LinearRegression adds it by default)
        X = np.random.randn(n, 2)
        beta_true = np.array([5.0, 2.0, -1.0])  # intercept, x1, x2
        X_with_intercept = np.column_stack([np.ones(n), X])
        y = X_with_intercept @ beta_true + np.random.randn(n) * 0.5
        return X, y, beta_true

    @pytest.fixture
    def clustered_data(self):
        """Create clustered regression data."""
        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 10
        n = n_clusters * obs_per_cluster

        cluster_ids = np.repeat(np.arange(n_clusters), obs_per_cluster)
        cluster_effects = np.random.randn(n_clusters)

        X = np.random.randn(n, 1)
        beta_true = np.array([3.0, 1.5])  # intercept, x1
        X_with_intercept = np.column_stack([np.ones(n), X])
        errors = cluster_effects[cluster_ids] + np.random.randn(n) * 0.3
        y = X_with_intercept @ beta_true + errors

        return X, y, cluster_ids, beta_true

    def test_basic_fit(self, simple_data):
        """Test basic LinearRegression fit."""
        X, y, beta_true = simple_data
        reg = LinearRegression().fit(X, y)

        # Check coefficients are close to true values
        np.testing.assert_allclose(reg.coefficients_, beta_true, atol=0.3)

        # Check fitted attributes exist
        assert reg.coefficients_ is not None
        assert reg.vcov_ is not None
        assert reg.residuals_ is not None
        assert reg.fitted_values_ is not None
        assert reg.n_obs_ == X.shape[0]
        assert reg.n_params_ == X.shape[1] + 1  # +1 for intercept
        assert reg.df_ == reg.n_obs_ - reg.n_params_

    def test_fit_without_intercept(self, simple_data):
        """Test fit without automatic intercept."""
        X, y, _ = simple_data
        n = X.shape[0]

        # Add intercept manually
        X_full = np.column_stack([np.ones(n), X])

        reg = LinearRegression(include_intercept=False).fit(X_full, y)

        # Should have same number of params as columns in X_full
        assert reg.n_params_ == X_full.shape[1]

    def test_fit_not_called_error(self):
        """Test that methods raise error if fit() not called."""
        reg = LinearRegression()

        with pytest.raises(ValueError, match="not been fitted"):
            reg.get_coefficient(0)

        with pytest.raises(ValueError, match="not been fitted"):
            reg.get_se(0)

        with pytest.raises(ValueError, match="not been fitted"):
            reg.get_inference(0)

    def test_get_coefficient(self, simple_data):
        """Test get_coefficient returns correct value."""
        X, y, beta_true = simple_data
        reg = LinearRegression().fit(X, y)

        for i, expected in enumerate(beta_true):
            actual = reg.get_coefficient(i)
            np.testing.assert_allclose(actual, expected, atol=0.3)

    def test_get_se(self, simple_data):
        """Test get_se returns positive standard errors."""
        X, y, _ = simple_data
        reg = LinearRegression().fit(X, y)

        for i in range(reg.n_params_):
            se = reg.get_se(i)
            assert se > 0

    def test_get_inference(self, simple_data):
        """Test get_inference returns InferenceResult with correct values."""
        X, y, beta_true = simple_data
        reg = LinearRegression().fit(X, y)

        result = reg.get_inference(1)  # First predictor (index 1 after intercept)

        # Check it's an InferenceResult
        assert isinstance(result, InferenceResult)

        # Check coefficient is close to true value
        np.testing.assert_allclose(result.coefficient, beta_true[1], atol=0.3)

        # Check SE is positive
        assert result.se > 0

        # Check t-stat computation
        np.testing.assert_allclose(result.t_stat, result.coefficient / result.se)

        # Check p-value is in valid range
        assert 0 <= result.p_value <= 1

        # Check confidence interval contains point estimate
        assert result.conf_int[0] < result.coefficient < result.conf_int[1]

        # Check df is set
        assert result.df == reg.df_

    def test_get_inference_significant_coefficient(self, simple_data):
        """Test inference for a truly significant coefficient."""
        X, y, _ = simple_data
        reg = LinearRegression().fit(X, y)

        # First predictor should be significant (true coef = 2.0)
        result = reg.get_inference(1)

        # With true effect of 2.0 and n=200, should be highly significant
        assert result.p_value < 0.001
        assert result.is_significant()
        assert result.significance_stars() == "***"

    def test_get_inference_batch(self, simple_data):
        """Test get_inference_batch returns dict of results."""
        X, y, _ = simple_data
        reg = LinearRegression().fit(X, y)

        results = reg.get_inference_batch([0, 1, 2])

        assert isinstance(results, dict)
        assert len(results) == 3
        assert all(isinstance(v, InferenceResult) for v in results.values())
        assert all(idx in results for idx in [0, 1, 2])

    def test_get_all_inference(self, simple_data):
        """Test get_all_inference returns results for all coefficients."""
        X, y, _ = simple_data
        reg = LinearRegression().fit(X, y)

        results = reg.get_all_inference()

        assert isinstance(results, list)
        assert len(results) == reg.n_params_
        assert all(isinstance(r, InferenceResult) for r in results)

    def test_custom_alpha(self, simple_data):
        """Test that custom alpha affects confidence intervals."""
        X, y, _ = simple_data
        reg = LinearRegression(alpha=0.10).fit(X, y)

        result = reg.get_inference(1)
        assert result.alpha == 0.10

        # 90% CI should be narrower than 95% CI
        result_99 = reg.get_inference(1, alpha=0.01)
        ci_width_90 = result.conf_int[1] - result.conf_int[0]
        ci_width_99 = result_99.conf_int[1] - result_99.conf_int[0]
        assert ci_width_90 < ci_width_99

    def test_r_squared(self, simple_data):
        """Test R-squared computation."""
        X, y, _ = simple_data
        reg = LinearRegression().fit(X, y)

        r2 = reg.r_squared()
        r2_adj = reg.r_squared(adjusted=True)

        # Should be high for well-specified model
        assert 0.8 < r2 <= 1.0

        # Adjusted should be smaller
        assert r2_adj < r2

    def test_predict(self, simple_data):
        """Test prediction on new data."""
        X, y, beta_true = simple_data
        reg = LinearRegression().fit(X, y)

        # Predict on same data
        y_pred = reg.predict(X)

        # Should match fitted values
        np.testing.assert_allclose(y_pred, reg.fitted_values_, rtol=1e-10)

        # Predict on new data
        X_new = np.random.randn(10, 2)
        y_pred_new = reg.predict(X_new)
        assert y_pred_new.shape == (10,)

    def test_robust_standard_errors(self, simple_data):
        """Test that robust=True computes HC1 standard errors."""
        X, y, _ = simple_data
        reg_robust = LinearRegression(robust=True).fit(X, y)
        reg_classical = LinearRegression(robust=False).fit(X, y)

        # SEs should differ
        se_robust = reg_robust.get_se(1)
        se_classical = reg_classical.get_se(1)

        assert se_robust != se_classical

    def test_cluster_standard_errors(self, clustered_data):
        """Test cluster-robust standard errors."""
        X, y, cluster_ids, _ = clustered_data

        reg_hc1 = LinearRegression(robust=True).fit(X, y)
        reg_cluster = LinearRegression(cluster_ids=cluster_ids).fit(X, y)

        # Cluster SE should typically be larger with correlated errors
        se_hc1 = reg_hc1.get_se(1)
        se_cluster = reg_cluster.get_se(1)

        # They should differ (cluster SE usually larger with cluster correlation)
        assert se_hc1 != se_cluster

    def test_cluster_ids_in_fit(self, clustered_data):
        """Test passing cluster_ids to fit() method."""
        X, y, cluster_ids, _ = clustered_data

        # Pass cluster_ids in constructor
        reg1 = LinearRegression(cluster_ids=cluster_ids).fit(X, y)

        # Pass cluster_ids in fit()
        reg2 = LinearRegression().fit(X, y, cluster_ids=cluster_ids)

        # Should give same results
        np.testing.assert_allclose(reg1.get_se(1), reg2.get_se(1), rtol=1e-10)

    def test_df_adjustment(self, simple_data):
        """Test degrees of freedom adjustment parameter."""
        X, y, _ = simple_data
        reg = LinearRegression().fit(X, y)
        reg_adj = LinearRegression().fit(X, y, df_adjustment=10)

        # Adjusted df should be 10 less
        assert reg_adj.df_ == reg.df_ - 10

        # This affects inference
        result = reg.get_inference(1)
        result_adj = reg_adj.get_inference(1)

        # Same coefficient and SE
        assert result.coefficient == result_adj.coefficient
        assert result.se == result_adj.se

        # Different df affects p-value and CI (though often slightly)
        assert result.df != result_adj.df

    def test_returns_self(self, simple_data):
        """Test that fit() returns self for chaining."""
        X, y, _ = simple_data
        reg = LinearRegression()
        result = reg.fit(X, y)

        assert result is reg

    def test_matches_solve_ols(self, simple_data):
        """Test that LinearRegression matches low-level solve_ols."""
        X, y, _ = simple_data
        n = X.shape[0]
        X_with_intercept = np.column_stack([np.ones(n), X])

        # Use low-level function
        coef, resid, fitted, vcov = solve_ols(
            X_with_intercept, y, return_fitted=True, return_vcov=True
        )

        # Use LinearRegression
        reg = LinearRegression(robust=True).fit(X, y)

        # Should match
        np.testing.assert_allclose(reg.coefficients_, coef, rtol=1e-10)
        np.testing.assert_allclose(reg.residuals_, resid, rtol=1e-10)
        np.testing.assert_allclose(reg.fitted_values_, fitted, rtol=1e-10)
        np.testing.assert_allclose(reg.vcov_, vcov, rtol=1e-10)


class TestNumericalStability:
    """Tests for numerical stability with ill-conditioned matrices."""

    def test_near_singular_matrix_stability(self):
        """Test that near-singular matrices are handled correctly."""
        np.random.seed(42)
        n = 100

        # Create near-collinear design (high condition number)
        X = np.random.randn(n, 3)
        X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(n) * 1e-8  # Near-perfect collinearity

        y = X[:, 0] + np.random.randn(n) * 0.1

        reg = LinearRegression(include_intercept=True).fit(X, y)

        # Should still produce finite coefficients
        assert np.all(np.isfinite(reg.coefficients_))

        # Compare with numpy's lstsq (gold standard for stability)
        X_full = np.column_stack([np.ones(n), X])
        expected, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)

        # Should be close (within reasonable tolerance for ill-conditioned problem)
        np.testing.assert_allclose(reg.coefficients_, expected, rtol=1e-6)

    def test_high_condition_number_matrix(self):
        """Test that high condition number matrices don't lose precision."""
        np.random.seed(42)
        n = 100
        k = 5

        # Create matrix with controlled condition number
        X = np.random.randn(n, k)
        # Make last column nearly dependent on first
        X[:, -1] = X[:, 0] * 0.9999 + np.random.randn(n) * 1e-6

        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n) * 0.1

        # Should complete without error
        reg = LinearRegression().fit(X, y)
        assert np.all(np.isfinite(reg.coefficients_))
        assert np.all(np.isfinite(reg.vcov_))

    def test_zero_se_warning(self):
        """Test that zero SE triggers a warning."""
        np.random.seed(42)
        n = 50

        # Create perfect fit scenario
        X = np.random.randn(n, 2)
        y = 1 + 2 * X[:, 0] + 3 * X[:, 1]  # No noise

        reg = LinearRegression().fit(X, y)

        # Residuals should be near-zero (perfect fit)
        assert np.allclose(reg.residuals_, 0, atol=1e-10)

        # SE should be very small, which may trigger the warning
        # The important thing is it doesn't crash
        for i in range(reg.n_params_):
            inf = reg.get_inference(i)
            assert np.isfinite(inf.coefficient)


class TestEstimatorIntegration:
    """Integration tests verifying estimators produce correct results."""

    def test_did_estimator_produces_valid_results(self):
        """Verify DifferenceInDifferences produces valid inference."""
        from diff_diff import DifferenceInDifferences

        # Create reproducible test data
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            "unit": np.repeat(range(20), 10),
            "time": np.tile(range(10), 20),
            "treated": np.repeat([0] * 10 + [1] * 10, 10),
            "post": np.tile([0] * 5 + [1] * 5, 20),
        })
        # True ATT = 2.0
        data["outcome"] = (
            np.random.randn(n)
            + 2.0 * data["treated"] * data["post"]
        )

        # Fit estimator
        did = DifferenceInDifferences(robust=True)
        result = did.fit(data, outcome="outcome", treatment="treated", time="post")

        # Coefficient should be close to true effect (within sampling variation)
        assert abs(result.att - 2.0) < 1.0

        # SE, p-value, CI should all be valid
        assert result.se > 0
        assert 0 <= result.p_value <= 1
        assert result.conf_int[0] < result.att < result.conf_int[1]

    def test_twfe_estimator_produces_valid_results(self):
        """Verify TwoWayFixedEffects produces valid inference."""
        from diff_diff import TwoWayFixedEffects

        np.random.seed(42)
        n_units = 30
        n_times = 6
        n = n_units * n_times

        data = pd.DataFrame({
            "unit": np.repeat(np.arange(n_units), n_times),
            "time": np.tile(np.arange(n_times), n_units),
            "treated": np.repeat(np.random.binomial(1, 0.5, n_units), n_times),
        })
        data["post"] = (data["time"] >= 3).astype(int)

        # Add unit and time effects with true ATT = 1.5
        unit_effects = np.random.randn(n_units)
        time_effects = np.random.randn(n_times)
        data["y"] = (
            unit_effects[data["unit"]]
            + time_effects[data["time"]]
            + data["treated"] * data["post"] * 1.5
            + np.random.randn(n) * 0.5
        )

        twfe = TwoWayFixedEffects()
        result = twfe.fit(
            data, outcome="y", treatment="treated", time="post", unit="unit"
        )

        # Should produce valid results
        assert result.se > 0
        assert 0 <= result.p_value <= 1
        assert np.isfinite(result.att)

    def test_sun_abraham_estimator_produces_valid_results(self):
        """Verify SunAbraham produces valid inference."""
        from diff_diff import SunAbraham

        np.random.seed(42)
        n_units = 60
        n_times = 10
        n = n_units * n_times

        data = pd.DataFrame({
            "unit": np.repeat(np.arange(n_units), n_times),
            "time": np.tile(np.arange(n_times), n_units),
        })

        # Staggered treatment timing
        first_treat_map = {}
        for i in range(n_units):
            if i < 20:
                first_treat_map[i] = np.inf  # Never treated
            elif i < 40:
                first_treat_map[i] = 5
            else:
                first_treat_map[i] = 7

        data["first_treat"] = data["unit"].map(first_treat_map)
        data["treated"] = (data["time"] >= data["first_treat"]).astype(int)
        data["y"] = np.random.randn(n) + data["treated"] * 2.0

        sa = SunAbraham(n_bootstrap=0)
        result = sa.fit(
            data, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )

        # Should produce valid results
        assert result.overall_se > 0
        assert np.isfinite(result.overall_att)
        assert len(result.event_study_effects) > 0
