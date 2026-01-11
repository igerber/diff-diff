"""Tests for the unified linear algebra backend."""

import numpy as np
import pandas as pd
import pytest

from diff_diff.linalg import compute_r_squared, compute_robust_vcov, solve_ols


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
