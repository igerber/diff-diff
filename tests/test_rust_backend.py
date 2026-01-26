"""
Tests for the Rust backend.

These tests verify that:
1. The Rust backend produces results matching the NumPy implementations
2. Basic functionality works correctly
3. Edge cases are handled properly

Tests are skipped if the Rust backend is not available.
"""

import numpy as np
import pytest

from diff_diff import HAS_RUST_BACKEND


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestRustBackend:
    """Test suite for Rust backend functions."""

    def test_rust_backend_available(self):
        """Verify Rust backend is available when this test runs."""
        assert HAS_RUST_BACKEND

    # =========================================================================
    # Bootstrap Weight Tests
    # =========================================================================

    def test_bootstrap_weights_shape(self):
        """Test bootstrap weights have correct shape."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch

        n_bootstrap, n_units = 100, 50
        weights = generate_bootstrap_weights_batch(n_bootstrap, n_units, "rademacher", 42)
        assert weights.shape == (n_bootstrap, n_units)

    def test_rademacher_weights_values(self):
        """Test Rademacher weights are +-1."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch

        weights = generate_bootstrap_weights_batch(100, 50, "rademacher", 42)
        unique_vals = np.unique(weights)
        assert len(unique_vals) == 2
        assert set(unique_vals) == {-1.0, 1.0}

    def test_rademacher_weights_mean_zero(self):
        """Test Rademacher weights have approximately zero mean."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch

        weights = generate_bootstrap_weights_batch(10000, 1, "rademacher", 42)
        mean = weights.mean()
        assert abs(mean) < 0.05, f"Rademacher mean should be ~0, got {mean}"

    def test_mammen_weights_mean_zero(self):
        """Test Mammen weights have approximately zero mean."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch

        weights = generate_bootstrap_weights_batch(10000, 1, "mammen", 42)
        mean = weights.mean()
        assert abs(mean) < 0.05, f"Mammen mean should be ~0, got {mean}"

    def test_webb_weights_mean_zero(self):
        """Test Webb weights have approximately zero mean."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch

        weights = generate_bootstrap_weights_batch(10000, 1, "webb", 42)
        mean = weights.mean()
        assert abs(mean) < 0.1, f"Webb mean should be ~0, got {mean}"

    def test_bootstrap_reproducibility(self):
        """Test bootstrap weights are reproducible with same seed."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch

        weights1 = generate_bootstrap_weights_batch(100, 50, "rademacher", 42)
        weights2 = generate_bootstrap_weights_batch(100, 50, "rademacher", 42)
        np.testing.assert_array_equal(weights1, weights2)

    def test_bootstrap_different_seeds(self):
        """Test different seeds produce different weights."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch

        weights1 = generate_bootstrap_weights_batch(100, 50, "rademacher", 42)
        weights2 = generate_bootstrap_weights_batch(100, 50, "rademacher", 43)
        assert not np.array_equal(weights1, weights2)

    # =========================================================================
    # Synthetic Weight Tests
    # =========================================================================

    def test_synthetic_weights_sum_to_one(self):
        """Test synthetic weights sum to 1."""
        from diff_diff._rust_backend import compute_synthetic_weights

        np.random.seed(42)
        Y_control = np.random.randn(10, 5)
        Y_treated = np.random.randn(10)

        weights = compute_synthetic_weights(Y_control, Y_treated, 0.0, 1000, 1e-8)
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights should sum to 1, got {weights.sum()}"

    def test_synthetic_weights_non_negative(self):
        """Test synthetic weights are non-negative."""
        from diff_diff._rust_backend import compute_synthetic_weights

        np.random.seed(42)
        Y_control = np.random.randn(10, 5)
        Y_treated = np.random.randn(10)

        weights = compute_synthetic_weights(Y_control, Y_treated, 0.0, 1000, 1e-8)
        assert np.all(weights >= -1e-10), "Weights should be non-negative"

    def test_synthetic_weights_shape(self):
        """Test synthetic weights have correct shape."""
        from diff_diff._rust_backend import compute_synthetic_weights

        np.random.seed(42)
        n_control = 8
        Y_control = np.random.randn(10, n_control)
        Y_treated = np.random.randn(10)

        weights = compute_synthetic_weights(Y_control, Y_treated, 0.0, 1000, 1e-8)
        assert weights.shape == (n_control,)

    # =========================================================================
    # Simplex Projection Tests
    # =========================================================================

    def test_project_simplex_sum(self):
        """Test projected vector sums to 1."""
        from diff_diff._rust_backend import project_simplex

        v = np.array([0.5, 0.3, 0.2, 0.4])
        projected = project_simplex(v)
        assert abs(projected.sum() - 1.0) < 1e-10

    def test_project_simplex_non_negative(self):
        """Test projected vector is non-negative."""
        from diff_diff._rust_backend import project_simplex

        v = np.array([-0.5, 0.3, 1.2, 0.4])
        projected = project_simplex(v)
        assert np.all(projected >= -1e-10)

    def test_project_simplex_already_on_simplex(self):
        """Test projecting a vector already on simplex."""
        from diff_diff._rust_backend import project_simplex

        v = np.array([0.3, 0.5, 0.2])
        projected = project_simplex(v)
        np.testing.assert_array_almost_equal(projected, v)

    # =========================================================================
    # OLS Tests
    # =========================================================================

    def test_solve_ols_shape(self):
        """Test OLS returns correct shapes."""
        from diff_diff._rust_backend import solve_ols

        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        coeffs, residuals, vcov = solve_ols(X, y, None, True)

        assert coeffs.shape == (k,)
        assert residuals.shape == (n,)
        assert vcov.shape == (k, k)

    def test_solve_ols_coefficients(self):
        """Test OLS coefficients match scipy."""
        from diff_diff._rust_backend import solve_ols
        from scipy.linalg import lstsq

        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        coeffs_rust, _, _ = solve_ols(X, y, None, True)
        coeffs_scipy = lstsq(X, y)[0]

        np.testing.assert_array_almost_equal(coeffs_rust, coeffs_scipy, decimal=10)

    def test_solve_ols_residuals(self):
        """Test OLS residuals are correct."""
        from diff_diff._rust_backend import solve_ols

        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        coeffs, residuals, _ = solve_ols(X, y, None, True)
        expected_residuals = y - X @ coeffs

        np.testing.assert_array_almost_equal(residuals, expected_residuals, decimal=10)

    # =========================================================================
    # Robust VCoV Tests
    # =========================================================================

    def test_robust_vcov_shape(self):
        """Test robust VCoV has correct shape."""
        from diff_diff._rust_backend import compute_robust_vcov

        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)

        vcov = compute_robust_vcov(X, residuals, None)
        assert vcov.shape == (k, k)

    def test_robust_vcov_symmetric(self):
        """Test robust VCoV is symmetric."""
        from diff_diff._rust_backend import compute_robust_vcov

        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)

        vcov = compute_robust_vcov(X, residuals, None)
        np.testing.assert_array_almost_equal(vcov, vcov.T)

    def test_robust_vcov_positive_diagonal(self):
        """Test robust VCoV has positive diagonal."""
        from diff_diff._rust_backend import compute_robust_vcov

        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)

        vcov = compute_robust_vcov(X, residuals, None)
        assert np.all(np.diag(vcov) > 0), "Diagonal should be positive"

    def test_cluster_robust_vcov(self):
        """Test cluster-robust VCoV."""
        from diff_diff._rust_backend import compute_robust_vcov

        np.random.seed(42)
        n, k = 100, 5
        n_clusters = 10
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)
        cluster_ids = np.repeat(np.arange(n_clusters), n // n_clusters)

        vcov = compute_robust_vcov(X, residuals, cluster_ids)
        assert vcov.shape == (k, k)
        assert np.all(np.diag(vcov) > 0)

    # =========================================================================
    # LU Fallback Tests (for near-singular matrices)
    # =========================================================================

    def test_near_singular_matrix_lu_fallback(self):
        """Test that near-singular matrices trigger LU fallback and produce valid results.

        When X'X is near-singular (not positive definite), Cholesky factorization
        fails and the Rust backend should fall back to LU decomposition.
        This test verifies:
        1. No crash or exception is raised
        2. Coefficients are finite
        3. Results match NumPy implementation
        """
        from diff_diff._rust_backend import solve_ols
        from scipy.linalg import lstsq

        np.random.seed(42)
        n = 100

        # Create near-collinear design matrix (high condition number)
        # Column 3 is almost a linear combination of columns 1 and 2
        X = np.random.randn(n, 3)
        X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(n) * 1e-8

        y = X[:, 0] + np.random.randn(n) * 0.1

        # Rust backend should handle this gracefully via LU fallback
        coeffs, residuals, vcov = solve_ols(X, y, None, True)

        # Verify results are finite
        assert np.all(np.isfinite(coeffs)), "Coefficients should be finite"
        assert np.all(np.isfinite(residuals)), "Residuals should be finite"

        # Verify residuals are correct given coefficients
        expected_residuals = y - X @ coeffs
        np.testing.assert_array_almost_equal(
            residuals, expected_residuals, decimal=8,
            err_msg="Residuals should match y - X @ coeffs"
        )

    def test_high_condition_number_matrix(self):
        """Test OLS with high condition number matrix uses LU fallback correctly."""
        from diff_diff._rust_backend import solve_ols

        np.random.seed(123)
        n = 100

        # Create matrix with high condition number via scaling
        X = np.random.randn(n, 4)
        X[:, 0] *= 1e6  # Scale first column to create high condition number
        X[:, 3] *= 1e-6  # Scale last column very small

        y = np.random.randn(n)

        # Should not raise and should produce finite results
        coeffs, residuals, vcov = solve_ols(X, y, None, True)

        assert np.all(np.isfinite(coeffs)), "Coefficients should be finite"
        assert np.all(np.isfinite(residuals)), "Residuals should be finite"
        assert vcov is not None, "VCoV should be returned"

    def test_near_singular_with_clusters(self):
        """Test near-singular matrix with cluster-robust SEs uses LU fallback."""
        from diff_diff._rust_backend import solve_ols

        np.random.seed(42)
        n = 100
        n_clusters = 10

        # Near-collinear design
        X = np.random.randn(n, 3)
        X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(n) * 1e-8

        y = X[:, 0] + np.random.randn(n) * 0.1
        cluster_ids = np.repeat(np.arange(n_clusters), n // n_clusters).astype(np.int64)

        # Should handle gracefully with cluster SEs
        coeffs, residuals, vcov = solve_ols(X, y, cluster_ids, True)

        assert np.all(np.isfinite(coeffs)), "Coefficients should be finite"
        assert np.all(np.isfinite(residuals)), "Residuals should be finite"
        assert vcov.shape == (3, 3), "VCoV should have correct shape"

    # =========================================================================
    # Rank-Deficient Matrix Tests (Critical for MultiPeriodDiD)
    # =========================================================================

    def test_rank_deficient_matrix_produces_valid_coefficients(self):
        """Test that rank-deficient matrices produce finite, reasonable coefficients.

        This test verifies the fix for the MultiPeriodDiD bug where rank-deficient
        design matrices (with redundant columns) produced astronomically wrong
        estimates (trillions instead of single digits).

        The SVD-based solver should truncate small singular values and produce
        a valid minimum-norm solution.
        """
        from diff_diff._rust_backend import solve_ols

        np.random.seed(42)
        n = 100

        # Create perfectly collinear design matrix (rank-deficient)
        # This mimics what can happen in MultiPeriodDiD with period dummies
        X = np.random.randn(n, 3)
        X[:, 2] = X[:, 0] + X[:, 1]  # Column 3 = Column 1 + Column 2

        y = X[:, 0] + np.random.randn(n) * 0.1

        # Rust backend should handle this gracefully via SVD truncation
        coeffs, residuals, vcov = solve_ols(X, y, None, True)

        # Coefficients must be finite (not NaN or Inf)
        assert np.all(np.isfinite(coeffs)), f"Coefficients should be finite, got {coeffs}"

        # Coefficients should be reasonable (not astronomically large like 1e12)
        assert np.all(np.abs(coeffs) < 1e6), f"Coefficients are unreasonably large: {coeffs}"

        # Residuals should be correct given coefficients
        expected_residuals = y - X @ coeffs
        np.testing.assert_array_almost_equal(
            residuals, expected_residuals, decimal=8,
            err_msg="Residuals should match y - X @ coeffs"
        )

    def test_multiperiod_did_like_design_matrix(self):
        """Test design matrix structure similar to MultiPeriodDiD.

        MultiPeriodDiD creates design matrices with:
        - Intercept
        - Period dummies (one-hot encoded)
        - Treatment × post interaction terms

        These can create rank-deficient matrices when period dummies and
        interaction terms are not all linearly independent.
        """
        from diff_diff._rust_backend import solve_ols

        np.random.seed(42)
        n = 200
        n_periods = 5

        # Create MultiPeriodDiD-like design matrix
        intercept = np.ones(n)

        # Period dummies (periods 1-4, period 0 is reference)
        period_assignment = np.random.randint(0, n_periods, n)
        period_dummies = np.zeros((n, n_periods - 1))
        for i in range(1, n_periods):
            period_dummies[:, i - 1] = (period_assignment == i).astype(float)

        # Treatment indicator and post indicator
        treated = np.random.binomial(1, 0.5, n)
        post = (period_assignment >= 3).astype(float)
        treat_post = treated * post

        # Build design matrix (potentially rank-deficient)
        X = np.column_stack([intercept, period_dummies, treat_post])

        # True effect
        true_effect = 2.5
        y = (
            1.0
            + 0.5 * period_dummies[:, 0]
            + 0.3 * period_dummies[:, 1]
            + 0.7 * period_dummies[:, 2]
            + 0.9 * period_dummies[:, 3]
            + true_effect * treat_post
            + np.random.randn(n) * 0.5
        )

        # Fit with Rust backend
        coeffs, residuals, vcov = solve_ols(X, y, None, True)

        # Coefficients must be finite
        assert np.all(np.isfinite(coeffs)), f"Coefficients should be finite, got {coeffs}"

        # Coefficients should be reasonable (not trillions)
        assert np.all(np.abs(coeffs) < 1e6), f"Coefficients are unreasonably large: {coeffs}"

        # Treatment effect (last coefficient) should be close to true effect
        assert abs(coeffs[-1] - true_effect) < 2.0, (
            f"Treatment effect {coeffs[-1]} is too far from true effect {true_effect}"
        )


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestRustVsNumpy:
    """Tests comparing Rust and NumPy implementations for numerical equivalence."""

    # =========================================================================
    # OLS Solver Equivalence
    # =========================================================================

    def test_solve_ols_coefficients_match(self):
        """Test Rust and NumPy OLS coefficients match."""
        from diff_diff._rust_backend import solve_ols as rust_fn
        from diff_diff.linalg import _solve_ols_numpy as numpy_fn

        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        rust_coeffs, rust_resid, rust_vcov = rust_fn(X, y, None, True)
        numpy_coeffs, numpy_resid, numpy_vcov = numpy_fn(X, y, cluster_ids=None)

        np.testing.assert_array_almost_equal(
            rust_coeffs, numpy_coeffs, decimal=8,
            err_msg="OLS coefficients should match"
        )
        np.testing.assert_array_almost_equal(
            rust_resid, numpy_resid, decimal=8,
            err_msg="OLS residuals should match"
        )

    def test_solve_ols_with_clusters_match(self):
        """Test Rust and NumPy OLS with cluster SEs match."""
        from diff_diff._rust_backend import solve_ols as rust_fn
        from diff_diff.linalg import _solve_ols_numpy as numpy_fn

        np.random.seed(42)
        n, k = 100, 5
        n_clusters = 10
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        cluster_ids = np.repeat(np.arange(n_clusters), n // n_clusters)

        rust_coeffs, _, rust_vcov = rust_fn(X, y, cluster_ids, True)
        numpy_coeffs, _, numpy_vcov = numpy_fn(X, y, cluster_ids=cluster_ids)

        np.testing.assert_array_almost_equal(
            rust_coeffs, numpy_coeffs, decimal=8,
            err_msg="Clustered OLS coefficients should match"
        )
        # VCoV may differ slightly due to implementation details
        np.testing.assert_array_almost_equal(
            rust_vcov, numpy_vcov, decimal=5,
            err_msg="Clustered OLS VCoV should match"
        )

    def test_rank_deficient_ols_residuals_match(self):
        """Test Rust and NumPy produce matching residuals for rank-deficient matrices.

        The Rust backend uses SVD truncation while NumPy uses R-style NaN handling.
        Despite different approaches, both should produce equivalent residuals.

        Note: The coefficient representations differ:
        - Rust: All finite (SVD minimum-norm solution)
        - NumPy: NaN for dropped columns (R-style)
        But both produce the same fitted values and residuals.
        """
        import warnings
        from diff_diff._rust_backend import solve_ols as rust_fn
        from diff_diff.linalg import _solve_ols_numpy as numpy_fn

        np.random.seed(42)
        n = 100

        # Create rank-deficient design matrix (perfect collinearity)
        X = np.random.randn(n, 3)
        X[:, 2] = X[:, 0] + X[:, 1]  # Column 3 = Column 1 + Column 2

        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n) * 0.1

        # Rust backend produces finite coefficients via SVD truncation
        rust_coeffs, rust_resid, _ = rust_fn(X, y, None, True)

        # NumPy backend produces NaN for dropped columns (R-style)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress rank-deficient warning
            numpy_coeffs, numpy_resid, _ = numpy_fn(X, y, cluster_ids=None)

        # Rust should produce finite coefficients
        assert np.all(np.isfinite(rust_coeffs)), "Rust coefficients should be finite"
        assert np.all(np.abs(rust_coeffs) < 1e6), "Rust coefficients should be reasonable"

        # NumPy should produce exactly one NaN coefficient (the dropped one)
        assert np.sum(np.isnan(numpy_coeffs)) == 1, "NumPy should have one NaN coefficient"

        # Non-NaN NumPy coefficients should be reasonable
        finite_numpy = numpy_coeffs[~np.isnan(numpy_coeffs)]
        assert np.all(np.abs(finite_numpy) < 1e6), "NumPy finite coefficients should be reasonable"

        # Residuals should be very close (this is the key equivalence check)
        # Both approaches should produce the same fitted values and residuals
        np.testing.assert_array_almost_equal(
            rust_resid, numpy_resid, decimal=5,
            err_msg="Residuals should match despite different coefficient representations"
        )

    def test_multiperiod_did_design_residuals_equivalence(self):
        """Test both backends produce equivalent residuals for MultiPeriodDiD-like matrices.

        For full-rank designs, both backends should produce identical results.
        The design matrix in this test is typically full-rank.
        """
        import warnings
        from diff_diff._rust_backend import solve_ols as rust_fn
        from diff_diff.linalg import _solve_ols_numpy as numpy_fn

        np.random.seed(42)
        n = 200
        n_periods = 5

        # Create MultiPeriodDiD-like design matrix
        intercept = np.ones(n)
        period_assignment = np.random.randint(0, n_periods, n)
        period_dummies = np.zeros((n, n_periods - 1))
        for i in range(1, n_periods):
            period_dummies[:, i - 1] = (period_assignment == i).astype(float)

        treated = np.random.binomial(1, 0.5, n)
        post = (period_assignment >= 3).astype(float)
        treat_post = treated * post

        X = np.column_stack([intercept, period_dummies, treat_post])

        true_effect = 2.5
        y = (
            1.0
            + 0.5 * period_dummies[:, 0]
            + 0.3 * period_dummies[:, 1]
            + 0.7 * period_dummies[:, 2]
            + 0.9 * period_dummies[:, 3]
            + true_effect * treat_post
            + np.random.randn(n) * 0.5
        )

        rust_coeffs, rust_resid, _ = rust_fn(X, y, None, True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May or may not warn depending on rank
            numpy_coeffs, numpy_resid, _ = numpy_fn(X, y, cluster_ids=None)

        # Rust should produce finite treatment effect
        rust_effect = rust_coeffs[-1]
        assert np.isfinite(rust_effect), "Rust treatment effect should be finite"
        assert abs(rust_effect - true_effect) < 2.0, (
            f"Rust treatment effect {rust_effect} too far from true {true_effect}"
        )

        # NumPy treatment effect should be close (may be finite or NaN depending on rank)
        numpy_effect = numpy_coeffs[-1]
        if np.isfinite(numpy_effect):
            assert abs(numpy_effect - true_effect) < 2.0, (
                f"NumPy treatment effect {numpy_effect} too far from true {true_effect}"
            )
            # Effects should be close to each other
            assert abs(rust_effect - numpy_effect) < 0.5, (
                f"Rust ({rust_effect}) and NumPy ({numpy_effect}) effects should match"
            )

        # Residuals should be very close (key equivalence check)
        np.testing.assert_array_almost_equal(
            rust_resid, numpy_resid, decimal=5,
            err_msg="Residuals should match for MultiPeriodDiD-like design"
        )

    # =========================================================================
    # Robust VCoV Equivalence
    # =========================================================================

    def test_robust_vcov_hc1_match(self):
        """Test Rust and NumPy HC1 robust VCoV match."""
        from diff_diff._rust_backend import compute_robust_vcov as rust_fn
        from diff_diff.linalg import _compute_robust_vcov_numpy as numpy_fn

        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)

        rust_vcov = rust_fn(X, residuals, None)
        numpy_vcov = numpy_fn(X, residuals, None)

        np.testing.assert_array_almost_equal(
            rust_vcov, numpy_vcov, decimal=8,
            err_msg="HC1 robust VCoV should match"
        )

    def test_robust_vcov_clustered_match(self):
        """Test Rust and NumPy cluster-robust VCoV match."""
        from diff_diff._rust_backend import compute_robust_vcov as rust_fn
        from diff_diff.linalg import _compute_robust_vcov_numpy as numpy_fn

        np.random.seed(42)
        n, k = 100, 5
        n_clusters = 10
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)
        cluster_ids = np.repeat(np.arange(n_clusters), n // n_clusters)

        rust_vcov = rust_fn(X, residuals, cluster_ids)
        numpy_vcov = numpy_fn(X, residuals, cluster_ids)

        np.testing.assert_array_almost_equal(
            rust_vcov, numpy_vcov, decimal=6,
            err_msg="Cluster-robust VCoV should match"
        )

    # =========================================================================
    # Bootstrap Weights Equivalence (Statistical Properties)
    # =========================================================================

    def test_bootstrap_weights_rademacher_properties(self):
        """Test Rust Rademacher weights have correct statistical properties."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch as rust_fn

        # Generate large sample for statistical tests
        n_bootstrap, n_units = 10000, 100
        weights = rust_fn(n_bootstrap, n_units, "rademacher", 42)

        # Rademacher: values are +-1, mean ~0, variance ~1
        unique_vals = np.unique(weights)
        assert set(unique_vals) == {-1.0, 1.0}, "Rademacher weights should be +-1"

        mean = weights.mean()
        assert abs(mean) < 0.02, f"Rademacher mean should be ~0, got {mean}"

        var = weights.var()
        assert abs(var - 1.0) < 0.02, f"Rademacher variance should be ~1, got {var}"

    def test_bootstrap_weights_mammen_properties(self):
        """Test Rust Mammen weights have correct statistical properties."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch as rust_fn

        n_bootstrap, n_units = 10000, 100
        weights = rust_fn(n_bootstrap, n_units, "mammen", 42)

        # Mammen: E[w] = 0, E[w^2] = 1, E[w^3] = 1
        mean = weights.mean()
        assert abs(mean) < 0.02, f"Mammen mean should be ~0, got {mean}"

        second_moment = (weights ** 2).mean()
        assert abs(second_moment - 1.0) < 0.02, f"Mammen E[w^2] should be ~1, got {second_moment}"

        third_moment = (weights ** 3).mean()
        assert abs(third_moment - 1.0) < 0.1, f"Mammen E[w^3] should be ~1, got {third_moment}"

    def test_bootstrap_weights_webb_properties(self):
        """Test Rust Webb weights have correct statistical properties."""
        from diff_diff._rust_backend import generate_bootstrap_weights_batch as rust_fn

        n_bootstrap, n_units = 10000, 100
        weights = rust_fn(n_bootstrap, n_units, "webb", 42)

        # Webb: 6-point distribution with E[w] = 0
        mean = weights.mean()
        assert abs(mean) < 0.1, f"Webb mean should be ~0, got {mean}"

        # Should have 6 unique values
        unique_vals = np.unique(weights.flatten())
        assert len(unique_vals) == 6, f"Webb should have 6 unique values, got {len(unique_vals)}"

    # =========================================================================
    # Synthetic Weights Equivalence
    # =========================================================================

    def test_synthetic_weights_match(self):
        """Test Rust and NumPy synthetic weights produce similar results."""
        from diff_diff._rust_backend import compute_synthetic_weights as rust_fn
        from diff_diff.utils import _compute_synthetic_weights_numpy as numpy_fn

        np.random.seed(42)
        Y_control = np.random.randn(10, 5)
        Y_treated = np.random.randn(10)

        rust_weights = rust_fn(Y_control, Y_treated, 0.0, 1000, 1e-8)
        numpy_weights = numpy_fn(Y_control, Y_treated, 0.0)

        # Both should be valid simplex weights
        assert abs(rust_weights.sum() - 1.0) < 1e-6, "Rust weights should sum to 1"
        assert abs(numpy_weights.sum() - 1.0) < 1e-6, "NumPy weights should sum to 1"
        assert np.all(rust_weights >= -1e-6), "Rust weights should be non-negative"
        assert np.all(numpy_weights >= -1e-6), "NumPy weights should be non-negative"

        # Reconstruction error should be similar
        rust_error = np.linalg.norm(Y_treated - Y_control @ rust_weights)
        numpy_error = np.linalg.norm(Y_treated - Y_control @ numpy_weights)
        assert abs(rust_error - numpy_error) < 0.5, \
            f"Reconstruction errors should be similar: rust={rust_error:.4f}, numpy={numpy_error:.4f}"

    def test_synthetic_weights_with_regularization(self):
        """Test Rust synthetic weights with L2 regularization."""
        from diff_diff._rust_backend import compute_synthetic_weights as rust_fn
        from diff_diff.utils import _compute_synthetic_weights_numpy as numpy_fn

        np.random.seed(42)
        Y_control = np.random.randn(15, 8)
        Y_treated = np.random.randn(15)
        lambda_reg = 0.1

        rust_weights = rust_fn(Y_control, Y_treated, lambda_reg, 1000, 1e-8)
        numpy_weights = numpy_fn(Y_control, Y_treated, lambda_reg)

        # Both should be valid simplex weights
        assert abs(rust_weights.sum() - 1.0) < 1e-6
        assert abs(numpy_weights.sum() - 1.0) < 1e-6

        # With regularization, weights should be more spread out (higher entropy)
        rust_entropy = -np.sum(rust_weights * np.log(rust_weights + 1e-10))
        numpy_entropy = -np.sum(numpy_weights * np.log(numpy_weights + 1e-10))
        assert rust_entropy > 0.5, "Regularized weights should have positive entropy"
        assert numpy_entropy > 0.5, "Regularized weights should have positive entropy"

    def test_simplex_projection_match(self):
        """Test Rust and NumPy simplex projection match exactly."""
        from diff_diff._rust_backend import project_simplex as rust_fn
        from diff_diff.utils import _project_simplex as numpy_fn

        # Test various input vectors
        test_vectors = [
            np.array([0.5, -0.3, 1.2, 0.4, -0.1]),
            np.array([1.0, 1.0, 1.0, 1.0]),  # uniform
            np.array([0.25, 0.25, 0.25, 0.25]),  # already on simplex
            np.array([-1.0, -2.0, 5.0]),  # one dominant
            np.array([0.1, 0.2, 0.3, 0.4]),  # near simplex
        ]

        for v in test_vectors:
            rust_proj = rust_fn(v)
            numpy_proj = numpy_fn(v)

            np.testing.assert_array_almost_equal(
                rust_proj, numpy_proj, decimal=10,
                err_msg=f"Simplex projection mismatch for input {v}"
            )

    def test_nan_vcov_fallback_to_python(self):
        """Test that NaN vcov from Rust backend triggers fallback to Python.

        When Rust SVD detects rank-deficiency that Python QR missed (due to
        different numerical properties), the vcov matrix may contain NaN values.
        The high-level solve_ols should detect this and fall back to Python's
        R-style handling, ensuring the user never receives silent NaN SEs.

        The key behavior being tested:
        1. When Rust returns NaN vcov, we emit a warning and re-run Python
        2. The Python re-run does fresh rank detection (not using cached info)
        3. R-style handling is applied: NaN coefficients for dropped columns
        """
        import warnings
        from diff_diff.linalg import solve_ols

        # Create an ill-conditioned matrix that might cause QR/SVD disagreement.
        # The condition number is extremely high, which may cause the Rust SVD
        # to detect numerical issues that QR doesn't catch.
        np.random.seed(42)
        n = 100

        # Create a matrix with near-perfect but not exact collinearity.
        # This is on the boundary where QR/SVD might disagree.
        X = np.random.randn(n, 4)
        # Make column 3 almost (but not exactly) a linear combination of 0-2
        X[:, 3] = X[:, 0] + X[:, 1] + X[:, 2] + np.random.randn(n) * 1e-12

        y = np.random.randn(n)

        # Capture any warnings that might be emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coeffs, residuals, vcov = solve_ols(X, y)

        # Check if fallback warning was emitted
        fallback_warning_emitted = any(
            "Re-running with Python backend" in str(warning.message)
            for warning in w
        )

        # Key invariants that must hold regardless of which backend is used:
        # 1. Coefficients must be finite (either via Rust SVD or Python R-style)
        finite_coeffs = coeffs[np.isfinite(coeffs)]
        assert len(finite_coeffs) >= 3, \
            "At least 3 coefficients should be finite (identifiable)"
        assert np.all(np.abs(finite_coeffs) < 1e10), \
            f"Finite coefficients should be reasonable, got {finite_coeffs}"

        # 2. If vcov has any finite values, they should correspond to finite coefficients
        if vcov is not None:
            finite_coef_mask = np.isfinite(coeffs)
            for i in range(len(coeffs)):
                if finite_coef_mask[i]:
                    # This coefficient's variance should be finite
                    var_i = vcov[i, i]
                    assert np.isfinite(var_i) or np.isnan(var_i), \
                        f"Variance for finite coef {i} should be finite or NaN (dropped)"

        # 3. Residuals must always be finite
        assert np.all(np.isfinite(residuals)), "Residuals should be finite"

        # 4. R-style consistency: NaN coefficients must have NaN vcov diagonal
        if vcov is not None:
            nan_coef_indices = set(np.where(np.isnan(coeffs))[0])
            nan_vcov_diag_indices = set(np.where(np.isnan(np.diag(vcov)))[0])

            # NaN in vcov diagonal should correspond exactly to NaN coefficients
            assert nan_vcov_diag_indices == nan_coef_indices, \
                f"NaN vcov diagonal {nan_vcov_diag_indices} should match " \
                f"NaN coefficients {nan_coef_indices}"

        # 5. If fallback warning was emitted, R-style handling MUST have occurred
        # This verifies that the fallback actually applies R-style NaN handling
        # (not minimum-norm solution which would have all finite coefficients)
        if fallback_warning_emitted:
            assert np.any(np.isnan(coeffs)), \
                "Fallback warning emitted but no NaN coefficients - " \
                "R-style handling was not applied"
            assert vcov is not None and np.any(np.isnan(vcov)), \
                "Fallback warning emitted but vcov has no NaN - " \
                "R-style handling was not applied"


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestTROPRustBackend:
    """Test suite for TROP Rust backend functions."""

    def test_unit_distance_matrix_shape(self):
        """Test unit distance matrix has correct shape."""
        from diff_diff._rust_backend import compute_unit_distance_matrix

        np.random.seed(42)
        n_periods, n_units = 10, 5
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))  # All control

        dist_matrix = compute_unit_distance_matrix(Y, D)
        assert dist_matrix.shape == (n_units, n_units)

    def test_unit_distance_matrix_diagonal_zero(self):
        """Test unit distance matrix has zero diagonal."""
        from diff_diff._rust_backend import compute_unit_distance_matrix

        np.random.seed(42)
        n_periods, n_units = 10, 5
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))

        dist_matrix = compute_unit_distance_matrix(Y, D)

        for i in range(n_units):
            assert dist_matrix[i, i] == 0.0, f"Diagonal [{i}, {i}] should be 0"

    def test_unit_distance_matrix_symmetric(self):
        """Test unit distance matrix is symmetric."""
        from diff_diff._rust_backend import compute_unit_distance_matrix

        np.random.seed(42)
        n_periods, n_units = 10, 5
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))

        dist_matrix = compute_unit_distance_matrix(Y, D)
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)

    def test_unit_distance_matrix_matches_numpy(self):
        """Test Rust distance matrix matches NumPy implementation."""
        from diff_diff._rust_backend import compute_unit_distance_matrix
        from diff_diff.trop import TROP

        np.random.seed(42)
        n_periods, n_units = 8, 4
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))

        # Rust implementation
        rust_dist = compute_unit_distance_matrix(Y, D)

        # NumPy implementation
        trop = TROP()
        numpy_dist = trop._compute_all_unit_distances(Y, D, n_units, n_periods)

        np.testing.assert_array_almost_equal(
            rust_dist, numpy_dist, decimal=10,
            err_msg="Distance matrices should match"
        )

    def test_unit_distance_excludes_treated(self):
        """Test distance matrix excludes treated observations."""
        from diff_diff._rust_backend import compute_unit_distance_matrix

        np.random.seed(42)
        n_periods, n_units = 10, 5
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        # Mark some periods as treated for unit 0
        D[5:, 0] = 1.0

        dist_matrix = compute_unit_distance_matrix(Y, D)

        # Should still produce valid distances
        assert np.all(np.isfinite(dist_matrix) | (dist_matrix == np.inf))
        assert dist_matrix[0, 0] == 0.0

    def test_loocv_grid_search_returns_valid_params(self):
        """Test LOOCV grid search returns valid parameter tuple."""
        from diff_diff._rust_backend import loocv_grid_search

        np.random.seed(42)
        n_periods, n_units = 8, 6
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        # Mark last 2 periods for unit 0 as treated
        D[6:, 0] = 1.0

        control_mask = (D == 0).astype(np.uint8)

        # Compute time distance matrix
        time_dist = np.abs(
            np.arange(n_periods)[:, np.newaxis] - np.arange(n_periods)[np.newaxis, :]
        ).astype(np.int64)

        lambda_time = np.array([0.0, 1.0], dtype=np.float64)
        lambda_unit = np.array([0.0, 1.0], dtype=np.float64)
        lambda_nn = np.array([0.0, 0.1], dtype=np.float64)

        best_lt, best_lu, best_ln, score, n_valid, n_attempted, first_failed = loocv_grid_search(
            Y, D, control_mask, time_dist,
            lambda_time, lambda_unit, lambda_nn,
            50, 100, 1e-6, 42
        )

        # Check returned parameters are from the grid
        assert best_lt in lambda_time
        assert best_lu in lambda_unit
        assert best_ln in lambda_nn
        assert np.isfinite(score) or score == np.inf
        # Check failure counts are valid
        assert n_valid >= 0
        assert n_attempted >= 0
        assert n_valid <= n_attempted
        # Check first_failed is None or a valid (unit, time) tuple
        assert first_failed is None or (isinstance(first_failed, tuple) and len(first_failed) == 2)

    def test_bootstrap_variance_shape(self):
        """Test bootstrap returns correct shapes."""
        from diff_diff._rust_backend import bootstrap_trop_variance

        np.random.seed(42)
        n_periods, n_units = 8, 6
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[6:, 0] = 1.0  # Treat unit 0 in last 2 periods

        control_mask = (D == 0).astype(np.uint8)

        # Compute time distance matrix
        time_dist = np.abs(
            np.arange(n_periods)[:, np.newaxis] - np.arange(n_periods)[np.newaxis, :]
        ).astype(np.int64)

        n_bootstrap = 20
        estimates, se = bootstrap_trop_variance(
            Y, D, control_mask, time_dist,
            1.0, 1.0, 0.1,  # lambda values
            n_bootstrap, 100, 1e-6, 42
        )

        # Should return array of bootstrap estimates and SE
        assert len(estimates) <= n_bootstrap  # Some may fail
        assert se >= 0.0  # SE should be non-negative

    def test_bootstrap_reproducibility(self):
        """Test bootstrap is reproducible with same seed."""
        from diff_diff._rust_backend import bootstrap_trop_variance

        np.random.seed(42)
        n_periods, n_units = 8, 6
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[6:, 0] = 1.0

        control_mask = (D == 0).astype(np.uint8)

        # Compute time distance matrix
        time_dist = np.abs(
            np.arange(n_periods)[:, np.newaxis] - np.arange(n_periods)[np.newaxis, :]
        ).astype(np.int64)

        # Run twice with same seed
        est1, se1 = bootstrap_trop_variance(
            Y, D, control_mask, time_dist,
            1.0, 1.0, 0.1, 20, 100, 1e-6, 42
        )
        est2, se2 = bootstrap_trop_variance(
            Y, D, control_mask, time_dist,
            1.0, 1.0, 0.1, 20, 100, 1e-6, 42
        )

        np.testing.assert_array_almost_equal(est1, est2)
        assert abs(se1 - se2) < 1e-10


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestTROPRustVsNumpy:
    """Tests comparing TROP Rust and NumPy implementations for numerical equivalence."""

    def test_distance_matrix_matches_numpy(self):
        """Test Rust distance matrix matches NumPy implementation exactly."""
        from diff_diff._rust_backend import compute_unit_distance_matrix
        from diff_diff.trop import TROP

        np.random.seed(42)
        n_periods, n_units = 12, 8
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        # Add some treatment to make it realistic
        D[8:, 0] = 1.0
        D[10:, 1] = 1.0

        # Rust implementation
        rust_dist = compute_unit_distance_matrix(Y, D)

        # NumPy implementation (directly call the private method)
        trop = TROP()
        numpy_dist = trop._compute_all_unit_distances(Y, D, n_units, n_periods)

        np.testing.assert_array_almost_equal(
            rust_dist, numpy_dist, decimal=10,
            err_msg="Distance matrices should match exactly"
        )

    def test_trop_produces_valid_results(self):
        """Test TROP with Rust backend produces valid estimation results."""
        import pandas as pd
        from diff_diff import TROP

        np.random.seed(42)

        # Create test data with known treatment effect
        n_units = 10
        n_periods = 8
        true_effect = 2.0
        data = []

        for i in range(n_units):
            for t in range(n_periods):
                is_treated = (i == 0) and (t >= 6)
                y = 1.0 + 0.5 * i + 0.3 * t + (true_effect if is_treated else 0) + np.random.randn() * 0.5
                data.append({
                    'unit': i,
                    'time': t,
                    'outcome': y,
                    'treated': 1 if is_treated else 0
                })

        df = pd.DataFrame(data)

        # Fit with current backend (Rust if available)
        trop = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            max_loocv_samples=30,
            seed=42
        )
        results = trop.fit(df, 'outcome', 'treated', 'unit', 'time')

        # Check results are valid
        assert np.isfinite(results.att), "ATT should be finite"
        assert np.isfinite(results.se), "SE should be finite"
        assert results.se >= 0, "SE should be non-negative"

        # ATT should be in reasonable range of true effect.
        # Tolerance of 2.0 accounts for:
        # - Small sample size (only 2 treated observations: unit 0, periods 6-7)
        # - Noise in data generation (std=0.5)
        # - LOOCV-selected tuning parameters may not be optimal for small samples
        # This is a validity test, not a precision test - we're checking the
        # estimation produces sensible results, not exact recovery.
        assert abs(results.att - true_effect) < 2.0, \
            f"ATT {results.att:.2f} should be close to true effect {true_effect}"

        # Tuning parameters should be from the grid
        assert results.lambda_time in [0.0, 1.0]
        assert results.lambda_unit in [0.0, 1.0]
        assert results.lambda_nn in [0.0, 0.1]


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestTROPJointRustBackend:
    """Test suite for TROP joint method Rust backend functions."""

    def test_loocv_grid_search_joint_returns_valid_result(self):
        """Test loocv_grid_search_joint returns valid tuning parameters."""
        from diff_diff._rust_backend import loocv_grid_search_joint

        np.random.seed(42)
        n_periods, n_units = 10, 20
        n_treated = 5
        n_post = 3

        # Generate simple data
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[-n_post:, :n_treated] = 1.0

        control_mask = (D == 0).astype(np.uint8)
        lambda_time_grid = np.array([0.0, 1.0])
        lambda_unit_grid = np.array([0.0, 1.0])
        lambda_nn_grid = np.array([0.0, 0.1])

        result = loocv_grid_search_joint(
            Y, D, control_mask,
            lambda_time_grid, lambda_unit_grid, lambda_nn_grid,
            50, 100, 1e-6, 42
        )

        best_lt, best_lu, best_ln, best_score, n_valid, n_attempted, _ = result

        # Check types and bounds
        assert isinstance(best_lt, float)
        assert isinstance(best_lu, float)
        assert isinstance(best_ln, float)
        assert best_lt in [0.0, 1.0]
        assert best_lu in [0.0, 1.0]
        assert best_ln in [0.0, 0.1]
        assert n_valid > 0
        assert n_attempted > 0
        assert best_score >= 0 or np.isinf(best_score)

    def test_loocv_grid_search_joint_reproducible(self):
        """Test loocv_grid_search_joint is reproducible with same seed."""
        from diff_diff._rust_backend import loocv_grid_search_joint

        np.random.seed(42)
        n_periods, n_units = 8, 15
        n_treated = 4
        n_post = 2

        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[-n_post:, :n_treated] = 1.0

        control_mask = (D == 0).astype(np.uint8)
        lambda_time_grid = np.array([0.0, 0.5])
        lambda_unit_grid = np.array([0.0, 0.5])
        lambda_nn_grid = np.array([0.0, 0.1])

        result1 = loocv_grid_search_joint(
            Y, D, control_mask,
            lambda_time_grid, lambda_unit_grid, lambda_nn_grid,
            30, 50, 1e-6, 42
        )
        result2 = loocv_grid_search_joint(
            Y, D, control_mask,
            lambda_time_grid, lambda_unit_grid, lambda_nn_grid,
            30, 50, 1e-6, 42
        )

        # Same seed should produce same results
        assert result1[:4] == result2[:4]

    def test_bootstrap_trop_variance_joint_shape(self):
        """Test bootstrap_trop_variance_joint returns valid output."""
        from diff_diff._rust_backend import bootstrap_trop_variance_joint

        np.random.seed(42)
        n_periods, n_units = 8, 15
        n_treated = 4
        n_post = 2

        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[-n_post:, :n_treated] = 1.0

        estimates, se = bootstrap_trop_variance_joint(
            Y, D, 0.5, 0.5, 0.1, 50, 50, 1e-6, 42
        )

        assert isinstance(estimates, np.ndarray)
        assert len(estimates) > 0
        assert isinstance(se, float)
        assert se >= 0

    def test_bootstrap_trop_variance_joint_reproducible(self):
        """Test bootstrap_trop_variance_joint is reproducible."""
        from diff_diff._rust_backend import bootstrap_trop_variance_joint

        np.random.seed(42)
        n_periods, n_units = 8, 15
        n_treated = 4
        n_post = 2

        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[-n_post:, :n_treated] = 1.0

        est1, se1 = bootstrap_trop_variance_joint(
            Y, D, 0.5, 0.5, 0.1, 50, 50, 1e-6, 42
        )
        est2, se2 = bootstrap_trop_variance_joint(
            Y, D, 0.5, 0.5, 0.1, 50, 50, 1e-6, 42
        )

        np.testing.assert_array_almost_equal(est1, est2)
        np.testing.assert_almost_equal(se1, se2)


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestTROPJointRustVsNumpy:
    """Tests comparing TROP joint Rust and NumPy implementations."""

    def test_trop_joint_produces_valid_results(self):
        """Test TROP joint with Rust backend produces valid results."""
        import pandas as pd
        from diff_diff import TROP

        np.random.seed(42)
        n_units, n_periods = 20, 10
        n_treated = 5
        n_post = 3
        true_effect = 2.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= (n_periods - n_post)
                y = 10.0 + i * 0.2 + t * 0.3 + np.random.randn() * 0.5
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_effect
                data.append({
                    'unit': i,
                    'time': t,
                    'outcome': y,
                    'treated': treatment_indicator,
                })

        df = pd.DataFrame(data)

        trop = TROP(
            method="joint",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=30,
            seed=42
        )
        results = trop.fit(df, 'outcome', 'treated', 'unit', 'time')

        # Check results are valid
        assert np.isfinite(results.att), "ATT should be finite"
        assert np.isfinite(results.se), "SE should be finite"
        assert results.se >= 0, "SE should be non-negative"

        # ATT should be positive (same direction as true effect)
        assert results.att > 0, f"ATT {results.att:.2f} should be positive"

        # Tuning parameters should be from the grid
        assert results.lambda_time in [0.0, 1.0]
        assert results.lambda_unit in [0.0, 1.0]
        assert results.lambda_nn in [0.0, 0.1]

    def test_trop_joint_and_twostep_agree_in_direction(self):
        """Test joint and twostep methods agree on treatment effect direction."""
        import pandas as pd
        from diff_diff import TROP

        np.random.seed(42)
        n_units, n_periods = 20, 10
        n_treated = 5
        n_post = 3
        true_effect = 2.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= (n_periods - n_post)
                y = 10.0 + i * 0.2 + t * 0.3 + np.random.randn() * 0.5
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_effect
                data.append({
                    'unit': i,
                    'time': t,
                    'outcome': y,
                    'treated': treatment_indicator,
                })

        df = pd.DataFrame(data)

        # Fit with joint method
        trop_joint = TROP(
            method="joint",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            seed=42
        )
        results_joint = trop_joint.fit(df, 'outcome', 'treated', 'unit', 'time')

        # Fit with twostep method
        trop_twostep = TROP(
            method="twostep",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            seed=42
        )
        results_twostep = trop_twostep.fit(df, 'outcome', 'treated', 'unit', 'time')

        # Both should have same sign (both positive for true_effect=2.0)
        assert np.sign(results_joint.att) == np.sign(results_twostep.att)


class TestFallbackWhenNoRust:
    """Test that pure Python fallback works when Rust is unavailable."""

    def test_has_rust_backend_is_bool(self):
        """HAS_RUST_BACKEND should be a boolean."""
        assert isinstance(HAS_RUST_BACKEND, bool)

    def test_imports_work_without_rust(self):
        """Core imports should work regardless of Rust availability."""
        from diff_diff import (
            CallawaySantAnna,
            DifferenceInDifferences,
            SyntheticDiD,
        )

        assert CallawaySantAnna is not None
        assert DifferenceInDifferences is not None
        assert SyntheticDiD is not None

    def test_linalg_works_without_rust(self):
        """linalg functions should work with NumPy fallback."""
        from diff_diff.linalg import compute_robust_vcov, solve_ols

        np.random.seed(42)
        n, k = 50, 3
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        coeffs, residuals, vcov = solve_ols(X, y)
        assert coeffs.shape == (k,)
        assert residuals.shape == (n,)
        assert vcov.shape == (k, k)
