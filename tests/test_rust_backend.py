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
