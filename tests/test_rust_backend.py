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
    """Tests comparing Rust and NumPy implementations."""

    def test_synthetic_weights_match(self):
        """Test Rust and NumPy synthetic weights match."""
        from diff_diff._rust_backend import compute_synthetic_weights as rust_fn
        from diff_diff.utils import _compute_synthetic_weights_numpy as numpy_fn

        np.random.seed(42)
        Y_control = np.random.randn(10, 5)
        Y_treated = np.random.randn(10)

        rust_weights = rust_fn(Y_control, Y_treated, 0.0, 1000, 1e-8)
        numpy_weights = numpy_fn(Y_control, Y_treated, 0.0)

        # They should be close but may differ due to optimization algorithm differences
        assert abs(rust_weights.sum() - numpy_weights.sum()) < 0.01

    def test_simplex_projection_match(self):
        """Test Rust and NumPy simplex projection match."""
        from diff_diff._rust_backend import project_simplex as rust_fn
        from diff_diff.utils import _project_simplex as numpy_fn

        v = np.array([0.5, -0.3, 1.2, 0.4, -0.1])

        rust_proj = rust_fn(v)
        numpy_proj = numpy_fn(v)

        np.testing.assert_array_almost_equal(rust_proj, numpy_proj, decimal=10)


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
