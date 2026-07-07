"""
Tests for the Rust backend.

These tests verify that:
1. The Rust backend produces results matching the NumPy implementations
2. Basic functionality works correctly
3. Edge cases are handled properly

Tests are skipped if the Rust backend is not available.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import HAS_RUST_BACKEND


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestRustBackend:
    """Test suite for Rust backend functions."""

    def test_rust_backend_available(self):
        """Verify Rust backend is available when this test runs."""
        assert HAS_RUST_BACKEND

    def test_rust_backend_info(self):
        """Test rust_backend_info returns valid diagnostics dict."""
        from diff_diff._backend import rust_backend_info

        info = rust_backend_info()
        assert isinstance(info, dict)
        assert "blas" in info
        assert "accelerate" in info
        assert "openblas" in info
        assert isinstance(info["blas"], bool)
        assert isinstance(info["accelerate"], bool)
        assert isinstance(info["openblas"], bool)
        # If either platform BLAS is enabled, blas should be True
        if info["accelerate"] or info["openblas"]:
            assert info["blas"] is True

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

    def test_bootstrap_weights_bit_identity_snapshot(self):
        """Pin fixed-seed bootstrap weight output byte-for-byte.

        Regression guard against silent RNG output drift across
        `rand` / `rand_xoshiro` crate upgrades. Distributional moment
        tests would not catch a byte shift that preserves the
        distribution (e.g. `rand 0.9`'s `random_range` algorithm
        change relative to `rand 0.8`'s `gen_range`).

        If this test fails after a Rust dependency bump, the byte stream
        has shifted. Decide deliberately whether to accept the new
        baseline (regenerate these values) or pin to a compatible
        crate version.
        """
        from diff_diff._rust_backend import generate_bootstrap_weights_batch

        # Captured under rand 0.10 + rand_xoshiro 0.8 with seed=42.
        # Rademacher and Mammen bytes match rand 0.8 + rand_xoshiro 0.6;
        # Webb bytes shifted in the rand 0.9 random_range algorithm change.
        expected = {
            "rademacher": np.array(
                [
                    [1.0, -1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0, 1.0],
                ]
            ),
            "mammen": np.array(
                [
                    [
                        1.618033988749895,
                        -0.6180339887498949,
                        1.618033988749895,
                        -0.6180339887498949,
                    ],
                    [
                        -0.6180339887498949,
                        -0.6180339887498949,
                        1.618033988749895,
                        1.618033988749895,
                    ],
                ]
            ),
            "webb": np.array(
                [
                    [1.0, -1.0, 1.224744871391589, 1.0],
                    [-1.0, 0.7071067811865476, 1.224744871391589, 1.224744871391589],
                ]
            ),
        }
        for weight_type, expected_arr in expected.items():
            actual = generate_bootstrap_weights_batch(2, 4, weight_type, 42)
            # Strict bit-identity: the snapshot values are either exact
            # (Rademacher = +/-1.0) or computed once via correctly-rounded
            # IEEE 754 sqrt in Rust (Mammen, Webb), so cross-platform
            # bit-equality holds on conformant hardware.
            np.testing.assert_array_equal(
                actual,
                expected_arr,
                err_msg=f"{weight_type} bootstrap weights drifted from pinned baseline",
            )

    # =========================================================================
    # Synthetic Weight Tests
    # =========================================================================

    # Tests for `compute_synthetic_weights` direct Rust binding removed in
    # the silent-failures audit post-cleanup (finding #22). The helper was
    # deleted from the Python layer and the Rust symbol was subsequently
    # removed from `rust/src/weights.rs` + unregistered in `rust/src/lib.rs`.

    def test_compute_synthetic_weights_is_removed(self):
        """Regression guard against accidental re-export of the deleted
        `compute_synthetic_weights` PyO3 binding (silent-failures finding
        #22). If this test fails, someone reintroduced the binding — audit
        the reason before adding it back."""
        import diff_diff._rust_backend as rb

        with pytest.raises(ImportError):
            from diff_diff._rust_backend import (  # noqa: F401
                compute_synthetic_weights,
            )

        assert not hasattr(rb, "compute_synthetic_weights"), (
            "compute_synthetic_weights was removed from the Rust backend "
            "in the post-audit cleanup for finding #22; its presence here "
            "indicates accidental re-export."
        )

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
            residuals,
            expected_residuals,
            decimal=8,
            err_msg="Residuals should match y - X @ coeffs",
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
            residuals,
            expected_residuals,
            decimal=8,
            err_msg="Residuals should match y - X @ coeffs",
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
        assert (
            abs(coeffs[-1] - true_effect) < 2.0
        ), f"Treatment effect {coeffs[-1]} is too far from true effect {true_effect}"


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
            rust_coeffs, numpy_coeffs, decimal=8, err_msg="OLS coefficients should match"
        )
        np.testing.assert_array_almost_equal(
            rust_resid, numpy_resid, decimal=8, err_msg="OLS residuals should match"
        )

    def test_solve_ols_underdetermined_match(self):
        """n < k through the RAW rust backend's slimmed marshalling: the
        thin-SVD U/V shapes flip (U is n x n, V is k x n), exercising the
        uty and drop paths on the underdetermined branch. This is a
        direct-kernel test - the PUBLIC solve_ols rejects n < k outright -
        asserting the engines' shared residual/exact-fit contract."""
        from diff_diff._rust_backend import solve_ols as rust_fn
        from diff_diff.linalg import _solve_ols_numpy as numpy_fn

        np.random.seed(7)
        n, k = 6, 9
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        import warnings as _w

        rust_coeffs, rust_resid, _ = rust_fn(X, y, None, True)
        with _w.catch_warnings():
            _w.simplefilter("ignore")  # numpy path warns on the rank drop
            _, numpy_resid, _ = numpy_fn(X, y, cluster_ids=None)

        # Coefficient CONVENTIONS legitimately differ here (rust kernel:
        # truncated-SVD minimum-norm over all k; numpy path: column-drop with
        # NaN); the public solve_ols never routes n < k to either engine (it
        # rejects such designs up front), so like
        # test_rank_deficient_ols_residuals_match this asserts the engines'
        # shared contract: an exact fit with matching residuals.
        assert np.all(np.isfinite(rust_coeffs))
        np.testing.assert_array_almost_equal(
            rust_resid, np.zeros(n), decimal=10, err_msg="rust residuals ~0"
        )
        np.testing.assert_array_almost_equal(
            rust_resid, numpy_resid, decimal=8, err_msg="residuals should match"
        )
        # the rust min-norm solution reproduces y exactly (fitted = X @ beta)
        np.testing.assert_array_almost_equal(
            X @ rust_coeffs, y, decimal=10, err_msg="exact fit expected"
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
            rust_coeffs, numpy_coeffs, decimal=8, err_msg="Clustered OLS coefficients should match"
        )
        # VCoV may differ slightly due to implementation details
        np.testing.assert_array_almost_equal(
            rust_vcov, numpy_vcov, decimal=5, err_msg="Clustered OLS VCoV should match"
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
            rust_resid,
            numpy_resid,
            decimal=5,
            err_msg="Residuals should match despite different coefficient representations",
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
        assert (
            abs(rust_effect - true_effect) < 2.0
        ), f"Rust treatment effect {rust_effect} too far from true {true_effect}"

        # NumPy treatment effect should be close (may be finite or NaN depending on rank)
        numpy_effect = numpy_coeffs[-1]
        if np.isfinite(numpy_effect):
            assert (
                abs(numpy_effect - true_effect) < 2.0
            ), f"NumPy treatment effect {numpy_effect} too far from true {true_effect}"
            # Effects should be close to each other
            assert (
                abs(rust_effect - numpy_effect) < 0.5
            ), f"Rust ({rust_effect}) and NumPy ({numpy_effect}) effects should match"

        # Residuals should be very close (key equivalence check)
        np.testing.assert_array_almost_equal(
            rust_resid,
            numpy_resid,
            decimal=5,
            err_msg="Residuals should match for MultiPeriodDiD-like design",
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
            rust_vcov, numpy_vcov, decimal=8, err_msg="HC1 robust VCoV should match"
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
            rust_vcov, numpy_vcov, decimal=6, err_msg="Cluster-robust VCoV should match"
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

        second_moment = (weights**2).mean()
        assert abs(second_moment - 1.0) < 0.02, f"Mammen E[w^2] should be ~1, got {second_moment}"

        third_moment = (weights**3).mean()
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

    # Rust/NumPy synthetic_weights parity tests removed in the silent-failures
    # audit post-cleanup (finding #22). Helper deleted; parity is now a
    # non-question since both paths route through the shared `_sc_weight_fw`
    # dispatcher in `utils.py`.

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
                rust_proj,
                numpy_proj,
                decimal=10,
                err_msg=f"Simplex projection mismatch for input {v}",
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
            "Re-running with Python backend" in str(warning.message) for warning in w
        )

        # Key invariants that must hold regardless of which backend is used:
        # 1. Coefficients must be finite (either via Rust SVD or Python R-style)
        finite_coeffs = coeffs[np.isfinite(coeffs)]
        assert len(finite_coeffs) >= 3, "At least 3 coefficients should be finite (identifiable)"
        assert np.all(
            np.abs(finite_coeffs) < 1e10
        ), f"Finite coefficients should be reasonable, got {finite_coeffs}"

        # 2. If vcov has any finite values, they should correspond to finite coefficients
        if vcov is not None:
            finite_coef_mask = np.isfinite(coeffs)
            for i in range(len(coeffs)):
                if finite_coef_mask[i]:
                    # This coefficient's variance should be finite
                    var_i = vcov[i, i]
                    assert np.isfinite(var_i) or np.isnan(
                        var_i
                    ), f"Variance for finite coef {i} should be finite or NaN (dropped)"

        # 3. Residuals must always be finite
        assert np.all(np.isfinite(residuals)), "Residuals should be finite"

        # 4. R-style consistency: NaN coefficients must have NaN vcov diagonal
        if vcov is not None:
            nan_coef_indices = set(np.where(np.isnan(coeffs))[0])
            nan_vcov_diag_indices = set(np.where(np.isnan(np.diag(vcov)))[0])

            # NaN in vcov diagonal should correspond exactly to NaN coefficients
            assert nan_vcov_diag_indices == nan_coef_indices, (
                f"NaN vcov diagonal {nan_vcov_diag_indices} should match "
                f"NaN coefficients {nan_coef_indices}"
            )

        # 5. If fallback warning was emitted, R-style handling MUST have occurred
        # This verifies that the fallback actually applies R-style NaN handling
        # (not minimum-norm solution which would have all finite coefficients)
        if fallback_warning_emitted:
            assert np.any(np.isnan(coeffs)), (
                "Fallback warning emitted but no NaN coefficients - "
                "R-style handling was not applied"
            )
            assert vcov is not None and np.any(np.isnan(vcov)), (
                "Fallback warning emitted but vcov has no NaN - " "R-style handling was not applied"
            )


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
            rust_dist, numpy_dist, decimal=10, err_msg="Distance matrices should match"
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
            Y,
            D,
            control_mask,
            time_dist,
            lambda_time,
            lambda_unit,
            lambda_nn,
            100,
            1e-6,
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

    @staticmethod
    def _stratified_indices(n_control, n_treated, n_bootstrap, seed):
        """Build stratified bootstrap index arrays via the shared helper."""
        from diff_diff.bootstrap_utils import stratified_bootstrap_indices

        rng = np.random.default_rng(seed)
        return stratified_bootstrap_indices(rng, n_control, n_treated, n_bootstrap)

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
        # Stratified pools: 1 treated unit (index 0), 5 control units
        ctrl_idx, trt_idx = self._stratified_indices(
            n_control=5, n_treated=1, n_bootstrap=n_bootstrap, seed=42
        )
        estimates, se = bootstrap_trop_variance(
            Y,
            D,
            control_mask,
            time_dist,
            1.0,
            1.0,
            0.1,  # lambda values
            n_bootstrap,
            100,
            1e-6,
            ctrl_idx,
            trt_idx,
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

        # Run twice with same seed (helper is deterministic given the same seed)
        ctrl_idx_a, trt_idx_a = self._stratified_indices(
            n_control=5, n_treated=1, n_bootstrap=20, seed=42
        )
        ctrl_idx_b, trt_idx_b = self._stratified_indices(
            n_control=5, n_treated=1, n_bootstrap=20, seed=42
        )
        est1, se1 = bootstrap_trop_variance(
            Y,
            D,
            control_mask,
            time_dist,
            1.0,
            1.0,
            0.1,
            20,
            100,
            1e-6,
            ctrl_idx_a,
            trt_idx_a,
        )
        est2, se2 = bootstrap_trop_variance(
            Y,
            D,
            control_mask,
            time_dist,
            1.0,
            1.0,
            0.1,
            20,
            100,
            1e-6,
            ctrl_idx_b,
            trt_idx_b,
        )

        np.testing.assert_array_almost_equal(est1, est2)
        assert abs(se1 - se2) < 1e-10

    def test_bootstrap_rejects_negative_index(self):
        """Rust local bootstrap must raise PyValueError on a negative index."""
        from diff_diff._rust_backend import bootstrap_trop_variance

        np.random.seed(42)
        n_periods, n_units = 8, 6
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[6:, 0] = 1.0
        control_mask = (D == 0).astype(np.uint8)
        time_dist = np.abs(
            np.arange(n_periods)[:, np.newaxis] - np.arange(n_periods)[np.newaxis, :]
        ).astype(np.int64)

        ctrl_idx, trt_idx = self._stratified_indices(
            n_control=5, n_treated=1, n_bootstrap=5, seed=0
        )
        ctrl_idx[2, 3] = -1  # negative
        with pytest.raises(ValueError, match="control_indices.*out-of-range"):
            bootstrap_trop_variance(
                Y,
                D,
                control_mask,
                time_dist,
                1.0,
                1.0,
                0.1,
                5,
                100,
                1e-6,
                ctrl_idx,
                trt_idx,
            )

    def test_bootstrap_rejects_out_of_range_index(self):
        """Rust local bootstrap must raise PyValueError on an index >= pool size."""
        from diff_diff._rust_backend import bootstrap_trop_variance

        np.random.seed(42)
        n_periods, n_units = 8, 6
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[6:, 0] = 1.0
        control_mask = (D == 0).astype(np.uint8)
        time_dist = np.abs(
            np.arange(n_periods)[:, np.newaxis] - np.arange(n_periods)[np.newaxis, :]
        ).astype(np.int64)

        ctrl_idx, trt_idx = self._stratified_indices(
            n_control=5, n_treated=1, n_bootstrap=5, seed=0
        )
        trt_idx[1, 0] = 99  # >> n_treated=1
        with pytest.raises(ValueError, match="treated_indices.*out-of-range"):
            bootstrap_trop_variance(
                Y,
                D,
                control_mask,
                time_dist,
                1.0,
                1.0,
                0.1,
                5,
                100,
                1e-6,
                ctrl_idx,
                trt_idx,
            )


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
            rust_dist, numpy_dist, decimal=10, err_msg="Distance matrices should match exactly"
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
                y = (
                    1.0
                    + 0.5 * i
                    + 0.3 * t
                    + (true_effect if is_treated else 0)
                    + np.random.randn() * 0.5
                )
                data.append({"unit": i, "time": t, "outcome": y, "treated": 1 if is_treated else 0})

        df = pd.DataFrame(data)

        # Fit with current backend (Rust if available)
        trop = TROP(
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            seed=42,
        )
        results = trop.fit(df, "outcome", "treated", "unit", "time")

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
        assert (
            abs(results.att - true_effect) < 2.0
        ), f"ATT {results.att:.2f} should be close to true effect {true_effect}"

        # Tuning parameters should be from the grid
        assert results.lambda_time in [0.0, 1.0]
        assert results.lambda_unit in [0.0, 1.0]
        assert results.lambda_nn in [0.0, 0.1]


@pytest.mark.slow
@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestTROPGlobalRustBackend:
    """Test suite for TROP global method Rust backend functions."""

    def test_loocv_grid_search_global_returns_valid_result(self):
        """Test loocv_grid_search_global returns valid tuning parameters."""
        from diff_diff._rust_backend import loocv_grid_search_global

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

        result = loocv_grid_search_global(
            Y,
            D,
            control_mask,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            100,
            1e-6,
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

    def test_loocv_grid_search_global_reproducible(self):
        """Test loocv_grid_search_global is deterministic (no subsampling)."""
        from diff_diff._rust_backend import loocv_grid_search_global

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

        result1 = loocv_grid_search_global(
            Y,
            D,
            control_mask,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            50,
            1e-6,
        )
        result2 = loocv_grid_search_global(
            Y,
            D,
            control_mask,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            50,
            1e-6,
        )

        # Without subsampling, results should be deterministic
        assert result1[:4] == result2[:4]

    @staticmethod
    def _global_stratified_indices(n_control, n_treated, n_bootstrap, seed):
        """Build stratified bootstrap index arrays via the shared helper."""
        from diff_diff.bootstrap_utils import stratified_bootstrap_indices

        rng = np.random.default_rng(seed)
        return stratified_bootstrap_indices(rng, n_control, n_treated, n_bootstrap)

    def test_bootstrap_trop_variance_global_shape(self):
        """Test bootstrap_trop_variance_global returns valid output."""
        from diff_diff._rust_backend import bootstrap_trop_variance_global

        np.random.seed(42)
        n_periods, n_units = 8, 15
        n_treated = 4
        n_post = 2

        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[-n_post:, :n_treated] = 1.0

        # Stratified pools: 4 treated units, 11 control units
        ctrl_idx, trt_idx = self._global_stratified_indices(
            n_control=n_units - n_treated, n_treated=n_treated, n_bootstrap=50, seed=42
        )
        estimates, se = bootstrap_trop_variance_global(
            Y,
            D,
            0.5,
            0.5,
            0.1,
            50,
            50,
            1e-6,
            ctrl_idx,
            trt_idx,
        )

        assert isinstance(estimates, np.ndarray)
        assert len(estimates) > 0
        assert isinstance(se, float)
        assert se >= 0

    def test_bootstrap_trop_variance_global_reproducible(self):
        """Test bootstrap_trop_variance_global is reproducible."""
        from diff_diff._rust_backend import bootstrap_trop_variance_global

        np.random.seed(42)
        n_periods, n_units = 8, 15
        n_treated = 4
        n_post = 2

        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[-n_post:, :n_treated] = 1.0

        ctrl_a, trt_a = self._global_stratified_indices(
            n_control=n_units - n_treated, n_treated=n_treated, n_bootstrap=50, seed=42
        )
        ctrl_b, trt_b = self._global_stratified_indices(
            n_control=n_units - n_treated, n_treated=n_treated, n_bootstrap=50, seed=42
        )
        est1, se1 = bootstrap_trop_variance_global(
            Y,
            D,
            0.5,
            0.5,
            0.1,
            50,
            50,
            1e-6,
            ctrl_a,
            trt_a,
        )
        est2, se2 = bootstrap_trop_variance_global(
            Y,
            D,
            0.5,
            0.5,
            0.1,
            50,
            50,
            1e-6,
            ctrl_b,
            trt_b,
        )

        np.testing.assert_array_almost_equal(est1, est2)
        np.testing.assert_almost_equal(se1, se2)

    def test_bootstrap_global_rejects_negative_index(self):
        """Rust global bootstrap must raise PyValueError on a negative index."""
        from diff_diff._rust_backend import bootstrap_trop_variance_global

        np.random.seed(42)
        n_periods, n_units = 8, 15
        n_treated = 4
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[-2:, :n_treated] = 1.0

        ctrl_idx, trt_idx = self._global_stratified_indices(
            n_control=n_units - n_treated, n_treated=n_treated, n_bootstrap=10, seed=0
        )
        ctrl_idx[3, 2] = -5  # negative
        with pytest.raises(ValueError, match="control_indices.*out-of-range"):
            bootstrap_trop_variance_global(
                Y,
                D,
                0.5,
                0.5,
                0.1,
                10,
                50,
                1e-6,
                ctrl_idx,
                trt_idx,
            )

    def test_bootstrap_global_rejects_out_of_range_index(self):
        """Rust global bootstrap must raise PyValueError on an index >= pool size."""
        from diff_diff._rust_backend import bootstrap_trop_variance_global

        np.random.seed(42)
        n_periods, n_units = 8, 15
        n_treated = 4
        Y = np.random.randn(n_periods, n_units)
        D = np.zeros((n_periods, n_units))
        D[-2:, :n_treated] = 1.0

        ctrl_idx, trt_idx = self._global_stratified_indices(
            n_control=n_units - n_treated, n_treated=n_treated, n_bootstrap=10, seed=0
        )
        trt_idx[5, 1] = 99  # >> n_treated=4
        with pytest.raises(ValueError, match="treated_indices.*out-of-range"):
            bootstrap_trop_variance_global(
                Y,
                D,
                0.5,
                0.5,
                0.1,
                10,
                50,
                1e-6,
                ctrl_idx,
                trt_idx,
            )


@pytest.mark.slow
@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestTROPGlobalRustVsNumpy:
    """Tests comparing TROP global Rust and NumPy implementations."""

    def test_trop_global_produces_valid_results(self):
        """Test TROP global with Rust backend produces valid results."""
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
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        trop = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=30,
            seed=42,
        )
        results = trop.fit(df, "outcome", "treated", "unit", "time")

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

    def test_trop_global_and_local_agree_in_direction(self):
        """Test global and local methods agree on treatment effect direction."""
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
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        # Fit with global method
        trop_global = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            seed=42,
        )
        results_global = trop_global.fit(df, "outcome", "treated", "unit", "time")

        # Fit with local method
        trop_local = TROP(
            method="local",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            seed=42,
        )
        results_local = trop_local.fit(df, "outcome", "treated", "unit", "time")

        # Both should have same sign (both positive for true_effect=2.0)
        assert np.sign(results_global.att) == np.sign(results_local.att)

    def test_trop_global_handles_nan_outcomes(self):
        """Test TROP global method handles NaN outcome values gracefully."""
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
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        # Introduce NaN values in control observations (pre-treatment periods)
        # Set 5% of control pre-treatment observations to NaN
        nan_indices = []
        for idx, row in df.iterrows():
            if row["treated"] == 0 and row["time"] < (n_periods - n_post):
                if np.random.rand() < 0.05:
                    nan_indices.append(idx)
        df.loc[nan_indices, "outcome"] = np.nan

        n_nan = len(nan_indices)
        assert n_nan > 0, "Should have introduced some NaN values"

        trop = TROP(
            method="global",
            lambda_time_grid=[0.0, 1.0],
            lambda_unit_grid=[0.0, 1.0],
            lambda_nn_grid=[0.0, 0.1],
            n_bootstrap=20,
            seed=42,
        )
        results = trop.fit(df, "outcome", "treated", "unit", "time")

        # Results should be finite (NaN observations are excluded)
        assert np.isfinite(results.att), f"ATT {results.att} should be finite with NaN data"
        assert np.isfinite(results.se), f"SE {results.se} should be finite with NaN data"
        assert results.se >= 0, "SE should be non-negative"

        # ATT should still be positive (true effect is positive)
        assert results.att > 0, f"ATT {results.att:.2f} should be positive"

    def test_trop_global_no_valid_pre_unit_gets_zero_weight(self):
        """Test that units with no valid pre-period data get zero weight.

        When a control unit has all NaN values in the pre-treatment period,
        it should receive zero weight (not maximum weight). This prevents
        such units from influencing the counterfactual estimation.

        This tests the fix for PR #113 Round 3 feedback (P1-1) where Rust
        backend was setting dist=0 -> delta_unit=exp(0)=1.0 (max weight)
        instead of dist=inf -> delta_unit=exp(-inf)=0.0 (zero weight).
        """
        import pandas as pd
        from diff_diff import TROP

        np.random.seed(42)
        n_units, n_periods = 15, 10
        n_treated = 3
        n_post = 3
        true_effect = 2.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= (n_periods - n_post)
                y = 10.0 + i * 0.2 + t * 0.3 + np.random.randn() * 0.3
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_effect
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        # Set ALL pre-period outcomes to NaN for one control unit (unit n_treated)
        # This unit has no valid pre-period data and should get zero weight
        control_unit_with_no_pre = n_treated  # First control unit
        pre_mask = (df["unit"] == control_unit_with_no_pre) & (df["time"] < (n_periods - n_post))
        df.loc[pre_mask, "outcome"] = np.nan

        # Verify we set NaN correctly
        unit_pre_data = df[
            (df["unit"] == control_unit_with_no_pre) & (df["time"] < (n_periods - n_post))
        ]
        assert (
            unit_pre_data["outcome"].isna().all()
        ), "Control unit should have all NaN in pre-period"

        # Fit with global method - should handle gracefully
        trop = TROP(
            method="global",
            lambda_time_grid=[0.5, 1.0],
            lambda_unit_grid=[0.5, 1.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=20,
            seed=42,
        )
        results = trop.fit(df, "outcome", "treated", "unit", "time")

        # Results should be finite - the unit with no valid pre-period data
        # should get zero weight and not break estimation
        assert np.isfinite(results.att), f"ATT {results.att} should be finite"
        assert np.isfinite(results.se), f"SE {results.se} should be finite"

        # ATT should be in reasonable range of true effect
        # The no-valid-pre unit getting zero weight shouldn't corrupt the estimate
        assert (
            abs(results.att - true_effect) < 1.5
        ), f"ATT {results.att:.2f} should be close to true effect {true_effect}"

    def test_trop_global_nan_exclusion_rust_python_parity(self):
        """Test Rust and Python backends produce matching results with NaN data.

        This verifies that when data contains NaN values:
        1. Both backends exclude NaN observations consistently
        2. ATT estimates are close (within tolerance)
        3. Neither backend produces corrupt results

        This tests the fix for PR #113 Round 3 feedback (P2-1).
        """
        import os
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
                y = 10.0 + i * 0.2 + t * 0.3 + np.random.randn() * 0.3
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_effect
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        # Introduce scattered NaN values (5% of control pre-period observations)
        np.random.seed(123)  # Different seed for NaN placement
        for idx, row in df.iterrows():
            if row["treated"] == 0 and row["time"] < (n_periods - n_post):
                if np.random.rand() < 0.05:
                    df.loc[idx, "outcome"] = np.nan

        n_nan = df["outcome"].isna().sum()
        assert n_nan > 0, "Should have some NaN values"

        # Common TROP parameters
        trop_params = dict(
            method="global",
            lambda_time_grid=[0.5, 1.0],
            lambda_unit_grid=[0.5, 1.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=20,
            seed=42,
        )

        # Run with Rust backend (current default when available)
        trop_rust = TROP(**trop_params)
        results_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        # Run with Python-only backend using mock.patch to avoid module reload issues
        # (Module reload breaks isinstance() checks in other tests due to class identity)
        from unittest.mock import patch
        import sys

        trop_global_module = sys.modules["diff_diff.trop_global"]

        with (
            patch.object(trop_global_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_global_module, "_rust_loocv_grid_search_global", None),
            patch.object(trop_global_module, "_rust_bootstrap_trop_variance_global", None),
        ):

            trop_python = TROP(**trop_params)
            results_python = trop_python.fit(df.copy(), "outcome", "treated", "unit", "time")

        # Both should produce finite results
        assert np.isfinite(results_rust.att), f"Rust ATT {results_rust.att} should be finite"
        assert np.isfinite(results_python.att), f"Python ATT {results_python.att} should be finite"

        # ATT estimates should be close (within reasonable tolerance)
        # Allow some difference due to LOOCV randomness and numerical differences
        att_diff = abs(results_rust.att - results_python.att)
        assert att_diff < 0.5, (
            f"Rust ATT ({results_rust.att:.3f}) and Python ATT ({results_python.att:.3f}) "
            f"differ by {att_diff:.3f}, should be < 0.5"
        )

        # Both should recover true effect direction
        assert results_rust.att > 0, f"Rust ATT {results_rust.att} should be positive"
        assert results_python.att > 0, f"Python ATT {results_python.att} should be positive"

    def test_trop_global_treated_pre_nan_rust_python_parity(self):
        """Test Rust/Python parity when treated units have pre-period NaN.

        When all treated units have NaN at a pre-period, average_treated[t] = NaN.
        Both backends should exclude this period from unit distance calculation
        (both numerator and denominator) to avoid inflating valid_count.

        This tests the fix for PR #113 Round 5 feedback (P2).
        """
        import os
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
                y = 10.0 + i * 0.2 + t * 0.3 + np.random.randn() * 0.3
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_effect
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        # Set ALL treated units' outcomes at period 3 (a pre-period) to NaN
        # This makes average_treated[3] = NaN
        target_period = 3
        treated_units = list(range(n_treated))
        mask = df["unit"].isin(treated_units) & (df["time"] == target_period)
        df.loc[mask, "outcome"] = np.nan

        # Verify we set NaN correctly
        n_nan = df.loc[mask, "outcome"].isna().sum()
        assert n_nan == n_treated, f"Should have {n_treated} NaN, got {n_nan}"

        # Common TROP parameters
        trop_params = dict(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=20,
            seed=42,
        )

        # Run with Rust backend (current default when available)
        trop_rust = TROP(**trop_params)
        results_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        # Run with Python-only backend using mock.patch to avoid module reload issues
        # (Module reload breaks isinstance() checks in other tests due to class identity)
        from unittest.mock import patch
        import sys

        trop_global_module = sys.modules["diff_diff.trop_global"]

        with (
            patch.object(trop_global_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_global_module, "_rust_loocv_grid_search_global", None),
            patch.object(trop_global_module, "_rust_bootstrap_trop_variance_global", None),
        ):

            trop_python = TROP(**trop_params)
            results_python = trop_python.fit(df.copy(), "outcome", "treated", "unit", "time")

        # Both should produce finite results
        assert np.isfinite(results_rust.att), f"Rust ATT {results_rust.att} should be finite"
        assert np.isfinite(results_python.att), f"Python ATT {results_python.att} should be finite"

        # ATT estimates should be close (within reasonable tolerance)
        att_diff = abs(results_rust.att - results_python.att)
        assert att_diff < 0.5, (
            f"Rust ATT ({results_rust.att:.3f}) and Python ATT ({results_python.att:.3f}) "
            f"differ by {att_diff:.3f}, should be < 0.5"
        )

    def test_trop_global_solver_parity_no_lowrank(self):
        """Test Rust/Python solver parity for no-lowrank path (lambda_nn >= 1e10).

        Both backends should produce matching (mu, alpha, beta) at atol=1e-6.
        This validates the convergence criterion fix (checking all params, not just mu).
        """
        import pandas as pd
        from diff_diff import TROP
        from unittest.mock import patch
        import sys

        np.random.seed(42)
        n_units, n_periods = 15, 8
        n_treated = 4
        n_post = 3

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= (n_periods - n_post)
                y = 5.0 + i * 0.5 + t * 0.4 + np.random.randn() * 0.2
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += 2.0
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )
        df = pd.DataFrame(data)

        # Fixed lambda with lambda_nn=inf (no low-rank)
        trop_params = dict(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=2,
            seed=42,
        )

        # Rust backend
        trop_rust = TROP(**trop_params)
        results_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        # Python-only backend
        trop_global_module = sys.modules["diff_diff.trop_global"]
        with (
            patch.object(trop_global_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_global_module, "_rust_loocv_grid_search_global", None),
            patch.object(trop_global_module, "_rust_bootstrap_trop_variance_global", None),
        ):
            trop_python = TROP(**trop_params)
            results_python = trop_python.fit(df.copy(), "outcome", "treated", "unit", "time")

        # ATT should match closely
        assert (
            abs(results_rust.att - results_python.att) < 1e-6
        ), f"No-lowrank ATT mismatch: Rust={results_rust.att:.8f}, Python={results_python.att:.8f}"

        # Unit and time effects should match
        for key in results_rust.unit_effects:
            r_val = results_rust.unit_effects[key]
            p_val = results_python.unit_effects[key]
            assert (
                abs(r_val - p_val) < 1e-6
            ), f"Unit effect mismatch for {key}: Rust={r_val:.8f}, Python={p_val:.8f}"

        for key in results_rust.time_effects:
            r_val = results_rust.time_effects[key]
            p_val = results_python.time_effects[key]
            assert (
                abs(r_val - p_val) < 1e-6
            ), f"Time effect mismatch for {key}: Rust={r_val:.8f}, Python={p_val:.8f}"

    def test_trop_global_solver_parity_with_lowrank(self):
        """Test Rust/Python solver parity for with-lowrank path (finite lambda_nn).

        Both backends should produce matching (mu, alpha, beta) at atol=1e-6.
        The with-lowrank solver calls no-lowrank as its inner step, so the
        convergence fix cascades here too.
        """
        import pandas as pd
        from diff_diff import TROP
        from unittest.mock import patch
        import sys

        np.random.seed(42)
        n_units, n_periods = 15, 8
        n_treated = 4
        n_post = 3

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= (n_periods - n_post)
                y = 5.0 + i * 0.5 + t * 0.4 + np.random.randn() * 0.2
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += 2.0
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )
        df = pd.DataFrame(data)

        # Fixed lambda with finite lambda_nn (low-rank enabled)
        trop_params = dict(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=42,
        )

        # Rust backend
        trop_rust = TROP(**trop_params)
        results_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        # Python-only backend
        trop_global_module = sys.modules["diff_diff.trop_global"]
        with (
            patch.object(trop_global_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_global_module, "_rust_loocv_grid_search_global", None),
            patch.object(trop_global_module, "_rust_bootstrap_trop_variance_global", None),
        ):
            trop_python = TROP(**trop_params)
            results_python = trop_python.fit(df.copy(), "outcome", "treated", "unit", "time")

        # ATT should match closely
        assert (
            abs(results_rust.att - results_python.att) < 1e-6
        ), f"With-lowrank ATT mismatch: Rust={results_rust.att:.8f}, Python={results_python.att:.8f}"

        # Unit and time effects should match
        for key in results_rust.unit_effects:
            r_val = results_rust.unit_effects[key]
            p_val = results_python.unit_effects[key]
            assert (
                abs(r_val - p_val) < 1e-6
            ), f"Unit effect mismatch for {key}: Rust={r_val:.8f}, Python={p_val:.8f}"


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestSDIDRustBackend:
    """Test suite for SDID Frank-Wolfe Rust backend functions."""

    def test_noise_level_matches_numpy(self):
        """Test Rust noise level matches NumPy implementation."""
        from diff_diff._rust_backend import compute_noise_level as rust_fn
        from diff_diff.utils import _compute_noise_level_numpy as numpy_fn

        np.random.seed(42)
        Y_pre = np.random.randn(10, 5)
        rust_nl = rust_fn(Y_pre)
        numpy_nl = numpy_fn(Y_pre)
        assert (
            abs(rust_nl - numpy_nl) < 1e-10
        ), f"Noise levels differ: rust={rust_nl}, numpy={numpy_nl}"

    def test_noise_level_single_period(self):
        """Test noise level returns 0 for single pre-period."""
        from diff_diff._rust_backend import compute_noise_level as rust_fn

        Y_pre = np.random.randn(1, 5)
        assert rust_fn(Y_pre) == 0.0

    def test_sc_weight_fw_on_simplex(self):
        """Test Frank-Wolfe solver produces valid simplex weights."""
        from diff_diff._rust_backend import sc_weight_fw as rust_fn

        np.random.seed(42)
        Y = np.random.randn(5, 4)  # 5 rows, 3 pre-periods + 1 target
        weights = rust_fn(Y, 0.1, True, None, 1e-3, 100)
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights should sum to 1, got {weights.sum()}"
        assert np.all(weights >= -1e-6), "Weights should be non-negative"
        assert weights.shape == (3,), f"Expected shape (3,), got {weights.shape}"

    def test_sc_weight_fw_matches_numpy(self):
        """Test Rust Frank-Wolfe matches Python implementation."""
        from diff_diff._rust_backend import sc_weight_fw as rust_fn
        from diff_diff.utils import _sc_weight_fw_numpy as numpy_fn

        np.random.seed(42)
        Y = np.random.randn(8, 6)  # 8 rows, 5 pre + 1 target
        rust_w = rust_fn(Y, 0.5, True, None, 1e-3, 1000)
        numpy_w = numpy_fn(Y, 0.5, True, None, 1e-3, 1000)
        np.testing.assert_array_almost_equal(
            rust_w, numpy_w, decimal=6, err_msg="Frank-Wolfe weights should match"
        )

    def test_sc_weight_fw_with_init_weights(self):
        """Test Frank-Wolfe with initial weights."""
        from diff_diff._rust_backend import sc_weight_fw as rust_fn
        from diff_diff.utils import _sc_weight_fw_numpy as numpy_fn

        np.random.seed(42)
        Y = np.random.randn(6, 5)
        init_w = np.array([0.5, 0.3, 0.15, 0.05])
        rust_w = rust_fn(Y, 0.2, True, init_w, 1e-3, 500)
        numpy_w = numpy_fn(Y, 0.2, True, init_w, 1e-3, 500)
        np.testing.assert_array_almost_equal(
            rust_w, numpy_w, decimal=6, err_msg="Frank-Wolfe with init weights should match"
        )

    def test_time_weights_on_simplex(self):
        """Test Rust time weights are on simplex."""
        from diff_diff._rust_backend import compute_time_weights as rust_fn

        np.random.seed(42)
        Y_pre = np.random.randn(8, 5)
        Y_post = np.random.randn(3, 5)
        weights = rust_fn(Y_pre, Y_post, 0.01, True, 1e-3, 1000)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert np.all(weights >= -1e-6)
        assert weights.shape == (8,)

    def test_time_weights_match_numpy(self):
        """Test Rust and NumPy time weights match (2-pass with sparsification)."""
        from diff_diff._rust_backend import compute_time_weights as rust_fn
        from diff_diff.utils import _sc_weight_fw_numpy, _sparsify

        np.random.seed(42)
        Y_pre = np.random.randn(6, 4)
        Y_post = np.random.randn(2, 4)

        min_decrease = 1e-3
        max_iter_pre = 100
        max_iter = 1000

        # Rust implementation (2-pass with sparsification)
        rust_w = rust_fn(Y_pre, Y_post, 0.01, True, min_decrease, max_iter_pre, max_iter)

        # Python implementation (manual 2-pass matching Rust)
        post_means = np.mean(Y_post, axis=0)
        Y_time = np.column_stack([Y_pre.T, post_means])
        lam = _sc_weight_fw_numpy(Y_time, 0.01, True, None, min_decrease, max_iter_pre)
        lam = _sparsify(lam)
        numpy_w = _sc_weight_fw_numpy(Y_time, 0.01, True, lam, min_decrease, max_iter)

        np.testing.assert_array_almost_equal(
            rust_w, numpy_w, decimal=6, err_msg="Time weights should match"
        )

    def test_time_weights_single_preperiod(self):
        """Test time weights with single pre-period returns [1.0]."""
        from diff_diff._rust_backend import compute_time_weights as rust_fn

        Y_pre = np.random.randn(1, 5)
        Y_post = np.random.randn(2, 5)
        weights = rust_fn(Y_pre, Y_post, 0.01)
        assert weights.shape == (1,)
        assert abs(weights[0] - 1.0) < 1e-10

    def test_unit_weights_on_simplex(self):
        """Test Rust unit weights are on simplex."""
        from diff_diff._rust_backend import compute_sdid_unit_weights as rust_fn

        np.random.seed(42)
        Y_pre = np.random.randn(8, 5)
        Y_tr_mean = np.random.randn(8)
        weights = rust_fn(Y_pre, Y_tr_mean, 0.5, True, 1e-3, 100, 1000)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert np.all(weights >= -1e-6)
        assert weights.shape == (5,)

    def test_unit_weights_match_numpy(self):
        """Test Rust and NumPy unit weights match."""
        from diff_diff._rust_backend import compute_sdid_unit_weights as rust_fn
        from diff_diff.utils import _sc_weight_fw_numpy, _sparsify

        np.random.seed(42)
        Y_pre = np.random.randn(6, 4)
        Y_tr_mean = np.random.randn(6)

        # Rust implementation
        rust_w = rust_fn(Y_pre, Y_tr_mean, 0.5, True, 1e-3, 100, 1000)

        # Python implementation (manual)
        Y_unit = np.column_stack([Y_pre, Y_tr_mean.reshape(-1, 1)])
        omega = _sc_weight_fw_numpy(Y_unit, 0.5, True, None, 1e-3, 100)
        omega = _sparsify(omega)
        numpy_w = _sc_weight_fw_numpy(Y_unit, 0.5, True, omega, 1e-3, 1000)

        np.testing.assert_array_almost_equal(
            rust_w, numpy_w, decimal=6, err_msg="Unit weights should match"
        )

    def test_unit_weights_single_control(self):
        """Test unit weights with single control returns [1.0]."""
        from diff_diff._rust_backend import compute_sdid_unit_weights as rust_fn

        Y_pre = np.random.randn(5, 1)
        Y_tr_mean = np.random.randn(5)
        weights = rust_fn(Y_pre, Y_tr_mean, 0.5)
        assert weights.shape == (1,)
        assert abs(weights[0] - 1.0) < 1e-10

    def test_fw_gram_vs_standard_equivalence(self):
        """Test Gram path (T0 < N) and standard path produce equivalent results.

        Creates a problem where T0 < N (triggers Gram path in Rust), then
        verifies the Rust result matches pure Python exactly. This validates
        that the Gram precomputation optimization produces identical weights.
        """
        from diff_diff._rust_backend import sc_weight_fw as rust_fn
        from diff_diff.utils import _sc_weight_fw_numpy as numpy_fn

        np.random.seed(42)
        # N=50 rows, T0=8 columns + 1 target = 9 cols total
        # This triggers Gram path (T0=8 < N=50)
        Y = np.random.randn(50, 9)

        rust_w = rust_fn(Y, 0.3, True, None, 1e-5, 10000)
        numpy_w = numpy_fn(Y, 0.3, True, None, 1e-5, 10000)

        # Weights must match to high precision
        np.testing.assert_array_almost_equal(
            rust_w, numpy_w, decimal=6, err_msg="Gram path weights should match Python"
        )
        assert abs(rust_w.sum() - 1.0) < 1e-6
        assert np.all(rust_w >= -1e-6)

    def test_fw_standard_path_equivalence(self):
        """Test standard path (T0 >= N) produces results matching Python.

        Creates a problem where T0 >= N (triggers standard path in Rust),
        then verifies the Rust result matches pure Python exactly.
        """
        from diff_diff._rust_backend import sc_weight_fw as rust_fn
        from diff_diff.utils import _sc_weight_fw_numpy as numpy_fn

        np.random.seed(42)
        # N=5 rows, T0=12 columns + 1 target = 13 cols total
        # This triggers standard path (T0=12 >= N=5)
        Y = np.random.randn(5, 13)

        rust_w = rust_fn(Y, 0.5, True, None, 1e-5, 10000)
        numpy_w = numpy_fn(Y, 0.5, True, None, 1e-5, 10000)

        np.testing.assert_array_almost_equal(
            rust_w, numpy_w, decimal=6, err_msg="Standard path weights should match Python"
        )
        assert abs(rust_w.sum() - 1.0) < 1e-6
        assert np.all(rust_w >= -1e-6)

    def test_sdid_intercept_false_rust_vs_python(self):
        """Test intercept=false produces matching weights in both backends.

        Verifies both Gram and standard paths handle intercept=false correctly
        (no column centering applied).
        """
        from diff_diff._rust_backend import sc_weight_fw as rust_fn
        from diff_diff.utils import _sc_weight_fw_numpy as numpy_fn

        np.random.seed(42)

        # Gram path: T0 < N
        Y_gram = np.random.randn(30, 6)
        rust_w_gram = rust_fn(Y_gram, 0.2, False, None, 1e-5, 10000)
        numpy_w_gram = numpy_fn(Y_gram, 0.2, False, None, 1e-5, 10000)
        np.testing.assert_array_almost_equal(
            rust_w_gram,
            numpy_w_gram,
            decimal=6,
            err_msg="Gram path intercept=false weights should match Python",
        )

        # Standard path: T0 >= N
        Y_std = np.random.randn(4, 10)
        rust_w_std = rust_fn(Y_std, 0.2, False, None, 1e-5, 10000)
        numpy_w_std = numpy_fn(Y_std, 0.2, False, None, 1e-5, 10000)
        np.testing.assert_array_almost_equal(
            rust_w_std,
            numpy_w_std,
            decimal=6,
            err_msg="Standard path intercept=false weights should match Python",
        )

    def test_full_sdid_rust_vs_python(self):
        """Test full SDID estimation produces same results with Rust and Python."""
        import pandas as pd
        from unittest.mock import patch
        import sys
        from diff_diff import SyntheticDiD

        np.random.seed(42)
        n_control, n_treated, n_pre, n_post = 8, 1, 6, 2
        true_effect = 3.0
        data = []
        for i in range(n_control + n_treated):
            is_treated = i >= n_control
            for t in range(n_pre + n_post):
                post = t >= n_pre
                y = 1.0 + 0.5 * i + 0.3 * t + np.random.randn() * 0.3
                if is_treated and post:
                    y += true_effect
                data.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "treated": 1 if is_treated else 0,
                        "post": 1 if post else 0,
                    }
                )
        df = pd.DataFrame(data)
        post_periods = list(range(n_pre, n_pre + n_post))

        # Run with Rust backend
        sdid_rust = SyntheticDiD(variance_method="placebo", seed=42)
        results_rust = sdid_rust.fit(df, "outcome", "treated", "unit", "time", post_periods)

        # Run with Python backend
        utils_mod = sys.modules["diff_diff.utils"]
        with patch.object(utils_mod, "HAS_RUST_BACKEND", False):
            sdid_py = SyntheticDiD(variance_method="placebo", seed=42)
            results_py = sdid_py.fit(df.copy(), "outcome", "treated", "unit", "time", post_periods)

        # ATT should be very close
        np.testing.assert_almost_equal(
            results_rust.att, results_py.att, decimal=4, err_msg="Rust and Python ATT should match"
        )


# ---------------------------------------------------------------------------
# Silent-failure audit axis-G coverage (findings #21, #22, #23).
# The motivating incident (CS covariate scaling, early 2026) was a silent
# Rust-vs-Python divergence on rank-deficient / mixed-scale designs that
# sailed through the existing happy-path parity tests. These three classes
# extend backend-parity coverage to the edge cases the Phase-2 audit flagged.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestSolveOLSSkipRankCheckParity:
    """Finding #21: `solve_ols(..., skip_rank_check=True)` parity.

    Both backends skip the pivoted-QR rank check and trust the caller's
    full-rank assertion. Rust uses SVD; Python uses scipy.linalg.lstsq
    with ``cond=1e-7``. Parity is expected in well-conditioned cases, but
    near-singular and mixed-scale inputs could divergence on singular-
    value truncation ordering between LAPACK backends.

    Rust dispatch is gated by ``vcov_type='hc1'`` and ``weights is None``
    (linalg.py:621-634), so tests scope to that intersection — the only
    path where Rust runs under ``skip_rank_check=True``.

    Assertions target fitted values (X @ beta), not beta directly: for
    rank-deficient designs, kept-column beta values depend on pivot order
    while fitted values are backend-invariant.
    """

    def _run_both_backends(self, X, y):
        """Call solve_ols with skip_rank_check=True under Rust and under
        forced-Python, return (coef_rust, coef_py)."""
        import sys
        from unittest.mock import patch

        from diff_diff.linalg import solve_ols

        coef_rust, resid_rust, _ = solve_ols(
            X, y, skip_rank_check=True, vcov_type="hc1", return_vcov=False
        )

        linalg_module = sys.modules["diff_diff.linalg"]
        with (
            patch.object(linalg_module, "HAS_RUST_BACKEND", False),
            patch.object(linalg_module, "_rust_solve_ols", None),
        ):
            coef_py, resid_py, _ = solve_ols(
                X, y, skip_rank_check=True, vcov_type="hc1", return_vcov=False
            )
        return coef_rust, coef_py

    def test_mixed_scale_full_rank(self):
        """Full-rank X with column-norm ratio > 1e6. Both SVD backends
        should truncate the same singular values."""
        rng = np.random.default_rng(11)
        n, k = 80, 3
        X = np.column_stack(
            [
                rng.normal(0, 1, n),  # unit scale
                rng.normal(0, 1e6, n),  # 1e6 scale
                rng.normal(0, 1, n),  # unit scale
            ]
        )
        y = 1.0 + 0.5 * X[:, 0] + 1e-7 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 0.1, n)

        coef_rust, coef_py = self._run_both_backends(X, y)

        fitted_rust = X @ coef_rust
        fitted_py = X @ coef_py
        np.testing.assert_allclose(
            fitted_rust,
            fitted_py,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Rust vs Python fitted-value divergence on mixed-scale X",
        )

    def test_near_singular_full_rank(self):
        """Near-collinear full-rank X (cond(X'X) > 1e10). Backends use
        the same SVD threshold (cond=1e-7), so should agree on truncation."""
        rng = np.random.default_rng(22)
        n = 80
        x1 = rng.normal(0, 1, n)
        x2 = x1 + rng.normal(0, 1e-6, n)  # nearly parallel to x1
        x3 = rng.normal(0, 1, n)
        X = np.column_stack([x1, x2, x3])
        y = 1.0 + 0.5 * x1 - 0.3 * x3 + rng.normal(0, 0.1, n)

        coef_rust, coef_py = self._run_both_backends(X, y)

        fitted_rust = X @ coef_rust
        fitted_py = X @ coef_py
        np.testing.assert_allclose(
            fitted_rust,
            fitted_py,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Rust vs Python fitted-value divergence on near-singular X",
        )

    def test_rank_deficient_collinear(self):
        """Perfectly collinear columns. ``skip_rank_check=True`` bypasses
        the QR detector; both backends must still produce matching fitted
        values via minimum-norm SVD solve, even if individual coefficients
        differ on dropped columns."""
        rng = np.random.default_rng(33)
        n = 80
        x1 = rng.normal(0, 1, n)
        X = np.column_stack([x1, 2.0 * x1, rng.normal(0, 1, n)])  # x2 ≡ 2*x1
        y = 1.0 + 0.5 * x1 + 0.3 * X[:, 2] + rng.normal(0, 0.1, n)

        coef_rust, coef_py = self._run_both_backends(X, y)

        fitted_rust = X @ coef_rust
        fitted_py = X @ coef_py
        # Fitted values are the backend-invariant object under rank deficiency.
        np.testing.assert_allclose(
            fitted_rust,
            fitted_py,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Rust vs Python fitted-value divergence on rank-deficient X",
        )


# TestSyntheticWeightsBackendParity removed in the silent-failures audit
# post-cleanup (finding #22). The wrapper it tested was deleted; the FW
# computation is inlined in `rank_control_units` (prep.py:990) and covered
# there by tests/test_prep.py::TestRankControlUnits.


@pytest.mark.slow
@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestTROPRustEdgeCaseParity:
    """Finding #23: TROP Rust grid-search + bootstrap parity on edge cases.

    Existing parity tests (this file, ~line 1687 / 1757) compare ATT and
    effect dictionaries on well-conditioned random data at ``atol=1e-6``.
    Gap: rank-deficient Y on the grid-search path, seed reproducibility
    on the bootstrap path.

    Sizing kept minimal (n_units=6, n_periods=5–6) per the
    `feedback_trop_heavy_tests` memory.
    """

    @staticmethod
    def _make_correlated_panel(n_units=6, n_periods=5, n_treated=2):
        """Panel with two control units nearly parallel to each other,
        making the pre-period Y matrix near rank-deficient."""
        import pandas as pd

        rng = np.random.default_rng(13)
        data = []
        shared_path = rng.normal(0, 1, n_periods)
        for i in range(n_units):
            is_treated = i < n_treated
            if i in (n_treated, n_treated + 1):
                # Two control units share a near-identical trajectory
                base = shared_path + 1e-10 * rng.normal(0, 1, n_periods)
            else:
                base = rng.normal(0, 1, n_periods)
            for t in range(n_periods):
                y = 5.0 + i * 0.3 + base[t]
                treated = 1 if (is_treated and t >= n_periods - 2) else 0
                if treated:
                    y += 1.5
                data.append({"unit": i, "time": t, "outcome": y, "treated": treated})
        return pd.DataFrame(data)

    def test_grid_search_rank_deficient_Y(self):
        """Grid-search ATT parity on rank-deficient Y.

        Silent-failures audit Finding #23 (grid-search half) regression
        guard. Previously a ~6% ATT divergence on two near-parallel
        control units because the Rust inner solver used iterative block
        coordinate descent while the Python fallback used SVD-based
        minimum-norm least squares. Fixed by porting the Rust inner
        solver to an SVD-based WLS path (numpy-compatible
        rcond = eps*max(n,k)) that mirrors Python's
        `np.linalg.lstsq(rcond=None)` step-for-step. This test asserts
        the backends now agree at atol=1e-6 on rank-deficient Y.
        """
        import sys
        from unittest.mock import patch

        from diff_diff import TROP

        df = self._make_correlated_panel()
        # n_bootstrap>=2 is required by TROP.__init__; we set the minimum.
        # Bootstrap SE is NOT asserted here (see separate test below +
        # xfail baseline for the RNG-algorithm mismatch between backends).
        trop_params = dict(
            method="global",
            lambda_time_grid=[0.1, 1.0, 10.0],
            lambda_unit_grid=[0.1, 1.0, 10.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=2,
            seed=42,
        )

        trop_rust = TROP(**trop_params)
        res_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        trop_global_module = sys.modules["diff_diff.trop_global"]
        with (
            patch.object(trop_global_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_global_module, "_rust_loocv_grid_search_global", None),
            patch.object(trop_global_module, "_rust_bootstrap_trop_variance_global", None),
        ):
            trop_py = TROP(**trop_params)
            res_py = trop_py.fit(df.copy(), "outcome", "treated", "unit", "time")

        # Primary assertion: the ATT point estimate at the chosen λ matches.
        # This catches both (a) same λ chosen and (b) tied λ producing same fit.
        np.testing.assert_allclose(
            res_rust.att,
            res_py.att,
            atol=1e-6,
            err_msg="Grid-search ATT divergence on rank-deficient Y: "
            f"Rust={res_rust.att:.8f}, Python={res_py.att:.8f}",
        )

    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_bootstrap_seed_reproducibility(self, seed):
        """Bootstrap SE parity under a fixed seed (global method).

        Silent-failures audit Finding #23 (bootstrap half) regression guard.
        Previously a ~28% SE divergence on tiny panels because Rust seeded
        ``rand_xoshiro::Xoshiro256PlusPlus`` per replicate while Python
        consumed ``numpy.random.default_rng`` (PCG64). Fixed by pre-generating
        stratified bootstrap indices via numpy on the Python side and passing
        them to Rust through the PyO3 surface (Python-canonical RNG); both
        backends now consume bit-identical index streams under the same seed.
        """
        import sys
        from unittest.mock import patch

        from diff_diff import TROP

        df = self._make_correlated_panel(n_units=6, n_periods=6, n_treated=2)
        trop_params = dict(
            method="global",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[np.inf],
            n_bootstrap=10,
            seed=seed,
        )

        trop_rust = TROP(**trop_params)
        res_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        trop_global_module = sys.modules["diff_diff.trop_global"]
        with (
            patch.object(trop_global_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_global_module, "_rust_loocv_grid_search_global", None),
            patch.object(trop_global_module, "_rust_bootstrap_trop_variance_global", None),
        ):
            trop_py = TROP(**trop_params)
            res_py = trop_py.fit(df.copy(), "outcome", "treated", "unit", "time")

        np.testing.assert_allclose(
            res_rust.se,
            res_py.se,
            atol=1e-14,
            rtol=1e-14,
            err_msg=f"Bootstrap SE divergence under seed={seed}: "
            f"Rust={res_rust.se:.16f}, Python={res_py.se:.16f}",
        )

    @pytest.mark.parametrize(
        "seed,lambda_nn",
        [
            (0, np.inf),
            (42, np.inf),
            (12345, np.inf),
            (42, 0.1),
        ],
    )
    def test_bootstrap_seed_reproducibility_local(self, seed, lambda_nn):
        """Backend-invariant bootstrap SE parity for the local method.

        Post-methodology-alignment regression guard covering both the
        ``lambda_nn=inf`` regime (no-lowrank path, closed by the Python
        ``_precomputed`` cache-fallthrough removal) and the finite
        ``lambda_nn`` regime (with-lowrank FISTA path, closed by the Rust
        weight-matrix normalization removal). With the RNG fix from
        PR #354 plus both methodology fixes landed here, local-method
        Rust and Python bootstraps consume bit-identical stratified
        indices AND bit-identical raw-exponential weights. Main-fit ATT
        is bit-identical (see ``test_local_method_main_fit_parity``),
        but per-replicate bootstrap fits route through Rust's
        ``estimate_model`` vs numpy's ``lstsq``, which use different
        matrix factorization paths and accumulate different BLAS
        roundoff. Empirically the residual gap is ~1e-7 relative;
        asserted at ``atol=1e-5`` which is ~100x the observed gap and
        comfortable across CI runner variance.

        Follow-up to tighten to ``atol=1e-14``: unify Rust
        ``estimate_model`` to use ``solve_wls_svd`` (the same SVD path
        used by global-method since PR #348). Tracked in ``TODO.md``.
        """
        import sys
        from unittest.mock import patch

        from diff_diff import TROP

        df = self._make_correlated_panel(n_units=6, n_periods=6, n_treated=2)
        trop_params = dict(
            method="local",
            lambda_time_grid=[1.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[lambda_nn],
            n_bootstrap=10,
            seed=seed,
        )

        trop_rust = TROP(**trop_params)
        res_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        trop_local_module = sys.modules["diff_diff.trop_local"]
        with (
            patch.object(trop_local_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_local_module, "_rust_bootstrap_trop_variance", None),
        ):
            trop_py = TROP(**trop_params)
            res_py = trop_py.fit(df.copy(), "outcome", "treated", "unit", "time")

        np.testing.assert_allclose(
            res_rust.se,
            res_py.se,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"Local-method bootstrap SE divergence under "
            f"seed={seed}, lambda_nn={lambda_nn}: "
            f"Rust={res_rust.se:.16f}, Python={res_py.se:.16f}",
        )

    @pytest.mark.parametrize(
        "lambda_nn,tol",
        [
            (np.inf, 1e-14),
            (0.1, 1e-10),
        ],
    )
    def test_local_method_main_fit_parity(self, lambda_nn, tol):
        """Backend-invariant ATT parity for the local-method main fit.

        Companion to the bootstrap seed-parity test above. Exercises both
        regimes: ``lambda_nn=inf`` (no-lowrank, bit-identical minimum-norm
        WLS argmin under aligned raw-exponential weights) and a finite
        ``lambda_nn`` (with-lowrank, FISTA inner loop; tolerance relaxed
        to ``1e-10`` because FISTA iteration ordering and BLAS reduction
        ordering introduce sub-1e-10 noise across Rust faer and numpy BLAS
        paths).

        Regression guard for the normalization fix and cache-fallthrough
        fix landed in this PR. Before the fix, Rust ATT diverged from
        Python ATT by O(10%) at finite ``lambda_nn`` and O(0) at
        ``lambda_nn=inf``; after the fix both regimes match to tolerance.

        Uses multi-candidate lambda grids so LOOCV selection exercises
        Rust's `compute_weight_matrix` (the surface the normalization fix
        changed). Patches both the LOOCV dispatch in ``diff_diff.trop``
        and the bootstrap dispatch in ``diff_diff.trop_local`` on the
        Python side so the comparison is Rust-LOOCV-and-fit vs
        Python-LOOCV-and-fit end-to-end.
        """
        import sys
        from unittest.mock import patch

        from diff_diff import TROP

        df = self._make_correlated_panel(n_units=6, n_periods=6, n_treated=2)
        # Multi-candidate grids so LOOCV selection isn't trivial; the Rust
        # weight-normalization fix changes per-lambda LOOCV scores and thus
        # potentially the selected lambda.
        trop_params = dict(
            method="local",
            lambda_time_grid=[0.1, 1.0, 10.0],
            lambda_unit_grid=[0.1, 1.0, 10.0],
            lambda_nn_grid=[lambda_nn],
            n_bootstrap=2,  # minimum allowed; we assert ATT, not SE
            seed=42,
        )

        trop_rust = TROP(**trop_params)
        res_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        trop_module = sys.modules["diff_diff.trop"]
        trop_local_module = sys.modules["diff_diff.trop_local"]
        with (
            patch.object(trop_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_module, "_rust_loocv_grid_search", None),
            patch.object(trop_local_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_local_module, "_rust_bootstrap_trop_variance", None),
        ):
            trop_py = TROP(**trop_params)
            res_py = trop_py.fit(df.copy(), "outcome", "treated", "unit", "time")

        np.testing.assert_allclose(
            res_rust.att,
            res_py.att,
            atol=tol,
            rtol=tol,
            err_msg=f"Local-method ATT divergence at lambda_nn={lambda_nn}: "
            f"Rust={res_rust.att:.16f}, Python={res_py.att:.16f}",
        )

    @pytest.mark.parametrize("lambda_nn,tol", [(np.inf, 1e-14), (0.1, 1e-10)])
    def test_local_method_same_cohort_donor_parity(self, lambda_nn, tol):
        """Backend-invariant ATT when multiple units share the same treatment cohort.

        Isolates the ``D[t, j] == 1`` target-period case the prior ``_compute_
        observation_weights`` gate silently dropped: three treated units all
        starting at ``t=3``. Under the paper's Eq. 2/3, ``ω_j`` is
        distance-based for all ``j ≠ i`` (same-cohort donors included); their
        pre-treatment rows contribute via ``θ_s · ω_j`` and post-treatment
        cells are zeroed by the control mask ``(1 - W_{js})``. Python now
        matches this convention (gate removed). This regression asserts that
        the main-fit ATT is bit-identical across backends on a fixture where
        the gate would previously have excluded donors' pre-treatment
        information.
        """
        import sys
        from unittest.mock import patch

        import pandas as pd

        from diff_diff import TROP

        # 3 treated units + 3 controls, all treated units share cohort at t=3.
        # Pre-treatment trajectories are distinct so same-cohort donors carry
        # non-trivial information; without the fix Python and Rust would have
        # different effective donor pools at each treated-observation target.
        rng = np.random.default_rng(7)
        rows = []
        for i in range(6):
            is_treated = i < 3
            base = rng.normal(0, 1, 8)
            for t in range(8):
                y = 3.0 + i * 0.2 + 0.5 * t + base[t]
                treated = 1 if (is_treated and t >= 5) else 0
                if treated:
                    y += 1.5
                rows.append({"unit": i, "time": t, "outcome": y, "treated": treated})
        df = pd.DataFrame(rows)

        trop_params = dict(
            method="local",
            lambda_time_grid=[0.1, 1.0, 10.0],
            lambda_unit_grid=[0.1, 1.0, 10.0],
            lambda_nn_grid=[lambda_nn],
            n_bootstrap=2,
            seed=42,
        )

        trop_rust = TROP(**trop_params)
        res_rust = trop_rust.fit(df.copy(), "outcome", "treated", "unit", "time")

        trop_module = sys.modules["diff_diff.trop"]
        trop_local_module = sys.modules["diff_diff.trop_local"]
        with (
            patch.object(trop_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_module, "_rust_loocv_grid_search", None),
            patch.object(trop_local_module, "HAS_RUST_BACKEND", False),
            patch.object(trop_local_module, "_rust_bootstrap_trop_variance", None),
        ):
            trop_py = TROP(**trop_params)
            res_py = trop_py.fit(df.copy(), "outcome", "treated", "unit", "time")

        np.testing.assert_allclose(
            res_rust.att,
            res_py.att,
            atol=tol,
            rtol=tol,
            err_msg=f"Same-cohort donor ATT divergence at lambda_nn={lambda_nn}: "
            f"Rust={res_rust.att:.16f}, Python={res_py.att:.16f}",
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


from diff_diff._backend import _rust_demean_map as _demean_map_symbol


@pytest.mark.skipif(
    not HAS_RUST_BACKEND or _demean_map_symbol is None,
    reason="Rust backend or demean_map kernel not available",
)
class TestDemeanMapKernel:
    """Rust demean_map vs the canonical numpy engine (_demean_map_numpy).

    Contract: identical sweep order, row-order scatter-add accumulation, and
    max|x - x_old| < tol convergence per column (incl. NaN poisoning). The
    assertion order is iteration-count EQUALITY first (deterministic under
    the pinned op-order contract), then allclose on outputs.
    """

    @staticmethod
    def _fixture(kind, seed=0, k=3):
        rng = np.random.default_rng(seed)
        if kind == "balanced":
            n_units, n_periods = 40, 8
            unit = np.repeat(np.arange(n_units), n_periods)
            period = np.tile(np.arange(n_periods), n_units)
        elif kind == "unbalanced":
            n_units, n_periods = 60, 12
            unit = np.repeat(np.arange(n_units), n_periods)
            period = np.tile(np.arange(n_periods), n_units)
            keep = rng.random(unit.size) > 0.35
            unit, period = unit[keep], period[keep]
        elif kind == "contiguous":  # slow-convergence regime (>100 iterations)
            n_units, n_periods, span = 120, 40, 6
            unit = np.repeat(np.arange(n_units), n_periods)
            period = np.tile(np.arange(n_periods), n_units)
            entry = rng.integers(0, n_periods - span, n_units)
            keep = (period >= entry[unit]) & (period < entry[unit] + span)
            unit, period = unit[keep], period[keep]
        else:
            raise ValueError(kind)
        n = unit.size
        x_cols = [rng.normal(size=n) for _ in range(k)]
        codes_list = [
            pd.factorize(unit, sort=False)[0].astype(np.intp),
            pd.factorize(period, sort=False)[0].astype(np.intp),
        ]
        n_groups = [len(np.unique(unit)), len(np.unique(period))]
        w = rng.uniform(0.5, 2.0, n)
        return x_cols, codes_list, n_groups, w

    @staticmethod
    def _run_both(x_cols, codes_list, n_groups, weights, tol=1e-10, max_iter=10_000):
        from diff_diff.utils import _demean_map_numpy, _demean_map_rust

        rust = _demean_map_rust(x_cols, codes_list, n_groups, weights, tol, max_iter)
        assert rust is not None, "rust kernel unexpectedly fell back"
        numpy_res = _demean_map_numpy(x_cols, codes_list, n_groups, weights, tol, max_iter)
        return rust, numpy_res

    @pytest.mark.parametrize("kind", ["balanced", "unbalanced", "contiguous"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_equivalence_two_way(self, kind, weighted):
        x_cols, codes_list, n_groups, w = self._fixture(kind)
        (r_cols, r_iters), (p_cols, p_iters) = self._run_both(
            x_cols, codes_list, n_groups, w if weighted else None
        )
        assert r_iters == p_iters  # deterministic under the pinned op order
        if kind == "contiguous":
            assert all(it > 100 or it < 0 for it in p_iters) or max(p_iters) > 100
        for rc, pc in zip(r_cols, p_cols):
            np.testing.assert_allclose(rc, pc, rtol=0, atol=1e-12)

    def test_equivalence_three_way(self):
        rng = np.random.default_rng(3)
        x_cols, codes_list, n_groups, w = self._fixture("unbalanced", seed=3)
        n = x_cols[0].size
        firm = rng.integers(0, 7, n)
        codes_list = codes_list + [pd.factorize(firm, sort=False)[0].astype(np.intp)]
        n_groups = n_groups + [len(np.unique(firm))]
        (r_cols, r_iters), (p_cols, p_iters) = self._run_both(x_cols, codes_list, n_groups, w)
        assert r_iters == p_iters
        for rc, pc in zip(r_cols, p_cols):
            np.testing.assert_allclose(rc, pc, rtol=0, atol=1e-12)

    def test_zero_total_weight_group_rows_inert_parity(self):
        x_cols, codes_list, n_groups, w = self._fixture("unbalanced", seed=4)
        w = w.copy()
        zero_rows = codes_list[0] == 0
        w[zero_rows] = 0.0
        (r_cols, r_iters), (p_cols, p_iters) = self._run_both(x_cols, codes_list, n_groups, w)
        assert r_iters == p_iters
        for rc, pc in zip(r_cols, p_cols):
            np.testing.assert_allclose(rc, pc, rtol=0, atol=1e-12)
            assert np.isfinite(rc).all()

    def test_nan_in_variable_never_converges_both(self):
        x_cols, codes_list, n_groups, _ = self._fixture("unbalanced", seed=5, k=1)
        x_cols[0][3] = np.nan
        (_, r_iters), (_, p_iters) = self._run_both(
            x_cols, codes_list, n_groups, None, tol=1e-8, max_iter=25
        )
        assert r_iters == [-1]
        assert p_iters == [-1]

    @pytest.mark.parametrize("k", [1, 64])
    def test_column_counts(self, k):
        x_cols, codes_list, n_groups, _ = self._fixture("unbalanced", seed=6, k=k)
        (r_cols, r_iters), (p_cols, p_iters) = self._run_both(x_cols, codes_list, n_groups, None)
        assert len(r_cols) == k and r_iters == p_iters
        for rc, pc in zip(r_cols, p_cols):
            np.testing.assert_allclose(rc, pc, rtol=0, atol=1e-12)

    def test_nonconvergence_flag_parity_at_max_iter_1(self):
        x_cols, codes_list, n_groups, _ = self._fixture("unbalanced", seed=7)
        (_, r_iters), (_, p_iters) = self._run_both(
            x_cols, codes_list, n_groups, None, tol=1e-15, max_iter=1
        )
        assert r_iters == p_iters == [-1] * len(x_cols)

    def test_forced_fallback_runs_numpy_engine(self, monkeypatch):
        """Wrapper returning None must route demean_by_groups to the numpy
        engine and still produce a correct result."""
        import diff_diff.utils as utils_mod
        from diff_diff.utils import demean_by_groups

        rng = np.random.default_rng(8)
        df = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(20), 5),
                "period": np.tile(np.arange(5), 20),
                "y": rng.normal(size=100),
            }
        )
        calls = {"numpy": 0}
        orig_numpy = utils_mod._demean_map_numpy

        def counting_numpy(*a, **kw):
            calls["numpy"] += 1
            return orig_numpy(*a, **kw)

        monkeypatch.setattr(utils_mod, "_demean_map_rust", lambda *a, **kw: None)
        monkeypatch.setattr(utils_mod, "_demean_map_numpy", counting_numpy)
        out, _ = demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm")
        assert calls["numpy"] == 1
        assert np.abs(out["y_dm"].values.mean()) < 1e-12

    def test_nonconvergence_warning_parity_under_rust(self):
        """Same 'did not converge' warning, same variable names, via the
        rust dispatch path."""
        from diff_diff.utils import demean_by_groups

        rng = np.random.default_rng(9)
        df = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(30), 6),
                "period": np.tile(np.arange(6), 30),
            }
        )
        df = df[rng.random(len(df)) > 0.3].reset_index(drop=True)
        df["y"] = rng.normal(size=len(df))
        with pytest.warns(UserWarning, match=r"\['y'\].*did not converge"):
            demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm", max_iter=1, tol=1e-15)

    def test_estimator_level_att_parity(self, monkeypatch):
        """SunAbraham + DiD(absorb=) ATT/SE identical across engines,
        including the FE-spanned-covariate snap decision."""
        import warnings as _w

        import diff_diff.utils as utils_mod
        from diff_diff import DifferenceInDifferences, SunAbraham

        rng = np.random.default_rng(10)
        n_units, n_periods = 90, 10
        unit = np.repeat(np.arange(n_units), n_periods)
        time_ = np.tile(np.arange(n_periods), n_units)
        keep = rng.random(unit.size) > 0.25
        unit, time_ = unit[keep], time_[keep]
        first = np.where(np.arange(n_units) % 3 == 0, 0, 5)[unit]
        treated = (unit < n_units // 2).astype(int)
        post = (time_ >= n_periods // 2).astype(int)
        y = 0.3 * treated * post + rng.normal(size=unit.size)
        df = pd.DataFrame(
            {
                "y": y,
                "unit": unit,
                "time": time_,
                "first_treat": first,
                "treated": treated,
                "post": post,
            }
        )
        # FE-spanned covariate: snap decisions must match across engines
        a = rng.normal(size=n_units)
        b = rng.normal(size=n_periods)
        df["xspan"] = a[unit] + b[time_]

        def fits():
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                sa = SunAbraham().fit(
                    df, outcome="y", unit="unit", time="time", first_treat="first_treat"
                )
                did = DifferenceInDifferences().fit(
                    df,
                    outcome="y",
                    treatment="treated",
                    time="post",
                    absorb=["unit", "time"],
                    covariates=["xspan"],
                )
            return sa, did

        sa_r, did_r = fits()
        monkeypatch.setattr(utils_mod, "_rust_demean_map", None)
        sa_p, did_p = fits()
        assert sa_r.att == pytest.approx(sa_p.att, abs=1e-10)
        assert sa_r.se == pytest.approx(sa_p.se, rel=1e-8)
        assert did_r.att == pytest.approx(did_p.att, abs=1e-10)
        assert did_r.se == pytest.approx(did_p.se, rel=1e-8)
        # snap decision parity: spanned covariate NaN under BOTH engines
        assert np.isnan(did_r.coefficients["xspan"])
        assert np.isnan(did_p.coefficients["xspan"])

    def test_stale_symbol_none_falls_back_to_numpy(self, monkeypatch):
        """A mixed-version extension missing demean_map (symbol None) must
        route the PUBLIC entry point to the numpy engine, not raise."""
        import diff_diff.utils as utils_mod
        from diff_diff.utils import demean_by_groups

        rng = np.random.default_rng(11)
        df = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(15), 4),
                "period": np.tile(np.arange(4), 15),
                "y": rng.normal(size=60),
            }
        )
        monkeypatch.setattr(utils_mod, "_rust_demean_map", None)
        out, _ = demean_by_groups(df, ["y"], ["unit", "period"], suffix="_dm")
        assert np.abs(out["y_dm"].values.mean()) < 1e-12
        # the wrapper itself honors its documented None contract too
        assert (
            utils_mod._demean_map_rust(
                [df["y"].values], [np.zeros(60, dtype=np.intp)], [1], None, 1e-10, 10
            )
            is None
        )

    @staticmethod
    def _counting_kernel(monkeypatch):
        """Wrap the kernel symbol to count invocations (proves the chunked
        path actually ran, per-chunk)."""
        import diff_diff.utils as utils_mod

        calls = {"kernel": 0}
        orig = utils_mod._rust_demean_map

        def counting(*a, **kw):
            calls["kernel"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(utils_mod, "_rust_demean_map", counting)
        return calls

    @pytest.mark.parametrize("weighted", [False, True])
    def test_chunked_dispatch_exactly_equals_single_call(self, monkeypatch, weighted):
        """Chunking is exact partitioning: per-column outputs are IDENTICAL
        (assert_array_equal - same code path, no cross-backend caveat) and
        iteration counts equal, vs both single-chunk rust and the numpy
        engine."""
        import diff_diff.utils as utils_mod
        from diff_diff.utils import _demean_map_numpy, _demean_map_rust

        monkeypatch.delenv("DIFF_DIFF_DEMEAN_CHUNK_COLS", raising=False)
        x_cols, codes_list, n_groups, w = self._fixture("unbalanced", seed=13, k=8)
        weights = w if weighted else None

        monkeypatch.setattr(utils_mod, "_DEMEAN_MAP_CHUNK_COLS", 1000)
        single = _demean_map_rust(x_cols, codes_list, n_groups, weights, 1e-10, 10_000)
        assert single is not None

        calls = self._counting_kernel(monkeypatch)
        monkeypatch.setattr(utils_mod, "_DEMEAN_MAP_CHUNK_COLS", 3)
        chunked = _demean_map_rust(x_cols, codes_list, n_groups, weights, 1e-10, 10_000)
        assert chunked is not None
        assert calls["kernel"] == 3  # ceil(8 / 3)

        assert chunked[1] == single[1]
        for c_col, s_col in zip(chunked[0], single[0]):
            np.testing.assert_array_equal(c_col, s_col)

        numpy_res = _demean_map_numpy(x_cols, codes_list, n_groups, weights, 1e-10, 10_000)
        assert chunked[1] == numpy_res[1]
        for c_col, p_col in zip(chunked[0], numpy_res[0]):
            np.testing.assert_allclose(c_col, p_col, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("k", [1, 4, 5])  # below / at / above the chunk boundary
    def test_chunk_boundaries(self, monkeypatch, k):
        """k <= chunk is a single kernel call; k = chunk+1 exercises the
        two-chunk boundary (balanced partition: 5 -> 2+3). All cases match
        the numpy engine."""
        import math

        import diff_diff.utils as utils_mod
        from diff_diff.utils import _demean_map_numpy, _demean_map_rust

        monkeypatch.delenv("DIFF_DIFF_DEMEAN_CHUNK_COLS", raising=False)
        monkeypatch.setattr(utils_mod, "_DEMEAN_MAP_CHUNK_COLS", 4)
        x_cols, codes_list, n_groups, _ = self._fixture("balanced", seed=14, k=k)
        calls = self._counting_kernel(monkeypatch)
        rust = _demean_map_rust(x_cols, codes_list, n_groups, None, 1e-10, 10_000)
        assert rust is not None
        assert calls["kernel"] == math.ceil(k / 4)
        numpy_res = _demean_map_numpy(x_cols, codes_list, n_groups, None, 1e-10, 10_000)
        assert rust[1] == numpy_res[1]
        for r_col, p_col in zip(rust[0], numpy_res[0]):
            np.testing.assert_allclose(r_col, p_col, rtol=0, atol=1e-12)

    def test_nonconverged_variable_in_second_chunk_still_named(self, monkeypatch):
        """The non-convergence warning names a variable whose column lands in
        the SECOND chunk (iters aggregation preserves variable order)."""
        import diff_diff.utils as utils_mod
        from diff_diff.utils import demean_by_groups

        monkeypatch.delenv("DIFF_DIFF_DEMEAN_CHUNK_COLS", raising=False)
        monkeypatch.setattr(utils_mod, "_DEMEAN_MAP_CHUNK_COLS", 2)
        rng = np.random.default_rng(15)
        df = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(30), 6),
                "period": np.tile(np.arange(6), 30),
            }
        )
        df = df[rng.random(len(df)) > 0.3].reset_index(drop=True)
        for name in ["x1", "x2", "y"]:  # 'y' is column 3 -> chunk 2 of 2
            df[name] = rng.normal(size=len(df))
        calls = self._counting_kernel(monkeypatch)
        with pytest.warns(UserWarning, match=r"\['x1', 'x2', 'y'\].*did not converge"):
            demean_by_groups(
                df, ["x1", "x2", "y"], ["unit", "period"], suffix="_dm", max_iter=1, tol=1e-15
            )
        assert calls["kernel"] == 2  # ceil(3 / 2): multi-chunk path exercised

    def test_estimator_level_chunked_att_parity(self, monkeypatch):
        """DiD(absorb=) ATT/SE with chunk=2 identical to the default chunk."""
        import diff_diff.utils as utils_mod
        from diff_diff import DifferenceInDifferences

        monkeypatch.delenv("DIFF_DIFF_DEMEAN_CHUNK_COLS", raising=False)
        rng = np.random.default_rng(16)
        n_units, n_periods = 60, 8
        df = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(n_units), n_periods),
                "period": np.tile(np.arange(n_periods), n_units),
            }
        )
        df["treated"] = (df["unit"] < 30).astype(int)
        df["post"] = (df["period"] >= 4).astype(int)
        df["x1"] = rng.normal(size=len(df))
        df["x2"] = rng.normal(size=len(df))
        df["y"] = (
            1.5 * df["treated"] * df["post"]
            + 0.5 * df["x1"]
            - 0.3 * df["x2"]
            + rng.normal(0, 0.5, len(df))
        )

        def fit():
            return DifferenceInDifferences().fit(
                df,
                outcome="y",
                treatment="treated",
                time="post",
                absorb=["unit", "period"],
                covariates=["x1", "x2"],
            )

        r_default = fit()
        monkeypatch.setattr(utils_mod, "_DEMEAN_MAP_CHUNK_COLS", 2)
        r_chunked = fit()
        np.testing.assert_allclose(r_chunked.att, r_default.att, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r_chunked.se, r_default.se, rtol=0, atol=1e-12)

    def test_default_is_single_dispatch(self, monkeypatch):
        """With the env unset and the default constant (None), a k=8 dispatch
        makes exactly ONE kernel call - chunking is opt-in."""
        from diff_diff.utils import _demean_map_rust

        monkeypatch.delenv("DIFF_DIFF_DEMEAN_CHUNK_COLS", raising=False)
        x_cols, codes_list, n_groups, _ = self._fixture("balanced", seed=17, k=8)
        calls = self._counting_kernel(monkeypatch)
        result = _demean_map_rust(x_cols, codes_list, n_groups, None, 1e-10, 10_000)
        assert result is not None
        assert calls["kernel"] == 1


class TestDemeanChunkResolver:
    """Pure-Python env-override logic - runs regardless of the Rust build."""

    def test_default_when_unset_is_none(self, monkeypatch):
        """Chunking is OPT-IN: env unset -> None -> single dispatch."""
        import diff_diff.utils as utils_mod

        monkeypatch.delenv("DIFF_DIFF_DEMEAN_CHUNK_COLS", raising=False)
        assert utils_mod._DEMEAN_MAP_CHUNK_COLS is None
        assert utils_mod._resolve_demean_chunk_cols() is None

    def test_valid_override_honored(self, monkeypatch):
        import diff_diff.utils as utils_mod

        monkeypatch.setenv("DIFF_DIFF_DEMEAN_CHUNK_COLS", "7")
        assert utils_mod._resolve_demean_chunk_cols() == 7

    @pytest.mark.parametrize("bad", ["abc", "0", "-4", "", "3.5"])
    def test_invalid_values_fall_back_to_default(self, monkeypatch, bad):
        import diff_diff.utils as utils_mod

        monkeypatch.setenv("DIFF_DIFF_DEMEAN_CHUNK_COLS", bad)
        assert utils_mod._resolve_demean_chunk_cols() is utils_mod._DEMEAN_MAP_CHUNK_COLS


from diff_diff._backend import (  # noqa: E402
    _rust_batched_ridge_chol_solve as _batched_chol_symbol,
)


@pytest.mark.skipif(
    not HAS_RUST_BACKEND or _batched_chol_symbol is None,
    reason="Rust backend or batched-Cholesky kernel not available",
)
class TestBatchedRidgeCholSolve:
    """Rust `batched_ridge_chol_solve_ones` vs the numpy LU reference.

    Contract: solves (A_i + ridge_i * I) x = 1 per matrix via hand-rolled
    Cholesky; non-SPD rows fall back to faer LU; an exactly-singular row is
    NaN-poisoned (so the Python dispatch recomputes it via the legacy
    chain). Cholesky-vs-LU parity is cond*eps-bounded, NOT bit-identical:
    ~1e-12 on well-conditioned stacks, ~1e-5 budget on the cond~1e6-1e8
    ridged near-singular stacks production feeds it (measured ~4e-7 max on
    real Omega* batches). Per-row op order is fixed, so results are
    bit-deterministic across batch splits and thread counts.
    """

    @staticmethod
    def _numpy_reference(a_stack, ridge):
        m, h, _ = a_stack.shape
        a_ridged = a_stack + ridge[:, None, None] * np.eye(h)[None]
        return np.linalg.solve(a_ridged, np.ones((m, h, 1)))[..., 0]

    @staticmethod
    def _spd_stack(m, h, seed, eps=1.0):
        rng = np.random.default_rng(seed)
        b = rng.standard_normal((m, h, h))
        return b @ b.transpose(0, 2, 1) + eps * np.eye(h)

    @pytest.mark.parametrize("h", [2, 3, 30, 60])
    @pytest.mark.parametrize("m", [1, 200])
    def test_well_conditioned_parity(self, h, m):
        a = self._spd_stack(m, h, seed=h * 1000 + m)
        ridge = 1e-6 * np.trace(a, axis1=1, axis2=2) / h
        got = _batched_chol_symbol(a, ridge)
        want = self._numpy_reference(a, ridge)
        # atol covers near-zero solution entries: Cholesky-vs-LU error scales
        # with the solution norm (cond*eps*||x||), not per-entry.
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)

    def test_ill_conditioned_parity_cond_bounded(self):
        """Production regime: numerically singular PSD + trace-scaled ridge
        (floors relative eigenvalues at ~1e-8 -> cond ~1e8). Cholesky and LU
        then differ at the cond*eps level; budget 1e-5 (~30x the measured
        max on real Omega* stacks)."""
        rng = np.random.default_rng(9)
        m, h = 50, 20
        q, _ = np.linalg.qr(rng.standard_normal((h, h)))
        eigs = np.logspace(0, -12, h)  # exact-null tail beyond fp precision
        a1 = (q * eigs) @ q.T
        a = np.repeat(a1[None], m, axis=0) + 0.0
        # jitter each matrix a little to vary the batch (stay PSD)
        jit = rng.standard_normal((m, h, h)) * 1e-9
        a = a + jit @ jit.transpose(0, 2, 1)
        ridge = 1e-6 * np.trace(a, axis1=1, axis2=2) / h
        got = _batched_chol_symbol(a, ridge)
        want = self._numpy_reference(a, ridge)
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-8)

    def test_nan_row_non_finite(self):
        a = self._spd_stack(3, 4, seed=1)
        a[1] = np.nan
        ridge = np.full(3, 1e-6)
        got = _batched_chol_symbol(a, ridge)
        assert not np.isfinite(got[1]).all()
        np.testing.assert_allclose(got[[0, 2]], self._numpy_reference(a, ridge)[[0, 2]], rtol=1e-12)

    def test_exact_singular_zero_ridge_nan_poisoned(self):
        """diag(1, -2, 0) with zero ridge: Cholesky fails (negative pivot),
        faer LU sees an exactly-zero U pivot -> whole row NaN (mirrors
        LAPACK's exact-singularity signal; a finite-garbage row would
        silently skip the dispatch's legacy pinv recompute)."""
        a = np.zeros((1, 3, 3))
        a[0, 0, 0] = 1.0
        a[0, 1, 1] = -2.0
        got = _batched_chol_symbol(a, np.zeros(1))
        assert np.isnan(got).all()

    def test_indefinite_full_rank_lu_fallback(self):
        """diag(1, -1) with zero ridge: not SPD but invertible - the LU
        fallback returns the exact solution [1, -1]."""
        a = np.zeros((1, 2, 2))
        a[0, 0, 0] = 1.0
        a[0, 1, 1] = -1.0
        got = _batched_chol_symbol(a, np.zeros(1))
        np.testing.assert_array_equal(got[0], [1.0, -1.0])

    def test_batch_split_bit_identity(self):
        """Full-stack result == concatenated sub-batch results, bitwise.
        Locks the per-row-fixed-op-order determinism the EfficientDiD
        tile-invariance twins depend on."""
        a = self._spd_stack(31, 12, seed=5)
        ridge = 1e-6 * np.trace(a, axis1=1, axis2=2) / 12
        full = _batched_chol_symbol(a, ridge)
        parts = np.vstack(
            [
                _batched_chol_symbol(a[:7], ridge[:7]),
                _batched_chol_symbol(a[7:20], ridge[7:20]),
                _batched_chol_symbol(a[20:], ridge[20:]),
            ]
        )
        np.testing.assert_array_equal(full, parts)

    def test_degenerate_shapes(self):
        """m=0 and H=0 are no-ops with the right shape; H=1 is the scalar
        1/(a+ridge) via the 1x1 factorization (within 1 ulp)."""
        out_m0 = _batched_chol_symbol(np.zeros((0, 4, 4)), np.zeros(0))
        assert out_m0.shape == (0, 4)
        out_h0 = _batched_chol_symbol(np.zeros((3, 0, 0)), np.zeros(3))
        assert out_h0.shape == (3, 0)
        out_h1 = _batched_chol_symbol(np.full((1, 1, 1), 4.0), np.zeros(1))
        np.testing.assert_allclose(out_h1, [[0.25]], rtol=1e-14)

    def test_strided_input_defensive(self):
        """Non-contiguous views produce identical results to a contiguous
        copy (defensive: the live dispatch path's fancy-indexed stacks are
        always C-contiguous)."""
        a = self._spd_stack(20, 5, seed=8)
        ridge = np.full(20, 1e-6)
        strided = a[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        got = _batched_chol_symbol(strided, ridge[::2])
        want = _batched_chol_symbol(np.ascontiguousarray(strided), np.ascontiguousarray(ridge[::2]))
        np.testing.assert_array_equal(got, want)

    def test_shape_validation_errors(self):
        with pytest.raises(ValueError, match="square"):
            _batched_chol_symbol(np.zeros((2, 3, 4)), np.zeros(2))
        with pytest.raises(ValueError, match="ridge length"):
            _batched_chol_symbol(np.zeros((2, 3, 3)), np.zeros(5))
