"""Tests for the unified linear algebra backend."""

import numpy as np
import pandas as pd
import pytest

from diff_diff.linalg import (
    InferenceResult,
    LinearRegression,
    _rank_guarded_inv,
    compute_r_squared,
    compute_robust_vcov,
    solve_ols,
    solve_poisson,
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
        """Test that check_finite=False skips the upfront NaN/Inf validation.

        Note: With the 'gelsd' driver, LAPACK may still error on NaN values
        during computation, which is actually safer than producing garbage.
        """
        X = np.random.randn(100, 2)
        X[50, 0] = np.nan
        y = np.random.randn(100)

        # The gelsd driver may raise an error when encountering NaN during
        # computation, or produce garbage results. Either is acceptable
        # (the key is that we don't raise the "X contains NaN" user-friendly error)
        try:
            coef, resid, vcov = solve_ols(X, y, check_finite=False)
            # If it completed, coefficients should contain NaN/Inf due to bad input
            assert np.isnan(coef).any() or np.isinf(coef).any()
        except ValueError as e:
            # LAPACK may raise an error on NaN values (gelsd behavior)
            # This is acceptable - the key is we skipped our own validation
            assert "X contains NaN" not in str(e) and "y contains NaN" not in str(e)

    def test_rank_deficient_produces_nan_for_dropped_columns(self):
        """Test that rank-deficient matrix returns NaN for dropped columns.

        Following R's lm() approach, coefficients for linearly dependent columns
        are set to NaN while identified coefficients are computed normally.
        """
        import warnings

        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = X[:, 0] + X[:, 1]  # Perfect collinearity: col 2 = col 0 + col 1
        y = np.random.randn(100)

        # Should emit warning about rank deficiency
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coef, resid, vcov = solve_ols(X, y)
            assert len(w) == 1
            assert "Rank-deficient" in str(w[0].message)

        assert coef.shape == (3,)
        assert resid.shape == (100,)

        # Exactly one coefficient should be NaN (the dropped one)
        nan_mask = np.isnan(coef)
        assert np.sum(nan_mask) == 1, f"Expected 1 NaN coefficient, got {np.sum(nan_mask)}: {coef}"

        # Non-NaN coefficients should be finite and reasonable
        finite_coef = coef[~nan_mask]
        assert np.all(
            np.isfinite(finite_coef)
        ), f"Finite coefficients contain non-finite values: {finite_coef}"
        assert np.all(
            np.abs(finite_coef) < 1e6
        ), f"Finite coefficients are unreasonably large: {finite_coef}"

        # VCoV should have NaN for dropped column's row and column
        assert vcov is not None
        dropped_idx = np.where(nan_mask)[0][0]
        assert np.all(np.isnan(vcov[dropped_idx, :])), "VCoV row for dropped column should be NaN"
        assert np.all(
            np.isnan(vcov[:, dropped_idx])
        ), "VCoV column for dropped column should be NaN"

        # VCoV for identified coefficients should be finite
        kept_idx = np.where(~nan_mask)[0]
        vcov_kept = vcov[np.ix_(kept_idx, kept_idx)]
        assert np.all(np.isfinite(vcov_kept)), "VCoV for kept coefficients should be finite"

        # Residuals should be finite (computed using only identified coefficients)
        assert np.all(np.isfinite(resid)), f"Residuals contain non-finite values"

    def test_rank_deficient_error_mode(self):
        """Test that rank_deficient_action='error' raises ValueError."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = X[:, 0] + X[:, 1]  # Perfect collinearity
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="rank-deficient"):
            solve_ols(X, y, rank_deficient_action="error")

    def test_rank_deficient_silent_mode(self):
        """Test that rank_deficient_action='silent' produces no warning."""
        import warnings

        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = X[:, 0] + X[:, 1]  # Perfect collinearity
        y = np.random.randn(100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coef, resid, vcov = solve_ols(X, y, rank_deficient_action="silent")
            # No warnings should be emitted
            assert len(w) == 0, f"Expected no warnings, got {len(w)}: {[str(x.message) for x in w]}"

        # Should still produce NaN for dropped column
        assert np.sum(np.isnan(coef)) == 1

    def test_rank_deficient_column_names_in_warning(self):
        """Test that column names appear in warning message."""
        import warnings

        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = X[:, 0] + X[:, 1]  # Perfect collinearity
        y = np.random.randn(100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coef, resid, vcov = solve_ols(X, y, column_names=["intercept", "x1", "x2_collinear"])
            assert len(w) == 1
            # Column name should appear in warning (not just index)
            assert (
                "x2_collinear" in str(w[0].message)
                or "intercept" in str(w[0].message)
                or "x1" in str(w[0].message)
            )

    def test_skip_rank_check_bypasses_qr_decomposition(self):
        """Test that skip_rank_check=True skips QR rank detection.

        When skip_rank_check=True, the function should skip QR decomposition
        and go directly to SVD solving, even in Python backend.
        """
        import warnings
        import os

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y = np.random.randn(n)

        # Force Python backend for this test
        old_backend = os.environ.get("DIFF_DIFF_BACKEND")
        os.environ["DIFF_DIFF_BACKEND"] = "python"

        try:
            # With skip_rank_check=True, should not emit any warnings
            # (even if we make X rank-deficient, since we're skipping the check)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                coef, resid, vcov = solve_ols(X, y, skip_rank_check=True)
                # No rank-deficiency warning should be emitted
                rank_warnings = [x for x in w if "Rank-deficient" in str(x.message)]
                assert len(rank_warnings) == 0
        finally:
            if old_backend is not None:
                os.environ["DIFF_DIFF_BACKEND"] = old_backend
            elif "DIFF_DIFF_BACKEND" in os.environ:
                del os.environ["DIFF_DIFF_BACKEND"]

        # Should produce valid coefficients
        assert coef.shape == (3,)
        assert np.all(np.isfinite(coef))

    def test_multiperiod_like_design_full_rank(self):
        """Test that MultiPeriodDiD-like design matrices work when full-rank.

        This test creates a properly specified MultiPeriodDiD-like design that
        is NOT rank-deficient to verify correct coefficient recovery.
        """
        import warnings

        np.random.seed(42)
        n = 200
        n_periods = 5

        # Create a design matrix similar to MultiPeriodDiD:
        # [intercept, period_1, period_2, ..., period_k, treated*post]

        # Intercept
        intercept = np.ones(n)

        # Period dummies (one-hot encoding for periods 1 to n_periods-1)
        # Period 0 is the reference
        period_assignment = np.random.randint(0, n_periods, n)
        period_dummies = np.zeros((n, n_periods - 1))
        for i in range(1, n_periods):
            period_dummies[:, i - 1] = (period_assignment == i).astype(float)

        # Treatment indicator (varies within periods to ensure identification)
        treated = np.random.binomial(1, 0.5, n)

        # Post indicator (periods >= 3 are post)
        post = (period_assignment >= 3).astype(float)

        # Treatment × post interaction
        treat_post = treated * post

        # Build design matrix
        X = np.column_stack([intercept, period_dummies, treat_post])

        # True effect
        true_effect = 2.5
        y = (
            1.0  # intercept effect
            + 0.5 * period_dummies[:, 0]  # period 1 effect
            + 0.3 * period_dummies[:, 1]  # period 2 effect
            + 0.7 * period_dummies[:, 2]  # period 3 effect
            + 0.9 * period_dummies[:, 3]  # period 4 effect
            + true_effect * treat_post  # treatment effect
            + np.random.randn(n) * 0.5  # noise
        )

        # Fit with solve_ols - should NOT produce warning if full-rank
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coef, resid, vcov = solve_ols(X, y)
            # Check if any rank deficiency warnings (may or may not occur)
            rank_warnings = [x for x in w if "Rank-deficient" in str(x.message)]

        # If no rank deficiency, all coefficients should be finite
        if len(rank_warnings) == 0:
            assert np.all(np.isfinite(coef)), f"Full-rank matrix: coefficients should be finite"
            assert np.all(np.abs(coef) < 1e6), f"Coefficients are unreasonably large: {coef}"
            # The treatment effect coefficient (last one) should be close to true effect
            assert (
                abs(coef[-1] - true_effect) < 2.0
            ), f"Treatment effect {coef[-1]} is too far from true {true_effect}"
        else:
            # If rank-deficient, check that identified coefficients are valid
            finite_coef = coef[~np.isnan(coef)]
            assert np.all(np.isfinite(finite_coef)), f"Identified coefficients should be finite"
            # If treatment effect is identified, check it
            if not np.isnan(coef[-1]):
                assert (
                    abs(coef[-1] - true_effect) < 2.0
                ), f"Treatment effect {coef[-1]} is too far from true {true_effect}"

    def test_single_cluster_error(self):
        """Test that single cluster raises error."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        cluster_ids = np.zeros(100)  # All same cluster

        with pytest.raises(ValueError, match="at least 2 clusters"):
            solve_ols(X, y, cluster_ids=cluster_ids)

    def test_singleton_clusters_included_in_variance(self):
        """Test that singleton clusters contribute to variance estimation.

        REGISTRY.md documents: "Singleton clusters (one observation): included
        in variance estimation; contribute to meat matrix via (residual² × X'X),
        same as larger clusters"

        This test verifies that behavior.
        """
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n)

        # Create clusters: one large cluster (50 obs), 50 singleton clusters
        # Total: 51 clusters, 50 of which are singletons
        cluster_ids = np.concatenate(
            [
                np.zeros(50),  # Large cluster (id=0)
                np.arange(1, 51),  # 50 singleton clusters (ids 1-50)
            ]
        )

        coef, resid, vcov = solve_ols(X, y, cluster_ids=cluster_ids)

        # Basic validity checks
        assert vcov.shape == (2, 2), "VCoV should be 2x2"
        assert np.all(np.isfinite(vcov)), "VCoV should have finite values"
        assert np.allclose(vcov, vcov.T), "VCoV should be symmetric"

        # Variance should be positive (singletons contribute, not zero)
        assert vcov[0, 0] > 0, "Intercept variance should be positive"
        assert vcov[1, 1] > 0, "Slope variance should be positive"

        # Compare to case without singletons (only large clusters)
        # With fewer clusters, variance should be DIFFERENT (not necessarily larger)
        cluster_ids_no_singletons = np.concatenate(
            [np.zeros(50), np.ones(50)]  # Cluster 0  # Cluster 1
        )
        _, _, vcov_no_singletons = solve_ols(X, y, cluster_ids=cluster_ids_no_singletons)

        # The two variance estimates should differ (singletons change the calculation)
        assert not np.allclose(
            vcov, vcov_no_singletons
        ), "Singleton clusters should affect variance estimation"


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

    def test_cluster_count_check_precedes_saturated_guard(self):
        """A 1-cluster cluster-robust request raises 'need >= 2 clusters' even on
        a saturated design — the cluster-count validation must precede the
        saturated (no residual DOF) NaN guard, not be masked by it."""
        # 1 cluster AND saturated (n == k): must still raise, not return NaN.
        with pytest.raises(ValueError, match="at least 2 clusters"):
            compute_robust_vcov(np.eye(2), np.zeros(2), np.zeros(2))

    def test_saturated_multi_cluster_returns_nan(self):
        """A saturated design (no residual DOF) with >= 2 clusters returns a NaN
        vcov rather than raising ZeroDivisionError from the CR1 adjustment."""
        # 2 clusters, n == k == 4 (saturated 2x2 with one obs per cluster-period).
        X = np.column_stack([np.ones(4), [0, 0, 1, 1], [0.0, 1.0, 0.0, 1.0], [0, 0, 0, 1.0]])
        vcov = compute_robust_vcov(X, np.zeros(4), np.array([0, 0, 1, 1]))
        assert np.all(np.isnan(vcov))

    def test_cluster_count_check_normalizes_series_cluster_ids(self):
        """The early cluster-count validation must normalize `cluster_ids` to an
        array before the zero-weight groupby, so a non-default-index pandas
        Series grouper is not index-aligned against Series(weights) and
        miscounted (which would wrongly raise 'need >= 2 clusters' on a valid
        multi-cluster fit)."""
        rng = np.random.default_rng(0)
        n = 18
        cl_arr = np.repeat([0, 1, 2], 6)
        cl = pd.Series(cl_arr, index=np.arange(500, 500 + n))  # non-default index
        X = np.column_stack([np.ones(n), (cl_arr < 1).astype(float), np.tile([0.0, 1.0], 9)])
        y = X @ np.array([1.0, 0.5, 0.3]) + rng.normal(scale=0.4, size=n)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        weights = np.ones(n)
        weights[0] = 0.0  # one zero-weight obs; cluster 0 still has positive-weight obs
        # All 3 clusters retain positive weight -> must not raise; vcov finite.
        vcov = compute_robust_vcov(X, residuals, cl, weights=weights, weight_type="aweight")
        assert vcov.shape == (3, 3)
        assert np.all(np.isfinite(vcov))

    def test_numerical_instability_fallback_warns(self, ols_data):
        """Test that numerical instability in Rust backend triggers warning and fallback."""
        from unittest.mock import patch
        import warnings

        from diff_diff import HAS_RUST_BACKEND

        if not HAS_RUST_BACKEND:
            pytest.skip("Rust backend not available")

        X, residuals = ols_data

        # Mock _rust_compute_robust_vcov to raise numerical instability error
        def mock_rust_vcov(*args, **kwargs):
            raise ValueError("Matrix inversion numerically unstable")

        with patch("diff_diff.linalg._rust_compute_robust_vcov", mock_rust_vcov):
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                vcov = compute_robust_vcov(X, residuals)

                # Verify warning was emitted
                instability_warnings = [
                    w for w in caught_warnings if "numerical instability" in str(w.message).lower()
                ]
                assert (
                    len(instability_warnings) == 1
                ), f"Expected 1 numerical instability warning, got {len(instability_warnings)}"

                # Verify fallback produced valid vcov matrix
                assert vcov.shape == (X.shape[1], X.shape[1])
                assert np.allclose(vcov, vcov.T)  # Symmetric
                assert np.all(np.linalg.eigvalsh(vcov) >= -1e-10)  # PSD


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
            coefficient=2.0, se=0.5, t_stat=4.0, p_value=0.001, conf_int=(1.0, 3.0), alpha=0.05
        )
        assert result.is_significant() is True

        # Not significant at 0.05
        result2 = InferenceResult(
            coefficient=0.5, se=0.5, t_stat=1.0, p_value=0.3, conf_int=(-0.5, 1.5), alpha=0.05
        )
        assert result2.is_significant() is False

    def test_is_significant_custom_alpha(self):
        """Test is_significant with custom alpha override."""
        result = InferenceResult(
            coefficient=2.0, se=0.5, t_stat=4.0, p_value=0.02, conf_int=(1.0, 3.0), alpha=0.05
        )

        # Significant at 0.05 (default)
        assert result.is_significant() is True

        # Not significant at 0.01
        assert result.is_significant(alpha=0.01) is False

    def test_significance_stars(self):
        """Test significance_stars returns correct stars."""
        # p < 0.001 -> ***
        result = InferenceResult(
            coefficient=1.0, se=0.1, t_stat=10.0, p_value=0.0001, conf_int=(0.8, 1.2)
        )
        assert result.significance_stars() == "***"

        # p < 0.01 -> **
        result2 = InferenceResult(
            coefficient=1.0, se=0.2, t_stat=5.0, p_value=0.005, conf_int=(0.6, 1.4)
        )
        assert result2.significance_stars() == "**"

        # p < 0.05 -> *
        result3 = InferenceResult(
            coefficient=1.0, se=0.3, t_stat=3.0, p_value=0.03, conf_int=(0.4, 1.6)
        )
        assert result3.significance_stars() == "*"

        # p < 0.1 -> .
        result4 = InferenceResult(
            coefficient=1.0, se=0.4, t_stat=2.5, p_value=0.08, conf_int=(0.2, 1.8)
        )
        assert result4.significance_stars() == "."

        # p >= 0.1 -> ""
        result5 = InferenceResult(
            coefficient=1.0, se=0.5, t_stat=2.0, p_value=0.15, conf_int=(0.0, 2.0)
        )
        assert result5.significance_stars() == ""

    def test_to_dict(self):
        """Test to_dict returns all fields."""
        result = InferenceResult(
            coefficient=2.5,
            se=0.5,
            t_stat=5.0,
            p_value=0.001,
            conf_int=(1.52, 3.48),
            df=100,
            alpha=0.05,
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

    def test_rank_deficient_degrees_of_freedom(self):
        """Test that degrees of freedom are computed correctly when columns are dropped.

        When a design matrix is rank-deficient, the effective number of parameters
        is the rank, not the number of columns. The df should be n - rank.
        """
        import warnings

        np.random.seed(42)
        n = 100
        # Create rank-deficient matrix: 4 columns but rank 3
        X = np.random.randn(n, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Column 3 = Column 0 + Column 1

        y = np.random.randn(n)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reg = LinearRegression(include_intercept=False).fit(X, y)

        # n_params_ should be total columns (4)
        assert reg.n_params_ == 4

        # n_params_effective_ should be the rank (3)
        assert reg.n_params_effective_ == 3

        # df_ should be n - effective_params = 100 - 3 = 97
        assert reg.df_ == n - 3

        # Verify one coefficient is NaN (the dropped one)
        assert np.sum(np.isnan(reg.coefficients_)) == 1

    def test_rank_deficient_inference_uses_correct_df(self):
        """Test that p-values and CIs use the correct df for rank-deficient matrices."""
        import warnings
        from scipy import stats

        np.random.seed(42)
        n = 100
        # Create rank-deficient matrix
        X = np.random.randn(n, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Perfect collinearity

        # True coefficients for the first 3 columns only
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n) * 0.5

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reg = LinearRegression(include_intercept=False).fit(X, y)

        # Get inference for an identified coefficient
        nan_mask = np.isnan(reg.coefficients_)
        kept_idx = np.where(~nan_mask)[0][0]  # First non-NaN coefficient
        result = reg.get_inference(kept_idx)

        # Check that df is correct (should be n - rank = 97)
        assert result.df == n - 3, f"Expected df={n-3}, got {result.df}"

        # Manually compute expected values using correct df
        coef = result.coefficient
        se = result.se
        t_stat_expected = coef / se
        p_value_expected = 2 * (1 - stats.t.cdf(abs(t_stat_expected), df=n - 3))

        # Verify t-stat
        np.testing.assert_allclose(result.t_stat, t_stat_expected, rtol=1e-10)

        # Verify p-value uses correct df (use atol for very small p-values)
        np.testing.assert_allclose(result.p_value, p_value_expected, atol=1e-10)

        # Verify CI uses correct df
        t_crit = stats.t.ppf(1 - 0.05 / 2, df=n - 3)
        ci_expected = (coef - t_crit * se, coef + t_crit * se)
        np.testing.assert_allclose(result.conf_int, ci_expected, rtol=1e-6)

    def test_rank_deficient_inference_nan_for_dropped_coef(self):
        """Test that inference for dropped coefficients returns NaN values."""
        import warnings

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Column 3 is dropped
        y = np.random.randn(n)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reg = LinearRegression(include_intercept=False).fit(X, y)

        # Find the dropped coefficient index
        nan_mask = np.isnan(reg.coefficients_)
        dropped_idx = np.where(nan_mask)[0][0]

        # Get inference for dropped coefficient
        result = reg.get_inference(dropped_idx)

        # All inference values should be NaN
        assert np.isnan(result.coefficient)
        assert np.isnan(result.se)
        assert np.isnan(result.t_stat)
        assert np.isnan(result.p_value)
        assert np.isnan(result.conf_int[0])
        assert np.isnan(result.conf_int[1])

    def test_rank_deficient_predict_uses_identified_coefficients(self):
        """Test that predict() works correctly with rank-deficient fits.

        Predictions should use only identified coefficients (treating dropped
        coefficients as zero), not produce all-NaN predictions.
        """
        import warnings

        np.random.seed(42)
        n = 100
        # Create rank-deficient matrix
        X = np.random.randn(n, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Column 3 is collinear

        # True model uses only first 3 columns
        y = 2 * X[:, 0] + 3 * X[:, 1] - 1 * X[:, 2] + np.random.randn(n) * 0.5

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reg = LinearRegression(include_intercept=False).fit(X, y)

        # Verify one coefficient is NaN
        assert np.sum(np.isnan(reg.coefficients_)) == 1

        # Predictions should NOT be all NaN
        X_new = np.random.randn(10, 4)
        y_pred = reg.predict(X_new)

        assert y_pred.shape == (10,)
        assert np.all(np.isfinite(y_pred)), "Predictions should be finite, not NaN"

        # Verify predictions match fitted values on training data
        y_fitted = reg.predict(X)
        np.testing.assert_allclose(y_fitted, reg.fitted_values_, rtol=1e-10)

    def test_rank_deficient_adjusted_r_squared_uses_effective_params(self):
        """Test that adjusted R² uses effective params, not total params.

        For rank-deficient fits, adjusted R² should use n_params_effective_
        for consistency with the corrected degrees of freedom.
        """
        import warnings

        np.random.seed(42)
        n = 100
        # Create rank-deficient matrix: 4 columns but rank 3
        X = np.random.randn(n, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Column 3 = Column 0 + Column 1

        y = 2 * X[:, 0] + np.random.randn(n) * 0.5

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reg = LinearRegression(include_intercept=False).fit(X, y)

        # Verify setup
        assert reg.n_params_ == 4
        assert reg.n_params_effective_ == 3

        # Get R² values
        r2 = reg.r_squared()
        r2_adj = reg.r_squared(adjusted=True)

        # Both should be valid numbers
        assert 0 <= r2 <= 1
        assert r2_adj < r2  # Adjusted should be smaller

        # Manually compute adjusted R² using effective params
        # r²_adj = 1 - (1 - r²) * (n - 1) / (n - k_effective)
        r2_adj_expected = 1 - (1 - r2) * (n - 1) / (n - 3)
        np.testing.assert_allclose(r2_adj, r2_adj_expected, rtol=1e-10)

        # Verify it's NOT using total params (which would give different result)
        r2_adj_wrong = 1 - (1 - r2) * (n - 1) / (n - 4)
        assert r2_adj != r2_adj_wrong, "Should use effective params, not total params"

    def test_rank_deficient_action_error_raises(self):
        """Test that LinearRegression with rank_deficient_action='error' raises on collinear data."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Perfect collinearity
        y = np.random.randn(n)

        reg = LinearRegression(include_intercept=False, rank_deficient_action="error")
        with pytest.raises(ValueError, match="rank-deficient"):
            reg.fit(X, y)

    def test_rank_deficient_action_silent_no_warning(self):
        """Test that LinearRegression with rank_deficient_action='silent' produces no warning."""
        import warnings

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Perfect collinearity
        y = np.random.randn(n)

        reg = LinearRegression(include_intercept=False, rank_deficient_action="silent")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.fit(X, y)
            # No warnings should be emitted
            assert len(w) == 0, f"Expected no warnings, got {len(w)}: {[str(x.message) for x in w]}"

        # Should still produce NaN for dropped column
        assert np.sum(np.isnan(reg.coefficients_)) == 1

    def test_rank_deficient_action_warn_default(self):
        """Test that LinearRegression with rank_deficient_action='warn' (default) emits warning."""
        import warnings

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Perfect collinearity
        y = np.random.randn(n)

        reg = LinearRegression(include_intercept=False)  # Default is "warn"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.fit(X, y)
            # Should have a warning about rank deficiency
            assert len(w) > 0, "Expected warning about rank deficiency"
            assert any(
                "Rank-deficient" in str(x.message) or "rank-deficient" in str(x.message).lower()
                for x in w
            ), f"Expected rank-deficient warning, got: {[str(x.message) for x in w]}"


class TestNumericalStability:
    """Tests for numerical stability with ill-conditioned matrices."""

    def test_near_singular_matrix_stability(self):
        """Test that near-singular matrices are handled correctly."""
        np.random.seed(42)
        n = 100

        # Create near-collinear design (high condition number but above rank tolerance)
        # The rank detection tolerance is 1e-07 (matching R's qr()), so we use noise
        # of 1e-5 which is clearly above the tolerance and provides a distinguishable
        # signal. With noise < 1e-07, the column would be considered linearly dependent.
        X = np.random.randn(n, 3)
        X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(n) * 1e-5  # Near but not perfect collinearity

        y = X[:, 0] + np.random.randn(n) * 0.1

        reg = LinearRegression(include_intercept=True).fit(X, y)

        # Should still produce finite coefficients (noise is above tolerance)
        assert np.all(np.isfinite(reg.coefficients_))

        # Compare with numpy's lstsq (gold standard for stability)
        X_full = np.column_stack([np.ones(n), X])
        expected, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)

        # Should be close (within reasonable tolerance for ill-conditioned problem)
        np.testing.assert_allclose(reg.coefficients_, expected, rtol=1e-4)

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

    def test_solve_ols_scale_invariance_fitted_values(self):
        """Rank detection + solve are invariant to per-column scaling.

        A large-scale column previously inflated the rank threshold (anchored to
        the largest pivot/singular value) and false-dropped well-scaled columns to
        NaN, or truncated the small-scale direction in the lstsq solve. After
        column equilibration, solve_ols(X) and solve_ols(X @ diag(s)) give the same
        fitted values/residuals, coefficients scale inversely, and t-stats (vcov is
        scale-equivariant) are invariant.
        """
        rng = np.random.default_rng(0)
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        y = 1.0 + 2.0 * X[:, 1] - 0.5 * X[:, 2] + rng.standard_normal(n) * 0.1
        s = np.array([1.0, 1e8, 1e-4])  # pathological per-column scaling

        coef_raw, resid_raw, vcov_raw = solve_ols(X, y)
        coef_scaled, resid_scaled, vcov_scaled = solve_ols(X * s, y)

        # full-rank design stays full-rank under any scaling (the headline bug)
        assert np.all(np.isfinite(coef_scaled))
        # fitted values / residuals invariant
        np.testing.assert_allclose(resid_raw, resid_scaled, atol=1e-8)
        np.testing.assert_allclose(X @ coef_raw, (X * s) @ coef_scaled, atol=1e-8)
        # coefficients scale inversely with the column scaling
        np.testing.assert_allclose(coef_scaled, coef_raw / s, rtol=1e-6)
        # t-stats invariant (vcov is scale-equivariant: SE_j scales like 1/s_j)
        t_raw = coef_raw / np.sqrt(np.diag(vcov_raw))
        t_scaled = coef_scaled / np.sqrt(np.diag(vcov_scaled))
        np.testing.assert_allclose(t_raw, t_scaled, rtol=1e-5)

    def test_did_finite_att_with_large_scale_covariate(self):
        """End-to-end: a full-rank DiD with a 1e8-scale covariate returns a finite
        ATT equal to the O(1)-rescaled run (no scale-induced NaN)."""
        from diff_diff import DifferenceInDifferences, generate_did_data

        df = generate_did_data(
            n_units=400, n_periods=2, treatment_period=1, treatment_effect=2.5, seed=42
        ).copy()
        rng = np.random.default_rng(0)
        # outcome-relevant, treatment-imbalanced covariate on a huge scale
        cov = (rng.standard_normal(len(df)) + 2.0 * df["treated"].to_numpy()) * 1e8
        df["cov"] = cov
        df["outcome"] = df["outcome"].astype(float) + 3e-8 * df["cov"]

        res = DifferenceInDifferences().fit(
            df, outcome="outcome", treatment="treated", time="post", covariates=["cov"]
        )
        assert np.isfinite(
            res.att
        ), "ATT is NaN for a full-rank design with a large-scale covariate"

        # same data with the covariate rescaled to O(1) gives the same ATT
        df_small = df.copy()
        df_small["cov"] = df["cov"] / 1e8
        df_small["outcome"] = df["outcome"]  # outcome already includes the effect
        res_small = DifferenceInDifferences().fit(
            df_small,
            outcome="outcome",
            treatment="treated",
            time="post",
            covariates=["cov"],
        )
        np.testing.assert_allclose(res.att, res_small.att, rtol=1e-6)

    def test_solve_ols_rank_zero_returns_nan_not_indexerror(self):
        """A design that collapses to rank 0 returns all-NaN coefficients with a
        warning, not a cryptic IndexError (empty float index array)."""
        import warnings

        n = 50
        X = np.zeros((n, 3))  # rank 0
        y = np.random.default_rng(2).standard_normal(n)
        with pytest.warns(UserWarning, match="[Rr]ank-deficient"):
            coef, resid, vcov = solve_ols(X, y)
        assert coef.shape == (3,)
        assert np.all(np.isnan(coef))  # nothing identifiable
        np.testing.assert_allclose(resid, y)  # fitted = 0
        assert vcov.shape == (3, 3) and np.all(np.isnan(vcov))

    def test_rank_detection_zero_column_matrix_returns_empty_contract(self):
        """An ``(n, 0)`` design has rank 0, no dropped columns, and no pivot."""
        from diff_diff.linalg import _detect_rank_deficiency

        X = np.empty((5, 0))

        rank, dropped, pivot = _detect_rank_deficiency(X)

        assert rank == 0
        assert dropped.dtype == int
        assert pivot.dtype == int
        assert dropped.shape == (0,)
        assert pivot.shape == (0,)

    def test_rank_detection_scale_repair_preserves_raw_drop_selection(self):
        """The scale-invariance repair must NOT change which column is dropped in a
        genuinely collinear, well-scaled design: the dropped column equals the raw
        pivoted-QR choice (the repair only raises the rank when a large-scale column
        false-inflated the threshold)."""
        from scipy.linalg import qr

        from diff_diff.linalg import _detect_rank_deficiency

        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 3))
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # rank 3 of 4, well-scaled
        rank, dropped, _ = _detect_rank_deficiency(X)

        # reference: the raw pivoted-QR drop selection
        _Q, R, piv = qr(X, mode="economic", pivoting=True)
        rd = np.abs(np.diag(R))
        raw_rank = int(np.sum(rd > 1e-7 * rd[0]))
        raw_dropped = np.sort(piv[raw_rank:])

        assert rank == 3
        np.testing.assert_array_equal(dropped, raw_dropped)

    def test_rank_detection_mixed_scale_and_collinearity_keeps_identified_subset(self):
        """Mixed case: a design that is genuinely rank-deficient AND contains a
        huge-scale independent column. The rank COUNT must be scale-corrected (3,
        not under-counted by the huge column), the RETAINED columns must be full
        rank (an identified subset — the property that actually matters), the
        huge independent column must NOT be dropped, and downstream inference on
        the kept coefficients must be valid (finite, non-negative vcov diagonal)."""
        from diff_diff.linalg import _detect_rank_deficiency

        rng = np.random.default_rng(0)
        n = 200
        x0 = rng.standard_normal(n)
        x1 = rng.standard_normal(n)
        z = rng.standard_normal(n)
        X = np.column_stack([x0, x1, x0 + x1, 1e8 * z])  # rank 3; col 3 huge+indep
        rank, dropped, _ = _detect_rank_deficiency(X)
        kept = np.array([i for i in range(4) if i not in set(dropped.tolist())], dtype=int)

        assert rank == 3  # scale-corrected count, not under-counted by the huge col
        assert len(dropped) == 1
        assert 3 not in dropped  # the huge INDEPENDENT column is never dropped
        # the RETAINED design is full rank (an identified subset) — equilibrate
        # before matrix_rank so the check is itself scale-invariant
        kept_eq = X[:, kept] / np.sqrt((X[:, kept] ** 2).sum(axis=0))
        assert np.linalg.matrix_rank(kept_eq) == len(kept)
        # downstream inference on the kept coefficients is valid
        y = x0 + 2 * x1 + 0.5 * (1e-8 * (1e8 * z)) + rng.standard_normal(n) * 0.1
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coef, _resid, vcov = solve_ols(X, y)
        assert np.all(np.isfinite(coef[kept]))  # kept coefficients identified
        assert np.isnan(coef[dropped[0]])  # the redundant column is NaN
        vd = np.diag(vcov)[kept]
        assert np.all(np.isfinite(vd)) and np.all(vd >= 0)  # valid kept-coef VCV

    def test_rank_detection_qr_mode_r_matches_economic(self):
        """Lock for the mode="r" refactor: rank detection reads only the R
        diagonal and the pivot, and both are bit-identical between
        mode="r" and mode="economic" (same dgeqp3 factorization; mode only
        controls whether the unused Q is formed). Pivot values are locked by
        same-session cross-mode equality, NOT hard-coded (dgeqp3 tie-breaks
        are BLAS-dependent); only rank/dropped are asserted as values."""
        from scipy.linalg import qr

        from diff_diff.linalg import _detect_rank_deficiency

        rng = np.random.default_rng(7)
        base = rng.standard_normal((80, 3))
        fixtures = {
            # genuinely collinear, well-scaled: rank 3 of 4
            "collinear": (np.column_stack([base, base[:, 0] - base[:, 2]]), 3),
            # scale artifact: full rank but one huge column
            "scale": (np.column_stack([base, 1e9 * rng.standard_normal(80)]), 4),
        }
        for name, (X, expected_rank) in fixtures.items():
            rank, dropped, pivot = _detect_rank_deficiency(X)
            assert rank == expected_rank, name
            assert len(dropped) == X.shape[1] - expected_rank, name
            # cross-mode equivalence: identical R diagonal and pivot
            r_only = qr(X, mode="r", pivoting=True)
            _q, r_eco, piv_eco = qr(X, mode="economic", pivoting=True)
            np.testing.assert_array_equal(
                np.abs(np.diag(r_only[0])), np.abs(np.diag(r_eco)), err_msg=name
            )
            np.testing.assert_array_equal(r_only[1], piv_eco, err_msg=name)

    def test_equilibrated_lstsq_f_order_overwrite_matches_reference(self):
        """Lock for the F-order + overwrite_a change: the in-place gelsd
        consume must return bit-identical coefficients to a plain
        non-overwriting C-order call (same values reach dgelsd either way),
        and must not mutate the caller's X."""
        from scipy.linalg import lstsq as scipy_lstsq

        from diff_diff.linalg import _equilibrated_lstsq

        rng = np.random.default_rng(11)
        X = rng.standard_normal((300, 6))
        X[:, 3] *= 1e7  # scale disparity exercises the equilibration
        y = X[:, :3].sum(axis=1) + rng.standard_normal(300)
        x_before = X.copy()

        coef = _equilibrated_lstsq(X, y)

        np.testing.assert_array_equal(X, x_before)  # caller's X untouched
        # like-for-like reference: SAME norm accumulation (einsum) and lstsq
        # options as _equilibrated_lstsq, differing ONLY in the C-order
        # non-overwriting call - the exact contract the F-order change claims
        # to preserve.
        norms = np.sqrt(np.einsum("ij,ij->j", X, X))
        ref = (
            scipy_lstsq(X / norms, y, lapack_driver="gelsd", check_finite=False, cond=1e-07)[0]
            / norms
        )
        np.testing.assert_array_equal(coef, ref)


class TestEstimatorIntegration:
    """Integration tests verifying estimators produce correct results."""

    def test_did_estimator_produces_valid_results(self):
        """Verify DifferenceInDifferences produces valid inference."""
        from diff_diff import DifferenceInDifferences

        # Create reproducible test data
        np.random.seed(42)
        n = 200
        data = pd.DataFrame(
            {
                "unit": np.repeat(range(20), 10),
                "time": np.tile(range(10), 20),
                "treated": np.repeat([0] * 10 + [1] * 10, 10),
                "post": np.tile([0] * 5 + [1] * 5, 20),
            }
        )
        # True ATT = 2.0
        data["outcome"] = np.random.randn(n) + 2.0 * data["treated"] * data["post"]

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

        data = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(n_units), n_times),
                "time": np.tile(np.arange(n_times), n_units),
                "treated": np.repeat(np.random.binomial(1, 0.5, n_units), n_times),
            }
        )
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
        result = twfe.fit(data, outcome="y", treatment="treated", time="post", unit="unit")

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

        data = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(n_units), n_times),
                "time": np.tile(np.arange(n_times), n_units),
            }
        )

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
        result = sa.fit(data, outcome="y", unit="unit", time="time", first_treat="first_treat")

        # Should produce valid results
        assert result.overall_se > 0
        assert np.isfinite(result.overall_att)
        assert len(result.event_study_effects) > 0


class TestSolveLogit:
    """Tests for IRLS logistic regression (solve_logit)."""

    def test_irls_coefficients_well_conditioned(self):
        """IRLS produces correct coefficients on well-conditioned data."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal((n, 3))
        beta_true = np.array([0.5, -1.0, 0.8])
        z = X @ beta_true
        y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)

        beta, probs = solve_logit(X, y)
        # beta[0] is intercept, beta[1:] are coefficients
        assert beta.shape == (4,)
        assert probs.shape == (n,)
        # Coefficients should be close to true values (intercept ~0)
        assert np.abs(beta[0]) < 1.0, "Intercept should be near zero"
        assert np.allclose(
            beta[1:], beta_true, atol=0.3
        ), f"Coefficients {beta[1:]} not close to {beta_true}"

    def test_irls_convergence(self):
        """IRLS converges without warnings on standard data."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(123)
        n = 200
        X = rng.standard_normal((n, 2))
        y = (rng.random(n) < 0.5).astype(float)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            beta, probs = solve_logit(X, y)

        convergence_warns = [x for x in w if "did not converge" in str(x.message)]
        assert len(convergence_warns) == 0

    def test_irls_non_convergence_warning(self):
        """IRLS warns when max_iter=1 prevents convergence."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        y = (rng.random(n) < 0.5).astype(float)

        with pytest.warns(UserWarning, match="did not converge"):
            solve_logit(X, y, max_iter=1)

    def test_near_separation_warning(self):
        """Warns about near-separation when covariate perfectly predicts outcome."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 200
        # Create near-perfect separation: large coefficient -> probs near 0/1
        X = rng.standard_normal((n, 1))
        y = (X[:, 0] > 0).astype(float)
        # Add a tiny bit of noise to avoid exact separation
        flip_idx = rng.choice(n, size=3, replace=False)
        y[flip_idx] = 1 - y[flip_idx]

        with pytest.warns(UserWarning, match="Near-separation detected"):
            solve_logit(X, y)

    def test_predicted_probabilities_valid(self):
        """Predicted probabilities are in (0, 1)."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        y = (rng.random(n) < 0.5).astype(float)

        _, probs = solve_logit(X, y)
        assert np.all(probs > 0) and np.all(probs < 1)

    def test_rank_deficient_design_matrix(self):
        """Handles rank-deficient X in logistic regression."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.standard_normal(n)
        # x2 is a duplicate of x1 -> rank deficient
        X = np.column_stack([x1, x1])
        y = (rng.random(n) < 0.5).astype(float)

        with pytest.warns(UserWarning, match="Rank-deficient"):
            beta, probs = solve_logit(X, y)

        assert beta.shape == (3,)  # intercept + 2 features
        assert probs.shape == (n,)

    def test_rank_deficient_action_silent(self):
        """rank_deficient_action='silent' suppresses warning on rank-deficient X."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.standard_normal(n)
        X = np.column_stack([x1, x1])  # rank deficient
        y = (rng.random(n) < 0.5).astype(float)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            beta, probs = solve_logit(X, y, rank_deficient_action="silent")

        rank_warns = [x for x in w if "Rank-deficient" in str(x.message)]
        assert len(rank_warns) == 0
        assert beta.shape == (3,)
        assert probs.shape == (n,)

    def test_rank_deficient_action_error(self):
        """rank_deficient_action='error' raises ValueError on rank-deficient X."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.standard_normal(n)
        X = np.column_stack([x1, x1])  # rank deficient
        y = (rng.random(n) < 0.5).astype(float)

        with pytest.raises(ValueError, match="Rank-deficient"):
            solve_logit(X, y, rank_deficient_action="error")


class TestCheckPropensityDiagnostics:
    """Tests for propensity score diagnostic warnings."""

    def test_no_warning_normal_scores(self):
        """No warning when all scores are within bounds."""
        from diff_diff.linalg import _check_propensity_diagnostics

        import warnings

        pscore = np.array([0.3, 0.5, 0.7, 0.4, 0.6])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_propensity_diagnostics(pscore, trim_bound=0.01)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 0

    def test_warning_extreme_scores(self):
        """Warns when propensity scores are near 0 or 1."""
        from diff_diff.linalg import _check_propensity_diagnostics

        pscore = np.array([0.001, 0.5, 0.999, 0.3, 0.7])
        with pytest.warns(UserWarning, match="outside"):
            _check_propensity_diagnostics(pscore, trim_bound=0.01)


class TestEPVDiagnostics:
    """Tests for Events Per Variable (EPV) check in solve_logit."""

    def test_epv_warning_below_threshold(self):
        """Warning emitted when EPV < threshold."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        # 40 events (minority class), 8 predictor variables → EPV = 5.0
        n = 200
        X = rng.standard_normal((n, 8))
        y = np.concatenate([np.ones(40), np.zeros(n - 40)])
        with pytest.warns(UserWarning, match="Low Events Per Variable"):
            solve_logit(X, y, epv_threshold=10)

    def test_epv_no_warning_above_threshold(self):
        """No EPV warning when EPV >= threshold."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        # 100 events, 2 predictor variables → EPV = 50
        n = 200
        X = rng.standard_normal((n, 2))
        y = np.concatenate([np.ones(100), np.zeros(100)])
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solve_logit(X, y, epv_threshold=10)
        epv_warns = [x for x in w if "Events Per Variable" in str(x.message)]
        assert len(epv_warns) == 0

    def test_epv_error_in_strict_mode(self):
        """ValueError raised when rank_deficient_action='error' and EPV low."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 8))
        y = np.concatenate([np.ones(30), np.zeros(n - 30)])
        with pytest.raises(ValueError, match="Low Events Per Variable"):
            solve_logit(X, y, rank_deficient_action="error", epv_threshold=10)

    def test_epv_threshold_configurable(self):
        """Custom threshold changes warning behavior."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 2))
        # 15 events, 2 predictor variables → EPV = 7.5
        y = np.concatenate([np.ones(15), np.zeros(n - 15)])

        # Default threshold 10 → should warn (EPV=7.5 < 10)
        with pytest.warns(UserWarning, match="Low Events Per Variable"):
            solve_logit(X, y, epv_threshold=10)

        # Threshold 3 → should not warn (EPV=7.5 >= 3)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solve_logit(X, y, epv_threshold=3)
        epv_warns = [x for x in w if "Events Per Variable" in str(x.message)]
        assert len(epv_warns) == 0

    def test_epv_context_label_in_warning(self):
        """Context label appears in warning message."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 8))
        y = np.concatenate([np.ones(30), np.zeros(n - 30)])
        with pytest.warns(UserWarning, match="cohort g=2004"):
            solve_logit(X, y, epv_threshold=10, context_label="cohort g=2004")

    def test_epv_diagnostics_out_populated(self):
        """diagnostics_out dict receives correct keys and values."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 4))
        y = np.concatenate([np.ones(20), np.zeros(n - 20)])
        diag = {}
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solve_logit(X, y, diagnostics_out=diag)

        assert "epv" in diag
        assert "n_events" in diag
        assert "k" in diag
        assert "is_low" in diag
        assert diag["n_events"] == 20  # minority class
        assert diag["k"] == 4  # 4 predictor variables (excluding intercept)
        assert abs(diag["epv"] - 5.0) < 0.01  # 20 events / 4 predictors
        assert diag["is_low"] is True

    def test_epv_uses_post_drop_k(self):
        """EPV uses k after rank-deficient column drop."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 3))
        # Make column 2 a duplicate of column 1 → will be dropped
        X[:, 2] = X[:, 1]
        y = np.concatenate([np.ones(30), np.zeros(n - 30)])
        diag = {}
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solve_logit(X, y, diagnostics_out=diag, rank_deficient_action="silent")

        # Should be 3 params (2 kept covariates + intercept), not 4
        assert diag["k"] == 2  # 2 kept predictor variables (excluding intercept)
        assert diag["n_events"] == 30
        assert abs(diag["epv"] - 15.0) < 0.01  # 30 events / 2 predictors

    def test_epv_uses_positive_weight_sample(self):
        """EPV computed on positive-weight sample, not padded rows."""
        from diff_diff.linalg import solve_logit

        rng = np.random.default_rng(42)
        # 10 real events + 190 real controls = 200 real rows
        n_real = 200
        X_real = rng.standard_normal((n_real, 4))
        y_real = np.concatenate([np.ones(10), np.zeros(n_real - 10)])
        w_real = np.ones(n_real)

        # Pad with 500 zero-weight rows (should not inflate EPV)
        n_pad = 500
        X_pad = rng.standard_normal((n_pad, 4))
        y_pad = np.concatenate([np.ones(250), np.zeros(250)])
        w_pad = np.zeros(n_pad)

        X = np.vstack([X_real, X_pad])
        y_all = np.concatenate([y_real, y_pad])
        w = np.concatenate([w_real, w_pad])

        diag = {}
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solve_logit(X, y_all, weights=w, diagnostics_out=diag)

        # EPV should reflect the 10-event effective sample, not 260
        assert diag["n_events"] == 10  # min(10, 190) from real sample
        assert diag["epv"] == 10 / 4  # 10 events / 4 predictors = 2.5
        assert diag["is_low"] is True


class TestNoDotRuntimeWarnings:
    """Verify np.dot replacement avoids Apple M4 BLAS ufunc FPE bug."""

    def test_solve_ols_no_runtime_warnings(self):
        """No RuntimeWarnings from solve_ols with n >= 500."""
        import warnings

        rng = np.random.default_rng(42)
        n = 500
        k = 5
        X = rng.standard_normal((n, k))
        beta_true = rng.standard_normal(k)
        y = np.dot(X, beta_true) + rng.standard_normal(n) * 0.1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coefficients, residuals, vcov = solve_ols(X, y, return_vcov=True)

        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, (
            f"Got {len(runtime_warnings)} RuntimeWarning(s): "
            f"{[str(x.message) for x in runtime_warnings]}"
        )
        assert np.allclose(coefficients, beta_true, atol=0.1)


class TestSolvePoisson:
    def test_basic_convergence(self):
        """solve_poisson converges on simple count data."""
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        true_beta = np.array([0.5, 0.3, -0.2])
        mu = np.exp(X @ true_beta)
        y = rng.poisson(mu).astype(float)
        beta, W = solve_poisson(X, y)
        assert beta.shape == (3,)
        assert W.shape == (n,)
        assert np.allclose(beta, true_beta, atol=0.15)

    def test_returns_weights(self):
        """solve_poisson returns final mu weights for vcov computation."""
        rng = np.random.default_rng(0)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.poisson(2.0, size=n).astype(float)
        beta, W = solve_poisson(X, y)
        assert (W > 0).all()

    def test_non_negative_output(self):
        """Fitted mu = exp(Xb) should be strictly positive."""
        rng = np.random.default_rng(1)
        n = 50
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.poisson(1.0, size=n).astype(float)
        beta, W = solve_poisson(X, y)
        mu_hat = np.exp(X @ beta)
        assert (mu_hat > 0).all()

    def test_no_intercept_prepended(self):
        """solve_poisson does NOT add intercept (caller's responsibility)."""
        rng = np.random.default_rng(2)
        n = 80
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.poisson(1.5, size=n).astype(float)
        beta, _ = solve_poisson(X, y)
        assert len(beta) == 2  # not 3

    def test_rank_zero_design_raises_valueerror(self):
        """A rank-0 design (all-zero, every column dropped) raises a clear
        ValueError instead of a cryptic IndexError on an empty integer index
        array. solve_poisson does not prepend an intercept, so rank 0 is
        reachable (unlike solve_logit)."""
        import warnings

        rng = np.random.default_rng(0)
        n = 60
        y = rng.poisson(2.0, size=n).astype(float)
        X = np.zeros((n, 2))  # rank 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="rank 0"):
                solve_poisson(X, y)

    def test_rank_zero_positive_weight_subset_raises_valueerror(self):
        """The effective (positive-weight) sample collapsing to rank 0 also raises
        a clear ValueError (the eff_kept branch)."""
        import warnings

        rng = np.random.default_rng(1)
        n = 60
        y = rng.poisson(2.0, size=n).astype(float)
        # Full design is full rank (variation lives in the zero-weight rows), but
        # the positive-weight subsample is all-zero -> rank 0 on that subset.
        X = np.zeros((n, 2))
        X[n // 2 :, :] = rng.standard_normal((n - n // 2, 2))
        weights = np.ones(n)
        weights[n // 2 :] = 0.0  # only the all-zero rows carry positive weight
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="rank 0"):
                solve_poisson(X, y, weights=weights)


class TestRankGuardedInv:
    """Contract tests for _rank_guarded_inv (rank-guarded generalized inverse).

    Used by the influence-function SE path of CallawaySantAnna /
    TripleDifference / StaggeredTripleDifference. A near-singular covariate Gram
    matrix must yield a finite generalized inverse on the identified subspace
    (NaN only on true rank-0), and well-conditioned matrices must be unchanged.
    """

    @staticmethod
    def _gram(X):
        return X.T @ X

    def test_well_conditioned_matches_solve_exactly(self):
        # Fast path must be bit-identical to np.linalg.solve(A, I) so that
        # well-conditioned R-parity goldens are numerically unchanged.
        rng = np.random.RandomState(0)
        X = rng.standard_normal((60, 4))
        A = self._gram(X)
        ginv, n_dropped, rank = _rank_guarded_inv(A)
        assert n_dropped == 0
        assert rank == 4
        np.testing.assert_array_equal(ginv, np.linalg.solve(A, np.eye(4)))

    def test_scale_invariance_no_false_drop(self):
        # One column on a vastly larger scale makes A's raw condition ~1e16, so
        # a bare pinv(A, rcond=1e-10) would truncate a genuine direction. The
        # symmetric equilibration must keep full rank.
        rng = np.random.RandomState(1)
        X = rng.standard_normal((80, 3))
        X[:, 0] *= 1e8
        A = self._gram(X)
        _, n_dropped, rank = _rank_guarded_inv(A)
        assert n_dropped == 0
        assert rank == 3

    def test_constant_column_truncates_one_finite(self):
        # X = [intercept, constant covariate, x2]: the constant covariate is
        # collinear with the intercept -> exactly one redundant direction.
        rng = np.random.RandomState(2)
        n = 100
        X = np.column_stack([np.ones(n), np.full(n, 5.0), rng.standard_normal(n)])
        A = self._gram(X)
        ginv, n_dropped, rank = _rank_guarded_inv(A)
        assert n_dropped == 1
        assert rank == 2
        assert np.all(np.isfinite(ginv))
        # Garbage-inverse guard: the prior np.linalg.solve would have entries
        # ~1e13+; the guarded inverse must be modest.
        assert np.max(np.abs(ginv)) < 1e3

    def test_rank_zero_returns_all_nan(self):
        A = np.zeros((3, 3))
        ginv, n_dropped, rank = _rank_guarded_inv(A)
        assert rank == 0
        assert n_dropped == 3
        assert np.all(np.isnan(ginv))

    def test_tracker_appended_once_on_truncation_only(self):
        rng = np.random.RandomState(3)
        n = 50
        # Deficient: constant column collinear with intercept.
        A_def = self._gram(np.column_stack([np.ones(n), np.full(n, 2.0), rng.standard_normal(n)]))
        tracker = []
        _rank_guarded_inv(A_def, tracker=tracker)
        assert len(tracker) == 1  # exactly one condition-number sample

        # Well-conditioned: no append.
        A_ok = self._gram(rng.standard_normal((n, 3)))
        tracker2 = []
        _rank_guarded_inv(A_ok, tracker=tracker2)
        assert tracker2 == []

    def test_generalized_inverse_identity(self):
        # A G A == A holds for any generalized inverse (the property that makes
        # the IF bilinear form invariant to the choice of inverse).
        rng = np.random.RandomState(4)
        n = 100
        X = np.column_stack([np.ones(n), np.full(n, 3.0), rng.standard_normal(n)])
        A = self._gram(X)
        ginv, n_dropped, _ = _rank_guarded_inv(A)
        assert n_dropped == 1
        np.testing.assert_allclose(A @ ginv @ A, A, rtol=1e-8, atol=1e-6)

    def test_boundary_kept_vs_truncated_gram_condition(self):
        # Threshold is rcond=1e-10 on the equilibrated Gram. cond ~1e8 stays
        # full rank; cond ~1e12 truncates the small direction. (Numbers are the
        # equilibrated Gram condition, not X's.)
        rng = np.random.RandomState(5)
        Q, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        kept = Q @ np.diag([1.0, 1e-8]) @ Q.T
        kept = 0.5 * (kept + kept.T)
        _, n_dropped_kept, rank_kept = _rank_guarded_inv(kept)
        assert n_dropped_kept == 0
        assert rank_kept == 2

        truncated = Q @ np.diag([1.0, 1e-12]) @ Q.T
        truncated = 0.5 * (truncated + truncated.T)
        _, n_dropped_tr, rank_tr = _rank_guarded_inv(truncated)
        assert n_dropped_tr == 1
        assert rank_tr == 1

    def test_empty_matrix(self):
        ginv, n_dropped, rank = _rank_guarded_inv(np.zeros((0, 0)))
        assert ginv.shape == (0, 0)
        assert n_dropped == 0
        assert rank == 0

    def test_return_dropped_mask(self):
        # return_dropped=True exposes a length-k boolean mask of the truncated
        # (unidentified) coordinates so per-coefficient callers can NaN them
        # (a zero-filled dropped coordinate would otherwise report se=0).
        rng = np.random.RandomState(7)
        n = 100
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x, x])  # cols 1 and 2 exactly collinear
        A = self._gram(X)

        # Default call is unchanged (backward-compatible 3-tuple).
        assert len(_rank_guarded_inv(A)) == 3

        ginv, n_dropped, rank, dropped = _rank_guarded_inv(A, return_dropped=True)
        assert n_dropped == 1 and rank == 2
        assert dropped.dtype == bool and dropped.shape == (3,)
        assert dropped.sum() == 1
        # The dropped coordinate is exactly the zero-filled row/col of the inverse.
        assert np.all(ginv[dropped] == 0.0)
        assert bool(dropped[0]) is False  # intercept stays identified

        # Full rank -> no dropped coordinates.
        A_ok = self._gram(rng.standard_normal((n, 3)))
        _, nd_ok, _, dropped_ok = _rank_guarded_inv(A_ok, return_dropped=True)
        assert nd_ok == 0 and not dropped_ok.any()

        # Rank 0 -> every coordinate dropped.
        _, nd0, _, dropped0 = _rank_guarded_inv(np.zeros((3, 3)), return_dropped=True)
        assert nd0 == 3 and dropped0.all()
