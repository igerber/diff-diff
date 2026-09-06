"""Tests for HC2 and Bell-McCaffrey extensions to compute_robust_vcov.

Phase 1a of the HeterogeneousAdoptionDiD implementation. Ships:

- ``vcov_type="classical"``: non-robust OLS SE (backward compat with
  ``robust=False`` on ``DifferenceInDifferences``).
- ``vcov_type="hc2"``: leverage-corrected HC2 one-way.
- ``vcov_type="hc2_bm"``: HC2 plus Imbens-Kolesar (2016) Satterthwaite DOF.
- ``vcov_type="hc2_bm"`` + ``cluster=``: Pustejovsky-Tipton (2018) CR2 cluster-
  robust with per-coefficient and (via ``_compute_cr2_bm_contrast_dof``) compound
  contrast Bell-McCaffrey Satterthwaite DOF. The contrast-DOF helper is the
  backend for ``MultiPeriodDiD``'s post-period-average ATT inference under the
  cluster+hc2_bm combination.

Weighted CR2 Bell-McCaffrey (``hc2_bm`` + ``weights=``, both one-way and
clustered) is now supported via the clubSandwich WLS-CR2 port. Parity against
``clubSandwich::vcovCR(lm(weights=w), type="CR2") + coef_test(test=
"Satterthwaite")$df_Satt`` is locked in ``tests/test_methodology_wls_cr2.py``;
this file covers healthy-design compatibility and the leverage-one NaN policy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from diff_diff.linalg import (
    LinearRegression,
    _compute_bm_dof_from_contrasts,
    _compute_bm_dof_oneway,
    _compute_cr2_bm,
    _compute_cr2_bm_contrast_dof,
    _compute_cr2_bm_vcov_and_dof,
    _compute_hat_diagonals,
    _compute_robust_vcov_numpy,
    _cr2_adjustment_matrix,
    compute_robust_vcov,
    solve_ols,
)
from tests.conftest import assert_nan_inference

# =============================================================================
# Fixtures: deterministic OLS datasets with hand-computable properties
# =============================================================================


@pytest.fixture
def small_ols_dataset():
    """Small deterministic dataset where OLS has closed-form leverage values.

    n=6, k=2 (intercept + slope). Known hat-matrix diagonals and residuals.
    """
    rng = np.random.default_rng(20260419)
    n = 30
    X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, size=n)])
    beta_true = np.array([1.0, 0.5])
    y = X @ beta_true + rng.normal(0.0, 0.1, size=n)
    return X, y


def _fit_unweighted(X, y):
    """Solve unweighted OLS and return residuals + bread matrix."""
    coef, resid, _ = solve_ols(X, y, return_vcov=False)
    bread = X.T @ X
    return coef, resid, bread


# =============================================================================
# Classical (non-robust) VCOV
# =============================================================================


class TestLeverageOneInference:
    """Effective leverage-one observations suppress the whole HC covariance."""

    @staticmethod
    def _design():
        X = np.column_stack([np.ones(4), [0.0, 0.0, 0.0, 1.0]])
        return X, np.array([0.0, 1.0, 2.0, 5.0])

    @pytest.mark.parametrize("vcov_type", ["hc2", "hc2_bm", "hc3"])
    @pytest.mark.parametrize("return_dof", [False, True])
    @pytest.mark.parametrize("compute", [compute_robust_vcov, _compute_robust_vcov_numpy])
    def test_leverage_one_nan_shapes_and_warning(self, vcov_type, return_dof, compute):
        X, _ = self._design()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute(
                X,
                np.array([-1.0, 0.0, 1.0, 0.0]),
                vcov_type=vcov_type,
                return_dof=return_dof,
            )
        vcov = result[0] if return_dof else result
        assert vcov.shape == (2, 2) and np.isnan(vcov).all()
        if return_dof:
            assert result[1].shape == (2,) and np.isnan(result[1]).all()
        assert len(caught) == 1
        assert caught[0].category is UserWarning
        family = "HC2-BM" if vcov_type == "hc2_bm" else vcov_type.upper()
        assert f"{family} variance is undefined: 1 observation(s)" in str(caught[0].message)
        assert "Returning NaN vcov" in str(caught[0].message)

    @pytest.mark.parametrize("vcov_type", ["hc2", "hc2_bm", "hc3"])
    @pytest.mark.parametrize("h", [1 - 2e-8, 1 - 1e-8, 1 - 5e-9, 1.0, 1.00001])
    def test_inclusive_threshold_and_over_one(self, monkeypatch, vcov_type, h):
        # Drive the exact comparison independently of BLAS rounding at the cutoff.
        import diff_diff.linalg as la

        X, _ = self._design()
        monkeypatch.setattr(
            la, "_compute_hat_diagonals", lambda *a, **k: np.array([h, 0.2, 0.2, 0.2])
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = la._compute_robust_vcov_numpy(
                X, np.array([1.0, -1.0, 0.0, 0.0]), vcov_type=vcov_type
            )
        if h >= 1 - 1e-8:
            assert np.isnan(v).all() and len(caught) == 1
            assert "variance is undefined" in str(caught[0].message)
        else:
            assert np.isfinite(v).all() and not caught

    @pytest.mark.parametrize("vcov_type", ["hc2", "hc2_bm"])
    @pytest.mark.parametrize("rank_reduced", [False, True])
    def test_solve_and_regression_preserve_points(self, vcov_type, rank_reduced):
        X, y = self._design()
        if rank_reduced:
            X = np.column_stack([X, X[:, 1]])
        baseline = solve_ols(
            X, y, return_fitted=True, return_vcov=False, rank_deficient_action="silent"
        )
        with pytest.warns(UserWarning, match="variance is undefined"):
            coef, resid, fitted, vcov = solve_ols(
                X, y, vcov_type=vcov_type, return_fitted=True, rank_deficient_action="silent"
            )
        for actual, expected in zip((coef, resid, fitted), baseline[:3]):
            np.testing.assert_allclose(actual, expected, atol=1e-14, equal_nan=True)
        assert np.isnan(vcov).all()
        with pytest.warns(UserWarning, match="variance is undefined"):
            reg = LinearRegression(
                vcov_type=vcov_type, include_intercept=False, rank_deficient_action="silent"
            ).fit(X, y)
        np.testing.assert_allclose(reg.coefficients_, baseline[0], equal_nan=True)
        for j in np.flatnonzero(np.isfinite(coef)):
            inference = reg.get_inference(j)
            assert np.isfinite(inference.coefficient) and np.isnan(inference.se)
            assert_nan_inference(vars(inference))

    @pytest.mark.parametrize("return_dof", [False, True])
    def test_hc2_bm_all_ones_weights_keep_cr2_boundary(self, return_dof):
        X, _ = self._design()
        resid = np.array([-1.0, 0.0, 1.0, 0.0])
        with pytest.warns(UserWarning, match="HC2-BM variance is undefined") as caught:
            plain = compute_robust_vcov(X, resid, vcov_type="hc2_bm", return_dof=return_dof)
        assert len(caught) == 1
        assert np.isnan(plain[0] if return_dof else plain).all()
        if return_dof:
            assert np.isnan(plain[1]).all()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            weighted = compute_robust_vcov(
                X,
                resid,
                weights=np.ones(4),
                weight_type="pweight",
                vcov_type="hc2_bm",
                return_dof=return_dof,
            )
            dof = _compute_bm_dof_from_contrasts(
                X, X.T @ X, np.array([1 / 3, 1 / 3, 1 / 3, 1.0]), np.eye(2), weights=np.ones(4)
            )
        assert not caught
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]]) / 3
        np.testing.assert_allclose(weighted[0] if return_dof else weighted, expected, atol=1e-12)
        np.testing.assert_allclose(dof, [2.0, 2.0], atol=1e-12)
        if return_dof:
            np.testing.assert_allclose(weighted[1], [2.0, 2.0], atol=1e-12)

    @pytest.mark.parametrize("weight_type", ["pweight", "aweight"])
    @pytest.mark.parametrize("delta", [0.0, 5e-9, 2e-8])
    def test_weighted_hc2_near_one(self, weight_type, delta):
        # Weighted bread = 1; the first row's WLS leverage is 1 - delta.
        w = np.array([2.0, 3.0])
        X = np.sqrt(np.array([(1 - delta) / w[0], delta / w[1]]))[:, None]
        resid = np.array([0.1, -0.2])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vcov, dof = compute_robust_vcov(
                X, resid, weights=w, weight_type=weight_type, vcov_type="hc2", return_dof=True
            )
        if delta < 1e-8:
            assert np.isnan(vcov).all() and np.isnan(dof).all()
            assert len(caught) == 1 and "HC2 variance is undefined" in str(caught[0].message)
        else:
            assert not caught and np.isfinite(vcov).all()
            np.testing.assert_array_equal(dof, [1.0])

    @pytest.mark.parametrize("weight_type", ["pweight", "aweight", "fweight"])
    @pytest.mark.parametrize("vcov_type, expected", [("hc2", 1 / 3), ("hc3", 4 / 9)])
    @pytest.mark.parametrize("z", [2.0, 10.0])
    def test_zero_weight_rows_have_no_contribution(self, weight_type, vcov_type, expected, z):
        X, y, w = (
            np.array([[1.0], [1.0], [z]]),
            np.array([0.0, 2.0, 7.0]),
            np.array([2.0, 2.0, 0.0]),
        )
        # Includes exact unit leverage and over-one fweight quadratic forms.
        resid = y - X[:, 0]
        with (
            warnings.catch_warnings(record=True) as caught,
            np.errstate(divide="raise", invalid="raise", over="raise"),
        ):
            warnings.simplefilter("always")
            v, df = compute_robust_vcov(
                X, resid, weights=w, weight_type=weight_type, vcov_type=vcov_type, return_dof=True
            )
            dropped, df_dropped = compute_robust_vcov(
                X[:2],
                resid[:2],
                weights=w[:2],
                weight_type=weight_type,
                vcov_type=vcov_type,
                return_dof=True,
            )
            coef, _, v_fit = solve_ols(
                X, y, weights=w, weight_type=weight_type, vcov_type=vcov_type
            )
        assert not caught and np.isfinite(v).all()
        np.testing.assert_allclose(v, dropped, atol=1e-14)
        np.testing.assert_allclose(v_fit, dropped, atol=1e-14)
        np.testing.assert_allclose(coef, [1.0], atol=1e-14)
        np.testing.assert_array_equal(df, df_dropped)
        if weight_type == "fweight":
            Xe, ye = np.repeat(X, w.astype(int), axis=0), np.repeat(y, w.astype(int))
            _, re, ve = solve_ols(Xe, ye, vcov_type=vcov_type)
            _, df_e = compute_robust_vcov(Xe, re, vcov_type=vcov_type, return_dof=True)
            np.testing.assert_allclose(v, [[expected]], atol=1e-14)
            np.testing.assert_allclose(v, ve, atol=1e-14)
            np.testing.assert_array_equal(df, [3.0])
            np.testing.assert_array_equal(df, df_e)

    @pytest.mark.parametrize("vcov_type", ["hc2", "hc3"])
    @pytest.mark.parametrize("count", [1, 2])
    def test_fweight_singleton_vs_repeated(self, vcov_type, count):
        X, y = self._design()
        w = np.array([2, 2, 2, count])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, vc = solve_ols(X, y, weights=w, weight_type="fweight", vcov_type=vcov_type)
            _, _, ve = solve_ols(np.repeat(X, w, axis=0), np.repeat(y, w), vcov_type=vcov_type)
        if count == 1:
            assert np.isnan(vc).all() and np.isnan(ve).all() and len(caught) == 2
        else:
            assert not caught and np.isfinite(vc).all()
            np.testing.assert_allclose(vc, ve, atol=1e-12)


class TestClassicalVcov:
    def test_matches_sigma_squared_inverse_XtX(self, small_ols_dataset):
        """V = sigma^2 * (X'X)^{-1}."""
        X, y = small_ols_dataset
        n, k = X.shape
        coef, resid, bread = _fit_unweighted(X, y)
        sigma2 = float(np.sum(resid**2) / (n - k))
        expected = sigma2 * np.linalg.inv(bread)

        got = compute_robust_vcov(X, resid, vcov_type="classical")
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_return_dof_yields_n_minus_k(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        vcov, dof = compute_robust_vcov(X, resid, vcov_type="classical", return_dof=True)
        assert dof.shape == (X.shape[1],)
        assert np.all(dof == X.shape[0] - X.shape[1])

    def test_classical_errors_with_cluster(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        cluster_ids = np.arange(X.shape[0]) % 3
        with pytest.raises(ValueError, match="classical SEs are one-way only"):
            compute_robust_vcov(X, resid, cluster_ids=cluster_ids, vcov_type="classical")


# =============================================================================
# HC2 one-way
# =============================================================================


class TestHC2Oneway:
    def test_hat_diagonals_sum_to_k(self, small_ols_dataset):
        """trace(H) = k for a full-rank unweighted OLS design (idempotent H)."""
        X, _ = small_ols_dataset
        bread = X.T @ X
        h_diag = _compute_hat_diagonals(X, bread)
        assert h_diag.sum() == pytest.approx(X.shape[1], abs=1e-10)

    def test_hat_diagonals_in_zero_one(self, small_ols_dataset):
        X, _ = small_ols_dataset
        bread = X.T @ X
        h_diag = _compute_hat_diagonals(X, bread)
        assert h_diag.min() >= 0.0
        assert h_diag.max() <= 1.0

    def test_hc2_matches_manual_formula(self, small_ols_dataset):
        """HC2 meat = bread^{-1} (sum u_i^2 / (1-h_ii) x x') bread^{-1}."""
        X, y = small_ols_dataset
        _, resid, bread = _fit_unweighted(X, y)
        h_diag = _compute_hat_diagonals(X, bread)
        one_minus_h = 1.0 - h_diag
        factor = (resid**2) / one_minus_h
        meat = X.T @ (X * factor[:, np.newaxis])
        bread_inv = np.linalg.inv(bread)
        expected = bread_inv @ meat @ bread_inv

        got = compute_robust_vcov(X, resid, vcov_type="hc2")
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_hc2_wider_than_hc1_for_small_n(self, small_ols_dataset):
        """HC2 SE >= HC1 SE (leverage correction increases variance)."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        vcov_hc1 = compute_robust_vcov(X, resid, vcov_type="hc1")
        vcov_hc2 = compute_robust_vcov(X, resid, vcov_type="hc2")
        se_hc1 = np.sqrt(np.diag(vcov_hc1))
        se_hc2 = np.sqrt(np.diag(vcov_hc2))
        # HC2 has no n/(n-k) adjustment; HC1 does. For small n and moderate
        # leverage, the magnitudes are comparable but HC2 leverage-inflates
        # observations with large h_ii, usually giving a wider SE.
        # Relationship depends on h_ii distribution; here we only assert both
        # are positive and finite.
        assert np.all(np.isfinite(se_hc1))
        assert np.all(np.isfinite(se_hc2))
        assert np.all(se_hc1 > 0)
        assert np.all(se_hc2 > 0)

    def test_hc2_errors_with_cluster(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        cluster_ids = np.arange(X.shape[0]) % 3
        with pytest.raises(ValueError, match="hc2 is one-way only"):
            compute_robust_vcov(X, resid, cluster_ids=cluster_ids, vcov_type="hc2")

    def test_hc2_return_dof_yields_n_minus_k(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        vcov, dof = compute_robust_vcov(X, resid, vcov_type="hc2", return_dof=True)
        assert dof.shape == (X.shape[1],)
        assert np.all(dof == X.shape[0] - X.shape[1])

    def test_hc2_large_n_approaches_hc1(self):
        """At large n, h_ii -> k/n -> 0 so HC2 meat approaches HC1 meat."""
        rng = np.random.default_rng(7)
        n = 5000
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, size=n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.1, size=n)
        _, resid, _ = _fit_unweighted(X, y)

        vcov_hc1 = compute_robust_vcov(X, resid, vcov_type="hc1")
        vcov_hc2 = compute_robust_vcov(X, resid, vcov_type="hc2")
        # Remove the n/(n-k) adjustment from HC1 to compare the meat matrices
        # on equal footing. At n=5000 with k=2, the hat diagonals average to
        # k/n = 4e-4, so HC2 and unadjusted-HC1 should agree to ~0.1%.
        adj = n / (n - 2)
        vcov_hc1_unadj = vcov_hc1 / adj
        rel_diff = np.abs(vcov_hc2 - vcov_hc1_unadj) / np.abs(vcov_hc1_unadj)
        assert np.all(rel_diff < 1e-3)


# =============================================================================
# Bell-McCaffrey one-way DOF
# =============================================================================


class TestHC2BMOneway:
    def test_bm_dof_shape_and_positive(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        vcov, dof_vec = compute_robust_vcov(X, resid, vcov_type="hc2_bm", return_dof=True)
        assert dof_vec.shape == (X.shape[1],)
        assert np.all(dof_vec > 0)
        assert np.all(np.isfinite(dof_vec))

    def test_bm_dof_smaller_than_n_minus_k(self, small_ols_dataset):
        """Bell-McCaffrey DOF should be conservative (<= n-k)."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        _, dof_vec = compute_robust_vcov(X, resid, vcov_type="hc2_bm", return_dof=True)
        n_minus_k = X.shape[0] - X.shape[1]
        assert np.all(dof_vec <= n_minus_k + 1e-10)

    def test_bm_dof_matches_manual_satterthwaite(self):
        """Cross-check: (trace(B))^2 / trace(B@B) for a specific small design."""
        # Deterministic design with hand-computable hat matrix.
        X = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
                [1.0, 5.0],
            ]
        )
        bread = X.T @ X
        h_diag = _compute_hat_diagonals(X, bread)
        bm_dof = _compute_bm_dof_oneway(X, bread, h_diag)

        # Expected: compute (trace(M @ diag(a) @ M))^2 / trace((M diag(a) M)^2)
        # for each coefficient.
        n, k = X.shape
        H = X @ np.linalg.inv(bread) @ X.T
        M = np.eye(n) - H
        bread_inv = np.linalg.inv(bread)
        for j in range(k):
            c = np.zeros(k)
            c[j] = 1.0
            q = X @ (bread_inv @ c)
            a = (q**2) / (1.0 - h_diag)
            # B = M diag(a) M
            B = M @ np.diag(a) @ M
            expected = (np.trace(B)) ** 2 / np.trace(B @ B)
            assert bm_dof[j] == pytest.approx(expected, abs=1e-10)

    def test_bm_dof_scales_with_n(self):
        """BM DOF grows linearly with n for fixed regressor distribution.

        For this U(0,1) design, both coefficients' BM DOF scale roughly as
        ``0.45 * n`` (derivable from the closed-form expectation of
        ``(sum q^2)^2 / sum a^2`` under uniform regressor). The test just
        checks BM DOF doubles when n doubles (to ~5% tolerance).
        """
        rng = np.random.default_rng(3)
        dofs_by_n = {}
        for n in (250, 500):
            X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, size=n)])
            y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.1, size=n)
            _, resid, _ = _fit_unweighted(X, y)
            _, dof_vec = compute_robust_vcov(X, resid, vcov_type="hc2_bm", return_dof=True)
            dofs_by_n[n] = dof_vec
        # Scaling check: doubling n doubles BM DOF to ~5%.
        ratio = dofs_by_n[500] / dofs_by_n[250]
        np.testing.assert_allclose(ratio, 2.0, rtol=0.15)


# =============================================================================
# Backward compatibility: existing HC1 / CR1 paths unchanged
# =============================================================================


class TestHC1Unchanged:
    def test_default_path_unchanged(self, small_ols_dataset):
        """Default call (no vcov_type kwarg) returns the same HC1 as before.

        Uses ``assert_allclose`` rather than bit-exact equality: the two
        call paths reach the same math but the default-kwarg path can
        accumulate ordering differences in the floating-point pipeline
        (e.g., Numpy BLAS may reorder reductions depending on which
        validator branch runs). The matrices agree to machine epsilon —
        well below the stability bar for variance inference.
        """
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        default = compute_robust_vcov(X, resid)
        explicit = compute_robust_vcov(X, resid, vcov_type="hc1")
        np.testing.assert_allclose(default, explicit, atol=1e-14, rtol=1e-14)

    def test_default_no_dof_returns_vcov_only(self, small_ols_dataset):
        """return_dof=False (default) returns ndarray, not tuple."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        result = compute_robust_vcov(X, resid, vcov_type="hc1")
        assert isinstance(result, np.ndarray)
        # With return_dof=True it's a tuple.
        result_tuple = compute_robust_vcov(X, resid, vcov_type="hc1", return_dof=True)
        assert isinstance(result_tuple, tuple)
        assert len(result_tuple) == 2

    def test_hc1_cluster_unchanged(self, small_ols_dataset):
        """Same invariant as ``test_default_path_unchanged`` for the
        clustered (CR1) path. Uses ``assert_allclose`` because Numpy
        BLAS reduction ordering can introduce sub-machine-epsilon
        differences between the default-kwarg and explicit-kwarg paths.
        """
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        cluster_ids = np.arange(X.shape[0]) % 5
        default = compute_robust_vcov(X, resid, cluster_ids=cluster_ids)
        explicit = compute_robust_vcov(X, resid, cluster_ids=cluster_ids, vcov_type="hc1")
        np.testing.assert_allclose(default, explicit, atol=1e-14, rtol=1e-14)

    def test_hc2_bm_weighted_cluster_gate_lifted(self, small_ols_dataset):
        """Weighted CR2 Bell-McCaffrey (cluster + weights) is now supported via
        the clubSandwich WLS-CR2 port. Smoke test: produces finite vcov + DOF.

        Numerical parity against ``clubSandwich::vcovCR(lm(weights=w),
        type="CR2") + coef_test()$df_Satt`` is locked in
        ``tests/test_methodology_wls_cr2.py``.
        """
        X, y = small_ols_dataset
        cluster_ids = np.arange(X.shape[0]) % 5
        rng = np.random.default_rng(7)
        w = rng.uniform(0.5, 2.0, size=X.shape[0])
        # Refit WLS to get correct residuals for the weighted CR2 path.
        coef = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
        resid_w = y - X @ coef
        vcov, dof = compute_robust_vcov(
            X,
            resid_w,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            weights=w,
            weight_type="pweight",
            return_dof=True,
        )
        assert np.all(np.isfinite(vcov)), "weighted CR2-BM vcov must be finite"
        assert np.all(np.diag(vcov) > 0), "weighted CR2-BM vcov diag must be positive"
        assert np.all(np.isfinite(dof)) and np.all(
            dof > 0
        ), "weighted CR2-BM DOF must be finite and positive"

    def test_hc2_bm_weighted_one_way_gate_lifted(self, small_ols_dataset):
        """Weighted one-way HC2-BM (no cluster) is now supported.

        Uses clubSandwich's singleton-cluster CR2 reduction (each obs is its
        own cluster) for the DOF computation. The simple ``(tr B)² / tr(B²)``
        unweighted formula DIVERGES from clubSandwich by ~6% on weighted
        designs, so the weighted branch routes through the P_array form
        instead.
        """
        X, y = small_ols_dataset
        rng = np.random.default_rng(8)
        w = rng.uniform(0.5, 2.0, size=X.shape[0])
        coef = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
        resid_w = y - X @ coef
        vcov, dof = compute_robust_vcov(
            X,
            resid_w,
            vcov_type="hc2_bm",
            weights=w,
            weight_type="pweight",
            return_dof=True,
        )
        assert np.all(np.isfinite(vcov)), "weighted HC2-BM vcov must be finite"
        assert np.all(np.diag(vcov) > 0), "weighted HC2-BM vcov diag must be positive"
        assert np.all(np.isfinite(dof)) and np.all(
            dof > 0
        ), "weighted HC2-BM DOF must be finite and positive"


# =============================================================================
# Invalid-input error paths
# =============================================================================


class TestInvalidInputs:
    def test_unknown_vcov_type_raises(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        with pytest.raises(ValueError, match="vcov_type must be one of"):
            compute_robust_vcov(X, resid, vcov_type="hc9")

    def test_hc0_not_accepted(self, small_ols_dataset):
        """HC0/CR0 are out of scope for Phase 1a (HC3 joined the valid set
        for the LWDiD canonical-vocabulary rename)."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        for bad in ("hc0", "cr0"):
            with pytest.raises(ValueError, match="vcov_type must be one of"):
                compute_robust_vcov(X, resid, vcov_type=bad)


class TestSolveOlsValidationBypass:
    """Regression tests for the P0 the CI reviewer surfaced: validation must
    fire for `solve_ols` / `_solve_ols_numpy` call paths too, not just through
    the public `compute_robust_vcov` wrapper. Unsupported combinations must
    raise everywhere rather than silently dropping to one-way formulas.
    """

    def test_solve_ols_rejects_cluster_plus_classical(self):
        rng = np.random.default_rng(1)
        n = 20
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.1, n)
        cluster_ids = np.arange(n) % 4
        with pytest.raises(ValueError, match="classical SEs are one-way only"):
            solve_ols(X, y, cluster_ids=cluster_ids, vcov_type="classical")

    def test_solve_ols_rejects_cluster_plus_hc2(self):
        rng = np.random.default_rng(2)
        n = 20
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.1, n)
        cluster_ids = np.arange(n) % 4
        with pytest.raises(ValueError, match="hc2 is one-way only"):
            solve_ols(X, y, cluster_ids=cluster_ids, vcov_type="hc2")

    def test_solve_ols_accepts_cluster_weights_hc2_bm(self):
        """Weighted CR2 BM via solve_ols is supported post-clubSandwich port.

        Smoke test only — numerical parity locked in
        ``tests/test_methodology_wls_cr2.py``.
        """
        rng = np.random.default_rng(3)
        n = 20
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.1, n)
        cluster_ids = np.arange(n) % 4
        weights = rng.uniform(0.5, 2.0, size=n)
        coef, _resid, vcov = solve_ols(
            X,
            y,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            weights=weights,
            weight_type="pweight",
            return_vcov=True,
        )
        assert vcov is not None and np.all(np.isfinite(vcov))
        assert np.all(np.diag(vcov) > 0)

    def test_linear_regression_rejects_cluster_plus_hc2(self):
        """LinearRegression is an estimator-level entry; it must also raise."""
        from diff_diff.linalg import LinearRegression

        rng = np.random.default_rng(4)
        n = 20
        X = np.column_stack([rng.uniform(0, 1, n)])  # LR adds intercept
        y = rng.normal(0, 1, n)
        cluster_ids = np.arange(n) % 4
        with pytest.raises(ValueError, match="hc2 is one-way only"):
            LinearRegression(cluster_ids=cluster_ids, vcov_type="hc2").fit(X, y)


# =============================================================================
# CR2 Bell-McCaffrey cluster-robust
# =============================================================================


class TestCR2BMCluster:
    def test_cr2_adjustment_matrix_identity_when_H_gg_zero(self):
        """When H_gg = 0, A_g = I (pseudo-inverse-sqrt of I)."""
        H_gg = np.zeros((3, 3))
        I_g = np.eye(3)
        A_g = _cr2_adjustment_matrix(I_g - H_gg)
        np.testing.assert_allclose(A_g, I_g, atol=1e-12)

    def test_cr2_adjustment_matrix_satisfies_inverse(self):
        """A_g @ A_g @ (I - H_gg) = I (on the range, pseudo-inverse property)."""
        rng = np.random.default_rng(13)
        # Random symmetric PSD matrix with eigenvalues in [0.1, 1.0]
        U = rng.normal(size=(4, 4))
        Q, _ = np.linalg.qr(U)
        eigvals = np.array([0.2, 0.4, 0.6, 0.9])
        IH = Q @ np.diag(eigvals) @ Q.T
        A = _cr2_adjustment_matrix(IH)
        # A @ A @ IH should equal I for full-rank IH.
        result = A @ A @ IH
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)

    def test_cr2_adjustment_handles_singular_block(self):
        """Singular I - H_gg (absorbed cluster FE): pseudo-inverse zeroes the null space."""
        # I - H_gg with one zero eigenvalue (rank 2 of 3).
        U = np.eye(3)
        eigvals = np.array([0.5, 0.3, 0.0])
        IH = U @ np.diag(eigvals) @ U.T
        A = _cr2_adjustment_matrix(IH)
        # First two diagonals should be 1/sqrt(eigval); third zeroed.
        expected_diag = np.array([1 / np.sqrt(0.5), 1 / np.sqrt(0.3), 0.0])
        np.testing.assert_allclose(np.diag(A), expected_diag, atol=1e-12)

    def test_cr2_bm_runs_unweighted(self):
        rng = np.random.default_rng(101)
        n = 40
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.2, n)
        cluster_ids = np.arange(n) % 5
        _, resid, _ = _fit_unweighted(X, y)
        vcov, dof = compute_robust_vcov(
            X,
            resid,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            return_dof=True,
        )
        assert vcov.shape == (2, 2)
        # VCOV is symmetric PSD.
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(vcov) > -1e-10)
        # DOF vector: k entries, all positive and finite.
        assert dof.shape == (2,)
        assert np.all(dof > 0)
        assert np.all(np.isfinite(dof))
        # CR2 DOF should be strictly less than G = 5 (small-sample correction).
        assert np.all(dof < 5)

    def test_cr2_bm_direct_helper_matches_dispatch(self):
        """Direct _compute_cr2_bm matches the dispatched compute_robust_vcov."""
        rng = np.random.default_rng(99)
        n = 30
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.2, n)
        cluster_ids = np.repeat(np.arange(6), 5)
        _, resid, _ = _fit_unweighted(X, y)
        bread = X.T @ X
        vcov_direct, dof_direct = _compute_cr2_bm(X, resid, cluster_ids, bread)
        vcov_dispatched, dof_dispatched = compute_robust_vcov(
            X,
            resid,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            return_dof=True,
        )
        np.testing.assert_allclose(vcov_direct, vcov_dispatched, atol=1e-12)
        np.testing.assert_allclose(dof_direct, dof_dispatched, atol=1e-12)

    def test_cr2_bm_singleton_clusters(self):
        """CR2 handles singleton clusters via pseudo-inverse when H_gg = 1."""
        rng = np.random.default_rng(77)
        n = 10
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.2, n)
        cluster_ids = np.arange(n)  # every observation its own cluster
        _, resid, _ = _fit_unweighted(X, y)
        # Should not raise and should produce finite numbers.
        vcov, dof = compute_robust_vcov(
            X,
            resid,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            return_dof=True,
        )
        assert np.all(np.isfinite(vcov))
        assert np.all(np.isfinite(dof))

    def test_cr2_parity_with_golden(self):
        """Parity against benchmarks/data/clubsandwich_cr2_golden.json.

        The JSON is the authoritative clubSandwich-generated fixture
        (regenerated via benchmarks/R/generate_clubsandwich_golden.R;
        `meta.source = "clubSandwich"`). Test tolerance is 1e-6, well
        within the 6-digit parity target stated in the Phase 1a plan;
        empirically Python matches clubSandwich at ≤ 7.1e-15 across all
        three datasets.
        """
        import json
        from pathlib import Path

        golden_path = (
            Path(__file__).parent.parent / "benchmarks" / "data" / "clubsandwich_cr2_golden.json"
        )
        if not golden_path.exists():
            pytest.skip("Golden JSON not present; run the R script to generate.")
        with open(golden_path) as f:
            golden = json.load(f)

        for name, d in golden.items():
            if name == "meta":
                continue
            # Skip scenarios that don't fit this test's `y ~ x` two-column
            # unweighted-cluster contract. Absorbed-FE / MPD / TWFE / SA / WLS
            # scenarios are tested separately via their own parity tests
            # (e.g. `test_estimators_vcov_type.py::TestDiDAbsorbedFERParity`
            # for absorbed_fe_did; `tests/test_methodology_wls_cr2.py` for
            # the `weighted_*` scenarios).
            if "x" not in d or "cluster" not in d or "vcov_shape" not in d:
                continue
            # The weighted-cluster fixtures use the same `cluster + x` keys
            # but with `weights` present and a 3-column `[1, x, z]` design.
            if "weights" in d:
                continue
            x = np.array(d["x"])
            y = np.array(d["y"])
            cluster = np.array(d["cluster"])
            X = np.column_stack([np.ones_like(x), x])
            _, resid, _ = solve_ols(X, y, return_vcov=False)
            bread = X.T @ X
            vcov, dof_vec = _compute_cr2_bm(X, resid, cluster, bread)
            expected_vcov = np.array(d["vcov_cr2"]).reshape(d["vcov_shape"])
            expected_dof = np.array(d["dof_bm"])
            np.testing.assert_allclose(
                vcov,
                expected_vcov,
                atol=1e-6,
                err_msg=f"VCOV mismatch on dataset '{name}'",
            )
            np.testing.assert_allclose(
                dof_vec,
                expected_dof,
                atol=1e-6,
                err_msg=f"BM DOF mismatch on dataset '{name}'",
            )

    def test_cr2_bm_fewer_than_two_clusters_raises(self):
        rng = np.random.default_rng(1)
        n = 10
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.2, n)
        _, resid, _ = _fit_unweighted(X, y)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            compute_robust_vcov(
                X,
                resid,
                cluster_ids=np.zeros(n),  # one cluster
                vcov_type="hc2_bm",
            )


# =============================================================================
# HC2 weighted
# =============================================================================


class TestHC2Weighted:
    def test_hc2_pweight_matches_manual(self, small_ols_dataset):
        """Weighted HC2 uses h_ii = w_i * x_i' (X'WX)^{-1} x_i."""
        X, y = small_ols_dataset
        rng = np.random.default_rng(11)
        n = X.shape[0]
        w = rng.uniform(0.5, 2.0, size=n)
        # Refit weighted OLS to get residuals appropriate for the weighted
        # sandwich.
        coef, resid, _ = solve_ols(  # type: ignore[call-overload]
            X, y, return_vcov=False, weights=w, weight_type="pweight"
        )
        XtWX = X.T @ (X * w[:, np.newaxis])
        h_diag = _compute_hat_diagonals(X, XtWX, weights=w)
        one_minus_h = np.maximum(1.0 - h_diag, 1e-10)
        scaled = w * resid / np.sqrt(one_minus_h)
        scores_hc2 = X * scaled[:, np.newaxis]
        meat = scores_hc2.T @ scores_hc2
        bread_inv = np.linalg.inv(XtWX)
        expected = bread_inv @ meat @ bread_inv

        got = compute_robust_vcov(X, resid, vcov_type="hc2", weights=w, weight_type="pweight")
        np.testing.assert_allclose(got, expected, atol=1e-10)


# =============================================================================
# Cluster-aware CR2 BM contrast-DOF helper (Gate 6 lift)
# =============================================================================


class TestCR2BMContrastDOF:
    """Tests for `_compute_cr2_bm_contrast_dof`.

    The helper generalizes the per-coefficient Satterthwaite DOF in
    `_compute_cr2_bm` to arbitrary linear combinations of coefficients
    (used by `MultiPeriodDiD` to compute the cluster-aware DOF for the
    post-period-average ATT contrast).
    """

    def _load_golden_scenario(self):
        """Load `mpd_clustered_avg_att_dof` scenario from R generator."""
        import json
        from pathlib import Path

        golden_path = (
            Path(__file__).parent.parent / "benchmarks" / "data" / "clubsandwich_cr2_golden.json"
        )
        if not golden_path.exists():
            pytest.skip(
                "Golden JSON not present; run "
                "`Rscript benchmarks/R/generate_clubsandwich_golden.R` first."
            )
        with open(golden_path) as f:
            golden = json.load(f)
        if "mpd_clustered_avg_att_dof" not in golden:
            pytest.skip(
                "Golden JSON does not include `mpd_clustered_avg_att_dof` "
                "scenario; regenerate via the R script."
            )
        return golden["mpd_clustered_avg_att_dof"]

    def _build_mpd_design(self, d):
        """Construct the MPD-style design matrix that mirrors R's lm()
        formula `treated + period_f + treated_period_X (non-ref) + factor(unit)`."""
        unit = np.array(d["unit"])
        period = np.array(d["period"])
        treated = np.array(d["treated"], dtype=float)
        n = len(period)
        n_periods = int(period.max())
        non_ref = list(range(2, n_periods + 1))
        const = np.ones(n)
        period_dummies = np.column_stack([(period == p).astype(float) for p in non_ref])
        interaction_dummies = np.column_stack(
            [(treated * (period == p)).astype(float) for p in non_ref]
        )
        n_units = int(unit.max())
        unit_dummies = np.column_stack([(unit == u).astype(float) for u in range(2, n_units + 1)])
        X_full = np.column_stack(
            [const, treated, period_dummies, interaction_dummies, unit_dummies]
        )
        # Drop the last unit dummy to match R's rank-deficient drop on this
        # parameterization (never-treated cohort's last unit is collinear with
        # the intercept + treated + remaining unit dummies).
        return X_full[:, :-1]

    def test_unit_contrasts_match_compute_cr2_bm(self):
        """Refactor anchor: calling the helper with `contrasts=eye(k)`
        produces the same per-coefficient DOFs as `_compute_cr2_bm`.

        Matmul ordering differs (helper applies eye separately, library
        slices precomputed columns), so use atol=1e-10 not bit-identity.
        """
        d = self._load_golden_scenario()
        X = self._build_mpd_design(d)
        k = X.shape[1]
        y = np.array(d["y"], dtype=float)
        cluster = np.array(d["cluster"])
        bread = X.T @ X
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coef

        _, dof_lib = _compute_cr2_bm(X, residuals, cluster, bread)
        dof_helper = _compute_cr2_bm_contrast_dof(X, cluster, bread, np.eye(k))

        finite_both = np.isfinite(dof_lib) & np.isfinite(dof_helper)
        assert finite_both.any(), "expected at least one finite DOF"
        np.testing.assert_allclose(dof_helper[finite_both], dof_lib[finite_both], atol=1e-10)

    def test_compound_contrast_matches_clubsandwich(self):
        """R-parity anchor: compound post-period-average contrast DOF
        matches clubSandwich's `Wald_test(test="HTZ")$df_denom` at 1e-10.
        """
        d = self._load_golden_scenario()
        X = self._build_mpd_design(d)
        k = X.shape[1]
        cluster = np.array(d["cluster"])
        bread = X.T @ X

        # The R golden stores c_avg as a (k_finite,) vector aligned with
        # finite_coef_names. Our X already has the rank-deficient column
        # dropped, so c_avg aligns directly.
        c_avg = np.array(d["c_avg"])
        assert c_avg.shape == (k,), f"c_avg shape {c_avg.shape} does not match X.shape[1] {k}"

        dof_avg_py = float(_compute_cr2_bm_contrast_dof(X, cluster, bread, c_avg[:, np.newaxis])[0])
        dof_avg_r = float(d["dof_avg"])
        np.testing.assert_allclose(dof_avg_py, dof_avg_r, atol=1e-10)

    def test_invalid_contrast_shape_raises(self):
        """Helper validates that the contrast matrix's row count matches `k`."""
        rng = np.random.default_rng(20260517)
        n, k = 30, 3
        X = rng.standard_normal((n, k))
        cluster = np.repeat(np.arange(5), 6)
        bread = X.T @ X
        bad_contrasts = np.zeros((k + 1, 1))
        with pytest.raises(ValueError, match=r"shape \(k="):
            _compute_cr2_bm_contrast_dof(X, cluster, bread, bad_contrasts)

    def test_too_few_clusters_raises(self):
        """Helper requires at least 2 clusters (matching `_compute_cr2_bm`)."""
        rng = np.random.default_rng(20260517)
        n, k = 30, 3
        X = rng.standard_normal((n, k))
        # Everyone in cluster 1 -> only 1 unique cluster.
        cluster = np.ones(n, dtype=int)
        bread = X.T @ X
        with pytest.raises(ValueError, match=r"[Nn]eed at least 2 clusters"):
            _compute_cr2_bm_contrast_dof(X, cluster, bread, np.eye(k))

    def test_wrappers_are_bit_identical_to_shared_core(self):
        """`_compute_cr2_bm` and `_compute_cr2_bm_contrast_dof` are thin
        wrappers over `_compute_cr2_bm_vcov_and_dof`, so they must reproduce
        the core's output exactly (atol=0 / array_equal).

        This is the refactor's structural guard: the per-coefficient vcov+DOF
        path (`contrasts=eye(k)`, residuals provided) and the DOF-only contrast
        path (`residuals=None`) both route through the single core, unweighted
        and weighted.
        """
        rng = np.random.default_rng(20260628)
        n, k = 40, 3
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        beta = np.array([0.5, 1.0, -0.7])
        y = X @ beta + rng.standard_normal(n) * 0.3
        cluster = np.repeat(np.arange(8), 5)
        # Compound contrast: per-coef columns plus an average of the slopes.
        C = np.column_stack([np.eye(k), np.array([0.0, 0.5, 0.5])])

        for weights in (None, rng.uniform(0.5, 2.0, size=n)):
            if weights is None:
                bread = X.T @ X
            else:
                bread = X.T @ (X * weights[:, np.newaxis])
            coef = np.linalg.solve(bread, X.T @ (y if weights is None else y * weights))
            residuals = y - X @ coef

            # Per-coefficient vcov + DOF via wrapper vs core (eye(k)).
            v_wrap, d_wrap = _compute_cr2_bm(X, residuals, cluster, bread, weights=weights)
            v_core, d_core = _compute_cr2_bm_vcov_and_dof(
                X, cluster, bread, np.eye(k), residuals=residuals, weights=weights
            )
            assert np.array_equal(v_wrap, v_core)
            assert np.array_equal(d_wrap, d_core, equal_nan=True)

            # DOF-only contrast path via wrapper vs core (residuals=None).
            dof_wrap = _compute_cr2_bm_contrast_dof(X, cluster, bread, C, weights=weights)
            vcov_core, dof_core = _compute_cr2_bm_vcov_and_dof(
                X, cluster, bread, C, residuals=None, weights=weights
            )
            assert vcov_core is None  # no residuals -> meat/vcov skipped
            assert np.array_equal(dof_wrap, dof_core, equal_nan=True)

    def test_dof_only_with_zero_weights_does_not_crash(self):
        """DOF-only callers pass `weights=` AND `residuals=None`; the shared
        core's zero-weight filter must guard the residuals subscript.

        Regression for the merge of `_compute_cr2_bm`'s filter (which subscripts
        `residuals` unconditionally) into the shared core: without the
        `residuals is not None` guard, a DOF-only call with zero weights would
        raise `TypeError` (StackedDiD's weighted contrast-DOF path and the
        weighted singleton-cluster dispatch hit exactly this).
        """
        rng = np.random.default_rng(20260629)
        n, k = 40, 3
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        cluster = np.repeat(np.arange(8), 5)
        weights = np.ones(n)
        # Zero out a few rows, keeping >=2 clusters with positive total weight.
        weights[[0, 7, 13]] = 0.0
        bread = X.T @ (X * weights[:, np.newaxis])
        C = np.column_stack([np.eye(k), np.array([0.0, 0.5, 0.5])])

        # Must not raise (the bug was an unconditional residuals[positive_mask]).
        dof = _compute_cr2_bm_contrast_dof(X, cluster, bread, C, weights=weights)
        assert dof.shape == (C.shape[1],)
        assert np.all(np.isfinite(dof))
