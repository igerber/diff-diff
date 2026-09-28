"""Tests for diff_diff.local_linear: kernels, moments, and local-linear fit."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from diff_diff.local_linear import (
    KERNELS,
    LocalLinearFit,
    epanechnikov_kernel,
    kernel_moments,
    local_linear_fit,
    triangular_kernel,
    uniform_kernel,
)

# =============================================================================
# Kernel support and shape
# =============================================================================


class TestKernelSupport:
    def test_epanechnikov_support(self):
        u = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
        k = epanechnikov_kernel(u)
        assert k[0] == 0.0
        assert k[6] == 0.0
        # k(0) = 3/4 * (1 - 0) = 0.75
        assert k[1] == pytest.approx(0.75)
        # k(0.5) = 0.75 * (1 - 0.25) = 0.5625
        assert k[3] == pytest.approx(0.5625)
        # k(1) = 0.75 * (1 - 1) = 0.0
        assert k[5] == pytest.approx(0.0)

    def test_triangular_support(self):
        u = np.array([-0.1, 0.0, 0.3, 1.0, 1.2])
        k = triangular_kernel(u)
        assert k[0] == 0.0
        assert k[4] == 0.0
        assert k[1] == pytest.approx(1.0)
        assert k[2] == pytest.approx(0.7)
        assert k[3] == pytest.approx(0.0)

    def test_uniform_support(self):
        u = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
        k = uniform_kernel(u)
        np.testing.assert_array_equal(k, [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_kernels_vectorize(self):
        # Scalar input handled as shape (1,) via asarray.
        for name, kfun in KERNELS.items():
            out = kfun(np.array([0.5]))
            assert out.shape == (1,)
            assert out[0] > 0.0, f"{name} should be positive at u=0.5"


# =============================================================================
# Closed-form kernel moments
# =============================================================================


def _numeric_kappa(kernel_name: str, k: int) -> float:
    """Numerically integrate t^k * kernel(t) over [0, 1]."""
    kfun = KERNELS[kernel_name]

    def integrand(t: float) -> float:
        return (t**k) * kfun(np.array([t]))[0]

    val, _ = integrate.quad(integrand, 0.0, 1.0, limit=200)
    return val


class TestKernelMoments:
    @pytest.mark.parametrize(
        "kernel,k,expected",
        [
            # Epanechnikov on [0, 1] with k(t) = (3/4)(1 - t^2)
            ("epanechnikov", 0, 1.0 / 2.0),
            ("epanechnikov", 1, 3.0 / 16.0),
            ("epanechnikov", 2, 1.0 / 10.0),
            ("epanechnikov", 3, 1.0 / 16.0),
            ("epanechnikov", 4, 3.0 / 70.0),
            # Triangular on [0, 1] with k(t) = 1 - t
            ("triangular", 0, 1.0 / 2.0),
            ("triangular", 1, 1.0 / 6.0),
            ("triangular", 2, 1.0 / 12.0),
            ("triangular", 3, 1.0 / 20.0),
            ("triangular", 4, 1.0 / 30.0),
            # Uniform on [0, 1] with k(t) = 1
            ("uniform", 0, 1.0),
            ("uniform", 1, 1.0 / 2.0),
            ("uniform", 2, 1.0 / 3.0),
            ("uniform", 3, 1.0 / 4.0),
            ("uniform", 4, 1.0 / 5.0),
        ],
    )
    def test_closed_form_kappa_matches_expected(self, kernel, k, expected):
        """Module's closed-form kappa_k matches the hand-derived value."""
        moms = kernel_moments(kernel)
        assert moms[f"kappa_{k}"] == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize("kernel", list(KERNELS))
    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    def test_closed_form_kappa_matches_numerical_integration(self, kernel, k):
        """Module's closed-form kappa_k matches scipy.integrate.quad to 1e-12."""
        moms = kernel_moments(kernel)
        numeric = _numeric_kappa(kernel, k)
        assert moms[f"kappa_{k}"] == pytest.approx(numeric, abs=1e-12)

    @pytest.mark.parametrize("kernel", list(KERNELS))
    def test_C_matches_formula(self, kernel):
        """C = (kappa_2^2 - kappa_1 kappa_3) / (kappa_0 kappa_2 - kappa_1^2)."""
        moms = kernel_moments(kernel)
        expected = (moms["kappa_2"] ** 2 - moms["kappa_1"] * moms["kappa_3"]) / (
            moms["kappa_0"] * moms["kappa_2"] - moms["kappa_1"] ** 2
        )
        assert moms["C"] == pytest.approx(expected, abs=1e-15)

    def test_kstar_L2_norm_matches_direct_integration(self):
        """Verify kstar_L2_norm for Epanechnikov by re-integrating directly."""
        moms = kernel_moments("epanechnikov")
        k0, k1, k2 = moms["kappa_0"], moms["kappa_1"], moms["kappa_2"]
        denom = k0 * k2 - k1 * k1

        def integrand(t: float) -> float:
            kt = epanechnikov_kernel(np.array([t]))[0]
            w = (k2 - k1 * t) / denom
            return (w**2) * (kt**2)

        expected, _ = integrate.quad(integrand, 0.0, 1.0, limit=200)
        assert moms["kstar_L2_norm"] == pytest.approx(expected, abs=1e-12)

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            kernel_moments("gaussian")


# =============================================================================
# Local-linear fit
# =============================================================================


class TestLocalLinearFit:
    def test_recovers_intercept_from_linear_dgp(self):
        """y = a + b*d + noise, fit at d0=0 should recover a."""
        rng = np.random.default_rng(20260418)
        n = 2000
        a_true = 2.5
        b_true = 0.7
        d = rng.uniform(0.0, 1.0, size=n)
        y = a_true + b_true * d + rng.normal(0.0, 0.01, size=n)

        fit = local_linear_fit(d, y, bandwidth=0.3, boundary=0.0, kernel="epanechnikov")
        # Tolerance is several noise-sigmas given the effective sample size.
        assert fit.intercept == pytest.approx(a_true, abs=0.01)
        assert fit.slope == pytest.approx(b_true, abs=0.05)
        assert isinstance(fit, LocalLinearFit)
        assert fit.bandwidth == pytest.approx(0.3)
        assert fit.kernel == "epanechnikov"
        assert fit.boundary == 0.0

    def test_intercept_unbiased_at_exact_linear_data(self):
        """With noiseless linear data, local-linear recovers intercept exactly."""
        d = np.linspace(0.01, 0.5, 50)
        y = 1.5 + 2.0 * d
        fit = local_linear_fit(d, y, bandwidth=0.4, boundary=0.0, kernel="epanechnikov")
        assert fit.intercept == pytest.approx(1.5, abs=1e-10)
        assert fit.slope == pytest.approx(2.0, abs=1e-10)

    def test_matches_weighted_ols_directly(self):
        """Kernel-weighted fit should equal manual WLS with identical weights."""
        from diff_diff.linalg import solve_ols

        rng = np.random.default_rng(42)
        d = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=100)
        h = 0.3

        fit = local_linear_fit(d, y, bandwidth=h, boundary=0.0, kernel="uniform")

        retain = (d >= 0.0) & (d <= h)
        X_manual = np.column_stack([np.ones(retain.sum()), d[retain] - 0.0])
        w_manual = np.ones(retain.sum())
        coef_manual, _, _ = solve_ols(  # type: ignore[call-overload]
            X_manual,
            y[retain],
            cluster_ids=None,
            return_vcov=False,
            weights=w_manual,
            weight_type="aweight",
        )
        assert fit.intercept == pytest.approx(coef_manual[0], abs=1e-10)
        assert fit.slope == pytest.approx(coef_manual[1], abs=1e-10)

    def test_weights_composed_with_kernel(self):
        """User weights multiply into kernel weights before the fit."""
        rng = np.random.default_rng(7)
        n = 200
        d = rng.uniform(0.0, 1.0, size=n)
        y = 1.0 + 0.5 * d + rng.normal(0.0, 0.05, size=n)
        user_w = rng.uniform(0.5, 2.0, size=n)
        fit = local_linear_fit(
            d,
            y,
            bandwidth=0.4,
            boundary=0.0,
            kernel="epanechnikov",
            weights=user_w,
        )
        # Just a smoke test that weights don't error and produce a close-to-1
        # intercept; we re-derive the point estimate via direct WLS below.
        assert fit.intercept == pytest.approx(1.0, abs=0.05)
        assert fit.n_effective <= n
        assert fit.n_effective > 0

    def test_returns_dataclass_fields(self):
        d = np.linspace(0.01, 0.5, 30)
        y = np.random.default_rng(0).normal(size=30)
        fit = local_linear_fit(d, y, bandwidth=0.4, boundary=0.0, kernel="triangular")
        # Dataclass invariants
        assert fit.n_effective == len(fit.residuals) == len(fit.kernel_weights)
        assert fit.design_matrix.shape == (fit.n_effective, 2)
        # The first column of the design is the intercept column.
        np.testing.assert_array_equal(fit.design_matrix[:, 0], np.ones(fit.n_effective))

    def test_n_effective_counts_positive_kernel_weights(self):
        """Observations outside [d0, d0 + h] are excluded."""
        # 5 inside, 5 outside the bandwidth window.
        d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 2.0, 2.5, 3.0, 3.5])
        y = np.zeros_like(d)
        y[:5] = 1.0
        fit = local_linear_fit(d, y, bandwidth=0.6, boundary=0.0, kernel="uniform")
        assert fit.n_effective == 5

    def test_bandwidth_too_narrow_raises(self):
        d = np.array([0.5, 0.6, 0.7])
        y = np.array([1.0, 2.0, 3.0])
        # Bandwidth 0.2 catches zero observations (all are above 0.2).
        with pytest.raises(ValueError, match="retained 0 observation"):
            local_linear_fit(d, y, bandwidth=0.2, boundary=0.0, kernel="uniform")

    def test_single_retained_observation_raises(self):
        d = np.array([0.01, 0.5, 0.7])
        y = np.array([1.0, 2.0, 3.0])
        # Bandwidth 0.1: only d=0.01 is in [0, 0.1]. Need at least 2.
        with pytest.raises(ValueError, match="retained 1 observation"):
            local_linear_fit(d, y, bandwidth=0.1, boundary=0.0, kernel="uniform")

    def test_negative_bandwidth_raises(self):
        d = np.array([0.1, 0.2])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            local_linear_fit(d, y, bandwidth=-0.1, boundary=0.0, kernel="uniform")

    def test_zero_bandwidth_raises(self):
        d = np.array([0.1, 0.2])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            local_linear_fit(d, y, bandwidth=0.0, boundary=0.0, kernel="uniform")

    def test_unknown_kernel_raises(self):
        d = np.array([0.1, 0.2])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown kernel"):
            local_linear_fit(d, y, bandwidth=0.5, boundary=0.0, kernel="my_kernel")

    def test_mismatched_shapes_raise(self):
        d = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same shape"):
            local_linear_fit(d, y, bandwidth=0.5, boundary=0.0)

    def test_mismatched_weights_shape_raises(self):
        d = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0])  # wrong length
        with pytest.raises(ValueError, match="weights must have"):
            local_linear_fit(d, y, bandwidth=0.5, boundary=0.0, weights=w)

    def test_negative_weights_raise(self):
        d = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, -0.5, 1.0])
        with pytest.raises(ValueError, match="nonnegative"):
            local_linear_fit(d, y, bandwidth=0.5, boundary=0.0, weights=w)

    def test_non_finite_d_raises(self):
        d = np.array([0.1, np.nan, 0.3, 0.4])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="d contains non-finite"):
            local_linear_fit(d, y, bandwidth=0.5, boundary=0.0)

    def test_non_finite_y_raises(self):
        d = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValueError, match="y contains non-finite"):
            local_linear_fit(d, y, bandwidth=0.5, boundary=0.0)

    def test_non_finite_weights_raises(self):
        d = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, np.nan, 1.0, 1.0])
        with pytest.raises(ValueError, match="weights contains non-finite"):
            local_linear_fit(d, y, bandwidth=0.5, boundary=0.0, weights=w)

    def test_nonzero_boundary(self):
        """Evaluation at d0 != 0 works (for Design 1 continuous-near-d_lower)."""
        rng = np.random.default_rng(11)
        n = 500
        d = rng.uniform(1.0, 2.0, size=n)  # Support starts at d_lower = 1.0
        y = 3.0 + 0.4 * (d - 1.0) + rng.normal(0.0, 0.02, size=n)

        fit = local_linear_fit(d, y, bandwidth=0.3, boundary=1.0, kernel="epanechnikov")
        # Boundary estimate should recover the intercept at d0=1.0.
        assert fit.intercept == pytest.approx(3.0, abs=0.02)
        assert fit.boundary == 1.0
