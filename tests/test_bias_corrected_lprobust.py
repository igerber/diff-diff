"""End-to-end tests for ``bias_corrected_local_linear`` (Phase 1c).

Wrapper-level parity, input-contract, CI, and parameter-interaction tests.
Port-level parity lives in ``tests/test_nprobust_port.py::TestLprobustSingleEval``
(wrapper-vs-port are orthogonal: the port test exercises ``lprobust`` with
R-supplied bandwidths; this file exercises the public API including the
auto-bandwidth path and every input-contract branch).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diff_diff.local_linear import (
    BandwidthResult,
    BiasCorrectedFit,
    bias_corrected_local_linear,
)

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "data"
    / "nprobust_lprobust_golden.json"
)


@pytest.fixture(scope="module")
def golden():
    """Load R-generated nprobust::lprobust golden values. Skip if absent."""
    if not GOLDEN_PATH.exists():
        pytest.skip(
            "Golden values file not found; "
            "run: Rscript benchmarks/R/generate_nprobust_lprobust_golden.R"
        )
    with GOLDEN_PATH.open() as f:
        return json.load(f)


# =============================================================================
# Shape / return-type tests
# =============================================================================


class TestReturnShape:
    def _smoke_data(self, seed=7):
        rng = np.random.default_rng(seed)
        G = 800
        d = rng.uniform(0.0, 1.0, G)
        y = d + d ** 2 + rng.normal(0, 0.4, G)
        return d, y

    def test_returns_bias_corrected_fit(self):
        d, y = self._smoke_data()
        fit = bias_corrected_local_linear(d, y, boundary=0.0)
        assert isinstance(fit, BiasCorrectedFit)
        assert np.isfinite(fit.estimate_classical)
        assert np.isfinite(fit.estimate_bias_corrected)
        assert fit.se_classical > 0
        assert fit.se_robust > 0
        assert fit.ci_low < fit.ci_high
        assert fit.h > 0
        assert fit.b > 0
        assert fit.n_used > 0
        assert fit.n_total == d.shape[0]
        assert fit.kernel == "epanechnikov"
        assert fit.boundary == 0.0

    def test_auto_bandwidth_populates_diagnostics(self):
        d, y = self._smoke_data()
        fit = bias_corrected_local_linear(d, y, boundary=0.0)
        assert fit.bandwidth_source == "auto"
        assert isinstance(fit.bandwidth_diagnostics, BandwidthResult)
        # Under nprobust rho=1 default, we set b = h (ignoring b_mse).
        assert fit.h == fit.b
        # h_mse still surfaced via diagnostics for inspection.
        assert fit.bandwidth_diagnostics.h_mse == pytest.approx(fit.h, rel=1e-14)

    def test_user_bandwidth_source(self):
        d, y = self._smoke_data()
        fit = bias_corrected_local_linear(d, y, boundary=0.0, h=0.3, b=0.3)
        assert fit.bandwidth_source == "user"
        assert fit.bandwidth_diagnostics is None
        assert fit.h == 0.3
        assert fit.b == 0.3

    def test_user_h_only_uses_rho_1(self):
        d, y = self._smoke_data()
        fit = bias_corrected_local_linear(d, y, boundary=0.0, h=0.3)
        # b defaults to h under rho=1.
        assert fit.h == 0.3
        assert fit.b == 0.3

    def test_b_without_h_raises(self):
        d, y = self._smoke_data()
        with pytest.raises(ValueError, match="b provided without h"):
            bias_corrected_local_linear(d, y, boundary=0.0, b=0.3)


# =============================================================================
# Parity tests against R nprobust::lprobust
# =============================================================================


class TestParity:
    """Tiered tolerances per plan commit criterion #1.

    - tau_cl, se_cl: atol=1e-14 where achievable (same arithmetic path as
      Phase 1b); relaxed to 1e-12 when the Q.q-combined residual path
      differs slightly.
    - tau_bc, se_rb: atol=1e-12.
    - ci_low, ci_high: atol=1e-13 (R's z exported in golden JSON).
    """

    def _dgp(self, golden, name):
        g = golden[name]
        return np.asarray(g["d"], dtype=np.float64), np.asarray(g["y"], dtype=np.float64), g

    def test_dgp1_all_six_outputs(self, golden):
        d, y, g = self._dgp(golden, "dgp1")
        fit = bias_corrected_local_linear(d, y, boundary=0.0, kernel="epanechnikov")
        # DGP 1 hits bit-parity through the Cholesky path for tau_cl / se_cl.
        np.testing.assert_allclose(fit.estimate_classical, g["tau_cl"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.se_classical, g["se_cl"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.estimate_bias_corrected, g["tau_bc"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.se_robust, g["se_rb"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.ci_low, g["ci_low"],
                                   atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(fit.ci_high, g["ci_high"],
                                   atol=1e-13, rtol=1e-13)
        # Bandwidth also should match (Phase 1b bit-parity path).
        np.testing.assert_allclose(fit.h, g["h"], atol=1e-12, rtol=1e-12)

    def test_dgp2_all_six_outputs(self, golden):
        d, y, g = self._dgp(golden, "dgp2")
        fit = bias_corrected_local_linear(d, y, boundary=0.0, kernel="epanechnikov")
        np.testing.assert_allclose(fit.estimate_classical, g["tau_cl"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.estimate_bias_corrected, g["tau_bc"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.se_classical, g["se_cl"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.se_robust, g["se_rb"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.ci_low, g["ci_low"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.ci_high, g["ci_high"],
                                   atol=1e-12, rtol=1e-12)

    def test_dgp3_all_six_outputs(self, golden):
        d, y, g = self._dgp(golden, "dgp3")
        fit = bias_corrected_local_linear(d, y, boundary=0.0, kernel="epanechnikov")
        np.testing.assert_allclose(fit.estimate_classical, g["tau_cl"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.estimate_bias_corrected, g["tau_bc"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.se_classical, g["se_cl"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.se_robust, g["se_rb"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.ci_low, g["ci_low"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.ci_high, g["ci_high"],
                                   atol=1e-12, rtol=1e-12)

    def test_clustered_parity_dgp_4(self, golden):
        """DGP 4 uses manual h=b=0.3 to sidestep an nprobust-internal
        singleton-cluster bug in the mse-dpi pilot fits."""
        g = golden["dgp4"]
        d = np.asarray(g["d"], dtype=np.float64)
        y = np.asarray(g["y"], dtype=np.float64)
        cluster = np.asarray(g["cluster"])
        fit = bias_corrected_local_linear(
            d, y, boundary=0.0, kernel="epanechnikov",
            h=g["h"], b=g["b"], cluster=cluster,
        )
        # Cluster IDs are sequential (1..50), so np.unique order matches R's
        # unique(); bit-parity is achievable.
        np.testing.assert_allclose(fit.estimate_classical, g["tau_cl"],
                                   atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(fit.estimate_bias_corrected, g["tau_bc"],
                                   atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(fit.se_classical, g["se_cl"],
                                   atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(fit.se_robust, g["se_rb"],
                                   atol=1e-14, rtol=1e-14)
        # CI bounds at bit-parity too (Python's scipy ppf and R's qnorm
        # agree to ULP; the remaining drift is pure tau.bc +/- z * se.rb
        # floating-point arithmetic).
        np.testing.assert_allclose(fit.ci_low, g["ci_low"],
                                   atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(fit.ci_high, g["ci_high"],
                                   atol=1e-14, rtol=1e-14)

    def test_shifted_boundary_parity_dgp_5(self, golden):
        """Design 1 continuous-near-d_lower: boundary = d.min() > 0."""
        g = golden["dgp5"]
        d = np.asarray(g["d"], dtype=np.float64)
        y = np.asarray(g["y"], dtype=np.float64)
        eval_point = float(g["eval_point_override"])
        fit = bias_corrected_local_linear(
            d, y, boundary=eval_point, kernel="epanechnikov",
        )
        np.testing.assert_allclose(fit.estimate_classical, g["tau_cl"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.estimate_bias_corrected, g["tau_bc"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.se_classical, g["se_cl"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.se_robust, g["se_rb"],
                                   atol=1e-12, rtol=1e-12)
        # CI bounds at the same tolerance (Python scipy ppf vs R qnorm
        # agree to ULP; tau.bc +/- z * se.rb inherits se_rb's drift).
        np.testing.assert_allclose(fit.ci_low, g["ci_low"],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.ci_high, g["ci_high"],
                                   atol=1e-12, rtol=1e-12)


# =============================================================================
# CI behavior
# =============================================================================


class TestCI:
    def _smoke(self):
        rng = np.random.default_rng(0)
        d = rng.uniform(0.0, 1.0, 1000)
        y = d + rng.normal(0, 0.3, 1000)
        return d, y

    def test_ci_symmetric_about_tau_bc(self):
        d, y = self._smoke()
        fit = bias_corrected_local_linear(d, y, boundary=0.0, h=0.2, b=0.2)
        midpoint = 0.5 * (fit.ci_low + fit.ci_high)
        np.testing.assert_allclose(
            midpoint, fit.estimate_bias_corrected, atol=1e-14, rtol=1e-14,
        )

    def test_alpha_01_wider_than_alpha_05(self):
        d, y = self._smoke()
        fit_05 = bias_corrected_local_linear(d, y, h=0.2, b=0.2, alpha=0.05)
        fit_01 = bias_corrected_local_linear(d, y, h=0.2, b=0.2, alpha=0.01)
        # alpha=0.01 gives 99% CI > 95% CI width.
        width_05 = fit_05.ci_high - fit_05.ci_low
        width_01 = fit_01.ci_high - fit_01.ci_low
        assert width_01 > width_05

    def test_alpha_out_of_range_raises(self):
        d = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 20)
        y = d + 0.1
        for bad_alpha in (0.0, 1.0, -0.1, 1.1):
            with pytest.raises(ValueError, match="alpha must be in"):
                bias_corrected_local_linear(d, y, h=0.3, b=0.3, alpha=bad_alpha)


# =============================================================================
# Input contract (mirrors Phase 1b test_bandwidth_selector.py)
# =============================================================================


class TestInputContract:
    def test_empty_raises(self):
        d = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="non-empty"):
            bias_corrected_local_linear(d, y, boundary=0.0, h=0.3)

    def test_non_finite_d_raises(self):
        d = np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        y = np.arange(10, dtype=np.float64)
        with pytest.raises(ValueError, match="d contains non-finite"):
            bias_corrected_local_linear(d, y, boundary=0.0, h=0.3)

    def test_non_finite_y_raises(self):
        d = np.linspace(0.05, 1.0, 20)
        y = d.copy()
        y[5] = np.inf
        with pytest.raises(ValueError, match="y contains non-finite"):
            bias_corrected_local_linear(d, y, boundary=0.0, h=0.3)

    def test_shape_mismatch_raises(self):
        d = np.linspace(0.05, 1.0, 20)
        y = np.linspace(0.0, 1.0, 15)
        with pytest.raises(ValueError, match="same shape"):
            bias_corrected_local_linear(d, y, boundary=0.0, h=0.3)

    def test_negative_dose_raises(self):
        rng = np.random.default_rng(1)
        d = rng.uniform(-0.5, 1.0, 500)
        y = d + rng.normal(0, 0.3, 500)
        with pytest.raises(ValueError, match="Negative dose"):
            bias_corrected_local_linear(d, y, boundary=0.0)

    def test_off_support_boundary_raises(self):
        rng = np.random.default_rng(2)
        d = rng.uniform(0.3, 1.0, 500)
        y = d + rng.normal(0, 0.3, 500)
        with pytest.raises(ValueError, match="not compatible with a Design 1'"):
            bias_corrected_local_linear(d, y, boundary=0.0)

    def test_mass_point_raises(self):
        # d.min() > 0 with > 2% at the mode.
        rng = np.random.default_rng(3)
        d = np.concatenate(
            [np.full(100, 0.1), rng.uniform(0.1, 1.0, 400)]
        )
        y = d + rng.normal(0, 0.3, 500)
        with pytest.raises(NotImplementedError, match="mass-point"):
            bias_corrected_local_linear(d, y, boundary=0.1)

    def test_single_cluster_in_active_window_nans_se(self):
        """Cluster-robust SE is NaN when fewer than two clusters fall inside the
        active kernel window, even with >= 2 clusters globally (the window is a
        subset of the sample selected by the bandwidth). Unidentified clustered
        inference must not report a finite se_robust."""
        rng = np.random.default_rng(0)
        d = np.linspace(0.0, 1.0, 200)
        y = 0.5 * d + 0.1 * rng.standard_normal(200)
        # cluster 0 = doses <= 0.2 (near boundary 0); cluster 1 = doses > 0.2
        cluster = (d > 0.2).astype(int)
        # Narrow bandwidth -> the window at boundary 0 sees only cluster 0.
        bc_narrow = bias_corrected_local_linear(d, y, boundary=0.0, h=0.1, cluster=cluster)
        assert np.isnan(bc_narrow.se_robust)
        assert np.isnan(bc_narrow.ci_low) and np.isnan(bc_narrow.ci_high)
        # Wide bandwidth spanning both clusters -> identified -> finite SE.
        bc_wide = bias_corrected_local_linear(d, y, boundary=0.0, h=0.6, cluster=cluster)
        assert np.isfinite(bc_wide.se_robust)

    def test_cluster_nan_raises(self):
        """Float NaN in cluster IDs is rejected."""
        d = np.linspace(0.0, 1.0, 100)
        y = d.copy()
        cluster = np.repeat(np.arange(10), 10).astype(np.float64)
        cluster[5] = np.nan
        with pytest.raises(ValueError, match="cluster contains missing"):
            bias_corrected_local_linear(d, y, h=0.3, cluster=cluster)

    def test_cluster_object_none_raises(self):
        """Object-dtype cluster with a ``None`` sentinel is rejected.
        Float-only checks let this through; the dtype-agnostic helper
        catches it (CI review PR #340 follow-up P1)."""
        d = np.linspace(0.0, 1.0, 100)
        y = d.copy()
        cluster = np.array(
            [i // 10 if i != 5 else None for i in range(100)], dtype=object
        )
        with pytest.raises(ValueError, match="cluster contains missing"):
            bias_corrected_local_linear(d, y, h=0.3, cluster=cluster)

    def test_cluster_object_nan_raises(self):
        """Object-dtype cluster with np.nan is rejected."""
        d = np.linspace(0.0, 1.0, 100)
        y = d.copy()
        cluster = np.array(
            [i // 10 if i != 5 else np.nan for i in range(100)], dtype=object
        )
        with pytest.raises(ValueError, match="cluster contains missing"):
            bias_corrected_local_linear(d, y, h=0.3, cluster=cluster)

    def test_unknown_kernel_raises(self):
        d = np.linspace(0.05, 1.0, 100)
        y = d.copy()
        with pytest.raises(ValueError, match="Unknown kernel"):
            bias_corrected_local_linear(d, y, h=0.3, kernel="cosine")


# =============================================================================
# Parameter interactions
# =============================================================================


class TestParameterInteractions:
    def test_cluster_with_manual_bandwidth(self):
        """Cluster + user-supplied h/b should just work."""
        rng = np.random.default_rng(11)
        G = 400
        d = rng.uniform(0.0, 1.0, G)
        y = d + rng.normal(0, 0.3, G)
        cluster = np.repeat(np.arange(20), 20)
        fit = bias_corrected_local_linear(
            d, y, h=0.3, b=0.3, cluster=cluster,
        )
        assert np.isfinite(fit.estimate_bias_corrected)
        assert fit.se_robust > 0

    def test_vce_hc0_raises_not_implemented(self):
        """Only vce='nn' is golden-tested in Phase 1c; hc-modes raise."""
        d = np.linspace(0.0, 1.0, 100)
        y = d.copy()
        with pytest.raises(NotImplementedError, match="vce='hc0'"):
            bias_corrected_local_linear(d, y, h=0.3, vce="hc0")

    def test_vce_hc1_raises_not_implemented(self):
        d = np.linspace(0.0, 1.0, 100)
        y = d.copy()
        with pytest.raises(NotImplementedError, match="vce='hc1'"):
            bias_corrected_local_linear(d, y, h=0.3, vce="hc1")

    def test_vce_hc2_raises_not_implemented(self):
        d = np.linspace(0.0, 1.0, 100)
        y = d.copy()
        with pytest.raises(NotImplementedError, match="vce='hc2'"):
            bias_corrected_local_linear(d, y, h=0.3, vce="hc2")

    def test_vce_hc3_raises_not_implemented(self):
        d = np.linspace(0.0, 1.0, 100)
        y = d.copy()
        with pytest.raises(NotImplementedError, match="vce='hc3'"):
            bias_corrected_local_linear(d, y, h=0.3, vce="hc3")


# =============================================================================
# End-to-end
# =============================================================================


class TestEndToEnd:
    def test_auto_bandwidth_covers_truth_at_95(self):
        """Full chain: data -> mse_optimal_bandwidth -> bias_corrected fit ->
        reasonable CI. Directional check (single draw), not a coverage test.

        True DGP: y = beta0 + beta1 * d + noise at d=0 gives m(0) = beta0.
        """
        rng = np.random.default_rng(20260419)
        G = 3000
        d = rng.uniform(0.0, 1.0, G)
        beta0_true = 0.0  # intercept
        beta1 = 1.0
        y = beta0_true + beta1 * d + rng.normal(0, 0.3, G)
        fit = bias_corrected_local_linear(d, y, boundary=0.0)
        # CI should contain the true boundary intercept (=0) for a single
        # seed with G=3000 with high probability. Do a loose sanity check.
        assert fit.ci_low <= beta0_true <= fit.ci_high


# =============================================================================
# NaN-safe CI (CI review PR #340 P0)
# =============================================================================


class TestNaNSafeCI:
    """``bias_corrected_local_linear`` must route the CI through
    ``safe_inference`` so degenerate cases with ``se_robust <= 0`` or
    non-finite ``se_robust`` surface as ``ci_low = ci_high = NaN`` rather
    than a misleading zero-width or infinite CI."""

    def test_constant_y_returns_nan_ci(self):
        """Constant y makes residuals zero; se_robust collapses to 0. CI
        must be (NaN, NaN), not a finite zero-width CI."""
        d = np.linspace(0.0, 1.0, 200)
        y = np.full_like(d, 1.5)  # zero residuals everywhere
        fit = bias_corrected_local_linear(d, y, h=0.3, b=0.3)
        assert fit.se_robust == 0.0 or not np.isfinite(fit.se_robust)
        assert np.isnan(fit.ci_low)
        assert np.isnan(fit.ci_high)

    def test_near_zero_se_returns_nan_ci(self):
        """Near-constant y produces a tiny se_robust; the NaN-safe gate
        fires when it hits zero exactly (covers the exact-fit edge case
        the CI review flagged without tripping a pre-existing Phase 1b
        ZeroDivisionError in the auto-bandwidth selector on truly
        constant y, which is tracked separately)."""
        rng = np.random.default_rng(0)
        d = np.linspace(0.0, 1.0, 500)
        # Residuals near machine epsilon; tau_bc stays finite.
        y = 0.1 * np.ones_like(d) + rng.normal(0, 1e-300, size=d.shape)
        fit = bias_corrected_local_linear(d, y, h=0.3, b=0.3)
        # Either the inference should be fully valid, OR the CI gate has
        # correctly fired. The contract is: se_rb <= 0 / non-finite =>
        # NaN CI.
        if not (np.isfinite(fit.se_robust) and fit.se_robust > 0):
            assert np.isnan(fit.ci_low)
            assert np.isnan(fit.ci_high)


# =============================================================================
# Auto-bandwidth parameter forwarding (CI review PR #340 P1)
# =============================================================================


class TestAutoBandwidthForwardsParameters:
    """Auto-bandwidth must forward ``cluster``, ``vce``, and ``nnmatch`` to
    the bandwidth selector. Calling the public ``mse_optimal_bandwidth``
    wrapper would hard-code ``cluster=None, vce="nn", nnmatch=3`` and
    silently mismatch the downstream ``lprobust`` fit — a methodology
    bug. These tests pin the correct wiring."""

    def _smoke_data(self, seed=33):
        rng = np.random.default_rng(seed)
        G = 600
        d = rng.uniform(0.0, 1.0, G)
        y = d + d ** 2 + rng.normal(0, 0.3, G)
        return d, y

    def test_auto_cluster_returns_finite(self):
        """Auto-bandwidth with cluster produces a finite BiasCorrectedFit.

        No R parity anchor (nprobust's internal lpbwselect has a
        singleton-cluster bug on the pilot fits); this test pins that the
        Python path completes and uses the clustered bandwidth downstream,
        not the unclustered one.
        """
        d, y = self._smoke_data()
        cluster = np.repeat(np.arange(30), 20)
        fit_cluster = bias_corrected_local_linear(d, y, cluster=cluster)
        fit_uncluster = bias_corrected_local_linear(d, y)
        assert fit_cluster.bandwidth_source == "auto"
        assert np.isfinite(fit_cluster.estimate_bias_corrected)
        assert np.isfinite(fit_cluster.se_robust)
        # The clustered bandwidth should differ from the unclustered one
        # (different residual meat feeds into Stage-2/3 AMSE minimization).
        # If the wrapper were silently passing cluster=None, these would
        # be identical to bit-parity.
        assert fit_cluster.h != fit_uncluster.h

    def test_auto_vce_hc1_rejected_in_phase_1c(self):
        """Phase 1c public wrapper restricts vce to 'nn'; hc-modes raise.
        Port-level lprobust still forwards vce through, so
        lpbwselect_mse_dpi gets the correct setting when unlocked in
        Phase 2+."""
        d, y = self._smoke_data()
        with pytest.raises(NotImplementedError, match="vce='hc1'"):
            bias_corrected_local_linear(d, y, vce="hc1")

    def test_auto_nnmatch_non_default_returns_finite(self):
        """Auto-bandwidth with non-default nnmatch must forward it to the
        selector, not silently use the hard-coded default of 3."""
        d, y = self._smoke_data()
        fit_nn5 = bias_corrected_local_linear(d, y, nnmatch=5)
        fit_nn3 = bias_corrected_local_linear(d, y, nnmatch=3)
        assert fit_nn5.bandwidth_source == "auto"
        assert np.isfinite(fit_nn5.estimate_bias_corrected)
        # nnmatch controls the NN residual construction; different values
        # give different meat matrices and therefore different stage
        # bandwidths.
        assert fit_nn5.h != fit_nn3.h


# =============================================================================
# Validator idempotence (regression gate for the Phase 1b extraction)
# =============================================================================


class TestValidatorIdempotence:
    """Plan commit criterion #5: _validate_had_inputs extraction must be
    idempotent (coerced output remains valid on re-validation)."""

    def test_idempotent_on_valid_inputs(self):
        from diff_diff.local_linear import _validate_had_inputs

        rng = np.random.default_rng(42)
        d_raw = rng.uniform(0.0, 1.0, 200)
        y_raw = rng.normal(0, 0.3, 200)
        d1, y1 = _validate_had_inputs(d_raw, y_raw, 0.0)
        d2, y2 = _validate_had_inputs(d1, y1, 0.0)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(y1, y2)

    def test_idempotent_at_boundary_d_min(self):
        from diff_diff.local_linear import _validate_had_inputs

        rng = np.random.default_rng(7)
        d_raw = rng.uniform(0.2, 1.0, 200)
        y_raw = rng.normal(0, 0.3, 200)
        boundary = float(d_raw.min())
        d1, y1 = _validate_had_inputs(d_raw, y_raw, boundary)
        d2, y2 = _validate_had_inputs(d1, y1, boundary)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(y1, y2)


class TestWeightedBiasCorrectedLocalLinear:
    """Phase 4.5 survey-support validation for the public wrapper.

    Phase 1c's unweighted R-parity path stays unchanged (regression lock
    via the existing TestParity class). This class adds:
      1. Uniform-weights bit-parity: weights=np.ones(G) ≡ no weights
      2. Weight-validator contract mirroring the port's front-door guards
      3. Informative-weight mechanism-has-teeth sanity
    """

    def _panel(self, seed=42, G=300, boundary=0.0):
        rng = np.random.default_rng(seed)
        d = rng.uniform(boundary, boundary + 1.0, G)
        y = (
            2.0 * (d - boundary)
            + 0.3 * (d - boundary) ** 2
            + rng.normal(0, 0.2, G)
        )
        return d, y

    def test_uniform_weights_bit_parity_full_struct(self):
        from diff_diff.local_linear import bias_corrected_local_linear

        d, y = self._panel()
        base = bias_corrected_local_linear(d, y, boundary=0.0, h=0.3, b=0.3)
        w1 = bias_corrected_local_linear(
            d, y, boundary=0.0, h=0.3, b=0.3, weights=np.ones(d.size)
        )
        np.testing.assert_allclose(
            w1.estimate_classical, base.estimate_classical, atol=1e-14, rtol=1e-14
        )
        np.testing.assert_allclose(
            w1.estimate_bias_corrected, base.estimate_bias_corrected,
            atol=1e-14, rtol=1e-14,
        )
        np.testing.assert_allclose(
            w1.se_classical, base.se_classical, atol=1e-14, rtol=1e-14
        )
        np.testing.assert_allclose(
            w1.se_robust, base.se_robust, atol=1e-14, rtol=1e-14
        )
        np.testing.assert_allclose(w1.ci_low, base.ci_low, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.ci_high, base.ci_high, atol=1e-14, rtol=1e-14)
        assert w1.n_used == base.n_used
        assert w1.h == base.h
        assert w1.b == base.b

    def test_uniform_weights_bit_parity_auto_bandwidth(self):
        """Auto-bandwidth path: the DPI selector is unweighted in this PR
        (documented gap), so weights=ones must produce the same h,b,estimate."""
        from diff_diff.local_linear import bias_corrected_local_linear

        d, y = self._panel()
        base = bias_corrected_local_linear(d, y, boundary=0.0)
        w1 = bias_corrected_local_linear(
            d, y, boundary=0.0, weights=np.ones(d.size)
        )
        np.testing.assert_allclose(w1.h, base.h, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(w1.b, base.b, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(
            w1.estimate_bias_corrected, base.estimate_bias_corrected,
            atol=1e-14, rtol=1e-14,
        )

    def test_nontrivial_weights_change_estimate(self):
        from diff_diff.local_linear import bias_corrected_local_linear

        d, y = self._panel()
        base = bias_corrected_local_linear(d, y, boundary=0.0, h=0.3, b=0.3)
        w = np.exp(-d * 3.0)  # informative: concentrate near boundary
        r_w = bias_corrected_local_linear(
            d, y, boundary=0.0, h=0.3, b=0.3, weights=w
        )
        assert not np.isclose(
            r_w.estimate_bias_corrected, base.estimate_bias_corrected, atol=1e-6
        )

    def test_negative_weights_reject(self):
        from diff_diff.local_linear import bias_corrected_local_linear

        d, y = self._panel()
        w = np.ones(d.size)
        w[0] = -0.5
        with pytest.raises(ValueError, match="non-negative"):
            bias_corrected_local_linear(
                d, y, boundary=0.0, h=0.3, b=0.3, weights=w
            )

    def test_nan_weights_reject(self):
        from diff_diff.local_linear import bias_corrected_local_linear

        d, y = self._panel()
        w = np.ones(d.size)
        w[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            bias_corrected_local_linear(
                d, y, boundary=0.0, h=0.3, b=0.3, weights=w
            )

    def test_zero_sum_weights_reject(self):
        from diff_diff.local_linear import bias_corrected_local_linear

        d, y = self._panel()
        with pytest.raises(ValueError, match="sum to zero"):
            bias_corrected_local_linear(
                d, y, boundary=0.0, h=0.3, b=0.3, weights=np.zeros(d.size)
            )

    def test_weights_length_mismatch_reject(self):
        from diff_diff.local_linear import bias_corrected_local_linear

        d, y = self._panel()
        with pytest.raises(ValueError, match="weights length"):
            bias_corrected_local_linear(
                d, y, boundary=0.0, h=0.3, b=0.3, weights=np.ones(d.size - 1)
            )
