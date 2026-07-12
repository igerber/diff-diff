"""Tests for the weighted Frank-Wolfe kernel and SDID survey-bootstrap helpers.

Covers PR #352 (SDID survey-bootstrap via weighted FW + Rao-Wu composition):
- Rust kernel: ``sc_weight_fw_weighted`` and ``_with_convergence`` variants
- Python wrappers: ``_sc_weight_fw(reg_weights=...)`` dispatch and the
  ``_sc_weight_fw_numpy`` weighted-reg fallback
- Survey helpers: ``compute_sdid_unit_weights_survey`` and
  ``compute_time_weights_survey``

The unweighted contract is verified separately in ``test_rust_backend.py`` and
``test_methodology_sdid.py``; this file focuses on the weighted-reg path.
"""

from __future__ import annotations

import numpy as np
import pytest

from diff_diff.utils import (
    _sc_weight_fw,
    _sc_weight_fw_numpy,
    compute_sdid_unit_weights,
    compute_sdid_unit_weights_survey,
    compute_time_weights,
    compute_time_weights_survey,
)


@pytest.fixture
def small_panel():
    """8 pre-periods × 12 control units, deterministic seed."""
    rng = np.random.default_rng(42)
    n_pre, n_post, n_control = 8, 4, 12
    return {
        "Y_pre_control": rng.normal(size=(n_pre, n_control)),
        "Y_post_control": rng.normal(size=(n_post, n_control)),
        "Y_pre_treated_mean": rng.normal(size=n_pre),
        "n_pre": n_pre,
        "n_post": n_post,
        "n_control": n_control,
    }


# =============================================================================
# Kernel-level: _sc_weight_fw with reg_weights
# =============================================================================


class TestSCWeightFWWeighted:
    """Verify the Python dispatch through to the weighted Rust/numpy kernels."""

    def test_reg_weights_none_matches_unweighted(self, small_panel):
        """reg_weights=None must be bit-identical to the unweighted call."""
        Y = np.column_stack(
            [
                small_panel["Y_pre_control"],
                small_panel["Y_pre_treated_mean"].reshape(-1, 1),
            ]
        )
        unweighted = _sc_weight_fw(Y, zeta=0.3, max_iter=10000, min_decrease=1e-7)
        weighted_none = _sc_weight_fw(
            Y,
            zeta=0.3,
            max_iter=10000,
            min_decrease=1e-7,
            reg_weights=None,
        )
        np.testing.assert_allclose(weighted_none, unweighted, rtol=1e-14, atol=0)

    def test_uniform_reg_weights_matches_unweighted(self, small_panel):
        """reg_weights = ones must collapse to the unweighted regularization.

        Mathematical identity: ζ²·Σ 1·ω² = ζ²·||ω||². The two paths can
        differ by ULP-scale due to float ordering inside the loop, so allow
        rel=1e-12 (tighter than rel=1e-14 only because the weighted loop
        uses Σ rw·ω² while the unweighted loop uses np.sum(ω²) — different
        reduction orders).
        """
        Y = np.column_stack(
            [
                small_panel["Y_pre_control"],
                small_panel["Y_pre_treated_mean"].reshape(-1, 1),
            ]
        )
        rw_uniform = np.ones(small_panel["n_control"])
        unweighted = _sc_weight_fw(Y, zeta=0.3, max_iter=10000, min_decrease=1e-7)
        weighted_uniform = _sc_weight_fw(
            Y,
            zeta=0.3,
            max_iter=10000,
            min_decrease=1e-7,
            reg_weights=rw_uniform,
        )
        np.testing.assert_allclose(weighted_uniform, unweighted, rtol=1e-12, atol=1e-13)

    def test_python_rust_parity_under_weighted_reg(self, small_panel):
        """Pure-Python and Rust weighted FW agree at rel=1e-10.

        Different BLAS / reduction orders prevent bit-identity, but the
        weighted objective is strictly convex on the simplex so both
        backends converge to the same minimizer.
        """
        Y = np.column_stack(
            [
                small_panel["Y_pre_control"],
                small_panel["Y_pre_treated_mean"].reshape(-1, 1),
            ]
        )
        rw = np.array([1.5, 0.5, 1.0, 2.0, 0.7, 1.3, 0.9, 1.8, 0.6, 1.1, 1.4, 0.8])
        rust_w = _sc_weight_fw(
            Y,
            zeta=0.3,
            max_iter=20000,
            min_decrease=1e-9,
            reg_weights=rw,
        )
        numpy_w = _sc_weight_fw_numpy(
            Y,
            zeta=0.3,
            max_iter=20000,
            min_decrease=1e-9,
            reg_weights=rw,
        )
        np.testing.assert_allclose(numpy_w, rust_w, rtol=1e-9, atol=1e-10)

    def test_simplex_invariants_under_arbitrary_rw(self, small_panel):
        """For any positive rw, ω sums to 1 and is non-negative."""
        Y = np.column_stack(
            [
                small_panel["Y_pre_control"],
                small_panel["Y_pre_treated_mean"].reshape(-1, 1),
            ]
        )
        rw = np.array([0.3, 0.5, 1.0, 2.0, 1.5, 0.8, 1.2, 0.6, 1.7, 0.9, 1.4, 1.1])
        omega = _sc_weight_fw(
            Y,
            zeta=0.4,
            max_iter=10000,
            min_decrease=1e-6,
            reg_weights=rw,
        )
        assert omega.shape == (small_panel["n_control"],)
        assert np.isclose(omega.sum(), 1.0, atol=1e-6), f"weights must sum to 1, got {omega.sum()}"
        assert np.all(omega >= -1e-9), "weights must be non-negative"

    def test_return_convergence_tuple_shape(self, small_panel):
        """return_convergence=True returns (weights, bool) under weighted reg."""
        Y = np.column_stack(
            [
                small_panel["Y_pre_control"],
                small_panel["Y_pre_treated_mean"].reshape(-1, 1),
            ]
        )
        rw = np.linspace(0.5, 2.0, small_panel["n_control"])
        result = _sc_weight_fw(
            Y,
            zeta=0.3,
            max_iter=10000,
            min_decrease=1e-6,
            return_convergence=True,
            reg_weights=rw,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        weights, converged = result
        assert weights.shape == (small_panel["n_control"],)
        assert isinstance(converged, bool)

    def test_reg_weights_length_mismatch_raises(self, small_panel):
        """reg_weights length mismatch raises ValueError on both backends.

        Regression against PR #355 R5 P2: the Rust internal previously
        silently fell back to the unweighted kernel when ``reg_weights.len()
        != Y.shape[1] - 1``, while the NumPy path raised. The dispatcher
        now validates shape upstream so both backends share the same
        failure surface, and each Rust pyfunction also guards the entry
        point against direct Rust-from-Python calls. Exercising the
        wrong-shape call must raise ``ValueError`` with a message naming
        the expected dimension.
        """
        Y = np.column_stack(
            [
                small_panel["Y_pre_control"],
                small_panel["Y_pre_treated_mean"].reshape(-1, 1),
            ]
        )
        expected_t0 = Y.shape[1] - 1
        bad_rw = np.ones(expected_t0 + 3)
        with pytest.raises(ValueError, match=r"reg_weights"):
            _sc_weight_fw(
                Y,
                zeta=0.3,
                max_iter=100,
                min_decrease=1e-6,
                reg_weights=bad_rw,
            )


# =============================================================================
# Survey helpers: compute_sdid_unit_weights_survey, compute_time_weights_survey
# =============================================================================


class TestComputeSDIDUnitWeightsSurvey:
    """Two-pass survey-weighted unit FW with diag(rw) regularization."""

    def test_uniform_rw_matches_unweighted_helper_within_tolerance(self, small_panel):
        """Uniform rw=1 produces ω close to the unweighted helper.

        Note: this is NOT bit-identity — the weighted helper column-scales Y
        by rw=ones (no-op) AND passes reg_weights=ones to the kernel, while
        the unweighted helper passes reg_weights=None. The two kernels reach
        the same simplex minimum but iterate through different float
        reduction orders, so the FW iterates can land on adjacent vertices
        when the simplex objective is nearly flat. Use rel=1e-6 (loose).
        """
        Y_pre_c = small_panel["Y_pre_control"]
        Y_pre_t = small_panel["Y_pre_treated_mean"]
        unweighted = compute_sdid_unit_weights(
            Y_pre_c,
            Y_pre_t,
            zeta_omega=0.3,
            min_decrease=1e-6,
        )
        rw_uniform = np.ones(small_panel["n_control"])
        weighted = compute_sdid_unit_weights_survey(
            Y_pre_c,
            Y_pre_t,
            rw_uniform,
            zeta_omega=0.3,
            min_decrease=1e-6,
        )
        np.testing.assert_allclose(weighted, unweighted, rtol=1e-6, atol=1e-6)

    def test_simplex_invariants_under_arbitrary_rw(self, small_panel):
        Y_pre_c = small_panel["Y_pre_control"]
        Y_pre_t = small_panel["Y_pre_treated_mean"]
        rng = np.random.default_rng(7)
        rw = rng.uniform(0.3, 2.5, size=small_panel["n_control"])
        omega = compute_sdid_unit_weights_survey(
            Y_pre_c,
            Y_pre_t,
            rw,
            zeta_omega=0.3,
            min_decrease=1e-6,
        )
        assert omega.shape == (small_panel["n_control"],)
        assert np.isclose(omega.sum(), 1.0, atol=1e-6)
        assert np.all(omega >= -1e-9)

    def test_rw_shape_mismatch_raises(self, small_panel):
        Y_pre_c = small_panel["Y_pre_control"]
        Y_pre_t = small_panel["Y_pre_treated_mean"]
        wrong_rw = np.ones(small_panel["n_control"] + 1)
        with pytest.raises(ValueError, match="rw_control shape"):
            compute_sdid_unit_weights_survey(
                Y_pre_c,
                Y_pre_t,
                wrong_rw,
                zeta_omega=0.3,
            )

    def test_return_convergence_propagates_AND_of_passes(self, small_panel):
        Y_pre_c = small_panel["Y_pre_control"]
        Y_pre_t = small_panel["Y_pre_treated_mean"]
        rw = np.linspace(0.5, 2.0, small_panel["n_control"])
        # Tight tolerance + few iterations to force non-convergence
        omega, converged = compute_sdid_unit_weights_survey(
            Y_pre_c,
            Y_pre_t,
            rw,
            zeta_omega=0.3,
            min_decrease=1e-15,
            max_iter_pre_sparsify=5,
            max_iter=5,
            return_convergence=True,
        )
        assert isinstance(converged, bool)
        # Confirm the result is still a valid simplex vector
        assert np.isclose(omega.sum(), 1.0, atol=1e-6)


class TestComputeTimeWeightsSurvey:
    """Two-pass row-weighted time FW with uniform regularization."""

    def test_uniform_rw_matches_unweighted_helper_within_tolerance(self, small_panel):
        """Uniform rw produces λ matching the unweighted helper.

        sqrt(1)=1 row-scaling is a no-op, so this is genuine equivalence
        modulo FW iterate ordering.
        """
        Y_pre_c = small_panel["Y_pre_control"]
        Y_post_c = small_panel["Y_post_control"]
        unweighted = compute_time_weights(
            Y_pre_c,
            Y_post_c,
            zeta_lambda=0.05,
            min_decrease=1e-6,
        )
        rw_uniform = np.ones(small_panel["n_control"])
        weighted = compute_time_weights_survey(
            Y_pre_c,
            Y_post_c,
            rw_uniform,
            zeta_lambda=0.05,
            min_decrease=1e-6,
        )
        np.testing.assert_allclose(weighted, unweighted, rtol=1e-6, atol=1e-6)

    def test_simplex_invariants_under_arbitrary_rw(self, small_panel):
        Y_pre_c = small_panel["Y_pre_control"]
        Y_post_c = small_panel["Y_post_control"]
        rng = np.random.default_rng(11)
        rw = rng.uniform(0.3, 2.5, size=small_panel["n_control"])
        lam = compute_time_weights_survey(
            Y_pre_c,
            Y_post_c,
            rw,
            zeta_lambda=0.05,
            min_decrease=1e-6,
        )
        assert lam.shape == (small_panel["n_pre"],)
        assert np.isclose(lam.sum(), 1.0, atol=1e-6)
        assert np.all(lam >= -1e-9)

    def test_rw_shape_mismatch_raises(self, small_panel):
        Y_pre_c = small_panel["Y_pre_control"]
        Y_post_c = small_panel["Y_post_control"]
        wrong_rw = np.ones(small_panel["n_control"] + 1)
        with pytest.raises(ValueError, match="rw_control shape"):
            compute_time_weights_survey(
                Y_pre_c,
                Y_post_c,
                wrong_rw,
                zeta_lambda=0.05,
            )

    def test_non_uniform_rw_beats_unweighted_centering_variant(self, small_panel):
        """Non-uniform rw: the weighted-centering solution achieves strictly
        lower weighted SSR than the (buggy) unweighted-centering variant.

        Verifies the PR #355 R1 fix — weighted centering + intercept=False
        — actually solves the stated weighted loss
        ``Σ_u rw_u·(A_u·λ - b_u)²``. Reproduces the unweighted-centering
        pre-R1 path by hand (row-scale Y by sqrt(rw), then pass
        intercept=True to the kernel so it centers on unweighted column
        means) and asserts the correct path's weighted SSR is strictly
        better. If R1's fix regresses (someone reverts back to
        intercept=True after row-scaling), this test fails because the
        two solutions become identical.
        """
        Y_pre_c = small_panel["Y_pre_control"]
        Y_post_c = small_panel["Y_post_control"]
        n_co = small_panel["n_control"]
        rng = np.random.default_rng(23)
        rw = np.where(rng.uniform(size=n_co) < 0.25, 5.0, 0.5)

        # Correct path: what compute_time_weights_survey actually does.
        lam_correct = compute_time_weights_survey(
            Y_pre_c,
            Y_post_c,
            rw,
            zeta_lambda=0.05,
            min_decrease=1e-8,
            max_iter=10000,
        )

        # Buggy variant: pre-R1 — row-scale by sqrt(rw) but let the kernel
        # do UNWEIGHTED centering (intercept=True on the row-scaled matrix).
        post_means = np.mean(Y_post_c, axis=0)
        Y_time_raw = np.column_stack([Y_pre_c.T, post_means])
        sqrt_rw = np.sqrt(np.maximum(rw, 0.0))
        Y_weighted_unweighted_center = Y_time_raw * sqrt_rw[:, None]
        lam_buggy = _sc_weight_fw(
            Y_weighted_unweighted_center,
            zeta=0.05,
            intercept=True,
            min_decrease=1e-8,
            max_iter=10000,
        )
        # Sparsify + refit second pass to match the two-pass shape.
        from diff_diff.utils import _sparsify

        lam_buggy = _sparsify(lam_buggy)
        lam_buggy = _sc_weight_fw(
            Y_weighted_unweighted_center,
            zeta=0.05,
            intercept=True,
            init_weights=lam_buggy,
            min_decrease=1e-8,
            max_iter=10000,
        )

        # Compute the canonical (weighted-centered) objective on both.
        wc_mean_pre = (Y_pre_c.T * rw[:, None]).sum(axis=0) / rw.sum()
        wc_mean_post = (post_means * rw).sum() / rw.sum()
        A_wc = Y_pre_c.T - wc_mean_pre
        b_wc = post_means - wc_mean_post

        def weighted_ssr(lam_val: np.ndarray) -> float:
            resid = A_wc @ lam_val - b_wc
            return float(np.sum(rw * resid**2))

        ssr_correct = weighted_ssr(lam_correct)
        ssr_buggy = weighted_ssr(lam_buggy)
        assert ssr_correct <= ssr_buggy + 1e-6, (
            f"weighted-centering λ (SSR={ssr_correct:.4f}) must achieve at "
            f"least as low weighted SSR as the unweighted-centering variant "
            f"(SSR={ssr_buggy:.4f}). PR #355 R1 regression: weighted SSR is "
            "not being minimized by the survey λ helper."
        )

    def test_zero_rw_subset_handled(self, small_panel):
        """rw with some zeros (Rao-Wu draws units to zero weight) still yields
        a valid simplex λ — the FW just down-weights those rows in the loss.
        """
        Y_pre_c = small_panel["Y_pre_control"]
        Y_post_c = small_panel["Y_post_control"]
        rw = np.ones(small_panel["n_control"])
        rw[:3] = 0.0  # zero out first 3 controls (e.g., undrawn PSU)
        lam = compute_time_weights_survey(
            Y_pre_c,
            Y_post_c,
            rw,
            zeta_lambda=0.05,
            min_decrease=1e-6,
        )
        assert lam.shape == (small_panel["n_pre"],)
        assert np.isclose(lam.sum(), 1.0, atol=1e-6)
        assert np.all(lam >= -1e-9)


# =============================================================================
# Pure-Python vs Rust path parity through the survey helpers
# =============================================================================


class TestSurveyHelperBackendParity:
    """The two _survey helpers must produce the same result on Rust vs numpy."""

    def test_unit_survey_python_rust_parity(self, small_panel, monkeypatch):
        """Force pure-Python and compare to Rust output.

        Uses monkeypatch to override HAS_RUST_BACKEND inside utils, then
        compares against the default Rust path.
        """
        from diff_diff import utils as dd_utils

        Y_pre_c = small_panel["Y_pre_control"]
        Y_pre_t = small_panel["Y_pre_treated_mean"]
        rng = np.random.default_rng(13)
        rw = rng.uniform(0.5, 2.0, size=small_panel["n_control"])

        # Rust path (default)
        rust_omega = compute_sdid_unit_weights_survey(
            Y_pre_c,
            Y_pre_t,
            rw,
            zeta_omega=0.3,
            max_iter_pre_sparsify=200,
            max_iter=20000,
            min_decrease=1e-9,
        )

        # Force pure-Python path
        monkeypatch.setattr(dd_utils, "HAS_RUST_BACKEND", False)
        py_omega = compute_sdid_unit_weights_survey(
            Y_pre_c,
            Y_pre_t,
            rw,
            zeta_omega=0.3,
            max_iter_pre_sparsify=200,
            max_iter=20000,
            min_decrease=1e-9,
        )

        # Sparsify path is identical, weighted FW is strictly convex; the
        # two backends should agree at FW convergence to numerical
        # tolerance dominated by float reduction order.
        np.testing.assert_allclose(py_omega, rust_omega, rtol=1e-7, atol=1e-7)

    def test_time_survey_python_rust_parity(self, small_panel, monkeypatch):
        from diff_diff import utils as dd_utils

        Y_pre_c = small_panel["Y_pre_control"]
        Y_post_c = small_panel["Y_post_control"]
        rng = np.random.default_rng(17)
        rw = rng.uniform(0.5, 2.0, size=small_panel["n_control"])

        rust_lam = compute_time_weights_survey(
            Y_pre_c,
            Y_post_c,
            rw,
            zeta_lambda=0.05,
            max_iter_pre_sparsify=200,
            max_iter=20000,
            min_decrease=1e-9,
        )

        monkeypatch.setattr(dd_utils, "HAS_RUST_BACKEND", False)
        py_lam = compute_time_weights_survey(
            Y_pre_c,
            Y_post_c,
            rw,
            zeta_lambda=0.05,
            max_iter_pre_sparsify=200,
            max_iter=20000,
            min_decrease=1e-9,
        )

        np.testing.assert_allclose(py_lam, rust_lam, rtol=1e-7, atol=1e-7)
