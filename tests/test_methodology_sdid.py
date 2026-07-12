"""
Methodology tests for Synthetic Difference-in-Differences (SDID).

Tests verify the implementation matches Arkhangelsky et al. (2021) and
R's synthdid package behavior: Frank-Wolfe solver, collapsed form,
auto-regularization, sparsification, and correct bootstrap/placebo SE.
"""

import warnings
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from diff_diff.synthetic_did import SyntheticDiD
from diff_diff.utils import (
    _compute_noise_level,
    _compute_noise_level_numpy,
    _compute_regularization,
    _fw_step,
    _sc_weight_fw,
    _sc_weight_fw_numpy,
    _sparsify,
    _sum_normalize,
    compute_sdid_estimator,
    compute_sdid_unit_weights,
    compute_time_weights,
    safe_inference,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _make_panel(n_control=20, n_treated=3, n_pre=5, n_post=3, att=5.0, seed=42):
    """Create a simple panel dataset for testing."""
    rng = np.random.default_rng(seed)
    data = []
    for unit in range(n_control + n_treated):
        is_treated = unit >= n_control
        unit_fe = rng.normal(0, 2)
        for t in range(n_pre + n_post):
            y = 10.0 + unit_fe + t * 0.3 + rng.normal(0, 0.5)
            if is_treated and t >= n_pre:
                y += att
            data.append(
                {
                    "unit": unit,
                    "period": t,
                    "treated": int(is_treated),
                    "outcome": y,
                }
            )
    import pandas as pd

    return pd.DataFrame(data)


# =============================================================================
# Phase A: Noise Level and Regularization
# =============================================================================


class TestNoiseLevel:
    """Verify _compute_noise_level matches hand-computed first-diff sd."""

    def test_known_values(self):
        """Test with a simple matrix where first-diffs are known."""
        # 3 time periods, 2 control units
        # Unit 0: [1, 3, 6] -> diffs: [2, 3]
        # Unit 1: [2, 2, 5] -> diffs: [0, 3]
        # All diffs: [2, 3, 0, 3], sd(ddof=1) = std([2,3,0,3], ddof=1)
        Y = np.array([[1.0, 2.0], [3.0, 2.0], [6.0, 5.0]])
        expected = np.std([2.0, 3.0, 0.0, 3.0], ddof=1)
        result = _compute_noise_level(Y)
        assert abs(result - expected) < 1e-10

    def test_single_period(self):
        """Single period -> no diffs possible -> noise level = 0."""
        Y = np.array([[1.0, 2.0, 3.0]])
        assert _compute_noise_level(Y) == 0.0

    def test_two_periods(self):
        """Two periods -> one diff per unit."""
        Y = np.array([[1.0, 4.0], [3.0, 7.0]])
        # Diffs: [2.0, 3.0], sd(ddof=1)
        expected = np.std([2.0, 3.0], ddof=1)
        assert abs(_compute_noise_level(Y) - expected) < 1e-10


class TestRegularization:
    """Verify _compute_regularization formula with known inputs."""

    def test_formula(self):
        """Check zeta_omega = (N1*T1)^0.25 * sigma, zeta_lambda = 1e-6 * sigma."""
        # Use a simple Y_pre_control where sigma is easy to compute
        Y = np.array([[1.0, 2.0], [3.0, 4.0], [6.0, 7.0]])
        sigma = _compute_noise_level(Y)
        n_treated, n_post = 2, 3

        zeta_omega, zeta_lambda = _compute_regularization(Y, n_treated, n_post)

        expected_omega = (n_treated * n_post) ** 0.25 * sigma
        expected_lambda = 1e-6 * sigma

        assert abs(zeta_omega - expected_omega) < 1e-10
        assert abs(zeta_lambda - expected_lambda) < 1e-15

    def test_zero_noise(self):
        """Constant outcomes -> zero noise -> zero regularization."""
        Y = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        zo, zl = _compute_regularization(Y, 2, 3)
        assert zo == 0.0
        assert zl == 0.0


# =============================================================================
# Phase B: Frank-Wolfe Solver
# =============================================================================


class TestFrankWolfe:
    """Verify Frank-Wolfe step and solver behavior."""

    def test_fw_step_descent(self):
        """A single FW step should not increase the half-gradient objective.

        The FW step minimizes the linearized objective at the current point.
        After the step with exact line search, the true objective should
        not increase when measured correctly.
        """
        rng = np.random.default_rng(42)
        N, T0 = 10, 5
        A = rng.standard_normal((N, T0))
        b = rng.standard_normal(N)
        eta = N * 0.1**2  # eta = N * zeta^2 matching R's formulation

        x = np.ones(T0) / T0

        # Run a few steps to get away from the initial uniform point
        # (first step from uniform can have numerical issues)
        for _ in range(5):
            x = _fw_step(A, x, b, eta)

        def objective(lam):
            err = A @ lam - b
            return (eta / N) * np.sum(lam**2) + np.sum(err**2) / N

        obj_before = objective(x)
        x_new = _fw_step(A, x, b, eta)
        obj_after = objective(x_new)

        assert obj_after <= obj_before + 1e-8

    def test_fw_step_on_simplex(self):
        """FW step should return a vector on the simplex."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 4))
        b = rng.standard_normal(8)
        x = np.array([0.25, 0.25, 0.25, 0.25])

        x_new = _fw_step(A, x, b, eta=1.0)
        assert np.all(x_new >= -1e-10)
        assert abs(np.sum(x_new) - 1.0) < 1e-10

    def test_sc_weight_fw_converges(self):
        """Full FW solver should converge on a known QP."""
        rng = np.random.default_rng(42)
        N, T0 = 15, 6
        Y = rng.standard_normal((N, T0 + 1))  # last col is target

        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=1000)
        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-6

    def test_sc_weight_fw_max_iter_zero(self):
        """max_iter=0 should return initial uniform weights."""
        Y = np.random.randn(5, 4)  # (N, T0+1) with T0=3
        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=0)
        expected = np.ones(3) / 3
        np.testing.assert_allclose(lam, expected, atol=1e-10)

    def test_sc_weight_fw_max_iter_one(self):
        """max_iter=1 should return weights after one step (still on simplex)."""
        rng = np.random.default_rng(99)
        Y = rng.standard_normal((8, 5))
        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=1)
        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-10

    def test_intercept_centering(self):
        """With intercept=True, column-centering should occur."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((10, 5)) + 100  # large offset
        lam_intercept = _sc_weight_fw(Y, zeta=0.1, intercept=True)
        lam_no_intercept = _sc_weight_fw(Y, zeta=0.1, intercept=False)
        # Both should be on simplex but may differ
        assert abs(np.sum(lam_intercept) - 1.0) < 1e-6
        assert abs(np.sum(lam_no_intercept) - 1.0) < 1e-6
        # They should be different because centering matters
        assert not np.allclose(lam_intercept, lam_no_intercept, atol=1e-3)

    def test_fw_warns_on_nonconvergence(self):
        """Silent-failure audit axis B: _sc_weight_fw_numpy must warn when max_iter exhausts."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((15, 7))  # (N, T0+1) with T0=6

        with pytest.warns(UserWarning, match="did not converge"):
            _sc_weight_fw_numpy(Y, zeta=0.1, max_iter=1, min_decrease=1e-12)

    def test_fw_no_warning_on_convergence(self):
        """Silent-failure audit axis B: no warning on well-conditioned convergent input."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((15, 7))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _sc_weight_fw_numpy(Y, zeta=0.1, max_iter=10000, min_decrease=1e-3)
        assert not any("did not converge" in str(x.message) for x in w)

    def test_fw_wrapper_warns_on_nonconvergence_without_rust(self):
        """Silent-failure audit axis B: public _sc_weight_fw wrapper must route
        warnings through even when called via the dispatcher with the Rust
        backend disabled. Pins the contract against refactors that would
        bypass the numpy path."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((15, 7))

        with patch("diff_diff.utils.HAS_RUST_BACKEND", False):
            with pytest.warns(UserWarning, match="did not converge"):
                _sc_weight_fw(Y, zeta=0.1, max_iter=1, min_decrease=1e-12)

    def test_fw_wrapper_no_warning_on_convergence_without_rust(self):
        """Silent-failure audit axis B: wrapper-level negative control."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((15, 7))

        with patch("diff_diff.utils.HAS_RUST_BACKEND", False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _sc_weight_fw(Y, zeta=0.1, max_iter=10000, min_decrease=1e-3)
            assert not any("did not converge" in str(x.message) for x in w)

    def test_fw_max_iter_zero_warns(self):
        """Silent-failure audit axis B: max_iter=0 produces the uniform init
        without iterating, which cannot converge by construction. The warning
        must fire (consistent with the convention: if we exited the loop
        without hitting the tolerance gate, we signal). Pins this contract."""
        Y = np.random.default_rng(0).standard_normal((5, 4))

        with patch("diff_diff.utils.HAS_RUST_BACKEND", False):
            with pytest.warns(UserWarning, match="did not converge"):
                _sc_weight_fw(Y, zeta=0.1, max_iter=0)


class TestSparsify:
    """Verify sparsification behavior."""

    def test_basic(self):
        """Weights below max/4 should be zeroed."""
        v = np.array([0.8, 0.1, 0.05, 0.05])
        result = _sparsify(v)
        assert result[0] > 0
        # 0.1 < 0.8/4 = 0.2, so should be zeroed
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_all_zero(self):
        """All-zero input should return uniform weights."""
        v = np.zeros(5)
        result = _sparsify(v)
        np.testing.assert_allclose(result, np.ones(5) / 5)

    def test_single_nonzero(self):
        """Single nonzero element -> that element becomes 1.0."""
        v = np.array([0.0, 0.0, 0.5, 0.0])
        result = _sparsify(v)
        assert result[2] == 1.0
        assert np.sum(result) == 1.0

    def test_equal_weights(self):
        """Equal weights: all equal max, so max/4 threshold keeps them all."""
        v = np.array([0.25, 0.25, 0.25, 0.25])
        result = _sparsify(v)
        # 0.25 > 0.25/4 = 0.0625, so all kept
        np.testing.assert_allclose(result, v, atol=1e-10)


class TestSumNormalize:
    """Verify _sum_normalize helper."""

    def test_basic(self):
        v = np.array([2.0, 3.0, 5.0])
        result = _sum_normalize(v)
        np.testing.assert_allclose(result, [0.2, 0.3, 0.5])

    def test_zero_sum(self):
        """Zero-sum vector -> uniform weights."""
        v = np.array([0.0, 0.0, 0.0])
        result = _sum_normalize(v)
        np.testing.assert_allclose(result, [1.0 / 3, 1.0 / 3, 1.0 / 3])


# =============================================================================
# Phase C/D: Unit and Time Weights
# =============================================================================


class TestUnitWeights:
    """Verify compute_sdid_unit_weights behavior."""

    def test_simplex_constraint(self):
        """Weights should be on the simplex (sum to 1, non-negative)."""
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((8, 15))
        Y_pre_treated_mean = rng.standard_normal(8)

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert np.all(omega >= -1e-10)
        assert abs(np.sum(omega) - 1.0) < 1e-6

    def test_single_control(self):
        """Single control unit -> weight = [1.0]."""
        Y_pre = np.array([[1.0], [2.0], [3.0]])
        Y_pre_treated_mean = np.array([1.5, 2.5, 3.5])

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert len(omega) == 1
        assert abs(omega[0] - 1.0) < 1e-10

    def test_empty_control(self):
        """No control units -> empty array."""
        Y_pre = np.zeros((5, 0))
        Y_pre_treated_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert len(omega) == 0

    def test_sparsification_occurs(self):
        """With enough controls, some weights should be zeroed by sparsification."""
        rng = np.random.default_rng(42)
        # Many controls with varied patterns — expect some to get zeroed
        Y_pre = rng.standard_normal((10, 50))
        Y_pre_treated_mean = rng.standard_normal(10)

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=0.5)

        # At least some weights should be exactly zero after sparsification
        assert np.sum(omega == 0) > 0


class TestTimeWeights:
    """Verify compute_time_weights behavior."""

    def test_simplex_constraint(self):
        """Weights should be on the simplex."""
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((8, 15))
        Y_post = rng.standard_normal((3, 15))

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-6

    def test_single_pre_period(self):
        """Single pre-period -> weight = [1.0]."""
        Y_pre = np.array([[1.0, 2.0, 3.0]])
        Y_post = np.array([[4.0, 5.0, 6.0]])

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        assert len(lam) == 1
        assert abs(lam[0] - 1.0) < 1e-10

    def test_collapsed_form_correctness(self):
        """Verify the collapsed form matrix is built correctly.

        The time weight optimization solves on (N_co, T_pre+1) where the
        last column is the per-control post-period mean.
        """
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((4, 3))  # 4 pre-periods, 3 controls
        Y_post = rng.standard_normal((2, 3))  # 2 post-periods, 3 controls

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        # Should have 4 weights (one per pre-period)
        assert len(lam) == 4
        assert abs(np.sum(lam) - 1.0) < 1e-6


# =============================================================================
# Full Pipeline
# =============================================================================


class TestATTFullPipeline:
    """Test full SDID estimation pipeline."""

    def test_estimation_produces_reasonable_att(self, ci_params):
        """Full estimation on canonical data should produce reasonable ATT."""
        df = _make_panel(n_control=20, n_treated=3, n_pre=6, n_post=3, att=5.0, seed=42)
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42, variance_method="placebo")
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(6, 9)),
        )

        # ATT should be positive and reasonably close to true value
        assert results.att > 0
        assert abs(results.att - 5.0) < 3.0

    def test_results_have_regularization_info(self):
        """Results should include noise_level, zeta_omega, zeta_lambda."""
        df = _make_panel(seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.noise_level is not None
        assert results.noise_level >= 0
        assert results.zeta_omega is not None
        assert results.zeta_omega >= 0
        assert results.zeta_lambda is not None
        assert results.zeta_lambda >= 0

    def test_user_override_regularization(self):
        """User-specified zeta_omega/zeta_lambda should be used instead of auto."""
        df = _make_panel(seed=42)
        sdid = SyntheticDiD(zeta_omega=99.0, zeta_lambda=0.5, variance_method="placebo", seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.zeta_omega == 99.0
        assert results.zeta_lambda == 0.5


# =============================================================================
# Placebo SE
# =============================================================================


class TestPlaceboSE:
    """Verify placebo variance formula."""

    def test_placebo_se_formula(self):
        """SE should be sqrt((r-1)/r) * sd(estimates, ddof=1)."""
        df = _make_panel(n_control=15, n_treated=2, seed=42)
        sdid = SyntheticDiD(variance_method="placebo", n_bootstrap=100, seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.se > 0
        assert results.variance_method == "placebo"

        # Verify the formula: se = sqrt((r-1)/r) * sd(placebo_estimates)
        if results.variance_effects is not None:
            r = len(results.variance_effects)
            expected_se = np.sqrt((r - 1) / r) * np.std(results.variance_effects, ddof=1)
            assert abs(results.se - expected_se) < 1e-10


# =============================================================================
# Bootstrap SE
# =============================================================================


class TestBootstrapSE:
    """Verify the paper-faithful refit pairs bootstrap.

    ``variance_method="bootstrap"`` re-estimates ω̂_b and λ̂_b via two-pass
    sparsified Frank-Wolfe on each bootstrap draw (Arkhangelsky et al. 2021
    Algorithm 2 step 2, and R's default ``synthdid::vcov(method="bootstrap")``).
    Survey designs are rejected upstream in ``fit()``.
    """

    def test_bootstrap_se_positive(self, ci_params):
        """Bootstrap SE should be positive."""
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(variance_method="bootstrap", n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.se > 0
        assert results.variance_method == "bootstrap"
        assert results.n_bootstrap == n_boot

    def test_bootstrap_with_zeta_overrides(self, ci_params):
        """Bootstrap SE should work with user-specified zeta overrides."""
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(
            variance_method="bootstrap",
            n_bootstrap=n_boot,
            zeta_omega=99.0,
            zeta_lambda=0.5,
            seed=42,
        )
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.zeta_omega == 99.0
        assert results.zeta_lambda == 0.5
        assert results.variance_method == "bootstrap"
        assert results.se > 0

    def test_bootstrap_se_tracks_placebo_se_exchangeable(self, ci_params):
        """Bootstrap SE tracks placebo SE under control-pool exchangeability.

        Placebo (Algorithm 4) already re-estimates ω and λ per permutation,
        so under exchangeability of the control pool it should produce a
        similar variance to the paper-faithful refit bootstrap. Divergence
        flags either a refit implementation bug or a genuine exchangeability
        violation in the DGP.

        Skipped under pure-Python mode: ``ci_params.bootstrap(min_n=...)``
        caps ``min_n`` at 49 to keep pure-Python CI fast (see
        ``tests/conftest.py:210``), but the 0.40 tolerance is calibrated
        for B∈[100, 200] — at B=49 MC noise on the bootstrap SE can push
        rel-diff beyond 0.40 without any correctness issue (B=100/200
        runs converge to rel-diff ≈ 0.27 on the same seed). The 15
        Rust-backed matrix jobs (macOS/Linux x86/Linux ARM/Windows × 3
        Python versions) exercise the regression guard at the designed
        B=200, so the contract is still covered for the default user
        install path.
        """
        from diff_diff import utils as dd_utils

        if not dd_utils.HAS_RUST_BACKEND:
            pytest.skip(
                "Pure-Python mode caps ci_params.bootstrap min_n at 49, "
                "but the 0.40 tolerance requires B≥100. Rust-backend CI "
                "jobs exercise this regression guard at B=200."
            )
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        n_boot = ci_params.bootstrap(200, min_n=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_placebo = SyntheticDiD(variance_method="placebo", n_bootstrap=n_boot, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
            )
            r_boot = SyntheticDiD(variance_method="bootstrap", n_bootstrap=n_boot, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
            )
        rel_diff = abs(r_boot.se - r_placebo.se) / r_placebo.se
        # Tolerance chosen for B=100–200 with MC noise floor ~7–10% on the
        # SE ratio; 0.40 leaves headroom without hiding order-of-magnitude
        # regressions.
        assert rel_diff < 0.40, (
            f"bootstrap SE {r_boot.se:.6f} does not track placebo SE "
            f"{r_placebo.se:.6f} on exchangeable DGP (rel diff {rel_diff:.4f})"
        )

    def test_bootstrap_succeeds_on_pweight_survey(self):
        """Survey + bootstrap succeeds (pweight-only) (PR #352).

        The weighted-FW path treats per-control survey weights as
        constant per draw (no Rao-Wu rescaling, since no PSU). ATT is
        finite, SE is positive, variance_method is preserved.
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=20, n_treated=3, seed=42)
        df["wt"] = 1.0
        result = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=1).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
            survey_design=SurveyDesign(weights="wt"),
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert result.variance_method == "bootstrap"

    def test_bootstrap_succeeds_on_full_design_survey(self):
        """Survey + bootstrap with strata/PSU succeeds via Rao-Wu (PR #352).

        Composes per-draw Rao-Wu rescaled weights with the weighted-FW
        helpers (compute_sdid_unit_weights_survey /
        compute_time_weights_survey). Asserts finite SE and survey_metadata
        records the strata/PSU layout.
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=20, n_treated=3, seed=42)
        df["wt"] = 1.0
        df["stratum"] = df["unit"] % 2
        # Globally unique PSU labels (avoids needing nest=True). Each unit
        # gets its own PSU id; stratum partitions the PSUs into two groups.
        df["psu"] = df["unit"]
        result = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=1).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
            survey_design=SurveyDesign(weights="wt", strata="stratum", psu="psu"),
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert result.variance_method == "bootstrap"
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_strata is not None
        assert result.survey_metadata.n_psu is not None

    def test_bootstrap_summary_shows_replications(self, ci_params):
        """result.summary() shows "Bootstrap replications" line for bootstrap.

        Cross-surface guard: the result-class gating at ``results.py:960``
        must keep ``"bootstrap"`` in its allow-list so the replications row
        renders for bootstrap fits.
        """
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        n_boot = ci_params.bootstrap(50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = SyntheticDiD(variance_method="bootstrap", n_bootstrap=n_boot, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
            )
        summary = r.summary()
        assert "Bootstrap replications" in summary
        assert str(n_boot) in summary

    def test_bootstrap_pweight_only_retries_zero_treated_mass_draws(self):
        """Pweight-only bootstrap: zero-mass treated draws must be retried,
        not silently fall back to an unweighted treated mean (PR #355 R2 P0).

        Regression: prior to R2, when a bootstrap draw's treated units all
        had survey weight 0 — reachable when at least one treated unit has
        pweight 0 AND the resample picks only those units — the code fell
        through to ``np.mean(Y_boot_pre_t, axis=1)``. That silently
        dropped survey weighting on that draw while the fit-time ATT uses
        the survey-weighted treated mean, corrupting the bootstrap
        distribution used for SE/p-value/CI.

        Fix: extend the degenerate-retry to any survey branch where
        ``rw_treated_draw.sum() == 0``. This test constructs a panel
        where one of three treated units has weight 0; bootstrap must
        still produce a valid finite SE, and the draws that hit the
        zero-mass condition are retried rather than getting a wrong τ.
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=25, n_treated=3, seed=42)
        # Give one treated unit weight 0 and the other two weight 1.
        # Control units get unit weights (any positive). Treated_idx 25,
        # 26, 27; set 25 → weight 0, 26 and 27 → weight 1.
        df["wt"] = 1.0
        df.loc[df["unit"] == 25, "wt"] = 0.0
        result = SyntheticDiD(variance_method="bootstrap", n_bootstrap=100, seed=1).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
            survey_design=SurveyDesign(weights="wt"),
        )
        assert np.isfinite(result.att), f"att not finite: {result.att}"
        assert np.isfinite(result.se), f"se not finite: {result.se}"
        assert result.se > 0, f"se={result.se} must be positive"
        # Cross-surface: the variance_method label should still be
        # bootstrap and the bootstrap replications line should render.
        assert result.variance_method == "bootstrap"

    def test_bootstrap_full_design_without_explicit_weights(self):
        """SurveyDesign(strata=..., psu=..., weights=None) fits successfully.

        Regression for PR #355 R1 code-quality finding: `SurveyDesign` allows
        `weights=None` (resolve() synthesizes unit weights of 1), but the
        SDID helper `_extract_unit_survey_weights` used to index
        `survey_design.weights` directly and would fail before bootstrap
        could run. The helper now returns ones for this configuration.
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=20, n_treated=3, seed=42)
        df["stratum"] = df["unit"] % 2
        df["psu"] = df["unit"]
        result = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=1).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
            survey_design=SurveyDesign(strata="stratum", psu="psu"),  # weights=None
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert result.variance_method == "bootstrap"
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_strata is not None
        assert result.survey_metadata.n_psu is not None

    def test_bootstrap_full_design_rao_wu_boot_idx_slice(self, monkeypatch):
        """Full-design bootstrap slices Rao-Wu weights by ``boot_idx``.

        Documented in REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap
        composition)``: the hybrid path first performs unit-level pairs-
        bootstrap (``boot_idx = rng.choice(n_total)``) and THEN slices
        the Rao-Wu rescaled weights over the resampled units. Monkeypatch
        ``generate_rao_wu_weights`` to return a known vector and capture
        the ``rw_control`` fed into ``compute_sdid_unit_weights_survey``;
        assert the captured vector matches the expected slice
        ``known_rw[:n_control][boot_idx[boot_is_control]]``.

        Regression against a subtle class of bug where either the slice
        index arithmetic or the Rao-Wu call site could drift (e.g.,
        someone refactors ``resolved_survey_unit`` indexing to skip the
        boot_idx slicing, or the rw-then-slice order gets swapped to
        slice-then-rw). Both would silently produce wrong bootstrap SE.
        """
        from diff_diff import synthetic_did as sdid_mod
        from diff_diff import utils as dd_utils
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=15, n_treated=3, seed=42)
        df["wt"] = 1.0
        df["stratum"] = df["unit"] % 2
        df["psu"] = df["unit"]

        # Known Rao-Wu weight vector. Length = n_total = 18; distinct
        # values per unit so a slice of the first n_control=15 positions
        # by boot_idx[boot_is_control] is identifiable.
        n_total = 18
        known_rw = np.arange(1, n_total + 1, dtype=np.float64)

        def fake_rao_wu(resolved_survey, rng):
            return known_rw.copy()

        monkeypatch.setattr(sdid_mod, "generate_rao_wu_weights", fake_rao_wu)

        captured: list = []

        real_helper = dd_utils.compute_sdid_unit_weights_survey

        def capturing_helper(Y_pre_c, Y_pre_t_mean, rw, *args, **kwargs):
            captured.append(np.array(rw, copy=True))
            return real_helper(Y_pre_c, Y_pre_t_mean, rw, *args, **kwargs)

        monkeypatch.setattr(sdid_mod, "compute_sdid_unit_weights_survey", capturing_helper)

        bootstrap_seed = 1
        SyntheticDiD(
            variance_method="bootstrap",
            n_bootstrap=10,
            seed=bootstrap_seed,
        ).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
            survey_design=SurveyDesign(weights="wt", strata="stratum", psu="psu"),
        )

        # Exact-equality check against a reproduced RNG stream (PR #355 R4
        # P3). The captured rw vectors must match known_rw[:n_control]
        # sliced by boot_idx[boot_is_control] value-for-value. Reproducing
        # the bootstrap's rng externally works because:
        #  - fake_rao_wu does NOT consume rng (just returns known_rw),
        #    so the only per-draw rng advance is ``rng.choice(n_total, ...)``
        #    which yields boot_idx;
        #  - known_rw is strictly positive, so the zero-mass retry branch
        #    (synthetic_did.py ``_bootstrap_se``) never fires;
        #  - a 15/3 split makes the no-control and all-control retries
        #    vanishingly rare.
        # An exact-equality regression catches the sibling bugs the old
        # range check missed: permuted indices, deduplicated boot_idx, or
        # substituted ``resolved_survey_unit.weights`` lookup in place of
        # the known_rw slice — any of which would silently change
        # bootstrap SE.
        n_control = 15
        rng_sim = np.random.default_rng(bootstrap_seed)
        expected_slices = []
        while len(expected_slices) < len(captured):
            boot_idx = rng_sim.choice(n_total, size=n_total, replace=True)
            boot_is_control = boot_idx < n_control
            n_co_b = int(boot_is_control.sum())
            if n_co_b == 0 or n_co_b == n_total:
                continue
            expected_slices.append(known_rw[:n_control][boot_idx[boot_is_control]])

        assert len(captured) >= 1, "no FW calls captured — survey dispatch broken"
        for i, (rw_captured, rw_expected) in enumerate(zip(captured, expected_slices)):
            np.testing.assert_array_equal(
                rw_captured,
                rw_expected,
                err_msg=(
                    f"draw {i}: captured rw_control differs from expected "
                    f"known_rw[:n_control][boot_idx[boot_is_control]]. "
                    "Regression in hybrid pairs-bootstrap + Rao-Wu "
                    "slice ordering."
                ),
            )

    def test_fit_raises_on_zero_total_treated_survey_mass(self):
        """Fit-time positive-mass guard: zero treated survey mass raises.

        ``SurveyDesign.resolve()`` accepts non-negative unit weights
        (``survey.py`` L171-L176), so a user can legitimately assign unit
        survey weights of 0 to every treated unit — encoding an
        unidentified target population. Without the front-door guard, the
        fit-time survey-weighted ATT (``np.average(Y, weights=w_treated)``)
        would hit ``0/0`` and silently propagate NaN into the bootstrap
        loop, defeating the per-draw zero-mass retry (PR #355 R2 P0).
        Regression against PR #355 R7 P1: the guard must fire before the
        bootstrap is even dispatched.
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=10, n_treated=3, seed=42)
        # Every treated unit gets weight 0; controls keep positive weight.
        df["wt"] = np.where(df["treated"] == 1, 0.0, 1.0)
        with pytest.raises(ValueError, match=r"treated arm has zero total mass"):
            SyntheticDiD(variance_method="bootstrap", n_bootstrap=20, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
                survey_design=SurveyDesign(weights="wt"),
            )

    def test_fit_raises_on_zero_total_control_survey_mass(self):
        """Fit-time positive-mass guard: zero control survey mass raises.

        Mirror of the treated-arm case (PR #355 R7 P1). Downstream
        ``omega_eff = unit_weights * w_control / (unit_weights * w_control).sum()``
        would hit 0/0; the guard front-doors.
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=10, n_treated=3, seed=42)
        df["wt"] = np.where(df["treated"] == 0, 0.0, 1.0)
        with pytest.raises(ValueError, match=r"control arm has zero total mass"):
            SyntheticDiD(variance_method="bootstrap", n_bootstrap=20, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
                survey_design=SurveyDesign(weights="wt"),
            )

    def test_fit_raises_on_zero_treated_mass_under_full_design(self):
        """Fit-time positive-mass guard fires under full strata/PSU/FPC too.

        The guard sources w_control / w_treated from the **resolved
        unit-level** design (PR #355 R4 P0), so zero total treated mass
        under a strata/PSU/FPC configuration must fire the same front-door
        ValueError as the pweight-only case (PR #355 R7 P1).
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=10, n_treated=3, seed=42)
        df["wt"] = np.where(df["treated"] == 1, 0.0, 1.0)
        df["stratum"] = df["unit"] % 2
        df["psu"] = df["unit"]
        with pytest.raises(ValueError, match=r"treated arm has zero total mass"):
            SyntheticDiD(variance_method="bootstrap", n_bootstrap=20, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
                survey_design=SurveyDesign(weights="wt", strata="stratum", psu="psu"),
            )

    def test_fit_raises_on_zero_effective_control_support(self, monkeypatch):
        """Fit-time guard: FW mass concentrated on zero-weight controls raises.

        Zero survey weights are valid input (``survey.py`` L171-L176
        rejects only negatives), and ``compute_sdid_unit_weights``
        sparsifies ``unit_weights`` to exact zeros by design. If FW
        concentrates on a control whose survey weight is 0, the composed
        ``omega_eff = unit_weights * w_control`` has zero total mass and
        the subsequent normalization hits ``0/0``, silently propagating
        NaN into the ATT / SE / CI. The R7 P1 guard
        (``w_control.sum() > 0``) is not sufficient here because at least
        one control has positive weight — the degeneracy is in the
        composed vector, not the raw survey mass.

        Regression against PR #355 R12 P1: the fit-time guard must raise
        a targeted ``ValueError`` before the normalization step.

        We monkeypatch ``compute_sdid_unit_weights`` to return a
        canonical sparse unit-weight vector concentrated on the first
        control, bypassing FW convergence dynamics. This makes the test
        about the guard contract (behavior when the degenerate
        configuration is reached), not about how easy it is to force FW
        into that configuration on synthetic data.
        """
        from diff_diff import synthetic_did as sdid_mod
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=5, n_treated=2, seed=42)
        # First control gets survey weight 0; others get weight 1.
        unique_units = np.sort(df["unit"].unique())
        treated_ids = set(df.loc[df["treated"] == 1, "unit"].unique())
        control_ids = [u for u in unique_units if u not in treated_ids]
        zero_weight_unit = control_ids[0]
        df["wt"] = np.where(df["unit"] == zero_weight_unit, 0.0, 1.0).astype(float)

        # Force unit_weights to concentrate on control_ids[0]. The call
        # sites in fit() use _create_outcome_matrices with control_units
        # in sorted order (same order _make_panel produces), so index 0
        # of the unit-weights vector corresponds to control_ids[0].
        def sparse_unit_weights(Y_pre_control, *args, **kwargs):
            n_ctrl = Y_pre_control.shape[1]
            w = np.zeros(n_ctrl)
            w[0] = 1.0
            return w

        monkeypatch.setattr(sdid_mod, "compute_sdid_unit_weights", sparse_unit_weights)

        with pytest.raises(ValueError, match=r"unidentified|zero survey weight"):
            SyntheticDiD(variance_method="placebo", n_bootstrap=20, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
                survey_design=SurveyDesign(weights="wt"),
            )

    def test_fit_raises_on_zero_effective_control_support_full_design(self, monkeypatch):
        """Fit-time effective-control guard fires under strata/PSU/FPC too.

        Mirror of the pweight-only zero-effective-control case (PR #355
        R12 P1) under a full survey design. The guard is part of the
        ``w_control is not None`` branch, which fires for both
        pweight-only and strata/PSU/FPC fits.
        """
        from diff_diff import synthetic_did as sdid_mod
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=5, n_treated=2, seed=42)
        unique_units = np.sort(df["unit"].unique())
        treated_ids = set(df.loc[df["treated"] == 1, "unit"].unique())
        control_ids = [u for u in unique_units if u not in treated_ids]
        zero_weight_unit = control_ids[0]
        df["wt"] = np.where(df["unit"] == zero_weight_unit, 0.0, 1.0).astype(float)
        df["stratum"] = df["unit"] % 2
        df["psu"] = df["unit"]

        def sparse_unit_weights(Y_pre_control, *args, **kwargs):
            n_ctrl = Y_pre_control.shape[1]
            w = np.zeros(n_ctrl)
            w[0] = 1.0
            return w

        monkeypatch.setattr(sdid_mod, "compute_sdid_unit_weights", sparse_unit_weights)

        with pytest.raises(ValueError, match=r"unidentified|zero survey weight"):
            SyntheticDiD(variance_method="bootstrap", n_bootstrap=20, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
                survey_design=SurveyDesign(weights="wt", strata="stratum", psu="psu"),
            )

    def test_fit_raises_on_implicit_psu_fpc_below_unit_count_unstratified(self):
        """Fit-time FPC validation fires when psu=None and FPC < n_units.

        When ``SurveyDesign(fpc=...)`` is declared without an explicit
        ``psu=``, SDID Rao-Wu (via ``bootstrap_utils.generate_rao_wu_weights``
        L654-L655) treats each unit as its own PSU. The helper rejects
        ``FPC < n_PSU`` mid-draw (``bootstrap_utils.py`` L684-L688); without
        a front-door guard, every bootstrap draw raises, ``_bootstrap_se``
        swallows the ``ValueError`` in its retry loop, and the user sees a
        generic bootstrap-exhaustion error. Regression against PR #355 R8
        P1: ``fit()`` must validate FPC vs unit count before dispatching
        the bootstrap.
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=10, n_treated=3, seed=42)
        df["wt"] = 1.0
        # 13 units total; FPC says population size is 5 — infeasible for
        # 13 implicit PSUs.
        df["fpc_pop"] = 5.0
        with pytest.raises(ValueError, match=r"FPC.*less than the number of units"):
            SyntheticDiD(variance_method="bootstrap", n_bootstrap=20, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
                survey_design=SurveyDesign(weights="wt", fpc="fpc_pop"),
            )

    def test_fit_raises_on_implicit_psu_fpc_below_stratum_unit_count(self):
        """Fit-time FPC validation fires per stratum under implicit PSU.

        Mirror of the unstratified case but with strata present. Each
        stratum's FPC must be >= its unit count (PR #355 R8 P1).
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=12, n_treated=4, seed=42)
        df["wt"] = 1.0
        df["stratum"] = df["unit"] % 2  # 2 strata, ~8 units each
        # FPC says 3 per stratum — infeasible for 8 implicit PSUs/stratum.
        df["fpc_pop"] = 3.0
        with pytest.raises(ValueError, match=r"FPC.*less than the number of units in that stratum"):
            SyntheticDiD(variance_method="bootstrap", n_bootstrap=20, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
                survey_design=SurveyDesign(
                    weights="wt",
                    strata="stratum",
                    fpc="fpc_pop",
                ),
            )

    def test_bootstrap_scale_invariance_under_pweight_rescaling(self):
        """Survey-bootstrap SE / p / CI are invariant to a global pweight rescaling.

        ``SurveyDesign.resolve()`` normalizes pweights/aweights to mean=1
        (survey.py L189-L203), which is the library's scale-invariance
        contract for survey-weighted fits. This test fits the same SDID
        panel under two SurveyDesigns — weights column ``"wt"`` vs a
        10x-rescaled copy ``"wt_scaled"`` — and asserts bootstrap SE,
        p-value, and CI agree to machine-epsilon tolerance.

        Regression against PR #355 R4 P0: the initial PR #352 pweight-only
        bootstrap branch bypassed the resolved (normalized) unit-level
        weights and fed raw panel-column weights into the weighted-FW
        objective. That objective is NOT invariant to a global rescale
        of rw — the loss term scales as rw^2 (``A-tilde = A * diag(rw)``)
        while the reg term scales as rw (``zeta^2 * sum rw * omega^2``) —
        so any user who rescaled their pweight column (e.g. switched
        units) would see silently different SEs. The fix
        (synthetic_did.py ``fit()`` around the ``resolved_survey`` block)
        sources ``w_control`` and ``w_treated`` from
        ``resolved_survey_unit.weights`` (post-normalization) rather
        than re-extracting raw weights via ``_extract_unit_survey_weights``.
        Tolerance is machine-epsilon tight because floating-point multiply-
        reduce ordering inside ``raw * (n / (c*raw_sum))`` vs
        ``raw * (n / raw_sum)`` can drift by ~1 ULP; a raw-weight fallback
        would produce differences on the order of 1 or larger.
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=12, n_treated=3, seed=42)
        unique_units = np.sort(df["unit"].unique())
        unit_weights = np.linspace(0.5, 2.5, len(unique_units))
        wt_map = dict(zip(unique_units, unit_weights))
        df["wt"] = df["unit"].map(wt_map)
        df["wt_scaled"] = df["wt"] * 10.0

        kwargs = dict(
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
        )
        result_base = SyntheticDiD(
            variance_method="bootstrap",
            n_bootstrap=50,
            seed=1,
        ).fit(df, survey_design=SurveyDesign(weights="wt"), **kwargs)
        result_scaled = SyntheticDiD(
            variance_method="bootstrap",
            n_bootstrap=50,
            seed=1,
        ).fit(df, survey_design=SurveyDesign(weights="wt_scaled"), **kwargs)

        assert np.isfinite(result_base.se) and result_base.se > 0
        np.testing.assert_allclose(
            result_scaled.se,
            result_base.se,
            rtol=1e-13,
            atol=0,
            err_msg="bootstrap SE is not invariant to pweight global rescaling",
        )
        np.testing.assert_allclose(
            result_scaled.p_value,
            result_base.p_value,
            rtol=1e-12,
            atol=1e-14,
            err_msg="bootstrap p-value is not invariant to pweight global rescaling",
        )
        np.testing.assert_allclose(
            result_scaled.conf_int,
            result_base.conf_int,
            rtol=1e-13,
            atol=0,
            err_msg="bootstrap CI is not invariant to pweight global rescaling",
        )

    def test_bootstrap_single_psu_returns_nan(self):
        """Unstratified single-PSU survey design returns NaN SE (PR #352).

        Resampling one PSU yields the same subset every draw, so the
        bootstrap distribution is degenerate and the SE is unidentified.
        ``_bootstrap_se`` short-circuits to (NaN, []) for this case rather
        than emitting a misleading SE estimate. Recovered from the pre-PR
        #351 fixed-weight Rao-Wu branch (commit 91082e5).
        """
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=20, n_treated=3, seed=42)
        df["wt"] = 1.0
        df["psu_single"] = 0  # all units in the same PSU; no strata
        sdid = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=1)
        result = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
            survey_design=SurveyDesign(weights="wt", psu="psu_single"),
        )
        assert np.isnan(
            result.se
        ), f"single-PSU unstratified bootstrap should return NaN SE, got {result.se}"

    def test_bootstrap_fw_nonconvergence_warning_fires_under_rust(self, monkeypatch):
        """Aggregate FW non-convergence warning surfaces on the Rust backend.

        Regression against the silent-failure mode previously masked by the
        ``warnings.catch_warnings`` tally: the Rust FW solver is silent on
        ``max_iter`` exhaustion, so a purely warnings-based per-draw count
        read zero even when the Rust solver did not converge. After wiring
        ``return_convergence=True`` through
        ``compute_sdid_unit_weights`` / ``compute_time_weights`` via the new
        ``sc_weight_fw_with_convergence`` Rust entry point, the helpers
        thread an explicit bool per pass and the aggregate warning above 5%
        of valid draws fires.

        We force every Rust FW call to report ``converged=False`` by monkey-
        patching the helper; under the fixed wiring, the aggregate warning
        must fire. Prior to the fix, this same monkeypatch would have had
        no effect on the tally.
        """
        from diff_diff import utils as dd_utils

        if not dd_utils.HAS_RUST_BACKEND:
            pytest.skip("Test targets the Rust backend specifically.")

        real_rust = dd_utils._rust_sc_weight_fw_with_convergence

        def _always_not_converged(Y, zeta, intercept, init, min_decrease, max_iter):
            weights, _ = real_rust(Y, zeta, intercept, init, min_decrease, max_iter)
            return weights, False

        monkeypatch.setattr(dd_utils, "_rust_sc_weight_fw_with_convergence", _always_not_converged)

        df = _make_panel(n_control=20, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sdid.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=list(range(5, 8)),
            )

        fw_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "Frank-Wolfe did not converge" in str(w.message)
        ]
        assert len(fw_warnings) >= 1, (
            "Expected the aggregate Frank-Wolfe non-convergence warning "
            "to fire when every FW call reports converged=False under "
            "the Rust backend. Got no such warning — the Rust FW path is "
            "silent on non-convergence and the bootstrap loop is not "
            "surfacing it."
        )


# =============================================================================
# Jackknife SE
# =============================================================================


class TestJackknifeSE:
    """Verify jackknife SE with fixed weights (Algorithm 3)."""

    def test_jackknife_se_positive(self):
        """Jackknife SE should be positive for well-specified data."""
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        assert results.se > 0
        assert results.variance_method == "jackknife"

    def test_jackknife_deterministic(self):
        """Jackknife should produce identical results regardless of seed."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        results1 = SyntheticDiD(variance_method="jackknife", seed=1).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        results2 = SyntheticDiD(variance_method="jackknife", seed=999).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        # ATT should be identical (same weights, no randomness in point est)
        assert results1.att == results2.att
        # SE should be identical (jackknife is deterministic)
        assert results1.se == results2.se

    def test_jackknife_se_formula(self):
        """Verify SE matches sqrt((n-1)/n * sum((u - ubar)^2))."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        assert results.variance_effects is not None
        u = results.variance_effects
        n = len(u)
        u_bar = np.mean(u)
        expected_se = np.sqrt((n - 1) / n * np.sum((u - u_bar) ** 2))
        assert abs(results.se - expected_se) < 1e-10

    def test_jackknife_n_iterations(self):
        """Number of jackknife estimates = n_control + n_treated."""
        n_co, n_tr = 15, 3
        df = _make_panel(n_control=n_co, n_treated=n_tr, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        assert results.variance_effects is not None
        assert len(results.variance_effects) == n_co + n_tr

    def test_jackknife_single_treated_nan(self):
        """Single treated unit -> NaN SE (matches R's NA)."""
        df = _make_panel(n_control=15, n_treated=1, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdid = SyntheticDiD(variance_method="jackknife", seed=42)
            results = sdid.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=list(range(5, 7)),
            )
        assert np.isnan(results.se)
        assert np.isnan(results.t_stat)
        assert np.isnan(results.p_value)

    def test_jackknife_analytical_pvalue(self):
        """Jackknife should use analytical p-value, not empirical."""
        from scipy.stats import norm

        df = _make_panel(n_control=20, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        if np.isfinite(results.t_stat):
            expected_p = 2 * (1 - norm.cdf(abs(results.t_stat)))
            assert abs(results.p_value - expected_p) < 1e-10

    def test_jackknife_same_att_as_placebo(self):
        """Jackknife should produce the same point estimate as placebo."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        res_jk = SyntheticDiD(variance_method="jackknife", seed=42).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        res_pl = SyntheticDiD(variance_method="placebo", seed=42, n_bootstrap=50).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        assert abs(res_jk.att - res_pl.att) < 1e-10

    def test_jackknife_n_bootstrap_ignored(self):
        """n_bootstrap=1 should not raise for jackknife (it's ignored)."""
        sdid = SyntheticDiD(variance_method="jackknife", n_bootstrap=1)
        assert sdid.n_bootstrap == 1
        assert sdid.variance_method == "jackknife"

    def test_jackknife_n_bootstrap_none_in_results(self):
        """Results should have n_bootstrap=None for jackknife."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        assert results.n_bootstrap is None

    def test_jackknife_with_pweights(self):
        """Jackknife should produce finite SE with survey pweights."""
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=15, n_treated=3, seed=42)
        # Add unit-constant survey weights
        unit_weights = {u: 1.0 + u * 0.1 for u in df["unit"].unique()}
        df["weight"] = df["unit"].map(unit_weights)

        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
            survey_design=SurveyDesign(weights="weight"),
        )
        assert results.se > 0
        assert np.isfinite(results.se)
        assert results.variance_method == "jackknife"

    def test_jackknife_zero_effective_control_nan(self):
        """Zero-weight controls after composition -> NaN SE."""
        from diff_diff.survey import SurveyDesign

        # 3 controls, 2 treated. Set all but 1 control survey weight to 0
        # so effective support <= 1.
        df = _make_panel(n_control=3, n_treated=2, seed=42)
        weights = {}
        control_units = sorted(df.loc[df["treated"] == 0, "unit"].unique())
        treated_units = sorted(df.loc[df["treated"] == 1, "unit"].unique())
        # Only first control gets positive weight
        for i, u in enumerate(control_units):
            weights[u] = 1.0 if i == 0 else 0.0
        for u in treated_units:
            weights[u] = 1.0
        df["weight"] = df["unit"].map(weights)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdid = SyntheticDiD(variance_method="jackknife", seed=42)
            results = sdid.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=list(range(5, 7)),
                survey_design=SurveyDesign(weights="weight"),
            )
        assert np.isnan(results.se)

    def test_jackknife_zero_treated_weight_nan(self):
        """Single positive-weight treated unit with survey -> NaN SE."""
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=10, n_treated=2, seed=42)
        weights = {}
        treated_units = sorted(df.loc[df["treated"] == 1, "unit"].unique())
        control_units = sorted(df.loc[df["treated"] == 0, "unit"].unique())
        for u in control_units:
            weights[u] = 1.0
        # Only first treated unit gets positive weight
        weights[treated_units[0]] = 1.0
        weights[treated_units[1]] = 0.0
        df["weight"] = df["unit"].map(weights)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdid = SyntheticDiD(variance_method="jackknife", seed=42)
            results = sdid.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=list(range(5, 7)),
                survey_design=SurveyDesign(weights="weight"),
            )
        assert np.isnan(results.se)


# =============================================================================
# Jackknife SE - R Golden Value Parity
# =============================================================================


class TestJackknifeSERParity:
    """Verify jackknife SE matches R's synthdid::vcov(method='jackknife').

    Golden values generated with R 4.5.2, synthdid package:

        library(synthdid)
        set.seed(42)
        N0 <- 20; N1 <- 3; T0 <- 5; T1 <- 3
        N <- N0 + N1; T <- T0 + T1
        Y <- matrix(0, nrow=N, ncol=T)
        for (i in 1:N) {
          unit_fe <- rnorm(1, sd=2)
          for (t in 1:T) {
            Y[i,t] <- 10 + unit_fe + (t-1)*0.3 + rnorm(1, sd=0.5)
            if (i > N0 && t > T0) Y[i,t] <- Y[i,t] + 5.0
          }
        }
        tau_hat <- synthdid_estimate(Y, N0, T0)
        se_jk <- sqrt(vcov(tau_hat, method="jackknife")[1,1])
    """

    # R's Y matrix (23 units x 8 periods), row-major
    Y_FLAT = [
        12.459567808595292,
        13.223481099962006,
        13.658348196773856,
        13.844051055863837,
        13.888854636247594,
        14.997677893012806,
        14.494587375086788,
        15.851128751231856,
        10.527006629006900,
        11.317894498245712,
        9.780141451338988,
        10.635177418486473,
        11.007911133698329,
        11.692547000930196,
        11.532445341187122,
        10.646344091442769,
        5.779122815714058,
        5.265746845809725,
        4.828411925858962,
        5.933107464969151,
        6.926403492435262,
        7.566662873481445,
        6.703831577045862,
        7.090431451464497,
        6.703722507026075,
        6.453676391630379,
        7.301398891231049,
        7.726092498224848,
        8.191225590595401,
        7.669210641906834,
        8.526151391259425,
        7.715169490073769,
        8.005628186152748,
        7.523978158267692,
        9.049143286687135,
        9.434081283341134,
        9.450553333966674,
        10.310163601090766,
        9.867729569702721,
        9.846941461031360,
        10.459939463684098,
        11.887686682638062,
        11.249912950470762,
        12.093459993478538,
        12.226598684379407,
        11.973716581337246,
        13.453499811673423,
        13.287085704636093,
        10.317796666844943,
        10.819165701226847,
        10.824437736488752,
        9.582976251622744,
        11.521962769964540,
        11.495903971828724,
        12.072136575632017,
        12.570433156881965,
        12.435827624848123,
        13.750744970607428,
        13.567397714461393,
        14.218726703934166,
        14.459837938730677,
        14.659912736018788,
        14.077914185301429,
        14.854380461280002,
        10.770274645112915,
        11.275621916712160,
        12.137534572839927,
        12.531125692916383,
        12.678920118269170,
        12.304148175294246,
        12.497145874675160,
        14.103389828901550,
        10.560062989643855,
        10.755394606294518,
        10.518678427483797,
        11.721841324084256,
        11.607272952190801,
        11.924464521898100,
        12.782516039349641,
        13.026729430318186,
        12.546145790341205,
        13.409407032231695,
        14.079787980063543,
        13.128838312144593,
        13.553836458429620,
        13.718363411441658,
        13.854625752117343,
        14.924224028489123,
        11.906891367097627,
        12.128784222882244,
        11.404804355878456,
        13.130649630134753,
        12.173021974919472,
        12.859165585526416,
        12.895280738363951,
        13.345233593320895,
        10.435966548001499,
        10.663839793569295,
        11.030422432974012,
        11.033668451079661,
        11.324277503659044,
        11.045836529045589,
        11.985219205566086,
        12.220060940064094,
        14.722723885094736,
        15.772410109968900,
        15.256969467031452,
        15.568564129971197,
        16.666133193788099,
        16.405462433247578,
        17.202870693537243,
        17.289652559976691,
        7.760317864391456,
        8.460282811921017,
        9.462415007659978,
        9.956467084312777,
        9.726218110324272,
        10.272688229133685,
        11.134101608790994,
        11.592584658589104,
        7.747112683063268,
        8.706521663648207,
        8.170907672905205,
        8.679537720718859,
        8.962718814069811,
        8.861932954235140,
        9.383430460745986,
        9.891050023644237,
        9.728955313568255,
        9.231765881057163,
        9.555677785583788,
        10.420693590160205,
        9.844078095298698,
        10.651913064308546,
        10.196489890710358,
        11.855847076501993,
        9.218785934915712,
        9.133582433258733,
        10.048827580363175,
        9.952567508276010,
        10.385962432276619,
        11.596546220044132,
        11.164945662130776,
        11.016817405176500,
        10.145044557120791,
        10.921420538928436,
        11.642624728800259,
        10.730067509380019,
        11.753738913724906,
        11.868862794274008,
        12.574196556067037,
        12.311524695461632,
        10.800710206252880,
        12.817967597577915,
        12.705627126180516,
        12.497850142478354,
        12.148734571851643,
        13.494742486942219,
        13.714835068828613,
        13.770060323710533,
        10.010857300549947,
        10.787315152039971,
        11.050238955584605,
        11.063282099053561,
        10.834793458278272,
        17.153286194944865,
        17.380010096861866,
        16.984758489324143,
        6.913302966281331,
        6.938279687001069,
        7.537129527669741,
        7.063822443245238,
        7.531238453797332,
        13.853711102827464,
        13.812711128345372,
        14.204067444347162,
        13.694867606609098,
        12.929992273442151,
        14.397345491024691,
        15.116119455987304,
        15.860226513457558,
        19.442026093187646,
        19.855029109494353,
        20.377546194927845,
    ]
    N0, N1, T0, T1 = 20, 3, 5, 3
    R_ATT = 4.980848860060929
    R_JACKKNIFE_SE = 0.613846670319004

    @pytest.fixture
    def r_panel_df(self):
        """Reconstruct the R panel as a pandas DataFrame."""
        N = self.N0 + self.N1
        T = self.T0 + self.T1
        Y = np.array(self.Y_FLAT).reshape(N, T)
        rows = []
        for i in range(N):
            for t in range(T):
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": Y[i, t],
                        "treated": int(i >= self.N0),
                    }
                )
        return pd.DataFrame(rows)

    def test_att_matches_r(self, r_panel_df):
        """ATT should match R's synthdid_estimate to machine precision."""
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            r_panel_df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[5, 6, 7],
        )
        assert abs(results.att - self.R_ATT) < 1e-10

    def test_jackknife_se_matches_r(self, r_panel_df):
        """Jackknife SE should match R's vcov(method='jackknife')."""
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            r_panel_df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[5, 6, 7],
        )
        assert abs(results.se - self.R_JACKKNIFE_SE) < 1e-10

    def test_placebo_se_matches_r(self, r_panel_df):
        """Placebo SE should match R's vcov(method='placebo') under the
        same panel and the exact permutation sequence R consumed.

        Loads ``tests/data/sdid_placebo_indices_r.json``, which pins
        R's per-rep permutation indices (200 × N0, 0-indexed) and the
        SE from R's ``placebo_se`` loop.

        Intercepts ``_placebo_variance_se`` to capture the per-fit args
        the dispatcher would have passed (Y_*_n, zetas, init_omega,
        init_lambda, etc.), then re-invokes the method directly with
        the same args plus the ``_placebo_indices`` test seam carrying
        R's permutations. Avoids manually reconstructing the
        normalization constants.

        Python uses the same warm-start ``weights.boot$omega =
        sum_normalize(omega[ind[1:N0_placebo]])`` that R's
        ``placebo_se`` consumes from ``attr(estimate, "weights")`` —
        the warm-start matters for finite-iter FW convergence under
        R's ``update.omega=TRUE`` semantics, even though the global
        FW optimum is init-independent.

        Skip if the fixture file is missing — CI's isolated-install
        job copies only ``tests/``, but ``tests/data/`` is included
        (this skip is a defensive guard per the existing convention).
        """
        import json
        import pathlib

        fixture_path = pathlib.Path(__file__).parent / "data" / "sdid_placebo_indices_r.json"
        if not fixture_path.exists():
            pytest.skip(
                f"Missing R-parity fixture {fixture_path}; regenerate via "
                "`Rscript benchmarks/R/generate_sdid_placebo_parity_fixture.R`."
            )
        payload = json.loads(fixture_path.read_text())
        r_se = payload["R_PLACEBO_SE"]
        r_perms = np.asarray(payload["R_PERMUTATIONS"], dtype=np.int64)
        replications = payload["metadata"]["replications"]
        assert r_perms.shape == (replications, self.N0)

        sdid = SyntheticDiD(
            variance_method="placebo",
            n_bootstrap=replications,
            seed=42,
        )
        # Capture the dispatcher's call args for `_placebo_variance_se`.
        captured: Dict[str, Any] = {}
        original_method = sdid._placebo_variance_se

        def capture_then_call(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return original_method(*args, **kwargs)

        sdid._placebo_variance_se = capture_then_call  # type: ignore[assignment]
        sdid.fit(
            r_panel_df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[5, 6, 7],
        )
        sdid._placebo_variance_se = original_method  # type: ignore[assignment]

        # Re-call `_placebo_variance_se` with the SAME normalized inputs
        # the dispatcher used, plus R's permutations through the seam.
        # The recovered Y_scale rescales the normalized SE back to user
        # outcome scale (matches the dispatcher's `se = se_n * Y_scale`
        # at synthetic_did.py around L1164).
        kwargs = dict(captured["kwargs"])
        kwargs["replications"] = replications
        kwargs["_placebo_indices"] = r_perms
        se_n, placebo_effects_n = sdid._placebo_variance_se(*captured["args"], **kwargs)
        Y_scale = sdid.results_.zeta_omega / kwargs["zeta_omega"]
        py_se = se_n * Y_scale
        py_taus = np.asarray(placebo_effects_n) * Y_scale
        # Match R within cross-library FW tolerance (Rust vs R BLAS
        # reductions differ at sub-ULP; 1e-8 absorbs that without
        # masking a real divergence).
        assert (
            abs(py_se - r_se) < 1e-8
        ), f"Python placebo SE {py_se} != R {r_se} (delta {py_se - r_se})"
        # Per-draw τ regression: equal-SE doesn't imply equal sample, and
        # the placebo τ vector is user-visible through ``variance_effects``
        # and feeds the empirical placebo p-value (synthetic_did.py
        # around L1164-L1170). Compare elementwise so a permutation that
        # diverged at a single draw — but happened to leave sd() unchanged
        # — still trips the regression.
        r_taus = np.asarray(payload["R_PLACEBO_TAUS"], dtype=float)
        assert (
            r_taus.shape == py_taus.shape == (replications,)
        ), f"shape mismatch: r {r_taus.shape}, py {py_taus.shape}"
        np.testing.assert_allclose(
            py_taus,
            r_taus,
            atol=1e-8,
            rtol=1e-8,
            err_msg="Per-draw placebo τ diverges from R despite SE match",
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_n_bootstrap_validation(self):
        """n_bootstrap < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            SyntheticDiD(n_bootstrap=0)
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            SyntheticDiD(n_bootstrap=1)
        # n_bootstrap=2 should be accepted
        sdid = SyntheticDiD(n_bootstrap=2)
        assert sdid.n_bootstrap == 2

    def test_single_treated_unit(self, ci_params):
        """Estimation should work with a single treated unit."""
        df = _make_panel(n_control=10, n_treated=1, n_pre=5, n_post=2, att=3.0, seed=42)
        n_boot = ci_params.bootstrap(30)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6],
        )
        assert np.isfinite(results.att)

    def test_insufficient_controls_for_placebo(self):
        """Placebo with n_control <= n_treated should warn and return SE=0."""
        df = _make_panel(n_control=2, n_treated=3, n_pre=5, n_post=2, seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = sdid.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6],
            )
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # Should warn about insufficient controls for placebo
            assert len(user_warnings) > 0

        assert results.se == 0.0

    def test_se_zero_propagation(self):
        """When SE=0, t_stat and p_value should be NaN, CI should be NaN."""
        df = _make_panel(n_control=2, n_treated=3, n_pre=5, n_post=2, seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = sdid.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6],
            )

        if results.se == 0.0:
            assert np.isnan(results.t_stat)
            assert np.isnan(results.p_value)
            assert np.isnan(results.conf_int[0])
            assert np.isnan(results.conf_int[1])

    def test_nonfinite_tau_filtered_in_bootstrap(self):
        """Non-finite tau values trigger retry in Python bootstrap path.

        Under the R-matching retry-to-B contract, a non-finite estimator
        result is treated like a degenerate draw: it triggers another
        attempt rather than being silently dropped. The output must
        accumulate exactly `n_bootstrap` finite draws, and the estimator
        must have been called strictly more than `n_bootstrap` times
        (the retry path fired).
        """
        call_count = [0]

        def mock_estimator(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return np.inf
            return 1.0 + call_count[0] * 0.01

        rng = np.random.default_rng(42)
        n_pre, n_post, n_control, n_treated = 5, 2, 10, 3
        Y_pre_c = rng.normal(size=(n_pre, n_control))
        Y_post_c = rng.normal(size=(n_post, n_control))
        Y_pre_t = rng.normal(size=(n_pre, n_treated))
        Y_post_t = rng.normal(size=(n_post, n_treated))
        unit_weights = np.ones(n_control) / n_control
        time_weights = np.ones(n_pre) / n_pre

        sdid = SyntheticDiD(n_bootstrap=20, seed=42)

        with (
            patch("diff_diff.synthetic_did.compute_sdid_estimator", side_effect=mock_estimator),
            patch("diff_diff._backend.HAS_RUST_BACKEND", False),
        ):
            se, estimates = sdid._bootstrap_se(
                Y_pre_c,
                Y_post_c,
                Y_pre_t,
                Y_post_t,
                unit_weights,
                time_weights,
            )

        # All retained estimates must be finite (non-finite never leaks).
        assert np.all(np.isfinite(estimates)), "Non-finite tau leaked into bootstrap estimates"
        # Retry contract: accumulate exactly B valid draws (matches R).
        assert len(estimates) == 20
        # Retry fired: estimator was called more than B times because every
        # third call returned inf and triggered another attempt.
        assert (
            call_count[0] > 20
        ), f"expected retry path to fire (call_count > 20); got {call_count[0]}"

    def test_nonfinite_tau_filtered_in_placebo(self):
        """Non-finite tau values are filtered in Python placebo path (matches Rust)."""
        call_count = [0]

        def mock_estimator(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return np.nan
            return 2.0 + call_count[0] * 0.01

        rng = np.random.default_rng(42)
        n_pre, n_post, n_control, n_treated = 5, 2, 15, 3
        Y_pre_c = rng.normal(size=(n_pre, n_control))
        Y_post_c = rng.normal(size=(n_post, n_control))
        Y_pre_t_mean = rng.normal(size=(n_pre,))
        Y_post_t_mean = rng.normal(size=(n_post,))

        sdid = SyntheticDiD(seed=42)

        with (
            patch("diff_diff.synthetic_did.compute_sdid_estimator", side_effect=mock_estimator),
            patch(
                "diff_diff.synthetic_did.compute_sdid_unit_weights",
                return_value=np.ones(n_control - n_treated) / (n_control - n_treated),
            ),
            patch(
                "diff_diff.synthetic_did.compute_time_weights", return_value=np.ones(n_pre) / n_pre
            ),
            patch("diff_diff._backend.HAS_RUST_BACKEND", False),
        ):
            se, estimates = sdid._placebo_variance_se(
                Y_pre_c,
                Y_post_c,
                Y_pre_t_mean,
                Y_post_t_mean,
                n_treated=n_treated,
                replications=20,
            )

        # All retained estimates must be finite (non-finite filtered out)
        assert np.all(np.isfinite(estimates)), "Non-finite tau leaked into placebo estimates"
        # Some estimates should have been filtered (every 3rd call returns nan)
        assert len(estimates) < 20

    def test_inf_se_produces_nan_inference(self):
        """SE=inf should produce NaN t_stat, p_value, and CI."""
        df = _make_panel(n_control=10, n_treated=3, n_pre=5, n_post=2, seed=42)
        sdid = SyntheticDiD(variance_method="bootstrap", n_bootstrap=10, seed=42)

        with patch.object(
            SyntheticDiD,
            "_bootstrap_se",
            return_value=(np.inf, np.array([1.0, 2.0, 3.0])),
        ):
            results = sdid.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6],
            )

        assert results.se == np.inf
        assert np.isnan(results.t_stat)
        assert np.isnan(results.p_value)
        assert np.isnan(results.conf_int[0])
        assert np.isnan(results.conf_int[1])


# =============================================================================
# get_params / set_params
# =============================================================================


class TestGetSetParams:
    """Verify parameter accessors."""

    def test_get_params_includes_new_names(self):
        """get_params should include zeta_omega/zeta_lambda."""
        sdid = SyntheticDiD(zeta_omega=1.0, zeta_lambda=0.5)
        params = sdid.get_params()
        assert "zeta_omega" in params
        assert "zeta_lambda" in params
        assert params["zeta_omega"] == 1.0
        assert params["zeta_lambda"] == 0.5

    def test_get_params_excludes_old_names(self):
        """get_params should NOT include lambda_reg or zeta."""
        sdid = SyntheticDiD()
        params = sdid.get_params()
        assert "lambda_reg" not in params
        assert "zeta" not in params

    def test_set_params_new_names(self):
        """set_params with new names should work."""
        sdid = SyntheticDiD()
        sdid.set_params(zeta_omega=2.0, zeta_lambda=0.1)
        assert sdid.zeta_omega == 2.0
        assert sdid.zeta_lambda == 0.1

    def test_set_params_deprecated_names_warn(self):
        """set_params with old names should emit DeprecationWarning."""
        sdid = SyntheticDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid.set_params(lambda_reg=1.0)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1

    def test_set_params_unknown_raises(self):
        """set_params with unknown name should raise ValueError."""
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            sdid.set_params(nonexistent_param=1.0)

    def test_set_params_rejects_invalid_variance_method(self):
        """set_params with invalid variance_method raises ValueError.

        Parity with __init__: the setter path must enforce the same enum
        contract, not silently accept arbitrary strings.
        """
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="variance_method must be one of"):
            sdid.set_params(variance_method="not_a_method")

    def test_set_params_rejects_incoherent_n_bootstrap(self):
        """set_params(variance_method='bootstrap', n_bootstrap=1) raises.

        Parity with __init__: n_bootstrap >= 2 required unless jackknife.
        """
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            sdid.set_params(variance_method="bootstrap", n_bootstrap=1)

    def test_set_params_allows_n_bootstrap_one_for_jackknife(self):
        """Jackknife is deterministic; n_bootstrap is ignored and need not be >= 2."""
        sdid = SyntheticDiD()
        sdid.set_params(variance_method="jackknife", n_bootstrap=1)
        assert sdid.variance_method == "jackknife"
        assert sdid.n_bootstrap == 1

    def test_set_params_rolls_back_on_validation_failure(self):
        """On validation failure, ``set_params`` restores the pre-call state.

        Guards against leaving the instance in a partially-mutated invalid
        state if one of multiple simultaneous updates fails validation.
        """
        sdid = SyntheticDiD(variance_method="placebo", n_bootstrap=200)
        with pytest.raises(ValueError):
            sdid.set_params(variance_method="bootstrap", n_bootstrap=1)
        # Pre-call values preserved despite mid-sequence setattr on the
        # (now-partially-applied) instance.
        assert sdid.variance_method == "placebo"
        assert sdid.n_bootstrap == 200


class TestDeprecatedParams:
    """Test deprecated parameter handling in __init__."""

    def test_lambda_reg_warns(self):
        """SyntheticDiD(lambda_reg=...) emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid = SyntheticDiD(lambda_reg=0.1)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 1
            assert "lambda_reg" in str(dep[0].message)

        # Deprecated param is ignored — auto-computed used
        assert sdid.zeta_omega is None

    def test_zeta_warns(self):
        """SyntheticDiD(zeta=...) emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid = SyntheticDiD(zeta=2.0)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 1
            assert "zeta" in str(dep[0].message)

        assert sdid.zeta_lambda is None

    def test_both_deprecated_params(self):
        """Both deprecated params at once should emit two warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SyntheticDiD(lambda_reg=0.5, zeta=1.5)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 2

    def test_default_variance_method_is_placebo(self):
        """Default variance_method should be 'placebo' (matching R)."""
        sdid = SyntheticDiD()
        assert sdid.variance_method == "placebo"


class TestVarianceEffectsRename:
    """`placebo_effects` was renamed to `variance_effects`; the old name is a
    deprecated read-only alias (removed in v4.0.0)."""

    def test_placebo_effects_deprecated_alias(self):
        """result.placebo_effects warns and returns the same array as
        result.variance_effects; the alias is read-only."""
        df = _make_panel(seed=42)
        res = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=1).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
        )
        with pytest.warns(DeprecationWarning, match="placebo_effects is deprecated"):
            aliased = res.placebo_effects
        # Alias returns the identical object held by the renamed field.
        assert aliased is res.variance_effects
        # Read-only: assignment to the deprecated property raises.
        with pytest.raises(AttributeError):
            res.placebo_effects = aliased

    def test_variance_effects_access_emits_no_warning(self):
        """Normal use must not route through the deprecated alias: reading
        variance_effects and calling get_loo_effects_df() (which reads the
        field internally) must emit no DeprecationWarning."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        res = SyntheticDiD(variance_method="jackknife", seed=42).fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert res.variance_effects is not None
            # get_loo_effects_df() reads the field internally; if it routed
            # through the alias this would raise the escalated warning.
            res.get_loo_effects_df()


class TestNoiseLevelEdgeCases:
    """Edge case tests for _compute_noise_level_numpy."""

    def test_noise_level_single_control_two_periods(self):
        """noise_level returns 0.0 (not NaN) for 1 control, 2 pre-periods.

        With shape (2, 1), first_diffs has size=1, and np.std([x], ddof=1)
        would divide by zero → NaN. Guard ensures 0.0 is returned instead,
        matching the Rust backend behavior.
        """
        Y = np.array([[1.0], [2.0]])  # (2, 1)
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0
        assert not np.isnan(result)

    def test_noise_level_single_element_returns_zero(self):
        """noise_level returns 0.0 when first_diffs has exactly 1 element."""
        # (2, 1) → diff → (1, 1) → size=1 → return 0.0
        Y = np.array([[5.0], [10.0]])
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0

    def test_noise_level_empty_returns_zero(self):
        """noise_level returns 0.0 for single time period (no diffs possible)."""
        Y = np.array([[1.0, 2.0, 3.0]])  # (1, 3)
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0


class TestPlaceboReestimation:
    """Tests verifying placebo variance re-estimates weights (not fixed)."""

    def test_placebo_reestimates_weights_not_fixed(self):
        """Placebo variance re-estimates omega/lambda per replication (matching R).

        Verifies the methodology choice: R's vcov(method='placebo') passes
        update.omega=TRUE, update.lambda=TRUE, so weights are re-estimated
        via Frank-Wolfe on each permutation — NOT renormalized from originals.

        We verify this by comparing the actual placebo SE against a manual
        fixed-weight computation; if they differ, re-estimation is happening.
        """
        # Need enough controls for placebo to work (n_control > n_treated)
        # and enough variation for weights to differ between re-estimation
        # and renormalization.
        df = _make_panel(n_control=15, n_treated=2, n_pre=6, n_post=3, att=5.0, seed=123)
        post_periods = list(range(6, 9))

        # Fit SDID to get original weights and matrices
        sdid = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
        )
        actual_se = results.se

        # Now compute a "fixed weight" placebo manually:
        # permute controls, renormalize original omega (no Frank-Wolfe),
        # keep original lambda unchanged.
        rng = np.random.default_rng(42)

        # Build the outcome matrix (T, N) as the estimator does
        pivot = df.pivot(index="period", columns="unit", values="outcome")
        control_units = sorted(results.unit_weights.keys())
        treated_mask = df.groupby("unit")["treated"].max().values.astype(bool)
        control_idx = np.where(~treated_mask)[0]
        treated_idx = np.where(treated_mask)[0]
        Y = pivot.values  # (T, N)
        pre_periods_arr = np.array(post_periods)
        pre_mask = ~np.isin(pivot.index.values, pre_periods_arr)

        Y_pre_control = Y[np.ix_(pre_mask, control_idx)]
        Y_post_control = Y[np.ix_(~pre_mask, control_idx)]

        # Extract numpy arrays from result dicts (ordered by control unit)
        unit_weights_arr = np.array([results.unit_weights[u] for u in control_units])
        time_weights_arr = np.array(
            [results.time_weights[t] for t in sorted(results.time_weights.keys())]
        )

        n_control = len(control_idx)
        n_treated_count = len(treated_idx)
        n_pseudo_control = n_control - n_treated_count

        fixed_estimates = []
        for _ in range(50):
            perm = rng.permutation(n_control)
            pc_idx = perm[:n_pseudo_control]
            pt_idx = perm[n_pseudo_control:]

            # Fixed weights: renormalize original omega for pseudo-controls
            fixed_omega = _sum_normalize(unit_weights_arr[pc_idx])
            fixed_lambda = time_weights_arr  # unchanged

            Y_pre_pc = Y_pre_control[:, pc_idx]
            Y_post_pc = Y_post_control[:, pc_idx]
            Y_pre_pt_mean = np.mean(Y_pre_control[:, pt_idx], axis=1)
            Y_post_pt_mean = np.mean(Y_post_control[:, pt_idx], axis=1)

            try:
                tau = compute_sdid_estimator(
                    Y_pre_pc,
                    Y_post_pc,
                    Y_pre_pt_mean,
                    Y_post_pt_mean,
                    fixed_omega,
                    fixed_lambda,
                )
                fixed_estimates.append(tau)
            except (ValueError, np.linalg.LinAlgError):
                continue

        if len(fixed_estimates) >= 2:
            n_s = len(fixed_estimates)
            fixed_se = np.sqrt((n_s - 1) / n_s) * np.std(fixed_estimates, ddof=1)
            # The two SEs should differ because re-estimation produces
            # different weights than renormalization
            assert actual_se != pytest.approx(fixed_se, rel=0.01), (
                f"Placebo SE ({actual_se:.6f}) matches fixed-weight SE "
                f"({fixed_se:.6f}), suggesting weights are NOT being "
                f"re-estimated as R's synthdid does."
            )


# =============================================================================
# Treatment Validation
# =============================================================================


class TestTreatmentValidation:
    """Test that SDID rejects time-varying treatment (staggered designs)."""

    def test_varying_treatment_within_unit_raises(self):
        """Unit whose treatment switches over time should raise ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "unit": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "outcome": np.random.randn(12),
                # Unit 1: treatment turns on at time 3 (staggered)
                "treated": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Treatment indicator varies within"):
            sdid.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[3, 4],
            )

    def test_constant_treatment_passes(self):
        """Normal block-treatment data should pass validation."""
        np.random.seed(42)
        n_units, n_periods = 10, 8
        rows = []
        for u in range(n_units):
            is_treated = 1 if u < 3 else 0
            for t in range(n_periods):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "outcome": np.random.randn() + (2.0 if is_treated and t >= 5 else 0),
                        "treated": is_treated,
                    }
                )
        data = pd.DataFrame(rows)
        sdid = SyntheticDiD()
        result = sdid.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[5, 6, 7],
        )
        assert result is not None


# =============================================================================
# Balanced Panel Validation
# =============================================================================


class TestBalancedPanelValidation:
    """Test that SDID rejects unbalanced panels."""

    def test_unbalanced_panel_raises(self):
        """Unit missing a period should raise ValueError."""
        np.random.seed(42)
        rows = []
        for u in range(6):
            is_treated = 1 if u < 2 else 0
            for t in range(5):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "outcome": np.random.randn(),
                        "treated": is_treated,
                    }
                )
        data = pd.DataFrame(rows)
        # Drop one observation to make panel unbalanced
        data = data[~((data["unit"] == 3) & (data["time"] == 2))].reset_index(drop=True)

        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Panel is not balanced"):
            sdid.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[3, 4],
            )

    def test_balanced_panel_passes(self):
        """Fully balanced panel should pass validation."""
        np.random.seed(42)
        rows = []
        for u in range(8):
            is_treated = 1 if u < 2 else 0
            for t in range(6):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "outcome": np.random.randn() + (1.5 if is_treated and t >= 4 else 0),
                        "treated": is_treated,
                    }
                )
        data = pd.DataFrame(rows)
        sdid = SyntheticDiD()
        result = sdid.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[4, 5],
        )
        assert result is not None


# =============================================================================
# Pre-treatment Fit Warning
# =============================================================================


class TestPreTreatmentFitWarning:
    """Test that poor pre-treatment fit emits a warning."""

    def test_poor_fit_emits_warning(self):
        """Treated units at very different level from controls should warn."""
        np.random.seed(42)
        rows = []
        for u in range(10):
            is_treated = 1 if u < 2 else 0
            # Large level difference: treated ~100, control ~10
            level = 100.0 if is_treated else 10.0
            for t in range(8):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "outcome": level + np.random.randn() * 0.5,
                        "treated": is_treated,
                    }
                )
        data = pd.DataFrame(rows)
        sdid = SyntheticDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7],
            )
            fit_warnings = [x for x in w if "Pre-treatment fit is poor" in str(x.message)]
            assert (
                len(fit_warnings) >= 1
            ), "Expected warning about poor pre-treatment fit but none was raised"

    def test_good_fit_no_warning(self):
        """Parallel trends data with similar levels should not warn."""
        np.random.seed(42)
        rows = []
        for u in range(10):
            is_treated = 1 if u < 3 else 0
            for t in range(8):
                # Same level, parallel trends, treatment effect only in post
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "outcome": t
                        + np.random.randn() * 0.3
                        + (2.0 if is_treated and t >= 5 else 0),
                        "treated": is_treated,
                    }
                )
        data = pd.DataFrame(rows)
        sdid = SyntheticDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid.fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[5, 6, 7],
            )
            fit_warnings = [x for x in w if "Pre-treatment fit is poor" in str(x.message)]
            assert (
                len(fit_warnings) == 0
            ), f"Unexpected pre-treatment fit warning: {fit_warnings[0].message}"


class TestMinDecreaseFloor:
    """Test min_decrease floor when noise_level == 0."""

    def test_floor_equivalence(self):
        """min_decrease=1e-5 floor vs 0.0 gives same weights on zero-noise data."""
        # Build zero-noise collapsed-form matrix (N_co=5, T_pre+1=4)
        # All rows identical → noise_level would be 0
        np.random.seed(42)
        Y = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (5, 1))

        w_floor = _sc_weight_fw(Y, zeta=0.0, intercept=True, min_decrease=1e-5, max_iter=10000)
        w_zero = _sc_weight_fw(Y, zeta=0.0, intercept=True, min_decrease=0.0, max_iter=10000)

        np.testing.assert_allclose(
            w_floor,
            w_zero,
            atol=1e-10,
            err_msg="Floor min_decrease should give same weights as 0.0",
        )
        # Both should be valid simplex weights
        assert np.all(w_floor >= -1e-12), "Weights should be non-negative"
        assert abs(w_floor.sum() - 1.0) < 1e-10, "Weights should sum to 1"


class TestBackendSEConsistency:
    """Test that pure Python and Rust-accelerated backends produce identical SEs."""

    def test_placebo_se_matches_across_backends(self):
        """SyntheticDiD placebo SE is identical regardless of backend.

        After removing the Rust outer-loop variance fast paths, both backends
        use the same Python loop for placebo/bootstrap variance. The only
        difference is that inner Frank-Wolfe weight calls may dispatch to Rust
        (faer) vs NumPy. With the same seed, RNG sequences are identical, so
        SEs should match within floating-point tolerance (rtol=1e-4 allows for
        accumulated faer vs NumPy differences across hundreds of iterations).
        """
        import sys

        df = _make_panel(n_control=20, n_treated=3, n_pre=5, n_post=3, att=5.0, seed=42)
        post_periods = list(range(5, 8))

        # Run with default backend (Rust-accelerated inner calls if available)
        sdid_default = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        results_default = sdid_default.fit(df, "outcome", "treated", "unit", "period", post_periods)

        # Run with pure Python backend
        utils_mod = sys.modules["diff_diff.utils"]
        with patch.object(utils_mod, "HAS_RUST_BACKEND", False):
            sdid_py = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
            results_py = sdid_py.fit(
                df.copy(), "outcome", "treated", "unit", "period", post_periods
            )

        # ATT must be identical (same data, same algorithm)
        np.testing.assert_allclose(
            results_default.att,
            results_py.att,
            rtol=1e-6,
            err_msg="ATT should match between backends",
        )

        # SE must match within tolerance (faer vs NumPy in FW inner loop)
        np.testing.assert_allclose(
            results_default.se,
            results_py.se,
            rtol=1e-4,
            err_msg="Placebo SE should match between backends",
        )

    def test_bootstrap_se_matches_across_backends(self):
        """SyntheticDiD bootstrap SE is identical regardless of backend."""
        import sys

        df = _make_panel(n_control=20, n_treated=3, n_pre=5, n_post=3, att=5.0, seed=42)
        post_periods = list(range(5, 8))

        # Run with default backend
        sdid_default = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)
        results_default = sdid_default.fit(df, "outcome", "treated", "unit", "period", post_periods)

        # Run with pure Python backend
        utils_mod = sys.modules["diff_diff.utils"]
        with patch.object(utils_mod, "HAS_RUST_BACKEND", False):
            sdid_py = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)
            results_py = sdid_py.fit(
                df.copy(), "outcome", "treated", "unit", "period", post_periods
            )

        # ATT must match
        np.testing.assert_allclose(
            results_default.att,
            results_py.att,
            rtol=1e-6,
            err_msg="ATT should match between backends",
        )

        # Bootstrap SE must match (same RNG, same loop, only inner FW differs)
        np.testing.assert_allclose(
            results_default.se,
            results_py.se,
            rtol=1e-4,
            err_msg="Bootstrap SE should match between backends",
        )


class TestEmptyPostGuard:
    """Test compute_time_weights guard for empty Y_post_control."""

    def test_empty_post_raises(self):
        """compute_time_weights raises ValueError when Y_post_control has 0 rows."""
        Y_pre = np.random.randn(4, 5)  # 4 pre-periods, 5 controls
        Y_post = np.empty((0, 5))  # 0 post-periods

        with pytest.raises(ValueError, match="Y_post_control has no rows"):
            compute_time_weights(Y_pre, Y_post, zeta_lambda=0.0)


# =============================================================================
# Validation Diagnostics: Fit Snapshot, Trajectories, LOO, Concentration,
#                         In-Time Placebo, Regularization Sensitivity
# =============================================================================


class TestFitSnapshot:
    """Verify the private _fit_snapshot bundle carries the right state."""

    def test_shapes_and_ordering(self):
        df = _make_panel(n_control=10, n_treated=2, n_pre=5, n_post=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        snap = res._fit_snapshot
        assert snap is not None
        assert snap.Y_pre_control.shape == (5, 10)
        assert snap.Y_post_control.shape == (3, 10)
        assert snap.Y_pre_treated.shape == (5, 2)
        assert snap.Y_post_treated.shape == (3, 2)
        assert len(snap.control_unit_ids) == 10
        assert len(snap.treated_unit_ids) == 2
        assert len(snap.pre_periods) == 5
        assert len(snap.post_periods) == 3
        # No unit appears in both sides
        assert set(snap.control_unit_ids).isdisjoint(set(snap.treated_unit_ids))

    def test_matrices_are_read_only(self):
        df = _make_panel(seed=7)
        sdid = SyntheticDiD(variance_method="jackknife", seed=7)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        snap = res._fit_snapshot
        assert not snap.Y_pre_control.flags.writeable
        assert not snap.Y_post_control.flags.writeable
        assert not snap.Y_pre_treated.flags.writeable
        assert not snap.Y_post_treated.flags.writeable
        with pytest.raises(ValueError):
            snap.Y_pre_control[0, 0] = -999.0


class TestTrajectories:
    """Synthetic and treated trajectories are exposed on results."""

    def test_synthetic_pre_matches_weighted_sum(self):
        df = _make_panel(seed=11)
        sdid = SyntheticDiD(variance_method="jackknife", seed=11)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        omega_eff = np.array([res.unit_weights[u] for u in res._fit_snapshot.control_unit_ids])
        expected = res._fit_snapshot.Y_pre_control @ omega_eff
        assert np.allclose(res.synthetic_pre_trajectory, expected, atol=1e-12)

    def test_treated_trajectory_matches_mean(self):
        df = _make_panel(seed=13)
        sdid = SyntheticDiD(variance_method="jackknife", seed=13)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        assert np.allclose(
            res.treated_pre_trajectory,
            np.mean(res._fit_snapshot.Y_pre_treated, axis=1),
            atol=1e-12,
        )
        assert np.allclose(
            res.treated_post_trajectory,
            np.mean(res._fit_snapshot.Y_post_treated, axis=1),
            atol=1e-12,
        )

    def test_treated_trajectory_survey_weighted(self):
        """Under survey weights, treated trajectories should equal the
        survey-weighted mean per the registry; the synthetic trajectory
        should match Y_pre_control @ omega_eff using composed weights."""
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=10, n_treated=3, n_pre=5, n_post=3, seed=83)
        w_by_unit = {u: 1.0 for u in df["unit"].unique()}
        w_by_unit[df["unit"].iloc[-1]] = 5.0  # skew one treated unit
        df = df.assign(weight=df["unit"].map(w_by_unit))
        sdid = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=83)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            survey_design=SurveyDesign(weights="weight", weight_type="pweight"),
        )
        snap = res._fit_snapshot
        w_t = snap.w_treated
        expected_pre = np.average(snap.Y_pre_treated, axis=1, weights=w_t)
        expected_post = np.average(snap.Y_post_treated, axis=1, weights=w_t)
        assert np.allclose(res.treated_pre_trajectory, expected_pre, atol=1e-12)
        assert np.allclose(res.treated_post_trajectory, expected_post, atol=1e-12)

        omega_eff = np.array([res.unit_weights[u] for u in snap.control_unit_ids])
        expected_synth_pre = snap.Y_pre_control @ omega_eff
        assert np.allclose(res.synthetic_pre_trajectory, expected_synth_pre, atol=1e-12)

    def test_public_trajectory_arrays_are_read_only(self):
        df = _make_panel(seed=87)
        sdid = SyntheticDiD(variance_method="jackknife", seed=87)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        for arr in (
            res.synthetic_pre_trajectory,
            res.synthetic_post_trajectory,
            res.treated_pre_trajectory,
            res.treated_post_trajectory,
            res.time_weights_array,
        ):
            assert not arr.flags.writeable

    def test_pre_fit_rmse_recoverable(self):
        df = _make_panel(seed=17)
        sdid = SyntheticDiD(variance_method="jackknife", seed=17)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        rmse = float(
            np.sqrt(np.mean((res.treated_pre_trajectory - res.synthetic_pre_trajectory) ** 2))
        )
        assert abs(rmse - res.pre_treatment_fit) < 1e-10


class TestLooEffectsDf:
    """get_loo_effects_df joins jackknife pseudo-values to unit identities."""

    def _fit_jackknife(self, seed=42, **kwargs):
        df = _make_panel(n_control=10, n_treated=3, n_pre=5, n_post=3, seed=seed, **kwargs)
        sdid = SyntheticDiD(variance_method="jackknife", seed=seed)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(5, 8)),
        )
        return res

    def test_columns_and_roles(self):
        res = self._fit_jackknife()
        loo = res.get_loo_effects_df()
        assert list(loo.columns) == ["unit", "role", "att_loo", "delta_from_full"]
        assert set(loo["role"]) == {"control", "treated"}
        assert (loo["role"] == "control").sum() == res.n_control
        assert (loo["role"] == "treated").sum() == res.n_treated
        assert set(loo["unit"]) == set(res.unit_weights) | set(res._fit_snapshot.treated_unit_ids)

    def test_delta_from_full_matches_math(self):
        res = self._fit_jackknife()
        loo = res.get_loo_effects_df()
        deltas = loo["att_loo"] - res.att
        assert np.allclose(loo["delta_from_full"], deltas, equal_nan=True)

    def test_sorted_by_absolute_delta_descending(self):
        res = self._fit_jackknife()
        loo = res.get_loo_effects_df().dropna(subset=["delta_from_full"])
        abs_delta = loo["delta_from_full"].abs().to_numpy()
        assert np.all(abs_delta[:-1] >= abs_delta[1:])

    def test_placebo_raises_value_error(self):
        df = _make_panel(seed=21)
        sdid = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=21)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        with pytest.raises(ValueError, match="variance_method='jackknife'"):
            res.get_loo_effects_df()

    def test_bootstrap_raises_value_error(self):
        df = _make_panel(seed=23)
        sdid = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=23)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        with pytest.raises(ValueError, match="variance_method='jackknife'"):
            res.get_loo_effects_df()

    def test_positional_mapping_matches_variance_effects(self):
        """First n_control positions in variance_effects map to control_units,
        next n_treated map to treated_units."""
        res = self._fit_jackknife()
        pe = res.variance_effects
        ids = res._loo_unit_ids
        roles = res._loo_roles
        assert list(ids[: res.n_control]) == res._fit_snapshot.control_unit_ids
        assert list(ids[res.n_control :]) == res._fit_snapshot.treated_unit_ids
        assert roles[: res.n_control].count("control") == res.n_control
        assert roles[res.n_control :].count("treated") == res.n_treated
        # The DataFrame values equal variance_effects values (up to row permutation)
        loo = res.get_loo_effects_df()
        assert np.allclose(
            sorted(loo["att_loo"].dropna().to_numpy()),
            sorted(pe[np.isfinite(pe)]),
            atol=1e-12,
        )


class TestWeightConcentration:
    """get_weight_concentration returns correct concentration metrics."""

    def test_equal_weights_yield_effective_n_equal_to_count(self):
        """With uniform weights 1/n, effective_n = n."""
        # Manually construct a results object with uniform weights
        from diff_diff.results import SyntheticDiDResults

        uniform = {f"u{i}": 0.1 for i in range(10)}
        res = SyntheticDiDResults(
            att=0.0,
            se=0.0,
            t_stat=0.0,
            p_value=1.0,
            conf_int=(0.0, 0.0),
            n_obs=100,
            n_treated=1,
            n_control=10,
            unit_weights=uniform,
            time_weights={},
            pre_periods=[],
            post_periods=[],
        )
        c = res.get_weight_concentration()
        assert abs(c["effective_n"] - 10.0) < 1e-10
        assert abs(c["herfindahl"] - 0.1) < 1e-10  # 10 * 0.01
        assert c["top_k"] == 5
        assert abs(c["top_k_share"] - 0.5) < 1e-10

    def test_single_unit_yields_effective_n_one(self):
        from diff_diff.results import SyntheticDiDResults

        res = SyntheticDiDResults(
            att=0.0,
            se=0.0,
            t_stat=0.0,
            p_value=1.0,
            conf_int=(0.0, 0.0),
            n_obs=10,
            n_treated=1,
            n_control=1,
            unit_weights={"only": 1.0},
            time_weights={},
            pre_periods=[],
            post_periods=[],
        )
        c = res.get_weight_concentration()
        assert abs(c["effective_n"] - 1.0) < 1e-10
        assert abs(c["herfindahl"] - 1.0) < 1e-10

    def test_top_k_clamped_to_n(self):
        from diff_diff.results import SyntheticDiDResults

        res = SyntheticDiDResults(
            att=0.0,
            se=0.0,
            t_stat=0.0,
            p_value=1.0,
            conf_int=(0.0, 0.0),
            n_obs=30,
            n_treated=1,
            n_control=3,
            unit_weights={"a": 0.5, "b": 0.3, "c": 0.2},
            time_weights={},
            pre_periods=[],
            post_periods=[],
        )
        c = res.get_weight_concentration(top_k=10)
        assert c["top_k"] == 3
        assert abs(c["top_k_share"] - 1.0) < 1e-10

    def test_negative_top_k_raises(self):
        from diff_diff.results import SyntheticDiDResults

        res = SyntheticDiDResults(
            att=0.0,
            se=0.0,
            t_stat=0.0,
            p_value=1.0,
            conf_int=(0.0, 0.0),
            n_obs=30,
            n_treated=1,
            n_control=3,
            unit_weights={"a": 0.5, "b": 0.3, "c": 0.2},
            time_weights={},
            pre_periods=[],
            post_periods=[],
        )
        with pytest.raises(ValueError, match="top_k must be non-negative"):
            res.get_weight_concentration(top_k=-1)

    def test_uses_composed_weights_under_survey(self):
        """Metrics come from self.unit_weights which stores composed ω_eff
        for survey fits."""
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=8, n_treated=2, n_pre=4, n_post=2, seed=29)
        # Add survey weights heavily favoring one control unit
        w_by_unit = {u: 1.0 for u in df["unit"].unique()}
        w_by_unit[0] = 20.0  # control unit 0 gets a heavy weight
        df = df.assign(weight=df["unit"].map(w_by_unit))
        sdid = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=29)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            survey_design=SurveyDesign(weights="weight", weight_type="pweight"),
        )
        c = res.get_weight_concentration()
        # Composed metrics reflect unit_weights dict exactly
        import numpy as np

        weights = np.array(list(res.unit_weights.values()))
        expected_effective_n = 1.0 / np.sum(weights**2)
        assert abs(c["effective_n"] - expected_effective_n) < 1e-10


class TestInTimePlacebo:
    """in_time_placebo re-estimates on shifted fake dates."""

    def test_default_sweep_shape(self):
        df = _make_panel(n_control=10, n_treated=2, n_pre=8, n_post=3, att=0.0, seed=31)
        sdid = SyntheticDiD(variance_method="jackknife", seed=31)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(8, 11)),
        )
        placebo = res.in_time_placebo()
        # Feasible positions: i in [2, n_pre - 1] = [2, 7] -> 6 rows
        assert len(placebo) == 6
        assert list(placebo.columns) == [
            "fake_treatment_period",
            "att",
            "pre_fit_rmse",
            "n_pre_fake",
            "n_post_fake",
        ]

    def test_no_effect_dgp_gives_small_placebo_atts(self):
        """On a clean no-effect DGP, placebo ATTs should be small vs the
        DGP's noise level."""
        df = _make_panel(n_control=20, n_treated=3, n_pre=8, n_post=3, att=0.0, seed=33)
        sdid = SyntheticDiD(variance_method="jackknife", seed=33)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(8, 11)),
        )
        placebo = res.in_time_placebo()
        median_abs = float(placebo["att"].abs().median())
        # Noise sd in _make_panel is 0.5; ATTs should be well under that at
        # the median. Loose threshold to stay stable under seed variation.
        assert median_abs < 1.0

    def test_explicit_list_overrides_sweep(self):
        df = _make_panel(n_control=10, n_treated=2, n_pre=8, n_post=3, seed=37)
        sdid = SyntheticDiD(variance_method="jackknife", seed=37)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(8, 11)),
        )
        placebo = res.in_time_placebo(fake_treatment_periods=[4, 6])
        assert len(placebo) == 2
        assert set(placebo["fake_treatment_period"]) == {4, 6}

    def test_post_period_raises(self):
        df = _make_panel(n_control=10, n_treated=2, n_pre=6, n_post=3, seed=41)
        sdid = SyntheticDiD(variance_method="jackknife", seed=41)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(6, 9)),
        )
        with pytest.raises(ValueError, match="post_periods"):
            res.in_time_placebo(fake_treatment_periods=[6])

    def test_missing_period_raises(self):
        df = _make_panel(n_control=10, n_treated=2, n_pre=6, n_post=3, seed=43)
        sdid = SyntheticDiD(variance_method="jackknife", seed=43)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(6, 9)),
        )
        with pytest.raises(ValueError, match="not found in pre_periods"):
            res.in_time_placebo(fake_treatment_periods=[999])

    def test_empty_default_sweep_preserves_schema(self):
        """When n_pre < 3, the default sweep is empty. The DataFrame must
        still carry the documented columns."""
        df = _make_panel(n_control=10, n_treated=2, n_pre=2, n_post=3, seed=50)
        sdid = SyntheticDiD(variance_method="jackknife", seed=50)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(2, 5)),
        )
        placebo = res.in_time_placebo()
        assert len(placebo) == 0
        assert list(placebo.columns) == [
            "fake_treatment_period",
            "att",
            "pre_fit_rmse",
            "n_pre_fake",
            "n_post_fake",
        ]

    def test_zeta_override_changes_result(self):
        df = _make_panel(n_control=10, n_treated=2, n_pre=8, n_post=3, seed=47)
        sdid = SyntheticDiD(variance_method="jackknife", seed=47)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(8, 11)),
        )
        default_ = res.in_time_placebo(fake_treatment_periods=[5])
        override_ = res.in_time_placebo(
            fake_treatment_periods=[5],
            zeta_omega_override=res.zeta_omega * 100,
        )
        # Different regularization should change at least one of att/rmse
        assert not np.isclose(
            default_["pre_fit_rmse"].iloc[0],
            override_["pre_fit_rmse"].iloc[0],
            atol=1e-8,
        )


class TestSensitivityToZetaOmega:
    """sensitivity_to_zeta_omega sweeps regularization values."""

    def test_multiplier_one_reproduces_original_att(self):
        df = _make_panel(n_control=12, n_treated=2, n_pre=6, n_post=3, att=2.0, seed=53)
        sdid = SyntheticDiD(variance_method="jackknife", seed=53)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(6, 9)),
        )
        sens = res.sensitivity_to_zeta_omega()
        # Row where zeta_omega == self.zeta_omega (multiplier 1.0)
        match = sens.loc[np.isclose(sens["zeta_omega"], res.zeta_omega)]
        assert len(match) == 1
        assert abs(match["att"].iloc[0] - res.att) < 1e-10

    def test_grid_length_matches_default_multipliers(self):
        df = _make_panel(seed=59)
        sdid = SyntheticDiD(variance_method="jackknife", seed=59)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        sens = res.sensitivity_to_zeta_omega()
        assert len(sens) == 5

    def test_default_grid_values_match_documented_multipliers(self):
        """Assert the exact documented default grid values, not just length,
        so the contract cannot drift unnoticed."""
        df = _make_panel(seed=60)
        sdid = SyntheticDiD(variance_method="jackknife", seed=60)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        sens = res.sensitivity_to_zeta_omega()
        expected = np.array([0.25, 0.5, 1.0, 2.0, 4.0]) * res.zeta_omega
        assert np.allclose(sens["zeta_omega"].to_numpy(), expected, atol=1e-12)

    def test_explicit_grid_overrides_multipliers(self):
        df = _make_panel(seed=61)
        sdid = SyntheticDiD(variance_method="jackknife", seed=61)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        sens = res.sensitivity_to_zeta_omega(zeta_grid=[0.1, 1.0, 10.0])
        assert len(sens) == 3
        assert np.allclose(sens["zeta_omega"], [0.1, 1.0, 10.0])

    def test_effective_n_grows_with_zeta(self):
        """Higher regularization pushes weights toward uniform, so effective_n
        should be monotone non-decreasing across the default grid."""
        df = _make_panel(n_control=15, n_treated=2, n_pre=8, n_post=3, seed=67)
        sdid = SyntheticDiD(variance_method="jackknife", seed=67)
        res = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=list(range(8, 11)),
        )
        sens = res.sensitivity_to_zeta_omega()
        eff_n = sens["effective_n"].to_numpy()
        # Allow tiny wobble from solver tolerance
        assert np.all(np.diff(eff_n) >= -1e-6)

    def test_columns_shape(self):
        df = _make_panel(seed=71)
        sdid = SyntheticDiD(variance_method="jackknife", seed=71)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        sens = res.sensitivity_to_zeta_omega()
        assert list(sens.columns) == [
            "zeta_omega",
            "att",
            "pre_fit_rmse",
            "max_unit_weight",
            "effective_n",
        ]

    def test_empty_zeta_grid_preserves_schema(self):
        df = _make_panel(seed=91)
        sdid = SyntheticDiD(variance_method="jackknife", seed=91)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        sens = res.sensitivity_to_zeta_omega(zeta_grid=[])
        assert sens.shape == (0, 5)
        assert list(sens.columns) == [
            "zeta_omega",
            "att",
            "pre_fit_rmse",
            "max_unit_weight",
            "effective_n",
        ]

    def test_empty_multipliers_preserves_schema(self):
        df = _make_panel(seed=93)
        sdid = SyntheticDiD(variance_method="jackknife", seed=93)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        sens = res.sensitivity_to_zeta_omega(multipliers=())
        assert sens.shape == (0, 5)
        assert list(sens.columns) == [
            "zeta_omega",
            "att",
            "pre_fit_rmse",
            "max_unit_weight",
            "effective_n",
        ]


class TestPractitionerSdidReferences:
    """_handle_synthetic in practitioner.py references real callables."""

    def test_snippets_reference_existing_methods(self):
        """Each code snippet in _handle_synthetic() should reference methods
        that actually exist on SyntheticDiDResults."""
        from diff_diff.practitioner import _handle_synthetic
        from diff_diff.results import SyntheticDiDResults

        df = _make_panel(seed=73)
        sdid = SyntheticDiD(variance_method="jackknife", seed=73)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        steps, _ = _handle_synthetic(res)
        expected_methods = [
            "get_weight_concentration",
            "in_time_placebo",
            "get_loo_effects_df",
            "sensitivity_to_zeta_omega",
        ]
        joined_code = "\n".join(step["code"] for step in steps)
        for method in expected_methods:
            assert method in joined_code, f"Expected {method}() referenced in practitioner guidance"
            assert hasattr(SyntheticDiDResults, method), (
                f"Practitioner guidance references {method}() but it's not "
                "on SyntheticDiDResults"
            )

    def test_snippets_parse_as_python(self):
        """Each snippet in _handle_synthetic should be syntactically valid."""
        import ast

        from diff_diff.practitioner import _handle_synthetic

        df = _make_panel(seed=77)
        sdid = SyntheticDiD(variance_method="jackknife", seed=77)
        res = sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")
        steps, _ = _handle_synthetic(res)
        for step in steps:
            ast.parse(step["code"])

    def test_jackknife_loo_snippet_handles_unavailable_loo(self):
        """When variance_method='jackknife' but LOO is unavailable
        (e.g., n_treated=1 returns empty jackknife array), the LOO snippet
        should degrade gracefully instead of raising."""
        from diff_diff.practitioner import _handle_synthetic

        df = _make_panel(n_control=10, n_treated=1, n_pre=5, n_post=3, seed=97)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdid = SyntheticDiD(variance_method="jackknife", seed=97)
            res = sdid.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=list(range(5, 8)),
            )
        assert res.variance_method == "jackknife"
        assert res._loo_unit_ids is None  # LOO intentionally unavailable

        steps, _ = _handle_synthetic(res)
        loo_snippet = next(s["code"] for s in steps if "get_loo_effects_df" in s["code"])
        # Executing the snippet against this result must not raise.
        import contextlib
        import io

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(loo_snippet, {"results": res})
        assert "LOO not available" in captured.getvalue()


class TestSyntheticDiDResultsPickle:
    """Pickle round-trip drops the fit snapshot; diagnostic methods raise
    with the documented recovery message."""

    def _fit(self, seed=101):
        df = _make_panel(seed=seed)
        sdid = SyntheticDiD(variance_method="jackknife", seed=seed)
        return sdid.fit(df, outcome="outcome", treatment="treated", unit="unit", time="period")

    def test_snapshot_dropped_on_pickle(self):
        import pickle

        res = self._fit()
        assert res._fit_snapshot is not None  # present pre-pickle

        restored = pickle.loads(pickle.dumps(res))
        assert restored._fit_snapshot is None
        # Public fields survive
        assert restored.att == res.att
        assert restored.se == res.se
        assert np.allclose(restored.synthetic_pre_trajectory, res.synthetic_pre_trajectory)

    def test_legacy_pickle_state_maps_placebo_effects(self):
        """A result pickled before the placebo_effects → variance_effects
        rename (state carries the old key, no variance_effects) loads with the
        draws under variance_effects and the deprecated alias still surfaces
        them."""
        res = self._fit(seed=109)  # jackknife → variance_effects populated
        effects = np.asarray(res.variance_effects)
        assert effects.size > 0
        # Simulate a <=3.5.x pickle state: old field name, no variance_effects.
        legacy_state = res.__getstate__()
        legacy_state["placebo_effects"] = legacy_state.pop("variance_effects")
        assert "variance_effects" not in legacy_state
        restored = type(res).__new__(type(res))
        restored.__setstate__(legacy_state)
        assert np.allclose(np.asarray(restored.variance_effects), effects)
        with pytest.warns(DeprecationWarning, match="placebo_effects is deprecated"):
            assert np.allclose(np.asarray(restored.placebo_effects), effects)

    def test_in_time_placebo_raises_after_pickle(self):
        import pickle

        res = self._fit(seed=103)
        restored = pickle.loads(pickle.dumps(res))
        with pytest.raises(ValueError, match="fit snapshot"):
            restored.in_time_placebo()

    def test_sensitivity_raises_after_pickle(self):
        import pickle

        res = self._fit(seed=105)
        restored = pickle.loads(pickle.dumps(res))
        with pytest.raises(ValueError, match="fit snapshot"):
            restored.sensitivity_to_zeta_omega()

    def test_live_instance_snapshot_untouched_by_getstate(self):
        """__getstate__ must not mutate the live object's snapshot —
        only the returned state dict carries the nulled field."""
        res = self._fit(seed=107)
        snap_before = res._fit_snapshot
        assert snap_before is not None
        _ = res.__getstate__()
        # Live instance unchanged after __getstate__ call
        assert res._fit_snapshot is snap_before
        # Diagnostics still work in the live session
        _ = res.in_time_placebo(fake_treatment_periods=[2])

    def test_snapshot_excluded_from_dataclass_fields(self):
        """_fit_snapshot and _loo_* must be plain instance attributes, not
        dataclass fields, so dataclass-recursive serializers (asdict,
        fields, replace) cannot reach retained panel state."""
        import dataclasses

        res = self._fit(seed=109)
        field_names = {f.name for f in dataclasses.fields(res)}
        assert "_fit_snapshot" not in field_names
        assert "_loo_unit_ids" not in field_names
        assert "_loo_roles" not in field_names

    def test_asdict_excludes_internal_diagnostic_state(self):
        """dataclasses.asdict() must not recurse into the retained panel
        snapshot or the LOO unit ID arrays."""
        import dataclasses

        res = self._fit(seed=111)
        assert res._fit_snapshot is not None  # live instance retains it
        d = dataclasses.asdict(res)
        assert "_fit_snapshot" not in d
        assert "_loo_unit_ids" not in d
        assert "_loo_roles" not in d


# =============================================================================
# Scale equivariance — internal Y normalization prevents catastrophic
# cancellation when outcomes are on extreme scales (millions-to-billions).
# τ is location-invariant and scale-equivariant, so fitting on ``a*Y + b``
# should produce τ/SE that scale exactly with ``a`` and leave p-values
# unchanged. On normally-scaled data the fix must be a numerical no-op.
# =============================================================================


class TestScaleEquivariance:
    """SDID is location-invariant and scale-equivariant in Y."""

    # Hard-coded baselines captured pre-fix on a well-scaled panel. If these
    # drift the fix is not a true no-op on normal data and review is warranted.
    _BASELINE = {
        # placebo = R-default warm-start (PR follow-up to #349 R-parity
        # work): per-draw FW is initialized with ``sum_normalize(
        # unit_weights[pseudo_control_idx])`` for ω and with the
        # fit-time ``time_weights`` for λ, matching R's
        # ``vcov.R::placebo_se`` ``weights.boot$omega = sum_normalize(
        # weights$omega[ind[1:N0_placebo]])`` warm-start. Drift from
        # the cold-start capture (0.29385822261006445) is the same
        # finite-iter convergence-pattern shift as the bootstrap warm-
        # start landed in PR #349. Warm-start matches R at machine
        # precision (test_placebo_se_matches_r in
        # TestJackknifeSERParity). Captured value is platform-divergent
        # at sub-ULP — warm-start carries ``unit_weights`` (fit-time
        # FW output) into per-draw init, and BLAS reduction-order
        # differences accumulate across 200 draws with path-dependent
        # sparsification to ~1e-9 SE divergence between Apple Accelerate
        # (macOS local: 0.293840360160448) and OpenBLAS (Linux CI:
        # 0.2938403592163006). The placebo row's SE assertion is
        # therefore loosened to ``rel=1e-7`` below; bit-identity
        # protection moves to ``test_placebo_se_matches_r``, which
        # bypasses the platform-divergent fit-time path through a test
        # seam. Pinned value is the Linux/OpenBLAS capture.
        "placebo": (4.603349837478791, 0.2938403592163006, 0.004975124378109453, 200),
        # bootstrap = paper-faithful refit with R-default warm-start: FW is
        # initialized with ``sum_normalize(unit_weights[boot_control_idx])``
        # for ω and with the fit-time ``time_weights`` for λ on each draw,
        # matching R's ``vcov.R::bootstrap_sample`` opts-rebind shape.
        # Drift from the cold-start capture (0.21424970…) is confined to
        # a handful of bootstrap draws where the 100-iter pre-sparsify pass
        # converged to a different sparsification pattern under uniform
        # init; strict-convexity of the FW objective means the converged
        # answer is unique, so the warm-start matches R more faithfully on
        # problems where the pre-sparsify budget is tight.
        "bootstrap": (4.6033498374787865, 0.21427381053829253, 2.2215821875845446e-102, 200),
        "jackknife": (4.603349837478791, 0.19908075946622925, 2.716551077849484e-118, 23),
    }

    # (a, b) pairs. Includes extreme scales where pre-fix SDID loses
    # ~6 mantissa digits in the double-difference subtraction.
    _SCALES = [(1e-6, 0.0), (1.0, 0.0), (1e6, 1e9), (1e9, -1e6), (1.0, 1e9)]

    @staticmethod
    def _rescale(df, a, b):
        out = df.copy()
        out["outcome"] = a * out["outcome"] + b
        return out

    @staticmethod
    def _fit(data, variance_method, *, seed=1, n_bootstrap=200):
        return SyntheticDiD(
            variance_method=variance_method, n_bootstrap=n_bootstrap, seed=seed
        ).fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
        )

    @pytest.mark.parametrize("variance_method", ["placebo", "bootstrap", "jackknife"])
    def test_baseline_parity_small_scale(self, variance_method):
        """Existing-fixture results match captured literals at bit-identity
        tolerance.

        The bootstrap refactor that removed the fixed-weight path reuses the
        same rng.choice sequence and the same compute_sdid_{unit,time}_weights
        + compute_sdid_estimator call chain that the refit branch already ran
        under the previous enum name. Any drift above machine epsilon means
        the numerical path changed, not just a rename — investigate before
        accepting. Placebo / jackknife are untouched by that refactor.
        """
        att0, se0, p0, n0 = self._BASELINE[variance_method]
        data = _make_panel(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = self._fit(data, variance_method)
        assert r.att == pytest.approx(att0, rel=1e-14)
        # Placebo SE is warm-start-driven (PR #369) and the per-draw FW
        # init carries platform-divergent ``unit_weights``; ~1e-9 macOS-
        # vs-Linux drift on 200 draws makes bit-identity unattainable.
        # Bit-identity protection moves to ``test_placebo_se_matches_r``
        # (TestJackknifeSERParity). Bootstrap and jackknife stay at the
        # original gate; their paths don't accumulate the same way.
        se_rel = 1e-7 if variance_method == "placebo" else 1e-14
        assert r.se == pytest.approx(se0, rel=se_rel)
        assert r.p_value == pytest.approx(p0, rel=1e-14)
        assert len(r.variance_effects) == n0

    @pytest.mark.parametrize("variance_method", ["placebo", "bootstrap", "jackknife"])
    def test_scale_equivariance(self, variance_method, ci_params):
        """τ/a, SE/|a|, p-value, and n_successful must be invariant under
        (Y → a*Y + b) across ~15 orders of magnitude."""
        # Pure invariance check (baseline captured at runtime, not vs _BASELINE), so the
        # absolute n_bootstrap is irrelevant: r0 and the scaled refits all use the same
        # (ci_params-scaled in pure-Python, 200 under Rust) count, preserving equivariance.
        nb = ci_params.bootstrap(200)
        data = _make_panel(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r0 = self._fit(data, variance_method, n_bootstrap=nb)
        att0, se0, p0 = r0.att, r0.se, r0.p_value
        n0 = len(r0.variance_effects)
        noise0 = r0.noise_level
        zeta_omega0 = r0.zeta_omega

        for a, b in self._SCALES:
            scaled = self._rescale(data, a, b)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r = self._fit(scaled, variance_method, n_bootstrap=nb)
            # Variance-method success count must be identical; divergence
            # would shift the empirical p-value floor 1/(n+1).
            assert len(r.variance_effects) == n0, (
                f"(a={a}, b={b}) yielded {len(r.variance_effects)} effects, " f"baseline had {n0}"
            )
            assert r.att / a == pytest.approx(att0, rel=1e-8), f"att failed at a={a}, b={b}"
            assert r.se / abs(a) == pytest.approx(se0, rel=1e-6), f"se failed at a={a}, b={b}"
            assert r.p_value == pytest.approx(p0, rel=1e-6), f"p failed at a={a}, b={b}"
            # Reported diagnostics scale with Y just like SE (noise_level,
            # zeta_omega are both in outcome units).
            assert r.noise_level / abs(a) == pytest.approx(noise0, rel=1e-8)
            assert r.zeta_omega / abs(a) == pytest.approx(zeta_omega0, rel=1e-8)

    @pytest.mark.parametrize("variance_method", ["placebo", "bootstrap", "jackknife"])
    def test_detects_true_effect_at_extreme_scale(self, variance_method):
        """Pre-fix regression: catastrophic cancellation at Y~1e9 degraded
        SEs so p-values clustered near 0.5 regardless of true effect. Here
        the true ATT is 0.5% of baseline — below the pre-fix precision
        floor — and must still produce a detectable, correctly-scaled τ."""
        rng = np.random.default_rng(0)
        n_control, n_treated, n_pre, n_post = 25, 3, 6, 4
        baseline_level = 1e9  # deliberately extreme
        true_att = 5e6  # 0.5% of baseline — lost in 6-digit cancellation pre-fix
        rows = []
        for unit in range(n_control + n_treated):
            is_treated = unit >= n_control
            unit_fe = rng.normal(0, 2e6)
            for t in range(n_pre + n_post):
                y = baseline_level + unit_fe + t * 3e5 + rng.normal(0, 5e5)
                if is_treated and t >= n_pre:
                    y += true_att
                rows.append(
                    {
                        "unit": unit,
                        "period": t,
                        "treated": int(is_treated),
                        "outcome": y,
                    }
                )
        data = pd.DataFrame(rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = SyntheticDiD(variance_method=variance_method, n_bootstrap=200, seed=7).fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=list(range(n_pre, n_pre + n_post)),
            )

        # τ must land near the true effect (within ~3 SE); SE must be
        # positive, finite, and small enough that the effect is significant
        # both by z-statistic and by p-value. Placebo uses the empirical null
        # formula (permutations approximate the null distribution); bootstrap
        # and jackknife use the analytical normal-theory p-value from the SE.
        assert np.isfinite(r.att) and np.isfinite(r.se)
        assert r.se > 0
        assert abs(r.att - true_att) < max(3 * r.se, 0.1 * true_att)
        z = abs(r.att / r.se)
        assert z > 3, (
            f"Effect at Y~1e9 must be detectable by z-stat; att={r.att}, "
            f"se={r.se}, z={z} (variance_method={variance_method})"
        )
        assert r.p_value < 0.05, (
            f"Effect at Y~1e9 must reject null; p_value={r.p_value} "
            f"(variance_method={variance_method})"
        )


class TestPValueSemantics:
    """P-value dispatch is variance-method dependent.

    Placebo (Algorithm 4) permutes control indices to generate null-
    distribution draws and uses the empirical formula
    ``max(mean(|draws| >= |att|), 1/(r+1))``. Bootstrap (Algorithm 2)
    resamples units to approximate the sampling distribution of τ̂ —
    draws are centered on τ̂, not 0 — so bootstrap uses the analytical
    normal-theory p-value from the SE. Jackknife pseudo-values are not
    null draws either and also use the analytical p-value.
    """

    def test_bootstrap_p_value_matches_analytical(self, ci_params):
        """Bootstrap p-value must equal safe_inference(att, se)[1]."""
        # Self-consistency check (reported p vs the analytical formula on the reported se) —
        # independent of the bootstrap draw count, so ci_params scaling is safe.
        df = _make_panel(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = SyntheticDiD(
                variance_method="bootstrap", n_bootstrap=ci_params.bootstrap(200), seed=1
            ).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
            )
        _, expected_p, _ = safe_inference(r.att, r.se, alpha=0.05)
        assert (
            abs(r.p_value - expected_p) < 1e-12
        ), f"bootstrap p_value={r.p_value} != analytical {expected_p}"

    def test_placebo_p_value_uses_empirical_formula(self, ci_params):
        """Placebo p-value must equal max(mean(|draws| >= |att|), 1/(r+1))."""
        # Self-consistency check (reported p vs the empirical formula on the reported
        # variance_effects) — independent of the draw count, so ci_params scaling is safe.
        df = _make_panel(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = SyntheticDiD(
                variance_method="placebo", n_bootstrap=ci_params.bootstrap(200), seed=1
            ).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
            )
        variance_effects = np.asarray(r.variance_effects)
        empirical_p = float(np.mean(np.abs(variance_effects) >= np.abs(r.att)))
        expected_p = max(empirical_p, 1.0 / (len(variance_effects) + 1))
        assert (
            abs(r.p_value - expected_p) < 1e-12
        ), f"placebo p_value={r.p_value} != empirical {expected_p}"

    def test_bootstrap_p_value_detects_large_effect(self):
        """Bootstrap p-value must reject decisively when z is large.

        Regression guard: pre-fix this would return ~0.5 regardless of
        effect size because the empirical null formula was applied to
        draws centered on τ̂.
        """
        df = _make_panel(att=5.0, seed=123)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = SyntheticDiD(variance_method="bootstrap", n_bootstrap=200, seed=1).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
            )
        z = abs(r.att / r.se)
        assert z > 6, f"setup error: z={z} too small to test rejection"
        assert r.p_value < 1e-6, f"bootstrap p_value={r.p_value} too large at z={z}"

    @pytest.mark.slow
    def test_bootstrap_p_value_null_dispersion(self):
        """Bootstrap p-values on null data must be dispersed and fall in a
        loose calibration-agnostic band — regression guard against the
        pre-fix p-clustering dispatch bug and against SE-collapse.

        Semantic: this is a calibration-agnostic regression test, not a
        nominal-calibration assertion. Under paper-faithful refit
        bootstrap (Arkhangelsky et al. 2021 Algorithm 2 step 2) the
        REGISTRY-cited coverage MC puts α=0.05 rejection near nominal on
        this DGP (≈0.08 at 500 seeds × B=200); the previous fixed-weight
        path over-rejected at ~0.18. We deliberately do NOT assert
        directionally on rejection rate because the reviewer's P2 was
        that the prior lower bound (`> 0.05`) biased the test toward
        anti-conservative behavior.

        Assertions (calibration-agnostic):

        - ``np.std(p_values) > 0.10``: dispersion floor — catches the
          pre-fix dispatch bug where p clustered at ~0.5 on every seed
          (std → 0).
        - ``0.01 <= rejection_rate <= 0.40``: loose band — catches both
          SE-collapse (rate → 1) and SE-explosion (rate → 0).
        """
        p_values = []
        for seed in range(100):
            df = _make_panel(att=0.0, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r = SyntheticDiD(
                    variance_method="bootstrap",
                    n_bootstrap=200,
                    seed=seed,
                ).fit(
                    df,
                    outcome="outcome",
                    treatment="treated",
                    unit="unit",
                    time="period",
                    post_periods=[5, 6, 7],
                )
            if np.isfinite(r.p_value):
                p_values.append(r.p_value)
        p_arr = np.asarray(p_values)
        assert len(p_arr) >= 90, f"only {len(p_arr)}/100 fits produced finite p-values"

        p_std = float(np.std(p_arr))
        assert p_std > 0.10, (
            f"p-value std {p_std:.3f} <= 0.10 — p-values too tightly "
            "clustered (pre-fix dispatch bug regression; fixed-weight→"
            "refit porting bug would manifest as p≈0.5 on every seed)"
        )

        rejection_rate = float(np.mean(p_arr < 0.05))
        assert 0.01 <= rejection_rate <= 0.40, (
            f"rejection rate {rejection_rate:.3f} outside loose "
            "calibration-agnostic band [0.01, 0.40] — likely SE-collapse "
            "or SE-explosion regression (compare against the REGISTRY "
            "coverage MC table for the nominal band on this DGP)"
        )


class TestDiagnosticScaleParity:
    """Post-PR #312: in_time_placebo() and sensitivity_to_zeta_omega() must
    inherit the same original-scale normalization contract that the main fit
    path uses. Both diagnostics re-run Frank-Wolfe on the stored fit-snapshot
    arrays, so at extreme Y they previously re-created the catastrophic
    cancellation PR #312 fixed on the main path (audit finding D-4)."""

    _SCALES = [(1.0, 0.0), (1e6, 1e9), (1e9, -1e6)]

    @staticmethod
    def _rescale(df, a, b):
        out = df.copy()
        out["outcome"] = a * out["outcome"] + b
        return out

    @staticmethod
    def _fit(data, seed=1):
        return SyntheticDiD(variance_method="jackknife", seed=seed).fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
        )

    def test_in_time_placebo_scale_equivariance(self):
        """in_time_placebo att/pre_fit_rmse must scale by |a| across
        (Y → a*Y + b). Pre-fix at extreme scale the diagnostic re-ran FW on
        original-scale snapshot arrays and cancellation corrupted att."""
        data = _make_panel(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r0 = self._fit(data)
        placebo0 = r0.in_time_placebo()
        fake_periods = placebo0["fake_treatment_period"].tolist()

        for a, b in self._SCALES:
            scaled = self._rescale(data, a, b)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r = self._fit(scaled)
            placebo = r.in_time_placebo(fake_treatment_periods=fake_periods)

            assert list(placebo["fake_treatment_period"]) == fake_periods
            for row0, row in zip(placebo0.to_dict("records"), placebo.to_dict("records")):
                if np.isnan(row0["att"]):
                    assert np.isnan(row["att"]), f"att at (a={a}, b={b})"
                    assert np.isnan(row["pre_fit_rmse"])
                    continue
                assert row["att"] / a == pytest.approx(row0["att"], rel=1e-6), (
                    f"att at (a={a}, b={b}), " f"fake_period={row0['fake_treatment_period']}"
                )
                assert row["pre_fit_rmse"] / abs(a) == pytest.approx(
                    row0["pre_fit_rmse"], rel=1e-6
                ), (
                    f"pre_fit_rmse at (a={a}, b={b}), "
                    f"fake_period={row0['fake_treatment_period']}"
                )

    def test_sensitivity_to_zeta_omega_scale_equivariance(self):
        """sensitivity_to_zeta_omega att/pre_fit_rmse must scale by |a|;
        unit-weight diagnostics (max_unit_weight, effective_n) must be
        scale-invariant."""
        data = _make_panel(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r0 = self._fit(data)
        sens0 = r0.sensitivity_to_zeta_omega()
        zeta_grid = sens0["zeta_omega"].tolist()

        for a, b in self._SCALES:
            scaled = self._rescale(data, a, b)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r = self._fit(scaled)
            sens = r.sensitivity_to_zeta_omega(zeta_grid=[a * z for z in zeta_grid])
            for row0, row in zip(sens0.to_dict("records"), sens.to_dict("records")):
                assert row["att"] / a == pytest.approx(
                    row0["att"], rel=1e-6
                ), f"att at (a={a}, b={b}), zeta={row0['zeta_omega']}"
                assert row["pre_fit_rmse"] / abs(a) == pytest.approx(row0["pre_fit_rmse"], rel=1e-6)
                # omega-derived diagnostics are scale-invariant.
                assert row["max_unit_weight"] == pytest.approx(row0["max_unit_weight"], rel=1e-6)
                assert row["effective_n"] == pytest.approx(row0["effective_n"], rel=1e-6)

    def test_in_time_placebo_detectable_at_extreme_scale(self):
        """Pre-fix regression: at Y~1e9 the placebo re-fit corrupted ATTs via
        cancellation so diagnostic numbers were garbage. Post-fix, all
        placebo rows on a zero-effect DGP must be finite and at least one
        must land within 5*noise_level in original-Y units."""
        rng = np.random.default_rng(0)
        n_control, n_treated, n_pre, n_post = 20, 3, 7, 3
        baseline_level = 1e9
        rows = []
        for unit in range(n_control + n_treated):
            unit_fe = rng.normal(0, 2e6)
            for t in range(n_pre + n_post):
                y = baseline_level + unit_fe + t * 3e5 + rng.normal(0, 5e5)
                rows.append(
                    {"unit": unit, "period": t, "treated": int(unit >= n_control), "outcome": y}
                )
        data = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = self._fit(data, seed=7)

        placebo = r.in_time_placebo()
        finite = placebo.dropna(subset=["att"])
        assert len(finite) > 0
        assert np.all(np.isfinite(finite["att"].values))
        assert np.all(np.isfinite(finite["pre_fit_rmse"].values))
        assert (np.abs(finite["att"]) < 5.0 * r.noise_level).any(), (
            f"At least one placebo row should be within 5*noise_level "
            f"({5.0 * r.noise_level}); got atts {finite['att'].tolist()}"
        )


class TestDiagnosticSnapshotBackwardCompat:
    """Locks in the backward-compatibility contract for legacy
    _SyntheticDiDFitSnapshot objects that pre-date Y_shift/Y_scale. Their
    defaults (0.0, 1.0) must make the new normalization a pure no-op so
    older cached snapshots still drive diagnostic refits unchanged."""

    def test_legacy_snapshot_defaults_are_noop(self):
        from diff_diff.results import _SyntheticDiDFitSnapshot

        data = _make_panel(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = SyntheticDiD(variance_method="jackknife", seed=1).fit(
                data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                post_periods=[5, 6, 7],
            )

        # Baseline diagnostic output with the real (fit-captured) normalization.
        placebo0 = r.in_time_placebo()
        sens0 = r.sensitivity_to_zeta_omega()

        # Overwrite the snapshot with a legacy one built without Y_shift /
        # Y_scale — the defaults must make the two diagnostic paths produce
        # the same output as the fit-captured version, because the main
        # fit's Y-shift/scale choice is a no-op on a small, well-scaled
        # panel (Y_shift ~ 10, Y_scale ~ O(1), so (Y - shift)/scale is just
        # a shifted/scaled copy of Y).
        snap = r._fit_snapshot
        legacy_snap = _SyntheticDiDFitSnapshot(
            Y_pre_control=np.array(snap.Y_pre_control),
            Y_post_control=np.array(snap.Y_post_control),
            Y_pre_treated=np.array(snap.Y_pre_treated),
            Y_post_treated=np.array(snap.Y_post_treated),
            control_unit_ids=list(snap.control_unit_ids),
            treated_unit_ids=list(snap.treated_unit_ids),
            pre_periods=list(snap.pre_periods),
            post_periods=list(snap.post_periods),
            w_control=snap.w_control,
            w_treated=snap.w_treated,
            # Defaults — no Y_shift/Y_scale captured.
        )
        # Confirm the defaults are what we expect.
        assert legacy_snap.Y_shift == 0.0
        assert legacy_snap.Y_scale == 1.0

        r._fit_snapshot = legacy_snap
        placebo_legacy = r.in_time_placebo()
        sens_legacy = r.sensitivity_to_zeta_omega()

        # Shape and columns must match.
        assert list(placebo_legacy.columns) == list(placebo0.columns)
        assert list(sens_legacy.columns) == list(sens0.columns)
        assert len(placebo_legacy) == len(placebo0)
        assert len(sens_legacy) == len(sens0)


class TestHeterogeneousAndRampingScale:
    """D-4b: the existing TestScaleEquivariance suite is affine-only
    (Y → a*Y + b with a single scalar a). These pathways are not covered:

    - Cross-unit heterogeneous scale: different units span 1e6 to 1e9.
    - Cross-period ramping: baseline trend growing several orders of
      magnitude across periods.

    Both were candidate triggers for the original SDID silent-failure report
    and must stay detectable after any future refactor of the normalization
    contract."""

    @staticmethod
    def _fit(data, seed=1):
        return SyntheticDiD(variance_method="jackknife", seed=seed).fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=[5, 6, 7],
        )

    def test_cross_unit_heterogeneous_scale(self):
        """Units spanning 1e6 to 1e9 must still produce finite fit and
        diagnostic output. Heterogeneous levels historically triggered the
        cancellation pathway even at modest Y; this is a regression trap
        for future refactors of the normalization contract."""
        rng = np.random.default_rng(11)
        n_control, n_treated, n_pre, n_post = 20, 3, 6, 3
        rows = []
        for unit in range(n_control + n_treated):
            unit_level = 10 ** rng.uniform(6, 9)
            is_treated = unit >= n_control
            for t in range(n_pre + n_post):
                y = unit_level * (1 + 0.02 * t) + rng.normal(0, unit_level * 0.01)
                if is_treated and t >= n_pre:
                    y += 0.05 * unit_level
                rows.append({"unit": unit, "period": t, "treated": int(is_treated), "outcome": y})
        data = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = self._fit(data)

        assert np.isfinite(r.att) and np.isfinite(r.se)
        assert r.se > 0
        placebo = r.in_time_placebo()
        assert np.all(np.isfinite(placebo["att"].dropna()))
        assert np.all(np.isfinite(placebo["pre_fit_rmse"].dropna()))
        sens = r.sensitivity_to_zeta_omega()
        assert np.all(np.isfinite(sens["att"]))
        assert np.all(np.isfinite(sens["pre_fit_rmse"]))

    def test_cross_period_ramping_trend(self):
        """A strong cross-period trend (baseline level multiplies across
        periods) must still produce a detectable, finite ATT and finite
        diagnostic output."""
        rng = np.random.default_rng(13)
        n_control, n_treated, n_pre, n_post = 20, 3, 6, 3
        rows = []
        for unit in range(n_control + n_treated):
            unit_fe = rng.normal(0, 1.0)
            is_treated = unit >= n_control
            for t in range(n_pre + n_post):
                trend = 10 ** (5 + 0.4 * t)
                y = trend + unit_fe * trend * 0.01 + rng.normal(0, trend * 0.005)
                if is_treated and t >= n_pre:
                    y += 0.01 * trend
                rows.append({"unit": unit, "period": t, "treated": int(is_treated), "outcome": y})
        data = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = self._fit(data)

        assert np.isfinite(r.att) and np.isfinite(r.se)
        assert r.se > 0
        placebo = r.in_time_placebo()
        assert np.all(np.isfinite(placebo["att"].dropna()))
        assert np.all(np.isfinite(placebo["pre_fit_rmse"].dropna()))
        sens = r.sensitivity_to_zeta_omega()
        assert np.all(np.isfinite(sens["att"]))
        assert np.all(np.isfinite(sens["pre_fit_rmse"]))


# =============================================================================
# Coverage MC calibration artifact (generated by benchmarks/python/coverage_sdid.py)
# =============================================================================


class TestCoverageMCArtifact:
    """Schema smoke-check on ``benchmarks/data/sdid_coverage.json``.

    The full Monte Carlo study (500 seeds × B=200 × 4 DGPs × 3 methods)
    runs outside CI; its JSON output underwrites the calibration table in
    REGISTRY.md §SyntheticDiD. The 4th DGP (``stratified_survey``)
    exercises the bootstrap survey-composition path (PR #355) and the
    jackknife PSU-level LOO path (this PR); placebo is structurally
    infeasible on this DGP because its cohort packs into a single stratum
    with 0 never-treated units, so the harness skips placebo for the
    `stratified_survey` block. This test verifies the artifact is present
    and structured correctly. Per ``feedback_golden_file_pytest_skip.md``,
    skip if missing — CI's isolated-install job copies only ``tests/``,
    not ``benchmarks/``.
    """

    def test_coverage_artifacts_present(self):
        import json
        import pathlib

        artifact = (
            pathlib.Path(__file__).parent.parent / "benchmarks" / "data" / "sdid_coverage.json"
        )
        if not artifact.exists():
            pytest.skip(
                f"Missing coverage MC artifact {artifact}; regenerate via "
                "`python benchmarks/python/coverage_sdid.py --n-seeds 500 "
                "--n-bootstrap 200 --output benchmarks/data/sdid_coverage.json`."
            )
        payload = json.loads(artifact.read_text())

        for key in ("metadata", "dgps", "per_dgp"):
            assert key in payload, f"missing top-level key: {key}"

        meta = payload["metadata"]
        for key in (
            "n_seeds",
            "n_bootstrap",
            "library_version",
            "backend",
            "generated_at",
            "methods",
            "alphas",
        ):
            assert key in meta, f"missing metadata key: {key}"
        assert meta["n_seeds"] >= 100, (
            f"n_seeds={meta['n_seeds']} too small; the REGISTRY calibration "
            "table cites 500-seed rates — regenerate with documented settings."
        )
        assert set(meta["methods"]) == {"placebo", "bootstrap", "jackknife"}, (
            "coverage artifact methods list must be exactly the 3 supported "
            "SDID variance methods (placebo / bootstrap / jackknife); the old "
            "fixed-weight 'bootstrap' and the additive 'bootstrap_refit' "
            "enum value are both gone."
        )

        for dgp in ("balanced", "unbalanced", "aer63", "stratified_survey"):
            assert dgp in payload["per_dgp"], f"missing DGP block: {dgp}"
            per_method = payload["per_dgp"][dgp]
            for method in ("placebo", "bootstrap", "jackknife"):
                assert method in per_method, f"missing method block {method!r} under DGP {dgp!r}"
                block = per_method[method]
                for field in (
                    "rejection_rate",
                    "mean_se",
                    "true_sd_tau_hat",
                    "se_over_truesd",
                    "n_successful_fits",
                ):
                    assert field in block, f"missing field {field!r} in {dgp}/{method}"
                for alpha_key in ("0.01", "0.05", "0.10"):
                    assert (
                        alpha_key in block["rejection_rate"]
                    ), f"missing alpha {alpha_key} in {dgp}/{method} rejection_rate"

        # Post-PR #365: stratified_survey runs bootstrap (validation
        # gate) + jackknife (anti-conservative but reported for
        # transparency); placebo is skipped on this DGP because its
        # cohort packs all treated into stratum 1 which has 0 never-
        # treated units (Case B at fit-time), so its block reports
        # n_successful_fits=0. Bootstrap must have the full 500
        # successful fits + finite rejection rate at α=0.05 inside the
        # calibration gate [0.02, 0.10].
        survey_block = payload["per_dgp"]["stratified_survey"]
        assert survey_block["bootstrap"]["n_successful_fits"] >= 100, (
            "stratified_survey bootstrap must have ≥100 successful fits; "
            "the survey-bootstrap path is broken if this drops to 0."
        )
        rej_05 = survey_block["bootstrap"]["rejection_rate"]["0.05"]
        assert 0.02 <= rej_05 <= 0.10, (
            f"stratified_survey bootstrap α=0.05 rejection {rej_05} outside "
            "calibration gate [0.02, 0.10]; weighted FW + Rao-Wu is "
            "miscalibrated. See PR #355 §3c rollback protocol."
        )
        # Placebo is structurally infeasible on this DGP: the DGP packs
        # all treated units into stratum 1, which has 0 never-treated
        # units, so the stratified-permutation allocator raises Case B
        # (zero controls in a treated-containing stratum) at fit-time.
        assert survey_block["placebo"]["n_successful_fits"] == 0, (
            "stratified_survey placebo should have 0 successful fits "
            "(stratified-permutation allocator raises Case B at fit-time "
            "because stratum 1 has 0 never-treated units — all treated "
            "cohort packs into stratum 1 by DGP construction)."
        )
        # Jackknife should now succeed (full-design support added). Its SE
        # is known anti-conservative with only 2 PSUs per stratum — that's
        # a methodology limitation documented in REGISTRY, not a regression.
        # Here we just check that the fit returned a finite SE (survey path
        # dispatched correctly); calibration-gate bands are intentionally
        # not asserted for jackknife on this DGP.
        assert survey_block["jackknife"]["n_successful_fits"] >= 100, (
            "stratified_survey jackknife must have ≥100 successful fits; "
            "the PSU-level LOO + stratum aggregation path is broken if "
            "this drops to 0."
        )
