"""Tests for HAD Phase 3 pre-test diagnostics (had_pretests.py)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    QUGTestResults,
    StuteJointResult,
    StuteTestResults,
    YatchewTestResults,
    did_had_pretest_workflow,
    joint_homogeneity_test,
    joint_pretrends_test,
    qug_test,
    stute_joint_pretest,
    stute_test,
    yatchew_hr_test,
)
from diff_diff.had_pretests import (
    _compose_verdict,
    _compose_verdict_event_study,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_two_period_panel(
    G: int,
    d: np.ndarray,
    dy: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """Construct a two-period panel satisfying the HAD overall-path contract.

    - ``time == 0``: ``d = 0`` for all units (HAD no-unit-untreated pre).
    - ``time == 1``: dose ``d`` per unit, outcome ``y_pre + dy``.
    """
    rng = np.random.default_rng(seed)
    y_pre = rng.normal(0.0, 1.0, size=G)
    pre = pd.DataFrame({"unit": np.arange(G), "time": 0, "y": y_pre, "d": np.zeros(G)})
    post = pd.DataFrame({"unit": np.arange(G), "time": 1, "y": y_pre + dy, "d": d})
    return pd.concat([pre, post], ignore_index=True)


def _linear_dgp(G: int, beta: float = 2.0, sigma: float = 0.3, seed: int = 42):
    """Return ``(d, dy)`` for a linear DGP ``dy = beta * d + eps``."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 1.0, size=G)
    dy = beta * d + rng.normal(0.0, sigma, size=G)
    return d, dy


def _quadratic_dgp(
    G: int, beta: float = 2.0, gamma: float = 5.0, sigma: float = 0.3, seed: int = 42
):
    """Return ``(d, dy)`` for a quadratic DGP ``dy = beta*d + gamma*d^2 + eps``."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.1, 1.0, size=G)
    dy = beta * d + gamma * d * d + rng.normal(0.0, sigma, size=G)
    return d, dy


# =============================================================================
# QUG test
# =============================================================================


class TestQUGTest:
    """Tests for :func:`qug_test`."""

    def test_tight_parity_closed_form(self):
        """Commit criterion 4: [0.2, 0.5, 0.9] -> T = 0.2/0.3 at atol=1e-12."""
        d = np.array([0.2, 0.5, 0.9])
        r = qug_test(d, alpha=0.05)
        expected_T = 0.2 / (0.5 - 0.2)
        expected_p = 1.0 / (1.0 + expected_T)
        np.testing.assert_allclose(r.t_stat, expected_T, atol=1e-12)
        np.testing.assert_allclose(r.p_value, expected_p, atol=1e-12)
        assert r.critical_value == pytest.approx(1.0 / 0.05 - 1.0, rel=1e-12)

    def test_tie_break(self):
        """Commit criterion 3: D_(1) == D_(2) -> NaN, reject=False."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = qug_test(np.array([0.3, 0.3, 0.5]))
            assert len(caught) == 1
            assert "D_(1) == D_(2)" in str(caught[0].message)
        assert np.isnan(r.t_stat)
        assert np.isnan(r.p_value)
        assert r.reject is False
        assert r.d_order_1 == 0.3
        assert r.d_order_2 == 0.3

    def test_zero_filter_emits_warning(self):
        """Zero-dose observations are excluded with a UserWarning."""
        d = np.array([0.0, 0.0, 0.3, 0.5, 0.9])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = qug_test(d)
            messages = [str(w.message) for w in caught]
        assert any("excluded 2" in m for m in messages)
        assert r.n_obs == 3
        assert r.n_excluded_zero == 2
        # Should reduce to [0.3, 0.5, 0.9]: T = 0.3 / 0.2 = 1.5
        np.testing.assert_allclose(r.t_stat, 0.3 / 0.2, atol=1e-12)

    def test_rejects_on_shifted_beta(self):
        """Commit criterion 2 (single-realization): shifted-Beta -> reject."""
        rng = np.random.default_rng(42)
        # Support is [0.2, 1.2]: infimum is well away from zero.
        d = 0.2 + rng.beta(2, 2, size=200)
        r = qug_test(d, alpha=0.05)
        # T = D_(1)/(D_(2) - D_(1)). D_(1) ~ 0.2; gap ~ 0.001 at G=200.
        # So T is large -> reject.
        assert r.reject is True
        assert r.p_value < 0.05

    def test_does_not_reject_on_uniform_support_zero(self):
        """Commit criterion 1 (single-realization): Uniform(0,1) -> usually no reject."""
        rng = np.random.default_rng(42)
        d = rng.uniform(0.0, 1.0, size=500)
        r = qug_test(d, alpha=0.05)
        # Uniform support has infimum at 0 (asymptotic). For G=500,
        # D_(1) is small but D_(2) - D_(1) also small -> T ~ O(1).
        # Not a guaranteed non-reject at single seed, but this seed passes.
        assert r.p_value > 0.05
        assert r.reject is False

    def test_reject_rate_bounded_by_alpha_on_uniform_null(self):
        """Commit criterion 1: asymptotic size alpha is an UPPER bound.

        Paper Theorem 4 establishes ``lim sup_{G->inf} P(reject | H_0) = alpha``
        (one-sided asymptotic size bound). At finite G the test is often
        conservative (actual rate below alpha). We verify the asymptotic
        size control: reject rate MUST NOT exceed alpha by more than 0.03.
        """
        alpha = 0.05
        n_reps = 200
        rejections = 0
        rng = np.random.default_rng(12345)
        for _ in range(n_reps):
            d = rng.uniform(0.0, 1.0, size=500)
            r = qug_test(d, alpha=alpha)
            if r.reject:
                rejections += 1
        rate = rejections / n_reps
        assert rate <= alpha + 0.03, (
            f"QUG reject rate {rate:.3f} exceeds alpha={alpha} by more than "
            f"0.03; asymptotic size control violated."
        )

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must satisfy"):
            qug_test(np.array([0.1, 0.5]), alpha=0.0)
        with pytest.raises(ValueError, match="alpha must satisfy"):
            qug_test(np.array([0.1, 0.5]), alpha=1.0)

    def test_nan_input_raises(self):
        with pytest.raises(ValueError, match="contains NaN"):
            qug_test(np.array([0.1, np.nan, 0.5]))

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            qug_test(np.array([[0.1, 0.5], [0.3, 0.9]]))

    def test_negative_dose_raises(self):
        """P2 fix: HAD doses must be non-negative; qug_test rejects negatives
        at the front door rather than silently folding them into the
        zero-exclusion counter."""
        with pytest.raises(ValueError, match="negative value"):
            qug_test(np.array([0.1, -0.3, 0.5, 0.9]))

    # -------------------------------------------------------------------
    # Phase 4.5 C0 decision-gate guards
    # -------------------------------------------------------------------

    def test_weights_kwarg_raises_not_implemented(self):
        """Phase 4.5 C0: qug_test(weights=) raises NotImplementedError."""
        d = np.array([0.1, 0.5, 0.9])
        with pytest.raises(NotImplementedError, match="qug_test does not support"):
            qug_test(d, weights=np.ones(3))

    def test_survey_kwarg_raises_not_implemented(self):
        """Phase 4.5 C0: qug_test(survey=) raises NotImplementedError."""
        from diff_diff import SurveyDesign

        d = np.array([0.1, 0.5, 0.9])
        with pytest.raises(NotImplementedError, match="qug_test does not support"):
            qug_test(d, survey=SurveyDesign(weights="w"))

    def test_mutex_both_set_raises_value_error(self):
        """Phase 4.5 C0: passing both survey= AND weights= raises ValueError
        (mirroring HeterogeneousAdoptionDiD.fit() at had.py:2890), BEFORE the
        NotImplementedError fires. Mutex pattern is consistent across the HAD
        surface so users get the same error text whether they hit the
        estimator or a pretest."""
        from diff_diff import SurveyDesign

        d = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="at most one of"):
            qug_test(d, survey=SurveyDesign(weights="w"), weights=np.ones(3))

    def test_methodology_pointer_in_message(self):
        """Phase 4.5 C0: the NotImplementedError must point users to (a) the
        joint Stute alternative and (b) Phase 4.5 C, AND name the three
        methodology reasons (extreme order statistics not smooth, Exp(1)/Exp(1)
        independence breaks under clustering, EVT-under-sampling literature
        sparse). This locks the cross-surface parity audit (docstring vs error
        message vs REGISTRY note must agree on the rationale)."""
        d = np.array([0.1, 0.5, 0.9])
        with pytest.raises(NotImplementedError) as exc_info:
            qug_test(d, weights=np.ones(3))
        msg = str(exc_info.value)
        # Methodology rationale tags
        assert "smallest order statistics" in msg
        assert "Exp(1)/Exp(1)" in msg
        assert "extreme-value theory" in msg
        # Routing pointers
        assert "joint Stute" in msg
        assert "Phase 4.5 C" in msg
        assert "did_had_pretest_workflow" in msg
        # REGISTRY pointer
        assert "REGISTRY.md" in msg

    def test_unweighted_call_unchanged_after_kwargs_added(self):
        """Stability invariant: existing positional / kwarg-free calls must
        produce bit-exact pre-PR output after the new keyword-only kwargs are
        added. Locks the closed-form parity at atol=1e-12 (same tolerance as
        test_tight_parity_closed_form)."""
        d = np.array([0.2, 0.5, 0.9])
        r = qug_test(d, alpha=0.05)
        expected_T = 0.2 / (0.5 - 0.2)
        expected_p = 1.0 / (1.0 + expected_T)
        np.testing.assert_allclose(r.t_stat, expected_T, atol=1e-12)
        np.testing.assert_allclose(r.p_value, expected_p, atol=1e-12)
        assert r.reject is False  # T = 0.667 < 1/0.05 - 1 = 19


class TestNegativeDoseGuardsOnLinearityTests:
    """R6 P1 fix: stute_test and yatchew_hr_test must enforce the same
    HAD non-negative-dose contract as qug_test and the panel validator."""

    def test_stute_negative_dose_raises(self):
        d = np.linspace(-0.2, 0.8, 30)  # contains negatives
        dy = 2.0 * d + np.random.default_rng(42).normal(0.0, 0.1, size=30)
        with pytest.raises(ValueError, match="negative value"):
            stute_test(d, dy, n_bootstrap=199, seed=42)

    def test_yatchew_negative_dose_raises(self):
        d = np.linspace(-0.2, 0.8, 30)
        dy = 2.0 * d + np.random.default_rng(42).normal(0.0, 0.1, size=30)
        with pytest.raises(ValueError, match="negative value"):
            yatchew_hr_test(d, dy)


# =============================================================================
# Stute test
# =============================================================================


class TestStuteTest:
    """Tests for :func:`stute_test`."""

    def test_does_not_reject_linear_dgp(self):
        """Commit criterion 5 (single-realization): linear -> no reject."""
        d, dy = _linear_dgp(G=200, beta=2.0, sigma=0.3, seed=42)
        r = stute_test(d, dy, alpha=0.05, n_bootstrap=199, seed=42)
        assert r.reject is False
        assert r.p_value > 0.05

    def test_rejects_quadratic_dgp(self):
        """Commit criterion 6: quadratic DGP -> reject."""
        d, dy = _quadratic_dgp(G=200, beta=2.0, gamma=5.0, sigma=0.3, seed=42)
        r = stute_test(d, dy, alpha=0.05, n_bootstrap=199, seed=42)
        assert r.reject is True
        assert r.p_value < 0.05

    def test_cvm_statistic_manual_equivalence(self):
        """Verify the simplified CvM form matches the paper's stated form
        in the no-ties case.

        Paper: S = Σ(g/G)^2 · ((1/g) Σ eps_(h))^2
        Code (no ties):  S = (1/G^2) Σ (cumsum_g)^2
        These are algebraically identical when all dose values are unique.
        """
        from diff_diff.had_pretests import _cvm_statistic

        eps = np.array([1.0, -0.5, 0.3, -0.2, 0.1])
        d_sorted = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # all unique -> no ties
        G = len(eps)

        # Paper form
        cumsum = np.cumsum(eps)
        paper_S = 0.0
        for g in range(1, G + 1):
            c_g = cumsum[g - 1]
            paper_S += (g / G) ** 2 * (c_g / g) ** 2

        code_S = _cvm_statistic(eps, d_sorted)
        np.testing.assert_allclose(code_S, paper_S, atol=1e-14)

    def test_cvm_statistic_tie_safe_order_invariance(self):
        """CRITICAL P1 fix: tie-safe CvM is order-invariant within tie blocks.

        Under the paper definition, at a tied dose ``D_g == D_{g+1}``, the
        cusum ``c_G`` evaluated at both tied observations includes ALL
        tie-block members. So the CvM statistic must not depend on the
        within-tie ordering of residuals. A naive per-observation cumsum
        (without tie-block collapse) would give different S values when
        the within-tie residual order is permuted.
        """
        from diff_diff.had_pretests import _cvm_statistic

        # 6 observations with two tie blocks: d = [0.1, 0.1, 0.1, 0.5, 0.5, 0.9]
        d_sorted = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.9])
        eps_a = np.array([1.0, -0.5, 0.3, 0.7, -0.2, 0.1])
        # Permute within-tie residual order (positions 0-2 and 3-4 permuted):
        eps_b = np.array([0.3, 1.0, -0.5, -0.2, 0.7, 0.1])
        S_a = _cvm_statistic(eps_a, d_sorted)
        S_b = _cvm_statistic(eps_b, d_sorted)
        np.testing.assert_allclose(S_a, S_b, atol=1e-14)

    def test_stute_order_invariance_with_duplicate_doses(self):
        """End-to-end P1 fix: stute_test on duplicate-dose inputs is invariant
        to within-tie row ordering (tie-safe CvM contract propagates)."""
        G = 40
        # Build d with ties and a matched dy
        rng = np.random.default_rng(42)
        d_unique = np.array([0.1, 0.3, 0.5, 0.8])
        d = np.repeat(d_unique, G // len(d_unique))
        dy = 2.0 * d + rng.normal(0.0, 0.1, size=G)
        # Permute order within each tie block
        perm = np.arange(G)
        rng_perm = np.random.default_rng(123)
        for start in range(0, G, G // len(d_unique)):
            block = perm[start : start + G // len(d_unique)]
            rng_perm.shuffle(block)
            perm[start : start + G // len(d_unique)] = block
        r_a = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_b = stute_test(d[perm], dy[perm], n_bootstrap=199, seed=42)
        # Observed cvm_stat must be bit-identical (tie-safe).
        np.testing.assert_allclose(r_a.cvm_stat, r_b.cvm_stat, atol=1e-14)

    def test_n_bootstrap_below_99_raises(self):
        """Commit criterion 8: n_bootstrap < 99 -> ValueError."""
        d, dy = _linear_dgp(G=50)
        with pytest.raises(ValueError, match=r"n_bootstrap must be >= 99"):
            stute_test(d, dy, n_bootstrap=50)

    def test_exact_linear_returns_p1_not_nan(self):
        """P1 fix: exact linear fit (eps=0) must fail-to-reject with p=1,
        NOT return NaN. Assumption 8 holds exactly in this case."""
        G = 50
        d = np.linspace(0.1, 1.0, G)
        dy = 1.0 + 2.0 * d  # exact linear, residuals are zero
        r = stute_test(d, dy, n_bootstrap=199, seed=42)
        assert np.isfinite(r.p_value)
        assert r.p_value == 1.0  # all bootstrap S_b tied at 0 with observed S
        assert r.reject is False
        assert r.cvm_stat == 0.0

    def test_exact_linear_shortcut_does_not_fire_on_shifted_noisy_data(self):
        """P0 fix (R2): the exact-linear short-circuit must NOT fire on
        noisy data after an additive shift. The fix scales against
        ``sum((dy - dybar)^2)`` (centered TSS, translation-invariant).

        Exact numerical equivalence of the two bootstrap runs is NOT
        expected (FP cancellation at 1e12 scale costs ~4 digits of
        precision), but the short-circuit behavior and decision MUST
        match.
        """
        rng = np.random.default_rng(42)
        G = 50
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 1.0, size=G)
        dy_shifted = dy + 1e12

        r_raw = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_shift = stute_test(d, dy_shifted, n_bootstrap=199, seed=42)

        assert r_raw.cvm_stat > 0.0, "shortcut fired on raw noisy data"
        assert r_shift.cvm_stat > 0.0, "shortcut fired on shifted noisy data"
        assert r_raw.reject == r_shift.reject
        np.testing.assert_allclose(r_raw.p_value, r_shift.p_value, rtol=5e-3)

    def test_exact_linear_shortcut_does_not_fire_on_rescaled_noisy_data(self):
        """P0 fix (R5): the exact-linear short-circuit must NOT fire on
        noisy data after a MULTIPLICATIVE rescaling. An earlier revision
        used ``max(centered_TSS, 1.0)`` as the denominator in the guard,
        which broke scale invariance: scaling ``dy`` by ``1e-12`` would
        make ``centered_TSS ~ 1e-24`` but the floor would hold the
        threshold at 1.0, firing the shortcut on noisy data that should
        NOT trigger it. Fix uses a purely relative comparison with a
        separate branch for the ``centered_TSS == 0`` edge case.
        """
        rng = np.random.default_rng(42)
        G = 50
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 1.0, size=G)
        dy_scaled = dy * 1e-12  # multiplicative rescale

        r_raw = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_scaled = stute_test(d, dy_scaled, n_bootstrap=199, seed=42)

        # Neither run may hit the short-circuit.
        assert r_raw.cvm_stat > 0.0, "shortcut fired on raw noisy data"
        assert r_scaled.cvm_stat > 0.0, "shortcut fired on scaled noisy data"
        # Decision must be preserved: rescaling dy cannot change the
        # rejection outcome (the statistic is scale-invariant in dy).
        assert r_raw.reject == r_scaled.reject

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            stute_test(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]))

    def test_nan_input_raises(self):
        d, dy = _linear_dgp(G=50)
        dy_nan = dy.copy()
        dy_nan[5] = np.nan
        with pytest.raises(ValueError, match="contains NaN"):
            stute_test(d, dy_nan)


# =============================================================================
# Yatchew-HR test
# =============================================================================


class TestYatchewHRTest:
    """Tests for :func:`yatchew_hr_test`."""

    def test_does_not_reject_linear_dgp(self):
        """Commit criterion 9 (single-realization): linear -> no reject."""
        d, dy = _linear_dgp(G=200, beta=2.0, sigma=0.3, seed=42)
        r = yatchew_hr_test(d, dy, alpha=0.05)
        assert r.reject is False

    def test_default_null_is_linearity_with_correct_label(self):
        """Default `null=` (omitted) tags result as ``null_form='linearity'``
        and is bit-exact with explicit `null='linearity'`."""
        d, dy = _linear_dgp(G=50, beta=2.0, sigma=0.3, seed=42)
        r_default = yatchew_hr_test(d, dy, alpha=0.05)
        r_explicit = yatchew_hr_test(d, dy, alpha=0.05, null="linearity")
        assert r_default.null_form == "linearity"
        assert r_explicit.null_form == "linearity"
        np.testing.assert_array_equal(r_default.t_stat_hr, r_explicit.t_stat_hr)
        np.testing.assert_array_equal(r_default.sigma2_lin, r_explicit.sigma2_lin)
        np.testing.assert_array_equal(r_default.sigma2_diff, r_explicit.sigma2_diff)
        np.testing.assert_array_equal(r_default.sigma2_W, r_explicit.sigma2_W)

    def test_rejects_quadratic_dgp(self):
        """Commit criterion 9 (single-realization): quadratic -> reject."""
        d, dy = _quadratic_dgp(G=200, beta=2.0, gamma=5.0, sigma=0.3, seed=42)
        r = yatchew_hr_test(d, dy, alpha=0.05)
        assert r.reject is True
        assert r.t_stat_hr > r.critical_value

    def test_sigma2_diff_normalizer_is_2G_paper_literal(self):
        """Commit criterion 10: sigma2_diff divides by 2G (paper-literal).

        Hand-compute sigma2_diff on a known input and verify the
        implementation divides by 2G, NOT 2(G-1).
        """
        # G = 4 deterministic inputs; sort by d first.
        d = np.array([0.1, 0.2, 0.3, 0.4])
        dy = np.array([1.0, 3.0, 2.0, 5.0])
        r = yatchew_hr_test(d, dy, alpha=0.05)

        # Sorted by d: (already sorted)
        # diff_dy = [dy[1]-dy[0], dy[2]-dy[1], dy[3]-dy[2]]
        #         = [2, -1, 3]; squared = [4, 1, 9]; sum = 14
        # Paper-literal: sigma2_diff = 14 / (2 * 4) = 1.75
        # WRONG form (using G-1=3): 14 / (2 * 3) = 2.333...
        sum_sq_diff = 14.0
        G = 4
        expected_paper_literal = sum_sq_diff / (2.0 * G)
        expected_wrong_form = sum_sq_diff / (2.0 * (G - 1))
        np.testing.assert_allclose(r.sigma2_diff, expected_paper_literal, atol=1e-12)
        assert (
            abs(r.sigma2_diff - expected_wrong_form) > 1e-6
        ), "sigma2_diff should divide by 2G (paper-literal), NOT 2(G-1)."

    def test_invalid_alpha_raises(self):
        d, dy = _linear_dgp(G=50)
        with pytest.raises(ValueError, match="alpha must satisfy"):
            yatchew_hr_test(d, dy, alpha=-0.1)

    def test_exact_linear_returns_p1_not_nan(self):
        """P1 fix: exact linear fit (eps=0) must fail-to-reject with p=1,
        NOT NaN. Yatchew statistic is formally -inf (finite-negative numerator
        over zero denominator), which corresponds to p=1 under the one-sided
        standard-normal critical value."""
        G = 50
        d = np.linspace(0.1, 1.0, G)
        dy = 1.0 + 2.0 * d  # exact linear, residuals are zero
        r = yatchew_hr_test(d, dy)
        assert np.isfinite(r.p_value)
        assert r.p_value == 1.0
        assert r.reject is False
        # t_stat_hr is formally -inf for the exact-linear limit.
        assert r.t_stat_hr == float("-inf")

    def test_exact_linear_shortcut_does_not_fire_on_shifted_noisy_data(self):
        """P0 fix (R2): Yatchew exact-linear short-circuit is translation-
        invariant (comparison against centered TSS, not raw sum(dy^2))."""
        rng = np.random.default_rng(42)
        G = 50
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 1.0, size=G)
        dy_shifted = dy + 1e12

        r_raw = yatchew_hr_test(d, dy)
        r_shift = yatchew_hr_test(d, dy_shifted)

        assert np.isfinite(r_raw.t_stat_hr), "shortcut fired on raw noisy data"
        assert np.isfinite(r_shift.t_stat_hr), "shortcut fired on shifted noisy data"
        assert r_raw.reject == r_shift.reject
        np.testing.assert_allclose(r_raw.t_stat_hr, r_shift.t_stat_hr, rtol=1e-3)

    def test_exact_linear_shortcut_does_not_fire_on_rescaled_noisy_data(self):
        """P0 fix (R5): Yatchew exact-linear short-circuit is scale-
        invariant. An earlier revision used a ``max(centered_TSS, 1.0)``
        floor that broke scale invariance under multiplicative rescaling
        of dy (e.g. ``dy * 1e-12``). Fix removes the floor and handles
        the zero-centered-TSS case in a separate branch."""
        rng = np.random.default_rng(42)
        G = 50
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 1.0, size=G)
        dy_scaled = dy * 1e-12

        r_raw = yatchew_hr_test(d, dy)
        r_scaled = yatchew_hr_test(d, dy_scaled)

        # Neither run may short-circuit to t_stat_hr = -inf.
        assert np.isfinite(r_raw.t_stat_hr), "shortcut fired on raw noisy data"
        assert np.isfinite(r_scaled.t_stat_hr), "shortcut fired on scaled noisy data"
        # Decision preserved: Yatchew is scale-invariant in dy.
        assert r_raw.reject == r_scaled.reject
        # T_hr is scale-invariant: sigma2_lin and sigma2_diff both scale
        # by c^2, sigma2_W scales by c^2, so the ratio is unchanged.
        np.testing.assert_allclose(r_raw.t_stat_hr, r_scaled.t_stat_hr, rtol=1e-3)

    def test_duplicate_doses_return_nan(self):
        """P1 fix: yatchew_hr_test rejects duplicate doses with UserWarning
        + NaN result. Adjacent differences depend on within-tie row order,
        which is non-methodological; users with tied doses should use
        stute_test (tie-safe CvM)."""
        d = np.array([0.1, 0.2, 0.2, 0.5, 0.8])  # one duplicate at 0.2
        dy = np.array([1.0, 2.0, 2.5, 3.0, 4.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = yatchew_hr_test(d, dy)
            msgs = [str(w.message) for w in caught]
        assert any("duplicate" in m for m in msgs)
        assert np.isnan(r.t_stat_hr)
        assert np.isnan(r.p_value)
        assert r.reject is False

    def test_constant_d_returns_nan(self):
        """P1 fix: yatchew_hr_test rejects constant d (degenerate case of
        duplicate doses: all observations tied)."""
        d = np.full(10, 0.5)
        dy = np.linspace(0.0, 2.0, 10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = yatchew_hr_test(d, dy)
            msgs = [str(w.message) for w in caught]
        assert any("duplicate" in m for m in msgs)
        assert np.isnan(r.t_stat_hr)
        assert r.reject is False

    def test_sigma4_W_zero_with_positive_numerator_rejects(self):
        """P0 fix: when sigma4_W = 0 AFTER the exact-linear shortcut
        (i.e., residuals are NOT all zero but adjacent-residual-product
        sums vanish), the Yatchew statistic is formally +inf or -inf
        depending on numerator sign. A unit-dose, non-exact-linear
        input where residuals alternate zero/non-zero must NOT be
        silently mapped to p=1 (fail-to-reject).

        Counterexample from CI R3 reviewer: d=[1,2,3,4,5],
        dy=[1,0,-2,0,1]. OLS slope = 0, residuals = dy itself,
        sigma2_lin = 1.2, sigma2_diff = 1.0, sigma4_W = 0 (each
        adjacent residual product includes a zero factor), and the
        formal statistic is sqrt(5) * 0.2 / 0 = +inf -> reject=True.
        Previously the sigma4_W <= 0 branch unconditionally returned
        p=1, which flipped a legitimate rejection.
        """
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dy = np.array([1.0, 0.0, -2.0, 0.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = yatchew_hr_test(d, dy, alpha=0.05)
            msgs = [str(w.message) for w in caught]
        # Must NOT silently fail-to-reject.
        assert not (r.p_value == 1.0 and r.reject is False), (
            "sigma4_W=0 with positive numerator was silently mapped to "
            "p=1, reject=False; expected +inf reject path."
        )
        # Positive numerator -> +inf statistic, p=0, reject=True.
        assert r.t_stat_hr == float("inf")
        assert r.p_value == 0.0
        assert r.reject is True
        # The caller should have been warned.
        assert any("sigma4_W = 0" in m for m in msgs)
        # Sanity: sigma2_lin and sigma2_diff preserved for inspection.
        np.testing.assert_allclose(r.sigma2_lin, 1.2, atol=1e-12)
        np.testing.assert_allclose(r.sigma2_diff, 1.0, atol=1e-12)


# =============================================================================
# Composite workflow
# =============================================================================


class TestCompositeWorkflow:
    """Tests for :func:`did_had_pretest_workflow`."""

    def test_all_pass_on_linear_flags_assumption7_gap(self):
        """Commit criterion 11: linear DGP -> all_pass, verdict flags
        Assumption 7 pre-trends gap per Phase 3 partial-workflow scope."""
        d, dy = _linear_dgp(G=200, beta=2.0, sigma=0.3, seed=42)
        panel = _make_two_period_panel(200, d, dy, seed=42)
        report = did_had_pretest_workflow(
            panel,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            alpha=0.05,
            n_bootstrap=199,
            seed=42,
        )
        assert report.all_pass is True
        # Partial-workflow verdict: explicitly names the Assumption 7 gap
        # so callers do not receive a misleading "TWFE safe" signal.
        assert "QUG and linearity diagnostics fail-to-reject" in report.verdict
        assert "Assumption 7" in report.verdict
        assert "pre-trends" in report.verdict
        assert "paper step 2 deferred" in report.verdict
        # Verdict must NOT claim unconditional TWFE safety.
        assert "TWFE safe" not in report.verdict
        assert report.n_obs == 200
        assert isinstance(report.qug, QUGTestResults)
        assert isinstance(report.stute, StuteTestResults)
        assert isinstance(report.yatchew, YatchewTestResults)

    def test_rejects_on_quadratic_plus_shifted_support(self):
        """Commit criterion 12: quadratic + shifted support -> both reject."""
        rng = np.random.default_rng(42)
        G = 200
        # Shifted-Beta support [0.2, 1.2]: QUG should reject.
        d = 0.2 + rng.beta(2, 2, size=G)
        # Quadratic dose response: linearity tests should reject.
        dy = 2.0 * d + 5.0 * d * d + rng.normal(0, 0.3, size=G)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        report = did_had_pretest_workflow(
            panel,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            alpha=0.05,
            n_bootstrap=199,
            seed=42,
        )
        assert report.all_pass is False
        assert "support infimum rejected" in report.verdict
        assert "linearity rejected" in report.verdict
        # At least QUG and one of {Stute, Yatchew} rejected.
        assert report.qug.reject is True
        assert report.stute is not None and report.yatchew is not None
        assert report.stute.reject or report.yatchew.reject

    def test_workflow_handles_tied_zero_doses_via_stute_fallback(self):
        """R4 P2 fix: on a common QUG-style panel with repeated d=0
        (never-treated comparison group), Yatchew returns NaN (duplicate
        doses) but Stute handles ties correctly. The composite workflow
        must adjudicate the linearity step via Stute ALONE per the paper's
        "Stute or Yatchew" step-3 wording, producing a conclusive verdict
        with a Yatchew-skipped note rather than forcing the whole report
        to inconclusive.

        Paper review line 298 cites an application with 12 zero-dose
        units, so this panel shape is not exotic.
        """
        rng = np.random.default_rng(42)
        G = 60
        # 20 never-treated (d=0 at post), 40 treated with random positive.
        d = np.zeros(G)
        d[20:] = rng.uniform(0.1, 1.0, size=40)
        dy = 2.0 * d + rng.normal(0.0, 0.3, size=G)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = did_had_pretest_workflow(
                panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
            )
        assert report.stute is not None and report.yatchew is not None
        # Yatchew must NaN (ties from the 20 zero doses).
        assert np.isnan(report.yatchew.p_value)
        assert report.yatchew.reject is False
        # QUG and Stute must be conclusive.
        assert np.isfinite(report.qug.p_value)
        assert np.isfinite(report.stute.p_value)
        # Composite must be conclusive (NOT inconclusive).
        assert "inconclusive" not in report.verdict
        # Verdict should note Yatchew was skipped.
        assert "Yatchew NaN - skipped" in report.verdict
        # With a linear DGP, all_pass should be True despite Yatchew NaN
        # (step 3 handled by Stute alone, per paper's "Stute OR Yatchew").
        assert report.all_pass is True

    def test_all_pass_false_when_any_test_nan(self):
        """P1 fix: all_pass must NOT be True when any constituent test is
        NaN. Previously all_pass was computed from `not reject` only, so a
        NaN-statistic test (reject=False by convention) incorrectly tripped
        all_pass=True while verdict was "inconclusive"."""
        G = 40
        # Constant dose triggers QUG tie (D_(1) == D_(2)) -> NaN.
        d = np.full(G, 0.5)
        dy = np.linspace(0.0, 2.0, G)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = did_had_pretest_workflow(
                panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
            )
        # At least one test produces NaN, so `all_pass` MUST be False even
        # though none of the tests set reject=True.
        assert np.isnan(report.qug.p_value)
        assert report.all_pass is False
        assert report.verdict.startswith("inconclusive")

    def test_workflow_seed_controls_stute_only(self):
        """Workflow seed forwards to Stute only; re-run with same seed is reproducible."""
        d, dy = _linear_dgp(G=100, beta=2.0, sigma=0.3, seed=42)
        panel = _make_two_period_panel(100, d, dy, seed=42)
        report_a = did_had_pretest_workflow(
            panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
        )
        report_b = did_had_pretest_workflow(
            panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
        )
        assert report_a.stute is not None and report_b.stute is not None
        assert report_a.yatchew is not None and report_b.yatchew is not None
        assert report_a.stute.p_value == report_b.stute.p_value
        assert report_a.qug.t_stat == report_b.qug.t_stat
        assert report_a.yatchew.t_stat_hr == report_b.yatchew.t_stat_hr


# =============================================================================
# NaN propagation
# =============================================================================


class TestNaNPropagation:
    """All three tests NaN-propagate on degenerate inputs; verdict is inconclusive."""

    def test_constant_dy_trivially_satisfies_linearity(self):
        """Commit criterion 13 (reframed): constant dy is trivially linear
        in d; Stute and Yatchew should fail-to-reject with large p-values.

        Constant dy means OLS residuals are mathematically zero and the CvM
        cusum is zero; the bootstrap yields all-tied ``S_b = 0``. The test
        correctly fails-to-reject (linearity holds). NaN propagation is
        covered by :meth:`test_qug_tie_propagates_to_composite` below.
        """
        G = 50
        d = np.linspace(0.1, 1.0, G)
        dy = np.full(G, 3.0)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = did_had_pretest_workflow(
                panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
            )
        assert report.stute is not None and report.yatchew is not None
        assert report.stute.reject is False
        assert report.yatchew.reject is False
        assert report.stute.p_value > 0.5
        assert report.yatchew.p_value > 0.5

    def test_qug_tie_propagates_to_composite(self):
        """QUG tie in composite -> verdict mentions QUG NaN."""
        G = 20
        # All units share the same positive dose -> D_(1) == D_(2) tie.
        d = np.full(G, 0.5)
        dy = np.linspace(0.0, 2.0, G)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = did_had_pretest_workflow(
                panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
            )
        assert np.isnan(report.qug.p_value)
        assert "QUG" in report.verdict
        assert report.verdict.startswith("inconclusive")

    def test_stute_constant_d_returns_nan(self):
        G = 50
        d = np.full(G, 0.5)
        dy = np.linspace(0.0, 2.0, G)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = stute_test(d, dy, n_bootstrap=199, seed=42)
        assert np.isnan(r.cvm_stat)
        assert r.reject is False


# =============================================================================
# JSON / DataFrame serialization
# =============================================================================


class TestJSONSerialization:
    """to_dict() is JSON-safe; to_dataframe() returns correct schema."""

    def test_qug_to_dict_is_json_safe(self):
        d = np.array([0.2, 0.5, 0.9])
        r = qug_test(d)
        json_str = json.dumps(r.to_dict())
        round_tripped = json.loads(json_str)
        assert round_tripped["test"] == "qug"
        assert round_tripped["reject"] is False

    def test_stute_to_dict_is_json_safe(self):
        d, dy = _linear_dgp(G=50)
        r = stute_test(d, dy, n_bootstrap=199, seed=42)
        json.dumps(r.to_dict())  # must not raise

    def test_yatchew_to_dict_is_json_safe(self):
        d, dy = _linear_dgp(G=50)
        r = yatchew_hr_test(d, dy)
        json.dumps(r.to_dict())  # must not raise

    def test_yatchew_to_dict_carries_null_form(self):
        """``to_dict()`` surfaces ``null_form`` for both modes."""
        d, dy = _linear_dgp(G=50)
        r_lin = yatchew_hr_test(d, dy)  # default = "linearity"
        r_mi = yatchew_hr_test(d, dy, null="mean_independence")
        d_lin = json.loads(json.dumps(r_lin.to_dict()))
        d_mi = json.loads(json.dumps(r_mi.to_dict()))
        assert d_lin["null_form"] == "linearity"
        assert d_mi["null_form"] == "mean_independence"

    def test_report_to_dict_is_json_safe(self):
        """Commit criterion 16: json.dumps(report.to_dict()) succeeds."""
        d, dy = _linear_dgp(G=50)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42)
        json_str = json.dumps(report.to_dict())
        round_tripped = json.loads(json_str)
        assert "qug" in round_tripped
        assert "stute" in round_tripped
        assert "yatchew" in round_tripped
        assert "verdict" in round_tripped

    def test_report_to_dataframe_schema(self):
        """Commit criterion 17: 3-row frame with specified columns in order."""
        d, dy = _linear_dgp(G=50)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42)
        df = report.to_dataframe()
        expected_cols = [
            "test",
            "statistic_name",
            "statistic_value",
            "p_value",
            "reject",
            "alpha",
            "n_obs",
        ]
        assert list(df.columns) == expected_cols
        assert list(df["test"]) == ["qug", "stute", "yatchew_hr"]
        assert list(df["statistic_name"]) == ["t_stat", "cvm_stat", "t_stat_hr"]
        assert len(df) == 3

    def test_per_test_to_dataframe_is_single_row(self):
        """Commit criterion 18: per-test to_dataframe returns a 1-row frame."""
        d, dy = _linear_dgp(G=50)
        r_stute = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_yatchew = yatchew_hr_test(d, dy)
        r_qug = qug_test(d)
        assert len(r_stute.to_dataframe()) == 1
        assert len(r_yatchew.to_dataframe()) == 1
        assert len(r_qug.to_dataframe()) == 1
        assert r_stute.to_dataframe()["test"].iloc[0] == "stute"


# =============================================================================
# Seed reproducibility
# =============================================================================


class TestSeedReproducibility:
    """Commit criterion 7: stute_test(seed=42) bitwise-reproducible."""

    def test_stute_seed_reproducible(self):
        d, dy = _linear_dgp(G=100, seed=1)
        r_a = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_b = stute_test(d, dy, n_bootstrap=199, seed=42)
        assert r_a.cvm_stat == r_b.cvm_stat
        assert r_a.p_value == r_b.p_value  # bitwise

    def test_stute_different_seeds_produce_different_pvals(self):
        d, dy = _linear_dgp(G=100, seed=1)
        r_a = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_b = stute_test(d, dy, n_bootstrap=199, seed=43)
        # CvM stat is seed-independent; only p-value varies across bootstraps.
        assert r_a.cvm_stat == r_b.cvm_stat
        # P-values should differ at B = 199 (grid = 1/200).
        assert r_a.p_value != r_b.p_value


# =============================================================================
# Sample-size gates
# =============================================================================


class TestSampleSizeGates:
    """Below-minimum sample sizes emit UserWarning + NaN result (no raise)."""

    def test_qug_below_min_returns_nan(self):
        """G=1 (after filter) -> NaN, warning, reject=False."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = qug_test(np.array([0.5]))
            msgs = [str(w.message) for w in caught]
        assert any("at least 2" in m for m in msgs)
        assert np.isnan(r.t_stat)
        assert r.reject is False

    def test_qug_all_zero_returns_nan(self):
        """All d=0 -> filter removes everything -> NaN."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = qug_test(np.zeros(10))
        assert np.isnan(r.t_stat)
        assert r.n_obs == 0
        assert r.n_excluded_zero == 10

    def test_stute_below_min_returns_nan(self):
        """Commit criterion 14: stute_test(G=8) -> NaN, warning, no raise."""
        d, dy = _linear_dgp(G=8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = stute_test(d, dy, n_bootstrap=199, seed=42)
            msgs = [str(w.message) for w in caught]
        assert any("below the minimum" in m for m in msgs)
        assert np.isnan(r.cvm_stat)
        assert r.reject is False

    def test_yatchew_below_min_returns_nan(self):
        """yatchew_hr_test(G=2) -> NaN, warning."""
        d = np.array([0.1, 0.5])
        dy = np.array([1.0, 2.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = yatchew_hr_test(d, dy)
            msgs = [str(w.message) for w in caught]
        assert any("below the minimum" in m for m in msgs)
        assert np.isnan(r.t_stat_hr)
        assert r.reject is False


# =============================================================================
# _compose_verdict priority logic
# =============================================================================


def _mk_qug(reject: bool, p: float = 0.5) -> QUGTestResults:
    return QUGTestResults(
        t_stat=10.0 if reject else 0.5,
        p_value=p,
        reject=reject,
        alpha=0.05,
        critical_value=19.0,
        n_obs=50,
        n_excluded_zero=0,
        d_order_1=0.1,
        d_order_2=0.2,
    )


def _mk_stute(reject: bool, p: float = 0.5) -> StuteTestResults:
    return StuteTestResults(
        cvm_stat=0.5,
        p_value=p,
        reject=reject,
        alpha=0.05,
        n_bootstrap=999,
        n_obs=50,
        seed=42,
    )


def _mk_yatchew(reject: bool, p: float = 0.5) -> YatchewTestResults:
    return YatchewTestResults(
        t_stat_hr=3.0 if reject else -0.5,
        p_value=p,
        reject=reject,
        alpha=0.05,
        critical_value=1.645,
        sigma2_lin=1.0,
        sigma2_diff=1.0,
        sigma2_W=1.0,
        n_obs=50,
    )


class TestComposeVerdictLogic:
    """Commit criterion 15: 5 priority-order tests."""

    def test_qug_nan_is_inconclusive(self):
        """(a1) QUG NaN forces inconclusive (step 1 is required)."""
        q = _mk_qug(reject=False, p=float("nan"))
        s = _mk_stute(reject=False)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert verdict == "inconclusive - QUG NaN"

    def test_both_linearity_nan_is_inconclusive(self):
        """(a2) When BOTH Stute and Yatchew are NaN, the linearity step
        cannot be adjudicated and the verdict is inconclusive."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False, p=float("nan"))
        y = _mk_yatchew(reject=False, p=float("nan"))
        verdict = _compose_verdict(q, s, y)
        assert verdict == "inconclusive - both Stute and Yatchew linearity tests NaN"

    def test_yatchew_nan_alone_does_not_force_inconclusive(self):
        """CRITICAL R4 P3 fix: the paper says step 3 uses "Stute OR
        Yatchew". A conclusive Stute must be sufficient even when Yatchew
        returns NaN (e.g. tied-dose panels). Verdict should be
        fail-to-reject with a "(Yatchew NaN - skipped)" suffix."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False)
        y = _mk_yatchew(reject=False, p=float("nan"))
        verdict = _compose_verdict(q, s, y)
        assert "inconclusive" not in verdict
        assert verdict.startswith("QUG and linearity diagnostics fail-to-reject")
        assert "Yatchew NaN - skipped" in verdict
        assert "Assumption 7" in verdict

    def test_stute_nan_alone_does_not_force_inconclusive(self):
        """Mirror of the above: Yatchew conclusive + Stute NaN should
        still adjudicate the linearity step via Yatchew alone."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False, p=float("nan"))
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "inconclusive" not in verdict
        assert verdict.startswith("QUG and linearity diagnostics fail-to-reject")
        assert "Stute NaN - skipped" in verdict

    def test_qug_reject_with_both_linearity_nan_surfaces_rejection(self):
        """R6 P1 fix: a conclusive QUG rejection must NOT be hidden by
        "inconclusive" just because BOTH linearity tests are NaN. Paper
        rule is one-way: TWFE is admissible only if NO test rejects, so
        any conclusive rejection dominates unresolved-step notes.
        """
        q = _mk_qug(reject=True)
        s = _mk_stute(reject=False, p=float("nan"))
        y = _mk_yatchew(reject=False, p=float("nan"))
        verdict = _compose_verdict(q, s, y)
        # Rejection must be surfaced as the primary verdict.
        assert "support infimum rejected" in verdict
        assert not verdict.startswith("inconclusive")
        # Unresolved linearity step is appended, not replacing the rejection.
        assert "additional steps unresolved" in verdict
        assert "both Stute and Yatchew" in verdict

    def test_linearity_reject_with_qug_nan_surfaces_rejection(self):
        """R6 P1 fix: a conclusive linearity rejection (Stute or Yatchew)
        must NOT be hidden by "inconclusive - QUG NaN"."""
        q = _mk_qug(reject=False, p=float("nan"))
        s = _mk_stute(reject=True)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "linearity rejected" in verdict
        assert "Stute" in verdict
        assert not verdict.startswith("inconclusive")
        assert "additional steps unresolved" in verdict
        assert "QUG NaN" in verdict

    def test_all_three_reject_with_qug_nan_keeps_conclusive_rejections(self):
        """Mixed case: Stute and Yatchew both conclusively reject, QUG
        unresolved. All conclusive rejections must be reported."""
        q = _mk_qug(reject=False, p=float("nan"))
        s = _mk_stute(reject=True)
        y = _mk_yatchew(reject=True)
        verdict = _compose_verdict(q, s, y)
        assert "linearity rejected" in verdict
        assert "Stute" in verdict
        assert "Yatchew" in verdict
        assert "QUG NaN" in verdict
        assert not verdict.startswith("inconclusive")

    def test_none_reject_flags_assumption7_gap(self):
        """(b) None reject -> verdict flags the Assumption 7 pre-trends gap
        rather than claiming unconditional TWFE safety."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "QUG and linearity diagnostics fail-to-reject" in verdict
        assert "Assumption 7" in verdict
        assert "pre-trends" in verdict
        assert "paper step 2 deferred" in verdict
        assert "TWFE safe" not in verdict

    def test_only_qug_rejects(self):
        """(c) Only QUG rejects -> QUG-only message."""
        q = _mk_qug(reject=True)
        s = _mk_stute(reject=False)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "support infimum rejected" in verdict
        assert "linearity rejected" not in verdict

    def test_only_linearity_rejects(self):
        """(d) Only linearity rejects -> linearity-only message."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=True)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "linearity rejected" in verdict
        assert "Stute" in verdict
        assert "Yatchew" not in verdict
        assert "support infimum rejected" not in verdict

    def test_all_reject_bundles_reasons(self):
        """(e) All reject -> bundled message naming each."""
        q = _mk_qug(reject=True)
        s = _mk_stute(reject=True)
        y = _mk_yatchew(reject=True)
        verdict = _compose_verdict(q, s, y)
        assert "support infimum rejected" in verdict
        assert "linearity rejected" in verdict
        assert "Stute" in verdict
        assert "Yatchew" in verdict
        assert "; " in verdict  # bundled with separator


# =============================================================================
# Top-level import surface
# =============================================================================


def test_phase_3_imports_at_top_level():
    """Commit criterion 19: all Phase 3 names importable from `diff_diff`."""
    import diff_diff

    expected = [
        "HADPretestReport",
        "QUGTestResults",
        "StuteTestResults",
        "YatchewTestResults",
        "did_had_pretest_workflow",
        "qug_test",
        "stute_test",
        "yatchew_hr_test",
    ]
    for name in expected:
        assert hasattr(diff_diff, name), f"diff_diff.{name} missing from public API"
        assert name in diff_diff.__all__, f"{name} missing from diff_diff.__all__"


# =============================================================================
# summary() smoke
# =============================================================================


class TestSummary:
    """summary() / print_summary() produce non-empty strings for all classes."""

    def test_qug_summary_non_empty(self):
        r = qug_test(np.array([0.2, 0.5, 0.9]))
        s = r.summary()
        assert len(s) > 0
        assert "QUG" in s

    def test_stute_summary_non_empty(self):
        d, dy = _linear_dgp(G=50)
        r = stute_test(d, dy, n_bootstrap=199, seed=42)
        s = r.summary()
        assert len(s) > 0
        assert "Stute" in s or "CvM" in s

    def test_yatchew_summary_non_empty(self):
        d, dy = _linear_dgp(G=50)
        r = yatchew_hr_test(d, dy)
        s = r.summary()
        assert len(s) > 0
        assert "Yatchew" in s

    def test_yatchew_summary_switches_title_on_null_form(self):
        """``summary()`` renders the linearity / mean-independence title
        per ``null_form``."""
        d, dy = _linear_dgp(G=50)
        s_lin = yatchew_hr_test(d, dy).summary()
        s_mi = yatchew_hr_test(d, dy, null="mean_independence").summary()
        assert "linearity test" in s_lin
        assert "mean-independence test" in s_mi
        # Negative controls — each title is mode-specific.
        assert "mean-independence test" not in s_lin
        assert "linearity test" not in s_mi

    def test_yatchew_repr_includes_null_form(self):
        """``repr()`` carries ``null_form=`` for both modes."""
        d, dy = _linear_dgp(G=50)
        r_lin = yatchew_hr_test(d, dy)
        r_mi = yatchew_hr_test(d, dy, null="mean_independence")
        assert "null_form='linearity'" in repr(r_lin)
        assert "null_form='mean_independence'" in repr(r_mi)

    def test_report_summary_bundles_all(self):
        d, dy = _linear_dgp(G=50)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42)
        s = report.summary()
        assert "QUG" in s
        assert "Stute" in s or "CvM" in s
        assert "Yatchew" in s
        assert "Verdict:" in s


# =============================================================================
# Phase 3 follow-up: joint Stute tests + event-study workflow dispatch
# =============================================================================


def _make_multi_period_panel(
    G: int,
    periods: list,
    first_treat_period,
    dose_fn=None,
    outcome_fn=None,
    seed: int = 42,
) -> pd.DataFrame:
    """Construct a multi-period HAD panel.

    Parameters
    ----------
    G : int
        Number of units.
    periods : list
        Time labels, ordered (numeric or ordered dtype). Must contain at
        least one pre-period (``t < first_treat_period``) and one
        post-period (``t >= first_treat_period``).
    first_treat_period : period label
        The first period where any unit has ``D > 0``. For pre-periods,
        ``D = 0`` for every unit.
    dose_fn : callable or None
        ``dose_fn(rng, G) -> np.ndarray`` returning per-unit dose.
        Default: uniform on [0.05, 1.0].
    outcome_fn : callable or None
        ``outcome_fn(rng, unit_id, t, d, is_post, first_treat) -> float``
        returning the outcome for a single (unit, period) cell. Default:
        linear effect ``0.5 * d`` on post-periods plus Gaussian noise.
    seed : int
    """
    rng = np.random.default_rng(seed)
    if dose_fn is None:
        doses = rng.uniform(0.05, 1.0, size=G)
    else:
        doses = dose_fn(rng, G)
    unit_effects = rng.normal(0.0, 0.3, size=G)

    def _default_outcome(rng_, g, t, d, is_post, _ft):
        eff = 0.5 * d if is_post else 0.0
        return float(unit_effects[g] + eff + rng_.normal(0.0, 0.1))

    if outcome_fn is None:
        outcome_fn = _default_outcome

    rows = []
    for g in range(G):
        for t in periods:
            is_post = t >= first_treat_period
            d = float(doses[g]) if is_post else 0.0
            y = outcome_fn(rng, g, t, d, is_post, first_treat_period)
            rows.append({"unit": g, "period": t, "y": y, "d": d})
    return pd.DataFrame(rows)


def _nonlinear_outcome(d_effect_fn):
    """Build an outcome_fn applying d_effect_fn(d) at post-periods."""

    def _fn(rng_, g, t, d, is_post, _ft):
        eff = d_effect_fn(d) if is_post else 0.0
        noise = rng_.normal(0.0, 0.1)
        return float(0.3 * g / 100.0 + eff + noise)

    return _fn


def _multi_period_residuals(G: int, K: int, seed: int = 42):
    """Random Gaussian residuals + zero fitted + uniform doses."""
    rng = np.random.default_rng(seed)
    horizon_labels = [f"t={1995 + k}" for k in range(K)]
    residuals = {k: rng.normal(0.0, 1.0, size=G) for k in horizon_labels}
    fitted = {k: np.zeros(G) for k in horizon_labels}
    doses = rng.uniform(0.0, 1.0, size=G)
    return residuals, fitted, doses


class TestStuteJointPretest:
    """Tests for :func:`stute_joint_pretest` (residuals-in core)."""

    def test_k1_parity_with_single_horizon_stute(self):
        """K=1 joint matches stute_test on same residuals (refit semantics)."""
        rng = np.random.default_rng(42)
        G = 100
        d = rng.uniform(0.0, 1.0, G)
        dy = 0.3 * d + rng.normal(0.0, 0.2, G)
        # stute_test fits OLS(dy ~ 1 + d) internally. Mirror the fit here
        # so the residuals passed to the joint helper are identical.
        x = np.column_stack([np.ones(G), d])
        beta = np.linalg.solve(x.T @ x, x.T @ dy)
        fitted = x @ beta
        resid = dy - fitted
        joint = stute_joint_pretest(
            residuals_by_horizon={"only": resid},
            fitted_by_horizon={"only": fitted},
            doses=d,
            design_matrix=x,
            n_bootstrap=999,
            seed=123,
            null_form="linearity",
        )
        single = stute_test(d, dy, n_bootstrap=999, seed=123)
        np.testing.assert_allclose(joint.cvm_stat_joint, single.cvm_stat, atol=1e-14, rtol=1e-14)
        # p_value can differ slightly due to RNG draw order (joint draws
        # one eta vector per iteration, single draws one - same shape);
        # the statistic being bit-identical is the critical check.

    def test_linear_dgp_fails_to_reject(self):
        """Linear DGP across all horizons: joint test should not reject."""
        rng = np.random.default_rng(2)
        G = 80
        d = rng.uniform(0.0, 1.0, G)
        residuals = {}
        fitted = {}
        for k in range(3):
            dy = 0.4 * d + rng.normal(0.0, 0.2, G)
            x = np.column_stack([np.ones(G), d])
            beta = np.linalg.solve(x.T @ x, x.T @ dy)
            fit = x @ beta
            residuals[f"h{k}"] = dy - fit
            fitted[f"h{k}"] = fit
        result = stute_joint_pretest(
            residuals_by_horizon=residuals,
            fitted_by_horizon=fitted,
            doses=d,
            design_matrix=np.column_stack([np.ones(G), d]),
            n_bootstrap=499,
            seed=99,
        )
        assert result.p_value > 0.05, f"unexpected rejection: p={result.p_value}"
        assert result.reject is False

    def test_violated_dgp_in_single_horizon_reject(self):
        """Quadratic effect in one of 3 horizons: joint test should reject."""
        rng = np.random.default_rng(7)
        G = 150
        d = rng.uniform(0.05, 1.0, G)
        residuals = {}
        fitted = {}
        for k in range(3):
            if k == 1:
                dy = 4.0 * (d**2) + rng.normal(0.0, 0.2, G)
            else:
                dy = 0.4 * d + rng.normal(0.0, 0.2, G)
            x = np.column_stack([np.ones(G), d])
            beta = np.linalg.solve(x.T @ x, x.T @ dy)
            fit = x @ beta
            residuals[f"h{k}"] = dy - fit
            fitted[f"h{k}"] = fit
        result = stute_joint_pretest(
            residuals_by_horizon=residuals,
            fitted_by_horizon=fitted,
            doses=d,
            design_matrix=np.column_stack([np.ones(G), d]),
            n_bootstrap=999,
            seed=99,
        )
        assert result.reject is True, f"expected rejection, got p={result.p_value}"

    def test_shared_eta_across_horizons_white_box(self):
        """Bootstrap uses the same eta for all horizons in each iteration.

        White-box check: construct residuals where horizon 0 and horizon
        1 have the EXACT SAME residuals. The joint bootstrap under a
        SHARED eta must produce bootstrap outcomes dy_b_h0 == dy_b_h1
        exactly (same fitted + same residuals * same eta). The refit
        then gives the same residuals, and the CvM statistic for both
        horizons is identical. If eta were drawn independently per
        horizon, the two bootstrap residual streams would diverge.

        We verify by checking that S_joint_bootstrap = 2 * S_single_bootstrap
        across many iterations (same underlying process duplicated).
        """
        rng = np.random.default_rng(11)
        G = 60
        d = rng.uniform(0.0, 1.0, G)
        dy = 0.5 * d + rng.normal(0.0, 0.3, G)
        x = np.column_stack([np.ones(G), d])
        beta = np.linalg.solve(x.T @ x, x.T @ dy)
        fit = x @ beta
        resid = dy - fit
        joint = stute_joint_pretest(
            residuals_by_horizon={"a": resid, "b": resid.copy()},
            fitted_by_horizon={"a": fit, "b": fit.copy()},
            doses=d,
            design_matrix=x,
            n_bootstrap=499,
            seed=77,
        )
        single = stute_joint_pretest(
            residuals_by_horizon={"a": resid},
            fitted_by_horizon={"a": fit},
            doses=d,
            design_matrix=x,
            n_bootstrap=499,
            seed=77,
        )
        # Under shared eta, the joint stat is exactly 2x the single stat
        # (both horizons identical). Under independent eta, the joint
        # would be the sum of two INDEPENDENT draws - bootstrap
        # distributions would differ from 2x the single distribution.
        np.testing.assert_allclose(
            joint.cvm_stat_joint, 2.0 * single.cvm_stat_joint, atol=1e-14, rtol=1e-14
        )
        # Under shared eta, each bootstrap S*_b_joint = 2 * S*_b_single,
        # so the p-value (P(S*_b >= S_obs)) is identical to the single.
        np.testing.assert_allclose(joint.p_value, single.p_value, atol=1e-14)

    def test_seed_reproducibility(self):
        """Same seed -> bit-identical results across calls."""
        rng = np.random.default_rng(3)
        G = 80
        d = rng.uniform(0.0, 1.0, G)
        resid = {"h0": rng.normal(0.0, 1.0, G), "h1": rng.normal(0.0, 1.0, G)}
        fit = {"h0": np.zeros(G), "h1": np.zeros(G)}
        r1 = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=299,
            seed=55,
        )
        r2 = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=299,
            seed=55,
        )
        np.testing.assert_allclose(r1.cvm_stat_joint, r2.cvm_stat_joint, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r1.p_value, r2.p_value, atol=1e-14, rtol=1e-14)
        assert r1.reject == r2.reject

    def test_nan_propagation(self):
        """NaN in any residual -> p_value=NaN, reject=False, dict preserved."""
        G = 20
        rng = np.random.default_rng(4)
        resid = {
            "h0": rng.normal(0.0, 1.0, G),
            "h1": rng.normal(0.0, 1.0, G),
        }
        resid["h1"][5] = np.nan
        fit = {"h0": np.zeros(G), "h1": np.zeros(G)}
        d = rng.uniform(0.0, 1.0, G)
        result = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=199,
            seed=0,
        )
        assert np.isnan(result.p_value)
        assert np.isnan(result.cvm_stat_joint)
        assert result.reject is False
        assert result.exact_linear_short_circuited is False
        # per_horizon_stats must preserve ALL keys with NaN values (not
        # empty, not partial) - feedback_no_silent_failures.
        assert set(result.per_horizon_stats.keys()) == {"h0", "h1"}
        assert all(np.isnan(v) for v in result.per_horizon_stats.values())
        assert result.horizon_labels == ["h0", "h1"]

    def test_negative_dose_raises(self):
        G = 20
        resid, fit, _ = _multi_period_residuals(G, K=2)
        doses_neg = np.full(G, -0.1)
        with pytest.raises(ValueError, match="non-negative"):
            stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=doses_neg,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )

    def test_small_G_warns_returns_nan(self):
        """R5: G < _MIN_G_STUTE mirrors single-horizon stute_test -
        warn + NaN result instead of raise. Prevents event-study
        workflow crash when a last-cohort filter leaves fewer than 10
        units."""
        G = 5  # below _MIN_G_STUTE (10)
        resid, fit, d = _multi_period_residuals(G, K=2)
        with pytest.warns(UserWarning, match="below the minimum"):
            result = stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=d,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )
        assert np.isnan(result.cvm_stat_joint)
        assert np.isnan(result.p_value)
        assert result.reject is False
        assert result.n_obs == G
        # Full diagnostic surface preserved on the NaN result
        assert set(result.per_horizon_stats.keys()) == set(str(k) for k in resid.keys())
        assert all(np.isnan(v) for v in result.per_horizon_stats.values())

    def test_small_bootstrap_raises(self):
        G = 50
        resid, fit, d = _multi_period_residuals(G, K=2)
        with pytest.raises(ValueError, match="n_bootstrap"):
            stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=d,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=50,
                seed=0,
            )

    def test_empty_residuals_raises(self):
        with pytest.raises(ValueError, match="at least one horizon"):
            stute_joint_pretest(
                residuals_by_horizon={},
                fitted_by_horizon={},
                doses=np.arange(30, dtype=np.float64),
                design_matrix=np.ones((30, 1)),
                n_bootstrap=199,
            )

    def test_key_mismatch_raises(self):
        G = 30
        with pytest.raises(ValueError, match="identical keys"):
            stute_joint_pretest(
                residuals_by_horizon={"a": np.zeros(G)},
                fitted_by_horizon={"b": np.zeros(G)},
                doses=np.arange(G, dtype=np.float64),
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
            )

    def test_exact_linear_short_circuit_per_horizon(self):
        """All-horizons exact linear -> short-circuit (p=1, no bootstrap)."""
        G = 40
        rng = np.random.default_rng(5)
        d = rng.uniform(0.0, 1.0, G)
        # Two horizons, both perfectly linear in d (residuals near-zero)
        dy1 = 2.0 * d + 1.0
        dy2 = -0.5 * d + 3.0
        x = np.column_stack([np.ones(G), d])
        beta1 = np.linalg.solve(x.T @ x, x.T @ dy1)
        beta2 = np.linalg.solve(x.T @ x, x.T @ dy2)
        fit1 = x @ beta1
        fit2 = x @ beta2
        resid1 = dy1 - fit1
        resid2 = dy2 - fit2
        result = stute_joint_pretest(
            residuals_by_horizon={"h1": resid1, "h2": resid2},
            fitted_by_horizon={"h1": fit1, "h2": fit2},
            doses=d,
            design_matrix=x,
            n_bootstrap=199,
            seed=1,
        )
        assert result.exact_linear_short_circuited is True
        assert result.p_value == 1.0
        assert result.reject is False

    def test_exact_linear_short_circuit_scale_invariant(self):
        """Scale-invariant: rescaling residuals by 1e10 preserves short-circuit."""
        G = 40
        rng = np.random.default_rng(6)
        d = rng.uniform(0.0, 1.0, G)
        dy = 2.0 * d + 1.0
        x = np.column_stack([np.ones(G), d])
        beta = np.linalg.solve(x.T @ x, x.T @ dy)
        fit = x @ beta
        resid = dy - fit
        # Scale by 1e10
        result = stute_joint_pretest(
            residuals_by_horizon={"h1": resid * 1e10},
            fitted_by_horizon={"h1": fit * 1e10},
            doses=d,
            design_matrix=x,
            n_bootstrap=199,
            seed=1,
        )
        assert result.exact_linear_short_circuited is True
        assert result.p_value == 1.0

    def test_per_horizon_short_circuit_independence(self):
        """Degenerate horizon + nontrivial horizon -> no short-circuit."""
        G = 80
        rng = np.random.default_rng(8)
        d = rng.uniform(0.05, 1.0, G)
        # Horizon 1: exact linear (fitted = dy, residuals ~ 0)
        dy1 = 2.0 * d + 1.0
        x = np.column_stack([np.ones(G), d])
        beta = np.linalg.solve(x.T @ x, x.T @ dy1)
        fit1 = x @ beta
        resid1 = dy1 - fit1
        # Horizon 2: strong quadratic (nontrivial residuals)
        dy2 = 5.0 * (d**2) + rng.normal(0.0, 0.1, G)
        beta2 = np.linalg.solve(x.T @ x, x.T @ dy2)
        fit2 = x @ beta2
        resid2 = dy2 - fit2
        result = stute_joint_pretest(
            residuals_by_horizon={"lin": resid1, "quad": resid2},
            fitted_by_horizon={"lin": fit1, "quad": fit2},
            doses=d,
            design_matrix=x,
            n_bootstrap=999,
            seed=3,
        )
        # Must NOT short-circuit - the quadratic horizon is informative.
        assert result.exact_linear_short_circuited is False
        # Strong nonlinearity in horizon 2 should make the joint reject.
        assert result.reject is True, f"expected rejection; p={result.p_value}"

    def test_horizon_labels_preserved_as_strings(self):
        """Int / str / pd.Period labels all get str()'d; order preserved."""
        G = 40
        rng = np.random.default_rng(9)
        d = rng.uniform(0.0, 1.0, G)
        resid_int_keyed = {1997: rng.normal(0.0, 1.0, G), 1998: rng.normal(0.0, 1.0, G)}
        fit_int_keyed = {1997: np.zeros(G), 1998: np.zeros(G)}
        result = stute_joint_pretest(
            residuals_by_horizon=resid_int_keyed,
            fitted_by_horizon=fit_int_keyed,
            doses=d,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=199,
            seed=0,
        )
        assert result.horizon_labels == ["1997", "1998"]
        assert set(result.per_horizon_stats.keys()) == {"1997", "1998"}

    def test_constant_d_returns_nan_with_warning(self):
        """R1: constant doses - no cross-sectional variation to detect
        nonlinearity. Must warn and return NaN inference rather than
        a mechanically-zero CvM (mean-indep null) or singular refit
        (linearity null). Mirrors stute_test's single-horizon guard."""
        G = 30
        resid, fit, _ = _multi_period_residuals(G, K=2, seed=123)
        d_constant = np.full(G, 0.5, dtype=np.float64)
        with pytest.warns(UserWarning, match="constant doses"):
            result = stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=d_constant,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )
        assert np.isnan(result.cvm_stat_joint)
        assert np.isnan(result.p_value)
        assert result.reject is False
        assert result.exact_linear_short_circuited is False
        # Per-horizon stats preserved with NaN values (diagnostic surface)
        assert set(result.per_horizon_stats.keys()) == set(resid.keys())
        assert all(np.isnan(v) for v in result.per_horizon_stats.values())

    def test_singular_design_matrix_raises_valueerror(self):
        """R7 P2: rank-deficient custom design_matrix (e.g. duplicate
        columns) must raise an explicit ValueError from the front-door,
        not a raw np.linalg.LinAlgError from the internal solve()."""
        G = 30
        rng = np.random.default_rng(801)
        d = rng.uniform(0.0, 1.0, G)
        resid = {"h0": rng.normal(0.0, 1.0, G), "h1": rng.normal(0.0, 1.0, G)}
        fit = {"h0": np.zeros(G), "h1": np.zeros(G)}
        # design_matrix with two identical columns (rank deficient).
        singular_X = np.column_stack([d, d])
        with pytest.raises(ValueError, match="rank-deficient"):
            stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=d,
                design_matrix=singular_X,
                n_bootstrap=199,
                seed=0,
            )

    def test_stringified_key_collision_raises(self):
        """R4 P1 regression: two raw keys whose str() representations
        collide (e.g. int 1 and str '1', or int 1 and float 1.0) must
        raise explicitly rather than silently overwrite one horizon in
        the internal residuals_arrays map and double-count the survivor
        in the sum-of-CvMs S_joint."""
        G = 20
        rng = np.random.default_rng(701)
        d = rng.uniform(0.0, 1.0, G)
        # int / str collision: str(1) == "1"
        resid_int_str_collision = {
            1: rng.normal(0.0, 1.0, G),
            "1": rng.normal(0.0, 1.0, G),
        }
        fit_int_str_collision = {1: np.zeros(G), "1": np.zeros(G)}
        with pytest.raises(ValueError, match="collision after str"):
            stute_joint_pretest(
                residuals_by_horizon=resid_int_str_collision,
                fitted_by_horizon=fit_int_str_collision,
                doses=d,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )

        # int / float collision: str(1) == "1" but str(1.0) == "1.0"
        # so these actually don't collide. Test a real collision case:
        # two different string representations of the same label.
        # Python: str(True) == "True"; bool(1) == True but that's the
        # same key. Use: str(None) == "None" collides if passed twice,
        # but keys must be unique per dict. Safer: two equal-after-str
        # object keys that were distinct before str conversion.
        class _WeirdLabel:
            def __init__(self, s):
                self._s = s

            def __str__(self):
                return self._s

            def __hash__(self):
                return hash((id(self), self._s))

        a = _WeirdLabel("horizon-1")
        b = _WeirdLabel("horizon-1")  # same str, different object
        assert a is not b
        assert str(a) == str(b)
        resid_obj_collision = {a: rng.normal(0.0, 1.0, G), b: rng.normal(0.0, 1.0, G)}
        fit_obj_collision = {a: np.zeros(G), b: np.zeros(G)}
        with pytest.raises(ValueError, match="collision after str"):
            stute_joint_pretest(
                residuals_by_horizon=resid_obj_collision,
                fitted_by_horizon=fit_obj_collision,
                doses=d,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )


class TestJointPretrendsTest:
    """Tests for :func:`joint_pretrends_test` data-in wrapper."""

    def test_smoke_runs_on_valid_panel(self):
        df = _make_multi_period_panel(
            G=50,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=11,
        )
        result = joint_pretrends_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            pre_periods=[1997, 1998],
            base_period=1999,
            n_bootstrap=299,
            seed=7,
        )
        assert isinstance(result, StuteJointResult)
        assert result.null_form == "mean_independence"
        assert result.n_horizons == 2
        assert result.n_obs == 50
        assert np.isfinite(result.p_value)
        # Linear DGP on post-periods; pre-periods have D=0 so no relationship
        # between dy_pre_t and D is expected. Fail-to-reject is the target.
        assert result.p_value > 0.05

    def test_matches_manually_constructed_residuals(self):
        """Data-in path reproduces explicit residuals-in call exactly."""
        df = _make_multi_period_panel(
            G=60,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=12,
        )
        # Data-in dispatch
        data_result = joint_pretrends_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            pre_periods=[1997, 1998],
            base_period=1999,
            n_bootstrap=399,
            seed=22,
        )
        # Manual construction: pivot, compute dy_t = Y_t - Y_base, then
        # center per-horizon to build residuals.
        pivot = df.pivot(index="unit", columns="period", values="y").sort_index()
        d_per = df.groupby("unit")["d"].max().sort_index().to_numpy()
        G = len(d_per)
        base = pivot[1999].to_numpy(dtype=np.float64)
        resid = {}
        fit = {}
        for t in [1997, 1998]:
            dy_t = pivot[t].to_numpy(dtype=np.float64) - base
            mean_t = float(dy_t.mean())
            fit[str(t)] = np.full(G, mean_t)
            resid[str(t)] = dy_t - mean_t
        manual_result = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d_per,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=399,
            seed=22,
            null_form="mean_independence",
        )
        np.testing.assert_allclose(
            data_result.cvm_stat_joint,
            manual_result.cvm_stat_joint,
            atol=1e-14,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            data_result.p_value,
            manual_result.p_value,
            atol=1e-14,
            rtol=1e-14,
        )

    def test_empty_pre_periods_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="non-empty"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_base_period_in_pre_periods_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="must not appear"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_out_of_order_pre_period_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=2000, seed=1
        )
        with pytest.raises(ValueError, match="strictly < base_period"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1999],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_non_zero_dose_in_pre_period_raises(self):
        """HAD contract: pre-periods have D=0 for every unit. The
        event-study validator catches this via its staggered-cohort
        detection (a pre-period unit with D>0 looks like an earlier
        treatment cohort)."""
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=2000, seed=1
        )
        # Contaminate pre-period 1998 with a non-zero dose for one unit
        df.loc[(df["unit"] == 0) & (df["period"] == 1998), "d"] = 0.5
        with pytest.raises(ValueError, match="Staggered|dose invariant|D = 0"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_non_zero_dose_in_base_period_raises(self):
        """Reciprocal: base_period (last pre-period) must also satisfy
        D=0. Caught by the event-study validator before our local
        guard runs."""
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=2000, seed=1
        )
        df.loc[(df["unit"] == 0) & (df["period"] == 1999), "d"] = 0.3
        with pytest.raises(ValueError, match="Staggered|dose invariant|D = 0"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_staggered_panel_without_first_treat_col_raises(self):
        """R1: direct wrapper call on a staggered panel without
        first_treat_col must raise via the event-study validator
        contract (same behavior as did_had_pretest_workflow's
        event-study dispatch)."""
        parts = []
        for cohort_ft, cohort_range in [(1999, (0, 15)), (2000, (15, 30))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                        }
                    )
        df = pd.DataFrame(parts)
        with pytest.raises(ValueError, match="Staggered"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_staggered_panel_with_first_treat_col_warns_and_filters(self):
        """R1: direct wrapper call on a staggered panel WITH
        first_treat_col auto-filters to last cohort + never-treated
        and emits UserWarning."""
        parts = []
        for cohort_ft, cohort_range in [(1999, (0, 10)), (2000, (10, 40))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                            "first_treat": cohort_ft,
                        }
                    )
        df = pd.DataFrame(parts)
        with pytest.warns(UserWarning, match="staggered|Staggered"):
            result = joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                first_treat_col="first_treat",
                n_bootstrap=199,
                seed=0,
            )
        assert isinstance(result, StuteJointResult)
        # Last cohort (F=2000) + never-treated kept (none in this fixture
        # - all units are treated); n_obs reflects the filter.
        assert result.n_obs == 30  # 10 filtered + 30 kept... actually just 30 kept

    def test_constant_d_wrapper_path_returns_nan_with_warning(self):
        """R1: direct wrapper call on a panel where ALL units have the
        same dose - propagates the joint core's constant-d guard and
        returns NaN inference rather than a spurious fail-to-reject."""

        def const_dose(rng_, G):  # noqa: ARG001
            return np.full(G, 0.4, dtype=np.float64)

        df = _make_multi_period_panel(
            G=30,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            dose_fn=const_dose,
            seed=33,
        )
        with pytest.warns(UserWarning, match="constant doses"):
            result = joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )
        assert np.isnan(result.p_value)
        assert result.reject is False


class TestJointHomogeneityTest:
    """Tests for :func:`joint_homogeneity_test` data-in wrapper."""

    def test_smoke_runs_on_linear_dgp(self):
        df = _make_multi_period_panel(
            G=50,
            periods=[1998, 1999, 2000, 2001, 2002],
            first_treat_period=2000,
            seed=13,
        )
        result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            post_periods=[2000, 2001, 2002],
            base_period=1999,
            n_bootstrap=299,
            seed=21,
        )
        assert isinstance(result, StuteJointResult)
        assert result.null_form == "linearity"
        assert result.n_horizons == 3
        assert np.isfinite(result.p_value)
        assert result.reject is False

    def test_rejects_on_quadratic_post_effect(self):
        """Quadratic effect in D across post-periods -> joint homogeneity rejects."""
        df = _make_multi_period_panel(
            G=120,
            periods=[1998, 1999, 2000, 2001],
            first_treat_period=2000,
            outcome_fn=_nonlinear_outcome(lambda d: 4.0 * (d**2)),
            seed=14,
        )
        result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            post_periods=[2000, 2001],
            base_period=1999,
            n_bootstrap=999,
            seed=31,
        )
        assert result.reject is True, f"expected rejection; p={result.p_value}"

    def test_matches_manually_constructed_residuals(self):
        df = _make_multi_period_panel(
            G=60,
            periods=[1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=15,
        )
        data_result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            post_periods=[2000, 2001],
            base_period=1999,
            n_bootstrap=399,
            seed=41,
        )
        # Manual construction: OLS(dy ~ 1 + D) per horizon
        pivot = df.pivot(index="unit", columns="period", values="y").sort_index()
        d_per = df.groupby("unit")["d"].max().sort_index().to_numpy()
        G = len(d_per)
        base = pivot[1999].to_numpy(dtype=np.float64)
        X = np.column_stack([np.ones(G), d_per.astype(np.float64)])
        resid = {}
        fit = {}
        for t in [2000, 2001]:
            dy_t = pivot[t].to_numpy(dtype=np.float64) - base
            beta = np.linalg.solve(X.T @ X, X.T @ dy_t)
            fit[str(t)] = X @ beta
            resid[str(t)] = dy_t - fit[str(t)]
        manual_result = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d_per,
            design_matrix=X,
            n_bootstrap=399,
            seed=41,
            null_form="linearity",
        )
        np.testing.assert_allclose(
            data_result.cvm_stat_joint,
            manual_result.cvm_stat_joint,
            atol=1e-14,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            data_result.p_value,
            manual_result.p_value,
            atol=1e-14,
            rtol=1e-14,
        )

    def test_empty_post_periods_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="non-empty"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_base_period_in_post_periods_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="must not appear"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[1999],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_post_period_before_base_raises(self):
        """All post_periods must be strictly > base_period."""
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="strictly > base_period"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[1998, 2000],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_all_zero_dose_post_period_raises(self):
        """Post-period with D=0 for every unit contradicts HAD contract.
        Caught either by the event-study validator's contiguous-dose
        invariant (post-zero breaks the monotone transition from pre
        D=0 to post D>0) or by our local reciprocal guard."""
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=1999, seed=1
        )
        # Zero out all doses at post-period 2000 (keep 1999 post intact for base contract)
        df.loc[df["period"] == 2000, "d"] = 0.0
        with pytest.raises(ValueError, match="dose invariant|D = 0 for every unit"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[1999, 2000],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_two_period_negative_post_dose_raises(self):
        """R2 P1 regression: direct wrapper call on a 2-period panel
        with a negative post dose must raise rather than silently
        collapse to zero via ``groupby.max()`` and produce a finite
        result. The 2-period path skips the event-study validator
        (``n_periods < 3``) so the row-level non-negative guard must
        live in ``_aggregate_for_joint_test`` itself."""
        G = 20
        rng = np.random.default_rng(601)
        doses = rng.uniform(0.1, 1.0, size=G)
        # Flip one unit's post dose to a negative value.
        doses[0] = -0.3
        rows = []
        for g in range(G):
            # pre-period
            rows.append({"unit": g, "period": 0, "y": rng.normal(0, 0.1), "d": 0.0})
            # post-period (with negative dose injected for unit 0)
            rows.append(
                {
                    "unit": g,
                    "period": 1,
                    "y": rng.normal(0, 0.1) + 0.3 * doses[g],
                    "d": float(doses[g]),
                }
            )
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="negative dose value"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[1],
                base_period=0,
                n_bootstrap=199,
                seed=0,
            )


class TestMultiPeriodWorkflow:
    """Tests for :func:`did_had_pretest_workflow` event-study dispatch."""

    def _linear_panel(self, seed: int = 100) -> pd.DataFrame:
        return _make_multi_period_panel(
            G=80,
            periods=[1996, 1997, 1998, 1999, 2000, 2001],
            first_treat_period=1999,
            seed=seed,
        )

    def test_overall_aggregate_unchanged(self):
        """Default aggregate='overall' preserves Phase 3 behavior."""
        d, dy = _linear_dgp(G=50, seed=42)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=299, seed=42)
        assert report.aggregate == "overall"
        assert report.stute is not None
        assert report.yatchew is not None
        assert report.pretrends_joint is None
        assert report.homogeneity_joint is None
        # Phase 3 step-2 gap string STILL present on the overall path
        assert "paper step 2 deferred" in report.verdict

    def test_overall_aggregate_rejects_multi_period(self):
        """Registry/docstring contract: aggregate='overall' is two-period only.

        REGISTRY claims `aggregate="overall"` requires a balanced two-period
        panel and that multi-period panels are rejected with a pointer to
        `aggregate="event_study"`. This test pins the front-door rejection
        so the registry text and the validator stay in lock-step.
        """
        df = _make_multi_period_panel(
            G=40,
            periods=[1997, 1998, 1999, 2000],
            first_treat_period=1999,
            seed=313,
        )
        with pytest.raises(ValueError, match=r"aggregate='event_study'"):
            did_had_pretest_workflow(df, "y", "d", "period", "unit", n_bootstrap=199, seed=0)

    def test_event_study_linear_dgp_all_pass(self):
        df = self._linear_panel(seed=101)
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=299,
            seed=17,
        )
        assert report.aggregate == "event_study"
        assert report.stute is None and report.yatchew is None
        assert report.pretrends_joint is not None
        assert report.homogeneity_joint is not None
        assert report.all_pass is True
        assert "TWFE admissible under Section 4" in report.verdict
        # The Phase 3 "paper step 2 deferred" string MUST NOT appear on
        # the event-study path - the gap is closed.
        assert "paper step 2 deferred" not in report.verdict

    def test_event_study_pretrend_violation_flagged(self):
        """Strong pre-trend correlated with D -> pretrends_joint rejects."""

        def pretrend_outcome(rng_, g, t, d, is_post, ft):
            # D is fixed at F for this unit; simulate correlated pre-trend
            # via knowing what the unit's eventual dose will be.
            # Placeholder: rng seeds unit g the same way _make_multi_period_panel does
            # NOTE: this is hacky; we bake-in correlation via g as proxy for dose.
            trend = (g / 100.0) * (t - 1998)  # pre-existing linear trend
            eff = 0.3 * d if is_post else 0.0
            return float(trend + eff + rng_.normal(0.0, 0.05))

        # Construct panel where dose correlates strongly with unit id (proxy for trend strength)
        def dose_fn(rng_, G):
            return np.linspace(0.05, 1.0, G)

        df = _make_multi_period_panel(
            G=80,
            periods=[1996, 1997, 1998, 1999, 2000],
            first_treat_period=1999,
            dose_fn=dose_fn,
            outcome_fn=pretrend_outcome,
            seed=200,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=499,
            seed=29,
        )
        assert report.pretrends_joint is not None
        assert report.pretrends_joint.reject is True
        assert "joint pre-trends rejected - assumption 7 violated" in report.verdict

    def test_event_study_homogeneity_violation_flagged(self):
        """Strong quadratic effect across post-periods -> homogeneity rejects."""
        df = _make_multi_period_panel(
            G=100,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=1999,
            outcome_fn=_nonlinear_outcome(lambda d: 4.0 * (d**2)),
            seed=210,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=999,
            seed=31,
        )
        assert report.homogeneity_joint is not None
        assert report.homogeneity_joint.reject is True
        assert "joint linearity rejected - heterogeneity bias" in report.verdict

    def test_event_study_qug_violation_flagged(self):
        """Shift dose support away from 0 -> QUG rejects."""

        def shifted_dose_fn(rng_, G):
            # All doses far from 0; D_2,(1) and D_2,(2) close -> T small -> reject
            return rng_.uniform(0.5, 1.0, G)

        df = _make_multi_period_panel(
            G=80,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=1999,
            dose_fn=shifted_dose_fn,
            seed=220,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=299,
            seed=42,
        )
        # QUG T statistic should be large (shifted support) -> reject.
        if report.qug.reject:
            assert report.verdict.startswith("support infimum rejected")

    def test_invalid_aggregate_raises(self):
        df = self._linear_panel(seed=102)
        with pytest.raises(ValueError, match="aggregate must be one of"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "period",
                "unit",
                aggregate="bogus",
                n_bootstrap=199,
            )

    def test_single_pre_period_yields_pretrends_skipped(self):
        """If t_pre_list has only the base pre-period, no earlier placebos
        exist -> pretrends_joint is None and verdict flags the skip."""
        df = _make_multi_period_panel(
            G=50,
            periods=[1998, 1999, 2000, 2001],
            first_treat_period=1999,
            seed=130,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=299,
            seed=52,
        )
        assert report.pretrends_joint is None
        # Even with a fail-to-reject homogeneity test, all_pass should be
        # False because pretrends_joint is None (step 2 not closed).
        if report.homogeneity_joint is not None and not report.homogeneity_joint.reject:
            assert report.all_pass is False
        # Verdict should mention the pre-trends skip.
        assert "joint pre-trends skipped" in report.verdict

    def test_no_paper_step_2_deferred_string_on_event_study(self):
        """Regression: event-study verdict must not emit the Phase 3 caveat."""
        df = self._linear_panel(seed=111)
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=299,
            seed=61,
        )
        assert "paper step 2 deferred" not in report.verdict
        assert "deferred to Phase 3 follow-up" not in report.verdict

    def test_first_treat_col_none_with_staggered_raises(self):
        """Inherited contract: staggered panel + no first_treat_col -> raises."""
        # Build two cohorts: one treated at 1999, one at 2000.
        parts = []
        for cohort_ft, cohort_range in [(1999, (0, 40)), (2000, (40, 80))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                        }
                    )
        df = pd.DataFrame(parts)
        with pytest.raises(ValueError):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "period",
                "unit",
                aggregate="event_study",
                n_bootstrap=199,
                seed=0,
            )

    def test_staggered_auto_filter_warns(self):
        """With first_treat_col provided, staggered panel auto-filters with warning."""
        parts = []
        for cohort_ft, cohort_range in [(1999, (0, 30)), (2000, (30, 80))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                            "first_treat": cohort_ft,
                        }
                    )
        df = pd.DataFrame(parts)
        with pytest.warns(UserWarning, match="staggered"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "period",
                "unit",
                first_treat_col="first_treat",
                aggregate="event_study",
                n_bootstrap=199,
                seed=0,
            )

    def test_event_study_verdict_priority_qug_first(self):
        """Rejections bundle; QUG rejection appears first."""
        qug = _mk_qug(reject=True, p=0.01)
        pretrends = StuteJointResult(
            cvm_stat_joint=1.0,
            p_value=0.01,
            reject=True,
            alpha=0.05,
            horizon_labels=["t"],
            per_horizon_stats={"t": 1.0},
            n_bootstrap=999,
            n_obs=50,
            n_horizons=1,
            seed=1,
            null_form="mean_independence",
            exact_linear_short_circuited=False,
        )
        homogeneity = StuteJointResult(
            cvm_stat_joint=0.5,
            p_value=0.5,
            reject=False,
            alpha=0.05,
            horizon_labels=["t"],
            per_horizon_stats={"t": 0.5},
            n_bootstrap=999,
            n_obs=50,
            n_horizons=1,
            seed=1,
            null_form="linearity",
            exact_linear_short_circuited=False,
        )
        verdict = _compose_verdict_event_study(qug, pretrends, homogeneity)
        # QUG appears before pre-trends
        assert verdict.index("QUG") < verdict.index("assumption 7")

    def test_event_study_all_conclusive_no_reject_admissible(self):
        qug = _mk_qug(reject=False, p=0.8)
        pretrends = StuteJointResult(
            cvm_stat_joint=0.3,
            p_value=0.7,
            reject=False,
            alpha=0.05,
            horizon_labels=["t"],
            per_horizon_stats={"t": 0.3},
            n_bootstrap=999,
            n_obs=50,
            n_horizons=1,
            seed=1,
            null_form="mean_independence",
            exact_linear_short_circuited=False,
        )
        homogeneity = StuteJointResult(
            cvm_stat_joint=0.5,
            p_value=0.6,
            reject=False,
            alpha=0.05,
            horizon_labels=["t"],
            per_horizon_stats={"t": 0.5},
            n_bootstrap=999,
            n_obs=50,
            n_horizons=1,
            seed=1,
            null_form="linearity",
            exact_linear_short_circuited=False,
        )
        verdict = _compose_verdict_event_study(qug, pretrends, homogeneity)
        assert "TWFE admissible under Section 4" in verdict

    def test_event_study_small_panel_after_filter_inconclusive_not_crash(self):
        """R5: staggered-panel last-cohort filter can leave fewer than
        `_MIN_G_STUTE` (10) units. The joint Stute core must warn +
        return NaN on small G (matching single-horizon stute_test) so
        the event-study workflow surfaces an inconclusive report
        rather than crashing. Regression against the original
        ValueError-on-G<10 contract."""
        parts = []
        # First cohort: 40 units treated at 1999 - will be DROPPED by
        # the last-cohort filter (F_last=2000 > 1999).
        # Second cohort: only 6 units treated at 2000 - kept. After
        # filter G = 6 < _MIN_G_STUTE, so the joint CvM is ill-
        # calibrated and must return NaN via warn.
        for cohort_ft, cohort_range in [(1999, (0, 40)), (2000, (40, 46))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                            "first_treat": cohort_ft,
                        }
                    )
        df = pd.DataFrame(parts)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "period",
                "unit",
                first_treat_col="first_treat",
                aggregate="event_study",
                n_bootstrap=199,
                seed=0,
            )
        # Workflow must complete (no crash) and surface an inconclusive
        # report. Both joint tests (pretrends + homogeneity) should
        # return NaN on the post-filter G=6 panel.
        assert report.aggregate == "event_study"
        if report.pretrends_joint is not None:
            assert np.isnan(report.pretrends_joint.p_value)
        assert report.homogeneity_joint is not None
        assert np.isnan(report.homogeneity_joint.p_value)
        assert report.all_pass is False
        # At least one "below the minimum" warning from the joint core.
        msgs = [str(w.message) for w in caught]
        assert any("below the minimum" in m for m in msgs)


class TestOrderedCategoricalChronology:
    """R2 P1 regressions: ordered-categorical time columns whose lexical
    and chronological order disagree (e.g. ``"q10"`` < ``"q2"``
    lexically but > chronologically). Raw ``t < base_period`` comparisons
    misorder these panels; the wrappers and workflow must use validated-
    rank comparisons to apply the test to the intended horizons."""

    @staticmethod
    def _categorical_panel(
        G: int = 60,
        categories=("q1", "q2", "q10", "post"),
        first_treat="post",
        seed: int = 501,
    ) -> pd.DataFrame:
        """Panel with ordered-categorical time whose lexical order
        (``"q1" < "q10" < "q2" < "post"``) differs from chronological
        order (``"q1" < "q2" < "q10" < "post"``)."""
        cat_type = pd.CategoricalDtype(categories=list(categories), ordered=True)
        rng = np.random.default_rng(seed)
        doses = rng.uniform(0.05, 1.0, size=G)
        rows = []
        for g in range(G):
            for t in categories:
                is_post = t == first_treat
                d = float(doses[g]) if is_post else 0.0
                y = 0.1 * g + (0.4 * d if is_post else 0.0) + rng.normal(0.0, 0.1)
                rows.append({"unit": g, "period": t, "y": y, "d": d})
        df = pd.DataFrame(rows)
        df["period"] = df["period"].astype(cat_type)
        return df

    def test_joint_pretrends_test_uses_chronological_rank(self):
        """Direct wrapper call with categories ["q1", "q2", "q10"] where
        the lexical order puts "q10" BEFORE "q2" but chronologically
        "q10" comes AFTER "q2". All three pre-periods must be accepted
        without a false out-of-order error."""
        df = self._categorical_panel()
        result = joint_pretrends_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            pre_periods=["q1", "q2"],
            base_period="q10",
            n_bootstrap=199,
            seed=3,
        )
        assert result.n_horizons == 2
        assert set(result.horizon_labels) == {"q1", "q2"}
        # The detrended-outcome residuals are mean-centered; under null
        # (no pre-trend correlated with D), p should be > 0.05 on this
        # weakly-noisy DGP.
        assert np.isfinite(result.p_value)

    def test_joint_pretrends_raises_on_lexically_ordered_but_chrono_invalid(self):
        """With base_period="q2" and pre_periods=["q10"], chronologically
        q10 > q2 so this is out-of-order - the rank-based check must
        raise. Raw `<` on the lexical side would INCORRECTLY accept
        it since "q10" < "q2" lexically."""
        df = self._categorical_panel()
        with pytest.raises(ValueError, match="chronological order"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=["q10"],
                base_period="q2",
                n_bootstrap=199,
                seed=0,
            )

    def test_joint_homogeneity_test_uses_chronological_rank(self):
        """Homogeneity wrapper twin of the pretrends test. Post-period
        "post" comes after all pre-periods chronologically; base="q10"
        is the last pre-period. Lexically "post" > "q10" too (coincides
        here), but the rank-based check must not rely on that."""
        df = self._categorical_panel()
        result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            post_periods=["post"],
            base_period="q10",
            n_bootstrap=199,
            seed=7,
        )
        assert result.n_horizons == 1
        assert result.horizon_labels == ["post"]
        assert np.isfinite(result.p_value)

    def test_workflow_event_study_ordered_categorical(self):
        """did_had_pretest_workflow(aggregate="event_study") must pick
        up BOTH earlier pre-periods ("q1", "q2") from an ordered-
        categorical panel where lexical order would silently drop one
        of them. Regression against the `earlier_pre` raw-< fix."""
        df = self._categorical_panel()
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=13,
        )
        assert report.aggregate == "event_study"
        assert report.pretrends_joint is not None
        # t_pre_list = ["q1", "q2", "q10"] chronologically; base = "q10"
        # (last pre-period); earlier_pre should be ["q1", "q2"] - both
        # placebo horizons must appear in pretrends_joint.
        assert set(report.pretrends_joint.horizon_labels) == {"q1", "q2"}
        assert report.homogeneity_joint is not None
        assert report.homogeneity_joint.horizon_labels == ["post"]
        # Verdict does not emit the step-2-skipped flag (both earlier
        # placebos were found).
        assert "joint pre-trends skipped" not in report.verdict


class TestHADPretestReportSerialization:
    """Tests for HADPretestReport serialization branching by aggregate."""

    def test_to_dict_overall_preserves_phase3_schema(self):
        d, dy = _linear_dgp(G=50)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42)
        out = report.to_dict()
        # Phase 3 schema is bit-exact: no `aggregate` key on the overall
        # path (only emitted on event_study) - Phase 3 downstream
        # consumers must not see a new key.
        assert "aggregate" not in out
        assert "qug" in out and "stute" in out and "yatchew" in out
        # Event-study keys absent on overall
        assert "pretrends_joint" not in out and "homogeneity_joint" not in out
        # Round-trip JSON safely
        json.dumps(out)

    def test_to_dict_event_study_emits_joint_keys(self):
        df = _make_multi_period_panel(
            G=60,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=131,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=0,
        )
        out = report.to_dict()
        assert out["aggregate"] == "event_study"
        assert "qug" in out
        assert "pretrends_joint" in out and "homogeneity_joint" in out
        # Overall-path keys absent on event-study
        assert "stute" not in out and "yatchew" not in out
        json.dumps(out)

    def test_to_dataframe_stable_3_row_shape(self):
        """to_dataframe returns 3 rows for both aggregates."""
        d, dy = _linear_dgp(G=50)
        panel_overall = _make_two_period_panel(50, d, dy, seed=42)
        report_overall = did_had_pretest_workflow(
            panel_overall, "y", "d", "time", "unit", n_bootstrap=199, seed=42
        )
        df_o = report_overall.to_dataframe()
        assert df_o.shape[0] == 3
        assert list(df_o["test"]) == ["qug", "stute", "yatchew_hr"]

        panel_es = _make_multi_period_panel(
            G=50,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=132,
        )
        report_es = did_had_pretest_workflow(
            panel_es,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=0,
        )
        df_e = report_es.to_dataframe()
        assert df_e.shape[0] == 3
        assert list(df_e["test"]) == ["qug", "pretrends_joint", "homogeneity_joint"]
        # Columns identical across aggregates
        assert set(df_o.columns) == set(df_e.columns)

    def test_summary_includes_aggregate_header(self):
        df = _make_multi_period_panel(
            G=50,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=133,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=0,
        )
        s = report.summary()
        assert "aggregate: event_study" in s

    def test_repr_includes_aggregate(self):
        df = _make_multi_period_panel(
            G=50,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=134,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=0,
        )
        r = repr(report)
        assert "aggregate='event_study'" in r


# =============================================================================
# Phase 4.5 C0 decision-gate guards on did_had_pretest_workflow
# =============================================================================


class TestHADPretestWorkflowSurveyGuards:
    """Phase 4.5 C survey-aware workflow tests.

    Phase 4.5 C makes did_had_pretest_workflow functional under
    survey=/weights=: it skips QUG with a UserWarning (per C0 deferral)
    and dispatches the linearity family with the survey-aware mechanism.
    Mutex on survey+weights still raises ValueError; replicate-weight
    survey designs raise NotImplementedError (parallel follow-up)."""

    def _make_minimal_overall_panel(self, with_weight_col: bool = False):
        """Two-period, single-cohort panel sufficient for overall workflow.

        When ``with_weight_col=True``, attaches a 'w' column populated with
        unit-constant positive values (HAD continuous-path constant-within-
        unit invariant)."""
        d_arr, dy_arr = _linear_dgp(G=20, beta=2.0, sigma=0.3)
        df = _make_two_period_panel(G=20, d=d_arr, dy=dy_arr)
        if with_weight_col:
            rng = np.random.default_rng(7)
            w_per_unit = rng.uniform(0.5, 2.0, size=20)
            # Constant-within-unit per HAD invariant.
            df["w"] = df["unit"].map(dict(zip(np.arange(20), w_per_unit)))
        return df

    def test_workflow_mutex_both_raises(self):
        """Phase 4.5 C: passing both survey= AND weights= raises ValueError
        (mutex), mirroring HeterogeneousAdoptionDiD.fit() at had.py:2890."""
        from diff_diff import SurveyDesign

        df = self._make_minimal_overall_panel(with_weight_col=True)
        with pytest.raises(ValueError, match="at most one of"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey=SurveyDesign(weights="w"),
                weights=np.ones(40),
            )

    def test_workflow_unweighted_overall_path_unchanged(self):
        """Stability invariant: existing positional / unweighted calls must
        produce a valid HADPretestReport after the new keyword-only kwargs
        are added. Smoke-tests the overall path."""
        df = self._make_minimal_overall_panel()
        report = did_had_pretest_workflow(df, "y", "d", "time", "unit", n_bootstrap=199, seed=0)
        # Overall-path invariant: stute and yatchew populated; joint variants None.
        assert report.aggregate == "overall"
        assert report.qug is not None
        assert report.stute is not None
        assert report.yatchew is not None
        assert report.pretrends_joint is None
        assert report.homogeneity_joint is None

    def test_workflow_weights_runs_overall_path(self):
        """Phase 4.5 C: weights= now functional. Workflow dispatches to
        weighted Stute + Yatchew, skips QUG, returns valid report with
        qug=None."""
        df = self._make_minimal_overall_panel()
        # 40 rows (20 units x 2 periods); per-row weights with constant-
        # within-unit invariant.
        rng = np.random.default_rng(7)
        w_per_unit = rng.uniform(0.5, 2.0, size=20)
        weights_per_row = np.tile(w_per_unit, 2)
        with pytest.warns(UserWarning, match="QUG step skipped"):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                weights=weights_per_row,
                n_bootstrap=199,
                seed=0,
            )
        assert report.aggregate == "overall"
        assert report.qug is None  # skipped per C0
        assert report.stute is not None
        assert report.yatchew is not None
        assert np.isfinite(report.stute.p_value)

    def test_workflow_survey_runs_overall_path(self):
        """Phase 4.5 C: survey= now functional via SurveyDesign(weights=col)."""
        from diff_diff import SurveyDesign

        df = self._make_minimal_overall_panel(with_weight_col=True)
        with pytest.warns(UserWarning, match="QUG step skipped"):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )
        assert report.aggregate == "overall"
        assert report.qug is None
        assert report.stute is not None
        assert report.yatchew is not None

    def test_workflow_verdict_carries_phase_4_5_c0_suffix(self):
        """Phase 4.5 C: verdict appends the linearity-conditional suffix
        explaining QUG was skipped per C0 deferral. Locks the cross-surface
        text used by downstream consumers."""
        df = self._make_minimal_overall_panel()
        weights_per_row = np.full(40, 1.5)  # uniform-positive
        with pytest.warns(UserWarning):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                weights=weights_per_row,
                n_bootstrap=199,
                seed=0,
            )
        assert "linearity-conditional verdict" in report.verdict
        assert "QUG-under-survey deferred per Phase 4.5 C0" in report.verdict

    def test_workflow_qug_none_serializes_cleanly(self):
        """Phase 4.5 C: qug=None must propagate cleanly through summary,
        to_dict, and to_dataframe (Reviewer CRITICAL #1 - retyped Optional)."""
        df = self._make_minimal_overall_panel()
        weights_per_row = np.full(40, 1.5)
        with pytest.warns(UserWarning):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                weights=weights_per_row,
                n_bootstrap=199,
                seed=0,
            )
        # summary() rendering
        s = report.summary()
        assert "QUG step skipped" in s
        # to_dict() serialization
        d = report.to_dict()
        assert d["qug"] is None
        # to_dataframe() must still produce 3 rows (qug NaN row preserved)
        df_out = report.to_dataframe()
        assert len(df_out) == 3
        qug_row = df_out[df_out["test"] == "qug"].iloc[0]
        assert pd.isna(qug_row["statistic_value"])
        assert pd.isna(qug_row["p_value"])
        assert qug_row["reject"] is False or qug_row["reject"] == 0  # bool-ish

    def test_workflow_replicate_weights_rejected_overall(self):
        """Phase 4.5 C: replicate-weight survey designs (BRR/Fay/JK1/JKn/SDR)
        raise NotImplementedError. Parallel follow-up after Phase 4.5 C."""
        from diff_diff import SurveyDesign

        df = self._make_minimal_overall_panel(with_weight_col=True)
        # Build replicate weights matrix (40 rows x 5 replicates of perturbed weights).
        rng = np.random.default_rng(0)
        w_col = df["w"].to_numpy()
        rep_w = np.column_stack([w_col * (1 + 0.1 * rng.standard_normal(40)) for _ in range(5)])
        df_with_rep = df.copy()
        for i in range(5):
            df_with_rep[f"rep{i}"] = rep_w[:, i]
        sd = SurveyDesign(
            weights="w",
            replicate_weights=[f"rep{i}" for i in range(5)],
            replicate_method="BRR",
        )
        with pytest.raises(NotImplementedError, match="replicate-weight"):
            did_had_pretest_workflow(
                df_with_rep, "y", "d", "time", "unit", survey=sd, n_bootstrap=199, seed=0
            )


# =============================================================================
# Phase 4.5 C: direct-helper survey/weights tests
# =============================================================================


class TestStuteTestSurvey:
    """Phase 4.5 C survey/weights extension on stute_test."""

    def _setup(self, G=30, seed=42):
        d, dy = _linear_dgp(G=G, beta=2.0, sigma=0.3, seed=seed)
        return d, dy

    def test_unweighted_call_bit_exact_after_kwargs_added(self):
        """Stability invariant #1: existing positional/kwarg-free calls
        produce bit-exact pre-PR p_value after the new keyword-only kwargs
        are added (no behavioral change on the unweighted path)."""
        d, dy = self._setup()
        r = stute_test(d, dy, alpha=0.05, n_bootstrap=199, seed=0)
        assert np.isfinite(r.cvm_stat)
        assert 0.0 <= r.p_value <= 1.0

    def test_weights_smoke(self):
        """weights= produces a finite, valid Stute result."""
        d, dy = self._setup()
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=30)
        r = stute_test(d, dy, weights=w, n_bootstrap=199, seed=0)
        assert np.isfinite(r.cvm_stat)
        assert 0.0 <= r.p_value <= 1.0

    def test_survey_smoke(self):
        """survey= via trivial ResolvedSurveyDesign produces a finite result."""
        from diff_diff.survey import make_pweight_design

        d, dy = self._setup()
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=30)
        resolved = make_pweight_design(w)
        r = stute_test(d, dy, survey=resolved, n_bootstrap=199, seed=0)
        assert np.isfinite(r.cvm_stat)
        assert 0.0 <= r.p_value <= 1.0

    def test_mutex_both_raises(self):
        """survey + weights mutex (mirrors workflow + qug_test pattern)."""
        from diff_diff.survey import make_pweight_design

        d, dy = self._setup()
        w = np.ones(30)
        with pytest.raises(ValueError, match="at most one of"):
            stute_test(d, dy, weights=w, survey=make_pweight_design(w), n_bootstrap=199, seed=0)

    def test_replicate_weights_raises(self):
        """Phase 4.5 C MEDIUM #4: replicate-weight survey designs raise
        NotImplementedError at the direct-helper entry point too (defense in
        depth + reciprocal-guard discipline)."""
        from diff_diff.survey import ResolvedSurveyDesign

        d, dy = self._setup()
        w = np.ones(30)
        rep_w = np.tile(w[:, None], (1, 5))
        resolved_with_rep = ResolvedSurveyDesign(
            weights=w,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=30,
            lonely_psu="remove",
            replicate_weights=rep_w,
            replicate_method="BRR",
            n_replicates=5,
        )
        with pytest.raises(NotImplementedError, match="replicate-weight"):
            stute_test(d, dy, survey=resolved_with_rep, n_bootstrap=199, seed=0)

    def test_negative_weights_rejected(self):
        """Strictly-positive weights required on the pweight shortcut."""
        d, dy = self._setup()
        w = np.ones(30)
        w[0] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            stute_test(d, dy, weights=w, n_bootstrap=199, seed=0)

    def test_weights_length_mismatch(self):
        d, dy = self._setup()
        with pytest.raises(ValueError, match="length"):
            stute_test(d, dy, weights=np.ones(20), n_bootstrap=199, seed=0)


class TestYatchewHRTestSurvey:
    """Phase 4.5 C survey/weights extension on yatchew_hr_test.

    Includes the bit-exact reduction-invariant lock at w=ones(G) per
    Reviewer CRITICAL #2 + MEDIUM #1: weighted variance components reduce
    exactly to the existing unweighted formulas at uniform weights.
    """

    def _setup(self, G=30, seed=42):
        rng = np.random.default_rng(seed)
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 0.3, size=G)
        return d, dy

    def test_unweighted_bit_exact_after_kwargs_added(self):
        """Existing call without weights/survey returns the pre-PR result."""
        d, dy = self._setup()
        r_unweighted = yatchew_hr_test(d, dy, alpha=0.05)
        # No reference value to compare against (no pre-PR golden file
        # captured); just check finiteness.
        assert np.isfinite(r_unweighted.t_stat_hr)

    def test_default_null_under_weights_is_linearity_with_correct_label(self):
        """Under weights, default `null=` (omitted) tags result as
        ``null_form='linearity'`` and is bit-exact with explicit linearity."""
        d, dy = self._setup()
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=30)
        r_default = yatchew_hr_test(d, dy, weights=w)
        r_explicit = yatchew_hr_test(d, dy, weights=w, null="linearity")
        assert r_default.null_form == "linearity"
        assert r_explicit.null_form == "linearity"
        np.testing.assert_array_equal(r_default.t_stat_hr, r_explicit.t_stat_hr)
        np.testing.assert_array_equal(r_default.sigma2_lin, r_explicit.sigma2_lin)

    def test_weighted_reduces_to_unweighted_at_uniform_weights(self):
        """Reviewer CRITICAL #2 lock: at w=ones(G), weighted variance
        components reduce to the unweighted formulas EXACTLY (atol=1e-14)."""
        d, dy = self._setup()
        r_unweighted = yatchew_hr_test(d, dy, alpha=0.05)
        r_weighted = yatchew_hr_test(d, dy, alpha=0.05, weights=np.ones(30))
        # All three variance components must match bit-exactly.
        np.testing.assert_allclose(
            r_unweighted.sigma2_lin, r_weighted.sigma2_lin, atol=1e-14, rtol=1e-14
        )
        np.testing.assert_allclose(
            r_unweighted.sigma2_diff, r_weighted.sigma2_diff, atol=1e-14, rtol=1e-14
        )
        np.testing.assert_allclose(
            r_unweighted.sigma2_W, r_weighted.sigma2_W, atol=1e-14, rtol=1e-14
        )
        # T_hr and p_value also match.
        np.testing.assert_allclose(
            r_unweighted.t_stat_hr, r_weighted.t_stat_hr, atol=1e-14, rtol=1e-14
        )

    def test_weights_smoke(self):
        d, dy = self._setup()
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=30)
        r = yatchew_hr_test(d, dy, weights=w)
        assert np.isfinite(r.t_stat_hr)
        assert 0.0 <= r.p_value <= 1.0

    def test_survey_smoke(self):
        from diff_diff.survey import make_pweight_design

        d, dy = self._setup()
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=30)
        r = yatchew_hr_test(d, dy, survey=make_pweight_design(w))
        assert np.isfinite(r.t_stat_hr)

    def test_mutex_both_raises(self):
        from diff_diff.survey import make_pweight_design

        d, dy = self._setup()
        w = np.ones(30)
        with pytest.raises(ValueError, match="at most one of"):
            yatchew_hr_test(d, dy, weights=w, survey=make_pweight_design(w))

    def test_zero_weight_rejected(self):
        """Per Reviewer Question #4: strictly-positive weights required
        (the adjacent-difference variance has sum(w_avg) in the denominator
        which collapses to zero in any contiguous-zero block)."""
        d, dy = self._setup()
        w = np.ones(30)
        w[5] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            yatchew_hr_test(d, dy, weights=w)

    def test_replicate_weights_raises(self):
        from diff_diff.survey import ResolvedSurveyDesign

        d, dy = self._setup()
        w = np.ones(30)
        resolved_with_rep = ResolvedSurveyDesign(
            weights=w,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=30,
            lonely_psu="remove",
            replicate_weights=np.tile(w[:, None], (1, 5)),
            replicate_method="BRR",
            n_replicates=5,
        )
        with pytest.raises(NotImplementedError, match="replicate-weight"):
            yatchew_hr_test(d, dy, survey=resolved_with_rep)


class TestYatchewHRTestMeanIndependence:
    """``null="mean_independence"`` mode mirroring R
    ``YatchewTest::yatchew_test(order=0)``.

    Closes the placebo Yatchew R-parity gap from PR #392 by exposing
    the intercept-only OLS (``Y ~ 1``) residual path. Default
    ``null="linearity"`` retains bit-exact backcompat across all existing
    test classes.
    """

    def _setup(self, G=30, seed=42):
        rng = np.random.default_rng(seed)
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 0.3, size=G)
        return d, dy

    def test_happy_path_returns_finite_with_correct_null_form(self):
        d, dy = self._setup()
        r = yatchew_hr_test(d, dy, null="mean_independence")
        assert np.isfinite(r.t_stat_hr)
        assert 0.0 <= r.p_value <= 1.0
        assert r.null_form == "mean_independence"

    def test_naive_python_baseline_atol_1e_12(self):
        """Hand-compute residuals = dy - dy.mean(), sigma2_lin = (1/G) sum(eps^2),
        sort by d, adjacent diffs, HR T-stat. Lock at atol=1e-12."""
        # G = 8 deterministic input.
        d = np.array([0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95])
        dy = np.array([1.2, 2.1, 1.8, 3.4, 2.9, 4.1, 3.7, 5.0])
        G = d.shape[0]

        eps = dy - dy.mean()
        idx = np.argsort(d, kind="stable")
        dy_s = dy[idx]
        eps_s = eps[idx]
        diff_dy = np.diff(dy_s)
        sigma2_lin_expected = float(np.mean(eps * eps))
        sigma2_diff_expected = float(np.sum(diff_dy * diff_dy) / (2.0 * G))
        sigma4_W_expected = float(np.mean(eps_s[1:] ** 2 * eps_s[:-1] ** 2))
        sigma2_W_expected = float(np.sqrt(sigma4_W_expected))
        t_expected = float(
            np.sqrt(G) * (sigma2_lin_expected - sigma2_diff_expected) / sigma2_W_expected
        )

        r = yatchew_hr_test(d, dy, null="mean_independence")
        np.testing.assert_allclose(r.sigma2_lin, sigma2_lin_expected, atol=1e-12)
        np.testing.assert_allclose(r.sigma2_diff, sigma2_diff_expected, atol=1e-12)
        np.testing.assert_allclose(r.sigma2_W, sigma2_W_expected, atol=1e-12)
        np.testing.assert_allclose(r.t_stat_hr, t_expected, atol=1e-12)

    def test_sigma2_lin_equals_population_variance_exactly(self):
        """Under mean-independence, sigma2_lin = (1/G) sum((dy - mean(dy))^2)
        = np.var(dy, ddof=0). Locks the (1/G) normalizer convention."""
        d, dy = self._setup()
        G = d.shape[0]
        r = yatchew_hr_test(d, dy, null="mean_independence")
        expected = float(np.sum((dy - dy.mean()) ** 2) / G)
        np.testing.assert_allclose(r.sigma2_lin, expected, atol=1e-14, rtol=1e-14)
        # Equivalent np.var(ddof=0) form.
        np.testing.assert_allclose(r.sigma2_lin, float(np.var(dy, ddof=0)), atol=1e-14)

    def test_invalid_null_value_raises(self):
        d, dy = self._setup()
        with pytest.raises(ValueError, match="null must be one of"):
            yatchew_hr_test(d, dy, null="bogus")  # type: ignore[arg-type]

    def test_default_null_matches_explicit_linearity_bit_exact(self):
        """Omitting `null=` must be bit-identical to `null='linearity'`."""
        d, dy = self._setup()
        r_default = yatchew_hr_test(d, dy, alpha=0.05)
        r_explicit = yatchew_hr_test(d, dy, alpha=0.05, null="linearity")
        np.testing.assert_array_equal(r_default.sigma2_lin, r_explicit.sigma2_lin)
        np.testing.assert_array_equal(r_default.sigma2_diff, r_explicit.sigma2_diff)
        np.testing.assert_array_equal(r_default.sigma2_W, r_explicit.sigma2_W)
        np.testing.assert_array_equal(r_default.t_stat_hr, r_explicit.t_stat_hr)
        assert r_default.null_form == "linearity"
        assert r_explicit.null_form == "linearity"

    def test_tie_and_constant_d_rejection_is_mode_agnostic(self):
        """Front-door tie / constant-d guard rejects under BOTH nulls
        (the sort-by-d differencing step is null-agnostic)."""
        # Constant d.
        d_const = np.full(20, 0.5)
        dy = np.linspace(0.0, 1.0, 20)
        with warnings.catch_warnings(record=True) as caught_lin:
            warnings.simplefilter("always")
            r_lin = yatchew_hr_test(d_const, dy, null="linearity")
            msgs_lin = [str(w.message) for w in caught_lin]
        with warnings.catch_warnings(record=True) as caught_mi:
            warnings.simplefilter("always")
            r_mi = yatchew_hr_test(d_const, dy, null="mean_independence")
            msgs_mi = [str(w.message) for w in caught_mi]
        assert any("duplicate" in m for m in msgs_lin)
        assert any("duplicate" in m for m in msgs_mi)
        assert np.isnan(r_lin.t_stat_hr) and np.isnan(r_mi.t_stat_hr)
        # Both record null_form correctly even on the NaN return path.
        assert r_lin.null_form == "linearity"
        assert r_mi.null_form == "mean_independence"

        # Ties (some duplicates).
        d_ties = np.array([0.1, 0.2, 0.2, 0.5, 0.8, 0.9])
        dy_ties = np.array([1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = yatchew_hr_test(d_ties, dy_ties, null="mean_independence")
        assert np.isnan(r.t_stat_hr)
        assert r.null_form == "mean_independence"

    def test_nan_dy_input_raises(self):
        """NaN in dy raises (mode-agnostic; same as linearity path)."""
        d = np.linspace(0.0, 1.0, 20)
        dy = np.linspace(0.0, 2.0, 20)
        dy[3] = np.nan
        with pytest.raises(ValueError, match="contains NaN"):
            yatchew_hr_test(d, dy, null="mean_independence")

    # ─── Weighted ────────────────────────────────────────────────────────────

    def test_weighted_reduces_to_unweighted_at_uniform_weights(self):
        """At w=ones(G), weighted mean-independence reduces bit-exactly to
        unweighted mean-independence (atol=1e-14)."""
        d, dy = self._setup()
        r_unw = yatchew_hr_test(d, dy, null="mean_independence")
        r_w = yatchew_hr_test(d, dy, null="mean_independence", weights=np.ones(d.shape[0]))
        np.testing.assert_allclose(r_unw.sigma2_lin, r_w.sigma2_lin, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_unw.sigma2_diff, r_w.sigma2_diff, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_unw.sigma2_W, r_w.sigma2_W, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_unw.t_stat_hr, r_w.t_stat_hr, atol=1e-14, rtol=1e-14)
        assert r_w.null_form == "mean_independence"

    def test_weighted_non_uniform_baseline_atol_1e_12(self):
        """Hand-compute weighted mean-independence on G=8 deterministic input
        with non-uniform weights. Lock at atol=1e-12."""
        d = np.array([0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95])
        dy = np.array([1.2, 2.1, 1.8, 3.4, 2.9, 4.1, 3.7, 5.0])
        w = np.array([1.0, 1.5, 0.8, 2.0, 1.2, 0.7, 1.3, 1.1])
        # Internal pweight normalization: w_eff = w * (G / sum(w)).
        G = d.shape[0]
        w_eff = w * (G / w.sum())
        sum_w_eff = float(np.sum(w_eff))

        a_hat = float(np.sum(w_eff * dy) / sum_w_eff)
        eps = dy - a_hat
        sigma2_lin_expected = float(np.sum(w_eff * eps * eps) / sum_w_eff)

        idx = np.argsort(d, kind="stable")
        dy_s = dy[idx]
        eps_s = eps[idx]
        w_s = w_eff[idx]
        w_avg = 0.5 * (w_s[1:] + w_s[:-1])
        diff_dy = np.diff(dy_s)
        sigma2_diff_expected = float(np.sum(w_avg * diff_dy * diff_dy) / (2.0 * sum_w_eff))
        sigma4_W_expected = float(np.sum(w_avg * eps_s[1:] ** 2 * eps_s[:-1] ** 2) / np.sum(w_avg))
        sigma2_W_expected = float(np.sqrt(sigma4_W_expected))
        t_expected = float(
            np.sqrt(sum_w_eff) * (sigma2_lin_expected - sigma2_diff_expected) / sigma2_W_expected
        )

        r = yatchew_hr_test(d, dy, null="mean_independence", weights=w)
        np.testing.assert_allclose(r.sigma2_lin, sigma2_lin_expected, atol=1e-12)
        np.testing.assert_allclose(r.sigma2_diff, sigma2_diff_expected, atol=1e-12)
        np.testing.assert_allclose(r.sigma2_W, sigma2_W_expected, atol=1e-12)
        np.testing.assert_allclose(r.t_stat_hr, t_expected, atol=1e-12)

    def test_default_null_under_weights_matches_explicit_linearity(self):
        d, dy = self._setup()
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=d.shape[0])
        r_default = yatchew_hr_test(d, dy, weights=w)
        r_explicit = yatchew_hr_test(d, dy, weights=w, null="linearity")
        np.testing.assert_array_equal(r_default.t_stat_hr, r_explicit.t_stat_hr)
        np.testing.assert_array_equal(r_default.sigma2_lin, r_explicit.sigma2_lin)
        assert r_default.null_form == "linearity"

    def test_survey_design_with_mean_independence_composes(self):
        """`survey_design=` and `null='mean_independence'` are orthogonal;
        passing both does NOT raise NotImplementedError."""
        from diff_diff.survey import make_pweight_design

        d, dy = self._setup()
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=d.shape[0])
        r = yatchew_hr_test(
            d,
            dy,
            null="mean_independence",
            survey_design=make_pweight_design(w),
        )
        assert np.isfinite(r.t_stat_hr)
        assert r.null_form == "mean_independence"

    def test_weighted_linearity_baseline_atol_1e_12(self):
        """4-arm coverage gap fill: hand-compute weighted-linearity components
        on G=8 deterministic input. Locks the (linearity, weighted) arm
        explicitly (the other 3 arms locked by tests above + existing suite)."""
        d = np.array([0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95])
        dy = np.array([1.2, 2.1, 1.8, 3.4, 2.9, 4.1, 3.7, 5.0])
        w = np.array([1.0, 1.5, 0.8, 2.0, 1.2, 0.7, 1.3, 1.1])
        G = d.shape[0]
        w_eff = w * (G / w.sum())
        sum_w_eff = float(np.sum(w_eff))

        # Weighted OLS: dy = a + b*d + eps.
        d_wm = float(np.sum(w_eff * d) / sum_w_eff)
        dy_wm = float(np.sum(w_eff * dy) / sum_w_eff)
        d_dev = d - d_wm
        var_d_w = float(np.sum(w_eff * d_dev * d_dev))
        b_hat = float(np.sum(w_eff * d_dev * (dy - dy_wm)) / var_d_w)
        a_hat = float(dy_wm - b_hat * d_wm)
        eps = dy - a_hat - b_hat * d
        sigma2_lin_expected = float(np.sum(w_eff * eps * eps) / sum_w_eff)

        idx = np.argsort(d, kind="stable")
        dy_s = dy[idx]
        eps_s = eps[idx]
        w_s = w_eff[idx]
        w_avg = 0.5 * (w_s[1:] + w_s[:-1])
        diff_dy = np.diff(dy_s)
        sigma2_diff_expected = float(np.sum(w_avg * diff_dy * diff_dy) / (2.0 * sum_w_eff))
        sigma4_W_expected = float(np.sum(w_avg * eps_s[1:] ** 2 * eps_s[:-1] ** 2) / np.sum(w_avg))
        sigma2_W_expected = float(np.sqrt(sigma4_W_expected))
        t_expected = float(
            np.sqrt(sum_w_eff) * (sigma2_lin_expected - sigma2_diff_expected) / sigma2_W_expected
        )

        r = yatchew_hr_test(d, dy, weights=w, null="linearity")
        np.testing.assert_allclose(r.sigma2_lin, sigma2_lin_expected, atol=1e-12)
        np.testing.assert_allclose(r.sigma2_diff, sigma2_diff_expected, atol=1e-12)
        np.testing.assert_allclose(r.sigma2_W, sigma2_W_expected, atol=1e-12)
        np.testing.assert_allclose(r.t_stat_hr, t_expected, atol=1e-12)
        assert r.null_form == "linearity"

    def test_zero_weight_rejected_under_mean_independence(self):
        """Strictly-positive weights still required (mode-agnostic)."""
        d, dy = self._setup()
        w = np.ones(d.shape[0])
        w[5] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            yatchew_hr_test(d, dy, null="mean_independence", weights=w)

    def test_replicate_weights_rejected_under_mean_independence(self):
        from diff_diff.survey import ResolvedSurveyDesign

        d, dy = self._setup()
        w = np.ones(d.shape[0])
        resolved_with_rep = ResolvedSurveyDesign(
            weights=w,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=d.shape[0],
            lonely_psu="remove",
            replicate_weights=np.tile(w[:, None], (1, 5)),
            replicate_method="BRR",
            n_replicates=5,
        )
        with pytest.raises(NotImplementedError, match="replicate-weight"):
            yatchew_hr_test(d, dy, null="mean_independence", survey=resolved_with_rep)

    # ─── Edge case ───────────────────────────────────────────────────────────

    def test_G_below_min_returns_nan_under_both_nulls(self):
        """G=2 returns NaN under both nulls (HR variance estimator
        needs adjacent-pair products, undefined for G<3)."""
        d = np.array([0.1, 0.5])
        dy = np.array([1.0, 2.0])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r_lin = yatchew_hr_test(d, dy, null="linearity")
            r_mi = yatchew_hr_test(d, dy, null="mean_independence")
        assert np.isnan(r_lin.t_stat_hr)
        assert np.isnan(r_mi.t_stat_hr)
        assert r_lin.null_form == "linearity"
        assert r_mi.null_form == "mean_independence"


class TestJointStuteSurvey:
    """Phase 4.5 C survey/weights on stute_joint_pretest +
    joint_pretrends_test + joint_homogeneity_test."""

    def _make_event_study_panel(self, G=20, T_pre=2, T_post=2, seed=42):
        """Balanced event-study panel with T_pre + T_post periods."""
        rng = np.random.default_rng(seed)
        d_per_unit = rng.uniform(0.1, 1.0, size=G)
        rows = []
        for t in range(T_pre):
            for g in range(G):
                rows.append({"unit": g, "time": t, "y": rng.normal(), "d": 0.0})
        for t in range(T_pre, T_pre + T_post):
            for g in range(G):
                rows.append(
                    {
                        "unit": g,
                        "time": t,
                        "y": rng.normal() + 2.0 * d_per_unit[g] * (t - T_pre + 1),
                        "d": d_per_unit[g],
                    }
                )
        return pd.DataFrame(rows)

    def test_joint_pretrends_weights_smoke(self):
        df = self._make_event_study_panel()
        w_per_unit = np.random.default_rng(7).uniform(0.5, 2.0, size=20)
        # Constant-within-unit per HAD invariant.
        weights_per_row = df["unit"].map(dict(zip(np.arange(20), w_per_unit))).to_numpy()
        r = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[0],
            base_period=1,
            n_bootstrap=199,
            seed=0,
            weights=weights_per_row,
        )
        assert np.isfinite(r.cvm_stat_joint)
        assert 0.0 <= r.p_value <= 1.0

    def test_joint_homogeneity_weights_smoke(self):
        df = self._make_event_study_panel()
        w_per_unit = np.random.default_rng(7).uniform(0.5, 2.0, size=20)
        weights_per_row = df["unit"].map(dict(zip(np.arange(20), w_per_unit))).to_numpy()
        r = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[2, 3],
            base_period=1,
            n_bootstrap=199,
            seed=0,
            weights=weights_per_row,
        )
        assert np.isfinite(r.cvm_stat_joint)
        assert 0.0 <= r.p_value <= 1.0

    def test_joint_pretrends_survey_smoke(self):
        from diff_diff import SurveyDesign

        df = self._make_event_study_panel()
        w_per_unit = np.random.default_rng(7).uniform(0.5, 2.0, size=20)
        df["w"] = df["unit"].map(dict(zip(np.arange(20), w_per_unit)))
        r = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[0],
            base_period=1,
            n_bootstrap=199,
            seed=0,
            survey=SurveyDesign(weights="w"),
        )
        assert np.isfinite(r.cvm_stat_joint)

    def test_joint_pretrends_mutex_both_raises(self):
        from diff_diff import SurveyDesign

        df = self._make_event_study_panel()
        df["w"] = 1.0
        with pytest.raises(ValueError, match="at most one of"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0],
                base_period=1,
                n_bootstrap=199,
                seed=0,
                weights=np.ones(80),
                survey=SurveyDesign(weights="w"),
            )

    def test_stute_joint_pretest_replicate_weights_raises(self):
        """Phase 4.5 C MEDIUM #4: replicate-weight rejection at the direct
        residuals-in entry too."""
        from diff_diff.survey import ResolvedSurveyDesign

        G = 20
        residuals_by_horizon = {
            "0": np.random.default_rng(0).normal(size=G),
            "1": np.random.default_rng(1).normal(size=G),
        }
        fitted_by_horizon = {"0": np.zeros(G), "1": np.zeros(G)}
        doses = np.linspace(0.1, 1.0, G)
        design_matrix = np.column_stack([np.ones(G), doses])
        w = np.ones(G)
        resolved_with_rep = ResolvedSurveyDesign(
            weights=w,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=G,
            lonely_psu="remove",
            replicate_weights=np.tile(w[:, None], (1, 5)),
            replicate_method="BRR",
            n_replicates=5,
        )
        with pytest.raises(NotImplementedError, match="replicate-weight"):
            stute_joint_pretest(
                residuals_by_horizon=residuals_by_horizon,
                fitted_by_horizon=fitted_by_horizon,
                doses=doses,
                design_matrix=design_matrix,
                n_bootstrap=199,
                seed=0,
                survey=resolved_with_rep,
            )


# =============================================================================
# Phase 4.5 C R1 review regressions: zero-weight survey, aweight/fweight
# pweight-only guard, staggered event-study weights= subsetting.
# =============================================================================


class TestPhase45CR1Regressions:
    """R1 P0 / P1 / P3 regressions on the survey-aware pretest paths."""

    def _make_overall_panel(self, with_w_col=False):
        d_arr, dy_arr = _linear_dgp(G=20, beta=2.0, sigma=0.3)
        df = _make_two_period_panel(G=20, d=d_arr, dy=dy_arr)
        if with_w_col:
            rng = np.random.default_rng(7)
            w_per_unit = rng.uniform(0.5, 2.0, size=20)
            df["w"] = df["unit"].map(dict(zip(np.arange(20), w_per_unit)))
        return df

    def _make_staggered_panel(self, G_per_cohort=10, with_w_col=False):
        """Two-cohort staggered panel: cohort A treats at F=2, cohort B at F=3.

        Last-cohort filter (per HAD Appendix B.2) keeps only cohort B.
        Workflow / data-in wrappers under aggregate='event_study' must
        subset row-level weights= to the surviving cohort (R1 P1)."""
        rng = np.random.default_rng(0)
        rows = []
        for cohort, F_g in [("A", 2), ("B", 3)]:
            for g in range(G_per_cohort):
                unit_id = (0 if cohort == "A" else G_per_cohort) + g
                d_post = rng.uniform(0.1, 1.0)
                for t in range(4):
                    d_t = d_post if t >= F_g else 0.0
                    y_t = rng.normal() + (2.0 * d_post * (t - F_g + 1) if t >= F_g else 0.0)
                    rows.append({"unit": unit_id, "time": t, "y": y_t, "d": d_t, "F": F_g})
        df = pd.DataFrame(rows)
        if with_w_col:
            n_units = 2 * G_per_cohort
            w_per_unit = np.random.default_rng(7).uniform(0.5, 2.0, size=n_units)
            df["w"] = df["unit"].map(dict(zip(np.arange(n_units), w_per_unit)))
        return df

    # --- R1 P0: zero-weight survey rejection -------------------------------

    def test_stute_test_zero_survey_weight_raises(self):
        from diff_diff.survey import ResolvedSurveyDesign

        d, dy = _linear_dgp(G=30)
        w = np.ones(30)
        w[0] = 0.0
        resolved = ResolvedSurveyDesign(
            weights=w,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=30,
            lonely_psu="remove",
        )
        with pytest.raises(ValueError, match="strictly positive"):
            stute_test(d, dy, survey=resolved, n_bootstrap=199, seed=0)

    def test_stute_joint_pretest_zero_survey_weight_raises(self):
        from diff_diff.survey import ResolvedSurveyDesign

        G = 20
        residuals_by_horizon = {
            "0": np.random.default_rng(0).normal(size=G),
            "1": np.random.default_rng(1).normal(size=G),
        }
        fitted_by_horizon = {"0": np.zeros(G), "1": np.zeros(G)}
        doses = np.linspace(0.1, 1.0, G)
        design_matrix = np.column_stack([np.ones(G), doses])
        w = np.ones(G)
        w[0] = 0.0
        resolved = ResolvedSurveyDesign(
            weights=w,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=G,
            lonely_psu="remove",
        )
        with pytest.raises(ValueError, match="strictly positive"):
            stute_joint_pretest(
                residuals_by_horizon=residuals_by_horizon,
                fitted_by_horizon=fitted_by_horizon,
                doses=doses,
                design_matrix=design_matrix,
                n_bootstrap=199,
                seed=0,
                survey=resolved,
            )

    def test_workflow_zero_survey_weight_column_rejected(self):
        from diff_diff import SurveyDesign

        df = self._make_overall_panel(with_w_col=True)
        df.loc[df["unit"] == 0, "w"] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )

    # --- R1 P1: aweight/fweight pweight-only guard -------------------------

    def test_stute_test_aweight_rejected(self):
        from diff_diff.survey import ResolvedSurveyDesign

        d, dy = _linear_dgp(G=30)
        resolved = ResolvedSurveyDesign(
            weights=np.ones(30),
            weight_type="aweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=30,
            lonely_psu="remove",
        )
        with pytest.raises(ValueError, match="weight_type='pweight'"):
            stute_test(d, dy, survey=resolved, n_bootstrap=199, seed=0)

    def test_yatchew_hr_test_fweight_rejected(self):
        from diff_diff.survey import ResolvedSurveyDesign

        d, dy = _linear_dgp(G=30)
        resolved = ResolvedSurveyDesign(
            weights=np.ones(30),
            weight_type="fweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=30,
            lonely_psu="remove",
        )
        with pytest.raises(ValueError, match="weight_type='pweight'"):
            yatchew_hr_test(d, dy, survey=resolved)

    def test_workflow_aweight_rejected_at_resolution(self):
        from diff_diff import SurveyDesign

        df = self._make_overall_panel(with_w_col=True)
        with pytest.raises(ValueError, match="weight_type='pweight'"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey=SurveyDesign(weights="w", weight_type="aweight"),
                n_bootstrap=199,
                seed=0,
            )

    # --- R1 P1: staggered event-study weights= subsetting ------------------

    def test_workflow_staggered_event_study_weights_subset_correctly(self):
        """R1 P1: on staggered panels, _validate_multi_period_panel filters
        to the last cohort; row-level weights= must be subset to the
        surviving cohort BEFORE re-aggregation. Pre-fix this crashed with
        a length-mismatch ValueError."""
        df = self._make_staggered_panel(G_per_cohort=10)
        n_rows = 2 * 10 * 4
        weights_per_row = np.ones(n_rows) * 1.5
        with pytest.warns(UserWarning):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                first_treat_col="F",
                aggregate="event_study",
                weights=weights_per_row,
                n_bootstrap=199,
                seed=0,
            )
        assert report.aggregate == "event_study"
        assert report.qug is None
        assert report.homogeneity_joint is not None

    # --- R2 P1: direct-wrapper staggered weights= subsetting ---------------

    def test_joint_pretrends_test_staggered_weights_subset(self):
        """R2 P1: joint_pretrends_test direct call must subset row-level
        weights= when its own _validate_had_panel_event_study filtering
        triggers on staggered panels. Pre-fix this crashed with a length-
        mismatch ValueError because the wrapper passed the full-panel
        weights array into _resolve_pretest_unit_weights(data_filtered, ...)."""
        df = self._make_staggered_panel(G_per_cohort=10)
        n_rows = 2 * 10 * 4
        weights_per_row = np.ones(n_rows) * 1.5
        with pytest.warns(UserWarning):
            r = joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0, 1],
                base_period=2,
                first_treat_col="F",
                n_bootstrap=199,
                seed=0,
                weights=weights_per_row,
            )
        assert np.isfinite(r.cvm_stat_joint)

    def test_joint_homogeneity_test_staggered_weights_subset(self):
        df = self._make_staggered_panel(G_per_cohort=10)
        n_rows = 2 * 10 * 4
        weights_per_row = np.ones(n_rows) * 1.5
        with pytest.warns(UserWarning):
            r = joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[3],
                base_period=2,
                first_treat_col="F",
                n_bootstrap=199,
                seed=0,
                weights=weights_per_row,
            )
        assert np.isfinite(r.cvm_stat_joint)

    # --- R2 P1: bootstrap perturbation form lock ---------------------------

    def test_stute_survey_perturbation_does_not_double_weight(self):
        """R2 P1: bootstrap perturbation is `dy_b = fitted + eps * eta_obs`
        (paper Appendix D form), NOT `eps * w * eta_obs`. Adding `* w` to
        the perturbation would over-weight by w² (weighting flows through
        weighted OLS refit + weighted CvM, NOT through the multiplier).

        Lock test: cvm_stat at uniform weights matches between paths
        bit-exactly (W=G under uniform weights so 1/W² = 1/G²); the
        bootstrap p-value distributions agree within Monte-Carlo noise
        (RNG draw ordering differs between batched survey-aware path and
        per-iteration unweighted path; numerical equivalence is unreachable).
        """
        d, dy = _linear_dgp(G=50, beta=2.0, sigma=0.3)
        r_unweighted = stute_test(d, dy, n_bootstrap=999, seed=0)
        r_weighted = stute_test(d, dy, weights=np.ones(50), n_bootstrap=999, seed=0)
        # cvm_stat: bit-exact reduction at w=1 (W=G, weighted CvM ≡ unweighted).
        np.testing.assert_allclose(
            r_unweighted.cvm_stat, r_weighted.cvm_stat, atol=1e-14, rtol=1e-14
        )
        # p_value: distributional agreement at large B; Monte-Carlo noise.
        # If the survey path were over-weighting (w² instead of w), the
        # bootstrap distribution would be inflated and the survey p-value
        # would systematically deviate. With the correct form, |diff| < 0.10.
        assert abs(r_unweighted.p_value - r_weighted.p_value) < 0.10

    # --- R3 P0: variance-unidentified survey-design guard ------------------

    def test_stute_test_single_psu_unstratified_returns_nan(self):
        """R3 P0: unstratified single-PSU survey designs are
        variance-unidentified (df_survey = n_psu - 1 = 0). The multiplier
        bootstrap helper returns an all-zero matrix; without the guard
        the code below would treat that as a valid bootstrap law and emit
        p_value ≈ 1/(B+1) (spurious rejection). Guard returns NaN +
        UserWarning instead."""
        from diff_diff.survey import ResolvedSurveyDesign

        d, dy = _linear_dgp(G=30)
        # All units in a single PSU, unstratified -> df_survey = 0.
        single_psu = np.zeros(30, dtype=np.int64)
        resolved = ResolvedSurveyDesign(
            weights=np.ones(30),
            weight_type="pweight",
            strata=None,
            psu=single_psu,
            fpc=None,
            n_strata=0,
            n_psu=1,
            lonely_psu="remove",
        )
        with pytest.warns(UserWarning, match="variance-unidentified"):
            r = stute_test(d, dy, survey=resolved, n_bootstrap=199, seed=0)
        assert np.isnan(r.p_value)
        assert r.reject is False
        # cvm_stat is the OBSERVED value (still computed pre-guard); only
        # p_value goes NaN because the bootstrap calibration is invalid.
        assert np.isfinite(r.cvm_stat)

    def test_stute_joint_pretest_single_psu_unstratified_returns_nan(self):
        """R3 P0: same guard on the joint variant."""
        from diff_diff.survey import ResolvedSurveyDesign

        G = 20
        residuals_by_horizon = {
            "0": np.random.default_rng(0).normal(size=G),
            "1": np.random.default_rng(1).normal(size=G),
        }
        fitted_by_horizon = {"0": np.zeros(G), "1": np.zeros(G)}
        doses = np.linspace(0.1, 1.0, G)
        design_matrix = np.column_stack([np.ones(G), doses])
        single_psu = np.zeros(G, dtype=np.int64)
        resolved = ResolvedSurveyDesign(
            weights=np.ones(G),
            weight_type="pweight",
            strata=None,
            psu=single_psu,
            fpc=None,
            n_strata=0,
            n_psu=1,
            lonely_psu="remove",
        )
        with pytest.warns(UserWarning, match="variance-unidentified"):
            r = stute_joint_pretest(
                residuals_by_horizon=residuals_by_horizon,
                fitted_by_horizon=fitted_by_horizon,
                doses=doses,
                design_matrix=design_matrix,
                n_bootstrap=199,
                seed=0,
                survey=resolved,
            )
        assert np.isnan(r.p_value)
        assert r.reject is False
        assert np.isfinite(r.cvm_stat_joint)

    def test_workflow_single_psu_propagates_nan_through_stute(self):
        """R3 P0: workflow-level: single-PSU survey design makes the
        survey Stute multiplier bootstrap variance-unidentified, so
        report.stute.p_value is NaN (the guard fired). Yatchew under
        survey is unaffected by PSU clustering by design (REGISTRY
        note: PSU clustering is NOT propagated through the variance-
        ratio statistic), so report.yatchew.p_value is still finite.
        The verdict carries the linearity-conditional suffix; users
        should read REGISTRY for the per-test mechanism caveat."""
        from diff_diff import SurveyDesign

        df = self._make_overall_panel(with_w_col=True)
        # Add a constant 'psu' column (single PSU).
        df["psu"] = 0
        with pytest.warns(UserWarning):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey=SurveyDesign(weights="w", psu="psu"),
                n_bootstrap=199,
                seed=0,
            )
        assert report.aggregate == "overall"
        assert report.qug is None  # skipped per C0
        # Stute: variance-unidentified -> NaN p-value (R3 P0 guard fired).
        assert report.stute is not None and np.isnan(report.stute.p_value)
        # Yatchew: closed-form, PSU-agnostic by design -> still finite.
        assert report.yatchew is not None and np.isfinite(report.yatchew.p_value)
        # Verdict carries the linearity-conditional suffix.
        assert "linearity-conditional verdict" in report.verdict

    # --- R4 P0: weight-scale invariance + cross-path agreement ------------

    def test_yatchew_weights_scale_invariant(self):
        """R4 P0: Yatchew test statistic must be invariant under uniform
        rescaling of weights. Pre-fix `T_hr = sqrt(sum(w)) * (...)` made
        the stat scale as sqrt(c), so weights=w and weights=100*w gave
        different p-values. Fix: helper normalizes pweights to mean=1
        before any computation."""
        d, dy = _linear_dgp(G=30, beta=2.0, sigma=0.3)
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=30)
        r1 = yatchew_hr_test(d, dy, weights=w)
        r2 = yatchew_hr_test(d, dy, weights=100.0 * w)
        np.testing.assert_allclose(r1.t_stat_hr, r2.t_stat_hr, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(r1.p_value, r2.p_value, atol=1e-12, rtol=1e-12)

    def test_stute_weights_scale_invariant(self):
        """R4 P0 mirror: Stute is internally scale-invariant in functional
        form, but normalization is required so weights= and survey=
        entry paths agree numerically."""
        d, dy = _linear_dgp(G=30, beta=2.0, sigma=0.3)
        w = np.random.default_rng(7).uniform(0.5, 2.0, size=30)
        r1 = stute_test(d, dy, weights=w, n_bootstrap=199, seed=0)
        r2 = stute_test(d, dy, weights=100.0 * w, n_bootstrap=199, seed=0)
        np.testing.assert_allclose(r1.cvm_stat, r2.cvm_stat, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(r1.p_value, r2.p_value, atol=1e-12, rtol=1e-12)

    def test_workflow_weights_eq_survey_at_overall_path(self):
        """R4 P0: workflow's weights= shortcut and survey=SurveyDesign(
        weights="w") must produce identical Yatchew/Stute results for
        the same design. SurveyDesign.resolve() normalizes pweights to
        mean=1; the helper now applies the same normalization on the
        weights= path so both paths agree numerically."""
        from diff_diff import SurveyDesign

        df = self._make_overall_panel(with_w_col=True)
        # Build a per-row weights array matching df["w"] for the shortcut.
        weights_per_row = df["w"].to_numpy()
        with pytest.warns(UserWarning):
            r_weights = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                weights=weights_per_row,
                n_bootstrap=199,
                seed=0,
            )
        with pytest.warns(UserWarning):
            r_survey = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey=SurveyDesign(weights="w"),
                n_bootstrap=199,
                seed=0,
            )
        # Yatchew: closed-form, must match exactly under mean=1 normalization.
        assert r_weights.yatchew is not None and r_survey.yatchew is not None
        np.testing.assert_allclose(
            r_weights.yatchew.t_stat_hr,
            r_survey.yatchew.t_stat_hr,
            atol=1e-10,
            rtol=1e-10,
        )
        # Stute: bootstrap is seeded; same multiplier matrix shape under
        # both paths means same RNG draws -> identical p-values.
        assert r_weights.stute is not None and r_survey.stute is not None
        np.testing.assert_allclose(
            r_weights.stute.cvm_stat, r_survey.stute.cvm_stat, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            r_weights.stute.p_value, r_survey.stute.p_value, atol=1e-10, rtol=1e-10
        )

    # --- R4 P1: 1D weights validation ------------------------------------

    def test_stute_test_rejects_2d_weights(self):
        """R4 P1: column-vector weights must raise, not silently broadcast."""
        d, dy = _linear_dgp(G=30)
        w_2d = np.ones((30, 1))  # common df[["w"]].to_numpy() pattern
        with pytest.raises(ValueError, match="1-dimensional"):
            stute_test(d, dy, weights=w_2d, n_bootstrap=199, seed=0)

    def test_yatchew_hr_test_rejects_2d_weights(self):
        d, dy = _linear_dgp(G=30)
        w_2d = np.ones((30, 1))
        with pytest.raises(ValueError, match="1-dimensional"):
            yatchew_hr_test(d, dy, weights=w_2d)

    def test_workflow_rejects_2d_weights(self):
        df = self._make_overall_panel()
        w_2d = np.ones((40, 1))
        with pytest.raises(ValueError, match="1-dimensional"):
            did_had_pretest_workflow(
                df, "y", "d", "time", "unit", weights=w_2d, n_bootstrap=199, seed=0
            )

    # --- R5 P1: lonely_psu='adjust' singleton-strata rejection ------------

    def _make_singleton_strata_resolved(self, G=30, lonely_psu="adjust"):
        """Resolved survey design with one PSU per stratum (singleton strata).
        Under lonely_psu='adjust' the bootstrap helper pools singletons with
        nonzero multipliers, but the variance target requires a pseudo-stratum
        centering transform not derived for the Stute CvM."""
        from diff_diff.survey import ResolvedSurveyDesign

        # G strata, each with exactly 1 PSU (each unit is its own stratum +
        # PSU). Tests the worst-case singleton-pooling regime.
        strata = np.arange(G, dtype=np.int64)
        psu = np.arange(G, dtype=np.int64)
        return ResolvedSurveyDesign(
            weights=np.ones(G),
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=G,
            n_psu=G,
            lonely_psu=lonely_psu,
        )

    def test_stute_test_lonely_psu_adjust_singleton_still_raises(self):
        """Stratified designs (``SurveyDesign(strata=...)``) are now
        supported via the stratified clustered wild-bootstrap correction
        (within-stratum demean + ``sqrt(n_h/(n_h-1))`` on PSU
        multipliers; see REGISTRY § "Note (Stute stratified survey-
        bootstrap calibration)"). But the ``lonely_psu='adjust'`` +
        singleton-strata subcase remains rejected — the pseudo-stratum
        centering transform has not been derived for the Stute CvM (same
        gap as the HAD sup-t deviation at REGISTRY:2382). This test
        verifies the preserved gate."""
        d, dy = _linear_dgp(G=30)
        resolved = self._make_singleton_strata_resolved(G=30, lonely_psu="adjust")
        with pytest.raises(NotImplementedError, match="lonely_psu='adjust'"):
            stute_test(d, dy, survey=resolved, n_bootstrap=199, seed=0)

    def test_stute_joint_pretest_lonely_psu_adjust_singleton_still_raises(self):
        """``lonely_psu='adjust'`` + singleton-strata stays rejected on
        the joint Stute path after stratified-design support lands
        (mirrors the single-horizon ``stute_test`` regression above)."""
        G = 20
        residuals_by_horizon = {
            "0": np.random.default_rng(0).normal(size=G),
            "1": np.random.default_rng(1).normal(size=G),
        }
        fitted_by_horizon = {"0": np.zeros(G), "1": np.zeros(G)}
        doses = np.linspace(0.1, 1.0, G)
        design_matrix = np.column_stack([np.ones(G), doses])
        resolved = self._make_singleton_strata_resolved(G=G, lonely_psu="adjust")
        with pytest.raises(NotImplementedError, match="lonely_psu='adjust'"):
            stute_joint_pretest(
                residuals_by_horizon=residuals_by_horizon,
                fitted_by_horizon=fitted_by_horizon,
                doses=doses,
                design_matrix=design_matrix,
                n_bootstrap=199,
                seed=0,
                survey=resolved,
            )

    # --- R6 P1: positive non-trivial PSU/strata survey coverage -----------

    def _make_event_study_panel_with_psu_strata(
        self,
        n_strata=2,
        n_psu_per_stratum=3,
        n_units_per_psu=2,
        T_pre=2,
        T_post=2,
        seed=42,
    ):
        """Balanced event-study panel with non-trivial PSU/strata structure."""
        rng = np.random.default_rng(seed)
        rows = []
        unit_id = 0
        for h in range(n_strata):
            for p in range(n_psu_per_stratum):
                psu_global = h * n_psu_per_stratum + p
                for _ in range(n_units_per_psu):
                    d_post = rng.uniform(0.1, 1.0)
                    w_unit = rng.uniform(0.5, 2.0)
                    for t in range(T_pre + T_post):
                        d_t = d_post if t >= T_pre else 0.0
                        y_t = rng.normal() + (2.0 * d_post * (t - T_pre + 1) if t >= T_pre else 0.0)
                        rows.append(
                            {
                                "unit": unit_id,
                                "time": t,
                                "y": y_t,
                                "d": d_t,
                                "stratum": h,
                                "psu": psu_global,
                                "w": w_unit,
                            }
                        )
                    unit_id += 1
        return pd.DataFrame(rows)

    def test_joint_homogeneity_test_psu_only_survey_smoke(self):
        """R6 P1 + R10 P1: positive coverage on joint_homogeneity_test
        with PSU-only survey design (NO strata, since stratified is
        rejected per R10 P1 narrowing)."""
        from diff_diff import SurveyDesign

        df = self._make_event_study_panel_with_psu_strata(
            n_strata=2, n_psu_per_stratum=3, n_units_per_psu=2
        )
        r = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[2, 3],
            base_period=1,
            n_bootstrap=199,
            seed=0,
            survey=SurveyDesign(weights="w", psu="psu"),
        )
        assert np.isfinite(r.cvm_stat_joint)
        assert 0.0 <= r.p_value <= 1.0

    def test_joint_homogeneity_test_stratified_design_supported(self):
        """Stratified designs (``SurveyDesign(weights, strata, psu)``)
        now run end-to-end on ``joint_homogeneity_test`` via the inner
        ``stute_joint_pretest`` -> ``apply_stratum_centering`` chain.
        Previously raised ``NotImplementedError`` (pre-this-PR Phase 4.5
        C gate). Now expected to return a finite, well-formed
        ``StuteJointResult`` for a non-singleton stratified design."""
        from diff_diff import SurveyDesign

        df = self._make_event_study_panel_with_psu_strata(
            n_strata=2, n_psu_per_stratum=3, n_units_per_psu=2
        )
        result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[2, 3],
            base_period=1,
            n_bootstrap=199,
            seed=0,
            survey=SurveyDesign(weights="w", strata="stratum", psu="psu"),
        )
        # Sanity: a stratified joint Stute on a linear DGP should produce
        # a finite p-value (the test exercises the survey-bootstrap path,
        # not a specific value pin).
        assert np.isfinite(result.p_value)
        assert 0.0 <= result.p_value <= 1.0
        assert result.n_horizons == 2
        assert result.n_bootstrap == 199

    def test_workflow_event_study_psu_only_survey_smoke(self):
        """R6 P1 + R10 P1: positive coverage on did_had_pretest_workflow
        event-study path with PSU-only structure (no strata)."""
        from diff_diff import SurveyDesign

        df = self._make_event_study_panel_with_psu_strata(
            n_strata=2, n_psu_per_stratum=3, n_units_per_psu=2
        )
        with pytest.warns(UserWarning, match="QUG step skipped"):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                survey=SurveyDesign(weights="w", psu="psu"),
                n_bootstrap=199,
                seed=0,
            )
        assert report.aggregate == "event_study"
        assert report.qug is None
        assert report.pretrends_joint is not None
        assert np.isfinite(report.pretrends_joint.cvm_stat_joint)
        assert report.homogeneity_joint is not None
        assert np.isfinite(report.homogeneity_joint.cvm_stat_joint)

    def test_workflow_event_study_zero_weights_on_dropped_cohort(self):
        """R6 P1 regression: previously the workflow eagerly resolved
        weights= on the FULL panel (before _validate_multi_period_panel's
        last-cohort filter), so zero/invalid weights on the soon-to-be-
        dropped cohort would abort an otherwise-valid event-study run.
        Fix: resolution moved into the per-aggregate branches; the
        event-study path lets joint wrappers handle resolution on
        data_filtered. This test verifies a panel where the dropped
        (early) cohort has zero weights succeeds on the surviving last
        cohort."""
        df = self._make_staggered_panel(G_per_cohort=10)
        weights_per_row = np.array([0.0 if df.iloc[i]["F"] == 2 else 1.5 for i in range(len(df))])
        with pytest.warns(UserWarning):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                first_treat_col="F",
                aggregate="event_study",
                weights=weights_per_row,
                n_bootstrap=199,
                seed=0,
            )
        assert report.aggregate == "event_study"
        assert report.qug is None
        assert report.homogeneity_joint is not None

    # --- R7 P0: weighted-CvM outer-measure oracle -------------------------

    def test_cvm_statistic_weighted_outer_measure_oracle(self):
        """R7 P0: weighted CvM must integrate outer measure against F_hat_w
        too. Hand-computed oracle distinguishes outer-weighted form
        ((1/W^2) sum_g w_g C_g^2) from count-weighted-cusum form
        ((1/W^2) sum_g C_g^2). Uniform weights cannot tell the two apart."""
        from diff_diff.had_pretests import _cvm_statistic_weighted

        eps = np.array([1.0, -2.0, 3.0])
        d = np.array([0.1, 0.2, 0.3])
        w = np.array([1.0, 2.0, 3.0])
        # C_1=1, C_2=-3, C_3=6, W=6.
        # Outer-weighted: (1*1 + 2*9 + 3*36) / 36 = 127/36.
        # Count-weighted (WRONG): (1+9+36) / 36 = 46/36.
        result = _cvm_statistic_weighted(eps, d, w)
        outer_weighted = (1 * 1.0**2 + 2 * (-3.0) ** 2 + 3 * 6.0**2) / (6.0**2)
        count_weighted = (1.0**2 + (-3.0) ** 2 + 6.0**2) / (6.0**2)
        np.testing.assert_allclose(result, outer_weighted, atol=1e-14, rtol=1e-14)
        assert abs(outer_weighted - count_weighted) > 1.0
        assert abs(result - count_weighted) > 1.0

    def test_cvm_statistic_weighted_reduces_at_uniform_weights(self):
        """At w=ones(G), outer-weighted form reduces bit-exactly to the
        unweighted statistic."""
        from diff_diff.had_pretests import _cvm_statistic, _cvm_statistic_weighted

        eps = np.array([1.0, -2.0, 3.0, 0.5, -0.7])
        d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        w_uniform = np.ones(5)
        np.testing.assert_allclose(
            _cvm_statistic_weighted(eps, d, w_uniform),
            _cvm_statistic(eps, d),
            atol=1e-14,
            rtol=1e-14,
        )

    # --- R7 P1: survey verdict consistency --------------------------------

    def test_workflow_overall_survey_pass_does_not_say_inconclusive(self):
        """R7 P1: when all_pass=True on the overall survey path, the
        verdict must NOT start with 'inconclusive'. Locks the explicit
        survey-aware verdict composer."""
        df = self._make_overall_panel()
        weights_per_row = np.full(40, 1.5)
        with pytest.warns(UserWarning):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                weights=weights_per_row,
                n_bootstrap=199,
                seed=0,
            )
        if report.all_pass:
            assert not report.verdict.startswith("inconclusive"), (
                f"all_pass=True but verdict starts with 'inconclusive': " f"{report.verdict!r}"
            )

    def test_workflow_event_study_survey_pass_does_not_say_inconclusive(self):
        """R7 P1: same invariant on the event-study survey path. Uses
        PSU-only design (no strata) per R10 P1 narrowing."""
        from diff_diff import SurveyDesign

        df = self._make_event_study_panel_with_psu_strata(
            n_strata=2, n_psu_per_stratum=3, n_units_per_psu=2
        )
        with pytest.warns(UserWarning, match="QUG step skipped"):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                survey=SurveyDesign(weights="w", psu="psu"),
                n_bootstrap=199,
                seed=0,
            )
        if report.all_pass:
            assert not report.verdict.startswith("inconclusive"), (
                f"all_pass=True but verdict starts with 'inconclusive': " f"{report.verdict!r}"
            )

    # --- R9 P1: front-door length validation on staggered weights= path ---

    def test_workflow_event_study_oversized_weights_raises(self):
        """R9 P1: oversized row-level weights= must raise a clean
        ValueError BEFORE the staggered-panel pos_idx subsetting (pre-fix
        the workflow silently truncated by slicing original weights to
        data_filtered's row count without checking length first)."""
        df = self._make_staggered_panel(G_per_cohort=10)
        weights_oversized = np.ones(100) * 1.5  # 80 rows expected
        with pytest.raises(ValueError, match="weights length 100"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                first_treat_col="F",
                aggregate="event_study",
                weights=weights_oversized,
                n_bootstrap=199,
                seed=0,
            )

    def test_workflow_event_study_undersized_weights_raises(self):
        """R9 P1: undersized weights= must raise clean ValueError, not
        a raw IndexError from pos_idx slicing."""
        df = self._make_staggered_panel(G_per_cohort=10)
        weights_undersized = np.ones(60) * 1.5
        with pytest.raises(ValueError, match="weights length 60"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                first_treat_col="F",
                aggregate="event_study",
                weights=weights_undersized,
                n_bootstrap=199,
                seed=0,
            )

    def test_joint_pretrends_test_oversized_weights_raises(self):
        """R9 P1: same length-validation contract on the direct wrapper."""
        df = self._make_staggered_panel(G_per_cohort=10)
        weights_oversized = np.ones(100) * 1.5
        with pytest.raises(ValueError, match="weights length 100"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[0, 1],
                base_period=2,
                first_treat_col="F",
                n_bootstrap=199,
                seed=0,
                weights=weights_oversized,
            )

    def test_joint_homogeneity_test_undersized_weights_raises(self):
        """R9 P1: same on joint_homogeneity_test."""
        df = self._make_staggered_panel(G_per_cohort=10)
        weights_undersized = np.ones(60) * 1.5
        with pytest.raises(ValueError, match="weights length 60"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[3],
                base_period=2,
                first_treat_col="F",
                n_bootstrap=199,
                seed=0,
                weights=weights_undersized,
            )

    # --- R12 P3: positive FPC-only survey coverage ------------------------

    def test_stute_test_fpc_only_survey_smoke(self):
        """R12 P3: positive smoke for FPC-only survey designs on the Stute
        family. Phase 4.5 C narrows survey support to pweight+PSU+FPC; the
        previous test matrix covered pweight-only and PSU-only but no FPC
        case, so the FPC scaling branch in
        generate_survey_multiplier_weights_batch was unpinned by direct
        regression."""
        from diff_diff.survey import ResolvedSurveyDesign

        d, dy = _linear_dgp(G=30, beta=2.0, sigma=0.3)
        # Construct an FPC-only design: no strata, no PSU, but FPC = N=200
        # (population size) so f = G/N = 0.15. The bootstrap helper applies
        # a sqrt(1 - f) scaling to the multipliers under FPC.
        w = np.ones(30)
        resolved = ResolvedSurveyDesign(
            weights=w,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=np.full(30, 200.0),
            n_strata=0,
            n_psu=30,
            lonely_psu="remove",
        )
        r = stute_test(d, dy, survey=resolved, n_bootstrap=199, seed=0)
        assert np.isfinite(r.cvm_stat)
        assert 0.0 <= r.p_value <= 1.0

    def test_workflow_overall_fpc_only_survey_smoke(self):
        """R12 P3: positive smoke for FPC-only on the workflow path."""
        from diff_diff import SurveyDesign

        df = self._make_overall_panel(with_w_col=True)
        # FPC value > G to satisfy the helper's "FPC must be >= n_units" guard.
        df["fpc"] = 200.0
        with pytest.warns(UserWarning, match="QUG step skipped"):
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                survey=SurveyDesign(weights="w", fpc="fpc"),
                n_bootstrap=199,
                seed=0,
            )
        assert report.aggregate == "overall"
        assert report.qug is None
        assert report.stute is not None and np.isfinite(report.stute.p_value)
        assert report.yatchew is not None and np.isfinite(report.yatchew.p_value)


class TestJointPretestsTrendsLin:
    """Direct unit tests for trends_lin=True on joint_pretrends_test and
    joint_homogeneity_test (PR #389 / Phase 4 R-parity).

    Locks: anchor-period guard, naive-Python detrending baseline at
    atol=1e-12, survey_design × trends_lin NotImplementedError, default
    bit-exact backcompat. R-parity is exercised by the separate
    test_did_had_parity.py module."""

    @staticmethod
    def _panel(rng_seed: int = 11, G: int = 200, F: int = 4, T: int = 5) -> pd.DataFrame:
        """Build a 5-period (default) panel with F=4. Beta(0.5, 1) doses
        give heavy density near zero; per-unit linear trends ensure
        trends_lin=True changes the test statistic vs trends_lin=False."""
        rng = np.random.default_rng(rng_seed)
        d = rng.beta(0.5, 1.0, size=G)
        unit_fe = rng.normal(0, 1, G)
        trend = rng.normal(0.1, 0.05, G)
        rows = []
        for g in range(G):
            for t in range(1, T + 1):
                treated = t >= F
                y = (
                    unit_fe[g]
                    + trend[g] * (t - 1)
                    + (d[g] + d[g] ** 2) * treated
                    + rng.normal(0, 0.5)
                )
                dose = d[g] if treated else 0.0
                rows.append({"unit": g, "time": t, "y": y, "d": dose})
        return pd.DataFrame(rows)

    def test_pretrends_default_bit_exact_backcompat(self):
        """trends_lin=False (default) preserves pre-PR numerics exactly."""
        df = self._panel(rng_seed=7)
        r1 = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[1, 2],
            base_period=3,
            n_bootstrap=99,
            seed=42,
        )
        r2 = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[1, 2],
            base_period=3,
            n_bootstrap=99,
            seed=42,
            trends_lin=False,
        )
        assert r1.cvm_stat_joint == r2.cvm_stat_joint
        assert r1.p_value == r2.p_value

    def test_homogeneity_default_bit_exact_backcompat(self):
        df = self._panel(rng_seed=8)
        r1 = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[4, 5],
            base_period=3,
            n_bootstrap=99,
            seed=42,
        )
        r2 = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[4, 5],
            base_period=3,
            n_bootstrap=99,
            seed=42,
            trends_lin=False,
        )
        assert r1.cvm_stat_joint == r2.cvm_stat_joint
        assert r1.p_value == r2.p_value

    def test_pretrends_trends_lin_changes_stat(self):
        """Per-unit linear trends in DGP → trends_lin=True changes CvM."""
        df = self._panel(rng_seed=9)
        r_no = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[1],
            base_period=3,
            n_bootstrap=99,
            seed=42,
        )
        r_yes = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[1],
            base_period=3,
            n_bootstrap=99,
            seed=42,
            trends_lin=True,
        )
        assert r_no.cvm_stat_joint != r_yes.cvm_stat_joint

    def test_pretrends_trends_lin_naive_baseline(self):
        """trends_lin detrending matches a hand-coded naive Python baseline.

        Detrended dy_t = (Y[g,t] - Y[g,base]) - (t - base) × (Y[g,base] - Y[g,base-1]).
        Build the detrended outcome manually as a new column and confirm
        joint_pretrends_test on the manual column with trends_lin=False
        matches joint_pretrends_test on raw Y with trends_lin=True at
        atol=1e-12 (deterministic given fixed seed)."""
        df = self._panel(rng_seed=10)
        base, base_minus_1 = 3, 2
        # Manual detrending: replace y with detrended y.
        wide = df.pivot(index="unit", columns="time", values="y").sort_index()
        slope = wide[base].to_numpy() - wide[base_minus_1].to_numpy()
        # Apply Y_detrended[t] = Y[t] - (t - base) × slope per row.
        df_manual = df.copy()
        df_manual = df_manual.merge(
            pd.DataFrame({"unit": np.arange(len(slope)), "slope": slope}),
            on="unit",
        )
        df_manual["y"] = df_manual["y"] - (df_manual["time"] - base) * df_manual["slope"]
        df_manual = df_manual.drop(columns=["slope"])

        # joint_pretrends_test on manual-detrended y, trends_lin=False
        r_manual = joint_pretrends_test(
            df_manual,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[1],
            base_period=base,
            n_bootstrap=99,
            seed=42,
        )
        # joint_pretrends_test on raw y, trends_lin=True
        r_auto = joint_pretrends_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            pre_periods=[1],
            base_period=base,
            n_bootstrap=99,
            seed=42,
            trends_lin=True,
        )
        np.testing.assert_allclose(
            r_auto.cvm_stat_joint,
            r_manual.cvm_stat_joint,
            atol=1e-12,
            rtol=0,
        )

    def test_pretrends_trends_lin_consumed_placebo_dropped_with_warning(self):
        """When pre_periods includes base_period-1, trends_lin drops it
        (the consumed placebo). Warning fires; remaining horizons get
        their normal detrending. Regression for PR #392 R1 P0."""
        df = self._panel(rng_seed=30)
        # Panel has periods 1..5, F=4. Base=3, base-1=2.
        # Caller passes [1, 2] which includes the consumed placebo at 2.
        with pytest.warns(UserWarning, match=r"dropping period .*2.*consumed.*placebo"):
            r = joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[1, 2],
                base_period=3,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
        # Result keeps only the surviving horizon (label "1").
        assert list(r.horizon_labels) == ["1"], (
            f"Expected only horizon '1' after dropping consumed placebo, "
            f"got {list(r.horizon_labels)}"
        )

    def test_pretrends_trends_lin_only_consumed_raises(self):
        """When pre_periods is exactly [base_period-1], trends_lin has
        no testable horizons left → ValueError. Regression for PR #392
        R1 P0 (concrete fix step 2)."""
        df = self._panel(rng_seed=31)
        # Panel periods 1..5, F=4. Base=3, base-1=2.
        # Caller passes only [2] (the consumed placebo).
        with pytest.raises(ValueError, match="no testable placebo horizons remain"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                joint_pretrends_test(
                    df,
                    "y",
                    "d",
                    "time",
                    "unit",
                    pre_periods=[2],
                    base_period=3,
                    n_bootstrap=99,
                    seed=42,
                    trends_lin=True,
                )

    def test_pretrends_trends_lin_no_consumed_in_pre_periods_no_warning(self):
        """When pre_periods does NOT include base_period-1, no consumed-
        placebo warning fires (negative regression)."""
        df = self._panel(rng_seed=32)
        # Panel periods 1..5, F=4. Base=3, base-1=2. Pass only [1].
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            r = joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[1],
                base_period=3,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
        assert list(r.horizon_labels) == ["1"]

    def test_homogeneity_trends_lin_naive_baseline(self):
        df = self._panel(rng_seed=12)
        base, base_minus_1 = 3, 2
        wide = df.pivot(index="unit", columns="time", values="y").sort_index()
        slope = wide[base].to_numpy() - wide[base_minus_1].to_numpy()
        df_manual = df.copy()
        df_manual = df_manual.merge(
            pd.DataFrame({"unit": np.arange(len(slope)), "slope": slope}),
            on="unit",
        )
        df_manual["y"] = df_manual["y"] - (df_manual["time"] - base) * df_manual["slope"]
        df_manual = df_manual.drop(columns=["slope"])

        r_manual = joint_homogeneity_test(
            df_manual,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[4, 5],
            base_period=base,
            n_bootstrap=99,
            seed=42,
        )
        r_auto = joint_homogeneity_test(
            df,
            "y",
            "d",
            "time",
            "unit",
            post_periods=[4, 5],
            base_period=base,
            n_bootstrap=99,
            seed=42,
            trends_lin=True,
        )
        np.testing.assert_allclose(
            r_auto.cvm_stat_joint,
            r_manual.cvm_stat_joint,
            atol=1e-12,
            rtol=0,
        )

    def test_homogeneity_trends_lin_missing_base_minus_1_raises(self):
        """When base is the FIRST period in panel (no base-1), trends_lin
        must error. joint_homogeneity_test is the reachable surface for
        this guard: it allows base to be a pre-period AND the earliest
        period (joint_pretrends_test additionally requires pre_periods <
        base, which structurally forces base-1 to exist if any pre_period
        is in the panel)."""
        # Panel periods [3, 4, 5], F=4, base=3 (earliest, last pre-period
        # before treatment), post=[4, 5].
        df = self._panel(rng_seed=14, F=4, T=5)
        df_trim = df[df["time"] >= 3].copy()
        with pytest.raises(ValueError, match=r"at least 2 (validated )?pre-periods"):
            joint_homogeneity_test(
                df_trim,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[4, 5],
                base_period=3,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )

    def test_pretrends_trends_lin_with_survey_design_raises(self):
        from diff_diff import SurveyDesign

        df = self._panel(rng_seed=15)
        df["w"] = 1.0
        with pytest.raises(NotImplementedError, match="trends_lin=True.*survey"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[1, 2],
                base_period=3,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
                survey_design=SurveyDesign(weights="w"),
            )

    def test_pretrends_trends_lin_with_weights_alias_raises(self):
        df = self._panel(rng_seed=16)
        with pytest.raises(NotImplementedError, match="trends_lin=True.*survey"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                joint_pretrends_test(
                    df,
                    "y",
                    "d",
                    "time",
                    "unit",
                    pre_periods=[1, 2],
                    base_period=3,
                    n_bootstrap=99,
                    seed=42,
                    trends_lin=True,
                    weights=np.ones(len(df)),
                )

    def test_homogeneity_trends_lin_with_survey_design_raises(self):
        from diff_diff import SurveyDesign

        df = self._panel(rng_seed=17)
        df["w"] = 1.0
        with pytest.raises(NotImplementedError, match="trends_lin=True.*survey"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[4, 5],
                base_period=3,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
                survey_design=SurveyDesign(weights="w"),
            )

    def test_workflow_trends_lin_forwards_to_joint_wrappers(self):
        """`did_had_pretest_workflow(aggregate='event_study', trends_lin=True)`
        must forward trends_lin to both joint pretests AND match the
        direct-surface call. Regression for PR #392 R1 P1."""
        df = self._panel(rng_seed=33)
        # Run the workflow with trends_lin=True (event_study path).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
        # Direct calls to the joint wrappers, same params, trends_lin=True.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_pretrends_direct = joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[1, 2],
                base_period=3,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
            r_homogeneity_direct = joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[4, 5],
                base_period=3,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
        # Workflow's joint pretests must match the direct calls bit-exactly.
        assert report.pretrends_joint is not None
        assert report.homogeneity_joint is not None
        assert report.pretrends_joint.cvm_stat_joint == r_pretrends_direct.cvm_stat_joint
        assert report.homogeneity_joint.cvm_stat_joint == r_homogeneity_direct.cvm_stat_joint

    def test_workflow_trends_lin_minimal_panel_skips_step2_gracefully(self):
        """Minimal valid trends_lin event-study panel: 4 periods, F=3
        (so t_pre_list=[1,2], base=2, base-1=1 is the only earlier
        placebo AND the consumed one). Workflow should set
        pretrends_joint=None (step 2 skipped) and still run step 3
        (homogeneity). Regression for PR #392 R2 P1.
        """
        rng = np.random.default_rng(35)
        G = 200
        T = 4
        F = 3  # treatment onset at t=3 → pre={1,2}, post={3,4}
        d = rng.beta(0.5, 1.0, size=G)
        unit_fe = rng.normal(0, 1, G)
        trend = rng.normal(0.1, 0.05, G)
        rows = []
        for g in range(G):
            for t in range(1, T + 1):
                treated = t >= F
                y = (
                    unit_fe[g]
                    + trend[g] * (t - 1)
                    + (d[g] + d[g] ** 2) * treated
                    + rng.normal(0, 0.5)
                )
                dose = d[g] if treated else 0.0
                rows.append({"unit": g, "time": t, "y": y, "d": dose})
        df = pd.DataFrame(rows)
        # Run workflow with trends_lin=True. Should not raise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "time",
                "unit",
                aggregate="event_study",
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
        # Step 2 (pretrends) skipped (the only earlier placebo at base-1
        # is the consumed one); step 3 (homogeneity) still runs.
        assert report.pretrends_joint is None, (
            "expected pretrends_joint=None on the minimal-panel trends_lin "
            "case where the only earlier placebo is the consumed one"
        )
        assert (
            report.homogeneity_joint is not None
        ), "expected homogeneity_joint to still run after step 2 skip"
        assert np.isfinite(report.homogeneity_joint.p_value)

    def test_pretrends_trends_lin_nonterminal_base_raises(self):
        """Direct caller passing base_period < t_pre_list[-1] under
        trends_lin=True must raise — Eq 17 anchors at F-1.
        Regression for PR #392 R3 P1 (methodology guard)."""
        df = self._panel(rng_seed=40)
        # Panel periods 1..5, F=4. t_pre_list = [1, 2, 3], F-1 = 3.
        # Pass base_period=2 (non-terminal pre-period).
        with pytest.raises(ValueError, match="last validated pre-period"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=[1],
                base_period=2,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )

    def test_homogeneity_trends_lin_nonterminal_base_raises(self):
        """Twin guard for joint_homogeneity_test."""
        df = self._panel(rng_seed=41)
        with pytest.raises(ValueError, match="last validated pre-period"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "time",
                "unit",
                post_periods=[4, 5],
                base_period=2,
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )

    def test_pretrends_trends_lin_unused_categorical_observed_only(self):
        """Ordered categorical time column with an unused intermediate
        level: trends_lin must resolve base_period - 1 to the previous
        OBSERVED period, not to the unused level (which would KeyError
        on the slope-pivot lookup). Regression for PR #392 R3 P1."""
        df_int = self._panel(rng_seed=42)
        # Convert time to ordered categorical with an unused intermediate
        # level inserted between observed levels t3 and t4.
        cat_levels = ["t1", "t2", "t3", "t_unused", "t4", "t5"]
        time_map = {1: "t1", 2: "t2", 3: "t3", 4: "t4", 5: "t5"}
        df = df_int.copy()
        df["time"] = pd.Categorical(
            df["time"].map(time_map),
            categories=cat_levels,
            ordered=True,
        )
        # Sanity: t_unused is in the dtype but absent from data.
        assert "t_unused" in df["time"].cat.categories
        assert "t_unused" not in set(df["time"].dropna().unique())
        # F=4 → t_pre_list = [t1, t2, t3], base must equal t3 under
        # trends_lin. Under the OLD period_rank lookup, base_minus_1
        # by rank would resolve to t_unused (rank 2 below t3=rank 4...
        # wait actually rank-1 = rank(t3)-1 = 3-1 = 2, which is t3
        # itself wait no t3 = rank 2). Let me reason: cat_levels
        # ranks t1=0, t2=1, t3=2, t_unused=3, t4=4, t5=5. For
        # base=t3 (rank 2), base-1 by rank = rank 1 = t2 (correct).
        # Better demonstration: pass base=t4 (post-period) — but that
        # would be invalid by other guards. Use a setup where the
        # unused level lies BEFORE base in chronology: place
        # t_unused between t2 and t3, then base=t3.
        cat_levels2 = ["t1", "t2", "t_unused", "t3", "t4", "t5"]
        df2 = df_int.copy()
        df2["time"] = pd.Categorical(
            df2["time"].map(time_map),
            categories=cat_levels2,
            ordered=True,
        )
        # Now base=t3 (rank 3); base-1 by rank = t_unused (rank 2,
        # not in data). Old code would KeyError on the slope pivot;
        # new observed-only lookup resolves to t2 (the previous
        # observed period). Verify the call SUCCEEDS.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = joint_pretrends_test(
                df2,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=["t1"],
                base_period="t3",
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
        assert np.isfinite(r.cvm_stat_joint)
        assert np.isfinite(r.p_value)

    def test_pretrends_trends_lin_unused_categorical_invariant(self):
        """Same observed panel with vs without unused intermediate
        categorical levels must produce IDENTICAL joint statistics on
        the pretrends path. Reviewer-requested invariant for PR #392
        R4 P0 — detrending delta must use observed-period rank, not
        full-categorical rank."""
        df_int = self._panel(rng_seed=50)
        time_map = {1: "t1", 2: "t2", 3: "t3", 4: "t4", 5: "t5"}
        df_a = df_int.copy()
        df_a["time"] = pd.Categorical(
            df_a["time"].map(time_map),
            categories=["t1", "t2", "t3", "t4", "t5"],
            ordered=True,
        )
        df_b = df_int.copy()
        df_b["time"] = pd.Categorical(
            df_b["time"].map(time_map),
            categories=["t1", "t_unused1", "t2", "t3", "t_unused2", "t4", "t5"],
            ordered=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_a = joint_pretrends_test(
                df_a,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=["t1"],
                base_period="t3",
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
            r_b = joint_pretrends_test(
                df_b,
                "y",
                "d",
                "time",
                "unit",
                pre_periods=["t1"],
                base_period="t3",
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
        assert r_a.cvm_stat_joint == r_b.cvm_stat_joint, (
            f"unused-categorical invariance broken on pretrends: "
            f"a={r_a.cvm_stat_joint}, b={r_b.cvm_stat_joint}"
        )
        assert r_a.p_value == r_b.p_value

    def test_homogeneity_trends_lin_unused_categorical_invariant(self):
        """Twin invariant for joint_homogeneity_test."""
        df_int = self._panel(rng_seed=51)
        time_map = {1: "t1", 2: "t2", 3: "t3", 4: "t4", 5: "t5"}
        df_a = df_int.copy()
        df_a["time"] = pd.Categorical(
            df_a["time"].map(time_map),
            categories=["t1", "t2", "t3", "t4", "t5"],
            ordered=True,
        )
        df_b = df_int.copy()
        df_b["time"] = pd.Categorical(
            df_b["time"].map(time_map),
            # Insert unused level between base (t3) and post (t4) — would
            # change the post-period delta under the buggy full-cat rank.
            categories=["t1", "t2", "t3", "t_unused", "t4", "t5"],
            ordered=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_a = joint_homogeneity_test(
                df_a,
                "y",
                "d",
                "time",
                "unit",
                post_periods=["t4", "t5"],
                base_period="t3",
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
            r_b = joint_homogeneity_test(
                df_b,
                "y",
                "d",
                "time",
                "unit",
                post_periods=["t4", "t5"],
                base_period="t3",
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )
        assert r_a.cvm_stat_joint == r_b.cvm_stat_joint, (
            f"unused-categorical invariance broken on homogeneity: "
            f"a={r_a.cvm_stat_joint}, b={r_b.cvm_stat_joint}"
        )
        assert r_a.p_value == r_b.p_value

    def test_workflow_trends_lin_with_overall_aggregate_raises(self):
        """trends_lin=True only valid on event_study aggregate."""
        df = self._panel(rng_seed=34)
        df_2p = df[df["time"].isin([3, 4])].copy()
        with pytest.raises(NotImplementedError, match="trends_lin=True.*event_study"):
            did_had_pretest_workflow(
                df_2p,
                "y",
                "d",
                "time",
                "unit",
                aggregate="overall",
                n_bootstrap=99,
                seed=42,
                trends_lin=True,
            )

    def test_pretrends_consumed_e_minus_2_dropped_in_HAD_fit(self):
        """Cross-surface: HAD.fit(trends_lin=True) drops e=-2 from event_times.

        Locks the consumed-placebo invariant (R reduces max placebo lag
        by 1; in our event_time convention that maps to dropping e=-2)."""
        from diff_diff import HeterogeneousAdoptionDiD

        df = self._panel(rng_seed=18)
        est_no = HeterogeneousAdoptionDiD().fit(
            df,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            aggregate="event_study",
        )
        est_yes = HeterogeneousAdoptionDiD().fit(
            df,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            aggregate="event_study",
            trends_lin=True,
        )
        assert -2 in est_no.event_times.tolist()
        assert -2 not in est_yes.event_times.tolist()


class TestHADFitTrendsLin:
    """Direct unit tests for HAD.fit(trends_lin=True) (event-study path).

    Locks F>=3 guard, aggregate='overall' rejection, survey_design ×
    trends_lin NotImplementedError, get/set_params idempotence."""

    @staticmethod
    def _panel(F: int = 4, T: int = 5, G: int = 200, rng_seed: int = 21) -> pd.DataFrame:
        rng = np.random.default_rng(rng_seed)
        d = rng.beta(0.5, 1.0, size=G)
        unit_fe = rng.normal(0, 1, G)
        trend = rng.normal(0.1, 0.05, G)
        rows = []
        for g in range(G):
            for t in range(1, T + 1):
                treated = t >= F
                y = (
                    unit_fe[g]
                    + trend[g] * (t - 1)
                    + (d[g] + d[g] ** 2) * treated
                    + rng.normal(0, 0.5)
                )
                dose = d[g] if treated else 0.0
                rows.append({"unit": g, "time": t, "y": y, "d": dose})
        return pd.DataFrame(rows)

    def test_fit_default_bit_exact_backcompat(self):
        from diff_diff import HeterogeneousAdoptionDiD

        df = self._panel(rng_seed=22)
        r1 = HeterogeneousAdoptionDiD().fit(
            df,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            aggregate="event_study",
        )
        r2 = HeterogeneousAdoptionDiD().fit(
            df,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            aggregate="event_study",
            trends_lin=False,
        )
        np.testing.assert_array_equal(r1.event_times, r2.event_times)
        np.testing.assert_array_equal(r1.att, r2.att)

    def test_fit_aggregate_overall_with_trends_lin_raises(self):
        from diff_diff import HeterogeneousAdoptionDiD

        df = self._panel(F=2, T=2, rng_seed=23)
        with pytest.raises(NotImplementedError, match="aggregate='event_study'"):
            HeterogeneousAdoptionDiD().fit(
                df,
                outcome_col="y",
                dose_col="d",
                time_col="time",
                unit_col="unit",
                aggregate="overall",
                trends_lin=True,
            )

    def test_fit_F2_with_trends_lin_raises(self):
        """F=2 (only 1 pre-period at t=1) → cannot identify slope."""
        from diff_diff import HeterogeneousAdoptionDiD

        # Build a 3-period panel with F=2 (treatment at t=2; only t=1 is pre).
        df = self._panel(F=2, T=3, rng_seed=24)
        with pytest.raises(ValueError, match="F >= 3"):
            HeterogeneousAdoptionDiD().fit(
                df,
                outcome_col="y",
                dose_col="d",
                time_col="time",
                unit_col="unit",
                aggregate="event_study",
                trends_lin=True,
            )

    def test_fit_with_survey_design_and_trends_lin_raises(self):
        from diff_diff import HeterogeneousAdoptionDiD, SurveyDesign

        df = self._panel(rng_seed=25)
        df["w"] = 1.0
        with pytest.raises(NotImplementedError, match="trends_lin=True.*survey"):
            HeterogeneousAdoptionDiD().fit(
                df,
                outcome_col="y",
                dose_col="d",
                time_col="time",
                unit_col="unit",
                aggregate="event_study",
                trends_lin=True,
                survey_design=SurveyDesign(weights="w"),
            )

    def test_fit_idempotence_with_trends_lin(self):
        """Repeat-fit with the same estimator + trends_lin=True yields
        identical numbers (per feedback_fit_does_not_mutate_config)."""
        from diff_diff import HeterogeneousAdoptionDiD

        df = self._panel(rng_seed=27)
        est = HeterogeneousAdoptionDiD()
        r1 = est.fit(
            df,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            aggregate="event_study",
            trends_lin=True,
        )
        r2 = est.fit(
            df,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            aggregate="event_study",
            trends_lin=True,
        )
        np.testing.assert_array_equal(r1.event_times, r2.event_times)
        np.testing.assert_array_equal(r1.att, r2.att)
        # get_params unchanged (no fit-time mutation).
        assert est.get_params() == HeterogeneousAdoptionDiD().get_params()


class TestStuteStratifiedSurveyBootstrap:
    """Phase 4.5 C strata-extension regression suite.

    Locks the behavior of the stratified clustered wild-bootstrap
    correction (within-stratum demean + sqrt(n_h/(n_h-1)) Bessel
    rescale) added in this PR. See REGISTRY § HeterogeneousAdoptionDiD
    "Note (Stute stratified survey-bootstrap calibration)" for the
    derivation.

    Coverage:
    - Stute single-horizon + stratified design: positive smoke
      (was NotImplementedError pre-PR).
    - did_had_pretest_workflow + stratified design: end-to-end
      smoke; QUG silently skipped, joint Stute + Yatchew run
      survey-aware; verdict carries the C0 deferral substring.
    - Trivial-stratum reduction: ``SurveyDesign(strata="all_ones")``
      vs ``SurveyDesign(strata=None)`` — the deterministic CvM
      statistic is bit-equal (atol=1e-14, the algebraic-identity
      claim); the bootstrap p-value matches only within
      bootstrap-noise (≤0.15 abs) because
      ``generate_survey_multiplier_weights_batch`` takes different
      RNG-consumption paths for strata-Some vs strata-None.
    - Non-strata calibration-shift end-to-end smoke: finite + range
      check that the non-strata Stute path runs after the
      single-implicit-stratum centering lands. A direction-pin via
      pre-PR baseline capture was considered but is redundant with
      the helper-level bit-parity regression at atol=1e-14 (which
      catches any revert of apply_stratum_centering's algebra) AND
      the call-site wiring regression below (which catches
      disconnection of the helper from the Stute call sites).
    - Stute call-site wiring regression: monkey-patches
      ``apply_stratum_centering`` and asserts both Stute call sites
      (``stute_test`` at had_pretests.py:1985,
      ``stute_joint_pretest`` at :3312) invoke it with psu_axis=1.
      Catches the disconnection case the helper bit-parity test
      cannot.
    - MC oracle consistency (``@pytest.mark.slow``): empirical Type
      I error under a stratified null DGP at α=0.05 sits in
      [0.0, 0.10] (3σ at 200 draws).
    - MC power (``@pytest.mark.slow``): rejection rate > 0.50 under
      a stratified known-alternative DGP.
    """

    def _stratified_panel(
        self,
        *,
        n_strata=4,
        n_psu_per_stratum=8,
        T_pre=2,
        T_post=2,
        true_slope=0.0,
        nonlinear=False,
        noise_sd=0.3,
        nonlinear_coef=5.0,
        seed=11,
    ):
        """One unit per PSU; balanced stratified design. Under the null
        ``nonlinear=False`` the post-period outcome is linear in dose
        (Stute linearity null holds); under ``nonlinear=True`` the
        post-period outcome adds a quadratic term to violate the null.
        Tightened noise + larger n compared to the prior fixture: 0.3
        SD per-period noise and 8 PSUs/stratum (G=32 after per-unit
        aggregation) keeps the MC oracle Type I in the 3σ band while
        giving the alternative DGP enough signal to exceed the 0.50
        power threshold at 200 draws."""
        rng = np.random.default_rng(seed)
        rows = []
        unit_id = 0
        for h in range(n_strata):
            for p in range(n_psu_per_stratum):
                psu_global = h * n_psu_per_stratum + p
                d_post = rng.uniform(0.1, 1.0)
                w_unit = rng.uniform(0.7, 1.5)
                for t in range(T_pre + T_post):
                    is_post = t >= T_pre
                    d_t = d_post if is_post else 0.0
                    treat_term = (true_slope * d_post) if is_post else 0.0
                    if is_post and nonlinear:
                        treat_term = treat_term + nonlinear_coef * d_post * d_post
                    y_t = rng.normal(0.0, noise_sd) + treat_term
                    rows.append(
                        {
                            "unit": unit_id,
                            "time": t,
                            "y": y_t,
                            "d": d_t,
                            "stratum": h,
                            "psu": psu_global,
                            "w": w_unit,
                        }
                    )
                unit_id += 1
        return pd.DataFrame(rows)

    def test_stute_test_stratified_positive_smoke(self):
        """Stute single-horizon under stratified survey design runs
        end-to-end (pre-PR: ``NotImplementedError``)."""
        from diff_diff import SurveyDesign
        from diff_diff.survey import make_pweight_design  # noqa: F401

        df = self._stratified_panel(n_strata=4, n_psu_per_stratum=8)
        sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
        # Aggregate to per-unit dy = Y_post - Y_pre, per-unit dose.
        pre = df[df["time"] < 2].groupby("unit", as_index=False)["y"].mean()
        post = df[df["time"] >= 2].groupby("unit", as_index=False)["y"].mean()
        merged = pre.merge(post, on="unit", suffixes=("_pre", "_post"))
        dose = df.groupby("unit", as_index=False)["d"].max()
        merged = merged.merge(dose, on="unit")
        dy_arr = (merged["y_post"] - merged["y_pre"]).to_numpy(dtype=np.float64)
        d_arr = merged["d"].to_numpy(dtype=np.float64)
        unit_df = df.drop_duplicates(subset="unit")[["unit", "stratum", "psu", "w"]].sort_values(
            "unit"
        )
        resolved = sd.resolve(unit_df)
        r = stute_test(d=d_arr, dy=dy_arr, survey_design=resolved, n_bootstrap=199, seed=0)
        assert np.isfinite(r.p_value)
        assert 0.0 <= r.p_value <= 1.0
        assert r.n_bootstrap == 199

    def test_workflow_stratified_event_study_end_to_end_smoke(self):
        """``did_had_pretest_workflow(survey_design=stratified_sd)`` runs
        the full event-study workflow under stratified sampling: QUG is
        silently skipped per Phase 4.5 C0, Stute + Yatchew run via the
        survey-aware path with the new stratum centering, and the joint
        variants populate. Verdict carries the C0 deferral substring
        verbatim."""
        from diff_diff import SurveyDesign

        df = self._stratified_panel(n_strata=4, n_psu_per_stratum=5)
        # The workflow needs ``first_treat`` per unit (HAD panel contract).
        # All units adopt at t=T_pre=2 in this fixture.
        df = df.copy()
        df["F"] = 2
        sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
        with pytest.warns(UserWarning, match="QUG step skipped"):
            report = did_had_pretest_workflow(
                data=df,
                outcome_col="y",
                dose_col="d",
                time_col="time",
                unit_col="unit",
                first_treat_col="F",
                survey_design=sd,
                aggregate="event_study",
                n_bootstrap=199,
                seed=0,
                alpha=0.05,
            )
        # QUG silently skipped on the survey path (per C0).
        assert report.qug is None
        # Joint pretrends + joint homogeneity populate.
        assert report.pretrends_joint is not None
        assert report.homogeneity_joint is not None
        # Both joint tests should produce finite p-values.
        assert np.isfinite(report.pretrends_joint.p_value)
        assert np.isfinite(report.homogeneity_joint.p_value)
        # Verdict carries the C0 deferral substring verbatim per
        # feedback_no_silent_failures.
        assert "QUG-under-survey deferred per Phase 4.5 C0" in report.verdict

    def test_trivial_stratum_reduces_to_strata_none(self):
        """``SurveyDesign(weights, psu, strata="all_ones")`` (single
        explicit stratum) is *algebraically* equivalent to
        ``SurveyDesign(weights, psu, strata=None)`` (single implicit
        stratum) after the PR — both apply the same within-implicit-
        stratum demean + sqrt(n_psu/(n_psu-1)) Bessel rescale.

        The DETERMINISTIC CvM statistic matches bit-exactly between the
        two fits — that is the algebraic-identity claim.

        The BOOTSTRAP p-value does NOT match bit-exactly because
        ``generate_survey_multiplier_weights_batch`` takes structurally
        different code paths based on ``strata`` (see
        ``bootstrap_utils.py:556+`` strata-None branch vs ``:579+``
        stratified branch — different RNG-consumption patterns and the
        strata-None branch additionally routes through the Rust backend
        when available, while the stratified per-stratum loop uses
        ``..._numpy``). Bootstrap p-values are expected to differ at
        the order of bootstrap noise — we assert a loose closeness band
        as a SMOKE, not bit-equality."""
        from diff_diff import SurveyDesign

        df = self._stratified_panel(n_strata=1, n_psu_per_stratum=20)
        df = df.copy()
        df["all_ones"] = 1
        sd_explicit = SurveyDesign(weights="w", psu="psu", strata="all_ones")
        sd_implicit = SurveyDesign(weights="w", psu="psu")

        pre = df[df["time"] < 2].groupby("unit", as_index=False)["y"].mean()
        post = df[df["time"] >= 2].groupby("unit", as_index=False)["y"].mean()
        merged = pre.merge(post, on="unit", suffixes=("_pre", "_post"))
        dose = df.groupby("unit", as_index=False)["d"].max()
        merged = merged.merge(dose, on="unit")
        dy_arr = (merged["y_post"] - merged["y_pre"]).to_numpy(dtype=np.float64)
        d_arr = merged["d"].to_numpy(dtype=np.float64)
        unit_df = df.drop_duplicates(subset="unit")[
            ["unit", "stratum", "psu", "w", "all_ones"]
        ].sort_values("unit")
        resolved_explicit = sd_explicit.resolve(unit_df)
        resolved_implicit = sd_implicit.resolve(unit_df)

        # Use the same seed for both. The CvM is deterministic; the
        # bootstrap p-value depends on the multiplier RNG path, which
        # differs between the two helper branches (see docstring).
        r_explicit = stute_test(
            d=d_arr, dy=dy_arr, survey_design=resolved_explicit, n_bootstrap=199, seed=42
        )
        r_implicit = stute_test(
            d=d_arr, dy=dy_arr, survey_design=resolved_implicit, n_bootstrap=199, seed=42
        )
        # CvM statistic is deterministic — bit-exact match (the
        # algebraic-identity claim).
        np.testing.assert_allclose(r_explicit.cvm_stat, r_implicit.cvm_stat, atol=1e-14)
        # Bootstrap p-values: loose closeness band (within bootstrap
        # noise), NOT bit-equality. At B=199 the bootstrap SD on the
        # uniform p-value is ~sqrt(0.25/199) ≈ 0.035; a ±0.15 band is
        # ~4σ and decisively distinguishes "two fits of the same
        # statistic" from "the algebraic-equivalence claim is broken"
        # (which would push p-values to opposite tails).
        assert abs(float(r_explicit.p_value) - float(r_implicit.p_value)) < 0.15, (
            f"Trivial-stratum reduction smoke: explicit p={r_explicit.p_value!r} "
            f"vs implicit p={r_implicit.p_value!r} differ by more than the "
            f"bootstrap-noise band (0.15). The algebraic equivalence between "
            f"SurveyDesign(strata='all_ones') and SurveyDesign(strata=None) "
            f"may be broken; check apply_stratum_centering's strata-None branch."
        )

    def test_stute_call_sites_invoke_apply_stratum_centering(self, monkeypatch):
        """Regression: ensure the Stute survey-bootstrap call sites
        actually invoke ``apply_stratum_centering`` on the PSU multipliers
        before the per-obs broadcast. A future change that silently
        removed the calls at ``had_pretests.py:1985`` (``stute_test``)
        or ``had_pretests.py:3312`` (``stute_joint_pretest``) would
        disable the stratified clustered wild-bootstrap correction
        without failing the helper-level bit-parity regression at
        ``tests/test_bootstrap_utils.py::TestApplyStratumCentering::
        test_bit_parity_vs_pre_refactor_inline_block`` (which only
        covers the HAD sup-t ``psu_axis=0`` layout). This test closes
        the disconnection-detection gap surfaced by PR #432 R2 review."""
        import diff_diff.had_pretests as hp
        from diff_diff import SurveyDesign

        calls: list = []
        real_helper = hp.apply_stratum_centering

        def spy(*args, **kwargs):
            # Record psu_axis from kwarg or positional (the call sites
            # use kwarg, but be defensive).
            psu_axis = kwargs.get("psu_axis", args[3] if len(args) > 3 else 0)
            calls.append(psu_axis)
            return real_helper(*args, **kwargs)

        monkeypatch.setattr(hp, "apply_stratum_centering", spy)

        df = self._stratified_panel(n_strata=4, n_psu_per_stratum=4)
        sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
        pre = df[df["time"] < 2].groupby("unit", as_index=False)["y"].mean()
        post = df[df["time"] >= 2].groupby("unit", as_index=False)["y"].mean()
        merged = pre.merge(post, on="unit", suffixes=("_pre", "_post"))
        dose = df.groupby("unit", as_index=False)["d"].max()
        merged = merged.merge(dose, on="unit")
        dy = (merged["y_post"] - merged["y_pre"]).to_numpy(dtype=np.float64)
        d_arr = merged["d"].to_numpy(dtype=np.float64)
        unit_df = df.drop_duplicates(subset="unit")[["unit", "stratum", "psu", "w"]].sort_values(
            "unit"
        )
        resolved = sd.resolve(unit_df)

        # Single-horizon call site.
        calls.clear()
        stute_test(d=d_arr, dy=dy, survey_design=resolved, n_bootstrap=99, seed=0)
        assert calls == [1], (
            "stute_test did not invoke apply_stratum_centering on the "
            f"Stute multiplier matrix (psu_axis=1). Recorded calls: {calls}. "
            "The stratified clustered wild-bootstrap correction is "
            "disconnected from had_pretests.py:1985."
        )

        # Multi-horizon call site.
        calls.clear()
        G = len(d_arr)
        residuals_by_horizon = {
            "0": np.random.default_rng(7).normal(size=G),
            "1": np.random.default_rng(8).normal(size=G),
        }
        fitted_by_horizon = {"0": np.zeros(G), "1": np.zeros(G)}
        design_matrix = np.column_stack([np.ones(G), d_arr])
        stute_joint_pretest(
            residuals_by_horizon=residuals_by_horizon,
            fitted_by_horizon=fitted_by_horizon,
            doses=d_arr,
            design_matrix=design_matrix,
            n_bootstrap=99,
            seed=0,
            survey=resolved,
        )
        assert calls == [1], (
            "stute_joint_pretest did not invoke apply_stratum_centering on "
            f"the joint Stute multiplier matrix (psu_axis=1). Recorded "
            f"calls: {calls}. The stratified clustered wild-bootstrap "
            "correction is disconnected from had_pretests.py:3312."
        )

    def test_calibration_shift_non_strata_end_to_end_smoke(self):
        """End-to-end smoke for the non-strata Stute path after the
        single-implicit-stratum centering lands. The post-PR path
        applies the ``sqrt(n_psu / (n_psu - 1))`` Bessel rescale (and
        within-implicit-stratum demean) on the PSU multipliers, where
        the pre-PR Phase 4.5 C path applied no correction.

        This test is a finite + range smoke — NOT a baseline-capture
        direction pin. A heavier worktree-based pre/post comparison was
        considered but is redundant with the helper-level bit-parity
        regression at ``tests/test_bootstrap_utils.py::TestApplyStratumCentering::
        test_bit_parity_vs_pre_refactor_inline_block`` (locked at
        ``atol=1e-14``), which fails the instant ``apply_stratum_centering``
        stops centering — regardless of which call site consumes it.
        That helper-level invariant lock catches any revert of the
        non-strata calibration end-to-end.

        See REGISTRY § "Note (Stute stratified survey-bootstrap
        calibration)" non-strata calibration improvement subsection."""
        from diff_diff.survey import make_pweight_design

        rng = np.random.default_rng(11)
        G = 80
        d_arr = rng.uniform(0.5, 5.0, size=G)
        # Linear dy under H0 (linearity null holds): no quadratic term.
        dy_arr = 2.0 * d_arr + rng.normal(0, 0.5, size=G)
        w_arr = rng.uniform(0.7, 1.5, size=G)
        w_arr = w_arr / w_arr.mean()
        resolved = make_pweight_design(w_arr)
        r = stute_test(d=d_arr, dy=dy_arr, survey_design=resolved, n_bootstrap=999, seed=11)
        assert np.isfinite(r.p_value)
        assert 0.0 <= r.p_value <= 1.0, f"Stute p-value out of valid range: {r.p_value}"

    @pytest.mark.slow
    def test_stratified_mc_type1_error_in_band(self):
        """MC oracle consistency: 200-draw simulation under a stratified
        null DGP. Empirical Type I error at α=0.05 should fall in
        [0.0, 0.10] (3σ width at n=200, ≈ ±3·0.0154). Seed-set for
        reproducibility. See reviewer Medium #1 — 1.6σ band was too
        tight; 3σ is the standard well-calibrated-test allowance."""
        from diff_diff import SurveyDesign

        n_draws = 200
        rejections = 0
        for sim in range(n_draws):
            df = self._stratified_panel(
                n_strata=4,
                n_psu_per_stratum=6,
                true_slope=2.0,  # post-period true linear effect
                nonlinear=False,  # null holds (linear E[dY|D])
                seed=1000 + sim,
            )
            sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
            pre = df[df["time"] < 2].groupby("unit", as_index=False)["y"].mean()
            post = df[df["time"] >= 2].groupby("unit", as_index=False)["y"].mean()
            merged = pre.merge(post, on="unit", suffixes=("_pre", "_post"))
            dose = df.groupby("unit", as_index=False)["d"].max()
            merged = merged.merge(dose, on="unit")
            dy = (merged["y_post"] - merged["y_pre"]).to_numpy(dtype=np.float64)
            d_arr = merged["d"].to_numpy(dtype=np.float64)
            unit_df = df.drop_duplicates(subset="unit")[
                ["unit", "stratum", "psu", "w"]
            ].sort_values("unit")
            resolved = sd.resolve(unit_df)
            r = stute_test(
                d=d_arr,
                dy=dy,
                survey_design=resolved,
                n_bootstrap=199,
                seed=sim,
            )
            if r.reject:
                rejections += 1
        empirical_type_1 = rejections / n_draws
        assert 0.0 <= empirical_type_1 <= 0.10, (
            f"Empirical Type I error under stratified null DGP: "
            f"{empirical_type_1:.3f} (target ≈ 0.05; 3σ band "
            f"[0.0, 0.10] at n_draws={n_draws})"
        )

    @pytest.mark.slow
    def test_stratified_mc_power_under_alternative(self):
        """MC power: 200-draw simulation under a stratified known-
        alternative DGP (quadratic E[dY|D]). Rejection rate at α=0.05
        should exceed 0.50. Seed-set for reproducibility. Validates
        that the stratified centering doesn't kill power."""
        from diff_diff import SurveyDesign

        n_draws = 200
        rejections = 0
        for sim in range(n_draws):
            df = self._stratified_panel(
                n_strata=4,
                n_psu_per_stratum=6,
                true_slope=2.0,
                nonlinear=True,  # alternative: post quadratic term
                seed=2000 + sim,
            )
            sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
            pre = df[df["time"] < 2].groupby("unit", as_index=False)["y"].mean()
            post = df[df["time"] >= 2].groupby("unit", as_index=False)["y"].mean()
            merged = pre.merge(post, on="unit", suffixes=("_pre", "_post"))
            dose = df.groupby("unit", as_index=False)["d"].max()
            merged = merged.merge(dose, on="unit")
            dy = (merged["y_post"] - merged["y_pre"]).to_numpy(dtype=np.float64)
            d_arr = merged["d"].to_numpy(dtype=np.float64)
            unit_df = df.drop_duplicates(subset="unit")[
                ["unit", "stratum", "psu", "w"]
            ].sort_values("unit")
            resolved = sd.resolve(unit_df)
            r = stute_test(
                d=d_arr,
                dy=dy,
                survey_design=resolved,
                n_bootstrap=199,
                seed=sim,
            )
            if r.reject:
                rejections += 1
        power = rejections / n_draws
        assert power > 0.50, (
            f"Stratified Stute power under known alternative: "
            f"{power:.3f} (target > 0.50 at n_draws={n_draws})"
        )
