"""
Methodology verification tests for Honest DiD (Rambachan & Roth, 2023).

These tests verify the corrected implementation against the paper's
equations, known analytical cases, and expected mathematical properties.
"""


import json
import os
import warnings

import numpy as np
import pytest

from diff_diff.honest_did import (
    HonestDiD,
    _compute_flci,
    _compute_optimal_flci,
    _compute_pre_first_differences,
    _construct_A_sd,
    _construct_constraints_rm_component,
    _construct_constraints_sd,
    _cv_alpha,
    _solve_bounds_lp,
    _solve_rm_bounds_union,
)

# =============================================================================
# TestDeltaSDConstraintMatrix
# =============================================================================


class TestDeltaSDConstraintMatrix:
    """Verify DeltaSD constraint matrix accounts for delta_0 = 0 boundary."""

    def test_row_count(self):
        """T+Tbar-1 rows, not T+Tbar-2 (accounts for delta_0 = 0)."""
        for T, Tbar in [(2, 2), (3, 3), (4, 2), (1, 1), (3, 1), (1, 3)]:
            A = _construct_A_sd(T, Tbar)
            expected_rows = T + Tbar - 1
            assert A.shape == (expected_rows, T + Tbar), (
                f"T={T}, Tbar={Tbar}: expected {expected_rows} rows, got {A.shape[0]}"
            )

    def test_2pre_2post_hand_computed(self):
        """Hand-computed matrix for 2 pre + 2 post periods."""
        # delta = [d_{-2}, d_{-1}, d_1, d_2]
        A = _construct_A_sd(2, 2)
        expected = np.array([
            [1, -2, 0, 0],   # t=-1: d_{-2} - 2*d_{-1} + 0
            [0,  1, 1, 0],   # t= 0: d_{-1} + d_1 (bridge)
            [0,  0, -2, 1],  # t= 1: 0 - 2*d_1 + d_2
        ])
        np.testing.assert_array_equal(A, expected)

    def test_bridge_constraint_present(self):
        """The bridge constraint delta_{-1} + delta_1 is always present."""
        for T, Tbar in [(1, 1), (2, 2), (4, 3)]:
            A = _construct_A_sd(T, Tbar)
            # Find the bridge row: non-zero only at positions T-1 and T
            bridge_found = False
            for row in A:
                if row[T - 1] != 0 and row[T] != 0:
                    # This should be [0, ..., 1, 1, ..., 0]
                    assert row[T - 1] == 1, f"Bridge row should have 1 at delta_{{-1}}"
                    assert row[T] == 1, f"Bridge row should have 1 at delta_1"
                    bridge_found = True
            assert bridge_found, f"Bridge constraint not found for T={T}, Tbar={Tbar}"

    def test_constraints_span_all_periods(self):
        """Constraints involve both pre and post periods (not pre-only)."""
        A = _construct_A_sd(3, 3)
        # Some rows should have non-zero entries in post-period columns
        post_cols = A[:, 3:]  # columns for delta_1, delta_2, delta_3
        assert np.any(post_cols != 0), "No constraints involve post-period deltas"


# =============================================================================
# TestIdentifiedSetLP
# =============================================================================


class TestIdentifiedSetLP:
    """Verify identified set LP pins delta_pre = beta_pre."""

    def test_m0_linear_extrapolation(self):
        """M=0 with linear pre-trends gives finite point-identified bounds."""
        # Pre-trends: linear decline with slope -0.1
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        l_vec = np.array([1.0])

        A, b = _construct_constraints_sd(3, 1, M=0.0)
        lb, ub = _solve_bounds_lp(beta_pre, beta_post, l_vec, A, b, 3)

        # Linear extrapolation: slope = -0.1, so delta_1 = 0 - 0.1 = -0.1
        # theta = beta_post - delta_post = 2.0 - (-0.1) = 2.1
        assert np.isfinite(lb), "M=0 should give finite lower bound"
        assert np.isfinite(ub), "M=0 should give finite upper bound"
        np.testing.assert_allclose(lb, 2.1, atol=1e-6)
        np.testing.assert_allclose(ub, 2.1, atol=1e-6)

    def test_bounds_widen_with_m(self):
        """Identified set widens monotonically with M."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        l_vec = np.array([1.0])

        prev_width = 0
        for M in [0.0, 0.1, 0.5, 1.0]:
            A, b = _construct_constraints_sd(3, 1, M=M)
            lb, ub = _solve_bounds_lp(beta_pre, beta_post, l_vec, A, b, 3)
            width = ub - lb
            assert width >= prev_width - 1e-10, (
                f"Width should increase: M={M}, width={width}, prev={prev_width}"
            )
            prev_width = width

    def test_three_period_analytical(self):
        """Paper Section 2.3: three-period example (T=1, Tbar=1)."""
        # delta = [d_{-1}, d_1], with delta_0 = 0
        # DeltaSD(M): |d_1 + d_{-1}| <= M (bridge constraint only)
        # With d_{-1} = beta_{-1} pinned:
        #   d_1 in [-(beta_{-1} + M), -(beta_{-1} - M)] = [-beta_{-1} - M, -beta_{-1} + M]
        # theta = beta_1 - d_1
        #   lb = beta_1 - (-beta_{-1} + M) = beta_1 + beta_{-1} - M
        #   ub = beta_1 - (-beta_{-1} - M) = beta_1 + beta_{-1} + M
        beta_pre = np.array([0.5])
        beta_post = np.array([3.0])

        for M in [0.0, 0.2, 1.0]:
            A, b = _construct_constraints_sd(1, 1, M=M)
            lb, ub = _solve_bounds_lp(beta_pre, beta_post, np.array([1.0]), A, b, 1)
            expected_lb = 3.0 + 0.5 - M
            expected_ub = 3.0 + 0.5 + M
            np.testing.assert_allclose(lb, expected_lb, atol=1e-6,
                                       err_msg=f"M={M}: lb mismatch")
            np.testing.assert_allclose(ub, expected_ub, atol=1e-6,
                                       err_msg=f"M={M}: ub mismatch")


# =============================================================================
# TestDeltaRMFirstDifferences
# =============================================================================


class TestDeltaRMFirstDifferences:
    """Verify DeltaRM constrains first differences, not levels."""

    def test_pre_first_differences_computation(self):
        """Pre-period first differences include delta_0=0 boundary."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        diffs = _compute_pre_first_differences(beta_pre)

        # Interior: |0.2-0.3|=0.1, |0.1-0.2|=0.1
        # Boundary: |0 - 0.1| = 0.1
        np.testing.assert_allclose(diffs, [0.1, 0.1, 0.1], atol=1e-10)

    def test_pre_first_differences_boundary(self):
        """The boundary term |0 - beta_{-1}| is included."""
        beta_pre = np.array([0.0, 0.0, 0.5])
        diffs = _compute_pre_first_differences(beta_pre)

        # Interior: |0-0|=0, |0.5-0|=0.5
        # Boundary: |0 - 0.5| = 0.5
        np.testing.assert_allclose(diffs, [0.0, 0.5, 0.5], atol=1e-10)

    def test_rm_constraints_are_first_differences(self):
        """RM constraint matrix constrains consecutive differences, not levels."""
        A, b = _construct_constraints_rm_component(2, 3, Mbar=1.0, max_pre_first_diff=0.1)

        # 3 post-period first diffs: |d_1|, |d_2-d_1|, |d_3-d_2|
        # Each needs pos/neg constraint = 6 rows total
        assert A.shape[0] == 6
        assert A.shape[1] == 5  # 2 pre + 3 post

        # First pair: d_1 <= 0.1 and -d_1 <= 0.1
        assert A[0, 2] == 1   # d_1
        assert A[1, 2] == -1  # -d_1

        # Second pair: d_2 - d_1 <= 0.1
        assert A[2, 3] == 1 and A[2, 2] == -1   # d_2 - d_1
        assert A[3, 3] == -1 and A[3, 2] == 1   # -(d_2 - d_1)

    def test_mbar0_gives_point_estimate(self):
        """Mbar=0: all post first diffs = 0, theta = l'beta_post."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0, 2.5])
        l_vec = np.array([0.5, 0.5])

        lb, ub = _solve_rm_bounds_union(beta_pre, beta_post, l_vec, 3, Mbar=0.0)

        theta = np.dot(l_vec, beta_post)
        np.testing.assert_allclose(lb, theta, atol=1e-6)
        np.testing.assert_allclose(ub, theta, atol=1e-6)

    def test_rm_bounds_widen_with_mbar(self):
        """Identified set widens monotonically with Mbar."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0, 2.5])
        l_vec = np.array([0.5, 0.5])

        prev_width = 0
        for Mbar in [0.0, 0.5, 1.0, 2.0]:
            lb, ub = _solve_rm_bounds_union(beta_pre, beta_post, l_vec, 3, Mbar)
            width = ub - lb
            assert width >= prev_width - 1e-10, f"Mbar={Mbar}: width decreased"
            prev_width = width


# =============================================================================
# TestOptimalFLCI
# =============================================================================


class TestOptimalFLCI:
    """Verify optimal FLCI properties."""

    def test_cv_alpha_at_zero(self):
        """cv_alpha(0, alpha) = z_{alpha/2} (standard normal quantile)."""
        from scipy.stats import norm
        np.testing.assert_allclose(_cv_alpha(0, 0.05), norm.ppf(0.975), atol=1e-4)
        np.testing.assert_allclose(_cv_alpha(0, 0.01), norm.ppf(0.995), atol=1e-4)

    def test_cv_alpha_monotonic(self):
        """cv_alpha(t) increases with |t| (more bias -> wider CI)."""
        cvs = [_cv_alpha(t, 0.05) for t in [0, 0.5, 1.0, 2.0, 5.0]]
        assert all(cvs[i] <= cvs[i + 1] + 1e-10 for i in range(len(cvs) - 1))

    def test_optimal_flci_is_finite_and_valid(self):
        """Optimal FLCI should produce finite CIs that cover identified set."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        sigma = np.eye(4) * 0.01
        l_vec = np.array([1.0])

        ci_lb_opt, ci_ub_opt = _compute_optimal_flci(
            beta_pre, beta_post, sigma, l_vec, 3, 1, M=0.5, alpha=0.05
        )

        # CI should be finite
        assert np.isfinite(ci_lb_opt) and np.isfinite(ci_ub_opt)
        # CI should cover the identified set
        A, b = _construct_constraints_sd(3, 1, 0.5)
        lb, ub = _solve_bounds_lp(beta_pre, beta_post, l_vec, A, b, 3)
        assert ci_lb_opt <= lb, "CI lower should be <= identified set lower"
        assert ci_ub_opt >= ub, "CI upper should be >= identified set upper"

    def test_m0_short_circuit(self):
        """M=0 takes the bias=0 fast path and never invokes the LP solver.

        ``_compute_worst_case_bias`` returns ``0.0`` immediately when ``M=0``
        (diff_diff/honest_did.py:1650), so ``scipy.optimize.linprog`` is
        never reached. Patching the LP solver and asserting ``call_count
        == 0`` is a direct correctness signal — CI-safe (no wall-clock
        dependency) and faster than the prior timing-based proxy.
        """
        from unittest.mock import patch

        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        sigma = np.eye(4) * 0.01
        l_vec = np.array([1.0])

        with patch("diff_diff.honest_did.optimize.linprog") as mock_linprog:
            ci_lb, ci_ub = _compute_optimal_flci(
                beta_pre, beta_post, sigma, l_vec, 3, 1, M=0.0
            )

        assert mock_linprog.call_count == 0, (
            f"M=0 must skip the LP solver (fast path at "
            f"_compute_worst_case_bias:1650); got "
            f"{mock_linprog.call_count} linprog call(s)."
        )
        # End-to-end correctness: M=0 CI is still well-defined.
        assert np.isfinite(ci_lb) and np.isfinite(ci_ub), (
            f"M=0 CI must be finite; got [{ci_lb}, {ci_ub}]"
        )
        assert ci_lb <= ci_ub, f"M=0 CI must be ordered; got [{ci_lb}, {ci_ub}]"

    def test_smoothness_flci_with_survey_df(self):
        """Survey df should widen the smoothness FLCI (folded t vs folded normal)."""
        beta_pre = np.array([0.1, 0.05])
        beta_post = np.array([2.0])
        sigma = np.eye(3) * 0.01

        # Without df: uses folded normal
        ci_lb_norm, ci_ub_norm = _compute_optimal_flci(
            beta_pre, beta_post, sigma, np.array([1.0]), 2, 1, M=0.5
        )
        # With df=2: uses folded non-central t (wider critical values)
        ci_lb_t, ci_ub_t = _compute_optimal_flci(
            beta_pre, beta_post, sigma, np.array([1.0]), 2, 1, M=0.5, df=2
        )
        width_norm = ci_ub_norm - ci_lb_norm
        width_t = ci_ub_t - ci_lb_t
        assert width_t > width_norm, (
            f"Survey df=2 should widen CI: norm={width_norm:.4f}, t={width_t:.4f}"
        )

    def test_m0_se_includes_pre_period_variance(self):
        """M=0 SE should account for pre-period variance, not just post."""
        # Use off-diagonal covariance to make pre-period SE matter
        sigma = np.array([
            [0.04, 0.02, 0.01],  # pre-1 has high variance
            [0.02, 0.01, 0.005],
            [0.01, 0.005, 0.01],
        ])
        beta_pre = np.array([0.2, 0.1])  # linear pre-trend
        beta_post = np.array([2.0])
        l_vec = np.array([1.0])

        ci_lb, ci_ub = _compute_optimal_flci(
            beta_pre, beta_post, sigma, l_vec, 2, 1, M=0.0
        )
        # CI should be finite and the width should reflect pre-period variance
        assert np.isfinite(ci_lb) and np.isfinite(ci_ub), "M=0 CI should be finite"
        width = ci_ub - ci_lb

        # Compare to post-only SE: sqrt(l'Sigma_post l) = sqrt(0.01) = 0.1
        post_only_width = 2 * 1.96 * np.sqrt(sigma[2, 2])
        assert width > post_only_width, (
            f"M=0 width ({width:.4f}) should exceed post-only ({post_only_width:.4f})"
        )

    def test_optimal_flci_width_increases_with_m_positive(self):
        """Regression for P0: smoothness CI width must increase with M for M > 0."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        sigma = np.eye(4) * 0.01

        # Test monotonicity for M > 0 only. The M=0 path uses a different
        # SE calculation (conservative, includes pre-period variance) which
        # can produce a wider CI than small M > 0 where the optimizer is active.
        widths = []
        for M in [0.1, 0.5, 1.0, 2.0]:
            ci_lb, ci_ub = _compute_optimal_flci(
                beta_pre, beta_post, sigma, np.array([1.0]), 3, 1, M=M
            )
            widths.append(ci_ub - ci_lb)

        for i in range(len(widths) - 1):
            assert widths[i + 1] >= widths[i] - 1e-4, (
                f"CI width must increase with M: M[{i}]={widths[i]:.4f}, "
                f"M[{i+1}]={widths[i+1]:.4f}"
            )

    def test_optimal_flci_bias_nonzero_for_nonzero_m(self):
        """Regression for P0: bias should be nonzero when M > 0."""
        from diff_diff.honest_did import _compute_worst_case_bias

        # T=3: 3 slopes (including boundary), sum(w)=1 for l=[1]
        w = np.array([0.2, 0.3, 0.5])
        l_vec = np.array([1.0])

        bias = _compute_worst_case_bias(w, l_vec, num_pre=3, num_post=1, M=0.5)
        assert bias > 0, f"Bias should be nonzero for M>0, got {bias}"

    def test_three_period_m0_flci_center(self):
        """T=1, Tbar=1, M=0: FLCI centered on beta_1 + beta_{-1}."""
        beta_pre = np.array([0.5])
        beta_post = np.array([3.0])
        sigma = np.eye(2) * 0.01

        ci_lb, ci_ub = _compute_optimal_flci(
            beta_pre, beta_post, sigma, np.array([1.0]), 1, 1, M=0.0
        )
        center = (ci_lb + ci_ub) / 2
        expected_center = 3.0 + 0.5  # beta_1 + beta_{-1}
        np.testing.assert_allclose(center, expected_center, atol=1e-4,
                                   err_msg="M=0 FLCI should be centered on beta_1 + beta_{-1}")

    def test_multi_post_m0_finite(self):
        """Default l_vec with Tbar>1: M=0 gives finite CI."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0, 2.5])
        sigma = np.eye(5) * 0.01
        l_vec = np.array([0.5, 0.5])  # average of 2 post periods

        ci_lb, ci_ub = _compute_optimal_flci(
            beta_pre, beta_post, sigma, l_vec, 3, 2, M=0.0
        )
        assert np.isfinite(ci_lb) and np.isfinite(ci_ub), (
            f"Multi-post M=0 should give finite CI, got [{ci_lb}, {ci_ub}]"
        )

    def test_multi_post_m_positive_finite(self):
        """Default l_vec with Tbar>1: M>0 gives finite CI."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0, 2.5])
        sigma = np.eye(5) * 0.01
        l_vec = np.array([0.5, 0.5])

        ci_lb, ci_ub = _compute_optimal_flci(
            beta_pre, beta_post, sigma, l_vec, 3, 2, M=0.5
        )
        assert np.isfinite(ci_lb) and np.isfinite(ci_ub), (
            f"Multi-post M=0.5 should give finite CI, got [{ci_lb}, {ci_ub}]"
        )

    def test_infeasible_lp_returns_nan(self):
        """Regression for P1: infeasible LP should return NaN, not [-inf, inf]."""
        # Non-linear pre-trends that are inconsistent with M=0 smoothness
        beta_pre = np.array([1.0, 0.0, 1.0])  # quadratic, not linear
        beta_post = np.array([2.0])
        A, b = _construct_constraints_sd(3, 1, M=0.0)

        lb, ub = _solve_bounds_lp(beta_pre, beta_post, np.array([1.0]), A, b, 3)
        # M=0 with non-linear pre-trends: should be infeasible
        assert np.isnan(lb) and np.isnan(ub), (
            f"Infeasible LP should return NaN, got [{lb}, {ub}]"
        )

    def test_infeasible_smoothness_fit_returns_flci_with_empty_idset(self):
        """Fit-level: an empty ESTIMATED identified set still yields a finite FLCI.

        When the observed pre-trend's curvature exceeds M, the identified-set LP
        (which pins delta_pre = beta_pre) is infeasible, so ``lb``/``ub`` are NaN.
        The FLCI does not depend on that LP - it is an affine estimator whose
        worst-case bias is taken over delta in Delta^SD(M) treating beta as random,
        so it is well-defined given (sigma, M). R's ``HonestDiD::createSensitivityResults``
        returns the FLCI in exactly this case, and so do we (previously the fit
        NaN-propagated the whole result, silently yielding no inference).
        """
        from diff_diff.results import MultiPeriodDiDResults, PeriodEffect

        # Non-linear pre-trends: inconsistent with Delta^SD(M=0.0)
        period_effects = {
            1: PeriodEffect(period=1, effect=1.0, se=0.1, t_stat=10.0,
                           p_value=0.0, conf_int=(0.8, 1.2)),
            2: PeriodEffect(period=2, effect=0.0, se=0.1, t_stat=0.0,
                           p_value=1.0, conf_int=(-0.2, 0.2)),
            3: PeriodEffect(period=3, effect=1.0, se=0.1, t_stat=10.0,
                           p_value=0.0, conf_int=(0.8, 1.2)),
            5: PeriodEffect(period=5, effect=2.0, se=0.1, t_stat=20.0,
                           p_value=0.0, conf_int=(1.8, 2.2)),
        }
        results = MultiPeriodDiDResults(
            avg_att=2.0, avg_se=0.1, avg_t_stat=20.0, avg_p_value=0.0,
            avg_conf_int=(1.8, 2.2), n_obs=500, n_treated=250, n_control=250,
            period_effects=period_effects, pre_periods=[1, 2, 3], post_periods=[5],
            vcov=np.eye(4) * 0.01,
            interaction_indices={1: 0, 2: 1, 3: 2, 5: 3},
        )

        honest = HonestDiD(method="smoothness", M=0.0)
        r = honest.fit(results)
        # Estimated identified set is empty -> NaN bounds ...
        assert np.isnan(r.lb) and np.isnan(r.ub), f"Expected NaN id-set bounds, got [{r.lb}, {r.ub}]"
        # ... but the FLCI is finite and matches R createSensitivityResults(M=0):
        # [2.082644, 2.488866] (verified against HonestDiD 0.2.6).
        assert np.isfinite(r.ci_lb) and np.isfinite(r.ci_ub), (
            f"Expected finite FLCI, got [{r.ci_lb}, {r.ci_ub}]"
        )
        assert abs(r.ci_lb - 2.082644) < 1e-3, f"ci_lb={r.ci_lb:.6f} vs R 2.082644"
        assert abs(r.ci_ub - 2.488866) < 1e-3, f"ci_ub={r.ci_ub:.6f} vs R 2.488866"
        # Finite CI excluding 0 -> significant, with a star.
        assert r.is_significant, "CI [2.08, 2.49] excludes 0 -> significant"
        assert r.significance_stars == "*"

    def test_smoothness_df_survey_zero_returns_nan(self):
        """Smoothness with df_survey=0 should return NaN CI."""
        from diff_diff.honest_did import _compute_optimal_flci

        beta_pre = np.array([0.1, 0.05])
        beta_post = np.array([2.0])
        sigma = np.eye(3) * 0.01

        # df=0 → NaN for all M
        ci_lb, ci_ub = _compute_optimal_flci(
            beta_pre, beta_post, sigma, np.array([1.0]), 2, 1, M=0.5, df=0
        )
        assert np.isnan(ci_lb) and np.isnan(ci_ub), "df=0 should give NaN CI"

    def test_smoothness_flci_finite_and_matches_r_across_M_grid(self):
        """Smoothness FLCI is finite AND matches R across the M grid (not just M=0).

        Regression guard for the identified-set NaN-gate bug (an empty estimated
        identified set must not suppress the FLCI). Also the B2b center-parity lock:
        the nested optimizer now matches R HonestDiD's optimal FLCI CENTER at
        intermediate M (previously the flat Nelder-Mead drifted up to ~9%). This
        ``[1, 0, 1]`` curved pre-trend is the kink-prone case; R centers are the
        deterministic (analytical-quantile) values, matching R's own MC output to its
        simulation noise. See ``TestHonestFLCIParityR`` for the full stress grid.
        """
        from diff_diff.honest_did import _compute_optimal_flci

        beta_pre = np.array([1.0, 0.0, 1.0])
        beta_post = np.array([2.0])
        sigma = np.eye(4) * 0.01
        lvec = np.array([1.0])
        # R HonestDiD 0.2.6, deterministic (analytical .qfoldednormal) centers.
        r_center = {0.0: 2.285755, 0.02: 2.295042, 0.05: 2.326349, 0.10: 2.474526}
        for M in [0.0, 0.02, 0.05, 0.1]:
            ci_lb, ci_ub = _compute_optimal_flci(
                beta_pre, beta_post, sigma, lvec, 3, 1, M=M, df=None
            )
            assert np.isfinite(ci_lb) and np.isfinite(ci_ub), f"M={M}: non-finite FLCI"
            assert ci_ub > ci_lb, f"M={M}: degenerate CI"
            center = (ci_lb + ci_ub) / 2
            assert abs(center - r_center[M]) < 1e-3, (
                f"M={M} center={center:.6f} vs R {r_center[M]:.6f}"
            )
        # M=0: tight R parity on the endpoints too (R 0.2.6: [2.082880, 2.488631]).
        ci_lb, ci_ub = _compute_optimal_flci(
            beta_pre, beta_post, sigma, lvec, 3, 1, M=0.0, df=None
        )
        assert abs(ci_lb - 2.082880) < 1e-3, f"M=0 ci_lb={ci_lb:.6f} vs R 2.082880"
        assert abs(ci_ub - 2.488631) < 1e-3, f"M=0 ci_ub={ci_ub:.6f} vs R 2.488631"

    def test_zero_sd_covariance_returns_nan(self):
        """Degenerate (zero) covariance -> zero estimator SD -> NaN inference (no
        ZeroDivisionError in M*bias/h), for M=0 and M>0 (zero-SE convention)."""
        from diff_diff.honest_did import _compute_optimal_flci

        beta_pre = np.array([0.1, -0.05, 0.1])
        beta_post = np.array([1.0])
        sigma = np.zeros((4, 4))
        lvec = np.array([1.0])
        for M in [0.0, 0.1, 0.5]:
            ci_lb, ci_ub = _compute_optimal_flci(beta_pre, beta_post, sigma, lvec, 3, 1, M=M)
            assert np.isnan(ci_lb) and np.isnan(ci_ub), f"M={M}: expected NaN CI on zero SD"

    def test_signed_contrast_lvec_matches_r(self):
        """A signed contrast target (l_vec=[1, -1], a difference of post-period
        effects) matches R HonestDiD's optimal FLCI. Our worst-case-bias objective
        is a faithful port of R's ``.createObjectiveObjectForBias``; for signed
        (non-averaging) l_vec that closed form is CONSERVATIVE relative to the exact
        LP-oracle bias ``_compute_worst_case_bias`` (the two coincide only for
        nonnegative l_vec) -- we match R either way. R HonestDiD 0.2.6 (deterministic
        quantile): center 0.200000, half-length 4.637049.
        """
        from diff_diff.honest_did import _compute_optimal_flci

        lb, ub = _compute_optimal_flci(
            np.array([0.1, 0.0, 0.1]), np.array([1.2, 0.9]),
            np.eye(5) * 0.05, np.array([1.0, -1.0]), 3, 2, M=1.0,
        )
        assert abs((lb + ub) / 2 - 0.200000) < 1e-3, f"center={(lb + ub) / 2:.6f} vs R 0.200000"
        assert abs((ub - lb) / 2 - 4.637049) < 1e-3, f"half={(ub - lb) / 2:.6f} vs R 4.637049"

    def test_negative_M_raises(self):
        """Negative smoothness in the direct FLCI helper raises (cv_alpha's abs()
        would otherwise silently treat M=-0.1 like M=0.1)."""
        from diff_diff.honest_did import _compute_optimal_flci

        with pytest.raises(ValueError, match="non-negative"):
            _compute_optimal_flci(
                np.array([0.1, 0.0, 0.1]), np.array([1.0]),
                np.eye(4) * 0.01, np.array([1.0]), 3, 1, M=-0.1,
            )


# =============================================================================
# TestHonestFLCIParityR - B2b optimal-FLCI parity vs R HonestDiD 0.2.6
# =============================================================================

_FLCI_GOLDEN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks",
    "data",
    "honest_flci_golden.json",
)


@pytest.mark.skipif(
    not os.path.exists(_FLCI_GOLDEN),
    reason="HonestDiD FLCI golden absent (partial checkout); "
    "run benchmarks/R/generate_honest_flci_golden.R",
)
class TestHonestFLCIParityR:
    """B2b: the nested optimal-FLCI optimizer matches R HonestDiD 0.2.6 across a
    stress grid (num_pre x num_post x AR(1) rho x M x l_vec).

    ``center``/``half_length``/``optimal_vec`` in the golden are override-R
    (R's Monte-Carlo ``.qfoldednormal`` replaced by an analytical quantile, so R
    solves the same deterministic outer problem as diff-diff's analytical
    ``_cv_alpha``) -> parity to 1e-3. The same diff-diff values also match stock
    (MC) R (``stock_center``) to ~1e-2, the residual being R's simulation noise on
    the near-flat width surface (diff-diff is strictly more accurate).
    """

    @staticmethod
    def _cases():
        with open(_FLCI_GOLDEN) as fh:
            return json.load(fh)["cases"]

    @staticmethod
    def _unpack(c):
        npre, npost = c["num_pre"], c["num_post"]
        beta = np.asarray(c["beta"], float)
        sigma = np.asarray(c["sigma"], float)
        l_vec = np.atleast_1d(np.asarray(c["l_vec"], float))  # npost==1 auto-unboxed
        return beta[:npre], beta[npre:], sigma, l_vec, npre, npost

    def test_center_and_halflength_match_override_r(self):
        max_dc = max_dw = 0.0
        for c in self._cases():
            if c["center"] is None:
                continue
            bp, bq, sigma, l_vec, npre, npost = self._unpack(c)
            lb, ub = _compute_optimal_flci(bp, bq, sigma, l_vec, npre, npost, c["M"], c["alpha"])
            center, half = (lb + ub) / 2, (ub - lb) / 2
            tag = f"npre={npre} npost={npost} M={c['M']}"
            assert abs(center - c["center"]) < 1e-3, f"{tag}: center {center:.6f} vs R {c['center']:.6f}"
            assert abs(half - c["half_length"]) < 1e-3, f"{tag}: half {half:.6f} vs R {c['half_length']:.6f}"
            max_dc = max(max_dc, abs(center - c["center"]))
            max_dw = max(max_dw, abs(half - c["half_length"]))
        assert max_dc < 1e-3 and max_dw < 1e-3

    def test_optimal_vec_matches_override_r(self):
        from diff_diff.honest_did import _flci_solve

        for c in self._cases():
            if c["optimal_vec"] is None:
                continue
            bp, bq, sigma, l_vec, npre, npost = self._unpack(c)
            _, _, v = _flci_solve(bp, bq, sigma, l_vec, npre, npost, c["M"], c["alpha"])
            r_v = np.atleast_1d(np.asarray(c["optimal_vec"], float))
            assert v is not None and np.max(np.abs(v - r_v)) < 3e-3, (
                f"npre={npre} M={c['M']}: optvec gap {np.max(np.abs(v - r_v)):.2e}"
            )

    def test_center_matches_stock_r_within_mc_noise(self):
        for c in self._cases():
            if c["stock_center"] is None:
                continue
            bp, bq, sigma, l_vec, npre, npost = self._unpack(c)
            lb, ub = _compute_optimal_flci(bp, bq, sigma, l_vec, npre, npost, c["M"], c["alpha"])
            # Loose tier: R's MC .qfoldednormal noise reaches ~1.4e-2 on flat-surface
            # cases (max observed 1.39e-2); the tight parity is vs override-R (1e-3).
            assert abs((lb + ub) / 2 - c["stock_center"]) < 1.5e-2, (
                f"npre={npre} M={c['M']}: vs stock-R {abs((lb + ub) / 2 - c['stock_center']):.2e}"
            )

    def test_inner_solve_failure_nan_consistent(self, monkeypatch):
        """A failed / infeasible inner solve NaN-propagates the full FLCI (no silent
        fallback to a wrong estimator), distinct from R's legitimate Inf-bias branch."""
        import diff_diff.honest_did as hd

        monkeypatch.setattr(
            hd, "_flci_min_bias_given_h",
            lambda P, h, x0_w=None: (np.zeros(P["num_pre"]), np.nan, False),
        )
        lb, ub = _compute_optimal_flci(
            np.array([0.1, -0.05, 0.1]), np.array([1.2]), np.eye(4) * 0.01, np.array([1.0]), 3, 1, M=0.1
        )
        assert np.isnan(lb) and np.isnan(ub)


# =============================================================================
# TestBreakdownValueMethodology
# =============================================================================


class TestBreakdownValueMethodology:
    """Verify breakdown value properties."""

    def test_breakdown_monotonicity(self):
        """If significant at M=k, should be significant at all M < k."""
        from diff_diff.results import MultiPeriodDiDResults, PeriodEffect

        # Use a weak effect so breakdown is reachable at moderate M
        period_effects = {
            1: PeriodEffect(period=1, effect=0.1, se=0.05, t_stat=2.0,
                           p_value=0.05, conf_int=(0.0, 0.2)),
            2: PeriodEffect(period=2, effect=0.05, se=0.05, t_stat=1.0,
                           p_value=0.32, conf_int=(-0.05, 0.15)),
            4: PeriodEffect(period=4, effect=0.15, se=0.05, t_stat=3.0,
                           p_value=0.003, conf_int=(0.05, 0.25)),
        }
        results = MultiPeriodDiDResults(
            avg_att=0.15, avg_se=0.05, avg_t_stat=3.0, avg_p_value=0.003,
            avg_conf_int=(0.05, 0.25), n_obs=500, n_treated=250, n_control=250,
            period_effects=period_effects, pre_periods=[1, 2], post_periods=[4],
            vcov=np.eye(3) * 0.0025,
            interaction_indices={1: 0, 2: 1, 4: 2},
        )

        honest = HonestDiD(method="smoothness")
        # Check that CI at M=0 does not include zero
        r0 = honest.fit(results, M=0.0)
        assert r0.ci_lb > 0, "Should be significant at M=0"

        # At sufficiently large M, CI should include zero.
        # The optimal FLCI is efficient, so need large M for a weak effect.
        r_large = honest.fit(results, M=20.0)
        assert r_large.ci_lb <= 0 <= r_large.ci_ub, "Should lose significance at large M"


class TestARPVertexEnumeration:
    """Diagnostic warnings on `_enumerate_vertices` vertex-search pathologies."""

    def test_enumerate_vertices_warns_on_exhausted_search(self):
        """All-LinAlgError path: fully-zero nuisance column makes A_sys
        singular on every basis, so the enumeration exhausts without
        feasible vertices and the user should see a RuntimeWarning rather
        than a silent empty-list return."""
        from diff_diff.honest_did import _enumerate_vertices

        # 4 moments, 1 nuisance column (all zeros) → A_sys singular on every basis
        X_tilde = np.zeros((4, 1))
        sigma_tilde_diag = np.array([1.0, 1.0, 1.0, 1.0])
        with pytest.warns(RuntimeWarning, match="exhausted"):
            vertices = _enumerate_vertices(X_tilde, sigma_tilde_diag, n_moments=4)
        assert vertices == []

    def test_enumerate_vertices_warns_on_heavy_rejection(self):
        """Mixed-basis path: 5 moments, 1 nuisance column. C(5, 2) = 10
        bases. By design, 6 bases hit LinAlgError (the singular pairs
        among indices {0,1,2,3} that share aligned nuisance/sigma values)
        and 4 bases produce feasible vertices (the (i, 4) pairs that pair
        a positive-X_tilde row with the unique negative-X_tilde row at
        index 4). 60% rejection rate trips the `heavily constrained`
        branch specifically, not the exhausted branch."""
        from diff_diff.honest_did import _enumerate_vertices

        X_tilde = np.array([[1.0], [1.0], [1.0], [2.0], [-1.0]])
        sigma_tilde_diag = np.array([1.0, 1.0, 1.0, 2.0, 1.0])
        with pytest.warns(RuntimeWarning, match="heavily constrained"):
            vertices = _enumerate_vertices(X_tilde, sigma_tilde_diag, n_moments=5)
        assert len(vertices) >= 1, (
            f"Heavy-rejection construction must still produce some feasible "
            f"vertices (otherwise the exhausted branch fires); got "
            f"{len(vertices)} vertices."
        )

    def test_enumerate_vertices_quiet_on_healthy_enumeration(self):
        """Well-conditioned X_tilde: most bases solve cleanly and feasible
        vertices are recovered. No RuntimeWarning should fire."""
        from diff_diff.honest_did import _enumerate_vertices

        rng = np.random.default_rng(0)
        # 4 moments, 1 nuisance — small and well-conditioned
        X_tilde = rng.normal(size=(4, 1))
        sigma_tilde_diag = np.array([1.0, 1.0, 1.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            vertices = _enumerate_vertices(X_tilde, sigma_tilde_diag, n_moments=4)
        diag_warnings = [
            w for w in caught
            if "exhausted" in str(w.message) or "heavily constrained" in str(w.message)
        ]
        assert not diag_warnings, (
            f"Healthy enumeration must not emit ARP diagnostics; got "
            f"{[str(w.message) for w in diag_warnings]}"
        )
        # Sanity: we expect some feasible vertices on a well-conditioned input
        assert isinstance(vertices, list)
