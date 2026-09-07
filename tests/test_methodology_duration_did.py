"""Methodology tests for DurationDiD against Deaner & Ku (2026).

Checks the Theorem 1 identities on exact population survival curves, the
hand-computed micro panel (CD holds exactly in sample), the PH mean-of-ratios
choice against the printed reciprocal, the centered-bootstrap conventions on
fixed replicate matrices (Appendix B Algorithm 1), the fixed-anchor Algorithm
2 contrast, whole-individual resampling, time-rescaling invariance, and
sampling behavior on the exact absorbing DGP. Imports the private seams of
``diff_diff.duration_did`` directly. API/validation tests live in
``tests/test_duration_did.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import diff_diff.duration_did as dd_module
from diff_diff import DurationDiD
from diff_diff.duration_did import (
    _centered_bootstrap_summary,
    _curve_status,
    _estimate_from_survival,
    _group_survival,
    _log_survival_moments,
    _pretest_contrasts,
    _quantile_inverted_cdf,
)
from tests.test_duration_did import build_from_survivors, fit_quiet, micro_panel

# ---------------------------------------------------------------------------
# Exact population DGP (per-period integrated hazard increments)
# ---------------------------------------------------------------------------


def population_curves(n_periods=8, tstar_idx=3, c=0.05, beta=0.3, method="cd", s11=0.6, s21=0.8):
    """Exact group survival curves under CD (additive) or PH (ratio).

    Returns ``S1_fact`` (treated factual), ``S1_cf`` (treated counterfactual),
    ``S2`` (control) and the population post-date effects ``tau``.
    """
    s = np.arange(2, n_periods + 1)
    lam2 = 0.15 + 0.02 * s
    lam1 = lam2 + c if method == "cd" else c * lam2
    lam1_fact = lam1.copy()
    lam1_fact[tstar_idx:] += beta
    S2 = s21 * np.exp(-np.concatenate([[0.0], np.cumsum(lam2)]))
    S1_cf = s11 * np.exp(-np.concatenate([[0.0], np.cumsum(lam1)]))
    S1_fact = s11 * np.exp(-np.concatenate([[0.0], np.cumsum(lam1_fact)]))
    return S1_fact, S1_cf, S2, (S1_cf - S1_fact)[tstar_idx + 1 :]


def simulate(n, S1, S2, seed, n_periods=None):
    rng = np.random.default_rng(seed)
    n_periods = len(S1) if n_periods is None else n_periods
    u = rng.uniform(size=2 * n)
    g = np.repeat([1, 0], n)
    S_fact = np.where(g[:, None] == 1, S1[None, :], S2[None, :])
    Y = (u[:, None] > S_fact).astype(int)
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(2 * n), n_periods),
            "time": np.tile(np.arange(1, n_periods + 1), 2 * n),
            "treated": np.repeat(g, n_periods),
            "exited": Y.ravel(),
        }
    )


def core_on_curves(S1, S2, elapsed, fit_idx, weights, method):
    S = np.stack([S1, S2])[None]
    R, D, H = _log_survival_moments(S, elapsed)
    c, R0, S0, tau = _estimate_from_survival(S, R, D, elapsed, fit_idx, weights, method)
    return c[0], R0[0], S0[0], tau[0], D[0], H[0]


# ---------------------------------------------------------------------------
# Population identities (Theorem 1)
# ---------------------------------------------------------------------------


class TestPopulationIdentities:
    @pytest.mark.parametrize("method, c", [("cd", 0.05), ("cd", -0.05), ("ph", 1.5)])
    @pytest.mark.parametrize("weights", [None, np.array([0.7, 0.2, 0.1])])
    def test_exact_recovery(self, method, c, weights):
        S1_fact, S1_cf, S2, tau_true = population_curves(c=c, method=method)
        elapsed = np.arange(8, dtype=float)
        fit_idx = np.array([1, 2, 3])
        w = np.ones(3) / 3 if weights is None else weights
        c_hat, R0, S0, tau, _, _ = core_on_curves(S1_fact, S2, elapsed, fit_idx, w, method)
        assert c_hat == pytest.approx(c, abs=1e-12)
        np.testing.assert_allclose(S0, S1_cf, atol=1e-12)
        np.testing.assert_allclose(tau[4:], tau_true, atol=1e-12)

    def test_unequal_baseline_and_ph_baseline_outside_exponent(self):
        # S_1t(0) = S_11 (S_2t / S_21)^c: the treated baseline is a factor,
        # not inside the exponent (review lines 262-265).
        S1_fact, S1_cf, S2, _ = population_curves(method="ph", c=1.5, s11=0.6, s21=0.8)
        elapsed = np.arange(8, dtype=float)
        _, _, S0, _, _, _ = core_on_curves(
            S1_fact, S2, elapsed, np.array([1, 2, 3]), np.ones(3) / 3, "ph"
        )
        np.testing.assert_allclose(S0, 0.6 * (S2 / 0.8) ** 1.5, atol=1e-12)
        assert S0[0] == pytest.approx(0.6)

    def test_ph_mean_ratio_vs_printed_reciprocal(self):
        # Review lines 1015-1019: the printed 3.5 slope returns 1/c under exact PH.
        d2 = np.array([0.0, 0.10, 0.30, 0.45, 0.60])
        d1 = 2 * d2
        pre = np.array([1, 2])
        ratio = np.mean(d1[pre] / d2[pre])
        printed = np.dot(d1[pre], d2[pre]) / np.dot(d1[pre], d1[pre])
        assert ratio == 2 and printed == 0.5

    def test_three_finite_sample_ph_choices_differ(self):
        # Review lines 1020-1023.
        x, y, dt = np.array([0.1, 0.3]), np.array([0.2, 0.9]), np.array([1, 2])
        choices = (
            np.mean(y / x),
            np.dot(x, y) / np.dot(x, x),
            np.dot(x / dt, y / dt) / np.dot(x / dt, x / dt),
        )
        np.testing.assert_allclose(choices, [2.5, 2.9, 2.6923076923076925])
        # The estimator implements the first (mean of ratios).
        S1 = np.exp(-np.array([0.0, 0.2, 0.9]))
        S2 = np.exp(-np.array([0.0, 0.1, 0.3]))
        c_hat, *_ = core_on_curves(
            S1, S2, np.array([0.0, 1.0, 2.0]), np.array([1, 2]), np.array([0.5, 0.5]), "ph"
        )
        assert c_hat == pytest.approx(2.5)

    def test_max_abs_sign_symmetry(self):
        z = np.array([-4.0, 1.0])
        assert np.max(np.abs(z)) == 4 and abs(np.max(z)) == 1


# ---------------------------------------------------------------------------
# Hand-computed micro panel
# ---------------------------------------------------------------------------


class TestMicroPanel:
    def test_cd_exact(self):
        r = fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3)
        assert r.coefficient == pytest.approx(math.log(1.5), abs=1e-14)
        np.testing.assert_allclose(r.att_by_period, [1 / 30, 8 / 270], atol=1e-14)
        assert r.survival_treated[-1] == 0.0  # zero treated post-survival is fine
        assert abs(r.pretest.contrast[0]) < 1e-12
        assert r.curve_status == ["ok"] * 5
        assert r.n_treated_survivors_at_last_pre == 4 and r.n_control_survivors_at_horizon == 2

    def test_ph_values(self):
        r = fit_quiet(DurationDiD(method="ph", n_bootstrap=0), micro_panel(), last_pre_period=3)
        assert r.coefficient == pytest.approx(2.2896729, abs=1e-6)
        np.testing.assert_allclose(r.att_by_period, [0.0301737, 0.0250975], atol=1e-6)
        assert r.pretest.contrast[0] == pytest.approx(0.2394958, abs=1e-6)
        assert r.curve_status == ["ok"] * 5

    def test_gemm_path_bit_identical_to_loop(self):
        df = micro_panel()
        arranged = dd_module._validate_and_arrange(df, "exited", "unit", "time", "treated", 3)
        Y, n1 = arranged["Y"], arranged["n_treated"]
        rng = np.random.default_rng(0)
        idx = rng.integers(0, Y.shape[0], size=(7, Y.shape[0]))
        W = np.stack([np.bincount(row, minlength=Y.shape[0]) for row in idx]).astype(float)
        S, _, _ = _group_survival(Y, n1, W)
        for b in range(7):
            Yb = Y[idx[b]]
            gb = np.concatenate([np.ones(n1), np.zeros(Y.shape[0] - n1)])[idx[b]]
            S1 = 1 - Yb[gb == 1].mean(axis=0)
            S2 = 1 - Yb[gb == 0].mean(axis=0)
            assert np.array_equal(S[b, 0], S1) and np.array_equal(S[b, 1], S2)

    def test_algorithm2_anchor_distinct_from_moving_window(self):
        # Fixed anchor (Algorithm 2) vs the main-text moving final window at t=2
        # (review lines 449-465): both are valid null contrasts but differ.
        df = build_from_survivors(30, [18, 9, 5, 1, 0], 10, [8, 6, 4, 3, 2])
        arranged = dd_module._validate_and_arrange(df, "exited", "unit", "time", "treated", 3)
        S, _, _ = _group_survival(arranged["Y"], arranged["n_treated"], np.ones((1, 40)))
        R, D, H = _log_survival_moments(S, arranged["elapsed"])
        fixed = _pretest_contrasts(D, H, np.array([1]), 2, "cd")[0, 0]
        # moving window: (D_12 - D_22)/1 - [(R_13 - R_12) - (R_23 - R_22)]/1
        moving = (D[0, 0, 1] - D[0, 1, 1]) - ((R[0, 0, 2] - R[0, 0, 1]) - (R[0, 1, 2] - R[0, 1, 1]))
        assert fixed != pytest.approx(moving)


# ---------------------------------------------------------------------------
# Centered bootstrap conventions (fixed replicate matrices)
# ---------------------------------------------------------------------------


class TestCenteredBootstrap:
    @pytest.mark.parametrize("n", [20, 100, 999, 1000])
    def test_quantile_inverted_cdf(self, n):
        x = np.arange(1, n + 1, dtype=float)
        rng = np.random.default_rng(n)
        rng.shuffle(x)
        assert _quantile_inverted_cdf(x, 0.95) == math.ceil(0.95 * n)
        assert _quantile_inverted_cdf(x, 0.5) == math.ceil(0.5 * n)
        assert _quantile_inverted_cdf(x, 1e-9) == 1.0

    def test_fixed_matrix_conventions(self):
        rng = np.random.default_rng(1)
        draws = rng.normal(size=(20, 3)) * np.array([1.0, 2.0, 0.5]) + np.array([0.5, -0.2, 0.1])
        point = np.array([0.4, -0.1, 0.1])
        s = _centered_bootstrap_summary(point, draws, alpha=0.05)
        np.testing.assert_array_equal(s["se"], np.sqrt(np.diag(s["vcov"])))
        np.testing.assert_allclose(s["se"], np.std(draws, axis=0, ddof=1), rtol=1e-12)
        z = np.abs(draws - point) / s["se"]
        for k in range(3):
            assert s["crit"][k] == np.sort(z[:, k])[math.ceil(0.95 * 20) - 1]
            assert s["crit_sim"] >= s["crit"][k]
            assert s["band_upper"][k] - s["band_lower"][k] >= s["ci_upper"][k] - s["ci_lower"][k]
        assert s["p_joint"] == np.mean(z.max(axis=1) >= np.max(np.abs(point / s["se"])))

    def test_band_p_duality_including_ties(self):
        # B=20, alpha=0.05: crit = 19th order statistic; band excludes zero
        # iff p <= 0.05 iff at most one draw is >= |t|.
        z_vals = np.linspace(0.1, 2.0, 20)  # exact centered pivots we will force
        for t_abs in [z_vals[18] + 1e-9, z_vals[18], z_vals[18] - 1e-9, z_vals[17]]:
            draws = t_abs + z_vals[:, None]  # point + z (se ~ std of z)
            point = np.array([t_abs])
            s = _centered_bootstrap_summary(point, draws, alpha=0.05)
            se = s["se"][0]
            z = np.abs(draws[:, 0] - t_abs) / se
            t_stat = abs(t_abs / se)
            band_excludes = t_stat > s["crit"][0]
            p = np.mean(z >= t_stat)
            assert band_excludes == (p <= 0.05)
            assert s["p"][0] == p

    def test_single_column_covariance_shape(self):
        s = _centered_bootstrap_summary(
            np.array([0.3]), np.random.default_rng(0).normal(size=(10, 1)), 0.05
        )
        assert s["vcov"].shape == (1, 1)
        assert s["crit_sim"] == s["crit"][0]


# ---------------------------------------------------------------------------
# Sampling behavior on the exact DGP
# ---------------------------------------------------------------------------


class TestSampling:
    @pytest.mark.parametrize("method, c", [("cd", 0.05), ("ph", 1.5)])
    def test_point_estimates_near_truth(self, method, c):
        S1, _, S2, tau_true = population_curves(c=c, method=method)
        df = simulate(20_000, S1, S2, seed=11)
        r = fit_quiet(DurationDiD(method=method, n_bootstrap=0), df, last_pre_period=4)
        np.testing.assert_allclose(r.att_by_period, tau_true, atol=0.03)
        assert r.coefficient == pytest.approx(c, abs=0.05 if method == "cd" else 0.4)

    def test_null_effect(self):
        S1, _, S2, tau_true = population_curves(beta=0.0)
        assert np.allclose(tau_true, 0)
        df = simulate(20_000, S1, S2, seed=5)
        r = fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=4)
        np.testing.assert_allclose(r.att_by_period, 0, atol=0.03)

    def test_se_calibration(self, ci_params):
        S1, _, S2, _ = population_curves()
        n_rep = ci_params.bootstrap(300, min_n=60)
        ests = np.array(
            [
                fit_quiet(
                    DurationDiD(n_bootstrap=0), simulate(1000, S1, S2, seed=s), last_pre_period=4
                ).att_by_period
                for s in range(n_rep)
            ]
        )
        mc_sd = ests.std(axis=0, ddof=1)
        r = fit_quiet(
            DurationDiD(n_bootstrap=ci_params.bootstrap(300, min_n=60), seed=0),
            simulate(1000, S1, S2, seed=999),
            last_pre_period=4,
        )
        ratio = r.se_by_period / mc_sd
        assert np.all((ratio > 0.7) & (ratio < 1.4)), ratio

    def test_time_rescaling_invariance(self):
        df = micro_panel()
        r1 = fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)
        df3 = df.copy()
        df3["time"] = 10 + 3 * (df3["time"] - 1)
        r3 = fit_quiet(DurationDiD(n_bootstrap=0), df3, last_pre_period=16)
        np.testing.assert_allclose(r3.att_by_period, r1.att_by_period, atol=1e-12)
        assert r3.coefficient == pytest.approx(r1.coefficient / 3)
        rp1 = fit_quiet(DurationDiD(method="ph", n_bootstrap=0), df, last_pre_period=3)
        rp3 = fit_quiet(DurationDiD(method="ph", n_bootstrap=0), df3, last_pre_period=16)
        assert rp3.coefficient == pytest.approx(rp1.coefficient)

    def test_headline_uses_same_weights_in_every_draw(self):
        S1, _, S2, _ = population_curves()
        r = fit_quiet(
            DurationDiD(n_bootstrap=30, seed=2), simulate(300, S1, S2, seed=3), last_pre_period=4
        )
        assert r.att == pytest.approx(np.mean(r.att_by_period))
        head_draws = r.bootstrap_effects.mean(axis=1)
        assert r.se == pytest.approx(np.std(head_draws, ddof=1), rel=1e-10)

    def test_curve_status_helper(self):
        assert (
            _curve_status(np.array([0.0, 0.5, 1.0, 1.5]), np.array([1, 0.5, 0.3, 0.1]), 1)
            == ["ok"] * 4
        )
        assert _curve_status(np.array([0.0, -0.1, 1.0, 1.5]), np.array([1, 0.5, 0.3, 0.1]), 1)[
            1
        ] == ("counterfactual_survival_above_one")
        assert _curve_status(np.array([0.0, 0.5, 0.4, 1.5]), np.array([1, 0.5, 0.3, 0.1]), 1)[
            2
        ] == ("counterfactual_nonmonotone")
        assert _curve_status(np.array([0.0, 0.5, np.inf, np.inf]), np.array([1, 0.5, 0.0, 0.0]), 1)[
            2:
        ] == [
            "control_survival_zero",
            "control_survival_zero",
        ]
        assert _curve_status(np.array([0.0, np.nan, 1.0, 1.5]), np.array([1, 0.5, 0.3, 0.1]), 1)[
            1
        ] == ("counterfactual_nonfinite")


@pytest.mark.slow
class TestCoverage:
    def test_uniform_coverage_near_nominal(self, ci_params):
        S1, _, S2, tau_true = population_curves()
        n_rep = ci_params.bootstrap(200, min_n=40)
        covered = 0
        for s in range(n_rep):
            r = fit_quiet(
                DurationDiD(n_bootstrap=200, seed=s),
                simulate(500, S1, S2, seed=1000 + s),
                last_pre_period=4,
            )
            covered += int(np.all((r.cband_lower <= tau_true) & (tau_true <= r.cband_upper)))
        assert abs(covered / n_rep - 0.95) < 0.06
