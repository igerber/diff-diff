"""Methodology tests for the Chang (2020) Case 1 orthogonal score.

The score objects under test are pure functions (``diff_diff/_dr_scores.py``);
no estimator is involved. The DGP uses ORACLE (true) nuisances so the tests
check the score algebra, not any learner.

DGP: X ~ N(0, I_3); g0(X) = expit(0.8*X0 - 0.5*X1); D ~ Bernoulli(g0);
dY = ell0(X) + theta0 * D + eps, ell0(X) = X0 + 0.5*X1^2, theta0 = 2.5.
Under this DGP E[dY(0) | X, D=0] = ell0(X) and the ATT is theta0.
"""

import numpy as np
import pytest

from diff_diff._dr_scores import (
    chang_panel_score,
    chang_panel_score_augmented,
    chang_rcs_lambda_slope,
    chang_rcs_score,
    chang_rcs_score_augmented,
)

THETA0 = 2.5

# DoubleML golden parity numbers, captured from the committed spike
# benchmarks/doubleml/chang_case1_parity.py under pinned doubleml==0.11.4
# (sklearn 1.9.0, macOS arm64, 2026-08-22). The spike matches its own
# hand-rolled Chang estimator to ~4e-16; the native reproduction below swaps
# sklearn's lbfgs logistic MLE for linalg.solve_logit's IRLS, so agreement is
# solver-tolerance-level (~5e-5 ATT / ~1e-6 SE) — far below the ~2.3e-3 shift
# that a p-hat-convention change produces, which is what this test guards.
DOUBLEML_GOLDEN_ATT = 3.260530717619
DOUBLEML_GOLDEN_SE = 0.173029660048


def _dgp(n=200_000, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    g0 = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
    D = (rng.uniform(size=n) < g0).astype(float)
    ell0 = X[:, 0] + 0.5 * X[:, 1] ** 2
    dY = ell0 + THETA0 * D + rng.normal(scale=0.7, size=n)
    return X, g0, D, ell0, dY


class TestChangScoreMethodology:
    def test_score_mean_recovers_att_with_oracle_nuisances(self):
        X, g0, D, ell0, dY = _dgp()
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        summand = chang_panel_score(dY, D, ell0, ps, float(D.mean()))
        assert abs(summand.mean() - THETA0) < 0.02

    def test_double_robustness_misspecified_propensity(self):
        # Wrong ps (constant), correct outcome regression -> still recovers ATT.
        X, g0, D, ell0, dY = _dgp()
        ps_wrong = np.full_like(g0, 0.3)
        summand = chang_panel_score(dY, D, ell0, ps_wrong, float(D.mean()))
        assert abs(summand.mean() - THETA0) < 0.02

    def test_double_robustness_misspecified_outcome_regression(self):
        # Wrong m (zero), correct propensity -> still recovers ATT.
        X, g0, D, ell0, dY = _dgp()
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        m_wrong = np.zeros_like(ell0)
        summand = chang_panel_score(dY, D, m_wrong, ps, float(D.mean()))
        assert abs(summand.mean() - THETA0) < 0.03

    def test_augmented_score_variance_matches_hand_formula(self):
        """SE = sqrt(mean(psi_bar**2)/N) with psi_bar = summand - D*theta/p_hat.

        Hand-recompute the whole chain on a small fixed sample and compare
        element-wise and in the SE aggregate.
        """
        X, g0, D, ell0, dY = _dgp(n=500, seed=4)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat = float(D.mean())
        summand = chang_panel_score(dY, D, ell0, ps, p_hat)
        theta = float(summand.mean())
        psi_bar = chang_panel_score_augmented(summand, D, theta, p_hat)

        # Hand formula, written out independently:
        w = (D - ps * (1 - D) / (1 - ps)) / p_hat
        hand_summand = w * (dY - ell0)
        hand_psi_bar = hand_summand - D * theta / p_hat
        np.testing.assert_allclose(psi_bar, hand_psi_bar, atol=1e-14, rtol=0)

        se = np.sqrt(np.mean(psi_bar**2) / len(D))
        hand_se = np.sqrt(np.mean(hand_psi_bar**2) / len(D))
        np.testing.assert_allclose(se, hand_se, atol=1e-15, rtol=0)
        assert 0.0 < se < 1.0  # sane magnitude on this DGP

    def test_augmented_score_is_mean_centered_at_theta_hat(self):
        """psi_bar has (near-)zero mean when theta is the score-implied ATT."""
        X, g0, D, ell0, dY = _dgp(n=2_000, seed=9)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat = float(D.mean())
        summand = chang_panel_score(dY, D, ell0, ps, p_hat)
        theta = float(summand.mean())
        psi_bar = chang_panel_score_augmented(summand, D, theta, p_hat)
        # mean(psi_bar) = theta - theta * mean(D)/p_hat = 0 exactly (p_hat = mean(D))
        assert abs(psi_bar.mean()) < 1e-12

    @pytest.mark.parametrize("direction", ["g", "ell"])
    def test_neyman_orthogonality_vs_nonorthogonal_comparator(self, direction):
        """First-order insensitivity to perturbations of the INFINITE-dimensional
        nuisances (g, ell) ONLY — p_hat is deliberately excluded (Chang Lemma 1
        orthogonalizes w.r.t. the infinite-dimensional nuisances; the estimated
        p_hat is handled by the variance correction instead).

        The population Gateaux derivative of the orthogonal score's mean is
        zero at the truth, so its finite-sample shift under an eps-perturbation
        is only Monte Carlo noise; a NON-orthogonal comparator score suffers an
        O(eps) shift under the SAME perturbation. The test asserts the
        orthogonal shift is at least 20x smaller than the comparator's.
        """
        X, g0, D, ell0, dY = _dgp(n=400_000, seed=21)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat = float(D.mean())
        eps = 0.05
        h = 0.5 * X[:, 2] + 0.3  # fixed perturbation direction with E[h] != 0

        base_orth = chang_panel_score(dY, D, ell0, ps, p_hat).mean()
        if direction == "ell":
            # Orthogonal score vs regression-adjustment plug-in (no ps
            # re-weighting of controls): the plug-in's m-derivative is
            # -E[D h]/p != 0, the orthogonal score's is E[w h] = 0.
            pert_orth = chang_panel_score(dY, D, ell0 + eps * h, ps, p_hat).mean()
            base_naive = (D * (dY - ell0) / p_hat).mean()
            pert_naive = (D * (dY - (ell0 + eps * h)) / p_hat).mean()
        else:
            # Orthogonal score vs Abadie IPW plug-in (no outcome adjustment):
            # the IPW score's g-derivative involves E[dY | X, D=0] = ell0 != 0.
            ps_pert = np.clip(ps + eps * h * ps * (1 - ps), 1e-3, 1 - 1e-3)
            pert_orth = chang_panel_score(dY, D, ell0, ps_pert, p_hat).mean()

            def ipw(p):
                return ((D - p * (1 - D) / (1 - p)) * dY / p_hat).mean()

            base_naive = ipw(ps)
            pert_naive = ipw(ps_pert)

        shift_orth = abs(pert_orth - base_orth)
        shift_naive = abs(pert_naive - base_naive)
        assert shift_orth < shift_naive / 20.0, (shift_orth, shift_naive)


class TestDoubleMLGoldenParity:
    def test_native_chang_estimator_matches_doubleml_goldens(self):
        """Dependency-free reproduction of the committed DoubleML parity spike.

        Same DGP (default_rng(42) draw order), same fold construction, same
        clipping; nuisances via linalg.solve_logit / solve_ols instead of
        sklearn. Guards the global-p-hat convention: switching to the
        fold-mean convention moves the ATT by ~2.3e-3, two orders of
        magnitude beyond the tolerance.
        """
        import warnings as _w

        from diff_diff.linalg import solve_logit, solve_ols

        rng = np.random.default_rng(42)
        N, d = 500, 5
        Xd = rng.standard_normal((N, d))
        g0 = 1 / (1 + np.exp(-(Xd[:, 0] - 0.5 * Xd[:, 1])))
        D = (rng.uniform(size=N) < g0).astype(float)
        dY = Xd[:, 0] + 0.5 * Xd[:, 2] ** 2 + 3.0 * D + rng.standard_normal(N)
        K, TRIM = 5, 1e-2
        perm = rng.permutation(N)
        tests = [np.sort(perm[i::K]) for i in range(K)]
        p_glob = float(D.mean())

        oof_g = np.empty(N)
        oof_m = np.empty(N)
        with _w.catch_warnings():
            _w.simplefilter("ignore")  # near-separation warnings are expected
            for te in tests:
                tr = np.setdiff1d(np.arange(N), te)
                beta, _ = solve_logit(Xd[tr], D[tr], max_iter=200, tol=1e-10)
                eta = np.column_stack([np.ones(te.size), Xd[te]]) @ beta
                oof_g[te] = np.clip(1 / (1 + np.exp(-eta)), TRIM, 1 - TRIM)
                ctrl = tr[D[tr] == 0]
                Xi = np.column_stack([np.ones(ctrl.size), Xd[ctrl]])
                coefs, _, _ = solve_ols(Xi, dY[ctrl], return_vcov=False)
                oof_m[te] = np.column_stack([np.ones(te.size), Xd[te]]) @ coefs

        summand = chang_panel_score(dY, D, oof_m, oof_g, p_glob)
        theta = float(summand.mean())
        psi_bar = chang_panel_score_augmented(summand, D, theta, p_glob)
        se = float(np.sqrt(np.mean(psi_bar**2) / N))

        np.testing.assert_allclose(theta, DOUBLEML_GOLDEN_ATT, atol=2e-4, rtol=0)
        np.testing.assert_allclose(se, DOUBLEML_GOLDEN_SE, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Chang (2020) Case 2 (repeated cross sections) score methodology
# ---------------------------------------------------------------------------
#
# LIBRARY-AUTHORED low-dimensional RCS design in the spirit of Chang Sec. 4
# (Gaussian X, logistic PS, theta0 = 3) — NOT the paper's own Sec. 4 RCS
# parameterization. The paper's Sec. 4.2 RCS DGPs are now extracted into
# docs/methodology/papers/chang-2020-review.md; the Sec. 4.2.2 kernel design
# is replicated in tests/test_methodology_dml_did.py ("Chang Sec. 4.2.2"
# section), and the Sec. 4.2.1 ML design (p in {100, 300}) is tracked in the
# narrowed TODO.md row (needs a penalized propensity learner).
# Rows i.i.d.: X ~ N(0, I_2); D ~ Bernoulli(sigmoid(0.5 X1 - 0.5 X2));
# T ~ Bernoulli(lam0) independent; levels
# Y = 1 + X1 + 0.5 X2 + T*(0.5 + 0.4 X1) + D*1.0 + T*D*theta0 + eps.
# Under stationary sampling the ORACLE Case 2 outcome nuisance is
# l20(X) = E[(T - lam0) Y | X, D=0] = lam0 (1 - lam0) * (0.5 + 0.4 X1).

RCS_THETA0 = 3.0
RCS_LAM0 = 0.5


def _rcs_dgp(n=200_000, seed=13):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    g0 = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.5 * X[:, 1])))
    D = (rng.uniform(size=n) < g0).astype(float)
    T = (rng.uniform(size=n) < RCS_LAM0).astype(float)
    trend = 0.5 + 0.4 * X[:, 0]
    y = (
        1.0
        + X[:, 0]
        + 0.5 * X[:, 1]
        + T * trend
        + D * 1.0
        + T * D * RCS_THETA0
        + rng.normal(scale=1.0, size=n)
    )
    ell20 = RCS_LAM0 * (1.0 - RCS_LAM0) * trend
    return X, g0, D, T, ell20, y


class TestChangRCSScoreMethodology:
    def test_score_mean_recovers_att_with_oracle_nuisances(self):
        X, g0, D, T, ell20, y = _rcs_dgp()
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        summand = chang_rcs_score(y, D, T, ell20, ps, float(D.mean()), float(T.mean()))
        assert abs(summand.mean() - RCS_THETA0) < 0.05

    def test_double_robustness_misspecified_propensity(self):
        # Wrong ps (constant), correct ell2 -> still recovers the ATT: the
        # control-side residual E[(T-lam)Y - ell2 | X, D=0] is zero, and the
        # wrong ps cancels from the treated-side weight.
        X, g0, D, T, ell20, y = _rcs_dgp()
        ps_wrong = np.full_like(g0, 0.3)
        summand = chang_rcs_score(y, D, T, ell20, ps_wrong, float(D.mean()), float(T.mean()))
        assert abs(summand.mean() - RCS_THETA0) < 0.05

    def test_double_robustness_misspecified_outcome_regression(self):
        # Wrong ell2 (zero), correct propensity -> still recovers the ATT:
        # E[(D - g0)/(1 - g0) | X] = 0 kills the wrong-nuisance term.
        X, g0, D, T, ell20, y = _rcs_dgp()
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        m_wrong = np.zeros_like(ell20)
        summand = chang_rcs_score(y, D, T, m_wrong, ps, float(D.mean()), float(T.mean()))
        assert abs(summand.mean() - RCS_THETA0) < 0.05

    def test_closed_form_hand_fixture(self):
        """Element-wise hand recomputation of summand, G_2lambda, psi_bar."""
        X, g0, D, T, ell20, y = _rcs_dgp(n=500, seed=6)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat = float(D.mean())
        lam = float(T.mean())
        summand = chang_rcs_score(y, D, T, ell20, ps, p_hat, lam)
        theta = float(summand.mean())
        psi_bar = chang_rcs_score_augmented(summand, D, T, y, ell20, ps, theta, p_hat, lam)
        g2 = chang_rcs_lambda_slope(y, D, T, ell20, ps, p_hat, lam)

        # Hand formulas, written out independently (Eq 3.2 / Thm 2):
        w = (D - ps) / (p_hat * lam * (1 - lam) * (1 - ps))
        hand_summand = w * ((T - lam) * y - ell20)
        np.testing.assert_allclose(summand, hand_summand, atol=1e-15, rtol=0)
        odds = (D - ps) / (1 - ps)
        hand_g2 = float(
            np.mean(
                -((1 - 2 * lam) / (lam**2 * (1 - lam) ** 2))
                * (odds / p_hat)
                * ((T - lam) * y - ell20)
                - (y / (p_hat * lam * (1 - lam))) * odds
            )
        )
        np.testing.assert_allclose(g2, hand_g2, atol=1e-12, rtol=0)
        hand_psi_bar = hand_summand - D * theta / p_hat + hand_g2 * (T - lam)
        np.testing.assert_allclose(psi_bar, hand_psi_bar, atol=1e-12, rtol=0)

        se = np.sqrt(np.mean(psi_bar**2) / len(D))
        hand_se = np.sqrt(np.mean(hand_psi_bar**2) / len(D))
        np.testing.assert_allclose(se, hand_se, atol=1e-15, rtol=0)

    def test_lambda_slope_first_term_algebraic_identity(self):
        # term1 of d_lam psi_2 equals -((1-2*lam)/(lam*(1-lam))) * summand_i.
        X, g0, D, T, ell20, y = _rcs_dgp(n=400, seed=8)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat, lam = float(D.mean()), 0.4  # lam != mean(T): identity is algebraic
        summand = chang_rcs_score(y, D, T, ell20, ps, p_hat, lam)
        odds = (D - ps) / (1 - ps)
        term1 = (
            -((1 - 2 * lam) / (lam**2 * (1 - lam) ** 2)) * (odds / p_hat) * ((T - lam) * y - ell20)
        )
        np.testing.assert_allclose(
            term1, -((1 - 2 * lam) / (lam * (1 - lam))) * summand, atol=1e-12, rtol=0
        )

    def test_lambda_slope_matches_finite_difference(self):
        # G_2lambda is d/d(lam) of mean(summand) holding the nuisances fixed:
        # central differences of the SCORE function in lam_hat must match.
        X, g0, D, T, ell20, y = _rcs_dgp(n=2_000, seed=10)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat, lam = float(D.mean()), float(T.mean())
        g2 = chang_rcs_lambda_slope(y, D, T, ell20, ps, p_hat, lam)
        h = 1e-6
        up = chang_rcs_score(y, D, T, ell20, ps, p_hat, lam + h).mean()
        dn = chang_rcs_score(y, D, T, ell20, ps, p_hat, lam - h).mean()
        np.testing.assert_allclose(g2, (up - dn) / (2 * h), rtol=1e-6, atol=1e-8)

    def test_p_derivative_identity_matches_finite_difference(self):
        # d_p psi_2 = -(1/p)(psi_2 + theta) = -summand/p; check via central
        # differences of mean(summand) in p_hat.
        X, g0, D, T, ell20, y = _rcs_dgp(n=2_000, seed=12)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat, lam = float(D.mean()), float(T.mean())
        summand = chang_rcs_score(y, D, T, ell20, ps, p_hat, lam)
        expected = -float(summand.mean()) / p_hat
        h = 1e-6
        up = chang_rcs_score(y, D, T, ell20, ps, p_hat + h, lam).mean()
        dn = chang_rcs_score(y, D, T, ell20, ps, p_hat - h, lam).mean()
        np.testing.assert_allclose(expected, (up - dn) / (2 * h), rtol=1e-5, atol=1e-8)

    def test_augmented_score_is_mean_centered_at_theta_hat(self):
        # mean(psi_bar) = theta - theta*mean(D)/p + G2*(mean(T)-lam) = 0
        # exactly when p_hat = mean(D) and lam_hat = mean(T).
        X, g0, D, T, ell20, y = _rcs_dgp(n=2_000, seed=14)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat, lam = float(D.mean()), float(T.mean())
        summand = chang_rcs_score(y, D, T, ell20, ps, p_hat, lam)
        theta = float(summand.mean())
        psi_bar = chang_rcs_score_augmented(summand, D, T, y, ell20, ps, theta, p_hat, lam)
        assert abs(psi_bar.mean()) < 1e-12

    def test_lambda_correction_changes_the_variance(self):
        # The Thm 2 SE with the lambda term differs from the bare
        # (summand - D*theta/p) variance — the "plausible implementation bug"
        # regression at the score level.
        X, g0, D, T, ell20, y = _rcs_dgp(n=5_000, seed=16)
        ps = np.clip(g0, 1e-3, 1 - 1e-3)
        p_hat, lam = float(D.mean()), float(T.mean())
        summand = chang_rcs_score(y, D, T, ell20, ps, p_hat, lam)
        theta = float(summand.mean())
        psi_bar = chang_rcs_score_augmented(summand, D, T, y, ell20, ps, theta, p_hat, lam)
        psi_no_lambda = summand - D * theta / p_hat
        se_full = np.sqrt(np.mean(psi_bar**2) / len(D))
        se_no_lambda = np.sqrt(np.mean(psi_no_lambda**2) / len(D))
        assert abs(se_full - se_no_lambda) / se_full > 1e-4


# ===========================================================================
# CCPS (2026) bad-control score (Equation 10) - pure-function methodology
# ===========================================================================
from diff_diff._dr_scores import ccps_panel_score, ccps_panel_score_augmented  # noqa: E402

CCPS_TAU = 1.0


def _ccps_dgp(n=300_000, seed=17):
    """Two-period bad-control DGP with CLOSED-FORM nuisances (W empty).

    xb, Z ~ N(0, 1); p(S) = expit(0.5 xb + 0.3 Z) is a function of
    S = (xb, Z) only, so omega0(R) = p/(1-p) EXACTLY (Equation 9 with the
    odds already S-measurable); xt(0) = 0.7 xb + 0.3 Z + e (independent of D
    given S: Assumption 4); xt = xt(0) + 0.5 D (a genuine bad control);
    m0(xt, xb, Z) = xt + 0.5 xb + 0.3 Z at the UNTREATED bad control;
    dY = m0 + tau D + noise. Hence nu0(S) = E[m0 | S, D=0] = 1.2 xb + 0.6 Z and
    the ATT is tau. Returns arrays evaluated at the OBSERVED bad control (the
    score only reads m_hat/omega_hat on controls, where xt == xt(0)).
    """
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal(n)
    Z = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-(0.5 * xb + 0.3 * Z)))
    D = (rng.uniform(size=n) < p).astype(float)
    xt0 = 0.7 * xb + 0.3 * Z + 0.3 * rng.standard_normal(n)
    xt = xt0 + 0.5 * D
    m0_untreated = xt0 + 0.5 * xb + 0.3 * Z
    dY = m0_untreated + CCPS_TAU * D + 0.5 * rng.standard_normal(n)
    m_true = xt + 0.5 * xb + 0.3 * Z
    nu_true = 1.2 * xb + 0.6 * Z
    omega_true = p / (1.0 - p)
    return dict(xt=xt, xb=xb, Z=Z, D=D, dY=dY, p=p, m=m_true, nu=nu_true, omega=omega_true)


class TestCCPSScore:
    def _score_mean(self, d, m=None, nu=None, ps=None, omega=None):
        m = d["m"] if m is None else m
        nu = d["nu"] if nu is None else nu
        ps = d["p"] if ps is None else ps
        omega = d["omega"] if omega is None else omega
        ps = np.clip(ps, 1e-3, 1 - 1e-3)
        return float(
            ccps_panel_score(d["dY"], d["D"], m, nu, ps, omega, float(d["D"].mean())).mean()
        )

    def test_score_mean_recovers_att_with_oracle_nuisances(self):
        d = _ccps_dgp()
        assert abs(self._score_mean(d) - CCPS_TAU) < 0.02

    def test_double_robustness_paired_outcome_side(self):
        # Proposition 6 (i): correct (m0, nu0) with WRONG (p, omega0) (an
        # internally consistent wrong pair: constant propensity, its odds).
        d = _ccps_dgp()
        ps_wrong = np.full_like(d["p"], 0.4)
        omega_wrong = np.full_like(d["p"], 0.4 / 0.6)
        assert abs(self._score_mean(d, ps=ps_wrong, omega=omega_wrong) - CCPS_TAU) < 0.02

    def test_double_robustness_paired_weighting_side(self):
        # Proposition 6 (ii): correct (p, omega0) with WRONG (m0, nu0), keeping
        # nu_wrong = E[m_wrong | S, D=0] (the pair stays functionally linked).
        d = _ccps_dgp()
        h = 0.8 * d["xb"] ** 2
        assert abs(self._score_mean(d, m=d["m"] + h, nu=d["nu"] + h) - CCPS_TAU) < 0.02

    def test_single_swaps_break_double_robustness(self):
        # The pairing is m0 <-> omega0 and nu0 <-> p (Lemma S2 / Lemma S3):
        # breaking BOTH members of a cancellation pair leaves an O(1) bias.
        d = _ccps_dgp()
        h = 0.8 * d["xb"] ** 2
        omega_const = np.full_like(d["p"], 0.2)
        bias_m_omega = self._score_mean(d, m=d["m"] + h, omega=omega_const) - CCPS_TAU
        assert abs(bias_m_omega) > 0.05, bias_m_omega
        ps_wrong = np.full_like(d["p"], 0.4)
        bias_nu_p = self._score_mean(d, nu=d["nu"] + h, ps=ps_wrong) - CCPS_TAU
        assert abs(bias_nu_p) > 0.05, bias_nu_p

    @pytest.mark.parametrize("direction", ["m", "nu", "p", "omega"])
    def test_neyman_orthogonality_vs_nonorthogonal_comparator(self, direction):
        """Gateaux derivative of E[score] is zero in each nuisance direction at
        the truth (Lemma S3's cross-terms), so the finite-sample shift under an
        eps-perturbation is Monte Carlo noise, while a purpose-built
        NON-orthogonal comparator shifts by O(eps). Orthogonal shift must be at
        least 20x smaller.
        """
        d = _ccps_dgp(n=400_000, seed=23)
        dY, D, xt, xb, Z = d["dY"], d["D"], d["xt"], d["xb"], d["Z"]
        m, nu, omega = d["m"], d["nu"], d["omega"]
        ps = np.clip(d["p"], 1e-3, 1 - 1e-3)
        p_hat = float(D.mean())
        eps = 0.05

        def score(m_=m, nu_=nu, ps_=ps, om_=omega):
            return float(ccps_panel_score(dY, D, m_, nu_, ps_, om_, p_hat).mean())

        base_orth = score()
        if direction == "m":
            # h is a function of R; its S-projection is E[h | S, D=0].
            h = 0.5 * xt + 0.3
            h_proj = 0.5 * (0.7 * xb + 0.3 * Z) + 0.3
            pert_orth = score(m_=m + eps * h)
            # Comparator: regression-adjustment plug-in mean_D(dY - nu) with nu
            # re-projected from the perturbed m (no reweighting terms).
            base_naive = float((D * (dY - nu) / p_hat).mean())
            pert_naive = float((D * (dY - (nu + eps * h_proj)) / p_hat).mean())
        elif direction == "nu":
            h = 0.5 * xb + 0.3
            pert_orth = score(nu_=nu + eps * h)
            base_naive = float((D * (dY - nu) / p_hat).mean())
            pert_naive = float((D * (dY - (nu + eps * h)) / p_hat).mean())
        elif direction == "p":
            h = 0.5 * xb + 0.3
            ps_pert = np.clip(ps + eps * h * ps * (1 - ps), 1e-3, 1 - 1e-3)
            pert_orth = score(ps_=ps_pert)

            def ipw(p_):  # Abadie IPW plug-in (no outcome adjustment)
                return float(((D - p_ * (1 - D) / (1 - p_)) * dY / p_hat).mean())

            base_naive = ipw(ps)
            pert_naive = ipw(ps_pert)
        else:
            h = 0.5 * xt + 0.3
            d_omega = eps * h
            pert_orth = score(om_=omega + d_omega)
            # Comparator: the omega term multiplies a NON-mean-zero residual
            # (a biased outcome regression m + c inside the omega term only),
            # so its omega-derivative is +c E[(1-D) d_omega]/p_hat != 0.
            c = 0.5

            def biased(om_):
                return score(om_=om_) + float(((1 - D) / p_hat * c * om_).mean())

            base_naive = biased(omega)
            pert_naive = biased(omega + d_omega)

        shift_orth = abs(pert_orth - base_orth)
        shift_naive = abs(pert_naive - base_naive)
        assert shift_orth < shift_naive / 20.0, (direction, shift_orth, shift_naive)

    def test_reduces_to_chang_score_without_bad_control(self):
        # nu = m and omega = ps/(1-ps): Equation 10 IS Chang's Case 1 score.
        d = _ccps_dgp(n=5000, seed=3)
        ps = np.clip(d["p"], 1e-3, 1 - 1e-3)
        p_hat = float(d["D"].mean())
        a = ccps_panel_score(d["dY"], d["D"], d["m"], d["m"], ps, ps / (1 - ps), p_hat)
        b = chang_panel_score(d["dY"], d["D"], d["m"], ps, p_hat)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)

    def test_augmented_step_is_chang_augmented(self):
        # phi_i = phi_1i - ATT - (ATT/pi)(D_i - pi) == summand - D theta / pi.
        d = _ccps_dgp(n=5000, seed=4)
        ps = np.clip(d["p"], 1e-3, 1 - 1e-3)
        p_hat = float(d["D"].mean())
        s = ccps_panel_score(d["dY"], d["D"], d["m"], d["nu"], ps, d["omega"], p_hat)
        theta = float(s.mean())
        aug = ccps_panel_score_augmented(s, d["D"], theta, p_hat)
        hand = s - theta - (theta / p_hat) * (d["D"] - p_hat)
        np.testing.assert_allclose(aug, hand, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(aug, chang_panel_score_augmented(s, d["D"], theta, p_hat))
        assert abs(aug.mean()) < 1e-12

    def test_input_validation(self):
        d = _ccps_dgp(n=50, seed=5)
        ps = np.clip(d["p"], 1e-3, 1 - 1e-3)
        with pytest.raises(ValueError, match="omega_hat must be non-negative"):
            ccps_panel_score(d["dY"], d["D"], d["m"], d["nu"], ps, -d["omega"], 0.5)
        with pytest.raises(ValueError, match="ps must lie in"):
            ccps_panel_score(d["dY"], d["D"], d["m"], d["nu"], np.ones(50), d["omega"], 0.5)
        with pytest.raises(ValueError, match="has length"):
            ccps_panel_score(d["dY"], d["D"], d["m"][:-1], d["nu"], ps, d["omega"], 0.5)
        with pytest.raises(ValueError, match="p_hat must satisfy"):
            ccps_panel_score(d["dY"], d["D"], d["m"], d["nu"], ps, d["omega"], 1.0)
