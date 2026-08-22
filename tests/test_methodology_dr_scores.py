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

from diff_diff._dr_scores import chang_panel_score, chang_panel_score_augmented

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
