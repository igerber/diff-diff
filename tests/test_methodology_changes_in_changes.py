"""Methodology verification for ChangesInChanges (CiC) and QDiD.

Verifies the implementation against:
1. The canonical formulas of Athey & Imbens (2006), Econometrica 74(2),
   as documented in docs/methodology/papers/athey-imbens-2006-review.md and
   docs/methodology/REGISTRY.md (eqs. 34-36, 17-18, A.1-A.2; Lemma A.1).
2. Population-level relations the paper establishes: QDiD's mean effect
   coincides with standard DiD's ATT under continuity (p. 447); CiC and DiD
   probability limits coincide under the nested additive-linear model
   (p. 463); CiC is equivariant to monotone transformations of the outcome.
3. Hand-calculated micro-examples for every estimator formula.

R-package parity lives in test_changes_in_changes_parity.py.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import ChangesInChanges, DifferenceInDifferences, QDiD
from diff_diff.changes_in_changes import (
    _cic_point,
    _ecdf,
    _interior_range,
    _qdid_point,
    _quantile_type1,
    _split_cells,
)


def fit_quiet(est, df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(df, outcome="y", treatment="treated", time="post", **kwargs)


# =============================================================================
# Eq. (34)-(35)/(A.1)-(A.2) inverse-CDF conventions
# =============================================================================


class TestInverseCDFConventions:
    """The eq. (A.2) sandwich identities and Lemma A.1 Galois identities hold
    for the (_ecdf, _quantile_type1) pair - the load-bearing property of the
    paper's asymptotic theory, violated by interpolating quantile definitions
    (e.g. numpy's default 'linear').

    Exactness caveat: the identities are exact-arithmetic statements. In
    floating point they hold bit-exactly whenever the ECDF values k/n are
    binary-exact (n a power of two); for other n the float product n*(k/n)
    can land one ulp above k and shift the selection by one order statistic -
    R's quantile.default behaves identically (pinned by the bit-exact type-1
    micro-fixtures in test_changes_in_changes_parity.py), so the estimator
    still matches R exactly. Power-of-two sizes test exactness; a non-dyadic
    size tests the 1/n sandwich bound.
    """

    @pytest.fixture(params=[8, 16, 64, 256])
    def sample(self, request):
        rng = np.random.default_rng(request.param)
        return np.sort(rng.normal(0, 1, request.param))

    def test_sandwich_upper(self, sample):
        # q <= F(F^{-1}(q)) < q + 1/N, equality iff q = j/N.
        n = sample.size
        q = np.linspace(0.001, 0.999, 97)
        fq = _ecdf(sample, _quantile_type1(sample, q))
        assert np.all(fq >= q)
        assert np.all(fq < q + 1.0 / n + 1e-15)
        j_over_n = np.arange(0, n + 1) / n
        fq_exact = _ecdf(sample, _quantile_type1(sample, j_over_n[1:]))
        np.testing.assert_array_equal(fq_exact, j_over_n[1:])

    def test_sandwich_upper_non_dyadic(self):
        # Non-power-of-two n: the sandwich bound still holds within 1/n slack
        # on both sides (one-order-statistic float slippage, identical to R).
        rng = np.random.default_rng(53)
        sample = np.sort(rng.normal(0, 1, 53))
        n = sample.size
        q = np.linspace(0.001, 0.999, 197)
        fq = _ecdf(sample, _quantile_type1(sample, q))
        assert np.all(fq >= q - 1e-15)
        assert np.all(fq < q + 2.0 / n)

    def test_sandwich_lower_at_sample_points(self, sample):
        # F^{-1}(F(y)) == y exactly at all sample values (eq. A.2).
        np.testing.assert_array_equal(_quantile_type1(sample, _ecdf(sample, sample)), sample)

    def test_inverse_at_zero_is_minimum(self, sample):
        # eq. (35): F^{-1}(0) = sample minimum, never -inf.
        assert _quantile_type1(sample, np.array([0.0]))[0] == sample[0]

    def test_inverse_at_one_is_maximum(self, sample):
        assert _quantile_type1(sample, np.array([1.0]))[0] == sample[-1]

    def test_galois_idempotence(self, sample):
        # Lemma A.1 (iii)/(iv): g(g^{-1}(g(y))) = g(y); g^{-1}(g(g^{-1}(u))) = g^{-1}(u).
        # The identities are stated on the support Y (Lemma A.1's domain): below
        # the sample minimum, g(y) = 0 while g^{-1}(0) is the minimum under the
        # eq. (35) convention, so (iii) intentionally does not extend there.
        y = np.linspace(sample[0], sample[-1] + 1, 41)
        gy = _ecdf(sample, y)
        np.testing.assert_array_equal(_ecdf(sample, _quantile_type1(sample, gy)), gy)
        u = np.linspace(0.01, 0.99, 37)
        ginv_u = _quantile_type1(sample, u)
        np.testing.assert_array_equal(_quantile_type1(sample, _ecdf(sample, ginv_u)), ginv_u)


# =============================================================================
# Hand-calculated micro-example
# =============================================================================


class TestHandCalculated:
    """4-cell micro-example small enough to verify every number by hand.

    y00 = [1, 2, 3, 4]   (control pre)
    y01 = [2, 4, 6, 8]   (control post: doubling transformation h(u,1) = 2u)
    y10 = [2, 3]         (treated pre)
    y11 = [10, 12]       (treated post)

    ECDF ranks of y10 in y00: F00(2) = 2/4, F00(3) = 3/4.
    Counterfactual draws: Q1(y01, 0.50) = y01_(2) = 4; Q1(y01, 0.75) = y01_(3) = 6.
    CiC ATT = mean(10, 12) - mean(4, 6) = 11 - 5 = 6.
    """

    def setup_method(self):
        y = np.array([1, 2, 3, 4, 2, 4, 6, 8, 2, 3, 10, 12], dtype=float)
        g = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        t = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
        self.cells = _split_cells(y, g, t)

    def test_cic_att(self):
        att, _, cf = _cic_point(self.cells, np.array([0.5]))
        np.testing.assert_array_equal(cf, [4.0, 6.0])
        assert att == 6.0

    def test_cic_qte(self):
        # Q1(y11, 0.5) = 10 (first order stat, ceil(2*0.5)=1); Q1(cf, 0.5) = 4.
        _, qte, _ = _cic_point(self.cells, np.array([0.5]))
        assert qte[0] == 10.0 - 4.0

    def test_interior_range(self):
        # q_lower = F10(min y00) = F10(1) = 0; q_upper = F10(max y00) = F10(4) = 1.
        q_lower, q_upper = _interior_range(self.cells)
        assert q_lower == 0.0
        assert q_upper == 1.0

    def test_qdid_att(self):
        # Own-sample ranks of y10: F10(2) = 1/2, F10(3) = 1.
        # Type-7 quantiles: Q7(y01, [.5, 1]) = [5, 8]; Q7(y00, [.5, 1]) = [2.5, 4].
        # ATT = 11 - (2.5 + mean(5, 8) - mean(2.5, 4)) = 11 - (2.5 + 6.5 - 3.25) = 5.25.
        att, _ = _qdid_point(self.cells, np.array([0.5]))
        assert att == 11.0 - (2.5 + 6.5 - 3.25)

    def test_qdid_qte(self):
        # Q7 at 0.5: y11 -> 11, y10 -> 2.5, y01 -> 5, y00 -> 2.5.
        # qte(.5) = 11 - (2.5 + 5 - 2.5) = 6.
        _, qte = _qdid_point(self.cells, np.array([0.5]))
        assert qte[0] == 6.0


# =============================================================================
# Population-level relations from the paper
# =============================================================================


def make_additive_panel(n, seed, effect=1.5):
    """Additive-linear DGP: the standard DiD model (eqs. 1, 4-6), where CiC,
    QDiD, and DiD all identify the same ATT."""
    rng = np.random.default_rng(seed)
    treat = np.repeat([1, 0], [n // 2, n - n // 2])
    eps_pre = rng.normal(0, 1, n)
    eps_post = rng.normal(0, 1, n)
    gamma = 0.7 * treat
    y_pre = 0.2 + gamma + eps_pre
    y_post = 0.2 + 0.6 + gamma + eps_post + treat * effect
    return pd.DataFrame(
        {
            "post": np.repeat([0, 1], n),
            "treated": np.tile(treat, 2),
            "y": np.concatenate([y_pre, y_post]),
        }
    )


class TestPopulationRelations:
    def test_qdid_att_matches_did_at_large_n(self):
        # p. 447: E[k^DID(Y_10)] = E[k^QDID(Y_10)] under continuity - QDiD's
        # mean effect is standard DiD's ATT in population. Finite-sample
        # deviation comes from qte's rank-formula ATT (REGISTRY Note), so the
        # comparison uses a large sample and a loose tolerance.
        df = make_additive_panel(20000, seed=42)
        qdid = fit_quiet(QDiD(n_bootstrap=0), df)
        did = DifferenceInDifferences().fit(df, outcome="y", treatment="treated", post="post")
        assert qdid.att == pytest.approx(did.att, abs=0.05)

    def test_cic_matches_did_on_additive_dgp(self):
        # p. 463: under the nested linear model with full independence the CiC
        # and DiD probability limits coincide.
        df = make_additive_panel(20000, seed=7)
        cic = fit_quiet(ChangesInChanges(n_bootstrap=0), df)
        did = DifferenceInDifferences().fit(df, outcome="y", treatment="treated", post="post")
        assert cic.att == pytest.approx(did.att, abs=0.05)

    def test_cic_scale_invariance_nonlinear_dgp(self):
        # CiC's assumptions are invariant to monotone transformations (p. 437):
        # on a multiplicative DGP, CiC estimated in logs then exponentiated
        # differs from DiD in levels, but CiC handles both consistently. Here
        # we check the sharper finite-sample property: monotone-transform
        # equivariance of the counterfactual draws (interpolation-free
        # pipeline), via quantile effects of exp(y).
        rng = np.random.default_rng(3)
        n = 400
        treat = np.repeat([1, 0], n // 2)
        u = rng.normal(0, 0.5, n)
        y_pre = u + rng.normal(0, 0.1, n)
        y_post = 1.4 * u + 0.3 + rng.normal(0, 0.1, n) + treat * 0.5
        df = pd.DataFrame(
            {
                "post": np.repeat([0, 1], n),
                "treated": np.tile(treat, 2),
                "y": np.concatenate([y_pre, y_post]),
            }
        )
        grid = np.array([0.2, 0.5, 0.8])

        res_log = fit_quiet(ChangesInChanges(quantiles=grid, n_bootstrap=0), df)
        df_exp = df.assign(y=np.exp(df["y"]))
        res_exp = fit_quiet(ChangesInChanges(quantiles=grid, n_bootstrap=0), df_exp)

        # Equivariance: counterfactual and observed quantiles commute with exp
        # EXACTLY (both are order statistics / selections, never interpolated),
        # so exp-scale QTEs equal the exp of the log-scale quantile endpoints.
        cells = _split_cells(
            df["y"].to_numpy(),
            df["treated"].to_numpy(),
            df["post"].to_numpy(),
        )
        _, _, cf_log = _cic_point(cells, grid)
        q11_log = _quantile_type1(np.sort(cells["y11"]), grid)
        qcf_log = _quantile_type1(cf_log, grid)
        expected_exp_qte = np.exp(q11_log) - np.exp(qcf_log)
        np.testing.assert_array_equal(res_exp.quantile_effects["qte"].to_numpy(), expected_exp_qte)
        # And the log-scale run is internally consistent with the same endpoints.
        np.testing.assert_array_equal(res_log.quantile_effects["qte"].to_numpy(), q11_log - qcf_log)

    def test_cic_recovers_heterogeneous_quantile_effects(self):
        # Monotone-in-u heterogeneous effects: the QTE curve should be
        # increasing across quantiles and bracket the true effect range.
        rng = np.random.default_rng(11)
        n = 30000
        treat = np.repeat([1, 0], n // 2)
        u = rng.normal(0, 1, n)
        y_pre = u + rng.normal(0, 0.05, n)
        effect = 0.5 + 0.5 * (u > 0)  # 0.5 below median-u, 1.0 above
        y_post = u + 0.3 + rng.normal(0, 0.05, n) + treat * effect
        df = pd.DataFrame(
            {
                "post": np.repeat([0, 1], n),
                "treated": np.tile(treat, 2),
                "y": np.concatenate([y_pre, y_post]),
            }
        )
        res = fit_quiet(ChangesInChanges(quantiles=np.array([0.1, 0.9]), n_bootstrap=0), df)
        qte = res.quantile_effects["qte"].to_numpy()
        assert qte[0] == pytest.approx(0.5, abs=0.1)
        assert qte[1] == pytest.approx(1.0, abs=0.1)


# =============================================================================
# Interior range hand-check on constructed data
# =============================================================================


def test_interior_range_construction():
    # y00 spans [0, 1]; y10 has exactly 25 of 100 points below 0 and 25 above 1:
    # q_lower = F10(min y00) = F10(0) = 0.25, q_upper = F10(max y00) = F10(1) = 0.75.
    y00 = np.linspace(0, 1, 50)
    y10 = np.concatenate(
        [np.linspace(-1, -0.01, 25), np.linspace(0.01, 0.99, 50), np.linspace(1.01, 2, 25)]
    )
    y01 = np.linspace(0, 1, 50)
    y11 = np.linspace(0, 1, 50)
    y = np.concatenate([y00, y01, y10, y11])
    g = np.concatenate([np.zeros(100), np.ones(150)])
    t = np.concatenate([np.zeros(50), np.ones(50), np.zeros(100), np.ones(50)])
    cells = _split_cells(y, g, t)
    q_lower, q_upper = _interior_range(cells)
    assert q_lower == 0.25
    assert q_upper == 0.75


# =============================================================================
# Covariate path (qte xformla parity route - Melly-Santangelo QR pipeline)
# =============================================================================


class TestCovariateMethodology:
    """Methodology checks for the conditional (quantile-regression) path.

    Deliberate non-test: monotone-transform equivariance does NOT extend to
    the covariate branch (linear-in-covariates quantile regression is not
    equivariant to nonlinear monotone transforms of the outcome), so the
    unconditional equivariance test has no covariate analogue - documented in
    the REGISTRY covariate Note rather than asserted here.
    """

    @staticmethod
    def _make_shift_dgp(n_per_cell=299, seed=0, effect=1.0, shift=0.8, trend_slope=0.6):
        """Covariate-composition confounding: x distributions differ by group
        and the time trend depends on x, so unconditional CiC/QDiD are biased
        while the conditional versions recover the constant effect.

        Cell size is coprime to 100 (see the parity-test header): avoids QR
        vertex degeneracy so the test isn't sensitive to solver tie-breaking.
        """
        rng = np.random.default_rng(seed)
        frames = []
        for g in (0, 1):
            for t in (0, 1):
                x = rng.uniform(0, 2, n_per_cell) + shift * g
                u = rng.normal(0, 0.4, n_per_cell)
                # The x -> LEVEL link is deliberately weak (0.15) while the
                # x -> TREND link is strong: unconditional CiC absorbs any
                # time change expressible as a monotone transformation of the
                # outcome, so a trend strongly correlated with the outcome
                # level would be (correctly!) soaked up even without
                # covariates. Decoupling trend from level makes the
                # unconditional estimator genuinely biased while the
                # conditional one stays valid (within x, the period change is
                # an additive shift).
                y = 0.5 + 0.15 * x + trend_slope * x * t + u + effect * g * t
                frames.append(pd.DataFrame({"treated": g, "post": t, "x": x, "y": y}))
        return pd.concat(frames, ignore_index=True)

    @pytest.mark.parametrize("cls", [ChangesInChanges, QDiD], ids=["cic", "qdid"])
    def test_irrelevant_covariate_matches_unconditional(self, cls):
        # x independent of group, period, and outcome: the conditional
        # estimator adds only QR estimation noise and must agree with the
        # unconditional one at moderate N.
        rng = np.random.default_rng(1)
        n = 251  # coprime to 100
        frames = []
        for g in (0, 1):
            for t in (0, 1):
                y = rng.normal(0.3 * t + 0.5 * g * t, 1.0, n)
                frames.append(
                    pd.DataFrame({"treated": g, "post": t, "x": rng.uniform(0, 1, n), "y": y})
                )
        df = pd.concat(frames, ignore_index=True)
        r_cov = fit_quiet(cls(n_bootstrap=0), df, covariates=["x"])
        r_unc = fit_quiet(cls(n_bootstrap=0), df)
        assert r_cov.att == pytest.approx(r_unc.att, abs=0.1)
        # Loose atol: tail-quantile QR estimates are noisy at this N (the
        # 0.05/0.90 grid points drive the worst deviations); a composition
        # bug would err at O(0.5-1).
        np.testing.assert_allclose(
            r_cov.quantile_effects["qte"].to_numpy(),
            r_unc.quantile_effects["qte"].to_numpy(),
            atol=0.35,
        )

    @pytest.mark.parametrize("cls", [ChangesInChanges, QDiD], ids=["cic", "qdid"])
    def test_covariates_correct_compositional_confounding(self, cls):
        # The Melly-Santangelo motivation (their Figures 1-2 analysis): when
        # group covariate compositions differ AND trends vary with x, the
        # unconditional estimator is biased; conditioning on x restores the
        # constant effect. Bias magnitude ~ trend_slope * E[x_treated - x_control].
        df = self._make_shift_dgp(seed=2)
        r_cov = fit_quiet(cls(n_bootstrap=0), df, covariates=["x"])
        r_unc = fit_quiet(cls(n_bootstrap=0), df)
        # Measured across seeds: cov bias <= 0.14, unc bias >= 0.33.
        assert abs(r_cov.att - 1.0) < 0.2
        assert abs(r_unc.att - 1.0) > 0.25
        assert abs(r_cov.att - 1.0) < abs(r_unc.att - 1.0)

    def test_constant_covariate_reduces_to_order_statistics(self):
        # Hand-check via the constant-covariate reduction: with x constant,
        # each per-cell QR collapses to an intercept-only fit whose fitted
        # value at tau is the check-loss minimizer - the ceiling order
        # statistic y_(ceil(n*tau)) (unique here because no grid tau puts
        # n*tau on an integer when n = 7). From those known predictions the
        # whole CiC composition is hand-derivable with the module's own
        # step-function helpers, so the full fit must reproduce it.
        from diff_diff.changes_in_changes import (
            _QR_TAU_GRID,
            _design_matrix,
            _fhat_eval,
            _qhat_eval,
            _rq_fit,
        )

        rng = np.random.default_rng(3)
        n = 7
        cells_y = {
            (0, 0): np.sort(rng.normal(0.0, 1.0, n)),
            (0, 1): np.sort(rng.normal(0.5, 1.2, n)),
            (1, 0): np.sort(rng.normal(0.2, 0.9, n)),
            (1, 1): np.sort(rng.normal(1.1, 1.1, n)),
        }
        frames = [
            pd.DataFrame({"treated": g, "post": t, "x": 2.0, "y": y})
            for (g, t), y in cells_y.items()
        ]
        df = pd.concat(frames, ignore_index=True)

        # 1. The LP's fitted values equal the ceiling order statistics.
        x_const = np.full(n, 2.0)
        for y_cell in cells_y.values():
            coefs = _rq_fit(y_cell, x_const[:, None], _QR_TAU_GRID)
            assert coefs is not None
            fitted = _design_matrix(x_const[:1, None] * 0 + 2.0) @ coefs.T  # 1 x 99
            expected = np.array([y_cell[int(np.ceil(n * tau)) - 1] for tau in _QR_TAU_GRID])
            np.testing.assert_allclose(fitted[0], expected, atol=1e-7)

        # 2. End-to-end CiC equals the hand-derived composition.
        def order_stat_preds(y_cell):
            row = np.array([y_cell[int(np.ceil(n * tau)) - 1] for tau in _QR_TAU_GRID])
            return np.tile(row, (n, 1))

        y10 = cells_y[(1, 0)]
        ranks = _fhat_eval(order_stat_preds(cells_y[(0, 0)]), _QR_TAU_GRID, y10)
        y0t = _qhat_eval(order_stat_preds(cells_y[(0, 1)]), _QR_TAU_GRID, ranks)
        expected_att = float(np.mean(cells_y[(1, 1)]) - np.mean(y0t))

        res = fit_quiet(ChangesInChanges(n_bootstrap=0), df, covariates=["x"])
        assert res.att == pytest.approx(expected_att, abs=1e-6)

        # 3. QDiD analogue: own-cell ranks, additive imputation, and the
        #    asymmetric Q7/Q1 quantile-type pair ported from qte.
        ranks_q = _fhat_eval(order_stat_preds(y10), _QR_TAU_GRID, y10)
        y0t_q = (
            y10
            + _qhat_eval(order_stat_preds(cells_y[(0, 1)]), _QR_TAU_GRID, ranks_q)
            - _qhat_eval(order_stat_preds(cells_y[(0, 0)]), _QR_TAU_GRID, ranks_q)
        )
        expected_att_q = float(np.mean(cells_y[(1, 1)]) - np.mean(y0t_q))
        res_q = fit_quiet(QDiD(n_bootstrap=0), df, covariates=["x"])
        assert res_q.att == pytest.approx(expected_att_q, abs=1e-6)
