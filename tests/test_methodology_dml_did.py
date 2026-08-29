"""DMLDiD methodology suite (PR-B1): ATT recovery, double robustness, oracle
equivalence, degenerate-cell bit-for-bit equivalence, DoubleML golden parity,
and (slow) Monte Carlo coverage.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import DMLDiD
from diff_diff._crossfit import assign_folds, cross_fit_predict
from diff_diff._dr_scores import (
    chang_panel_score,
    chang_panel_score_augmented,
    chang_rcs_score,
    chang_rcs_score_augmented,
)
from diff_diff._learners import LinearLearner, LogitLearner


def chang_2period_panel(n=500, theta=3.0, seed=42):
    """Chang (2020) §4.1.1-style DGP in long panel form (2 periods)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    g0 = 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))
    D = (rng.uniform(size=n) < g0).astype(float)
    ell0 = X[:, 0] + 0.5 * X[:, 2] ** 2
    y0 = rng.standard_normal(n)
    dY = ell0 + theta * D + rng.standard_normal(n)
    rows = []
    for i in range(n):
        for t, y in ((0, y0[i]), (1, y0[i] + dY[i])):
            rows.append((i, t, 1 if D[i] else 0, y, *X[i]))
    cols = ["unit", "time", "first_treat", "y", "x1", "x2", "x3", "x4", "x5"]
    return pd.DataFrame(rows, columns=cols), theta


XCOLS = ["x1", "x2", "x3", "x4", "x5"]
FIT_KW = dict(outcome="y", unit="unit", time="time", first_treat="first_treat")


class TestATTRecovery:
    def test_chang_dgp_theta_recovery(self):
        df, theta = chang_2period_panel(n=2000, seed=0)
        res = DMLDiD(outcome_learner="sieve", seed=0).fit(df, **FIT_KW, covariates=XCOLS)
        assert abs(res.att - theta) < 4 * res.se
        assert abs(res.att - theta) < 0.35

    def test_staggered_heterogeneous_effects_per_cell(self):
        # Cohort- and time-varying effects: each cell's estimate tracks its
        # own truth, not a pooled average.
        rng = np.random.default_rng(3)
        n_units, periods = 600, [1, 2, 3, 4]
        cohort = rng.choice([0, 3, 4], size=n_units, p=[0.5, 0.25, 0.25])
        X = rng.standard_normal((n_units, 2))
        effect = {(3, 3): 1.0, (3, 4): 2.0, (4, 4): 3.0}
        rows = []
        for i in range(n_units):
            for t in periods:
                y = 0.8 * X[i, 0] - 0.4 * X[i, 1] + 0.3 * t + rng.normal(scale=0.7)
                if cohort[i] > 0 and t >= cohort[i]:
                    y += effect[(cohort[i], t)]
                rows.append((i, t, cohort[i], y, X[i, 0], X[i, 1]))
        df = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "x2"])
        res = DMLDiD(seed=0).fit(df, **FIT_KW, covariates=["x1", "x2"])
        for (g, t), truth in effect.items():
            e = res.group_time_effects[(g, t)]
            assert abs(e["effect"] - truth) < 4 * e["se"], (g, t)

    def test_double_robustness_misspecified_propensity(self):
        # Non-logistic treatment assignment (threshold rule) misspecifies the
        # logit propensity; the DR score still recovers theta because the
        # outcome regression is correctly specified (linear in X).
        rng = np.random.default_rng(11)
        n, theta = 3000, 1.5
        X = rng.standard_normal((n, 2))
        D = ((X[:, 0] + 0.5 * rng.standard_normal(n)) > 0).astype(float)
        y0 = rng.standard_normal(n)
        dY = 1.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + theta * D + rng.standard_normal(n)
        rows = []
        for i in range(n):
            for t, y in ((0, y0[i]), (1, y0[i] + dY[i])):
                rows.append((i, t, int(D[i]), y, X[i, 0], X[i, 1]))
        df = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "x2"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # extreme-propensity warnings expected
            res = DMLDiD(seed=0).fit(df, **FIT_KW, covariates=["x1", "x2"])
        assert abs(res.att - theta) < 0.2


class TestOracleEquivalence:
    def _cell_arrays(self, seed=21, n=400):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 2))
        ps_true = 1 / (1 + np.exp(-X[:, 0]))
        D = (rng.uniform(size=n) < ps_true).astype(float)
        dY = X[:, 0] + 2.0 * D + rng.standard_normal(n)
        return X, D, dY, ps_true

    def test_oracle_learners_match_closed_form(self):
        # With ORACLE nuisances (true propensity, true control regression),
        # theta must equal the closed-form score mean and se the hand
        # augmented-score formula exactly.
        X, D, dY, ps_true = self._cell_arrays()
        m_true = X[:, 0]  # E[dY | X, D=0]

        class OracleProp:
            def fit(self, Xf, yf, sample_weight=None):
                return self

            def predict_proba(self, Xf):
                p = 1 / (1 + np.exp(-Xf[:, 0]))
                return np.column_stack([1 - p, p])

        class OracleReg:
            def fit(self, Xf, yf, sample_weight=None):
                return self

            def predict(self, Xf):
                return Xf[:, 0]

        n = len(D)
        rows = []
        y0 = np.zeros(n)
        for i in range(n):
            rows.append((i, 0, int(D[i]), y0[i], X[i, 0], X[i, 1]))
            rows.append((i, 1, int(D[i]), y0[i] + dY[i], X[i, 0], X[i, 1]))
        df = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "x2"])
        trim = 0.01
        res = DMLDiD(
            propensity_learner=OracleProp(),
            outcome_learner=OracleReg(),
            seed=0,
            pscore_trim=trim,
        ).fit(df, **FIT_KW, covariates=["x1", "x2"])
        ps = np.clip(ps_true, trim, 1 - trim)
        p_hat = D.mean()
        summand = chang_panel_score(dY, D, m_true, ps, p_hat)
        theta = float(np.mean(summand))
        psi_bar = chang_panel_score_augmented(summand, D, theta, p_hat)
        se = float(np.sqrt(np.mean(psi_bar**2) / n))
        np.testing.assert_allclose(res.att, theta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res.se, se, rtol=0, atol=1e-12)

    def test_two_period_degenerate_cell_equivalence(self):
        # The DMLDiD single-cell fit must equal the hand computation built
        # from the SAME spawned fold seed + cross_fit_predict + chang scores.
        # Tolerance 1e-14, not bit-for-bit: the hand pipeline materializes X
        # via pandas pivot/reindex while the estimator uses its own
        # precompute, and BLAS reduces differently-laid-out (but equal)
        # matrices in different orders per platform — ~1-ULP divergence on
        # Linux/Windows that macOS/Accelerate masks.
        df, _ = chang_2period_panel(n=300, seed=5)
        seed = 9
        res = DMLDiD(seed=seed).fit(df, **FIT_KW, covariates=XCOLS)
        # Hand computation. Cell = every unit (one cohort, one post period);
        # g_idx = t_idx = 0 under sorted rosters ([1], [0, 1] -> t=1 at
        # index... t roster is [0, 1]; the estimated cell is t=1 -> t_idx=1).
        unit_info = df.groupby("unit")["first_treat"].first()
        D = (unit_info.to_numpy() > 0).astype(float)
        wide = df.pivot(index="unit", columns="time", values="y").reindex(unit_info.index)
        dY = wide[1].to_numpy() - wide[0].to_numpy()
        X = df[df["time"] == 0].set_index("unit").reindex(unit_info.index)[XCOLS].to_numpy()
        n = len(D)
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(0, 1)))
        folds = assign_folds(n, 5, rng=rng, stratify=D)
        ps_res = cross_fit_predict(LogitLearner(), X, D, folds, predict_method="predict_proba")
        or_res = cross_fit_predict(
            LinearLearner(), X, dY, folds, predict_method="predict", fit_mask=(D == 0.0)
        )
        ps = np.clip(ps_res.oof_predictions, 0.01, 0.99)
        p_hat = D.mean()
        summand = chang_panel_score(dY, D, or_res.oof_predictions, ps, p_hat)
        theta = float(np.mean(summand))
        psi_bar = chang_panel_score_augmented(summand, D, theta, p_hat)
        se = float(np.sqrt(np.mean(psi_bar**2) / n))
        np.testing.assert_allclose(res.att, theta, rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(res.se, se, rtol=1e-14, atol=0.0)
        cells = [e for e in res.group_time_effects.values() if e["skip_reason"] is None]
        assert len(cells) == 1 and res.att == cells[0]["effect"]


class TestDoubleMLGoldenParity:
    """Golden literals from benchmarks/doubleml/chang_staggered_parity.py
    (doubleml 0.11.4, sklearn 1.9.0, seed 7): DoubleMLDIDBinary per cell
    under shared folds and pinned config (score="observational",
    in_sample_normalization=False, trimming 0.01). The native reproduction
    swaps sklearn's lbfgs logit for the library IRLS solver — tolerance
    atol 2e-4 (ATT) / 1e-5 (SE), the measured optimizer gap (B0 precedent).
    """

    GOLDEN = {
        (3, 3, 2): (2.030643126019, 0.173107723124),
        (3, 4, 2): (2.268834143564, 0.187095112140),
        (4, 3, 2): (-0.177672235406, 0.167181769564),
        (4, 4, 3): (2.139119845409, 0.186007089606),
    }

    def test_native_reproduction_matches_golden(self):
        # Reproduce the spike's DGP + shared folds exactly, then swap in the
        # library learners for the nuisances.
        rng = np.random.default_rng(7)
        n_units, periods = 400, [1, 2, 3, 4]
        cohort = rng.choice([0, 3, 4], size=n_units, p=[0.5, 0.25, 0.25])
        X_unit = rng.standard_normal((n_units, 2))
        y = np.empty((n_units, len(periods)))
        for i in range(n_units):
            for j, t in enumerate(periods):
                val = (
                    0.8 * X_unit[i, 0]
                    - 0.4 * X_unit[i, 1]
                    + 0.3 * t
                    + 0.5 * X_unit[i, 0] * t / 4
                    + rng.standard_normal()
                )
                if cohort[i] > 0 and t >= cohort[i]:
                    val += 2.0 + 0.2 * (t - cohort[i])
                y[i, j] = val
        col = {t: j for j, t in enumerate(periods)}
        for (g, t_eval, base), (att_gold, se_gold) in self.GOLDEN.items():
            idx = np.flatnonzero((cohort == g) | (cohort == 0))
            dY = y[idx, col[t_eval]] - y[idx, col[base]]
            D = (cohort[idx] == g).astype(float)
            X = X_unit[idx]
            n = idx.shape[0]
            cell_rng = np.random.default_rng(1000 + g * 10 + t_eval)
            perm = cell_rng.permutation(n)
            test_folds = [np.sort(perm[k::5]) for k in range(5)]
            smpls = [(np.setdiff1d(np.arange(n), te), te) for te in test_folds]
            oof_g = np.empty(n)
            oof_l = np.empty(n)
            for tr, te in smpls:
                lg = LogitLearner().fit(X[tr], D[tr])
                oof_g[te] = lg.predict_proba(X[te])[:, 1]
                ctrl = tr[D[tr] == 0]
                oof_l[te] = LinearLearner().fit(X[ctrl], dY[ctrl]).predict(X[te])
            ps = np.clip(oof_g, 0.01, 0.99)
            p_hat = D.mean()
            summand = chang_panel_score(dY, D, oof_l, ps, p_hat)
            theta = float(np.mean(summand))
            psi_bar = chang_panel_score_augmented(summand, D, theta, p_hat)
            se = float(np.sqrt(np.mean(psi_bar**2) / n))
            np.testing.assert_allclose(theta, att_gold, atol=2e-4, rtol=0)
            np.testing.assert_allclose(se, se_gold, atol=1e-5, rtol=0)


@pytest.mark.slow
class TestMonteCarloCoverage:
    def test_coverage_sanity(self, ci_params):
        # Sanity check, not a calibration claim: nominal 95% CI on the Chang
        # DGP. Acceptance band CONDITIONAL on the scaled rep count (at ~22
        # reps the tight band fails ~42% of the time at nominal coverage).
        n_reps = ci_params.bootstrap(200)
        band = (0.90, 0.99) if n_reps >= 100 else (0.80, 1.00)
        theta = 3.0
        hits = 0
        for rep in range(n_reps):
            df, _ = chang_2period_panel(n=500, theta=theta, seed=10_000 + rep)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = DMLDiD(seed=0).fit(df, **FIT_KW, covariates=XCOLS)
            lo, hi = res.overall_conf_int
            hits += int(lo <= theta <= hi)
        coverage = hits / n_reps
        assert band[0] <= coverage <= band[1], f"coverage {coverage:.3f} outside {band}"


class TestDoubleMLEndToEndGoldenParity:
    """Part-2 golden literals from benchmarks/doubleml/chang_staggered_parity.py:
    the PUBLIC ``DMLDiD.fit()`` (sklearn learner objects, seed=11) vs
    ``DoubleMLDIDBinary`` under DMLDiD's OWN reconstructed fold assignments —
    machine-precision in the spike. Here the same public fit runs with the
    NATIVE learners (IRLS logit instead of sklearn lbfgs), so tolerance is
    atol 2e-4 (ATT) / 1e-5 (SE), the measured optimizer gap. This covers
    public cell construction, base-period selection, IF scattering, and
    result wiring end to end.
    """

    GOLDEN = {
        (3, 3): (2.013640496612, 0.173611909306),
        (3, 4): (2.312506769326, 0.182958385142),
        (4, 3): (-0.153758273010, 0.164671337301),
        (4, 4): (2.157511261169, 0.191218867463),
    }

    def test_public_fit_matches_end_to_end_golden(self):
        rng = np.random.default_rng(7)
        n_units, periods = 400, [1, 2, 3, 4]
        cohort = rng.choice([0, 3, 4], size=n_units, p=[0.5, 0.25, 0.25])
        X_unit = rng.standard_normal((n_units, 2))
        rows = []
        for i in range(n_units):
            for t in periods:
                val = (
                    0.8 * X_unit[i, 0]
                    - 0.4 * X_unit[i, 1]
                    + 0.3 * t
                    + 0.5 * X_unit[i, 0] * t / 4
                    + rng.standard_normal()
                )
                if cohort[i] > 0 and t >= cohort[i]:
                    val += 2.0 + 0.2 * (t - cohort[i])
                rows.append((i, t, cohort[i], val, X_unit[i, 0], X_unit[i, 1]))
        df = pd.DataFrame(rows, columns=["id", "t", "g", "y", "x1", "x2"])
        res = DMLDiD(seed=11, n_folds=5, pscore_trim=0.01).fit(
            df, outcome="y", unit="id", time="t", first_treat="g", covariates=["x1", "x2"]
        )
        for (g, t), (att_gold, se_gold) in self.GOLDEN.items():
            cell = res.group_time_effects[(g, t)]
            np.testing.assert_allclose(cell["effect"], att_gold, atol=2e-4, rtol=0)
            np.testing.assert_allclose(cell["se"], se_gold, atol=1e-5, rtol=0)


class TestDoubleRobustnessConverse:
    def test_misspecified_outcome_correct_propensity(self):
        # Converse DR lane: the outcome-change regression is misspecified
        # (linear learner on a quadratic ell0), the propensity is correctly
        # specified (logistic in X) — the DR score still recovers theta.
        rng = np.random.default_rng(23)
        n, theta = 3000, 1.5
        X = rng.standard_normal((n, 2))
        g0 = 1 / (1 + np.exp(-(0.8 * X[:, 0])))
        D = (rng.uniform(size=n) < g0).astype(float)
        y0 = rng.standard_normal(n)
        dY = 1.0 + X[:, 0] ** 2 - 0.5 * X[:, 1] + theta * D + rng.standard_normal(n)
        rows = []
        for i in range(n):
            for t, y in ((0, y0[i]), (1, y0[i] + dY[i])):
                rows.append((i, t, int(D[i]), y, X[i, 0], X[i, 1]))
        df = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "x2"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(outcome_learner="linear", seed=0).fit(
                df, **FIT_KW, covariates=["x1", "x2"]
            )
        assert abs(res.att - theta) < 0.25


# ===========================================================================
# Repeated cross sections (Chang Case 2, panel=False)
# ===========================================================================
#
# Shared fixed RCS DGP (library-authored, in the spirit of Chang Sec. 4 —
# NOT the paper's own Sec. 4 RCS parameterization; the paper's Sec. 4.2.2
# kernel-design DGP is replicated in the "Chang Sec. 4.2.2" section below,
# and the Sec. 4.2.1 ML design is tracked in the narrowed TODO.md row):
# X ~ N(0, I_2), D ~ Bernoulli(sigmoid(0.5 X1 - 0.5 X2)), T ~
# Bernoulli(0.5), levels
# Y = 1 + X1 + 0.5 X2 + T*(0.5 + 0.4 X1) + D + T*D*theta0 + eps.

RCS_THETA0 = 3.0


def _rcs_frame(n_rows, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, 2))
    g0 = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.5 * X[:, 1])))
    D = (rng.uniform(size=n_rows) < g0).astype(int)
    T = (rng.uniform(size=n_rows) < 0.5).astype(int)
    trend = 0.5 + 0.4 * X[:, 0]
    y = (
        1.0
        + X[:, 0]
        + 0.5 * X[:, 1]
        + T * trend
        + D * 1.0
        + T * D * RCS_THETA0
        + rng.normal(size=n_rows)
    )
    return pd.DataFrame(
        {
            "unit": np.arange(n_rows),
            "time": T + 1,  # periods {1, 2}; cohort 2 = treated at period 2
            "first_treat": D * 2,
            "y": y,
            "x1": X[:, 0],
            "x2": X[:, 1],
        }
    )


class TestRCSOracleEquivalence:
    def test_oracle_learners_match_closed_form(self):
        # ORACLE nuisances on BOTH sides: the true DGP propensity and the
        # true l20(X) = E[(T - lam0)Y | X, D=0] = 0.25*(0.4 + 0.3*x1) at
        # lam0 = 0.5 (T independent of (X, D), so the x1 and eps terms are
        # killed by E[T - lam0] = 0 and E[(T - lam0)T] = 0.25 survives).
        # The public DMLDiD(panel=False) single-cell fit must equal the
        # hand Eq 3.2 / Thm 2 pipeline at 1e-12 (algebra equivalence —
        # exact for ANY fixed nuisance; using the true one keeps "oracle"
        # honest).
        rng = np.random.default_rng(31)
        n = 600
        X = rng.standard_normal((n, 2))
        ps_true = 1 / (1 + np.exp(-X[:, 0]))
        D = (rng.uniform(size=n) < ps_true).astype(float)
        T = (rng.uniform(size=n) < 0.5).astype(float)
        y = X[:, 0] + T * (0.4 + 0.3 * X[:, 0]) + 2.0 * T * D + rng.standard_normal(n)
        df = pd.DataFrame(
            {
                "unit": np.arange(n),
                "time": T.astype(int) + 1,
                "first_treat": (D * 2).astype(int),
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
            }
        )

        class OracleProp:
            def fit(self, Xf, yf, sample_weight=None):
                return self

            def predict_proba(self, Xf):
                p = 1 / (1 + np.exp(-Xf[:, 0]))
                return np.column_stack([1 - p, p])

        class OracleReg:
            def fit(self, Xf, yf, sample_weight=None):
                return self

            def predict(self, Xf):
                return 0.1 + 0.075 * Xf[:, 0]

        trim = 0.01
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(
                propensity_learner=OracleProp(),
                outcome_learner=OracleReg(),
                seed=0,
                pscore_trim=trim,
                panel=False,
            ).fit(df, **FIT_KW, covariates=["x1", "x2"])
        ps = np.clip(ps_true, trim, 1 - trim)
        m2 = 0.1 + 0.075 * X[:, 0]
        p_hat = float(D.mean())
        lam = float(T.mean())
        summand = chang_rcs_score(y, D, T, m2, ps, p_hat, lam)
        theta = float(np.mean(summand))
        psi_bar = chang_rcs_score_augmented(summand, D, T, y, m2, ps, theta, p_hat, lam)
        se = float(np.sqrt(np.mean(psi_bar**2) / n))
        np.testing.assert_allclose(res.att, theta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res.se, se, rtol=0, atol=1e-12)

    def test_two_period_cell_equivalence_vs_hand_pipeline(self):
        # Real (non-oracle) learners: the fit equals a hand pipeline that
        # replays the SAME spawned folds + cross_fit_predict + rcs scores.
        # Tolerance 1e-14 rtol, not bit-for-bit (BLAS layout lesson, B1 CI).
        from diff_diff._crossfit import assign_folds, cross_fit_predict
        from diff_diff._learners import LinearLearner, LogitLearner

        df = _rcs_frame(800, seed=33)
        seed = 9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=seed, panel=False).fit(df, **FIT_KW, covariates=["x1", "x2"])
        D = df["first_treat"].to_numpy().astype(float) / 2.0
        T = (df["time"].to_numpy() == 2).astype(float)
        y = df["y"].to_numpy()
        X = df[["x1", "x2"]].to_numpy()
        n = len(df)
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(0, 1)))
        folds = assign_folds(n, 5, rng=rng, stratify=D + 2.0 * T)
        ps_res = cross_fit_predict(LogitLearner(), X, D, folds, predict_method="predict_proba")
        lam = float(T.mean())
        r = (T - lam) * y
        or_res = cross_fit_predict(
            LinearLearner(), X, r, folds, predict_method="predict", fit_mask=(D == 0.0)
        )
        ps = np.clip(ps_res.oof_predictions, 0.01, 0.99)
        p_hat = float(D.mean())
        summand = chang_rcs_score(y, D, T, or_res.oof_predictions, ps, p_hat, lam)
        theta = float(np.mean(summand))
        psi_bar = chang_rcs_score_augmented(
            summand, D, T, y, or_res.oof_predictions, ps, theta, p_hat, lam
        )
        se = float(np.sqrt(np.mean(psi_bar**2) / n))
        np.testing.assert_allclose(res.att, theta, rtol=1e-14, atol=0)
        np.testing.assert_allclose(res.se, se, rtol=1e-14, atol=0)


class TestLambdaCorrectionRegression:
    def test_reported_se_carries_the_lambda_term(self):
        # Headline guard for the review's "plausible implementation bug":
        # from the fit's payload + g2_lambda diagnostic, recompute the SE
        # with the lambda term REMOVED and assert it differs from the
        # reported SE in the pinned direction.
        df = _rcs_frame(2000, seed=35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=0, panel=False).fit(df, **FIT_KW, covariates=["x1", "x2"])
        (gt_key,) = [k for k, e in res.group_time_effects.items() if e["skip_reason"] is None]
        entry = res.group_time_effects[gt_key]
        diag = res.cross_fit_diagnostics[gt_key]
        g2 = diag["g2_lambda"]
        assert np.isfinite(g2) and g2 != 0.0
        kit = res._aggregation_kit
        ii = kit.influence[gt_key]
        idx = np.concatenate([ii["treated_idx"], ii["control_idx"]])
        phi = np.concatenate([ii["treated_inf"], ii["control_inf"]])
        n_cell = len(idx)
        psi_bar = phi * n_cell  # payload entries are psi_bar / n_cell
        T_cell = (df["time"].to_numpy()[idx] == 2).astype(float)
        lam = diag["lam_hat"]
        psi_no_lambda = psi_bar - g2 * (T_cell - lam)
        se_no_lambda = float(np.sqrt(np.mean(psi_no_lambda**2) / n_cell))
        np.testing.assert_allclose(
            entry["se"], float(np.sqrt(np.mean(psi_bar**2) / n_cell)), rtol=1e-12
        )
        rel_gap = abs(se_no_lambda - entry["se"]) / entry["se"]
        assert rel_gap > 1e-5, (se_no_lambda, entry["se"])


class TestRCSATTRecovery:
    def test_theta_recovery(self):
        df = _rcs_frame(4000, seed=37)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=0, panel=False).fit(df, **FIT_KW, covariates=["x1", "x2"])
        assert abs(res.att - RCS_THETA0) < 0.3

    def test_double_robustness_misspecified_propensity(self):
        # Threshold (non-logistic) treatment rule misspecifies the logit
        # propensity; correct-in-X outcome regression still recovers theta.
        rng = np.random.default_rng(39)
        n = 5000
        X = rng.standard_normal((n, 2))
        # Noise scale 1.5 keeps the misspecified fitted propensities away
        # from the clip bounds — the RCS weight carries a 1/(lam(1-lam))
        # amplification on level outcomes, so a near-deterministic rule
        # turns clipping bias into the dominant finite-sample error.
        D = ((X[:, 0] + 1.5 * rng.standard_normal(n)) > 0).astype(int)
        T = (rng.uniform(size=n) < 0.5).astype(int)
        y = (
            1.0
            + X[:, 0]
            + 0.5 * X[:, 1]
            + T * (0.5 + 0.4 * X[:, 0])
            + D * 1.0
            + T * D * RCS_THETA0
            + rng.normal(size=n)
        )
        df = pd.DataFrame(
            {
                "unit": np.arange(n),
                "time": T + 1,
                "first_treat": D * 2,
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=0, panel=False).fit(df, **FIT_KW, covariates=["x1", "x2"])
        assert abs(res.att - RCS_THETA0) < 0.3

    def test_double_robustness_misspecified_outcome_regression(self):
        # Nonlinear-in-X outcome trend misspecifies the linear l2 learner;
        # the correctly-specified (logistic) propensity still recovers theta.
        rng = np.random.default_rng(41)
        n = 5000
        X = rng.standard_normal((n, 2))
        g0 = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.5 * X[:, 1])))
        D = (rng.uniform(size=n) < g0).astype(int)
        T = (rng.uniform(size=n) < 0.5).astype(int)
        y = (
            1.0
            + X[:, 0]
            + 0.5 * X[:, 1] ** 2  # nonlinear: linear learner misspecified
            + T * (0.5 + 0.4 * X[:, 0] ** 2)
            + D * 1.0
            + T * D * RCS_THETA0
            + rng.normal(size=n)
        )
        df = pd.DataFrame(
            {
                "unit": np.arange(n),
                "time": T + 1,
                "first_treat": D * 2,
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=0, panel=False).fit(df, **FIT_KW, covariates=["x1", "x2"])
        assert abs(res.att - RCS_THETA0) < 0.3


def _characterization_spike_frame():
    rng = np.random.default_rng(7)
    n_rows = 4000
    cohort = rng.choice([0, 3, 4], size=n_rows, p=[0.5, 0.25, 0.25])
    tt = rng.choice([1, 2, 3, 4], size=n_rows)
    X_row = rng.standard_normal((n_rows, 2))
    y = (
        0.8 * X_row[:, 0]
        - 0.4 * X_row[:, 1]
        + 0.3 * tt
        + 0.5 * X_row[:, 0] * tt / 4
        + rng.standard_normal(n_rows)
    )
    post = (cohort > 0) & (tt >= cohort)
    y = y + post * (2.0 + 0.2 * (tt - cohort))
    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "t": tt,
            "g": cohort,
            "y": y,
            "x1": X_row[:, 0],
            "x2": X_row[:, 1],
        }
    )


@pytest.fixture(scope="module")
def characterization_spike_fit():
    df = _characterization_spike_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DMLDiD(seed=11, panel=False).fit(
            df, outcome="y", unit="id", time="t", first_treat="g", covariates=["x1", "x2"]
        )


class TestRCSGoldenCharacterization:
    """Golden literals from benchmarks/doubleml/chang_rcs_characterization.py
    (doubleml 0.11.4, sklearn 1.9.0, SEED=7/SEED_DML=11). Part 2's CHANG-side
    values (public DMLDiD(panel=False) == hand pipeline at 0.0 diff under
    sklearn learners); the native reproduction swaps sklearn's lbfgs logit
    for the library IRLS solver — tolerance atol 2e-4 (ATT) / 1e-5 (SE), the
    B0/B1 optimizer-gap precedent. The Part 1 Chang-vs-DoubleML gaps are
    additionally pinned NONZERO by recomputing them from a LIVE native fit
    against the committed DoubleML ATT literals (characterization honesty:
    the two scores must not coincide, and a native regression onto
    DoubleML's score fails here).
    """

    GOLDEN = {
        (3, 3, 2): (1.867260790882, 0.244040011648),
        (3, 4, 2): (2.347579854514, 0.251882814022),
        (4, 3, 2): (-0.089752999600, 0.187258740736),
        (4, 4, 3): (1.822066292980, 0.260472487961),
    }
    # Part 1 DoubleMLDIDCSBinary ATT literals (shared folds; spike
    # transcript). The gap test recomputes CHANG - DoubleML from a live
    # native fit, so it fails if the native estimator ever drifts onto
    # DoubleML's Sant'Anna-Zhao score.
    DML_ATT = {
        (3, 3, 2): 2.121300323064,
        (3, 4, 2): 2.293055793734,
        (4, 3, 2): -0.057963388953,
        (4, 4, 3): 1.995045506760,
    }

    def test_native_reproduction_matches_golden(self, characterization_spike_fit):
        for (g, t_eval, _base), (att_gold, se_gold) in self.GOLDEN.items():
            e = characterization_spike_fit.group_time_effects[(g, t_eval)]
            np.testing.assert_allclose(e["effect"], att_gold, rtol=0, atol=2e-4)
            np.testing.assert_allclose(e["se"], se_gold, rtol=0, atol=1e-5)

    def test_characterization_gaps_are_nonzero(self, characterization_spike_fit):
        # Honesty pin, recomputed LIVE: the native Chang ATT minus the
        # committed DoubleML ATT must stay nonzero and bounded — a
        # near-zero gap would mean the scores coincide (they must not;
        # DoubleMLDIDCSBinary is NOT an oracle for Eq 3.2). Native-vs-spike
        # optimizer drift is atol 2e-4, far below the 1e-3 floor; the
        # smallest recorded gap is 1.73e-2.
        for (g, t_eval, _base), dml_att in self.DML_ATT.items():
            gap = characterization_spike_fit.group_time_effects[(g, t_eval)]["effect"] - dml_att
            assert 1e-3 < abs(gap) < 0.5


@pytest.mark.slow
class TestRCSMonteCarloCoverage:
    def test_coverage_sanity(self, ci_params):
        n_reps = ci_params.bootstrap(200)
        n = 800
        hits = 0
        for rep in range(n_reps):
            df = _rcs_frame(n, seed=10_000 + rep)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = DMLDiD(seed=rep, panel=False).fit(df, **FIT_KW, covariates=["x1", "x2"])
            lo, hi = res.conf_int
            hits += int(lo <= RCS_THETA0 <= hi)
        coverage = hits / n_reps
        lo_band, hi_band = (0.90, 0.99) if n_reps >= 100 else (0.80, 1.00)
        assert lo_band <= coverage <= hi_band, coverage


# ===========================================================================
# Chang Sec. 4.2.2 kernel-design RCS DGP (paper replication, DML PR-B2)
# ===========================================================================
#
# The paper's OWN repeated-cross-section simulation design (arXiv:1812.10846v3
# p. 19, Sec. 4.2.2 "Kernel Estimation"), replicated verbatim: D ~
# Bernoulli(0.5), scalar X | D ~ N(D, 1), Y0(0) = e1, Y0(1) = Y0(0) + X + e2,
# Y1(1) = theta0 + Y0(1) + e3, theta0 = 3, T ~ Bernoulli(0.5), observed
# Y = Y(0) + T*(Y(1) - Y(0)). "N(0, 0.1)" is read as VARIANCE 0.1
# (sigma = sqrt(0.1)); the printed notation is ambiguous and the adopted
# reading is recorded in docs/methodology/papers/chang-2020-review.md.
#
# The design's trend (X + e2) with group-imbalanced X | D ~ N(D, 1) violates
# UNCONDITIONAL parallel trends by construction: the unadjusted 2x2 group-mean
# DiD converges to theta0 + (E[X|D=1] - E[X|D=0]) = 4, which is what makes it
# a genuine covariate-adjustment fixture. Under our native learners both
# nuisances are correctly specified (true propensity = sigmoid(X - 1/2) by
# Bayes' rule; true outcome nuisance l20(X) = E[(T - lam0) Y | X, D=0] =
# lam0(1 - lam0) X = 0.25 X, from the identity E[(T - lam0) T] =
# lam0(1 - lam0) — the l20 definition is in diff_diff/_dr_scores.py), so this
# section adds paper-DGP faithfulness, not new discriminating power vs the
# incumbent _rcs_frame; the discriminating assertion below compares against
# the design's own confounded contrast instead. The paper estimates this
# design with Gaussian-KERNEL first stages; the fixtures use the library's
# native learners (documented REGISTRY Note). The Sec. 4.2.1 ML design is
# NOT replicable with the bundled unpenalized learners (narrowed TODO.md row).
# The paper publishes histograms only, so every assertion is recovery /
# centering / self-coverage — never a published-number pin.

CHANG_S422_THETA0 = 3.0


def chang_s422_kernel_frame(n, seed):
    """Sec. 4.2.2 verbatim; the EXACT draw order below is load-bearing (the
    seed-pinned tolerances in TestChangS422Recovery were measured under it)."""
    rng = np.random.default_rng(seed)
    D = (rng.uniform(size=n) < 0.5).astype(int)
    X = D + rng.standard_normal(n)  # X | D ~ N(D, 1)
    e1, e2, e3 = rng.normal(0, np.sqrt(0.1), (3, n))
    y00 = e1
    y01 = y00 + X + e2
    y11 = CHANG_S422_THETA0 + y01 + e3
    T = (rng.uniform(size=n) < 0.5).astype(int)
    y = np.where(T == 1, np.where(D == 1, y11, y01), y00)
    return pd.DataFrame(
        {
            "unit": np.arange(n),
            "time": T + 1,  # waves {1, 2}; cohort 2 = treated at period 2
            "first_treat": D * 2,
            "y": y,
            "x1": X,
        }
    )


def _fit_s422_expecting_only_a23(est, df):
    """Fit recording ALL warnings; tolerate only the deliberate panel=False
    Assumption 2.3 lane warning and the occasional propensity-trimming
    notice (the design's unbounded-X tails put rare fitted propensities
    outside [trim, 1-trim] — the documented overlap quirk; fires on some MC
    replicates, not at the pinned recovery seeds). Any OTHER warning
    (learner, degenerate cell, inference) fails the fixture instead of
    being blanket-suppressed (review round: simplefilter("ignore") weakened
    the regression guard)."""
    tolerated = ("Assumption 2.3", "will be trimmed")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = est.fit(df, **FIT_KW, covariates=["x1"])
    unexpected = [
        f"{w.category.__name__}: {w.message}"
        for w in caught
        # both tolerated messages are documented UserWarnings - a matching
        # text under a different category is still unexpected
        if not (w.category is UserWarning and any(pat in str(w.message) for pat in tolerated))
    ]
    assert not unexpected, f"unexpected warnings: {unexpected}"
    return res


def _unadjusted_did(df):
    """The covariate-blind 2x2 group-mean DiD contrast on a Sec.-4.2.2 frame
    (shared by the shape pin and every recovery test's discriminating
    comparison). Converges to theta0 + 1 on this design."""
    T = (df["time"] == 2).to_numpy()
    D = (df["first_treat"] > 0).to_numpy()
    m = lambda mask: float(df.loc[mask, "y"].mean())  # noqa: E731
    return (m(T & D) - m(~T & D)) - (m(T & ~D) - m(~T & ~D))


class TestChangS422FrameShape:
    """Pin the generator to the paper's distributions: recovery alone passes
    under plausible mis-codings (dropped mean shift, dropped trend, SD-vs-
    variance misread), so the defining moments are asserted directly."""

    def test_frame_matches_paper_dgp(self):
        df = chang_s422_kernel_frame(200_000, seed=40_000)
        x = df["x1"].to_numpy()
        T = (df["time"] == 2).to_numpy()
        D = (df["first_treat"] > 0).to_numpy()
        y = df["y"].to_numpy()

        # X | D ~ N(D, 1); shares 0.5
        assert abs(x[D].mean() - 1.0) < 0.02 and abs(x[~D].mean() - 0.0) < 0.02
        assert abs(x[D].var() - 1.0) < 0.03 and abs(x[~D].var() - 1.0) < 0.03
        assert abs(D.mean() - 0.5) < 0.01 and abs(T.mean() - 0.5) < 0.01

        # Stationary RCS sampling (Assumption 2.3): the wave draw is
        # independent of (D, X) — composition stable across waves.
        assert abs(T[D].mean() - T[~D].mean()) < 0.01
        assert abs(x[T].mean() - x[~T].mean()) < 0.02
        assert abs(np.corrcoef(T.astype(float), x)[0, 1]) < 0.01

        # Control-post regression on X: slope 1, residual var = Var(e1+e2) = 0.2
        mask = T & ~D
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        resid = y[mask] - (slope * x[mask] + intercept)
        assert abs(slope - 1.0) < 0.03
        assert abs(resid.var() - 0.2) < 0.01

        # Pre-wave outcome is pure e1: Var = 0.1
        assert abs(y[~T].var() - 0.1) < 0.005

        # Treated-post residual (OLS of y[T=1,D=1] on x1) = Var(e1+e2+e3) = 0.3
        mask = T & D
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        resid = y[mask] - (slope * x[mask] + intercept)
        assert abs(resid.var() - 0.3) < 0.015

        # The design's built-in confounding: the UNADJUSTED contrast converges
        # to theta0 + (E[X|D=1] - E[X|D=0]) = 4, NOT theta0.
        assert abs(_unadjusted_did(df) - 4.0) < 0.05

        # Correct-specification facts the recovery narrative rests on:
        # true propensity = sigmoid(X - 1/2) (logistic in X)...
        logit = LogitLearner().fit(x.reshape(-1, 1), D.astype(float))
        assert abs(logit.intercept_ - (-0.5)) < 0.05
        assert abs(float(np.asarray(logit.coef_).ravel()[0]) - 1.0) < 0.05
        # ...and l20(X) = 0.25 X (OLS of (T - 0.5) y on x1 over controls).
        mask = ~D
        slope, _ = np.polyfit(x[mask], (T[mask] - 0.5) * y[mask], 1)
        assert abs(slope - 0.25) < 0.02


class TestChangS422Recovery:
    """Seed-pinned recovery on the paper's design. The absolute bounds are
    SEED-PINS at the named data seeds (measured margins 0.242 / 0.095 /
    0.442), NOT distributional bounds (~0.5% of N=500 replicates exceed
    1.25); a failure under the exact chang_s422_kernel_frame draw order is a
    frame-implementation discrepancy, not a methodology signal — do not
    retune. Each test also asserts the DISCRIMINATING comparison: the
    adjusted estimate must land strictly closer to theta0 than the design's
    own confounded unadjusted contrast (4.008 / 3.978 / 3.617 at these
    seeds), which a covariate-blind estimator cannot do."""

    @pytest.mark.parametrize(
        "n,learner,seed,abs_bound",
        [
            (500, "sieve", 40_001, 1.25),
            (500, "linear", 40_002, 1.25),
            # the paper's small cell; centered but noisier at small N (the
            # review's Figures 9-14 finding), hence the wider pin
            (200, "linear", 40_003, 2.0),
        ],
    )
    def test_theta_recovery(self, n, learner, seed, abs_bound):
        df = chang_s422_kernel_frame(n, seed=seed)
        res = _fit_s422_expecting_only_a23(DMLDiD(outcome_learner=learner, seed=0, panel=False), df)
        err = abs(res.overall_att - CHANG_S422_THETA0)
        assert err < 4 * res.overall_se, (res.overall_att, res.overall_se)
        assert err < abs_bound, (res.overall_att, res.overall_se)
        confounded_err = abs(_unadjusted_did(df) - CHANG_S422_THETA0)
        assert err < confounded_err, (err, confounded_err)


@pytest.mark.slow
class TestChangS422MonteCarloCoverage:
    def test_coverage_sanity(self, ci_params):
        # Acceptance band CONDITIONAL on the scaled rep count (at ~22 reps the
        # tight band fails ~42% of the time at nominal coverage).
        n_reps = ci_params.bootstrap(200)
        n = 500
        hits = 0
        for rep in range(n_reps):
            df = chang_s422_kernel_frame(n, seed=41_000 + rep)
            res = _fit_s422_expecting_only_a23(
                DMLDiD(outcome_learner="linear", seed=rep, panel=False), df
            )
            lo, hi = res.conf_int
            hits += int(lo <= CHANG_S422_THETA0 <= hi)
        coverage = hits / n_reps
        lo_band, hi_band = (0.90, 0.99) if n_reps >= 100 else (0.80, 1.00)
        assert lo_band <= coverage <= hi_band, coverage
