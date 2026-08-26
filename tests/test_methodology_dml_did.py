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
# NOT the paper's own high-dimensional Sec. 4 RCS parameterization, whose
# replication is a tracked TODO row): X ~ N(0, I_2), D ~
# Bernoulli(sigmoid(0.5 X1 - 0.5 X2)), T ~ Bernoulli(0.5), levels
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
