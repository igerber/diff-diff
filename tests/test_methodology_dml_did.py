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
from diff_diff._dr_scores import chang_panel_score, chang_panel_score_augmented
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
