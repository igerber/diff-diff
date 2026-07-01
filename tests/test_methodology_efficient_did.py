"""EfficientDiD methodology test file — Chen, Sant'Anna & Xie (2025) walkthrough.

Companion to ``tests/test_efficient_did.py`` (API/unit surface) and
``tests/test_efficient_did_validation.py`` (HRS Table 6 + Compustat MC anchors):
this file validates the EfficientDiD *math* against the specific paper
equations/theorems, with paper-equation-numbered assertions. Mirrors the
structure of ``tests/test_methodology_power.py``.

Paper: Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). *Efficient
Difference-in-Differences and Event Study Estimators.* arXiv:2506.17729v1.
Paper review on file: ``docs/methodology/papers/chen-santanna-xie-2025-review.md``.

Decision pinned in PR-B (the source-validation pass):

- **D1 (sieve outcome regression):** the covariate doubly-robust path estimates
  the outcome regression ``m_hat(X)`` with a polynomial sieve (AIC/BIC order
  selection), matching the paper's flexible-nuisance specification (Section 4).
  Together with the sieve propensity ratio (Eq 4.1-4.2) and kernel-smoothed
  ``Omega*(X)`` this lets the covariate path attain the semiparametric efficiency
  bound under the paper's regularity conditions; it ELIMINATES the prior
  linear-OLS working-model deviation. Degree 1 reproduces the previous linear OLS
  up to floating point. ``TestCovariateSieveOutcomeRegression`` pins it.

Class structure (Verified Components):

- ``TestEfficientWeights`` — inverse-covariance optimal weights
  ``w = 1' Omega*^-1 / (1' Omega*^-1 1)`` (Eq 3.5 single-date / Eq 3.13 staggered),
  the min-variance property, and the singular-``Omega*`` pseudoinverse path.
- ``TestGeneratedOutcomeNoCov`` — the no-covariate generated outcome (Eq 3.9): the
  ``g'=g`` same-cohort pair telescopes to the per-baseline DiD
  ``E[Y_t-Y_tpre|g] - E[Y_t-Y_tpre|inf]`` (Eq 3.3); the ``g'=inf`` pairs telescope
  to the period-1 long-difference and are equal across ``t_pre``.
- ``TestNoCovariateClosedForm`` — Section 4.1 / Eq 3.13: the fitted efficient
  ``ATT(g,t)`` equals ``weights @ generated_outcomes`` rebuilt independently from
  the within-group sample means/covariances.
- ``TestPTPostReducesToCS`` — Corollary 3.1 / 3.2: under PT-Post EfficientDiD is
  just-identified (single never-treated, ``g-1`` baseline) and reduces to the
  standard single-baseline Callaway-Sant'Anna ``ATT(g,t)``.
- ``TestAnalyticalSE`` — Theorem 4.1: the analytical SE equals
  ``sqrt(mean(EIF^2)/n)``.
- ``TestHausmanStatistic`` — Theorem A.1 (Eq A.2): ``H = delta' V^+ delta`` with
  ``V = cov_post - cov_all`` (restricted MINUS efficient, footnote 21);
  well-conditioned ``V`` gives ``df = |E|``, a rank-deficient ``V`` gives
  ``df = effective_rank < |E|`` (the documented finite-sample DOF safeguard),
  reversing the covariance order flags an efficiency reversal, and the end-to-end
  pretest is internally consistent.
- ``TestCovariateSieveOutcomeRegression`` — D1: the sieve recovers a nonlinear-in-X
  conditional mean (selects ``K >= 2``), reproduces linear OLS on linear data
  (``K = 1``), and the growing sieve is more accurate than degree-1-capped
  sieves when the nuisances are nonlinear.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, EfficientDiD
from diff_diff.efficient_did import _hausman_quadratic_form
from diff_diff.efficient_did_covariates import (
    _silverman_bandwidth,
    estimate_inverse_propensity_sieve,
    estimate_outcome_regression,
    estimate_propensity_ratio_sieve,
)
from diff_diff.efficient_did_weights import (
    compute_efficient_weights,
    compute_generated_outcomes_nocov,
    compute_omega_star_nocov,
    enumerate_valid_triples,
)

# =============================================================================
# Helpers
# =============================================================================


def _staggered_panel(
    groups=(2, 3), n_per_group=120, n_control=120, n_periods=4, effect=2.0, sigma=0.4, seed=42
):
    """Balanced staggered panel with parallel trends (PT-All holds)."""
    rng = np.random.default_rng(seed)
    n_units = n_per_group * len(groups) + n_control
    ft = np.full(n_units, np.inf)
    for j, g in enumerate(groups):
        ft[j * n_per_group : (j + 1) * n_per_group] = g
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    ft_col = np.repeat(ft, n_periods)
    unit_fe = np.repeat(rng.normal(0, 1, n_units), n_periods)
    time_fe = np.tile(np.arange(1, n_periods + 1) * 0.3, n_units)
    tau = np.where((ft_col < np.inf) & (times >= ft_col), effect, 0.0)
    y = unit_fe + time_fe + tau + rng.normal(0, sigma, len(units))
    return pd.DataFrame({"unit": units, "time": times, "first_treat": ft_col, "y": y})


def _single_date_panel(
    n_treated=150, n_control=150, n_periods=3, treat_period=2, effect=2.0, sigma=0.4, seed=7
):
    """Two-group single-treatment-date panel (G in {treat_period, inf})."""
    rng = np.random.default_rng(seed)
    n_units = n_treated + n_control
    ft = np.full(n_units, np.inf)
    ft[:n_treated] = treat_period
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    ft_col = np.repeat(ft, n_periods)
    unit_fe = np.repeat(rng.normal(0, 1, n_units), n_periods)
    time_fe = np.tile(np.arange(1, n_periods + 1) * 0.5, n_units)
    tau = np.where((ft_col < np.inf) & (times >= ft_col), effect, 0.0)
    y = unit_fe + time_fe + tau + rng.normal(0, sigma, len(units))
    return pd.DataFrame({"unit": units, "time": times, "first_treat": ft_col, "y": y})


def _pivot_nocov(df):
    """Rebuild the no-covariate pivoted structures the fit derives internally.

    Means/covariances are permutation-invariant, so this reproduces the exact
    ``Omega*`` / generated outcomes regardless of internal unit ordering.
    """
    units = np.sort(df["unit"].unique())
    times = np.sort(df["time"].unique())
    period_to_col = {float(t): i for i, t in enumerate(times)}
    wide = df.pivot(index="unit", columns="time", values="y").loc[units, times].to_numpy()
    ft = df.groupby("unit")["first_treat"].first().loc[units].to_numpy(dtype=float)
    n = len(units)
    treatment_groups = sorted({float(g) for g in ft if np.isfinite(g)})
    cohort_masks = {float(g): (ft == g) for g in treatment_groups}
    never_treated_mask = ~np.isfinite(ft)
    cohort_fractions = {float(g): float((ft == g).sum()) / n for g in treatment_groups}
    cohort_fractions[np.inf] = float(never_treated_mask.sum()) / n
    return {
        "wide": wide,
        "period_to_col": period_to_col,
        "period_1_col": period_to_col[float(times[0])],
        "period_1": float(times[0]),
        "cohort_masks": cohort_masks,
        "never_treated_mask": never_treated_mask,
        "cohort_fractions": cohort_fractions,
        "time_periods": [float(t) for t in times],
        "treatment_groups": treatment_groups,
    }


def _cs_effect(cs_result, g, t):
    for (gg, tt), info in cs_result.group_time_effects.items():
        if int(gg) == int(g) and int(tt) == int(t):
            return info["effect"]
    return None


def _delta_y_arrays(n, x1, dy):
    """Pack a single-group outcome-change problem for estimate_outcome_regression.

    Returns (outcome_wide[n,2] with col0=Y_tpre=0, col1=Y_t=dy), covariate matrix
    [n,1]=x1, and an all-True group mask.
    """
    ow = np.column_stack([np.zeros(n), dy])
    return ow, x1.reshape(-1, 1), np.ones(n, dtype=bool)


# =============================================================================
# Eq 3.5 / 3.13 — inverse-covariance optimal weights
# =============================================================================


class TestEfficientWeights:
    """Optimal weights ``w = 1' Omega*^-1 / (1' Omega*^-1 1)`` (Eq 3.5 / 3.13)."""

    def test_inverse_covariance_weights_hand_computed(self):
        omega = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 3.0]])
        w, used_pinv, _ = compute_efficient_weights(omega)
        inv = np.linalg.inv(omega)
        ones = np.ones(3)
        expected = (ones @ inv) / (ones @ inv @ ones)
        assert not used_pinv
        np.testing.assert_allclose(w, expected, atol=1e-10)
        assert abs(w.sum() - 1.0) < 1e-12

    def test_weights_are_minimum_variance(self):
        """The inverse-covariance weights minimize ``v' Omega v`` over ``1'v = 1``
        (the semiparametric-efficiency property of Theorem 3.1/3.2)."""
        omega = np.array([[1.0, 0.3], [0.3, 2.0]])
        w, _, _ = compute_efficient_weights(omega)
        var_w = float(w @ omega @ w)
        rng = np.random.default_rng(0)
        for _ in range(500):
            a = rng.uniform(-1.0, 2.0)
            v = np.array([a, 1.0 - a])
            assert v @ omega @ v >= var_w - 1e-12

    def test_singular_omega_uses_pseudoinverse(self):
        """Two identical rows -> singular Omega* -> pinv path; weights still sum to 1."""
        omega = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, used_pinv, _ = compute_efficient_weights(omega)
        assert used_pinv
        assert abs(w.sum() - 1.0) < 1e-8


# =============================================================================
# Eq 3.9 — no-covariate generated outcome (telescoping)
# =============================================================================


class TestGeneratedOutcomeNoCov:
    """The no-covariate generated outcome ``Y_hat_j`` (Eq 3.9), built from
    within-group sample means, telescopes as the paper describes."""

    def _setup(self):
        # 5 units in cohort g=3, 5 never-treated; periods {1,2,3}.
        # g=3 outcomes [1,2,5]; never-treated [0,3,4] (deliberately non-parallel
        # so the tpre=2 per-baseline DiD differs from the period-1 long-difference).
        wide = np.array([[1.0, 2.0, 5.0]] * 5 + [[0.0, 3.0, 4.0]] * 5)
        cohort_masks = {3.0: np.array([True] * 5 + [False] * 5)}
        never = np.array([False] * 5 + [True] * 5)
        period_to_col = {1.0: 0, 2.0: 1, 3.0: 2}
        return wide, cohort_masks, never, period_to_col

    def test_telescoping_to_per_baseline_and_long_difference(self):
        wide, cohort_masks, never, p2c = self._setup()
        pairs = enumerate_valid_triples(3.0, [3.0], [1.0, 2.0, 3.0], 1.0, "all")
        y_hat = compute_generated_outcomes_nocov(
            3.0, 3.0, pairs, wide, cohort_masks, never, p2c, period_1_col=0
        )
        gen = dict(zip(pairs, y_hat))

        # g'=g=3, t_pre=2: per-baseline DiD (Eq 3.3) = (Y3-Y2|g) - (Y3-Y2|inf)
        per_baseline = (5.0 - 2.0) - (4.0 - 3.0)  # = 2.0
        assert gen[(3.0, 2.0)] == pytest.approx(per_baseline, abs=1e-12)

        # g'=inf telescopes to the period-1 long-difference (Y3-Y1|g) - (Y3-Y1|inf),
        # independent of t_pre.
        long_diff = (5.0 - 1.0) - (4.0 - 0.0)  # = 0.0
        assert gen[(np.inf, 2.0)] == pytest.approx(long_diff, abs=1e-12)
        assert gen[(np.inf, 3.0)] == pytest.approx(long_diff, abs=1e-12)

        # The two mechanisms genuinely differ here (non-trivial telescoping).
        assert abs(gen[(3.0, 2.0)] - gen[(np.inf, 2.0)]) > 1.0


# =============================================================================
# Section 4.1 / Eq 3.13 — no-covariate closed form
# =============================================================================


class TestNoCovariateClosedForm:
    """The fitted efficient ATT equals ``weights @ generated_outcomes`` (Eq 3.13)
    rebuilt independently from within-group sample means/covariances (Section 4.1)."""

    def test_efficient_att_equals_weighted_generated_outcomes(self):
        df = _staggered_panel(
            groups=(2, 3), n_per_group=120, n_control=120, n_periods=4, sigma=0.4, seed=11
        )
        res = EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")
        P = _pivot_nocov(df)

        # Pick an overidentified (g,t) (|H| >= 2).
        target, valid_pairs = None, None
        for g, t in res.group_time_effects:
            if t < g:
                continue
            pairs = enumerate_valid_triples(
                float(g), P["treatment_groups"], P["time_periods"], P["period_1"], "all"
            )
            if len(pairs) >= 2:
                target, valid_pairs = (float(g), float(t)), pairs
                break
        assert target is not None, "no overidentified (g,t) found"
        assert valid_pairs is not None
        g, t = target

        omega = compute_omega_star_nocov(
            g,
            t,
            valid_pairs,
            P["wide"],
            P["cohort_masks"],
            P["never_treated_mask"],
            P["period_to_col"],
            P["period_1_col"],
            P["cohort_fractions"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, _, _ = compute_efficient_weights(omega)
        y_hat = compute_generated_outcomes_nocov(
            g,
            t,
            valid_pairs,
            P["wide"],
            P["cohort_masks"],
            P["never_treated_mask"],
            P["period_to_col"],
            P["period_1_col"],
        )
        att_manual = float(w @ y_hat)

        fit_effect = None
        for (gg, tt), info in res.group_time_effects.items():
            if float(gg) == g and float(tt) == t:
                fit_effect = info["effect"]
                break
        assert fit_effect == pytest.approx(att_manual, abs=1e-9)


# =============================================================================
# Corollary 3.1 / 3.2 — PT-Post reduces to Callaway-Sant'Anna
# =============================================================================


class TestPTPostReducesToCS:
    """Under PT-Post EfficientDiD is just-identified and matches CS single-baseline."""

    def test_pt_post_is_just_identified_single_baseline(self):
        # Corollary 3.2: PT-Post -> exactly one moment (never-treated, g-1 baseline).
        pairs = enumerate_valid_triples(3.0, [3.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], 1.0, "post")
        assert pairs == [(np.inf, 2.0)]

    def test_pt_post_matches_cs_single_date(self):
        # Corollary 3.1: two-group single-date PT-Post == CS, exactly.
        df = _single_date_panel(n_treated=200, n_control=200, seed=3)
        edid = EfficientDiD(pt_assumption="post").fit(df, "y", "unit", "time", "first_treat")
        cs = CallawaySantAnna(control_group="never_treated").fit(
            df, "y", "unit", "time", "first_treat"
        )
        checked = 0
        for (g, t), info in edid.group_time_effects.items():
            if t >= g:
                c = _cs_effect(cs, g, t)
                assert c is not None
                assert info["effect"] == pytest.approx(c, abs=1e-9)
                checked += 1
        assert checked >= 1

    def test_pt_post_matches_cs_staggered(self):
        # Corollary 3.2: staggered PT-Post == CS single-baseline (post-treatment).
        df = _staggered_panel(groups=(2, 3), n_per_group=150, n_control=150, seed=5)
        edid = EfficientDiD(pt_assumption="post").fit(df, "y", "unit", "time", "first_treat")
        cs = CallawaySantAnna(control_group="never_treated").fit(
            df, "y", "unit", "time", "first_treat"
        )
        for (g, t), info in edid.group_time_effects.items():
            if t >= g and np.isfinite(info["effect"]):
                c = _cs_effect(cs, g, t)
                if c is not None:
                    assert info["effect"] == pytest.approx(c, abs=1e-8)


# =============================================================================
# Theorem 4.1 — analytical SE = sqrt(mean(EIF^2)/n)
# =============================================================================


class TestAnalyticalSE:
    """The default analytical SE is ``sqrt(mean(EIF^2)/n)`` (Theorem 4.1, p.21)."""

    def test_se_equals_sqrt_mean_eif_squared_over_n(self):
        df = _staggered_panel(groups=(2, 3), n_per_group=120, n_control=120, seed=9)
        res = EfficientDiD(pt_assumption="all").fit(
            df, "y", "unit", "time", "first_treat", store_eif=True
        )
        assert res.influence_functions is not None
        checked = 0
        for (g, t), info in res.group_time_effects.items():
            se = info["se"]
            if not (np.isfinite(se) and se > 0):
                continue
            eif = res.influence_functions[(g, t)]
            expected = float(np.sqrt(np.mean(eif**2) / len(eif)))
            assert se == pytest.approx(expected, rel=1e-9)
            checked += 1
        assert checked >= 1


# =============================================================================
# Theorem A.1 (Eq A.2) — Hausman statistic
# =============================================================================


class TestHausmanStatistic:
    """``H = delta' V^+ delta`` with ``V = cov_post - cov_all`` (restricted minus
    efficient); effective-rank degrees of freedom; covariance-direction guard."""

    def test_well_conditioned_v_df_equals_number_of_horizons(self):
        from scipy.stats import chi2

        cov_all = np.array([[0.04, 0.01], [0.01, 0.05]])  # efficient (smaller)
        cov_post = np.array([[0.09, 0.02], [0.02, 0.10]])  # restricted (larger)
        delta = np.array([0.3, -0.2])
        H, rank, p, n_neg, ok = _hausman_quadratic_form(delta, cov_post, cov_all)
        V = cov_post - cov_all
        assert ok and n_neg == 0 and rank == 2
        assert H == pytest.approx(float(delta @ np.linalg.inv(V) @ delta), rel=1e-10)
        assert p == pytest.approx(float(chi2.sf(H, df=2)), rel=1e-10)

    def test_rank_deficient_v_uses_effective_rank(self):
        from scipy.stats import chi2

        # Second coordinate has equal restricted/efficient variance -> V[1,1] = 0
        # -> a zero eigenvalue -> df = effective_rank = 1 < |E| = 2.
        cov_all = np.array([[0.04, 0.0], [0.0, 0.05]])
        cov_post = np.array([[0.09, 0.0], [0.0, 0.05]])
        delta = np.array([0.3, 0.2])
        H, rank, p, n_neg, ok = _hausman_quadratic_form(delta, cov_post, cov_all)
        assert ok and rank == 1 and n_neg == 0
        # Only the positive-variance direction contributes: 0.3^2 / (0.09 - 0.04).
        assert H == pytest.approx(0.3**2 / (0.09 - 0.04), rel=1e-9)
        assert p == pytest.approx(float(chi2.sf(H, df=1)), rel=1e-9)

    def test_covariance_direction_restricted_minus_efficient(self):
        # V must be cov_post - cov_all. Reversing the order gives a negative-definite
        # V (efficiency reversal): n_negative > 0 and no positive eigenvalues.
        cov_all = np.array([[0.04]])
        cov_post = np.array([[0.09]])
        delta = np.array([0.2])
        H1, r1, _, nneg1, _ = _hausman_quadratic_form(delta, cov_post, cov_all)
        H2, r2, _, nneg2, _ = _hausman_quadratic_form(delta, cov_all, cov_post)
        assert nneg1 == 0 and r1 == 1
        assert H1 == pytest.approx(0.2**2 / (0.09 - 0.04), rel=1e-9)
        assert nneg2 == 1 and r2 == 0 and np.isnan(H2)

    def test_end_to_end_pretest_consistent(self):
        df = _staggered_panel(groups=(3, 5), n_per_group=150, n_control=200, n_periods=7, seed=21)
        pretest = EfficientDiD.hausman_pretest(df, "y", "unit", "time", "first_treat")
        assert pretest.gt_details is not None
        n_horizons = len(pretest.gt_details)
        assert n_horizons >= 1
        assert 1 <= pretest.df <= n_horizons  # effective rank <= |E|
        assert pretest.statistic >= 0
        assert 0.0 <= pretest.p_value <= 1.0


# =============================================================================
# D1 — sieve outcome regression (the PR-B upgrade)
# =============================================================================


class TestCovariateSieveOutcomeRegression:
    """The outcome regression m_hat(X) is a polynomial sieve (Section 4)."""

    def test_sieve_recovers_nonlinear_conditional_mean(self):
        # E[dY|x1] = x1^2 : the sieve (K>=2) tracks the curvature; a forced-linear
        # (K=1) fit cannot. A pure-linear m_hat would have corr(m, x1^2) ~ 0.
        rng = np.random.default_rng(0)
        n = 400
        x1 = rng.normal(size=n)
        dy = x1**2 + rng.normal(scale=0.3, size=n)
        ow, cov, mask = _delta_y_arrays(n, x1, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_sieve = estimate_outcome_regression(ow, cov, mask, 1, 0, k_max=4, criterion="bic")
            m_linear = estimate_outcome_regression(ow, cov, mask, 1, 0, k_max=1, criterion="bic")
        truth = x1**2
        mse_sieve = float(np.mean((m_sieve - truth) ** 2))
        mse_linear = float(np.mean((m_linear - truth) ** 2))
        assert np.corrcoef(m_sieve, truth)[0, 1] > 0.9
        assert mse_sieve < 0.5 * mse_linear  # sieve clearly better
        assert not np.allclose(m_sieve, m_linear)  # K>=2 was actually selected

    def test_sieve_degrades_to_linear_on_linear_data(self):
        # E[dY|x1] = 3*x1 (+ noise): BIC selects K=1 and m_hat matches linear OLS.
        rng = np.random.default_rng(1)
        n = 400
        x1 = rng.normal(size=n)
        dy = 3.0 * x1 + rng.normal(scale=0.5, size=n)
        ow, cov, mask = _delta_y_arrays(n, x1, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_sieve = estimate_outcome_regression(ow, cov, mask, 1, 0, k_max=4, criterion="bic")
        Xd = np.column_stack([np.ones(n), x1])
        beta = np.linalg.lstsq(Xd, dy, rcond=None)[0]
        m_ols = Xd @ beta
        np.testing.assert_allclose(m_sieve, m_ols, atol=1e-8)

    def test_all_degrees_skipped_falls_back_to_group_mean(self):
        # A constant covariate makes every design rank-deficient -> fall back to the
        # intercept-only within-group mean of dY (the documented fallback semantic).
        rng = np.random.default_rng(2)
        n = 50
        x1 = np.ones(n)  # zero variance -> [1, X] is singular at every K
        dy = 2.0 + rng.normal(scale=0.1, size=n)
        ow, cov, mask = _delta_y_arrays(n, x1, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = estimate_outcome_regression(ow, cov, mask, 1, 0, k_max=4, criterion="bic")
        np.testing.assert_allclose(m, np.full(n, dy.mean()), atol=1e-9)

    def test_sieve_order_selection_invariant_to_survey_weight_scale(self):
        # Rescaling survey weights w -> c*w leaves the selected order and m̂
        # unchanged: the OLS IC uses the positive-weight support count for both n
        # and the penalty, and only RSS is weighted, so the criterion shifts by a
        # K-independent constant (REGISTRY outcome-regression Note).
        rng = np.random.default_rng(7)
        n = 300
        x1 = rng.normal(size=n)
        dy = x1**2 + rng.normal(scale=0.4, size=n)  # nonlinear -> K>=2 selected
        ow, cov, mask = _delta_y_arrays(n, x1, dy)
        w = rng.uniform(0.5, 2.0, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = estimate_outcome_regression(
                ow, cov, mask, 1, 0, k_max=4, criterion="bic", unit_weights=w
            )
            m2 = estimate_outcome_regression(
                ow, cov, mask, 1, 0, k_max=4, criterion="bic", unit_weights=100.0 * w
            )
        np.testing.assert_allclose(m1, m2, atol=1e-9)

    def test_weighted_all_degrees_skipped_uses_weighted_group_mean(self):
        # Survey weights + a constant covariate (every design singular) -> the
        # fallback is the WEIGHTED within-group mean of dY.
        rng = np.random.default_rng(8)
        n = 40
        x1 = np.ones(n)
        dy = 1.0 + rng.normal(scale=0.1, size=n)
        ow, cov, mask = _delta_y_arrays(n, x1, dy)
        w = rng.uniform(0.5, 2.0, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = estimate_outcome_regression(
                ow, cov, mask, 1, 0, k_max=4, criterion="bic", unit_weights=w
            )
        expected = float(np.average(dy, weights=w))
        np.testing.assert_allclose(m, np.full(n, expected), atol=1e-9)

    def test_zero_weight_padding_does_not_change_auto_selected_order(self):
        # Zero-weight (survey-subpopulation / padded) rows must be INERT for sieve
        # ORDER SELECTION, not merely for the WLS coefficients at a fixed K. The
        # auto order, the n_basis admissibility cap, and the IC sample-size terms
        # all key off the positive-weight support; if they used the raw row count
        # instead, padding the panel with zero-weight rows would push floor(n^{1/5})
        # to a higher candidate degree and could silently change the selected K
        # (hence the DR estimate). This exercises the AUTO path (NO explicit k_max)
        # with a genuinely cubic conditional mean: 240 positive-weight units give
        # floor(240^{1/5}) = 2, but raw-count logic on the 300 padded rows would
        # give floor(300^{1/5}) = 3 and select the cubic term -- diverging from the
        # positive-weight-only fit. This test FAILS under raw-count selection and
        # passes once selection uses the positive-weight support.
        rng = np.random.default_rng(101)
        n_real, n_pad = 240, 60  # floor(240^.2)=2 ; floor(300^.2)=3
        x1_real = rng.uniform(-2.0, 2.0, size=n_real)
        dy_real = 2.0 * x1_real**3 + rng.normal(scale=0.3, size=n_real)  # cubic -> K=3 if reachable
        # Padded rows: arbitrary (deliberately wild) covariates/outcomes, weight 0.
        x1_pad = rng.uniform(-2.0, 2.0, size=n_pad)
        dy_pad = rng.normal(scale=5.0, size=n_pad)
        x1 = np.concatenate([x1_real, x1_pad])
        dy = np.concatenate([dy_real, dy_pad])
        ow = np.column_stack([np.zeros(n_real + n_pad), dy])
        cov = x1.reshape(-1, 1)
        w = np.concatenate([rng.uniform(0.5, 2.0, n_real), np.zeros(n_pad)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # group_mask spans all rows; 60 carry zero weight -> support = 240.
            m_padded = estimate_outcome_regression(
                ow, cov, np.ones(n_real + n_pad, dtype=bool), 1, 0, criterion="bic", unit_weights=w
            )
            # group_mask = positive-weight rows only (the analytically-equivalent fit).
            m_support = estimate_outcome_regression(
                ow, cov, w > 0, 1, 0, criterion="bic", unit_weights=w
            )
        assert np.all(np.isfinite(m_padded))
        # Strict invariance: zero-weight padding changes neither the auto-selected
        # order nor the fitted m_hat anywhere.
        np.testing.assert_allclose(m_padded, m_support, atol=1e-9)

    def test_sieve_recovers_multivariate_nonlinear_conditional_mean(self):
        # d=2 covariates with a nonlinear conditional mean including a cross
        # term: E[dY|x1,x2] = x1^2 + x1*x2. The multivariate sieve basis
        # (dimension p_K = comb(K+2, 2)) captures it at K>=2; a degree-1 fit
        # (linear in x1, x2, no quadratic/cross terms) cannot. Exercises the
        # degree-vs-dimension distinction (p_K = comb(K+d, d), not K) that
        # matters once d > 1.
        rng = np.random.default_rng(13)
        n = 600  # floor(n^(1/5)) = 3 -> candidate degrees 1..3
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        cov = np.column_stack([x1, x2])
        truth = x1**2 + x1 * x2
        dy = truth + rng.normal(scale=0.4, size=n)
        ow = np.column_stack([np.zeros(n), dy])
        mask = np.ones(n, dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_sieve = estimate_outcome_regression(ow, cov, mask, 1, 0, criterion="bic")
            m_lin = estimate_outcome_regression(ow, cov, mask, 1, 0, k_max=1, criterion="bic")
        assert np.corrcoef(m_sieve, truth)[0, 1] > 0.9
        assert np.mean((m_sieve - truth) ** 2) < 0.5 * np.mean((m_lin - truth) ** 2)

    def test_growing_sieve_allows_order_above_5_for_large_group(self):
        # The sieve is a *growing* sieve: the candidate order is floor(n^(1/5))
        # with NO fixed ceiling, which is what satisfies Assumption C.1's
        # uniform-consistency / product-rate conditions (Theorem 4.1). For a
        # large group (n=8000 -> floor(n^(1/5))=6) a genuinely degree-6
        # conditional mean is captured, which the previous hard K<=5 cap could
        # not represent.
        rng = np.random.default_rng(11)
        n = 8000  # floor(n^(1/5)) = 6
        x1 = rng.uniform(-2.0, 2.0, size=n)
        truth = x1**6 - 3.0 * x1**4  # needs degree 6
        dy = truth + rng.normal(scale=0.5, size=n)
        ow, cov, mask = _delta_y_arrays(n, x1, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_grow = estimate_outcome_regression(ow, cov, mask, 1, 0, criterion="bic")
            m_cap5 = estimate_outcome_regression(ow, cov, mask, 1, 0, k_max=5, criterion="bic")
        # The growing sieve (auto k_max=6) captures the degree-6 truth that a
        # K<=5-capped fit structurally cannot.
        assert np.mean((m_grow - truth) ** 2) < np.mean((m_cap5 - truth) ** 2)
        assert np.corrcoef(m_grow, truth)[0, 1] > 0.95

    def test_outcome_regression_rejects_bad_criterion(self):
        # The helper validates its criterion for direct callers (the estimator
        # surface validates upstream, but the helper must fail closed too).
        ow, cov, mask = _delta_y_arrays(10, np.arange(10.0), np.arange(10.0))
        with pytest.raises(ValueError, match="criterion"):
            estimate_outcome_regression(ow, cov, mask, 1, 0, criterion="xic")

    def test_covariate_dr_path_last_cohort_with_anticipation(self):
        # DR (covariate) path under control_group="last_cohort" (all units
        # eventually treated -> last cohort is the pseudo-control) with
        # anticipation trimming. Mirrors the no-covariate last_cohort test on
        # the new sieve outcome-regression path.
        rng = np.random.default_rng(17)
        groups = (3, 5, 7)
        n_per, n_periods = 50, 7
        rows, uid = [], 0
        for g in groups:
            for _ in range(n_per):
                x1 = rng.normal()
                ufe = rng.normal(0, 0.5)
                for t in range(1, n_periods + 1):
                    y = ufe + 0.4 * x1 + 0.2 * t + (1.5 if t >= g else 0.0) + rng.normal(0, 0.3)
                    rows.append((uid, t, float(g), y, x1))
                uid += 1
        df = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = EfficientDiD(
                pt_assumption="all", control_group="last_cohort", anticipation=1
            ).fit(df, "y", "unit", "time", "first_treat", covariates=["x1"], aggregate="all")
        # anticipation=1, last_g=7 -> effective last cohort at 6 -> time_periods 1..5
        assert max(res.time_periods) == 5
        assert 7 not in res.groups  # last cohort reclassified as pseudo-control
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)

    @pytest.mark.slow
    def test_covariate_path_beats_forced_linear_under_nonlinear_nuisance(self, ci_params):
        """Conditional-PT DGP with nonlinear selection AND nonlinear Y(0) trend:
        the auto growing sieve recovers ATT (both nuisances captured), while
        sieve_k_max=1 -- which degree-1-constrains ALL covariate-path sieves
        (the outcome regression AND both propensity sieves) -- misspecifies both
        nuisances and so has larger error. This compares growing vs degree-1
        sieves; it does not isolate the outcome regression alone."""
        tau = 2.0
        n_sims = ci_params.bootstrap(60, min_n=20)
        err_sieve, err_linear = [], []
        for s in range(n_sims):
            rng = np.random.default_rng(5000 + s)
            n = 320
            x1 = rng.normal(size=n)
            # Nonlinear selection on x1^2 (kept within (0,1) for overlap).
            p_treat = 1.0 / (1.0 + np.exp(-0.8 * (x1**2 - 1.0)))
            treated = rng.uniform(size=n) < p_treat
            ft = np.where(treated, 2.0, np.inf)
            slope = 0.5 * x1 + 0.6 * x1**2  # nonlinear Y(0) trend in x1
            rows = []
            unit_fe = rng.normal(0, 0.5, n)
            for i in range(n):
                for t in (1, 2, 3):
                    y0 = unit_fe[i] + slope[i] * (t - 1) + rng.normal(0, 0.3)
                    y = y0 + (tau if (np.isfinite(ft[i]) and t >= ft[i]) else 0.0)
                    rows.append((i, t, ft[i], y, x1[i]))
            df = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a_sieve = (
                    EfficientDiD(pt_assumption="all")
                    .fit(df, "y", "unit", "time", "first_treat", covariates=["x1"], aggregate="all")
                    .overall_att
                )
                a_linear = (
                    EfficientDiD(pt_assumption="all", sieve_k_max=1)
                    .fit(df, "y", "unit", "time", "first_treat", covariates=["x1"], aggregate="all")
                    .overall_att
                )
            if np.isfinite(a_sieve):
                err_sieve.append(abs(a_sieve - tau))
            if np.isfinite(a_linear):
                err_linear.append(abs(a_linear - tau))
        mae_sieve = float(np.mean(err_sieve))
        mae_linear = float(np.mean(err_linear))
        # The growing sieve recovers ATT and is no worse than the all-degree-1
        # fit; under this nonlinear-nuisance DGP it is materially better.
        assert mae_sieve < 0.5
        assert mae_sieve <= mae_linear + 0.05


# =============================================================================
# Growing sieve (order > 5) — the two propensity-path sieves
# =============================================================================


def _cheb_t6(u):
    """6th Chebyshev polynomial: its energy is purely degree 6 (orthogonal to all
    lower degrees over [-1, 1]), so a degree-5 sieve captures none of it and a
    degree-6 sieve is required -- a clean way to force order-6 selection."""
    return 32 * u**6 - 48 * u**4 + 18 * u**2 - 1


class TestPropensitySieveGrowingOrder:
    """The two propensity-path sieves are growing sieves with no K<=5 cap.

    Companion to ``test_growing_sieve_allows_order_above_5_for_large_group`` (which
    covers the outcome regression): the hard ``K<=5`` ceiling was removed from ALL
    THREE nuisance sieves, so for a large support (n with floor(n^{1/5}) = 6) a
    genuinely degree-6 nuisance is selected. At that support the auto k_max is
    exactly 6, so ``auto != k_max=5 fit`` is a sharp proof that order 6 was
    selected (and ``auto == k_max=6 fit`` confirms it). The fitted target is made
    pure degree-6 via the 6th Chebyshev polynomial so degree 5 cannot represent it.
    """

    def test_propensity_ratio_sieve_selects_order_six(self):
        # r_{g,g'}(X) = p_g/p_{g'} is built as a pure degree-6 polynomial (0.3 +
        # 0.18*T6, in [0.12, 0.48] so well inside the ratio clip). The comparison
        # group g' is the majority (~78%), so its support clears 7776 and the auto
        # k_max reaches 6.
        rng = np.random.default_rng(0)
        n = 20000  # comparison-group support ~15.6k -> floor(.^{1/5}) = 6
        x1 = rng.uniform(-2.0, 2.0, size=n)
        r_true = 0.3 + 0.18 * _cheb_t6(x1 / 2.0)
        p_g = r_true / (1.0 + r_true)
        g = rng.uniform(size=n) < p_g
        cov = x1.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = estimate_propensity_ratio_sieve(cov, g, ~g, criterion="bic")
            cap5 = estimate_propensity_ratio_sieve(cov, g, ~g, k_max=5, criterion="bic")
            cap6 = estimate_propensity_ratio_sieve(cov, g, ~g, k_max=6, criterion="bic")
        assert int((~g).sum() ** 0.2) == 6  # comparison support gives auto k_max = 6
        assert not np.allclose(auto, cap5, atol=1e-9)  # order > 5 was selected
        np.testing.assert_allclose(auto, cap6, atol=1e-9)  # specifically order 6

    def test_inverse_propensity_sieve_selects_order_six(self):
        # s_{g'}(X) = 1/p_{g'}(X) is built as a pure degree-6 polynomial
        # (3.0 + 1.8*T6, >= 1.2 so p in (0, 1]); the group support (~8.5k) clears
        # 7776 so the auto k_max reaches 6.
        rng = np.random.default_rng(0)
        n = 20000
        x1 = rng.uniform(-2.0, 2.0, size=n)
        s_true = 3.0 + 1.8 * _cheb_t6(x1 / 2.0)
        p = 1.0 / s_true
        g = rng.uniform(size=n) < p
        cov = x1.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = estimate_inverse_propensity_sieve(cov, g, criterion="bic")
            cap5 = estimate_inverse_propensity_sieve(cov, g, k_max=5, criterion="bic")
            cap6 = estimate_inverse_propensity_sieve(cov, g, k_max=6, criterion="bic")
        assert int(g.sum() ** 0.2) == 6  # group support gives auto k_max = 6
        assert not np.allclose(auto, cap5, atol=1e-9)  # order > 5 was selected
        np.testing.assert_allclose(auto, cap6, atol=1e-9)  # specifically order 6


# =============================================================================
# Survey-weighted Silverman bandwidth (conditional Omega* kernel)
# =============================================================================


class TestSurveyWeightedSilvermanBandwidth:
    """`_silverman_bandwidth` uses a survey-weighted per-dimension dispersion on
    the positive-weight support, with the rate term `n` kept as the positive-
    weight count (REGISTRY EfficientDiD covariates Note). These lock the
    refinement's behavior and its invariances.
    """

    @staticmethod
    def _weighted_median_std_bandwidth(X, w):
        """Reference: median-of-weighted-std Silverman bandwidth over the
        positive-weight support, n = positive-weight count."""
        support = w > 0
        Xs, ws = X[support], w[support]
        n, d = Xs.shape
        wn = ws / ws.sum()
        mean = wn @ Xs
        var = wn @ (Xs - mean) ** 2
        stds = np.sqrt(var)
        stds[stds < 1e-10] = 1.0
        median_std = float(np.median(stds))
        return (4.0 / (d + 2)) ** (1.0 / (d + 4)) * median_std * n ** (-1.0 / (d + 4))

    def test_matches_weighted_reference_formula(self):
        # Under non-uniform weights the bandwidth equals the hand-computed
        # weighted-dispersion formula (not the unweighted one).
        rng = np.random.default_rng(11)
        X = rng.normal(size=(250, 3))
        w = rng.uniform(0.4, 3.0, size=250)
        got = _silverman_bandwidth(X, w)
        expected = self._weighted_median_std_bandwidth(X, w)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
        # And it genuinely differs from the unweighted bandwidth.
        assert abs(got - _silverman_bandwidth(X)) > 1e-4

    def test_reduces_to_unweighted_under_uniform_weights(self):
        # Uniform positive weights -> weighted std == unweighted population std.
        rng = np.random.default_rng(12)
        X = rng.normal(size=(200, 2))
        np.testing.assert_allclose(
            _silverman_bandwidth(X, np.full(200, 3.7)),
            _silverman_bandwidth(X),
            rtol=0.0,
            atol=1e-12,
        )

    def test_invariant_to_weight_scale(self):
        # w -> c*w leaves the weighted mean/std and the count unchanged.
        rng = np.random.default_rng(13)
        X = rng.normal(size=(180, 3))
        w = rng.uniform(0.5, 2.0, size=180)
        np.testing.assert_allclose(
            _silverman_bandwidth(X, w), _silverman_bandwidth(X, 137.0 * w), rtol=0.0, atol=1e-12
        )

    def test_invariant_to_zero_weight_padding(self):
        # Appending extreme zero-weight rows must not move the bandwidth: they
        # leave the support, the weighted moments, and the count all unchanged.
        rng = np.random.default_rng(14)
        X = rng.normal(size=(150, 2))
        w = rng.uniform(0.5, 2.0, size=150)
        X_pad = np.vstack([X, rng.uniform(30.0, 60.0, size=(40, 2))])
        w_pad = np.concatenate([w, np.zeros(40)])
        np.testing.assert_allclose(
            _silverman_bandwidth(X, w), _silverman_bandwidth(X_pad, w_pad), rtol=0.0, atol=0.0
        )


# =============================================================================
# End-to-end survey invariance: zero-weight padding must not change the estimate
# =============================================================================


class TestSurveyZeroWeightInvariance:
    """EfficientDiD's DR (covariate) estimate is invariant to zero-weight padding.

    Adding survey rows with weight 0 carries no information, so the point estimate
    must not change. Both auto-selectors on the covariate path key off the
    positive-weight support, so two distinct paths are covered:

    - **Just-identified (H=1)** — ``test_..._invariant``: pins the sieve ORDER
      selection. The never-treated comparison group has 240 positive-weight units
      (floor(240^{1/5}) = 2); padding it with 60 zero-weight units would push the
      raw count to 300 (floor = 3) under the buggy raw-count selection and select
      the genuinely-cubic conditional mean (CI codex P0, round 1). At H=1
      ``compute_per_unit_weights`` short-circuits to 1, so the kernel/bandwidth
      path is NOT exercised here.
    - **Overidentified (H>1)** — ``test_..._invariant_overidentified``: pins the
      auto Silverman BANDWIDTH for the kernel-smoothed ``Omega*(X)``. With H>1 the
      efficient weights depend on ``Omega*(X)``, whose auto bandwidth must ignore
      zero-weight rows — else extreme-covariate padding inflates the unweighted std
      and silently moves ``ATT(g,t)`` (CI codex P0, round 2).
    """

    def _panel(self, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        uid = 0

        def add(n, ft, weight, fe_sd, noise_sd):
            nonlocal uid
            for _ in range(n):
                x1 = rng.uniform(-2.0, 2.0)
                ufe = rng.normal(0.0, fe_sd)
                cubic = 1.3 * x1**3  # nonlinear conditional trend (needs degree 3)
                for t in (1, 2):
                    treated_now = np.isfinite(ft) and t >= ft
                    y = (
                        ufe
                        + 0.3 * (t - 1)
                        + cubic * (t - 1)
                        + (2.0 if treated_now else 0.0)
                        + rng.normal(0.0, noise_sd)
                    )
                    rows.append((uid, t, ft, y, x1, weight))
                uid += 1

        add(120, 2.0, 1.0, 0.5, 0.3)  # treated cohort
        add(240, np.inf, 1.0, 0.5, 0.3)  # real never-treated (positive weight)
        base = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "w"])
        n_base_rows = len(rows)
        add(60, np.inf, 0.0, 3.0, 3.0)  # zero-weight padded controls (wild outcomes)
        padded = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "w"])
        # ``base`` is exactly the positive-weight subset of ``padded``.
        assert (padded["w"].to_numpy()[:n_base_rows] > 0).all()
        return base, padded

    def test_zero_weight_padding_leaves_att_invariant(self):
        from diff_diff.survey import SurveyDesign

        base, padded = self._panel(seed=0)

        def fit(df):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return EfficientDiD(pt_assumption="all").fit(
                    df,
                    "y",
                    "unit",
                    "time",
                    "first_treat",
                    covariates=["x1"],
                    survey_design=SurveyDesign(weights="w"),
                    aggregate="all",
                )

        r_filtered = fit(base)
        r_padded = fit(padded)
        # The DR point estimate is exactly invariant to zero-weight padding (the
        # sieve-selection fix): the WLS fit, the weighted RSS, and now the order
        # selection are all keyed off the positive-weight support. NOTE this
        # single-date fixture is just-identified (H=1 per cell), so
        # ``compute_per_unit_weights`` short-circuits to 1 and the kernel/bandwidth
        # path is NOT exercised here — the overidentified test below covers that.
        np.testing.assert_allclose(
            r_padded.overall_att, r_filtered.overall_att, atol=1e-9, rtol=0.0
        )
        # The SE is invariant up to a tiny finite-sample DOF correction: the shared
        # survey sandwich (compute_survey_vcov / _compute_stratified_psu_meat) counts
        # zero-weight units as PSUs in its n_psu/(n_psu-1)-style correction. The
        # weighted scores themselves are zero for zero-weight rows, so this is a
        # second-order (~2e-4 relative) cross-cutting survey-infra effect, NOT the
        # sieve selection (tracked in TODO.md).
        np.testing.assert_allclose(r_padded.overall_se, r_filtered.overall_se, rtol=2e-3, atol=0.0)

    def _overid_panel(self, seed=0):
        """Staggered panel (cohorts 2 & 3 + never-treated, 4 periods) whose PT-All
        cells are OVERIDENTIFIED (H > 1): each ATT(g,t) has several admissible
        (g', t_pre) comparison moments, so ``compute_per_unit_weights`` forms the
        inverse-Omega* efficient combination rather than short-circuiting to 1 —
        i.e. the kernel-smoothed Omega*(X) and its auto Silverman bandwidth ARE on
        the path. The zero-weight padded units carry EXTREME covariates so that any
        bandwidth dependence on zero-weight rows is forced to show up (an extreme
        x1 inflates the unweighted std -> a larger auto bandwidth -> different
        Omega*)."""
        rng = np.random.default_rng(seed)
        rows = []
        uid = 0

        def add(n, ft, weight, xlo, xhi, fe_sd, noise_sd):
            nonlocal uid
            for _ in range(n):
                x1 = rng.uniform(xlo, xhi)
                ufe = rng.normal(0.0, fe_sd)
                for t in (1, 2, 3, 4):
                    trend = 0.3 * (t - 1) + (0.5 * x1 + 0.3 * x1**2) * (t - 1)  # nonlinear Y(0)
                    treated_now = np.isfinite(ft) and t >= ft
                    y = ufe + trend + (2.0 if treated_now else 0.0) + rng.normal(0.0, noise_sd)
                    rows.append((uid, t, ft, y, x1, weight))
                uid += 1

        add(90, 2.0, 1.0, -2.0, 2.0, 0.5, 0.3)  # cohort 2
        add(90, 3.0, 1.0, -2.0, 2.0, 0.5, 0.3)  # cohort 3
        add(140, np.inf, 1.0, -2.0, 2.0, 0.5, 0.3)  # never-treated
        base = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "w"])
        n_base_rows = len(rows)
        # zero-weight padded never-treated units with EXTREME covariates (x1 ~ [40, 60])
        add(40, np.inf, 0.0, 40.0, 60.0, 3.0, 3.0)
        padded = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "w"])
        assert (padded["w"].to_numpy()[:n_base_rows] > 0).all()
        return base, padded

    def test_zero_weight_padding_leaves_att_invariant_overidentified(self):
        # Overidentified (H>1) DR survey path with the AUTO Silverman bandwidth
        # (kernel_bandwidth=None): the auto bandwidth must ignore zero-weight rows,
        # else extreme-covariate padding inflates the unweighted std, widens the
        # kernel, changes Omega*(X) and the per-unit efficient weights, and silently
        # moves ATT(g,t). This FAILS under a raw (all-row) bandwidth and passes once
        # the bandwidth is keyed off the positive-weight support.
        from diff_diff.survey import SurveyDesign

        base, padded = self._overid_panel(seed=0)

        def fit(df):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return EfficientDiD(pt_assumption="all").fit(
                    df,
                    "y",
                    "unit",
                    "time",
                    "first_treat",
                    covariates=["x1"],
                    survey_design=SurveyDesign(weights="w"),
                    aggregate="all",
                )

        r_filtered = fit(base)
        r_padded = fit(padded)
        # Staggered multi-cohort design with a never-treated comparison -> PT-All
        # cells are overidentified (the kernel/bandwidth path is live).
        assert len(r_filtered.groups) >= 2
        # Exact point-estimate invariance per (g,t) and overall.
        for k, info in r_filtered.group_time_effects.items():
            np.testing.assert_allclose(
                r_padded.group_time_effects[k]["effect"], info["effect"], atol=1e-9, rtol=0.0
            )
        np.testing.assert_allclose(
            r_padded.overall_att, r_filtered.overall_att, atol=1e-9, rtol=0.0
        )
        # SE invariant up to the same second-order survey-vcov DOF correction.
        np.testing.assert_allclose(r_padded.overall_se, r_filtered.overall_se, rtol=2e-3, atol=0.0)
