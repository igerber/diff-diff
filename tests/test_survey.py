"""Tests for survey data support (diff_diff.survey)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DifferenceInDifferences,
    MultiPeriodDiD,
    SurveyDesign,
    SurveyMetadata,
)
from diff_diff.linalg import LinearRegression, compute_robust_vcov, solve_ols
from diff_diff.survey import (
    ResolvedSurveyDesign,
    compute_survey_metadata,
    compute_survey_vcov,
)
from diff_diff.twfe import TwoWayFixedEffects
from diff_diff.utils import within_transform

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def survey_2x2_data():
    """2x2 DiD panel with survey design columns.

    100 units, 2 periods, known ATT = 3.0.
    Strata, PSU, FPC, and sampling weights are included.
    """
    np.random.seed(42)
    n_units = 100
    n_treated = 50
    rows = []
    for unit in range(n_units):
        is_treated = unit < n_treated
        stratum = unit // 20  # 5 strata (20 units each)
        psu = unit // 5  # 20 PSUs (4 per stratum, globally unique)
        fpc_val = 200.0  # Population size per stratum
        # Sampling weight proportional to stratum population / sample count
        wt = 1.0 + 0.5 * stratum

        for period in [0, 1]:
            y = 10.0 + unit * 0.1
            if period == 1:
                y += 5.0
            if is_treated and period == 1:
                y += 3.0
            y += np.random.normal(0, 0.5)

            rows.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": period,
                    "outcome": y,
                    "stratum": stratum,
                    "psu": psu,
                    "fpc": fpc_val,
                    "weight": wt,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def twfe_panel_data():
    """Balanced panel for TWFE: 50 units, 2 periods, survey columns."""
    np.random.seed(123)
    n_units = 50
    rows = []
    for unit in range(n_units):
        is_treated = unit < 25
        stratum = unit // 10  # 5 strata (10 units each)
        psu = unit // 5  # 10 PSUs (2 per stratum, globally unique)
        wt = 1.0 + 0.3 * stratum

        for period in [0, 1]:
            y = 5.0 + unit * 0.05
            if period == 1:
                y += 2.0
            if is_treated and period == 1:
                y += 4.0
            y += np.random.normal(0, 0.3)

            rows.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": period,
                    "outcome": y,
                    "stratum": stratum,
                    "psu": psu,
                    "fpc": 100.0,
                    "weight": wt,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def multiperiod_data():
    """Multi-period panel: 60 units, 5 periods, treatment at period 3."""
    np.random.seed(99)
    n_units = 60
    n_treated = 30
    periods = [1, 2, 3, 4, 5]
    rows = []
    for unit in range(n_units):
        is_treated = unit < n_treated
        stratum = unit // 20  # 3 strata (20 units each)
        psu_within = (unit % 20) // 5  # 4 PSUs within each stratum
        psu = stratum * 4 + psu_within  # globally unique PSU ID
        wt = 1.0 + 0.4 * stratum

        for t in periods:
            y = 8.0 + unit * 0.03 + t * 0.5
            if is_treated and t >= 3:
                y += 2.5  # True ATT = 2.5
            y += np.random.normal(0, 0.4)

            rows.append(
                {
                    "unit": unit,
                    "period": t,
                    "treated": int(is_treated),
                    "outcome": y,
                    "stratum": stratum,
                    "psu": psu,
                    "fpc": 200.0,
                    "weight": wt,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Tier 1: Analytical Verification (exact match, tol <= 1e-10)
# =============================================================================


class TestAnalyticalVerification:
    """Exact-match tests against manual computation."""

    def test_wls_coefficients_manual(self):
        """WLS coefficients match manual (X'WX)^{-1} X'Wy computation."""
        np.random.seed(10)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([2.0, 3.0])
        y = X @ beta_true + np.random.randn(n) * 0.5
        weights = np.abs(np.random.randn(n)) + 0.5

        # Normalise pweights to mean=1
        w = weights * (n / np.sum(weights))

        # Manual WLS: (X'WX)^{-1} X'Wy
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta_manual = np.linalg.solve(XtWX, XtWy)

        # Via solve_ols
        coef, _resid, _vcov = solve_ols(X, y, weights=w, weight_type="pweight")

        np.testing.assert_allclose(coef, beta_manual, atol=1e-10)

    def test_uniform_weights_equals_unweighted(self):
        """Uniform weights (all ones) give identical results to no weights."""
        np.random.seed(20)
        n = 80
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 2.0 * X[:, 1] + np.random.randn(n) * 0.3

        coef_unw, resid_unw, vcov_unw = solve_ols(X, y)
        coef_w, resid_w, vcov_w = solve_ols(X, y, weights=np.ones(n), weight_type="pweight")

        np.testing.assert_allclose(coef_w, coef_unw, atol=1e-10)
        np.testing.assert_allclose(resid_w, resid_unw, atol=1e-10)
        np.testing.assert_allclose(vcov_w, vcov_unw, atol=1e-10)

    def test_weight_normalization(self):
        """resolve() normalises pweights to mean=1; fweights are unchanged."""
        n = 60
        df = pd.DataFrame(
            {
                "y": np.random.randn(n),
                "pw": np.random.uniform(1, 5, n),
                "fw": np.random.choice([1, 2, 3], n).astype(float),
            }
        )

        # pweight normalisation: mean should be 1 (sum should be n)
        sd_pw = SurveyDesign(weights="pw", weight_type="pweight")
        resolved_pw = sd_pw.resolve(df)
        np.testing.assert_allclose(np.mean(resolved_pw.weights), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(resolved_pw.weights), n, atol=1e-10)

        # fweight: unchanged
        sd_fw = SurveyDesign(weights="fw", weight_type="fweight")
        resolved_fw = sd_fw.resolve(df)
        np.testing.assert_allclose(resolved_fw.weights, df["fw"].values, atol=1e-12)

    def test_scale_invariance(self):
        """Scaling all pweights by a constant c does not change coefs or SEs."""
        np.random.seed(30)
        n = 60
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 5.0 + X[:, 1] * 2.0 + np.random.randn(n) * 0.5
        base_weights = np.abs(np.random.randn(n)) + 0.5

        # After normalisation, scale is irrelevant, but let's check via
        # full DiD pipeline with DataFrames.
        df = pd.DataFrame(
            {
                "outcome": y,
                "treated": np.random.randint(0, 2, n),
                "post": np.random.randint(0, 2, n),
                "w1": base_weights,
                "w2": base_weights * 100.0,
            }
        )

        did = DifferenceInDifferences()
        sd1 = SurveyDesign(weights="w1", weight_type="pweight")
        sd2 = SurveyDesign(weights="w2", weight_type="pweight")
        r1 = did.fit(df, outcome="outcome", treatment="treated", time="post", survey_design=sd1)
        r2 = did.fit(df, outcome="outcome", treatment="treated", time="post", survey_design=sd2)

        np.testing.assert_allclose(r1.att, r2.att, atol=1e-10)
        np.testing.assert_allclose(r1.se, r2.se, atol=1e-10)

    def test_weighted_residuals(self):
        """Residuals are on the original (unweighted) scale: y - X @ beta."""
        np.random.seed(40)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 3.0 + X[:, 1] * 1.5 + np.random.randn(n) * 0.3
        w = np.abs(np.random.randn(n)) + 0.5
        w = w * (n / np.sum(w))  # normalise

        coef, resid, _vcov = solve_ols(X, y, weights=w, weight_type="pweight")

        expected_resid = y - X @ coef
        np.testing.assert_allclose(resid, expected_resid, atol=1e-10)


# =============================================================================
# Tier 2: Exact Manual Oracle Tests
# =============================================================================


class TestReferenceValues:
    """Tier 2: Exact manual oracle tests (replaces placeholder R cross-validation)."""

    def test_wls_coefficients_exact_oracle(self):
        """Verify WLS coefficients against hand-computed (X'WX)^{-1}X'Wy for n=4."""
        # 2x2 DiD design with 2 obs per cell (n=8 > k=4)
        X = np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )
        y = np.array([10.0, 10.5, 12.0, 12.5, 11.0, 11.5, 16.0, 16.5])
        w = np.array([1.0, 1.5, 2.0, 1.0, 1.0, 1.5, 2.0, 1.0])

        # Hand-compute: beta = (X'WX)^{-1} X'Wy
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta_expected = np.linalg.solve(XtWX, XtWy)

        coef, resid, vcov = solve_ols(X, y, weights=w, weight_type="pweight")
        np.testing.assert_allclose(coef, beta_expected, atol=1e-10)

    def test_weighted_hc1_vcov_exact_oracle(self):
        """Verify weighted HC1 vcov against hand-computed sandwich formula."""
        np.random.seed(42)
        X = np.column_stack([np.ones(8), np.random.randn(8)])
        y = 1.0 + 2.0 * X[:, 1] + np.random.randn(8) * 0.5
        w = np.array([1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 1.5, 0.5])

        # Normalize weights to sum=n
        w_norm = w * len(w) / np.sum(w)

        # Hand-compute WLS
        W = np.diag(w_norm)
        XtWX = X.T @ W @ X
        beta = np.linalg.solve(XtWX, X.T @ W @ y)
        u = y - X @ beta

        # Hand-compute weighted HC1 sandwich: (X'WX)^{-1} [Σ w²u² xx'] (X'WX)^{-1}
        # Score-based: s_i = w_i x_i u_i, meat = Σ s_i s_i'
        n, k = X.shape
        meat = X.T @ np.diag(w_norm**2 * u**2) @ X
        bread_inv = np.linalg.inv(XtWX)
        adjustment = n / (n - k)
        vcov_expected = adjustment * bread_inv @ meat @ bread_inv

        coef, resid, vcov = solve_ols(X, y, weights=w_norm, weight_type="pweight")
        np.testing.assert_allclose(vcov, vcov_expected, atol=1e-10)
        np.testing.assert_allclose(coef, beta, atol=1e-10)


# =============================================================================
# Tier 3: Consistency and Invariance Tests
# =============================================================================


class TestConsistencyInvariance:
    """Statistical properties and invariance checks."""

    def test_fpc_monotonicity(self):
        """Increasing FPC (larger population) weakly increases SEs.

        FPC correction = 1 - n_h/N_h. As N_h grows, the correction gets
        closer to 1 (less reduction), so SEs should weakly increase.
        """
        np.random.seed(50)
        n = 60
        strata = np.repeat(np.arange(3), 20)
        psu = np.arange(n)  # each obs is own PSU
        weights = np.ones(n)

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + X[:, 1] * 0.5 + np.random.randn(n) * 0.3

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        ses = []
        fpc_values = [25.0, 50.0, 200.0, 10000.0]
        for fpc_val in fpc_values:
            fpc_arr = np.full(n, fpc_val)
            resolved = ResolvedSurveyDesign(
                weights=weights,
                weight_type="pweight",
                strata=strata,
                psu=psu,
                fpc=fpc_arr,
                n_strata=3,
                n_psu=n,
                lonely_psu="remove",
            )
            vcov = compute_survey_vcov(X, resid, resolved)
            ses.append(np.sqrt(np.diag(vcov)))

        # Monotonically non-decreasing as FPC grows (less correction)
        for i in range(len(ses) - 1):
            assert np.all(ses[i + 1] >= ses[i] - 1e-12), (
                f"SEs did not weakly increase from FPC={fpc_values[i]} " f"to FPC={fpc_values[i+1]}"
            )

    def test_stratification_reduces_variance(self):
        """Stratification reduces variance when strata have different means."""
        np.random.seed(60)
        n = 120

        # Create data where strata have very different means
        strata = np.repeat(np.arange(3), 40)
        strata_means = np.array([0.0, 5.0, 10.0])
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = strata_means[strata] + X[:, 1] * 0.5 + np.random.randn(n) * 0.3
        psu = np.arange(n)
        weights = np.ones(n)

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        # With stratification
        resolved_strat = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=3,
            n_psu=n,
            lonely_psu="remove",
        )
        vcov_strat = compute_survey_vcov(X, resid, resolved_strat)

        # Without stratification (single implicit stratum)
        resolved_nostrat = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=n,
            lonely_psu="remove",
        )
        vcov_nostrat = compute_survey_vcov(X, resid, resolved_nostrat)

        se_strat = np.sqrt(np.diag(vcov_strat))
        se_nostrat = np.sqrt(np.diag(vcov_nostrat))

        # Stratification should reduce (or not increase) the SE of the intercept
        # because strata capture between-stratum variance
        assert se_strat[0] <= se_nostrat[0] + 1e-6

    def test_psu_equals_individual(self):
        """When each observation is its own PSU, TSL reduces to weighted HC1."""
        np.random.seed(70)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2.0 + X[:, 1] + np.random.randn(n) * 0.5
        weights = np.abs(np.random.randn(n)) + 0.5
        weights = weights * (n / np.sum(weights))
        psu = np.arange(n)  # each obs is its own PSU

        coef, resid, hc1_vcov = solve_ols(X, y, weights=weights, weight_type="pweight")

        # TSL with individual-level PSUs, no strata
        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=n,
            lonely_psu="remove",
        )
        tsl_vcov = compute_survey_vcov(X, resid, resolved)

        # Both should be finite and positive-definite
        assert np.all(np.diag(tsl_vcov) > 0)
        assert np.all(np.diag(hc1_vcov) > 0)

        # Ratio should be close to 1 (not exact due to small-sample adjustments)
        ratio = np.sqrt(np.diag(tsl_vcov)) / np.sqrt(np.diag(hc1_vcov))
        np.testing.assert_allclose(ratio, 1.0, atol=0.3)

    def test_no_strata_degeneracy(self):
        """No strata + multiple PSUs: survey vcov matches hand-computed cluster-robust sandwich."""
        np.random.seed(80)
        n = 100
        n_psu = 20
        psu = np.repeat(np.arange(n_psu), 5)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + X[:, 1] + np.random.randn(n) * 0.5
        weights = np.ones(n)

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=n_psu,
            lonely_psu="remove",
        )
        survey_vcov = compute_survey_vcov(X, resid, resolved)

        # Hand-compute cluster-robust sandwich at PSU level with weights
        # Bread: (X'WX)^{-1}
        XtWX = X.T @ (X * weights[:, np.newaxis])
        XtWX_inv = np.linalg.inv(XtWX)
        # Weighted scores: w_i * X_i * e_i
        scores = X * (weights * resid)[:, np.newaxis]
        # Aggregate to PSU level
        psu_scores = np.zeros((n_psu, X.shape[1]))
        for g in range(n_psu):
            psu_scores[g] = scores[psu == g].sum(axis=0)
        # Center and compute meat with HC1-like adjustment
        psu_mean = psu_scores.mean(axis=0, keepdims=True)
        centered = psu_scores - psu_mean
        adjustment = n_psu / (n_psu - 1)
        meat = adjustment * (centered.T @ centered)
        oracle_vcov = XtWX_inv @ meat @ XtWX_inv

        np.testing.assert_allclose(survey_vcov, oracle_vcov, atol=1e-12)

    def test_weights_only_oracle(self):
        """Weights-only design: survey vcov uses implicit-PSU TSL."""
        np.random.seed(81)
        n = 60
        raw_weights = np.random.uniform(0.5, 3.0, n)
        weights = raw_weights * (n / np.sum(raw_weights))
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2.0 + X[:, 1] * 0.5 + np.random.randn(n) * 0.3

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        survey_vcov = compute_survey_vcov(X, resid, resolved)

        # TSL with implicit per-observation PSUs:
        # meat = sum_i (s_i - s_bar)(s_i - s_bar)' * n/(n-1)
        # where s_i = w_i * X_i * e_i
        k = X.shape[1]
        XtWX = X.T @ (X * weights[:, np.newaxis])
        XtWX_inv = np.linalg.inv(XtWX)
        scores = X * (weights * resid)[:, np.newaxis]
        scores_mean = scores.mean(axis=0, keepdims=True)
        centered = scores - scores_mean
        adjustment = n / (n - 1)
        meat = adjustment * (centered.T @ centered)
        oracle_vcov = XtWX_inv @ meat @ XtWX_inv

        np.testing.assert_allclose(survey_vcov, oracle_vcov, atol=1e-12)

    def test_fweight_expansion_equivalence(self):
        """fweight=k gives same coefficients as duplicating each row k times."""
        np.random.seed(90)
        n = 30
        X_base = np.column_stack([np.ones(n), np.random.randn(n)])
        y_base = 2.0 + X_base[:, 1] * 1.5 + np.random.randn(n) * 0.3
        freq = np.random.choice([1, 2, 3], n).astype(float)

        # WLS with fweights
        coef_fw, _, _ = solve_ols(X_base, y_base, weights=freq, weight_type="fweight")

        # Expanded dataset
        X_exp = np.repeat(X_base, freq.astype(int), axis=0)
        y_exp = np.repeat(y_base, freq.astype(int))
        coef_exp, _, _ = solve_ols(X_exp, y_exp)

        np.testing.assert_allclose(coef_fw, coef_exp, atol=1e-10)

    def test_kish_effective_n(self):
        """Verify effective N = (sum w)^2 / sum(w^2) (Kish formula)."""
        raw_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n = len(raw_weights)
        dummy_resolved = ResolvedSurveyDesign(
            weights=raw_weights * (n / np.sum(raw_weights)),
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        meta = compute_survey_metadata(dummy_resolved, raw_weights)

        expected_eff_n = np.sum(raw_weights) ** 2 / np.sum(raw_weights**2)
        np.testing.assert_allclose(meta.effective_n, expected_eff_n, atol=1e-12)

        expected_deff = n * np.sum(raw_weights**2) / np.sum(raw_weights) ** 2
        np.testing.assert_allclose(meta.design_effect, expected_deff, atol=1e-12)


# =============================================================================
# Tier 4: Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration with estimators."""

    def test_did_with_survey_design(self, survey_2x2_data):
        """Full pipeline: DifferenceInDifferences with SurveyDesign."""
        did = DifferenceInDifferences()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc",
            weight_type="pweight",
        )
        result = did.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            survey_design=sd,
        )

        assert isinstance(result.att, float)
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert np.isfinite(result.t_stat)
        assert np.isfinite(result.p_value)

        # ATT should be close to 3.0 (the true effect)
        assert abs(result.att - 3.0) < 1.0

        # Survey metadata should be populated
        assert result.survey_metadata is not None
        assert isinstance(result.survey_metadata, SurveyMetadata)
        assert result.survey_metadata.weight_type == "pweight"
        assert result.survey_metadata.n_strata == 5
        assert result.survey_metadata.n_psu == 20

    def test_psu_overrides_cluster(self, survey_2x2_data):
        """When both PSU and cluster specified, PSU is used with a warning."""
        did = DifferenceInDifferences(cluster="unit")
        sd = SurveyDesign(
            weights="weight",
            psu="psu",
            weight_type="pweight",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = did.fit(
                survey_2x2_data,
                outcome="outcome",
                treatment="treated",
                time="post",
                survey_design=sd,
            )
            psu_warnings = [
                x
                for x in w
                if "PSU will be used" in str(x.message) or "survey_design.psu" in str(x.message)
            ]
            assert len(psu_warnings) >= 1

        assert np.isfinite(result.att)
        assert result.se > 0

    def test_absorb_with_weights(self, survey_2x2_data):
        """Weighted demeaning produces different results from unweighted."""
        did_w = DifferenceInDifferences()
        sd = SurveyDesign(weights="weight", weight_type="pweight")
        result_w = did_w.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            absorb=["stratum"],
            survey_design=sd,
        )

        did_uw = DifferenceInDifferences()
        result_uw = did_uw.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            absorb=["stratum"],
        )

        # ATT should be similar (both near 3.0) but not identical
        assert abs(result_w.att - 3.0) < 1.5
        assert abs(result_uw.att - 3.0) < 1.5
        # SEs should differ because of weighting
        assert result_w.se != result_uw.se

    def test_summary_output(self, survey_2x2_data):
        """Survey Design block appears in summary()."""
        did = DifferenceInDifferences()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = did.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            survey_design=sd,
        )
        summary_text = result.summary()

        assert "Survey Design" in summary_text
        assert "pweight" in summary_text
        assert "Strata:" in summary_text
        assert "PSU/Cluster:" in summary_text
        assert "Effective sample size:" in summary_text
        assert "Kish DEFF (weights):" in summary_text

    def test_to_dict_survey_fields(self, survey_2x2_data):
        """Survey metadata fields appear in to_dict()."""
        did = DifferenceInDifferences()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = did.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            survey_design=sd,
        )
        d = result.to_dict()

        assert "weight_type" in d
        assert d["weight_type"] == "pweight"
        assert "effective_n" in d
        assert d["effective_n"] > 0
        assert "design_effect" in d
        assert d["design_effect"] > 0
        assert "sum_weights" in d
        assert "n_strata" in d
        assert d["n_strata"] == 5
        assert "n_psu" in d
        assert d["n_psu"] == 20
        assert "df_survey" in d

    def test_wild_bootstrap_survey_guard(self, survey_2x2_data):
        """NotImplementedError raised when wild bootstrap + survey combined."""
        did = DifferenceInDifferences(inference="wild_bootstrap")
        sd = SurveyDesign(weights="weight", weight_type="pweight")

        with pytest.raises(NotImplementedError, match="Wild bootstrap"):
            did.fit(
                survey_2x2_data,
                outcome="outcome",
                treatment="treated",
                time="post",
                survey_design=sd,
            )

    def test_survey_design_validation_bad_weight_type(self):
        """Invalid weight_type raises ValueError."""
        with pytest.raises(ValueError, match="weight_type"):
            SurveyDesign(weights="w", weight_type="invalid")

    def test_survey_design_validation_bad_lonely_psu(self):
        """Invalid lonely_psu raises ValueError."""
        with pytest.raises(ValueError, match="lonely_psu"):
            SurveyDesign(lonely_psu="invalid")

    def test_survey_design_validation_missing_column(self):
        """Missing column in data raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sd = SurveyDesign(weights="missing_col")
        with pytest.raises(ValueError, match="not found"):
            sd.resolve(df)

    def test_survey_design_validation_zero_weights_allowed(self):
        """Zero weights are allowed (for subpopulation analysis)."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 0.0, 1.0]})
        sd = SurveyDesign(weights="w")
        resolved = sd.resolve(df)
        assert resolved.weights[1] == 0.0

    def test_survey_design_validation_all_zero_weights_rejected(self):
        """All-zero weights raise ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [0.0, 0.0, 0.0]})
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="All weights are zero"):
            sd.resolve(df)

    def test_survey_design_validation_nan_weights(self):
        """NaN weights raise ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, np.nan, 1.0]})
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="NaN"):
            sd.resolve(df)

    def test_survey_design_validation_inf_weights(self):
        """Inf weights raise ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, np.inf, 1.0]})
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="Inf"):
            sd.resolve(df)

    def test_survey_design_validation_missing_strata_column(self):
        """Missing strata column raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sd = SurveyDesign(weights="w", strata="missing")
        with pytest.raises(ValueError, match="not found"):
            sd.resolve(df)

    def test_survey_design_validation_missing_psu_column(self):
        """Missing PSU column raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sd = SurveyDesign(weights="w", psu="missing")
        with pytest.raises(ValueError, match="not found"):
            sd.resolve(df)

    def test_survey_design_validation_missing_fpc_column(self):
        """Missing FPC column raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sd = SurveyDesign(weights="w", fpc="missing")
        with pytest.raises(ValueError, match="not found"):
            sd.resolve(df)

    def test_survey_design_validation_nan_fpc(self):
        """NaN FPC values raise ValueError."""
        df = pd.DataFrame(
            {
                "y": [1, 2, 3],
                "w": [1.0, 1.0, 1.0],
                "fpc": [100.0, np.nan, 100.0],
            }
        )
        sd = SurveyDesign(weights="w", fpc="fpc")
        with pytest.raises(ValueError, match="finite"):
            sd.resolve(df)

    def test_survey_design_validation_fpc_too_small(self):
        """FPC smaller than sample size raises ValueError."""
        n = 20
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": np.repeat([0, 1], 10),
                "psu": np.arange(n),
                "fpc": np.full(n, 5.0),  # FPC=5 but n_h=10 per stratum
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", fpc="fpc")
        with pytest.raises(ValueError, match="FPC"):
            sd.resolve(df)

    def test_unstratified_fpc_must_be_constant(self):
        """Row-varying FPC without strata raises ValueError."""
        n = 20
        psu = np.repeat(np.arange(4), 5)
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "psu": psu,
                # FPC varies across rows (some 100, some 200)
                "fpc": np.where(psu < 2, 100.0, 200.0),
            }
        )
        sd = SurveyDesign(weights="w", psu="psu", fpc="fpc")
        with pytest.raises(ValueError, match="constant"):
            sd.resolve(df)

    def test_survey_design_type_error(self, survey_2x2_data):
        """Passing non-SurveyDesign object raises TypeError."""
        did = DifferenceInDifferences()
        with pytest.raises(TypeError, match="SurveyDesign"):
            did.fit(
                survey_2x2_data,
                outcome="outcome",
                treatment="treated",
                time="post",
                survey_design={"weights": "weight"},  # dict instead of SurveyDesign
            )

    def test_nest_true(self):
        """nest=True makes repeated PSU IDs unique across strata."""
        n = 40
        strata = np.repeat([0, 1], 20)
        # PSU IDs repeat across strata: 0..9 in each stratum
        psu_raw = np.tile(np.arange(10), 4)[:n]

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu_raw,
            }
        )

        # nest=False rejects repeated PSU labels across strata
        sd_no_nest = SurveyDesign(weights="w", strata="s", psu="psu", nest=False)
        with pytest.raises(ValueError, match="PSU labels.*multiple strata"):
            sd_no_nest.resolve(df)

        # nest=True makes them unique: PSU 0 in stratum 0 != PSU 0 in stratum 1
        sd_nest = SurveyDesign(weights="w", strata="s", psu="psu", nest=True)
        resolved_nest = sd_nest.resolve(df)
        assert resolved_nest.n_psu == 20  # 10 per stratum × 2 strata
        assert resolved_nest.df_survey == 18  # 20 - 2

    def test_twfe_with_survey_design(self, twfe_panel_data):
        """TwoWayFixedEffects accepts and uses survey_design."""
        twfe = TwoWayFixedEffects()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = twfe.fit(
            twfe_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            survey_design=sd,
        )

        assert isinstance(result.att, float)
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0

        # ATT should be close to 4.0
        assert abs(result.att - 4.0) < 1.5

        # Survey metadata present
        assert result.survey_metadata is not None
        assert result.survey_metadata.weight_type == "pweight"

    def test_multiperiod_with_survey_design(self, multiperiod_data):
        """MultiPeriodDiD accepts and uses survey_design."""
        mpd = MultiPeriodDiD()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = mpd.fit(
            multiperiod_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            survey_design=sd,
        )

        assert np.isfinite(result.avg_att)
        assert np.isfinite(result.avg_se)
        assert result.avg_se > 0

        # Average ATT should be close to 2.5
        assert abs(result.avg_att - 2.5) < 1.5

    def test_fweight_error_for_fractional(self):
        """Fractional fweights raise ValueError."""
        df = pd.DataFrame(
            {
                "y": [1, 2, 3],
                "w": [1.5, 2.0, 3.0],  # 1.5 is fractional
            }
        )
        sd = SurveyDesign(weights="w", weight_type="fweight")
        with pytest.raises(ValueError, match="Frequency weights.*must be non-negative integers"):
            sd.resolve(df)

    def test_lonely_psu_remove_warning(self):
        """Singleton stratum with lonely_psu='remove' emits warning."""
        n = 30
        # Stratum 2 has only 1 PSU
        strata = np.array([0] * 10 + [1] * 10 + [2] * 10)
        psu = np.array(
            list(range(5)) * 2  # stratum 0: 5 PSUs
            + list(range(5, 10)) * 2  # stratum 1: 5 PSUs
            + [10] * 10  # stratum 2: 1 PSU (singleton)
        )
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", lonely_psu="remove")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sd.resolve(df)
            lonely_warnings = [
                x for x in w if "only" in str(x.message).lower() and "PSU" in str(x.message)
            ]
            assert len(lonely_warnings) >= 1

    def test_aweight_type(self):
        """aweight type resolves and normalises correctly."""
        n = 40
        df = pd.DataFrame(
            {
                "y": np.random.randn(n),
                "w": np.random.uniform(0.5, 3.0, n),
            }
        )
        sd = SurveyDesign(weights="w", weight_type="aweight")
        resolved = sd.resolve(df)

        assert resolved.weight_type == "aweight"
        # aweights are normalised to mean=1
        np.testing.assert_allclose(np.mean(resolved.weights), 1.0, atol=1e-12)

    def test_survey_metadata_df_survey(self):
        """df_survey = n_psu - n_strata when both present."""
        n = 60
        strata = np.repeat(np.arange(3), 20)
        psu = np.repeat(np.arange(12), 5)
        weights = np.ones(n)
        weights_norm = weights * (n / np.sum(weights))

        resolved = ResolvedSurveyDesign(
            weights=weights_norm,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=3,
            n_psu=12,
            lonely_psu="remove",
        )
        meta = compute_survey_metadata(resolved, weights)

        assert meta.df_survey == 12 - 3  # n_psu - n_strata = 9

    def test_survey_metadata_df_survey_no_strata(self):
        """df_survey = n_psu - 1 when no strata."""
        n = 50
        psu = np.repeat(np.arange(10), 5)
        weights = np.ones(n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=10,
            lonely_psu="remove",
        )
        meta = compute_survey_metadata(resolved, weights)

        assert meta.df_survey == 10 - 1  # n_psu - 1 = 9

    def test_survey_metadata_weight_range(self):
        """weight_range reports (min, max) of raw weights."""
        raw_w = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
        n = len(raw_w)
        resolved = ResolvedSurveyDesign(
            weights=raw_w * (n / np.sum(raw_w)),
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        meta = compute_survey_metadata(resolved, raw_w)

        assert meta.weight_range == (1.0, 5.0)

    def test_no_weights_survey_design(self):
        """SurveyDesign without weights resolves to uniform weights."""
        n = 20
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "s": np.repeat([0, 1], 10),
            }
        )
        sd = SurveyDesign(strata="s")  # no weights
        resolved = sd.resolve(df)

        np.testing.assert_allclose(resolved.weights, np.ones(n), atol=1e-12)


# =============================================================================
# Tier 5: Monte Carlo (slow)
# =============================================================================


class TestMonteCarlo:
    """Monte Carlo coverage tests."""

    @pytest.mark.slow
    def test_survey_coverage_monte_carlo(self, ci_params):
        """95% CI coverage should be between 92% and 98% over 500 simulations.

        DGP: weighted 2x2 DiD with known ATT = 2.0, stratified sampling.
        """
        n_sims = ci_params.bootstrap(500, min_n=199)
        true_att = 2.0
        covers = 0

        for sim in range(n_sims):
            np.random.seed(sim + 1000)
            n = 80
            treated = np.array([1] * 40 + [0] * 40)
            post = np.tile([0, 1], 40)
            strata = np.repeat(np.arange(4), 20)
            psu = np.repeat(np.arange(16), 5)
            weights = 1.0 + 0.5 * strata.astype(float)

            y = (
                5.0
                + 1.0 * treated
                + 2.0 * post
                + true_att * treated * post
                + np.random.randn(n) * 1.0
            )

            df = pd.DataFrame(
                {
                    "outcome": y,
                    "treated": treated,
                    "post": post,
                    "stratum": strata,
                    "psu": psu,
                    "weight": weights,
                }
            )

            sd = SurveyDesign(
                weights="weight",
                strata="stratum",
                psu="psu",
                weight_type="pweight",
            )
            did = DifferenceInDifferences()
            try:
                result = did.fit(
                    df,
                    outcome="outcome",
                    treatment="treated",
                    time="post",
                    survey_design=sd,
                )
                if result.conf_int[0] <= true_att <= result.conf_int[1]:
                    covers += 1
            except Exception:
                # Skip failed simulations (rank deficiency, etc.)
                n_sims -= 1
                continue

        if n_sims < 30:
            pytest.skip("Too few valid simulations for coverage test")

        coverage = covers / n_sims
        # Adjust tolerance for small n_sims (pure Python mode)
        tol_low = 0.88 if n_sims < 200 else 0.92
        tol_high = 0.99 if n_sims < 200 else 0.98
        assert tol_low <= coverage <= tol_high, (
            f"95% CI coverage = {coverage:.3f} ({covers}/{n_sims}) "
            f"outside [{tol_low}, {tol_high}]"
        )


# =============================================================================
# Tier 6: P0/P1 Fix Verification Tests
# =============================================================================


class TestP0P1Fixes:
    """Verification tests for P0 and P1 fixes."""

    def test_all_singleton_strata_nan_inference(self):
        """All-singleton strata design returns NaN inference (P0-3)."""
        np.random.seed(200)
        n = 30
        # Every stratum has exactly 1 PSU -> no within-stratum variance
        strata = np.arange(n)
        psu = np.arange(n)
        weights = np.ones(n)

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + X[:, 1] * 0.5 + np.random.randn(n) * 0.3

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=n,
            n_psu=n,
            lonely_psu="remove",
        )
        vcov = compute_survey_vcov(X, resid, resolved)

        # All singleton strata -> all removed -> NaN vcov
        assert np.all(np.isnan(vcov))

    def test_fpc_validates_against_psu_count(self):
        """FPC >= n_PSU passes validation even when n_obs > FPC (P1)."""
        n = 40
        # 2 strata, 4 PSUs per stratum, 5 obs per PSU
        strata = np.repeat([0, 1], 20)
        psu = np.repeat(np.arange(8), 5)
        # FPC = 10 (>= 4 PSUs per stratum, but < 20 obs per stratum)
        fpc_arr = np.full(n, 10.0)

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu,
                "fpc": fpc_arr,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", fpc="fpc")
        # Should not raise: FPC=10 >= n_PSU=4 per stratum
        resolved = sd.resolve(df)
        assert resolved.fpc is not None

    def test_fpc_nonconstant_within_stratum(self):
        """Non-constant FPC within a stratum raises ValueError (P1)."""
        n = 20
        strata = np.repeat([0, 1], 10)
        psu = np.repeat(np.arange(4), 5)
        # FPC varies within stratum 0
        fpc_arr = np.array([100.0] * 5 + [200.0] * 5 + [150.0] * 10)

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu,
                "fpc": fpc_arr,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", fpc="fpc")
        with pytest.raises(ValueError, match="constant within each stratum"):
            sd.resolve(df)

    def test_fpc_only_without_psu_resolves(self):
        """FPC without psu or strata resolves — validated later against effective PSUs."""
        n = 10
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "fpc": np.full(n, 100.0),
            }
        )
        sd = SurveyDesign(weights="w", fpc="fpc")
        resolved = sd.resolve(df)
        assert resolved.fpc is not None

    def test_weighted_within_transform_matches_explicit_wls(self):
        """Weighted within_transform + OLS matches explicit WLS with dummies (P0-2).

        For a balanced panel with 2-way FE, the within-transformed WLS
        should yield the same treatment coefficient as WLS with explicit
        unit + time dummies.
        """
        np.random.seed(300)
        n_units = 10
        n_periods = 3
        rows = []
        for u in range(n_units):
            for t in range(n_periods):
                treated = 1 if u < 5 else 0
                post = 1 if t >= 2 else 0
                y = 5.0 + u * 0.1 + t * 0.5 + 3.0 * treated * post
                y += np.random.randn() * 0.1
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "treated": treated,
                        "post": post,
                        "outcome": y,
                    }
                )
        df = pd.DataFrame(rows)
        weights = np.array([1.0 + 0.5 * (i % 3) for i in range(len(df))])
        weights = weights * len(df) / np.sum(weights)

        # Method 1: Within-transform + OLS on treatment*post
        df["treat_post"] = df["treated"] * df["post"]
        df_wt = within_transform(df, ["outcome", "treat_post"], "unit", "time", weights=weights)
        y_wt = df_wt["outcome_demeaned"].values
        X_wt = np.column_stack([np.ones(len(df)), df_wt["treat_post_demeaned"].values])
        coef_wt, _, _ = solve_ols(X_wt, y_wt, weights=weights, weight_type="pweight")

        # Method 2: Explicit WLS with unit + time dummies
        unit_dummies = pd.get_dummies(df["unit"], prefix="u", drop_first=True)
        time_dummies = pd.get_dummies(df["time"], prefix="t", drop_first=True)
        X_full = np.column_stack(
            [
                np.ones(len(df)),
                df["treat_post"].values,
                unit_dummies.values,
                time_dummies.values,
            ]
        )
        y_full = df["outcome"].values
        coef_full, _, _ = solve_ols(X_full, y_full, weights=weights, weight_type="pweight")

        # Treatment coefficient should match
        np.testing.assert_allclose(coef_wt[1], coef_full[1], atol=1e-6)

    def test_multiperiod_survey_metadata_populated(self, multiperiod_data):
        """MultiPeriodDiD populates survey_metadata in results (P2-1)."""
        mpd = MultiPeriodDiD()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = mpd.fit(
            multiperiod_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            survey_design=sd,
        )

        assert result.survey_metadata is not None
        assert isinstance(result.survey_metadata, SurveyMetadata)
        assert result.survey_metadata.weight_type == "pweight"
        assert result.survey_metadata.n_strata == 3
        assert result.survey_metadata.n_psu > 0  # Varies with fixture PSU structure

        # Survey info should appear in summary
        summary_text = result.summary()
        assert "Survey Design" in summary_text
        assert "pweight" in summary_text

        # Survey info should appear in to_dict
        d = result.to_dict()
        assert "weight_type" in d
        assert d["weight_type"] == "pweight"
        assert "effective_n" in d


# =============================================================================
# P1-1 / P1-2 regression tests (PR #218 review round 3)
# =============================================================================


class TestFweightInference:
    """Verify fweight df = sum(w) - k throughout the stack."""

    def test_fweight_se_matches_expanded_oracle(self):
        """Fweight SEs must match unweighted OLS on frequency-expanded data."""
        np.random.seed(42)
        n = 30
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.5
        fw = np.random.choice([1, 2, 3], size=n)

        # Oracle: expand each row fw[i] times, run unweighted OLS
        X_exp = np.repeat(X, fw, axis=0)
        y_exp = np.repeat(y, fw)
        coef_exp, _, vcov_exp = solve_ols(X_exp, y_exp)
        se_exp = np.sqrt(np.diag(vcov_exp))

        # Fweight path: compressed data with integer weights
        coef_fw, _, vcov_fw = solve_ols(X, y, weights=fw.astype(float), weight_type="fweight")
        se_fw = np.sqrt(np.diag(vcov_fw))

        np.testing.assert_allclose(coef_fw, coef_exp, atol=1e-10)
        np.testing.assert_allclose(se_fw, se_exp, atol=1e-10)

    def test_linear_regression_fweight_df(self):
        """LinearRegression with fweight must set df_ = sum(w) - k."""
        np.random.seed(42)
        n = 30
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.5
        fw = np.random.choice([1, 2, 3], size=n)

        model = LinearRegression(
            weights=fw.astype(float),
            weight_type="fweight",
            include_intercept=False,
        )
        model.fit(X, y)

        k = X.shape[1]
        expected_df = int(np.sum(fw)) - k
        assert model.df_ == expected_df


class TestWeightedRankDeficiency:
    """Weighted rank-deficient fits must not produce all-NaN residuals."""

    def test_weighted_rank_deficient_solver_finite_residuals(self):
        """solve_ols with weights + duplicate column: residuals must be finite."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        # Duplicate column -> rank-deficient
        X = np.column_stack([np.ones(n), x1, x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            coef, resid, vcov = solve_ols(
                X,
                y,
                weights=pw,
                weight_type="pweight",
                rank_deficient_action="warn",
            )

        # Exactly one coefficient should be NaN (dropped duplicate)
        assert np.sum(np.isnan(coef)) == 1

        # Residuals must all be finite (not all-NaN)
        assert np.all(np.isfinite(resid)), "Residuals must not be all-NaN"

        # Vcov: NaN rows/cols only for the dropped column
        nan_col = np.where(np.isnan(coef))[0][0]
        assert np.all(np.isnan(vcov[nan_col, :])), "Dropped col vcov row should be NaN"
        assert np.all(np.isnan(vcov[:, nan_col])), "Dropped col vcov col should be NaN"

        # Identified coefficients should have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0

    def test_linear_regression_weighted_rank_deficient_robust(self):
        """LinearRegression with weights + robust + rank deficiency: finite residuals."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)

        model = LinearRegression(
            weights=pw,
            weight_type="pweight",
            robust=True,
            include_intercept=False,
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X, y)

        coef = model.coefficients_
        resid = model.residuals_
        vcov = model.vcov_

        # One dropped coefficient
        assert np.sum(np.isnan(coef)) == 1

        # Residuals all finite
        assert np.all(np.isfinite(resid))

        # Identified coefficients have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0

    def test_fweight_survey_oracle(self):
        """fweight SurveyDesign: survey vcov uses implicit-PSU TSL."""
        np.random.seed(55)
        n = 30
        X_base = np.column_stack([np.ones(n), np.random.randn(n)])
        y_base = 2.0 + X_base[:, 1] * 1.5 + np.random.randn(n) * 0.3
        freq = np.random.choice([1, 2, 3], n).astype(float)

        # WLS with fweights via survey
        coef_fw, resid_fw, _ = solve_ols(X_base, y_base, weights=freq, weight_type="fweight")
        resolved = ResolvedSurveyDesign(
            weights=freq,
            weight_type="fweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        survey_vcov = compute_survey_vcov(X_base, resid_fw, resolved)

        # Oracle: TSL with implicit per-observation PSUs
        # scores = w_i * X_i * e_i, meat = n/(n-1) * (scores - mean)' (scores - mean)
        k = X_base.shape[1]
        XtWX = X_base.T @ (X_base * freq[:, np.newaxis])
        XtWX_inv = np.linalg.inv(XtWX)
        scores = X_base * (freq * resid_fw)[:, np.newaxis]
        scores_mean = scores.mean(axis=0, keepdims=True)
        centered = scores - scores_mean
        adjustment = n / (n - 1)
        meat = adjustment * (centered.T @ centered)
        oracle_vcov = XtWX_inv @ meat @ XtWX_inv

        np.testing.assert_allclose(survey_vcov, oracle_vcov, atol=1e-10)

    def test_survey_rank_deficient_with_psu(self):
        """LinearRegression + survey design (PSU) + rank deficiency: no crash."""
        np.random.seed(43)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x1])  # duplicate col
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)
        psu = np.arange(n)  # each obs is its own PSU

        resolved = ResolvedSurveyDesign(
            weights=pw,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=n,
            lonely_psu="remove",
        )

        model = LinearRegression(
            survey_design=resolved,
            include_intercept=False,
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X, y)

        coef = model.coefficients_
        resid = model.residuals_
        vcov = model.vcov_

        # One dropped coefficient
        assert np.sum(np.isnan(coef)) == 1

        # Residuals all finite
        assert np.all(np.isfinite(resid))

        # Identified coefficients have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0

        # Dropped column has NaN vcov
        dropped = np.where(np.isnan(coef))[0]
        for i in dropped:
            assert np.all(np.isnan(vcov[i, :]))
            assert np.all(np.isnan(vcov[:, i]))

    def test_survey_rank_deficient_weights_only(self):
        """Weights-only survey + rank deficiency: no crash, correct NaN pattern."""
        np.random.seed(44)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x1])  # duplicate col
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)

        resolved = ResolvedSurveyDesign(
            weights=pw,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        model = LinearRegression(
            survey_design=resolved,
            include_intercept=False,
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X, y)

        coef = model.coefficients_
        resid = model.residuals_
        vcov = model.vcov_

        # One dropped coefficient
        assert np.sum(np.isnan(coef)) == 1

        # Residuals all finite
        assert np.all(np.isfinite(resid))

        # Identified coefficients have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0

        # Dropped column has NaN vcov
        dropped = np.where(np.isnan(coef))[0]
        for i in dropped:
            assert np.all(np.isnan(vcov[i, :]))
            assert np.all(np.isnan(vcov[:, i]))

    def test_linear_regression_weighted_rank_deficient_classical(self):
        """LinearRegression with weights + classical vcov + rank deficiency."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)

        model = LinearRegression(
            weights=pw,
            weight_type="pweight",
            robust=False,
            include_intercept=False,
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X, y)

        coef = model.coefficients_
        resid = model.residuals_
        vcov = model.vcov_

        # One dropped coefficient
        assert np.sum(np.isnan(coef)) == 1

        # Residuals all finite
        assert np.all(np.isfinite(resid))

        # Identified coefficients have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0


# =============================================================================
# Round 5 Fixes (PR #218)
# =============================================================================


class TestRound5Fixes:
    """Tests for P1-A (implicit PSU df), P1-B (lonely_psu unstratified),
    P1-C (LinearRegression weight auto-derivation)."""

    def test_weights_only_survey_df(self):
        """P1-A: weights-only design uses n_obs - 1 for df."""
        n = 50
        weights = np.random.uniform(0.5, 3.0, size=n)
        weights = weights * (n / np.sum(weights))  # normalize

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        assert resolved.df_survey == n - 1

        # Metadata should also report implicit n_psu
        raw_w = np.random.uniform(0.5, 3.0, size=n)
        meta = compute_survey_metadata(resolved, raw_w)
        assert meta.n_psu == n

    def test_stratified_no_psu_survey_df(self):
        """P1-A: stratified-no-PSU design uses n_obs - n_strata for df."""
        n = 60
        n_strata = 3
        weights = np.ones(n, dtype=np.float64)
        strata = np.repeat(np.arange(n_strata), n // n_strata)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=None,
            fpc=None,
            n_strata=n_strata,
            n_psu=0,
            lonely_psu="remove",
        )

        assert resolved.df_survey == n - n_strata

    def test_single_psu_unstratified_lonely_psu_remove(self):
        """P1-B: single PSU unstratified, lonely_psu='remove' -> NaN vcov."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 0.5 * X[:, 1] + np.random.randn(n) * 0.3
        weights = np.ones(n, dtype=np.float64)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=np.zeros(n, dtype=int),  # all same PSU
            fpc=None,
            n_strata=0,
            n_psu=1,
            lonely_psu="remove",
        )

        vcov = compute_survey_vcov(X, residuals, resolved)
        assert np.all(np.isnan(vcov))

    def test_single_psu_unstratified_lonely_psu_certainty(self):
        """P1-B: single PSU unstratified, lonely_psu='certainty' -> NaN vcov."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 0.5 * X[:, 1] + np.random.randn(n) * 0.3
        weights = np.ones(n, dtype=np.float64)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=np.zeros(n, dtype=int),
            fpc=None,
            n_strata=0,
            n_psu=1,
            lonely_psu="certainty",
        )

        vcov = compute_survey_vcov(X, residuals, resolved)
        assert np.all(np.isnan(vcov))

    def test_single_psu_unstratified_lonely_psu_adjust(self):
        """P1-B: single PSU unstratified, lonely_psu='adjust' -> NaN vcov."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 0.5 * X[:, 1] + np.random.randn(n) * 0.3
        weights = np.ones(n, dtype=np.float64)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=np.zeros(n, dtype=int),
            fpc=None,
            n_strata=0,
            n_psu=1,
            lonely_psu="adjust",
        )

        vcov = compute_survey_vcov(X, residuals, resolved)
        assert np.all(np.isnan(vcov))

    def test_linear_regression_auto_derives_weights_from_survey(self):
        """P1-C: LinearRegression auto-derives weights from survey_design."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.5
        weights = np.random.uniform(0.5, 3.0, size=n)
        weights_norm = weights * (n / np.sum(weights))

        psu = np.repeat(np.arange(10), n // 10)

        resolved = ResolvedSurveyDesign(
            weights=weights_norm,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=10,
            lonely_psu="remove",
        )

        # Explicit weights path
        model_explicit = LinearRegression(
            weights=weights_norm,
            weight_type="pweight",
            robust=True,
            survey_design=resolved,
        )
        X = np.column_stack([np.ones(n), x1])
        model_explicit.fit(X, y)

        # Auto-derive path: no explicit weights
        model_auto = LinearRegression(
            robust=True,
            survey_design=resolved,
        )
        model_auto.fit(X, y)

        # Weights should be populated after fit
        assert model_auto.weights is not None

        # Coefficients should match
        np.testing.assert_allclose(
            model_auto.coefficients_, model_explicit.coefficients_, rtol=1e-10
        )

        # Vcov should match
        np.testing.assert_allclose(model_auto.vcov_, model_explicit.vcov_, rtol=1e-10)

    def test_resolve_warns_single_psu_unstratified(self):
        """P1-B: SurveyDesign.resolve() warns for single PSU unstratified."""
        n = 10
        df = pd.DataFrame(
            {
                "w": np.ones(n),
                "psu_col": np.zeros(n, dtype=int),
                "y": np.random.randn(n),
            }
        )

        for mode in ["remove", "certainty", "adjust"]:
            design = SurveyDesign(
                weights="w",
                psu="psu_col",
                lonely_psu=mode,
            )
            with pytest.warns(UserWarning, match=r"Only 1 PSU"):
                design.resolve(df)


class TestRound6Fixes:
    """Tests for round-6 review fixes (PR #218)."""

    def test_weights_only_matches_explicit_individual_psu(self):
        """Weights-only design produces identical vcov to explicit psu=arange(n)."""
        np.random.seed(600)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + X[:, 1] * 0.5 + np.random.randn(n) * 0.4
        weights = np.random.uniform(0.5, 3.0, n)
        weights = weights * (n / np.sum(weights))

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        # Weights-only design (no PSU)
        resolved_wo = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        vcov_wo = compute_survey_vcov(X, resid, resolved_wo)

        # Explicit individual-PSU design
        resolved_psu = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=np.arange(n),
            fpc=None,
            n_strata=0,
            n_psu=n,
            lonely_psu="remove",
        )
        vcov_psu = compute_survey_vcov(X, resid, resolved_psu)

        np.testing.assert_allclose(vcov_wo, vcov_psu, atol=1e-12)

    def test_conflicting_weights_warns_and_uses_survey(self):
        """Explicit weights differing from survey_design triggers warning."""
        np.random.seed(601)
        n = 40
        X = np.random.randn(n, 2)
        y = X @ [1.0, 0.5] + np.random.randn(n) * 0.3
        survey_weights = np.random.uniform(0.5, 3.0, n)
        different_weights = np.random.uniform(1.0, 5.0, n)

        resolved = ResolvedSurveyDesign(
            weights=survey_weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        # Fit with conflicting explicit weights — should warn
        reg_conflict = LinearRegression(weights=different_weights, survey_design=resolved)
        with pytest.warns(UserWarning, match="differ from survey_design"):
            reg_conflict.fit(X, y)

        # Fit with only survey_design — no explicit weights
        reg_survey = LinearRegression(survey_design=resolved)
        reg_survey.fit(X, y)

        np.testing.assert_allclose(reg_conflict.coefficients_, reg_survey.coefficients_, atol=1e-14)
        np.testing.assert_allclose(reg_conflict.vcov_, reg_survey.vcov_, atol=1e-14)

    def test_matching_weights_no_warning(self):
        """Same array object passed as weights and in survey_design: no warning."""
        np.random.seed(602)
        n = 40
        X = np.random.randn(n, 2)
        y = X @ [1.0, 0.5] + np.random.randn(n) * 0.3
        weights = np.random.uniform(0.5, 3.0, n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        # Same array object — should NOT warn
        reg = LinearRegression(weights=weights, survey_design=resolved)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reg.fit(X, y)


class TestRound7Fixes:
    """Tests for round-7 review fixes (PR #218)."""

    @staticmethod
    def _make_cluster_data(seed=700):
        """Create 2-period DiD data with 10 clusters of 5 obs each."""
        np.random.seed(seed)
        n_clusters = 10
        obs_per_cluster = 5
        rows = []
        for c in range(n_clusters):
            is_treated = c >= 5
            for i in range(obs_per_cluster):
                for period in [0, 1]:
                    y = 10.0 + c * 0.3 + np.random.randn() * 0.5
                    if period == 1 and is_treated:
                        y += 3.0
                    rows.append(
                        {
                            "unit": c * obs_per_cluster + i,
                            "period": period,
                            "treated": int(is_treated),
                            "y": y,
                            "cluster_id": c,
                            "w": 1.0 + 0.2 * c,
                        }
                    )
        return pd.DataFrame(rows)

    def test_cluster_injected_as_psu_did(self):
        """Cluster IDs injected as PSU produce identical SEs to explicit PSU."""
        data = self._make_cluster_data()

        # Fit with cluster= and weights-only survey (no PSU)
        result_inject = DifferenceInDifferences(cluster="cluster_id").fit(
            data,
            "y",
            "treated",
            "period",
            survey_design=SurveyDesign(weights="w"),
        )

        # Fit with explicit PSU in survey design
        result_explicit = DifferenceInDifferences(cluster="cluster_id").fit(
            data,
            "y",
            "treated",
            "period",
            survey_design=SurveyDesign(weights="w", psu="cluster_id"),
        )

        np.testing.assert_allclose(result_inject.se, result_explicit.se, atol=1e-12)
        assert result_inject.survey_metadata.n_psu == 10
        assert result_inject.survey_metadata.df_survey == 9

    def test_cluster_injected_as_psu_twfe(self):
        """TWFE: cluster IDs injected as PSU produce identical SEs to explicit PSU."""
        data = self._make_cluster_data()

        result_inject = TwoWayFixedEffects(cluster="cluster_id").fit(
            data,
            "y",
            "treated",
            "period",
            unit="unit",
            survey_design=SurveyDesign(weights="w"),
        )

        result_explicit = TwoWayFixedEffects(cluster="cluster_id").fit(
            data,
            "y",
            "treated",
            "period",
            unit="unit",
            survey_design=SurveyDesign(weights="w", psu="cluster_id"),
        )

        np.testing.assert_allclose(result_inject.se, result_explicit.se, atol=1e-12)
        assert result_inject.survey_metadata.n_psu == 10
        assert result_inject.survey_metadata.df_survey == 9

    def test_cluster_injected_as_psu_linear_regression(self):
        """Standalone LinearRegression: cluster injection matches explicit PSU."""
        np.random.seed(701)
        n = 50
        cluster_ids = np.repeat(np.arange(10), 5)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + X[:, 1] * 0.5 + np.random.randn(n) * 0.4
        weights = np.random.uniform(0.5, 3.0, n)

        # No PSU in resolved design
        resolved_no_psu = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        reg_inject = LinearRegression(
            include_intercept=False,
            cluster_ids=cluster_ids,
            survey_design=resolved_no_psu,
        )
        reg_inject.fit(X, y)

        # Explicit PSU
        codes, uniques = pd.factorize(cluster_ids)
        resolved_psu = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=codes,
            fpc=None,
            n_strata=0,
            n_psu=len(uniques),
            lonely_psu="remove",
        )
        reg_explicit = LinearRegression(
            include_intercept=False,
            cluster_ids=cluster_ids,
            survey_design=resolved_psu,
        )
        reg_explicit.fit(X, y)

        np.testing.assert_allclose(reg_inject.vcov_, reg_explicit.vcov_, atol=1e-12)

    def test_cluster_injection_no_effect_when_psu_present(self):
        """When PSU is already present, _inject_cluster_as_psu is a no-op."""
        from diff_diff.survey import _inject_cluster_as_psu

        existing_psu = np.array([0, 0, 1, 1, 2, 2])
        resolved = ResolvedSurveyDesign(
            weights=np.ones(6),
            weight_type="pweight",
            strata=None,
            psu=existing_psu,
            fpc=None,
            n_strata=0,
            n_psu=3,
            lonely_psu="remove",
        )
        result = _inject_cluster_as_psu(resolved, np.array([10, 10, 20, 20, 30, 30]))
        assert result is resolved  # Same object — no replacement

    def test_invalid_weight_type_raises(self):
        """Invalid weight_type raises ValueError in solve_ols and LinearRegression."""
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        w = np.ones(n)

        with pytest.raises(ValueError, match="weight_type must be one of"):
            solve_ols(X, y, weights=w, weight_type="pwieght")

        with pytest.raises(ValueError, match="weight_type must be one of"):
            LinearRegression(weights=w, weight_type="bad").fit(X, y)

    def test_nan_weights_raises(self):
        """NaN weights raise ValueError."""
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        w = np.ones(n)
        w[5] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            solve_ols(X, y, weights=w)

    def test_negative_weights_raises(self):
        """Negative weights raise ValueError."""
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        w = np.ones(n)
        w[3] = -0.5

        with pytest.raises(ValueError, match="non-negative"):
            solve_ols(X, y, weights=w)

    def test_inf_weights_raises(self):
        """Inf weights raise ValueError."""
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        w = np.ones(n)
        w[0] = np.inf

        with pytest.raises(ValueError, match="Inf"):
            solve_ols(X, y, weights=w)

    def test_zero_weights_accepted(self):
        """Zero weights are accepted (intentional divergence from SurveyDesign)."""
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        w = np.ones(n)
        w[0] = 0.0

        # Should NOT raise
        coef, resid, vcov = solve_ols(X, y, weights=w)
        assert coef is not None


class TestRound8Fixes:
    """Tests for round-8 review fixes (PR #218)."""

    def test_weighted_hc1_cluster_consistency(self):
        """Weighted HC1 SEs match cluster-robust SEs when each obs is its own cluster.

        When cluster_ids=np.arange(n), the cluster-robust estimator reduces to HC1
        because G=n makes the small-sample adjustments identical:
        HC1: n/(n-k), Cluster: (n/(n-1))*((n-1)/(n-k)) = n/(n-k).
        """
        np.random.seed(801)
        n = 30
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 2.0 * X[:, 1] + np.random.randn(n) * 0.5
        w = np.abs(np.random.randn(n)) + 0.1

        # HC1 path
        _, resid, vcov_hc1 = solve_ols(X, y, weights=w, weight_type="pweight")

        # Cluster path with each obs as its own cluster
        vcov_cluster = compute_robust_vcov(
            X, resid, cluster_ids=np.arange(n), weights=w, weight_type="pweight"
        )

        np.testing.assert_allclose(vcov_hc1, vcov_cluster, atol=1e-12)

    def test_compute_robust_vcov_invalid_weight_type(self):
        """Invalid weight_type raises ValueError."""
        np.random.seed(802)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        resid = np.random.randn(n)
        w = np.ones(n)

        with pytest.raises(ValueError, match="weight_type"):
            compute_robust_vcov(X, resid, weights=w, weight_type="bad")

    def test_compute_robust_vcov_nan_weights(self):
        """NaN weights raise ValueError via compute_robust_vcov."""
        np.random.seed(803)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        resid = np.random.randn(n)
        w = np.ones(n)
        w[2] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            compute_robust_vcov(X, resid, weights=w)

    def test_compute_robust_vcov_inf_weights(self):
        """Inf weights raise ValueError via compute_robust_vcov."""
        np.random.seed(804)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        resid = np.random.randn(n)
        w = np.ones(n)
        w[0] = np.inf

        with pytest.raises(ValueError, match="Inf"):
            compute_robust_vcov(X, resid, weights=w)

    def test_compute_robust_vcov_negative_weights(self):
        """Negative weights raise ValueError via compute_robust_vcov."""
        np.random.seed(805)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        resid = np.random.randn(n)
        w = np.ones(n)
        w[3] = -0.5

        with pytest.raises(ValueError, match="non-negative"):
            compute_robust_vcov(X, resid, weights=w)

    def test_fweight_df_rounding(self):
        """Near-integer fweights use rounded (not truncated) sum for df."""
        np.random.seed(806)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        # Near-integer weights that would truncate incorrectly
        w = np.full(n, 2.0 - 1e-14)

        reg = LinearRegression(weights=w, weight_type="fweight", include_intercept=False)
        reg.fit(X, y)
        # sum(w) ≈ 20 - 1e-13; round → 20, truncate → 19
        assert reg.df_ == 20 - reg.n_params_effective_

        # Also verify compute_robust_vcov path produces valid output
        _, resid, _ = solve_ols(X, y, weights=w, weight_type="fweight")
        vcov = compute_robust_vcov(X, resid, weights=w, weight_type="fweight")
        assert np.all(np.isfinite(vcov))
        assert np.all(np.diag(vcov) > 0)


class TestRound9Fixes:
    """Tests for round-9 review fixes (PR #218)."""

    def test_did_absorb_matches_explicit_wls_dummies(self):
        """Absorbed WLS matches explicit WLS with dummy variables (P0).

        Uses absorb=["region"] (cross-cuts treatment) so all regressors
        survive demeaning, giving a clean equivalence test.
        """
        np.random.seed(901)
        n_units = 10
        n_periods = 2
        rows = []
        for u in range(n_units):
            for t in range(n_periods):
                treated = 1 if u < 5 else 0
                post = 1 if t >= 1 else 0
                region = u % 3  # cross-cuts treatment
                y = 5.0 + region * 0.5 + t * 0.5 + 3.0 * treated * post
                y += np.random.randn() * 0.1
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "treated": treated,
                        "post": post,
                        "outcome": y,
                        "region": region,
                    }
                )
        df = pd.DataFrame(rows)
        weights = np.array([1.0 + 0.5 * (i % 3) for i in range(len(df))])
        weights = weights * len(df) / np.sum(weights)
        df["weight"] = weights

        sd = SurveyDesign(weights="weight", weight_type="pweight")

        # Method 1: absorb region FE
        did_absorb = DifferenceInDifferences()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result_absorb = did_absorb.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="post",
                absorb=["region"],
                survey_design=sd,
            )

        # Method 2: explicit region dummies
        did_explicit = DifferenceInDifferences()
        result_explicit = did_explicit.fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="post",
            fixed_effects=["region"],
            survey_design=sd,
        )

        np.testing.assert_allclose(result_absorb.att, result_explicit.att, atol=1e-6)

    def test_multiperiod_absorb_matches_explicit_wls_dummies(self):
        """MultiPeriodDiD absorbed WLS matches explicit WLS with dummies (P0)."""
        np.random.seed(902)
        n_units = 10
        periods = [1, 2, 3, 4]
        rows = []
        for u in range(n_units):
            for t in periods:
                treated = 1 if u < 5 else 0
                post = 1 if t >= 3 else 0
                region = u % 3
                y = 5.0 + region * 0.5 + t * 0.3 + 2.0 * treated * post
                y += np.random.randn() * 0.1
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "treated": treated,
                        "post": post,
                        "outcome": y,
                        "region": region,
                    }
                )
        df = pd.DataFrame(rows)
        weights = np.array([1.0 + 0.3 * (i % 4) for i in range(len(df))])
        weights = weights * len(df) / np.sum(weights)
        df["weight"] = weights

        sd = SurveyDesign(weights="weight", weight_type="pweight")

        # Method 1: absorb region FE
        mpd_absorb = MultiPeriodDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_absorb = mpd_absorb.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="time",
                post_periods=[3, 4],
                absorb=["region"],
                survey_design=sd,
            )

        # Method 2: explicit region dummies
        mpd_explicit = MultiPeriodDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_explicit = mpd_explicit.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="time",
                post_periods=[3, 4],
                fixed_effects=["region"],
                survey_design=sd,
            )

        np.testing.assert_allclose(result_absorb.avg_att, result_explicit.avg_att, atol=1e-6)

    def test_fractional_fweight_rejected_solve_ols(self):
        """Fractional fweights raise ValueError via solve_ols."""
        np.random.seed(903)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        w = np.array([1.5, 2.3, 1.0, 2.0, 1.7, 3.0, 1.0, 2.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="non-negative integers"):
            solve_ols(X, y, weights=w, weight_type="fweight")

    def test_fractional_fweight_rejected_compute_robust_vcov(self):
        """Fractional fweights raise ValueError via compute_robust_vcov."""
        np.random.seed(904)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        resid = np.random.randn(n)
        w = np.array([1.5, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="non-negative integers"):
            compute_robust_vcov(X, resid, weights=w, weight_type="fweight")

    def test_integer_fweight_accepted(self):
        """Integer fweights as float are accepted without error."""
        np.random.seed(905)
        n = 10
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        w = np.array([2.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0])

        # Should NOT raise
        coef, resid, vcov = solve_ols(X, y, weights=w, weight_type="fweight")
        assert coef is not None
        assert vcov is not None

    def test_all_certainty_psu_zero_vcov(self):
        """All-certainty-PSU stratified design returns zero vcov, not NaN (P1-B)."""
        np.random.seed(906)
        n = 30
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 2.0 * X[:, 1] + np.random.randn(n) * 0.5
        w = np.ones(n)
        w_norm = w * n / np.sum(w)

        # Each stratum has exactly 1 PSU → all certainty
        strata = np.arange(n)  # 30 strata, 1 obs each
        psu = np.arange(n)

        coef, resid, _ = solve_ols(X, y, weights=w_norm, weight_type="pweight")

        resolved = ResolvedSurveyDesign(
            weights=w_norm,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            lonely_psu="certainty",
            n_psu=n,
            n_strata=n,
        )
        vcov = compute_survey_vcov(X, resid, resolved)

        # Should be zero (not NaN) — every stratum is certainty
        assert np.all(vcov == 0.0)
        assert not np.any(np.isnan(vcov))

    def test_multiperiod_fweight_df_rounding(self):
        """MultiPeriodDiD uses rounded (not truncated) fweight df."""
        np.random.seed(907)
        n_units = 10
        periods = [1, 2, 3]
        rows = []
        for u in range(n_units):
            for t in periods:
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "treated": 1 if u < 5 else 0,
                        "post": 1 if t >= 3 else 0,
                        "outcome": np.random.randn(),
                        "fw": 2,  # integer fweight
                    }
                )
        df = pd.DataFrame(rows)

        sd = SurveyDesign(weights="fw", weight_type="fweight")
        mpd = MultiPeriodDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mpd.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="time",
                post_periods=[3],
                survey_design=sd,
            )

        # Should produce valid results (not crash on df truncation)
        assert np.isfinite(result.avg_att)
        assert np.isfinite(result.avg_se)
        assert result.avg_se > 0


class TestRound10Fixes:
    """Tests for PR #218 review round 10 fixes."""

    def test_zero_se_estimator_nan_inference(self):
        """Zero-SE path in LinearRegression.get_inference() returns NaN, not ±inf."""
        # Build a design where all strata are certainty PSUs → zero vcov → zero SE
        np.random.seed(42)
        n = 40
        strata = np.repeat([0, 1, 2, 3], 10)
        psu = strata.copy()  # 1 PSU per stratum → all certainty
        df = pd.DataFrame(
            {
                "outcome": np.random.randn(n),
                "treated": np.array([1] * 20 + [0] * 20),
                "post": np.tile([0, 1], 20),
                "w": np.ones(n),
                "strat": strata,
                "cluster": psu,
            }
        )
        sd = SurveyDesign(
            weights="w",
            weight_type="pweight",
            strata="strat",
            psu="cluster",
            lonely_psu="certainty",
        )
        did = DifferenceInDifferences()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = did.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="post",
                survey_design=sd,
            )
        # SE should be 0 (all certainty strata), inference should be NaN
        assert result.se == 0.0
        assert np.isnan(result.t_stat)
        assert np.isnan(result.p_value)
        assert np.isnan(result.conf_int[0])
        assert np.isnan(result.conf_int[1])

    def test_full_census_fpc_stratified_zero_vcov(self):
        """Full-census FPC (f_h=1) returns zero vcov, not NaN."""
        np.random.seed(42)
        n = 60
        strata = np.repeat([0, 1, 2], 20)
        psu = np.tile(np.arange(5), 12)  # 5 PSUs per stratum

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        residuals = np.random.randn(n)
        weights = np.ones(n)

        # FPC = n_psu per stratum (full census: f_h = 5/5 = 1)
        fpc = np.array([5.0] * n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=fpc,
            n_strata=3,
            n_psu=15,
            lonely_psu="remove",
        )
        vcov = compute_survey_vcov(X, residuals, resolved=resolved)
        # Full census → zero variance → zero vcov
        np.testing.assert_array_equal(vcov, np.zeros((2, 2)))

    def test_full_census_fpc_unstratified_zero_vcov(self):
        """Unstratified full-census FPC returns zero vcov, not NaN."""
        np.random.seed(42)
        n = 30
        psu = np.repeat(np.arange(6), 5)  # 6 PSUs

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        residuals = np.random.randn(n)
        weights = np.ones(n)

        # FPC = n_psu (full census: f_h = 6/6 = 1)
        fpc = np.array([6.0] * n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=fpc,
            n_strata=0,
            n_psu=6,
            lonely_psu="remove",
        )
        vcov = compute_survey_vcov(X, residuals, resolved=resolved)
        # Full census → (1-f_h)=0 → zero meat → zero vcov
        np.testing.assert_array_equal(vcov, np.zeros((2, 2)))

    def test_absorbed_did_sample_counts(self):
        """n_treated/n_control reflect raw data, not demeaned values after absorb."""
        np.random.seed(42)
        n_units = 20
        n_times = 4
        rows = []
        for u in range(n_units):
            for t in range(n_times):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "treated": 1 if u < 8 else 0,
                        "post": 1 if t >= 2 else 0,
                        "outcome": np.random.randn(),
                        "region": u % 3,
                    }
                )
        df = pd.DataFrame(rows)

        did = DifferenceInDifferences()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = did.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="post",
                absorb=["region"],
            )

        # Raw counts: 8 treated units * 4 times = 32 treated obs
        raw_treated = int(df["treated"].sum())
        raw_control = len(df) - raw_treated
        assert result.n_treated == raw_treated
        assert result.n_control == raw_control


class TestRound11Fixes:
    """Tests for PR #218 review round 11 fixes."""

    def test_repeated_fit_fresh_psu(self):
        """Repeated LinearRegression.fit() uses fresh PSU, not stale from prior fit."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        weights = np.ones(n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        lr = LinearRegression(survey_design=resolved, robust=True)

        # First fit: 2 clusters → survey_df = 2 - 1 = 1
        cluster_1 = np.array([0] * 10 + [1] * 10)
        lr.fit(X, y, cluster_ids=cluster_1)
        assert lr.survey_df_ == 1

        # Second fit: 4 clusters → survey_df = 4 - 1 = 3
        cluster_2 = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
        lr.fit(X, y, cluster_ids=cluster_2)
        assert lr.survey_df_ == 3

        # Original survey_design must be immutable
        assert lr.survey_design.psu is None

    def test_multi_absorb_survey_rejected_did(self):
        """DiD with multi-absorb + survey weights raises ValueError."""
        np.random.seed(42)
        n = 40
        df = pd.DataFrame(
            {
                "outcome": np.random.randn(n),
                "treated": np.array([1] * 20 + [0] * 20),
                "post": np.tile([0, 1], 20),
                "w": np.ones(n),
                "a": np.repeat(np.arange(4), 10),
                "b": np.tile(np.arange(5), 8),
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight")
        did = DifferenceInDifferences()
        with pytest.raises(ValueError, match="Multiple absorbed fixed effects"):
            did.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="post",
                absorb=["a", "b"],
                survey_design=sd,
            )

    def test_multi_absorb_survey_rejected_multiperiod(self):
        """MultiPeriodDiD with multi-absorb + survey weights raises ValueError."""
        np.random.seed(42)
        n = 60
        df = pd.DataFrame(
            {
                "outcome": np.random.randn(n),
                "treated": np.array([1] * 30 + [0] * 30),
                "time": np.tile([0, 1, 2], 20),
                "w": np.ones(n),
                "a": np.repeat(np.arange(6), 10),
                "b": np.tile(np.arange(5), 12),
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight")
        mpd = MultiPeriodDiD()
        with pytest.raises(ValueError, match="Multiple absorbed fixed effects"):
            mpd.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="time",
                post_periods=[2],
                absorb=["a", "b"],
                survey_design=sd,
            )

    def test_single_absorb_survey_allowed(self):
        """Single-absorb with survey weights should still work (regression guard)."""
        np.random.seed(42)
        n = 40
        df = pd.DataFrame(
            {
                "outcome": np.random.randn(n),
                "treated": np.array([1] * 20 + [0] * 20),
                "post": np.tile([0, 1], 20),
                "w": np.ones(n),
                "region": np.repeat(np.arange(4), 10),
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight")
        did = DifferenceInDifferences()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = did.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="post",
                absorb=["region"],
                survey_design=sd,
            )
        # Should succeed without error
        assert np.isfinite(result.att)

    def test_multiperiod_nonpositive_df_fallback(self):
        """MultiPeriodDiD with df=0 falls back to normal distribution."""
        np.random.seed(42)
        n = 40
        # 4 strata, 1 PSU per stratum → n_PSU=4, n_strata=4, df_survey=0
        strata = np.repeat([0, 1, 2, 3], 10)
        psu = strata.copy()  # 1 PSU per stratum → singleton strata
        df = pd.DataFrame(
            {
                "outcome": np.random.randn(n),
                "treated": np.array([1] * 20 + [0] * 20),
                "time": np.tile([0, 1, 2, 3], 10),
                "w": np.ones(n),
                "strat": strata,
                "cluster": psu,
            }
        )
        sd = SurveyDesign(
            weights="w",
            weight_type="pweight",
            strata="strat",
            psu="cluster",
            lonely_psu="adjust",
        )
        mpd = MultiPeriodDiD()
        with pytest.warns(UserWarning, match="non-positive"):
            result = mpd.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="time",
                post_periods=[2, 3],
                survey_design=sd,
            )
        # Period effects with finite SE > 0 should have finite p-values
        # (normal distribution fallback, not NaN from t(df=0))
        for period, pe in result.period_effects.items():
            if np.isfinite(pe.se) and pe.se > 0:
                assert np.isfinite(
                    pe.p_value
                ), f"Period {period}: finite SE={pe.se} but p_value={pe.p_value}"


class TestRound13Fixes:
    """Tests for PR #218 review round 13: zero-score-dispersion vcov."""

    def test_zero_score_dispersion_weights_only(self):
        """Weights-only design with identical scores returns zero vcov, not NaN."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        # All residuals zero → all scores zero → meat is zero from computation
        residuals = np.zeros(n)
        weights = np.ones(n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        vcov = compute_survey_vcov(X, residuals, resolved=resolved)
        # Zero residuals → zero scores → zero meat → zero vcov (not NaN)
        np.testing.assert_array_equal(vcov, np.zeros((2, 2)))

    def test_zero_score_dispersion_stratified_psu(self):
        """Stratified PSU design with identical PSU scores returns zero vcov."""
        np.random.seed(42)
        n = 30
        strata = np.repeat([0, 1, 2], 10)
        psu = np.tile(np.arange(5), 6)  # 5 PSUs per stratum

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        # All residuals zero → zero dispersion within each stratum
        residuals = np.zeros(n)
        weights = np.ones(n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=3,
            n_psu=15,
            lonely_psu="remove",
        )
        vcov = compute_survey_vcov(X, residuals, resolved=resolved)
        # Zero residuals → zero scores → zero V_h per stratum → zero vcov
        np.testing.assert_array_equal(vcov, np.zeros((2, 2)))


class TestRound14Fixes:
    """Tests for PR #218 review round 14 fixes."""

    def test_multiperiod_bootstrap_survey_fallback(self):
        """MultiPeriodDiD with wild_bootstrap + survey_design falls back gracefully."""
        np.random.seed(42)
        n = 40
        df = pd.DataFrame(
            {
                "outcome": np.random.randn(n),
                "treated": np.array([1] * 20 + [0] * 20),
                "time": np.tile([0, 1, 2, 3], 10),
                "w": np.ones(n),
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight")
        mpd = MultiPeriodDiD(inference="wild_bootstrap")
        # Should warn about fallback and produce valid analytical results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mpd.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="time",
                post_periods=[2, 3],
                survey_design=sd,
            )
        assert np.isfinite(result.avg_att)


class TestRound15Fixes:
    """Tests for PR #218 review round 15: NA validation for survey identifiers."""

    def test_strata_with_na_rejected(self):
        """SurveyDesign.resolve() rejects NA values in strata column."""
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "w": [1.0, 1.0, 1.0, 1.0],
                "strat": [0, 1, None, 0],  # NA in strata
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight", strata="strat")
        with pytest.raises(ValueError, match="Strata column.*missing values"):
            sd.resolve(df)

    def test_psu_with_na_rejected(self):
        """SurveyDesign.resolve() rejects NA values in PSU column."""
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "w": [1.0, 1.0, 1.0, 1.0],
                "cluster": [0, 1, np.nan, 0],  # NA in PSU
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight", psu="cluster")
        with pytest.raises(ValueError, match="PSU column.*missing values"):
            sd.resolve(df)

    def test_cluster_as_psu_with_na_rejected(self):
        """_inject_cluster_as_psu rejects NA values in cluster IDs."""
        from diff_diff.survey import _inject_cluster_as_psu

        resolved = ResolvedSurveyDesign(
            weights=np.ones(4),
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        cluster_ids = np.array([0, 1, np.nan, 0])
        with pytest.raises(ValueError, match="Cluster IDs contain missing"):
            _inject_cluster_as_psu(resolved, cluster_ids)


class TestRound16Fixes:
    """Tests for PR #218 review round 16: cluster-as-PSU nesting and FPC."""

    def test_injected_cluster_nested_in_strata(self):
        """Injected cluster IDs with repeated labels across strata get unique codes."""
        from diff_diff.survey import _inject_cluster_as_psu

        # 2 strata, cluster "1" appears in both → should produce 4 unique PSUs
        strata = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        resolved = ResolvedSurveyDesign(
            weights=np.ones(8),
            weight_type="pweight",
            strata=strata,
            psu=None,
            fpc=None,
            n_strata=2,
            n_psu=0,
            lonely_psu="remove",
        )
        cluster_ids = np.array([1, 1, 2, 2, 1, 1, 2, 2])  # labels repeat across strata
        result = _inject_cluster_as_psu(resolved, cluster_ids)
        # Should produce 4 unique PSUs (2 per stratum), not 2
        assert result.n_psu == 4
        # df_survey = n_psu - n_strata = 4 - 2 = 2
        assert result.df_survey == 2

    def test_fpc_with_strata_no_psu_accepted(self):
        """FPC + strata (no PSU) resolves — FPC validated later against effective PSUs."""
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "w": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "strat": [0, 0, 0, 1, 1, 1],
                "pop": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight", strata="strat", fpc="pop")
        # Should not raise at resolve time — FPC >= n_PSU validated at vcov time
        resolved = sd.resolve(df)
        assert resolved.fpc is not None

    def test_fpc_alone_no_strata_no_psu_accepted(self):
        """FPC alone (no PSU/strata) resolves — clusters may be injected later."""
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "w": [1.0, 1.0, 1.0, 1.0],
                "pop": [100.0, 100.0, 100.0, 100.0],
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight", fpc="pop")
        resolved = sd.resolve(df)
        assert resolved.fpc is not None

    def test_fpc_lt_effective_npsu_rejected_at_vcov(self):
        """FPC < effective n_PSU is rejected at compute_survey_vcov time."""
        np.random.seed(42)
        n = 12
        strata = np.repeat([0, 1], 6)
        psu = np.tile(np.arange(3), 4)  # 3 PSUs per stratum

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        residuals = np.random.randn(n)
        weights = np.ones(n)

        # FPC = 2 per stratum, but we have 3 PSUs → invalid
        fpc = np.array([2.0] * n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=fpc,
            n_strata=2,
            n_psu=6,
            lonely_psu="remove",
        )
        with pytest.raises(ValueError, match="FPC.*less than.*effective PSUs"):
            compute_survey_vcov(X, residuals, resolved=resolved)


class TestRound18Fixes:
    """Tests for PR #218 review round 18: implicit-PSU FPC handling."""

    def test_weights_only_fpc_reduces_variance(self):
        """Weights-only design with FPC > n_obs produces smaller vcov than without FPC."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        residuals = np.random.randn(n)
        weights = np.ones(n)

        # Without FPC
        resolved_no_fpc = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        vcov_no_fpc = compute_survey_vcov(X, residuals, resolved=resolved_no_fpc)

        # With FPC = 100 (sampling 20 from 100)
        fpc = np.full(n, 100.0)
        resolved_fpc = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=fpc,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        vcov_fpc = compute_survey_vcov(X, residuals, resolved=resolved_fpc)

        # FPC should reduce variance: (1 - 20/100) = 0.8 multiplier
        assert np.all(np.diag(vcov_fpc) < np.diag(vcov_no_fpc))
        np.testing.assert_allclose(np.diag(vcov_fpc), np.diag(vcov_no_fpc) * 0.8, rtol=1e-10)

    def test_weights_only_fpc_full_census_zero_vcov(self):
        """Weights-only FPC == n_obs (full census) produces zero vcov."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        residuals = np.random.randn(n)
        weights = np.ones(n)
        fpc = np.full(n, float(n))  # Full census: FPC == n_obs

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=fpc,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        vcov = compute_survey_vcov(X, residuals, resolved=resolved)
        np.testing.assert_array_equal(vcov, np.zeros((2, 2)))

    def test_weights_only_fpc_lt_nobs_rejected(self):
        """Weights-only FPC < n_obs is rejected at vcov time."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        residuals = np.random.randn(n)
        weights = np.ones(n)
        fpc = np.full(n, 10.0)  # FPC < n_obs → invalid

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=fpc,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        with pytest.raises(ValueError, match="FPC.*less than.*observations"):
            compute_survey_vcov(X, residuals, resolved=resolved)

    def test_did_with_fpc_only_survey(self):
        """DiD with weights-only FPC design produces finite results (estimator path)."""
        np.random.seed(42)
        n = 40
        df = pd.DataFrame(
            {
                "outcome": np.random.randn(n),
                "treated": np.array([1] * 20 + [0] * 20),
                "post": np.tile([0, 1], 20),
                "w": np.ones(n),
                "pop": np.full(n, 200.0),  # Sampling 40 from 200
            }
        )
        sd = SurveyDesign(weights="w", weight_type="pweight", fpc="pop")
        did = DifferenceInDifferences()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = did.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="post",
                survey_design=sd,
            )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0


class TestRound19Fixes:
    """Tests for PR #218 review round 19: PSU nesting validation."""

    def test_repeated_psu_labels_nest_false_rejected(self):
        """Repeated PSU labels across strata with nest=False are rejected."""
        n = 40
        strata = np.repeat([0, 1], 20)
        psu_raw = np.tile(np.arange(10), 4)[:n]  # labels repeat

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu_raw,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", nest=False)
        with pytest.raises(ValueError, match="PSU labels.*multiple strata"):
            sd.resolve(df)

    def test_repeated_psu_labels_nest_true_accepted(self):
        """Repeated PSU labels with nest=True produce correct n_psu."""
        n = 40
        strata = np.repeat([0, 1], 20)
        psu_raw = np.tile(np.arange(10), 4)[:n]

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu_raw,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", nest=True)
        resolved = sd.resolve(df)
        assert resolved.n_psu == 20  # 10 per stratum × 2
        assert resolved.df_survey == 18  # 20 - 2

    def test_unique_psu_labels_nest_false_accepted(self):
        """Globally unique PSU labels with nest=False work correctly."""
        n = 40
        strata = np.repeat([0, 1], 20)
        psu_raw = np.arange(n) // 2  # 20 unique PSUs, no overlap

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu_raw,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", nest=False)
        resolved = sd.resolve(df)
        assert resolved.n_psu == 20


class TestRound21Fixes:
    """Reject absorb + fixed_effects (FWL violation)."""

    def _make_panel(self):
        """Create a simple panel dataset for testing."""
        np.random.seed(42)
        n_units, n_periods = 20, 4
        df = pd.DataFrame(
            {
                "unit": np.repeat(range(n_units), n_periods),
                "time": np.tile(range(n_periods), n_units),
                "treated": np.repeat([1] * (n_units // 2) + [0] * (n_units // 2), n_periods),
                "post": np.tile([0, 0, 1, 1], n_units),
                "outcome": np.random.randn(n_units * n_periods),
                "region": np.repeat(["A", "B"] * (n_units // 2), n_periods),
                "sw": np.random.uniform(0.5, 2.0, n_units * n_periods),
            }
        )
        return df

    def test_absorb_fe_survey_rejected_did(self):
        """DiD rejects absorb + fixed_effects + survey_design."""
        df = self._make_panel()
        model = DifferenceInDifferences()
        with pytest.raises(ValueError, match="Cannot use both absorb and fixed_effects"):
            model.fit(
                df,
                formula="outcome ~ treated * post",
                absorb=["unit"],
                fixed_effects=["region"],
                survey_design=SurveyDesign(weights="sw"),
            )

    def test_absorb_fe_survey_rejected_multi_period(self):
        """MultiPeriodDiD rejects absorb + fixed_effects + survey_design."""
        df = self._make_panel()
        model = MultiPeriodDiD()
        with pytest.raises(ValueError, match="Cannot use both absorb and fixed_effects"):
            model.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="time",
                absorb=["unit"],
                fixed_effects=["region"],
                survey_design=SurveyDesign(weights="sw"),
            )

    def test_absorb_fe_rejected_without_survey_did(self):
        """DiD rejects absorb + fixed_effects even without survey weights."""
        df = self._make_panel()
        model = DifferenceInDifferences()
        with pytest.raises(ValueError, match="Cannot use both absorb and fixed_effects"):
            model.fit(
                df,
                formula="outcome ~ treated * post",
                absorb=["unit"],
                fixed_effects=["region"],
            )

    def test_absorb_fe_rejected_without_survey_multi_period(self):
        """MultiPeriodDiD rejects absorb + fixed_effects even without survey weights."""
        df = self._make_panel()
        model = MultiPeriodDiD()
        with pytest.raises(ValueError, match="Cannot use both absorb and fixed_effects"):
            model.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="time",
                absorb=["unit"],
                fixed_effects=["region"],
            )


class TestRound23Fixes:
    """TWFE: no-PSU survey path when cluster=None."""

    def test_twfe_weights_only_no_cluster_uses_no_psu_path(self, twfe_panel_data):
        """Weights-only survey with cluster=None uses implicit per-obs PSUs."""
        df = twfe_panel_data
        n_obs = len(df)
        twfe = TwoWayFixedEffects()
        result = twfe.fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            survey_design=SurveyDesign(weights="weight"),
        )
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_psu == n_obs
        assert result.survey_metadata.df_survey == n_obs - 1

    def test_twfe_stratified_no_psu_no_cluster(self, twfe_panel_data):
        """Stratified survey with no PSU and cluster=None uses n_obs - n_strata df."""
        df = twfe_panel_data
        n_obs = len(df)
        n_strata = df["stratum"].nunique()
        twfe = TwoWayFixedEffects()
        result = twfe.fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            survey_design=SurveyDesign(weights="weight", strata="stratum"),
        )
        assert result.survey_metadata is not None
        assert result.survey_metadata.df_survey == n_obs - n_strata

    def test_twfe_explicit_cluster_still_injects_psu(self, twfe_panel_data):
        """Explicit cluster= still injects cluster IDs as PSUs."""
        df = twfe_panel_data
        n_units = df["unit"].nunique()
        twfe = TwoWayFixedEffects(cluster="unit")
        result = twfe.fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            survey_design=SurveyDesign(weights="weight"),
        )
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_psu == n_units

    def test_twfe_non_survey_default_clustering_unaffected(self, twfe_panel_data):
        """Non-survey TWFE still uses unit-level clustering by default."""
        df = twfe_panel_data
        twfe = TwoWayFixedEffects()
        result = twfe.fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
        )
        assert result is not None
        assert np.isfinite(result.se)
        assert result.se > 0
