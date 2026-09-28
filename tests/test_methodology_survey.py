"""Survey Data Support methodology verification tests.

Targets Binder, D. A. (1983), *On the Variances of Asymptotically Normal
Estimators from Complex Surveys*, International Statistical Review 51(3),
279-292. DOI: 10.2307/1402588.

Secondary sources:
- Lumley, T. (2004). Analysis of Complex Survey Samples. *Journal of
  Statistical Software* 9(8), 1-19. (R ``survey`` package; ``svyglm``/``svydesign``.)
- Korn, E. L. & Graubard, B. I. (1990). Simultaneous Testing of Regression
  Coefficients with Complex Survey Data: Use of Bonferroni t Statistics.
  *The American Statistician* 44(4), 270-276. (Survey df = n_PSU - n_strata.)

Paper review on file: ``docs/methodology/papers/binder-1983-review.md``.
Theory note: ``docs/methodology/survey-theory.md`` (§4.4 Binder's result, §5 TSL
sandwich + df + singleton handling, §6 replicate variance).

Equation / section walk-through (HYBRID scope — these tests are the canonical
Binder-equation-numbered map; where an identity already has a tight oracle in the
broad survey suite, the bullet REFERENCES it instead of re-implementing it, and the
new first-class assertions concentrate on the genuinely-untested structures):

- **Binder Eq. 4.7 TSL sandwich** ``V = (X'WX)^-1 [sum_h V_h] (X'WX)^-1`` — the
  full sandwich *structure* is already exact-tested by
  ``test_survey.py::test_weighted_hc1_vcov_exact_oracle`` (a ``solve_ols`` HC1
  oracle, atol=1e-10) and ``::test_weights_only_oracle`` (atol=1e-12). Here
  ``TestTSLSandwich`` adds the genuinely-untested **residual-scale == score-scale**
  cross-function identity (survey-theory §5) and the PSU-only clustered-meat form.
- **Binder §4.4 IF variance** ``V = sum_h (1-f_h)(n_h/(n_h-1)) sum_j (psi_hj - psi_h_bar)^2``
  for *arbitrary* psi (not only fitted residuals) — ``TestBinder1983Variance``.
- **Stratum meat + FPC** ``V_h = (1-f_h)(n_h/(n_h-1)) sum_j (T_hj - T_h_bar)(...)'``
  — the single-group Bessel factor is already exact in
  ``test_survey.py::test_no_strata_degeneracy``; ``TestStratumMeatAndFPC`` adds the
  **multi-stratum decomposition** (>=2 strata, heterogeneous n_h, distinct Bessel
  factors summed) plus the FPC scaling / full-census-zero / ``N_h < n_psu`` raise.
- **Singleton handling + zero-vs-NaN identification** — ``TestSingletonStratum``.
- **Survey df = n_PSU - n_strata** (4-way branch + replicate QR-rank-1; Korn-Graubard
  1990) — ``TestSurveyDegreesOfFreedom``.
- **Weight-type meat** pweight ``sum w^2 x x' u^2`` / fweight ``X'diag(w u^2)X`` +
  ``df=sum(w)-k`` / aweight unweighted — pweight is already exact in
  ``test_weighted_hc1_vcov_exact_oracle``; ``TestWeightTypeMeat`` concentrates on the
  untested **fweight** and **aweight** structures.
- **Replicate variance** ``V = c * sum_r s_r (theta_r - theta_center)^2`` with BRR/Fay/
  JK1/JKn/SDR factors — ``TestReplicateVariance``.
- **DEFF = design_var / srs_var** (exact ratio identity) — ``TestDEFF``.
- **WLS estimating equations** ``X'W(y - X beta) = 0`` (Binder Eq. 2.1/2.3) —
  ``TestSurveyWLSEstimation``.
- **R parity** (machine-precision goldens vs R ``survey``) — pointer in
  ``TestSurveyParityR``; the full grids live in ``test_survey_r_crossvalidation.py``,
  ``test_survey_estimator_validation.py``, ``test_survey_real_data.py``.

Warning-firing coverage (lonely-PSU removal, ill-conditioned ``X'WX``) lives in the
broad ``tests/test_survey*.py`` suites; this methodology file asserts the variance
*identities* and defers the defensive surface, mirroring how
``tests/test_methodology_conley.py`` defers to ``tests/test_conley_vcov.py``.
"""

import json
import os

import numpy as np
import pytest

from diff_diff.linalg import compute_robust_vcov, solve_ols
from diff_diff.survey import (
    ResolvedSurveyDesign,
    _compute_stratified_psu_meat,
    _replicate_variance_factor,
    compute_deff_diagnostics,
    compute_replicate_if_variance,
    compute_survey_if_variance,
    compute_survey_vcov,
)

ATOL = 1e-12


def _resolved(weights, weight_type="pweight", strata=None, psu=None, fpc=None, lonely_psu="remove"):
    """Build a TSL ResolvedSurveyDesign directly from arrays (test helper)."""
    n_strata = int(len(np.unique(strata))) if strata is not None else 0
    if psu is not None:
        n_psu = int(len(np.unique(psu)))
    elif strata is not None:
        n_psu = 0
    else:
        n_psu = 0
    return ResolvedSurveyDesign(
        weights=np.asarray(weights, dtype=float),
        weight_type=weight_type,
        strata=None if strata is None else np.asarray(strata),
        psu=None if psu is None else np.asarray(psu),
        fpc=None if fpc is None else np.asarray(fpc, dtype=float),
        n_strata=n_strata,
        n_psu=n_psu,
        lonely_psu=lonely_psu,
    )


# =============================================================================
# Binder Eq. 2.1 / 2.3 — survey-weighted estimating equations
# =============================================================================
class TestSurveyWLSEstimation:
    """``B`` solves ``X'W(y - X B) = 0`` (Binder Eq. 2.1/2.3); WLS bread ``X'WX``."""

    def test_wls_solves_weighted_normal_equations(self):
        """At the WLS solution the weighted scores sum to zero (Binder FOC)."""
        rng = np.random.default_rng(7)
        n = 40
        X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
        w = rng.uniform(0.5, 3.0, size=n)
        y = X @ np.array([1.0, 2.0, -0.5]) + rng.normal(size=n) * 0.4

        _, resid, _ = solve_ols(X, y, weights=w, weight_type="pweight")
        # Sum_i w_i x_i u_i = 0 — the estimating equation the TSL meat is built on.
        score_total = X.T @ (w * resid)
        np.testing.assert_allclose(score_total, np.zeros(X.shape[1]), atol=1e-8)

    def test_pweight_scale_invariance(self):
        """beta is invariant to scaling all weights by a constant (sum(w)=n convention)."""
        rng = np.random.default_rng(11)
        n = 30
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        w = rng.uniform(0.5, 2.0, size=n)
        y = X @ np.array([0.5, 1.5]) + rng.normal(size=n) * 0.3

        coef_a, _, _ = solve_ols(X, y, weights=w, weight_type="pweight")
        coef_b, _, _ = solve_ols(X, y, weights=3.0 * w, weight_type="pweight")
        np.testing.assert_allclose(coef_a, coef_b, atol=ATOL)


# =============================================================================
# Binder §4.4 — IF-based design variance for arbitrary psi
# =============================================================================
class TestBinder1983Variance:
    """``V = sum_h (1-f_h)(n_h/(n_h-1)) sum_j (psi_hj - psi_h_bar)^2`` (Binder §4.4)."""

    def test_if_variance_matches_binder_formula(self):
        """Stratified PSU-total IF variance equals the hand-computed Binder sum."""
        # 2 strata; stratum 0 has 3 PSUs (2 obs each), stratum 1 has 2 PSUs (3 obs each).
        strata = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        psu = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
        N_h = {0: 50.0, 1: 40.0}
        fpc = np.array([N_h[h] for h in strata])
        rng = np.random.default_rng(3)
        psi = rng.normal(size=12)

        resolved = _resolved(np.ones(12), strata=strata, psu=psu, fpc=fpc)
        got = compute_survey_if_variance(psi, resolved)

        # Hand reference: within-stratum PSU totals, centered, Bessel + FPC.
        expected = 0.0
        for h in (0, 1):
            mh = strata == h
            df = _psu_totals(psi[mh], psu[mh])
            n_h = df.shape[0]
            f_h = n_h / N_h[h]
            centered = df - df.mean()
            expected += (1.0 - f_h) * (n_h / (n_h - 1)) * float(np.sum(centered**2))
        assert np.isclose(got, expected, atol=ATOL)

    def test_within_stratum_centering_for_arbitrary_psi(self):
        """Centering is WITHIN stratum (global mean != within-stratum means)."""
        # Stratum means deliberately far apart so global centering would differ.
        strata = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        psu = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        psi = np.array([1.0, 1.2, 0.8, 1.1, 9.0, 9.4, 8.6, 9.2])
        resolved = _resolved(np.ones(8), strata=strata, psu=psu)
        got = compute_survey_if_variance(psi, resolved)

        expected = 0.0
        for h in (0, 1):
            mh = strata == h
            tot = _psu_totals(psi[mh], psu[mh])
            n_h = tot.shape[0]
            expected += (n_h / (n_h - 1)) * float(np.sum((tot - tot.mean()) ** 2))
        assert np.isclose(got, expected, atol=ATOL)
        # Global-centering reference would be materially different.
        all_tot = _psu_totals(psi, psu)
        global_ref = (4 / 3) * float(np.sum((all_tot - all_tot.mean()) ** 2))
        assert not np.isclose(got, global_ref, atol=1e-6)

    def test_no_design_reduces_to_centered_sum_of_squares(self):
        """Weights-only (implicit per-obs PSU) → V = (n/(n-1)) sum (psi - psi_bar)^2."""
        psi = np.array([0.4, -1.1, 2.0, 0.3, -0.7, 1.2])
        resolved = _resolved(np.ones(6))
        got = compute_survey_if_variance(psi, resolved)
        n = 6
        expected = (n / (n - 1)) * float(np.sum((psi - psi.mean()) ** 2))
        assert np.isclose(got, expected, atol=ATOL)


# =============================================================================
# Binder Eq. 4.7 — TSL sandwich (residual-scale == score-scale; PSU-clustered meat)
# =============================================================================
class TestTSLSandwich:
    """``V_TSL = (X'WX)^-1 [sum_h V_h] (X'WX)^-1`` (Binder Eq. 4.7; survey-theory §5).

    The full sandwich structure is already exact-tested by
    ``test_survey.py::test_weighted_hc1_vcov_exact_oracle`` / ``::test_weights_only_oracle``.
    These assert the untested cross-function and PSU-only forms.
    """

    def test_residual_scale_equals_score_scale(self):
        """survey-theory §5: compute_survey_vcov(X=ones, eif) == compute_survey_if_variance(w*eif/sum w)."""
        strata = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        psu = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        rng = np.random.default_rng(5)
        eif = rng.normal(size=8)
        w = rng.uniform(0.5, 2.0, size=8)
        resolved = _resolved(w, strata=strata, psu=psu)

        ones = np.ones((8, 1))
        vcov = compute_survey_vcov(ones, eif, resolved)  # internally scores = w*eif
        if_var = compute_survey_if_variance(w * eif / np.sum(w), resolved)
        assert np.isclose(float(vcov[0, 0]), if_var, atol=ATOL)

    def test_psu_no_strata_reduces_to_psu_clustered_meat(self):
        """No strata, explicit PSU → sandwich with PSU-total clustered meat (G/(G-1))."""
        psu = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
        n = 10
        rng = np.random.default_rng(9)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        w = rng.uniform(0.5, 2.0, size=n)
        y = X @ np.array([1.0, 0.7]) + rng.normal(size=n) * 0.3
        _, resid, _ = solve_ols(X, y, weights=w, weight_type="pweight")
        resolved = _resolved(w, psu=psu)

        got = compute_survey_vcov(X, resid, resolved)

        XtWX = X.T @ (X * w[:, None])
        scores = X * (w * resid)[:, None]
        psu_tot = _psu_totals(scores, psu)
        G = psu_tot.shape[0]
        meat = (G / (G - 1)) * (psu_tot - psu_tot.mean(axis=0)).T @ (psu_tot - psu_tot.mean(axis=0))
        bread_inv = np.linalg.inv(XtWX)
        expected = bread_inv @ meat @ bread_inv
        np.testing.assert_allclose(got, expected, atol=ATOL)

    def test_vcov_symmetric_and_shape(self):
        """Sandwich is (k, k) and symmetric."""
        strata = np.repeat([0, 1], 6)
        psu = np.repeat(np.arange(4), 3)
        n = 12
        rng = np.random.default_rng(13)
        X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
        w = rng.uniform(0.5, 2.0, size=n)
        y = X @ np.array([1.0, 0.5, -0.3]) + rng.normal(size=n) * 0.3
        _, resid, _ = solve_ols(X, y, weights=w, weight_type="pweight")
        resolved = _resolved(w, strata=strata, psu=psu)
        vcov = compute_survey_vcov(X, resid, resolved)
        assert vcov.shape == (3, 3)
        np.testing.assert_allclose(vcov, vcov.T, atol=ATOL)


# =============================================================================
# Stratum meat + FPC — multi-stratum Bessel decomposition (genuine gap)
# =============================================================================
class TestStratumMeatAndFPC:
    """``V_h = (1-f_h)(n_h/(n_h-1)) sum_j (T_hj - T_h_bar)(...)'`` (survey-theory §5).

    Single-group Bessel is already exact in ``test_survey.py::test_no_strata_degeneracy``;
    this asserts the MULTI-STRATUM sum of distinct ``n_h/(n_h-1)`` factors.
    """

    def test_multi_stratum_bessel_decomposition(self):
        """>=2 strata with heterogeneous n_h: distinct Bessel factors summed (no FPC)."""
        # Stratum 0: 3 PSUs (factor 3/2). Stratum 1: 5 PSUs (factor 5/4).
        strata = np.array([0] * 6 + [1] * 10)
        psu = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        rng = np.random.default_rng(17)
        scores = rng.normal(size=(16, 2))  # (n, k=2) score vectors
        resolved = _resolved(np.ones(16), strata=strata, psu=psu)  # no FPC

        meat, computed, _ = _compute_stratified_psu_meat(scores, resolved)
        assert computed

        expected = np.zeros((2, 2))
        for h, n_h_expect in ((0, 3), (1, 5)):
            mh = strata == h
            tot = _psu_totals(scores[mh], psu[mh])
            n_h = tot.shape[0]
            assert n_h == n_h_expect  # heterogeneous PSU counts
            c = tot - tot.mean(axis=0)
            expected += (n_h / (n_h - 1)) * (c.T @ c)
        np.testing.assert_allclose(meat, expected, atol=ATOL)
        # Sanity: the two strata genuinely use different Bessel factors.
        assert abs((3 / 2) - (5 / 4)) > 0.1

    def test_fpc_scales_meat_by_one_minus_f(self):
        """FPC multiplies the stratum meat by (1 - f_h) exactly."""
        strata = np.zeros(8, dtype=int)
        psu = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        rng = np.random.default_rng(19)
        scores = rng.normal(size=(8, 1))
        N_h = 20.0
        f_h = 4 / N_h

        meat_no_fpc, _, _ = _compute_stratified_psu_meat(
            scores, _resolved(np.ones(8), strata=strata, psu=psu)
        )
        meat_fpc, _, _ = _compute_stratified_psu_meat(
            scores, _resolved(np.ones(8), strata=strata, psu=psu, fpc=np.full(8, N_h))
        )
        np.testing.assert_allclose(meat_fpc, (1.0 - f_h) * meat_no_fpc, atol=ATOL)

    def test_full_census_contributes_zero_finite(self):
        """Full census (N_h = n_psu, f_h = 1) → exactly zero variance, finite (not NaN)."""
        strata = np.zeros(6, dtype=int)
        psu = np.array([0, 0, 1, 1, 2, 2])
        psi = np.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0])
        resolved = _resolved(np.ones(6), strata=strata, psu=psu, fpc=np.full(6, 3.0))
        got = compute_survey_if_variance(psi, resolved)
        assert got == 0.0  # legitimate zero, finite
        assert not np.isnan(got)

    def test_fpc_below_npsu_raises(self):
        """N_h < n_psu is a fail-closed ValueError."""
        strata = np.zeros(6, dtype=int)
        psu = np.array([0, 0, 1, 1, 2, 2])
        resolved = _resolved(np.ones(6), strata=strata, psu=psu, fpc=np.full(6, 2.0))
        with pytest.raises(ValueError, match="FPC"):
            compute_survey_if_variance(np.arange(6.0), resolved)


# =============================================================================
# Singleton / lonely-PSU handling + zero-vs-NaN identification contract
# =============================================================================
class TestSingletonStratum:
    """``lonely_psu`` remove / certainty / adjust + the zero-vs-NaN contract (§5)."""

    def test_remove_skips_singleton_stratum(self):
        """remove: total meat equals the non-singleton stratum's meat alone."""
        # Stratum 0: 2 PSUs; stratum 1: 1 PSU (singleton).
        strata = np.array([0, 0, 0, 0, 1, 1])
        psu = np.array([0, 0, 1, 1, 2, 2])
        rng = np.random.default_rng(23)
        scores = rng.normal(size=(6, 1))
        full, _, _ = _compute_stratified_psu_meat(
            scores, _resolved(np.ones(6), strata=strata, psu=psu, lonely_psu="remove")
        )
        # Stratum 0 alone.
        m0 = strata == 0
        s0, _, _ = _compute_stratified_psu_meat(
            scores[m0], _resolved(np.ones(4), strata=strata[m0], psu=psu[m0])
        )
        np.testing.assert_allclose(full, s0, atol=ATOL)

    def test_adjust_centers_singleton_at_grand_psu_mean(self):
        """adjust: singleton contributes (T_singleton - grand_PSU_mean)^2."""
        strata = np.array([0, 0, 0, 0, 1, 1])
        psu = np.array([0, 0, 1, 1, 2, 2])
        psi = np.array([1.0, 1.5, 0.5, 0.8, 4.0, 4.6])
        resolved = _resolved(np.ones(6), strata=strata, psu=psu, lonely_psu="adjust")
        got = compute_survey_if_variance(psi, resolved)

        # Grand PSU mean across all 3 PSU totals.
        all_tot = _psu_totals(psi, psu)
        grand = all_tot.mean()
        # Stratum 0 (2 PSUs, Bessel 2/1, centered within); stratum 1 singleton, adjust.
        s0 = _psu_totals(psi[strata == 0], psu[strata == 0])
        expected = (2 / 1) * float(np.sum((s0 - s0.mean()) ** 2))
        s1 = _psu_totals(psi[strata == 1], psu[strata == 1])  # one PSU total
        expected += float(np.sum((s1 - grand) ** 2))
        assert np.isclose(got, expected, atol=ATOL)

    def test_all_singleton_remove_is_nan(self):
        """All strata singleton + remove → unidentified variance → NaN."""
        strata = np.array([0, 1, 2])
        psu = np.array([0, 1, 2])
        resolved = _resolved(np.ones(3), strata=strata, psu=psu, lonely_psu="remove")
        got = compute_survey_if_variance(np.array([1.0, 2.0, 3.0]), resolved)
        assert np.isnan(got)

    def test_all_singleton_certainty_is_zero(self):
        """All strata singleton + certainty → legitimate zero (finite), NOT NaN."""
        strata = np.array([0, 1, 2])
        psu = np.array([0, 1, 2])
        resolved = _resolved(np.ones(3), strata=strata, psu=psu, lonely_psu="certainty")
        got = compute_survey_if_variance(np.array([1.0, 2.0, 3.0]), resolved)
        assert got == 0.0
        assert not np.isnan(got)


# =============================================================================
# Survey degrees of freedom — n_PSU - n_strata (Korn-Graubard 1990)
# =============================================================================
class TestSurveyDegreesOfFreedom:
    """``df`` 4-way branch + replicate QR-rank-1 (survey-theory §5; matches R ``degf()``).

    Functional coverage of each branch also lives in
    ``test_survey.py::test_survey_metadata_df_survey`` (+ siblings).
    """

    def test_df_psu_plus_strata(self):
        strata = np.repeat([0, 1, 2], 4)
        psu = np.repeat(np.arange(6), 2)
        assert _resolved(np.ones(12), strata=strata, psu=psu).df_survey == 6 - 3

    def test_df_psu_only(self):
        psu = np.repeat(np.arange(5), 2)
        assert _resolved(np.ones(10), psu=psu).df_survey == 5 - 1

    def test_df_strata_only(self):
        strata = np.repeat([0, 1, 2], 5)
        assert _resolved(np.ones(15), strata=strata).df_survey == 15 - 3

    def test_df_weights_only(self):
        assert _resolved(np.ones(20)).df_survey == 20 - 1

    def test_df_replicate_qr_rank_minus_one(self):
        """Replicate df = QR-rank(analysis weights) - 1 (R ``survey::degf``)."""
        rng = np.random.default_rng(29)
        n, R = 20, 8
        rep = rng.uniform(0.5, 1.5, size=(n, R))  # full column rank a.s.
        resolved = ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
            replicate_weights=rep,
            replicate_method="BRR",
            n_replicates=R,
            combined_weights=True,
        )
        assert resolved.df_survey == R - 1


# =============================================================================
# Weight-type meat — fweight (one power of w) + aweight (unweighted) [genuine gap]
# =============================================================================
class TestWeightTypeMeat:
    """HC1 meat by weight type (REGISTRY "Weight Type Effects on Inference").

    The pweight ``sum w^2 u^2 x x'`` meat is already exact-tested by
    ``test_survey.py::test_weighted_hc1_vcov_exact_oracle``; this concentrates on the
    untested **fweight** (``X'diag(w u^2)X`` + ``df=sum(w)-k``) and **aweight**
    (unweighted meat) structures, plus the survey-TSL aweight path.
    """

    def _fit(self, seed, w, weight_type="pweight"):
        rng = np.random.default_rng(seed)
        n = w.shape[0]
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = X @ np.array([1.0, 0.6]) + rng.normal(size=n) * 0.4
        # beta is the same across weight_type for given w; fit once.
        _, resid, _ = solve_ols(X, y, weights=w, weight_type=weight_type)
        return X, resid

    def test_fweight_meat_one_power_w(self):
        """fweight: V = (n_eff/(n_eff-k)) (X'WX)^-1 [X'diag(w u^2)X] (X'WX)^-1, n_eff=sum(w)."""
        w = np.array([1, 2, 3, 1, 2, 4, 1, 2], dtype=float)
        X, u = self._fit(31, w, weight_type="fweight")
        k = X.shape[1]
        n_eff = int(np.sum(w))
        XtWX = X.T @ (X * w[:, None])
        meat = X.T @ (X * (w * u**2)[:, None])  # ONE power of w
        bread_inv = np.linalg.inv(XtWX)
        expected = (n_eff / (n_eff - k)) * bread_inv @ meat @ bread_inv
        got = compute_robust_vcov(X, u, weights=w, weight_type="fweight")
        np.testing.assert_allclose(got, expected, atol=ATOL)

    def test_fweight_df_is_sum_w_minus_k(self):
        """fweight degrees of freedom = sum(w) - k (frequency expansion)."""
        w = np.array([1, 2, 3, 1, 2, 4, 1, 2], dtype=float)
        X, u = self._fit(31, w, weight_type="fweight")
        _, dof = compute_robust_vcov(X, u, weights=w, weight_type="fweight", return_dof=True)
        np.testing.assert_allclose(dof, np.full(X.shape[1], np.sum(w) - X.shape[1]))

    def test_aweight_meat_is_unweighted(self):
        """aweight: meat = X'diag(u^2)X (no w), bread still X'WX, adjustment n/(n-k)."""
        rng = np.random.default_rng(37)
        w = rng.uniform(0.5, 2.0, size=10)
        X, u = self._fit(37, w, weight_type="aweight")
        n, k = X.shape
        XtWX = X.T @ (X * w[:, None])
        meat = X.T @ (X * (u**2)[:, None])  # NO weight in the meat
        bread_inv = np.linalg.inv(XtWX)
        expected = (n / (n - k)) * bread_inv @ meat @ bread_inv
        got = compute_robust_vcov(X, u, weights=w, weight_type="aweight")
        np.testing.assert_allclose(got, expected, atol=ATOL)

    def test_pweight_and_aweight_meat_differ(self):
        """The weight type genuinely changes the meat (pweight w^2 vs aweight unweighted)."""
        rng = np.random.default_rng(41)
        w = rng.uniform(0.5, 2.5, size=12)
        X, u = self._fit(41, w)
        v_p = compute_robust_vcov(X, u, weights=w, weight_type="pweight")
        v_a = compute_robust_vcov(X, u, weights=w, weight_type="aweight")
        assert not np.allclose(v_p, v_a, atol=1e-8)

    def test_survey_tsl_aweight_drops_weight_from_scores(self):
        """compute_survey_vcov aweight forms unweighted scores (≠ pweight w-scaled)."""
        rng = np.random.default_rng(43)
        n = 12
        w = rng.uniform(0.5, 2.0, size=n)
        psu = np.repeat(np.arange(4), 3)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(size=n) * 0.3
        _, u, _ = solve_ols(X, y, weights=w, weight_type="aweight")

        got_a = compute_survey_vcov(X, u, _resolved(w, weight_type="aweight", psu=psu))
        # Hand: aweight scores have NO weight; bread is still X'WX.
        XtWX = X.T @ (X * w[:, None])
        scores = X * u[:, None]
        psu_tot = _psu_totals(scores, psu)
        G = psu_tot.shape[0]
        meat = (G / (G - 1)) * (psu_tot - psu_tot.mean(0)).T @ (psu_tot - psu_tot.mean(0))
        bread_inv = np.linalg.inv(XtWX)
        np.testing.assert_allclose(got_a, bread_inv @ meat @ bread_inv, atol=ATOL)
        # And it differs from the pweight (w-scaled scores) survey vcov.
        got_p = compute_survey_vcov(X, u, _resolved(w, weight_type="pweight", psu=psu))
        assert not np.allclose(got_a, got_p, atol=1e-8)


# =============================================================================
# Replicate variance — V = c * sum_r (theta_r - theta_center)^2 (survey-theory §6)
# =============================================================================
class TestReplicateVariance:
    """Per-method replicate factors + the IF-reweighting variance formula (§6)."""

    def test_method_factors(self):
        assert _replicate_variance_factor("BRR", 20, 0.0) == 1.0 / 20
        assert np.isclose(_replicate_variance_factor("Fay", 20, 0.3), 1.0 / (20 * 0.7**2))
        assert _replicate_variance_factor("JK1", 20, 0.0) == 19.0 / 20
        assert _replicate_variance_factor("SDR", 20, 0.0) == 4.0 / 20

    def _rep_resolved(self, rep, method, *, mse=True, fay_rho=0.0, rscales=None, rep_strata=None):
        n, R = rep.shape
        return ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
            replicate_weights=rep,
            replicate_method=method,
            fay_rho=fay_rho,
            n_replicates=R,
            replicate_strata=None if rep_strata is None else np.asarray(rep_strata),
            combined_weights=True,
            replicate_rscales=None if rscales is None else np.asarray(rscales, dtype=float),
            mse=mse,
        )

    def test_if_replicate_matches_direct_formula_brr(self):
        """BRR IF variance = (1/R) sum_r (theta_r - theta_full)^2, theta_r = sum(w_r psi)."""
        rng = np.random.default_rng(47)
        n, R = 10, 6
        psi = rng.normal(size=n)
        rep = rng.uniform(0.4, 1.6, size=(n, R))  # combined weights vs full=ones → ratio=rep
        resolved = self._rep_resolved(rep, "BRR", mse=True)
        got, n_valid = compute_replicate_if_variance(psi, resolved)
        assert n_valid == R

        theta_full = float(np.sum(psi))
        theta_reps = np.array([float(np.sum(rep[:, r] * psi)) for r in range(R)])
        expected = (1.0 / R) * float(np.sum((theta_reps - theta_full) ** 2))
        assert np.isclose(got, expected, atol=ATOL)

    def test_jkn_per_stratum_factor(self):
        """JKn: V = sum_h ((n_h-1)/n_h) sum_{r in h} (theta_r - theta_full)^2."""
        rng = np.random.default_rng(53)
        n = 10
        psi = rng.normal(size=n)
        rep_strata = np.array([0, 0, 0, 1, 1])  # stratum 0: 3 reps, stratum 1: 2 reps
        R = len(rep_strata)
        rep = rng.uniform(0.4, 1.6, size=(n, R))
        resolved = self._rep_resolved(rep, "JKn", mse=True, rep_strata=rep_strata)
        got, n_valid = compute_replicate_if_variance(psi, resolved)
        assert n_valid == R

        theta_full = float(np.sum(psi))
        theta_reps = np.array([float(np.sum(rep[:, r] * psi)) for r in range(R)])
        expected = 0.0
        for h, n_h in ((0, 3), (1, 2)):
            mask = rep_strata == h
            expected += ((n_h - 1) / n_h) * float(np.sum((theta_reps[mask] - theta_full) ** 2))
        assert np.isclose(got, expected, atol=ATOL)

    def test_n_valid_below_two_returns_nan(self):
        """Fewer than 2 valid replicates → NaN variance (not estimable)."""
        psi = np.array([1.0, 2.0, 3.0])
        # Only one replicate column has any positive weight.
        rep = np.zeros((3, 4))
        rep[:, 0] = np.array([1.0, 1.2, 0.8])
        resolved = self._rep_resolved(rep, "BRR", mse=True)
        got, n_valid = compute_replicate_if_variance(psi, resolved)
        assert n_valid < 2
        assert np.isnan(got)


# =============================================================================
# DEFF = design_var / srs_var (exact ratio identity) [genuine gap]
# =============================================================================
class TestDEFF:
    """``DEFF = diag(survey_vcov) / diag(srs_hc1_vcov)`` exactly (REGISTRY "DEFF")."""

    def test_deff_is_exact_ratio_of_design_to_srs_variance(self):
        """DEFF reconstructs as the design/SRS variance ratio, not Kish or effective-n."""
        rng = np.random.default_rng(59)
        n = 60
        strata = np.repeat([0, 1, 2], 20)
        psu = np.repeat(np.arange(12), 5)
        w = rng.uniform(0.5, 2.0, size=n)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(size=n) * 0.4
        _, resid, _ = solve_ols(X, y, weights=w, weight_type="pweight")
        resolved = _resolved(w, strata=strata, psu=psu, fpc=np.full(n, 500.0))

        survey_vcov = compute_survey_vcov(X, resid, resolved)
        deff_obj = compute_deff_diagnostics(X, resid, survey_vcov, w, weight_type="pweight")
        srs_vcov = compute_robust_vcov(X, resid, weights=w, weight_type="pweight")  # HC1 default

        expected = np.diag(survey_vcov) / np.diag(srs_vcov)
        np.testing.assert_allclose(deff_obj.deff, expected, atol=ATOL)

    def test_deff_above_one_under_positive_intra_psu_correlation(self):
        """Strong within-PSU correlation inflates design variance: DEFF > 1 (soft)."""
        rng = np.random.default_rng(61)
        n_psu, per = 30, 8
        n = n_psu * per
        psu = np.repeat(np.arange(n_psu), per)
        # Strong shared PSU effect → positive intra-PSU correlation.
        psu_effect = rng.normal(scale=4.0, size=n_psu)[psu]
        treat = (np.arange(n_psu) % 2)[psu].astype(float)  # treatment by PSU parity
        X = np.column_stack([np.ones(n), treat])
        y = 1.0 + 0.5 * treat + psu_effect + rng.normal(scale=0.5, size=n)
        w = np.ones(n)
        _, resid, _ = solve_ols(X, y, weights=w, weight_type="pweight")
        resolved = _resolved(w, psu=psu)
        survey_vcov = compute_survey_vcov(X, resid, resolved)
        deff_obj = compute_deff_diagnostics(X, resid, survey_vcov, w, weight_type="pweight")
        # Treatment coefficient DEFF should exceed 1 under clustering.
        assert deff_obj.deff[1] > 1.0


# =============================================================================
# R parity — pointer to the machine-precision goldens (no duplication)
# =============================================================================
class TestSurveyParityR:
    """Machine-precision R ``survey`` parity is asserted in dedicated suites.

    - ``tests/test_survey_r_crossvalidation.py`` — ``svyglm`` / ``svydesign`` /
      ``svrepdesign`` (DiD, CallawaySantAnna, BRR).
    - ``tests/test_survey_estimator_validation.py`` — S1-S4
      (ImputationDiD / StackedDiD / SunAbraham / TripleDifference).
    - ``tests/test_survey_real_data.py`` — API / NHANES / RECS at atol 1e-8.
    """

    def test_r_parity_goldens_present_and_referenced(self):
        golden = os.path.join(
            os.path.dirname(__file__),
            "..",
            "benchmarks",
            "data",
            "synthetic",
            "survey_crossvalidation_r_results.json",
        )
        if not os.path.exists(golden):
            pytest.skip("R survey cross-validation golden not present (isolated install).")
        with open(golden) as fh:
            data = json.load(fh)
        assert len(data) > 0  # the svyglm/svrepdesign reference scenarios exist


def _psu_totals(values, psu):
    """Sum ``values`` (1-D or 2-D) within each unique PSU id; return (n_psu[, k])."""
    values = np.asarray(values, dtype=float)
    order = np.unique(psu)
    if values.ndim == 1:
        return np.array([values[psu == g].sum() for g in order])
    return np.array([values[psu == g].sum(axis=0) for g in order])
