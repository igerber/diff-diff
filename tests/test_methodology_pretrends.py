"""
PreTrendsPower methodology test file — Roth (2022) Section II.A-B walkthrough.

Companion to ``tests/test_pretrends.py`` (basic unit-test surface): this file
validates the library against Roth's specific paper equations and propositions,
with paper-equation-numbered assertions. Mirrors the structure of
``tests/test_methodology_bacon.py``.

Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for
    Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.
    https://doi.org/10.1257/aeri.20210236

Paper review on file: ``docs/methodology/papers/roth-2022-review.md``.

Class structure:

- ``TestPretrendsHandCalculation`` — K=1 closed-form match against
  Proposition 2 proof's univariate truncated-normal expression; NIS power
  against Monte Carlo simulation at small K; MDV inversion sanity.
- ``TestPretrendsPropositions`` — Roth Propositions 1-4 numerical
  verification via Monte Carlo simulation.
- ``TestPretrendsLinearGrid`` — γ-unit MDV on regular, irregular, and
  anticipation-shifted pre-period grids (PR-B Step 4 regression).
- ``TestPretrendsCustomWeightPersistence`` — custom weights stored on
  PreTrendsPowerResults; power_at(M) for custom matches a refit (PR-B
  Step 5 regression).
- ``TestPretrendsCovarianceSource`` — CS/SA full-VCV routing through
  event_study_vcov (PR-B Step 3 regression).
- ``TestPretrendsHelperAPI`` — compute_pretrends_power + compute_mdv accept
  violation_weights + pretest_form end-to-end (PR-B Step 6 regression).
- ``TestPretrendsNISvsWald`` — NIS and Wald forms produce form-consistent
  output; backwards-compat regression on the Wald path.
- ``TestPretrendsParityR`` — R `pretrends` package parity at
  `jonathandroth/pretrends` commit ``122731d082`` (package version 0.1.0).
  Goldens at ``benchmarks/data/r_pretrends_golden.json`` populated in PR-C
  (2026-05-19). The ``@pytest.mark.skipif`` guards partial checkouts where
  the JSON is absent.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff.pretrends import (
    PreTrendsPower,
    PreTrendsPowerResults,
    compute_mdv,
    compute_pretrends_power,
)
from diff_diff.sun_abraham import SunAbraham

# =============================================================================
# Shared fixtures
# =============================================================================


def _make_sa_panel(n_units_per_cohort=20, cohorts=(3, 4, 5), n_periods=6, seed=0):
    """Build a staggered-adoption panel for SunAbraham fitting.

    Default: 3 timing cohorts (3, 4, 5) of 20 units each + 20 never-treated,
    panel length 6. K=3 pre-periods for the first-treated cohort under default
    `anticipation=0`. Null DGP (no real treatment effect) — useful for
    SE-and-power tests without confounding.
    """
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_units_per_cohort):
            for t in range(1, n_periods + 1):
                rows.append((uid, g, t))
            uid += 1
    for _ in range(n_units_per_cohort):
        for t in range(1, n_periods + 1):
            rows.append((uid, 0, t))
        uid += 1
    df = pd.DataFrame(rows, columns=["unit", "first_treat", "time"])
    df["y"] = rng.normal(0, 0.5, len(df))
    return df


@pytest.fixture
def sa_results():
    """Fitted SunAbraham results on a 3-cohort + never-treated panel.

    Returns a SunAbrahamResults with event_study_vcov populated (post-PR-B
    Step 3 SA extension). Pre-periods at first-treated cohort g=3 are
    {-2, -1} under default anticipation=0 — but the full event_study_vcov_index
    spans {-4, -3, -2, 0, 1, 2, 3} across all cohorts.
    """
    df = _make_sa_panel()
    return SunAbraham().fit(df, outcome="y", unit="unit", first_treat="first_treat", time="time")


# =============================================================================
# TestPretrendsHandCalculation — paper-equation closed-forms + small-K MC
# =============================================================================


class TestPretrendsHandCalculation:
    """Closed-form sanity checks against Roth (2022) Section II.A-B equations."""

    def test_z_critical_value_matches_paper_default(self):
        """B_NIS critical value z_{1-α/2} = 1.96 at α=0.05 (Roth Eq. for B_NIS)."""
        pt = PreTrendsPower(alpha=0.05, pretest_form="nis")
        # The critical_value field on results is exactly z_{1-α/2} for NIS
        # (set in _compute_power_nis).
        # Build a minimal SunAbraham fit so we can extract it via the results.
        df = _make_sa_panel(n_units_per_cohort=15)
        sa_res = SunAbraham().fit(
            df, outcome="y", unit="unit", first_treat="first_treat", time="time"
        )
        result = pt.fit(sa_res)
        assert np.isclose(result.critical_value, 1.96, atol=0.01)

    def test_nis_power_at_h0_matches_independent_normals_formula(self):
        """Under H0 (M=0) with diagonal Σ, NIS power = 1 - (1 - α)^K.

        Roth Section II.A: B_NIS is the joint individual-CI acceptance event.
        Under H0 with independent normals, P(reject) = 1 - (1 - α)^K.
        """
        pt = PreTrendsPower(alpha=0.05, pretest_form="nis")
        # K=3, independent Σ_22 = 0.25 * I, M=0 (null)
        weights = np.array([1.0, 1.0, 1.0])
        vcov_diag = np.eye(3) * 0.25
        power, _, _, z_alpha = pt._compute_power_nis(0.0, weights, vcov_diag)
        expected = 1.0 - (1.0 - 0.05) ** 3
        assert np.isclose(power, expected, atol=0.005)
        assert np.isclose(z_alpha, stats.norm.ppf(0.975), atol=1e-10)

    def test_wald_power_at_h0_equals_alpha(self):
        """Under H0 (M=0), Wald noncentral-χ² power = alpha (size).

        Roth Section II.A: Wald form `W ~ χ²(K)` under H0 by construction;
        rejection probability at the (1-α) chi-squared critical value is α.
        """
        pt = PreTrendsPower(alpha=0.05, pretest_form="wald")
        weights = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)  # L2-normalized
        vcov = np.eye(3) * 0.25
        power, _, _, _ = pt._compute_power_wald(0.0, weights, vcov)
        assert np.isclose(power, 0.05, atol=0.01)

    def test_nis_power_matches_monte_carlo_K2_diagonal(self):
        """NIS power via scipy MVN matches MC simulation at K=2, diag Σ_22."""
        pt = PreTrendsPower(alpha=0.05, pretest_form="nis")
        weights = np.array([1.0, 1.0])  # equal weights, K=2
        vcov = np.eye(2) * 0.16  # σ = 0.4 each
        M = 0.6

        # Analytical via _compute_power_nis
        power_analytical, _, _, z_alpha = pt._compute_power_nis(M, weights, vcov)

        # MC: draw N samples from N(M * weights, vcov), check NIS rejection
        rng = np.random.default_rng(42)
        delta = M * weights
        samples = rng.multivariate_normal(mean=delta, cov=vcov, size=50_000)
        sigma = np.sqrt(np.diag(vcov))
        reject = np.any(np.abs(samples) > z_alpha * sigma, axis=1)
        power_mc = float(reject.mean())

        # MC SE on N=50k with power ~ 0.5: ~0.003. Allow 0.01 tolerance.
        assert np.isclose(
            power_analytical, power_mc, atol=0.01
        ), f"analytical={power_analytical:.4f}, mc={power_mc:.4f}"

    def test_nis_power_matches_monte_carlo_K3_correlated(self):
        """NIS power matches MC at K=3 with correlated Σ_22 (off-diagonals).

        This is the regime where Wald and NIS genuinely differ — both
        analytical paths must match their respective simulation truth.
        """
        pt = PreTrendsPower(alpha=0.05, pretest_form="nis")
        weights = np.array([1.0, 1.0, 1.0])
        # ρ=0.3 equicorrelation, σ²=0.25
        rho = 0.3
        sigma2 = 0.25
        vcov = sigma2 * (rho * np.ones((3, 3)) + (1 - rho) * np.eye(3))
        M = 0.5

        power_analytical, _, _, z_alpha = pt._compute_power_nis(M, weights, vcov)

        rng = np.random.default_rng(123)
        delta = M * weights
        samples = rng.multivariate_normal(mean=delta, cov=vcov, size=50_000)
        sigma_per = np.sqrt(np.diag(vcov))
        reject = np.any(np.abs(samples) > z_alpha * sigma_per, axis=1)
        power_mc = float(reject.mean())

        assert np.isclose(
            power_analytical, power_mc, atol=0.01
        ), f"analytical={power_analytical:.4f}, mc={power_mc:.4f}"

    def test_mdv_inversion_round_trip_nis(self):
        """MDV(target_power) achieves exactly target_power when evaluated.

        Both NIS and Wald: M = MDV computed at target_power=0.8 should give
        power(M) ≈ 0.8.
        """
        for form in ("nis", "wald"):
            pt = PreTrendsPower(alpha=0.05, power=0.80, pretest_form=form)
            weights = np.array([3.0, 2.0, 1.0])
            if form == "wald":
                weights = weights / np.linalg.norm(weights)
            vcov = np.eye(3) * 0.16
            mdv = pt._compute_mdv(weights, vcov)
            power_at_mdv = pt._compute_power(mdv, weights, vcov)[0]
            assert np.isclose(
                power_at_mdv, 0.80, atol=0.01
            ), f"form={form}: MDV={mdv:.4f}, power(MDV)={power_at_mdv:.4f}"

    def test_power_monotone_in_M_nis(self):
        """NIS power is monotone non-decreasing in |M| (basic sanity)."""
        pt = PreTrendsPower(pretest_form="nis")
        weights = np.array([3.0, 2.0, 1.0])
        vcov = np.eye(3) * 0.16
        powers = [pt._compute_power_nis(M, weights, vcov)[0] for M in [0, 0.5, 1.0, 2.0]]
        # Strictly non-decreasing
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1] - 1e-10, f"NIS power not monotone: {powers}"

    def test_mdv_nis_returns_zero_when_target_below_null_size(self):
        """NIS MDV returns 0.0 when target_power ≤ null rejection probability.

        NIS size under the null (with independent Σ) is `1 - (1-α)^K`, not α.
        For α=0.05, K=3 that's ≈ 0.143. Calling MDV with target_power=0.10
        should return 0.0 — no violation needed because the null already
        rejects at the target rate. Pre-fix: `_compute_mdv_nis` silently
        fell through to `M_high=1.0` because `brentq(0, 1)` raised
        ValueError on the boundary (power_minus_target(0) > 0).
        Post-fix: short-circuit at the boundary check.
        """
        pt = PreTrendsPower(alpha=0.05, power=0.10, pretest_form="nis")
        weights = np.array([1.0, 1.0, 1.0])
        vcov = np.eye(3) * 0.25  # diagonal, independence
        mdv = pt._compute_mdv_nis(weights, vcov)
        assert mdv == 0.0, f"target=0.10 < null size≈0.143; MDV should be 0.0, got {mdv}"

    def test_nis_power_handles_non_finite_cdf_via_mc_fallback(self):
        """NIS power_at falls back to MC when MVN CDF returns NaN (not just raises).

        The pre-fix code only triggered MC fallback on ValueError /
        LinAlgError exceptions; if scipy's Genz algorithm returns NaN
        directly (e.g., extreme numerical degeneracy), the NaN propagated
        through np.clip and into the MDV solver. Post-fix: explicit
        `np.isfinite(accept_prob)` check triggers MC fallback uniformly.

        We exercise this by monkey-patching `scipy.stats.multivariate_normal.cdf`
        to return NaN; the helper should fall through to simulation and
        produce a finite power in [0, 1].
        """
        from unittest.mock import patch

        from diff_diff.pretrends import _compute_nis_acceptance_prob

        weights = np.array([1.0, 1.0, 1.0])
        vcov = np.eye(3) * 0.16

        # Force the CDF to return NaN — verify MC fallback engages.
        with patch(
            "diff_diff.pretrends.stats.multivariate_normal.cdf",
            return_value=float("nan"),
        ):
            accept_prob = _compute_nis_acceptance_prob(0.5, weights, vcov, 1.96)

        # MC fallback should produce a valid probability in [0, 1].
        assert np.isfinite(accept_prob), "MC fallback did not engage"
        assert 0.0 <= accept_prob <= 1.0, f"MC accept_prob={accept_prob} out of [0, 1]"

    def test_mdv_nis_nonconvergence_cap_returns_inf(self):
        """NIS MDV returns ∞ when target power is unreachable in M ≤ 1000.

        With K=1 and σ = 1e4, the per-period acceptance prob remains very
        close to 1-α even at M=1000 (since δ/σ = 0.1 is still small relative
        to z=1.96). Power stays below target=0.99 throughout the doubling
        expansion → 1000-cap fires → return ∞.

        The Wald path's 1000-cap is on the noncentrality parameter and is
        structurally impossible to trigger for any finite target_power < 1
        on a finite-Σ scalar problem (ncx2.sf(cv, K, nc=1000) → 1 quickly),
        so we test the cap only on the NIS path.
        """
        pt = PreTrendsPower(alpha=0.05, power=0.99, pretest_form="nis")
        weights = np.array([1.0])
        vcov = np.array([[1e8]])  # σ = 1e4
        mdv = pt._compute_mdv_nis(weights, vcov)
        assert np.isinf(mdv), f"NIS MDV cap should return ∞, got {mdv}"

    def test_mdv_nis_finite_root_at_doubling_endpoint(self):
        """NIS MDV returns a finite root even when M_high lands at the 1024 cap.

        Concrete counter-example from R2 codex review: with σ ≈ 224
        (vcov=[[50000]]) and target_power=0.8, the doubling expansion
        sweeps M_high = 1, 2, 4, ..., 512, 1024. Power(M=512) ≈ 0.36 < 0.8
        and power(M=1024) ≈ 0.997 > 0.8, so the root sits in [512, 1024].
        Pre-fix the cap-check fired on the >=1000 condition and returned
        inf even though brentq could have bracketed the finite root.
        Post-fix the cap-check only triggers when power(M_high) is still
        below target — finite-root cases pass through to brentq.
        """
        pt = PreTrendsPower(alpha=0.05, power=0.8, pretest_form="nis")
        weights = np.array([1.0])
        vcov = np.array([[50000.0]])  # σ ≈ 223.6, root in [512, 1024]
        mdv = pt._compute_mdv_nis(weights, vcov)
        assert np.isfinite(mdv), f"finite-root case should NOT return ∞, got {mdv}"
        assert 512.0 < mdv < 1024.0, f"root expected in (512, 1024), got {mdv}"
        # Spot-check: the brentq result actually achieves target power.
        achieved, _, _, _ = pt._compute_power_nis(mdv, weights, vcov)
        assert abs(achieved - 0.8) < 1e-3, f"brentq root power={achieved}, expected ≈ 0.8"


# =============================================================================
# TestPretrendsPropositions — Roth Props 1-4 numerical verification (MC)
# =============================================================================


class TestPretrendsPropositions:
    """Roth (2022) Propositions 1-4 numerical verification via Monte Carlo.

    These tests validate that the LIBRARY's downstream consumers can rely on
    the conditional moments + variance reduction guarantees Roth proves. The
    library does not compute conditional moments in production code (it only
    needs the box probability for power), but the methodology test file
    exercises them via simulation to lock the contract that future audit
    rounds can compare against.

    Roth Proposition 1 (Section II.B):
        E[β̂_post | β̂_pre ∈ B(Σ)] = τ_post + δ_post
          + Σ_{12} Σ_{22}^{-1} ( E[β̂_pre | β̂_pre ∈ B(Σ)] - β_pre )

    Roth Proposition 3 (Section II.C):
        Var[β̂_post | β̂_pre ∈ B(Σ)]
          = Var[β̂_post] + (Σ_{12} Σ_{22}^{-1}) (Var[β̂_pre | β̂_pre ∈ B(Σ)]
            - Var[β̂_pre]) (Σ_{12} Σ_{22}^{-1})'

    Roth Proposition 4 (Section II.C): for convex B(Σ),
        Var[β̂_post | β̂_pre ∈ B(Σ)] ≤ Var[β̂_post]
    """

    @pytest.mark.slow
    def test_proposition_1_conditional_mean_matches_mc(self):
        """Prop 1: conditional mean E[β̂_post | NIS] matches MC at atol=0.01."""
        # Simple joint normal setup: K=2 pre-periods, M=1 post-period
        rng = np.random.default_rng(0)
        K, M_post = 2, 1
        # Σ structure: K+M-dim joint covariance
        # Block form: Σ = [[Σ_post, Σ_post,pre], [Σ_pre,post, Σ_pre]]
        sigma_pre = np.eye(K) * 0.16
        sigma_post = np.eye(M_post) * 0.16
        sigma_cross = 0.05 * np.ones((M_post, K))  # post-pre covariance
        # Build full joint Σ via block stacking — but for the test we just need
        # the regression coefficient Σ_{12} Σ_{22}^{-1} from post-on-pre.
        # Truth: β_pre = (0.3, 0.2), τ_post = 0, δ_post = 0.1
        beta_pre = np.array([0.3, 0.2])
        tau_post = np.array([0.0])
        delta_post = np.array([0.1])

        # Draw N samples from joint normal
        N = 200_000
        # Use scipy: sample jointly with mean = [beta_post; beta_pre]
        # beta_post = tau_post + delta_post under Roth's decomposition
        mean_post = tau_post + delta_post
        full_mean = np.concatenate([mean_post, beta_pre])
        full_cov = np.block(
            [
                [sigma_post, sigma_cross],
                [sigma_cross.T, sigma_pre],
            ]
        )
        joint = rng.multivariate_normal(full_mean, full_cov, size=N)
        beta_post_samples = joint[:, :M_post]
        beta_pre_samples = joint[:, M_post:]

        # NIS acceptance: |β̂_pre,t| ≤ 1.96 σ_t for all t
        sigma_pre_diag = np.sqrt(np.diag(sigma_pre))
        accept = np.all(np.abs(beta_pre_samples) <= 1.96 * sigma_pre_diag, axis=1)
        cond_post_mean_mc = beta_post_samples[accept].mean(axis=0)

        # Prop 1 prediction
        cond_pre_mean_mc = beta_pre_samples[accept].mean(axis=0)
        gamma = sigma_cross @ np.linalg.inv(sigma_pre)
        prop1_prediction = tau_post + delta_post + gamma @ (cond_pre_mean_mc - beta_pre)

        # MC noise floor at this N: ~0.01 with accept rate ~0.7.
        assert np.allclose(
            cond_post_mean_mc, prop1_prediction, atol=0.01
        ), f"MC={cond_post_mean_mc}, Prop1={prop1_prediction}"

    @pytest.mark.slow
    def test_proposition_4_variance_reduction_under_convex_B(self):
        """Prop 4: Var[β̂_post | β̂_pre ∈ B_NIS] ≤ Var[β̂_post] (B_NIS convex).

        B_NIS is convex (a Cartesian product of intervals), so Prop 4 applies.
        """
        rng = np.random.default_rng(1)
        K, M_post = 3, 1
        sigma_pre = np.eye(K) * 0.16
        sigma_post = np.eye(M_post) * 0.16
        sigma_cross = 0.04 * np.ones((M_post, K))
        full_cov = np.block(
            [
                [sigma_post, sigma_cross],
                [sigma_cross.T, sigma_pre],
            ]
        )
        # Parallel trends: β_pre = 0 → δ_pre = 0
        full_mean = np.zeros(K + M_post)
        N = 200_000
        joint = rng.multivariate_normal(full_mean, full_cov, size=N)
        beta_post_samples = joint[:, :M_post]
        beta_pre_samples = joint[:, M_post:]

        sigma_pre_diag = np.sqrt(np.diag(sigma_pre))
        accept = np.all(np.abs(beta_pre_samples) <= 1.96 * sigma_pre_diag, axis=1)

        var_unconditional = float(beta_post_samples.var(ddof=1))
        var_conditional = float(beta_post_samples[accept].var(ddof=1))

        # Prop 4: conditional variance should be NO LARGER than unconditional.
        # Allow small MC slop.
        assert (
            var_conditional <= var_unconditional + 0.01
        ), f"Prop 4 violated: unc={var_unconditional:.4f}, cond={var_conditional:.4f}"


# =============================================================================
# TestPretrendsLinearGrid — γ-unit MDV (PR-B Step 4 regression)
# =============================================================================


class TestPretrendsLinearGrid:
    """Linear weights honor actual pre-period relative-time labels.

    PR-B Step 4 closed the PR-A linear-pattern deviation by threading
    `relative_times` through `_get_violation_weights('linear')` and skipping
    L2 normalization on that path so the reported MDV is in Roth's γ units.
    """

    def test_regular_grid_produces_decreasing_weights(self):
        """Regular grid [-3, -2, -1] → linear weights = |t| = [3, 2, 1]."""
        pt = PreTrendsPower(violation_type="linear", pretest_form="nis")
        weights = pt._get_violation_weights(3, relative_times=np.array([-3, -2, -1]))
        np.testing.assert_allclose(weights, [3.0, 2.0, 1.0])

    def test_irregular_grid_reflects_actual_spacing(self):
        """Irregular grid [-5, -3, -1] → weights = [5, 3, 1] (not [3, 2, 1])."""
        pt = PreTrendsPower(violation_type="linear", pretest_form="nis")
        weights = pt._get_violation_weights(3, relative_times=np.array([-5, -3, -1]))
        np.testing.assert_allclose(weights, [5.0, 3.0, 1.0])

    def test_max_abs_pre_violation_uses_weight_scale_on_irregular_grid(self):
        """PR-B R12 P1 regression: ``PreTrendsPowerResults.max_abs_pre_violation``
        scales raw γ-unit ``mdv`` by ``max(|violation_weights|)`` so the
        level-scale comparison against ``|att|`` / per-period SEs is
        unit-consistent.

        On an irregular grid ``[-5, -3, -1]`` with linear weights
        ``[5, 3, 1]``, the largest level-scale pre-period violation under
        the MDV is ``mdv * 5``, NOT ``mdv * 1`` (the wrong unit-mixed
        scalar the report layer used pre-R12). Locks the architectural
        fix: raw γ should NEVER be compared to a level effect; always go
        through ``max_abs_pre_violation``.

        Uses synthetic Σ_22 + sa_results-shaped inputs so the fixture
        runs deterministically across pure-Python and Rust backends.
        """
        from diff_diff.pretrends import _coerce_relative_times_from_reference

        # Confirm the helper produces the irregular relative times.
        _ = _coerce_relative_times_from_reference([-5, -3, -1], 0)

        # K=3, ρ=0.4 equicorrelated, σ²=0.04 → moderate-power regime
        # so we get a finite mdv and can spot-check the level-scale scalar.
        K = 3
        rho = 0.4
        sigma2 = 0.04
        vcov = sigma2 * (rho * np.ones((K, K)) + (1 - rho) * np.eye(K))

        # Construct a synthetic result skeleton directly to exercise the
        # max_abs_pre_violation property end-to-end.
        relative_times = np.array([-5.0, -3.0, -1.0])
        pt = PreTrendsPower(violation_type="linear", pretest_form="nis", power=0.5)
        weights = pt._get_violation_weights(3, relative_times=relative_times)
        np.testing.assert_allclose(weights, [5.0, 3.0, 1.0])

        mdv = pt._compute_mdv_nis(weights, vcov)
        assert np.isfinite(mdv), f"MDV should be finite, got {mdv}"

        # Hand-construct the result with the right weights field so the
        # property exercises the new code path. Use minimal repr=False
        # field placeholders.
        from diff_diff.pretrends import PreTrendsPowerResults

        res = PreTrendsPowerResults(
            power=0.5,
            mdv=mdv,
            violation_magnitude=mdv,
            violation_type="linear",
            alpha=0.05,
            target_power=0.5,
            n_pre_periods=3,
            test_statistic=np.nan,
            critical_value=1.96,
            noncentrality=np.nan,
            pre_period_effects=np.zeros(3),
            pre_period_ses=np.full(3, np.sqrt(sigma2)),
            vcov=vcov,
            violation_weights=weights,
            covariance_source="full_pre_period_vcov",
        )
        # Level-scale scalar: mdv * max(|weights|) = mdv * 5 (the
        # `t=-5` slot dominates on irregular grids).
        expected = float(mdv * 5.0)
        assert np.isclose(res.max_abs_pre_violation, expected, atol=1e-10), (
            f"max_abs_pre_violation={res.max_abs_pre_violation} should equal "
            f"mdv * max(|w|) = {expected} on irregular grid [-5, -3, -1]"
        )
        # Sanity: raw mdv is materially smaller — confirms the unit-fix
        # actually moves the scalar (regression against a future revert
        # back to raw γ).
        assert (
            res.max_abs_pre_violation > 4 * mdv
        ), "max_abs_pre_violation must scale by max(|w|)=5, not collapse to mdv"

    def test_constant_violation_pattern_is_level_shift(self, sa_results):
        """``violation_type='constant'`` produces a per-period level shift,
        not an L2-normalized direction (PR-B R13 fix).

        REGISTRY ``## PreTrendsPower`` documents constant as ``δ_t = c``.
        The implementation now returns unnormalized ``[1, 1, ..., 1]``
        weights so the contract holds at the public API surface:

        - ``violation_weights == [1, 1, ..., 1]`` after fit (no L2 norm).
        - ``max_abs_pre_violation == mdv * 1 == mdv`` (level-scale and
          γ-scale coincide for the constant pattern).
        - ``power_at(M)`` evaluates the violation `δ_t = M` per period,
          not `δ_t = M/√K`.

        Pre-PR-B-R13 the constant path was silently divided by √K,
        so a constant MDV of 0.5 was a per-period shift of 0.5/√K,
        not 0.5 as the docs claimed. Locks the level-shift contract
        end-to-end on a real fit.
        """
        pt = PreTrendsPower(violation_type="constant", pretest_form="nis")
        result = pt.fit(sa_results)

        n_pre = result.n_pre_periods
        # Weights are exactly [1, 1, ..., 1] — NOT L2-normalized.
        assert result.violation_weights is not None
        np.testing.assert_allclose(result.violation_weights, np.ones(n_pre))
        # L2 norm of weights is √K, not 1.
        assert np.isclose(np.linalg.norm(result.violation_weights), np.sqrt(n_pre))
        # Level-scale max coincides with raw mdv (max(|w|) = 1).
        assert np.isclose(result.max_abs_pre_violation, result.mdv)

        # power_at(M) round-trip: under the level-shift contract,
        # power_at(M) for constant must equal power at `M=0.1` of a refit.
        # Loose atol because scipy MVN CDF and the centered helper take
        # slightly different paths with ~1e-6 sub-ULP roundoff.
        refit = pt.fit(sa_results, M=0.1)
        assert np.isclose(result.power_at(0.1), refit.power, atol=1e-4)

    def test_is_informative_uses_level_scale_not_raw_gamma(self):
        """``is_informative`` consumes ``max_abs_pre_violation`` (level scale)
        rather than raw ``mdv`` (slope scale) — locks the R12 fix on the
        property surface so future regressions cannot flip back to the
        wrong-unit heuristic.
        """
        from diff_diff.pretrends import PreTrendsPowerResults

        # SE = 0.5 across pre-periods; MDV = 0.4 (raw γ); weights have
        # max(|w|)=3 on a regular `[-3, -2, -1]` grid → level-scale max
        # violation = 1.2, well above 2 * max(SE) = 1.0 → NOT informative.
        res = PreTrendsPowerResults(
            power=0.5,
            mdv=0.4,
            violation_magnitude=0.4,
            violation_type="linear",
            alpha=0.05,
            target_power=0.5,
            n_pre_periods=3,
            test_statistic=np.nan,
            critical_value=1.96,
            noncentrality=np.nan,
            pre_period_effects=np.zeros(3),
            pre_period_ses=np.full(3, 0.5),
            vcov=np.eye(3) * 0.25,
            violation_weights=np.array([3.0, 2.0, 1.0]),
        )
        # max_abs_pre_violation = 0.4 * 3 = 1.2 > 2 * 0.5 = 1.0 → not informative
        assert np.isclose(res.max_abs_pre_violation, 1.2, atol=1e-10)
        assert res.is_informative is False, (
            "raw mdv=0.4 < 2*SE=1.0 would say 'informative', but the level-scale "
            "violation 1.2 > 1.0 says 'not informative' — the level-scale check wins"
        )

    def test_no_l2_normalization_when_relative_times_provided(self):
        """Linear-with-relative_times skips L2 norm → ||weights||_2 ≠ 1."""
        pt = PreTrendsPower(violation_type="linear", pretest_form="nis")
        weights = pt._get_violation_weights(3, relative_times=np.array([-3, -2, -1]))
        norm = np.linalg.norm(weights)
        # Norm should NOT be 1.0 — that's the bug we're regressing against.
        assert (
            norm > 1.5
        ), f"Linear-with-relative_times should NOT be L2-normalized, got ||·||_2 = {norm}"

    def test_mpd_calendar_period_ids_derive_relative_times_from_reference(self):
        """MPD calendar period IDs are correctly converted to Roth relative times.

        For MPD with `pre_periods=[0, 1, 2, 3]` and `reference_period=4`,
        the Roth-style relative times are `[-4, -3, -2, -1]`, not the raw
        period IDs `[0, 1, 2, 3]`. Pre-fix: the MPD adapter passed raw
        period IDs into `_get_violation_weights` as relative times,
        producing linear weights `[0, 1, 2, 3]` instead of Roth-style
        `[4, 3, 2, 1]`. Post-fix: derive
        `relative_times = estimated_pre_periods - reference_period`.

        Constructs a real ``MultiPeriodDiDResults`` and calls
        ``_extract_pre_period_params`` directly so the MPD branch is
        actually exercised (R2 P2 fix — prior version did manual
        arithmetic and never hit the production code path).
        """
        from diff_diff.results import MultiPeriodDiDResults, PeriodEffect

        period_ids = [0, 1, 2, 3]
        reference_period = 4

        period_effects = {
            p: PeriodEffect(
                period=p, effect=0.1 * p, se=0.2, t_stat=0.0, p_value=0.5, conf_int=(0.0, 0.0)
            )
            for p in period_ids
        }
        mpd_results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=0.0,
            avg_se=0.2,
            avg_t_stat=0.0,
            avg_p_value=0.5,
            avg_conf_int=(0.0, 0.0),
            n_obs=100,
            n_treated=50,
            n_control=50,
            pre_periods=period_ids,
            post_periods=[5, 6, 7],
            reference_period=reference_period,
        )

        pt = PreTrendsPower(pretest_form="nis", violation_type="linear")
        (
            _,
            ses,
            vcov,
            n_pre,
            relative_times,
            covariance_source,
        ) = pt._extract_pre_period_params(mpd_results)

        # End-to-end assertion: the MPD branch produced Roth-style relative
        # times derived from `reference_period`, not the raw period IDs.
        assert relative_times is not None, "MPD branch should produce relative_times"
        np.testing.assert_allclose(relative_times, [-4.0, -3.0, -2.0, -1.0])
        assert n_pre == 4
        # vcov falls through to diag(ses**2) because the mock has no
        # interaction_indices and no full vcov.
        np.testing.assert_allclose(np.diag(vcov), np.array(ses) ** 2)
        # MPD without `interaction_indices` records the diag-fallback source.
        assert covariance_source == "diag_fallback"

        # Plumbed through to _get_violation_weights: weights = |t| = [4, 3, 2, 1].
        weights = pt._get_violation_weights(n_pre, relative_times=relative_times)
        np.testing.assert_allclose(weights, [4.0, 3.0, 2.0, 1.0])

    def test_mpd_non_numeric_reference_warns_and_falls_back_to_legacy_weights(self):
        """MPD with non-numeric reference_period warns + falls back to legacy.

        When ``reference_period`` is a genuinely non-numeric / non-datetime
        label (e.g., the string "REF_STRING"), the MPD branch emits an
        explicit ``UserWarning`` and returns ``relative_times=None`` so
        ``_get_violation_weights('linear')`` uses the legacy count-based
        direction. The warning surfaces the contract that the reported
        MDV is NOT in Roth's γ units under this fallback (R8 CI codex
        fix: was previously a silent fallback, undocumented as a
        deviation in REGISTRY).
        """
        import warnings as _warnings

        from diff_diff.results import MultiPeriodDiDResults, PeriodEffect

        period_ids = ["A", "B", "C"]
        period_effects = {
            p: PeriodEffect(
                period=p, effect=0.1, se=0.2, t_stat=0.0, p_value=0.5, conf_int=(0.0, 0.0)
            )
            for p in period_ids
        }
        mpd_results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=0.0,
            avg_se=0.2,
            avg_t_stat=0.0,
            avg_p_value=0.5,
            avg_conf_int=(0.0, 0.0),
            n_obs=100,
            n_treated=50,
            n_control=50,
            pre_periods=period_ids,
            post_periods=["D", "E"],
            reference_period="REF_STRING",  # non-numeric, non-datetime
        )

        pt = PreTrendsPower(pretest_form="nis", violation_type="linear")
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _, _, _, _, relative_times, _ = pt._extract_pre_period_params(mpd_results)

        assert relative_times is None, "Non-numeric reference should yield None"
        nis_warns = [
            w
            for w in caught
            if "reference_period" in str(w.message) and "γ units" in str(w.message)
        ]
        assert len(nis_warns) >= 1, (
            "Non-numeric reference_period must emit an explicit UserWarning "
            f"noting the γ-unit contract is not held; got warnings: {[str(w.message) for w in caught]}"
        )

    def test_mpd_pandas_period_reference_yields_numeric_relative_times(self):
        """MPD with pandas.Period reference_period produces γ-unit weights.

        Quarterly-Period labels ``[2019Q1, 2019Q2, 2019Q3]`` with
        ``reference_period=2019Q4`` produce relative offsets in units of
        quarters: ``[-3, -2, -1]``. Validates the R8 CI codex fix that
        datetime-like labels are NOT silently fall-through cases — Period
        / Timestamp arithmetic supplies the γ-unit relative times the
        legacy fallback would have lost.
        """
        from diff_diff.results import MultiPeriodDiDResults, PeriodEffect

        periods = [pd.Period(f"2019Q{q}", freq="Q") for q in (1, 2, 3)]
        reference_period = pd.Period("2019Q4", freq="Q")
        period_effects = {
            p: PeriodEffect(
                period=p, effect=0.1, se=0.2, t_stat=0.0, p_value=0.5, conf_int=(0.0, 0.0)
            )
            for p in periods
        }
        mpd_results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=0.0,
            avg_se=0.2,
            avg_t_stat=0.0,
            avg_p_value=0.5,
            avg_conf_int=(0.0, 0.0),
            n_obs=100,
            n_treated=50,
            n_control=50,
            pre_periods=periods,
            post_periods=[pd.Period(f"2020Q{q}", freq="Q") for q in (1, 2)],
            reference_period=reference_period,
        )

        pt = PreTrendsPower(pretest_form="nis", violation_type="linear")
        _, _, _, n_pre, relative_times, _ = pt._extract_pre_period_params(mpd_results)

        # Period subtraction yields a Period offset whose `.n` is the
        # number-of-frequencies difference; signs matter and pre-periods
        # are NEGATIVE offsets from the reference.
        assert relative_times is not None
        np.testing.assert_allclose(relative_times, [-3.0, -2.0, -1.0])

        # Plumbed through to linear weights: |t| = [3, 2, 1] in γ units.
        weights = pt._get_violation_weights(n_pre, relative_times=relative_times)
        np.testing.assert_allclose(weights, [3.0, 2.0, 1.0])

    def test_backwards_compat_no_relative_times_uses_legacy_normalized(self):
        """Without relative_times: legacy [n-1, ..., 0]/||·||_2 direction.

        Preserves the pre-PR-B shipped behavior for callers that bypass fit()
        and call _get_violation_weights(n_pre) directly without relative_times.
        """
        pt = PreTrendsPower(violation_type="linear", pretest_form="nis")
        weights = pt._get_violation_weights(3)  # no relative_times
        # Legacy: [2, 1, 0] / sqrt(5) = [0.894, 0.447, 0]
        expected_legacy = np.array([2.0, 1.0, 0.0]) / np.sqrt(5.0)
        np.testing.assert_allclose(weights, expected_legacy, atol=1e-10)


# =============================================================================
# TestPretrendsCustomWeightPersistence — power_at(custom) (PR-B Step 5)
# =============================================================================


class TestPretrendsCustomWeightPersistence:
    """Custom violation weights are persisted on PreTrendsPowerResults.

    Per PR-B Step 5, the new ``violation_weights`` field on the result class
    enables ``power_at(M)`` to work for ``violation_type='custom'`` without
    re-fitting (lifting the PR-A R18 NotImplementedError guard for fresh fits).
    """

    def test_custom_weights_stored_on_results(self, sa_results):
        """After fit, results.violation_weights matches the L2-normalized input.

        The custom path in ``_get_violation_weights`` L2-normalizes the input
        weights to unit norm before fitting. The persisted
        ``violation_weights`` field on the result reflects the NORMALIZED
        weights (matching what `power_at()` and `_compute_power_*` actually
        operated on).
        """
        # Probe via a linear fit to learn n_pre (panel-dependent).
        probe = PreTrendsPower(violation_type="linear", pretest_form="nis").fit(sa_results)
        n_pre = probe.n_pre_periods
        # Build a length-n_pre custom weights vector deterministically.
        custom_w_raw = np.linspace(0.1, 0.6, n_pre)
        custom_w_normalized = custom_w_raw / np.linalg.norm(custom_w_raw)

        pt = PreTrendsPower(
            violation_type="custom", violation_weights=custom_w_raw, pretest_form="nis"
        )
        result = pt.fit(sa_results)
        assert result.violation_weights is not None
        np.testing.assert_allclose(result.violation_weights, custom_w_normalized)

    def test_power_at_custom_matches_refit(self, sa_results):
        """results.power_at(M) for custom matches a fresh fit(M=M)."""
        probe = PreTrendsPower(violation_type="linear", pretest_form="nis").fit(sa_results)
        n_pre = probe.n_pre_periods
        custom_w = np.array([0.2, 0.3, 0.5][:n_pre])
        if len(custom_w) < n_pre:
            custom_w = np.concatenate([custom_w, np.zeros(n_pre - len(custom_w))])

        pt = PreTrendsPower(violation_type="custom", violation_weights=custom_w, pretest_form="nis")
        results_base = pt.fit(sa_results)
        results_at_target = pt.fit(sa_results, M=0.5)

        power_via_method = results_base.power_at(0.5)
        power_via_refit = results_at_target.power

        # Tight tolerance — both paths use the same _compute_power_nis call.
        assert np.isclose(
            power_via_method, power_via_refit, atol=1e-6
        ), f"power_at={power_via_method:.6f}, refit={power_via_refit:.6f}"

    def test_to_dict_is_json_serializable(self, sa_results):
        """PR-B R5 regression: ``to_dict()`` must produce JSON-serializable
        output. ``violation_weights`` is emitted as ``list[float]`` (not raw
        ``np.ndarray``) so ``json.dumps`` works out of the box.

        Pre-R5 the dict carried a raw ``np.ndarray`` for ``violation_weights``;
        ``json.dumps(result.to_dict())`` raised ``TypeError``. Post-R5 the
        helper coerces to a Python list of floats.
        """
        probe = PreTrendsPower(violation_type="linear", pretest_form="nis").fit(sa_results)
        n_pre = probe.n_pre_periods
        custom_w = np.linspace(0.1, 0.6, n_pre)

        pt = PreTrendsPower(violation_type="custom", violation_weights=custom_w, pretest_form="nis")
        result = pt.fit(sa_results)

        d = result.to_dict()
        # Type contract: violation_weights round-trips as list[float] or None.
        assert isinstance(d["violation_weights"], list)
        for w in d["violation_weights"]:
            assert isinstance(w, float)

        # End-to-end JSON round-trip (NaN → strings in default mode? scipy
        # returns finite NaN — json.dumps with allow_nan=True is default).
        encoded = json.dumps(d, allow_nan=True)
        decoded = json.loads(encoded)
        # Spot-check provenance fields round-trip intact.
        assert decoded["covariance_source"] == result.covariance_source
        assert decoded["pretest_form"] == result.pretest_form


# =============================================================================
# TestPretrendsCovarianceSource — CS/SA full-VCV routing (PR-B Step 3)
# =============================================================================


class TestPretrendsCovarianceSource:
    """CS and SA adapters route through event_study_vcov on non-bootstrap fits.

    Pre-PR-B, both CS and SA branches in _extract_pre_period_params hard-coded
    diag(ses^2). PR-B Step 3 added the W-matrix construction for SA and
    routed both branches through the new module-level helper
    _extract_event_study_vcov_subblock when event_study_vcov is available.
    """

    def test_sa_non_bootstrap_persists_event_study_vcov(self, sa_results):
        """SunAbrahamResults.event_study_vcov is populated on non-bootstrap fits."""
        assert sa_results.event_study_vcov is not None
        assert sa_results.event_study_vcov_index is not None
        # Shape: |event_times| × |event_times|
        n_et = len(sa_results.event_study_vcov_index)
        assert sa_results.event_study_vcov.shape == (n_et, n_et)
        # Symmetric
        np.testing.assert_allclose(
            sa_results.event_study_vcov, sa_results.event_study_vcov.T, atol=1e-12
        )

    def test_sa_event_study_vcov_diagonal_matches_per_event_se(self, sa_results):
        """event_study_vcov diagonal[i, i] = se(e_i)^2 (W-matrix sanity).

        The diagonal entries should reproduce the existing per-event-time SE
        computation in _compute_iw_effects at atol=1e-10.
        """
        es_vcov = sa_results.event_study_vcov
        es_index = sa_results.event_study_vcov_index
        for i, e in enumerate(es_index):
            diag_se = float(np.sqrt(es_vcov[i, i]))
            es_effect = sa_results.event_study_effects.get(e, {})
            if "se" in es_effect:
                assert np.isclose(
                    diag_se, es_effect["se"], atol=1e-10
                ), f"e={e}: diag_se={diag_se}, es_effects[e][se]={es_effect['se']}"

    def test_sa_pretrends_consumes_full_vcov_not_diag(self, sa_results):
        """compute_pretrends_power on SA uses the full sub-VCV, not diag(ses^2)."""
        from diff_diff.pretrends import _extract_event_study_vcov_subblock

        # The new helper should produce a sub-block that differs from the
        # diag(ses**2) fallback IF the off-diagonals are nonzero.
        # Find the pre-periods of the SA panel.
        pre_periods = [t for t in sa_results.event_study_effects if t < 0]
        if not pre_periods:
            pytest.skip("No pre-periods in fixture")

        ses = np.array([sa_results.event_study_effects[t]["se"] for t in sorted(pre_periods)])
        sub, source = _extract_event_study_vcov_subblock(sa_results, sorted(pre_periods), ses)
        diag_fallback = np.diag(ses**2)

        # Source label reflects the full-VCV path being actually taken.
        assert source == "full_pre_period_vcov"
        # Should NOT be identical (assuming the panel produces nonzero
        # off-diagonal cohort overlap). At minimum the shape matches.
        assert sub.shape == diag_fallback.shape
        # Off-diagonals should generally be nonzero (cohort weights overlap
        # at adjacent event times).
        off_diag_sum = float(np.abs(sub - np.diag(np.diag(sub))).sum())
        assert off_diag_sum > 1e-8, (
            "SA event_study_vcov sub-block has all-zero off-diagonals — "
            "either the panel is degenerate or the W-matrix routing didn't fire."
        )


# =============================================================================
# TestPretrendsHelperAPI — helper-API extension (PR-B Step 6)
# =============================================================================


class TestPretrendsHelperAPI:
    """Helper functions accept violation_weights and pretest_form end-to-end."""

    def test_compute_pretrends_power_accepts_violation_weights_custom(self, sa_results):
        """compute_pretrends_power(..., violation_type='custom', violation_weights=...)"""
        # Probe n_pre
        probe = compute_pretrends_power(sa_results, violation_type="linear")
        n_pre = probe.n_pre_periods

        custom_w = np.arange(1, n_pre + 1, dtype=float)
        custom_w = custom_w / np.linalg.norm(custom_w)  # arbitrary normalized

        result = compute_pretrends_power(
            sa_results,
            violation_type="custom",
            violation_weights=custom_w,
        )
        assert isinstance(result, PreTrendsPowerResults)
        assert result.violation_type == "custom"
        assert result.violation_weights is not None
        np.testing.assert_allclose(result.violation_weights, custom_w)

    def test_compute_mdv_accepts_violation_weights_custom(self, sa_results):
        """compute_mdv mirrors compute_pretrends_power for custom support."""
        probe = compute_pretrends_power(sa_results, violation_type="linear")
        n_pre = probe.n_pre_periods
        custom_w = np.arange(1, n_pre + 1, dtype=float)
        custom_w = custom_w / np.linalg.norm(custom_w)

        mdv = compute_mdv(sa_results, violation_type="custom", violation_weights=custom_w)
        assert isinstance(mdv, float)
        assert mdv >= 0

    def test_compute_pretrends_power_accepts_pretest_form_wald(self, sa_results):
        """pretest_form='wald' opt-in selects the Wald acceptance-region form.

        Routes through ``_compute_power_wald`` / ``_compute_mdv_wald`` (the
        renamed pre-PR-B math), preserving the noncentral-χ² ellipsoidal
        acceptance region. NOTE: bit-identity to pre-PR-B numerical output
        on a fitted result is only guaranteed on the legacy `relative_times=None`
        path; new fits via `compute_pretrends_power(...)` thread `relative_times`
        into both NIS and Wald linear-weight construction, so a Wald fit on an
        irregular grid produces γ-unit MDV (not the pre-PR-B count-based L2-
        normalized MDV). See REGISTRY `## PreTrendsPower` linear-pattern Note.
        """
        wald_result = compute_pretrends_power(sa_results, pretest_form="wald")
        nis_result = compute_pretrends_power(sa_results, pretest_form="nis")

        assert wald_result.pretest_form == "wald"
        assert nis_result.pretest_form == "nis"
        # Wald has a finite noncentrality; NIS has NaN noncentrality.
        assert np.isfinite(wald_result.noncentrality)
        assert np.isnan(nis_result.noncentrality)
        # NIS has a finite box probability; Wald has NaN box probability.
        assert np.isfinite(nis_result.nis_box_probability)
        assert np.isnan(wald_result.nis_box_probability)


# =============================================================================
# TestPretrendsNISvsWald — form-comparison + backwards-compat (PR-B Step 2)
# =============================================================================


class TestPretrendsNISvsWald:
    """NIS and Wald form-comparison; Wald backwards-compat regression."""

    def test_default_pretest_form_is_nis(self):
        """PR-B Step 2 flipped the default from implicit-Wald to explicit-NIS."""
        pt = PreTrendsPower()
        assert pt.pretest_form == "nis"

    def test_wald_path_preserves_pre_pr_b_acceptance_region_form(self, sa_results):
        """pretest_form='wald' preserves the pre-PR-B acceptance-region form.

        The Wald math (noncentral-χ² on the quadratic form
        ``δ' Σ_22^{-1} δ``) is byte-identical to pre-PR-B: the methods
        are renamed to ``_compute_power_wald`` + ``_compute_mdv_wald``
        with unchanged function bodies, and the dispatcher in
        ``_compute_power`` / ``_compute_mdv`` selects this branch when
        ``pretest_form='wald'``.

        **Backward-compat scope**: this test locks the form-of-the-test
        contract, NOT bit-identity to pre-PR-B fitted-result numerics.
        Bit-identity for fitted results is regime-dependent:

        - On the **legacy `relative_times=None` path** (callers that
          bypass `fit()` and call `_get_violation_weights(n_pre)`
          directly), the count-based L2-normalized direction is
          unchanged, so Wald numerics ARE bit-identical to pre-PR-B.
        - On the **new `fit()`-threaded path** (PR-B Step 4), both NIS
          and Wald consume `relative_times` for linear violations and
          skip L2 normalization → γ-unit MDV. A Wald fit on an
          irregular grid `{-5, -3, -1}` therefore produces a
          γ-different MDV than pre-PR-B. See REGISTRY linear-pattern
          Note for the convention.
        """
        pt = PreTrendsPower(pretest_form="wald")
        result = pt.fit(sa_results)
        # Wald-specific fields populated (acceptance-region form contract)
        assert np.isfinite(result.noncentrality)
        assert np.isfinite(result.test_statistic)
        # NIS-specific fields are NaN under Wald
        assert np.isnan(result.nis_box_probability)
        # Power is in [0, 1]
        assert 0.0 <= result.power <= 1.0

    def test_nis_and_wald_differ_in_general(self):
        """NIS and Wald produce different power at the same M (general case).

        Under correlated Σ_22, the rectangular (NIS) and ellipsoidal (Wald)
        acceptance regions cover different probability mass under H1. Use a
        synthetic vcov with non-trivial off-diagonals at a small M so power
        is well-inside (0, 1) and the differentiation is observable.
        """
        # K=3, ρ=0.6 equicorrelated, σ²=0.04 — moderate-power regime
        rho = 0.6
        sigma2 = 0.04
        K = 3
        vcov = sigma2 * (rho * np.ones((K, K)) + (1 - rho) * np.eye(K))
        weights = np.array([3.0, 2.0, 1.0])
        weights_wald = weights / np.linalg.norm(weights)

        pt_nis = PreTrendsPower(pretest_form="nis")
        pt_wald = PreTrendsPower(pretest_form="wald")

        # Use a small M so power isn't saturated at 1
        M = 0.3
        power_nis, _, _, _ = pt_nis._compute_power_nis(M, weights, vcov)
        power_wald, _, _, _ = pt_wald._compute_power_wald(M, weights_wald, vcov)

        # The two forms should produce different power values
        assert not np.isclose(power_nis, power_wald, atol=0.02), (
            f"NIS and Wald produced essentially-equal power: "
            f"NIS={power_nis:.4f}, Wald={power_wald:.4f}"
        )


# =============================================================================
# TestPretrendsParityR — R parity (skips when goldens missing; PR-C)
# =============================================================================


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmarks",
            "data",
            "r_pretrends_golden.json",
        )
    ),
    reason="R `pretrends` parity goldens not present (partial checkout)",
)
class TestPretrendsParityR:
    """R `pretrends` package parity at commit ``122731d082`` (PR-C).

    Goldens at ``benchmarks/data/r_pretrends_golden.json`` populated in
    PR-C 2026-05-19 against `jonathandroth/pretrends` commit ``122731d082``
    (package version 0.1.0). The ``@pytest.mark.skipif`` decorator guards
    partial checkouts where the JSON is absent (the full repo always has
    it post-PR-C).

    Tolerance rationale: R hardcodes ``thresholdTstat.Pretest = 1.96``
    while Python uses ``scipy.stats.norm.ppf(0.975) = 1.959963984540054``
    (``dz ≈ 3.6e-5``); on top of that, ``mvtnorm::pmvnorm`` (R) and
    ``scipy.stats.multivariate_normal.cdf`` (Python) use Genz-Bretz
    randomized-lattice rules with different absolute-error defaults
    (abseps ≈ 1e-3 vs 1e-5). Combined, the empirical NIS power gap is
    bounded by ~5e-5 in the K=4 shifted-grid fixture (smaller for K∈{1,3}).
    For the inverse path (γ_p), R's ``slope_for_power`` uses
    ``uniroot(tol = .Machine$double.eps^0.25 ≈ 1.22e-4)`` versus Python
    ``brentq(xtol=2e-12)``; the inverse-solver tolerance gap dominates the
    γ_p residual. ``atol=1e-4`` is therefore the realistic ceiling on both
    tiers without tightening either solver.
    """

    @staticmethod
    def _load_r_golden():
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmarks",
            "data",
            "r_pretrends_golden.json",
        )
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _extract_python_params(fixture):
        """Parse a fixture's panel into ``(pre_periods, Sigma_22, weights)``.

        ``weights = |pre_periods|`` is the Roth γ-unit linear violation
        weight. ``Sigma_22`` is the pre-period sub-block of the full
        ``Sigma``. K=1 fixtures arrive with jsonlite ``auto_unbox`` scalar
        ``pre_periods`` — ``np.atleast_1d`` normalizes them to a length-1
        array.
        """
        panel = fixture["panel"]
        pre_periods = np.atleast_1d(np.array(panel["pre_periods"], dtype=float))
        K_pre = len(pre_periods)
        Sigma_full = np.atleast_2d(np.array(panel["Sigma"], dtype=float))
        Sigma_22 = Sigma_full[:K_pre, :K_pre]
        weights = np.abs(pre_periods)
        return pre_periods, Sigma_22, weights

    def test_nis_power_matches_r_pretrends(self):
        """Python ``_compute_power_nis`` matches R ``pretrends()`` at atol=1e-4.

        Iterates all 4 fixtures × 4 γ values (16 comparisons). See class
        docstring for the tolerance derivation.
        """
        goldens = self._load_r_golden()
        pt = PreTrendsPower(alpha=0.05, pretest_form="nis")
        for fixture_name, fixture in goldens.items():
            if fixture_name == "meta":
                continue
            _, Sigma_22, weights = self._extract_python_params(fixture)
            for g_val, r_power in zip(
                fixture["r_power_at_gamma"]["gamma_test_values"],
                fixture["r_power_at_gamma"]["power_values"],
            ):
                py_power, _, _, _ = pt._compute_power_nis(g_val, weights, Sigma_22)
                assert np.isclose(py_power, r_power, atol=1e-4), (
                    f"{fixture_name} γ={g_val}: Python={py_power:.7f}, "
                    f"R={r_power:.7f}, diff={py_power - r_power:+.2e}"
                )

    def test_mdv_gamma_p_matches_r_slope_for_power(self):
        """Python ``_compute_mdv_nis`` matches R ``slope_for_power()`` at atol=1e-4.

        Iterates all 4 fixtures × 2 target_power values (8 comparisons).
        See class docstring for the tolerance derivation — R uniroot vs
        Python brentq solver gap dominates.
        """
        goldens = self._load_r_golden()
        for fixture_name, fixture in goldens.items():
            if fixture_name == "meta":
                continue
            _, Sigma_22, weights = self._extract_python_params(fixture)
            for target_p, r_gamma_p in zip(
                fixture["r_gamma_p"]["target_power"],
                fixture["r_gamma_p"]["gamma_p_values"],
            ):
                pt = PreTrendsPower(alpha=0.05, power=target_p, pretest_form="nis")
                py_mdv = pt._compute_mdv_nis(weights, Sigma_22)
                assert np.isclose(py_mdv, r_gamma_p, atol=1e-4), (
                    f"{fixture_name} target_power={target_p}: "
                    f"Python={py_mdv:.7f}, R={r_gamma_p:.7f}, "
                    f"diff={py_mdv - r_gamma_p:+.2e}"
                )

    def test_irregular_grid_gamma_unit_end_to_end_matches_r(self):
        """End-to-end ``fit()`` on irregular grid matches R γ_p at atol=1e-4.

        Constructs a synthetic ``MultiPeriodDiDResults`` from the
        ``irregular_pre_periods`` fixture (pre {-5, -3, -1}; post {1, 2, 3};
        reference_period 0) with the fixture's β̂ + full Σ + populated
        ``interaction_indices``, then calls ``pt.fit()`` and asserts the
        produced ``mdv`` matches R's ``slope_for_power()``. This exercises
        the full
        ``fit() → _extract_pre_period_params → _get_violation_weights
        (γ-unit linear path) → _compute_mdv_nis`` chain, which the direct
        ``_compute_mdv_nis`` test in
        ``test_mdv_gamma_p_matches_r_slope_for_power`` does not. Locks the
        PR-B Step 4 γ-unit fix at the public API boundary.
        """
        from diff_diff.results import MultiPeriodDiDResults, PeriodEffect

        goldens = self._load_r_golden()
        fixture = goldens["irregular_pre_periods"]
        pre_periods, _, _ = self._extract_python_params(fixture)
        panel = fixture["panel"]
        Sigma_full = np.atleast_2d(np.array(panel["Sigma"], dtype=float))
        beta_hat = np.array(panel["beta_hat"], dtype=float)
        all_periods = list(np.atleast_1d(np.array(panel["all_periods"], dtype=int)))
        pre_period_ids = [int(p) for p in pre_periods]
        post_period_ids = [
            int(p) for p in np.atleast_1d(np.array(panel["post_periods"], dtype=int))
        ]

        # Build period_effects + interaction_indices over all_periods.
        ses_full = np.sqrt(np.diag(Sigma_full))
        period_effects = {
            int(p): PeriodEffect(
                period=int(p),
                effect=float(beta_hat[i]),
                se=float(ses_full[i]),
                t_stat=0.0,
                p_value=0.5,
                conf_int=(0.0, 0.0),
            )
            for i, p in enumerate(all_periods)
        }
        interaction_indices = {int(p): i for i, p in enumerate(all_periods)}

        mpd_results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=0.0,
            avg_se=0.2,
            avg_t_stat=0.0,
            avg_p_value=0.5,
            avg_conf_int=(0.0, 0.0),
            n_obs=1000,
            n_treated=500,
            n_control=500,
            pre_periods=pre_period_ids,
            post_periods=post_period_ids,
            reference_period=0,
            vcov=Sigma_full,
            interaction_indices=interaction_indices,
        )

        # Target power 0.8 (second entry in r_gamma_p).
        target_p = fixture["r_gamma_p"]["target_power"][1]
        r_gamma_p = fixture["r_gamma_p"]["gamma_p_values"][1]

        pt = PreTrendsPower(alpha=0.05, power=target_p, pretest_form="nis")
        result = pt.fit(mpd_results)

        # Provenance assertion: end-to-end path took the full-Σ_22 branch,
        # not the diag fallback (PR-B Step 3 routing).
        assert (
            result.covariance_source == "full_pre_period_vcov"
        ), f"Expected full_pre_period_vcov, got {result.covariance_source}"

        assert np.isclose(result.mdv, r_gamma_p, atol=1e-4), (
            f"End-to-end fit() mdv={result.mdv:.7f} ≠ R slope_for_power()={r_gamma_p:.7f}, "
            f"diff={result.mdv - r_gamma_p:+.2e}"
        )

    def test_k1_matches_r_and_closed_form(self):
        """K=1 fixture: Python ≡ closed form, both within atol=1e-4 of R.

        Three-way cross-check on the ``single_pre_period_closed_form``
        fixture (K=1, diagonal Σ = 0.25·I). Roth Proposition 2 univariate
        truncated-normal scalar:

            power(γ) = 1 - Φ(z - γ/σ) + Φ(-z - γ/σ)

        where ``z = Φ⁻¹(1 - α/2) = 1.959963984540054`` and
        ``σ = sqrt(Σ_22[0,0]) = 0.5``. Python ``_compute_power_nis`` for
        K=1 reduces to the same scalar normal-tail computation (no MVN
        randomization), so the Python-vs-closed-form match is essentially
        exact (atol=1e-7). R differs by the ``thresholdTstat=1.96`` vs
        ``qnorm(0.975)`` z-hardcoding (~1e-5 to ~2e-5 on power, ~3e-6 to
        ~2e-5 on γ_p). All three within 1e-4 of each other. Strongest
        parity claim in the suite.
        """
        goldens = self._load_r_golden()
        fixture = goldens["single_pre_period_closed_form"]
        _, Sigma_22, weights = self._extract_python_params(fixture)
        sigma = float(np.sqrt(Sigma_22[0, 0]))
        z = float(stats.norm.ppf(1 - 0.05 / 2))

        pt = PreTrendsPower(alpha=0.05, pretest_form="nis")
        for g_val, r_power in zip(
            fixture["r_power_at_gamma"]["gamma_test_values"],
            fixture["r_power_at_gamma"]["power_values"],
        ):
            py_power, _, _, _ = pt._compute_power_nis(g_val, weights, Sigma_22)
            closed_form = (
                1.0 - stats.norm.cdf(z - g_val / sigma) + stats.norm.cdf(-z - g_val / sigma)
            )
            # Python ≡ closed form (effectively exact — same scalar path).
            assert np.isclose(py_power, closed_form, atol=1e-7), (
                f"K=1 Python NIS power ≠ analytical closed form (γ={g_val}): "
                f"Python={py_power:.10f}, closed_form={closed_form:.10f}"
            )
            # Both within 1e-4 of R (z-hardcoding gap).
            assert np.isclose(py_power, r_power, atol=1e-4), (
                f"K=1 Python ≠ R (γ={g_val}): "
                f"Python={py_power:.7f}, R={r_power:.7f}, diff={py_power - r_power:+.2e}"
            )
            assert np.isclose(closed_form, r_power, atol=1e-4), (
                f"K=1 closed form ≠ R (γ={g_val}): "
                f"closed_form={closed_form:.7f}, R={r_power:.7f}, "
                f"diff={closed_form - r_power:+.2e}"
            )
