"""Methodology verification tests for HeterogeneousAdoptionDiD.

Targets de Chaisemartin, Ciccia, D'Haultfoeuille & Knau (2026) arXiv:2405.04465v6,
*Difference-in-Differences Estimators When No Unit Remains Untreated*.

Equation walk-through:

- Eq. 3  / Theorem 1: Design 1' WAS = [E(delta_Y) - lim_{d down 0} E(delta_Y | D <= d)] / E(D).
                         The library estimates the boundary intercept via
                         bias-corrected local-linear (Phase 1c) and computes
                         ``att = (mean(delta_Y) - tau_bc) / mean(D)``; the
                         test class exercises both the simple-DGP case
                         (boundary intercept ~ 0) AND a nonzero-boundary-
                         intercept case (``delta_Y = c + beta*D + eps`` with
                         ``c != 0``) so the ``mean(delta_Y) - tau_bc``
                         subtraction is verified, not just the
                         ``tau_bc ~ 0`` special case.
- Eq. 7  / (Algorithm): local-linear estimator with bias-corrected CI
- Eq. 11 / Theorem 3:  WAS_{d_lower} under Assumption 6 (mass-point path)
- Theorem 4 (QUG):     T_lambda = (lambda + E_1) / E_2 limit law, lambda=0
                         under H_0: d_lower = 0
- Eq. 18 / (Algorithm): joint Stute pre-trends + homogeneity
                         (mean-independence and linearity nulls).
                         The trends_lin=True linear-trend-detrended
                         variant is shipped in the library (R-parity
                         locked against DIDHAD::did_had(trends_lin=TRUE)
                         in tests/test_did_had_parity.py) but is
                         OUT OF SCOPE for this methodology file (no
                         coverage duplication).
- Eq. 29 / Theorem 7:  T_hr = sqrt(G) (sigma2_lin - sigma2_diff) / sigma2_W

See:

- ``docs/methodology/papers/dechaisemartin-2026-review.md`` (paper review)
- ``docs/methodology/REGISTRY.md`` ``## HeterogeneousAdoptionDiD`` block
- ``METHODOLOGY_REVIEW.md`` ``HeterogeneousAdoptionDiD`` section

Companion files (NOT duplicated here):

- ``tests/test_did_had_parity.py``         (R chaisemartin::did_had parity, 5 tests, atol=1e-8)
- ``tests/test_nprobust_port.py``          (Calonico-Cattaneo-Farrell port at atol=1e-14)
- ``tests/test_bias_corrected_lprobust.py`` (weighted bias-corrected, atol=1e-12)
- ``tests/test_had.py``, ``tests/test_had_pretests.py`` (implementation-detail unit tests)

Class structure:

- ``TestHADTheorem1Design1Prime`` — Eq. 3 + Theorem 1 (Design 1' boundary-
  subtracted identification; tests both the simple zero-boundary DGP and a
  nonzero-boundary-intercept DGP)
- ``TestHADTheorem3MassPoint`` — Eq. 11 + Theorem 3 (WAS_{d_lower} via 2SLS sample-average)
- ``TestHADTheorem4QUG`` — Theorem 4 (QUG null test, limit law Exp(1)/Exp(1))
- ``TestHADTheorem7YatchewHR`` — Eq. 29 + Theorem 7 (heteroskedasticity-robust linearity)
- ``TestHADJointStute`` — Section 4.2 step 2 + 4.3 (joint Stute pre-trends + homogeneity)
- ``TestHADDeviations`` — locks library deviations: equal-weighting, sup-t gating,
  staggered-timing fail-closed, safe_inference invariant
"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff import (
    HeterogeneousAdoptionDiD,
    HeterogeneousAdoptionDiDEventStudyResults,
    HeterogeneousAdoptionDiDResults,
    SurveyDesign,
    joint_homogeneity_test,
    joint_pretrends_test,
    qug_test,
    yatchew_hr_test,
)

# Per-test sub-seed bases (decorrelates MC tests within a class to avoid
# seed-correlation flake — review Medium #1 + Question #1).
_BASE_SEED_THEOREM1 = 4242
_BASE_SEED_THEOREM3 = 3333
_BASE_SEED_THEOREM4 = 5151
_BASE_SEED_THEOREM7 = 2929
_BASE_SEED_JOINT_STUTE = 7373
_BASE_SEED_DEVIATIONS = 9090


# =============================================================================
# Helpers — build minimal two-period HAD panels for direct estimator calls
# =============================================================================


def _make_two_period_panel(
    rng: np.random.Generator,
    G: int,
    *,
    dose_dist: str,
    was_true: float,
    sigma: float = 0.1,
    d_lower: float = 0.0,
    boundary_intercept: float = 0.0,
) -> pd.DataFrame:
    """Build a balanced two-period HAD panel.

    Period 1: D = 0 for all units (HAD pre-period contract).
    Period 2: D drawn from ``dose_dist`` on ``[d_lower, ...]``; outcome
    delta = boundary_intercept + was_true * D + N(0, sigma).

    Population WAS = was_true regardless of ``boundary_intercept``,
    because Eq. 3 / Theorem 1 subtracts off the boundary limit:
    ``WAS = (E[ΔY] - lim_{d↓0} E[ΔY | D ≤ d]) / E[D]
          = (boundary_intercept + was_true * E[D] - boundary_intercept) / E[D]
          = was_true``.
    Setting ``boundary_intercept != 0`` makes the library's
    ``att = (mean(ΔY) - τ_bc) / mean(D)`` actually exercise the
    ``τ_bc`` subtraction term (otherwise τ_bc ~ 0 and the test only
    verifies the ``mean(ΔY) / mean(D)`` ratio).
    """
    if dose_dist == "uniform_0_1":
        d_post = rng.uniform(0.0, 1.0, G)
    elif dose_dist == "uniform_d_lower_5":
        d_post = rng.uniform(d_lower, 5.0, G)
    elif dose_dist == "mass_point_d_lower_uniform":
        # 30% at d_lower, 70% Uniform(d_lower, d_lower + 4)
        n_mass = int(0.30 * G)
        n_cont = G - n_mass
        d_post = np.concatenate(
            [
                np.full(n_mass, d_lower),
                rng.uniform(d_lower, d_lower + 4.0, n_cont),
            ]
        )
        rng.shuffle(d_post)
    else:  # pragma: no cover - test scaffolding
        raise ValueError(f"unknown dose_dist={dose_dist!r}")

    delta_y = boundary_intercept + was_true * d_post + sigma * rng.standard_normal(G)
    y_pre = np.zeros(G)
    y_post = y_pre + delta_y

    units = np.repeat(np.arange(G), 2)
    periods = np.tile([1, 2], G)
    dose = np.column_stack([np.zeros(G), d_post]).ravel()
    outcome = np.column_stack([y_pre, y_post]).ravel()

    return pd.DataFrame(
        {
            "unit": units,
            "period": periods,
            "dose": dose,
            "outcome": outcome,
        }
    )


def _fit_overall(panel: pd.DataFrame, **kwargs) -> HeterogeneousAdoptionDiDResults:
    """Fit HAD with `aggregate="overall"` and return the result."""
    est = HeterogeneousAdoptionDiD(**kwargs)
    with warnings.catch_warnings():
        # The Design 1 family (mass_point / continuous_near_d_lower) emits
        # a UserWarning about Assumption 5/6 non-testability; filter so
        # test output isn't dominated by warning noise. The warning IS
        # locked elsewhere by
        # ``TestHADDeviations::test_assumption_5_6_userwarning_fires_on_design_1_family``,
        # which uses ``pytest.warns(UserWarning, match=r"Assumption [56]")``
        # on a mass-point fit to assert the warning fires (so this helper
        # suppression doesn't mask a regression).
        warnings.filterwarnings(
            "ignore",
            message=r".*Assumption [56].*",
            category=UserWarning,
        )
        result = est.fit(
            panel,
            outcome_col="outcome",
            dose_col="dose",
            time_col="period",
            unit_col="unit",
        )
    assert isinstance(result, HeterogeneousAdoptionDiDResults)
    return result


# =============================================================================
# TestHADTheorem1Design1Prime — Eq. 3 + Theorem 1
# =============================================================================


class TestHADTheorem1Design1Prime:
    """Eq. 3 + Theorem 1: Design 1' identification of WAS.

    Paper Section 3.1.2 / Theorem 1 (boundary-subtracted form):

        WAS = ( E[delta_Y] - lim_{d down 0} E[delta_Y | D_2 <= d] ) / E[D_2]

    The library implements this via :func:`bias_corrected_local_linear`
    (Phase 1c) composed into ``HeterogeneousAdoptionDiD._fit_continuous``
    on the ``continuous_at_zero`` design path:

        att = ( mean(delta_Y) - tau_bc ) / mean(D)

    where ``tau_bc`` is the bias-corrected local-linear estimate of the
    boundary intercept ``lim_{d down 0} E[delta_Y | D_2 <= d]``.

    This class exercises BOTH the simple case (boundary intercept ~ 0,
    where ``tau_bc`` is a small noise term) AND a NONZERO-boundary-
    intercept case (``delta_Y = c + beta*D + eps`` with ``c != 0``),
    so the ``mean(delta_Y) - tau_bc`` subtraction logic is verified
    rather than just the ``tau_bc ~ 0`` special case.
    """

    def test_eq3_was_recovery_uniform_dose(self) -> None:
        """Eq. 3: WAS recovered on Uniform(0,1) DGP within MC error.

        DGP: D ~ Uniform(0, 1), delta_y = 0.3 * D + N(0, 0.1).
        Population WAS = 0.3. Boundary intercept ~ 0 so the
        ``mean(delta_Y) - tau_bc`` subtraction reduces to
        ``mean(delta_Y)``; see ``test_eq3_was_recovery_nonzero_boundary``
        below for the nonzero-boundary case that explicitly exercises
        the subtraction term.
        """
        rng = np.random.default_rng(_BASE_SEED_THEOREM1 + 0)
        panel = _make_two_period_panel(
            rng, G=2000, dose_dist="uniform_0_1", was_true=0.3, sigma=0.1
        )
        result = _fit_overall(panel, design="auto")
        assert result.design == "continuous_at_zero"
        # Population WAS = 0.3. MC band ~ +/- 3 * se covers truth.
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert abs(result.att - 0.3) < 3.0 * result.se

    def test_eq3_was_recovery_nonzero_boundary_intercept(self) -> None:
        """Eq. 3: WAS recovered when boundary intercept c != 0.

        DGP: delta_y = 0.2 + 0.3 * D + N(0, 0.1). The boundary intercept
        is ``c = 0.2`` (constant additive component to delta_Y),
        so the library's

            tau_bc -> 0.2 (estimating ``lim_{d down 0} E[delta_Y | D <= d]``)
            mean(delta_Y) -> 0.2 + 0.3 * 0.5 = 0.35
            att = (0.35 - 0.2) / 0.5 = 0.30 = WAS_true

        verifies the ``mean(delta_Y) - tau_bc`` subtraction explicitly.
        Were the library to compute ``mean(delta_Y) / mean(D)`` without
        the boundary subtraction, the recovered att would be 0.70 (= 0.35
        / 0.5), so a non-trivial ``c != 0`` immediately distinguishes
        the two formulas.
        """
        rng = np.random.default_rng(_BASE_SEED_THEOREM1 + 10)
        panel = _make_two_period_panel(
            rng,
            G=2000,
            dose_dist="uniform_0_1",
            was_true=0.3,
            sigma=0.1,
            boundary_intercept=0.2,
        )
        result = _fit_overall(panel, design="auto")
        assert result.design == "continuous_at_zero"
        # Population WAS = 0.3; boundary intercept c = 0.2 must be
        # subtracted via tau_bc. MC band ~ +/- 3 * se covers truth.
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert abs(result.att - 0.3) < 3.0 * result.se
        # Guard against the regression-to-no-subtraction failure mode:
        # the wrong formula ``mean(delta_Y) / mean(D)`` would give
        # att ~ 0.7, far outside the 3-sigma band.
        assert abs(result.att - 0.7) > 5.0 * result.se

    def test_design_autodetect_lands_on_continuous_at_zero(self) -> None:
        """Design auto-detect picks continuous_at_zero when d.min() ~ 0."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM1 + 1)
        panel = _make_two_period_panel(rng, G=500, dose_dist="uniform_0_1", was_true=0.5, sigma=0.1)
        result = _fit_overall(panel, design="auto")
        assert result.design == "continuous_at_zero"
        assert result.d_lower == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.slow
    def test_eq3_normal_pivot_coverage(self, ci_params) -> None:
        """Eq. 8 + Theorem 1: bias-corrected CI 95% coverage at G=1000.

        Run n_replicates fits on the Design 1' DGP (gated by
        ``ci_params.bootstrap(200, min_n=25)`` so constrained CI can
        downshift the replication count while preserving the
        code-path coverage), collect (att_hat - WAS_true) / se_hat,
        assert empirical 95% coverage of WAS_true exceeds 0.85
        (matching paper Table 1's documented under-coverage band at
        G=100-500).
        """
        was_true = 0.3
        n_reps = ci_params.bootstrap(200, min_n=25)
        ats = []
        ses = []
        for idx in range(n_reps):
            rng = np.random.default_rng(_BASE_SEED_THEOREM1 + 100 + idx)
            panel = _make_two_period_panel(
                rng, G=1000, dose_dist="uniform_0_1", was_true=was_true, sigma=0.1
            )
            result = _fit_overall(panel, design="auto")
            ats.append(result.att)
            ses.append(result.se)
        ats = np.asarray(ats)
        ses = np.asarray(ses)
        valid = np.isfinite(ats) & np.isfinite(ses) & (ses > 0)
        assert valid.sum() >= 0.95 * n_reps  # at least 95% of fits valid
        z = (ats[valid] - was_true) / ses[valid]
        # CCT bias-corrected CI is normal-pivot at z_{1-alpha/2} = 1.96.
        coverage = float(np.mean(np.abs(z) <= 1.96))
        # Paper Table 1 documents under-coverage at small G (89% at
        # G=100, 95% at G=2500); at G=1000 we expect ~0.90-0.95.
        # MC standard error on coverage is sqrt(0.95*0.05/n_reps), so
        # the floor must absorb a few standard errors of slack at
        # reduced n. Full n=200: 0.85; reduced n=25: 0.65 (~6 SE below
        # 0.95).
        coverage_floor = 0.85 if n_reps >= 100 else 0.65
        assert coverage >= coverage_floor, (
            f"empirical coverage {coverage:.3f} below {coverage_floor} " f"(n_reps={n_reps})"
        )

    def test_zero_dose_units_dont_break_fit(self) -> None:
        """A continuous-at-zero panel with mass at exactly d=0 still fits."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM1 + 2)
        panel = _make_two_period_panel(
            rng, G=1000, dose_dist="uniform_0_1", was_true=0.4, sigma=0.1
        )
        # Force some exact zeros — common in real treatment-rollout data.
        zero_mask = (panel["period"] == 2) & (panel.index % 17 == 0)
        panel.loc[zero_mask, "dose"] = 0.0
        result = _fit_overall(panel, design="auto")
        assert result.design == "continuous_at_zero"
        assert np.isfinite(result.att)

    def test_constant_y_panel_returns_nan_inference(self) -> None:
        """Constant outcome -> safe_inference joint NaN contract.

        With sigma=0 + was_true=0, delta_Y is identically zero. The
        bias-corrected local-linear cannot estimate a slope (zero
        variance in the response) and returns NaN for both att and se.
        safe_inference then NaNs out (t_stat, p_value, conf_int) under
        the joint NaN convention.
        """
        rng = np.random.default_rng(_BASE_SEED_THEOREM1 + 3)
        panel = _make_two_period_panel(rng, G=500, dose_dist="uniform_0_1", was_true=0.0, sigma=0.0)
        result = _fit_overall(panel, design="auto")
        # Joint NaN invariant on degenerate panel: all inference fields
        # go NaN together (no partial-NaN leakage).
        assert np.isnan(result.att)
        assert np.isnan(result.se)
        assert np.isnan(result.t_stat)
        assert np.isnan(result.p_value)
        assert np.isnan(result.conf_int[0]) and np.isnan(result.conf_int[1])

    def test_d_lower_attribute_pinned_to_zero(self) -> None:
        """``result.d_lower`` is 0.0 (machine precision) on Design 1'."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM1 + 4)
        panel = _make_two_period_panel(rng, G=500, dose_dist="uniform_0_1", was_true=0.2, sigma=0.1)
        result = _fit_overall(panel, design="auto")
        assert result.d_lower == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# TestHADTheorem3MassPoint — Eq. 11 + Theorem 3
# =============================================================================


class TestHADTheorem3MassPoint:
    """Eq. 11 + Theorem 3: WAS_{d_lower} under Assumption 6, mass-point path.

    Paper Section 3.2.4: when ``d_lower > 0`` and ``D_2`` has a mass
    point at ``d_lower``, ``WAS_{d_lower}`` is identified via the 2SLS
    sample-average estimator with instrument ``1{D_2 > d_lower}``:

        WAS_{d_lower} = ( E[delta_Y | D_2 > d_lower] - E[delta_Y | D_2 = d_lower] )
                      / ( E[D_2 | D_2 > d_lower] - d_lower )

    The library implements this in ``_fit_mass_point_2sls``. This class
    exercises mass-point auto-detect + the closed-form 2SLS algebra.
    """

    def test_eq11_was_d_lower_recovery_30pct_mass(self) -> None:
        """Eq. 11: WAS_{d_lower} recovered on 30% mass-at-1.0 DGP.

        DGP: 30% at d_lower=1.0, 70% Uniform(1.0, 5.0). Linear
        delta_y = 0.4 * D + N(0, 0.1). Under linearity, WAS_{d_lower} = 0.4.
        """
        rng = np.random.default_rng(_BASE_SEED_THEOREM3 + 0)
        panel = _make_two_period_panel(
            rng,
            G=2000,
            dose_dist="mass_point_d_lower_uniform",
            was_true=0.4,
            sigma=0.1,
            d_lower=1.0,
        )
        result = _fit_overall(panel, design="auto")
        assert result.design == "mass_point"
        assert result.d_lower == pytest.approx(1.0, abs=1e-9)
        # Population WAS_{d_lower} = 0.4 under linear DGP.
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert abs(result.att - 0.4) < 3.0 * result.se

    def test_mass_point_design_autodetect(self) -> None:
        """Auto-detect picks mass_point when modal-fraction at d.min() > 2%."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM3 + 1)
        panel = _make_two_period_panel(
            rng,
            G=500,
            dose_dist="mass_point_d_lower_uniform",
            was_true=0.3,
            sigma=0.05,
            d_lower=2.0,
        )
        result = _fit_overall(panel, design="auto")
        assert result.design == "mass_point"

    def test_explicit_mass_point_on_continuous_sample_rejects(self) -> None:
        """Explicit design='mass_point' on a continuous sample raises."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM3 + 2)
        panel = _make_two_period_panel(rng, G=300, dose_dist="uniform_0_1", was_true=0.3, sigma=0.1)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.05)
        with pytest.raises(ValueError, match=r"(mass[_-]point|d_lower|modal)"):
            est.fit(
                panel,
                outcome_col="outcome",
                dose_col="dose",
                time_col="period",
                unit_col="unit",
            )

    def test_mass_point_n_at_d_lower_and_above_populated(self) -> None:
        """``n_mass_point`` and ``n_above_d_lower`` fields are populated."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM3 + 3)
        panel = _make_two_period_panel(
            rng,
            G=1000,
            dose_dist="mass_point_d_lower_uniform",
            was_true=0.3,
            sigma=0.1,
            d_lower=1.0,
        )
        result = _fit_overall(panel, design="auto")
        # 30% at d_lower => ~300; 70% above => ~700.
        assert result.n_mass_point is not None
        assert result.n_above_d_lower is not None
        assert result.n_mass_point + result.n_above_d_lower == 1000
        # Bandwidth diagnostics absent on mass-point path.
        assert result.bandwidth_diagnostics is None
        assert result.bias_corrected_fit is None

    def test_mass_point_wald_iv_equivalence(self) -> None:
        """Mass-point WAS matches the closed-form Wald-IV gap.

        WAS_{d_lower} = ( mean(delta_Y | D > d_lower) - mean(delta_Y | D = d_lower) )
                      / ( mean(D | D > d_lower) - d_lower )
        """
        rng = np.random.default_rng(_BASE_SEED_THEOREM3 + 4)
        panel = _make_two_period_panel(
            rng,
            G=1500,
            dose_dist="mass_point_d_lower_uniform",
            was_true=0.4,
            sigma=0.05,
            d_lower=1.0,
        )
        result = _fit_overall(panel, design="auto")
        # Recompute the closed-form Wald-IV from the panel.
        post = panel[panel["period"] == 2].copy()
        post["delta_y"] = post["outcome"].values  # pre-period y == 0 by construction
        at_d_lower = np.abs(post["dose"].values - 1.0) < 1e-9
        above = post["dose"].values > 1.0 + 1e-9
        wald = (
            float(post.loc[above, "delta_y"].mean()) - float(post.loc[at_d_lower, "delta_y"].mean())
        ) / (float(post.loc[above, "dose"].mean()) - 1.0)
        # Wald-IV closed form should match the 2SLS estimator at machine
        # precision (both are the same algebra on the same data).
        assert result.att == pytest.approx(wald, abs=1e-9)


# =============================================================================
# TestHADTheorem4QUG — Theorem 4 (QUG null test, Exp(1)/Exp(1) limit law)
# =============================================================================


class TestHADTheorem4QUG:
    """Theorem 4 (QUG): the order-statistic ratio test for ``d_lower = 0``.

    Paper Theorem 4: under ``H_0: d_lower = 0`` (and regularity), the
    statistic ``T = D_{(1)} / ( D_{(2)} - D_{(1)} )`` converges in
    distribution to ``T_lambda = (lambda + E_1) / E_2`` with
    ``E_i ~ Exp(1)`` iid; at ``lambda = 0`` the CDF is

        F(t) = t / (1 + t)

    so the asymptotic p-value is ``1 / (1 + T)``. The library implements
    this in ``qug_test``. This class exercises the limit-law
    distributional match + the closed-form p-value at machine precision
    + the tie-break and zero-dose conventions.
    """

    @pytest.mark.slow
    def test_theorem4_limit_law_distributional_match(self, ci_params) -> None:
        """Empirical CDF of T converges to F(t) = t/(1+t) at G=2000.

        Monte Carlo (gated by ``ci_params.bootstrap(5000, min_n=200)``):
        draw T from a Uniform(0,1) dose DGP (under H_0: d_lower = 0)
        and compare empirical CDF to ``F(t) = t / (1 + t)`` via
        Kolmogorov-Smirnov.

        Tolerance: KS-stat <= 0.05. Rationale: KS critical at n=5000,
        alpha=0.05 is ~1.36/sqrt(5000) = 0.0192; 0.05 provides ~2.6x
        margin to absorb heavy upper-tail truncation under
        T_lambda = (E_1) / E_2 (Cauchy-like tails — needs more samples
        for empirical-CDF stability in the upper percentiles).
        Reduced replication count under ``ci_params.bootstrap`` still
        exercises the same code path; pure-Python CI runs at n=200 and
        full runs at 5000.
        """
        n_draws = ci_params.bootstrap(5000, min_n=200)
        G_per_draw = 2000
        t_stats = np.empty(n_draws)
        for idx in range(n_draws):
            rng = np.random.default_rng(_BASE_SEED_THEOREM4 + idx)
            d = rng.uniform(0.0, 1.0, G_per_draw)
            res = qug_test(d, alpha=0.05)
            t_stats[idx] = res.t_stat
        valid = np.isfinite(t_stats)
        assert valid.sum() >= 0.99 * n_draws
        # Compare to closed-form F(t) = t/(1+t).
        ks_stat, _ = stats.kstest(t_stats[valid], lambda t: t / (1.0 + t))
        # KS critical at n=5000, alpha=0.05 is ~0.0192; at n=200 it's
        # ~0.096. Conditional tolerance per `ci_params.bootstrap` /
        # `feedback_bootstrap_drift_tests_need_backend_tolerance`: 0.05
        # at full n, 0.15 at reduced n.
        ks_tol = 0.05 if n_draws >= 1000 else 0.15
        assert (
            ks_stat <= ks_tol
        ), f"KS stat {ks_stat:.4f} exceeds {ks_tol} tolerance (n_draws={n_draws})"

    def test_theorem4_p_value_closed_form_precision(self) -> None:
        """Asymptotic p-value ``1/(1+T)`` at machine precision."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM4 + 99)
        d = rng.uniform(0.1, 1.0, 500)  # all positive — no zero-dose drop
        res = qug_test(d, alpha=0.05)
        assert np.isfinite(res.t_stat)
        assert np.isfinite(res.p_value)
        expected_p = 1.0 / (1.0 + res.t_stat)
        assert res.p_value == pytest.approx(expected_p, abs=1e-12)

    def test_tie_break_returns_all_nan_inference(self) -> None:
        """``D_{(1)} == D_{(2)}`` returns all-NaN with UserWarning, not raise."""
        d = np.array([0.5, 0.5, 1.0, 1.5, 2.0])  # tied minimum
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=UserWarning)
            res = qug_test(d, alpha=0.05)
        assert np.isnan(res.t_stat)
        assert np.isnan(res.p_value)
        assert res.reject is False
        # At least one UserWarning fired (tie-break or similar).
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_zero_dose_observations_filtered_with_warning(self) -> None:
        """Zero-dose units are dropped from QUG with a UserWarning."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM4 + 7)
        d_positive = rng.uniform(0.1, 1.0, 500)
        d_with_zeros = np.concatenate([d_positive, np.zeros(20)])
        rng.shuffle(d_with_zeros)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=UserWarning)
            res = qug_test(d_with_zeros, alpha=0.05)
        # Result is still computed on the positive subset.
        assert np.isfinite(res.t_stat)
        # Zero-dose-drop UserWarning fired.
        assert any(
            issubclass(w.category, UserWarning) and ("zero" in str(w.message).lower())
            for w in caught
        )

    def test_rejection_region_threshold_T_gt_alpha_inv_minus_one(self) -> None:
        """Rejection rule: ``T > 1/alpha - 1`` is the boundary of reject region."""
        # Construct d so T sits just above the alpha=0.05 threshold (= 19).
        # T = d[0] / (d[1] - d[0]); choose d[0] = 19, d[1] = 20.
        d = np.array([19.0, 20.0, 25.0, 30.0, 40.0])
        res = qug_test(d, alpha=0.05)
        assert res.t_stat == pytest.approx(19.0, abs=1e-12)
        # 1/alpha - 1 = 19.0; T = 19.0 is NOT strictly above -> no reject.
        assert res.reject is False
        # Push T above 19.0.
        d2 = np.array([19.01, 20.0, 25.0, 30.0, 40.0])
        res2 = qug_test(d2, alpha=0.05)
        assert res2.t_stat > 19.0
        assert res2.reject is True

    def test_finite_sample_under_alternative_rejects_at_d_lower_positive(self) -> None:
        """When d_lower > 0 (alternative true), QUG rejects with high power."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM4 + 50)
        d = rng.uniform(2.0, 5.0, 1000)  # d_lower ~ 2, far from 0
        res = qug_test(d, alpha=0.05)
        # T = D_(1) / (D_(2) - D_(1)) is very large when d_lower >> spacing.
        # Should reject H_0: d_lower = 0 with high probability.
        assert res.t_stat > 19.0  # well above 1/0.05 - 1 = 19
        assert res.reject is True


# =============================================================================
# TestHADTheorem7YatchewHR — Eq. 29 + Theorem 7
# =============================================================================


class TestHADTheorem7YatchewHR:
    """Eq. 29 + Theorem 7: heteroskedasticity-robust Yatchew linearity test.

    Paper Eq. 29 / Theorem 7:

        T_hr = sqrt(G) * (sigma2_lin - sigma2_diff) / sigma2_W

    where

        sigma2_lin  = (1/G) * sum(eps^2)              # OLS residuals under H0
        sigma2_diff = (1/(2G)) * sum((dy_{(g)} - dy_{(g-1)})^2)
        sigma2_W    = sqrt((1/(G-1)) * sum(eps_{(g)}^2 * eps_{(g-1)}^2))

    Under H0 (linearity), ``T_hr`` converges in distribution to
    ``N(0, 1)``. Note paper-literal normalization is ``1/(2G)`` for
    sigma2_diff (NOT finite-sample ``1/(2(G-1))``); the library pins
    the paper-literal form, and this class locks that convention.
    """

    @pytest.mark.slow
    def test_eq29_standard_normal_limit_under_linearity(self, ci_params) -> None:
        """T_hr converges to N(0,1) under H_0 (linearity) at G=2000.

        DGP: dy = a + b * d + N(0, sigma). Run n_reps draws (gated by
        ``ci_params.bootstrap(200, min_n=25)`` so constrained CI can
        downshift), assert empirical KS-stat vs N(0,1) below an n-
        dependent tolerance. KS critical at n=200, alpha=0.05 is
        ~1.36/sqrt(200) = 0.096; at n=25 it's ~0.272. Conditional
        tolerance: 0.15 at full n, 0.35 at reduced n.
        """
        n_reps = ci_params.bootstrap(200, min_n=25)
        G = 2000
        t_stats = np.empty(n_reps)
        for idx in range(n_reps):
            rng = np.random.default_rng(_BASE_SEED_THEOREM7 + idx)
            d = rng.uniform(0.0, 1.0, G)
            dy = 0.3 * d + 0.1 * rng.standard_normal(G)
            res = yatchew_hr_test(d, dy, alpha=0.05, null="linearity")
            t_stats[idx] = res.t_stat_hr
        # All draws should be finite (no ties on Uniform).
        assert np.all(np.isfinite(t_stats))
        ks_stat, _ = stats.kstest(t_stats, "norm")
        # KS critical at n=200, alpha=0.05 is ~0.096; at n=25 it's ~0.272.
        # Full-n run: 0.10 (slim margin, validated locally on the pinned
        # seed sequence); reduced-n CI: 0.35 (safety band over the
        # asymptotic critical at min_n).
        ks_tol = 0.10 if n_reps >= 100 else 0.35
        assert (
            ks_stat <= ks_tol
        ), f"KS stat {ks_stat:.4f} exceeds {ks_tol} tolerance (n_reps={n_reps})"

    def test_eq29_normalizer_2G_not_2Gminus1(self) -> None:
        """Locks the paper-literal sigma2_diff normalizer = 2G (NOT 2(G-1)).

        Hand-computed on a small panel:
          d  = [0.1, 0.2, 0.3, 0.4]
          dy = [1.0, 1.5, 2.0, 2.7]   (sorted by d; close to linear)

        sigma2_diff = (1/(2G)) * sum((dy_{(g)} - dy_{(g-1)})^2)
                    = (1/8) * ( (1.5-1.0)^2 + (2.0-1.5)^2 + (2.7-2.0)^2 )
                    = (1/8) * (0.25 + 0.25 + 0.49)
                    = 0.99 / 8 = 0.12375
        """
        d = np.array([0.1, 0.2, 0.3, 0.4])
        dy = np.array([1.0, 1.5, 2.0, 2.7])
        res = yatchew_hr_test(d, dy, alpha=0.05, null="linearity")
        # 2G normalization
        expected_sigma2_diff = 0.99 / 8.0  # 2*G = 8
        # finite-sample alternative would be 0.99 / 6 (= 2*(G-1)) = 0.165
        wrong_normalizer = 0.99 / 6.0
        assert res.sigma2_diff == pytest.approx(expected_sigma2_diff, abs=1e-12)
        # Confirm we are NOT computing the wrong (finite-sample) normalizer.
        assert abs(res.sigma2_diff - wrong_normalizer) > 1e-4

    def test_eq29_one_sided_critical_value_phi_inv(self) -> None:
        """Reject rule uses one-sided z_{1-alpha} = Phi^{-1}(1-alpha)."""
        rng = np.random.default_rng(_BASE_SEED_THEOREM7 + 999)
        G = 1500
        d = rng.uniform(0.0, 1.0, G)
        # Strongly nonlinear DGP: dy = sin(5*d) -> Yatchew should reject.
        dy = np.sin(5.0 * d) + 0.01 * rng.standard_normal(G)
        res = yatchew_hr_test(d, dy, alpha=0.05, null="linearity")
        assert res.t_stat_hr > stats.norm.ppf(0.95)  # > z_{0.95} ~ 1.645
        assert res.reject is True

    def test_constant_dy_short_circuits_to_p1_no_reject(self) -> None:
        """Exact-linear short-circuit: residuals ~ 0 -> p=1.0, reject=False."""
        # dy is exactly linear in d -> OLS residuals are at IEEE precision.
        d = np.linspace(0.0, 1.0, 200)
        dy = 0.5 + 0.7 * d  # exactly linear
        res = yatchew_hr_test(d, dy, alpha=0.05, null="linearity")
        assert res.p_value == pytest.approx(1.0, abs=0.0)
        assert res.reject is False

    def test_tied_dose_returns_nan_with_warning(self) -> None:
        """Tied doses -> Yatchew returns NaN with UserWarning (not raise)."""
        d = np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.5])
        dy = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=UserWarning)
            res = yatchew_hr_test(d, dy, alpha=0.05, null="linearity")
        assert np.isnan(res.t_stat_hr)
        # Tied-dose UserWarning fired.
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_mean_independence_mode_matches_R_order0(self) -> None:
        """``null="mean_independence"`` uses dy - mean(dy) residuals.

        Sanity: under truly mean-independent DGP (dy ~ N(0, 1), d
        independent), T_hr should NOT reject at alpha=0.05 most of the
        time.
        """
        rng = np.random.default_rng(_BASE_SEED_THEOREM7 + 1234)
        d = rng.uniform(0.0, 1.0, 1000)
        dy = rng.standard_normal(1000)  # mean-independent of d
        res = yatchew_hr_test(d, dy, alpha=0.05, null="mean_independence")
        assert np.isfinite(res.t_stat_hr)
        # Under H0, ~5% rejection rate. Single draw should usually
        # fail-to-reject; pinned seed makes this deterministic.
        assert res.reject is False


# =============================================================================
# TestHADJointStute — Eq. 18 (mean-independence variant) joint pre-trends + homogeneity
# =============================================================================


class TestHADJointStute:
    """Section 4.2 step 2 + Section 4.3: joint Stute tests for pre-trends
    and homogeneity.

    Paper Eq. 18 specifies a sum-of-CvMs joint statistic across multiple
    pre-period placebo horizons with a shared-eta Mammen wild bootstrap.
    The library ships the mean-independence variant in
    ``joint_pretrends_test`` (residuals from OLS Y_t - Y_base ~ 1) and
    the linearity (homogeneity) variant in ``joint_homogeneity_test``
    (residuals from OLS Y_t - Y_base ~ 1 + D).

    **Coverage scope of this class:** H0 fail-to-reject is exercised
    for both ``joint_pretrends_test`` (mean-independence null) and
    ``joint_homogeneity_test`` (linearity null) on a linear-DGP panel
    where D is independent of pre-Y; H1 rejection is demonstrated on
    ``joint_homogeneity_test`` only, via a nonlinear (D + D^2) post-
    period DGP. An H1 violating-pretrends test for
    ``joint_pretrends_test`` is not added here (a synthetic
    correlated-D-vs-pre-Y DGP would re-verify the bootstrap
    calibration covered by ``test_had_pretests.py``).

    The ``trends_lin=True`` Eq. 17 / Eq. 18 linear-trend-detrended
    variant is SHIPPED in the library and R-parity-locked against
    ``DIDHAD::did_had(..., trends_lin=TRUE)`` in
    ``tests/test_did_had_parity.py`` (3 DGPs x 5 method combos at
    ``atol=1e-8``). It is OUT OF SCOPE for this methodology file.
    """

    def _build_multi_period_panel(
        self,
        rng: np.random.Generator,
        *,
        G: int,
        pre_periods: list,
        base_period: int,
        post_periods: list,
        was_true: float,
        nonlinear_post: bool = False,
    ) -> pd.DataFrame:
        """Build a multi-period HAD panel with the given pre/base/post layout.

        Pre-periods: D = 0 for all units.
        Base period: D = 0 for all units (the F-1 anchor; pre-treatment).
        Post-periods: D = D_post (drawn once per unit, time-constant).

        Outcome model: Y_{g,t} = Y_{g, base} + (t > base) * (was_true * D
        + N(0, 0.1)). If ``nonlinear_post`` is True, replace with
        was_true * D + was_true * D**2 (so the effect is nonlinear in D).
        """
        d_post = rng.uniform(0.0, 1.0, G)
        # Time-constant base level per unit.
        y_base = 0.1 * rng.standard_normal(G)
        rows = []
        all_periods = pre_periods + [base_period] + post_periods
        for t in all_periods:
            for g in range(G):
                if t > base_period:
                    if nonlinear_post:
                        delta = was_true * d_post[g] + was_true * d_post[g] ** 2
                    else:
                        delta = was_true * d_post[g]
                    eps_t = 0.1 * rng.standard_normal()
                    outcome = y_base[g] + delta + eps_t
                    dose = d_post[g]
                else:
                    outcome = y_base[g] + 0.05 * rng.standard_normal()
                    dose = 0.0
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": dose,
                        "outcome": outcome,
                    }
                )
        return pd.DataFrame(rows)

    def test_joint_pretrends_fails_to_reject_under_h0(self) -> None:
        """Joint pre-trends test fails-to-reject when D is independent of pre-Y."""
        rng = np.random.default_rng(_BASE_SEED_JOINT_STUTE + 0)
        panel = self._build_multi_period_panel(
            rng,
            G=300,
            pre_periods=[1, 2],
            base_period=3,
            post_periods=[4, 5],
            was_true=0.3,
        )
        res = joint_pretrends_test(
            data=panel,
            outcome_col="outcome",
            dose_col="dose",
            time_col="period",
            unit_col="unit",
            pre_periods=[1, 2],
            base_period=3,
            n_bootstrap=199,
            seed=_BASE_SEED_JOINT_STUTE + 100,
        )
        # D is iid of Y_pre under the DGP -> fail-to-reject expected.
        assert np.isfinite(res.cvm_stat_joint)
        assert np.isfinite(res.p_value)
        assert res.reject is False

    def test_joint_homogeneity_fails_to_reject_under_linear_dgp(self) -> None:
        """Joint homogeneity (linearity) test fails-to-reject on linear DGP."""
        rng = np.random.default_rng(_BASE_SEED_JOINT_STUTE + 1)
        panel = self._build_multi_period_panel(
            rng,
            G=300,
            pre_periods=[1, 2],
            base_period=3,
            post_periods=[4, 5],
            was_true=0.3,
            nonlinear_post=False,
        )
        res = joint_homogeneity_test(
            data=panel,
            outcome_col="outcome",
            dose_col="dose",
            time_col="period",
            unit_col="unit",
            post_periods=[4, 5],
            base_period=3,
            n_bootstrap=199,
            seed=_BASE_SEED_JOINT_STUTE + 101,
        )
        assert np.isfinite(res.cvm_stat_joint)
        assert np.isfinite(res.p_value)
        assert res.reject is False

    def test_joint_homogeneity_rejects_under_nonlinear_dgp(self) -> None:
        """Joint homogeneity test rejects when delta_y is nonlinear in D."""
        rng = np.random.default_rng(_BASE_SEED_JOINT_STUTE + 2)
        panel = self._build_multi_period_panel(
            rng,
            G=500,
            pre_periods=[1, 2],
            base_period=3,
            post_periods=[4, 5],
            was_true=1.0,  # large nonlinearity (D + D^2)
            nonlinear_post=True,
        )
        res = joint_homogeneity_test(
            data=panel,
            outcome_col="outcome",
            dose_col="dose",
            time_col="period",
            unit_col="unit",
            post_periods=[4, 5],
            base_period=3,
            n_bootstrap=199,
            seed=_BASE_SEED_JOINT_STUTE + 102,
        )
        # Strong nonlinearity at G=500 with low noise -> should reject.
        assert np.isfinite(res.cvm_stat_joint)
        assert res.reject is True

    def test_n_bootstrap_lower_bound_validates(self) -> None:
        """``n_bootstrap < 99`` raises ValueError (bootstrap stability gate)."""
        rng = np.random.default_rng(_BASE_SEED_JOINT_STUTE + 3)
        panel = self._build_multi_period_panel(
            rng,
            G=100,
            pre_periods=[1, 2],
            base_period=3,
            post_periods=[4],
            was_true=0.3,
        )
        with pytest.raises(ValueError, match=r"n_bootstrap.*99"):
            joint_pretrends_test(
                data=panel,
                outcome_col="outcome",
                dose_col="dose",
                time_col="period",
                unit_col="unit",
                pre_periods=[1, 2],
                base_period=3,
                n_bootstrap=49,
                seed=42,
            )

    def test_per_horizon_stats_dict_populated(self) -> None:
        """``per_horizon_stats`` records the per-horizon CvM for diagnostics."""
        rng = np.random.default_rng(_BASE_SEED_JOINT_STUTE + 4)
        panel = self._build_multi_period_panel(
            rng,
            G=200,
            pre_periods=[1, 2],
            base_period=3,
            post_periods=[4, 5],
            was_true=0.3,
        )
        res = joint_pretrends_test(
            data=panel,
            outcome_col="outcome",
            dose_col="dose",
            time_col="period",
            unit_col="unit",
            pre_periods=[1, 2],
            base_period=3,
            n_bootstrap=199,
            seed=_BASE_SEED_JOINT_STUTE + 104,
        )
        # Per-horizon stats keyed by horizon label.
        assert isinstance(res.per_horizon_stats, dict)
        assert len(res.per_horizon_stats) == 2  # two pre-periods
        for v in res.per_horizon_stats.values():
            assert np.isfinite(v)


# =============================================================================
# TestHADDeviations — locks library deviations + safe_inference invariant
# =============================================================================


class TestHADDeviations:
    """Locks library deviations from paper and from naive defaults.

    Five deviation surfaces:
    1. Equal-vs-cell-size weighting on the continuous path (locked
       in REGISTRY Deviations Note #1).
    2. Sup-t bootstrap gating: runs only when event-study + weighted +
       cband=True (locked in REGISTRY Deviations Note #2).
    3. Staggered-timing fail-closed ValueError (locked in REGISTRY
       Deviations Library extension #5).
    4. ``first_treat_col`` last-cohort auto-filter (HAD's Appendix B.2
       prescription).
    5. ``safe_inference`` joint NaN invariant on degenerate inputs.
    """

    def test_equal_weighting_is_per_row_not_per_dose_cell(self) -> None:
        """Per-row equal weighting: selective region replication shifts att.

        The library uses per-row equal weighting (`w_g = 1`) on the
        continuous path. A cell-size-weighting counterfactual would
        rescale per-observation weights by inverse cell density, so
        replicating a dose region would shrink each per-row weight and
        leave the att invariant.

        Under per-row equal weighting on a NONLINEAR DGP, replicating
        one dose region shifts the empirical distribution and the att
        moves with it. This test probes the deviation directly:

        DGP: ΔY = 0.5 * D + 1.0 * D². Population WAS depends on
        ``E[D²] / E[D]``; replicating low-D units shrinks this ratio,
        so att shifts downward.

        Under cell-size weighting (counterfactual): both panels would
        give approximately the same att because the per-cell aggregate
        weight is preserved across the replication.
        """
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 0)
        G = 1500
        d_post = rng.uniform(0.0, 1.0, G)
        # Nonlinear DGP: linear-plus-quadratic.
        delta_y = 0.5 * d_post + 1.0 * d_post**2 + 0.05 * rng.standard_normal(G)
        units = np.repeat(np.arange(G), 2)
        periods = np.tile([1, 2], G)
        dose = np.column_stack([np.zeros(G), d_post]).ravel()
        outcome = np.column_stack([np.zeros(G), delta_y]).ravel()
        panel_a = pd.DataFrame({"unit": units, "period": periods, "dose": dose, "outcome": outcome})
        result_a = _fit_overall(panel_a, design="auto")

        # Build B by selectively replicating ONLY the low-D units
        # (D <= 0.15) 4x extra. This shifts the empirical distribution
        # toward the boundary, reducing E[D²]/E[D].
        post_a = panel_a[panel_a["period"] == 2]
        low_d_units = post_a.loc[post_a["dose"] <= 0.15, "unit"].values
        n_reps = 4
        extra_panels = []
        max_unit = int(panel_a["unit"].max())
        for r in range(1, n_reps + 1):
            extra = panel_a[panel_a["unit"].isin(low_d_units)].copy()
            extra["unit"] = extra["unit"] + max_unit * r + r
            extra_panels.append(extra)
        panel_b = pd.concat([panel_a] + extra_panels, ignore_index=True)
        result_b = _fit_overall(panel_b, design="auto")

        # Verify the shift: on a nonlinear DGP with per-row equal
        # weighting, panel B's att should differ from panel A's by
        # MORE than MC noise. Bound the expected shift size from below
        # by ~1.5 * max(se) — large enough to reject the no-shift null
        # (cell-size-weighting counterfactual) but small enough to
        # tolerate stochastic variation in the boundary intercept.
        shift = abs(result_b.att - result_a.att)
        max_se = max(result_a.se, result_b.se)
        assert shift > 1.5 * max_se, (
            f"selective low-D replication did not shift att enough "
            f"(shift={shift:.4f}, max_se={max_se:.4f}); "
            f"cell-size-weighting counterfactual would predict shift ~ 0"
        )
        # And the shift goes DOWN (cell-size weighting would predict shift = 0;
        # equal weighting on this DGP predicts att_B < att_A because the
        # nonlinear DGP's WAS depends on mean(D²)/mean(D), and replicating
        # low-D units reduces this ratio).
        assert result_b.att < result_a.att

    def test_sup_t_bootstrap_skipped_when_cband_false(self) -> None:
        """``cband=False`` on weighted event-study disables sup-t bootstrap.

        With ``cband=False``, the simultaneous-band machinery doesn't
        run; the result class's ``cband_low`` / ``cband_high`` fields
        (typed ``Optional[np.ndarray]``) stay ``None`` rather than
        being populated with a band.
        """
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 1)
        panel = self._make_event_study_panel(rng, G=200)
        panel = panel.assign(w=np.ones(len(panel)))  # uniform pweight (equivalent to unweighted)
        est = HeterogeneousAdoptionDiD(design="auto", n_bootstrap=99, seed=42)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = est.fit(
                panel,
                outcome_col="outcome",
                dose_col="dose",
                time_col="period",
                unit_col="unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
                cband=False,
            )
        assert isinstance(result, HeterogeneousAdoptionDiDEventStudyResults)
        # cband=False -> no simultaneous band. Result class has Optional[ndarray]
        # cband_low/high: None when bootstrap skipped.
        assert result.cband_low is None
        assert result.cband_high is None

    def test_sup_t_bootstrap_skipped_when_overall_aggregate(self) -> None:
        """``aggregate="overall"`` never invokes sup-t bootstrap."""
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 2)
        panel = _make_two_period_panel(rng, G=300, dose_dist="uniform_0_1", was_true=0.3, sigma=0.1)
        panel = panel.assign(w=np.ones(len(panel)))
        # Patch the bootstrap helper; should NOT be called on overall path.
        with patch("diff_diff.had._sup_t_multiplier_bootstrap") as mock_boot:
            est = HeterogeneousAdoptionDiD(design="auto")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                _ = est.fit(
                    panel,
                    outcome_col="outcome",
                    dose_col="dose",
                    time_col="period",
                    unit_col="unit",
                    aggregate="overall",
                    survey_design=SurveyDesign(weights="w"),
                    cband=True,  # request cband on overall — should be ignored
                )
            assert mock_boot.call_count == 0

    def test_staggered_timing_fail_closed_value_error(self) -> None:
        """Multi-cohort panel without ``first_treat_col`` raises ValueError.

        Locks the Library extension #5 design: paper prescribes "Warn",
        library raises. ``UserWarning`` would let the silent-misuse bug
        class through (only the last cohort is identified under
        Appendix B.2).
        """
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 3)
        # Multi-cohort panel: 3 periods, half-treated-at-t=2, half-treated-at-t=3.
        G = 100
        rows = []
        for g in range(G):
            first_treat = 2 if g < G // 2 else 3
            for t in [1, 2, 3]:
                if t < first_treat:
                    dose = 0.0
                else:
                    dose = rng.uniform(0.1, 1.0)  # cohort-specific dose
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": dose,
                        "outcome": 0.3 * dose + 0.1 * rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        est = HeterogeneousAdoptionDiD(design="auto")
        with pytest.raises(ValueError, match=r"(staggered|cohort|first_treat_col|HAD)"):
            est.fit(
                panel,
                outcome_col="outcome",
                dose_col="dose",
                time_col="period",
                unit_col="unit",
                aggregate="event_study",
            )

    def test_first_treat_col_activates_last_cohort_auto_filter(self) -> None:
        """``first_treat_col=`` activates last-cohort + never-treated auto-filter."""
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 4)
        # G large enough that the surviving (last-cohort + never-treated)
        # subset of ~2/3 of G has enough distinct dose values for the
        # bandwidth selector + local-linear fit.
        G = 600
        rows = []
        for g in range(G):
            # 3 cohorts: 1/3 never-treated, 1/3 treated at t=2, 1/3 treated at t=3.
            third = G // 3
            if g < third:
                first_treat = 0  # never treated
            elif g < 2 * third:
                first_treat = 2  # earlier cohort (dropped by auto-filter)
            else:
                first_treat = 3  # last cohort (kept)
            d_unit = rng.uniform(0.0, 1.0)  # uniform support so Design 1' resolves
            for t in [1, 2, 3]:
                if first_treat == 0 or t < first_treat:
                    dose = 0.0
                else:
                    dose = d_unit
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": dose,
                        "outcome": 0.3 * dose + 0.1 * rng.standard_normal(),
                        "first_treat": first_treat,
                    }
                )
        panel = pd.DataFrame(rows)
        est = HeterogeneousAdoptionDiD(design="auto")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = est.fit(
                panel,
                outcome_col="outcome",
                dose_col="dose",
                time_col="period",
                unit_col="unit",
                first_treat_col="first_treat",
                aggregate="event_study",
            )
        # Should produce a valid event-study result (no raise).
        assert isinstance(result, HeterogeneousAdoptionDiDEventStudyResults)
        # Paper Appendix B.2: filter keeps LAST cohort + never-treated;
        # drops earlier-cohort units. With G=600 and 3 equal-sized
        # cohorts (third=200 each), kept count = 200 never-treated +
        # 200 last cohort = 400. The earlier cohort (first_treat=2) is
        # the 200 dropped units. Lock the exact partition via the
        # result's filter_info metadata (the canonical source of truth
        # for what the auto-filter actually did, NOT the input panel).
        assert result.n_units == 400
        assert result.filter_info is not None
        assert result.filter_info["F_last"] == 3
        assert result.filter_info["n_kept"] == 400
        assert result.filter_info["n_dropped"] == 200
        assert result.filter_info["dropped_cohorts"] == [2]

    def test_assumption_5_6_userwarning_fires_on_design_1_family(self) -> None:
        """Design 1 family (continuous_near_d_lower / mass_point) emits the
        Assumption 5/6 non-testability ``UserWarning`` at fit time.

        Locks the documentation-closure claim: the
        ``HeterogeneousAdoptionDiD`` class docstring's "Non-testable
        assumptions" Notes block + the paper-review L192 closure both
        cite a fit-time warning at the "---- Assumption 5/6 warning on
        Design 1 paths ----" block. Without this regression test the
        warning could silently regress and the docs would still claim
        the surface exists.
        """
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 6)
        # Mass-point DGP triggers the Design 1 path -> warning fires.
        panel = _make_two_period_panel(
            rng,
            G=500,
            dose_dist="mass_point_d_lower_uniform",
            was_true=0.3,
            sigma=0.1,
            d_lower=1.0,
        )
        est = HeterogeneousAdoptionDiD(design="auto")
        with pytest.warns(UserWarning, match=r"Assumption [56]"):
            est.fit(
                panel,
                outcome_col="outcome",
                dose_col="dose",
                time_col="period",
                unit_col="unit",
            )

    def test_safe_inference_no_partial_nan_on_degenerate_panel(self) -> None:
        """safe_inference contract: no partial-NaN state on a degenerate panel.

        On a constant-outcome panel (all delta_Y = 0, no noise), the
        att/se/t_stat/p_value/conf_int fields must EITHER all be
        finite (degenerate path not triggered at this seed/G) OR all
        be NaN (degenerate path triggered) — never a mix. Locks the
        ``safe_inference()`` invariant that downstream inference fields
        move jointly with ``se``.
        """
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 5)
        panel = _make_two_period_panel(
            rng,
            G=400,
            dose_dist="uniform_0_1",
            was_true=0.0,
            sigma=0.0,  # delta_y identically zero
        )
        result = _fit_overall(panel, design="auto")
        # On a strictly degenerate panel, all inference fields move
        # together: either all finite or all NaN. Check the contract.
        inf_fields_nan = [
            np.isnan(result.t_stat),
            np.isnan(result.p_value),
            np.isnan(result.conf_int[0]),
            np.isnan(result.conf_int[1]),
        ]
        # Either all NaN (degenerate path triggered) or all finite
        # (degenerate path not triggered at this seed). Verify the
        # safe_inference invariant: no partial-NaN state.
        assert all(inf_fields_nan) or not any(
            inf_fields_nan
        ), f"safe_inference partial-NaN state: {inf_fields_nan}"

    @staticmethod
    def _make_event_study_panel(rng: np.random.Generator, G: int) -> pd.DataFrame:
        """Build a balanced multi-period HAD panel for event-study fits."""
        d_post = rng.uniform(0.1, 1.0, G)
        rows = []
        for t in [1, 2, 3, 4, 5]:
            for g in range(G):
                if t < 3:  # pre-periods + base anchor
                    dose = 0.0
                    delta = 0.0
                else:
                    dose = d_post[g]
                    delta = 0.3 * d_post[g]
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": dose,
                        "outcome": delta + 0.05 * rng.standard_normal(),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _zero_fraction_panel(n_zero: int, seed: int, G: int = 200) -> pd.DataFrame:
        """continuous_at_zero 2-period panel with EXACTLY ``n_zero`` zero doses."""
        rng = np.random.default_rng(seed)
        d = rng.uniform(0.2, 1.0, G)
        d[:n_zero] = 0.0
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        units = np.repeat(np.arange(G), 2)
        periods = np.tile([1, 2], G)
        dose = np.column_stack([np.zeros(G), d]).ravel()
        outcome = np.column_stack([np.zeros(G), dy]).ravel()
        return pd.DataFrame({"unit": units, "period": periods, "dose": dose, "outcome": outcome})

    def test_extensive_margin_warning_is_10pct_library_convention(self) -> None:
        """Locks TODO L74: the extensive-margin warning is a 10% library convention.

        The paper (de Chaisemartin et al. 2026, Section 2 / Assumption 3)
        prescribes warning users with a positive untreated mass but gives NO
        numeric cutoff, and explicitly RETAINS small untreated shares (Garrett
        et al. 12/2954 ~ 0.4%, nominal coverage). The library picks a 10%
        exactly-zero-dose fraction as the fire threshold — documented in
        REGISTRY § HeterogeneousAdoptionDiD. This pins both the constant and
        the fire/no-fire boundary so the convention cannot drift silently.
        """
        from diff_diff.had import _HAD_EXTENSIVE_MARGIN_ZERO_DOSE_FRAC

        assert _HAD_EXTENSIVE_MARGIN_ZERO_DOSE_FRAC == 0.10

        substr = "exactly-zero post-period dose"
        # At/above 10% (20/200) -> fires.
        with pytest.warns(UserWarning, match=substr):
            HeterogeneousAdoptionDiD().fit(
                self._zero_fraction_panel(20, seed=_BASE_SEED_DEVIATIONS + 10),
                "outcome",
                "dose",
                "period",
                "unit",
            )
        # Just below 10% (19/200 = 9.5%) -> does not fire.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD().fit(
                self._zero_fraction_panel(19, seed=_BASE_SEED_DEVIATIONS + 11),
                "outcome",
                "dose",
                "period",
                "unit",
            )
        assert not any(substr in str(w.message) for w in rec)

    def test_covariates_not_implemented_is_documented(self) -> None:
        """Locks TODO L73: fit(covariates=...) raises NotImplementedError.

        Covariate-adjusted HAD identification (de Chaisemartin et al. 2026,
        Appendix B.1 / Theorem 6) is deferred; the explicit ``covariates=``
        param raises NotImplementedError with the paper pointer rather than a
        bare TypeError. Documented in REGISTRY § HeterogeneousAdoptionDiD.
        """
        panel = self._zero_fraction_panel(1, seed=_BASE_SEED_DEVIATIONS + 12)
        with pytest.raises(NotImplementedError, match="Theorem 6"):
            HeterogeneousAdoptionDiD().fit(
                panel, "outcome", "dose", "period", "unit", covariates=["x1"]
            )
