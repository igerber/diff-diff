"""Methodology verification tests for ImputationDiD.

Targets Borusyak, Jaravel & Spiess (2024), *Revisiting Event-Study Designs:
Robust and Efficient Estimation*, Review of Economic Studies 91(6), 3253-3285
(DOI 10.1093/restud/rdae007).

Paper-equation walk-through (each Verified Component class maps to a numbered
result, verified against the source PDF in
``docs/methodology/papers/borusyak-jaravel-spiess-2024-review.md``):

- **Theorem 1 / 2** (p. 3267-8) — the 3-step imputation estimator (Step 1 fit on
  the untreated set Omega_0 only via eq. 5; Step 2 impute Y(0); Step 3 weighted
  aggregation) recovers the target ATT (``TestB2024Theorem2Imputation``).
- **Theorem 3 / Eqs. 6-8** (p. 3271-2) — conservative clustered variance and the
  *unit-clustered* Equation 8 auxiliary aggregator
  (``TestB2024Theorem3Variance``, ``TestB2024Eq8AuxiliaryAggregator``).
- **Proposition 5** (p. 3266) — without never-treated units, horizons
  ``K_it >= H_bar = max(E_i) - min(E_i)`` are not identified -> NaN + warning
  (``TestB2024Proposition5NoNeverTreated``).
- **Test 1 / Eq. 9 + Proposition 9** (p. 3273-4) — robust pre-trend test on
  Omega_0 only, independent of the treatment-effect estimate
  (``TestB2024Proposition9Test1``).
- Library extensions / deviations (multiplier bootstrap, survey TSL,
  ``aux_partition`` defaults, NaN inference, Prop-5 refuse-to-estimate)
  (``TestB2024LibraryDeviations``).

R-parity (bottom of file, NOT a methodology walk-through): ``TestImputationDiDParityR``
pins Python output against R ``didimputation::did_imputation()`` on fixed-seed
goldens. R ``didimputation`` implements the paper's Equation 8 only at the
cohort x event-time partition (where it equals ``sum(v^2 * tau)/sum(v^2)``); see
``docs/methodology/REGISTRY.md`` ``## ImputationDiD`` "Deviation from R".

See also:

- ``docs/methodology/papers/borusyak-jaravel-spiess-2024-review.md`` (primary-source review)
- ``docs/methodology/REGISTRY.md`` ``## ImputationDiD`` block
- ``METHODOLOGY_REVIEW.md`` ``ImputationDiD`` section
- ``tests/test_imputation.py`` (implementation-detail unit tests)
- ``benchmarks/R/generate_didimputation_golden.R`` (R goldens generator)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from diff_diff import ImputationDiD

# =============================================================================
# Module-level R-fixture availability + per-class seed decorrelation
# =============================================================================

GOLDEN_PATH = Path(__file__).parent.parent / "benchmarks" / "data" / "didimputation_golden.json"
PANEL_PATH = Path(__file__).parent.parent / "benchmarks" / "data" / "didimputation_test_panel.csv"
_R_FIXTURE_AVAILABLE = GOLDEN_PATH.is_file() and PANEL_PATH.is_file()

_BASE_SEED_THM2 = 9101
_BASE_SEED_THM3 = 9202
_BASE_SEED_EQ8 = 9303
_BASE_SEED_PROP5 = 9404
_BASE_SEED_PROP9 = 9505
_BASE_SEED_DEVIATIONS = 9606


# =============================================================================
# Helpers
# =============================================================================


def _make_staggered_panel(
    rng: np.random.Generator,
    *,
    cohorts: List[int],
    n_per_cohort: int = 100,
    n_periods: int = 6,
    tau_constant: Optional[float] = None,
    tau_by_horizon: Optional[Dict[int, float]] = None,
    sigma: float = 0.1,
    include_never_treated: bool = True,
    pretrend_slope: float = 0.0,
) -> pd.DataFrame:
    """Balanced staggered-adoption panel satisfying parallel trends.

    DGP (BJS Assumption 1): ``y_it = c_i + beta_t + w_it * tau_{K_it} + u_it``,
    with ``c_i ~ N(0,1)``, common time trend ``beta_t = 0.5 t`` (parallel
    trends hold -- no cohort-specific trends unless ``pretrend_slope != 0``),
    ``u_it ~ N(0, sigma^2)``. Treatment is absorbing from the cohort's event
    date. ``first_treat = 0`` denotes never-treated.

    ``pretrend_slope != 0`` injects a cohort-specific linear trend
    ``pretrend_slope * cohort_rank * t`` that violates parallel trends (used to
    exercise the pre-trend test's power).
    """
    if tau_constant is None and tau_by_horizon is None:
        tau_constant = 1.0
    rows: List[Dict[str, Any]] = []
    unit_id = 0
    all_cohorts = ([0] + list(cohorts)) if include_never_treated else list(cohorts)
    cohort_rank = {g: r for r, g in enumerate(sorted(cohorts))}
    for g in all_cohorts:
        for _ in range(n_per_cohort):
            c_i = rng.standard_normal()
            for t in range(1, n_periods + 1):
                beta_t = 0.5 * t
                u = sigma * rng.standard_normal()
                treated = g > 0 and t >= g
                if treated:
                    k = t - g
                    if tau_by_horizon is not None:
                        tau = tau_by_horizon.get(k, 0.0)
                    else:
                        tau = tau_constant if tau_constant is not None else 0.0
                else:
                    tau = 0.0
                trend = pretrend_slope * cohort_rank.get(g, 0) * t if g > 0 else 0.0
                y = c_i + beta_t + trend + (tau if treated else 0.0) + u
                rows.append(
                    {
                        "unit": unit_id,
                        "time": t,
                        "first_treat": g,
                        "outcome": y,
                    }
                )
            unit_id += 1
    return pd.DataFrame(rows)


# =============================================================================
# Theorem 1 / 2 — the imputation estimator
# =============================================================================


class TestB2024Theorem2Imputation:
    """Theorem 1/2 (p. 3267-8): 3-step imputation recovers the target ATT,
    fitting the counterfactual model on the untreated set Omega_0 only."""

    def test_recovers_constant_att(self) -> None:
        """Under a constant treatment effect tau=2.0, the overall ATT is recovered.

        DGP: 2 cohorts + never-treated, N=300 units, sigma=0.1. The per-obs SE is
        ~sigma/sqrt(N_1); a 0.05 band is >5 sigma.
        """
        rng = np.random.default_rng(_BASE_SEED_THM2 + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=100, tau_constant=2.0, sigma=0.1
        )
        res = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert abs(res.overall_att - 2.0) < 0.05

    def test_recovers_heterogeneous_event_study(self) -> None:
        """Horizon-specific effects tau_K = 1 + 0.5*K are recovered per horizon."""
        rng = np.random.default_rng(_BASE_SEED_THM2 + 2)
        tau_by_h = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5}
        panel = _make_staggered_panel(
            rng, cohorts=[2, 3], n_per_cohort=120, tau_by_horizon=tau_by_h, sigma=0.1
        )
        res = ImputationDiD().fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        for h, expected in tau_by_h.items():
            assert h in res.event_study_effects, f"missing horizon {h}"
            got = res.event_study_effects[h]["effect"]
            assert abs(got - expected) < 0.06, f"h={h}: {got:.4f} vs {expected}"

    def test_step1_uses_untreated_only(self) -> None:
        """Perturbing a single treated outcome by delta shifts the overall ATT by
        exactly delta/N_1 -- proving treated observations never feed back into the
        Step-1 counterfactual model (eq. 5 is fit on Omega_0 only)."""
        rng = np.random.default_rng(_BASE_SEED_THM2 + 3)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=60, tau_constant=1.0, sigma=0.1
        )
        base = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        n_1 = int(((panel["first_treat"] > 0) & (panel["time"] >= panel["first_treat"])).sum())

        perturbed = panel.copy()
        treated_idx = perturbed.index[
            (perturbed["first_treat"] > 0) & (perturbed["time"] >= perturbed["first_treat"])
        ][0]
        delta = 100.0
        perturbed.loc[treated_idx, "outcome"] += delta
        pert = ImputationDiD().fit(
            perturbed, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        # Only the perturbed obs's own tau_hat changes (weight 1/N_1).
        assert abs((pert.overall_att - base.overall_att) - delta / n_1) < 1e-6


# =============================================================================
# Theorem 3 — conservative clustered variance
# =============================================================================


class TestB2024Theorem3Variance:
    """Theorem 3 (p. 3271-2): the conservative clustered SE is finite/positive and
    (being conservative) is no smaller than a within-cohort-homogeneous benchmark."""

    def test_se_finite_and_positive(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_THM3 + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=80, tau_constant=1.0, sigma=0.2
        )
        res = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_event_study_ses_finite(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_THM3 + 2)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 3], n_per_cohort=80, tau_constant=1.0, sigma=0.2
        )
        res = ImputationDiD().fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        for h, eff in res.event_study_effects.items():
            # Skip the normalized reference period (effect=se=0 by construction).
            if h >= 0 and np.isfinite(eff["effect"]):
                assert np.isfinite(eff["se"]) and eff["se"] > 0, f"h={h}"

    def test_singular_omega0_routes_to_dense_fallback(self) -> None:
        """Regression: a rank-deficient Ω₀ makes A₀'[W]A₀ exactly singular, where
        ``scipy.sparse.linalg.factorized`` raises ``RuntimeError`` at factorization
        time. The variance projection must still route to the dense-`lstsq`
        fallback (not silently zero the untreated influence contributions via
        `np.nan_to_num`), emitting a `UserWarning` so the degraded path is visible.
        """
        # A period observed ONLY among treated obs -> its time FE is unidentified
        # in Ω₀ -> A₀ has an all-zero column -> A₀'A₀ is singular. Drop the
        # never-treated units at t=4 so t=4 appears only for the treated cohort.
        rng = np.random.default_rng(_BASE_SEED_THM3 + 9)
        rows: List[Dict[str, Any]] = []
        uid = 0
        for g in (0, 2):
            for _ in range(30):
                c_i = rng.standard_normal()
                for t in (1, 2, 3, 4):
                    if g == 0 and t == 4:
                        continue  # never-treated not observed at t=4
                    treated = g > 0 and t >= g
                    y = c_i + 0.5 * t + (1.0 if treated else 0.0) + 0.1 * rng.standard_normal()
                    rows.append({"unit": uid, "time": t, "first_treat": g, "outcome": y})
                uid += 1
        panel = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ImputationDiD(rank_deficient_action="silent").fit(
                panel,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
        # The build-time RuntimeError on the singular factorization must trigger
        # the dense-lstsq fallback (with a UserWarning carrying "dense lstsq").
        fallback_warnings = [w for w in caught if "dense lstsq" in str(w.message)]
        assert fallback_warnings, "expected the dense-lstsq fallback under a singular Ω₀"
        # Factorize-once: the build-time singular warning fires a single time for
        # this single-target (overall-only) fit, not once per (g,t).
        assert len(fallback_warnings) == 1
        assert np.isfinite(res.overall_se)


# =============================================================================
# Theorem 3 variance — per-fit factorization cache (perf refactor, #141)
# =============================================================================


class TestImputationVarianceFactorizationCache:
    """The untreated imputation projection
    ``v_untreated = -A_0 (A_0'[W]A_0)^{-1} A_1' w`` has a target-INVARIANT design
    (``A_0``/``A_1``/factorization) and a target-SPECIFIC RHS (``A_1' w``). The
    perf refactor (#141) builds + factorizes the design once per ``fit()``
    (cached) and solves only the RHS per target, replacing the prior per-target
    ``spsolve``. These tests pin that the cached factorize-once / solve-many path
    is **bit-identical** to the prior per-target ``spsolve`` and that the cache is
    actually exercised (built once per fit) and leak-free across fits.
    """

    @staticmethod
    def _cov_panel(seed: int = 4242, n_per_cohort: int = 40):
        """Staggered covariate panel + a positive survey-weight column."""
        rng = np.random.default_rng(seed)
        rows = []
        uid = 0
        for g in (0, 3, 4, 5):
            for _ in range(n_per_cohort):
                c_i = rng.standard_normal()
                x_i = rng.standard_normal()
                for t in range(1, 8):
                    treated = g > 0 and t >= g
                    x1 = x_i + 0.1 * rng.standard_normal()
                    x2 = rng.standard_normal()
                    y = c_i + 0.5 * t + 0.4 * x1 - 0.2 * x2 + (1.0 if treated else 0.0)
                    y += 0.1 * rng.standard_normal()
                    rows.append(
                        {
                            "unit": uid,
                            "time": t,
                            "first_treat": g,
                            "outcome": y,
                            "x1": x1,
                            "x2": x2,
                            "weight": 1.0 + 0.5 * abs(x_i),
                        }
                    )
                uid += 1
        return pd.DataFrame(rows)

    def _split(self, panel):
        treated = (panel["first_treat"] > 0) & (panel["time"] >= panel["first_treat"])
        return panel.loc[~treated], panel.loc[treated]

    @pytest.mark.parametrize("survey", [False, True])
    def test_cached_factorized_matches_spsolve_bit_identical(self, survey: bool) -> None:
        """``_solve_untreated_v`` on the cached factorization reproduces the prior
        per-target ``spsolve`` solution EXACTLY (atol=0) for several RHS, on both
        the unweighted and the survey-weighted (W_0) normal equations. This is the
        productized bit-identity spike justifying the spsolve -> factorized swap.
        """
        from scipy.sparse.linalg import spsolve

        panel = self._cov_panel()
        df_0, df_1 = self._split(panel)
        est = ImputationDiD()
        rng = np.random.default_rng(7)
        sw_0 = rng.uniform(0.5, 2.0, size=len(df_0)) if survey else None

        ctx = est._build_untreated_projection(
            df_0, df_1, "unit", "time", ["x1", "x2"], survey_weights_0=sw_0
        )
        assert not ctx.singular and ctx.solver is not None

        n1 = len(df_1)
        weight_vecs = [
            np.full(n1, 1.0 / n1),
            rng.standard_normal(n1),
            rng.uniform(0.0, 1.0, size=n1),
        ]
        for w in weight_vecs:
            v_cached = est._solve_untreated_v(ctx, w)
            # Reference: the prior per-target path (fresh spsolve on the same
            # cached normal-equations matrix), with the WLS left-weight reapplied.
            a1_w = ctx.A_1.T @ w
            z_ref = spsolve(ctx.A0tA0_csc, a1_w)
            v_ref = -(ctx.A_0 @ z_ref)
            if sw_0 is not None:
                v_ref = v_ref * sw_0
            np.testing.assert_array_equal(v_cached, v_ref)

    def test_cache_reuse_matches_fresh_build(self) -> None:
        """Reusing a cached projection across targets is bit-identical to building
        a fresh projection per target (the cache is a numerical no-op)."""
        panel = self._cov_panel()
        df_0, df_1 = self._split(panel)
        est = ImputationDiD()
        rng = np.random.default_rng(11)
        n1 = len(df_1)
        weights = [np.full(n1, 1.0 / n1), rng.standard_normal(n1)]

        ctx_shared = est._build_untreated_projection(df_0, df_1, "unit", "time", ["x1"])
        for w in weights:
            v_shared = est._solve_untreated_v(ctx_shared, w)
            ctx_fresh = est._build_untreated_projection(df_0, df_1, "unit", "time", ["x1"])
            v_fresh = est._solve_untreated_v(ctx_fresh, w)
            np.testing.assert_array_equal(v_shared, v_fresh)

    @pytest.mark.parametrize("survey", [False, True])
    def test_projection_built_once_per_fit(self, survey: bool) -> None:
        """The cache collapses the (1 + #horizons + #groups) per-target projection
        builds -- and, with bootstrap, the analytical + precompute builds -- to a
        single ``_build_untreated_projection`` per ``fit()``. Holds for the survey
        path too (survey_weights is excluded from the cache key)."""
        from diff_diff.imputation import ImputationDiD as _Cls

        panel = self._cov_panel()
        base = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        kw: Dict[str, Any] = dict(covariates=["x1", "x2"], aggregate="all")
        if survey:
            from diff_diff import SurveyDesign

            kw["survey_design"] = SurveyDesign(weights="weight")

        orig = _Cls._build_untreated_projection
        calls = {"n": 0}

        def counting(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        _Cls._build_untreated_projection = counting  # type: ignore[method-assign]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ImputationDiD().fit(panel, **base, **kw)
        finally:
            _Cls._build_untreated_projection = orig  # type: ignore[method-assign]

        # Multiple aggregation targets exist (so the naive count would be > 1).
        assert res.event_study_effects and res.group_effects
        assert calls["n"] == 1, f"expected 1 projection build, got {calls['n']}"

    def test_projection_built_once_per_fit_with_bootstrap(self) -> None:
        """With ``n_bootstrap > 0`` the analytical aggregation AND the bootstrap
        precompute both consume the projection (overall + #horizons + #groups
        each), yet the shared fit-local cache still builds it exactly once -- the
        bootstrap precompute path threads the same ``proj_cache``."""
        from diff_diff.imputation import ImputationDiD as _Cls

        panel = self._cov_panel()
        base = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        orig = _Cls._build_untreated_projection
        calls = {"n": 0}

        def counting(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        _Cls._build_untreated_projection = counting  # type: ignore[method-assign]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ImputationDiD(n_bootstrap=50, seed=99).fit(
                    panel, **base, covariates=["x1", "x2"], aggregate="all"
                )
        finally:
            _Cls._build_untreated_projection = orig  # type: ignore[method-assign]

        # Bootstrap actually ran (so the precompute path executed), and the
        # analytical + precompute targets still collapse to a single build.
        assert res.bootstrap_results is not None
        assert res.event_study_effects and res.group_effects
        assert calls["n"] == 1, f"expected 1 projection build, got {calls['n']}"

    def test_fit_idempotent_cache_no_leak(self) -> None:
        """The cache is a fit-time local, so refitting the same estimator yields
        identical SEs (no cross-fit cache leak)."""
        panel = self._cov_panel()
        base = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        est = ImputationDiD()
        r1 = est.fit(panel, **base, covariates=["x1", "x2"], aggregate="all")
        r2 = est.fit(panel, **base, covariates=["x1", "x2"], aggregate="all")
        assert r1.overall_se == r2.overall_se
        assert r1.event_study_effects is not None and r2.event_study_effects is not None
        for h in r1.event_study_effects:
            assert r1.event_study_effects[h]["se"] == r2.event_study_effects[h]["se"]


# =============================================================================
# Equation 8 — the unit-clustered auxiliary aggregator
# =============================================================================


class TestB2024Eq8AuxiliaryAggregator:
    """Equation 8 (p. 3272): the auxiliary treatment-effect model uses the
    *unit-clustered* aggregator
    ``tau_tilde_g = sum_i(sum_t v)(sum_t v*tau) / sum_i(sum_t v)^2`` -- NOT the
    naive observation-level mean ``sum(v*tau)/sum(v)``. The two differ whenever a
    unit contributes several observations to a group (coarser partitions) or the
    weights are non-uniform."""

    def test_unit_clustered_formula_handcalc(self) -> None:
        """White-box hand-calculation of _compute_auxiliary_residuals_treated.

        Construct one cohort group (cohort=2) under aux_partition='cohort' with:
          unit A: two obs, tau_hat = (0, 0), v = (1, 1)
          unit B: one obs,  tau_hat = 5,       v = 1
        Unit-clustered Eq. 8:
          a_A = 1+1 = 2, b_A = 0;  a_B = 1, b_B = 5
          tau_tilde = (2*0 + 1*5) / (2^2 + 1^2) = 5/5 = 1.0
        Observation-level mean (the OLD, wrong form):
          sum(v*tau)/sum(v) = (0+0+5)/(1+1+1) = 5/3 ~ 1.667
        So the returned residuals eps = tau_hat - tau_tilde must equal the
        unit-clustered values [-1, -1, 4], not the obs-level [-1.67, -1.67, 3.33].
        Weights are uniform here -- the divergence is driven purely by unit A
        contributing two observations to the group.
        """
        df_1 = pd.DataFrame(
            {
                "unit": ["A", "A", "B"],
                "time": [2, 3, 2],
                "first_treat": [2, 2, 2],
                "_rel_time": [0, 1, 0],
                "outcome": [0.0, 0.0, 5.0],
            }
        )
        est = ImputationDiD(aux_partition="cohort")
        # grand_mean=0 and all-zero FE => y_hat_0 = 0 => tau_hat = outcome.
        eps = est._compute_auxiliary_residuals_treated(
            df_1,
            "outcome",
            "unit",
            "time",
            "first_treat",
            None,
            {"A": 0.0, "B": 0.0},
            {2: 0.0, 3: 0.0},
            0.0,
            None,
            np.array([1.0, 1.0, 1.0]),
        )
        np.testing.assert_allclose(eps, [-1.0, -1.0, 4.0], atol=1e-12)
        # And NOT the observation-level form:
        assert not np.allclose(eps, [5.0 / 3 * -1, 5.0 / 3 * -1, 5 - 5.0 / 3], atol=1e-3)

    def test_nan_tau_co_group_obs_is_no_op(self) -> None:
        """A NaN-tau_hat observation (always v=0 by construction) must NOT poison
        its group's tau_tilde via 0*NaN=NaN. Add unit C with a missing FE (NaN
        tau_hat) and v=0 to the group above; the A/B residuals must be unchanged
        and C's residual is NaN (zeroed downstream in the variance product)."""
        df_1 = pd.DataFrame(
            {
                "unit": ["A", "A", "B", "C"],
                "time": [2, 3, 2, 2],
                "first_treat": [2, 2, 2, 2],
                "_rel_time": [0, 1, 0, 0],
                "outcome": [0.0, 0.0, 5.0, 7.0],
            }
        )
        est = ImputationDiD(aux_partition="cohort")
        # C is absent from unit_fe => NaN alpha_i => NaN tau_hat; its v_treated=0.
        eps = est._compute_auxiliary_residuals_treated(
            df_1,
            "outcome",
            "unit",
            "time",
            "first_treat",
            None,
            {"A": 0.0, "B": 0.0},
            {2: 0.0, 3: 0.0},
            0.0,
            None,
            np.array([1.0, 1.0, 1.0, 0.0]),
        )
        np.testing.assert_allclose(eps[:3], [-1.0, -1.0, 4.0], atol=1e-12)
        assert np.isnan(eps[3])

    def test_public_se_regression_pin_cohort_partition(self) -> None:
        """Public-API guard: the overall SE under aux_partition='cohort' on a
        fixed unbalanced design flows through the unit-clustered Eq. 8 aggregator.

        Correctness of the aggregator is established by the hand-calc above and by
        the R-parity class; this pins the public SE path against regression (a
        revert to the observation-level form would move this number).
        """
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 7)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 4], n_per_cohort=50, n_periods=6, tau_constant=1.0, sigma=0.3
        )
        res = ImputationDiD(aux_partition="cohort").fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        # Regression pin (value produced by the unit-clustered Eq. 8 code path).
        assert res.overall_se == pytest.approx(_EQ8_COHORT_SE_PIN, abs=1e-8)


# Pin value for test_public_se_regression_pin_cohort_partition, produced by the
# unit-clustered Eq. 8 implementation (see the test docstring). Deterministic
# given the fixed-seed design (the SE computation itself has no randomness).
_EQ8_COHORT_SE_PIN = 0.042000264835


# =============================================================================
# Supplementary Appendix A.9 — leave-one-out conservative variance
# =============================================================================


def _synthetic_group(a: np.ndarray, T: np.ndarray):
    """Build (per_unit, per_group) for a single group with per-unit weight sums
    ``a_i`` and unit effects ``T_i``, matching the layout that
    ``_compute_auxiliary_residuals_treated`` passes to ``_leave_one_out_factor``:
    ``per_unit`` MultiIndex (g, u) with column ``a``; ``per_group`` index g with
    column ``den = sum_i a_i^2``.
    """
    idx = pd.MultiIndex.from_arrays([[0] * len(a), list(range(len(a)))], names=["g", "u"])
    per_unit = pd.DataFrame({"a": a, "b": a * T}, index=idx)
    per_group = pd.DataFrame(
        {"num": [float((a**2 * T).sum())], "den": [float((a**2).sum())]},
        index=pd.Index([0], name="g"),
    )
    return per_unit, per_group


class TestB2024AppendixA9LeaveOneOut:
    """BJS 2024 Supplementary Appendix A.9 leave-one-out conservative variance.

    The efficient rescale ``eps_tilde^LO = eps_tilde / (1 - v_ig^2/sum_j v_jg^2)``
    reproduces the direct leave-one-out aggregate ``tau_tilde_it^LO`` exactly at
    the per-unit cluster sum ``psi_i`` (App. A.9). Source: arXiv 2108.12419v5
    App. A.9 (the REStud Supplementary Material is canonical); see REGISTRY.
    """

    _COMMON = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")

    def test_loo_factor_reproduces_direct_leave_one_out(self):
        """PAPER-FIDELITY GATE: the rescaled per-unit cluster sum psi_i equals the
        direct-LOO psi_i (recomputed with the paper's tau_tilde_it^LO), exactly."""
        a = np.array([1.3, 0.7, 2.1, 0.9, 1.6])
        T = np.array([2.0, 3.0, 1.5, 2.5, 1.9])
        per_unit, per_group = _synthetic_group(a, T)
        factor, n_single = ImputationDiD._leave_one_out_factor(per_unit, per_group)
        assert n_single == 0

        N = float((a**2 * T).sum())
        D = float((a**2).sum())
        tau_g = N / D
        # one obs per unit: eps_i = T_i - tau_tilde_g, psi_i = a_i * eps_i * factor_i
        psi_rescale = a * (T - tau_g) * factor.to_numpy()
        # direct LOO: tau_tilde_i^LO = (N - a_i^2 T_i)/(D - a_i^2)
        psi_direct = np.array(
            [a[i] * (T[i] - (N - a[i] ** 2 * T[i]) / (D - a[i] ** 2)) for i in range(len(a))]
        )
        np.testing.assert_allclose(psi_rescale, psi_direct, rtol=0, atol=1e-12)
        # and the closed-form factor 1/(1 - a^2/D)
        np.testing.assert_allclose(factor.to_numpy(), 1.0 / (1.0 - a**2 / D), rtol=0, atol=1e-14)

    def test_loo_factor_equal_weight_is_k_over_k_minus_1(self):
        """Equal-weight K-unit group -> per-unit factor = K/(K-1)."""
        for K in (2, 3, 5, 10, 50):
            a = np.ones(K)
            T = 2.0 + 0.1 * np.arange(K)
            per_unit, per_group = _synthetic_group(a, T)
            factor, n_single = ImputationDiD._leave_one_out_factor(per_unit, per_group)
            assert n_single == 0
            np.testing.assert_allclose(
                factor.to_numpy(), np.full(K, K / (K - 1)), rtol=0, atol=1e-12
            )

    def test_loo_factor_unit_dominated_group_keeps_large_finite_factor(self):
        """A >=2-unit group so dominated that `D - a_i^2` cancels to 0 in float64
        (a=[1.0, 1e-8] -> D==1.0 exactly) must keep its large FINITE factor via
        the exact sum-of-others denominator, NOT silently fall back to non-LOO."""
        a = np.array([1.0, 1e-8])  # D = 1 + 1e-16 rounds to 1.0; D - a_0^2 == 0.0
        T = np.array([2.0, 3.0])
        per_unit, per_group = _synthetic_group(a, T)
        factor, n_single = ImputationDiD._leave_one_out_factor(per_unit, per_group)
        assert n_single == 0  # 2 positive-weight units -> not a singleton
        f0 = factor.to_numpy()[0]
        assert np.isfinite(f0) and f0 > 1e10  # ~1/1e-16, NOT the fallback 1.0

    def test_loo_factor_positive_near_cancellation_is_exact(self):
        """POSITIVE near-cancellation (a=[1.0, 1e-6] -> D - a_0^2 stays > 0 but
        loses ~4 digits) must use the exact sum-of-others denominator, not the
        finite-but-wrong `D - a_i^2`. Exact factor = D / a_1^2."""
        a = np.array([1.0, 1e-6])
        T = np.array([2.0, 3.0])
        per_unit, per_group = _synthetic_group(a, T)
        factor, _ = ImputationDiD._leave_one_out_factor(per_unit, per_group)
        D = float((a**2).sum())
        exact = D / (a[1] ** 2)  # = D / sum_{j!=0} a_j^2
        # rel=1e-9 fails against the lossy `D - a_0^2` (off ~2e-4) but passes exact.
        assert factor.to_numpy()[0] == pytest.approx(exact, rel=1e-9)

    def test_loo_factor_effective_singleton_falls_back(self):
        """A >=2-ROW group with only one positive-weight unit (a=[1.5, 0.0]) is an
        effective singleton (fn. 51) -> factor 1.0 and counted as single."""
        a = np.array([1.5, 0.0])
        T = np.array([2.0, 3.0])
        per_unit, per_group = _synthetic_group(a, T)
        factor, n_single = ImputationDiD._leave_one_out_factor(per_unit, per_group)
        assert n_single == 1
        np.testing.assert_allclose(factor.to_numpy(), [1.0, 1.0], rtol=0, atol=1e-14)

    def test_loo_factor_single_unit_group_falls_back(self):
        """Single positive-weight unit (App. A.9 fn. 51) -> factor 1.0, counted."""
        a = np.array([1.5])
        T = np.array([2.0])
        per_unit, per_group = _synthetic_group(a, T)
        factor, n_single = ImputationDiD._leave_one_out_factor(per_unit, per_group)
        assert n_single == 1
        np.testing.assert_allclose(factor.to_numpy(), [1.0], rtol=0, atol=1e-14)

    def test_loo_se_geq_nonloo_at_unit_clustering(self):
        """Prop. A8 direction: at the default unit clustering, LOO SE >= non-LOO;
        strict on a multi-unit panel. ATT is unchanged."""
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 11)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=40, n_periods=6, tau_constant=1.0
        )
        r0 = ImputationDiD(leave_one_out=False).fit(panel, **self._COMMON)
        r1 = ImputationDiD(leave_one_out=True).fit(panel, **self._COMMON)
        np.testing.assert_allclose(r1.overall_att, r0.overall_att, rtol=0, atol=1e-12)
        assert r1.overall_se > r0.overall_se

    def test_loo_false_is_byte_identical_to_default(self):
        """leave_one_out=False changes nothing on the default path."""
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 12)
        panel = _make_staggered_panel(rng, cohorts=[3], n_per_cohort=50, n_periods=5)
        se_default = ImputationDiD().fit(panel, **self._COMMON).overall_se
        se_loo_false = ImputationDiD(leave_one_out=False).fit(panel, **self._COMMON).overall_se
        assert se_default == se_loo_false

    def test_loo_single_unit_group_warns_and_returns_finite(self):
        """A singleton treated cohort makes each cohort x horizon group a single
        unit -> UserWarning + non-LOO fallback, finite SE."""
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 13)
        rows: List[Dict[str, Any]] = []
        uid = 0
        for _ in range(8):  # never-treated controls
            c_i = rng.standard_normal()
            for t in range(1, 6):
                rows.append(
                    dict(
                        unit=uid,
                        time=t,
                        first_treat=0,
                        outcome=c_i + 0.5 * t + 0.1 * rng.standard_normal(),
                    )
                )
            uid += 1
        c_i = rng.standard_normal()  # ONE treated unit, cohort 3
        for t in range(1, 6):
            rows.append(
                dict(
                    unit=uid,
                    time=t,
                    first_treat=3,
                    outcome=c_i + 0.5 * t + (1.0 if t >= 3 else 0.0) + 0.1 * rng.standard_normal(),
                )
            )
        panel = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match="single positive-weight unit"):
            res = ImputationDiD(leave_one_out=True).fit(panel, **self._COMMON)
        assert np.isfinite(res.overall_se)

    def test_loo_param_validation_and_roundtrip(self):
        """leave_one_out is in get/set_params; a non-bool is rejected (TypeError)
        in __init__ AND at fit-time (closing the naive-setattr set_params bypass)."""
        assert ImputationDiD().get_params()["leave_one_out"] is False
        assert ImputationDiD(leave_one_out=True).get_params()["leave_one_out"] is True
        with pytest.raises(TypeError, match="leave_one_out must be a bool"):
            ImputationDiD(leave_one_out="yes")  # type: ignore[arg-type]
        # set_params is a naive setattr; the fit-time re-check must catch it
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 14)
        panel = _make_staggered_panel(rng, cohorts=[3], n_per_cohort=20, n_periods=5)
        est = ImputationDiD()
        est.set_params(leave_one_out="yes")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="leave_one_out must be a bool"):
            est.fit(panel, **self._COMMON)

    def test_loo_fit_is_idempotent_on_config(self):
        """Repeated fits with leave_one_out=True give identical SE (no config mutation)."""
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 15)
        panel = _make_staggered_panel(rng, cohorts=[3, 4], n_per_cohort=25, n_periods=6)
        est = ImputationDiD(leave_one_out=True)
        se_a = est.fit(panel, **self._COMMON).overall_se
        se_b = est.fit(panel, **self._COMMON).overall_se
        assert se_a == se_b
        assert est.leave_one_out is True

    def test_loo_composes_with_cluster_and_bootstrap(self):
        """LOO applies the same residual rescale under a coarser cluster= and
        under the multiplier bootstrap: finite SE (NOT asserting the >= direction
        away from unit clustering)."""
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 16)
        panel = _make_staggered_panel(rng, cohorts=[3, 4], n_per_cohort=30, n_periods=6)
        panel["state"] = panel["unit"] % 5  # coarser cluster
        se_cluster = (
            ImputationDiD(leave_one_out=True, cluster="state").fit(panel, **self._COMMON).overall_se
        )
        assert np.isfinite(se_cluster)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            se_boot = (
                ImputationDiD(leave_one_out=True, n_bootstrap=49, seed=7)
                .fit(panel, **self._COMMON)
                .overall_se
            )
        assert np.isfinite(se_boot)

    def test_loo_replicate_weight_survey_raises(self):
        """Replicate-weight variance bypasses the conservative IF path where LOO
        lives, so leave_one_out=True must fail closed (not silently no-op)."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(_BASE_SEED_EQ8 + 17)
        panel = _make_staggered_panel(rng, cohorts=[3, 4], n_per_cohort=20, n_periods=5)
        panel["weight"] = 1.0
        panel["rw1"] = 1.0
        panel["rw2"] = 1.0
        design = SurveyDesign(
            weights="weight",
            replicate_weights=["rw1", "rw2"],
            replicate_method="JK1",
            weight_type="pweight",
        )
        with pytest.raises(NotImplementedError, match="leave_one_out=True"):
            ImputationDiD(leave_one_out=True).fit(panel, survey_design=design, **self._COMMON)

    def test_loo_recorded_in_results_metadata(self):
        """The result object is self-describing: leave_one_out is persisted on the
        result, in to_dict(), and surfaced in summary() (it changes reported SEs)."""
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 18)
        panel = _make_staggered_panel(rng, cohorts=[3, 4], n_per_cohort=20, n_periods=5)
        r1 = ImputationDiD(leave_one_out=True).fit(panel, **self._COMMON)
        r0 = ImputationDiD(leave_one_out=False).fit(panel, **self._COMMON)
        assert r1.leave_one_out is True and r0.leave_one_out is False
        assert r1.to_dict()["leave_one_out"] is True
        assert r0.to_dict()["leave_one_out"] is False
        assert "Leave-one-out" in r1.summary()
        assert "Leave-one-out" not in r0.summary()

    @staticmethod
    def _assert_inference_consistent(effects):
        """Every cell is either a genuine estimate (se > 0 -> effect/t/p/CI all
        finite) or a degenerate/reference cell (se == 0 or NaN -> NaN t/p, per the
        safe_inference contract), and at least one cell is a genuine estimate."""
        assert effects is not None and len(effects) > 0
        n_finite = 0
        for eff in effects.values():
            se = eff["se"]
            if np.isfinite(se) and se > 0:
                assert np.isfinite(eff["effect"])
                assert np.isfinite(eff["t_stat"])
                assert np.isfinite(eff["p_value"])
                lo, hi = eff["conf_int"]
                assert np.isfinite(lo) and np.isfinite(hi)
                n_finite += 1
            else:
                # zero or non-finite SE -> undefined t-stat / p-value (NaN)
                assert np.isnan(eff["t_stat"]) and np.isnan(eff["p_value"])
        assert n_finite > 0

    def test_loo_aggregate_all_analytical(self):
        """aggregate="all": LOO routes through the event-study AND group
        aggregators (not just overall). ATT unchanged vs non-LOO, overall LOO SE
        >= non-LOO, and both aggregation surfaces have consistent finite inference."""
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 19)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=40, n_periods=6, tau_constant=1.0
        )
        r0 = ImputationDiD(leave_one_out=False).fit(panel, aggregate="all", **self._COMMON)
        r1 = ImputationDiD(leave_one_out=True).fit(panel, aggregate="all", **self._COMMON)
        np.testing.assert_allclose(r1.overall_att, r0.overall_att, rtol=0, atol=1e-12)
        assert r1.overall_se > r0.overall_se
        self._assert_inference_consistent(r1.event_study_effects)
        self._assert_inference_consistent(r1.group_effects)

    def test_loo_aggregate_all_bootstrap(self):
        """aggregate="all" under the multiplier bootstrap: event-study and group
        inference fields are populated and consistent."""
        rng = np.random.default_rng(_BASE_SEED_EQ8 + 20)
        panel = _make_staggered_panel(rng, cohorts=[3, 4], n_per_cohort=30, n_periods=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ImputationDiD(leave_one_out=True, n_bootstrap=99, seed=3).fit(
                panel, aggregate="all", **self._COMMON
            )
        assert np.isfinite(r.overall_se)
        self._assert_inference_consistent(r.event_study_effects)
        self._assert_inference_consistent(r.group_effects)

    def test_loo_aggregate_all_analytical_survey(self):
        """aggregate="all" with an analytical survey design (weights + PSU): the
        LOO rescale composes through the survey TSL variance with consistent
        inference on every surface."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(_BASE_SEED_EQ8 + 21)
        panel = _make_staggered_panel(rng, cohorts=[3, 4], n_per_cohort=40, n_periods=6)
        panel["weight"] = 1.0 + (panel["unit"] % 3) * 0.1
        panel["psu"] = panel["unit"] % 8
        design = SurveyDesign(weights="weight", psu="psu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ImputationDiD(leave_one_out=True).fit(
                panel, aggregate="all", survey_design=design, **self._COMMON
            )
        assert np.isfinite(r.overall_se)
        self._assert_inference_consistent(r.event_study_effects)
        self._assert_inference_consistent(r.group_effects)

    @pytest.mark.slow
    def test_loo_coverage_geq_nominal(self):
        """MC coverage: on an overfit-prone fine partition, LOO coverage is >=
        non-LOO coverage and near/above nominal (LOO removes the downward bias)."""
        true_att = 1.0
        n_rep = 200
        cov0 = cov1 = 0
        for rep in range(n_rep):
            rng = np.random.default_rng(20000 + rep)
            panel = _make_staggered_panel(
                rng,
                cohorts=[3, 4, 5],
                n_per_cohort=8,
                n_periods=6,
                tau_constant=true_att,
                sigma=1.0,
            )
            r0 = ImputationDiD(leave_one_out=False).fit(panel, **self._COMMON)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r1 = ImputationDiD(leave_one_out=True).fit(panel, **self._COMMON)
            for r, hit in ((r0, "cov0"), (r1, "cov1")):
                lo = r.overall_att - 1.96 * r.overall_se
                hi = r.overall_att + 1.96 * r.overall_se
                if lo <= true_att <= hi:
                    if hit == "cov0":
                        cov0 += 1
                    else:
                        cov1 += 1
        cov0f, cov1f = cov0 / n_rep, cov1 / n_rep
        assert cov1f >= cov0f - 0.02  # LOO no worse than non-LOO
        assert cov1f >= 0.90  # near/above nominal 95%


# =============================================================================
# Proposition 5 — non-identification without never-treated units
# =============================================================================


class TestB2024Proposition5NoNeverTreated:
    """Proposition 5 (p. 3266): with no never-treated units and H_bar =
    max(E_i)-min(E_i), horizons K >= H_bar are not identified -> NaN + warning."""

    def test_horizons_at_or_above_hbar_are_nan_with_warning(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_PROP5 + 1)
        # Cohorts 3 and 5, NO never-treated => H_bar = 5 - 3 = 2.
        panel = _make_staggered_panel(
            rng,
            cohorts=[3, 5],
            n_per_cohort=80,
            n_periods=8,
            tau_constant=1.0,
            sigma=0.1,
            include_never_treated=False,
        )
        with pytest.warns(UserWarning, match="identified"):
            res = ImputationDiD().fit(
                panel,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        assert res.event_study_effects is not None
        h_bar = 2
        for h, eff in res.event_study_effects.items():
            if h >= h_bar:
                assert np.isnan(eff["effect"]), f"h={h} >= H_bar should be NaN"

    def test_never_treated_present_identifies_all_horizons(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_PROP5 + 2)
        panel = _make_staggered_panel(
            rng,
            cohorts=[3, 5],
            n_per_cohort=80,
            n_periods=8,
            tau_constant=1.0,
            sigma=0.1,
            include_never_treated=True,
        )
        res = ImputationDiD().fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        # With never-treated controls, post horizons are identified (finite).
        assert any(
            np.isfinite(eff["effect"]) and h >= 2 for h, eff in res.event_study_effects.items()
        )


# =============================================================================
# Test 1 / Equation 9 + Proposition 9 — robust pre-trend test
# =============================================================================


class TestB2024Proposition9Test1:
    """Test 1 / Eq. 9 (p. 3273): pre-trend test on Omega_0 only; Proposition 9
    (p. 3274): the test is independent of the treatment-effect estimate."""

    def test_pretrend_test_does_not_reject_under_parallel_trends(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_PROP9 + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[4, 5], n_per_cohort=120, n_periods=7, tau_constant=1.0, sigma=0.1
        )
        res = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        pt = res.pretrend_test()
        assert "p_value" in pt
        assert pt["p_value"] > 0.05

    def test_pretrend_test_rejects_under_violation(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_PROP9 + 2)
        panel = _make_staggered_panel(
            rng,
            cohorts=[4, 5],
            n_per_cohort=120,
            n_periods=7,
            tau_constant=1.0,
            sigma=0.1,
            pretrend_slope=0.4,
        )
        res = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        pt = res.pretrend_test()
        assert pt["p_value"] < 0.05

    def test_estimate_independent_of_pretrend_request(self) -> None:
        """Proposition 9: requesting pre-period coefficients does not change the
        treatment-effect estimate (estimation is orthogonal to pre-testing)."""
        rng = np.random.default_rng(_BASE_SEED_PROP9 + 3)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=80, n_periods=7, tau_constant=1.5, sigma=0.1
        )
        common = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        base = ImputationDiD().fit(panel, **common)
        with_pre = ImputationDiD(pretrends=True).fit(panel, **common)
        assert with_pre.overall_att == pytest.approx(base.overall_att, abs=1e-10)


# =============================================================================
# Library extensions / deviations (not in the paper)
# =============================================================================


class TestB2024LibraryDeviations:
    """Library extensions beyond BJS 2024 (documented in REGISTRY.md)."""

    def test_multiplier_bootstrap_is_library_extension(self, ci_params) -> None:
        """The paper proposes only analytical SEs; the multiplier bootstrap on the
        Theorem-3 influence function is a library extension."""
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=80, tau_constant=1.0, sigma=0.2
        )
        n_boot = ci_params.bootstrap(99)
        res = ImputationDiD(n_bootstrap=n_boot, seed=7).fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.bootstrap_results is not None
        assert np.isfinite(res.bootstrap_results.overall_att_se)

    def test_aux_partition_options_all_run(self) -> None:
        """aux_partition choices (cohort_horizon/cohort/horizon) are library
        defaults; the paper does not prescribe the partition."""
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 2)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=60, tau_constant=1.0, sigma=0.2
        )
        common = dict(outcome="outcome", unit="unit", time="time", first_treat="first_treat")
        for partition in ("cohort_horizon", "cohort", "horizon"):
            res = ImputationDiD(aux_partition=partition).fit(panel, **common)
            assert np.isfinite(res.overall_se), partition


# =============================================================================
# R parity — didimputation::did_imputation (skip-guarded)
# =============================================================================


@pytest.fixture(scope="module")
def golden() -> dict:
    if not _R_FIXTURE_AVAILABLE:
        pytest.skip(
            "R didimputation parity fixture not present. Run "
            "`Rscript benchmarks/R/generate_didimputation_golden.R` to regenerate "
            "`benchmarks/data/didimputation_golden.json`."
        )
    with GOLDEN_PATH.open("r") as f:
        return json.loads(f.read())


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _R_FIXTURE_AVAILABLE:
        pytest.skip("R didimputation parity fixture not present.")
    return pd.read_csv(PANEL_PATH)


class TestImputationDiDParityR:
    """Pin Python ImputationDiD against R ``didimputation::did_imputation()``.

    The fixture is an unweighted staggered panel at the cohort x event-time
    partition (R's only mode), which validates the FULL variance machinery — the
    untreated `v_it` projection (Supplementary Proposition A3, otherwise not
    analytically verifiable) and the clustering — against the reference: SEs match
    to ~1e-10 and point estimates to ~1e-7 on the reference platform (the tests
    assert ATT ``abs=1e-6`` / SE ``abs=1e-7`` for cross-platform robustness). At
    this partition with uniform weights
    the unit-clustered Equation 8 coincides with both R's ``sum(v^2*tau)/sum(v^2)``
    and the old observation-level mean, so this class confirms *correctness*; the
    Eq. 8 *distinction* from the old form (which needs non-uniform weights or a
    coarser partition, with no R analogue) is proven by the white-box hand-calc in
    ``TestB2024Eq8AuxiliaryAggregator``.
    """

    def test_overall_att_matches_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = ImputationDiD().fit(
            panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.overall_att == pytest.approx(golden["overall"]["att"], abs=1e-6)

    def test_overall_se_matches_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = ImputationDiD().fit(
            panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.overall_se == pytest.approx(golden["overall"]["se"], abs=1e-7)

    def test_event_study_atts_match_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = ImputationDiD().fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        es = golden["event_study"]
        assert len(es["horizons"]) > 0
        for h, att in zip(es["horizons"], es["att"]):
            # Every golden horizon must be present and finite -- no silent skips.
            assert h in res.event_study_effects, f"missing horizon {h}"
            got = res.event_study_effects[h]["effect"]
            assert np.isfinite(got), f"non-finite ATT at h={h}"
            assert got == pytest.approx(att, abs=1e-6), f"h={h}"

    def test_event_study_ses_match_r(self, golden: dict, panel: pd.DataFrame) -> None:
        """Per-horizon SEs match R didimputation (the variance machinery, not just
        the point estimates) -- ~1e-10 observed on the reference platform, asserted
        here at abs=1e-7 for cross-platform robustness."""
        res = ImputationDiD().fit(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        es = golden["event_study"]
        assert len(es["horizons"]) > 0
        for h, se in zip(es["horizons"], es["se"]):
            # Every golden horizon must be present and finite -- no silent skips.
            assert h in res.event_study_effects, f"missing horizon {h}"
            got = res.event_study_effects[h]["se"]
            assert np.isfinite(got), f"non-finite SE at h={h}"
            assert got == pytest.approx(se, abs=1e-7), f"h={h}"
