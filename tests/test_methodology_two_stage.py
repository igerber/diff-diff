"""Methodology verification tests for TwoStageDiD.

Targets Gardner, J. (2022), *Two-stage differences in differences*,
arXiv:2207.05943 [econ.EM]. Each Verified Component class maps to a numbered
section / equation of the paper, verified against the source PDF in
``docs/methodology/papers/gardner-2022-review.md``:

- **§3, eqs. (4) / (6)** — the two-stage procedure (Stage 1 fits unit+time FE on
  the untreated set Omega_0 only; Stage 2 regresses the residualized outcome on
  treatment indicators) recovers the overall ATT ``E(beta_gp | D_gp = 1)`` (eq. 4)
  and the event-study horizons (Step 2' / eq. 6) under arbitrary heterogeneity
  (``TestGardner2022Section3TwoStageProcedure``).
- **§3.3 + Appendix B** — the joint-GMM Newey-McFadden (1994) Theorem 6.1
  sandwich with the GLOBAL Jacobian inverse, meat clustered at the unit, NO
  finite-sample multiplier; the variance folds in Stage-1 estimation error via
  the ``gamma_hat' c_g`` correction (``TestGardner2022Section33GMMVariance``).
- **fn. 19 + Proposition 5 of Borusyak et al. (2024)** — always-treated units are
  excluded (no untreated obs for Stage 1); without never-treated units horizons
  ``h >= h_bar = max(groups) - min(groups)`` are not identified -> NaN + warning;
  REGISTRY edge cases ``balance_e`` and zero-obs cohorts
  (``TestGardner2022Identification``).
- Library extensions / deviations (multiplier bootstrap on the GMM influence
  function; ``vcov_type`` narrowed to the GMM sandwich) -- Gardner prescribes
  analytical GMM SEs only (``TestGardner2022LibraryDeviations``).

R-parity (bottom of file, NOT a methodology walk-through): ``TestTwoStageDiDParityR``
pins Python output against R ``did2s::did2s()`` on a fixed-seed golden. R ``did2s``
defaults to analytical corrected clustered SEs (``bootstrap = FALSE``), the same
GMM sandwich the library computes; see ``docs/methodology/REGISTRY.md``
``## TwoStageDiD``.

Point estimates coincide with ImputationDiD (Borusyak, Jaravel & Spiess 2024) by
construction -- *functional* equivalence is covered by
``tests/test_two_stage.py::TestTwoStageDiDEquivalence``; this file asserts the
paper-grounded contract and the did2s cross-language parity.

See also:

- ``docs/methodology/papers/gardner-2022-review.md`` (primary-source review)
- ``docs/methodology/REGISTRY.md`` ``## TwoStageDiD`` block
- ``METHODOLOGY_REVIEW.md`` ``TwoStageDiD`` section
- ``tests/test_two_stage.py`` (implementation-detail unit tests)
- ``benchmarks/R/generate_did2s_golden.R`` (R golden generator)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from diff_diff import ImputationDiD, TwoStageDiD

# =============================================================================
# Module-level R-fixture availability + per-class seed decorrelation
# =============================================================================

GOLDEN_PATH = Path(__file__).parent.parent / "benchmarks" / "data" / "did2s_golden.json"
PANEL_PATH = Path(__file__).parent.parent / "benchmarks" / "data" / "did2s_test_panel.csv"
_R_FIXTURE_AVAILABLE = GOLDEN_PATH.is_file() and PANEL_PATH.is_file()

_BASE_SEED_PROC = 7101
_BASE_SEED_VAR = 7202
_BASE_SEED_IDENT = 7303
_BASE_SEED_DEV = 7404


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

    DGP (Gardner eq. 1): ``y_it = c_i + beta_t + w_it * tau_{K_it} + u_it``,
    with ``c_i ~ N(0,1)``, common time trend ``beta_t = 0.5 t`` (parallel
    trends hold -- no cohort-specific trends unless ``pretrend_slope != 0``),
    ``u_it ~ N(0, sigma^2)``. Treatment is absorbing from the cohort's event
    date. ``first_treat = 0`` denotes never-treated.
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
# Section 3, eqs. (4) / (6) — the two-stage procedure
# =============================================================================


class TestGardner2022Section3TwoStageProcedure:
    """§3 (eqs. 4/6): Stage 1 fits unit+time FE on the untreated subsample
    (Omega_0), Stage 2 regresses the residualized outcome on treatment status;
    this recovers the overall ATT (eq. 4) and event-study horizons (eq. 6)."""

    def test_recovers_overall_att_eq4(self) -> None:
        """Under a constant effect tau=2.0, the overall ATT E(beta|D=1) (eq. 4)
        is recovered. DGP: 2 cohorts + never-treated, N=300, sigma=0.1."""
        rng = np.random.default_rng(_BASE_SEED_PROC + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=100, tau_constant=2.0, sigma=0.1
        )
        res = TwoStageDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert abs(res.overall_att - 2.0) < 0.05

    def test_recovers_heterogeneous_event_study_eq6(self) -> None:
        """Horizon-specific effects tau_K = 1 + 0.5*K are recovered per horizon
        via the Step 2' event-study spec (eq. 6)."""
        rng = np.random.default_rng(_BASE_SEED_PROC + 2)
        tau_by_h = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5}
        panel = _make_staggered_panel(
            rng, cohorts=[2, 3], n_per_cohort=120, tau_by_horizon=tau_by_h, sigma=0.1
        )
        res = TwoStageDiD().fit(
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

    def test_stage1_uses_untreated_only(self) -> None:
        """Perturbing a single treated outcome by delta shifts the overall ATT by
        exactly delta/N_treated -- proving treated observations never feed back
        into the Stage-1 counterfactual model (FE fit on Omega_0 only)."""
        rng = np.random.default_rng(_BASE_SEED_PROC + 3)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=60, tau_constant=1.0, sigma=0.1
        )
        base = TwoStageDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        n_treated = int(
            ((panel["first_treat"] > 0) & (panel["time"] >= panel["first_treat"])).sum()
        )

        perturbed = panel.copy()
        treated_idx = perturbed.index[
            (perturbed["first_treat"] > 0) & (perturbed["time"] >= perturbed["first_treat"])
        ][0]
        delta = 100.0
        perturbed.loc[treated_idx, "outcome"] += delta
        pert = TwoStageDiD().fit(
            perturbed, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        # Only the perturbed obs's own residual changes (Stage-2 weight 1/N_treated).
        assert abs((pert.overall_att - base.overall_att) - delta / n_treated) < 1e-6

    def test_coincides_with_imputation_estimand(self) -> None:
        """Gardner's two-stage estimator coincides in point estimates with the
        imputation estimator (BJS 2024 p. 3258 / paper review "Relation"). The
        overall ATT matches ImputationDiD to machine precision (they differ only
        in the variance estimator)."""
        rng = np.random.default_rng(_BASE_SEED_PROC + 4)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 4], n_per_cohort=70, n_periods=6, tau_constant=1.5, sigma=0.2
        )
        ts = TwoStageDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        imp = ImputationDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert ts.overall_att == pytest.approx(imp.overall_att, abs=1e-10)

    def test_covariates_residualized_in_both_stages(self) -> None:
        """The live ``covariates=`` path (fn. 9 in-both-stages): a time-varying
        covariate correlated with treatment status would bias the ATT if ignored,
        but residualizing it in both stages (Stage-1 delta_hat fit on Omega_0,
        subtracted from all obs) recovers the true ATT.

        DGP: y = c_i + beta_t + theta*x + tau*D + u, with x shifted by +s on
        treated-post obs (so x is correlated with D). Omitting x leaves theta*x in
        y_tilde -> loads onto D -> ATT biased by ~theta*s; including x removes it.
        """
        rng = np.random.default_rng(_BASE_SEED_PROC + 5)
        theta, tau, shift = 2.0, 1.0, 1.0
        rows: List[Dict[str, Any]] = []
        uid = 0
        for g in (0, 3, 4):
            for _ in range(80):
                c_i = rng.standard_normal()
                for t in range(1, 7):
                    treated = g > 0 and t >= g
                    x = rng.standard_normal() + (shift if treated else 0.0)
                    y = (
                        c_i
                        + 0.5 * t
                        + theta * x
                        + (tau if treated else 0.0)
                        + 0.1 * rng.standard_normal()
                    )
                    rows.append({"unit": uid, "time": t, "first_treat": g, "x": x, "outcome": y})
                uid += 1
        panel = pd.DataFrame(rows)

        att_no_cov = (
            TwoStageDiD()
            .fit(panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat")
            .overall_att
        )
        att_cov = (
            TwoStageDiD()
            .fit(
                panel,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x"],
            )
            .overall_att
        )
        # Without adjustment, the confounded covariate biases the ATT well away
        # from tau; with adjustment, the true ATT is recovered.
        assert abs(att_no_cov - tau) > 0.5, f"expected bias, got {att_no_cov:.4f}"
        assert abs(att_cov - tau) < 0.05, f"covariate path failed: {att_cov:.4f}"


# =============================================================================
# Section 3.3 + Appendix B — joint-GMM Newey-McFadden Theorem 6.1 variance
# =============================================================================


class TestGardner2022Section33GMMVariance:
    """§3.3 + App. B: the GMM sandwich with the GLOBAL Jacobian inverse, meat
    clustered at the unit, NO finite-sample multiplier; the variance folds in
    Stage-1 estimation error via the gamma_hat' c_g correction."""

    def test_overall_se_finite_and_positive(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_VAR + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=80, tau_constant=1.0, sigma=0.2
        )
        res = TwoStageDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_event_study_ses_finite(self) -> None:
        rng = np.random.default_rng(_BASE_SEED_VAR + 2)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 3], n_per_cohort=80, tau_constant=1.0, sigma=0.2
        )
        res = TwoStageDiD().fit(
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

    def test_first_stage_correction_is_nontrivial(self) -> None:
        """The GMM sandwich folds Stage-1 FE estimation uncertainty into the score
        via the gamma_hat' c_g correction. On a design with limited untreated
        support per FE (so the first-stage uncertainty is non-negligible), the
        GMM SE is materially LARGER than a naive benchmark that treats the
        residualized outcome y_tilde as raw data and ignores first-stage
        estimation -- the iid SE of its mean, ``std(y_tilde) / sqrt(N_treated)``.

        Asserts a directional/magnitude gap (not bare inequality) so a
        near-homoskedastic coincidence can't pass it. The overall ATT is the mean
        of ``tau_hat`` (= y_tilde for treated obs) over treated observations, so
        the naive floor is exactly the no-first-stage SE of that mean.
        """
        rng = np.random.default_rng(_BASE_SEED_VAR + 3)
        # Few periods + modest units -> each time/unit FE rests on little
        # untreated support, so the first-stage correction visibly inflates the SE.
        panel = _make_staggered_panel(
            rng, cohorts=[2, 3], n_per_cohort=40, n_periods=4, tau_constant=1.0, sigma=0.5
        )
        res = TwoStageDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        gmm_se = res.overall_se

        # Naive benchmark: y_tilde for the treated obs is exposed as `tau_hat`;
        # the overall ATT is its (unweighted) mean. The SE of that mean IGNORING
        # the generated-regressand (first-stage) correction is std/sqrt(N_treated).
        tau_hat = res.treatment_effects["tau_hat"].values
        tau_hat = tau_hat[np.isfinite(tau_hat)]
        naive_floor = float(np.std(tau_hat, ddof=1)) / np.sqrt(len(tau_hat))

        assert np.isfinite(gmm_se) and gmm_se > 0
        assert gmm_se > naive_floor * 1.05, (
            f"GMM SE {gmm_se:.5f} should exceed the no-first-stage floor "
            f"{naive_floor:.5f} by a clear margin (the gamma_hat' c_g correction)"
        )

    def test_no_fsa_global_inverse_se_pin(self) -> None:
        """Regression pin of the overall SE on a fixed-seed panel. Locks the
        global-inverse + no-finite-sample-multiplier GMM sandwich convention; a
        revert to a per-cluster inverse or an FSA multiplier would move this
        number. Deterministic given the seed (the SE computation has no
        randomness)."""
        rng = np.random.default_rng(_BASE_SEED_VAR + 7)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 4], n_per_cohort=50, n_periods=6, tau_constant=1.0, sigma=0.3
        )
        res = TwoStageDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        assert res.overall_se == pytest.approx(_SE_PIN, abs=1e-8)

    def test_pretrends_event_study_ses_finite(self) -> None:
        """With pretrends=True the Stage-2 design adds pre-period leads; their GMM
        SEs are finite and the overall ATT is unchanged (REGISTRY edge case)."""
        rng = np.random.default_rng(_BASE_SEED_VAR + 4)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=80, n_periods=6, tau_constant=1.0, sigma=0.2
        )
        base = TwoStageDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        res = TwoStageDiD(pretrends=True).fit(
            panel,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert res.event_study_effects is not None
        # At least one pre-period lead is present and has a finite SE.
        leads = [h for h in res.event_study_effects if h < 0]
        assert leads, "pretrends=True should expose pre-period leads"
        for h in leads:
            eff = res.event_study_effects[h]
            if np.isfinite(eff["effect"]):
                assert np.isfinite(eff["se"]), f"lead h={h} SE non-finite"
        # Overall ATT is unchanged by requesting pre-period leads.
        assert res.overall_att == pytest.approx(base.overall_att, abs=1e-8)

    def test_unit_clustered_default(self) -> None:
        """cluster=None clusters the meat at the unit level (Appendix B
        vce(cluster id)); n_clusters equals the number of units."""
        rng = np.random.default_rng(_BASE_SEED_VAR + 5)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=50, tau_constant=1.0, sigma=0.2
        )
        res = TwoStageDiD().fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.n_clusters == panel["unit"].nunique()

    def test_singular_omega0_warns_and_falls_back(self) -> None:
        """A rank-deficient Stage-1 design (a period observed only among treated
        obs -> its time FE is unidentified in Omega_0 -> X_10'X_10 singular) makes
        the sparse factorization in the GMM variance fail; the code must emit a
        UserWarning and route to the certified sparse-LSMR fallback, still
        returning a finite SE."""
        rng = np.random.default_rng(_BASE_SEED_VAR + 9)
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
            res = TwoStageDiD(rank_deficient_action="silent").fit(
                panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )
        assert any(
            "falling back to sparse LSMR" in str(w.message) for w in caught
        ), "expected the sparse-LSMR fallback warning under a singular Omega_0"
        assert np.isfinite(res.overall_se)

    def test_bootstrap_scores_use_exact_residuals(self) -> None:
        """The multiplier bootstrap builds its per-cluster GMM scores from the SAME
        exact Stage-1 / Stage-2 residuals as the analytical variance (the shared
        `_exact_gmm_residuals` helper), NOT the ~1e-7 iterative residualized
        outcome. Since bootstrap SEs override the analytical SE when
        `n_bootstrap > 0`, a residual mismatch would silently report the pre-fix
        ~1% approximate SE. White-box: the bootstrap influence variance
        `bread @ (S'S) @ bread` must equal the analytical `overall_se**2` to
        machine precision on an unbalanced untreated panel (where iterative and
        exact residuals differ)."""
        rng = np.random.default_rng(_BASE_SEED_VAR + 11)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 4], n_per_cohort=50, n_periods=6, tau_constant=1.0, sigma=0.3
        )
        est = TwoStageDiD()
        res = est.fit(panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        # Reconstruct the Stage-1 internals the bootstrap score-builder consumes.
        df = panel.copy()
        omega_0_mask = ~((df["first_treat"] > 0) & (df["time"] >= df["first_treat"]))
        unit_fe, time_fe, gm, delta_hat, kcm = est._fit_untreated_model(
            df, "outcome", "unit", "time", None, omega_0_mask
        )
        df["_y_tilde"] = est._residualize(
            df, "outcome", "unit", "time", None, unit_fe, time_fe, gm, delta_hat
        )
        treated = ((df["first_treat"] > 0) & (df["time"] >= df["first_treat"])).values
        X_2 = treated.astype(float).reshape(-1, 1)
        S, bread, _, _ = est._compute_cluster_S_scores(
            df,
            "unit",
            "time",
            None,
            omega_0_mask,
            unit_fe,
            time_fe,
            delta_hat,
            kcm,
            X_2,
            df["unit"].values,
            None,
        )
        v_boot = bread @ (S.T @ S) @ bread
        # Identical to the analytical exact SE (would be ~1% off with iterative residuals).
        assert np.sqrt(v_boot[0, 0]) == pytest.approx(res.overall_se, abs=1e-9)

    def test_bootstrap_event_study_scores_use_exact_residuals(self) -> None:
        """Multi-column (event-study) bootstrap GMM scores also use the shared
        exact-residual helper (CI-review D1). White-box: for a multi-horizon
        Stage-2 design, the bootstrap influence variance `bread @ (S'S) @ bread`
        equals the analytical `_compute_gmm_variance` vcov **elementwise** to
        machine precision — both are single-sourced through `_exact_gmm_residuals`,
        so the k>1 path cannot silently diverge onto the iterative residuals."""
        rng = np.random.default_rng(_BASE_SEED_VAR + 12)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 4], n_per_cohort=50, n_periods=6, tau_constant=1.0, sigma=0.3
        )
        est = TwoStageDiD()
        est.fit(panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        df = panel.copy()
        omega_0_mask = ~((df["first_treat"] > 0) & (df["time"] >= df["first_treat"]))
        unit_fe, time_fe, gm, delta_hat, kcm = est._fit_untreated_model(
            df, "outcome", "unit", "time", None, omega_0_mask
        )
        df["_y_tilde"] = est._residualize(
            df, "outcome", "unit", "time", None, unit_fe, time_fe, gm, delta_hat
        )
        # Multi-column Stage-2 design: indicators for event-time horizons 0 and 1.
        treated = (df["first_treat"] > 0) & (df["time"] >= df["first_treat"])
        horizon = (df["time"] - df["first_treat"]).where(treated, other=-1).to_numpy()
        X_2 = np.column_stack([(horizon == 0).astype(float), (horizon == 1).astype(float)])
        cluster_ids = df["unit"].to_numpy()

        v_analytical = est._compute_gmm_variance(
            df,
            "unit",
            "time",
            None,
            omega_0_mask,
            unit_fe,
            time_fe,
            delta_hat,
            kcm,
            X_2,
            cluster_ids,
        )
        S, bread, _, _ = est._compute_cluster_S_scores(
            df,
            "unit",
            "time",
            None,
            omega_0_mask,
            unit_fe,
            time_fe,
            delta_hat,
            kcm,
            X_2,
            cluster_ids,
            None,
        )
        v_boot = bread @ (S.T @ S) @ bread
        np.testing.assert_allclose(v_boot, v_analytical, atol=1e-9)


# Pin value for test_no_fsa_global_inverse_se_pin, produced by the global-inverse
# GMM sandwich code path (see the test docstring). Deterministic given the
# fixed-seed design (the SE computation itself has no randomness).
_SE_PIN = 0.03679572008170665


# =============================================================================
# Identification — always-treated, Proposition 5, edge cases
# =============================================================================


class TestGardner2022Identification:
    """fn. 19 (always-treated exclusion) + Proposition 5 of Borusyak et al.
    (2024) (no never-treated -> horizons h >= h_bar unidentified) + REGISTRY
    edge cases (balance_e, zero-obs cohorts)."""

    def test_always_treated_excluded_with_warning(self) -> None:
        """Units treated in the first observed period have no untreated obs for
        Stage 1; they are excluded with a warning listing the IDs, and their
        treated obs do not enter Stage 2."""
        rng = np.random.default_rng(_BASE_SEED_IDENT + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=50, n_periods=6, tau_constant=1.0, sigma=0.1
        )
        n_treated_orig = (
            TwoStageDiD()
            .fit(panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat")
            .n_treated_units
        )

        # Force 5 units from a treated cohort to onset at period 1 (= min time),
        # making them always-treated (no untreated obs for Stage 1).
        cohort3_units = panel.loc[panel["first_treat"] == 3, "unit"].unique()[:5]
        panel.loc[panel["unit"].isin(cohort3_units), "first_treat"] = 1

        with pytest.warns(UserWarning, match="treated in all observed periods"):
            res = TwoStageDiD().fit(
                panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )
        # The 5 always-treated units are dropped, so the treated-unit count falls
        # by exactly 5, and their obs do not enter Stage 2.
        assert res.n_treated_units == n_treated_orig - 5
        assert np.isfinite(res.overall_att)

    def test_no_never_treated_horizons_nan(self) -> None:
        """Proposition 5 (Borusyak et al. 2024): with no never-treated units and
        h_bar = max(groups) - min(groups), horizons h >= h_bar are not identified
        -> NaN effect with n_obs > 0 and a warning. Regression-pins the behavior
        implemented at two_stage.py:2531-2674 (mirror of ImputationDiD)."""
        rng = np.random.default_rng(_BASE_SEED_IDENT + 2)
        # Cohorts 3 and 5, NO never-treated => h_bar = 5 - 3 = 2.
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
            res = TwoStageDiD().fit(
                panel,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
        assert res.event_study_effects is not None
        h_bar = 2
        flagged = [
            h
            for h, eff in res.event_study_effects.items()
            if h >= h_bar and np.isnan(eff["effect"])
        ]
        assert flagged, "expected NaN horizons at or above h_bar"
        for h in flagged:
            assert res.event_study_effects[h]["n_obs"] > 0, f"h={h} should keep n_obs>0"

    def test_balance_e_no_qualifying_cohorts_warns(self) -> None:
        """balance_e larger than any cohort's available pre-window -> no cohort
        qualifies -> warning + event study contains only the reference period."""
        rng = np.random.default_rng(_BASE_SEED_IDENT + 3)
        panel = _make_staggered_panel(
            rng, cohorts=[2, 3], n_per_cohort=60, n_periods=5, tau_constant=1.0, sigma=0.1
        )
        with pytest.warns(UserWarning, match="balance_e"):
            res = TwoStageDiD().fit(
                panel,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
                balance_e=10,  # impossible pre-window
            )
        assert res.event_study_effects is not None
        # Only the reference period survives.
        finite_effects = [
            h for h, eff in res.event_study_effects.items() if eff.get("n_obs", 0) > 0
        ]
        assert finite_effects == [] or all(
            res.event_study_effects[h]["effect"] == 0.0 for h in res.event_study_effects
        )

    def test_zero_obs_cohort_nan(self) -> None:
        """A cohort whose treated obs ALL fall at an unidentified period -> its
        group effect is NaN with n_obs=0 (REGISTRY edge case). Constructed: period
        4 is observed only among treated obs (never-treated absent at t=4), so its
        time FE is unidentified in Ω₀ and every t=4 outcome has NaN y_tilde. Cohort
        4 is treated only at t=4 -> all its treated obs are NaN -> n_obs=0; cohort 3
        keeps its identified t=3 effect (n_obs > 0)."""
        rng = np.random.default_rng(_BASE_SEED_IDENT + 4)
        rows: List[Dict[str, Any]] = []
        uid = 0
        # Never-treated: observed t=1,2,3 only (absent at t=4 -> t=4 has no
        # untreated obs anywhere -> its time FE is unidentified).
        for _ in range(40):
            c_i = rng.standard_normal()
            for t in (1, 2, 3):
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "first_treat": 0,
                        "outcome": c_i + 0.5 * t + 0.1 * rng.standard_normal(),
                    }
                )
            uid += 1
        # Cohort 3 (treated t>=3) and cohort 4 (treated only at t=4), t=1..4.
        for g in (3, 4):
            for _ in range(40):
                c_i = rng.standard_normal()
                for t in (1, 2, 3, 4):
                    treated = t >= g
                    y = c_i + 0.5 * t + (1.0 if treated else 0.0) + 0.1 * rng.standard_normal()
                    rows.append({"unit": uid, "time": t, "first_treat": g, "outcome": y})
                uid += 1
        panel = pd.DataFrame(rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # rank / NaN-y_tilde / lstsq warnings expected
            res = TwoStageDiD(rank_deficient_action="silent").fit(
                panel,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="group",
            )
        assert res.group_effects is not None
        # Cohort 4's only treated obs is the unidentified t=4 -> NaN, n_obs=0.
        assert 4 in res.group_effects
        c4 = res.group_effects[4]
        assert c4["n_obs"] == 0, f"cohort 4 should have n_obs=0, got {c4['n_obs']}"
        assert np.isnan(c4["effect"]), "cohort 4 effect must be NaN"
        assert np.isnan(c4["se"]) and np.isnan(c4["t_stat"]) and np.isnan(c4["p_value"])
        # Cohort 3 keeps an identified t=3 effect.
        assert 3 in res.group_effects and res.group_effects[3]["n_obs"] > 0


# =============================================================================
# Library extensions / deviations
# =============================================================================


class TestGardner2022LibraryDeviations:
    """Gardner (2022) prescribes analytical GMM SEs only (§3.3 + Appendix B).
    The library adds a multiplier bootstrap and narrows vcov_type to the GMM
    sandwich -- both documented in REGISTRY ## TwoStageDiD."""

    def test_multiplier_bootstrap_is_library_extension(self, ci_params) -> None:
        """The multiplier bootstrap on the GMM influence function runs and yields
        a finite bootstrap SE; it is a library extension (the paper proposes no
        bootstrap; R did2s defaults to analytical SEs)."""
        rng = np.random.default_rng(_BASE_SEED_DEV + 1)
        panel = _make_staggered_panel(
            rng, cohorts=[3, 4], n_per_cohort=60, tau_constant=1.0, sigma=0.2
        )
        n_boot = ci_params.bootstrap(99)
        res = TwoStageDiD(n_bootstrap=n_boot, seed=123).fit(
            panel, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.bootstrap_results is not None
        assert np.isfinite(res.bootstrap_results.overall_att_se)
        assert res.bootstrap_results.overall_att_se > 0

    def test_vcov_type_narrow_to_hc1(self) -> None:
        """vcov_type is permanently narrow to the GMM sandwich: classical/hc2/
        hc2_bm are rejected at construction (the GMM-corrected meat has no single
        hat matrix spanning both stages)."""
        for bad in ("classical", "hc2", "hc2_bm"):
            with pytest.raises(ValueError, match="rejected"):
                TwoStageDiD(vcov_type=bad)
        # conley is deferred (separate error), and an unknown type is invalid.
        with pytest.raises(ValueError):
            TwoStageDiD(vcov_type="conley")
        with pytest.raises(ValueError):
            TwoStageDiD(vcov_type="bogus")


# =============================================================================
# R parity — did2s::did2s() (skip-guarded golden)
# =============================================================================


@pytest.fixture(scope="module")
def golden() -> dict:
    if not _R_FIXTURE_AVAILABLE:
        pytest.skip(
            "R did2s parity fixture not present. Run "
            "`Rscript benchmarks/R/generate_did2s_golden.R` to regenerate "
            "`benchmarks/data/did2s_golden.json`."
        )
    with GOLDEN_PATH.open("r") as f:
        return json.loads(f.read())


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _R_FIXTURE_AVAILABLE:
        pytest.skip("R did2s parity fixture not present.")
    return pd.read_csv(PANEL_PATH)


class TestTwoStageDiDParityR:
    """Pin Python TwoStageDiD output against R ``did2s::did2s()`` (analytical
    corrected clustered SEs, bootstrap = FALSE, clustered at unit)."""

    def test_overall_att_matches_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = TwoStageDiD().fit(
            panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.overall_att == pytest.approx(golden["overall"]["att"], abs=1e-6)

    def test_overall_se_matches_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = TwoStageDiD().fit(
            panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert res.overall_se == pytest.approx(golden["overall"]["se"], abs=1e-7)

    def test_event_study_atts_match_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = TwoStageDiD().fit(
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
            assert h in res.event_study_effects, f"missing horizon {h}"
            got = res.event_study_effects[h]["effect"]
            assert np.isfinite(got), f"non-finite ATT at h={h}"
            assert got == pytest.approx(att, abs=1e-6), f"h={h}"

    def test_event_study_ses_match_r(self, golden: dict, panel: pd.DataFrame) -> None:
        res = TwoStageDiD().fit(
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
            assert h in res.event_study_effects, f"missing horizon {h}"
            got = res.event_study_effects[h]["se"]
            assert np.isfinite(got), f"non-finite SE at h={h}"
            assert got == pytest.approx(se, abs=1e-7), f"h={h}"
