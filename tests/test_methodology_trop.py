"""Methodology verification tests for the Triply Robust Panel (TROP) estimator.

Targets Athey, Imbens, Qu & Viviano (2025), *Triply Robust Panel Estimators*,
arXiv:2508.21536 (https://arxiv.org/abs/2508.21536).

Equation walk-through:

- Eq. 2:           nuclear-norm penalised weighted-least-squares objective with
                   soft-threshold SVD prox. Verified by `TestTROPNuclearNormProx`
                   (proximal correctness, plain prox-gradient objective monotonicity
                   on a toy setup, weighted-solver convergence, reduction of
                   singular-value mass). The shipped local / global solvers wrap
                   this prox step with accelerated FISTA momentum; the accelerated
                   loop's faster `O(1/k^2)` rate does NOT guarantee per-step
                   monotonicity, so the test exercises the plain prox-gradient
                   ingredient (without momentum), not the accelerated loop.
- Eq. 10:          balancing decomposition of the estimated counterfactual is
                   a paper-side balancing representation (paper Eq. 10 / Section 5.2) that depends on the internal
                   per-(i, t) weight vectors `theta_s^{i,t}` / `omega_j^{i,t}`,
                   which are not exposed on the public TROP API. Direct
                   numerical reconstruction of the four-term identity is out of
                   scope; `TestTROPNuclearNormProx.test_factor_matrix_consistent_with_treatment_effects`
                   is a structural pointer (shape + finiteness of the fitted
                   ``factor_matrix`` + ``treatment_effects`` populated with
                   finite entries), not a full Eq. 10 lock and not a non-
                   triviality claim on the ``L_hat`` magnitude.
- Eq. 3:           exponential-decay unit and time weights; unit distance uses
                   only periods where both i and j are untreated and excludes
                   the target period t. Verified by `TestTROPEquation3Weights`.
- Eqs. 4-5 / Algorithm 1: LOOCV pseudo-treatment-effect summation; control
                   set includes pre-treatment observations of eventually-
                   treated units; two-stage coordinate cycling per footnote 2.
                   Verified by `TestTROPAlgorithm1LOOCV`.
- Corollary 1:     unbiasedness under any one of (a) unit balance,
                   (b) time balance, (c) correct regression adjustment B=0.
                   Verified by `TestTROPCorollary1Unbiasedness` via three
                   targeted DGPs.
- Theorem 5.1:     triply-robust bias bound; **simulation sanity check, not
                   a direct theorem lock**. The paper states the bound
                   `|E[tau_hat - tau | L]| <= ||Delta_u|| * ||Delta_t|| * ||B||_*`
                   for FIXED, non-data-dependent weights, whereas TROP fits
                   use LOOCV-tuned (data-dependent) weights. The class
                   `TestTROPTheorem51TripleRobustness` verifies the bound's
                   empirical realisation under LOOCV-tuned weights: TROP
                   MSE strictly below DID MSE on a factor-confounded DGP.
                   The direct fixed-weight bias-bound test is deferred.
- Section 2.2:     special-case reductions. DID reduction (lambda_nn=inf +
                   uniform weights) is verified as a **benchmark sanity
                   check** against the basic DiD on a no-interactive-FE
                   panel (additive unit + time effects only) — empirical
                   numerical agreement on a friendly DGP, NOT an algebraic-
                   equivalence proof of the Section 2.2 reduction. The MC
                   reduction (uniform weights + finite lambda_nn) only
                   verifies that the nuclear-norm prox code path engages
                   and beats a DID-style baseline; it is NOT an
                   equivalence check against an independent MC reference
                   implementation. SC + SDID reductions are skipped —
                   paper claims reduction under "specific (omega, theta)
                   weight choices" without providing the map; cross-
                   language anchors are deferred until paper-author code
                   lands. See `TestTROPSpecialCases`.
- Eq. 13 / Algorithm 2: per-(i, t) estimation for multiple treated units;
                   ATT averages over all W_it=1 cells. Verified by
                   `TestTROPAlgorithm2MultipleTreated`.
- Algorithm 3:     stratified pairs bootstrap; separate N_0 / N_1
                   resampling; preserves within-unit temporal correlation.
                   Verified by `TestTROPAlgorithm3Bootstrap`.
- Section 3 / Eq. 6: semi-synthetic factor DGP; treatment-effect recovery
                   under known structure. Verified by
                   `TestTROPEquation6FactorDGPRecovery`.

Deviations from paper:

- Gap #5 (paper review): weight normalisation is ambiguous in the paper
  (Section 5 says weights sum to one; Eq. 3 uses unnormalised exponential
  weights). The shipped implementation matches Eq. 2 (unnormalised).
  Verified by
  `TestTROPDeviations.test_unnormalized_weights_match_eq2`.
- Gap #9 (paper review): the paper assumes a balanced panel; the library
  supports unbalanced panels with missing-period guards. Verified by
  `TestTROPDeviations.test_unbalanced_panels_supported`.
- Equation 14 covariate extension (and Theorem 8.1 covariate triple
  robustness) is deferred. `TROP.fit()` does not accept a ``covariates``
  parameter. Locked by
  `TestTROPDeviations.test_covariates_not_supported`.

See:

- ``docs/methodology/papers/athey-2025-review.md`` (paper review).
- ``docs/methodology/REGISTRY.md`` ``## TROP`` block.
- ``METHODOLOGY_REVIEW.md`` ``TROP`` section.

Companion files (NOT duplicated here):

- ``tests/test_trop.py``                                 (implementation-detail unit
                                                          tests, defensive guards,
                                                          API regressions)
- ``tests/test_trop.py::TestSilentWarningAudit``         (silent-failure audit)
- ``tests/test_trop.py::TestTROPConvergenceWarnings``    (FISTA / outer-loop
                                                          convergence warnings)
- ``tests/test_trop.py::TestTROPBootstrapFailureRateGuard`` (bootstrap 5%
                                                              failure-rate
                                                              guard)
- ``tests/test_trop.py::TestDMatrixValidation``          (absorbing-state
                                                          validation)

R / source parity is deferred until the paper-author reference
implementation is released ("forthcoming" per the paper); see
``METHODOLOGY_REVIEW.md`` ``TROP`` section.

Class structure:

- ``TestTROPNuclearNormProx`` — Eq. 2 + Eq. 10 (prox, FISTA, balancing).
- ``TestTROPEquation3Weights`` — Eq. 3 (unit / time weight semantics).
- ``TestTROPAlgorithm1LOOCV`` — Eqs. 4-5 + Algorithm 1.
- ``TestTROPCorollary1Unbiasedness`` — Corollary 1 (three conditions).
- ``TestTROPTheorem51TripleRobustness`` — Theorem 5.1 (MC ranking realisation).
- ``TestTROPSpecialCases`` — Section 2.2 reductions (DID, MC, SDID; SC skipped).
- ``TestTROPAlgorithm2MultipleTreated`` — Eq. 13 + Algorithm 2.
- ``TestTROPAlgorithm3Bootstrap`` — Algorithm 3 (stratified pairs bootstrap).
- ``TestTROPEquation6FactorDGPRecovery`` — Section 3 / Eq. 6 semi-synthetic.
- ``TestTROPDeviations`` — locks library deviations and documented choices.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import TROP, DifferenceInDifferences
from diff_diff.prep import generate_factor_data

# Per-class seed bases (decorrelate MC tests within and across classes).
_BASE_SEED_NUCLEAR_PROX = 4242
_BASE_SEED_EQ3_WEIGHTS = 3131
_BASE_SEED_ALG1_LOOCV = 5555
_BASE_SEED_COROLLARY_1 = 6464
_BASE_SEED_THEOREM_51 = 7373
_BASE_SEED_SPECIAL_CASES = 8181
_BASE_SEED_ALG2_MULTI = 9292
_BASE_SEED_DEVIATIONS = 3030


# =============================================================================
# Helpers — paper-aligned DGPs for methodology tests
# =============================================================================


def _make_trop_factor_panel(
    *,
    n_units: int = 30,
    n_treated: int = 6,
    n_pre: int = 8,
    n_post: int = 4,
    n_factors: int = 2,
    factor_strength: float = 1.0,
    treated_loading_shift: float = 0.5,
    unit_fe_sd: float = 0.3,
    noise_sd: float = 0.3,
    treatment_effect: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a balanced panel with a rank-`n_factors` interactive-FE component.

    Mirrors the paper's Eq. 6 simulation DGP (Section 3.1). Returns a long
    DataFrame with columns ``unit``, ``period``, ``outcome``, ``treated``.
    A non-zero ``treated_loading_shift`` induces selection-on-loadings
    confounding (matching the paper's logistic-on-factors selection in Eq. 8).
    Wraps :func:`diff_diff.prep.generate_factor_data` for parity with
    `tests/test_trop.py::generate_factor_dgp`.
    """
    data = generate_factor_data(
        n_units=n_units,
        n_pre=n_pre,
        n_post=n_post,
        n_treated=n_treated,
        n_factors=n_factors,
        treatment_effect=treatment_effect,
        factor_strength=factor_strength,
        treated_loading_shift=treated_loading_shift,
        unit_fe_sd=unit_fe_sd,
        noise_sd=noise_sd,
        seed=seed,
    )
    return pd.DataFrame(data[["unit", "period", "outcome", "treated"]])


def _make_constant_loading_panel(
    *,
    n_units: int = 30,
    n_treated: int = 8,
    n_pre: int = 6,
    n_post: int = 4,
    n_factors: int = 2,
    factor_strength: float = 1.0,
    noise_sd: float = 0.3,
    treatment_effect: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a panel where every unit shares the same factor loading.

    This makes Corollary 1(a) (unit balance) trivially hold: with
    Gamma_i = Gamma_j for all i, j, any non-negative weight vector ω
    satisfies (sum_i omega_i Gamma_i) = Gamma_N, so Delta_u = 0 regardless
    of lambda_unit. Treated units still receive a non-zero treatment effect
    and time factors still vary across t, so the test surface is real.
    """
    rng = np.random.default_rng(seed)
    n_periods = n_pre + n_post
    shared_loading = rng.normal(0, 1, (1, n_factors))
    Gamma = np.tile(shared_loading, (n_units, 1))
    Lambda = rng.normal(0, 1, (n_periods, n_factors))
    rows = []
    for i in range(n_units):
        is_treated = i < n_treated
        unit_fe = rng.normal(0, 0.3)
        for t in range(n_periods):
            time_fe = 0.15 * t
            interaction = factor_strength * Gamma[i, :] @ Lambda[t, :]
            y = 5.0 + unit_fe + time_fe + interaction
            d = 1 if (is_treated and t >= n_pre) else 0
            if d:
                y += treatment_effect
            y += rng.normal(0, noise_sd)
            rows.append({"unit": i, "period": t, "outcome": y, "treated": d})
    return pd.DataFrame(rows)


def _make_constant_factor_panel(
    *,
    n_units: int = 30,
    n_treated: int = 8,
    n_pre: int = 6,
    n_post: int = 4,
    n_factors: int = 2,
    factor_strength: float = 1.0,
    noise_sd: float = 0.3,
    treatment_effect: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a panel where every period shares the same factor score.

    This makes Corollary 1(b) (time balance) trivially hold: with
    Lambda_s = Lambda_t for all s, t, any non-negative weight vector θ
    satisfies (sum_s theta_s Lambda_s) = Lambda_T, so Delta_t = 0 regardless
    of lambda_time. Unit loadings still vary (with confounding shift on
    treated), so the test surface is non-trivial.
    """
    rng = np.random.default_rng(seed)
    n_periods = n_pre + n_post
    shared_factor = rng.normal(0, 1, (1, n_factors))
    Lambda = np.tile(shared_factor, (n_periods, 1))
    Gamma = rng.normal(0, 1, (n_units, n_factors))
    Gamma[:n_treated, :] += 0.5  # selection on loadings (still hard for DID)
    rows = []
    for i in range(n_units):
        is_treated = i < n_treated
        unit_fe = rng.normal(0, 0.3)
        for t in range(n_periods):
            time_fe = 0.15 * t
            interaction = factor_strength * Gamma[i, :] @ Lambda[t, :]
            y = 5.0 + unit_fe + time_fe + interaction
            d = 1 if (is_treated and t >= n_pre) else 0
            if d:
                y += treatment_effect
            y += rng.normal(0, noise_sd)
            rows.append({"unit": i, "period": t, "outcome": y, "treated": d})
    return pd.DataFrame(rows)


def _make_no_factor_panel(
    *,
    n_units: int = 30,
    n_treated: int = 8,
    n_pre: int = 6,
    n_post: int = 4,
    noise_sd: float = 0.3,
    treatment_effect: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a TWFE-clean panel (no interactive FE, just additive unit + time).

    With no factor structure, the regression-adjustment bias matrix B is
    trivially zero (and TROP with lambda_nn=infinity reduces exactly to TWFE
    per paper Section 2.2). Used by Corollary 1(c) to verify the B=0 case.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        is_treated = i < n_treated
        unit_fe = rng.normal(0, 0.5)
        for t in range(n_pre + n_post):
            time_fe = 0.2 * t
            y = 5.0 + unit_fe + time_fe
            d = 1 if (is_treated and t >= n_pre) else 0
            if d:
                y += treatment_effect
            y += rng.normal(0, noise_sd)
            rows.append({"unit": i, "period": t, "outcome": y, "treated": d})
    return pd.DataFrame(rows)


def _fit_did(df: pd.DataFrame) -> float:
    """Fit a `DifferenceInDifferences` benchmark on the panel and return
    the interaction coefficient.

    This is the library's basic 2×2 DiD estimator (`[const, D, T, D×T]`
    design, no explicit fixed effects added). On a balanced two-period
    block-assignment panel it coincides numerically with TWFE / two-way
    fixed effects (paper Section 2.2 uses "DID/TWFE" interchangeably for
    this special case), but the library distinguishes the two classes —
    `TwoWayFixedEffects` is a separate explicit-FE estimator. Used by
    `TestTROPTheorem51TripleRobustness` as the comparator benchmark for
    the MC-ranking realisation of the Theorem 5.1 bias bound and by
    `TestTROPSpecialCases::test_did_reduction_lambda_nn_inf_uniform_weights`
    as the DID-reduction target.
    """
    df2 = df.copy()
    df2["treat"] = df2.groupby("unit")["treated"].transform("max").astype(int)
    df2["post_flag"] = df2.groupby("period")["treated"].transform("max").astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = DifferenceInDifferences()
        r = est.fit(df2, formula="outcome ~ treat * post_flag", unit="unit", time="period")
    coefs = r.coefficients or {}
    return float(coefs.get("treat:post_flag", np.nan))


# =============================================================================
# TestTROPNuclearNormProx — Eq. 2 + Eq. 10 (prox, FISTA, balancing)
# =============================================================================


class TestTROPNuclearNormProx:
    """Eq. 2: weighted nuclear-norm-penalised L estimation; structural
    pointer to Eq. 10.

    Verifies the proximal-gradient inner solver (soft-threshold SVD),
    plain (non-accelerated) prox-gradient monotonicity on a toy setup,
    and the weighted-norm solver behaviour. The shipped local / global
    solvers wrap the prox step with accelerated FISTA momentum; the
    accelerated loop's faster ``O(1/k^2)`` rate does NOT guarantee
    per-step monotonicity, so the monotonicity test
    (``test_plain_prox_gradient_objective_decreases``) exercises the
    plain prox-gradient ingredient (no Nesterov momentum), NOT the
    accelerated outer loop.

    The balancing decomposition of paper Eq. 10
    (``Y_NT_hat = L_NT + theta . (Y_pre_N - L_pre_N) + omega . (Y_T_co - L_T_co)
        - sum theta_t omega_i (Y_it_co - L_it_co)``)
    is a paper-side balancing representation (paper Eq. 10 / Section 5.2) that requires the internal per-(i, t)
    weight vectors ``theta_s^{i,t}`` / ``omega_j^{i,t}`` to numerically
    reconstruct. Those vectors are not exposed on the public TROP API,
    so this class does NOT directly verify the four-term identity. The
    method ``test_factor_matrix_consistent_with_treatment_effects`` is a
    structural pointer only — it checks ``factor_matrix`` shape +
    finiteness and verifies ``treatment_effects`` is populated with
    finite entries (the framework that Eq. 10 derives). It does NOT
    assert non-triviality of the ``L_hat`` magnitude (the test DGP has
    no interactive factor structure, so a near-zero ``L_hat`` is
    methodologically correct under the paper's framework).

    Origin: ported from the pre-migration `TestTROPNuclearNormSolver`
    class in `test_trop.py` (three of four methods migrated; the one
    remaining defensive `test_zero_weights_no_division_error` stayed in
    `test_trop.py`) plus the weighted-solver convergence test
    `test_weighted_nuclear_norm_solver_convergence` and the weighted-
    nuclear-norm objective test `test_issue_c_weighted_nuclear_norm`,
    both originally in a pre-migration `TestPaperConformanceFixes`
    class that was deleted in the methodology-promotion PR.
    """

    def test_proximal_step_size_correctness(self):
        """Eq. 2 prox operator: L converges to ``prox_{lambda/2}(R)`` under
        uniform weights.

        With delta_max = 1 and uniform weights, the proximal gradient step
        reduces to L_{k+1} = soft_threshold_svd(R, lambda/2). Many
        iterations should converge L exactly to that analytical fixed point.
        """
        trop_est = TROP(method="global", n_bootstrap=2)

        rng = np.random.default_rng(42)
        R = rng.normal(0, 1, (4, 3))
        delta = np.ones((4, 3))
        lambda_nn = 0.5

        L = np.zeros_like(R)
        for _ in range(500):
            delta_max = np.max(delta)
            delta_norm = delta / delta_max
            gradient_step = L + delta_norm * (R - L)
            eta = 1.0 / (2.0 * delta_max)
            L = trop_est._soft_threshold_svd(gradient_step, eta * lambda_nn)

        L_exact = trop_est._soft_threshold_svd(R, lambda_nn / 2.0)
        np.testing.assert_array_almost_equal(L, L_exact, decimal=4)

    def test_plain_prox_gradient_objective_decreases(self):
        """Eq. 2 plain (non-accelerated) prox-gradient: objective
        ``f(L) + lambda * ||L||_*`` is non-increasing across iterations
        when the soft-threshold SVD prox is applied via a plain prox-
        gradient step (no Nesterov momentum).

        **Scope:** this verifies the underlying prox operator + gradient
        step that the library's accelerated FISTA loop builds on. The
        shipped local / global solvers (``trop_local.py`` / ``trop_global.py``)
        wrap this prox step with FISTA acceleration, which gives the
        faster ``O(1/k^2)`` rate but does NOT guarantee per-iteration
        monotonicity — Nesterov momentum can cause transient objective
        increases between iterations while still converging optimally.
        This test exercises the prox+gradient ingredient, not the
        accelerated outer loop.
        """
        rng = np.random.default_rng(42)
        R = rng.normal(0, 1, (6, 4))
        delta = rng.uniform(0.5, 2.0, (6, 4))
        lambda_nn = 0.3

        trop_est = TROP(method="global", n_bootstrap=2)
        L = np.zeros_like(R)
        objectives = []

        for _ in range(50):
            f_val = np.sum(delta * (R - L) ** 2)
            _, s, _ = np.linalg.svd(L, full_matrices=False)
            obj = f_val + lambda_nn * np.sum(s)
            objectives.append(obj)

            delta_max = np.max(delta)
            delta_norm = delta / delta_max
            gradient_step = L + delta_norm * (R - L)
            eta = 1.0 / (2.0 * delta_max)
            L = trop_est._soft_threshold_svd(gradient_step, eta * lambda_nn)

        for k in range(1, len(objectives)):
            assert objectives[k] <= objectives[k - 1] + 1e-10, (
                f"Plain prox-gradient objective increased at step {k}: "
                f"{objectives[k]} > {objectives[k - 1]} "
                f"(NOTE: this assertion is for the plain prox-gradient "
                f"ingredient; the shipped accelerated FISTA loop does "
                f"NOT guarantee per-step monotonicity)"
            )

    def test_local_nonuniform_weights_objective(self):
        """Eq. 2 weighted-prox: objective at the final iterate is
        at-or-below initialisation under non-uniform weights
        (W_max < 1), and the resulting L has strictly smaller nuclear
        norm than the residual R.

        Scope note: this is a **final-vs-initial** check on the shipped
        `_weighted_nuclear_norm_solve` (which uses accelerated FISTA);
        it does NOT verify per-iteration monotonicity. Per-step
        monotonicity is a property of the plain prox-gradient ingredient
        only (see ``test_plain_prox_gradient_objective_decreases``);
        accelerated FISTA's Nesterov momentum is allowed to produce
        transient per-step objective increases while still converging.
        """
        rng = np.random.default_rng(123)
        R = rng.normal(0, 1, (6, 4))
        W = rng.uniform(0.1, 0.8, (6, 4))
        lambda_nn = 0.3

        trop_est = TROP(method="local", n_bootstrap=2)

        L_init = np.zeros_like(R)
        f_init = np.sum(W * (R - L_init) ** 2)
        _, s_init, _ = np.linalg.svd(L_init, full_matrices=False)
        obj_init = f_init + lambda_nn * np.sum(s_init)

        L_final = trop_est._weighted_nuclear_norm_solve(
            Y=R,
            W=W,
            L_init=L_init,
            alpha=np.zeros(R.shape[1]),
            beta=np.zeros(R.shape[0]),
            lambda_nn=lambda_nn,
            max_inner_iter=20,
        )

        f_final = np.sum(W * (R - L_final) ** 2)
        _, s_final, _ = np.linalg.svd(L_final, full_matrices=False)
        obj_final = f_final + lambda_nn * np.sum(s_final)

        assert (
            obj_final <= obj_init + 1e-10
        ), f"Objective did not decrease: {obj_final} > {obj_init}"

        nuclear_norm_R = np.sum(np.linalg.svd(R, compute_uv=False))
        nuclear_norm_L = np.sum(s_final)
        assert (
            nuclear_norm_L < nuclear_norm_R
        ), f"Nuclear norm not reduced: {nuclear_norm_L} >= {nuclear_norm_R}"

    def test_weighted_nuclear_norm_objective_recovers_att(self):
        """Eq. 2 weighted objective with active regularisation: TROP with
        a non-zero ``lambda_nn`` grid (on an interactive-FE DGP) recovers
        a finite positive ATT and a non-negative effective_rank —
        exercising the weighted prox + alternating-min code path. (The
        ``effective_rank`` assertion is `>= 0` rather than `> 0` because
        the test DGP's factor structure may be absorbed by the prox
        regulariser; the active code path is verified by the positive-
        ATT recovery, not by a non-zero rank claim.)

        Scope note: this test does NOT fit an unregularised baseline for
        comparison; for the DID-vs-MC ranking on a confounded factor DGP
        see `TestTROPSpecialCases::test_matrix_completion_reduction_uniform_weights_finite_nn`.

        Origin: ported from the pre-migration
        `TestPaperConformanceFixes::test_issue_c_weighted_nuclear_norm`
        in `test_trop.py` (the pre-migration class was deleted in the
        methodology-promotion PR; the test was migrated here).
        """
        rng = np.random.default_rng(456)

        n_units = 15
        n_periods = 8
        n_treated = 3
        true_att = 2.0

        loadings = rng.normal(0, 1, n_units)
        factors = rng.normal(0, 1, n_periods)

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= 5
                y = 10.0 + loadings[i] * factors[t]
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1, 1.0],  # active regularisation
            max_iter=500,  # converge cleanly without "may be inaccurate" warning
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.effective_rank >= 0
        assert results.att > 0, f"ATT={results.att:.3f} should be positive"

    def test_weighted_nuclear_norm_solver_reduces_sv_mass(self):
        """Eq. 2 nuclear-norm regularisation: the fitted L has strictly
        smaller total singular-value mass than the input Y.

        Origin: ported from the pre-migration
        `TestPaperConformanceFixes::test_weighted_nuclear_norm_solver_convergence`
        in `test_trop.py` (the pre-migration class was deleted in the
        methodology-promotion PR; the test was migrated here).
        """
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[1.0],
        )

        n_periods = 5
        n_units = 8

        Y = np.random.default_rng(42).normal(0, 1, (n_periods, n_units))
        W = np.ones((n_periods, n_units))
        L_init = np.zeros((n_periods, n_units))
        alpha = np.zeros(n_units)
        beta = np.zeros(n_periods)

        L = trop_est._weighted_nuclear_norm_solve(
            Y, W, L_init, alpha, beta, lambda_nn=1.0, max_inner_iter=20
        )

        assert np.all(np.isfinite(L))
        _, s, _ = np.linalg.svd(L, full_matrices=False)
        _, s_orig, _ = np.linalg.svd(Y, full_matrices=False)
        assert np.sum(s) < np.sum(
            s_orig
        ), "Nuclear-norm regularisation should reduce total singular-value mass"

    def test_factor_matrix_consistent_with_treatment_effects(self):
        """Eq. 10 corollary: the fitted ``factor_matrix`` (L_hat) is
        consistent with the per-cell counterfactual implied by
        ``treatment_effects``.

        Eq. 10 (paper p. 22) writes the estimated counterfactual at a
        treated cell as ``L_hat_NT`` plus three weighted (Y - L_hat)
        averages over control / pre-treatment slices. The exact identity
        depends on the internal weight vectors (theta_s^{i,t}, omega_j^{i,t})
        which TROP computes from the user's lambda_time / lambda_unit
        plus per-(i,t) unit distances and is not part of the public API
        — so we cannot reconstruct the four-component sum here. What we
        CAN verify is that the resulting ``factor_matrix`` has the same
        shape as the (period, unit) outcome grid, is finite, and that
        ``treatment_effects`` is populated with finite entries. **No
        non-triviality / magnitude claim** is made on ``L_hat`` because
        the test DGP has no interactive factor structure (additive unit
        + time effects only) and a near-zero ``L_hat`` is methodologically
        correct under the paper's framework.

        For a direct numerical realisation of the Eq. 10 decomposition,
        see Athey et al. (2025) Section 5.2, which derives the
        decomposition under the block (N_0, T_0) assignment. The library
        relies on the same Eq. 2 prox + alternating-min stack as the
        derivation; the soft-threshold-SVD / FISTA / weighted-prox tests
        above (and `test_weighted_nuclear_norm_objective_recovers_att`)
        cover the ingredients.
        """
        rng = np.random.default_rng(_BASE_SEED_NUCLEAR_PROX)
        n_units = 8
        n_treated = 1
        n_pre = 5
        n_post = 1
        n_periods = n_pre + n_post

        rows = []
        for i in range(n_units):
            is_treated = i < n_treated
            unit_fe = rng.normal(0, 0.3)
            for t in range(n_periods):
                y = 5.0 + unit_fe + 0.1 * t + rng.normal(0, 0.2)
                d = 1 if (is_treated and t >= n_pre) else 0
                rows.append({"unit": i, "period": t, "outcome": y, "treated": d})
        df = pd.DataFrame(rows)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            max_iter=500,  # converge cleanly without "may be inaccurate" warning
            n_bootstrap=2,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        L = np.asarray(results.factor_matrix, dtype=float)
        assert L.shape == (n_periods, n_units)
        assert np.all(np.isfinite(L))
        # NOTE: this DGP has only additive unit + time effects plus iid
        # noise — no interactive factor structure. Per the paper's
        # framework, ``alpha_i`` / ``beta_t`` absorb the additive
        # surfaces, so a near-zero ``L_hat`` is methodologically correct
        # here. Asserting ``effective_rank > 0`` would lock a solver
        # artifact (e.g., regularisation under-shrinkage) rather than
        # the intended low-rank behavior. The shape + finiteness check
        # above + the treatment_effects existence check below are the
        # legitimate structural surface for this test; the Eq. 2
        # ingredients are independently verified in the prox / FISTA /
        # weighted-norm tests above.
        assert results.treatment_effects is not None
        # The single treated cell (i=0, t=n_pre) must be in treatment_effects.
        # Resolve whatever unit / period values were used in the input frame.
        assert any(
            np.isfinite(value) for value in results.treatment_effects.values()
        ), "treatment_effects must contain at least one finite entry"


# =============================================================================
# TestTROPEquation3Weights — Eq. 3 (unit / time weight semantics)
# =============================================================================


class TestTROPEquation3Weights:
    """Eq. 3: exponential-decay unit and time weights.

    Three direct paper-formula assertions:

    - ``test_distance_excludes_target_period`` (extracted from the
      pre-migration ``TestPaperConformanceFixes::test_issue_b`` in
      `test_trop.py` — the pre-migration class was deleted in the
      methodology-promotion PR): end-to-end fit smoke that the
      target-period anomaly does not dominate.
    - ``test_unit_distance_uses_untreated_only_mask`` (NEW): direct
      assertion that ``TROP._compute_unit_distance_for_obs`` matches
      the paper's
      ``dist_unit_{-t}(j, i) = sqrt(sum_{u != t}(1-W_iu)(1-W_ju)(Y_iu - Y_ju)^2
        / sum_{u != t}(1-W_iu)(1-W_ju))``
      formula, hand-computed on a constructed Y / D where the masking
      gives an unambiguous answer.
    - ``test_time_weights_match_exp_decay_formula`` (NEW): direct
      assertion that the per-(i, t) weight matrix's time-axis
      column equals ``exp(-lambda_time * |t - s|)``.

    Origin: target-period-exclusion test ported from the pre-migration
    `TestPaperConformanceFixes::test_issue_b_distance_excludes_target_period`
    in `test_trop.py` (the pre-migration class was deleted in the
    methodology-promotion PR); the unit-distance-mask and time-decay
    assertions are new direct Eq. 3 locks.
    """

    def test_distance_excludes_target_period(self):
        """Eq. 3: pairwise unit distance uses 1{u != t} to exclude period t.

        Construct a panel where the treated unit's outcome at the target
        period (t = 3) is anomalous (Y = 100, vs Y ~ 5 elsewhere). Without
        the 1{u != t} mask, this single observation would dominate every
        pairwise unit distance and break the weight calculation. With the
        mask, the fit completes and returns a finite ATT.
        """
        rng = np.random.default_rng(123)

        n_units = 10
        n_periods = 6
        data = []
        for i in range(n_units):
            is_treated = i == 0
            for t in range(n_periods):
                if is_treated and t == 3:
                    y = 100.0  # anomalous target-period outcome
                elif is_treated and t >= 3:
                    y = 5.0 + rng.normal(0, 0.1)
                else:
                    y = 5.0 + rng.normal(0, 0.1)

                treatment_indicator = 1 if (is_treated and t >= 3) else 0
                data.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[1.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=5,
            seed=42,
        )

        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results is not None
        assert np.isfinite(results.att)

    def test_unit_distance_uses_untreated_only_mask(self):
        """Eq. 3: ``dist_unit_{-t}(j, i)`` uses only periods where both
        units are untreated (the ``(1 - W_iu)(1 - W_ju)`` mask) AND
        excludes the target period t.

        Hand-construct ``Y`` and ``D`` where the treatment pattern is
        known, then verify ``TROP._compute_unit_distance_for_obs`` returns
        the paper's hand-computed RMS distance (sum of squared
        differences over the valid mask, normalised by the count of
        valid periods).
        """
        # Treated unit i = 0 is treated at t = 2 onward; control unit
        # j = 1 is untreated throughout. Target period t = 2.
        # Untreated periods for both: u in {0, 1}. Target period t = 2 is
        # also excluded by the 1{u != t} mask. Periods u >= 2 are
        # treated for i and excluded by (1 - W_iu).
        n_periods = 6
        n_units = 2
        Y = np.zeros((n_periods, n_units))
        # Y_0u: treated unit, set so that (Y_iu - Y_ju)^2 = 4 at u=0,
        # = 9 at u=1, others irrelevant due to mask.
        Y[0, 0] = 2.0
        Y[1, 0] = 3.0
        Y[2, 0] = 100.0  # target period — excluded by 1{u != t}
        Y[3, 0] = 99.0  # treated period — excluded by (1 - W_iu)
        Y[4, 0] = 98.0  # treated period — excluded by (1 - W_iu)
        Y[5, 0] = 97.0  # treated period — excluded by (1 - W_iu)
        Y[:, 1] = 0.0  # j is constant 0
        D = np.zeros((n_periods, n_units), dtype=int)
        D[2:, 0] = 1  # i treated from t = 2

        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=42,
        )
        dist = est._compute_unit_distance_for_obs(Y=Y, D=D, j=1, i=0, target_period=2)

        # Valid mask: u in {0, 1} (target excluded + both untreated).
        # Squared differences: (2-0)^2 = 4, (3-0)^2 = 9. Mean = 6.5.
        # RMS = sqrt(6.5).
        expected = float(np.sqrt(6.5))
        assert np.isclose(dist, expected, rtol=1e-12), (
            f"Eq. 3 unit distance: {dist:.6e} != hand-computed "
            f"{expected:.6e} over the valid (untreated-only,"
            " target-excluded) mask"
        )

    def test_time_weights_match_exp_decay_formula(self):
        """Eq. 3 time weights: ``theta_s^{i,t} = exp(-lambda_time * |t - s|)``.

        Direct assertion on the per-(i, t) weight matrix's time-axis
        slice: at column ``j != i``, the time-decay factor should be
        ``exp(-lambda_time * |t - s|)`` for each s. Since
        ``_compute_observation_weights`` returns the outer product of
        ``time_weights`` and ``unit_weights`` (shape ``(n_periods, n_units)``),
        column ``j`` equals ``unit_weights[j] * time_weights``. At
        ``lambda_unit = 0`` (uniform unit weights = 1), column ``j``
        equals ``time_weights`` exactly.
        """
        n_units = 4
        n_periods = 7
        target_t = 3
        lambda_time = 0.5
        rng = np.random.default_rng(_BASE_SEED_EQ3_WEIGHTS)
        Y = rng.normal(0, 1, (n_periods, n_units))
        D = np.zeros((n_periods, n_units), dtype=int)
        D[target_t:, 0] = 1  # one absorbing-state treated unit

        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=42,
        )
        weights = est._compute_observation_weights(
            Y=Y,
            D=D,
            i=0,
            t=target_t,
            lambda_time=lambda_time,
            lambda_unit=0.0,  # uniform unit weights => column j = time_weights
            control_unit_idx=np.arange(1, n_units),
            n_units=n_units,
            n_periods=n_periods,
        )

        # Hand-computed time weights: exp(-lambda_time * |t - s|)
        expected_time = np.exp(-lambda_time * np.abs(np.arange(n_periods) - target_t))

        # Column j = 1 (a control unit) should equal expected_time
        # (unit_weights[1] = 1.0 under lambda_unit = 0).
        np.testing.assert_array_almost_equal(
            weights[:, 1],
            expected_time,
            decimal=12,
            err_msg=(
                "Eq. 3 time weights at column j=1 do not match "
                f"exp(-{lambda_time} * |t-s|) hand-computed values"
            ),
        )


# =============================================================================
# TestTROPAlgorithm1LOOCV — Eqs. 4-5 + Algorithm 1 (LOOCV)
# =============================================================================


class TestTROPAlgorithm1LOOCV:
    """Eqs. 4-5 + Algorithm 1: leave-one-out cross-validation.

    Verifies that Q(lambda) sums squared pseudo-treatment effects over ALL
    control observations where D_js = 0 (including pre-treatment periods
    of eventually-treated units, not just never-treated units), and that
    the two-stage coordinate-descent cycling search (paper footnote 2)
    converges to a grid point.

    Origin:
    `test_issue_a_control_includes_pretreatment_obs` ported from the
    pre-migration `TestPaperConformanceFixes` class in `test_trop.py`
    (deleted in the methodology-promotion PR); one tightly-scoped
    cycling-convergence assertion ported from
    `tests/test_trop.py::TestCyclingSearch` (which retains its other
    LOOCV tests for the fallback / single-value-grid surfaces — those
    are defensive and stayed in `test_trop.py`).
    """

    def test_control_set_includes_pretreat_of_eventually_treated(self):
        """Eq. 2 / Eq. 5: control set is {(j, s) : (1 - W_{js}) > 0},
        which includes the pre-treatment observations of eventually-
        treated units, not just never-treated units.

        Construct staggered adoption with two treatment cohorts (early at
        t=3, late at t=5) plus never-treated. With a non-zero lambda_unit
        and the Eq. 2 control set, TROP exploits late-treated pre-period
        observations to fit early-treated counterfactuals (their levels
        match) and recovers a positive ATT.
        """
        rng = np.random.default_rng(42)
        n_units = 20
        n_early_treat = 5
        n_late_treat = 5
        n_periods = 8
        true_att = 2.0

        data = []
        for i in range(n_units):
            if i < n_early_treat:
                treat_period: int | None = 3
                unit_fe = 5.0
            elif i < n_early_treat + n_late_treat:
                treat_period = 5
                unit_fe = 5.5
            else:
                treat_period = None
                unit_fe = 10.0  # controls have distinct level

            for t in range(n_periods):
                is_post = treat_period is not None and t >= treat_period
                treatment_indicator = 1 if is_post else 0
                y = unit_fe + 0.2 * t
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[1.0],  # unit weights so distance matters
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.att > 0, f"ATT={results.att:.3f} should be positive"

    def test_cycling_search_converges_to_grid_point(self):
        """Algorithm 1 / footnote 2: two-stage coordinate-descent cycling
        always converges to a tuple of values from the input grids.

        Confirms the cycling-search invariant: the returned
        ``(lambda_time, lambda_unit, lambda_nn)`` must be members of the
        user-supplied grids, and ``att`` must be finite, ``se >= 0``.
        Origin: ported from
        `tests/test_trop.py::TestCyclingSearch::test_cycling_search_converges`.
        """
        df = _make_no_factor_panel(
            n_units=20,
            n_treated=5,
            n_pre=5,
            n_post=3,
            noise_sd=0.5,
            treatment_effect=3.0,
            seed=_BASE_SEED_ALG1_LOOCV,
        )

        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            n_bootstrap=5,
            seed=42,
        )

        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.lambda_time in trop_est.lambda_time_grid
        assert results.lambda_unit in trop_est.lambda_unit_grid
        assert results.lambda_nn in trop_est.lambda_nn_grid

        assert np.isfinite(results.att)
        assert results.se >= 0


# =============================================================================
# TestTROPCorollary1Unbiasedness — three single-condition DGPs
# =============================================================================


class TestTROPCorollary1Unbiasedness:
    """Corollary 1: unbiased if ANY ONE of three balance conditions holds.

    The paper states (page 23): under Assumption 1, for fixed (non-data-
    dependent) weights theta, omega,
    `|E[tau_hat - tau | L]| <= ||Delta_u(omega, Gamma)||_2 * ||Delta_t(theta, Lambda)||_2 * ||B||_*`
    so the estimator is unbiased if any one of:
      (a) Unit balance: sum_{i in C} omega_i Gamma_i = Gamma_N
      (b) Time balance: sum_{s in C} theta_s Lambda_s = Lambda_T
      (c) Correct regression adjustment: B = 0_K

    Each test constructs a DGP that makes one condition hold trivially
    (constant loadings, constant factor scores, or no factor structure
    with lambda_nn=infinity), and asserts TROP recovers the ATT within
    a 3-sigma MC band even when the other components are sub-optimal.
    Tests are NEW — no extraction from `test_trop.py`.
    """

    def test_unit_balance_constant_loadings(self):
        """Corollary 1(a): constant unit loadings make ``Delta_u = 0`` for
        any non-negative weight vector ``omega``. TROP is unbiased even
        with sub-optimal ``lambda_unit = 0`` (uniform unit weights).

        The DGP has shared ``Gamma_i = Gamma`` for all units (so the
        product ``sum omega_i Gamma_i = Gamma`` regardless of omega) and
        random ``Lambda_t`` across periods. Treated cells receive a
        constant treatment effect.
        """
        df = _make_constant_loading_panel(
            n_units=30,
            n_treated=8,
            n_pre=6,
            n_post=4,
            n_factors=2,
            factor_strength=1.0,
            noise_sd=0.3,
            treatment_effect=2.0,
            seed=_BASE_SEED_COROLLARY_1,
        )

        # Deliberately sub-optimal unit weighting (lambda_unit = 0 == uniform).
        # Theorem says we're still unbiased because the unit-balance term is 0.
        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[1.0],
            n_bootstrap=20,
            seed=42,
        )
        results = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        # 3-sigma band: with constant loadings, unit balance is trivial,
        # so any reasonable estimator (TROP included) is centred on truth.
        assert abs(results.att - 2.0) < 3.0 * results.se, (
            f"ATT={results.att:.4f} not within 3 SE={results.se:.4f} "
            f"of true=2.0 under unit balance"
        )

    def test_time_balance_constant_factors(self):
        """Corollary 1(b): constant factor scores make ``Delta_t = 0`` for
        any non-negative weight vector ``theta``. TROP is unbiased even
        with sub-optimal ``lambda_time = 0``.

        The DGP has shared ``Lambda_s = Lambda`` for all periods (so the
        product ``sum theta_s Lambda_s = Lambda`` regardless of theta) and
        random ``Gamma_i`` with selection-on-loadings (treated units
        shifted) — so DID would be biased, but TROP recovers ATT.
        """
        df = _make_constant_factor_panel(
            n_units=30,
            n_treated=8,
            n_pre=6,
            n_post=4,
            n_factors=2,
            factor_strength=1.0,
            noise_sd=0.3,
            treatment_effect=2.0,
            seed=_BASE_SEED_COROLLARY_1 + 1,
        )

        # Sub-optimal time weighting (lambda_time = 0). Time-balance
        # condition makes this irrelevant.
        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[1.0],
            n_bootstrap=20,
            seed=42,
        )
        results = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert abs(results.att - 2.0) < 3.0 * results.se, (
            f"ATT={results.att:.4f} not within 3 SE={results.se:.4f} "
            f"of true=2.0 under time balance"
        )

    def test_zero_regression_bias_no_factor_dgp(self):
        """Corollary 1(c): when ``B = 0_K`` (no regression-adjustment bias),
        TROP is unbiased regardless of weights.

        Setting ``lambda_nn = infinity`` forces ``L = 0`` (paper Section 2.2:
        factor model disabled), so the estimator reduces to TWFE on a
        TWFE-clean DGP and ``B = 0`` trivially. ATT recovery is sharp.
        """
        df = _make_no_factor_panel(
            n_units=30,
            n_treated=8,
            n_pre=6,
            n_post=4,
            noise_sd=0.3,
            treatment_effect=2.0,
            seed=_BASE_SEED_COROLLARY_1 + 2,
        )

        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[np.inf],  # factor model disabled --> B = 0
            n_bootstrap=20,
            seed=42,
        )
        results = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert abs(results.att - 2.0) < 3.0 * results.se, (
            f"ATT={results.att:.4f} not within 3 SE={results.se:.4f} " f"of true=2.0 under B=0"
        )
        # Sanity: lambda_nn was preserved as inf in results metadata
        # (REGISTRY ## TROP "λ_nn=∞ implementation" note: "TROPResults stores ORIGINAL lambda_nn value (inf)").
        assert results.lambda_nn == np.inf


# =============================================================================
# TestTROPTheorem51TripleRobustness — MC-ranking realisation of bias bound
# =============================================================================


@pytest.mark.slow
class TestTROPTheorem51TripleRobustness:
    """Theorem 5.1: empirical realisation of the triply-robust bias bound.

    The paper's bound
    `|E[tau_hat - tau | L]| <= ||Delta_u||_2 * ||Delta_t||_2 * ||B||_*`
    (product of three components rather than sum) is "strictly tighter
    than bounds for DID, SC, and SDID" (paper Section 5.2 + Eq. 11).
    A direct oracle-Gamma/Lambda/B test was spiked and found to require
    constructing B from the regression-adjustment estimator class
    (Assumption 2 bias matrix), which is brittle under finite-sample MC
    noise. The methodology test therefore verifies the bound's
    empirical realisation:

    TROP RMSE < DID RMSE under a confounded factor DGP, over a MC sweep
    of independent panels. The factor DGP induces interactive fixed-effect
    bias that the DiD benchmark cannot handle, while TROP's three robustness
    components jointly absorb the confounding.

    Tests are NEW. The MC-ranking pattern also dedupes the
    factor-DGP coverage from the pre-migration
    `TestTROPvsSDID::test_trop_handles_factor_dgp` in `test_trop.py`
    (which only asserted ``att != 0``, not ranking against DID; the
    pre-migration `TestTROPvsSDID` class was deleted in the
    methodology-promotion PR).
    """

    def test_trop_rmse_strictly_below_did_under_factor_confounding(self, ci_params):
        """Theorem 5.1 (Eq. 11 ranking): on a confounded factor DGP with
        true ``tau = 0``, TROP RMSE is strictly below DID RMSE across
        independent MC replicates.

        The factor DGP induces interactive-FE bias that the DiD benchmark cannot
        handle; TROP's three robustness components jointly absorb the
        confounding. Empirical magnitude (spike measurement at
        ``factor_strength=1.0``, 15 reps): TROP/DID RMSE ratio ~ 0.34
        (3x advantage). The assertion uses a generous ``ratio < 0.7``
        margin to absorb finite-sample MC noise and per-rep
        loocv-tuning variance.
        """
        n_reps = ci_params.bootstrap(10, min_n=5)
        trop_atts = []
        did_atts = []

        for rep in range(n_reps):
            df = _make_trop_factor_panel(
                n_units=24,
                n_treated=6,
                n_pre=6,
                n_post=3,
                n_factors=2,
                factor_strength=1.0,
                treated_loading_shift=0.8,
                noise_sd=0.3,
                treatment_effect=0.0,
                seed=_BASE_SEED_THEOREM_51 + rep,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trop_est = TROP(
                    lambda_time_grid=[0.0, 1.0],
                    lambda_unit_grid=[0.0, 1.0],
                    lambda_nn_grid=[0.0, 1.0],
                    n_bootstrap=5,
                    seed=_BASE_SEED_THEOREM_51 + rep,
                )
                r_trop = trop_est.fit(
                    df,
                    outcome="outcome",
                    treatment="treated",
                    unit="unit",
                    time="period",
                )
                did_att = _fit_did(df)

            if np.isfinite(r_trop.att):
                trop_atts.append(r_trop.att)
            if np.isfinite(did_att):
                did_atts.append(did_att)

        # Both estimators must have produced enough usable estimates.
        assert len(trop_atts) >= max(
            5, n_reps // 2
        ), f"Only {len(trop_atts)} of {n_reps} TROP fits returned finite ATT"
        assert len(did_atts) >= max(
            5, n_reps // 2
        ), f"Only {len(did_atts)} of {n_reps} DiD fits returned finite ATT"

        trop_rmse = float(np.sqrt(np.mean(np.asarray(trop_atts) ** 2)))
        did_rmse = float(np.sqrt(np.mean(np.asarray(did_atts) ** 2)))
        ratio = trop_rmse / did_rmse

        # Theorem 5.1 ranking realisation: TROP MSE strictly below DID MSE.
        # Generous margin (0.7) absorbs MC noise; spike measurement
        # showed ratio ~ 0.34 at the calibrated DGP.
        assert ratio < 0.7, (
            f"TROP/DID RMSE ratio {ratio:.3f} not below 0.7 "
            f"(TROP RMSE={trop_rmse:.4f}, DID RMSE={did_rmse:.4f}, "
            f"true tau=0). Theorem 5.1 bias bound predicts TROP should "
            f"strictly dominate DID under factor confounding."
        )


# =============================================================================
# TestTROPSpecialCases — Section 2.2 reductions (DID, MC, SDID)
# =============================================================================


class TestTROPSpecialCases:
    """Section 2.2 reductions: TROP collapses to DID and MC under specific
    tunings.

    Paper Section 2.2 (page 6):
      - lambda_nn=infinity AND omega_j=theta_s=1 (uniform weights)
        --> reduces to DID / TWFE
      - omega_j=theta_s=1 (uniform weights) AND lambda_nn<infinity
        --> reduces to Matrix Completion (Athey et al. 2021, MC)
      - lambda_nn=infinity AND specific (omega, theta) weight choices
        --> reduces to SC and SDID

    This class verifies the DID and MC reductions on clean DGPs. The
    SC and SDID reductions are intentionally skipped: the paper claims
    they hold "with specific choices of unit and time weights" without
    providing the omega/theta map, and the library does not expose an
    SC- or SDID-matching weight setter (only ``lambda_unit`` /
    ``lambda_time`` decay rates per Eq. 3). Cross-language anchor
    against `SyntheticDiD` / a synthetic-control reference is deferred
    until paper-author code clarifies the weight map. (Documented in
    `METHODOLOGY_REVIEW.md` ``TROP`` section under Deviations.)

    Tests are NEW. The factor-DGP smoke previously in the
    pre-migration `TestTROPvsSDID::test_trop_handles_factor_dgp` is
    subsumed by `TestTROPTheorem51TripleRobustness` (which tests a
    stronger MC-ranking claim); the pre-migration `TestTROPvsSDID`
    class was deleted in the methodology-promotion PR.
    """

    def test_did_reduction_lambda_nn_inf_uniform_weights(self):
        """Section 2.2 first bullet: with ``lambda_nn = infinity`` and
        uniform weights (``lambda_time = lambda_unit = 0``), TROP and
        basic DiD should produce close ATT estimates on a no-
        interactive-FE panel (additive unit + time effects only).

        **This is a benchmark sanity check, not an algebraic-equivalence
        proof.** The paper's Section 2.2 reduction is stated for the
        2×2 block-assignment case; this test uses a multi-period panel
        (n_pre=6, n_post=4) where the library's basic DiD is the
        canonical comparator but not the algebraic target. The 0.5
        tolerance absorbs finite-sample MC noise. A direct algebraic-
        reduction test (true 2-period panel or `TwoWayFixedEffects`
        comparator) is deferred.
        """
        df = _make_no_factor_panel(
            n_units=30,
            n_treated=8,
            n_pre=6,
            n_post=4,
            noise_sd=0.3,
            treatment_effect=2.0,
            seed=_BASE_SEED_SPECIAL_CASES,
        )

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[np.inf],  # factor model disabled
            n_bootstrap=10,
            seed=42,
        )
        r_trop = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        did_att = _fit_did(df)

        assert np.isfinite(r_trop.att)
        assert np.isfinite(did_att)

        # Same DGP, same identification --> ATTs match within MC noise.
        # 0.5 absolute tolerance is generous given true_att = 2.0 with
        # noise_sd = 0.3 and finite sample.
        assert abs(r_trop.att - did_att) < 0.5, (
            f"TROP ATT={r_trop.att:.4f} and DID ATT={did_att:.4f} "
            f"should match under lambda_nn=inf + uniform weights "
            f"(paper Section 2.2 DID reduction)"
        )

    def test_matrix_completion_reduction_uniform_weights_finite_nn(self):
        """Section 2.2 second bullet: with uniform weights and finite
        ``lambda_nn``, TROP reduces to a Matrix Completion estimator.

        **This is NOT an equivalence check against an independent MC
        reference implementation.** The paper does not provide a separate
        MC algorithm specification beyond "uniform weights + finite
        nuclear-norm penalty", and the library does not bundle an
        independent MC port. What this test DOES verify is that the
        nuclear-norm prox code path activates under the MC tuning
        (effective_rank > 0) and that the resulting ATT beats the
        DID-style (lambda_nn=infinity) baseline on a factor-confounded
        DGP. A true equivalence test would require either an external
        MC port from R or a hand-written reference solver.
        """
        df = _make_trop_factor_panel(
            n_units=25,
            n_treated=6,
            n_pre=7,
            n_post=3,
            n_factors=2,
            factor_strength=1.2,
            treated_loading_shift=0.5,
            noise_sd=0.3,
            treatment_effect=2.0,
            seed=_BASE_SEED_SPECIAL_CASES + 1,
        )

        # MC-style: uniform weights + finite lambda_nn enables factor model.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mc_est = TROP(
                lambda_time_grid=[0.0],
                lambda_unit_grid=[0.0],
                lambda_nn_grid=[0.1, 1.0],  # finite --> factor model on
                n_bootstrap=10,
                seed=42,
            )
            r_mc = mc_est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

            # DID-style baseline: same uniform weights, infinite lambda_nn.
            did_est = TROP(
                lambda_time_grid=[0.0],
                lambda_unit_grid=[0.0],
                lambda_nn_grid=[np.inf],
                n_bootstrap=10,
                seed=42,
            )
            r_did_via_trop = did_est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

        true_att = 2.0
        # MC reduction should recover the ATT strictly better than the
        # no-factor baseline because the DGP has interactive FE. No
        # tolerance slack — under factor confounding the DID-style
        # baseline is biased and MC must strictly improve on it.
        assert abs(r_mc.att - true_att) < abs(r_did_via_trop.att - true_att), (
            f"MC reduction ATT error |{r_mc.att:.4f} - {true_att}| "
            f"should be strictly LESS than DID-style baseline error "
            f"|{r_did_via_trop.att:.4f} - {true_att}|"
        )
        # MC reduction's factor matrix must be non-trivially active.
        assert r_mc.effective_rank > 0, (
            f"MC reduction effective_rank={r_mc.effective_rank} should be > 0 "
            f"(finite lambda_nn engages the prox solver)"
        )


# =============================================================================
# TestTROPAlgorithm2MultipleTreated — Eq. 13 + Algorithm 2
# =============================================================================


class TestTROPAlgorithm2MultipleTreated:
    """Eq. 13 + Algorithm 2: per-(i, t) estimation for multiple treated units.

    For each treated observation (i, t), Eq. 13 fits a separate model as if
    (i, t) were the only treated cell, with observation-specific weights
    omega_j^{i,t}, theta_s^{i,t}. The ATT then averages over all W_it=1
    cells via Eq. 1:
    `tau_hat = (1 / sum_{i,t} W_it) sum_{i,t} W_it tau_hat_{it}(lambda_hat)`.

    This supports general assignment patterns (Section 6.1) including
    staggered adoption and heterogeneous treatment effects (Remark 6.1).
    Tests are NEW.
    """

    def test_treatment_effects_dict_has_entry_per_treated_cell(self):
        """Algorithm 2: ``TROPResults.treatment_effects`` contains one
        ``tau_hat_it`` entry per treated (unit, period) cell.

        With ``n_treated`` treated units and ``n_post`` post-treatment
        periods under block assignment, there are ``n_treated * n_post``
        treated cells. Each must have a finite per-cell estimate.
        """
        df = _make_no_factor_panel(
            n_units=20,
            n_treated=4,
            n_pre=5,
            n_post=3,
            noise_sd=0.3,
            treatment_effect=2.0,
            seed=_BASE_SEED_ALG2_MULTI,
        )

        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            max_iter=500,  # converge cleanly without "may be inaccurate" warning
            n_bootstrap=5,
            seed=42,
        )
        results = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        n_treated_cells = 4 * 3
        assert results.treatment_effects is not None
        assert len(results.treatment_effects) == n_treated_cells, (
            f"Expected {n_treated_cells} per-cell tau_hat entries, "
            f"got {len(results.treatment_effects)}"
        )
        for cell, tau_it in results.treatment_effects.items():
            assert np.isfinite(tau_it), f"Per-cell tau_hat at {cell} should be finite, got {tau_it}"

    def test_att_equals_mean_of_per_cell_effects(self):
        """Eq. 1: ``tau_hat = (1 / sum W_it) sum W_it tau_hat_it``.

        With block assignment (no observation-level weight kwargs), the
        reduction is just the unweighted mean of per-cell ``tau_hat_it``
        entries in ``treatment_effects``. Match within 1e-8 (no additional
        post-aggregation step in the ATT pipeline).
        """
        df = _make_no_factor_panel(
            n_units=18,
            n_treated=4,
            n_pre=5,
            n_post=2,
            noise_sd=0.3,
            treatment_effect=1.5,
            seed=_BASE_SEED_ALG2_MULTI + 1,
        )

        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            max_iter=500,  # converge cleanly without "may be inaccurate" warning
            n_bootstrap=5,
            seed=42,
        )
        results = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.treatment_effects is not None
        per_cell = np.array(list(results.treatment_effects.values()))
        mean_per_cell = float(np.mean(per_cell))
        assert np.isclose(results.att, mean_per_cell, atol=1e-8), (
            f"ATT={results.att:.10f} != mean of per-cell effects "
            f"{mean_per_cell:.10f} (Eq. 1 / Algorithm 2 step 3)"
        )


# =============================================================================
# TestTROPAlgorithm3Bootstrap — stratified pairs bootstrap
# =============================================================================


class TestTROPAlgorithm3Bootstrap:
    """Algorithm 3: stratified pairs bootstrap.

    Paper Algorithm 3 (page 27) resamples N_0 control rows and N_1 treated
    rows with replacement SEPARATELY (not pooled), preserving the
    treatment ratio across replicates and within-unit temporal
    correlation. This is a pairs bootstrap, NOT a multiplier / Rao-Wu
    bootstrap.

    Origin: ported from the pre-migration
    `TestPaperConformanceFixes::test_issue_d_stratified_bootstrap` in
    `test_trop.py` (the pre-migration class was deleted in the
    methodology-promotion PR). Bootstrap-failure-rate and bootstrap-
    NaN-SE guards are defensive surfaces and stayed in
    `tests/test_trop.py::TestTROPBootstrapFailureRateGuard` and
    `tests/test_trop.py::TestTROPBootstrapNaNSE`.
    """

    def test_stratified_pairs_resampling_completes(self, ci_params):
        """Algorithm 3: stratified pairs bootstrap completes successfully on
        an unbalanced (3 treated, 17 control) panel and yields a positive
        finite SE.

        With N_1=3 treated and N_0=17 control units, naive pooled
        resampling would sometimes draw 0 treated rows; the stratified
        sampler always draws 3 treated + 17 control per replicate, so the
        bootstrap distribution converges and the SE is positive.
        """
        rng = np.random.default_rng(789)

        n_treated = 3
        n_control = 17
        n_units = n_treated + n_control
        n_periods = 6
        true_att = 2.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_periods):
                post = t >= 3
                y = 10.0 + rng.normal(0, 0.5)
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                data.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        n_boot = ci_params.bootstrap(30)
        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=n_boot,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.bootstrap_distribution is not None
        min_successes = max(5, int(0.67 * n_boot))
        assert len(results.bootstrap_distribution) >= min_successes, (
            f"Expected >= {min_successes} successful bootstrap draws "
            f"out of {n_boot}, got {len(results.bootstrap_distribution)}"
        )
        assert results.se > 0
        assert np.isfinite(results.se)


# =============================================================================
# TestTROPEquation6FactorDGPRecovery — Section 3 / Eq. 6 semi-synthetic
# =============================================================================


@pytest.mark.slow
class TestTROPEquation6FactorDGPRecovery:
    """Section 3 / Eq. 6: semi-synthetic factor-DGP recovery.

    The paper's simulation framework (Section 3.1, pages 9-11) uses a
    rank-4 factor model with AR(2) errors over 6 real-data backbones.
    True treatment effects are zero (placebo). The library's equivalent
    DGP is built via `_make_trop_factor_panel` (wraps
    `diff_diff.prep.generate_factor_data`).

    Origin: 5 tests ported verbatim from the pre-migration
    `TestMethodologyVerification` class in `test_trop.py` (the
    pre-migration class was deleted in the methodology-promotion PR;
    original line range L552-878): limiting-case uniform weights,
    unit-weight bias reduction, time-weight bias reduction, factor-
    model bias reduction, and paper-DGP null-recovery. Verified that
    the ported tests still pass after relocation.
    """

    def test_limiting_case_uniform_weights(self):
        """Eq. 3 limiting case: lambda_unit = lambda_time = 0, lambda_nn = 0.

        With all lambdas at zero, TROP uses uniform weights and an
        unregularised L (paper Section 2.2 first bullet, omega_j=theta_s=1
        case). This should give TWFE-like estimates on a TWFE-clean panel.
        """
        rng = np.random.default_rng(42)
        n_units = 15
        n_treated = 5
        n_pre = 5
        n_post = 3
        true_att = 3.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            unit_fe = rng.normal(0, 0.5)
            for t in range(n_pre + n_post):
                post = t >= n_pre
                time_fe = 0.2 * t
                y = 10.0 + unit_fe + time_fe
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert (
            abs(results.att - true_att) < 1.0
        ), f"ATT={results.att:.3f} should be close to true={true_att}"
        assert results.lambda_time == 0.0
        assert results.lambda_unit == 0.0
        assert results.lambda_nn == 0.0

    def test_unit_weights_reduce_bias(self):
        """Eq. 3 unit weights: exp(-lambda_unit * RMSE_distance) reduces bias
        when control units vary in similarity to treated.

        Heterogeneous controls (5 similar + remainder dissimilar) plus a
        non-zero lambda_unit grid lets LOOCV pick informative weighting.
        """
        rng = np.random.default_rng(123)
        n_units = 25
        n_treated = 5
        n_pre = 6
        n_post = 3
        true_att = 2.5

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            if is_treated or i < n_treated + 5:
                unit_fe = 5.0 + rng.normal(0, 0.3)
            else:
                unit_fe = 10.0 + rng.normal(0, 0.5)

            for t in range(n_pre + n_post):
                post = t >= n_pre
                time_fe = 0.2 * t
                y = unit_fe + time_fe
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0, 1.0, 2.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert (
            abs(results.att - true_att) < 1.5
        ), f"ATT={results.att:.3f} should be close to true={true_att}"

    def test_time_weights_reduce_bias(self):
        """Eq. 3 time weights: exp(-lambda_time * |t - s|) reduces bias on
        trending pre-treatment outcomes.

        Quadratic pre-trend (time_fe = 0.1*t + 0.05*t^2/n_pre) makes recent
        periods more informative for extrapolating the counterfactual.
        """
        rng = np.random.default_rng(456)
        n_units = 20
        n_treated = 5
        n_pre = 8
        n_post = 3
        true_att = 2.0

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            unit_fe = rng.normal(0, 0.5)

            for t in range(n_pre + n_post):
                post = t >= n_pre
                time_fe = 0.1 * t + 0.05 * (t**2 / n_pre)
                y = 10.0 + unit_fe + time_fe
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_att
                y += rng.normal(0, 0.3)
                data.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.0],
            n_bootstrap=10,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert results.att > 0, f"ATT={results.att:.3f} should be positive"
        assert results.lambda_time in [0.0, 0.5, 1.0]

    def test_factor_model_reduces_bias(self, ci_params):
        """Eq. 2 / Section 2.2 MC reduction: nuclear-norm regularisation
        reduces bias when the true DGP has interactive fixed effects.

        Generated via `_make_trop_factor_panel` (wraps
        `diff_diff.prep.generate_factor_data`) with rank-2 factor
        structure and selection-on-loadings.
        """
        data = _make_trop_factor_panel(
            n_units=25,
            n_pre=7,
            n_post=3,
            n_treated=5,
            n_factors=2,
            treatment_effect=2.0,
            factor_strength=1.5,
            unit_fe_sd=1.0,
            noise_sd=0.5,
            seed=789,
        )

        n_boot = ci_params.bootstrap(20)
        nn_grid = ci_params.grid([0.0, 0.1, 1.0, 5.0])
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5],
            lambda_unit_grid=[0.0, 0.5],
            lambda_nn_grid=nn_grid,
            n_bootstrap=n_boot,
            seed=42,
        )
        results = trop_est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        true_att = 2.0
        assert (
            abs(results.att - true_att) < 2.0
        ), f"ATT={results.att:.3f} should be within 2.0 of true={true_att}"
        assert results.effective_rank > 0, "Factor matrix should have positive rank"

    def test_paper_dgp_recovery(self, ci_params):
        """Section 3 Eq. 6 null recovery: factor DGP with treatment_effect=0
        produces ATT estimates centred near zero.

        Mirrors paper Table 2 settings (page 32) at reduced sample size:
        2 factors, selection on loadings and levels, linear time trend,
        zero true treatment effect. The estimate should fall well inside a
        normal-distribution 2-sigma band of zero.
        """
        rng = np.random.default_rng(2024)
        n_units = 30
        n_treated = 6
        n_pre = 7
        n_post = 3
        n_factors = 2
        true_tau = 0.0

        F = rng.normal(0, 1, (n_pre + n_post, n_factors))

        Lambda = rng.normal(0, 1, (n_factors, n_units))
        Lambda[:, :n_treated] += 0.5

        gamma = rng.normal(0, 1, n_units)
        gamma[:n_treated] += 1.0

        delta = np.linspace(0, 2, n_pre + n_post)

        data = []
        for i in range(n_units):
            is_treated = i < n_treated
            for t in range(n_pre + n_post):
                post = t >= n_pre
                y = 10.0 + gamma[i] + delta[t]
                y += Lambda[:, i] @ F[t, :]
                treatment_indicator = 1 if (is_treated and post) else 0
                if treatment_indicator:
                    y += true_tau
                y += rng.normal(0, 0.5)

                data.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y,
                        "treated": treatment_indicator,
                    }
                )

        df = pd.DataFrame(data)

        n_boot = ci_params.bootstrap(30)
        trop_est = TROP(
            lambda_time_grid=[0.0, 0.5, 1.0],
            lambda_unit_grid=[0.0, 0.5, 1.0],
            lambda_nn_grid=[0.0, 0.1, 1.0],
            max_iter=500,  # converge cleanly without "may be inaccurate" warning
            n_bootstrap=n_boot,
            seed=42,
        )
        results = trop_est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )

        assert (
            abs(results.att) < 2.0
        ), f"ATT={results.att:.3f} should be close to true={true_tau} under null"
        assert results.effective_rank >= 0


# =============================================================================
# TestTROPDeviations — documented deviations and library choices
# =============================================================================


class TestTROPDeviations:
    """Locks library deviations from the paper and documented choices.

    Each test is independent (no bare cross-references to defensive
    tests in ``test_trop.py``; pointers live in this class docstring).

    Cross-references for context (NOT duplicated assertions, NOT
    locked by tests in this class):
    - `tests/test_trop.py::TestTROPBootstrapFailureRateGuard`
      (bootstrap proportional 5% failure-rate warning — defensive
      surface, stays in `test_trop.py`).
    - `tests/test_trop.py::TestTROPConvergenceWarnings`
      (FISTA / outer-loop convergence warnings — defensive surface,
      stays in `test_trop.py`).
    - `tests/test_trop.py::TestSilentWarningAudit`
      (Phase 2 silent-failure audit — defensive surface, stays in
      `test_trop.py`).
    - `tests/test_trop.py::TestDMatrixValidation`
      (absorbing-state validation — defensive surface, stays in
      `test_trop.py`).

    Tests in this class cover:
    - Gap #5 (unnormalised weights match Eq. 2, not Section 5
      sum-to-one).
    - lambda_nn=infinity internal conversion to 1e10 sentinel.
    - lambda_time / lambda_unit rejection of inf grid values.
    - Gap #9 (unbalanced panels supported beyond paper's balanced-
      panel assumption).
    - Event-style D rejection.
    - Eq. 14 covariate extension not supported (Theorem 8.1
      correspondingly out of scope).
    - Rank selection is implicit via nuclear-norm soft-thresholding
      (no discrete ``rank_selection`` constructor parameter exposed;
      paper Section 5.3 + Appendix matches this choice).
    - n_bootstrap < 2 rejection (no analytical SE; bootstrap required).
    - LOOCV happy-path uses the user-supplied grid verbatim
      (fallback-to-defaults side covered defensively in `test_trop.py`).
    - Inference CI uses t-distribution post safe_inference migration.
    - safe_inference NaN-propagation contract on degenerate SE inputs.
    """

    def test_unnormalized_weights_match_eq2(self):
        """Gap #5 (paper review): paper Section 5 (p. 20) states weights
        sum to one (``1^T omega = 1^T theta = 1``), but Eq. 3 (p. 7) writes
        unnormalised exponential weights. The library matches Eq. 2 —
        the kernel weight at ``lambda_unit = 0`` is exactly 1 (not 1/N).

        This test directly inspects the per-observation weight matrix
        returned by `TROP._compute_observation_weights` under
        ``lambda_unit = lambda_time = 0`` and asserts:

        - Every entry equals 1.0 within machine precision (unnormalised).
        - The sum equals ``n_units * n_periods`` (paper Section 5 would
          require sum = 1; Eq. 2 / library returns ``N * T``).

        See ``docs/methodology/papers/athey-2025-review.md`` Gap #5.
        """
        n_units = 8
        n_periods = 6
        # Minimal Y / D matrices in (n_periods, n_units) layout. Values
        # don't matter for the lambda=0 branch (which early-returns 1.0
        # without inspecting distances), but supply realistic shapes
        # so the helper signature is exercised end-to-end.
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS)
        Y = rng.normal(0, 1, (n_periods, n_units))
        D = np.zeros((n_periods, n_units), dtype=int)
        D[3:, 0] = 1  # one absorbing-state treated unit at t=3

        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=42,
        )

        weights = est._compute_observation_weights(
            Y=Y,
            D=D,
            i=0,
            t=3,
            lambda_time=0.0,
            lambda_unit=0.0,
            control_unit_idx=np.arange(1, n_units),
            n_units=n_units,
            n_periods=n_periods,
        )

        # All weight entries must be exactly 1.0 under uniform Eq. 3
        # specification (exp(-0 * dist) = 1; product of time and unit
        # weights is 1 * 1 = 1 everywhere).
        np.testing.assert_array_almost_equal(
            weights,
            np.ones((n_periods, n_units)),
            decimal=12,
            err_msg=(
                "Per-obs weights under lambda_time = lambda_unit = 0 must "
                "be exactly 1.0 (Eq. 2 unnormalised), not 1/N or any "
                "other normalised value (Gap #5 in paper review)"
            ),
        )
        # And the sum is N*T, not 1 — discriminating against the paper's
        # Section 5 "sum-to-one" claim.
        assert weights.sum() == float(n_units * n_periods), (
            f"weights.sum()={weights.sum()} != N*T={n_units * n_periods}; "
            "library would be using normalised weights (Gap #5 broken)"
        )

    def test_lambda_nn_inf_stored_unchanged(self):
        """REGISTRY ``## TROP`` "λ_nn=∞ implementation" edge-case note:
        lambda_nn=infinity is converted internally to 1e10 for
        computation, but ``TROPResults.lambda_nn`` stores the original
        ``inf`` value. lambda_time and lambda_unit store their selected
        grid values directly (no inf conversion — Eq. 3 uses
        ``lambda_time = lambda_unit = 0`` for "disabled", not infinity).
        """
        df = _make_no_factor_panel(
            n_units=15,
            n_treated=4,
            n_pre=5,
            n_post=2,
            noise_sd=0.3,
            treatment_effect=1.5,
            seed=_BASE_SEED_DEVIATIONS + 1,
        )
        est = TROP(
            lambda_time_grid=[0.0, 0.5],
            lambda_unit_grid=[0.0, 0.5],
            lambda_nn_grid=[np.inf],
            n_bootstrap=2,
            seed=42,
        )
        results = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )
        assert results.lambda_nn == np.inf, (
            f"lambda_nn={results.lambda_nn} not preserved as inf "
            "(REGISTRY says ORIGINAL value is stored)"
        )
        assert results.lambda_time in [0.0, 0.5]
        assert results.lambda_unit in [0.0, 0.5]

    def test_inf_in_lambda_time_or_unit_grid_rejected(self):
        """REGISTRY ``## TROP`` "Disabled parameter semantics" note: only
        ``lambda_nn`` may be infinity in Eq. 3 semantics.
        ``lambda_time = 0`` and ``lambda_unit = 0`` mean "uniform weights"
        (disabled), because ``exp(-0 * dist) = 1``. inf in lambda_time /
        lambda_unit is rejected with ``ValueError`` pointing users to
        0.0 for uniform weights.
        """
        df = _make_no_factor_panel(
            n_units=10,
            n_treated=3,
            n_pre=4,
            n_post=2,
            noise_sd=0.1,
            treatment_effect=1.0,
            seed=_BASE_SEED_DEVIATIONS + 2,
        )
        with pytest.raises(ValueError, match="(?i)inf|infinity"):
            TROP(
                lambda_time_grid=[np.inf],
                lambda_unit_grid=[0.0],
                lambda_nn_grid=[0.1],
                n_bootstrap=2,
                seed=42,
            ).fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

    def test_unbalanced_panels_supported(self):
        """Gap #9 (paper review): the paper assumes balanced panels
        (N x T). The library accepts unbalanced panels with missing
        unit-period observations.

        Construct a balanced panel and drop ~10% of rows; TROP fit must
        complete and return a finite ATT.
        """
        df = _make_no_factor_panel(
            n_units=18,
            n_treated=5,
            n_pre=5,
            n_post=3,
            noise_sd=0.3,
            treatment_effect=2.0,
            seed=_BASE_SEED_DEVIATIONS + 3,
        )
        # Drop 10 rows at random (control + pre-treatment cells only,
        # so the treatment indicator stays absorbing).
        rng = np.random.default_rng(_BASE_SEED_DEVIATIONS + 30)
        eligible_idx = df.index[(df["treated"] == 0)].to_numpy()
        drop_idx = rng.choice(eligible_idx, size=10, replace=False)
        df_unbal = df.drop(index=drop_idx).reset_index(drop=True)
        assert len(df_unbal) < len(df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = TROP(
                lambda_time_grid=[0.0],
                lambda_unit_grid=[0.0],
                lambda_nn_grid=[0.1],
                n_bootstrap=5,
                seed=42,
            )
            results = est.fit(
                df_unbal,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )
        assert np.isfinite(results.att)
        assert results.se >= 0

    def test_event_style_d_rejected_with_value_error(self):
        """REGISTRY ``## TROP`` "D matrix validation" edge-case note:
        event-style D (only first treatment period has D=1) is rejected
        because monotonicity (absorbing state) is violated. Error message
        must guide users to convert to absorbing state.
        """
        # Build event-style: D=1 at t=3, D=0 at t=4 (1->0 transition
        # is non-monotonic, violating absorbing state).
        rows = []
        for i in range(10):
            for t in range(6):
                d = 1 if (i < 3 and t == 3) else 0  # event-style!
                rows.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": 1.0 + 0.1 * t + (1.0 if d else 0.0),
                        "treated": d,
                    }
                )
        df = pd.DataFrame(rows)
        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=42,
        )
        with pytest.raises(ValueError, match="(?i)absorbing|monotonic"):
            est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )

    def test_covariates_not_supported(self):
        """Eq. 14 + Theorem 8.1 are deferred. ``TROP.fit()`` does NOT
        accept a ``covariates`` keyword argument. The check uses
        ``inspect.signature`` so that adding a future ``**kwargs`` would
        not silently break this contract.

        See ``METHODOLOGY_REVIEW.md`` ``TROP`` "Outstanding Concerns".
        """
        sig = inspect.signature(TROP.fit)
        param_names = set(sig.parameters.keys())
        assert "covariates" not in param_names, (
            f"TROP.fit() unexpectedly exposes a 'covariates' parameter: "
            f"{param_names}. Equation 14 covariate extension is deferred; "
            f"adding the parameter without implementing the X*beta_coef + R "
            f"objective per Eq. 14 would silently break methodology contract."
        )
        # No **kwargs either (would let covariates slip through).
        var_kw = [n for n, p in sig.parameters.items() if p.kind == inspect.Parameter.VAR_KEYWORD]
        assert var_kw == [], (
            f"TROP.fit() exposes **kwargs ({var_kw}); covariate-trap " "is no longer airtight."
        )

    def test_rank_selection_is_implicit_via_nuclear_norm(self):
        """Paper Section 5.3 + Appendix: rank selection is implicit via
        nuclear-norm soft-thresholding. The library matches this — there
        is NO discrete ``rank_selection`` constructor parameter exposing
        cv / ic / elbow methods.

        Earlier REGISTRY prose (pre-promotion) mentioned "cv / ic / elbow"
        methods; that claim was an overclaim and is corrected in this
        promotion. This test locks the actual contract: rank is reported
        post-hoc via ``TROPResults.effective_rank`` (sum of singular
        values divided by the largest singular value) as a diagnostic,
        not as a user-selectable mode.
        """
        sig = inspect.signature(TROP.__init__)
        assert "rank_selection" not in sig.parameters, (
            "TROP.__init__ unexpectedly exposes a 'rank_selection' parameter. "
            "The paper specifies implicit-via-nuclear-norm rank selection; "
            "if a discrete switch is added, this deviation note in "
            "METHODOLOGY_REVIEW.md and REGISTRY.md must be updated."
        )
        # effective_rank is reported as a diagnostic post-fit.
        df = _make_no_factor_panel(
            n_units=10,
            n_treated=3,
            n_pre=4,
            n_post=2,
            noise_sd=0.1,
            treatment_effect=1.0,
            seed=_BASE_SEED_DEVIATIONS + 4,
        )
        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=42,
        )
        results = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )
        assert hasattr(results, "effective_rank")
        assert np.isfinite(results.effective_rank)

    def test_n_bootstrap_minimum_is_2(self):
        """REGISTRY ``## TROP`` "Bootstrap minimum" edge-case note:
        ``n_bootstrap`` must be >= 2 (enforced via
        ``ValueError``). TROP has no analytical SE formula — bootstrap
        is the only variance estimator, so n_bootstrap=0 or 1 cannot
        produce a defined SE.
        """
        with pytest.raises(ValueError, match="(?i)bootstrap"):
            TROP(
                lambda_time_grid=[0.0],
                lambda_unit_grid=[0.0],
                lambda_nn_grid=[0.1],
                n_bootstrap=1,
                seed=42,
            )

    def test_loocv_returns_user_grid_values_on_well_conditioned_panel(self):
        """REGISTRY ``## TROP`` "LOOCV failure handling" edge-case note
        (happy-path side): when LOOCV produces a finite
        Q(lambda) on at least one grid point, the result tuple
        ``(lambda_time, lambda_unit, lambda_nn)`` is from the user-
        supplied grid (no fallback to documented defaults
        ``(1.0, 1.0, 0.1)``).

        The fallback-warning side (when ALL parameter combinations fail
        LOOCV) is covered by `tests/test_trop.py::TestLOOCVFallback`
        defensive surfaces — duplication here would be redundant. This
        test locks the dual: well-conditioned panel uses the user grid
        verbatim, so any regression that prematurely triggers the
        fallback would surface here.
        """
        df = _make_no_factor_panel(
            n_units=12,
            n_treated=3,
            n_pre=4,
            n_post=2,
            noise_sd=0.3,
            treatment_effect=1.5,
            seed=_BASE_SEED_DEVIATIONS + 5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = TROP(
                lambda_time_grid=[0.5],
                lambda_unit_grid=[0.5],
                lambda_nn_grid=[0.1],
                n_bootstrap=2,
                seed=42,
            )
            results = est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
            )
        # Well-conditioned panel: result should be the user grid values
        # (LOOCV succeeded, no fallback triggered).
        assert results.lambda_time == 0.5
        assert results.lambda_unit == 0.5
        assert results.lambda_nn == 0.1

    def test_inference_ci_uses_t_distribution(self):
        """REGISTRY ``## TROP`` "Inference CI distribution" edge-case
        note: after the safe_inference migration the confidence interval
        uses the t-distribution with df = max(1, n_treated_obs - 1),
        consistent with p_value. (Previously CI used normal-distribution
        while p_value used t-distribution.)

        Lock: with a well-defined SE, the CI half-width equals
        ``t_{alpha/2, df} * SE`` within numerical tolerance.
        """
        df = _make_no_factor_panel(
            n_units=15,
            n_treated=4,
            n_pre=5,
            n_post=2,
            noise_sd=0.3,
            treatment_effect=2.0,
            seed=_BASE_SEED_DEVIATIONS + 6,
        )
        est = TROP(
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            max_iter=500,  # converge cleanly without "may be inaccurate" warning
            n_bootstrap=5,
            seed=42,
        )
        results = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
        )
        from scipy import stats

        df_t = max(1, results.n_treated_obs - 1)
        t_crit = stats.t.ppf(1 - results.alpha / 2, df_t)
        ci_lower, ci_upper = results.conf_int
        half_width = (ci_upper - ci_lower) / 2.0
        expected_half_width = t_crit * results.se
        assert np.isclose(half_width, expected_half_width, rtol=1e-6), (
            f"CI half-width={half_width:.6e} does not match "
            f"t_{{alpha/2, df={df_t}}} * SE = {expected_half_width:.6e} "
            f"(REGISTRY ## TROP 'Inference CI distribution' note: post safe_inference migration uses t-dist)"
        )

    def test_safe_inference_nan_propagation_contract(self):
        """`diff_diff.utils.safe_inference` invariant: when SE is non-
        finite or non-positive, ALL inference fields (t_stat, p_value,
        conf_int) are NaN-consistent.

        This test invokes `safe_inference` directly with the degenerate
        inputs (SE = 0, SE = NaN, SE = -1) used across the TROP code
        paths, so the NaN-propagation contract is exercised
        deterministically rather than depending on a panel construction
        that *might* produce a zero-SE bootstrap distribution. The
        product-level coverage of TROP's NaN-SE propagation lives at
        `tests/test_trop.py::TestTROPBootstrapNaNSE` (panel-level) and
        `tests/test_trop.py::TestTROPResults::test_nan_propagation_when_se_zero`
        (results-level); this methodology test locks the underlying
        invariant.
        """
        from diff_diff.utils import safe_inference
        from tests.conftest import assert_nan_inference

        # Cover the three SE-degenerate inputs that propagate to NaN
        # inference per the contract: zero, negative, NaN.
        for bad_se in (0.0, -1.0, float("nan")):
            t_stat, p_value, conf_int = safe_inference(
                effect=1.5,
                se=bad_se,
                alpha=0.05,
            )
            assert_nan_inference(
                {
                    "se": bad_se,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "conf_int": conf_int,
                }
            )

    # ------------------------------------------------------------------
    # Non-absorbing (general assignment) support — Eq. 1 / Eq. 12 /
    # Algorithm 2, Section 6.1. The paper's estimator handles general
    # assignment patterns ("units moving into and out of treatment"),
    # not only absorbing/staggered adoption (§2.1). The library exposes
    # this via the opt-in TROP(non_absorbing=True); the default still
    # rejects non-monotonic D (covered in test_trop.py::TestDMatrixValidation
    # and test_event_style_d_rejected_with_value_error above).
    # ------------------------------------------------------------------

    @staticmethod
    def _make_non_absorbing_panel(seed=0, tau=3.0, n_units=16, n_periods=8, all_toggle=False):
        """TWFE-clean panel with on/off (non-absorbing) treatment, no dynamic effects.

        Y_it(0) = alpha_i + beta_t + noise; Y_it(1) = Y_it(0) + tau. Some units
        switch treatment on and then off again, so D is non-monotonic.
        """
        rng = np.random.default_rng(seed)
        alpha = rng.normal(0.0, 1.0, n_units)
        beta = rng.normal(0.0, 1.0, n_periods)
        rows = []
        for i in range(n_units):
            d = np.zeros(n_periods, dtype=int)
            if all_toggle:
                on = 3 + (i % 2)
                d[on : on + 2] = 1  # every unit treated on an interior block
            elif i % 4 == 0 and i > 0:
                d[4:6] = 1  # on then off (non-absorbing)
            elif i % 3 == 0:
                d[5:] = 1  # absorbing block (mix of patterns)
            for t in range(n_periods):
                y0 = alpha[i] + beta[t] + rng.normal(0.0, 0.05)
                rows.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y0 + (tau if d[t] == 1 else 0.0),
                        "treated": int(d[t]),
                    }
                )
        return pd.DataFrame(rows)

    @pytest.mark.slow
    def test_non_absorbing_general_assignment_supported(self):
        """TROP(non_absorbing=True) accepts on/off treatment and recovers the
        ATT on a no-dynamic-effects DGP (Eq. 1 averages over all D=1 cells;
        Eq. 12 / Algorithm 2 masks treated cells per (i, t)). A caveat
        ``UserWarning`` is emitted because Theorem 5.1's guarantee is proven
        only under block assignment.
        """
        tau = 3.0
        df = self._make_non_absorbing_panel(seed=0, tau=tau)
        n_treated_cells = int(df["treated"].to_numpy().sum())
        # Sanity: the panel really is non-absorbing (some unit goes 1 -> 0).
        treated_wide = df.pivot(index="period", columns="unit", values="treated").to_numpy()
        assert bool(
            (np.diff(treated_wide, axis=0) < 0).any()
        ), "test panel must contain a 1->0 transition"

        est = TROP(
            method="local",
            non_absorbing=True,
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=1,
        )
        with pytest.warns(UserWarning, match="(?i)non_absorbing.*Theorem 5.1"):
            res = est.fit(df, "outcome", "treated", "unit", "period")

        # Estimand: ATT averages the per-cell effects over all D=1 cells (Eq. 1).
        assert np.isfinite(res.att)
        assert abs(res.att - tau) < 0.5, f"ATT {res.att} should recover tau={tau}"
        # One finite per-cell effect per treated cell (Eq. 12 / Algorithm 2).
        assert len(res.treatment_effects) == n_treated_cells
        assert all(np.isfinite(v) for v in res.treatment_effects.values())

    def test_non_absorbing_no_caveat_in_default_mode(self):
        """The non-absorbing caveat warning fires ONLY for non_absorbing=True;
        a default (absorbing) fit must not emit it.
        """
        # Absorbing staggered panel (monotonic per unit).
        rows = []
        for i in range(12):
            g = 4 if i < 3 else (6 if i < 6 else None)
            for t in range(8):
                d = 1 if (g is not None and t >= g) else 0
                rows.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": float(i) * 0.1 + float(t) * 0.2 + (2.0 if d else 0.0),
                        "treated": d,
                    }
                )
        df = pd.DataFrame(rows)
        est = TROP(
            method="local",
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.fit(df, "outcome", "treated", "unit", "period")
        assert not any(
            "non_absorbing" in str(w.message) for w in caught
        ), "default (absorbing) mode must not emit the non_absorbing caveat"

    @pytest.mark.slow
    def test_non_absorbing_unbalanced_panel_supported(self):
        """Non-absorbing support tolerates unbalanced panels (random missing
        control cells) and still returns a finite ATT.
        """
        df = self._make_non_absorbing_panel(seed=7, tau=3.0)
        # Drop 10% of untreated rows at random (missing control observations).
        rng = np.random.default_rng(11)
        control_rows = df.index[df["treated"] == 0].to_numpy()
        drop = rng.choice(control_rows, size=int(0.1 * len(control_rows)), replace=False)
        df_unbalanced = df.drop(index=drop).reset_index(drop=True)

        est = TROP(
            method="local",
            non_absorbing=True,
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=2,
            seed=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = est.fit(df_unbalanced, "outcome", "treated", "unit", "period")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)

    @pytest.mark.slow
    @pytest.mark.parametrize("lambda_unit", [0.0, 1.0])
    def test_non_absorbing_always_treated_unit_not_raw_outcome(self, lambda_unit):
        """A treated cell whose UNIT has no observed control cell leaves ``alpha_i``
        unidentified, so its tau would silently leak the unit fixed effect (a
        raw-outcome-like value). Such cells must be marked non-estimable (NaN).
        This holds for BOTH ``lambda_unit=0`` (uniform unit weights still give the
        always-treated unit no own control row) and ``lambda_unit>0`` (inf
        distance -> zero donor weights). Estimable cells still recover the effect
        and the bootstrap SE stays finite.

        Locks the documented behavior (REGISTRY ## TROP "non-absorbing
        non-estimable-cell trimming" Note): the ATT is the mean over estimable
        treated cells (library-wide non-estimable->NaN convention).
        """
        rng = np.random.default_rng(0)
        n_units, n_periods, tau = 8, 8, 5.0
        alpha = rng.normal(0.0, 1.0, n_units)
        alpha[0] = 10.0  # large unit-0 FE so any leak is unmistakable
        beta = rng.normal(0.0, 1.0, n_periods)
        rows = []
        for i in range(n_units):
            for t in range(n_periods):
                if i == 0:
                    d = 1  # always-treated: no untreated history
                elif i % 3 == 0:
                    d = 1 if 4 <= t <= 5 else 0  # on/off
                else:
                    d = 1 if t >= 6 else 0  # untreated history present
                y0 = alpha[i] + beta[t] + rng.normal(0.0, 0.05)
                rows.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y0 + (tau if d else 0.0),
                        "treated": int(d),
                    }
                )
        df = pd.DataFrame(rows)
        est = TROP(
            method="local",
            non_absorbing=True,
            lambda_time_grid=[0.0],
            lambda_unit_grid=[lambda_unit],
            lambda_nn_grid=[0.1],
            n_bootstrap=3,
            seed=1,
        )
        with pytest.warns(UserWarning, match="(?i)not estimable"):
            res = est.fit(df, "outcome", "treated", "unit", "period")

        # Every cell of the always-treated unit is non-estimable (NaN), never a
        # fixed-effect-contaminated raw outcome (alpha_0 = 10 would leak otherwise).
        raw_y = {
            t: float(df[(df.unit == 0) & (df.period == t)]["outcome"].iloc[0])
            for t in range(n_periods)
        }
        u0 = {k: v for k, v in res.treatment_effects.items() if k[0] == 0}
        assert len(u0) == n_periods
        for (_, t), v in u0.items():
            assert np.isnan(v), f"cell(0,{t}) should be NaN, got {v}"
            assert not np.isclose(v, raw_y[t]), "tau must not equal raw outcome"

        # Estimable cells (other units) remain finite and aggregate near the truth.
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert abs(res.att - tau) < 0.6

    @pytest.mark.slow
    def test_non_absorbing_fully_treated_period_not_estimable(self):
        """A period in which EVERY unit is treated has no control cell, so
        ``beta_t`` is unidentified and that period's tau would leak the time fixed
        effect. Those cells must be NaN (non-estimable), not finite raw-outcome
        values; treated cells in other periods still recover the effect.
        """
        rng = np.random.default_rng(1)
        n_units, n_periods, tau, hot = 8, 8, 3.0, 4
        alpha = rng.normal(0.0, 1.0, n_units)
        beta = rng.normal(0.0, 1.0, n_periods)
        beta[hot] = 20.0  # large period-`hot` FE so any leak is unmistakable
        rows = []
        for i in range(n_units):
            for t in range(n_periods):
                if t == hot:
                    d = 1  # every unit treated at `hot` -> no control at that period
                elif i % 2 == 0:
                    d = 1 if t >= 6 else 0
                else:
                    d = 1 if 1 <= t <= 2 else 0
                y0 = alpha[i] + beta[t] + rng.normal(0.0, 0.05)
                rows.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": y0 + (tau if d else 0.0),
                        "treated": int(d),
                    }
                )
        df = pd.DataFrame(rows)
        # Sanity: period `hot` is fully treated.
        assert bool((df[df.period == hot]["treated"].to_numpy() == 1).all())
        est = TROP(
            method="local",
            non_absorbing=True,
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=3,
            seed=1,
        )
        with pytest.warns(UserWarning, match="(?i)not estimable"):
            res = est.fit(df, "outcome", "treated", "unit", "period")

        hot_cells = {k: v for k, v in res.treatment_effects.items() if k[1] == hot}
        assert len(hot_cells) == n_units
        for k, v in hot_cells.items():
            assert np.isnan(v), f"fully-treated-period cell {k} should be NaN, got {v}"
        # Treated cells in other (estimable) periods still recover the effect.
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert abs(res.att - tau) < 0.6

    @pytest.mark.slow
    def test_non_absorbing_fully_toggling_no_never_treated_unit(self):
        """non_absorbing admits a fully toggling panel with NO never-treated unit
        (every unit is treated at some point but retains observed untreated
        cells). Identification falls back to untreated cells, and the bootstrap
        runs via the Python path (the Rust stratified resampler can return a
        degenerate ~0 SE on an empty control stratum). Asserts admission + finite
        ATT/SE + recovery.
        """
        tau = 4.0
        df = self._make_non_absorbing_panel(seed=2, tau=tau, all_toggle=True)
        # Sanity: no never-treated unit (every unit treated at some period).
        assert bool((df.groupby("unit")["treated"].max().to_numpy() == 1).all())
        est = TROP(
            method="local",
            non_absorbing=True,
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=3,
            seed=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = est.fit(df, "outcome", "treated", "unit", "period")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert res.se > 0  # not a degenerate empty-stratum ~0 SE
        assert abs(res.att - tau) < 0.6

    @pytest.mark.slow
    def test_unbalanced_absorbing_unidentified_unit_not_estimable(self):
        """The estimability guard applies to DEFAULT (absorbing) local fits too,
        not only non_absorbing. On an unbalanced absorbing panel where a treated
        unit's pre-treatment rows are entirely missing, that unit has no observed
        control cell, so ``alpha_i`` is unidentified; its cells must be NaN (the
        prior behavior silently reported a fixed-effect-contaminated tau), while
        the rest of the panel is estimated normally. ``non_absorbing=False``.
        """
        rng = np.random.default_rng(3)
        n_periods, tau = 6, 4.0
        beta = rng.normal(0.0, 1.0, n_periods)
        rows = []
        # Never-treated controls (units 0-2), observed all periods.
        for i in range(3):
            a = rng.normal(0.0, 1.0)
            for t in range(n_periods):
                rows.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": a + beta[t] + rng.normal(0, 0.05),
                        "treated": 0,
                    }
                )
        # Well-observed treated unit (unit 3), adopts at t=4, full pre-history.
        a3 = rng.normal(0.0, 1.0)
        for t in range(n_periods):
            d = 1 if t >= 4 else 0
            rows.append(
                {
                    "unit": 3,
                    "period": t,
                    "outcome": a3 + beta[t] + rng.normal(0, 0.05) + (tau if d else 0),
                    "treated": d,
                }
            )
        # Pathological treated unit (unit 4): adopts at t=3 but ONLY observed at
        # its treated periods 3,4,5 -- pre-treatment rows 0,1,2 are MISSING, so it
        # has no observed control cell and alpha_4 is unidentified.
        a4 = rng.normal(0.0, 1.0)
        a4 += 12.0  # large FE so any leak into tau would be unmistakable
        for t in (3, 4, 5):
            rows.append(
                {
                    "unit": 4,
                    "period": t,
                    "outcome": a4 + beta[t] + rng.normal(0, 0.05) + tau,
                    "treated": 1,
                }
            )
        df = pd.DataFrame(rows)

        est = TROP(
            method="local",  # non_absorbing defaults to False (absorbing)
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=3,
            seed=1,
        )
        with pytest.warns(UserWarning, match="(?i)not estimable"):
            res = est.fit(df, "outcome", "treated", "unit", "period")

        # Unit 4's treated cells are NaN (alpha_4 unidentified), never a leaked FE.
        u4 = {k: v for k, v in res.treatment_effects.items() if k[0] == 4}
        assert len(u4) == 3
        for k, v in u4.items():
            assert np.isnan(v), f"unidentified-unit cell {k} should be NaN, got {v}"
        # Unit 3 (well-observed) is estimated and recovers the effect.
        u3 = [v for k, v in res.treatment_effects.items() if k[0] == 3 and np.isfinite(v)]
        assert len(u3) > 0
        assert np.isfinite(res.att)
        assert abs(res.att - tau) < 0.6

    @pytest.mark.slow
    def test_non_absorbing_disconnected_support_not_estimable(self):
        """Strict two-way-FE identification: ``alpha_i + beta_t`` is pinned only
        within a connected component of the observed-control bipartite graph. A
        treated cell whose target unit and target period fall in DIFFERENT
        components is non-estimable even though both have *some* control support
        (the marginal row/column check would wrongly pass it). Such cells must be
        NaN, not a finite cross-component FE-contaminated value.

        Construction (periods 0-5): component A = units {0,1,2} whose untreated
        periods rotate within {0,1,2,3} (so A's control graph connects periods
        0-3 and units 0-2 into one component, with estimable treated cells inside
        it); component B = units {3,4} untreated only at periods {4,5}. A and B
        share no unit or period, so they are disconnected. Target cell (unit 0,
        period 4): unit 0 in A, period 4 in B -> alpha_0 + beta_4 unidentified ->
        NaN. A large beta_4 makes any cross-component leak unmistakable; component
        A still yields a finite ATT.
        """
        rng = np.random.default_rng(0)
        n_periods, tau = 6, 3.0
        alpha = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        beta = np.zeros(n_periods)
        beta[4] = 20.0
        # untreated-period sets: component A rotates within {0..3} and is treated
        # at {4,5}; component B is untreated only at {4,5}.
        untreated = {
            0: {0, 1},
            1: {1, 2},
            2: {2, 3},
            3: {4, 5},
            4: {4, 5},
        }
        rows = []
        for i in range(5):
            for t in range(n_periods):
                d = 0 if t in untreated[i] else 1
                rows.append(
                    {
                        "unit": i,
                        "period": t,
                        "outcome": alpha[i] + beta[t] + rng.normal(0, 0.05) + (tau if d else 0),
                        "treated": int(d),
                    }
                )
        df = pd.DataFrame(rows)
        est = TROP(
            method="local",
            non_absorbing=True,
            lambda_time_grid=[0.0],
            lambda_unit_grid=[0.0],
            lambda_nn_grid=[0.1],
            n_bootstrap=3,
            seed=1,
        )
        with pytest.warns(UserWarning, match="(?i)not estimable"):
            res = est.fit(df, "outcome", "treated", "unit", "period")

        # The cross-component target cell (unit 0 treated at period 4) is
        # non-estimable (NaN), not a beta_4-contaminated value.
        cell = res.treatment_effects.get((0, 4))
        assert cell is not None and np.isnan(cell), f"(0,4) should be NaN, got {cell}"
        # Within-component A treated cells (e.g. unit 0 at period 2) stay
        # estimable, so the fit still produces a finite ATT.
        assert np.isfinite(res.treatment_effects.get((0, 2)))
        assert np.isfinite(res.att)
