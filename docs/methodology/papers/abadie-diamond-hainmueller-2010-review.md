# Paper Review: Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program

**Authors:** Alberto Abadie, Alexis Diamond, Jens Hainmueller
**Citation:** Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493–505.
**PDF reviewed:** https://doi.org/10.1198/jasa.2009.ap08746 (published JASA version)
**Review date:** 2026-05-29

> Scope note: this review captures **only** what is in Abadie, Diamond & Hainmueller (2010). The method originates in Abadie & Gardeazabal (2003) — cited here for the V-selection numerics (App. B of that paper) — and the leave-one-out / detailed in-time-placebo diagnostics are developed in ADH (2015); both are reviewed separately. Nothing here is sourced from outside this paper.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Copy the `## SyntheticControl` section into the "Counterfactual / Synthetic Estimators" category.*

## SyntheticControl

**Primary source:** Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies." *JASA*, 105(490), 493–505. https://doi.org/10.1198/jasa.2009.ap08746

**Key implementation requirements:**

*Assumption checks / warnings:*
- **One treated unit, block assignment.** Region `i = 1` is uninterruptedly exposed to the intervention after period `T0` (`1 ≤ T0 < T`); regions `i = 2, …, J+1` are the never-exposed "donor pool" (Section 2.2). No staggered adoption.
- **No interference / no spillovers (SUTVA across units).** Untreated regions' outcomes are unaffected by the treated unit's intervention (Section 2.2, citing Rosenbaum 2007). Section 3 discusses violations: e.g., cigarette smuggling into California or tobacco-industry ad-fund diversion to control states would contaminate the donor pool (journal pp. 501).
- **No anticipation.** `Y_it^I = Y_it^N` for all `t ≤ T0`. If the outcome reacts before formal implementation (anticipation), `T0` must be redefined to the first period the outcome may respond (journal p. 494).
- **Good pre-treatment fit is required, not assumed.** Equation (2) can hold exactly only if the treated unit's pre-period characteristics lie in the **convex hull** of the donors'. "In some instances, the fit may be poor and then we would not recommend using a synthetic control" (journal p. 495). The method *forces the researcher to demonstrate* fit (Table 1 predictor balance; Figure 2 pre-period trajectory).
- **Interpolation bias.** Even with good fit, bias can be large if the linear factor model does not hold across regions with very different characteristics; mitigate by restricting the donor pool to similar units, or by adding penalty terms to the weight objective (journal pp. 495–496).
- **Donor-pool curation.** Exclude units exposed to the same/similar intervention or to large confounding shocks (in the application: states with their own tobacco programs or large cigarette-tax hikes, and DC, are dropped; journal pp. 498–499).

*Notation (Section 2.2):*
- `J+1` regions, `i = 1` treated; `t = 1, …, T`; `T0` = number of pre-intervention periods.
- `Y_it^N` = outcome without intervention; `Y_it^I` = outcome with intervention; observed `Y_it`.
- `Z_i` = `(r×1)` observed covariates **not affected by** the intervention; `δ_t` common factor; `θ_t` `(1×r)`; `λ_t` `(1×F)` unobserved common factors; `μ_i` `(F×1)` factor loadings; `ε_it` transitory shocks (mean zero).

*Observed outcome and target (Section 2.2, journal p. 495):*

    Y_it = Y_it^N + α_it · D_it,    D_it = 1{ i = 1 and t > T0 }

    α_1t = Y_1t^I − Y_1t^N = Y_1t − Y_1t^N,   for t > T0    ← target (per-period treatment effect on the treated unit)

`Y_1t` is observed, so estimating `α_1t` reduces to imputing the counterfactual `Y_1t^N`.

*Model justifying the estimator (Equation 1):*

    Y_it^N = δ_t + θ_t·Z_i + λ_t·μ_i + ε_it

This **generalizes the two-way fixed-effects / DiD model**: if `λ_t` is constant across `t`, Equation (1) collapses to the standard DiD (unobserved confounder effects constant in time, removed by differencing). The factor model lets the effect of unobserved confounders `μ_i` vary over time (journal p. 495).

*Synthetic control and the weight vector `W`:*

    W = (w_2, …, w_{J+1})',   w_j ≥ 0,   Σ_{j=2}^{J+1} w_j = 1     (simplex constraint)

    synthetic-control outcome at t:   Σ_{j=2}^{J+1} w_j · Y_jt

*Identifying weights (Equation 2):* there exist `(w_2*, …, w_{J+1}*)` such that

    Σ_{j=2}^{J+1} w_j*·Y_jt = Y_1t   for t = 1, …, T0,   and   Σ_{j=2}^{J+1} w_j*·Z_j = Z_1

*Estimator (as implemented):*

    α̂_1t = Y_1t − Σ_{j=2}^{J+1} w_j*·Y_jt,    for t ∈ {T0+1, …, T}

*Bias control (Equation 3, proved in Appendix B):* when `Σ_{t=1}^{T0} λ_t'λ_t` is nonsingular,

    Y_1t^N − Σ_j w_j*·Y_jt = Σ_j w_j* Σ_{s=1}^{T0} λ_t (Σ_{n=1}^{T0} λ_n'λ_n)^{-1} λ_s'(ε_js − ε_1s) − Σ_j w_j*(ε_jt − ε_1t)

Appendix B (journal pp. 503–505) bounds `E|R_1t|` via Cauchy–Schwarz + Hölder + Rosenthal inequalities and shows **the bias → 0 as the number of pre-treatment periods `T0` grows** relative to the scale of transitory shocks. Practical implication: a long, well-fit pre-period is the key requirement.

*Why fitting pre-period outcomes suffices (Equation 4):* if `Σ w_j* Z_j = Z_1` **and** `Σ w_j* μ_j = μ_1`, the estimator is unbiased. `μ_j` is unobserved, but fitting `Z_1` and a long set of pre-period outcomes `Y_11, …, Y_1T0` implies `Σ w_j* μ_j ≈ μ_1`, so (4) holds approximately (journal p. 495).

*Single-pretreatment-period case (Equations 5–6):* under an autoregressive model with time-varying coefficients (Eq. 5), if weights satisfy `Σ w_j* Y_jT0 = Y_1T0` and `Σ w_j* Z_jT0 = Z_1T0` (Eq. 6), the estimator is unbiased even with one pre-period (Appendix B).

*Weight estimation — the nested ("V-matrix") optimization (Section 2.3, journal p. 496):*

Predictor vectors stack covariates and `M` linear combinations of pre-period outcomes. With `K_m = (k_1, …, k_{T0})'` defining `Ȳ_i^{K_m} = Σ_{s=1}^{T0} k_s·Y_is`:

    X_1 = (Z_1', Ȳ_1^{K_1}, …, Ȳ_1^{K_M})'      (k×1),   k = r + M
    X_0 = (k×J) matrix, jth column (Z_j', Ȳ_j^{K_1}, …, Ȳ_j^{K_M})'

**Inner problem (W given V):**

    W*(V) = argmin_W  (X_1 − X_0·W)' V (X_1 − X_0·W)
            s.t.  w_j ≥ 0 (j = 2,…,J+1),  Σ w_j = 1

where `V` is a `(k×k)` symmetric positive-semidefinite matrix weighting the predictors. The discrepancy norm is `‖X_1 − X_0 W‖_V = sqrt((X_1−X_0W)'V(X_1−X_0W))`.

**Outer problem (choosing V):** "Although our inferential procedures are valid for any choice of `V`, the choice of `V` influences the mean square error." `V` may be chosen subjectively or **data-driven**. The paper's data-driven choice (journal p. 496):
- *(method used in the application)* Choose `V` among **positive-definite diagonal** matrices so that the synthetic control minimizes the **mean squared prediction error of the outcome over the pre-intervention periods** — i.e. `V* = argmin_V Σ_{t≤T0} (Y_1t − Σ_j w_j*(V)·Y_jt)²`. The numerical details are referenced to Abadie & Gardeazabal (2003, App. B).
- *(alternative)* **Cross-validation:** split the pre-period into a training period and a validation period; compute `W*(V)` on training data, then choose `V` to minimize the MSPE produced by `W*(V)` over the validation period.

One obvious predictor choice is to use **all** pre-period outcomes `Y_i1, …, Y_iT0` as the `Ȳ_i^{K_m}` (journal p. 496).

*Standard errors / inference (Section 2.4 + Section 3.4):*
- **No analytical standard error.** Large-sample inference is "not well suited" to comparative case studies with few units. The paper proposes **exact, permutation-style ("placebo") inference** valid regardless of the number of comparison units, periods, or aggregation level (journal pp. 496–497).
- **In-space placebo (permutation) test (journal pp. 501–503):** iteratively reassign the intervention to *each* donor unit, re-estimate the synthetic control, and obtain that unit's post-period gap. This yields a distribution of placebo gaps; the treated unit's effect is "significant" if its gap is unusually large relative to the placebo distribution.
- **RMSPE-ratio test statistic (preferred; journal p. 503):** for each unit compute

      ratio = (post-period MSPE) / (pre-period MSPE),
      where MSPE over a window = average of squared gaps (Y_unit,t − synthetic_unit,t)² over that window.

  Rank the treated unit's ratio among all `J+1` units. The exact permutation p-value is `rank / (J+1)`. For California the ratio is ≈130× the next, the largest of all 39 units, giving **p = 1/39 = 0.026** (the only formal "significance" number in the paper). The ratio normalizes by pre-period fit, which **obviates choosing a pre-fit cutoff** for excluding ill-fitting placebos.
- **Pre-fit filtering (robustness display, not the primary test; journal p. 502):** alternative placebo plots discard donors whose pre-period MSPE exceeds 20× / 5× / 2× the treated unit's (Figures 5–7). The RMSPE-ratio test makes this filtering unnecessary.
- **In-time placebo:** mentioned as a related falsification idea — set the intervention date at random in the pre-period (citing Bertrand-Duflo-Mullainathan 2004; Heckman-Hotz 1989; journal p. 497) — but **no detailed procedure is given in this paper** (see the ADH 2015 review).

*Edge cases:*
- Treated unit's pre-period vector far from the donor convex hull → poor fit → **do not use SCM** (journal p. 495).
- Highly nonlinear outcome–predictor relationship with wide predictor support → severe interpolation bias → restrict donor pool, or add penalty terms to `‖X_1 − X_0W‖` (journal p. 496).
- A predictor with near-zero `V` diagonal element has little predictive power for the pre-period outcome (in the application, log GDP per capita got a very small weight; journal p. 500).
- Placebo unit with poor pre-period fit produces a large post-period gap for the wrong reason → handle via the **RMSPE ratio** (normalizes by pre-fit) rather than raw gap (journal p. 502).
- Donor that itself experienced a similar intervention/shock → exclude from donor pool (journal pp. 498–499).

*Algorithm (reconstructed from Sections 2.3–2.4 and Section 3):*
1. Build the donor pool (curate out contaminated/treated-like units) and the predictor set: covariates `Z` plus `M` linear combinations of pre-period outcomes (commonly some pre-period outcome averages and/or all pre-period outcomes).
2. Form `X_1` (treated predictors) and `X_0` (donor predictors).
3. **Inner:** for a candidate `V`, solve `W*(V) = argmin (X_1 − X_0W)'V(X_1 − X_0W)` over the unit simplex.
4. **Outer:** choose diagonal PSD `V*` minimizing pre-period outcome MSPE of `W*(V)` (or via train/validation cross-validation).
5. Counterfactual `Ŷ_1t^N = Σ_j w_j*(V*)·Y_jt`; effect path `α̂_1t = Y_1t − Ŷ_1t^N` for `t > T0`.
6. **Inference:** repeat steps 2–5 treating each donor as the pseudo-treated unit; compute each unit's post/pre MSPE ratio; the treated unit's permutation p-value is its rank among all `J+1` ratios divided by `J+1`.

**Reference implementation(s):**
- The authors' `Synth` package for **MATLAB, R, and Stata** (companion software, journal pp. 493–494). (R: `Synth::synth()`.)

**Requirements checklist:**
- [ ] Weights on the unit simplex (`w_j ≥ 0`, `Σ w_j = 1`); one treated unit, block assignment.
- [ ] Predictor matrix `X_1`/`X_0` = covariates `Z` + `M` linear combinations of pre-period outcomes (support "all pre-period outcomes" as a choice).
- [ ] Inner solve `W*(V)` = simplex-constrained weighted least squares with predictor-importance matrix `V` (diagonal PSD).
- [ ] Outer `V` selection: pre-period-MSPE minimization (default) and/or train/validation cross-validation; allow user-supplied `V`.
- [ ] Effect = gap path `Y_1t − Σ w_j* Y_jt` for post periods; report pre-period RMSPE (fit diagnostic) and predictor-balance table.
- [ ] In-space placebo permutation inference + post/pre RMSPE-ratio p-value (`rank/(J+1)`); pre-fit-filtered placebo plots as robustness.
- [ ] No analytical SE — inference is permutation/placebo only.

---

## Implementation Notes

### Data Structure Requirements
- Balanced panel: outcome `Y_it` for all units `i = 1, …, J+1` over all periods `t = 1, …, T`; exactly **one** treated unit with **block** (absorbing, common-date) treatment after `T0`.
- Time-invariant covariates `Z_i` (not affected by the intervention) and the pre-period outcome series (for the `Ȳ^{K}` predictors).
- Donor pool explicitly curated (analyst-supplied exclusions).

### Computational Considerations
- Two-level optimization: an inner simplex-constrained quadratic program `W*(V)` nested inside an outer search over diagonal `V`. The outer objective is non-smooth in `V` (the inner argmin has kinks where the simplex active set changes); the paper references AG (2003, App. B) for numerics and does not specify the optimizer.
- Inference cost: the in-space placebo loop re-runs the full nested estimation once per donor (`J` extra fits).
- Aggregate-level data suffice; no micro-data needed (a stated advantage, journal p. 497).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| Donor pool | set of units | all eligible controls | Analyst curation (exclude treated-like / shocked units) |
| Predictors `X` (covariates `Z` + `Ȳ^{K_m}`) | matrix | covariates + pre-period outcome summaries; or all pre-period outcomes | Analyst choice of predictive variables |
| `V` (predictor-importance matrix) | diagonal PSD `k×k` | data-driven | Minimize pre-period outcome MSPE of `W*(V)`; or train/validation cross-validation; or user-supplied |
| `T0` (pre/post split) | period index | intervention date | Set to first period outcome may react (anticipation guard) |

### Relation to Existing diff-diff Estimators
- **`SyntheticDiD` (Arkhangelsky et al. 2021)** is the closest existing estimator: it uses unit *and* time weights with ridge regularization and a double-difference estimator; classic SCM uses **only donor (unit) weights** and a level-matching estimator (no time weights, no ridge). Equation (1) here shows classic SCM **generalizes DiD** (recovered when `λ_t` is constant).
- The inner simplex solve is the same shape as the Frank-Wolfe weight problem already in `diff_diff/utils.py` (`_sc_weight_fw`) once `V^½` is folded into the predictor matrix — but classic SCM adds the **outer `V` search**, which SyntheticDiD has no analog for.
- The placebo/permutation inference resembles SyntheticDiD's `variance_method="placebo"` in spirit, but the **post/pre RMSPE-ratio statistic** and the `rank/(J+1)` p-value are specific to this paper.

---

## Gaps and Uncertainties

- **V-optimization numerics are not in this paper.** Section 2.3 (journal p. 496) describes the outer objective (minimize pre-period outcome MSPE over diagonal PSD `V`) and a CV alternative, but defers the numerical details to **Abadie & Gardeazabal (2003), Appendix B** and the `Synth` software. The exact optimizer, starting values, and any normalization of `V` must be pinned from the `Synth` source / AG 2003 at implementation time, not from this paper.
- **Outer-objective norm.** The inner discrepancy uses `‖·‖_V`; the *outer* `V`-selection minimizes the plain (unweighted) pre-period outcome MSPE. The paper is explicit that inferential validity holds for *any* `V`, so the outer choice is an efficiency device, not an identification requirement (journal p. 496).
- **p-value granularity.** The permutation p-value is `rank/(J+1)`; with a small donor pool the smallest attainable p-value is `1/(J+1)` (here `1/39 = 0.026`). No confidence intervals are produced (a separate inference layer — conformal — is reviewed via CWZ 2021).
- **In-time placebos** are mentioned (journal p. 497) but not proceduralized here; the leave-one-out donor-robustness diagnostic is **absent** from this paper (both belong to the ADH 2015 review).
- **Cross-validation `V`** is described but **not** the method used in the Prop 99 application (which minimized pre-period MSPE directly); the paper does not give a default train/validation split.
- **Penalty-augmented weights** ("`‖X_1 − X_0W‖` plus penalty terms") are mentioned for interpolation-bias control (journal p. 496) but not formalized into a specific penalty (this anticipates later penalized-SC work, out of scope here).
