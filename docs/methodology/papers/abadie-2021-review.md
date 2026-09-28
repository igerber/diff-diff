# Paper Review: Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects

**Authors:** Alberto Abadie
**Citation:** Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391–425.
**PDF reviewed:** https://doi.org/10.1257/jel.20191450 (published JEL version)
**Review date:** 2026-05-29

> Scope note: this is a **practical-guide / review article**. It recaps the synthetic-control estimator (attributed to Abadie & Gardeazabal 2003 and ADH 2010/2015) and contributes a synthesis on **feasibility, data requirements, contextual requirements, and inference**, plus a survey of extensions. Where it surveys other methods (Chernozhukov-Wüthrich-Zhu conformal inference; Arkhangelsky et al. synthetic DiD; Abadie-L'Hour / Ben-Michael et al. penalized & bias-corrected SC; Doudchenko-Imbens; Athey et al. matrix completion), those are **citations** — captured here only as Abadie frames them. The dedicated CWZ 2021 review is authoritative for conformal inference; the others are out of scope for this initiative. Nothing here is sourced from outside this paper.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md. This is the richest source for the `## SyntheticControl` **assumption / warning** and **edge-case** sections.*

## SyntheticControl

**Primary source (this document):** Abadie, A. (2021). "Using Synthetic Controls…" *JEL*, 59(2), 391–425. https://doi.org/10.1257/jel.20191450

**Key implementation requirements:**

*Notation (Section 3.1):*
- `J+1` units, `j=1` treated, donors `j=2,…,J+1`; `T` periods, first `T0` pre-intervention. `Y_jt` observed; `Ŷ^N_jt` synthetic prediction of the untreated potential outcome. `X_1` `(k×1)` treated-unit predictors (may include pre-period outcomes); `X_0` `(k×J)` donor predictors. `Z_j` observed covariates; `μ_j` unobserved factor loadings.

*Target and estimator (Equations 1–3, 7–8):*

    (1)  τ_{1t} = Y^I_{1t} − Y^N_{1t}            (t > T0)
    (2)  Ŷ^N_{1t} = Σ_{j=2}^{J+1} w_j · Y_jt
    (3)/(8)  τ̂_{1t} = Y_{1t} − Σ_{j=2}^{J+1} w_j*·Y_jt

    (7)  W* = argmin_W ( Σ_{h=1}^{k} v_h·(X_{h1} − Σ_{j} w_j·X_{hj})^2 )^{1/2}
             s.t.  w_j ≥ 0,  Σ w_j = 1            ("constrained quadratic optimization")

Footnote 8: assumptions are on `Y^N` only; since `Y_{1t}=Y^I_{1t}` is observed for `t>T0`, **no assumptions on the process generating `Y^I` are needed**. Equation (1) lets the effect vary freely over time. Special cases: equal weights `w_j=1/J` (4), population weights (5), single nearest neighbor `w_m=1` (6).

*The justifying model and the identifying condition (Section 3.3):*

    (10)  Y^N_{jt} = δ_t + θ_t·Z_j + λ_t·μ_j + ε_jt        (linear factor / interactive-FE model)

- **Generalizes DiD/TWFE:** restricting `λ_t = λ` (time-invariant) recovers parallel trends; the factor model relaxes this by letting loadings on `μ_j` vary in time (Bai 2009 cited).
- **Identifying condition:** if `X_1 = X_0 W*` (the synthetic control reproduces the treated unit's predictors **including pre-period outcomes**), then `τ̂_{1t}` is unbiased under (10). `μ_1` is unobserved and cannot be matched directly; a good pre-period-outcome match approximates it **only when the transitory-shock scale is small or `T0` is large**. A small `T0` with enough shock variation can produce a spurious pre-period match → **overfitting / bias**.
- **Bias bound (cited to ADH 2010):** bias is bounded by a function **inversely proportional to `T0`**, *provided the pre-period fit is good*. "**A large `T0` cannot drive down the bias if the fit is bad.**" The bound **increases with `J`** (donor-pool size) and with the **number of unobserved factors** (components of `μ_j`).

*Feasibility / convex hull (Sections 3.3, 5):*
- In practice `X_1 = X_0 W*` is replaced by `X_1 ≈ X_0 W*`; **there are no ex-ante guarantees** on the size of `X_1 − X_0 W*`. When it is large, ADH 2010 recommend **against** using synthetic controls (potential for substantial bias).
- The treated unit's predictor point `(X_{11},…,X_{k1})` must fall **close to the convex hull** of the donors' points. If the treated unit is **"extreme"** in some predictor (or in pre-period outcomes), no weighted average reproduces it → "the conventional synthetic control estimator should not be used in that case."
- The simplex constraint **prevents extrapolation** but **not interpolation bias**: averaging away large discrepancies between dissimilar donors biases the estimate → **restrict the donor pool to similar units**.

*`V` (predictor-importance) selection (Section 3.2; this paper formalizes the options):*
- **(a) Inverse-variance:** set `v_h = 1/Var(X_{h·})` (rescales each predictor row to unit variance).
- **(b) Nested MSPE minimization (AG 2003 / ADH 2010):** choose `V` so `W(V)` minimizes pre-period outcome MSPE `Σ_{t∈𝒯0} (Y_{1t} − Σ_j w_j(V)·Y_jt)²` over a set `𝒯0 ⊆ {1,…,T0}`.
- **(c) Out-of-sample cross-validation (ADH 2015), formalized 4-step (Equation 9):** split pre-period into training `1..t0` and validation `t0+1..T0` (concretely `t0 = T0/2`); compute `W̃(V)` on training data; pick `V*` minimizing validation MSPE (9); recompute `W* = W(V*)` using the validation-window predictors.
- **Footnote 7 (non-uniqueness):** CV weights need not be unique; can add a ridge-type penalty `γ·Σ_h v_h²` (`γ>0`) favoring dense weights. Demonstrate robustness to the `V` choice (Klößner et al. 2018 cited).

*Predictor / variable selection (Section 3.4):*
- Predictors typically combine **pre-period outcomes** (crucial for matching `μ_j`; arise organically under a VAR DGP) **and** other covariates `Z_j`. Covariates omitted from `Z_j` are "mechanically absorbed into `μ_j`," increasing the bias bound — so **include real covariates**, don't rely on lagged outcomes alone.
- Flexibility: need not use every pre-period outcome; a **summary** (e.g., a pre-period mean) can suffice when outcomes co-move, and **increases weight sparsity** (number of nonzero `w_j` is controlled by the number of predictors).
- **Post-intervention outcomes are NOT used** to compute weights → weights are a **design-phase** object (safeguard against specification search / p-hacking; can be pre-registered).

*Standard errors / inference (Sections 3.5, 8):*
- **No SEs in the classical sense.** Inference is **permutation / placebo-based** (design-based, conditioning on the sample), **not** sampling-based. Rationale: small / single treated unit, no randomization, sample often = population.
- **RMSPE-ratio permutation test (Equations 11–12):**

      (11)  R_j(t1,t2) = ( (1/(t2−t1+1)) · Σ_{t=t1}^{t2} (Y_jt − Ŷ^N_jt)^2 )^{1/2}     (RMSPE for unit j)
      (12)  r_j = R_j(T0+1, T) / R_j(1, T0)                                            (post/pre ratio)

  `Ŷ^N_jt` is the synthetic control built treating unit `j` as treated (other `J` units as donors). p-value:

      p = (1/(J+1)) · Σ_{j=1}^{J+1} 𝟙₊(r_j − r_1)        (fraction of units with ratio ≥ the treated unit's r_1)

  Alternative: use the distribution of post-period `R_j(T0+1,T)` after discarding placebos with pre-period `R_j(1,T0)` ≫ `R_1(1,T0)`.
- **Confidence intervals by test inversion** (Firpo & Possebom 2018 cited) — invert the permutation test over hypothesized effect values.
- **One-sided tests** via positive/negative parts `(Y_jt − Ŷ^N_jt)^±` of the gap → power gain (treated-unit-contaminated placebos tend to produce opposite-sign effects).
- **Visualize** the permutation distribution of `r_j` or of placebo gaps `Y_jt − Ŷ^N_jt` (conveys magnitude, not just a p-value).
- **Surveyed alternatives (citations — see dedicated reviews):** Chernozhukov-Wüthrich-Zhu (2021) **conformal inference** (time-permutation of constrained-LS residuals under the null, valid under residual **exchangeability**, weights re-estimated under the null using all periods); CWZ (2019b) bias-corrected CIs (asymptotically pivotal t-stat + cross-fitting, large `T0` and `T−T0`); Cattaneo-Feng-Titiunik **predictive intervals** (estimation + irreducible-error uncertainty); Hahn-Shi / Andrews (2003) **end-of-sample instability** test.

*Edge cases / contextual requirements (Section 5 — the failure modes):*
- **Effect size vs. volatility:** small effects are masked by volatile outcomes; high *unit-specific* volatility raises overfitting risk → consider de-noising/filtering (only unit-specific noise hurts; common-factor volatility is differenced out by the SC).
- **No suitable comparison group:** exclude donors that (i) adopted a similar intervention, or (ii) suffered large idiosyncratic shocks not shared by the treated unit; restrict to comparable units (interpolation-bias control).
- **Anticipation:** if agents react before formal implementation, **backdate** the intervention. Backdating does **not** mechanically bias the estimator because (1)/(3) allow time-varying effects (unlike constant-effect panel models).
- **Interference / spillovers (SUTVA, Rubin 1980):** enforce in design (drop possibly-affected donors) or reason about the **sign of the bias** (e.g., negative spillover onto contributing donors → estimate is a *lower bound*). Sparsity + transparency of weights makes this feasible.
- **Outcome transformations & a differencing pitfall:** level mismatch can be handled via differences, growth rates, or **demeaning** `Ȳ_jt = Y_jt − (1/T0)Σ_{h≤T0} Y_jh` (≡ Doudchenko-Imbens constant shift). **But** differencing inflates the noise variance when `ε_jt` is roughly independent in time → higher overfitting/bias; the differenced model retains the factor structure `ΔY^N_jt = Δδ_t + Δθ_t Z_j + Δλ_t μ_j + Δε_jt`.
- **Short pre-period:** spurious (near-)perfect fit → unreliable counterfactual; mitigate with powerful non-outcome predictors (reduce residual variance).
- **Structural breaks:** a long `T0` risks violating constant-factor-loadings; up-weight (`v_h`) the most recent predictors to alleviate.
- **Time horizon:** effects may emerge slowly → need enough post-periods, or surrogate/leading indicators.

*Sparsity (Section 4):* synthetic-control weights are **sparse** — when `X_1` is outside the donor convex hull and donors are in "general position," the solution is **unique with ≤ `k` nonzero weights** (projection of `X_1` onto the hull). Sparsity here is for **interpretability** (the identity/magnitude of nonzero weights matters), unlike lasso where sparsity is an anti-overfitting device. With many treated units inside the hull, weights may be non-unique (penalized SC restores uniqueness).

**Reference implementation(s):**
- Authors' `Synth` package for **R, MATLAB, and Stata** (Section 3.2 footnote; documented in Abadie, Diamond & Hainmueller 2011, *J. Stat. Software* 42(13)).

**Requirements checklist (guidance this paper adds beyond 2010/2015):**
- [ ] Convex-hull / "extreme treated unit" guard → warn / refuse when pre-period fit is poor or the treated unit is extreme.
- [ ] `V`-selection: inverse-variance, nested-MSPE, and CV (with a documented `t0=T0/2`-style default + optional ridge `γΣv_h²` for non-uniqueness).
- [ ] Encourage covariates in addition to lagged outcomes; allow pre-period-outcome summaries (sparsity).
- [ ] Permutation inference: RMSPE-ratio p-value `(#{r_j≥r_1})/(J+1)`; one-sided variants; CI by test inversion; placebo-distribution visualization.
- [ ] Weights computed from **pre-intervention data only** (design-phase guarantee).
- [ ] Diagnostics: in-time placebo / backdating, leave-one-out, donor-pool & predictor robustness.
- [ ] Warnings for the failure modes (volatility, contamination, anticipation, interference, differencing, short pre-period, structural breaks).

---

## Implementation Notes

### Data Structure Requirements
- Aggregate panel (outcome + predictors) for the treated unit and a curated donor pool; **large pre-intervention window**; enough post-periods for the effect to manifest; balanced panel; single (or few) treated units with block timing.

### Computational Considerations
- Inner weight solve = constrained quadratic optimization over the simplex (Section 3.2 names it as such).
- `V` selection adds an outer loop (nested-MSPE or CV-validation evaluation). Permutation inference re-runs estimation `J` times (one pseudo-treated donor each).

### Tuning Parameters

| Parameter | Type | Default guidance (this paper) | Selection Method |
|-----------|------|-------------------------------|------------------|
| `V` (predictor importance) | nonneg vector | data-driven | inverse-variance; nested pre-period MSPE; or CV (`t0=T0/2`); optional ridge `γΣv_h²` for non-uniqueness |
| Predictors `X` | matrix | lagged outcomes + covariates | include real covariates; outcome summaries increase sparsity; data-driven via train/validation |
| Donor pool | set | curated, similar units | exclude treated-like / shocked / dissimilar units; limit size (overfitting) |
| Pre/post window | indices | as long a pre-window as structurally stable | backdate under anticipation; up-weight recent predictors under break risk |

### Relation to Existing diff-diff Estimators
- Same `SyntheticControl` estimator as the 2010/2015 reviews. This paper is the source for the **assumptions/warnings** and **edge-case** REGISTRY content and for the **formalized CV `V`-selection** (`t0=T0/2`) and the **CI-by-test-inversion / one-sided** inference refinements (relevant to PR-2/PR-3).
- It positions **synthetic DiD (Arkhangelsky et al.)** — already implemented as `SyntheticDiD` — as "an SC that additionally weights pre-intervention time periods," confirming classic SCM is the unit-weights-only special case.
- It positions **conformal inference (CWZ)** as the sampling-based complement to permutation inference — the basis for PR-3 (authoritative details in the CWZ review).

---

## Gaps and Uncertainties

- **No new estimator/algorithm numerics.** The inner solver, `V`-search routine, and starting values are not specified (referenced to AG 2003 / ADH 2010 and the `Synth` software). The CV `t0=T0/2` split is explicitly "heuristic."
- **CV-weight non-uniqueness** is acknowledged (footnote 7) with a ridge remedy `γΣv_h²` but no default `γ`; an implementation must pick a deterministic tie-break.
- **Surveyed inference methods are citation-level here.** The conformal recipe (CWZ), predictive intervals (Cattaneo et al.), and bias-corrected CIs (CWZ 2019b) are summarized but their exact algorithms/assumptions must come from the primary papers (CWZ 2021 is reviewed separately; the others are out of scope).
- **Multiple treated units, penalized SC, bias correction, matrix completion** (Section 8) are surveyed (Eqs. 13–18 transcribed as Abadie presents them) but are **deferred** (augmented SC) or out of scope; not part of the classic-SCM implementation.
- **Effect-size/volatility de-noising** (singular-value thresholding, Amjad-Shah-Shen) is mentioned as mitigation but not prescribed — a judgment call left to the analyst.
- **"Extreme treated unit" / convex-hull check** is qualitative ("falls close to the convex hull") — a concrete numerical hull-distance or fit threshold for a warning must be chosen at implementation.
