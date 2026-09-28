# Paper Review: On the Variances of Asymptotically Normal Estimators from Complex Surveys

**Authors:** David A. Binder (Institutional & Agriculture Survey Methods Division, Statistics Canada)
**Citation:** Binder, D. A. (1983). On the Variances of Asymptotically Normal Estimators from Complex Surveys. *International Statistical Review / Revue Internationale de Statistique*, 51(3), 279-292.
**PDF reviewed:** ISR published version (DOI: 10.2307/1402588; JSTOR stable URL https://www.jstor.org/stable/1402588), 15 PDF pages = JSTOR cover + journal pp. 279-292. Local PDFs are gitignored under `/papers/` (`papers/Binder-VariancesAsymptoticallyNormal-1983.pdf`); the journal/DOI version is the authoritative source.
**Review date:** 2026-06-27

---

## Methodology Registry Entry

*This paper is the canonical source for the design-based Taylor-series-linearization (TSL) variance used on the library's survey path. It is a variance-estimation methodology, not a treatment-effect estimator; the template's DiD-specific slots are filled with the paper's analogous content. Maps to `docs/methodology/REGISTRY.md` `## Survey Data Support` -> "Taylor Series Linearization (TSL) Variance".*

## Binder (1983) Design-Based TSL Variance

**Primary source:** Binder, D. A. (1983), *International Statistical Review* 51(3), 279-292. DOI 10.2307/1402588.

**Scope (Section 1, Summary):** The problem of specifying and estimating the variance of estimated parameters based on complex sample designs from **finite populations**. The results are "particularly useful when the parameter estimators cannot be defined explicitly as a function of other statistics from the sample" — i.e. *implicitly defined* parameters (regression, logistic regression, log-linear contingency-table models). The sampling distribution of interest is the **design-based** distribution (randomness from sampling), not a superpopulation/model-based one (Section 1: the finite-population parameter `B` is "defined in terms of the finite population values"; "we are only concerned with the design-based sampling distribution of `B̂`").

**Key implementation requirements:**

*Assumption checks / warnings (Section 3.1, p.282; formalized as Conditions 1-11 in the Appendix):*
- **(a)** A parameter space `Θ` containing a neighbourhood of the parameter of interest `θ₀` (Appendix Conditions 3-4).
- **(b)** A sequence of sample designs/populations that admits **asymptotically normal estimators of population totals** and **consistent estimators for the variance of the estimated totals** (Appendix Condition 5; this is the load-bearing design assumption — the whole method reduces variance estimation of `θ̂` to variance estimation of a *total*).
- **(c)** Continuity and limiting conditions on `W_N(θ)` and its partial derivatives (Appendix Conditions 1, 6, 7, 9, 10).
- **(d)** A continuity condition on the variance of the estimated total (Appendix Condition 11: a consistent estimator `Φ̂(θ)` for `Φ(θ)` exists under the sample design).
- **Full-rank requirement:** the estimating-function Jacobian `∂W_N/∂θ` must be of full rank for the sandwich (3.3) to be invertible (stated below 3.3, p.283; Appendix Condition 8 requires `J(θ₀)Φ(θ₀)J(θ₀)` full rank). For OLS this is the usual `X` full-column-rank condition (Section 2.1).
- The method depends on **asymptotic normality of the estimators**; Section 6 (Discussion) explicitly warns that "many standard statistical packages may be used for the estimation of the parameters ... but the variances and tests of hypotheses given in these packages will not be valid" under complex designs, and that "empirical studies on the validity of these approximations are important."

*Parameter definition — implicit estimating equations (Equations 2.5-2.6, the general form):*

    Σ_{k=1}^{N} u(Z_k; θ) - v(θ) = 0          (general, 2.5 -> 2.6: W_N(θ) = Σ u(Z_k; θ) - v(θ) = 0)

where:
- `Z_k = (Z_{1k}, ..., Z_{qk})ᵀ` = the `q`-dimensional data vector for the `k`-th unit of the finite population (`k = 1, ..., N`).
- `u(·)` = a `p`-dimensional vector-valued **estimating function** of the data `(Y_k, X_k)` and the parameter `θ` (= `B` for regression).
- `v(θ)` = a term allowing **explicitly defined** parameters; "for a given `θ` we know the value of `v(θ)`, but `u(Z_k; θ)` is only known for those units in the sample" (p.281) — this distinction drives the estimation in Section 3.
- `θ = (θ₁, ..., θ_p)` = the finite-population parameter, defined purely in terms of the finite-population values (Section 2.3).

*Regression specialization (Equation 2.1):* `B` is defined **implicitly** by `XᵀXB = XᵀY` (rather than explicitly as `B = (XᵀX)⁻¹XᵀY`); the implicit definition "gives more tractable results for the variances of the estimated regression coefficients" (Section 2.1). `X` is `N×p`, `Y` is `N×1`, `X` of full rank. This view follows Frankel (1971), Kish & Frankel (1974).

*Generalized linear model specialization (Equations 2.2-2.4):* density `p(y; θ, φ) = exp[α(φ){yθ - g(θ) + h(y)} + γ(φ, y)]`, with `E(Y) = g'(θ) = μ(θ)`, `V(Y) = μ'(θ)/α(φ)`, and `θ = f(Σ x_i β_i)`. The finite-population parameter `B` solves

    Σ_{k=1}^{N} [Y_k - μ{f(X_kᵀB)}] f'(X_kᵀB) X_{ik} = 0   (i = 1, ..., p)   (2.4)

unique when `μ(·)` and `f'(·)` are strictly monotone and `X` is full rank. Covers normal linear regression, probit, logit, log-linear contingency tables, variance-components.

*Estimator (Section 3.2, Equations 3.1-3.2):*
- `Û(θ)` = the survey estimator of the total `U(θ) = Σ_{k=1}^{N} u(Z_k; θ)`, assumed **asymptotically normal** with mean `U(θ)` and variance `Σ_U(θ)`, with a consistent variance estimator `Σ̂_U(θ)`.
- `Ŵ(θ) = Û(θ) - v(θ)`   (3.1)
- `θ̂` is defined by solving `Ŵ(θ̂) = 0`   (3.2).
- Linear-regression instance: `u(y_k, x_k; B) = -(y_k - x_kᵀB)x_k`, `v(B) = 0`, giving `S_XX B̂ - S_XY = 0` with `S_XX`, `S_XY` survey estimators of `XᵀX`, `XᵀY`.

*Variance — the sandwich (Section 3.3, Equations 3.3-3.4; THE central result):*
Taylor-expand `Ŵ(θ̂)` about `θ̂ = θ₀`: `0 = Ŵ(θ̂) ≈ Ŵ(θ₀) + [∂Ŵ(θ₀)/∂θ₀](θ̂ - θ₀)`, so `Ŵ(θ₀) ≈ -[∂Ŵ(θ₀)/∂θ₀](θ̂ - θ₀)`. Taking variances in the limit:

    V(θ̂) = [∂W_N(θ₀)/∂θ₀]⁻¹ Σ_U(θ₀) [∂W_N(θ₀)ᵀ/∂θ₀]⁻¹          (3.3)

with consistent estimator

    V̂(θ̂) = [∂Ŵ(θ̂)/∂θ̂]⁻¹ Σ̂_U(θ̂) [∂Ŵ(θ̂)ᵀ/∂θ̂]⁻¹             (3.4)

This is a **sandwich**: the "bread" is the inverse Jacobian of the estimating function (`∂W/∂θ`), and the **"meat" `Σ_U(θ)` is the design-based variance of the survey-estimated total of the estimating functions `u`** — so any standard survey total-variance estimator (stratified, multistage, with FPC) plugs straight in. Formal justification: Appendix (Lemma 1, Corollaries 1-2).

*Linear-model variance (Section 4.1, Equations 4.1-4.6):* For `f(Y) = Y`, `W_N(B) = Σ[Y_k - μ(X_kᵀB)]X_k` (4.2), and `∂W_N(B)/∂B = -Σ μ'(X_kᵀB)X_k X_kᵀ = -XᵀΛ(B)X` with `Λ(B) = diag[μ'(X_1ᵀB), ..., μ'(X_NᵀB)]`. Hence

    V(B̂) = [XᵀΛ(B)X]⁻¹ Σ(B) [XᵀΛ(B)X]⁻¹                     (4.4)
    V̂(B̂) = [Ĵ⁻¹(B̂)] Σ̂(B̂) [Ĵ⁻¹(B̂)],   Ĵ(B̂) = XᵀΛ(B̂)X        (4.5)

where **`Σ(B)` is the variance of a survey-estimated total based on the score vectors `{e_k x_k}`**, with residual `e_k = y_k - μ(x_kᵀB)` (estimated residual `ê_k = y_k - μ(x_kᵀB̂)`). Newton-Raphson (4.6) reuses `Ĵ(B̂)` as the per-iteration derivative matrix: `B̂^{(i+1)} = B̂^{(i)} - Ĵ⁻¹(B̂^{(i)})[Q̂(B̂^{(i)}) - Ĥ]`.

*OLS regression (Section 4.2, Equation 4.7; the diff-diff anchor):* with `μ(X_kᵀB) = X_kᵀB`, `W_N(B) = XᵀY - XᵀXB`, `∂W_N/∂B = -XᵀX`, so

    V̂(B̂) = S_XX⁻¹ Σ̂(B̂) S_XX⁻¹                              (4.7)

where `S_XX` estimates `XᵀX` and `Σ̂(B̂)` is the **estimated design-variance of the total of the score vectors `{ê_k x_k}`**, `ê_k = y_k - x_kᵀB̂`. "Fuller (1975) has obtained these results for stratified and for two-stage sampling" (p.284). The paper also derives the variance of a derived statistic `R̂²` (coefficient of multiple determination, Equations 4.8-4.11) as a worked example of variances for parameters that are *ratios/functions* of totals via the same `∂W` machinery.

*Logistic regression (Section 4.3):* `μ(X_kᵀB) = exp(X_kᵀB)/[1 + exp(X_kᵀB)]`; the `k`-th diagonal of `Λ(B)` in (4.4) is `μ(X_kᵀB)[1 - μ(X_kᵀB)]`; variance via (4.5).

*Log-linear / categorical models (Section 4.4, Equations 4.12-4.15):* multinomial-logit cell probabilities `p_i(B) = exp(a_iᵀB)/Σ_i exp(a_iᵀB)`; estimator `Ŵ(B̂) = AᵀN̂ - [Aᵀp(B̂)]1ᵀN̂ = 0` (4.13) using a consistent asymptotically-normal estimator `N̂` of category counts with `V̂[N̂]`. Variance (4.14):

    V[B̂] = (Nᵀ1)⁻² (AᵀH(B)A)⁻¹ Aᵀ(I - p(B)1ᵀ) V[N̂] (I - 1p(B)ᵀ) A (AᵀH(B)A)⁻¹   (4.14)

with `D(B) = diag[p(B)]`, `H(B) = D(B) - p(B)p(B)ᵀ`; simplifies to (4.15) when `N/Nᵀ1 ≈ p(B)`. Extends to product-multinomial models with known margins.

*Standard errors / variance — summary:*
- **Default:** analytical TSL sandwich (3.4)/(4.5)/(4.7). The meat is the design-based variance of the **estimated total of the score vectors**, computed under the actual sample design (stratification, multistage, unequal probabilities, FPC).
- **Stratified / two-stage:** Fuller (1975) results plug into `Σ̂(B̂)` (Section 4.2).
- **Resampling alternatives:** the paper frames variance estimation around the design-variance of a total; jackknife / balanced-repeated-replication estimators of that total-variance are admissible (Krewski & Rao 1981 is cited for the linearization/jackknife/BRR comparison and the with-replacement CLT — Section 5.1, Appendix).
- **Clustering level:** the PSU (primary sampling unit) within strata — see the Canada Health Survey design (Section 5.1): with-replacement PSU sampling within strata.

*Edge cases / boundary conditions:*
- **Full rank** of `∂W_N/∂θ` (and of `X` in the linear case) required for the sandwich inverse (Section 2.1; Appendix Condition 8).
- **Unique solution** of the GLM estimating equation requires `μ(·)`, `f'(·)` strictly monotone and `X` full rank (Section 2.3).
- **Validity hinges on asymptotic normality** of the total estimators; the approximation can fail in small samples / sparse cells (Section 6 caveat).
- **Implicit vs explicit parameters:** for parameters expressible explicitly as functions of totals (e.g. ratios within subpopulations), Tepping (1968) and Woodruff (1971) apply directly; Binder's contribution is the *implicitly* defined case (Section 2.1).

*Algorithm (general, from Sections 3.2 + 4.1):*
1. Choose the estimating function `u(Z_k; θ)` defining the finite-population parameter via `W_N(θ) = Σ u(Z_k; θ) - v(θ) = 0` (2.6).
2. Form the survey-weighted estimator `Ŵ(θ) = Û(θ) - v(θ)` (3.1) using the design weights.
3. Solve `Ŵ(θ̂) = 0` for `θ̂` (3.2) — closed form for OLS (`B̂ = S_XX⁻¹ S_XY`), Newton-Raphson (4.6) for GLMs.
4. Form the "bread" `Ĵ(θ̂) = ∂Ŵ(θ̂)/∂θ̂` (`= -XᵀX` for OLS, `= -XᵀΛ(B̂)X` for GLM).
5. Form the "meat" `Σ̂_U(θ̂)` = the **design-based variance estimate of the survey total of the score vectors** `u(Z_k; θ̂)` (`= {ê_k x_k}` for regression), under the actual stratified/multistage design.
6. Assemble `V̂(θ̂) = Ĵ⁻¹ Σ̂_U Ĵ⁻ᵀ` (3.4)/(4.5)/(4.7).

**Reference implementation(s) (as cited in the paper):**
- "Super Carp" — Hidiroglou, Fuller & Hickman (1980), Statistical Laboratory, Iowa State University (cited p.284 for the ratio-of-random-variables variance).
- Prior model-specific treatments: Fuller (1975) (stratified / two-stage regression), Freeman & Koch (1976) (raked contingency tables). Imrey, Koch & Stokes (1981, 1982) functional asymptotic regression is noted as an alternative that "also falls within the general framework" (Section 6).
- *(The modern realizations — R `survey::svyglm`/`svydesign` (Lumley), Stata `svy:`, SUDAAN — postdate this 1983 paper and are noted here only for orientation, not sourced from it.)*

**Requirements checklist (for the diff-diff survey TSL path):**
- [ ] Parameter defined as the solution to survey-weighted estimating equations (regression/GLM), not assumed explicit.
- [ ] Bread = inverse estimating-function Jacobian (`XᵀWX` analog for the weighted case).
- [ ] Meat = design-based variance of the **survey total of the score vectors** `{ê_k x_k}`, computed under strata/PSU/FPC.
- [ ] Sandwich assembled as `bread⁻¹ · meat · bread⁻ᵀ` (Eq. 4.7).
- [ ] Full-rank guard on the bread.
- [ ] Degrees of freedom / critical values consistent with the design (PSU-stratum count); small-sample caveat surfaced.

---

## Implementation Notes

### Data Structure Requirements
- **Finite population** of `N` units, `q`-dimensional data vector `Z_k` per unit; a complex sample design (stratification, multistage clustering, unequal selection probabilities) drawn from it (Section 1).
- Inputs for the regression case: design matrix `X` (`N×p`, full rank), outcome `Y` (`N×1`), and the survey design (strata, PSU identifiers, selection-probability/design weights, optional FPC).
- The method needs, from the design, **(i)** an asymptotically normal estimator of population totals and **(ii)** a consistent estimator of the variance of those estimated totals (Section 3.2 / Appendix Condition 5).

### Computational Considerations
- Closed-form for OLS (`B̂ = S_XX⁻¹ S_XY`); Newton-Raphson for GLMs (Eq. 4.6), reusing `Ĵ(B̂) = XᵀΛ(B̂)X` as both the update Hessian and the variance "bread".
- Variance cost is dominated by computing the design-variance of the score-vector total — `O(p²)` accumulation over PSUs within strata.
- No matrix beyond `p×p` is inverted for the variance (the meat is a `p×p` total-variance, the bread a `p×p` Jacobian).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|------------------|
| Estimating function `u(·)` | functional form | model-implied (OLS / logit / log-linear) | dictated by the parameter being estimated (Sections 2.2-4.4) |
| Score-total variance estimator | design-based method | linearization (TSL) | TSL by default; jackknife / BRR admissible (Krewski & Rao 1981) |
| FPC / with-replacement assumption | design option | with-replacement within strata (paper's CHS example) | from the actual design (Section 5.1) |

### Relation to Existing diff-diff Estimators
- This is the **theoretical foundation for the library's survey TSL variance path** (`diff_diff/survey.py`; `docs/methodology/REGISTRY.md` `## Survey Data Support` -> "Taylor Series Linearization (TSL) Variance"; `docs/methodology/survey-theory.md` §4.4-§5). Equation (4.7) `V̂(B̂) = S_XX⁻¹ Σ̂(B̂) S_XX⁻¹` is precisely the diff-diff regression-based TSL sandwich, with the meat `Σ̂(B̂)` realized as the stratified PSU-level variance of the score-vector total.
- The "variance of `θ̂` reduces to the variance of a *total* of influence/score contributions" result (Eq. 3.3) is the same mechanism used by the library's **influence-function-based** survey variance for the modern DiD estimators (Callaway-Sant'Anna, Sun-Abraham, ImputationDiD, TwoStageDiD, etc.): each estimator's IF plays the role of `u(Z_k; θ)`, and design-consistency follows whenever the IF satisfies Binder's smoothness conditions (a)-(d).
- The GLM specialization (4.5, `Λ` diagonal) underlies survey-weighted logit/Poisson DiD (WooldridgeDiD QMLE path).

---

## Gaps and Uncertainties

- **Degrees of freedom / small-sample inference is not prescribed.** The paper establishes asymptotic normality (Appendix Corollaries 1-2) but gives no finite-sample df rule; the `df = n_PSU − n_strata` convention used downstream comes from later work (Korn & Graubard 1990), not Binder. The Discussion (Section 6) only flags that "empirical studies on the validity of these approximations are important."
- **Lonely/singleton-PSU handling is not addressed.** The Canada Health Survey example assumes 2-4 PSUs per stratum sampled with replacement (Section 5.1); the paper does not treat strata with a single sampled PSU. Singleton-PSU conventions (`remove`/`certainty`/`adjust`) are downstream/practitioner choices.
- **FPC appears only implicitly.** The CHS example assumes with-replacement PSU selection (so no FPC); the general meat `Σ_U(θ)` is "the variance of a total" and inherits whatever FPC the design's total-variance estimator carries — the paper does not write the `(1 − f_h)` factor explicitly.
- **Weight-type semantics (pweight vs aweight vs fweight) are not distinguished.** Binder works with design (selection-probability) weights via the survey total `Û(θ)`; analytic/frequency-weight conventions are outside the paper.
- **The Appendix proof leans on cited regularity results.** Condition 5's asymptotic normality of the standardized estimating function is asserted to hold "in the literature" (Madow 1948; Hájek 1960, 1964; Rosen 1972; von Bahr 1972; Fuller 1975; Krewski & Rao 1981); the paper does not reprove the design CLT, so the validity in any specific design rests on those references (p.291).
- **Notation note:** in (4.3) and the worked `R̂²` example the paper uses `Ĥ`, `Q̂` for the survey-estimated totals `Σ Y_k X_k` and `Σ μ(X_kᵀB)X_k`; these are transcribed as-is. In the log-linear variance (4.14)/(4.15), every superscript `ᵀ` denotes a matrix **transpose** — there is no separate `T` matrix or operator: `Nᵀ1` is the scalar population total (`= 1ᵀN`, consistent with the derivative `∂W/∂B = −(1ᵀN)AᵀH(B)A` on p.285), and `Aᵀ(I − p(B)1ᵀ)` is `A`-transpose times the `(I − p(B)1ᵀ)` projector.
