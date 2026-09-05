# Paper Review: Difference-in-differences with "bad controls"

**Authors:** Carolina Caetano, Brantly Callaway, Stroud Payne, Hugo Sant'Anna
**Citation:** Caetano, C., Callaway, B., Payne, S., & Sant'Anna, H. (2026). Difference-in-differences with "bad controls". arXiv preprint arXiv:2608.03881. https://arxiv.org/abs/2608.03881
**PDF reviewed:** https://arxiv.org/pdf/2608.03881v2 (arXiv v2, 2 Sep 2026; title page dated September 3, 2026; 42 pages; SHA-256 `fbaef047fb0552f79eaea33f55405795f6c8d949fe50fe2d4c8908a51b2e02f4`) plus the Supplementary Appendix https://bcallaway11.github.io/files/badcontrols/CCPS_2026_SA_v1.pdf (SA v1; 23 pages; SHA-256 `15c7dafe2cae21ed4c3316bc73cba240d089d1b00ab2023a5a3910c9d95b48e5`)
**Review date:** 2026-09-05

> **Preprint status.** This is an unpublished arXiv preprint (v2, 2 Sep 2026; title page
> dated September 3, 2026) with a separately hosted Supplementary Appendix (CCPS_2026_SA_v1,
> dated September 1, 2026). The library's usual rule is to source methodology from published
> papers; this review is an explicit exception. **All equation, assumption, condition,
> theorem, proposition, remark, lemma, algorithm, table, figure, and page references below
> are pinned to arXiv v2 (main text, "p. N") and SA v1 ("S"-prefixed items, "SA p. N").**
> The title-page footnote (p. 1) states that some results originally appeared in
> "Difference-in-differences with time-varying covariates" (Caetano, Callaway, Payne, and
> Sant'Anna 2022) and that this paper, together with the companion
> "Difference-in-differences when parallel trends holds conditional on covariates" (Caetano
> and Callaway 2025), replaces that working paper. Author disambiguation: "Sant'Anna" here is
> **Hugo Sant'Anna** (UAB; the `hugosantanna` GitHub account hosts the R package), NOT Pedro
> H. C. Sant'Anna of Callaway and Sant'Anna (2021) and Sant'Anna and Zhao (2020), both of
> which this paper builds on. Re-check every number against any later arXiv version or a
> published version before citing in user-facing docs.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## BadControlsDiD` section into the "Modern Staggered Estimators" category of the registry once the estimator ships (rename to the shipped class name).*

## BadControlsDiD (prospective; name provisional)

**Primary source:** Caetano, C., Callaway, B., Payne, S., & Sant'Anna, H. (2026). Difference-in-differences with "bad controls". arXiv:2608.03881v2 + Supplementary Appendix CCPS_2026_SA_v1 (numbering below follows those versions; unpublished preprint - see the status note above).

The paper studies DiD when conditional parallel trends holds only after conditioning on a
time-varying covariate `X_t` that is itself affected by the treatment - a **bad control**.
With two periods `t* - 1` (nobody treated) and `t*` (some units treated), treatment `D`,
outcome `Y_t`, bad control `X_t` with potential versions `X_t(0)`, `X_t(1)`, exogenous
covariates `Z`, and `ΔY_{t*} := Y_{t*} - Y_{t*-1}`, a covariate is a bad control when it
satisfies both **Condition 1 (Outcome Relevance)** and **Condition 2 (Affected by the
Treatment)** (**Definition 1**, p. 6). Under conditional parallel trends (Assumption 2) the
ATT can be written as `E[ΔY_{t*} | D = 1] - E[m_0(X_{t*}(0), X_{t*-1}, Z) | D = 1]`
(Equation 2, p. 7), but this is NOT an identification result: the average is over the
distribution `F̃_1` of the **untreated potential** bad control `X_{t*}(0)` in the treated
group, which is never observed (Equation 3, p. 8). Section 3 shows that both traditional
practices are generally biased when `X_t` is a genuine bad control (the paper's own prose
says "bias whenever", pp. 9 and 11): including the post-treatment `X_{t*}` as a covariate
(`τ^use`, Section 3.1, p. 9) has zero bias when Condition 2 fails, and dropping the
covariate (`τ^discard`, Section 3.2, pp. 10-11) has zero bias when Condition 1 fails.
Strictly, each displayed bias is an integral that could also vanish by moment cancellation
with the conditions still holding, so failure of Condition 2 / Condition 1 is the paper's
sufficient benchmark case for zero bias (Figures 2-3), not a necessary condition.

Section 4 proposes two new approaches. **Approach 1** (Section 4.1, pp. 12-14) adds either
**Assumption 4 (Simple Covariate Unconfoundedness)** `X_{t*}(0) ⊥⊥ D | X_{t*-1}, Z` or the
non-nested **Assumption 5 (Bad Control Redundancy)**; under either, **Theorem 1** /
**Proposition 1** reduce the ATT to the standard Callaway and Sant'Anna (2021) estimand with
the PRE-treatment bad control `X_{t*-1}` (and `Z`) as covariates - existing DiD tools apply
immediately. **Approach 2** (Section 4.2, pp. 15-17) instead adds **Assumption 6 (Covariate
Unconfoundedness)** `X_{t*}(0) ⊥⊥ D | X_{t*-1}, W, Z` with observed confounders `W` of the
bad control (a natural choice being the lagged outcome, Remark 2), giving the nested
estimand of **Theorem 2 (Equation 4)**: an inner untreated-group regression of `ΔY` on
`(X_{t*}, X_{t*-1}, Z)`, a middle untreated-group regression of that fit on
`(X_{t*-1}, W, Z)`, and an outer average over treated units.

Section 5 extends to staggered adoption with group-time parameters `ATT(g, t)`:
**Theorem 3** (full bad-control history, never-treated comparison), **Proposition 2**
(first-order Markov dimension reduction, conditioning only on `(X_t, X_{g-1}, Z)` and
`(X_{g-1}, W, Z)`, not-yet-treated comparison, base period `g - 1`) - **the result the
paper says its application and accompanying R package implement (p. 21)** - and
**Proposition 3** (staggered Approach 1: Callaway and Sant'Anna 2021 with `X_{g-1}` as a
covariate). Section 6 provides two estimators for the Theorem 2 / Proposition 2 estimand:
an **imputation estimator** with linear working models (Section 6.1, Assumption 8,
Equations 5-7, Proposition 4) and a **Neyman-orthogonal, doubly robust estimator**
(Section 6.2, Equations 8-11, Propositions 5-7, Algorithm 1) that is combinable with
machine-learning nuisances under a product-rate condition (Assumption 9). Remark 6 gives
pre-tests, Supplementary Appendix SC an alternative under parallel trends for the bad
control plus linearity, SD Monte Carlo evidence, and Section 7 a job-displacement
application. Per **Remark 1** (p. 11) the approaches require panel data; per **Remark 8**
(p. 28) they do not map onto a single TWFE regression.

**Key implementation requirements:**

*Assumption checks / warnings:*

Two-period assumptions (Sections 2 and 4), verbatim. Footnote 4 (p. 5): no-anticipation and
SUTVA are implicitly imposed throughout; all expectations are assumed to exist and
conditional-expectation statements hold almost surely.

- **Condition 1 (Outcome Relevance)** (p. 6): for `d ∈ {0, 1}`,
  ```
  E[ΔY_{t*}(0) | X_{t*}(0), X_{t*-1}, Z, D = d]  ≠  E[ΔY_{t*}(0) | Z, D = d]
  ```
  Not data-checkable as stated (involves `X_{t*}(0)` for `D = 1`); for the untreated group
  it is the ordinary "does `X_t` predict the outcome trend" check.
- **Condition 2 (Affected by the Treatment)** (p. 6):
  ```
  ( X_{t*}(0) | X_{t*-1}, Z, D = 1 )  ≁  ( X_{t*}(1) | X_{t*-1}, Z, D = 1 )
  ```
  The complement (equality of the two conditional distributions) is what Lechner (2011) and
  Caetano and Callaway (2025) call *covariate exogeneity*; a leading case is
  `X_{it*}(1) = X_{it*}(0)` for all units. Not directly checkable; the post-treatment
  `ATT_X` of Remark 6 / Figure 6 is a MEAN-effect diagnostic only: a nonzero `ATT_X` is
  evidence that Condition 2 holds, but a zero `ATT_X` does not establish equality of the
  two conditional distributions (distributional or sign-cancelling heterogeneous effects
  can leave the mean at zero), and `ATT_X` says nothing about Condition 1 (outcome
  relevance), which must be assessed separately before calling `X_t` a bad control.
- **Definition 1 (Bad Control)** (p. 6): "`X_t` is a *bad control* if it satisfies both
  Conditions 1 and 2." The bad control is scalar in the notation; "it is straightforward to
  allow `X_it` to be a vector" (p. 5).
- **Assumption 1 (Random Sampling)** (p. 6): the observed data
  `{Y_{it*}, Y_{it*-1}, X_{it*}, X_{it*-1}, Z_i, D_i}_{i=1}^n` are i.i.d. Footnote 8
  (p. 15): under Approach 2 `W` is also observed, which "expands Assumption 1 slightly"
  (the expanded statement is not written out). A substantive sampling/design assumption,
  not data-checkable; the separate input requirement that the data form a balanced
  two-period panel (one row per unit with both periods present) is a structural
  validation, not a test of Assumption 1.
- **Assumption 2 (Conditional Parallel Trends)** (p. 7):
  ```
  E[ΔY_{t*}(0) | X_{t*}(0), X_{t*-1}, Z, D = 1] = E[ΔY_{t*}(0) | X_{t*}(0), X_{t*-1}, Z, D = 0]
  ```
  Conditions on the untreated potential bad control in both periods plus `Z`; compatible
  with `X_t` being affected by treatment. Untestable (footnote 7: cannot be drawn as a
  standard DAG because it conditions on `X_{t*}(0) ≠ X_{t*}`); assessed by pre-tests
  (Remark 6).
- **Assumption 3 (Overlap)** (p. 7): `P(D = 1 | X_{t*}(0), X_{t*-1}, Z) < 1`. Not checkable
  as stated (conditions on `X_{t*}(0)`).
- **Assumption 4 (Simple Covariate Unconfoundedness)** (p. 12): `X_{t*}(0) ⊥⊥ D | X_{t*-1}, Z`.
  An unconfoundedness assumption with `X_{t*}` as the outcome; no restriction on `X_{t*}(1)`;
  rules out confounders of `X_{t*}` other than `(X_{t*-1}, Z)` (Figure 4a). Untestable
  post-treatment; pre-period `ATT_X` pseudo-estimates assess it (Remark 6).
- **Assumption 5 (Bad Control Redundancy)** (p. 14):
  ```
  E[ΔY_{t*}(0) | X_{t*}(0), X_{t*-1}, Z, D = 0] = E[ΔY_{t*}(0) | X_{t*-1}, Z, D = 0]
  ```
  Non-nested with Assumption 4; still allows `X_t` to be a bad control (Figure 4b). The
  assumption is an equality of conditional-mean FUNCTIONS in the untreated group, so it is
  in principle assessable from `D = 0` data (compare a restricted `E[ΔY | X_{t*-1}, Z]` fit
  with an unrestricted `E[ΔY | X_{t*}, X_{t*-1}, Z]` fit, e.g. flexible terms or
  cross-fitted prediction comparison); a single `X_{t*}` coefficient test is equivalent
  ONLY under a correctly specified linear/additive model such as Assumption 8's `m_0`
  (nonlinear or interaction effects of `X_{t*}` can pass a zero-coefficient test while
  violating redundancy). No such test is proposed in the paper.
- **Assumption 6 (Covariate Unconfoundedness)** (p. 15): `X_{t*}(0) ⊥⊥ D | X_{t*-1}, W, Z`.
  Generalizes Assumption 4 with observed confounders `W` (Figure 5); the covariates for
  unconfoundedness of `X_{t*}` need not equal those for parallel trends of `ΔY_{t*}(0)`.
  Untestable; pre-period `ATT_X` pseudo-estimates assess it.
- **Assumption 7 (Overlap)** (p. 15): `P(D = 1 | X_{t*-1}, W, Z) < 1`. Data-checkable via the
  fitted propensity score `p̂(X_{t*-1}, W, Z)` (Equation 8); see Edge cases.
- **Assumption 8 (Linearity Conditions for Imputation)** (p. 24) and **Assumption 9
  (Product Rate for Nuisance Estimators)** (p. 27): estimator-level; stated in the
  imputation / DR sections below.

Staggered-adoption assumptions (Section 5), verbatim. Units first treated at `G_i`,
never-treated `G_i = ∞`, `𝒢 ⊆ {2, ..., T} ∪ {∞}`, `𝒢̄ := 𝒢 \ {∞}`; no group is treated in
period 1 (an "already treated" group is dropped - parallel trends does not identify its
effects and it is not useful as a comparison group without further assumptions, p. 18);
`D_it` is treatment status in period `t` (`D_t = 0` = not yet treated); `𝐗 = (X_1, ..., X_T)'`;
`𝐗_{s1:s2}` is the sub-vector for periods `s1..s2`; `ΔY_t(0) = Y_t(0) - Y_{t-1}(0)`.

- **Assumption MP-1 (Staggered Treatment Adoption)** (p. 17): for `t = 2, ..., T`,
  `D_{it-1} = 1 ⟹ D_{it} = 1`. Data-checkable (absorbing treatment).
- **Assumption MP-2 (No anticipation)** (p. 18): for all `t < G_i`, `Y_it = Y_it(0)` and
  `X_it = X_it(0)` - extends no-anticipation to the bad control; relaxable to limited
  anticipation "at the cost of notation". Observation rule under MP-1/MP-2 (p. 18):
  ```
  Y_it = 1{t >= G_i} Y_it(G_i) + 1{t < G_i} Y_it(0)
  X_it = 1{t >= G_i} X_it(G_i) + 1{t < G_i} X_it(0)
  ```
- **Assumption MP-3 (Multi-Period Random Sampling)** (p. 19):
  `{Y_i1, ..., Y_iT, X_i1, ..., X_iT, W_i, Z_i, D_i}_{i=1}^n` i.i.d. across units
  (balanced panel; `W`, `Z` time-invariant per unit in this notation).
- **Assumption MP-4 (Multi-Period Conditional Parallel Trends)** (p. 19): for all `g ∈ 𝒢`
  and `t = 2, ..., T`,
  ```
  E[ΔY_t(0) | 𝐗(0), Z, G = g] = E[ΔY_t(0) | 𝐗(0), Z]
  ```
  (conditions on the FULL vector of untreated potential bad controls). Untestable.
- **Assumption MP-5 (Multi-Period Covariate Unconfoundedness)** (p. 19): for all `t ≥ 2`,
  ```
  X_t(0) ⊥⊥ ( G, X_{t-2}(0), ..., X_1(0) ) | ( X_{t-1}(0), W, Z )
  ```
  Extends Assumption 6 AND embeds a first-order Markov structure (`X_t(0)` depends on its
  history only through `X_{t-1}(0)`). Untestable; assessed by pre-period `ATT_X(g, t)`
  (Remark 6).
- **Assumption MP-6 (Multi-Period Overlap)** (p. 19): for all `g ∈ 𝒢̄` there exists `ε > 0`
  with
  ```
  P(G = g) > ε   and   P(G = ∞ | 𝐗(0), W, Z) > ε
  ```
  Cohort masses `P(G = g)` are directly checkable. The conditional-overlap component is
  NOT directly checkable as stated: it conditions on the full untreated potential history
  `𝐗(0)`, whose post-treatment components are unobserved for treated cohorts. A fitted
  never-treated propensity on the observed reduced conditioning set of Proposition 2
  (`X_{g-1}, W, Z`) is a model-dependent, necessary-but-NOT-sufficient diagnostic: by
  iterated expectations, MP-6 implies overlap on the coarser observed set, but coarse
  overlap cannot establish MP-6's full-history overlap. The reduction of the future
  untreated bad-control history to `(X_{g-1}, W, Z)` is supplied by MP-5 (with MP-6) via
  Lemma 3 (Appendix A, p. 41); MP-7 reduces the OUTCOME regression only and does not by
  itself justify the reduced propensity.
- **Assumption MP-7 (Multi-Period Bad Control Dimension Reduction)** (p. 20): for all
  `1 ≤ t_1 < t_2 ≤ T` and `g ∈ 𝒢`,
  ```
  E[Y_{t2}(0) - Y_{t1}(0) | 𝐗(0), Z, G = g] = E[Y_{t2}(0) - Y_{t1}(0) | X_{t2}(0), X_{t1}(0), Z, G = g]
  ```
  (cf. Wooldridge 2010, Eq. 10.12). Untestable in general.
- **Assumption MP-8 (Multi-Period Simple Covariate Unconfoundedness)** (p. 21): for all
  `t ≥ 2`,
  ```
  X_t(0) ⊥⊥ ( G, X_{t-2}(0), ..., X_1(0) ) | ( X_{t-1}(0), Z )
  ```
  (MP-5 without `W`; staggered Assumption 4).
- **Assumption MP-9 (Multi-Period Bad Control Redundancy)** (p. 21): for all
  `1 ≤ t_1 < t_2 ≤ T` and `g ∈ 𝒢`,
  ```
  E[Y_{t2}(0) - Y_{t1}(0) | 𝐗(0), Z, G = g] = E[Y_{t2}(0) - Y_{t1}(0) | X_{t1}(0), Z, G = g]
  ```
  (staggered Assumption 5: controlling for the pre-treatment bad control suffices).
- **Remark 1 (Implications for Dealing with Bad Controls in Different Settings)** (p. 11),
  verbatim: "The discussion above effectively applies to cross-sectional settings with
  little modification, except that there would be one period t* and by replacing ΔY_{t*}
  with Y_{t*}. The approaches that we discuss below rely on the researcher having access to
  panel data and would not be available (at all) with cross-sectional data and (to a large
  extent) with repeated cross-sections data. Therefore, an important implication of our
  results is that a researcher's ability to deal with important covariates being affected by
  the treatment depends on the setting and nature of data that is available." Implementation:
  **panel data required**; the Section 3 bias analysis applies to cross-sections but the
  Section 4/5 identification results (Theorems 1-3, Propositions 1-3) do not.
- **Warning the paper itself issues** (p. 8): in a given application none of the extra
  assumptions may be plausible, in which case the ATT is simply not identified.

*Target parameters and identification results:*

Target parameters:
```
ATT      := E[Y_{t*}(1) - Y_{t*}(0) | D = 1]                 (two-period, p. 6)
ATT(g,t) := E[Y_t(g) - Y_t(0) | G = g]                       (staggered, p. 18)
```
Observation rules (p. 5): `Y_{it*-1} = Y_{it*-1}(0)`, `Y_{it*} = D_i Y_{it*}(1) + (1 - D_i) Y_{it*}(0)`,
`X_{it*-1} = X_{it*-1}(0)`, `X_{it*} = D_i X_{it*}(1) + (1 - D_i) X_{it*}(0)`.

Identification failure (pp. 7-8). Define the untreated-group outcome-change regression
(Equation 1; footnote 5: `m_0`'s argument list changes with context):
```
m_0(X_{t*}(0), X_{t*-1}, Z) := E[ΔY_{t*} | X_{t*}(0), X_{t*-1}, Z, D = 0]                       (1)
```
Under Assumption 2 (Equation 2) and in integral form (Equation 3):
```
ATT = E[ΔY_{t*} | D = 1] - E[ m_0(X_{t*}(0), X_{t*-1}, Z) | D = 1 ]                             (2)

ATT = E[ΔY_{t*} | D = 1]
      - ∫∫ m_0(x_{t*}(0), x_{t*-1}, z) dF̃_1(x_{t*}(0) | x_{t*-1}, z) dF_1(x_{t*-1}, z)           (3)

F̃_1(x_{t*}(0) | x_{t*-1}, z) := F_{X_{t*}(0) | X_{t*-1}, Z, D=1}(x_{t*}(0) | x_{t*-1}, z)   (NOT identified)
F_1(x_{t*-1}, z)             := F_{X_{t*-1}, Z | D=1}(x_{t*-1}, z)
```
Every approach in the paper either recovers `F̃_1` or adds assumptions that make it
irrelevant. Footnote 6: `F_1`, `F_0` denote cdfs conditional on `D = 1`, `D = 0`.

Traditional approaches (Section 3):
```
τ^use     := E[ΔY_{t*} | D = 1] - E[ m_0(X_{t*}, X_{t*-1}, Z) | D = 1 ]                          (p. 9)
           = E[ΔY_{t*} | D = 1] - ∫∫ m_0(x_{t*}, x_{t*-1}, z) dF_1(x_{t*} | x_{t*-1}, z) dF_1(x_{t*-1}, z)
τ^use - ATT = ∫∫ m_0(x_{t*}(0), x_{t*-1}, z) d( F̃_1(x_{t*}(0) | x_{t*-1}, z) - F_1(x_{t*} | x_{t*-1}, z) ) dF_1(x_{t*-1}, z)

τ^discard := E[ΔY_{t*} | D = 1] - E[ m_0(Z) | D = 1 ] = E[ΔY_{t*} | D = 1] - ∫ m_0(z) dF_1(z)   (p. 10)
τ^discard - ATT = ∫∫ ( m_0(x_{t*}(0), x_{t*-1}, z) - m_0(z) ) dF̃_1(x_{t*}(0) | x_{t*-1}, z) dF_1(x_{t*-1}, z)   (under Assumptions 1-3)
```
where `F_1(x_{t*} | x_{t*-1}, z) := F_{X_{t*} | X_{t*-1}, Z, D=1}(x_{t*} | x_{t*-1}, z)` is the
OBSERVED conditional cdf of the treated group's bad control. The paper's prose (mirrored
here): `τ^use - ATT` "is non-zero when the distribution of `X_{t*}(0)` differs from the
distribution of `X_{t*}(1)` for the treated group - this corresponds exactly to Condition 2
holding", so the approach "leads to bias whenever `X_{t*}` is a bad control" (p. 9;
Figure 2: arrow `D → X_{t*}` removed rationalizes it); `τ^discard - ATT` "is equal to 0
when `m_0(Z) = m_0(X_{t*}(0), X_{t*-1}, Z)`, which arises only in the case where `X_t` does
not affect the change in untreated potential outcomes", i.e. Condition 1 fails (pp. 10-11;
Figure 3: arrows `X_{t*}(0) → ΔY_{t*}(0)` and `X_{t*-1} → ΔY_{t*}(0)` removed). Dropping a
bad control "only changes the form of the bias relative to including it" (p. 11).
Qualification (not in the paper): both bias expressions are integrals of `m_0` against a
signed measure / difference of functions, so they can also be exactly zero by cancellation
while Conditions 1-2 hold; the Figure 2 / Figure 3 conditions are sufficient benchmark
cases for zero bias, and "generally biased" is the precise reading.

Assumption-to-result map:

| Result | Assumptions | Estimand conditions on | Comparison group | Base period |
|---|---|---|---|---|
| Theorem 1 (p. 12) | 1, 2, 3, 4 | `(X_{t*-1}, Z)` | `D = 0` | `t* - 1` |
| Proposition 1 (p. 14) | 1, 2, 3, 5 | `(X_{t*-1}, Z)` | `D = 0` | `t* - 1` |
| Theorem 2 / Eq. 4 (p. 15) | 1, 2, 6, 7 | inner `(X_{t*}, X_{t*-1}, Z)`; middle `(X_{t*-1}, W, Z)` | `D = 0` | `t* - 1` |
| Theorem 3 (p. 19) | MP-1 to MP-6 | inner `(𝐗_{g:T}, 𝐗_{1:(g-1)}, Z)`; middle `(𝐗_{1:(g-1)}, W, Z)` | never-treated `G = ∞` | `g - 1` |
| Proposition 2 (p. 20) | MP-1 to MP-7 | inner `(X_t, X_{g-1}, Z)`; middle `(X_{g-1}, W, Z)` | not-yet-treated `D_t = 0` | `g - 1` |
| Proposition 3 (p. 21) | MP-1, MP-2, MP-3, MP-4, MP-6, MP-7, and (MP-8 or MP-9) | `(X_{g-1}, Z)` | not-yet-treated `D_t = 0` | `g - 1` |
| Lemma 1 (Appendix A, p. 39) | MP-1 to MP-4, MP-6 | `(𝐗(0), Z)` | any single not-yet-treated group `g' > t` | `g - 1` |

- **Theorem 1** (p. 12). Under Assumptions 1 to 4,
  ```
  ATT = E[ΔY_{t*} | D = 1] - E[ m_0(X_{t*-1}, Z) | D = 1 ]
  ```
  A standard DiD estimand (Heckman, Ichimura, and Todd 1997; Abadie 2005; Callaway and
  Sant'Anna 2021) with covariates = pre-treatment bad control + `Z`; `X_{t*}` does not appear
  at all (p. 13).
- **Proposition 1** (Section 4.1.1, p. 14). Under Assumptions 1 to 3 and 5, the SAME
  estimand as Theorem 1. Its proof is stated (p. 14) to be "provided in Appendix A" but is
  not present in Appendix A as printed (see Gaps).
- **Theorem 2 (Equation 4)** (p. 15). Under Assumptions 1, 2, 6 and 7,
  ```
  ATT = E[ΔY_{t*} | D = 1]
        - E[ E[ m_0(X_{t*}, X_{t*-1}, Z) | X_{t*-1}, W, Z, D = 0 ] | D = 1 ]                     (4)
  ```
  Inner: untreated-group outcome-change regression given `(X_{t*}(0) = X_{t*}, X_{t*-1}, Z)`
  (the counterfactual path under Assumption 2). Middle: expectation over `X_{t*}(0)` given
  `(X_{t*-1}, W, Z)` in the UNTREATED group, which equals the treated group's conditional
  distribution under Assumption 6. Outer: over the TREATED group's `(X_{t*-1}, W, Z)`.
  `W` appears ONLY in the middle set; `X_{t*}` ONLY in the inner set (pp. 15-16; proof p. 39).
- **Remark 2 (Lagged Outcomes as W)** (p. 16). With `W = Y_{t*-1}`,
  ```
  ATT = E[ΔY_{t*} | D = 1]
        - E[ E[ m_0(X_{t*}, X_{t*-1}, Z) | X_{t*-1}, Y_{t*-1}, Z, D = 0 ] | D = 1 ]
  ```
  "genuinely difference-in-differences" while the lagged outcome handles the bad control;
  `Y_{t*-1}` enters only the middle (unconfoundedness) conditioning set, never `m_0`. Cites
  Chabé-Ferret 2017; Daw and Hatfield 2018; Imai, Kim, and Wang 2023; Marx, Tamer, and Tang
  2025 on the usual awkwardness of conditioning on lagged outcomes.
- **Remark 3 (Parallel Trends for Bad Control)** (p. 17). Parallel trends for `X_t` would
  require identifying the ENTIRE conditional distribution of `X_{t*}(0)` for the treated
  group; distributional DiD (Bonhomme and Sauder 2011; Callaway and Li 2019; Callaway, Li,
  and Oka 2018) or changes-in-changes (Athey and Imbens 2006; Melly and Santangelo 2015)
  could in principle be adapted but typically require continuous outcomes, so are less
  suitable for discrete or mixed covariates. With an extra linearity assumption only
  `E[ΔX_{t*}(0) | D = 1]` is needed - Supplementary Appendix SC (below).
- **Theorem 3** (p. 19). Under Assumptions MP-1 to MP-6 and `t ≥ g`,
  ```
  ATT(g, t) = E[Y_t - Y_{g-1} | G = g]
              - E[ E[ E[ Y_t - Y_{g-1} | 𝐗_{g:T}, 𝐗_{1:(g-1)}, Z, G = ∞ ] | 𝐗_{1:(g-1)}, W, Z, G = ∞ ] | G = g ]
  ```
  Base period `g - 1`; never-treated comparison group "as it is the only group for which
  the untreated potential bad control is observed in all periods" (pp. 19-20).
  Conditioning on the full history is high-dimensional - motivates Section 5.1.
- **Proposition 2** (Section 5.1.1, Extension 1: Dimension Reduction, p. 20). Under
  Assumptions MP-1 to MP-7 and `t ≥ g`,
  ```
  ATT(g, t) = E[Y_t - Y_{g-1} | G = g]
              - E[ E[ E[ Y_t - Y_{g-1} | X_t, X_{g-1}, Z, D_t = 0 ] | X_{g-1}, W, Z, D_t = 0 ] | G = g ]
  ```
  Not-yet-treated comparison (`D_t = 0`, i.e. `G > t`, includes never-treated), "though we
  note that effectively the same arguments can rationalize using the never-treated comparison
  group as well" (p. 21). **Verbatim (p. 21): "Proposition 2 is a mild, but practically
  useful, extension of our earlier identification result, and it is the one that is
  implemented in our application and accompanying R package."**
- **Proposition 3** (Section 5.1.2, Extension 2: Simple Covariate Unconfoundedness, p. 21).
  Under MP-1, MP-2, MP-3, MP-4, MP-6, MP-7, `t ≥ g`, and either MP-8 or MP-9,
  ```
  ATT(g, t) = E[Y_t - Y_{g-1} | G = g] - E[ E[ Y_t - Y_{g-1} | X_{g-1}, Z, D_t = 0 ] | G = g ]
  ```
  "one can directly use Callaway and Sant'Anna (2021) for estimation as long as the
  pre-treatment value of the bad control is included as a covariate" (pp. 21-22). MP-5 is
  NOT required.
- **Remark 4 (Aggregating ATT(g,t)'s)** (p. 22): see the staggered-estimation section.
- **Remark 5 (Lagged Outcomes as W in the Multi-Period Case)** (p. 22). `W = Y_{g-1}` works
  in the staggered setting. The more natural alternative
  ```
  X_t(0) ⊥⊥ ( G, X_{t-2}(0), ..., X_1(0) ) | ( X_{t-1}(0), Y_{t-1}(0), Z )
  ```
  is NOT pursued: `Y_{t-1}(0)` is not always observed for group `g` (unlike `Y_{g-1}(0)`),
  and it would require handling feedback from treatment to the bad control through the
  outcome (Bonhomme 2025; Marx, Tamer, and Tang 2025). **Endorsed choice: `W = Y_{g-1}`,
  the outcome at the group's base period, not `Y_{t-1}`.**
- **Remark 6 (Pre-Testing)** (pp. 22-23). "None of Assumptions MP-4, MP-5 and MP-8 is
  directly testable, but the same sort of pre-tests that are commonly used in
  difference-in-differences applications are useful to assess their plausibility. In
  particular, one can compute the same estimand as in Proposition 2 or 3 but for
  pre-treatment periods, in which case these pre-treatment pseudo-ATT(g, t)'s should be
  equal to zero. To pre-test assumptions about the bad control specifically, under our
  assumptions,
  ```
  ATT_X(g, t) := E[X_t(g) - X_t(0) | G = g]
  ```
  is identified. Estimating ATT_X(g, t) in pre-treatment periods provides a way to assess
  Assumption MP-5 or MP-8 in the periods before the treatment occurred. In post-treatment
  periods, it [is] a test for X_t actually being a bad control as it should be affected by
  the treatment." (verbatim, p. 23; "[is]" supplied - the printed text reads "it a test").
  The paper writes no estimand or estimator for `ATT_X(g, t)`; the application (Section 7)
  obtains it by "implementing the first step of the Section 6 imputation estimator" with
  the bad control as the outcome (see the Application section and Gaps). Scope
  qualification (not in the paper): `ATT_X(g, t)` is a conditional-MEAN effect of the
  treatment on `X`. Its evidentiary scope is one-sided: a nonzero post-treatment value
  supports Condition 2 (treatment affects `X`), but a zero value does not rule out
  distributional or heterogeneous effects, and it carries no information about
  Condition 1 (outcome relevance). It therefore cannot by itself establish that `X_t`
  "actually" is a bad control; an implementation should label it a mean-effect
  diagnostic and pair it with a separate outcome-relevance assessment.

Appendix A proof structure (pp. 38-42) - intermediate identities useful as implementation
checks:
```
ATT = E[ΔY_{t*} | D = 1] - E[ΔY_{t*}(0) | D = 1]                                                 (A.1, p. 38)
```
Proof of Theorem 1 (p. 39): LIE, LIE, Assumption 2, Assumption 4 (switches the middle
conditioning from `D = 1` to `D = 0`), LIE, definition of `m_0`. Proof of Theorem 2 (p. 39):
LIE, Assumption 2, definition of `m_0`, LIE over `(X_{t*-1}, W, Z)`, Assumption 6, then
`X_{t*}(0) = X_{t*}` for the untreated group.
- **Lemma 1** (pp. 39-40). Under MP-1 to MP-4 and MP-6, for `t ≥ g` and any not-yet-treated
  group `g' > t`,
  ```
  ATT(g, t) = E[Y_t - Y_{g-1} | G = g] - E[ E[Y_t - Y_{g-1} | 𝐗(0), Z, G = g'] | G = g ]
  ```
  (telescoping one-period parallel trends MP-4 from `g` through `t`; requires `g'` untreated
  through `t`).
- **Lemma 2** (p. 40). Under MP-5 and MP-6: `(X_g(0), ..., X_T(0)) ⊥⊥ G | X_{g-1}(0), ..., X_1(0), W, Z`
  (proof omitted, "very similar argument as for Lemma 3").
- **Proof of Theorem 3** (p. 40, Equation A.2): Lemma 1 with `g' = ∞`, LIE over
  `(𝐗_{1:(g-1)}, W, Z)`, then Lemma 2 to switch `G = g` to `G = ∞` in the middle set.
- **Lemma 3** (p. 41). Under MP-5 and MP-6: `(X_g(0), ..., X_T(0)) ⊥⊥ G | X_{g-1}(0), W, Z`.
  Proof factorizes the joint cdf of `𝐗_{g:T}(0)` into one-step transitions
  `dF_{X_s(0) | X_{s-1}(0), ..., X_{g-1}(0), W, Z, G}` and drops `G` from each by MP-5 - the
  first-order Markov structure is the source of Proposition 2's dimension reduction.
- **Proof of Proposition 2** (p. 41, Equation A.3): `ATT(g, t) = E[Y_t - Y_{g-1} | G = g] - E[ E[Y_t - Y_{g-1} | X_t(0), X_{g-1}, Z, D_t = 0] | G = g ]`
  by Lemma 1's argument plus MP-7; then LIE over `(X_{g-1}, W, Z)` and Lemma 3.
- **Lemma 4** (p. 41). Under MP-6 and MP-8: `(X_g(0), ..., X_T(0)) ⊥⊥ G | X_{g-1}(0), Z`
  (proof omitted).
- **Proof of Proposition 3** (p. 42, Equation A.4): `ATT(g, t) = E[Y_t - Y_{g-1} | G = g] - E[ E[Y_t - Y_{g-1} | 𝐗(0), Z, D_t = 0] | G = g ]`;
  under MP-9 the inner set collapses to `(X_{g-1}, Z)` directly; under MP-8 apply MP-7, LIE,
  Lemma 4, LIE.

Causal graphs (Figures 1-5; convention p. 10: red = change relative to Figure 1, dashed =
removed arrow, dashed box = unobserved, solid red box = observed):
- **Figure 1** (p. 8): (a) SWIG with `D = 0` for Assumption 2 - arrows `X_{t*-1} → X_{t*}(0)`,
  `Z → X_{t*}(0)`, `X_{t*-1} → ΔY_{t*}(0)`, `Z → ΔY_{t*}(0)`, `X_{t*}(0) → ΔY_{t*}(0)`;
  (b) DAG for `X_{t*}` - `W` (unobserved) `→ D`, `W → X_{t*}`, `D → X_{t*}`, `X_{t*-1} → D`,
  `X_{t*-1} → X_{t*}`, `Z → D`, `Z → X_{t*}`.
- **Figure 2** (p. 10): Figure 1(b) without `D → X_{t*}` (rationalizes `τ^use`; Condition 2 fails).
- **Figure 3** (p. 11): Figure 1(a) without `X_{t*}(0) → ΔY_{t*}(0)` and `X_{t*-1} → ΔY_{t*}(0)`
  (rationalizes `τ^discard`; Condition 1 fails).
- **Figure 4** (p. 13; the figure title says "Approach 3" while the text calls it Section 4.1
  / new Approach 1 - see Gaps): (a) DAG for Assumption 4 = Figure 1(b) without `W → D`,
  `W → X_{t*}`; (b) SWIG for Assumption 5 = Figure 1(a) without `X_{t*}(0) → ΔY_{t*}(0)`.
- **Figure 5** (p. 15): Figure 1(b) with `W` observed (solid red box) - Assumption 6.

*Imputation estimator (Section 6.1, Assumption 8, Equations 5-7):*

Section 6 (p. 23) treats the two-period Theorem 2 case ("the extension to estimating
ATT(g,t) is straightforward"; Theorem 1's case is covered by Callaway and Sant'Anna 2021
"essentially immediately"). Write
```
ATT = E[ΔY_{t*} | D = 1] - τ,        τ := E[ν_0(X_{t*-1}, W, Z) | D = 1],
ν_0(X_{t*-1}, W, Z) := E[ m_0(X_{t*}, X_{t*-1}, Z) | X_{t*-1}, W, Z, D = 0 ]                     (5)
```
`ν_0` is the untreated-group regression of `m_0` (evaluated at the unit's own
`X_{t*}, X_{t*-1}, Z`) on `(X_{t*-1}, W, Z)`: drops `X_{t*}`, adds `W`.

- **Assumption 8 (Linearity Conditions for Imputation)** (p. 24), verbatim: "The outcome
  regression for the untreated group is given by
  ```
  m_0(X_{t*}, X_{t*-1}, Z) = X_{t*} β_1 + X_{t*-1} β_2 + Z' β_3,
  ```
  and the conditional mean of the post-treatment bad control in the untreated group follows
  ```
  E[X_{t*} | X_{t*-1}, W, Z, D = 0] = X_{t*-1} γ_1 + W' γ_2 + Z' γ_3."
  ```
  (`X` scalar with scalar `β_1, β_2, γ_1`; `W`, `Z` vectors.) Builds on Gardner, Thakral,
  Tô, and Yap 2023; Borusyak, Jaravel, and Spiess 2024; Liu, Wang, and Xu 2024 (p. 23);
  allows treatment-effect heterogeneity. **Intercept convention (paper silent, see Gaps
  item 19):** neither linear model, nor `R_i` / `S_i` in SA, shows an explicit constant,
  yet the Monte Carlo DGPs contain constants (`θ_2 - θ_1 = 0.3` in `ΔY`, `0.15` in
  `X_2(0)`) and Table S1 classifies the imputation estimator as consistent in DGPs 1 and
  4 - which requires an intercept in both regressions (or `Z` containing a constant). An
  implementation must add an intercept to every nuisance regression (equivalently treat
  `Z` as including `1`); `R_i`, `S_i`, `R̃_i` and the S8 influence function are then read
  with the constant included in `Z`.
- Steps (p. 24; SA p. 1 writes the OLS fits with explicit `(1 - D_i)` weights, with
  `R_i := (X_{it*}, X_{i,t*-1}, Z_i')'` and `S_i := (X_{i,t*-1}, W_i', Z_i')'`):
  1. OLS of `ΔY_{t*}` on `(X_{t*}, X_{t*-1}, Z)` in the UNTREATED sample:
     `β̂ = (Σ_i (1 - D_i) R_i R_i')^{-1} Σ_i (1 - D_i) R_i ΔY_{it*}`.
  2. OLS of `X_{t*}` on `(X_{t*-1}, W, Z)` in the UNTREATED sample:
     `γ̂ = (Σ_i (1 - D_i) S_i S_i')^{-1} Σ_i (1 - D_i) S_i X_{it*}`;
     `Ê[X_{t*} | X_{t*-1}, W, Z, D = 0] = X_{t*-1} γ̂_1 + W' γ̂_2 + Z' γ̂_3`.
  3. Plug the fitted conditional mean into the `m_0` model in place of `X_{t*}`:
     ```
     ν̂_0(X_{t*-1}, W, Z) = Ê[X_{t*} | X_{t*-1}, W, Z, D = 0] β̂_1 + X_{t*-1} β̂_2 + Z' β̂_3
     ```
     Under Assumption 8, `ν_0 = S'γ β_1 + X_{t*-1} β_2 + Z' β_3 = R̃' β` with
     `R̃_i := ((S_i'γ)', X_{i,t*-1}', Z_i')'` = `R_i` with `X_{it*}` replaced by its
     conditional mean (SA pp. 1-2).
  4. With `n_1 := Σ_i D_i`,
     ```
     m̂_1  = (1/n_1) Σ_{i=1}^n D_i ΔY_{it*}                                                      (6)
     τ̂_ra = (1/n_1) Σ_{i=1}^n D_i ν̂_0(X_{i,t*-1}, W_i, Z_i)                                     (7)
     ```
     (SA p. 3 writes `τ̂_ra = (1/n) Σ_i (D_i/π̂) ν̂_{0,i}` with `π̂ = n_1/n` - identical.)
  5. `ATT̂_ra = m̂_1 - τ̂_ra`.
- **Proposition 4** (p. 24), verbatim: "Under Assumptions 1, 2, 6, 7, 8, and S1, the
  imputation estimator ATT̂_ra is consistent and satisfies
  ```
  √n ( ATT̂_ra - ATT ) →_d N(0, Ω),
  ```
  where Ω = Var(ψ^ra) and ψ^ra is the influence function defined in Equation S8 in the
  Supplementary Appendix."
- **Assumption S1 (Regularity Conditions for Imputation Estimator)** (SA p. 1), verbatim:
  "(i) E‖X_{t*}‖^4, E‖X_{t*-1}‖^4, E‖W‖^4, E‖Z‖^4, and E[ΔY_{t*}^4] are finite. (ii) The
  matrices E[RR' | D = 0] and E[SS' | D = 0] are positive definite."
- The Callaway and Sant'Anna (2021)-based "imputation" estimators used as comparators in
  the application (`Imp: include BC`, `Imp: exclude BC`, Estimators 3-4, p. 30) are NOT
  this Section 6.1 estimator; they are CS-style regression adjustments with/without the bad
  control in both periods.

*Neyman-orthogonal / doubly robust estimator (Section 6.2, Equations 8-11, Algorithm 1):*

Motivation (p. 25): the imputation estimator's bias under misspecification of `m_0` or
`E[X_{t*} | X_{t*-1}, W, Z, D = 0]` "does not vanish with sample size"; the DR estimator
allows (i) double robustness and (ii) machine-learning nuisances.

Notation (p. 25): `π := P(D = 1)`; observed data `O = (Y_{t*-1}, Y_{t*}, X_{t*-1}, X_{t*}, W, Z, D)`;
nuisance vector `η = (m_0, ν_0, p, ω_0)` with
```
p(X_{t*-1}, W, Z) = P(D = 1 | X_{t*-1}, W, Z)                                                    (8)
ω_0( X_{t*}(0), X_{t*-1}, Z ) := E[ p(X_{t*-1}, W, Z) / (1 - p(X_{t*-1}, W, Z)) | X_{t*}(0), X_{t*-1}, Z, D = 0 ]   (9)
```
`p` conditions on the PRE-treatment bad control (never on `X_{t*}`); `ω_0` is the
untreated-group regression of the odds `p/(1 - p)` on `(X_{t*}, X_{t*-1}, Z)` (includes
`X_{t*}`, drops `W`) - the mirror image of the `m_0 → ν_0` nesting.

Score (Equation 10, p. 25), exactly:
```
φ_1(O; η) = (D/π) ΔY_{t*}  -  (D/π) ν_0(X_{t*-1}, W, Z)
            - [(1 - D)/π] ( m_0(X_{t*}, X_{t*-1}, Z) - ν_0(X_{t*-1}, W, Z) ) · p(X_{t*-1}, W, Z) / (1 - p(X_{t*-1}, W, Z))
            - [(1 - D)/π] ( ΔY_{t*} - m_0(X_{t*}, X_{t*-1}, Z) ) · ω_0(X_{t*}, X_{t*-1}, Z)                     (10)
```
Term by term: (1) treated mean of `ΔY` (`E[ΔY_{t*} | D = 1]`); (2) treated mean of the nested
regression (`τ`); (3) untreated residual of `m_0` around its projection `ν_0`, reweighted by
the treatment odds given `(X_{t*-1}, W, Z)` - corrects the second-stage nuisance `ν_0`;
(4) untreated outcome-regression residual reweighted by `ω_0` - corrects the first-stage
nuisance `m_0`. All untreated terms are normalized by `π` (not `1 - π`); `(1 - D)/π` times
`p/(1 - p)` or `ω_0` implements the change of measure from `D = 0` to `D = 1` (Lemma S2).
The score never divides by `p` or by `π`-weighted `1 - p` on the treated side, so overlap is
needed only away from 1.

- **Proposition 5** (p. 25), verbatim: "Under Assumptions 1, 2, 6 and 7, ATT = E[φ_1(O; η)]."
  Proof (SB.1, SA pp. 4-5): line 1 of (10) equals ATT by LIE + Theorem 2 + definition of
  `ν_0`; line 2 has expectation `-((1-π)/π) E[m_0 · p/(1-p) | D = 0] + ((1-π)/π) E[ν_0 · p/(1-p) | D = 0] = 0`
  (LIE + definition of `ν_0`); line 3 has expectation `-((1-π)/π) E[(ΔY_{t*} - m_0) ω_0 | D = 0] = 0`
  (LIE + definition of `m_0`). No Gateaux-derivative computation is written; orthogonality
  is exhibited operationally in Lemma S3 (second-order conditional bias).
- **Lemma S2** (SA p. 5), verbatim: "Under Assumptions 6 and 7, for any integrable function
  h(X_{t*}(0), X_{t*-1}, Z),
  ```
  E[ h(X_{t*}(0), X_{t*-1}, Z) ω_0(X_{t*}(0), X_{t*-1}, Z) | D = 0 ] = (π/(1-π)) E[ h(X_{t*}(0), X_{t*-1}, Z) | D = 1 ]."
  ```
  (definition of `ω_0` + LIE, Assumption 6, change of measure.)
- Sample analog (Equation 11, pp. 25-26), with `η̂ = (m̂_0, ν̂_0, p̂, ω̂_0)`,
  `π̂ = n^{-1} Σ_i D_i`, `τ̂ = n_1^{-1} Σ_i D_i ν̂_{0,i}`, `m̂_{0,i} = m̂_0(X_{it*}, X_{i,t*-1}, Z_i)`,
  `p̂_i = p̂(X_{i,t*-1}, W_i, Z_i)`, `ω̂_{0,i} = ω̂_0(X_{it*}, X_{i,t*-1}, Z_i)`:
  ```
  ATT̂_dr = m̂_1 - τ̂ - (1/n) Σ_{i=1}^n [ ((1 - D_i)/π̂) (m̂_{0,i} - ν̂_{0,i}) · p̂_i/(1 - p̂_i)  +  ((1 - D_i)/π̂) (ΔY_{it*} - m̂_{0,i}) ω̂_{0,i} ]   (11)
  ```
- Parametric working models (p. 26). `η(θ) = (m_0(θ_m), ν_0(θ_ν), p(θ_p), ω_0(θ_ω))`. The
  "leading example": the Assumption 8 linear models for `m_0(θ_m)` and `ν_0(θ_ν)`; a
  **logit** working model for `p(θ_p)`; and `ω_0(θ_ω)` "to come from a regression of the
  estimated odds ratio, p̂(θ̂_p)/(1 - p̂(θ̂_p)) on X_{t*}, X_{t*-1}, and Z using the untreated
  group." Hence: `m_0` = OLS of `ΔY` on `(X_{t*}, X_{t*-1}, Z)` (untreated); `ν_0` = plug
  the untreated OLS fit of `X_{t*}` on `(X_{t*-1}, W, Z)` into the `m_0` model; `p` = logit
  of `D` on `(X_{t*-1}, W, Z)` on ALL units (Algorithm 1 step 2(a)); `ω_0` = OLS of the
  fitted odds on `(X_{t*}, X_{t*-1}, Z)` (untreated). Misspecification is allowed; under
  mild regularity `η̂(θ̂) →_p η(θ)` and `ATT̂_dr(θ̂) →_p ATT_dr(θ)`, possibly `≠ ATT`.
- **Proposition 6 (double robustness)** (p. 26), verbatim: "Under Assumptions 1, 2, 6 and 7,
  given parametric working models for the nuisance functions η(θ), and assuming that
  η̂(θ̂) →_p η(θ) and ATT̂_dr(θ̂) →_p ATT_dr(θ), and that either
  (i) ( m_0(θ_m), ν_0(θ_ν) ) = ( m_0, ν_0 )
  (ii) ( p(θ_p), ω_0(θ_ω) ) = ( p, ω_0 )
  Then, ATT̂_dr(θ̂) →_p ATT."
  **Pairing:** BOTH outcome nuisances `(m_0, ν_0)` correct, OR BOTH weighting nuisances
  `(p, ω_0)` correct. Condition (i) is Assumption 8. The nuisances are functionally related:
  a misspecified `m_0(θ_m)` generally implies a misspecified `ν_0(θ_ν)`, and likewise
  `p(θ_p)` / `ω_0(θ_ω)` (p. 26; the discussion sentence reads "Proposition 5 shows that
  ATT̂_dr is doubly robust" - see Gaps).
  Proof (SA pp. 5-7): `ATT_dr(θ) = m_1 - τ(θ) - T_3(θ) - T_4(θ)` (S9) with
  `T_3(θ) = ((1-π)/π) E[(m_0(θ_m) - ν_0(θ_ν)) · p(θ_p)/(1 - p(θ_p)) | D = 0]`,
  `T_4(θ) = ((1-π)/π) E[(ΔY_{t*} - m_0(θ_m)) ω_0(θ_ω) | D = 0]`. Case (i): `τ(θ) = τ`,
  `T_3 = T_4 = 0` by LIE for ARBITRARY `p(θ_p)`, `ω_0(θ_ω)`. Case (ii):
  `T_3^A(θ) = ((1-π)/π) E[m_0(θ_m) ω_0 | D = 0] = E[m_0(θ_m) | D = 1]` (S10, via Lemma S2),
  `T_3^B(θ) = E[ν_0(θ_ν) | D = 1] = τ(θ)` (S11),
  `T_4^A(θ) = ((1-π)/π) E[m_0 ω_0 | D = 0] = E[m_0 | D = 1] = τ` (S12),
  `T_4^B(θ) = E[m_0(θ_m) | D = 1]` (S13), so `ATT_dr(θ) = m_1 - τ` for ARBITRARY `m_0(θ_m)`,
  `ν_0(θ_ν)`. (In S12 "`E[m_0 | D = 1] = τ`" is the Lemma S2 change-of-measure object, not
  the treated-group mean of `m_0` at observed `X_{t*}(1)`.)
- **Assumption 9 (Product Rate for Nuisance Estimators)** (p. 27), verbatim: "The nuisance
  estimators η̂ = (m̂_0, ν̂_0, p̂, ω̂_0) satisfy
  ```
  (i)   ‖m̂_0 - m_0‖_2 · ‖ω̂_0 - ω_0‖_2 = o_p(n^{-1/2}),
  (ii)  ‖ν̂_0 - ν_0‖_2 · ‖p̂ - p‖_2 = o_p(n^{-1/2}),
  (iii) ‖m̂_0 - m_0‖_2 · ‖p̂ - p‖_2 = o_p(n^{-1/2})."
  ```
  All three hold when each nuisance converges at `o_p(n^{-1/4})` in `L^2`, "achievable by
  random forests (Wager and Athey 2018) and many other nonparametric and machine learners
  (Chernozhukov et al. 2018)" (p. 27). **Nested-error discussion (pp. 27-28):** the
  conditions are not independent. Since `ν_0 = T_{m_0} m_0` with `T_{m_0}` the
  conditional-expectation operator of (5), `‖ν̂_0 - ν_0‖_2` contains (a) the error from
  using `m̂_0` in the true operator, bounded by `‖m̂_0 - m_0‖_2` (conditional expectations
  are `L^2`-contractions), and (b) the second-stage approximation error
  `‖(T̂_{m_0} - T_{m_0}) m_0‖_2` from estimating the operator; both must converge fast
  enough for (ii). "Poor performance in estimating m_0 can spill over into poor performance
  in estimating ν_0. Likewise ... poor performance in estimating p can lead to poor
  performance in estimating ω_0."
- **Proposition 7 (asymptotic normality)** (p. 28), verbatim: "Under Assumptions 1, 2, 6, 7
  and 9, and S2, ATT̂_dr is consistent and satisfies
  ```
  √n( ATT̂_dr - ATT ) = (1/√n) Σ_{i=1}^n φ(O_i; η) + o_p(1) →_d N(0, V_dr),
  φ(O_i; η) := ( φ_1(O_i; η) - ATT ) - (ATT/π)(D_i - π),
  ```
  and V_dr = Var(φ(O; η)). V̂_dr, given in Algorithm 1, is consistent for V_dr." The
  `-(ATT/π)(D_i - π)` term comes from estimating `π` (SB.2: `ATT √n(π/π̂ - 1) = -(ATT/π)(1/√n) Σ (D_i - π) + o_p(1)`).
- **Assumption S2 (Regularity Conditions for DR Estimator)** (SA p. 7), verbatim:
  "(i) E[φ(O; η)^2] < ∞ and Var(φ(O; η)) > 0. (ii) ‖m̂_0^{-k} - m_0‖_2 = o_p(1),
  ‖ν̂_0^{-k} - ν_0‖_2 = o_p(1), ‖p̂^{-k} - p‖_2 = o_p(1), and ‖ω̂_0^{-k} - ω_0‖_2 = o_p(1)
  for each fold k. (iii) p(X_{t*-1}, W, Z) is uniformly bounded away from 1 and
  p̂^{-k}(X_{t*-1}, W, Z) is uniformly bounded away from 1 for each fold k. (iv) ω̂_0^{-k} is
  uniformly bounded for each fold k. (v) E[ΔY_{t*}^2 | X_{t*}, X_{t*-1}, Z, D = 0] is
  uniformly bounded."
- Proof of Proposition 7 (SB.2, SA pp. 7-11): `√n(ATT̂_dr - ATT) = A + B + o_p(1)` with
  `B = -(ATT/π)(1/√n) Σ_i (D_i - π)` and `A = T_1 + T_2 + T_3`, where
  `T_1 = (1/√n) Σ_i (φ_1(O_i, η) - ATT)`,
  `T_2 = √n Σ_k (|I_k|/n) E[φ_1(O_i, η̂^{-k}) - φ_1(O_i, η) | η̂^{-k}]`, and `T_3` its centered
  empirical-process counterpart. **Lemma S3** (SA pp. 8-10; `T_2 = o_p(1)`): expanding
  `E[T_2(k) | η̂^{-k}] = -A - B - C + D + E + F - G` over seven nuisance-difference terms,
  `C` and `E` cancel by LIE, `G = 0` by LIE, and the rest reduce to
  ```
  E[T_2(k) | η̂^{-k}] = ((1-π)/π) E[(ν̂_0^{-k} - ν_0)(p̂^{-k} - p)/((1-p)(1-p̂^{-k})) | η^{-k}, D=0]
                       - ((1-π)/π) E[(m̂_0^{-k} - m_0)(p̂^{-k} - p)/((1-p)(1-p̂^{-k})) | η^{-k}, D=0]
                       + ((1-π)/π) E[(m̂_0^{-k} - m_0)(ω̂_0^{-k} - ω_0) | η^{-k}, D=0]
                     ≤ C( ‖ν̂_0^{-k} - ν_0‖_2 ‖p̂^{-k} - p‖_2 + ‖m̂_0^{-k} - m_0‖_2 ‖p̂^{-k} - p‖_2 + ‖m̂_0^{-k} - m_0‖_2 ‖ω̂_0^{-k} - ω_0‖_2 ) = o_p(n^{-1/2})
  ```
  (Cauchy-Schwarz under S2 and Assumption 7; then Assumption 9 (ii), (iii), (i)
  respectively). **Lemma S4** (SA pp. 10-11; `T_3 = o_p(1)`): conditional variance bound
  `Var(φ̇_1(k) | η̂^{-k}) ≤ C(‖m̂_0 - m_0‖_2^2 + ‖ν̂_0 - ν_0‖_2^2 + ‖p̂ - p‖_2^2 + ‖ω̂_0 - ω_0‖_2^2) = o_p(1)` (S14),
  `Var(T_3(k) | η̂^{-k}) = (|I_k|/n) Var(φ̇_1(k) | η̂^{-k}) = o_p(1)` (S15), conditional
  Chebyshev `P(|T_3(k)| > ε | η̂^{-k}) ≤ Var(T_3(k) | η̂^{-k})/ε^2 ≤ δ/ε^2` (S16). Requires
  fixed `K` with every `|I_k|` of the same order as `n` (roughly equal folds) and nuisances
  for fold `k` fit on `I_{-k}` only.

*Standard errors (Propositions 4 and 7; SA Equation S8; Algorithm 1 step 4):*

- **Imputation estimator - influence function ψ^ra (Equation S8, SA p. 4), exactly:**
  ```
  ψ_i^{ra} := (D_i/π) ( ΔY_{it*} - E[ΔY_{t*} | D = 1] - ν_{0,i} + E[ν_0 | D = 1] )
              - [(1 - D_i)/(1 - π)] ( E[(D/π) R̃]' Σ_R^{-1} R_i u_i  +  E[(D/π) β_1' S'] Σ_S^{-1} S_i v_i )     (S8)
  ```
  with `π = P(D = 1)`, `ν_{0,i} = R̃_i'β`, `E[ν_0 | D = 1] = τ`, residuals
  `u_i = ΔY_{it*} - R_i'β`, `v_i = X_{it*} - S_i'γ` (`E[u | R, D = 0] = 0`, `E[v | S, D = 0] = 0`
  under Assumption 8), untreated second-moment matrices
  `Σ_R = E[(1-D)/(1-π) RR']`, `Σ_S = E[(1-D)/(1-π) SS']`, and treated-group means
  `E[(D/π) R̃]` (vector) and `E[(D/π) β_1' S']` = `β_1` times the treated mean of `S'` (S6
  writes this factor as `E[(D/π) β_1 S']`; same object, `β_1` scalar). Then
  `√n(ATT̂_ra - ATT) = (1/√n) Σ_i ψ_i^{ra} + o_p(1) →_d N(0, Var(ψ^ra))` by combining (S4),
  (S6), (S7). Derivation (SA pp. 2-4): **Lemma S1** (OLS linearizations, S1-S2:
  `√n(β̂ - β) = Σ_R^{-1} (1/√n) Σ_i [(1-D_i)/(1-π)] R_i u_i + o_p(1)`, analogously for `γ̂`);
  decomposition (S3) into `A = √n(m̂_1 - E[ΔY_{t*} | D = 1])` (S4) and
  `B = B_1 + B_2` where the generated-regressor term (S5)
  `ν̂_{0,i} - ν_{0,i} = S_i'(γ̂ - γ) β̂_1 + R̃_i'(β̂ - β)` yields (S6)
  `(1/√n) Σ_i (D_i/π̂)(ν̂_{0,i} - ν_{0,i}) = E[(D/π) β_1 S'] √n(γ̂ - γ) + E[(D/π) R̃'] √n(β̂ - β) + o_p(1)`,
  and (S7) handles `B_2`. Treated units contribute ONLY the first line of S8; untreated
  units ONLY the second (OLS-residual) line. A naive SE that treats `ν̂_0` as known omits the
  entire second line.
  **Note:** the paper never writes an explicit plug-in variance estimator `Ω̂` for the
  imputation estimator (contrast Algorithm 1 step 4); the natural implementation replaces
  `π, β, γ, Σ_R, Σ_S, E[·|D=1], u_i, v_i` by `π̂, β̂, γ̂`, untreated-sample second-moment
  matrices, treated means, and OLS residuals, and takes the sample variance of `ψ̂_i^{ra}`
  (`SE = √(Ω̂/n)`). Implementation-required; record as a REGISTRY Note when shipped.
- **DR / DDML estimator:** `V_dr = Var(φ(O; η))` with `φ(O; η) = (φ_1(O; η) - ATT) - (ATT/π)(D - π)`.
  Plug-in (Algorithm 1 step 4): `V̂_dr = n^{-1} Σ_{i=1}^n φ̂_i^2` with
  `φ̂_i = φ̂_{1,i} - ATT̂_dr - (ATT̂_dr/π̂)(D_i - π̂)` (cross-fitted score for unit `i`), and
  `se(ATT̂) = √(V̂_dr/n)`. `φ̂_i` has mean exactly zero by construction
  (`(1/n) Σ φ̂_{1,i} = ATT̂_dr`, `Σ (D_i - π̂) = 0`), so `V̂_dr` is the `1/n`-normalized sample
  variance. Proposition 7: `V̂_dr` is consistent for `V_dr`. No degrees-of-freedom or
  small-sample correction; no trimming rule (S2(iii)-(iv) are high-level conditions).
- **Bootstrap:** Not discussed in paper (neither the main text nor SA mentions any
  bootstrap; all inference is analytical via influence functions).
- **Clustering:** Not discussed in paper (unit-level i.i.d. sampling per Assumptions 1 /
  MP-3; influence functions are per unit).
- Normal critical values are implied throughout (`→_d N(0, ·)`); no t-distribution or
  small-sample adjustment is mentioned.

*Staggered adoption estimation (SA SB.3, SA pp. 11-12):*

- Scope: SB.3 covers "the estimator implied by the identification result in Proposition 2
  as we think this is the most likely result to be used in empirical work."
- **Per-(g,t) substitution rule** (verbatim): "Estimating ATT(g,t) based on this expression
  is as simple as replacing (Y_{it*} - Y_{it*-1}) with (Y_{it} - Y_{ig-1}), X_{it*} with
  X_{it}, and X_{it*-1} with X_{ig-1} in the preceding estimators." I.e. long difference
  from base period `g - 1` to `t`; bad control at `t` (inner set) and at `g - 1` (both
  sets); `W`, `Z` as in the two-period case (with `W = Y_{g-1}` per Remark 5). The
  comparison group is Proposition 2's not-yet-treated `D_t = 0` (SB.3 does not restate
  this; Section 5 does); the "treated" indicator is `G = g`.
- **Per-(g,t) influence functions:** "Consistency and asymptotic normality follow from
  effectively the same arguments as above and involve the same influence function and
  asymptotic variance up to adjusting the periods, which ... we denote by ψ_{g,t}(O_i) and
  φ_{g,t}(O_i), respectively" - `ψ_{g,t}` = (S8) and `φ_{g,t}` = Proposition 7's `φ`, per
  cell. The SA does not spell out the `(g,t)`-specific analog of `π` or of the `(D - π)`
  correction (see Gaps).
- **Remark 4 (Aggregating ATT(g,t)'s)** (p. 22), with `e` = exposure length and
  `𝒢_e := {g ∈ 𝒢̄ : g + e ∈ [2, T]}`:
  ```
  ATT^{es}(e) = Σ_{g ∈ 𝒢_e} ATT(g, g + e) P(G = g | G ∈ 𝒢_e)
  ATT^o       = Σ_{g ∈ 𝒢̄} Σ_{t=g}^{T} [ P(G = g | G ∈ 𝒢̄) / (T - g + 1) ] ATT(g, t)
  ```
  "See Callaway and Sant'Anna (2021) for details and more examples."
- **Aggregation inference (SB.3):** generic aggregate and estimator
  ```
  θ  = Σ_{g ∈ 𝒢̄} Σ_{t=g}^{T} w_θ(g,t) ATT(g,t),        θ̂ = Σ_{g ∈ 𝒢̄} Σ_{t=g}^{T} ŵ_θ(g,t) ATT̂(g,t)
  ```
  with event-study weights `1{t = g + e} P(G = g | G + e ≤ T)` and overall weights
  `P(G = g | G ∈ 𝒢̄)/(T - g + 1)`. High-level assumption on the estimated weights, for all
  `g ∈ 𝒢̄`, `t = g, ..., T`:
  ```
  √n( ŵ_θ(g,t) - w_θ(g,t) ) = (1/√n) Σ_{i=1}^n ξ_{g,t}^w(O_i) + o_p(1),   E[ξ_{g,t}^w] = 0, Var(ξ_{g,t}^w) finite and positive definite
  ```
  ("satisfied for all the weights considered in Callaway and Sant'Anna (2021) under mild
  regularity conditions"). Then
  ```
  √n( θ̂ - θ ) = (1/√n) Σ_{i=1}^n ξ_θ(O_i) + o_p(1) →_d N(0, E[ξ_θ(O)^2])
  ξ_θ(O) := Σ_{g ∈ 𝒢̄} Σ_{t=g}^{T} ( w_θ(g,t) ψ_{g,t}(O) + ξ_{g,t}^w(O) ATT(g,t) )
  ```
  "The expression above is for the influence function from the imputation estimator, but
  the same argument applies for the Neyman Orthogonal estimator with φ_{g,t} replacing
  ψ_{g,t}." Cross-cell covariance enters automatically through the per-unit sum; no
  separate multivariate CLT is stated. No multiplier bootstrap is mentioned; the
  Callaway and Sant'Anna (2021) reference is for the weight-IF `ξ^w_{g,t}`.

*Edge cases:*

- **Overlap:** Assumption 3 / 7 / MP-6 and S2(iii) bound the propensity away from 1 ONLY
  (`P(D = 1 | ·) < 1`; `p` and each `p̂^{-k}` "uniformly bounded away from 1"); no bound away
  from 0 is stated, consistent with the score never dividing by `p`. S2(iv): each `ω̂_0^{-k}`
  must be uniformly bounded (the estimated-odds regression must not blow up). **No trimming
  or clipping rule is given** - an implementation must choose one (clip `p̂` to `≤ 1 - κ`,
  bound `ω̂_0`) and document it as a deviation; flag as implementation-required.
- **Binary vs continuous bad control:** the paper's notation treats `X_t` as a generic
  real-valued random variable; the application's bad control (occupation score) and all
  simulation DGPs are continuous; SC's Assumptions S4-S5 and Corollary S1 estimator are
  linear-regression based on a real-valued `X`. Remark 3 notes distributional-DiD
  alternatives are "less suitable for discrete or mixed discrete-continuous covariates",
  which is the paper's motivation for the unconfoundedness route (Assumptions 4/6 make no
  continuity requirement). No binary/discrete-specific handling is discussed anywhere
  (with a binary `X`, Assumption 8's linear `E[X_{t*} | ·]` model is a linear probability
  model; not addressed in the paper).
- **Vector bad control:** scalar in all notation; "straightforward to allow `X_it` to be a
  vector" (p. 5). Assumption 8 / S8 would need matrix `β_1`, `γ`.
- **Already-treated group:** dropped (p. 18; the application drops workers displaced in
  1992). **No group treated in period 1** by construction of `𝒢 ⊆ {2, ..., T} ∪ {∞}`.
- **Never-treated vs not-yet-treated:** Theorem 3 requires never-treated; Propositions 2-3
  use not-yet-treated, and the paper says the same arguments rationalize never-treated for
  Proposition 2 (p. 21).
- **Panel only:** Remark 1 (p. 11); all estimators difference within unit; the RCS setting
  is "to a large extent" unavailable.
- **Sample construction in the application** (p. 29): balanced panel with positive earnings
  and non-missing outcome, bad control, and covariates in EVERY period; no missing-data
  handling is described.
- **Time-invariant `Z`:** absorbed by TWFE (Estimators 1-2 carry no covariates) but usable
  by all CS-style and Section 6 estimators (p. 30).
- **`W` choice:** application uses `W` = pre-treatment `log(Earnings)` (Remarks 2/5); the
  simulations use a separate `W` correlated with unobserved heterogeneity `η` and entering
  treatment assignment.
- **Nonlinearity in the bad-control model is the failure mode of the parametric
  estimators** (SD): DGP 3 (`X_1 Z`, `X_1^2`) breaks Imputation and PT for X; DGP 5
  (`W^2`, `X_1 W`) degrades them; parametric DR stays nearly unbiased in all five DGPs.
- **ML SE calibration:** random-forest ML can under-cover at larger `n` in strongly
  nonlinear DGPs (SE/SD 0.60 in DGP 3 at `n = 2000`); parametric DR in DGP 3 has SE/SD
  0.76-0.87 with a much larger SD than the other estimators.
- **Few treated / small samples / unbalanced panels / propensity extremes:** not discussed
  (simulations use `n ∈ {500, 1000, 2000}` with roughly half treated).
- **Two-step overfitting in ML nuisances** (footnote 9, p. 31): `ν̂_0^{-k}` and `ω̂_0^{-k}`
  regress on pseudo-outcomes built from `m̂_0^{-k}` and `p̂^{-k}`; using the same training
  units for both steps can overfit. The authors use each unit's **out-of-bag** first-stage
  prediction as the pseudo-outcome; for learners without OOB predictions, further split the
  training fold (Farbmacher et al. 2022).

*Algorithm (Algorithm 1 in paper, p. 27: "DDML Estimator of ATT under Covariate Unconfoundedness"), transcribed exactly:*

**Input:** Data `{(ΔY_{it*}, D_i, X_{it*}, X_{i,t*-1}, W_i, Z_i)}_{i=1}^n`; number of folds `K`.
1. Partition `{1, ..., n}` into `K` folds `{I_1, ..., I_K}`. Set `π̂ = n^{-1} Σ_{i=1}^n D_i`.
2. For each fold `k = 1, ..., K`, estimate nuisance functions on training sample `I_{-k}`:
   (a) *First stage.* Estimate `m̂_0^{-k}` using untreated units, and `p̂^{-k}` using all units.
   (b) *Second stage.* Given the first stage estimate of `m̂_0^{-k}`, estimate `ν̂_0^{-k}`.
   Likewise, given the first stage estimate of `p̂^{-k}`, estimate `ω̂_0^{-k}`, both using
   untreated units.
   (c) *Third Stage.* Evaluate the doubly robust score `φ̂_{1,i}^k` on held-out fold `I_k`
   based on Equation (10) with cross-fitted nuisance estimates,
   `(m̂_0^{-k}, ν̂_0^{-k}, p̂^{-k}, ω̂_0^{-k})`.
3. Compute `ATT̂_dr = n^{-1} Σ_{k=1}^K Σ_{i ∈ I_k} φ̂_{1,i}^k`.
4. Compute `V̂_dr = n^{-1} Σ_{i=1}^n φ̂_i^2`, and `se(ATT̂) = √(V̂_dr / n)`, where
   `φ̂_i = φ̂_{1,i} - ATT̂_dr - (ATT̂_dr / π̂)(D_i - π̂)`.

"The above algorithm implements a cross-fitting estimation procedure as in Chernozhukov
et al. (2018)." Reading notes: `I_{-k}` is the complement of `I_k`; `ν̂_0^{-k}` regresses
`m̂_0^{-k}(X_{t*}, X_{t*-1}, Z)` on `(X_{t*-1}, W, Z)` over untreated units of `I_{-k}`
(Equation 5); `ω̂_0^{-k}` regresses `p̂^{-k}/(1 - p̂^{-k})` on `(X_{t*}, X_{t*-1}, Z)` over
untreated units of `I_{-k}` (Equation 9); `π̂` is computed ONCE on the full sample (step 1),
not per fold; step 3 averages the (already `π̂`-normalized) score over all `n` units, which
equals Equation 11. No repeated cross-fitting / median aggregation is mentioned. For the
imputation estimator (Section 6.1) there is no cross-fitting: the two OLS fits use the full
untreated sample.

**Remark 7 (Monte Carlo Simulations)** (p. 28): pointer to SA SD. **Remark 8 (TWFE
Regressions)** (p. 28): the estimators build on Sant'Anna and Zhao (2020) and Callaway and
Sant'Anna (2021) rather than
```
Y_{it} = θ_t + η_i + α D_{it} + X_{it} β + Z_i δ_t + e_{it}
```
Caetano and Callaway (2025) document TWFE limitations from treatment-effect heterogeneity
and "hidden linearity bias" (differencing out `η_i` also differences the time-varying
covariates); with a bad control an additional bias arises if `X_{it}` is included
(Section 3.1) and a different one if excluded (Section 3.2); "because our approach above
effectively dealt with the bad control in a distinct step from recovering the ATT, it seems
difficult to operationalize a TWFE version of our approaches Sections 4 and 5."

**Supplementary Appendix SC (parallel trends for the bad control under linearity; SA pp. 12-14):**

Two-period setting ("the extension to staggered treatment adoption follows along the same
lines as the arguments in Section 5"); `X_t` generic real-valued.

- **Assumption S3 (Bad Control Parallel Trends):** `E[ΔX_{t*}(0) | W, Z, D = 1] = E[ΔX_{t*}(0) | W, Z, D = 0]`.
  Under S3, by standard DiD arguments,
  ```
  ATT_X := E[X_{t*}(1) - X_{t*}(0) | D = 1] = E[ΔX_{t*} | D = 1] - E[ E[ΔX_{t*} | W, Z, D = 0] | D = 1 ]      (S17)
  ```
  "identifying ATT_X is not sufficient for identifying ATT as it depends on the entire
  distribution of X_{t*}(0) for the treated group (see Equation (3) in the main text)."
- **Assumption S4 (Linear Conditional Expectation of ΔY_{t*}(0)):**
  `E[ΔY_{t*}(0) | X_{t*}(0), X_{t*-1}, Z] = X_{t*}(0) β_1 + X_{t*-1} β_2 + Z' β_3`;
  `(β_1, β_2, β_3)` identified from the untreated-group regression of `ΔY_{t*}` on
  `(X_{t*}, X_{t*-1}, Z)` - "the same assumption as we used for the imputation estimator".
- **Proposition S1.** Under Assumptions 1, 2, 7, S3, and S4,
  ```
  ATT = E[ΔY_{t*} | D=1] - ( E[ E[ΔX_{t*} | W, Z, D=0] | D=1 ] β_1 + E[X_{t*-1} | D=1] (β_2 + β_1) + E[Z | D=1]' β_3 )
  ```
  Proof (SA pp. 13-14): Assumption 2 (as in Theorems 1-2), Assumption S4, add/subtract
  `E[X_{t*-1} | D=1] β_1`, then the S17 argument. Linearity makes ATT depend on `X_{t*}(0)`
  only through its conditional mean.
- **Assumption S5 (Linear Conditional Expectation of ΔX_{t*}(0)):** `E[ΔX_{t*}(0) | W, Z] = W' δ_1 + Z' δ_2`;
  under S3, `(δ_1, δ_2)` identified from the untreated-group regression of `ΔX_{t*}` on `(W, Z)`.
- **Corollary S1.** Under Assumptions 1, 2, 7, S3, S4, and S5,
  ```
  ATT = E[ΔY_{t*} | D=1] - ( E[W | D=1]' δ_1 β_1 + E[X_{t*-1} | D=1] (β_2 + β_1) + E[Z | D=1]' (δ_2 β_1 + β_3) )
  ```
- **Implied estimator (SA p. 14):** first estimate `(β̂_1, β̂_2, β̂_3)` from `ΔY_{t*}` on
  `(X_{t*}, X_{t*-1}, Z)` and `(δ̂_1, δ̂_2)` from `ΔX_{t*}` on `(W, Z)`, both on the untreated
  group; then
  ```
  ATT̂ = (1/n) Σ_{i=1}^{n} (D_i / π̂) ( ΔY_{i,t*} - W_i' δ̂_1 β̂_1 - X_{i,t*-1} (β̂_2 + β̂_1) - Z_i' (δ̂_2 β̂_1 + β̂_3) )
  ```
  This is the `PT for X` estimator (Estimator 5) of the application and simulations. No
  variance formula is given for it in SC (simulation SE/SD ratios near 1 show that the
  authors' implementation carries analytical SEs - not printed).

**Monte Carlo evidence (SA SD, SA pp. 14-23; Tables S1-S6, Figure S1):**

Common DGP (SA pp. 14-15): two periods (`t = 1, 2`; `t* = 2`); `Z_i ~ N(0,1)`, `η_i ~ N(0,1)`
(unobserved heterogeneity) independent; noise `ε_i^W, ε_i^D, ε_i^{X_1}, ε_i^{X_2}, ε_{i1}^Y, ε_{i2}^Y ~ N(0,1)`
mutually independent and independent of `(Z_i, η_i)`.
```
W_i        = 0.8 η_i + 0.3 Z_i + 0.2 ε_i^W
D_i        = 1{ 0.2 Z_i + 0.4 W_i + 0.3 η_i + ε_i^D > 0 }
X_i1       = 0.5 η_i + 0.4 Z_i + 0.3 ε_i^{X_1}
X_i2(1)    = X_i2(0) + λ,   λ = 0.5
Y_it(0)    = θ_t + 0.5 η_i + 0.3 Z_i + X_it(0) + 0.3 ε_it^Y,   θ_1 = 0.3, θ_2 = 0.6
Y_{it*}(1) = Y_{it*}(0) + (X_{it*}(1) - X_{it*}(0)) + δ,   δ = 0.5
True ATT   = δ + λ = 1.00

DGP 1 (Covariate Unconfoundedness, Linear W):                 X_i2(0) = 0.7 X_i1 + 0.3 Z_i + 0.2 W_i + 0.15 + 0.3 ε_i^{X_2}
DGP 2 (Covariate Unconfoundedness, Nonlinear W):              X_i2(0) = 0.7 X_i1 + 0.3 Z_i + 0.2 W_i + 0.03 W_i^2 + 0.15 + 0.3 ε_i^{X_2}
DGP 3 (Simple Covariate Unconfoundedness, Nonlinear X):       X_i2(0) = 0.7 X_i1 + 0.3 Z_i + 0.4 X_i1 Z_i + 0.2 X_i1^2 + 0.15 + 0.3 ε_i^{X_2}
DGP 4 (Bad Control Parallel Trends):                          X_i2(0) = X_i1 + 0.3 Z_i + 0.2 W_i + 0.15 + 0.3 ε_i^{X_2}
DGP 5 (Covariate Unconfoundedness, Nonlinear W, X/W Interaction): X_i2(0) = 0.7 X_i1 + 0.3 Z_i + 0.2 W_i + 0.03 W_i^2 + 0.05 X_i1 W_i + 0.15 + 0.3 ε_i^{X_2}
```
Untreated potential outcome is linear in `X_it(0)` in every DGP; DGPs differ only in
`X_i2(0)`. Methods: seven of the nine application estimators (`Imp. include BC` and
`Imp. exclude BC` omitted "for conciseness"). 1000 replications; `n ∈ {500, 1000, 2000}`;
metrics: Bias (mean of `ATT̂ - ATT`), SD, RMSE, SE/SD (mean analytical SE over SD), coverage
of nominal 95% CIs (SA footnote 1, p. 16: SE/SD near 1.0 = well-calibrated SEs).

**Table S1: Theoretical Consistency of Each Estimator, by DGP** (SA p. 15)

| Method | dgp1 | dgp2 | dgp3 | dgp4 | dgp5 |
|---|---|---|---|---|---|
| TWFE Include BC | x | x | x | x | x |
| TWFE Exclude BC | x | x | x | x | x |
| PT for X | x | x | x | ok | x |
| ML (Pre-treatment) | x | x | ok | x | x |
| Imputation | ok | x | x | ok | x |
| DR | ok | x | x | ok | x |
| ML | ok | ok | ok | ok | ok |

(ok = consistent, x = not; printed as check marks / crosses.)

**Table S2: Monte Carlo Results: DGP 1** (SA p. 19)

| Method | n | Bias | SD | RMSE | SE/SD | Coverage |
|---|---|---|---|---|---|---|
| TWFE Include BC | 500 | -0.500 | 0.051 | 0.502 | 0.97 | 0.000 |
| TWFE Include BC | 1000 | -0.499 | 0.035 | 0.501 | 0.98 | 0.000 |
| TWFE Include BC | 2000 | -0.501 | 0.025 | 0.501 | 1.00 | 0.000 |
| TWFE Exclude BC | 500 | 0.015 | 0.049 | 0.051 | 0.99 | 0.932 |
| TWFE Exclude BC | 1000 | 0.013 | 0.036 | 0.038 | 0.95 | 0.924 |
| TWFE Exclude BC | 2000 | 0.013 | 0.025 | 0.028 | 0.99 | 0.918 |
| PT for X | 500 | -0.002 | 0.057 | 0.057 | 1.00 | 0.954 |
| PT for X | 1000 | -0.004 | 0.042 | 0.042 | 0.97 | 0.948 |
| PT for X | 2000 | -0.005 | 0.029 | 0.029 | 0.99 | 0.945 |
| ML (Pre-treatment) | 500 | 0.031 | 0.059 | 0.067 | 1.01 | 0.917 |
| ML (Pre-treatment) | 1000 | 0.036 | 0.043 | 0.057 | 1.00 | 0.870 |
| ML (Pre-treatment) | 2000 | 0.042 | 0.031 | 0.052 | 1.02 | 0.749 |
| Imputation | 500 | 0.002 | 0.056 | 0.056 | 1.00 | 0.954 |
| Imputation | 1000 | -0.000 | 0.041 | 0.041 | 0.96 | 0.947 |
| Imputation | 2000 | -0.000 | 0.028 | 0.028 | 1.00 | 0.947 |
| DR | 500 | 0.003 | 0.062 | 0.062 | 0.97 | 0.948 |
| DR | 1000 | -0.000 | 0.043 | 0.043 | 0.96 | 0.941 |
| DR | 2000 | 0.000 | 0.029 | 0.029 | 0.99 | 0.948 |
| ML | 500 | 0.011 | 0.060 | 0.061 | 0.99 | 0.935 |
| ML | 1000 | 0.007 | 0.045 | 0.045 | 0.97 | 0.940 |
| ML | 2000 | 0.007 | 0.032 | 0.033 | 0.97 | 0.937 |

**Table S3: Monte Carlo Results: DGP 2** (SA p. 20)

| Method | n | Bias | SD | RMSE | SE/SD | Coverage |
|---|---|---|---|---|---|---|
| TWFE Include BC | 500 | -0.499 | 0.050 | 0.501 | 0.99 | 0.000 |
| TWFE Include BC | 1000 | -0.501 | 0.035 | 0.503 | 1.00 | 0.000 |
| TWFE Include BC | 2000 | -0.499 | 0.025 | 0.500 | 0.99 | 0.000 |
| TWFE Exclude BC | 500 | 0.013 | 0.048 | 0.050 | 1.00 | 0.941 |
| TWFE Exclude BC | 1000 | 0.012 | 0.034 | 0.036 | 1.01 | 0.939 |
| TWFE Exclude BC | 2000 | 0.013 | 0.024 | 0.027 | 1.02 | 0.910 |
| PT for X | 500 | 0.016 | 0.059 | 0.061 | 0.96 | 0.929 |
| PT for X | 1000 | 0.015 | 0.040 | 0.042 | 1.02 | 0.940 |
| PT for X | 2000 | 0.016 | 0.028 | 0.032 | 1.02 | 0.922 |
| ML (Pre-treatment) | 500 | 0.036 | 0.061 | 0.071 | 0.98 | 0.902 |
| ML (Pre-treatment) | 1000 | 0.041 | 0.041 | 0.058 | 1.06 | 0.859 |
| ML (Pre-treatment) | 2000 | 0.046 | 0.030 | 0.055 | 1.07 | 0.706 |
| Imputation | 500 | 0.021 | 0.058 | 0.062 | 0.97 | 0.925 |
| Imputation | 1000 | 0.019 | 0.039 | 0.043 | 1.02 | 0.931 |
| Imputation | 2000 | 0.020 | 0.028 | 0.034 | 1.02 | 0.901 |
| DR | 500 | 0.002 | 0.063 | 0.063 | 0.97 | 0.938 |
| DR | 1000 | 0.002 | 0.042 | 0.042 | 1.00 | 0.951 |
| DR | 2000 | 0.003 | 0.029 | 0.030 | 0.99 | 0.952 |
| ML | 500 | 0.016 | 0.061 | 0.063 | 0.98 | 0.931 |
| ML | 1000 | 0.014 | 0.042 | 0.044 | 1.03 | 0.940 |
| ML | 2000 | 0.011 | 0.032 | 0.034 | 0.99 | 0.939 |

**Table S4: Monte Carlo Results: DGP 3** (SA p. 21)

| Method | n | Bias | SD | RMSE | SE/SD | Coverage |
|---|---|---|---|---|---|---|
| TWFE Include BC | 500 | -0.502 | 0.040 | 0.504 | 1.02 | 0.000 |
| TWFE Include BC | 1000 | -0.499 | 0.029 | 0.500 | 1.01 | 0.000 |
| TWFE Include BC | 2000 | -0.500 | 0.020 | 0.500 | 1.00 | 0.000 |
| TWFE Exclude BC | 500 | -0.128 | 0.060 | 0.141 | 1.00 | 0.409 |
| TWFE Exclude BC | 1000 | -0.128 | 0.041 | 0.134 | 1.02 | 0.143 |
| TWFE Exclude BC | 2000 | -0.127 | 0.030 | 0.131 | 1.00 | 0.017 |
| PT for X | 500 | 0.165 | 0.084 | 0.185 | 0.97 | 0.474 |
| PT for X | 1000 | 0.163 | 0.057 | 0.173 | 1.02 | 0.193 |
| PT for X | 2000 | 0.163 | 0.040 | 0.168 | 1.01 | 0.013 |
| ML (Pre-treatment) | 500 | 0.009 | 0.067 | 0.068 | 1.08 | 0.961 |
| ML (Pre-treatment) | 1000 | 0.007 | 0.046 | 0.047 | 1.11 | 0.957 |
| ML (Pre-treatment) | 2000 | 0.005 | 0.043 | 0.044 | 0.85 | 0.952 |
| Imputation | 500 | 0.172 | 0.084 | 0.192 | 0.99 | 0.445 |
| Imputation | 1000 | 0.171 | 0.057 | 0.180 | 1.02 | 0.169 |
| Imputation | 2000 | 0.170 | 0.042 | 0.175 | 1.00 | 0.009 |
| DR | 500 | 0.004 | 0.157 | 0.157 | 0.81 | 0.896 |
| DR | 1000 | 0.013 | 0.103 | 0.104 | 0.87 | 0.889 |
| DR | 2000 | 0.027 | 0.080 | 0.085 | 0.76 | 0.845 |
| ML | 500 | 0.010 | 0.070 | 0.070 | 1.04 | 0.956 |
| ML | 1000 | 0.008 | 0.052 | 0.053 | 0.99 | 0.951 |
| ML | 2000 | 0.013 | 0.063 | 0.064 | 0.60 | 0.892 |

**Table S5: Monte Carlo Results: DGP 4** (SA p. 22)

| Method | n | Bias | SD | RMSE | SE/SD | Coverage |
|---|---|---|---|---|---|---|
| TWFE Include BC | 500 | -0.500 | 0.051 | 0.503 | 1.04 | 0.000 |
| TWFE Include BC | 1000 | -0.501 | 0.039 | 0.503 | 0.96 | 0.000 |
| TWFE Include BC | 2000 | -0.499 | 0.026 | 0.500 | 1.00 | 0.000 |
| TWFE Exclude BC | 500 | 0.140 | 0.048 | 0.148 | 1.02 | 0.194 |
| TWFE Exclude BC | 1000 | 0.140 | 0.036 | 0.144 | 0.97 | 0.022 |
| TWFE Exclude BC | 2000 | 0.139 | 0.025 | 0.141 | 0.98 | 0.000 |
| PT for X | 500 | 0.000 | 0.054 | 0.054 | 1.04 | 0.956 |
| PT for X | 1000 | -0.001 | 0.040 | 0.040 | 1.00 | 0.949 |
| PT for X | 2000 | -0.001 | 0.029 | 0.029 | 0.96 | 0.933 |
| ML (Pre-treatment) | 500 | 0.043 | 0.057 | 0.071 | 1.07 | 0.910 |
| ML (Pre-treatment) | 1000 | 0.042 | 0.042 | 0.060 | 1.05 | 0.846 |
| ML (Pre-treatment) | 2000 | 0.043 | 0.032 | 0.054 | 1.00 | 0.727 |
| Imputation | 500 | 0.000 | 0.054 | 0.054 | 1.05 | 0.953 |
| Imputation | 1000 | -0.001 | 0.040 | 0.040 | 1.00 | 0.949 |
| Imputation | 2000 | -0.001 | 0.029 | 0.029 | 0.96 | 0.934 |
| DR | 500 | 0.001 | 0.058 | 0.058 | 1.03 | 0.951 |
| DR | 1000 | -0.001 | 0.041 | 0.041 | 1.00 | 0.946 |
| DR | 2000 | -0.000 | 0.030 | 0.030 | 0.95 | 0.929 |
| ML | 500 | 0.021 | 0.058 | 0.061 | 1.06 | 0.945 |
| ML | 1000 | 0.013 | 0.043 | 0.045 | 1.02 | 0.937 |
| ML | 2000 | 0.006 | 0.033 | 0.034 | 0.95 | 0.935 |

**Table S6: Monte Carlo Results: DGP 5** (SA p. 23)

| Method | n | Bias | SD | RMSE | SE/SD | Coverage |
|---|---|---|---|---|---|---|
| TWFE Include BC | 500 | -0.503 | 0.050 | 0.506 | 0.97 | 0.000 |
| TWFE Include BC | 1000 | -0.499 | 0.036 | 0.501 | 0.96 | 0.000 |
| TWFE Include BC | 2000 | -0.500 | 0.025 | 0.500 | 1.00 | 0.000 |
| TWFE Exclude BC | 500 | 0.011 | 0.048 | 0.050 | 1.00 | 0.944 |
| TWFE Exclude BC | 1000 | 0.013 | 0.037 | 0.039 | 0.94 | 0.923 |
| TWFE Exclude BC | 2000 | 0.013 | 0.024 | 0.027 | 1.03 | 0.918 |
| PT for X | 500 | 0.037 | 0.057 | 0.068 | 1.02 | 0.901 |
| PT for X | 1000 | 0.041 | 0.043 | 0.060 | 0.94 | 0.830 |
| PT for X | 2000 | 0.040 | 0.028 | 0.049 | 1.04 | 0.719 |
| ML (Pre-treatment) | 500 | 0.040 | 0.059 | 0.071 | 1.03 | 0.901 |
| ML (Pre-treatment) | 1000 | 0.049 | 0.046 | 0.067 | 0.97 | 0.798 |
| ML (Pre-treatment) | 2000 | 0.052 | 0.031 | 0.060 | 1.05 | 0.641 |
| Imputation | 500 | 0.042 | 0.055 | 0.069 | 1.03 | 0.892 |
| Imputation | 1000 | 0.046 | 0.043 | 0.063 | 0.94 | 0.788 |
| Imputation | 2000 | 0.044 | 0.027 | 0.052 | 1.04 | 0.646 |
| DR | 500 | -0.002 | 0.066 | 0.066 | 0.98 | 0.958 |
| DR | 1000 | 0.005 | 0.048 | 0.049 | 0.93 | 0.923 |
| DR | 2000 | 0.008 | 0.033 | 0.034 | 0.95 | 0.927 |
| ML | 500 | 0.020 | 0.059 | 0.062 | 1.01 | 0.937 |
| ML | 1000 | 0.020 | 0.046 | 0.050 | 0.95 | 0.911 |
| ML | 2000 | 0.017 | 0.033 | 0.037 | 0.97 | 0.900 |

**Figure S1** (SA p. 18): kernel densities of `ATT̂` over 1,000 replications at `n = 2,000`,
one panel per DGP (a)-(e), dashed line at 1.00 (the figure note says "for the four main
DGPs" though five panels are shown - see Gaps). DGPs 1, 2, 5: `TWFE Include BC` isolated at
~0.50, all others near 1.00 with `ML (Pre-treatment)` shifted right (~1.04-1.05); DGP 3:
`TWFE Exclude BC` ~0.87, `PT for X` and `Imputation` ~1.17, `DR` centered near 1.00 but
visibly wider; DGP 4: `TWFE Exclude BC` ~1.14.

Headline findings (authors' summary, SA pp. 16-17, with table facts):
- `TWFE: include BC`: bias `≈ -0.500 = -λ` in every cell (it nets out the treatment's effect
  on the bad control); coverage 0.000 everywhere - "traditional approaches that directly
  include the bad control can perform extremely poorly."
- `TWFE: exclude BC`: severely biased in DGP 3 (-0.128) and DGP 4 (+0.140); small positive
  bias (0.011-0.015) in DGPs 1, 2, 5 with coverage drifting down in `n`. SA footnote 2: its
  decent performance in some DGPs "is an artifact of the parameters of these DGPs ... It is
  also noteworthy that this approach is biased in all five DGPs, which results in coverage
  rates shrinking in the sample size."
- `PT for X`: unbiased with nominal coverage in DGP 4 (consistent) and DGP 1; "performs
  decently well" elsewhere except DGP 3 (bias +0.163, coverage 0.013 at n = 2000) and, to
  some extent, DGP 5 (+0.040, coverage 0.719).
- `ML: pre-treatment`: small but non-zero bias (+0.03 to +0.05) in DGPs 1, 2, 4, 5 with
  coverage falling in `n` (0.749, 0.706, 0.727, 0.641 at n = 2000); correct in DGP 3.
  Authors' conjecture: for both `PT for X` and `ML: pre-treatment`, "in most realistic
  applications, the alternative assumptions imposed by these approaches are likely to, in
  some sense, be closer to holding than assumptions that lead to fully including or
  excluding the bad control."
- `Imputation`, `DR`, `ML` "all ... perform decently well across DGPs"; main exception is
  Imputation in DGP 3 (bias +0.170, coverage 0.009 at n = 2000) and to some extent DGP 5
  (+0.044, coverage 0.646); DGP 2 mild bias (+0.020, coverage 0.901).
- `DR` (parametric): bias `≤ 0.027` in every cell; "even in the DGPs where its functional
  form assumptions do not hold, the DR estimator performs almost on par with or sometimes
  slightly better than the ML estimator." But in DGP 3 its SD is much larger
  (0.157/0.103/0.080) with SE/SD 0.81/0.87/0.76 and coverage 0.896/0.889/0.845.
- `ML`: the only estimator consistent in all five DGPs, but "sometimes undercovers. This is
  driven by standard errors that are sometimes too small relative to the standard deviation
  of the estimator" (worst: DGP 3, n = 2000, SE/SD 0.60, coverage 0.892).

**Empirical application (Section 7, pp. 29-34; Table 1, Figures 6-8):**

- **Data (7.1):** NLSY79, biennial 1992-2002 (respondents aged 28-35 in 1992). Outcome:
  `log(Earnings)`. Treatment: job displacement (involuntarily left a job in the previous two
  years, reason "layoff/job eliminated" or "plant/company/office/workplace closed";
  excludes quits, firings, temporary-job endings), recorded as the first displacement year
  - staggered adoption. Bad control `X`: **occupation score** = median log hourly wage of
  employed wage/salary workers aged 16-64 in the worker's occupation (IPUMS USA 1990 5%
  sample); constant per occupation over time, so a worker's score changes only on
  occupation change. Rationale: parallel trends "seems more plausible if it holds
  conditional on the occupation a worker would hold in the absence of job displacement"
  and "job displacement can shift workers into different occupations" (Kambourov and
  Manovskii 2009; Gathmann and Schönberg 2010). `Z`: race, sex, years of education
  (time-invariant). `W`: the individual's `log(Earnings)` in the pre-treatment period.
  Sample: balanced panel of 3,231 individuals with positive earnings and non-missing
  outcome / bad control / covariates in every period; displaced `N = 748` (displaced in any
  period 1994-2002), non-displaced `N = 2,483`; workers displaced in 1992 dropped.
- **Table 1: Summary Statistics** (p. 30; means, SDs in parentheses; Diff = displaced minus
  non-displaced):

  | Variable | Displaced | Non-displaced | Diff | se(Diff) |
  |---|---|---|---|---|
  | Log earnings, 1992 | 9.74 (0.88) | 9.91 (0.79) | -0.17 | 0.04 |
  | Log earnings, 2002 | 10.30 (0.84) | 10.52 (0.82) | -0.22 | 0.03 |
  | Occupation score, 1992 | 2.25 (0.33) | 2.32 (0.35) | -0.07 | 0.01 |
  | Occupation score, 2002 | 2.30 (0.32) | 2.36 (0.34) | -0.06 | 0.01 |
  | Black (%) | 29.4 (45.6) | 23.8 (42.6) | 5.7 | 1.9 |
  | Hispanic (%) | 19.9 (40.0) | 16.2 (36.9) | 3.7 | 1.6 |
  | Female (%) | 43.4 (49.6) | 47.3 (49.9) | -3.8 | 2.1 |
  | Education (years) | 13.57 (2.32) | 14.12 (2.63) | -0.56 | 0.10 |
  | N | 748 | 2,483 | | |

  Authors' reading: displaced workers have lower earnings in both years (gap larger in
  2002), lower occupation scores in both years "though the difference is roughly constant
  over time", less education, more likely Black or Hispanic, less likely female.
- **The nine estimators** (pp. 30-31, verbatim labels): (1) `TWFE: include BC` - TWFE with
  `D_it` and `X_it`; (2) `TWFE: exclude BC` - TWFE with `D_it` only; (3) `Imp: include BC`
  - Callaway and Sant'Anna (2021)-based imputation estimator with the bad control in both
  periods as a covariate; (4) `Imp: exclude BC` - same without the bad control in either
  period; (5) `PT for X` - imputation estimator under Assumption S3 / Proposition S1;
  (6) `ML: pre-treatment` - ML estimator under simple covariate unconfoundedness with the
  pre-treatment bad control as a covariate, group-time ATTs per Sant'Anna and Zhao (2020)
  with random-forest nuisances ("the same implementation as ML below, but without its bad
  control nuisance steps"); (7) `Imputation` - Section 6.1 under covariate
  unconfoundedness; (8) `DR` - Section 6.2 with parametric nuisances; (9) `ML` - Section 6.2
  with ML nuisances (footnote 9). Estimators 1-2 include no covariates (all `Z` are
  time-invariant); 3-9 include race, sex, education, estimate group-time ATTs, then
  aggregate; 7-9 use `W_i` = pre-treatment `log(Earnings)`.
- **Footnote 9 (ML details, p. 31):** cross-fitting with **five folds**; every nuisance by
  random forests via R `grf` (Tibshirani, Athey, Sverdrup, and Wager 2026) with `grf`'s
  default tuning; out-of-bag first-stage predictions as pseudo-outcomes for the second-stage
  `ν̂_0^{-k}` / `ω̂_0^{-k}` fits (or further-split the training fold per Farbmacher et al.
  2022 for learners without OOB predictions).
- **Footnote 10 (software, p. 31):** Estimators 1-2 via R `fixest` (Bergé, Butts, and
  McDermott 2026); 3-4 via R `ptetools` (Callaway 2026); 5-9 via the authors' R
  `badcontrols` package (Caetano, Callaway, Payne, and Sant'Anna 2026).
- **Event-study conventions** (Figure 6/7 notes): `e = 0` is the displacement period;
  post-treatment estimates use the period immediately before displacement as base;
  pre-treatment estimates are pseudo-ATTs using the immediately preceding period as base
  ("each is the estimate that would be obtained by treating that period as the start of
  displacement"; cf. Remark 6).
- **Pre-test "Is the Occupation Score a Bad Control?"** (pp. 31-32, Figure 6): maintaining
  MP-5 and "implementing the first step of the Section 6 imputation estimator" with the
  occupation score as outcome gives **overall `ATT_X = -0.0261 (SE 0.0092)`**; event study
  (visual readings): e = -8 ~-0.003; -6 ~+0.005; -4 ~-0.020; -2 ~+0.005; 0 ~-0.033;
  2 ~-0.020; 4 ~+0.001; 6 ~-0.009; 8 ~0.000. All pre-period error bars cover zero; e = 0
  excludes zero. Interpretation: "job displacement results in their moving to an occupation
  with 2.6% lower wages than they would have been in absent job displacement"; larger for
  workers who actually switch; largest at displacement, shrinking afterwards.
- **Main results** (pp. 32-34, Figures 7-8): **baseline `Imputation` overall
  `ATT = -0.0672 (SE 0.0242)`**; event study (visual): e = -8 ~-0.015; -6 ~-0.010;
  -4 ~-0.013; -2 ~+0.008; 0 ~-0.100; 2 ~-0.125; 4 ~-0.050; 6 ~+0.035; 8 ~+0.045
  (pre-periods near zero; e = 0, 2 significant; e ≥ 4 not). "job displacement reduced
  displaced workers' earnings by about 7% ... concentrated in the four years following
  displacement and ... broadly in line with the job displacement literature." Figure 8
  plots each estimator's overall ATT minus the baseline; exact differences are NOT printed
  (visual readings, not validation targets):

  | Estimator | Difference from Imputation (approx.) | Approx. error bar | Implied ATT (approx.) | Text statement |
  |---|---|---|---|---|
  | TWFE: include BC | ~-0.028 | ~[-0.054, -0.002] | ~-0.095 | "almost 10%", "roughly 40% larger in magnitude" |
  | TWFE: exclude BC | ~-0.030 | ~[-0.056, -0.004] | ~-0.097 | same |
  | Imp: include BC | ~+0.003 | ~[-0.003, +0.007] | ~-0.065 | "3-4% smaller in magnitude" |
  | Imp: exclude BC | ~+0.002 | ~[-0.001, +0.004] | ~-0.065 | same |
  | PT for X | ~+0.0015 | ~[-0.001, +0.003] | ~-0.066 | "similar estimate to our baseline" |
  | ML: pre-treatment | ~-0.006 | ~[-0.011, -0.001] | ~-0.073 | "9% larger in magnitude ... statistically significant" |
  | DR | ~-0.001 | ~[-0.002, 0.000] | ~-0.068 | "quite similar" |
  | ML | ~-0.001 | ~[-0.004, +0.003] | ~-0.068 | same |

- **Authors' interpretation** (pp. 33-34): all results qualitatively similar (6-10% lower
  earnings; unreported event studies share the pattern; footnote 11: with smaller main
  effects "the qualitative results would tend to be more sensitive to the estimator"), but
  "there are reasonably large quantitative differences across estimators." The include /
  exclude pairs (TWFE and Imp) "are closer to each other than they are to our baseline
  imputation estimator, which arguably indicates that the common robustness check of
  estimating separate regressions that include or exclude the bad control does not
  necessarily indicate robustness to the tension between including and excluding the bad
  control that is a main motivation of our paper." DR and ML "are notably the only ones in
  Figure 8 that are based on the same covariate unconfoundedness assumption as the baseline
  imputation estimate."
- Not stated for the application: control group (never- vs not-yet-treated), anticipation,
  SE method, or bootstrap draws (see Gaps).

**Reference implementation(s):**
- R: `badcontrols` (Caetano, Callaway, Payne, and Sant'Anna 2026),
  https://github.com/hugosantanna/badcontrols - "implementing all proposed estimators"
  (footnote 1, p. 1); used for application Estimators 5-9 (`PT for X`, `ML: pre-treatment`,
  `Imputation`, `DR`, `ML`; footnote 10, p. 31); the staggered estimand it implements is
  Proposition 2 (p. 21). Supporting R packages named by the paper: `ptetools` (Callaway
  2026; Estimators 3-4), `fixest` (Estimators 1-2), `grf` (random-forest nuisances).
  **Note:** `badcontrols` is GPL-3 and diff-diff is MIT - the library must NOT port or
  translate its source; it may be used only as a black-box numerical oracle (run it, compare
  outputs) for parity fixtures.
- Stata: Not discussed in paper.
- Python: Not discussed in paper.

**Requirements checklist:**

Approach 1 (Theorem 1 / Proposition 1 / Proposition 3 - docs + CallawaySantAnna):
- [ ] Document that passing the PRE-treatment bad control (read at base period `g - 1`) plus `Z` as CallawaySantAnna covariates computes the Proposition 3 estimand (two-period: Theorem 1 / Proposition 1), and state Assumptions 4 / 5 (MP-8 / MP-9) explicitly
- [ ] Guard: the post-treatment bad control must NOT be in the covariate set (`τ^use` bias); warn or refuse when a time-varying covariate is read at `t` rather than `g - 1`
- [ ] Pre-test surface: pre-period pseudo-`ATT(g, t)` (existing CS pre-period cells) and `ATT_X(g, t)` (bad control as outcome)

Imputation estimator (Section 6.1):
- [ ] Untreated-only OLS of `ΔY` on `(X_{t*}, X_{t*-1}, Z)` and of `X_{t*}` on `(X_{t*-1}, W, Z)`; `ν̂_0` by plug-in (Eq. 5 under Assumption 8); `ATT̂_ra = m̂_1 - τ̂_ra` (Eqs. 6-7)
- [ ] Influence-function SE from `ψ^ra` (Eq. S8) with the generated-regressor second line; plug-in convention documented as a Note (the paper writes none)
- [ ] Rank / positive-definiteness checks on the two untreated design matrices (Assumption S1(ii)); NaN inference via `safe_inference()` when violated

DR estimator, parametric (Section 6.2):
- [ ] Score `φ_1` (Eq. 10) exactly; sample analog Eq. 11; `π̂` global
- [ ] Nuisances: `m_0` (untreated OLS), `ν_0` (nested plug-in), `p` (logit on ALL units, on `(X_{t*-1}, W, Z)`), `ω_0` (untreated OLS of fitted odds on `(X_{t*}, X_{t*-1}, Z)`)
- [ ] Variance `V̂_dr` from `φ̂_i = φ̂_{1,i} - ATT̂_dr - (ATT̂_dr/π̂)(D_i - π̂)` (Algorithm 1 step 4); `se = √(V̂_dr/n)`; no DoF correction
- [ ] Overlap enforcement on `p̂` (away from 1) and boundedness of `ω̂_0` - implementation-chosen rule, documented deviation
- [ ] Double-robustness tests: correct `(m_0, ν_0)` with wrong `(p, ω_0)` and vice versa (Proposition 6 pairing)

DR estimator, ML / cross-fitting (Algorithm 1):
- [ ] K-fold partition; per-fold first stage (`m̂_0^{-k}` untreated, `p̂^{-k}` all) then nested second stage (`ν̂_0^{-k}`, `ω̂_0^{-k}` untreated) on `I_{-k}`; score on `I_k`
- [ ] Pseudo-outcome overfitting control for the nested stage (OOB predictions or a further split of the training fold - footnote 9)
- [ ] Pluggable regressor / classifier learners; document the `o_p(n^{-1/4})` requirement (Assumption 9)
- [ ] Degenerate-fold guards (no untreated rows in a training complement; no treated rows)

Staggered cells + aggregation (Section 5, SB.3):
- [ ] Per-(g,t) substitution: `Y_t - Y_{g-1}`, `X_t`, `X_{g-1}`, `W = Y_{g-1}` default (Remark 5), not-yet-treated (default) or never-treated comparison
- [ ] Per-cell IFs `ψ_{g,t}` / `φ_{g,t}`; aggregation IF `ξ_θ` with estimated-weight terms `ξ^w_{g,t}` (CS 2021 conventions); simple / group / event-study / calendar aggregations
- [ ] Drop already-treated group; require no group treated in period 1

Pre-tests and guards:
- [ ] Pre-period pseudo-`ATT(g, t)` with base = immediately preceding period (Figure 6/7 convention)
- [ ] `ATT_X(g, t)` estimator (bad control as outcome) with event study; estimand convention documented (the paper writes none)
- [ ] Panel-only: fail closed for repeated cross-sections (Remark 1)
- [ ] Balanced-panel / missing-data policy (paper: complete cases only)

Validation:
- [ ] Recovery tests on the SD DGPs 1-5 (true ATT = 1.00) matching Tables S2-S6 qualitatively (bias, SE/SD, coverage)
- [ ] Black-box parity against R `badcontrols` on a shared fixture (no source porting)
- [ ] Reduction test: with no bad control and no `W`, the DR score reduces to the Sant'Anna-Zhao / Chang panel DR score (see Relation section)

---

## Implementation Notes

### Data Structure Requirements
- **Balanced panel**, one row per unit-period (MP-3; the application requires non-missing
  data in every period). Columns: outcome `Y_it`; unit id; time `t`; first-treatment group
  `G_i` (never-treated coded `∞`, e.g. `0`/`NaN` in the library convention); bad control
  `X_it` (time-varying, scalar in the paper); `Z_i` (pre-treatment/baseline or exogenous
  time-varying covariates - the staggered notation carries a single `Z_i` per unit); `W_i`
  (pre-treatment confounders of `X`, per unit; default `W = Y_{g-1}`).
- **"Pre-treatment value" is cell-specific:** for cell `(g, t)` the bad control is read at
  `t` (inner nuisance only) and at the base period `g - 1` (both nuisances and the
  propensity); `W = Y_{g-1}` is the outcome at the group's base period, NOT `t - 1`
  (Remark 5). Two-period case: `t* - 1` and `t*`.
- **Comparison units per cell:** not-yet-treated (`G > t`, includes never-treated) for
  Propositions 2-3; never-treated only for Theorem 3. Treated units in the cell: `G = g`.
- Treatment must be absorbing (MP-1); no anticipation for `Y` AND `X` (MP-2); no group
  treated in period 1; already-treated units dropped.
- Panel only; repeated cross-sections must fail closed (Remark 1).

### Computational Considerations
- Imputation estimator: two untreated-sample OLS fits per cell plus treated means; the S8
  influence function needs the two untreated second-moment matrices, their inverses, treated
  means of `R̃` and `S`, and OLS residuals - `O(n k^2 + k^3)` per fit (the `k^3` term is the
  factorization / inversion of the `k x k` second-moment matrix).
- DR parametric: four nuisance fits per cell (OLS, OLS-plug-in, logit, OLS of fitted odds).
  Proposition 6 gives parametric CONSISTENCY without cross-fitting, but the inference
  result (Proposition 7, Assumption S2) is stated for the fold-specific cross-fitted
  nuisances of Algorithm 1 only; the paper provides no non-cross-fitted parametric
  influence function or variance. An implementation claiming Proposition 7 inference must
  therefore cross-fit even parametric nuisances, or separately derive and document a
  non-cross-fitted parametric variance as a methodology deviation (the parametric `DR`
  application estimator's fold usage is not stated).
- DR ML: `4 nuisances x K folds x (number of (g, t) cells)` learner fits, with the second
  stage nested inside each fold (it consumes first-stage OOB / split predictions on the
  training complement). `m̂_0`, `ν̂_0`, `ω̂_0` are fit on untreated-only subsamples of the
  training complement; `p̂` on all training units. Random forests (the paper's choice) make
  this the dominant cost; folds and cells are embarrassingly parallel.
- Aggregation: per-unit IF vectors across cells (`ψ_{g,t}` / `φ_{g,t}`) plus weight IFs
  `ξ^w_{g,t}`; memory `O(n x cells)` as in CallawaySantAnna.
- Never fit a nuisance on the fold it is evaluated on; the nested-stage pseudo-outcome
  should not be the in-sample first-stage fit (footnote 9).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| Cross-fitting folds `K` | int, fixed | none recommended in the paper; application uses 5 (footnote 9) | user-fixed; theory needs fixed `K` with equal-order fold sizes (Lemmas S3-S4) |
| Nuisance learners (`m_0`, `ν_0`, `p`, `ω_0`) | regressor / classifier | none prescribed; parametric leading example = OLS / OLS plug-in / logit / OLS-of-odds (p. 26); ML example = random forests via `grf` with default tuning (footnote 9) | must achieve `o_p(n^{-1/4})` in `L^2` (Assumption 9 discussion, p. 27) |
| Nested-stage pseudo-outcome | OOB vs split | OOB first-stage predictions (footnote 9) | further split the training fold for learners without OOB (Farbmacher et al. 2022) |
| Overlap threshold on `p̂` | float | none in the paper (S2(iii) is high-level, away from 1 only) | implementation choice - documented deviation |
| Bound on `ω̂_0` | float / rule | none in the paper (S2(iv)) | implementation choice - documented deviation |
| Comparison group | not-yet-treated / never-treated | not-yet-treated (Propositions 2-3; R package); never-treated rationalized too (p. 21); Theorem 3 requires never-treated | user choice |
| Base period | period | `g - 1` (all staggered results); pre-period pseudo-ATTs use the immediately preceding period (Figure 6/7 notes) | fixed by the identification result |
| Anticipation | int | 0 (MP-2; "limited anticipation" possible "at the cost of notation", p. 18) | not developed in the paper |
| `W` choice | column(s) | `Y_{g-1}` endorsed (Remarks 2, 5; application uses pre-treatment `log(Earnings)`) | substantive; must satisfy Assumption 6 / MP-5 |
| Working models for imputation | linear (Assumption 8) | linear in `(X_{t*}, X_{t*-1}, Z)` and `(X_{t*-1}, W, Z)` | fixed by Assumption 8 |

### Relation to Existing diff-diff Estimators

The following are facts about the diff-diff library supplied by the caller; they are kept
separate from the paper's claims.

- **`CallawaySantAnna`** (`diff_diff/staggered.py`): per-(g,t) cells; base period `g - 1`
  (positional; "varying" / "universal"); never-treated or not-yet-treated controls;
  covariates read at the cell's BASE PERIOD; estimation methods `reg` / `ipw` / `dr`
  (Sant'Anna and Zhao 2020); multiplier bootstrap; aggregations (simple, group, event
  study, calendar). **Consequence:** Approach 1 (Theorem 1 / Proposition 1 / Proposition 3)
  is ALREADY computable by passing the bad control in `covariates` (it is read at `g - 1`,
  exactly the paper's `X_{g-1}`); only documentation and the explicit assumptions
  (4 / 5 / MP-8 / MP-9 and the `τ^use` warning) are missing. The paper's Proposition 3
  statement ("one can directly use Callaway and Sant'Anna (2021) ... as long as the
  pre-treatment value of the bad control is included as a covariate") maps onto this
  directly, and the paper's own `Imp: include BC` / `Imp: exclude BC` comparators
  (Estimators 3-4) are CS-style regression adjustments of the same family.
- **`DMLDiD`** (`diff_diff/dml_did.py`): Chang (2020) Neyman-orthogonal score per (g,t) cell
  with K-fold cross-fitting; private infrastructure `diff_diff/_learners.py` (duck-typed
  regressor / classifier learners; native linear / ridge / logit / sieve),
  `diff_diff/_crossfit.py` (replayable unit-level fold assignment, out-of-fold predictions),
  `diff_diff/_dr_scores.py` (score functions); plug-in influence-function SE; multiplier
  bootstrap and CS aggregations via mixins; a `panel=False` repeated-cross-section lane.
  **Consequence:** with no bad control and no `W`, Equation 10 reduces algebraically to the
  Sant'Anna-Zhao / Chang panel DR score DMLDiD already computes (with `X_{t*}` absent,
  `ν_0 = m_0` and `ω_0 = p/(1 - p)`, so lines 2-4 of Equation 10 collapse to the familiar
  `(D/π) ΔY - (D/π) m_0 - ((1-D)/π)(ΔY - m_0) p/(1-p)` form), so the paper's DR estimator is
  a strict generalization that can live on DMLDiD: new arguments for the bad control and
  `W`; a new nested second stage inside each fold (`ν̂_0^{-k}`, `ω̂_0^{-k}` from first-stage
  OOB / split predictions); a new score in `_dr_scores.py`; the `(D - π)` variance
  correction is structurally the same as DMLDiD's existing `Ĝ_1p (D - p̂)` term. The RCS
  lane must fail closed per Remark 1. Reusable: fold machinery, learner protocol, IF-SE and
  bootstrap / aggregation mixins, degenerate-fold guards, overlap clipping convention.
  New: the four-nuisance nested pipeline, the pseudo-outcome overfitting control, the
  `ω_0` odds-regression learner, and - if offered - a non-cross-fitted parametric-DR path
  (Proposition 6 consistency only; its variance would be a documented deviation, see
  Computational Considerations).
- **`ImputationDiD`** (`diff_diff/imputation.py`): Borusyak-Jaravel-Spiess unit + time
  fixed-effects imputation. **NOT the same object** as this paper's imputation estimator,
  which is a per-cell differenced regression adjustment with a nested first stage (an
  untreated OLS of `X_t` on `(X_{g-1}, W, Z)` plugged into an untreated OLS of the long
  difference on `(X_t, X_{g-1}, Z)`). Structurally the paper's imputation estimator is
  closer to CallawaySantAnna's `estimation_method="reg"` with a nested first stage and the
  generated-regressor IF term (S8, second line). Do not route it through ImputationDiD.
- **Reuse vs new, summary:** reuse CS cell construction, base-period covariate reading,
  control-group selection, aggregation weights and their IFs (the paper's SB.3 defers to
  CS 2021 for exactly these), `safe_inference()`, the multiplier bootstrap (a library
  extension - the paper has none), `_crossfit` / `_learners` / `_dr_scores`. New: bad
  control + `W` column handling per cell, the nested nuisances, the S8 and Equation 10 /
  Proposition 7 influence functions, the `ATT_X(g, t)` pre-test surface, and the
  Approach 1 documentation / guards.

---

## Gaps and Uncertainties

1. **No explicit plug-in variance estimator for the imputation estimator.** Proposition 4
   (p. 24) and SA pp. 2-4 give only `Ω = Var(ψ^ra)` with `ψ^ra` in (S8); no `Ω̂` is written
   (contrast Algorithm 1 step 4 for the DR estimator). The sample-analog plug-in described
   above is an implementation choice to record as a REGISTRY Note.
2. **No bootstrap discussion anywhere** (main text pp. 23-28, SA SB.1-SB.3); no multiplier
   bootstrap even though the aggregation argument defers to Callaway and Sant'Anna (2021).
   A library bootstrap would be an extension.
3. **No clustering** (i.i.d. units per Assumptions 1 / MP-3); no small-sample or DoF
   corrections; normal critical values throughout.
4. **No trimming / overlap rule.** Assumptions 3, 7, MP-6 and S2(iii)-(iv) are high-level
   (propensity away from 1 only; `ω̂_0` bounded); no threshold or clipping recipe is given.
5. **`K` not recommended.** Algorithm 1 takes `K` as input; the application uses 5
   (footnote 9); Lemmas S3-S4 need fixed `K` with equal-order fold sizes. No repeated
   cross-fitting is discussed.
6. **`ATT_X(g, t)` is stated to be identified "under our assumptions" but no estimand or
   estimator is written** (Remark 6, pp. 22-23). The application obtains it by
   "implementing the first step of the Section 6 imputation estimator" (p. 31), which in
   the two-period case is `E[ΔX_{t*} | D = 1] - E[Ê[X_{t*} | X_{t*-1}, W, Z, D = 0] - X_{t*-1} | D = 1]`
   under Assumption 8's second equation; SC's (S17) gives a different (parallel-trends-for-X)
   expression under Assumption S3. Which conditioning set and which assumption the
   `ATT_X` figure rests on is not spelled out; do not guess - document the chosen estimand.
7. **Figure S1 note vs panels:** the note says "for the four main DGPs" but five panels
   (a)-(e) are shown (SA p. 18). Record as printed.
8. **Application details not stated:** control group (never- vs not-yet-treated),
   anticipation, SE method (analytical IF vs bootstrap), number of bootstrap draws, and the
   fold usage of the parametric `DR` estimator. Only point estimates with SEs / error bars
   appear (pp. 30-34). Exact Figure 8 differences are not printed for Estimators 1-6, 8, 9.
9. **Binary bad control** is not discussed in the paper text; all notation is generic
   real-valued, the application and simulations are continuous, and SC's linear models are
   written for a real-valued `X`. With binary `X`, Assumption 8's second equation is a
   linear probability model - unaddressed.
10. **Proposition 1 proof location:** p. 14 says the proof is "provided in Appendix A", but
    Appendix A (pp. 38-42) contains only the proofs of Theorems 1-3, Lemmas 1-4, and
    Propositions 2-3. Possibly in the SA (not in SA pp. 1-23 as read either). Its
    Section 4.1.1 argument (Assumption 5 collapses the inner conditioning set) is
    elementary; do not cite an Appendix location.
11. **Whether the `(D - π)` correction persists in staggered aggregation:** SB.3 says the
    per-cell IFs are "the same ... up to adjusting the periods" and denotes them
    `φ_{g,t}`, but does not write the `(g, t)`-specific analog of `π` (the `G = g` share
    within the cell's estimation sample?) nor restate the correction; `ξ_θ` then combines
    `φ_{g,t}` with estimated-weight IFs. Implementation must define the per-cell `π̂_{g,t}`
    and carry the correction; record the convention (SA pp. 11-12).
12. **Lemma 2 and Lemma 4 proofs are omitted** ("very similar" to Lemma 3; pp. 40-41).
13. **Assumption 1 expansion** to include `W` (footnote 8, p. 15) is not written out.
14. **Suspected typos / labeling inconsistencies (as printed):**
    - Figure 4's title says "Causal Graphs for Approach 3" while the text refers to
      Section 4.1 / new Approach 1 (p. 13).
    - The double-robustness discussion reads "Proposition 5 shows that ATT̂_dr is doubly
      robust" (p. 26) where the result is Proposition 6.
    - Remark 6, p. 23: "In post-treatment periods, it a test for X_t ..." (missing "is").
    - (S6) writes the first-stage factor as `E[(D/π) β_1 S']` while (S8) writes
      `E[(D/π) β_1' S']` (SA pp. 3-4); same object since `β_1` is scalar.
    - SA p. 3 writes `τ̂_ra = (1/n) Σ_i (D_i/π̂) ν̂_{0,i}` while Equation 7 writes
      `(1/n_1) Σ_i D_i ν̂_{0,i}`; identical since `π̂ = n_1/n`.
    - SB.2 (SA p. 7) opens with `ATT̂_dr = (1/n) Σ_i (π/π̂) φ_1(O_i, η̂^{-k})`, i.e. writes
      the estimator with the population `π` inside `φ_1` and a `π/π̂` rescaling; Equation 11
      / Algorithm 1 use `π̂` directly. Equivalent.
15. **Proposition 5 proof shows mean-zero of the correction terms, not a Gateaux-derivative
    orthogonality computation** (SB.1, SA pp. 4-5); orthogonality is exhibited via the
    second-order conditional bias in Lemma S3. Fine for implementation; note the
    proof style if citing "Neyman orthogonality".
16. **No `PT for X` (SC) variance formula** is written (SA p. 14), though the simulations
    report SE/SD for it; the `badcontrols` implementation's convention is unknown.
17. **Section 8 (Conclusion, p. 35)** restates contributions only: including or excluding
    bad controls can both bias the ATT; explicit conditions under which conditioning on the
    pre-treatment bad control suffices; a generalization based on unconfoundedness for the
    bad control. No limitations / future-work list is given beyond the body (Remark 5's
    open feedback problem; Remark 8's TWFE non-operationalizability).
18. **Extraction-coverage note:** the three section extractions covered main pp. 1-34 and
    38-42 plus SA pp. 1-23, with Remark 6's tail (p. 23) and Section 8 (p. 35) supplied
    verbatim as gap fills; References (pp. 36-37 approximately) and SA p. 17 (references)
    were not extracted. No contradictions between extractors were found beyond the
    as-printed items above.
19. **Intercept convention for the linear working models is never stated.** Assumption 8
    (p. 24) and the SA regressor vectors `R_i`, `S_i` (SA p. 1) show no constant, `Z` is
    described only as "a vector of other covariates" (p. 5), and no regression in
    Sections 6-7 or SA mentions an intercept. The Monte Carlo DGPs (SA pp. 14-15) contain
    constants (`θ_t` in `Y`, so `ΔY` has mean shift `0.3`; `0.15` in every `X_2(0)`
    equation) and Table S1 marks the imputation estimator consistent in DGPs 1 and 4,
    which is only true with an intercept (or a constant inside `Z`) in both OLS fits.
    Implementation-required: include an intercept in every nuisance regression and in
    the parametric propensity / odds models; document it as an implementation
    convention, not a paper-stated one.
