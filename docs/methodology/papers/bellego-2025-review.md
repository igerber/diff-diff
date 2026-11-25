# Paper Review: The Chained Difference-in-Differences

**Authors:** Christophe Bellégo, David Benatia, Vincent Dortet-Bernadet
**Citation:** Bellégo, C., Benatia, D., & Dortet-Bernadet, V. (2025). The chained difference-in-differences. *Journal of Econometrics*, 248, 105783. https://doi.org/10.1016/j.jeconom.2024.105783
**PDF reviewed:** https://arxiv.org/pdf/2301.01085v4 (arXiv v4, [econ.EM], 31 Jan 2025; title page dated February 4, 2025; 84 pages; SHA-256 `5260eb75bfb1a24e095e14dea0f02752979247ad62e3d7f426a46aaf79f77851`)
**Review date:** 2026-11-25

> **Version reviewed.** The paper is published (Journal of Econometrics 248, March 2025,
> article 105783, per the Crossref record for the DOI). This review was written from the
> arXiv v4 preprint, whose Online Appendix header (p. 50) reads "To appear in the Journal of
> Econometrics". The journal's typeset version was not available to the reviewer. **All
> equation, assumption, theorem, proposition, corollary, footnote, table, figure, and page
> references below are pinned to arXiv v4.** Re-check numbering against the published
> version before citing it in user-facing docs.
>
> **Page convention.** "p. N" is always the PDF page index. The main text (Sections 1-5)
> occupies pp. 1-43 and its printed page numbers equal the PDF page. The reference list is
> pp. 44-49. The Online Appendices (A: proofs, B: application tables) occupy pp. 50-84 and
> restart their printed numbering at 1, so **printed appendix page = PDF page - 49**.
>
> **Labels used in this review.** Text in quotation marks and equations in code blocks are
> transcribed from the paper. "[as printed]" marks a literal transcription of something
> that looks like a typo or an internal inconsistency. Anything marked **[Reviewer check -
> not in paper]** is this review's own algebra, arithmetic, or reading of a table; it is
> not a statement made by the paper. Where the review says the paper has no statement on a
> topic, the claim rests on a full-text search of a `pdftotext -layout` dump of all 84 pages
> for the terms listed at that point (a term split across a line break could be missed).
>
> **Notation.** In code blocks `T` stands for the paper's calligraphic T (the number of
> periods in Section 2). Section 3 uses an italic T with a different meaning (see the
> simulation section). `E_M[.]` is the expectation under the paper's mixture distribution
> and `Ê_M[.]` its sample mean. `p'_g` stands for the paper's dotted p (the derivative of
> the propensity link).

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## ChainedDiD` section into the staggered-adoption part of the registry once a feature ships (rename to the shipped class name).*

## ChainedDiD (prospective; name provisional)

**Primary source:** Bellégo, C., Benatia, D., & Dortet-Bernadet, V. (2025). The chained difference-in-differences. *Journal of Econometrics*, 248, 105783. DOI 10.1016/j.jeconom.2024.105783. Numbering below follows arXiv:2301.01085v4.

The paper studies "long-term (binary) treatment effect parameters when balanced panel data
is not available, or consists of only a subset of the available data" (p. 1). The estimator
"consists in aggregating a collection of short-term treatment effects estimated on multiple
incomplete panels" (p. 1). The core idea (p. 2): a long difference `Y_{t2} - Y_{t0}` equals
the sum of short differences `(Y_{t2} - Y_{t1}) + (Y_{t1} - Y_{t0})`, so "one subsample of
individuals might be used to estimate the DiD from t_0 to t_1 while another would be used to
estimate the DiD from t_1 to t_2. The sum of both effects identifies the long-term DiD from
t_0 to t_2".

The paper develops three layers, and an implementation must say which one it follows:

1. **Simple setting** (Section 2.1, pp. 8-17): one treatment date, no covariates, each unit
   observed in two consecutive periods. Used for intuition and for the efficiency
   comparison (Proposition 1).
2. **Rotating panel** (Section 2.2.1, pp. 18-23): staggered timing, covariates through an
   inverse-probability-weighted control group, each unit observed in two consecutive
   periods. Theorems 1 and 2.
3. **General missing data patterns** (Section 2.2.2, pp. 24-27): units observed at any two
   or more periods; all pairwise differences are combined by GMM. Theorem 3.

**Key implementation requirements:**

*Setting and notation (pp. 8-9, 17-18, 20):*
- `T` periods, `t = 1, ..., T`. `G_g` is "a binary variable that is equal to one if an
  individual is first treated in period g"; "The control group binary variable C denotes
  individuals who are never treated"; "for each individual" `sum_{g=2}^{T} G_g + C = 1`
  (p. 17). `D_t` equals one "if an individual is treated in t" (p. 17).
- `S_t` is "a binary variable equal to one for individuals observed at t and zero
  otherwise", and `S_{t,t+1}` denotes `S_t S_{t+1}` (p. 9).
- Generalized propensity score (pp. 17-18): `p_g(X) = P(G_g = 1 | X, G_g + C = 1)`.
- "Remark that X and D_i are assumed to be observed for all individuals." (p. 20).
  Footnote 6 (p. 10): "we do not consider cases where treatment status (or covariates) may
  also be missing".
- Covariates carry no time index in the assumptions (`X_i`). The application adds that
  propensity-score variables "must be observed in g - 1, t, and t + 1 to compute the
  elementary building block constituting each chain link" (p. 36).

*Target parameter (Equation 12, p. 18):*

    ATT(g, t) = E[ Y_t(1) - Y_t(0) | G_g = 1 ]

"This parameter is referred to as the group-time average treatment effect in Callaway and
Sant'Anna (2021)" (p. 18).

*Assumptions, rotating panel (pp. 18-19), verbatim:*
- **Assumption 1 (Sampling):** "For all t = 1, ..., T-1, {Y_it, Y_it+1, X_i, D_i1, D_i2,
  ..., D_iT}_{i=1}^{n_t} is independent and identically distributed (iid) conditional on
  S_it,t+1 = 1."
- **Assumption 2 (Missing Trends At Random), Equation 13:** "For all t = 1, ..., T - 1,"
  `S_{it,t+1} ⊥ Y_{it+1} - Y_it, X_i | D_i`.
- **Assumption 3 (Conditional Parallel Trends), Equation 14:** "For all t = 2, ..., T,
  g = 2, ..., T, such that g ≤ t,"
  `E[Y_{t+1}(0) - Y_t(0) | X, G_g = 1] = E[Y_{t+1}(0) - Y_t(0) | X, C = 1] a.s.`
- **Assumption 4 (Irreversibility of Treatment), Equation 15:** "For all t = 2, ..., T,"
  `D_{t-1} = 1 implies that D_t = 1`.
- **Assumption 5 (Overlap):** "For all g = 2, ..., T, P(G_g = 1) > 0 and for some ε > 0,
  p_g(X) < 1 - ε a.s."

*Assumptions, general missing data patterns (pp. 25-26), verbatim:* "we must modify
Assumptions 1 to 2 as follows."
- **Assumption 8 (Sampling):** "For all t = 1, ..., T, {Y_it-k, Y_it, X_i, D_i1, D_i2, ...,
  D_iT}_{i=1}^{n_{t,k}} is independent and identically distributed (iid) conditional on
  S_it-k,t = 1, for k = 1, ..., t - 1."
- **Assumption 9 (Missing Trends At Random 3), Equation 25:** "For all t = 1, ..., T and
  k = 1, ..., t - 1," `S_{it-k,t} ⊥ Y_it - Y_it-k, X_i | D_i`.

*Assumption checks / warnings:*
- **What the sampling process may depend on.** The cross-section DiD needs the sampling
  process to be "(conditionally) independent of the levels of the outcome variable"; "our
  approach requires that the sampling process be (conditionally) independent of the trend
  of the outcomes" (pp. 5-6). Assumption 2 "constitutes the principal departure from
  Callaway and Sant'Anna (2021)" (p. 19). "In practice, we only require conditional
  mean-independence between S_t,t+1 and Y_t+1 - Y_t." (p. 19). The two sampling
  assumptions "are generally non-nested" (p. 11).
- **Draws are not representative.** Assumption 1 "does not imply that observations are
  representative of the population of interest because we do not assume that the iid draws
  are taken from the population distribution" (p. 19). The data are "random draws from the
  mixture distribution F_M" with weights `λ_{t,t+1} = P(S_{t,t+1} = 1)`, and "Expectations
  under the mixture distribution does not correspond to population expectations" (p. 20).
- **Parallel trends is stated only for periods at or after treatment.** Assumption 3 is
  restricted to "g ≤ t"; Theorems 1-3 are stated "for 2 ≤ g ≤ t ≤ T" (pp. 21, 27). As
  printed, Equation 14 is about the increment from `t` to `t+1`, while Theorem 1 uses
  increments from `τ-1` to `τ` for `τ = g, ..., t` [as printed; see "Printed
  inconsistencies", item 1].
- **Anticipation.** A full-text search for "anticipat" returns no hits. The chain starts at
  `τ = g` (first link is the pair `(g-1, g)`), and the proof of Theorem 1 writes
  `ATT_X(g, t)` as `sum_{τ=g}^{t} ΔATT_X(g, τ)` (Equation 32, p. 55).
  **[Reviewer check - not in paper]** That telescoping requires `ATT_X(g, g-1) = 0`, i.e.
  no effect in the period before treatment; the paper does not state this as a numbered
  assumption.
- **Control group.** The main theoretical development uses never-treated units `C`. Online
  Appendix A.2.5 develops not-yet-treated controls (see below), and the main-text
  application uses them (p. 34).
- **Overlap is on treatment only.** Assumption 5 bounds `p_g(X)`. The efficiency proof for
  the simple setting assumes a common pair-sampling probability,
  "P(S_iτ S_iτ+1) = q ∈ (0, 1) for all τ, i" (p. 52). For the general estimator, no explicit
  lower bound on the pair-sampling probabilities `λ` and no minimum count per pair of
  periods was found: searches for "bounded
  away", "minimum" (one hit, p. 54, "minimum variance"), "sparse", "few " (hits pp. 5 and
  41, both about too few units for the long DiD) and "positiv" find no such condition. The nearest
  statements are "there are at least two different subsamples observed in each period
  1 < t < T" (p. 9) and "The key requirement is that individuals must be observed at least
  twice to be used." (p. 24).
- **Testable implication.** "under general missing data patterns, the sampling assumption
  can lead to testable hypotheses about trends in various data subsets. Indeed, Assumption 9
  implies a certain level of homogeneity across different subpanels." (p. 26).

*Estimator equation, rotating panel (Theorem 1 and the two-step estimator, pp. 20-21):*

Population weights (p. 20):

    w^G_{τ,τ-1}(g)    = G_g S_{τ,τ-1} / E_M[ G_g S_{τ,τ-1} ]

    w^C_{τ,τ-1}(g, X) = [ p_g(X) C S_{τ,τ-1} / (1 - p_g(X)) ] / E_M[ p_g(X) C S_{τ,τ-1} / (1 - p_g(X)) ]

**Theorem 1** (p. 21): "Under Assumptions 1 - 5, and for 2 ≤ g ≤ t ≤ T, the long-term
average treatment effect in period t is nonparametrically identified, and given by"

    ATT_CD(g, t) = sum_{τ=g}^{t} ΔATT(g, τ)

    ΔATT(g, τ) = E_M[ w^G_{τ,τ-1}(g) (Y_τ - Y_{τ-1}) ] - E_M[ w^C_{τ,τ-1}(g, X) (Y_τ - Y_{τ-1}) ]

"In addition, the cross-section DiD does not identify ATT(g,t) under Assumption 2."

Two-step estimator (p. 21):

    ATT^_CD(g, t) = (1/n) sum_{i=1}^{n} sum_{τ=g}^{t} { ŵ^G_{iτ,τ-1}(g) (Y_iτ - Y_iτ-1) - ŵ^C_{iτ,τ-1}(g, X) (Y_iτ - Y_iτ-1) }

    ŵ^G_{iτ,τ-1}(g) = G_ig S_iτ-1 S_iτ / [ (1/n) sum_i G_ig S_iτ-1 S_iτ ]

    ŵ^C_{iτ,τ-1}(g) = [ p̂_g(X_i) C_i S_iτ-1 S_iτ / (1 - p̂_g(X_i)) ] / [ (1/n) sum_i p̂_g(X_i) C_i S_iτ-1 S_iτ / (1 - p̂_g(X_i)) ]

"with p̂_g(·) being an estimated parametric propensity score function, such as logit or
probit, obtained in a first step." For each `(g, τ)`, each arm's raw weight is divided by
its sample mean over all `n` units; units not observed in both `τ-1` and `τ` enter that mean
as zeros, and the outer estimator also averages over all `n` units.
**[Reviewer check - not in paper]** The two `1/n` factors cancel, so each arm is the
weighted sum of outcome changes over the jointly observed units divided by the sum of the
same weights.

*Estimator equation, general missing data patterns (Section 2.2.2, pp. 24-27; Proof 5,
pp. 68-71):*

"We now consider not only first-differences Δ_1 ATT(t) but also k-period differences
Δ_k ATT(t) from t - k to t for k = 1, ..., t - 1." (p. 24; the dependence on `g` is dropped
"For simplicity of exposition"). "Consider the estimates of all possible Δ_k ATT(t), for all
t ≥ 2 and t - 1 ≥ k ≥ 1, stacked altogether into a vector ΔATT of length L_Δ. ... By
definition Δ_k ATT(t) = ATT(t) - ATT(t - k), hence we can write" (pp. 24-25)

    ΔATT = W ATT                                                    (Equation 23)

"where ATT is the vector of ATT(t) for t ≥ 2 of length L ≤ L_Δ and W is a matrix where each
element takes value in {-1, 0, 1}." The three-period example of Table 2 (p. 24) is written
as (Equation 24, p. 25):

    [ Δ_1 ATT(2)^{BP,IP1} ]   [  1  0 ]
    [ Δ_1 ATT(3)^{BP,IP2} ] = [ -1  1 ] [ ATT(2) ]
    [ Δ_2 ATT(3)^{BP,IP3} ]   [  0  1 ] [ ATT(3) ]

"where we pool all subsamples that identify each Δ_k ATT(t)." Footnote 21 (p. 25): "we
could also estimate these parameters separately for each subsample before stacking them
into ΔATT. That would require to estimate propensity scores conditional on subsample
membership."

Moment conditions (Equations 42-43, p. 68): "Let us define the vector of parameters
Θ = ATT that includes all θ_τ = ATT(τ), for all τ > 1 since θ_1 = ATT(1) = 0 by
construction." Each element of the `L_Δ`-vector of moments is

    w^G_{iτ,τ-k}(g) (Y_iτ - Y_iτ-k) - w^C_{iτ,τ-k}(g, X) (Y_iτ - Y_iτ-k) - θ_τ + θ_{τ-k}

"possibly for all τ ≥ 2, and 1 ≤ k < τ", with the weights of p. 20 where `S_{τ,τ-1}` is
replaced by `S_{τ,τ-k}` (p. 69).

"We propose to solve this problem using a GMM approach. Denoting Ω the covariance matrix of
ΔATT, the optimal GMM estimator of ATT corresponds to" (p. 25; Equation 47, p. 69)

    ATT^ = (W' Ω⁻¹ W)⁻¹ W' Ω⁻¹ ΔATT^

*Which weighting matrix the paper proposes:*
- The proposed estimator is the optimal-weighting GMM: step 4 of the procedure is "Estimate
  the optimal GMM estimator" (p. 27), and "our estimator brings efficiency gains by using an
  optimal weighting matrix" (p. 27).
- Identity weighting appears in two places. Footnote 23 (p. 27): "this approach embeds the
  estimator proposed above in the rotating panel data setting when using only the ΔATT_1(t)
  and replacing Ω by the identity matrix." Section 3 (p. 31): "We show estimates from the
  chained DiD GMM estimator using two weighting matrices: (1) Ch DiD uses the identity
  matrix, and (2) CD-GMM uses the optimal weighting matrix presented earlier."
- The paper's own caveat (p. 31): "the (asymptotically) optimal weighting matrix does not
  always deliver more precise estimates than the identity matrix in small samples, at least
  for this simulation design." And (pp. 31-32): "the optimal weights allow mitigating the
  precision loss for longer-term effects (e.g. β_6) at the cost of losing some precision for
  smaller-term effects (e.g. β_1)."
- A full-text search for "identity" returns pp. 27, 31 and 32 only. No rule for choosing
  between the two matrices was found beyond the sentences above.

*Which differences enter the stack, and pre-treatment periods:*
- The stack is described with `g` suppressed: "all possible Δ_k ATT(t), for all t ≥ 2 and
  t - 1 ≥ k ≥ 1" (p. 24), with period 1 as the zero reference ("θ_1 = ATT(1) = 0 by
  construction", p. 68). Step 1 of the procedure restores `g`: "for all k, g, t" (p. 26).
- Theorem 3 is stated "for all 2 ≤ g ≤ t ≤ T" (p. 27) and invokes Assumption 3, which is
  restricted to "g ≤ t".
- The displayed stacking and moment ranges therefore include, for a cohort `g`, links that
  begin before `g - 1`: step 1 says "for all k, g, t", and Equation 43 supplies the generic
  mapping through `-θ_τ + θ_{τ-k}` with the ranges "τ ≥ 2, and 1 ≤ k < τ" (pp. 68-69). The
  paper does not reconcile this broad construction with the cohort-specific post-treatment
  ranges of Assumption 3 and Theorem 3. The Table 2 example has every difference starting
  at or after the reference period.
  **[Reviewer check - not in paper]** With the zero reference at period 1 and `g ≥ 3`, the
  parameter `θ_t` for `t ≥ g` is a difference-in-differences from period 1 to `t`. It equals
  `ATT(g, t)` only if trends are also parallel between period 1 and `g - 1`, which
  Assumption 3 as printed does not state. Moving the reference to `g - 1` and leaving the
  pre-treatment parameters unrestricted is one way to reconcile the two; it is an
  implementation choice, not the paper's stated construction.
- Pre-treatment effects are not defined by Theorems 1-3 (all restricted to `g ≤ t`), and the
  bootstrap matrix is sized for post-treatment cells only: `K = T(T-1)/2`, "the number of
  (g, t) element for any 2 ≤ g ≤ t ≤ T" (p. 63). The paper nevertheless says of Callaway
  and Sant'Anna (2021) that "Their multiplier bootstrap and pre-trend tests remain
  asymptotically valid in our setting" (p. 3), and the application tables report
  `β_{-3}`, `β_{-2}`, `β_{-1}`, a "ref." row equal to 0, and a "Pre-trend" row for the
  chained DiD (Tables B.1-B.6, pp. 79-84). The regression of Equation 26 (p. 27) also
  carries cohort-by-period coefficients `β_gτ` for every `g = 2, ..., T` and
  `τ = 2, ..., T`, which includes `τ < g`, and that regression is stated to be equivalent to
  the chained DiD without covariates when each individual has two consecutive observations
  (p. 28). Searches for "pre-trend" (pp. 3, 38, 42), "pre-treatment" (pp. 13, 17), "placebo"
  (p. 42), "reference period" and "base period" (no hits) find no explanation of how the
  broad moment ranges are reconciled with cohort-specific pre-treatment restrictions, and
  no formula for the application's pre-treatment chained estimates.

*Unidentified elements and regularisation (footnote 22, p. 25; p. 69):*
- "if an element of ATT is not identified, the matrix W'Ω⁻¹W (or W'W) will not be invertible
  but a (Moore-Penrose) pseudo-inverse can still be used to identify the other elements. In
  addition, pseudo-inverses, e.g. Tikhonov's, will deliver a stable inverse of Ω if ΔATT is
  high-dimensional (Carrasco, Florens and Renault, 2007)." No regularisation parameter or
  selection rule is given ("Tikhonov" has one hit, p. 25).
- Rank condition (p. 69): "The necessary and sufficient rank condition for GMM
  identification in this linear setting is that the rank of Ω⁻¹W is equal to the number of
  columns (Newey and McFadden, 1994). This condition is satisfied if both the covariance
  matrix Ω and the weight matrix W are non-singular. Remark further that if W is not full
  row rank then some ATT(g, t) are not identified by the collection of Δ_k ATT(g, t)'s
  identified in the dataset at hand."
  **[Reviewer check - not in paper]** `W` is `L_Δ × L`. Identification requires full column
  rank `L`. "Non-singular" applies only to a square `W`, and full row rank is unnecessary
  here and impossible when `L_Δ > L`.

*Propensity score (pp. 21, 57, 63, 70):*
- Parametric: `p_g(X_i) = Λ(X_i' π_g^0)` "with Λ(·) being a known function (logit or
  probit), can be parametrically estimated by maximum likelihood" (p. 57), under "Assumption
  5 in Callaway and Sant'Anna (2021) or 4.4 in Abadie (2005)" (pp. 21, 27).
- Defined on the subpopulation `G_g + C = 1`, one score per cohort `g`.
- Its influence function `ξ^π_g` "does not depend on the sampling process since X is
  observed for all individuals" (p. 57), and the printed formula for `ξ^π_g(i)` carries the
  factor `(G_g + C)` and no sampling indicator (pp. 63, 70).
  **[Reviewer check - not in paper]** Read together with "X and D_i are assumed to be
  observed for all individuals" (p. 20), this means the paper's first step fits one
  propensity model per cohort on all cohort-`g` and never-treated individuals, not on the
  subsample observed in a given pair of periods. Footnote 21 names estimation "separately
  for each subsample" (each missing-data pattern, as in Table 2) as an alternative that
  "would require to estimate propensity scores conditional on subsample membership"; that
  is not the same as one fit per pair of periods on the pooled pairwise sample.

*Standard errors (Theorems 2-3; Online Appendix A.2.1, A.2.2, A.2.6):*

- **Asymptotic normality.** Theorem 2 (p. 21): under Assumptions 1-5 and the propensity
  score assumption, `sqrt(n) (ATT^_{g≤t} - ATT_{g≤t}) →d N(0, Σ)`. Theorem 3 (p. 27): the
  same for the GMM estimator under Assumptions 3-5, 8 and 9. In both, "the covariance Σ is
  detailed in the proof."
- **Influence function of one chain link** (Equation 37, p. 58, and the recipe on p. 63):

      ψ^G_gt(i) = w^G_{it,t-1}(g)    [ (Y_it - Y_it-1) - E_M[ w^G_{t,t-1}(g) (Y_t - Y_t-1) ] ]

      ψ^C_gt(i) = w^C_{it,t-1}(g, X) [ (Y_it - Y_it-1) - E_M[ w^C_{t,t-1}(g, X) (Y_t - Y_t-1) ] ] + M_gt' ξ^π_g(i)

      M_gt = E[ X ( C S_t S_{t-1} / (1 - p_g(X)) )² p'_g(X) [ (Y_it - Y_it-1) - E[ w^C_{t,t-1}(g, X) (Y_t - Y_it-1) ] ] ] / E[ p_g(X) C / (1 - p_g(X)) ]

      ξ^π_g(i) = E_M[ (G_g + C) p'_g(X)² / ( p_g(X_i) (1 - p_g(X_i)) ) X X' ]⁻¹ X_i (G_g + C) (G_g - p_g(X_i)) p'_g(X_i) / ( p_g(X_i) (1 - p_g(X_i)) )

  `M_gt` and `ξ^π_g` are transcribed from p. 63, which prints plain `E` in `M_gt`; p. 58
  prints `E_M` there. `M_gt` is "a k dimensional vector (k being the number of covariates
  in X)". The paper
  prints `ψ_gt = ψ^G_gt + ψ^C_gt` (p. 63; Equation 37 prints `ψ^G_gt + ψ^G_gt`). See
  "Printed inconsistencies", items 2 and 3, before implementing.
- **Chained influence function and variance, rotating panel** (p. 62): "the influence
  function of the chained DiD estimator corresponds to the sum of influence functions of the
  short-term DiD estimator":

      φ_gt(i) = sum_{τ=g}^{t} ψ_gτ(i),        Σ = E_M[ Φ Φ' ]

- **Covariance of the stacked differences, general patterns** (p. 70): "For every (g, t, k),
  compute the n-dimensional vector ψ_gtk" (Equation 48, the same formulas with `t-1`
  replaced by `t-k`), "Concatenate all ψ_gtk's into a L_Δ × n matrix Ψ, and compute
  Ω̂ = Ê[Ψ(i)Ψ(i)']." The main text writes this as `Ω̂ = n⁻¹ΨΨ'` (step 3, p. 26).
- **Variance of the GMM estimator** (Equations 49-50, pp. 70-71):

      Σ = (W' Ω⁻¹ W)⁻¹,        Φ = (W' Ω⁻¹ W)⁻¹ W' Ω⁻¹ Ψ

  This is the optimal-weighting form. **For the general overidentified stack, no variance
  formula for a non-optimal weighting matrix (including the identity matrix) was found**:
  "sandwich" has no hits, and the only variance expressions in A.2.6 are Equations 49 and
  50. The paper does give the rotating-panel covariance `Σ = E_M[ΦΦ']` (p. 62), and
  footnote 23 embeds that estimator as the first-difference, identity-weighted case.
  **[Reviewer check - not in paper]** For a fixed symmetric weighting matrix `A`, the same
  linear algebra gives `ATT^_A = (W'AW)⁻¹ W'A ΔATT^`, influence function
  `Φ_A = (W'AW)⁻¹ W'A Ψ`, and variance `(W'AW)⁻¹ W'A Ω A W (W'AW)⁻¹`, which reduces to
  Equation 49 at `A = Ω⁻¹`. An identity-weighted implementation needs this form and must
  document it as a derivation, not as a formula from the paper.
- **GMM regularity conditions** (pp. 69-70): "(i) Ω and the weight matrix W are
  non-singular; (ii) the true value Θ_0 lies within a compact set; and (iii)
  E_M[sup_Θ ||h_i(W_i|Θ)||] < ∞" for consistency (Theorem 2.6 in Newey and McFadden, 1994);
  "(iv) Θ_0 lies in the interior of the compact set; (v) E[||h_i(W_i|Θ_0)||², ] < ∞; (vi)
  W'Ω⁻¹W non-singular" for asymptotic normality (their Theorem 3.4).
- **Bootstrap** (A.2.2, pp. 62-63), "Bootstrapped confidence bands for ATT^(g,t)":
  1. "Draw a vector of V^b = (V_1, ..., V_i, ..., V_n)', where V_i's are iid zero mean random
     variables with unit variance, such as Bernoulli random variables with
     Pr(V = 1 - κ) = κ/√5 with κ = (√5 + 1)/2 as suggested by Mammen (1993)."
  2. "Compute the bootstrap draw ATT^*b_{g≤t} = ATT^_{g≤t} + Φ̂_{g≤τ} V^b where Φ̂_{g≤τ} is a
     consistent estimator of Φ_{g≤τ} (see below)."
  3. "Compute R̂*b(g, t) = √n (ATT^*b(g, t) - ATT^(g, t)) for each element of the vector
     ATT^*b_{g≤t}."
  4. "Repeat steps 1-3 B times. Note: do not re-estimate propensity scores and parameters
     for each draw."
  5. "Compute the bootstrapped covariance for each (g, t) as
     Σ̂^{1/2}(g, t) = (q_0.75(g, t) - q_0.25(g, t)) / (z_0.75 - z_0.25), where q_p(g, t) is
     the p^th sample quantile of R̂* (across B draws) and z(g, t) is the p^th quantile the
     standard normal distribution."
  6. "For each b, compute t-stat^b_{g≤t} = max_{(g,t)} |R̂*b(g, t)| Σ̂^{-1/2}(g, t)."
  7. "Construct ĉ_{1-α} as the empirical (1 - α) quantile of the B boostrap draws of
     t-stat^b_{g≤t}."
  8. "Construct the bootstrapped simultaneous confidence band for ATT(g, t) as
     Ĉ(g, t) = [ATT^(g, t) ± ĉ_{1-α} Σ̂^{-1/2}(g, t)/√n]." [as printed; see "Printed
     inconsistencies", item 5]

  For the GMM estimator the same algorithm is run with `Φ` from Equation 50 (pp. 70-71).
  The number of draws `B` is not given a value.
- **What the paper's inference does not include.** The paper describes simultaneous
  bootstrap bands for the vector of group-time effects (above) and a bootstrap "confidence
  interval for θ" for a summary parameter (A.2.4, p. 67). No pointwise analytical interval
  formula is printed. Searches for "Rademacher", "wild", "degrees of freedom", "finite sample" and
  "finite-sample" return no hits. "cluster" appears only on pp. 35-36, where it refers to
  French competitiveness clusters; the multipliers `V_i` are iid across individuals.

*Aggregation (Online Appendix A.2.3-A.2.4, pp. 63-67; p. 71):*

"we briefly mention the three main parameters of interest as proposed in Callaway and
Sant'Anna (2021)" (pp. 63-64). As printed (p. 64):

    θ_S(g) = 1/(T - g + 1) sum_{t=g}^{T} ATT(g, t)          θ_S = sum_{g=2}^{T} θ_S(g) Pr(G = g)

    θ_D(e) = sum_{g=2}^{T} sum_{t=g+e}^{T} ATT(g, t) Pr(G = g | t = g + e)      θ_D = 1/(T - 1) sum_{e=1}^{T-1} θ_D(e)

    θ_C(t) = sum_{g=2}^{t} ATT(g, t) Pr(G = g | g ≤ t)      θ_C = 1/(T - 1) sum_{t=2}^{T} θ_C(t)

- Corollary 3 (p. 65): `sqrt(n) (θ̂ - θ) →d N(0, Σ_θ)` with influence function (pp. 65-66)

      l^w(W_i) = sum_{g=2}^{T} sum_{t=2}^{T} { w_gt sum_{τ=g}^{t} ψ_gτ(W_i) + ξ^w_gt(W_i) ATT(g, t) }

  where `ξ^w_gt` "correspond to the estimation errors of weights". The aggregate's
  influence function therefore has a weight-estimation term in addition to the weighted
  chained influence functions.
- A.2.4 (pp. 66-67) repeats the bootstrap for a summary parameter with
  `L(i) = sum_g sum_t w_gt φ_gt(i)`, and "If the weights w_gt's are random, the influence
  function is changed to" `L(i) = sum_g sum_t w_gt φ_gt(i) + γ^w_gt(i) ATT(g, t)`, with the
  worked example `γ̂^w_gt(i) = 1/(T - g + 1) (G_i - (1/n) sum_i G_i)`.
- Under GMM (p. 71): "the choice of the optimal weighting Ω⁻¹ is the same if the objective
  is instead to minimize the variance of a linear transformation R'ATT", and the summary
  bootstrap uses `L(i) = R'Φ(i) + γ^w(i)'ATT`.
- The printed `θ_D(e)` sums over all `t ≥ g + e` and `θ_D` averages `e = 1, ..., T-1` [as
  printed; see "Printed inconsistencies", item 8]. The main text defers to Callaway and
  Sant'Anna (2021) "for a complete discussion of summary parameters" (p. 22).

*Sample selection on observables (pp. 22-23; Proofs 6-8, pp. 72-78):*
- **Assumption 6 (Missing Trends At Random 2), Equation 16** (p. 22):
  `S_{itt+1} ⊥ Y_it+1 - Y_it | X_i, D_i`. **Corollary 1** (Equations 17-18): Theorem 1
  continues to hold with

      w̃^G_{τ,τ-1}(g, X) = E[S_τ S_{τ-1} | G_g] / E[S_τ S_{τ-1} | X, G_g] · w^G_{τ,τ-1}(g, X)

      w̃^C_{τ,τ-1}(g, X) = E[S_τ S_{τ-1} | C]   / E[S_τ S_{τ-1} | X, C]   · w^C_{τ,τ-1}(g, X)

  "one can estimate an attrition model ex-ante for each treatment group and the control
  group ... the attrition model does not require to use the same explanatory variable X than
  the propensity score p_g(X). Remark, however, that we must be able to identify E(G_g) in
  the population." (pp. 73-74). Proof 7 (pp. 74-77) gives the influence function with the
  extra terms `N^G_gt' ξ^ρ_g` and `N^C_gt' ξ^ρ_C` for the two attrition models
  (Equation 58). Proof 7 assumes parametric selection probabilities
  `q_a(X_i) = E[S_τ S_{τ-1} | X_i, a] = Λ(X_i' ρ_a^0)` with "Λ(·) being a known function
  (logit or probit), that can be parametrically estimated by maximum likelihood", and
  asymptotically linear `ρ̂_a` (p. 74). Its influence function uses the modified weights and
  a modified `M_gt` that carries the selection factor `l_t(q_C)`, in addition to the two
  `N' ξ^ρ` terms. Footnote 18 (p. 23) offers a variant: "one can also opt for applying the
  stabilized weights P(S_τ S_{τ-1}|X_i, a_i)⁻¹ / n⁻¹ sum_{i=1}^{n} P(S_τ S_{τ-1}|X_i, a_i)⁻¹
  ex-ante."
- **Assumption 7 (Sequential Missing At Random), Equations 19-20** (p. 23):
  `Y_t ⊥ S_t | X, D, S_{t-1} = 0` and `Y_{t+1} ⊥ S_{t+1} | Y_t, X, D, S_t = 1`.
  **Corollary 2** (Equations 21-22) replaces the denominator by
  `E[S_τ | X, Y_{τ-1}, S_{τ-1} = 1, ·] E[S_{τ-1} | X, ·]`. It is stated under "the
  assumption that an individual is sampled for at most two consecutive periods" (p. 23), and
  "We do not derive the asymptotic distribution for this case, but it follows from the same
  steps as in Corollary 1." (p. 78).
- For general patterns the paper says Assumption 9 "can be relaxed to allow for sample
  selection based on observables, including subpanel membership, as described in
  Corollary 1" (p. 26). No `k`-period version of the Corollary 1 weights is written out.
- Efficiency losses from these first-step estimates are not corrected: "This is left for
  future research." (p. 27; footnote 24 cites Frazier and Renault (2017), Chaudhuri, Min and
  Barnwell (2019) and Sant'Anna and Zhao (2020)).

*Not-yet-treated controls (Online Appendix A.2.5, pp. 67-68):*
- **Assumption 15, Equation 41:** "For all t = 2, ..., T, g = 2, ..., T, such that g ≤ t,"
  `E[Y_t(0) - Y_{t-1}(0) | X, G_g = 1] = E[Y_t(0) - Y_{t-1}(0) | X, D_t = 0] a.s.`
- "using the 'not yet treated' as the control group only changes the weight" to
  `[p_{g,t}(X) (1 - D_t) S_{τ,τ-1} / (1 - p_{g,t}(X))] / E_M[same]`, with
  `p_{g,t}(X) = P(G_g = 1 | X, (G_g = 1 ∪ D_t = 1))` [as printed]. "The propensity scores
  must hence be estimated for pairs (g, t) because the control group evolves through time.
  The asymptotic properties of the two-step estimator remain similar, with minor changes to
  the asymptotic covariance." No influence function is given for this case.

*Other treatment-adjustment variants:* only the inverse-probability-weighted form is
developed. "outcome regression" has no hits; "doubly" appears on p. 27 (footnote 24, as
future work) and in the reference list (pp. 45, 49).

*Regression equivalence (pp. 15, 27-28):*
- Simple setting: the TWFE event-study regression of Equation 10 with individual fixed
  effects reproduces the chained DiD when "the sample is limited to two consecutive
  observations for each individual" (p. 15).
- General framework (Equation 26, p. 27): "This TWFE event-study regression is equivalent to
  the chained DiD estimator, provided we use individual fixed-effects (and not cohort
  fixed-effects), as well as exclude covariates and focus on cases where only two
  consecutive periods are observed per individual." (p. 28). No equivalence is stated for
  the general-pattern GMM estimator.

*Efficiency (Proposition 1, p. 16; Proof 1, pp. 50-54):*
- Under the components-of-variance model of Equation 4 with AR(1) errors
  (`ε_{it+1} = ρ ε_it + η_{it+1}`), "the standard assumptions of DiD settings as presented
  in the proof, and a Missing Completely At Random assumption", the chained DiD "is
  efficient for ρ = 1. In addition, it is more efficient than" the cross-section DiD "if"

      2(t - 1)/(1 + ρ) σ²_η  ≤  1.5 σ²_η/(1 - ρ²) + σ²_α,     for ρ ∈ [0, 1)

- Footnote 15 (p. 16): "'efficient estimator' refers to one with the lowest variance among
  all unbiased estimators."
- The proof derives the cross-section variance behind the `1.5` factor "For t < T" (p. 53),
  because "we have two overlapping samples in each period except for the first and last";
  Proposition 1 does not repeat that restriction.
- Asymptotic variances in the proof share the prefactor `1/(q p (1 - p))` with
  `q = P(S_iτ S_iτ+1)` and `p = P(G_i = 1)`: chained DiD `2(t-1) σ²_η/(1+ρ)` (pp. 52-53);
  long DiD `(ρ^{t-1} - 1)² σ²_{ε_1} + σ²_η sum_{τ=0}^{t-2} ρ^{2τ}` (p. 54).
- The simulations use iid errors, "although the chained DiD perform best for random-walk
  errors" (p. 31).

*Edge cases:*
- **Unit observed once**: contributes to no pairwise outcome difference ("individuals must
  be observed at least twice to be used", p. 24). Footnote 20 (p. 24): refreshment samples
  "do not identify any parameter on their own. They could still be used to address attrition
  ex-ante (Hoonhout and Ridder, 2019)." Such a unit can also enter the propensity-score
  step, whose printed influence function has no sampling indicator.
- **Unit not observed in a given pair**: its weight for that pair is zero. Per the printed
  `ξ^π_g`, it can still enter the propensity-score correction term (see the propensity
  score block above).
- **Unidentified ATT element**: Moore-Penrose pseudo-inverse "to identify the other
  elements" (footnote 22).
- **High-dimensional ΔATT**: Tikhonov-type inverse of Ω, no tuning rule (footnote 22).
- **Few observations in a pair of periods**: no handling specified (see the overlap bullet
  under "Assumption checks / warnings").
- **Stratified sampling**: "The same results apply using a suitably reweighted sample
  (Wooldridge, 2010; Davezies and D'Haultfœuille, 2009)." (footnote 17, p. 20). No formula.
- **Treatment reversal**: excluded by Assumption 4; footnote 16 points to de Chaisemartin
  and D'Haultfœuille (2022c).

*Algorithm (the paper's estimation procedure, pp. 26-27), verbatim:*
1. "Compute Δ_k ATT(g, t) = 1/n sum_{i=1}^{n} [(ŵ(g)^G_{it,t-k} - ŵ(g)^C_{it,t-k})(Y_it - Y_it-k)], for all k, g, t, and stack them into a L_Δ-dimensional vector ΔATT;"
2. "Define the matrix W appropriately;"
3. "Estimate the asymptotic covariance matrix Ω̂ = n⁻¹ΨΨ', where Ψ is a L_Δ × n matrix with elements defined in (48)."
4. "Estimate the optimal GMM estimator ATT^ = (W'Ω̂⁻¹W)⁻¹W'Ω̂⁻¹ΔATT;"

Inference then follows the bootstrap of A.2.2 with `Φ` from Equation 50.

**Reference implementation(s):**
- R: "The R package cdid is available on CRAN." (p. 43). The paper gives no URL, version,
  function name or license. For the cross-section DiD comparator, footnote 7 (p. 11) names
  "the R/Stata package did/csdid".
  **Note:** package metadata checked on the review date (CRAN page for `cdid`, version
  1.4.0.0, published 2026-11-25) lists `License: GPL-3`. diff-diff is MIT: the library must
  NOT port or translate that package's source; it may be used only as a black-box numerical
  oracle (run it, compare outputs). This review was written from the paper alone; no package
  source was opened. The package also offers options that are not in the paper (the review
  date's README describes a default weighting that the paper does not contain), so parity
  with package defaults is not the same thing as fidelity to the paper.
- Stata: no chained-DiD implementation is named in the paper.
- Python: no implementation is named.

**Requirements checklist:**
- [ ] Target `ATT(g, t)` for `2 ≤ g ≤ t ≤ T` with never-treated controls (Theorems 1-3).
- [ ] Each pairwise difference uses only units observed in both periods; each arm's raw
      weight is divided by its sample mean over all `n` units (zeros for units not observed
      in both periods), with the outer average also over all `n` units.
- [ ] Control-arm weights use the odds `p̂_g/(1 - p̂_g)` of a parametric (logit or probit)
      propensity score defined on cohort `g` plus never-treated units.
- [ ] Stack the pairwise estimates, build `W` from `Δ_k ATT(t) = ATT(t) - ATT(t - k)`, and
      solve by GMM. State which weighting matrix is used; the paper's proposed estimator
      uses `Ω̂⁻¹`.
- [ ] Unidentified `ATT` elements are reported as such (pseudo-inverse identifies "the other
      elements"), not filled with minimum-norm values.
- [ ] Influence functions include the weight-normalisation recentring and the
      propensity-score correction `M' ξ^π_g`, with the sign and denominator issues in
      "Printed inconsistencies" resolved and documented.
- [ ] Variance: Equation 49, `Σ = (W'Ω⁻¹W)⁻¹`, is derived under optimal weighting. Any
      other weighting of an overidentified stack needs its own covariance formula (the
      sandwich form), documented as a derivation.
- [ ] Aggregates carry the weight-estimation term of `l^w` (Corollary 3).
- [ ] Every choice the paper leaves open (see "Gaps and Uncertainties") is documented as an
      implementation choice with its own validation.

---

## Simulation Evidence (Section 3, pp. 28-32)

**Design.** Potential outcomes follow Equation 27 (p. 28):
`Y_it(D_i) = α_i + δ_t + sum_{τ=2}^{t} β_τ D_iτ + ε_it`, with `α_i ~ N(1, 2)`,
`δ_t ~ N(1, 1)`, `ε_it ~ N(0, 0.5)` iid, and `X_i ~ N(1, 1)` (pp. 28-29). Treatment and
sampling probabilities (p. 29):

    Pr(G_ig = 1 | X_i, α_i, G_ig + C_i = 1) = 1 / (1 + exp(θ_0 + θ_1 X_i + θ_2 α_i × g)),   θ_0 = -1, θ_1 = 0.4, θ_2 ∈ {0, 0.2}

    Pr(S_itt+1 = 1 | α_i) = 1 / (1 + exp(λ_0 + λ_1 α_i × t)),                               λ_0 = -1, λ_1 ∈ {0, 0.2}

"we set T = 6, so there is a total of 8 periods. In each period, the population size is
4800, and we draw 150 individuals such that S_itt+1 = 1." (p. 30). `β_τ` takes "the values
{1.75, 1.50, 1.25, 1.00, 0.75, 0.50} for τ = {1, 2, 3, 4, 5, 6}" (pp. 30-31). Step 2(d):
"Repeat steps 1(b)-2(c) 1,000 times and report the mean and standard deviation of the
estimators." (p. 30).

| DGP | θ_2 | λ_1 | Description (p. 30) |
|---|---|---|---|
| 1 | 0 | 0 | "the baseline case ... all estimators are unbiased" |
| 2 | 0.2 | 0.2 | treatment and sampling depend on individual heterogeneity, "leading to biased estimates for the cross-section DiD" |
| 3 | 0.2 | 0.2 | "stratified sample where 90% of the individuals are sampled on a rotating basis ... and those with α_i superior to the 90th percentile are always observed (10%)" |
| 4 | 0.2 | 0.2 | "individuals with α_i superior to the 60th percentile are always observed (40%) and the rest is drawn as a rotating subpanel (60%)" |

**Metrics reported.** Means and standard deviations over 1,000 replications only. A
full-text search for "coverage", "RMSE" and "power" returns no hits; "rejection" (pp. 33,
34) refers to project selection and "size" (pp. 9, 30, 33, 34) to sample, population, firm
or project size, not to test size. **The paper
reports no evidence on confidence-interval coverage or on the accuracy of estimated standard
errors.**

**Table 3 (p. 31), "Simulation results for a rotating panel"** - mean, with the standard
deviation in parentheses:

| | DGP 1 Chained DiD | DGP 1 CS DiD | DGP 1 Long DiD | DGP 2 Chained DiD | DGP 2 CS DiD | DGP 2 Long DiD |
|---|---|---|---|---|---|---|
| β_1 | 1.748 (0.099) | 1.745 (0.199) | 1.75 (0.017) | 1.752 (0.097) | 1.894 (0.148) | 1.75 (0.016) |
| β_2 | 1.5 (0.164) | 1.498 (0.305) | 1.5 (0.02) | 1.499 (0.157) | 1.774 (0.218) | 1.501 (0.018) |
| β_3 | 1.248 (0.231) | 1.25 (0.355) | 1.251 (0.023) | 1.254 (0.224) | 1.651 (0.257) | 1.25 (0.02) |
| β_4 | 1 (0.3) | 1.006 (0.412) | 1 (0.027) | 1.002 (0.293) | 1.522 (0.291) | 1 (0.023) |
| β_5 | 0.741 (0.406) | 0.765 (0.521) | 0.75 (0.033) | 0.739 (0.395) | 1.369 (0.365) | 0.75 (0.028) |
| β_6 | 0.499 (0.586) | 0.522 (0.711) | 0.499 (0.046) | 0.5 (0.603) | 1.209 (0.515) | 0.503 (0.04) |

For DGPs 1-2 the long DiD assumes "that all sampled individuals are observed for the entire
time frame" and "is infeasible but serves as a benchmark" (p. 31). The weighting matrix
behind Table 3's "Chained DiD" column is not stated.

**Table 4 (p. 32), "Simulation results for a stratified sample":**

| | DGP 3 Ch DiD | DGP 3 CD-GMM | DGP 3 CS DiD | DGP 3 Long DiD | DGP 4 Ch DiD | DGP 4 CD-GMM | DGP 4 CS DiD | DGP 4 Long DiD |
|---|---|---|---|---|---|---|---|---|
| β_1 | 1.753 (0.085) | 1.748 (0.151) | 1.714 (0.14) | 1.753 (0.085) | 1.754 (0.052) | 1.755 (0.057) | 1.715 (0.096) | 1.754 (0.052) |
| β_2 | 1.501 (0.127) | 1.488 (0.195) | 1.413 (0.208) | 0.897 (0.315) | 1.502 (0.061) | 1.503 (0.06) | 1.422 (0.137) | 1.504 (0.069) |
| β_3 | 1.256 (0.177) | 1.237 (0.234) | 1.112 (0.239) | 0.868 (0.311) | 1.252 (0.072) | 1.25 (0.062) | 1.135 (0.148) | 1.25 (0.068) |
| β_4 | 1.005 (0.215) | 0.988 (0.285) | 0.8 (0.268) | 0.812 (0.336) | 1.003 (0.077) | 1.002 (0.065) | 0.855 (0.158) | 1.002 (0.071) |
| β_5 | 0.749 (0.285) | 0.749 (0.331) | 0.459 (0.339) | 0.706 (0.352) | 0.75 (0.09) | 0.751 (0.074) | 0.567 (0.183) | 0.751 (0.078) |
| β_6 | 0.508 (0.412) | 0.52 (0.398) | 0.123 (0.479) | 0.511 (0.393) | 0.509 (0.121) | 0.511 (0.094) | 0.316 (0.237) | 0.511 (0.097) |

"For DGPs 3 and 4, the long DiD is estimated only for individuals that belong to the
balanced subpanel." (p. 31). "It appears that when the balanced subpanel consists of only
10% of the data, the chained DiD estimators outperform the long DiD." (p. 31).

**[Reviewer check - read from Table 4, not statements by the authors]**
- DGP 3 (10% always observed): the optimal-weighting standard deviation exceeds the
  identity-weighting one for `β_1` to `β_5` (0.151 vs 0.085; 0.195 vs 0.127; 0.234 vs 0.177;
  0.285 vs 0.215; 0.331 vs 0.285) and is smaller only for `β_6` (0.398 vs 0.412).
- DGP 4 (40% always observed): the optimal-weighting standard deviation exceeds the identity
  one only for `β_1` (0.057 vs 0.052) and is smaller for `β_2` to `β_6`.
- DGP 3: the long DiD means for `β_2` to `β_5` (0.897, 0.868, 0.812, 0.706) are far from the
  true values (1.50, 1.25, 1.00, 0.75). The text does not explain this pattern.

---

## Application (Section 4, pp. 32-43; Online Appendix B, pp. 79-84)

French collaborative R&D subsidies, 2010-2016, firm data 2007-2017, "about 30,000 firms in
the sample" (p. 36). The R&D survey has "a stratified sampling design: firms with intramural
R&D expenditures above 750,000 euros are systematically surveyed the following year, while
others are surveyed only two years in a row" (p. 35). The roughly 30,000 firms describe the
sample before balancing. Points that bear on implementation:

- **Sample construction.** "We balance the data to both facilitate the comparison across
  estimators and get closer to our theoretical framework. That is, we keep the firms that
  are consistently observed from 2007 to 2017 in the administrative data." (p. 37). "Then,
  we construct an unbalanced version of this panel by discarding all observations whenever
  a firm was not sampled in the R&D survey in a given year." (p. 38). Tables B.4-B.6 repeat
  the analysis "with the complete original sample from the administrative data, without
  discarding the individual firms that are not consistently observed throughout the period
  to create a balanced panel" (pp. 42-43). The appendix describes the same tables as
  results "where administrative data are not reweighted using the exhaustively observed
  variables" (p. 81) [as printed].
- **Controls.** "a treated company at a given period often serves as a control for other
  companies in the period preceding its treatment (the 'not yet treated')" (p. 34).
- **Selection correction.** "inverse propensity weighting of the first-differences
  Y_t - Y_{t-1} following Corollary 1 under a variation of Assumption 6", with "a logit
  model for the conditional probability P(S_tt+1 = 1|Z)" (p. 37). Footnote 33: "this
  simplified approach assumes sampling S_{t-1}S_t to be mean-independent of treatment status
  D and treatment selection variables X, conditional on Z."
- **Inference.** "95% confidence intervals obtained from the multiplier bootstrap" (p. 38).
  The number of draws, the band type (pointwise or simultaneous) and the weighting matrix
  used for the chained DiD are not stated in Section 4 or in the table notes.
- **Pre-trends.** p. 38: "pre-trend tests rejecting the null hypothesis of pre-trends";
  p. 42: "placebo tests for pre-trends show no significant employment differences upstream
  of initial treatment period ... (effects β_{-3}, β_{-2} and β_{-1} in Tables B.1 to B.3)."
  [as printed; p. 38 says "null hypothesis of pre-trends", which is ambiguous. The two
  sentences conflict only if that is read as a null of no pre-trends.]

**Table B.1 (p. 79), "Effects on total workforce (exhaustively observed outcome)"**,
outcome log(total workforce), point estimate with 95% interval:

| | Long DiD, Exhaustive (1) | Chained DiD, Exhaustive (2) | Chained DiD, Unbalanced (3) | Cross Section DiD, Exhaustive (4) | Cross Section DiD, Unbalanced (5) |
|---|---|---|---|---|---|
| β_{-3} | -0.013 [-0.084,0.058] | -0.013 [-0.079,0.053] | 0.032 [-0.059,0.123] | -0.013 [-0.144,0.117] | 0.075 [-0.609,0.76] |
| β_{-2} | -0.023 [-0.091,0.044] | -0.023 [-0.087,0.041] | -0.008 [-0.097,0.081] | -0.023 [-0.13,0.083] | 0.012 [-0.549,0.574] |
| β_{-1} | -0.005 [-0.054,0.044] | -0.005 [-0.052,0.041] | -0.005 [-0.066,0.056] | -0.005 [-0.09,0.08] | 0.024 [-0.442,0.491] |
| ref. | 0 | 0 | 0 | 0 | 0 |
| β_1 | 0.027 [-0.026,0.08] | 0.027 [-0.023,0.078] | 0.046 [-0.021,0.112] | 0.028 [-0.103,0.159] | 0.062 [-0.372,0.495] |
| β_2 | 0.057** [-0.006,0.121] | 0.056** [-0.005,0.118] | 0.081** [0.002,0.159] | 0.057 [-0.048,0.163] | 0.092 [-0.426,0.61] |
| β_3 | 0.07** [0,0.14] | 0.068** [0,0.135] | 0.089** [0.006,0.172] | 0.069 [-0.075,0.214] | 0.051 [-0.506,0.607] |
| β_4 | 0.084** [0.002,0.167] | 0.077** [0,0.154] | 0.119*** [0.024,0.214] | 0.083 [-0.091,0.257] | 0.087 [-0.53,0.704] |
| β_5 | 0.123*** [0.032,0.213] | 0.111*** [0.025,0.197] | 0.149*** [0.037,0.261] | 0.119* [-0.032,0.269] | 0.076 [-0.533,0.686] |
| Pre-trend | -0.014 [-0.048,0.021] | -0.014 [-0.049,0.021] | 0.006 [-0.04,0.053] | -0.014 [-0.079,0.051] | 0.037 [-0.294,0.368] |

Note to the table (p. 79): "'Exhaustive' refers to the use of an exhaustively observed
outcome without pretending that this variable is imperfectly observed. 'Unbalanced' refers
to the use of an exhaustively observed outcome pretending that this variable is observed
from R&D survey, that is, from an unbalanced repeated panel. 95% confidence intervals are
obtained from the multiplier bootstrap." The table prints the legend `* p < 0.10`,
`** p < 0.05`, `*** p < 0.01`. Tables B.2, B.4 and B.5 (pp. 80, 82, 83) use the same
five-column layout; Tables B.3 and B.6 (pp. 81, 84) use four columns, total workforce and
researchers under the chained and the cross-section DiD. Tables B.2-B.6 are not transcribed
here. The application data are French administrative and survey
sources (footnote 29, p. 35) and are not a replication target for the library.

**[Reviewer check - read from Table B.1]** Several starred estimates have a printed 95%
interval that includes zero (for example `β_2`, column (1): 0.057** with [-0.006, 0.121]).
The notes give the star thresholds but do not say how the underlying p-values are computed
or why a starred estimate can have a 95% interval containing zero.

---

## Implementation Notes

### Data Structure Requirements
- A panel in which a unit may be observed at any subset of periods; a unit needs at least
  two observed outcomes to contribute directly to a pairwise outcome difference (p. 24).
  Units without such a pair can still inform the propensity score or an attrition model.
- Cohort membership (first treatment period, or never treated) and covariates for every
  unit, including units with missing outcomes: "X and D_i are assumed to be observed for all
  individuals" (p. 20).
- Binary, absorbing treatment (Assumption 4). Cohorts are indexed from `g = 2`.
- Covariates enter the assumptions without a time index.

### Computational Considerations
- One pairwise estimate per `(g, t, k)`; the stack has `L_Δ` rows and `Ψ` is `L_Δ × n`.
  The paper claims "efficiency gains ... without much additional computational complexity"
  (p. 25) and gives no complexity analysis.
- Optimal weighting inverts the `L_Δ × L_Δ` matrix `Ω̂`; footnote 22 anticipates that this
  is unstable "if ΔATT is high-dimensional".
- The bootstrap perturbs fixed influence functions and does "not re-estimate propensity
  scores and parameters for each draw" (p. 62).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| GMM weighting matrix | choice | optimal (`Ω̂⁻¹`) is the proposed estimator; identity is the variant of footnote 23 and of "Ch DiD" in Section 3 | No rule; see the small-sample caveat on p. 31 |
| Propensity link `Λ` | choice | "logit or probit" | Not discussed |
| Regularisation of `Ω̂` | scalar | none given | Not discussed (footnote 22 names Tikhonov without a parameter) |
| Bootstrap draws `B` | integer | none given | Not discussed |
| Multiplier distribution | choice | Mammen (1993) two-point "such as" | Any iid mean-zero, unit-variance draw (p. 62) |
| Control group | choice | never treated | Not-yet-treated per Online Appendix A.2.5 |

### Relation to Existing diff-diff Estimators
*Library facts below were read from the source on the review date; they are not from the paper.*
- **CallawaySantAnna** (`diff_diff/staggered.py`) estimates the same `ATT(g, t)` target and
  already offers an inverse-probability-weighting estimation method, never-treated and
  not-yet-treated control groups, a multiplier bootstrap whose weight options include
  Mammen two-point draws, and sup-t simultaneous bands for the event-study aggregation.
  Its unbalanced-panel option routes pooled observations through the
  repeated-cross-section levels estimator. That route is the paper's "cross-section DiD",
  which Theorem 1 says "does not identify ATT(g,t) under Assumption 2".
- **No chained estimator on the review date.** A search of `diff_diff/` for "chained did",
  "chained difference" and "Bellégo" (case-insensitive) finds no estimator that combines
  pairwise differences across differently observed subsamples.
- **Reduction claims must be verified numerically.** On a balanced panel every pair of
  periods is observed for the same units, and Table 4 prints identical "Ch DiD" and "Long
  DiD" cells for `β_1`. Any claim that an implementation reduces to an existing library
  estimator on a balanced panel (and under which base-period and control-group settings)
  is a test to write, not a statement from the paper.
- **Aggregation.** The paper's summary parameters are those of Callaway and Sant'Anna
  (2021) with the chained influence functions substituted; the library's existing
  aggregation machinery is the natural host, provided the weight-estimation term of `l^w`
  is carried.

---

## Printed Inconsistencies

Transcribed as printed above. Each needs a documented resolution in an implementation.

1. **Index range of Assumption 3 (p. 18).** Equation 14 is stated for the increment
   `Y_{t+1}(0) - Y_t(0)` with `g ≤ t ≤ T`. Read literally it omits the increment from `g-1`
   to `g` that the first chain link uses, and it involves `Y_{T+1}`. Theorem 1 and its proof
   use increments from `τ-1` to `τ` for `τ = g, ..., t`, and the not-yet-treated analogue
   (Equation 41) is written for `Y_t(0) - Y_{t-1}(0)`.
2. **Sign and superscript in the influence function.** Equation 37 (p. 58) prints
   `ψ_gt = ψ^G_gt + ψ^G_gt`; p. 63 and Equation 48 (p. 70) print `ψ^G + ψ^C`; Equation 58
   (p. 74) again prints `ψ^G_gt + ψ^G_gt`. The estimator is
   `ΔATT^_g(g, τ) - ΔATT^_C(g, τ)` (p. 58), and Equations 38-39 assign `ψ^G` and `ψ^C` to
   those two pieces. **[Reviewer check - not in paper]** Those three statements together
   imply `ψ^G - ψ^C`. The sign matters because `M' ξ^π_g(i)` is non-zero for cohort-`g`
   units as well as controls.
3. **Denominator of `M_gt`.** Printed as `E_M[p_g(X) C / (1 - p_g(X))]` on p. 58 and as
   plain `E[p_g(X) C / (1 - p_g(X))]` on pp. 63 and 70; all three omit the sampling
   indicator. The derivation on p. 61 divides by `E_M[w_t(p_g)]` with
   `w_t(g) = g(X) C S_{t,t-1} / (1 - g(X))`.
4. **Bootstrap scaling (p. 62).** Step 2 prints `ATT^ + Φ̂ V^b` with no `1/n`.
   **[Reviewer check - not in paper]** For `R̂*b = √n (ATT^*b - ATT^)` to have variance `Σ`
   the draw needs a `1/n` factor on `Φ̂ V^b`.
5. **Band half-width (p. 63).** Step 8 prints `ĉ_{1-α} Σ̂^{-1/2}(g, t)/√n`, while step 5
   defines the scale as `Σ̂^{1/2}(g, t)`. The same exponent appears in A.2.4 step 8 (p. 67).
6. **Table 2 example (pp. 24-25).** The text says "all treatments take place at t = 1",
   while Equation 24 has no column for `ATT(1)` and the proof sets `ATT(1) = 0` "by
   construction".
7. **Proposition 1 example (pp. 17, 54).** "If ρ = 0, this condition becomes σ²_η/2 ≤ σ²_α
   for t = 2, and 5σ²_η/2 ≤ σ²_α for t = 6." **[Reviewer check - not in paper]** At `ρ = 0`
   the printed inequality is `(2(t-1) - 1.5) σ²_η ≤ σ²_α`: 0.5 at `t = 2`, 2.5 at `t = 3`,
   and 8.5 at `t = 6`.
8. **`θ_D(e)` (p. 64).** The inner sum runs over all `t = g+e, ..., T` while the weight is
   `Pr(G = g | t = g + e)`, and `θ_D` averages `e = 1, ..., T-1` although `e` ranges over
   `{0, 1, 2, ...}`.
9. **Assumption ranges.** Assumption 7 is stated "For all t = 1, ..., T - ∞" (p. 23);
   Assumption 6 "For all t = 2, ..., T" against Assumption 2's "t = 1, ..., T - 1".
10. **Not-yet-treated propensity score (p. 68).** Printed as
    `P(G_g = 1 | X, (G_g = 1 ∪ D_t = 1))` although the control group is `1 - D_t`.
11. **Simulation population size (pp. 29-30).** Step 1(a) draws
    `N = 2 × max_t { n / E_α[Pr(S_itt+1)] }` individuals per period, and the text then says
    "the population size is 4800". **[Reviewer check - not in paper]** With `n = 150`,
    `λ_0 = -1`, `α_i ~ N(1, 2)` and `t = 0, ..., 7`, a Monte Carlo evaluation of that formula
    gives about 410 for `λ_1 = 0` and about 683 for `λ_1 = 0.2`.
12. **Simulation indexing (pp. 28-31).** Equation 27 sums `β_τ D_iτ` over calendar index
    `τ = 2, ..., t`, while the values are assigned "for τ = {1, 2, 3, 4, 5, 6}" as effects
    "relative to the treatment starting date". Step 1(b) writes `G_it` where the definition
    uses `G_ig`, and step 2(d) repeats "steps 1(b)-2(c)", which excludes the population
    draw in step 1(a).
13. **Summary-parameter bootstrap (A.2.4, pp. 66-67).** Step 2 prints
    `θ̂*b = θ̂ + L̂' V^b` with no `1/n`, the same scaling issue as item 4. Step 6 keeps the
    group-time indices and a maximum, `max_{(g,t)} |R̂*b(g, t)| Σ̂^{-1/2}(g, t)`, although
    step 3 defines a scalar `R̂*b` for the summary parameter.
14. **Centring in `N^G_gt` (p. 75).** The treated-arm selection correction `N^G_gt` is
    printed with the control-arm weighted mean, `E_M[w̃^C_{it,t-1}(g, X)(Y_it - Y_it-1)]`,
    inside its recentring term, whereas `ψ^G_gt` (p. 74) recentres with the treated-arm
    weight `w̃^G`. `N^C_gt` and `N^G_gt` also print `q(X; π̄)` where the selection model's
    parameter is `ρ`. Corollary 1 inference needs these re-derived before implementation.

---

## Gaps and Uncertainties

These are the points where the paper does not determine an implementation. Each is a
library design choice that needs its own documentation and validation.

1. **Pre-treatment estimates and earlier periods.** The numbered identification and limit
   theorems cover `2 ≤ g ≤ t ≤ T` only. The paper reports pre-treatment coefficients in its
   application, and it provides a regression containing pre-treatment cohort-by-period
   coefficients (Equation 26) that it states is equivalent to the chained DiD in the
   no-covariate, two-consecutive-observation setting (pp. 27-28). It does not specify
   cohort-specific pre-treatment moments for the covariate-adjusted general-pattern GMM
   estimator or the application's pre-treatment calculations. Its displayed stack ("for all
   k, g, t", Equation 43) includes links that begin before `g - 1`, but the paper does not
   reconcile that construction, or its period-1 reference, with the post-treatment-only
   ranges of Assumption 3 and Theorem 3 (pp. 24-28, 63, 68-69, 79-84). The reference period
   per cohort and the treatment of pre-treatment parameters are therefore implementation
   choices.
2. **Variance under identity weighting.** The paper uses identity weighting in one
   simulation table. For the general overidentified GMM estimator it prints the asymptotic
   variance only for optimal weighting (Equation 49); the rotating-panel covariance
   `Σ = E_M[ΦΦ']` (p. 62) covers the first-difference chain that footnote 23 identifies with
   identity weighting. The general sandwich form above is this review's derivation.
3. **Analytical pointwise inference.** The paper's inference is a multiplier bootstrap:
   simultaneous bands for the group-time effects and a bootstrap interval for each summary
   parameter. Analytical standard errors from the influence functions follow the
   Callaway and Sant'Anna (2021) pattern the paper adapts, but no such formula is printed.
4. **Thinly observed pairs.** No condition, exclusion rule or weighting adjustment for pairs
   of periods with few jointly observed units appears, and the simulations (150 units per
   pair, 1,000 replications) report means and standard deviations only. The paper offers no
   evidence on interval coverage in any design.
5. **Propensity-score estimation sample.** The printed influence function corresponds to one
   fit per cohort on all cohort-`g` and never-treated individuals. Footnote 21 describes
   estimating each link separately within each subsample (missing-data pattern) before
   stacking, with propensity scores conditional on subsample membership. The paper does not
   describe fitting a separate propensity score on the pooled observations available for
   each pair of periods; that choice has a different influence function, which the paper
   does not derive.
6. **Clustering and dependence.** All results assume iid sampling units; no cluster-robust
   variance or clustered multiplier draw is discussed.
7. **Anticipation.** Not addressed by name; the chain's first link assumes the period
   before treatment is unaffected.
8. **Time-varying covariates.** The assumptions use `X_i`. The application says covariates
   "must be observed in g - 1, t, and t + 1" (p. 36), which suggests dated covariates, but
   the theory has no time index on `X`.
9. **Selection corrections beyond two consecutive periods.** Corollary 1 is written for
   one-period links and Corollary 2 for units "sampled for at most two consecutive periods".
   Corollary 2's asymptotic distribution is not derived; the paper says it "follows from the
   same steps as in Corollary 1" (p. 78).
10. **Not-yet-treated controls.** One page (A.2.5), no influence function, and a printed
    conditioning set that looks wrong (item 10 above).
11. **Regularisation and rank.** Pseudo-inverse and Tikhonov are named without parameters.
    How an estimated `Ω̂` behaves when some stacked estimates rest on very few units is not
    discussed.
12. **Published version.** Numbering and any corrections in the journal version were not
    checked.

---

## Verification Stamp

The synthesis above was written by the reviewer from five page-range extractions (pp. 1-20,
17-31, 28-43, 50-66, 64-84). The following load-bearing items were re-read directly against
the PDF page images by the reviewer and match the transcriptions above:

- pp. 18-27: Assumptions 1-9, Equations 12-26, Theorems 1-3, Corollaries 1-2, the weights,
  Table 2, footnotes 16-24, and the four-step estimation procedure.
- pp. 54-58: Proof 2 (Equations 32-36), the propensity-score paragraph, Equation 37, `M_gt`,
  the short-term estimator and Equations 38-39.
- pp. 62-63: the asymptotic result for `ATT`, the eight-step bootstrap, and the `Φ̂` recipe.
- pp. 68-71: A.2.5 weight, A.2.6 moments (Equations 42-43), the GMM derivation
  (Equations 44-47), conditions (i)-(vi), Equation 48, `M_gtk`, `ξ^π_g`, Equations 49-50.

Quoted sentences on pp. 3, 9, 11, 20, 22, 24, 31, 32, 34, 36, 38, 42, 43 and 50 were checked
against the text dump. Tables 3, 4 and B.1 were transcribed by an extraction pass that
compared every cell between the page image and the text dump; the reviewer did not re-read
those tables cell by cell. The absence claims rest on the full-text searches named at each
point.
