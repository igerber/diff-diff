# Paper Review: Omitted Variable Bias in Difference-in-Differences Designs

**Authors:** Juejue Wang, Pedro H. C. Sant'Anna, Victor Chernozhukov, Carlos Cinelli
**Citation:** Wang, J., Sant'Anna, P. H. C., Chernozhukov, V., & Cinelli, C. (2026). Omitted Variable Bias in Difference-in-Differences Designs. arXiv preprint arXiv:2609.19386. https://arxiv.org/abs/2609.19386
**PDF reviewed:** https://arxiv.org/pdf/2609.19386v1 (arXiv v1, [stat.ME], 16 Sep 2026; 115 pages; SHA-256 `b2d24d135f15ce0b129b49a4689272267569a0c2fa674613f3c6719c8b7b5df2`)
**Review date:** 2026-11-25

> **Preprint status.** This is an unpublished arXiv preprint (v1). The library's usual rule
> is to source methodology from published papers; reviewing it is an explicit exception made
> to evaluate a contributed implementation. **All equation, assumption, lemma, theorem,
> proposition, corollary, remark, definition, algorithm, table, figure, and page references
> below are pinned to arXiv v1.** Re-check every number against any later arXiv version or a
> published version before citing in user-facing docs.
>
> **Page convention.** "p. N" is always the PDF page index. The main text (Sections 1-7 and
> its reference list) occupies PDF pp. 1-36 and its printed page numbers equal the PDF page.
> The Supplementary Materials (Appendices A-G and a second reference list) occupy PDF
> pp. 37-115 and restart their printed numbering at 1, so **printed appendix page = PDF
> page - 36** (PDF p. 59 prints "23").
>
> **Labels used in this review.** Text in quotation marks and equations in code blocks are
> transcribed from the paper. Anything marked **[Reviewer check - not in paper]** is this
> review's own algebra or arithmetic on the paper's stated formulas and numbers; it is not a
> statement made by the paper. Where the review says the paper has no statement on a topic,
> the claim rests on a full-text search of a `pdftotext -layout` dump of all 115 pages for
> the terms listed at that point (a term split across a line break could be missed).

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## DiD OVB Sensitivity` section into the diagnostics / sensitivity part of the registry once a feature ships (rename to the shipped class name).*

## DiD OVB Sensitivity (prospective; name provisional)

**Primary source:** Wang, J., Sant'Anna, P. H. C., Chernozhukov, V., & Cinelli, C. (2026). Omitted Variable Bias in Difference-in-Differences Designs. arXiv:2609.19386v1 (numbering below follows v1; unpublished preprint - see the status note above).

The paper studies the canonical two-group, two-period DiD when parallel trends holds only
after conditioning on observed covariates `X` **and** an unobserved vector `U`. The
researcher can estimate only the "short" estimand `θ_s` (conditioning on `X`); the target is
the "long" estimand `θ` (conditioning on `X` and `U`). **Theorem 1** (p. 14) gives an exact
formula for the omitted variable bias, localized on the untreated group:
`θ - θ_s = -ρ_0 C_{0ΔY} C_{0D} S_0`, the product of three non-identified bias factors
(alignment, trend, selection) and one identified scale factor. The trend and alignment
factors are bounded by one; the selection factor is unbounded and "must always be
restricted" (p. 15). Section 3.3 re-expresses the selection factor as an increase in average
treatment odds among the treated and as a chi-squared covariate imbalance. Section 3.4 and
Appendix F compare this "conditional" parameterization with the "unconditional" one of
Chernozhukov et al. (2026) and the "centered" one of Huang and Pimentel (2025).

Section 4 gives two benchmarking procedures: against observed covariates (relative-strength
multipliers `k_{0ΔY,j}`, `k_{0D,j}` and an observed alignment `ρ_{0,j}`) and against one
pre-treatment period (two ways of extrapolating the pre-trend bias). Section 5 estimates the
three identified components `θ_s`, `σ²_{0s}`, `ν²_{0s}` by debiased machine learning with
Neyman-orthogonal scores (Equations 9-11, Algorithm 1), gives delta-method confidence limits
for the bounds `θ_±` (**Theorem 2**), and defines two routine-reporting statistics: the
**robustness value (RV)** and the **extreme robustness value (XRV)** (Section 5.3,
Definitions 3-4, Corollary 4). Section 6 applies everything to the 2007 minimum-wage cohort
of Callaway and Sant'Anna (2021). Appendix E reports Monte Carlo evidence on a Gaussian
design with closed-form bias factors.

**Key implementation requirements:**

*Assumption checks / warnings:*

Setup (Section 3.1, p. 11). Periods `t ∈ {1, 2}` (pre, post); `D ∈ {0, 1}` is treatment
group assignment; `Y_t(1)`, `Y_t(0)` are potential outcomes; `ΔY := Y_2 - Y_1`;
`ΔY(0) := Y_2(0) - Y_1(0)`; `X` observed covariates; `U` unobserved covariates.

- **Consistency / SUTVA** (Equation 1, p. 11):
  ```
  Y_t = D Y_t(1) + (1 - D) Y_t(0).                                           (1)
  ```
- **Conditional parallel trends given (X, U)** (Equation 3, p. 11; Assumption A.3, p. 44):
  ```
  E[ΔY(0) | D = 1, X, U] = E[ΔY(0) | D = 0, X, U].                           (3)
  ```
  Parallel trends given `X` alone is NOT assumed: "the parallel trends assumption may not
  hold if we condition on X alone, and our estimates may thus suffer from omitted variable
  bias" (p. 12).
- **Assumption 1 (Regularity Conditions)** (p. 12): "We assume the outcome evolution is
  square-integrable `E[ΔY²] < ∞` and weak overlap"
  ```
  E[ P(D = 1)^{-1} P(D = 0 | X, U)^{-1} | D = 1 ] < ∞.
  ```
  This is the only condition Theorem 1, Lemma 1, Lemma 2 and Corollary 1 invoke.
- **Appendix A.4 identification assumptions** (p. 44), used for Proposition A.4 (the long
  estimand equals the ATT):
  - **Assumption A.1 (Consistency):** "Observed outcomes are generated as `Y := Y(D)`."
  - **Assumption A.2 (No anticipation):** `E[Y_1(0) | D = 1, X, U] = E[Y_1(1) | D = 1, X, U]` a.s.
  - **Assumption A.3 (Conditional parallel trends):** Equation 3, a.s.
  - **Assumption A.4 (Strong overlap):** "There exists `ε > 0` such that
    `p := P(D = 1) ≥ ε`, and `π_{XU} := P(D = 1 | X, U) ≤ 1 - ε`, a.s." Only an upper bound
    on the long propensity score is imposed.
  - **Remark 3** (p. 44): "Assumption A.4 is stronger than what is needed to derive the OVB
    formula of Theorem 1, in which we impose only the weak overlap condition of
    Assumption 1."
- **Sampling** (p. 25): "we assume we have an i.i.d. sample from the observed data
  distribution." The paper has no statement on clustered, weighted, or survey sampling:
  the dump has zero hits for `cluster`, `survey`, `unbalanced`, `missing`; the hits for
  `weights` (pp. 6, 13, 86, 87, 94, 108) are all rebalancing / balancing / bootstrap
  multiplier weights, not sampling weights. In the application treatment is assigned at the
  state level and the unit is the county (p. 7); the paper does not discuss inference at the
  state level.
- **Data structure.** `ΔY` is a unit-level variable, so both periods must be observed for
  the same unit. The paper never uses the phrases "panel data" or "repeated cross" (zero
  hits each); a repeated-cross-sections version is not developed.
- **Remark 1** (p. 12): the results "apply to any such estimand that can be expressed in the
  generic form (4)", including group-time ATTs of Callaway and Sant'Anna (2021) "after
  appropriately redefining assumptions and variables", and the ATT under unconfoundedness,
  for which "one simply replaces ΔY with Y".
- **Overlap in practice.** The paper prescribes no propensity-score trimming, clipping, or
  weight normalization rule: zero hits for `trim`, `clip`, `truncat`, `winsor`, `ajek`
  (Hajek). The nearest statements are Remark 6 (pp. 65-66: strong overlap implies
  `E[α_0² | D = 0] < ∞`), footnote 2 (p. 66: the simulation sets `σ²_{0·}` "sufficiently
  larger than" `σ²_{1·}` "to ensure overlap"), and the misspecification finding (p. 68) that
  parametric estimates of `ν²_{0s}` "are highly sensitive to the estimated propensity
  scores".

*Target parameters and identification (Section 3.1-3.2, pp. 11-14; Appendix A.4):*

```
ATT := E[Y_2(1) | D = 1] - E[Y_2(0) | D = 1].                                (2)

θ   := E[ E[ΔY | D = 1, X, U] - E[ΔY | D = 0, X, U]  |  D = 1 ].             (4)   "long"

θ_s := E[ E[ΔY | D = 1, X]    - E[ΔY | D = 0, X]     |  D = 1 ].             (5)   "short"
```

- **Proposition A.4** (p. 44): under Assumptions A.1-A.4, `ATT = θ = E[ΔY - g_0(X, U) | D = 1]`.
- OVB as a parallel-trends deviation (unnumbered display, p. 12; **Proposition F.7**, p. 96):
  ```
  θ - θ_s = - E[ E[ΔY(0) | X, D = 1] - E[ΔY(0) | X, D = 0]  |  D = 1 ].
  ```
- Untreated regression functions and compact forms (p. 13):
  ```
  g_0 := E[ΔY | D = 0, X, U],        g_{0s} := E[ΔY | D = 0, X].

  θ = E[ΔY - g_0 | D = 1],           θ_s = E[ΔY - g_{0s} | D = 1].

  θ - θ_s = -(θ_0 - θ_{0s}),   θ_0 := E[g_0 | D = 1],   θ_{0s} := E[g_{0s} | D = 1].     (6)
  ```
  "The average treated trend `E[ΔY | D = 1]` is the same in both the long and short
  parameters, and it cancels out from the bias."
- Odds (p. 13; Appendix A.1, p. 37):
  ```
  O := P(D = 1) / P(D = 0) = p / (1 - p),
  O_{XU} := P(D = 1 | X, U) / P(D = 0 | X, U),
  O_X := P(D = 1 | X) / P(D = 0 | X) = π / (1 - π),      p = P(D = 1),  π = P(D = 1 | X).
  ```
- **Lemma 1** (p. 14): under Assumption 1,
  ```
  θ_0 = E[ g_0 (O_{XU} / O) | D = 0 ],      θ_{0s} = E[ g_{0s} (O_X / O) | D = 0 ].
  ```
  "Moreover, `g_{0s} = E[g_0 | D = 0, X]` and `O_X = E[O_{XU} | D = 0, X]`." Appendix B
  (p. 45) names the weights `α_0 := O_{XU}/O`, `α_{0s} := O_X/O` and states
  `E[α_0 | D = 0] = E[α_{0s} | D = 0] = 1`. Remark 6 (p. 65) identifies `α_0` as the Riesz
  representer of `θ_0(g_0)` on `L²(P_{(X,U)|D=0})`.

*OVB formula (Theorem 1, p. 14; proof pp. 45-48):*

"Under the regularity conditions of Assumption 1, the OVB is given by the (negative of the)
covariance of errors induced by the omission of U both in the regression function and in the
odds ratio, among the untreated units,"

```
θ - θ_s = - Cov( g_0 - g_{0s},  O_{XU}/O - O_X/O  |  D = 0 ).

θ - θ_s = -   ρ_0     C_{0ΔY}   C_{0D}    S_0 ,
           (alignment) (trend) (selection) (scale)
```

where

```
ρ_0       := Cor( g_0 - g_{0s},  O_{XU} - O_X  |  D = 0 ),
C²_{0ΔY}  := η²_{ΔY~U|X,D=0},
C²_{0D}   := (1 - R²_{O_XU~O_X|D=0}) / R²_{O_XU~O_X|D=0},
S²_0      := σ²_{0s} ν²_{0s},
σ²_{0s}   := E[ Var(ΔY | X, D = 0) | D = 0 ],
ν²_{0s}   := E[ (O_X / O)² | D = 0 ].
```

- **This is an exact signed equality, not a bound.** Only the squares `C²_{0ΔY}`,
  `C²_{0D}`, `S²_0` are defined; the paper does not say in words that the un-squared factors
  are the non-negative roots, but the proof defines them through
  `|θ - θ_s|² = ρ²_0 C²_{0ΔY} C²_{0D} S²_0` (p. 46) and every reported value is positive
  with the sign carried by `ρ_0` (p. 9, Table 7).
- **Ratio forms from the proof** (pp. 46-48), all over the untreated group:
  ```
  C²_{0ΔY} = E[(g_0 - g_{0s})² | D = 0] / E[(ΔY - g_{0s})² | D = 0]
           = R²_{ΔY-g_0s ~ g_0-g_0s | D=0} = η²_{ΔY~U|X,D=0},

  C²_{0D}  = E[(α_0 - α_{0s})² | D = 0] / E[α²_{0s} | D = 0],

  R²_{O_XU~O_X|D=0} = E[O²_X | D = 0] / E[O²_{XU} | D = 0] = R²_{α_0~α_0s|D=0},

  σ²_{0s}  = E[(ΔY - g_{0s})² | D = 0] = Var(ΔY - g_{0s} | D = 0),

  ν²_{0s}  = E[α²_{0s} | D = 0].
  ```
- **The selection R² is uncentered.** `R²_{Y~X}` denotes "the (possibly uncentered) R² from
  the orthogonal linear projection" (p. 6; Appendix A.2, p. 38:
  `R²_{Y~X} := E[(Ŷ^X)²]/E[Y²]`), and the proof uses second moments, not variances (p. 47).
  The paper calls this "a non-centered parameterization of selection strength, which
  naturally avoids a zero-denominator issue" (p. 6). The centered analogue is written with
  an explicit "1" in the conditioning set, e.g. `R²_{O_XU~O_X|1,D=0}` (p. 86). The
  nonparametric partial `η²` "is automatically centered" (p. 39).
- **Ranges** (p. 15): the trend component "can always be left unconstrained by the
  investigator, as it is naturally upper bounded by 1"; the alignment component's
  "magnitude is upper bounded by 1"; `1 - R²_{O_XU~O_X|D=0}` "is also an R² which can be
  upper bounded by 1", but the ratio `C²_{0D}` "can be arbitrarily large. Therefore, the
  treatment selection component must always be restricted." The two sensitivity parameters
  are `η²_{ΔY~U|X,D=0}` and `1 - R²_{O_XU~O_X|D=0}` (p. 15).
- "Note that bias arises only when all bias factors are non-zero." (p. 15). Degenerate case
  (p. 102): "We implicitly assume that `Var(g_0 - g_{0s} | D = 0)`,
  `Var(α_0 - α_{0s} | D = 0) > 0`; otherwise, the bias is zero."
- Worst-case bound implied by Theorem 1 (stated in Section 3.4, p. 19):
  `|θ - θ_s| ≤ C_{0D} S_0`.
- Scale interpretation (pp. 15-16): `σ²_{0s}` "measures the leftover variation of the trend
  in the control group after taking into account the part explained by X"; `ν²_{0s}`
  "measures how well observed covariates separate treated from control units. The stronger
  this separation, the less identifying variation is being used to estimate the ATT, and the
  more leverage unobserved confounding has to drive the bias".
- **[Reviewer check - not in paper]** With an empty covariate set, `O_X = O`, so
  `ν²_{0s} = 1` and `S_0 = sd(ΔY | D = 0)`; Table 7's note prints exactly this case
  (`ν̂²_{0s,∅} = 1`, `S_{0,∅} = 0.154`, p. 109).

*Selection-strength interpretations (Section 3.3, pp. 16-18; Appendix A.3, pp. 40-43):*

- **Lemma 2** (p. 17), under weak overlap:
  ```
  (i)   O_{XU} / O = dP_{X,U|D=1} / dP_{X,U|D=0}.
  (ii)  E[O²_{XU} | X, D = 0] = O_X E[O_{XU} | X, D = 1].
  (iii) χ²(P_{X,U|D=1} || P_{X,U|D=0}) = E[O_{XU}/O | D = 1] - 1.
  ```
- **Corollary 1** (p. 17):
  ```
  1. Increase in average treatment odds among the treated:
     C²_{0D} = ( E[O_{XU} | D = 1] - E[O_X | D = 1] ) / E[O_X | D = 1],
     ν²_{0s} = E[O_X | D = 1] / O.

  2. Covariate imbalance between treated and control units:
     C²_{0D} = ( χ²(P_{X,U|D=1} || P_{X,U|D=0}) - χ²(P_{X|D=1} || P_{X|D=0}) )
               / ( χ²(P_{X|D=1} || P_{X|D=0}) + 1 ),
     ν²_{0s} = χ²(P_{X|D=1} || P_{X|D=0}) + 1.
  ```
  So `ν²_{0s}` has two estimable representations: `E[(O_X/O)² | D = 0]` (controls) and
  `E[O_X/O | D = 1]` (treated). Equation 11 uses both.
- Worked mappings: if `U` "at most doubles the average odds of treatment, among treated
  units, then Corollary 1 implies `C²_{0D} ≤ 1`" (p. 17). Under the marginal sensitivity
  model of Tan (2006), `Λ^{-1} ≤ O_{XU}/O_X ≤ Λ` a.s. for some `Λ > 1`, "the sharp bound"
  is `C²_{0D} ≤ (Λ - 1)²/Λ` (pp. 17-18, Corollary 2); `Λ = 2` gives `C²_{0D} ≤ 1/2`.
- Equivalence used throughout Appendix D (p. 63): for `0 ≤ k < 1`,
  ```
  1 - R²_{O_XU~O_X|D=0} = k   <=>   C²_{0D} = k / (1 - k).
  ```
- "the imbalance measure directly relevant for quantifying the bias of the ATT is the
  χ²-divergence" rather than the standardized mean difference (p. 18).
- Supporting identities (**Proposition A.2**, p. 40; **Proposition A.3**, pp. 41-42):
  ```
  O_{XU}/O_X = dP_{U|X,D=1}/dP_{U|X,D=0};            O_X = E[O_{XU} | X, D = 0];
  E[O²_{XU} | X, D = 0] = O_X × E[O_{XU} | X, D = 1];
  E[O_{XU}] = p × ( E[O_{XU} | D = 1] + 1 ) = (1 - p) E[O²_{XU} | D = 0] + p;
  χ²(P_{X,U|D=1} || P_{X,U|D=0}) = Var( O_{XU}/O | D = 0 ) = E[O_{XU}/O | D = 1] - 1.
  ```

*Alternative parameterizations (Section 3.4, pp. 18-20; Appendix F, pp. 79-106):*

The paper's "preferred conditional parameterization" (p. 5) is Theorem 1. Two alternatives
target the same estimand and the same total bias:

- **Unconditional** (Chernozhukov et al. 2026 applied to the ATT; Equation 7, p. 18):
  ```
  θ - θ_s = ρ C_{ΔY} C_D S,                                                  (7)
  C²_{ΔY} := η²_{ΔY~U|X,D},   C²_D := ( E[O_{XU}] - E[O_X] ) / E[O_X],
  S² := E[Var(ΔY | X, D)] E[O_X] / P(D = 1)²,
  ρ := Cor( E[ΔY | D, X, U] - E[ΔY | D, X],  -(1 - D)(O_{XU} - O_X) ).
  ```
  Equation 7 has no leading minus sign (it is inside `ρ`). **Proposition 1** (p. 19):
  ```
  ρ² C²_{ΔY} = ( P(D = 0) σ²_{0s} / E[Var(ΔY | X, D)] ) ρ²_0 C²_{0ΔY}
             ≤  P(D = 0) σ²_{0s} / E[Var(ΔY | X, D)],
  ```
  so bounding `|ρ|` and `C_{ΔY}` separately by one gives the looser `|θ - θ_s| ≤ C_D S`,
  while the joint restriction recovers `C_{0D} S_0`. **Proposition F.2** (pp. 80-81):
  `C²_D = W_{0D} C²_{0D}` with
  `W_{0D} = O × (χ²(P_{X|1} || P_{X|0}) + 1) / (O × (χ²(P_{X|1} || P_{X|0}) + 1) + 1)`, and analogous weighted
  forms for `C²_{ΔY}` and `ρ²`. **Proposition F.1** (p. 80):
  `C²_D = (χ²_{XU} - χ²_X) / (χ²_X + 1/p)`.
- **Centered** (Huang and Pimentel 2025, "HP"; Equation 8, p. 19; Appendix F.2):
  ```
  θ - θ_s = - ρ_{w0} C_{w0D} S_{w0},                                         (8)
  ρ_{w0} := Cor( ΔY, O_{XU} - O_X | D = 0 ),
  C²_{w0D} := ( E[O_{XU}] - E[O_X] ) / ( E[O_X] - O ),
  S²_{w0} := Var(ΔY | D = 0) × P(D = 0) { E[O_X] - O } / P(D = 1)².
  ```
  "If X does not predict treatment, which includes the canonical 2x2 DiD setup without
  covariates, `C_{w0D}` has a zero denominator, and the decomposition (8) is not well
  defined" (p. 19). **Proposition 2** (p. 20):
  `|ρ_{w0}| ≤ sqrt(σ²_{0s} / Var(ΔY | D = 0)) ≤ ρ̄_{w0}`, with
  `ρ̄_{w0} := sqrt(1 - Cor²(O_X, ΔY | D = 0))` HP's estimable bound. **Proposition F.4**
  (p. 87): `C²_{w0D} = C²_{0D} × E[O_X | D = 1] / (E[O_X | D = 1] - O)` and
  `ρ²_{w0} = ρ²_0 × C²_{0ΔY} × (1 - R²_{ΔY~g_0s|1,D=0})`. Remark 7 (p. 86):
  `ν²_{w0s} = χ²(P_{X|1} || P_{X|0}) = ν²_{0s} - 1`, and estimation noise "can lead to
  negative estimates of `ν²_{w0s}`. This issue is mitigated in the uncentered
  parameterization." Section F.2.5 (pp. 91-92) gives a DML score for the centered scale.
- **Corollary 2** (p. 20), under the marginal sensitivity model:
  ```
  C²_D ≤ ((E[O_X] - p)/E[O_X]) (Λ - 1)²/Λ,   C²_{0D} ≤ (Λ - 1)²/Λ,
  C²_{w0D} ≤ ((E[O_X] - p)/(E[O_X] - O)) (Λ - 1)²/Λ.
  ```
  "All three bounds are sharp"; only the `C²_{0D}` bound "is invariant to the observed
  treatment assignment".
- **Statistics are not comparable across parameterizations** (p. 94): "sensitivity
  statistics should only be compared within the same parameterization." In the simulation
  design the population RV / XRV are 0.4228 / 0.2365 (this paper), 0.7446 / 0.6848
  (centered), and, for `p ∈ {0.2, 0.5, 0.8}`, {0.2175, 0.2731, 0.2360} /
  {0.0570, 0.0931, 0.0679} (unconditional) (pp. 84, 94).
- **Benchmarking differs from HP's** (Proposition F.6 and the display that follows it, p. 89):
  `k_{0D,j} = k^{HP}_{0D,j} × R²_{O_XU~O_{X_-j}|D=0}`, "which shows that, for the same value
  of k, their benchmarking exercise will always posit a weaker confounder than ours."
  (pp. 89-90)

*Bounds under direct restrictions (Section 5.1, p. 25):*

"Given plausibility judgments on the magnitude of sensitivity parameters `|ρ_0|`,
`C_{0ΔY}`, `C_{0D}`, we have the following bounds on θ:"

```
θ_± = θ_s ± |ρ_0| C_{0ΔY} C_{0D} S_0,     with   S²_0 = σ²_{0s} ν²_{0s}.
```

On the two R²-scale sensitivity parameters (`η² := η²_{ΔY~U|X,D=0}`,
`1 - R² := 1 - R²_{O_XU~O_X|D=0}`; shorthand of p. 60) the multiplier is
`|ρ_0| × sqrt(η²) × sqrt((1 - R²)/R²)`. The proof of Corollary 4 (p. 62) uses exactly this
at `(ρ²_0, η², 1 - R²) = (1, r, r)`: `θ* = θ_s - S_0 r / sqrt(1 - r)`.

*Estimation (Section 5.1, Equations 9-11 and Algorithm 1, pp. 25-26):*

"We estimate these components using debiased machine learning (DML), which combines Neyman
orthogonal scores with cross-fitting". Let `Z := (ΔY, D, X)`, `p := P(D = 1)`,
`π := P(D = 1 | X)`. Orthogonal scores:

```
ψ_{θ_s}(Z; g_{0s}, π, p)  = ( D/p - ((1 - D)/(1 - p)) (O_X/O) ) (ΔY - g_{0s}) - D θ_s / p,        (9)

ψ_{σ²_0s}(Z; g_{0s}, p)   = ((1 - D)/(1 - p)) ( (ΔY - g_{0s})² - σ²_{0s} ),                        (10)

ψ_{ν²_0s}(Z; π, p)        = 2 (D/p) ( O_X/O - ν²_{0s} ) - ((1 - D)/(1 - p)) ( (O_X/O)² - ν²_{0s} ). (11)
```

```
θ̂_s := DML(ψ_{θ_s}),      σ̂²_{0s} := DML(ψ_{σ²_0s}),      ν̂²_{0s} := DML(ψ_{ν²_0s}).
```

"Let `β ∈ {θ_s, σ²_{0s}, ν²_{0s}}` denote a generic target parameter and
`η = (g_{0s}, π, p)` the vector of nuisance parameters. The scores (9)-(11) admit the
representation `ψ_β(Z; η) = ψ^a_β(Z; η) β + ψ^b_β(Z; η)`." (p. 25)

- **Equation 11 is a debiased score, not the control-arm plug-in.** It combines the treated
  representation `E[O_X/O | D = 1]` (weight 2) and the control representation
  `E[(O_X/O)² | D = 0]` (weight -1). Appendix B (pp. 51-54) verifies Neyman orthogonality of
  all three scores with respect to `g_{0s}`, `π` and `p`. The plug-in mean of `(O_X/O)²`
  over controls alone is not orthogonal to `π`.
- **Remark 2** (p. 26): the DML estimator of `ν²_{0s}` "can be used to estimate the
  chi-squared divergence of observed covariates, since
  `ν²_{0s} - 1 = χ²(P_{X|D=1} || P_{X|D=0})`".
- `g_{0s} = E[ΔY | D = 0, X]` is a regression among the untreated by definition (p. 13);
  the only nuisances are `g_{0s}`, `π`, `p`. There is no separately learned Riesz
  representer: `O_X/O` is a function of `(π, p)`.
- **[Reviewer check - not in paper]** Reading off the coefficient on `β`:
  `ψ^a_{θ_s} = -D/p`, `ψ^a_{σ²_0s} = -(1 - D)/(1 - p)`,
  `ψ^a_{ν²_0s} = -2D/p + (1 - D)/(1 - p)`; each has expectation `-1`. Solving the pooled
  moment equation of Algorithm 1 therefore gives ratio estimators:
  ```
  θ̂_s    = Σ_i ( D_i/p̂ - ((1 - D_i)/(1 - p̂)) α̂_i ) (ΔY_i - ĝ_i)  /  Σ_i D_i/p̂,
  σ̂²_0s  = Σ_i ((1 - D_i)/(1 - p̂)) (ΔY_i - ĝ_i)²                  /  Σ_i (1 - D_i)/(1 - p̂),
  ν̂²_0s  = Σ_i ( 2 (D_i/p̂) α̂_i - ((1 - D_i)/(1 - p̂)) α̂_i² )      /  Σ_i ( 2 D_i/p̂ - (1 - D_i)/(1 - p̂) ),
  ```
  with `α̂_i = (π̂_i/(1 - π̂_i)) ((1 - p̂)/p̂)` and nuisances evaluated at the fold that
  holds out unit `i`. The paper writes "the solution of 0 = ..." and does not print these.

**Algorithm 1** (p. 26), transcribed exactly:

```
Algorithm 1  DML for estimable components: DML(ψ_β)

Input: The Neyman orthogonal score ψ_β(Z; η), sample {Z_i}_{i=1}^n, and number of folds L.
Sample Splitting: Randomly partition the sample into L folds of approximately equal size.
   Denote each fold by I_l and its complement by I_l^c, for l = 1, ..., L.
for l = 1, ..., L do
    Estimate the nuisance parameters using observations in I_l^c. Denote the estimates by η̂_l.
end for
Estimate: Construct the estimator β̂ as the solution of
              0 = (1/n) Σ_{l=1}^{L} Σ_{i ∈ I_l} ψ_β̂(Z_i; η̂_l).
Return: The estimate β̂ and the estimated scores ψ_β̂(Z_i; η̂_l) for each i ∈ I_l and each l.
```

- One pooled moment equation across folds. The paper does not use the labels DML1 / DML2
  (zero hits).
- `p` is listed inside `η`, so as literally written it is estimated on `I_l^c` per fold.
  The paper does not single it out for full-sample estimation.
- **Number of folds.** Algorithm 1 takes `L` as an input with no value. The only values are
  in Appendix E: "ten-fold cross-fitting" and footnote 2, "We use more than the default five
  cross-fitting folds to stabilize estimation of `ν²_{0s}`" (p. 66). The fold count of the
  application is not stated.
- No repeated cross-fitting or aggregation across splits is described (`median` hits only
  "median income", pp. 32, 108, 109; `seed` has zero hits).
- The paper does not say whether the three `DML(·)` calls share one fold partition and one
  set of nuisance fits. The delta method of Theorem 2 needs their influence functions
  aligned observation by observation.

*Standard errors (Lemma 3 and Theorem 2, pp. 26-27; proofs pp. 50-54):*

- **Lemma 3** (p. 26): "Suppose that Assumptions 3.1 and 3.2 from Chernozhukov et al. (2018)
  hold for each of the scores `ψ_β` above, and for the estimators of the nuisance parameters
  `η̂_l`. Then, the DML estimators `β̂` are asymptotically linear and Gaussian":
  ```
  sqrt(n)(β̂ - β) = (1/sqrt(n)) Σ_i ψ_β(Z_i; η) / ( -E[ψ^a_β(Z; η)] ) + o_p(1)  →_d  N(0, σ²_{φ_β}),

  φ^0_β(Z_i) := ψ_β(Z_i; η) / ( -E[ψ^a_β(Z; η)] ),      σ²_{φ_β} := E[(φ^0_β(Z))²],

  φ̂_β̂(·) := - ψ_β̂(·; η̂_l) / E_n[ψ^a(Z; η̂_l)],        σ̂²_{φ_β} := E_n[(φ̂_β̂(Z))²].
  ```
  The regularity and rate conditions are imported by reference and not restated.
- Plug-in bounds (p. 26): `θ̂_± := θ̂_s ± |ρ_0| C_{0ΔY} C_{0D} sqrt(σ̂²_{0s} ν̂²_{0s})`.
- **Theorem 2 (Confidence intervals for the bounds)** (pp. 26-27): under the assumptions of
  Lemma 3,
  ```
  sqrt(n)(θ̂_± - θ_±) = (1/sqrt(n)) Σ_i φ^0_{θ_±}(Z_i) + o_p(1)  →_d  N(0, σ²_{φ_θ±}),

  φ^0_{θ_±}(Z) = φ^0_{θ_s}(Z) ± |ρ_0| C_{0ΔY} C_{0D} × φ^0_{S_0}(Z),

  φ^0_{S_0}(Z) := (1/(2 S_0)) ( σ²_{0s} φ^0_{ν²_0s}(Z) + ν²_{0s} φ^0_{σ²_0s}(Z) ).
  ```
  "Let Φ denote the CDF of the standard normal and α denote the significance level. The
  confidence interval"
  ```
  [l, u] = [ θ̂_- - Φ^{-1}(1 - α) sqrt( E[(φ^0_{θ_-}(Z))²] / n ),
             θ̂_+ + Φ^{-1}(1 - α) sqrt( E[(φ^0_{θ_+}(Z))²] / n ) ],
  ```
  "satisfies `P(θ_- ∈ [l, ∞)) → 1 - α` and `P(θ_+ ∈ (-∞, u]) → 1 - α`. These results
  continue to hold if we replace `E[(φ^0_{θ_±}(Z))²]` with `E_n[(φ̂_θ̂±(Z))²]`."
- The lower limit uses the standard error of `θ̂_-` and the upper limit that of `θ̂_+`;
  they differ because `φ^0_{S_0}` enters with opposite signs. Coverage is stated one-sided
  for each bound separately; Theorem 2 states no joint coverage for `θ`.
- The variance divides `E_n[φ̂²]` by `n`; no finite-sample (`n - 1`) correction and no
  t-distribution appear (zero hits for `degrees of freedom`, `student`).
- **Critical-value convention - internal tension in v1.** The paper prints three things
  that do not obviously agree:
  1. Theorem 2 and `CI^max` (p. 60) use `Φ^{-1}(1 - α)` at each endpoint; Tables 9, 10, 12
     are captioned "One-sided 95% confidence intervals" (pp. 110, 113).
  2. Definitions 3-4 (p. 61) set RV = XRV = 0 when "the two sided 1 - α confidence interval
     from the original analysis under no confounding contains θ*", and say the
     point-estimate versions are obtained "by setting the critical value to zero. We denoted
     them by `RV_{θ*,α=1}(θ)` and `XRV_{θ*,α=1}(θ)`."
  3. Table 1 (p. 30) reports -0.0366, SE 0.0105, "Confidence Interval" [-0.0572; -0.0160],
     and Figure 2 labels the zero-confounding point "Observed (-0.016)" with the text
     "corresponds to the upper bound of the confidence interval reported in Table 1" (p. 31).

  **[Reviewer check - not in paper]** `Φ^{-1}(1 - α)` at `α = 1` is `-∞`, not zero; a zero
  critical value at `α = 1` corresponds to `Φ^{-1}(1 - α/2)`. Table 1's interval equals
  `-0.0366 ± 1.960 × 0.0105 = [-0.0572, -0.0160]` (with 1.645 the upper limit would be
  -0.0193); Table 11's equals `-0.074 ± 1.960 × 0.0109`. Fitting
  `label = θ̂_s + |ρ_0| S_0 sqrt(G_{0ΔY,j}) sqrt(G_{0D,j}/(1 - G_{0D,j})) + c` to the eight
  benchmark labels of Figure 2 with the Table 8 gains gives `S_0 ≈ 0.436`, `c ≈ 0.0172`
  (maximum residual 1.3e-4), i.e. `c / 0.0105 ≈ 1.64` - although the standard error of the
  bound also varies across the surface, so this is indicative only. The dump has zero hits
  for `α/2`, `1.96`, `1.645`, `0.975`. An implementation must choose and document a
  convention; the paper does not resolve it.

*Sensitivity statistics (Section 5.3, pp. 28-29; Appendix D, pp. 59-64):*

- **Compatible confidence interval** (D.1, pp. 59-60). With `η²` and `1 - R²` as above:
  ```
  CI^max_{1-α, ρ̄², R̄²_ΔY, R̄²_D}(θ) = [L^min, U^max],

  L^min = min ( θ̂_-(ρ²_0, η², R²) - Φ^{-1}(1 - α) sqrt( E[(φ^0_{θ_-}(Z; ρ²_0, η², R²))²] / n ) ),
  U^max = max ( θ̂_+(ρ²_0, η², R²) + Φ^{-1}(1 - α) sqrt( E[(φ^0_{θ_+}(Z; ρ²_0, η², R²))²] / n ) ),

  each subject to   ρ²_0 ≤ ρ̄²,   η² ≤ R̄²_ΔY,   1 - R² ≤ R̄²_D.
  ```
  "When `ρ̄² = 1`, we omit it from the subscript." **Corollary 3** (p. 60): the endpoints
  "have asymptotic one-sided coverage of at least 1 - α whenever the stated restrictions
  hold, and the interval is nested as any of the three upper bounds increases."
- The paper gives no computational method for the min / max (zero hits for `bisect`,
  `monoton`; `grid` hits only the learner tuning grids, pp. 75, 107).
  **[Reviewer check - not in paper]** Write the multiplier `m = |ρ_0| C_{0ΔY} C_{0D}`, which
  is increasing in each of `ρ²_0`, `η²`, `1 - R²`. Then
  `U(m) = θ̂_s + m Ŝ_0 + z × se_+(m)` with
  `se_+(m)² = E_n[(φ̂_{θ_s} + m φ̂_{S_0})²]/n`. By Cauchy-Schwarz
  `|d se_+/dm| ≤ se(Ŝ_0)`, so `dU/dm ≥ Ŝ_0 - z × se(Ŝ_0)`: the upper limit is increasing in
  `m` (and the optimum is at the corner of the constraint set) whenever `Ŝ_0 > z se(Ŝ_0)`,
  and symmetrically for the lower limit. This condition is sufficient, not necessary;
  when it fails this argument does not establish the corner, and monotonicity should be
  checked rather than assumed.
- **Extreme robustness value** (p. 28; Definition 4, p. 61): "leaving alignment and trend
  entirely unrestricted, what is the minimum strength of confounding with treatment
  selection alone that would lead one to not reject `H_0 : θ = θ*`?"
  ```
  XRV_{θ*,α}(θ) := inf { XRV : θ* ∈ CI^max_{1-α, 1, XRV}(θ) }.
  ```
- **Robustness value** (p. 29; Definition 3, p. 61): "the minimum strength of confounding in
  explaining both the untreated outcome trend and treatment selection jointly."
  ```
  RV_{θ*,α}(θ) := inf { RV : θ* ∈ CI^max_{1-α, RV, RV}(θ) }.
  ```
  In both, `|ρ_0| ≤ 1` is unrestricted. Both are defined "for fixed θ* and α ∈ (0, 0.5)" and
  are set to zero when the two-sided no-confounding interval contains `θ*` (Definitions 3-4).
- **Population values** (p. 62). "let `f_{θ*} := |θ* - θ_s| / S_0`":
  ```
  RV_{θ*}(θ) = (1/2) ( sqrt( f⁴_{θ*} + 4 f²_{θ*} ) - f²_{θ*} ),      XRV_{θ*}(θ) = f²_{θ*} / (1 + f²_{θ*}).
  ```
  No closed form is printed for the finite-sample `α` versions, whose definition runs
  through `CI^max` with a standard error that depends on the sensitivity parameters.
- **Corollary 4** (p. 62): "Let `f_{θ*} > 0`. Under the conditions of Theorem 2,
  `RV_{θ*,α}(θ)` and `XRV_{θ*,α}(θ)` are asymptotically valid 1 - α lower confidence bounds
  for their population counterparts." The two coverage events coincide (p. 63):
  `RV_{θ*,α}(θ) ≤ RV_{θ*}(θ) <=> XRV_{θ*,α}(θ) ≤ XRV_{θ*}(θ)`, so "all simulation results
  in this appendix report a single common (X)RV empirical coverage."
- **Interpretation** (D.2.3, pp. 63-64):
  - "If k is the value of `XRV_{0,α}`: if unobserved confounders increase treatment odds or
    covariate imbalance by less than `k/(1-k)` then such confounders are not capable of
    overturning the original results, at the significance level of α, *regardless* of how
    much variation such confounders explain of the outcome trend."
  - "If k is the value of `RV_{0,α}(θ)`: unobserved confounders explaining less than k of the
    residual variation in untreated trend and inducing less than `k/(1-k)` increase in
    average odds or covariate imbalance cannot overturn the conclusions of the study, at the
    significance level of α."
- **[Reviewer check - not in paper]** The population formulas reproduce every printed
  population value: with the Appendix E design (`θ_s = 1.7`, `S²_0 = 8 × 1.16631`),
  `f² = 0.30974`, RV = 0.42282, XRV = 0.23649 (paper: 0.4228, 0.2365). `RV = r` solves
  `r²/(1 - r) = f²`, consistent with the `S_0 r / sqrt(1 - r)` display of p. 62. In Table 1,
  RV = 4.38% and XRV = 0.20% imply the same total bias-to-scale ratio
  (`0.0438/sqrt(1 - 0.0438) = 0.04479`, `sqrt(0.0020/0.9980) = 0.04477`), and
  `0.0438/(1 - 0.0438) = 0.0458` matches the text's "4.58% relative increase in the observed
  average treatment odds" (p. 30).

*Benchmarking against observed covariates (Section 4.1, pp. 21-22; Section 5.2, p. 27; Appendix C.1, pp. 54-58):*

Let `X_j` be "a subvector of the observed covariates X" used as the benchmark and `X_{-j}`
the rest (p. 54). Leave-out objects (pp. 54-55):

```
g_{0s,-j} := E[ΔY | X_{-j}, D = 0],     α_{0s,-j} = O_{X_-j} / O,
θ_{0s,-j} := E[g_{0s,-j} | D = 1],      θ_{s,-j} := E[ΔY | D = 1] - θ_{0s,-j},
σ²_{0s,-j} := E[ Var(ΔY | X_{-j}, D = 0) | D = 0 ],
ν²_{0s,-j} := E[ α²_{0s,-j} | D = 0 ] = E[ α_{0s,-j} | D = 1 ].
```

- **Definition 1** (p. 55) and **Definition 2** (p. 56), also p. 21:
  ```
  k_{0D,j}  := ( R²_{O_X~O_{X_-j}|D=0} - R²_{O_XU~O_{X_-j}|D=0} ) / ( 1 - R²_{O_X~O_{X_-j}|D=0} ),

  k_{0ΔY,j} := ( η²_{ΔY~U,X_j|X_-j,D=0} - η²_{ΔY~X_j|X_-j,D=0} ) / η²_{ΔY~X_j|X_-j,D=0}.
  ```
  "`k_{0ΔY,j} = 1` implies that further adding U to the trend regression would lead to
  similar additive gains in explanatory power to those observed by adding `X_j`" (p. 21).
- **Proposition C.1** (p. 55) and **Remark 4** (p. 56):
  ```
  1 - R²_{O_XU~O_X|D=0} = k_{0D,j} C²_{0D,j},
  C²_{0D,j} := (1 - R²_{O_X~O_{X_-j}|D=0}) / R²_{O_X~O_{X_-j}|D=0},
  R²_{O_X~O_{X_-j}|D=0} = ν²_{0s,-j} / ν²_{0s},
  1 - R²_{O_XU~O_X|D=0} = k_{0D,j} ( ν²_{0s} - ν²_{0s,-j} ) / ν²_{0s,-j},
  k_{0D,j} ≤ 1 / C²_{0D,j}.
  ```
- **Proposition C.2** (p. 56) and **Remark 5** (pp. 56-57):
  ```
  η²_{ΔY~U|X,D=0} = k_{0ΔY,j} C²_{0ΔY,j} / (1 - C²_{0ΔY,j}),     C²_{0ΔY,j} := η²_{ΔY~X_j|X_-j,D=0},
  σ²_{0s,-j} - σ²_{0s} = E[(g_{0s} - g_{0s,-j})² | D = 0],
  η²_{ΔY~U|X,D=0} = k_{0ΔY,j} ( σ²_{0s,-j} - σ²_{0s} ) / σ²_{0s},
  k_{0ΔY,j} ≤ (1 - C²_{0ΔY,j}) / C²_{0ΔY,j}.
  ```
- **Alignment** (p. 22; C.1.4, p. 57):
  ```
  ρ_{0,j} := Cor( g_{0s} - g_{0s,-j},  O_X - O_{X_-j}  |  D = 0 ),
  θ_s - θ_{s,-j} = - ρ_{0,j} sqrt( (σ²_{0s,-j} - σ²_{0s}) (ν²_{0s} - ν²_{0s,-j}) ),
  ρ_{0,j} = - (θ_s - θ_{s,-j}) / sqrt( (σ²_{0s,-j} - σ²_{0s}) (ν²_{0s} - ν²_{0s,-j}) ).
  ```
  "`ρ_0` is not a measure of explanatory power of the confounders. Rather, it measures how
  systematically U shifts the odds of treatment and outcome evolution in the same
  direction" (pp. 21-22). The benchmark rule is "setting `|ρ_0| = |ρ_{0,j}|`" (p. 27); the
  application also fixes `|ρ_0|` at 0.3 and 1 (p. 110).
- **Gain metrics** (C.1.5, p. 58) and **plug-in bound** (Section 5.2, p. 27):
  ```
  G_{0D,j}  := C²_{0D,j} = ( ν²_{0s} - ν²_{0s,-j} ) / ν²_{0s,-j},
  G_{0ΔY,j} := C²_{0ΔY,j} / (1 - C²_{0ΔY,j}) = ( σ²_{0s,-j} - σ²_{0s} ) / σ²_{0s},

  θ̂_± = θ̂_s ± |ρ̂_{0,j}| × sqrt( k_{0ΔY,j} Ĝ_{0ΔY,j} × k_{0D,j} Ĝ_{0D,j} / (1 - k_{0D,j} Ĝ_{0D,j}) )
              × sqrt( σ̂²_{0s} ν̂²_{0s} ).
  ```
  The selection gain divides by the leave-out `ν²_{0s,-j}`; the trend gain divides by the
  full-covariate `σ²_{0s}`. "The orthogonal scores for these parameters have the same form
  as (9)-(11), with the only difference that `X_{-j}` is used for estimation in place of X."
  (p. 27)
- **Inference** (C.1.5, p. 58). "we assume `G_{0ΔY,j} > 0`, `G_{0D,j} > 0`, `ρ_{0,j} ≠ 0`,
  and `0 < k_{0D,j} G_{0D,j} < 1`." Delta-method influence functions:
  ```
  φ^0_{G_0D,j}(Z)  = ( ν²_{0s,-j} φ^0_{ν²_0s}(Z) - ν²_{0s} φ^0_{ν²_0s,-j}(Z) ) / ν⁴_{0s,-j},

  φ^0_{G_0ΔY,j}(Z) = ( σ²_{0s} φ^0_{σ²_0s,-j}(Z) - σ²_{0s,-j} φ^0_{σ²_0s}(Z) ) / σ⁴_{0s},

  φ^0_{ρ_0,j}(Z)   =   ( φ^0_{θ_s,-j}(Z) - φ^0_{θ_s}(Z) )
                         / ( (σ²_{0s,-j} - σ²_{0s})^{1/2} (ν²_{0s} - ν²_{0s,-j})^{1/2} )
                     - (θ_{s,-j} - θ_s) ( φ^0_{σ²_0s,-j}(Z) - φ^0_{σ²_0s}(Z) )
                         / ( 2 (σ²_{0s,-j} - σ²_{0s})^{3/2} (ν²_{0s} - ν²_{0s,-j})^{1/2} )
                     - (θ_{s,-j} - θ_s) ( φ^0_{ν²_0s}(Z) - φ^0_{ν²_0s,-j}(Z) )
                         / ( 2 (σ²_{0s,-j} - σ²_{0s})^{1/2} (ν²_{0s} - ν²_{0s,-j})^{3/2} ).
  ```
  **The influence function of the covariate-benchmarked bound `θ̂_±` itself is not printed.**
  "The plug-in estimator of the bias and the corresponding influence function for `θ_±`
  follow directly from the discussion in Appendix E.6 of Chernozhukov et al. (2026), after
  replacing the 'unconditional' components with their corresponding 'conditional'
  counterparts in our results." (p. 58). The final delta-method assembly, including the
  derivative through `|ρ̂_{0,j}|`, must be derived or sourced from that paper.
- The paper does not say what to do when an estimated difference
  `σ̂²_{0s,-j} - σ̂²_{0s}` or `ν̂²_{0s} - ν̂²_{0s,-j}` is zero or negative (both sit under
  square roots), beyond the positivity assumptions above. The one cap it reports is in the
  Table 7 caption (p. 109): "If the plug-in estimate of `ρ_{0,j}` exceeds one in absolute
  value, it is capped at ±1".

*Benchmarking against a pre-treatment period (Section 4.2, pp. 22-24; Section 5.2, pp. 27-28; Appendix C.2, pp. 58-59):*

"Consider three time periods `t ∈ {0, 1, 2}`, with treatment occurring after `t = 1`."
`ΔY^pre := Y_1 - Y_0`; `θ^pre`, `θ_s^pre ≠ 0`, `ρ_0^pre`, `C^pre_{0ΔY}`, `C^pre_{0D}`,
`S_0^pre = σ^pre_{0s} ν^pre_{0s}` are the Section 3 objects "computed with
`(ΔY^pre, X^pre, U^pre)` in place of `(ΔY, X, U)`, with D unchanged throughout" (pp. 22-23).
Since `θ^pre = 0`, `|ρ_0^pre| C^pre_{0ΔY} C^pre_{0D} S_0^pre = |θ_s^pre|`. "We consider the
simplest case of one additional pre-treatment period, and leave extensions to multiple time
periods to future work." (p. 22)

```
Extrapolating the bias magnitude:   θ_± = θ_s ± k |θ_s^pre|,                   k = k_ρ k_ΔY k_D k_S,
Extrapolating the bias factors:     θ_± = θ_s ± k (S_0 / S_0^pre) |θ_s^pre|,   k = k_ρ k_ΔY k_D,

k_ρ := |ρ_0|/|ρ_0^pre|,  k_ΔY := C_{0ΔY}/C^pre_{0ΔY},  k_D := C_{0D}/C^pre_{0D},  k_S := S_0/S_0^pre.
```

Note `k` means different things in the two approaches. Influence functions (C.2, p. 59),
with `Z := (ΔY, D, X, ΔY^pre, X^pre)`:

```
First approach:    φ^0_{θ_±}(Z) := φ^0_{θ_s}(Z) ± k sign(θ_s^pre) φ^0_{θ_s^pre}(Z).

Second approach (additionally assuming S_0 > 0 and S_0^pre > 0):
    φ^0_{θ_±}(Z) = φ^0_{θ_s}(Z) ± k |θ_s^pre| (S_0 / S_0^pre)
                   × ( φ^0_{θ_s^pre}(Z)/θ_s^pre + φ^0_{S_0}(Z)/S_0 - φ^0_{S_0^pre}(Z)/S_0^pre ).
```

When `(X^pre, U^pre) = (X, U)`, "the treatment selection component is then immediately
transportable across periods, since `C^{2,pre}_{0D} = C²_{0D}` and `ν^{2,pre}_{0s} = ν²_{0s}`"
(p. 23). The paper states no preference between the two approaches.

*Contour plots (Section 6.2, pp. 31-32; Figure 2):*

Horizontal axis `1 - R²_{O_XU~O_X|D=0}`, vertical axis the trend `η²`, "both on the same R²
scale"; one panel per fixed alignment (the application uses `|ρ_0| = 0.3` and `|ρ_0| = 1`).
"Each contour line corresponds to confounding scenarios that yield the same upper confidence
bound for θ" - the upper bound is used because the estimate is negative. A black triangle
marks the no-confounding confidence limit, a red dashed line "the threshold scenarios that
render the effect insignificant", red diamonds the benchmark points "1 x covariate" located
at the gain metrics, and a blue dashed line "the scenario implied by extrapolating the 2006
pre-trend deviation" (drawn at contour level 0.0185; its computation is not stated).

*Edge cases:*
- **Selection parameter at one.** `C²_{0D} = k/(1 - k)` diverges as
  `1 - R²_{O_XU~O_X|D=0} → 1` (p. 63); benchmark inference requires
  `0 < k_{0D,j} G_{0D,j} < 1` (p. 58).
- **No covariates.** The uncentered parameterization stays well defined with `ν²_{0s} = 1`
  (Table 7 note, p. 109); the centered one does not (p. 19).
- **Null inside the baseline interval.** RV = XRV = 0 (Definitions 3-4, p. 61).
- **Zero bias factors.** If `Var(g_0 - g_{0s} | D = 0) = 0` or
  `Var(α_0 - α_{0s} | D = 0) = 0`, the bias is zero (p. 102).
- **Infinite divergence.** `E[α²_0 | D = 0] < ∞` is required (Remark 6, pp. 65-66); in the
  Gaussian design the divergences are "finite if and only if `2σ²_{0X} > σ²_{1X}` and
  `2σ²_{0U} > σ²_{1U}`" (p. 65).
- **Pre-trend benchmark.** Requires `θ_s^pre ≠ 0`, and `S_0 > 0`, `S_0^pre > 0` for the
  bias-factor approach (p. 59).
- **Parametric nuisances under misspecification.** Confidence bounds over-cover and the
  RV / XRV lower-limit coverage falls below nominal (pp. 67-68, Figures 6 and 11).
- **Estimated `ν̂²_{0s}` below one, or non-positive `σ̂²_{0s}`.** The population value of
  `ν²_{0s}` is at least one (Corollary 1); the debiased estimate from Equation 11 need not
  be. The paper has no rule for this case.

**Algorithm 2 (multiplier bootstrap for uniform bands)** (p. 108) is "the multiplier
bootstrap procedure of Callaway and Sant'Anna (2021), which we use to construct the uniform
95% confidence bands shown in Figure 1" (p. 107). Input: an `n × T` influence-function matrix
of the short ATT across periods. For `b = 1, ..., B`: draw Mammen weights
(`κ = (sqrt(5) + 1)/2`; `V_i = 1 - κ` with probability `κ/sqrt(5)`, `κ` otherwise), compute
`R̂* = (1/sqrt(n)) Σ_i V_i φ^0_{θ_s}(Z_i)`. Scale:
`σ̂_t = IQR(R̂*(t)) / (Φ^{-1}(0.75) - Φ^{-1}(0.25))`; statistic `max_t |R̂*(t)| / σ̂_t`;
critical value its `1 - α` quantile; band `θ̂_s(t) ± ĉ_{1-α} σ̂_t / sqrt(n)`. `B` is not
given. It is used only for the event-study bands of Figures 1 and 21, not for the
sensitivity bounds.

**Monte Carlo evidence (Appendix E, pp. 64-78):**

*Data generating process (E.1, pp. 64-65):*

```
Step 1.  D_i ~ i.i.d. Bernoulli(p);   (X_i, U_i) | D_i = d ~ N(μ_d, Σ_d),
         μ_d = (μ_dX, μ_dU),   Σ_d = diag(σ²_dX, σ²_dU),   d ∈ {0, 1}.

Step 2.  Y_{i,t}(0) = α_i + ρ_t + X_i β_xt + U_i β_ut + ε_{i,t},
         Y_{i,t}(1) = Y_{i,t}(0) + 1{t ≥ 2} θ + (ν_{i,t} - ε_{i,t}),
         ρ_t = β_xt = β_ut = t,   α_i | D_i = d ~ N(2d, 1),   ε_{i,t}, ν_{i,t} ~ N(0, 1).

Step 3.  ΔY_i = (Δρ + X_i Δβ_x + U_i Δβ_u) + θ D_i + Δε̃_i,
         Δρ = Δβ_x = Δβ_u = 1,   Δε̃_i = D_i Δν_i + (1 - D_i) Δε_i ~ N(0, 2).
```

Design values (p. 66): `θ = 2`, `μ_0 = (0.3, 0.3)`, `μ_1 = (0, 0)`,
`(σ²_{0X}, σ²_{0U}) = (6, 6)`, `(σ²_{1X}, σ²_{1U}) = (3, 3)` (variances);
`p ∈ {0.2, 0.5, 0.8}`, `n ∈ {500, 2,000, 5,000}`, `α ∈ {0.01, 0.05, 0.1}`; "ten-fold
cross-fitting"; `B = 5,000` repetitions. Correct specification (p. 65): the propensity score
is "a logistic regression with quadratic terms in the covariates for both the 'long' and
'short' estimations" and the outcome-evolution model is linear.

*True quantities (pp. 65, 70-72):*

```
C²_{0ΔY} = σ²_{0U} / (σ²_{0U} + 2),      C²_{0D} = χ²(P_{U|1} || P_{U|0}),

ρ²_0 = (μ_{1U} - μ_{0U})² / ( σ²_{0U} × (χ²(P_{X|1} || P_{X|0}) + 1) × χ²(P_{U|1} || P_{U|0}) ),

θ_s = θ + (μ_{1U} - μ_{0U}),       E[(ΔY - g_{0s})² | D = 0] = σ²_{0U} + 2,

χ²(P_{X|1} || P_{X|0}) = σ²_{0X} / ( σ_{1X} sqrt(2σ²_{0X} - σ²_{1X}) )
                         × exp( (μ_{1X} - μ_{0X})² / (2σ²_{0X} - σ²_{1X}) ) - 1     (same form for U).
```

**[Reviewer check - not in paper]** Plugging in the design values: both divergences equal
`0.166305`; `C²_{0ΔY} = 0.75`; `C²_{0D} = 0.166305`; `ρ²_0 = 0.077334` (`ρ_0 < 0`);
`θ_s = 1.7`; `σ²_{0s} = 8`; `ν²_{0s} = 1.166305`; and
`ρ²_0 C²_{0ΔY} C²_{0D} σ²_{0s} ν²_{0s} = 0.09 = (θ - θ_s)²`. The paper prints
`ν²_{0s} ≈ 1.166` and the population RV / XRV 0.4228 / 0.2365 (p. 94), not the other values.

*Coverage of the confidence bounds with the true bias factors plugged in* (Table 2, p. 73;
z-statistic = (empirical - nominal) / `sqrt(α(1 - α)/5,000)`):

| p   | n     | 90% Cov. | z     | 95% Cov. | z     | 99% Cov. | z     |
|-----|-------|----------|-------|----------|-------|----------|-------|
| 0.2 | 500   | 0.8940   | -1.41 | 0.9456   | -1.43 | 0.9906   |  0.43 |
| 0.2 | 2,000 | 0.8988   | -0.28 | 0.9466   | -1.10 | 0.9884   | -1.14 |
| 0.2 | 5,000 | 0.8936   | -1.51 | 0.9458   | -1.36 | 0.9882   | -1.28 |
| 0.5 | 500   | 0.8974   | -0.61 | 0.9484   | -0.52 | 0.9906   |  0.43 |
| 0.5 | 2,000 | 0.9036   |  0.85 | 0.9488   | -0.39 | 0.9906   |  0.43 |
| 0.5 | 5,000 | 0.9000   |  0.00 | 0.9486   | -0.45 | 0.9916   |  1.14 |
| 0.8 | 500   | 0.9014   |  0.33 | 0.9496   | -0.13 | 0.9872   | -1.99 |
| 0.8 | 2,000 | 0.9090   |  2.12 | 0.9534   |  1.10 | 0.9914   |  0.99 |
| 0.8 | 5,000 | 0.8958   | -0.99 | 0.9470   | -0.97 | 0.9876   | -1.71 |

*Sensitivity statistics* (pp. 66-67; Figures 4-5): the Monte Carlo mean of the
point-estimate RV / XRV tracks the population values; the `α` versions are evaluated as
lower confidence limits, `P_n(RV_{θ*=0,α} ≤ RV_{θ*=0})`, and "the empirical coverage closely
tracks the nominal levels".

*Misspecification* (pp. 67-68; Table 3, p. 75; Figures 6-11): the same design, but "fit the
short nuisance models using `X* = exp(X/2)` in place of X". The "parametric" model (linear
for `g_{0s}`, logistic for `π`, on `X*`, `(X*)²`) is compared with a "nonparametric" pipeline
selected by RMSE among ridge and lasso (`glmnet`; `X*`, `(X*)²`, `(X*)³`) and a random forest
(`ranger`, `num.trees = 500`, `mtry ∈ {2, 3}`, `min.node.size ∈ {50, 100, ..., 300}`,
`splitrule ∈ {variance, extratrees}`). "Across 500 repetitions, random forest is selected in
87% of replications for the propensity score π and in 98.8% of replications for the outcome
evolution model `g_{0s}`." Findings: parametric bounds "tend to be wider and, consequently,
exhibit over-coverage"; "in some repetitions the parametric model produces excessively wide
confidence bounds. This occurs because its estimates of `ν²_{0s}` are highly sensitive to
the estimated propensity scores". Figure 11 shows parametric RV / XRV coverage well below
nominal and worsening with `n` (numbers are not printed; the figure is the only source).

**Empirical application (Sections 2 and 6, pp. 7-11 and 29-33; Appendix G, pp. 106-113):**

- **Sample** (p. 7; Table 6 caption, p. 108): the 2007 cohort of Callaway and Sant'Anna
  (2021). "The treated group consists of the 584 counties in those states, and the control
  group consists of the 1,377 counties in states that never raised their minimum wage above
  the federal floor before the end of 2007." Outcome: teen employment (effects are reported
  in percent). Covariates (Tables 4 and 6): `region`, `white`, `hs`, `pov`, `lpop`,
  `lmedinc`, "the covariates from Callaway and Sant'Anna (2021)" (p. 106). The paper does not
  name the data file or provider.
- **Learners** (G.1, pp. 106-107; Tables 4-5): parametric (`lm` / `glm`; `region`, `white`,
  `hs`, `pov`, `lpop`, `lpop²`, `lmedinc`, `lmedinc²`), ridge and lasso (`glmnet`; cubic
  polynomials and region interactions), and a random forest (`ranger`, `num.trees = 1,000`,
  `mtry ∈ {2, 4, 6}`, `min.node.size ∈ {10, 15, 25, 50, 75, ..., 150}`,
  `splitrule ∈ {variance, extratrees}`). Cross-fitted RMSE(D) / RMSE(ΔY): parametric
  0.420 / 0.1548, lasso 0.3803 / 0.1540, ridge 0.3859 / 0.1539, random forest
  0.3719 / 0.1529. "Random Forest attains the lowest RMSE for both, and we therefore use it
  as the learner for our main analysis." The selected hyperparameters and the fold count are
  not reported.
- **Headline numbers** (pp. 8-9): unconditional estimate -2.77%, conditional estimate
  -3.66%; 2006 pre-trend -3.9% (unconditional) and -3.8% (conditional). Exact decomposition
  of the covariate adjustment (p. 9):
  ```
  (-0.0277) - (-0.0366) = 0.0089 = 0.194 × 0.117 × 2.548 × 0.154
                                   (alignment) (trend) (selection) (scale)
  ```
- **Table 1: Minimal Sensitivity Reporting** (p. 30):

  | Short Estimate (θ̂_{s,2007}) | Std. Error | Confidence Interval | RV_{θ*=0, α=0.05} | XRV_{θ*=0, α=0.05} |
  |---|---|---|---|---|
  | -0.0366 | 0.0105 | [-0.0572; -0.0160] | 4.38% | 0.20% |

- **Table 7: decomposition of the observed confounding strength** (p. 109), adding one
  covariate at a time to the empty set, `θ_{s,j} - θ_{s,∅} = -ρ_{0,j} C_{0ΔY,j} C_{0D,j} S_{0,∅}`
  (standard errors in parentheses):

  | Covariate | Bias | Alignment | Trend | Imbalance |
  |---|---|---|---|---|
  | Region (Overall) | -0.0010 (0.0015) | 0.291 (0.425) | 0.036 (0.023) | 0.589 (0.036) |
  | Region (Midwest) | 0.0016 (0.0006) | -1.000 (0.498) | 0.032 (0.021) | 0.263 (0.030) |
  | Region (South) | 0.0013 (0.0012) | -1.000 (1.921) | 0.015 (0.030) | 0.504 (0.029) |
  | Region (West) | -0.0025 (0.0011) | 1.000 (0.611) | 0.033 (0.028) | 0.432 (0.039) |
  | Population (1000s) | -0.0016 (0.0007) | 0.765 (0.373) | 0.045 (0.010) | 0.295 (0.032) |
  | log(Population) | -0.0017 (0.0007) | 1.000 (1.834) | 0.023 (0.027) | 0.286 (0.032) |
  | Median Inc. (1000s) | -0.0002 (0.0006) | 0.051 (0.186) | 0.110 (0.018) | 0.179 (0.034) |
  | log(Median Inc.) | -0.0002 (0.0006) | 0.058 (0.210) | 0.110 (0.019) | 0.162 (0.037) |
  | White | 0.0011 (0.0010) | -0.307 (0.261) | 0.060 (0.026) | 0.406 (0.029) |
  | HS degree | -0.0026 (0.0010) | 0.607 (0.246) | 0.072 (0.016) | 0.389 (0.026) |
  | Poverty rate | 0.0001 (0.0008) | -0.029 (0.283) | 0.076 (0.026) | 0.251 (0.026) |
  | X_all | -0.0089 (0.0079) | 0.194 (0.166) | 0.117 (0.045) | 2.548 (0.170) |

  Note: "`S_{0,∅} = sqrt(σ̂²_{0s,∅} ν̂²_{0s,∅})` = 0.154 with an SE of 0.006. `σ̂²_{0s,∅}` = 0.024 with an SE of 0.002, and
  `ν̂²_{0s,∅}` = 1 because there are no covariates."
- **Table 8: leave-one-out gains** (p. 110; "All estimates are debiased and cross-fitted"):

  | Observed covariate | G_{0ΔY,j} | G_{0D,j} | ρ_{0,j} | θ_s - θ_{s,-j} |
  |---|---|---|---|---|
  | lmedinc | 0.0051 | 0.3701 | -0.2296 | 0.0036 |
  | region | 0.0049 | 0.7389 | 0.1327 | -0.0025 |
  | white | 0.0021 | 0.5190 | -0.0544 | 0.0006 |
  | pov | 0.0047 | 0.2392 | 0.0834 | -0.0011 |

- **Table 9: one-sided 95% confidence intervals from covariate benchmarks** (p. 110):

  | Covariate | \|ρ_0\| = \|ρ_{0,j}\| Lower | Upper | \|ρ_0\| = 0.3 Lower | Upper | \|ρ_0\| = 1 Lower | Upper |
  |---|---|---|---|---|---|---|
  | lmedinc | -0.0622 | -0.0147 | -0.0613 | -0.0112 | -0.0840 | 0.0122 |
  | white | -0.0554 | -0.0161 | -0.0648 | -0.0106 | -0.1021 | 0.0253 |
  | pov | -0.0541 | -0.0156 | -0.0595 | -0.0144 | -0.0751 | -0.0002 |
  | region | -0.0838 | -0.0098 | -0.0841 | 0.0011 | -0.1679 | 0.0828 |

- **Table 10: one-sided 95% confidence intervals from the 2006 pre-trend** (p. 110): bias
  magnitude [-0.0922, 0.0272]; bias factors [-0.0917, 0.0270]. The value of `k` is not
  stated.
- **Figure 2 benchmark labels** (p. 32; upper confidence limit at `α = 0.05`):
  `|ρ_0| = 0.3`: pov -0.0143, lmedinc -0.0122, white -0.0131, region -0.0041;
  `|ρ_0| = 1`: pov -0.0027, lmedinc 0.0044, white 0.0015, region 0.032; "Observed" -0.016.
- **Anticipation variant** (G.5, pp. 111-113), ascribing "the full decrease in teen
  employment from 2005 to 2007 to the minimum wage increase in 2007". Table 11: estimate
  -0.074, SE 0.0109, CI [-0.0954; -0.0526], RV 10.46%, XRV 1.21%. Table 13 gains
  (`G_{0ΔY,j}`, `G_{0D,j}`, `ρ_{0,j}`, `θ_s - θ_{s,-j}`): lmedinc 0.0083, 0.3701, -0.2915,
  0.0068; region 0.0147, 0.7389, 0.2156, -0.0084; white 0.0082, 0.5190, -0.0801, 0.0021.
  Table 12 intervals (`|ρ_0| = |ρ_{0,j}|`; 0.3; 1): lmedinc [-0.1057, -0.0464],
  [-0.1068, -0.0425], [-0.1548, 0.0047]; white [-0.0978, -0.0500], [-0.1086, -0.0389],
  [-0.1600, 0.0127]; region [-0.1360, -0.0255], [-0.1382, -0.0106], [-0.2716, 0.1226].
- **Table 6 covariate balance** (p. 108; standardized mean difference
  `(μ_1 - μ_0)/sqrt((σ²_1 + σ²_0)/2)`): Midwest 0.303, South -0.613, West 0.419, Population
  (1000s) 0.183, log(Population) 0.346, Median Inc. (1000s) 0.153, log(Median Inc.) 0.150,
  White 0.397, HS Graduates 0.336, Poverty Rate -0.324.

**Reference implementation(s):**
- R: the paper says "Open-source software for R implements the methods discussed in this
  paper" (p. 5) and names "our R package `dml.sensemakr`" (pp. 82, 92). It gives no URL,
  version, function name, or license (zero hits for `http`, `github`, `CRAN`, `license`,
  `GPL`). Nuisance learners named: `stats` (`lm`, `glm`), `glmnet` (`cv.glmnet`), `ranger`.
  **Note:** package metadata checked on the review date (the DESCRIPTION file of
  `github.com/carloscinelli/dml.sensemakr`, development version 0.2.3.9000, not on CRAN)
  lists `License: GPL-3`. diff-diff is MIT: the library must NOT port or translate that
  package's source; it may be used only as a black-box numerical oracle (run it, compare
  outputs). This review was written from the paper alone; no package source was opened.
- Stata: no implementation is named.
- Python: no implementation is named.

**Requirements checklist:**

Core (Sections 3 and 5.1):
- [ ] Balanced two-period unit-level input; `ΔY` formed per unit; binary time-invariant group indicator
- [ ] `g_{0s}` fit on the untreated only; propensity `π` fit on both groups; `O_X/O = (π/(1 - π))((1 - p)/p)`
- [ ] `θ̂_s` from Equation 9 (ratio form normalized by `Σ D/p̂`)
- [ ] `σ̂²_{0s}` from Equation 10
- [ ] `ν̂²_{0s}` from Equation 11 (the debiased score with treated and control terms, not the control-arm plug-in)
- [ ] Component influence functions aligned observation by observation for the delta method of Theorem 2 (reviewer recommendation, not prescribed by the paper: one fold partition and one set of cross-fitted nuisances shared by the three components is the simplest way to get this)
- [ ] Influence functions `φ̂ = -ψ̂ / E_n[ψ̂^a]` and variance `E_n[φ̂²]/n` (Lemma 3)
- [ ] Bounds `θ̂_± = θ̂_s ± |ρ_0| sqrt(η²) sqrt((1 - R²)/R²) Ŝ_0` with the selection input on the `1 - R²_{O_XU~O_X|D=0}` scale, uncentered
- [ ] Delta-method influence function of Theorem 2, with separate standard errors for the lower and upper bound
- [ ] A documented critical-value convention (see the tension flagged above), applied consistently to the bounds, the RV = 0 gate, and the contour origin
- [ ] A documented overlap policy (the paper prescribes none), surfaced to the user

Sensitivity statistics (Section 5.3, Appendix D):
- [ ] RV with trend and selection bounds equal and `|ρ_0| ≤ 1`; XRV with trend bound 1
- [ ] RV = XRV = 0 when the two-sided no-confounding interval contains `θ*`
- [ ] Point-estimate versions match the closed forms `(sqrt(f⁴ + 4f²) - f²)/2` and `f²/(1 + f²)`
- [ ] Search over `CI^max` that is correct when the confidence limit is not monotone in the multiplier, or a guard for that case

Benchmarking (Section 4, Section 5.2, Appendix C):
- [ ] Covariate benchmarks: leave-`X_j`-out refit of both nuisances, gains `G_{0ΔY,j}`, `G_{0D,j}`, alignment `ρ_{0,j}`, plug-in bound, and the positivity conditions of p. 58
- [ ] Influence function of the benchmarked bound (not printed in this paper; derive or source from Chernozhukov et al. 2026, Appendix E.6)
- [ ] Pre-trend benchmarks: both extrapolation rules and the influence functions of p. 59

Validation targets:
- [ ] Appendix E design: closed-form `θ_s`, `σ²_{0s}`, `ν²_{0s}`, bias factors, and population RV / XRV 0.4228 / 0.2365
- [ ] Table 2 coverage within Monte Carlo error under the paper's convention for the bounds
- [ ] Application: Table 1 and Tables 7-10 as tolerance targets only (random-forest nuisances; fold count and selected hyperparameters unreported)

---

## Implementation Notes

### Data Structure Requirements
- One row per unit with the outcome change `ΔY = Y_2 - Y_1`, a binary group indicator `D`,
  and covariates `X`; equivalently a balanced two-period panel. Covariates are treated as
  pre-treatment characteristics (in Appendix E they "are fixed over time", p. 65).
- Pre-trend benchmarking additionally needs one earlier period: `ΔY^pre = Y_1 - Y_0` and
  `X^pre` (p. 58).
- Multi-period use is by reduction to the generic form of Equation 4 (Remark 1): choose a
  cohort, a comparison group and a base period, then apply the two-period analysis. The
  application does this with the 2007 cohort, never-treated controls, and "the preceding
  year as base period" (Figure 1 caption, p. 8). The conclusion (p. 34) lists staggered
  adoption ("performed period-by-period and aggregated across cohorts and time") and
  continuous or multi-valued treatments as extensions, not results.

### Computational Considerations
- One cross-fitted propensity model and one cross-fitted outcome-change regression on the
  untreated can serve all three components (an implementation choice; the paper defines
  three separate `DML(·)` calls and does not say whether they share fits). Each covariate
  benchmark needs one more pair of fits on `X_{-j}`.
- RV and XRV are one-dimensional root-finding problems in the common strength; contour
  plots evaluate the upper (or lower) confidence limit on a two-dimensional grid for a
  fixed alignment.
- `ν̂²_{0s}` depends on squared estimated odds and is the numerically fragile component:
  the paper uses ten folds "to stabilize estimation of `ν²_{0s}`" (p. 66) and reports that
  parametric estimates are "highly sensitive to the estimated propensity scores" (p. 68).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| Folds `L` | int | Not fixed by Algorithm 1; footnote 2 calls five "the default" and the simulations use ten | User choice; more folds stabilize `ν̂²_{0s}` (p. 66) |
| Significance level `α` | float in (0, 0.5) | 0.05 in the application | User choice; critical-value convention unresolved (see above) |
| Null value `θ*` | float | 0 in the application | User choice |
| Alignment bound | float in [0, 1] | 1 in RV and XRV; 0.3 and 1 in the contour plots; `\|ρ_{0,j}\|` in benchmarks | Plausibility judgment or benchmark |
| Trend bound on `η²_{ΔY~U\|X,D=0}` | float in [0, 1] | 1 in XRV | Plausibility judgment or benchmark |
| Selection bound on `1 - R²_{O_XU~O_X\|D=0}` | float in [0, 1) | None - "must always be restricted" (p. 15) | Plausibility judgment, odds / imbalance reading, MSM `Λ`, or benchmark |
| Benchmark multipliers `k_{0ΔY,j}`, `k_{0D,j}` | float ≥ 0 | 1 in the application's "1 x covariate" points | Plausibility judgment; `k_{0D,j} G_{0D,j} < 1` required |
| Pre-trend multiplier `k` | float ≥ 0 | Not stated for Table 10 | Plausibility judgment |
| Nuisance learners | regressor, classifier | Random forest in the application, chosen by cross-fitted RMSE | Any DML-compatible learner; "can also be used with standard parametric regressions" (p. 4) |
| Propensity trimming | - | None in the paper | Library decision; must be documented |
| Bootstrap draws `B` (Algorithm 2 only) | int | Not stated | Event-study bands only |

### Relation to Existing diff-diff Estimators
- **DMLDiD** (Chang 2020; `diff_diff/dml_did.py`) already cross-fits a propensity model and
  an untreated outcome-change regression and evaluates a Neyman-orthogonal ATT score per
  two-period cell. Equation 9 is the same family of score (`D/p` minus the odds-reweighted
  control residual). The shared infrastructure in `diff_diff/_crossfit.py` and
  `diff_diff/_learners.py` is the natural host for the nuisances. Any reduction claim
  ("with these settings `θ̂_s` equals the DMLDiD cell estimate") must be verified
  numerically, since normalization and `p̂` handling may differ.
- **Built-in learners.** The learner factory currently resolves the string names linear,
  ridge and sieve for regressors and logit for classifiers, and accepts user-supplied
  learner objects. The paper's application uses random forests; reproducing it needs a
  user-supplied learner, and forest implementations differ across languages.
- **HonestDiD** (Rambachan and Roth 2023; `diff_diff/honest_did.py`) is the library's
  pre-trend extrapolation tool. Section 4.2 and Proposition F.7 are the bridge: the
  parallel-trends violation `δ` equals `θ_0 - θ_{0s}`, and the bias-magnitude rule
  `θ_s ± k|θ_s^pre|` is a one-pre-period relative-magnitudes restriction. The paper's
  version covers one pre-period only.
- **PreTrendsPower** (Roth 2022) addresses a different question (power of pre-trend tests);
  the paper's point that pre-trends "are not dispositive" (p. 2) is complementary.
- **Multiplier bootstrap.** Algorithm 2 is the Callaway-Sant'Anna multiplier bootstrap with
  Mammen weights; the library's bootstrap utilities already offer Mammen weights. It is not
  needed for the sensitivity bounds.
- **Example data.** The library ships a loader for the `did` package's `mpdta` example
  (county teen employment, 2003-2007, log population as the only covariate). That is not
  the paper's sample (584 treated and 1,377 control counties, six covariates, event-study
  estimates for 2002-2007); the application cannot be reproduced from the shipped loader.
- **No existing OVB sensitivity surface.** A grep of `diff_diff/` and `docs/api/` on the
  review date for `robustness_value`, `sensemakr`, `omitted variable` and `Cinelli` returned
  no hits.
- **Inference conventions.** The paper's inference is i.i.d. with normal critical values. The
  library's cluster-robust and survey-design variance paths have no counterpart in the
  paper; extending the influence-function variance to clusters would be a library extension
  to document as such, not a paper result.

---

## Gaps and Uncertainties

1. **Critical-value convention (pp. 27, 30, 60, 61).** Theorem 2 and `CI^max` use
   `Φ^{-1}(1 - α)` per endpoint; Definitions 3-4 gate on a "two sided 1 - α" interval and
   describe `α = 1` as "setting the critical value to zero"; Table 1's interval is the two-sided 1.96
   interval. The paper does not reconcile these. Resolving it needs the authors or a
   black-box run of their package; it changes every reported confidence limit, RV and XRV.
2. **Benchmarked-bound influence function (p. 58).** Deferred to Appendix E.6 of
   Chernozhukov et al. (2026), which is not in the review set. Per the paper-source-only
   rule, add that paper to the review set before implementing benchmark inference.
3. **Regularity conditions (p. 26).** Lemma 3 imports Assumptions 3.1 and 3.2 of
   Chernozhukov et al. (2018) without restating them; no nuisance-rate condition is printed.
4. **Fold count and split handling.** `L` has no default in Algorithm 1; "the default five"
   appears only in a footnote (p. 66). Algorithm 1 as written cross-fits `p` with the other
   nuisances (`p` is an element of `η`) and gives no exception for full-sample estimation.
   Whether the three components share folds and whether splits are repeated are not stated.
5. **Overlap handling.** No trimming or clipping rule; any library rule is a library choice.
   `ν̂²_{0s}` is the component most exposed to extreme propensities.
6. **Computation of `CI^max`, RV and XRV.** No algorithm is given. A sufficient (not
   necessary) condition for the corner solution is `Ŝ_0 > z se(Ŝ_0)` (reviewer derivation
   above); the paper states no condition.
7. **Finite-sample pathologies.** No guidance for `ν̂²_{0s} < 1`, non-positive leave-out
   differences in the gains, or `|ρ̂_{0,j}| > 1` inside the benchmark bounds (the ±1 cap is
   stated only for the Table 7 display).
8. **Application reproducibility.** The data source, selected forest hyperparameters, fold
   count, the full-covariate `Ŝ_0`, `σ̂²_{0s}`, `ν̂²_{0s}`, and the `k` behind Table 10 and
   the 0.0185 contour are not printed. The 2006 `θ̂_s^pre` is given only as a rounded
   percentage ("the conditional 2006 deviation of -3.8%", p. 9), with no standard error. The reviewer back-calculation `S_0 ≈ 0.436` is indicative only.
9. **Sampling design.** i.i.d. only. State-level treatment assignment with county-level
   units is not addressed; clustered or survey inference is outside the paper.
10. **Scope.** Two periods, two groups, binary treatment, unit-level differences. Repeated
    cross-sections, staggered aggregation, more than one pre-period, and non-binary
    treatments are not developed.
11. **Printed oddities in v1.** Appendix A.2 writes the partial-R² joint regressor as `U+X`
    while the nonparametric version uses `(X,U)` (pp. 38-39); p. 46 prints a factor
    `(g_0 - g_0)` in one proof line; p. 69 says "analogously to the long case" under the
    long case; Proposition F.2(i) is headed `C²_{0ΔY}` vs `C²_Y` (p. 81); Table 3 lists one
    variable for the forest with `mtry ∈ {2, 3}` and "500 repetitions" (p. 75); Figures 19-20
    say "up to 5,000 repetitions" (p. 95); the centered population RV is printed 0.7446 where
    the closed form gives 0.74470 (p. 94); the main-text reference list has one Cinelli and
    Hazlett (2025) entry while the appendix list has 2025a and 2025b (pp. 35, 114).
12. **Figure-only evidence.** Coverage under misspecification (Figures 6, 11) and the
    comparison simulations (Figures 12-20) are reported in figures without tables; no
    numbers from those figures are recorded here as validation targets.

**Verification stamp.** Extraction used four parallel readers over PDF pp. 1-113; the
synthesis was written by the reviewer in the main context. The reviewer read the following
directly from the rendered PDF pages and confirmed them against the extractions: Section 2
numbers (pp. 7-10); Equations 1-8, Assumption 1, Lemmas 1-2, Theorem 1, Corollaries 1-2,
Propositions 1-2 (pp. 11-20); Section 4 definitions (pp. 20-24); Equations 9-11,
Algorithm 1, Lemma 3, Theorem 2, Section 5.2-5.3 (pp. 24-29); Table 1 and Figure 2 labels
(pp. 30-32); Appendix C.1, C.2 and Appendix D (pp. 54-64); the Appendix E design and
settings (pp. 64-68). Tables 2-13, Algorithm 2, Assumptions A.1-A.4 and both reference
lists were checked against the text layer. The remaining appendix material (Appendix A.2-A.3,
Appendix B proofs, E.3, Appendix F) rests on single-reader extraction cross-checked for internal
consistency (the algebraic identities reproduce one another). All reviewer arithmetic was
recomputed.
