# Paper Review: Synthetic Control Method: Inference, Sensitivity Analysis and Confidence Sets

**Authors:** Sergio Firpo (Insper), Vitor Possebom (Yale University)
**Citation:** Firpo, S., & Possebom, V. (2018). "Synthetic Control Method: Inference, Sensitivity Analysis and Confidence Sets." *Journal of Causal Inference*, 6(2), 20160026.
**PDF reviewed:** https://doi.org/10.1515/jci-2016-0026 (published *Journal of Causal Inference* version, open access; received 15 Nov 2016, revised 6 Aug 2018, accepted 11 Aug 2018, 26 pp). Per the project's PDFs-never-committed convention the local PDF is kept outside the repository; the published J. Causal Inference version (DOI 10.1515/jci-2016-0026) is the authoritative source. All equation, section, and footnote numbers below are pinned to that version.
**Review date:** 2026-06-01

> Scope note: this paper extends the **permutation / placebo inference** procedure of Abadie, Diamond & Hainmueller (the SCM benchmark) in two ways — (1) a **sensitivity analysis** that parametrically re-weights the placebo p-value away from the equal-weights benchmark, and (2) testing **any sharp null hypothesis** (not only "no effect whatsoever") via a modified RMSPE statistic, which it **inverts to construct confidence sets** for the treatment-effect path. It also generalizes to arbitrary test statistics, multiple outcomes (familywise error control), and multiple treated units (a pooled effect). This review is the **Step-1 fidelity artifact** for the SCM **confidence-set / CI-by-test-inversion** implementation (PR-B, **shipped** — `SyntheticControlResults.test_sharp_null()` / `confidence_set()`) layered on the existing `SyntheticControl` estimator; the sensitivity-analysis and multiple-outcome / multiple-treated extensions are documented here but flagged **deferred**. The estimator itself (donor weights `W`, predictor importance `V`) is taken as given from ADH 2010/2015 — already implemented as `SyntheticControl` — and is recapped only as the paper frames it. Nothing here is sourced from outside this paper.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md. This documents an **inference procedure on the existing `SyntheticControl` estimator**, not a new estimator — the `## SyntheticControl` heading mirrors `abadie-2021-review.md`. The REGISTRY implementation contract (`docs/methodology/REGISTRY.md` §SyntheticControl) was unchanged by the docs-only PR-A; PR-B (shipped) added the confidence-set methodology subsection there and flipped the relevant checklist items below.*

## SyntheticControl

**Primary source (this document):** Firpo, S., & Possebom, V. (2018). "Synthetic Control Method: Inference, Sensitivity Analysis and Confidence Sets." *Journal of Causal Inference*, 6(2), 20160026. https://doi.org/10.1515/jci-2016-0026

**Key implementation requirements:**

*Notation (Section 2.1):*
- `J+1` regions over `T` periods; region 1 is treated from `T0+1` to `T` (`T0 ∈ (1,T) ∩ ℕ`); regions `2..J+1` are the never-treated donor pool. `Y^N_{j,t}` / `Y^I_{j,t}` = potential outcomes without / with the intervention; `D_{j,t}` = treatment dummy (`= 1` iff `j = 1` and `t > T0`). Observed outcome `Y_{j,t} = Y^N_{j,t} + α_{j,t}·D_{j,t}`.
- `Y_j` = `(T0×1)` pre-period outcome vector; `X_j` = `(K×1)` predictors; `X_0` = `(K×J)` donor predictor matrix, `X_1` = `(K×1)` treated predictors. Some rows of `X_j` may be linear combinations of `Y_j` (footnote 6).

*Target — the intervention-effect path (Equation 1):*

    (1)  α_{j,t} := Y^I_{j,t} − Y^N_{j,t}

The object of interest is the treated path `(α_{1,T0+1}, …, α_{1,T})`. Since `Y^I_{1,t}` is observed for `t > T0`, only `Y^N_{1,t}` must be estimated.

*Estimator — the synthetic control (taken from ADH; Equations 2–3):*

    Ŷ^N_{1,t} = Σ_{j=2}^{J+1} ŵ_j · Y_{j,t}
    (2)  Ŵ(V) := argmin_{W∈𝒲} (X_1 − X_0 W)' V (X_1 − X_0 W),   𝒲 = {W : w_j ≥ 0, Σ_{j=2}^{J+1} w_j = 1}
    (3)  V̂   := argmin_{V∈𝒱} (Y_1 − Y_0 Ŵ(V))' (Y_1 − Y_0 Ŵ(V)),   V diagonal PSD, trace = 1
    estimated gap:  α̂_{1,t} := Y_{1,t} − Ŷ^N_{1,t}

**Footnote 7 (V-selection — cross-reference to PR-1 #523):** besides the nested pre-period-MSPE choice of `V` (Eq. 3), the authors note two alternatives — (a) subjective / prior weights (discouraged, as it undermines SCM's objectivity), and (b) **cross-validation**: split the pre-period into a *training* and a *validation* window, solve Eq. 2 on the training window, and pick `V` minimizing the validation-window outcome MSPE (Eq. 3 evaluated on the validation window) — *exactly the `v_method="cv"` procedure shipped in PR-1 (#523)*. The Stata `Synth` command also offers a **regression-based** `V`, `v_k = |β_k| / (Σ_k |β_k|)` from regressing `Y_1` on `X_1` (not implemented in this library). The choice presented in the main text (nested MSPE) is the most common in the empirical literature.

### Inference (Section 2.2) — the benchmark permutation test

Following Fisher's Exact Hypothesis Testing Procedure (Fisher 1935; Imbens & Rubin; Rosenbaum), ADH permute which region is treated: for each `j ∈ {2,…,J+1}` re-estimate `α̂_{j,t}` and compare the treated unit's effect vector to the placebo distribution.

*RMSPE-ratio test statistic (Equation 4):*

    (4)  RMSPE_j := [ Σ_{t=T0+1}^{T} (Y_{j,t} − Ŷ^N_{j,t})² / (T − T0) ]  /  [ Σ_{t=1}^{T0} (Y_{j,t} − Ŷ^N_{j,t})² / T0 ]

(post-intervention MSPE ÷ pre-intervention MSPE — controls for imperfect pre-fit; ADH 2010 introduce this to handle abnormally large `|α̂_{1,t}|` driven by poor pre-fit rather than a real effect.)

*Benchmark p-value (Equation 5) and the exact null (Equation 6):*

    (5)  p := ( Σ_{j=1}^{J+1} 𝟙[ RMSPE_j ≥ RMSPE_1 ] ) / (J + 1)
    (6)  H_0:  Y^I_{j,t} = Y^N_{j,t}   for each region j and period t   (the "exact null" / no effect whatsoever)

Reject `H_0` if `p < γ` (e.g. `γ = 0.1`). Rejecting the exact null implies *some* region has a non-zero effect in *some* period. **Footnote 8:** `γ` must be chosen carefully given the **discrete, finite** number of regions — the p-value granularity is `1/(J+1)`, which may preclude the usual 5% / 10% levels. The exact null is also known as the *sharp* null of no effect (it is stronger than "the average/typical effect is zero").

### Contribution 1 — Sensitivity analysis (Section 3)

The benchmark Eq. 5 weights all units equally; that choice is restrictive and the decision may depend on it. Generalize to `p := Σ_{j=1}^{J+1} π_j · 𝟙[RMSPE_j ≥ RMSPE_1]` (Equation 7) and impose a **parametric weight family** that distorts the uniform weights as little as possible (à la Rosenbaum / Cattaneo et al.). Step-by-step in the SCM framework (Section 3):

1. Estimate `RMSPE_1,…,RMSPE_{J+1}` for all placebo assignments; `RMSPE_1 = RMSPE^obs`.
2. Rename in decreasing order: `RMSPE_(1) > RMSPE_(2) > … > RMSPE_(J+1)`.
3. Define `j̄ ∈ Ω := {(1),…,(J+1)}` with `RMSPE_j̄ = RMSPE^obs` (the largest such index on ties).
4. Parametric weights (Equation 8):

       (8)  π_(j)(φ, v) = exp(φ·v_(j)) / Σ_{j'∈Ω} exp(φ·v_(j'))

   with sensitivity parameter `φ ∈ ℝ₊`, indicators `v_(j') ∈ {0,1}`, `v = (v_1,…,v_{J+1})`. At `φ = 0` all weights are equal (recovers the benchmark Eq. 5). Interpretation: a region with `v = 1` carries weight `Φ := exp(φ) − 1` times larger than a region with `v = 0`.
5. Generalized p-value (Equation 9):

       (9)  p(φ, v) = Σ_{(j)∈Ω} [ exp(φ·v_(j)) / Σ_{j'∈Ω} exp(φ·v_(j')) ] · 𝟙[ RMSPE_(j) ≥ RMSPE_j̄ ]

   reject the exact null if `p(φ, v) < γ`; `φ = 0` reduces to Eq. 5.
6. If the exact null is **rejected**: the worst-case scenario sets `v_(j) = 1 if (j) ≤ j̄, else 0`; define `φ̲ ∈ ℝ₊` solving `p(φ̲, v) = γ`. A **small** `φ̲` ⇒ the rejection is **not robust** (small deviations from equal weights flip the decision).
7. If the exact null is **not rejected**: the best-case scenario sets `v_(j) = 0 if (j) ≤ j̄, else 1`; define `φ̄ ∈ ℝ₊` solving `p(φ̄, v) = γ`.
8. Plot `φ` (x-axis) vs `p(φ, v)` (y-axis); a curve that moves too fast ⇒ the test is too sensitive to the weight choice.

Large `φ̲` / `φ̄` boost confidence that the conclusion is robust to deviations from the equal-weights benchmark, in the same spirit as ADH's benchmark.

### Contribution 2 — Sharp nulls and confidence sets (Section 4)

*Testing sharp nulls (Section 4.1).* Generalize the exact null to any sharp null:

    (10)  H_0^f:  Y^I_{j,t} = Y^N_{j,t} + f_j(t)        (region-specific effect function f_j, j ∈ {1,…,J+1})
    (11)  H_0^f:  Y^I_{j,t} = Y^N_{j,t} + f(t)          (common effect function f : {1,…,T} → ℝ; the practical case)

Under a sharp null all potential outcomes are known, so `Y^N` is recoverable from the observed data. The RMSPE statistic becomes (Equation 12):

    (12)  RMSPE^f_j := [ Σ_{t=T0+1}^{T} (Y_{j,t} − Ŷ^N_{j,t} − f(t))² / (T − T0) ]  /  [ Σ_{t=1}^{T0} (Y_{j,t} − Ŷ^N_{j,t} − f(t))² / T0 ]

`f(t)` appears in **both** windows because Eq. 11 defines `f` over all `t ∈ {1,…,T}`; for the operational constant (Eq. 15) and linear (Eq. 17) families `f` carries a `𝟙[t ≥ T0+1]` factor, so `f(t) = 0` in the pre-period and the denominator reduces to the plain pre-period MSPE of Eq. 4. The p-value (Equation 13):

    (13)  p^f(φ, v) := Σ_{j=1}^{J+1} [ exp(φ·v_j) / Σ_{j'=1}^{J+1} exp(φ·v_j') ] · 𝟙[ RMSPE^f_j ≥ RMSPE^f_1 ]

reject the sharp null Eq. 11 if `p^f(φ, v) < γ`. The exact null (Eq. 6) is the special case `f ≡ 0`. Three highlighted `(φ, v)` choices: `(φ = 0, v = (1,…,1))` = the ADH benchmark; the worst-case `φ̲` if rejected; the best-case `φ̄` if not. Choice of `f`: a linear / quadratic / exponential fit to the estimated `(α̂_{1,t})` (to predict future effects); the **cost path** of the intervention (test "effect = cost" for cost-benefit analysis); or a theory-predicted shape (e.g. the inverted-U / U / decreasing shapes for natural-disaster GDP effects).

*Confidence sets by test inversion (Section 4.2).* Inverting the test over effect functions gives the confidence set (Equation 14):

    (14)  CS_{(1−γ)}(φ, v) := { f ∈ ℝ^{{1,…,T}} : p^f(φ, v) > γ }

= every effect function whose associated sharp null is **not rejected**. The general `ℝ^T` set is computationally infeasible and "too general to be informative," so the paper restricts to two **one-parameter** families:

    (15)  H_0^c:  Y^I_{j,t} = Y^N_{j,t} + c · 𝟙[t ≥ T0+1]                     (constant-in-time effect, c ∈ ℝ)
    (16)  CI_{(1−γ)}(φ, v) := { f : f = c and p^c(φ) > γ } ⊆ CS_{(1−γ)}(φ, v)    (confidence INTERVAL)

    (17)  H_0^c̃:  Y^I_{j,t} = Y^N_{j,t} + c̃ · (t − T0) · 𝟙[t ≥ T0+1]           (linear-in-time, zero intercept)
    (18)  C̃S_{(1−γ)}(φ, v) := { f : f = c̃·(t−T0)·𝟙[t ≥ T0+1] and p^c̃(φ) > γ }   (confidence SET)

Operationally: grid over the scalar `c` (or `c̃`), test each value via Eqs. 12–13, and keep those satisfying the set's defining **strict** inequality `p^c(φ) > γ` (Eqs. 14/16/18). **Boundary/equality convention (paper-sourced, stated once).** The paper's inequalities are not uniform at the boundary `p = γ`: the RMSPE-based tests *reject* at `p < γ` (Eqs. 5/9/13), the general-statistic test rejects at `p ≤ γ` (Eq. 19), and the confidence set is the *strict* `p^f > γ` (Eq. 14). Eq. 14's set is therefore **not** the exact complement of the Eq. 13 rejection region — they differ at `p^f = γ` (Eq. 14 *excludes* it, while Eq. 13 does *not* reject it; Eq. 19, by contrast, *would* reject at `p = γ`). This matters because the permutation p-value is **discrete** (a multiple of `1/(J+1)`), so `p = γ` is reachable. A PR-B implementation should pin a single boundary convention — we recommend Eq. 14's strict `p^f > γ` for confidence-set membership (i.e. exclude `p^f = γ`) — and document it. Extending to two-parameter functions (quadratic / exponential / logarithmic) is "theoretically straightforward" from Eq. 14 but computationally heavier; the paper restricts its main examples to one parameter. Confidence sets summarize **significance** (is `f ≡ 0` excluded?), **precision** (narrower ⇒ stronger conclusions), and **robustness** (compare set areas across `φ`). They are **uniform** over time (they combine information across all post periods to describe which effect *functions* are not rejected); a **point-wise** per-period CI instead uses `α̂_{1,t'}` as the test statistic separately for each `t' > T0` (Section 6.1 cautions that a point-wise interval may be inadequate).

### Other test statistics + Monte Carlo (Section 5)

The sensitivity analysis and confidence sets work with *any* test statistic `θ^f(ι, τ, Y, X, f)` (Equation 19), where `ι` = treatment-assignment vector, `τ` = post-period indicator, `Y` = observed-outcome matrix, `X` = predictors, `f` = the sharp-null function; permutation replaces region 1's assignment with each canonical basis vector `e_j`:

    (19)  p_{θ^f}(φ, v) := Σ_{j=1}^{J+1} [ exp(φ·v_j) / Σ_{j'} exp(φ·v_j') ] · 𝟙[ θ(e_j, τ, Y, X, f) ≥ θ^{f,obs} ] ≤ γ

Five test statistics are compared: `θ¹ = mean(|α̂_{j,t}| : t > T0)` (ADH); **`θ² = RMSPE` (Eq. 4, ADH-recommended)**; `θ³ = |t-stat of the mean post effect vs 0|` (Mideksa); `θ⁴` = simple post-period difference-in-means, treated − controls (Imbens & Rubin); `θ⁵` = the interaction coefficient in a DiD regression (Equation 20). **Monte Carlo (T = 25, T0 = 15, K = 10, J+1 = 20; factor-model DGP Eq. 21; linear intervention effect Eq. 22 with `λ ∈ {0,.05,.1,.25,.5,1,2}`; 21,000 reps):** all five permutation tests have the correct size (0.10); **RMSPE (`θ²`) is uniformly more powerful than the simple `θ⁴`/`θ⁵`** and out-powers the Conley–Taber asymptotic test (which is mis-sized at this small `N`); the t-test `θ³` is the most powerful **but** fails when positive and negative effects cancel in the post-period mean — for sign-varying effects, use the multiple-outcome framework (§6.1). Excluding poor-pre-fit donors (pre-period MSPE > 5× the treated unit's) raises `θ¹`'s power but slightly over-rejects, and makes `θ²`/`θ³` slightly conservative. No single statistic dominates — match it to the research question (Eudey et al.).

### Extensions (Section 6)

*Multiple outcomes (Section 6.1) — familywise error control.* For `M` outcomes `Y^1,…,Y^M` with sharp null `H_0^f: Y^{m,I}_{j,t} = Y^{m,N}_{j,t} + f_m(t)` (Equation 23), compute a per-outcome observed p-value, then a **FWER-controlled** p-value (adapting Anderson 2008): order outcomes by observed p-value, take running minima, apply the parametric weights (Equation 24), enforce monotonicity, and reject outcome `m` if `p^{fwer}_m ≤ γ`. A single-outcome study where each post-period is treated as a separate "outcome" reduces to this; Anderson's **summary index test** is more powerful for "is there *any* effect?", whereas FWER control is for the *timing* of the effect.

*Multiple treated units (Section 6.2) — pooled effect.* For `G` similar interventions (region `1^g` treated in interventon `g`), the pooled estimator is `ᾱ_{1,t} := Σ_{g=1}^{G} α̂_{1^g,t} / G` with sharp null Eq. 25. A single pooled test statistic `θ_{pld,f}` summarizes all time periods (to avoid over-rejection); placebo assignments permute which region is treated in each intervention via canonical bases, giving `Q := Π_{g=1}^{G} (J^g + 1)` pooled placebo assignments and the p-value Equation 26. Confidence sets (§4.2) extend by using Eq. 26.

### Empirical application (Section 7)

A re-analysis of ETA terrorism on Basque Country GDP per capita (Abadie & Gardeazabal 2003; `J+1 = 17`, `T0 = 1969`, post 1970–1997). A **one-sided** statistic (only negative effects are of interest) `θ = −ᾱ_post/(T−T0) ÷ (σ̂/√(T−T0))` gives `p = 3/17` (a marginal rejection of the exact null); excluding poor-pre-fit regions (Madrid, Extremadura, Balearic) → `p = 2/14`. Sensitivity: `φ̲ = 0.495` suffices to stop rejecting at the `3/14` level ⇒ the Basque region's weight need only be ~64% larger than `v = 0` units to overturn the result ⇒ **not very robust** (small sample). One-sided `12/14`-confidence sets (constant Eq. 16 and linear Eq. 18) lie below zero ⇒ economically relevant negative effects. A quadratic effect is not rejected (`p_quadratic = 6/14`, robust at `φ̄ = 1.905`) ⇒ the impact is initially negative but **attenuates toward zero in the long run**. Treating each year as a separate outcome (§6.1) localizes the negative impact to the 1980s with a recovery in the late 1990s.

**Reference implementation(s):**
- Authors' R and Stata code for the confidence sets in Eqs. 16 & 18 (footnote 15; the `goo.gl/RBYomh` short-link is stale) and a **Code Ocean** replication capsule (DOI `10.24433/CO.23bd238f-38c5-4b3e-82f4-3a1624fd8a33`).
- Built on the authors' `Synth` package (R / MATLAB / Stata) for the underlying SCM fit.

**Requirements checklist** (features this paper adds beyond ADH 2010/2015; **PR-B** = the planned next implementation target, **deferred** = later):
- [x] (PR-B) Sharp-null `RMSPE^f` test (Eqs. 12–13) reusing the in-space placebo permutation — subtract the hypothesized `f(t)` from the post-period gaps and re-rank. **Shipped:** `SyntheticControlResults.test_sharp_null(effect, gamma=...)`.
- [x] (PR-B) Confidence **interval** for a constant-in-time effect (Eqs. 15–16) by test inversion over a `c`-grid. **Shipped:** `confidence_set(family="constant")`.
- [x] (PR-B) Confidence **set** for a linear-in-time effect (Eqs. 17–18) by test inversion over a `c̃`-grid. **Shipped:** `confidence_set(family="linear")`.
- [x] (PR-B) Benchmark `(φ = 0, v = (1,…,1))` p-value (reuse `in_space_placebo`'s RMSPE-ratio): shipped — `test_sharp_null(0)` is identically `placebo_p_value`. **One-sided variant (Section 7): still `[ ]` deferred** — §7 uses the signed-`t` statistic `θ³` from the deferred general-`θ` menu (Eq. 19), so it ships with that menu, not here.
- [ ] (deferred) Sensitivity-analysis parametric weights `π_(j)(φ, v)` (Eqs. 7–9) + worst/best-case `φ̲`/`φ̄` robustness curve (Section 3).
- [ ] (deferred) General test-statistic menu `θ¹`–`θ⁵` (Eq. 19, Section 5).
- [ ] (deferred) Multiple-outcome FWER control (Eqs. 23–24) and multiple-treated-unit pooled confidence sets (Eqs. 25–26, Section 6).

---

## Implementation Notes

### Data Structure Requirements
- Same as `SyntheticControl`: a balanced aggregate panel (one treated unit + a curated donor pool), a long pre-period, and an absorbing block-treatment suffix. The inference layer adds **no new data requirements** — it consumes the fitted gap path `(α̂_{j,t})` and the per-unit pre/post MSPEs the estimator already computes.
- The sharp-null test and the confidence sets need the **full placebo reference set** (one synthetic-control refit per donor) — exactly the object the existing `in_space_placebo()` builds.

### Computational Considerations
- The benchmark test (Eq. 5) is `O(J)` synthetic-control refits (the permutation reference set). The sensitivity analysis (Eqs. 8–9) is a **closed-form re-weighting** of the *already-computed* `RMSPE_j` plus a one-dimensional root-find for `φ̲`/`φ̄` — no refits.
- **Test-inversion CI = a grid search × the permutation test.** For each grid value `c` (or `c̃`): subtract `f(t)` from the relevant post-period outcomes, recompute `RMSPE^f` for all `J+1` units (Eq. 12), and evaluate Eq. 13. Because the donor synthetic controls and the pre-period denominators are unchanged across the grid (only the post-period gap shifts by `f(t)`), the per-grid-value cost is dominated by re-ranking, not refitting. Cost scales with grid resolution × `J`.
- The general `ℝ^T` confidence set (Eq. 14) is computationally infeasible — an implementation must restrict to the constant / linear (or a small parametric) family and choose a finite grid.

### Tuning Parameters

| Parameter | Type | Default (this paper) | Selection method |
|-----------|------|----------------------|------------------|
| `φ` (sensitivity) | `≥ 0` | `0` (equal-weights benchmark) | swept to report `φ̲`/`φ̄`; `φ = 0` reproduces ADH |
| `v` (weight indicators) | `{0,1}^{J+1}` | `(1,…,1)` | worst / best-case patterns (steps 6–7) for the robustness bound |
| `γ` (significance level) | `∈ (0,1)` | `0.1` | chosen given the discrete `1/(J+1)` granularity (fn 8) |
| effect family `f` | constant / linear (/ parametric) | — | constant (Eq. 16) or linear (Eq. 18); two-parameter possible but costly |
| grid bounds + resolution | scalar grid | **unspecified by the paper** | implementation choice (documented deviation) |

### Relation to Existing diff-diff Estimators
- This is the **inference layer for the existing `SyntheticControl`** estimator (`diff_diff/synthetic_control.py`); it introduces **no new estimator**. PR-B would reuse `SyntheticControlResults.in_space_placebo`, `_placebo_fit_unit`, and the `_rmspe_ratio` / `_mspe` helpers: the benchmark test (`φ = 0`) is literally the existing in-space placebo (ADH 2010 §2.4, already shipped), and the CI adds the sharp-null `f(t)` subtraction + the grid inversion on top.
- The paper's **footnote 7** cross-validation `V` selection is the `v_method="cv"` shipped in PR-1 (#523); the sensitivity analysis is orthogonal to (and composes with) the existing `placebo_p_value`.
- Complements **conformal inference (Chernozhukov–Wüthrich–Zhu 2021)** — the other SCM inference track on the roadmap (review already on file). Firpo–Possebom is permutation / Fisher-randomization-based (finite-sample, valid under exchangeability of placebo assignments); CWZ is residual-exchangeability conformal. They are alternative routes to SCM uncertainty quantification.

---

## Gaps and Uncertainties

- **Grid bounds and resolution for the test-inversion CI are not specified.** Section 4.2 gives the set definitions (Eqs. 14/16/18) but not how to grid `c`/`c̃` or locate the interval endpoints — an implementation choice for PR-B (a documented deviation), e.g. a bracketing search on `p^c(φ) − γ`.
- **The general confidence set `CS ⊆ ℝ^T` (Eq. 14) is computationally infeasible** and "too general to be informative" (the paper's own framing); only the one-parameter constant (Eq. 16) and linear (Eq. 18) subsets are operationalized. Two-parameter families are called "theoretically straightforward" but are not demonstrated.
- **`γ` and finite-sample granularity:** with `J+1` regions the permutation p-value is a multiple of `1/(J+1)`, so not every conventional level is attainable (fn 8). The empirical application reports `3/17`, `2/14`, `12/14`, etc., rather than 0.05 / 0.10.
- **Point-wise vs uniform confidence sets:** the constructed sets are uniform over the post-period; a per-period point-wise interval (using `α̂_{1,t'}`) is mentioned but the paper cautions (Section 6.1) it may be inadequate without a multiplicity correction.
- **Sensitivity `v` worst/best-case patterns** (steps 6–7) define `φ̲`/`φ̄`, but selecting among multiple `v` that achieve a given decision rests on the "distort the uniform weights as little as possible" heuristic — a deterministic tie-break is left to the implementer.
- **Replication code:** the `goo.gl/RBYomh` short-link (fn 15) is stale; the live artifact is the Code Ocean capsule (DOI `10.24433/CO.23bd238f-38c5-4b3e-82f4-3a1624fd8a33`). Not consulted for this review (paper-sourced only).
