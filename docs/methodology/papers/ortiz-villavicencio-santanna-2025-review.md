# Paper Review: Better Understanding Triple Differences Estimators

**Authors:** Marcelo Ortiz-Villavicencio, Pedro H. C. Sant'Anna (both Emory University)
**Citation:** Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). *Better Understanding Triple Differences Estimators.* Working paper, arXiv:2505.09942 [econ.EM]. JEL: C10; C14; C21; C23.
**PDF reviewed:** **arXiv:2505.09942v3** (v3 posted **18 Jul 2025** per arXiv's submission history; original submission 15 May 2025) — version-pinned public URL https://arxiv.org/abs/2505.09942v3. Per the project's PDFs-never-committed convention, the local PDF is kept outside the repository; the arXiv v3 page is the authoritative source. **Version-pinned to v3** (all equation, theorem, corollary, and remark numbers below are v3's; other arXiv versions may renumber them — do not reconcile against a different version's rendering, e.g. ar5iv).
**Review date:** 2026-05-30

---

## Methodology Registry Entry

*Formatted to match `docs/methodology/REGISTRY.md` structure. This file is the canonical **scholarly paper review** for arXiv:2505.09942 and is the **Primary source** for **two** library estimators that share this paper: `TripleDifference` (the two-period DDD **estimand and three-DiD decomposition**, paper §3.1 / §4 specialized to a single treatment date — the library's repeated-cross-section data path follows the companion `triplediff` package; see Relation to diff-diff) and `StaggeredTripleDifference` (the multi-period / staggered case, paper §3.2 / §4). It **supplements — it does not supersede** — the existing REGISTRY sections `## TripleDifference` and `## StaggeredTripleDifference`, which remain the implementation contract. The equation anchors those REGISTRY sections already cite were verified accurate against v3 (see Gaps and Uncertainties).*

**Note (paper-vs-package scope):** This paper is the primary source for the **DDD estimand and three-DiD decomposition** that both library estimators implement. `TripleDifference` runs the **repeated-cross-section** path (`triplediff::ddd(panel=FALSE)`); its repeated-cross-section data handling and the associated inference follow the companion `triplediff` package, because the paper itself assumes a balanced panel and defers repeated-cross-section / unbalanced data to future work (§7). Read this review as authoritative for the DDD *estimand, identification, and balanced-panel inference*, not for the `panel=FALSE` data path.

## TripleDifference & StaggeredTripleDifference (DDD)

**Primary source:** Ortiz-Villavicencio & Sant'Anna (2025), *Better Understanding Triple Differences Estimators*, arXiv:2505.09942v3.

**Reference implementations:** R `triplediff` package — `ddd()` (group-time estimation, `panel = TRUE/FALSE`) and `agg_ddd()` (aggregation). Freely available on GitHub; "builds on and supersedes the material used for the 2022 Causal Solutions YouTube Lecture."

**Central thesis (Abstract; §1; §7):** Common DDD implementations — taking the *difference between two DiDs*, or running *three-way fixed-effects (3WFE) regressions* — are **generally invalid when identification requires conditioning on covariates**, and in staggered settings **pooling all not-yet-treated units as one comparison group introduces bias even without covariates**. The paper develops regression-adjustment (RA), inverse-probability-weighting (IPW), and doubly-robust (DR) DDD estimators that remain valid under covariate-adjusted DDD parallel trends, and a two-step GMM aggregator (based on re-centered influence functions) that combines multiple comparison groups for precision.

---

### Setup & Notation (§2)

- Fixed-`T`, large-`n` panel; units `i = 1, …, n`, periods `t = 1, …, T`. Treatment is an **absorbing** binary state (once treated, stays treated).
- `S_i ∈ S ⊆ {2, …, T} ∪ {∞}` — the **enabling group**: first period the policy is *enabled* for unit `i`'s group (e.g. a state). `S_i = ∞` ⇒ never enabled by `T`.
- `Q_i ∈ {0, 1}` — the **eligibility / qualifying partition** (e.g. "is a woman", "is a specific crop"), **time-invariant**.
- Treatment indicator: `D_{i,t} = 1{t ≥ S_i, Q_i = 1}` — a unit is treated only if it satisfies **both** criteria (its group enabled treatment **and** it qualifies).
- `G_i = min{t : D_{i,t} = 1}` — first treated period; `G_i = S_i` if `Q_i = 1`, and `G_i = ∞` if `Q_i = 0`. `G_trt = G \ {∞}`.
- A **never-enabled** group (`S_i = ∞`) is assumed always to exist. If every unit eventually enables, the paper drops all observations after `max S_i` and treats the last-enabling group as "never-eligible."
- Pre-treatment covariates `X_i` (support `X ⊆ R^d`), time-invariant.
- Potential outcomes (Robins 1986 sequential framework, simplified to staggered adoption): `Y_{i,t}(g)` = outcome if first treated at `g`; `Y_{i,t}(∞)` = never-treated outcome. Observed (Eq. 2.1):

  ```
  Y_{i,t} = Σ_{g ∈ G} 1{G_i = g} · Y_{i,t}(g)                                   (2.1)
  ```

- **Generalized propensity score** (§4.1): `p^{S=g,Q=1}_{g',q'}(X) ≡ Pr[S=g, Q=1 | X, (S=g,Q=1) ∪ (S=g',Q=q')]` — probability of being in the treated cell `(S=g, Q=1)` conditional on `X` and on being in one of the two cells being compared. This "sequence of two-group comparisons" device (Lechner 2002; Callaway–Sant'Anna 2021) is how the DR/IPW estimators are constructed.
- **Outcome regression**: `m^{S=g,Q=q}_{Y_t − Y_{t'}}(X) ≡ E[Y_t − Y_{t'} | S=g, Q=q, X]` — conditional expectation of the outcome *change*.

### Target Estimands (§2.1)

Group-time average treatment effect on the treated (Eq. 2.2):

```
ATT(g,t) ≡ E[Y_t(g) − Y_t(∞) | G_i = g] = E[Y_t(g) − Y_t(∞) | S_i = g, Q_i = 1]   (2.2)
```

Event-study (elapsed treatment time `e = t − g`) aggregation (Eq. 2.3) and its scalar average (Eq. 2.4):

```
ES(e) ≡ E[ ATT(G, G+e) | G+e ∈ [2,T] ]
      = Σ_{g ∈ G_trt} P(G=g | G+e ∈ [2,T]) · ATT(g, g+e)                          (2.3)

ES_avg ≡ (1/N_E) Σ_{e ∈ E} ES(e)                                                  (2.4)
```

The cohort-share weights `P(G=g | …)` are defined over `G_i`, which (by construction) is finite **only for `Q_i = 1` units** — i.e. the **eligible-treated** population. This is load-bearing for the aggregation-weight deviation noted under Implementation Notes.

### Identifying Assumptions (§2.2)

- **Assumption S (Random Sampling).** `{(Y_{i,1},…,Y_{i,T}, X_i', G_i, S_i, Q_i)}_{i=1}^n` is i.i.d.
- **Assumption SO (Strong Overlap).** For every `(g,q) ∈ S × {0,1}` and some `ε > 0`, `Pr[S=g, Q=q | X] > ε` w.p.1 — every `(g,q)` cell is populated at every `X`; rules out irregular identification (Khan–Tamer 2010).
- **Assumption NA (No-Anticipation).** For `g ∈ G_trt` and pre-treatment `t < g`: `E[Y_t(g) | S=g, Q=1, X] = E[Y_t(∞) | S=g, Q=1, X]` w.p.1. Lets pre-treatment periods serve as effectively-untreated.
- **Assumption DDD-CPT (DDD Conditional Parallel Trends)** — the key identifying restriction. For each `g ∈ G_trt`, `g' ∈ S`, and `t` with `t ≥ g` and `g' > max{g,t}`:

  ```
  E[Y_t(∞) − Y_{t-1}(∞) | S=g,  Q=1, X] − E[Y_t(∞) − Y_{t-1}(∞) | S=g,  Q=0, X]
    =
  E[Y_t(∞) − Y_{t-1}(∞) | S=g', Q=1, X] − E[Y_t(∞) − Y_{t-1}(∞) | S=g', Q=0, X]
  ```

  Interpretation (§2.2): DDD-CPT generalizes the unconditional two-period DDD parallel-trend of Olden–Møen (2022) to multiple periods, staggered adoption, and conditioning on `X`; it is the DDD analog of Callaway–Sant'Anna (2021)'s not-yet-treated conditional PT. Crucially it does **not** impose DiD-type PT *within* `S=g` (across `Q`) nor *across* treated groups — it only requires the *eligibility-gap trend* to be stable across enabling groups. This extra flexibility (allowing enabling-group- and partition-specific trend violations) is what makes DDD appealing — and is exactly why naive DiD-style shortcuts fail (below). Taking `X = 1` a.s. recovers the unconditional case.

---

### Why naive DDD fails — implications for practice (§3)

The paper motivates its estimators by showing standard practice breaks in two distinct ways.

**§3.1 — Two periods, covariates important.** With `t ∈ {1,2}`, `S ∈ {2, ∞}`, and no covariates, the 3WFE regression (Eq. 3.1) `Y_{i,t} = γ_i + γ_{s,t} + γ_{q,t} + β_3wfe·D_{i,t} + ε_{i,t}` recovers `ATT(2,2)`, and `β_3wfe` equals the **difference of two DiDs** (Eq. 3.2, Olden–Møen 2022):

```
β_3wfe = { E[Y2−Y1 | S=2, Q=1] − E[Y2−Y1 | S=2, Q=0] }      (DiD among S=2)
       − { E[Y2−Y1 | S=∞, Q=1] − E[Y2−Y1 | S=∞, Q=0] }      (DiD among S=∞)
       = ATT(2,2)         [only when X = 1 a.s.]                              (3.2)
```

When covariates **are** needed, three intuitive fixes **all fail** (Figure 1, n=5000, 1000 reps, true `ATT(2,2)=0`):
- (a) 3WFE with covariates interacted with post (Eq. 3.3) — biased;
- (b) Mundlak-device 3WFE: replace unit FE with S-by-Q FE plus linear `X` (Eq. 3.4) — biased;
- (c) difference of two **doubly-robust** DiDs (Sant'Anna–Zhao 2020) — biased.

The econometric reason (Remark 4.1): the DDD estimand must **integrate `X` over the covariate distribution of the *treated* units** (`S=2, Q=1`); proceeding as a difference of two DiDs integrates over the *untreated* distribution, producing bias. The fix is the **DR DDD estimator for `ATT(2,2)`** (Eq. 3.5) — which "cannot be expressed as the difference between two DR DiD estimators" but **is a function of three DR DiD estimators**, each using a different subset of untreated units as comparison:

```
ATT_dr(2,2) = E_n[ (ŵ^{S=2,Q=1}_trt − ŵ^{S=2,Q=0}_comp)(Y2 − Y1 − m̂^{S=2,Q=0}) ]
            + E_n[ (ŵ^{S=2,Q=1}_trt − ŵ^{S=∞,Q=1}_comp)(Y2 − Y1 − m̂^{S=∞,Q=1}) ]
            − E_n[ (ŵ^{S=2,Q=1}_trt − ŵ^{S=∞,Q=0}_comp)(Y2 − Y1 − m̂^{S=∞,Q=0}) ]    (3.5)
```

(Term 1: treated vs same-group-ineligible; Term 2: treated vs never-enabled-eligible; Term 3, subtracted: treated vs never-enabled-ineligible.)

**§3.2 — Staggered timing, even without covariates.** With `t ∈ {1,2,3}`, `S ∈ {2,3,∞}`, taking `X = 1`, the natural extension that **pools all not-yet-treated units** as the comparison group (Eq. 3.6, `ATT_cs-nyt`) is **systematically biased** for `ATT(g,t)` in DDD (Figure 2a — always negative when the truth is `+10`). Reason (Remark 4.2): DDD-CPT permits enabling-group- and partition-specific trends, and the eligible proportion `Q` can differ across enabling groups `S`; pooling not-yet-treated units conflates heterogeneous populations. The fix: **use one not-yet-enabled comparison cohort `g_c > t` at a time** (Eq. 3.7; `g_c = ∞` ⇒ never-enabled comparison), which yields an **over-identified** model, then **combine the valid comparisons by optimal GMM** (Eq. 3.8). CIs from the never-treated-only estimator are ≈ 50% wider than the GMM estimator that uses all not-yet-treated units.

---

### Identification (§4.1)

For `g_c ∈ S` with `g_c > max{g,t}` and post-treatment `t ≥ g`, the paper defines three estimands of `ATT(g,t)`:

- **DR DDD estimand (Eq. 4.1)** — the staggered analog of (3.5), a three-term combination using treated `(S=g, Q=1)` against `(S=g, Q=0)`, `(S=g_c, Q=1)`, and `(S=g_c, Q=0)`, with the generalized-propensity-score weights of Eq. 4.2.
- **RA DDD estimand (Eq. 4.3)** — outcome-regression-only form.
- **IPW DDD estimand (Eq. 4.4)** — propensity-weight-only form.

**Theorem 4.1 (nonparametric identification) — the first main result.** Under S, SO, NA, DDD-CPT, for all `g ∈ G_trt`, `t ∈ {2,…,T}`, and `g_c ∈ S` with `t ≥ g` and `g_c > t`:

```
ATT(g,t) = ATT_dr,gc(g,t) = ATT_ra,gc(g,t) = ATT_ipw,gc(g,t)                       (4.5)
```

This extends Callaway–Sant'Anna (2021) to DDD, and the RA / IPW / DR identification strands (Heckman-Ichimura-Todd 1997 / Abadie 2005 / Sant'Anna-Zhao 2020) to multi-period staggered DDD. DR is preferred: it is **Neyman-orthogonal** (Belloni et al. 2017) and hence more robust to nuisance misspecification.

**Corollary 4.1 (over-identification / multiple comparison groups).** Because every admissible `g_c` identifies the same `ATT(g,t)`, any weighted average over `G^c_{g,t} = {g_c ∈ S : g_c > max{g,t}}` with weights summing to one also identifies it:

```
ATT(g,t) = Σ_{g_c ∈ G^c_{g,t}} w^{g,t}_{g_c} · ATT_dr,gc(g,t),   Σ w = 1            (Cor. 4.1)
```

This opens the door to choosing weights for **minimum variance** (developed in §4.2 via GMM on re-centered influence functions).

Event-study identification follows (Remarks 4.3–4.4): post-treatment `ES(e)` (Eq. 4.6) and the pre-treatment analog (Eq. 4.7, with `ES(−1) = 0` by the baseline-period normalization at `g−1`), enabling event-study plots and Rambachan–Roth (2023) sensitivity analysis for DDD-CPT. Remark 4.5 notes the DR DDD estimand builds on an **efficient influence function** for two-period DDD (Appendix Lemma A.2).

### Estimation (§4.2)

Plug-in DR estimator — sample analog of (4.1), with estimated working models `m̂`, `p̂` (Eq. 4.8). Extends Callaway–Sant'Anna (2021)'s DR DiD to DDD; **consistent if, within each of the three DR-DiD components, *either* its outcome regression *or* its generalized propensity score is correctly specified** (multiply-robust — see Theorem 4.2).

Combining comparison groups (Corollary 4.1) gives `ATT_dr,ŵ(g,t) = ŵ'·ATT_dr(g,t)` (Eq. 4.9), where `ATT_dr(g,t)` is the `k_{g,t} × 1` vector over all valid `g_c`. The **minimum-variance weights** solve `min_w w'Ω̂_{g,t}w s.t. 1'w = 1` (Eq. 4.10), with closed form (Eq. 4.11) and the resulting **optimal GMM DDD estimator** (Eq. 4.12):

```
ŵ_gmm^{g,t} = Ω̂_{g,t}^{-1} 1 / (1' Ω̂_{g,t}^{-1} 1)                                (4.11)

ATT_dr,gmm(g,t) = ( 1' Ω̂_{g,t}^{-1} / (1' Ω̂_{g,t}^{-1} 1) ) · ATT_dr(g,t)          (4.12)
```

where `Ω̂_{g,t}` is a consistent estimate of the variance-covariance matrix of `ATT_dr(g,t)`. **Remark 4.6** shows (4.12) is the **optimal two-step GMM estimator based on re-centered influence functions** `RIF_dr,gc = IF_dr,gc + ATT_dr,gc`: the moment conditions `E[RIF_dr(g,t) − θ_{g,t}] = 0` (with `θ_{g,t} = ATT(g,t)`) yield exactly the efficient GMM weight `1'Ω^{-1}/1'Ω^{-1}1` (Newey–McFadden 1994).

Event-study and overall aggregations are plug-ins (Eqs. 4.13–4.14):

```
ES_dr,gmm(e) = Σ_{g ∈ G_trt} P_n(G=g | G+e ∈ [1,T]) · ATT_dr,gmm(g, g+e)            (4.13)

ES_avg,gmm   = (1/N_E) Σ_{e ∈ E} ES_dr,gmm(e)                                       (4.14)
```

(`ES_dr,gc` defined analogously by substituting the single-comparison estimator in 4.13.)

### Standard Errors / Inference (§4.2.1–§4.2.2)

Additional regularity assumptions: **WM (Working Model Conditions)**, **ALR (√n-Asymptotically Linear Representation)**, **IC (Integrability Conditions)** (with the nuisance pseudo-true parameters `κ` and the score `h^{g,t}_{g_c}`).

**Theorem 4.2 (Consistency & Asymptotic Normality) — pointwise inference.** Under S, SO, NA, DDD-CPT, WM, ALR, IC, both the single-comparison and the GMM estimators are √n-asymptotically normal with influence-function representations:

```
√n( ATT_dr,gc(g,t) − ATT(g,t) )   →d  N(0, Ω_{g,t,gc})
√n( ATT_dr,gmm(g,t) − ATT(g,t) )  →d  N(0, Ω_{g,t,gmm}),   Ω_{g,t,gmm} = (1'Ω_{g,t}^{-1}1)^{-1}
```

with the **efficiency ordering** `Ω_{g,t,gmm} ≤ Ω_{g,t,gc}` for every `g_c`, and `≤ Ω_{g,t,w}` for any summing-to-one weights `w` — i.e. the GMM combination is the minimum-variance combination. Theorem 4.2 also restates the **doubly / multiply robust** property: consistency holds provided each of the three DR-DiD components has a correct outcome regression *or* generalized propensity score — **8 (= 2³) working-model combinations** suffice.

**Remark 4.7 (simultaneous inference & clustering).** Pointwise results extend to hold simultaneously across all `(g,t)`; **simultaneous confidence bands** use the **multiplier bootstrap of Callaway–Sant'Anna (2021), Theorem 3 / Algorithm 1**, and **cluster-robust inference** follows their **Remark 10**. (The paper does not restate these to conserve space — they are inherited from the CS-2021 machinery.)

**Corollary 4.2 (event-study inference).** `ÊS_dr,gmm(e)` is √n-asymptotically normal with influence function `l^{es,e}_gmm` that accounts for estimation of the cohort-share weights `P_n(G=g | …)` (whose own IF `ξ^{g,e}` is given in Eq. 4.19). Inference for the overall `ES_avg,gmm` (4.14) follows by the **delta method**.

### Doubly / multiply robust property (§4.2.1, restated)

Each `ATT(g,t)` decomposes into **3 DR-DiD components**; within each component, correct specification of the outcome regression **or** the propensity score suffices. Hence `2 × 2 × 2 = 8` working-model combinations yield consistent DDD estimates — strictly more forgiving than a single DR DiD.

---

### Monte Carlo (§5) — reference

Two designs: (i) two periods with four covariates important for identification, `S ∈ {2, ∞}`; (ii) staggered, three periods, `S ∈ {2, 3, ∞}`, abstracting from covariates in the main text (covariate-staggered results in the Supplemental Appendix). Main-text comparison is graphical (density of point estimates; CI length per draw); the Supplement reports bias, RMSE, empirical 95% coverage, and average CI length over 1000 reps. Takeaway: the proposed DR / GMM DDD estimators are correctly centered and substantially more precise than 3WFE, difference-of-two-DiDs, and pooled-not-yet-treated alternatives.

### Empirical Illustrations (§6) — reference

Three applications quantify the practical stakes:
- **§6.1 Cai (2016)** — agricultural insurance on rural Chinese household financial decisions. The three-way fixed-effects estimator's confidence intervals are **up to 115% wider** than the proposed DR DDD estimator's.
- **§6.2 Cui, Zhang & Zheng (2018)** — emission-trading scheme on carbon emissions (staggered adoption; Eqs. 6.1–6.4). The proposed DR DDD estimates yield a **more modest, statistically insignificant** effect on the share of low-carbon patents, whereas 3WFE indicates a significant effect.
- **§6.3 Hansen & Wingender (2023)** — genetically-modified crop adoption on countrywide log-yields (Eq. 6.4 = the 3WFE event-study spec; estimates aggregated via Eq. 4.13, 999 bootstrap reps clustered at the country-crop level). Findings are **robust to dropping the never-enabling crop-country group** (≈ 15% yield increase) when the DDD estimator is used — unlike 3WFE, which shows much smaller effects and non-negligible pre-trends on that subsample.

### Concluding remarks / future work (§7)

The paper recommends practitioners **favor the optimal GMM DR DDD estimator** that pools information across comparison groups. Future directions explicitly named: data-adaptive / machine-learning nuisance estimation; **unbalanced panels and repeated cross-sections** (currently the framework assumes a balanced panel; cf. Abadie 2005, Callaway–Sant'Anna 2021, Sant'Anna–Xu 2023); **semiparametric efficiency bounds and efficient estimation in over-identified DDD** (citing **Chen, Sant'Anna & Xie 2025** — the same paper backing the library's `EfficientDiD`); and DDD-based persuasion effects (Jun–Lee 2024).

---

## Implementation Notes

### Data Structure Requirements
- Paper framework (§2): balanced panel with outcome `Y`, unit id, period, **enabling group `S_i`** (`first_treat`; `0`/`∞` for never-enabled), **binary time-invariant eligibility `Q_i`** (`eligibility`), and optional time-invariant covariates `X_i`. (The library's `StaggeredTripleDifference` uses a balanced panel; `TripleDifference` is repeated cross-section — see Relation to diff-diff below.)
- Both eligible (`Q=1`) and ineligible (`Q=0`) units are always required. A **never-enabled** cohort (`S=∞`) is required for `control_group="nevertreated"` and to identify the last-enabling cohort (whose only admissible comparison is never-enabled); otherwise later **not-yet-enabled** cohorts (`control_group="notyettreated"`, `g_c > t`) can serve as comparisons without a never-enabled cohort. The paper also accommodates the all-eventually-enabled case by trimming observations after `max S_i` and treating the last-enabling group as the effective never-eligible comparison (§2).

### Computational Considerations
- Per `(g, t)` cell: up to `k_{g,t}` DR-DiD fits (one per admissible comparison cohort `g_c`), each an OLS outcome regression plus a logistic generalized-propensity-score fit, combined via the closed-form GMM weights (Eq. 4.11) — no iterative optimization for the combination step.
- Influence functions are accumulated per unit so that `Ω̂_{g,t}` and the aggregation IFs (Cor. 4.2) are available analytically; simultaneous bands need the CS-2021 multiplier bootstrap (Remark 4.7).

### Tuning Parameters

| Parameter | Type | Paper prescription / example usage | Selection |
|-----------|------|-----------------------------------|-----------|
| Comparison cohorts `G^c_{g,t}` | set | `{g_c ∈ S : g_c > max(g,t)}` | All admissible not-yet-enabled cohorts + never-enabled |
| Comparison-group combination weights | vector | optimal GMM `Ω̂^{-1}1 / 1'Ω̂^{-1}1` (Eq. 4.11) | Minimum asymptotic variance |
| Estimation method | enum | DR (recommended) | RA / IPW / DR (Thm 4.1) |
| Bootstrap reps (simultaneous bands) | int | 999 (used in §6.3) | CS-2021 multiplier bootstrap |
| Clustering | level | application-specific (e.g. country-crop) | CS-2021 Remark 10 |

### Relation to Existing diff-diff Estimators
- **`TripleDifference`** (library, **Complete**) implements the **two-period DDD estimand and three-DR-DiD-component decomposition** of Eq. (3.5)/(4.1) with CS-2021-style influence-function SEs — this estimand and decomposition are what the paper is the primary source for. The library's path is **repeated cross-section** (`triplediff::ddd(panel=FALSE)`); the paper assumes a balanced panel and leaves repeated-cross-section / unbalanced data handling to future work (§7), so that data-path detail follows the companion `triplediff` package rather than the paper directly.
- **`StaggeredTripleDifference`** (library, **Complete**; DEPRECATED in 3.9, removed in 4.0 - the staggered case is served by `TripleDifference().fit(..., first_treat=)` since the phase-3(b) merge, ledger row M-013, running this same engine) implements the staggered / multi-period case — group-time `ATT(g,t)` via Eq. (4.1), the optimal-GMM combination across comparison cohorts (Eqs. 4.11–4.12), event-study via the CS aggregation mixin (Eq. 4.13), IF-based SEs, and a multiplier bootstrap for simultaneous bands. R reference `triplediff::ddd(panel = TRUE)` + `agg_ddd()`.
- Both reuse Callaway–Sant'Anna machinery (the DR-DiD building block, the cohort-share event-study aggregation, and the multiplier-bootstrap inference of Remark 4.7).

*The following are points where the **library's implementation choices** differ from the **paper and/or the R `triplediff` package** — for example, the comparison-cohort admissibility rule (a) differs from the paper and (b) matches R. They are recorded here for the implementation record and are formalized as REGISTRY deviations separately; the methodology summary above sources only from the paper.*
- **Comparison-cohort admissibility.** The paper's Theorem 4.1 / Corollary 4.1 define admissible comparison cohorts as `g_c > max(g, t)`. The library follows the R `triplediff` convention `g_c > max(t, base_period) + anticipation`; the two can differ for pre-treatment cells when a later cohort lies in `(t, g)`.
- **Aggregation weights.** The paper's `ES(e)` (Eqs. 2.3 / 4.13) weights cohorts by `P(G=g | …)`, where `G_i` is finite only for eligible-treated (`Q=1`) units — i.e. weights use `P(S=g, Q=1)`. R's `agg_ddd()` uses `P(S=g)` (all enabling-group units, including ineligible). This is the source of the larger tolerance on aggregated quantities in the library's existing R cross-validation.
- **Per-cohort group-effect WIF.** The library propagates the weight-influence-function (WIF) term through per-cohort group aggregations (conservative), where R's `agg_ddd(type="group")` uses `wif=NULL`.
- **Cluster-robust analytical SEs** are contemplated by Remark 4.7 (via CS-2021 Remark 10) but are accepted-but-not-wired in the library (deferred).

---

## Gaps and Uncertainties

- **REGISTRY equation anchors verified accurate against v3.** The existing `## StaggeredTripleDifference` REGISTRY section cites *"Equation 4.1"* for the three-DiD decomposition, *Eqs. 4.11–4.12* for the GMM combination, *Eq. 4.13* for event-study, and *Eq. 4.14* for the overall aggregation. Cross-checking v3: **Eq. (4.1) is indeed the three-term DR DDD estimand** (Remark 4.1 confirms it "leads to a combination of three DiD estimands, not just two"); **(4.11)/(4.12)** are the optimal weights / GMM estimator; **(4.13)/(4.14)** the event-study / overall aggregations; and the *identification* equivalence is separately **Theorem 4.1 / Eq. (4.5)**. So no equation-number reconciliation is required for the citations — they verify. (The library's **overall** aggregation intentionally differs from Eq. 4.14: it uses a simple post-treatment `(g,t)`-average rather than averaging over event-study effects — a pre-existing implementation departure currently described in REGISTRY prose (its formalization under the `**Note:**` / `**Deviation from R:**` label convention is deferred to PR-B), not a citation error.)
- **Balanced-panel / no repeated-cross-section.** The paper's framework assumes a balanced panel (§2); unbalanced panels and repeated cross-sections are named as future work (§7). DDD with `panel = FALSE` in R reflects the two-period repeated-cross-section path; staggered RC is out of the paper's current scope.
- **`Ω̂_{g,t}` estimation detail.** The paper specifies the optimal GMM combination requires a consistent `Ω̂_{g,t}` (Eq. 4.10) but, in the main text, leaves the finite-sample estimator of the comparison-group cross-covariance to the influence-function machinery; implementers should consult the asymptotic-theory derivations (§4.2.1, Theorem 4.2) and Appendix A for the exact `ψ`/IF forms. Page reference: Theorem 4.2 (p. 23) and Lemma A.2 (Appendix A).
- **Machine-learning nuisances.** The paper allows richer (e.g. ML) working models for `m̂`, `p̂` (citing Ahrens et al. 2025) but the asymptotic theory in v3 is stated for parametric working models under Assumption WM; ML nuisances are flagged as future work.
