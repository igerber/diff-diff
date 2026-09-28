# Paper Review: Covariate-Balanced Weighted Stacked Difference-in-Differences

**Authors:** Vadim Ustyuzhanin (HSE University)
**Citation:** Ustyuzhanin, V. (2026). *Covariate-Balanced Weighted Stacked Difference-in-Differences.* arXiv:2604.02293v1 [econ.EM], submitted 2 Apr 2026.
**PDF reviewed:** **arXiv:2604.02293v1** (https://arxiv.org/abs/2604.02293v1), dated April 2, 2026, 16 pages (content §1–§8 on pp. 1–15; References pp. 15–16). Per the project's PDFs-never-committed convention the local PDF is kept outside the repository; the arXiv v1 page is the authoritative source. The paper **does not number its equations, propositions, or theorems** — it has only labeled **Assumptions** (1–4 for the absorbing design; R1–R6 for the repeated-treatment design) and no algorithm boxes. **All references below are therefore pinned to section numbers (§2.1, §3.1, …) of v1**, which is the only arXiv version.
**Review date:** 2026-06-06

---

## Methodology Registry Entry

*This file is the canonical **in-repo scholarly paper review** for arXiv:2604.02293v1 — the **Step-1 fidelity artifact** (PR-A) for prospective CBWSDID support in diff-diff. There is **no existing `docs/methodology/REGISTRY.md` entry** for this method yet; PR-A is **docs-only** (touching only this review file, following the SyntheticControl PR-A #497 scope precedent). **The implementation packaging is an open PR-B decision and is deliberately not committed here:** because at `b_{sa}=1` the estimator reduces to weighted stacked DID, it can be realized either as (a) a **new estimator class** or (b) a **covariate-balancing (`b_{sa}`) path on the existing `StackedDiD`** — the latter being attractive because CBWSDID's mechanism is control *reweighting* (estimand-preserving under treatment-effect heterogeneity — see the §3.1 "estimand unchanged" note), not outcome-regression adjustment. Implementation, an R `cbwsdid` cross-language parity anchor, tests, and the REGISTRY/`doc-deps.yaml`/`references.rst` wiring are deferred to **PR-B** (planned separately, and contingent on the still-open decision of whether to pursue PR-B at all given this is a single-author 2026 preprint — see Gaps). This review records **paper facts only** and makes **no** code-deviation verdicts. The proposed class name (`CBWSDID` / `CovariateBalancedStackedDiD`) and the new-class-vs-`StackedDiD`-extension choice are PR-B decisions, not locked here.*

## CBWSDID

**Primary source:** [Ustyuzhanin, V. (2026). *Covariate-Balanced Weighted Stacked Difference-in-Differences*, arXiv:2604.02293v1.](https://arxiv.org/abs/2604.02293v1) R package `cbwsdid` (https://github.com/vadvu/cbwsdid).

**Central thesis (Abstract; §1):** Stacked DID reorganizes staggered-adoption data into cohort-specific sub-experiments and so avoids the bad-comparison problems of TWFE event studies, but it still leaves **two distinct design problems**. (1) **Across sub-experiments:** the ordinary stacked estimator aggregates the treated and control sides with *different* cohort shares, so untreated trends do not cancel even under within-sub-experiment parallel trends — Wing, Freedman & Hollingsworth (2024) fix this with *corrective stacked weights* that recover the trimmed aggregate ATT. (2) **Within a sub-experiment:** treated and clean-control units may still differ substantially on lagged outcomes or other pre-treatment characteristics, so correct aggregation alone does not guarantee credible treated-control comparisons. CBWSDID adds a **within-sub-experiment design stage** — matching *or* balancing weights — *before* the corrective aggregation stage, representing both refinement families through nonnegative first-stage **design weights** `b_{sa}` that compose with the Wing et al. corrective weights into a single weighted-least-squares stacked estimator. The estimator (a) accommodates matching- and weighting-based refinement within one estimator; (b) extends the logic beyond absorbing `0→1` adoption to **repeated `0→1` (and `1→0`) episodes** under a finite-memory assumption, bridging weighted stacked DID and episode-based panel matching (PanelMatch, Imai et al. 2023); and (c) ships an R package `cbwsdid`. The author positions CBWSDID as a **bridge** between modern DID estimators and design-based panel matching — it preserves the transparent estimand and aggregation logic of weighted stacked DID while importing the design sensitivity of matching/weighting (§8).

**Scope of what a library implementation would target:** the **absorbing staggered-adoption core** (§2–§3): the covariate-balanced cohort-specific DID `DID^b_{a,e}`, the final stacked weights `W_{sa}`, the pooled estimator `DID^{CBWSDID}_e`, and the weighted stacked event-study regression that returns `β_e`. The **repeated-treatment extension** (§4) and the **matching-based** refinement family are natural follow-ons (see PR-B scope pointers below).

**Key implementation requirements:**

### Setup and notation (§2.1)

- Groups `s = 1,…,S` indexed; calendar time `t = 1,…,T`. Treatment is staggered: each group has a first-treatment time `A_s ∈ {1,…,T,∞}`, `A_s=∞` = never treated.
- Potential outcomes: `Y_{s,t}(0)` = untreated potential outcome; `Y_{s,t}(a)` = potential outcome at time `t` if group `s` first adopts in period `a` and remains treated thereafter.
- For a cohort first treated in period `a` and event time `e = t − a`, the group-time ATT is:

      ATT(a, a+e) = E[ Y_{s,a+e}(a) − Y_{s,a+e}(0) | A_s = a ]

- Fix a **uniform event window** `κ = (κ_pre, κ_post)`. For each sub-experiment `a`, the treated and **clean-control** sets are:

      D_a = { s : A_s = a }
      C_a = { s : A_s > a + κ_post }

  The clean-control definition is *stronger* than "not yet treated at event time `e`": it keeps the control group fixed across the **entire** event window. Let `Ω_κ` = the set of (trimmed) treated cohorts `a` for which the full event window is observed and the clean-control rule is feasible (`C_a ≠ ∅`). A control unit may appear in **more than one** sub-experiment.
- Counts:

      N^D_a = ||D_a||,   N^C_a = ||C_a||,
      N^D_{Ω_κ} = Σ_{a∈Ω_κ} N^D_a,   N^C_{Ω_κ} = Σ_{a∈Ω_κ} N^C_a

- **Target parameter** — the trimmed aggregate ATT of Wing et al. (2024):

      θ^e_κ = Σ_{a∈Ω_κ} ATT(a, a+e) × ( N^D_a / N^D_{Ω_κ} )

  i.e. cohort-specific effects averaged with **treated-cohort shares that are stable across event time**.
- For each admissible `a`, build a sub-experiment from `D_a ∪ C_a` over event times `e ∈ {−κ_pre,…,κ_post}` (calendar `t = a+e`). Change from the reference period `−1`:

      ΔY_{s,a+e} = Y_{s,a+e} − Y_{s,a−1}

- Treated and control mean changes, and the cohort-specific DID:

      Δ̄^D_{a,e} = (1/N^D_a) Σ_{s∈D_a} ΔY_{s,a+e}
      Δ̄^C_{a,e} = (1/N^C_a) Σ_{s∈C_a} ΔY_{s,a+e}
      DID_{a,e} = Δ̄^D_{a,e} − Δ̄^C_{a,e}

  Under no anticipation and within-sub-experiment parallel trends, `DID_{a,e}` identifies `ATT(a, a+e)`.

### Weighted stacked DID (§2.2) — the baseline being refined

- Ordinary stacked DID pools all treated and all clean-control observations:

      DID^{SDID}_e = Σ_{a∈Ω_κ} (N^D_a / N^D_{Ω_κ}) Δ̄^D_{a,e}
                   − Σ_{a∈Ω_κ} (N^C_a / N^C_{Ω_κ}) Δ̄^C_{a,e}

  The **central problem**: the treated side uses treated-cohort shares `N^D_a/N^D_{Ω_κ}`, while the control side uses *different* shares `N^C_a/N^C_{Ω_κ}` — so untreated trends do **not** generally cancel even if parallel trends holds within each sub-experiment.
- Wing et al. (2024) **corrective weights** assign each control observation in sub-experiment `a` the sample weight:

      Q_{sa} = 1,                                         if s ∈ D_a
      Q_{sa} = (N^D_a / N^D_{Ω_κ}) / (N^C_a / N^C_{Ω_κ}),  if s ∈ C_a

  Re-aggregating the control trend with the *same* cohort shares as the treated trend gives:

      DID^{WSDID}_e = Σ_{a∈Ω_κ} (N^D_a/N^D_{Ω_κ}) Δ̄^D_{a,e}
                    − Σ_{a∈Ω_κ} (N^D_a/N^D_{Ω_κ}) Δ̄^C_{a,e}
                    = Σ_{a∈Ω_κ} (N^D_a/N^D_{Ω_κ}) DID_{a,e}

  which identifies `θ^e_κ`.
- **WLS implementation / regression spec (§2.2).** Re-index unit `s` as `sa` (a unit *within* a sub-experiment). The model is:

      Y_{sa,e} = α_{sa} + γ_{ae} + Σ_{h=κ_pre, h≠−1}^{κ_post} β_h · D_{sa} · 1{e=h} + ε_{sa,e}

  weighted by `Q_{sa}`, where:
  - `α_{sa}` = **unit-by-sub-experiment fixed effects** (each unit's *appearance* in a sub-experiment gets its own intercept — the paper labels `α_{sa}` and `γ_{ae}` jointly as "sub-experiment fixed effects", but `α_{sa}` is indexed by unit `s` within sub-experiment `a`);
  - `γ_{ae}` = **sub-experiment-by-event-time fixed effects** (indexed by sub-experiment `a` and event time `e`);
  - `D_{sa}` = a **time-invariant** indicator of treatment in sub-experiment `a`;
  - the dynamic coefficients `β_e` are the trimmed aggregate ATT at event time `e` (the reference period `−1` is omitted).
  - **Summation range / `κ_pre` sign (verified against arXiv v1 — paper notation is faithfully reproduced here, but is internally inconsistent):** the paper writes the lower limit verbatim as `h=κ_pre` (no minus), and its numeric examples write the window as `κ=(−3,2)`, `(−10,5)`, `(−4,10)` — i.e. **`κ_pre` is the *signed, negative* lower event-time bound**. So the sum spans **all** event times from `κ_pre (< 0)` to `κ_post`, including the pre-treatment **leads** `h ∈ {κ_pre,…,−2}` and the lags `h ∈ {0,…,κ_post}`, omitting only the reference period `−1` (it does **not** drop the leads). ⚠️ The §2.1 prose and the §4.1 reversal sentence instead write the event-time set as `{−κ_pre,…,κ_post}`, treating `κ_pre` as a *positive length* — the opposite convention. This `κ_pre`-sign inconsistency is the paper's, not a transcription error; the numeric examples are authoritative (`κ_pre < 0`). See *Gaps and Uncertainties*.
  - **This saturated two-way-FE-within-the-stack specification is the key structural delta from diff-diff's existing means-based `StackedDiD`** — see *Relation to Existing diff-diff Estimators* below.

### Covariate-balanced design weights and final stacked weights (§3.1) — the contribution

- The baseline assumes `Δ̄^C_{a,e}` is already a credible estimate of the treated cohort's untreated trend. CBWSDID adds a within-sub-experiment **design stage** when that is implausible. For each admissible cohort `a ∈ Ω_κ`, let `X_{sa}` be a vector of **pre-treatment characteristics constructed only from information at `t ≤ a−1`** (baseline covariates, lagged outcomes, pre-trend summaries). Using `D_a ∪ C_a`, construct **nonnegative design weights `b_{sa}` for control units `s ∈ C_a`**. Treated units are left unchanged (they keep unit weight). The weighted control mean change and the covariate-balanced cohort-specific DID:

      Δ̄^{C,b}_{a,e} = ( Σ_{s∈C_a} b_{sa} ΔY_{s,a+e} ) / ( Σ_{s∈C_a} b_{sa} )
      DID^b_{a,e}   = Δ̄^D_{a,e} − Δ̄^{C,b}_{a,e}

- **Effective control mass** (controls are reweighted; treated are not, so `Ñ^D_a = N^D_a`, `Ñ^D_{Ω_κ} = N^D_{Ω_κ}`):

      Ñ^C_a = Σ_{s∈C_a} b_{sa},   Ñ^C_{Ω_κ} = Σ_{a∈Ω_κ} Ñ^C_a

- **Final sample weights** `W_{sa}` (the composition of design + corrective weights):

      W_{sa} = 1,                                              if s ∈ D_a
      W_{sa} = b_{sa} × (N^D_a / N^D_{Ω_κ}) / (Ñ^C_a / Ñ^C_{Ω_κ}),  if s ∈ C_a

- **Scale invariance of `b_{sa}`.** It is **not** necessary to normalize `b_{sa}` to sum to one within a sub-experiment — `Δ̄^{C,b}_{a,e}` is already normalized by `Σ b_{sa}`, so replacing `b_{sa}` by `c_a·b_{sa}` for any positive constant `c_a` leaves the within-sub-experiment DID unchanged. Across sub-experiments the normalization role is played by the effective control mass and the corrective factor:

      Σ_{s∈C_a} W_{sa} = Ñ^C_{Ω_κ} × (N^D_a / N^D_{Ω_κ})

  so the total weighted control mass in sub-experiment `a` is automatically proportional to the treated-cohort share, regardless of the raw scale of `b_{sa}`.
- **Pooled CBWSDID estimator** (treated-share average of within-sub-experiment balanced DID contrasts):

      DID^{CBWSDID}_e = Σ_{a∈Ω_κ} (N^D_a / N^D_{Ω_κ}) DID^b_{a,e}

- **Estimand is unchanged by refinement.** Because the treated side of each cohort is left unchanged and the full set of treated cohorts `Ω_κ` is retained, the target parameter remains `θ^e_κ = Σ_{a∈Ω_κ} ATT(a,a+e)·(N^D_a/N^D_{Ω_κ})`. The estimator only changes *how the untreated counterfactual trends are estimated*, not the cohort weights in the aggregate ATT. ⚠️ **More aggressive preprocessing that discards treated units or entire treated cohorts changes the estimand to something like an overlap-trimmed ATT** — i.e. *control-only* reweighting preserves the estimand; *treated*-side trimming does not.
- **How to construct `b_{sa}` — matching OR weighting** (Stuart 2010; Austin & Stuart 2015), both theoretically equivalent here because both produce a nonnegative control weight `b_{sa}` for each `s ∈ C_a`. **With no covariate adjustment, `b_{sa}=1` for all clean controls, which returns the Wing et al. (2024) weighted stacked DID** (CBWSDID nests WSDID). Concretely:
  - *Matching:* NN without replacement → `b_{sa}` binary (matched = 1, unmatched = 0); NN with replacement → integer counts (how often a control is reused) or normalized versions; optimal/full matching → nonnegative design weights.
  - *Weighting:* `b_{sa}` is a continuous balancing weight from IPW, **entropy balancing**, overlap weighting, CBPS, or related methods.
  - The estimator does **not** depend on a particular refinement method.

*Assumption checks / warnings — absorbing staggered-adoption design (§3.2):*

- **Assumption 1 (No anticipation):** for every admissible `a ∈ Ω_κ`, `ATT(a, a+e) = 0` for all `e < 0`.
- **Assumption 2 (Within-sub-experiment weighted parallel trends):** for every `a ∈ Ω_κ` and every `e ∈ {−κ_pre,…,κ_post}`,

      E[ Y_{s,a+e}(0) − Y_{s,a−1}(0) | A_s = a ]
        = E_{b_a}[ Y_{s,a+e}(0) − Y_{s,a−1}(0) | s ∈ C_a ]

  (the `b_a`-weighted clean-control untreated trend equals the treated cohort's expected untreated trend). This **replaces unconditional parallel trends with a sub-experiment-specific, design-weighted version**.
- **Assumption 3 (Overlap and nondegeneracy):** for every `a ∈ Ω_κ`, the treated and clean-control sets are nonempty and `0 < Ñ^C_a < ∞`.
- **Assumption 4 (Pre-treatment refinement):** `b_{sa}` is constructed **only from information dated before `a−1`**, so no post-treatment bias is introduced. ⚠️ This is the operational guard the implementation must enforce: covariate construction (and any matching/weighting fit) may read only `t ≤ a−1` data.

Under Assumptions 1–4, `DID^b_{a,e} = ATT(a, a+e)` for each admissible `a, e`, hence `DID^{CBWSDID}_e = θ^e_κ`. Equivalently, the coefficient `β_e` from the `W_{sa}`-weighted stacked event-study regression identifies the trimmed aggregate ATT at event time `e`. The author summarizes CBWSDID as combining the **covariate-adjustment logic of Callaway & Sant'Anna (2021) within sub-experiments** with the **aggregation logic of Wing et al. (2024) across sub-experiments**.

*Estimator equation — repeated `0→1` / `1→0` extension (§4), if implemented:*

- Binary treatment `D_{s,t} ∈ {0,1}` may switch on, off, and on again. Introduce a history window length `L ≥ 1`; recent treatment history `H^{(L)}_{s,τ} = (D_{s,τ−L},…,D_{s,τ−1}) ∈ {0,1}^L`. For a history profile `h` and switch time `τ`:

      D^{01}_{τ,h} = { s : H^{(L)}_{sτ}=h, D_{s,τ−1}=0, D_{s,τ+r}=1 ∀ r=0,…,κ_post }   (treated switch-on episodes)
      C^0_{τ,h}    = { s : H^{(L)}_{sτ}=h, D_{s,τ+r}=0 ∀ r=0,…,κ_post }              (stable untreated control episodes)

  `L` is **distinct from `κ_pre`**: `κ_pre` controls displayed pre-period effects; `L` controls how much recent treatment history is held fixed when constructing comparable switch-on episodes. Prior treatment/reversals may occur elsewhere in the panel and may appear in the recent-history vector `H^{(L)}` (that vector is precisely how prior paths are conditioned on); the §4.1 prose states verbatim that such reversals do "not [occur] in `{−κ_pre,…,κ_post}`". ⚠️ **This prose is stronger than the formal episode-set definitions and Assumption R3, which constrain treatment status only over the *post-treatment* window** `r=0,…,κ_post` (treated: `D_{s,τ−1}=0` and `D_{s,τ+r}=1`; control: `D_{s,τ+r}=0`) — neither the set definitions above nor R3 explicitly restrict the lead window `r ∈ {κ_pre,…,−2}`. A literal "no reversal anywhere in the full window" reading would exclude valid prior-exposure histories and change the episode-weighted estimand; the formal definitions + R3 are the operative identification objects. See *Gaps and Uncertainties*. The control set is indexed by **stable untreated episodes, not never-treated units** — deliberately, since in a non-absorbing design what matters is local-window comparability, not lifetime never-treatment.
- A **finite-memory restriction** (Assumption R1) makes the extension operational: in the window of interest, potential outcomes depend on the pre-`τ` path only through `H^{(L)}_{sτ}`. The history-conditional treatment effect:

      ATT^{01}(τ, h, τ+e) = E[ Y_{s,τ+e}(h→1) − Y_{s,τ+e}(h→0) | s ∈ D^{01}_{τ,h} ]

- Reversals (`1→0`) are handled **mechanically** by setting `D^{1→0} = 1 − D^{0→1}`; estimation is otherwise unchanged.
- **Episode-weighted** (not unit-weighted) estimand. With `M^D_{τ,h}=||D^{01}_{τ,h}||`, `M^C_{τ,h}=||C^0_{τ,h}||`, `M^D_Ω = Σ M^D_{τ,h}` over `Ω^{01}_{L,κ}`:

      θ^{01}_e(L, κ) = Σ_{(τ,h)∈Ω^{01}_{L,κ}} ATT^{01}(τ, h, τ+e) × (M^D_{τ,h} / M^D_Ω)

  ⚠️ This is **episode-weighted**: a unit contributing multiple admissible episodes gets more total weight. It answers "the average effect of an admissible switch-on episode," **not** "the average effect for a switching unit." A unit-weighted alternative is acknowledged but **not developed** in the paper.
- Episode-level stacked weights and pooled estimator (with `M̃^C_{τ,h}=Σ_{s∈C^0_{τ,h}} b_{s,τ,h}`, `M̃^C_Ω = Σ M̃^C_{τ,h}`):

      W^{01}_{s,τ,h} = 1,                                                      if s ∈ D^{01}_{τ,h}
      W^{01}_{s,τ,h} = b_{s,τ,h} × (M^D_{τ,h}/M^D_Ω) / (M̃^C_{τ,h}/M̃^C_Ω),     if s ∈ C^0_{τ,h}

      Δ̄^{C,b}_{τ,h,e} = ( Σ_{s∈C^0_{τ,h}} b_{s,τ,h}(Y_{s,τ+e} − Y_{s,τ−1}) ) / ( Σ_{s∈C^0_{τ,h}} b_{s,τ,h} )
      DID^{01,CBWSDID}_e = Σ_{(τ,h)∈Ω^{01}_{L,κ}} (M^D_{τ,h}/M^D_Ω) ( Δ̄^D_{τ,h,e} − Δ̄^{C,b}_{τ,h,e} )

*Assumption checks / warnings — repeated-treatment design (§4.3):* parallel Assumptions 1–4 with the conditioning object changing from cohort `a` to episode type `(τ,h)`, plus the finite-memory restriction:
- **R1 (Finite memory):** potential outcomes in the window depend on the pre-`τ` path only through `H^{(L)}_{sτ}`.
- **R2 (No anticipation of switch-on episodes):** `E[Y_{s,τ+e}(h→1) − Y_{s,τ+e}(h→0) | s∈D^{01}_{τ,h}] = 0` for all `e<0`.
- **R3 (Stable episode design):** treated episodes stay treated and control episodes stay untreated throughout the post-treatment event window. *Can be relaxed* to onset-only (`e=0`) treated episodes that may subsequently reverse within the window (PanelMatch-style).
- **R4 (History-conditional weighted parallel trends):** `E[Y_{s,τ+e}(h→0) − Y_{s,τ−1}(h→0) | s∈D^{01}_{τ,h}] = E_{b_{τ,h}}[ … | s∈C^0_{τ,h}]`.
- **R5 (Episode-level overlap and nondegeneracy):** treated and stable-untreated episode sets nonempty and `0 < M̃^C_{τ,h} < ∞`.
- **R6 (Pre-treatment design and episode invariance):** `b_{s,τ,h}` built only from info dated before `τ−1`, and held **fixed across all event times** within a fixed admissible episode type.

Under R1–R6, `DID^{01,CBWSDID}_e = θ^{01}_e(L,κ)`. ⚠️ **Important control-eligibility implication:** a never-treated unit has an all-zero history in every lag window, so under exact/near-exact history matching never-treated units are admissible controls **only** for switch-on episodes whose recent history is also all zeros — they are *not* generally valid controls for episodes that occur after earlier exposures. Later switchers / previously treated units *can* serve as controls if, at `τ`, they are untreated throughout the event window and share the relevant recent history.

*Standard errors (§5):*
- **Default:** condition on the estimated design weights and treat the final weights `W_{sa}` / `W^{01}_{s,τ,h}` as **fixed** in the second-stage regression → a **cluster-robust variance estimator for the weighted stacked regression, clustered at the unit level `s`** (the same unit appears multiple times in the stacked sample, so unit-level clustering handles that dependence). This conditional-on-weights approach is close to how PanelMatch computes SEs (conditional on the matching-implied weights rather than treating the whole matching procedure as part of the stochastic expansion) (Imai et al. 2023). The repeated-treatment extension makes within-unit dependence even more salient (a unit can contribute multiple episodes), *further supporting unit-level clustering*.
  - Per the underlying WSDID spec (§2.2): variance is based on observations being dependent within units `s` but independent across them; clustering on `sa` is also possible, but (Wing et al. 2024) both give the same desired rejection rates (both suffer from a small number of clusters), and clustering on `s` is preferred to additionally account for repeated observations across sub-experiments.
- **Bootstrap:**
  - *Cluster bootstrap conditional on the design weights* — resample units/groups, keep the first-stage matching/weighting structure **fixed**, recompute only the second-stage stacked estimator.
  - *Cluster bootstrap with re-estimation of smooth balancing weights* — when the first stage is **smooth** (balancing-weight estimators), rebuild the first-stage design weights in each replication to more fully propagate first-stage estimation uncertainty.
- ⚠️ **Caveat (nonsmooth matching):** for nonsmooth matching estimators, the standard bootstrap that recomputes NN (or similar) matching in each replication is **not generally valid** (Abadie & Imbens 2008). → For matching-based refinement, prefer the conditional-on-weights variance or a conditional cluster bootstrap; do **not** re-run NN matching inside bootstrap replications.
- **Clustering level:** unit `s` (default and recommended).

*Edge cases:*
- **No covariate adjustment** (`b_{sa}=1` ∀ clean controls): reduces to Wing et al. (2024) weighted stacked DID — a useful equivalence/regression test target against the library `StackedDiD` (exact under the matching Q-weight count convention / on balanced stacks; see *Relation to Existing diff-diff Estimators*).
- **Empty clean-control set for a cohort** (`C_a = ∅`): cohort `a` is not in `Ω_κ` (excluded by construction / trimming).
- **Degenerate design weights** (`Ñ^C_a = 0`, e.g. all `b_{sa}=0`): violates Assumption 3 / R5 → cohort/episode is non-identified; must be detected and excluded or errored.
- **Control reused across sub-experiments:** legitimate and expected; do not de-duplicate. Drives the unit-level clustering requirement.
- **`b_{sa}` raw scale arbitrary within a sub-experiment:** the paper notes it is **not necessary** to normalize `b_{sa}` to sum to one within a cohort — the estimator is invariant to positive within-cohort rescaling (`b_{sa} → c_a·b_{sa}`, `c_a>0`), because `Δ̄^{C,b}_{a,e}` divides by `Σ b_{sa}` and the final weights are recomputed from the resulting `b_{sa}`. So within-cohort normalization is a **harmless implementation choice, not required and not prohibited**; only consistency (compute `Ñ^C_a` and `W_{sa}` from the same `b_{sa}`) matters.
- **Treated-side trimming / discarding cohorts:** silently changes the estimand to overlap-trimmed ATT — should be surfaced, not done silently.

*Algorithm (synthesized from §2–§3; the paper has NO numbered algorithm box):*
1. Trim cohorts to `Ω_κ`: keep cohorts `a` with the full event window observed and a feasible clean-control set `C_a = {s : A_s > a + κ_post} ≠ ∅`.
2. For each `a ∈ Ω_κ`, build `X_{sa}` from `t ≤ a−1` data only (Assumption 4) and fit a within-sub-experiment refinement on `D_a ∪ C_a` to obtain nonnegative control design weights `b_{sa}` (treated units keep weight 1). With no refinement, set `b_{sa}=1`.
3. Compute the effective control mass `Ñ^C_a = Σ b_{sa}` and `Ñ^C_{Ω_κ}`; form the final weights `W_{sa}` (=1 for treated; `= b_{sa}·(N^D_a/N^D_{Ω_κ})/(Ñ^C_a/Ñ^C_{Ω_κ})` for controls).
4. Stack the sub-experiments and run the `W_{sa}`-weighted event-study regression `Y_{sa,e} = α_{sa} + γ_{ae} + Σ_h β_h D_{sa}1{e=h} + ε`; the `β_e` are the event-time aggregate ATTs. (Equivalently, compute `DID^{CBWSDID}_e = Σ_a (N^D_a/N^D_{Ω_κ}) DID^b_{a,e}` directly.)
5. Variance: cluster-robust at unit `s`, conditional on the estimated `W_{sa}` (or a conditional/smooth-weight cluster bootstrap).
6. *Repeated-treatment variant:* replace cohorts `a` by episode types `(τ,h)` with history window `L`; use `D^{01}_{τ,h}`, `C^0_{τ,h}`, the episode-weighted shares `M^D_{τ,h}/M^D_Ω`, and `W^{01}_{s,τ,h}`.

**Reference implementation(s):**
- R: `cbwsdid` (Vadim Ustyuzhanin, https://github.com/vadvu/cbwsdid) — the **cross-language parity anchor for PR-B**. (Function/argument surface to be cataloged during PR-B against the package source; not transcribed here.)
- Related R packages the paper benchmarks against: `did` (Callaway & Sant'Anna), `fixest::sunab` (Sun & Abraham), `PanelMatch` (Imai et al.), and entropy-balancing / matching tooling.

**Requirements checklist (for PR-B):**
- [ ] Cohort trimming to `Ω_κ` with the clean-control rule `A_s > a + κ_post` over a uniform window `κ=(κ_pre,κ_post)`.
- [ ] Pre-treatment-only (`t ≤ a−1`) covariate / design-weight construction (Assumption 4 guard).
- [ ] At least one within-sub-experiment **weighting** refinement producing nonnegative `b_{sa}` (IPW or entropy balancing — see PR-B pointers).
- [ ] `b_{sa}=1` fallback that reduces to Wing et al. (2024) WSDID (equivalence test vs the library `StackedDiD` under the matching count convention).
- [ ] Final-weight composition `W_{sa}` with effective control mass `Ñ^C_a` (no within-cohort `b` normalization required).
- [ ] Pooled `DID^{CBWSDID}_e` and/or the `W_{sa}`-weighted two-way-FE-within-stack event-study regression returning `β_e`.
- [ ] Unit-level (`s`) cluster-robust SEs conditional on `W_{sa}`; documented bootstrap option(s); nonsmooth-matching bootstrap caveat honored.
- [ ] R `cbwsdid` numerical parity anchor.
- [ ] (Deferred candidate) repeated `0→1`/`1→0` episode extension with history window `L` and R1–R6.

---

## Implementation Notes

### Data Structure Requirements
- **Balanced or unbalanced panel** with unit id `s`, calendar time `t`, outcome `Y_{s,t}`, and either a first-treatment time `A_s` (absorbing) or a binary treatment path `D_{s,t}` (repeated). The absorbing core needs `A_s` (with `∞`/sentinel for never-treated).
- Pre-treatment covariates and/or lagged outcomes available at `t ≤ a−1` for every cohort `a` (the simulation uses 3 lags of `Y` plus baseline covariates; the empirical examples use 2–4 outcome lags plus covariate lags).
- A uniform event window `κ=(κ_pre,κ_post)` shared across cohorts (the clean-control set is defined relative to `κ_post`).
- For the repeated design: a history window length `L`, and the ability to compute `H^{(L)}_{s,τ}` lag vectors.

### Computational Considerations
- Two stages: (1) per-sub-experiment (or per-episode-type) refinement fit — the cost driver if matching/optimization is used; (2) a single weighted stacked regression on the pooled sample.
- Stacking duplicates control observations across sub-experiments, so the stacked design matrix is larger than the raw panel; the two-way-FE-within-stack regression should be implemented with within-transformation / sparse FE absorption (cf. diff-diff's `absorb` path) rather than dense dummies for `α_{sa}`/`γ_{ae}`.
- The repeated-treatment design can generate many `(τ,h)` episode types (up to `2^L` history profiles × switch times); episode enumeration and admissibility checks dominate.
- Bootstrap with first-stage re-estimation multiplies the refinement cost by the number of replications (only valid for *smooth* weighting, not matching).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `κ = (κ_pre, κ_post)` | event window (ints; `κ_pre < 0`) | application-specific, signed per the paper: sim `(−3,2)`; Trounstine `(−10,5)`; Acemoglu `(−4,10)` | substantive; governs trimming and the clean-control rule |
| refinement method | matching *or* weighting | none specified as universal default; **entropy balancing performed best in the paper's simulation** | by data/overlap; weighting (entropy balancing/IPW/CBPS/overlap) vs matching (NN±replacement, optimal, full) |
| `X_{sa}` (refinement covariates) | vector from `t ≤ a−1` | application-specific (lagged outcomes + baseline covariates; optional exact match on time-invariant discretes) | substantive; must be pre-treatment only |
| `L` (history window, repeated design only) | int `≥ 1` | application-specific (Acemoglu uses `L=4`) | how much recent treatment history to hold fixed |
| SE method | conditional cluster-robust / bootstrap | conditional cluster-robust at unit `s` | smooth-weight bootstrap optional; avoid NN re-matching in bootstrap |

### Relation to Existing diff-diff Estimators
- **`StackedDiD` (Wing, Freedman & Hollingsworth 2024) — the direct baseline.** At `b_{sa}=1` the CBWSDID estimator reduces to the **Wing et al. (2024) weighted stacked DID** (the paper-level `DID^{WSDID}_e` — the corrective `Q_{sa}` weights are the paper's, §2.2). ⚠️ **Whether that bit-for-bit reproduces the *library's* existing `StackedDiD` is a PR-B verification item, not an assertion here, because the Q-weight count convention differs:** the paper uses sub-experiment-level *unit* counts `N^D_a`, `N^C_a`; diff-diff's `StackedDiD` default ("aggregate" weighting, matching its R reference `compute_weights()`) uses *observation* counts per `(event_time, sub_exp)` — equal to the paper's unit counts **only on balanced stacks**, while diff-diff's `population`/`sample_share` weighting follows the paper's `N^D_a/N^C_a` directly (`diff_diff/stacked_did.py` `_compute_q_weights`). So the `b_{sa}=1` equivalence is exact under the matching count convention / on balanced panels; on unbalanced panels PR-B must decide whether CBWSDID follows the arXiv/R-`cbwsdid` sub-experiment-unit-count convention or diff-diff's event-time observation-count convention. ⚠️ **Structural delta to flag for PR-B:** the paper's WLS spec (§2.2) is a *saturated* regression with **unit-by-sub-experiment FE `α_{sa}` and sub-experiment-by-event-time FE `γ_{ae}`**, whereas diff-diff's `StackedDiD` is a **means-based weighted-pooled** event study (intercept + treatment dummy + event-time dummies + `D×event-time` interactions, carried by Q-weights, with **no** unit/time FE). A faithful CBWSDID either (a) adds the saturated FE on top of the stacked design, or (b) reuses the existing means-based machinery and verifies algebraic equivalence of the point estimates under the design weights. Determining which path matches the R `cbwsdid` output is a **PR-B** task — *not asserted here*.
- **`CallawaySantAnna` (2021).** CBWSDID's within-sub-experiment design weights play the same conditional-PT role as CS's covariate adjustment (the author makes this comparison explicitly in §3.2). CBWSDID differs by aggregating via Wing-style stacked corrective weights rather than CS's group-time aggregation, and by representing the adjustment as control-only design weights inside a stacked regression.
- **Inverse-propensity / balancing machinery.** diff-diff currently depends on numpy/pandas/scipy only and has **no** matching or balancing-weight machinery. The within-sub-experiment weighting stage (IPW / entropy balancing / CBPS / overlap weights) would be **new** code; matching (NN/optimal/full) would be newer still. This is the main net-new surface for PR-B.
- **`SpilloverDiD` / repeated-treatment.** The `0→1`/`1→0` episode extension overlaps conceptually with PanelMatch-style designs; diff-diff has no PanelMatch estimator, so the repeated extension is a larger, separable effort.

### PR-B scope pointers (neutral — not a plan, not a decision)
- **Packaging — leading candidate: a covariate-balancing path on the existing `StackedDiD`, not a new top-level estimator.** Because the estimator reduces to weighted stacked DID at `b_{sa}=1` and its refinement is control *reweighting* (estimand-preserving under heterogeneity — §3.1), it is naturally expressed as `StackedDiD` + a `b_{sa}` design-weight stage rather than a separate class. ⚠️ **Gating verification before committing to "extension":** confirm the `DID^{CBWSDID}_e` point estimate is recoverable in `StackedDiD`'s existing **means-based** machinery by folding `b_{sa}` into the `W_{sa}` regression weights (route (b) in *Relation to Existing diff-diff Estimators*), validated against R `cbwsdid` — especially on **unbalanced panels** and under the unit-count-vs-observation-count Q-weight convention. If that equivalence does not hold, the saturated-FE regression route (a) implies a larger change to `StackedDiD`'s internals and a standalone class becomes more attractive. Also check how `StackedDiD` currently computes SEs (analytic means-based vs WLS regression) to know how cleanly unit-clustered conditional-on-weights inference drops in.
- A natural **first** PR-B target is the **absorbing staggered-adoption core** with **one weighting-based balancing method** (IPW or entropy balancing, both implementable in scipy without new heavy dependencies), validated against R `cbwsdid` with the same refinement.
- **Matching-based** refinement (which needs distance metrics, replacement bookkeeping, optimal/full matching) and the **repeated `0→1`/`1→0` episode** extension are natural **later** increments and are likely **deferred** from a first implementation PR. The episode extension redefines the unit of analysis (episodes, not cohorts) and therefore **cannot** be a `StackedDiD` parameter even if the absorbing core is — it would be separate work regardless of the packaging choice.
- The `b_{sa}=1` ⇒ Wing et al. (2024) WSDID reduction gives a correctness anchor against the existing `StackedDiD` — exact under the matching Q-weight count convention (and on balanced stacks); the unbalanced-panel count convention is a PR-B decision (see *Relation to Existing diff-diff Estimators*).
- Naming, the new-class-vs-`StackedDiD`-extension packaging, the FE-vs-means-based regression-path decision, the exact R `cbwsdid` argument surface, and the deferral boundary are **PR-B decisions** to be made with the user. The new-class-vs-extension choice also bears on attribution: an extension leans on reputable components (Wing 2024 + entropy balancing/IPW + CS-style conditional PT) and cites Ustyuzhanin 2026 for the specific composition.

---

## Gaps and Uncertainties

- **Single-author 2026 preprint, not yet field-vetted.** arXiv:2604.02293v1 is a single-author working paper dated April 2, 2026, with no journal publication and (as of review) one arXiv version. It has **no numbered theorems/propositions/corollaries and no algorithm boxes** — identification is stated through labeled **Assumptions** (1–4, R1–R6) and the consistency conclusions `DID^b_{a,e}=ATT(a,a+e)` / `DID^{CBWSDID}_e=θ^e_κ` are asserted *under* those assumptions without formal proof in the main text. Treat the methodology as a faithful transcription of a preprint, **not** as a peer-reviewed result.
- **No equation numbers.** All equations are unnumbered in v1; this review references them by **section** (§2.1, §2.2, §3.1, §4.2, …). PR-B cross-references must use section numbers, not "Equation N."
- **`κ_pre` sign-convention inconsistency (§2.1/§2.2/§4.1 vs the numeric examples) — verified against arXiv v1.** The paper mixes two conventions for the event window `κ = (κ_pre, κ_post)`: the §2.2 WLS regression sum writes its lower limit verbatim as `h=κ_pre` (no minus) and **all** numeric examples write `κ=(−3,2)`/`(−10,5)`/`(−4,10)` — i.e. `κ_pre` is the *signed, negative* lower event-time bound; but §2.1 (event-time set `e ∈ {−κ_pre,…,κ_post}`) and the §4.1 reversal sentence (`… but not in {−κ_pre,…,κ_post}`) write `−κ_pre`, treating `κ_pre` as a *positive length*. A faithful implementation must pick one convention consistently; the **numeric examples are authoritative (`κ_pre < 0`)**, so the regression sum correctly spans all leads `{κ_pre,…,−2}` and lags `{0,…,κ_post}` minus the reference `−1`. (This is the paper's inconsistency, not a transcription error — the literal symbols are reproduced above.)
- **Repeated-treatment "no-reversal window" — prose vs formalization (§4.1 vs §4.2/§4.3).** The §4.1 prose says treatment reversals "may appear in the lag-history vector `H^{(L)}`, but **not in `{−κ_pre,…,κ_post}`**" (the full event window). But the formal episode-set definitions (`D^{01}_{τ,h}`: `D_{s,τ−1}=0`, `D_{s,τ+r}=1` for `r=0,…,κ_post`; `C^0_{τ,h}`: `D_{s,τ+r}=0` for `r=0,…,κ_post`) and **Assumption R3** restrict treatment status only over the **post-treatment** window `r=0,…,κ_post` (plus `τ−1` for treated) — they do not explicitly restrict the lead window `r ∈ {κ_pre,…,−2}`. The prose is therefore stronger than the operative identification conditions. PR-B must decide whether to enforce lead-window no-reversal (prose) or only post-window stability (set definitions + R3); the choice changes which prior-exposure episodes are admissible controls and hence the episode-weighted estimand. The formal definitions + R3 are the safer operative reading.
- **Regression FE labeling ambiguity (§2.2).** The paper writes "`α_{sa}` and `γ_{ae}` are sub-experiment fixed effects." Read literally with the `sa` re-indexing convention, `α_{sa}` is a *unit-by-sub-experiment* effect and `γ_{ae}` a *sub-experiment-by-event-time* effect (standard stacked two-way FE). PR-B should confirm the exact FE structure against the R `cbwsdid` source/`Y_{sa,e}` design before locking the regression path (esp. whether the library reuses the means-based `StackedDiD` design or adds saturated FE). p.3.
- **Design-weight estimation left to the user.** The paper deliberately does not commit to a single refinement; it lists matching (NN ±replacement, optimal, full) and weighting (IPW, entropy balancing, overlap, CBPS) as interchangeable for the estimator. The **specific estimator and its tuning** (distance metric, calipers, replacement, ratio; balancing moments, tolerance) are out of the paper's scope and must be pinned by the implementation/R-package. The simulation uses NN-with-replacement + Mahalanobis (ratio 4) and entropy balancing (Hainmueller 2012); Trounstine uses ratio 10; Acemoglu uses NN without replacement, 4 controls.
- **Unit- vs episode-weighting (§4.2).** The repeated-treatment estimand is **episode-weighted** by construction; the paper explicitly leaves the unit-weighted alternative *undeveloped*. Any unit-level interpretation in a repeated-treatment implementation would require an extra re-aggregation step not specified in the paper.
- **Variance is conditional-on-weights by default.** The default SE treats the estimated design weights as fixed — it does **not** propagate first-stage estimation uncertainty unless the (smooth-weight-only) re-estimating bootstrap is used. PR-B should document this as a known conservative/anti-conservative tradeoff and honor the Abadie–Imbens (2008) prohibition on bootstrapping nonsmooth matching. §5.
- **No finite-sample / small-cluster guidance beyond a pointer.** The paper notes (via Wing et al. 2024) that both clustering choices suffer from a small number of clusters but gives no small-sample correction recipe. p.3, §5.
- **Simulation/empirical numbers are illustrative, not validation targets.** Table 1 / Figures 2,4,6 are demonstrations on specific DGPs and replication datasets; they are not a parity suite. The authoritative numerical anchor for PR-B is the R `cbwsdid` package run on a shared dataset, not the paper's reported figures.
- **`Ω_κ` trimming interacts with the estimand.** Which cohorts survive trimming (full-window-observed + feasible clean control) defines `θ^e_κ`; different `κ` choices yield different (trimmed) target parameters. This is inherent to stacked DID but worth surfacing to users.
