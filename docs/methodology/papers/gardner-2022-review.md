# Paper Review: Two-stage differences in differences

**Authors:** John Gardner (Department of Economics, University of Mississippi)
**Citation:** Gardner, J. (2022). *Two-stage differences in differences.* arXiv:2207.05943 [econ.EM]. Working paper. URL: [https://arxiv.org/abs/2207.05943](https://arxiv.org/abs/2207.05943).
**PDF reviewed:** arXiv:2207.05943**v1** (`papers/gardner-2207.05943.pdf`, 34 PDF pages = printed pp. 1–33: content pp. 1–23, References pp. 24–25, Figures/Tables/Appendix C pp. 26–33, incl. Table 4 "Application event study estimates" on printed p. 33). Per the project's PDFs-never-committed convention the local PDF is kept outside the repo (gitignored `papers/`); arXiv is the authoritative source. The title page reads "This version: April 2021"; the arXiv stamp is "arXiv:2207.05943v1 [econ.EM] 13 Jul 2022", so the canonical citation year is **2022** (matching `docs/references.rst` and `diff_diff/two_stage.py`). All equation/section numbers below are this version's.
**Review date:** 2026-06-07

---

> **Scope note — full paper reviewed (no separate supplement).** Unlike the BJS (2024) REStud article, this is a self-contained working paper: all proofs (Appendix A), the reference Stata GMM implementation (Appendix B), and the full simulation/event-study tables (Appendix C) are **in this PDF**. There is therefore no supplement-only **GAP** class. Equations are numbered **(1)–(8)**; figures **1–4**; tables **1–4** (Table 4 = the Autor-application event-study estimates behind Figure 4, printed p. 33). **The paper contains no Gardner-numbered theorems or propositions** — Appendix A's consistency/unbiasedness results are prose proofs. The variance relies on **Newey & McFadden (1994), Theorem 6.1** (§3.3); §2.2 separately cites **de Chaisemartin & D'Haultfœuille (2020), Theorem 1** for the TWFE-weight representation. So there is no "Gardner Theorem 1" (the REGISTRY attribution of that name was a misattribution; see below). This matters for the REGISTRY reconciliation below.

> **Personal verification stamp (load-bearing items spot-checked against the source).** The reviewer read the full PDF (pp. 1–23 content) directly and verified verbatim: the parallel-trends decomposition eq. (1) and 2×2 regression eq. (2) (pp. 4–5); the DiD-estimand weights eq. (3) (p. 7); the two-stage procedure Steps 1–2 (pp. 8–9); the estimands eqs. (4)–(5) (p. 10); the event-study spec **eq. (6)** and Step 2′ (pp. 11–12); the inference §3.3 GMM moment condition and the **unnumbered** Newey-McFadden Theorem 6.1 sandwich (pp. 12–13); the Appendix A FWL derivation eqs. (7)–(8) (pp. 20–21); and the **Appendix B reference Stata GMM** (`vce(cluster id)`, `onestep`, `winitial(identity)`) (p. 23). **Two REGISTRY misattributions were found and corrected in this PR (see "Gaps and Uncertainties"):** (i) the `## TwoStageDiD` "Note on Equation 6 discrepancy" claimed the *paper's Equation 6* uses a per-cluster inverse `(D_c'D_c)^{-1}` — but **eq. (6) is the event-study regression specification, and the variance (unnumbered, §3.3) uses the GLOBAL Jacobian inverse** `E[∂f/∂(λ,γ,β)]^{-1}`, exactly what the library computes; (ii) an edge-case note attributed the first-stage variance correction to "Gardner 2022, Theorem 1", which does not exist (the paper has no Theorem 1).

---

## Methodology Registry Entry

*This file is the canonical **scholarly review** of the **primary source** (arXiv:2207.05943) for the library estimator `TwoStageDiD` (`two_stage.py`, `two_stage_bootstrap.py`, `two_stage_results.py`). It **supplements — it does not supersede** — the existing `docs/methodology/REGISTRY.md` `## TwoStageDiD` section, which remains the implementation contract. Heading/format aligned with REGISTRY conventions for ease of cross-reference.*

## TwoStageDiD

**Primary source:** Gardner (2022), *Two-stage differences in differences*, arXiv:2207.05943.

**Central thesis (§1; Abstract):** When treatment adoption is staggered and average effects are heterogeneous across groups and periods, the usual difference-in-differences (and event-study) regression is **misspecified** for conditional mean outcomes — outcomes are not linear in group, period, and treatment status — so its coefficient is a hard-to-interpret (possibly negatively) weighted average of group×period effects (§2.2, eq. (3); following de Chaisemartin & D'Haultfœuille 2020, Goodman-Bacon 2018, Sun & Abraham 2020). Gardner's fix exploits a simple observation: **under parallel trends, untreated potential outcomes are linear in group and period effects.** So estimate those effects from the *untreated* subsample (Stage 1), subtract them from observed outcomes, and regress the adjusted outcomes on treatment status (Stage 2). This recovers the overall ATT `E(β_gp | D_gp = 1)` even under arbitrary heterogeneity, is trivial to implement, and yields valid asymptotic SEs via a joint-GMM reading of the two stages.

**Key implementation requirements:**

*Setup & notation (§2.1):*
- Units `i`, calendar time `t`. Treatment **groups** `g ∈ {0,1,…,G}` and **periods** `p ∈ {0,1,…,P}` defined by staggered adoption: group `0` is untreated in all periods, group `1` is treated from period `1`, groups `1,2` from period `2`, etc. (group = cohort; period = treatment-onset epoch).
- `Y_gpit`, `Y_1gpit`, `Y_0gpit` = observed, treated, and untreated potential outcomes for the `i`th member of group `g` at time `t` of period `p`.
- `D_gp` = indicator that members of group `g` are treated in period `p`.
- `β_gp = E(Y_1gpit − Y_0gpit | g, p)` = the **group×period average treatment effect**.

*Identifying assumption — Parallel trends, eq. (1), p. 4:*
```
E(Y_gpit | g, p, D_gp) = λ_g + γ_p + β_gp D_gp          ... (1)
```
- Absent treatment, mean untreated outcomes decompose additively into group effects `λ_g` and period effects `γ_p`. This is the *only* identifying assumption; everything else is estimation.

*The DiD-regression problem (§2.2):* the 2×2 regression `Y_gpit = λ_g + γ_p + β D_gp + ε` (eq. (2)) has probability limit
```
β* = Σ_{g=1}^{G} Σ_{p=g}^{P} ω_gp β_gp ,   ω_gp per eq. (3)          ... (3)
```
with weights `ω_gp` that can be **negative** and that systematically under-weight long-treated groups/early periods (the heterogeneity-robustness failure this paper fixes). `ω_11 = 1` with a single treatment group, so the 2×2 case is unaffected (fn 6).

*Main estimator — two-stage procedure (§3):*
```
Step 1 (untreated subsample, D_gp = 0):   Y_gpit = λ_g + γ_p + ε_gpit   →  λ̂_g, γ̂_p
Step 2 (all observations):                regress  (Y_gpit − λ̂_g − γ̂_p)  on  D_gp   →  β̂
```
- **Identification (§3, Appendix A):** as long as there are untreated *and* treated observations for each group and period, `λ_g, γ_p` are identified from the untreated subpopulation, and Step 2 consistently estimates `E(β_gp | D_gp = 1)` even under arbitrary group×period heterogeneity. Consistency/unbiasedness proved in Appendix A (prose proofs, not numbered).
- **First-stage flexibility (fn 8):** Stage 1 need not use *only* untreated observations — any correctly specified conditional-mean model works (e.g. a full-sample model with treatment-status×period interactions, or group×period treatment indicators). Using the full sample can be **more efficient** (this is the "Two-stage (incl. treat×time in first stage)" row of Table 2). Restricting to untreated obs introduces **no** sample-selection bias because selection is on treatment status, a deterministic function of `(g,p)` (fn 10).

*Estimands (§3.1):*
```
Overall ATT (default):     E(β_gp | D_gp = 1) = Σ_g Σ_p β_gp · P(g, p | D_gp = 1)        ... (4)
P̄-period average:          Σ_g Σ_{p=g}^{g+P̄-1} β_gp · P(g | D_g = 1) / P̄                  ... (5)
```
- Eq. (4) is the default target (expectation over all observed treated units and periods). Eq. (5) averages group-specific effects over a **common completed duration `P̄`** — recover it by restricting the Stage-2 sample to treatment durations `≤ P̄`. The `P̄`-average is more comparable across groups when durations differ (it ignores effects beyond `P̄`).

*Event studies (§3.2):*
```
Standard (misspecified under heterogeneity), eq. (6):
   Y_gpit = λ_g + γ_p + Σ_{r=-R}^{P} β_r D_rgp + ε_gpit                  ... (6)
   where, for r ≤ 0, D_rgp ∈ {D_{-R,gp},…,D_{0gp}} are (r+1)-period treatment-adoption LEADS,
   and for r > 0, D_rgp ∈ {D_{1gp},…,D_{Pgp}} are r-period treatment-DURATION indicators.

Two-stage event-study, Step 2′:
   regress (Y_gpit − λ̂_g − γ̂_p) on D_{-R,gp},…,D_{0gt},…,D_{Pgp}   →  Ê(β_rgp | D_rgp = 1)
```
- **`D_rgp` is the event-study (relative-time) design** — leads for `r ≤ 0` (placebo/pre-trend tests), duration indicators for `r > 0`. (fn 12: calendar time `t` may replace coarse periods `p`. fn 13: duration↔period correspondence `β_rgp = β_{g,p-g+1}`.)
- Like the static case, the **standard** event-study spec (eq. 6) is contaminated under heterogeneity — its leads can be nonzero even under true parallel trends (Sun & Abraham 2020). The two-stage Step 2′ fixes this (Figure 2 confirms: standard spec shows a spurious pre-trend dip; two-stage does not). fn 14: the Step-2′ estimand averages over groups with duration `≥ r` (the Sun–Abraham interaction-weighted target).

*Standard errors — joint GMM, Newey-McFadden (1994) Theorem 6.1, §3.3, pp. 12–13:*
SEs must account for the **generated dependent variable** `Y_gpit − λ̂_g − γ̂_p` (Dumont et al. 2005). Read the two stages as a single GMM estimator (Hansen 1982) with parameter `(λ, γ, β)` solving the moment condition
```
E[ f(λ, γ, β; W_gpit) ] = E [ [Y − (1g)'λ − (1p)'γ] · [(1g), (1p)]' · (1 − D_gp)     ]  = 0
                              [ [Y − (1g)'λ − (1p)'γ − β D_gp] · D_gp                  ]
```
where `W_gpit = [Y_gpit, (1g)_gpit, (1p)_gpit, D_gp]`. The first moment block is restricted to **untreated** observations (the `(1 − D_gp)` factor) and identifies `λ, γ`; the second block identifies `β`. By Newey-McFadden Theorem 6.1, `√N(β̂ − β) →_d N(0, v)`, where `v` is the **last element** of
```
   E[ ∂f/∂(λ,γ,β) ]^{-1} · E[ f f' ] · E[ ∂f/∂(λ,γ,β) ]^{-1′}
```
- **This is the standard GMM sandwich with the GLOBAL Jacobian inverse `E[∂f/∂(λ,γ,β)]^{-1}` (the "bread") and the score outer-product `E[ff']` (the "meat").** There is **no per-cluster inverse** anywhere in the paper.
- "Asymptotics for variations on the two-stage estimator (such as the event-study version) are similar" (p. 13).
- **Reference implementation = Appendix B Stata GMM (p. 23):** `gmm (eq1: (y - {xb: i.year} - {xg: ibn.id})*(1-d)) (eq2: y - {xb:} - {xg:} - {delta}*d), instruments(eq1: i.year ibn.id) instruments(eq2: d) winitial(identity) onestep quickderivatives vce(cluster id)`. Key choices: **`vce(cluster id)`** clusters the meat at the unit (`id`) level with the **global** GMM bread; **`onestep` + `winitial(identity)`** (no efficient-GMM reweighting); **no finite-sample multiplier**. The empirical application clusters on **state** (Table 2 / Figure 4 notes; fn 19).

*Edge cases / boundary conditions (from the paper):*
- **Already-treated-in-first-period units → exclude.** The empirical application "exclude[s] units that were already treated in 1979, the first year of the sample, since no treatment effects are identified for this group" (fn 19). Maps to the library's always-treated exclusion.
- **Rank/identification:** `λ_g, γ_p` require untreated *and* treated observations for each group and period. A post-adoption period with no untreated units leaves its `γ_p` unidentified (the Gardner-native analogue of the no-never-treated problem). The library's rank-deficiency guard (`rank_deficient_action`) handles this.
- **Pre-trend testing:** event-study leads `D_rgp, r ≤ 0` serve as placebo tests, but only the **two-stage** leads (Step 2′) are heterogeneity-robust; the standard-spec leads (eq. 6) are not.
- **Covariates (fn 9):** include time-varying covariates in *both* regressions (analogous to regression DiD — assumes treatment doesn't influence the covariates and covariates don't influence the effect); or adjust by the last pre-treatment realization `X*_gpit`; or a doubly-robust combination. All are "more parametric than" the CS (2020) inverse-probability-weighting approach. (Library covariate support is the same `covariates=` story as ImputationDiD.)

**Reference implementation(s):**
- Stata: Appendix B GMM syntax (above); the `did2s` Stata command (Butts & Gardner).
- R: **`did2s`** (Kyle Butts & John Gardner) — **the library's R parity reference for PR-B.**
- The two-stage estimator coincides in point estimates with the imputation estimator (Borusyak, Jaravel & Spiess 2024) and with `ImputationDiD` in this library (the natural validation pair).

**Requirements checklist** (paper → library; ✓ = implemented + verified in source review, ⚠️ = paper-permitted but **not exposed** by the library, ☐ = PR-B deliverable):
- [x] Stage 1 group+period FE fit on **untreated obs only** (`D_gp = 0`), iterative for unbalanced panels (`_iterative_fe`).
- [x] Stage 2 regress residualized `Y − λ̂ − γ̂` on `D_gp` → overall ATT eq. (4) (`_stage2_static`); event-study Step 2′ → eq. (6) horizons (`_stage2_event_study`; `horizon_max`, `pretrends`); plus a library cohort/group aggregation (`_stage2_group`, a CS-style extension — not a distinct Gardner estimand).
- [x] **Variance = joint-GMM Newey-McFadden Thm 6.1 sandwich with the GLOBAL Jacobian inverse, meat clustered at the unit** (Appendix B `vce(cluster id)`); **no finite-sample multiplier**. Library matches this (the global-bread sandwich `(X'₂WX₂)^{-1}(S'S)(X'₂WX₂)^{-1}` with unit-clustered GMM-corrected score `S_g = γ̂'c_g − X'_{2g}ε_{2g}`).
- [x] Always-treated exclusion (fn 19); covariates in both stages (fn 9, `covariates=` — the in-both-regressions approach); `anticipation` handling.
- ⚠️ **Paper-permitted but NOT exposed by `TwoStageDiD`** (recorded as scope, not a deliverable): the **`P̄`-average estimand eq. (5)** (duration-restricted Stage-2 sample) and the **full-sample first-stage variant (fn 8)** — both described in the paper as valid modifications, but the library has no public parameter for either (`get_params()` = `anticipation, alpha, cluster, n_bootstrap, bootstrap_weights, seed, rank_deficient_action, horizon_max, pretrends, vcov_type`; Stage 1 is always untreated-only). The fn-9 last-pre-treatment / doubly-robust covariate variants are likewise not exposed (only the in-both-stages approach is).
- [x] **Dedicated `tests/test_methodology_two_stage.py`** with eq./section-numbered Verified Components (Stage-1 FE recovery; Stage-2 overall ATT eq. 4 + event-study eq. 6; GMM first-stage-correction behavior; always-treated drop). — *PR-B.*
- [x] **R parity fixture vs `did2s`** (`benchmarks/R/generate_did2s_golden.R` + `benchmarks/data/did2s_golden.json` + `did2s_test_panel.csv`); overall ATT (abs 1e-6) + event-study ATT (1e-6) / SE (1e-7). — *PR-B.*
- [x] **`two_stage.py` docstring/formula faithful; variance made exact in PR-B.** `_compute_gmm_variance`'s docstring (`two_stage.py:2842-2857`) states the code uses the GLOBAL Hessian inverse matching `did2s` and gives the correct sandwich `V = (X'₂X₂)^{-1}(Σ_g S_g S_g')(X'₂X₂)^{-1}` — the *formula* needed no correction (the Eq.6 misattribution was confined to the REGISTRY note, fixed in PR-A). However, the PR-B `did2s` SE parity surfaced a ~1% **numerical** gap: the variance derived its residuals from the *iterative* alternating-projection first-stage FE (`_iterative_fe`, converged to ~1e-7 on unbalanced untreated panels) while computing `gamma_hat` exactly. PR-B corrects this by re-solving the Stage-1 FE **exactly** inside the variance (sparse OLS, reusing the `gamma_hat` factorization) and adding an intercept to `_build_fe_design` so its column space spans the grand mean — yielding ~1e-7 `did2s` SE parity, mirroring ImputationDiD's exact-sparse variance. The point estimate stays on the iterative FE (twin-equivalence preserved). See REGISTRY `## TwoStageDiD` Notes.

---

## Implementation Notes

### Data Structure Requirements
- Long panel: unit id, time, outcome, and a cohort/event-date or absorbing-treatment column from which `D_gp` and relative time are derived. Panel need not be balanced; treatment must be staggered/absorbing.
- Optional: time-varying covariates `X_gpit` (treatment-unaffected & effect-neutral per fn 9), cluster/PSU columns, survey weights.

### Computational Considerations
- Dominant cost = the Stage-1 FE solve on the untreated subsample (alternating-projection/within transform; `_iterative_fe`). Stage 2 is a small OLS on the residualized outcome.
- The GMM variance needs the Stage-1→Stage-2 correction term (`γ̂' c_g`); the library precomputes `γ̂ = (X'₁₀WX₁₀)^{-1}(X'₁WX₂)` via a sparse factorization with a dense-`lstsq` fallback (both warn on near-singularity).

### Tuning Parameters

**Library-exposed knobs** (`TwoStageDiD.get_params()`):

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| Estimand | (output) | overall ATT (eq. 4) | Results carry overall ATT eq. (4) + event-study eq. (6) (`horizon_max`, `pretrends`) + a cohort/group aggregation; selected at read-time, not a fit param |
| `cluster` | column | unit (`cluster=None` → `unit`) | Appendix B clusters on `id`; the application clusters on state |
| `covariates=` (fit arg, fn 9) | list | none | Time-varying covariates residualized in **both** stages (the fn-9 in-both-regressions approach) |
| `anticipation` | int | 0 | Periods of anticipation (shifts the event date earlier) |
| `n_bootstrap` / `bootstrap_weights` | int / str | 0 / rademacher | Optional multiplier bootstrap (library extension; analytical GMM SE is the default) |

No regularization, bandwidth, factor-count, or cross-validation tuning (the estimator is unregularized OLS in each stage). **Paper-permitted variants NOT exposed:** the eq. (5) `P̄`-average / duration-restricted estimand and the fn-8 full-sample first-stage variant (see the ⚠️ row in the Requirements checklist).

### Relation to Existing diff-diff Estimators
- **`TwoStageDiD` (this estimator):** direct implementation of the §3 two-stage procedure + §3.3 GMM variance. The REGISTRY `## TwoStageDiD` section is the implementation contract; this review is its primary-source backing.
- **`ImputationDiD` (Borusyak, Jaravel & Spiess 2024):** **point-estimate-identical** twin — both estimate unit/group+time FE on untreated observations and recover effects from observed-minus-counterfactual. They differ in **inference**: TwoStageDiD uses the GMM sandwich that folds in Stage-1 estimation error; ImputationDiD uses the conservative Theorem-3 variance. (BJS 2024 p. 3258 lists Gardner's estimator as coinciding with theirs for Gardner's estimand class.)
- **`StackedDiD`:** the paper's §5/Appendix §6.1 "stacked" estimator is a *different* heterogeneity-robust device (group-specific datasets, dataset-specific FE); Gardner's Table 1–2 compare two-stage vs stacked vs aggregated.
- **Reusable library machinery:** `_iterative_fe` / within-transform (Stage 1); `solve_ols` (Stage 2); `safe_inference` for the GMM SE → t/p/CI; the SpilloverDiD `_compute_gmm_corrected_meat` is the sibling GMM-correction machinery (same first-stage-uncertainty structure); CallawaySantAnna/ImputationDiD NaN convention for unidentified cells; multiplier bootstrap is a **library extension** — Gardner (2022) proposes **no** bootstrap (analytical GMM SEs only, §3.3 + Appendix B Stata GMM). R `did2s` likewise **defaults to analytical corrected clustered SEs** (`bootstrap = FALSE`, the same GMM sandwich); its block bootstrap is *optional* (`bootstrap = TRUE`), not a Gardner-paper prescription. The library's multiplier bootstrap mirrors the CallawaySantAnna/ImputationDiD pattern.

---

## Gaps and Uncertainties

**No supplement-only GAP class** — the paper is self-contained (proofs in Appendix A, reference code in Appendix B, full tables in Appendix C). The reviewed PDF rendered cleanly; no `[UNREADABLE EQUATION]`. The only uncertainties are reconciliations against the existing REGISTRY:

| Item | Finding | Disposition |
|------|---------|-------------|
| **REGISTRY "Note on Equation 6 discrepancy"** (`## TwoStageDiD`) claimed *the paper's Equation 6* uses a per-cluster inverse `(D_c'D_c)^{-1}` that our global-inverse code "deviates" from. | **Misattribution.** Paper **eq. (6) is the event-study regression specification**; the variance is **unnumbered** (§3.3) and uses the **global Jacobian inverse** `E[∂f/∂(λ,γ,β)]^{-1}` (Newey-McFadden Thm 6.1). There is **no per-cluster inverse** in the paper. Our implementation **matches** the paper (and `did2s`) — it is **not** a deviation. | **Corrected in this PR** (REGISTRY note rewritten to state faithfulness; the cluster-summed filling, previously mislabeled "Bread", relabeled "Meat"). The `two_stage.py` docstring (`:2842-2857`) was checked and is **already accurate** (it describes the code's global inverse + `did2s` match and gives the correct sandwich; no fabricated paper deviation) — **no source change needed**. |
| **REGISTRY edge-case note** "the GMM sandwich variance accounts for Stage 1 estimation error (Gardner 2022, Theorem 1)". | **Misattribution.** Gardner (2022) has **no Theorem 1** (no numbered theorems at all). The Stage-1 correction comes from the joint-GMM reading + **Newey-McFadden (1994) Theorem 6.1**. | **Corrected in this PR** ("Gardner 2022, Theorem 1" → "Gardner 2022 §3.3; Newey-McFadden 1994, Theorem 6.1"). |
| **Proposition-5 / "no never-treated → horizons ≥ H̄ unidentified"** guard in the library + REGISTRY. | Gardner (2022) does **not** prove this — it is a **Borusyak, Jaravel & Spiess (2024)** result (their Proposition 5). Gardner's native identification requirement is the weaker "untreated obs for each period." | **No change** — the REGISTRY already correctly attributes this guard to "Proposition 5 of Borusyak et al. (2024)" (lines 1411, 1414). Noted here for clarity: it is a twin-derived extension, not a Gardner result. |
| **"No finite-sample multiplier" deviation note** (REGISTRY `## TwoStageDiD`). | **Accurate and correctly labeled.** Appendix B's `onestep`/`vce(cluster id)` GMM carries no `(n-1)/(n-p)` or `G/(G-1)` factor; the library matches (meat `= S'S`). | **No change** — verified faithful. |
| **Version.** Reviewed arXiv:2207.05943**v1** (title page "April 2021"; arXiv-stamped 13 Jul 2022). | Equation numbers are this version's. The misattributions above are **independent of versioning**: even if a later version renumbered the variance to "(6)", the per-cluster-inverse claim is false regardless (the variance uses a global inverse). | Citation pinned to **Gardner (2022)** per `docs/references.rst`. |

**PR-B deliverables** (tracked at the time in `TODO.md` Testing/Docs; since shipped): `tests/test_methodology_two_stage.py` (Stage-1/Stage-2/GMM-correction/always-treated Verified Components); `did2s` R parity fixture; then flip the `METHODOLOGY_REVIEW.md` `TwoStageDiD` row **In Progress → Complete**. (The variance *formula* and docstring needed no correction; the only documentation defect, the REGISTRY misattributions, was fixed in PR-A. PR-B did make one **numerical-accuracy** source change — re-solving the Stage-1 FE exactly inside `_compute_gmm_variance` + an intercept in `_build_fe_design` — so the analytical GMM SE matches R `did2s` to ~1e-7; see the checklist item above and REGISTRY `## TwoStageDiD` Notes.)
