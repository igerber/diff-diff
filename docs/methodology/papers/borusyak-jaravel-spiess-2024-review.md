# Paper Review: Revisiting Event-Study Designs: Robust and Efficient Estimation

**Authors:** Kirill Borusyak (UC Berkeley), Xavier Jaravel (LSE), Jann Spiess (Stanford University)
**Citation:** Borusyak, K., Jaravel, X., & Spiess, J. (2024). *Revisiting Event-Study Designs: Robust and Efficient Estimation.* The Review of Economic Studies, 91(6), 3253–3285. DOI: [10.1093/restud/rdae007](https://doi.org/10.1093/restud/rdae007). Open Access (CC BY 4.0). Advance access 6 February 2024; first version received April 2022; editorial decision August 2023; accepted January 2024.
**PDF reviewed:** **Published REStud version of record** (`rdae007`, 33 pages, journal pp. 3253–3285). Per the project's PDFs-never-committed convention, the local PDF is kept outside the repository (gitignored `papers/rdae007.pdf`); the DOI page is the authoritative source. **Version-pinned to the published REStud 91(6) typeset article** — all equation, theorem, proposition, and assumption numbers below are the *published* version's. The arXiv preprint (arXiv:2108.12419) and the 2021 working-paper draft **renumber** these; do not reconcile against a different version's rendering.
**Review date:** 2026-06-06

---

> **Scope note — main article only (Supplementary Material NOT reviewed here).** This PDF is the main REStud article (pp. 3253–3282 content; References pp. 3283–3285). REStud publishes proofs and several load-bearing results in a *separate* Supplementary Material document that is **not in this PDF**. The following supplement-only objects are referenced in the body and are flagged throughout as **GAP** (see "Gaps and Uncertainties"): **Supplementary Proposition A1** (identification rank/spanning condition), **A2** (imputation under restrictions `τ=Γθ`), **A3** (explicit efficient observation weights `v*_it`), **A4** (adjusted weights as a constrained quadratic program), **A7** (low-level short-panel asymptotic conditions); **Supplementary Appendix B** (ALL proofs); **Supplementary Appendices A.1–A.11** (incl. A.5 heteroskedasticity/serial-correlation generalization, A.8 minimal-excess-variance single-group result, A.9 leave-one-out variance, A.10 efficient `v*_it` algorithm, A.11 Monte Carlo); **Supplementary Table A1**; and **equation (17)** (FWL static weights, in the proof of Proposition 2).

> **Personal verification stamp (load-bearing items spot-checked against the source).** Beyond the multi-agent extraction, the reviewer read PDF pages 3266–3274 directly and verified verbatim: Theorem 1 + eq. (4) (p. 3267); Theorem 2 + eq. (5) (p. 3268); Proposition 6 (p. 3268); Assumptions 5 & 6 (p. 3269); Propositions 7 & 8 (p. 3270); Theorem 3 + eqs. (6)/(7) and the `σ²_τ≥0` conservative term (p. 3271–3272); the aux-model eq. (8) and default cohort×period partition (p. 3272); Test 1 + eq. (9) (p. 3273); Proposition 9 (p. 3274); Proposition 5 + the `H̄` non-identification statement (p. 3266); and footnote 23 confirming `v*_it` is derived in Supplementary Proposition A3 (not in this PDF). **The equation/theorem/proposition numbering used by the existing `docs/methodology/REGISTRY.md` `## ImputationDiD` section matches this published version** (Theorem 2 imputation; Theorem 3 / Eq. 7 variance; Eq. 8 aux model; Eq. 9 / Test 1 pre-trend; Proposition 5 for `H̄`; Proposition 9 for pre-test independence; Supplementary Proposition A3 for `v*_it`).

---

## Methodology Registry Entry

*This file is the canonical **scholarly review** of the **primary source** (DOI 10.1093/restud/rdae007) for the library estimator `ImputationDiD` (`imputation.py`, `imputation_bootstrap.py`). It **supplements — it does not supersede** — the existing `docs/methodology/REGISTRY.md` `## ImputationDiD` section, which remains the implementation contract and whose paper-numbering anchors were verified accurate against this published version (see verification stamp above). Heading/format aligned with REGISTRY conventions for ease of cross-reference.*

## ImputationDiD

**Primary source:** Borusyak, Jaravel & Spiess (2024), *Revisiting Event-Study Designs: Robust and Efficient Estimation*, Review of Economic Studies 91(6), 3253–3285. DOI: 10.1093/restud/rdae007.

**Central thesis (Abstract; §3; §4):** Conventional event-study practice — static/dynamic TWFE OLS (eqs. (1)–(3)) — *conflates* three choices that should be separate: (a) the identifying assumptions (parallel trends, no-anticipation), (b) restrictions on treatment-effect heterogeneity, and (c) the estimation target. This conflation causes three concrete failures: **under-identification** of the fully dynamic spec without never-treated units (§3.2, Prop. 1), **negative weighting / "forbidden comparisons"** in static regressions (§3.3, Props. 2–4), and **spurious identification** of long-run effects via extrapolation (§3.4, Prop. 5). The paper instead *explicitly separates* the researcher-chosen target `τ_w`, the identifying Assumptions 1–2, and any optional heterogeneity restriction (Assumption 3), then derives the **efficient, robust imputation estimator** and matching conservative inference.

**Key implementation requirements:**

*Setup & notation (§2):*
- Panel (need not be complete/balanced) of units `i`, periods `t`. **Conditional framework** — no random sampling; observations and event dates treated as non-stochastic; randomness is from `ε_it` only.
- Binary, **absorbing** treatment: `D_it = 1[K_it ≥ 0]`, `K_it = t − E_i` (horizon / relative time), `E_i` = event date (`E_i = ∞` for never-treated). Units sharing `E_i` = a cohort.
- `Ω` = all observations (size `N`); `Ω₁ = {it : D_it=1}` (treated, size `N₁`); `Ω₀ = {it : D_it=0}` = never-treated ∪ not-yet-treated (size `N₀`); `Ω_{1,h} = {it : K_it=h}`.
- Causal effect on a treated observation: `τ_it = E[Y_it − Y_it(0)]` for `it ∈ Ω₁`.

*Estimation target (§2):*
```
τ_w = Σ_{it∈Ω₁} w_it τ_it ≡ w₁′τ
```
- Pre-specified non-stochastic weights `w_it` (depend on assignment/timing, NOT on realized outcomes); **need not sum to one**.
- Overall ATT: `w_it = 1/N₁`. Horizon-`h`: `w_it = 1[K_it=h]/|Ω_{1,h}|`. Group differences: `Σ w_it = 0`. Heterogeneity-by-`R_it` slope (fn 7): `w_it = (R_it − R̄)/Σ(R_js − R̄)²`. ATT-per-intensity (non-binary): `w_it ∝ 1/R_it`.

*Assumption checks / warnings:*
- **Assumption 1 (Parallel trends), p. 3259:** ∃ non-stochastic `α_i, β_t` with `E[Y_it(0)] = α_i + β_t` ∀ `it∈Ω`. Imposed at unit level, on the whole sample, on *potential* outcomes (⇒ a-priori justifiable + testable).
- **Assumption 1′ (General model of Y(0)), p. 3260:** `E[Y_it(0)] = A'_it λ_i + X'_it δ`. `A'_itλ_i` nests unit FEs & unit-specific trends (observed `A_it`); `X'_itδ = β_t + X̃'_itδ̃` nests period FEs + time-varying covariates. **`X_it` must be unaffected by treatment and strictly exogenous** (Supp. App. A.1 — GAP).
- **Assumption 2 (No-anticipation), p. 3260:** `Y_it = Y_it(0)` ∀ `it∈Ω₀`. Weakenable to allow `k`-period anticipation by redefining event dates earlier.
- **Assumption 3 / 3′ (Optional heterogeneity restriction), p. 3260:** `Bτ=0` (`B` `M×N₁` full row rank) ≡ `τ=Γθ` (`Γ` `N₁×(N₁−M)` full column rank). **Null model `M=0, Γ=I_{N₁}` (unrestricted heterogeneity) is the conservative default.**
- **Assumption 4 (Spherical errors), p. 3267 — efficiency only:** `E[εε′]=σ²I_N`. Also holds under unit random effects `ε_it=η_i+ε̃_it`; generalizes to known heteroskedasticity/dependence by weighting Step 1 by `σ_it^{−2}` (fn 21).
- **Assumption 5 (Clustered errors), p. 3269 — inference:** `ε_it` independent across units `i`, `Var[ε_it] ≤ σ̄²`.
- **Assumption 6 (Herfindahl condition), p. 3269 — consistency:** `‖v‖²_H ≡ Σ_i(Σ_{t:it∈Ω}|v_it|)² → 0`; effective sample size `n_H = ‖v‖_H^{−2} → ∞`.
- **Rank/spanning identification (Supp. Prop. A1, summarized p. 3267):** covariate space of treated obs must be spanned by that of untreated obs. For the unit-FE/unit-trend + time-FE case: identified iff (1) `A_it` not collinear for any relevant unit, AND (2) **≥1 untreated unit at the end of the time period of interest**. Practitioner form: every treated unit needs ≥1 untreated period; every post period needs ≥1 untreated unit. (Full statement = Supp. Prop. A1 — GAP.)

*Main estimator — Theorem 1 (efficient, OLS form), eq. (4), p. 3267:*
```
Y_it = A'_it λ_i + X'_it δ + D_it Γ'_it θ + ε_it          ... (4)
Step 1. θ̂*  = OLS of (4)         (θ assumed identified)
Step 2. τ̂*  = Γ θ̂*
Step 3. τ̂*_w = w₁′ τ̂*
```
- Unique efficient linear unbiased estimator of `τ_w` under Assumptions 1′,2,3′,4 (Gauss–Markov).
- **Unbiased under Assumptions 1′,2,3′ ALONE — even with non-spherical errors.**

*Recommended default — Theorem 2 (imputation representation), eq. (5), p. 3268:*
Under the null Assumption 3′ (`Γ = I_{N₁}`, unrestricted heterogeneity), the Theorem-1 estimator equals the imputation estimator:
```
Step 1 (untreated only, it∈Ω₀):   Y_it = A'_it λ_i + X'_it δ + ε_it   → λ̂*_i, δ̂*   ... (5)
Step 2 (each treated it∈Ω₁, w_it≠0):  Ŷ_it(0) = A'_it λ̂*_i + X'_it δ̂*;   τ̂*_it = Y_it − Ŷ_it(0)
Step 3 (aggregate):                  τ̂*_w = Σ_{it∈Ω₁} w_it τ̂*_it
```
- **Steps 1–2 are identical regardless of the target estimand — only the Step-3 weights change.**
- Computationally cheap: a TWFE fit on untreated obs only (fast algorithms: Guimarães & Portugal 2010; Correia 2017), vs. the high-dimensional Theorem-1 regression.
- **Proposition 6 (p. 3268):** ANY unbiased linear estimator (null Assumption 3) has an imputation representation — they differ only in *how* `Y_it(0)` is imputed; Theorem 1/2's way is the efficient one.

*Linear-estimator weights `v*_it`:* the estimator is `τ̂*_w = Σ_{it∈Ω} v*_it Y_it` with non-stochastic `v*_it`. **Explicit `v*_it` formula = Supplementary Proposition A3 (GAP — not in this PDF).** Adjusted weights `v*₁` under restrictions solve a constrained quadratic variance-min program (Supp. Prop. A4; fn 23). Efficient computation with multiple high-dim FEs = Supp. App. A.10 (iterative LS) (GAP).

*Standard errors — Theorem 3 (conservative, clustered), eqs. (6)/(7), §4.3, p. 3271–3272:*
Target variance under unit-clustered errors (Assumption 5):
```
σ²_w = E[ Σ_i ( Σ_{t:it∈Ω} v_it ε_it )² ]
```
Plug-in (eq. 6) and the feasible conservative estimator (eq. 7):
```
σ̂²_w = Σ_i ( Σ_{t:it∈Ω} v_it ε̃_it )²                                  ... (6)
σ̂²_w = Σ_i ( Σ_{t:it∈Ω} v_it ε̃_it )² ,
        ε̃_it = Y_it − A'_it λ̂*_i − X'_it δ̂* − D_it τ̃_it             ... (7)
```
- For **untreated** obs, `ε̃_it = ε̂_it` (the Step-1 residual). For **treated** obs, the trivial residual `ε̂_it ≡ 0` (perfect fit) is replaced by the auxiliary-model residual using `τ̃_it`.
- **Asymptotically conservative:** `‖v‖_H^{−2}(σ̂²_w − σ²_w − σ²_τ) →_p 0`, where `σ²_τ = Σ_i(Σ_{t:D_it=1} v_it(τ_it − τ̄_it))² ≥ 0`. **Exact** (`σ²_τ=0`) iff the auxiliary model is correct (`τ̄_it = τ_it`).
- Exact estimation of `σ²_w` is impossible under unrestricted heterogeneity (cannot separate `τ_it` from `ε_it`; cf. Kline et al. 2020, Lemma 1) — hence the conservative auxiliary-model device, following de Chaisemartin & D'Haultfœuille (2020).
- Extends directly to **variance–covariance matrices** for vector-valued estimands (e.g. ATTs at multiple horizons).

*Auxiliary treatment-effect model, eq. (8), p. 3272:*
Partition `Ω₁ = ∪_g G_g`, impose `τ_it ≡ τ_g` within group, estimate:
```
            Σ_i ( Σ_{t:it∈G_g} v_it ) ( Σ_{t:it∈G_g} v_it τ̂*_it )
τ̃_g  =  ──────────────────────────────────────────────────────────       ... (8)
                     Σ_i ( Σ_{t:it∈G_g} v_it )²
```
- **Default partition (Stata `did_imputation`): groups by cohort × period** (when cohorts large); else by horizon relative to onset; single group = most conservative (minimal excess variance, Supp. App. A.8).
- Finite-sample refinement: **leave-one-out** modification of `τ̃_it` (Supp. App. A.9). **✓ Implemented** as the opt-in `leave_one_out` parameter (default False); the efficient-rescale formula and derivation live in REGISTRY (see the A.9 provenance note under "Other notes").
- **✓ Implemented (PR-B):** `_compute_auxiliary_residuals_treated` (`imputation.py`) computes the *unit-clustered* Eq. (8) above (per unit, the within-unit sums `a_{i,g}=Σ_t v_it` and `b_{i,g}=Σ_t v_it·τ̂*_it`, then `Σ_i a·b / Σ_i a²`). Non-target / unimputable (NaN `τ̂`, `v_it=0`) observations are excluded from the aggregation — exact for finite `τ̂` and NaN-safe. *Prior to the ImputationDiD methodology validation the code used the observation-level mean `Σ v·τ̂ / Σ v`, which coincides with Eq. (8) only under uniform within-group weights and diverges for survey-weighted / heterogeneity estimands (any partition) or the coarser `cohort` partition where a unit contributes multiple observations to a group.* The code docstring and the REGISTRY label now state the exact unit-clustered Eq. (8). Validated by white-box hand-calc + R `didimputation` parity in `tests/test_methodology_imputation.py`.
- **No bootstrap is proposed** anywhere in the paper — SEs are analytical/cluster-robust. No explicit `G/(G−1)` or `(N−k)` DOF multiplier in the main text (the only finite-sample device is LOO).

*Pre-trend / placebo test — Test 1, eq. (9), §4.4, p. 3273:*
```
Y_it = A'_it λ_i + X'_it δ + W'_it γ + ε_it          ... (9)   [OLS on untreated obs only]
Test  γ = 0  via heteroskedasticity- and cluster-robust Wald test.
```
- Valid because (9) is implied by Assumptions 1′,2 under `H₀: γ=0`. Natural `W_it` = indicators for `1,…,k` periods before onset, reference = periods before `E_i − k`.
- Uses **untreated obs only** ⇒ robust to treatment-effect heterogeneity (avoids Sun–Abraham 2021 contamination); under spherical+normal errors asymptotically equivalent to the homoskedastic F-test (UMP invariant; Lehmann–Romano 2006 §7.6). Hausman-test alternative (fn 26). Optimal `k` is open (fn 27).
- **Proposition 9 (Pre-test robustness), p. 3274:** under model (9) + Assumption 4, `τ̂*_w` is **uncorrelated** with any Test-1 `γ̂`; under normality they are **independent**, so inference on `τ_w` is **unaffected by pre-testing** (avoids Roth 2022). No adjustment needed for the Theorem-1 estimator under spherical errors.

*Edge cases:*
- **No never-treated units (Proposition 5, p. 3266):** with `H̄ = max_i E_i − min_i E_i`, any non-negative-weighted sum of effects over `K_it ≥ H̄` is **not identified** by Assumptions 1–2. Robust estimators are computed only for identified estimands → refuse-to-estimate (the library sets such horizons to NaN with a warning). Note (fn 20): *differences* across long horizons (e.g. `τ_A4 − τ_B4`) can still be identified.
- **Fully dynamic spec under-identification (Proposition 1, p. 3262):** without never-treated units, only a *linear* trend in `{τ_h}` is unidentified (`{τ_h + κ(h+1)}` fits equally well); non-linear paths are identified. Packages that silently drop a lead/period indicator can manufacture a spurious trend.
- **Negative weighting (Props. 2–4):** static estimand `τ^static = Σ w^static_it τ_it`, weights sum to 1 but can be negative ("forbidden comparisons"); no negative weighting iff `N*₁ ≥ N*₀` (Prop. 4).
- **Always-treated units:** no untreated reference periods ⇒ implicitly excluded by the spanning condition (not separately treated in the main text).
- **Incidental parameters:** `λ_i` not consistently estimable in short panels, but `τ̂_w` is still consistent (averages over units) and the cluster-robust variance does not require consistent `λ̂_i` (Stock & Watson 2008).

**Reference implementation(s):**
- Stata: `did_imputation` (estimator + inference) and `event_plot` (Borusyak, authors' commands).
- R: `didimputation` (community package; not named in the main text — appears in References/ecosystem). **This is the library's R parity reference for PR-B.**
- `csdid` is used in the paper only to run the dCDH/CS comparison.

**Requirements checklist** (paper → library):
- [x] Step 1 TWFE/`A,X` fit on **untreated obs only** (`Ω₀`), iterative for unbalanced panels.
- [x] Step 2 impute `Ŷ(0)`, form `τ̂*_it`; Step 3 weighted aggregation by estimand-specific `w_it`.
- [x] Conservative clustered variance eq. (7); cohort×period default partition.
- [x] Auxiliary `τ̃_g`: unit-clustered Eq. (8) `Σ_i a·b / Σ_i a²` across all `aux_partition` modes; matches R `didimputation` at the default cohort×event-time partition.
- [x] Proposition-5 refuse-to-estimate (NaN + warning) for `K_it ≥ H̄` with no never-treated units.
- [x] Test-1 pre-trend regression (eq. 9) on untreated obs, cluster-robust Wald; pre-period event-study leads.
- [x] **R parity fixture vs `didimputation`** — `benchmarks/data/didimputation_golden.json` (generator `benchmarks/R/generate_didimputation_golden.R`, v0.5.0); point estimates match to ~1e-7, SEs to ~1e-10 (observed; tests assert ATT `abs=1e-6` / SE `abs=1e-7`) (`TestImputationDiDParityR`).
- [x] **Dedicated `tests/test_methodology_imputation.py`** with eq./theorem-numbered Verified Components (Theorem 1/2/3, eqs. 5-9, Props. 5/9 + white-box Eq. 8 hand-calc).

---

## Implementation Notes

### Data Structure Requirements
- Long panel: unit id, time, outcome, and either an event-date/cohort column or a binary absorbing treatment indicator from which `E_i`/`K_it` are derived. Panel need not be balanced; treatment must be absorbing.
- Optional: time-varying covariates `X̃_it` (must be treatment-unaffected & strictly exogenous), unit-specific trends (`A_it`), cluster/PSU columns, survey weights.

### Computational Considerations
- Dominant cost = Step-1 TWFE solve on `Ω₀`; use alternating-projection/within transform (Guimarães–Portugal 2010; Correia 2017). Unbalanced panels ⇒ iterative demeaning (the library already does this; one-pass demeaning is exact only for balanced panels).
- Variance requires the implied weights `v*_it`; with multiple high-dim FEs this is the costly part (Supp. App. A.10 algorithm — consult supplement for the exact recursion).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|------------------|
| Auxiliary partition `{G_g}` (eq. 8) | categorical | cohort × period | Within-group heterogeneity < across-group; coarser ⇒ more conservative; single group = max conservative |
| Pre-trend lead count `k` (Test 1) | int | researcher choice | No automatic rule (fn 27); too-large `k` lowers power |
| Step-1 weights (efficiency under known variances) | weights | unweighted OLS | `∝ σ_it^{−2}` (e.g. `∝ n_it` for aggregated cells) (fn 21) |
| Leave-one-out variance refinement | bool | `leave_one_out=False` | Supp. App. A.9 finite-sample improvement — **✓ implemented** |
| Anticipation horizon | int | 0 | Redefine event dates earlier by `k` periods |

No regularization, bandwidth, factor-count, or cross-validation tuning (the estimator is unregularized OLS on untreated data).

### Relation to Existing diff-diff Estimators
- **`ImputationDiD` (this estimator):** direct implementation of Theorem 2 (imputation) + Theorem 3 (conservative variance) + Test 1 (pre-trends). The REGISTRY `## ImputationDiD` section already encodes this; this review is its primary-source backing.
- **`TwoStageDiD` (Gardner 2022):** coincides with the imputation estimator for Gardner's estimand class (p. 3258) — the natural validation pair; both single-effect imputation methods.
- **`WooldridgeDiD` (ETWFE):** Wooldridge (2021) two-way Mundlak ≡ imputation estimator for a restricted estimand class with time-invariant (but time-varying-effect) controls (p. 3258).
- **`CallawaySantAnna`:** equivalent when there is one pre-period and no covariates (fn 5, p. 3258).
- **Liu et al. (2022) / Gardner (2021) / Thakral–Tô (2022) / Athey et al. matrix completion (no factors/regularization):** all coincide with the imputation estimator for their estimand classes (p. 3258).
- **Reusable library machinery:** unified `linalg.py` `solve_ols`/within-transform for Step 1; `safe_inference` for the eq.-7 SE → t/p/CI; CallawaySantAnna NaN convention for Prop-5 refuse-to-estimate; multiplier-bootstrap pattern (a *library extension*, not in the paper).

---

## Gaps and Uncertainties

**Supplement-only content (NOT in the reviewed main-article PDF) — must consult the separate Supplementary Material for PR-B if these surfaces are validated:**

| Object | Referenced at | Why it matters for the library |
|--------|--------------|--------------------------------|
| **Supp. Prop. A1** — full identification rank/spanning conditions | p. 3267 | Backs the rank-deficiency / refuse-to-estimate guards |
| **Supp. Prop. A2** — imputation under restrictions `τ=Γθ` (weight adjustment `w₁→v₁`) | p. 3269, fn 25 | Only relevant if a non-null `Γ` is ever supported |
| **Supp. Prop. A3** — explicit efficient observation weights `v*_it` | p. 3269, fn 23 | **The eq.-7 variance needs `v*_it`; the REGISTRY's "Note on v_it derivation" reconstructs the FE-only closed form because A3 was unavailable. The reviewed PDF confirms A3 is the canonical source and is NOT here.** |
| **Supp. Prop. A4** — adjusted weights as constrained quadratic program | p. 3269, fn 23 | Efficiency under restrictions |
| **Supp. App. A.5** — heteroskedasticity/serial-correlation generalization | p. 3257, 3268 | Maps to any GLS-style variance extension |
| **Supp. App. A.7 / Supp. Prop. A7** — low-level short-panel asymptotics | p. 3270–3272 | Conditions on researcher weights `w₁` |
| **Supp. App. A.8** — minimal excess variance, single-group partition | p. 3272 | Justifies the conservative single-group default |
| **Supp. App. A.9** — leave-one-out variance modification | p. 3256, 3272 | Finite-sample SE refinement — **✓ implemented** (`leave_one_out`) |
| **Supp. App. A.10** — efficient `v*_it` algorithm (iterative LS) | p. 3273 | The practical variance computation recipe |
| **Supp. App. A.11** — full Monte Carlo study | p. 3268, 3282 | SD ratios 1.3–3.6×; coverage claims |
| **Supp. App. B** — **ALL proofs** | p. 3262 | — |
| **equation (17)** — FWL static weights (proof of Prop. 2) | p. 3264 | Negative-weight characterization detail |
| **Supp. Table A1** — binning vs no-binning estimates | p. 3276, fn 34 | Application detail |

**Other notes:**
- The application (§5, Broda–Parker 2008 tax-rebate MPX) is context only, not implementation-relevant; preferred imputation estimate = first-month MPX $30.5 (3.4% of rebate); notional MPC 7.8–11.4%.
- No contradictions were found between the three extraction passes or against the direct page reads; all numbered equations rendered cleanly (no `[UNREADABLE EQUATION]`).
- **Resolved in PR-B:** the auxiliary `τ̃_g` aggregator now implements the exact unit-clustered Eq. (8) (`_compute_auxiliary_residuals_treated` + docstring + REGISTRY label), validated by white-box hand-calc + R `didimputation` parity (`tests/test_methodology_imputation.py`). The earlier obs-level mean was a valid conservative simplification but not the variance-minimizing form.
- **Done in PR-B:** the **R `didimputation` parity fixture** and **`tests/test_methodology_imputation.py`** are on file. The `v*_it` weights (Supp. Prop. A3, absent from this PDF) were validated *empirically* against the reference: the exact two-way-FE projection `-A₀(A₀'A₀)⁻¹A₁'w` now matches `didimputation` (observed ~1e-10; tests assert SE `abs=1e-7`) — the prior FE-only closed form `-(w_i/n0_i + w_t/n0_t − w/N₀)` was a balanced-panel approximation that biased the SE ~27% on staggered (unbalanced-Ω₀) designs and was corrected in PR-B. The leave-one-out variance refinement (Supp. App. A.9) is now **implemented** as the opt-in `leave_one_out` parameter (default False, preserving R `didimputation` parity) — see the A.9 provenance note below.
- **Supp. App. A.9 provenance (leave-one-out, added post-review):** A.9 was a GAP in this main-article review (the REStud Supplementary Material is not in the reviewed PDF). It was subsequently sourced from the arXiv preprint supplement (**arXiv:2108.12419v5, Appendix A.9 "Leave-Out Conservative Variance Estimation"**) to implement the opt-in `leave_one_out` parameter. The **REStud Supplementary Material is the canonical version**; per this doc's version-pin scope note, the efficient-rescale formula `ε̃^LO = ε̃/(1 − v_ig²/Σ_j v_jg²)`, its exact ψ-level equivalence to the direct leave-one-out `τ̃_it^LO`, Prop. A8 (unbiased for an upper bound), and the footnote-51 single-positive-weight-unit edge are recorded in `docs/methodology/REGISTRY.md` `## ImputationDiD` (NOT transcribed here). Validated by the exact ψ-identity + hand-calc + MC coverage (`tests/test_methodology_imputation.py::TestB2024AppendixA9LeaveOneOut`); the authors' Stata `did_imputation` ships the same option (`leaveout`), now a committed Stata parity anchor matching the library LOO SE to ~1e-9 at the overall ATT and all event-study horizons (`benchmarks/stata/generate_imputation_loo_golden.do`; `tests/test_imputation_loo_stata_parity.py`).
