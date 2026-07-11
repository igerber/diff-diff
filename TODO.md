# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

## How this file is organized

- **[Actionable Backlog](#actionable-backlog)** — work with a clear implementation path and
  no external blocker. **Pull from here.** Effort (`Quick` ≤1 day · `Mid` 3-10 CI rounds ·
  `Heavy` derivation-free but large) is noted per row; `Priority` is carried from the
  originating PR review.
- **[Deferred / Documented](#deferred--documented)** — known gaps that are **not currently
  actionable**: blocked on a methodology derivation, on external tooling (R / Stata / Julia)
  absent from CI, parked pending user demand / out of paper scope, or explicitly won't-fix.
  Retained for provenance and AI-review deviation-documentation — **do not pull from here
  without first clearing the named blocker.**
- **[Reference / Status](#reference--status)** — not backlog: user-facing limitations,
  module-size monitoring, deprecations, and current-state notes.

The `Origin` column (Actionable tables) and the `PR` column (Deferred tables) both point to the originating PR number or review tag.

---

## Actionable Backlog

### Methodology / correctness

| Issue | Location | Origin | Effort | Priority |
|-------|----------|--------|--------|----------|
| `SyntheticControl` conformal (CWZ 2021) AR / innovation-permutation path (Lemmas 5-7) for time-series proxies — the residual-permutation shortcut is only valid for time-permutation-invariant proxies (SC/Lasso/DiD); an AR proxy needs innovation permutation. One-sided alternatives (Remark 1 signed statistic) and proxy covariates (eq 4/6 note) SHIPPED 2026-07. | `diff_diff/conformal.py`, `diff_diff/synthetic_control_results.py` | CWZ-2021 | Heavy | Low |
| `ContinuousDiD` CGBS-2024 extensions. (a) `covariates=` kwarg — **DONE (reg/dr)**; (b) discrete-treatment saturated regression (`treatment_type="discrete"`) — **DONE**; (c) lowest-dose-as-control per Remark 3.1 when `P(D=0)=0` (`control_group="lowest_dose"`) — **DONE** (discrete + continuous mass-point, single-cohort; estimand `ATT(d)−ATT(d_L)`; see REGISTRY Note #7). Remaining (all deferred `NotImplementedError`, documented): `estimation_method="ipw"` on the dose curve (scalar-adjustment / degenerate); `covariates=` × `survey_design=` (weighted OR + weighted nuisance IF); multi-cohort **heterogeneous-support** discrete aggregation (support-aware: average each dose only over the cohorts that observe it); **multi-cohort `lowest_dose`** (within-cohort `d_L` reference + support-aware cross-cohort aggregation); and **`covariates=` × `lowest_dose`** (conditional-PT-relative-to-`d_L` estimand). Single-cohort / 2-period / shared-support multi-cohort are supported. | `continuous_did.py` | CGBS-2024 | Heavy | Low |

### Performance

Consolidates the former `#### Performance` tech-debt table and the standalone
`## Performance Optimizations` section. (Speculative / low-value perf notes — numba JIT,
generic sparse-FE, QR+SVD rank-detection redundancy, `check_finite` bypass — moved to
[Deferred → Parked](#parked--pending-user-demand--out-of-scope).)

| Issue | Location | Origin | Effort | Priority |
|-------|----------|--------|--------|----------|
| `EfficientDiD` conditional path: the largest remaining O(n) stage is the sieve/nuisance construction outside the tiled pass (~9s at 10k). (The `_ridge_solve_weights` Python-prep shave landed 2026-07-07 — the `omega_stack[rest]` fancy-index copy and tail scatter are skipped when no row is zero-masked, byte-identical outputs; the `zero_mask` abs scan itself remains, needed for correctness.) | `efficient_did_covariates.py` | CS-scaling | Mid | Low |
| Evaluate flipping `DIFF_DIFF_SOLVE_OLS_FASTPATH` default-ON after an opt-in soak (the 2026-07 certified normal-equations Cholesky fast path, both backends). A flip needs: golden/parity-suite recapture at the tol-bounded posture (fitted ~1e-8 abs / SE ~1e-6 rel — the default today is byte-pinned in several benchmark conventions), certification-rate telemetry across real workloads (any decline is silent-correct but forfeits the speedup), and the staged default-flip protocol used for `df_convention` (v4-class change). | `diff_diff/linalg.py::_resolve_solve_ols_fastpath`, `rust/src/linalg.rs::solve_ols_chol` | CS-scaling | Mid | Low |

### Testing / docs

---

## Deferred / Documented

Not currently actionable. Retained for provenance + AI-review deviation-documentation.

### Paper-gated / needs methodology derivation

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| `PlaceboTests` `boundary_gap` — a permutation randomization-inference margin (SE-audit item (b)); NOT computed anywhere in code today, so this is a new feature + result field, not a coverage lock. **User-locked 2026-07-09: defer until a derivation/paper source exists** — do not design or implement from scratch. | `tests/test_methodology_placebo.py`, `diff_diff/diagnostics.py` | SE-audit | Low |
| `SyntheticControl` fit-snapshot residency (`_SyntheticControlFitSnapshot`) — **investigated 2026-07-07, parked**: the snapshot ALIASES the fit's own working pivots (zero extra construction cost); the retained residency implements the documented freeze contract (post-fit mutation of estimator inputs must not change `in_space_placebo()` / `leave_one_out()` / conformal output on an already-returned results object, and `__getstate__` already excludes it from pickles). A compact array representation saves only pandas overhead (the float panel dominates); releasing residency needs new API surface (`release`/opt-out flag) or a freeze-contract change. Revisit on user demand for very large donor panels. | `synthetic_control.py`, `synthetic_control_results.py` | follow-up | Low |
| Stratified survey-PSU multiplier-weight draw-tiling — **investigated 2026-07-07, parked**: the stratified generator (`generate_survey_multiplier_weights_batch`) consumes ONE sequential rng stream stratum-major (`rng.choice(size=(n_bootstrap, n_h))` per stratum, then lonely-PSU pooling), so draw-chunked assembly CANNOT reproduce the stream bit-identically (contra the old row's parenthetical) — it would need per-stratum generator state skipping (PCG64.advance + per-weight-type variate accounting; fragile) or a stream-layout change (MC-level SE changes → baseline/golden recapture + REGISTRY note). Stratified designs have few PSUs, so the full `(n_bootstrap × n_psu)` matrix rarely matters; unstratified (the large-`n_units` case) is already tiled. Revisit only if a large-PSU stratified design hits memory, as a documented stream change. | `diff_diff/bootstrap_chunking.py::iter_survey_multiplier_weight_blocks` | follow-up | Low |
| CBWSDID covariate balancing (`StackedDiD(balance="entropy")`) v1 supports only balanced event windows + `weighting="aggregate"`; unbalanced/ragged panels fail closed (unit-count vs observation-count corrector convention unresolved off balanced panels). Matching-based balancing and the repeated `0→1`/`1→0` episode extension are also deferred. Documented in REGISTRY StackedDiD "Covariate balancing (CBWSDID)" Notes. | `stacked_did.py`, `balancing.py`, REGISTRY | follow-up | Low |
| dCDH: Phase-1 per-period placebo `DID_M^pl` has NaN SE (no IF derivation for the per-period aggregation path). Multi-horizon placebos (`L_max ≥ 1`) have valid SE. | `chaisemartin_dhaultfoeuille.py` | #294 | Low |
| dCDH: survey cell-period allocator's post-period attribution is a library convention, not derived from the observation-level survey linearization. MC coverage is empirically close to nominal; a formal derivation (or covariance-aware two-cell alternative) is deferred. Documented in REGISTRY survey IF expansion Note. | `chaisemartin_dhaultfoeuille.py`, REGISTRY | #408 | Medium |
| dCDH by_path: survey-aware backward-horizon (`placebo + predict_het + survey_design`) raises `NotImplementedError` (and `_compute_heterogeneity_test` warn-and-skips to forward-horizon-only heterogeneity) — the Binder TSL cell-period allocator's REGISTRY justification is tied to post-period attribution; backward horizons would put ψ_g mass on a pre-period cell. Needs the pre-period cell allocator derived. | `chaisemartin_dhaultfoeuille.py`, REGISTRY | follow-up | Medium |
| **HonestDiD Δ^RM ARP confidence sets** (consolidates the former "Honest DiD Improvements" checklist): uses a naive FLCI instead of the paper's ARP conditional/hybrid sets (Sections 3.2.1-3.2.2). ARP infrastructure exists but the moment-inequality transformation needs calibration; CIs are conservative (valid coverage). Sub-items folded here: improved C-LF via direct optimization instead of grid search (`honest_did.py:947`); hybrid inference methods; event-study-specific bounds per post-period; simulation-based power analysis for honest bounds. (`CallawaySantAnnaResults` support has **landed**.) | `honest_did.py` | #248 | Medium |
| **Conley `vcov_type` for IF / GMM estimators** (consolidates 8 near-identical rows). No reference implementation exists for any of these spatial-HAC × influence-function/GMM compositions; each was rejected at `__init__` with a deferral pointer here. SunAbraham + WooldridgeDiD-OLS conley have **shipped** (within-transform via `solve_ols`). Per estimator: <br>• `CallawaySantAnna` — Conley kernel × per-(g,t) IF aggregation (`staggered.py`). <br>• `TripleDifference` — × the 3-pairwise-DiD IF decomposition `w3·IF_3 + w2·IF_2 - w1·IF_1` (`triple_diff.py`). <br>• `ImputationDiD` — × Theorem-3 per-unit IF `sigma_sq = (cluster_psi_sums**2).sum()` (`imputation.py`). <br>• `EfficientDiD` — × per-unit EIF `_compute_se_from_eif` (`efficient_did.py`). <br>• `TwoStageDiD` — thread into the GMM sandwich meat `_compute_gmm_variance`; the SpilloverDiD `_compute_gmm_corrected_meat` machinery could be adapted to score `S_g = gamma_hat' c_g - X'_{2g} eps_{2g}` but two-stage-GMM × Conley has no reference (`two_stage.py`). <br>• `StackedDiD` — **methodology-blocked, not plumbing**: the stacked design replicates each control unit across sub-experiments, so Conley's distance matrix sees same-unit copies at distance 0 (`K(0)=1`); needs a per-stack spatial identifier (`stacked_did.py`). <br>• `SyntheticDiD` — uses `variance_method ∈ {bootstrap, jackknife, placebo}`, no analytical sandwich for Conley to plug into; needs an analytical-sandwich path or a spatial-block bootstrap (Politis-Romano 1994) (`synthetic_did.py`). <br>• Conley + survey weights / `survey_design` — score-reweighting is mechanical but the PSU×spatial-kernel interaction and replicate-weight spatial variance are non-trivial (Bertanha-Imbens 2014 covers cluster-sample, not Conley); raises `NotImplementedError` at the linalg validator (`linalg.py::_validate_vcov_args`). | (per sub-item) | follow-up · Phase 1b · Phase 5 | Low-Med |
| `HeterogeneousAdoptionDiD` Phase 4.5 C still-open: (a) **replicate-weight designs** (BRR/Fay/JK1/JKn/SDR) — per-replicate weight-ratio rescaling for the OLS-on-residuals refit isn't covered by the multiplier-bootstrap composition; each linearity-family helper raises `NotImplementedError` on replicate weights. (b) **`lonely_psu='adjust'` + singleton-strata** on the Stute family — the pseudo-stratum centering transform isn't derived for the Stute CvM functional. | `had_pretests.py` | Phase 4.5 C | Low |
| `HeterogeneousAdoptionDiD` mass-point `vcov_type in {hc2, hc2_bm}` raises `NotImplementedError` — OLS leverage `x_i'(X'X)^{-1}x_i` is wrong for 2SLS; needs the `x_i'(Z'X)^{-1}(...)(X'Z)^{-1}x_i` correction plus an R/Stata (`ivreg2 small robust`) parity anchor. | `had.py::_fit_mass_point_2sls` | Phase 2a | Medium |
| `HeterogeneousAdoptionDiD` `trends_lin × survey_design`: per-group linear-trend slope under survey weighting is not derived from the paper. Raises `NotImplementedError` across all 3 `trends_lin` surfaces. | `had.py`, `had_pretests.py` | #389 | Low |
| `SpilloverDiD(survey_design=...)` replicate-weight variance (BRR/Fay/JK1/JKn/SDR): Wave E.1 ships Taylor-linearization only. Per Gerber (2026) Appendix A the IF-reweighting shortcut does NOT apply to TwoStageDiD-class estimators (`gamma_hat` is weight-sensitive); correct support needs per-replicate full re-fit of both stages. | `spillover.py`, `survey.py::compute_replicate_refit_variance` | follow-up | Low |
| `SpilloverDiD(vcov_type="conley", conley_lag_cutoff>0, survey_design=...)` no-effective-PSU serial Bartlett HAC: weights-only / strata-only designs without a cluster fallback raise `NotImplementedError` (each pseudo-PSU appears in one period, so the serial cross-period loop contributes zero). Needs a unit-level serial fallback derivation or routing through `conley_unit` with documented IF-allocator asymmetry. | `spillover.py`, `two_stage.py::_compute_stratified_serial_bartlett_meat` | Wave E.2 tail | Low |
| `SpilloverDiD` data-driven `d_bar` selection (Butts 2021b / 2023 JUE Insight cross-validation). | `spillover.py` | follow-up | Low |
| **`LPDiD` non-absorbing exit-event dynamics** (Dube et al. 2025 online Appendix C `eta_h^{g,n}`): the shipped `non_absorbing` modes estimate the **entry-effect** estimands (Eq. 12/13) only; separate dynamic event-studies for treatment switch-*offs* are not implemented. Needs the exit-event clean-sample derivation + estimand contract. | `lpdid.py`, REGISTRY | PR-C follow-up | Low |
| **`LPDiD` non-absorbing interior-gap support**: non-absorbing modes require a gap-free panel within each unit's observed span and raise on interior time gaps (the `[t-L, t+h]` window conditions can't be verified across a gap). The absorbing path already reindexes interior gaps to the calendar grid; extending that fail-closed handling (per-window gap masking) to non-absorbing is deferred. | `lpdid.py::_prepare_panel` | PR-C follow-up | Low |

### Needs external reference (R / Stata / Julia)

Blocked on tooling absent from CI (no workflow installs R/Stata/Julia). A clear path
exists but parity can't be verified without a local toolchain.

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| `StaggeredTripleDifference` R cross-validation: CSV fixtures not committed (gitignored); tests skip without local R + `triplediff`. Commit fixtures or generate deterministically. | `tests/test_methodology_staggered_triple_diff.py` | #245 | Medium |
| `StaggeredTripleDifference` R parity: benchmark only tests the no-covariate path (`xformla=~1`). Add covariate-adjusted scenarios + aggregation-SE parity assertions. | `benchmarks/R/benchmark_staggered_triplediff.R` | #245 | Medium |
| `StaggeredTripleDifference` per-cohort group-effect SEs include WIF (conservative vs R's `wif=NULL`); documented in REGISTRY. Could override the mixin for an exact R match (verification needs R `triplediff`). | `staggered_triple_diff.py` | #245 | Low |
| **WooldridgeDiD follow-up cluster** (PR-B Stage D/E fail-closed surfaces; re-enable after R/Stata validation): <br>• QMLE sandwich uses `aweight` cluster adjustment `(G/(G-1))·(n-1)/(n-k)` vs Stata's `G/(G-1)` (conservative); add a `qmle` weight type if Stata goldens confirm a material difference (`wooldridge.py`, `linalg.py`). <br>• response-scale APE / log-link coefficient bridge for R `etwfe(family=poisson|logit)` cell-level parity — needs `emfx()` APE extraction or link-inversion with baseline-mean adjustment (`generate_wooldridge_golden.R`, `test_methodology_wooldridge.py`). <br>• `aggregate(weights="cohort_share")` on survey-weighted fits: `_n_g_per_cohort` uses raw `unit.nunique()`; implement design-weighted unit totals per cohort (paper W2025 §7) and lift the `ValueError` gate (`wooldridge.py`, `wooldridge_results.py`). <br>• unconditional inference for `cohort_share` accounting for ω̂_g sampling uncertainty (W2025 §7.5); currently NaN-closed (`wooldridge_results.py`). <br>• `cohort_trends=True × survey_design` and `× control_group="never_treated"` raise `NotImplementedError` (unvalidated TSL variance / trend columns spanned by placebo cell-dummies) (`wooldridge.py`). <br>• Stata `jwdid` golden-value `TestReferenceValues` (`tests/test_wooldridge.py`). | `wooldridge.py`, `wooldridge_results.py`, `linalg.py`, benchmarks | #216 · PR-B | Med-Low |
| Extend `WooldridgeDiD` `method ∈ {logit, poisson}` with `vcov_type ∈ {classical, hc2, hc2_bm}`: composing HC2 leverage + Bell-McCaffrey DOF with the QMLE pseudo-residual sandwich needs derivation + R parity vs `clubSandwich::vcovCR(glm, type="CR2")`. Rejected at `__init__`. | `wooldridge.py` | follow-up | Medium |
| `PreTrendsPower` CS/SA `anticipation=1` R-parity fixture: R `pretrends` has no anticipation parameter, so the Python `_extract_pre_period_params` anticipation filter isn't R-parity-locked. Build a synthetic CS/SA result with `anticipation=1` and assert γ_p matches R's `slope_for_power()`. (Mechanism already covered by MC + full-VCV tests.) | `tests/test_methodology_pretrends.py`, `generate_pretrends_golden.R` | PR-C | Low |
| Harmonize SunAbraham's HC1 within-transform finite-sample correction with `fixest::sunab()` — SA applies `n/(n-k_dm)`, fixest applies `n/(n-k_total)` (counts absorbed FE); ~1-2% SE difference, documented as a "Deviation from R" and pinned at `atol=5e-3`. Either thread `df_adjustment` or keep as an intentional, R-verified difference. | `sun_abraham.py`, `linalg.py` | follow-up | Low |
| Absorbed-FE **clustered** CR1 with *non-nested* FE: for `absorb=[FE1,FE2], cluster=FE1` (e.g. `absorb=["unit","time"], cluster="unit"`), `fixest` counts the non-nested FE (time) in the CR1 `(n-1)/(n-k)` finite-sample denominator, but the clustered path uses only `k_visible`. D4 harmonized the *non-clustered* classical/hc1 full-K scale (`_absorbed_fe_vcov_scale`) and left the clustered path unchanged — correct for FE nested in the cluster, a small deviation for non-nested FE (documented in REGISTRY within-transform note). Thread a non-nested `df_adjustment` into the clustered CR1 factor; verify vs `fixest::feols(..., cluster=)`. | `linalg.py`, `estimators.py` | SE-audit D4 | Low |
| Rust multiplier-bootstrap weight RNG (`generate_bootstrap_weights_batch`) seeds `Xoshiro256PlusPlus::seed_from_u64(seed+i)` per row; audit Python callers (`sdid.py`, `efficient_did_bootstrap.py`, `bootstrap_utils.py`) for parity-test gaps and, where a numpy-canonical equivalent exists, pre-generate in Python and pass through PyO3 (same fix shape as TROP RNG parity #354). | `rust/src/bootstrap.rs`, `bootstrap_utils.py` | follow-up | Medium |
| `SyntheticDiD` bootstrap cross-language parity anchor vs R `synthdid::vcov(method="bootstrap")` or Julia `Synthdid.jl` (refit-native). Same-library validation is in place; Julia is the cleanest target. Tolerance ~1e-6 (BLAS+RNG paths preclude 1e-10). | `benchmarks/R/`, `benchmarks/julia/`, `tests/` | follow-up | Low |
| CS R helpers hard-code `xformla = ~1`; no covariate-adjusted R benchmark for the IRLS path. | `tests/test_methodology_callaway.py` | #202 | Low |
| `CallawaySantAnna` bootstrap: align p-value computation with R `did`'s symmetric-percentile method (former "CallawaySantAnna Bootstrap Improvements" section). | `staggered.py` | — | Low |
| **`bias_corrected_local_linear` (lprobust) Phase-1c follow-ups:** extend golden parity to `kernel ∈ {triangular, uniform}` (epa-only today); expose `vce ∈ {hc0,hc1,hc2,hc3}` on the public wrapper once R goldens exist (port supports all four; needs a per-mode generator + a hc2/hc3 q-fit-leverage decision); clustered-DGP auto-bandwidth parity is **blocked upstream** on an nprobust singleton-cluster bug in `lpbwselect.mse.dpi` (Phase-1c DGP 4 uses manual `h=b=0.3`). | `_nprobust_port.py`, `local_linear.py`, `generate_nprobust_lprobust_golden.R` | Phase 1c | Low-Med |
| `HeterogeneousAdoptionDiD` Stute-family Stata-bridge parity: no public R `Stutetest` package exists; would add `benchmarks/stata/generate_stute_golden.do` + a Stata dependency. | `benchmarks/stata/`, `tests/test_stute_test_parity.py` | follow-up | Low |
| **`LPDiD` regression-adjustment SE — no runnable R reference.** The RA influence-function cluster SE is canonically Stata `teffects ra ... atet vce(cluster)` only; no R package computes it (`alexCardazzi/lpdid` does direct covariate inclusion, not RA). Today the RA *point* is R-anchored (~1e-12), the SE is pinned + MC-coverage-validated (`coverage_lpdid_ra.py`). Follow-up: contribute the RA path to `alexCardazzi/lpdid` so a runnable R RA reference exists — only a *trusted* anchor once cross-checked vs Stata `teffects` (else circular). | `tests/test_methodology_lpdid.py`, `benchmarks/python/coverage_lpdid_ra.py` | #B2 follow-up | Low |
| **`LPDiD` survey scope gaps (PR-D1 deferrals).** Survey support covers the variance-weighted default path only. (a) `survey_design` + `reweight=True` (the equally-weighted / regression-adjustment IF path) is rejected: the weighted RA influence-function variance has **no runnable survey reference** (same class as the RA-SE row above - `survey::svyglm` anchors only the OLS/WLS path). (b) Replicate-weight survey designs (BRR/Fay/JK1/JKn/SDR) and (c) non-pweight (fweight/aweight) types are rejected pending demand. | `lpdid.py`, REGISTRY #8 | PR-D1 | Low |
| **`LPDiD` non-absorbing R-parity - DONE (PR-C2)** via an independent `fixest::feols` Eq. 12/13 reconstruction (point+SE ~1e-13/~1e-15 vw; `effect_stabilization` reweighted point + pinned SE). `alexCardazzi/lpdid`'s `nonabsorbing_lag` proved NOT a faithful Eq. 13 (off-switch clamp + non-paper boundary/placebo window; diverges ~0.01-0.05 even on a monotone panel), so it is recorded as a divergent reference, not a gate. **Residual external-reference gap:** the authors' canonical non-absorbing SE/RA is Stata `lpdid`/`teffects` only (no faithful R analogue) - same class as the absorbing RA-SE row above; revisit if a Stata toolchain or a corrected R package appears. | `benchmarks/R/generate_lpdid_golden.R`, `tests/test_methodology_lpdid.py` | PR-C2 | Low |
| `HeterogeneousAdoptionDiD` Phase-3 R-parity: ships coverage-rate validation on synthetic DGPs, not tight point parity vs `chaisemartin::stute_test` / `yatchew_test` (needs bootstrap-seed-semantics + `B` alignment across numpy/R). | `tests/test_had_pretests.py` | Phase 3 | Low |

### Parked — pending user demand / out of scope

Doable in principle, but no current caller and/or explicitly out of paper scope.

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| Rust-backend CR2 Bell-McCaffrey port (`return_dof` in the Rust vcov dispatch + CR2 algebra) — **premise re-scoped 2026-07-09**: the scores-based DOF + low-rank factored `A_g` changes made the NumPy CR2-BM path BLAS-bound (`O(n_g k²)` per cluster; 4.1s→38ms at n=100k/k=40), so a Rust port buys ~nothing and adds a parity surface. Revisit only if profiling shows CR2-BM hot again. | `rust/src/linalg.rs` | — | Low |
| Clustered-CR1 inference df **default flip to `"cluster"` (G−1) at v4** — the opt-in `df_convention=` knob landed 2026-07 (DiD/TWFE/MPD + LinearRegression; REGISTRY §TwoWayFixedEffects deviation note); the remaining work is the major-version default change (moves every clustered p-value/CI) + migration note + flipping `TestDfConvention`/`test_moderate_t_pins_residual_df_convention` expectations. Also evaluate extending the knob to standalone estimators with CR1-t inference at that time. | `diff_diff/linalg.py::LinearRegression`, `diff_diff/estimators.py`, `diff_diff/twfe.py` | — | Medium |
| CallawaySantAnna **unbalanced-panel R parity — LANDED** via `allow_unbalanced_panel=True` (matches R `did::att_gt(allow_unbalanced_panel=TRUE)` / `DRDID::reg_did_rc`: ATT bit-exact on cells AND dynamic aggregation via fixed unit-cohort-mass `pg` + a per-unit WIF; SE up to the documented CR1 `sqrt(G/(G-1))` factor). The earlier "weighting" framing was a mis-diagnosis — on unbalanced panels the dominant divergence from R is the *estimator* (within-cell differencing vs RC-on-pooled-obs), not only the weighting; both are resolved by the flag. The DEFAULT path keeps within-cell differencing as a documented design choice and now emits a `UserWarning` on unbalanced input (no-silent-failures). **Remaining deferred:** `survey_design=` × `allow_unbalanced_panel=` (per-obs vs per-unit weight resolution — currently fail-closed `NotImplementedError`); and covariate / ipw / dr × the flag R-parity verification (the RC path supports them; the committed golden covers `reg` no-cov). | `staggered.py`, `staggered_aggregation.py` | SE-audit D3 | Low |
| CallawaySantAnna event-study bucket/weight construction is duplicated between the analytical aggregator (`staggered_aggregation.py::_aggregate_event_study`) and the multiplier bootstrap (`staggered_bootstrap.py`): both group (g,t) by `e = t - g`, apply the finite/NaN/reference masks, and read cohort weights. Both already consume the same source-materialized universal reference cells (so they agree), but the bucket logic is copy-pasted. Extract one shared helper returning per-event-time buckets (finite cells, NaN cells, reference flags, cohort weights, combined-IF inputs) used by both. Pure refactor; gate on byte-identical analytical + bootstrap output. | `staggered_aggregation.py`, `staggered_bootstrap.py` | SE-audit D3 | Low |
| `StackedDiD` survey re-resolution intra-file dedup (raw-weight extraction ×3, compose-normalize ×3, resolve-on-stacked ×2). The cross-estimator ContinuousDiD/EfficientDiD panel-to-unit collapse consolidation LANDED (#226 shared helpers `ResolvedSurveyDesign.subset_to_units_by_row_idx` / `build_unit_first_row_index`); StackedDiD is deliberately NOT on that path (control units are duplicated across sub-experiments, so it re-resolves at stacked granularity rather than collapsing to one row per unit). The residual is stacked-specific, low value, and touches the numerically-sensitive composed-weight renormalization. Post-filter re-resolution / metadata-recompute unification across the three estimators was assessed and is not warranted — they use genuinely different mechanisms and already delegate to shared `_resolve_survey_for_fit` / `compute_survey_metadata`. | `stacked_did.py` | #226 | Low |
| `solve_ols` residual memory floor is inherent to SVD-based lstsq after the PR-E marshalling slim (rust-side allocator high-water 15.32 -> 7.81 GB on the 2.4M x 130 clustered solve; remaining = one fused equilibrated input copy + thin-SVD U transient + vcov scores block + LAPACK gelsd internals on the python path). Further reduction requires the tall-skinny-QR algorithm change - same trade-off as the parked QR-reuse row below (library-wide last-digit perturbation + loses the second independent collinearity detector); the two land together if ever reopened. Diagnosis + measurements in docs/performance-plan.md. (The 2026-07 opt-in `DIFF_DIFF_SOLVE_OLS_FASTPATH` Cholesky path sidesteps the floor when enabled — no thin-SVD U transient, scores reuse the one equilibrated buffer; this row tracks the DEFAULT path only.) | `rust/src/linalg.rs::solve_ols`, `diff_diff/linalg.py` | PR-E | Low |
| SunAbraham-family fits are SOLVER-bound after the v3.6.x demeaning speedups: on the county-class shape (3.1k units x 60 months, ~100 interaction columns) `solve_ols` = ~60% of fit time (faer SVD 1.24s + pivoted-QR rank detection 0.70s of 3.27s), and the share grows further under the Rust demean kernel; the pivoted QR alone is ~20% of the whole fit (measured attribution in docs/performance-plan.md). Candidate fix is QR-reuse (one factorization for rank detection + solve). **Parked per the 2026-07 correctness-first decision**: the change perturbs every coefficient in the library at the last digits (golden/parity/bit-identity recapture) and removes the SVD's independent singular-value truncation - a second, different near-collinearity detector that is deliberate defense-in-depth for the no-silent-failures contract (the FE-span guard episode showed borderline cases are real). Re-open only on practitioner demand for wide event studies; any implementation gates on fixest/R end-to-end parity, not just internal consistency. (2026-07 updates: the pivoted-QR share quoted above predates #634's stage-0 Gram certification — rank detection is now ~0.03s on the county shape — and the opt-in `DIFF_DIFF_SOLVE_OLS_FASTPATH` Cholesky path is the certification-gated alternative that captures the solver win WITHOUT the library-wide perturbation: the default stays byte-identical and keeps the SVD truncation as the second collinearity detector.) | `linalg.py::solve_ols`, `linalg.py::_detect_rank_deficiency` | PR-C attribution | Medium |
| dCDH parity-test SE/CI assertions only cover pure-direction scenarios; mixed-direction SE comparison is structurally apples-to-oranges (cell-count vs obs-count weighting). | `test_chaisemartin_dhaultfoeuille_parity.py` | #294 | Low |
| `HeterogeneousAdoptionDiD` joint cross-horizon covariance / sup-t bands: per-horizon SEs use independent sandwiches (paper-faithful pointwise CIs per Pierce-Schott Fig 2). Follow-ups (low demand): IF-based stacking for joint cross-horizon inference; analytical H×H covariance on the weighted ES path. (A sup-t band on the unweighted ES path shipped via the clustered band — `cluster=` fires the simultaneous band on the unweighted path.) | `had.py::_fit_event_study` | Phase 2b / 4.5 B | Low |
| `HeterogeneousAdoptionDiD` event-study staggered timing beyond the last cohort: Phase 2b auto-filters to the last cohort (paper App B.2); earlier-cohort effects aren't HAD-identified (redirect to dCDH). Full staggered HAD needs a different identification path (out of paper scope). | `had.py::_validate_had_panel_event_study` | Phase 2b | Low |
| `HeterogeneousAdoptionDiD` survey-aware support-endpoint test (**research, waits on literature**): needs a calibrated support-infimum test under complex sampling (endpoint EVT × survey-aware functional CLT × tail-empirical-process theory). Permanent `NotImplementedError` on `qug_test(survey_design=...)`; rationale in REGISTRY § "QUG Null Test" Note (Phase 4.5 C0). | `had_pretests.py::qug_test` | Phase 4.5 C0 | Low |
| `HeterogeneousAdoptionDiD` Phase-4.5 weight-aware auto-bandwidth MSE-DPI selector (~300 LoC); users pass `h`/`b` explicitly today. Plus replicate-weight SurveyDesigns on the continuous-dose paths (Rao-Wu-style per-replicate weight-ratio rescaling for the local-linear intercept IF). | `_nprobust_port.py::lpbwselect_mse_dpi`, `had.py::_aggregate_unit_resolved_survey` | Phase 4.5 | Low |
| `HeterogeneousAdoptionDiD` Phase-4 Pierce-Schott (2016) replication harness — **waived (2026-05-20)**: R parity at `atol=1e-8` on the same 3 DGPs is a strictly stronger anchor than reproducing Fig 2's pointwise CIs on the LBD-restricted PNTR panel (paper §5.2 self-acknowledges NP estimators too noisy there). Re-open only on user demand. See REGISTRY HAD Deviations Notes #3/#4. | `benchmarks/`, `tests/` | Phase 2a | Low |
| `HeterogeneousAdoptionDiD` time-varying dose on event study: Phase 2b rejects panels where `D_{g,t}` varies within a unit for `t≥F` (constant-dose convention, App B.2). A time-varying-dose estimator is a future PR; current behavior is front-door rejection. | `had.py::_validate_had_panel_event_study` | Phase 2b | Low |
| `HeterogeneousAdoptionDiD` repeated-cross-section support: paper §2 allows panel OR RCS, but Phase 2a is panel-only (RCS inputs rejected by the balanced-panel validator). Needs an RCS identification path (pre/post cell means) with its own validator + `data_mode` surface. | `had.py::_validate_had_panel` | Phase 2a | Medium |
| `HeterogeneousAdoptionDiD` Phase-3 nprobust bandwidth for Stute variants on continuous regressors (currently OLS residuals from a 2-parameter linear fit, no bandwidth selection). Not in paper scope. | `had_pretests.py::stute_test` | Phase 3 | Low |
| `SpilloverDiD(ring_method="count")`: count-of-treated-in-ring (paper §3.2) is methodologically supported by Butts but re-introduces functional-form dependence; expose behind an explicit kwarg gate + warning. | `spillover.py` | follow-up | Low |
| `TwoStageDiD` paper-permitted estimand variants (Gardner 2022): the Eq.(5) P̄-period-average estimand and the fn.8 full-sample first-stage variant have no public parameter. Documented ⚠️ in `gardner-2022-review.md`; surface as `estimand=` / `first_stage=` if a use case arises. | `two_stage.py` | follow-up | Low |
| `bias_corrected_local_linear` multi-eval grid (`neval > 1`) with cross-covariance (`covgrid=TRUE`). Not needed for HAD; useful for multi-dose diagnostics. | `_nprobust_port.py::lprobust` | Phase 1c | Low |
| Rust local-method `estimate_model` → unify to `solve_wls_svd` (the global-method's SVD helper) for sub-1e-14 bootstrap-SE parity. The local-method bootstrap parity test passes at `atol=1e-5`; the residual ~1e-7 is roundoff, not a user-visible correctness bug. | `rust/src/trop.rs`, `rust/src/linalg.rs` | follow-up | Low |
| Validate the `.txt` AI guides (`llms-full.txt`, `llms-practitioner.txt`) as executable snippets — **not low-lift** (re-scoped 2026-06-01): only ~20% of ~112 fenced blocks are standalone-runnable; the rest are signature pseudo-code, context fragments, or data-shape-specific. Needs signature-block detection + a context/data skip-allowlist + per-snippet fixtures. | `tests/test_doc_snippets.py` | #239 | Low |
| `TestWorkflowDoesNotExecutePRHeadCode` (CodeQL #14 guard) doesn't model `bash/sh/./source <script>` execution, multi-line `python3 -c` bodies, shell-var indirection, `eval`, `find -exec`, `xargs -I`. Catches common accidental regressions (16 forms); closing the residuals needs multi-line shell parsing + script-exec allowlists — diminishing return given the documented threat model. | `tests/test_openai_review.py`, `.github/workflows/ai_pr_review.yml` | #436 | Low |
| Calendar-time aggregation (R `did` feature gap) — blocks 1 ported test in `test-att_gt.R`. | — | — | Low |
| **Speculative / low-value performance notes** (relocated from the old `## Performance Optimizations`): numba JIT for bootstrap loops — **blocked by the numpy/pandas/scipy-only dependency policy**; generic sparse-matrix handling for large FE; QR+SVD rank-detection redundancy in `solve_ols` (QR overhead is minimal vs the SVD solve — correctness over micro-opt; `skip_rank_check` already exists for known-full-rank hot paths); incomplete `check_finite=False` bypass (scipy's QR in `_detect_rank_deficiency()` still validates; edge-case only). All `Low`, none correctness-affecting. | `linalg.py::solve_ols` | — | Low |

### Won't-fix / waived (decisions on the record)

| Decision | Location | Verified |
|----------|----------|----------|
| **StackedDiD intercept SEs not parity-lockable (SE-audit C1/(c)).** Measured ~0.3% divergence from R: a nuisance-parameter reference-cell/parameterization gap, not machine-precision lockable (the event-study interaction SEs already match ~2e-13). Surfacing `se_cr1/cr2_intercept` would add an unasserted, R-divergent public field. Waived with the SE-audit row closure. | `tests/test_methodology_stacked_did.py` | SE-audit / 2026-07-09 |
| **estimatr `classical` intercept SE excluded from parity (SE-audit (d)).** Same documented `O(1/n)` projection/DOF deviation as the slope; reference-only. Waived with the SE-audit row closure. | `tests/test_estimatr_iv_robust_parity.py` | SE-audit / 2026-07-09 |
| **`bread_inv` reuse not bit-identically achievable.** "Factor `(X'WX)` once, reuse across HC2/HC2-BM" can't be done bit-identically (the bar for a pure perf refactor of the inference path). Internal bread ops solve against *different* RHS (`X.T`, `eye`, `meat`+`temp.T`, `contrasts`); only same-RHS results are bit-reusable. Measured: `lu_solve(lu_factor(A),B)` differs from `solve(A,B)` up to 6.4e-15; the `inv(A)@meat@inv(A)` sandwich differs up to 1.24e-14 — both nonzero and *below* the affected goldens' tolerances (1e-12/1e-10), so a broad reuse would silently shift SEs without tripping the suite. The one genuine bit-identical redundancy (a duplicated `solve(bread, X.T)` in the unweighted one-way `hc2_bm`+`return_dof` path) is dwarfed by that path's dense `M=I−H` build, so the saving is negligible. | `linalg.py::compute_robust_vcov` | 2026-06-01 |
| **R-script-per-test consolidation has no CI impact.** No CI workflow installs R, so every R-parity test skips in CI behind a per-file availability gate — consolidating `Rscript` spawns yields zero CI speedup. `test_methodology_twfe.py` already session-caches its R fits. The only residual is a LOCAL-dev micro-opt for `test_methodology_continuous_did.py` / `test_methodology_callaway.py` (re-spawn `library(...)` per call). Low value; retained as a local-dev note. | `tests/test_methodology_continuous_did.py`, `tests/test_methodology_callaway.py` | #139 / 2026-06-07 |
| **`HeterogeneousAdoptionDiD` mass-point IV bread is non-symmetric — `_rank_guarded_inv` inapplicable.** The structural rank-guard sweep (continuous_did / two_stage / spillover / conley) excluded `had.py`'s `ZtWX = Zd'WX` ([1, instrument]' × [1, endogenous]): it is a non-symmetric 2×2 Wald-IV bread (`V = ZtWX_inv @ Omega @ ZtWX_inv.T`), and `_rank_guarded_inv` assumes a **symmetric PSD** Gram (symmetric `D=diag(A)` equilibration + eigendecomposition), so applying it would be methodologically wrong. The existing fallback already returns NaN SE on a singular bread; an IV-appropriate near-singular guard would need a different mechanism. | `had.py` | structural-rank-guard / 2026-06-28 |
| **`ImputationDiD` SE vcov is already rank-guarded upstream.** Excluded from the structural rank-guard sweep: the lead/effect vcov comes from `solve_ols(..., return_vcov=True, rank_deficient_action=...)` at the OLS fit (`imputation.py:~2316`), which already drops rank-deficient columns. The only raw inverse (`solve(V_gamma, gamma)`, `imputation.py:~2530`) is the pretrends **Wald F-test statistic** with a safe `NaN` fallback — a test statistic, not a sandwich bread — so there is no garbage-SE exposure. No structural rank-guard needed. | `imputation.py` | structural-rank-guard / 2026-06-28 |
| **TWFE HC2/HC2-BM full-dummy dedup: drift-prone duplication already resolved by the shared builder; full delegation waived.** The former Actionable row (origin: follow-up review, citing pre-#655 line numbers) asked to extract a shared dummy-construction helper or delegate TWFE's HC2/HC2-BM path to DiD's `fixed_effects=` branch. The shared-helper half SHIPPED in #655: both sites now delegate dummy construction, drop-first convention, FE column naming, and the duplicate-term backstop to the single `build_fe_dummy_blocks` (`utils.py`) + `validate_design_term_names` implementation (`twfe.py::fit` full-dummy branch; `estimators.py::DifferenceInDifferences.fit` `fixed_effects=` branch) — the FE-naming / survey-behavior drift risk the row targeted is gone. What remains per site is ~4 lines of genuinely estimator-specific design-matrix assembly (TWFE stacks `const`/`ATT`/covariates; DiD stacks its formula terms), which is not drift-prone duplication. The remaining full-delegation option — routing `TWFE.fit` through DiD machinery with TWFE-specific cluster-default threading — would touch TWFE's user-visible result surface (coefficient-dict keys, cluster-label conventions, warning text) for near-zero residual benefit; waived on cost/benefit. | `twfe.py::fit`, `estimators.py::DifferenceInDifferences.fit`, `utils.py::build_fe_dummy_blocks` | #655 / 2026-07-10 |
| **Survey TSL SE intentionally counts genuine-subpopulation zero-weight PSUs (matches R, NOT a bug).** Re-examined the former "count only positive-weight PSUs in the correction" item (origin PR-B). `_compute_stratified_psu_meat`'s finite-sample correction `(1-f_h)·n_{PSU,h}/(n_{PSU,h}-1)` and PSU-mean centering keep zero-weight PSUs — this is the **full-design domain-estimation convention** (Lumley 2004 §3.4; R `survey::svyrecvar(subset())`), already documented in REGISTRY § "Subpopulation Analysis" (the survey-vcov path deliberately differs from the positive-weight invariance applied *outside* it). The ATT is exactly invariant; the SE is intentionally NOT invariant to genuine-subpopulation zeroing (it *should* differ from a naive physical subset — that is the whole point of `subpopulation()`). Repro (`scratchpad`): zeroing a full PSU vs physically dropping it differs ~5e-3 rel — the Lumley-correct gap, and R's `svyrecvar(subset())` produces the matching SE (only `df` differs; see the § "Subpopulation Analysis" Deviation note). The only truly invariance-violating shape — appending *synthetic new* all-zero PSUs — does not arise in any estimator path (real padding reuses existing PSU labels and is already bit-invariant, or is genuine domain estimation via `prep.py`'s zero-padded full-design cell variance). "Fixing" the meat to positive-weight-only would break the documented Lumley/R parity. Regression-locked by `tests/test_survey.py::TestZeroWeightPsuConventionWaiver`. | `survey.py` (`_compute_stratified_psu_meat`) | PR-B / 2026-06-30 |

---

## Reference / Status

Not backlog — current-state notes, monitoring, and scheduled removals.

### Known Limitations (user-facing)

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported (falls back to analytical) | `estimators.py:1647` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:890-911` | Low | Rarely needed |

For survey-specific limitations (NotImplementedError paths), see the
[Current Limitations](docs/survey-roadmap.md#current-limitations) section of survey-roadmap.md.

### Large Module Files

Target: ideally < 1000 lines per module; modules ≥3000 lines are candidates for splitting,
2000-3000 are monitored, 1000-2000 are accepted as a cohesion / scope trade-off. Updated
2026-05-15.

| File | Lines | Action |
|------|-------|--------|
| `chaisemartin_dhaultfoeuille.py` | 8636 | Consider splitting (per-path / placebos / survey IF / aggregation) |
| `had_pretests.py` | 4951 | Consider splitting (Stute / Yatchew / QUG / joint pretests) |
| `had.py` | 4593 | Consider splitting (continuous / mass-point / event-study / survey paths) |
| `staggered.py` | 3963 | Consider splitting — grew through survey + aggregation features |
| `linalg.py` | 3601 | Consider splitting (vcov surfaces) only if cohesion preserved — unified backend; vcov / solver paths tightly coupled |
| `diagnostic_report.py` | 3380 | Consider splitting (per-method renderers + provenance) |
| `power.py` | 3196 | Consider splitting (power analysis + MDE + sample size) |
| `synthetic_did.py` | 2819 | Monitor — variance methods + survey paths |
| `honest_did.py` | 2785 | Monitor |
| `business_report.py` | 2653 | Monitor — per-method narrative renderers |
| `imputation.py` | 2475 | Monitor |
| `survey.py` | 2466 | Monitor — grew with Phase 6 features |
| `utils.py` | 2396 | Monitor |
| `prep_dgp.py` | 2057 | Monitor |
| `triple_diff.py` | 2053 | Monitor |
| `estimators.py` | 1991 | Acceptable |
| `two_stage.py` | 1985 | Acceptable |
| `chaisemartin_dhaultfoeuille_results.py` | 1981 | Acceptable |
| `prep.py` | 1876 | Acceptable |
| `efficient_did.py` | 1793 | Acceptable |
| `sun_abraham.py` | 1713 | Acceptable |
| `continuous_did.py` | 1682 | Acceptable |
| `results.py` | 1676 | Acceptable |
| `staggered_triple_diff.py` | 1619 | Acceptable |
| `_nprobust_port.py` | 1412 | Acceptable |
| `practitioner.py` | 1402 | Acceptable |
| `trop_global.py` | 1350 | Acceptable |
| `trop_local.py` | 1339 | Acceptable |
| `local_linear.py` | 1332 | Acceptable |
| `wooldridge.py` | 1305 | Acceptable |
| `chaisemartin_dhaultfoeuille_bootstrap.py` | 1175 | Acceptable |
| `bacon.py` | 1144 | Acceptable |
| `pretrends.py` | 1133 | Acceptable |
| `stacked_did.py` | 1050 | Acceptable |
| `conley.py` | 1006 | Acceptable |
| `visualization/` | 4316 | Subpackage (split across 7 files) — OK |

### Standard Error Consistency

`vcov_type` has subsumed the previously-proposed `se_type` knob. `DifferenceInDifferences`
and `TwoWayFixedEffects` accept `vcov_type ∈ {classical, hc1, hc2, hc2_bm, conley}`
(the validated set in `linalg.py::_VALID_VCOV_TYPES`); cluster-robust variance comes from
`cluster=` alongside the heteroscedasticity kind (`hc1+cluster` ⇒ CR1 Liang-Zeger;
`hc2_bm+cluster` ⇒ CR2 Bell-McCaffrey, including the weighted WLS-CR2 port; the N>1
absorbed-FE + weights composition is supported via iterative alternating-projection demeaning, #586);
wild cluster bootstrap is the separate `inference="wild_bootstrap"` path. Threading
`vcov_type` through the 8 standalone estimators is **complete** (Phase 1b); four
(`CallawaySantAnna`, `TripleDifference`, `ImputationDiD`, `EfficientDiD`) are permanently
narrow to `{hc1}` per their influence-function variance, and `TwoStageDiD` is likewise
narrow (Gardner GMM meat has no single cross-stage hat matrix). The per-estimator
`vcov_type="conley"` extensions: SunAbraham + WooldridgeDiD-OLS are **shipped**; the
IF/GMM estimators are tracked in
[Deferred → Paper-gated](#paper-gated--needs-methodology-derivation).

### Type Annotations

Mypy reports 0 errors. All mixin `attr-defined` errors resolved via `TYPE_CHECKING`-guarded
method stubs in the bootstrap mixin classes.

### Test Coverage

Visualization tests skip when matplotlib / plotly are not installed (see
`pytest.importorskip` markers in `tests/test_visualization*.py`).

### Deprecated Code

- `lambda_reg` and `zeta` in `SyntheticDiD` (`synthetic_did.py`) — deprecated in favor of
  `zeta_omega` / `zeta_lambda`; remove in v4.0.0 (public kwarg removal requires a major bump).

### RuntimeWarnings — Apple Silicon M4 BLAS bug (numpy < 2.3)

Spurious RuntimeWarnings ("divide by zero", "overflow", "invalid value") are emitted by
`np.matmul`/`@` on Apple Silicon M4 + macOS Sequoia with numpy < 2.3, for matrices with
≥260 rows. They **do not affect correctness** (coefficients/fitted values are valid, designs
full rank). Root cause: Apple's BLAS SME kernels corrupt the FP status register
([numpy#28687](https://github.com/numpy/numpy/issues/28687),
[#29820](https://github.com/numpy/numpy/issues/29820); fixed in numpy ≥ 2.3 via
[PR #29223](https://github.com/numpy/numpy/pull/29223)). Not reproducible on M3, Intel, or
Linux.

- `linalg.py:162` — warnings in fitted-value computation (`X @ coefficients`); seen in
  `test_prep.py` during treatment-effect recovery (n > 260).
- `triple_diff.py:307,323` — warnings in propensity-score computation (IPW/DR with
  covariates); logistic-regression overflow in edge cases (separate from the BLAS bug).
- **Long-term:** revert to the `@` operator when numpy ≥ 2.3 becomes the minimum supported
  version.
