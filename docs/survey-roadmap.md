# Survey Data Support: History and Current State

This document is the technical reference for survey-design support in
diff-diff. It records the build history (Phases 1-10) as shipped and
documents current limitations. Forward-looking roadmap items live in
[ROADMAP.md](../ROADMAP.md); this file is the historical and technical
companion.

---

## What's Shipped

### Phases 1-2: Core Infrastructure

- `SurveyDesign` class with weights, strata, PSU, FPC, weight_type, nest, lonely_psu
- Taylor Series Linearization (TSL) variance with strata + PSU + FPC
- Weighted OLS, sandwich estimator, demeaning, survey degrees of freedom
- `SurveyMetadata` on results (effective n, DEFF, weight_range)
- Base estimators: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD

### Phase 3: OLS-Based Standalone Estimators

| Estimator | Survey Support | Notes |
|-----------|----------------|-------|
| StackedDiD | pweight only | Q-weights compose multiplicatively; fweight/aweight rejected |
| SunAbraham | Full | Bootstrap via Rao-Wu rescaled |
| BaconDecomposition | Diagnostic | Weighted descriptives only, no inference |
| TripleDifference | Full | Regression, IPW, and DR methods with TSL on IFs |
| ContinuousDiD | Full | Weighted B-spline OLS + TSL; bootstrap via multiplier at PSU |
| EfficientDiD | Full | No-cov and DR covariate paths both survey-weighted; bootstrap via multiplier at PSU |

### Phase 4: Complex Estimators + Weighted Logit

| Estimator | Survey Support | Notes |
|-----------|----------------|-------|
| ImputationDiD | Full | Weighted iterative FE + conservative variance; bootstrap via multiplier at PSU |
| TwoStageDiD | Full | Weighted FE + GMM sandwich; bootstrap via multiplier at PSU |
| CallawaySantAnna | Full | Strata/PSU/FPC/replicate weights; IPW/DR covariates (Phase 7a); replicate IF variance |

Weighted `solve_logit()` in `linalg.py` — survey weights enter IRLS as
`w_survey * mu * (1 - mu)`.

### Phase 5: SyntheticDiD + TROP

| Estimator | Survey Support | Notes |
|-----------|----------------|-------|
| SyntheticDiD | pweight (placebo / jackknife / bootstrap); strata/PSU/FPC (all three methods — bootstrap via PR #355 weighted FW + Rao-Wu; placebo via stratified permutation + weighted FW; jackknife via PSU-level LOO with stratum aggregation). `lonely_psu="adjust"` not supported on the jackknife path (use `"remove"` / `"certainty"` or switch to `bootstrap`). | Treated means survey-weighted; omega composed with control weights post-optimization. Bootstrap survey path uses weighted-FW + Rao-Wu rescaling per draw. Placebo full-design permutes pseudo-treated within strata containing actual treated units (requires at least one stratum with `n_c > n_t`; exact-count designs raise Case D `ValueError`). Jackknife full-design leaves out one PSU at a time and aggregates per Rust & Rao (1996); full-census strata (`f_h ≥ 1`) short-circuit to zero contribution. |
| TROP | pweight | Population-weighted ATT aggregation; model fitting unchanged |

### Phase 6: Advanced Features (v2.7.6)

- **Survey-aware bootstrap** for bootstrap-using estimators:
  multiplier at PSU (CS, Imputation, TwoStage, Continuous, Efficient)
  and Rao-Wu rescaled (SA, SyntheticDiD, TROP). SyntheticDiD bootstrap
  composes Rao-Wu rescaled per-draw weights with the **weighted Frank-Wolfe**
  variant (PR #355): each draw solves the weighted objective
  ``min ||A·diag(rw)·ω - b||² + ζ²·Σ rw_i ω_i²`` and composes
  ``ω_eff = rw·ω/Σ(rw·ω)`` for the SDID estimator. See REGISTRY.md
  §SyntheticDiD ``Note (survey + bootstrap composition)`` for the full
  derivation. SyntheticDiD's `placebo` and `jackknife` methods now also
  support full strata/PSU/FPC designs: placebo via stratified permutation
  + the same weighted FW kernel; jackknife via PSU-level LOO with
  stratum aggregation (Rust & Rao 1996). See REGISTRY.md §SyntheticDiD
  "Note (survey + placebo composition)" and "Note (survey + jackknife
  composition)" for objectives and limitations.
- **Replicate weight variance**: BRR, Fay's BRR, JK1, JKn, SDR.
  12 of 16 estimators supported (not SyntheticDiD, TROP, BaconDecomposition, or WooldridgeDiD)
- **DEFF diagnostics**: per-coefficient design effects vs SRS baseline
- **Subpopulation analysis**: `SurveyDesign.subpopulation()` preserves
  full design structure for correct variance

### Phase 7: Completing the Survey Story (v2.8.0-v2.8.1)

- **7a.** CS IPW/DR covariates + survey: DRDID nuisance IF corrections
  (Sant'Anna & Zhao 2020, Theorem 3.1)
- **7b.** Repeated cross-sections: `CallawaySantAnna(panel=False)` matching
  `DRDID::reg_did_rc`, `drdid_rc`, `std_ipw_did_rc`
- **7c.** Survey tutorial: `docs/tutorials/16_survey_did.ipynb` with full
  workflow (strata, PSU, FPC, replicates, subpopulation, DEFF)
- **7d.** HonestDiD + survey: survey df and event-study VCV propagated
  to sensitivity analysis with t-distribution critical values
- **7e.** Staggered DDD survey support (only implementation in R or Python
  with design-based DDD variance). Reached via `TripleDifference` with
  `first_treat=` since 3.9; `StaggeredTripleDifference` is deprecated
  (row M-013) but runs the same engine until its 4.0 removal.

### Phase 8: Survey Maturity (v2.8.3-v2.8.4)

- **8a.** SDR replicate method for ACS PUMS (80 columns)
- **8b.** FPC in ImputationDiD and TwoStageDiD
- **8c.** Silent operation warnings (8 operations now emit `UserWarning`)
- **8d.** Lonely PSU "adjust" in bootstrap (Rust & Rao 1996)
- **8e.** CV on estimates, `trim_weights()`, survey-aware ImputationDiD pretrends
- **8f.** Compatibility matrix in `choosing_estimator.rst`

### Phase 9: Real-Data Validation (v2.9.0)

15 cross-validation tests against R's `survey` package using real federal
survey datasets:

| Dataset | Design | Key result |
|---------|--------|------------|
| API (R `survey`) | Strata + FPC | ATT, SE, df, CI match R (7 variants incl. subpopulation, Fay's BRR) |
| NHANES (CDC/NCHS) | Strata + PSU (nest=TRUE) | ACA DiD matches R for strata+PSU, covariates, subpopulation |
| RECS 2020 (U.S. EIA) | 60 JK1 replicate weights | Coefficients, SEs, df, CI match R |

Files: `benchmarks/R/benchmark_realdata_*.R`, `tests/test_survey_real_data.py`,
`benchmarks/data/real/*_realdata_golden.json`

### Documentation Remaining (Phase 8g)

- **Multi-stage design**: not yet documented. Single-stage (strata + PSU)
  is sufficient per Lumley (2004) Section 2.2.
- **Post-stratification / calibration**: DOCUMENTED (2026-07). `SurveyDesign`
  expects pre-calibrated weights; calibration stays upstream by design. The
  recommended companion is Meta's `balance` package (>= 0.21), which ships a
  dedicated `balance.interop.diff_diff` adapter (`pip install "balance[did]"`).
  The handoff is documented in `docs/api/prep.rst` ("Weight calibration with
  balance"), demonstrated end-to-end in
  `docs/tutorials/26_composition_drift_calibration.ipynb` (including when
  calibration is essential for the causal estimand, not just descriptives),
  and the consumed diff-diff surface is pinned by
  `tests/test_balance_interop_contract.py`. `samplics` (read-only; successor
  `svy` not yet released) and `weightipy` remain alternatives.

### Phase 10: Survey Completeness (v2.9.0–v3.0)

- **10a.** Survey theory document (`survey-theory.md`) — formal justification for design-based variance with modern DiD influence functions
- **10b.** Research-grade survey DGP — 9 parameters on `generate_survey_did_data()` (8 research-grade + `conditional_pt`)
- **10c.** R validation expansion — 8 of 16 estimators cross-validated against R's `survey::svyglm()`
- **10d.** Tutorial rewrite — flat-weight vs design-based comparison with known ground truth
- **10f.** WooldridgeDiD survey support — OLS, logit, Poisson paths with `pweight` + strata/PSU/FPC + TSL variance
- **10g.** LPDiD survey support — variance-weighted default path via `fit(survey_design=...)` with `pweight` + strata/PSU/FPC + stratified-PSU Taylor-linearization variance (`survey::svyglm` parity); reweight/regression-adjustment, replicate-weight, and non-pweight designs rejected (deferred follow-ups)

### v3.0.1: Survey Aggregation Helper

`aggregate_survey()` (in `diff_diff.prep`) bridges individual-level survey
microdata (BRFSS, ACS, CPS, NHANES) to geographic-period panels for
second-stage DiD estimation. Computes design-based cell means using domain
estimation (Lumley 2004 S3.4), with SRS fallback for small cells. Returns a
panel DataFrame plus a pre-configured `SurveyDesign` for the second-stage
fit. Default `second_stage_weights="pweight"` (population weights) is
compatible with all survey-capable estimators; opt-in `"aweight"` (precision
weights) provides efficiency-weighted estimates for estimators that accept it.
Supports both TSL and replicate-weight variance.

See `docs/api/prep.rst` for the API reference and `docs/methodology/REGISTRY.md`
for the methodology entry.

### Phase 4.5 C: HAD Stute Survey Workflow ✅ Shipped

The HeterogeneousAdoptionDiD pretest family (`stute_test`,
`stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`,
and the composite `did_had_pretest_workflow`) gained end-to-end
support for `SurveyDesign(strata=..., psu=..., weights=..., fpc=...)`
in PR #432 (2026-05). The Stute CvM bootstrap on stratified survey
designs uses a documented synthesis of clustered-wild-bootstrap
ingredients (Cameron-Gelbach-Miller 2008 cluster-level multipliers;
Davidson-Flachaire 2008 wild-bootstrap centering; Wu 1986 / Liu 1988
Bessel small-sample correction; Djogbenou-MacKinnon-Nielsen 2019
cluster-wild consistency for nonlinear functionals): within-stratum
demean + `sqrt(n_h/(n_h-1))` rescale on the PSU multipliers BEFORE
the per-obs broadcast in the wild-residual loop. The shared helper
`bootstrap_utils.apply_stratum_centering` backs both the new Stute
path and the existing HAD sup-t event-study cband bootstrap. The QUG
step remains permanently deferred under survey designs (Phase 4.5
C0); the workflow surfaces this in `report.qug=None` plus the
`_QUG_DEFERRED_SUFFIX` substring on `report.verdict`. Tutorial 22
(`docs/tutorials/22_had_survey_design.ipynb`) walks the workflow
end-to-end on a BRFSS-shape state-rollout panel.

Remaining HAD survey-path deferrals (separate follow-up PRs):
`lonely_psu='adjust'` + singleton strata (pseudo-stratum centering
transform not yet derived for the Stute functional — same gap as the
HAD sup-t deviation at REGISTRY:2382); replicate-weight designs
(BRR / Fay / JK1 / JKn / SDR — separate Rao-Wu / JKn bootstrap
composition).

---

## Phase 10: Academic Grounding (History)

The Phase 10 items established the theoretical and empirical foundation
for survey-design variance estimation on modern DiD influence functions.
All items below are shipped; this section documents what was done and
why.

### 10a. Theory Document ✅

`docs/methodology/survey-theory.md` lays out the formal argument for
design-based variance estimation with modern DiD influence functions:

1. Modern heterogeneity-robust DiD estimators (CS, SA, BJS) are smooth
   functionals of the weighted empirical distribution
2. Survey-weighted empirical distribution is design-consistent for the
   finite-population quantity (Hájek/design-weighted estimator)
3. The influence function is a property of the functional, not the
   sampling design — IFs remain valid under survey weighting
4. TSL (stratified cluster sandwich) and replicate-weight methods are
   valid variance estimators for smooth functionals of survey-weighted
   estimating equations (Binder 1983, Rao & Wu 1988, Shao 1996)

This is the short-term deliverable that can be linked from docs and README
immediately.

**Key references:**
- Binder, D.A. (1983). "On the Variances of Asymptotically Normal
  Estimators from Complex Surveys." *International Statistical Review* 51.
- Rao, J.N.K. & Wu, C.F.J. (1988). "Resampling Inference with Complex
  Survey Data." *JASA* 83(401).
- Shao, J. (1996). "Resampling Methods in Sample Surveys." *Statistics* 27.

### 10b. Survey Simulation DGP ✅

Enhanced `generate_survey_did_data()` with 8 research-grade parameters:
`icc`, `weight_cv`, `informative_sampling`, `heterogeneous_te_by_strata`,
`te_covariate_interaction`, `covariate_effects`, `strata_sizes`, and
`return_true_population_att`. All backward-compatible. Supports panel
and repeated cross-section modes.

**Resolved:** `conditional_pt` parameter added. When nonzero, shifts treated
units' x1 mean by +1 SD and adds `conditional_pt * x1_i * (t/T)` to the
outcome, creating X-dependent time trends. Unconditional PT fails; conditional
PT holds after covariate adjustment. DR/IPW estimators recover truth.

### 10c. Expand R Validation Coverage ✅

8 of 16 estimators now cross-validated against R's `survey::svyglm()`:
DifferenceInDifferences, TWFE, CallawaySantAnna, SyntheticDiD,
ImputationDiD, StackedDiD, SunAbraham, TripleDifference.

### 10d. Tutorial: Show the Pain ✅

Survey tutorial rewritten with side-by-side flat-weight vs design-based
comparison using the research-grade DGP from 10b, showing known ground
truth, coverage simulation, and false pre-trend detection rates.

### 10f. WooldridgeDiD Survey Support ✅

WooldridgeDiD (ETWFE) now supports `survey_design` for all three methods
(OLS, logit, Poisson) with `pweight` only (`fweight`/`aweight` rejected).
OLS uses survey-weighted within-transformation + WLS + TSL vcov.
Logit/Poisson use survey-weighted IRLS + X_tilde linearization for TSL
vcov. Replicate-weight designs raise `NotImplementedError`; bootstrap +
survey is rejected.

Two further combinations raise rather than subsetting the frame in place:
comparison-support filtering (periods with no untreated unit) and
unidentified-cohort exclusion. Both would delete rows, which under a complex
design removes their PSUs and strata from the variance — see the Current
Limitations table. The refusals are conditional: a survey fit that drops
nothing is unaffected.

### 10g. Practitioner Guidance ✅

Subsumed by the practitioner decision tree
(`docs/practitioner_decision_tree.rst`) and the practitioner
getting-started guide (`docs/practitioner_getting_started.rst`).
The Brand Awareness Survey DiD tutorial
(`docs/tutorials/17_brand_awareness_survey.ipynb`) demonstrates the
full workflow end-to-end; DEFF diagnostics provide the empirical signal
for whether survey design matters on a given dataset.

---

## Current Limitations

All items below raise an error when attempted, with a message describing
the limitation and suggested alternative.

| Estimator | Limitation | Alternative |
|-----------|-----------|-------------|
| LWDiD | Any `survey_design` / sampling weights | No weight argument exists on any path, so the failure mode is a bare `TypeError: unexpected keyword argument` rather than a descriptive error (the exception to the preamble above). The LW papers derive the transformation and exact-inference layer for unweighted panels; a weighted counterpart is DEFERRED pending user demand. Use `CallawaySantAnna` (or another survey-capable staggered estimator) when design-based variance is required. |
| SyntheticDiD | Replicate weights | Pre-existing limitation: no replicate-weight survey support on SDID. All three variance methods (bootstrap, placebo, jackknife) now support pweight-only and strata/PSU/FPC designs; replicate-weight designs remain rejected. |
| TROP | Replicate weights | Use strata/PSU/FPC design with Rao-Wu rescaled bootstrap |
| BaconDecomposition | Replicate weights | Diagnostic only, no inference |
| ImputationDiD | `pretrends=True` + replicate weights | Use analytical survey design instead |
| ImputationDiD | `pretrend_test()` + replicate weights | Use analytical survey design instead |
| DiD, TWFE | `inference='wild_bootstrap'` + `survey_design` | Use analytical survey inference (default) |
| EfficientDiD | `cluster` + `survey_design` | Use `survey_design` with PSU/strata |
| WooldridgeDiD | Unsupported-period filtering + `survey_design` | Restrict the frame to the supported periods explicitly and re-fit. Deleting rows in-place is naive subsetting: it removes their PSUs and strata from the TSL meat and from `df_survey = n_PSU - n_strata`. Exact only if every PSU and stratum survives the restriction (true on a balanced panel; NOT in general — an unbalanced frame can hold a PSU observed only at unsupported periods). Verify before relying on it. |
| WooldridgeDiD | Unidentified-cohort exclusion + `survey_design` | Same reason (ledger `M-123`). Drop the cohort from the frame yourself, or supply a panel where every cohort has a pre-treatment period. |
| All bootstrap estimators | Bootstrap + replicate weights | These are alternative variance methods; pick one |
| CS, DMLDiD, EfficientDiD, ImputationDiD, TwoStageDiD | `aggregate('total')` on fits declaring a `survey_design` | The estimator-owned total (3.10) is panel non-survey only: the realized-mass relay omits the survey mass-uncertainty (att*dC) variance term and design-aware population-scale totals are not implemented (retained weight scale differs by design family). Pass a caller-derived numeric `scale=` to the MMM exporters instead, or use `cluster=` (without `survey_design`) for an unweighted clustered fit. Tracked in DEFERRED.md (Paper-gated). |

**Warning/fallback (no error):** MultiPeriodDiD with `wild_bootstrap` +
`survey_design` warns and falls back to analytical inference.

**Resolved (2026-07):** CallawaySantAnna `reg`+covariates (survey and
unweighted) now carries the full `DRDID::reg_did_panel` estimation-effect
IF correction; the former "conservative plug-in IF" deviation is removed
(see REGISTRY.md, CallawaySantAnna standard-error notes).
