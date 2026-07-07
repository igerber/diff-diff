# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.7.0] - 2026-07-08

### Removed
- **HAD pretest helpers: deprecated `survey=` / `weights=` kwargs removed (3.7.x).**
  Completes the HAD survey-design API consolidation started with
  `HeterogeneousAdoptionDiD.fit` in 3.7.0: all 7 pretest surfaces
  (`did_had_pretest_workflow`, `qug_test`, `stute_test`, `yatchew_hr_test`,
  `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`) now accept
  `survey_design=` only — passing the old aliases raises `TypeError`. Migration:
  array-in helpers take `survey_design=make_pweight_design(arr)` (pweight-only) or a
  pre-resolved `ResolvedSurveyDesign`; data-in surfaces add the weights as a column and
  pass `survey_design=SurveyDesign(weights='col_name', ...)` — the former row-level
  `weights=` array shortcut is gone (per-unit aggregation + mean-1 normalization made
  the two forms numerically identical, so no results change under migration).
  Surviving `survey_design=` / unweighted paths are byte-identical (no computational
  code touched). Dead code removed with the aliases: the 3-way alias mutexes, the
  per-surface deprecation-message constants (`diff_diff/survey.py`), the workflow's
  internal row-level weight alignment + forwarding machinery, and the joint wrappers'
  staggered weight-subsetting blocks (`SurveyDesign` column references self-align).
  `qug_test` still permanently rejects `survey_design=` (Phase 4.5 C0).
- **`HeterogeneousAdoptionDiD.fit()` no longer accepts the deprecated `survey=` and
  `weights=` kwargs** (the pre-scheduled removal for the 3.7.0 minor bump; the
  `DeprecationWarning` shipped in a prior release with "will be removed in the next minor
  release"). Weighting is now expressed solely through the canonical `survey_design=`,
  matching `CallawaySantAnna` / `EfficientDiD` / `ImputationDiD` / `TwoStageDiD` (which all
  take `survey_design=` only and compute design-based Binder (1983) Taylor-linearization
  variance via the shared `compute_survey_if_variance`).
  - **`survey=SurveyDesign(...)`** -> **`survey_design=SurveyDesign(...)`** (pure rename; same
    Binder-TSL path, byte-identical output).
  - **`fit(weights=<array>)`** -> add the weights as a column on `data` and pass
    **`survey_design=SurveyDesign(weights='col')`**. This is a variance-family change: the old
    per-row `weights=` array produced a CCT-2014 pweight-robust / 2SLS-sandwich SE with Normal
    inference (`variance_formula="pweight"` / `"pweight_2sls"`); the canonical path produces
    Binder-TSL with `t(df_survey)` inference (`variance_formula="survey_binder_tsl"` /
    `"survey_binder_tsl_2sls"`). Point estimates are unchanged; SEs move ~0.1%+.
  - **`fit(weights=<array>, cluster=<col>)`** (the weighted-CR1 2SLS sandwich) has no drop-in
    equivalent - migrate to **`survey_design=SurveyDesign(weights='w', psu='cluster_col')`**
    (Binder-TSL clustered through the PSU).
  - **`cband` is now keyword-only** on `fit()` (it previously followed the removed positional
    `survey`/`weights` slots; keyword-only prevents a silent positional misbind).
  - Note: this public-kwarg removal in a *minor* bump is the pre-scheduled HAD exception
    documented since the deprecation shipped; other pending removals (e.g. `SyntheticDiD`
    `lambda_reg`/`zeta`) remain gated on the next major (v4.0.0). The equivalent
    `survey=`/`weights=` kwargs on the HAD pretest helpers were removed separately —
    completed by the pretest-helper removal entry above in this same Unreleased section.

### Fixed
- **Rust clustered vcov is now run-to-run deterministic.** The cluster-score aggregation
  in `compute_robust_vcov` (also reached via `solve_ols(return_vcov=True)`) built its
  (G, k) cluster-scores matrix in `HashMap` iteration order, which is SipHash-randomized
  per call — mathematically identical, but the GEMM accumulation order changed on every
  invocation, wobbling the vcov at ~1e-14 (3 distinct values observed across 8 identical
  calls; the Python backend was bit-stable). Rows now accumulate in first-appearance
  order — ascending for the factorized 0..G-1 ids the Python dispatcher passes, matching
  NumPy's groupby order. Verified: 1 distinct value across 50 identical calls (was 3/8);
  NumPy parity unchanged (~1e-14, the normal cross-backend GEMM tolerance); bit-identity
  regression tests cover both contiguous and non-contiguous unsorted cluster ids.
- **`ImputationDiD` pretrends lead model gains the FE-span snap guard.** The Test-1 lead
  indicators + covariates now route through `snap_absorbed_regressors` after the
  within-transform (the same two-stage snap + LSMR confirmation the `absorb=` estimators
  use). A lead whose calendar period contains only its cohort's rows on the untreated
  sample collapses to a calendar-time dummy in the span of the absorbed time FE; it now
  snaps to exact zero — deterministic NaN coefficient + cause-specific warning naming
  `lead[h]` — instead of relying on the raw rank check alone, which the documented
  truncated-MAP-iterate exposure can defeat in slow-convergence regimes (junk direction
  perturbing the identified lead coefficients). Identified leads are unchanged (the full
  imputation suites pass unmodified); behavioral tests lock both the spanned-NaN contract
  and the no-op case. REGISTRY ImputationDiD note added.
- **`HonestDiD` Δ^SD optimal-FLCI center parity with R (SE-audit B2b).** The optimal
  Fixed-Length CI optimizer was a flat Nelder-Mead over slope weights that landed on a
  different affine estimator than R `HonestDiD::findOptimalFLCI` at intermediate smoothness
  `M` — the CI **center** drifted up to ~9% (the width/coverage matched). Replaced it with a
  faithful port of R's **nested convex program**: an inner minimum-worst-case-bias problem at a
  fixed estimator SD `h` (a smooth convex QCQP over slope weights, solved with `scipy` SLSQP —
  no cvxpy) and an outer 1-D search over `h`. Now matches R's optimal FLCI **center +
  half-length + optimalVec** to ~1e-3 (median ~1e-5) across a stress grid. diff-diff's
  **analytical** folded-normal critical value is strictly more accurate than R's Monte-Carlo
  `.qfoldednormal`, so it also solves the deterministic version of R's problem more precisely
  than stock R. Verified vs R HonestDiD 0.2.6 (`benchmarks/data/honest_flci_golden.json`,
  `TestHonestFLCIParityR`); the M=0 result and all existing behaviour are unchanged.

### Testing
- **Callaway-Sant'Anna golden tolerances tightened to machine precision (SE-audit C6
  closure).** The no-covariate DR golden asserted ATT only, within 0.02, and never
  asserted the SE; measured agreement with R `did` is ~6e-11 ATT / ~2e-11 relative SE
  (no-covariate DR is deterministic algebra — no propensity IRLS enters), so both are
  now pinned at 1e-8 with the SE assertion added. The reg (0.02) and ipw (0.05) ATT
  bands — predating the DRDID estimation-effect IF terms — are tightened to 1e-8 on
  measured ~3e-11/~4e-11 agreement. The covariate-DR scenario keeps its documented 2e-3
  band (~1e-3 DR small-sample numerics via the propensity nuisance).
- **fixest hetero + cluster SE machine-precision locks on an unbalanced, heteroskedastic
  DGP (SE-audit G2 completion).** The committed `fixest_did_twfe_golden.json` gains two
  appended scenarios (error sd varying by arm/period, ~15% rows dropped; the original
  balanced scenarios' RNG draws precede them and reproduce value-identically): on the
  plain-OLS DiD path, `hetero` (HC1) no longer collapses to iid and is locked against
  `fixest` at machine precision — and the cluster-robust CR1 SE turns out to match fixest
  **exactly** on plain OLS (balanced and unbalanced), so the former ~0.5% DiD cluster
  band-pin is tightened to a machine-precision lock: the documented ~0.25% fixest-CR1
  DOF-convention deviation is an absorbed-FE (within-transform) phenomenon only. The TWFE
  cluster band-pin is retained and re-scoped to that documented non-nested-FE ssc
  deviation (~0.3% unbalanced); TWFE `hetero` has no public unclustered surface
  (auto-cluster-at-unit convention), so its scenario locks iid — which also pins the D4
  full-K rescale on an UNBALANCED panel for the first time.
- **`ImputationDiD` covariate-path R parity anchor.** The no-covariate staggered panel was
  the only `didimputation` R anchor; the covariate branch (first-stage imputation model
  `y ~ x | unit + time` on the untreated sample, R `first_stage = ~ 0 + x | unit + time`
  == diff-diff `covariates=["x"]`) now has its own golden: a time-varying, unit-correlated
  covariate panel appended to `generate_didimputation_golden.R` (the base scenario's RNG
  draws precede the new block, so the committed base panel and golden values reproduce
  byte-identically) and a `TestImputationDiDCovariateParityR` class pinning overall +
  per-horizon event-study ATT and SE. Observed agreement on the reference platform:
  SE ~2e-10 (the covariate-augmented untreated `v_it` projection + clustering machinery),
  ATT ~2e-7; asserted at abs=1e-6/1e-7 for cross-platform robustness.
- **reviewer-eval harness: tutorial-notebook cases now reviewed with CI-equivalent
  context.** `ci_prompt` reproduces the CI workflow's `<notebook-prose>` block for changed
  `docs/tutorials/*.ipynb` (extraction via `tools/notebook_md_extract.py` with the same
  per-output / per-notebook / aggregate caps, fail-soft per notebook, pre-extract
  test-then-append truncation with an omitted-notebooks marker, zero-extracted fallback,
  close-tag sanitization, and the untrusted wrapper + out-of-wrapper warning), appended
  after the unified diff exactly as CI does. The `verify-corpus`/`run` tutorial-case
  rejection guards are lifted. Documented divergence (same rationale as `pr_review.md`
  sourcing): the extractor runs from the current repo, not each case's base SHA. Covered
  by seven new adapter tests (wrapper + sanitization, zero-extracted, aggregate
  truncation, byte-vs-char cap parity, trusted-extractor sentinel, Git-quoted
  filename discovery, end-to-end prompt assembly order).
- **True half-sample BRR replicate regressions per estimator family.** The
  replicate-weight expansion tests used Fay-like 0.5/1.5 perturbations, under which every
  unit keeps positive weight; true BRR was covered only at the vcov-helper level. A new
  Hadamard-balanced half-sample generator (paired 2-PSU pseudo-strata; selected PSU w*2,
  the other exactly 0 — the `survey::brrweights` full-BRR convention) now backs a
  per-family regression class (DiD, DiD-absorb, MultiPeriodDiD, TWFE, SunAbraham,
  StackedDiD, ImputationDiD, TwoStageDiD): finite positive replicate SEs under genuine
  half-samples plus a base-weights point-estimate invariance check on EVERY family
  (replicate columns drive only the variance). Construction sanity is itself asserted (exactly half the
  paired PSUs zeroed per replicate, all multipliers in {0, 2}). Notably, genuine
  half-samples CAN lose identification inside a replicate refit on some designs (the
  staggered binary-interaction TWFE parameterization) — TWFE fails loudly there; the
  family test uses its dedicated 2-period panel, with the behavior noted in-test.
- **`CallawaySantAnna` ipw R-parity yardsticks folded into the golden fixture + no-covariate
  ipw structural-parity decision recorded.** `csdid_golden_values.json` regenerated (R 4.5.2,
  did 2.5.1, DRDID 1.3.0): all pre-existing data and result blocks reproduced byte-identically;
  the ipw scenario now carries the `aggte` simple/dynamic/group blocks (identical to the
  previously hardcoded 2026-07-05 yardsticks), and
  `test_golden_ipw_aggregation_se_vs_r_did_251` reads them from the JSON. The no-covariate ipw
  branch's unconditional-propensity treatment is now a recorded document-only decision
  (REGISTRY § CallawaySantAnna): R `did`'s intercept-only logit is deliberately not mirrored —
  its estimation-effect correction is identically zero, and no-covariate ipw/reg/dr reduce to
  the same difference-in-means IF, locked bit-identical per cell by a new
  `TestDRNoCovariateSEUniformity::test_ipw_no_cov_per_cell_identical_to_reg`.
- **CI-locked standard-error parity for flagship and previously-unasserted paths (SE-audit
  coverage batch).** These surfaces computed SEs matching R but had no CI assertion pinning them
  (the latent-risk pattern that once hid the CallawaySantAnna reg-method gap):
  - **New `fixest` DiD/TWFE golden** (`benchmarks/data/fixest_did_twfe_golden.json`, generated by
    `benchmarks/R/generate_fixest_did_twfe_golden.R`): the flagship 2×2 DiD and TWFE classical/iid
    ATT **and SE** now match `fixest::feols` to ~1e-16 in CI without R at test time (the prior
    live-`Rscript` tests only asserted `att` at rtol=1e-3 and skipped in CI). The TWFE assertion
    also locks the D4 full-K within-transform rescale.
  - **CR2 backbone**: `TwoWayFixedEffects(vcov_type="hc2_bm")`'s default CR2-clustered-at-unit SE
    and the one-way HC2-BM SE (previously only their Satterthwaite DOF was asserted); the
    `MultiPeriodDiD` average-effect CR2 SE (previously only checked finite).
  - **StackedDiD** event-study coefficient point estimates; **PlaceboTests** leave-one-out
    `t_stat`/`p_value`; **DIDHAD** QUG order-statistic test (`t_stat`/`p_value`, previously
    unasserted against the golden). All at ~1e-10.
  - **estimatr 2SLS intercept SE** (HAD mass-point path): the `HC1`/`CR1` intercept SE now
    matches `estimatr::iv_robust` `se_intercept` to ~1e-15 (the 2×2 sandwich already computed
    `V[0,0]`; an opt-in `return_intercept_se=` hook on the private `_fit_mass_point_2sls`
    surfaces it — the default 3-tuple return is byte-unchanged, no public-API change). The
    `classical` intercept is excluded (same documented `O(1/n)` projection/DOF deviation as the slope).
  - **CR2 Satterthwaite DOF** via CI-inversion (no new result fields): `TwoWayFixedEffects(hc2_bm)`
    `dof_hc2_bm` and `MultiPeriodDiD(hc2_bm)` per-period `dof_per_coef` are pinned by reconstructing
    `conf_int` from the golden DOF — with ATT+SE already locked, the CI matches iff the DOF matches.
  - **PlaceboTests** leave-one-out `df` (recovered from the public `leave_one_out_effects` count);
    **DIDHAD** Yatchew linearity test `p_value` and both `sigma2` components (through the documented
    `N/(N-1)` sample-vs-population convention shift). All at ~1e-10.
  - **`fixest` cluster-robust SE band**: the DiD/TWFE cluster-at-unit SE is pinned within the
    documented ~0.25% fixest-CR1 small-sample DOF-convention band (guards an unintended SE-formula
    change; the machine-precision hetero/cluster lock is deferred — needs an unbalanced-DGP golden).
- **Doc-snippet env leak fixed; dCDH pinned-baseline backend detection made order-robust.** The
  doc-snippet runner (`tests/test_doc_snippets.py`) executed documentation code blocks without
  sandboxing `os.environ`, so the troubleshooting page's backend-override snippet leaked
  `DIFF_DIFF_BACKEND='python'` into every later test in a full-suite run. The only victim was
  `test_survey_dcdh.py::test_bootstrap_se_matches_pre_pr4_baseline`, whose call-time env read then
  selected the pure-Python baseline arm while the fit still dispatched to the already-imported Rust
  backend (fails under full-suite order, passes standalone). The runner now snapshot/restores
  `os.environ` around each snippet (root cause), and the dCDH test derives its baseline arm from
  the same dispatch globals the fit consumes (`bootstrap_utils` / `linalg`), with a coherence
  assert between the two (defense in depth).

### Added
- **REGISTRY.md and REPORTING.md are now published on Read the Docs.** The two
  methodology markdown pages render as in-site Sphinx pages (MyST) under a new
  "Methodology" toctree section, so cross-references from the API docs use `:doc:` links
  instead of off-site `blob/main` GitHub URLs — stable-docs readers previously landed on
  a different revision than their installed version. `myst-parser` joins the docs
  dependency set (pyproject `docs` extra, RTD post_install, docs-tests workflow — kept in
  sync); every other repo-internal markdown under `docs/` (performance notes, paper
  reviews, roadmaps) is explicitly excluded from the build. Two latent doc defects
  surfaced by the `-W` build are fixed: a broken ToC anchor in REGISTRY.md (typo'd
  `#differenceinifferences`, broken on GitHub too) and a heading-level jump; the
  "Diagnostics & Sensitivity" heading is retitled to "Diagnostics and Sensitivity"
  because GitHub and docutils slug `&` differently (double vs collapsed hyphen). Full
  local `make -C docs html SPHINXOPTS="-W"` passes with 0 warnings.
- **`ImputationDiD` leave-one-out conservative variance** (`leave_one_out`, default `False`) — the
  Borusyak-Jaravel-Spiess (2024) Supplementary Appendix A.9 finite-sample refinement. The non-LOO
  auxiliary aggregate `tau_tilde_g` (eq. 8) is built from the fitted `tau_hat_it` and so partially
  overfits to the noise `epsilon_it`, biasing the conservative variance downward; `leave_one_out=True`
  recomputes each unit's group aggregate excluding that unit (implemented efficiently by rescaling each
  treated auxiliary residual by `1/(1 - v_ig**2/sum_j v_jg**2)`, exactly equivalent to the direct
  leave-one-out at the per-unit cluster sum), yielding a larger, less-downward-biased SE (Prop. A8:
  unbiased for an upper bound at the default unit clustering). Point estimates are unchanged. A group
  with a single positive-weight unit (LOO undefined, App. A.9 fn. 51) keeps the non-LOO residual with a
  `UserWarning`. `leave_one_out` is recorded on `ImputationDiDResults` (and `to_dict()` / `summary()`);
  replicate-weight survey designs raise `NotImplementedError` (their variance bypasses the
  influence-function path). Default `False` preserves R `didimputation` parity.
- **`CallawaySantAnna(allow_unbalanced_panel=True)`** — parity with R
  `did::att_gt(allow_unbalanced_panel=TRUE)` on unbalanced panels. When set and the input panel is
  unbalanced (some units unobserved in some periods), the pooled observations are routed through the
  repeated-cross-section levels estimator (`DRDID::reg_did_rc`), replacing the default within-cell
  panel differencing (a different estimand on unbalanced data), and the per-observation influence
  function is clustered by the original unit. **ATT matches R bit-for-bit** — per-cell AND dynamic
  aggregation (fixed unit-cohort-mass `pg` reweighting + a per-unit WIF correction); the analytical
  SE matches up to the documented CR1 `sqrt(G/(G-1))` finite-sample factor. **Inert on balanced
  panels** (byte-identical to the default). Independently, the default path now emits a `UserWarning`
  on unbalanced input (previously silent) pointing to the flag. `survey_design=` with the flag raises
  `NotImplementedError` (deferred). Verified against R `did` 2.5.1
  (`benchmarks/data/cs_unbalanced_golden.json`).
- **`ContinuousDiD` lowest-dose-as-control** (`control_group="lowest_dose"`, CGBS 2024 Remark 3.1) for
  settings with no untreated group (`P(D=0) = 0`): the lowest-dose group `d_L` becomes the comparison
  and the estimand is `ATT(d) − ATT(d_L)` (with `ATT(d_L) = 0` the omitted reference). It is a
  control-group swap — the entire linear influence-function / bootstrap / event-study / survey
  machinery is reused unchanged (`ee_control` already carries the reference-group variance) — so both
  the discrete (`treatment_type="discrete"`) and continuous (B-spline) paths are supported; the
  discrete ACRT backward-difference reference shifts from `0` to `d_L`
  (`ACRT(d_1) = ATT(d_1)/(d_1 − d_L)`). The continuous path requires a genuine mass point at the
  minimum dose (`P(D=d_L) > 0`). Results gain a `reference_dose` field (`= d_L`). Fail-closed:
  never-treated units present, a singleton `d_L`, no treated dose above `d_L`, `dvals ≤ d_L`, or a
  survey/subpopulation design leaving `< 2` positive-weight `d_L` units all raise; multi-cohort and
  `covariates=` × `lowest_dose` raise `NotImplementedError` (deferred). The default `never_treated` /
  `not_yet_treated` paths are unchanged.
- **`EfficientDiD` `omega_ridge` parameter** (default `1e-6`) — ridge-regularizes the Omega*
  inversion behind the efficient weights: solves `(Omega* + omega_ridge * max(trace/H, 0) * I) x = 1`
  instead of pseudo-inverting the numerically singular Omega* that PT-All's telescoping
  overidentified moments produce (measured cond 1e17–1e22 for 100% of units on realistic panels).
  Makes per-cell `ATT(g,t)` numerically well-defined and platform-stable: a 1-ulp input
  perturbation now moves per-cell values by ≤3e-9 relative (previously ~1e-4 through the
  pseudoinverse's rcond-cutoff cliff, meaning per-cell values silently depended on BLAS/platform).
  Calibrated against the HRS Table 6 replication anchors (all unchanged; worst deviation
  0.0257 SE, shift ≤ 0.0001 SE) and Monte Carlo (bias/RMSE/SE-calibration/coverage statistically
  identical to legacy). **One-time value shift:** covariate-path per-cell and event-study values
  move within their pre-existing indeterminacy band on upgrade (worst observed post-treatment
  cell shift ~0.6 of its own SE at n=500, shrinking with n; overall-ATT shift 1.6e-2 → 1.1e-3
  relative from n=500 → 2k); the no-covariates path is essentially unchanged (~1e-7).
  `omega_ridge=0` restores the entire legacy code path bit-for-bit. The ridge path also drops the
  degenerate pre-treatment self-pair `(g'=g, t_pre=t)` (an identically-zero moment) so
  pre-treatment placebos stay data-driven, and consolidates the legacy per-cell
  condition-number warnings into one fit-level warning. See the Omega* ridge Note in
  `docs/methodology/REGISTRY.md`.
- **`ContinuousDiD` discrete-treatment saturated regression** (`treatment_type="discrete"`) for
  multi-valued / discrete dose (CGBS 2024 Eq. 4.1). Each distinct dose level gets its own effect
  coefficient — `ATT(d_j) = mean_{D=d_j}(ΔY) − control` (a per-level 2×2 DiD) — instead of a B-spline
  curve; `ACRT(d_j)` is the paper's backward finite difference on the grid `{0, d_1, …, d_J}`
  (`ACRT(d_1) = ATT(d_1)/d_1`, so a binary `D ∈ {0,1}` gives `ACRT = ATT`). The saturated fit is an
  exact basis swap, so analytical, multiplier-bootstrap, covariate (`reg`/`dr`), and survey inference
  all compose and reduce analytically to the per-level 2×2 DiD standard error.
  Multi-cohort fits with heterogeneous dose support across cohorts raise `NotImplementedError`
  (support-aware aggregation deferred); an off-support `dvals` value raises `ValueError`. The default
  `treatment_type="continuous"` (B-spline) path is unchanged.
- **`ContinuousDiD` covariate support** (`covariates=`, `estimation_method ∈ {"reg", "dr"}`) for
  dose-response estimation under **conditional** parallel trends
  (`E[ΔY(0) | D=d, X] = E[ΔY(0) | D=0, X]`). `reg` uses an outcome-regression control counterfactual;
  `dr` (default) is doubly-robust (DRDID `drdid_panel`). The scalar `overall_att` + standard error
  match `DRDID::reg_did_panel` / `drdid_panel` to ~1e-8; analytical, multiplier-bootstrap, and
  event-study inference all compose with covariates. `reg` and `dr` share the dose-response *shape*
  and `ACRT(d)`, differing only in the `overall_att` / ATT(d) level and the doubly-robust SE.
  `estimation_method="ipw"` with covariates raises `NotImplementedError` (pure IPW's covariate
  adjustment is a scalar level shift and cannot adjust the curve shape); `covariates=` +
  `survey_design=` is deferred (`NotImplementedError`).
- **`HeterogeneousAdoptionDiD` event-study `cluster=` support** (both designs). On
  `aggregate="event_study"`, `cluster=` now produces cluster-robust per-horizon pointwise
  confidence intervals AND a cluster-robust simultaneous sup-t confidence band (`cband=True`),
  on the continuous (CCT local-linear) and mass-point (2SLS) paths, unweighted or weighted —
  previously `cluster=` was ignored on the nonparametric event-study path (with a `UserWarning`)
  and the weighted mass-point `cband=True` case raised `NotImplementedError`. The clustered band
  draws cluster-level multipliers on the per-unit influence function; the variance family is
  reconciled to each path's analytical cluster-robust SE (exact for continuous; `√(G/(G-1))` CR1
  scaling for mass-point). `cluster=` + `survey=` is rejected — route clustering through
  `survey_design=SurveyDesign(psu=<cluster_col>)`. No behavior change for unclustered fits.

### Fixed
- **`CallawaySantAnna` base period is now selected positionally (sorted-index), matching R
  `did::att_gt` on gapped panels.** Previously the base period used literal calendar arithmetic
  (`t-1` pre-treatment, `g-1-anticipation` post/universal), so on non-consecutive period grids
  (e.g. biennial surveys, skipped years) cells whose calendar base was unobserved were NaN'd as
  `missing_period` even though R estimates them from the nearest observed period — which also
  corrupted the aggregated event-study / group SEs. The base is now the nearest observed period
  (largest observed `p < t` pre-treatment; largest observed `p` with `p + anticipation < g`
  post/universal), and the pre/post split is on `t < g` (independent of anticipation, matching a
  deparse of `did` 2.5.1 `compute.att_gt`) — resolving the internal inconsistency with the
  library's own dCDH estimator, which already used positional neighbors. On consecutive grids
  this is byte-identical to the old rule (all existing goldens unchanged); on gapped panels we now
  reproduce every R cell (e.g. `fewer_periods` {1,3,4,6} 7/15 previously-NaN cells, `reg` to
  ~1e-11 / `dr` to ~1e-4). A single shared `_select_base_period` helper is used across all
  estimation paths (panel fast / vectorized / covariate-reg, and repeated cross-sections). For
  `base_period="universal"`, each cohort's positional base is now materialized as a zero reference
  cell (`att=0`, `se=NaN`) in `group_time_effects` / `to_dataframe("group_time")` at its positional
  base event time (`e = base - g`, which can be `-2`, `-3`, … on gapped grids), matching R's
  `att_gt` table and `aggte(type="dynamic")` — including the overlapping-reference case where the
  zero base dilutes another cohort's estimated pre-trend at the same event time (analytical AND
  multiplier-bootstrap paths verified vs `did` 2.5.1 to ~1e-5). The `group` / `simple` aggregations
  use post-treatment cells only and exclude the reference.
- **Absorbed-FE (`absorb=`) non-clustered classical/hetero standard errors now use the full-K
  finite-sample scale, matching `fixest`.** The within-transform variance scale (`sse/(n-k)` for
  `classical`, `n/(n-k)` for `hc1`) counted only the *visible* regressors `k`, excluding the
  absorbed fixed effects, so `TwoWayFixedEffects(vcov_type="classical")`,
  `DifferenceInDifferences(absorb=..., vcov_type in {classical,hc1})`, and
  `MultiPeriodDiD(absorb=..., vcov_type in {classical,hc1})` non-clustered SEs sat ~6.5% below
  `fixest::feols(vcov="iid"/"hetero")` — even though the reported t-`df` already counted the
  absorbed FE (an internal inconsistency). A single scalar rescale
  (`(n-k)/(n-k-df_adjustment)`, fail-closed when the full-K residual dof is non-positive) now
  aligns the SE's `k` with `K_full`, so the absorb path equals the explicit full-dummy
  (`fixed_effects=`) path and fixest. **Clustered** SEs are unchanged (fixest's nested-FE `ssc`
  convention already matches with `k_visible`); `hc2`/`hc2_bm` (leverage / Satterthwaite DOF) and
  survey vcov are unaffected; `SunAbraham` `hc1` auto-clusters at unit and so keeps its documented
  deviation.
- **Non-finite degrees of freedom now fail closed to all-NaN inference.**
  `safe_inference()` previously rejected `df <= 0` but let a non-finite `df` (NaN) through,
  producing an inconsistent tuple (finite t-stat, NaN p-value/CI). It now returns all-NaN for
  any non-finite `df`. `MultiPeriodDiD(vcov_type="hc2_bm")` likewise no longer falls back to
  the shared residual `df` when a coefficient's Bell-McCaffrey Satterthwaite DOF is non-finite
  (guard-suppressed) — such a coefficient's inference is now NaN rather than silently computed
  from a different `df`, preserving the joint-NaN inference contract. Only affects coefficients
  whose BM DOF was declared unreliable (high-leverage / collinear); well-conditioned
  treatment / event-study / average contrasts are unchanged.
- **Unweighted clustered CR2 / Bell-McCaffrey per-coefficient Satterthwaite DOF no
  longer returns non-physical values** for high-leverage FE-dummy / collinear nuisance
  columns. The simple `(tr B)²/tr(B²)` form could produce a garbage DOF there (float64
  noise in `trace_B2` → up to ~1e61, or a finite-but-inflated value above the cluster
  count). The unweighted path now carries the same noise-floor guard as the weighted
  path plus a `DOF ≤ G` (cluster-count) physical bound, NaN-ing those columns with a
  warning. The treatment / event-study / compound-average contrasts that estimators
  actually consume are unchanged and match R clubSandwich; this only affects direct
  `LinearRegression(vcov_type="hc2_bm", cluster_ids=...)` callers reading full
  per-coefficient DOF vectors.
- **`ChaisemartinDHaultfoeuille` phase-1 placebo (`DID_M^pl`) point estimate sign
  corrected.** The single-horizon (`L_max=None`) `placebo_effect` used the opposite
  (forward) difference order from the multi-horizon placebo path and R
  `did_multiplegt_dyn`, so it was sign-flipped on pure-direction panels (magnitude was
  bit-identical). It now uses the backward-difference × switch-direction convention,
  matching R to working precision on `joiners_only` / `leavers_only`. Mixed-direction
  panels retain the separately-documented period-vs-cohort control-set deviation. SEs
  (NaN by design for the single-period placebo) and the multi-horizon placebo path are
  unaffected.
- **`HonestDiD(method="smoothness")` now returns a finite FLCI when the estimated
  identified set is empty**, instead of silently NaN-propagating all inference. When the
  observed pre-trend's curvature exceeds `M`, the `δ_pre = β_pre`-pinned identified-set LP
  is infeasible (`lb`/`ub` = NaN) — but the optimal FLCI does not depend on that LP (its
  worst-case bias is taken over `δ ∈ Δ^SD(M)` treating β as random), so it is well-defined
  given `(Σ, M)`. R's `HonestDiD::createSensitivityResults` returns the FLCI in exactly this
  case; previously `fit()` returned all-NaN, so smoothness sensitivity analysis yielded no
  inference on any event study with non-trivial pre-trend curvature. The FLCI now matches R
  to ~1e-3 at `M=0`; a known optimizer/center divergence at intermediate `M` (up to ~9% on
  wide pre/post windows, CI *width* unaffected) is documented in `REGISTRY.md` and deferred.
- **`CallawaySantAnna` panel reg/ipw standard errors now match `DRDID` /
  R `did` exactly** (point estimates unchanged — the omitted terms are mean-zero,
  which is why ATTs always matched R at ~1e-11 while SEs drifted). Two defects:
  (1) the covariate-reg influence function omitted `DRDID::reg_did_panel`'s OLS
  estimation-effect term (`asy.lin.rep.ols %*% M1`) — per-cell SEs were 4-13% and
  aggregated (simple/group/event-study) SEs 3-20% from R, and **anti-conservative**
  on some designs (0.958x on the two-period golden fixture) despite the prior
  documentation calling the plug-in "conservative"; (2) the unweighted covariate-ipw
  per-cell SE used a weighted *population* variance never scaled by an effective
  sample size — **~7x inflated** — and its influence function lacked
  `std_ipw_did_panel`'s propensity-score estimation-effect correction
  (`asy.lin.rep.ps %*% M2`; aggregated SEs ~2.4% off). The survey ipw branch already
  carried both corrections (Phase 7a); the fix mirrors it method-uniformly. All panel
  reg/ipw per-cell SEs (including no-covariate, where ipw's
  `var_c*(1-p)/(n_c*p)` plug-in equaled R only at treated-share 0.5) are now derived
  from the same influence function that feeds aggregation: `sqrt(sum(phi^2))`, the
  convention dr always used. Everything downstream of the per-cell IFs inherits the
  fix: t-stats/p-values/confidence intervals, aggregated SEs, `event_study_vcov`
  (HonestDiD input), the bare-`cluster=` CR1 per-cell override, and
  multiplier-bootstrap SEs. The reg estimation-effect projection is evaluated in the
  centered basis (`1/n_c + (x-x̄_c)'G^{-1}(x̄_t-x̄_c)`), which is offset-invariant;
  reg/ipw fits with collinear covariates now fire the same aggregate rank-guard
  warning as dr. Post-fix parity: per-cell and aggregated reg/ipw vs the R golden
  fixtures at ~5e-12, and vs fresh R `did` 2.5.1 ipw aggregations at ~1e-10
  (previously-unasserted golden SE blocks are now enabled, plus R-free numpy
  reconstructions of both DRDID influence functions).
- **ImputationDiD/TwoStageDiD covariate fits with zero-weight replicate designs (JK1/plain
  BRR) now produce finite SEs.** Replicate weights that zero out whole PSUs reach Step 1
  unmasked; the previous per-estimator pandas demeaning loops divided 0/0 on zero-total-weight
  groups, NaN-poisoning the demeaned design so EVERY replicate refit failed inside
  `solve_ols` ("All replicate refits failed. Returning NaN variance." after a
  non-convergence warning storm — measured: 198 warnings and SE=NaN on a 350k-row panel
  with 16 JK1 replicates, now 0 warnings and a finite SE). A main fit combining zero-weight
  rows with covariates previously raised an opaque `ValueError`; it now fits, with the
  zero-weight unit's FE surfacing as NaN (spillover convention — keys retained for the
  rank-condition check, never a silent finite 0.0) and its unidentified cells NaN across
  all inference fields. TwoStageDiD's per-replicate "non-finite imputed outcomes" warning
  is suppressed inside replicate-refit closures only (`warn_nan=False`); the main-fit
  warning is unchanged.
- **Dube, Girardi, Jordà & Taylor (2025) citation corrected to *J. Applied Econometrics*
  40(**7**):741-758** (was cited as issue 5) across `docs/references.rst`,
  `docs/methodology/REGISTRY.md`, `docs/methodology/papers/dube-2025-review.md`,
  `docs/api/lpdid.rst`, and `carousel/generate_lpdid_carousel.py`. Verified against the
  published record (DOI 10.1002/jae.70000; IDEAS/RePEc `wly/japmet/v40y2025i7p741-758`).
  Historical CHANGELOG entries (the 3.6.0 release notes) intentionally retain the
  original attribution as a record of what was claimed at release time; this entry
  supersedes it.

### Changed
- **Explicit `vcov_type` with replicate-weight survey designs now warns and proceeds
  (was: inconsistent raise/silent-ignore).** With replicate variance the analytical
  sandwich is replaced wholesale — per-replicate refits return point estimates only,
  identical across vcov families (FWL), so the requested vcov family cannot influence any
  reported number. `TwoWayFixedEffects(vcov_type="hc2"/"hc2_bm")` previously raised
  `NotImplementedError` while `DifferenceInDifferences` silently ignored the kwarg; both
  (and `MultiPeriodDiD`) now share one contract: explicit non-`hc1` analytical
  `vcov_type` (`hc2`/`hc2_bm`/`classical`) emits a `UserWarning` and the discarded base fit remaps to `hc1` (avoids wasted CR2-BM work,
  one-way validator rejections, and the TWFE full-dummy route, which does not compose
  with per-replicate re-demeaning). Explicit `hc1` — the old error message's own
  workaround guidance — stays silent; `conley` keeps its own survey-design support contract and validators
  (excluded from the remap). A per-replicate full-dummy HC2 implementation
  (the TODO row) was investigated and rejected as a costly no-op: it cannot change the
  replicate variance. Tests lock warn+bit-identity-to-hc1 on all three estimators.
- **Shared FE-dummy design build (`build_fe_dummy_blocks`).** The drop-first
  `pd.get_dummies` design construction existed as three inline copies — the
  `DifferenceInDifferences` and `MultiPeriodDiD` `fixed_effects=` loops and the
  `TwoWayFixedEffects` HC2/HC2-BM full-dummy path — whose FE naming / dtype /
  column-order conventions could drift independently (the drift risk flagged in TODO).
  All three now delegate to one `diff_diff.utils.build_fe_dummy_blocks` helper whose
  names match `fe_dummy_names` (the reserved-name collision guard) by construction.
  Outputs are bit-identical (A/B against the previous implementation on DiD with a
  non-default-order Categorical FE + covariates, TWFE hc2 + hc2_bm, and MultiPeriodDiD
  multi-FE including the `fe == time` skip); the MPD/DiD paths also drop the
  per-column `np.column_stack` accumulation (O(k²) copies) for one block stack.
- **Per-cell solver fast paths for covariate fits (CallawaySantAnna and every
  estimator routing through the shared solvers).** Two pure-Python changes in
  `diff_diff/linalg.py`: (1) `solve_logit`'s IRLS inner step — previously a full
  SVD (`np.linalg.lstsq`) on the tall weighted design per iteration — now solves
  equilibrated normal equations by Cholesky under an explicit
  reciprocal-condition guard (LAPACK `dpocon`), falling back to the exact legacy
  solve for any iteration whose normal matrix cannot be certified
  well-conditioned; (2) `_detect_rank_deficiency` short-circuits the common
  full-rank case with a Gram/eigenvalue certification two orders stricter than
  the pivoted-QR boundary, so all rank counts, drop decisions, and pivot
  selections on deficient or uncertifiable designs are unchanged. Measured
  (3-rep medians, CallawaySantAnna, 100k units x 20 periods = 2M rows, 95 cells,
  `aggregate="all"`): dr fits 1.5s → 1.0s at 10 covariates (1.49x), 2.6s → 1.6s
  at 20 (1.64x), 5.7s → 3.1s at 40 (1.85x); ipw 1.8s → 0.9s (2.0x);
  survey-weighted dr 2.7s → 1.7s (1.6x); pure-Python backend gains are equal or
  larger. The solver stages themselves: IRLS solve 6.8x, rank detection 16.5x
  (at 40 covariates). Estimates are unchanged beyond machine precision (overall
  ATT/SE deltas exactly 0; per-cell max ~7e-15; the R-golden ipw SE parity at
  1e-6 abs and the dCDH bit-identity baselines hold unmodified), the Cholesky
  fallback fires on 0% of healthy fits (its near-separation trip path is locked
  by tests), and memory is flat.
- **`CallawaySantAnna` no-covariate `estimation_method="dr"` per-cell SE is now
  influence-function-based** (`sqrt(sum(phi^2))`), matching the `reg`/`ipw` branches and R's
  `DRDID::drdid_panel`. It was the last per-cell SE on the ddof=1 plug-in
  `sqrt(var_t/n_t + var_c/n_c)`, which deviated by O(1/n) (and `dr` is the default method, so a
  plain no-covariate fit was affected). Without covariates DR reduces to difference in means, so
  the per-cell SE is now bit-identical to the `reg` path. Point estimates, aggregated SEs, and
  event-study SEs are unchanged — the same influence function already fed aggregation.
- **`EfficientDiD` analytical covariate path rewritten as a fused unit-tiled GEMM pass**
  (kernel-covariance tables with `(t_pre_j, t_pre_k)` dedup, per-group kernel matrices reused
  across all `(g,t)` cells, batched ridge solves replacing the per-unit SVD + pseudoinverse loop;
  the no-covariates Omega* gets a Gram-form twin). Measured (3-rep medians, 20 periods,
  5 cohorts, 5 covariates, `aggregate="all"`): fit 35.9s → 1.8s at 500 units (20x),
  95.4s → 3.5s at 1k (27x), 337.3s → 7.8s at 2k (43x), ~2.3h (extrapolated O(n²)) → 58s at 10k;
  a 100k-unit fit now completes in ~86 min at 4.0 GB peak RSS (previously memory-killed).
  Survey-weighted: 95.3s → 3.3s at 1k units. Memory is tile-bounded
  (`_TARGET_OMEGA_TILE_BYTES` = 256 MB). Point estimates under the default ridge differ from
  the legacy pseudoinverse path as described under `omega_ridge` above; at `omega_ridge=0`
  results are bit-identical to the previous release. The conditional-path scalability warning
  now fires once per fit at n > 50,000 (previously per cell at n > 5,000).
- **`EfficientDiD` conditional-path kernel-covariance tables hoisted across `(g,t)` cells**
  (follow-up to the fused tiled pass above). Every Omega* term is a kernel covariance keyed only
  by outcome columns, so the tables the tiled pass rebuilt per cell dedup to one table of
  distinct product columns per comparison group per unit-tile (~26x fewer kernel GEMM columns
  on a PT-All fit); kernel weight matrices are built and freed one group at a time, so the tile
  memory budget is governed by the largest single group instead of the sum (proportionally
  fatter tiles at large n); each cell's Omega* is gathered from the group tables through
  compact per-cell column slices in the legacy per-entry operation order (value-exact, locked
  by test). Measured (3-rep medians, 20 periods, 5 cohorts, 5 covariates, `aggregate="all"`):
  fit 7.7s → 7.0s at 2k units, 57.2s → 35.7s at 10k (1.6x), 358s → 129s at 30k (2.8x), and
  the 100k-unit fit drops from ~86 min to **17.8 min (~4.8x)** at the same ~4 GB peak RSS;
  the kernel-covariance stage itself is 40x faster (21.2s → 0.5s at 10k). Results move only at
  floating-point reassociation level (post-treatment cells ~1e-12 relative, overall ATT ~1e-13);
  the no-covariates path is byte-identical and `omega_ridge=0` still routes the entire legacy
  path. A `pt_assumption="post"` covariate fit builds no tables at all (all cells are
  just-identified), keeping that path regression-free.
- **`EfficientDiD` conditional-path ridge solves dispatch to a Rust batched-Cholesky kernel**
  (follow-up to the table hoisting above; Rust backend only). `_ridge_solve_weights`' batched
  `np.linalg.solve` — a serial LAPACK LU sweep over the `(units × H × H)` ridged Omega* stacks
  and the top stage at every scale post-hoisting — now routes to
  `batched_ridge_chol_solve_ones` in the Rust backend: a hand-rolled unblocked Cholesky (the
  ridged Omega* is SPD by construction) in a reused per-thread scratch buffer, parallelized
  over the unit axis with rayon. Non-SPD rows (measured zero on realistic panels) fall back to
  LU in-kernel, and any non-finite row is recomputed through the exact legacy numpy chain, so
  the pseudoinverse edge-case semantics are unchanged. Measured (3-rep medians, 20 periods,
  5 cohorts, 5 covariates, `aggregate="all"`): fit 7.1s → 4.3s at 2k units (1.65x),
  36.2s → 22.7s at 10k (1.6x), 129s → 90s at 30k (1.4x), the 100k-unit fit 17.8 → 15.9 min at
  slightly lower peak RSS; survey-weighted 3.0s → 1.7s at 1k (1.8x). The ridge-solve stage
  itself is 4.7x faster at 10k (17.1s → 3.7s; the kernel alone is ~9.6x — the remainder is
  shared Python-side prep, tracked in TODO.md). Results on the Rust backend move at
  floating-point reassociation level only (post-treatment cells ~2e-12 relative, overall ATT
  ~1e-13); the pure-Python backend is byte-identical, the no-covariates path is untouched, and
  `omega_ridge=0` never reaches the kernel.
- **CallawaySantAnna multiplier bootstrap rewritten as a fused, column-tiled scatter-GEMM**
  (EfficientDiD's bootstrap routes through the same kernel). The former loop sliced the
  `(block × n_units)` weight matrix twice per (g,t) cell per weight block — at a 40-period,
  10-cohort, 100k-unit panel (390 cells, 999 draws) that was ~95% of bootstrap wall time as
  pure memory traffic. All perturbation columns (per-cell IFs, overall combined IF,
  per-event-time combined IFs) are now scattered into a column-tiled influence matrix
  (byte-capped tiles; each tile replays the bit-identical weight stream via RNG state
  snapshot/restore) and consumed by one BLAS GEMM per weight block. Measured (5-rep medians,
  rust backend + Accelerate): bootstrap fits 24.5s → 4.3s (5.7x) at the 4M-row flagship,
  5.6s → 0.31s (18.3x) no-covariate and 6.5s → 1.2s (5.5x) dr at 2M rows, 4.2s → 1.5s (2.7x)
  survey-PSU; peak memory flat or lower. Point estimates are bit-identical (the bootstrap
  never feeds them); bootstrap SEs/CIs/p-values match the previous loop to BLAS
  reassociation (≤ ~1e-15 relative — far below bootstrap Monte-Carlo error), the same
  numerics posture as the existing draw-axis chunking. EfficientDiD's unweighted path folds
  its `1/n` prefactor into the influence column (`W @ (eif/n)` instead of `(W @ eif)/n`) —
  the same reassociation-level equivalence.
- **CallawaySantAnna aggregation SE assembly rewritten to O(n_units)** (also inherited by
  `StaggeredTripleDifference` aggregation). The combined influence-function assembly behind
  simple/event-study/group aggregated SEs — previously ~56-85% of analytical fit time at
  scale — replaces its per-aggregation-target full-DataFrame cohort scans, per-unit Python
  loops, and dense `(n_units × n_gt)` weight-influence-function matrices with per-fit cohort
  tables and a closed-form WIF (algebraically identical; details in REGISTRY). Measured
  (medians of 3, Apple M4 Max, Rust backend): analytical fits at 100k units × 20 periods
  (2M rows) 1.32→0.21s no-covariate (6.3x), 2.49→1.08s 5-covariate DR (2.3x); long panels
  40p×10 cohorts 9.09→3.91s and 60p×15 cohorts 9.86→4.34s (2.3x); bootstrap-999 fits
  1.4-1.6x (the remaining draw-loop matmul is unchanged); repeated cross-sections
  4.17→1.50s no-covariate (2.8x) with peak memory 6.4→2.1 GB (-67%; the dense WIF matrices
  were observation-scale in RCS mode). Point estimates are bit-identical; aggregated SEs
  move only at floating-point reassociation level (measured ≤5e-16 relative, drift-bound
  tested at 1e-9 against a frozen copy of the prior implementation). For scale context,
  a full `fit(aggregate="all")` at 2M rows now runs 3-11x faster than R `did` 2.5.1
  (equal work, analytical or bootstrap-999, single- or multi-core R; R's `pl`/`cores`
  parallelism is BLAS-bound on this path so both R arms time identically).
- **ImputationDiD/TwoStageDiD demeaning modernized onto the shared MAP engine.** The
  private per-estimator pandas `_iterative_demean` loops (rebuilt a
  `pd.Series.groupby().transform()` hash table every alternating-projection iteration)
  are deleted; the covariate and pre-trend-lead within-transformations now route through
  `demean_by_groups` (factorize-once + `np.bincount`, optional Rust kernel, one dispatch
  for all columns), and both `_iterative_fe` FE solvers route through a new shared
  bincount Gauss-Seidel helper (`diff_diff.utils._iterative_fe_solve`, modeled on
  SpilloverDiD's). `max_iter` modernized 100 → 10,000 (the R `fixest`/`pyfixest`
  convention already used by the shared engine). Estimates are preserved: measured ATT
  deltas ≤ 2e-15 and SE deltas ≤ 2e-12 relative across a 5-scenario before/after grid
  (2.35M-row panels, R-parity suites unchanged at 1e-6/1e-7). Measured speedups (median
  of 3): ImputationDiD covariate fit 4.18s → 1.72s (2.4x) and no-covariate 1.40x at
  2.35M rows; replicate-weight survey variance 20.3s → 3.6s (5.7x, 32 replicates,
  350k rows) and 30.3s → 1.8s on zeroed-PSU designs (17x — the old loop burned
  `max_iter` futile iterations per replicate). TwoStageDiD fit time unchanged
  (GMM-variance-dominated). Accumulation-order numerics documented in
  `docs/methodology/REGISTRY.md` (both estimator sections + "Absorbed Fixed Effects").

### Performance
- **Rust-backend HC2 vcov.** The Rust vcov path supported only HC1/CR1; one-way
  (unclustered, unweighted) HC2 now dispatches to a new `compute_robust_vcov_hc2` kernel
  mirroring the NumPy branch exactly — hat diagonals off the same bread,
  `u²/max(1−h, 1e-10)` leverage meat, no n/(n−k) factor — matching NumPy at ~1e-15 on a
  seed grid. The near-singular hat-diagonal guard stays Python-side: the kernel returns a
  sentinel error and the documented warn-and-fall-back-to-HC1 fires in the dispatcher,
  identical to the NumPy branch (locked by a monkeypatched-sentinel test plus an exact
  h=1 clamp-parity test). The symbol is imported independently (mixed-version safe — a
  stale extension degrades HC2 to NumPy without disabling older Rust accelerations).
  `return_dof` / weighted / CR2-BM requests stay on NumPy (CR2-BM tracked in TODO).
- **Wild cluster bootstrap test inversion ~7x faster with bounded memory.** The WCR
  bootstrap DGP is linear in the candidate null `r` (`y*(r) = A + r·B` with r-independent
  `A`, `B`), so the per-cluster score decomposition and the studentizing variance reduce
  to five precomputed `(B,)` vectors (the variance is the PSD quadratic
  `qa + r·qb + r²·qc`). The CI inversion's ~O(100) `_t_star(r)` evaluations — each of
  which previously materialized fresh `(B×n)` bootstrap-outcome, `(k×B)` refit and `(n×B)`
  residual arrays — now cost O(B) arithmetic each, and the ONE precompute pass is chunked
  over draws (a conservative per-chunk byte budget sized against the peak count of live
  `(Bc, n)` temporaries) so peak memory is bounded for large `n`/`B`. Verified against
  origin/main on a 5-seed few-cluster grid: SE, p-value, and inverted CI endpoints all
  **bit-identical** (backend pinned), 6.8x end-to-end; the boottest parity suite
  (`tests/test_wild_bootstrap.py`, incl. pinned values) passes unchanged, plus a new
  chunk-count-invariance regression class (`TestPrecomputeChunking`). The
  quadratic-form evaluation is a ~1-ULP reassociation of the per-call `sum(scores²)`;
  the strict-inequality tie guard absorbs sub-1e-9 shifts by design.
- **`SpilloverDiD` staggered nearest-treated distances gain the sparse cKDTree branch.**
  The staggered cohort loop always built a dense `(n_units, n_treated_by_onset)` distance
  matrix per cohort; it now dispatches per cohort to the same cKDTree helper the static
  path uses (auto-activated when `n_units` exceeds the sparse threshold, built-in metrics
  only, `cutoff_km` set to the outermost ring edge). Within-cutoff distances are exact (the
  helper recomputes the true great-circle/planar metric for in-range matches) and
  beyond-cutoff units get `inf` — semantics-preserving because every staggered `d_it`
  consumer (ring membership, `S_it`, the far-away check, the event-study `d_bar` trigger)
  compares against thresholds at or below that cutoff. Helper- and fit-level equality
  tests pin the sparse arm against the dense path (atol 1e-12 end-to-end).
- **`CallawaySantAnna` per-(g,t) IF scatters converted from `np.add.at` to fancy `+=`**
  (`staggered.py::_cluster_robust_se_from_per_gt_if` — runs once per (g,t) cell when
  `cluster=` is set — and the general combined-IF assembly path in
  `staggered_aggregation.py`). The index arrays are duplicate-free by construction at
  every producer (`np.where` on disjoint masks — the invariant the aggregation fast
  path already relies on), so the fancy scatter is bit-identical to the unbuffered
  `np.add.at` while avoiding its 5-20x per-element overhead. Also removed the dead
  `_compute_aggregated_se` (zero callers; superseded by
  `_compute_aggregated_se_with_wif` since the WIF adjustment landed).
- **`EfficientDiD` `_ridge_solve_weights` Python-prep shave.** When no row of the Omega*
  stack is zero-masked (the common case), the batched ridge solve now consumes the stack
  directly instead of paying the `omega_stack[rest]` fancy-index copy, and returns the
  solved weights without the tail scatter through a preallocated uniform matrix — cutting
  ~O(n·H²) memory traffic per tile call on both backends (the Rust kernel takes a
  read-only borrow; the numpy chain builds its own ridged copy). Outputs are
  **byte-identical** (function-level identity checked on common and zero-masked stacks;
  cond-10k end-to-end ATT bit-identical, fit time 20.09s → 19.99s median — a
  memory-traffic cleanup, not a headline speedup). The `zero_mask` abs scan is retained
  (correctness); the remaining conditional-path lever is the sieve/nuisance stage
  (see TODO).
- **`CallawaySantAnna` / `StaggeredTripleDifference` per-cell unit-label arrays no longer
  materialized.** Every (g,t) cell's internal influence-function record carried
  `treated_units` / `control_units` label arrays (`all_units[positions]`, two O(n_control)
  allocations per cell) that no in-package code path ever read: the only consumers were the
  `precomputed`-less fallbacks of the combined-IF assembly and the multiplier bootstrap,
  which every in-package caller bypasses by threading the precomputed structures. Labels
  remain recoverable as `all_units[treated_idx]`. The two fallbacks now raise an actionable
  `ValueError` if a direct caller reaches them without label arrays (previously they would
  have been reached only by external callers hand-building the internal dict). No public
  API or numerical change — the dict is internal (`_influence_func_info` whitebox surface
  keeps `treated_idx`/`control_idx`/`treated_inf`/`control_inf`).
### Changed
- **`SpilloverDiD` stage-1 FE solver routed through the shared Gauss-Seidel engine.**
  `spillover._iterative_fe_subset` is now a thin Butts-subsample wrapper over
  `diff_diff.utils._iterative_fe_solve` (the engine ImputationDiD / TwoStageDiD already
  use), taking the FE-solver copy count in the library from 2 to 1. The wrapper keeps the
  SpilloverDiD front door (empty-Omega_0 / empty positive-weight-Omega_0 `ValueError`
  gates); the shared engine owns the iteration, the zero-weight NaN-FE convention, and
  the `warn_if_not_converged` non-convergence warning (now labelled "SpilloverDiD stage-1
  FE (Butts Omega_0 subsample)", replacing the caller-side message). Per sweep the shared
  engine computes the identical group means and convergence metric, so converged fits are
  bit-identical; `max_iter` is aligned from the historical local cap of 100 to the shared
  10,000 convention (fits that previously exhausted 100 iterations and warned may now
  converge instead — strictly more accurate FE; `tol=1e-10` unchanged). REGISTRY
  SpilloverDiD section documents the routing.
- **`stute_test` / `stute_joint_pretest` unweighted bootstrap ~2x faster, bit-identical.**
  The Appendix-D per-replicate OLS refit loops now hoist their loop invariants: Mammen
  multiplier draws are batched through memory-bounded `rng.choice` calls (numpy fills
  C-order, so the variate stream — and therefore every draw — is identical to the prior
  per-iteration draws), and the d-moments + tie-safe CvM tie-block indices are precomputed
  once per test instead of once per replicate. Each replicate still applies the same 1-D
  refit/CvM operations on the same values, so `bootstrap_S`, `cvm_stat`, and `p_value` are
  **bit-identical** to the literal per-iteration form (locked by a frozen byte-copy parity
  test + per-helper equality tests). Measured 2.5x/2.1x/1.9x at G=1e3/1e4/1e5 (B=999).
  This resolves the Phase-3 TODO row's ~2x target without its sketched O(G²)
  `M = I - X(X'X)^{-1}X'` materialization (which would not scale in memory) and with
  exact — not just "functional" — identity. The paper-faithful per-replicate refit form
  is unchanged; the large-G advisory now cites measured post-hoist timings.

## [3.6.2] - 2026-07-03

### Added
- **balance interop launch: composition-drift tutorial + `interop-notebooks` CI job.** Meta's
  `balance` package (>= 0.21) ships a one-way adapter `balance.interop.diff_diff`
  (facebookresearch/balance PR #465) whose `balance[did]` extra pins `diff-diff>=3.3,<4`.
  `docs/tutorials/26_composition_drift_calibration.ipynb` is the diff-diff-side companion to
  balance's `balance_diff_diff_brfss` tutorial, telling the failure-mode half of the story: a
  BRFSS-style smoking-ban DGP with no systematic arm-specific trends (parallel trends hold in
  expectation; planted ATT -3.0pp, realized -2.98pp under a rarely-binding probability floor) where
  treatment-correlated non-response drift biases the design-weight Callaway-Sant'Anna ATT to
  ~-4.1pp with *clean pre-trends*; a per-wave national rake fails (~-4.4pp - margins satisfied in
  aggregate while arm-level composition is untouched); per-state raking with balance (BRFSS's own
  granularity, population-count totals) recovers ~-3.2pp. Also covers the seam both ways (native
  `SurveyDesign` + `aggregate_survey` vs `bd.to_panel_for_did`/`bd.fit_did`, exact-parity assert),
  a 3-estimator x 2-weighting sweep, and `as_balance_diagnostic` cross-package diagnostics.
  `tests/test_t26_composition_drift_calibration_drift.py` re-derives every quoted number
  (auto-skips without balance). balance stays out of package requirements: the tutorial runs in a
  new isolated `interop-notebooks` job in `notebooks.yml` (python 3.12, installs
  `balance>=0.21`, also the drift guard's CI home; the workflow's weekly cron doubles as a
  cross-package integration smoke against latest PyPI balance), and the main notebooks job env is
  unchanged.
- **`balance.interop.diff_diff` contract tests.** `tests/test_balance_interop_contract.py` pins
  the diff-diff surface Meta's balance adapter consumes, importing no balance code:
  `aggregate_survey` forwarded-params superset + `(panel, SurveyDesign)` return schema, the
  `SurveyDesign` 15-field dataclass contract (plus TSL / replicate constructions), estimator and
  short-alias resolution (`CS`/`DiD`/`BJS`/`HAD`) with `survey_design=` accepted by all 17
  promised `fit()` signatures, the `_balance_adjustment` setattr provenance side-channel
  (guards against future `__slots__`), the CallawaySantAnna pweight-only guard, and the
  `SurveyMetadata.design_effect`/`effective_n`/`sum_weights` attribute names read by
  `as_balance_diagnostic`. Docs handoff closing the survey-roadmap Phase 8g gap: "Weight
  calibration with balance" section in `docs/api/prep.rst`, calibration pointers in
  `llms.txt`/`llms-full.txt`/`llms-practitioner.txt` and `README.md` Survey Support, and
  Deville & Särndal (1992) + Sarig, Galili & Eilat (2023) in `docs/references.rst`.
- **`SyntheticControl` ADH-2015 §4 tail diagnostics** (two opt-in `SyntheticControlResults`
  methods, closing the last two ADH-2015 §4 checklist items). `regression_weights()` reports the
  implied donor weights `W^reg = X0a'(X0a X0a')^{-1} X1a` of the regression counterfactual
  (intercept-augmented so they sum to 1 at full row rank) and flags donors outside `[0, 1]` — the
  extrapolation an OLS counterfactual incurs but the simplex-constrained synthetic control cannot;
  pure linear algebra, min-norm least-squares with a rank-deficient warning. `sparse_synthetic_control()`
  exhaustively searches `C(J, l)` size-`l` donor subsets holding `V` fixed at the baseline fit,
  reporting how fit / ATT degrade as the synthetic is forced sparse (per-size winner table +
  `get_sparse_synthetic_control_gaps()` overlay); the default `sizes=[1,2,3]` sweep skips over-cap
  sizes with a warning while an explicitly requested oversized `l` raises (`max_subsets` guard). Both
  are purely additive, surface under `estimator_native_diagnostics`, and leave the analytical
  inference contract unchanged (`se`/`t_stat`/`p_value`/`conf_int` stay NaN). No behavior change to
  any existing fit.
- **Rust `demean_map` kernel for FE absorption** (optional backend acceleration). When the Rust
  extension is available, the method-of-alternating-projections sweeps in
  `demean_by_groups`/`within_transform` run in a compiled kernel, rayon-parallel across the
  demeaned variables, with the GIL released during compute. The kernel mirrors the canonical
  numpy engine exactly (same sweep order, row-order bincount accumulation, zero-total-weight
  inert guard, and `max|x - x_old| < tol` stopping rule with NaN-poisoning semantics); numpy
  remains the reference implementation per the python-canonical policy, with equivalence tests
  asserting iteration-count equality plus `assert_allclose` at atol=1e-12. No behavior change
  when the extension is absent or `DIFF_DIFF_BACKEND=python`; kernel-side validation errors fall
  back to the numpy engine. Affects every MAP consumer (TWFE, SunAbraham, Bacon, Wooldridge,
  DiD/MPD `absorb=`, and the survey replicate refits).
- **Opt-in chunked dispatch for the Rust `demean_map` kernel** (internal
  `DIFF_DIFF_DEMEAN_CHUNK_COLS` env knob; default behavior unchanged - single full-width
  dispatch). When set to a positive integer, the wrapper feeds the kernel balanced
  near-equal column blocks, capping the kernel's transient input/working copies on wide
  absorbed designs for memory-constrained runs (a width near the machine's core count
  measured best). Results are identical by construction - each column's MAP loop is fully
  independent, so chunking changes neither demeaned values nor iteration counts (locked by
  exact-equality tests vs single-call dispatch). Shipped opt-in rather than default-on:
  measured end-to-end on the FE-absorption lane, the fit-level peak RSS on the widest
  workload is dominated by the downstream solver phase, so default chunking would cost
  ~2-7% wall-clock for only ~5-12% peak-RSS reduction (details + corrected memory
  attribution in `docs/performance-plan.md`).

### Changed
- **`solve_ols` marshalling slimmed on both backends** (memory + speed; outputs preserved).
  Rust: column norms computed from the borrowed numpy view, equilibration fused into the
  single owned faer copy, `U'y` computed directly off the faer SVD factor (the old path
  materialized a full ndarray copy of U for one k-vector dot), and factors dropped before
  the fitted/residual/vcov stage - rust-side allocator high-water on a 2.4M x 130 clustered
  solve falls from 15.32 GB to 7.81 GB (measured with the new feature-gated `alloc-profile`
  counting allocator; never compiled into wheels). Python: rank detection uses
  `qr(mode="r")` (same dgeqp3 R and pivot, skips forming the discarded Q) and
  `_equilibrated_lstsq` materializes its scaled temporary F-order with `overwrite_a=True`
  so gelsd consumes it in place - python-backend estimates are bit-identical (benchmark
  identity deltas exactly 0.0). The removed copies also cost time: frozen-code benchmark
  arms measured CV-clear improvements incl. firm-panel rust fits -21% (25.8 -> 20.3 s),
  county event studies -12%/-13% (rust/python), with no scenario regressing. Rust outputs
  move only at the cross-backend parity tolerances (REGISTRY-documented; never
  bit-identical across backends).
- **FE-absorption demeaning rewritten: factorize-once + `np.bincount` method of alternating
  projections** (`demean_by_groups` / `within_transform`). Each absorbed dimension is factorized
  once and group means are formed via `np.bincount` instead of re-hashing the group keys with
  `pandas.groupby().transform("mean")` on every iteration x variable x dimension. Affects every
  fit of `TwoWayFixedEffects`, `SunAbraham`, `BaconDecomposition`, `WooldridgeDiD`,
  `DifferenceInDifferences`/`MultiPeriodDiD` with `absorb=`, and the survey replicate-refit path.
  Convergence semantics (per-variable sweep order, `max|x - x_old| < tol` criterion, zero-total-
  weight group guard, non-convergence warning) are unchanged. **Numerical contract**: bincount
  accumulation is not Kahan-compensated the way pandas' grouped mean is, so demeaned values agree
  with the prior implementation to ~1e-10 order rather than bit-for-bit; estimator estimates are
  validated unchanged at the benchmark identity gate
  (`bench_fe_absorption.py --check-estimates`, coefficients atol=1e-9).
- **MAP iteration cap raised from 100 to 10,000** (`max_iter` in both `demean_by_groups` and
  `within_transform`), matching the R `fixest` (`fixef.iter`) and `pyfixest` (`fixef_maxiter`)
  defaults. Correlated FE incidence (e.g. stores active in contiguous week windows) genuinely
  needs hundreds of iterations; the old cap made such fits warn and return slightly-off residuals.
  Worst-case trade-off accepted deliberately: a truly non-convergent input now iterates 10,000x
  before warning. Documented in `docs/methodology/REGISTRY.md` "Absorbed Fixed Effects".
- **NaN in an absorbed group column now raises `ValueError`** naming the column (previously the
  unweighted path silently NaN-poisoned the affected rows and the weighted path silently passed
  them through un-demeaned).
- **FE-spanned regressors are now snapped to exact zero before the solver** (new
  `diff_diff.utils.snap_absorbed_regressors`, wired at every demeaning consumer: DiD/MPD
  `absorb=`, TWFE, SunAbraham, BaconDecomposition, WooldridgeDiD, and the DiD/MPD/TWFE
  replicate-refit closures). A regressor spanned by the absorbed FEs (treated-group indicator with
  the unit dimension absorbed, a period dummy with time absorbed, a unit-constant covariate, or an
  additive `a_unit + b_time` combination) demeans to numerical junk that previously survived the
  rank check via column equilibration and perturbed the identified coefficients - up to ~3e-3 on
  ATT with a ~1e14-scale garbage coefficient reported for the spanned column in the joint-span
  case. Detection is two-stage (fast relative-norm path at 1e-10, then an exact sparse-LSMR
  span-membership confirmation for candidates masked by MAP truncation on unbalanced/correlated
  panels); norms are `sqrt(w)`-weighted under WLS so zero-weight domain rows cannot mask spanning.
  Spanned regressors are dropped deterministically (coefficient NaN) with a cause-specific
  `UserWarning` naming them under `rank_deficient_action="warn"`.
  **Point-estimate note**: designs carrying such spanned regressors (e.g. 2x2 DiD with
  `absorb=[unit_dim, time_dim]`) see identified estimates shift by up to ~1e-5 relative to
  v3.6.1 - the removal of the junk direction, documented in REGISTRY "Absorbed Fixed Effects";
  estimates are now invariant (~1e-14) to the demeaning tolerance where they previously swung
  at ~1e-5.

### Added
- **FE-absorption benchmark suite** (`benchmarks/speed_review/bench_fe_absorption.py`, scenarios
  7-13 in `docs/performance-scenarios.md`): seven realistic workloads (county policy event study,
  firm panel with churn, scanner store-week, 5M-order geo experiment, survey BRR replicates,
  correlated-FE stress, small-panel guard) timing the MAP-demeaning hot path
  (`demean_by_groups` / `within_transform`) with subprocess isolation, multi-run CV reporting, and
  ATT/SE identity capture so optimization PRs can prove estimates are unchanged
  (`--check-estimates`). Includes an optional pyfixest yardstick lane
  (`bench_fe_absorption_pyfixest.py`, guarded on import - never a dependency) asserting < 1e-6
  coefficient parity on the exact-estimand scenarios, and committed BEFORE baselines
  (`baselines/fe_absorption_before.json`). Measurement-only: no library behavior change.
- **`HeterogeneousAdoptionDiD` cluster-robust SE on the continuous paths** (Phase 2a). `cluster=`
  is now threaded into `bias_corrected_local_linear` on the `continuous_at_zero` /
  `continuous_near_d_lower` designs, so the CCT-2014 robust variance becomes cluster-robust and the
  β̂-scale SE is `se_robust / |den|` (previously `cluster=` was ignored on the continuous path with a
  `UserWarning`). Composes with the `weights=` shortcut (weighted cluster-robust). The `cluster=` +
  `survey_design=` composition raises `NotImplementedError` (route clustering through
  `survey_design=SurveyDesign(psu=<cluster_col>)`). Cluster IDs must be unit-constant — a nonexistent
  column, NaN, or within-unit-varying cluster now raises (mirroring the mass-point path) instead of
  being silently ignored. Cluster-robust inference with fewer than two clusters in the active kernel
  window (the in-bandwidth subset the CCT variance is computed on) returns `se=nan` (and NaN t-stat /
  p-value / CI, `att` finite), matching the mass-point CR1 single-cluster contract; the guard lives in
  `_nprobust_port.lprobust` so it also covers the direct `bias_corrected_local_linear` API. Result metadata reports `vcov_type="cr1"` +
  `cluster_name`. The mass-point path and the event-study (Phase 2b) path are unchanged (Phase 2b
  still defers cluster with a warning).

### Changed
- **Internal: consolidated the panel-to-unit survey-design collapse** shared by `ContinuousDiD`
  and `EfficientDiD` into two `diff_diff/survey.py` helpers —
  `ResolvedSurveyDesign.subset_to_units_by_row_idx` (folds the index-and-recount preamble around
  the existing `subset_to_units`) and `build_unit_first_row_index` (first-panel-row index per
  unit). Removes four hand-rolled copies of the collapse plus a slow `df.iterrows()` index build.
  No behavior change: survey-weighted SEs, `df_survey`, and design-effect metadata are
  bit-identical (locked by an oracle test against the old inline logic).
- **`DiagnosticReport` now routes `SpilloverDiDResults`.** Previously a fitted
  `SpilloverDiD` (Butts 2021) result matched no `_APPLICABILITY` / `_PT_METHOD` entry, so
  `DiagnosticReport(spillover_result)` reported every check as `not_applicable`. It now routes
  to parallel-trends (event-study joint test on the per-event-time direct-effect dynamics,
  populated when `event_study=True`), design-effect (survey-weighted fits), and heterogeneity;
  Goodman-Bacon is intentionally excluded because SpilloverDiD identifies off far-away control
  units rather than TWFE 2×2 comparisons. Aggregate-only fits now skip parallel-trends with an
  estimator-accurate remediation (`SpilloverDiD(..., event_study=True)`). `BusinessReport` and the
  `describe_target_parameter` block pick up the routing automatically.
- **`SyntheticControl` in-space / leave-one-out placebo diagnostics now distinguish structural `cv`
  infeasibility from solver non-convergence.** Under `v_method="cv"`, an excluded `in_space_placebo()`
  / `leave_one_out()` refit whose pseudo-treated (in-space) or reduced (leave-one-out) donor pool is
  indistinguishable in a re-aggregated CV window — a structural identification failure, not an
  under-optimized solve — is now tallied in a new `n_infeasible` field (in-space) / `_loo_n_infeasible`
  (leave-one-out) with a `status="infeasible"` row, mirroring the split `in_time_placebo` already
  reported; `n_failed` now counts only genuine solver non-convergences. `_placebo_status` /
  `_loo_status` gain `all_placebos_infeasible` / `all_placebos_unusable` (resp.
  `all_refits_infeasible` / `all_refits_unusable`) codes, and `DiagnosticReport` surfaces a
  machine-readable `reason_code` alongside `n_failed` / `n_infeasible`. The permutation
  `placebo_p_value` / `n_placebos` are UNCHANGED — both causes are excluded from the rank / ATT range
  identically, so only the diagnostic attribution is refined. `n_infeasible` is 0 for the non-`cv`
  `v_method`s (no structural-identification gate). Internal: `_placebo_fit_unit` now returns a
  `(result, status)` tuple and `_outer_solve_V_cv` a structural-infeasible flag.

## [3.6.1] - 2026-07-01

### Added
- **`LPDiD` complex-survey-design support** (Phase D1). Adds a `survey_design=` argument to
  `LPDiD.fit()` (a `SurveyDesign` with probability weights + optional strata/PSU/FPC). On the
  variance-weighted default path the long-difference regression at each horizon is fit by WLS on
  the survey weights, and the standard error is the stratified-PSU Taylor-linearization (Binder
  TSL) sandwich with `df = n_PSU - n_strata`, reusing `diff_diff/survey.py`
  (`compute_survey_vcov` / `_compute_stratified_psu_meat`). The design is re-resolved on each
  realized (post-clean-control) sample so weights/strata/PSU align with the regression rows; with
  no explicit PSU the unit (LP-DiD's default cluster) is injected as the PSU. Rejects
  `survey_design` combined with `reweight=True` (the equally-weighted / regression-adjustment
  influence-function path), replicate-weight designs, and non-pweight (fweight/aweight) types,
  each a deferred follow-up. `LPDiDResults` gains `survey_metadata` / `n_strata` / `n_psu`, a
  `"survey_tsl"` `vcov_type`, and a Survey Design block in `summary()`. The non-survey path is
  byte-for-byte unchanged. Validated against `survey::svyglm` on the stacked long difference
  (full numeric golden parity in Phase D2, below).
- **`LPDiD` complex-survey R-parity validation** (Phase D2). Pins the D1 survey path end-to-end
  against `survey::svyglm` (Lumley) goldens on a dedicated staggered-absorbing survey panel:
  per-horizon point/SE/df + pooled-post/pre for three variance paths: the variance-weighted full
  design (strata+PSU+FPC), the weights-only unit-injected-PSU design, and the direct-covariate
  variant (point ~1e-6, SE ~1e-5, df exact via the per-design `n_PSU - n_strata` / `n_PSU - 1`
  formula). Each horizon uses a fresh `svydesign` over that horizon's clean sample, matching the
  library's per-sample resolution; `svyglm` is the reference implementation of the Binder TSL
  sandwich so it anchors the variance directly, and the clean-sample construction is independently
  cross-checked against `alexCardazzi/lpdid` (the unweighted variance-weighted event study matches
  to <1e-8). New `tests/test_methodology_lpdid.py::TestLPDiDSurveyParityR` +
  `benchmarks/R/generate_lpdid_survey_golden.R` + `lpdid_survey_panel.csv` /
  `lpdid_survey_golden.json` (own seed, so the absorbing / non-absorbing goldens stay
  byte-identical). No estimator change.
- **`TROP` non-absorbing (on/off) treatment support** (Athey, Imbens, Qu & Viviano 2025,
  §2.1 / Eq. 12 / Algorithm 2). New `non_absorbing` parameter (default `False`). The paper
  supports general assignment patterns ("units moving into and out of treatment"), not only
  absorbing/staggered adoption; `TROP(non_absorbing=True)` (`method='local'` only) now
  accepts treatment that switches on and off, imputing each treated cell's counterfactual via
  the paper's `(1-W)` masking. The default `non_absorbing=False` is unchanged and still
  rejects non-monotonic D with a `ValueError` (now also pointing to the opt-in), guarding
  against the common mistake of encoding absorbing treatment as an event-style spike. This
  *removes a prior implementation over-restriction* (the estimator was stricter than the
  paper) rather than adding a deviation. `method='global'` keeps its block-assignment
  requirement and rejects `non_absorbing=True`. A one-time `UserWarning` is emitted noting
  that validity relies on the no-dynamic-effects assumption and that the triple-robustness
  guarantee (Theorem 5.1) is proven only under block assignment. The Rust local LOOCV and
  point-estimate paths were already mask-driven and unchanged (Rust/Python ATT parity is
  regression-tested); the non-absorbing **bootstrap** is routed to the Python path, because
  the Rust resampler lacks the no-weighted-control-support guard and can return a degenerate
  ~0 SE on an empty control stratum. Treated cells with no weighted control support (e.g. an
  always-treated unit under `lambda_unit>0`) are materialized as NaN and excluded from the
  ATT (the library non-estimable->NaN convention), with a `UserWarning`.
- **`LPDiD` non-absorbing R-parity validation** (Phase C2). Pins both non-absorbing modes
  against an independent `fixest::feols` reconstruction of the paper's Eq. 12 (`first_entry`)
  and Eq. 13 (`effect_stabilization`) clean-sample restrictions: variance-weighted point and
  SE match to ~1e-13/~1e-15; the `effect_stabilization` reweighted point matches (its SE is
  pinned as a regression guard, a small weighted-cluster convention difference). New `tests/test_methodology_lpdid.py`
  parity class + separate `lpdid_nonabsorbing_panel.csv` / `lpdid_nonabsorbing_golden.json`
  (the absorbing B2 goldens stay byte-identical). `alexCardazzi/lpdid`'s `nonabsorbing_lag` is
  recorded as a divergent third-party reference (it clamps off-switches and uses a non-paper
  boundary/placebo convention, diverging ~0.01-0.05 from Eq. 13 even on a monotone panel), not
  a parity gate. No estimator change.
- **`LPDiD` non-absorbing (reversible) treatment** (Dube, Girardi, Jordà & Taylor 2025,
  Section 4.2). New `non_absorbing` parameter: `"first_entry"` (Eq. 12 — the effect of
  entering treatment for the first time and staying treated) and `"effect_stabilization"`
  (Eq. 13, with `stabilization_window=L` — units whose treatment has been stable for at least
  `L` periods serve as clean controls, so estimation is feasible with few or no never-treated
  units). The default `non_absorbing=None` is unchanged (absorbing path; still rejects
  non-absorbing input, bit-for-bit identical results). Both modes implement the entry-effect
  estimands with mode-aware clean-sample masks, a documented "untreated before the first
  observed period" boundary convention, and a gap-free-panel requirement; the Appendix-C
  exit-event dynamics and survey-design support remain follow-ups (R-parity is covered by
  the entry above).
  Pure-Python validation covers the absorbing reduction, the re-entry mechanism, pre-trend
  placebos, non-negative weighting, stabilized-control admission, and DGP recovery.
- **Weighted multiple absorbed fixed effects (`absorb=[a, b, ...]`) now supported in
  `DifferenceInDifferences` / `MultiPeriodDiD`.** The prior `ValueError` rejecting multi-absorb
  with survey weights is lifted: the absorb path now uses the method of alternating projections
  (`diff_diff.utils.demean_by_groups`), the exact weighted Frisch-Waugh-Lovell residualization for
  N > 1 dimensions. New `demean_by_groups()` N-way helper; the two-way `within_transform()` now
  delegates to it. Single-absorb and balanced-panel results are byte-stable (weighted
  `within_transform` output is bit-identical; balanced multi-way matches the prior closed-form
  demean to machine precision).
- **`LPDiD` R-parity validation (absorbing).** `tests/test_methodology_lpdid.py` pins the
  estimator against the method authors' own R recipes (`danielegirardi/lpdid` event-study /
  reweight / premean / pooled `fixest::feols` specifications) with an `alexCardazzi/lpdid`
  cross-check gate on the event-study variants, generated by
  `benchmarks/R/generate_lpdid_golden.R`. Variance-weighted, reweighted, premean,
  direct-covariate, pooled, and regression-adjustment-point estimates match to ~1e-12
  (pooled is anchored to the authors' fixed-composition recipe only — `alexCardazzi`'s pooled
  uses a laxer clean-control window and is recorded for transparency, not gated). The
  regression-adjustment standard error (canonically Stata `teffects ra` only — no R-package
  analogue) is pinned and its calibration validated by an ungated Monte-Carlo coverage study
  (`benchmarks/python/coverage_lpdid_ra.py`, ≈0.95 coverage across cluster counts). Confirms
  the two previously-provisional REGISTRY notes (RA influence-function variance convention;
  pooled fixed-composition estimand) resolve in the library's favour with no estimator change.

### Changed
- **`EfficientDiD` survey-weighted Silverman bandwidth** (covariate DR path). The auto
  Silverman bandwidth for the kernel-smoothed conditional `Omega*(X)` now uses a
  **survey-weighted** per-dimension dispersion (weighted mean/std over the positive-weight
  support) instead of the unweighted sample dispersion, so the bandwidth reflects the
  population covariate distribution the kernel targets. The rate term `n` remains the
  positive-weight support count. This shifts the DR point estimate and SE only in
  **overidentified (H>1) covariate cells under non-uniform survey weights**; under uniform
  weights it reduces to the previous bandwidth up to floating point, and it preserves the
  existing invariances to zero-weight (subpopulation / padded) rows and to weight rescaling.
  Non-survey and just-identified (H=1) paths are unchanged.

## [3.6.0] - 2026-06-29

### Added
- **`LPDiD` (Local Projections Difference-in-Differences; Dube, Girardi, Jordà & Taylor 2025,
  *J. Applied Econometrics* 40(5):741-758).** Per-horizon long-difference OLS
  (`y_{i,t+h} − y_{i,t−1}`) on a clean-control sample (newly-treated + not-yet-treated) with
  calendar-time fixed effects and no unit FE, so the default variance-weighted estimand has
  strictly non-negative weights (no TWFE negative-weighting). Options: `reweight=True`
  (equally-weighted ATT, numerically equivalent to Callaway-Sant'Anna), premean-differenced base
  period (`pmd`), `no_composition` (fixed post-treatment composition), pooled pre/post estimands,
  outcome/first-difference lag controls (`ylags`/`dylags`), and a regression-adjustment covariate
  path (ImputationDiD/BJS-family influence-function cluster variance). Cluster-robust SEs at the
  unit level by default. This release implements the **absorbing-treatment** path; per-unit
  interior time gaps are handled by calendar-correct feature construction. Non-absorbing
  treatment, survey-design support, and external R-package parity are tracked follow-ups.
- **`placebo_group_test` gained an optional `treatment` parameter.** When supplied, units
  that are ever real-treated are dropped before the placebo so it runs on never-treated units
  only (the uncontaminated design); without it, behavior is unchanged and the caller must pass
  control-only data. Degenerate designs (all fake-treated units dropped, or no controls
  remaining) now raise a clear `ValueError` instead of a cryptic `LinAlgError`, and a
  fake-treated unit that is itself real-treated emits a `UserWarning`.
- **PlaceboTests methodology validation:** `tests/test_methodology_placebo.py` (paper-anchored
  to Bertrand-Duflo-Mullainathan 2004) plus base-R exact-enumeration R parity
  (`benchmarks/R/generate_placebo_golden.R` → `benchmarks/data/placebo_golden.json`). The
  `PlaceboTests` methodology-review row is promoted to **Complete**.
- **Survey Data Support methodology validation:** `tests/test_methodology_survey.py` (33 tests
  anchored to Binder 1983 Eq. 4.7 / `docs/methodology/survey-theory.md` §5/§6) isolates the
  design-based TSL and replicate-weight variance identities — the multi-stratum Bessel
  decomposition, the fweight (`df=Σw−k`) / aweight (unweighted-meat) structures, the exact
  `DEFF = design_var/srs_var` ratio, and the residual-scale==score-scale identity — that the
  broad survey suite previously covered only indirectly. The core variance machinery
  (`compute_survey_vcov` / `_compute_stratified_psu_meat` / `compute_replicate_vcov` /
  `df_survey`) was read against Binder and verified faithful (no code change required). The
  `Survey Data Support` methodology-review row is promoted to **Complete** — the last In Progress
  row, so the methodology-review tracker is now fully Complete. Added Lumley (2004) JSS 9(8),
  Korn-Graubard (1990), and Solon-Haider-Wooldridge (2015) to `docs/references.rst`.

### Changed
- **CallawaySantAnna now materializes non-estimable `(g,t)` cells as NaN entries instead of
  omitting them.** Cells that cannot be estimated (missing base/post period, zero
  treated/control, zero survey-weight mass, or a non-finite regression solve) are stored in
  `group_time_effects` as NaN entries carrying a machine-readable `skip_reason`
  (`"missing_period"` / `"zero_treated_control"` / `"zero_weight_mass"` /
  `"non_finite_regression"`; estimable cells carry `None`), uniformly across all estimation paths
  (no-covariate regression, covariate regression, IPW/DR, repeated cross-section, survey-weighted)
  — previously only the covariate-regression singular case did this and the other paths dropped
  the cell silently from the grid. The cells are excluded from every aggregation
  (simple/group/event-study), from `balance_e`, and from the bootstrap, so all aggregate
  point estimates and standard errors — and the event-study `n_groups` / by-group `n_periods`
  metadata — are numerically **unchanged** and continue to match R `did`'s `aggte()`; a fit where
  no cell is estimable still raises `ValueError`. `to_dataframe("group_time")` now includes these
  NaN rows and a `skip_reason` column. This is a documented per-cell surface **deviation from R**'s
  `att_gt` (which omits the rows). See REGISTRY.md "CallawaySantAnna" edge cases.
- **CallawaySantAnna multiplier bootstrap now tiles weight generation over draws, cutting
  peak memory at large `n_units`.** The dense `(n_bootstrap × n_units)` multiplier-weight
  matrix (the dominant allocation for the default unit-level bootstrap — `cluster=None`,
  equivalently `cluster="unit"` — where each unit is its own
  PSU) is generated and consumed one draw-block at a time via the new
  `diff_diff/bootstrap_chunking.py` helper instead of being materialized in full. Measured peak
  RSS at 999 bootstrap reps drops ~79% at 500k units (11.6 GB → 2.4 GB) and ~68% at 1M units
  (10.8 GB → 3.4 GB); the previously out-of-reach millions-of-units × 999-rep regime now stays
  near the fit's memory floor. The weight *stream* is bit-identical on both backends (Rust
  absolute per-row seeding; NumPy in-order stream); end-to-end bootstrap SEs match to within
  floating-point reassociation of the BLAS reductions (~1 ULP, far below bootstrap Monte-Carlo
  error). Stratified survey designs (few PSUs) are unchanged (full generation + sliced blocks);
  see TODO.md for the deferred per-stratum tiling.
- **EfficientDiD and HeterogeneousAdoptionDiD multiplier bootstraps now tile weight generation
  over draws too, via the same `diff_diff/bootstrap_chunking.py` helper.** Both built the same
  dense `(n_bootstrap × n_units)` multiplier-weight matrix CallawaySantAnna did — EfficientDiD
  in its per-`(g,t)` EIF perturbation, HAD in its event-study sup-t band — the dominant
  allocation at large `n_units` (~40 GB at 5M units × 999 reps). It is now generated and
  consumed one draw-block at a time, capping peak memory at `O(block × n_units)`, so these
  estimators reach the same millions-of-units scale as the chunked CallawaySantAnna path. The
  weight *stream* is bit-identical on both backends; end-to-end bootstrap SEs and sup-t
  critical values match the un-chunked path to within floating-point reassociation of the BLAS
  reductions (~1 ULP, far below bootstrap Monte-Carlo error). As with CallawaySantAnna,
  stratified survey designs (few PSUs) are unchanged — full generation + sliced blocks — with
  the deferred per-stratum tiling tracked in TODO.md.
- **`run_placebo_test`'s `fake_group` path now filters ever-treated units by default.** The
  dispatcher threads its `treatment` column into `placebo_group_test`, so the fake-group
  placebo runs on never-treated units only (a more-correct placebo). Calling
  `placebo_group_test` directly without `treatment` retains the previous behavior.
- **Bumped the Rust backend's `blas-src` crate `0.10` → `0.14`.** `blas-src` is a
  linker-only crate pulled in **only by the `accelerate` (macOS) feature**; the Linux
  `openblas` path links system OpenBLAS via `build.rs` and the default/Windows builds use the
  pure-Rust `faer` backend, so neither touches `blas-src`. `0.14` links the **same** system
  Accelerate.framework as `0.10` (`accelerate-src` is unchanged at `0.3.2`), so there is no
  API or numerical change. Validated on the macOS Accelerate path: clean `cargo build` /
  `maturin develop --features accelerate` against the pinned `ndarray 0.17`, the Rust unit
  tests, and the full Python⇄Rust equivalence suite (`tests/test_rust_backend.py`).

### Performance
- **`within_transform` no longer takes a redundant full-frame copy.** The two-way within
  (fixed-effects) demeaning helper — shared by `TwoWayFixedEffects`, `SunAbraham`,
  `WooldridgeDiD`, and `BaconDecomposition` — copied the entire input frame defensively before
  `pd.concat`-ing the demeaned columns onto it, even though the demean is read-only and
  `concat` does not mutate its inputs. That copy is removed: the demeaned columns are attached
  as a single consolidated block via `pd.concat` (under pandas copy-on-write the original
  columns are shared, not copied). Peak RSS of a wide `TwoWayFixedEffects(vcov_type="hc1")` fit
  drops ~8% (e.g. 964 → 886 MB at 400k units × 6 covariates); the win scales with panel width.
  **Bit-identical** (proven at `atol=0` for TWFE incl. the replicate-weight path, SunAbraham,
  Wooldridge, and Bacon) — frame assembly only, the demean arithmetic is unchanged.
- **`ImputationDiD` conservative-variance: cache the untreated-projection factorization per
  fit.** The exact imputation projection `v_untreated = -A_0 (A_0'[W]A_0)^{-1} A_1'w` has a
  target-invariant design (`A_0`/`A_1`/factorization) and a target-specific RHS (`A_1'w`), but
  previously rebuilt the sparse design and re-`spsolve`d it for every estimand target (overall
  ATT, each event-study horizon, each group) and again for the bootstrap precompute. It now
  factorizes `(A_0'[W]A_0)` once per `fit()` via `scipy.sparse.linalg.factorized` and solves
  only the per-target RHS (factorize-once / solve-many), collapsing `O(1 + #horizons +
  #groups)` factorizations to one. **Bit-identical** to the prior per-target `spsolve` (proven
  at `atol=0` across FE-only, covariate, survey-weighted, and bootstrap paths); no methodology,
  numerical, or public-API change.
- **`LinearRegression(vcov_type="hc2_bm")`: compute the CR2 sandwich once, not twice.** The fit
  path previously computed the CR2/HC2-BM vcov inside `solve_ols`, then recomputed the entire
  Bell-McCaffrey sandwich (per-cluster `A_g`, the `O(n²k)` Satterthwaite loop) a second time via
  `compute_robust_vcov(return_dof=True)` just to extract the per-coefficient DOF. The vcov **and**
  DOF now come from a single combined call (the CR2 helper returns both together), halving the
  per-coefficient CR2 sandwich cost on every `hc2_bm` fit (weighted one-way WLS-CR2,
  weighted/unweighted clustered CR2-BM — i.e. `DiD`/`TWFE`/`MPD` with `vcov_type="hc2_bm"`).
  (`MultiPeriodDiD`'s separate post-period-average contrast DOF recomputation is addressed by the
  `MultiPeriodDiD` entry below.) **Bit-identical** SEs and DOF
  (proven at `atol=0`); no methodology, numerical, or public-API change. Minor side effect: the
  HC2→HC1 high-leverage-fallback `UserWarning` now fires once per fit instead of twice.
- **`MultiPeriodDiD(cluster=..., vcov_type="hc2_bm")`: build the CR2 precomputes once, not
  twice.** The cluster + CR2 Bell-McCaffrey path built the expensive per-cluster precomputes
  (the `A_g` adjustment-matrix eigendecompositions, `S_W`, the residual-maker `M = I − H`) twice
  per fit — once in `solve_ols`'s vcov path (whose per-coefficient DOF was then discarded) and
  again in the separate `_compute_cr2_bm_contrast_dof` call for the per-coefficient and
  post-period-average ATT Satterthwaite DOF. The vcov **and** all DOF now come from a single
  shared CR2 core (`_compute_cr2_bm_vcov_and_dof`) per fit; `_compute_cr2_bm` and
  `_compute_cr2_bm_contrast_dof` become thin wrappers over it (removing the formerly-duplicated
  ~50-line precompute block, so every CR2 caller routes through one implementation). The fit-level
  call computes vcov and the per-coefficient + avg-ATT DOF together from one precompute build.
  **Bit-identical** `vcov_`, per-period DOF, avg-ATT DOF, p-values, and CIs (proven at `atol=0`
  across balanced, unbalanced, and rank-deficient designs; a mechanism test asserts the
  per-cluster adjustment matrix is built exactly once per cluster, down from twice). Completes the
  `MultiPeriodDiD` follow-up noted in the `LinearRegression` entry above. No methodology,
  numerical, or public-API change.

### Fixed
- **Unbalanced-panel correctness: N > 1 absorbed fixed effects now use iterative alternating
  projections instead of single-pass sequential demeaning.** Affected `DifferenceInDifferences` /
  `MultiPeriodDiD` with `absorb=[a, b, ...]` and the shared unweighted two-way `within_transform`
  (used by `TwoWayFixedEffects`, `SunAbraham`, `BaconDecomposition`). A single sequential demean
  sweep is the exact Frisch-Waugh-Lovell residualization only when the fixed-effect subspaces are
  orthogonal (balanced fully-crossed panels); on unbalanced panels it was a biased approximation
  (coefficients off by ~1e-2 in tested cases). The within transformation now iterates to
  convergence (`diff_diff.utils.demean_by_groups`), matching R `fixest` / `reghdfe` / `lfe`.
  Balanced-panel and single-absorb results are unchanged to machine precision; the unweighted
  two-way path now also emits the non-convergence `UserWarning` (previously only the weighted path
  could).
- **Structural (non-covariate) matrix inverses are now rank-guarded.** The internal design-Gram
  bread inversions in `ContinuousDiD` (ACRT-variance `Psi'WPsi`), `TwoStageDiD` (Stage-2
  `X_2'WX_2`, both the analytical and multiplier-bootstrap surfaces), `SpilloverDiD` (Wave D
  `A_22`), and the Conley spatial-HAC variance (`X'WX`) sat on a `LinAlgError`-only fallback:
  `np.linalg.inv`/`solve` raise only on an *exactly* singular matrix, so a **near**-singular
  Gram returned a garbage inverse (~1e13) straight into the SE (and `ContinuousDiD`'s exact-
  singular fallback was a *silent* minimum-norm `pinv`; Conley *raised* `ValueError`). All now
  route through the shared `_rank_guarded_inv` (`diff_diff/linalg.py`) — the same generalized
  inverse already used for the covariate IF SEs — which truncates redundant directions on the
  equilibrated Gram to give a finite SE on the identified subspace (the well-conditioned
  near-collinear limit, not minimum-norm; NaN only at rank 0) and emits a `UserWarning` when a
  direction is dropped. For the per-coefficient reporters (TwoStageDiD event/group, SpilloverDiD
  rings, Conley), a dropped (unidentified) named coefficient is reported with **NaN** SE (not the
  zero-filled `0`); linear-combination reporters (ContinuousDiD dose curves) are unaffected, since
  a dropped direction correctly contributes 0 there. These invert *internal* bases users cannot
  perturb with `covariates=`.
  **Behavior change:** a rank-deficient Conley design no longer raises — it rank-reduces with a
  warning. Well-conditioned designs are unchanged (the fast path is `np.linalg.solve(A, I)`,
  R-parity preserved). (`HeterogeneousAdoptionDiD`'s non-symmetric IV bread and `ImputationDiD`
  — whose vcov is already rank-guarded upstream via `solve_ols` — were assessed and excluded;
  see `TODO.md`.)
- **`CallawaySantAnna` / `StaggeredTripleDifference` covariate outcome-regression is now
  scale-robust.** The covariate OR nuisance fits — `_compute_all_att_gt_covariate_reg` and
  `_doubly_robust` (CS) and `_compute_or` (StaggeredTripleDifference) — previously used an
  estimator-local `cho_solve(X'X)` cache fast path (with bare `scipy.linalg.lstsq(cond=1e-7)`
  fallbacks) that **bypassed the shared scale-equilibrated solver**. They now route through
  `solve_ols` (column-equilibrated SVD/gelsd), matching `TripleDifference._fit_predict_mu` and
  R's `lm()`/QR. A covariate **correlated with another regressor at a very large scale** (e.g. a
  large constant offset, near-collinear with the intercept) could perturb the point-estimate ATT —
  and the influence-function SE that follows it — because forming the normal equations squares the
  condition number; the equilibrated SVD avoids this (offset-invariant to ~1e-11 where the prior
  solve drifted, e.g. ~4e-6 at an offset of 1e6, growing with scale). Pure orthogonal ill-scaling
  was already safe (diagonal `X'X`), so the practical impact is confined to ill-conditioned /
  correlated covariate designs. **Not bit-identical** (cho/normal-equations → SVD) — well-scaled
  designs move only ~1e-12, well under the R-parity tolerances; the existing CS covariate R-`did`
  golden tests (`reg`/`ipw`) still pass, a new `with_covariates_dr` att+SE golden parity test is
  added (R-parity at ~1e-3), and scale-invariance tests pin the fix. No change to estimands,
  identifying assumptions, the no-covariate path, or the propensity-score fits.
- **Corrected the Korn & Graubard (1990) citation venue** in `docs/methodology/REGISTRY.md`
  (Survey Degrees of Freedom) from *JASA* 85(409) to *The American Statistician* 44(4), 270-276
  — the survey-df / Bonferroni-t paper (DOI 10.1080/00031305.1990.10475737).
- **`permutation_test` now reports the randomization-inference p-value
  `(1 + count) / (B + 1)`** (Phipson & Smyth 2010), replacing `count / B` floored at
  `1/(B+1)`. The `+1` includes the observed statistic in both numerator and denominator
  (the floor is now intrinsic). Because assignments are sampled with replacement, this is a
  valid but slightly conservative Monte-Carlo randomization-inference p-value (not an exact
  finite-sample value); it converges to the exact full-enumeration value `count/total` as the
  number of permutations grows. (Permutation p-values shift by a small amount, at most
  `~1/(B+1)`.)

### Security
- **Bumped the Rust backend's `pyo3` and `numpy` crates 0.28 → 0.29.** Resolves two RustSec
  advisories in `pyo3 < 0.29` — RUSTSEC-2026-0176 (out-of-bounds read in `PyList`/`PyTuple`
  `nth`/`nth_back`, High) and RUSTSEC-2026-0177 (missing `Sync` bound on
  `PyCFunction::new_closure`, Medium). Neither vulnerable path was reachable in this crate
  (no `PyList`/`PyTuple` iteration, no `new_closure`, no free-threaded wheels); `numpy` 0.29 is
  bumped in lockstep because it requires `pyo3` ^0.29. No API or numerical change — both crates
  are FFI/binding layers, and the math/RNG crates (`ndarray`, `faer`, `rand`, `rand_xoshiro`)
  are unchanged.

## [3.5.3] - 2026-06-25

### Added
- **`generate_synthetic_control_data()` data generator + a capstone `SyntheticControl` tutorial.** New public generator (`diff_diff/prep_dgp.py`, exported from `diff_diff`) builds a **single-treated-unit** factor-model panel for synthetic-control demos and tests: one treated unit whose latent factor loadings and baseline are an exact convex combination of a few donors (so the noiseless trajectory lies in the donor convex hull and a synthetic control reproduces it closely — the observed fit is approximate under added noise), persistent AR(1) factors, predictor covariates that each proxy a distinct factor, a common calendar time effect, and a known `"ramp"` or `"constant"` treatment effect emitted as `true_effect`. Tutorial **`docs/tutorials/25_synthetic_control_policy.ipynb`** walks the whole `SyntheticControl` surface end-to-end on a policy-evaluation story (one state adopts a clean-energy standard), structured around **two inference philosophies**: cross-unit permutation (`in_space_placebo` + Firpo–Possebom `confidence_set`, with `leave_one_out` / `in_time_placebo` robustness) versus over-time conformal (CWZ `conformal_test` / `conformal_confidence_intervals` / `conformal_average_effect`), with the per-period conformal band as the climax. A `tests/test_t25_synthetic_control_policy_drift.py` drift guard re-derives every quoted number from the generator.
- **`TwoStageDiD` methodology validation (Gardner 2022 / `did2s`).** New `tests/test_methodology_two_stage.py` with paper-equation-numbered Verified Components (§3 two-stage procedure / eqs. 4 & 6, §3.3 GMM variance, fn. 19 always-treated exclusion, Proposition 5, covariate path, `balance_e`, `vcov_type` narrowing) plus a `did2s::did2s()` cross-language parity fixture (`benchmarks/R/generate_did2s_golden.R` → `benchmarks/data/did2s_golden.json` + `did2s_test_panel.csv`), pinning overall + event-study ATT (abs 1e-6) and SE (abs 1e-7). `METHODOLOGY_REVIEW.md` `TwoStageDiD` row flipped to **Complete**.
- **`p_val_type` parameter on `DifferenceInDifferences` (and inherited `TwoWayFixedEffects`)** for the wild cluster bootstrap, mirroring `fwildclusterboot::boottest`: `"two-tailed"` (default — test on `|t|`, two-tailed inverted CI, which may be asymmetric) or `"equal-tailed"` (each tail at `alpha/2`, equal-tailed CI). Only used when `inference="wild_bootstrap"`; inert on `MultiPeriodDiD` (which falls back to analytical inference).

### Fixed
- **Wild cluster bootstrap (`inference="wild_bootstrap"`) now imposes the null — fixes an invalid p-value (issue #543).** `DifferenceInDifferences`/`TwoWayFixedEffects` with `inference="wild_bootstrap"` previously produced a p-value that contradicted its own confidence interval (e.g. CI `[2.30, 2.64]` excluding 0, yet `p = 0.86`). `diff_diff.utils.wild_bootstrap_se` *claimed* to run the Wild Cluster **Restricted** bootstrap but never actually imposed the null — it re-fit the full design (keeping the treatment column) to the unchanged outcome, so the "restricted" residuals equaled the unrestricted ones and the bootstrap coefficient distribution centered on the estimate instead of 0. The p-value `mean(|t*| ≥ |t₀|)` then measured noise around the estimate (≈0.5–0.86 regardless of significance) while the percentile-of-coefficients CI happened to look fine — an internal contradiction. The bootstrap now genuinely imposes H₀ (drops the coefficient's column for the restricted fit), studentizes with the analytical CR1 SE, and derives the CI by **test inversion** so the p-value and CI are exactly consistent (`0 ∈ CI ⟺ p ≥ alpha`). For Rademacher weights with few clusters the full `2**n_clusters` sign-vector set is enumerated (deterministic), matching R's `fwildclusterboot::boottest`. **Results change** for any prior `wild_bootstrap` use: the headline `p_value`/`conf_int` are corrected (a true effect is now correctly significant), and the reported `se` is now the analytical cluster-robust (CR1) SE (numerically ~unchanged in well-behaved cases). Validated against `fwildclusterboot::boottest()` (`benchmarks/R/generate_wild_cluster_boot_golden.R`; bootstrap t-distribution to ~6e-14, `se`/`t`/interior-`p` exact, CI to ~1e-4) and an independent full-refit enumeration. See `docs/methodology/REGISTRY.md` §"Wild cluster bootstrap (WCR)".
- **Cluster-robust / HC1 standard errors no longer raise `ZeroDivisionError` on a saturated design.** `linalg.compute_robust_vcov` (NumPy path) divided by `(n_eff - k)` in the HC1/CR1 small-sample adjustment without guarding a design with no residual degrees of freedom (`n_eff == k`, e.g. a 2×2 DiD with one observation per cluster-period); it now returns a NaN vcov so inference is degenerate (NaN), consistent with the all-or-nothing NaN convention, rather than crashing. Surfaced while hardening the wild cluster bootstrap (`wild_bootstrap_se` independently routes saturated / weak-identification designs to NaN, and represents a genuinely unbounded inverted CI with `±inf` instead of mixing finite point estimates with NaN endpoints).
- **Wild cluster bootstrap on a rank-deficient full-dummy design no longer crashes when storing the vcov.** `_run_wild_bootstrap_inference` computed the stored cluster-robust vcov via `compute_robust_vcov(X, ...)` on the full design, which inverts `X'X` directly and raises (or returns garbage) when a nuisance column is collinear (e.g. a fixed-effect dummy collinear with treatment) — even though the ATT is identified and the bootstrap itself drops such columns. It now computes the stored vcov through the rank-aware `solve_ols(..., rank_deficient_action="silent")` path, NaN-expanding the dropped column (bit-identical to the prior result on full-rank designs).
- **`TwoStageDiD` analytical GMM standard errors are now exact (match R `did2s` to ~1e-7).** The Gardner two-stage GMM sandwich `_compute_gmm_variance` derived its residuals from the *iterative* alternating-projection first-stage fixed effects (`_iterative_fe`, which converge only to ~1e-7 on unbalanced untreated panels) while computing `gamma_hat` exactly — leaving the variance ~1% off the analytical sandwich. The variance now re-solves the Stage-1 FE **exactly** (sparse OLS, reusing the `gamma_hat` factorization), and `_build_fe_design` gained an intercept column so its column space spans the grand mean (the prior intercept-free design omitted it, and the exact residual is first-order sensitive to it). Unidentified-FE obs (rank-deficient / Proposition-5) fall back to the iterative residual, so those edge cases are unchanged; the reported `overall_att` still uses the iterative FE (point-estimate equivalence with `ImputationDiD` preserved). Mirrors the same-class fix already applied to `ImputationDiD`'s exact-sparse variance.
- **`LinearRegression.get_se()` / `get_inference()` no longer return a `NaN` standard error from a tiny-negative variance artifact.** A high-leverage / degenerate coefficient (e.g. an absorbed-FE dummy near-collinear with the treatment, whose Bell-McCaffrey Satterthwaite DOF already hits the noise-floor guard) can have a CR2/HC variance of ~0 (≈1e-32) whose vcov diagonal lands just-below-zero under BLAS-dependent float rounding; `np.sqrt` of the negative then produced a `NaN` SE **nondeterministically** — passing single-threaded but failing under the parallel pure-Python full-suite run (`tests/test_methodology_wls_cr2.py::TestLinearRegressionFENanGuardEndToEnd::test_did_absorbed_fe_lr_inference_nan_for_guarded_coefs`). Both SE sites now clamp the vcov diagonal at 0, so the SE is finite (0 for a genuinely-zero variance), deterministic, and BLAS-independent. **No change for any positive variance** (the clamp is a no-op there); only the previously-`NaN` degenerate case is affected.
- **`TripleDifference` power analysis now honors `n_periods > 2`.** `simulate_power`,
  `simulate_mde`, and `simulate_sample_size` previously routed DDD to the
  cross-sectional 2×2×2 `generate_ddd_data` regardless of `n_periods` (emitting an
  "n_periods ignored" warning). They now route to the panel DGP
  `generate_ddd_panel_data` when `n_periods > 2`, honoring `n_periods`/`treatment_period`
  and sizing the panel by `n_units` directly (the sample-size search switches from the
  multiple-of-8 grid to a continuous step-1 search). Because `simulate_power` defaults
  to `n_periods=4`, the default DDD power call now uses the panel DGP. The panel DGP has
  within-unit serial correlation, so construct the estimator as
  `TripleDifference(cluster="unit")` for valid power — a `UserWarning` fires otherwise.
  `treatment_fraction` remains inert (balanced 2×2×2); pass `group_frac`/`partition_frac`
  via `data_generator_kwargs`. See `docs/methodology/REGISTRY.md` §PowerAnalysis.

## [3.5.2] - 2026-06-08

### Added
- **`StackedDiD` covariate balancing (CBWSDID; Ustyuzhanin 2026, arXiv:2604.02293).** New constructor parameter `balance="entropy"` plus `fit(..., covariates=[...])` add a within-sub-experiment design stage: entropy balancing (Hainmueller 2012) reweights the clean controls toward the treated cohort's covariate means (read at the last pre-treatment period), and the resulting design weights `b_sa` compose with the Wing et al. (2024) corrective weights via the effective control mass into the final stacked weights `W_sa`. This is **control-only reweighting**, so it estimates untreated trends under *conditional* parallel trends while preserving the trimmed-aggregate-ATT estimand (at `b_sa=1` it reduces to the paper's unit-count weighted stacked DID, equal to `StackedDiD(weighting="aggregate")` on balanced event windows). Inference reuses the existing conditional-on-weights cluster-robust path. Scope: requires `weighting="aggregate"` and **balanced event windows** (ragged windows raise — the unit-count vs observation-count convention is unresolved off balanced panels); `population`/`sample_share`/`survey_design=` and matching-based balancing / the repeated-treatment extension are not supported (raise `NotImplementedError`). Infeasible cohorts fail closed with a clear error. New `diff_diff/balancing.py` (entropy-balancing solver). Estimand validated end-to-end against the closed-form CBWSDID formula (`tests/test_methodology_stacked_did.py`).
- **`SyntheticControl` conformal inference (Chernozhukov, Wüthrich & Zhu 2021, *JASA* 116(536)).** Three opt-in `SyntheticControlResults` methods give valid p-values for the post-period effect trajectory and pointwise confidence intervals — what the in-space placebo / Firpo-Possebom test-inversion paths cannot. Unlike the Firpo path (which re-ranks the cross-unit placebo gaps), the conformal layer fits its **own** time-permutation-invariant constrained-LS synthetic-control proxy (CWZ §2.3 eqs 3–4 — simplex weights on raw outcomes over **all** periods under the null, no `V`-matrix, no intercept) and permutes residuals **over time** for the single treated unit (CWZ's exactness theory requires a time-symmetric proxy, which the headline ADH `V`-matrix fit is not). **`conformal_test(effect, q=1, scheme="moving_block", n_iid=10000, seed=None)`** computes the joint sharp-null permutation p-value (eqs 1–2) of `S_q(û) = ((1/√T*)·Σ_{t>T0}|û_t|^q)^{1/q}` (`q ∈ {1, 2, ∞}`); the proxy is fit once and only residuals are permuted (footnote 7). **`conformal_confidence_intervals(alpha=0.1, scheme="moving_block", bounds=None, n_grid=100, seed=None)`** returns pointwise per-period CIs by test inversion (Algorithm 1 — each period `t` uses `Z = (pre-periods, t)` with the other post-periods dropped, a clean `T*=1` test). **`conformal_average_effect(alpha=0.1, scheme="moving_block", bounds=None, n_grid=200, seed=None)`** returns a CI for the average post-period effect by collapsing the panel into non-overlapping `T*`-blocks and permuting the block residuals (Appendix A.1). Permutation schemes: `"moving_block"` (`Π_→` cyclic shifts, valid under serial dependence — the default) and `"iid"` (`Π_all`, sampled, finer p-values); both include the identity so the p-value floor is `1/|Π|` (no extra `+1`). Fail-closed handling for `<1` donor / unpickled result / non-finite panel / non-converged grid points (treated as indeterminate, not rejected) / grid-limited / empty / unbounded sets; a single donor and `T*≥T0` warn. Surfaced under `conformal_inference` / `get_conformal_grid_df()` and `DiagnosticReport`'s `estimator_native_diagnostics`; the analytical `se`/`t_stat`/`p_value`/`conf_int`/`is_significant` stay NaN throughout. Core in the new `diff_diff/conformal.py` (reuses the Frank-Wolfe simplex solver). *Deferred:* one-sided variants (§7), covariates folded into the proxy, and the AR/innovation-permutation path (Lemmas 5–7).
- **`SyntheticControl` confidence sets by test inversion (Firpo & Possebom 2018 §4, PR-B).** Classic SCM gains the uncertainty quantification it has lacked — a confidence set for the treatment-effect *path* — without changing its always-NaN analytical inference contract. Two opt-in `SyntheticControlResults` methods built ON TOP of the in-space placebo: `test_sharp_null(effect, gamma=0.1)` tests a sharp null `H_0: α_1t = f(t)` (Eq 11; `effect` a scalar constant effect or a length-`n_post` post-period path) by subtracting `f(t)` from every unit's post-period gaps and re-ranking the modified RMSPE ratio `RMSPE^f` (Eqs 12–13 at `φ=0`, `v=(1,…,1)`), and `confidence_set(family="constant"|"linear", gamma=0.1, bounds=None, n_grid=200)` inverts that test into a confidence set — a constant-in-time interval (Eqs 15–16) or a linear-in-time slope set (Eqs 17–18) — keeping every value whose sharp null is not rejected at the paper's **strict** `p^f > γ` boundary (Eq 14). The whole computation is a **pure re-ranking of the gap paths `in_space_placebo()` already computes** (no synthetic-control refits): under a common-effect null the donor synthetics and the pre-period MSPE denominators are unchanged — only the post gaps shift by `f(t)` — so each grid value costs an `O(J)` rank, not a refit. With `bounds=None` the set is recovered **EXACTLY** by piecewise-constant breakpoint inversion: `p^c` is constant between the real roots of the placebo-vs-treated comparison quadratics, so `p` is evaluated once per induced interval AND at each breakpoint (a tie under `≥` can lift `p` above γ there, yielding an isolated accepted point) — NO centering/monotonicity assumption, so accepted tails, disjoint components, and unbounded/empty sets are all handled (a poor-pre-fit treated unit can have its accepted region in the tails). `bounds=(lo,hi)` instead scans a fixed grid (grid-limited); `n_grid` controls only the returned inspection table when `bounds=None`. Results: a pickle-surviving `effect_confidence_set` summary (`{family, parameter, gamma, lower, upper, contiguous, status, …}`, `status ∈ {"ran","empty","unbounded"}`) + a `get_confidence_set_df()` grid table, surfaced under `estimator_native_diagnostics.confidence_set`. **The analytical `conf_int`/`se`/`t_stat`/`p_value` stay NaN** — this is a permutation set at level `1−γ` (γ granular in `1/(J+1)`), possibly a set / unbounded / non-contiguous, so it cannot be coerced into the Wald-interval `conf_int` tuple; it is kept separate exactly as `placebo_p_value` is kept off `p_value`. **Fail-closed:** `γ < 1/(J+1)` (no value rejectable — fn 8) or a treated unit lacking the best pre-fit → `"unbounded"` (`±inf` + warning); no interval or breakpoint accepted → `"empty"` (NaN endpoints); a non-contiguous accepted region (disjoint components / an isolated singleton) → the `[lower, upper]` hull with `contiguous=False` + warning; `< 2` donors / a non-converged treated fit / an unpickled result (no placebo reference set) → `ValueError`. `test_sharp_null(0)` is held bit-for-bit equal to `placebo_p_value` (Eq 5 = Eq 13) by reusing each unit's **per-unit** floored pre-period denominator persisted from the placebo run. **Scope:** the sensitivity-analysis weights (`φ≠0`, Eqs 7–9), the general test-statistic menu (Eq 19), one-sided (§7's signed-`t` statistic), and the multiple-outcome/treated extensions (§6) are deferred (flagged in the paper review checklist). **Validation:** no R anchor (R `Synth` has no test inversion; the authors' Code Ocean capsule was not consulted) — self-consistency to the (Basque-R-anchored) `placebo_p_value`, a numpy oracle on Eqs 12–14 (incl. the strict `p=γ` boundary and the per-unit floor), invariants (the point estimate lies in the constant set for a well-posed fit; a center-rejected/tails-accepted regression; an isolated-breakpoint singleton; monotone-in-γ), and a coverage simulation. Consumes the PR-A `firpo-possebom-2018-review.md`; documented in `docs/methodology/REGISTRY.md` §SyntheticControl (new methodology block + `**Note:**` labels for the boundary convention, the grid choice, the non-analytical `conf_int` contract, and the no-R-anchor validation), `docs/api/synthetic_control.rst`, and the LLM guides.

### Changed
- **`SyntheticDiDResults.placebo_effects` renamed to `variance_effects`.** The
  array's contents are method-specific — placebo treatment effects
  (`variance_method="placebo"`), per-draw bootstrap ATT estimates
  (`"bootstrap"`), or leave-one-out estimates (`"jackknife"`) — so the old name
  was misleading; the `variance_method` field disambiguates the contents. Read
  `result.variance_effects` going forward.
- **`ImputationDiD` methodology-review-tracker promotion: In Progress → Complete.** Completes the source-validation pass (PR-B) of the Borusyak, Jaravel & Spiess (2024, REStud 91(6)) audit — PR-A (#529) added the paper review on file; this PR validates the source against the code (uncovering and fixing the SE corrections above), adds paper-equation-numbered Verified Components (`tests/test_methodology_imputation.py`: Theorem 1/2 imputation; Theorem 3 / eqs. 6-8 variance + the unit-clustered Equation 8 hand-calc; Proposition 5 `K >= H_bar` non-identification; Test 1 / eq. 9 + Proposition 9 pre-trend independence), an R `didimputation` v0.5.0 parity fixture + generator (`benchmarks/R/generate_didimputation_golden.R`), and flips the tracker row to **Complete** with a Verified Components / Corrections Made / Deviations / R Comparison Results detail block. Documented deviations: the unit-clustered Equation 8 matches R only at the cohort×event-time partition (diff-diff additionally offers coarser partitions, no R analogue); the multiplier bootstrap and survey-design TSL variance are library extensions; leave-one-out variance (Supplementary Appendix A.9) is not implemented. REGISTRY `## ImputationDiD` updated (Equation 8 now the exact unit-clustered form; `**Note (deviation from R):**`); `TODO.md` PR-B rows removed.

### Deprecated
- **`SyntheticDiDResults.placebo_effects`** is now a read-only alias for
  `variance_effects` that emits a `DeprecationWarning` on access; it will be
  removed in v4.0.0. The alias is a property, not a dataclass field, so it is
  read-only (assignment raises `AttributeError`) and
  `dataclasses.replace(result, placebo_effects=...)` no longer works /
  `dataclasses.asdict(result)` now emits the `variance_effects` key — use
  `variance_effects`.

### Fixed
- **`ImputationDiD` conservative standard errors were biased downward without covariates (~27% on a staggered panel) — corrected to the exact two-way-FE imputation projection (behavior change).** The Theorem-3 conservative variance (Borusyak, Jaravel & Spiess 2024, eqs. 6-7) needs the implied observation weights `v_it = -A_0 (A_0' A_0)^{-1} A_1' w` of the untreated observations. The covariate-free path used a closed form `-(w_i/n0_i + w_t/n0_t - w/N_0)` that is exact only for a **balanced** untreated set, but `Omega_0` is generically **unbalanced** in staggered designs (treated observations are removed) — so every analytical SE / t-stat / p-value / CI **without covariates** was biased (the **point estimates were always correct** — they are computed by imputation, not via `v_it`). Replaced with the exact sparse two-way-FE projection (the path the covariate case already used), and corrected that projection's design matrix to keep **all** unit dummies (the previous drop-first-unit-with-no-intercept design projected onto a space one rank short of the true two-way-FE span — a further ~1.6% bias). The standard errors now match R `didimputation::did_imputation` to ~1e-10 (`tests/test_methodology_imputation.py::TestImputationDiDParityR`, goldens `benchmarks/data/didimputation_golden.json`). **Potentially breaking:** covariate-free `ImputationDiD` analytical SEs (and dependent t/p/CI) change — they were previously too small. The multiplier bootstrap (`n_bootstrap>0`) resamples the same Theorem-3 influence function (`v_it * epsilon_tilde_it`), so **bootstrap SEs may also shift**. A genuinely rank-deficient `Omega_0` (e.g. an unidentified period FE) now routes through a least-squares fallback with a `UserWarning` instead of a silent (balanced-assumption) value.
- **`ImputationDiD` auxiliary variance model (Equation 8) now uses the paper's unit-clustered aggregator.** `_compute_auxiliary_residuals_treated` computed the observation-level mean `tau_tilde_g = sum(v*tau_hat)/sum(v)`; corrected to BJS Equation 8's unit-clustered `sum_i (sum_t v)(sum_t v*tau_hat) / sum_i (sum_t v)^2` (the excess-variance-minimizing form, Supplementary Appendix A.8). The two coincide under uniform within-group weights (the unweighted overall ATT / single-horizon event study under the default `aux_partition="cohort_horizon"`) and differ for survey-weighted / heterogeneity estimands or the coarser `aux_partition="cohort"`. Zero-weight / unimputable (NaN `tau_hat`) rows are excluded from the aggregation (exact for finite `tau_hat`, NaN-safe). At the default partition this equals R `didimputation`'s `sum(v^2*tau)/sum(v^2)`.
- **`ImputationDiD` untreated Step-1 residuals preserve NaN for missing fixed effects** (symmetric with the treated path) instead of a silent `fillna(0.0)`. Provably inert on valid data — every untreated observation's unit and period appear in the Step-1 FE estimates — but it stops a rank-condition logic error from masquerading as a 0 residual (any NaN is zeroed downstream in the variance product, as before).

## [3.5.1] - 2026-06-02

### Added
- **`SyntheticControl` cross-validation + inverse-variance `V`-selection (ADH 2015 §; Abadie 2021 §3.2(a), Eq. 9).** Two new `v_method` values complete the ADH-2015/Abadie-2021 `V`-selection menu (joining `"nested"` / `"custom"`), each threaded through the in-space / leave-one-out / in-time placebo refits so a diagnostic uses the **same** estimator as the headline fit. **`v_method="cv"`** selects the diagonal predictor-importance `V` by out-of-sample cross-validation: the pre-period is split positionally at `v_cv_t0` (new constructor param; default `len(pre)//2`, Abadie 2021's `t0 = T0/2`) into a training and a validation window, `V` is chosen to minimize the validation-window outcome MSPE of the training-fit weights (`mspe_v` now reports this validation MSPE under cv), and the final reported weights are re-estimated on the validation-window predictors (ADH 2015 step 4). Each predictor spec is **re-aggregated** over each window (its mean/sum/identity recomputed over only the periods that fall in that window — a separate `dataprep` per window, exactly as ADH 2015's CV does, since R `Synth` has no built-in CV function), so the V-search is genuinely out-of-sample for every predictor type and the same `V*` drives both fits with no zeroed coordinate (`v_weights` reproduce `donor_weights` on the validation-window predictors, and `predictor_balance` is reported on that validation-window basis). **Fully-spanning precondition (fail-closed):** re-aggregating a predictor on each window requires it to be observed in **both** windows, so `cv` **requires every predictor to span both the training and validation windows** and raises `ValueError` otherwise — satisfied by ADH 2015's shared covariate / multi-period `special_predictors` (which span the windows) but NOT by the default per-period outcome lags (each is single-period and lives in one window only), so `cv` with the bare default predictors is rejected with guidance to pass spanning predictors. In-time-placebo truncation that breaks the fully-spanning precondition (a kept spec stops spanning both windows at the truncated split) marks that date `infeasible`. A second fail-closed gate covers windows that span but carry **no cross-donor variation** (every re-aggregated predictor constant across the donors, so `X0·W` is constant in `W` → a flat, unidentified weight solve that would otherwise return arbitrary "converged" weights — even when the treated unit differs, since donor distinguishability, not treated-vs-donor variation, identifies `W`): the headline fit raises `ValueError`, in-space placebo refits whose donor pool is indistinguishable in a window are dropped from the reference set, and such in-time-truncated dates are marked `infeasible`. Abadie 2021 footnote 7's CV non-uniqueness is handled by a **deterministic tie-break** (prefer the `V` closest to uniform among ties), making the selected `V*` among equally-good optima independent of the multistart evaluation order. The cv fit is reproducible for a fixed `seed` (like `nested`) but is not seed-independent — the multistart fills any slots beyond the distinct heuristic starts with seed-dependent random Dirichlet draws, so the tie-break removes start-order dependence among ties, not seed dependence. The tie-break is convergence-aware (a non-converged optimizer candidate cannot displace a converged incumbent on an objective tie). If the training-window solve that defines `mspe_v` truncates (e.g. `inner_max_iter` too small), the fit fails closed — `mspe_v=NaN` and the fit is marked non-converged — rather than reporting an invalid Eq. 9 criterion. **`v_method="inverse_variance"`** uses the closed form `v_h = 1/Var(X_h)` (variance over donors+treated on the unstandardized predictors), applied to the **raw** predictors so the effective objective is the unit-variance-rescaled `Σ_h diff_h²/Var_h` (Abadie 2021 §3.2(a)); the `standardize` pre-scaling is intentionally bypassed on this branch (inverse-variance weighting *is* the unit-variance rescaling — applying it on already-standardized rows would double-rescale to `Σ_h diff_h²/Var_h²`), so it is equivalent to uniform `V` on standardized predictors. No search (`mspe_v=None`); a zero-variance row gets 0 weight and an all-zero-variance panel falls back to uniform `V` with a warning. `custom_v` is rejected (fail-closed) for both methods and `v_cv_t0` is rejected unless `v_method="cv"`. On the degenerate **single-donor** path (`J=1` forces `w=[1]`) `V` is unidentified — every `V` yields the same synthetic — so `v_weights` is **uniform** and `mspe_v=None` for ALL `v_method`s (cv / inverse_variance included; their selected / closed-form `V` would be inert), with a `UserWarning`; the donor weights / gap / ATT are unaffected. An explicitly pinned `v_cv_t0` that no longer fits the truncated pre-fake window is nulled to the `//2` default for the placebo refit (a pinned value that still fits the truncated window is kept). **Validation:** R `Synth` has no built-in CV function (ADH 2015's CV is a manual `dataprep`+`synth` re-run), so cv is anchored by deterministic equivalence to the R-anchored `custom_v` path (the step-3 validation MSPE of the training-window fit and the step-4 validation-window weights each match a `custom_v=V*` fit on the correspondingly re-aggregated predictors) plus cv self-consistency (`in_time_placebo` under cv == a fresh cv fit on the backdated panel to 1e-7); inverse-variance is anchored bit-for-bit to a `custom_v=1/Var(X)` fit. Documented in `docs/methodology/REGISTRY.md` §SyntheticControl (new `**Note:**` labels for the per-window re-aggregation convention, the flat-MSPE tie-break, and inverse-variance), `docs/api/synthetic_control.rst`, the LLM guides, and `README.md`. The remaining ADH-2015 items (`W^reg` extrapolation diagnostic, sparse-SC subset search) stay tracked in `TODO.md`.
- **Firpo & Possebom (2018) SCM inference paper review on file (PR-A).** Added `docs/methodology/papers/firpo-possebom-2018-review.md`, a faithful, paper-sourced fidelity review of Firpo & Possebom (2018, *Journal of Causal Inference* 6(2), DOI 10.1515/jci-2016-0026) — the Step-1 artifact for the forthcoming SCM **confidence-set / CI-by-test-inversion** track (PR-B) layered on the existing `SyntheticControl` estimator (classic SCM has no analytical SE; `se`/`p_value`/`conf_int` are NaN). Transcribes (paper-sourced only, no code-deviation verdicts) the benchmark RMSPE-ratio permutation test (Eqs. 4–6), the sensitivity-analysis parametric p-value weights with worst/best-case `φ̲`/`φ̄` (Eqs. 7–9), the sharp-null `RMSPE^f` test (Eqs. 10–13), the **confidence sets by test inversion** (Eq. 14) with the operational constant-effect CI (Eqs. 15–16) and linear-effect CS (Eqs. 17–18), the general test-statistic framework + Monte Carlo size/power of five statistics (Eq. 19, Section 5), and the multiple-outcome FWER (Eqs. 23–24) and multiple-treated-unit pooled (Eqs. 25–26) extensions; the requirements checklist flags the PR-B target (sharp-null test + constant/linear CI + benchmark + one-sided) versus the deferred sensitivity-analysis and multi-outcome/treated extensions. Docs-only; no code change. Registered in `docs/references.rst` (Synthetic Control Method section) and `docs/doc-deps.yaml`; REGISTRY `## SyntheticControl` gains a `firpo-possebom-2018-review.md` reviews-on-file pointer.
- **`HeterogeneousAdoptionDiD.fit()` fit-time extensive-margin warning + `covariates=` not-implemented pointer.** Two UX additions to the HAD `fit()` surface, with **no change to any estimate or standard error**. (1) The **overall** path now emits a `UserWarning` when a non-trivial fraction (`>= 10%`, a library-convention cutoff in `_HAD_EXTENSIVE_MARGIN_ZERO_DOSE_FRAC`) of units have an exactly-zero post-period dose — a genuine untreated mass for which a standard DiD using those units as controls may be more appropriate (de Chaisemartin et al. 2026, Section 2 / Assumption 3). The paper retains *small* untreated shares (e.g. 12/2954 in Garrett et al., with close-to-nominal coverage), so the 10% cutoff sits ~25× above that; the warning is **overall-path-only** because the event-study path *requires* never-treated units per Appendix B.2. Previously the recommendation surfaced only via `qug_test()`'s zero-dose warning when the user ran the pre-tests. (2) `HeterogeneousAdoptionDiD.fit(covariates=...)` now raises `NotImplementedError` with a pointer to the deferred Appendix B.1 / Theorem 6 covariate-adjusted extension (via an explicit keyword-only `covariates=` param) instead of a bare `TypeError` from an unknown kwarg; pre-residualize the outcome on the covariates as a workaround. Documented in `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD; new tests in `tests/test_had.py` and `tests/test_methodology_had.py`.

### Fixed
- **Covariate names that collide with reserved structural terms now raise `ValueError` instead of silently corrupting the coefficient dict (`DifferenceInDifferences`, `MultiPeriodDiD`, `TwoWayFixedEffects`).** These estimators build their `coefficients` dict by zipping a variable-name list -- structural term names PLUS the user covariate column names appended verbatim -- with the fitted coefficient vector. A covariate whose name equaled a reserved structural name (`const`; the treatment/time column names; the `{treatment}:{time}` interaction; MultiPeriodDiD `period_{p}` dummies and `{treatment}:period_{p}` interactions; `TwoWayFixedEffects` `ATT`; fixed-effect / unit / time dummy names; or an internal `_`-prefixed working column such as `_treat_time` / `_did_treatment` / `_treatment_post`) silently **overwrote** that structural coefficient via Python dict last-write-wins -- e.g. a covariate named `const` dropped the intercept -- with no error or warning. A new shared `validate_covariate_names` helper (`diff_diff/utils.py`) is now called in each of the three `fit()` methods before the design matrix is built; it raises `ValueError` on a collision (the comparison is case-sensitive, so e.g. `Const` is still allowed) **and** on duplicate names within `covariates` (which collapse to a single dict entry the same way). Fixed-effect/unit/time dummy reserved names are taken from the same `pd.get_dummies(..., drop_first=True)` call used to build them, so they match exactly (including for pandas `Categorical` columns with a non-default category order). For `TwoWayFixedEffects` the guard fires on **all** variance paths: the default within-transform path returns only `{"ATT": att}` (no covariate is a dict key there), but a covariate named `_treatment_post` would still clobber the internal interaction column, so guarding both paths is uniform and forward-compatible. **Potentially breaking:** a fit that previously *succeeded* with a colliding (or duplicated) covariate name -- silently returning a corrupted coefficient dict -- now raises; rename the covariate column(s). The staggered / influence-function estimators (CallawaySantAnna, SunAbraham, StaggeredTripleDifference, EfficientDiD, TwoStageDiD, ImputationDiD, WooldridgeDiD, dCDH, StackedDiD) key results by `(g, t)` tuples / relative-time indices, never covariate names, and `TripleDifference` / `SyntheticControl` / `SyntheticDiD` do not expose covariates by name, so none are affected. New tests in `tests/test_utils.py`, `tests/test_estimators.py`, and `tests/test_estimators_vcov_type.py`.

### Changed
- **CI + local AI PR-reviewer model upgraded `gpt-5.4` → `gpt-5.5`.** The CI Codex reviewer (`.github/workflows/ai_pr_review.yml`) and the local `/ai-review-local` default (`.claude/scripts/openai_review.py` `DEFAULT_MODEL`) now run `gpt-5.5` @ `xhigh` effort / `read-only` sandbox (all other invocation settings unchanged). Validated empirically before the swap via the `tools/reviewer-eval/` A/B harness: on a real-bug corpus plus a k=6 big-diff de-risk, `gpt-5.5` matched-or-beat `gpt-5.4` on every test-backed recall case (including a bug buried in a ~3k-line methodology diff), added zero false positives, and ran faster; an end-to-end CI canary confirmed the action environment (`openai/codex-action@v1`, codex CLI 0.135.0) runs `gpt-5.5` and catches a planted P0. `gpt-5.5` is also added to the reasoning-model set (`_is_reasoning_model`, for the api-backend timeout/token limits) and to the `PRICING` table at OpenAI's confirmed standard rates (`gpt-5.5` $5/$30, `gpt-5.5-pro` $30/$180 per 1M input/output tokens) — the production CI + local reviewer run `gpt-5.5` via the flat-rate **codex backend**, but `--backend auto` falls back to the metered API path when the codex CLI is unavailable, so the cost estimate must stay accurate there. `gpt-5.4` remains accepted.
- **EfficientDiD methodology-review-tracker promotion: In Progress → Complete, with a covariate outcome-regression upgrade (behavior change).** Completes the source-validation pass (PR-B) of the Chen, Sant'Anna & Xie (2025, arXiv:2506.17729v1) audit — PR-A (#515) added the paper review on file; this PR validates the source against the code, eliminates the one real deviation, adds paper-equation Verified Components, and flips the tracker. **Behavior change:** the covariate doubly-robust path's outcome regression `m̂(X)` was a **linear OLS** working model — consistent (doubly robust) but attaining the semiparametric efficiency bound only when the conditional mean is linear in the covariates. It is replaced by a **polynomial sieve** (total degree up to K, AIC/BIC order selection, the same basis family as the propensity-ratio sieve), so with the sieve propensity ratio and the kernel-smoothed conditional `Ω*(X)` all nuisances are estimated nonparametrically and the covariate path attains the bound under the paper's regularity conditions (Section 4 / Theorem 4.1). The order is chosen by an OLS information criterion `IC = n·ln(RSS/n) + c_n·p_K`, where `p_K = comb(K+d, d)` is the sieve basis dimension (number of fitted coefficients; `c_n = 2` AIC, `ln(n)` BIC), on the within-group (survey-weighted) residual sum of squares, using the within-group **positive-weight support** count for both `n` and the penalty (the raw row count when unweighted) so the selected order — and hence `m̂` — is invariant both to the survey-weight scale and to zero-weight (survey-subpopulation / padded) rows. The weighted RSS / Gram / loss totals are already inert to zero-weight rows; keying order selection (auto-k_max, the `n_basis` admissibility cap, and the IC sample-size terms) off the positive-weight support — in the outcome regression **and** both propensity sieves — keeps selection inert too. The auto Silverman bandwidth for the kernel-smoothed conditional `Ω*(X)` is likewise evaluated on the positive-weight support, so the overidentified (`H > 1`) DR path — where `Ω*(X)` feeds the per-unit efficient weights — is invariant as well (otherwise a zero-weight row with an extreme covariate would inflate the unweighted std and move the bandwidth). So a zero-weight-padded survey fit returns a bit-identical point estimate to the positive-weight-only fit (verified end-to-end in `tests/test_methodology_efficient_did.py::TestSurveyZeroWeightInvariance`, both a just-identified `H=1` case pinning sieve-order selection and an overidentified `H>1` case pinning the kernel bandwidth, plus a helper-level auto-order regression test; each fails under the respective raw-count / all-row selector; the existing `test_survey_phase3.py` scale-invariance asserts still hold to `atol=1e-8`). **Degree 1 reproduces the prior linear OLS up to floating point**, so AIC/BIC degrades to linear when the conditional mean is linear and covariate-fit numbers change only when a higher order is selected (i.e. when linear was inadequate); `sieve_k_max=1` forces every covariate-path sieve to degree 1 (it recovers the linear outcome-regression component but also degree-1-constrains the propensity sieves, so it does **not** reproduce the exact pre-PR estimator). The sieve is a *growing* sieve — the candidate degree is `floor(n_pos^{1/5})` (the positive-weight support `n_pos`, the raw group size when unweighted) with **no fixed ceiling**, giving a basis dimension `p_K = comb(K+d,d)` bounded by `n_basis < n_pos` (so `p_K/n → 0` for the low-dimensional covariate settings typical of DiD; Assumption C.1's rate is on the dimension, not the degree). This satisfies C.1's growing-sieve uniform-consistency / `o_p(n^{-1/2})` product-rate conditions (Theorem 4.1) under which the bound is attained asymptotically; a frozen finite-order sieve would not. (High-dimensional `X` faces the usual curse of dimensionality, where the paper's ML-nuisance option applies.) This also removes the prior hard `K≤5` cap from the two pre-existing propensity-ratio / inverse-propensity sieves (a no-op for groups under ~3,125 units, where `floor(n^{1/5}) < 5` anyway; it only activates higher orders at large n). The small-group overfit cap (`n_basis < n_group`), the rank-guard + partial-skip warnings, and the WLS survey path mirror the propensity sieve; if every degree is rank-skipped the estimator falls back to the intercept-only within-group mean (distinct from the propensity sieve's constant-ratio-1 fallback). The no-covariate path, weights, generated outcomes, `Ω*`, SE, aggregation, and Hausman are **unchanged** — the audit verified them correct against the paper (no other corrections). The Theorem A.1 Hausman statistic computation was extracted into a behavior-preserving `_hausman_quadratic_form` helper for unit-testability. New `tests/test_methodology_efficient_did.py` with paper-equation-numbered Verified Components (Eq 3.5/3.13 inverse-covariance weights + the min-variance property; Eq 3.9 generated-outcome telescoping; Eq 3.13/§4.1 no-covariate closed form; Corollary 3.1/3.2 PT-Post = Callaway-Sant'Anna; Theorem 4.1 SE = `sqrt(mean(EIF²)/n)`; Theorem A.1 / Eq A.2 Hausman with the restricted−efficient covariance direction, the effective-rank DOF safeguard on a rank-deficient `V`, and the covariance-direction guard; plus the sieve nonlinear-recovery / linear-degradation / efficiency-gain checks). The HRS Table 6 anchor (`tests/test_efficient_did_validation.py::TestHRSReplication`, a derived openICPSR 116186 subset) is tightened from 0.1·SE to **0.05·SE** (the fit is deterministic; all cells are < 0.03·SE), with the data license/redistribution and the 656-vs-652 sample difference documented in `tests/data/README.md`. REGISTRY `## EfficientDiD` Notes updated (outcome regression now sieve + bound-attainment under Assumption C.1; new K=1-fallback edge-case Note); module/class docstrings and the paper review's "open working-model choice" pointer reconciled; `METHODOLOGY_REVIEW.md` row promoted to **Complete** (`Last Review = 2026-06-01`) with a Verified Components / Corrections Made / Deviations detail block; priority queue pruned.

## [3.5.0] - 2026-06-01

### Added
- **EfficientDiD methodology paper review on file (PR-A).** Added `docs/methodology/papers/chen-santanna-xie-2025-review.md`, a faithful, paper-sourced fidelity review of Chen, Sant'Anna & Xie (2025, arXiv:2506.17729v1) — the Step-1 artifact of the `EfficientDiD` methodology-review validation. Transcribes (paper-sourced only, no code-deviation verdicts) the identifying assumptions (S/O/NA/PT-Post/PT-All), the single-treatment-date EIF (Theorem 3.1, Eqs 3.2–3.6; Corollary 3.1) and the staggered analog (Theorem 3.2, Eqs 3.9–3.14; Corollary 3.2), the no-covariate closed form and the covariate sieve/kernel doubly-robust estimation path of §4 (Eqs 4.1–4.5), the efficiency + standard-error results (Theorem 4.1; SE = `sqrt(mean(EIF²)/n)` or multiplier bootstrap), the Hausman PT-All-vs-PT-Post pretest (Appendix A, Theorem A.1), and the HRS Table 6 / ARE cross-language anchor for the follow-up validation pass. Docs-only; no code change. `docs/references.rst` updated from "Working Paper" to the arXiv URL; REGISTRY `## EfficientDiD` gains a `Paper review on file:` pointer; the review is registered in `docs/doc-deps.yaml`. The `METHODOLOGY_REVIEW.md` row stays **In Progress** — the source-validation pass (PR-B) flips it.
- **PowerAnalysis methodology-review-tracker promotion: In Progress → Complete, with a panel-variance correction (behavior change).** Closes the Bloom (1995) + Burlig, Preonas & Woerman (2020) source audits on the tracker (PR-A #506 added both paper reviews + under-review Notes; this PR validates the source against the code and reconciles the discrepancies). **Behavior change:** the analytical *panel* DiD variance was the Moulton design-effect factor `(1+(T−1)·rho)/T`, wrong two ways versus the source — wrong period-scaling (~4× too small at `rho=0`, `m=r=5` versus the iid DiD benchmark) and the **opposite `rho`-sign** (it *raised* the MDE as within-unit correlation grew). It is replaced by the within-unit equicorrelated special case of Burlig et al. Eq. 2, `Var(ATT) = sigma² · (1/n_T + 1/n_C) · (1/n_pre + 1/n_post) · (1 − rho)`, in which within-unit (serial) correlation *lowers* the MDE because the difference-in-differences cancels the shared within-unit component. So `PowerAnalysis.mde` / `power` / `sample_size` (and the `compute_*` wrappers) now return a **smaller** MDE / required N as `rho` rises for **all** designs; the 2×2 path matches Bloom's `2σ²` at the default `rho = 0` and is continuous with the panel form at `n_pre = n_post = 1`. New input validation, enforced for **all** designs *before* the 2×2-vs-panel router: `n_pre >= 1`, `n_post >= 1`, `rho ∈ [−1/(T−1), 1)` (`T = n_pre + n_post`), finite `sigma >= 0`, positive group counts, and `treat_frac ∈ (0, 1)` now raise `ValueError` (previously invalid two-period shapes and out-of-range `rho` fell through to `basic_did` silently). The `(1 − rho)` factor applies at `T = 2` too — the 2×2 path is Burlig's `m = r = 1` special case (footnote 11), so a nonzero `rho` is no longer silently ignored there, while `rho = 0` still recovers Bloom's `2σ²`. The MDE multiplier stays the **normal (z)** Bloom multiplier (a deliberate large-sample approximation to Burlig's t, documented as `**Deviation from R:**`) — unchanged. New `tests/test_methodology_power.py` (Bloom Table 1 multipliers; 2×2 + panel closed forms; a literal-equicorrelated Monte-Carlo validation of the panel variance; `sample_size`↔`mde` round-trip; input-guard + `rho`-at-`T=2` + `compute_*` wrapper validation; base-R `qnorm` parity at `benchmarks/data/r_power_golden.json`, generator `benchmarks/R/generate_power_golden.R`); the two `tests/test_power.py` ICC-direction tests were inverted to Burlig's sign. REGISTRY `## PowerAnalysis` equation block rewritten (z not t; corrected 2×2 / panel SE + sample-size; removed the cluster-`m` and inverted-`R²` terms that matched neither code nor source); `docs/references.rst` adds Frison & Pocock (1992) + McKenzie (2012) as the equicorrelated lineage; tutorial `06_power_analysis.ipynb` corrected. `METHODOLOGY_REVIEW.md` row promoted to **Complete** (`Last Review = 2026-05-31`); priority queue pruned; the PR-A under-review Notes removed across REGISTRY / `power.py` / `references.rst`.
- **`WooldridgeDiD` outcome-fit hint:** `WooldridgeDiD(method="ols")` now emits a `UserWarning` when the outcome is binary (`{0, 1}`) or a non-negative integer count, noting that a matching nonlinear model (`method="logit"` / `method="poisson"`) is often the **more appropriate specification** for such outcomes. Following Wooldridge (2023): the nonlinear paths impose parallel trends on the link/index scale rather than in levels (level-PT is only valid for continuous/unbounded outcomes), and the paper's Section 5 simulations show the linear model both biased and less precise where the nonlinear mean holds. It is a **different identifying assumption** than linear OLS — which one fits depends on which parallel-trends restriction holds — so the warning frames it as a recommended comparison, not an automatic switch or free efficiency upgrade. OLS remains a valid QMLE for *any* response (Table 1). Always-on (suppress via `warnings.filterwarnings`); detection is high-signal (binary requires exactly `{0, 1}`; the count branch suggests Poisson — the natural unbounded-count model — for *any* non-negative integers with >2 distinct values, so bounded binomial / known-upper-bound integer outcomes are not separately distinguished from unbounded counts; fractional / continuous outcomes are not flagged).
- **`SyntheticControl` leave-one-out + in-time placebo robustness diagnostics (ADH 2015 §4).** Two opt-in `SyntheticControlResults` methods, each a thin re-run of the validated solver (analytical `se`/`t_stat`/`p_value`/`conf_int`/`is_significant` stay bound to the NaN analytical `p_value`). **`leave_one_out()`** drops each reportably-weighted donor (weight above the 1e-6 floor — the donors in `donor_weights`) in turn and re-fits the treated unit against the reduced pool, returning a per-drop ATT / `delta_att` table (a `status="baseline"` row first, then one row per dropped donor sorted by `|delta_att|`; non-converged refits → `status="failed"` with NaN metrics); a large `delta_att` flags single-donor dependence (a single *dominant* donor is still dropped — the others absorb its mass — and its large `delta_att` is the intended signal). **`in_time_placebo()`** reassigns the intervention to an earlier pre-date `t_f`, re-fits using only pre-`t_f` information, and reports the placebo "effect" over the held-out window `[t_f, T0)` — ~0 when there is no real pre-period effect (ADH 2015 Fig. 4). It sweeps every feasible interior pre-date by default (≥2 pre-fake + ≥1 post-fake); an explicit post-period / non-pre date raises, a dimensionally-infeasible valid date yields a `status="infeasible"` row. **Windowing = TRUNCATE** (documented `**Note:**` in REGISTRY): predictor specs are re-cut to the pre-`t_f` window (pre-period-outcome predictors become the pre-`t_f` outcomes; covariate/special windows are intersected), a window lying entirely in the held-out region is **dropped** (surfaced in `n_dropped_specs` + an aggregated warning) and `custom_v` is subset in lockstep with the surviving specs; the true post-treatment periods are excluded from the placebo fit entirely (no peeking). Both fail closed on a non-converged treated fit (and `leave_one_out` on `<2` donors). New accessors `get_leave_one_out_df()` / `get_in_time_placebo_df()` (survive pickling) and long-form `get_leave_one_out_gaps()` / `get_in_time_placebo_gaps()` for the overlay/backdating plots (panel-derived, dropped on pickle). **Validation:** R `Synth` has no in-time/LOO function (verified against its full CRAN function index), so — beyond the solver's existing Basque R parity — the diagnostics are anchored by deterministic self-consistency tests proving each equals a from-scratch `synthetic_control()` fit on the equivalent sub-problem (reduced donor pool / backdated panel) to 1e-7. **Reporting-stack integration:** `_scm_native` surfaces opt-in `leave_one_out` + `in_time_placebo` blocks (`status="not_run"` stub until run), `BusinessReport` lifts them into the SCM native robustness block, and `practitioner_next_steps` emits both as steps (non-`STEPS` tags so a caller's `completed_steps` cannot suppress them). The remaining ADH-2015 items (CV `V`-selection, `W^reg` extrapolation diagnostic, sparse-SC) are tracked in `TODO.md`. Documented in `docs/methodology/REGISTRY.md` §SyntheticControl, `docs/methodology/REPORTING.md`, `docs/api/synthetic_control.rst`, the LLM guides, and `README.md`.
- **New tutorial: `docs/tutorials/24_staggered_vs_collapsed_power.ipynb` — "Staggered Rollout or a Simple 2×2? A Power-Analysis Decision Guide".** A practitioner walkthrough for geo experiments (framed on a 50-state staggered rollout) on when to reach for Callaway-Sant'Anna vs collapsing to a familiar pre/post 2×2. Shows, with live paired Monte Carlo on `generate_staggered_data`, that the collapsed 2×2 silently targets a *diluted* estimand (reports ~60–94% of the true effect-on-treated as the rollout staggers, with near-zero CI coverage of the truth under a slow rollout), and that CS's minimum-detectable-lift penalty is a *fast-rollout* phenomenon that shrinks to parity as the rollout becomes more staggered. Fully self-contained (runs live, no committed data files); ends with a CS-vs-2×2 decision guide.
- **`SyntheticControl` in-space placebo permutation inference + reporting-stack integration (ADH 2010 §2.4).** New `SyntheticControlResults.in_space_placebo()` provides the significance test classic SCM lacks an analytical SE for: it reassigns treatment to each donor, refits a synthetic control for that pseudo-treated donor against the **other `J−1` donors** (the real treated unit is excluded from every placebo pool — its post-period is treatment-contaminated; matches `SCtools::generate.placebos`), and ranks the treated unit's post/pre **RMSPE ratio** among the `J+1` units. New fields `placebo_p_value` (`= rank/(n_placebos+1)`, an upper-tail rank test on the unsigned RMSPE ratio — direction-agnostic, so it detects an effect of *either* sign rather than a signed/one-directional hypothesis; ties counted via `≥`), `rmspe_ratio` (the treated statistic, set at fit), and `n_placebos`/`n_failed` (effective reference-set sizes; non-converged placebos are excluded from BOTH numerator and denominator, never penalized into the rank). `placebo_p_value` is a **separate field** from the (always-NaN) `p_value` — it is a permutation p-value with no SE/t-stat and does not flow through `safe_inference`; `is_significant` stays bound to `p_value`. Edge cases fail closed: scale-aware RMSPE-ratio floor (a perfect pre-fit gives a finite ratio, not `inf`), `J<2` → NaN+warn, `J==2` → degenerate+coarse warn, deterministic given `seed`. New `get_placebo_df()` returns the per-unit RMSPE-ratio summary table (incl. the treated row and any failed donors) used for the rank. The design keeps the placebo *compute* opt-in — the per-donor refit loop runs only on the explicit `in_space_placebo()` call. To support that opt-in call, every fit retains a `_SyntheticControlFitSnapshot` of the pivoted panel (memory O(units × periods × predictor-vars), like `SyntheticDiD`'s snapshot for `in_time_placebo`; excluded from pickling). A compact/lazy snapshot representation is tracked as a follow-up in `TODO.md`. **Reporting-stack integration:** `SyntheticControlResults` is now routed through `DiagnosticReport` (fit-based `scm_fit` parallel-trends analogue → verdict `design_enforced_pt` reading `pre_rmspe`; `_scm_native` surfaces `pre_rmspe` + donor-weight concentration + the placebo p-value when already computed — never triggering the refit loop implicitly), `practitioner_next_steps` (`_handle_synthetic_control` with the placebo as the headline significance step), and `BusinessReport` (fit-based assumption block, ADH 2010 attribution, robustness via `estimator_native_diagnostics`; HonestDiD passthrough rejected like SDiD/TROP). Also fixes a latent BR bug where the headline `is_significant` was a non-JSON-serializable numpy `bool_` when `p_value` is a numpy `NaN`. Documented in `docs/methodology/REGISTRY.md` §SyntheticControl (new `**Note:**` labels for the donor-pool construction, failure handling, RMSPE-ratio floor, and the non-analytical-p-value split), `docs/methodology/REPORTING.md`, `docs/api/synthetic_control.rst`, the LLM guides, and `README.md`.
- **New estimator: `SyntheticControl` — classic Synthetic Control Method (Abadie, Diamond & Hainmueller 2010; Abadie & Gardeazabal 2003).** Standalone estimator (`diff_diff/synthetic_control.py`) + `SyntheticControlResults` (`diff_diff/synthetic_control_results.py`) + `synthetic_control()` convenience function, exported from `diff_diff`. Builds a single treated unit's counterfactual as a convex combination of never-treated donor units — **donor (unit) weights only**, no time weights or ridge, distinct from `SyntheticDiD`. The inner simplex-constrained weighted-LS solve `W*(V)` reuses `utils._sc_weight_fw` (folding `V^½` into the predictor matrix, `intercept=False`, `zeta=0`); the diagonal predictor-importance matrix `V` is selected data-driven by minimizing pre-period outcome MSPE (`v_method="nested"`, softmax-on-simplex multistart Nelder-Mead + Powell polish) or supplied by the user (`v_method="custom"`). Predictors are built from `predictors`/`predictor_window`/`predictors_op`, `special_predictors`, and per-period outcome lags (`pre_period_outcomes`), in the R `Synth::dataprep` row order; per-row standardization (SD over donors+treated, ddof=1) matches the R `Synth::synth` source. Reports the gap path (`α̂_1t = Y_1t − Σ_j w_j Y_jt`), `att` (mean post-period gap), `pre_rmspe`, donor weights, `v_weights`, and a predictor-balance table. **No analytical standard error** — `se`/`t_stat`/`p_value`/`conf_int` are NaN; significance comes from in-space placebo permutation inference via `in_space_placebo()` (see the dedicated entry below). Ten validation gates baked in: predictor-period leakage, absorbing post-period suffix + no-anticipation cross-check against the treatment column, post-period canonicalization, donor-pool filtering before period derivation, empty-window rejection, poor-pre-fit `UserWarning` (RMSPE > SD of treated pre-outcomes), duplicate-predictor-label rejection, inner-solve non-convergence warning, order-independent gap-path rebuild, and the `standardize="none"` deviation; plus fail-closed `custom_v` cross-field rules and degenerate single-donor / single-pre-period handling. **R-`Synth` parity** (`tests/test_methodology_synthetic_control.py`, fixtures generated by `benchmarks/R/generate_synth_basque_golden.R` into `tests/data/`): two-tier on the Basque Country study — Tier-1 feeds R's `solution.v` via `custom_v` and reproduces the published donor weights (region 10 Cataluña 0.851 + region 14 Madrid 0.149) to `atol=1e-3` deterministically; Tier-2 (`@pytest.mark.slow`) checks the data-driven nested fit lands in a tolerance band (the nested `V` legitimately differs because the outer objective uses all pre periods, not R's `time.optimize.ssr` window). Documented in `docs/methodology/REGISTRY.md` §SyntheticControl (with `**Deviation from R:** standardize="none"` and `**Note:**` labels for the standardization formula, objective window, softmax `V` parametrization, and 1×SD poor-fit threshold), `docs/api/synthetic_control.rst`, the LLM guides, and `README.md`.
- **StaggeredTripleDifference methodology-review-tracker promotion: In Progress → Complete**, plus a new opt-in Eq-4.14 overall ATT. Closes the Ortiz-Villavicencio & Sant'Anna (2025, arXiv:2505.09942v3) primary-source review on the tracker (PR-A #499 added the paper review on file; this PR validates the source against it). New paper-equation-anchored Verified Components in `tests/test_methodology_staggered_triple_diff.py` (Theorem 4.1 / Eq. 4.5 RA=IPW=DR identification; Eq. 4.1 three-term DDD decomposition; Eqs. 4.11-4.12 optimal-GMM weight normalization + single-group reduction; Eq. 4.13 event-study cohort-share weighting; Eq. 4.14 / Cor. 4.2 overall) alongside the existing R cross-validation against `triplediff::ddd(panel=TRUE)` + `agg_ddd()`. **New feature — opt-in `overall_att_es` (paper Eq. 4.14 overall):** the unweighted mean of the post-treatment event-study effects ES(e), exposed on `StaggeredTripleDiffResults` (with `overall_se_es` / `overall_t_stat_es` / `overall_p_value_es` / `overall_conf_int_es`) and populated only when `aggregate="event_study"` / `"all"`. The default `overall_att` is unchanged (the Callaway-Sant'Anna simple post-treatment (g,t) average — the library-wide convention). Its analytical SE is the influence function of that mean (the average of the per-event-time combined IFs, routed through the same survey-aware variance estimator as the per-e effects via a new `_se_from_psi` helper); a multiplier-bootstrap SE replaces it under `n_bootstrap>0`. Computed via a side-channel stash on the shared `CallawaySantAnnaAggregationMixin._aggregate_event_study` (no return-signature change; CallawaySantAnna unaffected), over post-treatment `e >= -anticipation` (the library convention, matching `overall_att`). Cross-validated against R `agg_ddd(type="eventstudy")$overall.att` / `overall.se` (SE matches to ~0.1%). REGISTRY `## StaggeredTripleDifference`: the previously-unlabeled overall-aggregation prose is formalized under a `**Note:**` documenting both overalls, and the duplicate aggregation-weight deviation is consolidated (fixing a `P(G=g)` vs R `P(S=g)` mislabel). `METHODOLOGY_REVIEW.md` row L69 promoted to **Complete** (`Last Review = 2026-05-30`) with a Verified Components / R Comparison Results detail block; priority queue pruned. `docs/references.rst` Ortiz-Villavicencio entry pinned to arXiv:2505.09942v3.
- **SunAbraham + WooldridgeDiD-OLS `vcov_type="conley"` (Conley 1999 spatial-HAC) threading.** Both estimators now accept `vcov_type="conley"` with the five `conley_*` constructor params (`conley_coords`, `conley_cutoff_km`, `conley_metric`, `conley_kernel`, `conley_lag_cutoff`), reusing the already-`conleyreg`-validated `solve_ols` / `conley.py` machinery — within-period spatial HAC at `conley_lag_cutoff=0`, plus the within-unit Bartlett serial term at `conley_lag_cutoff>0` (the panel-aware path, since `conley_time`/`conley_unit` are always supplied — not pooled cross-sectional), no new variance code. Conley routes through each estimator's within-transform path; the unit auto-cluster is dropped on the conley path (an explicit `cluster=` enables the spatial+cluster product kernel); `survey_design=` / `weights` / `n_bootstrap>0` are rejected, and WooldridgeDiD conley is OLS-path-only (`method ∈ {logit, poisson}` + conley still rejected via the `method != "ols"` guard). `SunAbrahamResults` / `WooldridgeDiDResults` gain a `conley_lag_cutoff` field plus a Conley variance-label line in `summary()` (`SunAbrahamResults` also gains `cluster_name`). FWL-composability — the within-transform conley SE equals the full-dummy conley SE — is pinned in `tests/test_conley_vcov.py` (`TestConleySunAbraham` / `TestConleyWooldridge`). **`StackedDiD` conley remains deferred for a methodology reason** (the stacked design replicates units across sub-experiments, so Conley would see same-unit copies at distance 0; no `conleyreg` anchor; paper-gated) — its prior "same shape as the SunAbraham follow-up" framing is corrected in REGISTRY / TODO / the rejection message.
- **ConleySpatialHAC methodology-review-tracker promotion: In Progress → Complete.** Closes the Conley (1999) *Journal of Econometrics* 92(1) primary-source review on the methodology-review tracker. The paper review on file at `docs/methodology/papers/conley-1999-review.md` was previously merged (2026-05-09); this PR is the F.L.I.P. consolidation — new `tests/test_methodology_conley.py` with paper-equation-numbered Verified Components walk-through (~1600 LoC; 10 classes; 60 tests, 5 of them `@pytest.mark.slow`). Coverage: Eq. 4.2 cross-sectional sandwich (pairwise-distance specialization; the project's paper review identifies Eq. 4.2 page 18 as the real-valued/pairwise form, with Eq. 3.13 reserved for the lattice-indexed form), Eq. 4.2 HC0 + rank-1 limits, Andrews (1991) HAC lag truncation matching `conleyreg::time_dist.cpp`, haversine convention with Earth radius 6371.01 km, Phase 2 panel block-decomposed sandwich at `atol=1e-12`, sparse k-d-tree dense-vs-sparse bit-identity (Wave A #120 numerical correctness), and R `conleyreg` v0.1.9 parity at `atol=1e-6` on 6 fixtures (3 cross-sectional + 3 panel) plus the sparse-forced and time-asymmetric kernel parity contracts. Three dedicated deviations-area classes: `TestConleyLibraryExtensions` (Wave A library extensions — combined spatial+cluster product kernel #119, callable conley_metric validation #123, sparse k-d-tree activation #120, indefiniteness guard), `TestConleyDeviationsFromR` (1-D radial Bartlett vs paper's 2-D separable Eq. 3.14, time-label normalization via `np.unique`, independent temporal kernel deferred), and `TestConleyDeferrals` (5 fail-closed `NotImplementedError`/`TypeError` contracts: LinearRegression + survey_design, DiD/MPD/TWFE + survey_design, Conley + weights, SyntheticDiD + Conley, wild_bootstrap + Conley). Methodology-anchored tests extracted from `tests/test_conley_vcov.py`: full classes `TestConleyDirectHelper`, `TestConleyReductions`, `TestConleyReductionsAddendum`, `TestConleyParityR`, `TestConleyParitySpacetime`, `TestConleyPanelHelper`, `TestConleySparseRParityForced`; plus methodology-anchored tests from `TestConleyKernels`, `TestConleyDistanceMetrics`, `TestConleySparse`. File drops 4248 → 3113 lines after extraction. Defensive surface preserved: input validation, NaN/inf guards, dispatch-level validity, estimator-level integration smoke tests, set_params atomicity, sparse-path activation thresholds + density-gate fallback. `METHODOLOGY_REVIEW.md` row L91 promoted to **Complete** with `Last Review = 2026-05-26`; detail block rewritten with Verified Components / Test Coverage / R Comparison Results inline table / Corrections Made / Deviations / Outstanding Concerns. Priority queue at L1386 pruned: PreTrendsPower removed (already Complete since 2026-05-19) and ConleySpatialHAC removed (this PR); substantive-review-blocked renumbered #2-#5 → #1-#4 and consolidation-pass-blocked renumbered #6-#8 → #5-#6.

### Added / Changed
- **EfficientDiD `vcov_type` threading + Results metadata harmonization (Phase 1b interstitial #4, permanently narrow).** `EfficientDiD(vcov_type=...)` now accepts `{"hc1"}` only (default). Analytical-sandwich families `{classical, hc2, hc2_bm}` and `conley` are REJECTED at `__init__` / `set_params` with methodology-rooted messages — EfficientDiD uses influence-function-based variance per Chen-Sant'Anna-Xie (2025) achieving the semiparametric efficiency bound; the per-unit EIF aggregation has no single design matrix on which hat-matrix leverage or Bell-McCaffrey Satterthwaite DOF can be defined. `cluster=` (Liang-Zeger CR1 on cluster-aggregated EIF) and `survey_design=` (TSL on combined IF) paths are unchanged. **BC break on `EfficientDiDResults`:** the `cluster` field renamed to `cluster_name`; new `n_clusters` + `vcov_type` fields added; `to_dict()` method added (mirrors TripleDifferenceResults). `DiagnosticReport._pt_hausman` updated to read the renamed `cluster_name` field for the Hausman pretest replay (`diff_diff/diagnostic_report.py:2444`). `EfficientDiD.set_params(vcov_type=bad)` raises immediately rather than deferring to `fit()` — intentional eager-validation pattern matching EfficientDiD's existing handling of `pt_assumption`/`control_group` etc, diverging from `ImputationDiD`/`TripleDifference`/`CallawaySantAnna` (which use sklearn mutate-then-validate-at-use). Survey-PSU bootstrap path returns NaN SE when fewer than 2 independent PSUs are available (was ≈0 SE from BLAS roundoff). New summary block: `Variance estimator: <label>` line rendered after the survey block when not under bootstrap; suppressed under bootstrap (replaced with `Inference method: bootstrap` + `Bootstrap replications: <n>`). Default `cluster=None` (no survey) renders "HC1 heteroskedasticity-robust" — methodologically correct because the per-unit EIF SE `sqrt(mean(EIF²)/n)` is HC1-style (no Liang-Zeger G/(G-1) finite-sample correction); diverges from `ImputationDiD` which auto-clusters at unit per BJS Theorem 3.
- **TwoStageDiD `vcov_type` threading + Results metadata (Phase 1b interstitial #5, final, permanently narrow).** `TwoStageDiD(vcov_type=...)` now accepts `{"hc1"}` only (default), completing the Phase 1b initiative across all 8 standalone estimators. Analytical-sandwich families `{classical, hc2, hc2_bm}` and `conley` are REJECTED at `__init__` / `fit()` with methodology-rooted messages: TwoStageDiD's variance is the Gardner (2022) two-stage GMM cluster-sandwich whose meat is the per-cluster GMM-corrected score `S_g = gamma_hat' c_g - X'_{2g} eps_{2g}`, which folds first-stage FE estimation uncertainty into the score — there is no single hat matrix spanning both stages on which HC2 leverage or Bell-McCaffrey Satterthwaite DOF can be defined, and the Gardner correction has not been derived for the leverage-corrected/homoskedastic meat (no reference implementation; mirrors the SpilloverDiD `classical` rejection). `cluster=` and `survey_design=` paths are numerically unchanged (bit-identical for healthy fits). **`TwoStageDiDResults` additions (no rename, no BC break):** new `vcov_type` / `cluster_name` / `n_clusters` fields + `to_dict()` method. `summary()` renders a `Variance estimator: <label>` line after the survey block (suppressed under bootstrap — `Inference method: bootstrap` + `Bootstrap replications: <n>` shown instead — and under any survey design). Default `cluster=None` renders `"CR1 cluster-robust at <unit>, G=<n_units>"` because the Gardner sandwich auto-clusters at the unit column (did2s no-FSA convention — the `CR1` label carries no `(n-1)/(n-p)` factor, matching R `did2s`; same convention as ImputationDiD's Theorem 3 variance). Defensive `n_clusters<2` NaN guard added to the multiplier-bootstrap path (was ≈0 SE from BLAS roundoff) plus a survey-PSU `n_psu<2` parity guard. `cluster=` with a replicate-weight survey design now raises `NotImplementedError` (replicate-refit variance ignores `cluster=`). `vcov_type='conley'` deferred to a TODO follow-up row.

### Fixed
- **Scale-invariant rank detection and least-squares solve in the shared OLS backend (`diff_diff/linalg.py` + `rust/src/linalg.rs`).** `_detect_rank_deficiency` ran pivoted QR on the raw design matrix with a rank threshold anchored to the largest pivot diagonal, so a covariate on a large raw scale (e.g. population, income in cents, market cap) inflated the threshold and **false-dropped the intercept/treatment/interaction columns to NaN on an otherwise full-rank model** — a `DifferenceInDifferences` fit with a covariate ×1 or ×1e4 returned the correct ATT while the same covariate ×1e8 returned `ATT=NaN`. Even after detection, the `scipy.lstsq(cond=1e-7)` solve (and the Rust SVD truncation) truncated the genuine small-scale direction relative to the huge column, returning finite-but-wrong coefficients. Detection now runs a raw pivoted QR first and only re-checks on column-equilibrated (unit 2-norm) columns when the raw pass reports a deficiency, adopting the higher equilibrated rank only when the raw drop was scale-induced; the least-squares solve equilibrates columns and unscales the coefficients. This is applied in both the Python and Rust backends, making rank detection and the fit invariant to per-column scaling while leaving everything else unchanged: it is a no-op for full-rank well-conditioned designs (R-parity goldens unaffected) and does **not** change which column is dropped in a *well-scaled* collinear design (the established raw pivot selection is preserved); a scale-induced under-count instead adopts the scale-corrected equilibrated selection (which may differ from the raw choice but retains an identified full-rank subset). Also fixes a cryptic `IndexError: arrays used as indices must be of integer type` when a design collapsed to rank 0 (e.g. a constant FE-collinear covariate in `ImputationDiD`/`TwoStageDiD`): `solve_ols` now returns all-NaN coefficients cleanly, and `solve_poisson` raises a clear `ValueError`. (`solve_logit` always prepends an intercept, so rank 0 is unreachable there — its index array is just made `dtype=int` for consistency.) New regression tests in `tests/test_linalg.py` assert scale-invariance of fitted values/t-stats, a finite ATT through the public DiD estimator with a 1e8-scale covariate, rank-0 NaN handling (OLS) / clear `ValueError` (`solve_poisson`), that the scale repair preserves the raw collinear drop selection for well-scaled genuinely-collinear designs, and that the mixed scale+collinearity case retains an identified full-rank subset (huge independent column kept) with valid inference on the kept coefficients. Both backends verified equivalent (`tests/test_rust_backend.py`). **Scope:** this covers covariate fits routed through `solve_ols` — DiD, TWFE, MultiPeriodDiD, ImputationDiD, TwoStageDiD, and TripleDifference. CallawaySantAnna and StaggeredTripleDifference fit their covariate outcome-regression nuisance via estimator-local `cho_solve` on `X'X` / `scipy.lstsq(cond=1e-7)` that are not yet equilibrated; making those *point-estimate* solves scale-robust is deferred (see TODO.md). The companion DR/OR influence-function SE rank-guard for CS / TripleDifference / StaggeredTripleDifference (local matrix inverses) is now addressed in the dedicated entry below.
- **Rank-guarded doubly-robust / outcome-regression influence-function standard errors (`CallawaySantAnna`, `TripleDifference`, `StaggeredTripleDifference`).** The per-cell propensity-score Hessian and outcome-regression bread (`X'WX`) are inverted for the influence-function SE via `np.linalg.solve`/`inv`, which raise `LinAlgError` only on *exactly* singular matrices. A **near**-singular Gram from a constant or collinear covariate did not raise, so a garbage inverse (entries ~1e13) flowed straight into the SE — reproduced: `CallawaySantAnna(estimation_method="dr")` `overall_se` ~5.1e13, `TripleDifference(estimation_method="reg")` `se` ~1.8e17, `StaggeredTripleDifference` SEs inflated 30–100× — while the **ATT point estimate stayed valid** (only the SE was wrong). A new shared helper `_rank_guarded_inv` (`diff_diff/linalg.py`) symmetrically equilibrates the Gram (`D^{-1/2} A D^{-1/2}`, `D=diag(A)`) and, when rank-deficient, inverts a **column-dropped** principal submatrix — keeping the most-independent columns via pivoted QR on the **equilibrated** Gram (`rcond=1e-10` relative-eigenvalue rank threshold). This is a column-drop generalized inverse in the same *family* as the point estimate / R's `lm()` (drop redundant columns, not a minimum-norm pseudo-inverse), but its column *selection* is scale-invariant and so may drop a different *member* of a collinear set than the point estimate's raw-pivot `_detect_rank_deficiency` under mixed-scale *exact* collinearity — a documented deviation that does not affect the SE (the identified subspace, and hence the variance, is unchanged whichever redundant member is dropped; verified order-invariant for both column orders and under survey weighting). An all-NaN inverse (NaN SE) is returned only on true rank-0. Using a column-drop inverse (rather than a minimum-norm pseudo-inverse, which diverges sharply when the IF multiplier leaves `range(A)` — e.g. a control or treated-sub-cell bread multiplied by a mean from a cell where the covariate is not collinear) makes the analytical SE equal the **well-conditioned near-collinear limit**: replacing the exactly-collinear covariate with a near-collinear (full-rank) one yields the same SE to working precision (verified `se_ratio ≈ 1` across `reg`/`ipw`/`dr`, for every per-cell bread, control and treated). A covariate that is rank-deficient only within one cell still legitimately enters the other cells' full-rank fits, so the ATT/SE reflect that (poor) covariate specification. The well-conditioned fast path returns `np.linalg.solve(A, I)` unchanged (R-parity goldens unaffected). Each estimator emits ONE aggregate `UserWarning` reporting the dropped redundant direction(s) + max condition number (`TripleDifference` gains this consolidated warning, which it previously lacked), suppressed under `rank_deficient_action="silent"`. New regression tests in `tests/test_linalg.py` (helper contract), `tests/test_staggered.py`, `tests/test_methodology_triple_diff.py`, and `tests/test_methodology_staggered_triple_diff.py`. **Scope:** the live bug is the SE rank-guard; the estimator-local *point-estimate* coefficient solves remain scale-equilibration follow-ups (TODO.md) — for an exact duplicate covariate at a very large scale (e.g. `1e8·x`) the un-equilibrated `reg` local OR solve can perturb the point-estimate ATT (the IF SE consistently follows it); a *well-scaled* exact duplicate is exact (SE == dropping it) and `dr`/`ipw` are unaffected. `EfficientDiD` (covariate path) was diagnosed and is already rank-safe (`pinv(rcond=tol/max_eigval)`); `ContinuousDiD`/`SpilloverDiD` have no user-covariate path.
- **Bertanha-Imbens 2014 citation correction (16 sites across 5 files).** A verification spike confirmed the citation across `diff_diff/linalg.py` (×8), `diff_diff/conley.py` (×1), `diff_diff/guides/llms-full.txt` (×2), `docs/methodology/REGISTRY.md` (×4), and `docs/api/spillover.rst` (×1) was incorrect — NBER w20773 *External Validity in Fuzzy Regression Discontinuity Designs* (JBES 2020, 38(3):593-612) by Bertanha & Imbens covers fuzzy RD external validity, NOT weighted spatial-HAC under sampling weights. Replaced across all 16 sites with the open-problem framing: "weighted spatial-HAC under probability sampling is an open methodological question; no canonical extension of Conley (1999) exists for the combination." At the four `REGISTRY.md` sites the replacement is wrapped in the canonical `**Note (open methodological question):**` label per CLAUDE.md "Documenting Deviations (AI Review Compatibility)". REGISTRY ConleySpatialHAC section gains a new `**Note (deferral status, 2026-05-26):**` splitting the boundary into three parts: **Shipped** — SpilloverDiD + Conley + survey via Wave E.1/E.2/E.3 (PR #468/#474/#482), TwoStageDiD + Conley + survey via Wave E.3 parity (PR #485). **Deferred (generic linalg surface, any `weight_type`)** — DiD/MPD/TWFE/LinearRegression generic path + Conley + `survey_design=`; `LinearRegression` / `compute_robust_vcov` Conley + `weights=` rejected for `pweight`, `aweight`, AND `fweight` (weighted Conley is not implemented on the generic linalg surface). **Open methodological question (subset)** — the `pweight` / `survey_design` portion of the deferral additionally lacks a canonical methodological extension of Conley (1999) for weighted spatial-HAC under probability sampling. **No source-code logic changes:** verified via diff-in-diff pytest output before and after the citation strip (175 passed + 14 warnings, bit-identical pass set on `tests/test_conley_vcov.py`). **Historical CHANGELOG entries (pre-[Unreleased]) intentionally retain the Bertanha-Imbens 2014 attribution** as accurate records of what was claimed at the time of each release; the [Unreleased] entry above supersedes those rationales going forward.

## [3.4.2] - 2026-05-25

### Fixed
- **`CallawaySantAnna.cluster=` silent no-op (Phase 1b interstitial).** `CallawaySantAnna(cluster="state").fit(...)` previously accepted the argument, stored it, returned it from `get_params()`, but never consumed it anywhere in the fit / aggregator / bootstrap pipeline (`staggered.py:154-156` docstring claimed "Defaults to unit-level clustering" — but for bare `cluster=X`, the aggregator at `staggered_aggregation.py:193-213` computed per-unit IF variance regardless, and the bootstrap at `staggered_bootstrap.py:323-347` drew per-unit multiplier weights regardless). Users who explicitly set `cluster="state"` got per-unit inference with no warning — typically SE too small under intra-cluster correlation. **Survey-PSU clustering via `survey_design=SurveyDesign(psu="state")` was NOT affected** and continued to cluster correctly via `_compute_stratified_psu_meat`. The fix synthesizes a minimal `SurveyDesign(psu=self.cluster, weight_type="pweight")` when bare `cluster=` is set without an explicit survey design, threading the synthesized PSU through the existing survey-PSU machinery (aggregator + bootstrap). A new dedicated `df_inference` field on `CallawaySantAnnaResults` carries the cluster-level df for the bare-cluster-synthesize path ONLY (where `survey_metadata` is intentionally `None` to preserve the `DiagnosticReport.survey_metadata is not None` skip at `diagnostic_report.py:848-856` + `:1150-1158` for "Original fit used a survey design" reasoning, and the `summary()` survey block render at `staggered_results.py:235-238`). `HonestDiD` at `honest_did.py` prefers `survey_metadata.df_survey` first (the actual CS-internal df, which may be tightened post-resolve for replicate designs) and falls back to `df_inference` for bare-cluster fits — so downstream consumers always see the cluster df without overriding the post-recompute survey df. When `survey_design=SurveyDesign(weights=Y)` without PSU is provided AND `cluster=X` is also set, `_inject_cluster_as_psu` injects the bare cluster as the effective PSU AND an `effective_survey_design = replace(survey_design, psu=self.cluster)` is constructed so the downstream `_validate_unit_constant_survey` catches movers (units crossing clusters across periods) on panel data via the now-PSU-bearing design; `survey_metadata` is recomputed to reflect the injected PSU. When both `cluster=X` AND `survey_design.psu=Y` are set, the explicit PSU wins via `_resolve_effective_cluster` (emits `UserWarning` if partitions differ). **`cluster= + SurveyDesign(replicate_weights=[...])` raises `NotImplementedError`**: replicate-weight variance is computed by replicate reweighting (BRR / Fay / JK1 / JKn / SDR) and ignores PSU/cluster entirely (`survey.py:104-109` enforces replicate_weights are mutually exclusive with strata/psu/fpc); honoring bare `cluster=` would silently have no effect while populating `cluster_name`/`n_clusters` on Results dishonestly. Assertive regression tests pin the fix on both panel and repeated-cross-section paths plus the survey/non-survey contract boundaries: `test_cluster_robust_ses_differ_from_unit_level`, `test_bare_cluster_works_with_panel_false_rcs`, `test_bare_cluster_synthesizes_survey_design`, `test_inject_branch_panel_mover_raises`, `test_replicate_weight_plus_cluster_rejected`, `test_bare_cluster_populates_df_inference` (asserts the dedicated cluster-df carrier is set), `test_bare_cluster_does_not_set_survey_metadata` (asserts the survey/non-survey contract is preserved — DiagnosticReport / summary() must not treat a bare-cluster fit as survey-backed), `test_explicit_survey_design_does_populate_survey_metadata` (asserts the inject-branch path still populates survey_metadata for legitimate user-provided SurveyDesign), and `test_bare_cluster_honest_did_uses_df_inference` (end-to-end: HonestDiD threads df_inference into HonestDiDResults.df_survey, preventing silent normal-theory regression on a future refactor). When `cluster=None` (default), behavior is bit-equal to pre-PR (wiring guarded by `if self.cluster is not None:`). Audit verified the no-op was CS-specific — the other 7 Phase 1b estimators (SunAbraham, StackedDiD, WooldridgeDiD, ImputationDiD, TripleDifference, TwoStageDiD, EfficientDiD) handle bare `cluster=` correctly.

### Added
- **New tutorial: SpilloverDiD on synthetic TVA-style spillover panel (Butts 2021 §4 analogue) with spillover-bandwidth sensitivity and Conley spatial-HAC inference.** `docs/tutorials/23_spillover_tva.ipynb` walks a practitioner through the SpilloverDiD workflow on a 4-period 200-unit panel laid out as treated cluster + near-control band + far-control band. The DGP is tuned at locked seed 23 (`n_treated=25, n_near=120, n_far=55, tau_total=-7.4, delta_1=-4.5, d_bar=100 km`) so naive multi-period TWFE on the full sample understates the direct effect by ~42% — matching the Butts (2021) §4 Table 1 Panel A bias-correction direction documented at `docs/methodology/papers/butts-2021-review.md:257`. SpilloverDiD with `rings=[0.0, 100.0]` cleanly recovers both `tau_total = -7.34` and `delta_1 = -4.53`. The tutorial covers (1) problem framing with TVA / Kline-Moretti (2014) citation, (2) panel construction with the DGP equation inline, (3) naive headline and the bias mechanism, (4) `rings` sensitivity grid at outer edges 50/100/150/200 km (estimates stabilize once `d_bar` covers the true spillover horizon), (5) the headline `SpilloverDiD` fit, (6) Conley spatial-HAC variance via `vcov_type="conley", conley_cutoff_km=100, conley_lag_cutoff∈{0,1}` — the cutoff = `d_bar` choice follows Butts §3.1, while the `conley_lag_cutoff` serial term is the library's documented Wave E.2 follow-up synthesis with Newey-West-style serial Bartlett HAC (per REGISTRY "Variance (Wave E.2 follow-up)") — showcasing the Spillover-Conley work shipped through PRs #468 / #474 / #477 / #482 / #485 / #489. Drift detection in `tests/test_t23_spillover_tva_drift.py` (21 function-level tests, T19 pattern) pins panel composition, geographic bands, true parameters, naive coefficient endpoint (round-to-2 — well-conditioned MultiPeriodDiD fit, BLAS-stable to better than 0.005), recovery endpoints (round-to-2 on `tau_total = -7.34` and `delta_1 = -4.53`), seed-specific geometry numbers (max distance from origin, band diameters, cross-band max pair, within-band median + within-100km pair fractions), sensitivity grid `rings=[0, 50]` endpoint at round-to-1 (per reviewer guidance for BLAS safety on the borderline-rank-deficient point), notebook §2 constant-sync + AST-body sync against the test fixture, exact `tau_total` AND `delta_1` identity across the d_bar ∈ {100, 150, 200} grid (on THIS synthetic DGP only, because no units lie in the 80-200 km band — once `d_bar` covers the true 100 km spillover horizon, the additional ring bins are empty and contribute zero observations; this is NOT a generic Butts §4 result and the registry frames `d_bar` as a real bias/variance tradeoff in the general case), Conley SE divergence from HC1 (direction-pinned: Conley < HC1 on this DGP), and the platform-agnostic post-filter warning surface (T19 pattern: mirrors the notebook's narrow `.*encountered in matmul` `RuntimeWarning` filter inside the capture block and asserts no warnings remain; on Apple Silicon M4 + numpy<2.3 the three known Accelerate BLAS matmul warnings — documented at `TODO.md` "RuntimeWarnings in Linear Algebra Operations" — fire and are filtered, on M3 / Intel / Linux or numpy>=2.3 the filter is a no-op, EITHER WAY any UserWarning / FutureWarning / non-matmul RuntimeWarning surfaces immediately). Per `feedback_t19_drift_guards_test_file_only`, ZERO in-notebook asserts — all numerical guards live in the test file. Per `feedback_notebook_workflow`, the DGP was developed and locked in a temporary `_scratch/` script (gitignored) before being pasted into the notebook §2 cell and duplicated into the drift-test `panel` fixture. `docs/index.rst` toctree updated; `docs/references.rst` gains the Kline-Moretti (2014) entry; `docs/methodology/papers/butts-2021-review.md:257` cross-reference updated from "T22 tutorial" to "T23 tutorial" (slot 22 was occupied by `22_had_survey_design.ipynb`). The `SpilloverDiD T22 TVA tutorial` row in `TODO.md` (renumbered to T23 at delivery) is dropped.
- **TROP methodology-review-tracker promotion: In Progress → Complete.** Closes the Athey, Imbens, Qu & Viviano (2025) *Triply Robust Panel Estimators* (arXiv:2508.21536) primary-source review on the methodology-review tracker. PR-A (the paper review on file at `docs/methodology/papers/athey-2025-review.md`) was previously merged as #443; this PR is the F.L.I.P. consolidation — new `tests/test_methodology_trop.py` with paper-equation-numbered Verified Components walk-through (10 classes, 36 tests covering Eq. 2 soft-threshold SVD prox / plain prox-gradient monotonicity on a toy setup / weighted-prox solver (the shipped accelerated FISTA outer loop is NOT directly tested for per-step monotonicity because Nesterov momentum does not guarantee it), Eq. 3 unit + time weights, Eqs. 4-5 + Algorithm 1 LOOCV with two-stage cycling, Corollary 1 three-condition unbiasedness, Theorem 5.1 MC-ranking realisation of the triply-robust bias bound, Section 2.2 DID + MC reductions, Eq. 13 + Algorithm 2 per-(i, t) estimation, Algorithm 3 stratified pairs bootstrap, Section 3 / Eq. 6 factor-DGP recovery, plus a `TestTROPDeviations` class locking 11 documented library deviations). Migrated from `tests/test_trop.py`: `TestMethodologyVerification` (5 tests → `TestTROPEquation6FactorDGPRecovery`), four paper-conformance tests + one weighted-solver convergence test from `TestPaperConformanceFixes` (→ `TestTROPEquation3Weights` / `TestTROPAlgorithm1LOOCV` / `TestTROPNuclearNormProx` / `TestTROPAlgorithm3Bootstrap`), three prox / plain prox-gradient monotonicity / weighted-objective tests from `TestTROPNuclearNormSolver` (→ `TestTROPNuclearNormProx`), plus a cycling-convergence test from `TestCyclingSearch` and the factor-DGP smoke from `TestTROPvsSDID`; the `TestPaperConformanceFixes` and `TestTROPvsSDID` shells are deleted. `TestTROPNuclearNormSolver` retains its single defensive `test_zero_weights_no_division_error`. `METHODOLOGY_REVIEW.md` TROP row promoted with merge date 2026-05-24, full Verified Components / Test Coverage / Deviations / Outstanding Concerns / R Parity structure mirroring HAD (PR #473) / ContinuousDiD (PR #476) / DCDH (PR #481) / WooldridgeDiD (PR #486) precedents. **Documented deviations:** Gap #5 (unnormalised weights match Eq. 2, not Section 5 sum-to-one), Gap #9 (unbalanced panels supported beyond paper's balanced-panel assumption), rank selection is implicit via nuclear-norm soft-thresholding with no discrete `rank_selection` constructor parameter (matches paper Section 5.3 + Appendix; corrects an earlier REGISTRY overclaim that listed cv / ic / elbow methods), `λ_nn = ∞` → 1e10 internal sentinel with original-value storage on results. **Outstanding Concerns (deferred):** Equation 14 covariate extension (`TROP.fit()` does not accept a `covariates` kwarg; non-support locked by `TestTROPDeviations::test_covariates_not_supported` via `inspect.signature` to guard against future `**kwargs`) and Theorem 8.1 (covariate triple robustness) deferred until use cases motivate; SC / SDID reductions paper-claimed under "specific (omega, theta) weight choices" not provided in the paper text — cross-language anchor deferred. **R parity:** deferred until paper-author reference implementation is released ("forthcoming" per the paper). REGISTRY.md `## TROP` section gains a "Verified Components" expansion: 10 ticked requirements + four `**Note:**` / `**Note (library-side choice):**` / `**Note (deferral):**` annotations consolidating the deviation surface (Eq. 10 balancing-decomposition pointer, Gap #5 weight-normalisation library-side choice, Eq. 14 covariate deferral). No source-code changes to `diff_diff/trop*.py`. Methodology sign-off scope: paper-aligned identification ingredients (Eq. 2 prox, Eq. 3 weights, Eqs. 4-5 LOOCV, Algorithms 1-3, Corollary 1 unbiasedness, Eq. 6 simulation recovery, DID reduction, documented deviations) are directly locked by the new tests. Theorem 5.1 is verified as a simulation sanity check (TROP RMSE < DID RMSE under LOOCV-tuned weights), NOT as a direct fixed-weight conditional-bias-bound lock; the Matrix Completion reduction is verified as code-path activation (effective_rank > 0 + beats DID baseline), NOT as equivalence against an independent MC reference. The Eq. 14 covariate extension is documented as deferred (TROP.fit() has no `covariates` kwarg).
- **ImputationDiD `vcov_type` input contract (Phase 1b interstitial #3, permanently narrow).** `ImputationDiD(vcov_type=...)` now accepts `{"hc1"}` only (default). Analytical-sandwich families `{classical, hc2, hc2_bm}` and `conley` spatial-HAC are REJECTED at `__init__` with methodology-rooted messages mirroring the CallawaySantAnna (PR #487) and TripleDifference (PR #488) interstitials. The rejection is **library-architectural, not paper-prescribed**: ImputationDiD uses influence-function-based variance per Borusyak-Jaravel-Spiess (2024) Theorem 3 — the per-unit IF aggregation `psi_it = v_it · epsilon_tilde_it` has no equivalent single design matrix on which hat-matrix leverage `1/(1−h_ii)` or Bell-McCaffrey Satterthwaite DOF can be defined. `hc1` with `cluster=None` ≡ per-unit IF variance (Theorem 3 equation 7); `hc1` with `cluster=X` ≡ per-cluster IF summation `sigma_sq = (cluster_psi_sums**2).sum()` (plain CR1 — no Stata-style `(n-1)/(n-p)` finite-sample factor because the IF has no design-matrix `p`); `hc1` with `survey_design=` ≡ TSL on the combined IF via `compute_survey_if_variance()` (analytical strata/PSU/FPC or replicate BRR/Fay/JK1/JKn/SDR). All paths are unchanged at machine precision (default behavior bit-equal across `aggregate ∈ {None, "event_study", "group"}` and across analytical + bootstrap inference). `vcov_type`, `cluster_name`, and `n_clusters` fields added to `ImputationDiDResults`; threaded through new `to_dict()` method (also net-new, mirrors `TripleDifferenceResults.to_dict()`). `summary()` routes the variance-family label through the shared `_format_vcov_label` (`results.py:49-89`): default `cluster=None` fits render `"CR1 cluster-robust at <unit>, G=<n_units>"` (Theorem 3 still clusters at the `unit` column when `cluster=None`); explicit `cluster=<col>` fits render `"CR1 cluster-robust at <cluster_name>, G=<n>"`; survey-backed fits suppress the variance-estimator line (the Survey Design block already names design + n_psu + df); bootstrap fits suppress the analytical variance-family label and render `"Inference method: bootstrap"` + `"Bootstrap replications: <n>"` instead (mirrors the canonical `DiDResults.summary()` gate at `results.py:213-226` — the displayed SE/CI/p-value are bootstrap-derived, not analytical). **`cluster= + SurveyDesign(replicate_weights=[...])` raises `NotImplementedError`** at `fit()`: replicate-weight variance is computed by replicate reweighting and ignores PSU/cluster entirely; honoring bare `cluster=` would silently no-op while populating `cluster_name`/`n_clusters` on Results dishonestly. Mirrors the CS PR #487 + TD PR #488 fail-closed guards. **Bootstrap path returns NaN SE when fewer than 2 independent clusters/PSUs are available** (`n_clusters < 2` analytical path, `n_psu < 2` survey-PSU path); without this guard the multiplier bootstrap SE collapses to ≈0 from BLAS roundoff (NOT NaN) and downstream zero-SE checks miss the degenerate case. NaN propagates to all overall ATT inference fields plus per-horizon and per-group bootstrap dicts via the new `_build_nan_bootstrap_results` helper in `imputation_bootstrap.py`. `set_params(vcov_type=...)` mirrors CS+TD pattern (mutate-then-validate-at-use, no atomic validation); `fit()` re-validates `vcov_type` at use time. New `TestImputationDiDVcovType` class in `tests/test_imputation.py` covers the 7-surface contract (default / cluster / TSL-survey / replicate-survey + bootstrap × cluster + bootstrap × survey bit-equal — ALL parametrized over `aggregate ∈ {None, "event_study", "group"}` with per-horizon and per-group SE override branches pinned; `fit()`-time revalidation; bootstrap n_psu<2 + n_clusters<2 NaN propagation including `coef_var` NaN; `pretrends=True` × `vcov_type='hc1'` × cluster bit-equality) plus introspection (default attr, `get_params`, Results carries, `to_dict`, summary label, cluster_name suppression under survey, fit-clone idempotence, convenience function) and input-rejection tests with distinct keyword `match=` pins per family. REGISTRY.md "IF-based variance estimators vs analytical-sandwich estimators" cross-reference section updated to list `ImputationDiD` alongside `CallawaySantAnna` and `TripleDifference` in the "Enforced today" tier. **Interstitial PR #3** in the Phase 1b sequence (after CS #487, TD #488). Two estimators remaining: TwoStageDiD (methodology-heavy, sandwich + Gardner GMM-corrected meat) and EfficientDiD (IF-based interstitial #4, follows the same narrow-contract template).
- **TripleDifference `vcov_type` input contract (Phase 1b interstitial #2, permanently narrow).** `TripleDifference(vcov_type=...)` now accepts `{"hc1"}` only (default). The analytical-sandwich families `{classical, hc2, hc2_bm}` and `conley` spatial-HAC are REJECTED at `__init__` with methodology-rooted messages mirroring the CS interstitial. The rejection is **library-architectural, not paper-prescribed**: TripleDifference uses influence-function-based variance per Ortiz-Villavicencio & Sant'Anna (2025) arXiv:2505.09942 — the 3-pairwise-DiD decomposition `inf = w3·IF_3 + w2·IF_2 - w1·IF_1` has no single design matrix to compute hat-matrix leverage `1/(1-h_ii)` or Bell-McCaffrey Satterthwaite DOF on. The narrow contract is permanent and applies to the remaining IF-based estimators (`ImputationDiD`, `EfficientDiD`) when their `vcov_type` threading PRs land. `hc1` with `cluster=None` ≡ per-unit IF variance (`std(inf)/sqrt(n)`); `hc1` with `cluster=X` ≡ CR1 Liang-Zeger on the combined IF (`(G/(G-1)) · Σ_c (Σ_{i∈c} ψ_i)² / n²`, plain CR1 — no Stata-style `(n-1)/(n-p)` finite-sample factor because the IF has no design-matrix `p` in the OLS sense); `hc1` with `survey_design=` ≡ TSL on the combined IF (analytical or replicate). All three paths are unchanged at machine precision (default behavior bit-equal across all 3 estimation methods `{dr, reg, ipw}`). `vcov_type` and `cluster_name` fields added to `TripleDifferenceResults`, threaded through `to_dict()`. `summary()` routes the variance-family label through the shared `_format_vcov_label` (`results.py:49-89`): bare fits render `"HC1 heteroskedasticity-robust"`, clustered fits render `"CR1 cluster-robust at <cluster_name>, G=<n>"` (since the actual algebra is Liang-Zeger CR1 on the combined IF), and survey-backed fits suppress the variance-estimator line entirely (the Survey Design block already names design + n_psu + df, and the analytical SE is TSL on the combined IF — a raw "hc1" label would misstate the inference path). **`cluster= + SurveyDesign(replicate_weights=[...])` raises `NotImplementedError`** at `fit()`: replicate-weight variance is computed by replicate reweighting (BRR / Fay / JK1 / JKn / SDR) and ignores PSU/cluster entirely; honoring bare `cluster=` would silently have no effect on the variance estimate while populating `cluster_name`/`n_clusters` on Results dishonestly. Mirrors the `CallawaySantAnna` guard from PR #487. Under `survey_design.psu` (non-replicate path) `cluster_name`/`n_clusters` on Results are suppressed (set to None) so they can't misreport the raw cluster argument when the resolver picks the survey PSU instead. `set_params(vcov_type=...)` mirrors CS pattern (mutate-then-validate-at-use, no atomic validation); `fit()` re-validates `vcov_type` at use time so a `set_params(vcov_type="hc4")` mutation surfaces a clear error at fit-time rather than silently propagating to Results metadata. **Interstitial PR #2** (after CS PR #487) rather than full Phase 1b PR 4/8 vcov_type threading — the narrow surface is methodologically dictated by TripleDifference's IF-based variance, not a deferral. New `TestTripleDifferenceVcovType` class in `tests/test_triple_diff.py` covers the 5-surface contract (default/cluster/survey bit-equal, `__init__` rejection per family, `fit()`-time revalidation) plus 8 introspection / convenience-function tests. REGISTRY.md "IF-based variance estimators vs analytical-sandwich estimators" cross-reference section updated to list `TripleDifference` alongside `CallawaySantAnna` in the "Enforced today" tier. Phase 1b PR 4/8 (full `{classical, hc1, hc2, hc2_bm}` threading) resumes on a different estimator (TwoStageDiD) post-merge; the two remaining IF-based estimators (`ImputationDiD`, `EfficientDiD`) follow the same narrow-contract template.
- **CallawaySantAnna `vcov_type` input contract (Phase 1b interstitial, permanently narrow).** `CallawaySantAnna(vcov_type=...)` now accepts `{"hc1"}` only (default). The analytical-sandwich families `{classical, hc2, hc2_bm}` and `conley` spatial-HAC are REJECTED at `__init__` with methodology-rooted messages. The rejection is **library-architectural, not paper-prescribed**: CS uses influence-function-based variance per Callaway & Sant'Anna (2021) — per-(g,t) doubly-robust / IPW / outcome-regression structure — and has no single design matrix to compute hat-matrix leverage `1/(1-h_ii)` or Bell-McCaffrey Satterthwaite DOF on. The narrow contract is permanent and applies to other IF-based estimators (ImputationDiD, EfficientDiD) when their `vcov_type` threading PRs land. `hc1` with `cluster=None` ≡ per-unit IF variance (Williams 2000 form); `hc1` with `cluster=X` ≡ CR1 Liang-Zeger on the IF activated via the cluster= wiring fix above. Documentation in `docs/methodology/REGISTRY.md` "IF-based variance estimators vs analytical-sandwich estimators" subsection. `vcov_type`, `cluster_name`, `n_clusters`, `df_inference` added to `CallawaySantAnnaResults` (the canonical PSU column wins for `cluster_name` reporting — `survey_design.psu` when explicit PSU is provided, `self.cluster` when bare cluster synthesizes/injects). `set_params(vcov_type=...)` mirrors SA pattern (mutate-then-refresh `_vcov_type_explicit`, no atomic validation); `fit()` re-validates `vcov_type` at use time so a `set_params(vcov_type="hc4")` mutation surfaces a clear error at fit-time rather than silently propagating to Results metadata. **Interstitial PR** rather than full Phase 1b PR 4/8 vcov_type threading — the narrow surface is methodologically dictated by CS's IF-based variance, not a deferral. Phase 1b PR 4/8 (full {classical, hc1, hc2, hc2_bm} threading) resumes on a different estimator post-merge.
- **TripleDifference cluster-changes-SE defensive regression test.** Added `tests/test_triple_diff.py::TestTripleDifferenceClusterDefensive::test_cluster_changes_ses` asserting that `TripleDifference(cluster="state")` produces SE differing from `cluster=None` SE by `>1e-6` on a fixed-seed panel with state-level random effects. Defensive coverage closes a test gap identified during the Phase 1b cluster-wiring audit; TripleDifference's bare-cluster code path (`triple_diff.py:1245-1259`) was already correct but lacked a positive regression test. Mirrors `tests/test_two_stage.py::test_cluster_changes_ses`.
- **TwoStageDiD: parity with SpilloverDiD Wave E.3 — always-treated unit drop preserves full-domain survey design via zero-padded scores.** Closes the parity follow-up tracked at `TODO.md` after PR #482 (SpilloverDiD Wave E.3, merge `24de9062`). When TwoStageDiD detects always-treated units (`first_treat <= min_time`) and removes them from the OLS sample, the resolved survey design retains its FULL-DOMAIN `n_psu` / `n_strata` / `df_survey` / `strata` / `fpc` / `psu` arrays instead of being subsetted via `replace(resolved_survey, ...)`. Per-cluster stage-1 / stage-2 score aggregates are computed at the post-drop fit-sample length and then zero-padded onto the full-domain unique-PSU list before stratified-meat dispatch via two new optional kwargs on `_compute_gmm_variance`: `score_pad_mask` (full-domain boolean keep mask) and `cluster_ids_full` (full-domain post-injection PSU labels). PSUs containing only always-treated rows get zero score rows but still count toward `G_full` for `n_psu` / `df_survey` accounting. **Documented synthesis (library-convention adoption, NOT new methodology):** adopts the canonical "zero-pad scores + retain full-design resolved survey" convention from R `survey::svyrecvar(subset())` (Lumley 2010 §2.5) already established in `diff_diff/imputation.py:2175-2183` (PreTrendsImputation), `diff_diff/prep.py:1401-1432` (DCDH cell variance), and `diff_diff/spillover.py` (PR #482 Wave E.3). **Mechanical realization:** `two_stage.py:1485-1525` design-subset block deleted (the `replace(resolved_survey, ...)` subset + `n_psu` / `n_strata` recompute + post-drop `compute_survey_metadata` call); `keep_mask` promoted to `fit()`-level scope (always defined, all-True when no always-treated drop); `survey_weights = survey_weights[keep_mask.values]` retained for stage-1 / stage-2 OLS arithmetic; cluster injection block updated to source `cluster_ids_raw` from FULL-DOMAIN `data[cluster_var].values` (not post-drop `df[cluster_var].values`) so `_inject_cluster_as_psu`'s zip against `resolved_survey.strata` (full-domain) stays length-aligned; `df["_survey_cluster"]` aligned to post-drop length via `resolved_survey.psu[keep_mask.values]`; post-injection `compute_survey_metadata` uses full-domain `raw_w` from `data[survey_design.weights]`. `_compute_gmm_variance` adds the zero-pad expansion after the per-cluster aggregation (mapping fit-sample `unique_clusters` into `unique_clusters_full` positions via `np.searchsorted`) and updates the strata/fpc `obs_idx` lookups to use `cluster_ids_for_lookup = cluster_ids_full` when padding is active. The three inner stage-2 methods (`_stage2_static`, `_stage2_event_study`, `_stage2_group`) thread the new kwargs through; bootstrap-resample call sites keep default `None` (no behavior change on bootstrap path). **Always-treated warning text updated:** "Associated survey weights subsetted for stage-1 / stage-2 OLS; full-domain survey design retained for variance estimation (Wave E.3 parity)." replaces the prior "and design arrays adjusted" claim. **No-survey path unchanged:** when `resolved_survey is None`, both `score_pad_mask` and `cluster_ids_full` default to `None` and the existing post-drop scoring path runs bit-identically. **Replicate variance + always-treated drop:** existing path unchanged (replicate refit handles resampling at the survey-design level; `score_pad_mask_arg` is `None` on `_uses_replicate_ts` paths). **Tests:** new `TestTwoStageDiDWaveE3ParityAlwaysTreated` class in `tests/test_two_stage.py` (8 tests: no-always-treated baseline, full-domain `df_survey` preservation under drop, full-domain `n_psu` reporting, per-cluster zero-pad mock-spy on `_compute_stratified_meat_from_psu_scores`, subpopulation + always-treated composition, cluster-as-PSU + always-treated, no-survey path unchanged, PSU entirely-always-treated). REGISTRY.md TwoStageDiD section gains a "documented synthesis — Wave E.3 parity" note; SpilloverDiD Wave E.3 section updated to mark the TwoStageDiD parity follow-up as shipped.
- **WooldridgeDiD (ETWFE) methodology-review-tracker promotion: In Progress → Complete.** Closes the Wooldridge (2025) *Empirical Economics* 69(5) primary-source review on the methodology-review tracker (PR-A #484 added the paper review on file; this PR-B closes two paper gaps and completes the F.L.I.P. consolidation). `METHODOLOGY_REVIEW.md` L52 status flipped with merge date `2026-05-22`; detail section L584-605 rewritten to the Verified Components / Test Coverage / Corrections Made / Deviations / Outstanding Concerns template mirroring the HAD (PR #473) / ContinuousDiD (PR #476) / DCDH (PR #481) precedents; L27 In Progress example re-pointed to TROP; priority queue items #7-#10 renumbered to #6-#9.
- **WooldridgeDiD `weights="cohort_share"` on `aggregate()` (paper W2025 Eq. 7.4 / Eq. 7.6).** `WooldridgeDiDResults.aggregate(type, weights="cell" | "cohort_share")` exposes the paper's cohort-share aggregation as an opt-in alternative to the default cell-count weighting (which matches Stata `jwdid_estat`). Under `weights="cohort_share"`, per-cell weights are `∝ N_g` (per-cohort unit count): for `type="simple"` (paper Eq. 7.4) the simple-overall ATT normalizes across all post-treatment cells; for `type="event"` (paper Eq. 7.6 cohort-share-by-exposure) the normalization is per-event-time across cohorts present at event-time `e`. `type="group"` and `type="calendar"` raise `ValueError` under cohort_share (no paper closed-form). The Bell-McCaffrey contrast DOF (`vcov_type="hc2_bm"`) is rebuilt under the active weighting scheme so SE + DOF reflect the actual aggregation. On balanced panels with uniform within-cohort cell counts the two schemes coincide (paper Section 7.5 footnote). New `_n_g_per_cohort` field on `WooldridgeDiDResults` carries the per-cohort unit counts; populated in all three fit paths (OLS / logit / Poisson). Closes TODO row 95.
- **WooldridgeDiD `cohort_trends=True` for paper W2025 Section 8 / Eq. 8.1 heterogeneous cohort-specific linear trends.** `WooldridgeDiD(cohort_trends=True)` adds linear `dg_i · t` interactions for each treated cohort to the design matrix. Under the heterogeneous-trends DGP `y = c_i + α_t + δ_g · t + τ · w_{it} + u_{it}`, the parameter recovers `τ` even when parallel trends fails (paper Section 8.3). **OLS-path only:** `cohort_trends=True` + `method ∈ {"logit","poisson"}` raises `NotImplementedError` at `__init__` per paper Section 8's OLS scope; the error message cites the paper section explicitly. **Auto-routes to full-dummy mode** regardless of `vcov_type` (matching the absorb→fixed_effects auto-route pattern at `feedback_absorb_to_fixed_effects_auto_route`): composing `dg_i · t` with the within-transformation yields `(dg_i − mean(dg_i)) · (t − mean(t))` which is algebraically correct but non-trivial to verify on every panel shape; the full-dummy auto-route keeps math closure verified on the same paths already locked by PR #483's HC2 / HC2-BM / classical R-parity goldens. New `cohort_trend_coefs: Dict[g → δ_g]` attribute on `WooldridgeDiDResults` (empty dict under default `cohort_trends=False`). Closes the PR-A Requirements Checklist heterogeneous-trends gap (item 11 in `docs/methodology/papers/wooldridge-2025-review.md`).
- **WooldridgeDiD R-parity goldens for `etwfe(family="poisson")` + `etwfe(family="logit")`.** `benchmarks/R/generate_wooldridge_golden.R` extended to fit R `etwfe` on Poisson + logit DGPs and persist log-link coefficient + SE goldens to `benchmarks/data/wooldridge_golden.json` (Poisson + logit blocks alongside the existing OLS vcov_type blocks). `benchmarks/R/requirements.R` pins `etwfe >= 0.5.0`. The R goldens cover diff-diff's nonlinear surfaces only at the surface level (fit completes + log-link goldens present + structured correctly); numerical cell-level R-parity between diff-diff's response-scale ATT (paper W2023 ASF / APE) and R `etwfe`'s log-link cell coefficient is deferred — requires either `emfx()`-based APE extraction on the R side or link-function inversion with baseline-mean adjustment (new TODO row added).
- **`tests/test_methodology_wooldridge.py` extended with 6 paper-equation-numbered methodology classes + 1 library-deviations class.** Net ~70 new tests across 10 classes (joining the existing 12 vcov_type R-parity tests from PR #483) covering Theorem 3.1 Mundlak ≡ TWFE equivalence, Proposition 5.1 imputation ≡ POLS, Section 6 event study, Section 7 aggregation paths (paper Eqs. 7.4 / 7.6 hand-calc + survey/bootstrap/never-treated rejections), Section 8 heterogeneous trends (per-cohort identification + all-treated last-cohort drop + survey/never-treated cross-product rejections + reporting metadata), Section 10 unbalanced panels + time-varying covariates, plus a `TestW2025LibraryDeviations` class consolidating 5 surviving deviations (HC1 finite-sample factor, QMLE sandwich `(n-1)/(n-k)`, nonlinear-vs-fixest, logit cohort+time dummies, anticipation + aggregation). Per-class seed decorrelation via `_BASE_SEED_*` module constants mirrors the HAD precedent at `tests/test_methodology_had.py:78-83`. New DGP helpers (`_make_two_cohort_three_period_panel`, `_make_three_cohort_four_period_panel`, `_make_heterogeneous_trends_panel`, `_make_unbalanced_panel`) reusable across the methodology classes. Two new surface-only R-parity classes (`TestWooldridgeParityRPoisson`, `TestWooldridgeParityRLogit`) lock the Poisson + logit goldens at the structural level.
- **WooldridgeDiD `vcov_type` parameter, OLS path (Phase 1b PR 3/8).** `WooldridgeDiD(vcov_type=...)` now accepts `{"classical","hc1","hc2","hc2_bm"}` on `method="ols"` (defaults to `"hc1"`, preserves prior behavior at machine precision — the WLS-CR1 sandwich is algebraically invariant between the prior within-transform path and the new branched path, differing only by float64 multiplication ordering at sub-ULP scale; the full 106-test `tests/test_wooldridge.py` baseline still passes unchanged). `hc2_bm` auto-routes to a full-dummy saturated design (`[intercept, X_design, unit_dummies, time_dummies]`) + clubSandwich WLS-CR2 algebra (PR #475) — matches `clubSandwich::vcovCR(lm(...), type="CR2") + coef_test()$df_Satt` at `atol=1e-10` on the new `benchmarks/data/wooldridge_golden.json` fixture. `classical`/`hc2` supported via full-dummy + auto-drop of the unit auto-cluster (one-way families); explicit `cluster="X"` + one-way family raises at the linalg validator. Per-cell + aggregate p-values/CIs on `classical`/`hc2` paths use the residual DOF `n - rank(X)` (matches R `lm()` / `coef_test()` t-distribution), not normal-theory. **Bell-McCaffrey Satterthwaite DOF is threaded across ALL hc2_bm user-facing inference surfaces**: (1) per-cell `group_time_effects[(g, t)]` use `coef_test()$df_Satt` (matches R at atol=1e-6 from CI inversion); (2) overall ATT uses the post-period-aggregation contrast DOF from `_compute_cr2_bm_contrast_dof` (matches R `Wald_test(test="HTZ")$df_denom` at atol=1e-10); (3) `.aggregate("group" | "calendar" | "event")` recomputes contrast-specific BM DOFs lazily from BM artifacts stored on the Results object — the REDUCED kept-column design (`X_red`), cluster_ids, reduced bread matrix, and reduced-space coef-index map (using the reduced kept-column design after rank-deficient drops keeps the bread non-singular and matches the subspace `solve_ols` actually estimated in). Fail-closed (all-NaN inference) when BM DOF unavailable, mirrors PR #475 R7 and PR #479 R3. `method ∈ {"logit","poisson"}` + `vcov_type != "hc1"` raises `NotImplementedError` at `__init__` (GLM CR2-BM-on-pseudo-residuals composition needs derivation; deferred to follow-up TODO row). `SurveyDesign` + `vcov_type != "hc1"` raises `NotImplementedError` at `fit()` (survey TSL overrides analytical sandwich). `n_bootstrap > 0` + one-way (`hc2`/`classical`) raises at `fit()` regardless of `cluster=` setting (multiplier bootstrap is intrinsically clustered, but one-way vcov_type does not compose with cluster_ids — either the auto-cluster is dropped when `cluster=None` leaving the bootstrap with no cluster to draw at, or the linalg validator rejects one-way + cluster_ids when `cluster=X`). `conley` rejected at `__init__` with a deferral pointer. `vcov_type`, `cluster_name`, `n_clusters` added to `WooldridgeDiDResults` for downstream introspection (per `feedback_results_vcov_label_cluster_metadata`). Third PR of the Phase 1b standalone-estimator threading initiative (5 PRs to follow: CallawaySantAnna, ImputationDiD, TripleDifference, TwoStageDiD, EfficientDiD).
- **`SpilloverDiD(survey_design=SurveyDesign.subpopulation(...))` full-design retention via zero-pad scores (Wave E.3).** Closes the Wave E.1/E.2/follow-up documented limitation at `REGISTRY.md:3249`: `SurveyDesign.subpopulation()`-derived designs AND warn-and-drop fits now preserve the full-domain resolved survey design — `n_psu` / `n_strata` / `df_survey` / Binder TSL per-stratum centering reflect the FULL domain rather than the post-`finite_mask` fit sample. **Documented synthesis (library-convention adoption, NOT new methodology):** Wave E.3 adopts the canonical "zero-pad scores to full panel + retain full-design resolved survey" pattern from R `survey::svyrecvar(subset())` (Lumley 2010 §2.5) already established in `diff_diff/imputation.py:2175-2183` (PreTrendsImputation lead regression — Omega_0 scores zero-padded back to full panel length) and `diff_diff/prep.py:1401-1432` (DCDH cell variance — IF zero-padded outside the cell). Wave E.3 propagates the same convention to SpilloverDiD's Wave E.1 Binder TSL × Wave D Gardner GMM × Wave E.2/follow-up stratified-Conley + serial Bartlett meat. **Mechanical realization (one new `_compute_gmm_corrected_meat` kwarg):** the gamma_hat / Psi build stays on SURVEY-FINITE-MASK inputs (`X_1_sparse_fit`, `X_10_sparse_fit`, `eps_10_fit` built on `survey_finite_mask = finite_mask & survey_weights > 0`; `X_2_kept_gamma`, `eps_2_fit_gamma`, `survey_weights_fit_gamma` projected from the fit-sample frame down to survey_finite_mask) so the drop-first stage-1 FE column space is bit-identical to the pre-E.3 path. `_compute_gmm_corrected_meat` gains a new optional kwarg `score_pad_mask: Optional[np.ndarray] = None`: when supplied, the helper zero-pads the fit-sample `Psi` to full panel length AFTER construction but BEFORE kernel dispatch via `Psi_padded[score_pad_mask] = Psi`. Kernel-dispatch arrays (`cluster_ids`, `conley_coords`, `conley_time`, `conley_unit`, `resolved_survey`) are passed at FULL length so the meat helpers (Binder TSL / stratified-Conley / serial Bartlett) see the full-domain PSU / strata / centroid / time geometry. The `_validate_conley_kwargs` call inside the helper reads `n_for_conley = len(score_pad_mask)` when the kwarg is set so the Conley shape checks see the full-length geometry. **`gamma_hat` invariance:** the gamma_hat solve operates on fit-sample inputs throughout — bit-identical to the pre-E.3 path (critical for the case where `_build_butts_fe_design_csr`'s `pd.factorize` re-compaction would drop a different unit's column under a full-length FE build than under a fit-length one). **Bread invariance:** `A_22 = X_2_kept' W X_2_kept` at `spillover.py:3187-3214` still uses fit-length `X_2_kept` because `A_22_full = X_2_full' W_full X_2_full` equals `A_22_kept` when zero-weight rows contribute zero. **A2 invariant:** warn-and-drop and `SurveyDesign.subpopulation()` drops are treated identically — both apply the zero-pad mechanism. The "both mechanisms compose cleanly" case (subpop-excluded row that is ALSO warn-and-dropped) produces `Psi = 0` from either cause; the PSU still counts toward `n_psu_full`. Hand-computation methodology anchor at `_scratch/wave_e3_smoke.py` codifies the A2 invariant on 4 PSU × 4 period × 3 obs synthetic. **Subpopulation parity vs upstream-subset:** `df_survey` matches the full domain regardless of how many rows the subpopulation mask excludes (mirrors R `svyglm(design=subset(d, mask))` vs `svyglm(design=svydesign(data=data[mask], ...))`). SE may differ by design — subpopulation retains zero-padded PSU geometry; upstream-subset drops PSUs entirely. **Pre-E.3 baseline parity:** when `finite_mask.all() == True` AND all weights `> 0`, the Wave E.3 zero-pad is a no-op — ATT + SE + n_psu + df_survey match pre-E.3 baseline values via FIXED GOLDEN values at `test_c` (`rtol=1e-12, atol=1e-12`). **Cross-surface n_psu consistency:** top-level `res.n_psu` reads from `len(resolved_survey_fit.weights)` on the implicit-PSU branch (was `int(finite_mask.sum())` pre-codex-R1-P2-fix); this keeps `res.n_psu == res.survey_metadata.n_psu` on weights-only / strata-only survey designs under warn-and-drop. Regression at `test_c2`. **Restrictions inherited:** replicate-weight variance + subpopulation continues to raise `NotImplementedError` at the Wave E.1 gate. TwoStageDiD's analogous `finite_mask + design-subset` pattern at `two_stage.py:567-601` is NOT yet adopted to Wave E.3 — separate parity follow-up tracked in `TODO.md` (an expected-divergence test was attempted but TwoStageDiD's always-treated handling at `two_stage.py:294-336` differs from SpilloverDiD's per-unit Omega_0 check, so the divergence didn't materialize on the standard fixture; the parity follow-up should add its own targeted regression). **Implementation:** `spillover.py:2845-2896` design-subset block deleted; `survey_weights_fit = survey_weights[finite_mask]` retained for the stage-2 OLS solve which still operates on the fit sample; `cluster_ids_full[finite_mask]` subset dropped on the survey path. `_compute_gmm_corrected_meat` call at `spillover.py:3163` now receives FIT-LENGTH gamma_hat-construction inputs (unchanged) plus FULL-LENGTH kernel-dispatch arrays (`cluster_ids_for_meat`, `conley_*_for_meat`, `resolved_survey_fit`) plus the new `score_pad_mask=survey_finite_mask` kwarg; no-survey path passes `score_pad_mask=None` and uses fit-length variables throughout (bit-identical to pre-E.3). `_compute_gmm_corrected_meat` at `two_stage.py:62-80` adds one new optional kwarg `score_pad_mask: Optional[np.ndarray] = None` and one post-Psi-construction zero-pad block; the `_validate_conley_kwargs` call uses `n_for_conley = len(score_pad_mask)` when the kwarg is set. Within-unit-constancy validator at `spillover.py:2913` updated to operate on full-length unit array. Second `compute_survey_metadata` recompute at `spillover.py:2954-2959` uses full-length `raw_w`. No `_compute_stratified_meat_from_psu_scores` / `_compute_stratified_conley_meat` / `_compute_stratified_serial_bartlett_meat` signature changes. **Tests:** new `TestSpilloverDiDWaveE3SubpopulationFullDesign` and `TestSpilloverDiDWaveE3SubpopulationFullDesignEventStudy` classes in `tests/test_spillover.py` (19 tests: pre-E.3 baseline parity via pinned goldens, n_psu cross-surface consistency on implicit-PSU branch, A2 invariant (zero-pad mechanics via mock-spy), subpopulation × explicit-PSU parity, conley + lag>0 + subpopulation × explicit-PSU / cluster-injection / weights-only branches, cluster-as-PSU + subpopulation parity, unit with BOTH zero weight AND no Omega_0 support, gamma_hat-build sample excludes zero-weight rows, n_obs / n_treated / n_control / n_far_away_obs reflect count_mask, warn-drop SE drift golden, ATT bit-equality under PSU-last-sort exclusion, exact event-study n_obs propagation, event-study on both is_staggered branches with analytical + conley+lag variants). Pre-existing Wave E.1 `test_p2_finite_mask_forces_drop_under_survey` assertion flipped from `n_psu=8` (subset) to `n_psu=10` (full domain) to reflect the new contract.
- **ChaisemartinDHaultfoeuille (DCDH) methodology-review-tracker promotion.** Tracker row flipped **In Progress** → **Complete** with full Verified Components / Test Coverage / Corrections Made / Deviations / Outstanding Concerns structure mirroring the HAD precedent (PR #473) and ContinuousDiD precedent (PR #476). REGISTRY `## ChaisemartinDHaultfoeuille` gains a formal `### Deviations from the paper / from R / library extensions` block consolidating 7 documented deviations into a single AI-review-recognized labeled surface (per CLAUDE.md "Documenting Deviations (AI Review Compatibility)"): (D1) equal-cell weighting (deviation from BOTH AER 2020 Equation 3 AND R `DIDmultiplegtDYN`); (D2) period-based vs cohort-based stable controls; (D3) balanced-baseline panel + interior-gap drops + terminal-missingness retention + cell-period-allocator targeted `ValueError`; (D4) SE normalization `N_l` vs R `G` (~4% smaller analytical SE); (D5) singleton-cohort degeneracy → NaN with `UserWarning`; (D6) `<50%` switcher warning at far horizons (library extension citing Favara-Imbs application, footnote 14 of NBER WP 29873); (D7) Phase 3 `DID^X` covariate first-stage equal-cell weights. R cross-language coverage holds at documented tolerance bands in `tests/test_chaisemartin_dhaultfoeuille_parity.py` (`POINT_RTOL = 1e-4` on pure-direction point estimates, `MIXED_POINT_RTOL = 0.025` on mixed-direction, `PURE_DIRECTION_SE_RTOL = 0.05` on pure-direction SE, `SE_RTOL = 0.10` on multi-horizon SE, `se_rtol=0.15` on the long-panel `L_max=5` joiners-only scenario where cell-count-weighting compounds). No source code changes, no new tests, no new docstrings — consolidation only against the existing 12 methodology tests (`tests/test_methodology_chaisemartin_dhaultfoeuille.py`), 26 R-parity tests (`tests/test_chaisemartin_dhaultfoeuille_parity.py`), 352 unit tests (`tests/test_chaisemartin_dhaultfoeuille.py`), survey suites (`tests/test_survey_dcdh.py`, `tests/test_survey_dcdh_replicate_psu.py`, three cell-period coverage suites), and two primary-source DCDH paper reviews on disk (2020 AER + 2022/2023 NBER WP 29873 via PR #478; the `dechaisemartin-2026-review.md` on disk is HAD's primary source, not DCDH's, and is referenced as adjacent context only). The REGISTRY Deviations block uses semantic section-name anchors (rather than fragile line numbers) for back-references to other parts of the DCDH section — an intentional divergence from the PR #476 ContinuousDiD precedent reflecting PR-A wording-drift CI feedback that flagged line-number cross-references as drift-prone in long sections. `METHODOLOGY_REVIEW.md` DCDH row promoted **In Progress** → **Complete**; L27 In Progress example paragraph re-pointed to WooldridgeDiD; L1289 priority-order queue item #6 (DCDH) removed and items #7-#11 renumbered to #6-#10.

### Changed
- **Internal refactor: dedup serial Bartlett kernel construction and PSD guard between Conley no-survey and TwoStage survey paths.** Extracts `_serial_bartlett_kernel_matrix(t_codes, L)` and `_validate_meat_psd(M, *, error_msg, warning_template, stacklevel=3)` to `diff_diff/conley.py`. Replaces three inline kernel constructions (`conley.py` panel-block branch, `two_stage.py` survey singleton-adjust branch, `two_stage.py` survey multi-PSU branch) and two inline finite-plus-eigvalsh guards (`conley.py::_compute_conley_meat`, `two_stage.py` survey panel-block orchestrator) with helper calls. No behavior change — methodology anchor at `tests/test_spillover.py::TestSpilloverDiDWaveE2FollowupConleySurveyLagCutoff` (21 tests including hand-computed serial Bartlett HAC at L=1) and existing PSD-warning monkey-patch tests at `tests/test_conley_vcov.py::TestConleyDirectHelper::{test_uniform_kernel_negative_eigenvalue_warns, test_indefinite_meat_warning_fires_for_bartlett}` still pass unchanged (substring `"bartlett"` / `"uniform"` / `"negative eigenvalue"` in warning messages preserved byte-for-byte). New `TestSerialBartlettKernelMatrix`-grouped tests in `TestConleyKernels` (5 tests: hand-computed L=2 / L=1 / L=0 degenerate / single-element / int-vs-float bit-equality contract) and new `TestValidateMeatPsd` class (4 tests: non-finite raises with caller's `error_msg`, negative-eigenvalue warns with `{eigval:.2e}` substituted, PSD matrix silent, threshold boundary at -5e-13 silent). Closes `TODO.md` Bartlett-dedup row.

## [3.4.1] - 2026-05-21

### Added
- **StackedDiD `vcov_type` parameter (Phase 1b PR 2/8).** `StackedDiD(vcov_type=...)` now accepts `{"hc1","hc2_bm"}` (defaults to `"hc1"`, preserves prior behavior at machine precision — WLS-CR1 sandwich is algebraically invariant between the prior bake-Q-into-X pattern and the new `solve_ols(weights=composed_weights, vcov_type=...)` path, differing only by float64 multiplication ordering at ~2 ULPs at SE scale; pinned by `test_hc1_se_bit_equal_to_pre_pr_baseline` at `atol=1e-13`). `classical` and `hc2` rejected at `__init__` with cluster-incompatibility `ValueError` — StackedDiD clusters intrinsically at `'unit'` or `'unit_subexp'` (no `cluster=None` opt-out), so one-way families are not composable with the linalg validator's cluster_ids check. `hc2_bm` routes CR2 Bell-McCaffrey through the clubSandwich WLS-CR2 port (PR #475) — matches `clubSandwich::vcovCR(lm(weights=Q,...), cluster=~unit|unit_subexp, type="CR2") + coef_test()$df_Satt` at `atol=1e-10` on the new `benchmarks/data/stacked_did_golden.json` fixture (6 R-parity tests in `tests/test_methodology_stacked_did.py`). HC1 + cluster matches `clubSandwich::vcovCR(..., type="CR1S")` (Stata-style `G/(G-1) * (n-1)/(n-p)` correction; plain `type="CR1"` omits the `(n-1)/(n-p)` term and would diverge by ~1.4% on the test fixture). **Bell-McCaffrey Satterthwaite DOF is threaded into the user-facing aggregated inference for hc2_bm**: per-event-time `event_study_effects[h]['p_value']/['conf_int']` use the per-coefficient contrast DOF from `_compute_cr2_bm_contrast_dof`; `overall_p_value`/`overall_conf_int` use the post-period-average contrast DOF, matching R `Wald_test(test="HTZ")$df_denom` at atol=1e-10. Without this threading the small-sample inference would silently fall back to normal-theory (df=None) — mirrors the SunAbraham aggregated-inference pattern from PR #472 and addresses the R1 CI codex P0 caught at submission time. `vcov_type` propagated to `StackedDiDResults.vcov_type`. `SurveyDesign` combined with `vcov_type != "hc1"` raises `NotImplementedError`: survey TSL/replicate-refit overrides analytical sandwich. Reject order locked: fweight/aweight check at `stacked_did.py:309` fires before the survey + non-hc1 check at `stacked_did.py:~325` (pinned by `test_aweight_plus_hc2_bm_rejected_by_stacked_did_level_guard`). `conley` rejected with a deferral message. `weight_type ∈ {"aweight","fweight"}` + `hc2_bm` continues to raise per the linalg validator's pweight-only restriction in PR #475 (`vcov_type="hc1"` supports all three weight types via the existing StackedDiD Q-weight semantics reject). Second PR of the Phase 1b standalone-estimator threading initiative (6 PRs to follow: WooldridgeDiD-OLS, CallawaySantAnna, ImputationDiD, TripleDifference, TwoStageDiD, EfficientDiD).
- **ContinuousDiD methodology-review-tracker promotion.** Tracker row flipped **In Progress** → **Complete** with full Verified Components / Test Coverage / Corrections Made / Deviations / Outstanding Concerns structure mirroring the HAD precedent (PR #473). REGISTRY `## ContinuousDiD` gains a formal Deviations block consolidating the boundary-knots deviation from R `contdid` v0.1.0 (`range(dose)` vs `range(dvals)` — library avoids extrapolation), the `bspline_derivative` derivative-failure `UserWarning` (Phase 2 axis-C #12), the `+inf` → `0` never-treated recoding warning, and the zero-`first_treat`+nonzero-`dose` force-zeroing warning (both axis-E silent-coercion fixes) into a single AI-review-recognized labeled surface. R cross-language coverage for ContinuousDiD runs at relative tolerance across two surfaces: (a) **scalar parity with raw R `cont_did` / `pte_default`** at 1% on overall ATT for all 6 benchmarks and on overall ACRT for benchmarks 4-5 (benchmark 6 is event-study, scalar `overall_att` only); (b) **harmonized boundary-knot-normalized curve parity** with R-side ATT(d) / ACRT(d) reconstructed under `Boundary.knots = range(treated_doses)` (matching the library) on benchmarks 1-3 via the benchmark harness — `_run_r_contdid` does the R-side rebuild at `tests/test_methodology_continuous_did.py:333-367`, and `_compare_with_r` orchestrates the Python-vs-R comparison at `:395-459` — max ATT(d) at 1% and max ACRT(d) at 2%. NOT bit-exact (`atol=1e-8`) like HAD — the boundary-knots deviation precludes algorithmic bit-equality on aggregated dose-response curves. Surface (a) is direct raw-package parity; surface (b) is reconstructed-basis parity because raw `contdid` curves use `range(dvals)`. No source code changes, no new tests, no new docstrings — consolidation only against the existing 15 methodology tests (`tests/test_methodology_continuous_did.py`), 80 unit tests (`tests/test_continuous_did.py`), and `docs/methodology/continuous-did.md` theory note. `METHODOLOGY_REVIEW.md` ContinuousDiD row promoted **In Progress** → **Complete**.
- **`SpilloverDiD(vcov_type="conley", survey_design=...)` integration via stratified-Conley sandwich on PSU totals (Wave E.2).** Lifts the Wave E.1 `NotImplementedError` (`spillover.py:2201` upfront, `two_stage.py:217` helper-level) and adds spatial-HAC + design-based variance for the previously deferred composition. **Documented synthesis** of Conley (1999) spatial-HAC × Gerber (2026, arXiv:2605.04124) Proposition 1 Binder TSL (the Wave E.1 foundation) × Wave D Gardner GMM first-stage uncertainty correction (Butts 2021 §3.1 + Gardner 2022 §4) applied to SpilloverDiD's ring-indicator stage-2 design. No reference software combines all three ingredients on a two-stage influence function. **Mechanical composition (panel-aware):** preserves the library's existing `conley_lag_cutoff = 0` semantic at `diff_diff.conley._compute_conley_meat` ("within-period spatial only — exclude cross-period spatial pairs") by looping over periods. For each period `t`, SpilloverDiD's per-obs Hájek-weighted Wave D IF `psi_i` is aggregated to per-period PSU totals `S_psu_t[g] = sum_{i in PSU g, time t} psi_i` (via `np.add.at`); per-PSU spatial centroids are panel-constant (mean of per-observation `conley_coords` within each PSU, vectorized `np.add.at` sums / `np.bincount` counts); for each stratum the within-stratum sandwich is `M_h_t = (1 - f_h) * n_h/(n_h-1) * sum_{j,k in PSUs_h} K(d(centroid_j, centroid_k) / conley_cutoff_km) * (S_psu_t[j] - S_bar_h_t)(S_psu_t[k] - S_bar_h_t)'`, where K is the Bartlett kernel (SpilloverDiD currently exposes Bartlett only and hardcodes it; the survey helper accepts `"uniform"` too but exposing that on the SpilloverDiD constructor is a separate follow-up) and `d` is haversine / euclidean / callable per `ConleyMetric`. Cross-stratum kernel weights are exactly zero by sampling design (strata are independence partitions). Total meat is `sum_t sum_h M_h_t`. Cross-period spatial pairs are excluded by construction — the per-period loop matches the library's panel Conley contract exactly. **Reduction semantics (load-bearing for tests):** the orchestrator's panel-aware meat equals `sum_t` of per-period within-stratum stratified-Conley sandwiches on per-period PSU totals (pinned at `tests/test_spillover.py::TestSpilloverDiDWaveE2ConleySurveyDesign::test_b_panel_aware_per_period_sum_invariant`); single stratum (H = 1, FPC = inf) reduces to `sum_t` plain Conley sandwich on per-period PSU totals (NOT on time-collapsed totals). **Implementation:** new `_compute_stratified_conley_meat_from_psu_scores` helper in `diff_diff/survey.py` (parallel to existing `_compute_stratified_meat_from_psu_scores` 3-tuple `(meat, variance_computed, legitimate_zero_count)` contract; per-stratum loop replaces the inner `centered.T @ centered` with `_compute_conley_meat(scores=centered, coords=psu_coords_h, ...)` in cross-sectional mode); new dispatch wrapper `_compute_stratified_conley_meat` in `diff_diff/two_stage.py` (parallel to existing `_compute_binder_tsl_meat`, performs per-obs Psi → PSU aggregation + centroid derivation + dispatch to survey helper, intentionally drops `cluster_ids` at the dispatch boundary — see Restrictions). `_compute_gmm_corrected_meat` conley branch extended with `if resolved_survey is not None` routing to the new wrapper; the `resolved_survey is None` branch is bit-identical to Wave D. **Singleton-stratum `lonely_psu="adjust"` parity:** the survey helper mirrors the Binder helper's `continue` to skip the FPC scale on singleton strata (with `n_h = 1` the scale `n_h / (n_h - 1)` would divide by zero); the degenerate one-PSU kernel `K = [[K(0)]] = [[1.0]]` reduces to `centered.T @ centered`, matching Binder's singleton-adjust output. **Saturated `df_survey = 0` NaN-fail:** mirrors Wave E.1 (`_compute_stratified_conley_meat` returns NaN meat with `UserWarning` template "Wave E.2 stratified-Conley sandwich: df_survey = 0..." so callers can `pytest.warns(UserWarning, match="Wave E.2 stratified-Conley")`). **Public surface restrictions:** replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises `NotImplementedError` (inherits Wave E.1 gate; per-replicate full refit is separate follow-up scope); `cluster=<col> + survey_design.psu + vcov_type="conley"` coerces `cluster=<col>` to PSU per Wave E.1's warn-and-use-PSU pattern (the Conley cluster product kernel becomes a no-op after PSU aggregation, so `cluster_ids` is intentionally not threaded into the inner Conley kernel call — every PSU is its own cluster post-aggregation, which would zero all cross-PSU pairs); LinearRegression-side `vcov_type="conley" + survey_design=` gate at `diff_diff/linalg.py:2853` remains (separate Bertanha-Imbens 2014 weighted-Conley "Phase 5" roadmap, not Wave E); DiagnosticReport routing for `SpilloverDiDResults(vcov_type="conley", survey_design=)` requires `_APPLICABILITY` / `_PT_METHOD` registration (separate Wave F PR). **Tests:** new `TestSpilloverDiDWaveE2ConleySurveyDesign` and `TestSpilloverDiDWaveE2ConleySurveyDesignEventStudy` classes in `tests/test_spillover.py` (bit-identical no-survey fallback; panel-aware per-period sum invariant on the orchestrator + helper composition; hand-computation methodology anchor; single-stratum ≡ plain Conley on PSU totals; cross-stratum independence as a unit test on the survey helper with interleaved cross-stratum centroids; Binder vs Conley singleton-adjust FPC skip parity; lonely-PSU sensitivity across three modes; FPC large ≡ no-FPC and FPC = n_h zeros stratum; saturated NaN-fail with `pytest.warns(match="Wave E.2 stratified-Conley")`; replicate-weight + non-pweight rejections; cluster warn-and-use-PSU; fit idempotency; `finite_mask` survey-array subsetting; no-PSU coverage — weights-only `SurveyDesign(weights=...)`, strata-only `SurveyDesign(weights=..., strata=...)`, and a per-period re-index unit invariant pinning that no cross-period spatial pairs leak into the meat on implicit-PSU layouts; event-study path on both `is_staggered=True`/`False` branches per `feedback_cohort_loop_trigger_cache_both_branches`; drift goldens at `rtol=1e-12 / atol=1e-14`). The pre-existing `tests/test_spillover.py::test_fit_conley_plus_survey_design_not_implemented` Wave E.1-era gate-assertion test is removed (replaced by the positive-path tests above). Wave E.1 entry's "Public surface restrictions" bullet updated to past-tense the conley+survey gate reference.
- **`SpilloverDiD(vcov_type="conley", conley_lag_cutoff > 0, survey_design=...)` panel-block composition via spatial + serial Bartlett HAC (Wave E.2 follow-up).** Lifts the Wave E.2 upfront `NotImplementedError` at `spillover.py:2210` and extends the panel-aware stratified-Conley sandwich (cross-sectional `lag=0` shipped in Wave E.2) with a within-PSU serial Bartlett HAC over time (Newey-West 1987 form). **Documented synthesis** of Wave E.2's panel-aware Conley × Binder TSL × Wave D Gardner GMM composition with Newey-West (1987) serial Bartlett HAC, matching the no-survey panel-block decomposition at `diff_diff.conley._compute_conley_meat` (Conley 1999 + Newey-West 1987 separable form, NOT Driscoll-Kraay 2D-HAC). The composition is `meat = meat_spatial + meat_serial` with disjoint index sets: spatial is the shipped Wave E.2 per-period within-stratum sandwich on PSU totals; serial is the new within-PSU sum `meat_serial_h = FPC_h_panel * sum_{g in stratum h} sum_{|t-s| <= L, t != s, both periods present for PSU g} (1 - |t-s|/(L+1)) * S_centered_t[g] @ S_centered_s[g]'` where `S_centered_t[g] = S_psu_t[g] - S_bar_h(g)_t` is per-period within-stratum centered (Binder TSL form — matches the spatial helper's centering exactly), and `|t-s|` uses panel-wide dense time codes (mirrors `conley.py:940` R deviation that matches R `conleyreg::time_dist`). Serial Bartlett kernel is hardcoded regardless of `conley_kernel` (the user-selected kernel governs the spatial term only). **FPC convention (panel-wide per-stratum):** standalone Newey-West composition on stratified clusters — the serial sum is a PANEL-level construct, so the cluster set is the panel-wide PSU set in stratum h; FPC denominator uses `n_h_panel = |unique PSUs in stratum h across active sample|`, NOT per-period `n_h_t`. The spatial term keeps its existing per-period FPC unchanged. For balanced panels the two FPC denominators converge; the difference surfaces under unbalanced panels. Citation chain: Binder (1983) for the FPC factor form, Gerber (2026) Prop 1 for the Binder TSL composition with two-stage IF, Newey-West (1987) for the serial Bartlett kernel weights, Conley (1999) for the spatial kernel and panel-block decomposition (deliberately NOT by analogy to the Binder helper's cross-sectional per-stratum FPC convention). **Centering asymmetry vs no-survey reference:** the no-survey panel-block path at `conley.py:949-965` uses RAW scores for the serial term because it assumes `E[scores] = 0` under correct specification; the survey-weighted Binder TSL form centers explicitly (textbook stratified-cluster sandwich). Using raw scores in the survey case would inflate variance by twice the squared per-period stratum mean and would NOT reduce to the cross-sectional Wave E.2 form at lag=0. **Reduction semantics (load-bearing for tests):** `conley_lag_cutoff = 0` or `None` produces bit-identical ATT and scalar SE to shipped Wave E.2 (orchestrator skips the serial helper invocation; the spatial loop + saturation guard + new PSD/finite guard still run on the spatial-only meat). `assert_array_equal` regression pin at `test_a` covers user-visible ATT + scalar SE; `test_a2` mock-spy independently asserts the serial helper is NOT invoked at `lag=0`. The meat matrix itself is not exposed on `SpilloverDiDResults`, so full meat-matrix equality is implied (not asserted); `conley_time is None` or `T = 1` short-circuits the serial helper to zero meat (degenerate panel-block, not a saturation diagnostic); single-stratum H=1 with FPC=inf reduces to Newey-West Bartlett HAC on per-period within-stratum-CENTERED per-PSU score sequences (NOT raw scores — Binder TSL centering is retained at H=1; the panel-wide `G/(G-1)` survey factor replaces FPC); bandwidth → 0 with L > 0 reduces the spatial term to per-period within-stratum HC sandwich while leaving the serial term unchanged (separable form). **Singleton-stratum `lonely_psu="adjust"` panel-wide mean asymmetry:** for the serial helper, `_global_psu_mean` is the panel-wide mean of per-period PSU totals (averaged over all `(g, t)` with `present[g, t]`), NOT the per-period within-stratum mean used by the spatial helper. The `continue`-skip-FPC pattern matches the spatial helper at `survey.py:2007-2017` to avoid divide-by-zero on `n_h_panel = 1`. **Restrictions inherited:** replicate-weight variance still raises `NotImplementedError`; DiagnosticReport routing for the panel-block case is queued for the same Wave F follow-up as the cross-sectional case. **Implementation:** new sibling helper `_compute_stratified_serial_bartlett_meat` in `diff_diff/two_stage.py` (parallel to the Wave E.2 spatial orchestrator; ~200 LoC). Orchestrator `_compute_stratified_conley_meat` signature extended with `conley_lag_cutoff: Optional[int] = None`; spatial loop unchanged; serial helper called after spatial loop when `conley_lag_cutoff > 0`; saturation NaN-fail accounting merges both terms' `(variance_computed, legitimate_zero)` flags into the same template (`UserWarning` "Wave E.2 stratified-Conley sandwich: df_survey = 0..." covers both spatial-only and panel-block cases). Dispatch in `_compute_gmm_corrected_meat` conley branch threads `conley_lag_cutoff` through; the `cluster_ids` non-threading rationale (post-PSU-aggregation every PSU is its own cluster) still applies to the new serial branch. Spillover-side gate at `spillover.py:2210-2230` (Wave E.2-era `NotImplementedError` for `lag > 0 + survey`) deleted; gate rationale comment replaced with shipped-behaviour note. Stage-1 weighted FE solver, `finite_mask` survey-array subsetting, `df_survey` threading, bread weighting, and `SpilloverDiDResults` survey metadata are all inherited UNCHANGED — Psi construction is bit-identical regardless of vcov_type or lag. **Tests:** new `TestSpilloverDiDWaveE2FollowupConleySurveyLagCutoff` and `TestSpilloverDiDWaveE2FollowupConleySurveyLagCutoffEventStudy` classes in `tests/test_spillover.py`; existing `test_j0_panel_conley_lag_cutoff_rejected_under_survey` (Wave E.2-era gate-assertion) DELETED.
- **HeterogeneousAdoptionDiD methodology-review-tracker promotion.** New `tests/test_methodology_had.py` (6 classes, 36 tests) with paper-equation-numbered Verified Components walk-through against de Chaisemartin, Ciccia, D'Haultfœuille & Knau (2026) arXiv:2405.04465v6 (Equations 3 / 7 / 11 / 18 / 29 and Theorems 1 / 3 / 4 / 7): Design 1' MC recovery on both the zero-boundary DGP AND a nonzero-boundary-intercept DGP (`ΔY = c + β·D + ε` with `c != 0`) so the `att = (mean(ΔY) − τ_bc) / mean(D)` subtraction term is verified explicitly, N(0,1) coverage at `n_replicates=200`, mass-point Wald-IV closed-form equivalence at `atol=1e-9`, QUG limit-law distributional match at KS-stat ≤ 0.05 (n_draws=5000), Yatchew-HR paper-literal `σ²_diff = 1/(2G)` normalization lock, joint Stute pre-trends + homogeneity H0 fail-to-reject on both surfaces and H1 reject for joint homogeneity under a nonlinear DGP, and library-deviation locks (equal-weighting via selective low-dose-region replication, sup-t bootstrap gating, staggered-timing fail-closed `ValueError`). Added "Non-testable assumptions (paper Section 3.1.2)" Notes block to `HeterogeneousAdoptionDiD` class docstring + "Scope (what this test does NOT cover)" clauses to `qug_test` / `stute_test` / `yatchew_hr_test` / `did_had_pretest_workflow` Notes sections explicitly stating that the pre-tests verify ADJACENT assumptions (Assumption 4 / 7 / 8) and CANNOT test Assumptions 5 or 6. Phase-4 validation-harness items (Pierce-Schott 2016 Figure 2 replication, Table 1 coverage-rate reproduction across 3 DGPs × G ∈ {100, 500, 2500}) waived with documented rationale: R parity at `atol=1e-8` in `tests/test_did_had_parity.py` (3 DGPs × 5 method combos, bit-exact via `rtol=0`) is a strictly stronger anchor than coverage-rate Monte Carlo, and the paper itself self-acknowledges (Section 5.2) that NP estimators are too noisy to be informative on the LBD-restricted PNTR panel. REGISTRY HAD section gains a consolidated Deviations block (5 entries with framing header) and closes 2 of 3 unchecked Implementation Checklist items — the staggered-timing fail-closed `ValueError` and the Assumption 5/6 non-testability documentation; the `covariates=` Theorem 6 follow-up and the extensive-margin / "consider running standard DiD" warning both remain explicitly tracked in `TODO.md` as Low-priority follow-ups rather than claimed-closed. `dechaisemartin-2026-review.md:182-194` requirements checklist boxes the Phase 1a/1b/1c implementation-status closures + the Assumption 5/6 documentation + the staggered-timing closures; the extensive-margin item is acknowledged as partial (zero-dose `UserWarning` exists in `qug_test`; main-`fit()` "consider standard DiD" recommendation is the TODO follow-up). `METHODOLOGY_REVIEW.md` HAD row promoted **In Progress** → **Complete**.
- **WLS-CR2 Bell-McCaffrey gates lifted (clubSandwich port).** `vcov_type="hc2_bm" + weights` (both one-way and cluster-robust) is now supported via `compute_robust_vcov` / `solve_ols` / `LinearRegression`, matching `clubSandwich::vcovCR(..., type="CR2")` + `coef_test(test="Satterthwaite")$df_Satt` and `Wald_test(test="HTZ")$df_denom` at `atol=1e-10` on six new weighted scenarios in `benchmarks/data/clubsandwich_cr2_golden.json` (vcov + non-noise-floor per-coefficient DOF + compound-contrast DOF match; high-leverage FE-dummy coefficients fall at the noise floor and are suppressed to NaN per the precision-limit note below) (`tests/test_methodology_wls_cr2.py`). `LinearRegression` now populates per-coefficient `_bm_dof` on the weighted-cluster path (previously skipped), so `get_inference()` reports the correct Satterthwaite DOF instead of falling back to `n-k`. The lift applies to the **analytical** vcov surface; estimator-level `survey_design=` paths continue to use the Taylor-series linearization survey variance (which takes precedence over the analytical sandwich). Weight-type restriction: the port matches the `pweight` (sampling-weight) convention only — `aweight` and `fweight` raise `NotImplementedError` (separate methodology task); CR1 (`vcov_type="hc1"`) supports all three weight types. Critical implementation note: the diff-diff form matches **clubSandwich's specific algebra** (R source: `CR-adjustments.R::CR2`, `clubSandwich.R::vcov_CR`, `coef_test.R::Satterthwaite_df`), NOT a textbook Pustejovsky-Tipton (2018) §3.3 transform-once derivation — clubSandwich uses W (not √W) in the hat matrix, W² in the bias term, and unweighted residuals in the score construction. The Satterthwaite DOF uses the full `get_arrays.R::get_GH` H1/H2/H3 array construction; the simpler `(tr B)² / tr(B²)` form (which is exact for unweighted) diverges from clubSandwich by 0.5-30% on weighted designs. Unweighted CR2-BM behavior is bit-equal to prior at `atol=1e-14` (regression-safe via `TestUnweightedRegressionStillBitEqual` + `TestDOFFormulaDualPathEquivalence`). **Known precision limit**: the Satterthwaite DOF formula for high-leverage FE-dummy contrasts (where the contrast vector projects to near-zero on the design) is at the float64 noise floor; the helper detects this regime via two criteria applied union-wise — **(a) batch-relative** (per-contrast max|P| below `1e-10 ×` the largest contrast's max|P|; catches FE-dummies in a per-coefficient sweep), and **(b) absolute single-contrast safe** (per-contrast max|P| below `(EPS × n × k × max(bread_inv_scale, 1))²`; catches single-contrast calls like MPD avg_att where no batch reference exists) — and returns NaN with a `UserWarning` rather than ship BLAS-implementation-dependent DOF (15-30% disagreement vs R's clubSandwich on those specific contrasts). VCOV/SE remain valid; only the affected coefficients' DOF (and any t-test or CI that depends on it) is suppressed. The weighted clustered CR2 path also physically filters `weights > 0` rows before per-cluster computations to preserve subpopulation invariance (zero-weight rows inside otherwise positive-weight clusters were previously entering the CR2 adjustment matrices on the row side; fixed in PR #475 round 2). clubSandwich version pin: ≥ 0.7.0. Closes TODO.md rows 104-105 (Gates 4 and 5).
- **SunAbraham `vcov_type` parameter (Phase 1b PR 1/8).** `SunAbraham(vcov_type=...)` now accepts `{"classical","hc1","hc2","hc2_bm"}` (defaults to `"hc1"`, which preserves prior behavior bit-equally - SA historically hard-coded HC1). Auto-cluster-at-unit dropped when the user opts into explicit `vcov_type="hc2"` or `vcov_type="classical"` (one-way only); preserved for `"hc1"` and `"hc2_bm"`. When `vcov_type in {"classical","hc2","hc2_bm"}`, `_fit_saturated_regression` auto-routes to a full-dummy saturated design (mirrors TWFE Gate 1 from PR #469): FWL preserves cohort coefficients but not the hat matrix, so HC2 leverage and Bell-McCaffrey Satterthwaite DOF must be computed on the full FE projection. Empirically matches R `lm()` summary classical SE, `sandwich::vcovHC(type="HC2")`, and `clubSandwich::vcovCR(..., type="CR2")` + `coef_test()$df_Satt` at atol=1e-10 (cohort SE and BM DOF pinned in `tests/test_methodology_sun_abraham.py`). For `vcov_type="hc2_bm"`, the user-facing aggregated inference (`event_study_effects[e]['p_value']`/`['conf_int']`, `overall_p_value`/`overall_conf_int`) uses CR2 Bell-McCaffrey contrast DOF — matches `clubSandwich::Wald_test(test="HTZ")$df_denom` at atol=1e-10 (mirrors PR #465's `_compute_cr2_bm_contrast_dof` pattern for MultiPeriodDiD's post-period-average ATT). `vcov_type` is now propagated to `SunAbrahamResults.vcov_type` for downstream introspection. `SurveyDesign` (any kind — analytical weights, stratified, PSU, or replicate-weight) combined with `vcov_type in {"classical","hc2","hc2_bm"}` raises `NotImplementedError`: the survey-design TSL (or replicate-weight refit) variance overrides the analytical sandwich family, and the auto-cluster guard for one-way families would silently downgrade unit-level PSUs to per-observation PSUs. Use `vcov_type="hc1"` (default) for survey designs. `conley` rejected at `__init__` with a deferral message (would require threading 6+ `conley_*` params through the saturated regression call). **Deviation from R:** SA's within-transform HC1 SE differs from `fixest::sunab()` by ~1-2% (~2e-3 absolute) on typical panel sizes due to a different `(n-k)` finite-sample correction (fixest counts absorbed FE in k_total; SA's `solve_ols` counts only within-transformed columns); the IW aggregation step is otherwise identical (pinned at atol=5e-3, tracked in TODO.md). First PR of the Phase 1b standalone-estimator threading initiative (7 PRs to follow: StackedDiD, WooldridgeDiD-OLS, CallawaySantAnna, ImputationDiD, TripleDifference, TwoStageDiD, EfficientDiD).
- **PreTrendsPower R `pretrends` parity goldens (PR-C closes PR-B's deferred R-parity row).** JSON goldens at `benchmarks/data/r_pretrends_golden.json` generated from the committed `benchmarks/R/generate_pretrends_golden.R` script against `jonathandroth/pretrends` commit `122731d082` (package version 0.1.0, R 4.5.2). 4 fixtures cover regular K=3 grid (`uniform_3_pre_periods_no_anticipation`), irregular K=3 grid `[-5,-3,-1]` (`irregular_pre_periods` — locks the PR-B Step 4 γ-unit linear-weight fix), anticipation-shifted K=4 grid (`anticipation_shifted`), and K=1 closed form (`single_pre_period_closed_form` — Roth Proposition 2 univariate truncated-normal). `TestPretrendsParityR` in `tests/test_methodology_pretrends.py` now active (4 tests): NIS power vs R `pretrends::pretrends()` at `atol=1e-4` across all 4 fixtures × 4 γ values; γ_p MDV vs R `slope_for_power()` at `atol=1e-4` across all 4 fixtures × 2 target_power values; end-to-end `fit()` on irregular grid vs R γ_p at `atol=1e-4` (locks the full `fit() → _extract_pre_period_params → _get_violation_weights → _compute_mdv_nis` chain through the public API); K=1 three-way cross-check (Python ≡ analytical truncated-normal closed form `1 - Φ(z - γ/σ) + Φ(-z - γ/σ)` at `atol=1e-7`; both within `atol=1e-4` of R). Tolerance rationale: R hardcodes `thresholdTstat.Pretest=1.96` while Python uses `scipy.stats.norm.ppf(0.975) = 1.959963984540054` (`dz ≈ 3.6e-5`); R `slope_for_power` uses `uniroot(tol = .Machine$double.eps^0.25 ≈ 1.22e-4)` versus Python `brentq(xtol=2e-12)`; the inverse-solver tolerance gap dominates γ_p, and `mvtnorm::pmvnorm` (R) vs `scipy.stats.multivariate_normal.cdf` (Python) Genz-Bretz randomized-lattice differences bound the K=4 NIS power gap at ~5e-5. `METHODOLOGY_REVIEW.md` PreTrendsPower row promoted `**Complete** (R parity pending)` → `**Complete**`. Roth (2022) paper review's `R \`pretrends\` package version pin (provisional)` Gaps bullet struck. Closes the PR-C TODO row.
- **`SpilloverDiD(survey_design=...)` integration on HC1 / CR1 paths via Binder TSL (Wave E.1).** Lifts the Wave B/C/D upfront `NotImplementedError` and adds design-based variance for `vcov_type ∈ {"hc1"}` plus `cluster=<col>` (CR1). **Documented synthesis** of Gerber (2026, arXiv:2605.04124) Proposition 1 — Binder Taylor Series Linearization for IF representations of smooth functionals; explicitly derived for TwoStageDiD in the paper's Appendix — composed with the Wave D Gardner GMM first-stage uncertainty correction (Butts 2021 §3.1 + Gardner 2022 §4) applied to SpilloverDiD's ring-indicator stage-2 design. No reference software combines all ingredients. **Mechanical composition:** SpilloverDiD's per-obs Wave D IF `psi_i = gamma_hat' * X_{10,i} * eps_{10,i} - X_{2,i} * eps_{2,i}` (with survey weights threaded through `gamma_hat` solve, eps construction, and bread inversion via Hájek normalization) is aggregated to PSU totals and passed to the audited `_compute_stratified_meat_from_psu_scores` Binder TSL meat helper. Stage-1 FE estimation extends `_iterative_fe_subset` with a `weights=` kwarg implementing WLS-FE via weighted bincount (numerator `bincount(w*resid)` / denominator `bincount(w)`); the `weights is None` path is bit-identical to the Wave B / C / D unweighted bincount. **Degrees of freedom:** t-distribution lookup uses `ResolvedSurveyDesign.df_survey` (4-way branch: PSU+strata → `n_PSU - n_strata`; PSU only → `n_PSU - 1`; strata only → `n_obs - n_strata`; neither → `n_obs - 1`), threaded through all four `safe_inference` call sites (aggregate `tau_total`, per-ring `delta_j`, event-study per-event-time `tau_k` / `delta_jk`, scalar `att` lincom). **Survey-array subsetting:** when `finite_mask` drops baseline-treated rows, `survey_weights` and `ResolvedSurveyDesign.{weights, strata, psu, fpc, replicate_weights}` are subsetted in parallel; `n_psu`, `n_strata`, and `survey_metadata` are recomputed (mirrors `TwoStageDiD.fit:567-601`). **Cluster + survey resolution:** when `cluster=<col>` and `survey_design.psu` are both supplied with different groupings, a `UserWarning` fires and PSU wins (mirrors `_resolve_effective_cluster` at `survey.py:1253-1275`; TwoStageDiD parity). When `cluster=<col>` is supplied without `survey_design.psu`, the cluster column is injected as the effective PSU via `_inject_cluster_as_psu`, which now honors `SurveyDesign.nest`: under `nest=False`, cluster labels must be globally unique across strata (raises if they repeat, matching the explicit-PSU resolver's contract). **Saturated `df_survey = 0` NaN-fail:** when `lonely_psu="remove"` removes all strata (singleton PSUs), the meat helper returns `(_, var_computed=False, legit_zero=0)` and SpilloverDiD's Wave E.1 path returns NaN meat with a `UserWarning` matching `"df_survey"` so callers can `pytest.warns(UserWarning, match="df_survey")`. This is a **departure from TwoStageDiD** (`two_stage.py:2003-2005`) which currently NaN-fails SILENTLY; Wave E.1 surfaces the diagnostic per `feedback_no_silent_failures`. **Subpopulation limitation (closed in Wave E.3 — see top entry above):** the original Wave E.1 ship had `SurveyDesign.subpopulation()`-derived designs with zero-weight padding rows that lose stage-1 FE support physically removed by `finite_mask`, so `n_psu` / `df_survey` / Binder centering reflected the reduced fit sample rather than the full domain design. This was tracked as the Wave E.3 follow-up and is now closed by the Wave E.3 entry at the top of the [Unreleased] section (full-design retention via score_pad_mask zero-pad). **Public surface restrictions:** `vcov_type="conley" + survey_design=` originally raised `NotImplementedError` pointing at planned Wave E.2; lifted in the Wave E.2 entry above (stratified-Conley sandwich on PSU totals). Replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises `NotImplementedError` — per Gerber (2026) Appendix A, the IF-reweighting shortcut does not apply to TwoStageDiD-class estimators because `gamma_hat` is weight-sensitive; correct support requires per-replicate full re-fit and is queued as a follow-up; non-pweight (`weight_type ∈ {"fweight", "aweight"}`) raises `ValueError` (the Binder TSL assumes probability weights). **Implementation:** `_compute_gmm_corrected_meat` extended with `survey_weights` + `resolved_survey` kwargs at `diff_diff/two_stage.py:56` (TYPE_CHECKING forward reference for `ResolvedSurveyDesign` to avoid circular import); new module-level helper `_compute_binder_tsl_meat` at `diff_diff/two_stage.py` wraps `_compute_stratified_meat_from_psu_scores` with implicit per-obs PSU synthesis for no-PSU survey designs + the Wave E.1 NaN-fail + warning; `_iterative_fe_subset` weighted path at `diff_diff/spillover.py:1382` (in-place extension, bit-identical fallback, positive-weight identification gate); `_inject_cluster_as_psu` honors `nest` (shared survey-helper fix that also benefits TwoStageDiD); `ResolvedSurveyDesign` gains a `nest` field propagated through all 5 construction sites. `SpilloverDiDResults` extended with `survey_metadata`, `n_psu`, `n_strata` fields at `diff_diff/results.py`. **Tests:** new `TestSpilloverDiDWaveE1SurveyDesignHc1` (17 tests: bit-identity fallback, Binder TSL hand-check uniform + non-uniform weights, lonely_psu modes, FPC degenerate limits ×3, saturated NaN-fail with `pytest.warns(match="df_survey")`, cluster+survey warn-and-use-PSU, no-PSU regressions (weights-only, weights+strata, cluster-without-PSU, cluster overlap with nest=False/True), zero-weight Omega_0 exclusion + all-zero raises, replicate-weight + non-pweight + Conley+survey rejections, fit idempotency, finite_mask subsetting) and `TestSpilloverDiDWaveE1SurveyDesignEventStudy` (7 tests: event-study + survey on both `is_staggered` branches with `df_survey` lincom verification, distinguishability between survey-share and sample-share lincom rules via manual reconstruction with cohort-correlated weights + non-constant tau_k, aggregate-vs-event-study parity, drift goldens, subset-path invariant). Wave B/C/D bullets below are unchanged; this entry replaces the pre-Wave-E.1 `survey_design=` rejection.

## [3.4.0] - 2026-05-19

### Added
- **`TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"})` now supported** (`diff_diff/twfe.py:155`). Lifts Gate 1 of the six HC2/HC2-BM `NotImplementedError` gates — the last absorbed-FE gate (DiD-absorb shipped earlier, MPD-absorb shipped earlier, MPD cluster+contrast-DOF shipped earlier in this release). Unlike DiD / MPD, TWFE has no `absorb=` / `fixed_effects=` parameter to swap (unit + time FEs are baked into the estimator's identity), so the same auto-route trick isn't applicable. Instead, `TwoWayFixedEffects.fit()` bypasses the within-transform when `vcov_type in {"hc2","hc2_bm"}` and stacks the full-dummy design `[intercept, treated×post, covariates, factor(unit), factor(time)]` explicitly, then runs OLS through the standard `solve_ols` path so the leverage correction `h_ii = x_i' (X'X)^{-1} x_i` and CR2 Bell-McCaffrey adjustment `A_g = (I - H_gg)^{-1/2}` compute on the full FE projection (FWL preserves coefficients and residuals but NOT the hat matrix). Verified at `atol=1e-10` vs `lm(y ~ treat_post + factor(unit) + factor(post)) + sandwich::vcovHC(type="HC2")` for HC2, vs `clubSandwich::vcovCR(cluster=seq_len(n), type="CR2") + coef_test()$df_Satt` for the singleton-cluster one-way HC2-BM Satterthwaite DOF, and vs `vcovCR(cluster=unit, type="CR2")` for the auto-cluster CR2-BM path (new `twfe_two_period` scenario in `benchmarks/data/clubsandwich_cr2_golden.json`). **Auto-cluster default:** TWFE's unit auto-cluster is preserved on `hc2_bm` (routes to CR2-BM at unit) and on `hc2 + wild_bootstrap` (the bootstrap consumes the cluster structure for resampling regardless of the analytical sandwich choice); dropped on explicit `hc2 + analytical` to match the one-way contract (the linalg validator rejects `hc2 + cluster_ids`). **User-visible surface change** (matches the DiD-absorb / MPD-absorb disclosures above): under `vcov_type in {"hc2","hc2_bm"}`, `result.coefficients`, `result.vcov`, `result.residuals`, `result.fitted_values`, and `result.r_squared` reflect the full-dummy fit rather than the within-transformed reduced fit (FE-dummy entries are included alongside the `"ATT"` key; `r_squared` is computed on the un-demeaned outcome; residuals / fitted are on the original scale; `len(result.coefficients) == result.vcov.shape[0]` invariant upheld). `result.att`, its SE, and analytical inference are unchanged (FWL-equivalent). HC1 / CR1 / Conley / classical paths remain on the within-transform. **Survey-design scope** (mirrors DiD-absorb): when `survey_design=` is supplied, the existing survey variance path (Taylor-series linearization or replicate-weight variance) takes precedence over the analytical HC2/HC2-BM sandwich; the full-dummy build only changes FE handling. **Rejected combos:** `vcov_type in {"hc2","hc2_bm"}` + replicate-weight survey designs (BRR / Fay / JK1 / JKn / SDR) raises `NotImplementedError` at `twfe.py:~233` because the replicate path re-demeans per replicate, which doesn't compose with the full-dummy build (would require per-replicate full-dummy refit); workaround: use `vcov_type="hc1"` for replicate-weight CR1. `hc2_bm + weights` remains blocked at the linalg validator (same gate as Gates 4-5 — weighted CR2 variants). New tests: `tests/test_estimators_vcov_type.py::TestFitBehavior` (9 tests: rejection flip → behavioral; refactor regression vs `DifferenceInDifferences(fixed_effects=[unit, time])` at `atol=1e-12`; auto-cluster default coverage on `hc2_bm`; explicit `hc2 + analytical` no-auto-cluster; `hc2 + wild_bootstrap` auto-cluster preserved; `hc2 / hc2_bm + replicate` rejection; always-treated unit finite ATT; coefficients-vs-vcov alignment invariant); `tests/test_methodology_twfe.py::TestTWFEHC2RParity` (3 R-parity tests at `atol=1e-10`).
- **Agent-discoverability contract test (`tests/test_agent_discoverability.py`).** New static-snapshot test pinning the agent-facing surface introduced by PR #464: `__all__` membership of `agent_workflow` / `profile_panel` / `get_llm_guide` / `practitioner_next_steps` / `BusinessReport`; `dir(diff_diff)` head-first ordering against `_AGENT_FACING_ORDER` (catches drift in the `_OrderedName` `__lt__` ordering trick); `_OrderedName` `isinstance(_, str)` + str-method compatibility; `dir()` full-namespace + `inspect.getmembers` parity; top-level `__doc__` first-paragraph mention of `agent_workflow` + named references to the 5-step workflow primitives; `agent_workflow()` script content references each downstream helper by name; canonical estimator class names (CallawaySantAnna, ContinuousDiD, HeterogeneousAdoptionDiD, etc.) remain importable. No live API calls; runs in the default pytest suite. Closes [issue #461](https://github.com/igerber/diff-diff/issues/461) (snapshot variant — live-agent regression test deferred to a separate follow-up that depends on causal-llm-eval packaging its harness). Also closes the `__dir__()` contract-test row from `TODO.md` that PR #464 deferred here.
- **`diff_diff.agent_workflow(df, unit=..., time=..., treatment=..., outcome=...)` — stateless orchestrator for LLM-agent discoverability** (`diff_diff/agent_workflow.py`). Prints (and returns as dict) a copy-pasteable 5-step workflow with the caller's column names templated in: `profile_panel` → `get_llm_guide("autonomous")` → `<Estimator>(...).fit(df, ...)` → `practitioner_next_steps(result)` → `BusinessReport(result).full_report()`. The function calls nothing internally and does not inspect `df`; it is a guided tour, not a router. Surfaces the canonical workflow primitives (`profile_panel`, `get_llm_guide`, `practitioner_next_steps`, `BusinessReport`) that cold-start agent dry-passes at [igerber/causal-llm-eval](https://github.com/igerber/causal-llm-eval) showed agents practically never reach for on their own. Output structure: `{"profile_call", "guide_call", "fit_candidates", "validation_calls", "reporting_call", "script"}`; `fit_candidates` is a flat list of estimator/diagnostic class names referenced in the workflow patterns (each must remain importable on `diff_diff`, locked by `tests/test_agent_workflow.py::test_fit_candidates_all_importable`). Closes [issue #460](https://github.com/igerber/diff-diff/issues/460).
- **Top-level `__doc__` rewritten to lead with the agent workflow** (`diff_diff/__init__.py`). `help(diff_diff)` now opens with the `agent_workflow(df, ...)` recommendation as the first non-blank paragraph; `get_llm_guide("full")` and `get_llm_guide("practitioner")` pointers preserved for the existing `tests/test_guides.py::test_module_docstring_mentions_helper` guard.
- **`dir(diff_diff)` now surfaces agent-facing entrypoints first** via a module-level `__dir__()` override paired with a small `_OrderedName(str)` subclass that subverts CPython's unconditional alphabetic sort (PyList_Sort respects `__lt__` on the elements). Agent-facing names (`agent_workflow`, `profile_panel`, `get_llm_guide`, `practitioner_next_steps`, `BusinessReport`, `DiagnosticReport`) appear at the head of the list; the remainder stays alphabetic via the `str.__lt__` fallback. The underlying `__all__` membership is **unchanged** and `from diff_diff import *` semantics are unaffected (driven by `__all__`, not `dir()`). Elements are `isinstance(x, str)` and compatible with `inspect.getmembers`, dict-key lookup, f-strings, and standard `str` methods; tooling that re-sorts via `sorted(dir(diff_diff))` will see priority order (use `sorted(dir(diff_diff), key=str)` to recover plain alphabetic if needed). Internal: `_AGENT_FACING_ORDER` tuple is read by the new `tests/test_agent_discoverability.py` contract test (PR B). Addresses [issue #460](https://github.com/igerber/diff-diff/issues/460) item 3.
- **`MultiPeriodDiD(cluster=..., vcov_type="hc2_bm")` now supported** (`diff_diff/estimators.py:1657`). Pre-PR the combination raised `NotImplementedError` because the cluster-aware CR2 Bell-McCaffrey Satterthwaite DOF for the post-period-average ATT (`avg_att = (1/n_post) Σ_{t ≥ t_treat} β_t`) was not implemented — only the per-coefficient case existed in `_compute_cr2_bm`. New `_compute_cr2_bm_contrast_dof` helper in `diff_diff/linalg.py` generalizes the per-coefficient loop to arbitrary `(k, m)` contrast matrices using the identical Pustejovsky-Tipton 2018 Section 4 algebra; `_compute_cr2_bm` is refactored to call it with `contrasts=eye(k)` so the existing per-coefficient parity to clubSandwich's `coef_test$df_Satt` is preserved (refactor regression at atol=1e-10). `MultiPeriodDiD.fit()` extends its existing avg_att DOF block to branch on `effective_cluster_ids`: one-way `_compute_bm_dof_from_contrasts` when None, cluster-aware `_compute_cr2_bm_contrast_dof` otherwise. Cluster IDs are per-observation length `n` and are NOT subscripted by the rank-deficient column-drop mask. R parity verified at atol=1e-10 against clubSandwich's `Wald_test(constraints=matrix(c, 1), test="HTZ")$df_denom` on the new `mpd_clustered_avg_att_dof` fixture in `benchmarks/data/clubsandwich_cr2_golden.json` (Wald_test's HTZ on a 1-row constraint matrix yields the Satterthwaite t-test DOF). Per-coefficient `period_effects[t].p_value` / `conf_int` and `avg_att` `avg_p_value` / `avg_conf_int` now reflect the correct Satterthwaite DOF rather than the n-k fallback under cluster+hc2_bm. Weighted CR2-BM (`survey_design=` paths) remains a separate gate. New tests: `tests/test_linalg_hc2_bm.py::TestCR2BMContrastDOF` (4 tests: refactor regression, R-parity, shape validation, cluster-count validation); existing `test_multi_period_cluster_plus_hc2_bm_rejected` flipped to behavioral `test_multi_period_cluster_plus_hc2_bm_produces_finite_inference`.
- **PreTrendsPower: NIS box probability as the new primary test form (PR-B methodology audit, Roth 2022).** Implements Roth (2022) Section II.A-B no-individually-significant (NIS) box probability `P(β̂_pre ∈ B_NIS(Σ))` as the new default `pretest_form='nis'` on `PreTrendsPower`, `compute_pretrends_power`, and `compute_mdv`. The Wald noncentral-χ² form previously shipped as the implicit default is now opt-in via `pretest_form='wald'` and remains as a paper-supported alternative (Propositions 1+3+4 all apply — the Wald ellipsoid is convex). Computation uses `scipy.stats.multivariate_normal.cdf` with `lower_limit=` for the rectangular box probability on the centered change-of-variable `Y = β̂_pre - δ_pre ~ N(0, Σ_22)`; the MDV is solved via doubling expansion + `optimize.brentq` bisection with a 1000-cap non-convergence fallback returning `np.inf`. New private helpers `_compute_power_nis` and `_compute_mdv_nis`; the existing methods are renamed `_compute_power_wald` and `_compute_mdv_wald` with byte-identical math, and `_compute_power` / `_compute_mdv` become dispatchers on `self.pretest_form`. `power_curve()` and `PreTrendsPowerResults.power_at()` inherit the dispatch (power_at via the new persisted `pretest_form` field on the result). The `summary()` / `to_dict()` / `to_dataframe()` outputs dispatch on `pretest_form` — NIS fits print "NIS box probability: ..." instead of "Non-centrality parameter: ...".
- **PreTrendsPower: full Σ_22 routing on CS and SA event-study adapters (PR-B methodology audit, Σ_22 fidelity).** The shipped `compute_pretrends_power` adapter previously hard-coded `np.diag(ses**2)` for both `CallawaySantAnnaResults` and `SunAbrahamResults` regardless of whether the analytical event-study VCV was available, dropping the off-diagonal correlations Roth's framework relies on. PR-B routes non-bootstrap CS fits through the full `event_study_vcov` sub-block (already persisted at `staggered_results.py:126-128`) and extends `SunAbrahamResults` to also persist `event_study_vcov` + `event_study_vcov_index` constructed via the W-matrix aggregation `event_study_vcov = W @ vcov_cohort @ W.T` where W is the cohort-aggregation matrix (`|event_times| × n_interactions` sparse matrix with `W[i, j] = cohort_weights[e_i][g]` at column `j = coef_index_map[(g, e_i)]`). The new shared helper `_extract_event_study_vcov_subblock` at module level in `pretrends.py` consumes the full VCV when available with a `.index()` lookup on `event_study_vcov_index`; defensive ValueError on label mismatch. Bootstrap fits and replicate-weight survey fits clear `event_study_vcov` (mirroring the CS bootstrap-clear pattern at `staggered.py:2032-2036`) so they fall through to `diag(ses^2)` and the analytical VCV is never mixed with bootstrap/replicate SE overrides downstream. Diagonal-entry sanity check verifies that `event_study_vcov[i, i] = se(e_i)^2` matches the existing per-event-time SE computation in `_compute_iw_effects` at `atol=1e-10`. **Backwards-compatible field additions**: new `event_study_vcov` + `event_study_vcov_index` fields on `SunAbrahamResults` default to `None`, so existing consumers that don't read them see no change.
- **`PreTrendsPowerResults` now persists fitted `violation_weights` + `pretest_form` + `nis_box_probability` (PR-B Step 5).** New optional fields on the result dataclass enable `power_at(M)` to work for ALL four violation types (linear / constant / last_period / **custom**) on fresh fits, by reading the stored weights directly instead of reconstructing from `violation_type` alone. The PR-A R18 NotImplementedError silent-failure guard for `violation_type='custom'` is retained ONLY for legacy serialized results (`violation_weights=None`) — fresh fits no longer hit it.
- **Helper API: `compute_pretrends_power` and `compute_mdv` now accept `violation_weights` and `pretest_form` (PR-B Step 6).** Closes the PR-A R18 helper/class API gap that previously made `violation_type='custom'` unusable from the helper functions. Helpers now forward both new parameters to the underlying `PreTrendsPower` class. Default `pretest_form='nis'` matches the class default. All existing helper call sites in `test_pretrends.py` and `test_pretrends_event_study.py` continue to pass without changes because the form-invariance of most assertions allowed the default flip with only 3 tests needing targeted updates.
- **NEW `tests/test_methodology_pretrends.py` (PR-B Step 7).** Roth (2022) Section II.A-B paper-equation-numbered Verified Components walk-through. 8 classes, 30+ tests covering K=1 closed-form (Proposition 2 proof), NIS box probability via MC simulation cross-check, Propositions 1-4 simulation parity, linear-units γ-scale verification on regular / irregular / pandas.Period grids, custom-weight persistence regression, JSON-serializability of `to_dict`, CS/SA full-VCV adapter regression, helper API end-to-end, NIS-vs-Wald differentiation, and skip-gated `TestPretrendsParityR` stubs for PR-C R-package goldens.
- **`benchmarks/R/generate_pretrends_golden.R` (PR-B Step 12).** R generator script for the PR-C deferred goldens. Script committed with a `<PR-C-PIN>` placeholder commit reference; PR-C pins the audited `pretrends` revision, runs the script, commits the JSON goldens at `benchmarks/data/r_pretrends_golden.json`, and activates the parity tests.
- **`MultiPeriodDiD(absorb=..., vcov_type in {"hc2", "hc2_bm"})` now supported** (`diff_diff/estimators.py:1476`). Mirrors the DiD-absorb auto-route shipped earlier in this release: when `absorb=` is paired with `vcov_type in {"hc2","hc2_bm"}`, `MultiPeriodDiD.fit()` promotes the absorb columns to `fixed_effects=` internally so the existing full-dummy-design code path computes the algebraically correct vcov on the event-study design (`treated + period_X dummies + treated:period_X interactions + factor(unit)`). Verified at ~1e-10 vs `lm() + sandwich::vcovHC(type="HC2")` and `lm() + clubSandwich::vcovCR(cluster=1:n, type="CR2")` on a 5-cohort × 5-period event-study fixture (new `tests/test_estimators_vcov_type.py::TestMPDAbsorbedFERParity` against `benchmarks/data/clubsandwich_cr2_golden.json` scenario `mpd_absorbed_fe_did`). HC1/CR1 paths on `absorb=` are unchanged (no leverage term). (`TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"})` was lifted later in this same release via an inline full-dummy build — see the top entry; TWFE has no `fixed_effects=` equivalent inside the estimator, so it gets a separate full-dummy branch rather than the absorb→fixed_effects parameter swap used here.) **Behavioral note (full `MultiPeriodDiDResults` surface change under auto-route):** under the auto-route, the entire returned `MultiPeriodDiDResults` reflects the full-dummy fit rather than the within-transformed fit — `result.coefficients`, `result.vcov`, `result.residuals`, `result.fitted_values`, `result.r_squared` all include the FE-dummy entries / un-demeaned values. `result.period_effects[t].effect` / `.se` / `.p_value` / `.conf_int` and `result.avg_att` / `.avg_se` are invariant to this routing (FWL guarantee). MPD requires a time-invariant ever-treated indicator that lies in the span of the intercept and the post-auto-route unit FE dummies (the exact alias depends on the omitted FE reference category under `pd.get_dummies(drop_first=True)`, not just on "the sum of treated-cohort unit dummies"), so `solve_ols` drops one column from that collinear set under R-style rank-deficiency handling. Which specific column is dropped is pivot-order and dummy-coding dependent (in the shipped parity fixture it is a never-treated unit dummy, not the `treated` main effect itself). The per-period interaction coefficients (`treated:period_X`) and `avg_att` are identified and invariant to that choice; parity tests target those rather than the `treated` main effect. **Survey-design scope (replicate weights):** when `survey_design=` uses replicate weights, the auto-route short-circuits the absorb-refit branch at `estimators.py:1693` and routes through the standard `compute_replicate_vcov` path on the fixed full-dummy design — correct because the design does not depend on replicate weights so no per-replicate refit is needed. **Redundant time-FE skip:** when the routed (or directly-supplied) `fixed_effects` list contains the `time` column, MPD silently skips emitting `<time>_<X>` dummies for that entry because the design already absorbs the time dimension via the non-reference period dummies; without the skip, the two blocks would collide on dummy names and the `coefficients` dict would silently collapse duplicates under `var_names`-keyed construction, breaking the coefficients-vs-vcov alignment that downstream consumers rely on. This applies to both the new `absorb=` auto-route and the pre-existing `fixed_effects=[<time_col>]` invocation.
- **`DifferenceInDifferences(absorb=..., vcov_type in {"hc2", "hc2_bm"})` now supported** (`diff_diff/estimators.py:382`). Previously raised `NotImplementedError` because the HC2 leverage correction and CR2 Bell-McCaffrey DOF depend on the FULL FE hat matrix, while within-transformation (FWL) preserves coefficients and residuals but not the hat. Lift via internal auto-route: when `absorb=` is paired with `vcov_type in {"hc2","hc2_bm"}`, the fit promotes the absorb columns to `fixed_effects=` internally so the existing full-dummy-design code path computes the algebraically correct vcov. Empirically matches `lm() + sandwich::vcovHC(type="HC2")` and `lm() + clubSandwich::vcovCR(cluster=..., type="CR2")` at ~1e-10 (verified via new `tests/test_estimators_vcov_type.py::TestDiDAbsorbedFERParity` against `benchmarks/data/clubsandwich_cr2_golden.json` scenario `absorbed_fe_did`, with the R generator using the singleton-cluster CR2 trick for one-way HC2-BM Satterthwaite DOF). HC1/CR1 paths unchanged. (`MultiPeriodDiD(absorb=...)` and `TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"})` were both lifted later in this same release — see the top entries; both use the same algebra on different fit-path structures.) **Behavioral note (full `DiDResults` surface change under auto-route):** under the auto-route, the entire returned `DiDResults` reflects the full-dummy fit rather than the within-transformed fit. Specifically, `result.coefficients` and `result.vcov` include the FE-dummy entries (matching the `fixed_effects=` path), `result.residuals` and `result.fitted_values` are on the un-demeaned outcome scale, and `result.r_squared` is computed on the un-demeaned outcome (so it absorbs the FE variance and will typically be higher than the within-R²). `result.att` is invariant to this routing (FWL guarantee). Downstream consumers reading `result.att` are unaffected; consumers reading the broader result surface should expect the full-dummy values. **Survey-design scope:** the auto-route changes the FE handling (and removes the prior absorbed-FE rejection), but `survey_design=` continues to drive its own variance path (Taylor-series linearization or replicate-weight variance, per the existing survey contract) rather than the analytical HC2/HC2-BM sandwich. The auto-route is therefore methodologically meaningful for non-survey fits and for the FE-handling side of survey fits; analytical small-sample inference under `vcov_type in {"hc2","hc2_bm"}` is bypassed when a survey design is supplied.
- **`SpilloverDiD` Gardner GMM first-stage uncertainty correction across HC1 / Conley / cluster (Wave D).** Closes the documented Wave B/C "SEs biased downward by a few percent" caveat. **Documented synthesis** of Butts (2021) Section 3.1 (the IF construction for spillover-aware DiD) + Gardner (2022) Section 4 (the two-stage GMM sandwich) + Conley (1999) (the spatial kernel). No reference software combines all three — `did2s` (Butts & Gardner) implements the Gardner correction without rings or Conley; `conleyreg` and `acreg` implement Conley without the two-stage correction. Wave D is the synthesis. Applies unconditionally under `vcov_type ∈ {"hc1", "conley", "cluster"}` for both `event_study=False` AND `event_study=True`. **Formula** (Butts 2021 §3.1 + Gardner 2022 §4): `psi_i = gamma_hat' * X_{10,i} * eps_{10,i} - X_{2,i} * eps_{2,i}` where `gamma_hat = (X_10' X_10)^{-1} (X_1' X_2)` is the stage-1-projection-of-stage-2 cross-moment; meat = `Psi' K Psi` with `K` dispatched by `vcov_type` (identity for HC1, block-indicator for cluster, spatial kernel for Conley); vcov = `(X_2' X_2)^{-1} @ meat @ (X_2' X_2)^{-1}`. **Finite-sample multipliers:** `n/(n-p)` for HC1; `G/(G-1) * (n-1)/(n-p)` for cluster CR1; no multiplier for Conley (preserves `conleyreg` / Wave B convention). **Public surface:** `vcov_type="classical"` now raises `NotImplementedError` upfront (the Wave D synthesis has not been derived for the homoskedastic meat structure `sigma_hat^2 * (X_10' X_10)`); REGISTRY's "vcov_type restrictions" block updated accordingly. **Point estimates unchanged** (`tau_total`, `delta_j`, event-study `tau_k` / `delta_jk` are byte-identical to Wave B/C); SE values shift upward by 1-few percent depending on first-stage residual variance. **Implementation:** new module-level helper `_compute_gmm_corrected_meat` in `diff_diff/two_stage.py` (NOT a modification of the existing `_compute_gmm_variance` method — TwoStageDiD's path is unchanged); new module-level helper `_build_butts_fe_design_csr` in `diff_diff/spillover.py`; new module-level helper `_compute_conley_meat` in `diff_diff/conley.py` factored out of `_compute_conley_vcov` so the same kernel-application code path handles both standard sandwich (`X * residuals`) and Wave D IF outer product (`Psi`) cases. **No new public API kwarg** — the correction is unconditional. Wave D variance mode dispatch derives from the public contract: `vcov_type="conley"` → `"conley"`; `cluster=<col>` → `"cluster"` (CR1); otherwise `"hc1"`. **Wave B/C SE goldens re-pinned** at `tests/test_spillover.py::TestSpilloverDiDEventStudyBackwardCompat` (constants renamed `_WAVE_B_GOLDEN_*` → `_WAVE_D_GOLDEN_*`; pre-Wave-D references retained as commented baselines for the directional inflation invariant `_WAVE_B_UNCORRECTED_*`). **Tests:** new test classes `TestSpilloverDiDWaveDGmmCorrectedHc1Hand` (hand-derived `Psi` on a 4-unit × 3-period over-identified panel — matches at `atol=1e-12`), `TestSpilloverDiDWaveDGmmCorrectedEventStudy` (vcov shape on event-study path), `TestSpilloverDiDWaveDGmmCorrectedNanInferenceContract` (rank-deficient column propagation), `TestSpilloverDiDWaveDGmmCorrectedValidatorWiring` (Conley validator fires from the new helper), `TestSpilloverDiDWaveDGmmCorrectedFitIdempotence` (clone + repeat-fit bit-identity per `feedback_fit_does_not_mutate_config`), `TestSpilloverDiDWaveDPublicVarianceContract` (end-to-end public `cluster=<col>` CR1 routing, single-cluster rejection, classical NotImplementedError). Closes the Gardner-GMM follow-up row in `TODO.md`.
- **BaconDecomposition R parity goldens.** Closes the PR-B deferral row in `TODO.md`. JSON goldens at `benchmarks/data/r_bacondecomp_golden.json` generated from the committed `benchmarks/R/generate_bacon_golden.R` script (3 fixtures: `uniform_3groups_with_never_treated`, `two_groups_no_never_treated`, `always_treated_remapped`) against `bacondecomp 0.1.1` on R 4.5.2. `tests/test_methodology_bacon.py::TestBaconParityR` now active (4 tests, no skips): TWFE coefficient parity at `atol=1e-6` across all 3 fixtures; weights-sum parity at `atol=1e-6` across all 3 fixtures; per-component estimate + weight parity at `atol=1e-6` on the 2 non-remap fixtures **and on the 6 timing-vs-timing rows of `always_treated_remapped`** (carve-out narrowed to U-bucket rows only); plus a dedicated fold-back test (`test_always_treated_remapped_fold_back_matches_r`) that pins the **documented convention divergence** on `always_treated_remapped` (R keeps `first_treat=1` as a distinct timing cohort and emits `Later vs Always Treated` comparisons; Python's paper-footnote-11 convention remaps those units to `U` and folds them into a single `treated_vs_never` cell per treated cohort) by aggregating R's split rows per cohort and asserting they match Python's single fold at `atol=1e-6`. The aggregate is invariant per Theorem 1; the per-component breakdown differs structurally between conventions but the fold-back is now directly asserted. New `**Note (R parity convention divergence on always-treated)**` and `**Deviation (first-period boundary extension on always-treated remap)**` in `docs/methodology/REGISTRY.md`. **First-period boundary deviation:** the paper uses strict `t_i < 1` for the always-treated bucket; the library uses the inclusive `first_treat <= min(time)` rule and folds `first_treat == min(time)` cohorts into `U`. R does NOT apply this fold (it keeps such cohorts as their own bucket). When `min(time) > 1` the rules coincide. Explicitly labeled in REGISTRY's Deviations block and mirrored in `METHODOLOGY_REVIEW.md` and `bacon.py`. METHODOLOGY_REVIEW.md tracker row promoted `**Complete** (R parity goldens pending)` → `**Complete**`.
- **`generate_ddd_panel_data` — panel-structured DGP for Triple-Difference power analysis** (`diff_diff/prep_dgp.py`). New public function exported from `diff_diff` and `diff_diff.prep` for panel DDD simulations. Cross-sectional `generate_ddd_data` remains available unchanged. Produces a balanced panel of `n_units × n_periods` with two unit-level binary dimensions (`group`, `partition`) and a derived `post = 1[period >= treatment_period]` indicator; columns: `unit, period, outcome, group, partition, post, treated, true_effect` (+ `x1, x2` when `add_covariates=True`). DDD-CPT identification holds because the `group * partition` interaction enters as a unit-level (time-invariant) term, leaving the triple-interaction `treatment_effect * group * partition * post` as the sole source of differential group × partition trend. Compatible with `TripleDifference(cluster="unit").fit(..., time="post")` (the cluster kwarg is required because `TripleDifference` is the repeated-cross-section `panel=FALSE` estimator and unclustered SE on panel-generated rows understates variance under within-unit serial correlation; the point estimate `att` is invariant to clustering — see the new `TripleDifference` REGISTRY note on panel-shaped input). Users get panel-realistic unit fixed effects and within-unit serial correlation while the binary 2×2×2 estimator surface is unchanged. **Stratified allocation:** the partition split is drawn stratified-by-group at the requested `partition_frac` so every `(group, partition)` cell receives at least one unit; a targeted `ValueError` is raised at fit-time when the rounded cell counts (`n_units`, `group_frac`, `partition_frac`) would leave any cell empty. This guarantees the 2x2x2 DDD surface is populated for any valid input — independent marginal sampling (the cross-sectional `generate_ddd_data` convention) could collapse cells when marginals are small (e.g., `n_units=4, group_frac=partition_frac=0.25`). Validates `1 <= treatment_period < n_periods`, `group_frac` and `partition_frac` strictly in `(0, 1)`, and `n_units >= 4`. Deterministic recovery (`noise_sd=0`) matches `treatment_effect` to ~1e-15 (covered by `tests/test_prep.py::TestGenerateDddPanelData`, 16 tests including infeasible-config rejection and smallest-feasible-config round-trip through `TripleDifference.fit`). `power.simulate_power` is NOT yet auto-routed to the panel DGP for `TripleDifference` (the existing `_ddd_dgp_kwargs` registry entry still ignores `n_periods` and the existing `_check_ddd_dgp_compat` warning still fires on non-default kwargs) — that wiring is tracked as a follow-up in TODO.md.
- **BaconDecomposition: Goodman-Bacon (2021) methodology audit (PR-B).** Closes the BaconDecomposition row in `METHODOLOGY_REVIEW.md` (status flipped from **In Progress** → **Complete** — initially with an R-parity-goldens caveat that was closed by the parity-goldens bullet above in this same release). Builds on the PR #451 paper review at `docs/methodology/papers/goodman-bacon-2021-review.md`. **Audit outcomes:** (1) Rewrote `_recompute_exact_weights` in `bacon.py` to actually implement Theorem 1 (Eqs. 7-9 + 10e-g) — the prior "exact" implementation was missing the `(1-n_kU)` factor in the subsample variance, did not square the sample share, and added an extraneous `unit_share` factor not present in the paper; the post-hoc sum-to-1 normalization masked the relative-weight error but produced ~0.3% decomposition error vs TWFE on a 3-cohort + never-treated DGP. The rewrite computes the exact numerators of Eqs. 10e/f/g and lets the post-hoc normalization handle the `V̂^D` denominator (Theorem 1's identity guarantees `V̂^D = Σ numerators`). The TWFE-vs-weighted-sum identity now holds at `atol=1e-10` on both noisy and hand-calculable DGPs. (2) Added always-treated warn+remap per paper footnote 11: units whose `first_treat` is at or before the first observable period (`first_treat <= min(time)`, excluding the never-treated sentinels `0` and `np.inf`) are automatically remapped to the `U` (untreated) bucket via an internal column (`__bacon_first_treat_internal__`) with a `UserWarning`. Detection uses ordered-time logic on the **time axis**, so panels whose `time` column has negative or zero-crossing labels (event-time encodings) are handled correctly; the `0` sentinel restriction applies only to `first_treat`, not to `time`, and a real treatment cohort with `first_treat == 0` would still be folded into U today (re-label such cohorts to a non-sentinel value before fitting). The user's original `first_treat` column is preserved unchanged. The count is surfaced as a new `BaconDecompositionResults.n_always_treated_remapped` dataclass field, rendered in `summary()` output when nonzero. **`n_never_treated` reports TRUE never-treated only**, computed from the original user column before remap — remapped always-treated units appear separately as `n_always_treated_remapped`, no double-counting. (3) New methodology test file `tests/test_methodology_bacon.py` (34 tests across 6 classes post-release; the audit added ~24 tests and the R-parity-goldens bullet above expanded coverage: `TestBaconHandCalculation` hand-checks Eqs. 7-9 + 10b-d on a minimal balanced panel at `atol=1e-10`; `TestBaconParityR` (4 tests, all active post-release once the R parity goldens bullet above landed; skips cleanly with a regenerate-instructions pointer in partial-checkout scenarios where the JSON is unavailable); `TestBaconAlwaysTreatedRemap` regression-tests warn+remap mechanics including user-data-preservation; `TestBaconEdgeCases` exercises no-untreated, single-cohort, unbalanced panel, constant-ATT recovery; `TestBaconWeightModes` locks the new exact-is-default contract; `TestBaconSurveyDesignNarrowing` confirms survey_design composes with exact mode and warn+remap). (4) R `bacondecomp::bacon()` parity generator committed at `benchmarks/R/generate_bacon_golden.R` covering three DGP fixtures (3-groups-with-U, 2-groups-no-U, always-treated-remapped); the JSON goldens deferral at audit time was closed in this same release by the parity-goldens bullet above. (5) `docs/methodology/REGISTRY.md` `## BaconDecomposition` block replaced with the paper-review-sourced entry plus three new sub-notes: weight modes (exact vs approximate), always-treated remap, R parity status. **Explicit removal:** the prior REGISTRY block's "Weights may be negative for later-vs-earlier comparisons" claim was incorrect per Theorem 1 (decomposition weights are strictly positive and sum to 1; negative weights are an estimand-level phenomenon, not estimator-level) and is dropped from the new entry. Closes the BaconDecomposition follow-up tracked at `TODO.md` (the prior row added in PR #451 is replaced by a narrower R-parity-goldens deferral row).
- **`SpilloverDiD(event_study=True)` — per-event-time × ring decomposition (Butts 2021 Section 5 / Table 2).** Replaces the Wave B `NotImplementedError` gate with the full per-event-time × ring decomposition. Emits per-event-time direct effects `tau_k` and per-(ring, event-time) spillover effects `delta_jk` as `att_dynamic: pd.DataFrame` (indexed by event-time `k`) and a MultiIndex `spillover_effects: pd.DataFrame` (levels `(ring_label, event_time)`). A TwoStageDiD-compatible `event_study_effects: Dict[int, Dict[str, Any]]` alias (matching `two_stage.py:1355-1389` schema with `conf_int = (low, high)` tuple) is also emitted for consumption by `plot_event_study` (`SpilloverDiDResults` is wired into `_extract_plot_data` and prefers the new `reference_period` attribute over the legacy `n_obs==0` heuristic). `DiagnosticReport` integration is NOT wired in this PR — registering `SpilloverDiDResults` in `DiagnosticReport`'s applicability/method tables is queued as a follow-up. **Methodology spec:** the implementation operationalizes Butts Section 5's single `K_it` symbol as TWO event-time clocks — `K_direct = t - effective_first_treat(i)` for ever-treated unit rows, and `K_spill = t - earliest-in-range-cohort-onset(i)` for spillover rows (running min across activated cohorts; NaN for pre-trigger and far-away rows). `K_spill >= 0` structurally; negative-k spillover cells emit rectangularly with `coef = NaN, n_obs = 0`. **Reference period:** `ref_period = -1 - anticipation` (mirrors `TwoStageDiD` at `two_stage.py:486`); when `horizon_max` is set, `ref_period` must fall inside `[-horizon_max, +horizon_max]` or fit raises `ValueError` — silent floor-shift to `-horizon_max` would change identification (rejected per `feedback_no_silent_failures`). The reference row in `att_dynamic` / `event_study_effects` uses `coef = 0.0, se = 0.0, n_obs = 0, conf_int = (0.0, 0.0)` for TwoStageDiD parity. **`horizon_max` semantics (divergence from TwoStageDiD):** SpilloverDiD bins event-times outside `[-horizon_max, +horizon_max]` into endpoint pools (no observations dropped); TwoStageDiD filters those rows. The divergence is intentional and cross-documented. With `horizon_max=None`, the helper auto-detects the bin set from observed K values. **Scalar `att` aggregation:** when `event_study=True`, the top-level `att` is the **sample-share-weighted average** of post-treatment `tau_k` (`att = sum_{k >= 0} w_k * tau_k` with `w_k = n_treated_at_k / total`). SE comes from linear-combination inference `Var(att) = w' V_subset w` on the post-treatment block of the stage-2 vcov — no separate fit. **Reduce-to-aggregate equivalence:** under a constant-tau DGP with `horizon_max=None`, the lincom-weighted scalar `att` reproduces Wave B's aggregate `tau_total` bit-identically in the deterministic limit (verified by `TestSpilloverDiDEventStudyReduceToAggregate`). Note: `horizon_max=0` is **not supported** under `event_study=True` (rejected at validation): the single bin `k=0` leaves no event-time pair to anchor the reference period against. Use `event_study=False` for a single aggregate direct effect (Wave B static spec); event-study mode requires `horizon_max>=1` or `horizon_max=None`. **Post-finite_mask sample contract:** `att_dynamic["n_obs"]`, `event_study_effects[k]["n_obs"]`, AND the scalar `att` share weights all reflect the POST-`finite_mask` stage-2 estimation sample (not the pre-mask design). On warn-and-drop fits (baseline-treated units without Omega_0 rows excluded), the reported `n_obs` per cell counts only rows that actually entered `solve_ols`. **Fail-closed scalar `att`:** if any post-treatment direct-effect coefficient is NaN (rank-deficient drop by `solve_ols`), the scalar `att` is set to NaN with an explicit warning rather than silently zeroing the dropped column's contribution via `np.nansum` on a fixed weight vector — inspect `att_dynamic` for the per-event-time coefficients and re-aggregate manually if appropriate. **Backward compatibility:** `event_study=False` leaves all Wave C fields (`att_dynamic`, `event_study_effects`, `horizon_max`, `reference_period`) as `None`. The aggregate stage-2 design construction, fit, and extraction logic on this path are byte-identical to Wave B; `TestSpilloverDiDEventStudyBackwardCompat` pins att / se / per-ring goldens captured on the unchanged aggregate path so any future drift fails the regression. **Variance:** at original Wave C ship time per-event-time SEs used `solve_ols`'s standard variance (HC1 / Conley / cluster paths) WITHOUT the Gardner GMM first-stage uncertainty correction. **Superseded by the Wave D Gardner GMM first-stage correction in this same release** (see the Wave D bullet above): per-event-time SEs now apply the IF outer-product correction unconditionally and shift upward by 1-few percent relative to the original Wave C ship-time values. **Tests:** `tests/test_spillover.py` adds 30 new test methods across event-study API, two-clock K helper, horizon binning, design builder, reference period, reduce-to-aggregate, identification MC (50 seeds, per-event-time tau_k recovery within 0.025), placebo pre-trends (Type I rate ≤ 0.30 over 50 seeds at alpha=0.10), singularity (rectangular schema), Conley integration (vcov shape + non-negative diagonal), summary/to_dict/pickle round-trip, event_study_effects schema parity with TwoStageDiD, lincom-att hand-computed, validation (`horizon_max < 0`, `ref_period < -horizon_max`), and fit idempotence. DGP factory `generate_butts_staggered_dgp` extended with `tau_per_event_time` and `delta_per_ring_per_event_time` callable kwargs (backward-compatible — both default to `None`, producing the Wave B scalar DGP bit-identically; verified by `tests/test_dgp_utils.py` with pinned SHA-256 baselines).
- **`SpilloverDiD` — ring-indicator spillover-aware DiD (Butts 2021).** New standalone estimator at `diff_diff/spillover.py` implementing two-stage Gardner methodology with ring-indicator covariates that identify direct effect on treated (`tau_total`) alongside per-ring spillover effects on near-control units (`delta_j`). Documented synthesis of ingredients (no single published software covers the exact recipe — `did2s` implements Gardner two-stage without rings; the Butts ring estimator has no R/Stata package): Butts (2021) Section 5 / Table 2 identification, Gardner (2022) two-stage residualize-then-fit, and the Conley spatial-HAC vcov shipped in 3.3.3. Handles both panel non-staggered (Equations 5/6/8) and Section 5 staggered timing in one estimator — non-staggered is the special case where all treated units share an onset time. **API:** `SpilloverDiD(rings=[0, 50, 100, 200], conley_coords=("lat","lon"), ...).fit(data, outcome="y", unit="unit", time="t", treatment="D")` (binary D auto-converted to `first_treat`) or `.fit(..., first_treat="first_treat")` (Gardner convention). Result: `SpilloverDiDResults(DiDResults)` with `.att` = `tau_total`, `.spillover_effects` (per-ring `pd.DataFrame` with `coef`/`se`/`t_stat`/`p_value`/`ci_low`/`ci_high`), `.ring_breakpoints`, `.d_bar`, `.n_units_ever_in_ring`, `.n_far_away_obs`, `.is_staggered`. `.coefficients` exposes all `(1+K)` stage-2 entries (`"treatment"` + `"_spillover_<ring_label>"`) plus an `"ATT"` alias keyed to vcov columns. **Methodology spec (committed):** stage-2 regressor is the time-varying `(1 - D_it) * Ring_{it,j}` form (paper page 12's `S_it = S_i * 1{t >= t_treat}` notation; Section 5 Table 2's `S^k_{it}` / `Ring^k_{it,j}`). Reading the literal unit-static `(1 - D_it) * S_i` from Equation 5 is algebraically rank-deficient under TWFE (`(1-D_it) * S_i = S_i - D_it`, with `S_i` absorbed by `mu_i`, leaving `-D_it`); only the time-varying form supports the paper's identification (Proposition 2.3). Stage-1 subsample uses Butts' STRICTER `Omega_0 = {D_it = 0 AND S_it = 0}` (untreated AND unexposed), not TwoStageDiD's `{D_it = 0}` alone — this prevents spillover-contaminated near-controls in pre/post periods from biasing the time FE. **Gardner identity (non-staggered):** a 20-seed deterministic regression test pins `SpilloverDiD.att` against a direct single-stage TWFE ring regression on the full sample (`y ~ mu_i + lambda_t + tau * D_it + sum_j delta_j * (1 - D_it) * Ring_{it,j}`) at `atol=1e-10` — empirically bit-identical, so the reported non-staggered `tau_total` IS the Butts Eqs. 4-6 estimator. **Identification-check policy (period strict, unit warn-and-drop, plus connectivity):** every period must have at least one Omega_0 row (hard `ValueError` — dropping a period removes all units' cross-time identification). Units lacking Omega_0 rows (e.g. baseline-treated units with `D_it = 1` at every observed `t`) are warned-and-dropped: their unit FE is NaN, residualization writes NaN on their rows, and the downstream finite-mask path excludes them from stage 2 — mirrors `TwoStageDiD`'s always-treated convention. Additionally, the supported-units bipartite graph (units linked by shared Omega_0 periods) must form a single connected component; `K > 1` components raise `ValueError` because the FE solver would return only component-specific constants and residualization would silently mix them across components (defense-in-depth — under absorbing treatment the disconnected case may be unreachable through the upstream validators, but the check future-proofs Wave B follow-ups). **Public API restrictions (Wave B MVP):** `covariates=` raises `NotImplementedError` because Gardner-style two-stage requires covariate effects estimated on the untreated-and-unexposed subsample at stage 1 (appending raw covariates only at stage 2 silently biases `tau_total` / `delta_j` on panels with time-varying covariates); non-absorbing / reversible treatment patterns (e.g. `[0, 1, 0]`) raise `ValueError` rather than being silently coerced into "treated from first 1 onward"; non-constant `first_treat` values across rows of the same unit raise `ValueError`; `conley_coords` is required on every fit path (not just `vcov_type="conley"`) because ring construction always uses it. **Far-away control identification:** uses CURRENT-period untreated status (`D_it = 0`) rather than never-treated-only, so all-eventually-treated staggered designs (no never-treated units) can identify the counterfactual via not-yet-treated far-away rows. **Variance (Wave B MVP ship-time):** stage-2 OLS variance via `solve_ols` (HC1 / Conley / cluster paths all flow through) WITHOUT the Gardner GMM first-stage uncertainty correction. **Superseded by the Wave D Gardner GMM first-stage correction in this same release** (see the Wave D bullet above): the GMM correction now applies unconditionally across HC1 / Conley / CR1 (via `cluster=<col>`), shifting SE values upward by 1-few percent relative to the original Wave B ship-time values. **Deferred features (planned follow-ups, as of Wave B ship-time):** `event_study=True` per-event-time × ring coefficients (Butts Table 2), `survey_design=` integration, `ring_method="count"` (count-of-treated-in-ring), data-driven `d_bar` selection (Butts 2021b / Butts 2023 JUE Insight), Gardner GMM first-stage correction at stage 2, sparse staggered ring-distance path. **Shipped in same release:** `event_study=True` (Wave C bullet above) + Gardner GMM first-stage correction (Wave D bullet above); remaining items still queued. **Tests:** `tests/test_spillover.py` (157 tests across ring-construction primitives, validators, fit integration, raw-data invariant, identification MC — non-staggered DGP at 50 seeds + 200-seed `@pytest.mark.slow` variant recovers both `tau_total` and `delta_1`; staggered DGP at 30 seeds anchors both `tau_total` and `delta_1` — Conley plumbing (verifies `solve_ols` is called with `vcov_type="conley"` + Conley kwargs, no silent HC1 fallback), Gardner identity bit-identity, coefficients-vs-vcov alignment, warn-and-drop, rank_deficient_action validation, Omega_0 bipartite-graph connectivity, anticipation behavior on both fit paths). DGP factories `tests/_dgp_utils.py::generate_butts_nonstaggered_dgp` / `generate_butts_staggered_dgp` satisfy Butts Assumptions 1/3/5/7 by construction.
- **`ChaisemartinDHaultfoeuille.predict_het` × `placebo`: R-parity on both global and per-path surfaces.** R-verified — `did_multiplegt_dyn(predict_het, placebo)` emits heterogeneity OLS results on backward (placebo) horizons via R's `DIDmultiplegtDYN:::did_multiplegt_main` placebo block (`effect = matrix(-i, ...)` rbind site); the same block runs per-by_level under `did_multiplegt_dyn(by_path, predict_het, placebo)`, so both global `res$results$predict_het` and per-by_level `res$by_level_i$results$predict_het` slots emit backward rows. R's predict_het syntax with `placebo > 0` requires the `c(-1)` sentinel in the horizon vector to trigger "compute heterogeneity for ALL forward (1..effects) AND ALL placebo (1..placebo) positions" — passing positive-only horizons errors with "specified numbers in predict_het that exceed the number of placebos". Python mirrors via `_compute_heterogeneity_test(..., placebo=L_max)` (set automatically from `self.placebo` at both global and per-path call sites in `fit()`) — the function iterates forward (1..L_max) and backward (-1..-L_max) horizons in a single loop with an explicit `out_idx < 0` eligibility guard for backward horizons whose `F_g` is too small (would otherwise silently misread `N_mat` via numpy negative indexing). `results.heterogeneity_effects` uses negative-int keys for backward horizons; `path_heterogeneity_effects` does the same per path. Placebo rows in `to_dataframe(level="by_path")` have non-NaN `het_*` columns when `placebo=True` and `heterogeneity=` are both set. **Survey gate (warn + skip):** `survey_design + placebo + heterogeneity` emits a `UserWarning` at fit-time and falls back to forward-horizon-only heterogeneity on both surfaces — the Binder TSL cell-period allocator's REGISTRY justification is tied to **post-period** attribution; backward-horizon attribution puts ψ_g mass on a pre-period cell, a separate library-extension claim that needs its own derivation. Forward-horizon `predict_het + survey_design` continues to work unchanged on both global and per-path surfaces. The function-level `_compute_heterogeneity_test` keeps a per-iteration `NotImplementedError` backstop for direct callers that bypass fit(). Pre-period allocator derivation deferred to a follow-up methodology PR (tracked in TODO.md). R parity confirmed at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityHeterogeneityWithPlacebo` (scenario 23, `multi_path_reversible_predict_het_with_placebo_global`, `placebo=2, effects=3, no by_path`) and `::TestDCDHDynRParityByPathHeterogeneityWithPlacebo` (scenario 22, same DGP plus `by_path=3`); pinned at `BETA_RTOL=1e-6` / `SE_RTOL=1e-5` for `beta` / `se` / `t_stat` / `n_obs` and `INFERENCE_RTOL=1e-4` for `p_value` / `conf_int` across 3 paths × (3 forward + 2 placebo) = 15 horizons + 1 global × 5 horizons. Cross-surface invariants regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathPredictHetPlacebo` (placebo het column population, survey-gate warn+skip behavior, forward+survey anti-regression, `out_idx<0` eligibility guard, single-path telescope `path_heterogeneity_effects[(only_path,)] == heterogeneity_effects` bit-exactly, summary rendering, direct-call `NotImplementedError` backstop). Closes TODO #422.

### Changed
- **PreTrendsPower: default `pretest_form` flipped from implicit Wald to explicit `'nis'` (PR-B methodology audit, Roth 2022).** The new default uses the paper-analyzed NIS box probability — the form Roth (2022) actually tabulates in his Section I.C empirical exercise and the form the R `pretrends` package implements. `pretest_form='wald'` preserves the **acceptance-region form** (noncentral-χ² on the quadratic form `δ' Σ_22^{-1} δ`) byte-identically — the methods are renamed `_compute_power_wald` + `_compute_mdv_wald` with unchanged bodies, dispatched on `self.pretest_form`. **Caveat on bit-identity for fitted results**: the linear-weight contract changed independently in PR-B Step 4 (see the next bullet), so a Wald fit on an irregular pre-period grid produces γ-unit MDV via the new `relative_times`-threaded path, NOT the pre-PR-B count-based L2-normalized MDV. Pre-PR-B Wald numerics are bit-identical to post-PR-B Wald output only on the legacy `relative_times=None` callable path (callers that bypass `fit()` and call `_get_violation_weights(n_pre)` directly) and on the regular-grid case where `|t| ∝ [n_pre-1, ..., 0]`. All existing `tests/test_pretrends.py` numerical assertions (101 helper/class references; only 3 tests depended on the exact Wald size-at-null property and were pinned to `pretest_form='wald'`) continue to produce identical numerical output. The `docs/tutorials/07_pretrends_power.ipynb` walkthrough re-render to reflect the default flip is tracked as a follow-up (the existing tutorial does not exercise the irregular-grid regime).
- **PreTrendsPower: `_get_violation_weights('linear')` now honors actual pre-period relative-time labels and skips L2 normalization → reported MDV is in Roth's γ units (PR-B Step 4).** Pre-PR-B, the linear-violation direction was constructed as `[n_pre-1, ..., 1, 0] / ||·||_2` from `n_pre` count alone — irregular pre-period grids like `{-5, -3, -1}` were treated as if the periods were `{-3, -2, -1}`, and the L2-normalization meant the reported MDV equaled `γ · ||t||_2`, not γ. PR-B threads the actual `relative_times` array from `_extract_pre_period_params` into `_get_violation_weights` and, for `violation_type='linear'` with `relative_times not None`, uses `weights = |t|` directly with NO L2 normalization. Then `δ_pre = M · |t|` reflects Roth's `δ_t = γ · t` convention and the reported MDV equals γ exactly. Verified: regular grid `[-3, -2, -1]` → weights `[3, 2, 1]`; irregular grid `[-5, -3, -1]` → weights `[5, 3, 1]`; backwards-compat callers that bypass `fit()` and pass only `n_pre` retain the legacy normalized `[n_pre-1, ..., 0] / ||·||_2` behavior. The `_extract_pre_period_params` return type widened from a 4-tuple to a 6-tuple `(effects, ses, vcov, n_pre, relative_times, covariance_source)`; the `relative_times` element is populated by all three adapter branches from their respective sorted pre-period lists (MPD via `pandas.Period` / `Timestamp` / `np.datetime64` arithmetic when applicable, falling back to a warn + count-based normalized direction for genuinely non-numeric labels), and the new `covariance_source` element records the actual extraction path for downstream report-layer tier classification.
- **BaconDecomposition: default `weights` flipped from `"approximate"` to `"exact"` (PR-B methodology audit).** The new default uses Goodman-Bacon (2021) Theorem 1's exact Eqs. 7-9 + 10e-g weights, matching R `bacondecomp::bacon()` at `atol=1e-6` (validated via `tests/test_methodology_bacon.py::TestBaconParityR`; see the new Added entry above for the convention divergence on always-treated cohorts). Hand-calculation + TWFE-vs-weighted-sum identity also hold at `atol=1e-10`. The `weights="approximate"` path remains available as an opt-in fast diagnostic for speed-sensitive loops; its numerical output may differ from R. Three entry points were flipped: `BaconDecomposition(weights="exact")` (`bacon.py:397`), `bacon_decompose(weights="exact")` (`bacon.py:1064`), `TwoWayFixedEffects.decompose(weights="exact")` (`twfe.py:684`). **Behavior change for users not passing explicit `weights=`**: the decomposition weights are now paper-faithful by default. Users who depended on the previous `"approximate"` numerics for diagnostic plots or comparison-type weight shares can preserve the old behavior by passing `weights="approximate"` explicitly. **Survey-design behavior change**: `weights="exact"` (now the default) routes through `_validate_unit_constant_survey`, which rejects survey designs whose weights / strata / PSU / FPC columns vary within a unit across periods (the exact-mode path collapses to per-unit aggregation via `groupby().first()`). The previous `weights="approximate"` default tolerated time-varying within-unit survey weights via observation-level weighted means. Users whose survey-weighted Bacon calls used time-varying within-unit weights must now either (a) collapse their weights to be unit-constant or (b) pass explicit `weights="approximate"` to retain the legacy obs-level path. The production diagnostic surface (`diff_diff/diagnostic_report.py:1740`) was updated to pass explicit `weights="exact"`. Existing test assertions in `tests/test_bacon.py` continue to pass with the new default; the `test_weighted_sum_equals_twfe` tolerance was tightened from `< 0.1` to `< 1e-10` to lock the Theorem 1 algebraic-identity contract.

- **`ChaisemartinDHaultfoeuille.predict_het` inference: t-distribution df threading (closes TODO pilot-412).** `_compute_heterogeneity_test` now passes `df = n_obs - rank(design)` to `safe_inference` on the non-survey OLS path, matching R `did_multiplegt_dyn(predict_het=...)`'s t-distribution inference (`DIDmultiplegtDYN:::did_multiplegt_main` `t_stat <- qt(0.975, df.residual(model))` site). Pre-PR Python used `df=None` (normal Z critical), producing 0.1-2% rtol gaps on `p_value` and `conf_int` vs R. Parity tolerance tightened on the existing forward-horizon scenarios (`multi_path_reversible_predict_het`, `multi_path_reversible_by_path_predict_het`) from "unpinned" to `INFERENCE_RTOL=1e-4` on `p_value` and `conf_int`; `beta` / `se` / `t_stat` continue at `BETA_RTOL=1e-6` / `SE_RTOL=1e-5`. **Post-drop rank (post-2026-05-16 wrap-up):** the df denominator uses the post-drop numerical rank via `_detect_rank_deficiency`, which `solve_ols` already calls internally. For full-rank designs `rank == n_params` and behavior is bit-identical to the pre-PR `n_obs - n_params` path; for near-rank-deficient designs that `solve_ols` retains rather than NaN-out (e.g., cohort-collinearity at high horizons), the post-drop rank is strictly lower and the post-PR `df` is larger, matching R's `lm()` convention. The Z-vs-t REGISTRY deviation note is replaced with an "R parity (post-2026-05-15 df threading)" positive-claim note.

- **`ChaisemartinDHaultfoeuille.by_path` negative-baseline path regression coverage.** New `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary::test_negative_baseline_path_supported` exercises switchers with `D_{g,1} = -1` and asserts that `path_effects` correctly contains negative-baseline tuple keys (e.g., `(-1, 0, 0, 0)`, `(-1, 1, 1, 1)`). This closes the test-coverage gap from PR #419: the existing `test_negative_integer_D_supported` only covered paths with negative values in non-baseline positions (e.g., `(0, -1, -1, -1)`), which does not trigger R's documented `substr(path, 1, 1)` baseline-extraction bug. Python's tuple-key matching is correct under any baseline value; this test pins the contract. No R-parity fixture is added because R is the buggy side on this regime — the deviation is documented in the REGISTRY non-binary treatment Note.

### Fixed
- **PreTrendsPower: unit-consistent level-scale ratio for tier classification (PR-B R12 follow-up).** PR-B Step 4 made the linear MDV report Roth's γ units (a slope on relative time), but downstream tier-classification heuristics still divided the raw γ by level-scale quantities — `DiagnosticReport.pretrends_power` computed `mdv_share_of_att = mdv / abs(att)`, `is_informative` checked `mdv < 2 * max(pre_period_ses)`, and `sensitivity_to_honest_did` reported `mdv_in_ses = mdv / max_pre_se`. On irregular pre-period grids this silently mixed slope and level scales and could mis-tier the same fit as `well_powered` / `moderately_powered` / `underpowered`. Fix: new `PreTrendsPowerResults.max_abs_pre_violation` property exposes the level-scale scalar `mdv * max(|violation_weights|)` — the largest level-scale pre-period deviation under the MDV. `is_informative`, `sensitivity_to_honest_did`, `DiagnosticReport._check_pretrends_power`, and `_format_precomputed_pretrends_power` all switched to consume `max_abs_pre_violation` instead of raw `mdv` for level-scale comparisons. `mdv_share_of_att` is now defined as `max_abs_pre_violation / abs(att)`; the schema also surfaces the new `max_abs_pre_violation` field for inspection. Legacy serialized results without `violation_weights` fall back to raw `mdv` (preserves pre-PR-B count-based L2-normalized behavior where `mdv` was already roughly level-scale). On the live `cs_fit` fixture the ratio moves from `0.053` (slope/level mismatch) to `0.211` (level/level) — still `well_powered`, but now interpretable. New regressions: `test_max_abs_pre_violation_uses_weight_scale_on_irregular_grid` (γ * 5 on `[-5, -3, -1]`), `test_is_informative_uses_level_scale_not_raw_gamma` (level-scale check beats raw-γ check on a constructed mismatch), plus the updated BR `test_full_vcov_path_no_downgrade_on_real_cs_fit` which now pins `0.35 < max_abs_pre_violation < 0.40`.
- **PreTrendsPower: `PreTrendsPowerResults.power_at(M)` for `violation_type='custom'` (PR-B Step 5).** PR-A R18 added a `NotImplementedError` guard to prevent silent equal-weights output when `power_at()` couldn't reconstruct the fitted custom weights. PR-B Step 5 persists the normalized `violation_weights` on `PreTrendsPowerResults` at fit time, so `power_at(M)` now works correctly for all four violation types (linear / constant / last_period / custom) on fresh fits. The PR-A guard is retained only for legacy serialized results lacking the new `violation_weights` field (refit with current library version to lift). Verified by the new `test_power_at_works_for_custom_violation_type` regression test and the companion `test_power_at_raises_on_legacy_custom_result_without_weights` (simulates a legacy serialized result by clearing `violation_weights` to None).
- **`DiagnosticReport` / `BusinessReport` covariance-source provenance propagation (PR-B Step 3, R3 follow-up).** Before PR-B, `DiagnosticReport._infer_cov_source` flagged CS / SA fits with populated `event_study_vcov` as `"diag_fallback_available_full_vcov_unused"`, and `_apply_diag_fallback_downgrade` then conservatively downgraded the `well_powered` tier to `moderately_powered`. PR-B Step 3 routes those fits through the full `Σ_22` sub-block at the estimator layer — but the report layer kept the old type-based inference, so correctly-computed full-VCV power results were silently being downgraded. Fix: `PreTrendsPowerResults` gains a new `covariance_source` field that `pretrends.py:_extract_pre_period_params` populates with `"full_pre_period_vcov"` or `"diag_fallback"` based on the actual extraction path taken; `DiagnosticReport._check_pretrends_power` and `_format_precomputed_pretrends_power` prefer that persisted label and fall back to type-based inference only for legacy serialized results that lack the field. Two paths now coexist through the report layer: **new fits** (post-PR-B, `covariance_source` is persisted) consume the persisted label directly — non-bootstrap CS / SA report `"full_pre_period_vcov"` and are NOT downgraded; **legacy serialized results** (pre-PR-B, no `covariance_source` field on the object) fall through to `_infer_cov_source`, which STILL emits the conservative `"diag_fallback_available_full_vcov_unused"` sentinel for CS / SA + populated `event_study_vcov` because without the persisted label we cannot distinguish a pre-PR-B fit (which used `diag(ses^2)`) from a post-PR-B fit, and the PR-A conservative downgrade still applies to preserve backwards-compat. For `MultiPeriodDiDResults` without `interaction_indices`, the legacy fallback reports `"diag_fallback"` (a genuine fallback, no downgrade applies). Effect: non-bootstrap CS / SA pre-trends power blocks on fresh fits now keep their well_powered tier through the report layer (instead of being downgraded by the conservative sentinel); legacy serialized results are unchanged. Verified by `test_precomputed_pretrends_power_persisted_full_vcov_no_downgrade` (new fits), `test_precomputed_pretrends_power_legacy_missing_field_still_downgraded` (legacy fallback contract), `test_precomputed_pretrends_power_consumes_persisted_cov_source` (persisted label takes precedence over legacy inference), and `test_precomputed_pretrends_power_legacy_mpd_without_interaction_indices_reports_diag`.

## [3.3.3] - 2026-05-15

### Added
- **Tutorial 22: Survey-Weighted HAD** (`docs/tutorials/22_had_survey_design.ipynb`) — end-to-end walkthrough of `HeterogeneousAdoptionDiD` + `did_had_pretest_workflow` on a BRFSS-shape stratified household-survey panel (5 strata × 6 PSUs/stratum × 2 states/PSU = 60 states; post-stratification raking weights with CV ~ 0.30; FPC = 30 PSUs/stratum; PSU × period interaction shocks injected so cluster correlation survives DiD first-differencing). Demonstrates the `SurveyDesign(strata=...)` path through the Stute pretest family that the previous `[Unreleased]` entry unblocked. Eight numbered sections: motivation; panel + in-notebook helper for attaching survey columns to a HAD panel; naive vs survey-aware headline fit with a side-by-side ATT / SE / CI table (~10% SE inflation, sign-only direction asserted); a dedicated section explaining why the SE inflation is modest for HAD specifically (WAS-d_lower IF concentration at the boundary vs full-panel regression coefficients); event-study fit with sup-t cband under the survey design (per-horizon table + matplotlib gated plot); pretest workflow on both `aggregate="overall"` and `aggregate="event_study"` paths walking the Phase 4.5 C0 QUG-deferred verdict suffix and the now-supported stratified-clustered Stute multiplier bootstrap; "Communicating to Leadership" two-paragraph template (executive + methodologist); Extensions + Summary Checklist surfacing the still-deferred `lonely_psu='adjust'` + singleton-strata, replicate-weight designs, and the permanent QUG-under-survey C0 deferral. Companion drift-test file `tests/test_t22_had_survey_design_drift.py` (32 tests across 7 groups: panel + survey composition with deterministic exact pins; naive-vs-survey headline with sign-only SE-inflation anchor; event-study cband-vs-pointwise ordering and post/pre coverage; pretest overall path with `_QUG_DEFERRED_SUFFIX` lock and Yatchew `sigma2_*` deterministic pins; pretest event-study path with the SAME `_QUG_DEFERRED_SUFFIX` lock plus a SEPARATE substring lock on `report.summary()` for the L736 QUG-skip note; workflow-surface separation locking that overall has Stute+Yatchew while event-study has joint pretrends/joint linearity with `yatchew=None` and `stute=None`; and weighted point-estimation contract anchoring `survey.att != naive.att` plus the algebraic identity `att = (dy_mean_w - tau_bc) / den_w` from `_fit_continuous`). Bootstrap p-value pins use anchored windows of total width 0.30 (± 0.15 around seeded centers) per `feedback_strata_bootstrap_path_divergence` (stratified Mammen multiplier paths reduce effective dofs vs non-strata; PR #432 commit `aef07020` already had to relax bit-equality bands on this code path). T20 and T21 "Extensions" bullets updated with forward-pointers to T22; `docs/practitioner_decision_tree.rst` HAD universal-rollout and survey sections each gain a `.. tip::` cross-link to T22 (adjacent to T20 / T17, NOT displacing); `docs/api/had.rst` gains a "Survey-aware fit" cross-reference; `docs/survey-roadmap.md` gains a "Phase 4.5 C ✅ Shipped" entry; bundled `diff_diff/guides/llms.txt` and `llms-practitioner.txt` carry T22 inventory entries (the `llms-full.txt` reference guide is left as a follow-up to keep T22 PR scope tight); `docs/doc-deps.yaml` wires T22 as a dependent of both `had.py` and `had_pretests.py`. Closes the Phase 5 (wave 2 second slice) tutorial gap; the realistic survey-weighted HAD workflow on BRFSS / CPS / NHANES / ACS-shaped designs is now end-to-end documented for practitioners.
- **HAD pretest workflow: stratified survey-design support (Phase 4.5 C continuation).** Lifts the `NotImplementedError` gate on `SurveyDesign(strata=...)` in `stute_test` (`had_pretests.py:1927-1940` pre-PR) and `stute_joint_pretest` (`:3259-3271` pre-PR), and by inheritance in `joint_pretrends_test`, `joint_homogeneity_test`, and `did_had_pretest_workflow` (the wrappers delegate to the joint Stute helper). Implements a documented synthesis of clustered-wild-bootstrap ingredients (Cameron-Gelbach-Miller 2008 cluster-level multipliers; Davidson-Flachaire 2008 wild-bootstrap centering; Djogbenou-MacKinnon-Nielsen 2019 cluster-wild consistency for nonlinear functionals; Kreiss-Lahiri 2012 within-block centering analogy; Wu 1986 / Liu 1988 Bessel small-sample correction) — no single paper covers the exact composition for the stratified Stute CvM functional. The recipe: within-stratum demean + `sqrt(n_h/(n_h-1))` Bessel rescale applied to the PSU multipliers `psu_mults` BEFORE the per-obs broadcast `eta_obs = psu_mults[b, psu_col_idx]` in the wild-residual loop. Bootstrap CvM variance matches the analytical Binder-TSL stratified target `V_S = sum_h (1 - f_h) (n_h / (n_h - 1)) sum_j (psi_hj - psi_h_bar)²` exactly (the `(1 - f_h)` FPC factor was already baked in by `generate_survey_multiplier_weights_batch`; this PR bakes the remaining `(n_h / (n_h - 1))` factor and enforces within-stratum-mean-zero centering). New shared helper `bootstrap_utils.apply_stratum_centering(psu_mults, resolved_survey, psu_ids, psu_axis=...)` is called from both the new Stute path (psu_axis=1 on the multiplier matrix) AND the existing HAD sup-t event-study cband bootstrap (psu_axis=0 on the PSU-aggregated influence tensor; refactored bit-exactly from the inline block previously at `had.py:2172-2204`). Locks the algebraic identity architecturally instead of leaving parallel code blocks to drift. MC oracle consistency validated under a 4-stratum × 6-PSU/stratum stratified null DGP with weights+strata+PSU (200 seeded draws, empirical Type I at α=0.05 in `[0, 0.10]` — 3σ band; the FPC bake-in is covered separately by the helper-unit test `test_fpc_baked_in_helper_is_fpc_agnostic`); MC power validated under a known-alternative stratified DGP (rejection > 0.50). HAD sup-t event-study cband bit-parity preserved (`atol=1e-14, rtol=1e-14` on the refactored helper output + 29 existing cband tests passing post-refactor; that helper-level bit-parity test locks the axis-0 algebra). A separate wired-in regression at `tests/test_had_pretests.py::TestStuteStratifiedSurveyBootstrap::test_stute_call_sites_invoke_apply_stratum_centering` monkey-patches the helper and asserts both Stute call sites (`stute_test` at `had_pretests.py:1985` and `stute_joint_pretest` at `:3312`) invoke it with `psu_axis=1` — that test fails if either call site is disconnected (the axis-0 helper-parity test alone does not catch that case). See `docs/methodology/REGISTRY.md` § HeterogeneousAdoptionDiD — "Note (Stute stratified survey-bootstrap calibration)" for the full derivation. Remaining deferrals: `lonely_psu='adjust'` + singleton-strata (same pseudo-stratum centering gap as the HAD sup-t deviation at REGISTRY:2382) and replicate-weight designs (BRR/Fay/JK1/JKn/SDR — separate Rao-Wu / JKn bootstrap composition). Unblocks the realistic survey-weighted HAD workflow on BRFSS/CPS/NHANES/ACS-shaped designs.
- **Conley (1999) Wave A mechanical extensions** on top of the Phase 1+2 sandwich (`diff_diff/conley.py`, `diff_diff/linalg.py`, `diff_diff/estimators.py`, `diff_diff/twfe.py`). **(1) DiD support (#118):** `DifferenceInDifferences(vcov_type="conley").fit(..., unit="<col>")` is now supported. `unit` is a fit-time kwarg (NOT on `__init__`; unused unless Conley is set; not part of `get_params()` / `set_params()`) mirroring `MultiPeriodDiD.fit(unit=...)` / `TwoWayFixedEffects.fit(unit=...)`. DiD inherits the same panel block-decomposed sandwich as MPD/TWFE; on a 2-period panel it matches `MultiPeriodDiD(...).fit(..., post_periods=[1], reference_period=0)` bit-exactly. Missing `unit=`/`conley_lag_cutoff`/`conley_coords`/`conley_cutoff_km` raise `ValueError`; `survey_design=` + Conley raises `NotImplementedError` (Bertanha-Imbens 2014 follow-up); `inference="wild_bootstrap"` + Conley raises `NotImplementedError`. **(2) Combined spatial + cluster product kernel (#119):** `compute_robust_vcov(vcov_type="conley", cluster_ids=...)` / `LinearRegression(vcov_type="conley", cluster_ids=...)` / `TwoWayFixedEffects(vcov_type="conley", cluster="<col>")` / `MultiPeriodDiD(vcov_type="conley", cluster="<col>")` / `DifferenceInDifferences(vcov_type="conley", cluster="<col>")` apply `K_total[i, j] = K_space(d_ij/h) · 1{c_i = c_j}`. On the panel block-decomposed path the cluster indicator multiplies BOTH the spatial sandwich AND the serial sandwich; the validator enforces that `cluster_ids` is constant within each unit across periods (the within-unit serial mask is then trivially all-ones; cross-sectional path has no such constraint). TWFE's default auto-cluster on the Conley path remains silently dropped (combining with unit-level clusters would zero out all between-unit pairs and defeat the spatial pooling); users must pass an explicit above-unit cluster (e.g. region) to opt in. DiD has no auto-cluster — the choice is fully explicit. Two limit fixtures anchor correctness (no R parity — R `conleyreg` does not support combined kernels): all-unique-clusters reduces to HC0; huge-cutoff reduces to pure within-cluster CR1. The huge-cutoff reduction is EXACT only for `conley_kernel="uniform"` (`K(u) = 1` for `|u| ≤ 1`); for `conley_kernel="bartlett"` the identity is asymptotic since `K_bartlett(u) = 1 - |u| < 1` for `u > 0`. The fixture anchor uses uniform for an exact identity check. Per-slice mask construction (NOT full n×n) preserves memory on panel paths. **(3) Sparse k-d-tree fast path (#120):** auto-activates for the spatial Bartlett meat when `n > 5_000` AND metric is `"haversine"` or `"euclidean"` AND kernel is `"bartlett"`. Builds a CSR sparse kernel matrix via `scipy.spatial.cKDTree.query_ball_tree` instead of materializing the full n×n distance matrix; haversine projects to a 3-D unit-sphere chord representation with the exact great-circle recomputed for in-range neighbors only. Bit-identity parity vs the dense path at `atol=1e-10`; R parity at `atol=1e-6` is preserved on the existing 3 panel R fixtures with the sparse path force-enabled. The bartlett-only gate is for boundary correctness — bartlett at `u=1` is exactly 0, so the sparse path safely drops at-cutoff pairs; uniform at `u=1` is 1 and would require a closed-interval query semantic that haversine chord projection cannot reliably preserve. Constants: `_CONLEY_SPARSE_N_THRESHOLD = 5_000` (auto-toggle); `_CONLEY_DENSE_WARN_N` renamed `_CONLEY_DENSE_OOM_WARN_N = 20_000` (memory exhaustion threshold for the dense fallback — independent of the sparse threshold). Private `_conley_sparse: Optional[bool]` kwarg on `_compute_conley_vcov` controls the toggle (`None` = auto, `True` = force, `False` = force dense; `True` with an unsupported kernel/metric raises). The serial component (within-unit Bartlett over time) remains dense regardless — per-unit slices are small. **(4) Callable `conley_metric` validation (#123):** result must satisfy shape `(n, n)`, finite, non-negative, symmetric to `atol=1e-10`, AND zero on the diagonal (`|d(i, i)| ≤ 1e-10`); each failure raises a targeted `ValueError` naming the violated invariant. The zero-diagonal contract is load-bearing for the Conley sandwich: the `i = j` term must reduce to the HC0 diagonal `X_i ε_i² X_i'` via `K(0) = 1`; positive self-distance would silently attenuate the HC0 contribution by `K(d_ii / h) < 1`. Built-in metrics (`"haversine"`, `"euclidean"`) satisfy this by construction. Previously, malformed callables produced opaque BLAS errors deep in the pipeline. **Tests:** `tests/test_conley_vcov.py::TestConleySparse` (12), `::TestConleySparseRParityForced` (3), `::TestConleyCluster` (10), `::TestConleyDistanceMetrics` extended (7 new); existing rejection tests flipped to behavioral; `test_did_conley_matches_mpd_post_periods_1` locks the DiD-vs-MPD bit-exact agreement. **Docs:** REGISTRY `## ConleySpatialHAC` updates: new "Combined spatial + cluster product kernel" + "Performance / scale" subsections, DiD-vs-TWFE cluster asymmetry paragraph, updated panel-API restrictions table. TODO rows 118 / 119 / 120 / 123 removed; rows 121 (Conley + survey_design / weights, Bertanha-Imbens 2014) and 122 (`SyntheticDiD(vcov_type="conley")`, spatial-block bootstrap per Politis-Romano 1994) retained for future waves.
- **Conley (1999) spatial-HAC standard errors via `vcov_type="conley"`** on cross-sectional `LinearRegression` / `compute_robust_vcov` plus panel `MultiPeriodDiD` / `TwoWayFixedEffects` (Phases 1 and 2 of the spillover-conley initiative). **Cross-sectional contract:** `conley_coords` (n × 2 array of lat/lon or projected coords), `conley_cutoff_km=<float>` (positive finite bandwidth in km for haversine, or coord units for euclidean — REQUIRED, no default per the no-silent-failures contract), `conley_metric="haversine"|"euclidean"|callable` (default `"haversine"`; great-circle uses Earth's mean radius 6371.01 km matching R `conleyreg`), `conley_kernel="bartlett"|"uniform"` (default `"bartlett"`; both kernels emit a `UserWarning` if the resulting meat has a materially negative eigenvalue — neither the radial 1-D Bartlett nor the uniform kernel is formally PSD-guaranteed; Conley 1999's explicit PSD formula is the 2-D separable lattice product window at Eq 3.14). Cross-sectional variance estimator `Var̂(β) = (X'X)^{-1} · ( Σ_{i,j} K(d_ij/h) · X_i ε_i ε_j X_j' ) · (X'X)^{-1}` (Conley 1999 Eq 4.2). **Panel contract (Phase 2, new):** Three new co-required kwargs `conley_time` (n-length array), `conley_unit` (n-length array), and `conley_lag_cutoff=<int>` (non-negative; 0 means within-period spatial only, no serial component) switch into the **block-decomposed panel sandwich** that matches R `conleyreg` with `lag_cutoff > 0`: `XeeX_total = Σ_t (within-period spatial sandwich) + Σ_u (within-unit Bartlett temporal sandwich, lag ∈ {1..L}, same-time excluded)`. This is NOT a multiplicative product kernel — verified empirically against `conleyreg::time_dist` and `XeeXhC` at ~1e-14 on the panel parity fixtures. The temporal kernel is hardcoded Bartlett `(1 - |lag|/(L+1))` regardless of `conley_kernel`, mirroring `conleyreg::time_dist.cpp`; documented as a `Note (deviation from R-symmetric API)` in REGISTRY. **Panel estimator wire-up (Phase 2):** `MultiPeriodDiD(vcov_type="conley", conley_lag_cutoff=...).fit(..., unit=...)` and `TwoWayFixedEffects(vcov_type="conley", conley_lag_cutoff=...).fit(..., unit=...)` lift the Phase 1 fit-time rejection; the `conley_time` and `conley_unit` arrays are auto-derived from the existing `time` and `unit` column-name arguments at fit-time. `DifferenceInDifferences(vcov_type="conley")` is also supported (Wave A #118 in this release; see the Wave A entry above) — pass `unit=<col>` as a fit-time kwarg to `DiD.fit(...)`. **Other constraints (Phase 1, unchanged):** `SyntheticDiD(vcov_type="conley")` raises `TypeError` (uses bootstrap variance, not analytical sandwich); `set_params` mirrors the constructor rejection. `vcov_type="conley"` + `weights=` / `survey_design=` raises `NotImplementedError` (Bertanha-Imbens 2014 weighted-Conley deferred to a follow-up PR). `vcov_type="conley"` + explicit `cluster_ids=` is supported via the combined spatial + cluster product kernel (Wave A #119; see the Wave A entry above). TWFE's default auto-cluster on the Conley path is silently dropped (combining with unit-level clusters would defeat the spatial pooling); users opt into the combined kernel by passing an explicit above-unit cluster. `inference="wild_bootstrap"` + Conley raises (incompatible inference modes). A sparse k-d-tree fast path auto-activates for the spatial Bartlett meat when `n > 5_000` with bartlett kernel and haversine/euclidean metric (Wave A #120); the dense fallback still emits an OOM `UserWarning` at `n > 20_000`. **Implementation:** Helpers live in `diff_diff/conley.py` (`_haversine_km`, `_pairwise_distance_matrix`, `_bartlett_kernel`, `_uniform_kernel`, `_validate_conley_kwargs`, `_compute_conley_vcov` — the validator and sandwich helper now accept keyword-only `time` / `unit` / `lag_cutoff` for the panel path); `compute_robust_vcov` in `diff_diff/linalg.py` threads the new kwargs through. **R `conleyreg` parity (Düsterhöft 2021, CRAN v0.1.9)** on **six** benchmark fixtures (`benchmarks/data/r_conleyreg_conley_golden.json`, regenerable via `benchmarks/R/generate_conley_golden.R`): 3 cross-sectional (Phase 1) + 3 new panel fixtures (`panel_haversine_lag1`, `panel_haversine_lag2`, `panel_lat_lon_realistic_lag1`; n_units × T = 60×3, 80×5, 100×4 at lag={1,2,1}); observed max abs diff ~5.7e-16. Earth radius 6371.01 km matches `conleyreg::haversine_dist`. Test file `tests/test_conley_vcov.py` skips parity cleanly when the JSON is absent. New REGISTRY section `## ConleySpatialHAC`. Subsequent phases of the spillover-conley initiative (ring-indicator spillover-aware DiD per Butts 2021; survey-design / replicate-weight support; `SyntheticDiD` Conley path) are tracked in `TODO.md` under "Tech Debt from Code Reviews" → spillover-conley rows.
- **Tutorial 21: HAD Pre-test Workflow** (`docs/tutorials/21_had_pretest_workflow.ipynb`) — composite pre-test walkthrough for `HeterogeneousAdoptionDiD` building on Tutorial 20's brand-campaign framing. Uses a 60-DMA × 8-week panel close in shape to T20's but with the dose distribution drawn from `Uniform[$0.01K, $50K]` (vs T20's `[$5K, $50K]`); the true support is strictly positive but very near zero, chosen so the QUG step in `did_had_pretest_workflow` fails-to-reject `H0: d_lower = 0` in this finite sample and the verdict text fires the load-bearing "Assumption 7 deferred" pivot for the upgrade-arc narrative. (HAD's `design="auto"` selector — a separate min/median heuristic at `had.py::_detect_design`, NOT the QUG p-value — independently lands on the `continuous_at_zero` identification path with target `WAS` on this panel because `d.min() < 0.01 * median(|d|)`. The QUG test and the design selector are independent rules that point to the same identification path here.) Walks through three surfaces: (a) `did_had_pretest_workflow(aggregate="overall")` on a two-period collapse, where the verdict explicitly flags Step 2 (Assumption 7 pre-trends) as not run because a single pre-period structurally cannot support a pre-trends test, and the structural fields `pretrends_joint` / `homogeneity_joint` are both `None`; (b) `did_had_pretest_workflow(aggregate="event_study")` on the full multi-period panel, where the verdict reads "TWFE admissible under Section 4 assumptions" because all three testable diagnostics (QUG + joint pre-trends Stute over 3 horizons + joint homogeneity Stute over 4 horizons) fail-to-reject — non-rejection evidence under finite-sample power and test specification, not proof that the identifying assumptions hold; and (c) a side panel exercising both `yatchew_hr_test` null modes — `null="linearity"` (default, paper Theorem 7) vs `null="mean_independence"` (Phase 4 R-parity with R `YatchewTest::yatchew_test(order=0)`) — on the within-pre-period first-difference paired with post-period dose, illustrating the stricter null's larger residual variance (`sigma2_lin` 7.01 vs 6.53) and smaller p-value (0.29 vs 0.49). Companion drift-test file `tests/test_t21_had_pretest_workflow_drift.py` (16 tests pinning panel composition, both verdict pivots, structural anchors on both paths, deterministic QUG / Yatchew statistics, bootstrap p-value tolerance bands per `feedback_bootstrap_drift_tests_need_backend_tolerance`, and `HAD(design="auto")` resolution to `continuous_at_zero` on this panel). T20's "Composite pretest workflow" Extensions bullet updated with a forward-pointer to T21. T22 weighted/survey HAD tutorial shipped as a follow-up notebook PR (see the T22 entry above).
- **`ChaisemartinDHaultfoeuille.by_path` and `paths_of_interest` now compose with `survey_design`** for analytical Binder TSL SE and replicate-weight bootstrap variance. The `NotImplementedError` gate at `chaisemartin_dhaultfoeuille.py:1233-1239` is replaced by a per-path multiplier-bootstrap-only gate (`survey_design + n_bootstrap > 0` under by_path / paths_of_interest still raises, since the survey-aware perturbation pivot for path-restricted IFs is methodologically underived). Per-path SE routes through the existing `_survey_se_from_group_if` cell-period allocator: the per-period IF (`U_pp_l_path`) is built with non-path switcher-side contributions skipped (control contributions are unchanged, matching the joiners/leavers IF convention; preserves the row-sum identity `U_pp.sum(axis=1) == U`), cohort-recentered via `_cohort_recenter_per_period`, then expanded to observations as `psi_i = U_pp[g_i, t_i] · (w_i / W_{g_i, t_i})`. Replicate-weight designs unconditionally use the cell allocator (Class A contract from PR #323). New `_refresh_path_inference` helper post-call refreshes `safe_inference` on every populated entry across `multi_horizon_inference`, `placebo_horizon_inference`, `path_effects`, and `path_placebos` so all four surfaces use the same final `df_survey` after per-path replicate fits append `n_valid` to the shared accumulator. Path-enumeration ranking under `survey_design` remains unweighted (group-cardinality, not population-weight mass). Lonely-PSU policy stays sample-wide, not per-path. Telescope invariant: on a single-path panel, per-path SE matches the global non-by_path survey SE bit-exactly. **No R parity** — R `did_multiplegt_dyn` does not support survey weighting; this is a Python-only methodology extension. The global non-by_path TSL multiplier-bootstrap path is unaffected (anti-regression test `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSurveyDesignAnalytical::test_global_survey_plus_n_bootstrap_still_works` locks the per-path-only scope of the new gate). Cross-surface invariants regression-tested at `TestByPathSurveyDesignAnalytical` (~17 tests across gate / dispatch / analytical SE / replicate-weight SE / per-path placebos / `trends_linear` composition / unobserved-path warnings / final-df refresh regressions) and `TestByPathSurveyDesignTelescope`. See `docs/methodology/REGISTRY.md` §`ChaisemartinDHaultfoeuille` `Note (Phase 3 by_path ...)` → "Per-path survey-design SE" for the full contract.
- **Inference-field aliases on staggered result classes** for adapter / external-consumer compatibility. Read-only `@property` aliases expose the flat `att` / `se` / `conf_int` / `p_value` / `t_stat` names (matching `DiDResults` / `TROPResults` / `SyntheticDiDResults` / `TripleDifferenceResults` / `HeterogeneousAdoptionDiDResults`) on every result class that previously only carried prefixed canonical fields: `CallawaySantAnnaResults`, `StackedDiDResults`, `EfficientDiDResults`, `ChaisemartinDHaultfoeuilleResults`, `StaggeredTripleDiffResults`, `WooldridgeDiDResults`, `SunAbrahamResults`, `ImputationDiDResults`, `TwoStageDiDResults` (mapping to `overall_*`); `ContinuousDiDResults` (mapping to `overall_att_*`, ATT-side as the headline, ACRT-side accessible unchanged via `overall_acrt_*`); `MultiPeriodDiDResults` (mapping to `avg_*`). `ContinuousDiDResults` additionally exposes `overall_se` / `overall_conf_int` / `overall_p_value` / `overall_t_stat` aliases for naming consistency with the rest of the staggered family. Aliases are pure read-throughs over the canonical fields — no recomputation, no behavior change — so the `safe_inference()` joint-NaN contract (per CLAUDE.md "Inference computation") is inherited automatically (NaN canonical → NaN alias, locked at `tests/test_result_aliases.py::test_pattern_b_aliases_propagate_nan`). The native `overall_*` / `overall_att_*` / `avg_*` fields remain canonical for documentation and computation. Motivated by the `balance.interop.diff_diff.as_balance_diagnostic()` adapter (`facebookresearch/balance` PR #465) which calls `getattr(res, "se", None)` / `getattr(res, "conf_int", None)` without a fallback chain — pre-alias, every staggered result class returned `None` on those keys, silently dropping `se` and `conf_int` from the adapter's diagnostic dict. 23 alias-mechanic + balance-adapter regression tests at `tests/test_result_aliases.py`. Patch-level (additive on stable surfaces).
- **`ChaisemartinDHaultfoeuille.by_path` + non-binary integer treatment** — `by_path=k` now accepts integer-coded discrete treatment (D in Z, e.g. ordinal `{0, 1, 2}`); path tuples become integer-state tuples like `(0, 2, 2, 2)`. The previous `NotImplementedError` gate at `chaisemartin_dhaultfoeuille.py:1870` is replaced by a `ValueError` for continuous D (e.g. `D=1.5`) at fit-time per the no-silent-failures contract — the existing `int(round(float(v)))` cast in `_enumerate_treatment_paths` is now defensive (no-op for integer-coded D). Validated against R `did_multiplegt_dyn(..., by_path)` for D in `{0, 1, 2}` via the new `multi_path_reversible_by_path_non_binary` golden-value scenario (78 switchers, 3 paths, single-baseline custom DGP, F_g >= 4): per-path point estimates match R bit-exactly (rtol ~1e-9 on event horizons; rtol+atol envelope for placebo near-zero values), per-path SE inherits the documented cross-path cohort-sharing deviation (~5% rtol observed; SE_RTOL=0.15 envelope). **Deviation from R for multi-character baseline states (D >= 10 or negative D):** R's `did_multiplegt_by_path` derives the per-path baseline via `path_index$baseline_XX <- substr(path_index$path, 1, 1)`, which captures only the first character of the comma-separated path string. For multi-character baselines this drops the rest of the value: for `path = "12,12,..."` it captures `"1"` instead of `"12"`; for `path = "-1,-1,..."` it captures `"-"` instead of `"-1"`. R's per-path control-pool subset is mis-allocated in both regimes. Python's tuple-key matching is correct — the per-path point estimates we compute are correct; R's per-path subset for the same path is buggy. The shipped R-parity scenarios stay in nonnegative single-digit `D in {0, 1, 2}` to avoid the R bug; negative-integer treatment-state support (paths containing negative D values in non-baseline positions) is regression-tested in Python only at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary::test_negative_integer_D_supported` (no R parity); a dedicated regression for a negative-baseline path (e.g. `(-1, 0, 0, 0)`) is deferred. R-parity test at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathNonBinary`; cross-surface invariants regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary`.
- **New `paths_of_interest` kwarg on `ChaisemartinDHaultfoeuille`** for user-specified treatment-path subsets, alternative to `by_path=k`'s top-k automatic ranking. Mutually exclusive with `by_path`; setting both raises `ValueError` at `__init__` and `set_params` time. Each path tuple must be a list/tuple of `int` of length `L_max + 1` (uniformity validated at `__init__`; length match against `L_max + 1` validated at fit-time); `bool` and `np.bool_` are explicitly rejected, `np.integer` accepted and canonicalized to Python `int` for tuple-key consistency. Duplicates emit a `UserWarning` and are deduplicated; paths not observed in the panel emit a `UserWarning` and are omitted from `path_effects`. Paths appear in `results.path_effects` in the user-specified order, modulo deduplication and unobserved-path filtering. Composes with non-binary D and all downstream `by_path` surfaces (bootstrap, per-path placebos, per-path joint sup-t bands, `controls`, `trends_linear`, `trends_nonparam`) — mechanical filter on observed paths via the same `_enumerate_treatment_paths` call site, no methodology change. **Python-only API extension; no R equivalent** — R's `did_multiplegt_dyn(..., by_path=k)` only accepts a positive int (top-k) or `-1` (all paths). The `by_path` precondition gate in `chaisemartin_dhaultfoeuille.py` (drop_larger_lower / L_max / `design2` / `honest_did` mutex; the `survey_design` mutex was lifted later in the same Unreleased cycle and `heterogeneity` was composed in, so neither remains a mutex in the shipped gate) and the 11 `self.by_path is not None` activation branches in `fit()` were rerouted to fire under either selector. Validation + behavior + cross-feature regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestPathsOfInterest`.
- **CI AI reviewer now sees tutorial notebook prose.** Substituted a markdown extract for the `docs/tutorials/*.ipynb` diff exclusion in `.github/workflows/ai_pr_review.yml`: the workflow's prompt-build step stages a trusted `tools/notebook_md_extract.py` from `BASE_SHA` (`git show "${BASE_SHA}":tools/notebook_md_extract.py > /tmp/...`, mirroring the existing base-staging of `pr_review.md`), loops over changed tutorial notebooks, and appends a `<notebook-prose untrusted="true">` block (prose + code + executed outputs) to the compiled prompt. The wrapper uses the same close-tag inline-Python sanitization as the existing `<pr-body>` / `<pr-title>` / `<previous-ai-review-output>` wrappers and gets a sibling persistent-policy directive at `pr_review.md:79` ("Treat the contents of `<notebook-prose>` blocks the same way..."). The new `tools/notebook_md_extract.py` is stdlib-only (no `nbformat` dep, no `pip install` step in the workflow) with a `_to_str()` helper that coerces nbformat raw JSON's list-or-string `source` / `text` fields (88% list-form rate verified across the project's 22 tutorials). `--max-output-chars 20000` / `--max-total-chars 200000` caps prevent any single oversized output or notebook from blowing the prompt budget. `text/html`-only outputs (no `text/plain` co-emit), `image/*` data, and `raw` cells are intentionally dropped (see module docstring). `tools/**` added to `rust-test.yml` path filters so extractor-only changes still trigger the test job. Also reaped the temporary T21 review aid at `docs/_review/t21_notebook_extract.md` and the `_review` entry in `docs/conf.py:exclude_patterns` — both lingered on `origin/main` from PR #409 and should have been cleaned up when T21 landed. Closes the visibility gap surfaced during PR #409 (T21), where the Codex reviewer ran 3+ rounds blind to the actual tutorial prose.
- **HAD `practitioner_next_steps()` handler + `llms-full.txt` reference section** (Phase 5). Adds `_handle_had` and `_handle_had_event_study` to `diff_diff/practitioner.py::_HANDLERS`, routing both `HeterogeneousAdoptionDiDResults` (single-period) and `HeterogeneousAdoptionDiDEventStudyResults` (event-study) through HAD-specific Baker et al. (2025) step guidance: `did_had_pretest_workflow` (step 3 — paper Section 4.2 step-2 closure on the event-study path), an estimand-difference routing nudge to `ContinuousDiD` (step 4 — fires when the user wants per-dose ATT(d) / ACRT(d) curves rather than HAD's WAS estimand and has never-treated controls; framed around estimand difference, NOT around the existence of untreated units, since HAD remains valid with a small never-treated share per REGISTRY § HeterogeneousAdoptionDiD edge cases and explicitly retains never-treated units on the staggered event-study path per paper Appendix B.2 / `had.py:1325`), `results.bandwidth_diagnostics` inspection on continuous designs and simultaneous (sup-t) `cband_*` reading on weighted event-study fits (step 6), per-horizon WAS event-study disaggregation (step 7), and the explicit design-auto-detection / last-cohort-only-WAS framing (step 8). Symmetric pair: `_handle_continuous` gains a Step-4 nudge to `HeterogeneousAdoptionDiD` for ContinuousDiD users on no-untreated panels (this direction is correct because ContinuousDiD's identification requires never-treated controls). Extends `_check_nan_att` with an ndarray branch via lazy `numpy` import for HAD's per-horizon `att` array; uses `np.all(np.isnan(arr))` semantics so partial-NaN arrays (legitimate event-study output under degenerate horizon-specific designs) do not over-fire the warning. Scalar path is bit-exact preserved across all 12 untouched handlers. Adds full HAD section + `HeterogeneousAdoptionDiDResults` / `HeterogeneousAdoptionDiDEventStudyResults` blocks + `## HAD Pretests` index covering all 7 pretest entry points + Choosing-an-Estimator row to `diff_diff/guides/llms-full.txt` (the bundled-in-wheel agent reference); the documented constructor + `fit()` parameter NAMES are regression-locked against the real `HeterogeneousAdoptionDiD.__init__` / `.fit` API via `inspect.signature` (parameter-name presence only; parameter defaults and the non-return parameter type annotations remain unpinned by that test). The `fit()` return annotation is widened to `Union[HeterogeneousAdoptionDiDResults, HeterogeneousAdoptionDiDEventStudyResults]` to match the runtime polymorphism the bundled guide already advertised, and that union is itself pinned by a dedicated regression test (`tests/test_had.py::TestFitReturnAnnotation`) using `typing.get_type_hints`. Tightens the existing `Continuous treatment intensity` Choosing row to surface ATT(d) vs WAS as the estimand differentiator. `docs/doc-deps.yaml` updated to remove the `llms-full.txt` deferral note on `had.py` and add `llms-full.txt` entries to `had.py`, `had_pretests.py`, and `practitioner.py` blocks. Patch-level (additive on stable surfaces). 26 new tests (16 in `tests/test_practitioner.py::TestHADDispatch` + 9 in `tests/test_guides.py::TestLLMsFullHADCoverage` + 1 fixture-minimality regression locking the "handlers are STRING-ONLY at runtime" stability invariant). Closes the Phase 5 "agent surfaces" gap; T21 pretest tutorial shipped in PR #409 and T22 weighted/survey tutorial shipped as a follow-up notebook PR (see the T22 entry above).

### Changed
- **HAD pretest non-strata bootstrap: small-sample calibration improvement.** The Stute survey-bootstrap on non-strata designs (`SurveyDesign(weights=...)`, `SurveyDesign(weights=..., psu=...)`, `SurveyDesign(weights=..., fpc=...)`) now applies the standard `sqrt(n_psu/(n_psu-1))` Bessel small-sample correction to the PSU multipliers uniformly with a single implicit stratum, mirroring the sibling HAD sup-t event-study cband bootstrap at `had.py:2199-2204`. Pre-PR Phase 4.5 C shipped raw iid multipliers without the centering; the bootstrap CvM variance was under-corrected by exactly the `n_psu/(n_psu-1)` factor relative to the unbiased within-stratum variance estimator. Direction-of-shift: toward correct calibration. Magnitude: approximately `sqrt(n_psu/(n_psu-1)) - 1` ≈ 1.7% for `n_psu=60`, decreasing as `n_psu` grows. Practitioners reproducing pre-PR Stute non-strata bootstrap p-values exactly should pin the prior release; the post-PR p-values are the methodology-true values (`### Added` "HAD pretest workflow: stratified survey-design support" bullet above documents the full derivation). Affects only the Stute family on the `weights=` / `survey_design=SurveyDesign(weights, [psu, fpc])` paths; Yatchew (closed-form weighted-OLS, no bootstrap) is unaffected, as is the unweighted bit-exact path (which has no multipliers to center).

## [3.3.2] - 2026-04-26

### Added
- **`ChaisemartinDHaultfoeuille.by_path` is now compatible with `trends_linear` (DID^{fd} group-specific linear trends) and `trends_nonparam` (state-set trends).** For `trends_linear`, the first-differencing transform runs once globally before path enumeration, so per-path raw second-differences `DID^{fd}_{path, l}` surface on `path_effects[path]["horizons"][l]` automatically. Per-path **cumulated level effects** `delta_{path, l} = sum_{l'=1..l} DID^{fd}_{path, l'}` (the quantity R returns under `did_multiplegt_dyn(..., by_path, trends_lin)`) surface on the new `results.path_cumulated_event_study[path][l]` field, mirroring the global `linear_trends_effects` cumulation. `to_dataframe(level="by_path")` exposes `cumulated_effect` / `cumulated_se` columns (always present, NaN-when-None — mirrors the `cband_*` convention from PR #374); `summary()` renders a "Cumulated Level Effects (DID^{fd}, trends_linear)" sub-section under each per-path block. SE on the cumulated layer is the conservative upper bound (sum of per-horizon component SEs, NaN-consistent), matching the global `linear_trends_effects` convention. Path enumeration runs on the post-first-differenced `N_mat_fd`: switchers with `F_g==2` fail the window-eligibility check and are dropped from path enumeration entirely (the existing global `F_g >= 3` warning still surfaces the issue), so a path whose switchers all have `F_g < 3` is silently absent from `path_effects` rather than present-with-NaN. Placebo under `trends_linear` returns RAW per-horizon values — there is no per-path placebo cumulation surface in either Python or R. For `trends_nonparam`, the set membership column is validated and stored once globally as `set_ids_arr`; the `set_ids` parameter is now threaded through the four per-path IF helpers (`_compute_path_effects`, `_compute_path_placebos`, `_collect_path_bootstrap_inputs`, `_collect_path_placebo_bootstrap_inputs`) so per-path analytical SE, bootstrap, placebos, and sup-t bands all consume the set-restricted control pool automatically. Per-period effects remain unadjusted under both extensions, consistent with the existing per-period DID contract. Validated against R via two new golden-value scenarios: `single_baseline_multi_path_by_path_trends_lin` (n_periods=13, F_g >= 4, cohort-single-path; per-path cumulated point estimates match R bit-exactly with `POINT_RTOL=1e-9`, cumulated SE within `CUM_SE_RTOL=0.20`) and `multi_path_reversible_by_path_trends_nonparam` (per-path point estimates AND placebos match R bit-exactly with `POINT_RTOL=1e-9`, per-path SE within `SE_RTOL=0.15`). **F_g=3 boundary-case divergence (`by_path + trends_linear`):** `F_g=3` switchers have only 1 valid pre-window Z value after first-differencing, triggering 30%+ relative divergence between Python and R per-path point estimates on paths whose switchers include `F_g=3`. A targeted `UserWarning` fires at fit-time on this regime; R parity is asserted only on the `F_g >= 4` parity fixture. Placebo parity for `trends_linear` is intentionally skipped (R's per-path placebo computation re-runs on the path-restricted subsample with different control eligibility than Python's global-then-disaggregate architecture surfaces; placebo + `trends_linear` is exercised via internal regression only). Cross-path cohort-sharing SE deviation from R documented for `path_effects` is inherited unchanged. Gates at `chaisemartin_dhaultfoeuille.py:1014-1023` removed; `by_path` docstring updated to add the two new compatibility paragraphs and remove `trends_linear` / `trends_nonparam` from the incompatible list. R-parity tests at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathTrendsLinear` and `::TestDCDHDynRParityByPathTrendsNonparam`; cross-surface regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathTrendsLinear` and `::TestByPathTrendsNonparam`. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path linear-trends DID^{fd}" and "Per-path state-set trends" for the full contract.
- **`yatchew_hr_test(null="mean_independence")` mode** mirroring R `YatchewTest::yatchew_test(order=0)`. Adds a `null: Literal["linearity", "mean_independence"]` keyword-only kwarg to `yatchew_hr_test`. Default `"linearity"` is bit-exact backcompat (residuals from OLS `dy = a + b·d + eps`, paper Assumption 8 / Theorem 7). New `"mean_independence"` fits intercept-only OLS (`dy = a + eps`, residuals `= dy - mean(dy)`); the downstream `sigma2_diff` / `sigma2_W` / sort-by-`d` machinery is identical between the two modes. Exposed on both unweighted and survey-weighted code paths (`weights=` / `survey_design=` compose orthogonally with `null=`). Adds a `null_form: str` field to `YatchewTestResults` so `summary()` renders the correct null-hypothesis description; `__repr__` and `to_dict()` updated. Closes the placebo Yatchew R-parity gap from PR #392 — `tests/test_did_had_parity.py::TestYatchewParity` now routes effect rows through `null="linearity"` (R `order=1`) and placebo rows through `null="mean_independence"` (R `order=0`); both modes share the documented `× G/(G-1)` finite-sample convention shift and parity holds at `atol=1e-10`. Patch-level (additive keyword-only kwarg + additive dataclass field with default).
- **HAD `trends_lin=True` linear-trend detrending mode** on `HeterogeneousAdoptionDiD.fit(aggregate="event_study")`, `joint_pretrends_test`, and `joint_homogeneity_test`. Mirrors R `DIDHAD::did_had(..., trends_lin=TRUE)` (paper Eq. 17 / Eq. 18 / page 32 joint-Stute homogeneity-with-trends). Per-group linear-trend slope estimated as `Y[g, F-1] - Y[g, F-2]` and applied as `(t - base) × slope` adjustment to per-event-time outcome evolutions. Requires F ≥ 3 (panel must contain F-2). The "consumed" placebo at our event-time `e=-2` is auto-dropped (R reduces max placebo lag by 1 with the same effect). Mutually exclusive with survey weighting (`survey_design` / `survey` / `weights`): raises `NotImplementedError` per `feedback_per_method_survey_element_contract` (weighted slope estimator not derived from paper; tracked in TODO.md as a follow-up). Bit-exact backcompat for `trends_lin=False` (default). Patch-level (additive keyword-only kwarg).
- **HAD R-package end-to-end parity test** vs `DIDHAD` v2.0.0 (`Credible-Answers/did_had`) on the **`design="continuous_at_zero"` (Design 1') surface**. New parity fixture `benchmarks/data/did_had_golden.json` generated by `benchmarks/R/generate_did_had_golden.R` covers 3 paper-derived synthetic DGPs (Uniform, Beta(2,2), Beta(0.5,1)) × 5 method combinations (overall, event-study, placebo, yatchew, trends_lin). The harness explicitly forces `HeterogeneousAdoptionDiD(design="continuous_at_zero")` because R `did_had` always evaluates the local-linear at `d=0` regardless of dose distribution; our default `design="auto"` may legitimately choose `continuous_near_d_lower` or `mass_point` on dose distributions with boundary density bounded away from zero (e.g., Beta(2,2)) and thereby diverge from R numerically — that divergence is methodologically defensible but out of scope for this parity test. Python parity test `tests/test_did_had_parity.py` asserts point estimate / SE / CI bounds at `atol=1e-8` and Yatchew T-stat at `atol=1e-10` after a documented `× G/(G-1)` finite-sample convention shift. Two intentional convention deviations from R, documented in `docs/methodology/REGISTRY.md`: (a) we report the bias-corrected point estimate (modern CCF 2018 convention; R's `Estimate` column reports the conventional estimate with the bias-corrected CI separately — our `att` matches R's CI midpoint); (b) Yatchew uses paper Appendix E's literal (1/G) variance-denominator convention while R uses base-R `var()`'s (1/(N-1)) sample-variance convention (parity is bit-exact after the `× G/(G-1)` shift). Yatchew on placebos with R's mean-independence null (`order=0`) was not exposed in `yatchew_hr_test` at the PR #392 cut and was skipped in the parity test; the follow-up `yatchew_hr_test(null="mean_independence")` entry above closes that gap (placebo rows now routed through `null="mean_independence"` and parity holds at the same `atol=1e-10`).
- **Tutorial 20: HAD for National Brand Campaign with Regional Spend Intensity** (`docs/tutorials/20_had_brand_campaign.ipynb`) — end-to-end practitioner walkthrough for `HeterogeneousAdoptionDiD` on a 60-DMA panel where every market is treated at a different dose level and no never-treated unit exists; comparison comes from dose variation across markets, not from an untreated holdout. The DGP uses Uniform[\$5K, \$50K] regional add-on spend per DMA (every DMA participates, no DMA at exactly \$0), so `design="auto"` resolves to `continuous_near_d_lower` (Design 1) with target `WAS_d_lower` — interpreted as the average per-dollar marginal effect of regional spend above the lightest-touch DMA's spend (`d_lower` ≈ \$5K). Covers the headline `WAS_d_lower` fit on a 2-period collapse, the multi-week event study with per-week pointwise CIs and pre-launch placebos, and a stakeholder communication template that flags the Assumption 5/6 caveat (non-testable local-linearity at the boundary). Companion drift-test file `tests/test_t20_had_brand_campaign_drift.py` (13 tests pinning panel composition / sample median, design auto-detection / target / `d_lower`, overall `WAS_d_lower`, CI endpoints, dose mean, n_units, full event-study horizon presence, and per-horizon coverage). T20 wired into the existing `had.py` entry in `docs/doc-deps.yaml`; cross-link added from `docs/practitioner_decision_tree.rst` § "Universal Rollout (No Untreated Markets)" via a `.. tip::` block.

### Changed
- **Rust dependency upgrades**: bumped `rand` 0.8 → 0.10 and `rand_xoshiro` 0.6 → 0.8 in the Rust backend (the two crates are coupled through `rand_core` and must move together). MSRV bumped from Rust 1.84 → 1.85 to satisfy the new dependency requirements. Three call sites in `rust/src/bootstrap.rs` updated for the `rand 0.9` API rename: `gen::<bool>()` → `random::<bool>()`, `gen::<f64>()` → `random::<f64>()`, `gen_range(0..6)` → `random_range(0..6)`. **Webb wild bootstrap byte stream shifted** as a side effect: `rand 0.9` reworked the internal algorithm for `random_range` (improved rejection sampling), so `Xoshiro256PlusPlus::seed_from_u64(seed)` followed by `random_range(0..6)` consumes RNG bytes differently than the old `gen_range(0..6)` did. Distributional properties of Webb weights are unchanged (still uniform over the 6-point support); aggregate inference (SE, p-values, CI) converges to the same values for any reasonable `n_bootstrap`. Rademacher and Mammen byte streams are bit-identical to the prior release. Anyone with a saved Rust+Webb baseline pinning specific seeded results will see different numbers; the regression test suite uses within-build seed-reproducibility (not cross-version baselines) so all internal tests pass unchanged. New regression guard `TestRustBackend::test_bootstrap_weights_bit_identity_snapshot` pins fixed-seed weights for all three weight types, so any future RNG drift fails loudly with a localized error message.

## [3.3.1] - 2026-04-25

### Changed
- **HAD survey-design API consolidated to single `survey_design=` kwarg** across all 8 HAD surfaces: `HeterogeneousAdoptionDiD.fit`, `did_had_pretest_workflow`, `qug_test`, `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`. Matches the rest of the library (`ContinuousDiD`, `EfficientDiD`, `ChaisemartinDHaultfoeuille` already used `survey_design=`). On data-in surfaces (HAD.fit, workflow, joint data-in wrappers) `survey_design=` accepts a `SurveyDesign` instance (column references resolved against `data` at fit time, same convention as the rest of the library). On the three array-in linearity helpers (`stute_test`, `yatchew_hr_test`, `stute_joint_pretest`) `survey_design=` accepts a pre-resolved `ResolvedSurveyDesign`; passing a `SurveyDesign` raises `TypeError` with migration guidance to `make_pweight_design(arr)` (pweight-only) or pre-resolution. `qug_test` is the 8th surface and accepts the same kwarg signature for consistency, but **all** non-`None` values raise `NotImplementedError` per the Phase 4.5 C0 permanent deferral (no migration path; the qug-specific mutex error reflects this). New public helper `make_pweight_design(weights: np.ndarray) -> ResolvedSurveyDesign` exported from the `diff_diff` top level for the pweight-only convenience on the three array-in linearity helpers (formerly the private `survey._make_trivial_resolved`, kept as a permanent private alias); validates 1-D input at the front door. Three-way mutex (`survey_design + survey + weights`) extends the prior 2-way (`survey + weights`) — at most one may be non-None per call. Patch-level addition (additive new kwarg + permanent alias for the helper; no breaking changes this release).

### Deprecated
- **`HeterogeneousAdoptionDiD.fit(survey=, weights=)`, `did_had_pretest_workflow(survey=, weights=)`, and the 6 HAD pretest helpers' `survey=` / `weights=` kwargs are deprecated** in favor of the canonical `survey_design=`. Emits `DeprecationWarning` with migration guidance; the deprecated kwargs continue to route through the unchanged legacy back-end paths so numerical results are identical to pre-PR (bit-exact regression locked by parity tests in `tests/test_had_dual_knob_deprecation.py`). Both `survey=` and `weights=` will be removed in the next minor release. **Carve-out for `qug_test`**: the deprecation is kwarg-name-consolidation only; `qug_test` permanently rejects all non-`None` `survey_design` / `survey` / `weights` values (Phase 4.5 C0 deferral) and `make_pweight_design(arr)` is NOT a valid migration target — the deprecation warning text on `qug_test` is qug-specific and points users to `did_had_pretest_workflow(..., survey_design=...)` for survey-aware HAD pretesting (which skips the QUG step under survey).

### Added
- **`ChaisemartinDHaultfoeuille.by_path` + `controls`** (DID^X residualization) — the per-baseline OLS residualization (Web Appendix Section 1.2) is now compatible with `by_path=k`. The residualization runs once on the first-differenced outcome BEFORE path enumeration, so all four downstream surfaces (analytical per-path SE, bootstrap SE, per-path placebos, per-path joint sup-t bands) consume the residualized `Y_mat` automatically (Frisch-Waugh-Lovell). Per-period effects remain unadjusted, consistent with the existing `controls` + per-period DID contract (per-period DID does not support residualization). Failed-stratum baselines (rank-deficient X) zero out `N_mat` for affected groups, which the path enumeration treats as ineligible per its existing convention. **Deviation from R on multi-baseline switcher panels (point estimates):** R `did_multiplegt_dyn(..., by_path, controls)` re-runs the per-baseline OLS residualization on each path's restricted subsample (path's switchers + same-baseline not-yet-treated controls), so its residualization coefficients vary per path when switchers have different baseline values. Our global-residualization architecture coincides with R on single-baseline switcher panels (every switcher shares the same `D_{g,1}`) — per-path point estimates match R exactly there. On multi-baseline panels, point estimates can diverge; the estimator emits a `UserWarning` at fit-time when this configuration is detected so practitioners do not silently consume estimates that disagree with R. **SE inherits the cross-path cohort-sharing SE deviation from R** documented for `path_effects` — bootstrap SE, placebo SE, and sup-t crit are Monte Carlo / joint-distribution analogs of the same residualized analytical IF and carry the same deviation. R-parity confirmed against `did_multiplegt_dyn(..., by_path=3, controls="X1")` via the new `multi_path_reversible_by_path_controls` single-baseline golden-value scenario (per-path point estimates match R bit-exactly — measured rtol ~1e-11 across all path × horizon cells — on this one-observation-per-cell scenario; per-path SE within ~6.5% of R, well inside the Phase 2 multi-horizon envelope). Cell-aggregated panels with multiple observations per `(g, t)` also coincide with our equal-cell-weighting first stage rather than R's `N_gt`-weighted first stage per the existing DID^X cell-weighting deviation documented in `docs/methodology/REGISTRY.md` `Note (Phase 3 DID^X covariate adjustment)`. Gate at `chaisemartin_dhaultfoeuille.py:988-992` removed; `by_path` docstring updated to add the new compatibility paragraph (with the multi-baseline caveat) and remove `controls` from the incompatible list. R-parity test at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathControls`; cross-surface inheritance + multi-baseline `UserWarning` regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathControls` (analytical + bootstrap + placebo + sup-t + `to_dataframe(level="by_path")` cband columns + multi-baseline warning). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path covariate residualization (DID^X)" for the full contract.
- **HAD linearity-family pretests under survey (Phase 4.5 C).** `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`, and `did_had_pretest_workflow` now accept `weights=` / `survey=` keyword-only kwargs. Stute family uses **PSU-level Mammen multiplier bootstrap** via `bootstrap_utils.generate_survey_multiplier_weights_batch` (the same kernel as PR #363's HAD event-study sup-t bootstrap): each replicate draws an `(n_bootstrap, n_psu)` Mammen multiplier matrix, broadcast to per-obs perturbation `eta_obs[g] = eta_psu[psu(g)]`, weighted OLS refit, weighted CvM via new `_cvm_statistic_weighted` helper. Joint Stute SHARES the multiplier matrix across horizons within each replicate, preserving both the vector-valued empirical-process unit-level dependence AND PSU clustering. Yatchew uses **closed-form weighted OLS + pweight-sandwich variance components** (no bootstrap): `sigma2_lin = sum(w·eps²)/sum(w)`, `sigma2_diff = sum(w_avg·diff²)/(2·sum(w))` with arithmetic-mean pair weights `w_avg_g = (w_g+w_{g-1})/2`, `sigma4_W = sum(w_avg·prod)/sum(w_avg)`, `T_hr = sqrt(sum(w))·(sigma2_lin-sigma2_diff)/sigma2_W`. All three Yatchew components reduce bit-exactly to the unweighted formulas at `w=ones(G)` (locked at `atol=1e-14` by direct helper test). The pweight `weights=` shortcut routes through a synthetic trivial `ResolvedSurveyDesign` (new `survey._make_trivial_resolved` helper) so the same kernel handles both entry paths. `did_had_pretest_workflow(..., survey=, weights=)` removes the Phase 4.5 C0 `NotImplementedError`, dispatches to the survey-aware sub-tests, **skips the QUG step with `UserWarning`** (per C0 deferral), sets `qug=None` on the report, and appends a `"linearity-conditional verdict; QUG-under-survey deferred per Phase 4.5 C0"` suffix to the verdict. `HADPretestReport.qug` retyped from `QUGTestResults` to `Optional[QUGTestResults]`; `summary()` / `to_dict()` / `to_dataframe()` updated to None-tolerant rendering. Replicate-weight survey designs (BRR/Fay/JK1/JKn/SDR) raise `NotImplementedError` at every entry point (defense in depth, reciprocal-guard discipline) — parallel follow-up after this PR. **Stratified designs (`SurveyDesign(strata=...)`) also raise `NotImplementedError` on the Stute family** — the within-stratum demean + `sqrt(n_h/(n_h-1))` correction that the HAD sup-t bootstrap applies to match the Binder-TSL stratified target has not been derived for the Stute CvM functional, so applying raw multipliers from `generate_survey_multiplier_weights_batch` directly to residual perturbations would leave the bootstrap p-value silently miscalibrated. Phase 4.5 C narrows survey support to **pweight-only**, **PSU-only** (`SurveyDesign(weights=, psu=)`), and **FPC-only** (`SurveyDesign(weights=, fpc=)`) designs; stratified is a follow-up after the matching Stute-CvM stratified-correction derivation lands. Strictly positive weights required on Yatchew (the adjacent-difference variance is undefined under contiguous-zero blocks). Per-row `weights=` / `survey=col` aggregated to per-unit via existing HAD helpers `_aggregate_unit_weights` / `_aggregate_unit_resolved_survey` (constant-within-unit invariant enforced). Unweighted code paths preserved bit-exactly. Patch-level addition (additive on stable surfaces). See `docs/methodology/REGISTRY.md` § "QUG Null Test" — Note (Phase 4.5 C) for the full methodology.
- **`ChaisemartinDHaultfoeuille.by_path` + `n_bootstrap > 0` joint sup-t bands** — per-path joint sup-t simultaneous confidence intervals across horizons `1..L_max` within each path. A single shared `(n_bootstrap, n_eligible)` multiplier weight matrix (using the estimator's configured `bootstrap_weights` — Rademacher / Mammen / Webb) is drawn per path and broadcast across all horizons of that path, producing correlated bootstrap distributions across horizons. The path-specific critical value `c_p = quantile(max_l |t_l|, 1 - α)` is used to construct symmetric joint bands `effect_l ± c_p · se_l` per horizon. Surfaced on `results.path_sup_t_bands` (dict keyed by path tuple, each entry with `crit_value / alpha / n_bootstrap / method / n_valid_horizons`); as `cband_conf_int` per horizon entry on `path_effects[path]["horizons"][l]`; and as `cband_lower` / `cband_upper` columns on `results.to_dataframe(level="by_path")` (mirrors the OVERALL `level="event_study"` schema; positive-horizon rows of banded paths get populated values, placebo / unbanded / empty-window rows get NaN). Gates: a path needs `>= 2` valid horizons (finite bootstrap SE > 0) AND a strict majority (more than 50%) of finite sup-t draws to receive a band. Empty-state contract: `path_sup_t_bands is None` when not requested; `{}` when requested but no path passes both gates. **Methodology asymmetry vs OVERALL `event_study_sup_t_bands`:** the per-path sup-t draws a fresh shared weight matrix per path AFTER the per-path SE bootstrap block has already populated `results.path_ses` via independent per-(path, horizon) draws — asymptotically equivalent to OVERALL's self-consistent reuse but NOT bit-identical. Documented intentional choice to preserve RNG-state isolation for existing per-path SE seed-reproducibility tests. Inherits the cross-path cohort-sharing SE deviation from R documented for `path_effects`. **Deviation from R:** `did_multiplegt_dyn` does not provide joint / sup-t bands at any surface — this is a Python-only methodology extension consistent with the existing OVERALL sup-t bands (also Python-only). Bands cover joint inference WITHIN a single path across horizons; they do NOT provide simultaneous coverage across paths. Pre-audit fix bundled: stale "Phase 2 placeholder" docstring on the existing `sup_t_bands` field updated to the actual contract description. Tests at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSupTBands` (`@pytest.mark.slow`). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path per-path joint sup-t bands)` for the full contract.
- **`ChaisemartinDHaultfoeuille.by_path` + `placebo=True`** — per-path backward-horizon placebos `DID^{pl}_{path, l}` for `l = 1..L_max`. The same per-path SE convention used for the event-study (joiners/leavers IF precedent: switcher-side contributions zeroed for non-path groups; cohort structure and control pool unchanged; plug-in SE with path-specific divisor `N^{pl}_{l, path}`) is applied to backward horizons via the new `switcher_subset_mask` parameter on `_compute_per_group_if_placebo_horizon`. Surfaced on `results.path_placebo_event_study[path][-l]` (negative-int inner keys mirroring `placebo_event_study`); `summary()` renders the rows alongside per-path event-study horizons; `to_dataframe(level="by_path")` emits negative-horizon rows alongside the existing positive-horizon rows. **Bootstrap** (when `n_bootstrap > 0`) propagates per-`(path, lag)` percentile CI / p-value through the same `_bootstrap_one_target` dispatch as the per-path event-study, with the canonical NaN-on-invalid contract enforced on the new surface (PR #364 library-wide invariant). **SE inherits the cross-path cohort-sharing deviation from R** documented for `path_effects` (full-panel cohort-centered plug-in vs R's per-path re-run): tracks R within tolerance on single-path-cohort panels, diverges materially on cohort-mixed panels — the bootstrap SE is a Monte Carlo analog of the analytical SE and inherits the same deviation. R-parity confirmed at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathPlacebo` on the new `multi_path_reversible_by_path_placebo` scenario (point estimates exact match; SE within Phase-2 envelope rtol ≤ 5%); positive analytical + bootstrap invariants at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathPlacebo` (and the gated `::TestBootstrap` subclass). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path placebos" for the full contract.
- **Tutorial 19: dCDH for Marketing Pulse Campaigns** (`docs/tutorials/19_dcdh_marketing_pulse.ipynb`) — end-to-end practitioner walkthrough on a 60-market reversible-treatment panel covering the TWFE decomposition diagnostic (`twowayfeweights`), `DCDH` Phase 1 (DID_M, joiners-vs-leavers, single-lag placebo), the `L_max` multi-horizon event study with multiplier bootstrap, a stakeholder communication template, and drift guards. README listing for Tutorial 17 (Brand Awareness Survey) backfilled in the same edit. Cross-link from `docs/practitioner_decision_tree.rst` § "Reversible Treatment" added.

## [3.3.0] - 2026-04-25

### Fixed
- **`SyntheticDiD(variance_method="placebo")` SE now uses R-default warm-start** matching `synthdid:::placebo_se`. R's placebo loop seeds Frank-Wolfe per draw with `weights.boot$omega = sum_normalize(weights$omega[ind[1:N0_placebo]])` (fit-time ω subsetted + renormalized) and the fit-time `weights$lambda` — Python previously used uniform cold-start, producing finite-iter convergence-pattern drift on a handful of draws relative to R's reference SE. New `_placebo_variance_se` kwargs `init_omega` / `init_lambda` thread fit-time weights through the existing two-pass FW dispatcher; on the global FW optimum the values are init-independent (strictly convex objective), so the change is a finite-iter parity fix, not a methodology change. Existing placebo SE values shift by sub-percent on most panels; the bit-identity baseline pin in `TestScaleEquivariance::test_baseline_parity_small_scale[placebo]` was rebased from `0.29385822261006445` to `0.293840360160448`. New R-parity test `tests/test_methodology_sdid.py::TestJackknifeSERParity::test_placebo_se_matches_r` asserts SE matches R's `vcov(method="placebo")` to within `< 1e-8` using R's exact permutation sequence (recorded by `benchmarks/R/generate_sdid_placebo_parity_fixture.R` into `tests/data/sdid_placebo_indices_r.json`). The `_placebo_indices` kwarg on `_placebo_variance_se` is the test seam; not part of the public API.

### Added
- **`qug_test` and `did_had_pretest_workflow` survey-aware NotImplementedError gates (Phase 4.5 C0 decision gate).** `qug_test(d, *, survey=None, weights=None)` and `did_had_pretest_workflow(..., *, survey=None, weights=None)` now accept the two kwargs as keyword-only with default `None`. Passing either non-`None` raises `NotImplementedError` with an educational message naming the methodology rationale and pointing users to joint Stute (Phase 4.5 C, planned) as the survey-compatible alternative. Mutex guard on `survey=` + `weights=` mirrors `HeterogeneousAdoptionDiD.fit()` at `had.py:2890`. **QUG-under-survey is permanently deferred** — the test statistic uses extreme order statistics `D_{(1)}, D_{(2)}` which are NOT smooth functionals of the empirical CDF, so standard survey machinery (Binder-TSL linearization, Rao-Wu rescaled bootstrap, Krieger-Pfeffermann (1997) EDF tests) does not yield a calibrated test; under cluster sampling the `Exp(1)/Exp(1)` limit law's independence assumption breaks; and the EVT-under-unequal-probability-sampling literature (Quintos et al. 2001, Beirlant et al.) addresses tail-index estimation, not boundary tests. The workflow's gate is **temporary** — Phase 4.5 C will close it for the linearity-family pretests with mechanism varying by test: Rao-Wu rescaled bootstrap for `stute_test` and the joint variants (`stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`); weighted OLS residuals + weighted variance estimator for `yatchew_hr_test` (Yatchew 1997 is a closed-form variance-ratio test, not bootstrap-based). Sister pretests (`stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`) keep their closed signatures in this release — Phase 4.5 C will add kwargs and implementation together to avoid API churn. Unweighted `qug_test(d)` and `did_had_pretest_workflow(...)` calls are bit-exact pre-PR (kwargs are keyword-only after `*`; positional path unchanged). New tests at `tests/test_had_pretests.py::TestQUGTest` (5 rejection / mutex / message / regression tests) and the new `TestHADPretestWorkflowSurveyGuards` class (6 tests covering both kwarg paths, mutex, methodology pointer, both aggregate paths, and unweighted regression). See `docs/methodology/REGISTRY.md` § "QUG Null Test" — Note (Phase 4.5 C0) for the full methodology rationale plus a sketch of the (out-of-scope) theoretical bridge that combines endpoint-estimation EVT (Hall 1982, Aarssen-de Haan 1994, Hall-Wang 1999, Beirlant-de Wet-Goegebeur 2006), survey-aware functional CLTs (Boistard-Lopuhaä-Ruiz-Gazen 2017, Bertail-Chautru-Clémençon 2017), and tail-empirical-process theory (Drees 2003) — publishable methodology research, not engineering work.
- **`HeterogeneousAdoptionDiD` mass-point `survey=` / `weights=` + event-study `aggregate="event_study"` survey composition + multiplier-bootstrap sup-t simultaneous confidence band (Phase 4.5 B).** Closes the two Phase 4.5 A `NotImplementedError` gates: `design="mass_point" + weights/survey` and `aggregate="event_study" + weights/survey`. Weighted 2SLS sandwich in `_fit_mass_point_2sls` follows the Wooldridge 2010 Ch. 12 pweight convention (`w²` in the HC1 meat, `w·u` in the CR1 cluster score, weighted bread `Z'WX`); HC1 and CR1 ("stata" `se_type`) bit-parity with `estimatr::iv_robust(..., weights=, clusters=)` at `atol=1e-10` (new cross-language golden at `benchmarks/data/estimatr_iv_robust_golden.json`, generated by `benchmarks/R/generate_estimatr_iv_robust_golden.R`; `estimatr` added to `benchmarks/R/requirements.R`). `_fit_mass_point_2sls` gains `weights=` + `return_influence=` kwargs and now always returns a 3-tuple `(beta, se, psi)` — `psi` is the per-unit IF on the β̂-scale scaled so `compute_survey_if_variance(psi, trivial_resolved) ≈ V_HC1[1,1]` at `atol=1e-10` (PR #359 IF scale convention applied uniformly; no `sum(psi²)` claims). Event-study per-horizon variance: `survey=` path composes Binder-TSL via `compute_survey_if_variance`; `weights=` shortcut uses the analytical weighted-robust SE (continuous: CCT-2014 `bc_fit.se_robust / |den|`; mass-point: weighted 2SLS pweight sandwich from `_fit_mass_point_2sls` — HC1 / classical / CR1). `survey_metadata` / `variance_formula` / `effective_dose_mean` populated in both regimes (previously hardcoded `None` at `had.py:3366`). New multiplier-bootstrap sup-t: `_sup_t_multiplier_bootstrap` reuses `diff_diff.bootstrap_utils.generate_survey_multiplier_weights_batch` for PSU-level draws with stratum centering + sqrt(n_h/(n_h-1)) small-sample correction + FPC scaling + lonely-PSU handling. On the `weights=` shortcut, sup-t calibration is routed through a synthetic trivial `ResolvedSurveyDesign` so the centered + small-sample-corrected branch fires uniformly — targets the analytical HC1 variance family (`compute_survey_if_variance(IF, trivial) ≈ V_HC1` per the PR #359 IF scale invariant) rather than the raw `sum(ψ²) = ((n-1)/n) · V_HC1` that unit-level Rademacher multipliers would produce on the HC1-scaled IF. Perturbations: `delta = weights @ IF` with NO `(1/n)` prefactor (matching `staggered_bootstrap.py:373` idiom), normalized by per-horizon analytical SE, `(1-alpha)`-quantile of the sup-t distribution. At H=1 the quantile reduces to `Φ⁻¹(1 − alpha/2) ≈ 1.96` up to MC noise (regression-locked by `TestSupTReducesToNormalAtH1`). `HeterogeneousAdoptionDiD.__init__` gains `n_bootstrap: int = 999` and `seed: Optional[int] = None` (CS-parity singular seed); `fit()` gains `cband: bool = True` (only consulted on weighted event-study). `HeterogeneousAdoptionDiDEventStudyResults` extended with `variance_formula`, `effective_dose_mean`, `cband_low`, `cband_high`, `cband_crit_value`, `cband_method`, `cband_n_bootstrap` (all `None` on unweighted fits); surfaced in `to_dict`, `to_dataframe`, `summary`, `__repr__`. Unweighted event-study with `cband=False` preserves pre-Phase 4.5 B numerical output bit-exactly (stability invariant, locked by regression tests). Zero-weight subpopulation convention carries over from PR #359 (filter for design decisions; preserve full ResolvedSurveyDesign for variance). Non-pweight SurveyDesigns (`aweight`, `fweight`, replicate designs) raise `NotImplementedError` on both new paths (reciprocal-guard discipline). Pretest surfaces (`qug_test`, `stute_test`, `yatchew_hr_test`, joint variants, `did_had_pretest_workflow`) remain unweighted in this release — Phase 4.5 C / C0. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Weighted 2SLS (Phase 4.5 B)", "Event-study survey composition", and "Sup-t multiplier bootstrap" for derivations and invariants.
- **`PanelProfile.outcome_shape` and `PanelProfile.treatment_dose` extensions + `llms-autonomous.txt` worked examples (Wave 2 of the AI-agent enablement track).** `profile_panel(...)` now populates two new optional sub-dataclasses on the returned `PanelProfile`: `outcome_shape: Optional[OutcomeShape]` (numeric outcomes only — exposes `n_distinct_values`, `pct_zeros`, `value_min` / `value_max`, `skewness` and `excess_kurtosis` (NaN-safe; `None` when `n_distinct_values < 3` or variance is zero), `is_integer_valued`, `is_count_like` (heuristic: integer-valued AND has zeros AND right-skewed AND > 2 distinct values AND non-negative support, i.e. `value_min >= 0`; flags WooldridgeDiD QMLE consideration over linear OLS — the non-negativity clause aligns the routing signal with `WooldridgeDiD(method="poisson")`'s hard rejection of negative outcomes at `wooldridge.py:1105-1109`), `is_bounded_unit` ([0, 1] support)) and `treatment_dose: Optional[TreatmentDoseShape]` (continuous treatments only — exposes `n_distinct_doses`, `has_zero_dose`, `dose_min` / `dose_max` / `dose_mean` over non-zero doses). Both `OutcomeShape` and `TreatmentDoseShape` are mostly descriptive context. **`profile_panel` does not see the separate `first_treat` column** that `ContinuousDiD.fit()` consumes; the estimator's actual fit-time gates key off `first_treat` (defines never-treated controls as `first_treat == 0`, force-zeroes nonzero `dose` on those rows with a `UserWarning`, and rejects negative dose only among treated units `first_treat > 0`; see `continuous_did.py:276-327` and `:348-360`). In the canonical `ContinuousDiD` setup (Callaway, Goodman-Bacon, Sant'Anna 2024), the dose `D_i` is **time-invariant per unit** and `first_treat` is a **separate column** the caller supplies (not derived from the dose column). Under that setup, several facts on the dose column predict `fit()` outcomes: `PanelProfile.has_never_treated` (proxies `P(D=0) > 0` because the canonical convention ties `first_treat == 0` to `D_i == 0`); `PanelProfile.treatment_varies_within_unit == False` (the actual fit-time gate at line 222-228, holds regardless of `first_treat`); `PanelProfile.is_balanced` (the actual fit-time gate at line 329-338); absence of the `duplicate_unit_time_rows` alert (silent last-row-wins overwrite, must deduplicate before fit); and `treatment_dose.dose_min > 0` (predicts the strictly-positive-treated-dose requirement at line 287-294 because treated units carry their constant dose across all periods). When `has_never_treated == False` (no zero-dose controls but all observed doses non-negative), `ContinuousDiD` does not apply (Remark 3.1 lowest-dose-as-control is not implemented); `HeterogeneousAdoptionDiD` IS a routing alternative on this branch (HAD's own contract requires non-negative dose, which is satisfied). When `dose_min <= 0` (negative treated doses), `ContinuousDiD` does not apply AND `HeterogeneousAdoptionDiD` is **not** a fallback — HAD also raises on negative post-period dose (`had.py:1450-1459`); the applicable alternative is linear DiD with the treatment as a signed continuous covariate. Re-encoding the treatment column is an agent-side preprocessing choice that changes the estimand and is not documented in REGISTRY as a supported fallback. The estimator's force-zero coercion on `first_treat == 0` rows with nonzero `dose` is implementation behavior for inconsistent inputs, not a documented method for manufacturing never-treated controls. The agent must validate the supplied `first_treat` column independently — `profile_panel` does not see it. The shape extensions provide distributional context (effect-size range, count-shape detection) that supplements but does not replace those gates. Both fields are `None` when their classification gate is not met (e.g., `treatment_dose is None` for binary treatments). `to_dict()` serializes the nested dataclasses as JSON-compatible nested dicts. New exports: `OutcomeShape`, `TreatmentDoseShape` from top-level `diff_diff`. `llms-autonomous.txt` gains a new §5 "Worked examples" section with three end-to-end PanelProfile -> reasoning -> validation walkthroughs (binary staggered with never-treated controls, continuous dose with zero baseline, count-shaped outcome) plus §2 field-reference subsections for the new shape fields and §4.7 / §4.11 cross-references for outcome-shape considerations. Existing §5-§8 of the autonomous guide are renumbered to §6-§9. Descriptive only — no recommender language inside the worked examples.
- **`HeterogeneousAdoptionDiD.fit(survey=..., weights=...)` on continuous-dose paths (Phase 4.5 survey support).** The `continuous_at_zero` (paper Design 1') and `continuous_near_d_lower` (Design 1 continuous-near-d̲) designs accept survey weights through two interchangeable kwargs: `weights=<array>` (pweight shortcut, weighted-robust SE from the CCT-2014 lprobust port) and `survey=SurveyDesign(weights, strata, psu, fpc)` (design-based inference via Binder-TSL variance using the existing `compute_survey_if_variance` helper at `diff_diff/survey.py:1802`). Point estimates match across both entry paths; SE diverges by design (pweight-only vs PSU-aggregated). `HeterogeneousAdoptionDiDResults.survey_metadata` is a repo-standard `SurveyMetadata` dataclass (weight_type / effective_n / design_effect / sum_weights / weight_range / n_strata / n_psu / df_survey); HAD-specific extras (`variance_formula` label, `effective_dose_mean`) are separate top-level result fields. `to_dict()` surfaces the full `SurveyMetadata` object plus `variance_formula` + `effective_dose_mean`; `summary()` renders `variance_formula`, `effective_n`, `effective_dose_mean`, and (when the survey= path is used) `df_survey`; `__repr__` surfaces `variance_formula` + `effective_dose_mean` when present. The HAD `mass_point` design and `aggregate="event_study"` path raise `NotImplementedError` under survey/weights (deferred to Phase 4.5 B: weighted 2SLS + event-study survey composition); the HAD pretests stay unweighted in this release (Phase 4.5 C). Parity ceiling acknowledged — no public weighted-CCF bias-corrected local-linear reference exists in any language; methodology confidence comes from (1) uniform-weights bit-parity at `atol=1e-14` on the full lprobust output struct, (2) cross-language weighted-OLS parity (manual R reference) at `atol=1e-12`, and (3) Monte Carlo oracle consistency on known-τ DGPs. `_nprobust_port.lprobust` gains `weights=` and `return_influence=` (used internally by the Binder-TSL path); `bias_corrected_local_linear` removes the Phase 1c `NotImplementedError` on `weights=` and forwards. Auto-bandwidth selection remains unweighted in this release — pass `h`/`b` explicitly for weight-aware bandwidths. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Weighted extension (Phase 4.5 survey support)".
- **`stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test` + `StuteJointResult`** (HeterogeneousAdoptionDiD Phase 3 follow-up). Joint Cramér-von Mises pretests across K horizons with shared-η Mammen wild bootstrap (preserves vector-valued empirical-process unit-level dependence per Delgado-Manteiga 2001 / Hlávka-Hušková 2020). The core `stute_joint_pretest` is residuals-in; two thin data-in wrappers construct per-horizon residuals for the two nulls the paper spells out: mean-independence (step 2 pre-trends, `OLS(Y_t − Y_base ~ 1)` per pre-period) and linearity (step 3 joint, `OLS(Y_t − Y_base ~ 1 + D)` per post-period). Sum-of-CvMs aggregation (`S_joint = Σ_k S_k`); per-horizon scale-invariant exact-linear short-circuit. Closes the paper Section 4.2 step-2 gap that Phase 3 `did_had_pretest_workflow` previously flagged with an "Assumption 7 pre-trends test NOT run" caveat. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Joint Stute tests" for algorithm, invariants, and scope exclusion of Eq 18 linear-trend detrending (deferred to Phase 4 Pierce-Schott replication).
- **`did_had_pretest_workflow(aggregate="event_study")`**: multi-period dispatch on balanced ≥3-period panels. Runs QUG at `F` + joint pre-trends Stute across earlier pre-periods + joint homogeneity-linearity Stute across post-periods. Step 2 closure requires ≥2 pre-periods; with only a single pre-period (the base `F-1`) `pretrends_joint=None` and the verdict flags the skip. Reuses the Phase 2b event-study panel validator (last-cohort auto-filter under staggered timing with `UserWarning`; `ValueError` when `first_treat_col=None` and the panel is staggered). The data-in wrappers `joint_pretrends_test` and `joint_homogeneity_test` also route through that same validator internally, so direct wrapper calls inherit the last-cohort filter and constant-post-dose invariant. `HADPretestReport` extended with `pretrends_joint`, `homogeneity_joint`, and `aggregate` fields; serialization methods (`summary`, `to_dict`, `to_dataframe`, `__repr__`) preserve the Phase 3 output bit-exactly on `aggregate="overall"` — no `aggregate` key, no header row, no schema drift — and only surface the new fields on `aggregate="event_study"`.
- **`ChaisemartinDHaultfoeuille.by_path`** — per-path event-study disaggregation, mirroring R `did_multiplegt_dyn(..., by_path=k)`. Passing `by_path=k` (positive int) to the estimator reports separate `DID_{path,l}` + SE + inference for the top-k most common observed treatment paths in the window `[F_g-1, F_g-1+L_max]`, answering the practitioner question "is a single pulse enough, or do you need sustained exposure?" across paths like `(0,1,0,0)` vs `(0,1,1,0)` vs `(0,1,1,1)`. The per-path SE follows the joiners-only / leavers-only IF precedent (switcher-side contribution zeroed for non-path groups; control pool and cohort structure unchanged; plug-in SE with path-specific divisor). Requires `drop_larger_lower=False` (multi-switch groups are the object of interest) and `L_max >= 1`. Binary treatment was the only supported case at the initial cut; subsequent entries in this `[Unreleased]` block lifted that and the original gates one by one. Currently still gated: `design2` and `honest_did` raise `NotImplementedError` (deferred to follow-up PRs). All other combinations — `n_bootstrap > 0`, `placebo=True`, joint sup-t bands, `controls`, `trends_linear`, `trends_nonparam`, `survey_design`, `heterogeneity`, non-binary integer treatment, and the `paths_of_interest` user-specified selector — are now supported, with the per-feature contracts in their dedicated entries elsewhere in `[Unreleased]`. Results expose `results.path_effects: Dict[Tuple[int, ...], Dict[str, Any]]` and `results.to_dataframe(level="by_path")`; the summary grows a "Treatment-Path Disaggregation" block. Ties in path frequency are broken lexicographically on the path tuple for deterministic ranking. Overflow (`by_path > n_observed_paths`) returns all observed paths with a `UserWarning`. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path per-path event-study disaggregation)` for the full contract.
- **`ChaisemartinDHaultfoeuille.by_path` + `n_bootstrap > 0`** — bootstrap SE for per-path event-study effects. The top-k paths are enumerated once on the observed data (R-faithful path-stability semantics: matches `did_multiplegt_dyn(..., by_path=k, bootstrap=B)`, confirmed empirically against `DIDmultiplegtDYN 2.3.3`), and the existing multiplier bootstrap (`bootstrap_weights ∈ {"rademacher", "mammen", "webb"}`) runs per `(path, horizon)` target via the shared `_bootstrap_one_target` / `compute_effect_bootstrap_stats` helpers. Point estimates are unchanged from the analytical path. Bootstrap SE replaces the analytical SE in `path_effects[path]["horizons"][l]["se"]`, and `p_value` / `conf_int` propagate the **bootstrap percentile** statistics (library Round-10 convention, same as `overall` / `joiners` / `leavers` / `multi_horizon`); `t_stat` is SE-derived via `safe_inference` per the anti-pattern rule. Interpretation is *conditional on the observed path set* — practitioners wanting unconditional inference capturing path-selection uncertainty need a pairs-bootstrap (no R precedent). **SE inherits the analytical cross-path cohort-sharing deviation:** bootstrap input is the same full-panel cohort-centered path IF as the analytical path, so the bootstrap SE is a Monte Carlo analog of the analytical SE and inherits the existing analytical-path divergence from R on mixed-path cohorts (see REGISTRY.md for the full mechanism). On single-path-cohort panels, bootstrap and analytical SE both track R up to the Phase 2 envelope. **Deviation from R (CI method):** R's per-path bootstrap CI is normal-theory around the bootstrap SE (half-width ≈ `1.96·se`); ours is the bootstrap percentile CI, intentionally diverging from R to keep the dCDH inference surface internally consistent across all bootstrap targets. Positive regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathBootstrap` (`@pytest.mark.slow`): point-estimate invariance, finite SE on non-degenerate panels, bootstrap-vs-analytical SE within 30% rtol on cohort-clean panels, degenerate-cohort NaN propagation, Rademacher/Mammen/Webb parity, seed reproducibility, and percentile-vs-normal-theory CI pinning. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → **Bootstrap SE** for the full write-up.
- **R-parity for `ChaisemartinDHaultfoeuille.by_path`** against `DIDmultiplegtDYN 2.3.3`. Two new scenarios in `benchmarks/data/dcdh_dynr_golden_values.json` generated from `did_multiplegt_dyn(..., by_path=k)`: `mixed_single_switch_by_path` (2 paths, `by_path=2`) and `multi_path_reversible_by_path` (4 observed paths, `by_path=3`, via a new deterministic multi-path DGP pattern in the R generator). Per-path point estimates and per-path switcher counts match R exactly; per-path SE matches within the Phase 2 multi-horizon SE envelope (observed rtol ≤ 10.2% on the 2-path scenario, ≤ 4.2% on the 4-path scenario). Parity tests live at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPath`, matching paths by tuple label via set-equality (robust to R's undocumented frequency-tie tiebreak) and cross-checking per-path switcher counts before SE comparison. **Deviation documented:** cross-path cohort sharing — our full-panel cohort-centered plug-in vs R's per-path re-run diverges materially when a `(D_{g,1}, F_g, S_g)` cohort spans multiple observed paths; the two coincide when every cohort is single-path. The parity scenarios are constructed to keep cohorts single-path (scenario 13 by design, scenario 14 via path-assignment-deterministic-on-F_g). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path...)` for the full write-up.
- **`profile_panel()` utility + `llms-autonomous.txt` reference guide (agent-facing)** — new `diff_diff.profile_panel(df, *, unit, time, treatment, outcome)` returns a frozen `PanelProfile` dataclass of structural facts (panel balance, treatment-type classification — `"binary_absorbing"` / `"binary_non_absorbing"` / `"continuous"` / `"categorical"`, cohort structure, outcome characteristics, and a `tuple[Alert, ...]` of factual observations). `.to_dict()` returns a JSON-serializable view. Paired with a new bundled `"autonomous"` variant on `get_llm_guide()` — `get_llm_guide("autonomous")` returns a reference-shaped guide (distinct from the existing workflow-prose `"practitioner"` variant) with §1 audience disclaimer, §2 `PanelProfile` field reference, §3 embedded 17-estimator × 9-design-feature support matrix, §4 per-design-feature reasoning citing Baker et al. (2025) and Roth / Sant'Anna (2023), §5 post-fit validation index, §6 BR/DR schema reference, §7 citations, §8 intentional omissions. Both pieces are bundled inside the wheel (no GitHub / RTD dependency at runtime); `diff_diff/__init__.py` module docstring leads with an agent-entry block listing `profile_panel`, `get_llm_guide("autonomous")`, `get_llm_guide("practitioner")`, and `BusinessReport` so `help(diff_diff)` surfaces them. Descriptive, not opinionated — `profile_panel` alerts never recommend a specific estimator, and the guide enumerates trade-offs rather than dispatching. Exports: `profile_panel`, `PanelProfile`, `Alert` from top-level `diff_diff`.
- **`target_parameter` block in BR/DR schemas (experimental; schema version bumped to 2.0)** — `BUSINESS_REPORT_SCHEMA_VERSION` and `DIAGNOSTIC_REPORT_SCHEMA_VERSION` bumped from `"1.0"` to `"2.0"` because the new `"no_scalar_by_design"` value on the `headline.status` / `headline_metric.status` enum (dCDH `trends_linear=True, L_max>=2` configuration) is a breaking change per the REPORTING.md stability policy. BusinessReport and DiagnosticReport now emit a top-level `target_parameter` block naming what the headline scalar actually represents for each of the 16 result classes. Closes BR/DR foundation gap #6 (target-parameter clarity). Fields: `name`, `definition`, `aggregation` (machine-readable dispatch tag), `headline_attribute` (raw result attribute), `reference` (citation pointer). BR's summary emits the short `name` right after the headline; DR's overall-interpretation paragraph does the same; both full reports carry a "## Target Parameter" section with the full definition. Per-estimator dispatch is sourced from REGISTRY.md and lives in the new `diff_diff/_reporting_helpers.py::describe_target_parameter`. A few branches read fit-time config (`EfficientDiDResults.pt_assumption`, `StackedDiDResults.clean_control`, `ChaisemartinDHaultfoeuilleResults.L_max` / `covariate_residuals` / `linear_trends_effects`); others emit a fixed tag (the fit-time `aggregate` kwarg on CS / Imputation / TwoStage / Wooldridge does not change the `overall_att` scalar — disambiguating horizon / group tables is tracked under gap #9). See `docs/methodology/REPORTING.md` "Target parameter" section.
- SyntheticDiD coverage Monte Carlo calibration table added to `docs/methodology/REGISTRY.md` §SyntheticDiD — rejection rates at α ∈ {0.01, 0.05, 0.10} across `placebo` / `bootstrap` / `jackknife` on 3 representative DGPs (balanced / exchangeable, unbalanced, and Arkhangelsky et al. (2021) AER §6.3 non-exchangeable). Artifact at `benchmarks/data/sdid_coverage.json` (500 seeds × B=200), regenerable via `benchmarks/python/coverage_sdid.py`.

### Fixed
- **SyntheticDiD `variance_method="bootstrap"` now runs the paper-faithful refit bootstrap** with R-default warm-start. Re-estimates ω̂_b and λ̂_b via two-pass sparsified Frank-Wolfe on each pairs-bootstrap draw using the fit-time normalized-scale zeta — Arkhangelsky et al. (2021) Algorithm 2 step 2, matching the behavior of R's default `synthdid::vcov(method="bootstrap")` (which rebinds `attr(estimate, "opts")` so the renormalized ω serves as Frank-Wolfe initialization). The Python path threads that warm-start through `compute_sdid_unit_weights(..., init_weights=_sum_normalize(ω̂[boot_control_idx]))` and `compute_time_weights(..., init_weights=λ̂)` on each bootstrap draw. `compute_sdid_unit_weights` and `compute_time_weights` gain a new `init_weights` kwarg; when provided, the Rust top-level fast-path is skipped in favor of the Python two-pass dispatcher (whose inner FW calls still dispatch to Rust). Without this kwarg both helpers remain backward-compatible and keep the Rust fast-path. The previous fixed-weight bootstrap path is removed entirely — it was not paper-faithful and, despite prior documentation claiming otherwise, also did not match R's default bootstrap (the previous R-parity test fixture invoked `synthdid_estimate(weights=...)` without rebinding `opts`, which silently runs fixed-weight, so the 1e-10 parity was between two paths both wrong in the same direction). Coverage MC at the new artifact above quantifies the correctness fix on 3 representative null DGPs. **Users' existing `variance_method="bootstrap"` fits will return materially different SE / p-value / CI values on the next release** — same enum name, corrected semantics. Bootstrap is now ~5–30× slower per fit than the old fixed-weight shortcut (panel-size dependent; warm-start converges faster than cold-start so the slowdown is less than the 10–100× prior estimate). The PR #349 follow-on bullets below (analytical p-value dispatch, sqrt((r-1)/r) SE formula, retry-to-B contract) all carry over to the refit path unchanged.
- SyntheticDiD `variance_method="bootstrap"` now computes p-values from the analytical normal-theory formula using the bootstrap SE (matching R's `synthdid::vcov()` convention), rather than an empirical null-distribution formula that is not valid for bootstrap draws. `is_significant` and `significance_stars` are derived from `p_value` and will also change for bootstrap fits. Placebo and jackknife are unchanged. Point estimates are unaffected.
- SyntheticDiD bootstrap SE formula applies the `sqrt((r-1)/r)` correction matching R's synthdid and the placebo SE formula.
- SyntheticDiD bootstrap now retries degenerate resamples (all-control or all-treated, or non-finite `τ_b`) until exactly `n_bootstrap` valid replicates are accumulated, matching R's `synthdid::bootstrap_sample` and Arkhangelsky et al. (2021) Algorithm 2. Previously the Python path counted attempts (with degenerate draws silently dropped), producing fewer valid replicates than requested. A bounded-attempt guard (`20 × n_bootstrap`) prevents pathological-input hangs.
- **TROP global bootstrap SE backend parity under fixed seed** — Rust and Python backends now produce bit-identical bootstrap SE under the same `seed`. Previously Rust's `bootstrap_trop_variance_global` seeded `rand_xoshiro::Xoshiro256PlusPlus` per replicate while Python's fallback consumed `numpy.random.default_rng` (PCG64), producing ~28% SE divergence on tiny panels under `seed=42`. Fixed by extracting a shared `stratified_bootstrap_indices` helper in `diff_diff/bootstrap_utils.py` that pre-generates per-replicate stratified sample indices via numpy on the Python side; both backends consume the same integer arrays through the PyO3 surface. Sampling law (stratified: controls then treated, with replacement) is unchanged. Closes the bootstrap-RNG half of silent-failures audit finding #23 (grid-search half closed in PR #348; local-method methodology half closed by the two Fixed entries below). Local-method TROP also adopts the Python-canonical index contract for the RNG layer here.
- **TROP local-method Rust weight-matrix no longer normalized** — `rust/src/trop.rs::compute_weight_matrix` no longer divides time-weights or unit-weights by their respective sums before the outer product. The paper's Equation 2/3 (Athey, Imbens, Qu, Viviano 2025) and REGISTRY.md Requirements checklist (line 2037: `[x] Unit weights: exp(-λ_unit × distance) (unnormalized, matching Eq. 2)`) both specify raw-exponential weights; Python's `_compute_observation_weights` was already REGISTRY-compliant. **User-visible effect**: Rust local-method ATT values may shift for any fit with `lambda_nn < infinity` — normalizing the weight-matrix inflated the effective nuclear-norm penalty relative to the data-fit term, changing the regularization trade-off. For `lambda_nn = infinity` (factor model disabled) outputs are unchanged because uniform weight scaling leaves the minimum-norm WLS argmin invariant. Rust LOOCV-selected lambdas may also shift on this boundary; both backends now converge on the same REGISTRY-compliant selection.
- **TROP local-method Python `_compute_observation_weights` now uses the function-argument `Y, D` and treats all non-target units as donors** — two coupled changes that bring Python structurally in line with Rust and the paper's Eq. 2/3:
    1. Removed the `if self._precomputed is not None:` branch that silently substituted `self._precomputed["Y"]` / `["D"]` / `["time_dist_matrix"]` (original-panel cache populated during main fit) for the function-argument `Y, D`. Under bootstrap, `_fit_with_fixed_lambda` computes fresh `Y, D` from the resampled `boot_data` and passes them in; the helper was discarding those and recomputing unit distances from the original panel, so Python's local bootstrap resampled units but reused stale unit-distance weights. Rust's bootstrap was already correct (always consumed `y_boot, d_boot`).
    2. Removed the `valid_control_at_t = D[t, :] == 0` target-period donor gate that zeroed `ω_j` for any unit `j` treated at the target period (other than the target unit itself). Per REGISTRY Eq. 2/3 and Rust's `compute_weight_matrix`, `ω_j = exp(-λ_unit × dist(j, i))` for all `j ≠ i`; treated-cell exclusion happens via the `(1 − W_{js})` factor applied inside `_estimate_model`. Same-cohort donors now contribute via their pre-treatment rows. Empirically the main-fit ATT is unchanged on tested fixtures because same-cohort pre-treatment observations are exactly absorbed by their own unit fixed effect `alpha_j` without propagating into `mu`, `beta`, or other units' parameters — so this change is structural alignment rather than a numerical shift in output. Users on same-cohort panels with very few controls may still see tiny differences in edge cases; the new `test_local_method_same_cohort_donor_parity` regression guards the aligned behavior.
  Together with the normalization fix above, TROP local-method backend parity on the main-fit ATT is regime-dependent: `atol=rtol=1e-14` for `lambda_nn=inf` (no nuclear-norm regularization, uniform weight scaling leaves the WLS argmin invariant) and `atol=1e-10` for finite `lambda_nn` (FISTA inner loop + BLAS reduction ordering introduce sub-1e-10 roundoff across Rust `faer` vs numpy paths). Bootstrap SE parity is asserted at `atol=1e-5` to accommodate ~1e-7 roundoff between Rust's `estimate_model` matrix factorization and numpy's `lstsq` that accumulates across per-replicate fits; sub-1e-14 bootstrap parity is tracked as a follow-up in `TODO.md` under "unify Rust local-method solver path". Closes silent-failures audit finding #23 (local-method half; the RNG half closed in PR #354 and the grid-search half in PR #348).

### Changed
- **`did_had_pretest_workflow(aggregate="event_study")` verdict no longer emits the "paper step 2 deferred to Phase 3 follow-up" caveat** — the joint pre-trends Stute test closes that gap. The two-period `aggregate="overall"` path retains the existing caveat since the joint variant does not apply to single-pre-period panels. Downstream code that greps verdict strings for the Phase 3 caveat will see it suppressed on the event-study path.
- **SyntheticDiD bootstrap no longer supports survey designs** (capability regression in PR #351, **restored in PR #355** — see Added/Changed entries directly below). The removed fixed-weight bootstrap path was the only SDID variance method that supported strata/PSU/FPC (via Rao-Wu rescaled bootstrap); the PR #351 paper-faithful refit bootstrap initially rejected all survey designs (including pweight-only) with `NotImplementedError`. PR #355 restores the capability via a weighted-FW + Rao-Wu composition; the lock-out window applies only to the v3.2.x line that ships PR #351 alone (without PR #355). Composing Rao-Wu rescaled weights with Frank-Wolfe re-estimation: see `docs/methodology/REGISTRY.md` §SyntheticDiD `Note (survey + bootstrap composition)`.

### Added (PR #355)
- **SDID `variance_method="bootstrap"` survey support restored** via a hybrid pairs-bootstrap + Rao-Wu rescaling composed with a weighted Frank-Wolfe kernel. Each bootstrap draw first performs the unit-level pairs-bootstrap resampling specified by Arkhangelsky et al. (2021) Algorithm 2 (`boot_idx = rng.choice(n_total)`), and *then* applies Rao-Wu rescaled per-unit weights (Rao & Wu 1988) sliced over the resampled units — NOT a standalone Rao-Wu bootstrap. New Rust kernel `sc_weight_fw_weighted` (and `_with_convergence` sibling) accepts a per-coordinate `reg_weights` argument so the FW objective becomes `min ||A·ω - b||² + ζ²·Σ_j reg_w[j]·ω[j]²`. New Python helpers `compute_sdid_unit_weights_survey` and `compute_time_weights_survey` thread per-control survey weights through the two-pass sparsify-refit dispatcher (column-scaling Y by `rw` for the loss, `reg_weights=rw` for the penalty on the unit-weights side; weighted column-centering + row-scaling Y by `sqrt(rw)` for the loss with uniform reg on the time-weights side). `_bootstrap_se` survey branch composes the per-draw `rw` (Rao-Wu rescaling for full designs, constant `w_control` for pweight-only fits) with the weighted-FW helpers, then composes `ω_eff = rw·ω/Σ(rw·ω)` for the SDID estimator. Coverage MC artifact extended with a `stratified_survey` DGP (BRFSS-style: N=40, strata=2, PSU=2/stratum); the bootstrap row's near-nominal calibration is the validation gate (target rejection ∈ [0.02, 0.10] at α=0.05). New regression tests across `test_methodology_sdid.py::TestBootstrapSE` (single-PSU short-circuit, full-design and pweight-only succeeds-tests, zero-treated-mass retry, deterministic Rao-Wu × boot_idx slice) and `test_survey_phase5.py::TestSyntheticDiDSurvey` (full-design ↔ pweight-only SE differs assertion). See REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap composition)`` for the full objective and the argmin-set caveat.

### Changed (PR #355)
- **SDID bootstrap SE values under survey fits now differ numerically from the v3.2.x line that shipped PR #351 alone**: the fit no longer raises `NotImplementedError`, and instead returns the weighted-FW + Rao-Wu SE. Non-survey fits are unaffected (the bootstrap dispatcher routes only the survey branch through the new `_survey` helpers; non-survey fits continue to call the existing `compute_sdid_unit_weights` / `compute_time_weights` and stay bit-identical at rel=1e-14 on the `_BASELINE["bootstrap"]` regression). SDID's `placebo` and `jackknife` paths still reject `strata/PSU/FPC` on the v3.2.x line; full-design support for those methods lands separately in the entries below.

### Added
- **SDID `variance_method="placebo"` and `"jackknife"` now support strata/PSU/FPC designs.** Closes the last SDID survey gap. All three variance methods (bootstrap from PR #355, plus placebo and jackknife here) now handle full survey designs. New private methods `SyntheticDiD._placebo_variance_se_survey` and `_jackknife_se_survey` route the full-design path through method-specific allocators:
  - **Placebo** — stratified permutation (Pesarin 2001). Each draw samples pseudo-treated indices uniformly without replacement from controls *within each stratum* containing actual treated units; non-treated strata contribute their controls unconditionally. The weighted Frank-Wolfe kernel from PR #355 (`compute_sdid_unit_weights_survey` / `compute_time_weights_survey`) re-estimates ω and λ per draw with per-control survey weights threaded into both loss and regularization; post-optimization composition `ω_eff = rw·ω/Σ(rw·ω)`. Arkhangelsky Algorithm 4 SE formula unchanged.
  - **Jackknife** — PSU-level leave-one-out with stratum aggregation (Rust & Rao 1996). `SE² = Σ_h (1-f_h)·(n_h-1)/n_h·Σ_{j∈h}(τ̂_{(h,j)} - τ̄_h)²` with `f_h = n_h_sampled / fpc[h]` (population-count FPC form). λ held fixed across LOOs; ω subsetted, composed with rw, renormalized. Strata with `n_h < 2` silently skipped (matches R `survey::svyjkn` with `lonely_psu="remove"` / `"certainty"`; `"adjust"` raises `NotImplementedError`). Full-census strata (`f_h ≥ 1`) short-circuit to zero contribution before any LOO feasibility check. `SE = 0` is returned for legitimate zero variance (e.g., every stratum full-census); `SE = NaN` with a targeted `UserWarning` is reserved for undefined cases — all strata skipped, or any delete-one replicate in a non-full-census contributing stratum is undefined (all-treated-in-one-PSU LOO, kept ω_eff / w_treated mass zero, estimator raises). Unstratified single-PSU short-circuits to NaN.
  - **Fit-time feasibility guards** (placebo): `ValueError` on stratum-level infeasibility with targeted messages distinguishing three cases — **Case B** (treated-containing stratum has zero controls), **Case C** (fewer controls than treated in a treated stratum), **Case D** (every treated stratum is exact-count `n_c_h == n_t_h` → permutation support is 1, null distribution collapses). Partial-permutation fallback rejected because it would silently change the null-distribution semantics.
  - **Gate relaxed**: the fit-time guard at `synthetic_did.py:352-369` that rejected placebo/jackknife + strata/PSU/FPC is removed. Replicate-weight designs remain rejected (separate methodology — replicate variance is closed-form and would double-count with Rao-Wu-like rescaling). Non-survey and pweight-only paths bit-identical by construction — the new code is gated on `resolved_survey_unit.(strata|psu|fpc) is not None`.
  - **Coverage MC**: `benchmarks/data/sdid_coverage.json` extended with jackknife on `stratified_survey`. Bootstrap validates near-nominal (α=0.05 rejection = 0.058, SE/trueSD = 1.13). Jackknife reported with an anti-conservatism caveat: with only 2 PSUs per stratum the stratified jackknife formula has 1 effective DoF per stratum, a well-documented limitation of Rust & Rao (1996) — `se_over_truesd ≈ 0.46` on this DGP. Users needing tight SE calibration with few PSUs should prefer `variance_method="bootstrap"`. Placebo is structurally infeasible on the existing `stratified_survey` DGP (its cohort packs into one stratum with 0 never-treated units — by design a bootstrap-suited DGP); the placebo survey path is exercised via unit tests on a feasible fixture.
  - **Regression tests** across `tests/test_survey_phase5.py`: two new classes `TestSDIDSurveyPlaceboFullDesign` and `TestSDIDSurveyJackknifeFullDesign`. Placebo: pseudo-treated-stratum contract, Case B / Case C front-door guards with targeted-message regression, SE-differs-from-pweight-only, deterministic dispatch. Jackknife: stratum-aggregation self-consistency, **FPC magnitude regression** (2-stratum handcrafted panel asserts `SE_fpc == SE_nofpc · sqrt(1-f)` at `rtol=1e-10`), single-PSU-stratum skip, unstratified short-circuit, all-strata-skipped warning + NaN, SE-differs-from-pweight-only, deterministic dispatch. Existing `test_full_design_placebo_raises` and `test_full_design_jackknife_raises` flipped to `_succeeds` assertions. All 19 existing pweight-only and non-survey placebo/jackknife tests pass unchanged (bit-identity preserved via the new-path gating).
  - **Allocator asymmetry** (documented in REGISTRY): placebo ignores the PSU axis (unit-level within-stratum permutation — the classical stratified permutation test; PSU-level permutation on few PSUs is near-degenerate); jackknife respects PSU (PSU-level LOO is the canonical survey jackknife). Both respect strata. See `docs/methodology/REGISTRY.md` §SyntheticDiD `Note (survey + placebo composition)` and `Note (survey + jackknife composition)`.

## [3.2.0] - 2026-04-19

### Added
- **`BusinessReport` and `DiagnosticReport` (experimental preview)** (PR #318) - practitioner-ready output layer. `BusinessReport(results, ...)` produces plain-English narrative summaries (`.summary()`, `.full_report()`, `.export_markdown()`, `.to_dict()`) from any of the 16 fitted result types. `DiagnosticReport(results, ...)` orchestrates the existing diagnostic battery (parallel trends, pre-trends power, HonestDiD sensitivity, Goodman-Bacon, heterogeneity, design-effect, EPV) plus estimator-native diagnostics for SyntheticDiD (`pre_treatment_fit`, weight concentration, in-time placebo, zeta sensitivity) and TROP (factor-model fit metrics). Both classes expose an AI-legible `to_dict()` schema (single source of truth; prose renders from the dict). BR auto-constructs DR by default so summaries mention pre-trends, robustness, and design-effect findings in one call. See `docs/methodology/REPORTING.md` for methodology deviations including the no-traffic-light-gates decision, pre-trends verdict thresholds (0.05 / 0.30), and power-aware phrasing driven by `compute_pretrends_power`. **Both schemas are marked experimental in this release** - wording, verdict thresholds, and schema shape will change; do not anchor downstream tooling on them yet.
- **Kernel / local-linear / nonparametric infrastructure** (PRs #327, #335) - bandwidth selector, local linear regression, HC2 / Bell-McCaffrey variance helpers, and a port of R `nprobust`'s point-estimate path. Foundation for the upcoming `HeterogeneousAdoptionDiD` estimator (de Chaisemartin, Ciccia, D'Haultfœuille & Knau 2024 — "DiD with no untreated group"). Released as internal modules with full test coverage (`tests/test_bandwidth_selector.py`, `tests/test_local_linear.py`, `tests/test_linalg_hc2_bm.py`, `tests/test_nprobust_port.py`); the user-facing estimator ships in a later phase.
- **Cell-period IF allocator for dCDH survey variance (Class A contract)** (PR #323) - replaces the group-level allocator `ψ_i = ψ_g * (w_i / W_g)` with a cell-period allocator `ψ_i = ψ_g * (w_i / W_{g, out_idx})` on the post-period cell for the DID_l replicate-weight ATT path. Is the allocator shape that the v3.2.0 heterogeneity and bootstrap extensions below build on. Documents the post-period attribution convention in REGISTRY.md with a hand-computed row-sum identity test.

### Performance
- **`aggregate_survey` stratum-PSU scaffolding precompute** — the per-cell Taylor-series variance inside `aggregate_survey` no longer rebuilds stratum-PSU scaffolding on every cell. A frozen `_PsuScaffolding` (strata codes, global PSU codes unique across strata, per-stratum counts and FPC ratios, singleton mask, static legitimate-zero counts and variance-computable flag) is precomputed once per design at the top of `aggregate_survey` and threaded through `_cell_mean_variance` to a new `_compute_if_variance_fast` path that replaces the per-stratum pandas groupby with two vectorized `np.bincount` passes. BRFSS-shaped 50-state × 10-year × 1M-row microdata → state-year panel drops from ~24s to sub-2s under both backends (the path is pure Python, so Python and Rust track each other). Numerical output is preserved to sub-ULP tolerance; seven-case equivalence tests (`TestAggregateSurveyScaffolding`) assert `assert_allclose(atol=1e-14, rtol=1e-14)` between fast and legacy paths across stratified+PSU+FPC, stratified no FPC, PSU-only, weights-only, and all three `lonely_psu` modes (remove / certainty / adjust). Replicate-weight designs continue to route through `compute_replicate_if_variance` unchanged. `_compute_stratified_psu_meat` is untouched — all other TSL callers (DiD / TWFE / CS / etc.) are unaffected.

### Changed
- Add Zenodo DOI badge to README; upgrade the BibTeX citation block with the concept DOI (`10.5281/zenodo.19646175`) and list author as Isaac Gerber (matching `CITATION.cff`). `CITATION.cff` carries the concept DOI as its top-level `doi:` field — Zenodo auto-mints a versioned DOI for every release, but the CFF file tracks the concept DOI only so it doesn't need a follow-up edit per release. DOI was minted by Zenodo when v3.1.3 was released.
- **`ChaisemartinDHaultfoeuille` heterogeneity + within-group-varying PSU/strata now supported under Binder TSL** - `fit(heterogeneity=..., survey_design=...)` no longer raises `NotImplementedError` when the resolved design's PSU or strata vary across the cells of a group. On the **Binder TSL** branch (`compute_survey_if_variance`), the heterogeneity WLS coefficient IF is expanded to observation level via the cell-period allocator `ψ_i = ψ_g * (w_i / W_{g, out_idx})` on the post-period cell — the DID_l post-period single-cell convention shipped in v3.1.x. Under PSU=group the PSU-level Binder TSL variance is byte-identical to the previous release (PSU-level aggregate telescopes to `ψ_g`); under within-group-varying PSU, mass lands in the post-period PSU of the transition. The **Rao-Wu replicate-weight** branch (`compute_replicate_if_variance`) retains the legacy group-level allocator `ψ_i = ψ_g * (w_i / W_g)`: replicate variance computes `θ_r = sum_i ratio_ir * ψ_i` at observation level and is therefore not PSU-telescoping, so the cell-period allocator would silently change the replicate SE whenever a replicate column's ratios vary within group (e.g., per-row replicate matrices). Replicate + heterogeneity fits therefore produce byte-identical SE to the previous release, and the newly-unblocked `heterogeneity=` + within-group-varying PSU combination is unreachable under replicate designs by construction (`SurveyDesign` rejects `replicate_weights` combined with explicit `strata/psu/fpc`).
- **`ChaisemartinDHaultfoeuille.fit(survey_design=..., n_bootstrap > 0)` now supports within-group-varying PSU** — the PSU-level Hall-Mammen wild multiplier bootstrap has been extended from a group-level PSU map (one multiplier per group) to a cell-level PSU map (one multiplier per `(g, t)` cell's PSU). A dispatcher in `_compute_dcdh_bootstrap` detects PSU-within-group-constant regimes (including PSU=group auto-inject and strictly-coarser PSU with within-group constancy) and routes them through the legacy group-level path so the bootstrap SE is bit-identical to the previous release (guarded by the new `test_bootstrap_se_matches_pre_pr4_baseline` and the pre-existing `test_auto_inject_bit_identical_to_group_level`). Under within-group-varying PSU, a group contributing cells to multiple PSUs receives independent multiplier draws per PSU — the correct Hall-Mammen wild PSU clustering at cell granularity. Multi-horizon bootstraps draw a single shared `(n_bootstrap, n_psu)` PSU-level weight matrix per block and broadcast per-horizon via each horizon's cell-to-PSU map, so the sup-t simultaneous confidence band remains a valid joint distribution. Closes the last `NotImplementedError` gate in the dCDH survey contract; replicate-weight variance and `n_bootstrap > 0` remain mutually exclusive by construction. **Scope note:** panels with *terminal missingness* where the terminally-missing group is in a cohort whose other groups still contribute at the missing period now raise a targeted `ValueError` on every survey variance path that uses the cell-period allocator: Binder TSL with within-group-varying PSU, Rao-Wu replicate-weight ATT (which always uses the cell allocator per the Class A contract shipped in PR #323), and the cell-level wild PSU bootstrap. Cohort-recentering leaks centered IF mass onto cells with no positive-weight observations, which the cell-period allocator cannot attach to any observation/PSU. This closes a silent mass-drop bug the cell-period allocator introduced across all three paths in v3.1.x; pre-process the panel to remove terminal missingness (drop late-exit groups or trim to a balanced sub-panel) as the documented workaround. For Binder TSL only, using an explicit `psu=<group_col>` routes through the legacy group-level allocator where the row-sum identity makes the two allocators statistically equivalent. Replicate-weight ATT and within-group-varying-PSU bootstrap have no such allocator fallback — the panel itself must be pre-processed. PSU-within-group-constant Binder TSL (including PSU=group auto-inject) is unaffected.
- **Performance review: practitioner-scale scenarios + benchmark harness extension** (PR #333) - new `docs/performance-scenarios.md` documents 5-7 realistic practitioner workflows (marketing lift, geo-experiment, BRFSS state-policy, dCDH reversible treatment) grounded in the practitioner docs and the paper literature, not cookie-cutter textbook data. `benchmarks/speed_review/` extended with practitioner-scale scripts and per-backend bit-identity baselines. Baselines refreshed against current main. Finding: the biggest leverage areas are bootstrap resampling loops and per-replicate survey-design rebuilds in the bootstrap path; documented in `docs/performance-plan.md` for follow-up optimization PRs.
- **Wall-clock timing tests excluded from default CI** (PRs #330, #336) - `TestCallawaySantAnnaSEAccuracy.test_timing_performance` and `TestPerformanceRegression` marked `@pytest.mark.slow`, removing false-positive CI failures from runner-noise variance (BLAS path variation, neighbor VM contention). Tests remain runnable via `pytest -m slow` for ad-hoc local benchmarking; the perf-review harness above is the principled replacement for CI-gated performance tracking.

### Fixed
- **Silent-failures audit: axis A** (PR #334) — minor solver paths numerical-precision / scale-fragility closeouts, completing the SDID extreme-Y-scale work started in v3.1.2.
- **Silent-failures audit: axis C & J** (PR #339) — B-spline derivative warning scope broadened; `SurveyPowerConfig` stale-cache wording narrowed.
- **Silent-failures audit: axis E** (PR #331) — row-drop counters surfaced across estimator paths so silent validator row-drops leave an explicit count on the result.
- **Silent-failures audit: axis G** (PR #337) — Rust vs Python backend edge-case parity tests added for rank-deficient, extreme-scale, and constant-column inputs.
- **SyntheticDiD diagnostic Y-normalization parity** (PR #328) — extends the PR #312 catastrophic-cancellation fix from the main fit path into `SyntheticDiDResults.in_time_placebo()` and `.sensitivity_to_zeta_omega()`. Diagnostics now apply the same `Y_shift / Y_scale` normalization the main fit uses, pass `zeta / Y_scale` and a normalized `min_decrease` into Frank-Wolfe, then rescale `att` / `pre_fit_rmse` back to original-Y units.
- **TROP bootstrap failure-rate guards** (PR #324) — alternating-minimization bootstrap loops now emit a `UserWarning` on silent high-failure-rate runs (LOOCV and bootstrap aggregation paths both covered); attempt-count-based warning replaces the previous observation-count denominator that could silently mask sparse runs.
- **`simulate_power()` failure-count surface + narrow except clause** (PR #326) — power-simulation replicate loop narrows the exception whitelist from `except Exception` to estimation/data-path failures (`TypeError` and friends now propagate, not silently absorb), and surfaces `n_simulation_failures` on `SimulationPowerResults`. Failure count included in `summary()` and `to_dict()`.

## [3.1.3] - 2026-04-18

### Added
- **Replicate-weight variance and PSU-level bootstrap for dCDH** (PR #311) - `ChaisemartinDHaultfoeuille` now accepts `variance_method="replicate"` for BRR / Fay / JK1 / JKn / SDR inference, and PSU-level multiplier bootstrap when `survey_design.psu` is set. Adds df-aware inference (reduced effective df under replicate variance; propagated through delta / HonestDiD surfaces) plus group-level PSU map construction. Validated via per-cohort aggregation, shared-draw multi-horizon bootstrap alignment, and cross-surface df consistency.
- **Zenodo DOI auto-minting configuration** (PR #321) - `.zenodo.json` at repo root defines release metadata so the next GitHub Release automatically mints a Zenodo DOI (concept DOI + versioned DOI). Also adds a top-level `LICENSE` file for Zenodo archival.

### Fixed
- **Silent sparse→dense lstsq fallback in `ImputationDiD` and `TwoStageDiD`** (PR #319) - when the sparse solver fails and the dense fallback runs, the estimator now emits a `UserWarning` instead of silently switching paths. Regression tests assert the dense fallback SEs remain usable.
- **Non-convergence signaling in TROP alternating-minimization solvers** (PR #317) - the global- and local-TROP solvers now emit a `UserWarning` when the alternating-minimization loop exits without meeting tolerance, including LOOCV and bootstrap aggregation paths. Warnings aggregate at top-level call sites to avoid log spam.

### Changed
- **`/bump-version` skill updates `CITATION.cff`** (PR #320) - internal release-management tooling now keeps `CITATION.cff` `version:` and `date-released:` in sync with the other version surfaces. Resolves a single `RELEASE_DATE` upfront (from the CHANGELOG header if pre-populated, else today's date) and threads it through all date-bearing files — fixes drift that caused v3.1.2 to ship with `CITATION.cff` still pinned at 3.1.1.

## [3.1.2] - 2026-04-18

### Fixed
- **SyntheticDiD catastrophic cancellation at extreme Y scale** (PR #312) - the Frank-Wolfe weight solver lost precision when outcome magnitudes were very large or very small; results are now numerically stable across scales.
- **Non-convergence signaling in FE imputation alternating-projection solvers** (PR #314) - `ImputationDiD`, `TwoStageDiD`, and shared `within_transform` now emit a `UserWarning` when the alternating-projection / weighted-demean loop exits without meeting the tolerance. `max_iter` and `tol` are documented on `within_transform`.
- **Non-convergence signaling in SyntheticDiD Frank-Wolfe solver** (PR #315) - the numpy-path Frank-Wolfe SC weight solver now emits a `UserWarning` when the loop exits without meeting `min_decrease`. Wrapper-level and `max_iter=0` regression tests added.

### Changed
- Refresh `ROADMAP.md` to drop top-level phase numbering and reflect shipped state through v3.1.1 (PR #313). Absorbs dCDH into the Current State estimator list; adds Recently Shipped summary; reorganizes open work as Shipping Next / Under Consideration / AI-Agent Track / Long-term. Updates `docs/business-strategy.md`, `docs/survey-roadmap.md`, `docs/practitioner_decision_tree.rst`, `docs/choosing_estimator.rst`, `docs/api/chaisemartin_dhaultfoeuille.rst`, `README.md`, and `diff_diff/guides/llms-full.txt` to remove stale phase-deferral language now that the deferred items have shipped.
- Bump the `SyntheticDiD(lambda_reg=...)` and `SyntheticDiD(zeta=...)` deprecation warnings' removal target from `v3.1` to `v4.0.0`. Removing public kwargs in a patch / minor release would violate Semantic Versioning; the deprecation stays warning-only throughout the `3.x` line and will be removed in the next major release. Use `zeta_omega` / `zeta_lambda` instead.

## [3.1.1] - 2026-04-16

### Added
- **Jackknife variance estimation for SyntheticDiD** - `variance_method='jackknife'` implements the delete-one-unit jackknife from Arkhangelsky et al. (2021) Section 5. Supports both standard and survey-weighted jackknife with automatic `pweight` propagation. Validated against R `synthdid` package.
- **LinkedIn carousel** for dCDH estimator announcement (`carousel/diff-diff-dcdh-carousel.pdf`)

## [3.1.0] - 2026-04-14

### Added
- **dCDH Phase 3: Complete feature set for `ChaisemartinDHaultfoeuille`** - three sub-releases completing the estimator:
  - **Phase 3a** (PR #300): Placebo SE via multiplier bootstrap (resolves Phase 1 deferral), non-binary treatment support with crossing-cell detection and automatic cell dropping, R parity SE assertions tightened
  - **Phase 3b** (PR #302): Covariate adjustment via `controls` parameter (OLS residualization, Design 2 per-period path for non-binary treatment), group-specific linear trends via `trends_linear=True` (absorbs group-specific slopes before DiD), R `DIDmultiplegtDYN` parity tests for covariates and trends
  - **Phase 3c** (PR #303): HonestDiD sensitivity analysis integration - `honest_did()` method on results with automatic event-study-to-sensitivity bridge, support trimming for non-consecutive horizons, `l_vec` target specification, Delta-RM and Delta-SD smoothness bounds

### Changed
- ROADMAP.md updated: dCDH Phase 3 items marked shipped

## [3.0.2] - 2026-04-12

### Added
- **`ChaisemartinDHaultfoeuille`** (alias `DCDH`) - de Chaisemartin & D'Haultfœuille estimator for **non-absorbing (reversible) treatments**. The only modern staggered DiD estimator that handles treatment switching on AND off. Implements `DID_M` from AER 2020, validated against R `DIDmultiplegtDYN` v2.3.3. Ships Phases 1 and 2:
  - Phase 1: headline `DID_M` with analytical SE, joiners/leavers decompositions, single-lag placebo, multiplier bootstrap, TWFE decomposition diagnostic
  - Phase 2: multi-horizon event study (`L_max`), dynamic placebos, normalized estimator, cost-benefit aggregate (Lemma 4), sup-t simultaneous confidence bands, `plot_event_study()` integration
- **`twowayfeweights()`** - standalone TWFE decomposition diagnostic (Theorem 1, AER 2020)
- **`generate_reversible_did_data()`** - reversible-treatment panel data generator with 7 switch patterns
- **Survey-aware power analysis** - analytical helpers (`compute_power()`, `compute_mde()`, `compute_sample_size()`) accept a `deff` parameter for design-effect adjustment. Simulation helpers (`simulate_power`, `simulate_mde`, `simulate_sample_size`) accept a `survey_config` (`SurveyPowerConfig`) that generates data with complex survey structure and injects a `SurveyDesign` into each simulated fit.
- **`aggregate_survey()` `second_stage_weights` parameter** - choose `"pweight"` (default, population weights) or `"aweight"` (precision weights). pweight output is compatible with all survey-capable estimators; aweight is opt-in for GLS efficiency with estimators marked Full in the survey support matrix.
- **`conditional_pt` parameter** on `generate_survey_did_data()` - simulates scenarios where unconditional parallel trends fail but conditional PT holds after covariate adjustment
- **Tutorial 18: Geo-Experiment Analysis** (`18_geo_experiments.ipynb`) - SyntheticDiD walkthrough for marketing analytics: simulated DMA panel, 5 treated markets, fit + diagnostics + stakeholder summary
- **Practitioner decision tree** (`docs/practitioner_decision_tree.rst`) - "which method fits my business problem?" guide
- **Practitioner getting started guide** (`docs/practitioner_getting_started.rst`) - end-to-end walkthrough with terminology bridge
- **JOSS paper** (`paper.md`, `paper.bib`) - software paper for Journal of Open Source Software submission
- **CONTRIBUTORS.md** - author and contributor credit
- **Standalone CI Gate workflow** (`.github/workflows/ci-gate.yml`) - doc-only PRs no longer block on path-filtered test workflows

### Changed
- `aggregate_survey()` default second-stage weights changed from `aweight` (precision) to `pweight` (population). Users who need the old precision-weighting behavior can pass `second_stage_weights="aweight"`.
- README "For Data Scientists" section with practitioner-facing links and `aggregate_survey()` documentation
- CITATION.cff updated with version and release date
- ROADMAP.md updated: B1a-d marked done, B2b marked done, B3d marked shipped, dCDH entry updated with correct citations

### Fixed
- Doc-only PRs no longer block indefinitely on CI Gate (standalone gate workflow runs on all PRs regardless of path filters)
- `aggregate_survey()` docs no longer overclaim universal estimator compatibility - explicitly document aweight/pweight restrictions per the survey support matrix

## [3.0.1] - 2026-04-07

### Added
- **`aggregate_survey()`** — new function in `diff_diff.prep` that bridges individual-level survey microdata to geographic-period panels for DiD estimation. Computes design-based cell means and precision weights using domain estimation (Lumley 2004), with SRS fallback for small cells. Returns a panel DataFrame and pre-configured `SurveyDesign` for second-stage estimation. Supports both TSL and replicate-weight variance.
- **Python 3.14 support** — upgraded PyO3 from 0.22 to 0.28, updated CI and publish workflow matrices, bumped Rust MSRV to 1.84 for faer 0.24 compatibility.

### Changed
- Updated README Python support matrix to include 3.14

### Fixed
- Fix domain estimation zero-padding for correct design-based cell variance
- Fix SRS fallback weight normalization for scale invariance across replicate designs
- Validate numeric dtype for outcomes/covariates before aggregation (nullable dtype support)
- Validate grouping columns for NaN values

## [3.0.0] - 2026-04-07

v3.0 completes the survey support roadmap: all 16 estimators (15 inference-level +
BaconDecomposition diagnostic) now accept `survey_design`. See v2.8.0–v2.9.1 entries
for the full feature history leading to this release.

### Breaking Changes
- **Remove `bootstrap_weight_type` parameter** from CallawaySantAnna — use `bootstrap_weights` instead (deprecated since v1.0.1)
- **Remove TROP `method="twostep"` alias** — use `method="local"` (deprecated since v2.7.2)
- **Remove TROP `method="joint"` alias** — use `method="global"` (deprecated since v2.7.2)

### Upgrading from v2.x
- `CallawaySantAnna(bootstrap_weight_type="mammen")` → `CallawaySantAnna(bootstrap_weights="mammen")`
- `TROP(method="twostep")` → `TROP(method="local")`
- `TROP(method="joint")` → `TROP(method="global")`

### Deprecated
- SyntheticDiD `lambda_reg` and `zeta` parameters formally scheduled for removal in v3.1 — use `zeta_omega`/`zeta_lambda` instead

### Changed
- Internal attribute `bootstrap_weight_type` renamed to `bootstrap_weights` in bootstrap mixin and StaggeredTripleDifference for consistency
- TROP `set_params()` now validates `method` against `("local", "global")` — previously only validated in `__init__`
- Documentation updated: all survey gap notes for WooldridgeDiD removed, ROADMAP Phase 10 items marked shipped

## [2.9.1] - 2026-04-06

### Added
- **Survey theory document** (`docs/methodology/survey-theory.md`) — formal justification for design-based variance estimation with modern DiD influence functions, citing Binder (1983), Rao & Wu (1988), Shao (1996)
- **Research-grade survey DGP** — 8 new parameters on `generate_survey_did_data()`: `icc`, `weight_cv`, `informative_sampling`, `heterogeneous_te_by_strata`, `te_covariate_interaction`, `covariate_effects`, `strata_sizes`, `return_true_population_att`. All backward-compatible.
- **R validation expansion** — 4 additional estimators cross-validated against R's `survey::svyglm()`: ImputationDiD, StackedDiD, SunAbraham, TripleDifference. Survey R validation coverage now 8 of 16 estimators.
- **LinkedIn carousel** for Wooldridge ETWFE estimator announcement

### Changed
- Survey tutorial rewritten: leads with "Why Survey Design Matters" section showing flat-weight vs design-based comparison with known ground truth, coverage simulation, and false pre-trend detection rates
- Documentation refresh: ROADMAP.md, llms.txt, llms-full.txt, llms-practitioner.txt, choosing_estimator.rst updated for v2.9.0 — added WooldridgeDiD and StaggeredTripleDifference, DDD flowchart branch, standardized estimator counts, qualified survey claims
- Survey roadmap updated: Phase 10a-10d marked shipped, conditional PT noted for 10e

### Fixed
- Fix stale "EfficientDiD covariates + survey not supported" note in choosing_estimator.rst
- Fix WooldridgeDiD described as "ASF-based" for OLS path (OLS uses direct coefficients; ASF only for logit/Poisson)
- Fix dead StaggeredTripleDifference API link in llms.txt
- Fix survey example attribute: `.design_effect` not `.deff` in llms-full.txt
- Fix `subpopulation()` example to show tuple unpacking in llms-full.txt
- Remove 8 resolved items from TODO.md

## [2.9.0] - 2026-04-04

### Added
- **WooldridgeDiD (ETWFE)** estimator — Extended Two-Way Fixed Effects from Wooldridge (2025, 2023). Supports OLS, logit, and Poisson QMLE paths with ASF-based ATT and delta-method SEs. Four aggregation types (simple, group, calendar, event) matching Stata `jwdid_estat`. Alias: `ETWFE`. (PR #216, thanks @wenddymacro)
- **EfficientDiD survey + covariates** — doubly robust covariate path now threads survey weights through all four nuisance estimation stages (outcome regression, propensity ratio sieve, inverse propensity sieve, kernel-smoothed conditional Omega*). Previously raised `NotImplementedError`.
- **Survey real-data validation** (Phase 9) — 15 cross-validation tests against R's `survey` package using three real federal survey datasets:
  - **API** (R `survey` package): TSL variance with strata, FPC, subpopulations, covariates, and Fay's BRR replicates
  - **NHANES** (CDC/NCHS): TSL variance with strata + PSU + nest=TRUE, validating the ACA young adult coverage provision DiD
  - **RECS 2020** (U.S. EIA): JK1 replicate weight variance with 60 pre-computed replicate columns
  - ATT, SE, df, and CI match R to machine precision (< 1e-10) where directly comparable; known deviations documented in REGISTRY.md (TWFE SE differs due to unit FE absorption; subpopulation df differs due to strata preservation)
- **Label-gated CI** — test workflows now require `ready-for-ci` label before running, reducing wasted CI during AI review rounds. AI review workflow always runs.
- **Documentation dependency map** (`docs/doc-deps.yaml`) — maps source files to impacted documentation. New `/docs-impact` skill flags which docs need updating when source files change.

### Changed
- WooldridgeDiD: full interacted covariate basis (D_g × X, f_t × X) for OLS path
- `/submit-pr`, `/push-pr-update`, `/pre-merge-check`, `/docs-check` skills updated for label-gated CI and doc-deps workflow

### Fixed
- Fix WooldridgeDiD OLS unbalanced demeaning and nonlinear never-treated identification
- Fix WooldridgeDiD Poisson dropped-cell bug and anticipation propagation
- Fix EfficientDiD IF-scale mismatch in survey aggregation and zero-weight never-treated guard
- Fix bootstrap clustering and delta-method reduced space in WooldridgeDiD

## [2.8.4] - 2026-04-04

### Added
- **SDR replicate method** (Phase 8a) — Successive Difference Replication for ACS PUMS users. `SurveyDesign(replicate_method="SDR")` with variance formula `V = 4/R * sum((theta_r - theta)^2)`.
- **FPC support for ImputationDiD and TwoStageDiD** (Phase 8b) — finite population correction now threaded through TSL variance for both estimators.
- **Lonely PSU "adjust" in bootstrap** (Phase 8d) — `lonely_psu="adjust"` now works with survey-aware bootstrap (previously raised `NotImplementedError`). Uses Rust & Rao (1996) grand-mean centering.
- **CV on estimates** (Phase 8e) — `coef_var` property on all results objects (SE/estimate). Handles edge cases (SE=0, estimate=0).
- **Weight trimming utility** (Phase 8e) — `trim_weights(data, weight_col, upper=None, lower=None, quantile=None)` in `prep.py` for capping extreme survey weights.
- **ImputationDiD pretrends + survey** (Phase 8e) — pre-trends F-test now survey-aware using subpopulation approach for correct variance under complex designs.
- Updated ImputationDiD tutorial to demonstrate `pretrends=True` event study
- Updated survey tutorial: narrative improvements, chart rendering fixes

### Fixed
- Fix survey pretrend F-test df calculation and rank-deficient survey VCV handling
- Fix `trim_weights` NaN poisoning when weight column contains missing values
- Fix single-singleton PSU warning for lonely_psu="adjust"

## [2.8.3] - 2026-04-02

### Added
- **Silent operation warnings** — 8 operations that previously altered analysis results without informing the user now emit `UserWarning`:
  - TROP lstsq → pseudo-inverse numerical fallback
  - TwoStageDiD NaN masking of unidentified fixed effects (zeroed out with treatment indicator)
  - TwoStageDiD always-treated unit removal (sample size change)
  - CallawaySantAnna silent (g,t) pair skipping (zero treated or control observations)
  - TROP missing treatment indicator fill with 0 (control)
  - Rust → Python backend fallback (previously debug log only)
  - Survey weight normalization (pweights/aweights rescaled to mean=1)
  - `np.inf` → 0 never-treated convention conversion
- **ImputationDiD pre-period event study coefficients** — pre-treatment "effects" (should be ~0 under parallel trends) for visual pre-trends assessment, following BJS (2024) Test 1
- **TwoStageDiD pre-period event study coefficients** — same pre-trends extension
- **Replicate weight expansion** to 7 additional estimators: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD, SunAbraham, StackedDiD, ImputationDiD, TwoStageDiD (coverage: 4/13 → 11/13)

### Changed
- ImputationDiD pre-period coefficients use BJS Test 1 (impute Y(0) for treated units in pre-treatment periods)
- SunAbraham replicate weights use full interaction-weighted refit per replicate with cohort-level SEs

### Fixed
- Fix zero-weight demeaning safety in replicate weight paths
- Fix `df_survey` writeback for rank-deficient replicate designs (df=0)
- Fix ImputationDiD `balance_e` zero-qualifying-cohort fallback in pretrends path
- Fix survey zero-mass (g,t) skip warning gap
- Fix SunAbraham positional assignment in replicate loop

## [2.8.2] - 2026-04-02

### Added
- **EPV diagnostics for propensity score logit** — events-per-variable (EPV) checks with Peduzzi convention (predictors excluding intercept) for CallawaySantAnna IPW/DR, TripleDifference IPW/DR, and StaggeredTripleDifference
- `epv_summary()` / `epv_diagnostics` on post-fit results for CallawaySantAnna, TripleDifference, and StaggeredTripleDifference
- `diagnose_propensity()` pre-estimation helper on CallawaySantAnna
- EPV summary block in TripleDifference `summary()` output
- `epv_threshold` parameter for propensity score estimation — warns on low EPV (default) or escalates via `rank_deficient_action="error"`

### Changed
- Default propensity score fallback behavior: safer defaults with method-specific warning messages
- EPV denominator uses predictor count excluding intercept (Peduzzi et al. 1996 convention)

### Fixed
- Fix TripleDifference survey-weighted fallback propensity score
- Fix NaN cache poisoning in propensity score estimation
- Fix `epv_summary` column schema on empty results
- Fix SDDD EPV: use min-EPV across comparison cohorts with cache diagnostic propagation
- Fix `diagnose_propensity` `np.inf` handling

## [2.8.1] - 2026-04-01

### Added
- **Survey-aware DiD tutorial** (`docs/tutorials/16_survey_did.ipynb`) — Phase 7c complete. Full workflow with strata, PSU, FPC, replicate weights, subpopulation analysis, and DEFF diagnostics. Includes `generate_survey_did_data()` DGP function.
- **Survey R cross-validation** — benchmark scripts and tests comparing TSL variance against R's `survey::svyglm()` for basic DiD and TWFE with full survey designs (strata, PSU, FPC). Committed JSON fixtures for CI without R.
- **HonestDiD methodology review and validation** — 478 lines of methodology tests, paper review document, rewritten optimal FLCI with first-difference reparameterization.
- **StaggeredTripleDifference survey support** — full `SurveyDesign` integration with strata/PSU/FPC, replicate weights, and survey-aware bootstrap.

### Changed
- HonestDiD: rewrite optimal FLCI with proper first-difference reparameterization and centrosymmetric LP optimization
- HonestDiD: use `conf_int` from results instead of hardcoded `1.96*se` in event study plots
- Survey tutorial cross-referenced from choosing_estimator.rst and quickstart.rst

### Fixed
- Fix HonestDiD identified set computation and inference (F1-F6 from Rambachan & Roth 2023)
- Fix FLCI slope count (T not T-1) and constraint formula
- Fix NaN CI misclassification as significant (P0 finding)
- Fix M=0 linear extrapolation and survey df folded nct in REGISTRY.md
- Fix replicate-weight scale invariance and BRR test fixtures
- Fix JK1 populated-PSU guard and narrow warning filter

## [2.8.0] - 2026-03-31

### Added
- **Staggered Triple Difference estimator** (Ortiz-Villavicencio & Sant'Anna 2025)
  - `StaggeredTripleDifference` class with group-time ATT(g,t) for DDD designs with staggered adoption
  - Event study aggregation, pre-treatment placebo effects, multiplier bootstrap inference
  - R benchmark validation against `triplediff` package
  - DGP function `generate_staggered_ddd_data()` for simulation and testing
- **Survey Phase 7a: CS IPW/DR + covariates + survey**
  - DRDID panel nuisance-estimation IF corrections (PS + OR) under survey weights
  - Survey-weighted propensity score estimation and outcome regression
  - IFs account for nuisance parameter estimation uncertainty (Sant'Anna & Zhao 2020, Theorem 3.1)
- **Survey Phase 7b: Repeated cross-sections**
  - `CallawaySantAnna(panel=False)` for repeated cross-section surveys (BRFSS, ACS, CPS)
  - Cross-sectional DRDID: `reg` matches `DRDID::reg_did_rc`, `dr` matches `DRDID::drdid_rc`, `ipw` matches `DRDID::std_ipw_did_rc`
  - Survey weights, covariates, and all estimation methods supported
- **Survey Phase 7d: HonestDiD + survey variance**
  - Survey df and full event-study VCV from IF vectors propagated to sensitivity analysis
  - t-distribution critical values with survey degrees of freedom
  - Bootstrap/replicate designs fall back to diagonal VCV with warning
- **Plotly visualization styling**: thread `marker`, `markersize`, `linewidth`, `capsize`, `ci_linewidth` kwargs through plotly backends (previously silently ignored)
- AI agent discoverability for practitioner guide

### Changed
- HonestDiD now raises `ValueError` on non-consecutive event-time grid (was warning)
- HonestDiD validates full grid around reference period
- Panel IPW/DR PS correction scaling matches R's `H/n`, `asy_rep/n`, `colMeans` convention
- RC IF normalization follows R's `psi` convention with explicit `phi` conversion

### Fixed
- Fix HonestDiD reference-aware pre/post split for varying-base event studies
- Fix HonestDiD `_estimate_max_pre_violation` to use reference-aware pre_periods
- Fix panel M2 gradient scaling for IPW/DR nuisance IF corrections
- Fix VCV index alignment for repeated cross-section aggregation
- Fix replicate-weight df propagation: return per-statistic df instead of mutating shared state
- Fix WIF population consistency: zero df `first_treat` for ineligible units
- Fix bootstrap RCS cohort-mass weighting and stale event-study VCV reset

## [2.7.6] - 2026-03-28

### Added
- **AI practitioner guardrails** based on Baker et al. (2025) "Difference-in-Differences Designs: A Practitioner's Guide"
  - `practitioner.py` module with 8-step workflow enforcement for AI agents
  - Estimator-specific handlers ensuring correct diagnostic ordering (pre-trends before estimation, Bacon decomposition before estimator selection)
  - `docs/llms.txt`, `docs/llms-practitioner.txt`, `docs/llms-full.txt` for AI agent discoverability
  - Evaluation rubric (`docs/practitioner-guide-evaluation.md`) with correctness-aware scoring
- **Survey Phase 6: Advanced features**
  - Survey-aware bootstrap for all 8 bootstrap-using estimators (PSU-level multiplier for CS/Imputation/TwoStage/Continuous/Efficient; Rao-Wu rescaled for SA/SyntheticDiD/TROP)
  - Replicate weight variance estimation (BRR, Fay's BRR, JK1, JKn) for OLS-based and IF-based estimators
  - Per-coefficient DEFF diagnostics comparing survey vs SRS variance
  - Subpopulation analysis via `SurveyDesign.subpopulation()` preserving full design structure
  - CS analytical expansion: strata/PSU/FPC for aggregated SEs via `compute_survey_if_variance()`
  - TROP cross-classified pseudo-strata for survey-aware bootstrap

### Changed
- Estimator-specific guidance for parallel trends tests and placebo checks (no shared templates)
- SDiD and TROP split into separate decision tree branches in practitioner workflow

### Fixed
- Fix replicate weight df calculation using pivoted QR rank with R-compatible tolerance
- Fix replicate IF variance score scaling for EfficientDiD, TripleDiff, ContinuousDiD
- Fix panel-to-unit replicate weight propagation and normalization
- Fix CS zero-mass return type and vectorized guard for survey paths
- Fix `solve_logit` effective-sample validation for zero-weight designs
- Fix subpopulation mask validation and EfficientDiD bootstrap guard

## [2.7.5] - 2026-03-23

### Added
- **Phase 4 survey support** for ImputationDiD, TwoStageDiD, and CallawaySantAnna estimators
  - ImputationDiD/TwoStageDiD: analytical survey inference with weights, strata, and PSU (FPC not supported; bootstrap+survey deferred)
  - CallawaySantAnna: weights-only analytical IF/WIF inference matching R `did::wif()` (strata/PSU/FPC deferred)
  - Survey-aware aggregation for group-time, event-study, and overall ATT
- **EfficientDiD enhancements**: doubly robust covariates path, sieve inverse propensity (Eq 3.12), conditional Omega*
- **Cluster-robust SEs** for EfficientDiD with last-cohort control and Hausman pretest
- **Enhanced visualizations**: synth weights, staircase, dose-response, group-time heatmap, plotly backend
- **Local AI review skill** (`/ai-review-local`) with Responses API, delta-diff re-review, and cost visibility
- Add `plotly` optional dependency group (`pip install diff-diff[plotly]`)

### Changed
- Migrate AI local review from Chat Completions to Responses API
- Split TROP estimator into mixin modules (`trop_local.py`, `trop_global.py`) for maintainability
- Refactor `visualization.py` into `visualization/` subpackage
- Improve review script: full-file context, content-first parsing, tiered matching, fingerprint stability

### Fixed
- Fix CallawaySantAnna reg+cov control IF normalization and survey df calculation
- Fix TripleDifference TSL double-weighting and RA nuisance linearization with survey weights
- Fix ContinuousDiD bread normalization, fweight TSL scaling, and weighted-mass IF linearization
- Fix BaconDecomposition exact-weight survey unit_share and empty-cell guard
- Fix SunAbraham survey weight floor in overall ATT aggregation
- Fix plotly event study for non-numeric periods, heatmap masking, color parser

## [2.7.4] - 2026-03-21

### Added
- **Survey/sampling weights support** (`survey_design` parameter) for `DifferenceInDifferences` and `TwoWayFixedEffects`
  - Taylor-series linearization (TSL) variance estimation with stratified multi-stage designs
  - Probability weights (pweight), frequency weights (fweight), and analytic weights (aweight)
  - Finite population correction (FPC) support
  - PSU-based clustering with lonely PSU handling
  - New `diff_diff/survey.py` module with `SurveyDesign` and `compute_survey_vcov`
- **EfficientDiD validation tests** against Chen, Sant'Anna & Xie (2025) using HRS dataset
  - HRS validation fixture with provenance documentation
  - Shared DGP helper in `tests/helpers/edid_dgp.py`
- Simulation-based power analysis for all registry-backed estimators (MDE, sample size, power curves); unregistered estimators supported via custom `data_generator` and `result_extractor`

### Changed
- Extend power analysis to support all registry-backed estimators with `result_extractor` parameter
- Update power analysis tutorial with simulation-based features
- Reject `absorb + fixed_effects` combination (FWL violation) in both survey and non-survey paths

### Fixed
- TWFE cluster-as-PSU injection for no-PSU survey designs
- Non-unique PSU labels across strata with `nest=False`
- FPC validation moved to `compute_survey_vcov` for effective PSU structure
- Survey HC1 meat formula and weighted rank-deficiency handling
- Zero-SE inference, full-census FPC, fweight contract corrections
- Bootstrap+survey fallback in MultiPeriodDiD
- DDD `_snap_n` floor mismatch and `n_per_cell` suppression scope

## [2.7.3] - 2026-03-19

### Added
- Add aarch64 Linux wheel builds to publish workflow

### Changed
- Improve documentation information architecture
- Fix silent interpreter skip and consolidate Linux jobs in publish workflow

## [2.7.2] - 2026-03-18

### Added
- SEO infrastructure: meta tags, sitemap, llms.txt/llms-full.txt for AI discoverability

### Changed
- Rename TROP `method="twostep"` to `method="local"`; `"twostep"` deprecated, removal in v3.0
- Rename internal TROP `_joint_*` methods to `_global_*` for consistency

### Fixed
- Fix TROPResults schema: report unit counts not observation counts
- Fix llms-full.txt accuracy and dynamic canonical URLs

## [2.7.1] - 2026-03-15

### Changed
- Replace BFGS logit with IRLS for propensity score estimation in CallawaySantAnna
- Reject `pscore_trim=0.0` to prevent infinite IPW weights
- Honor `rank_deficient_action="error"` in propensity score paths
- Validate `pscore_trim` at `fit()` to guard against `set_params` bypass
- Mark slow tests (`@pytest.mark.slow`) and exclude by default for faster local iteration
- Use per-class slow markers in `test_trop.py` for faster pure Python CI

### Fixed
- Vectorize Sun-Abraham bootstrap resampling loop for improved performance

## [2.7.0] - 2026-03-15

### Added
- **EfficientDiD estimator** (`EfficientDiD`) implementing Chen, Sant'Anna & Xie (2025) efficient DiD
- CallawaySantAnna event study SEs (WIF-based) and simultaneous confidence bands (sup-t)
- R comparison tests for event-study SEs and cband critical values
- Non-finite outcome validation in `EfficientDiD.fit()`
- CallawaySantAnna speed benchmarks with baseline results
- Estimator alias documentation in README, quickstart, and API docs

### Changed
- **BREAKING: TROP nuclear norm solver step size fix** — The proximal gradient
  threshold for the L matrix (both `method="global"` and `method="twostep"` with
  finite `lambda_nn`) was over-shrinking singular values by a factor of 2. The
  soft-thresholding threshold was λ_nn/max(δ) when the correct value is
  λ_nn/(2·max(δ)), derived from the Lipschitz constant L_f=2·max(δ) of the
  quadratic gradient. This fix produces higher-rank L matrices and closer
  agreement with exact convex optimization solutions. Users with finite
  `lambda_nn` will observe different ATT estimates. Added FISTA/Nesterov
  acceleration to the twostep inner solver for faster L convergence.
- Add (1-W) weight masking to TROP global method, rename joint→global
- Optimize CallawaySantAnna covariate path with Cholesky and pscore caching
- Update Codex AI review model from gpt-5.2-codex to gpt-5.4

### Fixed
- Fix CallawaySantAnna event study SEs (missing WIF) and simultaneous confidence bands
- Fix analytical and bootstrap WIF pg scaling to use global N
- Fix TROP nuclear norm solver threshold scaling for non-uniform weights
- Fix stale coefficients in TROP global low-rank solver and NaN bootstrap poisoning
- Fix NaN-cell preservation in CallawaySantAnna balance_e aggregation
- Fix not-yet-treated cache keys and dropped-cell warning
- Fix rank-deficiency handling with Cholesky rank checks and reduced-column solve
- Fix Rust convergence criterion, n_valid_treated consistency, and NaN bootstrap SE

## [2.6.1] - 2026-03-08

### Added
- Short aliases for all estimators (e.g., `DiD`, `TWFE`, `EventStudy`, `CS`, `SDiD`)

### Changed
- Update roadmap for v2.6.0: reflect completed work and refresh priorities
- Add ContinuousDiD to ReadTheDocs API reference and choosing guide
- Add SPT identification caveat and data requirements per review
- Add time-invariant dose requirement to data requirements

### Fixed
- Fix alias docs wording: clarify TROP has no alias
- Fix ContinuousDiD SE method: influence function, not delta method
- Fix methodology doc: influence functions, not delta method for ContinuousDiD SEs
- Fix dollar sign escaping in continuous DiD tutorial
- Fix continuous DiD tutorial formatting: escape dollar signs and split chart cell
- Fix methodology claims and slide numbering per PR review

## [2.6.0] - 2026-02-22

### Added
- **Continuous DiD estimator** (`ContinuousDiD`) implementing Callaway, Goodman-Bacon & Sant'Anna (2024)
  for continuous treatment dose-response analysis
  - `ContinuousDiDResults` with dose-response curves and event-study effects
  - `DoseResponseCurve` with bootstrap p-values
  - Analytical and bootstrap event-study SEs
  - P(D=0) warning for low-probability control groups
- Stacked DiD tutorial (Tutorial 13) with Q-weight computation walkthrough

### Changed
- Clarify aggregate Q-weight computation for unbalanced panels in Stacked DiD tutorial
- Replace SunAbraham manual bootstrap stats with NaN-gated utility

### Fixed
- Fix not-yet-treated control mask to respect anticipation parameter in ContinuousDiD
- Guard non-finite `original_effect` in `compute_effect_bootstrap_stats`
- Fix bootstrap NaN propagation for rank-deficient cells
- Fix NaN propagation in rank-deficient spline predictions
- Guard bootstrap NaN propagation: SE/CI/p-value all NaN when SE invalid
- Fix bootstrap ACRT^{glob} centering bug
- Fix bootstrap percentile inference and analytical event-study SE scaling
- Fix control group bug and dose validation in ContinuousDiD

## [2.5.0] - 2026-02-19

### Added
- Stacked DiD estimator (`StackedDiD`) implementing Wing, Freedman & Hollingsworth (2024)
  with corrective Q-weights for compositional balance across event times
- Sub-experiment construction per adoption cohort with clean (never-yet-treated) controls
- IC1/IC2 trimming for compositional balance across event times
- Q-weights for aggregate, population, or sample share estimands (Table 1)
- WLS event study regression via sqrt(w) transformation
- `stacked_did()` convenience function
- R benchmark scripts for Stacked DiD validation (`benchmarks/R/benchmark_stacked_did.R`)
- Comprehensive test suite for Stacked DiD (`tests/test_stacked_did.py`)

### Fixed
- NaN inference handling in pure Python mode for edge cases

## [2.4.3] - 2026-02-19

### Changed
- Rewrite TripleDifference estimator to match R's `triplediff::ddd()` — all 3 estimation
  methods (DR, IPW, RA) now use three-DiD decomposition with influence function SE, achieving
  <0.001% relative difference from R across all 24 comparisons (4 DGPs × 3 methods × 2 covariate settings)
- Validate cluster column in TripleDifference for proper cluster-robust SEs
- Handle non-finite influence function propagation in TripleDifference edge cases
- Propensity score fallback uses Hessian-based SE when score optimization fails
- Improved R-squared consistency across estimation methods

### Fixed
- Fix low cell count warning and overlap detection in TripleDifference IPW
- Fix cluster SE computation to use functional (groupby) approach instead of loop
- Fix rank deficiency handling in TripleDifference regression adjustment

### Added
- 91 methodology verification tests for TripleDifference (`tests/test_methodology_triple_diff.py`)
- R benchmark scripts for triple difference validation (`benchmarks/R/benchmark_triplediff.R`)
- Update METHODOLOGY_REVIEW.md to reflect completed TripleDifference review

## [2.4.2] - 2026-02-18

### Added
- **Conditional BLAS linking for Rust backend** — Apple Accelerate on macOS, OpenBLAS on Linux.
  Pre-built wheels now use platform-optimized BLAS for matrix-vector and matrix-matrix
  operations across all Rust-accelerated code paths (weights, OLS, TROP). Windows continues
  using pure Rust (no external dependencies). Improves Rust backend performance at larger scales.
- `rust_backend_info()` diagnostic function in `diff_diff._backend` — reports compile-time
  BLAS feature status (blas, accelerate, openblas)

### Fixed
- **Rust SDID backend performance regression at scale** — Frank-Wolfe solver was 3-10x slower than pure Python at 1k+ scale
  - Gram-accelerated FW loop for time weights: precomputes A^T@A, reducing per-iteration cost from O(N×T0) to O(T0) (~100x speedup per iteration at 5k scale)
  - Allocation-free FW loop for unit weights: 1 GEMV per iteration (was 3), zero heap allocations (was ~8)
  - Dispatch based on problem dimensions: Gram path when T0 < N, standard path when T0 >= N
  - Rust backend now faster than pure Python at all scales

## [2.4.1] - 2026-02-17

### Added
- Tutorial notebook for Two-Stage DiD (Gardner 2022) (`docs/tutorials/12_two_stage_did.ipynb`)

### Changed
- Module splits for large files: ImputationDiD, TwoStageDiD, and TROP each split into separate results and bootstrap submodules
- Migrated remaining inline inference computations to `safe_inference()` utility
- Replaced `@` operator with `np.dot()` at observation-dimension sites to avoid Apple M4 BLAS warnings
- Updated TODO.md and ROADMAP.md for accuracy post-v2.4.0

### Fixed
- Matplotlib import guards added to tutorials 11 and 12
- Various bug fixes from code quality cleanup (diagnostics, estimators, linalg, staggered, sun_abraham, synthetic_did, triple_diff)

## [2.4.0] - 2026-02-16

### Added
- **Gardner (2022) Two-Stage DiD estimator** (`TwoStageDiD`)
  - Two-stage estimator: (1) estimate unit+time FE on untreated obs, (2) regress residualized outcomes on treatment indicators
  - `TwoStageDiDResults` with overall ATT, event study, group effects, per-observation treatment effects
  - `TwoStageBootstrapResults` for multiplier bootstrap inference on GMM influence function
  - `two_stage_did()` convenience function for quick estimation
  - Point estimates identical to ImputationDiD; different variance estimator (GMM sandwich vs. conservative)
  - No finite-sample adjustments (raw asymptotic sandwich, matching R `did2s`)
- Proposition 5 detection for unidentified long-run horizons without never-treated units

### Changed
- Workflow improvements to reduce PR review rounds

### Fixed
- Zero-observation horizons/cohorts producing se=0 instead of NaN in TwoStageDiD
- Edge case fixes for TwoStageDiD (PR review feedback)
- Grep PCRE patterns updated to use POSIX character classes

## [2.3.2] - 2026-02-16

### Added
- **Python 3.13 support** with upper version cap (`>=3.9,<3.14`)

### Changed
- **Sun-Abraham methodology review** (PR #153)
  - IW aggregation weights now use event-time observation counts (not group sizes)
  - Normalize `np.inf` never-treated encoding before treatment group detection
  - Add R benchmark scripts and methodology-aligned tests
- Use `rank_deficient_action` and `np.errstate` instead of broad `RuntimeWarning` filter in SDID tutorial

### Fixed
- Sun-Abraham bootstrap NaN propagation for non-finite ATT estimates
- Sun-Abraham df_adjustment off-by-one in analytical SE computation
- CI pandas compatibility for SunAbraham bootstrap inference
- SyntheticDiD tutorial: eliminate pre-treatment fit warnings

## [2.3.1] - 2026-02-15

### Fixed
- Fix docs/PyPI version mismatch (issue #146) — RTD now builds versioned docs from source
- Fix RTD docs build failure caused by Rust/maturin compilation timeout on ReadTheDocs

### Changed
- Remove Rust outer-loop variance estimation for SyntheticDiD (placebo and bootstrap)
  - Fixes SE mismatch between pure Python and Rust backends (different RNG sequences)
  - Fixes Rust performance regression at 1k+ scale (memory bandwidth saturation from rayon parallelism)
  - Inner Frank-Wolfe weight computation still uses Rust when available

### Documentation
- Re-run SyntheticDiD benchmarks against R after Frank-Wolfe methodology rewrite
- Updated `docs/benchmarks.rst` SDID validation results, performance tables, and known differences
- ATT now matches R to < 1e-10 (previously 0.3% diff) since both use Frank-Wolfe optimizer

## [2.3.0] - 2026-02-09

### Added
- **Borusyak-Jaravel-Spiess (2024) Imputation DiD estimator** (`ImputationDiD`)
  - Efficient imputation estimator for staggered DiD designs
  - OLS on untreated observations for unit+time FE, impute counterfactual Y(0), aggregate
  - Conservative variance (Theorem 3) with `aux_partition` parameter for SE tightness
  - Pre-trend test (Equation 9) via `results.pretrend_test()`
  - Percentile bootstrap inference
  - Influence-function bootstrap with sparse variance and weight/covariate fixes
  - Absorbing-treatment validation for non-constant `first_treat`
  - Empty event-study warning for unidentified long-run horizons
- **`/paper-review` skill** for academic paper methodology extraction
- **`/read-feedback-revise` skill** for addressing PR review comments
- **`--pr` flag for `/review-plan` skill** to review plans posted as PR comments
- **`--updated` flag for `/review-plan` skill** for re-reviewing revised plans
- **MultiPeriodDiD vs R (fixest) benchmark** for cross-language validation

### Changed
- Shortened test suite runtime with parallel execution and reduced iterations

### Fixed
- **TWFE within-transformation bug** identified during methodology review
- TWFE: added non-{0,1} binary time warning, ATT invariance tests, and R fixture caching
- TWFE: single-pass demeaning, HC1 test fix, fixest coeftable comparison
- MultiPeriodDiD: added unit FE and NaN guard for R comparison benchmark
- Removed tracked PDF from repo and gitignored papers directory

## [2.2.1] - 2026-02-07

### Changed
- **MultiPeriodDiD: Full event-study specification** (BREAKING)
  - Treatment × period interactions now created for ALL periods (pre and post),
    not just post-treatment
  - Pre-period coefficients available for parallel trends assessment
  - Default reference period changed from first to last pre-period (e=-1 convention)
    with FutureWarning for one release cycle
  - `period_effects` dict now contains both pre and post period effects
  - `to_dataframe()` includes `is_post` column
  - `summary()` output now shows pre-period effects section
  - t_stat uses `np.isfinite(se) and se > 0` guard (consistent with other estimators)

### Added
- Time-varying treatment warning when `unit` is provided and treatment varies
  within units (guides users toward ever-treated indicator D_i)
- `unit` parameter to `MultiPeriodDiD.fit()` for staggered adoption detection
- `reference_period` and `interaction_indices` attributes on `MultiPeriodDiDResults`
- `pre_period_effects` and `post_period_effects` convenience properties on results
- Pre-period section in `summary()` output with reference period indicator
- `ValueError` when `reference_period` is set to a post-treatment period
- Staggered adoption warning when treatment timing varies across units (with `unit` param)
- Informative KeyError when accessing reference period via `get_effect()`

### Removed
- **TROP `variance_method` parameter** — Jackknife variance estimation removed.
  Bootstrap (the only method specified in Athey et al. 2025) is now always used.
  The `variance_method` field has also been removed from `TROPResults`.
- **TROP `max_loocv_samples` parameter** — Control observation subsampling removed
  from LOOCV tuning parameter selection. Equation 5 of Athey et al. (2025) explicitly
  sums over ALL control observations where D=0; the previous subsampling (default 100)
  was not specified in the paper. LOOCV now uses all control observations, making
  tuning fully deterministic. Inner LOOCV loops in the Rust backend are parallelized
  to compensate for the increased observation count.

### Fixed
- HonestDiD: filter non-finite period effects from MultiPeriodDiD results
  (prevents NaN propagation into sensitivity bounds; raises ValueError
  when no finite pre- or post-period effects remain)
- HonestDiD VCV extraction: now uses interaction sub-VCV instead of full regression VCV
  (via `interaction_indices` period → column index mapping)
- MultiPeriodDiD: `avg_se` guard now checks `np.isfinite()` (matches per-period pattern;
  prevents `avg_t_stat=0` / `avg_p_value=1` when variance is infinite)
- HonestDiD: extraction now uses explicit pre-then-post ordering instead of sorted period
  labels (prevents misclassification when period labels don't sort chronologically)
- Backend-aware test parameter scaling for pure Python CI performance
- Lower TROP stratified bootstrap threshold floor from 11 to 5 for pure Python CI

## [2.2.0] - 2026-01-27

### Added
- **Windows wheel builds** using pure-Rust `faer` library for linear algebra (PR #115)
  - Eliminates external BLAS/LAPACK dependencies (no OpenBLAS or Intel MKL required)
  - Enables cross-platform wheel builds for Linux, macOS, and Windows
  - Simplifies installation on all platforms

### Changed
- **Rust backend migrated from nalgebra/ndarray to faer** (PR #115)
  - OLS solver now uses faer's SVD implementation
  - Robust variance estimation uses faer's matrix operations
  - TROP distance calculations use faer primitives
  - Maintains numerical parity with existing NumPy backend

### Fixed
- **Rust backend numerical stability improvements** (PR #115)
  - Improved singular matrix detection with condition number checks
  - NaN propagation in variance-covariance estimation
  - Fallback to Python backend on numerical instability with warning
  - Underdetermined SVD handling (n < k case)
- **macOS CI compatibility** for Python 3.14 with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY`

## [2.1.9] - 2026-01-26

### Added
- **Unified LOOCV for TROP joint method** with Rust acceleration (PR #113)
  - Leave-one-out cross-validation for rank and regularization parameter selection
  - Rust backend provides significant speedup for LOOCV grid search

### Fixed
- **TROP joint method Rust/Python parity** (PR #113)
  - Fixed valid_count bug in LOOCV computation
  - Proper NaN exclusion for units with no valid pre-period data
  - Zero weight assignment for units missing pre-period data
  - Jackknife variance estimation fixes
  - Staggered adoption validation and simultaneous adoption enforcement
  - Treated-pre NaN handling improvements
  - LOOCV subsampling fix for Python-only path

## [2.1.8] - 2026-01-25

### Added
- **`/push-pr-update` skill** for committing and pushing PR revisions
  - Commits local changes to current branch and pushes to remote
  - Triggers AI code review automatically
  - Robust handling for fork repos, unpushed commits, and upstream tracking

### Fixed
- **TROP estimator methodology alignment** (PR #110)
  - Aligned with paper methodology (Equation 5, D matrix semantics)
  - NaN propagation and LOOCV warnings improvements
  - Rust backend test alignment with new loocv_grid_search return signature
  - LOOCV cycling, D matrix validation fixes
  - Final estimation infinity handling and edge case fixes
  - Absorbing-state gap detection and n_post_periods fix

### Changed
- **`/submit-pr` skill improvements** (PR #111)
  - Case-insensitive secret scanning with POSIX ERE regex
  - Verify origin ref exists before push
  - Dynamic default branch detection with fallback
  - Robust handling for unpushed commits, fork repos
  - Files count display in PR summary

## [2.1.7] - 2026-01-25

### Fixed
- **`plot_event_study` reference period normalization behavior**
  - Effects are now only normalized when `reference_period` is explicitly provided
  - Auto-inferred reference periods only apply hollow marker styling (no normalization)
  - Reference period SE is set to NaN during normalization (constraint, not estimate)
  - Updated docstring to clarify explicit vs auto-inferred behavior

### Changed
- Refactored visualization tests to reuse `cs_results` fixture for better performance

## [2.1.6] - 2026-01-24

### Added
- **Methodology verification tests** for DifferenceInDifferences estimator
  - Comprehensive test suite validating all REGISTRY.md requirements
  - Tests for formula interface, coefficient extraction, rank deficiency handling
  - Singleton cluster variance estimation behavioral tests

### Changed
- **REGISTRY.md documentation improvements**
  - Clarified singleton cluster formula notation (u_i² X_i X_i' instead of ambiguous residual² × X'X)
  - Verified DifferenceInDifferences behavior against documented requirements

## [2.1.5] - 2026-01-22

### Added
- **METHODOLOGY_REVIEW.md** tracking document for methodology review progress
  - Review status summary table for all 12 estimators
  - Detailed notes template for each estimator by category
  - Review process guidelines with checklist and priority ordering
- **`base_period` parameter** for CallawaySantAnna pre-treatment effect computation
  - "varying" (default): Pre-treatment uses t-1 as base (consecutive comparisons)
  - "universal": All comparisons use g-anticipation-1 as base
  - Matches R `did::att_gt()` base_period parameter
- **Pre-merge-check skill** (`/pre-merge-check`) for automated PR validation
  - Pattern checks for NaN handling consistency
  - Context-specific checklist generation

### Changed
- **Tutorial 02 improvements**: Added pre-trends section, clarified base_period interaction with anticipation

### Fixed
- Not-yet-treated control group now properly excludes cohort g when computing ATT(g,t)
- Aggregation t_stat uses NaN (not 0.0) when SE is non-finite or zero
- Bootstrap inference for pre-treatment effects with `base_period="varying"`
- NaN propagation for empty post-treatment effects in CallawaySantAnna
- Grep word boundary pattern in pre-merge-check skill

## [2.1.4] - 2026-01-20

### Added
- **Development checklists and workflow improvements** in `CLAUDE.md`
  - Estimator inheritance map showing class hierarchy for `get_params`/`set_params`
  - Test writing guidelines for fallback paths, parameters, and warnings
  - Checklists for adding parameters and warning/error handling
- **R-style rank deficiency handling** across all estimators
  - `rank_deficient_action` parameter: "warn" (default), "error", or "silent"
  - Dropped columns have NaN coefficients (like R's `lm()`)
  - VCoV matrix has NaN for rows/cols of dropped coefficients
  - Propagated to all estimators: DifferenceInDifferences, MultiPeriodDiD, TwoWayFixedEffects, CallawaySantAnna, SunAbraham, TripleDifference, TROP, SyntheticDiD

### Fixed
- `get_params()` now includes `rank_deficient_action` parameter (fixes sklearn cloning)
- NaN vcov fallback in Rust backend for rank-deficient matrices
- MultiPeriodDiD vcov/df computation for rank-deficient designs
- Average ATT inference for rank-deficient designs

### Changed
- Rank tolerance aligned with R's `lm()` default for consistent behavior

## [2.1.3] - 2026-01-19

### Fixed
- TROP estimator paper conformance issues (Athey et al. 2025)
  - Control set now includes pre-treatment observations of eventually-treated units (Issue A)
  - Unit distance computation excludes target period per Equation 3 (Issue B)
  - Nuclear norm update uses weighted proximal gradient instead of unweighted soft-thresholding (Issue C)
  - Bootstrap sampling now stratifies by treatment status per Algorithm 3 (Issue D)
- TROP Rust backend alignment with paper specification
  - Weight normalization to sum to 1 (probability weights)
  - Weighted proximal gradient for L update with step size η ≤ 1/max(W)

### Changed
- Cleaned up unused parameters from TROP Rust API
  - Removed `control_unit_idx` and `unit_dist_matrix` from public functions
  - Per-observation distances now computed dynamically (more accurate, slightly slower)

## [2.1.2] - 2026-01-19

### Added
- **Consolidated DGP functions** in `prep.py` for all supported DiD designs
  - `generate_did_data()` - Basic 2x2 DiD data generation
  - `generate_staggered_data()` - Staggered adoption data for Callaway-Sant'Anna/Sun-Abraham
  - `generate_factor_data()` - Factor model data for TROP/SyntheticDiD
  - `generate_ddd_data()` - Triple Difference (DDD) design data
  - `generate_panel_data()` - Panel data with optional parallel trends violations
  - `generate_event_study_data()` - Event study data with simultaneous treatment

### Changed
- **Clean up development tracking files** for v2.1.1 release
  - Removed completed items from TODO.md (now tracked in CHANGELOG)
  - Updated ROADMAP.md version numbers and removed shipped TROP section
  - Updated `prep.py` line count in Large Module Files table (1338 → 1993)

## [2.1.1] - 2026-01-19

### Added
- **Rust backend acceleration for TROP estimator** delivering 5-20x overall speedup
  - `compute_unit_distance_matrix` - Parallel pairwise RMSE computation for donor matching
  - `loocv_grid_search` - Parallel leave-one-out cross-validation across 180 parameter combinations
  - `bootstrap_trop_variance` - Parallel bootstrap variance estimation
  - Automatic fallback to Python when Rust backend unavailable
  - Logging for Rust fallback events to aid debugging
- **`/bump-version` skill** for release management
  - Updates version in `__init__.py`, `pyproject.toml`, and `rust/Cargo.toml`
  - Generates CHANGELOG entries from git commits
  - Adds comparison links automatically
- **`/review-pr` skill** for code review workflow

### Changed
- **TROP estimator performance optimizations** (Python backend)
  - Vectorized distance matrix computation using NumPy broadcasting
  - Extracted tuning constants to module-level for clarity
  - Added `TROPTuningParams` TypedDict for parameter documentation

### Fixed
- Tutorial notebook validation errors in `10_trop.ipynb`
- Pre-existing RuntimeWarnings in CallawaySantAnna bootstrap (documented)
- TROP `pre_periods` parameter handling for edge cases

## [2.1.0] - 2026-01-17

### Added
- **Triply Robust Panel (TROP) estimator** implementing Athey, Imbens, Qu & Viviano (2025)
  - `TROP` class combining three robustness components:
    - Factor model adjustment via SVD (removes unobserved confounders with factor structure)
    - Synthetic control style unit weights
    - SDID style time weights
  - `TROPResults` dataclass with ATT, factors, loadings, unit/time weights
  - `trop()` convenience function for quick estimation
  - Automatic rank selection methods: cross-validation (`'cv'`), information criterion (`'ic'`), elbow detection (`'elbow'`)
  - Bootstrap and placebo-based variance estimation
  - Full integration with existing infrastructure (exports in `__init__.py`, sklearn-compatible API)
  - Tutorial notebook: `docs/tutorials/10_trop.ipynb`
  - Comprehensive test suite: `tests/test_trop.py`

**Reference**: Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). "Triply Robust Panel Estimators." *Working Paper*. [arXiv:2508.21536](https://arxiv.org/abs/2508.21536)

## [2.0.3] - 2026-01-17

### Changed
- **Rust backend performance optimizations** delivering up to 32x speedup for bootstrap operations
  - Bootstrap weight generation now 16x faster on average (up to 32x for Webb distribution)
  - Direct `Array2` allocation eliminates intermediate `Vec<Vec<f64>>` (~50% memory reduction)
  - Rayon chunk size tuning (`min_len=64`) reduces parallel scheduling overhead
  - Webb distribution uses lookup table instead of 6-way if-else chain

### Added
- **LinearRegression helper class** in `linalg.py` for code deduplication
  - High-level OLS wrapper with unified coefficient extraction and inference
  - Used by DifferenceInDifferences, TwoWayFixedEffects, SunAbraham, TripleDifference
  - Provides `InferenceResult` dataclass for coefficient-level statistics
- **Cholesky factorization** for symmetric positive-definite matrix inversion in Rust backend
  - ~2x faster than LU decomposition for well-conditioned matrices
  - Automatic fallback to LU for near-singular or indefinite matrices
- **Vectorized variance computation** in Rust backend
  - HC1 meat computation: `X' @ (X * e²)` via BLAS instead of O(n×k²) loop
  - Score computation: broadcast multiplication instead of O(n×k) loop
- **Static BLAS linking options** in `rust/Cargo.toml`
  - `openblas-static` and `intel-mkl-static` features for standalone distribution
  - Eliminates runtime BLAS dependency at cost of larger binary size

## [2.0.2] - 2026-01-15

### Fixed
- **CallawaySantAnna SE computation** now exactly matches R's `did` package
  - Fixed weight influence function (wif) formula for "simple" aggregation
  - Corrected `pg` computation: uses `n_g / n_all` (matching R) instead of `n_g / total_treated`
  - Fixed wif iteration: iterates over keepers (post-treatment pairs) with individual ATT(g,t) values
  - SE difference reduced from ~2.5% to <0.01% vs R's `did` package (essentially exact match)
  - Point estimates unchanged; all existing tests pass

## [2.0.1] - 2026-01-13

### Added
- **Shared within-transformation utilities** in `utils.py`
  - `demean_by_group()` - One-way fixed effects demeaning
  - `within_transform()` - Two-way (unit + time) FE transformation
  - Reduces code duplication across `estimators.py`, `twfe.py`, `sun_abraham.py`, `bacon.py`

### Fixed
- **DataFrame fragmentation warning** - Build columns in batch instead of iteratively

### Changed
- Reverted untested Rust backend optimizations (Cholesky factorization, reduced allocations) - these will be re-added when proper testing infrastructure is available

## [2.0.0] - 2026-01-12

### Added
- **Optional Rust backend** for accelerated computation
  - 4-8x speedup for SyntheticDiD and bootstrap operations
  - Parallel bootstrap weight generation (Rademacher, Mammen, Webb)
  - Accelerated OLS solver using OpenBLAS/MKL
  - Cluster-robust variance estimation
  - Synthetic control weight optimization with simplex projection
  - Pre-built wheels for Linux x86_64 and macOS ARM64
  - Pure Python fallback for all other platforms
- **`diff_diff/_backend.py`** - Backend detection and configuration module
  - `HAS_RUST_BACKEND` flag exported in main package
  - `DIFF_DIFF_BACKEND` environment variable for backend control:
    - `'auto'` (default) - Use Rust if available, fall back to Python
    - `'python'` - Force pure Python mode
    - `'rust'` - Force Rust mode (fails if unavailable)
- **Rust source code** in `rust/` directory
  - `rust/src/lib.rs` - PyO3 module definition
  - `rust/src/bootstrap.rs` - Parallel bootstrap weight generation
  - `rust/src/linalg.rs` - OLS solver and robust variance estimation
  - `rust/src/weights.rs` - Synthetic control weights and simplex projection
- **Rust backend test suite** - `tests/test_rust_backend.py` for equivalence testing

### Changed
- Package version bumped from 1.4.0 to 2.0.0 (major version for new backend)
- CI/CD updated to build Rust extensions with maturin
- ReadTheDocs now installs from PyPI (pre-built wheels with Rust backend)

## [1.4.0] - 2026-01-11

### Added
- **Unified linear algebra backend** (`diff_diff/linalg.py`)
  - `solve_ols()` - Optimized OLS solver using scipy's gelsy LAPACK driver
  - `compute_robust_vcov()` - Vectorized (clustered) robust variance-covariance
  - Single optimization point for all estimators; prepares for future Rust backend
  - New `tests/test_linalg.py` with comprehensive tests

### Changed
- **Major performance improvements** - All estimators now significantly faster
  - BasicDiD/TWFE @ 10K: 0.835s → 0.011s (76x faster, now 4.2x faster than R)
  - CallawaySantAnna @ 10K: 2.234s → 0.109s (20x faster, now 7.2x faster than R)
  - All results numerically identical to previous versions
- **CallawaySantAnna optimizations** (`staggered.py`)
  - Pre-computed wide-format outcome matrix and cohort masks
  - Vectorized ATT(g,t) computation using numpy operations (23x faster)
  - Batch bootstrap weight generation
  - Vectorized multiplier bootstrap using matrix operations (26x faster)
- **TWFE optimization** (`twfe.py`)
  - Cached groupby indexes for within-transformation
- **All estimators migrated** to unified `linalg.py` backend
  - `estimators.py`, `twfe.py`, `staggered.py`, `triple_diff.py`,
    `synthetic_did.py`, `sun_abraham.py`, `utils.py`

### Behavioral Changes
- **Rank-deficient design matrices**: The new `gelsy` LAPACK driver handles
  rank-deficient matrices gracefully (returning a least-norm solution) rather
  than raising an explicit error. Previously, `DifferenceInDifferences` would
  raise `ValueError("Design matrix is rank-deficient")`. Users relying on this
  error for collinearity detection should validate their design matrices
  separately. Results remain numerically correct for well-specified models.

## [1.3.1] - 2026-01-10

### Added
- **SyntheticDiD placebo-based variance estimation** matching R's `synthdid` package methodology
  - New `variance_method` parameter with options `"bootstrap"` (default) and `"placebo"`
  - Placebo method implements Algorithm 4 from Arkhangelsky et al. (2021):
    1. Randomly permutes control unit indices
    2. Designates N₁ controls as pseudo-treated (matching actual treated count)
    3. Renormalizes original unit weights for remaining pseudo-controls
    4. Computes SDID estimate with renormalized weights
    5. Repeats for `n_bootstrap` replications
    6. SE = sqrt((r-1)/r) × sd(estimates)
  - Provides methodological parity with R's `synthdid::vcov(method = "placebo")`
  - `n_bootstrap` parameter now used for both bootstrap and placebo replications
  - `SyntheticDiDResults` now tracks `variance_method` and `n_bootstrap` attributes
  - Results summary displays variance method and replications count

**Reference**: Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088-4118.

## [1.3.0] - 2026-01-09

### Added
- **Triple Difference (DDD) estimator** implementing Ortiz-Villavicencio & Sant'Anna (2025)
  - `TripleDifference` class for DDD designs where treatment requires two criteria (group AND partition)
  - `TripleDifferenceResults` dataclass with ATT, SEs, cell means, and diagnostics
  - `triple_difference()` convenience function for quick estimation
  - Three estimation methods: regression adjustment (`reg`), inverse probability weighting (`ipw`), and doubly robust (`dr`)
  - Proper covariate handling (unlike naive DDD implementations that difference two DiDs)
  - Propensity score trimming for IPW/DR methods
  - Cluster-robust standard errors support
  - Tutorial notebook: `docs/tutorials/08_triple_diff.ipynb`

**Reference**: Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). "Better Understanding Triple Differences Estimators." *Working Paper*. [arXiv:2505.09942](https://arxiv.org/abs/2505.09942)

## [1.2.1] - 2026-01-08

### Added
- **Expanded test coverage** for edge cases:
  - Wild bootstrap with very few clusters (< 5), including 2-3 cluster scenarios
  - Unbalanced panels with missing periods across units
  - Single treated unit scenarios for DiD and Synthetic DiD
  - Perfect collinearity detection (validates clear error messages)
  - CallawaySantAnna with single treatment cohort
  - SyntheticDiD with insufficient pre-treatment periods

### Changed
- **Refactored CallawaySantAnna bootstrap**: Extracted `_compute_effect_bootstrap_stats()` helper method for cleaner code and reduced duplication in bootstrap statistics computation.

## [1.2.0] - 2026-01-07

### Added
- **Pre-Trends Power Analysis** (Roth 2022) for assessing informativeness of pre-trends tests
  - `PreTrendsPower` class for computing power and minimum detectable violation (MDV)
  - `PreTrendsPowerResults` dataclass with power, MDV, and test statistics
  - `PreTrendsPowerCurve` for power curves across violation magnitudes
  - `compute_pretrends_power()` and `compute_mdv()` convenience functions
  - Multiple violation types: `linear`, `constant`, `last_period`, `custom`
  - Integration with Honest DiD via `sensitivity_to_honest_did()` method
  - `plot_pretrends_power()` visualization for power curves
  - Tutorial notebook: `docs/tutorials/07_pretrends_power.ipynb`
  - Full API documentation: `docs/api/pretrends.rst`

**Reference**: Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322.

### Fixed
- **Reference period handling in pre-trends analysis**: Fixed bug where reference period was incorrectly assigned `avg_se` instead of being excluded from power calculations. Now properly excludes the omitted reference period from the joint Wald test.

## [1.1.1] - 2026-01-06

### Fixed
- **SyntheticDiD bootstrap error handling**: Bootstrap now raises clear `ValueError` when all iterations fail, instead of silently returning SE=0.0. Added warnings for edge cases (single successful iteration, high failure rate).

- **Diagnostics module error handling**: Improved error messages in `permutation_test()` and `leave_one_out_test()` with actionable guidance. Added warnings when significant iterations fail. Enhanced `run_all_placebo_tests()` to return structured error info including error type.

### Changed
- **Code deduplication**: Extracted wild bootstrap inference logic to shared `_run_wild_bootstrap_inference()` method in `DifferenceInDifferences` base class, used by both `DifferenceInDifferences` and `TwoWayFixedEffects`.

- **Type hints**: Added missing type hints to nested functions:
  - `compute_trend()` in `utils.py`
  - `neg_log_likelihood()` and `gradient()` in `staggered.py`
  - `format_label()` in `prep.py`

## [1.1.0] - 2026-01-05

### Added
- **Sun-Abraham (2021) interaction-weighted estimator** for staggered DiD
  - `SunAbraham` class implementing saturated regression approach
  - `SunAbrahamResults` with event study effects, cohort weights, and overall ATT
  - `SABootstrapResults` for bootstrap inference (SEs, CIs, p-values)
  - Support for `never_treated` and `not_yet_treated` control groups
  - Analytical and cluster-robust standard errors
  - Multiplier bootstrap with Rademacher, Mammen, or Webb weights
  - Integration with `plot_event_study()` visualization
  - Useful robustness check alongside Callaway-Sant'Anna

**Reference**: Sun, L., & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199.

## [1.0.2] - 2026-01-04

### Changed
- Refactored `estimators.py` to reduce module size
  - Moved `TwoWayFixedEffects` to `diff_diff/twfe.py`
  - Moved `SyntheticDiD` to `diff_diff/synthetic_did.py`
  - Backward compatible re-exports maintained in `estimators.py`

### Fixed
- Fixed ReadTheDocs version display by importing from package `__version__`

## [1.0.1] - 2026-01-04

### Fixed
- Tech debt cleanup (Tier 1 + Tier 2)
  - Improved code organization and documentation
  - Fixed minor issues identified in tech debt review

## [1.0.0] - 2026-01-04

### Added
- **Goodman-Bacon decomposition** for TWFE diagnostics
  - `BaconDecomposition` class for decomposing TWFE into weighted 2x2 comparisons
  - `Comparison2x2` dataclass for individual comparisons (treated_vs_never, earlier_vs_later, later_vs_earlier)
  - `BaconDecompositionResults` with weights and estimates by comparison type
  - `bacon_decompose()` convenience function
  - `plot_bacon()` visualization for decomposition results
  - Integration via `TwoWayFixedEffects.decompose()` method
- **Power analysis** for study design
  - `PowerAnalysis` class for analytical power calculations
  - `PowerResults` and `SimulationPowerResults` dataclasses
  - `compute_mde()`, `compute_power()`, `compute_sample_size()` convenience functions
  - `simulate_power()` for Monte Carlo simulation-based power analysis
  - `plot_power_curve()` visualization for power analysis
  - Tutorial notebook: `docs/tutorials/06_power_analysis.ipynb`
- **Callaway-Sant'Anna multiplier bootstrap** for inference
  - `CSBootstrapResults` with standard errors, confidence intervals, p-values
  - Rademacher, Mammen, and Webb weight distributions
  - Bootstrap inference for all aggregation methods
- **Troubleshooting guide** in documentation
- **Standard error computation guide** explaining SE differences across estimators

### Changed
- Updated package status to Production/Stable (was Alpha)
- SyntheticDiD bootstrap now warns when >5% of iterations fail

### Fixed
- Silent bootstrap failures in SyntheticDiD now produce warnings

## [0.6.0]

### Added
- **CallawaySantAnna covariate adjustment** for conditional parallel trends
  - Outcome regression (`estimation_method='reg'`)
  - Inverse probability weighting (`estimation_method='ipw'`)
  - Doubly robust estimation (`estimation_method='dr'`)
  - Pass covariates via `covariates` parameter in `fit()`
- **Honest DiD sensitivity analysis** (Rambachan & Roth 2023)
  - `HonestDiD` class for computing bounds under parallel trends violations
  - Relative magnitudes restriction (`DeltaRM`) - bounds post-treatment violations by pre-treatment
  - Smoothness restriction (`DeltaSD`) - bounds second differences of trend violations
  - Combined restrictions (`DeltaSDRM`)
  - FLCI and C-LF confidence interval methods
  - Breakdown value computation via `breakdown_value()`
  - Sensitivity analysis over M grid via `sensitivity_analysis()`
  - `HonestDiDResults` and `SensitivityResults` dataclasses
  - `compute_honest_did()` convenience function
  - `plot_sensitivity()` for sensitivity analysis visualization
  - `plot_honest_event_study()` for event study with honest CIs
  - Tutorial notebook: `docs/tutorials/05_honest_did.ipynb`
- **API documentation site** with Sphinx
  - Full API reference auto-generated from docstrings
  - "Which estimator should I use?" decision guide
  - Comparison with R packages (did, HonestDiD)
  - Getting started / quickstart guide

### Changed
- Updated mypy configuration for better numpy type compatibility
- Modernized ruff configuration to use `[tool.ruff.lint]` section

### Fixed
- Fixed 21 ruff linting issues (import ordering, unused variables, ambiguous names)
- Fixed 94 mypy type checking issues (Optional types, numpy type casts, assertions)
- Added missing return statement in `run_placebo_test()`

## [0.5.0]

### Added
- **Wild cluster bootstrap** for valid inference with few clusters
  - Rademacher weights (default, good for most cases)
  - Webb's 6-point distribution (recommended for <10 clusters)
  - Mammen's two-point distribution
  - `WildBootstrapResults` dataclass
  - `wild_bootstrap_se()` utility function
  - Integration with `DifferenceInDifferences` and `TwoWayFixedEffects` via `inference='wild_bootstrap'`
- **Placebo tests module** (`diff_diff.diagnostics`)
  - `placebo_timing_test()` - fake treatment timing test
  - `placebo_group_test()` - fake treatment group test
  - `permutation_test()` - permutation-based inference
  - `leave_one_out_test()` - sensitivity to individual treated units
  - `run_placebo_test()` - unified dispatcher for all test types
  - `run_all_placebo_tests()` - comprehensive diagnostic suite
  - `PlaceboTestResults` dataclass
- **Tutorial notebooks** in `docs/tutorials/`
  - `01_basic_did.ipynb` - Basic 2x2 DiD, formula interface, covariates, fixed effects, wild bootstrap
  - `02_staggered_did.ipynb` - Staggered adoption with Callaway-Sant'Anna
  - `03_synthetic_did.ipynb` - Synthetic DiD with unit/time weights
  - `04_parallel_trends.ipynb` - Parallel trends testing and diagnostics
- Comprehensive test coverage (380+ tests)

## [0.4.0]

### Added
- **Callaway-Sant'Anna estimator** for staggered difference-in-differences
  - `CallawaySantAnna` class with group-time ATT(g,t) estimation
  - Support for `never_treated` and `not_yet_treated` control groups
  - Aggregation methods: `simple`, `group`, `calendar`, `event_study`
  - `CallawaySantAnnaResults` with group-time effects and aggregations
  - `GroupTimeEffect` dataclass for individual effects
- **Event study visualization** via `plot_event_study()`
  - Works with `MultiPeriodDiDResults`, `CallawaySantAnnaResults`, or DataFrames
  - Publication-ready formatting with customization options
- **Group effects visualization** via `plot_group_effects()`
- **Parallel trends testing utilities**
  - `check_parallel_trends()` - simple slope-based test
  - `check_parallel_trends_robust()` - Wasserstein distance test
  - `equivalence_test_trends()` - TOST equivalence test

## [0.3.0]

### Added
- **Synthetic Difference-in-Differences** (`SyntheticDiD`)
  - Unit weight optimization for synthetic control
  - Time weight computation for pre-treatment periods
  - Placebo-based and bootstrap inference
  - `SyntheticDiDResults` with weight accessors
- **Multi-period DiD** (`MultiPeriodDiD`)
  - Event-study style estimation with period-specific effects
  - `MultiPeriodDiDResults` with `period_effects` dictionary
  - `PeriodEffect` dataclass for individual period results
- **Data preparation utilities** (`diff_diff.prep`)
  - `generate_did_data()` - synthetic data generation
  - `make_treatment_indicator()` - create treatment from categorical/numeric
  - `make_post_indicator()` - create post-treatment indicator
  - `wide_to_long()` - reshape wide to long format
  - `balance_panel()` - ensure balanced panel data
  - `validate_did_data()` - data validation
  - `summarize_did_data()` - summary statistics by group
  - `create_event_time()` - event time for staggered designs
  - `aggregate_to_cohorts()` - aggregate to cohort means
  - `rank_control_units()` - rank controls by similarity

## [0.2.0]

### Added
- **Two-Way Fixed Effects** (`TwoWayFixedEffects`)
  - Within-transformation for unit and time fixed effects
  - Efficient handling of high-dimensional fixed effects via `absorb`
- **Fixed effects support** in base `DifferenceInDifferences`
  - `fixed_effects` parameter for dummy variable approach
  - `absorb` parameter for within-transformation approach
- **Cluster-robust standard errors**
  - `cluster` parameter for cluster-robust inference
- **Formula interface**
  - R-style formulas like `"outcome ~ treated * post"`
  - Support for covariates in formulas

## [0.1.0]

### Added
- Initial release
- **Basic Difference-in-Differences** (`DifferenceInDifferences`)
  - sklearn-like API with `fit()` method
  - Column name interface for outcome, treatment, time
  - Heteroskedasticity-robust (HC1) standard errors
  - `DiDResults` dataclass with ATT, SE, p-value, confidence intervals
  - `summary()` and `print_summary()` methods
  - `to_dict()` and `to_dataframe()` export methods
  - `is_significant` and `significance_stars` properties

[3.0.1]: https://github.com/igerber/diff-diff/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/igerber/diff-diff/compare/v2.9.1...v3.0.0
[2.9.1]: https://github.com/igerber/diff-diff/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/igerber/diff-diff/compare/v2.8.4...v2.9.0
[2.8.4]: https://github.com/igerber/diff-diff/compare/v2.8.3...v2.8.4
[2.8.3]: https://github.com/igerber/diff-diff/compare/v2.8.2...v2.8.3
[2.8.2]: https://github.com/igerber/diff-diff/compare/v2.8.1...v2.8.2
[2.8.1]: https://github.com/igerber/diff-diff/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/igerber/diff-diff/compare/v2.7.6...v2.8.0
[2.7.6]: https://github.com/igerber/diff-diff/compare/v2.7.5...v2.7.6
[2.7.5]: https://github.com/igerber/diff-diff/compare/v2.7.4...v2.7.5
[2.7.4]: https://github.com/igerber/diff-diff/compare/v2.7.3...v2.7.4
[2.7.3]: https://github.com/igerber/diff-diff/compare/v2.7.2...v2.7.3
[2.7.2]: https://github.com/igerber/diff-diff/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/igerber/diff-diff/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/igerber/diff-diff/compare/v2.6.1...v2.7.0
[2.6.1]: https://github.com/igerber/diff-diff/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/igerber/diff-diff/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/igerber/diff-diff/compare/v2.4.3...v2.5.0
[2.4.3]: https://github.com/igerber/diff-diff/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/igerber/diff-diff/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/igerber/diff-diff/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/igerber/diff-diff/compare/v2.3.2...v2.4.0
[2.3.2]: https://github.com/igerber/diff-diff/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/igerber/diff-diff/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/igerber/diff-diff/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/igerber/diff-diff/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/igerber/diff-diff/compare/v2.1.9...v2.2.0
[2.1.9]: https://github.com/igerber/diff-diff/compare/v2.1.8...v2.1.9
[2.1.8]: https://github.com/igerber/diff-diff/compare/v2.1.7...v2.1.8
[2.1.7]: https://github.com/igerber/diff-diff/compare/v2.1.6...v2.1.7
[2.1.6]: https://github.com/igerber/diff-diff/compare/v2.1.5...v2.1.6
[2.1.5]: https://github.com/igerber/diff-diff/compare/v2.1.4...v2.1.5
[2.1.4]: https://github.com/igerber/diff-diff/compare/v2.1.3...v2.1.4
[2.1.3]: https://github.com/igerber/diff-diff/compare/v2.1.2...v2.1.3
[2.1.2]: https://github.com/igerber/diff-diff/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/igerber/diff-diff/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/igerber/diff-diff/compare/v2.0.3...v2.1.0
[3.7.0]: https://github.com/igerber/diff-diff/compare/v3.6.2...v3.7.0
[3.6.2]: https://github.com/igerber/diff-diff/compare/v3.6.1...v3.6.2
[3.6.1]: https://github.com/igerber/diff-diff/compare/v3.6.0...v3.6.1
[3.6.0]: https://github.com/igerber/diff-diff/compare/v3.5.3...v3.6.0
[3.5.3]: https://github.com/igerber/diff-diff/compare/v3.5.2...v3.5.3
[3.5.2]: https://github.com/igerber/diff-diff/compare/v3.5.1...v3.5.2
[3.5.1]: https://github.com/igerber/diff-diff/compare/v3.5.0...v3.5.1
[3.5.0]: https://github.com/igerber/diff-diff/compare/v3.4.2...v3.5.0
[3.4.2]: https://github.com/igerber/diff-diff/compare/v3.4.1...v3.4.2
[3.4.1]: https://github.com/igerber/diff-diff/compare/v3.4.0...v3.4.1
[3.4.0]: https://github.com/igerber/diff-diff/compare/v3.3.3...v3.4.0
[3.3.3]: https://github.com/igerber/diff-diff/compare/v3.3.2...v3.3.3
[3.3.2]: https://github.com/igerber/diff-diff/compare/v3.3.1...v3.3.2
[3.3.1]: https://github.com/igerber/diff-diff/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/igerber/diff-diff/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/igerber/diff-diff/compare/v3.1.3...v3.2.0
[3.1.3]: https://github.com/igerber/diff-diff/compare/v3.1.2...v3.1.3
[3.1.2]: https://github.com/igerber/diff-diff/compare/v3.1.1...v3.1.2
[3.1.1]: https://github.com/igerber/diff-diff/compare/v3.1.0...v3.1.1
[3.1.0]: https://github.com/igerber/diff-diff/compare/v3.0.2...v3.1.0
[3.0.2]: https://github.com/igerber/diff-diff/compare/v3.0.1...v3.0.2
[2.0.3]: https://github.com/igerber/diff-diff/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/igerber/diff-diff/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/igerber/diff-diff/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/igerber/diff-diff/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/igerber/diff-diff/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/igerber/diff-diff/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/igerber/diff-diff/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/igerber/diff-diff/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/igerber/diff-diff/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/igerber/diff-diff/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/igerber/diff-diff/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/igerber/diff-diff/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/igerber/diff-diff/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/igerber/diff-diff/compare/v0.6.0...v1.0.0
[0.6.0]: https://github.com/igerber/diff-diff/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/igerber/diff-diff/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/igerber/diff-diff/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/igerber/diff-diff/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/igerber/diff-diff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/igerber/diff-diff/releases/tag/v0.1.0
