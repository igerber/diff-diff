# Reporting

This document records the methodology choices embedded in
`BusinessReport` and `DiagnosticReport` — the convenience layer that
produces plain-English stakeholder narratives from any diff-diff result.

Methodology for estimators lives in `REGISTRY.md`. This file is the
single source for reporting-layer decisions; `REGISTRY.md` cross-links
here rather than duplicating content.

## Module

- `diff_diff/business_report.py` — `BusinessReport`, `BusinessContext`.
- `diff_diff/diagnostic_report.py` — `DiagnosticReport`,
  `DiagnosticReportResults`.

Both modules dispatch by `type(results).__name__` lookup to avoid
circular imports across the 16 result classes. They do no estimator
fitting; every effect, SE, p-value, CI, and sensitivity bound is
either read from the fitted result, derived from the result's own
post-fit `aggregate('event_study')` surface, or produced by an
existing diff-diff utility (`compute_honest_did`,
`HonestDiD.sensitivity`, `BaconDecomposition`,
`check_parallel_trends`, `compute_pretrends_power`). The post-fit
derivation is the one report-layer data source beyond direct reads:
when a fit's raw `event_study_effects` field is absent (the modern,
no-fit-time-`aggregate=` path), `DiagnosticReport` resolves
`results.aggregate('event_study')` once (cached) and consumes the
returned `EventStudyResults` container, where applicable per
estimator: CallawaySantAnna and DMLDiD for parallel_trends /
pretrends_power / sensitivity (DMLDiD ALWAYS derives — it never
populates a fit-time surface by design), ImputationDiD and TwoStageDiD
for parallel_trends (`pretrends=True` fits) and heterogeneity,
ContinuousDiD for heterogeneity. StackedDiD, SunAbraham, and dCDH never reach the
derived route (raw surface always populated, or the
`placebo_event_study` branch). For CS the derivation is an
influence-kit recompute; ImputationDiD's
kit is a panel-backed Theorem-3 recompute from the retained working
panel, and TwoStageDiD's runs a fresh Stage-2 OLS + GMM sandwich over
a retained frame copy (ledger rows M-021/M-022) — so those two
producers DO recompute variance from their retained kits, exactly as
their own post-fit `aggregate()` does. Bootstrapped CS fits derive
successfully too: `aggregate('event_study')` replays the fit-time
multiplier bootstrap (percentile inference; the derived container
carries `vcov=None`, so parallel_trends rides the Bonferroni fallback
and pretrends_power/sensitivity the diagonal-covariance fallback, and
the replay's re-emitted fit-time warnings are recorded and
republished per section). Kit-less/legacy-pickle fits,
backend-mismatched CS bootstrap replays, and the other estimators'
bootstrap gates still fail closed: the derivation exception is caught
and surfaced as an explicit per-check skip reason, never substituted
with analytical numbers. The raw field, when present — including the
requested-but-empty `{}` sentinel, which encodes fit-time
configuration such as a `balance_e=` that emptied the window — always
takes precedence and is never re-derived. When the caller
passes the raw panel + column kwargs, `DiagnosticReport` may call
those utilities on the supplied data (2x2 PT via
`check_parallel_trends`, Goodman-Bacon decomposition via
`BaconDecomposition`, and the EfficientDiD Hausman PT-All vs PT-Post
pretest via `EfficientDiD.hausman_pretest`).

The `design_effect` section of `DiagnosticReport.to_dict()` is a
read-only surface: it echoes `survey_metadata.design_effect` and
`effective_n` from the fitted result along with a `band_label` enum
classifying the deviation from 1. The enum values are:

- `"improves_precision"` for `deff < 0.95` (effective N is LARGER
  than nominal N — a precision-improving design);
- `"trivial"` for `0.95 <= deff < 1.05` (effectively no effect on
  inference);
- `"slightly_reduces"` for `1.05 <= deff < 2`;
- `"materially_reduces"` for `2 <= deff < 5`;
- `"large_warning"` for `deff >= 5`;
- `None` when `deff` is missing or non-finite.

The section does not call `compute_deff_diagnostics` (that helper
needs per-fit internals the result objects do not expose). The report layer **does** compose a few
cross-period summary statistics from per-period inputs already
produced by the estimator — specifically the joint-Wald / Bonferroni
pre-trends p-value from pre-period event-study coefficients (see
`_pt_event_study`), the MDV-to-ATT ratio for power-tier selection,
and the heterogeneity dispersion block (CV / range / sign-
consistency over post-treatment group / event-study / group-time
effects, pre-period and reference-marker rows excluded). These are
reporting-layer aggregations of inputs already in the result object,
not new inference.

## Target parameter

The BusinessReport and DiagnosticReport schemas both carry a
top-level `target_parameter` block that names what scalar the
headline number actually represents. The 16 result classes have
meaningfully different estimands — a stakeholder reading
`overall_att = -0.0214` on a Callaway-Sant'Anna fit cannot tell
whether that is the simple-weighted average across `ATT(g,t)`
cells, an event-study-weighted aggregate, or a group-weighted
aggregate. Baker et al. (2025) Step 2 is "Define the target
parameter"; BR/DR does that work for the user.

Schema shape:

```json
"target_parameter": {
  "name": "overall ATT (cohort-size-weighted average of ATT(g,t))",
  "definition": "A cohort-size-weighted average of group-time ATTs ...",
  "aggregation": "simple",
  "headline_attribute": "overall_att",
  "reference": "Callaway & Sant'Anna (2021); REGISTRY.md Sec. CallawaySantAnna"
}
```

Field semantics:

- `name` — short stakeholder-facing name. Rendered verbatim in
  BR's summary paragraph and DR's overall-interpretation
  paragraph. Always non-empty.
- `definition` — plain-English description of what the scalar is
  and how it is aggregated. Rendered in BR's and DR's full-report
  markdown (under "## Target Parameter") but omitted from the
  summary paragraph so stakeholder prose stays within the 6-10-
  sentence target.
- `aggregation` — machine-readable tag dispatching agents can
  branch on. Complete enumeration per estimator:
  - `"did_or_twfe"` (DiDResults / TwoWayFixedEffects both route here — neutral tag; ambiguous at the result-class level until estimator provenance is persisted)
  - `"event_study"` (MultiPeriodDiDResults)
  - `"simple"` (CallawaySantAnna / DMLDiD / Imputation / TwoStage / Wooldridge)
  - `"iw"` (SunAbraham)
  - `"stacked"` (StackedDiD)
  - `"pt_all_combined"` / `"pt_post_single_baseline"` (EfficientDiD
    branched on `pt_assumption`)
  - `"dose_overall"` (ContinuousDiD)
  - `"ddd"` / `"staggered_ddd"` (TripleDifference / StaggeredTripleDiff)
  - `"spillover"` (SpilloverDiD — Butts 2021 total effect on the treated `tau_total`, identified off far-away controls; headline `att`)
  - dCDH dynamic branches follow the exact `overall_att`
    contract: `"M"` / `"M_x"` / `"M_fd"` / `"M_x_fd"` for
    `L_max=None`; `"DID_1"` / `"DID_1_x"` / `"DID_1_fd"` /
    `"DID_1_x_fd"` for `L_max=1`; `"delta"` / `"delta_x"` for
    `L_max>=2` without trend suppression; and
    `"no_scalar_headline"` when `trends_linear=True` AND
    `L_max>=2` (the scalar is intentionally NaN).
  - `"synthetic"` (SyntheticDiD) / `"factor_model"` (TROP) /
    `"twfe"` (BaconDecomposition read-out) / `"unknown"` (default
    fallback).
- `headline_attribute` — the raw result attribute the scalar
  comes from (`"overall_att"` / `"att"` / `"avg_att"` /
  `"twfe_estimate"`), OR `None` when `aggregation ==
  "no_scalar_headline"` (the dCDH `trends_linear=True,
  L_max>=2` branch where `overall_att` is intentionally NaN by
  design). Agents dispatching on this field must handle `None` by
  inspecting `headline.reason` (BR) / `headline_metric.reason`
  (DR), which distinguishes two subcases:

  - **Populated-surface subcase** (per-horizon
    `linear_trends_effects` dict is non-empty): `reason`
    directs callers to `results.linear_trends_effects[l]` for
    per-horizon cumulated level effects.
  - **Empty-surface subcase** (`linear_trends_effects is None`
    because no horizons survived estimation): `reason` names
    the empty state explicitly and directs callers toward
    re-fit remediation (larger `L_max` or
    `trends_linear=False`) rather than a nonexistent dict. The
    dCDH native estimand label is also branched — on this
    subcase `_estimand_label()` returns
    `DID^{fd}_l (no cumulated level effects survived estimation)`
    (or `DID^{X,fd}_l (...)` when covariates are active).

  Different result classes use different attribute names; agents
  that want to re-read the raw value can dispatch on
  `headline_attribute`.
- `reference` — one-line citation pointer to the canonical paper
  and the REGISTRY.md section.

Per-estimator dispatch lives in
`diff_diff/_reporting_helpers.py::describe_target_parameter`. Each
branch is sourced from the corresponding estimator's section in
REGISTRY.md; new result classes must add an explicit branch (the
exhaustiveness test `TestTargetParameterCoversEveryResultClass`
locks this in).

A few branches read fit-time config from the result object:

- `EfficientDiDResults.pt_assumption`: `"all"` (over-identified
  combined) vs `"post"` (just-identified single-baseline) branches
  `aggregation` between `"pt_all_combined"` and
  `"pt_post_single_baseline"`.
- `StackedDiDResults.control_group`: `"never_treated"` /
  `"strict"` / `"not_yet_treated"` varies the `definition` clause
  describing which units qualify as controls.
- `ChaisemartinDHaultfoeuilleResults.L_max` +
  `covariate_residuals` + `linear_trends_effects`: branches the
  dCDH estimand tag per the exact `overall_att` contract in
  `chaisemartin_dhaultfoeuille.py:2602-2634` and
  `chaisemartin_dhaultfoeuille.py:2828-2834`:
  - `L_max=None` → `DID_M` (Phase 1 per-period aggregate;
    `aggregation="M"`).
  - `L_max=1` → `DID_1` (single-horizon per-group estimand,
    Equation 3 of the dynamic companion paper;
    `aggregation="DID_1"`).
  - `L_max>=2` → cost-benefit `delta` (Lemma 4 cross-horizon
    aggregate; `aggregation="delta"`).
  - `trends_linear=True` AND `L_max>=2` → `overall_att` is
    intentionally NaN (no scalar aggregate; per-horizon level
    effects live on `results.linear_trends_effects[l]`).
    `aggregation="no_scalar_headline"` and
    `headline_attribute` is `None`.

  Covariates (`has_controls`) and/or linear trends
  (`has_trends`, when `L_max < 2`) add `_x` / `_fd` /
  `_x_fd` suffixes to the `aggregation` tag and the
  corresponding `^X` / `^{fd}` / `^{X,fd}` superscripts to the
  `name` (e.g. `DID^X_1`, `delta^X`, `DID^{fd}_M`), matching the
  result class's own `_estimand_label()` helper at
  `chaisemartin_dhaultfoeuille_results.py:454-490`.

A few branches emit a fixed tag regardless of fit-time config —
notably `CallawaySantAnna`, `ImputationDiD`, `TwoStageDiD`, and
`WooldridgeDiD`. For these estimators the `overall_att`
(or `att` / `avg_att`) scalar is ALWAYS the simple weighted
aggregation; post-fit `results.aggregate()` (the successor to the
deprecated fit-time `aggregate` kwarg, rows M-020..M-027) returns
the horizon / group tables without changing the headline scalar —
and `DiagnosticReport` now consumes those post-fit tables
internally for its event-study-gated checks when the raw fields are
absent. Disambiguating those tables in prose is
tracked under BR/DR gap #9 (per-cohort narrative rendering).

`ContinuousDiDResults` emits a single `"dose_overall"` tag with a
disjunctive definition (`ATT^loc` under PT; `ATT^glob` under
SPT) because the PT-vs-SPT regime is a user-level assumption, not
a library setting.

## Design deviations

- **Note:** No hard pass/fail gates. `DiagnosticReport` does not produce
  a traffic-light verdict. Severity is conveyed through natural-language
  phrasing ("robust", "fragile", "material share"). This is an explicit
  deviation from the strategy document's Gap 4 ("traffic-light
  assessment (green/yellow/red)"); the choice is motivated by the
  well-known risk of naive thresholds producing false confidence. A
  `ConservativeThresholds` opt-in layer remains available as a future
  addition if practitioner demand materialises.

- **Note:** Placebo battery is opt-in (`run_placebo=False` by default).
  `run_all_placebo_tests` on a typical panel (500 permutations times one
  DiD fit per permutation) adds tens of seconds of latency, which would
  be surprising as the default on a convenience wrapper. The schema
  reserves the `"placebo"` key; it is always rendered with
  `{"status": "skipped", "reason": "..."}` in MVP so agents parsing the
  schema see a stable shape.

- **Note:** `DiagnosticReport` does not call `check_parallel_trends` on
  event-study or staggered result objects. `check_parallel_trends` in
  `diff_diff/utils.py` assumes a single binary treatment with universal
  pre-periods; for staggered and event-study designs, DR reads the
  pre-period event-study coefficients directly off the fitted result —
  or, when the raw `event_study_effects` field is absent, off the
  internally derived post-fit `aggregate('event_study')` container,
  which carries the same coefficients — and constructs a joint
  Wald statistic (or Bonferroni fallback when `vcov` is missing). On
  the derived route the PT / pre-trends-power / sensitivity sections
  carry the additive schema key
  `pre_period_source = "aggregate_event_study"` (raw-field fits omit
  the key, mirroring the `df_denom` additive-key convention), and any
  warnings the kit recompute re-emits are captured and re-published on
  the consuming section's `warnings` list (record-and-republish; never
  swallowed). This
  mirrors the guidance in `practitioner._parallel_trends_step(staggered=True)`.

- **Note:** Survey-design threading for fit-faithful Bacon replay.
  `DiagnosticReport(survey_design=...)` and
  `BusinessReport(survey_design=...)` accept the original
  `SurveyDesign` object and forward it to
  `BaconDecomposition.fit(survey_design=...)` so the Goodman-Bacon
  decomposition is computed under the same design as the weighted
  estimate. When `survey_metadata` is set but `survey_design` is not
  supplied, Bacon skips with an explicit reason rather than replaying
  an unweighted decomposition for a design that differs from the
  weighted estimate; users can alternatively pass
  `precomputed={'bacon': ...}` with a survey-aware result.

- **Note:** Declared repeated-cross-section fits skip Bacon. Any
  panel-attribute-bearing producer fitted with `panel=False` (DMLDiD
  and CallawaySantAnna alike) skips the Goodman-Bacon check with an
  explicit reason instead of replaying: `BaconDecomposition` on
  one-observation-per-unit data is degenerate (treatment collinear
  with the absorbed unit fixed effects — it returns a meaningless
  `twfe_estimate` of 0.0 rather than raising). The gate sits AFTER the
  precomputed-Bacon passthrough, so `precomputed={'bacon': ...}`
  computed on a real panel stays honored. BusinessReport's DMLDiD
  identification narrative is likewise design-aware: RCS fits cite the
  stationary-sampling Assumption 2.3 and the Case 2 rate package.

  The simple 2x2 parallel-trends helper (`utils.check_parallel_trends`)
  has no survey-aware variant. On a survey-backed `DiDResults` the
  check is skipped **unconditionally**, regardless of whether
  `survey_design` is supplied, because the helper cannot consume the
  design even when it is available. Users must pass
  `precomputed={'parallel_trends': ...}` with a survey-aware pretest
  result to opt in. Event-study PT on staggered estimators is
  unaffected — it reads the weighted pre-period coefficients off the
  fitted result (or off its internally derived post-fit container,
  which carries the same weighted coefficients) and uses the finite-df
  reference described below, so no second replay is needed.

- **Note:** Survey finite-df PT policy. When the fitted result carries
  a finite `survey_metadata.df_survey`, `_pt_event_study` computes
  `F = W / k` (numerator df = k pre-period coefficients) against an
  F(k, df_survey) reference distribution rather than chi-square(k).
  The design-based SE already reflects the effective sample size, so
  the chi-square reference would systematically over-reject under the
  finite-sample correction the SE captures. The schema surfaces the
  survey branch via the `method` suffix `_survey`
  (e.g., `joint_wald_survey`, `joint_wald_event_study_survey`) and
  exposes the denominator df as `df_denom`, so BR / DR prose can flag
  the finite-sample correction rather than silently presenting a
  chi-square-style result. Non-finite `df_survey` (NaN / inf /
  non-positive) falls back to the chi-square path.

- **Note:** Estimator-native validation surfaces are surfaced rather
  than duplicated. `SyntheticDiDResults` routes parallel-trends to
  `pre_treatment_fit` (the RMSE of the synthetic-control fit on the
  pre-period), and routes sensitivity to `in_time_placebo()` +
  `sensitivity_to_zeta_omega()`. `TROPResults` surfaces factor-model
  diagnostics (`effective_rank`, `loocv_score`, selected `lambda_*`)
  under `estimator_native_diagnostics`. `SyntheticControlResults`
  routes parallel-trends to the `scm_fit` analogue (`pre_rmspe`,
  verdict `design_enforced_pt`) and surfaces `pre_rmspe`, donor-weight
  concentration, the in-space placebo permutation p-value, the
  ADH-2015 §4 leave-one-out (`leave_one_out`), in-time placebo
  (`in_time_placebo`), regression-weight extrapolation
  (`regression_weights`) and sparse-SC subset-search
  (`sparse_synthetic_control`) blocks, the Firpo-Possebom (2018)
  test-inversion `confidence_set`, and the Chernozhukov-Wüthrich-Zhu
  (2021) `conformal_inference` block (joint / pointwise / average) under
  `estimator_native_diagnostics` — each
  is populated only when the caller has already run the corresponding
  opt-in method (DR never triggers a refit loop implicitly; otherwise a
  `status="not_run"` stub), and it omits HonestDiD-style `sensitivity`
  (significance IS the placebo).
  `EfficientDiDResults` PT runs through `EfficientDiD.hausman_pretest`
  (the estimator's native PT-All vs PT-Post check).

- **Note:** Pre-trends verdict is a three-bin heuristic, not a field
  convention. DR maps the joint p-value as follows:

  - `joint_p >= 0.30` &rarr; `no_detected_violation`.
  - `0.05 <= joint_p < 0.30` &rarr; `some_evidence_against`.
  - `joint_p < 0.05` &rarr; `clear_violation`.

  These thresholds are diff-diff heuristics. The 0.30 upper bound draws
  on equivalence-testing intuition (Rambachan & Roth 2023 discuss the
  limitations of pre-tests). The `no_detected_violation` label
  deliberately avoids "parallel trends hold" language — the test did
  not detect a violation, but pre-trends tests are commonly
  underpowered. See the power-aware phrasing rule below.

- **Note:** Power-aware phrasing for `no_detected_violation`. DR calls
  `compute_pretrends_power(results, violation_type='linear',
  alpha=alpha, target_power=0.80)` for the estimator families that
  ship a `compute_pretrends_power` adapter: `MultiPeriodDiDResults`,
  `CallawaySantAnnaResults`, and `SunAbrahamResults` (see
  `_APPLICABILITY["pretrends_power"]` in
  `diff_diff/diagnostic_report.py`). `DMLDiDResults` reaches power via
  the DERIVED route instead: it ships no native adapter (its native
  isinstance route raises on the absent fit-time surface), and DR hands
  `compute_pretrends_power` its admitted `aggregate('event_study')`
  container (M-093 seventh amendment). Other staggered families with
  event-study output (`ImputationDiDResults`, `TwoStageDiDResults`,
  `SpilloverDiDResults`, `StackedDiDResults`, `EfficientDiDResults`,
  `StaggeredTripleDiffResults`, `WooldridgeDiDResults`,
  `ChaisemartinDHaultfoeuilleResults`) do not yet have a power
  adapter and therefore render the `no_detected_violation` tier as
  `underpowered` with the fallback reason recorded in
  `schema["pre_trends"]["power_reason"]` (plain-English explanation)
  while `schema["pre_trends"]["power_status"]` carries the
  machine-readable enum (`"ran"` / `"skipped"` / `"error"` /
  `"not_applicable"`). BusinessReport then reads
  `mdv_share_of_att = max_abs_pre_violation / abs(att)` and selects a tier.
  The numerator is the **level-scale max pre-period violation under the
  MDV**, computed as `mdv * max(|violation_weights|)` — NOT the raw `mdv`
  scalar. Post PR-B Step 4, raw `mdv` for `violation_type='linear'` is in
  Roth's γ units (a slope on relative time), so comparing it directly to
  a level-scale `|att|` would mix units on irregular pre-period grids and
  mis-tier the result. The level-scale quantity is exposed via the new
  `PreTrendsPowerResults.max_abs_pre_violation` property and the
  `DiagnosticReport.pretrends_power` block schema field of the same name.
  Tier thresholds:

  - `< 0.25` &rarr; `well_powered` &mdash; "the test has 80% power to
    detect a violation of magnitude M, which is only X% of the
    estimated effect; if a material pre-trend existed, this test would
    likely have caught it."
  - `>= 0.25 and < 1.0` &rarr; `moderately_powered` &mdash; "the test
    is informative but not definitive; see the sensitivity analysis
    below for bounded-violation guarantees."
  - `>= 1.0` &rarr; `underpowered` &mdash; "the test has limited
    power &mdash; a non-rejection does not prove the assumption. See
    the HonestDiD sensitivity analysis below for a more reliable
    signal."
  - Power analysis not runnable &rarr; fall back to `underpowered`
    phrasing; the fallback reason is recorded in
    `schema["pre_trends"]["power_reason"]` (plain-English explanation;
    `power_status` carries the enum).

  Rationale: always-hedging phrasing under-sells well-designed
  studies; always-confident phrasing over-sells underpowered ones.
  The library already ships `compute_pretrends_power()`, so using it
  is the honest default rather than hedging every non-violation.

- **Note:** Pre-period covariance routing for staggered-estimator power.
  As of the PR-B PreTrendsPower implementation audit (Roth 2022),
  `compute_pretrends_power()` consumes the full `event_study_vcov`
  sub-block when it is available — non-bootstrap CS fits
  (`staggered_results.py` populates the matrix), non-bootstrap SA
  fits (`sun_abraham.py` builds it via `W @ vcov_cohort @ W.T`), and
  admitted CS-/DMLDiD-/Stacked-sourced `aggregate('event_study')` containers
  carrying `vcov` (row M-024; StackedDiD persists its ES VCV in every
  inference mode, so Stacked containers always take this tier). The
  `PreTrendsPowerResults.covariance_source` field records the actual
  extraction path (`"full_pre_period_vcov"` vs `"diag_fallback"`), and
  the `DiagnosticReport.pretrends_power` block surfaces that label
  unchanged. There are two paths through the report layer with
  different downgrade semantics:

  - **New fits** (post-PR-B, `PreTrendsPowerResults.covariance_source`
    is populated): `DiagnosticReport` reads the persisted label
    directly. Non-bootstrap CS / SA fits report
    `"full_pre_period_vcov"` and are NOT downgraded; bootstrap /
    replicate-weight paths report `"diag_fallback"` and also pass
    through unchanged (no "available but unused" concern — the
    estimator did its best with what was available).
  - **Legacy serialized results** (pre-PR-B, no
    `covariance_source` field on the object): the report layer falls
    back to type-based inference in
    `_infer_cov_source(source_fit)`. For event-study result types
    (CS / SA / etc.) with populated `event_study_vcov`, the legacy-
    ambiguous case still emits the conservative
    `"diag_fallback_available_full_vcov_unused"` sentinel and the
    `well_powered → moderately_powered` downgrade still applies —
    because without the persisted provenance we cannot rule out that
    the stored power was computed from `diag(ses^2)` under PR-A
    semantics. For `MultiPeriodDiDResults` without
    `interaction_indices`, the legacy fallback reports
    `"diag_fallback"` (a genuine fallback, not the "available but
    unused" case, so no downgrade applies).

  Remaining `"diag_fallback"` cases on new fits — bootstrap /
  replicate-weight CS / SA, bootstrap / replicate-weight TwoStageDiD,
  plus ImputationDiD / EfficientDiD — pass through unchanged because
  nothing better is available on those result types in those modes.
  StackedDiD and TwoStageDiD now persist `event_study_vcov` (4.0
  program row M-092): StackedDiD in EVERY inference mode (its
  reported event-study SEs are always the persisted matrix's
  diagonal), TwoStageDiD on the analytical paths only (bootstrap and
  replicate-weight modes clear it). Where the covariance is present,
  the PT check takes the joint-Wald path (subject to the hc2_bm
  policy and rank guard below); on the derived route the same
  covariance arrives via the container's `vcov` / `vcov_index`
  fields. Pretrends POWER: natively
  `compute_pretrends_power()` supports MPD / CS / SunAbraham fits;
  since row M-024 a Stacked `results.aggregate('event_study')`
  CONTAINER also admits (kappa_pre >= 2). On plain (no fit-time
  `aggregate=`) CS fits, `DiagnosticReport` internally derives the CS
  container and feeds it to `compute_pretrends_power` /
  `HonestDiD.sensitivity_analysis` — CS is M-093-admitted with pinned
  raw-route parity; the admission set itself is unchanged (widening
  is a per-estimator methodology decision, ledger row M-093).
  Stacked/TwoStage NATIVE
  results remain outside DR's power applicability - within
  DiagnosticReport their covariance is consumed by the PT check and
  by the PRECOMPUTED-power provenance classifier only (a stored power
  result on a fit that now exposes a full VCV is labelled
  `"diag_fallback_available_full_vcov_unused"`).

- **Note:** hc2_bm parallel-trends policy (deviation by omission,
  documented). When the source fit's `vcov_type` is `"hc2_bm"`, the
  PT check does NOT run the covariance-based joint Wald even when a
  full event-study covariance is persisted: the generic chi-square /
  F(k, df_survey) reference ignores the CR2 / Bell-McCaffrey
  small-sample correction the estimator's own per-row inference
  applies, and the proper multi-constraint CR2 test (AHT/HTZ,
  `clubSandwich::Wald_test(test="HTZ")`) is not implemented (tracked
  in DEFERRED.md). The check instead runs Bonferroni over the
  BM-adjusted per-row p-values, preserving the small-sample
  correction. Additionally, the joint-Wald path fail-closes on
  provenance: if ANY retained pre-period row carries a non-finite
  p-value (hc2_bm BM-DOF failure, collapsed replicate-survey df), the
  check returns `method="inconclusive"` rather than computing a Wald
  statistic through effects whose inference the estimator refused to
  produce — the covariance-backed twin of the round-33/42 Bonferroni
  guards. Finally, the joint-Wald solve is rank-guarded: a
  cluster-sandwich covariance has rank bounded by the cluster-score
  dimension, so when the number of tested leads approaches the number
  of clusters the pre-period block is singular and `β′V⁻¹β` is a
  garbage (possibly negative) quadratic form. The check requires a
  finite, numerically full-rank, PSD block and a non-negative finite
  statistic; otherwise it falls back to Bonferroni over the per-row
  p-values (a documented conservative downgrade, not a defect).

- **Note:** Unit-translation policy. BusinessReport does not
  arithmetically translate log-points to percents or level effects to
  log-points. The estimate is rendered in the scale the estimator
  produced; `outcome_unit="log_points"` emits an informational
  caveat. The policy avoids guessing the underlying model (no
  estimator in the library currently exports both log and level
  coefficients), which would be unsafe in the presence of non-linear
  link functions (Poisson QMLE, logit).

- **Note:** Single-knob `alpha` with preserved-native-CI fallback.
  BusinessReport exposes only `alpha` (defaults to `results.alpha`);
  there is no separate `significance_threshold` parameter. When the
  requested `alpha` matches the fit's native level, it drives both the
  CI level (`(1 - alpha) * 100`% interval) and the phrasing tier
  threshold ("statistically significant at the (1 - alpha) * 100%
  level"). When the requested `alpha` differs from the fit's native
  level (e.g., the user asks for `alpha=0.10` on a result fit with
  `alpha=0.05`), BusinessReport does NOT recompute the CI at the
  requested level, because the stored CI is the only quantile the
  underlying estimator supplied (bootstrap distributions and
  finite-df analytical variances are not always retained on the
  result). Instead, the schema preserves the fit's native CI (with its
  original level) and uses the requested `alpha` only for the
  significance-phrasing threshold, and emits an
  `alpha_override_preserved` caveat describing the mismatch. This is
  the conservative choice: it avoids silently recomputing CIs under
  assumptions the estimator may not support.

- **Note:** Schema stability policy for the AI-legible `to_dict()`
  surface. New top-level keys count as additive (no version bump); new
  values in any `status` enum count as breaking (agents doing
  exhaustive pattern match will break on unknown enums); renames and
  removals count as breaking. The `BUSINESS_REPORT_SCHEMA_VERSION`
  and `DIAGNOSTIC_REPORT_SCHEMA_VERSION` constants bump independently.
  The v3.2 CHANGELOG marks both schemas experimental so users do not
  anchor tooling on them prematurely; a formal deprecation policy will
  land within two subsequent PRs.

- **Note:** Schema version 2.0 (both BR and DR). The BR/DR gap #6
  target-parameter PR adds the `headline.status` /
  `headline_metric.status` value `"no_scalar_by_design"` (used for
  the dCDH `trends_linear=True, L_max>=2` configuration where
  `overall_att` is intentionally NaN). Per the stability policy
  above, new enum values are breaking changes, so
  `BUSINESS_REPORT_SCHEMA_VERSION` and
  `DIAGNOSTIC_REPORT_SCHEMA_VERSION` bumped from `"1.0"` to
  `"2.0"`. The schemas remain marked experimental, so the formal
  deprecation policy does not yet apply.

- **Note:** `headline.ci_level` carries the EXACT confidence level in
  percent: an `int` for integral coverage (`95` at `alpha=0.05`, byte-
  identical to the historical value) and a `float` for fractional coverage
  (`97.5` at `alpha=0.025`, previously rounded to `98`). No version bump:
  the key, its meaning, and every integral value are unchanged; only
  previously misreported fractional levels change, and they change type
  rather than enum membership. Every text surface (summary headers,
  event-study table headers, sup-t band labels, report prose) derives its
  percent from the same `results_base._coverage_level` / `_coverage_pct`
  helpers, so display and schema cannot disagree.

## Reference implementation(s)

The phrasing rules follow the guidance in:

- Baker, A. C., Callaway, B., Cunningham, S., Goodman-Bacon, A., &
  Sant'Anna, P. H. C. (2025). *Difference-in-Differences Designs: A
  Practitioner's Guide.* (The 8-step workflow enforced through
  `diff_diff/practitioner.py`.)
- Rambachan, A., & Roth, J. (2023). *A More Credible Approach to
  Parallel Trends.* Review of Economic Studies. (HonestDiD sensitivity;
  the pre-test power caveat directly shaped the three-tier power
  phrasing.)
- Roth, J. (2022). *Pretest with Caution: Event-study Estimates after
  Testing for Parallel Trends.* American Economic Review: Insights.
  (Motivates the power-aware phrasing tiers.)
