# diff-diff 4.0 Design Specification

**Status: LIVING - normative until the 4.0 cut.**

**Review contract:** every Phase 2-5 PR of the 4.0 program is reviewed against
this document. A PR that deviates from it must edit this document (and the
matrix, `docs/v4-deprecations.yaml`) in the same diff - deviation-by-silence is
a review reject.

**Forward-reference disclaimer:** symbols described here as 3.9/4.0 surface
intentionally do NOT exist at HEAD. `tests/test_v4_matrix.py` enforces their
absence until their phase ships, and flips to enforcing their presence when the
matrix row flips. Nothing in this document is an undocumented deviation; it is
the documentation.

Companion artifacts: `docs/v4-deprecations.yaml` (the lifecycle ledger - single
source of truth for every old/new/version/status fact; this prose never
restates per-row lifecycle data and instead cites rows as `[M-###]`),
`tests/test_v4_matrix.py` (enforcement), ROADMAP.md "4.0 API unification"
(program pointer).

All decisions below were locked with the maintainer on 2026-07-18 (six program
decisions + eight naming-checkpoint decisions), with dated addenda: 2026-07-19
(the diagnostic-family decision, section 3.5) and 2026-07-31 (the four
consolidation-scope decisions appended to the section 8 checkpoint list).
Sections 3-8 are the target surface; section 9 maps it onto PRs.

---

## 1. Goals / Non-goals

**Goals.** (i) Three estimator merges: TwoWayFixedEffects absorbs
MultiPeriodDiD [M-010], TripleDifference absorbs StaggeredTripleDifference as
a facade [M-013], ChangesInChanges absorbs QDiD [M-015]. (ii) One contract
across everything that stays separate: column vocabulary (section 8), results
quintet (section 5), aggregation (section 6), inference surface (section 7),
alias table (section 3). (iii) Every deprecation queued anywhere in the repo
lands or is explicitly re-scheduled at the 4.0 cut - tracked by the matrix,
enforced by CI.

**Non-goals.** No numerical behavior changes in any phase except the two
scheduled default policies ([M-004]..[M-006] + [M-128]..[M-131]
df_convention, [M-080] auto-cluster), the ledgered 3.9 defect fixes of the
variance-consolidation program ([M-126] K_reference SEs, [M-127] tail-df
consolidation), and the documented estimate/SE shift of the MPD merge
(section 4.1). No
new estimators. No merging across identification strategies: the staggered
family (CallawaySantAnna, SunAbraham, ImputationDiD, TwoStageDiD, EfficientDiD,
WooldridgeDiD, StackedDiD, LPDiD) stays separate classes - ecosystem-wide,
method switches succeed only when estimators share an estimand and input
contract, and these differ structurally (ImputationDiD/TwoStageDiD have no
control-group concept at all). SyntheticControl and SyntheticDiD stay separate
(disjoint weight machinery). The 4.1 `event_study()` comparison front door is
sketched in section 9 but not specified here.

## 2. Release-line rule and warning policy

**Ladder:** 3.9 ships the new surface additively plus FutureWarnings on old
surface (sklearn playbook) -> 4.0 removes deprecated surface, flips results
storage (section 5) and the two default policies -> 4.1 adds the front door ->
5.0 removes the 4.0-era compatibility properties. Last fully-old-API release:
3.8.0 (tag `v3.8.0`).

**maint/3.8 branch rule:** 3.8.x patches are cut from main until the first
Phase 2 PR merges; after that, main carries minor-worthy content, so further
3.8.x patches come from a `maint/3.8` branch cut at tag `v3.8.0` (fix lands on
main, cherry-pick back).

**Warning categories:** all NEW shims emit `FutureWarning` (visible to end
users by default). Pre-existing `DeprecationWarning` sites keep their category
([M-001] [M-002] [M-003]) - churn-free.

**Per-PR gates.** Every shim PR ships a dedicated behavioral test file
asserting (a) `pytest.warns` on the old surface with the migration message, and
(b) bit-exact routing parity: the deprecated path routes to the same numbers as
before the PR. Every removal PR follows the
`tests/test_had_dual_knob_deprecation.py` pattern: canonical-surface positive
smoke + `TypeError`/`AttributeError` removal pin per surface. Matrix rows flip
in the same diff; the PR's CHANGELOG entry names the flipped row ids.

## 3. Target 4.0 surface

### 3.1 Estimator roster (fates)

Derived from `diff_diff.__all__` at v3.8.0. 24 estimator classes -> 21.

| Class | Fate |
|---|---|
| DifferenceInDifferences | Keep (2x2; `post` contract, section 8) |
| TwoWayFixedEffects | Keep + absorbs MultiPeriodDiD [M-010] |
| MultiPeriodDiD | Removed 4.0 [M-010] (results class [M-011], PeriodEffect [M-012]) |
| CallawaySantAnna, SunAbraham, ImputationDiD, TwoStageDiD, EfficientDiD, WooldridgeDiD, StackedDiD, LPDiD | Keep (staggered family - separate by design) |
| TripleDifference | Keep + absorbs StaggeredTripleDifference [M-013] |
| StaggeredTripleDifference | Removed 4.0 [M-013] (results class [M-014]) |
| ChangesInChanges | Keep + absorbs QDiD via `method=` [M-015] |
| QDiD | Removed 4.0 [M-015] |
| SyntheticDiD, SyntheticControl, TROP, ContinuousDiD, HeterogeneousAdoptionDiD, RegressionDiscontinuity, SpilloverDiD, ChaisemartinDHaultfoeuille | Keep |
| BaconDecomposition | Keep - moves into the diagnostic family (section 3.5) |

### 3.2 Final alias table

Kept aliases (14, the literature-standard names): DiD, TWFE, SDiD, CS, SA,
BJS, DDD, Bacon, EDiD, ETWFE, DCDH, CiC, HAD, RDD (EDiD was initially slated
for the diet but RETAINED on review evidence - it is the Chen-Sant'Anna-Xie
paper's own label for its estimator, the same criterion that keeps SDiD). Changes:
`EventStudy` is dropped, not retargeted [M-060] - "event study" names a design
that CS/SA/BJS/LPDiD also produce, and retargeting it to a class whose default
mode is the static ATT would make the name carry altered meaning. `SDDD` dies
with its class [M-013] [M-064]. `QDiDResults` dies with QDiD [M-061].
**Alias diet (2026-07-31; shipped in 3.9 by 2(d) PR-A):** `CDiD`,
`Stacked` and `Gardner` are
retired — [M-132]..[M-134] — deprecated 3.9, removed 4.0. Their target classes
survive, so no parent-class shim can carry the warning; the 3.9 FutureWarning
rides a module-level `__getattr__` [M-135]. The 3.9 consequence: the three
names leave module globals (gone from `dir()` and static autocomplete) while
staying importable and in `__all__`, and `from diff_diff import *` fires their
FutureWarnings — accepted diet behavior (the package import protocol
resolves each missing `__all__` name twice, so the star-import records
six warnings for the three aliases; the test asserts the message set).
New: `SCM` for SyntheticControl [M-062] ("SC" rejected - one
transposition from CS). The planned `Spillover` alias for SpilloverDiD is CANCELLED [M-063]
(never shipped). TROP and LPDiD are self-aliased acronyms.
After 4.0: every estimator has exactly one class name and at most one alias —
15 aliases total.

### 3.3 Module-level function wrappers

The 8 estimator wrapper functions are deprecated in 3.9 (shipped by 2(d)
PR-A: FutureWarning shims + docstring notes) and removed in 4.0
([M-070]..[M-077]): classes are the single canonical construction surface.
`twowayfeweights` stays (diagnostic function, not a class duplicate);
`compute_honest_did`, placebo/power helpers, and dataset loaders are
unaffected (not estimator duplicates).

### 3.4 Canonical signatures for renamed surfaces

Only surfaces with matrix rows are listed; unchanged estimators keep their
signatures. Lifecycle facts live in the cited rows.

- `DifferenceInDifferences.fit(data, outcome=None, treatment=None, post=None,
  formula=None, covariates=None, fixed_effects=None, absorb=None,
  survey_design=None, unit=None)` - `time` -> `post` [M-030].
- `TwoWayFixedEffects.fit(..., post=...)` in static mode - `time` -> `post`
  [M-082], with the 4.0 static-mode `time=` rejection [M-083]; section 4.1.
- `ContinuousDiD.fit(..., covariates=...)` - `covariates` moves from
  `__init__` to `fit()` [M-084] (the sklearn hyperparam/data split).
- `TripleDifference.fit(...)` - section 4.2; `time` -> `post` in 2x2x2 mode
  [M-031], with the NAME persisting as the staggered calendar column and the
  4.0 2x2x2-mode `time=` rejection [M-085] (mirrors the TWFE pair).
- `WooldridgeDiD.fit(..., first_treat=...)` - `cohort` -> `first_treat`
  [M-032]. The `exovar`/`xtvar`/`xgvar` covariate split is KEPT intentionally
  (the three-way split is methodologically meaningful: which covariates enter
  which interaction sets); documented as domain vocabulary under the section 8
  rules.
- `ChaisemartinDHaultfoeuille.fit(..., unit=..., covariates=...)` - `group` ->
  `unit` [M-033], `controls` -> `covariates` [M-034].
- `HeterogeneousAdoptionDiD.fit(outcome=, dose=, time=, unit=, first_treat=,
  ...)` - `_col` suffixes dropped [M-035]..[M-039].
- `RegressionDiscontinuity.fit(outcome=, running=, takeup=, ...)` -
  [M-040]..[M-042]; `takeup` rather than `treatment` because the column
  accepts non-binary dose take-up (rule 7's corollary, section 8), with the
  results-field mirror at [M-094].
- `StackedDiD(control_group=...)` - `clean_control` renamed [M-043].
- `WooldridgeDiDResults.to_dataframe(level=...)` - `aggregation` -> `level`
  [M-044]; the `"event"` value spelling unifies across its existing
  `aggregate(type=)` surface [M-086] and `summary(aggregation=)` is retired
  for the uniform `summary(alpha=None)` [M-087]. Transitional 3.9 shape
  (shipped by the 2(c)-ii PR-B): `summary(aggregation=SENTINEL, *,
  alpha=None)` - `alpha` is KEYWORD-ONLY while the deprecated `aggregation`
  still holds position 1 (a positional float raises a pointed TypeError);
  the uniform positional `summary(alpha=None)` arrives with the 4.0
  removal.
- `robust` constructor param dropped everywhere it exists
  [M-045]..[M-047] [M-115] - fully redundant with `vcov_type`, and its default
  even differed across estimators (True/True/False). Four sites, not three:
  [M-115] adds `LinearRegression`, which owns its `__init__`; the
  TwoWayFixedEffects / MultiPeriodDiD occurrences inherit
  `DifferenceInDifferences.__init__` and ride [M-045].

### 3.5 The diagnostic family

diff-diff 4.0 formalizes a THIRD object kind alongside estimators and
results: **diagnostics** (locked with the maintainer 2026-07-19). Exactly
ONE bit is load-bearing: an estimator's result carries a causal-effect
inference row (the section 5 quintet); a diagnostic's result does NOT -
it assesses a design, an identifying assumption, or robustness (a
decomposition table, binned plot data, a pre-trends p-value, sensitivity
bounds, a power curve).

**Membership: a NEW CANONICAL CONSOLIDATION anchored on the
"Diagnostics & Sensitivity" docs family** (which today is close but not
identical - the consolidation is the source of truth from 4.0):
BaconDecomposition (reclassified OUT of the README/API estimator lists),
RDPlot (consolidated from the RD grouping), parallel-trends testing, the
placebo suite, HonestDiD, PreTrendsPower, PowerAnalysis,
`twowayfeweights`, the HAD pretests, DiagnosticReport's results, and the
upcoming CJM 2020 density test (born into the family). Explicit
NON-members despite current docs placement: Conley spatial-HAC (an
embedded inference method on estimators, not a diagnostic object) and
other design elements listed under the README section. (A narrower
classes-only tier with separate categories for HonestDiD/PowerAnalysis
was drafted and REJECTED 2026-07-19: the boundary needed adjudication,
and the family line above is the decided one.)

**Mechanics - the marker lives on RESULT containers.** Consumers hold
results, not entry points, so the `Diagnostic` marker base (shipping
ADDITIVELY in Phase 2 [M-091]) attaches to the class-backed diagnostic
RESULT types: `BaconDecompositionResults`, `RDPlotResult`,
`HonestDiDResults` + `SensitivityResults`, `PreTrendsPowerResults` +
`PreTrendsPowerCurve`, `PowerResults` + the `Simulation*Results` family,
`PlaceboTestResults`, the HAD pretest containers (`QUGTestResults`,
`StuteTestResults`, `YatchewTestResults`, `StuteJointResult`,
`HADPretestReport` - all already exposing the serialization pair), and
`DiagnosticReportResults` (whose `summary()`/`to_dataframe()` live on the
BUILDER today - Phase 2 moves or delegates them onto the container to meet
the contract; BusinessReport exports no result container, so nothing to
mark there) - plus every future diagnostic's result type (the density
test's first). Entry-point classes are NOT marked (one
marker, one meaning). Contract: marked results expose `summary()` /
`to_dataframe()` and are exempt from the section 5 quintet BY TYPE
(isinstance-checkable, so BusinessReport / DiagnosticReport /
practitioner routing stop special-casing by result-class name). The
family's entry points keep their existing shapes - data-in (Bacon,
RDPlot) and results-in (HonestDiD, PreTrendsPower) both belong; the
marker does not constrain them. NARROWING: function-shaped members that
return raw dicts (`check_parallel_trends` and variants,
`equivalence_test_trends`) and `TWFEWeightsResult` (no
`summary()`/`to_dataframe()` today) participate in the DOCS family only,
not the type contract; upgrading them to marked containers is optional
Phase 2 follow-up work, not part of this contract. The Phase 2 gate
[M-091] requires a dedicated roster test AND consumer propagation: every
enumerated result type is `isinstance(result, Diagnostic)` and exposes
the serialization pair, representative ESTIMATOR results are NOT, and the
CONSUMERS actually switch to the marker - BusinessReport rejects marked
diagnostics as its primary estimator input by type (today it
special-cases only Bacon by name), `practitioner_next_steps()` routes
marked diagnostics through diagnostic-specific handling instead of its
unknown-result estimator fallback, and DiagnosticReport itself routes by
the marker: at least one non-Bacon marked result is handled as a
diagnostic (never through estimator fallback) while Bacon's existing
read-out behavior is retained and tested. Import paths do NOT move - the
flat top-level namespace is kept (rejected: `diff_diff.diagnostics.*`
moves). Zero new classes beyond the single marker.

**4.0 reorganization [M-090].** At 4.0 the public surface claims split
into "estimators + diagnostics": Bacon is re-homed OUT of the API
estimator roster (where it is misfiled today); RDPlot - already
documented under Diagnostics & Sensitivity - is consolidated under the
unified family grouping; README catalog, llms.txt, API reference and
alias-table groupings, and estimator-count claims on JOSS-adjacent
surfaces all split accordingly. Tracked as a behavior row so the cut
sweep cannot miss it.

**RDPlot contract compliance [M-088] [M-089].** `RDPlot.fit()` shipped
with `outcome_col` / `running_col` after the ledger froze - the section 8
missed-rename clause applies, and the params migrate to bare `outcome` /
`running` on the standard 3.9-shim / 4.0-removal path (matching
RegressionDiscontinuity's [M-040] [M-041]).

## 4. The three merges

### 4.1 TwoWayFixedEffects absorbs MultiPeriodDiD [M-010]

**Target API.**

```python
TWFE().fit(df, outcome, treatment, post, unit)            # static ATT
                                                          # (post = 0/1 dummy)
TWFE().fit(df, outcome, treatment, time="period",
           unit="id", event_study=True,
           post_periods=[3, 4, 5])                        # dynamic mode
                                                          # (time = calendar)
TWFE().fit(df, outcome, treatment, time="period",
           event_study=True, spec="pooled",
           post_periods=[3, 4, 5])                        # old MPD model;
                                                          # repeated cross-sections
```

(Amended 2026-08-07, Phase 3(a) review: the event-study calls pass
`time=`/`unit=` as KEYWORDS. Signature slot 4 belongs to `post` for the
whole M-082 shim window and beyond - the same slot cannot carry the
static dummy and the calendar column - so the calendar `time=` lives at
the signature tail, exactly like the M-031 merged staggered interface. A
positional 4th argument under `event_study=True` lands in `post` and is
rejected with a message steering to `time=`. Second amendment, same
review cycle: `post_periods=` is REQUIRED (non-empty) in event-study
mode. The treatment boundary is not observable from the documented
time-invariant ever-treated indicator, so MultiPeriodDiD's midpoint
default - last half of the calendar - is a silent guess; the merged mode
fails loud instead, consistent with its day-one wild raise and
no-legacy-defaults posture. MPD itself keeps the midpoint default
through 3.9.)

**The static `time`/`post` contract [M-082] [M-083].** Today's static TWFE
takes its 0/1 post dummy in a param NAMED `time` (the code and REGISTRY warn
on >2 unique values) - the same overload section 8 rule 1 abolishes. The
merged class resolves it without silent reinterpretation: static mode takes
`post` (the semantics of today's `time` argument, renamed [M-082]);
event-study mode takes calendar `time`. In 3.9, static `time=` still works
with a FutureWarning steering to `post=`. At 4.0, `time` means calendar ONLY,
and static mode (`event_study=False`) passing `time=` raises a `ValueError`
pointing at `post=` [M-083] - the name is never quietly re-meant.

Event-study mode params: `event_study: bool = False`, `spec: str = "within"`
(`"within"` = unit + time FE; `"pooled"` = treatment-group dummy + period
dummies, no unit FE - the only spec that works without a unit id),
`reference_period=None` (default: last pre-period, e=-1 convention - no
transition warning; the legacy-default FutureWarning dies with MPD [M-007]),
`post_periods=None`.

**Routing semantics.** `event_study=False` with `post=` is numerically
exactly today's TWFE. With `event_study=True`, `spec="within"` estimates the
unit-FE event study (`alpha_i + gamma_t + sum_e delta_e * D_i x 1[t=e]`);
`spec="pooled"` reproduces the MPD design matrix. There is NO auto-detection
from the presence of `unit` - spec is always explicit (rejected: silent
estimand switches). `spec="within"` without `unit` is an error.

**Documented estimate shift.** MPD's default spec had no unit FE; the merged
default does. Point estimates coincide ONLY in the restricted equivalence
case - balanced panel, no covariates, simultaneous adoption (REGISTRY section
MultiPeriodDiD); with unbalanced panels or covariates, the unit-FE projection
changes point estimates too, and standard errors shift in general. 3.x MPD
numbers are reproduced exactly via `spec="pooled"` - that, not "only SEs
move", is the migration message. Phase 3 test obligations: (a) balanced
no-covariate equivalence test (`within` == `pooled` point estimates), (b) an
unbalanced-or-covariate divergence test (`within` != `pooled`, locking the
documented behavior change), (c) `spec="pooled"` bit-exact parity vs 3.x MPD.
Migration guide gets a worked example of both specs.

**Inference.** The merged class carries TWFE's inference stack: auto-cluster
at unit (section 7), wild bootstrap for the static mode. Wild bootstrap in
event-study mode raises an explicit `ValueError` (MPD's current silent
analytical fallback violates the no-silent-failures principle); porting it is
backlog, not scope. (Amended 2026-08-07 with the Phase 3(a) ship, precision:
the raise is live from the mode's 3.9 BIRTH - a new surface needs no
deprecation window - and the original "at 4.0" phrasing described when MPD's
fallback dies with the class.) Auto-cluster decision (user-approved
2026-08-07): the event-study mode auto-clusters at unit FROM 3.9 - new API
adopts the section-7 end-state immediately, so [M-080]'s 4.0 flip never
re-touches it - WITH static TWFE's carve-outs mirrored lane-for-lane: the
auto-cluster is silently dropped on the Conley path (an implicit spatial x
unit product kernel would zero every between-unit pair), never injected as a
survey PSU (the documented implicit-per-observation-PSU rule), and dropped
for explicit one-way analytical families; explicit `cluster=` always passes
through and behaves exactly like MPD's own explicit cluster on every lane.
The pooled-parity gate therefore pins bit-exactness under MATCHED cluster
settings and unconditionally in the no-unit repeated-cross-sections form.

**Results.** 3.9's `TWFE(event_study=True)` returns the unified event-study
surface (section 5) from day one - no intermediate container churn.
`period_effects` becomes a FutureWarning property view over that surface at
4.0, removed 5.0 [M-016]. The HonestDiD / PreTrendsPower integrations (which
read `interaction_indices` off MultiPeriodDiDResults) are ported to the
unified surface in the same Phase 3 PR.

**Deprecation choreography.** 3.9: `event_study=`/`spec=` ship on TWFE;
`MultiPeriodDiD.__init__` emits FutureWarning; `EventStudy` alias warns
[M-060]. 4.0: MultiPeriodDiD, MultiPeriodDiDResults, PeriodEffect, EventStudy
removed [M-010] [M-011] [M-012] [M-060]. (Shipped 3.9, Phase 3(a): the mode,
the shim, the alias warning riding it, the `time=`->`post=` rename [M-082],
and the consumer ports - see the [M-010] ledger notes for the shipped
mechanics and the test triple in tests/test_v4_merge_mpd.py.)

### 4.2 TripleDifference absorbs StaggeredTripleDifference [M-013]

**Target API.**

```python
ddd = TripleDifference()
ddd.fit(df, outcome, group, partition, post)              # 2x2x2 (RC engine)
ddd.fit(df, outcome, unit, time, first_treat, partition,  # staggered (panel engine)
        covariates=None, ...)
```

**Routing semantics.** One class, two engines, both internally UNCHANGED in
this program (R-parity preserved; engine unification is a possible 4.x
internal refactor, explicitly out of scope). Dispatch is by signature shape:
the 2x2x2 engine serves the `(group, partition, post)` parameterization; the
staggered engine serves `(unit, time, first_treat, partition)`. Mixing the
two parameter sets is an error, not a guess. This mirrors the reference
implementation - Ortiz-Villavicencio & Sant'Anna's own `triplediff::ddd()`
serves both designs from one signature. The third dimension is `partition` in
both modes (the paper's and R package's vocabulary); the staggered engine's
`eligibility` name dies with its class.

**Inference note.** The two engines keep their existing inference stacks
(analytical influence-function SEs on the 2x2x2 engine; multiplier bootstrap +
GMM weighting on the staggered engine). `cluster=` analytical SEs remain
staggered-mode-unsupported (bootstrap required), as today - documented in the
class docstring, tracked as post-4.0 backlog.

**Results.** One results shape: the staggered container's structure with the
2x2x2 case as the degenerate single-ATT view (canonical quintet always
populated; group-time table empty in 2x2x2 mode). StaggeredTripleDiffResults
dies [M-014].

**Deprecation choreography.** 3.9: staggered params ship on TripleDifference;
`time` -> `post` in 2x2x2 mode [M-031] (the `time` NAME persists as the
staggered calendar column - same contract as TWFE's [M-082]);
StaggeredTripleDifference warns. 4.0: StaggeredTripleDifference + results +
SDDD alias removed [M-013] [M-014] [M-064]; 2x2x2 mode passing `time=`
raises with `post=` guidance [M-085], never silently reinterpreted.

### 4.3 ChangesInChanges absorbs QDiD [M-015]

**Target API.** `ChangesInChanges(method="cic")` with `method="qdid"` for the
quantile-DiD comparison estimator. The default encodes Athey & Imbens'
recommendation of CiC over QDiD. Internally these are already one dispatcher,
one bootstrap machinery, one results class - the merge is API-only. 3.9:
`method=` ships; QDiD class warns. 4.0: QDiD + QDiDResults export alias
removed [M-015] [M-061].

## 5. Results contract

**Canonical quintet.** `att`, `se`, `t_stat`, `p_value`, `conf_int` - bound to
ONE coherent inference row (locked library principle: uniform names never
carry altered meaning). The quintet contract applies to ESTIMATOR results;
diagnostic-family results (section 3.5) are exempt BY TYPE - they expose
`summary()` / `to_dataframe()` but no inference row. As of v3.8.0 every
ESTIMATOR result class already exposes the full quintet as properties
(verified 2026-07-18), so 3.9 needs no additive property work. At 4.0 the STORAGE flips [M-050]..[M-058]: the canonical names
become the native dataclass fields; `overall_att` (and sibling `overall_*`
inference fields, plus ContinuousDiD's deviant `overall_att_*` family
[M-058]) become FutureWarning properties, removed 5.0. MultiPeriodDiD's
`avg_*` family dies with its class [M-011] instead of flipping.

**Serialization policy.** `to_dict()` / `to_dataframe()` emit canonical names
only from 4.0 (deprecated names never leak into serialized output; the
warning properties are attribute-access-only). `to_dataframe`'s selector is
`level=` everywhere [M-044]; `summary(alpha=None)` is the uniform signature.
Every main results class supports `summary()`, `to_dict()`, `to_dataframe()` -
the seven classes currently missing `to_dict()` gain it in Phase 2
(additive).

**Unified event-study representation.** ONE representation for event-study
effects across all estimators (today the fourteen event-study producers use
four incompatible native container shapes). The `EventStudyResults`
container plus its per-producer builders ship additively in Phase 2 [M-092]
(the builder is package-internal; public exposure rides
`aggregate(type="event_study")` in the same phase, and the merged TWFE returns
it natively in Phase 3). This document pins the requirements it satisfies:
per-event-time canonical quintet rows, explicit reference-period marking (via
an `is_reference` column - no sentinel-value conventions; the
n_groups==0 / n_obs==0 sentinels are retired at 4.0 [M-093]), the event-study
vcov exposed uniformly where computed, per-row inference-df provenance (the
df each stored p-value/CI actually used, threaded via the producers'
`event_study_df` channels), producer-provenance fields for the container
consumers (`base_period`, `anticipation`, the scalar `df_survey`
channel beside the per-row one - the per-row column cannot encode the
replicate-undefined 0.0 sentinel - and `reference_event_times`, the
distinct per-cohort positional-base event times whose multi-entry case
fails the HonestDiD/PreTrendsPower common-reference guard on gapped
universal grids; amended into [M-092] pre-cut with the
2(b) consumer delivery), and `to_dataframe(level="event_study")`
emitting identical column schemas from every estimator. The container
OWNS its arrays: `__post_init__` copies every array field including the
`vcov`/`vcov_index` pair (amended with [M-024] - `np.asarray` aliased the
producer's stored matrix on the post-fit view route, so mutating a
container could corrupt the fitted result; `vcov_index` keeps its native
dtype, int labels never become floats).

**Pickle migration.** Renamed-field classes ship `__setstate__` migration
following the existing `SyntheticDiDResults.__setstate__` precedent
(diff_diff/results.py) so 3.x pickles load under 4.0.

## 6. Aggregation contract: `results.aggregate(type=...)`

**Pattern.** Estimate once, aggregate as a post-fit step - the ecosystem's
strongest norm (`did::aggte`, `etwfe::emfx`, Stata `estat aggregation`).
`fit(aggregate=)` is deprecated in 3.9 and removed in 4.0
([M-020]..[M-027]); `balance_e` moves to `aggregate()` alongside it.

**Vocabulary.** Closed set: `"simple"`, `"event_study"`, `"group"`,
`"calendar"`, plus per-estimator documented extras where the estimand demands
them (ContinuousDiD adds `"dose"` [M-025]; HAD's `"overall"` maps to
`"simple"` [M-027] - its fit-time mode selector, and the workflow twin
`did_had_pretest_workflow(aggregate=)` [M-139], both resolve by
panel-shape inference; Wooldridge's `"gt"` group-time table stays as a
documented extra). The drifted spellings die across ALL their surfaces:
`"eventstudy"` (ContinuousDiD [M-025]); Wooldridge's `"event"` on its
EXISTING post-fit `aggregate(type=)` - the emfx-style prior art for this
section's pattern - plus `summary(aggregation=)` and
`to_dataframe(aggregation=)` [M-044] [M-086] [M-087].

**Heterogeneous-`target` rendering** (added with [M-025], the first
producer of a mixed-target `AggregationResult` - ContinuousDiD's att/acrt
dual estimand; widened with [M-027]: the same target-column + neutral
`estimate`-heading rendering also fires when the SINGLE distinct target
is not `"att"` - HAD's WAS/WAS_d_lower and dCDH's estimand-labelled
relays previously rendered under the hard-coded `ATT` heading, a
mislabel; the target column sizes to the longest label so
uniform-`att` and att/acrt producers stay byte-stable): when a
container carries more than one distinct `target`,
`summary()` renders a `target` column and a neutral `estimate` heading
(the uniform-target `ATT` heading would mislabel the other estimand's
rows), and `to_dataframe()` orders rows by FIRST-APPEARANCE target blocks
(producer order - NOT lexicographic, which would invert att/acrt) with
labels ascending within each block under the same `_sortable` guard as
the uniform path (mixed-type labels keep producer order, never raise).
Uniform-target containers render byte-identically to the pre-[M-025]
output; the machine surfaces (`to_dataframe` columns, `to_dict`) carried
`target` from birth and are unchanged.

**Semantics.** `aggregate()` re-aggregates WITHOUT refitting, from influence
functions retained on the results object. **Correction (Phase 2b PR 1):** this
section previously said "CallawaySantAnna already stores them". It does not -
`CallawaySantAnnaResults.influence_functions` was declared and never assigned,
and the fit-time `precomputed` bookkeeping is a local. Retention is therefore
work in EVERY migrating PR, not just the ones "where missing", and the kit must
be built during `fit()` because nothing it needs survives the call. Memory cost
is documented per estimator, enumerating every retained buffer rather than only
the largest: for CallawaySantAnna the dominant payload is the per-(g,t)
influence-function dict at roughly O(n_units x n_gt), on top of the O(n_units)
bookkeeping; a replicate-weight survey design retains a further
O(n_units x n_replicates) matrix through the resolved design object; and on
repeated cross-sections several bookkeeping arrays are observation-length
rather than unit-length, so the O(n_units) figure is a panel-only bound. Raw
unit identifiers are NOT retained - the kit needs only position, so it stores
canonical 0..n-1 codes, keeping a shared results artifact free of names, emails
or administrative IDs. Analytical-vs-bootstrap inference of the aggregated
estimand follows the fit's inference method, and the bootstrap gate is
PER-LEVEL (converged with [M-027] across CS/EDiD/Imputation/TwoStage,
which previously failed closed on every level): a level that RELAYS the
fit's stored inference verbatim (`'simple'` on every adopter; all of
dCDH/StackedDiD/HAD) is faithful under any inference regime and stays
available on bootstrapped fits, with the df COLUMN NaN'd there - no df
governs percentile inference, so a relay never publishes analytical
provenance beside percentile statistics; where bootstrap draws are not
retained, a RECOMPUTE level on a bootstrapped fit RAISES rather than
silently returning analytical inference. **View-relay exception (Phase 2b PRs 1-2):**
estimators whose `aggregate()` RELAYS stored fields without recomputation
need no influence-function kit - there is nothing to re-weight. The
retention requirement binds RECOMPUTING estimators (the CallawaySantAnna
class); dCDH [M-026] relays its stored overall/event-study surfaces
verbatim, and StackedDiD [M-024] always materializes its event-study
surface at fit (the pooled regression always includes the interactions)
so both its levels are pure views - which is also why bootstrap-style
fail-closing does not apply to them: every stored inference mode relays
faithfully. Estimators whose
estimand is already a single aggregation (SunAbraham's saturated event study,
LPDiD's per-horizon design) expose `aggregate()` where meaningful as additive
surface; their native params (`only_event`/`only_pooled` etc.) are documented
domain vocabulary, not drift.

## 7. Inference surface

- `vcov_type` becomes the single SE-type selector wherever analytical SEs
  exist; the redundant `robust` flag dies [M-045]..[M-047] [M-115]. Estimators without
  analytical SEs (SyntheticControl, TROP, CiC) document their native variance
  methods instead of growing a dead param.
- Wild bootstrap is exposed via `inference="wild_bootstrap"` uniformly where
  supported [M-096] - "supported" meaning the estimator offers wild CLUSTER
  bootstrap as an ALTERNATIVE to analytical SEs. That roster is
  DifferenceInDifferences + TwoWayFixedEffects. (MultiPeriodDiD also carries
  the param but has no wild-bootstrap path - it warns and falls back to
  analytical; it is removed at 4.0 [M-010] and section 4.1 replaces that
  fallback with an explicit raise in the merged event-study mode, so the roster
  pins estimators that SUPPORT WCR, not those that merely expose the param.)

  **The selector must fail closed** - and since 3.9 (2(d) PR-B, [M-096]
  `done`) it does. Before the fix, `inference=` was stored without any
  valid-value check, and DiD routed `wild_bootstrap` WITHOUT `cluster=`
  silently to analytical (formerly pinned by
  `tests/test_wild_bootstrap.py::test_did_wild_bootstrap_requires_cluster`,
  flipped BY DESIGN), so a typo or a missing prerequisite quietly changed
  which SE/p-value/CI procedure ran - the no-silent-failures principle
  applied to inference selection. Fail closed means REFUSE, not
  warn-and-degrade: the accepted set is pinned to exactly
  `{"analytical", "wild_bootstrap"}` (string-typed) on `__init__` and
  transactional `set_params`, and `wild_bootstrap` without `cluster=`
  raises `ValueError` at fit. REGISTRY specifies WCR as
  `inference="wild_bootstrap"` *(with `cluster=`)*, so that combination is
  incoherent rather than merely unsupported - and a warning would still
  leave a procedure running that the caller did not ask for.

  **Implementation decisions (locked 2026-08-06; shipped in the 2(d)
  PR-B).** The accepted value set `{"analytical", "wild_bootstrap"}` is
  validated at `__init__` (transactional `set_params` inherits it via the
  BaseEstimator probe re-init); the COHERENCE checks run at fit - DiD
  with `inference="wild_bootstrap"` and no `cluster=` raises
  `ValueError`, and DiD/TWFE with `wild_bootstrap` and
  `n_bootstrap < 2` raise `ValueError`. **Amendment (2026-08-07,
  user-approved, same-diff with the [M-096] ledger notes): the floor is
  `< 2`, not the originally-locked `< 1`** - review verified by execution
  that `n_bootstrap ∈ {0, 1}` deterministically degenerates (the wild
  routing never consulted `n_bootstrap > 0`; both values ran WCR with too
  few draws and returned a wild-labeled all-NaN inference tuple with no
  warning), and `>= 2` is already the house floor in SyntheticDiD/TROP.
  TWFE's unit auto-cluster stays; MultiPeriodDiD's warn-and-fallback
  stays until its 4.0 removal (DEFERRED.md documents the limitation; the
  fallback is n_bootstrap-independent). The flipped fallback test carries
  its CHANGELOG disclosure. The roster guard is a dynamic sweep: the set
  of estimators exposing `inference` in `get_params()` is exactly
  {DifferenceInDifferences, MultiPeriodDiD, TwoWayFixedEffects} - a
  future estimator gaining WCR must adopt the selector or land its own
  row. The shipping PR also fixed a latent fit-idempotency bug in the
  same class: `_bootstrap_results` is now reset per-fit, so a refit after
  `set_params(inference="analytical")` no longer reports stale
  `inference_method="wild_bootstrap"` + bootstrap metadata.

  The apparent `inference=` vs `n_bootstrap>0` split is NOT drift: an estimator
  whose bootstrap IS its inference method runs a different procedure -
  CallawaySantAnna an influence-function multiplier bootstrap, SunAbraham a
  unit-pairs bootstrap (Rao-Wu rescaled under stratified/PSU survey designs),
  ChaisemartinDHaultfoeuille a group-level influence-function multiplier
  bootstrap that upgrades to PSU-level Hall-Mammen wild clustering under survey
  designs with strictly-coarser PSUs - against DiD/TWFE's residual wild cluster
  bootstrap with test-inverted CIs and no reference t-distribution. Spelling
  any of those `"wild_bootstrap"` would make a uniform name carry altered
  meaning, which rule 8 of section 8 forbids; they keep `n_bootstrap` as
  documented domain vocabulary on the section 6 precedent. Should one of them
  later want an `inference=` selector, the value names its own method - that is
  additive, post-4.0, minor-version work, not part of this program.
- `n_bootstrap` semantic unification [M-081, `done` since 3.9]: `0` =
  bootstrap off wherever an analytical path exists; bootstrap-only
  estimators document their positive defaults. Counts stay tuned per
  estimator (999 light / 200 compute-heavy) - NO numeric default changes.
  **Implementation decision (locked 2026-08-06; shipped in the 2(d)
  PR-B)**: a shared `validate_n_bootstrap` helper (non-negative int;
  accepts numpy integers; rejects bool/None/float/negative; 0 stays legal
  wherever it means off) is promoted to utils - verbatim from
  ChangesInChanges' local validator - and applied to EVERY estimator with
  a previously-unvalidated `n_bootstrap`: CallawaySantAnna, SunAbraham,
  EfficientDiD, ImputationDiD, TwoStageDiD, WooldridgeDiD, ContinuousDiD,
  the DiD family (DiD's `__init__`, inherited by MPD/TWFE), and
  StaggeredTripleDifference; CiC/QDiD re-point to the shared helper.
  **Named exception**: HeterogeneousAdoptionDiD has an analytical
  pointwise path but deliberately floors `n_bootstrap >= 1` - its
  `n_bootstrap` powers ONLY the optional sup-t band, whose off-switch is
  `fit(cband=False)`, so `n_bootstrap=0` would be a second, ambiguous
  off-switch; the 0=off clause does not apply to HAD and no HAD behavior
  changed. On the DiD/TWFE wild lane, 0 never meant off (the routing
  consults only the selector) - the [M-096] floor now rejects
  `n_bootstrap < 2` there at fit.
- **Auto-cluster policy** [M-080], flips at 4.0: every panel estimator
  (required `unit` column) defaults to clustering at unit
  (Bertrand-Duflo-Mullainathan practice), setting `cluster_name` /
  `n_clusters` metadata (locked cluster-label rule; verify each SE formula at
  implementation per that rule). `cluster=False` explicitly disables.
  Cross-sectional 2x2 estimators stay HC-robust unless `cluster=` is given.
  StackedDiD's hard-coded `cluster="unit"` default becomes an instance of the
  general policy rather than a special case.
- **df_convention default flip** [M-004]..[M-006] + [M-128]..[M-131]:
  `"residual"` -> `"cluster"` (G-1) at 4.0 - the locked #663 direction.
  PR C ([M-127], 3.9) resolved the "evaluate extending the knob" item EARLY:
  the knob is now a THREE-VALUE `{"residual","cluster","normal"}` surface on
  DiD/MPD/TWFE/LinearRegression plus SunAbraham/WooldridgeDiD-OLS/StackedDiD/
  ImputationDiD-pretrends (their 4.0 flips are [M-128]..[M-131]) and LPDiD
  (default already `"cluster"`, no flip row). The flip moves every clustered
  p-value/CI; the flip PR updates `TestDfConvention` /
  `test_moderate_t_pins_residual_df_convention` / the per-estimator knob
  suites' expectations and adds the migration-guide entry.
- Constructor hygiene rides Phase 2: ContinuousDiD's `covariates` moves from
  `__init__` to `fit()` [M-084] (the one estimator with a data column in the
  constructor), and the shared `BaseEstimator` mixin replaces the 25
  hand-rolled `get_params`/`set_params` copies (transactional set_params per
  the locked rule; `deep=` supported uniformly; the implementation measured
  25 defining classes - `changes_in_changes.py` holds two - correcting this
  document's earlier count of 24).

## 8. Contract-rename rules

The rules; the complete rename inventory is `docs/v4-deprecations.yaml`
(groups `renames-*`). **This section is normative for any rename the matrix
missed.**

1. `post` names a 0/1 post-period indicator; `time` always means the calendar
   period column. Never overload [M-030] [M-031] [M-082] [M-083] [M-085].
2. `first_treat` names the cohort / first-treatment-period column everywhere
   [M-032].
3. `unit` names the unit identifier; `group` is reserved for the treated-group
   0/1 indicator (TripleDifference) [M-033].
4. `covariates` names covariate column lists [M-034]; estimator-specific
   covariate structure may keep domain names when the split is methodological
   (Wooldridge `exovar`/`xtvar`/`xgvar` - kept).
5. No `_col` suffixes [M-035]..[M-042].
6. `control_group` is the param, `"never_treated"` / `"not_yet_treated"`
   (underscored) the values [M-043]. LPDiD's `"clean"` is kept as domain
   vocabulary (clean-control design, a different concept), documented in its
   docstring with the mapping.
7. `treatment` names a 0/1 indicator (group membership or take-up), never a
   cohort and never a treatment level; estimators with non-binary
   per-period treatment (dCDH) document theirs as domain vocabulary.
   Corollary: a column that MAY be non-binary does not get this name even
   when it is conceptually take-up. Fuzzy RD's observed-take-up column
   accepts dose values (matching R's `fuzzy=`; the estimand label degrades
   from complier-LATE to a bare local Wald ratio accordingly), so [M-042]
   [M-094] target `takeup` - the spelling the RD docstring already uses -
   rather than `treatment`.
8. Hybrid-naming principle (locked 2026-07-12): diff-diff names where the
   library owns the concept; domain vocabulary where it is the field's
   language; an R-equivalents mapping table (`yname/tname/idname/gname` ->
   `outcome/time/unit/first_treat`) ships in the docs, not as param aliases.
9. **A rename carries to every surface that mirrors it.** Renaming a
   constructor or `fit()` param renames any RESULTS field that echoes it back
   (and the `summary()` label that prints it), in the SAME diff and on the same
   shim schedule - a results field is public surface, is emitted by
   `to_dict()`, and section 5 forbids deprecated names in serialized output
   from 4.0. Mirrors added by amendment: [M-094] (mirrors [M-042]), [M-095]
   (mirrors [M-043]), [M-114] (mirrors [M-033]). When adding a rename row,
   grep the matching `*_results.py` for the old name before assuming the param
   row is the whole job.
10. **These rules bind module-level PUBLIC FUNCTIONS, not just classes.** The
   Phase 1 inventory was derived from the estimator roster, so exported
   functions went unaudited; the completeness sweep added [M-097] (the
   RETAINED `twowayfeweights` diagnostic - section 3.3 keeps it past 4.0, so
   no wrapper removal covers it), [M-098]..[M-112] (the three exported HAD
   pretest entry points, five `_col` params each), and [M-113]
   (`trim_weights`). A function that survives 4.0 gets the same shim schedule
   as a class surface; only surfaces scheduled for REMOVAL
   ([M-070]..[M-077]) are exempt, because the removal moots the name.

11. **A rename's scope includes every READER of the old name.** A row is not
   done when the definition site is renamed - it is done when nothing still
   reads the old name. Internal consumers migrate in the same diff, and the
   row's `code_refs` name them so the removal PR cannot miss them (see
   [M-043] / [M-095], whose readers span `power.py`, `practitioner.py`,
   `_reporting_helpers.py`, `business_report.py` and REPORTING.md). The
   specific hazard is `getattr(obj, "old_name", <default>)`: after removal it
   returns the DEFAULT instead of raising, so the consumer silently reports
   the wrong thing rather than crashing - a wrong-answer regression that no
   removal pin catches. Grep for the old name across `diff_diff/` and
   `docs/methodology/` before marking any rename row terminal. For an old
   name SHARED across rows (the `_col` family spans HAD, RDD and the pretest
   functions), a reader may be recorded on any row of that token family: the
   guard (`tests/test_naming_guard.py`) and removal PRs consult the family's
   `code_refs` UNION, and the pre-terminal repo-wide grep above remains the
   per-row safety net (decided 2026-08-01 over per-row duplication).

**Missed-rename amendments (2026-08-02, the 2(c)-ii PR-B sweep):** [M-136]
(LPDiD `to_dataframe(level="event")` - the drifted value spelling every
sibling already writes as `"event_study"`; same unification as [M-086]) and
[M-137]/[M-138] (`permutation_test[time]` / `leave_one_out_test[time]` -
public diagnostics params that forward verbatim into
`DifferenceInDifferences.fit`'s 0/1 post dummy, a rule-1 overload the
phase-1 estimator-roster inventory missed). The `run_placebo_test` /
`run_all_placebo_tests` WRAPPERS keep their single overloaded `time`
(calendar for the timing/group tests, post dummy for the two renamed
callees): a rename cannot express dual semantics - the signature redesign is
tracked in TODO.md and the guard carries honest allowlist reasons.

**Domain vocabulary that is NOT a violation** (recorded so the sweep is not
re-litigated): the staggered family's `group`/`groups` on results containers
and `GroupTimeEffect.group` name the ATT(g,t) COHORT in Callaway-Sant'Anna's
own notation - rule 8 protects exactly this, and renaming them to `unit` would
be actively wrong. `TripleDifference.fit[group]` is the treated-group 0/1
indicator that rule 3 explicitly reserves the name for. dCDH is the one
estimator where `group` genuinely means a unit id, which is why [M-033] and
[M-114] exist and the CS-family fields have no rows. `plot_group_effects`'s
`groups` selector is the same CS-cohort vocabulary on the plotting surface.

**Naming-checkpoint outcomes (2026-07-18 + 2026-07-31 addendum), with losing
candidates:**
`event_study=` bool (over `effects=` enum, `dynamic=` bool);
`spec="within"|"pooled"` (over `unit_fe=` bool, `model=` string);
`partition` (over `eligibility`); EventStudy alias dropped (over retarget -
altered-meaning trap - and over a convenience subclass); underscored
control_group values (over R-compact spellings, over accept-both);
unified event-study representation with `period_effects` as a 4.0->5.0
property (over keeping the dict canonical, over hard removal); n_bootstrap
semantic-only unification (over uniform 999, over uniform 200); panel
auto-cluster-at-unit (over never-auto-cluster, over status quo).
2026-07-31 addendum (consolidation scope): staggered-family mega-merge
rejected (over a `StaggeredDiD(method=)` union class - altered-meaning trap on
the union-params surface, and literature discoverability); ImputationDiD <->
TwoStageDiD merge rejected (different inference stacks - BJS conservative IF
SE vs GMM sandwich - and both independently cited); moderate alias diet
(section 3.2; over an aggressive diet dropping the author-initials shorthands
too, over no diet; EDiD was initially slated but retained - review evidence
showed it is the CSX paper's own estimator label); alias-diet 3.9
FutureWarning via module `__getattr__`
[M-135] (over silent removal - the only 4.0 removals that would have shipped
without a deprecation window).

## 9. Phase -> PR breakdown

Boundary rule, verbatim: **anything two later PRs could disagree about lives
above; anything only one PR cares about stays in that PR's plan.**

| Phase | Ships in | PRs (each: dedicated shim/removal tests + matrix flips + CHANGELOG naming flipped row ids) |
|---|---|---|
| 1 (this PR) | - | Spec + matrix + enforcement test + support edits |
| 2: contract foundations | 3.9 | (a) results base + unified event-study representation [M-092] + to_dict completion + the Diagnostic marker base on the diagnostic result roster [M-091] (section 3.5); (b) `aggregate()` + fit(aggregate=) shims [M-020..M-027] [M-139] (M-020's shim already shipped; M-139 is the HAD workflow twin, a pre-cut amendment); (c) param renames [M-030..M-047] [M-084] [M-086..M-089] + their results-field mirrors [M-094] [M-095] (section 8 rule 9) + the public-function completeness sweep [M-097..M-113] (section 8 rule 10) + the dCDH results mirror [M-114] + the fourth `robust` site [M-115] + the 2(c)-ii missed-rename amendments [M-136..M-138] (LPDiD `level` value; the two post-dummy diagnostics params) + BaseEstimator mixin + ContinuousDiD covariates move; (d) alias introduction [M-062] (the Spillover introduction is cancelled [M-063]) + the alias-diet `__getattr__` warning shim [M-135] + wrapper deprecations [M-070..M-077] + the two inference-surface policies: `n_bootstrap` semantic unification [M-081] and the wild-cluster-bootstrap roster guard [M-096]; shipped insertions (all done): the aggregate contract [M-122], the ETWFE reference-period family [M-123] [M-124] [M-125], and the variance-consolidation program [M-126] [M-127] |
| 3: merges | 3.9 | (a) TWFE event-study mode [M-010] + EventStudy warn [M-060] + the fit `time`->`post` rename [M-082] (gates: section 4.1's equivalence/divergence/pooled-parity test triple) (shipped: tests/test_v4_merge_mpd.py; consumer ports incl. HonestDiD/PreTrendsPower calendar routes); (b) TripleDifference facade [M-013] + the SDDD alias [M-064]; (c) CiC method= [M-015] |
| 4: release + soak | 3.9 cut | Migration guide written (skeleton: section 10); maintainer cuts 3.9; maint/3.8 rule active |
| 5: enforcement | 4.0 | Removals [M-010..M-015, M-020..M-027, M-139, M-030, M-032..M-047 old names, M-060, M-061, M-064, M-070..M-077, M-084, M-086..M-089, M-001..M-003, M-117, M-118, M-119, M-120] + the alias diet [M-132]..[M-134] + the amendment's old names [M-094] [M-095] [M-097..M-115] [M-136..M-138] (incl. their consumer migrations and the `clean_control` serialized reporting key); M-031's old `time` name persists as the merged class's calendar column, so it is deliberately absent from the removal roster (its 4.0 enforcement is the M-085 behavior entry below); property window: [M-016] property-flips at 4.0 (removal at 5.0); storage flips [M-050..M-058]; default policies [M-004..M-006, M-128..M-131, M-080]; merged-class behavior enforcements [M-083] [M-085]; warning retirement [M-007]; fastpath go/no-go [M-008]; diagnostic-family docs/roster reorganization [M-090]; sentinel retirement [M-093]; docs/llms.txt/README refresh |
| 6: front door | 4.1 | `event_study(data, outcome, unit, time, first_treat, estimator=...)` comparison entry point over the staggered family (sketch only; specified in its own plan) |

Citation semantic for the table: a cell may cite a row whose current `phase`
differs when that phase performs one of the row's lifecycle transitions (the
phase-5 removal roster cites rows still at their shim phase; the phase-2(b)
cell keeps M-020, whose shim shipped there). The row's `phase` field tracks
only the NEXT transition (section 11). Terminal rows (`done`/`removed`) are
exempt in both directions - citable for the historical record, never required.

**Remaining 3.9 sequence (2026-07-31).** The single canonical statement of the
remaining PR order; it records order and rationale only and does not
re-enumerate the cells' M-id lists:

1. Planning consolidation + the section 8 consolidation-scope decisions (the
   PR that wrote this subsection).
2. The naming-completeness guard test (shipped: `tests/test_naming_guard.py`,
   which carries the amended spec - surface sweep + phase-table agreement +
   consumer coverage - in its module docstring) - lands BEFORE 2(c), so the
   rename PR works from a mechanically-verified list.
3. 2(c)-i: the BaseEstimator mixin, front-loaded (shipped:
   `diff_diff/_base.py`, with the cross-estimator contract suite
   `tests/test_base_estimator.py` on a dynamic roster). Scope was section 7's
   normative statement verbatim: replace the hand-rolled
   `get_params`/`set_params` copies library-wide (3.9-cut checklist item 1) -
   not merely the standalone estimator classes; the exhaustive inventory
   found 25 defining classes (not 24). set_params is transactional via probe
   re-init - `type(self)(**merged)` validates before any mutation - so the
   renames build on a contract that can never drift from `__init__`.
4. 2(c)-ii: the rename sweep (the phase-2(c) cell); may split by rename group.
5. 2(b): post-fit `aggregate()` + `fit(aggregate=)` shims (the (b) cell).
   The reserved-id pool is spent for this wave (M-118..M-120 claimed;
   M-116/M-121 stay earmarked for the HAD rename and Wooldridge), so any
   further new row takes the next free id (M-139, the HAD workflow twin,
   was the first); the
   `EventStudyResults` downstream-consumability work (TODO.md row: the three
   consumers currently reject the unified container) lands before or inside
   this wave so the shims do not steer users into a dead end.
6. 2(d), split into TWO PRs (2026-08-06): PR-A - wrapper deprecations
   [M-070..M-077], the SCM introduction [M-062], and the alias-diet
   `__getattr__` shim [M-135] + dieted-alias surface sweep (reader
   surfaces are recorded in M-132..M-135's `code_refs`, the
   ledger-native home - this doc carries no file inventory; the PR
   starts from those `code_refs` and additionally greps each dieted
   alias repo-wide) (shipped: the eight wrapper shims +
   `tests/test_v4_wrapper_shims.py`, the module `__getattr__` +
   `tests/test_aliases.py`, SCM); then PR-B - the two inference-surface
   policies, `n_bootstrap` semantic unification [M-081] and the
   wild-cluster-bootstrap roster guard [M-096] (implementation
   decisions live in section 7, per this section's boundary rule)
   (shipped: the shared `validate_n_bootstrap` sweep, the fail-closed
   selector + `< 2` wild floor + per-fit bootstrap-state reset, and
   `tests/test_v4_inference_policy.py`).
7. Phase 3 merges (a)/(b)/(c) per the phase-3 cell.
8. Phase 4: migration guide, the 3.9-cut checklist below, cut.

**3.9-cut checklist (un-rowed obligations).** `test_due_rows_are_terminal`
gates everything that HAS a row; the following Phase 2 obligations are real but
not expressible as ledger rows, so the 3.9 release PR asserts them by hand:

1. The shared `BaseEstimator` mixin has replaced the hand-rolled
   `get_params`/`set_params` copies (section 7), with `deep=` uniform and
   set_params transactional per the locked rule. A pure refactor - no symbol
   changes - so no row can see it. DONE: `diff_diff/_base.py` +
   `tests/test_base_estimator.py` (25 classes converted; the ten
   formerly-lazy set_params surfaces now validate eagerly, REGISTRY
   EfficientDiD note updated in the same diff).
2. The R-equivalents mapping table (section 8 rule 8) ships in the docs.
3. The migration guide exists (section 10) and its TL;DR table has a row per
   breaking change, generated against the matrix rather than hand-listed.
4. The ledger and this document agree on the phase breakdown in the table
   above - any PR that re-scoped a phase edited both. Invariant and
   enforcement spec: `tests/test_naming_guard.py` (module docstring).

Everything else queued for 3.9 is row-gated, by one of two mechanisms. Symbol
rows that declare a `warning` gate on `deprecated_in` - the shim must have
shipped ([M-010] [M-013] [M-015], [M-020]..[M-027], [M-139], [M-030]..[M-047],
[M-070]..[M-077], [M-082],
[M-084], [M-086]..[M-089], [M-094] [M-095], [M-097]..[M-115]). Rows with no
shim to assert gate
on `introduced_in` instead - the new surface must have shipped: the
introduce-only alias [M-062] and the `behavior`-kind policies
[M-081] [M-091] [M-092] [M-096] [M-135]. That second mechanism is deliberate for
behavior rows: the early-flip guard keys off `deprecated_in`, so a flip version
would fail the very PR that implements the obligation (it lands while
`__version__` is still 3.8.x). Use `introduced_in`, not `deprecated_in`, for
any future behavior row that must ship in a release the repo has not yet
bumped to.

**4.0-cut checklist (final item):** the due-row sweep is AUTOMATED -
`tests/test_v4_matrix.py::test_due_rows_are_terminal` fails any release bump
whose version reaches a row's scheduled version while its status has not
flipped (rows must flip or be explicitly re-scheduled in the release PR). The
remaining manual item: re-run the repo-wide deprecation grep
(DeprecationWarning/FutureWarning/"will be removed"/"removed in v"/"next
major"/"flip") to catch anything born outside the matrix.

## 10. Migration-guide skeleton (`docs/migration-4.0.md`, written in Phase 4)

1. TL;DR table: one row per breaking change, old -> new, one-line fix.
2. The three merges (worked examples: MPD -> TWFE event_study incl. the SE
   shift and `spec="pooled"`; SDDD -> TripleDifference; QDiD -> CiC method=).
3. Renamed parameters (generated from the matrix, groups `renames-*`).
4. `results.aggregate()` (before/after snippets per estimator family).
5. Results fields (overall_att family -> canonical quintet; property window).
6. Inference defaults that moved numbers (df_convention, auto-cluster) - with
   how to reproduce 3.x numbers exactly.
7. Removed functions and aliases (wrappers, EventStudy, SDDD, QDiDResults,
   and the alias-diet three: CDiD, Stacked, Gardner).
8. Codemod section: the mechanical renames as a script/regex table.

## 11. Matrix mechanics (normative schema for `docs/v4-deprecations.yaml`)

**Fields.** `id` (`M-###`, unique, never reused), `kind`, `group` (required
- human clustering key), `old`,
`new`, `introduced_in`, `deprecated_in`, `removed_in`, `status`, `phase`,
`warning`, `test_ref`, `code_refs`, `notes`, plus kind-specific:
`old_default`/`new_default` (default-flip), `snippet` (warning-retirement),
`old_target`/`new_target` (alias), `env_var` + `decision_due` +
`decided_default` (env-default: `decision_due` is the version by which the
go/no-go must be recorded; `decided_default: on|off` records the outcome and
is required at `done` - "evaluated, kept off" is a first-class terminal
state). `deprecated_in`/`removed_in`/`new` are required-present but nullable;
versions match `\d+\.\d+(\.\d+)?`.

**Kinds.** `param` (constructor or method parameter), `param-value` (accepted
value spelling; full symbol lifecycle and due gate but NO reality probe -
accepted values are not introspectable, so behavioral enforcement lives in
the row's `test_ref` suite), `class`, `field` (results attribute),
`function`, `alias` (top-level export alias), `default-flip` (same param, new
default), `env-default` (environment-variable-resolved default),
`warning-retirement` (a warning message scheduled to disappear), `behavior`
(policy change not assertable by introspection; schema-checked only, flipped
manually, swept at the cut).

**Locator grammar.** `diff_diff:Name` (top-level export - REQUIRED for
class/function rows with `removed_in` set, so the locator survives module
deletion), `diff_diff:Class[param]` (`__init__` parameter),
`diff_diff:Class.method[param]`, `diff_diff:Class.attr` (field/property),
dotted module path (warning-retirement/env-default rows). The module part
must always import - a failed import is a hard test failure (typo guard),
never a legal absence.

**Status lifecycle.** Symbol kinds (param/class/field/function): `planned` ->
`shimmed` -> `removed`; the `shimmed` stop may be skipped only when the row's
deprecation rides a parent row (stated in `notes`, e.g. [M-011]).
`param-value` rows follow the same lifecycle (with `test_ref` required at
`shimmed`/`removed` and due-gate coverage). Remaining non-symbol kinds
(alias/default-flip/env-default/warning-retirement/behavior): `planned`
or `evaluate` -> `done`. Terminal rows (`removed`/`done`) keep asserting
forever - a removed symbol resurrecting is a test failure.

**Assertion semantics (enforced by `tests/test_v4_matrix.py`).**
- `param`/`class`/`function`: `planned` = old resolves AND new does NOT
  resolve; `shimmed` = both resolve + `test_ref` exists; `removed` = old
  absent, new (if non-null) resolves, `test_ref` exists.
- `field`: membership in `__dataclass_fields__` is the discriminator.
  `planned` = old is a dataclass field, new is NOT a dataclass field
  (property is fine); `shimmed` = new is the dataclass field, old resolves as
  a descriptor but is NOT a dataclass field; `removed` = old absent entirely.
- `alias`: `planned` with `old_target` = identity holds
  (`getattr(diff_diff, old) is resolve(old_target)`); introduce-only rows
  (`old_target: null`) = name absent from `diff_diff`; `done` = identity with
  `new_target`, or absent if `new_target` is null.
- `default-flip`: `inspect.signature` default equals `old_default`
  (`done`: equals `new_default`).
- `env-default`: resolver imports and returns False with the env var deleted
  (`done`: the resolver matches `decided_default` with the env var unset -
  "on" expects True, "off" expects False).
- `warning-retirement`: `snippet` present in the `code_refs` file (`done`:
  absent).
- `param-value`: schema + due gate + `test_ref` existence at
  `shimmed`/`removed`; no reality probe (value behavior asserted in the
  `test_ref` suite: old spelling warns at shim, rejected at removal, new
  spelling accepted).
- `behavior`: schema + `code_refs` existence only.
- Release gate (all kinds): once `diff_diff.__version__` reaches a row's
  `removed_in` (symbol/alias rows), flip version (`deprecated_in` on
  default-flip/warning-retirement/behavior rows), or `decision_due`
  (env-default rows), the status must be terminal; a due `introduced_in`
  means the row may no longer be `planned` OR `evaluate` (the new surface
  must have shipped; evaluate cannot satisfy an introduction - this is what
  gates introduce-only aliases and the Phase 2 marker); symbol rows with a
  due `deprecated_in` and a declared `warning` may no longer be `planned`.
  The gate is two-sided: an EARLY-removal guard fails any row that goes
  terminal while its scheduled version is still in the future (the shim
  window is a promise). Field-flip rows assert the whole family: at the 4.0
  flip, the full canonical quintet must be native fields and none of the
  ROW'S deprecated sibling names (both conventions: `overall_se`-style and
  `overall_att_se`-style) may remain fields - partial migrations fail. Other
  `overall_*` estimand families (e.g. ContinuousDiD's `overall_acrt*`) are
  separate estimands outside the quintet contract, untouched unless they get
  their own rows. Version comparisons pad to three components
  ('4.0' == '4.0.0').
  env-default `done` asserts the resolver matches `decided_default` with the
  env var unset - flip-on and evaluated-kept-off are both representable. A
  declared `test_ref` must exist at EVERY status, including terminal
  (removal pins survive forever); behavior / default-flip / env-default rows
  REQUIRE a `test_ref` at `done` (semantic flips need ledger-linked
  behavioral evidence). Alias rows must NOT declare `warning` - an alias is
  the same object as its target, so the warning rides the parent class row
  when the target is itself deprecated, and a `behavior`-row `__getattr__`
  mechanism ([M-135]) when the target survives (schema-enforced either way:
  the alias row itself never declares one). Top-level `diff_diff:Name`
  class/function rows
  and alias rows also assert `__all__` membership consistent with their
  status (stale `import *` entries fail). The shipped row ids are a
  committed snapshot in the enforcement test (121 as of 2(b) PR-4's
  HAD workflow-aggregate row: Phase 1 + the diagnostic-family
  amendment +
  the M-092/M-093 results-contract rows + the M-094..M-096 amendment rows +
  the M-097..M-115 completeness sweep + M-117..M-120/M-122 + the ETWFE
  reference-period pair M-123/M-124 + M-125 + M-126 + M-127..M-131 +
  the alias-diet family M-132..M-135 + the 2(c)-ii amendments
  M-136..M-138 + M-139;
  the snapshot extends by a new id range in the same diff that appends
  rows): ids are never deleted or reused, and the test fails if any
  snapshot id disappears.

**Cross-row migration rule.** Removing a symbol requires migrating, in the
same diff, every other row whose locators or `code_refs` reference it (e.g.
[M-016]'s locator moves to the successor container when [M-011] flips). The
enforcement test's hard-fail on unresolvable module parts makes forgetting
this loud, not silent.

**`phase` semantics.** `phase` names the program phase whose PR performs the
row's NEXT status transition; each flip updates it (terminal rows keep their
last value).

## 12. Open questions

None.
