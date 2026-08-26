# Migrating to 4.0

```{note}
This describes the **upcoming 4.0 release**. Most of the renames below can be adopted from
3.9 onward, well before 4.0 lands — the new spellings and the old ones both work during the
deprecation window, so you can migrate incrementally and silence the warnings as you go.
```

diff-diff 4.0 removes the deprecated surface that 3.9 shipped warnings for, merges three pairs
of estimators, and flips two families of inference defaults. Every change is tracked as a row in
`docs/v4-deprecations.yaml`, and the appendix below is checked against that ledger by
`tests/test_v4_matrix.py` — if a row is added, moved, or rescheduled and this page is not
updated, CI fails.

## What changes

| Area | What changes | Rows | Where |
|---|---|---|---|
| aggregate-postfit | Aggregation moves off `fit()` onto post-fit `results.aggregate(...)` | 13 | §4 |
| alias-table | Six export aliases are removed in favour of their canonical class names | 6 | §7 |
| constructor-hygiene | `covariates=` moves from the constructor to `fit()` | 1 | §7b |
| df-convention-flip | The `df_convention` default flips to `"cluster"` on seven estimators - numbers move | 7 | §6 |
| diagnostic-family | Bacon is re-homed out of the estimator roster into the diagnostics family | 1 | §7b |
| field-flip | Nine results containers rename `overall_att` to the canonical `att` | 9 | §5 |
| function-wrappers | Eight module-level wrapper functions are removed; call the classes | 8 | §7 |
| merge-ddd | `StaggeredTripleDifference` is absorbed by `TripleDifference` | 5 | §2 |
| merge-mpd | `MultiPeriodDiD` is absorbed by `TwoWayFixedEffects(event_study=True)` | 5 | §2 |
| merge-qdid | `QDiD` is absorbed by `ChangesInChanges(method="qdid")` | 2 | §2 |
| obligation-sdid-params | Two SyntheticDiD constructor params (inert since 3.0.0) are removed | 3 | §7b |
| obligation-warning-retirements | A transition `FutureWarning` stops being emitted | 1 | §7b |
| policy-auto-cluster | TwoWayFixedEffects auto-clusters on `unit` unless you pass `cluster=` | 1 | §6 |
| renames-cohort | `cohort=` becomes `first_treat=` | 1 | §3 |
| renames-col-suffix | The `*_col` suffix is dropped across 27 parameters and one field | 27 | §3 |
| renames-control-group | `clean_control=` becomes `control_group=` | 2 | §3 |
| renames-dcdh | dCDH's `group=`/`controls=` become `unit=`/`covariates=` | 4 | §3 |
| renames-level | Aggregation-level params and their accepted values unify on `level=`/`"event_study"` | 4 | §3 |
| renames-post | The post-dummy `time=` becomes `post=` | 3 | §3 |
| renames-robust-drop | `robust=` is dropped in favour of `vcov_type=` - read §3, the translation is per-estimator | 4 | §3 |
| results-contract | A legacy sentinel field is retired in favour of the unified event-study surface | 1 | §5 |

## The three merges

Each merge keeps one class and retires the other. The retired class still works in 3.9 and warns.

### MultiPeriodDiD → TwoWayFixedEffects(event_study=True)

```python
# 3.x
from diff_diff import MultiPeriodDiD
results = MultiPeriodDiD().fit(data, outcome="y", unit="id", time="period", treatment="treat")

# 4.0
from diff_diff import TwoWayFixedEffects
results = TwoWayFixedEffects().fit(
    data, outcome="y", unit="id", time="period", treatment="treat",
    event_study=True, spec="pooled", post_periods=[3, 4, 5],
)
```

Two things are easy to get wrong here, and both change numbers:

- **`spec="pooled"` reproduces the MultiPeriodDiD design.** The new default is `spec="within"`,
  which adds unit fixed effects. That moves **point estimates as well as standard errors** on
  unbalanced panels or with covariates — the two specs coincide only in the restricted
  equivalence case (balanced panel, no covariates, simultaneous adoption). "Only SEs move" is
  not the migration message.
- **`post_periods=` is required** in event-study mode. MultiPeriodDiD defaulted to a midpoint
  split of the calendar, which is a silent guess; the merged mode makes you state the treatment
  boundary. `spec="pooled"` is also the only spec valid for repeated cross-sections.

### StaggeredTripleDifference → TripleDifference

```python
# 3.x
from diff_diff import StaggeredTripleDifference
results = StaggeredTripleDifference().fit(data, outcome="y", unit="id", time="period",
                                          first_treat="g", eligibility="p")

# 4.0
from diff_diff import TripleDifference
results = TripleDifference().fit(data, outcome="y", unit="id", time="period",
                                 first_treat="g", partition="p")
```

The staggered fit parameters are keyword-only on the merged class. The results container is the
unified `TripleDifference` shape; the 2x2x2 design reads as a degenerate single-ATT view of it.

### QDiD → ChangesInChanges(method="qdid")

```python
# 3.x
from diff_diff import QDiD
results = QDiD(n_bootstrap=200, seed=42).fit(data, outcome="y", treatment="treat", time="post")

# 4.0
from diff_diff import ChangesInChanges
results = ChangesInChanges(method="qdid", n_bootstrap=200, seed=42).fit(
    data, outcome="y", treatment="treat", time="post")
```

**Only the class spelling is deprecated, not the estimator.** `method="qdid"` is a fully
supported comparison mode and emits no warning of its own — the numbers are unchanged, because
it is the same engine. `method="cic"` is the default and encodes Athey–Imbens' recommendation.

## Renamed parameters

Most renames are mechanical: the new spelling already exists, so you can adopt it today and the
old one keeps working until 4.0. See the appendix for the full list, and the
[R Comparison](r_comparison.rst) page for how these names map onto the R packages.

Three families need more than a search-and-replace:

**`robust=` → `vcov_type=` is per-estimator, not a single rule.** Verified against the current
release:

| estimator | `robust=True` | `robust=False` |
|---|---|---|
| `DifferenceInDifferences`, `LinearRegression` | already the `hc1` default — drop it | `vcov_type="classical"` |
| `TripleDifference` | drop it | drop it — only `hc1` is accepted, so `robust=` never changed inference, and `vcov_type="classical"` **raises** |
| `HeterogeneousAdoptionDiD` | `vcov_type="hc1"` | drop it — non-robust is its legacy default |

**`time=` → `post=` carries a semantic change, and the old name survives.** On
`DifferenceInDifferences`, `TripleDifference` and `TwoWayFixedEffects`, `time=` used to mean a
0/1 post dummy. The name is not removed — it is **repurposed** to mean the calendar column. From
4.0, passing `time=` in static mode (`TwoWayFixedEffects`) or 2x2x2 mode (`TripleDifference`)
**raises `ValueError`** rather than being silently reinterpreted. If you pass a post dummy, rename
it to `post=`; if you pass a calendar column to an event-study or staggered fit, leave it alone.

**Dropped parameters are not always deletions.** A parameter with no successor in the appendix
may still need a replacement call — see its `Fix` cell.

## Post-fit aggregation

Aggregation moves off `fit()` onto the results object:

```python
# 3.x
results = CallawaySantAnna().fit(data, ..., aggregate="event_study")

# 4.0
results = CallawaySantAnna().fit(data, ...)
event_study = results.aggregate("event_study")
```

`DiagnosticReport` / `BusinessReport` need no migration step of their own:
where applicable, their event-study-gated checks derive the surface
internally via the result's post-fit `aggregate('event_study')` when the
raw `event_study_effects` field is absent, so moving a fit off the
fit-time keyword no longer silently disables those checks. The routing is
estimator-specific: CallawaySantAnna derives for parallel trends,
pre-trends power, and sensitivity; ImputationDiD and TwoStageDiD for
parallel trends (`pretrends=True` fits) and heterogeneity; ContinuousDiD
for heterogeneity. StackedDiD and SunAbraham always populate the raw
surface (nothing to derive), and ChaisemartinDHaultfoeuille's pre-period
checks read `placebo_event_study` directly.

```{warning}
**Bootstrapped fits: CallawaySantAnna, DMLDiD, and EfficientDiD are covered; the remaining recompute adopters are not yet.**
On `CallawaySantAnna`, `DMLDiD` (which inherits the CS replay channel), and `EfficientDiD`, the post-fit recompute levels now REPLAY the
fit-time multiplier bootstrap from the fit-retained RNG state (percentile inference
matching a fit-time aggregation to floating-point reassociation) — no fit-time keyword
needed; only pre-replay legacy pickles and artifacts moved across the Rust/NumPy weight
backend fail closed with a refit message. On `ImputationDiD`, `TwoStageDiD` and
`ContinuousDiD`, the recompute levels still raise `NotImplementedError` when the fit used
`n_bootstrap > 0` — keep the fit-time call there for now and track their open `TODO.md`
rows.
`aggregate("simple")` — and, on its five adopters, `aggregate("total")` (3.10; DMLDiD joined in 3.11 — panel fits only, RCS fits fail `total` closed) — does relay,
and `StackedDiD`, `ChaisemartinDHaultfoeuille` and `HeterogeneousAdoptionDiD` are unaffected —
their `aggregate()` is a pure view over stored fields.
```

## Results fields

Nine containers rename `overall_att` to the canonical `att`. **The new accessor already works
today**, so this migration can be done immediately; the old name becomes a warning-emitting
property at 4.0 and is removed at 5.0.

```python
att = results.att          # canonical, works now
att = results.overall_att  # warns from 4.0, removed at 5.0
```

**The whole inference quintet moves, not just the point estimate.** Wherever a container
carried `overall_*` inference fields alongside `overall_att`, they flip to the canonical
names in the same step:

| old | new |
|---|---|
| `overall_att` | `att` |
| `overall_se` | `se` |
| `overall_t_stat` | `t_stat` |
| `overall_p_value` | `p_value` |
| `overall_conf_int` | `conf_int` |

```{note}
`ContinuousDiDResults` is the outlier: its sibling fields are spelled
`overall_att_se`, `overall_att_t_stat`, `overall_att_p_value` and
`overall_att_conf_int` — with the `att` infix — and they flip to the same canonical
names. If you grep for `overall_se` you will miss them.
```

## Inference defaults that move numbers

Two changes alter results without changing any call:

- **`df_convention` flips to `"cluster"`** on seven estimators. Pass `df_convention="residual"`
  to reproduce 3.x numbers exactly.
- **Panel estimators auto-cluster at `unit`** when `cluster=` is omitted. Pass
  **`cluster=False`** to disable it — `None` and omission both mean "auto-cluster", so
  `cluster=None` does *not* reproduce unclustered 3.x standard errors. Cross-sectional
  2x2 estimators stay HC-robust unless you pass `cluster=`. Note the TWFE **event-study**
  mode has auto-clustered since 3.9 (it shipped that way with the merge), so this flip
  changes the static and other panel paths, not that one.

## Removed functions and aliases

Eight module-level wrapper functions and six export aliases are removed. Both were thin
indirections: call the estimator class directly, or import the canonical name. The appendix lists
each one with its target.

## Remaining 4.0 changes

Smaller items that do not fit the families above — two inert `SyntheticDiD` constructor
parameters, the `covariates=` constructor-to-`fit()` move, a retired transition warning, the
Bacon roster re-homing, and the family-wide `anticipation` validation below. The appendix lists
the ledger-derived removals/flips; [M-144] is a behavior tightening with no removal/deprecation
fields and appears here only.

- `anticipation` is validated across the family ([M-144], landing at 4.0): whole-valued floats
  that previously fit identically to their integer now raise — pass the `int`; bool and
  negative/non-integer values raise too. Accepted numpy integers are normalized, so the public
  `anticipation` attribute and `get_params()["anticipation"]` are now always built-in `int`
  (previously a numpy scalar survived). WooldridgeDiD's error message text changed to the shared
  wording, and its constructor now reports a bad `bootstrap_weights`/`vcov_type`/`df_convention`
  before a bad `anticipation` (the ordering flipped).

One pending decision: the `DIFF_DIFF_SOLVE_OLS_FASTPATH` environment default has a go/no-go due
at 4.0 that has not been made. If it lands on, it is a numerics change and will be documented
then; "evaluated, kept off" is an equally valid outcome, so it carries no appendix row today.

## Codemod

The mechanical keyword renames can be applied with a regex sweep. This table covers only
**identifier** renames — it deliberately excludes dropped parameters (no successor to write),
results-field renames (`.groups` → `.units` would rewrite every pandas `groupby(...).groups` in
your code), and accepted-value renames, which change strings rather than names.

| find | replace |
|---|---|
| `\boutcome_col\s*=` | `outcome=` |
| `\bdose_col\s*=` | `dose=` |
| `\btime_col\s*=` | `time=` |
| `\bunit_col\s*=` | `unit=` |
| `\bfirst_treat_col\s*=` | `first_treat=` |
| `\brunning_col\s*=` | `running=` |
| `\btreatment_col\s*=` | `takeup=` |
| `\bweight_col\s*=` | `weights=` |
| `\bcohort\s*=` | `first_treat=` |
| `\bclean_control\s*=` | `control_group=` |
| `\bcontrols\s*=` | `covariates=` |
| `\baggregation\s*=` | `level=` |

Note `treatment_col` maps to **`takeup`**, not `treatment` — it is the one `*_col` parameter
whose replacement is not just the suffix dropped.

**Three renames are deliberately NOT in the table, because a global replace would corrupt
working code.** Each is a rename only in a specific call:

| rename | where it applies | why it cannot be global |
|---|---|---|
| `group=` → `unit=` | `ChaisemartinDHaultfoeuille.fit`, `twowayfeweights` | `TripleDifference.fit` takes a `group=` that is **not** renamed |
| `time=` → `post=` | `DifferenceInDifferences.fit`, `permutation_test`, `leave_one_out_test` | `time=` survives everywhere else as the calendar column — see the renamed-parameters section |
| `aggregate=` → post-fit call | the aggregation family | not a rename at all; it moves off `fit()` |

Review every hit even in the safe table: these are word-boundary matches on common words.

## Already shipped in 3.9

These landed in 3.9 and need no 4.0 action, but they moved behaviour and are easy to mistake for
4.0 changes:

**Numbers moved.** The ETWFE reference-period fixes (unidentified-cohort exclusion, and a
fail-closed guard on designs with no estimable post-treatment cell), the clustered-CR1
`K_reference` convergence, and the tail-df consolidation.

**Fail-closed validation.** `n_bootstrap` semantics unified; `inference="wild_bootstrap"` without
`cluster=` now raises; `TripleDifference(pscore_trim=0)` is rejected; an all-NaN Wooldridge
overall ATT raises rather than returning NaN. These are refusals, not default changes — no
numeric default moved.

**Additive API and container changes.** The unified `EventStudyResults` surface, the
`AggregationResult` container, and the `Diagnostic` marker base on the diagnostic result roster.
The alias-diet `__getattr__` shim also landed here: `CDiD`, `Stacked` and `Gardner` still import
but now emit a `FutureWarning`, and they no longer appear in `dir()`/`vars()`.

## Appendix: every 4.0 change

One row per ledger row that requires action at 4.0 — 108 in total. `Old` and `New` are the
ledger's own locators. An em dash in `New` means the ledger names no successor **locator**; it
does not mean no action is required, so read the `Fix` cell.

| Row | Group | Old | New | Fix |
|---|---|---|---|---|
| M-020 | aggregate-postfit | `diff_diff:CallawaySantAnna.fit[aggregate]` | `diff_diff:CallawaySantAnnaResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits replay the fit-time bootstrap post-fit - see the aggregation section. |
| M-021 | aggregate-postfit | `diff_diff:ImputationDiD.fit[aggregate]` | `diff_diff:ImputationDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. On a bootstrapped fit the recompute levels raise today - see the aggregation section. |
| M-022 | aggregate-postfit | `diff_diff:TwoStageDiD.fit[aggregate]` | `diff_diff:TwoStageDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. On a bootstrapped fit the recompute levels raise today - see the aggregation section. |
| M-023 | aggregate-postfit | `diff_diff:EfficientDiD.fit[aggregate]` | `diff_diff:EfficientDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits replay the fit-time bootstrap post-fit - see the aggregation section. |
| M-024 | aggregate-postfit | `diff_diff:StackedDiD.fit[aggregate]` | `diff_diff:StackedDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. |
| M-025 | aggregate-postfit | `diff_diff:ContinuousDiD.fit[aggregate]` | `diff_diff:ContinuousDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. On a bootstrapped fit the recompute levels raise today - see the aggregation section. |
| M-026 | aggregate-postfit | `diff_diff:ChaisemartinDHaultfoeuille.fit[aggregate]` | `diff_diff:ChaisemartinDHaultfoeuilleResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. |
| M-027 | aggregate-postfit | `diff_diff:HeterogeneousAdoptionDiD.fit[aggregate]` | `diff_diff:HeterogeneousAdoptionDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. |
| M-117 | aggregate-postfit | `diff_diff:CallawaySantAnna.fit[balance_e]` | `diff_diff:CallawaySantAnnaResults.aggregate[balance_e]` | Move `balance_e=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits replay the fit-time bootstrap post-fit - see the aggregation section. |
| M-118 | aggregate-postfit | `diff_diff:ImputationDiD.fit[balance_e]` | `diff_diff:ImputationDiDResults.aggregate[balance_e]` | Move `balance_e=` off `fit()` onto post-fit `results.aggregate(...)`. On a bootstrapped fit the recompute levels raise today - see the aggregation section. |
| M-119 | aggregate-postfit | `diff_diff:TwoStageDiD.fit[balance_e]` | `diff_diff:TwoStageDiDResults.aggregate[balance_e]` | Move `balance_e=` off `fit()` onto post-fit `results.aggregate(...)`. On a bootstrapped fit the recompute levels raise today - see the aggregation section. |
| M-120 | aggregate-postfit | `diff_diff:EfficientDiD.fit[balance_e]` | `diff_diff:EfficientDiDResults.aggregate[balance_e]` | Move `balance_e=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits replay the fit-time bootstrap post-fit - see the aggregation section. |
| M-139 | aggregate-postfit | `diff_diff:did_had_pretest_workflow[aggregate]` | — | Remove `aggregate=`; the battery is inferred from panel shape - two periods select the overall battery, more than two select the event-study battery. It is not selected post-fit (`HADPretestReport.aggregate` is a metadata field, not a method). |
| M-060 | alias-table | `EventStudy` | — | Import `TwoWayFixedEffects(...).fit(..., event_study=True)` - the alias is dropped, not retargeted instead of the `EventStudy` alias. |
| M-061 | alias-table | `QDiDResults` | — | Import `ChangesInChangesResults` instead of the `QDiDResults` alias. |
| M-064 | alias-table | `SDDD` | — | Import `TripleDifference` (staggered mode) instead of the `SDDD` alias. |
| M-132 | alias-table | `CDiD` | — | Import `ContinuousDiD` instead of the `CDiD` alias. |
| M-133 | alias-table | `Stacked` | — | Import `StackedDiD` instead of the `Stacked` alias. |
| M-134 | alias-table | `Gardner` | — | Import `TwoStageDiD` instead of the `Gardner` alias. |
| M-084 | constructor-hygiene | `diff_diff:ContinuousDiD[covariates]` | `diff_diff:ContinuousDiD.fit[covariates]` | `covariates=` moves from the constructor to `fit()` - pass it at fit time. |
| M-004 | df-convention-flip | `diff_diff:DifferenceInDifferences[df_convention]` | — | The `df_convention` default flips to `"cluster"`; pass `df_convention="residual"` to keep 3.x numbers. |
| M-005 | df-convention-flip | `diff_diff:TwoWayFixedEffects[df_convention]` | — | The `df_convention` default flips to `"cluster"`; pass `df_convention="residual"` to keep 3.x numbers. |
| M-006 | df-convention-flip | `diff_diff:LinearRegression[df_convention]` | — | The `df_convention` default flips to `"cluster"`; pass `df_convention="residual"` to keep 3.x numbers. |
| M-128 | df-convention-flip | `diff_diff:SunAbraham[df_convention]` | — | The `df_convention` default flips to `"cluster"`; pass `df_convention="residual"` to keep 3.x numbers. |
| M-129 | df-convention-flip | `diff_diff:WooldridgeDiD[df_convention]` | — | The `df_convention` default flips to `"cluster"`; pass `df_convention="residual"` to keep 3.x numbers. |
| M-130 | df-convention-flip | `diff_diff:StackedDiD[df_convention]` | — | The `df_convention` default flips to `"cluster"`; pass `df_convention="residual"` to keep 3.x numbers. |
| M-131 | df-convention-flip | `diff_diff:ImputationDiD[df_convention]` | — | The `df_convention` default flips to `"cluster"`; pass `df_convention="residual"` to keep 3.x numbers. |
| M-090 | diagnostic-family | `diff_diff:BaconDecomposition` | — | Bacon moves out of the estimator roster into the diagnostics family - update docs references and imports of the roster, not call sites. |
| M-050 | field-flip | `diff_diff:CallawaySantAnnaResults.overall_att` | `diff_diff:CallawaySantAnnaResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-051 | field-flip | `diff_diff:SunAbrahamResults.overall_att` | `diff_diff:SunAbrahamResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-052 | field-flip | `diff_diff:ImputationDiDResults.overall_att` | `diff_diff:ImputationDiDResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-053 | field-flip | `diff_diff:TwoStageDiDResults.overall_att` | `diff_diff:TwoStageDiDResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-054 | field-flip | `diff_diff:StackedDiDResults.overall_att` | `diff_diff:StackedDiDResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-055 | field-flip | `diff_diff:EfficientDiDResults.overall_att` | `diff_diff:EfficientDiDResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-056 | field-flip | `diff_diff:WooldridgeDiDResults.overall_att` | `diff_diff:WooldridgeDiDResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-057 | field-flip | `diff_diff:ChaisemartinDHaultfoeuilleResults.overall_att` | `diff_diff:ChaisemartinDHaultfoeuilleResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-058 | field-flip | `diff_diff:ContinuousDiDResults.overall_att` | `diff_diff:ContinuousDiDResults.att` | Read `att` instead of `overall_att`; the canonical accessor already works today. |
| M-070 | function-wrappers | `diff_diff:imputation_did` | — | Call the estimator class directly instead of `imputation_did`. |
| M-071 | function-wrappers | `diff_diff:two_stage_did` | — | Call the estimator class directly instead of `two_stage_did`. |
| M-072 | function-wrappers | `diff_diff:stacked_did` | — | Call the estimator class directly instead of `stacked_did`. |
| M-073 | function-wrappers | `diff_diff:trop` | — | Call the estimator class directly instead of `trop`. |
| M-074 | function-wrappers | `diff_diff:synthetic_control` | — | Call the estimator class directly instead of `synthetic_control`. |
| M-075 | function-wrappers | `diff_diff:triple_difference` | — | Call the estimator class directly instead of `triple_difference`. |
| M-076 | function-wrappers | `diff_diff:bacon_decompose` | — | Call the estimator class directly instead of `bacon_decompose`. |
| M-077 | function-wrappers | `diff_diff:chaisemartin_dhaultfoeuille` | — | Call the estimator class directly instead of `chaisemartin_dhaultfoeuille`. |
| M-013 | merge-ddd | `diff_diff:StaggeredTripleDifference` | `diff_diff:TripleDifference.fit[first_treat]` | See the merges section for the worked before/after. |
| M-014 | merge-ddd | `diff_diff:StaggeredTripleDiffResults` | — | Read the unified `TripleDifference` results shape (a degenerate single-ATT view for 2x2x2) instead of this container. |
| M-085 | merge-ddd | `diff_diff:TripleDifference.fit[time]` | — | In 2x2x2 mode, `time=` raises - pass `post=` for the 0/1 post dummy. `time=` now means the staggered calendar column only. |
| M-140 | merge-ddd | `diff_diff:TripleDifference.fit[aggregate]` | `diff_diff:TripleDifferenceResults.aggregate` *(not yet implemented)* | No successor yet - post-fit `aggregate()` is not implemented on the DDD container (open `TODO.md` row). Keep the fit-time call until it lands. |
| M-141 | merge-ddd | `diff_diff:TripleDifference.fit[balance_e]` | `diff_diff:TripleDifferenceResults.aggregate[balance_e]` *(not yet implemented)* | No successor yet - post-fit `aggregate()` is not implemented on the DDD container (open `TODO.md` row). Keep the fit-time call until it lands. |
| M-010 | merge-mpd | `diff_diff:MultiPeriodDiD` | `diff_diff:TwoWayFixedEffects.fit[event_study]` | See the merges section for the worked before/after. |
| M-011 | merge-mpd | `diff_diff:MultiPeriodDiDResults` | — | Read the unified event-study surface on the merged `TwoWayFixedEffects` results instead of this container. |
| M-012 | merge-mpd | `diff_diff:PeriodEffect` | — | Superseded by the unified event-study representation - read the event-study surface rather than per-period `PeriodEffect` objects. |
| M-016 | merge-mpd | `diff_diff:MultiPeriodDiDResults.period_effects` | — | Not removed at 4.0: it becomes a `FutureWarning` property view over the unified event-study surface on the successor container, and is removed at 5.0. Migrate the read to the successor class. |
| M-083 | merge-mpd | `diff_diff:TwoWayFixedEffects.fit[time]` | — | In static mode (`event_study=False`), `time=` raises - pass `post=` for the 0/1 post dummy. `time=` now means the calendar column only. |
| M-015 | merge-qdid | `diff_diff:QDiD` | `diff_diff:ChangesInChanges[method]` | See the merges section for the worked before/after. |
| M-143 | merge-qdid | `diff_diff:ChangesInChangesResults.estimator` | `diff_diff:ChangesInChangesResults.method` | See the merges section for the worked before/after. |
| M-001 | obligation-sdid-params | `diff_diff:SyntheticDiD[lambda_reg]` | — | Drop it. It has been IGNORED since 3.0.0, so 3.x already auto-computes regularization - do NOT copy its value into `zeta_omega=`, which activates an override that was inert and changes weights, ATT and inference. Set `zeta_omega=` only as a deliberate new choice. |
| M-002 | obligation-sdid-params | `diff_diff:SyntheticDiD[zeta]` | — | Drop it. It has been IGNORED since 3.0.0, so 3.x already auto-computes regularization - do NOT copy its value into `zeta_lambda=`, which activates an override that was inert and changes weights, ATT and inference. Set `zeta_lambda=` only as a deliberate new choice. |
| M-003 | obligation-sdid-params | `diff_diff:SyntheticDiDResults.placebo_effects` | `diff_diff:SyntheticDiDResults.variance_effects` | Read `variance_effects` instead of `placebo_effects`. |
| M-007 | obligation-warning-retirements | `diff_diff.estimators` | — | The MultiPeriodDiD `e=-1` transition FutureWarning stops being emitted; drop any warning filter that suppressed it. |
| M-080 | policy-auto-cluster | `diff_diff:TwoWayFixedEffects[cluster]` | — | Panel estimators auto-cluster at `unit` when `cluster=` is omitted; pass `cluster=False` to disable it and keep unclustered 3.x standard errors (`None`/omission means auto-cluster, not off). |
| M-032 | renames-cohort | `diff_diff:WooldridgeDiD.fit[cohort]` | `diff_diff:WooldridgeDiD.fit[first_treat]` | Rename the keyword: `cohort=` becomes `first_treat=`. |
| M-035 | renames-col-suffix | `diff_diff:HeterogeneousAdoptionDiD.fit[outcome_col]` | `diff_diff:HeterogeneousAdoptionDiD.fit[outcome]` | Rename the keyword: `outcome_col=` becomes `outcome=`. |
| M-036 | renames-col-suffix | `diff_diff:HeterogeneousAdoptionDiD.fit[dose_col]` | `diff_diff:HeterogeneousAdoptionDiD.fit[dose]` | Rename the keyword: `dose_col=` becomes `dose=`. |
| M-037 | renames-col-suffix | `diff_diff:HeterogeneousAdoptionDiD.fit[time_col]` | `diff_diff:HeterogeneousAdoptionDiD.fit[time]` | Rename the keyword: `time_col=` becomes `time=`. |
| M-038 | renames-col-suffix | `diff_diff:HeterogeneousAdoptionDiD.fit[unit_col]` | `diff_diff:HeterogeneousAdoptionDiD.fit[unit]` | Rename the keyword: `unit_col=` becomes `unit=`. |
| M-039 | renames-col-suffix | `diff_diff:HeterogeneousAdoptionDiD.fit[first_treat_col]` | `diff_diff:HeterogeneousAdoptionDiD.fit[first_treat]` | Rename the keyword: `first_treat_col=` becomes `first_treat=`. |
| M-040 | renames-col-suffix | `diff_diff:RegressionDiscontinuity.fit[outcome_col]` | `diff_diff:RegressionDiscontinuity.fit[outcome]` | Rename the keyword: `outcome_col=` becomes `outcome=`. |
| M-041 | renames-col-suffix | `diff_diff:RegressionDiscontinuity.fit[running_col]` | `diff_diff:RegressionDiscontinuity.fit[running]` | Rename the keyword: `running_col=` becomes `running=`. |
| M-042 | renames-col-suffix | `diff_diff:RegressionDiscontinuity.fit[treatment_col]` | `diff_diff:RegressionDiscontinuity.fit[takeup]` | Rename the keyword: `treatment_col=` becomes `takeup=`. |
| M-088 | renames-col-suffix | `diff_diff:RDPlot.fit[outcome_col]` | `diff_diff:RDPlot.fit[outcome]` | Rename the keyword: `outcome_col=` becomes `outcome=`. |
| M-089 | renames-col-suffix | `diff_diff:RDPlot.fit[running_col]` | `diff_diff:RDPlot.fit[running]` | Rename the keyword: `running_col=` becomes `running=`. |
| M-094 | renames-col-suffix | `diff_diff:RegressionDiscontinuityResults.treatment_col` | `diff_diff:RegressionDiscontinuityResults.takeup` | Read `takeup` instead of `treatment_col` on the results object. |
| M-098 | renames-col-suffix | `diff_diff:joint_pretrends_test[outcome_col]` | `diff_diff:joint_pretrends_test[outcome]` | Rename the keyword: `outcome_col=` becomes `outcome=`. |
| M-099 | renames-col-suffix | `diff_diff:joint_pretrends_test[dose_col]` | `diff_diff:joint_pretrends_test[dose]` | Rename the keyword: `dose_col=` becomes `dose=`. |
| M-100 | renames-col-suffix | `diff_diff:joint_pretrends_test[time_col]` | `diff_diff:joint_pretrends_test[time]` | Rename the keyword: `time_col=` becomes `time=`. |
| M-101 | renames-col-suffix | `diff_diff:joint_pretrends_test[unit_col]` | `diff_diff:joint_pretrends_test[unit]` | Rename the keyword: `unit_col=` becomes `unit=`. |
| M-102 | renames-col-suffix | `diff_diff:joint_pretrends_test[first_treat_col]` | `diff_diff:joint_pretrends_test[first_treat]` | Rename the keyword: `first_treat_col=` becomes `first_treat=`. |
| M-103 | renames-col-suffix | `diff_diff:joint_homogeneity_test[outcome_col]` | `diff_diff:joint_homogeneity_test[outcome]` | Rename the keyword: `outcome_col=` becomes `outcome=`. |
| M-104 | renames-col-suffix | `diff_diff:joint_homogeneity_test[dose_col]` | `diff_diff:joint_homogeneity_test[dose]` | Rename the keyword: `dose_col=` becomes `dose=`. |
| M-105 | renames-col-suffix | `diff_diff:joint_homogeneity_test[time_col]` | `diff_diff:joint_homogeneity_test[time]` | Rename the keyword: `time_col=` becomes `time=`. |
| M-106 | renames-col-suffix | `diff_diff:joint_homogeneity_test[unit_col]` | `diff_diff:joint_homogeneity_test[unit]` | Rename the keyword: `unit_col=` becomes `unit=`. |
| M-107 | renames-col-suffix | `diff_diff:joint_homogeneity_test[first_treat_col]` | `diff_diff:joint_homogeneity_test[first_treat]` | Rename the keyword: `first_treat_col=` becomes `first_treat=`. |
| M-108 | renames-col-suffix | `diff_diff:did_had_pretest_workflow[outcome_col]` | `diff_diff:did_had_pretest_workflow[outcome]` | Rename the keyword: `outcome_col=` becomes `outcome=`. |
| M-109 | renames-col-suffix | `diff_diff:did_had_pretest_workflow[dose_col]` | `diff_diff:did_had_pretest_workflow[dose]` | Rename the keyword: `dose_col=` becomes `dose=`. |
| M-110 | renames-col-suffix | `diff_diff:did_had_pretest_workflow[time_col]` | `diff_diff:did_had_pretest_workflow[time]` | Rename the keyword: `time_col=` becomes `time=`. |
| M-111 | renames-col-suffix | `diff_diff:did_had_pretest_workflow[unit_col]` | `diff_diff:did_had_pretest_workflow[unit]` | Rename the keyword: `unit_col=` becomes `unit=`. |
| M-112 | renames-col-suffix | `diff_diff:did_had_pretest_workflow[first_treat_col]` | `diff_diff:did_had_pretest_workflow[first_treat]` | Rename the keyword: `first_treat_col=` becomes `first_treat=`. |
| M-113 | renames-col-suffix | `diff_diff:trim_weights[weight_col]` | `diff_diff:trim_weights[weights]` | Rename the keyword: `weight_col=` becomes `weights=`. |
| M-043 | renames-control-group | `diff_diff:StackedDiD[clean_control]` | `diff_diff:StackedDiD[control_group]` | Rename the keyword: `clean_control=` becomes `control_group=`. |
| M-095 | renames-control-group | `diff_diff:StackedDiDResults.clean_control` | `diff_diff:StackedDiDResults.control_group` | Read `control_group` instead of `clean_control` on the results object. |
| M-033 | renames-dcdh | `diff_diff:ChaisemartinDHaultfoeuille.fit[group]` | `diff_diff:ChaisemartinDHaultfoeuille.fit[unit]` | Rename the keyword: `group=` becomes `unit=`. |
| M-034 | renames-dcdh | `diff_diff:ChaisemartinDHaultfoeuille.fit[controls]` | `diff_diff:ChaisemartinDHaultfoeuille.fit[covariates]` | Rename the keyword: `controls=` becomes `covariates=`. |
| M-097 | renames-dcdh | `diff_diff:twowayfeweights[group]` | `diff_diff:twowayfeweights[unit]` | Rename the keyword: `group=` becomes `unit=`. |
| M-114 | renames-dcdh | `diff_diff:ChaisemartinDHaultfoeuilleResults.groups` | `diff_diff:ChaisemartinDHaultfoeuilleResults.units` | Read `units` instead of `groups` on the results object. |
| M-044 | renames-level | `diff_diff:WooldridgeDiDResults.to_dataframe[aggregation]` | `diff_diff:WooldridgeDiDResults.to_dataframe[level]` | Rename the keyword: `aggregation=` becomes `level=`. |
| M-086 | renames-level | `diff_diff:WooldridgeDiDResults.aggregate[type]=event` | `diff_diff:WooldridgeDiDResults.aggregate[type]=event_study` | Change the accepted value: `"event"` becomes `"event_study"`. |
| M-087 | renames-level | `diff_diff:WooldridgeDiDResults.summary[aggregation]` | — | `summary()` unifies to the library-wide `summary(alpha=None)` signature - drop `aggregation=` and select the level on `aggregate()` instead. |
| M-136 | renames-level | `diff_diff:LPDiDResults.to_dataframe[level]=event` | `diff_diff:LPDiDResults.to_dataframe[level]=event_study` | Change the accepted value: `"event"` becomes `"event_study"`. |
| M-030 | renames-post | `diff_diff:DifferenceInDifferences.fit[time]` | `diff_diff:DifferenceInDifferences.fit[post]` | Rename the keyword: `time=` becomes `post=`. |
| M-137 | renames-post | `diff_diff:permutation_test[time]` | `diff_diff:permutation_test[post]` | Rename the keyword: `time=` becomes `post=`. |
| M-138 | renames-post | `diff_diff:leave_one_out_test[time]` | `diff_diff:leave_one_out_test[post]` | Rename the keyword: `time=` becomes `post=`. |
| M-045 | renames-robust-drop | `diff_diff:DifferenceInDifferences[robust]` | — | Drop `robust=True` (already the `hc1` default); `robust=False` becomes `vcov_type="classical"`. |
| M-046 | renames-robust-drop | `diff_diff:TripleDifference[robust]` | — | Drop it. `TripleDifference` accepts only `vcov_type="hc1"`, so `robust=` never changed its inference - do NOT write `vcov_type="classical"`, which raises. |
| M-047 | renames-robust-drop | `diff_diff:HeterogeneousAdoptionDiD[robust]` | — | HAD's legacy default is non-robust: `robust=True` becomes `vcov_type="hc1"`; drop `robust=False`. |
| M-115 | renames-robust-drop | `diff_diff:LinearRegression[robust]` | — | Drop `robust=True` (already the `hc1` default); `robust=False` becomes `vcov_type="classical"`. |
| M-093 | results-contract | `diff_diff:CallawaySantAnnaResults.event_study_effects` | `diff_diff:EventStudyResults` | The sentinel is retired - read the unified `EventStudyResults` surface instead of the raw field. |
