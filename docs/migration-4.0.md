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
| lwdid-results-contract | `LWDiDResults` joins the uniform results contract: the `pvalue`/`ci` alias properties warn in favour of `p_value`/`conf_int` (two rows); `to_dataframe()` is reshaped ([M-153], prose) | 2 | §7b |
| merge-ddd | `StaggeredTripleDifference` is absorbed by `TripleDifference` | 5 | §2 |
| merge-mpd | `MultiPeriodDiD` is absorbed by `TwoWayFixedEffects(event_study=True)` | 5 | §2 |
| merge-qdid | `QDiD` is absorbed by `ChangesInChanges(method="qdid")` | 2 | §2 |
| obligation-sdid-params | Two SyntheticDiD constructor params (inert since 3.0.0) are removed | 3 | §7b |
| obligation-sc-inner-budget | Two SyntheticControl constructor params (inert since the exact NNLS inner solve) warn at 4.0 and are removed at 5.0 | 2 | §7b |
| obligation-warning-retirements | A transition `FutureWarning` stops being emitted | 1 | §7b |
| policy-auto-cluster | SpilloverDiD auto-clusters on `unit` unless you pass `cluster=`; `cluster=False` becomes the uniform explicit off-switch | 1 | §6 |
| renames-cohort | `cohort=` becomes `first_treat=` | 1 | §3 |
| renames-col-suffix | The `*_col` suffix is dropped across 27 parameters and one field | 27 | §3 |
| renames-control-group | `clean_control=` becomes `control_group=` | 2 | §3 |
| renames-dcdh | dCDH's `group=`/`controls=` become `unit=`/`covariates=` | 4 | §3 |
| renames-level | Aggregation-level params and their accepted values unify on `level=`/`"event_study"` | 5 | §3 |
| renames-post | The post-dummy `time=` becomes `post=` | 3 | §3 |
| renames-robust-drop | `robust=` is dropped in favour of `vcov_type=` - read §3, the translation is per-estimator | 4 | §3 |
| results-contract | A legacy sentinel field is retired in favour of the unified event-study surface | 1 | §5 |

## The three merges

Each merge keeps one class and retires the other. The retired classes warned through 3.9 and are
gone at 4.0 (`ImportError` on `from diff_diff import MultiPeriodDiD`, `StaggeredTripleDifference`,
`QDiD`).

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
- **`unit=` changes the standard errors on `spec="pooled"`.** The 3.x call above passed
  `unit="id"`, which MultiPeriodDiD read only for Conley / staggered-adoption checks. The merged
  mode auto-clusters at `unit` whenever `unit=` is passed and the fit is not an explicit
  `classical`/`hc2`/`hc3`, Conley or survey fit (`hc1`, the default, is NOT an opt-out). To
  reproduce MultiPeriodDiD's unclustered HC1 numbers, drop `unit=`; an explicit `cluster=`
  reproduces a MultiPeriodDiD fit that passed the same `cluster=`. (`cluster=False` is the
  uniform off-switch since 4.0, [M-080].)

```python
# 4.0 - bit-exact with the 3.x unclustered MultiPeriodDiD default (no unit=)
results = TwoWayFixedEffects().fit(
    data, outcome="y", time="period", treatment="treat",
    event_study=True, spec="pooled", post_periods=[3, 4, 5],
)
```

- **`results.period_effects` still works, with a `FutureWarning`** ([M-016]; removed in 5.0).
  On a `TwoWayFixedEffects` event-study surface it returns the same calendar-keyed dict of
  per-period records (`period, effect, se, t_stat, p_value, conf_int, is_significant,
  significance_stars`; the reference period omitted; `is_significant` at the fit's `alpha`).
  The successor is the surface itself: `results.to_dataframe()` has one row per `event_time`
  (reference row included) and the columnar `att`/`se`/`p_value`/`conf_int_lower`/
  `conf_int_upper` arrays.

```python
# 4.0 - the per-period surface replaces the period_effects dict
es = TwoWayFixedEffects().fit(
    data, outcome="y", unit="id", time="period", treatment="treat",
    event_study=True, post_periods=[3, 4, 5], reference_period=2,
)
table = es.to_dataframe()          # one row per event_time, reference row included
post = table[table["event_time"].isin(es.post_periods)]
```

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

The staggered fit parameters are keyword-only on the merged class. Both designs return ONE
`TripleDifferenceResults` (row M-014; `results.design` is `"2x2x2"` or `"staggered"`); the
2x2x2 design reads as a degenerate single-ATT view of it (empty group-time table). The 3.x
`StaggeredTripleDiffResults` container is gone: read `results.att` / `results.se` / ... where you
read `overall_att` / `overall_se` / ..., and its `to_dict()` emits the canonical keys. The
fit-time `aggregate=` / `balance_e=` keywords are gone too (rows M-140 / M-141) - aggregation is
post-fit, and the paper's Eq. 4.14 overall (the 3.x `overall_att_es` field) is a `weights=`
selector on `aggregate('simple')`:

```python
# 4.0: post-fit aggregation on the staggered design (bootstrapped fits replay the
# fit-time multiplier bootstrap, so sup-t bands and percentile inference match)
es = results.aggregate("event_study", balance_e=None)   # EventStudyResults
cohorts = results.aggregate("group")                     # AggregationResult
eq414 = results.aggregate("simple", weights="event_time")  # the Eq. 4.14 overall, one row
```

3.x pickles of the 2x2x2 `TripleDifferenceResults` load unchanged (the new fields are
backfilled); 3.x `StaggeredTripleDiffResults` pickles cannot load, like every removed class.

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

**Only the class spelling was removed, not the estimator.** `method="qdid"` is a fully
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
| `DifferenceInDifferences`, `LinearRegression` | already the `hc1` default — drop it | `vcov_type="classical"` on an UNCLUSTERED fit; with `cluster=` (and on `TwoWayFixedEffects`, which always auto-clusters) just drop it — 3.x already remapped `robust=False` + clustering to `hc1`/CR1, and `vcov_type="classical"` with `cluster=` raises |
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

**Dropped `fit()` parameters shift the positional slots after them.** Removing the fit-time
`aggregate=`/`balance_e=` keywords moves every later positional parameter left: on
`CallawaySantAnna`, `ImputationDiD` and `TwoStageDiD` `survey_design` moves from slot 9 to 7;
on `EfficientDiD` `survey_design` 9 -> 7 and `store_eif` 10 -> 8; on `StackedDiD` (`aggregate`
sat before them) `population`/`survey_design`/`covariates` move 7/8/9 -> 6/7/8; on
`ContinuousDiD` `survey_design`/`covariates` move 8/9 -> 7/8; on `ChaisemartinDHaultfoeuille`
`L_max`, `covariates`, `trends_linear`, `trends_nonparam`, `honest_did`, `heterogeneity`,
`design2` and `survey_design` each move one slot left (7..14 -> 6..13). `HeterogeneousAdoptionDiD`
and `did_had_pretest_workflow` shift nothing (everything after the removed slot is keyword-only;
a seventh positional on `fit()` now raises). Pass these by keyword.

## Post-fit aggregation

Aggregation moves off `fit()` onto the results object:

```python
# 3.x (a TypeError from 4.0 - the keyword is gone):
#   results = CallawaySantAnna().fit(data, ..., aggregate="event_study")

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
**Bootstrapped fits: every adopter has a post-fit route - two mechanisms.**
On `CallawaySantAnna`, `DMLDiD` (which inherits the CS replay channel), and `EfficientDiD`, the post-fit recompute levels REPLAY the
fit-time multiplier bootstrap from the fit-retained RNG state (percentile inference
matching a fit-time aggregation to floating-point reassociation); only pre-replay legacy
pickles and artifacts moved across the Rust/NumPy weight backend fail closed with a refit
message. On `ImputationDiD`, `TwoStageDiD` and `ContinuousDiD` a fit with `n_bootstrap > 0`
ALWAYS computes the event-study (and, on Imputation/TwoStage, group) percentile surface at fit
and `aggregate('event_study')` / `aggregate('group')` RELAYS it - the same numbers the
3.x `fit(aggregate="all")` route produced, with a NaN `df` column and no `vcov`. Two limits
there: `balance_e=` is unavailable on those bootstrapped fits (`NotImplementedError` - the
stored surface was computed unrestricted; re-fit with `n_bootstrap=0` for a balanced window),
and a bootstrapped pickle written by 3.9-3.13 stores no surface (or, if it was fitted with
`balance_e=`, a restricted one), so re-fit under 4.0 for the relayed surface. If the surface
could not be materialized at fit, the fit warns once, the headline bootstrap is unchanged,
and the relay raises with that reason.
`aggregate("simple")` - and, on its five adopters, `aggregate("total")` (3.10; DMLDiD joined in 3.11 - panel non-survey fits only: RCS fits, declared-`survey_design=` fits, and bare-`cluster=` fits with incomplete treated support fail `total` closed) - relays on every adopter,
and `StackedDiD`, `ChaisemartinDHaultfoeuille` and `HeterogeneousAdoptionDiD` are unaffected -
their `aggregate()` is a pure view over stored fields. One residual delta on
replicate-weight `ImputationDiD`/`TwoStageDiD` designs: the plain fit's replicate stack is
`[overall]`-only, so on degenerate designs the OVERALL row's se/CI/df can differ from what a
3.x `fit(aggregate=...)` call reported (each surface self-consistent).
```

## Results fields

Nine containers rename `overall_att` to the canonical `att`. Since 4.0 the canonical name
is the stored dataclass field and the old name is a read-only warning-emitting property
(removed at 5.0); the new accessor already worked under 3.x, so a codebase that migrated
early sees no change.

```python
att = results.att          # canonical: the stored field
att = results.overall_att  # FutureWarning since 4.0, removed at 5.0
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
`ContinuousDiDResults` is the outlier: its sibling fields were spelled
`overall_att_se`, `overall_att_t_stat`, `overall_att_p_value` and
`overall_att_conf_int` — with the `att` infix — and they flipped to the same canonical
names. If you grep for `overall_se` you will miss them. Since 4.0 all nine of its legacy
spellings warn: the five former fields AND the four former silent `overall_se` /
`overall_t_stat` / `overall_p_value` / `overall_conf_int` aliases.
```

**Constructing a results object by hand** (test fixtures, adapters) takes the canonical
keywords: `CallawaySantAnnaResults(att=..., se=..., t_stat=..., p_value=..., conf_int=..., ...)`.
Passing a legacy keyword such as `overall_att=` raises `TypeError`. The fields were renamed in
their original slots, so positional construction order is unchanged.

**Pickles written under 3.x load unchanged.** Each flipped container's `__setstate__` rewrites
the legacy `overall_*` keys to the canonical names on load (an existing canonical key wins),
so the loaded object stores the quintet natively and `overall_*` never reappear in its
`__dict__` or in a re-pickle.

**The legacy names are read-only.** Assigning `results.overall_se = x` raises
`AttributeError`; assign the canonical field (`results.se = x`) instead. The bootstrap draw
containers (`results.bootstrap_results.overall_att_se` and friends) are NOT part of this
flip: they hold no point estimate, keep their payload names, and emit no warning.

**Report schemas.** `DiagnosticReport.to_dict()` reports `headline_metric.name` and
`target_parameter.headline_attribute` as `"att"` for the nine containers (`"overall_att"`
under 3.x), and `DIAGNOSTIC_REPORT_SCHEMA_VERSION` is `"3.0"`;
`BusinessReport.to_dict()` reports `target_parameter.headline_attribute` as `"att"` likewise
and `BUSINESS_REPORT_SCHEMA_VERSION` is `"4.0"` (its `headline` block carries no name key).

## Inference defaults that move numbers

Two changes alter results without changing any call:

- **`df_convention` flips to `"cluster"`** on seven estimators (rows M-004..M-006 and
  M-128..M-131): `DifferenceInDifferences`, `TwoWayFixedEffects`, `LinearRegression`,
  `SunAbraham`, `WooldridgeDiD` (OLS arms), `StackedDiD` and `ImputationDiD` (pretrends
  leads); `LPDiD` was already there. Clustered analytical p-values and confidence intervals
  now use the Stata/fixest `t(G − 1)` reference instead of the fitted residual df
  `t(n − K)`; point estimates, standard errors and t-statistics never move.
  (`SpilloverDiD` gained the knob at 4.0 with the same `"cluster"` default, [M-154] - no
  flip, a new surface; see the bullet below.)

  ```python
  TwoWayFixedEffects().fit(df, outcome="y", treatment="d", post="post", unit="unit")
  # 4.0: p-values / CIs at t(G - 1) on the auto-clustered fit
  TwoWayFixedEffects(df_convention="residual").fit(
      df, outcome="y", treatment="d", post="post", unit="unit"
  )  # 3.x inference, bit-identical
  ```

  **Which fits move.** Every fit whose analytical sandwich is clustered: `TwoWayFixedEffects`
  (static and event-study with `unit=`), `SunAbraham`, `WooldridgeDiD` OLS arms and
  `StackedDiD` cluster at `unit` by default, so their default fits move; `ImputationDiD`
  moves only on the `pretrends=True` lead rows; `DifferenceInDifferences` and
  `LinearRegression` move only when you pass `cluster=` / `cluster_ids=`. The analytical
  lanes that survive inside a bootstrapped fit move too: `WooldridgeDiD`'s per-cell inference
  and its post-fit `aggregate()`, and `ImputationDiD`'s bootstrapped pretrends leads. Code that
  relays these p-values moves with them: the fake-timing / fake-group placebo diagnostics
  (when `estimator_kwargs` carries `cluster=`) and the power simulators
  (`simulate_power` / `simulate_mde` / `simulate_sample_size`) on `TwoWayFixedEffects`,
  `SunAbraham` and `StackedDiD` profiles.

  **Which fits do not move.** Unclustered and Conley fits (the knob is inert without a
  cluster count), survey-design fits (the survey df keeps precedence), `vcov_type="hc2_bm"`
  (Bell-McCaffrey DOF keep precedence), bootstrap percentile p-values and CIs, the BJS
  overall/post inference of `ImputationDiD`, `WooldridgeDiD`'s logit/poisson arms, the
  permutation and leave-one-out diagnostics (they use only the point estimates), and every
  estimator without the knob. On the surfaces where the knob is a documented no-op —
  non-`pretrends` `ImputationDiD` fits and `WooldridgeDiD` logit/poisson fits — passing
  `df_convention="residual"` emits the inert-config `UserWarning` (there is nothing to
  restore there).
- **`SpilloverDiD` auto-clusters at `unit`** when `cluster=` is omitted — it is the one
  estimator whose default standard errors move. Pass **`cluster=False`** for the 3.x
  observation-level HC numbers; `None` and omission both mean "auto-cluster", so
  `cluster=None` does *not* reproduce them. No other default moves: static and event-study
  `TwoWayFixedEffects`, `SunAbraham`, `WooldridgeDiD`, `ImputationDiD`, `TwoStageDiD`,
  `LPDiD` and `StackedDiD` already clustered at `unit` in 3.x, and the influence-function
  and unit-cross-section estimators (`CallawaySantAnna`, `EfficientDiD`, `DMLDiD`,
  `ChaisemartinDHaultfoeuille`, `LWDiD`, `HeterogeneousAdoptionDiD`, the staggered
  `TripleDifference` mode) are unit-level by construction and keep reporting no cluster
  metadata. Cross-sectional 2x2 estimators stay HC-robust unless you pass `cluster=`.
  `cluster=False` becomes the explicit off-switch on every estimator whose clustering
  is a choice (in 3.x it was accepted nowhere): on `SpilloverDiD`, `TwoWayFixedEffects`,
  `SunAbraham`, `WooldridgeDiD` and `LPDiD` it yields the plain HC sandwich — on the
  last four that is **new** unclustered behaviour, not a 3.x reproduction (the
  within-transform lanes carry the absorbed-FE full-K rescale, so the SE equals HC1 on
  the explicit full-dummy design; the t reference is the residual df, `df_convention`
  being inert without a cluster count; `SunAbraham`'s survey lane stops injecting the
  unit PSU, so its TSL SE moves too; `WooldridgeDiD`'s logit/poisson arms give the plain
  QMLE sandwich under an unchanged normal-theory reference distribution - the SE and
  hence the p-value move); on the by-construction estimators
  and on `ImputationDiD`/`TwoStageDiD` it is an accepted no-op (their variance is
  unit-level by construction, so nothing can be switched off); `StackedDiD` keeps
  rejecting it (its `cluster` accepts only `"unit"`/`"unit_subexp"`, the stacked
  regression clusters by construction). Three combinations are rejected rather than
  given an unverified variance: `cluster=False` with `inference="wild_bootstrap"`
  (DiD/TWFE) or with `WooldridgeDiD(n_bootstrap > 0)` - those bootstraps draw at the
  analytical cluster - and `cluster=False` with `vcov_type="hc2_bm"` on `SunAbraham` /
  `WooldridgeDiD` (their Bell-McCaffrey DOF lane is cluster-only; `TwoWayFixedEffects`
  does run the one-way CR2-BM lane). `SunAbraham`'s pairs bootstrap accepts it as a no-op
  (it resamples units regardless of `cluster`).
- **`cluster` accepts `str | None | False` only** ([M-155]). The shared validator
  replaces 13 per-class lookups that accepted any hashable column label: an integer- or
  tuple-labelled cluster column must be renamed to a string, and `cluster=True` raises
  (previously a `KeyError` or silent unit clustering, depending on the class). The check
  runs at construction, through `set_params` and again at `fit()`.
- **Cluster metadata is populated on analytical fits.** `DifferenceInDifferences` and
  `TwoWayFixedEffects` report `n_clusters` from the regression's own effective cluster
  count on non-survey analytical fits (it was `None` unless the wild bootstrap ran), and
  `SunAbrahamResults` gains `n_clusters` and reports `cluster_name` on every clustered
  analytical fit (it was `None` off the Conley path). Under a survey design with no
  explicit `cluster=`, `TwoWayFixedEffects` now reports `cluster_name=None` (the
  auto-cluster is never injected as a PSU, so the former `"unit"` label described ids
  the regression never received); summaries render ", G=..." and `SunAbraham` summaries
  gain a "Std. errors:" line on analytical non-survey fits.
- **`SpilloverDiD` p-values and CIs on its clustered lanes move to t(G − 1)** ([M-154]):
  `SpilloverDiD` gained the three-value `df_convention` knob with the 4.0 `"cluster"`
  default, so the unit auto-cluster and an explicit `cluster=` now use the same reference
  distribution as every other auto-clustered default. `SpilloverDiD(df_convention="residual")`
  restores the 3.x residual-df inference bit-for-bit; `"cluster"` falls back to the residual df
  on `cluster=False` and Conley fits (`"normal"` still selects z there); the survey design df
  keeps precedence under every value. Results
  report `df_convention` and `inference_df`.

## Removed functions and aliases

Eight module-level wrapper functions and six export aliases are removed. Both were thin
indirections: call the estimator class directly, or import the canonical name. The appendix lists
each one with its target.

## Remaining 4.0 changes

Smaller items that do not fit the families above — two inert `SyntheticDiD` constructor
parameters, two `SyntheticControl` inner-solve parameters that became inert when the donor
weights moved to an exact non-negative least squares ([M-148], [M-149]: `DeprecationWarning`
at 4.0, removed at 5.0), the `covariates=` constructor-to-`fit()` move, a retired transition warning, the
Bacon roster re-homing, and the family-wide `anticipation` validation below. The appendix lists
the ledger-derived removals/flips; [M-144], [M-145], and [M-146] are behavior tightenings,
[M-147] is an additive policy, and [M-153] is a results-frame reshape. These have no
removal/deprecation fields and appear here only. [M-155] (the shared `cluster`
argument contract) and [M-154] (the `SpilloverDiD` `df_convention` knob), both in
"Inference defaults that move numbers", are the same kind of behavior tightening.

- `WooldridgeDiD(unsupported_period_action="error")` ([M-147]) refuses periods lacking
  comparison support before removing them. The default `"drop"` preserves filtering and
  warnings, so existing callers need no migration. The policy applies to all three methods
  independently of `rank_deficient_action`; other identification checks still apply.
  Survey validation runs first, then `"error"` raises `ValueError` for unsupported periods;
  `"drop"` retains the survey-domain `NotImplementedError` when removal would be required.

- `anticipation` is validated across the family ([M-144], landing at 4.0): whole-valued floats
  that previously fit identically to their integer now raise — pass the `int`; bool and
  negative/non-integer values raise too. Accepted numpy integers are normalized, so the public
  `anticipation` attribute and `get_params()["anticipation"]` are now always built-in `int`
  (previously a numpy scalar survived). WooldridgeDiD's error message text changed to the shared
  wording, and its constructor now reports a bad `bootstrap_weights`/`vcov_type`/`df_convention`
  before a bad `anticipation` (the ordering flipped).

- `pscore_trim` is validated via a shared helper ([M-145], landing at 4.0): `ContinuousDiD`
  tightens from `[0, 0.5)` to `(0, 0.5)` — `pscore_trim=0` (which disabled the overlap clip
  keeping IPW/DR weights finite) now raises, its message wording changed, and non-real-scalar
  inputs (`None`, strings, `Decimal`/`Fraction`, 1-element arrays) raise `ValueError` instead of
  `TypeError` (or being silently accepted, for the 1-element array). `CallawaySantAnna` gains the
  same type guard; `TripleDifference`, `CallawaySantAnna`, and `ContinuousDiD` now store the value
  coerced to built-in `float`; `LWDiD`'s message wording changed. (The removed
  `StaggeredTripleDifference` class kept its permissive construction shape through 3.x.)

- `summary(alpha=...)` / `print_summary(alpha=...)` on the results classes family-wide
  ([M-146], landing at 4.0; the staggered family first, then the non-staggered summaries -
  DiDResults incl. SpilloverDiDResults, MultiPeriodDiDResults, SyntheticDiDResults,
  TripleDifferenceResults, TROPResults, ContinuousDiDResults, and LWDiDResults - whose
  `summary()` previously took no `alpha` at all): a value different from the
  fit-time `alpha` now raises `ValueError` instead of silently relabeling the
  confidence-interval header over fit-time stored intervals; `alpha=0.0`, previously
  swallowed by a falsy-`or` default, raises too. Re-fit at the desired alpha instead.
  `SyntheticControlResults.summary` joins the contract as a carve-out: its `alpha` was a
  dead no-op (it prints no alpha-based interval; the displayed confidence set is keyed on
  its own stored `gamma`), and a non-fit value now raises like the rest of the family.

- `LWDiDResults.to_dataframe()` is reshaped ([M-153], landing at 4.0): it gains the uniform
  `level=` selector (`"simple"` default, `"group"`, `"event_study"`) and returns the shared
  aggregation / event-study schema (`level, label, target, att, se, t_stat, p_value,
  conf_int_lower, conf_int_upper, n, weight, df`) plus LWDiD's `rolling`, `estimation_method`
  and `vcov_type` columns; `to_csv()` takes the same `level`. What left the 3.10-3.13 frame:
  the `term` (common timing) / `cohort` (staggered) column is now `label`; the trailing
  `"Overall"` row of staggered fits is now `level="simple"` (the default returns ONE headline
  row - the former cohort rows are `level="group"`); `n_treated` is the schema's `n`;
  `n_obs` and `n_control` are no longer in the frame (read `to_dict()`); `ci_lower`/`ci_upper`
  are `conf_int_lower`/`conf_int_upper`.

- `BusinessReport.to_dict()` schema `2.0` -> `3.0` (landing at 4.0 with the [M-095] removal): the
  `assumption` block no longer carries the legacy `clean_control` key; read `control_group`.
  `DiagnosticReport`'s schema stays `2.0`.

The `DIFF_DIFF_SOLVE_OLS_FASTPATH` environment default was due a go/no-go at 4.0 ([M-008]):
decided 2026-10-18 as **evaluated, kept off**. No numerics change — the opt-in fast path
behaves exactly as in 3.x, and a later flip would be a major-version change on its own
schedule. It therefore carries no appendix row.

### Estimators added after 3.9

`LWDiD` (3.10), `DMLDiD` (3.11), `DurationDiD` (3.12), `RDDensityTest` and the TWFE weights
family (`attgt_weights`, `decompose_twfe_weights`, 3.13) were born on the 4.0 vocabulary, so
none of the renames above touches them. The `df_convention` flip does not touch them either:
they expose no such knob. `DMLDiDResults` inherits `CallawaySantAnnaResults` storage, so its
`overall_att` family rides [M-050] with the parent. `LWDiDResults` joined the uniform
results contract in the cut's conformance PR: `summary(alpha=None)` / `print_summary(alpha=None)`
under the [M-146] guard, the reshaped `to_dataframe(level=...)` / `to_csv(level=...)` described
under "Remaining 4.0 changes" ([M-153]), and FutureWarnings on its `pvalue`/`ci` alias
properties ([M-151], [M-152]: read `.p_value` / `.conf_int`; removed at 5.0). `DurationDiDResults`
gains `aggregate("simple")` (a verbatim relay of the stored bootstrap inference, `df` NaN) and
`RDDensityTestResult` gains `to_dict()`.

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

One row per ledger row that requires action at 4.0 — 113 in total. `Old` and `New` are the
ledger's own locators. An em dash in `New` means the ledger names no successor **locator**; it
does not mean no action is required, so read the `Fix` cell.

| Row | Group | Old | New | Fix |
|---|---|---|---|---|
| M-020 | aggregate-postfit | `diff_diff:CallawaySantAnna.fit[aggregate]` | `diff_diff:CallawaySantAnnaResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits replay the fit-time bootstrap post-fit - see the aggregation section. |
| M-021 | aggregate-postfit | `diff_diff:ImputationDiD.fit[aggregate]` | `diff_diff:ImputationDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits compute the surface at fit and `aggregate()` relays it - see the aggregation section. |
| M-022 | aggregate-postfit | `diff_diff:TwoStageDiD.fit[aggregate]` | `diff_diff:TwoStageDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits compute the surface at fit and `aggregate()` relays it - see the aggregation section. |
| M-023 | aggregate-postfit | `diff_diff:EfficientDiD.fit[aggregate]` | `diff_diff:EfficientDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits replay the fit-time bootstrap post-fit - see the aggregation section. |
| M-024 | aggregate-postfit | `diff_diff:StackedDiD.fit[aggregate]` | `diff_diff:StackedDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. |
| M-025 | aggregate-postfit | `diff_diff:ContinuousDiD.fit[aggregate]` | `diff_diff:ContinuousDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits compute the surface at fit and `aggregate()` relays it - see the aggregation section. |
| M-026 | aggregate-postfit | `diff_diff:ChaisemartinDHaultfoeuille.fit[aggregate]` | `diff_diff:ChaisemartinDHaultfoeuilleResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. |
| M-027 | aggregate-postfit | `diff_diff:HeterogeneousAdoptionDiD.fit[aggregate]` | `diff_diff:HeterogeneousAdoptionDiDResults.aggregate` | Move `aggregate=` off `fit()` onto post-fit `results.aggregate(...)`. |
| M-117 | aggregate-postfit | `diff_diff:CallawaySantAnna.fit[balance_e]` | `diff_diff:CallawaySantAnnaResults.aggregate[balance_e]` | Move `balance_e=` off `fit()` onto post-fit `results.aggregate(...)`. Bootstrapped fits replay the fit-time bootstrap post-fit - see the aggregation section. |
| M-118 | aggregate-postfit | `diff_diff:ImputationDiD.fit[balance_e]` | `diff_diff:ImputationDiDResults.aggregate[balance_e]` | Move `balance_e=` off `fit()` onto post-fit `results.aggregate(...)`. Unavailable on a bootstrapped fit (`NotImplementedError`; re-fit with `n_bootstrap=0`) - see the aggregation section. |
| M-119 | aggregate-postfit | `diff_diff:TwoStageDiD.fit[balance_e]` | `diff_diff:TwoStageDiDResults.aggregate[balance_e]` | Move `balance_e=` off `fit()` onto post-fit `results.aggregate(...)`. Unavailable on a bootstrapped fit (`NotImplementedError`; re-fit with `n_bootstrap=0`) - see the aggregation section. |
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
| M-050 | field-flip | `diff_diff:CallawaySantAnnaResults.overall_att` | `diff_diff:CallawaySantAnnaResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-051 | field-flip | `diff_diff:SunAbrahamResults.overall_att` | `diff_diff:SunAbrahamResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-052 | field-flip | `diff_diff:ImputationDiDResults.overall_att` | `diff_diff:ImputationDiDResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-053 | field-flip | `diff_diff:TwoStageDiDResults.overall_att` | `diff_diff:TwoStageDiDResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-054 | field-flip | `diff_diff:StackedDiDResults.overall_att` | `diff_diff:StackedDiDResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-055 | field-flip | `diff_diff:EfficientDiDResults.overall_att` | `diff_diff:EfficientDiDResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-056 | field-flip | `diff_diff:WooldridgeDiDResults.overall_att` | `diff_diff:WooldridgeDiDResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-057 | field-flip | `diff_diff:ChaisemartinDHaultfoeuilleResults.overall_att` | `diff_diff:ChaisemartinDHaultfoeuilleResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-058 | field-flip | `diff_diff:ContinuousDiDResults.overall_att` | `diff_diff:ContinuousDiDResults.att` | Read and construct with `att`; `overall_att` is a read-only FutureWarning property since 4.0; 3.x pickles load unchanged. |
| M-070 | function-wrappers | `diff_diff:imputation_did` | — | Call the estimator class directly instead of `imputation_did`. |
| M-071 | function-wrappers | `diff_diff:two_stage_did` | — | Call the estimator class directly instead of `two_stage_did`. |
| M-072 | function-wrappers | `diff_diff:stacked_did` | — | Call the estimator class directly instead of `stacked_did`. |
| M-073 | function-wrappers | `diff_diff:trop` | — | Call the estimator class directly instead of `trop`. |
| M-074 | function-wrappers | `diff_diff:synthetic_control` | — | Call the estimator class directly instead of `synthetic_control`. |
| M-075 | function-wrappers | `diff_diff:triple_difference` | — | Call the estimator class directly instead of `triple_difference`. |
| M-076 | function-wrappers | `diff_diff:bacon_decompose` | — | Call the estimator class directly instead of `bacon_decompose`. |
| M-077 | function-wrappers | `diff_diff:chaisemartin_dhaultfoeuille` | — | Call the estimator class directly instead of `chaisemartin_dhaultfoeuille`. |
| M-151 | lwdid-results-contract | `diff_diff:LWDiDResults.pvalue` | `diff_diff:LWDiDResults.p_value` | Read `.p_value`; the alias warns from 4.0 and is removed at 5.0. |
| M-152 | lwdid-results-contract | `diff_diff:LWDiDResults.ci` | `diff_diff:LWDiDResults.conf_int` | Read `.conf_int`; the alias warns from 4.0 and is removed at 5.0. |
| M-013 | merge-ddd | `diff_diff:StaggeredTripleDifference` | `diff_diff:TripleDifference.fit[first_treat]` | See the merges section for the worked before/after. |
| M-014 | merge-ddd | `diff_diff:StaggeredTripleDiffResults` | — | Both designs return `TripleDifferenceResults` (`results.design` says which); read `att` / `se` / ... where you read `overall_*`; event-study and cohort surfaces are `results.aggregate(...)`. 3.x staggered pickles cannot load. |
| M-085 | merge-ddd | `diff_diff:TripleDifference.fit[time]` | — | In 2x2x2 mode, `time=` raises - pass `post=` for the 0/1 post dummy. `time=` now means the staggered calendar column only. |
| M-140 | merge-ddd | `diff_diff:TripleDifference.fit[aggregate]` | `diff_diff:TripleDifferenceResults.aggregate` | Drop the keyword and call `results.aggregate('event_study')` / `aggregate('group')` / `aggregate('simple')` post-fit; the Eq. 4.14 overall (3.x `overall_att_es`) is `aggregate('simple', weights='event_time')`. |
| M-141 | merge-ddd | `diff_diff:TripleDifference.fit[balance_e]` | `diff_diff:TripleDifferenceResults.aggregate[balance_e]` | Pass `balance_e=` to `results.aggregate('event_study', balance_e=...)` or to `aggregate('simple', weights='event_time', balance_e=...)`. |
| M-010 | merge-mpd | `diff_diff:MultiPeriodDiD` | `diff_diff:TwoWayFixedEffects.fit[event_study]` | See the merges section for the worked before/after. |
| M-011 | merge-mpd | `diff_diff:MultiPeriodDiDResults` | — | Read the unified event-study surface on the merged `TwoWayFixedEffects` results instead of this container. |
| M-012 | merge-mpd | `diff_diff:PeriodEffect` | — | Superseded by the unified event-study representation - read the event-study surface rather than per-period `PeriodEffect` objects. |
| M-016 | merge-mpd | `diff_diff:EventStudyResults.period_effects` | — | Not removed at 4.0: `period_effects` is a `FutureWarning` property view over the unified event-study surface (calendar surfaces only; keyed by calendar period, reference omitted), removed at 5.0. Read `to_dataframe()` or the columnar arrays instead. |
| M-083 | merge-mpd | `diff_diff:TwoWayFixedEffects.fit[time]` | — | In static mode (`event_study=False`), `time=` raises - pass `post=` for the 0/1 post dummy. `time=` now means the calendar column only. |
| M-015 | merge-qdid | `diff_diff:QDiD` | `diff_diff:ChangesInChanges[method]` | See the merges section for the worked before/after. |
| M-143 | merge-qdid | `diff_diff:ChangesInChangesResults.estimator` | `diff_diff:ChangesInChangesResults.method` | See the merges section for the worked before/after. |
| M-148 | obligation-sc-inner-budget | `diff_diff:SyntheticControl[inner_max_iter]` | — | Drop it. The donor weights are solved exactly (one non-negative least squares) since #842, so there is no inner iteration budget; passing it warns `DeprecationWarning` at 4.0 and is a `TypeError` at 5.0. The former Frank-Wolfe budget is NOT a knob to carry over to `optimizer_options=` (that is the outer `V` search). |
| M-149 | obligation-sc-inner-budget | `diff_diff:SyntheticControl[inner_min_decrease]` | — | Drop it. An exact inner solve has no stopping tolerance; passing it warns `DeprecationWarning` at 4.0 and is a `TypeError` at 5.0. Do NOT translate it into `optimizer_options={"xatol": ...}` - that tolerance governs the outer `V` search, a different quantity. |
| M-001 | obligation-sdid-params | `diff_diff:SyntheticDiD[lambda_reg]` | — | Drop it. It has been IGNORED since 3.0.0, so 3.x already auto-computes regularization - do NOT copy its value into `zeta_omega=`, which activates an override that was inert and changes weights, ATT and inference. Set `zeta_omega=` only as a deliberate new choice. |
| M-002 | obligation-sdid-params | `diff_diff:SyntheticDiD[zeta]` | — | Drop it. It has been IGNORED since 3.0.0, so 3.x already auto-computes regularization - do NOT copy its value into `zeta_lambda=`, which activates an override that was inert and changes weights, ATT and inference. Set `zeta_lambda=` only as a deliberate new choice. |
| M-003 | obligation-sdid-params | `diff_diff:SyntheticDiDResults.placebo_effects` | `diff_diff:SyntheticDiDResults.variance_effects` | Read `variance_effects` instead of `placebo_effects`. |
| M-007 | obligation-warning-retirements | `diff_diff.estimators` | — | The MultiPeriodDiD `e=-1` transition FutureWarning is no longer emitted (the class is gone; the event-study mode defaults to the last pre-period silently); drop any warning filter that suppressed it. |
| M-080 | policy-auto-cluster | `diff_diff:SpilloverDiD[cluster]` | — | `SpilloverDiD` now clusters at `unit` when `cluster=` is omitted (TWFE's carve-outs mirrored: explicit `cluster=` wins, dropped on the Conley path, never injected as a survey PSU; its p-values/CIs use t(G − 1) since [M-154], item 7c); pass `cluster=False` for the 3.x observation-level HC standard errors (`None`/omission means auto-cluster, not off). Every other panel estimator already clustered at unit or is unit-level by construction, so its numbers do not move. `cluster=False` is also newly accepted on `TwoWayFixedEffects`/`SunAbraham`/`WooldridgeDiD`/`LPDiD` (plain HC sandwich - new behaviour; residual-df t), as a no-op on the by-construction estimators, on `ImputationDiD`/`TwoStageDiD` and on the 2x2 `DifferenceInDifferences` / 2x2x2 `TripleDifference` surfaces; `StackedDiD` still rejects it. Rejected with `inference="wild_bootstrap"` (DiD/TWFE), `WooldridgeDiD(n_bootstrap > 0)` and `vcov_type="hc2_bm"` on `SunAbraham`/`WooldridgeDiD`. `cluster` accepts `str | None | False` only ([M-155]). |
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
| M-150 | renames-level | `diff_diff:WooldridgeDiDResults.to_dataframe[level]=event` | `diff_diff:WooldridgeDiDResults.to_dataframe[level]=event_study` | Change the accepted value: `"event"` becomes `"event_study"`. |
| M-030 | renames-post | `diff_diff:DifferenceInDifferences.fit[time]` | `diff_diff:DifferenceInDifferences.fit[post]` | Rename the keyword: `time=` becomes `post=`. |
| M-137 | renames-post | `diff_diff:permutation_test[time]` | `diff_diff:permutation_test[post]` | Rename the keyword: `time=` becomes `post=`. |
| M-138 | renames-post | `diff_diff:leave_one_out_test[time]` | `diff_diff:leave_one_out_test[post]` | Rename the keyword: `time=` becomes `post=`. |
| M-045 | renames-robust-drop | `diff_diff:DifferenceInDifferences[robust]` | — | Drop `robust=True` (already the `hc1` default); `robust=False` becomes `vcov_type="classical"`. |
| M-046 | renames-robust-drop | `diff_diff:TripleDifference[robust]` | — | Drop it. `TripleDifference` accepts only `vcov_type="hc1"`, so `robust=` never changed its inference - do NOT write `vcov_type="classical"`, which raises. |
| M-047 | renames-robust-drop | `diff_diff:HeterogeneousAdoptionDiD[robust]` | — | HAD's legacy default is non-robust: `robust=True` becomes `vcov_type="hc1"`; drop `robust=False`. |
| M-115 | renames-robust-drop | `diff_diff:LinearRegression[robust]` | — | Drop `robust=True` (already the `hc1` default); `robust=False` becomes `vcov_type="classical"`. |
| M-093 | results-contract | `diff_diff:CallawaySantAnnaResults.event_study_effects` | `diff_diff:EventStudyResults` | The sentinel is retired - read the unified `EventStudyResults` surface instead of the raw field. |
