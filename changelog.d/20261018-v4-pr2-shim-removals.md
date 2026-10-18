### Removed
- **The 3.9 rename shims are gone (ledger rows flipped to `removed`).** Old keyword spellings
  now raise `TypeError`: `DifferenceInDifferences.fit(time=)` -> `post=` (`M-030`),
  `permutation_test(time=)` / `leave_one_out_test(time=)` -> `post=` (`M-137`, `M-138`),
  `WooldridgeDiD.fit(cohort=)` -> `first_treat=` (`M-032`), `ChaisemartinDHaultfoeuille.fit(group=,
  controls=)` -> `unit=`, `covariates=` (`M-033`, `M-034`), `twowayfeweights(group=)` -> `unit=`
  (`M-097`), the `*_col` spellings on `HeterogeneousAdoptionDiD.fit`, `RegressionDiscontinuity.fit`,
  `RDPlot.fit`, `joint_pretrends_test`, `joint_homogeneity_test`, `did_had_pretest_workflow`
  (`M-035`..`M-042`, `M-088`, `M-089`, `M-098`..`M-112`) and `trim_weights(weight_col=)` ->
  `weights=` (`M-113`), `StackedDiD(clean_control=)` -> `control_group=` (`M-043`),
  `WooldridgeDiDResults.to_dataframe(aggregation=)` -> `level=` (`M-044`) and
  `summary(aggregation)` (`M-087`; `summary(alpha=None)` is now the positional signature),
  `ContinuousDiD(covariates=)` -> `fit(covariates=)` (`M-084`). The `"event"` value spelling is
  rejected with `ValueError` on `WooldridgeDiDResults.aggregate(type=)`, `.to_dataframe(level=)` and
  `LPDiDResults.to_dataframe(level=)` - use `"event_study"` (`M-086`, `M-150`, `M-136`).
- **`robust=` removed** from `DifferenceInDifferences` (and so `TwoWayFixedEffects`,
  `MultiPeriodDiD`), `TripleDifference`, `HeterogeneousAdoptionDiD` and `LinearRegression`
  (`M-045`..`M-047`, `M-115`); `vcov_type=` is the single selector. The legacy-CR1 remap that
  turned `robust=False` + clustering into `hc1` went with it. Positional constructor slots after
  the removed parameter shift left by one (DiD slot 1, LinearRegression/TripleDifference slot 2,
  HeterogeneousAdoptionDiD slot 6); `SyntheticDiD`'s inert `lambda_reg`/`zeta` (`M-001`, `M-002`,
  constructor slots 7-8) are removed too, shifting `vcov_type` and the `conley_*` parameters two
  slots left. Pass these by keyword.
- **Results-field aliases removed**: `SyntheticDiDResults.placebo_effects` (`M-003`; use
  `variance_effects`), `RegressionDiscontinuityResults.treatment_col` (`M-094`; `takeup`),
  `StackedDiDResults.clean_control` and `StackedDiD.clean_control` (`M-095`, `M-043`;
  `control_group`), `ChaisemartinDHaultfoeuilleResults.groups` (`M-114`; `units`),
  `ChangesInChangesResults.estimator` (`M-143`; `method`). Their serialized dual keys are gone from
  `to_dict()` (`treatment_col`, `clean_control`, `estimator`, Wooldridge's
  `aggregation_weights["event"]`); 3.x pickles still load (`__setstate__` key migrations kept).
- **The eight module-level wrapper functions are removed** (`M-070`..`M-077`): `imputation_did`,
  `two_stage_did`, `stacked_did`, `trop`, `synthetic_control`, `triple_difference`,
  `bacon_decompose`, `chaisemartin_dhaultfoeuille`. Call the estimator classes. `diff_diff.trop`,
  `diff_diff.synthetic_control`, `diff_diff.stacked_did` and `diff_diff.chaisemartin_dhaultfoeuille`
  now name the submodules (as `diff_diff.bacon` always did).
- **The dieted aliases are removed** (`M-132`..`M-134`): `CDiD` -> `ContinuousDiD`, `Stacked` ->
  `StackedDiD`, `Gardner` -> `TwoStageDiD`; the module `__getattr__` warning shim (`M-135`) is gone.

### Changed
- **`WooldridgeDiDResults.summary(alpha=None)`** renders the headline (simple) row only; the
  group / calendar / event-study tables it used to render on request - and their `[pre]`/`[antic]`
  relative-period labels - are no longer part of `summary()`. Use `aggregate(...)` /
  `to_dataframe(level=...)`; `anticipation` is still reported on the results object.
  `to_dataframe(level=)` now rejects an unknown level with `ValueError` instead of returning an
  empty frame.
- `practitioner_next_steps` no longer recognises duck-typed results that carry only the legacy
  `estimator` attribute; `ChangesInChangesResults.method` is the routing field.

### Breaking Changes
- **`BusinessReport.to_dict()` schema `2.0` -> `3.0`**: the StackedDiD `assumption` block no longer
  carries the legacy `clean_control` key (read `control_group`). `DiagnosticReport`'s schema stays
  `2.0`.

### Internal
- `tests/test_v4_matrix.py` removed-status probes: a function row that resolves to a submodule
  counts as absent; a param row absent from the explicit signature counts as removed despite
  `**kwargs`. Naming-guard allowlist extended for the kept body-local aliases, `__setstate__` keys
  and shared vocabulary. The `diff_diff._deprecation` helpers survive for the remaining
  M-031/M-082, PR-3 and 5.0-window shims.
