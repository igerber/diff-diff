### Breaking Changes

- **4.0 cut, item 9b: the compatibility shims scheduled for a 5.0 removal are removed at 4.0** (user
  decision - no 4.x debt; none of these `FutureWarning` shims ever reached a release, so the released
  3.x names break without a warned release - `docs/migration-4.0.md` is the notice). The 3.x
  `overall_att` / `overall_se` / `overall_t_stat` / `overall_p_value` / `overall_conf_int` names on
  `CallawaySantAnnaResults` (and `DMLDiDResults`), `SunAbrahamResults`, `ImputationDiDResults`,
  `TwoStageDiDResults`, `StackedDiDResults`, `EfficientDiDResults`, `WooldridgeDiDResults`,
  `ChaisemartinDHaultfoeuilleResults` and - all nine spellings, `overall_att_*` included -
  `ContinuousDiDResults` raise `AttributeError`; read the canonical `att` / `se` / `t_stat` /
  `p_value` / `conf_int` (`M-050`..`M-058`). 3.x pickles still load (the `__setstate__` key migration
  stays). `EventStudyResults.period_effects` is gone; read `to_dataframe()` or the columnar arrays
  (`M-016`). `LWDiDResults.pvalue` / `.ci` are gone; read `.p_value` / `.conf_int` (`M-151`, `M-152`).
- **`SyntheticControl(inner_max_iter=, inner_min_decrease=)` are removed** (`M-148`, `M-149`; the
  `DeprecationWarning` shipped in 3.13.0): passing either is a `TypeError`, and the positional
  parameters after them (`standardize`, `alpha`, `seed`, `v_cv_t0`) move two slots left - pass
  them by keyword.
- **`run_placebo_test` / `run_all_placebo_tests`: `time` is the calendar column only and
  `post_periods` is required for every `test_type`** (`M-156`, `M-157`; spec section 8 rule 1). The
  0/1 post dummy the permutation and leave-one-out tests need is derived as
  `data[time].isin(post_periods)`; a missing `time` column, missing/empty `post_periods`, missing
  values in `time`, `post_periods` values absent from the column, or a pre-existing `_placebo_post`
  column raise `ValueError` (on the suite wrapper before any test runs). Calls that passed the 0/1
  indicator as `time=` with `post_periods=[1]` return identical numbers; calls that omitted
  `post_periods` must add it.

### Fixed

- `run_all_placebo_tests` and `run_placebo_test(test_type="permutation" | "leave_one_out")` no
  longer fail on a multi-period calendar column: through 3.x the wrappers forwarded `time` itself as
  the 0/1 post dummy, so `DifferenceInDifferences.fit` rejected it and the suite silently reported
  the two tests as error dicts (tutorial 04's placebo cell printed `ERROR` lines). The dummy is now
  derived from `post_periods`.

### Internal

- Ledger rows `M-016`, `M-050`..`M-058`, `M-151`, `M-152` flip to `removed` at 4.0 with
  `deprecated_in: null` (no release warned); `M-148`/`M-149` flip to `removed` with
  `deprecated_in: 3.13.0`; new behavior rows `M-156`/`M-157` (`tests/test_diagnostics.py`); the
  ledger header, spec ladder, section 5/8/9/11 notes and the phase-5 roster carry dated amendments.
  `diff_diff._deprecation` is reduced to the pickle-migration (`LEGACY_OVERALL_QUINTET`,
  `migrate_renamed_state_keys`, `migrate_event_study_reference_sentinel`) and required-arg
  (`NOT_SUPPLIED`, `require_arg`) helpers - `deprecated_field_property`, `resolve_renamed_kwarg`,
  `warn_deprecated_kwarg` and `deprecated_kwarg_message` are deleted with their last consumers.
  With the descriptors gone, assigning a legacy name to a results instance creates a shadow
  attribute instead of raising (the non-slotted dataclasses) - the naming guard's consumer sweep
  is the in-repo defense. Naming guard: the `DMLDiDResults.overall_att` surface entry and the
  `_deprecation.py` `placebo_effects` consumer entry are deleted, the two placebo wrappers join the
  rule-1 calendar `time` inventory, and the retired token families' remaining in-repo hits
  (body-local `overall_att` names, the kept `__setstate__` key maps, methodology history) are
  allowlisted. API autosummary stubs regenerated; TODO.md's placebo-wrapper redesign row closed;
  tutorial 17 re-executed (its stored practitioner snippet echoed the old `time='post'` advice).
