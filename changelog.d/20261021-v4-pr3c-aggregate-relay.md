### Removed
- **The last fit-time `aggregate=` / `balance_e=` keywords are gone (ledger rows flipped to
  `removed`).** `fit(aggregate=)` is removed from `ImputationDiD` (`M-021`), `TwoStageDiD`
  (`M-022`) and `ContinuousDiD` (`M-025`), `fit(balance_e=)` from `ImputationDiD` (`M-118`)
  and `TwoStageDiD` (`M-119`). With PR 3b's removals on CallawaySantAnna / EfficientDiD /
  StackedDiD / dCDH / HAD this completes the aggregate-postfit group: every adopter aggregates
  through post-fit `results.aggregate(type, balance_e=)` only, and `TripleDifference`'s
  staggered-mode keywords are the one surviving fit-time `aggregate=` (`M-140`/`M-141`,
  canonical until the DDD container unification). With the keywords go ContinuousDiD's
  no-underscore `"eventstudy"` spelling and its fit-time value validation, the dead analytic
  fit-time event-study inference arm in `continuous_did.py`, and ImputationDiD's unreachable
  pretrends+replicate fit-time gate.

### Breaking Changes
- Passing `aggregate=` or `balance_e=` to `ImputationDiD.fit`, `TwoStageDiD.fit` or
  `ContinuousDiD.fit` now raises Python's native `TypeError: ... unexpected keyword argument`.
- **Positional slots after the removed parameters shift left.** `ImputationDiD` and
  `TwoStageDiD`: `survey_design` 9 -> 7; `ContinuousDiD`: `survey_design`, `covariates`
  8/9 -> 7/8. Pass these by keyword.
- `aggregate('event_study', balance_e=k)` on a BOOTSTRAPPED `ImputationDiD` / `TwoStageDiD`
  fit raises `NotImplementedError` (the relayed surface was computed unrestricted; re-fit with
  `n_bootstrap=0` for a balanced window). Analytic fits recompute with `balance_e` as before.
- A bootstrapped `ImputationDiD` / `TwoStageDiD` / `ContinuousDiD` fit WITH bootstrap inference
  but WITHOUT a stored surface - a 3.9-3.13 bootstrapped plain-fit pickle, or a fit whose
  fail-soft materialization dropped it - raises `ValueError` from `aggregate('event_study')` /
  `aggregate('group')` (previously `NotImplementedError` from the bootstrap gate); the message
  names the refit. A 3.9-3.13 bootstrapped pickle fitted with `balance_e=k` stores a restricted
  surface and the relay returns it as stored.

### Changed
- **Bootstrapped `ImputationDiD` / `TwoStageDiD` / `ContinuousDiD` fits relay their event-study
  (and, on Imputation/TwoStage, group) surface.** The deprecated keyword was the only route to
  bootstrap event-study inference there (no per-draw state is retained for a replay), so a fit
  with `n_bootstrap > 0` now ALWAYS computes those surfaces at fit - exactly what
  `fit(aggregate="all")` computed, `balance_e=None` - stores them in the legacy
  `event_study_effects` / `group_effects` fields, and `aggregate('event_study')` /
  `aggregate('group')` RELAYS them (kit-snapshot alpha / anticipation / survey metadata; NaN
  `df` column; `vcov` `None`) instead of raising `NotImplementedError`. Analytic fits are
  unchanged (legacy fields `None`, kit recompute on demand). Storage rule: the legacy fields
  hold a surface IFF the fit carries bootstrap inference. On Imputation/TwoStage a bootstrapped
  fit that ends analytical (zero valid draws, total bootstrap failure) clears them and
  `aggregate()` recomputes analytically. ContinuousDiD's bootstrapped kit is scalars-only, so
  no analytical fallback exists there: a bootstrapped fit with no post-treatment cells never
  runs its bootstrap (overall ATT already NaN with a `UserWarning`), stores no surface, and
  `aggregate('event_study')` raises the absent-surface `ValueError`.
  - Materialization is FAIL-SOFT with a one-shot overall-only retry: an exception while
    building the surfaces, or a bootstrap attempt that raises / returns `None` with surfaces
    present, leaves them `None`, emits one `UserWarning` ("... could not be materialized ...")
    and re-runs the bootstrap overall-only (a seeded fit reproduces the first attempt's weight
    draws), so a plain bootstrapped fit's headline quintet and its success/abort outcome are
    unchanged by the removal. ContinuousDiD bootstrap failures still propagate as before.
  - Consequences on bootstrapped plain fits: the aggregators' own `UserWarning`s (ImputationDiD
    Prop-5 unidentified horizons, empty horizons, LOO single-unit auxiliaries; TwoStageDiD
    Stage-1/Stage-2 singular or rank-deficient fallbacks) and the per-target bootstrap-stats
    `RuntimeWarning`s now fire where a plain fit never emitted them; `summary()` prints the
    event-study/group tables and `to_dataframe(level="event_study"|"group")` returns frames
    instead of raising; `DiagnosticReport` / `BusinessReport` run heterogeneity (Imputation /
    TwoStage on the relayed group dict, ContinuousDiD on the event study) and parallel trends
    (Imputation / TwoStage) on the relayed surface instead of skipping with
    `NotImplementedError`; `plot_event_study(results)` plots them natively. Measured cost (199
    draws, 300 units x 12 periods): Imputation 0.01s -> 0.08s, TwoStage 0.01s -> 0.04s,
    Continuous unchanged at 0.05s; all three default to `n_bootstrap=0`.
- `ImputationDiD`'s inert-`df_convention` warning predicate is now exact: the per-lead inference
  is reachable iff `pretrends=True` and the design is not replicate-weighted (`M-127`); the
  unreachable pretrends+replicate fit-time gate is deleted (the combination is still rejected
  post-fit by `aggregate('event_study')`).
- Runtime messages (`plot_event_study`, the results-class "not stored" hints,
  `_reporting_helpers`, `DiagnosticReport` docstrings), `practitioner_next_steps` advice for
  bootstrapped ImputationDiD / TwoStageDiD / ContinuousDiD fits, the `llms*.txt` guides, the
  ContinuousDiD API page and tutorial, the migration guide and the performance docs describe
  the relay instead of steering users to the removed fit-time route.

### Internal
- `tests/test_aggregate_contract.py`: the remaining `Test*Shim` classes, fit-time fixtures and
  fail-closed bootstrap pins are retired for `Test*Removal` pins (TypeError, signature,
  positional-slot binding) and relay pins - the relayed surfaces equal the stored legacy dicts
  row-for-row and match the per-backend bootstrap oracles - plus failure-injection pins for the
  fail-soft materialization and the storage rule. The estimator suites, `DiagnosticReport`,
  `event_study_surface` (bootstrapped Imputation / TwoStage / Continuous enrolled as genuine
  native producers) and practitioner tests take their relay pins; the module-level
  FutureWarning filters for the fit-time route are gone. Naming-guard allowlist entries for the
  guides' post-fit `aggregate` vocabulary are finalized. Ledger row `M-093` gains the fit-side
  table of which producers populate the legacy fields; `M-127` records the exact predicate.
  `docs/v4-design.md` section 6 carries the relay amendment and the HYBRIDS paragraph; section 9
  marks sub-item 3c shipped.
