### Removed
- **The fit-time `aggregate=` / `balance_e=` keywords are gone (ledger rows flipped to
  `removed`).** `fit(aggregate=)` is removed from `CallawaySantAnna` (`M-020`), `ImputationDiD`
  (`M-021`), `TwoStageDiD` (`M-022`), `EfficientDiD` (`M-023`), `StackedDiD` (`M-024`),
  `ContinuousDiD` (`M-025`), `ChaisemartinDHaultfoeuille` (`M-026`) and
  `HeterogeneousAdoptionDiD` (`M-027`), `fit(balance_e=)` from `CallawaySantAnna` (`M-117`),
  `ImputationDiD` (`M-118`), `TwoStageDiD` (`M-119`) and `EfficientDiD` (`M-120`), and
  `aggregate=` from `did_had_pretest_workflow` (`M-139`; the `HADPretestReport.aggregate`
  output field survives). The successor everywhere is post-fit
  `results.aggregate(type, balance_e=)`; HAD selects its mode from the panel shape. With the
  keywords go ContinuousDiD's no-underscore `"eventstudy"` spelling, StackedDiD's fit-time
  `group`/`all`/unknown-value `ValueError`s (`aggregate('group')` already fails closed in the
  mixin), and the private `_VALID_AGGREGATES` sets in `had.py` / `had_pretests.py`
  (`_infer_aggregate_mode` is the sole selector).
- **CallawaySantAnna and EfficientDiD compute no event-study/group surface at fit.** Their
  legacy `event_study_effects` / `group_effects` fields - and CS's `cband_crit_value`,
  `event_study_vcov`, `event_study_vcov_index`, `event_study_df` - are `None` after EVERY fit;
  the post-fit replay (`aggregate('event_study'|'group', balance_e=)`) is the only bootstrap
  route and is unchanged.

### Breaking Changes
- Passing `aggregate=` or `balance_e=` to any of the eight `fit()` methods (or `aggregate=` to
  `did_had_pretest_workflow`) now raises Python's native
  `TypeError: ... unexpected keyword argument`. `TripleDifference`'s staggered-mode
  `aggregate=`/`balance_e=` are untouched (`M-140`/`M-141`, canonical until the DDD container
  unification).
- **Positional slots after the removed parameters shift left.** `CallawaySantAnna`,
  `ImputationDiD`, `TwoStageDiD`: `survey_design` 9 -> 7; `EfficientDiD`: `survey_design`
  9 -> 7, `store_eif` 10 -> 8; `StackedDiD` (`aggregate` sat before them): `population`,
  `survey_design`, `covariates` 7/8/9 -> 6/7/8; `ContinuousDiD`: `survey_design`, `covariates`
  8/9 -> 7/8; `ChaisemartinDHaultfoeuille`: `L_max`, `covariates`, `trends_linear`,
  `trends_nonparam`, `honest_did`, `heterogeneity`, `design2`, `survey_design` each move one
  slot left (7..14 -> 6..13). `HeterogeneousAdoptionDiD.fit` shifts nothing (everything after the
  removed slot is keyword-only; a seventh positional now raises). Pass these by keyword.
- `CallawaySantAnnaResults.to_dataframe(level="event_study"|"group")` and the `EfficientDiDResults`
  twins raise on every fresh fit (the legacy fields are always `None`); use
  `results.aggregate(...).to_dataframe()`. The column-schema rewire of these levels onto the
  container is `M-093`'s (PR 8).

### Changed
- **Bootstrapped `ImputationDiD` / `TwoStageDiD` / `ContinuousDiD` fits relay their event-study
  (and, on Imputation/TwoStage, group) surface.** The deprecated keyword was the only route to
  bootstrap event-study inference there (no per-draw state is retained for a replay), so a fit
  with `n_bootstrap > 0` now ALWAYS computes those surfaces at fit - exactly what
  `fit(aggregate="all")` computed, `balance_e=None` - stores them in the legacy fields, and
  `aggregate('event_study')` / `aggregate('group')` RELAYS them (kit-snapshot alpha /
  anticipation / survey metadata; NaN `df` column; `vcov` `None`) instead of raising
  `NotImplementedError`. Analytic fits are unchanged (legacy fields `None`, kit recompute on
  demand). `aggregate('event_study', balance_e=k)` on such a bootstrapped fit raises
  `NotImplementedError` (re-fit with `n_bootstrap=0`). Storage rule: the legacy fields hold a
  surface IFF the fit carries bootstrap inference - a bootstrapped fit that ends analytical
  (zero valid draws, total bootstrap failure) clears them and `aggregate()` recomputes
  analytically, while a fit WITH bootstrap inference and no stored surface (a 3.9-3.13 pickle,
  a fail-soft drop) raises a `ValueError` naming the refit. Known limitation: a 3.9-3.13
  bootstrapped pickle fitted with `balance_e=k` stores a restricted surface and the relay
  returns it as stored.
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
- Runtime messages, `practitioner_next_steps` advice, the four `llms*.txt` guides, the API
  docs, the migration guide and the benchmarks no longer steer users to the fit-time route;
  `CallawaySantAnna`'s class docstring, `plot_event_study`, `honest_did` / `pretrends` and the
  results-class "not stored" messages point at `results.aggregate('event_study')`.

### Internal
- Per-backend (NumPy / Rust weight stream) oracle pins for the removed route live in
  `tests/test_aggregate_contract.py` (captured from the unmodified tree by
  `tests/_capture_v4_aggregate_oracles.py`), with failure-injection pins for the fail-soft
  materialization; `tests/helpers/legacy_surface.py` provides `es_dict` / `group_dict` /
  `legacy_standin` for the migrated consumer tests. Naming-guard allowlist entries for the
  guides' canonical post-fit `aggregate` vocabulary. Ledger row `M-093` gains the fit-side
  table of which producers populate the legacy fields. `warn_deprecated_kwarg` loses its last
  library caller and is retained (spec section 8 rule 11) with a direct unit pin.
