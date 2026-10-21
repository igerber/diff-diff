### Removed
- **The fit-time `aggregate=` / `balance_e=` keywords are gone on the estimators whose post-fit
  route already existed (ledger rows flipped to `removed`).** `fit(aggregate=)` is removed from
  `CallawaySantAnna` (`M-020`), `EfficientDiD` (`M-023`), `StackedDiD` (`M-024`),
  `ChaisemartinDHaultfoeuille` (`M-026`) and `HeterogeneousAdoptionDiD` (`M-027`),
  `fit(balance_e=)` from `CallawaySantAnna` (`M-117`) and `EfficientDiD` (`M-120`), and
  `aggregate=` from `did_had_pretest_workflow` (`M-139`; the `HADPretestReport.aggregate`
  output field survives). The successor is post-fit `results.aggregate(type, balance_e=)`; HAD
  selects its mode from the panel shape. With the keywords go StackedDiD's fit-time
  `group`/`all`/unknown-value `ValueError`s (`aggregate('group')` already fails closed in the
  mixin) and the private `_VALID_AGGREGATES` sets in `had.py` / `had_pretests.py`
  (`_infer_aggregate_mode` is the sole selector). `ImputationDiD`, `TwoStageDiD` and
  `ContinuousDiD` keep their deprecated keywords until the next cut PR adds the
  bootstrapped-fit relay.
- **CallawaySantAnna and EfficientDiD compute no event-study/group surface at fit.** Their
  legacy `event_study_effects` / `group_effects` fields - and CS's `cband_crit_value`,
  `event_study_vcov`, `event_study_vcov_index`, `event_study_df` - are `None` after EVERY fit;
  the post-fit replay (`aggregate('event_study'|'group', balance_e=)`) is the only bootstrap
  route and is unchanged.

### Breaking Changes
- Passing `aggregate=` or `balance_e=` to `CallawaySantAnna.fit`, `EfficientDiD.fit`,
  `StackedDiD.fit`, `ChaisemartinDHaultfoeuille.fit` or `HeterogeneousAdoptionDiD.fit` (or
  `aggregate=` to `did_had_pretest_workflow`) now raises Python's native
  `TypeError: ... unexpected keyword argument`. `TripleDifference`'s staggered-mode
  `aggregate=`/`balance_e=` are untouched (`M-140`/`M-141`, canonical until the DDD container
  unification).
- **Positional slots after the removed parameters shift left.** `CallawaySantAnna`:
  `survey_design` 9 -> 7; `EfficientDiD`: `survey_design` 9 -> 7, `store_eif` 10 -> 8;
  `StackedDiD` (`aggregate` sat before them): `population`, `survey_design`, `covariates`
  7/8/9 -> 6/7/8; `ChaisemartinDHaultfoeuille`: `L_max`, `covariates`, `trends_linear`,
  `trends_nonparam`, `honest_did`, `heterogeneity`, `design2`, `survey_design` each move one
  slot left (7..14 -> 6..13). `HeterogeneousAdoptionDiD.fit` shifts nothing (everything after
  the removed slot is keyword-only; a seventh positional now raises). Pass these by keyword.
- `CallawaySantAnnaResults.to_dataframe(level="event_study"|"group")` and the
  `EfficientDiDResults` twins raise on every fresh fit (the legacy fields are always `None`);
  use `results.aggregate(...).to_dataframe()`. The column-schema rewire of these levels onto the
  container is `M-093`'s (PR 8).

### Changed
- Runtime messages, the four `llms*.txt` guides, the API docs, the migration guide and the
  benchmarks no longer steer users to the fit-time route on these five estimators;
  `CallawaySantAnna`'s class docstring, `plot_event_study`, `honest_did` / `pretrends` and the
  results-class "not stored" messages point at `results.aggregate('event_study')`.

### Internal
- `tests/test_aggregate_contract.py`: the `Test*Shim` classes, fit-time fixtures and
  inertness comparisons for the five estimators are retired in favour of `Test*Removal`
  pins (TypeError, signature, positional-slot binding); the committed oracle fixture is the
  numerical reference. `tests/_capture_v4_aggregate_oracles.py` is marked historical (it
  drives the removed route; runnable only at the recorded `captured_at_head`). Naming-guard
  allowlist entries for the `llms.txt` / `llms-full.txt` canonical post-fit `aggregate`
  vocabulary. Ledger row `M-093` records that CS/EfficientDiD populate the legacy fields on
  no fit. `warn_deprecated_kwarg` loses its last library caller and is retained (spec section
  8 rule 11) with a direct unit pin.
