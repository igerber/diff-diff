### Removed
- **The `MultiPeriodDiD` class, its `MultiPeriodDiDResults` container, the `PeriodEffect`
  record and the `EventStudy` alias are gone** (`M-010`, `M-011`, `M-012`, `M-060`; 4.0 cut,
  PR 4c-ii). `from diff_diff import MultiPeriodDiD` (and the other three names) raises
  `ImportError`. The estimator is `TwoWayFixedEffects().fit(..., event_study=True, spec=,
  post_periods=, reference_period=)`, which returns the unified `EventStudyResults` surface
  natively: `spec="pooled"` is the removed class's design (bit-exact without `unit=`; with
  `unit=` the pooled spec auto-clusters at unit), `spec="within"` is its `absorb=["unit"]`
  design; `post_periods=` is required (the midpoint default died with the class) and the
  post-period average `avg_att` has no successor field. Pinned field for field against the
  pre-refactor capture `tests/fixtures/v4_twfe_event_study_oracles.json`.
- The `pyproject.toml` pytest `filterwarnings` entry that silenced the removed class's
  construction `FutureWarning`, the `MultiPeriodDiD` power-simulation profile
  (`simulate_power` has no event-study profile yet; TODO row), and the `MultiPeriodDiDResults`
  branches of DiagnosticReport / BusinessReport / `practitioner_next_steps` (both reports still
  reject `EventStudyResults` by design; the admission TODO row records where the logic lived).

### Breaking Changes
- **`TwoWayFixedEffects.fit(time=)` in static mode now raises `ValueError`** naming `post=`
  (`M-083`). Through 3.x `time=` was the deprecated alias for the 0/1 post indicator and
  emitted a `FutureWarning` (`M-082`); since 4.0 `time` means the event-study calendar column
  only. Supplying both `post=` and `time=` keeps raising the "pass only post=" error.
- **`EventStudyResults.period_effects` is a `FutureWarning` compatibility view** (`M-016`;
  removed in 5.0): on a calendar surface it returns the former calendar-keyed dict of
  per-period records (`period, effect, se, t_stat, p_value, conf_int, is_significant,
  significance_stars`, reference period omitted; `is_significant` is evaluated at the fit's
  `alpha`, where the removed record hard-coded 0.05; duplicate labels raise). Relative-time
  `aggregate("event_study")` surfaces do not carry it (`AttributeError`, no warning). The
  successor is `to_dataframe()` and the columnar arrays.
- The 3.x "default reference_period has changed" `FutureWarning` is no longer emitted
  (`M-007`): the event-study mode defaults to the last pre-period silently.
- `simulate_power` and friends read the flat `att`/`se`/`p_value`/`conf_int` names from
  unregistered custom estimators' results and raise `TypeError` naming `_EstimatorProfile`
  otherwise (the `avg_*` fallback left with the removed container; pass `result_extractor=`
  for other shapes).

### Internal
- `DifferenceInDifferences._fit_event_study_core` serves only the TWFE event-study mode
  (`post_periods` required by the core; `include_treatment_main` / `cluster_override`
  keyword-required); the private payload is pruned to the fields the surface builder reads.
  The clustered `hc2_bm` contrast batch keeps the former post-period-average column as the
  batch-relative Bell-McCaffrey DOF guard's anchor so per-period NaN masks stay bit-identical
  (REGISTRY Note; TODO row for the guard redesign). `build_event_study_surface` opens with an
  `isinstance(EventStudyResults)` passthrough.
- Ledger: `M-010`/`M-011`/`M-012` -> `removed`, `M-060`/`M-007`/`M-083` -> `done`, `M-016`
  -> `shimmed` on `diff_diff:EventStudyResults.period_effects`, `M-082` keeps `shimmed` with
  `warning: null`. Spec item 4c-ii carries the SHIPPED note incl. the three recorded
  deviations (anchor column, new live benchmark script, DR/BR branch deletion).
- Tests: the MPD-only tests are deleted or re-fixtured onto `cs_fit` / the TWFE surface
  (`tests/test_v4_merge_mpd.py` carries the removal, `M-016` and `M-007` pins and the anchor
  pin); the oracle fixture gains two clustered `hc2_bm` lanes captured at `d45d5316` before
  the core pruning (its `mpd_twin` records are pruned in PR 4c-ii-b); the clustered
  per-period CR2 DOF R/clubSandwich pin and the event-study covariate-collision guard are
  migrated to TWFE; variance-conventions `twfe_es_*` rows re-pinned (`tail_df` 5) and the table
  regenerated. The methodology/benchmark prose half and the oracle fixture's
  `mpd_twin` prune follow in PR 4c-ii-b.
