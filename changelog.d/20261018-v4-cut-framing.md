### Documentation
- **4.0-cut framing** (the first PR of the 4.0 cut, at v3.13.0): `docs/v4-design.md`
  re-synced with the library as shipped since 3.9 - the estimator roster now counts
  LWDiD/DMLDiD/DurationDiD (27 -> 24 at 4.0, 23 estimators + Bacon in the diagnostic
  family), the diagnostic family names the TWFE weights containers and
  `DurationDiDPretestResults`, and section 9 carries the ten-PR cut sequence. `M-080`
  (auto-cluster policy) re-scoped to the measured roster: static `TwoWayFixedEffects`
  already auto-clusters at unit, `SpilloverDiD` is the one estimator whose default SE
  moves at 4.0, `cluster=False` becomes the uniform explicit off-switch, and the
  by-construction estimators keep no auto-cluster metadata by decision (new "Default
  clustering roster" section in `docs/methodology/variance-conventions.md`; REGISTRY
  Notes on SpilloverDiD, StackedDiD, ImputationDiD, TwoStageDiD, DMLDiD, DurationDiD).
  `M-008` (`DIFF_DIFF_SOLVE_OLS_FASTPATH` default) decided: evaluated, kept off. New
  ledger row `M-150` gives `WooldridgeDiDResults.to_dataframe(level="event")` its own
  locator (twin of `M-136`). The migration guide's auto-cluster section, appendix and
  orientation table follow the ledger (111 appendix rows).

### Internal
- **Ledger cut window** (`docs/v4-deprecations.yaml` header `cut_window: "4.0"`,
  `cut_base: "3.13.0"`; spec section 11): rows scheduled at 4.0 may flip terminal
  ahead of the version bump so the removals land over many PRs while `__version__`
  stays 3.13.0; `tests/test_v4_matrix.py` relaxes only the early-removal guard for the
  window version, refuses any other version bump while early-terminal rows exist, and
  keeps due enforcement unchanged. The id snapshot and `ROW_COUNT_FLOOR` now include
  `M-148`/`M-149` (shipped by #838 without extending them) and `M-150` (132 rows).
  `M-091`'s roster records the four later-born `Diagnostic` containers; `M-050`'s
  `code_refs` record that `diff_diff/_deprecation.py` survives 4.0 as 5.0-window
  infrastructure; `M-045`/`M-115` name the legacy-CR1 remap branches their removal
  deletes. A `maint/3.13` branch is cut at `v3.13.0` for 3.13.x patches.
