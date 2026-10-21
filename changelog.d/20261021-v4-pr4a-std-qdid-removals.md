### Removed
- **The `StaggeredTripleDifference` class and its `SDDD` alias are gone** (`M-013`, `M-064`; 4.0
  cut, PR 4a). `from diff_diff import StaggeredTripleDifference` and `SDDD` raise `ImportError`,
  and the `diff_diff.staggered_triple_diff` module no longer exists. The staggered DDD estimator
  is `TripleDifference().fit(..., unit=, time=, first_treat=, partition=)` - the same engine
  (`diff_diff/_staggered_triple_diff_engine.py`), pinned against the committed oracle in
  `tests/test_v4_merge_ddd.py`. R's compact `control_group` spellings
  (`"notyettreated"`/`"nevertreated"`) died with the class; the merged surface takes
  `"not_yet_treated"`/`"never_treated"`.
- **The `QDiD` class and the `QDiDResults` alias are gone** (`M-015`, `M-061`). The quantile-DiD
  comparison estimator is `ChangesInChanges(method="qdid")` (same engine, identical numbers);
  the results container is `ChangesInChangesResults` for both methods.
- The two `pyproject.toml` pytest `filterwarnings` entries that silenced the removed classes'
  construction `FutureWarning`s.

### Breaking Changes
- **`TripleDifference.fit(time=)` in the 2x2x2 design now raises `ValueError`** naming `post=`
  (`M-085`). Through 3.x `time=` was the deprecated alias for the 0/1 post indicator and emitted
  a `FutureWarning` (`M-031`); since 4.0 `time` means the staggered calendar column only.
  Supplying both `post=` and `time=` keeps raising the "pass only post=" error.
- Keyword mapping for callers of the removed class: `eligibility=` is `partition=` on
  `TripleDifference`; the staggered fit parameters are keyword-only; `cluster=` raises in
  staggered mode (use `n_bootstrap > 0` for unit-level clustered inference). Users of
  `balance[did]` stay on 3.x through that extra's `diff-diff<4` pin, where the old name still
  resolves.

### Internal
- The three StaggeredTripleDifference suites (`tests/test_staggered_triple_diff.py`,
  `tests/test_methodology_staggered_triple_diff.py`, `tests/test_survey_staggered_ddd.py`) are
  ported to `TripleDifference`'s staggered mode (keyword fits; the R golden's compact
  `control_group` values are mapped), so the engine keeps its behavior, methodology and survey
  coverage. `tests/test_v4_merge_ddd.py` / `tests/test_v4_merge_cic.py` drop their old-vs-new
  parity lanes for `Test*Removal` pins; the two capture scripts are marked historical and
  `make_2x2` lives in the CiC merge test.
- Ledger: `M-013`/`M-015` -> `removed`, `M-061`/`M-064`/`M-085` -> `done`; `M-031` keeps
  `shimmed` with `warning: null`; `M-080` drops the deleted module from its `code_refs`.
  Spec: cut-sequence item 4 split into 4a/4b/4c, the cross-row rule gains its alias clause.
  `docs/doc-deps.yaml` re-primes the `staggered_triple_diff` group on the results module.
- `tests/test_v4_matrix.py` no longer requires `StaggeredTripleDifference`/`QDiD` worked-example
  owners in the migration guide; the two autosummary stubs are deleted; REGISTRY keeps the
  `## StaggeredTripleDifference` and `### QDiD` sections as the engine's methodology record
  (archive framing lands with `M-090`).
