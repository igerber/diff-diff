### Internal
- **4.0 cut, PR 3a - tests-first migration to the post-fit aggregation containers.** The
  test and benchmark suites now read event-study / group surfaces from
  `results.aggregate("event_study")` / `results.aggregate("group")` (the container API that
  has existed since 3.9) instead of the deprecated fit-time `fit(aggregate=...)` keyword and
  the legacy `event_study_effects` / `group_effects` fields, so the keyword removals in PRs 3b
  and 3c land against already-migrated suites. No library source changes and no ledger row
  flips in this PR; the fit-time keywords still work and still warn. Bootstrapped
  ImputationDiD / TwoStageDiD / ContinuousDiD sites stay on the fit-time route until PR 3c
  adds the post-fit relay for them.
- **Committed per-backend aggregation oracles.** `tests/test_aggregate_contract.py` no longer
  uses the fit-time route as its numerical reference: every analytic surface is pinned at
  `rtol=1e-12` (confidence-interval bounds at `1e-9`: scipy's `t.ppf` moved by up to 3.4e-11
  relative between 1.13 and 1.14+) and every bootstrapped CS / EfficientDiD replay at the
  replay-parity tolerances against `tests/fixtures/v4_aggregate_oracles.json`, captured on the unmodified
  tree at `8724fecd` by `tests/_capture_v4_aggregate_oracles.py` under both weight backends
  (see `tests/fixtures/README.md`). New `tests/helpers/legacy_surface.py` (`es_dict`,
  `group_dict`, `legacy_standin`) gives consumer tests a faithful stand-in for the 3.x
  fit-time surface where their subject is the native (legacy-pickle) route.
- **Benchmarks** (`benchmarks/python/benchmark_callaway.py`, `benchmark_imputation.py`, the
  `speed_review` scripts) call `aggregate()` post-fit inside the timed region and emit the
  same JSON row shape the R refresh joins on.
- `docs/v4-design.md` section 9 records the three-way split of cut-sequence item 3 (3a
  tests-first, 3b mechanical removals, 3c the Imputation/TwoStage/ContinuousDiD removals +
  relay).
