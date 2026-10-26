### Internal
- PR 4c-ii-b (docs/benchmarks half of the MultiPeriodDiD removal): REGISTRY / REPORTING /
  survey-theory / METHODOLOGY_REVIEW and the remaining user-doc prose stop describing the
  removed class as live; TODO rows 33/35/38/95/99 and three DEFERRED rows that governed the
  class are reworded or retired. The oracle fixture `tests/fixtures/v4_twfe_event_study_oracles.
  json` drops its `mpd_twin` records and the lane metadata loses its MPD twin fields (the TWFE
  records are byte-identical); the capture script documents the per-lane capture SHAs.
- Benchmarks: the live event-study arm is the new `benchmarks/python/benchmark_twfe_event_study.py`
  (`spec="within"`, `cluster="unit"`; parity against R `fixest::feols(... | unit)` verified
  locally - ATT to 2e-12, SE to 0.0%, per-period effects correlation 1.000);
  `benchmark_multiperiod.py` and `benchmarks/refresh_2026_07/` stay frozen because that harness
  runs inside pinned 3.5.3/3.7.0 venvs that predate `event_study=True`.
