### Fixed
- **`plot_event_study` accepts `WooldridgeDiDResults`**: passing a Wooldridge
  ETWFE result (the object `results.aggregate("event_study")` returns) raised
  `KeyError: 'effect'`. The result is now converted to the unified event-study
  container at entry, so its stored intervals are drawn as-is, the
  stored-`alpha` mismatch warning applies, and a `weights="cohort_share"`
  aggregation (NaN intervals by design) draws no error bars. A fit on which
  `aggregate("event_study")` has not been called raises a `ValueError` naming
  that call. On every container input, a row whose stored interval is NaN now
  also stays without an interval under an explicit `reference_period=`
  (previously the normalization path recomputed one from the SE, which would
  have redrawn inference that a producer had disabled). A contract test now pins the one event-study idiom the tutorials
  teach (`es = results.aggregate("event_study")` where the class has
  `aggregate`, else the result; `plot_event_study(es)`) on all thirteen
  estimators that produce an event study.
