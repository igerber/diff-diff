### Fixed
- **Exact fractional confidence-level labels, family-wide**: every text surface
  that names a confidence level now prints the exact coverage (`97.5%` for
  `alpha=0.025`; previously truncated to `97%` by `int((1 - alpha) * 100)` or
  rounded to `98%` by `int(round(...))` / `:.0f`) via one shared
  `results_base._coverage_pct` formatter: the 14 `summary()` headers
  (DiD/TWFE/MultiPeriod/SyntheticDiD, CallawaySantAnna, staggered and 2x2x2
  TripleDifference, StackedDiD, ImputationDiD, TwoStageDiD, EfficientDiD,
  ContinuousDiD, dCDH, SunAbraham, TROP), the `EventStudyResults` / HAD / RDD /
  ETWFE / LWDiD / LPDiD / ChangesInChanges table headers, the CS and dCDH sup-t
  band labels, the dCDH HonestDiD block (whose "Significant at" line printed
  `2%` for `alpha=0.025`; now `2.5%`), `WildBootstrapResults` and the LWDiD
  wild-cluster-bootstrap summaries, `HonestDiDResults` / `PlaceboTestResults`
  summaries, and BusinessReport / DiagnosticReport prose. The BusinessReport
  headline `ci_level` field carries the exact level as an `int` when integral
  (`95` is byte-unchanged) and a `float` otherwise (`97.5`); no schema-version
  bump (REPORTING.md Note). Default-alpha output is byte-identical. A source
  guard (`tests/test_coverage_label.py`) rejects any reintroduced inline
  percent computation.
