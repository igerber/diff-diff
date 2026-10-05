### Fixed
- **Bootstrapped CallawaySantAnna fits now carry an event-study covariance**
  (#836): with `n_bootstrap > 0`, `event_study_vcov` (fit time) and the `vcov`
  of `aggregate("event_study")` (post-fit replay, CallawaySantAnna and DMLDiD)
  are the covariance (`ddof=1`) of the event-study bootstrap draws, indexed by
  the horizons with a finite bootstrap SE. Its diagonal is the bootstrap SEs
  squared, because both come from the same draws. Previously the bootstrap
  path cleared the covariance and HonestDiD fell back to `diag(se**2)` with a
  warning, which made its Delta^SD intervals under-cover in staggered designs
  with serially correlated errors. HonestDiD bounds on bootstrapped CS fits
  therefore change, and the diagonal-fallback warnings no longer fire. The
  matrix is still not stored, and the diagonal fallback still applies, when
  `n_bootstrap` does not exceed the number of event-study horizons (the
  covariance would be singular), when a draw is non-finite, or with fewer
  than two PSUs.

### Behavioral Changes
- **Pre-trend diagnostics on bootstrapped CallawaySantAnna fits read the
  bootstrap covariance** (#836): `compute_pretrends_power` now reports
  `covariance_source="full_pre_period_vcov"` instead of `"diag_fallback"`,
  and the `DiagnosticReport` parallel-trends check runs the joint Wald test
  on the event-study covariance (`method="joint_wald_event_study"`, or its
  `_survey` variant) instead of the Bonferroni fallback.
