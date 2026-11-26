### Added

- Add experimental native Python `ChainedDiD` for overlapping incomplete
  panels: never-treated controls, PT-post, normalised IPW and fixed
  identity-weighted GMM. Results include explicit support diagnostics,
  original-unit population accounting, full joint covariance, analytical
  unit/cluster inference and canonical post-fit aggregation. The pinned R
  comparison agrees on eight fixtures, with a documented propensity-fitting
  difference for incomplete continuous covariates. Fresh simulation coverage
  qualification remains incomplete. This does not implement conditional
  attrition correction or the complete `cdid` feature set.

- Require two observed units per arm for chained comparisons, record excluded
  candidates and warn on singleton exclusions or low retained support. Identity
  weighting still ignores pair precision; these safeguards do not establish
  calibrated intervals. Exclude reference identities from mixed event-study
  diagnostic rows, expose constituent weights with `aggregation_weights(level)`,
  and check frozen public R outputs offline without changing tolerances.

- Add a reproducible same-panel Gaussian diagnostic comparing native HC0 with
  known covariance and benchmark-only design precision weighting. Preserve the
  production metric and failed qualification gates; the prototype's bounded
  simulation agreement is not general inference qualification.
