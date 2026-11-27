### Fixed
- **PreTrendsPower NIS determinism and `power_adequate` tolerance**: seeded
  `scipy.stats.multivariate_normal.cdf` in `_compute_nis_acceptance_prob` (when
  supported by SciPy) so repeated NIS power and MDV calls return identical
  values, and applied the integration error tolerance (`1e-5` for NIS, `1e-12`
  for Wald) in `PreTrendsPowerResults.power_adequate` so evaluating at `M=None`
  (the MDV) consistently reports adequate power instead of flipping randomly.
