### Added
- **TWFE weight diagnostics** (port of Brantly Callaway's `twfeweights` R
  package, MIT): what a two-way fixed effects regression *implicitly* weights
  on staggered-adoption data.
  - `attgt_weights(results, type="twfe"|"overall"|"simple")` reports the
    weight a TWFE regression, ATT^O, or ATT^simple places on each ATT(g,t),
    plus post-period negative-weight counts. Returns `ATTGTWeightsResult`
    (`.level` records the type).
    The CS estimands honour the fit's `anticipation` window in raw time units
    (an explicit `anticipation=` on the frame path); `"twfe"` keeps `t >= g`,
    but a cohort with no estimable post cell under the window is dropped for
    every type.
  - `decompose_twfe_weights(data, ..., method="fwl")` re-derives the estimate
    from its ATT(g,t) building blocks and returns `TWFEDecompositionResult`
    with `pre_period_contribution` - the sample contribution of the
    pre-treatment cells, which can reflect parallel-trends violations or
    sampling variation - and, with `balance_covariates=`, implicit-weight
    covariate balance. `plot_twfe_weights()` renders either view (matplotlib
    or plotly).
  - Validation: rejects NaN / `-inf` cohort labels, covariate-adjusted fits
    under `type="twfe"`, unbalanced panels, fitted results that dropped any
    unit by their per-cell complete-case rules or that carry no completeness
    record (fitted by <= 3.12.0), non-finite outcomes / covariates,
    duplicated or non-finite ATT(g,t) cells, an incomplete group-time grid,
    and invalid sampling weights. Gaps the estimator itself could not fill
    (`skip_reason` missing_period / zero_treated_control / zero_weight_mass)
    are handled as `aggregate()` does instead of raising: a cohort with no
    estimable post cell under the window is dropped, and the CS estimands
    average over each surviving cohort's available post cells (`aggte`).
