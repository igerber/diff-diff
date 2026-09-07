### Added
- **TWFE weight diagnostics** (port of Brantly Callaway's `twfeweights` R
  package, MIT): what a two-way fixed effects regression *implicitly* weights
  on staggered-adoption data.
  - `attgt_weights(results, aggregation="twfe"|"overall"|"simple")` reports the
    weight a TWFE regression, ATT^O, or ATT^simple places on each ATT(g,t),
    plus post-period negative-weight counts. Returns `ATTGTWeightsResult`.
  - `decompose_twfe_weights(data, ..., method="fwl")` re-derives the estimate
    from its ATT(g,t) building blocks and returns `TWFEDecompositionResult`
    with `pretrend_bias` - the contribution of pre-treatment cells, i.e. of
    parallel-trends violations rather than of treatment - and, with
    `balance_covariates=`, implicit-weight covariate balance.
    `plot_twfe_weights()` renders either view (matplotlib or plotly).
  - Validation: rejects NaN / `-inf` cohort labels, covariate-adjusted fits
    under `aggregation="twfe"`, duplicated or non-finite ATT(g,t) cells, an
    incomplete group-time grid, and invalid sampling weights. Two structural
    gaps are handled as R does instead of raising: a cohort with no estimable
    post cell is dropped (`did`'s first-period drop), and under
    `control_group="not_yet_treated"` the CS estimands average over each
    cohort's available post periods (`aggte`).
