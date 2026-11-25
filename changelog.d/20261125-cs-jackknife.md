### Added
- **Cluster jackknife inference for `CallawaySantAnna`** (`vcov_type="jackknife"`):
  the delete-one-cluster jackknife (CV3) of Karim, Nielsen, MacKinnon & Webb
  (2026, arXiv:2602.12043), for designs with few clusters, where the default
  influence-function standard errors and the multiplier bootstrap reject a true
  null too often. Each cluster (the `cluster=` column, or the unit when none is
  given) is deleted in turn, the estimator is re-run, and the variance is
  `((H - 1) / H) * sum_h (theta_h - theta)^2` around the full-sample estimate,
  with a `t(H - 1)` reference. It applies to the overall ATT, every ATT(g,t) and
  the `event_study` / `group` aggregations (the event-study container carries
  the jackknife covariance). `results.get_loo_effects_df()` lists the
  delete-one overall ATTs. Overall, event-study and group standard errors match
  R `did` on delete-one samples to 1e-8.
  - In the library's size simulation (no effect, nominal 5%, 8 to 32 clusters)
    the jackknife rejected 3-10% for the overall ATT with two or more treated
    clusters per cohort, against 4-15% for the clustered default and 7-21% for
    the multiplier bootstrap; with one treated cluster per cohort it rejected
    8-18% against 15-35%. A fit warns when a reported quantity rests on fewer
    than three treated clusters.
  - A deletion that leaves no estimate makes every jackknife standard error
    NaN; a cell, horizon or cohort that is not estimable in every delete-one
    sample gets a NaN standard error.
  - Not combinable with `n_bootstrap > 0`, `survey_design=` or
    `allow_unbalanced_panel=True`; `panel=False` needs `cluster=`.
  - `DiagnosticReport` gives a jackknife fit a Bonferroni pre-trend check (no
    covariance-based joint Wald) and skips its pre-trends power section.
  - `CallawaySantAnna(vcov_type="jackknife")` can be passed to
    `diagnose_design` as a candidate.

### Changed
- **Tutorial 2 (staggered DiD)** no longer says the multiplier bootstrap gives
  valid inference with few clusters; it points to the cluster jackknife.
