### Fixed
- **Per-cell ATT(g,t) standard errors now honor replicate-weight and PSU survey
  designs** (`CallawaySantAnna`, staggered `TripleDifference`). The aggregate SEs
  (overall, event study, group) of these fits were already design-based, but the
  per-cell `group_time_effects` SE, t-statistic, p-value and confidence interval
  were silently those of a weights-only fit in three lanes:
  - `CallawaySantAnna` with `SurveyDesign(replicate_weights=...)`;
  - staggered `TripleDifference` with `SurveyDesign(replicate_weights=...)`;
  - staggered `TripleDifference` with `SurveyDesign(psu=...)`.

  On a one-cell panel, where the cell is the overall ATT, the cell SE now equals
  the overall SE: 0.3428 instead of 0.1568 for `CallawaySantAnna` under JK1
  replicates (cell p-value 0.016 instead of 4.9e-06), and 0.3253 instead of 0.2173
  for staggered `TripleDifference` under both JK1 replicates and a PSU design.
  Per-cell SEs under those designs will generally be larger than before; point
  estimates and all aggregate results are unchanged, as are fits with no survey
  design or a weights-only, strata-only or FPC-only design.
  - Replicate designs use IF reweighting on the cell's influence function with a
    cell-local df `min(df_survey, n_valid - 1)` (new helper
    `diff_diff.survey.cap_replicate_df`); PSU designs use CR1 on the PSU totals of
    the cell's influence function. For staggered `TripleDifference` both are
    applied to the GMM-combined influence function the aggregates consume.
  - Fail-closed: a cell whose replicate variance is zero, negative or non-finite,
    or whose influence function carries a non-finite value, reports a NaN SE with
    NaN inference. The non-finite guard also applies to the existing
    `CallawaySantAnna` and `DMLDiD` PSU per-cell lane, which could otherwise report
    a finite clustered SE for a partly-NaN influence function.
  - See the REGISTRY CallawaySantAnna and StaggeredTripleDifference per-cell
    survey Notes.
