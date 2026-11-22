### Fixed
- **Replicate-weight inference df: a dropped replicate now caps the design df
  instead of replacing it** (every estimator with replicate-weight support:
  `CallawaySantAnna`, `TripleDifference` (2x2x2 and staggered), `EfficientDiD`,
  `ContinuousDiD`, `DifferenceInDifferences`, `TwoWayFixedEffects` including
  event-study mode, `SunAbraham`, `StackedDiD`, `ImputationDiD`, `TwoStageDiD`,
  `DMLDiD` aggregates, and `LinearRegression`). When a replicate was dropped
  (an all-zero replicate column, or a failed replicate refit / solve) the
  inference df became `n_valid - 1`, ignoring the design df (QR rank of the
  replicate matrix minus 1). It is now `min(df_survey, n_valid - 1)`. Two
  outcomes were wrong and are fixed; SEs and point estimates are unchanged:
  - **Rank-deficient replicate matrix: df was raised above the design df.** With
    15 usable columns of rank 5 plus one all-zero column, t-stats were referred
    to t(14) instead of t(4): on a small staggered panel the `CallawaySantAnna`
    overall p-value was 9.1e-09 and is now 2.7e-04, the `TwoWayFixedEffects`
    p-value was 7.4e-11 and is now 6.5e-05, and `survey_metadata.df_survey`
    read 14 and now reads 4. p-values and confidence intervals on such designs
    get wider.
  - **Undefined design df (rank <= 1) plus a dropped column gave t-inference.**
    Statistics were referred to t(`n_valid - 1`) (e.g. `CallawaySantAnna`
    p = 0.013 at t(2)); they now report NaN t-stat, p-value and CI with the SE
    still reported, the same as an undefined design with no dropped column.
    `survey_metadata.df_survey` stays `None` (it previously read
    `n_valid - 1`).
  - `LinearRegression.survey_df_` follows the same rule: the capped df, or
    `None` on an undefined design df, in which case `get_inference()` emits its
    existing "undefined survey d.f." warning and returns NaN inference whether
    or not a replicate dropped.
  - Plain (no-absorb) `DifferenceInDifferences`: when failed replicate solves
    lowered the df, inference used the lower df but `survey_metadata.df_survey`
    and `summary()` kept showing the design df. They now report the df used.
  - Designs with no dropped replicate, and designs whose surviving replicate
    columns are full rank, are unaffected (the two rules coincide).
  - See the REGISTRY Survey Data Support replicate-df Note. New helper
    `diff_diff.survey.reduced_replicate_df`.

### Behavioral Changes
- **One statistic with fewer than two valid replicates no longer changes the
  inference of the others** (`CallawaySantAnna`, staggered `TripleDifference`
  and `DMLDiD` event-study / group aggregates, `EfficientDiD`,
  `ContinuousDiD`). Such a statistic has a NaN SE and NaN inference; it used to
  also alter the df shared with the rest of the fit:
  - `CallawaySantAnna` family aggregates: every event-study horizon or group
    row went NaN, and the overall relay could write `0` into
    `survey_metadata.df_survey`. Now only the affected row is NaN.
  - `EfficientDiD`: every statistic computed afterwards silently switched from
    the t reference to the normal reference. Now they keep the design df.
  - `ContinuousDiD`: every statistic's inference went NaN. Now only the
    affected statistic is NaN.
  - This needs a non-finite influence function on a single statistic, so it is
    not expected on well-formed fits. `ChaisemartinDHaultfoeuille` is unchanged:
    it deliberately takes one global minimum over all its replicate-variance
    sites (fail-closed).
