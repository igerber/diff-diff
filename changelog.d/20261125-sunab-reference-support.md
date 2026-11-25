### Fixed
- **`SunAbraham` no longer reports wrong finite numbers when a
  cohort-by-period coefficient is not identified.** The saturated regression
  omitted one relative period, `e = -1 - anticipation`, for every cohort.
  Whenever some coefficient was not identified in that design the QR step
  dropped one unnamed column (a data-dependent choice, often a
  post-treatment one), every coefficient sharing the dependency was then
  measured against the dropped one, and those numbers were pooled into the
  event-study rows and the overall ATT. On a noiseless panel with true
  effect `2 + 0.5 e` (90 units, 8 periods, cohorts 3 and 5):
  - a cohort treated at the first panel period: overall ATT NaN, seven wrong
    cohort coefficients, an `e = 0` row of 0.0 for a true 2.0;
  - a cohort first observed at `e = 0`: ATT NaN, five wrong coefficients, an
    `e = 0` row of -0.25;
  - a cohort whose units are observed at disjoint sets of periods: ATT NaN,
    four wrong coefficients, an `e = 0` row of 1.0;
  - never-treated units absent at one post-treatment calendar period: ATT
    NaN and an `e = 3` row of 2.25 for a true 3.5;
  - a cohort with a gap at `e = -1`: the baseline for that cohort was
    whichever of its columns QR dropped (a pre-period one on this panel,
    which left the estimates right and the `e = -2` row NaN).
  On 50 random-missingness panels, 14 had a wrong finite cohort coefficient,
  5 of them with no rank-deficiency warning at all (the demeaned design was
  not detected as deficient). Each case is now handled (see Behavioral
  Changes). `fixest::sunab` has the same defect on
  these panels, so the corrected behavior is a documented deviation from R.
- **`SunAbraham` bootstrap SEs were wrong, or the fit raised `IndexError`,
  on a frame whose index is not `0..n-1`.** Both the pairs bootstrap and the
  Rao-Wu bootstrap used index labels as row positions. A filtered frame
  (reversed labels on the regression-test panel: SE 0.1085 instead of
  0.0958), a frame with shifted labels, and a default-index frame holding a
  unit with a negative or missing `first_treat` were all affected.
- **`SunAbraham` under survey weights: a cohort-by-period cell with zero
  total weight no longer turns identified results into NaN.** Such a cell
  got an all-zero weighted design column, and its NaN coefficient was then
  pooled at weight zero (`0 * NaN`). With one cohort entirely at weight 0
  the overall ATT and five event-study rows were NaN although the other
  cohort identified them.

### Behavioral Changes
- **`SunAbraham`: per-cohort reference period.** A cohort that cannot be
  normalized at `e = -1 - anticipation` (it is not observed there, or no
  never-treated unit is observed at that calendar period) is normalized at
  its latest usable pre-treatment period instead, and `fit()` warns naming
  the cohort and its reference. `SunAbrahamResults.reference_event_times` (new field) lists the
  distinct bases, and `reference_period` reports the shared fallback when no
  cohort is observed at `e = -1 - anticipation` and all share one (for
  example `-2` on a biennial grid), so the event-study anchor row sits at the
  true base. Panels in which every cohort is observed at the reference
  period, with never-treated units observed at those calendar periods, are
  unchanged.
- **`SunAbraham`: cohorts with no pre-treatment observation are excluded.**
  `fit()` warns naming the cohort and estimates on the data without it; with
  no treated cohort left it raises `ValueError`. Under `survey_design=` it
  raises `NotImplementedError` instead, since dropping rows there would not
  be domain estimation.
- **`SunAbraham`: unidentified coefficients are NaN.** A cohort-by-period
  coefficient that the observed rows do not identify is reported as NaN with
  one warning naming the cells, as is every pooled event-study row that
  contains one (and the overall ATT when a post-treatment row does).
  `event_study_vcov` is NaN on those rows and columns only. Bootstrap draws
  that cannot identify a row are dropped for that row, and replicate-weight
  variances are computed over the identified entries.
  `rank_deficient_action="error"` raises on such a panel and `"silent"`
  suppresses the warning.
- **`compute_pretrends_power` refuses a `SunAbraham` fit whose cohorts are
  normalized against different reference periods** (`reference_event_times`
  with more than one entry): the hypothesized violation is defined relative
  to one reference. `DiagnosticReport` reports that as an error status on its
  `pretrends_power` section.
