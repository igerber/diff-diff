### Fixed
- **`StackedDiD` no longer returns shifted estimates on panels with missing
  observations.** The stacked regression contrasts (treated x event-time) and
  (control x event-time) cell means, which are within-unit changes only when
  the same units sit in every cell of a sub-experiment. The paper's stack is
  built that way (Wing, Freedman & Hollingsworth 2024, Section 5.1), but
  `fit()` kept every row of an eligible unit, so on a ragged panel level
  differences between units leaked into the estimates, with no warning. On a
  noiseless panel with a true ATT of 2.5 (90 units, 10 periods, cohorts 4 and
  6, `kappa_pre = kappa_post = 2`):
  - one never-treated observation missing: 2.4978;
  - 5% of rows missing at random: 2.6020;
  - half of one cohort missing its `e = -1` row: 2.2965; the whole cohort
    missing it, or having no pre-treatment rows at all: 1.9500;
  - never-treated units missing two post-treatment periods: 2.7000;
  - a calendar period absent for every unit: 1.7667 or 2.6833, depending on
    the period;
  - no never-treated rows inside the event windows: NaN with a rank warning.
  Each case now returns 2.5 exactly or raises (see Behavioral Changes). The
  authors' reference R code returns the same shifted numbers on such panels
  (the library matched it to 12 digits), so the corrected behavior is a
  documented deviation from it; R `cbwsdid` already builds its stack from
  complete units, and `balance="entropy"` now matches it on ragged panels.
- **`StackedDiD` no longer mis-assigns rows it cannot place on the event-time
  grid.** Duplicate `(unit, time)` rows were double-counted (2.5411 on the
  panel above), rows at a fractional `time` were absorbed into the reference
  category (1.3833), rows with a missing `time` were dropped silently, and a
  fractional `first_treat` was truncated to the period before it. All four
  now raise `ValueError`.
- **`StackedDiD(weighting="population")` and `weighting="sample_share"` now
  apply the paper's treatment weight.** Table 1 of Wing, Freedman &
  Hollingsworth (2024) weights the treated rows as well as the controls; the
  library set the treated weight to 1 under every scheme, so under these two
  the treated arm was aggregated by unit shares and the control arm by
  population or sample shares, the untreated trends did not cancel, and the
  estimate was not the named aggregate. On balanced noiseless panels with
  cohort-specific effects and a quadratic time trend the largest error in an
  event-study row was 0.0023 (`population`) and 0.10 (`sample_share`); both
  are now exact. The default `weighting="aggregate"` was correct and is
  bit-identical.
- **`StackedDiD` + `survey_design=`: the corrective weights now use
  survey-weighted cohort shares.** They were formed from raw unit counts and
  then multiplied by the survey weight, which left the treated arm on
  survey-weighted shares and the control arm on a mix of both. With
  unit-constant weights of 1.5 / 0.5 / 1 across two cohorts and the controls,
  a noiseless panel with a true ATT of 2.50 returned 2.46; with half of a
  cohort at weight 0 it returned 2.4733. Both are now exact, replicate-weight
  refits recompute the corrective weights from each replicate's weights, and
  an independent R implementation (`survey::svyglm`) of the same formula
  matches to 1e-11.
- **`StackedDiD` + `survey_design=` no longer accepts survey columns that
  change within a unit.** A weight that varies over a unit's rows changes
  each event-time cell's weighted composition exactly as a missing row does
  (2.6174 for a true 2.5 with positive time-varying weights); it now raises
  `ValueError`, as it does for the other panel estimators.
- **`StackedDiDResults.groups` no longer lists an adoption event whose
  sub-experiment was not built.**

### Behavioral Changes
- **`StackedDiD`: complete event windows.** A unit enters a sub-experiment
  only if it is observed at every event time of the window
  `[-kappa_pre - anticipation, kappa_post]`. On a panel with missing
  observations the units that are not are excluded from that sub-experiment
  and `fit()` warns with the per-event "kept of eligible" counts; the estimand
  is the trimmed aggregate ATT over the units observed across their whole
  window, and `n_treated_units`, `n_control_units` and `stacked_data` reflect
  them. Balanced panels are unchanged.
- **`StackedDiD`: events without complete units are dropped.** An adoption
  event left without a treated unit or a clean control observed across its
  window, including one whose window contains a calendar period no unit
  observes, is dropped with a warning naming the reason and is reported in
  `trimmed_groups` instead of `groups`. With no event left `fit()` raises
  `ValueError`, naming the gaps in the time grid when there are any (a
  biennial or datetime-coded `time` column must be re-indexed to consecutive
  integers). Panels that previously raised "would be fabricated" for a
  missing reference period now raise this error instead.
- **`StackedDiD` + `survey_design=` refuses a panel that would be trimmed**
  (`NotImplementedError`), whether a unit or a whole adoption event would be
  dropped: excluding them would remove their PSUs and strata from the variance
  and its degrees of freedom.
- **`StackedDiD` estimates change under `weighting="population"`,
  `weighting="sample_share"` and `survey_design=`** (see Fixed): under the two
  alternative schemes wherever their target shares differ from the treated-unit
  shares, and under a survey design wherever the survey weights differ across
  cohorts. Replicate-weight standard errors change as well, since each refit
  recomputes the corrective weights. A sub-experiment with no treated or
  control unit of positive weight, or a zero treated population, now carries
  zero weight and is named in a warning; `weighting="population"` requires a
  finite, non-negative, unit-constant population for the treated units.
- **`StackedDiD(balance="entropy")` trims ragged windows** like the default
  path, where it previously raised `ValueError`.
