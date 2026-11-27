### Fixed
- **`StackedDiD` with `anticipation > 0` no longer uses a later adopter as a
  control while that unit is already responding.** The clean-control cutoff
  ignored `anticipation`, so under `control_group="not_yet_treated"` (the
  default) a cohort adopting at `a + kappa_post + 1` up to
  `a + kappa_post + anticipation` stayed in event `a`'s control pool through
  periods in which it had started to respond, and the estimate was biased
  with no warning. On a noiseless balanced panel (periods 0-8, equal-sized
  cohorts adopting at 3 and 6 plus never-treated units, an effect of 2
  starting one period before adoption, `kappa_pre = kappa_post = 2`,
  `anticipation = 1`) the fit returned 1.5 at event time 2 and an overall ATT
  of 1.875; both are now 2 exactly. The cutoff is now
  `A_s > a + kappa_post + anticipation`, and
  `A_s > a + kappa_post + kappa_pre + anticipation` under
  `control_group="strict"`. A strict fit used a responding control only when
  `anticipation > kappa_pre`, but its control set changes more widely: the
  new cutoff also drops later cohorts adopting up to `anticipation` periods
  past the old one, so that their own window, which now starts
  `anticipation` periods earlier, stays disjoint from this one. The inclusion check uses the
  same rule, so an adoption event whose only later cohort falls inside the
  cutoff is now trimmed. Estimates change only for fits with
  `anticipation > 0` that had a later cohort between the old and the new
  cutoff; `anticipation=0` and `control_group="never_treated"` are unchanged.
  Under `survey_design=`, a fit in which the `anticipation` term leaves a
  later cohort out of every sub-experiment now raises `NotImplementedError`:
  the design is re-resolved on the stacked rows, so that cohort's PSUs or
  strata can leave the variance and its degrees of freedom (on a stratified
  census the fit would have reported SE 0.136 with 9 df where the full design
  gives SE 0 with 15 df).
  Each fit records the rule on the new
  `StackedDiDResults.control_cutoff_includes_anticipation` (`True`; `None` on
  a result saved by an earlier version). For a `"not_yet_treated"` or
  `"strict"` result with `anticipation > 0`, `BusinessReport` and
  `DiagnosticReport` state the adjusted cutoff when the result records it; when
  it does not, they show the earlier cutoff, label it unverified and say to
  rerun the fit. Results with `anticipation=0` or never-treated controls get
  no such caveat. Separately, the `BusinessReport` assumption text now gives
  the event window with its `anticipation` term whenever `anticipation > 0`.

- **`StackedDiD` + `survey_design=` without a `psu` column no longer uses
  `(unit, sub-experiment)` pairs as survey PSUs under
  `cluster="unit_subexp"` (Taylor-linearized variance; replicate-weight
  designs are unaffected).** Each original unit is the PSU, as it already was
  under the default `cluster="unit"`; a `UserWarning` says that
  `"unit_subexp"` does not apply to the survey variance. The pairs duplicated
  sampled units and their count moved with the stack: on a stratified panel
  with a finite-population correction the fit reported SE 0 with 21 survey
  degrees of freedom where the sampled units give SE 0.07906 with 15.
  Standard errors change for survey fits that combined `cluster="unit_subexp"`
  with a design that names no `psu`; fits with an explicit `psu` or the
  default cluster are unchanged.
- **`StackedDiD` + `survey_design=` with zero survey degrees of freedom now
  returns NaN inference.** When `n_PSU - n_strata = 0` under Taylor
  linearization (for example one PSU per stratum with `lonely_psu="adjust"`)
  the df was clamped to 1 and the fit reported t(1) p-values and confidence
  intervals. Estimates and standard errors are still reported; t-statistics,
  p-values and intervals are NaN, overall and per event time, with a
  `UserWarning`.
- **`StackedDiD` + `survey_design=` without a `psu` column validates its
  implicit unit PSUs like explicit ones.** A stratum with a single unit now
  warns that it is excluded from variance estimation under
  `lonely_psu="remove"`, and a finite-population size below a stratum's unit
  count raises `ValueError`; both checks were skipped before.
