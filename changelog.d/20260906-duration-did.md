### Added
- **DurationDiD estimator** (Deaner & Ku 2026, *Causal Duration Analysis with
  Diff-in-Diff*, arXiv:2405.05220v2): two-group, common-timing difference-in-
  differences for a binary absorbing outcome. `method="cd"` (constant additive
  gap between the groups' untreated hazards) or `method="ph"` (constant hazard
  ratio, mean-of-ratios estimator) is fitted on the pre-treatment cumulative
  hazards — by default with equal weights over every eligible pre-treatment
  date, or on a user window via `fit(pre_periods=..., pre_period_weights=...)`
  — with exact numeric date selection and finite real fitting weights.
  Integer dates preserve their identity across supported signed/unsigned
  dtype ranges: spacings and offsets are subtracted before float64 elapsed
  arithmetic, which must remain finite and strictly increasing. Floating
  dates must be losslessly representable as float64; complex inputs are
  rejected. Result labels, JSON and practitioner guidance retain the same
  dates. The treated group's counterfactual survival is imputed from the
  control group (Theorem 1). Reports the absorption ATT at every
  post-treatment date plus its uniform average as `att`, with the paper's
  whole-individual pooled bootstrap (Appendix B Algorithm 1: centered
  absolute-deviation pointwise intervals and a simultaneous max-|t| band) and
  the Algorithm 2 fixed-anchor pre-treatment specification test
  (`results.pretest`, a `DurationDiDPretestResults` diagnostic). Every
  inference family is either fully available or fully withheld with a named
  `inference_status` (invalid imputed counterfactual curve, failed bootstrap
  draws, zero SE); failed draws are never retried or silently dropped.
  `results.aggregate("event_study")` returns the unified `EventStudyResults`
  container (event time 0 = first post-treatment date, reference -1).
  Covariates, staggered adoption, censoring, survey and cluster inference are
  deferred. `DiagnosticReport` and `BusinessReport` reject `DurationDiDResults`
  by type (their batteries are keyed to mean-outcome parallel-trends
  diagnostics; admission is tracked in `TODO.md`); `practitioner_next_steps`
  gains a DurationDiD handler with a hazard-restriction assumptions step.
  A tutorial notebook is deferred (tracked in `TODO.md`), a documented
  deviation from the new-estimator documentation checklist; the executed
  examples on the API page are the hands-on reference.
