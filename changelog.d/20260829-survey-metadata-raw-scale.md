### Fixed
- **`survey_metadata` raw-scale provenance on the unit-level recompute**
  (CallawaySantAnna panel + repeated-cross-section lanes,
  `TripleDifference`/`StaggeredTripleDifference` staggered engine,
  `ContinuousDiD` analytical branch, `EfficientDiD`): the recompute passed
  the RESOLVED (mean-1 rescaled) weights as `compute_survey_metadata`'s
  raw weights, so `sum_weights`/`weight_range` reported the normalized
  scale instead of the user's original weight scale. They now report the
  raw scale, matching every other estimator (DMLDiD got the pattern in
  its survey PR). For previously-successful fits whose survey design does
  not alias a mutated role column, this is metadata-provenance only:
  estimates, SEs, p-values, CIs, `df_survey`, `n_strata`, `n_psu` are
  byte-identical, and `effective_n`/`design_effect` are scale-invariant
  (unchanged within floating-point round-off). Additionally,
  `ContinuousDiD`'s zero-dose-unit filter now re-resolves the survey
  design from pristine input rows: a design column aliasing a mutated
  role column (e.g. `weights` naming the dose column) previously
  zero-weighted every never-treated unit on filtered fits (failing with
  "No valid (g,t) cells"); such fits now estimate under the user's
  original weights, consistent with the unfiltered path.
