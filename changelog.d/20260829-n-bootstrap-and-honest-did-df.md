### Changed
- **`n_bootstrap` type guards aligned onto `utils.validate_n_bootstrap`**
  for the estimators the M-081 sweep deliberately left out —
  `HeterogeneousAdoptionDiD`, `ChaisemartinDHaultfoeuille`, `TROP`,
  `SyntheticDiD` (jackknife lane included), plus the two HAD pretest
  helpers (`stute_test`, `stute_joint_pretest`): previously-accepted
  type-blind values now raise the shared message — `True` (silently ran as
  1 replicate on HAD/dCDH), floats like `2.5` (passed the `>= 2` floors),
  and bool/negative under SyntheticDiD's jackknife floor exemption. The
  estimator-specific floors are unchanged and keep their own messages for
  non-negative sub-floor integers (TROP/SDiD `n_bootstrap=1`, HAD `0`);
  NEGATIVE values now surface the shared validator's message instead of
  each estimator's former wording.
- **`honest_did` inference-df resolution consolidated** onto the shared
  `aggregation.resolve_inference_df()` (three duplicated precedence blocks
  removed). Same precedence; `HonestDiDResults.df_survey` is now
  float-typed (`31.0` where it was `31`), and a fractional `df_inference`
  is preserved instead of truncated.
