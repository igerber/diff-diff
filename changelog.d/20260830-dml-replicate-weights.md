### Added
- **DMLDiD replicate-weight survey designs**: `DMLDiD` now accepts
  replicate-weight `SurveyDesign`s (BRR / Fay / JK1 / JKn / SDR) on both
  lanes (panel and repeated cross sections), computing per-cell AND
  aggregate variances by IF-reweighting the augmented cross-fitted scores
  (`compute_replicate_if_variance`; nuisances are not re-estimated per
  replicate). Inference uses `df = rank(replicate matrix) - 1` with
  `min(df_survey, n_valid - 1)` capping; degenerate cells (zero or
  non-finite replicate variance) fail closed to NaN inference. Replicate +
  `cluster=` and replicate + `n_bootstrap > 0` are rejected with targeted
  errors (previously all replicate designs failed closed with a blanket
  `NotImplementedError`).
