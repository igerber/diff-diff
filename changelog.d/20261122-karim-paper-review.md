### Documentation
- Added a paper review of Karim, Nielsen, MacKinnon and Webb's *Improved
  Inference for CSDID Using the Cluster Jackknife* (arXiv 2602.12043v1) at
  `docs/methodology/papers/karim-2026-review.md`: the cluster-jackknife variance
  for the aggregate ATT (equation 20), its caveats when a deleted cluster removes
  a cohort, the Monte Carlo rejection frequencies, and the points the paper
  leaves unspecified. Source audit for prospective few-cluster inference on
  `CallawaySantAnna`; no estimator or public API is introduced.
