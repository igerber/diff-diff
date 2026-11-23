### Added
- **Design lock: carry the pre-specified estimator from `diagnose_design` into
  the analysis.** `DesignDiagnosisResults.lock(label, data=)` records one
  estimator as the choice made before the analysis and returns a `DesignLock`;
  `DesignLock.reconcile(estimator, outcome=, unit=, time=, ...)` later checks
  the analysis that was run against it and labels it `pre-specified` or
  `deviates`.
  - Held-out periods: when more than one estimator is compared,
    `diagnose_design` now leaves the last periods of the panel out of the
    comparison (new `n_holdout=`; default half of the periods at or after the
    latest adoption period, `n_holdout=0` compares on the whole panel).
    `lock()` reruns only the chosen estimator with those periods included, on
    fresh placebo draws, so its reported power is not taken from the runs used
    to choose it.
  - `DesignLock` is plain JSON data (`to_json()`, `DesignLock.from_json()`,
    `DESIGN_LOCK_SCHEMA_VERSION`) holding the chosen estimator and its
    configuration, the declared design, a fingerprint of the panel, every
    estimator that was compared and the chosen estimator's diagnosands.
    `to_estimator()` builds the estimator exactly as locked.
  - `reconcile` reports one check per item (estimator class, each constructor
    parameter, `fit()` keywords, column names, and with `data=` the fingerprint
    of the diagnosed periods, unit count, panel balance, treated count, cohort
    count and post-window lengths) in a `DesignReconciliationResults`. It
    compares what the analyst declares with the lock; `pre-specified` covers
    the listed checks only and a deviation is not graded.
  - The reconciliation idea follows Blair et al. (2019); held-out periods, the
    lock format and the label are library choices documented as Notes in
    `docs/methodology/REGISTRY.md` `## DesignDiagnosis`.

### Changed
- **`diagnose_design` data fingerprint encodes numbers by value.**
  `data_sha256` no longer distinguishes `1` from `1.0`, so the digest of a set
  of rows does not change when pandas stores an integer column as float because
  some other row is missing. `diagnose_design` has not been released, so no
  stored fingerprint is affected.
