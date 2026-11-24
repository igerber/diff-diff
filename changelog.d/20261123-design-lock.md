### Added
- **Design lock: carry the estimator chosen with `diagnose_design` into the
  analysis.** `DesignDiagnosisResults.lock(label, data=)` records one estimator
  and its configuration as a `DesignLock`;
  `DesignLock.reconcile(estimator, outcome=, unit=, time=, ...)` later checks
  the analysis that was run against it and returns the status `match` or
  `deviates`.
  - Lock rerun: when more than one estimator was compared, `lock()` reruns only
    the chosen one on the same panel with new placebo draws (pass the panel
    again as `data=`) and reports that run. This removes the part of the
    overstatement of the chosen estimator's power that comes from the placebo
    draws; the part that comes from the panel itself remains (about one
    percentage point in the simulation behind the registry note). The lock
    records the rerun's size verdict and warns when it differs from the
    comparison's.
  - `DesignLock` is plain JSON data (`to_json()`, `DesignLock.from_json()`,
    `DESIGN_LOCK_SCHEMA_VERSION`) holding the chosen estimator and its
    configuration, the declared design, a fingerprint of the panel, every
    estimator that was compared and the chosen estimator's diagnosands.
    `to_estimator()` builds the estimator exactly as locked. The lock does not
    record when it was made.
  - `reconcile` reports one check per item (estimator class, each constructor
    parameter, `fit()` keywords, column names, and with `data=` the fingerprint
    of the diagnosed periods, unit count, panel balance, treated count, cohort
    count and post-window lengths) in a `DesignReconciliationResults`. It
    compares what the analyst declares with the lock; `match` covers the checks
    that were run, a deviation is not graded, and the summary shows how many
    checks were not run next to the status.
  - The summaries of `DesignDiagnosisResults` and `DesignLock` state that random
    placebo assignment tests inference (size, power), not parallel trends.
  - The reconciliation idea follows Blair et al. (2019); the rerun, the lock
    format and the status are library choices documented as Notes in
    `docs/methodology/REGISTRY.md` `## DesignDiagnosis`, with the simulation in
    `benchmarks/python/design_lock_rerun.py`.

### Changed
- **`diagnose_design` data fingerprint encodes numbers by value.**
  `data_sha256` no longer distinguishes `1` from `1.0`, so the digest of a set
  of rows does not change when pandas stores an integer column as float because
  some other row is missing. `diagnose_design` has not been released, so no
  stored fingerprint is affected.
