### Added
- **Fit from a design lock, and design reconciliation in the reports.**
  - `DesignLock.fit(data, first_treat= | treatment=, post=)` builds the locked
    estimator, fits it on the analysis data with the locked columns and `fit()`
    keywords, and reconciles the result against the lock on that same frame. It
    returns the `DesignReconciliationResults`, with the estimate in a new
    `results` field. The configuration is the locked one by construction; the
    realized design (units, treated units, cohorts, post-window lengths) and the
    data in the diagnosed periods can still deviate.
  - `DesignReconciliationResults` gains `results` and `declared` (`False` when
    `DesignLock.fit` ran the analysis, `True` when the estimator and columns
    were declared to `DesignLock.reconcile`).
  - `DiagnosticReport` and `BusinessReport` accept `design_reconciliation=` and
    state whether the analysis is `match` or `deviates` against the lock, the
    deviating items and the number of checks that were not run: a top-level
    `design_reconciliation` key in both `to_dict()` schemas (additive, no schema
    version change), one sentence in `summary()` and a section in
    `full_report()`. `BusinessReport` adds the caveat topics `design_deviation`
    and `design_not_checked`. A reconciliation that was not made for the
    reported results is refused. The reports do not say when the design was
    fixed, which a lock does not record.
  - New tutorial "Design Diagnosis and the Design Lock" (Study Design group):
    diagnose candidate estimators on untreated history, lock one, fit the locked
    analysis after the rollout and report the reconciliation.
