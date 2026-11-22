### Added
- **`diagnose_design`: design diagnosis on your own untreated panel.** Before
  analysing a study, check how each candidate estimator behaves on the panel
  you will actually use. `diagnose_design(data, outcome=, unit=, time=,
  estimators=, cohort_periods=, n_treated= | treatment_fraction=, ...)`
  repeatedly assigns placebo treatments that follow the declared design,
  optionally adds a known effect to the placebo-treated observations, fits
  every estimator on the same placebo draws, and returns a
  `DesignDiagnosisResults` with, per estimator, the rejection rate under no
  effect (size) and the power, bias, RMSE and coverage at each injected effect,
  plus an optional minimum detectable effect (`power=`).
  - Outcome-blind: no real treatment effect is estimated. Pass `first_treat=`
    to drop every period from the earliest real adoption onward.
  - Estimators are returned in input order and never ranked. `size_controlled`
    flags those whose no-effect rejection rate is consistent with `alpha`; no
    minimum detectable effect is reported for the others.
  - Supports `DifferenceInDifferences`, `TwoWayFixedEffects`,
    `CallawaySantAnna`, `SunAbraham`, `ImputationDiD`, `TwoStageDiD`,
    `StackedDiD` and `EfficientDiD`, under common or staggered timing.
  - Sources: Bertrand, Duflo & Mullainathan (2004), Mizushima & Powell (2025),
    Coglianese & Fang (2026) and Blair et al. (2019). The size check, the
    minimum detectable effect on real data and the other choices with no paper
    source are documented as Notes in `docs/methodology/REGISTRY.md`
    `## DesignDiagnosis`.
