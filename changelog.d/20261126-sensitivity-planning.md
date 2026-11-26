### Added
- **Planned sensitivity analysis at the design stage: `HonestDiD.power_by_M`.**
  `diagnose_design(..., event_study=True)` now keeps the event-study container
  of every no-effect placebo draw, and `HonestDiD.power_by_M(diagnosis,
  M_grid=, effect_size=, power=)` runs the sensitivity analysis on each of
  them. It returns one row per `M`: the rate at which the robust interval
  excluded zero although no effect was present (`placebo_rejection_rate`), the
  interval's mean width (`ci_width`), the minimum detectable effect (`mde`),
  the power at `effect_size` (`power`), and the number of replicates scored
  and lost.
  - The numbers are counted, not derived from a formula. An effect that is the
    same across treated units moves a draw's robust interval by the value of
    the target parameter and changes nothing else, so the interval at any
    effect size follows from the no-effect draw. The library's tests check
    this against real injection for `CallawaySantAnna` and `StackedDiD`.
  - `mde` and `power` are left blank on a row that fails the size rule
    `diagnose_design` applies to its own MDE (the interval excludes zero too
    often with no effect present, or more than 10% of the replicates were
    lost); the rate is shown on every row. In the library's simulation the
    `method="smoothness"` interval at `M = 0` excluded zero in 14% to 45% of
    no-effect draws on 40-unit panels with 10 treated units, and all of those
    rows are withheld.
  - The planned `mde` matched real injection on all 72 rows of that
    simulation (76% to 85% power on fresh draws against a target of 80%).
  - `method="combined"` is not planned, and a fitted model is refused (use
    `sensitivity_analysis`). `HonestDiD.fit`, `sensitivity_analysis` and
    `breakdown_value` still refuse a placebo event study.
  - Tutorial 34 gains a section that plans the sensitivity analysis of its
    design.

### Changed
- A `DesignDiagnosisResults` from `diagnose_design(..., event_study=True)` is
  larger than before: each placebo event study carries the event-study
  container of every replicate (about 3 KB each). Nothing else in the
  diagnosis changes.
