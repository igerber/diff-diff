### Added
- **Planned pre-trend power at the design stage.**
  `diagnose_design(..., event_study=True)` keeps, for each `CallawaySantAnna`
  and `StackedDiD` estimator, a placebo event study: the mean and the
  covariance, across the no-effect placebo draws, of the event-study
  coefficients of each fit (`DesignDiagnosisResults.event_study_results`, an
  `EventStudyResults` labelled `estimand="placebo"`). `PreTrendsPower.fit`,
  `power_at`, `power_curve`, `compute_pretrends_power` and `compute_mdv` accept
  the diagnosis (or one of its surfaces) and return planned pre-trend power from
  Roth's (2022) formula before any outcome exists
  (`covariance_source="placebo_draws"`). The new
  `PreTrendsPower.power_by_pre_periods(diagnosis)` tabulates the minimum
  detectable violation and the test's no-violation rejection rate by the number
  of pre-periods tested.
  - The formula assumes that each coefficient's own test rejects at its nominal
    level. For the pre-periods being planned, `PreTrendsPower` counts how often
    the pre-trend test rejected in the placebo draws (none of which contains a
    violation) and compares that with the rate the formula assumes. Planned
    power is reported only where the two agree within Monte Carlo error: `fit`
    raises with both rates otherwise, and the by-pre-period table shows both
    rates on every row (`power_no_violation`, `placebo_rejection_rate`) and
    leaves the planned numbers blank on rows that disagree. In the library's
    simulation the formula matched direct trend injection on all 28 cells that
    passed this check (power at the planned minimum detectable violation 73%
    to 84% against a target of 80%), and every cell on which it did not match
    was one the check withholds (mostly 40-unit panels with 10 treated units).
  - Planned power is available for the default pre-test form
    (`pretest_form="nis"`) only.
  - No placebo event study is recorded for a `vcov_type="jackknife"` candidate
    or for the other estimator classes.
  - `HonestDiD`, `PreTrendsPower.sensitivity_to_honest_did`, and
    `precomputed["pretrends_power"]` in `DiagnosticReport` / `BusinessReport`
    refuse a placebo event study or a result planned from one.

### Fixed
- `PreTrendsPower.fit`, `power_at`, `power_curve` and
  `PreTrendsPowerResults.power_at` reported a power of 100% (default pre-test
  form) or of `alpha` (Wald form) for a violation magnitude `M` that was NaN.
  A non-finite `M` now raises `ValueError`.
- The `PreTrendsPower` example in the bundled `llms-full.txt` guide called a
  method and an argument that do not exist (`ptp.compute`, `M_grid=` on
  `compute_pretrends_power`).
