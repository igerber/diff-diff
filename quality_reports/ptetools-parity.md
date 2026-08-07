# `ptetools` Compatibility Scope

Status: targeted validation complete. The full `diff-diff` regression suite is
intentionally out of scope for this work.

## Frozen Scope

- Panel setup and two-period group-time subsets
- Panel and repeated-cross-section ATT(g,t) primitives
- Generic `pte` loop and post-fit group/dynamic aggregation
- `attgt_if` and `attgt_noif` result containers
- Callaway--Li `covid_attgt` via the DRDID-validated doubly-robust core
- QTT/QoTT containers, aggregation, and empirical bootstrap surfaces
- Dose-response result containers and `process_dose_gt`
- `ggpte` event-study plotting and `ggpte_cont` dose-response plotting
- `plot_qtt` overall and dynamic QTT plotting
- RCS covariate adjustment through the DRDID repeated-cross-section core
- Optional multiplier-bootstrap pointwise and simultaneous dynamic bands
- R-style aliases and public exports documented in `docs/api/ptetools.rst`

## Validation Results

- Targeted `ptetools` and docs IA tests: **60 passed**, 2 warnings
- QTT, dose-processing, and R DRDID parity subset: **17 passed**
- Ruff: **passed**
- Black: **passed**
- Mypy with Python 3.12 target: **zero errors**

## R Parity Evidence

- `covid_attgt`: ATT and influence functions match R `ptetools`/`DRDID` for
  levels, first differences, and covariate changes at the tested tolerance.
- QTT/QoTT: numerical parity verified on single- and two-cohort panels;
  QTT critical values match the R implementation.
- B-spline basis: `splines2::bSpline`/`dbs` design matrices match the R
  reference.
- `twfeweights` and `badcontrols`: existing R parity fixtures remain covered.
- The new `dr_ml_attgt` adapter's direct parametric probe is **not** claimed as
  exact parity yet: on the small bad-control fixture Python returns ATT
  `5.000000` while the installed R function returns `5.003463`. The adapter is
  contract-tested, but the estimator-level discrepancy needs a separate
  nuisance-model investigation before adding a numeric golden.
- Dose-response `process_dose_gt`: container and internal consistency are
  tested; a complete R end-to-end golden is unavailable because the reference
  per-cell dose estimator is outside this repository's R fixture surface.
  A direct R probe with a hand-built per-cell result also fails before the
  estimator output is produced (``invalid type``), confirming that this is an
  input-contract/reference-surface limitation rather than a missing Python
  comparison assertion.

## Intentional Python Differences

- Python returns dataclasses/DataFrames and matplotlib/Plotly objects instead
  of R S3 lists and ggplot objects.
- Influence functions use the project convention `phi = psi / n`; R DRDID
  exposes the unnormalized `psi` representation.
- `PTEResults.aggregate()` adds a Python post-fit inference surface with
  influence-function standard errors and normal-based confidence intervals.
