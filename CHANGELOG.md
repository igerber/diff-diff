# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0]

### Added
- **CallawaySantAnna covariate adjustment** for conditional parallel trends
  - Outcome regression (`estimation_method='reg'`)
  - Inverse probability weighting (`estimation_method='ipw'`)
  - Doubly robust estimation (`estimation_method='dr'`)
  - Pass covariates via `covariates` parameter in `fit()`
- **Honest DiD sensitivity analysis** (Rambachan & Roth 2023)
  - `HonestDiD` class for computing bounds under parallel trends violations
  - Relative magnitudes restriction (`DeltaRM`) - bounds post-treatment violations by pre-treatment
  - Smoothness restriction (`DeltaSD`) - bounds second differences of trend violations
  - Combined restrictions (`DeltaSDRM`)
  - FLCI and C-LF confidence interval methods
  - Breakdown value computation via `breakdown_value()`
  - Sensitivity analysis over M grid via `sensitivity_analysis()`
  - `HonestDiDResults` and `SensitivityResults` dataclasses
  - `compute_honest_did()` convenience function
  - `plot_sensitivity()` for sensitivity analysis visualization
  - `plot_honest_event_study()` for event study with honest CIs
  - Tutorial notebook: `docs/tutorials/05_honest_did.ipynb`
- **API documentation site** with Sphinx
  - Full API reference auto-generated from docstrings
  - "Which estimator should I use?" decision guide
  - Comparison with R packages (did, HonestDiD)
  - Getting started / quickstart guide

### Changed
- Updated mypy configuration for better numpy type compatibility
- Modernized ruff configuration to use `[tool.ruff.lint]` section

### Fixed
- Fixed 21 ruff linting issues (import ordering, unused variables, ambiguous names)
- Fixed 94 mypy type checking issues (Optional types, numpy type casts, assertions)
- Added missing return statement in `run_placebo_test()`

## [0.5.0]

### Added
- **Wild cluster bootstrap** for valid inference with few clusters
  - Rademacher weights (default, good for most cases)
  - Webb's 6-point distribution (recommended for <10 clusters)
  - Mammen's two-point distribution
  - `WildBootstrapResults` dataclass
  - `wild_bootstrap_se()` utility function
  - Integration with `DifferenceInDifferences` and `TwoWayFixedEffects` via `inference='wild_bootstrap'`
- **Placebo tests module** (`diff_diff.diagnostics`)
  - `placebo_timing_test()` - fake treatment timing test
  - `placebo_group_test()` - fake treatment group test
  - `permutation_test()` - permutation-based inference
  - `leave_one_out_test()` - sensitivity to individual treated units
  - `run_placebo_test()` - unified dispatcher for all test types
  - `run_all_placebo_tests()` - comprehensive diagnostic suite
  - `PlaceboTestResults` dataclass
- **Tutorial notebooks** in `docs/tutorials/`
  - `01_basic_did.ipynb` - Basic 2x2 DiD, formula interface, covariates, fixed effects, wild bootstrap
  - `02_staggered_did.ipynb` - Staggered adoption with Callaway-Sant'Anna
  - `03_synthetic_did.ipynb` - Synthetic DiD with unit/time weights
  - `04_parallel_trends.ipynb` - Parallel trends testing and diagnostics
- Comprehensive test coverage (380+ tests)

## [0.4.0]

### Added
- **Callaway-Sant'Anna estimator** for staggered difference-in-differences
  - `CallawaySantAnna` class with group-time ATT(g,t) estimation
  - Support for `never_treated` and `not_yet_treated` control groups
  - Aggregation methods: `simple`, `group`, `calendar`, `event_study`
  - `CallawaySantAnnaResults` with group-time effects and aggregations
  - `GroupTimeEffect` dataclass for individual effects
- **Event study visualization** via `plot_event_study()`
  - Works with `MultiPeriodDiDResults`, `CallawaySantAnnaResults`, or DataFrames
  - Publication-ready formatting with customization options
- **Group effects visualization** via `plot_group_effects()`
- **Parallel trends testing utilities**
  - `check_parallel_trends()` - simple slope-based test
  - `check_parallel_trends_robust()` - Wasserstein distance test
  - `equivalence_test_trends()` - TOST equivalence test

## [0.3.0]

### Added
- **Synthetic Difference-in-Differences** (`SyntheticDiD`)
  - Unit weight optimization for synthetic control
  - Time weight computation for pre-treatment periods
  - Placebo-based and bootstrap inference
  - `SyntheticDiDResults` with weight accessors
- **Multi-period DiD** (`MultiPeriodDiD`)
  - Event-study style estimation with period-specific effects
  - `MultiPeriodDiDResults` with `period_effects` dictionary
  - `PeriodEffect` dataclass for individual period results
- **Data preparation utilities** (`diff_diff.prep`)
  - `generate_did_data()` - synthetic data generation
  - `make_treatment_indicator()` - create treatment from categorical/numeric
  - `make_post_indicator()` - create post-treatment indicator
  - `wide_to_long()` - reshape wide to long format
  - `balance_panel()` - ensure balanced panel data
  - `validate_did_data()` - data validation
  - `summarize_did_data()` - summary statistics by group
  - `create_event_time()` - event time for staggered designs
  - `aggregate_to_cohorts()` - aggregate to cohort means
  - `rank_control_units()` - rank controls by similarity

## [0.2.0]

### Added
- **Two-Way Fixed Effects** (`TwoWayFixedEffects`)
  - Within-transformation for unit and time fixed effects
  - Efficient handling of high-dimensional fixed effects via `absorb`
- **Fixed effects support** in base `DifferenceInDifferences`
  - `fixed_effects` parameter for dummy variable approach
  - `absorb` parameter for within-transformation approach
- **Cluster-robust standard errors**
  - `cluster` parameter for cluster-robust inference
- **Formula interface**
  - R-style formulas like `"outcome ~ treated * post"`
  - Support for covariates in formulas

## [0.1.0]

### Added
- Initial release
- **Basic Difference-in-Differences** (`DifferenceInDifferences`)
  - sklearn-like API with `fit()` method
  - Column name interface for outcome, treatment, time
  - Heteroskedasticity-robust (HC1) standard errors
  - `DiDResults` dataclass with ATT, SE, p-value, confidence intervals
  - `summary()` and `print_summary()` methods
  - `to_dict()` and `to_dataframe()` export methods
  - `is_significant` and `significance_stars` properties

[0.6.0]: https://github.com/igerber/diff-diff/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/igerber/diff-diff/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/igerber/diff-diff/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/igerber/diff-diff/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/igerber/diff-diff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/igerber/diff-diff/releases/tag/v0.1.0
