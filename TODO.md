# diff-diff Library Roadmap

This document tracks planned features and improvements for the diff-diff library.

## Priority 2: Advanced Methods

### Honest DiD / Sensitivity Analysis (Rambachan-Roth)
**Status**: Not Started
**Effort**: High
**Impact**: High

Pre-trends testing has low power and can exacerbate bias. Sensitivity analysis asks: "How robust are results to violations of parallel trends?"

**Features**:
- Compute bounds under restrictions on trend deviations
- Confidence intervals valid under partial identification
- Breakdown analysis visualization

**References**:
- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. Review of Economic Studies.
- R package: `HonestDiD`

### Borusyak-Jaravel-Spiess Imputation Estimator
**Status**: Not Started
**Effort**: High
**Impact**: Medium

Alternative to Callaway-Sant'Anna that's more efficient when parallel trends hold across all periods.

**Implementation Notes**:
- Impute Y(0) for treated observations using control outcomes
- Support both regression and matrix completion approaches
- Reference: Borusyak, Jaravel, and Spiess (2024)

### Sun-Abraham Estimator
**Status**: Not Started
**Effort**: Medium
**Impact**: Medium

Interaction-weighted estimator for staggered DiD. Focuses on "cohort-specific average treatment effects on the treated" (CATT).

**Reference**: Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics.

---

## Priority 3: Machine Learning Extensions

### Double/Debiased ML for DiD
**Status**: Not Started
**Effort**: High
**Impact**: Medium

For high-dimensional settings with many covariates. Uses machine learning for nuisance parameter estimation.

**Implementation Notes**:
- Integrate with scikit-learn estimators
- Support cross-fitting
- Implement DR-DiD with ML components
- Reference: Chernozhukov et al. (2018), Chang (2020)

### Parallel Trends Forest
**Status**: Not Started
**Effort**: High
**Impact**: Medium

Uses machine learning to construct optimal control samples when using DiD in relatively long panels with little randomization.

**Reference**: Shahn et al. (2023)

---

## Priority 4: Usability Enhancements

### Power Analysis Tools
**Status**: Not Started
**Effort**: Medium
**Impact**: Medium

Help practitioners determine sample size requirements:
- Minimum detectable effect given sample size
- Required sample size for target power
- Visualization of power curves

### Enhanced Visualization
**Status**: Partial
**Effort**: Low
**Impact**: Medium

Current: Basic event study plots implemented.

**Additions needed**:
- Pre-trends shading with significance markers
- Comparison plots across specifications
- Synthetic control weight visualization
- Interactive plots (optional Plotly support)

### Improved Formula Interface
**Status**: Not Started
**Effort**: Low
**Impact**: Low

Current formula support is basic. Enhancements:
- Support for multiple interactions
- Polynomial terms
- Factor notation (C() for categorical)
- Formula objects like patsy/formulaic

---

## Code Quality

### Implement `predict()` Method
**Status**: Not Started
**Effort**: Low
**Impact**: Low

`DifferenceInDifferences.predict()` currently raises `NotImplementedError`. Implementation requires storing column names during fit.

---

## Documentation

### API Reference
**Status**: Partial
**Effort**: Medium
**Impact**: Medium

Docstrings exist but no built API documentation site. Consider:
- Sphinx/ReadTheDocs setup
- mkdocs-material

---

## Completed Features (v0.5.1)

- [x] Comprehensive test coverage for `utils.py` module (72 new tests)
  - `validate_binary`, `compute_robust_se`, `compute_confidence_interval`, `compute_p_value`
  - `check_parallel_trends`, `check_parallel_trends_robust`, `equivalence_test_trends`
  - `compute_synthetic_weights`, `compute_time_weights`, `compute_placebo_effects`
  - `compute_sdid_estimator`, `_project_simplex`
- [x] Tutorial notebooks in `docs/tutorials/`
  - `01_basic_did.ipynb` - Basic DiD, formula interface, covariates, fixed effects, wild bootstrap
  - `02_staggered_did.ipynb` - Staggered adoption with Callaway-Sant'Anna
  - `03_synthetic_did.ipynb` - Synthetic DiD with unit/time weights
  - `04_parallel_trends.ipynb` - Parallel trends testing and diagnostics

## Completed Features (v0.5.0)

- [x] Wild cluster bootstrap for valid inference with few clusters (<50)
  - Rademacher, Webb (6-point), and Mammen weight types
  - Integration with DifferenceInDifferences and TwoWayFixedEffects via `inference='wild_bootstrap'`
  - Reference: Cameron, Gelbach, and Miller (2008)
- [x] Placebo tests module (`diff_diff/diagnostics.py`)
  - Fake timing test (`placebo_timing_test`)
  - Fake group test (`placebo_group_test`)
  - Permutation-based inference (`permutation_test`)
  - Leave-one-out sensitivity (`leave_one_out_test`)
  - Comprehensive suite (`run_all_placebo_tests`)
  - Reference: Bertrand, Duflo, and Mullainathan (2004)

## Completed Features (v0.4.0)

- [x] Callaway-Sant'Anna estimator for staggered DiD
- [x] Event study visualization (`plot_event_study`)
- [x] Group effects visualization (`plot_group_effects`)
- [x] Export TwoWayFixedEffects in public API
- [x] Export parallel trends testing utilities
- [x] CallawaySantAnnaResults with event study and group aggregation
- [x] Comprehensive test coverage for new estimator (17 tests)
