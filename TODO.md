# diff-diff Library Roadmap

This document tracks planned features and improvements for the diff-diff library.

## Priority 1: Critical Improvements

### Wild Cluster Bootstrap
**Status**: Not Started
**Effort**: Medium
**Impact**: High

Standard cluster-robust standard errors are biased with few clusters (<50). Wild bootstrap provides valid inference even with 5-10 clusters.

**Implementation Notes**:
- Add `wild_bootstrap_se()` function in `utils.py`
- Support Rademacher and Webb weights
- Integrate with existing estimators via parameter
- Reference: Cameron, Gelbach, and Miller (2008)

### Placebo Tests Module
**Status**: Not Started
**Effort**: Medium
**Impact**: Medium

Implement standard diagnostic tools for DiD:
- Fake treatment timing tests (assign treatment before it actually occurred)
- Fake treatment group tests (run DiD on never-treated units)
- Permutation-based inference

**Implementation Notes**:
- Create `diff_diff/diagnostics.py` module
- Add `run_placebo_test()` function
- Support multiple placebo specifications

---

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

### Add Test Coverage for Utils
**Status**: Not Started
**Effort**: Low
**Impact**: Medium

The `utils.py` module has no dedicated tests. Need coverage for:
- `check_parallel_trends()`
- `check_parallel_trends_robust()`
- `equivalence_test_trends()`
- Synthetic control weight functions

### Implement `predict()` Method
**Status**: Not Started
**Effort**: Low
**Impact**: Low

`DifferenceInDifferences.predict()` currently raises `NotImplementedError`. Implementation requires storing column names during fit.

---

## Documentation

### Example Notebooks
**Status**: Not Started
**Effort**: Medium
**Impact**: High

Create Jupyter notebooks demonstrating:
1. Basic 2x2 DiD with real-world data
2. Staggered adoption with CallawaySantAnna
3. Synthetic DiD walkthrough
4. Parallel trends testing and diagnostics
5. Visualization and reporting

### API Reference
**Status**: Partial
**Effort**: Medium
**Impact**: Medium

Docstrings exist but no built API documentation site. Consider:
- Sphinx/ReadTheDocs setup
- mkdocs-material

---

## Completed Features (v0.4.0)

- [x] Callaway-Sant'Anna estimator for staggered DiD
- [x] Event study visualization (`plot_event_study`)
- [x] Group effects visualization (`plot_group_effects`)
- [x] Export TwoWayFixedEffects in public API
- [x] Export parallel trends testing utilities
- [x] CallawaySantAnnaResults with event study and group aggregation
- [x] Comprehensive test coverage for new estimator (17 tests)
