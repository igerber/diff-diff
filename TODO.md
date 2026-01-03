# diff-diff Library Roadmap

This document tracks planned features and improvements for the diff-diff library.

## Quick Overview

| Feature | Status | Priority | Effort |
|---------|--------|----------|--------|
| CallawaySantAnna Bootstrap | Not Implemented | P1 | Medium |
| CallawaySantAnna Covariates | Not Implemented | P1 | High |
| MultiPeriodDiD Wild Bootstrap | Not Implemented | P1 | Medium |
| `predict()` Method | Not Implemented | P1 | Low |
| SyntheticDiD Robustness | Partial | P1 | Medium |
| Honest DiD (Rambachan-Roth) | Not Started | P2 | High |
| Borusyak-Jaravel-Spiess | Not Started | P2 | High |
| Sun-Abraham Estimator | Not Started | P2 | Medium |
| Double/Debiased ML | Not Started | P3 | High |
| Power Analysis | Not Started | P4 | Medium |
| Enhanced Visualization | Partial | P4 | Low-Medium |
| Goodman-Bacon Decomposition | Not Started | P4 | Medium |
| API Documentation Site | Not Started | Doc | Medium |

**Legend**: P1 = Complete existing, P2 = Advanced methods, P3 = ML extensions, P4 = Usability

---

## Priority 1: Complete Existing Implementations

These are features that are partially implemented or documented as limitations in existing estimators. Completing these would provide a more robust foundation before adding new methods.

### CallawaySantAnna Bootstrap Inference
**Status**: Not Implemented (raises NotImplementedError)
**Effort**: Medium
**Impact**: High

The `n_bootstrap` parameter exists but bootstrap inference is not implemented. Currently only analytical standard errors are available.

**Implementation Notes**:
- Implement unit-level block bootstrap for group-time ATT(g,t) effects
- Properly aggregate bootstrap samples for overall ATT and event study effects
- Handle covariance between group-time effects in aggregation
- Reference: `staggered.py:488-492` raises NotImplementedError

### CallawaySantAnna Covariate Adjustment
**Status**: Not Implemented (parameter accepted but unused)
**Effort**: High
**Impact**: High

Covariates parameter is accepted but currently unused. The implementation uses unconditional parallel trends.

**Implementation Notes**:
- Implement propensity score estimation for IPW
- Implement outcome regression for covariate adjustment
- Implement true doubly-robust estimation combining both
- Currently all three methods (dr, ipw, reg) reduce to difference-in-means without covariates
- Reference: `staggered.py:494-501` warns that covariates are not used

### MultiPeriodDiD Wild Bootstrap
**Status**: Not Implemented (warns and falls back to analytical)
**Effort**: Medium
**Impact**: Medium

Wild cluster bootstrap is supported for basic DiD and TWFE, but not for MultiPeriodDiD.

**Implementation Notes**:
- Challenge: Multiple coefficients of interest (period-specific effects)
- Need to handle joint inference across period effects
- Consider implementing Wald-type joint test
- Reference: `estimators.py:944-951` warns and falls back

### Implement `predict()` Method
**Status**: Not Implemented (raises NotImplementedError)
**Effort**: Low
**Impact**: Low

`DifferenceInDifferences.predict()` exists but raises NotImplementedError. Requires storing column names during fit.

**Implementation Notes**:
- Store column name information during `fit()`
- Reconstruct design matrix for new data
- Reference: `estimators.py:532-554`

### SyntheticDiD Robustness Improvements
**Status**: Partial
**Effort**: Medium
**Impact**: Medium

Bootstrap SE computation can silently fail and skip iterations.

**Improvements needed**:
- Better handling of failed bootstrap iterations
- Warning when significant proportion of bootstraps fail
- Support for multiple treated units with individual weights
- Jackknife-based inference as alternative to bootstrap
- Reference: `estimators.py:1580-1654` silently catches exceptions

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
- Simulation-based power analysis for staggered designs

### Enhanced Visualization
**Status**: Partial
**Effort**: Low-Medium
**Impact**: Medium

Current: `plot_event_study()` and `plot_group_effects()` implemented with matplotlib.

**Additions needed**:
- Pre-trends shading with significance markers (partially done)
- Comparison plots across specifications (e.g., overlay multiple models)
- Synthetic control weight visualization (unit weights bar chart, time weights)
- Treatment adoption "staircase" plot for staggered designs
- Interactive plots (optional Plotly support)
- Bacon decomposition visualization for TWFE diagnostics

**Current limitations**:
- matplotlib is required but only lazy-imported
- Reference: `visualization.py:157-163`

### Improved Formula Interface
**Status**: Partial
**Effort**: Low-Medium
**Impact**: Low

Current: Basic formula support (`outcome ~ treated * post`) works.

**Limitations**:
- Only single interaction supported (`estimators.py:443-444`)
- No polynomial terms (e.g., `I(x**2)`)
- No factor notation (`C()` for categorical)
- No transformation functions (`log()`, `scale()`)

**Enhancements**:
- Support for multiple interactions
- Integration with patsy/formulaic for full R-style formulas
- Better error messages for unsupported syntax

### Goodman-Bacon Decomposition
**Status**: Not Started
**Effort**: Medium
**Impact**: Medium

Diagnostic tool showing how TWFE estimate is a weighted average of 2x2 DiD comparisons.

**Implementation Notes**:
- Decompose TWFE into timing groups and clean/forbidden comparisons
- Visualization of weights by comparison type
- Reference: Goodman-Bacon (2021)

---

## Code Quality & Technical Debt

### Diagnostics Module Improvements
**Status**: Partial
**Effort**: Low
**Impact**: Medium

The `run_all_placebo_tests()` function can fail silently or produce confusing errors.

**Issues**:
- Permutation and leave-one-out tests require binary post indicator, not multi-period time column
- Error messages stored in dict but easy to miss
- Consider adding validation and clearer messaging
- Reference: `diagnostics.py:782-885`

### Standard Error Computation Consistency
**Status**: Review Needed
**Effort**: Medium
**Impact**: Medium

Different estimators compute SEs differently, which may cause confusion.

**Audit needed**:
- DifferenceInDifferences: HC1 or cluster-robust
- TwoWayFixedEffects: Always cluster-robust (at unit level by default)
- CallawaySantAnna: Simple difference-in-means SE (no clustering currently)
- SyntheticDiD: Bootstrap or placebo-based
- Consider consistent interface for SE type selection

### Test Coverage for Edge Cases
**Status**: Partial
**Effort**: Medium
**Impact**: Medium

Some edge cases may not be well-tested:
- Very few clusters (< 5) with wild bootstrap
- Unbalanced panels with missing periods
- Single treated unit scenarios
- Perfect collinearity detection
- Zero variance in outcomes

---

## Documentation

### API Reference
**Status**: Partial
**Effort**: Medium
**Impact**: Medium

Docstrings exist but no built API documentation site. Consider:
- Sphinx/ReadTheDocs setup
- mkdocs-material

### Tutorial Improvements
**Status**: Completed (v0.5.1)
**Effort**: Low
**Impact**: Medium

Tutorials exist but could be enhanced:
- Add troubleshooting section for common errors
- Include comparison of estimator outputs on same data
- Add real-world data examples (currently synthetic only)
- Cover when to use which estimator decision tree

---

## Future Considerations

### Alternative Inference Methods
**Status**: Research
**Effort**: High
**Impact**: Medium

Methods to consider for future versions:
- Randomization inference for small samples
- Bayesian DiD with prior on parallel trends
- Conformal inference for prediction intervals

### Integration with Other Libraries
**Status**: Not Started
**Effort**: Medium
**Impact**: Low

Potential integrations:
- scikit-learn Pipeline compatibility
- pandas accessor (e.g., `df.did.fit(...)`)
- Export to Stata/R formats for comparison

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
