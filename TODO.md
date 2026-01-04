# diff-diff Library Roadmap

This document tracks the path from the current version to a solid 1.0 release, prioritized by practitioner value and academic credibility.

## What Makes a Credible 1.0?

A production-ready DiD library needs:
1. ✅ **Core estimators** - Basic DiD, TWFE, MultiPeriod, Staggered (Callaway-Sant'Anna), Synthetic DiD
2. ✅ **Valid inference** - Robust SEs, cluster SEs, wild bootstrap for few clusters
3. ✅ **Assumption diagnostics** - Parallel trends tests, placebo tests
4. ⚠️ **Sensitivity analysis** - What if parallel trends is violated? (Rambachan-Roth)
5. ⚠️ **Conditional parallel trends** - Covariate adjustment for staggered DiD
6. ✅ **Documentation** - API reference site for discoverability

---

## Quick Overview

| Feature | Status | Priority | Why It Matters |
|---------|--------|----------|----------------|
| **Honest DiD (Rambachan-Roth)** | ✅ Implemented | 1.0 Blocker | Reviewers expect sensitivity analysis |
| **CallawaySantAnna Covariates** | ✅ Implemented | 1.0 Blocker | Conditional PT often required in practice |
| **API Documentation Site** | ✅ Implemented | 1.0 Blocker | Credibility and discoverability |
| Goodman-Bacon Decomposition | Not Started | 1.0 Target | Explains when TWFE fails |
| Power Analysis | Not Started | 1.0 Target | Study design tool |
| CallawaySantAnna Bootstrap | Not Implemented | 1.0 Target | Better inference with few clusters |
| Sun-Abraham Estimator | Not Started | Post-1.0 | Alternative to CS, some prefer it |
| Borusyak-Jaravel-Spiess | Not Started | Post-1.0 | More efficient under homogeneous effects |
| Double/Debiased ML | Not Started | Post-1.0 | High-dimensional covariates |

---

## 1.0 Blockers

These features are essential for a credible 1.0 release. Without them, the library has significant gaps compared to R alternatives.

### Honest DiD / Sensitivity Analysis (Rambachan-Roth)
**Status**: ✅ Implemented
**Effort**: High
**Practitioner Value**: ⭐⭐⭐⭐⭐

**Why this matters**: Pre-trends tests have low power and can exacerbate bias. Increasingly, journal reviewers and seminar audiences expect sensitivity analysis showing "how robust are results to violations of parallel trends?" This is becoming as standard as reporting robust SEs.

**Implemented features**:
- ✅ Relative magnitudes (ΔRM): Bounds post-treatment violations by M̄ × max pre-period violation
- ✅ Smoothness (ΔSD): Bounds on second differences of trend violations
- ✅ Combined restrictions (ΔSDRM): Both smoothness and relative magnitude bounds
- ✅ FLCI (Fixed Length Confidence Interval) for smoothness restrictions
- ✅ C-LF (Conditional Least Favorable) for relative magnitudes
- ✅ Breakdown analysis: Find smallest M where robust CI includes zero
- ✅ Sensitivity analysis over grid of M values
- ✅ Visualization: `plot_sensitivity()` and `plot_honest_event_study()`
- ✅ Comprehensive test suite (49 tests)
- ✅ Tutorial notebook: `docs/tutorials/05_honest_did.ipynb`

**Future extensions** (post-1.0):
- Improved C-LF implementation with direct optimization instead of grid search
- Support for CallawaySantAnnaResults (currently only MultiPeriodDiDResults)
- Event-study-specific bounds for each post-period
- Hybrid inference methods
- Simulation-based power analysis for honest bounds

**References**:
- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*.
- R package: `HonestDiD`

### CallawaySantAnna Covariate Adjustment
**Status**: ✅ Implemented
**Effort**: High
**Practitioner Value**: ⭐⭐⭐⭐⭐

**Why this matters**: In most applied settings, parallel trends only holds *conditional on covariates*. Without covariate adjustment, users must assume unconditional parallel trends, which is often implausible. The R `did` package supports this; we now do too.

**Implementation**:
- ✅ Outcome regression: Regresses outcome changes on covariates for control, predicts counterfactual for treated
- ✅ IPW: Estimates propensity scores via logistic regression, reweights controls to match treated covariate distribution
- ✅ Doubly robust: Combines outcome regression and IPW for double robustness (consistent if either model is correct)
- Covariates are extracted from base period for each group-time ATT(g,t) estimation
- Graceful fallback to unconditional estimation if covariate extraction fails

**Usage**:
```python
cs = CallawaySantAnna(estimation_method='dr')  # 'dr', 'ipw', or 'reg'
results = cs.fit(
    data,
    outcome='outcome',
    unit='unit',
    time='time',
    first_treat='first_treat',
    covariates=['x1', 'x2']  # Now fully functional!
)
```

### API Documentation Site
**Status**: ✅ Implemented
**Effort**: Medium
**Practitioner Value**: ⭐⭐⭐⭐

**Why this matters**: For a 1.0 release, users should be able to find comprehensive API documentation online.

**Implementation**:
- ✅ Sphinx + ReadTheDocs theme with autodoc and napoleon extensions
- ✅ Full API reference auto-generated from docstrings
- ✅ "Which estimator should I use?" decision guide with flowchart
- ✅ Comparison with R packages (`did`, `HonestDiD`, `synthdid`)
- ✅ Getting started / quickstart guide
- ✅ ReadTheDocs configuration for automated builds
- ✅ Module-by-module API documentation:
  - Estimators (DifferenceInDifferences, TWFE, MultiPeriodDiD, SyntheticDiD)
  - Staggered (CallawaySantAnna, CallawaySantAnnaResults, GroupTimeEffect)
  - Results (DiDResults, MultiPeriodDiDResults, SyntheticDiDResults, PeriodEffect)
  - Visualization (plot_event_study, plot_group_effects, plot_sensitivity, plot_honest_event_study)
  - Diagnostics (placebo tests, permutation tests, leave-one-out)
  - Honest DiD (HonestDiD, DeltaSD, DeltaRM, DeltaSDRM)
  - Utils (parallel trends testing, wild bootstrap)
  - Data Prep (generate_did_data, balance_panel, etc.)

**Build locally**: `cd docs && make html`

---

## 1.0 Target Features

These would strengthen the 1.0 release but aren't strictly blocking.

### Goodman-Bacon Decomposition
**Status**: Not Started
**Effort**: Medium
**Practitioner Value**: ⭐⭐⭐⭐

**Why this matters**: Helps users understand *why* TWFE can be biased with staggered adoption. Shows the weights on "forbidden comparisons" (already-treated as controls). Essential diagnostic before deciding whether to use Callaway-Sant'Anna.

**Implementation**:
- Decompose TWFE into 2x2 comparisons
- Show weights by comparison type (clean vs. forbidden)
- Visualization of decomposition
- Reference: Goodman-Bacon (2021)

### Power Analysis Tools
**Status**: Not Started
**Effort**: Medium
**Practitioner Value**: ⭐⭐⭐⭐

**Why this matters**: Practitioners need to know "how many units/periods do I need to detect an effect of size X?" Currently no Python tool does this well for DiD.

**Features**:
- Minimum detectable effect given sample size
- Required sample size for target power
- Simulation-based power for staggered designs
- Visualization of power curves

### CallawaySantAnna Bootstrap Inference
**Status**: Not Implemented (raises NotImplementedError)
**Effort**: Medium
**Practitioner Value**: ⭐⭐⭐

**Why this matters**: With few clusters or groups, analytical SEs may be unreliable. Bootstrap provides valid inference. The R `did` package uses multiplier bootstrap.

**Current state**:
- `n_bootstrap` parameter exists but raises NotImplementedError
- Reference: `staggered.py:488-492`

**Implementation Notes**:
- Implement multiplier/weighted bootstrap at unit level
- Aggregate bootstrap samples for overall ATT and event study
- Handle covariance between group-time effects

### Enhanced Visualization
**Status**: Partial
**Effort**: Low-Medium
**Practitioner Value**: ⭐⭐⭐

**Current**: `plot_event_study()` and `plot_group_effects()` work well.

**Additions for 1.0**:
- Synthetic control weight visualization (bar chart of unit weights)
- Bacon decomposition visualization
- Treatment adoption "staircase" plot

**Post-1.0**:
- Interactive Plotly support
- Comparison plots across specifications

---

## Post-1.0 Features

These are valuable but can wait for future versions.

### Sun-Abraham Estimator
**Status**: Not Started
**Effort**: Medium

Alternative to Callaway-Sant'Anna using interaction-weighted approach. Some practitioners prefer it; provides a robustness check.

**Reference**: Sun & Abraham (2021). *Journal of Econometrics*.

### Borusyak-Jaravel-Spiess Imputation Estimator
**Status**: Not Started
**Effort**: High

More efficient than Callaway-Sant'Anna when parallel trends holds across all periods. Uses imputation approach.

**Reference**: Borusyak, Jaravel, and Spiess (2024).

### Double/Debiased ML for DiD
**Status**: Not Started
**Effort**: High

For high-dimensional settings with many covariates. Uses ML for nuisance parameter estimation with cross-fitting.

**Reference**: Chernozhukov et al. (2018), Chang (2020).

### Alternative Inference Methods
**Status**: Research
**Effort**: High

- Randomization inference for small samples
- Bayesian DiD with prior on parallel trends
- Conformal inference for prediction intervals

---

## Technical Debt & Code Quality

Items to address as part of ongoing maintenance.

### Known Limitations in Current Code

| Issue | Location | Priority |
|-------|----------|----------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:944-951` | Low (edge case) |
| `predict()` raises NotImplementedError | `estimators.py:532-554` | Low (rarely needed) |
| SyntheticDiD bootstrap can fail silently | `estimators.py:1580-1654` | Medium |
| Diagnostics module error handling | `diagnostics.py:782-885` | Medium |

### Standard Error Consistency Audit
**Status**: Review Needed

Different estimators compute SEs differently:
- DifferenceInDifferences: HC1 or cluster-robust
- TwoWayFixedEffects: Always cluster-robust (unit level default)
- CallawaySantAnna: Simple difference-in-means SE (no clustering)
- SyntheticDiD: Bootstrap or placebo-based

Consider unified interface for SE type selection.

### Test Coverage for Edge Cases

Some edge cases to add tests for:
- Very few clusters (< 5) with wild bootstrap
- Unbalanced panels with missing periods
- Single treated unit scenarios
- Perfect collinearity detection

---

## Documentation Improvements

Beyond the API site:
- Troubleshooting section for common errors
- "Which estimator should I use?" decision tree
- Comparison of estimator outputs on same data
- Real-world data examples (currently synthetic only)

---

## Completed Features

### v0.5.2
- [x] **Honest DiD sensitivity analysis** (Rambachan & Roth 2023)
  - Relative magnitudes (ΔRM) and smoothness (ΔSD) restrictions
  - Combined restrictions (ΔSDRM)
  - FLCI and C-LF confidence interval methods
  - Breakdown value computation
  - Sensitivity analysis over M grid
  - `plot_sensitivity()` and `plot_honest_event_study()` visualization
  - HonestDiD, HonestDiDResults, SensitivityResults classes
  - DeltaSD, DeltaRM, DeltaSDRM restriction classes
  - Tutorial notebook: `05_honest_did.ipynb`
  - 49 comprehensive tests

### v0.5.1
- [x] Comprehensive test coverage for `utils.py` module (72 tests)
- [x] Tutorial notebooks in `docs/tutorials/`
  - Basic DiD, formula interface, covariates, fixed effects, wild bootstrap
  - Staggered adoption with Callaway-Sant'Anna
  - Synthetic DiD with unit/time weights
  - Parallel trends testing and diagnostics

### v0.5.0
- [x] Wild cluster bootstrap (Rademacher, Webb, Mammen weights)
- [x] Placebo tests module (fake timing, fake group, permutation, leave-one-out)

### v0.4.0
- [x] Callaway-Sant'Anna estimator for staggered DiD
- [x] Event study visualization
- [x] Group effects visualization
- [x] Parallel trends testing utilities

---

## Suggested 1.0 Milestone Plan

1. ✅ **CallawaySantAnna Covariates** - Makes the staggered estimator production-ready
2. ✅ **Honest DiD (Rambachan-Roth)** - Addresses the key credibility gap
3. ✅ **API Documentation Site** - Professional presentation
4. **Goodman-Bacon Decomposition** - Key diagnostic for TWFE users
5. **Power Analysis** - Study design tool practitioners need

With items 1-2 complete, diff-diff now has feature parity with R's `did` + `HonestDiD` ecosystem for core sensitivity analysis. The remaining items (3-5) will complete the 1.0 release.
