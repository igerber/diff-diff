# diff-diff Library Roadmap

This document tracks the path from the current version to a solid 1.0 release, prioritized by practitioner value and academic credibility.

## What Makes a Credible 1.0?

A production-ready DiD library needs:
1. ✅ **Core estimators** - Basic DiD, TWFE, MultiPeriod, Staggered (Callaway-Sant'Anna), Synthetic DiD
2. ✅ **Valid inference** - Robust SEs, cluster SEs, wild bootstrap for few clusters
3. ✅ **Assumption diagnostics** - Parallel trends tests, placebo tests
4. ⚠️ **Sensitivity analysis** - What if parallel trends is violated? (Rambachan-Roth)
5. ⚠️ **Conditional parallel trends** - Covariate adjustment for staggered DiD
6. ⚠️ **Documentation** - API reference site for discoverability

---

## Quick Overview

| Feature | Status | Priority | Why It Matters |
|---------|--------|----------|----------------|
| **Honest DiD (Rambachan-Roth)** | Not Started | 1.0 Blocker | Reviewers expect sensitivity analysis |
| **CallawaySantAnna Covariates** | Not Implemented | 1.0 Blocker | Conditional PT often required in practice |
| **API Documentation Site** | Not Started | 1.0 Blocker | Credibility and discoverability |
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
**Status**: Not Started
**Effort**: High
**Practitioner Value**: ⭐⭐⭐⭐⭐

**Why this matters**: Pre-trends tests have low power and can exacerbate bias. Increasingly, journal reviewers and seminar audiences expect sensitivity analysis showing "how robust are results to violations of parallel trends?" This is becoming as standard as reporting robust SEs.

**Features needed**:
- Compute bounds under restrictions on trend deviations (relative magnitudes)
- Confidence intervals valid under partial identification
- Breakdown analysis: "How much violation would nullify the result?"
- Visualization of sensitivity curves

**References**:
- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*.
- R package: `HonestDiD`

### CallawaySantAnna Covariate Adjustment
**Status**: Not Implemented (parameter accepted but unused)
**Effort**: High
**Practitioner Value**: ⭐⭐⭐⭐⭐

**Why this matters**: In most applied settings, parallel trends only holds *conditional on covariates*. Without covariate adjustment, users must assume unconditional parallel trends, which is often implausible. The R `did` package supports this; we should too.

**Current state**:
- `covariates` parameter is accepted but silently ignored
- All three methods (dr, ipw, reg) currently reduce to difference-in-means
- Reference: `staggered.py:494-501`

**Implementation Notes**:
- Implement propensity score estimation for IPW
- Implement outcome regression for covariate adjustment
- Implement true doubly-robust estimation combining both
- Consider using cross-fitting for DR estimator

### API Documentation Site
**Status**: Not Started
**Effort**: Medium
**Practitioner Value**: ⭐⭐⭐⭐

**Why this matters**: For a 1.0 release, users should be able to find comprehensive API documentation online. Currently only docstrings and README exist.

**Options**:
- Sphinx + ReadTheDocs (traditional, well-supported)
- mkdocs-material (modern, clean look)
- pdoc (simple, auto-generates from docstrings)

**Should include**:
- Full API reference
- "When to use which estimator" decision guide
- Comparison with R packages (`did`, `HonestDiD`, `synthdid`)

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

1. **CallawaySantAnna Covariates** - Makes the staggered estimator production-ready
2. **Honest DiD (Rambachan-Roth)** - Addresses the key credibility gap
3. **API Documentation Site** - Professional presentation
4. **Goodman-Bacon Decomposition** - Key diagnostic for TWFE users
5. **Power Analysis** - Study design tool practitioners need

With these five additions, diff-diff would be competitive with R's `did` + `HonestDiD` ecosystem.
