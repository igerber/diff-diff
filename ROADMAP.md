# diff-diff Roadmap

This document outlines the feature roadmap for diff-diff, prioritized by practitioner value and academic credibility.

## What Makes a Credible 1.0?

A production-ready DiD library needs:

1. ✅ **Core estimators** - Basic DiD, TWFE, MultiPeriod, Staggered (Callaway-Sant'Anna), Synthetic DiD
2. ✅ **Valid inference** - Robust SEs, cluster SEs, wild bootstrap for few clusters
3. ✅ **Assumption diagnostics** - Parallel trends tests, placebo tests
4. ✅ **Sensitivity analysis** - What if parallel trends is violated? (Rambachan-Roth)
5. ✅ **Conditional parallel trends** - Covariate adjustment for staggered DiD
6. ✅ **Documentation** - API reference site for discoverability

**All 1.0 blockers are complete.** diff-diff has feature parity with R's `did` + `HonestDiD` ecosystem for core DiD analysis.

---

## Status Overview

| Feature | Status | Priority | Why It Matters |
|---------|--------|----------|----------------|
| Honest DiD (Rambachan-Roth) | ✅ Done | — | Reviewers expect sensitivity analysis |
| CallawaySantAnna Covariates | ✅ Done | — | Conditional PT often required in practice |
| API Documentation Site | ✅ Done | — | Credibility and discoverability |
| Goodman-Bacon Decomposition | ✅ Done | — | Explains when TWFE fails |
| Power Analysis | Not Started | 1.0 Target | Study design tool |
| CallawaySantAnna Bootstrap | Not Started | 1.0 Target | Better inference with few clusters |
| Sun-Abraham Estimator | Not Started | Post-1.0 | Alternative to CS, some prefer it |
| Gardner's did2s | Not Started | Post-1.0 | Two-stage approach, available in pyfixest |
| Local Projections DiD | Not Started | Post-1.0 | Dynamic effects (Dube et al. 2023) |
| Borusyak-Jaravel-Spiess | Not Started | Post-1.0 | More efficient under homogeneous effects |
| Double/Debiased ML | Not Started | Post-1.0 | High-dimensional covariates |

---

## 1.0 Target Features

These would strengthen the 1.0 release but aren't strictly blocking.

### ✅ Goodman-Bacon Decomposition (Done)

Helps users understand *why* TWFE can be biased with staggered adoption. Shows weights on "forbidden comparisons" (already-treated as controls). Essential diagnostic before deciding whether to use Callaway-Sant'Anna.

- ✅ Decompose TWFE into 2x2 comparisons
- ✅ Show weights by comparison type (clean vs. forbidden)
- ✅ Visualization of decomposition (scatter and bar charts)
- ✅ Integration with `TwoWayFixedEffects.decompose()` method
- ✅ Automatic warning when TWFE detects staggered treatment timing

**Reference**: Goodman-Bacon (2021). *Journal of Econometrics*.

### Power Analysis Tools

Practitioners need to know "how many units/periods do I need to detect an effect of size X?" Currently no Python tool does this well for DiD.

- Minimum detectable effect given sample size
- Required sample size for target power
- Simulation-based power for staggered designs
- Visualization of power curves

### CallawaySantAnna Bootstrap Inference

With few clusters or groups, analytical SEs may be unreliable. Bootstrap provides valid inference. The R `did` package uses multiplier bootstrap.

- Multiplier/weighted bootstrap at unit level
- Aggregate bootstrap samples for overall ATT and event study
- Handle covariance between group-time effects

### Enhanced Visualization

- Synthetic control weight visualization (bar chart of unit weights)
- ✅ Bacon decomposition visualization (scatter and bar charts)
- Treatment adoption "staircase" plot

---

## Post-1.0 Features

These are valuable but can wait for future versions.

### Sun-Abraham Estimator

Alternative to Callaway-Sant'Anna using interaction-weighted approach. Some practitioners prefer it; provides a robustness check.

**Reference**: Sun & Abraham (2021). *Journal of Econometrics*.

### Gardner's Two-Stage DiD (did2s)

Two-stage approach to staggered DiD that first residualizes outcomes using untreated observations, then estimates treatment effects. Available in pyfixest (Python) and did2s (R).

**Reference**: Gardner (2022). *Two-stage differences in differences*.

### Local Projections DiD

Implements local projections for dynamic treatment effects. Flexible approach that doesn't require specifying the full dynamic structure. Gaining traction in applied work.

**Reference**: Dube, Girardi, Jordà, and Taylor (2023).

### Borusyak-Jaravel-Spiess Imputation Estimator

More efficient than Callaway-Sant'Anna when parallel trends holds across all periods. Uses imputation approach.

**Reference**: Borusyak, Jaravel, and Spiess (2024).

### Double/Debiased ML for DiD

For high-dimensional settings with many covariates. Uses ML for nuisance parameter estimation with cross-fitting.

**Reference**: Chernozhukov et al. (2018), Chang (2020).

### Alternative Inference Methods

- Randomization inference for small samples
- Bayesian DiD with prior on parallel trends
- Conformal inference for prediction intervals

---

## Release History

### v0.7.0 (Current)

- ✅ Goodman-Bacon decomposition for TWFE diagnostics
- ✅ `plot_bacon()` visualization (scatter and bar charts)
- ✅ `TwoWayFixedEffects.decompose()` integration
- ✅ Automatic staggered treatment warning in TWFE

### v0.6.0

- ✅ **All 1.0 Blockers Complete**
- ✅ Honest DiD sensitivity analysis (Rambachan & Roth 2023)
- ✅ CallawaySantAnna covariate adjustment (DR, IPW, Reg)
- ✅ API documentation site with Sphinx

### v0.5.0

- Wild cluster bootstrap (Rademacher, Webb, Mammen weights)
- Placebo tests module
- Tutorial notebooks

### v0.4.0

- Callaway-Sant'Anna estimator for staggered DiD
- Event study and group effects visualization
- Parallel trends testing utilities

### v0.3.0

- Synthetic Difference-in-Differences
- Multi-period DiD with event study
- Data preparation utilities

### v0.2.0

- Two-Way Fixed Effects estimator
- Fixed effects support (absorb parameter)
- Cluster-robust standard errors
- Formula interface

### v0.1.0

- Initial release with basic DiD estimator

---

## Contributing

Interested in contributing? See the [GitHub repository](https://github.com/igerber/diff-diff) for open issues. Features marked "Not Started" are good candidates for contributions.
