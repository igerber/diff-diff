# diff-diff Roadmap

This document outlines the feature roadmap for diff-diff, prioritized by practitioner value and academic credibility.

For past changes and release history, see [CHANGELOG.md](CHANGELOG.md).

---

## Current Status

diff-diff v2.7.5 is a **production-ready** DiD library with feature parity with R's `did` + `HonestDiD` + `synthdid` ecosystem for core DiD analysis, plus **unique survey support** — design-based variance estimation (Taylor linearization, replicate weights) integrated across all estimators. No R or Python package offers this combination:

- **Core estimators**: Basic DiD, TWFE, MultiPeriod, Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess Imputation, Synthetic DiD, Triple Difference (DDD), TROP, Two-Stage DiD (Gardner 2022), Stacked DiD (Wing et al. 2024), Continuous DiD (Callaway, Goodman-Bacon & Sant'Anna 2024)
- **Valid inference**: Robust SEs, cluster SEs, wild bootstrap, multiplier bootstrap, placebo-based variance
- **Assumption diagnostics**: Parallel trends tests, placebo tests, Goodman-Bacon decomposition
- **Sensitivity analysis**: Honest DiD (Rambachan-Roth), Pre-trends power analysis (Roth 2022)
- **Study design**: Power analysis tools
- **Data utilities**: Real-world datasets (Card-Krueger, Castle Doctrine, Divorce Laws, MPDTA), DGP functions for all supported designs
- **Survey support**: Full `SurveyDesign` with strata, PSU, FPC, weight types, replicate weights (BRR/Fay/JK1/JKn), Taylor linearization, DEFF diagnostics, subpopulation analysis — integrated across all estimators (see [survey-roadmap.md](docs/survey-roadmap.md))
- **Performance**: Optional Rust backend for accelerated computation; faster than R at scale (see [CHANGELOG.md](CHANGELOG.md) for benchmarks)

---

## Near-Term Enhancements (v2.8)

### Survey Phase 7: Completing the Survey Story

Close the remaining gaps for practitioners using major population surveys
(ACS, CPS, BRFSS, MEPS). See [survey-roadmap.md](docs/survey-roadmap.md) for
full details.

- **CS Covariates + IPW/DR + Survey** *(Implemented)*: DRDID nuisance IF
  corrections (PS + OR) under survey weights for all estimation methods.
- **Repeated Cross-Sections** *(Implemented)*: `panel=False` support for
  CallawaySantAnna using cross-sectional DRDID (Sant'Anna & Zhao 2020,
  Section 4). Supports BRFSS, ACS annual, CPS monthly.
- **Survey-Aware DiD Tutorial** *(Open)*: Jupyter notebook demonstrating
  the full workflow with realistic survey data.
- **HonestDiD + Survey Variance** *(Implemented)*: Survey df and full
  event-study VCV propagated to sensitivity analysis, with bootstrap/replicate
  diagonal fallback.

### Staggered Triple Difference (DDD)

Extend the existing `TripleDifference` estimator to handle staggered adoption settings.

- Group-time ATT(g,t) for DDD designs with variation in treatment timing
- Event study aggregation and pre-treatment placebo effects
- Multiplier bootstrap for valid inference in staggered settings

**Reference**: [Ortiz-Villavicencio & Sant'Anna (2025)](https://arxiv.org/abs/2505.09942). "Better Understanding Triple Differences Estimators." *Working Paper*. R package: `triplediff`.

---

## Medium-Term Enhancements

### Efficient DiD Estimators

Semiparametrically efficient versions of existing DiD/event-study estimators with 40%+ precision gains over current methods.

**Reference**: [Chen, Sant'Anna & Xie (2025)](https://arxiv.org/abs/2506.17729). *Working Paper*.

### de Chaisemartin-D'Haultfœuille Estimator

Handles treatment that switches on and off (reversible treatments), unlike most other methods.

- Allows units to move into and out of treatment
- Time-varying, heterogeneous treatment effects
- Comparison with never-switchers or flexible control groups

**Reference**: [de Chaisemartin & D'Haultfœuille (2020, 2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3980758). *American Economic Review*.

### Local Projections DiD

Implements local projections for dynamic treatment effects. Doesn't require specifying full dynamic structure.

- Flexible impulse response estimation
- Robust to misspecification of dynamics
- Natural handling of anticipation effects

**Reference**: Dube, Girardi, Jordà, and Taylor (2023).

### Nonlinear DiD

For outcomes where linear models are inappropriate (binary, count, bounded).

- Logit/probit DiD for binary outcomes
- Poisson DiD for count outcomes
- Proper handling of incidence rate ratios and odds ratios

**Reference**: [Wooldridge (2023)](https://academic.oup.com/ectj/article/26/3/C31/7250479). *The Econometrics Journal*.

### Causal Duration Analysis with DiD

Extends DiD to duration/survival outcomes where standard methods fail (hazard rates, time-to-event).

- Duration analogue of parallel trends on hazard rates
- Avoids distributional assumptions and hazard function specification

**Reference**: [Deaner & Ku (2025)](https://www.aeaweb.org/conference/2025/program/paper/k77Kh8iS). *AEA Conference Paper*.

---

## Long-Term Research Directions (v3.0+)

Frontier methods requiring more research investment.

### DiD with Interference / Spillovers

Standard DiD assumes SUTVA; spatial/network spillovers violate this. Two-stage imputation approach estimates treatment AND spillover effects under staggered timing.

**Reference**: [Butts (2024)](https://arxiv.org/abs/2105.03737). *Working Paper*.

### Quantile/Distributional DiD

Recover the full counterfactual distribution and quantile treatment effects (QTT), not just mean ATT. Goes beyond "what's the average effect" to "who gains, who loses."

- Changes-in-Changes (CiC) identification strategy
- QTT(τ) at user-specified quantiles
- Full counterfactual distribution function
- Two-period foundation, then staggered extension

**Reference**: [Athey & Imbens (2006)](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0262.2006.00668.x). *Econometrica*.

### CATT Meta-Learner for Heterogeneous Effects

ML-powered conditional ATT — discover who benefits most from treatment using doubly robust meta-learner.

**Reference**: [Lan, Chang, Dillon & Syrgkanis (2025)](https://arxiv.org/abs/2502.04699). *Working Paper*.

### Causal Forests for DiD

Machine learning methods for discovering heterogeneous treatment effects in DiD settings.

- Estimate treatment effect heterogeneity across covariates
- Data-driven subgroup discovery
- Honest confidence intervals for discovered heterogeneity

**References**:
- [Kattenberg, Scheer & Thiel (2023)](https://ideas.repec.org/p/cpb/discus/452.html). *CPB Discussion Paper*.
- Athey & Wager (2019). *Annals of Statistics*.

### Matrix Completion Methods

Unified framework encompassing synthetic control and regression approaches.

- Nuclear norm regularization for low-rank structure
- Bridges synthetic control (few units, many periods) and regression (many units, few periods)

**Reference**: [Athey et al. (2021)](https://arxiv.org/abs/1710.10251). *Journal of the American Statistical Association*.

### Double/Debiased ML for DiD

For high-dimensional settings with many potential confounders.

- ML for nuisance parameter estimation (propensity, outcome models)
- Cross-fitting for valid inference

**Reference**: Chernozhukov et al. (2018). *The Econometrics Journal*.

### Alternative Inference Methods

- **Randomization inference**: Exact p-values for small samples
- **Bayesian DiD**: Priors on parallel trends violations
- **Conformal inference**: Prediction intervals with finite-sample guarantees

---

## Infrastructure Improvements

- Video tutorials and worked examples

---

## Contributing

Interested in contributing? Features in the "Near-Term" and "Medium-Term" sections are good candidates. See the [GitHub repository](https://github.com/igerber/diff-diff) for open issues.

Key references for implementation:
- [Roth et al. (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0304407623001318). "What's Trending in Difference-in-Differences?" *Journal of Econometrics*.
- [Baker et al. (2025)](https://arxiv.org/pdf/2503.13323). "Difference-in-Differences Designs: A Practitioner's Guide."
