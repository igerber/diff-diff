# diff-diff Roadmap

This document outlines the feature roadmap for diff-diff, prioritized by practitioner value and academic credibility.

For past changes and release history, see [CHANGELOG.md](CHANGELOG.md).

---

## Current Status

diff-diff v2.0.0 is a **production-ready** DiD library with feature parity with R's `did` + `HonestDiD` + `synthdid` ecosystem for core DiD analysis:

- **Core estimators**: Basic DiD, TWFE, MultiPeriod, Callaway-Sant'Anna, Sun-Abraham, Synthetic DiD, Triple Difference (DDD)
- **Valid inference**: Robust SEs, cluster SEs, wild bootstrap, multiplier bootstrap, placebo-based variance
- **Assumption diagnostics**: Parallel trends tests, placebo tests, Goodman-Bacon decomposition
- **Sensitivity analysis**: Honest DiD (Rambachan-Roth), Pre-trends power analysis (Roth 2022)
- **Study design**: Power analysis tools
- **Performance**: Optional Rust backend for accelerated computation; faster than R at scale (see [CHANGELOG.md](CHANGELOG.md) for benchmarks)

---

## Near-Term Enhancements (v2.1)

High-value additions building on our existing foundation.

### Borusyak-Jaravel-Spiess Imputation Estimator

More efficient than Callaway-Sant'Anna when treatment effects are homogeneous across groups/time. Uses imputation rather than aggregation.

- Imputes untreated potential outcomes using pre-treatment data
- More efficient under homogeneous effects assumption
- Can handle unbalanced panels more naturally

**Reference**: Borusyak, Jaravel, and Spiess (2024). *Review of Economic Studies*.

### Gardner's Two-Stage DiD (did2s)

Two-stage approach gaining traction in applied work. First residualizes outcomes, then estimates effects.

- Stage 1: Estimate unit and time FEs using only untreated observations
- Stage 2: Regress residualized outcomes on treatment indicators
- Clean separation of identification and estimation

**Reference**: Gardner (2022). *Working Paper*.

### Staggered Triple Difference (DDD)

Extend the existing `TripleDifference` estimator to handle staggered adoption settings. The current implementation handles 2-period DDD; this extends to multi-period designs.

**Multi-period/Staggered Support:**
- Group-time ATT(g,t) for DDD designs with variation in treatment timing
- Handle settings where groups adopt at different times
- Multiple comparison groups (never-treated, not-yet-treated in either dimension)
- `StaggeredTripleDifference` class or extended `TripleDifference` with `first_treat` parameter

**Event Study Aggregation:**
- Dynamic treatment effects over time (event study coefficients)
- Pre-treatment placebo effects for parallel trends assessment
- `aggregate='event_study'` parameter like `CallawaySantAnna`
- Integration with `plot_event_study()` visualization

**Multiplier Bootstrap Inference:**
- Multiplier bootstrap for valid inference in staggered settings
- Rademacher, Mammen, and Webb weight options (matching existing estimators)
- `n_bootstrap` parameter and `DDDBootstrapResults` class
- Clustered bootstrap for panel data

**Reference**: [Ortiz-Villavicencio & Sant'Anna (2025)](https://arxiv.org/abs/2505.09942). *Working Paper*. R package: `triplediff`.

### Enhanced Visualization

- Synthetic control weight visualization (bar chart of unit weights)
- Treatment adoption "staircase" plot for staggered designs
- Interactive plots with plotly backend option

---

## Medium-Term Enhancements (v2.2+)

Extending diff-diff to handle more complex settings.

### Continuous Treatment DiD

Many treatments have dose/intensity rather than binary on/off. Active research area with recent breakthroughs.

- Treatment effect on treated (ATT) parameters under generalized parallel trends
- Dose-response curves and marginal effects
- Handle settings where "dose" varies across units and time
- Event studies with continuous treatments

**References**:
- [Callaway, Goodman-Bacon & Sant'Anna (2024)](https://arxiv.org/abs/2107.02637). *NBER Working Paper*.
- [de Chaisemartin, D'Haultfœuille & Vazquez-Bare (2024)](https://arxiv.org/abs/2402.05432). *AEA Papers and Proceedings*.

### de Chaisemartin-D'Haultfœuille Estimator

Handles treatment that switches on and off (reversible treatments), unlike most other methods.

- Allows units to move into and out of treatment
- Time-varying, heterogeneous treatment effects
- Comparison with never-switchers or flexible control groups
- Different assumptions than CS/SA—useful for different settings

**Reference**: [de Chaisemartin & D'Haultfœuille (2020, 2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3980758). *American Economic Review*.

### Local Projections DiD

Implements local projections for dynamic treatment effects. Doesn't require specifying full dynamic structure.

- Flexible impulse response estimation
- Robust to misspecification of dynamics
- Natural handling of anticipation effects
- Growing use in macroeconomics and policy evaluation

**Reference**: Dube, Girardi, Jordà, and Taylor (2023).

### Nonlinear DiD

For outcomes where linear models are inappropriate (binary, count, bounded).

- Logit/probit DiD for binary outcomes
- Poisson DiD for count outcomes
- Flexible strategies for staggered designs with nonlinear models
- Proper handling of incidence rate ratios and odds ratios

**Reference**: [Wooldridge (2023)](https://academic.oup.com/ectj/article/26/3/C31/7250479). *The Econometrics Journal*.

### Doubly Robust DiD + Synthetic Control

Unified framework combining DiD and synthetic control with doubly robust identification—valid under *either* parallel trends or synthetic control assumptions.

- ATT identified under parallel trends OR group-level SC condition
- Semiparametric estimation framework
- Multiplier bootstrap for valid inference under either assumption
- Strengthens credibility by avoiding the DiD vs. SC trade-off

**Reference**: [Sun, Xie & Zhang (2025)](https://arxiv.org/abs/2503.11375). *Working Paper*.

### Causal Duration Analysis with DiD

Extends DiD to duration/survival outcomes where standard methods fail (hazard rates, time-to-event).

- Duration analogue of parallel trends on hazard rates
- Avoids distributional assumptions and hazard function specification
- Visual and formal pre-trends assessment for duration data
- Handles absorbing states approaching probability bounds

**Reference**: [Deaner & Ku (2025)](https://www.aeaweb.org/conference/2025/program/paper/k77Kh8iS). *AEA Conference Paper*.

---

## Triply Robust Panel Estimator (v2.3)

### TROP Estimator

The Triply Robust Panel (TROP) estimator from Athey, Imbens, Qu & Viviano (2025) combines three components for robust causal effect estimation in panel data:

1. **Flexible outcome model**: Low-rank factor structure on top of two-way fixed effects
2. **Unit weights**: Upweight control units similar to treated units (like synthetic control)
3. **Time weights**: Upweight time periods close to treatment (like SDID)

**Triple robustness property**: The estimator is asymptotically unbiased if *any one* of the three components removes the underlying bias. The bias depends on the *product* of unit-level imbalance, time-level imbalance, and regression adjustment error—a tighter bound than TWFE, SC, or SDID alone.

**Key advantages:**
- Outperforms TWFE/DiD, synthetic control, matrix completion, and SDID in simulations matched to real datasets
- More robust than existing methods across a wide range of empirically relevant settings
- Combines the strengths of regression adjustment, synthetic control weighting, and SDID time weighting

**Reference**: [Athey, S., Imbens, G., Qu, Z., & Viviano, D. (2025)](https://arxiv.org/abs/2508.21536). "Triply Robust Panel Estimators." *Working Paper*. Presented at the Journal of Applied Econometrics lecture, ASSA meetings, January 2025.

---

## Long-Term Research Directions (v3.0+)

Frontier methods requiring more research investment.

### Matrix Completion Methods

Unified framework encompassing synthetic control and regression approaches. Moves seamlessly between cross-sectional and time-series patterns.

- Nuclear norm regularization for low-rank structure
- Handles missing data patterns common in panel settings
- Bridges synthetic control (few units, many periods) and regression (many units, few periods)
- Confidence intervals via debiasing

**Reference**: [Athey et al. (2021)](https://arxiv.org/abs/1710.10251). *Journal of the American Statistical Association*.

### Causal Forests for DiD

Machine learning methods for discovering heterogeneous treatment effects in DiD settings.

- Estimate treatment effect heterogeneity across covariates
- Data-driven subgroup discovery
- Combine with DiD identification for observational data
- Honest confidence intervals for discovered heterogeneity

**References**:
- [Kattenberg, Scheer & Thiel (2023)](https://ideas.repec.org/p/cpb/discus/452.html). *CPB Discussion Paper*.
- Athey & Wager (2019). *Annals of Statistics*.

### Double/Debiased ML for DiD

For high-dimensional settings with many potential confounders.

- ML for nuisance parameter estimation (propensity, outcome models)
- Cross-fitting for valid inference
- Handles many covariates without overfitting concerns
- Doubly-robust estimation with ML flexibility

**Reference**: Chernozhukov et al. (2018). *The Econometrics Journal*.

### Alternative Inference Methods

- **Randomization inference**: Exact p-values for small samples
- **Bayesian DiD**: Priors on parallel trends violations
- **Conformal inference**: Prediction intervals with finite-sample guarantees

---

## Infrastructure Improvements

Ongoing maintenance and developer experience.

### Code Quality

- Extract shared within-transformation logic to utils
- ~~Consolidate linear regression helpers~~ ✓ Done (v2.1): Added `LinearRegression` helper class and `InferenceResult` dataclass in `linalg.py`. All major estimators (DifferenceInDifferences, TwoWayFixedEffects, SunAbraham, TripleDifference) now use the unified helper for coefficient extraction and inference.
- Consider splitting `staggered.py` (1800+ lines)

### Documentation

- Real-world data examples (beyond synthetic)
- Video tutorials and worked examples

---

## Contributing

Interested in contributing? Features in the "Near-Term" and "Medium-Term" sections are good candidates. See the [GitHub repository](https://github.com/igerber/diff-diff) for open issues.

Key references for implementation:
- [Roth et al. (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0304407623001318). "What's Trending in Difference-in-Differences?" *Journal of Econometrics*.
- [Baker et al. (2025)](https://arxiv.org/pdf/2503.13323). "Difference-in-Differences Designs: A Practitioner's Guide."
