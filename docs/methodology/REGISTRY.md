# Methodology Registry

This document provides the academic foundations and key implementation requirements for each estimator in diff-diff. It serves as a reference for contributors and users who want to understand the theoretical basis of the methods.

## Table of Contents

1. [Core DiD Estimators](#core-did-estimators)
   - [DifferenceInDifferences](#differenceinifferences)
   - [MultiPeriodDiD](#multiperioddid)
   - [TwoWayFixedEffects](#twowayfixedeffects)
2. [Modern Staggered Estimators](#modern-staggered-estimators)
   - [CallawaySantAnna](#callawaysantanna)
   - [SunAbraham](#sunabraham)
3. [Advanced Estimators](#advanced-estimators)
   - [SyntheticDiD](#syntheticdid)
   - [TripleDifference](#tripledifference)
   - [TROP](#trop)
4. [Diagnostics & Sensitivity](#diagnostics--sensitivity)
   - [BaconDecomposition](#bacondecomposition)
   - [HonestDiD](#honestdid)
   - [PreTrendsPower](#pretrendspower)
   - [PowerAnalysis](#poweranalysis)

---

# Core DiD Estimators

## DifferenceInDifferences

**Primary source:** Canonical econometrics textbooks
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press.
- Angrist, J.D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment and post indicators must be binary (0/1) with variation in both
- Warns if no treated units in pre-period or no control units in post-period
- Parallel trends assumption is untestable but can be assessed with pre-treatment data

*Estimator equation (as implemented):*
```
ATT = (Ȳ_{treated,post} - Ȳ_{treated,pre}) - (Ȳ_{control,post} - Ȳ_{control,pre})
    = E[Y(1) - Y(0) | D=1]
```

Regression form:
```
Y_it = α + β₁(Treated_i) + β₂(Post_t) + τ(Treated_i × Post_t) + X'γ + ε_it
```
where τ is the ATT.

*Standard errors:*
- Default: HC1 heteroskedasticity-robust
- Optional: Cluster-robust (specify `cluster` parameter)
- Optional: Wild cluster bootstrap for small number of clusters

*Edge cases:*
- Empty cells (e.g., no treated-pre observations) raise ValueError
- Singleton groups in clustering are dropped with warning
- Rank-deficient design matrix (collinearity): warns and sets NA for dropped coefficients (R-style, matches `lm()`)
  - Tolerance: `1e-07` (matches R's `qr()` default), relative to largest diagonal element of R in QR decomposition
  - Controllable via `rank_deficient_action` parameter: "warn" (default), "error", or "silent"

**Reference implementation(s):**
- R: `fixest::feols()` with interaction term
- Stata: `reghdfe` or manual regression with interaction

**Requirements checklist:**
- [ ] Treatment and time indicators are binary 0/1 with variation
- [ ] ATT equals coefficient on interaction term
- [ ] Wild bootstrap supports Rademacher, Mammen, Webb weight distributions
- [ ] Formula interface parses `y ~ treated * post` correctly

---

## MultiPeriodDiD

**Primary source:** Event study methodology
- Freyaldenhoven, S., Hansen, C., Pérez, J.P., & Shapiro, J.M. (2021). Visualization, identification, and estimation in the linear panel event-study design. NBER Working Paper 29170.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires multiple pre and post periods
- Reference period (typically t=-1) must be specified or defaulted
- Warns if treatment timing varies across units (suggests staggered estimator)

*Estimator equation (as implemented):*
```
Y_it = α_i + γ_t + Σ_{e≠-1} δ_e × 1(t - E_i = e) + X'β + ε_it
```
where E_i is treatment time for unit i, and δ_e are event-study coefficients.

*Standard errors:*
- Default: Cluster-robust at unit level
- Event-study coefficients use appropriate degrees of freedom

*Edge cases:*
- Unbalanced panels: only uses observations where event-time is defined
- Never-treated units: event-time indicators are all zero
- Endpoint binning: distant event times can be binned
- Rank-deficient design matrix (collinearity): warns and sets NA for dropped coefficients (R-style, matches `lm()`)
- Average ATT (`avg_att`) is NA if any post-period effect is unidentified (R-style NA propagation)

**Reference implementation(s):**
- R: `fixest::feols()` with `i(event_time, ref=-1)`
- Stata: `eventdd` or manual indicator regression

**Requirements checklist:**
- [ ] Reference period coefficient is zero (normalized)
- [ ] Pre-period coefficients test parallel trends assumption
- [ ] Supports both balanced and unbalanced panels
- [ ] Returns PeriodEffect objects with confidence intervals

---

## TwoWayFixedEffects

**Primary source:** Panel data econometrics
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press, Chapter 10.

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Staggered treatment warning**: If treatment timing varies across units, warns about potential bias from negative weights (Goodman-Bacon 2021, de Chaisemartin & D'Haultfœuille 2020)
- Requires sufficient within-unit and within-time variation
- Warns if any fixed effect is perfectly collinear with treatment

*Estimator equation (as implemented):*
```
Y_it = α_i + γ_t + τ(D_it) + X'β + ε_it
```
Estimated via within-transformation (demeaning):
```
Ỹ_it = τD̃_it + X̃'β + ε̃_it
```
where tildes denote demeaned variables.

*Standard errors:*
- Default: Cluster-robust at unit level (accounts for serial correlation)
- Degrees of freedom adjusted for absorbed fixed effects

*Edge cases:*
- Singleton units/periods are automatically dropped
- Treatment perfectly collinear with FE raises error with informative message listing dropped columns
- Covariate collinearity emits warning but estimation continues (ATT still identified)
- Rank-deficient design matrix: warns and sets NA for dropped coefficients (R-style, matches `lm()`)
- Unbalanced panels handled via proper demeaning

**Reference implementation(s):**
- R: `fixest::feols(y ~ treat | unit + time, data)`
- Stata: `reghdfe y treat, absorb(unit time) cluster(unit)`

**Requirements checklist:**
- [ ] Staggered treatment automatically triggers warning
- [ ] Auto-clusters standard errors at unit level
- [ ] `decompose()` method returns BaconDecompositionResults
- [ ] Within-transformation correctly handles unbalanced panels

---

# Modern Staggered Estimators

## CallawaySantAnna

**Primary source:** [Callaway, B., & Sant'Anna, P.H.C. (2021). Difference-in-Differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.](https://doi.org/10.1016/j.jeconom.2020.12.001)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires never-treated units as comparison group (identified by `first_treat=0` or `never_treated=True`)
- Warns if no never-treated units exist (suggests alternative comparison strategies)
- Limited pre-treatment periods reduce ability to test parallel trends

*Estimator equation (as implemented):*

Group-time average treatment effect:
```
ATT(g,t) = E[Y_t - Y_{g-1} | G_g=1] - E[Y_t - Y_{g-1} | C=1]
```
where G_g=1 indicates units first treated in period g, and C=1 indicates never-treated.

With covariates (doubly robust):
```
ATT(g,t) = E[((G_g - p̂_g(X))/(1-p̂_g(X))) × (Y_t - Y_{g-1} - m̂_{0,g,t}(X) + m̂_{0,g,g-1}(X))] / E[G_g]
```

Aggregations:
- Simple: `ATT = Σ_{g,t} w_{g,t} × ATT(g,t)` weighted by group size
- Event-study: `ATT(e) = Σ_g w_g × ATT(g, g+e)` for event-time e
- Group: `ATT(g) = Σ_t ATT(g,t) / T_g` average over post-periods

*Standard errors:*
- Default: Analytical (influence function-based)
- Bootstrap: Multiplier bootstrap with Rademacher, Mammen, or Webb weights
- Block structure preserves within-unit correlation

*Edge cases:*
- Groups with single observation: included but may have high variance
- Missing group-time cells: ATT(g,t) set to NaN
- Anticipation: `anticipation` parameter shifts reference period
- Rank-deficient design matrix (covariate collinearity):
  - Detection: Pivoted QR decomposition with tolerance `1e-07` (R's `qr()` default)
  - Handling: Warns and drops linearly dependent columns, sets NA for dropped coefficients (R-style, matches `lm()`)
  - Parameter: `rank_deficient_action` controls behavior: "warn" (default), "error", or "silent"
- Non-finite inference values:
  - Analytic SE: Returns NaN to signal invalid inference (not biased via zeroing)
  - Bootstrap: Drops non-finite samples, warns, and adjusts p-value floor accordingly
  - Threshold: Returns NaN if <50% of bootstrap samples are valid
  - **Note**: This is a defensive enhancement over reference implementations (R's `did::att_gt`, Stata's `csdid`) which may error or produce unhandled inf/nan in edge cases without informative warnings

**Reference implementation(s):**
- R: `did::att_gt()` (Callaway & Sant'Anna's official package)
- Stata: `csdid`

**Requirements checklist:**
- [ ] Requires never-treated units (first_treat=0 or equivalent)
- [ ] Bootstrap weights support Rademacher, Mammen, Webb distributions
- [ ] Aggregations: simple, event_study, group all implemented
- [ ] Doubly robust estimation when covariates provided
- [ ] Multiplier bootstrap preserves panel structure

---

## SunAbraham

**Primary source:** [Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.](https://doi.org/10.1016/j.jeconom.2020.09.006)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires never-treated units as control group
- Warns if treatment effects may be heterogeneous across cohorts (which the method handles)
- Reference period must be specified (default: e=-1)

*Estimator equation (as implemented):*

Saturated regression with cohort-specific effects:
```
Y_it = α_i + γ_t + Σ_{g∈G} Σ_{e≠-1} δ_{g,e} × 1(G_i=g) × D^e_{it} + ε_it
```
where G_i is unit i's cohort (first treatment period), D^e_{it} = 1(t - G_i = e).

Interaction-weighted estimator:
```
δ̂_e = Σ_g ŵ_{g,e} × δ̂_{g,e}
```
where weights ŵ_{g,e} = n_{g,e} / Σ_g n_{g,e} (sample share of cohort g at event-time e).

*Standard errors:*
- Default: Cluster-robust at unit level
- Delta method for aggregated coefficients
- Optional: Pairs bootstrap for robustness

*Edge cases:*
- Single cohort: reduces to standard event study
- Cohorts with no observations at some event-times: weighted appropriately
- Extrapolation beyond observed event-times: not estimated
- Rank-deficient design matrix (covariate collinearity):
  - Detection: Pivoted QR decomposition with tolerance `1e-07` (R's `qr()` default)
  - Handling: Warns and drops linearly dependent columns, sets NA for dropped coefficients (R-style, matches `lm()`)
  - Parameter: `rank_deficient_action` controls behavior: "warn" (default), "error", or "silent"

**Reference implementation(s):**
- R: `fixest::sunab()` (Laurent Bergé's implementation)
- Stata: `eventstudyinteract`

**Requirements checklist:**
- [ ] Never-treated units required as controls
- [ ] Interaction weights sum to 1 within each relative time period
- [ ] Reference period defaults to e=-1, coefficient normalized to zero
- [ ] Cohort-specific effects recoverable from results
- [ ] Cluster-robust SEs with delta method for aggregates

---

# Advanced Estimators

## SyntheticDiD

**Primary source:** [Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088-4118.](https://doi.org/10.1257/aer.20190159)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires balanced panel (same units observed in all periods)
- Warns if pre-treatment fit is poor (high RMSE)
- Treatment must be "block" structure: all treated units treated at same time

*Estimator equation (as implemented):*

```
τ̂^sdid = Σ_t λ_t (Ȳ_{tr,t} - Σ_j ω_j Y_{j,t})
```

Unit weights ω solve:
```
min_ω ||Ȳ_{tr,pre} - Σ_j ω_j Y_{j,pre}||₂² + ζ² ||ω||₂²
s.t. ω ≥ 0, Σ_j ω_j = 1
```

Time weights λ solve analogous problem matching pre/post means.

Regularization parameter:
```
ζ = (N_tr × T_post)^(1/4) × σ̂
```
where σ̂ is estimated noise level.

*Standard errors:*
- Default: Placebo variance estimator (Algorithm 4 in paper)
```
V̂ = ((r-1)/r) × Var({τ̂^(j) : j ∈ controls})
```
where τ̂^(j) is placebo estimate treating unit j as treated
- Alternative: Block bootstrap

*Edge cases:*
- Negative weights attempted: projected onto simplex
- Perfect pre-treatment fit: regularization prevents overfitting
- Single treated unit: valid, uses jackknife-style variance

**Reference implementation(s):**
- R: `synthdid::synthdid_estimate()` (Arkhangelsky et al.'s official package)

**Requirements checklist:**
- [ ] Unit weights: sum to 1, non-negative (simplex constraint)
- [ ] Time weights: sum to 1, non-negative (simplex constraint)
- [ ] Placebo SE formula: `sqrt((r-1)/r) * sd(placebo_estimates)`
- [ ] Regularization scales with panel dimensions
- [ ] Returns both unit and time weights for interpretation

---

## TripleDifference

**Primary source:** [Ortiz-Villavicencio, M., & Sant'Anna, P.H.C. (2025). Better Understanding Triple Differences Estimators. arXiv:2505.09942.](https://arxiv.org/abs/2505.09942)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires all 8 cells of the 2×2×2 design: Group(0/1) × Period(0/1) × Treatment(0/1)
- Warns if any cell has fewer than threshold observations
- Propensity score overlap required for IPW/DR methods

*Estimator equation (as implemented):*

Eight-cell structure:
```
τ^DDD = [(Ȳ₁₁₁ - Ȳ₁₀₁) - (Ȳ₀₁₁ - Ȳ₀₀₁)] - [(Ȳ₁₁₀ - Ȳ₁₀₀) - (Ȳ₀₁₀ - Ȳ₀₀₀)]
```
where subscripts are (Group, Period, Treatment eligibility).

Regression form:
```
Y = β₀ + β_G(G) + β_P(P) + β_T(T) + β_{GP}(G×P) + β_{GT}(G×T) + β_{PT}(P×T) + τ(G×P×T) + X'γ + ε
```

Doubly robust estimator:
```
τ̂^DR = E[(ψ_IPW(Y,D,X;π̂) + ψ_RA(Y,X;μ̂) - ψ_bias(X;π̂,μ̂))]
```

*Standard errors:*
- Regression adjustment: HC1 or cluster-robust
- IPW: Influence function-based (accounts for estimated propensity)
- Doubly robust: Efficient influence function

*Edge cases:*
- Propensity scores near 0/1: trimmed at `pscore_trim` (default 0.01)
- Empty cells: raises ValueError with diagnostic message
- Collinear covariates: automatic detection and warning

**Reference implementation(s):**
- Authors' replication code (forthcoming)

**Requirements checklist:**
- [ ] All 8 cells (G×P×T) must have observations
- [ ] Propensity scores clipped at `pscore_trim` bounds
- [ ] Doubly robust consistent if either propensity or outcome model correct
- [ ] Returns cell means for diagnostic inspection
- [ ] Supports RA, IPW, and DR estimation methods

---

## TROP

**Primary source:** [Athey, S., Imbens, G.W., Qu, Z., & Viviano, D. (2025). Triply Robust Panel Estimators. arXiv:2508.21536.](https://arxiv.org/abs/2508.21536)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires sufficient pre-treatment periods for factor estimation
- Warns if estimated rank seems too high/low relative to panel dimensions
- Unit weights can become degenerate if λ_unit too large

*Estimator equation (as implemented):*

Factor model:
```
Y_it = L_it + τ D_it + ε_it
```
where L = UΣV' is low-rank factor structure.

Factor estimation via nuclear norm regularization:
```
L̂ = argmin_L ||Y_control - L||_F² + λ_nn ||L||_*
```
Solved via soft-thresholding of singular values:
```
L̂ = U × soft_threshold(Σ, λ_nn) × V'
```

Unit weights:
```
ω_j = exp(-λ_unit × d(j, treated)) / Σ_k exp(-λ_unit × d(k, treated))
```
where d(j, treated) is RMSE distance to treated units in pre-period.

Time weights: analogous construction for periods.

*Standard errors:*
- Default: Block bootstrap preserving panel structure
- Alternative: Jackknife (leave-one-unit-out)

*Edge cases:*
- Rank selection: automatic via cross-validation, information criterion, or elbow
- Zero singular values: handled by soft-thresholding
- Extreme distances: weights regularized to prevent degeneracy

**Reference implementation(s):**
- Authors' replication code (forthcoming)

**Requirements checklist:**
- [ ] Factor matrix estimated via soft-threshold SVD
- [ ] Unit weights: `exp(-λ_unit × distance)` with normalization
- [ ] LOOCV implemented for tuning parameter selection
- [ ] Multiple rank selection methods: cv, ic, elbow
- [ ] Returns factor loadings and scores for interpretation

---

# Diagnostics & Sensitivity

## BaconDecomposition

**Primary source:** [Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.](https://doi.org/10.1016/j.jeconom.2021.03.014)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires variation in treatment timing (staggered adoption)
- Warns if only one treatment cohort (decomposition not meaningful)
- Assumes no never-treated: uses not-yet-treated as controls

*Estimator equation (as implemented):*

TWFE decomposes as:
```
τ̂^TWFE = Σ_k s_k × τ̂_k
```
where k indexes 2×2 comparisons and s_k are Bacon weights.

Three comparison types:
1. **Treated vs. Never-treated** (if never-treated exist):
   ```
   τ̂_{T,U} = (Ȳ_{T,post} - Ȳ_{T,pre}) - (Ȳ_{U,post} - Ȳ_{U,pre})
   ```

2. **Earlier vs. Later-treated** (Earlier as treated, Later as control pre-treatment):
   ```
   τ̂_{k,l} = DiD using early-treated as treatment, late-treated as control
   ```

3. **Later vs. Earlier-treated** (problematic: uses post-treatment outcomes as control):
   ```
   τ̂_{l,k} = DiD using late-treated as treatment, early-treated (post) as control
   ```

Weights depend on group sizes and variance in treatment timing.

*Standard errors:*
- Not typically computed (decomposition is exact)
- Individual 2×2 estimates can have SEs

*Edge cases:*
- Continuous treatment: not supported, requires binary
- Weights may be negative for later-vs-earlier comparisons
- Single treatment time: no decomposition possible

**Reference implementation(s):**
- R: `bacondecomp::bacon()`
- Stata: `bacondecomp`

**Requirements checklist:**
- [ ] Three comparison types: treated_vs_never, earlier_vs_later, later_vs_earlier
- [ ] Weights sum to approximately 1 (numerical precision)
- [ ] TWFE coefficient ≈ weighted sum of 2×2 estimates
- [ ] Visualization shows weight vs. estimate by comparison type

---

## HonestDiD

**Primary source:** [Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555-2591.](https://doi.org/10.1093/restud/rdad018)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires event-study estimates with pre-treatment coefficients
- Warns if pre-treatment coefficients suggest parallel trends violation
- M=0 corresponds to exact parallel trends assumption

*Estimator equation (as implemented):*

Identified set under smoothness restriction (Δ^SD):
```
Δ^SD(M) = {δ : |δ_t - δ_{t-1}| ≤ M for all pre-treatment t}
```

Identified set under relative magnitudes (Δ^RM):
```
Δ^RM(M̄) = {δ : |δ_post| ≤ M̄ × max_t |δ_t^pre|}
```

Bounds computed via linear programming:
```
[τ_L, τ_U] = [min_δ∈Δ τ(δ), max_δ∈Δ τ(δ)]
```

Confidence intervals:
- FLCI (Fixed-Length Confidence Interval) for smoothness
- C-LF (Conditional Least-Favorable) for relative magnitudes

*Standard errors:*
- Inherits from underlying event-study estimation
- Sensitivity analysis reports bounds, not point estimates

*Edge cases:*
- Breakdown point: smallest M where CI includes zero
- M=0: reduces to standard parallel trends
- Negative M: not valid (constraints become infeasible)

**Reference implementation(s):**
- R: `HonestDiD` package (Rambachan & Roth's official package)

**Requirements checklist:**
- [ ] M parameter must be ≥ 0
- [ ] Breakdown point (breakdown_M) correctly identified
- [ ] Delta^SD (smoothness) and Delta^RM (relative magnitudes) both supported
- [ ] Sensitivity plot shows bounds vs. M
- [ ] FLCI and C-LF confidence intervals implemented

---

## PreTrendsPower

**Primary source:** [Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.](https://doi.org/10.1257/aeri.20210236)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires specification of variance-covariance matrix of pre-treatment estimates
- Warns if pre-trends test has low power (uninformative)
- Different violation types have different power properties

*Estimator equation (as implemented):*

Pre-trends test statistic (Wald):
```
W = δ̂_pre' V̂_pre^{-1} δ̂_pre ~ χ²(k)
```

Power function:
```
Power(δ_true) = P(W > χ²_{α,k} | δ = δ_true)
```

Minimum detectable violation (MDV):
```
MDV(power=0.8) = min{|δ| : Power(δ) ≥ 0.8}
```

Violation types:
- **Linear**: δ_t = c × t (linear pre-trend)
- **Constant**: δ_t = c (level shift)
- **Last period**: δ_{-1} = c, others zero
- **Custom**: user-specified pattern

*Standard errors:*
- Power calculations are exact (no sampling variability)
- Uncertainty comes from estimated Σ

*Edge cases:*
- Perfect collinearity in pre-periods: test not well-defined
- Single pre-period: power calculation trivial
- Very high power: MDV approaches zero

**Reference implementation(s):**
- R: `pretrends` package (Roth's official package)

**Requirements checklist:**
- [ ] MDV = minimum detectable violation at target power level
- [ ] Violation types: linear, constant, last_period, custom all implemented
- [ ] Power curve plotting over violation magnitudes
- [ ] Integrates with HonestDiD for combined sensitivity analysis

---

## PowerAnalysis

**Primary source:**
- Bloom, H.S. (1995). Minimum Detectable Effects: A Simple Way to Report the Statistical Power of Experimental Designs. *Evaluation Review*, 19(5), 547-556. https://doi.org/10.1177/0193841X9501900504
- Burlig, F., Preonas, L., & Woerman, M. (2020). Panel Data and Experimental Design. *Journal of Development Economics*, 144, 102458.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires specification of outcome variance and intraclass correlation
- Warns if power is very low (<0.5) or sample size insufficient
- Cluster randomization requires cluster-level parameters

*Estimator equation (as implemented):*

Minimum detectable effect (MDE):
```
MDE = (t_{α/2} + t_{1-κ}) × SE(τ̂)
```
where κ is target power (typically 0.8).

Standard error for DiD:
```
SE(τ̂) = σ × √(1/n_T + 1/n_C) × √(1 + ρ(m-1)) / √(1 - R²)
```
where:
- ρ = intraclass correlation
- m = cluster size
- R² = variance explained by covariates

Power function:
```
Power = Φ(|τ|/SE - z_{α/2})
```

Sample size for target power:
```
n = 2(t_{α/2} + t_{1-κ})² σ² / MDE²
```

*Standard errors:*
- Analytical formulas (no estimation uncertainty in power calculation)
- Simulation-based power accounts for finite-sample and model-specific factors

*Edge cases:*
- Very small effects: may require infeasibly large samples
- High ICC: dramatically reduces effective sample size
- Unequal allocation: optimal is often 50-50 but depends on costs

**Reference implementation(s):**
- R: `pwr` package (general), `DeclareDesign` (simulation-based)
- Stata: `power` command

**Requirements checklist:**
- [ ] MDE calculation given sample size and variance parameters
- [ ] Power calculation given effect size and sample size
- [ ] Sample size calculation given MDE and target power
- [ ] Simulation-based power for complex designs
- [ ] Cluster adjustment for clustered designs

---

# Cross-Reference: Standard Errors Summary

| Estimator | Default SE | Alternatives |
|-----------|-----------|--------------|
| DifferenceInDifferences | HC1 robust | Cluster-robust, wild bootstrap |
| MultiPeriodDiD | Cluster at unit | Wild bootstrap |
| TwoWayFixedEffects | Cluster at unit | Wild bootstrap |
| CallawaySantAnna | Analytical (influence fn) | Multiplier bootstrap |
| SunAbraham | Cluster-robust + delta method | Pairs bootstrap |
| SyntheticDiD | Placebo variance (Alg 4) | Block bootstrap |
| TripleDifference | HC1 / cluster-robust | Influence function for IPW/DR |
| TROP | Block bootstrap | Jackknife |
| BaconDecomposition | N/A (exact decomposition) | Individual 2×2 SEs |
| HonestDiD | Inherited from event study | FLCI, C-LF |
| PreTrendsPower | Exact (analytical) | - |
| PowerAnalysis | Exact (analytical) | Simulation-based |

---

# Cross-Reference: R Package Equivalents

| diff-diff Estimator | R Package | Function |
|---------------------|-----------|----------|
| DifferenceInDifferences | fixest | `feols(y ~ treat:post, ...)` |
| MultiPeriodDiD | fixest | `feols(y ~ i(event_time), ...)` |
| TwoWayFixedEffects | fixest | `feols(y ~ treat \| unit + time, ...)` |
| CallawaySantAnna | did | `att_gt()` |
| SunAbraham | fixest | `sunab()` |
| SyntheticDiD | synthdid | `synthdid_estimate()` |
| TripleDifference | - | (forthcoming) |
| TROP | - | (forthcoming) |
| BaconDecomposition | bacondecomp | `bacon()` |
| HonestDiD | HonestDiD | `createSensitivityResults()` |
| PreTrendsPower | pretrends | `pretrends()` |
| PowerAnalysis | pwr / DeclareDesign | `pwr.t.test()` / simulation |

---

# Version History

- **v1.0** (2025-01-19): Initial registry with 12 estimators
