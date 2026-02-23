# Methodology Registry

This document provides the academic foundations and key implementation requirements for each estimator in diff-diff. It serves as a reference for contributors and users who want to understand the theoretical basis of the methods.

## Table of Contents

1. [Core DiD Estimators](#core-did-estimators)
   - [DifferenceInDifferences](#differenceinifferences)
   - [MultiPeriodDiD](#multiperioddid)
   - [TwoWayFixedEffects](#twowayfixedeffects)
2. [Modern Staggered Estimators](#modern-staggered-estimators)
   - [CallawaySantAnna](#callawaysantanna)
   - [ContinuousDiD](#continuousdid)
   - [SunAbraham](#sunabraham)
   - [ImputationDiD](#imputationdid)
   - [TwoStageDiD](#twostagedid)
   - [StackedDiD](#stackeddid)
3. [Advanced Estimators](#advanced-estimators)
   - [SyntheticDiD](#syntheticdid)
   - [TripleDifference](#tripledifference)
   - [TROP](#trop)
4. [Diagnostics & Sensitivity](#diagnostics--sensitivity)
   - [PlaceboTests](#placebotests)
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
- Empty cells (e.g., no treated-pre observations) cause rank deficiency, handled per `rank_deficient_action` setting
  - With "warn" (default): emits warning, sets NaN for affected coefficients
  - With "error": raises ValueError
  - With "silent": continues silently with NaN coefficients
- Singleton clusters (one observation): included in variance estimation; contribute to meat matrix via u_i² X_i X_i' (same formula as larger clusters with n_g=1)
- Rank-deficient design matrix (collinearity): warns and sets NA for dropped coefficients (R-style, matches `lm()`)
  - Tolerance: `1e-07` (matches R's `qr()` default), relative to largest diagonal element of R in QR decomposition
  - Controllable via `rank_deficient_action` parameter: "warn" (default), "error", or "silent"

**Reference implementation(s):**
- R: `fixest::feols()` with interaction term
- Stata: `reghdfe` or manual regression with interaction

**Requirements checklist:**
- [x] Treatment and time indicators are binary 0/1 with variation
- [x] ATT equals coefficient on interaction term
- [x] Wild bootstrap supports Rademacher, Mammen, Webb weight distributions
- [x] Formula interface parses `y ~ treated * post` correctly

---

## MultiPeriodDiD

**Primary source:** Event study methodology
- Freyaldenhoven, S., Hansen, C., Pérez, J.P., & Shapiro, J.M. (2021). Visualization,
  identification, and estimation in the linear panel event-study design. NBER Working Paper 29170.
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed.
  MIT Press, Ch. 10, 13.
- Angrist, J.D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

**Scope:** Simultaneous adoption event study. All treated units receive treatment at the
same time. For staggered adoption (different units treated at different times), use
CallawaySantAnna or SunAbraham instead.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment indicator must be binary (0/1) with variation in both groups
- Requires at least 1 pre-treatment and 1 post-treatment period
- Warns when only 1 pre-period available (≥2 needed to test parallel trends;
  ATT is still valid but pre-trends assessment is not possible)
- Reference period defaults to last pre-treatment period (e=-1 convention)
- Treatment indicator should be time-invariant ever-treated (D_i);
  warns when time-varying D_it detected (requires `unit` parameter)
- Warns if treatment timing varies across units when `unit` is provided
  (suggests CallawaySantAnna or SunAbraham instead)
- Treatment must be an absorbing state (once treated, always treated)

*Estimator equation (target specification):*

With unit and time fixed effects absorbed:

```
Y_it = α_i + γ_t + Σ_{e≠-1} δ_e × D_i × 1(t = E + e) + X'β + ε_it
```

where:
- α_i = unit fixed effects (absorbed)
- γ_t = time fixed effects (absorbed)
- E = common treatment time (same for all treated units)
- D_i = treatment group indicator (1=treated, 0=control)
- e = t - E = event time (relative periods to treatment)
- δ_e = treatment effect at event time e
- δ_{-1} = 0 (reference period, omitted for identification)

For simultaneous treatment, this is equivalent to interacting treatment with
calendar-time indicators:

```
Y_it = α_i + γ_t + Σ_{t≠t_ref} δ_t × (D_i × Period_t) + X'β + ε_it
```

where interactions are included for ALL periods (pre and post), not just post-treatment.

Pre-treatment coefficients (e < -1) test the parallel trends assumption:
under H0 of parallel trends, δ_e = 0 for all e < 0.

Post-treatment coefficients (e ≥ 0) estimate dynamic treatment effects.

Average ATT over post-treatment periods:

```
ATT_avg = (1/|post|) × Σ_{e≥0} δ_e
```

with SE computed from the sub-VCV matrix:

```
Var(ATT_avg) = 1'V1 / |post|²
```

where V is the VCV sub-matrix for post-treatment δ_e coefficients.

*Standard errors:*
- Default: HC1 heteroskedasticity-robust (same as DifferenceInDifferences base class)
- Alternative: Cluster-robust at unit level via `cluster` parameter (recommended for panel data)
- Optional: Wild cluster bootstrap (complex for multi-coefficient testing;
  requires joint bootstrap distribution)
- Degrees of freedom adjusted for absorbed fixed effects

*Edge cases:*
- Reference period: omitted from design matrix; coefficient is zero by construction.
  Default is last pre-treatment period (e=-1). User can override via `reference_period`.
- Post-period reference: raises ValueError. Post-period references would exclude a
  post-treatment period from estimation, biasing avg_att and breaking downstream inference.
- Reference period default change: FutureWarning emitted when `reference_period` is not
  explicitly specified and ≥2 pre-periods exist, noting the default changed from first
  to last pre-period (e=-1 convention, matching fixest/did).
- Never-treated units: all event-time indicators are zero; they identify the time
  fixed effects and serve as comparison group.
- Endpoint binning: distant event times (e.g., e < -K or e > K) should be binned
  into endpoint indicators to avoid sparse cells. This prevents imprecise estimates
  at extreme leads/lags.
- Unbalanced panels: only uses observations where event-time is defined. Units
  not observed at all event times contribute to the periods they are present for.
- Rank-deficient design matrix (collinearity): warns and sets NA for dropped
  coefficients (R-style, matches `lm()`)
- Average ATT (`avg_att`) is NA if any post-period effect is unidentified
  (R-style NA propagation)
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - p_value and CI: Also NaN when t_stat is NaN
  - avg_se: Checked for finiteness before computing avg_t_stat
  - **Note**: Defensive enhancement matching CallawaySantAnna NaN convention
- Treatment reversal: warns if any unit transitions from treated to untreated
  (non-absorbing treatment violates the simultaneous adoption assumption)
- Time-varying treatment (D_it): warns when `unit` parameter is provided and
  within-unit treatment variation is detected. Advises creating an ever-treated
  indicator. Without ever-treated D_i, pre-period interaction coefficients are
  unidentified.
- Pre-test of parallel trends: joint F-test on pre-treatment δ_e coefficients.
  Low power in pre-test does not validate parallel trends (Roth 2022).

**Reference implementation(s):**
- R: `fixest::feols(y ~ i(time, treatment, ref=ref_period) | unit + time, data, cluster=~unit)`
  or equivalently `feols(y ~ i(event_time, ref=-1) | unit + time, data, cluster=~unit)`
- Stata: `reghdfe y ib(-1).event_time#1.treatment, absorb(unit time) cluster(unit)`

**Requirements checklist:**

- [x] Event-time indicators for ALL periods (pre and post), not just post-treatment
- [x] Reference period coefficient is zero (normalized by omission from design matrix)
- [x] Pre-period coefficients available for parallel trends assessment
- [ ] Default cluster-robust SE at unit level (currently HC1; cluster-robust via `cluster` param)
- [ ] Supports unit and time FE via absorption
- [ ] Endpoint binning for distant event times
- [x] Average ATT correctly accounts for covariance between period effects
- [x] Returns PeriodEffect objects with confidence intervals
- [x] Supports both balanced and unbalanced panels

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

**Note:** The interaction term `D_i × Post_t` is within-transformed (demeaned) alongside the
outcome and covariates before regression. This is required by the Frisch-Waugh-Lovell theorem:
all regressors must be projected out of the same fixed effects space as the dependent variable.
This matches the behavior of R's `fixest::feols()` with absorbed FE.

*Standard errors:*
- Default: Cluster-robust at unit level (accounts for serial correlation)
- Degrees of freedom adjusted for absorbed fixed effects: `df_adjustment = n_units + n_times - 2`

*Edge cases:*
- Singleton units/periods are automatically dropped
- Treatment perfectly collinear with FE raises error with informative message listing dropped columns
- Covariate collinearity emits warning but estimation continues (ATT still identified)
- Rank-deficient design matrix: warns and sets NA for dropped coefficients (R-style, matches `lm()`)
- Unbalanced panels handled via proper demeaning
- Multi-period `time` parameter: only binary (0/1) post indicator is recommended; multi-period values
  produce `treated × period_number` rather than `treated × post_indicator`. A `UserWarning` is
  emitted when `time` has >2 unique values, advising users to create a binary post column.
  Non-{0,1} binary time (e.g., {2020, 2021}) also emits a warning, though the ATT is mathematically
  correct — the within-transformation absorbs the scaling.
- Staggered warning limitation: requires `time` to have actual period values (not binary 0/1)
  so that different cohort first-treatment times can be distinguished. With binary `time="post"`,
  all treated units appear to start at `time=1`, making staggering undetectable. Users with
  staggered designs should use `decompose()` or `CallawaySantAnna` directly.

**Reference implementation(s):**
- R: `fixest::feols(y ~ treat:post | unit + post, data, cluster = ~unit)`
- Stata: `reghdfe y treat, absorb(unit time) cluster(unit)`

**Requirements checklist:**
- [ ] Staggered adoption detection warning (only fires when `time` has >2 unique values; with binary `time`, staggering is undetectable)
- [x] Multi-period time warning (fires when `time` has >2 unique values)
- [x] Auto-clusters standard errors at unit level
- [x] `decompose()` method returns BaconDecompositionResults
- [x] Within-transformation correctly handles unbalanced panels
- [x] Non-{0,1} binary time warning (fires when time has 2 unique values not in {0,1})
- [x] ATT invariance to time encoding (verified by test)

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

*Note:* This equation uses g-1 as the base period, which applies to post-treatment effects (t ≥ g) and `base_period="universal"`. With `base_period="varying"` (default), pre-treatment effects use t-1 as base for consecutive comparisons (see Base period selection in Edge cases).

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

*Bootstrap weight distributions:*

The multiplier bootstrap uses random weights w_i with E[w]=0 and Var(w)=1:

| Weight Type | Values | Probabilities | Properties |
|-------------|--------|---------------|------------|
| Rademacher | ±1 | 1/2 each | Simplest; E[w³]=0 |
| Mammen | -(√5-1)/2, (√5+1)/2 | (√5+1)/(2√5), (√5-1)/(2√5) | E[w³]=1; better for skewed data |
| Webb | ±√(3/2), ±1, ±√(1/2) | 1/6 each | 6-point; recommended for few clusters |

**Webb distribution details:**
- Values: {-√(3/2), -1, -√(1/2), √(1/2), 1, √(3/2)} ≈ {-1.225, -1, -0.707, 0.707, 1, 1.225}
- Equal probabilities (1/6 each) giving E[w]=0, Var(w)=1
- Matches R's `did` package implementation
- **Verification**: Implementation matches `fwildclusterboot` R package
  ([C++ source](https://github.com/s3alfisc/fwildclusterboot/blob/master/src/wildboottest.cpp))
  which uses identical `sqrt(1.5)`, `1`, `sqrt(0.5)` values with equal 1/6 probabilities.
  Some documentation shows simplified values (±1.5, ±1, ±0.5) but actual implementations
  use square root values to achieve unit variance.
- Reference: Webb, M.D. (2023). Reworking Wild Bootstrap Based Inference for Clustered Errors.
  Queen's Economics Department Working Paper No. 1315. (Updated from Webb 2014)

*Edge cases:*
- Groups with single observation: included but may have high variance
- Missing group-time cells: ATT(g,t) set to NaN
- Anticipation: `anticipation` parameter shifts reference period
  - Group aggregation includes periods t >= g - anticipation (not just t >= g)
  - Both analytical SE and bootstrap SE aggregation respect anticipation
  - Not-yet-treated + anticipation: control mask uses `G > t + anticipation`
    (not just `G > t`) to exclude cohorts in the anticipation window
- Rank-deficient design matrix (covariate collinearity):
  - Detection: Pivoted QR decomposition with tolerance `1e-07` (R's `qr()` default)
  - Handling: Warns and drops linearly dependent columns, sets NA for dropped coefficients (R-style, matches `lm()`)
  - Parameter: `rank_deficient_action` controls behavior: "warn" (default), "error", or "silent"
- Non-finite inference values:
  - Analytic SE: Returns NaN to signal invalid inference (not biased via zeroing)
  - Bootstrap: Drops non-finite samples, warns, and adjusts p-value floor accordingly. SE, CI, and p-value are all NaN if the original point estimate is non-finite, SE is non-finite or zero (e.g., n_valid=1 with ddof=1, or identical samples)
  - Threshold: Returns NaN if <50% of bootstrap samples are valid
  - Per-effect t_stat: Uses NaN (not 0.0) when SE is non-finite or zero (consistent with overall_t_stat)
  - **Note**: This is a defensive enhancement over reference implementations (R's `did::att_gt`, Stata's `csdid`) which may error or produce unhandled inf/nan in edge cases without informative warnings
- No post-treatment effects (all treatment occurs after data ends):
  - Overall ATT set to NaN (no post-treatment periods to aggregate)
  - All overall inference fields (SE, t-stat, p-value, CI) also set to NaN
  - Warning emitted: "No post-treatment effects for aggregation"
  - Individual pre-treatment ATT(g,t) are computed (for parallel trends assessment)
  - Bootstrap runs for per-effect SEs even without post-treatment; only overall statistics are NaN
  - **Principle**: NaN propagates consistently through overall inference fields; pre-treatment effects get full bootstrap inference
- Aggregated t_stat (event-study, group-level):
  - Uses NaN when SE is non-finite or zero (matches per-effect and overall t_stat behavior)
  - Previous behavior (0.0 default) was inconsistent and misleading
- Base period selection (`base_period` parameter):
  - "varying" (default): Pre-treatment uses t-1 as base (consecutive comparisons)
  - "universal": All comparisons use g-anticipation-1 as base
  - Both produce identical post-treatment ATT(g,t); differ only pre-treatment
  - Matches R `did::att_gt()` base_period parameter
  - **Event study output**: With "universal", includes reference period (e=-1-anticipation)
    with effect=0, se=NaN, conf_int=(NaN, NaN). Inference fields are NaN since this is
    a normalization constraint, not an estimated effect. Only added when real effects exist.
- Base period interaction with Sun-Abraham comparison:
  - CS with `base_period="varying"` produces different pre-treatment estimates than SA
  - This is expected: CS uses consecutive comparisons, SA uses fixed reference (e=-1-anticipation)
  - Use `base_period="universal"` for methodologically comparable pre-treatment effects
  - Post-treatment effects match regardless of base_period setting
- Control group with `control_group="not_yet_treated"`:
  - Always excludes cohort g from controls when computing ATT(g,t)
  - This applies to both pre-treatment (t < g) and post-treatment (t >= g) periods
  - For pre-treatment periods: even though cohort g hasn't been treated yet at time t, they are the treated group for this ATT(g,t) and cannot serve as their own controls
  - Control mask: `never_treated OR (first_treat > t + anticipation AND first_treat != g)`

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

## ContinuousDiD

**Primary Source:** Callaway, Goodman-Bacon & Sant'Anna (2024), "Difference-in-Differences with a Continuous Treatment," NBER Working Paper 32117.

**R Reference:** `contdid` v0.1.0 (CRAN).

### Identification

Two levels of parallel trends (following CGBS 2024, Assumptions 1-2):

**Parallel Trends (PT):** for all doses d in D_+,
`E[Y_t(0) - Y_{t-1}(0) | D = d] = E[Y_t(0) - Y_{t-1}(0) | D = 0]`.
Untreated potential outcome paths are the same across all dose groups and the
untreated group. Stronger than binary PT because it conditions on specific dose values.
Identifies: `ATT(d|d)`, `ATT^{loc}`. Does NOT identify `ATT(d)`, `ACRT`, or cross-dose comparisons.

**Strong Parallel Trends (SPT):** additionally, for all d in D,
`E[Y_t(d) - Y_{t-1}(0) | D > 0] = E[Y_t(d) - Y_{t-1}(0) | D = d]`.
No selection into dose groups on the basis of treatment effects.
Implies `ATT(d|d) = ATT(d)` for all d.
Additionally identifies: `ATT(d)`, `ACRT(d)`, `ACRT^{glob}`, and cross-dose comparisons.

See `docs/methodology/continuous-did.md` Section 4 for full details.

### Key Equations

**Target parameters:**
- `ATT(d|d) = E[Y_t(d) - Y_t(0) | D = d]` — effect of dose d on units who received dose d (PT)
- `ATT(d) = E[Y_t(d) - Y_t(0) | D > 0]` — dose-response curve (SPT required)
- `ACRT(d) = dATT(d)/dd` — average causal response / marginal effect (SPT required)
- `ATT^{loc} = E[ATT(D|D) | D > 0] = E[Delta Y | D > 0] - E[Delta Y | D = 0]` — binarized ATT (PT); equals `ATT^{glob}` under SPT
- `ATT^{glob} = E[ATT(D) | D > 0]` — global average dose-response level (SPT required)
- `ACRT^{glob} = E[ACRT(D_i) | D > 0]` — plug-in average marginal effect (SPT required)

**Estimation via B-spline OLS:**
1. Compute `Delta_tilde_Y = (Y_t - Y_{t-1})_treated - mean((Y_t - Y_{t-1})_control)`
2. Build B-spline basis `Psi(D_i)` from treated doses
3. OLS: `beta = (Psi'Psi)^{-1} Psi' Delta_tilde_Y`
4. `ATT(d) = Psi(d)' beta`, `ACRT(d) = dPsi(d)/dd' beta`

### Edge Cases

- **No untreated group**: Remark 3.1 (lowest-dose-as-control) not implemented; requires P(D=0) > 0.
- **Discrete treatment**: Detect integer-valued dose and warn; saturated regression deferred.
- **All-same dose**: B-spline basis collapses; ACRT(d) = 0 everywhere.
- **Rank deficiency**: When n_treated <= n_basis, cell is skipped.
- **Balanced panel required**: Matches R `contdid` v0.1.0.
- **Anticipation + not-yet-treated**: Control mask uses `G > t + anticipation`
  (not just `G > t`) to exclude cohorts in the anticipation window from
  not-yet-treated controls. When `anticipation=0` (default), behavior is
  unchanged.
- **Boundary knots**: Knots are built once from all treated doses (global, not per-cell) to ensure a common basis across (g,t) cells for aggregation. Evaluation grid is clamped to training-dose boundary knots (`range(dose)`). R's `contdid` v0.1.0 has an inconsistency where `splines2::bSpline(dvals)` uses `range(dvals)` instead of `range(dose)`, which can produce extrapolation artifacts at dose grid extremes. Our approach avoids extrapolation and is methodologically sound.

### Implementation Checklist

- [x] B-spline basis construction matching R's `splines2::bSpline` (global knots from all treated doses; boundary knots use training-dose range; see deviation note above)
- [x] Multi-period (g,t) cell iteration with base period selection
- [x] Dose-response and event-study aggregation with group-proportional weights (n_treated/n_total per group, divided among post-treatment cells; R `ptetools` convention)
- [x] Multiplier bootstrap for inference
- [x] Analytical SEs via influence functions
- [x] Equation verification tests (linear, quadratic, multi-period)
- [ ] Covariate support (deferred, matching R v0.1.0)
- [ ] Discrete treatment saturated regression
- [ ] Lowest-dose-as-control (Remark 3.1)

---

## SunAbraham

**Primary source:** [Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.](https://doi.org/10.1016/j.jeconom.2020.09.006)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires never-treated units as control group
- Warns if treatment effects may be heterogeneous across cohorts (which the method handles)
- Reference period: e=-1-anticipation (defaults to e=-1 when anticipation=0)

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
- Event-time range: no artificial cap (estimates all available relative times, matching R's `fixest::sunab()`)
- No post-treatment effects: returns `(NaN, NaN)` for overall ATT/SE; all inference fields (t_stat, p_value, conf_int) propagate NaN via `np.isfinite()` guards
- `min_pre_periods`/`min_post_periods` parameters: removed (previously deprecated with `FutureWarning`; callers passing these will now get `TypeError`)
- Variance fallback: when full weight vector cannot be constructed for overall ATT SE, uses simplified variance (ignores covariances between periods) with `UserWarning`
- Rank-deficient design matrix (covariate collinearity):
  - Detection: Pivoted QR decomposition with tolerance `1e-07` (R's `qr()` default)
  - Handling: Warns and drops linearly dependent columns, sets NA for dropped coefficients (R-style, matches `lm()`)
  - Parameter: `rank_deficient_action` controls behavior: "warn" (default), "error", or "silent"
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - Analytical inference: p_value and CI also NaN when t_stat is NaN (NaN propagates through `compute_p_value` and `compute_confidence_interval`)
  - Bootstrap inference: p_value and CI computed from bootstrap distribution. SE, CI, and p-value are all NaN if the original point estimate is non-finite, SE is non-finite or zero, or if <50% of bootstrap samples are valid
  - Applies to overall ATT, per-effect event study, and aggregated event study
  - **Note**: Defensive enhancement matching CallawaySantAnna behavior; R's `fixest::sunab()` may produce Inf/NaN without warning
- Inference distribution:
  - Cohort-level p-values: t-distribution (via `LinearRegression.get_inference()`)
  - Aggregated event study and overall ATT p-values: normal distribution (via `compute_p_value()`)
  - This is asymptotically equivalent and standard for delta-method-aggregated quantities
  - **Deviation from R**: R's fixest uses t-distribution at all levels; aggregated p-values may differ slightly for small samples

**Reference implementation(s):**
- R: `fixest::sunab()` (Laurent Bergé's implementation)
- Stata: `eventstudyinteract`

**Requirements checklist:**
- [x] Never-treated units required as controls
- [x] Interaction weights sum to 1 within each relative time period
- [x] Reference period defaults to e=-1, coefficient normalized to zero
- [x] Cohort-specific effects recoverable from results
- [x] Cluster-robust SEs with delta method for aggregates
- [x] R comparison: ATT matches within machine precision (<1e-11)
- [x] R comparison: SE matches within 0.3% (well within 1% threshold)
- [x] R comparison: Event study effects match perfectly (correlation 1.0)

---

## ImputationDiD

**Primary source:** [Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event-Study Designs: Robust and Efficient Estimation. *Review of Economic Studies*, 91(6), 3253-3285.](https://doi.org/10.1093/restud/rdae007)

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Parallel trends (Assumption 1):** `E[Y_it(0)] = alpha_i + beta_t` for all observations. General form allows `E[Y_it(0)] = alpha_i + beta_t + X'_it * delta` with time-varying covariates.
- **No-anticipation effects (Assumption 2):** `Y_it = Y_it(0)` for all untreated observations. Adjustable via `anticipation` parameter.
- Treatment must be absorbing: `D_it` switches from 0 to 1 and stays at 1.
- Covariate space of treated observations must be spanned by untreated observations (rank condition). For unit/period FE case: every treated unit must have ≥1 untreated period; every post-treatment period must have ≥1 untreated unit.
- Without never-treated units, long-run effects at horizon `K_it >= H_bar` (where `H_bar = max(first_treat) - min(first_treat)`) are not identified (Proposition 5). Set to NaN with warning.

*Estimator equation (Theorem 2, as implemented):*

```
Step 1. Estimate counterfactual model on untreated observations only (it in Omega_0):
    Y_it = alpha_i + beta_t [+ X'_it * delta] + epsilon_it

Step 2. For each treated observation (it in Omega_1), impute:
    Y_hat_it(0) = alpha_hat_i + beta_hat_t [+ X'_it * delta_hat]
    tau_hat_it  = Y_it - Y_hat_it(0)

Step 3. Aggregate:
    tau_hat_w = sum_{it in Omega_1} w_it * tau_hat_it
```

where:
- `Omega_0 = {it : D_it = 0}` — all untreated observations (never-treated + not-yet-treated)
- `Omega_1 = {it : D_it = 1}` — all treated observations
- `w_it` = pre-specified weights (overall ATT: `w_it = 1/N_1`)

*Common estimation targets (weighting schemes):*
- Overall ATT: `w_it = 1/N_1` for all `it in Omega_1`
- Horizon-specific: `w_it = 1[K_it = h] / |Omega_{1,h}|` for `K_it = t - E_i`
- Group-specific: `w_it = 1[G_i = g] / |Omega_{1,g}|`

*Standard errors (Theorem 3, Equation 7):*

Conservative clustered variance estimator:
```
sigma_hat^2_w = sum_i ( sum_{t: it in Omega} v_it * epsilon_tilde_it )^2
```

Observation weights `v_it`:
- For treated `(i,t) in Omega_1`: `v_it = w_it` (the aggregation weight)
- For untreated `(i,t) in Omega_0` (FE-only case): `v_it = -(w_i./n_{0,i} + w_.t/n_{0,t} - w../N_0)`
  where `w_i. = sum of w over treated obs of unit i`, `n_{0,i} = untreated periods for unit i`, etc.
- For untreated with covariates: `v_untreated = -A_0 (A_0' A_0)^{-1} A_1' w_treated`
  where `A_0`, `A_1` are design matrices for untreated/treated observations.

**Note on v_it derivation:** The paper's Supplementary Proposition A3 provides the explicit formula for `v_it^*`, but was not in the extraction range for the paper review. The FE-only closed form above is reconstructed from Theorem 3's general form — it follows from the chain rule of the imputation estimator's dependence on the Step 1 OLS estimates. The covariate case uses the OLS projection matrix directly.

Auxiliary model residuals (Equation 8):
- Partition `Omega_1` into groups `G_g` (default: cohort × horizon)
- Compute `tau_tilde_g` for each group (weighted average within group)
- `epsilon_tilde_it = Y_it - alpha_hat_i - beta_hat_t [- X'delta_hat] - tau_tilde_g` (treated)
- `epsilon_tilde_it = Y_it - alpha_hat_i - beta_hat_t [- X'delta_hat]` (untreated, i.e., Step 1 residuals)

The `aux_partition` parameter controls the partition: `"cohort_horizon"` (default, tightest SEs), `"cohort"` (coarser, more conservative), `"horizon"` (groups by relative time only).

*Pre-trend test (Test 1, Equation 9):*
```
Y_it = alpha_i + beta_t [+ X'_it * delta] + W'_it * gamma + epsilon_it
```
- Estimate on untreated observations only
- Test `gamma = 0` via cluster-robust Wald F-test
- Independent of treatment effect estimation (Proposition 9)

*Edge cases:*
- **Unbalanced panels:** FE estimated via iterative alternating projection (Gauss-Seidel), equivalent to OLS with unit+time dummies. Converges in O(max_iter) passes; typically 5-20 iterations for unbalanced panels, 1-2 for balanced. One-pass demeaning is only exact for balanced panels.
- **No never-treated units (Proposition 5):** Long-run effects at horizons `h >= H_bar` are not identified. Set to NaN with warning listing affected horizons.
- **Rank condition failure:** Every treated unit must have ≥1 untreated period; every post-treatment period must have ≥1 untreated unit. Behavior controlled by `rank_deficient_action`: "warn" (default), "error", or "silent". Missing FE produce NaN treatment effects for affected observations.
- **Always-treated units:** Units with `first_treat` at or before the earliest time period have no untreated observations. Warning emitted; these units are excluded from Step 1 OLS but their treated observations contribute to aggregation if imputation is possible.
- **NaN propagation:** If all `tau_hat` values for a given horizon or group are NaN, the aggregated effect and all inference fields (SE, t-stat, p-value, CI) are set to NaN. NaN in v*eps product (from missing FE) is zeroed for variance computation (matching R's did_imputation which drops unimputable obs).
- **NaN inference for undefined statistics:** t_stat uses NaN when SE is non-finite or zero; p_value and CI also NaN. Matches CallawaySantAnna NaN convention.
- **Pre-trend test:** Uses iterative demeaning (same as Step 1 FE) for exact within-transformation on unbalanced panels. One-pass demeaning is only exact for balanced panels.
- **Overall ATT variance:** Weights zero out non-finite tau_hat and renormalize, matching the ATT estimand (which averages only finite tau_hat). `_compute_conservative_variance` returns 0.0 for all-zeros weights, so the n_valid==0 guard is necessary to return NaN SE.
- **`balance_e` cohort filtering:** When `balance_e` is set, cohort balance is checked against the *full panel* (pre + post treatment) via `_build_cohort_rel_times()`, requiring observations at every relative time in `[-balance_e, max_h]`. Both analytical aggregation and bootstrap inference use the same `_compute_balanced_cohort_mask` with pre-computed cohort horizons.
- **Bootstrap clustering:** Multiplier bootstrap generates weights at `cluster_var` granularity (defaults to `unit` if `cluster` not specified). Invalid cluster column raises ValueError.
- **Non-constant `first_treat` within a unit:** Emits `UserWarning` identifying the count and example unit. The estimator proceeds using the first observed value per unit (via `.first()` aggregation), but results may be unreliable.
- **treatment_effects DataFrame weights:** `weight` column uses `1/n_valid` for finite tau_hat and 0 for NaN tau_hat, consistent with the ATT estimand.
- **Rank-deficient covariates in variance:** Covariates with NaN coefficients (dropped for rank deficiency in Step 1) are excluded from the variance design matrices `A_0`/`A_1`. Only covariates with finite coefficients participate in the `v_it` projection.
- **Sparse variance solver:** `_compute_v_untreated_with_covariates` uses `scipy.sparse.linalg.spsolve` to solve `(A_0'A_0) z = A_1'w` without densifying the normal equations matrix. Falls back to dense `lstsq` if the sparse solver fails.
- **Bootstrap inference:** Uses multiplier bootstrap on the Theorem 3 influence function: `psi_i = sum_t v_it * epsilon_tilde_it`. Cluster-level psi sums are pre-computed for each aggregation target (overall, per-horizon, per-group), then perturbed with multiplier weights (Rademacher by default; configurable via `bootstrap_weights` parameter to use Mammen or Webb weights, matching CallawaySantAnna). This is a library extension (not in the paper) consistent with CallawaySantAnna/SunAbraham bootstrap patterns.
- **Auxiliary residuals (Equation 8):** Uses v_it-weighted tau_tilde_g formula: `tau_tilde_g = sum(v_it * tau_hat_it) / sum(v_it)` within each partition group. Zero-weight groups (common in event-study SE computation) fall back to unweighted mean.

**Reference implementation(s):**
- Stata: `did_imputation` (Borusyak, Jaravel, Spiess; available from SSC)
- R: `didimputation` package (Kyle Butts)

**Requirements checklist:**
- [x] Step 1: OLS on untreated observations only (never-treated + not-yet-treated)
- [x] Step 2: Impute counterfactual `Y_hat_it(0)` for treated observations
- [x] Step 3: Aggregate with researcher-chosen weights `w_it`
- [x] Conservative clustered variance estimator (Theorem 3, Equation 7)
- [x] Auxiliary model for treated residuals (Equation 8) with configurable partition (`aux_partition`)
- [x] Supports unit FE, period FE, and time-varying covariates
- [x] Refuses to estimate unidentified estimands (Proposition 5) — sets NaN with warning
- [x] Pre-trend test uses only untreated observations (Test 1, Equation 9)
- [x] Supports balanced and unbalanced panels (iterative Gauss-Seidel demeaning for exact FE)
- [x] Event study and group aggregation

---

## TwoStageDiD

**Primary source:** [Gardner, J. (2022). Two-stage differences in differences. arXiv:2207.05943.](https://arxiv.org/abs/2207.05943)

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Parallel trends (same as ImputationDiD):** `E[Y_it(0)] = alpha_i + beta_t` for all observations.
- **No-anticipation effects:** `Y_it = Y_it(0)` for all untreated observations.
- Treatment must be absorbing: `D_it` switches from 0 to 1 and stays at 1.
- Always-treated units (treated in all periods) are excluded with a warning, since they have no untreated observations for Stage 1 FE estimation.

*Estimator equation (two-stage procedure, as implemented):*

```
Stage 1. Estimate unit + time fixed effects on untreated observations only (it in Omega_0):
    Y_it = alpha_i + beta_t + epsilon_it
    Compute residuals: y_tilde_it = Y_it - alpha_hat_i - beta_hat_t  (for ALL observations)

Stage 2. Regress residualized outcomes on treatment indicators (on treated observations):
    y_tilde_it = tau * D_it + eta_it
    (or event-study specification with horizon indicators)
```

Point estimates are identical to ImputationDiD (Borusyak et al. 2024). The two-stage procedure is algebraically equivalent to the imputation approach: both estimate unit+time FE on untreated observations and recover treatment effects from the difference between observed and counterfactual outcomes.

*Variance: GMM sandwich (Newey & McFadden 1994 Theorem 6.1):*

The variance accounts for first-stage estimation error propagating into Stage 2, following the GMM framework:

```
V(tau_hat) = (D'D)^{-1} * Bread * (D'D)^{-1}

Bread = sum_c ( sum_{i in c} psi_i )( sum_{i in c} psi_i )'
```

where `psi_i` is the stacked influence function for unit i across all its observations, combining the Stage 2 score and the Stage 1 correction term.

**Note on Equation 6 discrepancy:** The paper's Equation 6 uses a per-cluster inverse `(D_c'D_c)^{-1}` when forming the influence function contribution. The R `did2s` implementation and our code use the GLOBAL inverse `(D'D)^{-1}` following standard GMM theory (Newey & McFadden 1994). We follow the R implementation, which is consistent with standard GMM sandwich variance estimation.

**No finite-sample adjustments:** The variance estimator uses the raw asymptotic sandwich without degrees-of-freedom corrections (no HC1-style `n/(n-k)` adjustment). This matches the R `did2s` implementation.

*Bootstrap:*

Our implementation uses multiplier bootstrap on the GMM influence function: cluster-level `psi` sums are pre-computed, then perturbed with multiplier weights (Rademacher by default; configurable via `bootstrap_weights` parameter to use Mammen or Webb weights, matching CallawaySantAnna). The R `did2s` package defaults to block bootstrap (resampling clusters with replacement). Both approaches are asymptotically valid; the multiplier bootstrap is computationally cheaper and consistent with the CallawaySantAnna/ImputationDiD bootstrap patterns in this library.

*Edge cases:*
- **Always-treated units:** Units treated in all observed periods have no untreated observations for Stage 1 FE estimation. These are excluded with a warning listing the affected unit IDs. Their treated observations do NOT contribute to Stage 2.
- **Rank condition violations:** If the Stage 1 design matrix (unit+time dummies on untreated obs) is rank-deficient, or if certain unit/time FE are unidentified (e.g., a unit with no untreated periods after excluding always-treated), the affected FE produce NaN. Behavior controlled by `rank_deficient_action`: "warn" (default), "error", or "silent".
- **NaN y_tilde handling:** When Stage 1 FE are unidentified for some observations, the residualized outcome `y_tilde` is NaN. These observations are zeroed out (excluded) from the Stage 2 regression and variance computation, matching the treatment of unimputable observations in ImputationDiD.
- **NaN inference for undefined statistics:** t_stat uses NaN when SE is non-finite or zero; p_value and CI also NaN. Matches CallawaySantAnna/ImputationDiD NaN convention.
- **Event study aggregation:** Horizon-specific effects use the same two-stage procedure with horizon indicator dummies in Stage 2. Unidentified horizons (e.g., long-run effects without never-treated units, per Proposition 5 of Borusyak et al. 2024) produce NaN.
- **balance_e with no qualifying cohorts:** If no cohorts have sufficient pre/post coverage for the requested `balance_e`, a warning is emitted and event study results contain only the reference period.
- **No never-treated units (Proposition 5):** When there are no never-treated units and multiple treatment cohorts, horizons h >= h_bar (where h_bar = max(groups) - min(groups)) are unidentified per Proposition 5 of Borusyak et al. (2024). These produce NaN inference with n_obs > 0 (treated observations exist but counterfactual is unidentified) and a warning listing affected horizons. Matches ImputationDiD behavior. Proposition 5 applies to event study horizons only, not cohort aggregation — a cohort whose treated obs all fall at Prop 5 horizons naturally gets n_obs=0 in group effects because all its y_tilde values are NaN.
- **Zero-observation horizons after filtering:** When `balance_e` or NaN `y_tilde` filtering results in zero observations for some non-Prop-5 event study horizons, those horizons produce NaN for all inference fields (effect, SE, t-stat, p-value, CI) with n_obs=0.
- **Zero-observation cohorts in group effects:** If all treated observations for a cohort have NaN `y_tilde` (excluded from estimation), that cohort's group effect is NaN with n_obs=0.

**Reference implementation(s):**
- R: `did2s::did2s()` (Kyle Butts & John Gardner)

**Requirements checklist:**
- [x] Stage 1: OLS on untreated observations only for unit+time FE
- [x] Stage 2: Regress residualized outcomes on treatment indicators
- [x] Point estimates match ImputationDiD
- [x] GMM sandwich variance (Newey & McFadden 1994 Theorem 6.1)
- [x] Global `(D'D)^{-1}` in variance (matches R `did2s`, not paper Eq. 6)
- [x] No finite-sample adjustment (raw asymptotic sandwich)
- [x] Always-treated units excluded with warning
- [x] Multiplier bootstrap on GMM influence function
- [x] Event study and overall ATT aggregation

---

## StackedDiD

**Primary source:** Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked Difference-in-Differences. NBER Working Paper 32054. http://www.nber.org/papers/w32054

**Key implementation requirements:**

*Assumption checks / warnings:*
- Assumption 1 (No Anticipation): ATT(a, a+e) = 0 for all e < 0
- Assumption 2 (Common Trends): E[Y_{s,a+e}(0) - Y_{s,a-1}(0) | A_s = a] = E[Y_{s,a+e}(0) - Y_{s,a-1}(0) | A_s > a + e]
- Clean controls must exist for each sub-experiment (IC2)
- Event window must fit within observed data range (IC1)

*Target parameter (Equation 2):*

    theta_kappa^e = sum_{a in Omega_kappa} ATT(a, a+e) * (N_a^D / N_Omega_kappa^D)

where:
- `theta_kappa^e` = trimmed aggregate ATT at event time e
- `Omega_kappa` = trimmed set of adoption events satisfying IC1 and IC2
- `N_a^D` = number of treated units in sub-experiment a
- `N_Omega_kappa^D` = total treated units across all sub-experiments in trimmed set

*Estimator equation (Equation 3 — weighted saturated event study, recommended):*

    Y_sae = alpha_0 + alpha_1 * D_sa + sum_{h != -1} [lambda_h * 1(e=h) + delta_h * D_sa * 1(e=h)] + U_sae

Estimated via WLS with Q-weights. The delta_h coefficients identify theta_kappa^e.

*Q-weights (Section 5.3, Table 1):*

    Q_sa = 1                                           if D_sa = 1 (treated)
    Q_sa = (N_a^D / N^D) / (N_a^C / N^C)             if D_sa = 0 (control, aggregate weighting)
    Q_sa = (Pop_a^D / Pop^D) / (N_a^C / N^C)         if D_sa = 0 (control, population weighting)
    Q_sa = ((N_a + N_a^C)/(N^D+N^C)) / (N_a^C/N^C)  if D_sa = 0 (control, sample share weighting)

*Standard errors (Section 5.4):*
- Default: Cluster-robust standard errors at the group (unit) level
- Alternative: Cluster at group x sub-experiment level
- Both approaches yield approximately correct coverage when clusters > 100 (Table 2)
- No special bootstrap procedure specified; standard cluster-robust SEs recommended
- For post-period average: delta method or `lincom`/`marginaleffects`

*Edge cases:*
- All events trimmed: `len(Omega_kappa) == 0` -> ValueError suggesting reduced kappa
- No clean controls for event a: IC2 check fails -> Trim event, warn user
- Single cohort in trimmed set: Valid — Q-weights simplify
- Duplicate observations: Same (unit, time) appears in multiple sub-experiments -> handled by clustering at unit level
- Constant treatment share across sub-exps: Unweighted FE recovers correct estimand (special case, Section 5.5)
- Anticipation > 0: Reference period shifts to e = -1 - anticipation. Post-treatment includes anticipation periods (e >= -anticipation). Window expands by anticipation pre-periods.
- Group aggregation: Not supported — pooled stacked regression cannot produce cohort-specific effects. Use CallawaySantAnna or ImputationDiD.

*Algorithm (Section 5):*
1. Choose kappa_pre, kappa_post event window
2. Apply IC1 (window fits in data) and IC2 (clean controls exist) to get Omega_kappa
3. For each a in Omega_kappa: build sub-experiment with treated (A_s = a), clean controls (A_s > a + kappa_post), time window [a - kappa_pre, a + kappa_post] (with anticipation: [a - kappa_pre - anticipation, a + kappa_post])
4. Stack all sub-experiments vertically
5. Compute Q-weights: aggregate weighting uses observation counts per (event_time, sub_exp), matching R reference. Population/sample_share use unit counts per sub_exp (paper notation).
6. Run WLS regression of Equation 3 with Q-weights
7. Extract delta_h coefficients as event-study ATTs
8. Compute cluster-robust SEs at unit level

*IC1 (Adoption Event Window, Section 3):*

    IC1_a = 1[a - kappa_pre >= T_min  AND  a + kappa_post <= T_max]

Note: Matches R reference implementation (`focalAdoptionTime - kappa_pre >= minTime`).
The reference period a-1 is included in the window [a-kappa_pre, a+kappa_post] when kappa_pre >= 1.
The paper text states a stricter bound (T_min + 1) but the R code by the co-author uses T_min.

*IC2 (Clean Controls Exist, Section 3):*

    IC2_a = 1[exists s with A_s > a + kappa_post]    (not_yet_treated)
    IC2_a = 1[exists s with A_s > a + kappa_post + kappa_pre]  (strict)
    IC2_a = 1[exists s with A_s = infinity]           (never_treated)

**Reference implementation(s):**
- R: https://github.com/hollina/stacked-did-weights (`create_sub_exp()`, `compute_weights()`)
- No Stata or Python package; Stata estimation via standard `reghdfe` with Q-weight column

**Requirements checklist:**
- [x] Sub-experiment construction with treated + clean controls + time window
- [x] IC1 and IC2 trimming with warnings
- [x] Q-weight computation for all three weighting schemes (Table 1)
- [x] WLS via sqrt(w) transformation
- [x] Event study regression (Equation 3) with reference period e=-1
- [x] Cluster-robust SEs at unit or unit x sub-exp level
- [x] Overall ATT as average of post-treatment delta_h with delta-method SE
- [x] Anticipation parameter support
- [x] Never-treated encoding (0 and inf)

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

where Ȳ_{tr,t} is the mean treated outcome at time t, ω_j are unit weights, and λ_t are time weights.

*Unit weights ω (Frank-Wolfe on collapsed form):*

Build collapsed-form matrix Y_unit of shape (T_pre, N_co + 1), where the last column is the mean treated pre-period outcomes. Solve via Frank-Wolfe on the simplex:

```
min_{ω on simplex}  ζ_ω² ||ω||₂² + (1/T_pre) ||A_centered ω - b_centered||₂²
```

where A = Y_unit[:, :N_co], b = Y_unit[:, N_co], and centering is column-wise (intercept=True).

**Two-pass sparsification procedure** (matches R's `synthdid::sc.weight.fw` + `sparsify_function`):
1. First pass: Run Frank-Wolfe for 100 iterations (max_iter_pre_sparsify) from uniform initialization
2. Sparsify: `v[v <= max(v)/4] = 0; v = v / sum(v)` (zero out small weights, renormalize)
3. Second pass: Run Frank-Wolfe for 10000 iterations (max_iter) starting from sparsified weights

The sparsification step concentrates weights on the most important control units, improving interpretability and stability.

*Time weights λ (Frank-Wolfe on collapsed form):*

Build collapsed-form matrix Y_time of shape (N_co, T_pre + 1), where the last column is the per-control post-period mean (averaged across post-periods for each control unit). Solve:

```
min_{λ on simplex}  ζ_λ² ||λ||₂² + (1/N_co) ||A_centered λ - b_centered||₂²
```

where A = Y_time[:, :T_pre], b = Y_time[:, T_pre], and centering is column-wise.

*Auto-regularization (matching R's synthdid):*
```
noise_level = sd(first_differences of control outcomes)   # pooled across units
zeta_omega  = (N_treated × T_post)^(1/4) × noise_level
zeta_lambda = 1e-6 × noise_level
```

The noise level is computed as the standard deviation (ddof=1) of `np.diff(Y_pre_control, axis=0)`,
which computes first-differences across time for each control unit and pools all values.
This matches R's `sd(apply(Y[1:N0, 1:T0], 1, diff))`.

*Frank-Wolfe step (matches R's `fw.step()`):*
```
half_grad = A' (A x - b) + η x        # η = N × ζ²
i = argmin(half_grad)                   # vertex selection (simplex corner)
d_x = e_i - x                          # direction toward vertex i
step = -(half_grad · d_x) / (||A d_x||² + η ||d_x||²)
step = clip(step, 0, 1)
x_new = x + step × d_x
```

Convergence criterion: stop when objective decrease < min_decrease² (default min_decrease = 1e-5 × noise_level, max_iter = 10000, max_iter_pre_sparsify = 100).

*Standard errors:*

- Default: Placebo variance estimator (Algorithm 4 in paper)
  1. Randomly permute control unit indices
  2. Split into pseudo-controls (first N_co - N_tr) and pseudo-treated (last N_tr)
  3. Re-estimate unit weights (Frank-Wolfe) on pseudo-control/pseudo-treated data
  4. Re-estimate time weights (Frank-Wolfe) on pseudo-control data
  5. Compute SDID estimate with re-estimated weights
  6. Repeat `replications` times (default 200)
  7. `SE = sqrt((r-1)/r) × sd(placebo_estimates)` where r = number of successful replications

  This matches R's `synthdid::vcov(method="placebo")` which passes `update.omega=TRUE, update.lambda=TRUE` via `opts`.

- Alternative: Bootstrap at unit level (matching R's `synthdid::vcov(method="bootstrap")`)
  1. Resample ALL units (control + treated) with replacement
  2. Identify which resampled units are control vs treated
  3. Renormalize original unit weights for resampled controls: `ω_boot = _sum_normalize(ω[boot_control_idx])`
  4. Time weights λ remain **unchanged** from original estimation (fixed weights)
  5. Compute SDID estimate with renormalized ω and original λ
  6. `SE = sd(bootstrap_estimates, ddof=1)`

*Edge cases:*
- **Frank-Wolfe non-convergence**: Returns current weights after max_iter iterations. No warning emitted; the convergence check `vals[t-1] - vals[t] < min_decrease²` simply does not trigger early exit, and the final iterate is returned.
- **`_sparsify` all-zero input**: If `max(v) <= 0`, returns uniform weights `ones(len(v)) / len(v)`.
- **Single control unit**: `compute_sdid_unit_weights` returns `[1.0]` immediately (short-circuit before Frank-Wolfe).
- **Zero control units**: `compute_sdid_unit_weights` returns empty array `[]`.
- **Single pre-period**: `compute_time_weights` returns `[1.0]` when `n_pre <= 1` (Frank-Wolfe on a 1-element simplex is trivial).
- **Bootstrap with 0 control or 0 treated in resample**: Skip iteration (`continue`). If ALL bootstrap iterations fail, raises `ValueError`. If only 1 succeeds, warns and returns SE=0.0. If >5% failure rate, warns about reliability.
- **Placebo with n_control <= n_treated**: Warns that not enough control units for placebo variance estimation, returns SE=0.0 and empty placebo effects array. The check is `n_control - n_treated < 1`.
- **Negative weights attempted**: Frank-Wolfe operates on the simplex (non-negative, sum-to-1), so weights are always feasible by construction. The step size is clipped to [0, 1] and the move is toward a simplex vertex.
- **Perfect pre-treatment fit**: Regularization (ζ² ||ω||²) prevents overfitting by penalizing weight concentration.
- **Single treated unit**: Valid; placebo variance uses jackknife-style permutations of controls.
- **Noise level with < 2 pre-periods**: Returns 0.0, which makes both zeta_omega and zeta_lambda equal to 0.0 (no regularization). **Note (deviation from R):** `min_decrease` uses a `1e-5` floor when `noise_level == 0` to enable Frank-Wolfe early stopping. R would use `0.0`, causing FW to run all `max_iter` iterations; the result is equivalent since zero-noise data has no variation to optimize.
- **NaN inference for undefined statistics**: t_stat uses NaN when SE is zero or non-finite; p_value and CI also NaN. Matches CallawaySantAnna NaN convention.
- **Placebo p-value floor**: `p_value = max(empirical_p, 1/(n_replications + 1))` to avoid reporting exactly zero.
- **Varying treatment within unit**: Raises `ValueError`. SDID requires block treatment (constant within each unit). Suggests CallawaySantAnna or ImputationDiD for staggered adoption.
- **Unbalanced panel**: Raises `ValueError`. SDID requires all units observed in all periods. Suggests `balance_panel()`.
- **Poor pre-treatment fit**: Warns (`UserWarning`) when `pre_fit_rmse > std(treated_pre_outcomes, ddof=1)`. Diagnostic only; estimation proceeds.

**Reference implementation(s):**
- R: `synthdid::synthdid_estimate()` (Arkhangelsky et al.'s official package)
- Key R functions matched: `sc.weight.fw()` (Frank-Wolfe), `sparsify_function` (sparsification), `vcov.synthdid_estimate()` (variance)

**Requirements checklist:**
- [x] Unit weights: Frank-Wolfe on collapsed form (T_pre, N_co+1), two-pass sparsification (100 iters -> sparsify -> 10000 iters)
- [x] Time weights: Frank-Wolfe on collapsed form (N_co, T_pre+1), last column = per-control post mean
- [x] Unit and time weights: sum to 1, non-negative (simplex constraint)
- [x] Auto-regularization: noise_level = sd(first_diffs), zeta_omega = (N1*T1)^0.25 * noise_level, zeta_lambda = 1e-6 * noise_level
- [x] Sparsification: v[v <= max(v)/4] = 0; v = v/sum(v)
- [x] Placebo SE formula: sqrt((r-1)/r) * sd(placebo_estimates)
- [x] Placebo SE: re-estimates omega and lambda per replication (matching R's update.omega=TRUE, update.lambda=TRUE)
- [x] Bootstrap: fixed weights (original lambda unchanged, omega renormalized for resampled controls)
- [x] Returns both unit and time weights for interpretation
- [x] Column centering (intercept=True) in Frank-Wolfe optimization

---

## TripleDifference

**Primary source:** [Ortiz-Villavicencio, M., & Sant'Anna, P.H.C. (2025). Better Understanding Triple Differences Estimators. arXiv:2505.09942.](https://arxiv.org/abs/2505.09942)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires all 8 cells of the 2×2×2 design: Group(0/1) × Period(0/1) × Treatment(0/1)
- Warns if any cell has fewer than threshold observations
- Propensity score overlap required for IPW/DR methods

*Estimator equation (as implemented):*

Three-DiD decomposition (matching R's `triplediff::ddd()`):
```
Subgroups: 4=G1P1, 3=G1P0, 2=G0P1, 1=G0P0
DDD = DiD_3 + DiD_2 - DiD_1
```
where `DiD_j` is a pairwise DiD comparing subgroup j vs subgroup 4 (reference).

Each pairwise DiD uses the selected estimation method (DR, IPW, or RA) with
repeated cross-section implementation (`panel=FALSE` in R).

Regression adjustment (RA): Separate OLS per subgroup-time cell within each
pairwise comparison, imputed counterfactual means.

IPW: Propensity score P(subgroup=4|X) within {j, 4} subset, Hajek normalization.

Doubly robust (DR): Combines outcome regression and IPW with efficiency correction
(OR bias correction term).

*Standard errors (all methods):*

Individual-level (default):
```
SE = std(w₃·IF₃ + w₂·IF₂ - w₁·IF₁, ddof=1) / sqrt(n)
```
where `w_j = n / n_j`, `n_j = |{subgroup=j}| + |{subgroup=4}|`, and `IF_j` is the
per-observation influence function for pairwise DiD j (padded to full n with zeros).

Cluster-robust (when `cluster` parameter is provided):
```
SE = sqrt( (G/(G-1)) * (1/n²) * Σ_c ψ_c² )
```
where `G` is the number of clusters, `ψ_c = Σ_{i∈c} IF_i` is the sum of the combined
influence function within cluster `c`, and the `G/(G-1)` factor is the Liang-Zeger
finite-sample adjustment.

Note: IF-based SEs are inherently heteroskedasticity-robust; the `robust` parameter
has no additional effect.

*Edge cases:*
- Propensity scores near 0/1: trimmed at `pscore_trim` (default 0.01)
- Empty cells: raises ValueError with diagnostic message
- Low cell counts: warns when any cell has fewer than 10 observations
- Cluster-robust SE: requires at least 2 clusters (raises `ValueError`)
- Cluster IDs: must not contain NaN (raises `ValueError`)
- Overlap warning: emitted when >5% of observations are trimmed at pscore bounds (IPW/DR only)
- Propensity score estimation failure: falls back to unconditional probability P(subgroup=4),
  sets hessian=None (skipping PS correction in influence function), emits UserWarning
- Collinear covariates: detected via pivoted QR in `solve_ols()`, action controlled by
  `rank_deficient_action` ("warn", "error", "silent")
- Non-finite influence function values (e.g., from extreme propensity scores in IPW/DR
  or near-singular design): warns and sets SE to NaN, propagated to t_stat/p_value/CI
  via safe_inference()
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - p_value and CI: Also NaN when t_stat is NaN
  - **Note**: Defensive enhancement; reference implementation behavior not yet documented

**Reference implementation(s):**
- R `triplediff::ddd()` (v0.2.1, CRAN) — official companion by paper authors

**Requirements checklist:**
- [x] All 8 cells (G×P×T) must have observations
- [x] Propensity scores clipped at `pscore_trim` bounds
- [x] Doubly robust consistent if either propensity or outcome model correct
- [x] Returns cell means for diagnostic inspection
- [x] Supports RA, IPW, and DR estimation methods
- [x] Three-DiD decomposition: DDD = DiD_3 + DiD_2 - DiD_1 (matching R)
- [x] Influence function SE: std(w3·IF_3 + w2·IF_2 - w1·IF_1) / sqrt(n)
- [x] Cluster-robust SE via Liang-Zeger variance on influence function
- [x] ATT and SE match R within <0.001% for all methods and DGP types

---

## TROP

**Primary source:** [Athey, S., Imbens, G.W., Qu, Z., & Viviano, D. (2025). Triply Robust Panel Estimators. arXiv:2508.21536.](https://arxiv.org/abs/2508.21536)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires sufficient pre-treatment periods for factor estimation (at least 2)
- Warns if estimated rank seems too high/low relative to panel dimensions
- Unit weights can become degenerate if λ_unit too large
- Returns Q(λ) = ∞ if ANY LOOCV fit fails (Equation 5 compliance)

*Treatment indicator (D matrix) semantics:*

D must be an **ABSORBING STATE** indicator, not a treatment timing indicator:
- D[t, i] = 0 for all t < g_i (pre-treatment periods for unit i)
- D[t, i] = 1 for all t >= g_i (during and after treatment for unit i)

where g_i is the treatment start time for unit i.

For staggered adoption, different units have different treatment start times g_i.
The D matrix naturally handles this - distances use periods where BOTH units
have D=0, matching the paper's (1 - W_iu)(1 - W_ju) formula in Equation 3.

**Wrong D specification**: If user provides event-style D (only first treatment period
has D=1), ATT will be incorrect - document this clearly.

*ATT definition (Equation 1, Section 6.1):*
```
τ̂ = (1 / Σ_i Σ_t W_{it}) Σ_{i=1}^N Σ_{t=1}^T W_{it} τ̂_{it}(λ̂)
```
- ATT averages over ALL cells where D_it=1 (treatment indicator)
- No separate "post_periods" concept - D matrix is the sole input for treatment timing
- Supports general assignment patterns including staggered adoption

*Estimator equation (as implemented, Section 2.2):*

Working model (separating unit/time FE from regularized factor component):
```
Y_it(0) = α_i + β_t + L_it + ε_it,   E[ε_it | L] = 0
```
where α_i are unit fixed effects, β_t are time fixed effects, and L = UΣV' is a low-rank
factor structure. The FE are estimated separately from L because L is regularized but
the fixed effects are not.

Optimization (Equation 2):
```
(α̂, β̂, L̂) = argmin_{α,β,L} Σ_j Σ_s θ_s^{i,t} ω_j^{i,t} (1-W_js)(Y_js - α_j - β_s - L_js)² + λ_nn ||L||_*
```
Solved via alternating minimization with soft-thresholding of singular values for L:
```
L̂ = U × soft_threshold(Σ, λ_nn) × V'
```

Per-observation weights (Equation 3):
```
θ_s^{i,t}(λ) = exp(-λ_time × |t - s|)

ω_j^{i,t}(λ) = exp(-λ_unit × dist^unit_{-t}(j, i))

dist^unit_{-t}(j, i) = (Σ_u 1{u≠t}(1-W_iu)(1-W_ju)(Y_iu - Y_ju)² / Σ_u 1{u≠t}(1-W_iu)(1-W_ju))^{1/2}
```
Note: weights are per-(i,t) observation-specific. The distance formula excludes the
target period t and uses only periods where both units are untreated (W=0).

*Special cases (Section 2.2):*
- λ_nn=∞, ω_j=θ_s=1 (uniform weights) → recovers DID/TWFE
- ω_j=θ_s=1, λ_nn<∞ → recovers Matrix Completion (Athey et al. 2021)
- λ_nn=∞ with specific ω_j, θ_s → recovers SC/SDID

*LOOCV tuning parameter selection (Equation 5, Footnote 2):*
```
Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
```
- Score is **SUM** of squared pseudo-treatment effects on control observations
- **Two-stage procedure** (per paper's footnote 2):
  - Stage 1: Univariate grid searches with extreme fixed values
    - λ_time search: fix λ_unit=0, λ_nn=∞ (disabled)
    - λ_nn search: fix λ_time=0 (uniform time weights), λ_unit=0
    - λ_unit search: fix λ_nn=∞, λ_time=0
  - Stage 2: Cycling (coordinate descent) until convergence
- **"Disabled" parameter semantics** (per paper Section 4.3, Table 5, Footnote 2):
  - `λ_time=0`: Uniform time weights (disabled), because exp(-0 × dist) = 1
  - `λ_unit=0`: Uniform unit weights (disabled), because exp(-0 × dist) = 1
  - `λ_nn=∞`: Factor model disabled (L=0), because infinite penalty; converted to `1e10` internally
  - **Note**: `λ_nn=0` means NO regularization (full-rank L), which is the OPPOSITE of "disabled"
  - **Validation**: `lambda_time_grid` and `lambda_unit_grid` must not contain inf. A `ValueError` is raised if they do, guiding users to use 0.0 for uniform weights per Eq. 3.
- **LOOCV failure handling** (Equation 5 compliance):
  - If ANY LOOCV fit fails for a parameter combination, Q(λ) = ∞
  - A warning is emitted on the first failure with the observation (t, i) and λ values
  - Subsequent failures for the same λ are not individually warned (early return)
  - This ensures λ selection only considers fully estimable combinations

*Standard errors:*
- Block bootstrap preserving panel structure (Algorithm 3)

*Edge cases:*
- Rank selection: automatic via cross-validation, information criterion, or elbow
- Zero singular values: handled by soft-thresholding
- Extreme distances: weights regularized to prevent degeneracy
- LOOCV fit failures: returns Q(λ) = ∞ on first failure (per Equation 5 requirement that Q sums over ALL control observations where D==0); if all parameter combinations fail, falls back to defaults (1.0, 1.0, 0.1)
- **λ_nn=∞ implementation**: Only λ_nn uses infinity (converted to 1e10 for computation):
  - λ_nn=∞ → 1e10 (large penalty → L≈0, factor model disabled)
  - Conversion applied to grid values during LOOCV (including Rust backend)
  - Conversion applied to selected values for point estimation
  - Conversion applied to selected values for variance estimation (ensures SE matches ATT)
  - **Results storage**: `TROPResults` stores *original* λ_nn value (inf), while computations use 1e10. λ_time and λ_unit store their selected values directly (0.0 = uniform).
- **Empty control observations**: If no valid control observations exist, returns Q(λ) = ∞ with warning. A score of 0.0 would incorrectly "win" over legitimate parameters.
- **Infinite LOOCV score handling**: If best LOOCV score is infinite, `best_lambda` is set to None, triggering defaults fallback
- Validation: requires at least 2 periods before first treatment
- **D matrix validation**: Treatment indicator must be an absorbing state (monotonic non-decreasing per unit)
  - Detection: `np.diff(D, axis=0) < 0` for any column indicates violation
  - Handling: Raises `ValueError` with list of violating unit IDs and remediation guidance
  - Error message includes: "convert to absorbing state: D[t, i] = 1 for all t >= first treatment period"
  - **Rationale**: Event-style D (0→1→0) silently biases ATT; runtime validation prevents misuse
  - **Unbalanced panels**: Missing unit-period observations are allowed. Monotonicity validation checks each unit's *observed* D sequence for monotonicity, which correctly catches 1→0 violations that span missing period gaps (e.g., D[2]=1, missing [3,4], D[5]=0 is detected as a violation even though the gap hides the transition in adjacent-period checks).
  - **n_post_periods metadata**: Counts periods where D=1 is actually observed (at least one unit has D=1), not calendar periods from first treatment. In unbalanced panels where treated units are missing in some post-treatment periods, only periods with observed D=1 values are counted.
- Wrong D specification: if user provides event-style D (only first treatment period),
  the absorbing-state validation will raise ValueError with helpful guidance
- **Bootstrap minimum**: `n_bootstrap` must be >= 2 (enforced via `ValueError`). TROP uses bootstrap for all variance estimation — there is no analytical SE formula.
- **LOOCV failure metadata**: When LOOCV fits fail in the Rust backend, the first failed observation coordinates (t, i) are returned to Python for informative warning messages
- **Inference CI distribution**: After `safe_inference()` migration, CI uses t-distribution (df = max(1, n_treated_obs - 1)), consistent with p_value. Previously CI used normal-distribution while p_value used t-distribution (inconsistent). This is a minor behavioral change; CIs may be slightly wider for small n_treated_obs.

**Reference implementation(s):**
- Authors' replication code (forthcoming)

**Requirements checklist:**
- [x] Factor matrix estimated via soft-threshold SVD
- [x] Unit weights: `exp(-λ_unit × distance)` (unnormalized, matching Eq. 2)
- [x] LOOCV implemented for tuning parameter selection
- [x] LOOCV uses SUM of squared errors per Equation 5
- [x] Multiple rank selection methods: cv, ic, elbow
- [x] Returns factor loadings and scores for interpretation
- [x] ATT averages over all D==1 cells (general assignment patterns)
- [x] No post_periods parameter (D matrix determines treatment timing)
- [x] D matrix semantics documented (absorbing state, not event indicator)
- [x] Unbalanced panels supported (missing observations don't trigger false violations)

### TROP Joint Optimization Method

**Method**: `method="joint"` in TROP estimator

**Approach**: Joint weighted least squares with optional nuclear norm penalty.
Estimates fixed effects, factor matrix, and scalar treatment effect simultaneously.

**Objective function** (Equation J1):
```
min_{μ, α, β, L, τ}  Σ_{i,t} δ_{it} × (Y_{it} - μ - α_i - β_t - L_{it} - W_{it}×τ)² + λ_nn×||L||_*
```

where:
- δ_{it} = δ_time(t) × δ_unit(i) are observation weights (product of time and unit weights)
- μ is the intercept
- α_i are unit fixed effects
- β_t are time fixed effects
- L_{it} is the low-rank factor component
- τ is a **single scalar** (homogeneous treatment effect assumption)
- W_{it} is the treatment indicator

**Weight computation** (differs from twostep):
- Time weights: δ_time(t) = exp(-λ_time × |t - center|) where center = T - treated_periods/2
- Unit weights: δ_unit(i) = exp(-λ_unit × RMSE(i, treated_avg))
  where RMSE is computed over pre-treatment periods comparing to average treated trajectory

**Implementation approach** (without CVXPY):

1. **Without low-rank (λ_nn = ∞)**: Standard weighted least squares
   - Build design matrix with unit/time dummies + treatment indicator
   - Solve via iterative coordinate descent for (μ, α, β, τ)

2. **With low-rank (finite λ_nn)**: Alternating minimization
   - Alternate between:
     - Fix L, solve weighted LS for (μ, α, β, τ)
     - Fix (μ, α, β, τ), soft-threshold SVD for L (proximal step)
   - Continue until convergence

**LOOCV parameter selection** (unified with twostep, Equation 5):
Following paper's Equation 5 and footnote 2:
```
Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
```
where τ̂_js^loocv is the pseudo-treatment effect at control observation (j,s)
with that observation excluded from fitting.

For joint method, LOOCV works as follows:
1. For each control observation (t, i):
   - Zero out weight δ_{ti} = 0 (exclude from weighted objective)
   - Fit joint model on remaining data → obtain (μ̂, α̂, β̂, L̂)
   - Compute pseudo-treatment: τ̂_{ti} = Y_{ti} - μ̂ - α̂_i - β̂_t - L̂_{ti}
2. Score = Σ τ̂_{ti}² (sum of squared pseudo-treatment effects)
3. Select λ combination that minimizes Q(λ)

**Rust acceleration**: The LOOCV grid search is parallelized in Rust for 5-10x speedup.
- `loocv_grid_search_joint()` - Parallel LOOCV across all λ combinations
- `bootstrap_trop_variance_joint()` - Parallel bootstrap variance estimation

**Key differences from twostep method**:
- Treatment effect τ is a single scalar (homogeneous assumption) vs. per-observation τ_{it}
- Global weights (distance to treated block center) vs. per-observation weights
- Single model fit per λ combination vs. N_treated fits
- Faster computation for large panels

**Assumptions**:
- **Simultaneous adoption (enforced)**: The joint method requires all treated units
  to receive treatment at the same time. A `ValueError` is raised if staggered
  adoption is detected (units first treated at different periods). Treatment timing is
  inferred once and held constant for bootstrap variance estimation.
  For staggered adoption designs, use `method="twostep"`.

**Reference**: Adapted from reference implementation. See also Athey et al. (2025).

**Requirements checklist:**
- [x] Same LOOCV framework as twostep (Equation 5)
- [x] Global weight computation using treated block center
- [x] Weighted least squares with treatment indicator
- [x] Alternating minimization for nuclear norm penalty
- [x] Returns scalar τ (homogeneous treatment effect)
- [x] Rust acceleration for LOOCV and bootstrap

---

# Diagnostics & Sensitivity

## PlaceboTests

**Module:** `diff_diff/diagnostics.py`

*Edge cases:*
- NaN inference for undefined statistics:
  - `permutation_test`: t_stat is NaN when permutation SE is zero (all permutations produce identical estimates)
  - `leave_one_out_test`: t_stat, p_value, CI are NaN when LOO SE is zero (all LOO effects identical)
  - **Note**: Defensive enhancement matching CallawaySantAnna NaN convention

---

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

# Visualization

## Event Study Plotting (`plot_event_study`)

**Reference Period Normalization**

Normalization only occurs when `reference_period` is **explicitly specified** by the user:

- **Explicit `reference_period=X`**: Normalizes effects (subtracts ref effect), sets ref SE to NaN
  - Point estimates: `effect_normalized = effect - effect_ref`
  - Reference period SE → NaN (it's now a constraint, not an estimate)
  - Other periods' SEs unchanged (uncertainty relative to the constraint)
  - CIs recomputed from normalized effects and original SEs

- **Auto-inferred reference** (from CallawaySantAnna results): Hollow marker styling only, no normalization
  - Original effects are plotted unchanged
  - Reference period shown with hollow marker for visual indication
  - All periods retain their original SEs and error bars

This design prevents unintended normalization when the reference period isn't a true
identifying constraint (e.g., CallawaySantAnna with `base_period="varying"` where different
cohorts use different comparison periods).

The explicit-only normalization follows the `fixest` (R) convention where the omitted/reference
category is an identifying constraint with no associated uncertainty. Auto-inferred references
follow the `did` (R) package convention which does not normalize and reports full inference.

**Rationale**: When normalizing to a reference period, we're treating that period as an
identifying constraint (effect ≡ 0 by definition). The variance of a constant is zero,
but since it's a constraint rather than an estimated quantity, we report NaN rather than 0.
Auto-inferred references may not represent true identifying constraints, so normalization
should be a deliberate user choice.

**Edge Cases:**
- If `reference_period` not in data: No normalization applied
- If reference effect is NaN: No normalization applied
- Reference period CI becomes (NaN, NaN) after normalization (explicit only)
- Reference period is plotted with hollow marker (both explicit and auto-inferred)
- Reference period error bars: removed for explicit, retained for auto-inferred

**Reference implementation(s):**
- R: `fixest::coefplot()` with reference category shown at 0 with no CI
- R: `did::ggdid()` does not normalize; shows full inference for all periods

---

# Cross-Reference: Standard Errors Summary

| Estimator | Default SE | Alternatives |
|-----------|-----------|--------------|
| DifferenceInDifferences | HC1 robust | Cluster-robust, wild bootstrap |
| MultiPeriodDiD | HC1 robust | Cluster-robust (via `cluster` param), wild bootstrap |
| TwoWayFixedEffects | Cluster at unit | Wild bootstrap |
| CallawaySantAnna | Analytical (influence fn) | Multiplier bootstrap |
| SunAbraham | Cluster-robust + delta method | Pairs bootstrap |
| ImputationDiD | Conservative clustered (Thm 3) | Multiplier bootstrap (library extension; percentile CIs and empirical p-values, consistent with CS/SA) |
| TwoStageDiD | GMM sandwich (Newey & McFadden 1994) | Multiplier bootstrap on GMM influence function |
| SyntheticDiD | Placebo variance (Alg 4) | Unit-level bootstrap (fixed weights) |
| TripleDifference | Influence function (all methods) | SE = std(IF) / sqrt(n) |
| StackedDiD | Cluster-robust (unit) | Cluster at unit × sub-experiment |
| TROP | Block bootstrap | — |
| BaconDecomposition | N/A (exact decomposition) | Individual 2×2 SEs |
| HonestDiD | Inherited from event study | FLCI, C-LF |
| PreTrendsPower | Exact (analytical) | - |
| PowerAnalysis | Exact (analytical) | Simulation-based |

---

# Cross-Reference: R Package Equivalents

| diff-diff Estimator | R Package | Function |
|---------------------|-----------|----------|
| DifferenceInDifferences | fixest | `feols(y ~ treat:post, ...)` |
| MultiPeriodDiD | fixest | `feols(y ~ i(time, treat, ref=ref) \| unit + time)` |
| TwoWayFixedEffects | fixest | `feols(y ~ treat \| unit + time, ...)` |
| CallawaySantAnna | did | `att_gt()` |
| SunAbraham | fixest | `sunab()` |
| ImputationDiD | didimputation | `did_imputation()` |
| TwoStageDiD | did2s | `did2s()` |
| ContinuousDiD | contdid | `cont_did()` |
| SyntheticDiD | synthdid | `synthdid_estimate()` |
| TripleDifference | triplediff | `ddd()` |
| StackedDiD | stacked-did-weights | `create_sub_exp()` + `compute_weights()` |
| TROP | - | (forthcoming) |
| BaconDecomposition | bacondecomp | `bacon()` |
| HonestDiD | HonestDiD | `createSensitivityResults()` |
| PreTrendsPower | pretrends | `pretrends()` |
| PowerAnalysis | pwr / DeclareDesign | `pwr.t.test()` / simulation |

---

# Version History

- **v1.0** (2025-01-19): Initial registry with 12 estimators
