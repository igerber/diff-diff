# Continuous Difference-in-Differences: Methodology Reference

**Paper:** Callaway, Goodman-Bacon & Sant'Anna (2024/2025), "Difference-in-Differences
with a Continuous Treatment," NBER Working Paper 32117 (arXiv: 2107.02637v8).

**R reference implementation:** `contdid` v0.1.0 (CRAN)

---

## 1. Problem Statement

In many DiD applications, treatment has a **dose** or intensity rather than binary on/off.
Examples: pollution exposure varying by distance, different minimum wage levels, varying
tax rates, different subsidy shares.

Binary DiD cannot:
1. Handle settings where all units receive *some* treatment (no clean untreated group)
2. Estimate dose-response relationships
3. Identify effects of marginal changes in treatment intensity

### What goes wrong with TWFE

The standard TWFE regression for continuous DiD:

```
Y_{i,t} = theta_t + eta_i + beta^{twfe} * D_i * Post_t + v_{i,t}    (Eq. 1.1)
```

beta^{twfe} suffers from multiple simultaneous problems:
- **Negative weights on levels**: Weights on ATT(l|l) integrate to *zero*, not one.
  Below-average dose units get negative weight.
- **Selection bias under standard PT**: A contamination term persists even under
  standard parallel trends.
- **Non-representative weighting under strong PT**: Even without selection bias,
  weights don't match the dose density, making beta^{twfe} sensitive to the untreated
  group share.
- **Scale dependence**: Rescaling dose (0-1 to 0-100) changes beta^{twfe} proportionally,
  while natural target parameters are invariant.

---

## 2. Setup and Notation

### Two-period baseline case

- Two periods: t=1 (pre), t=2 (post)
- In t=1, no unit is treated (all have dose 0)
- In t=2, some units receive dose D_i in D_+ subset (0, inf), rest remain at D_i = 0

### Key variables

| Symbol | Definition |
|--------|-----------|
| Y_{i,t} | Outcome for unit i at time t |
| D_i | Dose (treatment intensity). D_i in D = {0} union D_+ |
| D_+ = (d_L, d_U) | Support of continuous dose among treated |
| Y_{i,t}(d) | Potential outcome under dose d |
| Delta Y | Y_{t=2} - Y_{t=1} |
| f_{D\|D>0}(d) | Dose density conditional on being treated |

### Multi-period staggered notation

| Symbol | Definition |
|--------|-----------|
| G_i | Timing group: period when unit i first treated |
| G = inf | Never-treated units |
| W_{i,t} = D_i * 1{t >= G_i} | Treatment exposure at time t |
| Y_{i,t}(g,d) | Potential outcome if first treated in period g with dose d |

### What is a "dose"? What is a "group"?

- **Dose**: The amount/intensity of treatment. Can be continuous or multi-valued discrete.
  Crucially, dose is **time-invariant** for each unit (the amount doesn't change once assigned).
- **Group**: In two-period case, all units with the same dose d. In multi-period case,
  characterized by both timing (G_i) and dose (D_i).

---

## 3. Target Parameters / Estimands

### 3.1 Level treatment effects

**Local ATT** (effect of dose d on units who received dose d'):
```
ATT(d|d') = E[Y_{t=2}(d) - Y_{t=2}(0) | D = d']
```

When d' = d (the "own-dose" case):
```
ATT(d|d) = E[Y_{t=2}(d) - Y_{t=2}(0) | D = d]
```

**Global ATT** (effect of dose d across all treated):
```
ATT(d) = E[Y_{t=2}(d) - Y_{t=2}(0) | D > 0]
```

Key distinction: ATT(d|d) != ATT(d) when there is selection into dose groups on
treatment effects. Under standard PT, only ATT(d|d) is identified. Under strong PT,
ATT(d|d) = ATT(d).

### 3.2 Average causal response (ACRT)

For **continuous** treatment:
```
ACRT(d|d') = d/dl ATT(l|d') |_{l=d}
ACRT(d)    = d/dd ATT(d)
```

For **discrete** (multi-valued) treatment:
```
ACRT(d_j|d_k) = [ATT(d_j|d_k) - ATT(d_{j-1}|d_k)] / (d_j - d_{j-1})
```

Level effects = height of dose-response curve (switching from 0 to d).
Causal responses = slope of dose-response curve (marginal increase in d).
These coincide for binary treatment; they diverge for continuous treatment.

### 3.3 Summary parameters

Four natural scalar summaries:
```
ATT^{loc}   = E[ATT(D|D)   | D > 0]    -- local levels, weighted by dose density
ATT^{glob}  = E[ATT(D)     | D > 0]    -- global levels, weighted by dose density
ACRT^{loc}  = E[ACRT(D|D)  | D > 0]    -- local slopes, weighted by dose density
ACRT^{glob} = E[ACRT(D)    | D > 0]    -- global slopes, weighted by dose density
```

`loc` = each dose group's own curve. `glob` = single curve for all treated.

### 3.4 Multi-period parameters

Group-time-dose specific:
```
ATT(g,t,d|g,d) = E[Y_t(g,d) - Y_t(0) | G=g, D=d]
```

Aggregated across groups/periods by dose:
```
ATT^{dose}(d|d) = weighted average of ATT(g,t,d|g,d) across (g,t) cells
```

Event-study versions:
```
ATT^{es}_{loc}(e) = E[ATT^{dose,es}(D|D,e) | G+e in [2,T], G <= T]
```

where e = event time (periods since treatment).

---

## 4. Identifying Assumptions

### Assumption PT (Standard / Weak Parallel Trends)

For all d in D_+:
```
E[Y_{t=2}(0) - Y_{t=1}(0) | D = d] = E[Y_{t=2}(0) - Y_{t=1}(0) | D = 0]
```

Untreated potential outcome paths are the same across all dose groups and the
untreated group. Direct analog of binary DiD parallel trends.

**Identifies**: ATT(d|d), ATT^{loc}. Does NOT identify ATT(d), ACRT, or any
cross-dose comparison.

### Assumption SPT (Strong Parallel Trends)

For all d in D:
```
E[Y_{t=2}(d) - Y_{t=1}(0) | D > 0] = E[Y_{t=2}(d) - Y_{t=1}(0) | D = d]
```

No selection into dose groups on the basis of treatment effects. Implies
ATT(d|d) = ATT(d) for all d.

**Additionally identifies**: ATT(d), ACRT(d), ACRT^{glob}, and cross-dose
comparisons have causal interpretation.

### Other assumptions

- **No anticipation**: Y_{i,t=1} = Y_{i,t=1}(0) for all units
- **Overlap**: P(D=0) > 0 (untreated units exist); f_{D|D>0}(d) > 0 on D_+
- **Multi-period PT-MP**: E[Delta Y_t(0) | G=g, D=d] = E[Delta Y_t(0) | G=inf, D=0]

### Comparison to binary DiD

When D in {0,1}: PT = SPT, ATT(1|1) = ATT(1) = ATT^{loc} = ATT^{glob},
ACRT = ATT. Everything collapses to standard Callaway & Sant'Anna (2021).

---

## 5. Estimation Procedures

### 5.1 Discrete treatment: saturated regression

When dose takes values d_1, ..., d_J (Eq. 4.1):
```
Delta Y_i = beta_0 + sum_{j=1}^{J} 1{D_i = d_j} * beta_j + epsilon_i
```

- beta_j estimates ATT(d_j)
- (beta_j - beta_{j-1}) / (d_j - d_{j-1}) estimates ACRT(d_j)
- Standard OLS inference applies

### 5.2 Continuous treatment: parametric (B-spline sieve)

**This is the default in the R package and the recommended starting point.**

For each (g,t) 2x2 cell:

1. Compute untreated counterfactual: E_n[Delta Y | D=0]
2. Demean treated outcomes: Delta tilde{Y}_i = Delta Y_i - E_n[Delta Y | D=0] for D_i > 0
3. Construct B-spline basis psi^K(D) of degree `degree` with `num_knots` interior knots
4. OLS regression among treated (Eq. 4.4):
   ```
   Delta tilde{Y}_i = psi^K(D_i)' beta_K + epsilon_i
   ```
5. Evaluate at dose grid (Eq. 4.5):
   ```
   ATT(d)  = psi^K(d)' * hat{beta}_K
   ACRT(d) = (d/dd psi^K(d))' * hat{beta}_K
   ```

**R package defaults**: degree=3 (cubic), num_knots=0 (global polynomial),
dose grid = quantiles P10 to P99 in 1% steps (90 points).

### 5.3 Continuous treatment: nonparametric CCK

Adapts Chen, Christensen & Kankanala (2025) sieve estimator with data-driven
dimension selection. Same regression framework as 5.2 but:

- Sieve dimension K is selected automatically via Lepski-type method (Algorithm 1)
- Provides honest, sup-norm rate-adaptive uniform confidence bands
- **Restricted to two-period settings** (no staggered adoption)

Algorithm for sieve dimension selection:
1. Define candidate set K = {2^k + 3 : k in N_+}
2. Compute K_max based on sample size
3. For each K in candidate set, test stability of estimates across K values
4. Select smallest K where estimates are stable (within bootstrap critical value)

### 5.4 Summary parameter estimation

**ATT^{glob} (binarized DiD)**: Under SPT (Eq. 4.6):
```
ATT^{glob} = E[Delta Y | D > 0] - E[Delta Y | D = 0]
```
Simple difference in means between any-treated and untreated.

**ACRT^{glob} (plug-in)**:
```
ACRT^{glob} = (1/n_{D>0}) * sum_{i: D_i > 0} ACRT(D_i)
```
Average the estimated ACRT curve over treated units' doses.

### 5.5 Multi-period estimation

For staggered adoption: apply two-period estimation to each (g,t) cell separately,
then aggregate. The R package handles this via the `ptetools` framework.

For event-study ATT^{es}_{loc}(e): can binarize treatment and use standard
Callaway & Sant'Anna (2021) machinery.

### 5.6 No untreated group (Remark 3.1)

When P(D=0) = 0 (all units receive some treatment), use the lowest dose group d_L
as comparison. Under PT, this recovers ATT(d|d) - ATT(d_L|d_L). Under SPT,
recovers ATT(d) - ATT(d_L).

---

## 6. Inference

### Parametric / discrete case

Standard OLS inference applies. Can cluster as needed.

### Nonparametric CCK case

**Multiplier (Gaussian) bootstrap** — NOT standard nonparametric bootstrap:
1. Draw omega_i iid N(0,1) for i=1,...,n
2. Compute weighted sums of influence functions using these weights
3. Repeat B=1000 times
4. No re-estimation needed per bootstrap draw (computationally efficient)

**Pointwise confidence intervals**:
```
ATT(d) +/- z_{0.975} * sigma_K(d) / sqrt(n)
```

**Uniform confidence bands (UCBs)**:
1. Compute bootstrap distribution of sup-t statistic across all dose values
2. Critical value c_alpha from (1-alpha) quantile
3. Band: ATT(d) +/- (c_alpha + A * gamma) * sigma(d) / sqrt(n)
4. These are honest and rate-adaptive

### Summary parameter inference

ACRT^{glob} plug-in estimator is sqrt(n)-consistent and asymptotically normal.
Standard errors via influence functions or multiplier bootstrap.

---

## 7. TWFE Decomposition (Theorem 3.4)

Four decompositions of beta^{twfe}, each revealing a different pathology:

| Decomposition | Weights positive? | Sum to 1? | Selection bias (under PT)? |
|:---|:---:|:---:|:---:|
| (a) Causal response | Yes | Yes | Yes |
| (b) Levels | No | No (sum to 0) | N/A |
| (c) Scaled levels | No | Yes | N/A |
| (d) Scaled 2x2 | Yes | Yes | Yes |

Even under SPT (best case), decomposition (a) uses TWFE-specific weights that
don't match the dose density, making beta^{twfe} an unappealing summary.

---

## 8. Key Theorems

| Theorem | Statement (plain English) |
|---------|--------------------------|
| 3.1 | Under PT: ATT(d\|d) = E[Delta Y \| D=d] - E[Delta Y \| D=0]. The local ATT for each dose group is identified by the standard DiD comparison. |
| 3.2 | Under PT: cross-dose comparisons mix causal effects with selection bias. The derivative of E[Delta Y\|D=d] does NOT identify ACRT without stronger assumptions. |
| 3.3 | Under SPT: ATT(d) = E[Delta Y\|D=d] - E[Delta Y\|D=0], and ACRT(d) = d/dd E[Delta Y\|D=d]. Cross-dose comparisons are causal. |
| 3.4 | TWFE decomposition: beta^{twfe} admits four representations, all problematic. |
| Cor 3.1 | ATT^{glob} = binarized DiD. ACRT^{glob} = weighted average of dose-specific slopes. |
| C.1 | Multi-period: ATT(g,t,d\|g,d) = E[Y_t - Y_{g-1}\|G=g,D=d] - E[Y_t - Y_{g-1}\|W_t=0]. |

---

## 9. R Package Implementation Details

### API surface

Main function: `cont_did()` returns `pte_dose_results` or `dose_obj`.

Key parameters:
```
cont_did(yname, dname, gname, tname, idname, data,
         target_parameter = "level"|"slope",
         aggregation = "dose"|"eventstudy"|"none",
         treatment_type = "continuous"|"discrete",
         dose_est_method = "parametric"|"cck",
         dvals = NULL,
         degree = 3, num_knots = 0,
         control_group = "notyettreated"|"nevertreated"|"eventuallytreated",
         anticipation = 0,
         bstrap = TRUE, boot_type = "multiplier", biters = 1000,
         cband = FALSE, alp = 0.05,
         base_period = "varying",
         ...)
```

### Core algorithm per (g,t) cell

1. Extract 2x2 subset: target group (g) + control group, pre-period + post-period
2. Construct B-spline basis from treated units' doses using `splines2::bSpline()`

   > **Boundary knot note**: The B-spline boundary knots are set from the
   > training doses (`range(dose_treated)`). Evaluation at `dvals` is clamped
   > to these boundaries. R's `contdid` v0.1.0 uses `range(dvals)` as boundary
   > knots when evaluating, which can cause extrapolation artifacts. This is an
   > intentional deviation.

3. OLS: regress Delta Y on B-spline basis
4. Evaluate fitted spline at `dvals` -> ATT(d) vector
5. Evaluate derivative of spline at `dvals` -> ACRT(d) vector
6. Return estimates + influence functions for bootstrap

### Aggregation

- **`"dose"`**: Average across (g,t) cells at each dose point -> dose-response curve
- **`"eventstudy"`**: Average across dose at each event time e -> dynamic effects
- **`"none"`**: Return disaggregated (g,t,d) results

### Data conventions

- **Dose is time-invariant**: Set to actual value in ALL periods (pre and post)
- **Never-treated**: G=0, dose forced to 0 internally
- **Balanced panel required** in v0.1.0
- Units with treatment timing but zero dose are dropped

### Default dose grid

```
dvals = quantile(dose[dose > 0], probs = seq(0.10, 0.99, 0.01))
```
90 evaluation points, P10 to P99. Provides implicit tail trimming.

### Knot placement

Quantile-based by default: `choose_knots_quantile(dose[dose > 0], num_knots)`.
With `num_knots=0`, no interior knots (global polynomial of given degree).
Knots are built **once globally** from all positive doses, not per (g,t) cell.
This ensures a common basis space across cells so that dose-response vectors
can be meaningfully aggregated.

### Dependencies mapping (R -> Python)

| R Package | Purpose | Python Equivalent |
|-----------|---------|-------------------|
| `splines2` | B-spline basis + derivatives | `scipy.interpolate.BSpline` + custom derivative |
| `sandwich` | Robust variance | Already in diff-diff `linalg.py` |
| `ptetools` | Group-time iteration, aggregation, bootstrap | Reimplement (mirrors existing CS framework) |
| `MASS::ginv` | Pseudo-inverse | `numpy.linalg.pinv` |
| `npiv` | CCK nonparametric | Reimplement for CCK method |

### Current limitations (v0.1.0)

- Covariates not supported (xformula = ~1 only)
- Discrete treatment not yet implemented
- Unbalanced panels not supported
- CCK restricted to 2-period case
- Repeated cross-sections not supported

---

## 10. Implementation Priorities for diff-diff

### Phase 1 (Core)
1. Parametric B-spline estimation for two-period case
2. ATT(d) and ACRT(d) dose-response curves
3. Summary parameters: ATT^{glob}, ACRT^{glob}
4. Bootstrap inference (multiplier)

### Phase 2 (Staggered)
5. Multi-period extension via (g,t) cell iteration
6. Dose aggregation and event-study aggregation
7. Control group options (not-yet-treated, never-treated)

### Phase 3 (Advanced)
8. CCK nonparametric estimation
9. Uniform confidence bands
10. Covariates support (DR/IPW/OR)

### Defer
- Discrete treatment (saturated regression — simpler, add later)
- TWFE decomposition diagnostics
