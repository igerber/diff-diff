# Design-Based Variance Estimation for Modern DiD Estimators

**Key references:**

- Binder, D.A. (1983). "On the Variances of Asymptotically Normal Estimators
  from Complex Surveys." *International Statistical Review* 51(3), 279--292.
- Lumley, T. (2004). "Analysis of Complex Survey Samples." *Journal of
  Statistical Software* 9(8), 1--19.
- Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics* 225(2), 200--230.

**Implementation:** `diff_diff/survey.py`

---

## 1. Motivation

### 1.1. The problem: naive standard errors under complex survey designs

Policy evaluations frequently rely on nationally representative surveys:
NHANES (health outcomes), ACS (demographics and housing), BRFSS (behavioral
risk factors), CPS (labor force), and MEPS (medical expenditure). Most of these
are repeated cross-sectional surveys (with the partial exception of CPS's
rotating panel); the sampling frame --- strata, PSUs --- may shift across waves,
adding a layer of complexity to design-based variance estimation that does not
arise with a fixed panel. These surveys employ stratified multi-stage cluster
sampling to achieve national coverage at manageable cost. The resulting data
carry two features that invalidate naive standard errors:
(i) observations within the same primary sampling unit (PSU) are correlated,
and (ii) stratification constrains the sampling variability.

Naive standard errors --- whether heteroskedasticity-robust (HC1) or clustered
at the individual level --- treat the sample as if it were drawn by simple
random sampling. Under complex survey designs this ignores intra-cluster
correlation within PSUs, which typically inflates variance relative to SRS, and
stratification, which typically deflates it. The net effect is design-specific;
naive SEs are generally incorrect --- and often too small --- under complex
survey designs. The ratio of design-based to naive variance is the *design
effect* (DEFF); values of 2--5 are common in health and social surveys.

This matters especially for difference-in-differences (DiD) estimation because:

1. Treatment is often assigned at a level that aligns with PSU structure ---
   state policies, county programs, school-district mandates --- so the
   within-PSU correlation of treatment intensifies the design effect on
   treatment-effect estimates.
2. DiD estimands involve contrasts across groups and time periods, amplifying
   any distortion in variance estimation.
3. Incorrect SEs can flip significance conclusions for policy-relevant effect
   sizes, undermining the credibility of program evaluations.

### 1.2. The gap: modern DiD theory assumes iid sampling

The modern DiD literature derives estimators and their asymptotic properties
under sampling assumptions that are incompatible with complex survey designs.
The foundational papers in this literature either assume iid sampling
explicitly, or adopt frameworks that do not incorporate complex survey design
features (strata, PSU clustering, FPC):

*Note on terminology.* The recent DiD literature uses "design-based" to refer
to treatment-assignment design (Athey & Imbens 2022), where uncertainty arises
from which units receive treatment; throughout this document, "design-based"
refers to survey sampling design (Binder 1983), where uncertainty arises from
which units are sampled. Same term, different referent.

- **Callaway & Sant'Anna (2021)** state iid as a numbered assumption
  (Assumption 2) and derive the multiplier bootstrap under it. The paper
  acknowledges design-based inference in the treatment-assignment sense ---
  citing Athey & Imbens (2018; published 2022) --- but does not pursue
  survey-design-based inference.
- **Sant'Anna & Zhao (2020)** assume iid (Assumption 1) and derive the doubly
  robust influence function and semiparametric efficiency bounds under it.
- **Borusyak, Jaravel & Spiess (2024)** adopt a conditional/fixed-design
  framework that avoids random sampling assumptions altogether, conditioning on
  the observation set. Their variance results do not address survey-sampling
  uncertainty.
- **Sun & Abraham (2021)** maintain iid as an unstated but operative
  assumption in deriving the interaction-weighted estimator.
- **de Chaisemartin & D'Haultfoeuille (2020)** assume group-level
  independence (Assumption 3), which does not map to the stratified-cluster
  structure of survey data.
- **Gardner (2022)** invokes standard GMM regularity conditions that
  implicitly require iid or ergodic stationary data.

The most comprehensive recent review of the DiD literature --- Roth, Sant'Anna,
Bilinski & Poe (2023), "What's Trending in Difference-in-Differences?" ---
discusses design-based inference in the treatment-assignment sense (Section 5.2),
where randomness comes from treatment assignment rather than sampling, but does
not address survey sampling design, survey weights, or strata/PSU/FPC-based
variance estimation.

### 1.3. The gap in software

Existing software implementations reflect this theoretical gap. R's `did`
package (Callaway & Sant'Anna) accepts a `weightsname` parameter for point
estimation and supports cluster-level multiplier bootstrap via `clustervars`
(drawing Rademacher weights at the cluster level rather than per unit), but
does not account for stratification or finite population corrections. Stata's
`csdid` (Rios-Avila, Sant'Anna & Callaway) accepts `pweight` for point
estimation and supports clustered wild bootstrap, but does not support the
`svy:` prefix --- there is no mechanism for strata or FPC.
`did_multiplegt_dyn` (de Chaisemartin & D'Haultfoeuille) clusters at the group
level by default but likewise lacks strata and FPC support.
`eventstudyinteract` (Sun & Abraham) does not accept probability weights.
`didimputation` (Borusyak, Jaravel & Spiess) accepts estimation weights via
`wname` but does not provide survey-design variance.

These implementations support weights for point estimation and allow
cluster-robust inference, but none provides full survey-design variance
estimation that jointly accounts for strata, PSU clustering, and finite
population corrections.

### 1.4. Adjacent work: survey inference for causal effects

The survey statistics literature has developed design-based variance theory for
smooth functionals (Binder 1983; Demnati & Rao 2004; Lumley 2004), and recent
work has extended this to causal inference --- but primarily for cross-sectional
estimators or simple two-period designs, not for modern staggered DiD:

- **DuGoff, Schuler & Stuart (2014)** provide practical guidance on combining
  propensity score methods with complex surveys using Stata's `svy:` framework,
  but address cross-sectional treatment effects, not DiD.
- **Zeng, Li & Tong (2025)** derive sandwich variance for survey-weighted
  propensity score estimators using influence functions --- the closest work to
  the bridge we describe --- but for cross-sectional IPW/augmented weighting,
  not staggered DiD.
- **Ye, Bilinski & Lee (2025)** study DiD with repeated cross-sectional survey
  data, combining propensity scores with survey weights. However, their
  estimator is limited to two periods and two groups, uses bootstrap-only
  variance (no analytical design-based derivation), and does not address the
  modern heterogeneity-robust estimators considered here.

No published work formally derives design-based variance --- in the survey-
statistics sense of strata/PSU/FPC-based Taylor series linearization --- for
the influence functions of modern heterogeneity-robust DiD estimators
(Callaway--Sant'Anna, Sun--Abraham, imputation DiD, etc.).

### 1.5. What this document provides

This document bridges the two literatures. The core argument (Section 4) is a
careful application of existing survey linearization theory (Binder 1983) to
modern DiD estimators: because these estimators are smooth functionals of the
empirical distribution, Binder's theorem guarantees that applying the
stratified-cluster variance formula to their influence function values produces
a design-consistent variance estimator. The argument applies existing theory to
a new setting --- it has not previously been stated for the modern
heterogeneity-robust DiD case.

diff-diff implements this connection: it is the only package --- across R,
Stata, and Python --- that provides design-based variance estimation
(Taylor Series Linearization with strata/PSU/FPC, and replicate weight methods)
for modern heterogeneity-robust DiD estimators.

For a code walkthrough, see the
[survey tutorial](https://github.com/igerber/diff-diff/blob/main/docs/tutorials/16_survey_did.ipynb).
For the compatibility matrix showing which estimators support which survey
features, see the [Survey Design Support](../choosing_estimator.rst#survey-design-support)
section.

---

## 2. Setup and Notation

### Finite population and survey design

Consider a finite population U = {1, ..., N}. The population is partitioned
into H non-overlapping strata. Within stratum h, there are N_h PSUs in the
population, of which n_h are sampled. Within each sampled PSU, observations
are either fully enumerated or sub-sampled. This describes the standard
stratified multi-stage design used by most federal statistical agencies.

### Sampling weights

Each sampled observation i carries a sampling weight w_i = 1 / pi_i, where
pi_i is the inclusion probability. Under probability-weight (`pweight`)
semantics, the raw weight w_i = 1/pi_i represents how many population units
observation i represents. diff-diff normalizes probability weights to mean 1
(sum = n) to avoid scale dependence in regression coefficients. After
normalization, weights preserve relative representativeness --- w_i = 2 means
observation i represents twice as many population units as the average --- but
no longer indicate absolute population counts.

### Finite population correction

The sampling fraction in stratum h is f_h = n_h / N_h. When f_h is close to
1, most of the finite population has been observed and sampling variability is
reduced. The finite population correction factor (1 - f_h) enters the variance
formula to account for this.

### Notation summary

| Symbol | Definition |
|--------|-----------|
| U = {1, ..., N} | Finite population |
| H | Number of strata |
| n_h | Number of sampled PSUs in stratum h |
| N_h | Total PSUs in stratum h (for FPC) |
| f_h = n_h / N_h | Sampling fraction in stratum h |
| w_i = 1 / pi_i | Sampling weight for observation i |
| F | Population distribution |
| F_hat_w | Survey-weighted empirical distribution |
| T(F) | Target functional (estimand) |
| theta_hat = T(F_hat_w) | Plug-in estimate |
| psi_i = IF(x_i; T, F) | Influence function value for observation i |

### Target estimand

The estimand is theta = T(F), where T is a functional mapping a distribution
to a real number (or vector). For DiD, T extracts treatment effects ---
average treatment effects on the treated (ATTs) --- from the joint distribution
of outcomes, treatment status, and time. The abstraction of theta as a
functional of F is what lets us bridge survey statistics and DiD: both
literatures reason about functionals, just from different perspectives.

---

## 3. Survey-Weighted Estimation

### Design consistency

Under the survey design, the survey-weighted empirical distribution is:

```
F_hat_w = sum_i w_i * delta_{x_i} / sum_i w_i
```

This is the Hájek (self-normalized) form of the design-weighted estimator,
preferred when the population size N is unknown. It is design-consistent for
the same target as the Horvitz-Thompson estimator.

The sum is over sampled observations and delta_{x_i} is the point mass
at x_i. When T is a smooth functional, the plug-in estimator theta_hat =
T(F_hat_w) is design-consistent for theta = T(F): as the sample size grows
within the finite-population asymptotic framework, theta_hat converges in
probability to theta.

### Regression-based estimators

For regression-based estimators (DifferenceInDifferences, TwoWayFixedEffects,
MultiPeriodDiD, SunAbraham, StackedDiD, ContinuousDiD), the point estimates
solve weighted estimating equations. The WLS formulation minimizes:

```
sum_i w_i * (Y_i - X_i' beta)^2
```

which yields the weighted normal equations:

```
sum_i w_i * X_i * (Y_i - X_i' beta) = 0
```

The implementation passes sqrt(w_i)-transformed data to `solve_ols()` in
`diff_diff/linalg.py`.

### Influence-function-based estimators

For IF-based estimators (CallawaySantAnna, ImputationDiD, TwoStageDiD,
EfficientDiD, TripleDifference, StaggeredTripleDifference), point estimates
are constructed from survey-weighted sample moments. For example,
CallawaySantAnna with `estimation_method='reg'` computes:

```
ATT(g,t) = sum_{i in G_g} w_i * Delta_Y_i / sum_{i in G_g} w_i
          - sum_{i in C} w_i * Delta_Y_i / sum_{i in C} w_i
```

Every step replaces simple sample averages (1/n) sum_i with weighted averages
(sum_i w_i)^{-1} sum_i w_i. For doubly-robust and IPW variants, the same
principle applies to the propensity score estimation (via survey-weighted
logistic regression) and outcome regression.

### When is weighting appropriate?

Solon, Haider & Wooldridge (2015) discuss when weighting is appropriate for
causal inference. Under design-based inference --- the perspective adopted by
diff-diff --- survey weights are needed to ensure that treatment effect
estimates correspond to the finite population, not just the sample. Without
weights, ATT estimates reflect the sample composition, which may
over-represent certain strata due to the sampling design.

---

## 4. Influence Functions and DiD

This section presents the core argument: why design-based variance estimation
is valid for modern DiD estimators. The argument proceeds in five steps.

### 4.1. Influence functions are properties of the functional

The influence function (IF) of a functional T at distribution F is the Gateaux
derivative:

```
IF(x; T, F) = lim_{eps -> 0} [T((1-eps)F + eps * delta_x) - T(F)] / eps
```

This is a property of the map T and the distribution F. It does not depend on
how the sample was drawn. The same functional T has the same IF regardless of
whether the data come from simple random sampling, stratified sampling, or
cluster sampling. The IF characterizes each observation's first-order
contribution to the estimator.

### 4.2. Modern DiD estimators are smooth functionals

Each modern DiD estimator can be written as theta = T(F) for a smooth
functional T that admits an influence function representation. The key
estimators and their smoothness arguments:

- **CallawaySantAnna (reg):** T(F) involves population means of outcomes
  within group-time cells. Sample means are smooth functionals of F.

- **CallawaySantAnna (dr/ipw):** T(F) additionally involves a propensity
  score model (smooth in population moments) and outcome regression (smooth in
  population moments). Sant'Anna & Zhao (2020) derive the full IF, including
  nuisance-function corrections.

- **SunAbraham:** T(F) is a linear functional of interaction-weighted
  regression coefficients, which are themselves smooth functionals of F via the
  implicit function theorem applied to the normal equations.

- **ImputationDiD:** T(F) involves OLS on untreated observations (smooth),
  counterfactual imputation (linear in coefficients), and averaging treatment-
  minus-imputed residuals (smooth). The IF follows from Theorem 3 of Borusyak,
  Jaravel & Spiess (2024).

- **EfficientDiD:** T(F) involves population means and covariances within
  cohort-time cells. The efficient influence function (EIF) is derived in the
  original paper.

- **ContinuousDiD:** T(F) involves B-spline regression coefficients, smooth
  functionals of F (Callaway, Goodman-Bacon & Sant'Anna 2024).

- **TripleDifference:** T(F) extends the two-group DiD sandwich to a triple
  contrast. The IF follows by the same arguments as DifferenceInDifferences.

- **StaggeredTripleDifference:** Staggered DDD with IF-based aggregation
  across group-time-subgroup cells. Smooth by the same logic as
  CallawaySantAnna.

- **TwoStageDiD:** Gardner (2022) two-stage imputation. The IF captures
  uncertainty from both the first-stage regression and the second-stage
  contrast.

- **WooldridgeDiD:** Poisson or OLS regression with saturated interaction
  terms. Smooth via the estimating-equation representation. *Note:* survey
  design support is not yet implemented for this estimator.

- **StackedDiD:** Q-weighted regression on stacked sub-experiments. Smooth in
  the population moments of each sub-experiment.

The common thread: all these estimators reduce to combinations of weighted
means, regression coefficients, and smooth transformations thereof. Each admits
an IF representation.

### 4.2a. Where the IF chain does not apply

Two estimators in diff-diff --- **SyntheticDiD** and **TROP** --- involve
non-smooth optimization steps (synthetic control weight selection, optimal
transport maps) that do not fit cleanly into the smooth-functional framework.
Their survey variance estimators bypass the TSL/IF framework entirely and use
resampling / permutation-style allocators tailored to each method's role:

- **TROP** uses Rao-Wu rescaled bootstrap at the PSU level. Per-observation
  treatment effects (tau_it) are deterministic given the data and do not
  depend on survey weights, so the Rao-Wu path precomputes tau values once
  and only varies the ATT aggregation weights across draws (see REGISTRY.md
  for the documented optimization).
- **SyntheticDiD** supports all three variance methods under full
  strata/PSU/FPC designs. ``bootstrap`` uses hybrid pairs-bootstrap + Rao-Wu
  rescaling composed with a weighted Frank-Wolfe kernel (each draw re-runs
  the full estimator on the resampled panel). ``placebo`` uses stratified
  permutation + weighted Frank-Wolfe (pseudo-treated sampled within each
  treated-containing stratum). ``jackknife`` uses PSU-level leave-one-out
  with Rust-Rao stratum aggregation (fixed ω, λ — no refit per LOO). See
  the bullets under "4.2b. SyntheticDiD survey resampling allocators" below
  and REGISTRY.md §SyntheticDiD for the full derivations. Replicate-weight
  designs remain rejected (no replicate-weight variance path). The TSL/IF-
  based argument in this document does not extend to these estimators.

### 4.3. Under survey weighting, the same IF form applies

Under survey weighting, we replace F with F_hat_w (the survey-weighted
empirical distribution). The estimator becomes theta_hat = T(F_hat_w). Because
the IF is a property of T, not the sampling design, the first-order von Mises
expansion is:

```
T(F_hat_w) - T(F) = sum_i d_i * psi_i + o_p(n^{-1/2})
```

where d_i = 1 if unit i is sampled (0 otherwise), and psi_i = w_i * IF(x_i;
T, F) / N is the scaled influence function value. (In practice, the population
size N is typically unknown and is estimated by N_hat = sum_i w_i. After
diff-diff normalizes pweights to mean 1, sum_i w_i = n; the scaling is
variance-equivalent because only relative weights affect the sandwich meat.) The key observation: this
linearized form is a weighted sum over the sampled observations, and its
variance is determined by the sampling design --- not by T. The IF transforms
the problem of estimating Var(theta_hat) into the simpler problem of estimating
the variance of a weighted total.

### 4.4. Binder's (1983) result

Binder (1983) formalized this insight. The key result: for any smooth
functional T, the design-based variance of theta_hat = T(F_hat_w) can be
consistently estimated by applying the standard stratified-cluster variance
formula to the per-unit IF values psi_i. Specifically:

```
V_hat(theta_hat) = sum_h (1 - f_h) * (n_h / (n_h - 1))
                   * sum_{j=1}^{n_h} (psi_hj - psi_h_bar)^2
```

where psi_hj = sum_{i in PSU j, stratum h} psi_i is the PSU-level total of IF
values, and psi_h_bar is the within-stratum mean of PSU totals.

This works because theta_hat is asymptotically equivalent to a linear function
of survey-weighted totals. Once linearized via the IF, the variance of
theta_hat inherits the same structure as the variance of a design-weighted
total, which the survey statistics literature has established formulas for.

### 4.5. Combining the pieces

The chain of reasoning:

1. Modern DiD estimators are smooth functionals of F (Section 4.2).
2. Their IFs are well-defined and do not depend on the sampling design
   (Section 4.1).
3. Under survey weighting, the estimator theta_hat = T(F_hat_w) has a
   first-order expansion in terms of the same IF values (Section 4.3).
4. Binder (1983) shows that applying the stratified-cluster variance formula
   to these IF values gives a consistent variance estimator (Section 4.4).

Therefore: plugging the IF values from any modern DiD estimator into the
stratified-cluster variance formula produces a design-consistent variance
estimator. This is exactly what diff-diff implements.

The argument requires that each DiD estimator satisfies the regularity
conditions for Binder's theorem (existence of a continuous IF, remainder
term of order o_p(n^{-1/2})). For regression-based estimators, this follows
from the implicit function theorem applied to the estimating equations. For
doubly-robust estimators, this follows from the semiparametric theory of
Sant'Anna & Zhao (2020). For imputation estimators, the IF from Theorem 3 of
Borusyak et al. (2024) satisfies these conditions.

---

## 5. Taylor Series Linearization (TSL) Variance

### Regression-based TSL sandwich

For regression-based estimators, the TSL variance-covariance matrix is the
stratified cluster sandwich (Binder 1983):

```
V_TSL = (X'WX)^{-1} [sum_h V_h] (X'WX)^{-1}
```

This is the standard sandwich estimator with the "meat" computed at the PSU
level within strata. The implementation is `compute_survey_vcov()` in
`diff_diff/survey.py`.

### Stratum-level meat

The variance contribution from stratum h is:

```
V_h = (1 - f_h) * (n_h / (n_h - 1)) * sum_{j=1}^{n_h} (T_hj - T_h_bar)(T_hj - T_h_bar)'
```

where:
- T_hj = sum_{i in PSU j, stratum h} w_i * X_i * u_i is the PSU-level score
  total (with u_i = Y_i - X_i' beta the residual),
- T_h_bar = (1/n_h) sum_j T_hj is the within-stratum mean of PSU-level scores,
- (1 - f_h) is the finite population correction,
- n_h / (n_h - 1) is the small-sample degrees-of-freedom adjustment.

The total meat is sum_h V_h, computed by `_compute_stratified_psu_meat()` in
`diff_diff/survey.py`.

### IF-based TSL variance

For scalar IF-based estimators (CallawaySantAnna, ImputationDiD, TwoStageDiD,
TripleDifference, StaggeredTripleDifference, EfficientDiD, DMLDiD), the variance is
computed directly from per-unit influence function values without the bread
matrix:

```
V_design = sum_h (1 - f_h) * (n_h / (n_h - 1)) * sum_{j=1}^{n_h} (psi_hj - psi_h_bar)^2
```

where psi_hj = sum_{i in PSU j, stratum h} psi_i is the PSU-level total of
IF values. This is the same formula as the meat in the regression sandwich, but
applied directly to the scalar IF values rather than to score vectors. The
implementation is `compute_survey_if_variance()` in `diff_diff/survey.py`.

**Residual-scale vs. score-scale.** These two functions accept inputs at
different scales. `compute_survey_vcov()` takes residuals on the original
scale (u_i = Y_i - X_i' beta) and multiplies by w_i internally to form
scores. `compute_survey_if_variance()` takes score-scale psi_i values
directly --- weights are already baked in. To see the connection: when
`compute_survey_vcov()` is called with X = [1]' and residuals = eif (raw
efficient influence function values), it internally forms scores = w_i * eif_i
and produces sandwich = (sum w)^{-2} * meat(w * eif). The scalar IF function
`compute_survey_if_variance(psi)` produces meat(psi) directly. These are
equivalent when psi_i = w_i * eif_i / sum(w) --- i.e., when the IF values
are on score-scale. EfficientDiD exploits this: the TSL path passes raw EIF
values to `compute_survey_vcov()` (which handles scaling), while the replicate
path explicitly converts to score-scale via psi = w * eif / sum(w) before
calling `compute_replicate_if_variance()`.

### Degrees of freedom

Inference uses the t-distribution with survey degrees of freedom:

| Design | df |
|--------|-----|
| Explicit PSU + strata | n_PSU - n_strata |
| Explicit PSU, no strata | n_PSU - 1 |
| Replicate weights | rank(W_rep) - 1 |
| No survey structure | n - 1 |

For replicate weights, the degrees of freedom are computed via the QR
rank of the analysis-weight matrix, matching R's `survey::degf()`.

### Singleton stratum handling

When a stratum contains only one sampled PSU (n_h = 1), the within-stratum
variance is undefined (division by n_h - 1 = 0). diff-diff provides three
strategies via the `lonely_psu` parameter:

- **"remove"**: Skip singleton strata and emit a warning. The variance estimate
  excludes these strata entirely.
- **"certainty"**: Treat singleton PSUs as sampled with certainty (f_h = 1),
  contributing zero to the variance.
- **"adjust"**: Center the singleton stratum's PSU total at the grand mean of
  all PSU totals instead of the (undefined) within-stratum mean (matching
  Stata's `singleunit(centered)` behavior).

---

## 6. Replicate Weight Variance

### Motivation

Replicate weights provide an alternative to TSL. Instead of linearizing the
estimator, they perturb the weights and observe the resulting variation in
estimates. This approach is useful when:

1. The survey agency provides pre-computed replicate weights with the
   public-use file (common for ACS, CPS, NHANES).
2. The estimator is too complex for easy linearization (though the IF-based
   approach in diff-diff largely eliminates this concern for smooth
   functionals).

Replicate weights are mutually exclusive with strata/PSU/FPC at the
`SurveyDesign` level: the design information is already embedded in the
replicate weight construction.

### Supported methods

diff-diff supports five replicate-weight methods. The general variance formula
is:

```
V_rep = c * sum_r s_r * (theta_r - theta_center)^2
```

where theta_r is the estimate from replicate r, theta_center is either the
full-sample estimate (mse=True) or the mean of replicate estimates (mse=False),
and c and s_r are method-specific factors.

The method-specific formulas (matching `_replicate_variance_factor()` in
`diff_diff/survey.py`):

```
BRR:  V = (1/R)              * sum_r (theta_r - theta)^2
Fay:  V = 1/[R * (1-rho)^2]  * sum_r (theta_r - theta)^2
JK1:  V = (R-1)/R            * sum_r (theta_r - theta)^2
SDR:  V = (4/R)              * sum_r (theta_r - theta)^2
JKn:  V = sum_h [(n_h-1)/n_h] * sum_{r in h} (theta_r - theta)^2
```

where R is the number of replicate columns, rho is the Fay perturbation
factor, and n_h is the number of replicates in stratum h (for JKn).

### Replicate variance for IF-based estimators

For regression-based estimators, `compute_replicate_vcov()` re-runs WLS for
each replicate weight column to obtain theta_r. For IF-based estimators, this
would require R complete re-fits of the estimator, which is computationally
expensive.

diff-diff avoids this for most IF-based estimators (CallawaySantAnna,
EfficientDiD, ContinuousDiD, TripleDifference, StaggeredTripleDifference)
using weight-ratio rescaling: the replicate estimate is computed by
reweighting the per-unit IF values rather than re-running the estimator.
The `SurveyDesign` parameter `combined_weights` controls the interpretation:

```
combined_weights=True:   theta_r = sum_i (w_{r,i} / w_i) * psi_i
combined_weights=False:  theta_r = sum_i  w_{r,i}         * psi_i
```

When `combined_weights=True`, the replicate columns w_{r,i} already
incorporate the full-sample weight, so the ratio w_{r,i} / w_i extracts the
perturbation factor. When `combined_weights=False`, the replicate columns are
the perturbation factors directly. This rescaling is numerically exact for
smooth functionals (to first order) and avoids the cost of R re-fits. The
implementation is in `compute_replicate_if_variance()` in `diff_diff/survey.py`.

**Exception: refit-based replicate variance.** For ImputationDiD and
TwoStageDiD, the first-stage regression (on untreated observations) must be
re-estimated with each replicate's weights to properly capture its
contribution to variance. These estimators use
`compute_replicate_refit_variance()`, which re-runs the full estimator for
each replicate column.

---

## 7. What Survey Weighting Does NOT Fix

Survey weighting addresses the sampling design. It does not resolve every
threat to valid causal inference with DiD. Practitioners should be aware of
the following limitations.

**Parallel trends.** Survey weighting ensures that treatment-effect estimates
target the correct population. It does not validate the parallel trends
assumption. Under the superpopulation model, parallel trends must hold for the
population, not just the sample. If parallel trends fail in the population,
survey-weighted estimates remain biased --- with correctly estimated standard
errors around the wrong estimand. Use `HonestDiD` for sensitivity analysis.

**Small-cluster asymptotics.** TSL variance requires at least 2 PSUs per
stratum (n_h >= 2). With few PSUs per stratum --- common in some state-based
surveys --- the t-distribution approximation with df = n_PSU - n_strata may
be anti-conservative. diff-diff reports the survey degrees of freedom so users
can assess this directly.

**Estimand dependence on weights.** The design-based framework treats population
values as fixed and relies on probability weighting to target finite-population
parameters. Binder's variance formula is consistent for the variance of
whatever the weighted estimator targets. However, if treatment effects vary
with inclusion probability in ways not captured by the stratification, the
survey-weighted estimator may target a different population quantity than the
intended ATT. In such cases, the variance estimate is correct for the estimand
actually being estimated, but that estimand may not correspond to the causal
parameter of interest.

**SUTVA.** Survey weighting does not address interference between units. If
treatment of one unit affects outcomes of another (spillovers), the ATT
estimand is not well-defined regardless of the variance estimator.

**Weight variability.** Highly variable weights reduce effective sample size.
The Kish design effect due to unequal weighting,
deff_w = n * sum(w_i^2) / (sum(w_i))^2, measures this: when deff_w >> 1,
estimates are less precise than the nominal sample size suggests. (This
captures only the weighting component of the full design effect discussed in
Section 1.1, which also incorporates clustering and stratification effects.)
diff-diff reports this quantity as `SurveyMetadata.design_effect` (the Kish
deff_w) to help users assess weight variability.

**Model misspecification.** For doubly-robust and IPW estimators
(CallawaySantAnna with `estimation_method='dr'` or `'ipw'`), the IF
corrections for propensity score and outcome regression uncertainty assume
correct specification of at least one nuisance model. Survey weighting does not
rescue a badly specified propensity score or outcome model.

---

## 8. Implementation in diff-diff

### Two variance paths

diff-diff provides two variance estimation strategies for survey data:

1. **Taylor Series Linearization (TSL):** Uses strata, PSU, and FPC to compute
   the stratified-cluster sandwich. Available for all estimators with
   analytical (non-bootstrap) survey variance.
2. **Replicate weights:** Uses pre-computed replicate weight columns (BRR, Fay,
   JK1, JKn, SDR). Available where indicated in the compatibility matrix.

These are mutually exclusive at the `SurveyDesign` level.

### Estimator survey variance dispatch

Each estimator uses one of three variance strategies under survey designs:

| Estimator | Variance path | Notes |
|-----------|--------------|-------|
| DifferenceInDifferences | TSL sandwich | OLS-based, all weight types |
| TwoWayFixedEffects | TSL sandwich | OLS-based, all weight types |
| MultiPeriodDiD | TSL sandwich | OLS-based, all weight types |
| CallawaySantAnna | TSL on IFs | pweight only |
| DMLDiD | TSL on IFs | pweight only; cross-fitted augmented scores; library extension of Chang (2020) i.i.d. theory |
| SunAbraham | TSL sandwich | OLS-based, all weight types |
| TripleDifference | TSL on IFs | pweight only |
| StaggeredTripleDifference | TSL on IFs | pweight only |
| ImputationDiD | TSL on IFs | pweight only |
| TwoStageDiD | TSL on IFs | pweight only |
| EfficientDiD | TSL on EIFs | all weight types |
| ContinuousDiD | TSL sandwich | all weight types |
| StackedDiD | TSL sandwich | pweight only |
| SyntheticDiD | Bootstrap / permutation / PSU-LOO | Not IF-amenable (Section 4.2a); all three variance methods support full strata/PSU/FPC designs |
| TROP | Bootstrap only | Not IF-amenable (Section 4.2a) |
| BaconDecomposition | Diagnostic only | Weighted descriptives, no inference |

For the definitive compatibility matrix including replicate weight and survey
bootstrap support, see the
[Survey Design Support](../choosing_estimator.rst#survey-design-support) section.

### IF-based variance path in detail

For IF-based estimators, the variance computation proceeds as:

1. The estimator computes per-unit influence function values psi_i for each
   group-time cell (g, t).
2. These are aggregated across cells with weight-influence-function (WIF)
   adjustment to produce a single per-unit IF vector for the overall ATT.
3. The aggregated IF vector is passed to `compute_survey_if_variance()`, which
   computes the design-based variance using `_compute_stratified_psu_meat()`.
4. For replicate weights, most IF-based estimators use
   `compute_replicate_if_variance()`, which reweights the IF vector via
   weight-ratio rescaling. ImputationDiD and TwoStageDiD instead use
   `compute_replicate_refit_variance()`, which re-runs the full estimator
   for each replicate column (see Section 6).

### Bootstrap and survey interaction

Two bootstrap strategies interact with survey designs:

- **Multiplier bootstrap at PSU level** (CallawaySantAnna, DMLDiD,
  ImputationDiD, TwoStageDiD, ContinuousDiD, EfficientDiD,
  StaggeredTripleDifference):
  Generates multiplier weights at the PSU level within strata, with FPC
  scaling. Each bootstrap draw reweights the IF values.

- **Rao-Wu rescaled bootstrap** (SunAbraham, TROP): Draws PSUs
  with replacement within strata and rescales observation weights. Each draw
  re-runs the full estimator on the resampled data.
- **Hybrid pairs-bootstrap + Rao-Wu rescaling** (SyntheticDiD, PR #355):
  SDID's full-design bootstrap is NOT a standalone Rao-Wu bootstrap. Each
  draw first performs the unit-level pairs-bootstrap resampling that
  Arkhangelsky et al. (2021) Algorithm 2 specifies (``boot_idx = rng.choice(n_total)``),
  and *then* applies the Rao-Wu rescaled per-unit weights sliced over the
  resampled units (``rw_control = rao_wu_rw[:n_control][boot_idx_control]``).
  The weighted-Frank-Wolfe kernel then solves
  ``min ||A·diag(rw)·ω - b||² + ζ²·Σ rw_i ω_i²`` on the resampled panel,
  and ``ω_eff = rw·ω / Σ(rw·ω)`` is composed for the SDID estimator.
  See REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap composition)``
  for the full objective and the argmin-set caveat.

- **Stratified permutation placebo** (SyntheticDiD): SDID's full-design
  placebo variance allocator (triggered when ``strata`` and/or ``psu``
  is declared on the ``SurveyDesign``). For each placebo draw,
  pseudo-treated indices are sampled uniformly without replacement
  from controls *within each stratum containing actual treated units*
  (classical stratified permutation test — Pesarin 2001).
  Pseudo-treated means are survey-weighted; weighted-FW re-estimates
  ω and λ per draw with ``rw_control`` threaded into both loss and
  regularization. Post-optimization composition
  ``ω_eff = rw · ω / Σ(rw · ω)`` with zero-mass retry. SE follows
  Arkhangelsky Algorithm 4:
  ``sqrt((r-1)/r) · std(placebo_estimates, ddof=1)``. Fit-time
  feasibility guards raise ``ValueError`` on three failure cases:
  Case B (treated stratum has 0 controls), Case C (fewer controls
  than treated in a treated stratum), and Case D (every treated
  stratum is exact-count ``n_c == n_t`` → permutation support = 1).
  ``SurveyDesign(fpc=...)`` is a documented no-op for placebo —
  permutation tests are conditional on the observed sample (Pesarin
  2001 §1.5), so the sampling fraction does not enter Algorithm 4 or
  its survey extension. An ``fpc=`` column emits a ``UserWarning`` and
  is not part of the placebo dispatch trigger. See REGISTRY.md
  §SyntheticDiD ``Note (survey + placebo composition)``.

- **PSU-level leave-one-out with stratum aggregation** (SyntheticDiD):
  SDID's full-design jackknife variance allocator, matching the
  canonical Rust & Rao (1996) stratified jackknife form:
  ``SE² = Σ_h (1 - f_h) · (n_h - 1)/n_h · Σ_{j∈h} (τ̂_{(h,j)} - τ̄_h)²``
  where ``f_h = n_h_sampled / fpc[h]`` is the per-stratum sampling
  fraction (population-count FPC form, matching ``SurveyDesign.resolve``).
  Fixed weights per LOO: ω subsetted over kept controls, composed with
  kept ``w_control``, renormalized; λ held at the fit-time value. Strata
  with ``n_h < 2`` are silently skipped (stratum-level variance
  unidentified; matches R ``survey::svyjkn`` under
  ``lonely_psu="remove"`` / ``"certainty"``). Full-census strata
  (``f_h ≥ 1``) short-circuit to zero contribution before any LOO
  feasibility check. ``SE = 0`` is returned for legitimate zero
  variance (every stratum full-census, or exact-zero within-stratum
  dispersion); ``SE = NaN`` with a ``UserWarning`` is reserved for
  undefined cases (all strata skipped, or any delete-one replicate in
  a non-full-census contributing stratum is undefined). Unstratified
  single-PSU designs short-circuit to ``SE = NaN``.
  ``SurveyDesign(lonely_psu="adjust")`` is **not yet supported** on
  this path and raises ``NotImplementedError``; use
  ``variance_method="bootstrap"`` or ``lonely_psu="remove"`` /
  ``"certainty"``. **Known limitation**: with ``n_h = 2`` per stratum,
  the stratified PSU-level jackknife has only 1 effective DoF per
  stratum and tends to be anti-conservative (see REGISTRY §SyntheticDiD
  calibration table for the ``stratified_survey × jackknife`` row).
  Users with few PSUs per stratum should prefer
  ``variance_method="bootstrap"``, which validates at near-nominal
  calibration on the same DGP.

---

## References

### Survey statistics

- Binder, D.A. (1983). "On the Variances of Asymptotically Normal Estimators
  from Complex Surveys." *International Statistical Review* 51(3), 279--292.
- Demnati, A. & Rao, J.N.K. (2004). "Linearization Variance Estimators for
  Survey Data." *Survey Methodology* 30(1), 17--26.
- Lumley, T. (2004). "Analysis of Complex Survey Samples." *Journal of
  Statistical Software* 9(8), 1--19.
- Rao, J.N.K. & Wu, C.F.J. (1988). "Resampling Inference with Complex Survey
  Data." *Journal of the American Statistical Association* 83(401), 231--241.
- Shao, J. (1996). "Resampling Methods in Sample Surveys." *Statistics*
  27(3--4), 203--237.

### Modern DiD

- Athey, S. & Imbens, G.W. (2022). "Design-Based Analysis in
  Difference-in-Differences Settings with Staggered Adoption." *Journal of
  Econometrics* 226(1), 62--79.
- Borusyak, K., Jaravel, X. & Spiess, J. (2024). "Revisiting Event-Study
  Designs: Robust and Efficient Estimation." *Review of Economic Studies*
  91(6), 3253--3285.
- Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics* 225(2), 200--230.
- Callaway, B., Goodman-Bacon, A. & Sant'Anna, P.H.C. (2024).
  "Difference-in-Differences with a Continuous Treatment." NBER Working Paper
  32117.
- de Chaisemartin, C. & D'Haultfoeuille, X. (2020). "Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects." *American Economic Review*
  110(9), 2964--2996.
- Gardner, J. (2022). "Two-Stage Differences in Differences." Working Paper.
- Roth, J., Sant'Anna, P.H.C., Bilinski, A. & Poe, J. (2023). "What's
  Trending in Difference-in-Differences? A Synthesis of the Recent
  Econometrics Literature." *Journal of Econometrics* 235(2), 2218--2244.
- Sant'Anna, P.H.C. & Zhao, J. (2020). "Doubly Robust Difference-in-
  Differences Estimators." *Journal of Econometrics* 219(1), 101--122.
- Sun, L. & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in
  Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*
  225(2), 175--199.

### Survey-weighted causal inference (cross-sectional)

- DuGoff, E.H., Schuler, M. & Stuart, E.A. (2014). "Generalizing
  Observational Study Results: Applying Propensity Score Methods to Complex
  Surveys." *Health Services Research* 49(1), 284--303.
- Solon, G., Haider, S.J. & Wooldridge, J.M. (2015). "What Are We Weighting
  For?" *Journal of Human Resources* 50(2), 301--316.
- Ye, K., Bilinski, A. & Lee, Y. (2025). "Difference-in-differences analysis
  with repeated cross-sectional survey data." *Health Services & Outcomes
  Research Methodology*. DOI: 10.1007/s10742-025-00364-7.
- Zeng, S., Li, F. & Tong, X. (2025). "Moving toward Best Practice when
  Using Propensity Score Weighting in Survey Observational Studies."
  arXiv:2501.16156.
