# Code Review: diff-diff Library

**Reviewer:** Claude (AI Code Review)
**Date:** 2026-01-02
**Repository:** diff-diff
**Version:** 0.1.0

---

## Executive Summary

This is a well-structured Python library implementing Difference-in-Differences (DiD) causal inference with an sklearn-like API. The codebase demonstrates solid understanding of econometric methodology and good software engineering practices. However, I've identified several methodological concerns and potential bugs that should be addressed before production use.

**Overall Assessment:** 🟡 Good foundation, but requires fixes before production use

---

# Part 1: Subject Matter Expert Review (Econometrics/Methodology)

## 1.1 Core DiD Estimation ✅ Mostly Correct

### What's Correct:
- The basic DiD formula is correctly implemented via OLS regression
- The interaction term (`treatment × post`) correctly identifies the ATT
- The design matrix construction is appropriate for the 2×2 case

### Methodological Issues:

#### 🔴 **CRITICAL: Absorbed Fixed Effects Implementation Has a Flaw**

**Location:** `estimators.py:187-195`

```python
if absorb:
    vars_to_demean = [outcome] + (covariates or [])
    for ab_var in absorb:
        n_absorbed_effects += working_data[ab_var].nunique() - 1
        for var in vars_to_demean:
            group_means = working_data.groupby(ab_var)[var].transform("mean")
            working_data[var] = working_data[var] - group_means
```

**Problem:** The absorbed fixed effects only demean the outcome and covariates, but **NOT** the treatment and time indicator variables. In a proper within-transformation for absorbed fixed effects, ALL variables in the regression (including `d`, `t`, and `dt`) should be demeaned by the absorbed groups.

**Impact:** This will produce biased ATT estimates when using `absorb` with variables that correlate with treatment assignment or timing.

**Correct Implementation:**
```python
vars_to_demean = [outcome, treatment, time] + (covariates or [])
```
And then the interaction term should be computed AFTER demeaning.

---

#### 🟡 **CONCERN: Two-Way Fixed Effects Not Demeaning Treatment Variables**

**Location:** `estimators.py:596-598`

```python
data_demeaned["_treatment_post"] = (
    data_demeaned[treatment] * data_demeaned[time]
)
```

**Problem:** The `_within_transform` method only demeans the outcome and covariates. The treatment indicator and time variables are NOT demeaned. However, since they're typically binary and may be time-invariant (treatment) or unit-invariant (time), this might be acceptable. BUT the interaction term should technically be created from demeaned components for correct TWFE estimation.

**Recommendation:** Document this as a limitation or implement proper TWFE demeaning.

---

#### 🟡 **CONCERN: Simple Parallel Trends Test Using Pooled Observations**

**Location:** `utils.py:206-229`

The `check_parallel_trends` function computes trends by pooling all observations within each group, treating them as independent observations in a simple linear regression. This ignores the panel structure and will produce incorrect standard errors.

**Issue:** When you have panel data with repeated observations per unit, treating all observations as independent will understate standard errors (because observations within units are correlated).

**Recommendation:** The test should either:
1. Aggregate to unit-period means first
2. Use clustered standard errors at the unit level
3. Use a proper panel data slope estimator

---

#### 🟡 **CONCERN: Degrees of Freedom Calculation in TWFE**

**Location:** `estimators.py:622-624`

```python
n_units = data[unit].nunique()
n_times = data[time].nunique()
df = len(y) - X.shape[1] - n_units - n_times + 2
```

**Issue:** The `+2` adjustment is attempting to account for the double-counting of the grand mean in two-way demeaning, but this formula may not be correct for all cases. The standard formula for TWFE degrees of freedom is:

```
df = N - K - n_units - n_times + 1
```

(The +1 accounts for the grand mean that's been absorbed by both unit and time effects.)

---

## 1.2 Robust Standard Errors ✅ Correct

### HC1 Implementation
The HC1 heteroskedasticity-robust standard errors are correctly implemented:

```python
adjustment = n / (n - k)
meat = X.T @ (X * u_squared[:, np.newaxis])
vcov = adjustment * XtX_inv @ meat @ XtX_inv
```

This is the correct sandwich estimator with HC1 small-sample adjustment.

### Cluster-Robust Standard Errors
The cluster-robust implementation is correct:

```python
adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
for cluster in unique_clusters:
    score_c = X_c.T @ u_c
    meat += np.outer(score_c, score_c)
```

This correctly sums the outer products of cluster-level score vectors.

---

## 1.3 Parallel Trends Testing (Wasserstein Distance) ✅ Novel and Mostly Correct

### What's Good:
- Using Wasserstein distance for distributional comparison is a valid and robust approach
- The permutation-based inference is correctly implemented
- Combining multiple tests (Wasserstein, KS) provides comprehensive assessment

### Issues:

#### 🟡 **CONCERN: Threshold of 0.2 for Normalized Wasserstein is Arbitrary**

**Location:** `utils.py:386-390`

```python
plausible = bool(
    wasserstein_p > 0.05 and
    (wasserstein_normalized < 0.2 if not np.isnan(wasserstein_normalized) else True)
)
```

The 0.2 threshold for normalized Wasserstein distance is described as a "rule of thumb" but has no theoretical justification. This should be:
1. Made configurable by the user
2. Documented with appropriate caveats about its arbitrary nature

---

#### 🟡 **CONCERN: Using `np.random.seed()` Globally**

**Location:** `utils.py:324-325`

```python
if seed is not None:
    np.random.seed(seed)
```

This sets the global random state, which can affect other code running in the same session.

**Recommendation:** Use `np.random.default_rng(seed)` for local random state:
```python
rng = np.random.default_rng(seed)
perm_idx = rng.permutation(n_total)
```

---

## 1.4 Equivalence Testing (TOST) ✅ Correct

The Two One-Sided Tests (TOST) implementation is methodologically correct:
- Proper Welch-Satterthwaite degrees of freedom approximation
- Correct formulation of the two one-sided tests
- Maximum p-value correctly identifies the binding constraint

The default equivalence margin of 0.5 × pooled SD is a reasonable choice (similar to Cohen's d effect size interpretation).

---

## 1.5 Missing Methodological Features

1. **No Staggered DiD Support:** The documentation warns about TWFE bias with staggered treatment, but no alternative estimators (Callaway-Sant'Anna, Sun-Abraham, etc.) are provided.

2. **No Event Study Plots:** Standard DiD analysis typically includes event study / leads-and-lags analysis to visually assess pre-trends.

3. **No Weights Support:** No ability to use sampling weights or propensity score weights.

4. **No Treatment Intensity:** Only binary treatment is supported; no continuous/dose treatment option.

---

# Part 2: Engineering Review (Bugs & Edge Cases)

## 2.1 Critical Bugs

### 🔴 **BUG: Fixed Effects Dummies Use Original Data Instead of Working Data**

**Location:** `estimators.py:223`

```python
dummies = pd.get_dummies(data[fe], prefix=fe, drop_first=True)
```

**Problem:** When both `absorb` and `fixed_effects` are used together, the dummies are created from `data` (original) instead of `working_data` (demeaned). This is inconsistent and could lead to unexpected behavior.

**Fix:**
```python
dummies = pd.get_dummies(working_data[fe], prefix=fe, drop_first=True)
```

---

### 🔴 **BUG: Division by Zero in Standard Error Calculation**

**Location:** `utils.py:227`

```python
se_slope = np.sqrt(mse / np.sum((time_norm - mean_t) ** 2))
```

**Problem:** If all time values are the same (after normalization), `np.sum((time_norm - mean_t) ** 2)` will be zero, causing division by zero.

**Edge Case:** This can happen if `pre_periods` contains only one unique time period.

---

### 🔴 **BUG: Division by Zero in Degrees of Freedom Calculation**

**Location:** `utils.py:555-556`

```python
df = ((var_t/n_t + var_c/n_c)**2 /
      ((var_t/n_t)**2/(n_t-1) + (var_c/n_c)**2/(n_c-1)))
```

**Problem:** If `n_t == 1` or `n_c == 1`, we get division by zero (`n_t-1 = 0`).

The check on line 526 requires `len() < 2`, but this should be `<= 1` to properly catch the case where exactly 2 data points exist (which is still problematic for variance calculation).

---

### 🟡 **BUG: Variance Calculation with Single Observation**

**Location:** `utils.py:377-378`

```python
var_treated = np.var(treated_changes, ddof=1)
var_control = np.var(control_changes, ddof=1)
```

**Problem:** If either array has only one element, `np.var(..., ddof=1)` returns `nan` (since you can't compute variance with ddof=1 from a single observation). This propagates through all subsequent calculations.

---

## 2.2 Edge Cases Not Handled

### 🟡 **Perfect Multicollinearity Detection Missing**

**Location:** `estimators.py:302-303`

```python
coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
```

**Problem:** If the design matrix has perfect multicollinearity (e.g., from fixed effects dummies that sum to a constant), `lstsq` will still return coefficients but they may be numerically unstable or arbitrary.

**Recommendation:** Add a rank check:
```python
if np.linalg.matrix_rank(X) < X.shape[1]:
    raise ValueError("Design matrix is rank-deficient (perfect multicollinearity)")
```

---

### 🟡 **No Check for Sufficient Variation in Fixed Effects**

**Location:** `estimators.py:220-226`

If a fixed effect variable has only one category, creating dummies with `drop_first=True` results in zero dummy columns, but no warning is raised.

---

### 🟡 **Formula Parsing Does Not Handle Whitespace Consistently**

**Location:** `estimators.py:352-357`

```python
parts = rhs.split("*")
treatment = parts[0].strip()
time = parts[1].strip()
```

**Problem:** The formula `"outcome ~ treated*post + cov"` works, but `"outcome ~ treated * post + cov"` (with spaces around `*`) also works. However, edge cases like `"outcome ~ treated *post"` (asymmetric spacing) may cause issues.

The current implementation handles this via `.strip()`, but the splitting logic doesn't account for cases like:
```
"outcome ~ treatment + time * something"
```
This would incorrectly parse `"treatment + time"` as the treatment variable.

---

### 🟡 **Missing Data Silently Filtered in Some Functions**

**Location:** `utils.py:442`

```python
changes_data = data_sorted.dropna(subset=["_outcome_change"])
```

This silently drops observations with missing changes. While this is expected for the first period, unexpected NaNs from the original data would be silently dropped without warning.

---

## 2.3 Performance Issues

### 🟡 **Inefficient Cluster Loop**

**Location:** `utils.py:84-90`

```python
for cluster in unique_clusters:
    mask = cluster_ids == cluster
    X_c = X[mask]
    u_c = residuals[mask]
    score_c = X_c.T @ u_c
    meat += np.outer(score_c, score_c)
```

**Problem:** For large numbers of clusters, this loop is slow. A vectorized implementation using pandas groupby or sparse matrices would be more efficient.

---

### 🟡 **Permutation Test Could Use Parallel Processing**

**Location:** `utils.py:362-367`

```python
for i in range(n_permutations):
    perm_idx = np.random.permutation(n_total)
    ...
```

For large datasets and many permutations, this could benefit from parallelization (e.g., using `joblib` or `concurrent.futures`).

---

## 2.4 Code Quality Issues

### 🟡 **Inconsistent Error Handling**

Some functions return NaN values for invalid inputs (e.g., `check_parallel_trends_robust`), while others raise exceptions (e.g., `DifferenceInDifferences.fit`). This inconsistency can confuse users.

---

### 🟡 **No Type Hints for Return Types in Some Functions**

**Location:** Multiple functions in `utils.py`

Functions like `_compute_outcome_changes` have type hints for parameters but not for return types.

---

### 🟡 **Magic Numbers**

Several magic numbers appear without constants:
- `0.2` for normalized Wasserstein threshold (`utils.py:389`)
- `0.5` for default equivalence margin multiplier (`utils.py:547`)
- `1000` for default permutations (`utils.py:260`)

These should be defined as module-level constants with documentation.

---

## 2.5 Test Coverage Gaps

### Missing Tests:

1. **No test for `absorb` + `fixed_effects` combination**
2. **No test for cluster-robust SE in base DiD estimator**
3. **No test for formula with covariates beyond interaction**
4. **No test for edge case with single pre-period**
5. **No test for the `TwoWayFixedEffects` class**
6. **No negative test for perfect multicollinearity**
7. **No test for missing values in data**
8. **No test for very small sample sizes (n < 10)**

---

# Summary of Recommended Fixes

## Critical (Must Fix)

| Issue | Location | Description |
|-------|----------|-------------|
| Absorbed FE not demeaning treatment vars | `estimators.py:187-195` | Demean all regression variables, not just outcome |
| FE dummies using wrong data source | `estimators.py:223` | Use `working_data` instead of `data` |
| Division by zero in SE | `utils.py:227` | Add guard for single-period data |
| Division by zero in df calculation | `utils.py:555-556` | Check for `n_t <= 2` or `n_c <= 2` |

## Important (Should Fix)

| Issue | Location | Description |
|-------|----------|-------------|
| Global random seed | `utils.py:324-325` | Use `np.random.default_rng()` |
| TWFE df formula | `estimators.py:622-624` | Verify +2 vs +1 adjustment |
| Rank deficiency check | `estimators.py:302-303` | Add matrix rank check |
| Arbitrary Wasserstein threshold | `utils.py:389` | Make configurable |

## Nice to Have

| Issue | Location | Description |
|-------|----------|-------------|
| Parallel trends SE correction | `utils.py:206-229` | Account for panel structure |
| Cluster loop optimization | `utils.py:84-90` | Vectorize for performance |
| Permutation parallelization | `utils.py:362-367` | Add parallel option |
| Add TwoWayFixedEffects tests | `test_estimators.py` | Expand test coverage |

---

# Conclusion

The **diff-diff** library is a solid foundation for DiD analysis in Python. The core methodology is largely correct, and the code is well-organized with good documentation. However, several issues need attention:

1. **The absorbed fixed effects implementation has a critical bug** that will produce biased estimates
2. **Edge cases with small samples** can cause division-by-zero errors
3. **Test coverage** should be expanded, especially for the TWFE estimator

With these fixes, the library would be suitable for production use in applied econometrics research.

---

*Review completed by Claude AI. All findings should be verified by a human domain expert before implementation.*
