# Code Review: PR #22 - CallawaySantAnna Covariate Adjustment

**Reviewer:** Claude
**Date:** 2026-01-03
**Status:** Approved with suggestions

## Summary

This PR implements covariate adjustment for the `CallawaySantAnna` estimator, enabling conditional parallel trends assumptions. This addresses a key 1.0 blocker from the roadmap.

## Changes Made During Review

The following issues were fixed directly in this review:

1. **Removed unused instance variable** (`staggered.py:596-597`)
   - `self._covariates = covariates` was stored but never used
   - Covariates are passed through the method chain instead

2. **Fixed empty influence function in IPW** (`staggered.py:883-886`)
   - The unconditional IPW case returned `np.array([])` as a placeholder
   - Now properly computes the influence function for consistency with other methods

3. **Added test for extreme propensity scores** (`test_staggered.py:656-707`)
   - Tests that propensity score clipping handles near-perfect separation gracefully

## Suggestions for Future Work

### Standard Error Approximation

The SE calculation in `_outcome_regression` (line 776) ignores estimation error in the regression coefficients:

```python
# Approximate SE (ignoring estimation error in beta for simplicity)
se = np.sqrt(var_t / n_t + var_c / n_c)
```

For a full sandwich variance estimator, see Sant'Anna & Zhao (2020). This is a reasonable approximation for now but could be improved in a future release.

### Propensity Score Model Caching

In `_doubly_robust`, the propensity score is estimated independently (line 932). For efficiency with large datasets, consider refactoring to share the propensity model between IPW and DR estimators when both are called.

### Additional Test Coverage

Consider adding tests for:
- Near-collinear covariates (testing `_linear_regression` fallback to pseudo-inverse)
- Missing values in covariates (testing the fallback warning path)
- Very small sample sizes per group-time cell

### Documentation

The class docstring could include an example showing covariate-adjusted usage:

```python
# When parallel trends only holds conditional on covariates
cs = CallawaySantAnna(estimation_method='dr')  # doubly robust
results = cs.fit(data, outcome='outcome', unit='unit',
                 time='time', first_treat='first_treat',
                 covariates=['age', 'income'])
```

## References

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with multiple time periods. Journal of Econometrics, 225(2), 200-230.
- Sant'Anna, P. H., & Zhao, J. (2020). Doubly robust difference-in-differences estimators. Journal of Econometrics, 219(1), 101-122.
