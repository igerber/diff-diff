# Code Review: PR #26 - Implement Honest DiD Sensitivity Analysis

**Reviewer**: Claude
**Date**: 2026-01-04
**PR Branch**: pr-26
**Commits**: 4 (15f6592, 66c37a8, 93dde89, 2426c24)

## Summary

This PR implements the Honest DiD sensitivity analysis framework (Rambachan & Roth 2023), which is a major feature addition that addresses a key credibility gap in the library. The implementation includes:

- New `honest_did.py` module with `HonestDiD` class and supporting types
- Visualization functions for sensitivity analysis
- Comprehensive test suite (699 lines)
- Tutorial notebook and documentation updates

**Overall Assessment**: ✅ **Approve with suggestions** - This is a well-structured implementation of an important feature. The code is well-documented and follows existing patterns. There are some minor issues and opportunities for improvement detailed below.

---

## Critical Issues

### 1. Potential Bug: Missing Pre-period Effects in `_extract_event_study_params`

**File**: `diff_diff/honest_did.py:433-464`

The function only extracts post-period effects from `MultiPeriodDiDResults`, but the identified set computation may need pre-period coefficients for proper constraint construction:

```python
# Current code only extracts post-period effects
for period in post_periods:
    pe = results.period_effects[period]
    effects.append(pe.effect)
    ses.append(pe.se)
```

**Concern**: If `MultiPeriodDiDResults.period_effects` only contains post-periods, the smoothness constraints may not work correctly since they span all periods.

**Recommendation**: Verify that the constraint matrices are built correctly for the post-period-only case, or add logic to extract pre-period coefficients when available (e.g., from the coefficients dict).

---

## High Priority Suggestions

### 2. Inefficient Breakdown Value Search

**File**: `diff_diff/honest_did.py:1343-1392`

The breakdown value search uses a coarse grid + bisection approach but has some inefficiencies:

```python
# This loop re-fits for every grid point
for M_test in M_grid:
    result = self.fit(results, M=M_test)
    if not result.is_significant:
        upper_bound = M_test
        if i > 0:
            lower_bound = M_grid[i - 1]
        break
```

**Recommendation**: Consider caching intermediate computations or using a more efficient root-finding algorithm. Also, the `_default_M` lookup could be cached:

```python
def __init__(...):
    ...
    self._default_M_cache: Dict[str, float] = {
        "relative_magnitude": 1.0,
        "smoothness": 0.0,
        "combined": 1.0,
    }
```

### 3. Hardcoded Significance Logic

**File**: `diff_diff/honest_did.py:193-198`

```python
@property
def significance_stars(self) -> str:
    """Return significance stars if CI excludes zero."""
    if self.is_significant:
        # Use p-value approximation based on CI
        return "*"
    return ""
```

The comment mentions "p-value approximation" but only returns a single star. This is inconsistent with `_get_significance_stars` used elsewhere in the codebase.

**Recommendation**: Either implement proper p-value based stars or update the docstring to clarify that this only indicates significance at the α level.

### 4. Unused Import in `__init__.py`

**File**: `diff_diff/__init__.py:65`

```python
from diff_diff.honest_did import (
    ...
    sensitivity_plot,  # This function doesn't exist in honest_did.py
)
```

**Issue**: `sensitivity_plot` is exported but the actual function in `honest_did.py` is named `plot_sensitivity_analysis` (lines 1395-1398), and there's also `plot_sensitivity` in `visualization.py`.

**Recommendation**: Either add an alias in `honest_did.py`:
```python
sensitivity_plot = plot_sensitivity_analysis
```
Or update the import to use the correct name.

---

## Medium Priority Suggestions

### 5. Inconsistent Error Messages

**File**: `diff_diff/honest_did.py`

Some validation errors use different styles:

```python
# Line 59
raise ValueError("M must be non-negative")

# Line 773
raise ValueError("method must be one of: 'smoothness', 'relative_magnitude', 'combined'")

# Line 778
raise ValueError("M must be non-negative")
```

**Recommendation**: Use consistent error message formatting, e.g., always include the parameter name:
```python
raise ValueError("M must be non-negative, got: {self.M}")
```

### 6. Magic Numbers in `_solve_bounds_lp`

**File**: `diff_diff/honest_did.py:573-631`

The LP solver uses hardcoded method and fallback behavior:

```python
result_min = optimize.linprog(
    c, A_ub=A_ineq, b_ub=b_ineq,
    bounds=(None, None),
    method='highs'  # Hardcoded solver
)
```

**Recommendation**: Consider making the solver configurable, as 'highs' may not be available in older scipy versions:
```python
def _solve_bounds_lp(..., method: str = 'highs'):
```

### 7. Missing Type Hints for Return Values in Some Methods

**File**: `diff_diff/honest_did.py`

Some methods have incomplete type hints:

```python
# Line 346 - Missing return type
def plot(self, ax=None, show_bounds: bool = True, ...):
```

**Recommendation**: Add return type hints for all public methods:
```python
def plot(self, ax: Optional[Any] = None, ...) -> Any:
```

### 8. Potential Numerical Instability

**File**: `diff_diff/honest_did.py:649-660`

The FLCI computation doesn't handle edge cases:

```python
def _compute_flci(lb: float, ub: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lb = lb - z * se
    ci_ub = ub + z * se
    return ci_lb, ci_ub
```

**Recommendation**: Add guards for edge cases:
```python
if se <= 0:
    raise ValueError("Standard error must be positive")
if not (0 < alpha < 1):
    raise ValueError("alpha must be between 0 and 1")
```

---

## Low Priority / Style Suggestions

### 9. Long Lines in Summary Output

**File**: `diff_diff/honest_did.py:225-276`

The summary method has many string literals exceeding 88 characters. While functional, this makes the output harder to maintain.

**Recommendation**: Use f-string formatting with variables for repeated widths:
```python
WIDTH = 70
DASHES = "-" * WIDTH
lines = [
    "=" * WIDTH,
    "Honest DiD Sensitivity Analysis Results".center(WIDTH),
    ...
]
```

### 10. Redundant `__repr__` Methods

**File**: `diff_diff/honest_did.py`

The Delta classes (`DeltaSD`, `DeltaRM`, `DeltaSDRM`) have custom `__repr__` methods that produce output identical to what dataclass would generate.

**Recommendation**: Remove the custom `__repr__` methods and let dataclass handle it, or differentiate them meaningfully.

### 11. Consider Using `@cached_property` for Expensive Properties

**File**: `diff_diff/honest_did.py:186-205`

Properties like `identified_set_width` and `ci_width` could be cached:

```python
from functools import cached_property

@cached_property
def identified_set_width(self) -> float:
    """Width of the identified set."""
    return self.ub - self.lb
```

Note: This is a minor optimization since these computations are trivial.

---

## Documentation Issues

### 12. Typo in README Example

**File**: `README.md:1001`

```python
from diff_diff import HonestDiD, MultiPeriodDiD
```

This import should work, but the example later uses:
```python
from diff_diff import plot_sensitivity, plot_honest_event_study
```

**Recommendation**: Consolidate into a single import block for clarity.

### 13. Missing CallawaySantAnna Support Note

**File**: `diff_diff/honest_did.py:468-494`

The code attempts to support `CallawaySantAnnaResults` but the implementation appears incomplete:

```python
try:
    from diff_diff.staggered import CallawaySantAnnaResults
    if isinstance(results, CallawaySantAnnaResults):
        if results.event_study_effects is None:
            raise ValueError(...)
```

The TODO.md mentions this as a future extension, but the code structure suggests partial implementation.

**Recommendation**: Either complete the implementation or add a clearer "not yet supported" error message.

---

## Test Coverage Observations

### Positive Notes
- Good coverage of edge cases (single post-period, large M, M=0)
- Integration tests with real estimator
- Mock fixtures enable fast unit tests

### Suggestions
1. Add test for negative `alpha` validation
2. Add test for CallawaySantAnna integration (even if just to verify the error message)
3. Add parametrized tests for different `l_vec` configurations

---

## Performance Considerations

### Memory
The `SensitivityResults` stores full `bounds` and `robust_cis` lists. For very fine grids, consider lazy evaluation.

### Computation
Each `fit()` call reconstructs constraint matrices. For sensitivity analysis over many M values, consider caching the structural parts that don't depend on M.

---

## Security

No security concerns identified. The code doesn't execute user input or access external resources beyond scipy's optimization routines.

---

## Compatibility

### Dependencies
- Uses `scipy.optimize.linprog` with `method='highs'` which requires scipy >= 1.6.0
- Uses `scipy.stats.norm.ppf` (widely available)

**Recommendation**: Add scipy version requirement to setup.py/pyproject.toml if not already specified.

---

## Final Checklist

- [x] Code follows existing patterns in the codebase
- [x] Type hints present (mostly complete)
- [x] Docstrings present and informative
- [x] Tests cover main functionality
- [ ] All exports in `__init__.py` are valid (see issue #4)
- [x] Tutorial notebook is comprehensive
- [x] README documentation added
- [x] CLAUDE.md updated

---

## Summary of Action Items

| Priority | Issue | Action |
|----------|-------|--------|
| **Critical** | #1 Pre-period effects | Verify constraint matrix construction |
| **High** | #4 Invalid export | Fix `sensitivity_plot` export |
| **High** | #2 Breakdown search | Consider optimization |
| **Medium** | #6 Hardcoded solver | Make configurable or document requirement |
| **Medium** | #8 Numerical edge cases | Add validation |
| **Low** | #3, #5, #7, #9-11 | Style/consistency improvements |

---

**Recommendation**: Merge after addressing issues #1 and #4. Other issues can be addressed in follow-up PRs.
