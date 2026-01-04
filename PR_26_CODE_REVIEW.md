# Code Review: PR #26 - Implement Honest DiD Sensitivity Analysis

**Reviewer**: Claude
**Date**: 2026-01-04 (Updated)
**PR Branch**: pr-26
**Commits**: 5 (15f6592, 66c37a8, 93dde89, 2426c24, e40d6b4)

## Summary

This PR implements the Honest DiD sensitivity analysis framework (Rambachan & Roth 2023), which is a major feature addition that addresses a key credibility gap in the library. The implementation includes:

- New `honest_did.py` module with `HonestDiD` class and supporting types
- Visualization functions for sensitivity analysis
- Comprehensive test suite (699 lines)
- Tutorial notebook and documentation updates

**Overall Assessment**: ✅ **Approve** - This is a well-structured implementation of an important feature. The code is well-documented and follows existing patterns. Initial review concerns have been addressed or verified as non-issues.

---

## Verified as Non-Issues

### ~~#1 Pre-period Effects Extraction~~ ✅ Correct by Design

The LP optimization correctly works over all periods (pre + post), but only post-period components contribute to the objective function. This is the correct approach for handling smoothness constraints that link pre and post periods.

**Revision**: Added clarifying comment in `_solve_bounds_lp`:
```python
# Note: The optimization is over delta for ALL periods (pre + post), but
# only the post-period components contribute to the objective function.
# This correctly handles smoothness constraints that link pre and post periods.
```

### ~~#4 sensitivity_plot Export~~ ✅ Valid

The function exists at `honest_did.py:1437`. Initial review missed this.

---

## Addressed in Revision (e40d6b4)

### #3 Significance Logic ✅ Fixed

Updated docstring to clarify partial identification semantics:
```python
@property
def significance_stars(self) -> str:
    """
    Return significance indicator if robust CI excludes zero.

    Note: Unlike point estimation, partial identification does not yield
    a single p-value. This returns "*" if the robust CI excludes zero
    at the specified alpha level, indicating the effect is robust to
    the assumed violations of parallel trends.
    """
    return "*" if self.is_significant else ""
```

### #5 Inconsistent Error Messages ✅ Fixed

All error messages now include parameter values:
```python
raise ValueError(f"M must be non-negative, got M={self.M}")
raise ValueError(f"Mbar must be non-negative, got Mbar={self.Mbar}")
raise ValueError(f"alpha must be between 0 and 1, got alpha={self.alpha}")
```

### #6 Hardcoded LP Solver ✅ Fixed

Added `lp_method` parameter with documentation:
```python
def _solve_bounds_lp(..., lp_method: str = 'highs') -> Tuple[float, float]:
    """
    ...
    lp_method : str
        LP solver method for scipy.optimize.linprog. Default 'highs' requires
        scipy >= 1.6.0. Alternatives: 'interior-point', 'revised simplex'.
    """
```

### #8 Numerical Edge Cases ✅ Fixed

Added validation in `_compute_flci`:
```python
if se <= 0:
    raise ValueError(f"Standard error must be positive, got se={se}")
if not (0 < alpha < 1):
    raise ValueError(f"alpha must be between 0 and 1, got alpha={alpha}")
```

### #13 CallawaySantAnna Error Message ✅ Improved

Better guidance for users:
```python
raise ValueError(
    "CallawaySantAnnaResults must have event_study_effects for HonestDiD. "
    "Re-run CallawaySantAnna.fit() with aggregate='event_study' to compute "
    "event study effects."
)
```

---

## Remaining Suggestions (Low Priority - Future PRs)

### #2 Breakdown Value Search Efficiency

The breakdown search re-fits for every grid point. Consider caching intermediate computations for performance with fine grids.

### #7 Missing Return Type Hints

Some methods have incomplete type hints (e.g., `plot` method). Minor style issue.

### #9-11 Style Suggestions

- Long lines in summary output
- Redundant `__repr__` methods in Delta classes
- Could use `@cached_property` for trivial computed properties

### #12 README Import Consolidation

Multiple import blocks in examples could be consolidated for clarity.

---

## Test Coverage Observations

### Positive Notes
- Good coverage of edge cases (single post-period, large M, M=0)
- Integration tests with real estimator
- Mock fixtures enable fast unit tests

### Suggestions for Future
1. Add test for negative `alpha` validation (now that validation exists)
2. Add test for CallawaySantAnna integration error message
3. Add parametrized tests for different `l_vec` configurations

---

## Final Checklist

- [x] Code follows existing patterns in the codebase
- [x] Type hints present (mostly complete)
- [x] Docstrings present and informative
- [x] Tests cover main functionality
- [x] All exports in `__init__.py` are valid
- [x] Tutorial notebook is comprehensive
- [x] README documentation added
- [x] CLAUDE.md updated
- [x] Error messages include parameter values
- [x] Edge case validation added

---

## Summary

| Original Issue | Status | Resolution |
|----------------|--------|------------|
| #1 Pre-period effects | ✅ Non-issue | Verified correct by design |
| #2 Breakdown search | ⏳ Future | Low priority optimization |
| #3 Significance logic | ✅ Fixed | Docstring clarified |
| #4 Invalid export | ✅ Non-issue | Function exists at line 1437 |
| #5 Error messages | ✅ Fixed | Now include parameter values |
| #6 Hardcoded solver | ✅ Fixed | `lp_method` parameter added |
| #7 Return type hints | ⏳ Future | Minor style issue |
| #8 Numerical edge cases | ✅ Fixed | Validation added |
| #9-11 Style | ⏳ Future | Low priority |
| #12 README imports | ⏳ Future | Minor docs cleanup |
| #13 CS&A error message | ✅ Fixed | Clearer guidance added |

---

**Recommendation**: ✅ **Approve and merge** - All critical and high-priority items have been addressed or verified as non-issues. Remaining suggestions are low-priority style improvements suitable for future PRs.
