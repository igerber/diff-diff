# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Priority Items for 1.0.1

### Linter/Type Errors (Blocking)

| Issue | Location | Severity |
|-------|----------|----------|
| Unused import `Union` | `power.py:25` | ruff F401 |
| Unsorted imports | `staggered.py:8` | ruff I001 |
| 10 mypy errors - Optional type handling | `staggered.py:843-1631` | mypy operator/index |

### Quick Wins

- [ ] Fix ruff errors (2 auto-fixable)
- [ ] Fix mypy errors in staggered.py (Optional dict access needs guards)
- [ ] Remove duplicate `_get_significance_stars()` from `diagnostics.py:24-34` (already in `results.py:183-193`)

---

## Known Limitations

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:1068-1074` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:532-554` | Low | Rarely needed |
| SyntheticDiD bootstrap can fail silently | `estimators.py:1580-1654` | Medium | Needs error handling |
| Diagnostics module error handling | `diagnostics.py:782-885` | Medium | Improve robustness |

---

## Code Quality Issues

### Bare Exception Handling

Replace broad `except Exception` with specific exceptions:

| Location | Recommendation |
|----------|----------------|
| `diagnostics.py:636` | Catch `ValueError`, `LinAlgError` |
| `diagnostics.py:747` | Catch `ValueError`, `LinAlgError` |
| `honest_did.py:807` | Catch specific optimization errors |
| `honest_did.py:821` | Catch specific optimization errors |

### Code Duplication

| Duplicate Code | Locations | Action |
|---------------|-----------|--------|
| `_get_significance_stars()` | `results.py:183`, `diagnostics.py:24` | Remove from diagnostics.py |
| Wild bootstrap inference block | `estimators.py:278-296`, `estimators.py:725-748` | Extract to shared method |
| Within-transformation logic | `estimators.py:217-232`, `estimators.py:787-833`, `bacon.py:567-642` | Extract to utils.py |
| Linear regression helper | `staggered.py:205-240`, `estimators.py:366-408` | Consider consolidation |

### API Inconsistencies

**Bootstrap parameter naming:**
| Estimator | Parameter | Should be |
|-----------|-----------|-----------|
| DifferenceInDifferences | `bootstrap_weights` | Keep |
| CallawaySantAnna | `bootstrap_weight_type` | Rename to `bootstrap_weights` |
| TwoWayFixedEffects | `bootstrap_weights` | Keep |

**Cluster variable defaults:**
- `TwoWayFixedEffects` silently defaults cluster to `unit` at runtime (`estimators.py:689`)
- Behavior should be documented in docstring or made explicit in `__init__`

---

## Large Module Files

Current line counts (target: < 1000 lines per module):

| File | Lines | Status |
|------|-------|--------|
| `staggered.py` | 1822 | Consider splitting |
| `estimators.py` | 1812 | Consider splitting |
| `honest_did.py` | 1491 | Acceptable |
| `utils.py` | 1350 | Acceptable |
| `power.py` | 1350 | Acceptable |
| `prep.py` | 1338 | Acceptable |
| `visualization.py` | 1388 | Acceptable |
| `bacon.py` | 1027 | OK |

**Potential splits:**
- `estimators.py` → `twfe.py`, `synthetic_did.py` (keep base classes in estimators.py)
- `staggered.py` → `staggered_bootstrap.py` (move bootstrap logic)

---

## Standard Error Consistency

Different estimators compute SEs differently. Consider unified interface.

| Estimator | Default SE Type |
|-----------|-----------------|
| DifferenceInDifferences | HC1 or cluster-robust |
| TwoWayFixedEffects | Always cluster-robust (unit level) |
| CallawaySantAnna | Simple difference-in-means SE |
| SyntheticDiD | Bootstrap or placebo-based |

**Action**: Audit and document SE computation across estimators. Consider adding `se_type` parameter for consistency.

---

## Test Coverage Gaps

Edge cases needing tests:

- [ ] Very few clusters (< 5) with wild bootstrap
- [ ] Unbalanced panels with missing periods
- [ ] Single treated unit scenarios
- [ ] Perfect collinearity detection
- [ ] CallawaySantAnna with single cohort
- [ ] SyntheticDiD with insufficient pre-periods

**Note**: 21 visualization tests are skipped when matplotlib unavailable - this is expected.

---

## Documentation Improvements

- [ ] Troubleshooting section for common errors
- [ ] Comparison of estimator outputs on same data
- [ ] Real-world data examples (currently synthetic only)
- [ ] Performance benchmarks vs. R packages

---

## CallawaySantAnna Bootstrap Improvements

Deferred improvements from code review (PR #32):

- [ ] Refactor `_run_multiplier_bootstrap` into smaller helper methods for maintainability
- [ ] Consider aligning p-value computation with R `did` package (symmetric percentile method)

---

## Honest DiD Future Improvements

Post-1.0 enhancements for `honest_did.py`:

- [ ] Improved C-LF implementation with direct optimization instead of grid search
- [ ] Support for CallawaySantAnnaResults (currently only MultiPeriodDiDResults)
- [ ] Event-study-specific bounds for each post-period
- [ ] Hybrid inference methods
- [ ] Simulation-based power analysis for honest bounds

---

## Performance

No major performance issues identified. Potential future optimizations:

- JIT compilation for bootstrap loops (numba)
- Parallel bootstrap iterations
- Sparse matrix handling for large fixed effects

---

## Type Hints

Missing type hints in internal functions:
- `utils.py:593` - `compute_trend()` nested function
- `staggered.py:173, 180` - Nested functions in `_logistic_regression()`
- `prep.py:604` - `format_label()` nested function
