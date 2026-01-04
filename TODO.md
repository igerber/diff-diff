# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Priority Items for 1.0.1

### Linter/Type Errors (Blocking) - COMPLETED

| Issue | Location | Status |
|-------|----------|--------|
| ~~Unused import `Union`~~ | `power.py:25` | Fixed |
| ~~Unsorted imports~~ | `staggered.py:8` | Fixed |
| ~~10 mypy errors - Optional type handling~~ | `staggered.py:843-1631` | Fixed |

### Quick Wins - COMPLETED

- [x] Fix ruff errors (2 auto-fixable)
- [x] Fix mypy errors in staggered.py (Optional dict access needs guards)
- [x] Remove duplicate `_get_significance_stars()` from `diagnostics.py` (now imports from `results.py`)

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

### Bare Exception Handling - COMPLETED

~~Replace broad `except Exception` with specific exceptions:~~

| Location | Status |
|----------|--------|
| ~~`diagnostics.py:624`~~ | Fixed - catches `ValueError`, `KeyError`, `LinAlgError` |
| ~~`diagnostics.py:735`~~ | Fixed - catches `ValueError`, `KeyError`, `LinAlgError` |
| ~~`honest_did.py:807`~~ | Fixed - catches `ValueError`, `TypeError` |
| ~~`honest_did.py:822`~~ | Fixed - catches `ValueError`, `TypeError` |

### Code Duplication

| Duplicate Code | Locations | Status |
|---------------|-----------|--------|
| ~~`_get_significance_stars()`~~ | `results.py:183`, ~~`diagnostics.py`~~ | Fixed in 1.0.1 |
| Wild bootstrap inference block | `estimators.py:278-296`, `estimators.py:725-748` | Future: extract to shared method |
| Within-transformation logic | `estimators.py:217-232`, `estimators.py:787-833`, `bacon.py:567-642` | Future: extract to utils.py |
| Linear regression helper | `staggered.py:205-240`, `estimators.py:366-408` | Future: consider consolidation |

### API Inconsistencies - PARTIALLY ADDRESSED

**Bootstrap parameter naming:**
| Estimator | Parameter | Status |
|-----------|-----------|--------|
| DifferenceInDifferences | `bootstrap_weights` | OK |
| CallawaySantAnna | `bootstrap_weights` | Fixed in 1.0.1 (deprecated `bootstrap_weight_type`) |
| TwoWayFixedEffects | `bootstrap_weights` | OK |

**Cluster variable defaults:**
- ~~`TwoWayFixedEffects` silently defaults cluster to `unit` at runtime~~ - Documented in 1.0.1

---

## Large Module Files

Current line counts (target: < 1000 lines per module):

| File | Lines | Status |
|------|-------|--------|
| `staggered.py` | 1822 | Consider splitting |
| `estimators.py` | ~975 | OK (refactored) |
| `twfe.py` | ~355 | OK (new) |
| `synthetic_did.py` | ~540 | OK (new) |
| `honest_did.py` | 1491 | Acceptable |
| `utils.py` | 1350 | Acceptable |
| `power.py` | 1350 | Acceptable |
| `prep.py` | 1338 | Acceptable |
| `visualization.py` | 1388 | Acceptable |
| `bacon.py` | 1027 | OK |

**Completed splits:**
- ~~`estimators.py` → `twfe.py`, `synthetic_did.py` (keep base classes in estimators.py)~~ - Done in 1.0.2

**Potential splits:**
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
