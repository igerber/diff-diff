# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Known Limitations

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:944-951` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:532-554` | Low | Rarely needed |
| SyntheticDiD bootstrap can fail silently | `estimators.py:1580-1654` | Medium | Needs error handling |
| Diagnostics module error handling | `diagnostics.py:782-885` | Medium | Improve robustness |

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

---

## Documentation Improvements

- [ ] Troubleshooting section for common errors
- [ ] Comparison of estimator outputs on same data
- [ ] Real-world data examples (currently synthetic only)
- [ ] Performance benchmarks vs. R packages

---

## Code Quality

### Refactoring Candidates

- `estimators.py` is large (~1600 lines). Consider splitting TWFE and SyntheticDiD into separate modules.
- Duplicate code in fixed effects handling between `DifferenceInDifferences` and `TwoWayFixedEffects`.

### Type Hints

- Most modules have type hints, but some internal functions lack them
- Consider stricter mypy settings

### Dependencies

- Core: numpy, pandas, scipy only (no statsmodels) - keep it this way
- Optional: matplotlib for visualization

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
