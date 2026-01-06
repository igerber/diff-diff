# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Known Limitations

Current limitations that may affect users:

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:1068-1074` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:532-554` | Low | Rarely needed |

---

## Code Quality

### Code Duplication

Consolidation opportunities for cleaner maintenance:

| Duplicate Code | Locations | Notes |
|---------------|-----------|-------|
| Within-transformation logic | `estimators.py:217-232`, `estimators.py:787-833`, `bacon.py:567-642` | Extract to utils.py |
| Linear regression helper | `staggered.py:205-240`, `estimators.py:366-408` | Consider consolidation |

### Large Module Files

Target: < 1000 lines per module for maintainability.

| File | Lines | Action |
|------|-------|--------|
| `staggered.py` | 1822 | Consider splitting to `staggered_bootstrap.py` |
| `honest_did.py` | 1491 | Acceptable |
| `visualization.py` | 1388 | Acceptable |
| `utils.py` | 1350 | Acceptable |
| `power.py` | 1350 | Acceptable |
| `prep.py` | 1338 | Acceptable |
| `bacon.py` | 1027 | OK |

### Standard Error Consistency

Different estimators compute SEs differently. Consider unified interface.

| Estimator | Default SE Type |
|-----------|-----------------|
| DifferenceInDifferences | HC1 or cluster-robust |
| TwoWayFixedEffects | Always cluster-robust (unit level) |
| CallawaySantAnna | Simple difference-in-means SE |
| SyntheticDiD | Bootstrap or placebo-based |

**Action**: Consider adding `se_type` parameter for consistency across estimators.

---

## Test Coverage Gaps

Edge cases needing tests:

- [ ] Very few clusters (< 5) with wild bootstrap
- [ ] Unbalanced panels with missing periods
- [ ] Single treated unit scenarios
- [ ] Perfect collinearity detection
- [ ] CallawaySantAnna with single cohort
- [ ] SyntheticDiD with insufficient pre-periods

**Note**: 21 visualization tests are skipped when matplotlib unavailable—this is expected.

---

## Documentation Improvements

- [ ] Troubleshooting section for common errors
- [ ] Comparison of estimator outputs on same data
- [ ] Real-world data examples (currently synthetic only)
- [ ] Performance benchmarks vs. R packages

---

## Honest DiD Improvements

Enhancements for `honest_did.py`:

- [ ] Improved C-LF implementation with direct optimization instead of grid search
- [ ] Support for CallawaySantAnnaResults (currently only MultiPeriodDiDResults)
- [ ] Event-study-specific bounds for each post-period
- [ ] Hybrid inference methods
- [ ] Simulation-based power analysis for honest bounds

---

## CallawaySantAnna Bootstrap Improvements

From code review (PR #32):

- [ ] Refactor `_run_multiplier_bootstrap` into smaller helper methods
- [ ] Consider aligning p-value computation with R `did` package (symmetric percentile method)

---

## Performance Optimizations

No major performance issues identified. Potential future optimizations:

- [ ] JIT compilation for bootstrap loops (numba)
- [ ] Parallel bootstrap iterations
- [ ] Sparse matrix handling for large fixed effects

---

## Type Hints

All previously identified missing type hints have been addressed in v1.1.1.
