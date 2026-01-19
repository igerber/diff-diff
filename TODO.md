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

### Large Module Files

Target: < 1000 lines per module for maintainability.

| File | Lines | Action |
|------|-------|--------|
| `prep.py` | 1993 | Grew with DGP functions; consider splitting |
| `staggered.py` | 1822 | Consider splitting to `staggered_bootstrap.py` |
| `honest_did.py` | 1491 | Acceptable |
| `visualization.py` | 1388 | Acceptable |
| `utils.py` | 1350 | Acceptable |
| `power.py` | 1350 | Acceptable |
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

## Test Coverage

**Note**: 21 visualization tests are skipped when matplotlib unavailable—this is expected.

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

- [ ] Consider aligning p-value computation with R `did` package (symmetric percentile method)
- [ ] Investigate RuntimeWarnings in influence function aggregation (`staggered.py:1722`, `staggered.py:1999-2018`)
  - Warnings: "divide by zero", "overflow", "invalid value" in matmul operations
  - Occurs during bootstrap SE computation with small sample sizes or edge cases
  - Does not affect correctness (results are still valid), but should be suppressed or handled gracefully

---

## RuntimeWarnings in Linear Algebra Operations

Pre-existing RuntimeWarnings in matrix operations that should be investigated:

- [ ] `linalg.py:162` - "divide by zero", "overflow", "invalid value" in fitted value computation
  - Occurs during `X @ coefficients` when coefficients contain extreme values
  - Seen in test_prep.py during treatment effect recovery tests
- [ ] `triple_diff.py:307,323` - Similar warnings in propensity score computation
  - Occurs in IPW and DR estimation methods with covariates
  - Related to logistic regression overflow in edge cases

**Note**: These warnings do not affect correctness of results but should be handled gracefully (e.g., with `np.errstate` context managers or input validation).

---

## Performance Optimizations

Potential future optimizations:

- [ ] JIT compilation for bootstrap loops (numba)
- [ ] Sparse matrix handling for large fixed effects

