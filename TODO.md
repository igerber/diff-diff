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
| ~~Within-transformation logic~~ | ~~Multiple files~~ | ✅ Extracted to `utils.py` as `demean_by_group()` and `within_transform()` (v2.0.1) |
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

## Test Coverage

**Note**: 21 visualization tests are skipped when matplotlib unavailable—this is expected.

---

## Documentation Improvements

- [ ] Comparison of estimator outputs on same data
- [ ] Real-world data examples (currently synthetic only)

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

---

## Rust Backend Optimizations

Deferred from PR #58 code review (can be done post-merge):

- [x] **Matrix inversion efficiency** (`rust/src/linalg.rs`): ~~Use Cholesky factorization for symmetric positive-definite matrices instead of column-by-column solve~~ (completed in v2.0.1)
- [x] **Reduce bootstrap allocations** (`rust/src/bootstrap.rs`): ~~Currently uses `Vec<Vec<f64>>` → flatten → `Array2` which allocates twice.~~ Now allocates directly into pre-allocated buffer. (completed in v2.0.1)
- [ ] **Consider static BLAS linking** (`rust/Cargo.toml`): Currently requires system BLAS libraries. Consider `openblas-static` or `intel-mkl-static` features for easier distribution.

---

## Performance Optimizations

Potential future optimizations:

- [ ] JIT compilation for bootstrap loops (numba)
- [ ] Parallel bootstrap iterations
- [ ] Sparse matrix handling for large fixed effects

