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
| `staggered.py` | 2301 | Consider splitting to `staggered_bootstrap.py` |
| `prep.py` | 1993 | Grew with DGP functions; consider splitting |
| `trop.py` | 1703 | Monitor size |
| `visualization.py` | 1627 | Acceptable but growing |
| `honest_did.py` | 1493 | Acceptable |
| `utils.py` | 1481 | Acceptable |
| `power.py` | 1350 | Acceptable |
| `triple_diff.py` | 1291 | Acceptable |
| `sun_abraham.py` | 1176 | Acceptable |
| `pretrends.py` | 1160 | Acceptable |
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

### Type Annotations

Pyright reports 282 type errors. Most are false positives from numpy/pandas type stubs.

| Category | Count | Notes |
|----------|-------|-------|
| reportArgumentType | 94 | numpy/pandas stub mismatches |
| reportAttributeAccessIssue | 89 | Union types (results classes) |
| reportReturnType | 21 | Return type mismatches |
| reportOperatorIssue | 16 | Operators on incompatible types |
| Others | 62 | Various minor issues |

**Genuine issues to fix (low priority):**
- [ ] Optional handling in `estimators.py:291,297,308` - None checks needed
- [ ] Union type narrowing in `visualization.py:325-345` - results classes
- [ ] numpy floating conversion in `diagnostics.py:669-673`

**Note:** Most errors are false positives from imprecise type stubs. Mypy config in pyproject.toml already handles these via `disable_error_code`.

### Rust Code Quality

Clippy reports 6 warnings (no errors):

- [ ] `rust/src/linalg.rs:32` - Define type alias for complex return type
- [ ] `rust/src/trop.rs` - Refactor 3 functions with >7 arguments to use param structs
  - `loocv_score_for_params` (12 args)
  - `compute_weight_matrix` (9 args)
  - `estimate_model` (9 args)

---

## Deprecated Code

Deprecated parameters still present for backward compatibility:

- [ ] `bootstrap_weight_type` in `CallawaySantAnna` (`staggered.py:746,763-771`)
  - Deprecated in favor of `bootstrap_weights` parameter
  - Warning text says "removed in v2.0" - update to "v3.0" when ready
  - Also used in: README.md (2x), tutorial 02, test_staggered.py
  - Remove in next major version (v3.0)

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

