# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Known Limitations

Current limitations that may affect users:

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:778-784` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:567-588` | Low | Rarely needed |

## Code Quality

### Large Module Files

Target: < 1000 lines per module for maintainability. Updated 2026-03-29.

| File | Lines | Action |
|------|-------|--------|
| `power.py` | 2588 | Consider splitting (power analysis + MDE + sample size) |
| `linalg.py` | 2289 | Monitor — unified backend, splitting would hurt cohesion |
| `staggered.py` | 2275 | Monitor — grew with survey support |
| `imputation.py` | 2009 | Monitor |
| `triple_diff.py` | 1921 | Monitor |
| `utils.py` | 1902 | Monitor |
| `two_stage.py` | 1708 | Monitor |
| `survey.py` | 1646 | Monitor — grew with Phase 6 features |
| `continuous_did.py` | 1626 | Monitor |
| `honest_did.py` | 1511 | Acceptable |
| `sun_abraham.py` | 1540 | Acceptable |
| `estimators.py` | 1357 | Acceptable |
| `trop_local.py` | 1261 | Acceptable |
| `trop_global.py` | 1251 | Acceptable |
| `prep.py` | 1225 | Acceptable |
| `pretrends.py` | 1105 | Acceptable |
| `trop.py` | 981 | Split done — trop_global.py + trop_local.py |
| `visualization/` | 4172 | Subpackage (split across 7 files) — OK |

---

### Tech Debt from Code Reviews

Deferred items from PR reviews that were not addressed before merge.

#### Methodology/Correctness

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| ImputationDiD dense `(A0'A0).toarray()` scales O((U+T+K)^2), OOM risk on large panels | `imputation.py` | #141 | Medium (deferred — only triggers when sparse solver fails) |
| Multi-absorb weighted demeaning needs iterative alternating projections for N > 1 absorbed FE with survey weights; unweighted multi-absorb also uses single-pass (pre-existing, exact only for balanced panels) | `estimators.py` | #218 | Medium |
| Replicate-weight survey df — **Resolved**. `df_survey = rank(replicate_weights) - 1` matching R's `survey::degf()`. For IF paths, `n_valid - 1` when dropped replicates reduce effective count. | `survey.py` | #238 | Resolved |
| CallawaySantAnna survey: strata/PSU/FPC — **Resolved**. Aggregated SEs (overall, event study, group) use `compute_survey_if_variance()`. Bootstrap uses PSU-level multiplier weights. | `staggered.py` | #237 | Resolved |
| CallawaySantAnna survey + covariates + IPW/DR — **Resolved**. DRDID panel nuisance IF corrections (PS + OR) implemented for both survey and non-survey DR paths (Phase 7a). IPW path unblocked. | `staggered.py` | #233 | Resolved |
| SyntheticDiD/TROP survey: strata/PSU/FPC — **Resolved**. Rao-Wu rescaled bootstrap implemented for both. TROP uses cross-classified pseudo-strata. Rust TROP remains pweight-only (Python fallback for full design). | `synthetic_did.py`, `trop.py` | — | Resolved |
| EfficientDiD hausman_pretest() clustered covariance stale `n_cl` — **Resolved**. Recompute `n_cl` and remap indices after `row_finite` filtering via `np.unique(return_inverse=True)`. | `efficient_did.py` | #230 | Resolved |
| EfficientDiD `control_group="last_cohort"` trims at `last_g - anticipation` but REGISTRY says `t >= last_g`. With `anticipation=0` (default) these are identical. With `anticipation>0`, code is arguably more conservative (excludes anticipation-contaminated periods). Either align REGISTRY with code or change code to `t < last_g` — needs design decision. | `efficient_did.py` | #230 | Low |
| TripleDifference power: `generate_ddd_data` is a fixed 2×2×2 cross-sectional DGP — no multi-period or unbalanced-group support. Add a `generate_ddd_panel_data` for panel DDD power analysis. | `prep_dgp.py`, `power.py` | #208 | Low |
| ContinuousDiD event-study aggregation anticipation filter — **Resolved**. `_aggregate_event_study()` now filters `e < -anticipation` when `anticipation > 0`, matching CallawaySantAnna behavior. Bootstrap paths also filtered. | `continuous_did.py` | #226 | Resolved |
| Survey design resolution/collapse patterns are inconsistent across panel estimators — ContinuousDiD rebuilds unit-level design in SE code, EfficientDiD builds once in fit(), StackedDiD re-resolves on stacked data; extract shared helpers for panel-to-unit collapse, post-filter re-resolution, and metadata recomputation | `continuous_did.py`, `efficient_did.py`, `stacked_did.py` | #226 | Low |
| Survey metadata formatting dedup — **Resolved**. Extracted `_format_survey_block()` helper in `results.py`, replaced 13 occurrences across 11 files. | `results.py` + 10 results files | — | Resolved |
| TROP: `fit()` and `_fit_global()` share ~150 lines of near-identical data setup (panel pivoting, absorbing-state validation, first-treatment detection, effective rank, NaN warnings). Both bootstrap methods also duplicate the stratified resampling loop. Extract shared helpers to eliminate cross-file sync risk. | `trop.py`, `trop_global.py`, `trop_local.py` | — | Low |
| StaggeredTripleDifference R cross-validation: CSV fixtures not committed (gitignored); tests skip without local R + triplediff. Commit fixtures or generate deterministically. | `tests/test_methodology_staggered_triple_diff.py` | #245 | Medium |
| StaggeredTripleDifference R parity: benchmark only tests no-covariate path (xformla=~1). Add covariate-adjusted scenarios and aggregation SE parity assertions. | `benchmarks/R/benchmark_staggered_triplediff.R` | #245 | Medium |
| StaggeredTripleDifference: per-cohort group-effect SEs include WIF (conservative vs R's wif=NULL). Documented in REGISTRY. Could override mixin for exact R match. | `staggered_triple_diff.py` | #245 | Low |

#### Performance

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| ImputationDiD event-study SEs recompute full conservative variance per horizon (should cache A0/A1 factorization) | `imputation.py` | #141 | Low |
| Rust faer SVD ndarray-to-faer conversion overhead (minimal vs SVD cost) | `rust/src/linalg.rs:67` | #115 | Low |

#### Testing/Docs

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| R comparison tests spawn separate `Rscript` per test (slow CI) | `tests/test_methodology_twfe.py:294` | #139 | Low |
| CS R helpers hard-code `xformla = ~ 1`; no covariate-adjusted R benchmark for IRLS path | `tests/test_methodology_callaway.py` | #202 | Low |
| ~376 `duplicate object description` Sphinx warnings — restructure `docs/api/*.rst` to avoid duplicate `:members:` + `autosummary` | `docs/api/*.rst` | — | Low |
| Doc-snippet smoke tests only cover `.rst` files; `.txt` AI guides outside CI validation | `tests/test_doc_snippets.py` | #239 | Low |

---

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

Mypy reports 0 errors. All mixin `attr-defined` errors resolved via
`TYPE_CHECKING`-guarded method stubs in bootstrap mixin classes.

## Deprecated Code

Deprecated parameters still present for backward compatibility:

- `bootstrap_weight_type` in `CallawaySantAnna` (`staggered.py`)
  - Deprecated in favor of `bootstrap_weights` parameter
  - Remove in next major version (v3.0)

---

## Test Coverage

**Note**: 21 visualization tests are skipped when matplotlib unavailable—this is expected.

---

## Honest DiD Improvements

Enhancements for `honest_did.py`:

- [ ] Improved C-LF implementation with direct optimization instead of grid search
  (current implementation uses simplified FLCI approach with estimation uncertainty
  adjustment; see `honest_did.py:947`)
- [x] Support for CallawaySantAnnaResults (implemented in `honest_did.py:612-653`;
  requires `aggregate='event_study'` when calling `CallawaySantAnna.fit()`)
- [ ] Event-study-specific bounds for each post-period
- [ ] Hybrid inference methods
- [ ] Simulation-based power analysis for honest bounds

---

## CallawaySantAnna Bootstrap Improvements

- [ ] Consider aligning p-value computation with R `did` package (symmetric percentile method)

---

## RuntimeWarnings in Linear Algebra Operations

### Apple Silicon M4 BLAS Bug (numpy < 2.3)

Spurious RuntimeWarnings ("divide by zero", "overflow", "invalid value") are emitted by `np.matmul`/`@` on Apple Silicon M4 + macOS Sequoia with numpy < 2.3. The warnings appear for matrices with ≥260 rows but **do not affect result correctness** — coefficients and fitted values are valid (no NaN/Inf), and the design matrices are full rank.

**Root cause**: Apple's BLAS SME (Scalable Matrix Extension) kernels corrupt the floating-point status register, causing spurious FPE signals. Tracked in [numpy#28687](https://github.com/numpy/numpy/issues/28687) and [numpy#29820](https://github.com/numpy/numpy/issues/29820). Fixed in numpy ≥ 2.3 via [PR #29223](https://github.com/numpy/numpy/pull/29223).

**Not reproducible** on M3, Intel, or Linux.

- [ ] `linalg.py:162` - Warnings in fitted value computation (`X @ coefficients`)
  - Caused by M4 BLAS bug, not extreme coefficient values
  - Seen in test_prep.py during treatment effect recovery tests (n > 260)
- [ ] `triple_diff.py:307,323` - Warnings in propensity score computation
  - Occurs in IPW and DR estimation methods with covariates
  - Related to logistic regression overflow in edge cases (separate from BLAS bug)

- **Long-term:** Revert to `@` operator when numpy ≥ 2.3 becomes the minimum supported version.

---

## Feature Gaps (from R `did` package comparison)

Features in R's `did` package that block porting additional tests:

| Feature | R tests blocked | Priority | Status |
|---------|----------------|----------|--------|
| Repeated cross-sections (`panel=FALSE`) | ~7 tests in test-att_gt.R + test-user_bug_fixes.R | High | **Resolved** — Phase 7b: `panel=False` on CallawaySantAnna |
| Sampling/population weights | 7 tests incl. all JEL replication | Medium | **Resolved** (Phases 1-6 + 7a: CS IPW/DR + covariates + survey) |
| Calendar time aggregation | 1 test in test-att_gt.R | Low | |

---

## Performance Optimizations

Potential future optimizations:

- [ ] JIT compilation for bootstrap loops (numba)
- [ ] Sparse matrix handling for large fixed effects

### QR+SVD Redundancy in Rank Detection

**Background**: The current `solve_ols()` implementation performs both QR (for rank detection) and SVD (for solving) decompositions on rank-deficient matrices. This is technically redundant since SVD can determine rank directly.

**Current approach** (R-style, chosen for robustness):
1. QR with pivoting for rank detection (`_detect_rank_deficiency()`)
2. scipy's `lstsq` with 'gelsd' driver (SVD-based) for solving

**Why we use QR for rank detection**:
- QR with pivoting provides the canonical ordering of linearly dependent columns
- R's `lm()` uses this approach for consistent dropped-column reporting
- Ensures consistent column dropping across runs (SVD column selection can vary)

**Potential optimization** (future work):
- Skip QR when `rank_deficient_action="silent"` since we don't need column names
- Use SVD rank directly in the Rust backend (already implemented)
- Add `skip_rank_check` parameter for hot paths where matrix is known to be full-rank (implemented in v2.2.0)

**Priority**: Low - the QR overhead is minimal compared to SVD solve, and correctness is more important than micro-optimization.

### Incomplete `check_finite` Bypass

**Background**: The `solve_ols()` function accepts a `check_finite=False` parameter intended to skip NaN/Inf validation for performance in hot paths where data is known to be clean.

**Current limitation**: When `check_finite=False`, our explicit validation is skipped, but scipy's internal QR decomposition in `_detect_rank_deficiency()` still validates finite values. This means callers cannot fully bypass all finite checks.

**Impact**: Minimal - the scipy check is fast and only affects edge cases where users explicitly pass `check_finite=False` with non-finite data (which would be a bug in their code anyway).

**Potential fix** (future work):
- Pass `check_finite=False` through to scipy's QR call (requires scipy >= 1.9.0)
- Or skip `_detect_rank_deficiency()` entirely when `check_finite=False` and `_skip_rank_check=True`

**Priority**: Low - this is an edge case optimization that doesn't affect correctness.

