# Performance Improvement Plan

This document outlines the strategy for improving diff-diff's performance on large datasets, particularly for BasicDiD/TWFE and CallawaySantAnna estimators.

## Problem Statement

Benchmark comparisons show that while diff-diff is competitive or faster than R for small datasets, performance degrades significantly at scale:

| Scale | BasicDiD Python | R (fixest) | Ratio |
|-------|-----------------|------------|-------|
| Small (<1K obs) | 0.003s | 0.041s | Python 16x faster |
| 5K (40-200K obs) | 0.180s | 0.046s | R 4x faster |
| 10K (100-500K obs) | 0.835s | 0.049s | R 17x faster |

| Scale | CallawaySantAnna Python | R (did) | Ratio |
|-------|-------------------------|---------|-------|
| Small | 0.048s | 0.077s | Python 1.6x faster |
| 5K | 0.793s | 0.382s | R 2x faster |
| 10K | 2.234s | 0.816s | R 2.7x faster |

Note: SyntheticDiD is already 37-1600x faster than R's synthdid package.

## Root Cause Analysis

### 1. OLS Solver (`estimators.py`)

Current implementation uses `np.linalg.lstsq` with default settings:
- General-purpose LAPACK driver (gelsd) rather than faster alternatives
- Preceded by expensive `matrix_rank()` check (O(min(n,k)^3))
- NumPy may not link to optimized BLAS

### 2. Cluster-Robust Standard Errors (`utils.py`)

Loop-based implementation:
```python
for cluster in unique_clusters:
    mask = cluster_ids == cluster  # O(n) per cluster
    ...
```
- O(n * n_clusters) complexity
- Creates boolean mask array for each cluster
- No vectorization or parallelization

### 3. Within-Transformation (`twfe.py`)

Multiple groupby operations:
```python
for var in variables:
    unit_means = data.groupby(unit)[var].transform("mean")
    time_means = data.groupby(time)[var].transform("mean")
    ...
```
- Multiple passes over data per variable
- No caching of groupby indexes
- Not using alternating projections algorithm

### 4. CallawaySantAnna Nested Loops (`staggered.py`)

```python
for g in treatment_groups:
    for t in valid_periods:
        att_gt = self._compute_att_gt(...)
```
- Repeated DataFrame indexing (`.set_index()`, `.loc[]`, `.isin()`) for each (g,t)
- No pre-computation of outcome changes
- Influence function dictionaries created per (g,t)

## Optimization Strategy

### Phase 1: Pure Python Optimizations (No New Dependencies)

Quick wins that improve performance without adding dependencies.

#### 1.1 Vectorized Cluster-Robust SE

Replace loop with vectorized groupby:
```python
scores = X * residuals[:, np.newaxis]
cluster_scores = pd.DataFrame(scores).groupby(cluster_ids).sum()
meat = cluster_scores.values.T @ cluster_scores.values
```

**Expected speedup:** 5-10x for SE computation

#### 1.2 scipy.linalg.lstsq with Optimized Driver

```python
from scipy.linalg import lstsq
coefficients = lstsq(X, y, lapack_driver='gelsy',
                     overwrite_a=True, overwrite_b=True,
                     check_finite=False)[0]
```

**Expected speedup:** 1.2-1.5x for OLS

#### 1.3 Cache Groupby Indexes

Create groupby objects once and reuse:
```python
unit_grouper = data.groupby(unit, sort=False)
time_grouper = data.groupby(time, sort=False)
```

**Expected speedup:** 1.5-2x for demeaning

#### 1.4 Pre-compute CallawaySantAnna Data Structures

Pivot to wide format once, pre-compute all period changes:
```python
outcome_wide = data.pivot(index=unit, columns=time, values=outcome)
changes = {(t0, t1): outcome_wide[t1] - outcome_wide[t0] for ...}
```

**Expected speedup:** 3-5x for CallawaySantAnna

### Phase 2: Rust Backend (PyO3)

Implement performance-critical components in Rust for maximum speed.

#### Why Rust over C++

- **Memory safety** - No segfaults or buffer overflows
- **Modern tooling** - Cargo + maturin for easy wheel building
- **Zero-copy NumPy interop** - rust-numpy crate
- **Parallelism** - rayon makes parallel iteration trivial
- **Proven approach** - Used by polars, pyfixest, cryptography, orjson

#### Architecture

```
diff_diff/
├── estimators.py          # Python API (unchanged interface)
├── _rust_backend/         # Optional Rust acceleration
│   └── ...                # Compiled .so/.pyd
└── _fallback.py           # Pure Python fallback

src/                       # Rust source (Cargo workspace)
├── lib.rs
├── ols.rs                 # Fast OLS with cluster SE
├── demeaning.rs           # Alternating projections
├── bootstrap.rs           # Parallel bootstrap
└── staggered.rs           # ATT(g,t) computation
```

#### Graceful Degradation

```python
try:
    from diff_diff._rust_backend import solve_ols_clustered
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

def _fit_ols(self, X, y, cluster_ids=None):
    if _HAS_RUST and self.backend == 'rust':
        return solve_ols_clustered(X, y, cluster_ids)
    else:
        # Existing NumPy implementation
        ...
```

#### Components to Implement in Rust

| Component | Current Bottleneck | Rust Benefit |
|-----------|-------------------|--------------|
| Cluster-robust SE | O(n * clusters) loop | rayon parallel iteration |
| Within-transformation | Multiple groupby passes | Single-pass with hash tables |
| OLS solving | NumPy lstsq overhead | faer or direct LAPACK |
| Bootstrap resampling | Sequential iterations | Embarrassingly parallel |
| ATT(g,t) computation | Repeated DataFrame indexing | Pre-indexed sparse structures |

#### Key Rust Dependencies

- `pyo3` - Python bindings
- `rust-numpy` - Zero-copy NumPy array access
- `ndarray` - N-dimensional arrays
- `faer` - Pure Rust linear algebra (competitive with Intel MKL)
- `rayon` - Data parallelism

#### Distribution

Using maturin + GitHub Actions to build wheels for:
- Linux (manylinux2014, x86_64 and aarch64)
- macOS (x86_64 and ARM64)
- Windows (x86_64)

Use abi3 stable ABI for Python version-independent wheels.

## Implementation Roadmap

| Phase | Scope | Effort | Expected Speedup |
|-------|-------|--------|------------------|
| 1.1 | Vectorize cluster SE | 1-2 days | 5-10x (SE only) |
| 1.2 | scipy lstsq optimization | 1 day | 1.2-1.5x (OLS) |
| 1.3 | Cache groupby indexes | 1 day | 1.5-2x (demeaning) |
| 1.4 | Pre-compute CS structures | 2-3 days | 3-5x (CS) |
| 2.1 | Rust cluster SE | 1-2 weeks | 10-50x (SE) |
| 2.2 | Rust parallel bootstrap | 1 week | 5-20x (bootstrap) |
| 2.3 | Rust demeaning | 2 weeks | 3-10x (TWFE) |
| 2.4 | Rust OLS solver | 2 weeks | Match R |
| 2.5 | Rust staggered ATT | 2-3 weeks | 5-10x (CS) |
| 2.6 | CI/CD wheel building | 1 week | N/A |

## Expected Outcomes

### After Phase 1 (Pure Python)

- BasicDiD @ 10K: 0.835s -> ~0.3-0.4s (2-3x improvement)
- CallawaySantAnna @ 10K: 2.2s -> ~0.6-0.8s (3x improvement)

### After Phase 2 (Rust Backend)

- BasicDiD @ 10K: Match or beat R's 0.049s
- CallawaySantAnna @ 10K: Match R's 0.816s
- Bootstrap inference: 10-20x faster

## References

- [PyO3 User Guide](https://pyo3.rs/)
- [rust-numpy](https://github.com/PyO3/rust-numpy)
- [maturin](https://github.com/PyO3/maturin)
- [faer - Fast linear algebra in Rust](https://github.com/sarah-ek/faer-rs)
- [Polars architecture](https://github.com/pola-rs/polars) - Example of Rust/Python hybrid
- [pyfixest](https://github.com/py-econometrics/pyfixest) - Similar approach for econometrics
- [fixest demeaning algorithm](https://rdrr.io/cran/fixest/man/demeaning_algo.html)
