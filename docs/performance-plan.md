# Performance Improvement Plan

This document outlines the strategy for improving diff-diff's performance on large datasets, particularly for BasicDiD/TWFE and CallawaySantAnna estimators.

---

## Results Achieved (v1.4.0)

**Phase 1 is complete.** Pure Python optimizations exceeded all targets:

| Estimator | v1.3 (10K scale) | v1.4 (10K scale) | Speedup | vs R |
|-----------|------------------|------------------|---------|------|
| BasicDiD/TWFE | 0.835s | **0.011s** | **76x** | **4.2x faster than R** |
| CallawaySantAnna | 2.234s | **0.109s** | **20x** | **7.2x faster than R** |
| SyntheticDiD | 32.6s | N/A | N/A | 37x faster than R |

### What Was Implemented

1. **Unified `linalg.py` backend** (`diff_diff/linalg.py`)
   - `solve_ols()` - scipy lstsq with gelsy LAPACK driver
   - `compute_robust_vcov()` - Vectorized cluster-robust SE via pandas groupby
   - Single optimization point for all estimators

2. **CallawaySantAnna optimizations** (`staggered.py`)
   - `_precompute_structures()` - Pre-computed wide-format outcome matrix, cohort masks
   - `_compute_att_gt_fast()` - Vectorized ATT(g,t) using numpy (23x faster)
   - `_generate_bootstrap_weights_batch()` - Batch weight generation
   - Vectorized bootstrap using matrix operations (26x faster)

3. **TWFE optimization** (`twfe.py`)
   - Cached groupby indexes for within-transformation

4. **All estimators migrated** to unified backend
   - `estimators.py`, `twfe.py`, `staggered.py`, `triple_diff.py`, `synthetic_did.py`, `sun_abraham.py`, `utils.py`

---

## Original Problem Statement

Benchmark comparisons showed that while diff-diff was competitive or faster than R for small datasets, performance degraded significantly at scale:

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

### Phase 2: Compiled Backend

Implement performance-critical components in a compiled language for maximum speed.

#### Backend Options: Rust vs C++

We have two viable options for a compiled backend. Both can achieve near-identical performance; the choice depends on team expertise and maintenance considerations.

##### Option A: Rust with PyO3

**Pros:**
- **Memory safety by design** - No segfaults, buffer overflows, or data races; compiler catches these at build time
- **Modern tooling** - Cargo package manager + maturin makes wheel building straightforward
- **Zero-copy NumPy interop** - rust-numpy crate provides direct array access without copying
- **Easy parallelism** - rayon crate makes parallel iteration trivial (`.par_iter()`)
- **Growing ecosystem** - Used by polars, pyfixest, cryptography, orjson, ruff
- **Low per-call overhead** - Research shows PyO3 has ~0.14ms overhead vs NumPy's ~3.5ms for simple operations
- **Single toolchain** - `cargo build` works the same on all platforms

**Cons:**
- **Learning curve** - Rust's ownership model takes time to learn
- **Smaller scientific ecosystem** - Fewer numerical libraries than C++ (though ndarray and faer are mature)
- **Slower compilation** - Rust compiles slower than C++
- **Newer language** - Less institutional knowledge, fewer Stack Overflow answers

**Key dependencies:** `pyo3`, `rust-numpy`, `ndarray`, `faer` (linear algebra), `rayon` (parallelism)

##### Option B: C++ with pybind11

**Pros:**
- **Mature ecosystem** - Eigen, Armadillo, Intel MKL, OpenBLAS all native C++
- **Familiar to more developers** - Larger pool of contributors
- **Proven in scientific Python** - NumPy, SciPy, scikit-learn, pandas all use C/C++ extensions
- **Excellent Eigen integration** - pybind11 has built-in support for Eigen matrices
- **Faster compilation** - C++ compiles faster than Rust
- **More optimization resources** - Decades of C++ performance tuning knowledge

**Cons:**
- **Memory safety risks** - Segfaults, buffer overflows, use-after-free possible; harder to debug
- **Manual memory management** - Must carefully manage lifetimes, especially with Python GC interaction
- **Complex build systems** - CMake configuration, compiler flags, platform-specific issues
- **Copy overhead by default** - pybind11 copies arrays unless carefully configured with `py::array_t`
- **Manual GIL management** - Easy to deadlock or corrupt state if GIL not handled correctly
- **Platform differences** - MSVC vs GCC vs Clang have different behaviors and flags

**Key dependencies:** `pybind11`, `Eigen` (linear algebra), `OpenMP` or `TBB` (parallelism)

##### Comparison Summary

| Factor | Rust (PyO3) | C++ (pybind11) |
|--------|-------------|----------------|
| Memory safety | Compile-time guarantees | Runtime risks |
| Build tooling | Cargo + maturin (simple) | CMake + scikit-build (complex) |
| NumPy interop | Zero-copy via rust-numpy | Zero-copy possible but tricky |
| Parallelism | rayon (trivial) | OpenMP/TBB (more boilerplate) |
| Linear algebra | faer, ndarray-linalg | Eigen, MKL, OpenBLAS |
| Ecosystem maturity | Growing | Established |
| Learning curve | Steeper (ownership) | Moderate (but footguns) |
| Wheel building | maturin-action (simple) | cibuildwheel (more config) |
| Debug experience | Good (cargo, clippy) | Variable (platform-dependent) |

##### Recommendation

**Rust with PyO3** is the recommended approach because:

1. **pyfixest validates this for our exact domain** - They use Rust/PyO3 for fixed effects econometrics
2. **Memory safety prevents production bugs** - No risk of segfaults in user code
3. **maturin simplifies distribution** - Single command builds wheels for all platforms
4. **rayon makes parallelization trivial** - Critical for bootstrap and cluster SE

However, **C++ is a viable alternative** if:
- Team has stronger C++ expertise
- Need to integrate with existing C++ econometrics code
- Want to leverage Eigen's mature linear algebra

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

#### Architecture by Backend

##### Rust Layout

```
diff_diff/
├── estimators.py          # Python API (unchanged)
├── _rust_backend/         # Compiled Rust module
│   └── ...
└── _fallback.py           # Pure Python fallback

src/                       # Rust source (Cargo workspace)
├── Cargo.toml
├── lib.rs
├── ols.rs                 # OLS with cluster SE
├── demeaning.rs           # Alternating projections
├── bootstrap.rs           # Parallel bootstrap
└── staggered.rs           # ATT(g,t) computation

pyproject.toml             # maturin build config
```

##### C++ Layout

```
diff_diff/
├── estimators.py          # Python API (unchanged)
├── _cpp_backend/          # Compiled C++ module
│   └── ...
└── _fallback.py           # Pure Python fallback

cpp/                       # C++ source
├── CMakeLists.txt
├── src/
│   ├── module.cpp         # pybind11 bindings
│   ├── ols.cpp            # OLS with cluster SE
│   ├── ols.hpp
│   ├── demeaning.cpp      # Within transformation
│   ├── demeaning.hpp
│   ├── bootstrap.cpp      # Parallel bootstrap
│   └── bootstrap.hpp
└── extern/
    └── eigen/             # Eigen submodule (or system install)

pyproject.toml             # scikit-build-core config
```

#### Distribution

##### Rust (maturin)

```yaml
# .github/workflows/wheels.yml
- uses: PyO3/maturin-action@v1
  with:
    command: build
    args: --release --out dist
```

- Simple single-action CI configuration
- Use abi3 stable ABI for Python version-independent wheels
- Cross-compilation via `--target` flag

##### C++ (cibuildwheel)

```yaml
# .github/workflows/wheels.yml
- uses: pypa/cibuildwheel@v2
  env:
    CIBW_BUILD: "cp39-* cp310-* cp311-* cp312-*"
```

- More configuration required for CMake integration
- Need to handle OpenMP linking per-platform
- Consider vcpkg or conan for dependency management

Both approaches build wheels for:
- Linux (manylinux2014, x86_64 and aarch64)
- macOS (x86_64 and ARM64)
- Windows (x86_64)

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

## Outcomes

### Phase 1 Results (v1.4.0) ✅

**Exceeded all targets:**

- BasicDiD @ 10K: 0.835s → **0.011s** (76x improvement, 4.2x faster than R)
- CallawaySantAnna @ 10K: 2.2s → **0.109s** (20x improvement, 7.2x faster than R)
- Bootstrap inference: 26x faster via vectorization

### Phase 2 (Rust Backend) - Optional Future Work

No longer required for R parity. May be pursued for:
- Further optimization at extreme scales (100K+ units)
- Parallel bootstrap across CPU cores
- Memory efficiency for very large datasets

## References

### Rust Backend

- [PyO3 User Guide](https://pyo3.rs/) - Rust bindings for Python
- [rust-numpy](https://github.com/PyO3/rust-numpy) - Zero-copy NumPy interop
- [maturin](https://github.com/PyO3/maturin) - Build and publish Rust Python packages
- [faer](https://github.com/sarah-ek/faer-rs) - Pure Rust linear algebra (competitive with MKL)
- [Polars](https://github.com/pola-rs/polars) - Example of Rust/Python hybrid architecture
- [pyfixest](https://github.com/py-econometrics/pyfixest) - Rust backend for fixed effects econometrics

### C++ Backend

- [pybind11 documentation](https://pybind11.readthedocs.io/) - C++ bindings for Python
- [pybind11 Eigen integration](https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html) - Zero-copy with Eigen
- [Eigen](https://eigen.tuxfamily.org/) - C++ linear algebra library
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) - CMake integration for Python packages
- [cibuildwheel](https://cibuildwheel.readthedocs.io/) - Build wheels for all platforms

### General

- [fixest demeaning algorithm](https://rdrr.io/cran/fixest/man/demeaning_algo.html) - Reference implementation
- [PyO3 vs C performance comparison](https://www.alphaxiv.org/overview/2507.00264v1) - Academic benchmark
- [Making Python 100x faster with Rust](https://ohadravid.github.io/posts/2023-03-rusty-python/) - Practical tutorial
