# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.6.0] - 2026-02-22

### Added
- **Continuous DiD estimator** (`ContinuousDiD`) implementing Callaway, Goodman-Bacon & Sant'Anna (2024)
  for continuous treatment dose-response analysis
  - `ContinuousDiDResults` with dose-response curves and event-study effects
  - `DoseResponseCurve` with bootstrap p-values
  - Analytical and bootstrap event-study SEs
  - P(D=0) warning for low-probability control groups
- Stacked DiD tutorial (Tutorial 13) with Q-weight computation walkthrough

### Changed
- Clarify aggregate Q-weight computation for unbalanced panels in Stacked DiD tutorial
- Replace SunAbraham manual bootstrap stats with NaN-gated utility

### Fixed
- Fix not-yet-treated control mask to respect anticipation parameter in ContinuousDiD
- Guard non-finite `original_effect` in `compute_effect_bootstrap_stats`
- Fix bootstrap NaN propagation for rank-deficient cells
- Fix NaN propagation in rank-deficient spline predictions
- Guard bootstrap NaN propagation: SE/CI/p-value all NaN when SE invalid
- Fix bootstrap ACRT^{glob} centering bug
- Fix bootstrap percentile inference and analytical event-study SE scaling
- Fix control group bug and dose validation in ContinuousDiD

## [2.5.0] - 2026-02-19

### Added
- Stacked DiD estimator (`StackedDiD`) implementing Wing, Freedman & Hollingsworth (2024)
  with corrective Q-weights for compositional balance across event times
- Sub-experiment construction per adoption cohort with clean (never-yet-treated) controls
- IC1/IC2 trimming for compositional balance across event times
- Q-weights for aggregate, population, or sample share estimands (Table 1)
- WLS event study regression via sqrt(w) transformation
- `stacked_did()` convenience function
- R benchmark scripts for Stacked DiD validation (`benchmarks/R/benchmark_stacked_did.R`)
- Comprehensive test suite for Stacked DiD (`tests/test_stacked_did.py`)

### Fixed
- NaN inference handling in pure Python mode for edge cases

## [2.4.3] - 2026-02-19

### Changed
- Rewrite TripleDifference estimator to match R's `triplediff::ddd()` — all 3 estimation
  methods (DR, IPW, RA) now use three-DiD decomposition with influence function SE, achieving
  <0.001% relative difference from R across all 24 comparisons (4 DGPs × 3 methods × 2 covariate settings)
- Validate cluster column in TripleDifference for proper cluster-robust SEs
- Handle non-finite influence function propagation in TripleDifference edge cases
- Propensity score fallback uses Hessian-based SE when score optimization fails
- Improved R-squared consistency across estimation methods

### Fixed
- Fix low cell count warning and overlap detection in TripleDifference IPW
- Fix cluster SE computation to use functional (groupby) approach instead of loop
- Fix rank deficiency handling in TripleDifference regression adjustment

### Added
- 91 methodology verification tests for TripleDifference (`tests/test_methodology_triple_diff.py`)
- R benchmark scripts for triple difference validation (`benchmarks/R/benchmark_triplediff.R`)
- Update METHODOLOGY_REVIEW.md to reflect completed TripleDifference review

## [2.4.2] - 2026-02-18

### Added
- **Conditional BLAS linking for Rust backend** — Apple Accelerate on macOS, OpenBLAS on Linux.
  Pre-built wheels now use platform-optimized BLAS for matrix-vector and matrix-matrix
  operations across all Rust-accelerated code paths (weights, OLS, TROP). Windows continues
  using pure Rust (no external dependencies). Improves Rust backend performance at larger scales.
- `rust_backend_info()` diagnostic function in `diff_diff._backend` — reports compile-time
  BLAS feature status (blas, accelerate, openblas)

### Fixed
- **Rust SDID backend performance regression at scale** — Frank-Wolfe solver was 3-10x slower than pure Python at 1k+ scale
  - Gram-accelerated FW loop for time weights: precomputes A^T@A, reducing per-iteration cost from O(N×T0) to O(T0) (~100x speedup per iteration at 5k scale)
  - Allocation-free FW loop for unit weights: 1 GEMV per iteration (was 3), zero heap allocations (was ~8)
  - Dispatch based on problem dimensions: Gram path when T0 < N, standard path when T0 >= N
  - Rust backend now faster than pure Python at all scales

## [2.4.1] - 2026-02-17

### Added
- Tutorial notebook for Two-Stage DiD (Gardner 2022) (`docs/tutorials/12_two_stage_did.ipynb`)

### Changed
- Module splits for large files: ImputationDiD, TwoStageDiD, and TROP each split into separate results and bootstrap submodules
- Migrated remaining inline inference computations to `safe_inference()` utility
- Replaced `@` operator with `np.dot()` at observation-dimension sites to avoid Apple M4 BLAS warnings
- Updated TODO.md and ROADMAP.md for accuracy post-v2.4.0

### Fixed
- Matplotlib import guards added to tutorials 11 and 12
- Various bug fixes from code quality cleanup (diagnostics, estimators, linalg, staggered, sun_abraham, synthetic_did, triple_diff)

## [2.4.0] - 2026-02-16

### Added
- **Gardner (2022) Two-Stage DiD estimator** (`TwoStageDiD`)
  - Two-stage estimator: (1) estimate unit+time FE on untreated obs, (2) regress residualized outcomes on treatment indicators
  - `TwoStageDiDResults` with overall ATT, event study, group effects, per-observation treatment effects
  - `TwoStageBootstrapResults` for multiplier bootstrap inference on GMM influence function
  - `two_stage_did()` convenience function for quick estimation
  - Point estimates identical to ImputationDiD; different variance estimator (GMM sandwich vs. conservative)
  - No finite-sample adjustments (raw asymptotic sandwich, matching R `did2s`)
- Proposition 5 detection for unidentified long-run horizons without never-treated units

### Changed
- Workflow improvements to reduce PR review rounds

### Fixed
- Zero-observation horizons/cohorts producing se=0 instead of NaN in TwoStageDiD
- Edge case fixes for TwoStageDiD (PR review feedback)
- Grep PCRE patterns updated to use POSIX character classes

## [2.3.2] - 2026-02-16

### Added
- **Python 3.13 support** with upper version cap (`>=3.9,<3.14`)

### Changed
- **Sun-Abraham methodology review** (PR #153)
  - IW aggregation weights now use event-time observation counts (not group sizes)
  - Normalize `np.inf` never-treated encoding before treatment group detection
  - Add R benchmark scripts and methodology-aligned tests
- Use `rank_deficient_action` and `np.errstate` instead of broad `RuntimeWarning` filter in SDID tutorial

### Fixed
- Sun-Abraham bootstrap NaN propagation for non-finite ATT estimates
- Sun-Abraham df_adjustment off-by-one in analytical SE computation
- CI pandas compatibility for SunAbraham bootstrap inference
- SyntheticDiD tutorial: eliminate pre-treatment fit warnings

## [2.3.1] - 2026-02-15

### Fixed
- Fix docs/PyPI version mismatch (issue #146) — RTD now builds versioned docs from source
- Fix RTD docs build failure caused by Rust/maturin compilation timeout on ReadTheDocs

### Changed
- Remove Rust outer-loop variance estimation for SyntheticDiD (placebo and bootstrap)
  - Fixes SE mismatch between pure Python and Rust backends (different RNG sequences)
  - Fixes Rust performance regression at 1k+ scale (memory bandwidth saturation from rayon parallelism)
  - Inner Frank-Wolfe weight computation still uses Rust when available

### Documentation
- Re-run SyntheticDiD benchmarks against R after Frank-Wolfe methodology rewrite
- Updated `docs/benchmarks.rst` SDID validation results, performance tables, and known differences
- ATT now matches R to < 1e-10 (previously 0.3% diff) since both use Frank-Wolfe optimizer

## [2.3.0] - 2026-02-09

### Added
- **Borusyak-Jaravel-Spiess (2024) Imputation DiD estimator** (`ImputationDiD`)
  - Efficient imputation estimator for staggered DiD designs
  - OLS on untreated observations for unit+time FE, impute counterfactual Y(0), aggregate
  - Conservative variance (Theorem 3) with `aux_partition` parameter for SE tightness
  - Pre-trend test (Equation 9) via `results.pretrend_test()`
  - Percentile bootstrap inference
  - Influence-function bootstrap with sparse variance and weight/covariate fixes
  - Absorbing-treatment validation for non-constant `first_treat`
  - Empty event-study warning for unidentified long-run horizons
- **`/paper-review` skill** for academic paper methodology extraction
- **`/read-feedback-revise` skill** for addressing PR review comments
- **`--pr` flag for `/review-plan` skill** to review plans posted as PR comments
- **`--updated` flag for `/review-plan` skill** for re-reviewing revised plans
- **MultiPeriodDiD vs R (fixest) benchmark** for cross-language validation

### Changed
- Shortened test suite runtime with parallel execution and reduced iterations

### Fixed
- **TWFE within-transformation bug** identified during methodology review
- TWFE: added non-{0,1} binary time warning, ATT invariance tests, and R fixture caching
- TWFE: single-pass demeaning, HC1 test fix, fixest coeftable comparison
- MultiPeriodDiD: added unit FE and NaN guard for R comparison benchmark
- Removed tracked PDF from repo and gitignored papers directory

## [2.2.1] - 2026-02-07

### Changed
- **MultiPeriodDiD: Full event-study specification** (BREAKING)
  - Treatment × period interactions now created for ALL periods (pre and post),
    not just post-treatment
  - Pre-period coefficients available for parallel trends assessment
  - Default reference period changed from first to last pre-period (e=-1 convention)
    with FutureWarning for one release cycle
  - `period_effects` dict now contains both pre and post period effects
  - `to_dataframe()` includes `is_post` column
  - `summary()` output now shows pre-period effects section
  - t_stat uses `np.isfinite(se) and se > 0` guard (consistent with other estimators)

### Added
- Time-varying treatment warning when `unit` is provided and treatment varies
  within units (guides users toward ever-treated indicator D_i)
- `unit` parameter to `MultiPeriodDiD.fit()` for staggered adoption detection
- `reference_period` and `interaction_indices` attributes on `MultiPeriodDiDResults`
- `pre_period_effects` and `post_period_effects` convenience properties on results
- Pre-period section in `summary()` output with reference period indicator
- `ValueError` when `reference_period` is set to a post-treatment period
- Staggered adoption warning when treatment timing varies across units (with `unit` param)
- Informative KeyError when accessing reference period via `get_effect()`

### Removed
- **TROP `variance_method` parameter** — Jackknife variance estimation removed.
  Bootstrap (the only method specified in Athey et al. 2025) is now always used.
  The `variance_method` field has also been removed from `TROPResults`.
- **TROP `max_loocv_samples` parameter** — Control observation subsampling removed
  from LOOCV tuning parameter selection. Equation 5 of Athey et al. (2025) explicitly
  sums over ALL control observations where D=0; the previous subsampling (default 100)
  was not specified in the paper. LOOCV now uses all control observations, making
  tuning fully deterministic. Inner LOOCV loops in the Rust backend are parallelized
  to compensate for the increased observation count.

### Fixed
- HonestDiD: filter non-finite period effects from MultiPeriodDiD results
  (prevents NaN propagation into sensitivity bounds; raises ValueError
  when no finite pre- or post-period effects remain)
- HonestDiD VCV extraction: now uses interaction sub-VCV instead of full regression VCV
  (via `interaction_indices` period → column index mapping)
- MultiPeriodDiD: `avg_se` guard now checks `np.isfinite()` (matches per-period pattern;
  prevents `avg_t_stat=0` / `avg_p_value=1` when variance is infinite)
- HonestDiD: extraction now uses explicit pre-then-post ordering instead of sorted period
  labels (prevents misclassification when period labels don't sort chronologically)
- Backend-aware test parameter scaling for pure Python CI performance
- Lower TROP stratified bootstrap threshold floor from 11 to 5 for pure Python CI

## [2.2.0] - 2026-01-27

### Added
- **Windows wheel builds** using pure-Rust `faer` library for linear algebra (PR #115)
  - Eliminates external BLAS/LAPACK dependencies (no OpenBLAS or Intel MKL required)
  - Enables cross-platform wheel builds for Linux, macOS, and Windows
  - Simplifies installation on all platforms

### Changed
- **Rust backend migrated from nalgebra/ndarray to faer** (PR #115)
  - OLS solver now uses faer's SVD implementation
  - Robust variance estimation uses faer's matrix operations
  - TROP distance calculations use faer primitives
  - Maintains numerical parity with existing NumPy backend

### Fixed
- **Rust backend numerical stability improvements** (PR #115)
  - Improved singular matrix detection with condition number checks
  - NaN propagation in variance-covariance estimation
  - Fallback to Python backend on numerical instability with warning
  - Underdetermined SVD handling (n < k case)
- **macOS CI compatibility** for Python 3.14 with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY`

## [2.1.9] - 2026-01-26

### Added
- **Unified LOOCV for TROP joint method** with Rust acceleration (PR #113)
  - Leave-one-out cross-validation for rank and regularization parameter selection
  - Rust backend provides significant speedup for LOOCV grid search

### Fixed
- **TROP joint method Rust/Python parity** (PR #113)
  - Fixed valid_count bug in LOOCV computation
  - Proper NaN exclusion for units with no valid pre-period data
  - Zero weight assignment for units missing pre-period data
  - Jackknife variance estimation fixes
  - Staggered adoption validation and simultaneous adoption enforcement
  - Treated-pre NaN handling improvements
  - LOOCV subsampling fix for Python-only path

## [2.1.8] - 2026-01-25

### Added
- **`/push-pr-update` skill** for committing and pushing PR revisions
  - Commits local changes to current branch and pushes to remote
  - Triggers AI code review automatically
  - Robust handling for fork repos, unpushed commits, and upstream tracking

### Fixed
- **TROP estimator methodology alignment** (PR #110)
  - Aligned with paper methodology (Equation 5, D matrix semantics)
  - NaN propagation and LOOCV warnings improvements
  - Rust backend test alignment with new loocv_grid_search return signature
  - LOOCV cycling, D matrix validation fixes
  - Final estimation infinity handling and edge case fixes
  - Absorbing-state gap detection and n_post_periods fix

### Changed
- **`/submit-pr` skill improvements** (PR #111)
  - Case-insensitive secret scanning with POSIX ERE regex
  - Verify origin ref exists before push
  - Dynamic default branch detection with fallback
  - Robust handling for unpushed commits, fork repos
  - Files count display in PR summary

## [2.1.7] - 2026-01-25

### Fixed
- **`plot_event_study` reference period normalization behavior**
  - Effects are now only normalized when `reference_period` is explicitly provided
  - Auto-inferred reference periods only apply hollow marker styling (no normalization)
  - Reference period SE is set to NaN during normalization (constraint, not estimate)
  - Updated docstring to clarify explicit vs auto-inferred behavior

### Changed
- Refactored visualization tests to reuse `cs_results` fixture for better performance

## [2.1.6] - 2026-01-24

### Added
- **Methodology verification tests** for DifferenceInDifferences estimator
  - Comprehensive test suite validating all REGISTRY.md requirements
  - Tests for formula interface, coefficient extraction, rank deficiency handling
  - Singleton cluster variance estimation behavioral tests

### Changed
- **REGISTRY.md documentation improvements**
  - Clarified singleton cluster formula notation (u_i² X_i X_i' instead of ambiguous residual² × X'X)
  - Verified DifferenceInDifferences behavior against documented requirements

## [2.1.5] - 2026-01-22

### Added
- **METHODOLOGY_REVIEW.md** tracking document for methodology review progress
  - Review status summary table for all 12 estimators
  - Detailed notes template for each estimator by category
  - Review process guidelines with checklist and priority ordering
- **`base_period` parameter** for CallawaySantAnna pre-treatment effect computation
  - "varying" (default): Pre-treatment uses t-1 as base (consecutive comparisons)
  - "universal": All comparisons use g-anticipation-1 as base
  - Matches R `did::att_gt()` base_period parameter
- **Pre-merge-check skill** (`/pre-merge-check`) for automated PR validation
  - Pattern checks for NaN handling consistency
  - Context-specific checklist generation

### Changed
- **Tutorial 02 improvements**: Added pre-trends section, clarified base_period interaction with anticipation

### Fixed
- Not-yet-treated control group now properly excludes cohort g when computing ATT(g,t)
- Aggregation t_stat uses NaN (not 0.0) when SE is non-finite or zero
- Bootstrap inference for pre-treatment effects with `base_period="varying"`
- NaN propagation for empty post-treatment effects in CallawaySantAnna
- Grep word boundary pattern in pre-merge-check skill

## [2.1.4] - 2026-01-20

### Added
- **Development checklists and workflow improvements** in `CLAUDE.md`
  - Estimator inheritance map showing class hierarchy for `get_params`/`set_params`
  - Test writing guidelines for fallback paths, parameters, and warnings
  - Checklists for adding parameters and warning/error handling
- **R-style rank deficiency handling** across all estimators
  - `rank_deficient_action` parameter: "warn" (default), "error", or "silent"
  - Dropped columns have NaN coefficients (like R's `lm()`)
  - VCoV matrix has NaN for rows/cols of dropped coefficients
  - Propagated to all estimators: DifferenceInDifferences, MultiPeriodDiD, TwoWayFixedEffects, CallawaySantAnna, SunAbraham, TripleDifference, TROP, SyntheticDiD

### Fixed
- `get_params()` now includes `rank_deficient_action` parameter (fixes sklearn cloning)
- NaN vcov fallback in Rust backend for rank-deficient matrices
- MultiPeriodDiD vcov/df computation for rank-deficient designs
- Average ATT inference for rank-deficient designs

### Changed
- Rank tolerance aligned with R's `lm()` default for consistent behavior

## [2.1.3] - 2026-01-19

### Fixed
- TROP estimator paper conformance issues (Athey et al. 2025)
  - Control set now includes pre-treatment observations of eventually-treated units (Issue A)
  - Unit distance computation excludes target period per Equation 3 (Issue B)
  - Nuclear norm update uses weighted proximal gradient instead of unweighted soft-thresholding (Issue C)
  - Bootstrap sampling now stratifies by treatment status per Algorithm 3 (Issue D)
- TROP Rust backend alignment with paper specification
  - Weight normalization to sum to 1 (probability weights)
  - Weighted proximal gradient for L update with step size η ≤ 1/max(W)

### Changed
- Cleaned up unused parameters from TROP Rust API
  - Removed `control_unit_idx` and `unit_dist_matrix` from public functions
  - Per-observation distances now computed dynamically (more accurate, slightly slower)

## [2.1.2] - 2026-01-19

### Added
- **Consolidated DGP functions** in `prep.py` for all supported DiD designs
  - `generate_did_data()` - Basic 2x2 DiD data generation
  - `generate_staggered_data()` - Staggered adoption data for Callaway-Sant'Anna/Sun-Abraham
  - `generate_factor_data()` - Factor model data for TROP/SyntheticDiD
  - `generate_ddd_data()` - Triple Difference (DDD) design data
  - `generate_panel_data()` - Panel data with optional parallel trends violations
  - `generate_event_study_data()` - Event study data with simultaneous treatment

### Changed
- **Clean up development tracking files** for v2.1.1 release
  - Removed completed items from TODO.md (now tracked in CHANGELOG)
  - Updated ROADMAP.md version numbers and removed shipped TROP section
  - Updated `prep.py` line count in Large Module Files table (1338 → 1993)

## [2.1.1] - 2026-01-19

### Added
- **Rust backend acceleration for TROP estimator** delivering 5-20x overall speedup
  - `compute_unit_distance_matrix` - Parallel pairwise RMSE computation for donor matching
  - `loocv_grid_search` - Parallel leave-one-out cross-validation across 180 parameter combinations
  - `bootstrap_trop_variance` - Parallel bootstrap variance estimation
  - Automatic fallback to Python when Rust backend unavailable
  - Logging for Rust fallback events to aid debugging
- **`/bump-version` skill** for release management
  - Updates version in `__init__.py`, `pyproject.toml`, and `rust/Cargo.toml`
  - Generates CHANGELOG entries from git commits
  - Adds comparison links automatically
- **`/review-pr` skill** for code review workflow

### Changed
- **TROP estimator performance optimizations** (Python backend)
  - Vectorized distance matrix computation using NumPy broadcasting
  - Extracted tuning constants to module-level for clarity
  - Added `TROPTuningParams` TypedDict for parameter documentation

### Fixed
- Tutorial notebook validation errors in `10_trop.ipynb`
- Pre-existing RuntimeWarnings in CallawaySantAnna bootstrap (documented)
- TROP `pre_periods` parameter handling for edge cases

## [2.1.0] - 2026-01-17

### Added
- **Triply Robust Panel (TROP) estimator** implementing Athey, Imbens, Qu & Viviano (2025)
  - `TROP` class combining three robustness components:
    - Factor model adjustment via SVD (removes unobserved confounders with factor structure)
    - Synthetic control style unit weights
    - SDID style time weights
  - `TROPResults` dataclass with ATT, factors, loadings, unit/time weights
  - `trop()` convenience function for quick estimation
  - Automatic rank selection methods: cross-validation (`'cv'`), information criterion (`'ic'`), elbow detection (`'elbow'`)
  - Bootstrap and placebo-based variance estimation
  - Full integration with existing infrastructure (exports in `__init__.py`, sklearn-compatible API)
  - Tutorial notebook: `docs/tutorials/10_trop.ipynb`
  - Comprehensive test suite: `tests/test_trop.py`

**Reference**: Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). "Triply Robust Panel Estimators." *Working Paper*. [arXiv:2508.21536](https://arxiv.org/abs/2508.21536)

## [2.0.3] - 2026-01-17

### Changed
- **Rust backend performance optimizations** delivering up to 32x speedup for bootstrap operations
  - Bootstrap weight generation now 16x faster on average (up to 32x for Webb distribution)
  - Direct `Array2` allocation eliminates intermediate `Vec<Vec<f64>>` (~50% memory reduction)
  - Rayon chunk size tuning (`min_len=64`) reduces parallel scheduling overhead
  - Webb distribution uses lookup table instead of 6-way if-else chain

### Added
- **LinearRegression helper class** in `linalg.py` for code deduplication
  - High-level OLS wrapper with unified coefficient extraction and inference
  - Used by DifferenceInDifferences, TwoWayFixedEffects, SunAbraham, TripleDifference
  - Provides `InferenceResult` dataclass for coefficient-level statistics
- **Cholesky factorization** for symmetric positive-definite matrix inversion in Rust backend
  - ~2x faster than LU decomposition for well-conditioned matrices
  - Automatic fallback to LU for near-singular or indefinite matrices
- **Vectorized variance computation** in Rust backend
  - HC1 meat computation: `X' @ (X * e²)` via BLAS instead of O(n×k²) loop
  - Score computation: broadcast multiplication instead of O(n×k) loop
- **Static BLAS linking options** in `rust/Cargo.toml`
  - `openblas-static` and `intel-mkl-static` features for standalone distribution
  - Eliminates runtime BLAS dependency at cost of larger binary size

## [2.0.2] - 2026-01-15

### Fixed
- **CallawaySantAnna SE computation** now exactly matches R's `did` package
  - Fixed weight influence function (wif) formula for "simple" aggregation
  - Corrected `pg` computation: uses `n_g / n_all` (matching R) instead of `n_g / total_treated`
  - Fixed wif iteration: iterates over keepers (post-treatment pairs) with individual ATT(g,t) values
  - SE difference reduced from ~2.5% to <0.01% vs R's `did` package (essentially exact match)
  - Point estimates unchanged; all existing tests pass

## [2.0.1] - 2026-01-13

### Added
- **Shared within-transformation utilities** in `utils.py`
  - `demean_by_group()` - One-way fixed effects demeaning
  - `within_transform()` - Two-way (unit + time) FE transformation
  - Reduces code duplication across `estimators.py`, `twfe.py`, `sun_abraham.py`, `bacon.py`

### Fixed
- **DataFrame fragmentation warning** - Build columns in batch instead of iteratively

### Changed
- Reverted untested Rust backend optimizations (Cholesky factorization, reduced allocations) - these will be re-added when proper testing infrastructure is available

## [2.0.0] - 2026-01-12

### Added
- **Optional Rust backend** for accelerated computation
  - 4-8x speedup for SyntheticDiD and bootstrap operations
  - Parallel bootstrap weight generation (Rademacher, Mammen, Webb)
  - Accelerated OLS solver using OpenBLAS/MKL
  - Cluster-robust variance estimation
  - Synthetic control weight optimization with simplex projection
  - Pre-built wheels for Linux x86_64 and macOS ARM64
  - Pure Python fallback for all other platforms
- **`diff_diff/_backend.py`** - Backend detection and configuration module
  - `HAS_RUST_BACKEND` flag exported in main package
  - `DIFF_DIFF_BACKEND` environment variable for backend control:
    - `'auto'` (default) - Use Rust if available, fall back to Python
    - `'python'` - Force pure Python mode
    - `'rust'` - Force Rust mode (fails if unavailable)
- **Rust source code** in `rust/` directory
  - `rust/src/lib.rs` - PyO3 module definition
  - `rust/src/bootstrap.rs` - Parallel bootstrap weight generation
  - `rust/src/linalg.rs` - OLS solver and robust variance estimation
  - `rust/src/weights.rs` - Synthetic control weights and simplex projection
- **Rust backend test suite** - `tests/test_rust_backend.py` for equivalence testing

### Changed
- Package version bumped from 1.4.0 to 2.0.0 (major version for new backend)
- CI/CD updated to build Rust extensions with maturin
- ReadTheDocs now installs from PyPI (pre-built wheels with Rust backend)

## [1.4.0] - 2026-01-11

### Added
- **Unified linear algebra backend** (`diff_diff/linalg.py`)
  - `solve_ols()` - Optimized OLS solver using scipy's gelsy LAPACK driver
  - `compute_robust_vcov()` - Vectorized (clustered) robust variance-covariance
  - Single optimization point for all estimators; prepares for future Rust backend
  - New `tests/test_linalg.py` with comprehensive tests

### Changed
- **Major performance improvements** - All estimators now significantly faster
  - BasicDiD/TWFE @ 10K: 0.835s → 0.011s (76x faster, now 4.2x faster than R)
  - CallawaySantAnna @ 10K: 2.234s → 0.109s (20x faster, now 7.2x faster than R)
  - All results numerically identical to previous versions
- **CallawaySantAnna optimizations** (`staggered.py`)
  - Pre-computed wide-format outcome matrix and cohort masks
  - Vectorized ATT(g,t) computation using numpy operations (23x faster)
  - Batch bootstrap weight generation
  - Vectorized multiplier bootstrap using matrix operations (26x faster)
- **TWFE optimization** (`twfe.py`)
  - Cached groupby indexes for within-transformation
- **All estimators migrated** to unified `linalg.py` backend
  - `estimators.py`, `twfe.py`, `staggered.py`, `triple_diff.py`,
    `synthetic_did.py`, `sun_abraham.py`, `utils.py`

### Behavioral Changes
- **Rank-deficient design matrices**: The new `gelsy` LAPACK driver handles
  rank-deficient matrices gracefully (returning a least-norm solution) rather
  than raising an explicit error. Previously, `DifferenceInDifferences` would
  raise `ValueError("Design matrix is rank-deficient")`. Users relying on this
  error for collinearity detection should validate their design matrices
  separately. Results remain numerically correct for well-specified models.

## [1.3.1] - 2026-01-10

### Added
- **SyntheticDiD placebo-based variance estimation** matching R's `synthdid` package methodology
  - New `variance_method` parameter with options `"bootstrap"` (default) and `"placebo"`
  - Placebo method implements Algorithm 4 from Arkhangelsky et al. (2021):
    1. Randomly permutes control unit indices
    2. Designates N₁ controls as pseudo-treated (matching actual treated count)
    3. Renormalizes original unit weights for remaining pseudo-controls
    4. Computes SDID estimate with renormalized weights
    5. Repeats for `n_bootstrap` replications
    6. SE = sqrt((r-1)/r) × sd(estimates)
  - Provides methodological parity with R's `synthdid::vcov(method = "placebo")`
  - `n_bootstrap` parameter now used for both bootstrap and placebo replications
  - `SyntheticDiDResults` now tracks `variance_method` and `n_bootstrap` attributes
  - Results summary displays variance method and replications count

**Reference**: Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088-4118.

## [1.3.0] - 2026-01-09

### Added
- **Triple Difference (DDD) estimator** implementing Ortiz-Villavicencio & Sant'Anna (2025)
  - `TripleDifference` class for DDD designs where treatment requires two criteria (group AND partition)
  - `TripleDifferenceResults` dataclass with ATT, SEs, cell means, and diagnostics
  - `triple_difference()` convenience function for quick estimation
  - Three estimation methods: regression adjustment (`reg`), inverse probability weighting (`ipw`), and doubly robust (`dr`)
  - Proper covariate handling (unlike naive DDD implementations that difference two DiDs)
  - Propensity score trimming for IPW/DR methods
  - Cluster-robust standard errors support
  - Tutorial notebook: `docs/tutorials/08_triple_diff.ipynb`

**Reference**: Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). "Better Understanding Triple Differences Estimators." *Working Paper*. [arXiv:2505.09942](https://arxiv.org/abs/2505.09942)

## [1.2.1] - 2026-01-08

### Added
- **Expanded test coverage** for edge cases:
  - Wild bootstrap with very few clusters (< 5), including 2-3 cluster scenarios
  - Unbalanced panels with missing periods across units
  - Single treated unit scenarios for DiD and Synthetic DiD
  - Perfect collinearity detection (validates clear error messages)
  - CallawaySantAnna with single treatment cohort
  - SyntheticDiD with insufficient pre-treatment periods

### Changed
- **Refactored CallawaySantAnna bootstrap**: Extracted `_compute_effect_bootstrap_stats()` helper method for cleaner code and reduced duplication in bootstrap statistics computation.

## [1.2.0] - 2026-01-07

### Added
- **Pre-Trends Power Analysis** (Roth 2022) for assessing informativeness of pre-trends tests
  - `PreTrendsPower` class for computing power and minimum detectable violation (MDV)
  - `PreTrendsPowerResults` dataclass with power, MDV, and test statistics
  - `PreTrendsPowerCurve` for power curves across violation magnitudes
  - `compute_pretrends_power()` and `compute_mdv()` convenience functions
  - Multiple violation types: `linear`, `constant`, `last_period`, `custom`
  - Integration with Honest DiD via `sensitivity_to_honest_did()` method
  - `plot_pretrends_power()` visualization for power curves
  - Tutorial notebook: `docs/tutorials/07_pretrends_power.ipynb`
  - Full API documentation: `docs/api/pretrends.rst`

**Reference**: Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322.

### Fixed
- **Reference period handling in pre-trends analysis**: Fixed bug where reference period was incorrectly assigned `avg_se` instead of being excluded from power calculations. Now properly excludes the omitted reference period from the joint Wald test.

## [1.1.1] - 2026-01-06

### Fixed
- **SyntheticDiD bootstrap error handling**: Bootstrap now raises clear `ValueError` when all iterations fail, instead of silently returning SE=0.0. Added warnings for edge cases (single successful iteration, high failure rate).

- **Diagnostics module error handling**: Improved error messages in `permutation_test()` and `leave_one_out_test()` with actionable guidance. Added warnings when significant iterations fail. Enhanced `run_all_placebo_tests()` to return structured error info including error type.

### Changed
- **Code deduplication**: Extracted wild bootstrap inference logic to shared `_run_wild_bootstrap_inference()` method in `DifferenceInDifferences` base class, used by both `DifferenceInDifferences` and `TwoWayFixedEffects`.

- **Type hints**: Added missing type hints to nested functions:
  - `compute_trend()` in `utils.py`
  - `neg_log_likelihood()` and `gradient()` in `staggered.py`
  - `format_label()` in `prep.py`

## [1.1.0] - 2026-01-05

### Added
- **Sun-Abraham (2021) interaction-weighted estimator** for staggered DiD
  - `SunAbraham` class implementing saturated regression approach
  - `SunAbrahamResults` with event study effects, cohort weights, and overall ATT
  - `SABootstrapResults` for bootstrap inference (SEs, CIs, p-values)
  - Support for `never_treated` and `not_yet_treated` control groups
  - Analytical and cluster-robust standard errors
  - Multiplier bootstrap with Rademacher, Mammen, or Webb weights
  - Integration with `plot_event_study()` visualization
  - Useful robustness check alongside Callaway-Sant'Anna

**Reference**: Sun, L., & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199.

## [1.0.2] - 2026-01-04

### Changed
- Refactored `estimators.py` to reduce module size
  - Moved `TwoWayFixedEffects` to `diff_diff/twfe.py`
  - Moved `SyntheticDiD` to `diff_diff/synthetic_did.py`
  - Backward compatible re-exports maintained in `estimators.py`

### Fixed
- Fixed ReadTheDocs version display by importing from package `__version__`

## [1.0.1] - 2026-01-04

### Fixed
- Tech debt cleanup (Tier 1 + Tier 2)
  - Improved code organization and documentation
  - Fixed minor issues identified in tech debt review

## [1.0.0] - 2026-01-04

### Added
- **Goodman-Bacon decomposition** for TWFE diagnostics
  - `BaconDecomposition` class for decomposing TWFE into weighted 2x2 comparisons
  - `Comparison2x2` dataclass for individual comparisons (treated_vs_never, earlier_vs_later, later_vs_earlier)
  - `BaconDecompositionResults` with weights and estimates by comparison type
  - `bacon_decompose()` convenience function
  - `plot_bacon()` visualization for decomposition results
  - Integration via `TwoWayFixedEffects.decompose()` method
- **Power analysis** for study design
  - `PowerAnalysis` class for analytical power calculations
  - `PowerResults` and `SimulationPowerResults` dataclasses
  - `compute_mde()`, `compute_power()`, `compute_sample_size()` convenience functions
  - `simulate_power()` for Monte Carlo simulation-based power analysis
  - `plot_power_curve()` visualization for power analysis
  - Tutorial notebook: `docs/tutorials/06_power_analysis.ipynb`
- **Callaway-Sant'Anna multiplier bootstrap** for inference
  - `CSBootstrapResults` with standard errors, confidence intervals, p-values
  - Rademacher, Mammen, and Webb weight distributions
  - Bootstrap inference for all aggregation methods
- **Troubleshooting guide** in documentation
- **Standard error computation guide** explaining SE differences across estimators

### Changed
- Updated package status to Production/Stable (was Alpha)
- SyntheticDiD bootstrap now warns when >5% of iterations fail

### Fixed
- Silent bootstrap failures in SyntheticDiD now produce warnings

## [0.6.0]

### Added
- **CallawaySantAnna covariate adjustment** for conditional parallel trends
  - Outcome regression (`estimation_method='reg'`)
  - Inverse probability weighting (`estimation_method='ipw'`)
  - Doubly robust estimation (`estimation_method='dr'`)
  - Pass covariates via `covariates` parameter in `fit()`
- **Honest DiD sensitivity analysis** (Rambachan & Roth 2023)
  - `HonestDiD` class for computing bounds under parallel trends violations
  - Relative magnitudes restriction (`DeltaRM`) - bounds post-treatment violations by pre-treatment
  - Smoothness restriction (`DeltaSD`) - bounds second differences of trend violations
  - Combined restrictions (`DeltaSDRM`)
  - FLCI and C-LF confidence interval methods
  - Breakdown value computation via `breakdown_value()`
  - Sensitivity analysis over M grid via `sensitivity_analysis()`
  - `HonestDiDResults` and `SensitivityResults` dataclasses
  - `compute_honest_did()` convenience function
  - `plot_sensitivity()` for sensitivity analysis visualization
  - `plot_honest_event_study()` for event study with honest CIs
  - Tutorial notebook: `docs/tutorials/05_honest_did.ipynb`
- **API documentation site** with Sphinx
  - Full API reference auto-generated from docstrings
  - "Which estimator should I use?" decision guide
  - Comparison with R packages (did, HonestDiD)
  - Getting started / quickstart guide

### Changed
- Updated mypy configuration for better numpy type compatibility
- Modernized ruff configuration to use `[tool.ruff.lint]` section

### Fixed
- Fixed 21 ruff linting issues (import ordering, unused variables, ambiguous names)
- Fixed 94 mypy type checking issues (Optional types, numpy type casts, assertions)
- Added missing return statement in `run_placebo_test()`

## [0.5.0]

### Added
- **Wild cluster bootstrap** for valid inference with few clusters
  - Rademacher weights (default, good for most cases)
  - Webb's 6-point distribution (recommended for <10 clusters)
  - Mammen's two-point distribution
  - `WildBootstrapResults` dataclass
  - `wild_bootstrap_se()` utility function
  - Integration with `DifferenceInDifferences` and `TwoWayFixedEffects` via `inference='wild_bootstrap'`
- **Placebo tests module** (`diff_diff.diagnostics`)
  - `placebo_timing_test()` - fake treatment timing test
  - `placebo_group_test()` - fake treatment group test
  - `permutation_test()` - permutation-based inference
  - `leave_one_out_test()` - sensitivity to individual treated units
  - `run_placebo_test()` - unified dispatcher for all test types
  - `run_all_placebo_tests()` - comprehensive diagnostic suite
  - `PlaceboTestResults` dataclass
- **Tutorial notebooks** in `docs/tutorials/`
  - `01_basic_did.ipynb` - Basic 2x2 DiD, formula interface, covariates, fixed effects, wild bootstrap
  - `02_staggered_did.ipynb` - Staggered adoption with Callaway-Sant'Anna
  - `03_synthetic_did.ipynb` - Synthetic DiD with unit/time weights
  - `04_parallel_trends.ipynb` - Parallel trends testing and diagnostics
- Comprehensive test coverage (380+ tests)

## [0.4.0]

### Added
- **Callaway-Sant'Anna estimator** for staggered difference-in-differences
  - `CallawaySantAnna` class with group-time ATT(g,t) estimation
  - Support for `never_treated` and `not_yet_treated` control groups
  - Aggregation methods: `simple`, `group`, `calendar`, `event_study`
  - `CallawaySantAnnaResults` with group-time effects and aggregations
  - `GroupTimeEffect` dataclass for individual effects
- **Event study visualization** via `plot_event_study()`
  - Works with `MultiPeriodDiDResults`, `CallawaySantAnnaResults`, or DataFrames
  - Publication-ready formatting with customization options
- **Group effects visualization** via `plot_group_effects()`
- **Parallel trends testing utilities**
  - `check_parallel_trends()` - simple slope-based test
  - `check_parallel_trends_robust()` - Wasserstein distance test
  - `equivalence_test_trends()` - TOST equivalence test

## [0.3.0]

### Added
- **Synthetic Difference-in-Differences** (`SyntheticDiD`)
  - Unit weight optimization for synthetic control
  - Time weight computation for pre-treatment periods
  - Placebo-based and bootstrap inference
  - `SyntheticDiDResults` with weight accessors
- **Multi-period DiD** (`MultiPeriodDiD`)
  - Event-study style estimation with period-specific effects
  - `MultiPeriodDiDResults` with `period_effects` dictionary
  - `PeriodEffect` dataclass for individual period results
- **Data preparation utilities** (`diff_diff.prep`)
  - `generate_did_data()` - synthetic data generation
  - `make_treatment_indicator()` - create treatment from categorical/numeric
  - `make_post_indicator()` - create post-treatment indicator
  - `wide_to_long()` - reshape wide to long format
  - `balance_panel()` - ensure balanced panel data
  - `validate_did_data()` - data validation
  - `summarize_did_data()` - summary statistics by group
  - `create_event_time()` - event time for staggered designs
  - `aggregate_to_cohorts()` - aggregate to cohort means
  - `rank_control_units()` - rank controls by similarity

## [0.2.0]

### Added
- **Two-Way Fixed Effects** (`TwoWayFixedEffects`)
  - Within-transformation for unit and time fixed effects
  - Efficient handling of high-dimensional fixed effects via `absorb`
- **Fixed effects support** in base `DifferenceInDifferences`
  - `fixed_effects` parameter for dummy variable approach
  - `absorb` parameter for within-transformation approach
- **Cluster-robust standard errors**
  - `cluster` parameter for cluster-robust inference
- **Formula interface**
  - R-style formulas like `"outcome ~ treated * post"`
  - Support for covariates in formulas

## [0.1.0]

### Added
- Initial release
- **Basic Difference-in-Differences** (`DifferenceInDifferences`)
  - sklearn-like API with `fit()` method
  - Column name interface for outcome, treatment, time
  - Heteroskedasticity-robust (HC1) standard errors
  - `DiDResults` dataclass with ATT, SE, p-value, confidence intervals
  - `summary()` and `print_summary()` methods
  - `to_dict()` and `to_dataframe()` export methods
  - `is_significant` and `significance_stars` properties

[2.6.0]: https://github.com/igerber/diff-diff/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/igerber/diff-diff/compare/v2.4.3...v2.5.0
[2.4.3]: https://github.com/igerber/diff-diff/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/igerber/diff-diff/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/igerber/diff-diff/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/igerber/diff-diff/compare/v2.3.2...v2.4.0
[2.3.2]: https://github.com/igerber/diff-diff/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/igerber/diff-diff/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/igerber/diff-diff/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/igerber/diff-diff/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/igerber/diff-diff/compare/v2.1.9...v2.2.0
[2.1.9]: https://github.com/igerber/diff-diff/compare/v2.1.8...v2.1.9
[2.1.8]: https://github.com/igerber/diff-diff/compare/v2.1.7...v2.1.8
[2.1.7]: https://github.com/igerber/diff-diff/compare/v2.1.6...v2.1.7
[2.1.6]: https://github.com/igerber/diff-diff/compare/v2.1.5...v2.1.6
[2.1.5]: https://github.com/igerber/diff-diff/compare/v2.1.4...v2.1.5
[2.1.4]: https://github.com/igerber/diff-diff/compare/v2.1.3...v2.1.4
[2.1.3]: https://github.com/igerber/diff-diff/compare/v2.1.2...v2.1.3
[2.1.2]: https://github.com/igerber/diff-diff/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/igerber/diff-diff/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/igerber/diff-diff/compare/v2.0.3...v2.1.0
[2.0.3]: https://github.com/igerber/diff-diff/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/igerber/diff-diff/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/igerber/diff-diff/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/igerber/diff-diff/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/igerber/diff-diff/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/igerber/diff-diff/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/igerber/diff-diff/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/igerber/diff-diff/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/igerber/diff-diff/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/igerber/diff-diff/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/igerber/diff-diff/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/igerber/diff-diff/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/igerber/diff-diff/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/igerber/diff-diff/compare/v0.6.0...v1.0.0
[0.6.0]: https://github.com/igerber/diff-diff/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/igerber/diff-diff/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/igerber/diff-diff/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/igerber/diff-diff/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/igerber/diff-diff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/igerber/diff-diff/releases/tag/v0.1.0
