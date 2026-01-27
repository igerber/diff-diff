# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

diff-diff is a Python library for Difference-in-Differences (DiD) causal inference analysis. It provides sklearn-like estimators with statsmodels-style output for econometric analysis.

## Common Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_estimators.py

# Run a specific test
pytest tests/test_estimators.py::TestDifferenceInDifferences::test_basic_did

# Format code
black diff_diff tests

# Lint code
ruff check diff_diff tests

# Type checking
mypy diff_diff
```

### Rust Backend Commands

```bash
# Build Rust backend for development (requires Rust toolchain)
maturin develop

# Build with release optimizations
maturin develop --release

# Force pure Python mode (disable Rust backend)
DIFF_DIFF_BACKEND=python pytest

# Force Rust mode (fail if Rust not available)
DIFF_DIFF_BACKEND=rust pytest

# Run Rust backend equivalence tests
pytest tests/test_rust_backend.py -v
```

**Note**: As of v2.2.0, the Rust backend uses the pure-Rust `faer` library for linear algebra,
eliminating external BLAS/LAPACK dependencies. This enables Windows wheel builds and simplifies
cross-platform compilation - no OpenBLAS or Intel MKL installation required.

## Architecture

### Module Structure

- **`diff_diff/estimators.py`** - Core estimator classes implementing DiD methods:
  - `DifferenceInDifferences` - Basic 2x2 DiD with formula or column-name interface
  - `MultiPeriodDiD` - Event-study style DiD with period-specific treatment effects
  - Re-exports `TwoWayFixedEffects` and `SyntheticDiD` for backward compatibility

- **`diff_diff/twfe.py`** - Two-Way Fixed Effects estimator:
  - `TwoWayFixedEffects` - Panel DiD with unit and time fixed effects (within-transformation)

- **`diff_diff/synthetic_did.py`** - Synthetic DiD estimator:
  - `SyntheticDiD` - Synthetic control combined with DiD (Arkhangelsky et al. 2021)

- **`diff_diff/staggered.py`** - Staggered adoption DiD main module:
  - `CallawaySantAnna` - Callaway & Sant'Anna (2021) estimator for heterogeneous treatment timing
  - Core estimation methods: `_precompute_structures()`, `_compute_att_gt_fast()`, `fit()`
  - Estimation approaches: `_outcome_regression()`, `_ipw_estimation()`, `_doubly_robust()`
  - Re-exports result and bootstrap classes for backward compatibility

- **`diff_diff/staggered_results.py`** - Result container classes:
  - `GroupTimeEffect` - Container for individual group-time effects
  - `CallawaySantAnnaResults` - Results with group-time ATT(g,t), `summary()`, `to_dataframe()`

- **`diff_diff/staggered_bootstrap.py`** - Bootstrap inference:
  - `CSBootstrapResults` - Bootstrap inference results (SEs, CIs, p-values for all aggregations)
  - `CallawaySantAnnaBootstrapMixin` - Mixin with bootstrap methods
  - `_generate_bootstrap_weights_batch()` - Vectorized weight generation
  - Multiplier bootstrap with Rademacher, Mammen, or Webb weights

- **`diff_diff/staggered_aggregation.py`** - Aggregation methods:
  - `CallawaySantAnnaAggregationMixin` - Mixin with aggregation methods
  - `_aggregate_simple()`, `_aggregate_event_study()`, `_aggregate_by_group()`
  - `_compute_aggregated_se_with_wif()` - SE with weight influence function adjustment

- **`diff_diff/sun_abraham.py`** - Sun-Abraham interaction-weighted estimator:
  - `SunAbraham` - Sun & Abraham (2021) estimator using saturated regression
  - `SunAbrahamResults` - Results with event study effects and cohort weights
  - `SABootstrapResults` - Bootstrap inference results
  - Alternative to Callaway-Sant'Anna with different weighting scheme
  - Useful robustness check when both estimators agree

- **`diff_diff/triple_diff.py`** - Triple Difference (DDD) estimator:
  - `TripleDifference` - Ortiz-Villavicencio & Sant'Anna (2025) estimator for DDD designs
  - `TripleDifferenceResults` - Results with ATT, SEs, cell means, diagnostics
  - `triple_difference()` - Convenience function for quick estimation
  - Regression adjustment, IPW, and doubly robust estimation methods
  - Proper covariate handling (unlike naive DDD implementations)

- **`diff_diff/trop.py`** - Triply Robust Panel (TROP) estimator (v2.1.0):
  - `TROP` - Athey, Imbens, Qu & Viviano (2025) estimator with factor model adjustment
  - `TROPResults` - Results with ATT, factors, loadings, unit/time weights
  - `trop()` - Convenience function for quick estimation
  - Three robustness components: factor adjustment, unit weights, time weights
  - Two estimation methods via `method` parameter:
    - `"twostep"` (default): Per-observation model fitting (Algorithm 2 of paper)
    - `"joint"`: Weighted least squares with homogeneous treatment effect (faster)
  - Automatic rank selection via cross-validation, information criterion, or elbow detection
  - Bootstrap and placebo-based variance estimation

- **`diff_diff/bacon.py`** - Goodman-Bacon decomposition for TWFE diagnostics:
  - `BaconDecomposition` - Decompose TWFE into weighted 2x2 comparisons (Goodman-Bacon 2021)
  - `BaconDecompositionResults` - Results with comparison weights and estimates by type
  - `Comparison2x2` - Individual 2x2 comparison (treated_vs_never, earlier_vs_later, later_vs_earlier)
  - `bacon_decompose()` - Convenience function for quick decomposition
  - Integrated with `TwoWayFixedEffects.decompose()` method

- **`diff_diff/linalg.py`** - Unified linear algebra backend (v1.4.0+):
  - `solve_ols()` - OLS solver with R-style rank deficiency handling
  - `_detect_rank_deficiency()` - Detect linearly dependent columns via pivoted QR
  - `compute_robust_vcov()` - Vectorized HC1 and cluster-robust variance-covariance estimation
  - `compute_r_squared()` - R-squared and adjusted R-squared computation
  - `LinearRegression` - High-level OLS helper class with unified coefficient extraction and inference
  - `InferenceResult` - Dataclass container for coefficient-level inference (SE, t-stat, p-value, CI)
  - Single optimization point for all estimators (reduces code duplication)
  - Cluster-robust SEs use pandas groupby instead of O(n × clusters) loop
  - **Rank deficiency handling** (R-style):
    - Detects rank-deficient matrices using pivoted QR decomposition
    - `rank_deficient_action` parameter: "warn" (default), "error", or "silent"
    - Dropped columns have NaN coefficients (like R's `lm()`)
    - VCoV matrix has NaN for rows/cols of dropped coefficients
    - Warnings include column names when provided

- **`diff_diff/_backend.py`** - Backend detection and configuration (v2.0.0):
  - Detects optional Rust backend availability
  - Handles `DIFF_DIFF_BACKEND` environment variable ('auto', 'python', 'rust')
  - Exports `HAS_RUST_BACKEND` flag and Rust function references
  - Other modules import from here to avoid circular imports with `__init__.py`

- **`rust/`** - Optional Rust backend for accelerated computation (v2.0.0+):
  - **`rust/src/lib.rs`** - PyO3 module definition, exports Python bindings
  - **`rust/src/bootstrap.rs`** - Parallel bootstrap weight generation (Rademacher, Mammen, Webb)
  - **`rust/src/linalg.rs`** - OLS solver (SVD-based) and cluster-robust variance estimation
  - **`rust/src/weights.rs`** - Synthetic control weights and simplex projection
  - **`rust/src/trop.rs`** - TROP estimator acceleration:
    - `compute_unit_distance_matrix()` - Parallel pairwise RMSE distance computation (4-8x speedup)
    - `loocv_grid_search()` - Parallel LOOCV across tuning parameters (10-50x speedup)
    - `bootstrap_trop_variance()` - Parallel bootstrap variance estimation (5-15x speedup)
  - Uses pure-Rust `faer` library for linear algebra (no external BLAS/LAPACK dependencies)
  - Cross-platform: builds on Linux, macOS, and Windows without additional setup
  - Provides 4-8x speedup for SyntheticDiD, 5-20x speedup for TROP

- **`diff_diff/results.py`** - Dataclass containers for estimation results:
  - `DiDResults`, `MultiPeriodDiDResults`, `SyntheticDiDResults`, `PeriodEffect`
  - Each provides `summary()`, `to_dict()`, `to_dataframe()` methods

- **`diff_diff/visualization.py`** - Plotting functions:
  - `plot_event_study` - Publication-ready event study coefficient plots
  - `plot_group_effects` - Treatment effects by cohort visualization
  - `plot_sensitivity` - Honest DiD sensitivity analysis plots (bounds vs M)
  - `plot_honest_event_study` - Event study with honest confidence intervals
  - `plot_bacon` - Bacon decomposition scatter/bar plots (weights vs estimates by comparison type)
  - `plot_power_curve` - Power curve visualization (power vs effect size or sample size)
  - `plot_pretrends_power` - Pre-trends test power curve (power vs violation magnitude)
  - Works with MultiPeriodDiD, CallawaySantAnna, SunAbraham, HonestDiD, BaconDecomposition, PowerAnalysis, PreTrendsPower, or DataFrames

- **`diff_diff/utils.py`** - Statistical utilities:
  - Robust/cluster standard errors (`compute_robust_se`)
  - Parallel trends tests (`check_parallel_trends`, `check_parallel_trends_robust`, `equivalence_test_trends`)
  - Synthetic control weight computation (`compute_synthetic_weights`, `compute_time_weights`)
  - Wild cluster bootstrap (`wild_bootstrap_se`, `WildBootstrapResults`)

- **`diff_diff/diagnostics.py`** - Placebo tests and DiD diagnostics:
  - `run_placebo_test()` - Main dispatcher for different placebo test types
  - `placebo_timing_test()` - Fake treatment timing test
  - `placebo_group_test()` - Fake treatment group test (DiD on never-treated)
  - `permutation_test()` - Permutation-based inference
  - `leave_one_out_test()` - Sensitivity to individual treated units
  - `run_all_placebo_tests()` - Comprehensive suite of diagnostics
  - `PlaceboTestResults` - Dataclass for test results

- **`diff_diff/datasets.py`** - Real-world datasets for teaching and examples:
  - `load_card_krueger()` - Card & Krueger (1994) minimum wage dataset (classic 2x2 DiD)
  - `load_castle_doctrine()` - Castle Doctrine / Stand Your Ground laws (staggered adoption)
  - `load_divorce_laws()` - Unilateral divorce laws (staggered adoption, Stevenson-Wolfers)
  - `load_mpdta()` - Minimum wage panel data from R `did` package (Callaway-Sant'Anna example)
  - `list_datasets()` - List available datasets with descriptions
  - `load_dataset(name)` - Load dataset by name
  - `clear_cache()` - Clear locally cached datasets
  - Datasets are downloaded from public sources and cached locally

- **`diff_diff/honest_did.py`** - Honest DiD sensitivity analysis (Rambachan & Roth 2023):
  - `HonestDiD` - Main class for computing bounds under parallel trends violations
  - `DeltaSD`, `DeltaRM`, `DeltaSDRM` - Restriction classes for smoothness and relative magnitudes
  - `HonestDiDResults` - Results with identified set bounds and robust CIs
  - `SensitivityResults` - Results from sensitivity analysis over M grid
  - `compute_honest_did()` - Convenience function for quick bounds computation
  - `sensitivity_plot()` - Convenience function for plotting sensitivity analysis

- **`diff_diff/power.py`** - Power analysis for study design:
  - `PowerAnalysis` - Main class for analytical power calculations
  - `PowerResults` - Results with MDE, power, sample size
  - `SimulationPowerResults` - Results from Monte Carlo power simulation
  - `simulate_power()` - Simulation-based power for any DiD estimator
  - `compute_mde()`, `compute_power()`, `compute_sample_size()` - Convenience functions

- **`diff_diff/pretrends.py`** - Pre-trends power analysis (Roth 2022):
  - `PreTrendsPower` - Main class for assessing informativeness of pre-trends tests
  - `PreTrendsPowerResults` - Results with power and minimum detectable violation (MDV)
  - `PreTrendsPowerCurve` - Power curve across violation magnitudes with plot method
  - `compute_pretrends_power()` - Convenience function for quick power computation
  - `compute_mdv()` - Convenience function for minimum detectable violation
  - Violation types: 'linear', 'constant', 'last_period', 'custom'
  - Integrates with HonestDiD for comprehensive sensitivity analysis

- **`diff_diff/prep.py`** - Data preparation utilities (core functions):
  - `make_treatment_indicator`, `make_post_indicator` - Create binary indicators
  - `wide_to_long`, `balance_panel` - Panel data reshaping
  - `validate_did_data`, `summarize_did_data` - Data validation and summary
  - `create_event_time` - Create event-time column for staggered adoption designs
  - `aggregate_to_cohorts` - Aggregate unit-level data to cohort means
  - `rank_control_units` - Rank control units by suitability for DiD/Synthetic control
  - Re-exports all functions from `prep_dgp.py` for backward compatibility

- **`diff_diff/prep_dgp.py`** - Data generation functions (DGP):
  - `generate_did_data` - Create synthetic data with known treatment effect (basic 2x2 DiD)
  - `generate_staggered_data` - Staggered adoption data for CallawaySantAnna/SunAbraham
  - `generate_factor_data` - Factor model data for TROP/SyntheticDiD
  - `generate_ddd_data` - Triple Difference (DDD) design data
  - `generate_panel_data` - Panel data with optional parallel trends violations
  - `generate_event_study_data` - Event study data with simultaneous treatment

### Key Design Patterns

1. **sklearn-like API**: Estimators use `fit()` method, `get_params()`/`set_params()` for configuration
2. **Formula interface**: Supports R-style formulas like `"outcome ~ treated * post"`
3. **Fixed effects handling**:
   - `fixed_effects` parameter creates dummy variables (for low-dimensional FE)
   - `absorb` parameter uses within-transformation (for high-dimensional FE)
4. **Results objects**: Rich dataclass objects with statistical properties (`is_significant`, `significance_stars`)
5. **Unified linear algebra backend**: All estimators use `linalg.py` for OLS and variance estimation
6. **Estimator inheritance**: Understanding inheritance prevents consistency bugs
   ```
   DifferenceInDifferences (base class)
   ├── TwoWayFixedEffects (inherits get_params/set_params)
   └── MultiPeriodDiD (inherits get_params/set_params)

   Standalone estimators (each has own get_params/set_params):
   ├── CallawaySantAnna
   ├── SunAbraham
   ├── TripleDifference
   ├── TROP
   ├── SyntheticDiD
   └── BaconDecomposition
   ```
   When adding params to `DifferenceInDifferences.get_params()`, subclasses inherit automatically.
   Standalone estimators must be updated individually.

### Performance Architecture (v1.4.0)

diff-diff achieved significant performance improvements in v1.4.0, now **faster than R** at all scales. Key optimizations:

#### Unified `linalg.py` Backend

All estimators use a single optimized OLS/SE implementation:

- **R-style rank deficiency handling**: Uses pivoted QR to detect linearly dependent columns, drops them, sets NaN for their coefficients, and emits informative warnings (following R's `lm()` approach)
- **Vectorized cluster-robust SE**: Uses pandas groupby aggregation instead of O(n × clusters) Python loop
- **Single optimization point**: Changes to `linalg.py` benefit all estimators

```python
# All estimators import from linalg.py
from diff_diff.linalg import solve_ols, compute_robust_vcov

# Example usage (warns on rank deficiency, sets NaN for dropped coefficients)
coefficients, residuals, vcov = solve_ols(X, y, cluster_ids=cluster_ids)

# Suppress warning or raise error:
coefficients, residuals, vcov = solve_ols(X, y, rank_deficient_action="silent")  # no warning
coefficients, residuals, vcov = solve_ols(X, y, rank_deficient_action="error")   # raises ValueError

# At estimator level (DifferenceInDifferences, MultiPeriodDiD):
from diff_diff import DifferenceInDifferences
did = DifferenceInDifferences(rank_deficient_action="error")   # raises on collinear data
did = DifferenceInDifferences(rank_deficient_action="silent")  # no warning
```

#### CallawaySantAnna Optimizations (`staggered.py`)

- **Pre-computed data structures**: `_precompute_structures()` creates wide-format outcome matrix and cohort masks once
- **Vectorized ATT(g,t)**: `_compute_att_gt_fast()` uses numpy operations (23x faster than loop-based)
- **Batch bootstrap weights**: `_generate_bootstrap_weights_batch()` generates all weights at once
- **Matrix-based bootstrap**: Bootstrap iterations use matrix operations instead of nested loops (26x faster)

#### Performance Results

| Estimator | v1.3 (10K scale) | v1.4 (10K scale) | vs R |
|-----------|------------------|------------------|------|
| BasicDiD/TWFE | 0.835s | 0.011s | **4x faster than R** |
| CallawaySantAnna | 2.234s | 0.109s | **8x faster than R** |
| SyntheticDiD | Already optimized | N/A | **37x faster than R** |

See `docs/performance-plan.md` for full optimization details and `docs/benchmarks.rst` for validation results.

### Documentation

- **`docs/methodology/REGISTRY.md`** - Methodology Registry:
  - Academic foundations and citations for each estimator
  - Key implementation requirements (equations, SE formulas, edge cases)
  - Reference implementations (R packages, Stata commands)
  - Requirements checklists for validation
  - **Must be consulted before implementing methodology-related changes**
  - **Must be updated when deviating from reference implementations**

- **`docs/tutorials/`** - Jupyter notebook tutorials:
  - `01_basic_did.ipynb` - Basic 2x2 DiD, covariates, fixed effects, wild bootstrap
  - `02_staggered_did.ipynb` - Staggered adoption with Callaway-Sant'Anna, bootstrap inference
  - `03_synthetic_did.ipynb` - Synthetic DiD with unit/time weights
  - `04_parallel_trends.ipynb` - Parallel trends testing and diagnostics
  - `05_honest_did.ipynb` - Honest DiD sensitivity analysis for parallel trends violations
  - `06_power_analysis.ipynb` - Power analysis for study design, MDE, simulation-based power
  - `07_pretrends_power.ipynb` - Pre-trends power analysis (Roth 2022), MDV, power curves
  - `08_triple_diff.ipynb` - Triple Difference (DDD) estimation with proper covariate handling
  - `09_real_world_examples.ipynb` - Real-world data examples (Card-Krueger, Castle Doctrine, Divorce Laws)
  - `10_trop.ipynb` - Triply Robust Panel (TROP) estimation with factor model adjustment

### Benchmarks

- **`benchmarks/`** - Validation benchmarks against R packages:
  - `run_benchmarks.py` - Main orchestrator for running all benchmarks
  - `compare_results.py` - Result comparison utilities
  - `R/` - R benchmark scripts (did, synthdid, fixest, HonestDiD)
  - `python/` - Python benchmark scripts mirroring R scripts
  - `data/synthetic/` - Generated test data (not committed, use `--generate-data-only`)
  - `results/` - JSON output files (not committed)

Run benchmarks:
```bash
# Generate synthetic data first
python benchmarks/run_benchmarks.py --generate-data-only

# Run all benchmarks
python benchmarks/run_benchmarks.py --all

# Run specific estimator
python benchmarks/run_benchmarks.py --estimator callaway
```

See `docs/benchmarks.rst` for full methodology and validation results.

### Test Structure

Tests mirror the source modules:
- `tests/test_estimators.py` - Tests for DifferenceInDifferences, TWFE, MultiPeriodDiD, SyntheticDiD
- `tests/test_staggered.py` - Tests for CallawaySantAnna
- `tests/test_sun_abraham.py` - Tests for SunAbraham interaction-weighted estimator
- `tests/test_triple_diff.py` - Tests for Triple Difference (DDD) estimator
- `tests/test_trop.py` - Tests for Triply Robust Panel (TROP) estimator
- `tests/test_bacon.py` - Tests for Goodman-Bacon decomposition
- `tests/test_linalg.py` - Tests for unified OLS backend, robust variance estimation, LinearRegression helper, and InferenceResult
- `tests/test_utils.py` - Tests for parallel trends, robust SE, synthetic weights
- `tests/test_diagnostics.py` - Tests for placebo tests
- `tests/test_wild_bootstrap.py` - Tests for wild cluster bootstrap
- `tests/test_prep.py` - Tests for data preparation utilities
- `tests/test_visualization.py` - Tests for plotting functions
- `tests/test_honest_did.py` - Tests for Honest DiD sensitivity analysis
- `tests/test_power.py` - Tests for power analysis
- `tests/test_pretrends.py` - Tests for pre-trends power analysis
- `tests/test_datasets.py` - Tests for dataset loading functions

### Test Writing Guidelines

**For fallback/error handling paths:**
- Don't just test that code runs without exception
- Assert the expected behavior actually occurred
- Bad: `result = func(bad_input)` (only tests no crash)
- Good: `result = func(bad_input); assert np.isnan(result.coef)` (tests behavior)

**For new parameters:**
- Test parameter appears in `get_params()` output
- Test `set_params()` modifies the attribute
- Test parameter actually affects behavior (not just stored)

**For warnings:**
- Capture warnings with `warnings.catch_warnings(record=True)`
- Assert warning message was emitted
- Assert the warned-about behavior occurred

### Dependencies

Core dependencies are numpy, pandas, and scipy only (no statsmodels). The library implements its own OLS, robust standard errors, and inference.

## Documentation Requirements

When implementing new functionality, **always include accompanying documentation updates**:

### For New Estimators or Major Features

1. **README.md** - Add:
   - Feature mention in the features list
   - Full usage section with code examples
   - Parameter documentation table
   - API reference section (constructor params, fit() params, results attributes/methods)
   - Scholarly references if applicable

2. **docs/api/*.rst** - Add:
   - RST documentation with `autoclass` directives
   - Method summaries
   - References to academic papers

3. **docs/tutorials/*.ipynb** - Update relevant tutorial or create new one:
   - Working code examples
   - Explanation of when/why to use the feature
   - Comparison with related functionality

4. **CLAUDE.md** - Update:
   - Module structure section
   - Test structure section
   - Any relevant design patterns

5. **ROADMAP.md** - Update:
   - Move implemented features from planned to current status
   - Update version numbers

### For Bug Fixes or Minor Enhancements

- Update relevant docstrings
- Add/update tests
- Update CHANGELOG.md (if exists)
- **If methodology-related**: Update `docs/methodology/REGISTRY.md` edge cases section

### Scholarly References

For methods based on academic papers, always include:
- Full citation in README.md references section
- Reference in RST docs with paper details
- Citation in tutorial summary

Example format:
```
Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in
event studies with heterogeneous treatment effects. *Journal of Econometrics*,
225(2), 175-199.
```

## Development Checklists

### Adding a New Parameter to Estimators

When adding a new `__init__` parameter that should be available across estimators:

1. **Implementation** (for each affected estimator):
   - [ ] Add to `__init__` signature with default value
   - [ ] Store as `self.param_name`
   - [ ] Add to `get_params()` return dict
   - [ ] Handle in `set_params()` (usually automatic via `hasattr`)

2. **Consistency** - apply to all applicable estimators per the **Estimator inheritance** map above

3. **Testing**:
   - [ ] Test `get_params()` includes new param
   - [ ] Test parameter affects estimator behavior
   - [ ] Test with non-default value

4. **Documentation**:
   - [ ] Update docstring in all affected classes
   - [ ] Update CLAUDE.md if it's a key design pattern

### Implementing Methodology-Critical Code

When implementing or modifying code that affects statistical methodology (estimators, SE calculation, inference, edge case handling):

1. **Before coding - consult the Methodology Registry**:
   - [ ] Read the relevant estimator section in `docs/methodology/REGISTRY.md`
   - [ ] Identify the reference implementation(s) listed
   - [ ] Note the edge case handling requirements

2. **During implementation**:
   - [ ] Follow the documented equations and formulas
   - [ ] Match reference implementation behavior for standard cases
   - [ ] For edge cases: either match reference OR document deviation

3. **When deviating from reference implementations**:
   - [ ] Add a **Note** in the Methodology Registry explaining the deviation
   - [ ] Include rationale (e.g., "defensive enhancement", "R errors here")
   - [ ] Ensure the deviation is an improvement, not a bug

4. **Testing methodology-aligned behavior**:
   - [ ] Test that edge cases produce documented behavior (NaN, warning, etc.)
   - [ ] Assert warnings are raised (not just captured)
   - [ ] Assert the warned-about behavior actually occurred
   - [ ] For NaN results: assert `np.isnan()`, don't just check "no exception"

### Adding Warning/Error/Fallback Handling

When adding code that emits warnings or handles errors:

1. **Consult Methodology Registry first**:
   - [ ] Check if behavior is documented in edge cases section
   - [ ] If not documented, add it before implementing

2. **Verify behavior matches message**:
   - [ ] Manually trace the code path after warning/error
   - [ ] Confirm the stated behavior actually occurs

3. **Write behavioral tests**:
   - [ ] Don't just test "no exception raised"
   - [ ] Assert the expected outcome occurred
   - [ ] For fallbacks: verify fallback behavior was applied
   - [ ] Example: If warning says "setting NaN", assert `np.any(np.isnan(result))`

4. **Protect arithmetic operations**:
   - [ ] Wrap ALL related operations in `np.errstate()`, not just the final one
   - [ ] Include division, matrix multiplication, and any operation that can overflow/underflow

### Reviewing New Features or Code Paths

When reviewing PRs that add new features, modes, or code paths (learned from PR #97 analysis):

1. **Edge Case Coverage**:
   - [ ] Empty result sets (no matching data for a filter condition)
   - [ ] NaN/Inf propagation through ALL inference fields (SE, t-stat, p-value, CI)
   - [ ] Parameter interactions (e.g., new param × existing aggregation methods)
   - [ ] Control/comparison group composition for all code paths

2. **Documentation Completeness**:
   - [ ] All new parameters have docstrings with type, default, and description
   - [ ] Methodology docs match implementation behavior (equations, edge cases)
   - [ ] Edge cases documented in `docs/methodology/REGISTRY.md`

3. **Logic Audit for New Code Paths**:
   - [ ] When adding new modes (like `base_period="varying"`), trace ALL downstream effects
   - [ ] Check aggregation methods handle the new mode correctly
   - [ ] Check bootstrap/inference methods handle the new mode correctly
   - [ ] Explicitly test control group composition in new code paths

4. **Pattern Consistency**:
   - [ ] Search for similar patterns in codebase (e.g., `t_stat = x / se if se > 0 else ...`)
   - [ ] Ensure new code follows established patterns or updates ALL instances
   - [ ] If fixing a pattern, grep for ALL occurrences first:
     ```bash
     grep -n "if.*se.*> 0.*else" diff_diff/*.py
     ```

### Fixing Bugs Across Multiple Locations

When a bug fix involves a pattern that appears in multiple places (learned from PR #97 analysis):

1. **Find All Instances First**:
   - [ ] Use grep/search to find ALL occurrences of the pattern before fixing
   - [ ] Document the locations found (file:line)
   - [ ] Example: `t_stat = effect / se if se > 0 else 0.0` appeared in 7 locations

2. **Fix Comprehensively in One Round**:
   - [ ] Fix ALL instances in the same PR/commit
   - [ ] Create a test that covers all locations
   - [ ] Don't fix incrementally across multiple review rounds

3. **Regression Test the Fix**:
   - [ ] Verify fix doesn't break other code paths
   - [ ] For early-return fixes: ensure downstream code still runs when needed
   - [ ] Example: Bootstrap early return must still compute per-effect SEs

4. **Common Patterns to Watch For**:
   - `if se > 0 else 0.0` → should be `else np.nan` for undefined statistics
   - `if len(data) > 0 else return` → check what downstream code expects
   - `mask = (condition)` → verify mask logic for all parameter combinations

### Pre-Merge Review Checklist

Final checklist before approving a PR:

1. **Behavioral Completeness**:
   - [ ] Happy path tested
   - [ ] Edge cases tested (empty data, NaN inputs, boundary conditions)
   - [ ] Error/warning paths tested with behavioral assertions

2. **Inference Field Consistency**:
   - [ ] If one inference field (SE, t-stat, p-value) can be NaN, all related fields handle it
   - [ ] Aggregation methods propagate NaN correctly
   - [ ] Bootstrap methods handle NaN in base estimates

3. **Documentation Sync**:
   - [ ] Docstrings updated for all changed signatures
   - [ ] README updated if user-facing behavior changes
   - [ ] REGISTRY.md updated if methodology edge cases change

## Task Implementation Workflow

When implementing features or fixes, follow this workflow to ensure quality and catch issues early:

### Phase 1: Planning
- Use `EnterPlanMode` for non-trivial tasks
- Consult `docs/methodology/REGISTRY.md` for methodology-critical code
- Identify all files and code paths that will be affected
- For bug fixes: grep for the pattern first to find ALL occurrences

### Phase 2: Implementation
- Follow the relevant development checklists above
- Write tests alongside implementation (not after)
- For bug fixes: fix ALL occurrences in the same commit
- For new parameters: ensure all aggregation/bootstrap paths handle them

### Phase 3: Pre-Merge Review
**Run `/pre-merge-check` before submitting**. This skill will:
1. Run automated pattern checks on changed files (NaN handling, etc.)
2. Check for missing test coverage
3. Display context-specific checklist items based on what changed
4. Optionally run the test suite

Address any warnings before proceeding.

### Phase 4: Submit
- Use `/submit-pr` to create the PR
- Automated AI review will run additional methodology and edge case checks
- The PR template will prompt for methodology references if applicable

### Quick Reference: Common Patterns to Check

Before submitting methodology changes, verify these patterns:

```bash
# Find potential NaN handling issues (should use np.nan, not 0.0)
grep -n "if.*se.*>.*0.*else 0" diff_diff/*.py

# Find all t_stat calculations to ensure consistency
grep -n "t_stat.*=" diff_diff/*.py

# Find all inference field assignments
grep -n "\(se\|t_stat\|p_value\|ci_lower\|ci_upper\).*=" diff_diff/*.py | head -30
```
