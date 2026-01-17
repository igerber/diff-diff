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

# Run Rust unit tests
cd rust && cargo test

# Force pure Python mode (disable Rust backend)
DIFF_DIFF_BACKEND=python pytest

# Force Rust mode (fail if Rust not available)
DIFF_DIFF_BACKEND=rust pytest

# Run Rust backend equivalence tests
pytest tests/test_rust_backend.py -v
```

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

- **`diff_diff/staggered.py`** - Staggered adoption DiD estimators:
  - `CallawaySantAnna` - Callaway & Sant'Anna (2021) estimator for heterogeneous treatment timing
  - `CallawaySantAnnaResults` - Results with group-time ATT(g,t) and aggregations
  - `CSBootstrapResults` - Bootstrap inference results (SEs, CIs, p-values for all aggregations)
  - `GroupTimeEffect` - Container for individual group-time effects
  - Multiplier bootstrap with Rademacher, Mammen, or Webb weights

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

- **`diff_diff/bacon.py`** - Goodman-Bacon decomposition for TWFE diagnostics:
  - `BaconDecomposition` - Decompose TWFE into weighted 2x2 comparisons (Goodman-Bacon 2021)
  - `BaconDecompositionResults` - Results with comparison weights and estimates by type
  - `Comparison2x2` - Individual 2x2 comparison (treated_vs_never, earlier_vs_later, later_vs_earlier)
  - `bacon_decompose()` - Convenience function for quick decomposition
  - Integrated with `TwoWayFixedEffects.decompose()` method

- **`diff_diff/linalg.py`** - Unified linear algebra backend (v1.4.0+):
  - `solve_ols()` - OLS solver using scipy's gelsy LAPACK driver (QR-based, faster than SVD)
  - `compute_robust_vcov()` - Vectorized HC1 and cluster-robust variance-covariance estimation
  - `compute_r_squared()` - R-squared and adjusted R-squared computation
  - `LinearRegression` - High-level OLS helper class with unified coefficient extraction and inference
  - `InferenceResult` - Dataclass container for coefficient-level inference (SE, t-stat, p-value, CI)
  - Single optimization point for all estimators (reduces code duplication)
  - Cluster-robust SEs use pandas groupby instead of O(n × clusters) loop

- **`diff_diff/_backend.py`** - Backend detection and configuration (v2.0.0):
  - Detects optional Rust backend availability
  - Handles `DIFF_DIFF_BACKEND` environment variable ('auto', 'python', 'rust')
  - Exports `HAS_RUST_BACKEND` flag and Rust function references
  - Other modules import from here to avoid circular imports with `__init__.py`

- **`rust/`** - Optional Rust backend for accelerated computation (v2.0.0):
  - **`rust/src/lib.rs`** - PyO3 module definition, exports Python bindings
  - **`rust/src/bootstrap.rs`** - Parallel bootstrap weight generation (Rademacher, Mammen, Webb)
  - **`rust/src/linalg.rs`** - OLS solver and cluster-robust variance estimation
  - **`rust/src/weights.rs`** - Synthetic control weights and simplex projection
  - Uses ndarray-linalg with OpenBLAS (Linux/macOS) or Intel MKL (Windows)
  - Provides 4-8x speedup for SyntheticDiD, minimal benefit for other estimators

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

- **`diff_diff/prep.py`** - Data preparation utilities:
  - `generate_did_data` - Create synthetic data with known treatment effect
  - `make_treatment_indicator`, `make_post_indicator` - Create binary indicators
  - `wide_to_long`, `balance_panel` - Panel data reshaping
  - `validate_did_data`, `summarize_did_data` - Data validation and summary
  - `create_event_time` - Create event-time column for staggered adoption designs
  - `aggregate_to_cohorts` - Aggregate unit-level data to cohort means
  - `rank_control_units` - Rank control units by suitability for DiD/Synthetic control

### Key Design Patterns

1. **sklearn-like API**: Estimators use `fit()` method, `get_params()`/`set_params()` for configuration
2. **Formula interface**: Supports R-style formulas like `"outcome ~ treated * post"`
3. **Fixed effects handling**:
   - `fixed_effects` parameter creates dummy variables (for low-dimensional FE)
   - `absorb` parameter uses within-transformation (for high-dimensional FE)
4. **Results objects**: Rich dataclass objects with statistical properties (`is_significant`, `significance_stars`)
5. **Unified linear algebra backend**: All estimators use `linalg.py` for OLS and variance estimation

### Performance Architecture (v1.4.0)

diff-diff achieved significant performance improvements in v1.4.0, now **faster than R** at all scales. Key optimizations:

#### Unified `linalg.py` Backend

All estimators use a single optimized OLS/SE implementation:

- **scipy.linalg.lstsq with 'gelsy' driver**: QR-based solving, faster than NumPy's default SVD-based solver
- **Vectorized cluster-robust SE**: Uses pandas groupby aggregation instead of O(n × clusters) Python loop
- **Single optimization point**: Changes to `linalg.py` benefit all estimators

```python
# All estimators import from linalg.py
from diff_diff.linalg import solve_ols, compute_robust_vcov

# Example usage
coefficients, residuals, vcov = solve_ols(X, y, cluster_ids=cluster_ids)
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
