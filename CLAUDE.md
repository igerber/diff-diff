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

- **`diff_diff/bacon.py`** - Goodman-Bacon decomposition for TWFE diagnostics:
  - `BaconDecomposition` - Decompose TWFE into weighted 2x2 comparisons (Goodman-Bacon 2021)
  - `BaconDecompositionResults` - Results with comparison weights and estimates by type
  - `Comparison2x2` - Individual 2x2 comparison (treated_vs_never, earlier_vs_later, later_vs_earlier)
  - `bacon_decompose()` - Convenience function for quick decomposition
  - Integrated with `TwoWayFixedEffects.decompose()` method

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
  - Works with MultiPeriodDiD, CallawaySantAnna, HonestDiD, BaconDecomposition, PowerAnalysis, or DataFrames

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

### Documentation

- **`docs/tutorials/`** - Jupyter notebook tutorials:
  - `01_basic_did.ipynb` - Basic 2x2 DiD, covariates, fixed effects, wild bootstrap
  - `02_staggered_did.ipynb` - Staggered adoption with Callaway-Sant'Anna, bootstrap inference
  - `03_synthetic_did.ipynb` - Synthetic DiD with unit/time weights
  - `04_parallel_trends.ipynb` - Parallel trends testing and diagnostics
  - `05_honest_did.ipynb` - Honest DiD sensitivity analysis for parallel trends violations
  - `06_power_analysis.ipynb` - Power analysis for study design, MDE, simulation-based power

### Test Structure

Tests mirror the source modules:
- `tests/test_estimators.py` - Tests for DifferenceInDifferences, TWFE, MultiPeriodDiD, SyntheticDiD
- `tests/test_staggered.py` - Tests for CallawaySantAnna
- `tests/test_bacon.py` - Tests for Goodman-Bacon decomposition
- `tests/test_utils.py` - Tests for parallel trends, robust SE, synthetic weights
- `tests/test_diagnostics.py` - Tests for placebo tests
- `tests/test_wild_bootstrap.py` - Tests for wild cluster bootstrap
- `tests/test_prep.py` - Tests for data preparation utilities
- `tests/test_visualization.py` - Tests for plotting functions
- `tests/test_honest_did.py` - Tests for Honest DiD sensitivity analysis
- `tests/test_power.py` - Tests for power analysis

### Dependencies

Core dependencies are numpy, pandas, and scipy only (no statsmodels). The library implements its own OLS, robust standard errors, and inference.
