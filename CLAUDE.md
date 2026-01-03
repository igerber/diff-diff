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
  - `TwoWayFixedEffects` - Panel DiD with unit and time fixed effects (within-transformation)
  - `MultiPeriodDiD` - Event-study style DiD with period-specific treatment effects
  - `SyntheticDiD` - Synthetic control combined with DiD (Arkhangelsky et al. 2021)

- **`diff_diff/staggered.py`** - Staggered adoption DiD estimators:
  - `CallawaySantAnna` - Callaway & Sant'Anna (2021) estimator for heterogeneous treatment timing
  - `CallawaySantAnnaResults` - Results with group-time ATT(g,t) and aggregations
  - `GroupTimeEffect` - Container for individual group-time effects

- **`diff_diff/results.py`** - Dataclass containers for estimation results:
  - `DiDResults`, `MultiPeriodDiDResults`, `SyntheticDiDResults`, `PeriodEffect`
  - Each provides `summary()`, `to_dict()`, `to_dataframe()` methods

- **`diff_diff/visualization.py`** - Plotting functions:
  - `plot_event_study` - Publication-ready event study coefficient plots
  - `plot_group_effects` - Treatment effects by cohort visualization
  - Works with MultiPeriodDiD, CallawaySantAnna, or DataFrames

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

### Dependencies

Core dependencies are numpy, pandas, and scipy only (no statsmodels). The library implements its own OLS, robust standard errors, and inference.
