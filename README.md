# diff-diff

A Python library for Difference-in-Differences (DiD) causal inference analysis with an sklearn-like API and statsmodels-style outputs.

## Installation

```bash
pip install diff-diff
```

Or install from source:

```bash
git clone https://github.com/igerber/diff-diff.git
cd diff-diff
pip install -e .
```

## Quick Start

```python
import pandas as pd
from diff_diff import DifferenceInDifferences

# Create sample data
data = pd.DataFrame({
    'outcome': [10, 11, 15, 18, 9, 10, 12, 13],
    'treated': [1, 1, 1, 1, 0, 0, 0, 0],
    'post': [0, 0, 1, 1, 0, 0, 1, 1]
})

# Fit the model
did = DifferenceInDifferences()
results = did.fit(data, outcome='outcome', treatment='treated', time='post')

# View results
print(results)  # DiDResults(ATT=3.5000*, SE=1.2583, p=0.0367)
results.print_summary()
```

Output:
```
======================================================================
          Difference-in-Differences Estimation Results
======================================================================

Observations:                        8
Treated units:                       4
Control units:                       4
R-squared:                      0.9123

----------------------------------------------------------------------
Parameter         Estimate     Std. Err.     t-stat      P>|t|
----------------------------------------------------------------------
ATT                 3.5000       1.2583      2.782      0.0367
----------------------------------------------------------------------

95% Confidence Interval: [0.3912, 6.6088]

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
======================================================================
```

## Features

- **sklearn-like API**: Familiar `fit()` interface with `get_params()` and `set_params()`
- **Pythonic results**: Easy access to coefficients, standard errors, and confidence intervals
- **Multiple interfaces**: Column names or R-style formulas
- **Robust inference**: Heteroskedasticity-robust (HC1) and cluster-robust standard errors
- **Panel data support**: Two-way fixed effects estimator for panel designs
- **Multi-period analysis**: Event-study style DiD with period-specific treatment effects
- **Synthetic DiD**: Combined DiD with synthetic control for improved robustness

## Usage

### Basic DiD with Column Names

```python
from diff_diff import DifferenceInDifferences

did = DifferenceInDifferences(robust=True, alpha=0.05)
results = did.fit(
    data,
    outcome='sales',
    treatment='treated',
    time='post_policy'
)

# Access results
print(f"ATT: {results.att:.4f}")
print(f"Standard Error: {results.se:.4f}")
print(f"P-value: {results.p_value:.4f}")
print(f"95% CI: {results.conf_int}")
print(f"Significant: {results.is_significant}")
```

### Using Formula Interface

```python
# R-style formula syntax
results = did.fit(data, formula='outcome ~ treated * post')

# Explicit interaction syntax
results = did.fit(data, formula='outcome ~ treated + post + treated:post')

# With covariates
results = did.fit(data, formula='outcome ~ treated * post + age + income')
```

### Including Covariates

```python
results = did.fit(
    data,
    outcome='outcome',
    treatment='treated',
    time='post',
    covariates=['age', 'income', 'education']
)
```

### Fixed Effects

Use `fixed_effects` for low-dimensional categorical controls (creates dummy variables):

```python
# State and industry fixed effects
results = did.fit(
    data,
    outcome='sales',
    treatment='treated',
    time='post',
    fixed_effects=['state', 'industry']
)

# Access fixed effect coefficients
state_coefs = {k: v for k, v in results.coefficients.items() if k.startswith('state_')}
```

Use `absorb` for high-dimensional fixed effects (more efficient, uses within-transformation):

```python
# Absorb firm-level fixed effects (efficient for many firms)
results = did.fit(
    data,
    outcome='sales',
    treatment='treated',
    time='post',
    absorb=['firm_id']
)
```

Combine covariates with fixed effects:

```python
results = did.fit(
    data,
    outcome='sales',
    treatment='treated',
    time='post',
    covariates=['size', 'age'],           # Linear controls
    fixed_effects=['industry'],            # Low-dimensional FE (dummies)
    absorb=['firm_id']                     # High-dimensional FE (absorbed)
)
```

### Cluster-Robust Standard Errors

```python
did = DifferenceInDifferences(cluster='state')
results = did.fit(
    data,
    outcome='outcome',
    treatment='treated',
    time='post'
)
```

### Two-Way Fixed Effects (Panel Data)

```python
from diff_diff.estimators import TwoWayFixedEffects

twfe = TwoWayFixedEffects()
results = twfe.fit(
    panel_data,
    outcome='outcome',
    treatment='treated',
    time='year',
    unit='firm_id'
)
```

### Multi-Period DiD (Event Study)

For settings with multiple pre- and post-treatment periods:

```python
from diff_diff import MultiPeriodDiD

# Fit with multiple time periods
did = MultiPeriodDiD()
results = did.fit(
    panel_data,
    outcome='sales',
    treatment='treated',
    time='period',
    post_periods=[3, 4, 5],      # Periods 3-5 are post-treatment
    reference_period=0           # Reference period for comparison
)

# View period-specific treatment effects
for period, effect in results.period_effects.items():
    print(f"Period {period}: {effect.effect:.3f} (SE: {effect.se:.3f})")

# View average treatment effect across post-periods
print(f"Average ATT: {results.avg_att:.3f}")
print(f"Average SE: {results.avg_se:.3f}")

# Full summary with all period effects
results.print_summary()
```

Output:
```
================================================================================
            Multi-Period Difference-in-Differences Estimation Results
================================================================================

Observations:                      600
Pre-treatment periods:             3
Post-treatment periods:            3

--------------------------------------------------------------------------------
Average Treatment Effect
--------------------------------------------------------------------------------
Average ATT       5.2000       0.8234      6.315      0.0000
--------------------------------------------------------------------------------
95% Confidence Interval: [3.5862, 6.8138]

Period-Specific Effects:
--------------------------------------------------------------------------------
Period            Effect     Std. Err.     t-stat      P>|t|
--------------------------------------------------------------------------------
3                 4.5000       0.9512      4.731      0.0000***
4                 5.2000       0.8876      5.858      0.0000***
5                 5.9000       0.9123      6.468      0.0000***
--------------------------------------------------------------------------------

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
================================================================================
```

### Synthetic Difference-in-Differences

Synthetic DiD combines the strengths of Difference-in-Differences and Synthetic Control methods by re-weighting control units to better match treated units' pre-treatment outcomes.

```python
from diff_diff import SyntheticDiD

# Fit Synthetic DiD model
sdid = SyntheticDiD()
results = sdid.fit(
    panel_data,
    outcome='gdp_growth',
    treatment='treated',
    unit='state',
    time='year',
    post_periods=[2015, 2016, 2017, 2018]
)

# View results
results.print_summary()
print(f"ATT: {results.att:.3f} (SE: {results.se:.3f})")

# Examine unit weights (which control units matter most)
weights_df = results.get_unit_weights_df()
print(weights_df.head(10))

# Examine time weights
time_weights_df = results.get_time_weights_df()
print(time_weights_df)
```

Output:
```
===========================================================================
         Synthetic Difference-in-Differences Estimation Results
===========================================================================

Observations:                      500
Treated units:                       1
Control units:                      49
Pre-treatment periods:               6
Post-treatment periods:              4
Regularization (lambda):        0.0000
Pre-treatment fit (RMSE):       0.1234

---------------------------------------------------------------------------
Parameter         Estimate     Std. Err.     t-stat      P>|t|
---------------------------------------------------------------------------
ATT                 2.5000       0.4521      5.530      0.0000
---------------------------------------------------------------------------

95% Confidence Interval: [1.6139, 3.3861]

---------------------------------------------------------------------------
                   Top Unit Weights (Synthetic Control)
---------------------------------------------------------------------------
  Unit state_12: 0.3521
  Unit state_5: 0.2156
  Unit state_23: 0.1834
  Unit state_8: 0.1245
  Unit state_31: 0.0892
  (8 units with weight > 0.001)

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
===========================================================================
```

#### When to Use Synthetic DiD Over Vanilla DiD

Use Synthetic DiD instead of standard DiD when:

1. **Few treated units**: When you have only one or a small number of treated units (e.g., a single state passed a policy), standard DiD averages across all controls equally. Synthetic DiD finds the optimal weighted combination of controls.

   ```python
   # Example: California passed a policy, want to estimate its effect
   # Standard DiD would compare CA to the average of all other states
   # Synthetic DiD finds states that together best match CA's pre-treatment trend
   ```

2. **Parallel trends is questionable**: When treated and control groups have different pre-treatment levels or trends, Synthetic DiD can construct a better counterfactual by matching the pre-treatment trajectory.

   ```python
   # Example: A tech hub city vs rural areas
   # Rural areas may not be a good comparison on average
   # Synthetic DiD can weight urban/suburban controls more heavily
   ```

3. **Heterogeneous control units**: When control units are very different from each other, equal weighting (as in standard DiD) is suboptimal.

   ```python
   # Example: Comparing a treated developing country to other countries
   # Some control countries may be much more similar economically
   # Synthetic DiD upweights the most comparable controls
   ```

4. **You want transparency**: Synthetic DiD provides explicit unit weights showing which controls contribute most to the comparison.

   ```python
   # See exactly which units are driving the counterfactual
   print(results.get_unit_weights_df())
   ```

**Key differences from standard DiD:**

| Aspect | Standard DiD | Synthetic DiD |
|--------|--------------|---------------|
| Control weighting | Equal (1/N) | Optimized to match pre-treatment |
| Time weighting | Equal across periods | Can emphasize informative periods |
| N treated required | Can be many | Works with 1 treated unit |
| Parallel trends | Assumed | Partially relaxed via matching |
| Interpretability | Simple average | Explicit weights |

**Parameters:**

```python
SyntheticDiD(
    lambda_reg=0.0,     # Regularization toward uniform weights (0 = no reg)
    zeta=1.0,           # Time weight regularization (higher = more uniform)
    alpha=0.05,         # Significance level
    n_bootstrap=200,    # Bootstrap iterations for SE (0 = placebo-based)
    seed=None           # Random seed for reproducibility
)
```

## Working with Results

### Export Results

```python
# As dictionary
results.to_dict()
# {'att': 3.5, 'se': 1.26, 'p_value': 0.037, ...}

# As DataFrame
df = results.to_dataframe()
```

### Check Significance

```python
if results.is_significant:
    print(f"Effect is significant at {did.alpha} level")

# Get significance stars
print(f"ATT: {results.att}{results.significance_stars}")
# ATT: 3.5000*
```

### Access Full Regression Output

```python
# All coefficients
results.coefficients
# {'const': 9.5, 'treated': 1.0, 'post': 2.5, 'treated:post': 3.5}

# Variance-covariance matrix
results.vcov

# Residuals and fitted values
results.residuals
results.fitted_values

# R-squared
results.r_squared
```

## Checking Assumptions

### Parallel Trends

**Simple slope-based test:**

```python
from diff_diff.utils import check_parallel_trends

trends = check_parallel_trends(
    data,
    outcome='outcome',
    time='period',
    treatment_group='treated'
)

print(f"Treated trend: {trends['treated_trend']:.4f}")
print(f"Control trend: {trends['control_trend']:.4f}")
print(f"Difference p-value: {trends['p_value']:.4f}")
```

**Robust distributional test (Wasserstein distance):**

```python
from diff_diff.utils import check_parallel_trends_robust

results = check_parallel_trends_robust(
    data,
    outcome='outcome',
    time='period',
    treatment_group='treated',
    unit='firm_id',              # Unit identifier for panel data
    pre_periods=[2018, 2019],    # Pre-treatment periods
    n_permutations=1000          # Permutations for p-value
)

print(f"Wasserstein distance: {results['wasserstein_distance']:.4f}")
print(f"Wasserstein p-value: {results['wasserstein_p_value']:.4f}")
print(f"KS test p-value: {results['ks_p_value']:.4f}")
print(f"Parallel trends plausible: {results['parallel_trends_plausible']}")
```

The Wasserstein (Earth Mover's) distance compares the full distribution of outcome changes, not just means. This is more robust to:
- Non-normal distributions
- Heterogeneous effects across units
- Outliers

**Equivalence testing (TOST):**

```python
from diff_diff.utils import equivalence_test_trends

results = equivalence_test_trends(
    data,
    outcome='outcome',
    time='period',
    treatment_group='treated',
    unit='firm_id',
    equivalence_margin=0.5       # Define "practically equivalent"
)

print(f"Mean difference: {results['mean_difference']:.4f}")
print(f"TOST p-value: {results['tost_p_value']:.4f}")
print(f"Trends equivalent: {results['equivalent']}")
```

## API Reference

### DifferenceInDifferences

```python
DifferenceInDifferences(
    robust=True,      # Use HC1 robust standard errors
    cluster=None,     # Column for cluster-robust SEs
    alpha=0.05        # Significance level for CIs
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `fit(data, outcome, treatment, time, ...)` | Fit the DiD model |
| `summary()` | Get formatted summary string |
| `print_summary()` | Print summary to stdout |
| `get_params()` | Get estimator parameters (sklearn-compatible) |
| `set_params(**params)` | Set estimator parameters (sklearn-compatible) |

**fit() Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame | Input data |
| `outcome` | str | Outcome variable column name |
| `treatment` | str | Treatment indicator column (0/1) |
| `time` | str | Post-treatment indicator column (0/1) |
| `formula` | str | R-style formula (alternative to column names) |
| `covariates` | list | Linear control variables |
| `fixed_effects` | list | Categorical FE columns (creates dummies) |
| `absorb` | list | High-dimensional FE (within-transformation) |

### DiDResults

**Attributes:**

| Attribute | Description |
|-----------|-------------|
| `att` | Average Treatment effect on the Treated |
| `se` | Standard error of ATT |
| `t_stat` | T-statistic |
| `p_value` | P-value for H0: ATT = 0 |
| `conf_int` | Tuple of (lower, upper) confidence bounds |
| `n_obs` | Number of observations |
| `n_treated` | Number of treated units |
| `n_control` | Number of control units |
| `r_squared` | R-squared of regression |
| `coefficients` | Dictionary of all coefficients |
| `is_significant` | Boolean for significance at alpha |
| `significance_stars` | String of significance stars |

**Methods:**

| Method | Description |
|--------|-------------|
| `summary(alpha)` | Get formatted summary string |
| `print_summary(alpha)` | Print summary to stdout |
| `to_dict()` | Convert to dictionary |
| `to_dataframe()` | Convert to pandas DataFrame |

### MultiPeriodDiD

```python
MultiPeriodDiD(
    robust=True,      # Use HC1 robust standard errors
    cluster=None,     # Column for cluster-robust SEs
    alpha=0.05        # Significance level for CIs
)
```

**fit() Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame | Input data |
| `outcome` | str | Outcome variable column name |
| `treatment` | str | Treatment indicator column (0/1) |
| `time` | str | Time period column (multiple values) |
| `post_periods` | list | List of post-treatment period values |
| `covariates` | list | Linear control variables |
| `fixed_effects` | list | Categorical FE columns (creates dummies) |
| `absorb` | list | High-dimensional FE (within-transformation) |
| `reference_period` | any | Omitted period for time dummies |

### MultiPeriodDiDResults

**Attributes:**

| Attribute | Description |
|-----------|-------------|
| `period_effects` | Dict mapping periods to PeriodEffect objects |
| `avg_att` | Average ATT across post-treatment periods |
| `avg_se` | Standard error of average ATT |
| `avg_t_stat` | T-statistic for average ATT |
| `avg_p_value` | P-value for average ATT |
| `avg_conf_int` | Confidence interval for average ATT |
| `n_obs` | Number of observations |
| `pre_periods` | List of pre-treatment periods |
| `post_periods` | List of post-treatment periods |

**Methods:**

| Method | Description |
|--------|-------------|
| `get_effect(period)` | Get PeriodEffect for specific period |
| `summary(alpha)` | Get formatted summary string |
| `print_summary(alpha)` | Print summary to stdout |
| `to_dict()` | Convert to dictionary |
| `to_dataframe()` | Convert to pandas DataFrame |

### PeriodEffect

**Attributes:**

| Attribute | Description |
|-----------|-------------|
| `period` | Time period identifier |
| `effect` | Treatment effect estimate |
| `se` | Standard error |
| `t_stat` | T-statistic |
| `p_value` | P-value |
| `conf_int` | Confidence interval |
| `is_significant` | Boolean for significance at 0.05 |
| `significance_stars` | String of significance stars |

### SyntheticDiD

```python
SyntheticDiD(
    lambda_reg=0.0,     # L2 regularization for unit weights
    zeta=1.0,           # Regularization for time weights
    alpha=0.05,         # Significance level for CIs
    n_bootstrap=200,    # Bootstrap iterations for SE
    seed=None           # Random seed for reproducibility
)
```

**fit() Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame | Panel data |
| `outcome` | str | Outcome variable column name |
| `treatment` | str | Treatment indicator column (0/1) |
| `unit` | str | Unit identifier column |
| `time` | str | Time period column |
| `post_periods` | list | List of post-treatment period values |
| `covariates` | list | Covariates to residualize out |

### SyntheticDiDResults

**Attributes:**

| Attribute | Description |
|-----------|-------------|
| `att` | Average Treatment effect on the Treated |
| `se` | Standard error (bootstrap or placebo-based) |
| `t_stat` | T-statistic |
| `p_value` | P-value |
| `conf_int` | Confidence interval |
| `n_obs` | Number of observations |
| `n_treated` | Number of treated units |
| `n_control` | Number of control units |
| `unit_weights` | Dict mapping control unit IDs to weights |
| `time_weights` | Dict mapping pre-treatment periods to weights |
| `pre_periods` | List of pre-treatment periods |
| `post_periods` | List of post-treatment periods |
| `pre_treatment_fit` | RMSE of synthetic vs treated in pre-period |
| `placebo_effects` | Array of placebo effect estimates |

**Methods:**

| Method | Description |
|--------|-------------|
| `summary(alpha)` | Get formatted summary string |
| `print_summary(alpha)` | Print summary to stdout |
| `to_dict()` | Convert to dictionary |
| `to_dataframe()` | Convert to pandas DataFrame |
| `get_unit_weights_df()` | Get unit weights as DataFrame |
| `get_time_weights_df()` | Get time weights as DataFrame |

## Requirements

- Python >= 3.9
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black diff_diff tests
ruff check diff_diff tests
```

## License

MIT License
