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
print(results)  # DiDResults(ATT=3.0000, SE=1.7321, p=0.1583)
results.print_summary()
```

Output:
```
======================================================================
             Difference-in-Differences Estimation Results
======================================================================

Observations:                      8
Treated units:                     4
Control units:                     4
R-squared:                    0.9055

----------------------------------------------------------------------
Parameter           Estimate    Std. Err.     t-stat      P>|t|
----------------------------------------------------------------------
ATT                   3.0000       1.7321      1.732     0.1583
----------------------------------------------------------------------

95% Confidence Interval: [-1.8089, 7.8089]

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
======================================================================
```

## Features

- **sklearn-like API**: Familiar `fit()` interface with `get_params()` and `set_params()`
- **Pythonic results**: Easy access to coefficients, standard errors, and confidence intervals
- **Multiple interfaces**: Column names or R-style formulas
- **Robust inference**: Heteroskedasticity-robust (HC1) and cluster-robust standard errors
- **Wild cluster bootstrap**: Valid inference with few clusters (<50) using Rademacher, Webb, or Mammen weights
- **Panel data support**: Two-way fixed effects estimator for panel designs
- **Multi-period analysis**: Event-study style DiD with period-specific treatment effects
- **Staggered adoption**: Callaway-Sant'Anna (2021) estimator for heterogeneous treatment timing
- **Synthetic DiD**: Combined DiD with synthetic control for improved robustness
- **Event study plots**: Publication-ready visualization of treatment effects
- **Parallel trends testing**: Multiple methods including equivalence tests
- **Placebo tests**: Comprehensive diagnostics including fake timing, fake group, permutation, and leave-one-out tests
- **Honest DiD sensitivity analysis**: Rambachan-Roth (2023) bounds and breakdown analysis for parallel trends violations
- **Data prep utilities**: Helper functions for common data preparation tasks

## Tutorials

We provide Jupyter notebook tutorials in `docs/tutorials/`:

| Notebook | Description |
|----------|-------------|
| `01_basic_did.ipynb` | Basic 2x2 DiD, formula interface, covariates, fixed effects, cluster-robust SE, wild bootstrap |
| `02_staggered_did.ipynb` | Staggered adoption with Callaway-Sant'Anna, group-time effects, aggregation methods |
| `03_synthetic_did.ipynb` | Synthetic DiD, unit/time weights, inference methods, regularization |
| `04_parallel_trends.ipynb` | Testing parallel trends, equivalence tests, placebo tests, diagnostics |
| `05_honest_did.ipynb` | Honest DiD sensitivity analysis, bounds, breakdown values, visualization |

## Data Preparation

diff-diff provides utility functions to help prepare your data for DiD analysis. These functions handle common data transformation tasks like creating treatment indicators, reshaping panel data, and validating data formats.

### Generate Sample Data

Create synthetic data with a known treatment effect for testing and learning:

```python
from diff_diff import generate_did_data, DifferenceInDifferences

# Generate panel data with 100 units, 4 periods, and a treatment effect of 5
data = generate_did_data(
    n_units=100,
    n_periods=4,
    treatment_effect=5.0,
    treatment_fraction=0.5,  # 50% of units are treated
    treatment_period=2,       # Treatment starts at period 2
    seed=42
)

# Verify the estimator recovers the treatment effect
did = DifferenceInDifferences()
results = did.fit(data, outcome='outcome', treatment='treated', time='post')
print(f"Estimated ATT: {results.att:.2f} (true: 5.0)")
```

### Create Treatment Indicators

Convert categorical variables or numeric thresholds to binary treatment indicators:

```python
from diff_diff import make_treatment_indicator

# From categorical variable
df = make_treatment_indicator(
    data,
    column='state',
    treated_values=['CA', 'NY', 'TX']  # These states are treated
)

# From numeric threshold (e.g., firms above median size)
df = make_treatment_indicator(
    data,
    column='firm_size',
    threshold=data['firm_size'].median()
)

# Treat units below threshold
df = make_treatment_indicator(
    data,
    column='income',
    threshold=50000,
    above_threshold=False  # Units with income <= 50000 are treated
)
```

### Create Post-Treatment Indicators

Convert time/date columns to binary post-treatment indicators:

```python
from diff_diff import make_post_indicator

# From specific post-treatment periods
df = make_post_indicator(
    data,
    time_column='year',
    post_periods=[2020, 2021, 2022]
)

# From treatment start date
df = make_post_indicator(
    data,
    time_column='year',
    treatment_start=2020  # All years >= 2020 are post-treatment
)

# Works with datetime columns
df = make_post_indicator(
    data,
    time_column='date',
    treatment_start='2020-01-01'
)
```

### Reshape Wide to Long Format

Convert wide-format data (one row per unit, multiple time columns) to long format:

```python
from diff_diff import wide_to_long

# Wide format: columns like sales_2019, sales_2020, sales_2021
wide_df = pd.DataFrame({
    'firm_id': [1, 2, 3],
    'industry': ['tech', 'retail', 'tech'],
    'sales_2019': [100, 150, 200],
    'sales_2020': [110, 160, 210],
    'sales_2021': [120, 170, 220]
})

# Convert to long format for DiD
long_df = wide_to_long(
    wide_df,
    value_columns=['sales_2019', 'sales_2020', 'sales_2021'],
    id_column='firm_id',
    time_name='year',
    value_name='sales',
    time_values=[2019, 2020, 2021]
)
# Result: 9 rows (3 firms × 3 years), columns: firm_id, year, sales, industry
```

### Balance Panel Data

Ensure all units have observations for all time periods:

```python
from diff_diff import balance_panel

# Keep only units with complete data (drop incomplete units)
balanced = balance_panel(
    data,
    unit_column='firm_id',
    time_column='year',
    method='inner'
)

# Include all unit-period combinations (creates NaN for missing)
balanced = balance_panel(
    data,
    unit_column='firm_id',
    time_column='year',
    method='outer'
)

# Fill missing values
balanced = balance_panel(
    data,
    unit_column='firm_id',
    time_column='year',
    method='fill',
    fill_value=0  # Or None for forward/backward fill
)
```

### Validate Data

Check that your data meets DiD requirements before fitting:

```python
from diff_diff import validate_did_data

# Validate and get informative error messages
result = validate_did_data(
    data,
    outcome='sales',
    treatment='treated',
    time='post',
    unit='firm_id',      # Optional: for panel-specific validation
    raise_on_error=False  # Return dict instead of raising
)

if result['valid']:
    print("Data is ready for DiD analysis!")
    print(f"Summary: {result['summary']}")
else:
    print("Issues found:")
    for error in result['errors']:
        print(f"  - {error}")

for warning in result['warnings']:
    print(f"Warning: {warning}")
```

### Summarize Data by Groups

Get summary statistics for each treatment-time cell:

```python
from diff_diff import summarize_did_data

summary = summarize_did_data(
    data,
    outcome='sales',
    treatment='treated',
    time='post'
)
print(summary)
```

Output:
```
                        n      mean       std       min       max
Control - Pre        250  100.5000   15.2340   65.0000  145.0000
Control - Post       250  105.2000   16.1230   68.0000  152.0000
Treated - Pre        250  101.2000   14.8900   67.0000  143.0000
Treated - Post       250  115.8000   17.5600   72.0000  165.0000
DiD Estimate           -    9.9000         -         -         -
```

### Create Event Time for Staggered Designs

For designs where treatment occurs at different times:

```python
from diff_diff import create_event_time

# Add event-time column relative to treatment timing
df = create_event_time(
    data,
    time_column='year',
    treatment_time_column='treatment_year'
)
# Result: event_time = -2, -1, 0, 1, 2 relative to treatment
```

### Aggregate to Cohort Means

Aggregate unit-level data for visualization:

```python
from diff_diff import aggregate_to_cohorts

cohort_data = aggregate_to_cohorts(
    data,
    unit_column='firm_id',
    time_column='year',
    treatment_column='treated',
    outcome='sales'
)
# Result: mean outcome by treatment group and period
```

### Rank Control Units

Select the best control units for DiD or Synthetic DiD analysis by ranking them based on pre-treatment outcome similarity:

```python
from diff_diff import rank_control_units, generate_did_data

# Generate sample data
data = generate_did_data(n_units=50, n_periods=6, seed=42)

# Rank control units by their similarity to treated units
ranking = rank_control_units(
    data,
    unit_column='unit',
    time_column='period',
    outcome_column='outcome',
    treatment_column='treated',
    n_top=10  # Return top 10 controls
)

print(ranking[['unit', 'quality_score', 'pre_trend_rmse']])
```

Output:
```
   unit  quality_score  pre_trend_rmse
0    35         1.0000          0.4521
1    42         0.9234          0.5123
2    28         0.8876          0.5892
...
```

With covariates for matching:

```python
# Add covariate-based matching
ranking = rank_control_units(
    data,
    unit_column='unit',
    time_column='period',
    outcome_column='outcome',
    treatment_column='treated',
    covariates=['size', 'age'],  # Match on these too
    outcome_weight=0.7,          # 70% weight on outcome trends
    covariate_weight=0.3         # 30% weight on covariate similarity
)
```

Filter data for SyntheticDiD using top controls:

```python
from diff_diff import SyntheticDiD

# Get top control units
top_controls = ranking['unit'].tolist()

# Filter data to treated + top controls
filtered_data = data[
    (data['treated'] == 1) | (data['unit'].isin(top_controls))
]

# Fit SyntheticDiD with selected controls
sdid = SyntheticDiD()
results = sdid.fit(
    filtered_data,
    outcome='outcome',
    treatment='treated',
    unit='unit',
    time='period',
    post_periods=[3, 4, 5]
)
```

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

### Wild Cluster Bootstrap

When you have few clusters (<50), standard cluster-robust SEs are biased. Wild cluster bootstrap provides valid inference even with 5-10 clusters.

```python
# Use wild bootstrap for inference
did = DifferenceInDifferences(
    cluster='state',
    inference='wild_bootstrap',
    n_bootstrap=999,
    bootstrap_weights='rademacher',  # or 'webb' for <10 clusters, 'mammen'
    seed=42
)
results = did.fit(data, outcome='y', treatment='treated', time='post')

# Results include bootstrap-based SE and p-value
print(f"ATT: {results.att:.3f} (SE: {results.se:.3f})")
print(f"P-value: {results.p_value:.4f}")
print(f"95% CI: {results.conf_int}")
print(f"Inference method: {results.inference_method}")
print(f"Number of clusters: {results.n_clusters}")
```

**Weight types:**
- `'rademacher'` - Default, ±1 with p=0.5, good for most cases
- `'webb'` - 6-point distribution, recommended for <10 clusters
- `'mammen'` - Two-point distribution, alternative to Rademacher

Works with `DifferenceInDifferences` and `TwoWayFixedEffects` estimators.

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

### Staggered Difference-in-Differences (Callaway-Sant'Anna)

When treatment is adopted at different times by different units, traditional TWFE estimators can be biased. The Callaway-Sant'Anna estimator provides unbiased estimates with staggered adoption.

```python
from diff_diff import CallawaySantAnna

# Panel data with staggered treatment
# 'first_treat' = period when unit was first treated (0 if never treated)
cs = CallawaySantAnna()
results = cs.fit(
    panel_data,
    outcome='sales',
    unit='firm_id',
    time='year',
    first_treat='first_treat',  # 0 for never-treated, else first treatment year
    aggregate='event_study'      # Compute event study effects
)

# View results
results.print_summary()

# Access group-time effects ATT(g,t)
for (group, time), effect in results.group_time_effects.items():
    print(f"Cohort {group}, Period {time}: {effect['effect']:.3f}")

# Event study effects (averaged by relative time)
for rel_time, effect in results.event_study_effects.items():
    print(f"e={rel_time}: {effect['effect']:.3f} (SE: {effect['se']:.3f})")

# Convert to DataFrame
df = results.to_dataframe(level='event_study')
```

Output:
```
=====================================================================================
          Callaway-Sant'Anna Staggered Difference-in-Differences Results
=====================================================================================

Total observations:                     600
Treated units:                           35
Control units:                           15
Treatment cohorts:                        3
Time periods:                             8
Control group:                never_treated

-------------------------------------------------------------------------------------
                  Overall Average Treatment Effect on the Treated
-------------------------------------------------------------------------------------
Parameter         Estimate     Std. Err.     t-stat      P>|t|   Sig.
-------------------------------------------------------------------------------------
ATT                 2.5000       0.3521       7.101     0.0000   ***
-------------------------------------------------------------------------------------

95% Confidence Interval: [1.8099, 3.1901]

-------------------------------------------------------------------------------------
                          Event Study (Dynamic) Effects
-------------------------------------------------------------------------------------
Rel. Period       Estimate     Std. Err.     t-stat      P>|t|   Sig.
-------------------------------------------------------------------------------------
0                   2.1000       0.4521       4.645     0.0000   ***
1                   2.5000       0.4123       6.064     0.0000   ***
2                   2.8000       0.5234       5.349     0.0000   ***
-------------------------------------------------------------------------------------

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
=====================================================================================
```

**When to use Callaway-Sant'Anna vs TWFE:**

| Scenario | Use TWFE | Use Callaway-Sant'Anna |
|----------|----------|------------------------|
| All units treated at same time | ✓ | ✓ |
| Staggered adoption, homogeneous effects | ✓ | ✓ |
| Staggered adoption, heterogeneous effects | ✗ | ✓ |
| Need event study with staggered timing | ✗ | ✓ |
| Fewer than ~20 treated units | ✓ | Depends on design |

**Parameters:**

```python
CallawaySantAnna(
    control_group='never_treated',  # or 'not_yet_treated'
    anticipation=0,                  # Periods before treatment with effects
    estimation_method='dr',          # 'dr', 'ipw', or 'reg'
    alpha=0.05,                      # Significance level
    cluster=None,                    # Column for cluster SEs
    n_bootstrap=0,                   # Must be 0 (bootstrap not yet implemented)
    seed=None                        # Random seed
)
```

**Current limitations:**
- Bootstrap inference (`n_bootstrap > 0`) is not yet implemented
- Covariate adjustment for conditional parallel trends is not yet implemented

### Event Study Visualization

Create publication-ready event study plots:

```python
from diff_diff import plot_event_study, MultiPeriodDiD, CallawaySantAnna

# From MultiPeriodDiD
did = MultiPeriodDiD()
results = did.fit(data, outcome='y', treatment='treated',
                  time='period', post_periods=[3, 4, 5])
plot_event_study(results, title="Treatment Effects Over Time")

# From CallawaySantAnna (with event study aggregation)
cs = CallawaySantAnna()
results = cs.fit(data, outcome='y', unit='unit', time='period',
                 first_treat='first_treat', aggregate='event_study')
plot_event_study(results, title="Staggered DiD Event Study")

# From a DataFrame
df = pd.DataFrame({
    'period': [-2, -1, 0, 1, 2],
    'effect': [0.1, 0.05, 0.0, 2.5, 2.8],
    'se': [0.3, 0.25, 0.0, 0.4, 0.45]
})
plot_event_study(df, reference_period=0)

# With customization
ax = plot_event_study(
    results,
    title="Dynamic Treatment Effects",
    xlabel="Years Relative to Treatment",
    ylabel="Effect on Sales ($1000s)",
    color="#2563eb",
    marker="o",
    shade_pre=True,           # Shade pre-treatment region
    show_zero_line=True,      # Horizontal line at y=0
    show_reference_line=True, # Vertical line at reference period
    figsize=(10, 6),
    show=False                # Don't call plt.show(), return axes
)
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

### Honest DiD Sensitivity Analysis (Rambachan-Roth)

Pre-trends tests have low power and can exacerbate bias. **Honest DiD** (Rambachan & Roth 2023) provides sensitivity analysis showing how robust your results are to violations of parallel trends.

```python
from diff_diff import HonestDiD, MultiPeriodDiD

# First, fit a standard event study
did = MultiPeriodDiD()
event_results = did.fit(
    data,
    outcome='outcome',
    treatment='treated',
    time='period',
    post_periods=[5, 6, 7, 8, 9]
)

# Compute honest bounds with relative magnitudes restriction
# M=1 means post-treatment violations can be up to 1x the worst pre-treatment violation
honest = HonestDiD(method='relative_magnitude', M=1.0)
honest_results = honest.fit(event_results)

print(honest_results.summary())
print(f"Original estimate: {honest_results.original_estimate:.4f}")
print(f"Robust 95% CI: [{honest_results.ci_lb:.4f}, {honest_results.ci_ub:.4f}]")
print(f"Effect robust to violations: {honest_results.is_significant}")
```

**Sensitivity analysis over M values:**

```python
# How do results change as we allow larger violations?
sensitivity = honest.sensitivity_analysis(
    event_results,
    M_grid=[0, 0.5, 1.0, 1.5, 2.0]
)

print(sensitivity.summary())
print(f"Breakdown value: M = {sensitivity.breakdown_M}")
# Breakdown = smallest M where the robust CI includes zero
```

**Breakdown value:**

The breakdown value tells you how robust your conclusion is:

```python
breakdown = honest.breakdown_value(event_results)
if breakdown >= 1.0:
    print("Result holds even if post-treatment violations are as bad as pre-treatment")
else:
    print(f"Result requires violations smaller than {breakdown:.1f}x pre-treatment")
```

**Smoothness restriction (alternative approach):**

```python
# Bounds second differences of trend violations
# M=0 means linear extrapolation of pre-trends
honest_smooth = HonestDiD(method='smoothness', M=0.5)
smooth_results = honest_smooth.fit(event_results)
```

**Visualization:**

```python
from diff_diff import plot_sensitivity, plot_honest_event_study

# Plot sensitivity analysis
plot_sensitivity(sensitivity, title="Sensitivity to Parallel Trends Violations")

# Event study with honest confidence intervals
plot_honest_event_study(event_results, honest_results)
```

### Placebo Tests

Placebo tests help validate the parallel trends assumption by checking whether effects appear where they shouldn't (before treatment or in untreated groups).

**Fake timing test:**

```python
from diff_diff import run_placebo_test

# Test: Is there an effect before treatment actually occurred?
# Actual treatment is at period 3 (post_periods=[3, 4, 5])
# We test if a "fake" treatment at period 1 shows an effect
results = run_placebo_test(
    data,
    outcome='outcome',
    treatment='treated',
    time='period',
    test_type='fake_timing',
    fake_treatment_period=1,  # Pretend treatment was in period 1
    post_periods=[3, 4, 5]    # Actual post-treatment periods
)

print(results.summary())
# If parallel trends hold, placebo_effect should be ~0 and not significant
print(f"Placebo effect: {results.placebo_effect:.3f} (p={results.p_value:.3f})")
print(f"Is significant (bad): {results.is_significant}")
```

**Fake group test:**

```python
# Test: Is there an effect among never-treated units?
# Get some control unit IDs to use as "fake treated"
control_units = data[data['treated'] == 0]['firm_id'].unique()[:5]

results = run_placebo_test(
    data,
    outcome='outcome',
    treatment='treated',
    time='period',
    unit='firm_id',
    test_type='fake_group',
    fake_treatment_group=list(control_units),  # List of control unit IDs
    post_periods=[3, 4, 5]
)
```

**Permutation test:**

```python
# Randomly reassign treatment and compute distribution of effects
# Note: requires binary post indicator (use 'post' column, not 'period')
results = run_placebo_test(
    data,
    outcome='outcome',
    treatment='treated',
    time='post',           # Binary post-treatment indicator
    unit='firm_id',
    test_type='permutation',
    n_permutations=1000,
    seed=42
)

print(f"Original effect: {results.original_effect:.3f}")
print(f"Permutation p-value: {results.p_value:.4f}")
# Low p-value indicates the effect is unlikely to be due to chance
```

**Leave-one-out sensitivity:**

```python
# Test sensitivity to individual treated units
# Note: requires binary post indicator (use 'post' column, not 'period')
results = run_placebo_test(
    data,
    outcome='outcome',
    treatment='treated',
    time='post',           # Binary post-treatment indicator
    unit='firm_id',
    test_type='leave_one_out'
)

# Check if any single unit drives the result
print(results.leave_one_out_effects)  # Effect when each unit is dropped
```

**Run all placebo tests:**

```python
from diff_diff import run_all_placebo_tests

# Comprehensive diagnostic suite
# Note: This function runs fake_timing tests on pre-treatment periods.
# The permutation and leave_one_out tests require a binary post indicator,
# so they may return errors if the data uses multi-period time column.
all_results = run_all_placebo_tests(
    data,
    outcome='outcome',
    treatment='treated',
    time='period',
    unit='firm_id',
    pre_periods=[0, 1, 2],
    post_periods=[3, 4, 5],
    n_permutations=500,
    seed=42
)

for test_name, result in all_results.items():
    if hasattr(result, 'p_value'):
        print(f"{test_name}: p={result.p_value:.3f}, significant={result.is_significant}")
    elif isinstance(result, dict) and 'error' in result:
        print(f"{test_name}: Error - {result['error']}")
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

### HonestDiD

```python
HonestDiD(
    method='relative_magnitude',  # 'relative_magnitude' or 'smoothness'
    M=None,               # Restriction parameter (default: 1.0 for RM, 0.0 for SD)
    alpha=0.05,           # Significance level for CIs
    l_vec=None            # Linear combination vector for target parameter
)
```

**fit() Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | MultiPeriodDiDResults | Results from MultiPeriodDiD.fit() |
| `M` | float | Restriction parameter (overrides constructor value) |

**Methods:**

| Method | Description |
|--------|-------------|
| `fit(results, M)` | Compute bounds for given event study results |
| `sensitivity_analysis(results, M_grid)` | Compute bounds over grid of M values |
| `breakdown_value(results, tol)` | Find smallest M where CI includes zero |

### HonestDiDResults

**Attributes:**

| Attribute | Description |
|-----------|-------------|
| `original_estimate` | Point estimate under parallel trends |
| `lb` | Lower bound of identified set |
| `ub` | Upper bound of identified set |
| `ci_lb` | Lower bound of robust confidence interval |
| `ci_ub` | Upper bound of robust confidence interval |
| `ci_width` | Width of robust CI |
| `M` | Restriction parameter used |
| `method` | Restriction method ('relative_magnitude' or 'smoothness') |
| `alpha` | Significance level |
| `is_significant` | True if robust CI excludes zero |

**Methods:**

| Method | Description |
|--------|-------------|
| `summary()` | Get formatted summary string |
| `to_dict()` | Convert to dictionary |
| `to_dataframe()` | Convert to pandas DataFrame |

### SensitivityResults

**Attributes:**

| Attribute | Description |
|-----------|-------------|
| `M_grid` | Array of M values analyzed |
| `results` | List of HonestDiDResults for each M |
| `breakdown_M` | Smallest M where CI includes zero (None if always significant) |

**Methods:**

| Method | Description |
|--------|-------------|
| `summary()` | Get formatted summary string |
| `plot(ax)` | Plot sensitivity analysis |
| `to_dataframe()` | Convert to pandas DataFrame |

### Data Preparation Functions

#### generate_did_data

```python
generate_did_data(
    n_units=100,          # Number of units
    n_periods=4,          # Number of time periods
    treatment_effect=5.0, # True ATT
    treatment_fraction=0.5,  # Fraction treated
    treatment_period=2,   # First post-treatment period
    unit_fe_sd=2.0,       # Unit fixed effect std dev
    time_trend=0.5,       # Linear time trend
    noise_sd=1.0,         # Idiosyncratic noise std dev
    seed=None             # Random seed
)
```

Returns DataFrame with columns: `unit`, `period`, `treated`, `post`, `outcome`, `true_effect`.

#### make_treatment_indicator

```python
make_treatment_indicator(
    data,                 # Input DataFrame
    column,               # Column to create treatment from
    treated_values=None,  # Value(s) indicating treatment
    threshold=None,       # Numeric threshold for treatment
    above_threshold=True, # If True, >= threshold is treated
    new_column='treated'  # Output column name
)
```

#### make_post_indicator

```python
make_post_indicator(
    data,                  # Input DataFrame
    time_column,           # Time/period column
    post_periods=None,     # Specific post-treatment period(s)
    treatment_start=None,  # First post-treatment period
    new_column='post'      # Output column name
)
```

#### wide_to_long

```python
wide_to_long(
    data,                  # Wide-format DataFrame
    value_columns,         # List of time-varying columns
    id_column,             # Unit identifier column
    time_name='period',    # Name for time column
    value_name='value',    # Name for value column
    time_values=None       # Values for time periods
)
```

#### balance_panel

```python
balance_panel(
    data,                  # Panel DataFrame
    unit_column,           # Unit identifier column
    time_column,           # Time period column
    method='inner',        # 'inner', 'outer', or 'fill'
    fill_value=None        # Value for filling (if method='fill')
)
```

#### validate_did_data

```python
validate_did_data(
    data,                  # DataFrame to validate
    outcome,               # Outcome column name
    treatment,             # Treatment column name
    time,                  # Time/post column name
    unit=None,             # Unit column (for panel validation)
    raise_on_error=True    # Raise ValueError or return dict
)
```

Returns dict with `valid`, `errors`, `warnings`, and `summary` keys.

#### summarize_did_data

```python
summarize_did_data(
    data,                  # Input DataFrame
    outcome,               # Outcome column name
    treatment,             # Treatment column name
    time,                  # Time/post column name
    unit=None              # Unit column (optional)
)
```

Returns DataFrame with summary statistics by treatment-time cell.

#### create_event_time

```python
create_event_time(
    data,                  # Panel DataFrame
    time_column,           # Calendar time column
    treatment_time_column, # Column with treatment timing
    new_column='event_time' # Output column name
)
```

#### aggregate_to_cohorts

```python
aggregate_to_cohorts(
    data,                  # Unit-level panel data
    unit_column,           # Unit identifier column
    time_column,           # Time period column
    treatment_column,      # Treatment indicator column
    outcome,               # Outcome variable column
    covariates=None        # Additional columns to aggregate
)
```

#### rank_control_units

```python
rank_control_units(
    data,                          # Panel data in long format
    unit_column,                   # Unit identifier column
    time_column,                   # Time period column
    outcome_column,                # Outcome variable column
    treatment_column=None,         # Treatment indicator column (0/1)
    treated_units=None,            # Explicit list of treated unit IDs
    pre_periods=None,              # Pre-treatment periods (default: first half)
    covariates=None,               # Covariate columns for matching
    outcome_weight=0.7,            # Weight for outcome trend similarity (0-1)
    covariate_weight=0.3,          # Weight for covariate distance (0-1)
    exclude_units=None,            # Units to exclude from control pool
    require_units=None,            # Units that must appear in output
    n_top=None,                    # Return only top N controls
    suggest_treatment_candidates=False,  # Identify treatment candidates
    n_treatment_candidates=5,      # Number of treatment candidates
    lambda_reg=0.0                 # Regularization for synthetic weights
)
```

Returns DataFrame with columns: `unit`, `quality_score`, `outcome_trend_score`, `covariate_score`, `synthetic_weight`, `pre_trend_rmse`, `is_required`.

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

## References

This library implements methods from the following scholarly works:

### Difference-in-Differences

- **Ashenfelter, O., & Card, D. (1985).** "Using the Longitudinal Structure of Earnings to Estimate the Effect of Training Programs." *The Review of Economics and Statistics*, 67(4), 648-660. [https://doi.org/10.2307/1924810](https://doi.org/10.2307/1924810)

- **Card, D., & Krueger, A. B. (1994).** "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania." *The American Economic Review*, 84(4), 772-793. [https://www.jstor.org/stable/2118030](https://www.jstor.org/stable/2118030)

- **Angrist, J. D., & Pischke, J.-S. (2009).** *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press. Chapter 5: Differences-in-Differences.

### Two-Way Fixed Effects

- **Wooldridge, J. M. (2010).** *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

- **Imai, K., & Kim, I. S. (2021).** "On the Use of Two-Way Fixed Effects Regression Models for Causal Inference with Panel Data." *Political Analysis*, 29(3), 405-415. [https://doi.org/10.1017/pan.2020.33](https://doi.org/10.1017/pan.2020.33)

### Robust Standard Errors

- **White, H. (1980).** "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)

- **MacKinnon, J. G., & White, H. (1985).** "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties." *Journal of Econometrics*, 29(3), 305-325. [https://doi.org/10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7)

- **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).** "Robust Inference With Multiway Clustering." *Journal of Business & Economic Statistics*, 29(2), 238-249. [https://doi.org/10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136)

### Wild Cluster Bootstrap

- **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008).** "Bootstrap-Based Improvements for Inference with Clustered Errors." *The Review of Economics and Statistics*, 90(3), 414-427. [https://doi.org/10.1162/rest.90.3.414](https://doi.org/10.1162/rest.90.3.414)

- **Webb, M. D. (2014).** "Reworking Wild Bootstrap Based Inference for Clustered Errors." Queen's Economics Department Working Paper No. 1315. [https://www.econ.queensu.ca/sites/econ.queensu.ca/files/qed_wp_1315.pdf](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/qed_wp_1315.pdf)

- **MacKinnon, J. G., & Webb, M. D. (2018).** "The Wild Bootstrap for Few (Treated) Clusters." *The Econometrics Journal*, 21(2), 114-135. [https://doi.org/10.1111/ectj.12107](https://doi.org/10.1111/ectj.12107)

### Placebo Tests and DiD Diagnostics

- **Bertrand, M., Duflo, E., & Mullainathan, S. (2004).** "How Much Should We Trust Differences-in-Differences Estimates?" *The Quarterly Journal of Economics*, 119(1), 249-275. [https://doi.org/10.1162/003355304772839588](https://doi.org/10.1162/003355304772839588)

### Synthetic Control Method

- **Abadie, A., & Gardeazabal, J. (2003).** "The Economic Costs of Conflict: A Case Study of the Basque Country." *The American Economic Review*, 93(1), 113-132. [https://doi.org/10.1257/000282803321455188](https://doi.org/10.1257/000282803321455188)

- **Abadie, A., Diamond, A., & Hainmueller, J. (2010).** "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493-505. [https://doi.org/10.1198/jasa.2009.ap08746](https://doi.org/10.1198/jasa.2009.ap08746)

- **Abadie, A., Diamond, A., & Hainmueller, J. (2015).** "Comparative Politics and the Synthetic Control Method." *American Journal of Political Science*, 59(2), 495-510. [https://doi.org/10.1111/ajps.12116](https://doi.org/10.1111/ajps.12116)

### Synthetic Difference-in-Differences

- **Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).** "Synthetic Difference-in-Differences." *American Economic Review*, 111(12), 4088-4118. [https://doi.org/10.1257/aer.20190159](https://doi.org/10.1257/aer.20190159)

### Parallel Trends and Pre-Trend Testing

- **Roth, J. (2022).** "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322. [https://doi.org/10.1257/aeri.20210236](https://doi.org/10.1257/aeri.20210236)

- **Lakens, D. (2017).** "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." *Social Psychological and Personality Science*, 8(4), 355-362. [https://doi.org/10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177)

### Honest DiD / Sensitivity Analysis

The `HonestDiD` module implements sensitivity analysis methods for relaxing the parallel trends assumption:

- **Rambachan, A., & Roth, J. (2023).** "A More Credible Approach to Parallel Trends." *The Review of Economic Studies*, 90(5), 2555-2591. [https://doi.org/10.1093/restud/rdad018](https://doi.org/10.1093/restud/rdad018)

  This paper introduces the "Honest DiD" framework implemented in our `HonestDiD` class:
  - **Relative Magnitudes (ΔRM)**: Bounds post-treatment violations by a multiple of observed pre-treatment violations
  - **Smoothness (ΔSD)**: Bounds on second differences of trend violations, allowing for linear extrapolation of pre-trends
  - **Breakdown Analysis**: Finding the smallest violation magnitude that would overturn conclusions
  - **Robust Confidence Intervals**: Valid inference under partial identification

- **Roth, J., & Sant'Anna, P. H. C. (2023).** "When Is Parallel Trends Sensitive to Functional Form?" *Econometrica*, 91(2), 737-747. [https://doi.org/10.3982/ECTA19402](https://doi.org/10.3982/ECTA19402)

  Discusses functional form sensitivity in parallel trends assumptions, relevant to understanding when smoothness restrictions are appropriate.

### Multi-Period and Staggered Adoption

- **Callaway, B., & Sant'Anna, P. H. C. (2021).** "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230. [https://doi.org/10.1016/j.jeconom.2020.12.001](https://doi.org/10.1016/j.jeconom.2020.12.001)

- **Sant'Anna, P. H. C., & Zhao, J. (2020).** "Doubly Robust Difference-in-Differences Estimators." *Journal of Econometrics*, 219(1), 101-122. [https://doi.org/10.1016/j.jeconom.2020.06.003](https://doi.org/10.1016/j.jeconom.2020.06.003)

- **Sun, L., & Abraham, S. (2021).** "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199. [https://doi.org/10.1016/j.jeconom.2020.09.006](https://doi.org/10.1016/j.jeconom.2020.09.006)

- **de Chaisemartin, C., & D'Haultfœuille, X. (2020).** "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." *American Economic Review*, 110(9), 2964-2996. [https://doi.org/10.1257/aer.20181169](https://doi.org/10.1257/aer.20181169)

- **Goodman-Bacon, A. (2021).** "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics*, 225(2), 254-277. [https://doi.org/10.1016/j.jeconom.2021.03.014](https://doi.org/10.1016/j.jeconom.2021.03.014)

### General Causal Inference

- **Imbens, G. W., & Rubin, D. B. (2015).** *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.

- **Cunningham, S. (2021).** *Causal Inference: The Mixtape*. Yale University Press. [https://mixtape.scunning.com/](https://mixtape.scunning.com/)

## License

MIT License
