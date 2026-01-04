# Implementation Plan: Honest DiD (Rambachan-Roth)

## Overview

This plan covers implementing the Honest DiD sensitivity analysis framework based on [Rambachan & Roth (2023)](https://academic.oup.com/restud/article/90/5/2555/7039335) "A More Credible Approach to Parallel Trends" published in *The Review of Economic Studies*.

### What is Honest DiD?

Honest DiD provides tools for robust inference in difference-in-differences designs when parallel trends may be violated. Instead of assuming parallel trends holds exactly, it:

1. **Imposes restrictions** on how post-treatment trend violations relate to pre-treatment violations
2. **Computes bounds** on the treatment effect under these restrictions (partial identification)
3. **Constructs robust confidence intervals** that cover the true parameter with high probability
4. **Calculates breakdown values** showing how much violation would nullify the result

### Key Restrictions Supported

1. **Relative Magnitudes (ΔRM)**: Post-treatment violations ≤ M̄ × max(|pre-treatment violations|)
   - M̄=1: Post-period violations no worse than observed pre-trends
   - M̄=0: No violations (standard parallel trends)
   - M̄>1: Allow larger post-period violations

2. **Smoothness (ΔSD)**: Bounds on second differences of the trend
   - M=0: Linear extrapolation of pre-trends
   - M>0: Allow non-linear trend changes up to M per period

---

## Architectural Decisions

### Decision 1: Standalone Module vs. Extension

**Recommendation: Standalone `honest_did.py` module**

Rationale:
- Follows library pattern (each major estimator family has its own module)
- Clean separation of concerns
- Can work as a wrapper around existing estimators
- Easier to test independently

### Decision 2: Class-Based vs. Functional API

**Recommendation: Hybrid approach**

- `HonestDiD` class for configuration and fitting
- Convenience functions for common use cases
- Follows existing library patterns (like `DifferenceInDifferences` class)

### Decision 3: Integration Points

**Primary integration with:**
- `MultiPeriodDiDResults` - event study with pre/post coefficients
- `CallawaySantAnnaResults` - staggered adoption event studies

**Secondary integration:**
- Could extend to basic `DiDResults` with minimal functionality

### Decision 4: Optimization Backend

**Options:**
1. `scipy.optimize.linprog` - Linear programming (for smoothness)
2. `scipy.optimize.minimize` - General optimization (for relative magnitudes)
3. Custom quadratic programming via projected gradient descent

**Recommendation: Use scipy.optimize**
- Already a dependency (via other utilities)
- Well-tested and stable
- Sufficient for our needs

---

## Module Structure

```
diff_diff/
├── honest_did.py (NEW - ~800 lines)
│   ├── # Delta restriction classes
│   ├── DeltaSD (smoothness restriction)
│   ├── DeltaRM (relative magnitudes restriction)
│   ├── DeltaSDRM (combined restriction)
│   │
│   ├── # Main class
│   ├── HonestDiD
│   │   ├── __init__(method, M, alpha, ...)
│   │   ├── fit(results) → HonestDiDResults
│   │   ├── sensitivity_analysis(results, M_grid) → SensitivityResults
│   │   ├── breakdown_value(results) → float
│   │   └── _compute_bounds(), _compute_robust_ci(), ...
│   │
│   ├── # Results classes
│   ├── HonestDiDResults
│   │   ├── bounds: Tuple[float, float]
│   │   ├── robust_ci: Tuple[float, float]
│   │   ├── M: float
│   │   ├── method: str
│   │   ├── original_results: Any
│   │   ├── summary(), to_dict(), to_dataframe()
│   │   └── is_significant (True if CI excludes 0)
│   │
│   └── SensitivityResults
│       ├── M_values: np.ndarray
│       ├── bounds: List[Tuple]
│       ├── robust_cis: List[Tuple]
│       ├── breakdown_M: float
│       └── plot()
│
├── visualization.py (EXTEND)
│   ├── plot_sensitivity() - Bounds as function of M
│   └── plot_honest_event_study() - Event study with Honest CIs
│
├── utils.py (EXTEND)
│   └── Internal helper functions if needed
│
└── tests/
    └── test_honest_did.py (NEW - ~400 lines)
```

---

## Implementation Details

### Phase 1: Core Infrastructure

#### 1.1 Delta Restriction Classes

```python
@dataclass
class DeltaSD:
    """Smoothness restriction on trend violations.

    Restricts the second differences: |δ_{t+1} - δ_t| ≤ M
    where δ_t is the violation at time t.
    """
    M: float = 0.0  # M=0 means linear extrapolation

@dataclass
class DeltaRM:
    """Relative magnitudes restriction.

    Post-treatment violations bounded by M̄ × max(|pre-treatment violations|).
    """
    Mbar: float = 1.0  # Mbar=1 means violations ≤ max pre-period violation
```

#### 1.2 Extract Event Study Coefficients

Need to extract from results objects:
- `beta_hat`: Vector of event study coefficients (pre + post periods)
- `sigma`: Variance-covariance matrix
- `num_pre_periods`: Number of pre-treatment periods
- `num_post_periods`: Number of post-treatment periods

```python
def _extract_event_study_params(results) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Extract event study parameters from various results types."""
    if isinstance(results, MultiPeriodDiDResults):
        # Extract from period_effects
        ...
    elif isinstance(results, CallawaySantAnnaResults):
        # Extract from event_study_effects
        ...
```

#### 1.3 Construct Constraint Matrices

The optimization problem requires constructing matrices A and d such that:
- δ ∈ Δ ⟺ A @ δ ≤ d

For smoothness (ΔSD):
```
|δ_{t+1} - δ_t - (δ_t - δ_{t-1})| ≤ M  for all t
```

For relative magnitudes (ΔRM):
```
|δ_post| ≤ Mbar × max(|δ_pre|)
```

### Phase 2: Bounds Computation

#### 2.1 Identified Set Bounds

For a scalar parameter θ = l'β (where l is a weighting vector):

**Lower bound:**
```
min_{δ ∈ Δ} l'(β - δ)
```

**Upper bound:**
```
max_{δ ∈ Δ} l'(β - δ)
```

This is a linear program when Δ is defined by linear constraints.

#### 2.2 Robust Confidence Intervals

Two methods from the paper:

1. **FLCI (Fixed Length Confidence Interval)** - for smoothness
   - Uses conditional inference
   - Uniform coverage over Δ

2. **C-LF (Conditional Least Favorable)** - for relative magnitudes
   - Hybrid bootstrap/analytic approach
   - Accounts for estimation of max pre-period violation

Implementation will follow the R package approach.

### Phase 3: Main Class Implementation

```python
class HonestDiD:
    """Honest DiD sensitivity analysis (Rambachan & Roth 2023).

    Parameters
    ----------
    method : str
        "smoothness" (ΔSD) or "relative_magnitude" (ΔRM)
    M : float
        Restriction parameter. For smoothness, bounds second differences.
        For relative_magnitude, scales max pre-period violation.
    alpha : float
        Significance level for confidence intervals (default 0.05)
    l_vec : array-like or str, optional
        Weighting vector for scalar parameter.
        "avg" for average post-treatment effect.
        None for all post-period effects.

    Examples
    --------
    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.honest_did import HonestDiD
    >>>
    >>> # Fit event study
    >>> mp_did = MultiPeriodDiD()
    >>> results = mp_did.fit(data, outcome='y', treatment='treated',
    ...                      time='period', post_periods=[4,5,6,7])
    >>>
    >>> # Sensitivity analysis with relative magnitudes
    >>> honest = HonestDiD(method='relative_magnitude', M=1.0)
    >>> bounds = honest.fit(results)
    >>> print(bounds.summary())
    """
```

### Phase 4: Sensitivity Curve and Breakdown

#### 4.1 Sensitivity Analysis

```python
def sensitivity_analysis(self, results, M_grid=None) -> SensitivityResults:
    """Compute bounds for a grid of M values.

    Returns object with:
    - M_values: array of M parameter values
    - bounds: list of (lower, upper) identified set bounds
    - robust_cis: list of (lower, upper) robust CIs
    - breakdown_M: smallest M where CI includes 0
    """
```

#### 4.2 Breakdown Value

The breakdown value is the smallest M where the robust CI includes zero:
```python
def breakdown_value(self, results) -> float:
    """Find M where robust CI first includes zero.

    Uses binary search over M values.
    """
```

### Phase 5: Visualization

#### 5.1 Sensitivity Plot

```python
def plot_sensitivity(
    sensitivity_results: SensitivityResults,
    ax=None,
    show_bounds: bool = True,
    show_ci: bool = True,
    breakdown_line: bool = True,
    **kwargs
):
    """Plot bounds and CIs as function of M.

    Similar to Figure 3 in Rambachan-Roth paper.
    """
```

#### 5.2 Honest Event Study Plot

```python
def plot_honest_event_study(
    honest_results: HonestDiDResults,
    original_results=None,
    ax=None,
    **kwargs
):
    """Event study plot with honest confidence intervals.

    Shows original point estimates with robust CIs.
    """
```

### Phase 6: Documentation

1. **Docstrings**: Comprehensive docstrings for all public classes/functions
2. **Tutorial notebook**: `docs/tutorials/05_honest_did.ipynb`
   - Motivation and theory
   - Basic usage with MultiPeriodDiD
   - Interpretation of bounds and breakdown values
   - Comparison with standard CIs
   - Sensitivity plots
3. **Update README**: Add Honest DiD to feature list

---

## Testing Strategy

### Unit Tests

```python
# tests/test_honest_did.py

class TestDeltaRestrictions:
    def test_delta_sd_constraint_matrix(self):
        """Test smoothness constraint construction."""

    def test_delta_rm_constraint_matrix(self):
        """Test relative magnitude constraint construction."""

class TestHonestDiD:
    def test_basic_smoothness(self):
        """Test M=0 (linear extrapolation) gives point-identified result."""

    def test_basic_relative_magnitude(self):
        """Test Mbar=0 recovers standard CI."""

    def test_bounds_widen_with_M(self):
        """Verify bounds get wider as M increases."""

    def test_breakdown_value(self):
        """Test breakdown value computation."""

    def test_integration_multiperiod(self):
        """Test with MultiPeriodDiDResults."""

    def test_integration_callaway_santanna(self):
        """Test with CallawaySantAnnaResults."""

class TestVisualization:
    def test_plot_sensitivity_runs(self):
        """Test sensitivity plot doesn't error."""

    def test_plot_honest_event_study_runs(self):
        """Test event study plot doesn't error."""
```

### Validation Against R Package

Where possible, validate results against the R `HonestDiD` package using synthetic data with known properties.

---

## Key Questions for Discussion

### Q1: Scope of Initial Release

**Option A: Full Implementation**
- Both smoothness (ΔSD) and relative magnitudes (ΔRM)
- Combined restrictions (ΔSDRM)
- Full FLCI and C-LF methods
- ~2-3 weeks of work

**Option B: Focused Implementation** (Recommended)
- Start with relative magnitudes (ΔRM) - most commonly used
- Add smoothness (ΔSD) as follow-up
- ~1-1.5 weeks of work

### Q2: Confidence Interval Method

The paper describes multiple CI methods with different properties:
- **FLCI**: Fixed length, simpler, good for smoothness
- **C-LF**: Conditional, better for relative magnitudes

Should we implement both or focus on one?

### Q3: Integration Depth

How tightly should this integrate with existing estimators?
- **Light**: Separate module that takes results objects
- **Deep**: Add `.honest_ci()` method to each estimator

Recommendation: Start light, add convenience methods later.

### Q4: Optimization Dependencies

The R package uses specialized quadratic programming. We can either:
- Use `scipy.optimize.linprog` (LP, already in deps)
- Use `scipy.optimize.minimize` with SLSQP (QP)
- Consider `cvxpy` for cleaner formulation (new dependency)

Recommendation: Use scipy to avoid new dependencies.

---

## Implementation Order

1. **Core infrastructure** (~2 days)
   - Delta restriction classes
   - Event study parameter extraction
   - Constraint matrix construction

2. **Bounds computation** (~2 days)
   - Linear programming for smoothness
   - Optimization for relative magnitudes
   - Basic identified set bounds

3. **Confidence intervals** (~2 days)
   - FLCI implementation
   - C-LF implementation
   - Hybrid bootstrap approach

4. **Main class and results** (~1 day)
   - HonestDiD class
   - HonestDiDResults dataclass
   - SensitivityResults dataclass

5. **Sensitivity analysis** (~1 day)
   - Grid computation
   - Breakdown value
   - Interpolation

6. **Visualization** (~1 day)
   - Sensitivity plots
   - Honest event study plots

7. **Testing** (~2 days)
   - Unit tests
   - Integration tests
   - Validation against R package

8. **Documentation** (~1 day)
   - Tutorial notebook
   - README update
   - Docstring refinement

**Total: ~12 working days**

---

## Success Criteria

1. **Correctness**: Results match R package on test cases
2. **Usability**: Clean API consistent with library patterns
3. **Performance**: Reasonable computation time (<10s for typical use)
4. **Documentation**: Clear tutorial explaining interpretation
5. **Testing**: >90% code coverage for new module

---

## References

1. Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *The Review of Economic Studies*, 90(5), 2555-2591.

2. R Package: https://github.com/asheshrambachan/HonestDiD

3. Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.
