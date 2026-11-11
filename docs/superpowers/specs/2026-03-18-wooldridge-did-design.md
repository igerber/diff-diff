# WooldridgeDiD Estimator — Design Spec

**Date:** 2026-03-18
**Status:** Approved
**Scope:** Integrate Stata `jwdid` (Wooldridge ETWFE) functionality into diff-diff

---

## 1. Background and Motivation

The Stata package `jwdid` (Friosavila 2021) implements Wooldridge's (2021, 2023) Extended
Two-Way Fixed Effects (ETWFE) estimator for staggered DiD. Its key advantages over existing
diff-diff estimators are:

- **Saturated regression**: estimates all cohort×time ATT(g,t) in a single pooled OLS,
  more efficient than Callaway-Sant'Anna's pair-wise approach
- **Nonlinear extension**: Wooldridge (2023) extends ETWFE to logit and Poisson, avoiding
  the incidental parameters problem — no other estimator in diff-diff supports this
- **Equivalence to CS**: under identical assumptions, ETWFE ATT(g,t) equals CS ATT(g,t)

**Primary references:**
- Wooldridge (2021). "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and
  Difference-in-Differences Estimators." SSRN 3906345.
- Wooldridge (2023). "Simple approaches to nonlinear difference-in-differences with panel
  data." *The Econometrics Journal*, 26(3), C31–C66.
- Friosavila (2021). `jwdid`: Stata module. SSC s459114.

---

## 2. Architecture Overview

### New files
| File | Purpose |
|------|---------|
| `diff_diff/wooldridge.py` | `WooldridgeDiD` estimator class |
| `diff_diff/wooldridge_results.py` | `WooldridgeDiDResults` dataclass |
| `tests/test_wooldridge.py` | Full test suite |

### Modified files
| File | Change |
|------|--------|
| `diff_diff/__init__.py` | Export `WooldridgeDiD`, `WooldridgeDiDResults` |
| `docs/methodology/REGISTRY.md` | Add ETWFE methodology section |

### Class hierarchy
`WooldridgeDiD` is a **standalone estimator** (same level as `CallawaySantAnna`,
`SunAbraham`, etc.), not inheriting from `DifferenceInDifferences`. It implements its own
`get_params` / `set_params`.

---

## 3. Public API

### Constructor

```python
class WooldridgeDiD:
    def __init__(
        self,
        method: str = "ols",                    # "ols" | "logit" | "poisson"
        control_group: str = "not_yet_treated", # "never_treated" | "not_yet_treated"
        anticipation: int = 0,                  # pre-treatment periods to include
        demean_covariates: bool = True,         # within cohort-period demeaning (jwdid default)
        alpha: float = 0.05,
        cluster: Optional[str] = None,          # default: unit identifier (jwdid default)
        n_bootstrap: int = 0,                   # >0 enables multiplier bootstrap
        bootstrap_weights: str = "rademacher",  # "rademacher" | "webb" | "mammen"
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",    # "warn" | "error" | "silent"
    ): ...
```

### fit()

```python
def fit(
    self,
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    cohort: str,                            # first treatment period; 0/NaN = never treated
    exovar: Optional[List[str]] = None,     # time-invariant covariates (no interaction)
    xtvar: Optional[List[str]] = None,      # time-varying covariates (demeaned within cohort-period)
    xgvar: Optional[List[str]] = None,      # cohort-interacted covariates
) -> "WooldridgeDiDResults": ...
```

**Notes:**
- `cohort` column convention: integer = first treatment period, 0 or NaN = never treated.
  Consistent with `CallawaySantAnna`'s `cohort` parameter.
- Default clustering is at the `unit` level (matches `jwdid` default of `vce(cluster ivar)`).
- `demean_covariates=True` corresponds to `jwdid` default; `False` corresponds to `xasis` option.

### get_params / set_params

```python
def get_params(self) -> Dict[str, Any]: ...       # returns all constructor params
def set_params(self, **params) -> "WooldridgeDiD": ...  # sklearn-compatible
```

---

## 4. Results Object

```python
@dataclass
class WooldridgeDiDResults:
    # Raw cohort×time estimates — core output
    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    # key = (g, t); value = {"att", "se", "t_stat", "p_value", "conf_int"}

    # Simple aggregation (always computed on fit)
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]

    # Other aggregations (populated by .aggregate())
    group_effects: Optional[Dict[Any, Dict]]       # keyed by cohort g
    calendar_effects: Optional[Dict[Any, Dict]]    # keyed by calendar period t
    event_study_effects: Optional[Dict[int, Dict]] # keyed by relative period k = t - g

    # Metadata
    method: str
    control_group: str
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05

    # Methods
    def aggregate(self, type: str) -> "WooldridgeDiDResults": ...
    # type: "simple" | "group" | "calendar" | "event"
    # fills corresponding fields, returns self for chaining

    def summary(self, aggregation: str = "simple") -> str: ...
    def to_dataframe(self, aggregation: str = "event") -> pd.DataFrame: ...
    def plot_event_study(self, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
```

**Inference rule:** ALL inference fields (t_stat, p_value, conf_int) computed together
via `safe_inference()` from `diff_diff.utils`. Never computed individually.

---

## 5. Internal Computation

### 5a. Linear ETWFE (`method="ols"`)

Faithful port of `jwdid` + `reghdfe`:

1. **Filter observations**: keep control group (never- or not-yet-treated at time t) plus
   all treated units. Drop observations where `t < g - anticipation`.

2. **Build interaction matrix**: for each (g, t) with `t >= g - anticipation`, create
   column `1(G_i = g) * 1(T = t)`. These are the β_{g,t} regressors.

3. **Covariate preparation**:
   - `exovar`: append as-is
   - `xtvar`: demean within (cohort × period) cells when `demean_covariates=True`
   - `xgvar`: interact with each cohort indicator

4. **Absorb unit + time FE**: within-transformation (existing `absorb` mechanism in
   `linalg.py`), not explicit dummies.

5. **Solve**: `linalg.solve_ols()` → extract β_{g,t} coefficients and vcov submatrix.

6. **Inference**: `linalg.compute_robust_vcov()` with unit-level clustering by default,
   then `safe_inference()` for each (g, t) cell.

7. **Bootstrap**: multiplier bootstrap supported for all inference;
   wild cluster bootstrap supported for linear only (same as `DifferenceInDifferences`).

### 5b. Nonlinear (`method="logit"|"poisson"`)

Following Wooldridge (2023) pooled QMLE approach:

- **Logit**: group-level FE (cohort × period), **not** individual FE — avoids incidental
  parameters problem. Log-likelihood: Bernoulli QLL.
- **Poisson**: individual FE absorbed via PPML (iterative within-transformation).
  Log-likelihood: Poisson QLL.

Optimization: `scipy.optimize.minimize` (L-BFGS-B). Vcov from numerical Hessian
(`scipy.optimize.approx_fprime` second differences).

**ATT computation via Average Structural Function (ASF):**
Coefficients on treatment interactions are not directly ATTs. Must compute:
```
ATT(g,t) = mean[ g(X_i'β̂ + δ̂_{g,t}) - g(X_i'β̂) ]  over treated units in (g,t)
```
where `g(·)` = logistic or exp. Delta method for SE propagation.

Bootstrap: multiplier bootstrap only (no wild cluster bootstrap for nonlinear).

### 5c. Aggregation Weights (exact jwdid_estat formula)

```
ω(g,t) = number of unit-time observations in cell (g,t)

simple:   Σ_{g,t: t≥g} ω(g,t)·ATT(g,t)  /  Σ_{g,t: t≥g} ω(g,t)
group:    Σ_{t≥g}       ω(g,t)·ATT(g,t)  /  Σ_{t≥g}       ω(g,t)    ∀g
calendar: Σ_{g: t≥g}    ω(g,t)·ATT(g,t)  /  Σ_{g: t≥g}    ω(g,t)    ∀t
event:    Σ_g            ω(g,g+k)·ATT(g,g+k) / Σ_g ω(g,g+k)          ∀k
```

Aggregation SEs: delta method for linear (variance of weighted sum); bootstrap
distribution used when `n_bootstrap > 0`.

---

## 6. Parallel Trends Assumptions

| `control_group` | Assumption | Pre-treatment effects |
|-----------------|------------|----------------------|
| `"not_yet_treated"` (default) | Parallel trends between each cohort and not-yet-treated units | Constrained to zero by design |
| `"never_treated"` | Parallel trends between each cohort and never-treated units | Estimable (visible in event study k < 0) |

---

## 7. Testing Strategy

### test_wooldridge.py structure

**API tests**
- Invalid `method` / `control_group` raises `ValueError`
- `get_params()` / `set_params()` round-trip
- Accessing `results_` before `fit()` raises

**Basic functionality**
- Fit on `mpdta` dataset, all fields non-NaN (`assert_nan_inference()`)
- All four aggregations callable and produce sensible output
- `to_dataframe()` and `summary()` run without error

**Methodology correctness**
- Linear ETWFE ATT(g,t) ≈ CallawaySantAnna ATT(g,t) on same data / same control group
  (tolerance ~1e-3, both theoretically equivalent under OLS / same assumptions)
- Nonlinear: simulated binary data, logit ATT sign correct
- Aggregation weight verification: manual weighted average == `simple` ATT

**Edge cases**
- `control_group="never_treated"` with pre-treatment k < 0 effects estimable
- `anticipation=1` shifts treatment window correctly
- All three covariate types passed simultaneously
- Single cohort degenerates to standard DiD

**Slow tests** (`@pytest.mark.slow`)
- Bootstrap SE convergence (`ci_params.bootstrap(n, min_n=199)`, threshold 0.40/0.15)
- Nonlinear bootstrap

---

## 8. Documentation

- `docs/methodology/REGISTRY.md`: add "WooldridgeDiD / ETWFE" section with:
  - Academic sources (Wooldridge 2021, 2023; Friosavila 2021)
  - Estimator equation (saturated model)
  - SE methods (unit-level cluster, multiplier bootstrap, wild cluster bootstrap for OLS)
  - Edge cases: nonlinear ASF computation, covariate demeaning
  - Note: `**Deviation from Stata:** nonlinear bootstrap uses multiplier (jwdid uses delta method)`
- Export as `WooldridgeDiD` and alias `ETWFE` in `__init__.py`
