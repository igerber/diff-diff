# DiD OVB Sensitivity Design

## Goal

Add a clean-room Python implementation of the omitted-variable-bias sensitivity
framework for canonical two-period DiD, integrated with `diff-diff`'s existing
learner and result conventions and validated against an independent R oracle.

## Scope

The first release targets the paper's canonical DiD application rather than a
general replacement for the GPL-3 `dml.sensemakr` package. It will provide:

- cross-fitted estimates of the short ATT, residual trend variance, and observed
  selection scale;
- the OVB decomposition into scale, trend, selection, and alignment factors;
- bias bounds and delta-method confidence intervals under user-supplied
  restrictions;
- point-estimate, confidence-interval, and contour-grid RV/XRV calculations;
- a structured results object with `summary()`, `to_dict()`, and
  `to_dataframe()`;
- a deterministic R/Python parity fixture using the same canonical two-period
  data and nuisance specifications.

The implementation will not copy R source, function bodies, tests, or package
layout from `dml.sensemakr`. It will cite Wang et al. and the R package as an
independent reference implementation. The existing `DMLDiD` estimator remains
unchanged in the first slice; integration is through a new estimator/module so
existing behavior stays bit-stable.

## Mathematical contract

For observations with outcome change `delta_y`, treatment `d`, and covariates
`x`, the module estimates the paper's short parameter and scale components:

\[
  \theta_s = E[\Delta Y-g_s(X)\mid D=1],\quad
  \sigma^2_{0s}=E[(\Delta Y-g_s(X))^2\mid D=0],\quad
  \nu^2_{0s}=E[(O_X/O)^2\mid D=0].
\]

The reported scale is `S0 = sqrt(sigma2_0s * nu2_0s)`. Given restrictions
`abs(rho) <= rho_max`, `C_delta_y <= trend_max`, and `C_d <= selection_max`,
the bias radius is `rho_max * trend_max * selection_max * S0` and the bound
estimates are `theta_s +/- radius`. Benchmarking against an observed covariate
and pre-trend extrapolation are separate, explicit result methods and are not
silently mixed into the baseline bound.

RV/XRV follow the paper's definitions: they are the smallest common selection
and trend strength compatible with the requested null entering the widest
confidence interval; if the original confidence interval already contains the
null, both values are zero. The implementation exposes the search grid and the
resulting endpoint calculations so parity failures are diagnosable.

## API and data flow

The public surface will be:

```python
from diff_diff import DIDOVBSensitivity

result = DIDOVBSensitivity(n_folds=5, seed=42).fit(
    data, outcome="y", treatment="treated", time="post",
    covariates=["x1", "x2"], unit="unit",
)
result.bounds(trend_strength=1.0, selection_strength=0.25)
result.robustness_value(null_value=0.0, alpha=0.05)
result.contour(null_value=0.0, alpha=0.05)
```

The estimator accepts either a supplied `outcome_change` column or a balanced
two-period panel from which the change is constructed. Nuisance learners use
the repository's existing learner factory and cross-fitting helpers. No hidden
network download is performed by the estimator.

## R parity

The parity harness will use a committed, small canonical fixture and an R script
under `benchmarks/R/`. It will pin fold assignments, seed, clipping, learner
configuration, and null/alpha settings. The Python test compares the short ATT,
sigma/nu scale components, decomposition radius, bounds, and RV/XRV against
machine-readable R output with tolerances documented next to the fixture.
Tests requiring R will skip explicitly when `Rscript` or the required R package
is unavailable; they will never replace missing R output with Python-generated
goldens.

## License and attribution

The Python implementation is original MIT-licensed work in this repository.
The paper, `dml.sensemakr`, and `CS_RR` are cited in module and documentation
references. Any reused MIT-licensed `CS_RR` material or data receives its own
copyright/license notice. GPL-3 source from `dml.sensemakr` is not copied or
translated mechanically.

## Non-goals for the first release

- porting every general-purpose ATE/ATT feature in `dml.sensemakr`;
- reproducing R plotting objects or R-specific S3 classes;
- adding automatic data downloads;
- changing the existing `DMLDiD` result schema;
- pushing to `upstream` before local tests, R parity, and review are complete.
