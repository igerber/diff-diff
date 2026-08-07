# Diagnosis: `dr_ml_attgt` Parametric Parity

## Symptom

On the shared 20-unit two-period bad-control fixture, the Python adapter
returns `ATT = 5.000000`, while the installed R
`badcontrols::dr_ml_attgt(..., nuisance_method="parametric")` returns
`ATT = 5.003463`. The difference is larger than the exact parity tolerance.

## Minimal Reproduction

- Two periods: `period = 0, 1`
- Ten treated and ten untreated units
- Bad control: `X`
- R call: `dr_ml_attgt(xformula=~1, bad_control_formula=~X,
  nuisance_method="parametric")`
- Python call: `dr_ml_attgt(xformula="~1", bad_control_formula="~X",
  nuisance_method="parametric")`

## Root Cause

The R implementation assigns cross-fitting folds with unseeded `sample()` in
`references-badcontrols/R/dr_ml.R` (the treated and comparison groups are
folded independently). The Python implementation uses a deterministic fold
assignment. Parametric nuisance predictions are therefore evaluated on
different held-out folds, so the finite-sample ATT differs even though the
score algebra is the same.

## Fix Status

No estimator change applied. Relaxing the tolerance or changing the score
would launder a fold-assignment mismatch into a false parity claim.

## Verification

- Python badcontrols/ptetools/twfeweights/docs target suite: `75 passed`.
- Ruff, Black, and Mypy: passed.
- Existing R parity fixtures remain unchanged.

## Prevention

Add a parity-only fold ingress to both implementations: generate one explicit
fold-id vector, pass it to R and Python, and compare the nuisance predictions,
ATT, and influence function under that shared fold assignment. Until that
protocol exists, this adapter is contract-compatible but not an exact numeric
parity surface.
