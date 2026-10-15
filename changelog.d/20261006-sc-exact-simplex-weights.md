### Fixed
- **`SyntheticControl` donor weights are the exact simplex least-squares
  solution** (#838): the inner weight solve, run at every `V` the nested
  search tries and in every placebo refit, was Frank-Wolfe with an iteration
  budget (`inner_max_iter=10000`) that ordinary factor-model panels often
  exhausted. The treated fit then warned "did not converge" and the
  diagnostics that need a converged fit were skipped (`placebo_p_value` NaN);
  placebos whose refit hit the budget were dropped from the reference set; and
  the nested search ranked `V` only among the solves that met their stopping
  rule. The problem reduces exactly to one non-negative least squares, solved
  by Lawson-Hanson's active set, which terminates in finitely many steps, so
  the solve now reaches the optimum and none of those paths fires for it (a
  solve that exhausted the active set's step guard, not reached in practice,
  would return a feasible iterate flagged non-converged and take the same
  fail-closed paths as before). On the issue's
  three designs, 40 panels each, the placebo test had failed closed or lost
  placebos on 40, 40 and 33 panels; it now keeps every placebo on all 120, and
  the treated pre-RMSPE equals the exact optimum to rounding.
  - The same solve replaces Frank-Wolfe in the conformal proxy
    (`conformal_test()`, `conformal_average_effect()` and
    `conformal_confidence_intervals()`: `proxy_converged` is true and
    `n_grid_nonconverged` is 0 in practice) and in the informational `synthetic_weight`
    column of `rank_control_units()`, which stopped at 1000 iterations; that
    column's values move, the ranking does not. A panel with a missing
    pre-period outcome keeps its uniform `synthetic_weight` column.

### Behavioral Changes
- **`SyntheticControl` results move by the former solver's gap** (#838): the
  donor weights, `att`, `pre_rmspe`, and the placebo, leave-one-out, in-time,
  sparse and conformal results. On generic panels the change is small: the
  issue's treated pre-RMSPE was a median 1e-4 above the optimum, relative.
  Where the treated unit's predictors lie inside the donors' convex hull,
  every `V` fits them exactly, so the predictors identify neither the weights
  nor `V`. The solve then returns a vertex of the optimal face; Frank-Wolfe
  stopped wherever its budget left it, and R `Synth`'s interior-point solver
  tends to a point inside the face. The vertex is the same under every SciPy
  version: the active set is diff-diff's own, because `scipy.optimize.nnls`
  returns a different vertex at a tie from one release to another (five
  different sets of T25's placebo ratios over SciPy 1.11 to 1.18). The T25
  tutorial panel is such a case: its weights go from five donors plus 0.0043
  on each of the other 15 to four donors, `att` from 6.79 to 6.69 and
  `pre_rmspe` from 0.987 to 0.934. On the Basque panel at R's `V`, the weights
  match R `Synth` to 4.5e-6 (9.6e-5 before).
- **Conformal p-values and intervals move** (#838): with the exact proxy the
  intervals are narrower on simulated panels and still cover above nominal.
  Over 50 panels (300 period intervals at nominal 90%), coverage goes from
  0.970 to 0.953 and the median width from 0.515 to 0.450.

### Deprecated
- **`SyntheticControl(inner_max_iter=, inner_min_decrease=)`** (#838, ledger
  M-148 / M-149): both now default to `None` and emit a `DeprecationWarning` when
  passed; the value is discarded (`get_params()` reports `None`). The inner
  weight solves have no iteration budget or stopping tolerance to set. Drop the
  arguments; they are removed in 5.0.
