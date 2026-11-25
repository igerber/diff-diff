### Fixed
- **`SyntheticControl` conformal confidence intervals no longer depend on a
  heuristic grid span** (`conformal_confidence_intervals()`,
  `conformal_average_effect()`; Chernozhukov-Wuthrich-Zhu 2021). The
  automatic range was `max(8 x sd(pre residuals), 0.5 x |point estimate|)`
  around the point estimate, scanned at `n_grid` points, with the outermost
  accepted grid points reported as the endpoints. Three consequences are
  fixed:
  - when the effect was large relative to the residual scale the grid step
    exceeded the interval: at effect / residual sd of 50 the reported 90%
    interval was 17% too narrow, and at 500 or more no grid point fell in the
    set and the result was a false `status="empty"` with NaN endpoints;
  - when the pre-period fit was near exact and the point estimate near zero
    (typical for the block-collapsed average-effect interval with as many
    donors as pre-blocks) the range was far too narrow and the result was a
    `grid_limited` sliver of the set;
  - the endpoints were always an inner approximation, short by up to one grid
    step per side (4-5% of the width in ordinary cases), on explicit
    `bounds=` as well.
  The scan now covers a range derived from the data that contains the finite
  boundary of the set (the proxy weights lie on the simplex, which bounds
  every residual), the point estimate is itself evaluated as a candidate,
  and each endpoint is located by bisection to 1e-6 of the interval width.
- **`status="unbounded"` is now decided by the attainable p-value floor of
  the single-post-slot test** rather than `1/|Pi|`. With one tested slot
  among `m`, the p-value cannot fall below the share of permutations that
  leave that slot in place: `1/m` for the moving-block set and for an exact
  i.i.d. enumeration alike. Under `scheme="iid"` the old check let such cases
  through and reported a finite `grid_limited` interval for a set that is the
  whole line. The warnings and docstrings no longer advise `scheme="iid"` as
  a way to obtain a finer p-value for these two intervals (it refines only
  the joint `conformal_test`).

### Behavioral Changes
- **`SyntheticControl` conformal intervals move**: reported intervals widen
  to the true boundary of the accepted set (by up to one former grid step
  per side), and cases formerly reported as `empty` or `grid_limited` on the
  default range now return `ran` with finite endpoints. On the default range
  `grid_limited` now arises only when an edge candidate's proxy fit does not
  converge. When `alpha` is below the single-slot floor the result is
  `unbounded` with infinite endpoints instead of a finite `grid_limited`
  interval. Non-contiguous sets, which occur on near-exact pre-period fits,
  are now seen in full and reported as their hull with `contiguous=False`.
- **`get_conformal_grid_df()` lists every candidate evaluated**, one row per
  distinct value: the rows of each scan, the point estimate, and the
  endpoint-refinement candidates. Its row count is therefore no longer
  `n_grid` (which is now the number of points per scan), `lower` / `upper`
  are rows of it, and `n_grid_in_set` / `n_grid_nonconverged` count its rows.
  Outcomes too large for a finite candidate range raise `ValueError`
  (previously a scan over non-finite values).
- **`bounds=(lo, hi)` whose width `hi - lo` overflows float64 now raises
  `ValueError`** in `conformal_confidence_intervals()`,
  `conformal_average_effect()` and `confidence_set()` (for example
  `(-1.7e308, 1.7e308)`). The grid built from such a pair contained NaN
  values. A pair with a finite width is still accepted, including one that
  ends at the largest float.
