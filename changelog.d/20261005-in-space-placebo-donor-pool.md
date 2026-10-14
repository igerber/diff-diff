### Fixed
- **`SyntheticControl` in-space placebo is now an exact permutation test**:
  `in_space_placebo()` now fits each placebo against all the other units, the
  real treated unit included, as Abadie, Diamond & Hainmueller (2010, §3.4,
  p. 501) do ("shifting California to the donor pool"). It used to drop the
  treated unit from every placebo's donor pool (the `SCtools::generate.placebos`
  construction); that reference set changes with the treated unit, so the
  permutation p-value was not exact. (The new p-value is exact under the null
  of no effect and random assignment, when every placebo refit enters the
  reference set; failed or infeasible refits are still excluded and reported via
  `n_failed` / `n_infeasible`.) Every placebo that puts weight on the
  treated unit now gets a different RMSPE ratio, so `placebo_p_value` and the
  placebo rows of `get_placebo_df()` change on every such fit, by whole ranks;
  the fit itself (`att`, weights, the treated unit's `rmspe_ratio`) does not
  change. On the 10-unit no-effect panel of issue #837, 5 of the 10
  assignments move by one rank (3 up, 2 down) and the 10 p-values become
  exactly 0.1, 0.2, ..., 1.0; before, 2 assignments gave `p = 0.1`. Over 300
  simulated 10-unit no-effect panels, with each unit treated in turn, the old
  construction gave `p <= 0.1` for 11.7% of the assignments (Monte Carlo SE 0.25
  points); the new one gives exactly 10% on every panel.
  `test_sharp_null()` and `confidence_set()` re-rank the new placebo gaps and
  use the treated unit's null-imputed outcome `Y_1 - f` inside each placebo's
  synthetic control (Firpo & Possebom 2018, fn 10), so their p-values and sets
  change too. With exactly 2 donors each placebo is now fit against 2 units
  (a real weight solve, and under `v_method="nested"` / `"cv"` a `V` search,
  instead of a forced `w = [1]`); fewer than 2 donors still gives a NaN p-value.
  `fit()` now processes an explicit `donor_pool` in the panel's sorted unit
  order (the order the placebo refits use) rather than the caller's list order,
  so `donor_weights` keys follow that order (`get_weights_df()` stays sorted by
  weight descending) and an exact weight tie between donors resolves the same
  way in the treated fit and in every placebo.

### Behavioral Changes
- **`SyntheticControlResults.in_space_placebo(n_starts=...)` rejects a multistart
  count that differs from the fit's under `v_method="nested"` / `"cv"`** (also
  reached through the lazy reference build in `test_sharp_null()` /
  `confidence_set()`). The treated unit's RMSPE ratio was produced at fit time
  with the fit's `n_starts`, and more starts can select a different `V` optimum
  even when every fit converges, so a different placebo budget would rank one
  statistic against references computed by another procedure and void the
  exactness of the permutation. Re-fit with the larger `n_starts` instead. The
  override stays inert under `custom` / `inverse_variance` (no `V` search).
