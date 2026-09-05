### Fixed
- **SyntheticDiD `pre_treatment_fit` is now shape-only**: the reported
  pre-treatment RMSE (and the "Pre-treatment fit is poor" warning) previously
  measured the raw level residual between the treated mean and the synthetic
  control, while the Frank-Wolfe unit weights are fit on column-centered
  outcomes (`intercept=True`, matching R `synthdid`) and deliberately leave a
  constant level gap to the DiD step. A parallel treated series sitting at a
  different level therefore reported a large RMSE and a false poor-fit
  warning even when the ATT was recovered exactly. The RMSE is now taken on
  the pre-period residual after removing its mean, which is the data-fit
  component of the centered Frank-Wolfe objective, computed on the
  normalized outcome scale and rescaled (so a large common outcome level
  cannot perturb it); `in_time_placebo()` and `sensitivity_to_zeta_omega()`
  report the same shape-only `pre_fit_rmse`. Results pickled before this
  release are migrated on load: the shape RMSE and the level gap are
  recomputed from the stored trajectories and replace the stale level RMSE
  (which is cleared when no trajectories were stored), never relabeled. Estimates, standard errors
  and weights are unchanged.

### Behavioral Changes
- **SyntheticDiD pre-fit diagnostic redefinition**: `pre_treatment_fit` and
  the `pre_fit_rmse` diagnostic columns drop the level gap, so their values
  fall for any design with a treated-vs-synthetic level offset, and are NaN
  with a single pre-period. The poor-fit warning is now anchored to an
  in-space placebo fit reference (Abadie, Diamond & Hainmueller 2010; Abadie
  2021): the treated fit is compared with the same statistic over up to 20
  placebo fits of control units treated as if treated (Algorithm 4 draws at
  the fit-time zeta), and the warning fires when the placebo p-value is at
  or below 0.05 (at the 20-draw default: worse than every placebo draw),
  replacing the unreachable `1 x std(treated pre-outcomes)` rule. The reference is computed for every variance method (one extra
  Frank-Wolfe solve per draw) from a private RNG stream, so SE draws are
  unchanged; it needs at least 19 successful draws (`n_bootstrap >= 19`) to
  fire and is absent when no pseudo-control remains. A treated unit far
  noisier than every control still fits worse than every placebo and warns;
  the warning text says so. `summary()` labels the values
  `Pre-fit RMSE (shape)`, `Pre-fit level gap` and `Pre-fit placebo p-value`.

### Added
- **`SyntheticDiDResults.pre_treatment_level_gap`**: signed mean pre-period
  gap (treated minus synthetic), the constant offset absorbed by the DiD
  step, reported in `summary()` and `to_dict()` for inspection.
- **`SyntheticDiDResults.pre_fit_placebo_rmse` / `pre_fit_placebo_pvalue`**:
  the placebo pre-fit reference distribution and the treated fit's placebo
  p-value behind the poor-fit warning (p-value also in `to_dict()`).
