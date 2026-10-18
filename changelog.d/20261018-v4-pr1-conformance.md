### Added
- **`LWDiDResults.summary(alpha=None)` / `print_summary(alpha=None)`** join the family-wide
  summary-alpha guard (ledger row `M-146`): a value other than the fit-time `alpha` raises
  `ValueError` instead of being ignored; the methods previously took no `alpha` at all.
- **`LWDiDResults.to_dataframe(level=...)` / `to_csv(path, level=...)`** gain the uniform
  `level` selector (`"simple"` default, `"group"`, `"event_study"`) - see Breaking Changes.
- **`DurationDiDResults.aggregate("simple")`**: a verbatim relay of the stored headline
  quintet (`n = n_units`, `df` NaN - inference is bootstrap-only), joining the spec section 6
  `simple` contract every aggregation adopter honours.
- **`RDDensityTestResult.to_dict()`**: JSON-serializable scalars with native casts (NaN
  inference passes through), the `*_conventional` block `None` unless `report_all=True`, and
  the bandwidth-selection table nested under `bandwidths` only when it was estimated.
- `diff_diff._deprecation.deprecated_field_property` takes `removed_in=` so the warning and the
  generated docstring name the row's actual removal release.

### Deprecated
- **`LWDiDResults.pvalue` and `LWDiDResults.ci`** (ledger rows `M-151`, `M-152`): the
  3.10-born alias properties for `p_value` / `conf_int` now emit a `FutureWarning` and are
  removed in 5.0. Read `.p_value` / `.conf_int`.

### Breaking Changes
- **`LWDiDResults.to_dataframe()` is reshaped onto the shared results schema** (ledger row
  `M-153`): each `level` returns `aggregate(level).to_dataframe()` - columns `level, label,
  target, att, se, t_stat, p_value, conf_int_lower, conf_int_upper, n, weight, df` (the
  event-study schema for `level="event_study"`) - plus LWDiD's `rolling`, `estimation_method`
  and `vcov_type` columns. The default `"simple"` is ONE headline row; the former per-cohort
  rows are `level="group"` (without the old trailing `"Overall"` row). The `term` / `cohort`
  column is now `label`, `n_treated` is the schema's `n`, `n_obs` and `n_control` are no longer
  in the frame (both stay in `to_dict()`), and `ci_lower` / `ci_upper` are
  `conf_int_lower` / `conf_int_upper`. `to_csv()` follows and takes the same `level`.
