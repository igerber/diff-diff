### Behavioral Changes
- **`summary(alpha=...)` never recomputes or relabels stored inference,
  family-wide** ([M-146] completion): a non-fit `alpha` now raises `ValueError`
  at seven more sites - `DiDResults` (and `SpilloverDiDResults` by
  inheritance), `MultiPeriodDiDResults`, `SyntheticDiDResults`,
  `TripleDifferenceResults`, `TROPResults`, `ContinuousDiDResults` (all
  previously printed a requested-alpha header over fit-time stored intervals),
  and `SyntheticControlResults` (previously a silent no-op `alpha`); `alpha=0.0`,
  previously swallowed by a falsy-`or` default, raises too. Re-fit at the
  desired alpha instead.
- **`plot_dose_response` honest bands and labels**: DataFrame-`se` input masks
  non-positive/non-finite `se` rows from the confidence band with a warning
  (previously a zero-SE row drew a finite zero-width band) and validates
  `alpha` strictly inside (0, 1); the band legend is alpha-derived on the `se`
  branch, `results.alpha`-derived on `results=` input, and the level-free
  "CI" for bare-curve/explicit-CI input (previously hard-coded "95% CI"
  regardless of the requested alpha); an explicitly passed `alpha` on
  non-`se` input warns instead of being silently ignored; the plotly band
  polygon filters non-finite-CI rows (a NaN vertex previously mangled the
  `toself` band) and both renderers suppress an all-masked band.
