### Documentation
- **Tutorials 02/04/05/07/23 are rebuilt on `TwoWayFixedEffects().fit(..., event_study=True)`**
  (`spec="within"`: unit + period fixed effects, unit-clustered SEs) ahead of the
  MultiPeriodDiD removal (`M-010`, 4.0 cut PR 4c). Point estimates match the former
  MultiPeriodDiD fits on these balanced panels; the SEs change because the within spec
  clusters by unit. Tutorial 23's naive headline is the equal-weight average of the two
  post-period coefficients computed in the cell (ATT unchanged at -4.29; the SE is now the
  unit-clustered one), and tutorial 02's period-effects loop reads `to_dataframe()`.
- The static `TwoWayFixedEffects().fit(..., time=...)` calls in tutorials 01/02/09 (the
  `M-082` alias, misused there as the post dummy) are replaced: 01 passes `post="post"`;
  02 and 09 report `TwoWayFixedEffects().decompose(...).twfe_estimate`, the unit + period
  fixed-effects coefficient their Goodman-Bacon sections decompose (the old cells printed
  `treated x period` artefacts: 0.6309 and 0.0001). Tutorial 02 cell 36's covariate draw is
  seeded. Tutorials 02 and 09 are re-executed on the Rust backend, which produced the
  committed outputs, so the seeded bootstrap cells are unchanged.
- Tutorials 08 and 16 describe `StaggeredTripleDifference` as the removed class (4.0 cut,
  PR 4a). `diff_diff/guides/llms-full.txt`'s event-study snippet calls `print(es.summary())`
  (`EventStudyResults` has no `print_summary`).

### Testing
- `tests/test_t23_spillover_tva_drift.py`'s `naive_fit` fixture mirrors the notebook's TWFE
  event-study headline cell, runs the fit with warnings escalated to errors, and pins the
  post-period-average identity and the unit-clustered vcov-block SE alongside the existing
  -4.29 endpoint.

### Internal
- `docs/v4-design.md` cut-sequence item 4b marked shipped; item 4c lists the two remaining
  markdown-only MultiPeriodDiD mentions (06 cell 34, 16 cell 40). `docs/doc-deps.yaml` maps
  the TWFE module to tutorials 01/02/04/05/07/09/23. No ledger row flips.
