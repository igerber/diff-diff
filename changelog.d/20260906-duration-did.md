### Added
- **Duration DiD**: `DurationDiD` estimates cumulative absorption effects in balanced
  individual panels with common treatment timing under common dynamics or proportional
  untreated hazards. Includes pooled individual-bootstrap pointwise/simultaneous inference,
  a stored fixed-anchor hazard pretest, owned results and event-study aggregation, native
  reporting, and an executed tutorial. Unsupported domains and failed bootstrap families
  retain explicit availability metadata; no survey, covariate or cluster extensions.
  Constructor parameters are revalidated on every fit, including after direct attribute
  updates; invalid configurations raise without replacing a previous fitted result.
