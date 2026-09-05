### Added
- **WooldridgeDiD comparison-support policy** ([M-147]): set
  `unsupported_period_action="error"` to refuse periods lacking eligible comparison
  support before removing them. The default `"drop"` preserves filtering and warnings.
  The option works across OLS, logit and Poisson independently of
  `rank_deficient_action`; results record the fit-time policy in `summary()` and
  `to_dict()`. Existing survey and identification checks remain active.
