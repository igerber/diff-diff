### Documentation
- Corrected the `CallawaySantAnnaResults.group_time_effects` entry in the
  bundled `llms-full.txt` guide: each ATT(g,t) cell is a plain dict with the
  keys `effect`, `se`, `t_stat`, `p_value`, `conf_int`, `n_treated`,
  `n_control` and `skip_reason` (read as `cell["effect"]`), not a
  `GroupTimeEffect` object. The `GroupTimeEffect` section of the staggered API
  page now says the same. No behavior change.
