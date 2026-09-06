### Added
- **`WooldridgeDiD(unsupported_period_action=...)`** ([M-147]): an opt-out for
  per-period comparison-support filtering. An *unsupported period* is one
  lacking the required comparison support: no positive-weight eligible
  comparison observation is observed there, so no `ATT(g, t)` at that period
  is identified. `"drop"` (the default) is unchanged: such periods are removed
  before the solve and the reduction is warned, exactly as before. `"error"`
  refuses with `ValueError` before removing any period, naming the periods,
  the would-be-dropped observation count and the cause (structural, or
  zero survey weight), for users who would rather see the refusal than
  estimate on a reduced sample. The refusal precedes the `survey_design=`
  refusal and is not gated on `rank_deficient_action`. No estimate changes
  under either value.

### Internal
- **Doc-snippet tests run inside `tmp_path`**: `tests/test_doc_snippets.py`
  previously executed snippets with the repository root as the working
  directory, so `savefig('<name>.png')` calls in the API docs wrote PNGs into
  the checkout (four such files were caught in review). Snippet side effects
  now land in the per-test temporary directory.
