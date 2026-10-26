### Internal
- 4.0 cut, PR 5a (the reader half of cut-sequence item 5, the `M-050`..`M-058` storage
  flip): every in-repo reader of the 3.x `overall_att` / `overall_se` / `overall_t_stat` /
  `overall_p_value` / `overall_conf_int` field names (and ContinuousDiD's `overall_att_*`
  spellings) on the nine flipping containers in SOURCE and TESTS - consumers, the
  containers' own `summary()` / `to_dict()` / `__repr__` / aggregation bodies, and every
  test suite - now reads the canonical `att` / `se` /
  `t_stat` / `p_value` / `conf_int` quintet. No user-visible change: the canonical names
  already resolve as properties, numbers are bit-identical and nothing warns yet; the
  storage flip itself, the `FutureWarning` shims and the report-schema changes follow in
  PR 5b; the documentation / guides / tutorial / benchmark half of the reader migration
  is PR 5a-ii (split off because the combined diff exceeded the AI review input budget). `power.py`'s `_extract_staggered` (a silent `getattr` fallback chain over the old
  names) is deleted - the staggered power profiles use `_extract_simple`; the
  `practitioner_next_steps` `overall_att` fallback is gone. Bootstrap draw containers
  (`CSBootstrapResults.overall_att_se` and siblings) and `StaggeredTripleDiffResults`
  (unified at cut item 6) are untouched.
