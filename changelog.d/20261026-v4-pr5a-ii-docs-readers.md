### Internal
- 4.0 cut, PR 5a-ii (the documentation half of cut-sequence item 5's reader migration,
  split from PR 5a because the combined diff exceeded the AI review input budget): every
  documented read of the 3.x `overall_att` / `overall_se` / `overall_t_stat` /
  `overall_p_value` / `overall_conf_int` names (and ContinuousDiD's `overall_att_*`
  spellings) on the nine `M-050`..`M-058` containers - RTD pages, the bundled
  `llms*.txt` guides, tutorial code cells, the tutorial drift tests that mirror them
  (tutorial 32's code-cell hash pins re-pinned), benchmark scripts and the carousel
  generator - now reads the canonical `att` / `se` / `t_stat` / `p_value` / `conf_int`
  quintet. No user-visible behaviour change (the canonical names already resolve);
  committed notebook outputs are unchanged except tutorial 18's echoed guidance string.
  Bootstrap draw containers (`bootstrap_results.overall_att_se`) and R / golden-JSON
  interface keys are untouched.
