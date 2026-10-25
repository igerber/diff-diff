# Test fixtures

## `codex_stderr_0_162_1*.txt` - sanitized codex exec stderr transcripts

Event-structure transcripts of `codex exec` (codex-cli 0.162.1) used by
`tests/test_openai_review.py` to pin the `call_codex` session parser against
the live stderr format:

- `codex_stderr_0_162_1.txt` - the 2026-10 control-arm harness smoke
  (`run_eval.py smoke`, gpt-5.6-sol @ xhigh, isolated `CODEX_HOME`): 8 shell
  commands, no non-shell event.
- `codex_stderr_0_162_1_websearch.txt` - the step-0 web-search probe
  (gpt-6-astra @ xhigh, prompt asking for a web search then `echo ok`, empty
  `--cd` dir): codex printed `web search: <query>` PREFIX lines, so the
  `session_other_events` counter uses `OTHER_EVENT_PREFIX_RE` as well as the
  bare-word `OTHER_EVENT_HEADERS` set.

A committed transcript keeps ONLY the event structure: every event-header line
(`codex`, `exec`, `user`, `thinking`, `tokens used`, `web search: ...`, any
other bare-word header), every `exec` START line (`<shell> -lc '...' in <cwd>`,
command text kept so the parser test is real) and every STATUS line
(` succeeded in Nms:` etc.). Every other run of consecutive lines is collapsed
into ONE placeholder named by the header that precedes it: `<PROMPT REDACTED>`
after `user` (the prompt echo carries the case diff), `<TEXT REDACTED>` after
`codex`/`thinking` (review prose), `<OUTPUT REDACTED>` after a status line
(command output), `<OTHER REDACTED>` otherwise (harness logging, token counts).
Absolute paths in the kept lines are rewritten to `<PATH>`. Line counts
therefore differ from the raw transcript; header, start and status counts do
not, which is all the fixture tests assert. The sanitizer is a one-off snippet
run in the authoring session (not committed code); re-create a fixture by
capturing `2> <file>` from a real run and applying the same rules.

There is deliberately NO 0.153.4 transcript: the format was first observed on
that CLI, but the live pin is 0.162.1 and the older CLI is no longer installed.

## `v4_aggregate_oracles.json` - pre-removal aggregation oracles (4.0 PRs 3a-3c)

Numeric reference for `tests/test_aggregate_contract.py`, captured by
`tests/_capture_v4_aggregate_oracles.py` on the UNMODIFIED tree at the commit
named in `provenance.captured_at_head` - i.e. while the deprecated fit-time
`fit(aggregate=...)` / `fit(balance_e=...)` route still existed and could serve
as an independent second implementation of every aggregation surface. Layout:

- `analytic` - per estimator, the `fit(aggregate="all")` (ContinuousDiD:
  `"eventstudy"`) and `balance_e=1` surfaces on every design arm the contract
  file exercises. The NumPy and Rust captures agreed to rtol 1e-12 /
  atol 1e-13, so one copy is stored; the post-fit kit recompute must reproduce
  it after the removal.
- `boot` - keyed by weight backend (`"numpy"` / `"rust"`, the
  `effective_weight_backend()` vocabulary), because the seeded multiplier
  stream differs per backend: the fit-time bootstrapped surfaces the CS/EDiD
  replays and the Imputation/TwoStage/ContinuousDiD relays must reproduce, plus
  the overall quintets of bootstrapped PLAIN fits, which the always-on surface
  materialization must leave unchanged. A pin whose backend key is absent skips.

Rows are stored as `{label: {effect, se, t_stat, p_value, ci_lo, ci_hi,
n_obs|n_groups, df_used, cband_lo/hi}}` with NaN serialized as `null` (the file
is written by the script's `dump_fixture()`: sorted keys, one indent level per
nesting level, every leaf row / label list / matrix row on one line); labels
are stringified numbers (`repr(float)` for fractional horizons). Every
event-study bundle also carries `ref_labels`, the labels the post-fit container
marks `is_reference` (the legacy zero-count sentinel rows where the post-fit
route was closed at capture): the contract test pins reference MEMBERSHIP
against it, so a surface that mis-marks an estimated horizon as reference
fails instead of having that row skipped. Re-capturing
on a post-removal tree is NOT a substitute: the fit-time route is gone, so
nothing independent remains to disagree with.

## `v4_twfe_event_study_oracles.json` - TWFE event-study decoupling oracles (4.0 PR 4c-i)

Numeric reference for `tests/test_twfe_event_study_decoupling.py`, captured by
`tests/_capture_v4_merge_mpd_oracles.py` on the UNMODIFIED tree at the commit
named in `provenance.captured_at_head` (`68e383bd`, origin/main at the PR 4c-i
branch cut) - i.e. while `TwoWayFixedEffects.fit(..., event_study=True)` still
built its surface through an internal `MultiPeriodDiDResults` and
`results_base._from_mpd`, and while `MultiPeriodDiD` still existed as the
independent twin of every lane. Layout:

- `provenance` - capture SHA, `diff_diff.__version__`, weight backend,
  numpy/scipy/pandas/python versions.
- `lanes.<name>` - one entry per lane of
  `tests/helpers/twfe_es_oracle_lanes.py::LANES` (pooled default / explicit
  cluster / hc2_bm; within + explicit cluster / hc2 carve-out / Conley /
  survey TSL / survey JK1 replicates; string period labels; a non-suffix
  partition; a rank-deficient NaN post period). Each carries `echo` (the
  lane's panel name, constructor and fit kwargs and survey label - the test
  asserts it equals the live lane, so an edited lane fails as lane drift, not
  as a numeric regression), `twfe` (every `dataclasses.fields(EventStudyResults)`
  value of the TWFE event-study surface) and `mpd_twin` (the same for
  `_from_mpd(MultiPeriodDiD(...).fit(...))`).

Arrays are stored as `{"dtype": <numpy dtype string>, "values": [...]}` (the
real dtype, e.g. `"<i8"`, `"<f8"`, `"<U2"`, `"|b1"`; 2-D arrays as nested
lists) with NaN serialized as `null`; tuples as lists; scalars verbatim. The
NumPy and Rust captures agreed to rtol 1e-12 / atol 1e-13 (CI bounds 1e-9),
so one copy is stored. The consumer checks `t_stat` against the bound those
tolerances imply for a ratio, `atol (1 + |t|) / se + 2 rtol |t|`, not a flat
atol: Linux OpenBLAS and Windows drift 1.6e-13 / 2.4e-13 from the macOS
Accelerate capture on the rank-deficient lane's near-zero effect (`se` 0.29),
where `att` and `se` each pass at atol 1e-13. Re-capturing on a post-4c-ii tree is NOT a substitute:
`MultiPeriodDiD` is gone there, so the `mpd_twin` half has no independent
producer left.
