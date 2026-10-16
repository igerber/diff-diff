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
