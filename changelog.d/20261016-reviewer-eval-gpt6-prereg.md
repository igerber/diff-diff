### Internal
- **Codex reviewer gpt-6 campaign pre-registration (PR-1 of 2)**: `tools/reviewer-eval`
  pre-registers the `gpt-5.6-sol` -> `gpt-6-astra` evaluation (`DECISION_RULE.md`:
  symmetric regression gate over every ground-truth bug, FP gate, downgrade rule,
  verifier-confirmed leaks void a cell and contest the gate, stop conditions; the
  2026-07 rule is archived under `campaigns/`) and splits the grader-facing text into
  `GRADING_RUBRIC.md` (marker-based `LEAK?` flag whose list renders from
  `reviewer_audit.LEAK_RULES`). New engines: `reviewer_verdict.py` (gates, precedence,
  binding to the sealed artifacts, protocol guards, PROVISIONAL/DEFINITIVE) and
  `reviewer_audit.py` (post-unblinding session-log audit, informational and
  over-matching by design). `run_eval.py` gains `audit` and `verdict`, runs every
  codex call under an isolated CI-faithful `CODEX_HOME`, records seven-key git
  provenance per artifact and in the manifest, seals artifacts at run time
  (`artifacts_sha`), binds the blinded bundle (`bundle_id`, `run_ids_sha`,
  rubric/prompt sha) and refuses post-run stages on a dirty or moved tree; `run`
  holds an exclusive per-subdir lock for the whole attempt, records the OK cells of
  aborted attempts too and refuses a rerun that would re-dispatch any cell an
  earlier attempt recorded OK; review worktree leaves are salted opaque hashes. Harness CLI re-pinned to
  codex-cli 0.162.1 (arm `E` = gpt-6-astra @ xhigh added; CI `codex-version` pinned
  to the same, lockstep-tested with `DEFAULT_MODEL`, the plan-review pin, the
  control arm and the prose pins).
- **`openai_review.py`**: `call_codex` parses codex's stderr session events into
  `session_commands` (+ start/exec/other-event counts, truncation flag) on every
  exit path and prints a one-line summary; `--timeout` now also caps the codex
  backend's wall clock when passed explicitly (previously ignored there), `0`
  means no timeout on both backends and negatives are rejected; `PRICING`
  corrected for gpt-5.6-* and extended with gpt-6-*; gpt-6* counts as a
  reasoning family; blinding scrubs gpt-6/astra and Unicode-dash spellings.
