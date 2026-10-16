# Codex Reviewer Comparison Harness

A small local tool for deciding whether to upgrade the Codex PR reviewer (e.g.
`gpt-5.5 → gpt-5.6-sol`) **before** it goes live. It runs the control and
candidate arms over a corpus of real diff-diff review cases, saves each arm's
raw review, and emits a side-by-side bundle you (or an LLM) read into a
caught / missed / false-positive table — optionally **blinded** (arm identities
stripped) so graders can't favor a model.

**Status:** harness supports N-arm matrices (current config: a 5-arm file; the
2026-10 campaign is `--configs B,E --k 2`, control gpt-5.6-sol vs candidate
gpt-6-astra, outcome: pending), per-arm repeat counts, blinded grading sealed to
the run artifacts, a post-unblinding session-log audit and a mechanical verdict
CLI. Local-only — not wired to CI. The go/no-go rule is pre-registered in
`DECISION_RULE.md`; the grader-facing rubric is `GRADING_RUBRIC.md` (graders
never see the rule or any campaign history). ChatGPT-account `codex login`
rejects `gpt-6-sol`, so a cheaper-tier probe arm is parked (DEFERRED.md).

## Why this exists

An early reviewer update regressed by *missing real issues*, and it changed the
model **and** the prompt at once — so the cause couldn't be isolated. This harness
keeps upgrades empirical: vary **only** the declared treatment fields, reproduce
the CI invocation faithfully, and compare every arm on the same cases.

## Layout

```
tools/eval_core/         # shared engine: models, store, runner, compare (+ blinding)
tools/reviewer-eval/
├── run_eval.py          # CLI: verify-corpus · smoke · run · compare · audit · verdict
├── reviewer_verdict.py  # mechanical gates + binding/protocol guards (DECISION_RULE.md as code)
├── reviewer_audit.py    # session-log audit: the ONE leak-marker list (LEAK_RULES) + coverage classes
├── adapters/            # diff-diff bindings: ci_prompt, codex_reviewer (+ isolated CODEX_HOME), corpus_loader, worktree (+ git_provenance)
├── config/configs.json  # the arms (one role=control) + declared treatment_fields
├── DECISION_RULE.md     # pre-registered gates (grader rubric is separate)
├── GRADING_RUBRIC.md    # grader-facing: definitions, LEAK? flag, marker list, output JSON
├── campaigns/           # archived rules + verdicts (2026-07-gpt56-*, 2026-10-gpt6-*)
└── corpus/              # cases/{s1_synthetic, s2_historical, s3_negative, s4_missed} + schema + synonyms
```

The engine lives in `tools/eval_core/` (shared across eval harnesses; this tool
was its first consumer) — generic glue with no diff-diff coupling.

## Usage

```bash
# 1. Verify the corpus materializes (no codex; fast)
python tools/reviewer-eval/run_eval.py verify-corpus

# 2. Smoke test (1 case, first real codex call) for the CONTROL only. Bare
#    `smoke` exercises the role=control arm (derived from configs.json). The
#    candidate's first corpus run is the campaign itself (pre-registration by
#    construction); `smoke` REFUSES a registered candidate arm, and `run`
#    accepts one only as the exact registered campaign shape below. A review
#    runs 2-15 min.
python tools/reviewer-eval/run_eval.py smoke

# 3. Full matrix — k=2 repeats on control vs candidate (one run = one experiment)
python tools/reviewer-eval/run_eval.py run --subdir gpt6 --configs B,E --k 2

# 4. Emit the side-by-side bundle to grade (+ the identity-stripped, sealed one)
python tools/reviewer-eval/run_eval.py compare --subdir gpt6 --blinded

# 5. After unblinding: the session-log audit (outputs go OUTSIDE the repo or
#    under the gitignored runs/ — a root-level file dirties the tree and trips
#    the identity guard) and the mechanical verdict (PROVISIONAL without
#    --audit-hits, DEFINITIVE with it)
python tools/reviewer-eval/run_eval.py audit --subdir gpt6 -o /tmp/<scratch>/hits.json
python tools/reviewer-eval/run_eval.py verdict --subdir gpt6 --grades <reconciled.json> \
    --control B --candidate E [--audit-hits /tmp/<scratch>/hits.json] [--expect-head <sha>]
```

Run artifacts and the bundles land under `runs/` (gitignored). `--blinded` also
writes `comparison.blinded.md` (with a visible Bundle ID graders copy into their
table) plus a sealed `blinding.json` (the label→arm mapping, `bundle_id`,
`run_ids_sha`, `artifacts_sha`, `rubric_sha256`, `prompt_sha256`) — graders
read ONLY the blinded bundle; `verdict` re-derives every binding and refuses a
grades file, bundle, hits file or artifact set that drifted.

Every harness codex call runs under an **isolated `CODEX_HOME`** that mirrors
CI's codex-action home (the credential plus `sandbox_mode = "read-only"` and
`approval_policy = "never"`; no MCP servers, hooks or memories; codex's default
features otherwise). Progress during a run: read `done:`/`INFRA` lines from
stdout and count artifact files; the manifest exists from the START of an
attempt as a `{"failed": true}` marker and is replaced only at the end.

### Provenance, sealing and the audit

Each artifact's `usage` records the seven `git_provenance` keys sampled when
its review started (`repo_head_sha`, `repo_tree_sha`, `repo_dirty`,
`repo_branch`, `repo_common_dir`, `repo_worktree_paths`, `repo_ref_names`),
its absolute `worktree_dir`, the isolated `codex_home_path` /
`codex_home_isolated` flag, and `session_commands` (the shell command lines
codex executed, parsed from its stderr by `openai_review.call_codex`, with
start/exec/other-event counts and a truncation flag). The manifest records the
same seven keys at write time plus aggregates (`run_head_shas`,
`run_head_missing`, `run_dirty_any`, `run_marker_mismatch`, `run_not_isolated`)
and `artifacts_sha`, which seals the artifacts the moment `run` exits.
`compare --blinded`, `audit` and `verdict` refuse unless the working tree is
clean, carries no ignored file under a registered surface, and sits at the
manifest's HEAD and tree sha (engine identity = tree identity).

A campaign runs on a FROZEN PR HEAD; any push voids it, and a voided campaign
is re-run into a FRESH subdir (`gpt6-r2`, ...) with a matching local tag
(`campaign-gpt6-r2`), never the old subdir. The campaign-global
`runs/campaign-ledger.jsonl` records every dispatch of the registered contrast
(first record = frozen exposure revision); `verdict` labels a re-exposure or a
post-exposure registered-surface change (beyond the single `CALL_TIMEOUT_S`
line) as RE-REGISTERED and non-mechanical. Stop conditions are enforced by
`run` through the append-only `runs/<subdir>-attempts.jsonl` (a START record
before dispatch, an END record with the per-cell failures and the OK cells'
artifact hashes after, aborted attempts included): at most 3 reruns after the
first attempt, a (case, arm, repeat) that times out in 2 consecutive attempts
stops the campaign, and a rerun that would re-dispatch any cell an earlier
attempt recorded OK is refused (all before any codex call). `run` also holds
an exclusive per-subdir lock (`runs/<subdir>.lock`) for the whole attempt, so
a concurrent `run` into the same subdir is refused rather than interleaved.

**Answer-key exposure (accepted, both arms equally):** cases are materialized
as detached worktrees INSIDE the checkout, sharing its object DB and refs, so a
reviewer with a shell can reach fix commits, the corpus and sibling run
artifacts. Review worktree leaves are salted opaque hashes (no case or arm id in
any path). The grading rubric's marker-based `LEAK?` flag (verifier-confirmed
leaks VOID the cell and contest the gate) plus the NON-gating post-unblinding
session-log audit (`audit`; informational, over-matching by design, best-effort
because codex's default non-shell channels such as web search leave no shell
trace - on codex-cli 0.162.1 BOTH arms web-search routinely, so expect most
cells `unaudited` and the audit vacuous; accepted by user decision, see
`DECISION_RULE.md`) cover this campaign; OS-level isolation is a TODO.md row.

## How scoring works

Each arm produces a raw markdown review. `compare` collates, per case, the
ground-truth bugs followed by every arm's **raw** review (plus a grading
instruction pointing at `.github/codex/prompts/pr_review.md` for the severity
rubric all arms were given). An LLM — a subagent, or you in-conversation —
reads that bundle top-to-bottom and fills the caught / missed / FP table. No
regex parsing of review prose: free-form, model-specific output (models format
findings differently) is read directly. For the grader protocol (definitions,
the `LEAK?` flag, the output JSON) see `GRADING_RUBRIC.md`; for the gates,
reconciliation and the pre-registered decision rule see `DECISION_RULE.md`.

## What faithful reproduction means

The candidate is measured the way CI will run it: `adapters/ci_prompt.py`
rebuilds the CI prompt (the **current** `pr_review.md` — the prompt under
validation, identical for all arms; deliberately NOT base-sourced, see
`adapters/ci_prompt.py` — + `git diff --name-status` +
`--unified=5` with the same pathspec exclusions; REGISTRY is **not** inlined —
Codex reads it from the worktree), `adapters/worktree.py` materializes each case
in a detached worktree, and `adapters/codex_reviewer.py` reuses the production
`openai_review.call_codex` with byte-identical flags. The runner asserts the
Codex CLI version is identical across arms and that the arms differ only in the
declared `treatment_fields` (model, and for the gpt-5.6 matrix also effort),
each arm in a clean single-field contrast.

## Corpus strata

- **S1 synthetic** — injected bugs (revert a real fix). Sanity floor.
- **S2 historical** — real PR states just before a confirmed bug was caught.
- **S3 negative** — clean PRs (precision controls; any finding above the case's `allow_severities` = false positive).
- **S4 missed** — real bugs the AI reviewer *missed* (the failure-mode probe).

## Known limitations

- **Tutorial-notebook cases are supported with CI-equivalent context.** CI
  special-cases only `docs/tutorials/*.ipynb`: it excludes them from the diff *and*
  appends a sanitized `<notebook-prose>` block extracted via
  `tools/notebook_md_extract.py`. The harness reproduces both (same per-output /
  per-notebook / aggregate caps, fail-soft extraction, truncation marker,
  zero-extracted fallback, close-tag sanitization — see
  `ci_prompt.build_notebook_prose_block`). Documented divergence: the extractor is
  sourced from the current repo rather than each case's base SHA (same rationale as
  `pr_review.md` sourcing). Non-tutorial `.ipynb` ride the normal diff path, exactly
  as CI handles them.
- One `run` invocation = one experiment (run all arms together; `--k-per` covers
  per-arm repeat counts). `compare` reads the per-run manifest
  (`runs/<subdir>-manifest.json`), so rerunning into the same `--subdir` with a
  changed model **replaces** the comparison rather than mixing the old and new runs.
- Blinding is best-effort: on a MIXED-k matrix the repeat counts partition arms
  into {k=2} vs {k=1} groups (an equal-k matrix such as the 2026-10 B/E run is a
  clean blind), and the sanitizer can't catch every conceivable self-reference.
  Graders are instructed to grade on content, never on guessed identity (see
  `GRADING_RUBRIC.md`).
