# Pre-Registered Decision Rule: gpt-5.6-sol -> gpt-6-astra Reviewer Evaluation

Date: 2026-10-16. Pre-registered in PR #849 (frozen at that PR's labeled, CI-green HEAD). The
candidate arm is NEVER run on the corpus before the campaign: its first corpus
run is the campaign itself, so pre-registration holds by construction. The
verdict is computed MECHANICALLY by `run_eval.py verdict` (`reviewer_verdict.py`)
from the reconciled grading table; this file may not be edited after the
campaign starts (the campaign's frozen PR-1 HEAD pins it).

This file is NOT grader-facing. Graders receive `GRADING_RUBRIC.md` only.

## The experiment

| Arm | Model | Effort | Repeats | Role |
|-----|-------|--------|---------|------|
| B | gpt-5.6-sol | xhigh | k=2 | control (current production) |
| E | gpt-6-astra | xhigh | k=2 | candidate |

No probe arms. CLI pinned to codex-cli 0.162.1 on every arm (the runner refuses a
live `codex --version` mismatch); CI pins the same version. Every harness codex
call runs under an isolated `CODEX_HOME` that mirrors CI's codex-action home
(credential + `sandbox_mode = "read-only"` + `approval_policy = "never"`; no MCP
servers, hooks or memories; codex's DEFAULT features otherwise, web search
included).

## Definitions

By reference to `GRADING_RUBRIC.md`. Grades are exactly caught / partial /
missed (partial = missed for every gate). "Leak" is NOT a grade but a separate
per-cell field (`leak: none | confirmed` in the reconciled table; graders write
`none` or `flag`, the verifier rules); a `confirmed` cell is VOID (below).

- **Reliably caught** (per arm, per bug) = caught in EVERY non-void repeat of
  that arm AND no void repeat on that (case, arm).
- **FP total per arm** = sum of FPs over all its non-void repeats on S3 cases,
  where an FP is a finding whose severity is above the case's
  `allow_severities`, with NO topic-based exemption: on this corpus the only
  known-FP topic (s3-wcr) has `would_be_severity_if_flagged` equal to the
  allowed P3, so a P3 mention is already allowed and anything above P3 is the
  2026-07 "calibration failure" FP. The older "known-FP topics are never
  counted" wording is retired as redundant; the sibling engine's
  `known_topic_id` exemption branch is NOT ported. The bundle's "do NOT flag"
  wording for known topics stays (reviewer advice, not grader instruction).

## Gates (E vs B; symmetric on catches)

1. **Regression NO-GO** = ANY ground-truth bug (`must_catch` true OR false)
   that B reliably catches and E has status MISSED (the status definitions
   under "Voided repeats" are the executable wording). `must_catch` stays in
   the case files as a reporting label only. B UNSTABLE / E MISSED is a
   judgment FLAG reported next to the mechanical outcome; it never changes the
   label by itself.
2. **FP NO-GO** = E's S3 FP total > B's. Small-sample note (2 S3 cases x k=2,
   B expected at 0): the verdict prints every FP's text next to this gate so
   the user can rule a control invalid, as happened in 2026-07; the mechanical
   verdict stands unless the user overrides it in writing in the verdict.
3. **GO** = zero regressions, the FP gate passes, FEWER THAN 2
   reliable-to-unstable downgrades (bugs B catches 2/2 and E catches 1/2), and
   >= 1 strict improvement: E reliably catches ANY bug on which B has status
   MISSED (the two s4-wcr bugs DO count), OR E has strictly fewer S3 FPs than B
   (only reachable if B posts >= 1 FP).
4. **PARITY** = zero regressions and the FP gate passes, and EITHER no strict
   improvement OR >= 2 reliable-to-unstable downgrades -> the user decides,
   with these inputs reported: judgment flags, the downgrade and upgrade
   lists, latency, the input-token cost estimate, hallucination and leakage
   counts, the session-log audit, and the vendor lifecycle.

### Voided repeats

A (case, arm, repeat) with a verifier-CONFIRMED leak is VOID for every gate,
both its bug rows and its FP list, whichever arm it belongs to. An arm's
per-bug status is computed from its remaining (non-void, OK) repeats:

- RELIABLE = caught in every remaining repeat AND no void repeat on that
  (case, arm);
- UNSTABLE = caught in some remaining repeat, or caught in all remaining
  repeats but at least one repeat void;
- MISSED = caught in no remaining repeat (>= 1 remaining);
- UNGRADEABLE = no remaining repeat.

Gates use these statuses (regression = B RELIABLE and E MISSED; downgrade = B
RELIABLE and E UNSTABLE; improvement = E RELIABLE and B MISSED; UNGRADEABLE
never satisfies a gate, it only contests). A void or ungradeable cell ALWAYS
contests: its case is reported "contested by leak" on whichever gates the case
feeds (gates 1/3/downgrades for a bug case, gate 2 for an S3 case). Audit hits
never contest a gate (see "Hit list"). A contested gate goes to the user, never
decided mechanically: the user records in the verdict one of "accept
mechanical", "override to GO" or "override to NO-GO", with the reason, and
PR-2 swaps exactly when that final call is GO. A void cell is NOT re-run
(resume would reuse the cached artifact and the leak is a property of that
review); it stays void.

### Precedence (the executable order)

(0) validate the grades against the manifest-scoped runs; (1) UNDETERMINED if
any compared (case, arm) has fewer OK repeats than scheduled because of
INFRA_ERROR (void cells do NOT count here, they are graded cells); (2) apply
void/ungradeable marking from `leak: confirmed`; (3) NO-GO on any regression;
(4) NO-GO on FP excess; (5) PARITY if >= 2 downgrades; (6) GO if >= 1 strict
improvement; (7) PARITY. "Contested" is an annotation added at steps (3)-(6)
for the gates a void/ungradeable cell touched; the session-log hit list is
attached to every gate as an informational annotation and never contests. A
live campaign with any INFRA_ERROR has a failed manifest that `verdict`
refuses outright, so (1) is reachable only on synthetic tables; incomplete
campaigns are handled by the stop conditions below.

## Informational, non-gating

`run_eval.py verdict` computes the first three reads from the sealed artifacts
(`campaign_reads` in its JSON output: per-arm n/mean/p95/max latency, the
2x-p95 flag, the input-cost estimate from `openai_review.PRICING`); the
vendor-lifecycle read is prose the verdict file carries.

- Per-arm latency (flag if E p95 > 2x B).
- Hallucination / unverified external-tool-claim count per arm (counted once
  per (case, arm, repeat)).
- Leakage count per arm (distinct confirmed cells) and unconfirmed-flag count.
- Rough CI input cost per arm = `prompt_tokens_est` x official input price (a
  +/-50% initial-prompt estimate; output/reasoning/tool tokens are not
  recorded).
- **Session-log audit** (`run_eval.py audit`, `reviewer_audit.py`), run AFTER
  unblinding: for every manifest run artifact it strips the run's own worktree
  path from each `usage.session_commands` entry (the shell command lines
  `call_codex` parsed from codex's stderr), then scans the remainder for the
  ONE authoritative leak marker list (source of truth `reviewer_audit.LEAK_RULES`:
  the literal `LEAK_MARKERS` plus the seven regexes, rendered verbatim in
  `GRADING_RUBRIC.md`; this file does not restate it) and for the computed
  per-campaign markers (the enclosing checkout, its branch name, the git
  common dir and its parent checkout, every other worktree, every other
  recorded ref name, the isolated codex home). It over-matches BY DESIGN
  (routine orientation such as `git log -1`, `git show HEAD`, `HEAD~` are
  markers) and is BEST-EFFORT: codex's default non-shell channels (web
  search, sub-agents, browser) leave no shell `exec` line - the 2026-10 astra
  latency probe used web search 26 times in one review, and the pre-freeze
  rehearsal showed the CONTROL (sol) web-searching too (12 events across its
  two S1 reviews, both cells `unaudited`, audit VACUOUS) - so any run that
  printed a non-shell event header is `unaudited` rather than covered. User
  decision (2026-10-16, after the rehearsal): the campaign ACCEPTS a vacuous
  or near-vacuous audit as-is rather than disabling web search (CI parity,
  decision 8) or capturing the queries; the grader `LEAK?` flag and the
  verifier passes remain the only leak check, and the verdict reports the
  `unaudited` labels and the vacuity verbatim. The rehearsal expectation is
  therefore "coverage classes computed and reported" - not `unaudited == 0`.
  - **Coverage classes** (the single definition, used here, by
    `reviewer_audit.coverage_class` and by every test). Both `covered` and
    `no_commands` REQUIRE `usage["codex_home_isolated"] == True` and
    `usage["session_other_events"] == 0`. `covered` = `session_commands` is
    non-empty, no entry ends in `[truncated]`, `session_commands_truncated`
    is exactly `False`, and `exec_lines == start_lines == len(session_commands)`.
    `no_commands` = the list is empty, `start_lines == 0`, `exec_lines == 0`
    and `session_commands_truncated` is exactly `False` (codex ran no shell
    command). `unaudited` = everything else (key absent, a truncated entry or
    stream, a missing or non-boolean truncation flag, an empty list with a
    non-zero count, `exec_lines > 0` with `start_lines == 0`, a non-zero
    `other_events`, a non-isolated home). The audit is VACUOUS if NO
    artifact is `covered` or `no_commands`; a vacuous audit is reported as
    such (the hit list is empty for a technical reason).
  - **Hit list** (user decision 6): the audit never decides whether a hit
    matters and never marks a gate contested. The verdict prints, for EACH
    gate, the cells that feed it (gates 1/3/downgrades: every (case, arm,
    repeat) of every bug case; gate 2: every S3 cell) with the cell's coverage
    class and its hits (marker names + the stripped command line). A hit is
    **orientation-only** iff all its marker names are in
    `reviewer_audit.ORIENTATION_MARKERS`; a cell is orientation-only iff all
    its hits are; the verdict's first paragraph states how many gate-feeding
    cells carry hits and how many are orientation-only, and lists the
    non-orientation markers of every remaining hit. The user reads that list
    and decides; the mechanical label stands unless overridden in writing.

## Grading protocol (blinded, multi-grader)

1. `compare --blinded` writes `comparison.blinded.md` (M* labels, model
   self-references scrubbed, latency redacted, a visible Bundle ID) and a
   sealed `blinding.json` (mapping, `bundle_id`, `run_ids_sha`,
   `artifacts_sha`, `rubric_sha256`, `prompt_sha256`) that graders NEVER read.
   `verdict` REGENERATES the blinded bundle deterministically from the sealed
   artifacts and refuses unless the file is byte-identical, so the
   grader-visible evidence is bound to the artifacts, not only to itself.
2. 3 INDEPENDENT blinded grader subagents receive ONLY the blinded bundle,
   `GRADING_RUBRIC.md` and `.github/codex/prompts/pr_review.md` (never this
   file, never the 2026-07 verdict, never raw artifacts). Isolation is by
   instruction plus two checks (the prompt lists the three permitted paths
   and forbids Bash; each returned table is grepped for paths, run ids,
   config ids and model names before acceptance), not a sandbox.
3. One verifier subagent (same three inputs) reconciles disagreements and
   rules on every `LEAK?` flag with quoted evidence (pass 1), then examines
   EVERY remaining bug cell and S3 cell against the confirmation criteria in
   one pass (pass 2, at most 48 cells). Only diff-independent evidence confirms.
4. Unblind only after the tables are final; apply the gates mechanically
   (`verdict` without `--audit-hits` = PROVISIONAL); then run `audit` and the
   same `verdict` with `--audit-hits` (= DEFINITIVE). Both outputs go in the
   verdict file.

Blinding caveat: with two k=2 arms the labels are a clean 50/50 blind.

## Corpus floor and provenance

The campaign runs only with >= 8 verified cases including >= 2 S4 cases
(`verdict` refuses below the floor). Provenance recorded in the verdict:
frozen PR-1 HEAD sha, tree sha, local tag, PR number, subdir, later main
merge sha. The manifest's `repo_head_sha`/`repo_tree_sha` must equal them,
with `run_head_shas == [<HEAD>]`, `run_head_missing == 0`,
`run_dirty_any == false`, `run_marker_mismatch == 0`, `run_not_isolated == 0`;
`verdict` recomputes the aggregates from the sealed artifacts and refuses on
any difference. Any push to PR-1 after the campaign starts VOIDS the run (a
voided campaign is re-run into a fresh subdir on the new HEAD).

**Campaign ledger (enforced by `run_eval.py`):** the first dispatch of the
registered contrast creates the local git ref
`refs/campaign/<control>-<candidate>/frozen` at the preflight HEAD (in the
repository, never pushed, outside the disposable `runs/` dir), and every
dispatch appends `{contrast, head, tree, subdir, ledger_reset}` to the
campaign-global `runs/campaign-ledger.jsonl`; the FIRST record is the frozen
candidate-exposure revision and must equal the ref. `verdict` refuses without
a ledger record, without the ref, when the ledger's first head differs from
the ref, or when the first record is marked `ledger_reset` (the ledger was
deleted after the ref existed) - all three are protocol-guard refusals, not
labels - and otherwise judges the manifest against the first record: mechanical iff this subdir IS
the first dispatch, or the HEAD moved (a voided campaign re-run into a fresh
subdir) and the registered surfaces (`tools/reviewer-eval/`,
`tools/eval_core/`, `.claude/scripts/openai_review.py`,
`.github/codex/prompts/pr_review.md`, `tools/notebook_md_extract.py`) are
untouched between the two commits. **Pre-registered exception:** a diff that
is exactly the single `CALL_TIMEOUT_S = <n>` line in
`adapters/codex_reviewer.py` with the value strictly INCREASED (a lower cap
changes censoring and is not exempt) keeps the campaign pre-registered,
because a timeout cannot change what a completed review says. A second subdir
on the SAME head (re-exposure), the FIRST subdir reused after a HEAD change
(a voided campaign re-runs into a fresh subdir), or any other
registered-surface change makes the verdict "RE-REGISTERED after candidate
exposure": still computed and
reported, labelled non-mechanical with the surface diff stat, final call the
user's, never a mechanical GO. The ledger lists every subdir the contrast
ran in.

Known, accepted limit: provenance is sampled at the start of each review and
at manifest write; a tracked input edited after one sample and restored
before the next is invisible to the fields (the tree-identity guard pins the
tree only at the post-run stages).

## Stop conditions (enforced by `run_eval.py run`)

Both integrity logs (`runs/<subdir>-attempts.jsonl`, `runs/campaign-ledger.jsonl`)
FAIL CLOSED: a malformed line or a record missing a required field refuses
the stage with the file and line number, never a silent skip.

At most 3 reruns after the first attempt (resume reuses OK artifacts). Every
`run` attempt appends a START record before dispatch and an END record (the
per-attempt failure record: case id, arm, repeat, first 120 chars of
`infra_error`) to the append-only `runs/<subdir>-attempts.jsonl` BEFORE
anything can overwrite the artifacts; an attempt that raised before dispatch
(`CLIVersionMismatch`, `ConfoundMismatch`) has no END record and does not
count. An attempt interrupted or failed AFTER dispatch (an interrupt, an
unexpected exception) gets an END record marked `aborted` and counts toward
the cap; a fully cached resume (no codex call) writes no END record and does
not count. Every END record lists the OK cells of that attempt with their
artifact hashes - an aborted attempt's record is rebuilt from the store for
every planned cell (cached and freshly persisted alike; each review is
persisted by the worker that produced it before the coordinator sees it, so
an interruption of the coordinator still protects every finished review); `run`
refuses a new attempt when any artifact listed OK by ANY earlier END record
(the cumulative set) is missing or changed (a rerun RESUMES, it never
re-samples a successful cell). One attempt per subdir at a time: `run` holds
an exclusive cross-process lock (`runs/<subdir>.lock`) from the stop-condition
check through the manifest write and refuses a concurrent `run` into the same
subdir, so no artifact set can be interleaved from two invocations and sealed
by the last one. `run` REFUSES a fifth dispatched attempt into the same subdir and
refuses to start when a (case, arm, repeat) timed out in the last 2
consecutive attempts; both checks run BEFORE anything on disk is touched, so
a refused attempt leaves the previous successful manifest and its sealed
artifacts intact; both are surfaced to the user (never silently excluded,
never graded partial). A provenance PREFLIGHT also runs before the manifest
is touched and before any codex call: a dirty working tree or an incomplete
git sample (a `None` ref or worktree list, an empty sha/branch/common dir)
refuses the attempt, so a campaign the post-run gate would reject never
spends the candidate's first corpus exposure. `run` also fails closed (failure marker,
exit 2) when its write-time provenance sample is incomplete (a `None` ref or
worktree list, an empty sha/branch/common dir), and `verdict` refuses such a
manifest at its protocol guard. The operator's run-log headers duplicate
this record for the verdict file.

## Candidate exposure (enforced by `run_eval.py`)

A registered candidate arm (`reviewer_verdict.REGISTERED_CONTRASTS`) can be
selected ONLY as the exact registered campaign: `run --configs <control>,<candidate>
--k 2` with no `--k-per` and no `--strata`; `smoke` refuses it outright. So
"the candidate's first corpus run is the campaign" holds by construction in
code, not only by protocol. An EMPTY `session_commands`
on the 0.162.1 control smoke was a STOP condition before the PR-1 commit (it
did not occur: the smoke parsed 8/8 exec blocks).
