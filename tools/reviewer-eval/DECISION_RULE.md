# Pre-Registered Decision Rule: gpt-6.1-sol -> gpt-6.2-sol Reviewer Evaluation (campaign 3)

Date: 2026-11-26. Pre-registered on a frozen LOCAL commit (anchors: the local tag
`campaign-gpt62sol` and the ledger ref `refs/campaign/F-G/frozen`; see "Corpus
floor and provenance"). The candidate arm is NEVER run on the corpus before the
campaign: its first corpus run is the campaign itself, so pre-registration holds
by construction. The verdict is computed MECHANICALLY by `run_eval.py verdict`
(`reviewer_verdict.py`) from the reconciled grading table; this file may not be
edited after the campaign starts (the frozen commit pins it).

**Relation to campaign 2 (gpt-6.1-sol, `campaigns/2026-10-gpt61sol-DECISION_RULE.md`):**
campaign 2 is complete (44/44 runs on `e1674c0e`, local tag `campaign-gpt61sol`,
mechanical DEFINITIVE PARITY, accepted as GO by the user on 2026-10-17, so
gpt-6.1-sol @ xhigh is production and arm F is the control). This rule is the
SAME rule with arm F = gpt-6.1-sol in the control seat and arm G = gpt-6.2-sol
in the candidate seat: the gates, definitions, precedence, grading protocol,
audit, ledger and stop conditions below are unchanged in substance. Three
things differ, all stated here before any candidate exposure:

1. **CLI.** gpt-6.2-sol is rejected on codex-cli 0.162.1
   (`model_not_in_accessible_catalog`, ChatGPT-account login) and accepted on
   0.172.0, so every arm is re-pinned to codex-cli 0.172.0. F is judged as a
   FRESH control sample on the new CLI (F is re-run in the new subdir; the
   campaign-2 F artifacts no longer exist and its grades are never mixed in).
2. **Freeze anchor.** The campaign runs on a local commit and ONE PR follows
   it (user direction of 2026-10-17, as campaign 2 actually ran); the PR-based
   wording of the earlier rules is replaced below.
3. **One added descriptive read** (non-gating, see "Informational"): the fresh
   F per-bug status (0.172.0) next to the committed campaign-2 F status
   (0.162.1), the CLI-bump read. No gate compares them.

`REGISTERED_CONTRASTS` keeps `(B, E)` and `(B, F)` so the earlier ledger
records stay interpretable; `(F, G)` is the contrast this campaign registers.

This file is NOT grader-facing. Graders receive `GRADING_RUBRIC.md` only.

## The experiment

| Arm | Model | Effort | Repeats | Role |
|-----|-------|--------|---------|------|
| F | gpt-6.1-sol | xhigh | k=2 | control (current production) |
| G | gpt-6.2-sol | xhigh | k=2 | candidate |

No probe arms. CLI pinned to codex-cli 0.172.0 on every arm (the runner refuses a
live `codex --version` mismatch); CI pins the same version. Every harness codex
call runs under an isolated `CODEX_HOME` that mirrors CI's codex-action home
(credential + `sandbox_mode = "read-only"` + `approval_policy = "never"` +
`[features] apps = false`; no MCP servers, hooks or memories; codex's DEFAULT
features otherwise, web search included). The default feature surface on 0.172.0 differs from 0.162.1
(`codex features list` under the isolated home on both versions, 2026-11-26,
gitignored `runs/features-<version>.txt`): 18 flags are newly on
(`api_key_cyber_access_programs`, `bedrock_setup_wizard`, `cli_daybreak`,
`code_mode_host_output_truncation`, `code_mode_interrupt`,
`deferred_tool_world_state`, `guardian_include_additional_context`,
`incremental_tools`, `incremental_tools_legacy`, `incremental_tools_legacy_gcp`,
`mcp_lazy_stdio_startup`, `mcp_oauth_refresh_coordination`,
`prediction_preserve_parent_tools`, `reasoning_effort_override`,
`retain_client_tool_outputs`, `tui_workspace`, `unified_image_budget`,
`untrusted_as_tool_output`) and one is newly off (`unified_exec_zsh_fork`).
Both arms run the same surface, and a channel that leaves no shell `exec`
line is classified `unaudited` exactly as before.

**One feature is switched off for every arm: `apps`** (user decision,
2026-11-26, before any candidate corpus exposure). The harness logs in with
the ChatGPT-account credential, which carries the account's app connectors
(`codex_apps/...`); CI logs in with an API key and is taken to have none (an
inference from the auth mode, not verified under an API key). In the
pre-registration off-corpus probes (merged PR #841's diff, xhigh, isolated
home, six gpt-6.2-sol reviews) the candidate called the GitHub connector in
two reviews (24 `github.fetch_file` calls in one, one
`github.fetch_pr_comments` in another); the one gpt-6.1-sol probe did not.
The connector reads the PUBLISHED repository rather than the case worktree,
so on historical cases it can return code that already contains the fix.
With `apps = false` one probe showed no `codex_apps` event. Web search stays
on, as in the earlier campaigns.

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

## Gates (G vs F; symmetric on catches)

1. **Regression NO-GO** = ANY ground-truth bug (`must_catch` true OR false)
   that F reliably catches and G has status MISSED (the status definitions
   under "Voided repeats" are the executable wording). `must_catch` stays in
   the case files as a reporting label only. F UNSTABLE / G MISSED is a
   judgment FLAG reported next to the mechanical outcome; it never changes the
   label by itself.
2. **FP NO-GO** = G's S3 FP total > F's. Small-sample note (2 S3 cases x k=2,
   F expected at 0): the verdict prints every FP's text next to this gate so
   the user can rule a control invalid, as happened in 2026-07; the mechanical
   verdict stands unless the user overrides it in writing in the verdict.
3. **GO** = zero regressions, the FP gate passes, FEWER THAN 2
   reliable-to-unstable downgrades (bugs F catches 2/2 and G catches 1/2), and
   >= 1 strict improvement: G reliably catches ANY bug on which F has status
   MISSED (the two s4-wcr bugs DO count), OR G has strictly fewer S3 FPs than F
   (only reachable if F posts >= 1 FP).
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

Gates use these statuses (regression = F RELIABLE and G MISSED; downgrade = F
RELIABLE and G UNSTABLE; improvement = G RELIABLE and F MISSED; UNGRADEABLE
never satisfies a gate, it only contests). A void or ungradeable cell ALWAYS
contests: its case is reported "contested by leak" on whichever gates the case
feeds (gates 1/3/downgrades for a bug case, gate 2 for an S3 case). Audit hits
never contest a gate (see "Hit list"). A contested gate goes to the user, never
decided mechanically: the user records in the verdict one of "accept
mechanical", "override to GO" or "override to NO-GO", with the reason, and
the single follow-up PR swaps exactly when that final call is GO. A void cell is NOT re-run
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

- Per-arm latency (flag if G p95 > 2x F).
- **CLI-bump read** (campaign 3 only, descriptive): the fresh F per-bug status
  on codex-cli 0.172.0 next to the committed campaign-2 F status on 0.162.1
  (`campaigns/2026-10-gpt61sol-VERDICT.md`). Same corpus, prompt and rubric
  bytes, same model id, different CLI and an independent sample, so a
  difference cannot be attributed to the CLI alone. No gate uses it; it
  informs the user's separate decision on the CI `codex-version` pin.
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
(`verdict` refuses below the floor). The campaign runs on a frozen LOCAL
commit. Provenance recorded in the verdict: that commit's sha and tree sha,
the local tag `campaign-gpt62sol` (never pushed; it keeps the sha resolvable
after the PR is squashed), the subdir, and later the PR number and main merge
sha. The manifest's `repo_head_sha`/`repo_tree_sha` must equal them,
with `run_head_shas == [<HEAD>]`, `run_head_missing == 0`,
`run_dirty_any == false`, `run_marker_mismatch == 0`, `run_not_isolated == 0`;
`verdict` recomputes the aggregates from the sealed artifacts and refuses on
any difference. Moving HEAD before the verdict is final VOIDS the run (the
post-run stages refuse a HEAD or tree that differs from the manifest's; a
voided campaign is re-run into a fresh subdir on the new HEAD and judged by
the ledger rule below). Once the DEFINITIVE verdict is recorded, the later
commits that add the verdict files and any swap, and the rebase/squash into
the single PR, do NOT void it: they are after the measurement, and the tag
and `refs/campaign/F-G/frozen` keep the frozen revision auditable. This
paragraph is procedural, not code-enforced, beyond the guards named in it.

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
on the 0.172.0 control smoke was a STOP condition before the freeze (it did
not occur: the smoke of arm F on 2026-11-26 parsed 32/32 exec blocks, with 6
non-shell events, under the isolated home with `apps = false`).
