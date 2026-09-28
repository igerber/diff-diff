---
name: plan-review
description: Review a Claude Code plan file with the campaign-selected dual engine (Opus reviewer + codex-sol, merge/verify), or revise a plan from its review. Produces the .review.md the ExitPlanMode content-hash gate requires.
---

# Plan Review (dual engine)

Owns the plan-review → revise cycle. The review ENGINE is **dual** — two blind
reviewers (Claude @ Opus 4.8 + codex `gpt-5.6-sol`) then a merge/verify pass —
the configuration Campaign 1 selected: its exploratory run had dual reliably
catch **7/9** must-catch plan defects vs **1/9** for any single reviewer. That
campaign was **NON-GATING** (contaminated negatives + a criteria regression);
the sensitivity signal is what selects dual, and a clean re-validation is
tracked — see `tools/plan-review-eval/verdicts/campaign-1.md`. The content-hash
approval gate (`.claude/hooks/check-plan-review.py`) is engine-agnostic and
unchanged; the snapshot helper's (`.claude/scripts/plan_snapshot.py`)
hash-certification contract is unchanged, though its lifecycle is extended here
(it emits a per-invocation work dir, creates transactionally, self-cleans on
persist failure, and its `abort` is strict).

Bundled in this skill dir (`.claude/skills/plan-review/`):
- `criteria.md`, `reviewer_prompt.md` — the detection prompts, **byte-identical**
  to what the campaign graded (never edit without re-validation).
- `merge_verify.md` — the merge/verify prompt, production-adapted to NAME the two
  reviewers (un-blinded from the graded copy, which blinded them only for
  grading); the verify-every-finding logic is unchanged.
- `render.py` — strict `__TOKEN__` renderer. **Always render prompts with it;
  never free-text-substitute the templates** (that is a different, unmeasured
  engine).
- `codex_review.py` — the codex (reviewer 2) half, with the campaign
  model/effort pins and loud-fallback exit codes.

## Modes

- **Review** a plan before approval (the CLAUDE.md "Plan Review Before
  Approval" gate invokes this).
- **Revise** a plan from its existing review.

The retired `/review-plan` carried `--updated` (delta re-review) and `--pr`
(feedback-coverage) flags. Neither is reimplemented in this initial skill —
they are a tracked follow-up (TODO.md). Re-reviewing a changed plan is the
**Revise phase**'s job (it re-runs Review over the new bytes); PR-feedback
coverage is dimension 9 of `criteria.md`, applied when the reviser hands the
plan a PR comment to fold in. Do NOT advertise or emulate the old flags here.

The Review phase runs in one of two ENGINE modes, chosen by the caller (the
gate's adaptive **Dual / Single / Skip** offer): **dual** (default — reviewer 1
+ codex + merge/verify) or **single** (reviewer 1 alone, a DELIBERATE
one-reviewer choice — distinct from the codex-unavailable fallback). Skip is
handled by the gate (a Skipped marker), not this skill.

Two path conventions below, because **shell variables do NOT persist across
separate Bash tool calls and the Write tool does NOT expand them**:
- The skill dir is the literal `.claude/skills/plan-review/`, used verbatim (not
  via a `$SKILL` variable).
- The per-worktree scratch dir is `git rev-parse --git-path plan-review`. Every
  Bash block that needs it **re-derives it inline** (`"$(git rev-parse
  --git-path plan-review)/…"`); a path handed to the **Write tool** uses the
  LITERAL value printed in step 1, never a `$SCRATCH` token.

---

## Review phase

### 1. Snapshot the plan (helper fixes the reviewed bytes)

First derive, create, and PRINT the scratch dir — the printed value is the
literal path you give the Write tool (which does not expand shell variables):

```bash
SCRATCH="$(git rev-parse --git-path plan-review)"; mkdir -p "$SCRATCH"; echo "$SCRATCH"
```

The raw plan path is untrusted and must never touch a shell: with the **Write
tool** (never echo/heredoc), write it to `<scratch>/plan-path.txt` — substitute
the LITERAL path just printed for `<scratch>` — then snapshot (this Bash block
re-derives the scratch path inline; the variable above does not carry over):

```bash
python3 .claude/scripts/plan_snapshot.py snapshot \
  --plan-path-file "$(git rev-parse --git-path plan-review)/plan-path.txt"
```

It prints `state_path`, `snapshot_path`, `body_path`, `meta_path`, `work_dir`,
`plan_path`, `plan_sha256`, `review_path`. **Confirm the printed `plan_path` is
the plan you supplied** (the ingress file is shared per-worktree; a concurrent
session may have overwritten it). If it differs, the snapshot you just took is of
the wrong plan — release it before retrying with
`python3 .claude/scripts/plan_snapshot.py abort --state-file "<state_path>"`,
then re-Write `plan-path.txt` and re-run (an un-aborted first snapshot is
orphaned). Non-zero exit → report and stop. Review the SNAPSHOT
(`snapshot_path`), never the live plan.

`work_dir` is a private per-invocation directory the helper emits (under its own
snapshots dir, safe-charset leaf, removed together with the snapshot) for the
prompt/review files this skill writes — every one below lives inside it. Two
concurrent reviews get distinct `work_dir`s, so their prompts and reviewer
outputs cannot cross-wire (one plan's merged report can never be persisted under
another's certified hash). Because it is **helper-derived, not built from the
repo/worktree path**, no `$()`/backtick in a checkout path can execute when the
path is later substituted into a command.

> **Release the snapshot exactly once.** Everything below holds an open snapshot
> (its `work_dir` included). `plan_snapshot.py persist` (step 7) releases the
> snapshot AND the `work_dir` on the success path. On ANY failure or early stop
> BEFORE persist — a render error, a reviewer/merge failure, a bad write — run
> `python3 .claude/scripts/plan_snapshot.py abort --state-file "<state_path>"`
> (plain `abort`: the state still exists, so a wrong/mistyped token fails loudly
> rather than silently no-op'ing). Abort removes the snapshot and the `work_dir`
> — there is no separate cleanup to run. A failure of the persist call itself
> needs NO abort: `persist` self-cleans the whole invocation on any failure
> (step 7).

### 2. Render the reviewer prompt (tested Python, never free-text)

```bash
python3 ".claude/skills/plan-review/render.py" ".claude/skills/plan-review/reviewer_prompt.md" \
  --token criteria=".claude/skills/plan-review/criteria.md" \
  --token plan="<snapshot_path>" \
  -o "<work_dir>/reviewer_prompt.txt"
```

### 3. Reviewer 1 — Claude @ Opus (blind)

**Dual mode: run reviewer 1 and reviewer 2 (step 4) concurrently.** They are
blind to each other and share no state, so issue the reviewer-1 Task-subagent
spawn (below) and the reviewer-2 `codex_review.py` call (step 4) in the SAME
batch and let both run in parallel; the merge (step 5) is the join point that
consumes both. (Single mode runs only this step.)

Spawn a Task subagent — `subagent_type: "general-purpose"`, **`model: "opus"`**
(the Task tool takes family aliases, not exact IDs, so this is a **runtime
alias** — it resolves to the current Opus, expected the campaign's 4.8 — NOT an
immutable pin; it does still keep the reviewer off the ambient session model),
read-only intent — whose prompt is the exact contents of
`<work_dir>/reviewer_prompt.txt`. It reviews the plan against the CURRENT repo
and returns the findings list + summary table. Write its output to
`<work_dir>/review_a.md`.

### 4. Reviewer 2 — codex-sol (blind) — DUAL mode only

**Single mode**: skip steps 4-5 and take the **Single-reviewer mode
(deliberate)** path below — the review body is reviewer 1's output with the
deliberate-single note.

> **codex read surface (documented, accepted).** codex runs `--sandbox
> read-only`, which blocks writes but does NOT confine READS to the repo
> (`codex_review.py` prints the sensitive-file notice before invoking). A
> prompt-injected plan could in principle steer codex to read outside the
> worktree — the SAME surface `/ai-review-local --backend codex` already
> documents and accepts. Real OS-level confinement is tracked in `TODO.md`
> (Codex reviewer isolation). Plans here are authored in the user's own
> session; treat this as the accepted, pre-existing codex surface, not a
> plan-review-specific gate.

```bash
python3 ".claude/skills/plan-review/codex_review.py" \
  --prompt-file "<work_dir>/reviewer_prompt.txt" \
  --repo-root "$(pwd)" \
  -o "<work_dir>/review_b.md"
```

- **Exit 0** → codex review is in `<work_dir>/review_b.md`; go to step 5.
- **Non-zero (2 = codex absent, 3 = timeout/error)** → codex is unavailable;
  take the **Loud fallback** below and skip steps 5-6. A hung codex cannot
  wedge the gate — `codex_review.py` caps at 2400s and exits 3.

### 5. Merge + verify — Claude @ Opus

```bash
python3 ".claude/skills/plan-review/render.py" ".claude/skills/plan-review/merge_verify.md" \
  --token criteria=".claude/skills/plan-review/criteria.md" \
  --token plan="<snapshot_path>" \
  --token review_a="<work_dir>/review_a.md" \
  --token review_b="<work_dir>/review_b.md" \
  -o "<work_dir>/merge_prompt.txt"
```

Spawn a Task subagent — `subagent_type: "general-purpose"`, **`model: "opus"`**,
read-only — with the contents of `<work_dir>/merge_prompt.txt`. It matches
findings across the two reviews (`[consensus]`/`[single reviewer]` tags),
**re-verifies every finding against the repo** (nothing trusted blindly), and
emits the merged report (findings, `## Rejected on verification`,
`## Disagreements`, summary table). **The review body is the provenance marker
`<!-- plan-review-engine: dual -->` as its first line, then this merged
report.** (The marker lets the Revise phase read the engine mode
deterministically instead of inferring it from the body prose.)

### 6. (dual path complete)

### Loud fallback (codex unavailable/timeout)

Skip the merge. The review body = the provenance marker
`<!-- plan-review-engine: single-fallback -->` (first line), then this warning
verbatim, then the contents of `<work_dir>/review_a.md`:

```
> ⚠ **codex unavailable — SINGLE-Claude fallback.** This is the un-validated
> weak arm (Campaign 1: caught 1/9 hard plan defects vs 7/9 for the dual
> engine). Run `codex login` and re-review for full coverage.
```

The review still persists and gates normally.

### Single-reviewer mode (deliberate)

The user deliberately chose one reviewer (the gate's **Single** option). Run
step 3 only. The review body = the provenance marker
`<!-- plan-review-engine: single -->` (first line), then this note verbatim
(distinct from the codex-unavailable warning — this was a choice, not a
failure), then the contents of `<work_dir>/review_a.md`:

```
> Single-Opus review (deliberate one-reviewer choice, not the dual engine).
> The dual engine caught 7/9 hard plan defects vs 1/9 single in Campaign 1 —
> re-review in dual mode if this plan turns out higher-risk than expected.
```

The review persists and gates normally.

### 7. Persist (helper stamps the gate contract)

Write the review body — the merged report (dual), or reviewer 1's output with
its prepended note (single / codex-fallback), each led by its
`<!-- plan-review-engine: … -->` marker — to the printed `body_path` with the
Write tool. **No YAML frontmatter**: the helper stamps `plan:`/`plan_sha256:`
itself.

Derive `assessment` from the ACTUAL verified findings, not the reviewer's
self-reported table: **count the finding lines by their severity tag**
(`[P0]`/`[P1]`/`[P2]`/`[P3]`) and cross-check that count against the summary
table. If they disagree, or there are no findings AND no parseable table, the
report is malformed — do NOT persist a guessed assessment: `abort` (plain — the
state still exists; it also removes the `work_dir`) and report the malformed
output. Otherwise derive the label from the line counts **by this deterministic
rule** (so identical findings always persist the same label):

- any **P0 or P1** → `Significant issues found`
- else any **P2** → `Minor revisions recommended`
- else (only P3, or no findings) → `Ready to implement`

Then write the meta JSON to `meta_path`:

```json
{"reviewed_at": "<ISO-8601 UTC>", "assessment": "<Ready to implement | Minor revisions recommended | Significant issues found>", "critical_count": <P0>, "medium_count": <P1+P2>, "low_count": <P3>, "flags": []}
```

Then `python3 .claude/scripts/plan_snapshot.py persist --state-file "<state_path>"`
(exit 0 = stamped + cleaned up, `work_dir` included; exit 3 = plan changed
mid-review → re-review; other non-zero → report). **Do NOT abort after persist**
— `persist` self-cleans the entire invocation (snapshot + `work_dir` + sidecars)
on ANY failure, so post-persist there is nothing left to release, and a
follow-up abort could only ever hit a wrong/stale token. There is no separate
`rm -rf` at any point — persist and abort both remove the `work_dir`. Display the
review in the conversation.

### 8. Surface the verified findings (advisory — do NOT edit the plan here)

The review is now persisted and stamps the CURRENT plan bytes, so the approval
gate is satisfied for the plan as-is. **Editing the plan now would invalidate
that stamp and re-deny `ExitPlanMode`** (the hook compares `plan_sha256` to the
live bytes). So this step only PRESENTS the merged review to the user and
changes nothing on disk.

Acting on the findings is the **Revise phase** (below): it applies edits and
re-reviews, which re-stamps the hash. Carry the triage directive there — when
the user opts to revise, each verified finding is either a **mechanical fix
applied directly** (a stale path, a missing test, a signature the plan can just
adopt) or a **genuine trade-off surfaced with options + a recommendation**,
never a judgment call silently written into the plan.

---

## Revise phase

Consumes the review artifact and applies revisions, then re-reviews. The
re-review re-runs the **Review phase in the SAME engine mode the existing review
used** — read from the review's `<!-- plan-review-engine: … -->` marker
(step 2). Never silently upgrade a deliberate-`single` review to dual, and never
fall back to a retired single-reviewer path.

1. **Locate the review** via the helper (review filenames carry a canonical
   path digest — never derive from the basename). Derive + print the scratch dir
   (`SCRATCH="$(git rev-parse --git-path plan-review)"; mkdir -p "$SCRATCH"; echo
   "$SCRATCH"`), then with the **Write tool** write the plan path to the printed
   `<scratch>/plan-path.txt` (a literal, not a `$SCRATCH` token — Write does not
   expand variables), then run (re-deriving the path inline)
   `python3 .claude/scripts/plan_snapshot.py check --plan-path-file "$(git rev-parse --git-path plan-review)/plan-path.txt"`
   → `plan_path`, `review_path`, `review_exists`, `fresh`. Confirm the printed
   `plan_path` matches (shared ingress).
   - `review_exists` false → run the **Review phase** first (nothing to revise
     from).
   - `review_exists` true but `fresh` false → the plan bytes already changed
     since the review was stamped, so the review describes stale content. Do
     NOT revise from it: re-run the **Review phase** over the current plan
     first, then revise from that fresh review.
2. **Display** the plan + review in the conversation. **Parse** issues by
   severity (CRITICAL/MEDIUM/LOW ↔ P0 / P1+P2 / P3) and checklist gaps, and read
   the review's **engine mode** from its provenance marker
   `<!-- plan-review-engine: … -->`: `single` ⇒ deliberate single;
   `single-fallback` or `dual` ⇒ dual (a fallback re-review retries codex, which
   is desired). Older reviews without the marker fall back to prose (a
   deliberate-single note ⇒ single, otherwise dual). Use this for the re-review
   in step 4.
3. **Apply revisions** to the plan file (triage rule from step 8).
4. **Re-review**: the plan bytes changed, so the stamped review is now stale —
   re-run the **Review phase** over the revised plan **in the engine mode noted
   in step 2** (a fresh review writes the new `plan_sha256`). If the user
   declines the re-review, write an honest **Skipped** marker instead: snapshot,
   then persist with meta
   `{"reviewed_at": "<ISO 8601>", "assessment": "Skipped", "critical_count": 0,
   "medium_count": 0, "low_count": 0, "flags": []}` and body `Review skipped by
   user.` — never re-stamp the old review's hash onto unexamined content. Apply
   the same snapshot lifecycle as the Review phase: on a failure BEFORE persist
   (a bad meta/body Write) run plain `plan_snapshot.py abort --state-file
   "<state_path>"`; do NOT abort after persist — it self-cleans its own failures.

---

## Rollback

To remove this skill and restore the pre-Step-3 workflow: delete
`.claude/skills/plan-review/`, restore `.claude/commands/review-plan.md` +
`revise-plan.md` from git history, and revert the CLAUDE.md "Plan Review Before
Approval" section to spawn the single review agent. The hook + `plan_snapshot.py`
+ the `.review.md` contract are unchanged and need no rollback.
