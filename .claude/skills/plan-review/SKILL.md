---
name: plan-review
description: Review a Claude Code plan file with the validated dual engine (Opus reviewer + codex-sol, merge/verify), or revise a plan from its review. Produces the .review.md the ExitPlanMode content-hash gate requires.
---

# Plan Review (dual engine)

Owns the plan-review → revise cycle. The review ENGINE is **dual** — two blind
reviewers (Claude @ Opus 4.8 + codex `gpt-5.6-sol`) then a merge/verify pass —
the configuration Campaign 1 validated (reliably caught **7/9** must-catch plan
defects vs **1/9** for any single reviewer; see
`tools/plan-review-eval/verdicts/campaign-1.md`). The content-hash approval
gate (`.claude/hooks/check-plan-review.py`) and snapshot helper
(`.claude/scripts/plan_snapshot.py`) are engine-agnostic and used UNCHANGED.

Bundled in this skill dir (`.claude/skills/plan-review/`):
- `criteria.md`, `reviewer_prompt.md`, `merge_verify.md` — the exact prompts
  the campaign graded (as-validated; never edit without re-validation).
- `render.py` — strict `__TOKEN__` renderer. **Always render prompts with it;
  never free-text-substitute the templates** (that is a different, un-validated
  engine).
- `codex_review.py` — the codex (reviewer 2) half, with the validated pins and
  loud-fallback exit codes.

## Modes

- **Review** a plan before approval (the CLAUDE.md "Plan Review Before
  Approval" gate invokes this): default; `--updated` = fresh re-review of a
  changed plan (add a `## Delta Assessment` section); `--pr <url>` = also assess
  PR-comment feedback coverage.
- **Revise** a plan from its existing review.

The Review phase runs in one of two ENGINE modes, chosen by the caller (the
gate's adaptive **Dual / Single / Skip** offer): **dual** (default — reviewer 1
+ codex + merge/verify) or **single** (reviewer 1 alone, a DELIBERATE
one-reviewer choice — distinct from the codex-unavailable fallback). Skip is
handled by the gate (a Skipped marker), not this skill.

All bash below assumes `SCRATCH="$(git rev-parse --git-path plan-review)"` with
`mkdir -p "$SCRATCH"` already run (temp files); `SKILL=.claude/skills/plan-review`.

---

## Review phase

### 1. Snapshot the plan (UNCHANGED helper — fixes the reviewed bytes)

The raw plan path is untrusted and must never touch a shell: Write it to
`$SCRATCH/plan-path.txt` with the **Write tool** (never echo/heredoc), then:

```bash
python3 .claude/scripts/plan_snapshot.py snapshot --plan-path-file "$SCRATCH/plan-path.txt"
```

It prints `state_path`, `snapshot_path`, `body_path`, `meta_path`, `plan_path`,
`plan_sha256`, `review_path`. **Confirm the printed `plan_path` is the plan you
supplied** (the ingress file is shared per-worktree; a concurrent session may
have overwritten it — if it differs, re-Write and re-run). Non-zero exit →
report and stop. Review the SNAPSHOT (`snapshot_path`), never the live plan.

### 2. Render the reviewer prompt (tested Python, never free-text)

```bash
python3 "$SKILL/render.py" "$SKILL/reviewer_prompt.md" \
  --token criteria="$SKILL/criteria.md" \
  --token plan="<snapshot_path>" \
  -o "$SCRATCH/reviewer_prompt.txt"
```

### 3. Reviewer 1 — Claude @ Opus (blind)

Spawn a Task subagent — `subagent_type: "general-purpose"`, **`model: "opus"`**
(the campaign graded Opus 4.8; pin it, don't inherit the ambient session
model), read-only intent — whose prompt is the exact contents of
`$SCRATCH/reviewer_prompt.txt`. It reviews the plan against the CURRENT repo and
returns the findings list + summary table. Write its output to
`$SCRATCH/review_a.md`.

### 4. Reviewer 2 — codex-sol (blind) — DUAL mode only

**Single mode**: skip steps 4-5 and take the **Single-reviewer mode
(deliberate)** path below — the review body is reviewer 1's output with the
deliberate-single note.

```bash
python3 "$SKILL/codex_review.py" \
  --prompt-file "$SCRATCH/reviewer_prompt.txt" \
  --repo-root "$(pwd)" \
  -o "$SCRATCH/review_b.md"
```

- **Exit 0** → codex review is in `$SCRATCH/review_b.md`; go to step 5.
- **Non-zero (2 = codex absent, 3 = timeout/error)** → codex is unavailable;
  take the **Loud fallback** below and skip steps 5-6. A hung codex cannot
  wedge the gate — `codex_review.py` caps at 600s and exits 3.

### 5. Merge + verify — Claude @ Opus

```bash
python3 "$SKILL/render.py" "$SKILL/merge_verify.md" \
  --token criteria="$SKILL/criteria.md" \
  --token plan="<snapshot_path>" \
  --token review_a="$SCRATCH/review_a.md" \
  --token review_b="$SCRATCH/review_b.md" \
  -o "$SCRATCH/merge_prompt.txt"
```

Spawn a Task subagent — `subagent_type: "general-purpose"`, **`model: "opus"`**,
read-only — with the contents of `$SCRATCH/merge_prompt.txt`. It matches
findings across the two reviews (`[consensus]`/`[single reviewer]` tags),
**re-verifies every finding against the repo** (nothing trusted blindly), and
emits the merged report (findings, `## Rejected on verification`,
`## Disagreements`, summary table). **This merged report is the review body.**

### 6. (dual path complete)

### Loud fallback (codex unavailable/timeout)

Skip the merge. The review body = the contents of `$SCRATCH/review_a.md` with
this warning prepended verbatim:

```
> ⚠ **codex unavailable — SINGLE-Claude fallback.** This is the un-validated
> weak arm (Campaign 1: caught 1/9 hard plan defects vs 7/9 for the dual
> engine). Run `codex login` and re-review for full coverage.
```

The review still persists and gates normally.

### Single-reviewer mode (deliberate)

The user deliberately chose one reviewer (the gate's **Single** option). Run
step 3 only. The review body = the contents of `$SCRATCH/review_a.md` with this
note prepended verbatim (distinct from the codex-unavailable warning — this was
a choice, not a failure):

```
> Single-Opus review (deliberate one-reviewer choice, not the dual engine).
> The dual engine caught 7/9 hard plan defects vs 1/9 single in Campaign 1 —
> re-review in dual mode if this plan turns out higher-risk than expected.
```

The review persists and gates normally.

### 7. Persist (UNCHANGED helper — stamps the gate contract)

Write the review body — the merged report (dual), or reviewer 1's output with
its prepended note (single / codex-fallback) — to the printed `body_path` with
the Write tool. **No YAML frontmatter**: the helper stamps `plan:`/`plan_sha256:`
itself. Set the meta `assessment`/counts from the report's summary table. Write
the meta JSON to `meta_path`:

```json
{"reviewed_at": "<ISO-8601 UTC>", "assessment": "<Ready to implement | Minor revisions recommended | Significant issues found>", "critical_count": <P0>, "medium_count": <P1+P2>, "low_count": <P3>, "flags": []}
```

Then `python3 .claude/scripts/plan_snapshot.py persist --state-file "<state_path>"`
(exit 0 = stamped + cleaned up; exit 3 = plan changed mid-review → run `abort
--state-file <state_path>` and re-review; other non-zero → report). Display the
review in the conversation.

### 8. Triage the verified findings

Per the standing directive: **apply mechanical fixes directly** (a stale path,
a missing test, a wrong signature the plan can just adopt); **surface genuine
trade-offs to the user** with options + a recommendation — never silently
incorporate a judgment call into the plan.

---

## Revise phase

Consumes the review artifact and applies revisions; the re-review it triggers
runs the **dual engine above**, not any retired single-reviewer path.

1. **Locate the review** via the helper (review filenames carry a canonical
   path digest — never derive from the basename): Write the plan path to
   `$SCRATCH/plan-path.txt`, then run
   `python3 .claude/scripts/plan_snapshot.py check --plan-path-file "$SCRATCH/plan-path.txt"`
   → `plan_path`, `review_path`, `review_exists`, `fresh`. Confirm the printed
   `plan_path` matches (shared ingress). If `review_exists` is false, run the
   **Review phase** first.
2. **Display** the plan + review in the conversation. **Parse** issues by
   severity (CRITICAL/MEDIUM/LOW ↔ P0 / P1+P2 / P3) and checklist gaps.
3. **Apply revisions** to the plan file (triage rule from step 8).
4. **Re-review**: the plan bytes changed, so the stamped review is now stale —
   re-run the **Review phase** over the revised plan (a fresh dual review
   writes the new `plan_sha256`). If the user declines the re-review, write an
   honest **Skipped** marker instead: snapshot, then persist with meta
   `{"reviewed_at": "<ISO 8601>", "assessment": "Skipped", "critical_count": 0,
   "medium_count": 0, "low_count": 0, "flags": []}` and body `Review skipped by
   user.` — never re-stamp the old review's hash onto unexamined content.

---

## Rollback

To remove this skill and restore the pre-Step-3 workflow: delete
`.claude/skills/plan-review/`, restore `.claude/commands/review-plan.md` +
`revise-plan.md` from git history, and revert the CLAUDE.md "Plan Review Before
Approval" section to spawn the single review agent. The hook + `plan_snapshot.py`
+ the `.review.md` contract are unchanged and need no rollback.
