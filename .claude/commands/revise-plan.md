---
description: Read plan review feedback and revise the plan with user overrides
argument-hint: "[<plan-path>] [-- <user-notes>]"
---

# Revise Plan

Read structured review feedback for a Claude Code plan, display it in the terminal for user consideration, and revise the plan to address the issues — incorporating user overrides and notes.

## Arguments

`$ARGUMENTS` may contain:
- **Plan file path** (optional): Path to the plan file, e.g., `~/.claude/plans/foo.md`. If omitted, auto-detected from the most recent `.md` file in `~/.claude/plans/` (excluding `*.review.md` files).
- `--` separator followed by **user notes** (optional): Free-form text with directives about which review items to accept, reject, or modify.

Parse `$ARGUMENTS` by splitting on ` -- ` (space-dash-dash-space). Everything before the separator is the plan path (if non-empty after trimming). Everything after is user notes. If `$ARGUMENTS` does not contain ` -- `, the entire argument is treated as the plan path (or empty if `$ARGUMENTS` is empty).

Examples:
- `/revise-plan` — auto-detect plan, accept all review feedback
- `/revise-plan ~/.claude/plans/foo.md` — specific plan, accept all feedback
- `/revise-plan -- Disagree with CRITICAL #2, the API handles this` — auto-detect, with overrides
- `/revise-plan ~/.claude/plans/foo.md -- Skip all LOW items, for MEDIUM #1 use a simpler approach` — specific plan with overrides

## Constraints

- **Plan file only**: Only modifies the plan file. Does NOT create, edit, or delete any project source code, tests, or documentation.
- **Works outside plan mode**: This skill is invoked from a normal (non-plan-mode) conversation. It enters plan mode via `EnterPlanMode` for the revision step.
- **Terminal-first**: The review content is always displayed in the terminal for the user to read before any revision begins.

## Instructions

### Step 1: Locate Plan File

Determine the plan file path:

1. **From arguments**: If a path was provided before ` -- `, use it.
2. **From most recent plan**: If no path provided, find the most recent plan:
   ```bash
   ls -t ~/.claude/plans/*.md 2>/dev/null | grep -v '\.review\.md$' | head -1
   ```
3. **Ask user**: If no plan file found, use AskUserQuestion:
   ```
   Which plan file would you like to revise?
   Enter the path (e.g., ~/.claude/plans/plan-name.md)
   ```

Verify the plan file exists by reading it. If it doesn't exist, report the error and stop.

### Step 2: Locate and Read Review File

Locate the review via the helper (review filenames carry a canonical-path
digest — NEVER derive them from the basename): first initialize the scratch
dir (it need not exist on a fresh worktree):
```bash
SCRATCH="$(git rev-parse --git-path plan-review)"
mkdir -p "$SCRATCH" && echo "$SCRATCH"
```
Write the raw plan path to `<scratch>/plan-path.txt` (Write tool), then `python3 .claude/scripts/plan_snapshot.py check --plan-path-file
"$SCRATCH/plan-path.txt"` — its JSON reports `plan_path`, `review_path`,
`review_exists`, and `fresh`. Confirm the printed `plan_path` is the plan you
supplied — the ingress file is shared per-worktree, so a concurrent session
may have overwritten it between your Write and the check; if it differs,
re-Write the path and re-run the check before using any of its output.

**If `review_exists` is true**: Read the file at the printed `review_path`,
proceed to Step 3.

**If `review_exists` is false**: Use AskUserQuestion:
- "Run a review now (spawns a review agent)" (Recommended)
- "Skip review and enter plan mode directly"

If "Run a review now" is chosen:
- **First take an immutable snapshot via the tested helper** (the raw plan
  path is untrusted — it reaches the helper via a file written with the Write
  tool, never a shell): derive `SCRATCH="$(git rev-parse --git-path
  plan-review)"` (mkdir -p it), Write the raw plan path to
  `<scratch>/plan-path.txt`, then run
  `python3 .claude/scripts/plan_snapshot.py snapshot --plan-path-file
  "$SCRATCH/plan-path.txt"` — it prints JSON with `state_path`,
  `snapshot_path`, `meta_path`, `body_path`, `plan_path`, `plan_sha256`,
  `review_path`. Confirm the printed `plan_path` is the plan you supplied
  (detects a concurrent ingress overwrite; re-run if not). Non-zero exit:
  report and stop.
- Use the Task tool with `subagent_type: "general-purpose"`, pointing the agent
  at the SNAPSHOT path (never the live plan). Prompt the agent:
  ```
  You are reviewing a Claude Code plan file as an independent reviewer.

  1. Read the review criteria from `.claude/commands/review-plan.md` (Steps 2 through 5)
  2. Read the plan file at: <snapshot-path> (an immutable snapshot of the plan)
  3. Follow the review instructions: read CLAUDE.md for project context, read referenced files, evaluate across 8 dimensions, present structured feedback
  4. Number each issue sequentially within its severity section (CRITICAL #1, MEDIUM #1, etc.)
  5. Return ONLY the structured review output (from "## Overall Assessment" through "## Summary"). Do NOT include the "## Plan Content" display (Step 4b) — it is for terminal display only and must not be persisted to the review file.
  ```
- After the agent returns: Write its review output (from "## Overall
  Assessment" through "## Summary" — no plan content) to the exact `body_path`
  printed by snapshot, Write the meta JSON (reviewed_at, assessment, counts,
  `"flags": []`) to the exact `meta_path`, then run
  `python3 .claude/scripts/plan_snapshot.py persist --state-file
  "<state-path>"` (the literal printed path). Exit 3 = the plan changed while
  being reviewed: NOT persisted — relay the message and re-run. The helper
  certifies the recorded snapshot digest and sets `plan:`/`plan_sha256:`
  itself. On any pre-persist failure or cancellation, run
  `plan_snapshot.py abort --state-file "<state-path>"` so temporary files
  never accumulate.
- Proceed to Step 3 with the review content

If "Skip review" is chosen:
- Skip Steps 3-5 (no review to display, check, or parse)
- In Step 6, since there are no review issues, present only:
  - "Enter plan mode with general guidance" (Recommended)
  - "Cancel"
  If "Cancel" is chosen, stop and report "Revision cancelled."
- In Step 7, since there are no review issues to address:
  - Skip rule-based revision (no CRITICAL/MEDIUM/LOW to process)
  - Apply user notes as general guidance for the revision
  - Write a "Skipped" marker via the helper (it snapshots, hashes, and
    stamps `plan:`/`plan_sha256:` itself): Write the raw plan path to
    `<scratch>/plan-path.txt`, run `plan_snapshot.py snapshot
    --plan-path-file ...` (as in Step 2 — confirm the printed `plan_path`
    is the plan you supplied before using its output), Write
    `{"reviewed_at": "<ISO 8601>", "assessment": "Skipped",
    "critical_count": 0, "medium_count": 0, "low_count": 0, "flags": []}` to
    the printed `meta_path`, Write `Review skipped by user.` to the printed
    `body_path`, then run `plan_snapshot.py persist --state-file
    "<state-path>"` before calling `ExitPlanMode` to satisfy the hook.
  - In `## Revision Notes`, record: "Review skipped — revision based on user notes only"
  - All issue counts are zero in the Addressed/Dismissed/Open sections
  - If the review marker write fails, report an error and stop — the hook requires this file on disk.

### Step 3: Display Plan and Review in Terminal

Display both the plan content and the review in the conversation. The plan was already read in Step 1, and the review in Step 2. This is the primary reading surface — the user reads both here.

```
## Plan: <plan-filename>

<full plan content>

---

## Review for <plan-filename>

<full review content (excluding YAML frontmatter)>

---
Source: <review-file-path>
```

### Step 4: Staleness Check

Compare the plan's content hash against the one recorded in the review file
via the helper's read-only probe (the raw path flows through file ingress —
never a shell): Write the raw plan path to `<scratch>/plan-path.txt`
(`SCRATCH="$(git rev-parse --git-path plan-review)"`), then:
```bash
SCRATCH="$(git rev-parse --git-path plan-review)"
python3 .claude/scripts/plan_snapshot.py check --plan-path-file "$SCRATCH/plan-path.txt"
```
Confirm the printed `plan_path` is the plan you supplied (shared per-worktree
ingress — a concurrent session may have overwritten it); if it differs,
re-Write the path and re-run the check.

If the printed `fresh` is false (hash mismatch, or no `plan_sha256` recorded),
the review is STALE — it reviewed different plan content. Warn:
```
Warning: The plan file was modified after this review was generated.
The review may be commenting on an older version of the plan.
Consider running `/review-plan <plan-path> --updated` for a fresh review.
```

### Step 5: Parse and Summarize Review

Extract from the review content:
- Issues by severity: CRITICAL #N, MEDIUM #N, LOW #N
- Checklist gaps
- Questions for Author
- Assessment

Display a summary:
```
Found: N CRITICAL, N MEDIUM, N LOW issues, N checklist gaps, N questions
Assessment: <assessment>
```

### Step 6: Collect User Input

If user notes were provided in `$ARGUMENTS` (after ` -- `), parse them for directives:
- "disagree with #N" or "dismiss #N" → mark that issue as dismissed
- "skip #N" → mark as dismissed
- "skip all LOW" → dismiss all LOW severity issues
- "for #N, do X instead" → override the suggested fix
- "address all" or no specific directives → accept all feedback
- Free-form text applies as general guidance

If no user notes were provided, use AskUserQuestion:
- "Address all issues" (Recommended)
- "Let me specify which items to address or dismiss"

If "Let me specify" is chosen, the user provides free-form text. Parse as above.

### Step 7: Enter Plan Mode and Revise

Call `EnterPlanMode` to transition into plan mode. In plan mode:

1. **Read the current plan file** in full
2. **Read source files** referenced in CRITICAL/MEDIUM issues and files the plan proposes to modify
3. **Revise the plan** following these rules:
   - **CRITICAL issues**: Address unless user explicitly dismissed with justification
   - **MEDIUM issues**: Address unless user dismissed
   - **LOW issues**: Skip unless user explicitly requested them
   - **Questions for Author**: Incorporate user's answers if provided; otherwise note as "Open — to be resolved during implementation"
   - **Checklist gaps**: Add missing items as plan steps where appropriate
4. **Append a `## Revision Notes` section** at the end of the plan:
   ```markdown
   ## Revision Notes

   Revised based on review at <review-file-path>.

   ### Addressed
   - CRITICAL #1: <brief description of what was changed>
   - MEDIUM #2: <brief description>

   ### Dismissed
   - MEDIUM #3: <reason — user's justification or "user override">

   ### Open
   - Question #1: <question text — to be resolved during implementation>
   ```
5. **Write the revised plan** using the Edit or Write tool
6. **Re-review the revised plan** — never merely re-stamp the old review's hash
   onto content it did not examine (that would recreate the old `touch` bypass
   with better cosmetics). Spawn the review agent again (Step 2's "Run a review
   now" flow) over the REVISED plan; that fresh review writes the new
   `plan_sha256`. If the user explicitly declines a re-review, write a fresh
   Skipped marker (Step 2's template, with the new hash) — its
   `assessment: "Skipped"` accurately records that the revised content was not
   independently reviewed. The ExitPlanMode hook denies on hash mismatch; there
   is no mtime/touch bypass.
7. **Call `ExitPlanMode`** for user approval

### Step 8: Report

After exiting plan mode, summarize:
```
Plan revised: <plan-path>
- Addressed: N issues
- Dismissed: N issues
- Open: N items

To re-review the revised plan:
  /review-plan <plan-path> --updated
```

## Notes

- This skill works outside plan mode and transitions into plan mode for the revision
- The review is always displayed in the terminal for the user to read before any revision begins
- If no review file exists, a review agent can be spawned automatically — no second window needed
- User notes/overrides take priority over review recommendations
- Only the plan file is modified; no project source code is touched
- The `## Revision Notes` section provides an audit trail of what was addressed and why
- Pairs with `/review-plan` for iterative revision: `/review-plan` generates feedback, `/revise-plan` addresses it
- For subsequent rounds, run `/review-plan <path> --updated` to get a delta assessment of improvements
