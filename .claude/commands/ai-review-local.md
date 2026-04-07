---
description: Run AI code review locally using OpenAI API before opening a PR
argument-hint: "[--context minimal|standard|deep] [--include-files <files>] [--token-budget <n>] [--force-fresh] [--full-registry] [--model <model>] [--timeout <seconds>] [--dry-run]"
---

# Local AI Code Review

Run a structured code review using the OpenAI Responses API. Reviews changes
against the same methodology criteria used by the CI reviewer, but adapted for local
pre-PR use. Designed for iterative review/revision cycles before submitting a PR.

## Arguments

`$ARGUMENTS` may contain optional flags:
- `--context {minimal,standard,deep}`: Context depth (default: `standard`)
  - `minimal`: Diff only (original behavior)
  - `standard`: Diff + full contents of changed `diff_diff/` source files
  - `deep`: Standard + import-graph expansion (files imported by changed files)
- `--include-files <file1,file2,...>`: Extra files to include as read-only context
  (filenames resolve under `diff_diff/`, or use paths relative to repo root)
- `--token-budget <n>`: Max estimated input tokens before dropping import-context
  files (default: 200000). Changed source files are always included regardless of budget.
- `--force-fresh`: Skip delta-diff mode, run a full fresh review even if previous state exists
- `--full-registry`: Include the entire REGISTRY.md instead of selective sections
- `--model <name>`: Override the OpenAI model (default: `gpt-5.4`)
- `--timeout <seconds>`: HTTP request timeout (default: 300). Use 900 for reasoning models.
- `--dry-run`: Print the compiled prompt without calling the API

**Reasoning models** (`gpt-5.4-pro`, `o3`, `o4-mini`, etc.): Reviews may take 10-15
minutes. For deep reviews with reasoning models, combine `--token-budget` with `--model`:
```
/ai-review-local --model gpt-5.4-pro --token-budget 500000 --context deep
```

## Constraints

This skill does not modify source code files. It may:
- Create a commit if there are uncommitted changes (Step 3)
- Create/update review artifacts in `.claude/reviews/` (gitignored)
- Write temporary files to `/tmp/` (cleaned up in Step 8)

Step 5 makes a single external API call to OpenAI. Step 3b runs a secret scan
before any data is sent externally.

## Instructions

### Step 1: Parse Arguments

Parse `$ARGUMENTS` for the optional flags listed above. All flags are optional —
the default behavior (standard context, selective registry, gpt-5.4, live API call)
requires no arguments.

### Step 2: Validate Prerequisites

Run these checks in parallel:

```bash
# Check API key is set (never echo/log the actual value)
test -n "$OPENAI_API_KEY" && echo "API key: set" || echo "API key: MISSING"

# Check script exists
test -f .claude/scripts/openai_review.py && echo "Script: found" || echo "Script: MISSING"
```

If the API key is missing (and not `--dry-run`):
```
Error: OPENAI_API_KEY is not set.

To set it up:
1. Get a key from https://platform.openai.com/api-keys
2. Add to your shell: echo 'export OPENAI_API_KEY=sk-...' >> ~/.zshrc
3. Reload: source ~/.zshrc
```

If the script is missing:
```
Error: .claude/scripts/openai_review.py not found.
This file should be checked into the repository.
```

Ensure the reviews directory exists:
```bash
mkdir -p .claude/reviews
```

### Step 3: Commit Changes and Generate Diff

Determine and validate the comparison ref (matching the pattern from `/push-pr-update`):

```bash
# Get the repo's default branch name
default_branch=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name' 2>/dev/null || echo "main")

# Resolve to a validated local ref (fallback chain for shallow/single-branch clones)
if git rev-parse --verify "$default_branch" >/dev/null 2>&1; then
    comparison_ref="$default_branch"
elif git rev-parse --verify "origin/$default_branch" >/dev/null 2>&1; then
    comparison_ref="origin/$default_branch"
else
    git fetch origin "$default_branch" --depth=1 2>/dev/null || true
    comparison_ref="origin/$default_branch"
fi
```

If the comparison ref still doesn't resolve, abort:
```
Error: Cannot resolve comparison ref for '$default_branch'. Ensure you have
fetched the default branch: git fetch origin $default_branch
```

Check for uncommitted changes (modified, staged, or untracked):
```bash
git status --porcelain
```

If there are uncommitted changes, commit them before proceeding:

1. Show the user what will be committed (the `git status --porcelain` output above)
2. Stage all changes: `git add -A`
3. Create a commit with a descriptive message summarizing the changes. Follow the
   repository's commit message conventions (see recent `git log --oneline`).
4. Report: "Committed changes: <commit message> (<short sha>)"

If the commit fails (e.g., pre-commit hook), display the error and stop.

Generate diff and metadata:
```bash
git diff --unified=5 "${comparison_ref}...HEAD" > /tmp/ai-review-diff.patch
git diff --name-status "${comparison_ref}...HEAD" > /tmp/ai-review-files.txt
branch_name=$(git branch --show-current)
```

If the diff is empty, report:
```
No committed changes vs ${comparison_ref} to review.
```
Clean up temp files and stop.

### Step 3b: Secret Scan Before API Upload

Before sending any diff content to OpenAI, run the canonical secret scan patterns
(from `/pre-merge-check` Section 2.6) against the same diff range:

```bash
# Content pattern — search diff content, output filenames only (never echo secret values)
secret_files=$(git diff -G "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])" --name-only "${comparison_ref}...HEAD" 2>/dev/null || true)

# Sensitive filename pattern
sensitive_files=$(git diff --name-only "${comparison_ref}...HEAD" | grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$" || true)
```

If either `secret_files` or `sensitive_files` is non-empty, use AskUserQuestion:
```
Warning: Potential secrets detected in files that would be sent to OpenAI:
- <list filenames>

The diff containing these files is about to be transmitted to the OpenAI API.

Options:
1. Abort — review and remove secrets before retrying
2. Continue — I confirm these are not real secrets
```

If user selects abort, clean up temp files and stop. If continue, proceed.

### Step 3c: Full-File Secret Scan (for standard/deep context)

When `--context` is not `minimal`, full source files will be uploaded to OpenAI. The diff-based scan in Step 3b only covers changed lines, so scan the full content of every file that will be transmitted:

```bash
# Category 1: Changed diff_diff/ source files (standard/deep)
upload_scan_files=""
if [ "$context_level" != "minimal" ]; then
    upload_scan_files=$(git diff --name-only "${comparison_ref}...HEAD" | grep "^diff_diff/.*\.py$" || true)
fi

# Category 2: All diff_diff/*.py for deep mode (conservative superset of import expansion)
if [ "$context_level" = "deep" ]; then
    upload_scan_files=$(find diff_diff -name "*.py" -not -path "*/__pycache__/*" 2>/dev/null | sort -u)
fi

# Category 3: --include-files (mirror script's path confinement)
if [ -n "$include_files" ]; then
    repo_root_real=$(pwd -P)
    for f in $(echo "$include_files" | tr ',' ' '); do
        # Reject absolute paths (mirrors script's os.path.isabs check)
        case "$f" in /*) continue ;; esac
        # Mirror script's branching: only bare filenames (no /) resolve under diff_diff/
        # Slash-containing inputs resolve relative to repo root only
        case "$f" in
            */*) if [ -f "$f" ]; then candidate="$f"; else continue; fi ;;
            *)   if [ -f "diff_diff/$f" ]; then candidate="diff_diff/$f"; else continue; fi ;;
        esac
        # Verify within repo root (mirrors script's realpath containment)
        real_candidate=$(cd "$(dirname "$candidate")" && pwd -P)/$(basename "$candidate")
        case "$real_candidate" in "$repo_root_real"/*) upload_scan_files="$upload_scan_files $candidate" ;; esac
    done
fi

# Scan using same canonical content patterns from Step 3b (never echo secret values)
if [ -n "$upload_scan_files" ]; then
    secret_hits=$(grep -rlE "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])" $upload_scan_files 2>/dev/null || true)
    sensitive_hits=$(echo "$upload_scan_files" | tr ' ' '\n' | grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$" || true)
fi
```

If either `secret_hits` or `sensitive_hits` is non-empty, use AskUserQuestion:
```
Warning: Potential secrets detected in source files that would be uploaded to OpenAI
(full-file scan, not just diff hunks):
- <list filenames>

Options:
1. Abort — review and remove secrets before retrying
2. Continue — I confirm these are not real secrets
```

If user selects abort, clean up temp files and stop. If continue, proceed.

### Step 4: Handle Re-Review State

Check for existing review state and generate delta diff if applicable. **Validate that the
stored state matches the current branch and base** before reusing — stale state from a
different branch can contaminate re-review context.

**If `--force-fresh` is set**, delete prior state but still seed a new baseline:
```bash
rm -f .claude/reviews/review-state.json
rm -f .claude/reviews/local-review-latest.md
rm -f .claude/reviews/local-review-previous.md
echo "Force-fresh: deleted all prior review state. Fresh review will seed new baseline."
# Do NOT pass --previous-review or --delta-diff/--delta-changed-files
# DO still pass --review-state, --commit-sha, --base-ref so the fresh run seeds a new baseline
```

**Otherwise**, validate existing review state using the Python validator (single-point
validation that checks schema version, branch/base match, and required finding fields):
```bash
if [ -f .claude/reviews/review-state.json ]; then
    # Use the script's validate_review_state() for comprehensive validation
    # Returns: last_reviewed_commit and is_valid flag
    validation_result=$(python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('openai_review', '.claude/scripts/openai_review.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_, _, commit, valid = mod.validate_review_state(
    '.claude/reviews/review-state.json', '$branch_name', '$comparison_ref'
)
print(f'{commit}|{valid}')
" 2>/dev/null || echo "|False")
    last_reviewed_commit=$(echo "$validation_result" | cut -d'|' -f1)
    state_valid=$(echo "$validation_result" | cut -d'|' -f2)

    if [ "$state_valid" != "True" ]; then
        echo "Warning: review-state.json validation failed. Running fresh review."
        rm -f .claude/reviews/review-state.json
        rm -f .claude/reviews/local-review-previous.md
        last_reviewed_commit=""
    fi

    # Only generate delta when state is valid AND commit is an ancestor of HEAD
    if [ -n "$last_reviewed_commit" ] && git merge-base --is-ancestor "$last_reviewed_commit" HEAD 2>/dev/null; then
        # SHA is a valid ancestor — generate delta diff
        git diff --unified=5 "${last_reviewed_commit}...HEAD" > /tmp/ai-review-delta-diff.patch
        git diff --name-status "${last_reviewed_commit}...HEAD" > /tmp/ai-review-delta-files.txt

        # Check if delta is empty
        if [ ! -s /tmp/ai-review-delta-diff.patch ]; then
            echo "No changes since last review (commit ${last_reviewed_commit:0:7}). Use --force-fresh to re-review."
            rm -f /tmp/ai-review-delta-diff.patch /tmp/ai-review-delta-files.txt
            rm -f /tmp/ai-review-diff.patch /tmp/ai-review-files.txt
            # Stop here
        fi
        # State validated and delta generated — preserve previous review for re-review context
        if [ -f .claude/reviews/local-review-latest.md ]; then
            cp .claude/reviews/local-review-latest.md .claude/reviews/local-review-previous.md
            echo "Previous review preserved for re-review context."
        fi
    else
        echo "Warning: Previous review commit is not an ancestor of HEAD (likely rebase). Running fresh review."
        rm -f .claude/reviews/review-state.json
        rm -f .claude/reviews/local-review-previous.md
    fi
fi

# Final cleanup: if delta mode was NOT activated (no delta files generated),
# ensure no stale previous-review file can leak into the fresh review.
# This covers the case where review-state.json is missing entirely but
# local-review-previous.md lingers from a prior run.
if [ ! -f /tmp/ai-review-delta-diff.patch ]; then
    rm -f .claude/reviews/local-review-previous.md
fi
```

**Important**: Previous review text is ONLY preserved when delta mode is active (delta files
generated and state validated). When delta mode is NOT active — whether because
`review-state.json` is missing, branch/base mismatch, non-ancestor, or `--force-fresh` —
the previous review file is deleted to prevent stale findings from leaking into a fresh
review via `--previous-review`.

### Step 5: Run the Review Script

Build and run the command. Include optional arguments only when their conditions are met:
- `--previous-review`: only if `.claude/reviews/local-review-previous.md` exists AND `--force-fresh` was NOT set
- `--delta-diff` and `--delta-changed-files`: only if delta files were generated in Step 4
- `--review-state`, `--commit-sha`, `--base-ref`: always include (even with `--force-fresh`, to seed a new baseline)
- `--context`, `--include-files`, `--token-budget`: pass through from parsed arguments

```bash
python3 .claude/scripts/openai_review.py \
    --review-criteria .github/codex/prompts/pr_review.md \
    --registry docs/methodology/REGISTRY.md \
    --diff /tmp/ai-review-diff.patch \
    --changed-files /tmp/ai-review-files.txt \
    --output .claude/reviews/local-review-latest.md \
    --branch-info "$branch_name" \
    --repo-root "$(pwd)" \
    --context "$context_level" \
    --review-state .claude/reviews/review-state.json \
    --commit-sha "$(git rev-parse HEAD)" \
    --base-ref "$comparison_ref" \
    [--previous-review .claude/reviews/local-review-previous.md] \
    [--delta-diff /tmp/ai-review-delta-diff.patch] \
    [--delta-changed-files /tmp/ai-review-delta-files.txt] \
    [--include-files "$include_files"] \
    [--token-budget "$token_budget"] \
    [--full-registry] \
    [--model <model>] \
    [--timeout <seconds>] \
    [--dry-run]
```

Note: `--force-fresh` is a skill-only flag — it controls whether delta diffs are
generated in Step 4 and is NOT passed to the script.

**Reasoning model handling:** If the model contains `-pro` or starts with `o1`/`o3`/`o4`
(e.g., `gpt-5.4-pro`, `o3`, `o4-mini`):
- Pass `--timeout 900` to the script (unless the user explicitly specified `--timeout`)
- Run the Bash command with `run_in_background: true` (bypasses the 600s Bash tool timeout cap)
- After the background command completes, continue to Step 6

If `--dry-run`: display the prompt output and stop. Report the estimated token count,
cost estimate, and model that would be used.

If the script exits non-zero, display the error output and stop.

### Step 6: Display the Review

Read and display the full contents of `.claude/reviews/local-review-latest.md`.

### Step 7: Summarize Findings and Offer Next Steps

Parse the review output to extract ALL findings. For each finding, capture:
- Severity (P0/P1/P2/P3)
- Section (Methodology, Code Quality, etc.)
- One-line summary of the issue

Note: The script handles writing `review-state.json` automatically (finding tracking
across rounds). The skill does NOT need to write JSON — just pass `--review-state`
and `--commit-sha` to the script.

Present a **findings summary** showing every finding, grouped by severity:

```
## Review Summary: <assessment emoji and label>

### P0 — Blockers
1. [Methodology] <one-line summary> — <file:line>
2. [Security] <one-line summary> — <file:line>

### P1 — Needs Changes
1. [Code Quality] <one-line summary> — <file:line>

### P2 — Should Fix
1. [Documentation] <one-line summary> — <file:line>

### P3 — Informational (no action required)
1. [Maintainability] <one-line summary> — <file:line>
2. [Tech Debt] <one-line summary> — <file:line>
```

Omit severity groups that have zero findings. The full review with details is already
displayed in Step 6 — this summary helps the user quickly assess what needs attention.

Then use AskUserQuestion, tailored to the severity:

**If no findings at all** (clean review):
```
Review passed with no findings. Suggested next steps:
- /submit-pr — commit and open a pull request
```

**For ⛔ or ⚠️ (P0/P1 findings)**:
```
Options:
1. Enter plan mode to address findings (Recommended)
2. Re-run with --full-registry for deeper methodology context
3. Skip — I'll address these manually
```

**For ✅ with P2/P3 findings only**:
```
Options:
1. Address findings before submitting
2. Skip — proceed to /submit-pr
```

**If user chooses to address findings**: Parse the findings from the review output.
The review context is already in the conversation. Start addressing the findings
directly — for P0/P1 issues use `EnterPlanMode` for a structured approach; for P2/P3
issues, fix them directly since they are minor.

After fixes are committed, the user re-runs `/ai-review-local` for a follow-up review.
On re-review, the script automatically activates delta-diff mode (comparing only
changes since the last reviewed commit) and shows a structured findings table
tracking which previous findings have been addressed.

### Step 8: Cleanup

```bash
rm -f /tmp/ai-review-diff.patch /tmp/ai-review-files.txt \
      /tmp/ai-review-delta-diff.patch /tmp/ai-review-delta-files.txt
```

Note: `.claude/reviews/review-state.json` is preserved across review rounds (it
tracks the last reviewed commit and findings). It is cleaned up when the user
runs `--force-fresh` or when a rebase invalidates the tracked commit.

## Error Handling

| Scenario | Response |
|---|---|
| `OPENAI_API_KEY` not set (non-dry-run) | Error with setup instructions (see Step 2) |
| Script file missing | Error suggesting it should be checked in |
| No committed changes | Clean exit with message |
| Script exits non-zero | Display stderr output from script |
| Previous review file missing on re-run | Script warns and continues as fresh review |
| User aborts due to uncommitted changes | Clean exit |
| No changes since last review (empty delta) | Report and stop, suggest `--force-fresh` |
| Rebase invalidates last reviewed commit | Warn, delete stale state, run fresh review |
| `review-state.json` schema mismatch | Script warns and starts fresh |

## Examples

```bash
# Standard review of current branch vs main (default: full source file context)
/ai-review-local

# Review with minimal context (diff only, original behavior)
/ai-review-local --context minimal

# Review with deep context (changed files + imported files)
/ai-review-local --context deep

# Include specific files as extra context
/ai-review-local --include-files linalg.py,utils.py

# Preview the compiled prompt without calling the API
/ai-review-local --dry-run

# Force a fresh review (ignore previous review state)
/ai-review-local --force-fresh

# Use a different model with full registry
/ai-review-local --model gpt-4.1 --full-registry

# Deep review with reasoning model (may take 10-15 minutes)
/ai-review-local --model gpt-5.4-pro --token-budget 500000 --context deep

# Limit token budget for faster/cheaper reviews
/ai-review-local --token-budget 100000
```

## Notes

- This skill does NOT modify source files — it only generates temp files and
  review artifacts in `.claude/reviews/` (which is gitignored). It may also
  create a commit if there are uncommitted changes (Step 3).
- **Context levels**: By default (`standard`), the full contents of changed
  `diff_diff/` source files are sent alongside the diff. This catches "sins of
  omission" — code that should have changed but wasn't (e.g., a wrapper missing
  a new parameter). Use `--context deep` to also include files imported by
  changed files as read-only reference.
- **Delta-diff re-review**: When `review-state.json` exists from a previous run,
  the script automatically generates a delta diff (changes since the last reviewed
  commit) and focuses the reviewer on those changes. The full branch diff is
  included as reference context. Use `--force-fresh` to bypass this.
- **Finding tracking**: The script writes structured findings to `review-state.json`
  after each review. On re-review, previous findings are shown in a table with
  their status (open/addressed), enabling the reviewer to focus on what changed.
- **Cost visibility**: The script shows estimated cost before the API call and
  actual cost (from the API response) after completion.
- Re-review mode activates automatically when a previous review exists in
  `.claude/reviews/local-review-latest.md`
- The review criteria are adapted from `.github/codex/prompts/pr_review.md` (same
  methodology axes, severity levels, and anti-patterns) but framed for local
  code-change review rather than PR review
- The CI review (Codex action with full repo access) remains the authoritative final
  check — local review is a fast first pass to catch most issues early
- **Data transmission**: In non-dry-run mode, this skill transmits the unified diff,
  changed-file metadata, full source file contents (in standard/deep mode),
  import-context files (in deep mode), selected methodology registry text, and
  prior review context (if present) to OpenAI via the Chat Completions API.
  Use `--dry-run` to preview exactly what would be sent.
- This skill pairs naturally with the iterative workflow:
  `/ai-review-local` -> address findings -> `/ai-review-local` -> `/submit-pr`
