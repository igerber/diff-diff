---
description: Push code revisions to an existing PR and trigger AI code review
argument-hint: "[--message <commit-msg>] [--no-review]"
---

# Push PR Update

Push local changes to an existing pull request branch and optionally trigger AI code review.

## Arguments

`$ARGUMENTS` may contain:
- `--message <msg>` (optional): Custom commit message. If omitted, auto-generate from changes.
- `--no-review` (optional): Skip triggering AI review after push.

## Instructions

### 1. Parse Arguments

Parse `$ARGUMENTS` to extract:
- **--message**: Custom commit message (everything after `--message` until next flag or end)
- **--no-review**: Boolean flag

### 2. Validate Current State

1. **Check current branch**:
   ```bash
   git branch --show-current
   ```
   - If on `main` (or the repository's default branch), abort:
     ```
     Error: Cannot push PR update from main branch.
     Switch to a feature branch or use /submit-pr to create a new PR.
     ```

2. **Check for changes to commit**:
   ```bash
   git status --porcelain
   ```
   - If output is empty, abort:
     ```
     No changes detected. Working directory is clean.
     Nothing to push.
     ```

3. **Get PR information**:
   ```bash
   gh pr view --json number,url,headRefName,baseRefName
   ```
   - If no PR exists for current branch, abort:
     ```
     Error: No open PR found for branch '<branch-name>'.
     Use /submit-pr to create a new pull request.
     ```
   - Store PR number and URL for later use.

### 3. Stage and Commit Changes

1. **Stage all changes**:
   ```bash
   git add -A
   ```

2. **Secret scanning check** (same as submit-pr):
   - **Run deterministic pattern check** (case-insensitive):
     ```bash
     git diff --cached | grep -iE "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|api[_-]?key\s*[=:]|secret[_-]?key\s*[=:]|password\s*[=:]|private[_-]?key|bearer\s+[a-zA-Z0-9_-]+|token\s*[=:])" || true
     ```
   - **Check for sensitive file names**:
     ```bash
     git diff --cached --name-only | grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$" || true
     ```
   - If patterns detected, **unstage and warn**:
     ```bash
     git reset HEAD
     ```
     Then use AskUserQuestion:
     ```
     Warning: Potential secrets detected in files:
     - <list of files/patterns>

     Files have been unstaged for safety.

     Options:
     1. Abort - review and remove secrets before retrying
     2. Continue anyway - I confirm these are not real secrets (will re-stage)
     ```
   - If user chooses to continue, re-stage with `git add -A`

3. **Generate or use commit message**:
   - If `--message` provided, use that message
   - Otherwise, generate from changes:
     - Run `git diff --cached --stat` to see what's being committed
     - Analyze the changes and generate a descriptive commit message
     - Use imperative mood ("Add", "Fix", "Update", "Refactor")
   - Format with HEREDOC and Co-Authored-By:
     ```bash
     git commit -m "$(cat <<'EOF'
     <commit message>

     Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
     EOF
     )"
     ```

### 4. Push to Remote

1. **Push to the tracked remote branch**:
   ```bash
   git push
   ```
   - If push fails (e.g., remote rejected), report error and suggest:
     ```
     Push failed: <error message>

     If the remote has new commits, try:
       git pull --rebase && /push-pr-update
     ```

2. **Get pushed commit info**:
   ```bash
   git log -1 --oneline
   ```

### 5. Trigger AI Review (unless `--no-review`)

If `--no-review` flag was NOT provided:

1. **Extract repository owner/repo from remote URL**:
   ```bash
   git remote get-url origin
   ```
   Parse to extract `<owner>` and `<repo>` from URL (handles both SSH and HTTPS formats).

2. **Add review comment using MCP tool**:
   ```
   mcp__github__add_issue_comment with parameters:
     - owner: <owner>
     - repo: <repo>
     - issue_number: <PR number from step 2>
     - body: "/ai-review"
   ```

### 6. Report Results

**If AI review triggered:**
```
Changes pushed to PR #<number>

Commit: <hash> - <message>
Files changed: <count>

AI code review triggered. Results will appear shortly.

PR URL: <url>
```

**If `--no-review` was used:**
```
Changes pushed to PR #<number>

Commit: <hash> - <message>
Files changed: <count>

PR URL: <url>

Tip: Run /ai-review to request AI code review.
```

## Error Handling

### Not on a Feature Branch
```
Error: Cannot push PR update from main branch.
Switch to a feature branch or use /submit-pr to create a new PR.
```

### No Changes to Commit
```
No changes detected. Working directory is clean.
Nothing to push.
```

### No Open PR for Branch
```
Error: No open PR found for branch '<branch-name>'.
Use /submit-pr to create a new pull request.
```

### Push Failed
```
Push failed: <error message>

If the remote has new commits, try:
  git pull --rebase && /push-pr-update
```

## Examples

```bash
# Push changes with auto-generated commit message and trigger AI review
/push-pr-update

# Push with custom commit message
/push-pr-update --message "Address PR feedback: fix edge case handling"

# Push without triggering AI review
/push-pr-update --no-review

# Both options together
/push-pr-update --message "Fix typo in docstring" --no-review
```

## Notes

- This skill is for updating existing PRs. Use `/submit-pr` to create new PRs.
- Always stages ALL changes (`git add -A`). Stage manually first for partial commits.
- The `/ai-review` comment triggers the repository's AI review workflow (if configured).
- Uses the same secret scanning as `/submit-pr` to prevent accidental credential commits.
