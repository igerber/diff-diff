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

1. **Get repository default branch**:
   ```bash
   gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name'
   ```
   Store as `<default-branch>`.

2. **Check current branch**:
   ```bash
   git branch --show-current
   ```
   - If current branch equals `<default-branch>`, abort:
     ```
     Error: Cannot push PR update from <default-branch> branch.
     Switch to a feature branch or use /submit-pr to create a new PR.
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

4. **Check for changes to commit or push**:
   ```bash
   git status --porcelain
   ```
   - If output is empty (working directory clean):
     - Check if branch has an upstream tracking branch:
       ```bash
       git rev-parse --abbrev-ref @{u} 2>/dev/null
       ```
     - If NO upstream exists:
       - Check if branch has commits ahead of default: `git rev-list --count <default-branch>..HEAD 2>/dev/null || echo "0"`
       - If ahead count > 0: Skip to Section 4 (Push to Remote) — will push with `-u` to set upstream
       - If ahead count = 0: Abort (new branch with nothing to push):
         ```
         No changes detected. Working directory is clean and branch has no commits ahead of <default-branch>.
         Nothing to push.
         ```
     - If upstream EXISTS:
       - Check if branch is ahead: `git rev-list --count @{u}..HEAD`
       - If ahead count > 0: Skip to Section 4 (Push to Remote) — there are committed changes to push
       - If ahead count = 0: Abort:
         ```
         No changes detected. Working directory is clean and branch is up to date.
         Nothing to push.
         ```

### 3. Stage and Commit Changes

1. **Stage all changes**:
   ```bash
   git add -A
   ```

2. **Capture file count for reporting**:
   ```bash
   git diff --cached --name-only | wc -l
   ```
   Store as `<files-changed-count>` for use in final report.

3. **Secret scanning check** (same as submit-pr):
   - **Run deterministic pattern check** (file names only, no content leaked):
     ```bash
     git diff --cached -G "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])" --name-only
     ```
     Note: Uses `-G` to search diff content but `--name-only` to output only file names, preventing secret values from appearing in logs.
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

4. **Generate or use commit message**:
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

1. **Check for upstream tracking branch**:
   ```bash
   git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null
   ```

2. **Push to remote**:
   - If upstream exists: `git push`
   - If no upstream: `git push -u origin HEAD`

   If push fails, report error and suggest:
   ```
   Push failed: <error message>

   If the remote has new commits, try:
     git pull --rebase && /push-pr-update
   ```

3. **Get pushed commit info**:
   ```bash
   git log -1 --oneline
   ```

### 5. Trigger AI Review (unless `--no-review`)

If `--no-review` flag was NOT provided:

1. **Get base repository from PR**:
   ```bash
   gh pr view --json baseRepository --jq '.baseRepository.owner.login + "/" + .baseRepository.name'
   ```
   Store as `<owner>/<repo>` (this is the upstream repo, correct for fork workflows).
   Parse to extract `<owner>` and `<repo>`.

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
Files changed: <files-changed-count>

AI code review triggered. Results will appear shortly.

PR URL: <url>
```

**If `--no-review` was used:**
```
Changes pushed to PR #<number>

Commit: <hash> - <message>
Files changed: <files-changed-count>

PR URL: <url>

Tip: Run /ai-review to request AI code review.
```

## Error Handling

### Not on a Feature Branch
```
Error: Cannot push PR update from <default-branch> branch.
Switch to a feature branch or use /submit-pr to create a new PR.
```

### No Changes to Commit or Push (with upstream)
```
No changes detected. Working directory is clean and branch is up to date.
Nothing to push.
```

### No Changes to Commit or Push (no upstream, no commits ahead)
```
No changes detected. Working directory is clean and branch has no commits ahead of <default-branch>.
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
