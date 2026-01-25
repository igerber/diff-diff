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
       - Determine comparison ref (handles shallow/single-branch clones):
         - If `<default-branch>` exists locally (`git rev-parse --verify <default-branch> 2>/dev/null`): use `<default-branch>`
         - Else if `origin/<default-branch>` exists (`git rev-parse --verify origin/<default-branch> 2>/dev/null`): use `origin/<default-branch>`
         - Else: fetch it first (`git fetch origin <default-branch> --depth=1 2>/dev/null || true`), then use `origin/<default-branch>`
         - Store as `<comparison-ref>`
       - Check if branch has commits ahead: `git rev-list --count <comparison-ref>..HEAD 2>/dev/null || echo "0"`
       - If ahead count > 0:
         - **Scan for secrets in commits to push** (see Section 3a below)
         - Compute `<files-changed-count>`: `git diff --name-only <comparison-ref>..HEAD | wc -l`
         - Skip to Section 4 (Push to Remote) — will push with `-u` to set upstream
       - If ahead count = 0: Abort (new branch with nothing to push):
         ```
         No changes detected. Working directory is clean and branch has no commits ahead of <default-branch>.
         Nothing to push.
         ```
     - If upstream EXISTS:
       - Check if branch is ahead: `git rev-list --count @{u}..HEAD`
       - If ahead count > 0:
         - **Scan for secrets in commits to push** (see Section 3a below)
         - Compute `<files-changed-count>`: `git diff --name-only @{u}..HEAD | wc -l`
         - Skip to Section 4 (Push to Remote) — there are committed changes to push
       - If ahead count = 0: Abort:
         ```
         No changes detected. Working directory is clean and branch is up to date.
         Nothing to push.
         ```

### 3a. Secret Scan for Already-Committed Changes (when skipping Section 3)

When the working tree is clean but commits are ahead, scan for secrets in the commits to be pushed before proceeding to Section 4:

1. **Get diff range**: Use `<comparison-ref>..HEAD` (from Section 2.4 — either `@{u}`, `<default-branch>`, or `origin/<default-branch>`)

2. **Run pattern check** (file names only, no content leaked):
   ```bash
   secret_files=$(git diff <comparison-ref>..HEAD -G "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])" --name-only 2>/dev/null || true)
   ```

3. **Check for sensitive file names**:
   ```bash
   sensitive_files=$(git diff --name-only <comparison-ref>..HEAD | grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$" || true)
   ```

4. **If patterns detected**, warn with AskUserQuestion:
   ```
   Warning: Potential secrets detected in committed changes:
   - <list of files/patterns>

   These changes are already committed. Options:
   1. Abort - use 'git reset --soft HEAD~N' to uncommit and remove secrets before retrying
   2. Continue anyway - I confirm these are not real secrets
   ```
   Note: Unlike Section 3, we cannot simply unstage these changes since they are already committed.

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
     secret_files=$(git diff --cached -G "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])" --name-only 2>/dev/null || true)
     ```
     Note: Uses `-G` to search diff content but `--name-only` to output only file names, preventing secret values from appearing in logs. The `|| true` prevents exit status 1 when patterns match from aborting strict runners.
   - **Check for sensitive file names**:
     ```bash
     sensitive_files=$(git diff --cached --name-only | grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$" || true)
     ```
   - **If patterns detected** (i.e., `secret_files` or `sensitive_files` is non-empty), **unstage and warn**:
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
