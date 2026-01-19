---
description: Commit changes to a new branch, push to GitHub, and open a PR with project template
argument-hint: "[title] [--branch <name>] [--base <branch>] [--draft]"
---

# Submit Pull Request

Commit work, push to a new branch, and open a pull request with the project-specific PR template.

## Arguments

`$ARGUMENTS` may contain:
- **title** (optional): PR title. If omitted, auto-generate from changes/commits.
- `--branch <name>` (optional): Branch name. If omitted, auto-generate from title.
- `--base <branch>` (optional): Base branch for PR. Default: `main`.
- `--draft` (optional): Create as draft PR.

## Instructions

### 1. Parse Arguments

Parse `$ARGUMENTS` to extract:
- **title**: Everything before any `--` flags
- **--branch**: Branch name (if provided)
- **--base**: Base branch (default: `main`)
- **--draft**: Boolean flag

### 2. Sync with Remote

1. **Fetch latest from remote**:
   ```bash
   git fetch origin
   ```

2. **Check if behind base branch**:
   ```bash
   git rev-list --count HEAD..origin/<base-branch>
   ```
   - If count > 0, we're behind. Warn user and offer options:
     ```
     Your branch is X commits behind origin/<base-branch>.

     Options:
     1. Rebase first: git pull --rebase origin <base-branch>
     2. Continue anyway (may have merge conflicts in PR)
     ```
   - Use AskUserQuestion to let user choose whether to continue or abort

### 3. Check for Changes

Run `git status` to check for uncommitted changes:
- If there are staged or unstaged changes, proceed to commit them
- If there are no changes and HEAD is not ahead of base branch, inform user and exit
- If there are no local changes but commits exist that haven't been pushed, skip to step 6

### 4. Stage and Commit Changes

1. **Stage all changes**: `git add -A`

2. **Generate commit message** (if changes exist):
   - Run `git diff --cached --stat` to see what's being committed
   - Analyze the changes and generate a descriptive commit message
   - Use imperative mood ("Add", "Fix", "Update", "Refactor")
   - Format with HEREDOC:
     ```bash
     git commit -m "$(cat <<'EOF'
     <generated commit message>

     Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
     EOF
     )"
     ```

### 5. Create and Switch to New Branch

1. **Generate branch name** (if not provided):
   - Slugify the PR title: lowercase, replace spaces with hyphens
   - Prefix based on change type: `feature/`, `fix/`, `refactor/`, `docs/`

2. **Create and checkout branch**:
   ```bash
   git checkout -b <branch-name>
   ```

### 6. Push Branch to Remote

```bash
git push -u origin <branch-name>
```

### 7. Extract Commit Information for PR Body

1. Get commits on this branch:
   ```bash
   git log <base-branch>..HEAD --oneline
   ```

2. Get changed files:
   ```bash
   git diff <base-branch>..HEAD --stat
   ```

3. Categorize changes for the template:
   - Estimator/math changes: files in `diff_diff/` with statistical content
   - Test changes: files in `tests/`
   - Documentation: files in `docs/`, `*.md`, `*.rst`

### 8. Generate PR Body

Fill in the template:

```markdown
## Summary
- <bullet point for each commit>

## Methodology references (required if estimator / math changes)
- Method name(s): <from code analysis or "N/A - no methodology changes">
- Paper / source link(s): <from docstrings or "N/A">
- Any intentional deviations from the source (and why): <if applicable or "None">

## Validation
- Tests added/updated: <list test files or "No test changes">
- Backtest / simulation / notebook evidence (if applicable): <if tutorials updated or "N/A">

## Security / privacy
- Confirm no secrets/PII in this PR: Yes

---
Generated with Claude Code
```

**Template logic:**
- **Methodology**: Mark "N/A" if no files in `diff_diff/` with estimator/algorithm changes
- **Validation**: List `test_*.py` files changed, note tutorial updates
- **Security**: Default "Yes", but warn if `.env`, credentials, or API key patterns detected

### 9. Create Pull Request

Use the MCP GitHub tool to create the PR:

```
mcp__github__create_pull_request with parameters:
  - owner: <extracted from git remote>
  - repo: <extracted from git remote>
  - title: <PR title>
  - head: <branch-name>
  - base: <base-branch>
  - body: <generated PR body>
  - draft: <true if --draft flag provided>
```

To extract owner/repo from git remote:
```bash
git remote get-url origin
# Parse: git@github.com:owner/repo.git or https://github.com/owner/repo.git
```

### 10. Report Results

```
Pull request created successfully!

Branch: <branch-name>
PR: #<number> - <title>
URL: https://github/<owner>/<repo>/pull/<number>

Changes included:
<list of changed files>

Next steps:
- Review the PR at the URL above
- Request reviewers if needed
- Run /review-pr <number> to get AI review
```

## Error Handling

### No Changes to Commit
```
No changes detected. Your working directory is clean.
Nothing to submit.
```

### Branch Already Exists
```
Branch '<name>' already exists.
Options:
1. Provide different name: /submit-pr "title" --branch <new-name>
2. Delete existing: git branch -D <name>
```

### Push/PR Creation Failed
Show the error and provide manual fallback commands.

## Examples

```bash
# Auto-generate everything
/submit-pr

# With custom title
/submit-pr "Add pre-trends power analysis"

# With custom branch
/submit-pr "Fix bootstrap variance" --branch fix/bootstrap-variance

# Draft PR against different base
/submit-pr "Refactor linalg module" --base develop --draft
```

## Notes

- Always stages ALL changes (`git add -A`). Stage manually first for partial commits.
- Branch names auto-prefixed: feature/, fix/, refactor/, docs/
- Uses MCP GitHub server for PR creation (requires PAT with repo access)
- Git push uses SSH or HTTPS based on remote URL configuration
