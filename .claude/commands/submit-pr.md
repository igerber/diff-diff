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
- If there are staged or unstaged changes, proceed to step 4
- If there are no changes and HEAD is not ahead of origin/<base-branch>, inform user and exit
- If there are no local changes but commits exist that haven't been pushed, skip to step 6

### 4. Resolve Branch Name (BEFORE any commits)

**IMPORTANT**: Always resolve the branch name before staging or committing to avoid commits on the base branch.

1. **Check current branch**:
   ```bash
   git branch --show-current
   ```

2. **If on base branch (e.g., `main`)**:
   - Generate or use provided branch name
   - **Generate branch name** (if not provided via `--branch`):
     - Analyze unstaged changes with `git diff --stat` to understand the change type
     - Slugify the PR title (if provided) or generate from changes: lowercase, replace spaces with hyphens
     - Prefix based on change type: `feature/`, `fix/`, `refactor/`, `docs/`
   - **Create and switch to the new branch BEFORE staging**:
     ```bash
     git checkout -b <branch-name>
     ```

3. **If already on a feature branch**:
   - Use the current branch name
   - No need to create a new branch

### 5. Stage and Commit Changes

1. **Secret scanning check**:
   - Run `git diff` and scan for potential secrets:
     - Files matching: `.env*`, `*credentials*`, `*secret*`, `*.pem`, `*.key`
     - Content patterns: `API_KEY=`, `SECRET=`, `PASSWORD=`, `ghp_`, `sk-`, AWS keys, etc.
   - If potential secrets detected, warn user and use AskUserQuestion to confirm:
     ```
     Warning: Potential secrets detected in staged files:
     - .env.local (contains API_KEY=)
     - config.json (contains "password":)

     Options:
     1. Abort - do not commit these files
     2. Continue - I confirm these are not real secrets
     ```

2. **Stage all changes**: `git add -A`

3. **Generate commit message** (if changes exist):
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

### 6. Push Branch to Remote

1. **Ensure branch name is resolved**:
   - If branch name was not resolved in step 4 (e.g., skipped due to no local changes):
     ```bash
     git branch --show-current
     ```
   - Use this as the branch name for push and PR creation

2. **Push to remote**:
   ```bash
   git push -u origin <branch-name>
   ```

### 7. Extract Commit Information for PR Body

1. Get commits on this branch (compare against remote to avoid stale data):
   ```bash
   git log origin/<base-branch>..HEAD --oneline
   ```

2. Get changed files:
   ```bash
   git diff origin/<base-branch>..HEAD --stat
   ```

3. Categorize changes for the template:
   - **Estimator/math changes**: files in `diff_diff/`, `rust/src/`, or changes to `docs/methodology/REGISTRY.md`
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
- **Methodology**: Mark "N/A" only if NO files changed in `diff_diff/`, `rust/src/`, or `docs/methodology/`. If methodology files changed, consult `docs/methodology/REGISTRY.md` for proper citations.
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
URL: https://github.com/<owner>/<repo>/pull/<number>

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
