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

### 2. Detect Remote Configuration

Determine if this is a fork-based workflow:

1. **Check for upstream remote**:
   ```bash
   git remote get-url upstream 2>/dev/null
   ```

2. **Set remote variables**:
   - If `upstream` exists → **fork workflow**:
     - `<base-remote>` = `upstream` (for base comparisons, PR target)
     - `<push-remote>` = `origin` (for pushing branches)
     - Extract `<upstream-owner>/<upstream-repo>` from upstream URL
     - Extract `<fork-owner>` from origin URL
   - If `upstream` does not exist → **direct workflow**:
     - `<base-remote>` = `origin`
     - `<push-remote>` = `origin`
     - Extract `<owner>/<repo>` from origin URL

3. **Fetch from base remote**:
   ```bash
   git fetch <base-remote>
   ```

### 3. Sync with Remote

1. **Check if behind base branch**:
   ```bash
   git rev-list --count HEAD..<base-remote>/<base-branch>
   ```
   - If count > 0, we're behind. Warn user and offer options:
     ```
     Your branch is X commits behind <base-remote>/<base-branch>.

     Options:
     1. Rebase first: git pull --rebase <base-remote> <base-branch>
     2. Continue anyway (may have merge conflicts in PR)
     ```
   - Use AskUserQuestion to let user choose whether to continue or abort

### 4. Check for Changes

1. **Check for uncommitted changes**:
   ```bash
   git status --porcelain
   ```
   - If output is non-empty, there are staged or unstaged changes → proceed to step 5

2. **Check for unpushed commits** (if no uncommitted changes):
   ```bash
   git rev-list --count <base-remote>/<base-branch>..HEAD
   ```
   - If count > 0, there are unpushed commits → skip to step 7
   - If count == 0, inform user and exit:
     ```
     No changes detected. Your working directory is clean and up-to-date with <base-remote>/<base-branch>.
     Nothing to submit.
     ```

### 5. Resolve Branch Name (BEFORE any commits)

**IMPORTANT**: Always resolve the branch name before staging or committing to avoid commits on the base branch.

1. **Check current branch**:
   ```bash
   git branch --show-current
   ```

2. **If on base branch (e.g., `main`)**:
   - Generate or use provided branch name
   - **Generate branch name** (if not provided via `--branch`):
     - Analyze changes to understand the change type:
       ```bash
       git diff --stat              # Unstaged changes
       git diff --cached --stat     # Staged changes
       git status --porcelain       # All changes summary
       ```
     - **Sanitize the branch name** (from title or generated):
       1. Lowercase the string
       2. Replace spaces with hyphens
       3. Remove invalid git ref characters: `:`, `?`, `*`, `[`, `]`, `^`, `~`, `\`, `@{`, `..`
       4. Replace consecutive hyphens/underscores with single hyphen
       5. Trim leading/trailing hyphens
       6. Truncate to reasonable length (50 chars max for branch name portion)
     - Prefix based on change type: `feature/`, `fix/`, `refactor/`, `docs/`
     - **Validate with git**:
       ```bash
       git check-ref-format --branch "<branch-name>"
       ```
       - If validation fails, prompt user for a valid branch name
     - If no diff output and no title provided, prompt user for branch name
   - **Create and switch to the new branch BEFORE staging**:
     ```bash
     git checkout -b <branch-name>
     ```

3. **If already on a feature branch**:
   - Use the current branch name
   - No need to create a new branch

### 6. Stage and Commit Changes

1. **Stage all changes first**:
   ```bash
   git add -A
   ```

2. **Secret scanning check** (AFTER staging to catch all files):
   - **Run deterministic pattern check** (case-insensitive with expanded patterns):
     ```bash
     git diff --cached | grep -iE "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|api[_-]?key[[:space:]]*[=:]|secret[_-]?key[[:space:]]*[=:]|password[[:space:]]*[=:]|private[_-]?key|bearer[[:space:]]+[a-zA-Z0-9_-]+|token[[:space:]]*[=:])" || true
     ```
   - **Check for sensitive file names** (case-insensitive):
     ```bash
     git diff --cached --name-only | grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$" || true
     ```
   - **Optional**: For more thorough scanning, use dedicated tools if available:
     ```bash
     # gitleaks detect --staged --no-git  # If gitleaks installed
     # trufflehog git file://. --only-verified --fail  # If trufflehog installed
     ```
   - Pay special attention to newly added files:
     ```bash
     git diff --cached --name-only --diff-filter=A
     ```
   - If pattern check returns matches or sensitive files detected, **unstage and warn**:
     ```bash
     git reset HEAD  # Unstage all files
     ```
     Then use AskUserQuestion:
     ```
     Warning: Potential secrets detected in files:
     - .env.local (contains API_KEY=)
     - config.json (contains "password":)

     Files have been unstaged for safety.

     Options:
     1. Abort - review and remove secrets before retrying
     2. Continue anyway - I confirm these are not real secrets (will re-stage)
     ```
   - If user chooses to continue, re-stage with `git add -A`

3. **Generate commit message**:
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

### 7. Push Branch to Remote

1. **Resolve and validate branch name**:
   ```bash
   git branch --show-current
   ```

2. **Guard: Prevent pushing from base branch**:
   - If current branch equals `<base-branch>` (e.g., `main`):
     - This can happen when step 4 skipped to step 7 due to unpushed commits on base
     - **Must create a new branch before proceeding**:
       - Generate branch name from unpushed commits (analyze `git log <base-remote>/<base-branch>..HEAD`)
       - Use provided `--branch` name if available
       - Create and switch:
         ```bash
         git checkout -b <branch-name>
         ```
     - If branch creation fails or is declined, abort with error:
       ```
       Error: Cannot create PR from base branch to itself.
       Please create a feature branch first or provide --branch <name>.
       ```

3. **Push to push-remote** (always `origin`, even in fork workflows):
   ```bash
   git push -u <push-remote> <branch-name>
   ```

### 8. Extract Commit Information for PR Body

1. Get commits on this branch (compare against base-remote to avoid stale data):
   ```bash
   git log <base-remote>/<base-branch>..HEAD --oneline
   ```

2. Get changed files:
   ```bash
   git diff <base-remote>/<base-branch>..HEAD --stat
   ```

3. Categorize changes for the template:
   - **Estimator/math changes**: files in `diff_diff/`, `rust/src/`, or `docs/methodology/`
   - Test changes: files in `tests/`
   - Documentation: files in `docs/`, `*.md`, `*.rst`

### 9. Generate PR Body

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

### 10. Create Pull Request

Use the MCP GitHub tool to create the PR:

```
mcp__github__create_pull_request with parameters:
  - owner: <target-owner>      # upstream-owner (fork) or owner (direct)
  - repo: <target-repo>        # upstream-repo (fork) or repo (direct)
  - title: <PR title>
  - head: <head-ref>           # See below for fork vs direct
  - base: <base-branch>
  - body: <generated PR body>
  - draft: <true if --draft flag provided>
```

**Head reference format**:
- **Direct workflow**: `head: <branch-name>`
- **Fork workflow**: `head: <fork-owner>:<branch-name>`

**Extract remote info**:
```bash
# For target (where PR is created)
git remote get-url <base-remote>
# Parse: git@github.com:owner/repo.git or https://github.com/owner/repo.git

# For fork owner (if fork workflow)
git remote get-url origin
# Extract owner from URL
```

### 11. Report Results

```
Pull request created successfully!

Branch: <branch-name>
PR: #<number> - <title>
URL: https://github.com/<target-owner>/<target-repo>/pull/<number>

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
- **Fork workflows supported**: If `upstream` remote exists, PRs target upstream with `<fork-owner>:<branch>` head reference
