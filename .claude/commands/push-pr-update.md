---
description: Push code revisions to an existing PR
argument-hint: "[--message <commit-msg>]"
---

# Push PR Update

Push local changes to an existing pull request branch.

For same-repo PRs, pushing **automatically starts the CI codex review** — this
command must never post an `/ai-review` comment to trigger one. Doing so
double-fires the reviewer and clutters the PR.

For **fork** PRs the reviewer does not run at all: `.github/workflows/ai_pr_review.yml`
gates `pull_request` runs on `head.repo.full_name == github.repository` to avoid
untrusted checkout. Ask the PR itself which case applies:

```bash
gh pr view --json isCrossRepository --jq '.isCrossRepository'
```

`true` means a fork PR — say the reviewer is security-gated and point at
`/ai-review-local`, rather than telling the user to poll for something that will
never arrive.

**Do not infer this by comparing `gh pr view --json headRepositoryOwner` against
`gh repo view --json owner`.** In a fork checkout `gh repo view` resolves to the
fork, so both sides return the fork owner and a genuine cross-repository PR is
misreported as same-repo — precisely backwards. `isCrossRepository` is computed by
GitHub from the PR's head and base repositories and does not depend on which
checkout the CLI is run from.

## Arguments

`$ARGUMENTS` may contain:
- `--message <msg>` (optional): Custom commit message. If omitted, auto-generate from changes.

## Instructions

### 1. Parse Arguments

Parse `$ARGUMENTS` to extract:
- **--message**: Custom commit message (everything after `--message` until next flag or end)

### 2. Validate Current State

> **Refs are data — resolve them into variables, never interpolate `<placeholder>`.**
> A git ref name can contain `$()` or backticks (git accepts them), so pasting a
> resolved default branch / comparison ref into a shell command executes it. Resolve
> them into shell variables via command substitution and use only **quoted** forms
> (`"$DEFAULT_BRANCH"`, `"$COMPARISON_REF..HEAD"`) everywhere below. Variables do not
> persist across separate Bash tool calls, so re-run the two-line resolution at the top
> of any later block that needs them (it is deterministic).

1. **Resolve the default branch into a variable**:
   ```bash
   DEFAULT_BRANCH="$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')"
   ```

2. **Check current branch**:
   ```bash
   CURRENT="$(git branch --show-current)"
   ```
   - If `"$CURRENT"` equals `"$DEFAULT_BRANCH"`, abort:
     ```
     Error: Cannot push PR update from the default branch.
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
     - **Resolve `COMPARISON_REF` into a variable** — quoted, deterministic, handles
       shallow/single-branch clones. Prefer the upstream `@{u}`; else the local or
       `origin/` default branch (fetched shallow if absent):
       ```bash
       DEFAULT_BRANCH="$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')"
       if UP="$(git rev-parse --abbrev-ref @{u} 2>/dev/null)"; then
         COMPARISON_REF="$UP"
       elif git rev-parse --verify "$DEFAULT_BRANCH" >/dev/null 2>&1; then
         COMPARISON_REF="$DEFAULT_BRANCH"
       elif git rev-parse --verify "origin/$DEFAULT_BRANCH" >/dev/null 2>&1; then
         COMPARISON_REF="origin/$DEFAULT_BRANCH"
       else
         git fetch origin "$DEFAULT_BRANCH" --depth=1 2>/dev/null || true
         COMPARISON_REF="origin/$DEFAULT_BRANCH"
       fi
       AHEAD="$(git rev-list --count "$COMPARISON_REF..HEAD" 2>/dev/null || echo 0)"
       ```
     - If `"$AHEAD"` > 0:
       - **Scan for secrets in commits to push** (Section 3a)
       - Files changed: `git diff --name-only "$COMPARISON_REF..HEAD" | wc -l`
       - Proceed to Section 3a (secret scan), 3b (methodology), Section 4 (push with `-u`)
     - If `"$AHEAD"` == 0: abort — "No changes detected; branch has no commits ahead of
       the default branch. Nothing to push."
     - If upstream EXISTS: `AHEAD="$(git rev-list --count "@{u}..HEAD")"`.
       - `"$AHEAD"` > 0 → scan (3a), `git diff --name-only "@{u}..HEAD" | wc -l`, then
         3a → 3b → Section 4.
       - `"$AHEAD"` == 0 → abort — "No changes detected; branch is up to date. Nothing
         to push."

### 3a. Secret Scan for Already-Committed Changes (when skipping Section 3)

When the working tree is clean but commits are ahead, scan for secrets in the commits to be pushed before proceeding to Section 4:

1. **Re-resolve `COMPARISON_REF`** at the top of this block (variables do not persist
   across tool calls) using the same deterministic snippet as Section 2.4, then use it
   **quoted** — never a raw `<comparison-ref>` placeholder.

2. **Run pattern check** using the canonical patterns from `/pre-merge-check` Section 2.6:
   ```bash
   secret_files=$(git diff "$COMPARISON_REF..HEAD" -G "<content pattern from Section 2.6>" --name-only 2>/dev/null || true)
   sensitive_files=$(git diff --name-only "$COMPARISON_REF..HEAD" | grep -iE "<filename pattern from Section 2.6>" || true)
   ```
   Read the actual regex values from `/pre-merge-check` Section 2.6 at execution time. Uses `-G` to search diff content but `--name-only` to output only file names. (`<content pattern>`/`<filename pattern>` are the fixed literal regexes from Section 2.6, not git-controlled data.)

3. **If patterns detected** (i.e., `secret_files` or `sensitive_files` is non-empty), warn with AskUserQuestion:
   ```
   Warning: Potential secrets detected in committed changes:
   - <list of files/patterns>

   These changes are already committed. Options:
   1. Abort - use 'git reset --soft HEAD~N' to uncommit and remove secrets before retrying
   2. Continue anyway - I confirm these are not real secrets
   ```
   Note: Unlike Section 3, we cannot simply unstage these changes since they are already committed.

### 3b. Methodology Checks for Already-Committed Changes (when skipping Section 3)

When the working tree is clean but commits are ahead, check for methodology issues before pushing:

1. **Methodology review of already-committed changes is deferred to `/pre-merge-check`.**
   The changes here are already committed, so this pattern check is non-blocking, and
   the pre-merge gate (run before committing) is the right place for it. Do **not**
   interpolate a comparison ref into a scan command here — `/pre-merge-check` covers the
   working-tree case safely via `premerge_scan.py`, and re-deriving a committed range in
   prose is where injection creeps back in. If you want the committed range checked, run
   `/pre-merge-check` on the branch before it was committed, or review the diff by eye.

3. **Documentation impact check**: Check which source files in `diff_diff/` are in the committed changes.
   If source files are present, read `docs/doc-deps.yaml` and check which dependent
   documentation files are NOT also in the committed changes. Warn about:
   - ALL docs with `type: methodology` (regardless of `drift_risk`)
   - All HIGH `drift_risk` docs (any type)
   ```
   Documentation impact: source files changed but related docs were not updated:
     [METHODOLOGY] docs/methodology/REGISTRY.md — <section hint>
     [HIGH] docs/survey-roadmap.md
   Run /docs-impact for full details.
   ```
   Also warn when the changes touch `diff_diff/` but the branch carries no
   `changelog.d/` fragment (see CONTRIBUTING.md "Changelog fragments").
   This is a WARNING, not a blocker.

Note: Section 3b checks are informational warnings only — no AskUserQuestion prompt, since changes are already committed and cannot be unstaged. This differs from the staged-changes path (Section 3) which offers a "fix vs continue" choice.

### 3. Stage and Commit Changes

1. **Stage all changes**:
   ```bash
   git add -A
   ```

2. **Quick pattern check** — run the argv-safe helper, never a shell grep over
   filenames:
   ```bash
   SCRATCH="$(git rev-parse --git-path premerge-scan)"; mkdir -p "$SCRATCH"
   python3 .claude/scripts/premerge_scan.py --scratch "$SCRATCH"
   ```
   Runs the methodology pattern checks (A–D) in pure Python; **exit 3** = a
   metacharacter-bearing path, **exit 4** = a git/read failure (incomplete scan —
   **stop and report**, do not push on an empty scan). See `/pre-merge-check`
   Section 2.1. If it reports findings:
   ```
   Pre-commit pattern check found N potential issues:
   <list warnings with file:line>

   Options:
   1. Fix issues before committing (recommended)
   2. Continue anyway
   ```
   Use AskUserQuestion. If user chooses to fix, abort the commit flow.

   **Documentation impact check** (if source files are staged):
   If source files in `diff_diff/` are present, read `docs/doc-deps.yaml` and check which
   dependent documentation files are NOT also in the staged set. Warn about:
   - ALL docs with `type: methodology` (regardless of `drift_risk`)
   - All HIGH `drift_risk` docs (any type)
   ```
   Documentation impact: source files changed but related docs were not updated:
     [METHODOLOGY] docs/methodology/REGISTRY.md — <section hint>
   Run /docs-impact for full details.
   ```
   Also warn when the changes touch `diff_diff/` but the branch carries no
   `changelog.d/` fragment (see CONTRIBUTING.md "Changelog fragments").
   This is a WARNING, not a blocker.

3. **Capture file count for reporting**:
   ```bash
   git diff --cached --name-only | wc -l
   ```
   Store as `<files-changed-count>` for use in final report.

4. **Secret scanning check** (same as submit-pr):
   - **Run deterministic pattern check** using the canonical patterns from `/pre-merge-check` Section 2.6:
     ```bash
     secret_files=$(git diff --cached -G "<content pattern from Section 2.6>" --name-only 2>/dev/null || true)
     sensitive_files=$(git diff --cached --name-only | grep -iE "<filename pattern from Section 2.6>" || true)
     ```
     Read the actual regex values from `/pre-merge-check` Section 2.6 at execution time. Uses `-G` to search diff content but `--name-only` to output only file names.
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

5. **Generate or use commit message**:
   - If `--message` provided, use that message
   - Otherwise, generate from changes:
     - Run `git diff --cached --stat` to see what's being committed
     - Analyze the changes and generate a descriptive commit message
     - Use imperative mood ("Add", "Fix", "Update", "Refactor")
   - **Commit via `git commit --file`, never a heredoc.** `--message` here is raw user
     input, and a `git commit -m "$(cat <<'EOF' … EOF)"` heredoc breaks if the message
     contains a line that is exactly `EOF`: the heredoc closes early and the following
     lines run as shell. The Write tool never invokes a shell, and `git commit --file`
     reads the file verbatim. Do this as **three ordered operations, not one shell
     block** (a Write tool call cannot run inside a Bash process):

     1. Derive and print the literal path (one Bash call):
        ```bash
        git rev-parse --git-path push-commit-msg.txt
        ```
     2. **Write the message to that literal path with the Write tool.**
     3. Commit, then clean up **while preserving the commit's exit status** — do not let
        `rm` become the block's successful last command and mask a failed commit (one
        Bash call, re-deriving the path):
        ```bash
        MSG_FILE="$(git rev-parse --git-path push-commit-msg.txt)"
        rc=0; git commit --file "$MSG_FILE" || rc=$?
        rm -f "$MSG_FILE"
        [ "$rc" -eq 0 ] || { echo "commit failed ($rc)"; exit "$rc"; }
        ```
     Do NOT append `Co-Authored-By`, `Claude-Session`, "Generated with Claude
     Code", or any other authorship trailer. The commit message describes the
     change, not who typed it.

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

### 5. Report Results

```
Changes pushed to PR #<number>

Commit: <hash> - <message>
Files changed: <files-changed-count>

PR URL: <url>

<AI review line — same-repo vs fork, per the note at the top of this file>
CI tests require the `ready-for-ci` label, which the user adds (never Claude).
```

- **Same-repo**: `AI code review started automatically on push — poll the PR for the bot's Overall Assessment rather than posting anything to request it.`
- **Fork**: `The CI AI reviewer is security-gated and will NOT run on fork PRs. Use /ai-review-local instead.`

## Error Handling

### Not on a Feature Branch
```
Error: Cannot push PR update from the default branch.
Switch to a feature branch or use /submit-pr to create a new PR.
```

### No Changes to Commit or Push (with upstream)
```
No changes detected. Working directory is clean and branch is up to date.
Nothing to push.
```

### No Changes to Commit or Push (no upstream, no commits ahead)
```
No changes detected. Working directory is clean and branch has no commits ahead of the default branch.
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
# Push changes with auto-generated commit message
/push-pr-update

# Push with custom commit message
/push-pr-update --message "Address PR feedback: fix edge case handling"
```

## Notes

- This command is for updating existing PRs. Use `/submit-pr` to create new PRs.
- Always stages ALL changes (`git add -A`). Stage manually first for partial commits.
- Pushing auto-starts the CI codex review. Never post `/ai-review` to trigger it.
- Uses the `gh` CLI throughout. The GitHub MCP server is NOT used in this repo.
- Uses the same secret scanning as `/submit-pr` to prevent accidental credential commits.
