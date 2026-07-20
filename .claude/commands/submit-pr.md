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

### 1. Parse Arguments and Resolve Safe Values

Parse `$ARGUMENTS` to extract the raw **title** (optional), optional **--branch**,
optional **--base** (default `main`), and the **--draft** flag. Determine a **change
type** (`feature`/`fix`/`refactor`/`docs`) from a quick `git diff --stat` /
`git status --porcelain`.

**Title when none was given.** `/submit-pr` with no title is supported. If you want a
descriptive title, derive it from **base-free** commands only — `git diff --cached
--stat`, `git status --porcelain`, `git log -1 --format=%s` — and write it in step 1b.
**Never build a title with `git log <base>..HEAD`**: at this point `<base>` is the raw,
unvalidated `--base`, and a malicious value would execute on that command line before
the script ever sees it. If you write no title, `pr_prepare.py` generates a base-free
fallback (the last commit subject), so the no-title path is safe either way.

**The raw title/branch/base must never touch a shell — not even a `VAR="…"`
assignment.** A backtick or `$(...)` in the title executes *at the assignment*,
before any script runs (`RAW_TITLE="Fix `` `id` `` "` fires `id`). So materialise the
untrusted values with the **Write tool** (which never invokes a shell) into files,
then pass the *file paths* — which you control and which are metacharacter-free — to
`.claude/scripts/pr_prepare.py`. That script is the single ingress that resolves and
validates every dynamic value; it is unit-tested (`tests/test_pr_prepare.py`),
including an end-to-end subprocess test that runs it with a `$(touch sentinel)` title
and asserts nothing executes.

**Do the three sub-steps in this exact order — they are not one shell block.**

**1a. Derive and clear the scratch directory** (one Bash call). Use a *deterministic*
path, not `mktemp -d`: shell variables do not persist across separate Bash tool calls,
so a random path captured here is gone in the next call. `git rev-parse --git-path`
recomputes the same path every call (and is correct inside worktrees). Clearing it
first prevents a prior interrupted run's stale `raw-branch.txt`/`raw-base.txt` from
being read as fresh explicit flags:

```bash
SCRATCH="$(git rev-parse --git-path pr-prepare)"   # e.g. .git/pr-prepare
rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
echo "$SCRATCH"                                     # note the literal path for 1b
```

**1b. Write the raw values with the Write tool** (NOT `echo`/heredoc/`printf` — those
re-invoke the shell on the content) into the literal path from 1a. Only write a file
for a value that exists:
- `<scratch>/raw-title.txt` ← the title, only if the user gave one *or* you generated
  a base-free one; otherwise omit it and the script generates a fallback
- `<scratch>/raw-branch.txt` ← the explicit `--branch`, only if one was given
- `<scratch>/raw-base.txt` ← the explicit `--base`, only if one was given

**1c. Run the script** (one Bash call — re-derive `SCRATCH`, do NOT clear it again).
**Substitute the change type and draft flag as literals** — they are trusted values
you determined during parsing (a fixed enum and a boolean), and shell variables set in
an earlier tool call are not defined here. Write the literal `--change-type <type>`,
and include the literal `--draft` line only for a draft PR:

```bash
SCRATCH="$(git rev-parse --git-path pr-prepare)"
python3 .claude/scripts/pr_prepare.py \
  --title-file "$SCRATCH/raw-title.txt" \
  --branch-file "$SCRATCH/raw-branch.txt" \
  --base-file "$SCRATCH/raw-base.txt" \
  --current-branch "$(git branch --show-current)" \
  --change-type fix \
  --draft \
  --scratch "$SCRATCH"
```

Replace `fix` with the resolved change type (`feature`/`fix`/`refactor`/`docs`), and
**drop the `--draft` line entirely** when the user did not pass `--draft`. A
missing/empty `--title-file`/`--branch-file`/`--base-file` means "not supplied".
`--current-branch` is trusted git output; the script still validates it.

If it exits non-zero it has **rejected** an unsafe or conflicting explicit
`--branch`/`--base`, or a fork remote that would not parse (it fails closed) — surface
its message and stop.

**Because variables do not persist across tool calls, every later step re-derives
`SCRATCH="$(git rev-parse --git-path pr-prepare)"` and re-reads the value files it
needs** (`BASE_BRANCH="$(cat "$SCRATCH/pr-base.txt")"`, etc.) inside that same call.
Do not assume a variable set in an earlier step is still defined.

**Cleanup on every exit.** A `trap … EXIT` cannot span tool calls, so make it a rule:
if you abort at any later step (rejection here, the user cancels the sync in step 3, a
commit or push fails), re-derive `$SCRATCH` and `rm -rf` it before stopping. Step 10
removes it on the success path.

The script wrote one value per file under `$SCRATCH`:
`pr-base.txt`, `pr-branch.txt`, `pr-headref.txt`, `pr-target-repo.txt`,
`pr-is-fork.txt`, `pr-draft.txt`, `pr-title.txt`. **Re-read the ones a step needs at
the top of that step** (variables do not carry over), always quoted:

```bash
SCRATCH="$(git rev-parse --git-path pr-prepare)"
BASE_BRANCH="$(cat "$SCRATCH/pr-base.txt")"
BRANCH_NAME="$(cat "$SCRATCH/pr-branch.txt")"
```

Every branch/base value matches `[A-Za-z0-9._/-]` and passed `git check-ref-format`;
the title is the opaque file. None can carry a shell metacharacter.

### 2. Resolve Remotes

Remote *names* are fixed literals (`origin`/`upstream`), not user input, so they are
safe to set in prose:

- `IS_FORK == true` → **fork workflow**: `BASE_REMOTE=upstream`, `PUSH_REMOTE=origin`.
  `TARGET_REPO` and `HEAD_REF` (`fork-owner:branch`) were already resolved by the
  script from the remote URLs.
- `IS_FORK == false` → **direct workflow**: `BASE_REMOTE=origin`, `PUSH_REMOTE=origin`.
  `gh` infers the target repo; `HEAD_REF` is just `BRANCH_NAME`.

Then fetch: `git fetch "$BASE_REMOTE"`.

### 3. Sync with Remote

1. **Check if behind base branch** (quoted variables throughout):
   ```bash
   git rev-list --count "HEAD..$BASE_REMOTE/$BASE_BRANCH"
   ```
   - If count > 0, we're behind. Warn user and offer options:
     ```
     Your branch is X commits behind $BASE_REMOTE/$BASE_BRANCH.

     Options:
     1. Rebase first: git pull --rebase "$BASE_REMOTE" "$BASE_BRANCH"
     2. Continue anyway (may have merge conflicts in PR)
     ```
   - Use AskUserQuestion to let user choose whether to continue or abort

### 4. Reconcile branch state (idempotent)

**submit-pr's goal is: this committed branch is pushed and has an open PR.** It is
idempotent — safe to run whether or not the work is already committed or already
pushed. The standard flow commits at `/ai-review-local` (which requires a commit) and
runs `/pre-merge-check` before this, so by here the work is normally *already
committed*; and a rebase/prep step may already have *pushed* it. The terminal state is
**"a PR exists,"** not "there was something to push" — so a fully-committed,
already-pushed branch still proceeds to open its PR (step 10), never dead-ends.

1. **Uncommitted changes?** `git status --porcelain`
   - **Non-empty** → these have NOT been through `/ai-review-local`. Warn and
     AskUserQuestion:
     ```
     N uncommitted file(s) have not been through /ai-review-local.
     1. Commit them anyway (goes in UNREVIEWED — bypasses the review step)
     2. Abort - commit and run /ai-review-local first
     ```
     On option 1, do steps 5, 5b, 6 (create branch if needed, stage, commit). On
     option 2, stop.
   - **Empty** → continue to the zero-diff guard below.

2. **Zero-diff guard — is there anything to submit at all?**
   ```bash
   git rev-list --count "$BASE_REMOTE/$BASE_BRANCH..HEAD"
   ```
   - **0 commits ahead of base** (e.g. a clean `main` that equals `origin/main`):
     there is genuinely nothing to submit. Check for an existing open PR (step 10's
     query) — if one exists, report its URL; if not, exit with **"Nothing to submit —
     no commits ahead of `$BASE_BRANCH`."** Do NOT create or push a branch (that would
     leave remote clutter, then `gh pr create` would fail on an empty diff).
   - **> 0 commits ahead** → real work exists. Continue to step 7 (push-if-needed) then
     step 10 (ensure a PR exists). An already-*pushed* branch with commits and no PR
     still proceeds — that is the idempotent path.

### 5. Create the Branch (BEFORE any commits)

`BRANCH_NAME` was already resolved and safety-validated by the script in step 1 —
whitelist-sanitised if generated, or rejected outright if an unsafe explicit
`--branch` was given. There is **no sanitisation to do here**; just use the quoted
variable.

1. **Check current branch**: `CURRENT="$(git branch --show-current)"`. This is empty
   on a **detached HEAD**.

2. **If on the base branch, OR detached** (`CURRENT` empty or equal to `$BASE_BRANCH`),
   create and switch before staging, so no commit lands on the base or on a detached
   HEAD:
   ```bash
   git checkout -b "$BRANCH_NAME"
   ```
   Detached HEAD must take this path: otherwise a commit lands on no branch, and step 7
   would try to push a generated ref that was never created. (`$BRANCH_NAME` from step 1
   is already the generated name in the detached case, since the script saw an empty
   `--current-branch`.)

3. **Otherwise** (already on a feature branch), use it as-is — no new branch needed.

### 5b. Stage and Quick Pattern Check

1. **Stage all changes**:
   ```bash
   git add -A
   ```

2. **Quick pattern check** — run the argv-safe helper, never a shell grep over
   filenames (a changed path like `diff_diff/$(touch x).py` would execute):
   ```bash
   SCRATCH="$(git rev-parse --git-path premerge-scan)"; mkdir -p "$SCRATCH"
   python3 .claude/scripts/premerge_scan.py --scratch "$SCRATCH"
   ```
   It runs the methodology pattern checks (A–D) in pure Python over file content;
   **exit 3** = a changed path carries shell metacharacters, **exit 4** = a git/read
   failure (the scan is incomplete — **stop and report**, do not commit on an empty
   scan). See `/pre-merge-check` Section 2.1 for the shared definition. If it reports
   findings:
   ```
   Pre-commit pattern check found N potential issues:
   <list warnings with file:line>

   Options:
   1. Fix issues before committing (recommended)
   2. Continue anyway
   ```
   Use AskUserQuestion. If user chooses to fix, abort the commit flow and let them address the issues.

3. **Documentation impact check** (if source files are staged):
   ```bash
   git diff --cached --name-only | grep "^diff_diff/.*\.py$"
   ```
   If source files are present, read `docs/doc-deps.yaml` and check which dependent
   documentation files are NOT also in the staged set. Warn about:
   - ALL docs with `type: methodology` (regardless of `drift_risk`)
   - All HIGH `drift_risk` docs (any type)
   ```
   Documentation impact: source files changed but related docs were not updated:
     [METHODOLOGY] docs/methodology/REGISTRY.md — <section hint>
     [HIGH] docs/survey-roadmap.md
   Run /docs-impact for full details.
   ```
   This is a WARNING, not a blocker.

### 6. Commit Changes

1. **Secret scanning check** (files already staged from 5b):
   - **Run deterministic pattern check** using the canonical patterns from `/pre-merge-check` Section 2.6:
     ```bash
     secret_files=$(git diff --cached -G "<content pattern from Section 2.6>" --name-only 2>/dev/null || true)
     sensitive_files=$(git diff --cached --name-only | grep -iE "<filename pattern from Section 2.6>" || true)
     ```
     Read the actual regex values from `/pre-merge-check` Section 2.6 at execution time. Uses `-G` to search diff content but `--name-only` to output only file names, preventing secret values from appearing in logs.
   - **Optional**: For more thorough scanning, use dedicated tools if available:
     ```bash
     # gitleaks detect --staged --no-git  # If gitleaks installed
     # trufflehog git file://. --only-verified --fail  # If trufflehog installed
     ```
   - Pay special attention to newly added files:
     ```bash
     git diff --cached --name-only --diff-filter=A
     ```
   - **If patterns detected** (i.e., `secret_files` or `sensitive_files` is non-empty), **unstage and warn**:
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
   - **Write the message to a file with the Write tool, then `git commit --file` —
     never a heredoc.** A `git commit -m "$(cat <<'EOF' … EOF)"` heredoc breaks if the
     message body contains a line that is exactly `EOF`: the heredoc closes early and
     the following lines execute as shell. This is the same shell-ingress class
     `pr_prepare.py` exists to prevent; the Write tool never invokes a shell, and
     `git commit --file` reads the file verbatim. Write the message to the literal
     scratch path (the one printed in step 1a), then commit — **re-deriving `SCRATCH`
     in this block**, since it does not carry over from an earlier tool call:
     ```bash
     SCRATCH="$(git rev-parse --git-path pr-prepare)"
     # write the message to "$SCRATCH/commit-msg.txt" with the Write tool first
     git commit --file "$SCRATCH/commit-msg.txt"
     ```
     Do NOT append `Co-Authored-By`, `Claude-Session`, "Generated with Claude
     Code", or any other authorship trailer. The commit message describes the
     change, not who typed it.

### 7. Push Branch to Remote

1. **Resolve and validate branch name**:
   ```bash
   git branch --show-current
   ```

2. **Guard: Prevent pushing from base branch or detached HEAD**:
   - If the current branch is empty (detached HEAD) **or** equals `"$BASE_BRANCH"`:
     - This can happen when step 4 skipped to step 7 due to unpushed commits on base,
       or when HEAD was detached
     - **Must switch to `$BRANCH_NAME` before proceeding** (already resolved safely
       in step 1):
       ```bash
       git checkout -b "$BRANCH_NAME"
       ```
     - If branch creation fails or is declined, abort with error:
       ```
       Error: Cannot create PR from base branch to itself.
       Please create a feature branch first or provide --branch <name>.
       ```

3. **Push to push-remote only if needed** (always `origin`, even in fork workflows).
   Push when there are unpushed commits or no upstream; it's a **no-op when the branch
   is already fully pushed** — that is not an error, just continue to step 10:
   ```bash
   if [ -z "$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null)" ]; then
     git push -u "$PUSH_REMOTE" "$BRANCH_NAME"          # no upstream yet
   elif [ "$(git rev-list --count @{u}..HEAD)" -gt 0 ]; then
     git push "$PUSH_REMOTE" "$BRANCH_NAME"             # unpushed commits
   fi
   # else: already fully pushed — nothing to push, proceed to step 10.
   ```

### 8. Extract Commit Information for PR Body

1. Get commits on this branch (compare against base-remote to avoid stale data):
   ```bash
   git log "$BASE_REMOTE/$BASE_BRANCH..HEAD" --oneline
   ```

2. Get changed files:
   ```bash
   git diff "$BASE_REMOTE/$BASE_BRANCH..HEAD" --stat
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
```

Do not add an authorship footer to the PR body.

**Template logic:**
- **Methodology**: Mark "N/A" only if NO files changed in `diff_diff/`, `rust/src/`, or `docs/methodology/`. If methodology files changed, consult `docs/methodology/REGISTRY.md` for proper citations.
- **Validation**: List `test_*.py` files changed, note tutorial updates
- **Security**: Default "Yes", but warn if `.env`, credentials, or API key patterns detected

### 10. Ensure a PR exists

The terminal state is "an open PR exists for this branch **that matches what was
requested**," so **first check whether one already does** — this is what makes
submit-pr idempotent and safe to re-run. Fetch enough to compare, not just the URL:

```bash
gh pr view --json url,state,baseRefName,isDraft \
  --jq 'select(.state=="OPEN") | "\(.url)\t\(.baseRefName)\t\(.isDraft)"' 2>/dev/null
```

- **An open PR exists and its `baseRefName` == `$BASE_BRANCH` and its `isDraft`
  matches the requested draft state** → report its URL; you are done. Do not open a
  second one.
- **An open PR exists but the base or draft state differs from what was requested**
  → do NOT silently treat it as success. Report the mismatch (e.g. "an open PR for
  this branch already targets `main`, but you asked for `--base develop`") and let the
  user decide — retarget, or accept the existing PR.
- **No open PR** → create it, as below.

(Fork workflows: pass the target repo explicitly to `gh pr view --repo "$TARGET_REPO"`,
since a bare `gh` resolves to the fork.)

All dynamic values were resolved and safety-checked by the script in step 1 and are
already loaded into `TITLE_FILE`/`BASE_BRANCH`/`BRANCH_NAME`/`HEAD_REF`/`TARGET_REPO`.
There is **no sanitisation, no remote-URL parsing, and no placeholder substitution
to do here** — that logic lives in `pr_prepare.py` where it is unit-tested. This step
only writes the body and invokes `gh` with quoted variables.

1. **Write the body with the Write tool** (not a shell heredoc) to `$SCRATCH/pr-body.md`.
   Using the tool means the body is never parsed by a shell and cannot collide with a
   heredoc delimiter. `pr_prepare.py` already wrote `$SCRATCH/pr-title.txt`.

2. **Invoke `gh`, re-deriving everything in this one call** (nothing persisted from
   earlier steps), everything quoted. Draft state comes from the file the script
   wrote, so a `--draft` request cannot be silently dropped:

   ```bash
   SCRATCH="$(git rev-parse --git-path pr-prepare)"
   TITLE="$(cat "$SCRATCH/pr-title.txt")"
   BODY_FILE="$SCRATCH/pr-body.md"
   BASE_BRANCH="$(cat "$SCRATCH/pr-base.txt")"
   HEAD_REF="$(cat "$SCRATCH/pr-headref.txt")"
   TARGET_REPO="$(cat "$SCRATCH/pr-target-repo.txt")"
   IS_FORK="$(cat "$SCRATCH/pr-is-fork.txt")"
   DRAFT=(); [ "$(cat "$SCRATCH/pr-draft.txt")" = "true" ] && DRAFT=(--draft)

   if [ "$IS_FORK" = "true" ]; then
     gh pr create \
       --repo "$TARGET_REPO" --head "$HEAD_REF" --base "$BASE_BRANCH" \
       --title "$TITLE" --body-file "$BODY_FILE" "${DRAFT[@]}"
   else
     # gh infers the target repo from the remotes in the direct workflow.
     gh pr create \
       --base "$BASE_BRANCH" --title "$TITLE" --body-file "$BODY_FILE" "${DRAFT[@]}"
   fi
   ```

   `DRAFT` is a bash array so an empty value expands to *no argument* (not an empty
   string), which the earlier `${DRAFT:+…}` form got wrong under zsh.

3. **Clean up**, on success and failure alike: `rm -rf "$SCRATCH"`.

`gh pr create` prints the PR URL on success — capture it for step 11 rather than
reconstructing it. If it fails, surface the error verbatim; do not retry silently.

**Regression coverage.** The safety property — no title/base/branch value can execute
regardless of backticks or `$()` — is enforced by `tests/test_pr_prepare.py`
(including a payload that would `touch` a sentinel), not by prose. If you change how
values reach the shell here, that suite must still pass.

### 10b. Do NOT force PR ref creation

There is deliberately no step here that pushes again, verifies
`refs/pull/<number>/head`, or creates an empty commit. An earlier version did, and
it was wrong three ways:

1. It probed `PUSH_REMOTE` for the ref. In a fork workflow that is the fork, while
   `refs/pull/<number>/head` only ever exists in the **base** repository — so the
   check came back empty even on success, every time.
2. The fallback then committed `Trigger PR ref creation` and pushed it, putting a
   junk empty commit in the history of every fork PR.
3. That extra push re-triggers the CI AI reviewer, which already started on PR open.

The workaround existed for PRs created through the raw API. `gh pr create` uses the
normal path and GitHub creates the ref itself. If a ref genuinely appears missing,
report it — do not modify the branch to provoke it.

### 11. Report Results

```
Pull request created successfully!

Branch: $BRANCH_NAME
PR: <the URL gh printed>

Changes included:
<list of changed files>

Next steps:
- Review the PR at the URL above
- <AI review line — see below, depends on direct vs fork>
- CI tests require the `ready-for-ci` label, which the user adds (never Claude)
```

**The AI review line is conditional.** `.github/workflows/ai_pr_review.yml` runs
`pull_request` reviews only when `head.repo.full_name == github.repository`, so fork
PRs are skipped at the workflow level as an untrusted-checkout guard.

- **Direct workflow**: `AI code review starts automatically on PR open — do NOT post /ai-review`
- **Fork workflow**: `The CI AI reviewer is security-gated and will NOT run on fork PRs. Use /ai-review-local instead.`

Telling a fork contributor to wait for a review that cannot appear is worse than
telling them nothing.

## Error Handling

### Nothing to do (idempotent success)
A clean, fully-pushed branch that already has an open PR is not an error — report the
existing PR URL. submit-pr never dead-ends with "nothing to submit" on a committed
branch: if no PR exists yet it opens one (step 10), even when there is nothing to push.

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
- Uses the `gh` CLI for PR creation (requires `gh auth login`). The GitHub MCP
  server is NOT used anywhere in this repo — `gh` is the only supported path.
- Git push uses SSH or HTTPS based on remote URL configuration
- **Fork workflows supported**: If `upstream` remote exists, PRs target upstream with `<fork-owner>:<branch>` head reference
