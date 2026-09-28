---
description: Create a new git worktree with full dev environment for parallel work
argument-hint: "<name> [base-branch]"
---

# Create Git Worktree

Create an isolated worktree for parallel development. Arguments: $ARGUMENTS

## Instructions

### 1. Parse Arguments

Parse `$ARGUMENTS` to extract:
- **name** (required): First argument — used as both directory suffix and branch name
- **base-ref** (optional): Second argument — existing branch, tag, or ref to branch
  from (creates branch `<name>` starting at that ref)

If no name is provided, abort with:
```
Error: Name required. Usage: /worktree-new <name> [base-branch]
Example: /worktree-new feature-bacon-fix
```

Validate that **name** starts with a letter or digit, followed by `[a-zA-Z0-9._-]`.
If it starts with `-` or contains spaces, slashes, or other shell metacharacters, abort:
```
Error: Name must start with a letter or digit and contain only letters, digits, dots, hyphens, and underscores.
Got: <name>
```

If **base-ref** is provided, apply the same character validation (must match
`^[a-zA-Z0-9][a-zA-Z0-9._/-]*$` — slashes are allowed for refs like `origin/main`).
Then verify the ref exists:

```bash
git rev-parse --verify --quiet "$BASE_REF"
```

If verification fails, abort:
```
Error: Ref not found: <base-ref>
Available branches:
<output of: git branch -a --format='%(refname:short)'>
```

### 2. Resolve Paths

Derive paths dynamically (do NOT hardcode the repo name):

```bash
MAIN_ROOT="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"
REPO_NAME="$(basename "$MAIN_ROOT")"
PARENT_DIR="$(dirname "$MAIN_ROOT")"
WORKTREE_PATH="${PARENT_DIR}/${REPO_NAME}-<name>"
```

Use `$WORKTREE_PATH` (the absolute path) for all subsequent commands.

### 3. Validate

```bash
git worktree list
```

- If a worktree already exists at `$WORKTREE_PATH`, abort with an error.
- If a branch named `<name>` already exists and no base-ref was given:
  - First check if the branch is already checked out in a worktree
    (parse `git worktree list --porcelain` for a `branch refs/heads/<name>` line).
  - If checked out elsewhere, abort:
    ```
    Error: Branch '<name>' is already checked out in worktree at <path>.
    Use a different name or remove that worktree first.
    ```
  - Otherwise, ask the user whether to check out that existing branch
    or pick a different name. If the user chooses to use it:
    ```bash
    git worktree add -- "$WORKTREE_PATH" "<name>"
    ```
    Then skip step 4 and continue to step 5.

### 4. Create the Worktree

**If base-ref was provided**, use it directly — the user stated their intent:

```bash
git worktree add -b "<name>" -- "$WORKTREE_PATH" "$BASE_REF"
```

**If no base-ref was provided**, do NOT silently branch from wherever HEAD happens
to be. Resolve what the base would actually be first:

```bash
CURRENT_REF="$(git rev-parse --abbrev-ref HEAD)"
# Base remote: upstream when this is a fork checkout, else origin — same resolution
# /submit-pr uses. In the common (no-upstream) case this is just origin.
BASE_REMOTE="$(git remote get-url upstream >/dev/null 2>&1 && echo upstream || echo origin)"
# Resolve the remote's default branch from the remote itself. Do NOT use
# `gh repo view "$BASE_REMOTE"` — gh treats the argument as an owner/repo, not a
# remote alias, so it errors ("Could not resolve to a Repository with the name
# '<you>/origin'") and silently falls back to `main`, breaking any repo whose
# default is `master`/`develop`.
DEFAULT_BRANCH="$(git ls-remote --symref "$BASE_REMOTE" HEAD 2>/dev/null \
  | awk '/^ref:/ {sub("refs/heads/","",$2); print $2; exit}')"

# Offline / remote unavailable? Fall back to the LOCAL record of the remote's HEAD
# rather than inventing `main` — assuming `main` on a `develop`/`master` repo would
# branch from the wrong ancestry.
if [ -z "$DEFAULT_BRANCH" ]; then
  DEFAULT_BRANCH="$(git symbolic-ref --quiet --short "refs/remotes/$BASE_REMOTE/HEAD" 2>/dev/null \
    | sed "s#^$BASE_REMOTE/##")"
fi
```

If `$DEFAULT_BRANCH` is still empty, **do not guess `main`** — ask the user for the
base branch (or to pass an explicit base-ref) and stop. A wrong default silently
poisons every worktree spun off it.

- If `$CURRENT_REF` equals `$DEFAULT_BRANCH`, check the base is actually current
  before branching from it. A worktree spun off a stale local default branch means a
  rebase later and phantom review findings in between:

  Measure **both** directions. Behind-only is the obvious case, but ahead matters
  just as much: local commits sitting unpushed on the default branch would be
  inherited by the new worktree and end up contaminating an unrelated PR.

  Distinguish "verified in sync" from "could not verify" — a failed fetch or a missing
  tracking ref must NOT masquerade as "in sync", or a stale/divergent base slips
  through silently:

  ```bash
  FRESH_OK=1
  git fetch "$BASE_REMOTE" "$DEFAULT_BRANCH" --quiet 2>/dev/null || FRESH_OK=0
  # Track ref EXISTENCE separately from freshness: a stale-but-present ref can be
  # offered as a fallback; an absent ref cannot (branching from it just fails).
  REF_EXISTS=1
  git rev-parse --verify --quiet "$BASE_REMOTE/$DEFAULT_BRANCH" >/dev/null || { REF_EXISTS=0; FRESH_OK=0; }
  # A failed rev-list must lower FRESH_OK — NOT silently become 0 ahead/behind (which
  # would read as "verified in sync"). Propagate the failure, then default the value.
  BEHIND=$(git rev-list --count "HEAD..$BASE_REMOTE/$DEFAULT_BRANCH" 2>/dev/null) || FRESH_OK=0
  AHEAD=$(git rev-list --count "$BASE_REMOTE/$DEFAULT_BRANCH..HEAD" 2>/dev/null) || FRESH_OK=0
  : "${BEHIND:=0}"; : "${AHEAD:=0}"
  ```

  - **`FRESH_OK=1` and `BEHIND == 0` and `AHEAD == 0`** — verified in sync:
    **proceed silently, no prompt.** This is the common path and must stay frictionless.
    ```bash
    git worktree add -b "<name>" -- "$WORKTREE_PATH"
    ```

  - **`FRESH_OK=0`** — freshness could not be verified (offline, fetch failed, or the
    tracking ref is absent). Do not assume sync. Surface it — and **only offer the
    stale-tracking-ref option when `REF_EXISTS=1`**; when the ref is absent that option
    would just fail:
    ```
    Could not verify <default-branch> against <base-remote>.
    Branching from local <default-branch> may inherit stale or local-only commits.

    Options:
    1. Branch from <base-remote>/<default-branch> anyway (may be stale) — ONLY if REF_EXISTS=1
    2. Branch from local <default-branch>
    3. Abort - I'll fetch first
    ```

  - **Otherwise** — verified but behind, ahead, or diverged — surface it before
    creating anything. State whichever applies:
    ```
    Local <default-branch> and <base-remote>/<default-branch> differ:
      behind: <BEHIND> commit(s)   ahead: <AHEAD> commit(s)

    Behind means you will rebase later. Ahead means those local-only commits
    become part of the new branch.

    Options:
    1. Branch from <base-remote>/<default-branch> (recommended)
    2. Branch from local <default-branch> anyway
    3. Abort
    ```
    Option 1 uses `git worktree add -b "<name>" -- "$WORKTREE_PATH" "$BASE_REMOTE/$DEFAULT_BRANCH"`;
    option 2 uses the plain form above.

  Never fetch-and-reset the user's checked-out default branch — this reads
  `$BASE_REMOTE/*` and branches from it. The branch they are standing on is left alone.

- If `$CURRENT_REF` is anything else, the new branch would inherit that branch's
  commits. That is occasionally intended (stacking work) and usually not. **Surface it
  with AskUserQuestion before creating anything:**

  ```
  No base branch given, and HEAD is on '<current-ref>', not '<default-branch>'.
  A new worktree created now would inherit every commit on '<current-ref>'.

  Options:
  1. Branch from <base-remote>/<default-branch> (recommended)
  2. Branch from current HEAD (<current-ref>) - stack on this work
  3. Abort - let me specify a base
  ```

  For option 1, the worktree add must be **conditional on the fetch succeeding** —
  otherwise a failed fetch silently branches from a stale `$BASE_REMOTE/$DEFAULT_BRANCH`,
  breaking the promise that the base is current:
  ```bash
  if git fetch "$BASE_REMOTE" "$DEFAULT_BRANCH" --quiet; then
    git worktree add -b "<name>" -- "$WORKTREE_PATH" "$BASE_REMOTE/$DEFAULT_BRANCH"
  else
    echo "Fetch of $BASE_REMOTE/$DEFAULT_BRANCH failed; its tracking ref may be stale."
    # Ask: branch from the (possibly stale) tracking ref, branch from current HEAD,
    # or abort and fetch manually. Do NOT create the worktree on a stale base silently.
  fi
  ```
  For option 2, use the no-base-ref form above. For option 3, stop.

### 5. Set Up Python Environment

Note to user: dependency installation may take a moment on a fresh venv.

```bash
python3 -m venv "$WORKTREE_PATH/.venv"
"$WORKTREE_PATH/.venv/bin/pip" install --upgrade pip
"$WORKTREE_PATH/.venv/bin/pip" install -e "$WORKTREE_PATH[dev]"
```

Do NOT use `-q` — let pip output stream so the user sees progress.

### 6. Verify Rust Backend Rebuild Capability (best-effort)

Step 5 already compiled the Rust extension via maturin (the build backend).
This step verifies maturin can rebuild into **this worktree's** venv after future
Rust changes.

**`maturin develop` installs into the venv it detects from `VIRTUAL_ENV` / the
current directory — NOT the venv implied by `--manifest-path`.** Passing only
`--manifest-path` from another directory builds this worktree's Rust code into
whatever venv happens to be active, silently. Pin both:

```bash
(cd "$WORKTREE_PATH" && VIRTUAL_ENV="$WORKTREE_PATH/.venv" \
  "$WORKTREE_PATH/.venv/bin/maturin" develop --release --features accelerate)
```

`--release --features accelerate` matches the macOS build used elsewhere in this
project (Apple Accelerate BLAS). On Linux substitute `--features openblas`; on
Windows drop `--features` entirely.

If this step fails, the Rust extension was likely still built successfully by pip
in step 5. Report it and continue — this is not an error.

### 7. Report

Print:

```
Worktree ready: $WORKTREE_PATH
Branch: <branch>

To start working:
  cd $WORKTREE_PATH && source .venv/bin/activate && claude
```
