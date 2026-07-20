---
description: Review a Claude Code plan file from a staff engineer perspective
argument-hint: "[--updated] [--pr <comment-url>] <path-to-plan-file>"
---

# Review Plan

Review a Claude Code plan file from a staff engineer perspective and provide structured feedback across 8 dimensions. Optionally, when given a `--pr <comment-url>`, also verify the plan covers all feedback items from a specific PR review comment (Dimension 9).

## Arguments

`$ARGUMENTS` may contain:
- **Plan file path** (required): Path to the plan file, e.g., `~/.claude/plans/dreamy-coalescing-brook.md`
- `--updated` (optional): Signal that the plan has been revised since a prior review. Forces a fresh full review and includes a delta assessment of what changed.
- `--pr <comment-url>` (optional): URL of the specific PR comment whose feedback
  the plan addresses. Accepts GitHub comment URLs in any of these formats:
    - `https://github.com/owner/repo/pull/123#issuecomment-456`
    - `https://github.com/owner/repo/pull/123#discussion_r789`
    - `https://github.com/owner/repo/pull/123#pullrequestreview-012`
  Enables branch verification and PR feedback coverage checking (Dimension 9).

Parse `$ARGUMENTS` to extract:
- **--updated**: Split `$ARGUMENTS` on whitespace and check if any token is exactly `--updated`. Remove that token to get the remaining text.
- **--pr**: Check if any token is exactly `--pr`. If found, take the next token as the comment URL and remove both tokens. If `--pr` is found with no following URL, use AskUserQuestion to request it:
```
What is the PR comment URL to check coverage against?

Supported formats:
1. PR-level comment: https://github.com/owner/repo/pull/42#issuecomment-123456
2. Inline review comment: https://github.com/owner/repo/pull/42#discussion_r789012
3. Full PR review: https://github.com/owner/repo/pull/42#pullrequestreview-345678
```
- **Plan file path**: The remaining non-flag tokens after removing `--updated` and `--pr <url>`, joined back together. All flags (`--updated`, `--pr <url>`) are position-independent relative to the path and to each other.
- If no path remains after stripping flags, use AskUserQuestion to request it:
```
Which plan file would you like me to review?

Options:
1. Enter the path (e.g., ~/.claude/plans/plan-name.md)
```

## Constraints

- **Read-only for project files**: Do NOT create, edit, or delete any project files (source code, tests, documentation, configuration). The only PERSISTENT output is the review file (the helper-derived `review_path` in `~/.claude/plans/` — its filename carries a canonical-path digest). The workflow also uses invocation-scoped TEMPORARY files, which are allowed: the ingress/scratch files under `$(git rev-parse --git-path plan-review)` (inside `.git/`, never project content) and the helper-managed snapshot/state/meta/body files under `~/.claude/plans/.snapshots/` (the helper deletes them on persist and on abort).
- **Advisory-only**: Provide feedback and recommendations. Do not implement fixes.
- **No code changes**: Do not modify any source code, test files, or documentation.
- Use the Read tool for files and the Glob/Grep tools for searching. Do not use Edit, NotebookEdit, or file-modifying Bash commands on project files. The Write tool and `mkdir -p` may target only `~/.claude/plans/` and the two temporary locations above.
- The `gh api` calls used with `--pr` are read-only API requests, consistent with the project-files read-only constraint.

## Instructions

### Step 1: Take an Immutable Snapshot of the Plan

**The review must certify exactly the bytes it examined.** The whole
snapshot/verify/persist protocol lives in the tested helper
`.claude/scripts/plan_snapshot.py` (see `tests/test_plan_snapshot.py`) — the
raw plan path is UNTRUSTED and never touches a shell; it reaches the helper
via a file written with the Write tool:

1. Derive the scratch dir (one Bash call; deterministic, worktree-correct):
   ```bash
   SCRATCH="$(git rev-parse --git-path plan-review)"
   mkdir -p "$SCRATCH" && echo "$SCRATCH"
   ```
2. **Write the raw plan path** (exactly as supplied, `~` and all) to
   `<scratch>/plan-path.txt` with the Write tool — never `echo`/heredoc.
3. **Run the helper** (re-derive `SCRATCH` in this call):
   ```bash
   SCRATCH="$(git rev-parse --git-path plan-review)"
   python3 .claude/scripts/plan_snapshot.py snapshot --plan-path-file "$SCRATCH/plan-path.txt"
   ```
   It normalizes the path as data (`~` expansion, canonical realpath — any
   absolute path is accepted; the path never touches a shell), reads the plan
   bytes ONCE,
   writes an invocation-unique immutable snapshot + state token, and prints
   JSON: `state_path`, `snapshot_path`, `meta_path`, `body_path`, `plan_path`,
   `plan_sha256`, `review_path`. Non-zero exit = invalid/unreadable path —
   report its message and stop. **Confirm the printed `plan_path` is the plan
   you supplied** (a concurrent session overwriting the ingress file is
   thereby detected — if it differs, re-run from step 2).
4. **Read the SNAPSHOT file** (Read tool, the printed `snapshot_path`) — it is
   the ONLY text this review examines. The state token keys the rest of the
   protocol; Step 6's persist certifies the RECORDED snapshot digest only
   after re-verifying the live plan against it. **If the review aborts for any
   reason before persisting** (error, user cancellation), clean up the
   invocation: `python3 .claude/scripts/plan_snapshot.py abort --state-file
   "<state-path>"` — temporary snapshot files must not accumulate.

### Step 1b: Handle Re-Review (if `--updated`)

If the `--updated` flag is present, this is a re-review of a revised plan.

**You MUST perform a complete fresh review** — do not skip or abbreviate any steps. Treat the plan file contents as the authoritative source, not your memory of a prior version.

After completing the standard 8-dimension review in Step 4, add a **Delta Assessment** section to the output (see Step 5 template for format). This section compares the revised plan against the prior review's feedback:
- Which previously-raised issues have been addressed?
- Which previously-raised issues remain unresolved?
- Are there any new issues introduced by the revisions?

Additionally, check for a prior review via the Step 1 snapshot output: its `review_path` is the canonical location (review filenames carry a canonical-path digest — never derive them from the basename). If a file exists there, read it as a supplementary source of prior review context. When conversation context has been compressed between rounds, use the review file's content for delta assessment instead. If both conversation context and the review file are available, prefer whichever source is more detailed.

If no prior review is available from either source (conversation context or review file), still include the Delta Assessment section but fill each subsection with: "Delta assessment unavailable — no prior review found in conversation context or review file. Full fresh review performed."

### Step 2: Read CLAUDE.md for Project Context

Read the project's `CLAUDE.md` file to understand:
- Key design patterns (sklearn-like API, formula interface, results objects, etc.)
- Estimator inheritance map
- Testing conventions
- Key reference file pointers (methodology registry, etc.)

Also read `CONTRIBUTING.md` for documentation requirements, test writing
guidelines, and implementation guidelines.

If the plan modifies estimator math, standard error formulas, inference logic, or edge-case handling, also read `docs/methodology/REGISTRY.md` to understand the academic foundations and reference implementations for the affected estimator(s).

### Step 2b: Parse Comment URL and Verify Branch (if `--pr`)

Only perform this step when `--pr <comment-url>` was provided. Otherwise skip to Step 3.

**Parse the URL:**
- Strip query parameters from the URL before parsing: remove the query string (the `?...` portion) while preserving the `#` fragment. For example, `https://github.com/o/r/pull/1?notification_referrer_id=abc#issuecomment-123` becomes `https://github.com/o/r/pull/1#issuecomment-123`. If the fragment itself contains `?` (e.g., `#discussion_r123?foo=bar`), strip the `?` and everything after it from the fragment before pattern matching, since GitHub fragments never contain `?` as meaningful data.
- Only `github.com` URLs are supported. If the URL host is not `github.com`, report an error and stop.
- Extract `owner`, `repo`, `pr_number` from the URL path. The `pr_number` is always the path segment immediately after `/pull/`.
- Extract comment type and ID from the fragment:

| Fragment | Type | `gh api` endpoint |
|---|---|---|
| `#issuecomment-{id}` | Issue comment | `repos/{owner}/{repo}/issues/comments/{id}` |
| `#discussion_r{id}` | Inline review comment | `repos/{owner}/{repo}/pulls/comments/{id}` |
| `#pullrequestreview-{id}` | PR review | `repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id}` |

If the URL doesn't match any fragment pattern (including bare PR URLs without a fragment), report:
```
Error: Unrecognized PR comment URL format. Expected a GitHub PR comment URL like:
  https://github.com/owner/repo/pull/123#issuecomment-456
The URL must point to a specific comment, not a PR page.
```

**Verify `gh` CLI availability:**

Run `gh auth status 2>/dev/null` (suppress output on success). If it fails, report a hard error:
```
Error: The --pr flag requires the GitHub CLI (gh) to be installed and authenticated.
Run `gh auth login` to authenticate, then retry.
```

**Verify branch state:**

```bash
gh pr view <number> --repo <owner>/<repo> --json headRefName,baseRefName,title --jq '.'
```

Compare `headRefName` against `git branch --show-current`:
- **Match**: Note "Branch verified" in output.
- **Mismatch**: Emit a warning in the output and note under Dimension 2 (Codebase Correctness) that code references may be inaccurate. Recommend the user checkout the PR branch first (`git checkout <headRefName>`), but do not block the review.

### Step 2c: Fetch the Specific Comment (if `--pr`)

Only perform this step when `--pr` was provided. Otherwise skip to Step 3.

Fetch the comment using the `gh api` endpoint from the table in Step 2b.

**For `pullrequestreview-` URLs**, fetch BOTH the review body AND its inline comments:
```bash
# Review body
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id} --jq '{body: .body, user: .user.login, state: .state}'

# All inline comments belonging to this review
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id}/comments --paginate --jq '.[] | {body: .body, path: .path, line: .line, diff_hunk: .diff_hunk}'
```

**For other comment types**, fetch the single comment:

**Issue comment:**
```bash
gh api repos/{owner}/{repo}/issues/comments/{id} --jq '{body: .body, user: .user.login, created_at: .created_at}'
```

**Inline review comment:**
```bash
gh api repos/{owner}/{repo}/pulls/comments/{id} --jq '{body: .body, user: .user.login, path: .path, line: .line, diff_hunk: .diff_hunk}'
```

**Error handling:**
- **404**: `Error: Comment not found at <url>. It may have been deleted or the URL may be incorrect.`
- **403 / other API errors**: `Error: GitHub API returned <status>. You may not have access to this repository, or you may be rate-limited. Check 'gh auth status' and try again.`
- **Empty comment body** (and no inline comments for review types): report and skip Dimension 9:
  ```
  Note: No feedback text found in the comment at <url>.
  Skipping PR Feedback Coverage (Dimension 9). Reviewing plan without PR context.
  ```

The response includes: `body` (comment text), `user.login` (author), `created_at`, and for inline comments: `path` (file), `line` (line number in the file — use `line`, not `position` which is the diff offset, and not `original_line` which is the base branch line), `diff_hunk` (surrounding diff context).

**Extract discrete feedback items** from the comment body:
- For AI review comments (structured markdown with P0/P1/P2/P3 or Critical/Medium/Minor sections): parse each severity section and extract individual items with their labeled severity
- For human comments with numbered/bulleted lists: each list item is one feedback item
- For human comments that are a single paragraph or conversational: the entire comment is one feedback item
- For inline review comments: each comment is one item, with `path` and `line` as its file/line reference
- **Default severity**: when a feedback item has no severity label, treat it as Medium
- Process all feedback items regardless of count

Each feedback item tracks: severity (labeled or default Medium), description, file path (if available), and line reference (if available).

### Step 3: Read Referenced Files

Identify all files the plan references (file paths, module names, class names). When `--pr` was provided, also include files referenced in the feedback comment — inline comment `path` fields and file paths mentioned in the comment body (e.g., `path/to/file.py:L123`). Then read them to validate the plan's assumptions:

**Priority order for reading files:**
1. **Files the plan proposes to modify** — read ALL of these first
2. **Files referenced for context** (imports, call sites, existing patterns) — read selectively to verify specific claims

**Scope restriction:**
- Only read files that are within the project repository (the working directory tree).
- The plan file itself (the `$ARGUMENTS` input) is exempt — it can be anywhere (e.g., `~/.claude/plans/`).
- If the plan references paths outside the repo (home directory configs, SSH keys, `/etc/` files, etc.), do NOT read them. Instead, note in the review output under Dimension 2 (Codebase Correctness) that those external paths were not verified.

**What to verify:**
- File paths exist
- Class names, function signatures, and method names match what the plan describes
- Line numbers (if referenced) are accurate
- The plan's description of existing code matches reality

If the plan references more than ~15 files, use judgment: read all files slated for modification, then spot-check context files as needed rather than reading every one.

### Step 4: Evaluate Across 8 Dimensions

#### Dimension 1: Completeness & Executability

Could a fresh Claude Code session — with no access to the conversation history that produced this plan — execute it without asking clarifying questions?

Check for:
- Are all file paths explicit? (No "the relevant file" or "the test file")
- Are code changes described concretely? (Function signatures, parameter names, not just "add a method")
- Are decision points resolved, not deferred? ("We'll figure out the API later" is a red flag)
- Are there implicit assumptions that require conversation context to understand?

#### Dimension 2: Codebase Correctness

Do file paths, class names, function signatures, and line-number references in the plan match the actual codebase?

Use your findings from Step 3. Flag:
- File paths that don't exist
- Function/class names that are misspelled or don't exist
- Line numbers that point to the wrong code
- Descriptions of existing behavior that don't match reality

#### Dimension 3: Scope

Is the scope right — not too much, not too little?

Check for **missing related changes**:
- Tests for new/changed functionality
- `__init__.py` export updates
- `get_params()` / `set_params()` updates for new parameters
- Documentation updates (`diff_diff/guides/llms.txt` for new public-API surfaces, `docs/api/*.rst`, `docs/references.rst` for new citations, tutorials, CONTRIBUTING.md, CLAUDE.md if design patterns change). README updates only if the change affects the landing page (new estimator catalog one-liner, hero/badges/tagline, top-level capability paragraph) - per CONTRIBUTING.md, README is not the place for usage examples or per-estimator sections.
- For bug fixes: did the plan grep for ALL occurrences of the pattern, or just the one reported?

Check for **unnecessary additions**:
- Docstrings/comments/type annotations for untouched code
- Premature abstractions or over-engineering
- Feature flags or backward-compatibility shims when the code can just be changed

#### Dimension 4: Edge Cases & Failure Modes

For methodology-critical code:
- NaN propagation through ALL inference fields (SE, t-stat, p-value, CI)
- Empty inputs / empty result sets
- Boundary conditions (single observation, single group, etc.)
- **Registry cross-check** (for plans modifying estimator math/SE/inference):
  - Read the relevant estimator section in `docs/methodology/REGISTRY.md`
  - For each equation the plan implements: verify it matches the Registry, or the plan documents the deviation
  - For each edge case in the Registry's "Edge cases" section: verify the plan handles it or explicitly defers it
  - CRITICAL if plan contradicts a Registry equation without documented deviation
  - MEDIUM if plan doesn't handle a documented Registry edge case
  - LOW if plan adds new edge case handling not yet in Registry (suggest updating it)

For all code:
- Error handling paths — are they tested with behavioral assertions (not just "runs without exception")?
- What happens when the feature interacts with other parameters/modes?

#### Dimension 5: Architecture & Patterns

Check against CLAUDE.md conventions:
- Does it respect the estimator inheritance map? (Adding a param to `DifferenceInDifferences` auto-propagates to `TwoWayFixedEffects` and `MultiPeriodDiD`; standalone estimators need individual updates)
- Does it use `linalg.py` for OLS/variance instead of reimplementing?
- Does it follow the sklearn-like `fit()` / results-object pattern?
- Is there a simpler alternative that avoids new abstraction?
- Does it match existing code patterns in the codebase?

#### Dimension 6: Plan Execution Risks

Plan-specific failure modes that wouldn't show up in a code review:

- **Ordering issues**: Does the plan propose changes in an order that would break things mid-implementation? (e.g., changing an import before the module it imports from exists, deleting a function before updating its callers)
- **Ambiguous decision points**: Does the plan defer decisions that should be made now? Vague phrases like "choose an appropriate approach" or "handle edge cases" without specifying which ones
- **Missing rollback path**: For risky changes (public API modifications, data format changes), does the plan consider what happens if something goes wrong?
- **Implicit dependencies**: Does step N assume step M was completed, but this ordering isn't stated?

#### Dimension 7: Backward Compatibility & API Risk

- Does the plan add, remove, or rename public API surface (parameters, methods, classes)?
- If so, does it acknowledge the breaking change and state the versioning decision (deprecation period vs clean removal)?
- Downstream effects on:
  - Convenience functions
  - Re-exports in `__init__.py`
  - Existing tutorials and documentation
  - User code that may depend on the current API

#### Dimension 8: Testing Strategy

- Are tests included in the plan? Do they cover the happy path AND the edge cases from Dimension 4?
- Are test assertions behavioral (checking outcomes) rather than just "runs without exception"?
- For bug fixes: does the plan fix all pattern instances and test all of them?
- Are there missing test scenarios? (Parameter interactions, error paths, boundary conditions)

#### Dimension 9: PR Feedback Coverage (only if `--pr` provided with non-empty comment)

Only evaluate this dimension when `--pr` was provided and a non-empty comment was fetched in Step 2c. For each feedback item extracted in Step 2c, assess:

- **Addressed**: Plan explicitly mentions the issue AND proposes a concrete fix
- **Partially addressed**: Plan touches the area but doesn't fully resolve the feedback
- **Not addressed**: Plan makes no mention of this feedback item
- **Dismissed with justification**: Plan acknowledges the feedback but explains why it won't be acted on (acceptable for Low/P3; flag for Critical/P0)

Use judgment, not just substring matching — the plan may use different words to describe the same fix.

**Assessment impact:**
- Unaddressed P0/P1/Critical items -> results in "Significant issues found"
- Unaddressed P2/Medium items count as Medium issues
- Unaddressed P3/Low items count as Low issues

### Step 4b: Display Plan Content

Before presenting the review, display the full plan content so the user can cross-reference the review findings against what was actually written:

```
## Plan Content: <plan-filename>

<full plan file content>

---
```

This ensures the user can read the plan immediately before reading the review findings. Display the full plan content as-is from the file.

Note: The plan content is displayed in the terminal only — it is NOT included in the `.review.md` file (Step 6), which contains only the review output. The plan is already persisted as its own file.

### Step 5: Present Structured Feedback

Present the review in the following format. Number each issue sequentially within its severity section (e.g., CRITICAL #1, CRITICAL #2, MEDIUM #1) to enable cross-referencing with `/revise-plan`. Do NOT skip any section — if a section has no findings, write "None." for that section. The Delta Assessment section is only included when the `--updated` flag was provided (see Step 1b). The PR Context and PR Feedback Coverage sections are only included when `--pr` was provided with a non-empty comment.

```
## Overall Assessment

[2-3 sentences: what the plan does, the reviewer's key observations, and the biggest concern if any]

---

## PR Context (only include if `--pr` was provided with non-empty comment)

**PR**: #<number> - <title> (<owner>/<repo>)
**Branch**: <headRefName> -> <baseRefName>
**Comment**: <comment-url>
**Comment author**: <user.login>
**Feedback items extracted**: N
**Branch match**: Yes / No (warning: recommend `git checkout <headRefName>`)

---

## Issues

### CRITICAL
[Issues that would cause implementation failure, incorrect results, or breaking changes if not addressed. Each issue should include: file path and/or line number if applicable, what's wrong, and a concrete suggestion for fixing it.]

### MEDIUM
[Issues that should be addressed but won't block implementation. Missing test cases, incomplete documentation updates, scope gaps.]

### LOW
[Minor suggestions. Style consistency, optional improvements, things to consider.]

---

## Convention Gaps

Cross-reference against `CLAUDE.md` and `CONTRIBUTING.md`. List project conventions
the plan does not account for.

[Draw on the conventions that actually apply to this plan — e.g. the estimator
inheritance map and `get_params`/`set_params` propagation for a new parameter;
`safe_inference()` for inference fields; the deviation-labelling rules for
methodology changes; grep-all-sites-then-fix-in-one-PR for pattern bugs; the
documentation surfaces in "README discipline". List only what the plan misses.]

**Registry Alignment** (if methodology files changed):
- [ ] Plan equations match REGISTRY.md (or deviations documented)
- [ ] All Registry edge cases handled or explicitly out-of-scope
- [ ] REGISTRY.md updated if new edge cases discovered

---

## PR Feedback Coverage (only include if `--pr` was provided with non-empty comment)

### Addressed
- [severity] <description> -- Plan step: <reference to plan section>

### Partially Addressed
- [severity] <description> -- Gap: <what's missing>

### Not Addressed
- [severity] <description>

### Dismissed
- [severity] <description> -- Plan's reason: "<quote>"

| Status | Count |
|--------|-------|
| Addressed | N |
| Partially addressed | N |
| Not addressed | N |
| Dismissed | N |

---

## Questions for the Author

[Ambiguities or missing information that should be clarified before implementation begins. Phrase as specific questions.]

---

## Delta Assessment (only include if `--updated` flag was provided)

### Addressed
[List prior issues that have been resolved in the revised plan]

### Unresolved
[List prior issues that remain. Include the original issue text for reference.]

### New Issues
[List any new issues introduced by the revisions, or "None."]

### PR Feedback Coverage Delta (only include if both `--updated` and `--pr` were provided)

The `--pr` URL must be the same across the initial review and the `--updated` re-review — this compares coverage of the same feedback comment. If the prior review's PR comment URL is no longer available in conversation context (e.g., context compressed), note: "PR coverage delta unavailable — prior PR context not found."

- **Newly addressed**: [list of feedback items now covered that were previously not addressed or partially addressed]
- **Still not addressed**: [list of feedback items still missing]

---

## Summary

| Category | Issues |
|----------|--------|
| Critical | [count] |
| Medium | [count] |
| Low | [count] |
| Checklist gaps | [count] |
| PR feedback gaps | [count of Not Addressed + Partially Addressed] (only if `--pr`) |
| Questions | [count] |

**Assessment**: [No critical issues found / Minor issues to address / Significant issues found]

- **No critical issues found**: No critical issues, few or no medium issues
- **Minor issues to address**: No critical issues, some medium issues that are straightforward to address
- **Significant issues found**: Has critical issues or many medium issues that require rethinking the approach
```

### Step 6: Save Review to File

After displaying the review in the conversation (Step 5), persist it via the
helper — it re-verifies the live plan against the reviewed snapshot, builds the
frontmatter (setting `plan:` and `plan_sha256:` itself from the snapshot — the
caller cannot mis-stamp them), writes atomically, and cleans the snapshot up:

1. **Write the review body** (everything from "## Overall Assessment" through
   "## Summary", exactly as displayed) to the exact `body_path` printed in
   Step 1, with the Write tool (invocation-unique — concurrent reviews cannot
   cross-wire inputs).

2. **Write the meta JSON** to the exact `meta_path` printed in Step 1, with
   the Write tool:
   ```json
   {"reviewed_at": "2026-02-15T14:30:00Z", "assessment": "Significant issues found",
    "critical_count": 2, "medium_count": 3, "low_count": 1, "flags": ["--updated", "--pr"]}
   ```
   (`reviewed_at` from `date -u +%Y-%m-%dT%H:%M:%SZ`; `flags` lists the CLI
   flags active during this review — `"--updated"`, `"--pr"`, or `[]`.)

3. **Persist via the state token** — substitute the literal `state_path`
   printed in Step 1 (helper-generated, safe charset):
   ```bash
   python3 .claude/scripts/plan_snapshot.py persist --state-file "<state-path>"
   ```
   - **Exit 3** means the plan was modified during the review: the review was
     NOT persisted (it examined the snapshot, and the live content is now
     something else). Relay the helper's message and stop — re-run
     /review-plan against the current plan. Do NOT proceed to the footer.
   - Any other non-zero exit: report the message and stop (the review file is
     required by the ExitPlanMode hook; a missing one blocks approval).
   - On success it prints the review path (canonical — derived from the
     realpath the hook also uses, so symlink aliases cannot split the key).

4. **Append a footer** to the conversation output:
   ```
   ---
   Review saved to: <review-file-path>
   Tip: In the planning window, the review will be read automatically before plan approval.
   ```

## Notes

- This skill is read-only for project files — its one persistent output is the review file at the helper-derived `review_path` (canonical basename + canonical-path digest, in `~/.claude/plans/`), whose `plan_sha256` frontmatter is what the ExitPlanMode hook validates against the plan's current content
- Plan files are typically located in `~/.claude/plans/`
- The review is displayed in the conversation (primary reading surface) and saved to a `.review.md` file alongside the plan (for persistence and cross-session exchange)
- On `--updated` re-reviews, the prior `.review.md` file is read for delta context and then overwritten with the new review
- Pairs with the in-plan-mode review workflow (CLAUDE.md) for in-session review
- For best results, run this before implementing a plan to catch issues early
- The 8 dimensions are tuned for plan-specific failure modes, not generic code review
- Use `--updated` when re-reviewing a revised plan to get a delta assessment of what changed since the prior review
- Use `--pr <comment-url>` when the plan addresses a specific PR review comment.
  This fetches the comment, extracts feedback items, and checks that the plan
  covers each one. Pairs naturally with `/read-feedback-revise` which creates the plan.
- The `--pr` flag requires the `gh` CLI to be installed and authenticated.
- For best results, run this while on the PR branch so file contents and line
  numbers match what reviewers commented on.
- The comment URL can be copied from the GitHub web UI by right-clicking the
  timestamp on any PR comment and selecting "Copy link".
- For `pullrequestreview-` URLs, both the review body and its inline comments
  are fetched (matching `/read-feedback-revise` behavior).
