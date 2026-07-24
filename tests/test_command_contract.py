"""Contract tests over the /submit-pr and /push-pr-update command markdown.

The safety of these commands lives partly in prose (the model follows the steps), so
`pr_prepare.py` and `git commit --file` being safe in isolation is not enough — the
commands must actually use them. These lightweight checks read the command files and
fail if a heredoc-based `git commit` is reintroduced, raw values are interpolated, or a
base-referencing git command runs before the base is validated. They scan **fenced
`bash` blocks only**, so prose references to an anti-pattern (in backticks, mid-
sentence) are never mistaken for executable lines.
"""

import pathlib
import re

import pytest

_COMMANDS = pathlib.Path(__file__).resolve().parent.parent / ".claude" / "commands"
_FILES = ["submit-pr.md", "push-pr-update.md"]


def _bash_block_lines(text):
    """Yield (line_number, line) for every line inside a ```bash fenced block.

    Line numbers are 1-indexed, matching editor display and the other checks here.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"^\s*```bash\b", lines[i]):
            j = i + 1
            while j < len(lines) and not re.match(r"^\s*```\s*$", lines[j]):
                yield j + 1, lines[j]
                j += 1
            i = j + 1
        else:
            i += 1


# A real heredoc commit is a *command line* beginning with `git commit -m "$(cat`.
_HEREDOC_COMMIT = re.compile(r"^\s*git commit -m \"\$\(cat")
# A git command performing a `..` range (needs a base ref) — matched anywhere on the
# line, so it also catches `TITLE="$(git log "$BASE"..HEAD)"`.
_BASE_RANGE_CMD = re.compile(r"git (?:log|diff|rev-list)\b[^\n]*\.\.")
# The actual helper invocation (not a prose mention).
_PREPARE_INVOKE = re.compile(r"^\s*python3 \.claude/scripts/pr_prepare\.py\b")


def _read(name):
    if not _COMMANDS.is_dir():
        # No command dir at all → installed distribution without repo tooling.
        pytest.skip("command dir not present (installed distribution)")
    path = _COMMANDS / name
    # In a real checkout a missing hardened command is a lost safety boundary, not a
    # reason to skip — fail loudly so deleting the file cannot leave the suite green.
    assert path.exists(), f"{name} is missing from a repo checkout — safety contract lost"
    return path.read_text()


_REQUIRED_COMMANDS = ["submit-pr.md", "push-pr-update.md", "worktree-new.md", "worktree-rm.md"]


@pytest.mark.parametrize("name", _REQUIRED_COMMANDS)
def test_required_command_present_in_checkout(name):
    if not _COMMANDS.is_dir():
        pytest.skip("not a repo checkout")
    assert (_COMMANDS / name).exists(), f"{name} deleted from a checkout — contract lost"


@pytest.mark.parametrize("name", _FILES)
def test_no_active_heredoc_commit(name):
    offenders = [(n, ln) for n, ln in _bash_block_lines(_read(name)) if _HEREDOC_COMMIT.match(ln)]
    assert not offenders, f"{name} has heredoc `git commit -m` line(s): {offenders}"


@pytest.mark.parametrize("name", _FILES)
def test_uses_git_commit_file(name):
    assert "git commit --file" in _read(
        name
    ), f"{name} must commit via `git commit --file`, not a heredoc"


def test_submit_pr_has_no_raw_value_interpolation():
    """No raw title/branch/base pasted into a shell command; all flow through files."""
    text = _read("submit-pr.md")
    assert '--title "$RAW' not in text
    assert not re.search(r'--(branch|base) "\$RAW', text)


def test_submit_pr_has_zero_diff_guard():
    """submit-pr must guard against a clean zero-commits-ahead branch (else it creates
    and pushes a useless branch, then gh pr create fails on an empty diff). The idempotent
    reframe removed the blanket 'nothing to submit' exit; the guard must be scoped to
    'zero commits ahead of base'."""
    text = _read("submit-pr.md")
    assert re.search(
        r'rev-list --count "\$BASE_REMOTE/\$BASE_BRANCH\.\.HEAD"', text
    ), "submit-pr must compare HEAD to the base to detect a zero-diff branch"
    assert "Nothing to submit" in text, "submit-pr must still exit on a genuine zero-diff"


def test_submit_pr_idempotency_check_compares_base_and_draft():
    """The existing-PR idempotency check must not treat ANY open PR as success — it must
    compare baseRefName/isDraft. Assert the fields are on the *executable* `gh pr view`
    command line (its --json argument), not merely somewhere in the prose."""
    view_lines = [
        ln
        for _, ln in _bash_block_lines(_read("submit-pr.md"))
        if "gh pr view" in ln and "--json" in ln
    ]
    assert view_lines, "no executable `gh pr view --json` idempotency query found"
    assert any(
        "baseRefName" in ln and "isDraft" in ln for ln in view_lines
    ), "the gh pr view idempotency query must fetch baseRefName AND isDraft to compare them"


def test_submit_pr_no_base_range_command_before_validation():
    """The raw --base must not reach a shell before pr_prepare.py validates it. No git
    command performing a base `..` range may appear, in any bash block, before the
    actual helper invocation — this is the exact P0 class where title generation ran
    `git log <base>..HEAD` on the raw base. Scans fenced bash only (prose references are
    not matched) and locates the real `python3 …pr_prepare.py` line, not a prose
    mention of it."""
    text = _read("submit-pr.md")
    lines = text.splitlines()
    inv_line = next((n for n, ln in enumerate(lines, 1) if _PREPARE_INVOKE.match(ln)), None)
    assert inv_line is not None, "no actual `python3 …/pr_prepare.py` invocation found"
    offenders = [
        (n, ln) for n, ln in _bash_block_lines(text) if n < inv_line and _BASE_RANGE_CMD.search(ln)
    ]
    assert not offenders, f"base-range git command before validation: {offenders}"


# ---------------------------------------------------------------------------
# worktree-new safety guards. (worktree-rm's destructive-logic rewrite was reverted
# to its main version in this PR — see PR notes — so it has no guards to assert here.)
# ---------------------------------------------------------------------------

_WORKTREE_NEW_REQUIRED = [
    "ls-remote --symref",  # default branch from the remote, not `gh repo view <alias>`
    "FRESH_OK",  # unverified freshness is surfaced, not treated as in-sync
]


@pytest.mark.parametrize("pattern", _WORKTREE_NEW_REQUIRED)
def test_worktree_new_keeps_guard(pattern):
    assert pattern in _read("worktree-new.md"), f"worktree-new.md lost guard: {pattern!r}"


def test_worktree_new_no_gh_repo_view_alias():
    # `gh repo view "$BASE_REMOTE"` treats a remote alias as owner/repo; must not recur.
    # Skip `#` comment lines — the fix documents the anti-pattern in a bash comment.
    offenders = [
        (n, ln)
        for n, ln in _bash_block_lines(_read("worktree-new.md"))
        if not ln.lstrip().startswith("#") and re.search(r'gh repo view "\$BASE_REMOTE"', ln)
    ]
    assert not offenders, f"worktree-new uses gh repo view with a remote alias: {offenders}"


# A shell assignment that pastes a raw <placeholder> value into shell source — the
# class behind every injection finding this PR closed (title, base, filename stem).
# A git filename/title with `$()` or backticks executes at such an assignment.
_PLACEHOLDER_ASSIGN = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*="<[^>]+>"')

# The hardened workflow commands. (review-plan.md / revise-plan.md retired in the
# Step-3 skill migration — their plan-review contracts moved to
# tests/test_plan_review_skill.py, targeting .claude/skills/plan-review/SKILL.md.)
_HARDENED = [
    "submit-pr.md",
    "push-pr-update.md",
    "pre-merge-check.md",
    "worktree-new.md",
    "worktree-rm.md",
]


@pytest.mark.parametrize("name", _HARDENED)
def test_no_raw_placeholder_assignment(name):
    """Invariant: no command pastes a raw value into a shell assignment. Discovered
    filenames, titles, and bases must be consumed as data (files, NUL streams, or the
    model's reasoning), never `VAR="<placeholder>"` — a `$(...)`/backtick payload would
    execute at the assignment. (This catches the ASSIGNMENT form; pre-merge-check, which
    must pass filenames as command ARGUMENTS, is additionally guarded below.)"""
    offenders = [
        (n, ln) for n, ln in _bash_block_lines(_read(name)) if _PLACEHOLDER_ASSIGN.match(ln)
    ]
    assert not offenders, f"{name} pastes a raw value into a shell assignment: {offenders}"


_SCAN_COMMANDS = ["pre-merge-check.md", "submit-pr.md", "push-pr-update.md"]


@pytest.mark.parametrize("name", _SCAN_COMMANDS)
def test_commands_invoke_premerge_scan(name):
    """The methodology pattern checks over changed filenames must go through the
    argv-safe helper (premerge_scan.py), never a shell grep over paths. All three
    commands that run those checks must invoke it."""
    assert "premerge_scan.py" in _read(
        name
    ), f"{name} must run pattern checks via premerge_scan.py, not a shell grep"


def test_push_pr_update_no_raw_ref_interpolation():
    """push-pr-update must not paste a git-controlled ref (`<comparison-ref>` /
    `<default-branch>`) into an executable command — git accepts `$()`/backticks in ref
    names. Refs are resolved into shell variables and used quoted. Scans bash blocks
    only, so the prose guard mentioning the placeholder is not matched."""
    offenders = [
        (n, ln)
        for n, ln in _bash_block_lines(_read("push-pr-update.md"))
        if re.search(r"<(comparison-ref|default-branch)>", ln)
    ]
    assert not offenders, f"push-pr-update interpolates a raw ref placeholder: {offenders}"


def test_push_pr_update_ref_vars_are_quoted():
    """It is not enough to reject the raw placeholder — a *bare* `$COMPARISON_REF..HEAD`
    (unquoted) would still let a `$()`/backtick in the ref's value execute. Assert every
    `$COMPARISON_REF`/`$DEFAULT_BRANCH` use in a bash block is quoted (preceded by `"` or
    inside a double-quoted span)."""
    bare = []
    for n, ln in _bash_block_lines(_read("push-pr-update.md")):
        for m in re.finditer(r"\$(?:COMPARISON_REF|DEFAULT_BRANCH)\b", ln):
            before = ln[: m.start()]
            quoted = before.rstrip().endswith('"') or (before.count('"') % 2 == 1)
            if not quoted:
                bare.append((n, ln.strip()))
    assert not bare, f"push-pr-update has BARE (unquoted) ref variable use: {bare}"


def test_quoted_ref_variable_does_not_execute(tmp_path):
    """The safety property behind the quoting: a ref *value* holding `$(...)` as data
    (as it would if git ever returned such a ref name) is inert when used through a
    quoted expansion — the exact shape the command uses, `git rev-list "$REF..HEAD"`.
    Single-quote assignment holds the payload as literal data; the quoted use must not
    re-execute it."""
    import subprocess

    script = "REF='x$(touch SHOULD_NOT)'; git rev-list --count \"$REF..HEAD\" 2>/dev/null; true"
    subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True)
    assert not (tmp_path / "SHOULD_NOT").exists(), "quoted ref-var use executed its payload"


# No prose may interpolate a plan path into a shell command at all any more —
# every plan-path operation (snapshot/persist/staleness check) flows through
# plan_snapshot.py via file ingress. The staleness probe included.
def test_pre_merge_check_has_no_filename_grep():
    """No `grep/pytest/git diff` over a `<changed-…files>` placeholder may remain in
    pre-merge-check — that was the filename-as-argument injection surface. Filenames now
    flow only through the helper and `xargs -0` over its NUL-delimited safe lists."""
    offenders = [
        (n, ln)
        for n, ln in _bash_block_lines(_read("pre-merge-check.md"))
        if re.search(r"(grep|pytest|git diff|ls)\b[^\n]*<changed", ln)
    ]
    assert not offenders, f"pre-merge-check still greps changed filenames: {offenders}"


def test_no_gnu_only_xargs_a():
    """`xargs -a` is a GNU extension that BSD/macOS xargs rejects. The run-lists must be
    fed over stdin (`xargs -0 pytest < file`) so pre-merge-check works on macOS."""
    offenders = [
        (n, ln)
        for n, ln in _bash_block_lines(_read("pre-merge-check.md"))
        if re.search(r"xargs\b[^\n]*\s-a\b", ln)
    ]
    assert not offenders, f"GNU-only `xargs -a` (breaks on macOS): {offenders}"


# NOTE: the plan-review command contracts (no-prose-shasum, check-subcommand
# staleness, helper-review-path lookups, scratch-init, per-ingress plan_path
# confirmation) moved to tests/test_plan_review_skill.py when /review-plan +
# /revise-plan were retired for the .claude/skills/plan-review/ skill.
