"""Tests for .claude/scripts/pr_prepare.py — safe ingress for /submit-pr values.

These guard the property that matters: no title, base, or branch value can carry a
shell metacharacter or command substitution into a command line, an unsafe *explicit*
branch/base is rejected rather than silently rewritten, generated names are valid git
refs, and an already-checked-out feature branch is honoured. The suite is skipped when
the script is absent (e.g. a pip-installed checkout without .claude/scripts/).
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest


def _find_script() -> "pathlib.Path | None":
    candidate = (
        pathlib.Path(__file__).resolve().parent.parent / ".claude" / "scripts" / "pr_prepare.py"
    )
    if candidate.exists():
        return candidate
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        candidate = pathlib.Path(root) / ".claude" / "scripts" / "pr_prepare.py"
        if candidate.exists():
            return candidate
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


_SCRIPT_PATH = _find_script()

pytestmark = pytest.mark.skipif(
    _SCRIPT_PATH is None,
    reason="pr_prepare.py not found (not in repo checkout)",
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("pr_prepare", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# sanitize_branch_portion — the injection-neutralising core
# ---------------------------------------------------------------------------

_MALICIOUS = [
    "Fix `safe_inference` and $(touch /tmp/sentinel)",
    "Add ${IFS}payload; rm -rf .",
    "Handle | pipe && chain > redirect",
    "Quote 'single' and \"double\" and `back`",
    "Newline\ninjection\rtest",
]


@pytest.mark.parametrize("title", _MALICIOUS)
def test_sanitized_portion_is_shell_safe(mod, title):
    out = mod.sanitize_branch_portion(title)
    assert mod.is_shell_safe_ref(out), f"{out!r} still contains metacharacters"
    for ch in "`$();|&><'\"\n\r/ ":
        assert ch not in out


def test_single_underscore_preserved(mod):
    out = mod.sanitize_branch_portion("Fix safe_inference NaN guard")
    assert "safe_inference" in out
    assert "__" not in out


def test_hyphen_runs_collapse(mod):
    assert mod.sanitize_branch_portion("a---b   c") == "a-b-c"


def test_portion_truncated_and_trimmed(mod):
    out = mod.sanitize_branch_portion("-" * 5 + "x" * 80)
    assert len(out) <= 50
    assert not out.startswith("-") and not out.endswith("-")


def test_empty_after_sanitize(mod):
    assert mod.sanitize_branch_portion("!!!///$$$") == ""


# ---------------------------------------------------------------------------
# normalize_ref_portion + generated refs are valid git refs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,forbidden",
    [
        ("a..b", ".."),
        ("foo.lock", ".lock"),
        (".hidden", None),  # leading dot stripped
        ("trailing.", None),  # trailing dot stripped
    ],
)
def test_normalize_fixes_git_invalid_forms(mod, raw, forbidden):
    out = mod.normalize_ref_portion(raw)
    if forbidden:
        assert forbidden not in out
    assert not out.startswith(".") and not out.endswith(".")


@pytest.mark.parametrize(
    "title",
    ["Fix v1.2..final", "Release foo.lock", "...", "Handle a....b edge"],
)
def test_generated_branch_is_valid_git_ref(mod, title):
    # These titles would produce git-invalid refs without normalization.
    b = mod.resolve_branch(None, None, "main", title, "fix")
    assert mod._git_ref_ok(b), f"generated {b!r} fails git check-ref-format"


# ---------------------------------------------------------------------------
# resolve_branch — precedence: explicit > existing feature branch > generated
# ---------------------------------------------------------------------------


def test_generated_branch_has_prefix_and_safe_portion(mod):
    b = mod.resolve_branch(None, None, "main", "Fix `safe_inference` bug", "fix")
    assert b.startswith("fix/")
    assert mod.is_shell_safe_ref(b)
    assert "`" not in b and "$" not in b


def test_generated_branch_slash_only_from_prefix(mod):
    b = mod.resolve_branch(None, None, "main", "touch /tmp/x", "fix")
    assert b.count("/") == 1


def test_existing_feature_branch_is_used_verbatim(mod):
    # On a feature branch, no explicit --branch: use the branch we are ON, not a
    # title-derived name (else push/HEAD_REF target the wrong branch).
    b = mod.resolve_branch(None, "fix/already-here", "main", "some title", "fix")
    assert b == "fix/already-here"


def test_on_base_generates_from_title(mod):
    b = mod.resolve_branch(None, "main", "main", "New thing", "feature")
    assert b == "feature/new-thing"


def test_explicit_branch_conflicts_with_feature_branch_rejected(mod):
    with pytest.raises(ValueError):
        mod.resolve_branch("fix/other", "fix/already-here", "main", "t", "fix")


def test_unknown_change_type_falls_back_to_feature(mod):
    b = mod.resolve_branch(None, None, "main", "whatever", "nonsense")
    assert b.startswith("feature/")


@pytest.mark.parametrize(
    "bad",
    ["evil`whoami`", "a$(id)b", "has space", "semi;colon", "pipe|x", 'quote"x'],
)
def test_explicit_unsafe_branch_rejected(mod, bad):
    with pytest.raises(ValueError):
        mod.resolve_branch(bad, None, "main", "title", "fix")


def test_explicit_safe_branch_used_verbatim(mod):
    assert (
        mod.resolve_branch("fix/manual-branch_1.2", None, "main", "t", "fix")
        == "fix/manual-branch_1.2"
    )


def test_base_default_when_absent(mod):
    assert mod.resolve_base(None) == "main"


def test_explicit_unsafe_base_rejected(mod):
    with pytest.raises(ValueError):
        mod.resolve_base("main`whoami`")


def test_explicit_safe_base_used(mod):
    assert mod.resolve_base("develop") == "develop"


def test_uppercase_explicit_ref_accepted(mod):
    # Uppercase has no shell semantics; real refs like Release/4.0 or V4 must pass.
    assert mod.resolve_branch("Release/4.0", None, "main", "t", "fix") == "Release/4.0"
    assert mod.resolve_base("V4") == "V4"


def test_build_head_ref_fork_without_owner_raises(mod):
    with pytest.raises(ValueError):
        mod.build_head_ref(True, None, "fix/x")
    with pytest.raises(ValueError):
        mod.build_head_ref(True, "", "fix/x")


# ---------------------------------------------------------------------------
# owner_repo / head ref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("git@github.com:owner/repo.git", "owner/repo"),
        ("https://github.com/owner/repo.git", "owner/repo"),
        ("https://github.com/owner/repo", "owner/repo"),
        ("ssh://git@github.com/owner/repo.git", "owner/repo"),
        ("git@github.com:owner/repo.git/", "owner/repo"),
        ("https://github.com/owner/repo/", "owner/repo"),  # trailing slash, no .git
        ("git@github.com:owner/repo/", "owner/repo"),
        ("https://github.com/owner/repo.git/", "owner/repo"),
    ],
)
def test_owner_repo_parses(mod, url, expected):
    assert mod.owner_repo(url) == expected


def test_build_head_ref_direct_vs_fork(mod):
    assert mod.build_head_ref(False, "", "fix/x") == "fix/x"
    assert mod.build_head_ref(True, "forkowner", "fix/x") == "forkowner:fix/x"


# ---------------------------------------------------------------------------
# File-based input: values arrive as FILE CONTENT, never as shell/argv strings
# ---------------------------------------------------------------------------


def test_main_reads_title_from_file_and_executes_nothing(mod, tmp_path):
    sentinel = tmp_path / "sentinel"
    title_file = tmp_path / "raw-title.txt"
    title_file.write_text(f"Fix `safe_inference` and $(touch {sentinel})")
    scratch = tmp_path / "scratch"

    rc = mod.main(
        [
            "--title-file",
            str(title_file),
            "--change-type",
            "fix",
            "--scratch",
            str(scratch),
        ]
    )
    assert rc == 0
    assert not sentinel.exists()

    assert (scratch / "pr-title.txt").read_text() == title_file.read_text().strip()
    branch_out = (scratch / "pr-branch.txt").read_text()
    assert mod.is_shell_safe_ref(branch_out)
    assert "`" not in branch_out and "$" not in branch_out


def test_main_rejects_unsafe_explicit_branch(mod, tmp_path):
    title_file = tmp_path / "t.txt"
    title_file.write_text("ok")
    branch_file = tmp_path / "b.txt"
    branch_file.write_text("evil`whoami`")
    rc = mod.main(
        [
            "--title-file",
            str(title_file),
            "--branch-file",
            str(branch_file),
            "--scratch",
            str(tmp_path / "s"),
        ]
    )
    assert rc == 2


def test_end_to_end_subprocess_no_execution(tmp_path):
    """The reviewer's requirement: exercise the real process boundary, not just an
    in-process call. A payload in the title FILE must not execute when the script is
    run as a subprocess, and the resolved branch must be metacharacter-free."""
    sentinel = tmp_path / "sentinel"
    title_file = tmp_path / "raw-title.txt"
    # Content includes backticks, $(), a semicolon, and a newline.
    title_file.write_text(f"Fix `id`; $(touch {sentinel})\nsecond line")
    scratch = tmp_path / "scratch"

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--title-file",
            str(title_file),
            "--change-type",
            "fix",
            "--scratch",
            str(scratch),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not sentinel.exists(), "payload executed via the subprocess boundary"
    branch_out = (scratch / "pr-branch.txt").read_text()
    for ch in "`$();\n":
        assert ch not in branch_out


def _init_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def test_no_title_clean_tree_uses_last_commit_subject(mod, tmp_path, monkeypatch):
    """No-title path must not error or need a base. On a CLEAN tree the last commit
    subject is the change, so it is the title."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "feat: my subject line"], cwd=repo, check=True)

    scratch = repo / "s"
    rc = mod.main(["--scratch", str(scratch)])  # no --title-file at all
    assert rc == 0
    assert (scratch / "pr-title.txt").read_text() == "feat: my subject line"


def test_no_title_dirty_tree_uses_neutral_not_previous_commit(mod, tmp_path, monkeypatch):
    """In /submit-pr's normal flow the title is resolved before the commit, so the tree
    is dirty. The fallback must NOT reuse the previous, unrelated commit subject."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "OLD unrelated subject"], cwd=repo, check=True)
    # Now dirty the tree (uncommitted change), as at title-resolution time.
    (repo / "g.txt").write_text("new work")

    assert mod.generate_fallback_title() == "Update working tree changes"
    assert "OLD unrelated subject" not in mod.generate_fallback_title()


def test_fallback_title_never_empty(mod):
    assert mod.generate_fallback_title()  # non-empty in this repo


def test_main_draft_flag_emitted(mod, tmp_path):
    title_file = tmp_path / "t.txt"
    title_file.write_text("Add thing")
    scratch = tmp_path / "s"
    # Without --draft
    mod.main(["--title-file", str(title_file), "--scratch", str(scratch)])
    assert (scratch / "pr-draft.txt").read_text() == "false"
    # With --draft
    mod.main(["--title-file", str(title_file), "--draft", "--scratch", str(scratch)])
    assert (scratch / "pr-draft.txt").read_text() == "true"
