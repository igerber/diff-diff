"""Tests for tools/reviewer-eval/reviewer_audit.py (the session-log audit).

Mirrors test_evals_runtime.py's isolation: module-level skip when the harness is
absent, sys.path insert guarded the same way, and NO top-level import of the
module under test (pytestmark applies only after import; a top-level import
would still raise at collection on the isolated /tmp/tests CI legs).
"""

import functools
import importlib.util
import json
import os
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_EVAL_ROOT = _REPO / "tools" / "reviewer-eval"

pytestmark = pytest.mark.skipif(
    not _EVAL_ROOT.exists(),
    reason="reviewer-eval harness not present (isolated install)",
)

if _EVAL_ROOT.exists() and str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))


@functools.lru_cache(maxsize=1)
def _audit():
    import reviewer_audit

    return reviewer_audit


@functools.lru_cache(maxsize=1)
def _run_eval():
    spec = importlib.util.spec_from_file_location(
        "reviewer_eval_run_eval", _EVAL_ROOT / "run_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WT = "/Users/igerber/diff-diff-codex-upgrade/tools/reviewer-eval/runs/.worktrees/0123456789abcdef0123456789abcdef-0123456789abcdef"
ROOT = "/Users/igerber/diff-diff-codex-upgrade"
MARKERS = {
    "repo_branch": "codex-upgrade",
    "repo_common_dir": "/Users/igerber/diff-diff/.git",
    "repo_worktree_paths": ["/Users/igerber/diff-diff-contributions"],
    "repo_ref_names": [
        "contributions",
        "eval-pins/pr590-r1",
        "origin/main",
        "origin",
        "v3.12.0",
        "codex-upgrade",
    ],
    "codex_home_path": None,
}


def _hits(cmd, markers=MARKERS, root=ROOT, literals=None, wt=WT):
    a = _audit()
    stripped = a.strip_own_path(cmd, wt, root)
    pm = a._path_markers(root, markers)
    rr = a._ref_name_regex(markers.get("repo_ref_names") or [], markers.get("repo_branch") or "")
    kw = {} if literals is None else {"literals": literals}
    return a.scan_entry(stripped, root, markers, pm, rr, **kw)


def _wrapped(cmd, quote="'"):
    return f"/bin/zsh -lc {quote}{cmd}{quote} in {WT}"


# --------------------------------------------------------------------------- #
# Own-path substitution
# --------------------------------------------------------------------------- #


def test_strip_own_path_all_forms_and_order():
    a = _audit()
    rel = WT[len(ROOT) + 1 :]
    leaf = os.path.basename(WT)
    for form in (WT, rel, leaf, os.path.realpath(WT)):
        out = a.strip_own_path(f"cat {form}/x.py", WT, ROOT)
        assert out == "cat <WT>/x.py", (form, out)
    # ABSOLUTE form leaves no <repo_root>/ residue (pins realpath -> absolute -> relative -> leaf).
    assert a.strip_own_path(f"grep -rn x {WT}", WT, ROOT) == "grep -rn x <WT>"
    assert "<WT>" in a.strip_own_path(
        f"ls ~/{WT[len(os.path.expanduser('~')) + 1:]}", WT, ROOT
    ) or not WT.startswith(os.path.expanduser("~"))
    # Continuations are joined before scanning.
    assert a.strip_own_path("git diff HEAD \\\n    main", WT, ROOT) == "git diff HEAD main"
    # Nothing else is touched.
    assert a.strip_own_path("echo hello", WT, ROOT) == "echo hello"


def test_own_worktree_git_commands_still_hit_after_substitution():
    assert "GIT_LOG_RE" in _hits(f"git -C {WT} log -1")
    assert "BARE_SHOW_RE" in _hits(f"git -C {WT} show HEAD")
    assert "repo_root" not in _hits(f"cat {WT}/diff_diff/x.py")


# --------------------------------------------------------------------------- #
# Literals
# --------------------------------------------------------------------------- #


def test_every_literal_hits():
    a = _audit()
    for lit in a.LEAK_MARKERS:
        assert lit in _hits(f"x {lit} y"), lit


@pytest.mark.parametrize(
    "cmd, marker",
    [
        ('grep -rl slug "${PWD%/tools/*}"', "${PWD"),
        ('find "$(dirname "$PWD")" -name case.json', "dirname"),
        ("grep -rn slug /Users/igerber --include=*.json", "--include=*.json"),
        ("g=git; $g show main:x.py", "=git"),
        ("git format-patch -1 --stdout", "format-patch"),
        ("git shortlog HEAD", "shortlog"),
        ("git whatchanged -1", "whatchanged"),
        ("git log @~1..@", "@~"),
        ("git status --porcelain=v2 --branch", "porcelain=v2"),
        ("grep branch.oid out.txt", "branch.oid"),
        ("git ls-remote .", "ls-remote"),
        ("cd ..", "cd .."),
        ("git rev-parse HEAD", "rev-parse"),
        ("git diff HEAD campaign-gpt6 -- x.py", "campaign-"),
        ("git describe --always", "describe"),
        ("git name-rev HEAD", "name-rev"),
        ("git log --walk-reflogs", "--walk-reflogs"),
        ("git diff HEAD^..HEAD", "HEAD^"),
        ("rm -r build", " -r "),
    ],
)
def test_literal_examples(cmd, marker):
    assert marker in _hits(cmd)
    assert marker in _hits(_wrapped(cmd)) and marker in _hits(_wrapped(cmd, '"'))


# --------------------------------------------------------------------------- #
# The seven regexes
# --------------------------------------------------------------------------- #

GIT_LOG_HITS = [
    "git log",
    "git log -1",
    "git log abc1234..def5678",
    "git log 0123456789ab",
    "git log -p HEAD~3..",
    "git log HEAD@{1}..HEAD",
    "git log --oneline base..head",
    "git log origin/main..HEAD",
    "git log -n 1000000",
    "git log --grep='eval:..*'",
    "git log -S cafe1234",
    "git --no-optional-locks log",
    "git log @^..@",
    "git log -n 3 --stat && git diff abc1234..def5678",
    "git -C /x log -1",
    "git --git-dir=/x/.git log",
    "git --no-replace-objects log -1",
    "git --exec-path=/x --no-advice log",
    "git -c core.pager=cat log",
]
GIT_LOG_MISSES = [
    "git diff abc1234..def5678",
    "git diff --stat base..head",
    "git logs-dir",
    "cat changelog.md",
    "git diff HEAD..main",
]


@pytest.mark.parametrize("cmd", GIT_LOG_HITS)
def test_git_log_re_hits(cmd):
    for form in (cmd, _wrapped(cmd), _wrapped(cmd, '"')):
        assert "GIT_LOG_RE" in _hits(form), form


@pytest.mark.parametrize("cmd", GIT_LOG_MISSES)
def test_git_log_re_misses(cmd):
    for form in (cmd, _wrapped(cmd), _wrapped(cmd, '"')):
        assert "GIT_LOG_RE" not in _hits(form), form


MAIN_HITS = [
    "git diff HEAD main",
    "git log -p main",
    "git checkout main",
    "git -C /x checkout main",
    "git --no-pager diff main",
    "git diff HEAD..main",
    "git diff main...HEAD",
    "git log -p HEAD..main",
    "git log master..HEAD",
    "git show main:diff_diff/x.py",
    "git diff HEAD heads/main",
    "git diff HEAD refs/heads/main",
    "git diff HEAD remotes/origin/main",
    "git --no-replace-objects diff HEAD..main",
]
MAIN_MISSES = [
    "git diff HEAD~1",
    "ls domain.py",
    "git diff -- main.py",
    "git log -- src/main/",
    "ls maintenance.md",
    "git diff HEAD",
]


@pytest.mark.parametrize("cmd", MAIN_HITS)
def test_main_ref_re_hits(cmd):
    for form in (cmd, _wrapped(cmd), _wrapped(cmd, '"')):
        assert "MAIN_REF_RE" in _hits(form), form


@pytest.mark.parametrize("cmd", MAIN_MISSES)
def test_main_ref_re_misses(cmd):
    for form in (cmd, _wrapped(cmd), _wrapped(cmd, '"')):
        assert "MAIN_REF_RE" not in _hits(form), form


def test_main_ref_two_line_entry_only_with_continuation():
    assert "MAIN_REF_RE" not in _hits("git diff HEAD\nmain")
    assert "MAIN_REF_RE" in _hits("git diff HEAD \\\n    main")


@pytest.mark.parametrize(
    "cmd",
    [
        "git show",
        "git show --stat",
        "git -C /x show",
        "git --no-pager show",
        "git --no-replace-objects show HEAD",
        "git --git-dir /x show HEAD",
    ],
)
def test_bare_show_re_hits(cmd):
    assert "BARE_SHOW_RE" in _hits(cmd) and "BARE_SHOW_RE" in _hits(_wrapped(cmd))


@pytest.mark.parametrize("cmd", ["git showcase", "github show", "git status"])
def test_bare_show_re_misses(cmd):
    assert "BARE_SHOW_RE" not in _hits(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "git tag",
        "git tag -l",
        "git branch --show-current",
        f"git -C {WT} branch -a",
        "git --no-pager tag",
        "git --git-dir=/x/.git branch",
    ],
)
def test_ref_list_re_hits(cmd):
    assert "REF_LIST_RE" in _hits(cmd) and "REF_LIST_RE" in _hits(_wrapped(cmd))


@pytest.mark.parametrize("cmd", ["git tagger", "cat branch.py"])
def test_ref_list_re_misses(cmd):
    assert "REF_LIST_RE" not in _hits(cmd)


@pytest.mark.parametrize(
    "cmd", ["rg secret ..", "ls ..", "find .. -name '*.json'", "diff -r diff_diff ..", "cd ..; ls"]
)
def test_dotdot_re_hits(cmd):
    assert "DOTDOT_RE" in _hits(cmd) and "DOTDOT_RE" in _hits(_wrapped(cmd, '"'))


@pytest.mark.parametrize(
    "cmd",
    [
        "git diff base..head",
        "git log HEAD~3..",
        "git diff main...HEAD",
        "git log --grep='eval:..*'",
        "cat ../x",
    ],
)
def test_dotdot_re_misses(cmd):
    assert "DOTDOT_RE" not in _hits(cmd)


@pytest.mark.parametrize(
    "cmd", ["cat .git", "p=$(cut -d' ' -f2 .git)", "cat .git/HEAD", "cat ../.git/worktrees/x/HEAD"]
)
def test_git_dir_re_hits(cmd):
    assert "GIT_DIR_RE" in _hits(cmd) and "GIT_DIR_RE" in _hits(_wrapped(cmd, '"'))


@pytest.mark.parametrize("cmd", ["cat .gitignore", "ls .github", "git status"])
def test_git_dir_re_misses(cmd):
    assert "GIT_DIR_RE" not in _hits(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "git diff HEAD v3.12.0 -- diff_diff/x.py",
        "git grep pat v3.12.0",
        "git ls-tree v3.12.0",
        "git -C /x show v3.9.1:x.py",
        "git diff HEAD tags/v3.12.0",
        "git diff HEAD refs/tags/v3.12.0 -- f",
        "git diff HEAD..v3.12.0",
    ],
)
def test_tag_ref_re_hits(cmd):
    assert "TAG_REF_RE" in _hits(cmd) and "TAG_REF_RE" in _hits(_wrapped(cmd))


@pytest.mark.parametrize(
    "cmd", ["git diff HEAD~1", "pip install diff-diff==3.12.0", "cat v3.12.0.md"]
)
def test_tag_ref_re_misses(cmd):
    assert "TAG_REF_RE" not in _hits(cmd)


# --------------------------------------------------------------------------- #
# Computed markers
# --------------------------------------------------------------------------- #


def test_computed_path_markers_and_longest_attribution():
    assert _hits("cat /Users/igerber/diff-diff/.git/worktrees/x/HEAD") == [
        "GIT_DIR_RE",
        "worktrees/",
        "repo_common_dir",
    ] or "repo_common_dir" in _hits("cat /Users/igerber/diff-diff/.git/worktrees/x/HEAD")
    assert "repo_main_checkout" in _hits("sed -n 1,80p /Users/igerber/diff-diff/diff_diff/utils.py")
    h = _hits("cat /Users/igerber/diff-diff-contributions/x")
    assert "repo_worktree_path" in h and "repo_main_checkout" not in h, h
    assert "repo_root" in _hits(f"git -C {ROOT} grep slug")
    assert "repo_root" in _hits(f"grep -rn slug {ROOT}/tests")
    assert "repo_root" not in _hits(f"cat {WT}/x.py")
    # Boundary: the primary checkout path is a prefix of repo_root's path but must not match inside it.
    assert "repo_main_checkout" not in _hits(f"cat {ROOT}/x.py")


def test_home_relative_forms_derive_from_recorded_paths(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/igerber")
    import importlib

    importlib.reload(_audit())
    assert "repo_main_checkout" in _hits("sed -n 1,80p ~/diff-diff/diff_diff/utils.py")
    assert "repo_worktree_path" in _hits("ls $HOME/diff-diff-contributions")


def test_union_worktree_marker_from_one_artifact_only(tmp_path):
    """A path present only in ONE artifact's repo_worktree_paths is a marker."""
    from eval_core.models import RunResult
    from eval_core.store import RunStore

    a = _audit()
    store = RunStore(str(tmp_path / "s"))
    u1 = _usage(
        ["/bin/zsh -lc 'ls /tmp/other-wt/x' in " + WT],
        extra={"repo_worktree_paths": ["/tmp/other-wt"]},
    )
    u2 = _usage([], start=0, execs=0)
    r1 = RunResult(case_id="c", config_id="B", repeat_idx=0, usage=u1, run_id="r1")
    r2 = RunResult(case_id="c", config_id="E", repeat_idx=0, usage=u2, run_id="r2")
    store.save("r1", r1)
    store.save("r2", r2)
    sealed = a.artifacts_sha(store.root, ["r1", "r2"])
    out = a.audit(
        [r1, r2], store.root, ROOT, {**MARKERS, "repo_worktree_paths": []}, "rsha", sealed
    )
    assert any("repo_worktree_path" in h["markers"] for h in out["hits"])
    assert out["coverage"] == {"covered": 1, "no_commands": 1, "unaudited": 0, "total": 2}
    assert (
        out["vacuous"] is False
        and out["run_ids_sha"] == "rsha"
        and out["branch"] == "codex-upgrade"
    )


def test_ref_name_marker_is_git_scoped():
    for cmd in (
        "git diff HEAD contributions -- f",
        "git log eval-pins/pr590-r1 -1",
        "git diff HEAD..contributions",
    ):
        assert "repo_ref_name" in _hits(cmd), cmd
    for cmd in ("rg contributions src/", "echo eval-pins/pr590-r1", "git diff HEAD origin"):
        assert "repo_ref_name" not in _hits(cmd), cmd


def test_branch_marker_is_git_scoped():
    assert "repo_branch" in _hits("git diff HEAD codex-upgrade")
    assert _hits("git diff HEAD") == []
    assert "repo_branch" not in _hits("echo codex-upgrade")


def test_codex_home_marker_without_literal():
    a = _audit()
    literals = tuple(m for m in a.LEAK_MARKERS if m != "codex-home-")
    markers = {**MARKERS, "codex_home_path": "/tmp/codex-home-abc"}
    assert _hits("cat /tmp/codex-home-abc/history.jsonl", markers=markers, literals=literals) == [
        "codex_home_path"
    ]


# --------------------------------------------------------------------------- #
# Coverage classes + orientation + rubric rendering
# --------------------------------------------------------------------------- #


def _usage(cmds, start=None, execs=None, other=0, truncated=False, isolated=True, extra=None):
    u = {
        "session_commands": cmds,
        "session_commands_start_lines": len(cmds) if start is None else start,
        "session_commands_exec_lines": len(cmds) if execs is None else execs,
        "session_other_events": other,
        "session_commands_truncated": truncated,
        "codex_home_isolated": isolated,
        "worktree_dir": WT,
    }
    if extra:
        u.update(extra)
    return u


def test_coverage_classes_exact_definition():
    cc = _audit().coverage_class
    assert cc({}) == "unaudited"  # absent key
    assert cc(_usage([])) == "no_commands"
    assert cc(_usage([], start=1)) == "unaudited"
    assert cc(_usage([], execs=1)) == "unaudited"
    u = _usage([])
    del u["session_commands_exec_lines"]
    assert cc(u) == "unaudited"
    assert cc(_usage(["/bin/zsh -lc 'ls' in /x"])) == "covered"
    assert cc(_usage(["/bin/zsh -lc 'ls' in /x"], isolated=False)) == "unaudited"
    u = _usage(["/bin/zsh -lc 'ls' in /x"])
    del u["codex_home_isolated"]
    assert cc(u) == "unaudited"
    assert cc(_usage(["/bin/zsh -lc 'ls' in /x"], other=2)) == "unaudited"
    assert cc(_usage(["/bin/zsh -lc 'ls' in /x\n[truncated]"])) == "unaudited"
    assert cc(_usage(["/bin/zsh -lc 'ls' in /x"], truncated=True)) == "unaudited"
    # Truncation state must be an explicit False: missing / None / 1 are unknown -> unaudited.
    u = _usage(["/bin/zsh -lc 'ls' in /x"])
    del u["session_commands_truncated"]
    assert cc(u) == "unaudited"
    assert cc(_usage(["/bin/zsh -lc 'ls' in /x"], truncated=None)) == "unaudited"
    assert cc(_usage(["/bin/zsh -lc 'ls' in /x"], truncated=1)) == "unaudited"
    assert cc(_usage([], truncated=None)) == "unaudited"
    # exec_lines == 2, start_lines == 1 (a command in a format the start regex missed).
    assert cc(_usage(["/bin/zsh -lc 'ls' in /x"], execs=2)) == "unaudited"


def test_orientation_only_definition():
    a = _audit()
    assert a.ORIENTATION_MARKERS <= set(a.LEAK_MARKERS) | {n for n, _ in a.REGEX_RULES}
    assert set(_hits("git log -1")) <= a.ORIENTATION_MARKERS
    assert set(_hits(f"git -C {WT} log -1")) <= a.ORIENTATION_MARKERS
    assert not set(_hits("git log --all")) <= a.ORIENTATION_MARKERS
    assert not set(_hits("git log main")) <= a.ORIENTATION_MARKERS
    assert not set(_hits("cat ../x")) <= a.ORIENTATION_MARKERS


def test_audit_vacuous_and_failed_manifest_refusal(tmp_path, monkeypatch):
    from eval_core.models import RunResult
    from eval_core.store import RunStore

    a = _audit()
    store = RunStore(str(tmp_path / "s"))
    r = RunResult(
        case_id="c",
        config_id="B",
        repeat_idx=0,
        usage=_usage(["/bin/zsh -lc 'ls' in /x\n[truncated]"]),
        run_id="r",
    )
    store.save("r", r)
    sealed = a.artifacts_sha(store.root, ["r"])
    out = a.audit([r], store.root, ROOT, MARKERS, "rsha", sealed)
    assert out["vacuous"] is True and out["coverage"]["unaudited"] == 1
    assert out["unaudited"] == [{"case": "c", "arm": "B", "repeat": 0}]
    with pytest.raises(ValueError, match="artifacts changed"):
        a.audit([r], store.root, ROOT, MARKERS, "rsha", "0" * 64)
    with pytest.raises(FileNotFoundError):
        a.artifacts_sha(store.root, ["missing"])
    # The CLI refuses a failed manifest (traversal refused too).
    run_eval = _run_eval()
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "f-manifest.json").write_text(json.dumps({"failed": True}))
    with pytest.raises(SystemExit, match="failed/incomplete"):
        run_eval.cmd_audit(type("N", (), {"subdir": "f", "output": str(tmp_path / "h.json")})())
    assert (
        run_eval.cmd_audit(type("N", (), {"subdir": "../x", "output": str(tmp_path / "h.json")})())
        == 1
    )


def test_rubric_renders_exactly_leak_rules():
    a = _audit()
    rubric = (_EVAL_ROOT / "GRADING_RUBRIC.md").read_text()
    begin, end = "<!-- LEAK_RULES:begin -->", "<!-- LEAK_RULES:end -->"
    assert begin in rubric and end in rubric
    rendered = rubric.split(begin, 1)[1].split(end, 1)[0].strip("\n")
    assert rendered == a.render_rules_for_rubric().strip("\n")
    # One source, two renderings: every literal and every regex source appears, in order.
    pos = -1
    for kind, src in a.LEAK_RULES:
        needle = f"`{src}`" if kind == "literal" else f"`{kind}` = `{src}`"
        nxt = rendered.find(needle, pos + 1)
        assert nxt > pos, (kind, src)
        pos = nxt


def test_rubric_hygiene_no_arm_identities():
    import re

    rubric = (_EVAL_ROOT / "GRADING_RUBRIC.md").read_text()
    low = rubric.lower()
    cfg = json.loads((_EVAL_ROOT / "config" / "configs.json").read_text())
    for arm in cfg["arms"]:
        assert arm["model"].lower() not in low
    for word in (
        "gpt-",
        "astra",
        "sol",
        "terra",
        "luna",
        "no-go",
        "parity",
        "2026-07",
        "arm b",
        "arm e",
    ):
        assert not re.search(r"\b" + re.escape(word) + r"\b", low), word
    for sentence in (
        'The "_optional_ (must_catch=false)" tag in the bundle is a legacy label with no grading meaning: grade every listed bug in full.',
        "This rubric's one-row-per-(case, bug, arm label, repeat) table overrides the bundle's \"How to grade\" row format.",
        "On a negative-control case, list EVERY finding whose severity is above the allowed severities, including findings on a known-FP topic; the engine, not the grader, decides what counts.",
    ):
        assert " ".join(sentence.split()) in " ".join(rubric.split()), sentence
