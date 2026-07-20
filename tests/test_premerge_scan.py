"""Tests for .claude/scripts/premerge_scan.py — argv-safe pre-merge scanning.

The property that matters: a git-controlled filename — however hostile — is only ever
consumed as data, never executed. Pattern checks run in pure Python over file content;
discovery is NUL-delimited argv; unsafe paths are screened out. The integration tests
place `$(touch sentinel)` filenames in the working tree (staged AND untracked) and
assert nothing executes. Skipped when the script is absent (installed distribution).
"""

import importlib.util
import pathlib
import subprocess

import pytest


def _find_script():
    cand = (
        pathlib.Path(__file__).resolve().parent.parent / ".claude" / "scripts" / "premerge_scan.py"
    )
    if cand.exists():
        return cand
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        cand = pathlib.Path(root) / ".claude" / "scripts" / "premerge_scan.py"
        if cand.exists():
            return cand
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


_SCRIPT = _find_script()
pytestmark = pytest.mark.skipif(_SCRIPT is None, reason="premerge_scan.py not found")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("premerge_scan", _SCRIPT)
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,safe",
    [
        ("diff_diff/staggered.py", True),
        ("diff_diff/visualization/_event_study.py", True),
        ("tests/test_foo.py", True),
        ("diff_diff/$(touch x).py", False),
        ("diff_diff/`whoami`.py", False),
        ("tests/test_a;b.py", False),
        ("diff_diff/has space.py", False),
        ("diff_diff/quote'.py", False),
    ],
)
def test_is_safe_path(mod, path, safe):
    assert mod.is_safe_path(path) is safe


def test_categorize(mod):
    cats = mod.categorize(
        [
            "diff_diff/staggered.py",
            "diff_diff/__init__.py",
            "diff_diff/visualization/_event_study.py",
            "tests/test_staggered.py",
            "docs/tutorials/01_intro.ipynb",
            "README.md",
        ]
    )
    assert cats["methodology"] == [
        "diff_diff/staggered.py",
        "diff_diff/visualization/_event_study.py",
    ]
    assert cats["tests"] == ["tests/test_staggered.py"]
    assert cats["notebooks"] == ["docs/tutorials/01_intro.ipynb"]
    assert "README.md" in cats["docs"]


def test_check_content_patterns(mod):
    lines = [
        "t_stat = effect / se",  # A
        "t_stat = safe_inference(effect, se)",  # not A (has safe_inference)
        "x = 1 if se > 0 else 0.0",  # B
        "compute_confidence_interval(se)",  # D (no guard)
        "if np.isfinite(se): compute_confidence_interval(se)",  # not D (guard)
    ]
    msgs = [m for _, m in mod.check_content_patterns(lines)]
    assert any("Check A" in m for m in msgs)
    assert any("Check B" in m for m in msgs)
    assert any("Check D" in m for m in msgs)
    # The guarded/safe_inference lines must NOT produce A or D.
    assert sum("Check A" in m for m in msgs) == 1
    assert sum("Check D" in m for m in msgs) == 1


def test_new_params_missing_from_get_params_ast(mod):
    # AST-based: restricted to __init__ self-assigns vs the class's own get_params.
    text = (
        "class E:\n"
        "    def __init__(self):\n"
        "        self.new_param = 1\n"
        "        self.kept = 2\n"
        "    def get_params(self):\n"
        "        return {'kept': self.kept}\n"
    )
    added = mod.param_names_from_added(["        self.new_param = 1", "        self.kept = 2"])
    assert mod.new_params_missing_from_get_params(added, text) == ["new_param"]


def test_check_c_ignores_assignments_outside_init(mod):
    # A self.X assigned in a *non*-__init__ method must not count as a param.
    text = (
        "class E:\n"
        "    def __init__(self):\n"
        "        self.real = 1\n"
        "    def fit(self):\n"
        "        self.other = 2\n"  # not a param
        "    def get_params(self):\n"
        "        return {'real': self.real}\n"
    )
    # 'other' is not in __init__, so it is not a missing param; 'real' is present.
    assert mod.new_params_missing_from_get_params(["real", "other"], text) == []


def test_load_groups(mod):
    yaml = (
        "groups:\n"
        "  staggered:\n"
        "    - diff_diff/staggered.py\n"
        "    - diff_diff/staggered_bootstrap.py\n"
        "\n"
        "other:\n"
    )
    m = mod.load_groups(yaml)
    assert m["diff_diff/staggered_bootstrap.py"] == "staggered"


def test_stem_of(mod):
    assert mod.stem_of("diff_diff/staggered.py") == "staggered"
    assert mod.stem_of("diff_diff/_nprobust_port.py") == "nprobust_port"  # leading _ dropped
    assert mod.stem_of("diff_diff/visualization/_event_study.py") == "visualization"  # package


def test_resolve_tests(mod):
    tests = [
        "tests/test_staggered.py",
        "tests/test_methodology_staggered.py",
        "tests/test_other.py",
    ]
    hits = mod.resolve_tests("staggered", tests)
    assert hits == ["tests/test_methodology_staggered.py", "tests/test_staggered.py"]


# ---------------------------------------------------------------------------
# Integration: hostile filenames — staged AND untracked — never execute
# ---------------------------------------------------------------------------


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def test_hostile_filenames_do_not_execute(mod, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "diff_diff").mkdir(parents=True)
    (repo / "tests").mkdir()
    _init_repo(repo)
    monkeypatch.chdir(repo)
    # A base commit so `git diff HEAD` works.
    (repo / "diff_diff" / "ok.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)

    # Sentinel is a bare name in the repo (no slashes), so it can be embedded in a
    # filename. If any codepath runs `$(touch sentinel)` as shell, the file appears.
    sentinel = repo / "sentinel"
    # A STAGED hostile filename and an UNTRACKED hostile filename.
    staged = repo / "diff_diff" / "$(touch sentinel).py"
    staged.write_text("t_stat = effect / se\n")
    subprocess.run(["git", "add", "--", str(staged)], cwd=repo, check=True)
    untracked = repo / "tests" / "test_`touch sentinel`.py"
    untracked.write_text("x\n")

    scratch = repo / "s"
    rc = mod.main(["--scratch", str(scratch)])

    assert not sentinel.exists(), "a hostile filename executed"
    assert rc == 3, "unsafe paths must make the scan exit non-zero so the caller stops"
    # The safe run-lists must not contain the hostile names.
    run_tests = (scratch / "run-tests.z").read_text()
    assert "touch" not in run_tests


def test_clean_repo_returns_zero(mod, tmp_path, monkeypatch):
    repo = tmp_path / "clean"
    (repo / "diff_diff").mkdir(parents=True)
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "diff_diff" / "ok.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    (repo / "diff_diff" / "ok.py").write_text("x = 2\n")  # a clean change

    rc = mod.main(["--scratch", str(repo / "s")])
    assert rc == 0


def test_changed_test_file_included_in_run_list(mod, tmp_path, monkeypatch):
    """A test-only change must appear in run-tests.z (DT-1)."""
    repo = tmp_path / "r"
    (repo / "tests").mkdir(parents=True)
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "tests" / "test_thing.py").write_text("def test_x(): pass\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    (repo / "tests" / "test_thing.py").write_text("def test_x(): assert True\n")

    scratch = repo / "s"
    assert mod.main(["--scratch", str(scratch)]) == 0
    assert "tests/test_thing.py" in (scratch / "run-tests.z").read_text()


def test_committed_range_scans_ahead_commit(mod, tmp_path, monkeypatch, capsys):
    """--range scans a committed range even when the working tree is clean (CQ-1), and
    the Check A finding is actually reported (not merely rc==0)."""
    repo = tmp_path / "r"
    (repo / "diff_diff").mkdir(parents=True)
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "diff_diff" / "m.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "baseref"], cwd=repo, check=True)
    (repo / "diff_diff" / "m.py").write_text("t_stat = effect / se\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "work"], cwd=repo, check=True)

    scratch = repo / "s"
    assert mod.main(["--scratch", str(scratch), "--range", "baseref..HEAD"]) == 0
    out = capsys.readouterr().out
    assert "diff_diff/m.py" in out and "Check A" in out


def test_untracked_estimator_check_c(mod, tmp_path, monkeypatch, capsys):
    """An untracked new estimator whose param is absent from get_params() must be
    flagged by Check C — `git diff HEAD` shows no added lines for it (CQ round-21)."""
    repo = tmp_path / "r"
    (repo / "diff_diff").mkdir(parents=True)
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "diff_diff" / "base.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    # UNTRACKED new estimator with a param missing from get_params.
    (repo / "diff_diff" / "new_est.py").write_text(
        "class E:\n"
        "    def __init__(self):\n"
        "        self.new_param = 1\n"
        "    def get_params(self):\n"
        "        return {}\n"
    )
    assert mod.main(["--scratch", str(repo / "s")]) == 0
    out = capsys.readouterr().out
    assert "Check C" in out and "new_param" in out


def test_deleted_methodology_file_does_not_crash(mod, tmp_path, monkeypatch):
    """Deleting a methodology file must not raise/exit 4 (round-21)."""
    repo = tmp_path / "r"
    (repo / "diff_diff").mkdir(parents=True)
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "diff_diff" / "gone.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    subprocess.run(["git", "rm", "-q", "diff_diff/gone.py"], cwd=repo, check=True)

    assert mod.main(["--scratch", str(repo / "s")]) == 0


def test_conftest_not_a_pytest_target(mod, tmp_path, monkeypatch):
    """A changed tests/conftest.py must not enter run-tests.z (would exit 5) (DT-1)."""
    repo = tmp_path / "r"
    (repo / "tests").mkdir(parents=True)
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "tests" / "conftest.py").write_text("import pytest\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    (repo / "tests" / "conftest.py").write_text("import pytest  # changed\n")

    scratch = repo / "s"
    assert mod.main(["--scratch", str(scratch)]) == 0
    assert "conftest.py" not in (scratch / "run-tests.z").read_text()


def test_parse_name_status_z(mod):
    # A modified, an added, a deleted, and a rename (dest → M).
    toks = ["M", "a.py", "A", "b.py", "D", "c.py", "R100", "old.py", "new.py"]
    st = mod.parse_name_status_z(toks)
    assert st == {"a.py": "M", "b.py": "A", "c.py": "D", "new.py": "M"}


def test_git_error_fails_closed(mod, tmp_path, monkeypatch):
    """A bad range (git error) must exit non-zero, not report 0 findings (CQ-2), and
    must not leave STALE run-lists in the reused scratch dir (CI review P2)."""
    repo = tmp_path / "r"
    repo.mkdir()
    _init_repo(repo)
    monkeypatch.chdir(repo)
    (repo / "f.py").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)

    scratch = repo / "s"
    # Seed a STALE run-list a prior run might have left behind.
    scratch.mkdir()
    (scratch / "run-tests.z").write_text("tests/test_unrelated.py")

    rc = mod.main(["--scratch", str(scratch), "--range", "nonexistent-ref..HEAD"])
    assert rc == 4
    # The stale list must have been truncated at startup, before the git error.
    assert (scratch / "run-tests.z").read_text() == ""
