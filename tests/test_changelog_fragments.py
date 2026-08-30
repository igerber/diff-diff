"""Guards for the changelog-fragment system (.claude/scripts/changelog_compile.py).

Three contracts:

1. **Repo-state invariant** — `CHANGELOG.md`'s `## [Unreleased]` section is
   pointer-only and every `changelog.d/` fragment is valid, so a PR that
   edits Unreleased directly (the old convention) fails CI in the docs-tests
   lane.
2. **Compiler behavior** — `check` negatives and the `compile` assembly
   (category order, contiguous joins, comparison links, the exit-4
   idempotent re-run, the downgrade/dirty/empty guards) in tmp fixtures.
3. **Workflow pins** — docs-tests.yml's `on:` path filters carry the
   fragment surfaces and the doc-snippets job actually invokes this file
   (path filters alone would go silently dead if the step were deleted).

Skipped when the script is absent (installed distribution). The module must
IMPORT cleanly on the Python 3.9 CI leg, so annotations stay 3.9-safe.
"""

import importlib.util
import pathlib
import shutil
import subprocess

import pytest


def _find_repo_root():
    cand = pathlib.Path(__file__).resolve().parent.parent
    if (cand / ".claude" / "scripts" / "changelog_compile.py").exists():
        return cand
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        cand = pathlib.Path(root)
        if (cand / ".claude" / "scripts" / "changelog_compile.py").exists():
            return cand
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


_REPO_ROOT = _find_repo_root()
pytestmark = pytest.mark.skipif(
    _REPO_ROOT is None, reason="changelog_compile.py not found (installed distribution)"
)

POINTER = (
    "<!-- entries live in changelog.d/*.md; "
    "compiled at release by .claude/scripts/changelog_compile.py -->"
)


@pytest.fixture(scope="module")
def mod():
    path = _REPO_ROOT / ".claude" / "scripts" / "changelog_compile.py"
    spec = importlib.util.spec_from_file_location("changelog_compile", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

MINIMAL_CHANGELOG = (
    "# Changelog\n\n"
    "## [Unreleased]\n\n" + POINTER + "\n\n"
    "## [1.2.0] - 2026-01-15\n\n"
    "### Added\n"
    "- old entry\n\n"
    "## [1.1.0] - 2026-01-01\n\n"
    "### Fixed\n"
    "- older entry\n\n"
    "[1.2.0]: https://github.com/x/y/compare/v1.1.0...v1.2.0\n"
    "[1.1.0]: https://github.com/x/y/releases/tag/v1.1.0\n"
)

GOOD_FRAGMENT = "### Fixed\n- **a fix**: details, wrapped with\n  a continuation line.\n"


def make_repo(tmp_path, changelog=MINIMAL_CHANGELOG, fragments=None, readme=True):
    d = tmp_path / "changelog.d"
    d.mkdir()
    if readme:
        (d / "README.md").write_text("spec\n")
    for name, content in (fragments or {}).items():
        (d / name).write_text(content)
    (tmp_path / "CHANGELOG.md").write_text(changelog)
    return tmp_path


def check_findings(mod, root):
    findings, _ = mod.run_check(root)
    return findings


@pytest.fixture
def git_commit_all():
    """git init + commit everything in a fixture repo (skips if git absent).

    Call with init_only=True to initialize without committing (for tests
    exercising the not-committed-in-HEAD path).
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("git unavailable")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin",
    }

    def _commit(root, init_only=False):
        subprocess.run([git, "init", "-q"], cwd=root, check=True)
        if not init_only:
            subprocess.run([git, "add", "-A"], cwd=root, check=True, env=env)
            subprocess.run([git, "commit", "-q", "-m", "seed"], cwd=root, check=True, env=env)

    return _commit


# ---------------------------------------------------------------------------
# 1. Live-repo invariant
# ---------------------------------------------------------------------------


class TestLiveRepo:
    def test_check_passes_on_repo(self, mod):
        assert check_findings(mod, _REPO_ROOT) == []

    def test_readme_mirrors_compiler_categories(self, mod):
        """README<->compiler parity (the TestLintWorkflowPinSync convention):
        every category, in order, appears as a backticked token in the
        README's category list."""
        readme = (_REPO_ROOT / "changelog.d" / "README.md").read_text()
        tokens = ["`%s`" % c for c in mod.CATEGORIES]
        positions = [readme.find(t) for t in tokens]
        assert all(
            p >= 0 for p in positions
        ), "changelog.d/README.md is missing categories: " + ", ".join(
            t for t, p in zip(tokens, positions) if p < 0
        )
        assert positions == sorted(positions), (
            "changelog.d/README.md lists categories in a different order "
            "than the compiler constant"
        )

    def test_direct_unreleased_edit_fails_check(self, mod, tmp_path):
        text = (_REPO_ROOT / "CHANGELOG.md").read_text()
        polluted = text.replace(
            POINTER, POINTER + "\n\n### Added\n- a bullet added the old way\n", 1
        )
        root = tmp_path
        (root / "changelog.d").mkdir()
        (root / "changelog.d" / "README.md").write_text("spec\n")
        (root / "CHANGELOG.md").write_text(polluted)
        assert any("pointer comment" in f for f in check_findings(mod, root))


# ---------------------------------------------------------------------------
# 2a. check negatives / positives
# ---------------------------------------------------------------------------


class TestCheck:
    @pytest.mark.parametrize(
        "name",
        [
            "no-date-prefix.md",
            "2026-08-30-dashes.md",
            "20260830_underscore.md",
            "20269999-impossible-date.md",
            "20260830-UPPER.md",
            "20260830-slug.txt",
        ],
    )
    def test_bad_fragment_names(self, mod, tmp_path, name):
        root = make_repo(tmp_path, fragments={name: GOOD_FRAGMENT})
        assert any(name in f for f in check_findings(mod, root))

    @pytest.mark.parametrize(
        "body,needle",
        [
            ("### Wat\n- bullet\n", "unknown category"),
            ("", "no '### <Category>' block"),
            ("### Fixed\n", "no top-level bullet"),
            ("### Fixed\n- \n", "empty top-level bullet"),
            ("### Fixed\n-  \n", "empty top-level bullet"),
            ("### Fixed\n- ok\n## [9.9.9]\n", "'## ' header"),
            ("- floating bullet\n### Fixed\n- ok\n", "outside any"),
            ("### Fixed\nnot a bullet\n", "not a top-level bullet"),
        ],
    )
    def test_bad_fragment_bodies(self, mod, tmp_path, body, needle):
        root = make_repo(tmp_path, fragments={"20260830-x.md": body})
        assert any(needle in f for f in check_findings(mod, root))

    def test_missing_readme_fails(self, mod, tmp_path):
        root = make_repo(tmp_path, readme=False)
        assert any("README.md" in f for f in check_findings(mod, root))

    def test_subdirectory_and_stray_file_fail(self, mod, tmp_path):
        root = make_repo(tmp_path)
        (root / "changelog.d" / "nested").mkdir()
        assert any("subdirectories" in f for f in check_findings(mod, root))

    def test_dotfile_ignored(self, mod, tmp_path):
        root = make_repo(tmp_path, fragments={"20260830-ok.md": GOOD_FRAGMENT})
        (root / "changelog.d" / ".DS_Store").write_bytes(b"\x00junk")
        assert check_findings(mod, root) == []

    def test_missing_pointer_fails(self, mod, tmp_path):
        root = make_repo(tmp_path, changelog=MINIMAL_CHANGELOG.replace(POINTER + "\n\n", ""))
        assert any("pointer comment" in f for f in check_findings(mod, root))

    def test_duplicated_pointer_fails(self, mod, tmp_path):
        root = make_repo(
            tmp_path, changelog=MINIMAL_CHANGELOG.replace(POINTER, POINTER + "\n" + POINTER)
        )
        assert any("pointer comment" in f for f in check_findings(mod, root))

    def test_missing_unreleased_header_fails(self, mod, tmp_path):
        root = make_repo(tmp_path, changelog=MINIMAL_CHANGELOG.replace("## [Unreleased]\n", ""))
        assert any("Unreleased" in f for f in check_findings(mod, root))

    def test_check_rejects_symlink_fragment_before_reading(self, mod, tmp_path):
        # A symlink at a fragment path must be a finding at check level (CI),
        # not just at compile time — its target is mutable out-of-band.
        root = make_repo(tmp_path)
        target = root / "real-content.md"
        target.write_text(GOOD_FRAGMENT)
        (root / "changelog.d" / "20260830-x.md").symlink_to(target)
        assert any("not a regular file" in f for f in check_findings(mod, root))

    def test_check_rejects_dangling_symlink_without_crashing(self, mod, tmp_path):
        root = make_repo(tmp_path)
        (root / "changelog.d" / "20260830-x.md").symlink_to(root / "does-not-exist.md")
        assert any("not a regular file" in f for f in check_findings(mod, root))

    def test_check_rejects_undecodable_fragment(self, mod, tmp_path):
        root = make_repo(tmp_path)
        (root / "changelog.d" / "20260830-x.md").write_bytes(b"### Fixed\n- \xff\xfe junk\n")
        assert any("unreadable" in f for f in check_findings(mod, root))

    def test_duplicate_unreleased_headers_fail(self, mod, tmp_path):
        # A second '## [Unreleased]' section could carry direct bullets that
        # the first-match slice never inspects; exactly one header is allowed.
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15\n",
            "## [Unreleased]\n\n### Added\n- smuggled direct bullet\n\n"
            "## [1.2.0] - 2026-01-15\n",
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert any("exactly one is allowed" in f for f in check_findings(mod, root))

    def test_eof_only_unreleased_header_is_finding_not_traceback(self, mod, tmp_path):
        # File ending exactly at the header with no trailing newline must
        # produce a validation finding, not a ValueError.
        changelog = "# Changelog\n\n## [Unreleased]"
        root = make_repo(tmp_path, changelog=changelog)
        assert any("pointer comment" in f for f in check_findings(mod, root))

    def test_blank_lines_around_pointer_tolerated(self, mod, tmp_path):
        root = make_repo(
            tmp_path, changelog=MINIMAL_CHANGELOG.replace(POINTER, "\n" + POINTER + "\n\n")
        )
        assert check_findings(mod, root) == []


# ---------------------------------------------------------------------------
# 2b. compile behavior
# ---------------------------------------------------------------------------


def run_compile(mod, root, version="1.3.0", date="2026-08-30", allow_dirty=True):
    return mod.run_compile(root, version, date, allow_dirty)


class TestCompile:
    def test_category_order_and_contiguous_join(self, mod, tmp_path):
        root = make_repo(
            tmp_path,
            fragments={
                "20260829-bbb.md": "### Fixed\n- fix from bbb\n\n### Added\n- add from bbb\n",
                "20260828-aaa.md": "### Fixed\n- fix from aaa\n",
                "20260830-ccc.md": "### Internal\n- internal from ccc\n",
            },
        )
        assert run_compile(mod, root) == 0
        text = (root / "CHANGELOG.md").read_text()
        section = text[text.index("## [1.3.0]") : text.index("## [1.2.0]")]
        # Category order: Added before Fixed before Internal.
        assert (
            section.index("### Added") < section.index("### Fixed") < section.index("### Internal")
        )
        # Within Fixed: ascending filename order, joined contiguously.
        assert "- fix from aaa\n- fix from bbb" in section

    def test_header_and_link(self, mod, tmp_path):
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root) == 0
        text = (root / "CHANGELOG.md").read_text()
        assert "## [1.3.0] - 2026-08-30\n" in text
        link = "[1.3.0]: https://github.com/x/y/compare/v1.2.0...v1.3.0\n"
        assert link in text
        # Inserted immediately above the PREV link line.
        assert text.index(link) < text.index("[1.2.0]:")

    def test_fragments_deleted_and_round_trip(self, mod, tmp_path):
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root) == 0
        remaining = [p.name for p in (root / "changelog.d").iterdir() if not p.name.startswith(".")]
        assert remaining == ["README.md"]
        # Post-compile tree still passes check, and re-compile exits 4.
        assert check_findings(mod, root) == []
        assert run_compile(mod, root) == 4

    def test_prev_is_semver_max_not_string_max(self, mod, tmp_path):
        changelog = (
            "# Changelog\n\n"
            "## [Unreleased]\n\n" + POINTER + "\n\n"
            "## [3.11.0] - 2026-08-29\n\n### Added\n- x\n\n"
            "## [3.9.1] - 2026-08-17\n\n### Fixed\n- y\n\n"
            "[3.11.0]: https://github.com/x/y/compare/v3.9.1...v3.11.0\n"
            "[3.9.1]: https://github.com/x/y/releases/tag/v3.9.1\n"
        )
        root = make_repo(tmp_path, changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT})
        # 3.10.0 <= semver-max 3.11.0 even though "3.9.1" is the string max.
        assert run_compile(mod, root, version="3.10.0") != 0
        assert run_compile(mod, root, version="3.12.0") == 0
        assert "compare/v3.11.0...v3.12.0" in (root / "CHANGELOG.md").read_text()

    def test_version_not_above_prev_rejected(self, mod, tmp_path):
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root, version="1.2.0") != 0  # equal->exit-4 lane guards apply
        assert run_compile(mod, root, version="1.1.9") == 1

    @pytest.mark.parametrize("version", ["03.1.0", "1.02.0", "1.0", "v1.3.0", "1.3.0.0"])
    def test_bad_versions_rejected(self, mod, tmp_path, version):
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root, version=version) == 2

    @pytest.mark.parametrize("date", ["20260830", "2026-8-30", "2026-13-01", "yesterday"])
    def test_bad_dates_rejected(self, mod, tmp_path, date):
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root, date=date) == 2

    def test_exit4_requires_no_fragments(self, mod, tmp_path):
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root) == 0
        (root / "changelog.d" / "20260831-late.md").write_text(GOOD_FRAGMENT)
        assert run_compile(mod, root) == 1  # header exists AND fragments remain

    def test_exit4_requires_header_is_prev(self, mod, tmp_path):
        root = make_repo(tmp_path)  # no fragments
        # 1.1.0 exists but is older than PREV=1.2.0 -> downgrade refusal.
        assert run_compile(mod, root, version="1.1.0") == 1

    def test_exit4_requires_nonempty_section(self, mod, tmp_path):
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15\n\n### Added\n- old entry\n",
            "## [1.2.0] - 2026-01-15\n",
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_exit4_requires_dated_header(self, mod, tmp_path):
        changelog = MINIMAL_CHANGELOG.replace("## [1.2.0] - 2026-01-15", "## [1.2.0]")
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_exit4_rejects_noncanonical_header_date(self, mod, tmp_path):
        # An impossible/malformed date must not be certified and propagated
        # into CITATION.cff via the exit-4 RELEASE_DATE reuse.
        changelog = MINIMAL_CHANGELOG.replace("## [1.2.0] - 2026-01-15", "## [1.2.0] - 2026-99-99")
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_compile_refuses_duplicate_unreleased_headers(self, mod, tmp_path):
        # check runs as compile's first step, so a duplicate Unreleased
        # section blocks compilation before any insertion or deletion.
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15\n",
            "## [Unreleased]\n\n### Added\n- smuggled direct bullet\n\n"
            "## [1.2.0] - 2026-01-15\n",
        )
        root = make_repo(tmp_path, changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root) == 1
        assert (root / "changelog.d" / "20260830-x.md").exists()

    def test_duplicate_release_headers_rejected_not_certified(self, mod, tmp_path):
        # Two '## [1.2.0]' sections with a nonempty first section and a valid
        # comparison link: exit 4 must NOT certify this (the loop inspects
        # only the first match); any duplicated release version is a corrupt
        # changelog and exits 1.
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.1.0] - 2026-01-01\n",
            "## [1.2.0] - 2026-01-15\n\n### Fixed\n- duplicate section\n\n"
            "## [1.1.0] - 2026-01-01\n",
        )
        root = make_repo(tmp_path, changelog=changelog)  # no fragments
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_exit4_target_header_at_eof_is_error_not_traceback(self, mod, tmp_path):
        # An existing target header ending the file with no trailing newline
        # must produce the empty-section error, not a ValueError (the same
        # EOF anti-pattern fixed in _unreleased_slice).
        changelog = (
            "# Changelog\n\n"
            "## [Unreleased]\n\n" + POINTER + "\n\n"
            "## [1.1.0] - 2026-01-01\n\n### Fixed\n- older entry\n\n"
            "[1.1.0]: https://github.com/x/y/releases/tag/v1.1.0\n"
            "## [1.2.0] - 2026-01-15"
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_exit4_rejects_sole_release_header(self, mod, tmp_path):
        # A lone target header (no predecessor to anchor the comparison
        # link) cannot be compiler output; exit 4 must not certify it.
        changelog = (
            "# Changelog\n\n"
            "## [Unreleased]\n\n" + POINTER + "\n\n"
            "## [1.2.0] - 2026-01-15\n\n"
            "### Added\n"
            "- only entry\n"
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_exit4_rejects_wrongly_sourced_comparison_link(self, mod, tmp_path):
        # The link must read compare/v<immediate-predecessor>...v<target>;
        # a link sourced from any other version was not produced by the
        # compiler and must not be certified via exit 4.
        changelog = MINIMAL_CHANGELOG.replace(
            "[1.2.0]: https://github.com/x/y/compare/v1.1.0...v1.2.0\n",
            "[1.2.0]: https://github.com/x/y/compare/v0.9.0...v1.2.0\n",
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_exit4_requires_target_comparison_link(self, mod, tmp_path):
        # A hand-built section without its comparison link is not a completed
        # compile; bump-version's step 6 assumes the link exists on exit 4.
        changelog = MINIMAL_CHANGELOG.replace(
            "[1.2.0]: https://github.com/x/y/compare/v1.1.0...v1.2.0\n", ""
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_no_fragments_no_header_rejected(self, mod, tmp_path):
        root = make_repo(tmp_path)
        assert run_compile(mod, root) == 1

    def test_dirty_guard(self, mod, tmp_path, git_commit_all):
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        # Non-git root without --allow-dirty: fail-closed.
        assert run_compile(mod, root, allow_dirty=False) == 1
        # Git root with an uncommitted fragment: refused; --allow-dirty passes.
        git_commit_all(root, init_only=True)
        assert run_compile(mod, root, allow_dirty=False) == 1
        assert run_compile(mod, root, allow_dirty=True) == 0

    def test_dirty_guard_catches_gitignored_fragment(self, mod, tmp_path, git_commit_all):
        # `git status` never sees ignored untracked files; the guard is a
        # committed-content check instead, so a fragment hidden by
        # .git/info/exclude must be refused (it would be compiled into the
        # release and deleted) and must survive the refused run. Baseline is
        # committed FIRST so the ignored fragment is the sole dirty entry.
        root = make_repo(tmp_path)
        git_commit_all(root)
        (root / ".git" / "info").mkdir(exist_ok=True)
        (root / ".git" / "info" / "exclude").write_text("changelog.d/20260830-x.md\n")
        (root / "changelog.d" / "20260830-x.md").write_text(GOOD_FRAGMENT)
        assert run_compile(mod, root, allow_dirty=False) == 1
        assert (root / "changelog.d" / "20260830-x.md").exists()

    def test_dirty_guard_catches_status_hidden_untracked(self, mod, tmp_path, git_commit_all):
        # status.showUntrackedFiles=no hides untracked files from
        # `git status`; the committed-content guard must still refuse the
        # uncommitted fragment.
        root = make_repo(tmp_path)
        git_commit_all(root)
        git = shutil.which("git")
        subprocess.run(
            [git, "-C", str(root), "config", "status.showUntrackedFiles", "no"],
            check=True,
        )
        (root / "changelog.d" / "20260830-x.md").write_text(GOOD_FRAGMENT)
        assert run_compile(mod, root, allow_dirty=False) == 1
        assert (root / "changelog.d" / "20260830-x.md").exists()

    def test_dirty_guard_rejects_symlink_fragment(self, mod, tmp_path, git_commit_all):
        # A symlink at a fragment path is refused outright: read_text()
        # follows mutable target content, so it can never be certified as
        # committed bytes.
        root = make_repo(tmp_path)
        git_commit_all(root)
        target = root / "real-content.md"
        target.write_text(GOOD_FRAGMENT)
        (root / "changelog.d" / "20260830-x.md").symlink_to(target)
        assert run_compile(mod, root, allow_dirty=False) == 1

    def test_dirty_guard_rejects_modified_committed_fragment(self, mod, tmp_path, git_commit_all):
        # A committed fragment whose worktree bytes drifted from the HEAD
        # blob is refused (a status-free byte comparison).
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        git_commit_all(root)
        (root / "changelog.d" / "20260830-x.md").write_text(
            GOOD_FRAGMENT + "- uncommitted extra bullet\n"
        )
        assert run_compile(mod, root, allow_dirty=False) == 1

    def test_dirty_guard_ignores_dotfiles(self, mod, tmp_path, git_commit_all):
        # A stray .DS_Store must NOT trip the guard — dotfiles are never
        # compiled or deleted, mirroring check's dotfile rule.
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        git_commit_all(root)
        (root / "changelog.d" / ".DS_Store").write_bytes(b"\x00junk")
        assert run_compile(mod, root, allow_dirty=False) == 0

    def test_fragment_free_recovery_via_committed_stub(self, mod, tmp_path, git_commit_all):
        # The documented fragment-free release recovery: write an
        # ### Internal stub, COMMIT it, re-run WITHOUT --allow-dirty.
        root = make_repo(tmp_path)
        assert run_compile(mod, root, allow_dirty=False) == 1  # nothing to release
        (root / "changelog.d" / "20260830-stub.md").write_text(
            "### Internal\n- metadata-only re-release\n"
        )
        git_commit_all(root)
        assert run_compile(mod, root, allow_dirty=False) == 0

    def test_byte_stability_outside_edits(self, mod, tmp_path):
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        before = (root / "CHANGELOG.md").read_text()
        assert run_compile(mod, root) == 0
        after = (root / "CHANGELOG.md").read_text()
        # The compiled file is exactly the original with (a) the new section
        # inserted below the pointer block and (b) the new link line inserted
        # above the [1.2.0]: link — everything else byte-identical.
        head = before[: before.index("## [1.2.0]")]
        mid = before[before.index("## [1.2.0]") : before.index("[1.2.0]:")]
        tail = before[before.index("[1.2.0]:") :]
        section = after[len(head) : after.index("## [1.2.0]")]
        link = "[1.3.0]: https://github.com/x/y/compare/v1.2.0...v1.3.0\n"
        assert after == head + section + mid + link + tail
        assert section.startswith("## [1.3.0] - 2026-08-30\n")


# ---------------------------------------------------------------------------
# 3. Workflow pins (bounded slicing — never split on "pull_request:" alone)
# ---------------------------------------------------------------------------

REQUIRED_FILTER_ENTRIES = (
    "tests/test_changelog_fragments.py",
    "changelog.d/**",
    "CHANGELOG.md",
    ".claude/scripts/changelog_compile.py",
)


def _on_block(text):
    """The workflow's `on:` block: from the `on:` line to the next
    top-level (column-0) key."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "on:")
    end = next(
        (
            i
            for i, ln in enumerate(lines[start + 1 :], start + 1)
            if ln.rstrip() and ln[0] not in (" ", "\t", "#")
        ),
        len(lines),
    )
    return "".join(lines[start:end])


class TestWorkflowPins:
    @pytest.fixture(scope="class")
    def docs_tests_text(self):
        path = _REPO_ROOT / ".github" / "workflows" / "docs-tests.yml"
        if not path.exists():
            pytest.skip("docs-tests.yml not present")
        return path.read_text()

    @pytest.mark.parametrize("trigger", ["push:", "pull_request:"])
    def test_path_filters_cover_fragment_surfaces(self, docs_tests_text, trigger):
        on_block = _on_block(docs_tests_text)
        tstart = on_block.index(trigger)
        others = [
            on_block.index(t)
            for t in ("push:", "pull_request:", "schedule:", "workflow_dispatch:")
            if t != trigger and t in on_block and on_block.index(t) > tstart
        ]
        tblock = on_block[tstart : min(others)] if others else on_block[tstart:]
        for entry in REQUIRED_FILTER_ENTRIES:
            assert f"'{entry}'" in tblock or f'"{entry}"' in tblock or f"- {entry}" in tblock, (
                f"docs-tests.yml {trigger} paths filter is missing {entry!r} — "
                "a fragment-only PR would run no CI guard"
            )

    def test_doc_snippets_step_invokes_guard(self, docs_tests_text):
        jobs = docs_tests_text[docs_tests_text.index("\njobs:") :]
        doc_snippets = jobs[jobs.index("doc-snippets:") :]
        nxt = [
            doc_snippets.index(j)
            for j in ("sphinx-build:", "docs-deps-py39-smoke:")
            if j in doc_snippets
        ]
        block = doc_snippets[: min(nxt)] if nxt else doc_snippets
        assert "pytest tests/test_changelog_fragments.py" in block, (
            "docs-tests.yml doc-snippets job no longer runs the changelog "
            "fragment guard — the path filters alone are silently dead"
        )
