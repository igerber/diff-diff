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
from pathlib import Path

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
    tmp_path.mkdir(parents=True, exist_ok=True)
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
            "20260830-topic-.md",
            "20260830-topic--detail.md",
            "20260830--topic.md",
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
            # CommonMark accepts ATX headings indented 1-3 spaces; they must
            # not slip through as "continuation" lines (they render as real
            # headings the compiler's column-zero scanners never see).
            ("### Fixed\n ## [9.9.9] - 2099-01-01\n- entry\n", "indented Markdown heading"),
            ("### Fixed\n  ## smuggled\n- entry\n", "indented Markdown heading"),
            ("### Fixed\n   ## smuggled\n- entry\n", "indented Markdown heading"),
            ("### Fixed\n- ok\n  ### Unknown\n", "indented Markdown heading"),
            # CommonMark registers link definitions inside list items, and
            # the FIRST definition wins once compiled - a version-link-like
            # construct has no place in a release-note fragment.
            ("### Fixed\n- [1.3.0]:https://evil.example/wrong\n", "version-link-like"),
            ("### Fixed\n- ok\n  - [1.3.0]: https://evil.example\n", "version-link-like"),
            ("### Fixed\n- see [ 1.3.0 ]: for details\n", "version-link-like"),
            # check<->compile parity: everything the assembled-output guards
            # refuse must fail at PR time, not at release time.
            ("### Documentation\n- Example:\n  ```python\n  print(1)\n  ```\n", "block context"),
            ("### Fixed\n- x, see\n  <pre>\n  y\n  </pre>\n", "block context"),
            ("### Fixed\n- see [1.3.0\n  ]: for details\n", "version-link-like"),
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

    def test_exit4_rejects_orphan_bullet_before_category(self, mod, tmp_path):
        # Any-header + any-bullet independently is not enough: an orphan
        # bullet above a (now empty) valid category is a state compile
        # could not have produced - the section grammar must reject it.
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15\n\n### Added\n- old entry\n",
            "## [1.2.0] - 2026-01-15\n\n- orphan bullet\n\n### Added\n",
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_exit4_rejects_bullet_only_under_unknown_category(self, mod, tmp_path):
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15\n\n### Added\n- old entry\n",
            "## [1.2.0] - 2026-01-15\n\n### Added\n\n### Unknown\n- smuggled\n",
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    @pytest.mark.parametrize(
        "smuggled",
        [
            " ## [9.9.9] - 2099-01-01",
            "   ## [9.9.9] - 2099-01-01",
            " [9.9.9]: https://github.com/x/y/compare/v1.2.0...v9.9.9",
            "##  [9.9.9] - 2099-01-01",
            "##\t[9.9.9] - 2099-01-01",
            "### [9.9.9] - 2099-01-01",
            "[ 9.9.9 ]: https://github.com/x/y/compare/vA...vB",
            "- [9.9.9]:https://evil.example/wrong",
            "  - [9.9.9]: https://evil.example/wrong",
        ],
        ids=[
            "indented-header-1sp",
            "indented-header-3sp",
            "indented-link-def",
            "double-space-after-hashes",
            "tab-after-hashes",
            "h3-release-like",
            "whitespace-normalized-label",
            "list-item-link-def",
            "nested-list-link-def",
        ],
    )
    def test_noncanonical_changelog_constructs_refused(self, mod, tmp_path, smuggled):
        # CommonMark renders headings indented 1-3 spaces, with a tab or
        # multiple spaces after the hashes, and at any hash depth - all
        # invisible to column-zero/single-space-anchored duplicate scans.
        # Every rendering-but-noncanonical shape must refuse, on BOTH the
        # fresh and exit-4 paths.
        changelog = MINIMAL_CHANGELOG.replace(
            "### Added\n- old entry\n",
            f"### Added\n- old entry\n{smuggled}\n",
        )
        fresh = make_repo(
            tmp_path / "fresh", changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT}
        )
        assert run_compile(mod, fresh) == 1
        # Refused before writing: no new section, fragment untouched.
        assert "## [1.3.0]" not in (fresh / "CHANGELOG.md").read_text()
        assert (fresh / "changelog.d" / "20260830-x.md").exists()
        exit4 = make_repo(tmp_path / "exit4", changelog=changelog)
        assert run_compile(mod, exit4, version="1.2.0") == 1

    @pytest.mark.parametrize(
        "wrapper",
        [
            "```\n{d}\n```",
            "<pre>\n{d}\n</pre>",
            # Unterminated blocks stay open through EOF in CommonMark and
            # would swallow the terminal link block itself.
            "```\n{d}",
            "<pre>\n{d}",
        ],
        ids=["fenced-code", "html-block", "unterminated-fence", "unterminated-pre"],
    )
    def test_definition_inside_block_context_refused(self, mod, tmp_path, wrapper):
        # A column-zero '[1.2.0]: url' inside a fenced/HTML block LOOKS
        # canonical to line-based scans but does not render - the anchor
        # search could insert the new link inside that block. Every
        # definition must live in the terminal contiguous link block.
        d = "[1.2.0]: https://github.com/x/y/compare/v1.1.0...v1.2.0"
        changelog = MINIMAL_CHANGELOG.replace(
            "### Added\n- old entry\n",
            "### Added\n- old entry\n\n" + wrapper.format(d=d) + "\n",
        ).replace(
            d + "\n", ""
        )  # the def now exists ONLY inside the block
        fresh = make_repo(
            tmp_path / "fresh", changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT}
        )
        assert run_compile(mod, fresh) == 1
        assert "## [1.3.0]" not in (fresh / "CHANGELOG.md").read_text()
        assert (fresh / "changelog.d" / "20260830-x.md").exists()
        exit4 = make_repo(tmp_path / "exit4", changelog=changelog)
        assert run_compile(mod, exit4, version="1.2.0") == 1

    def test_html_block_touching_link_block_refused(self, mod, tmp_path):
        # A type-6 HTML block ('<div>') persists until a blank line - one
        # sitting directly above the definitions would swallow them, so the
        # terminal run must be preceded by a blank line.
        changelog = MINIMAL_CHANGELOG.replace(
            "- older entry\n\n[1.2.0]:", "- older entry\n<div>\n[1.2.0]:"
        )
        fresh = make_repo(
            tmp_path / "fresh", changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT}
        )
        assert run_compile(mod, fresh) == 1
        assert (fresh / "changelog.d" / "20260830-x.md").exists()
        exit4 = make_repo(tmp_path / "exit4", changelog=changelog)
        assert run_compile(mod, exit4, version="1.2.0") == 1

    def test_nospace_link_definition_blocks_fresh_target(self, mod, tmp_path):
        # CommonMark allows '[x]:dest' with no whitespace after the colon,
        # and resolves a label to its FIRST definition - a hidden no-space
        # definition for the target must refuse the fresh compile, or the
        # appended canonical definition would be silently outranked.
        changelog = MINIMAL_CHANGELOG.replace(
            "[1.2.0]:",
            "[1.3.0]:https://github.com/x/y/compare/vWRONG...vWRONG\n[1.2.0]:",
        )
        root = make_repo(tmp_path, changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root) == 1
        assert "## [1.3.0]" not in (root / "CHANGELOG.md").read_text()
        assert (root / "changelog.d" / "20260830-x.md").exists()

    def test_nospace_duplicate_link_definition_blocks_exit4(self, mod, tmp_path):
        changelog = MINIMAL_CHANGELOG.replace(
            "[1.2.0]:",
            "[1.2.0]:https://github.com/x/y/compare/vWRONG...vWRONG\n[1.2.0]:",
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.2.0") == 1

    def test_ls_tree_enumeration_failure_fails_closed(
        self, mod, tmp_path, monkeypatch, git_commit_all
    ):
        # A failed HEAD enumeration means the deleted-fragment guard cannot
        # run; that must refuse the compile, never silently skip the check.
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        git_commit_all(root)
        before = (root / "CHANGELOG.md").read_text()
        real_run = mod.subprocess.run

        def failing_run(cmd, *a, **k):
            if "ls-tree" in cmd and "-r" in cmd:
                # A genuine nonzero-returncode CompletedProcess.
                return real_run(["git", "--no-such-flag"], capture_output=True, text=True)
            return real_run(cmd, *a, **k)

        monkeypatch.setattr(mod.subprocess, "run", failing_run)
        assert run_compile(mod, root, allow_dirty=False) == 1
        # Nothing written, nothing consumed.
        assert (root / "CHANGELOG.md").read_text() == before
        assert (root / "changelog.d" / "20260830-x.md").exists()

    def test_indented_heading_fragment_cannot_compile(self, mod, tmp_path):
        frag = "### Fixed\n ## [9.9.9] - 2099-01-01\n- entry\n"
        root = make_repo(tmp_path, fragments={"20260830-x.md": frag})
        assert run_compile(mod, root) == 1
        # Nothing consumed, nothing written.
        assert (root / "changelog.d" / "20260830-x.md").exists()
        assert "9.9.9" not in (root / "CHANGELOG.md").read_text()

    def test_exit4_rejects_indented_heading_in_section(self, mod, tmp_path):
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15\n\n### Added\n- old entry\n",
            "## [1.2.0] - 2026-01-15\n\n### Added\n- old entry\n ## smuggled\n",
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

    def test_previous_version_mismatch_leaves_state_untouched(self, mod, tmp_path):
        # bump-version passes its package-metadata OLD_VERSION; a drifted
        # changelog (latest release != OLD_VERSION) must refuse before any
        # write or fragment deletion.
        root = make_repo(tmp_path, fragments={"20260830-x.md": GOOD_FRAGMENT})
        before = (root / "CHANGELOG.md").read_text()
        rc = mod.run_compile(root, "1.3.0", "2026-08-30", True, previous_version="9.9.9")
        assert rc == 1
        assert (root / "CHANGELOG.md").read_text() == before
        assert (root / "changelog.d" / "20260830-x.md").exists()
        # Matching previous-version compiles normally.
        assert mod.run_compile(root, "1.3.0", "2026-08-30", True, previous_version="1.2.0") == 0
        # Interrupted-bump recovery: re-running with the SAME (now stale)
        # previous-version must reach the idempotent exit-4 path, not the
        # drift rejection (compile finished; package metadata not yet
        # bumped).
        assert mod.run_compile(root, "1.3.0", "2026-08-30", True, previous_version="1.2.0") == 4
        # A completed bump re-run (metadata already at the target) also
        # certifies; an unrelated value is drift.
        assert mod.run_compile(root, "1.3.0", "2026-08-30", True, previous_version="1.3.0") == 4
        assert mod.run_compile(root, "1.3.0", "2026-08-30", True, previous_version="9.9.9") == 1

    def test_dateless_prev_anchor_rejected_legacy_exempt(self, mod, tmp_path):
        # The ANCHOR (latest release) must carry a canonical date; older
        # legacy headers without dates stay legal (the real CHANGELOG has
        # pre-convention dateless releases).
        frag = {"20260830-x.md": GOOD_FRAGMENT}
        changelog = MINIMAL_CHANGELOG.replace("## [1.2.0] - 2026-01-15", "## [1.2.0]")
        root = make_repo(tmp_path / "dateless", changelog=changelog, fragments=frag)
        assert run_compile(mod, root, version="1.3.0") == 1
        # Impossible date on the anchor is equally rejected.
        changelog = MINIMAL_CHANGELOG.replace("2026-01-15", "2026-99-99")
        root = make_repo(tmp_path / "impossible", changelog=changelog, fragments=frag)
        assert run_compile(mod, root, version="1.3.0") == 1
        # A dateless OLDER header does not block (legacy exemption).
        changelog = MINIMAL_CHANGELOG.replace("## [1.1.0] - 2026-01-01", "## [1.1.0]")
        root = make_repo(tmp_path / "legacy", changelog=changelog, fragments=frag)
        assert run_compile(mod, root, version="1.3.0") == 0

    def test_alternate_whitespace_target_link_detected(self, mod, tmp_path):
        # A pre-existing target link with extra whitespace after ':' must
        # still trip the duplicate-definition refusal.
        changelog = MINIMAL_CHANGELOG + "[1.3.0]:   https://example.com/compare/v1.2.0...v1.3.0\n"
        root = make_repo(tmp_path, changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root, version="1.3.0") == 1

    def test_preexisting_target_link_definition_rejected(self, mod, tmp_path):
        # A '[X.Y.Z]:' link definition without its release header is an
        # inconsistent state; compiling would emit a DUPLICATE definition.
        changelog = MINIMAL_CHANGELOG + "[1.3.0]: https://example.com/compare/v1.2.0...v1.3.0\n"
        root = make_repo(tmp_path, changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root, version="1.3.0") == 1
        assert (root / "changelog.d" / "20260830-x.md").exists()

    def test_exit4_rejects_duplicate_link_definitions(self, mod, tmp_path):
        # One correct + one wrong definition for the target: the duplicate
        # check now runs BEFORE the existing-target path, so exit 4 cannot
        # certify the ambiguous state.
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15",
            "## [1.3.0] - 2026-01-16\n\n### Fixed\n- prior.\n\n## [1.2.0] - 2026-01-15",
        )
        changelog = changelog.replace(
            "[1.2.0]:",
            "[1.3.0]: https://example.com/compare/v1.2.0...v1.3.0\n"
            "[1.3.0]: https://evil.example/compare/v1.2.0...v1.3.0\n[1.2.0]:",
            1,
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.3.0") == 1

    def test_exit4_requires_predecessor_link(self, mod, tmp_path):
        # No link definition for the predecessor: never fall back to
        # accepting any base - refuse (not a compiler-produced state).
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15",
            "## [1.3.0] - 2026-01-16\n\n### Fixed\n- prior.\n\n## [1.2.0] - 2026-01-15",
        )
        changelog = changelog.replace(
            "[1.2.0]:", "[1.3.0]: https://example.com/compare/v1.2.0...v1.3.0\n[oops-1.2.0]:", 1
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.3.0") == 1

    def test_exit4_rejects_wrong_base_target_link(self, mod, tmp_path):
        # Correct predecessor/target versions but an UNRELATED repository
        # base: exit 4 must not certify this as already-compiled.
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15",
            "## [1.3.0] - 2026-01-16\n\n### Fixed\n- prior.\n\n## [1.2.0] - 2026-01-15",
        )
        changelog = changelog.replace(
            "[1.2.0]:",
            "[1.3.0]: https://evil.example/compare/v1.2.0...v1.3.0\n[1.2.0]:",
            1,
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.3.0") == 1

    def test_malformed_release_header_rejected(self, mod, tmp_path):
        # A release-LIKE header the canonical grammar does not match (extra
        # suffix) is invisible to _existing_headers, so compiling that
        # version would otherwise add a SECOND canonical section beside the
        # malformed one; the compiler must refuse before touching anything.
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.2.0] - 2026-01-15", "## [1.2.0] - 2026-01-15 draft"
        )
        root = make_repo(tmp_path, changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT})
        before = (root / "CHANGELOG.md").read_text()
        assert run_compile(mod, root, version="1.2.0") == 1
        assert (root / "changelog.d" / "20260830-x.md").exists()
        assert (root / "CHANGELOG.md").read_text() == before  # untouched

    def test_mixed_canonical_and_malformed_duplicate_rejected(self, mod, tmp_path):
        # Canonical 1.2.0 + a malformed 1.2.0 twin: the malformed-header
        # rejection fires (the duplicate check alone cannot see the twin).
        changelog = MINIMAL_CHANGELOG.replace(
            "## [1.1.0] - 2026-01-01\n",
            "## [1.2.0] - 2026-01-15 rc1\n\n### Fixed\n- twin\n\n" "## [1.1.0] - 2026-01-01\n",
        )
        root = make_repo(tmp_path, changelog=changelog)
        assert run_compile(mod, root, version="1.3.0") == 1

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

    def test_dirty_guard_catches_deleted_tracked_fragment(self, mod, tmp_path, git_commit_all):
        # A committed fragment DELETED from the worktree must refuse: the
        # compile would otherwise ship only the surviving fragment while
        # the git diff deletes both - silently losing a release note.
        root = make_repo(
            tmp_path,
            fragments={"20260830-a.md": GOOD_FRAGMENT, "20260830-b.md": GOOD_FRAGMENT},
        )
        git_commit_all(root)
        (root / "changelog.d" / "20260830-a.md").unlink()
        before = (root / "CHANGELOG.md").read_text()
        assert run_compile(mod, root, allow_dirty=False) == 1
        assert (root / "CHANGELOG.md").read_text() == before
        assert (root / "changelog.d" / "20260830-b.md").exists()

    def test_compile_rolls_back_on_unlink_failure(self, mod, tmp_path, monkeypatch):
        # Transactionality: if fragment deletion fails after CHANGELOG.md
        # was written, both the changelog and every fragment are restored.
        root = make_repo(
            tmp_path,
            fragments={"20260830-a.md": GOOD_FRAGMENT, "20260830-b.md": GOOD_FRAGMENT},
        )
        before = (root / "CHANGELOG.md").read_text()
        real_unlink = Path.unlink
        calls = {"n": 0}

        def failing_unlink(self, *a, **k):
            if self.suffix == ".md" and self.parent.name == "changelog.d":
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OSError("simulated unlink failure")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", failing_unlink)
        assert run_compile(mod, root, allow_dirty=True) == 1
        assert (root / "CHANGELOG.md").read_text() == before
        assert (root / "changelog.d" / "20260830-a.md").exists()
        assert (root / "changelog.d" / "20260830-b.md").exists()

    def test_second_release_over_tag_link_anchor(self, mod, tmp_path):
        # A repo whose sole prior release carries the conventional
        # releases/tag link must be able to compile its second release.
        changelog = (
            "# Changelog\n\n## [Unreleased]\n\n"
            + POINTER
            + "\n\n## [1.0.0] - 2026-01-01\n\n### Added\n- first.\n\n"
            "[1.0.0]: https://example.com/releases/tag/v1.0.0\n"
        )
        root = make_repo(tmp_path, changelog=changelog, fragments={"20260830-x.md": GOOD_FRAGMENT})
        assert run_compile(mod, root, version="1.1.0") == 0
        out = (root / "CHANGELOG.md").read_text()
        assert "[1.1.0]: https://example.com/compare/v1.0.0...v1.1.0" in out

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
