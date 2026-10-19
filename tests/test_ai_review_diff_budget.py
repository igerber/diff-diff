"""Tests for `tools/ai_review_diff_budget.py`.

`.github/workflows/ai_pr_review.yml` runs this filter (staged from BASE_SHA)
over the unified diff so the compiled Codex prompt stays under the
1,048,576-character input cap that killed the review of PR #861 (1.72M
compiled). Inline fixtures only; the skip-guard covers `rust-test.yml`'s
isolated-install matrix, which copies just `tests/` and has no `tools/`.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "ai_review_diff_budget.py"

if not SCRIPT_PATH.exists():
    pytest.skip(
        "tools/ai_review_diff_budget.py not present in working tree",
        allow_module_level=True,
    )


def _load():
    spec = importlib.util.spec_from_file_location("ai_review_diff_budget", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_block(path: str, body_lines: int, *, old_path: str | None = None) -> bytes:
    """A realistic per-file unified-diff block of roughly ``body_lines`` lines."""
    a = old_path or path
    head = (
        f"diff --git a/{a} b/{path}\n"
        f"index 0123456..89abcde 100644\n"
        f"--- a/{a}\n"
        f"+++ b/{path}\n"
        f"@@ -1,{body_lines} +1,{body_lines} @@\n"
    )
    body = "".join(f"+line {i} of {path}\n" for i in range(body_lines))
    return (head + body).encode()


@pytest.fixture(scope="module")
def mod():
    return _load()


class TestSplitBlocks:
    def test_splits_on_column_zero_header_only(self, mod):
        # A body line that CONTAINS "diff --git" carries a prefix char and must
        # not start a new block.
        trap = b"+diff --git a/trap b/trap\n diff --git a/trap2 b/trap2\n"
        patch = _file_block("diff_diff/a.py", 2) + trap + _file_block("tests/b.py", 1)
        preamble, blocks = mod.split_blocks(patch)
        assert preamble == b""
        assert [b.path for b in blocks] == ["diff_diff/a.py", "tests/b.py"]
        assert trap in blocks[0].data
        assert b"".join(b.data for b in blocks) == patch

    def test_preamble_kept(self, mod):
        patch = b"some preamble\n" + _file_block("x.py", 1)
        preamble, blocks = mod.split_blocks(patch)
        assert preamble == b"some preamble\n"
        assert len(blocks) == 1

    def test_deleted_file_uses_a_side_path(self, mod):
        block = (
            b"diff --git a/gone.py b/gone.py\n"
            b"deleted file mode 100644\n"
            b"--- a/gone.py\n"
            b"+++ /dev/null\n"
            b"@@ -1 +0,0 @@\n-x\n"
        )
        _, blocks = mod.split_blocks(block)
        assert blocks[0].path == "gone.py"

    def test_binary_block_path_comes_from_the_header(self, mod):
        block = b"diff --git a/img.png b/img.png\nBinary files a/img.png and b/img.png differ\n"
        _, blocks = mod.split_blocks(block)
        assert blocks[0].path == "img.png"

    def test_mode_only_block_with_space_b_slash_in_path(self, mod):
        block = b"diff --git a/x b/y.py b/x b/y.py\nold mode 100644\nnew mode 100755\n"
        _, blocks = mod.split_blocks(block)
        assert blocks[0].path == "x b/y.py"

    def test_pure_rename_uses_rename_to_and_keeps_source_tier(self, mod):
        block = (
            b"diff --git a/diff_diff/old.py b/diff_diff/new.py\n"
            b"similarity index 100%\nrename from diff_diff/old.py\nrename to diff_diff/new.py\n"
        )
        _, blocks = mod.split_blocks(block)
        assert blocks[0].path == "diff_diff/new.py"
        assert mod.tier(blocks[0].path) == 2

    def test_c_quoted_paths_are_unquoted(self, mod):
        new_file = (
            b'diff --git "a/caf\\303\\251 \\"x\\".py" "b/caf\\303\\251 \\"x\\".py"\n'
            b"new file mode 100644\n--- /dev/null\n"
            b'+++ "b/caf\\303\\251 \\"x\\".py"\n@@ -0,0 +1 @@\n+x\n'
        )
        binary = b'diff --git "a/tests/caf\\303\\251.png" "b/tests/caf\\303\\251.png"\nBinary files differ\n'
        _, blocks = mod.split_blocks(new_file + binary)
        assert [b.path for b in blocks] == ['caf\u00e9 "x".py', "tests/caf\u00e9.png"]
        assert mod.tier(blocks[1].path) == 0

    def test_quoted_header_with_escaped_quote_then_space_in_path(self, mod):
        # Path `diff_diff/end" x.py`: the a-side token contains `\" ` so a
        # naive split on `" "` picks the wrong boundary.
        mode_only = (
            b'diff --git "a/diff_diff/end\\" x.py" "b/diff_diff/end\\" x.py"\n'
            b"old mode 100644\nnew mode 100755\n"
        )
        binary = b'diff --git "a/diff_diff/end\\" x.png" "b/diff_diff/end\\" x.png"\nBinary files differ\n'
        _, blocks = mod.split_blocks(mode_only + binary)
        assert [b.path for b in blocks] == ['diff_diff/end" x.py', 'diff_diff/end" x.png']
        assert [mod.tier(b.path) for b in blocks] == [2, 2]
        out = mod.stub(blocks[0], "a" * 40, "b" * 40)
        assert b"""-- ':(literal)diff_diff/end" x.py'""" in out

    def test_c_unquote_escapes_and_octal(self, mod):
        assert mod._c_unquote(b'"a\\tb\\\\c\\"d\\303\\251"') == b'a\tb\\c"d\xc3\xa9'
        assert mod._c_unquote(b"plain") == b"plain"

    def test_empty_patch(self, mod):
        assert mod.split_blocks(b"") == (b"", [])


class TestTiering:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("tests/test_x.py", 0),
            ("tests/fixtures/big.json", 0),
            ("benchmarks/python/bench.py", 0),
            ("tools/foo.py", 0),
            (".claude/scripts/x.py", 0),
            ("docs/methodology/REGISTRY.md", 1),
            ("docs/v4-deprecations.yaml", 1),
            (".github/workflows/ci.yml", 1),
            ("README.md", 1),
            ("diff_diff/staggered.py", 2),
            ("rust/src/lib.rs", 2),
        ],
    )
    def test_tier(self, mod, path, expected):
        assert mod.tier(path) == expected


class TestFitToBudget:
    def test_in_budget_passthrough_is_byte_identical(self, mod):
        patch = _file_block("diff_diff/a.py", 50) + _file_block("tests/b.py", 50)
        assert mod.fit_to_budget(patch, len(patch)) == patch
        assert mod.fit_to_budget(patch, len(patch) + 1) == patch

    def test_drops_lowest_tier_largest_first(self, mod):
        src = _file_block("diff_diff/core.py", 400)  # tier 2, biggest
        doc = _file_block("docs/methodology/REGISTRY.md", 300)  # tier 1
        big_test = _file_block("tests/test_big.py", 200)  # tier 0, bigger
        small_test = _file_block("tests/test_small.py", 20)  # tier 0, smaller
        patch = src + doc + big_test + small_test
        budget = len(patch) - len(big_test) // 2  # dropping big_test alone suffices
        out = mod.fit_to_budget(patch, budget, "aaaaaaa", "bbbbbbb")
        assert len(out) <= budget
        assert src in out and doc in out and small_test in out
        assert big_test not in out
        assert b"[diff omitted for prompt budget] tests/test_big.py (" in out
        assert (
            b"git --no-pager diff --unified=5 aaaaaaa bbbbbbb -- ':(literal)tests/test_big.py'"
            in out
        )
        assert out.startswith(b"[prompt budget] 1 of 4 file diffs are omitted")

    def test_source_dropped_only_when_unavoidable(self, mod):
        src = _file_block("diff_diff/core.py", 400)
        test = _file_block("tests/test_big.py", 200)
        patch = src + test
        # Budget smaller than src alone: both must go.
        out = mod.fit_to_budget(patch, len(src) - 1, "aaaaaaa", "bbbbbbb")
        assert len(out) <= len(src) - 1
        assert b"tests/test_big.py (" in out and b"diff_diff/core.py (" in out
        # Budget that fits src + stub but not src + test: only the test goes.
        out2 = mod.fit_to_budget(patch, len(src) + 600, "aaaaaaa", "bbbbbbb")
        assert src in out2 and test not in out2

    def test_order_of_kept_and_stubbed_blocks_is_preserved(self, mod):
        blocks = [
            _file_block("diff_diff/a.py", 10),
            _file_block("tests/t1.py", 300),
            _file_block("diff_diff/b.py", 10),
        ]
        patch = b"".join(blocks)
        out = mod.fit_to_budget(patch, len(patch) - 10)
        ia = out.index(b"diff --git a/diff_diff/a.py")
        it = out.index(b"[diff omitted for prompt budget] tests/t1.py")
        ib = out.index(b"diff --git a/diff_diff/b.py")
        assert ia < it < ib

    def test_tiny_budget_emits_bounded_summary_only(self, mod):
        patch = _file_block("diff_diff/a.py", 10) + _file_block("tests/t.py", 10)
        out = mod.fit_to_budget(patch, 300)
        assert len(out) <= 300
        assert out.startswith(b"[prompt budget] the unified diff (")
        assert b"does not fit the remaining budget of 300 bytes" in out
        assert b"diff --git" not in out
        # The summary itself is truncated to the budget, down to empty.
        for budget in (0, 1, 17, 60):
            short = mod.fit_to_budget(patch, budget)
            assert len(short) <= budget
            assert short.startswith(b"[prompt budget] the unified diff ("[:budget])
        assert mod.fit_to_budget(patch, 0) == b""

    def test_no_headers_hard_truncates_within_budget(self, mod):
        patch = b"x" * 1000
        out = mod.fit_to_budget(patch, 100)
        assert len(out) <= 100
        assert out.endswith(b"[prompt budget] diff truncated here.\n")
        for budget in (0, 1, 10, 37, 38, 39):
            assert len(mod.fit_to_budget(patch, budget)) <= budget

    def test_non_shrinking_stubs_are_never_substituted(self, mod):
        """A stub larger than the hunks it replaces frees nothing; ten tiny test
        diffs must stay visible while the one large source diff is stubbed."""
        small = [_file_block(f"tests/t{i}.py", 1) for i in range(10)]
        big = _file_block("diff_diff/core.py", 60)
        patch = b"".join(small) + big
        stub_len = len(mod.stub(mod.split_blocks(small[0])[1][0], "a" * 40, "b" * 40))
        assert stub_len > len(small[0])
        budget = len(patch) - len(big) // 2
        out = mod.fit_to_budget(patch, budget, "a" * 40, "b" * 40)
        assert len(out) <= budget
        for blk in small:
            assert blk in out
        assert big not in out
        assert b"[diff omitted for prompt budget] diff_diff/core.py (" in out

    def test_stub_without_shas_has_no_git_hint(self, mod):
        patch = _file_block("tests/t.py", 100) + _file_block("diff_diff/a.py", 1)
        out = mod.fit_to_budget(patch, len(patch) - 1)
        assert b"[diff omitted for prompt budget] tests/t.py (" in out
        assert b"git --no-pager" not in out

    def test_path_with_quote_is_shell_quoted_in_hint(self, mod):
        patch = _file_block("tests/it's.py", 100) + _file_block("diff_diff/a.py", 1)
        out = mod.fit_to_budget(patch, len(patch) - 1, "aaaaaaa", "bbbbbbb")
        assert b"-- ':(literal)tests/it'\\''s.py'" in out

    def test_output_never_exceeds_budget_across_sizes(self, mod):
        blocks = [_file_block(f"tests/t{i}.py", 5 + 37 * i) for i in range(12)]
        blocks += [_file_block(f"diff_diff/m{i}.py", 50 + 11 * i) for i in range(6)]
        blocks += [_file_block(f"docs/d{i}.md", 30 * i) for i in range(5)]
        patch = b"".join(blocks)
        for budget in list(range(0, 400)) + list(range(400, len(patch) + 1, 997)):
            out = mod.fit_to_budget(patch, budget, "aaaaaaa", "bbbbbbb")
            assert len(out) <= budget, budget
        assert mod.fit_to_budget(patch, len(patch)) == patch


class TestCli:
    def _run(self, args, cwd):
        return subprocess.run(
            [sys.executable, "-I", str(SCRIPT_PATH), *args],
            cwd=cwd,
            capture_output=True,
            check=False,
        )

    def test_cli_passthrough_and_output_file(self, tmp_path):
        patch = _file_block("diff_diff/a.py", 3)
        src = tmp_path / "d.patch"
        src.write_bytes(patch)
        out = tmp_path / "o.patch"
        r = self._run(["--diff", str(src), "--budget", "99999", "-o", str(out)], tmp_path)
        assert r.returncode == 0, r.stderr
        assert out.read_bytes() == patch
        r2 = self._run(["--diff", str(src), "--budget", "99999"], tmp_path)
        assert r2.returncode == 0 and r2.stdout == patch

    def test_cli_trims_with_shas(self, tmp_path):
        patch = _file_block("tests/t.py", 200) + _file_block("diff_diff/a.py", 2)
        src = tmp_path / "d.patch"
        src.write_bytes(patch)
        r = self._run(
            [
                "--diff",
                str(src),
                "--budget",
                str(len(patch) - 1),
                "--base-sha",
                "0" * 40,
                "--head-sha",
                "f" * 40,
            ],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        assert len(r.stdout) <= len(patch) - 1
        assert b"[diff omitted for prompt budget] tests/t.py (" in r.stdout
        assert b"diff --git a/diff_diff/a.py" in r.stdout

    @pytest.mark.parametrize("bad", ["main", "$(id)", "abc;rm", "ZZZZZZZ"])
    def test_cli_rejects_non_hex_sha(self, tmp_path, bad):
        src = tmp_path / "d.patch"
        src.write_bytes(_file_block("x.py", 1))
        r = self._run(["--diff", str(src), "--budget", "10", "--base-sha", bad], tmp_path)
        assert r.returncode == 2
        assert b"not a hex object id" in r.stderr

    def test_cli_rejects_negative_budget(self, tmp_path):
        src = tmp_path / "d.patch"
        src.write_bytes(_file_block("x.py", 1))
        r = self._run(["--diff", str(src), "--budget", "-1"], tmp_path)
        assert r.returncode == 2


class TestWorkflowWiring:
    """The workflow must stage the filter from BASE_SHA and append the
    unified diff LAST under a budget derived from the compiled prompt size
    (so notebook prose, PR text and the previous review are all counted).
    The staging/exec ordering contract itself is enforced by
    `tests/test_openai_review.py::TestWorkflowDoesNotExecutePRHeadCode`."""

    WF = REPO_ROOT / ".github" / "workflows" / "ai_pr_review.yml"

    @pytest.fixture
    def text(self):
        if not self.WF.exists():
            pytest.skip("workflow not found")
        return self.WF.read_text()

    def test_filter_is_staged_from_base_sha(self, text):
        assert (
            'git show "${BASE_SHA}":tools/ai_review_diff_budget.py > /tmp/ai_review_diff_budget.py'
            in text
        )

    def test_budget_is_derived_from_the_compiled_prompt(self, text):
        live = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        assert any(ln.strip() == "PROMPT_BUDGET=1000000" for ln in live), (
            "PROMPT_BUDGET must stay a whole-prompt byte target below the "
            "1,048,576-character codex cap"
        )
        assert any('DIFF_BUDGET=$(( PROMPT_BUDGET - $(wc -c < "$PROMPT") ))' in ln for ln in live)
        assert any(
            "python3 /tmp/ai_review_diff_budget.py --diff /tmp/ai-review-diff.patch" in ln
            for ln in live
        )

    def test_unified_diff_is_appended_after_notebook_prose(self, text):
        # The diff must come after the LAST prose wrapper so the budget sees it.
        prose_end = text.rfind('echo "</notebook-prose>"')
        diff_header = text.rfind('echo "Unified diff (context=5):"')
        assert 0 < prose_end < diff_header
        assert text.find("- name: Run Codex") > diff_header

    def test_bootstrap_fallback_appends_untrimmed(self, text):
        assert 'cat /tmp/ai-review-diff.patch >> "$PROMPT"' in text
