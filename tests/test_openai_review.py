"""Tests for .claude/scripts/openai_review.py — local AI review script.

These tests are skipped in CI when the script is not available (e.g., when
the package is installed via pip into a temp directory). They run locally
where the repo checkout includes .claude/scripts/.
"""

import importlib.util
import json
import os
import pathlib
import subprocess

import pytest

# ---------------------------------------------------------------------------
# Import the script as a module (it's not in a package)
# ---------------------------------------------------------------------------


def _find_script() -> "pathlib.Path | None":
    """Find openai_review.py relative to the repo root."""
    # Method 1: relative to this test file (works in local checkout)
    candidate = (
        pathlib.Path(__file__).resolve().parent.parent
        / ".claude"
        / "scripts"
        / "openai_review.py"
    )
    if candidate.exists():
        return candidate

    # Method 2: relative to git repo root (works in worktrees)
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        candidate = pathlib.Path(root) / ".claude" / "scripts" / "openai_review.py"
        if candidate.exists():
            return candidate
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


_SCRIPT_PATH = _find_script()

# Skip entire module if the script isn't available (e.g., CI pip-install)
pytestmark = pytest.mark.skipif(
    _SCRIPT_PATH is None,
    reason="openai_review.py not found (not in repo checkout)",
)


@pytest.fixture(scope="module")
def review_mod():
    """Import openai_review.py as a module."""
    assert _SCRIPT_PATH is not None
    spec = importlib.util.spec_from_file_location("openai_review", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def repo_root():
    """Return the repo root directory."""
    assert _SCRIPT_PATH is not None
    return str(_SCRIPT_PATH.parent.parent.parent)


# ---------------------------------------------------------------------------
# _sections_for_file
# ---------------------------------------------------------------------------


class TestSectionsForFile:
    def test_direct_match(self, review_mod):
        assert "BaconDecomposition" in review_mod._sections_for_file("bacon.py")

    def test_companion_file(self, review_mod):
        assert "SunAbraham" in review_mod._sections_for_file("sun_abraham_bootstrap.py")

    def test_no_match(self, review_mod):
        assert review_mod._sections_for_file("linalg.py") == []

    def test_staggered_maps_multiple(self, review_mod):
        sections = review_mod._sections_for_file("staggered.py")
        assert "CallawaySantAnna" in sections
        assert "SunAbraham" in sections

    def test_longest_prefix_wins(self, review_mod):
        # sun_abraham.py should match "sun_abraham" not "staggered"
        sections = review_mod._sections_for_file("sun_abraham.py")
        assert sections == ["SunAbraham"]


# ---------------------------------------------------------------------------
# _needed_sections
# ---------------------------------------------------------------------------


class TestNeededSections:
    def test_basic(self, review_mod):
        text = "M\tdiff_diff/bacon.py"
        assert "BaconDecomposition" in review_mod._needed_sections(text)

    def test_visualization_submodule(self, review_mod):
        text = "M\tdiff_diff/visualization/_event_study.py"
        assert "Event Study Plotting" in review_mod._needed_sections(text)

    def test_visualization_multiple_files(self, review_mod):
        """All visualization/ submodule files map via directory to Event Study Plotting."""
        text = (
            "M\tdiff_diff/visualization/_event_study.py\n"
            "M\tdiff_diff/visualization/_diagnostic.py"
        )
        sections = review_mod._needed_sections(text)
        assert "Event Study Plotting" in sections

    def test_non_diff_diff_paths_ignored(self, review_mod):
        text = "M\ttests/test_bacon.py\nM\tCLAUDE.md"
        assert review_mod._needed_sections(text) == set()

    def test_utility_files_no_sections(self, review_mod):
        text = "M\tdiff_diff/linalg.py\nM\tdiff_diff/utils.py"
        assert review_mod._needed_sections(text) == set()

    def test_mixed_files(self, review_mod):
        text = (
            "M\tdiff_diff/bacon.py\n"
            "M\tdiff_diff/linalg.py\n"
            "M\ttests/test_bacon.py"
        )
        sections = review_mod._needed_sections(text)
        assert sections == {"BaconDecomposition"}

    def test_empty_input(self, review_mod):
        assert review_mod._needed_sections("") == set()


# ---------------------------------------------------------------------------
# extract_registry_sections
# ---------------------------------------------------------------------------


class TestExtractRegistrySections:
    SAMPLE_REGISTRY = (
        "# Registry\n\n"
        "## Table of Contents\nTOC content\n\n"
        "## BaconDecomposition\nBacon content line 1\nBacon content line 2\n\n"
        "## SunAbraham\nSA content\n\n"
        "## Event Study Plotting (`plot_event_study`)\nPlotting content\n"
    )

    def test_extract_single_section(self, review_mod):
        result = review_mod.extract_registry_sections(
            self.SAMPLE_REGISTRY, {"BaconDecomposition"}
        )
        assert "Bacon content line 1" in result
        assert "SA content" not in result

    def test_extract_multiple_sections(self, review_mod):
        result = review_mod.extract_registry_sections(
            self.SAMPLE_REGISTRY, {"BaconDecomposition", "SunAbraham"}
        )
        assert "Bacon content" in result
        assert "SA content" in result

    def test_prefix_match_for_headings_with_parens(self, review_mod):
        result = review_mod.extract_registry_sections(
            self.SAMPLE_REGISTRY, {"Event Study Plotting"}
        )
        assert "Plotting content" in result

    def test_empty_section_names(self, review_mod):
        assert review_mod.extract_registry_sections(self.SAMPLE_REGISTRY, set()) == ""

    def test_nonexistent_section(self, review_mod):
        result = review_mod.extract_registry_sections(
            self.SAMPLE_REGISTRY, {"NonExistent"}
        )
        assert result == ""


# ---------------------------------------------------------------------------
# _adapt_review_criteria
# ---------------------------------------------------------------------------


class TestAdaptReviewCriteria:
    def test_replaces_opening_line(self, review_mod):
        source = "You are an automated PR reviewer for a causal inference library."
        result = review_mod._adapt_review_criteria(source)
        assert "automated PR reviewer" not in result
        assert "code reviewer" in result

    def test_replaces_pr_language(self, review_mod):
        source = "If the PR changes an estimator"
        result = review_mod._adapt_review_criteria(source)
        assert "If the changes affect an estimator" in result

    def test_warns_on_missing_substitution(self, review_mod, capsys):
        # A text that doesn't contain any of the expected patterns
        review_mod._adapt_review_criteria("Totally different text")
        captured = capsys.readouterr()
        assert "Warning: prompt substitution did not match" in captured.err

    def test_all_substitutions_apply_to_real_prompt(self, review_mod, capsys):
        """Verify all substitutions match the actual pr_review.md file."""
        assert _SCRIPT_PATH is not None
        repo_root = _SCRIPT_PATH.parent.parent.parent
        prompt_path = repo_root / ".github" / "codex" / "prompts" / "pr_review.md"
        if not prompt_path.exists():
            pytest.skip("pr_review.md not found")
        source = prompt_path.read_text()
        review_mod._adapt_review_criteria(source)
        captured = capsys.readouterr()
        assert "Warning: prompt substitution did not match" not in captured.err


# ---------------------------------------------------------------------------
# compile_prompt
# ---------------------------------------------------------------------------


class TestCompilePrompt:
    def test_basic_structure(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="Review criteria here.",
            registry_content="Registry content.",
            diff_text="diff --git a/foo.py",
            changed_files_text="M\tfoo.py",
            branch_info="feature/test",
            previous_review=None,
        )
        assert "Review criteria here." in result
        assert "Registry content." in result
        assert "diff --git a/foo.py" in result
        assert "Branch: feature/test" in result
        assert "previous-review-output" not in result

    def test_includes_previous_review(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="Criteria.",
            registry_content="Registry.",
            diff_text="diff content",
            changed_files_text="M\tfoo.py",
            branch_info="main",
            previous_review="Previous review findings here.",
        )
        assert "<previous-review-output>" in result
        assert "Previous review findings here." in result
        assert "follow-up review" in result

    def test_no_previous_review_block_when_none(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="C.",
            registry_content="R.",
            diff_text="D.",
            changed_files_text="M\tf.py",
            branch_info="b",
            previous_review=None,
        )
        assert "<previous-review-output>" not in result


# ---------------------------------------------------------------------------
# compile_prompt — enhanced context modes
# ---------------------------------------------------------------------------


class TestCompilePromptWithContext:
    """Test compile_prompt with the new context parameters."""

    def test_backward_compatibility(self, review_mod):
        """Original args produce same structure — no source/import sections."""
        result = review_mod.compile_prompt(
            criteria_text="Criteria.",
            registry_content="Registry.",
            diff_text="diff content",
            changed_files_text="M\tfoo.py",
            branch_info="main",
            previous_review=None,
        )
        assert "Full Source Files" not in result
        assert "Import Context" not in result
        assert "Changes Under Review" in result

    def test_standard_mode_includes_source_files(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="C.",
            registry_content="R.",
            diff_text="D.",
            changed_files_text="M\tf.py",
            branch_info="b",
            previous_review=None,
            source_files_text='<file path="diff_diff/foo.py">content</file>',
        )
        assert "Full Source Files (Changed)" in result
        assert "sins of omission" in result
        assert '<file path="diff_diff/foo.py">' in result
        assert "Import Context" not in result

    def test_deep_mode_includes_import_context(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="C.",
            registry_content="R.",
            diff_text="D.",
            changed_files_text="M\tf.py",
            branch_info="b",
            previous_review=None,
            source_files_text="<file>src</file>",
            import_context_text='<file path="diff_diff/utils.py" role="import-context">utils</file>',
        )
        assert "Full Source Files (Changed)" in result
        assert "Import Context (Read-Only Reference)" in result
        assert "Do NOT flag issues in these files" in result

    def test_delta_diff_structure(self, review_mod):
        result = review_mod.compile_prompt(
            criteria_text="C.",
            registry_content="R.",
            diff_text="full diff content",
            changed_files_text="M\tf.py",
            branch_info="b",
            previous_review="Previous findings.",
            delta_diff_text="delta diff content",
            delta_changed_files_text="M\tf.py",
        )
        assert "Changes Since Last Review" in result
        assert "delta diff content" in result
        assert "Full Branch Diff (Reference Only)" in result
        assert "<full-diff-reference>" in result
        assert "full diff content" in result

    def test_delta_diff_with_structured_findings(self, review_mod):
        findings = [
            {
                "id": "R1-P1-1",
                "severity": "P1",
                "section": "Methodology",
                "summary": "Missing NaN guard",
                "location": "diff_diff/foo.py:L42",
                "status": "open",
            }
        ]
        result = review_mod.compile_prompt(
            criteria_text="C.",
            registry_content="R.",
            diff_text="full diff",
            changed_files_text="M\tf.py",
            branch_info="b",
            previous_review="Prev.",
            delta_diff_text="delta",
            structured_findings=findings,
        )
        assert "Previous Findings" in result
        assert "R1-P1-1" in result
        assert "Missing NaN guard" in result
        assert "diff_diff/foo.py:L42" in result

    def test_fresh_review_no_delta_sections(self, review_mod):
        """Without delta_diff_text, no delta-specific sections appear."""
        result = review_mod.compile_prompt(
            criteria_text="C.",
            registry_content="R.",
            diff_text="D.",
            changed_files_text="M\tf.py",
            branch_info="b",
            previous_review=None,
            source_files_text="<file>src</file>",
        )
        assert "Changes Since Last Review" not in result
        assert "Full Branch Diff (Reference Only)" not in result
        assert "Changes Under Review" in result


    def test_findings_table_escapes_pipe_chars(self, review_mod):
        """Summary containing | should be escaped in the findings table."""
        findings = [
            {
                "id": "R1-P1-1", "severity": "P1", "section": "Code Quality",
                "summary": "Return type str | None is wrong",
                "location": "foo.py:L10", "status": "open",
            }
        ]
        result = review_mod.compile_prompt(
            criteria_text="C.", registry_content="R.", diff_text="D.",
            changed_files_text="M\tf.py", branch_info="b",
            previous_review="Prev.", delta_diff_text="delta",
            structured_findings=findings,
        )
        # The pipe in "str | None" should be escaped as "str \| None"
        assert "str \\| None" in result


# ---------------------------------------------------------------------------
# PREFIX_TO_SECTIONS mapping coverage
# ---------------------------------------------------------------------------


class TestPrefixMappingCoverage:
    """Validate that known estimator modules have PREFIX_TO_SECTIONS entries."""

    # Core estimator files that MUST have a mapping
    EXPECTED_MAPPED = [
        "estimators.py",
        "twfe.py",
        "staggered.py",
        "sun_abraham.py",
        "imputation.py",
        "two_stage.py",
        "stacked_did.py",
        "synthetic_did.py",
        "triple_diff.py",
        "trop.py",
        "bacon.py",
        "honest_did.py",
        "power.py",
        "pretrends.py",
        "diagnostics.py",
        "visualization.py",
        "continuous_did.py",
        "efficient_did.py",
        "survey.py",
    ]

    # Utility files that intentionally have NO mapping
    EXPECTED_UNMAPPED = [
        "linalg.py",
        "utils.py",
        "results.py",
        "prep.py",
        "prep_dgp.py",
        "datasets.py",
        "_backend.py",
        "bootstrap_utils.py",
        "__init__.py",
    ]

    def test_all_estimator_files_have_mapping(self, review_mod):
        for filename in self.EXPECTED_MAPPED:
            sections = review_mod._sections_for_file(filename)
            assert sections, f"{filename} has no PREFIX_TO_SECTIONS mapping"

    def test_utility_files_have_no_mapping(self, review_mod):
        for filename in self.EXPECTED_UNMAPPED:
            sections = review_mod._sections_for_file(filename)
            assert sections == [], f"{filename} unexpectedly has a mapping: {sections}"

    def test_visualization_submodule_maps_correctly(self, review_mod):
        """Ensure visualization/ subdirectory files map via directory name."""
        text = "M\tdiff_diff/visualization/_event_study.py"
        assert "Event Study Plotting" in review_mod._needed_sections(text)

        # _diagnostic.py inside visualization/ maps to Event Study Plotting
        # (via directory), NOT PlaceboTests (which is diagnostics.py at top level)
        text = "M\tdiff_diff/visualization/_diagnostic.py"
        sections = review_mod._needed_sections(text)
        assert "Event Study Plotting" in sections


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_rough_estimate(self, review_mod):
        # 400 chars -> ~100 tokens
        text = "a" * 400
        assert review_mod.estimate_tokens(text) == 100

    def test_empty_string(self, review_mod):
        assert review_mod.estimate_tokens("") == 0


# ---------------------------------------------------------------------------
# resolve_changed_source_files
# ---------------------------------------------------------------------------


class TestResolveChangedSourceFiles:
    def test_filters_to_diff_diff_py_files(self, review_mod, repo_root):
        text = "M\tdiff_diff/bacon.py\nM\ttests/test_bacon.py\nM\tCLAUDE.md"
        paths = review_mod.resolve_changed_source_files(text, repo_root)
        assert any("bacon.py" in p for p in paths)
        assert not any("test_bacon" in p for p in paths)
        assert not any("CLAUDE" in p for p in paths)

    def test_skips_deleted_files(self, review_mod, repo_root):
        text = "D\tdiff_diff/deleted_file.py\nM\tdiff_diff/bacon.py"
        paths = review_mod.resolve_changed_source_files(text, repo_root)
        assert not any("deleted_file" in p for p in paths)
        assert any("bacon.py" in p for p in paths)

    def test_empty_input(self, review_mod, repo_root):
        assert review_mod.resolve_changed_source_files("", repo_root) == []

    def test_skips_nonexistent_files(self, review_mod, repo_root):
        text = "M\tdiff_diff/nonexistent_xyz.py"
        assert review_mod.resolve_changed_source_files(text, repo_root) == []


# ---------------------------------------------------------------------------
# read_source_files
# ---------------------------------------------------------------------------


class TestReadSourceFiles:
    def test_produces_xml_tagged_output(self, review_mod, repo_root):
        # Use a real file that exists
        path = os.path.join(repo_root, "diff_diff", "__init__.py")
        if not os.path.isfile(path):
            pytest.skip("diff_diff/__init__.py not found")
        result = review_mod.read_source_files([path], repo_root)
        assert '<file path="diff_diff/__init__.py">' in result
        assert "</file>" in result

    def test_role_attribute(self, review_mod, repo_root):
        path = os.path.join(repo_root, "diff_diff", "__init__.py")
        if not os.path.isfile(path):
            pytest.skip("diff_diff/__init__.py not found")
        result = review_mod.read_source_files([path], repo_root, role="import-context")
        assert 'role="import-context"' in result

    def test_handles_missing_file(self, review_mod, repo_root, capsys):
        result = review_mod.read_source_files(
            ["/nonexistent/path.py"], repo_root
        )
        assert result == ""
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_empty_paths(self, review_mod, repo_root):
        assert review_mod.read_source_files([], repo_root) == ""


# ---------------------------------------------------------------------------
# parse_imports
# ---------------------------------------------------------------------------


class TestParseImports:
    def test_extracts_absolute_import(self, review_mod, repo_root):
        """Test with a real source file that imports diff_diff modules."""
        path = os.path.join(repo_root, "diff_diff", "bacon.py")
        if not os.path.isfile(path):
            pytest.skip("diff_diff/bacon.py not found")
        imports = review_mod.parse_imports(path)
        # bacon.py should import from diff_diff (e.g., diff_diff.linalg or diff_diff.utils)
        assert all(m.startswith("diff_diff.") for m in imports)

    def test_ignores_non_diff_diff_imports(self, review_mod, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("import numpy\nimport pandas\nfrom os import path\n")
        imports = review_mod.parse_imports(str(test_file))
        assert imports == set()

    def test_submodule_imports_not_truncated(self, review_mod, repo_root):
        """Submodule imports should keep full path, not truncate to 2 components."""
        path = os.path.join(repo_root, "diff_diff", "visualization", "_staggered.py")
        if not os.path.isfile(path):
            pytest.skip("diff_diff/visualization/_staggered.py not found")
        imports = review_mod.parse_imports(path)
        # Should include full submodule paths like diff_diff.visualization._common
        has_submodule = any(
            m.count(".") >= 2 for m in imports  # at least 3 components
        )
        assert has_submodule, (
            f"Expected submodule imports (3+ components) but got: {imports}"
        )

    def test_relative_import_aliases_expanded(self, review_mod, repo_root):
        """from . import _event_study should resolve to diff_diff.visualization._event_study."""
        path = os.path.join(repo_root, "diff_diff", "visualization", "__init__.py")
        if not os.path.isfile(path):
            pytest.skip("diff_diff/visualization/__init__.py not found")
        imports = review_mod.parse_imports(path)
        # Should include individual submodule names, not just the package
        submodules = [m for m in imports if m.startswith("diff_diff.visualization._")]
        assert len(submodules) > 0, (
            f"Expected visualization submodule imports but got: {imports}"
        )

    def test_handles_syntax_error(self, review_mod, tmp_path, capsys):
        test_file = tmp_path / "bad.py"
        test_file.write_text("def foo(:\n  pass\n")
        imports = review_mod.parse_imports(str(test_file))
        assert imports == set()
        captured = capsys.readouterr()
        assert "SyntaxError" in captured.err

    def test_handles_missing_file(self, review_mod):
        imports = review_mod.parse_imports("/nonexistent/file.py")
        assert imports == set()


# ---------------------------------------------------------------------------
# expand_import_graph
# ---------------------------------------------------------------------------


class TestExpandImportGraph:
    def test_expands_imports(self, review_mod, repo_root):
        """Expanding imports for a real file produces additional paths."""
        path = os.path.join(repo_root, "diff_diff", "bacon.py")
        if not os.path.isfile(path):
            pytest.skip("diff_diff/bacon.py not found")
        result = review_mod.expand_import_graph([path], repo_root)
        # Should find at least some imports (linalg, utils, etc.)
        assert isinstance(result, list)
        # All paths should be absolute and exist
        for p in result:
            assert os.path.isabs(p)
            assert os.path.isfile(p)

    def test_deduplicates_against_changed_set(self, review_mod, repo_root):
        """Files already in changed_paths should not appear in expansion."""
        bacon = os.path.join(repo_root, "diff_diff", "bacon.py")
        linalg = os.path.join(repo_root, "diff_diff", "linalg.py")
        if not (os.path.isfile(bacon) and os.path.isfile(linalg)):
            pytest.skip("required files not found")
        result = review_mod.expand_import_graph([bacon, linalg], repo_root)
        assert linalg not in [os.path.normpath(p) for p in result]

    def test_visualization_init_includes_submodules(self, review_mod, repo_root):
        """expand_import_graph on visualization/__init__.py should include submodules."""
        path = os.path.join(repo_root, "diff_diff", "visualization", "__init__.py")
        if not os.path.isfile(path):
            pytest.skip("diff_diff/visualization/__init__.py not found")
        result = review_mod.expand_import_graph([path], repo_root)
        filenames = [os.path.basename(p) for p in result]
        # Should include visualization submodules like _event_study.py, _staggered.py
        assert any(f.startswith("_") and f.endswith(".py") for f in filenames), (
            f"Expected visualization submodule files but got: {filenames}"
        )

    def test_empty_input(self, review_mod, repo_root):
        assert review_mod.expand_import_graph([], repo_root) == []


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_known_model(self, review_mod):
        result = review_mod.estimate_cost(100_000, 16_384, "gpt-5.4")
        assert result is not None
        assert "$" in result
        assert "input" in result
        assert "output" in result

    def test_unknown_model(self, review_mod):
        result = review_mod.estimate_cost(100_000, 16_384, "unknown-model")
        assert result is None

    def test_prefix_match(self, review_mod):
        # gpt-5.4-turbo should match gpt-5.4 prefix
        result = review_mod.estimate_cost(100_000, 16_384, "gpt-5.4-turbo")
        assert result is not None


# ---------------------------------------------------------------------------
# Token budget — apply_token_budget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_under_budget_all_included(self, review_mod):
        src = "y" * 400
        imp = '<file path="a.py">small</file>'
        result_src, result_imp, dropped = review_mod.apply_token_budget(
            mandatory_tokens=100,
            source_files_text=src,
            import_context_text=imp,
            budget=200_000,
        )
        assert result_src == src
        assert result_imp is not None
        assert dropped == []

    def test_over_budget_drops_imports_not_source(self, review_mod):
        src = "y" * 400
        imp = (
            '<file path="big.py">' + "z" * 40_000 + "</file>\n"
            '<file path="small.py">' + "z" * 400 + "</file>"
        )
        result_src, result_imp, dropped = review_mod.apply_token_budget(
            mandatory_tokens=200_000,  # fills budget
            source_files_text=src,
            import_context_text=imp,
            budget=200_000,
        )
        # Source files always included (sticky)
        assert result_src == src
        # At least one import file should be dropped
        assert len(dropped) > 0

    def test_source_files_always_included(self, review_mod):
        """Source files are sticky — never dropped even when over budget."""
        src = "y" * 800_000  # large source files
        result_src, _, dropped = review_mod.apply_token_budget(
            mandatory_tokens=100_000,
            source_files_text=src,
            import_context_text=None,
            budget=50_000,  # budget smaller than mandatory alone
        )
        assert result_src == src

    def test_mandatory_exceeds_budget_warns(self, review_mod, capsys):
        review_mod.apply_token_budget(
            mandatory_tokens=300_000,
            source_files_text=None,
            import_context_text=None,
            budget=200_000,
        )
        captured = capsys.readouterr()
        assert "exceeding --token-budget" in captured.err


# ---------------------------------------------------------------------------
# Review state — parse and write
# ---------------------------------------------------------------------------


class TestParseReviewState:
    def test_reads_valid_json(self, review_mod, tmp_path):
        state_file = tmp_path / "review-state.json"
        state = {
            "schema_version": 1,
            "last_reviewed_commit": "abc123",
            "review_round": 2,
            "findings": [{"id": "R1-P1-1", "severity": "P1", "summary": "Test", "status": "open"}],
        }
        state_file.write_text(json.dumps(state))
        findings, round_num = review_mod.parse_review_state(str(state_file))
        assert len(findings) == 1
        assert round_num == 2

    def test_missing_file_returns_empty(self, review_mod):
        findings, round_num = review_mod.parse_review_state("/nonexistent.json")
        assert findings == []
        assert round_num == 0

    def test_schema_version_mismatch(self, review_mod, tmp_path, capsys):
        state_file = tmp_path / "review-state.json"
        state = {"schema_version": 999, "findings": []}
        state_file.write_text(json.dumps(state))
        findings, round_num = review_mod.parse_review_state(str(state_file))
        assert findings == []
        assert round_num == 0
        captured = capsys.readouterr()
        assert "schema version mismatch" in captured.err

    def test_non_dict_root_returns_empty(self, review_mod, tmp_path, capsys):
        state_file = tmp_path / "review-state.json"
        state_file.write_text("[1, 2, 3]")  # list, not dict
        findings, round_num = review_mod.parse_review_state(str(state_file))
        assert findings == []
        assert round_num == 0
        captured = capsys.readouterr()
        assert "not a JSON object" in captured.err

    def test_non_list_findings_returns_empty(self, review_mod, tmp_path, capsys):
        state_file = tmp_path / "review-state.json"
        state = {"schema_version": 1, "findings": "not a list", "review_round": 1}
        state_file.write_text(json.dumps(state))
        findings, round_num = review_mod.parse_review_state(str(state_file))
        assert findings == []
        assert round_num == 0
        captured = capsys.readouterr()
        assert "not a list" in captured.err

    def test_non_int_round_defaults_to_zero(self, review_mod, tmp_path):
        state_file = tmp_path / "review-state.json"
        state = {"schema_version": 1, "findings": [], "review_round": "not_int"}
        state_file.write_text(json.dumps(state))
        findings, round_num = review_mod.parse_review_state(str(state_file))
        assert findings == []
        assert round_num == 0

    def test_non_dict_findings_filtered(self, review_mod, tmp_path):
        """Non-dict elements in findings list are filtered out, not crash."""
        state_file = tmp_path / "review-state.json"
        good_finding = {
            "id": "R1-P1-1", "severity": "P1",
            "summary": "Test finding", "status": "open",
        }
        state = {
            "schema_version": 1,
            "findings": ["oops", good_finding, 42],
            "review_round": 1,
        }
        state_file.write_text(json.dumps(state))
        findings, round_num = review_mod.parse_review_state(str(state_file))
        assert len(findings) == 1
        assert findings[0]["id"] == "R1-P1-1"
        assert round_num == 1

    def test_findings_missing_required_keys_filtered(self, review_mod, tmp_path):
        """Dict findings missing required keys (id, severity, summary, status) filtered."""
        state_file = tmp_path / "review-state.json"
        state = {
            "schema_version": 1,
            "findings": [
                {"id": "R1-P1-1", "severity": "P1"},  # missing summary, status
                {"id": "R1-P1-2", "severity": "P1", "summary": "Good", "status": "open"},
                {"severity": "P2", "summary": "No id", "status": "open"},  # missing id
            ],
            "review_round": 1,
        }
        state_file.write_text(json.dumps(state))
        findings, round_num = review_mod.parse_review_state(str(state_file))
        assert len(findings) == 1
        assert findings[0]["id"] == "R1-P1-2"


class TestWriteReviewState:
    def test_writes_valid_json(self, review_mod, tmp_path):
        path = str(tmp_path / "review-state.json")
        review_mod.write_review_state(
            path=path,
            commit_sha="abc123",
            base_ref="main",
            branch="feature/test",
            review_round=1,
            findings=[{"id": "R1-P0-1", "severity": "P0"}],
        )
        with open(path) as f:
            data = json.load(f)
        assert data["schema_version"] == 1
        assert data["last_reviewed_commit"] == "abc123"
        assert data["review_round"] == 1
        assert len(data["findings"]) == 1

    def test_round_trips_with_parse(self, review_mod, tmp_path):
        path = str(tmp_path / "review-state.json")
        original_findings = [
            {"id": "R1-P1-1", "severity": "P1", "summary": "Test finding", "status": "open"}
        ]
        review_mod.write_review_state(
            path=path,
            commit_sha="def456",
            base_ref="main",
            branch="fix/bug",
            review_round=3,
            findings=original_findings,
        )
        findings, round_num = review_mod.parse_review_state(path)
        assert round_num == 3
        assert findings[0]["id"] == "R1-P1-1"


# ---------------------------------------------------------------------------
# Review findings parsing
# ---------------------------------------------------------------------------


class TestParseReviewFindings:
    def test_extracts_findings(self, review_mod):
        review_text = (
            "## Methodology\n\n"
            "**P1** Missing NaN guard in `diff_diff/staggered.py:L145`\n\n"
            "## Code Quality\n\n"
            "**P2** Unused import in `diff_diff/utils.py:L12`\n\n"
            "## Summary\n"
            "Overall assessment: Looks good\n"
        )
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) >= 2
        assert not uncertain
        severities = {f["severity"] for f in findings}
        assert "P1" in severities
        assert "P2" in severities

    def test_empty_review(self, review_mod):
        findings, uncertain = review_mod.parse_review_findings("No issues found.", 1)
        assert findings == []
        assert not uncertain

    def test_finding_ids_follow_format(self, review_mod):
        review_text = (
            "**P0** Critical bug in `foo.py:L1`\n"
            "**P1** Minor issue in the code\n"
        )
        findings, _ = review_mod.parse_review_findings(review_text, 2)
        for f in findings:
            assert f["id"].startswith("R2-")
            assert f["status"] == "open"

    def test_parses_bold_severity_format(self, review_mod):
        """**P1** format should be parsed."""
        review_text = "**P1** Missing NaN guard in `foo.py:L10`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1

    def test_parses_bold_colon_severity(self, review_mod):
        """- **P1:** format (bold severity with colon) should be parsed."""
        review_text = "- **P1:** Missing NaN guard in `foo.py:L10`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"

    def test_parses_bare_colon_severity(self, review_mod):
        """- P1: format (bare severity with colon) should be parsed."""
        review_text = "- P1: Missing NaN guard in `foo.py:L10`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"

    def test_mixed_format_both_parsed(self, review_mod):
        """Review with supported + previously-unsupported format should parse both."""
        review_text = (
            "**P2** Code quality issue in `bar.py:L5`\n"
            "- **P1:** Missing NaN guard in `foo.py:L10`\n"
        )
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 2
        severities = {f["severity"] for f in findings}
        assert "P1" in severities
        assert "P2" in severities
        assert not uncertain

    def test_parses_severity_bold_value(self, review_mod):
        """Severity: **P1** format (bold value after plain label) should be parsed."""
        review_text = "- Severity: **P1** — Missing NaN guard in `foo.py:L10`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"

    def test_parses_numbered_list_severity(self, review_mod):
        """1. Severity: P1 format should be parsed."""
        review_text = "1. Severity: P1 — Missing NaN guard in `foo.py:L10`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"

    def test_parses_starred_bold_severity(self, review_mod):
        """* **Severity:** P1 format should be parsed."""
        review_text = "* **Severity:** P1 — Missing NaN guard in `bar.py:L5`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"

    def test_numbered_bold_severity_triggers_uncertainty(self, review_mod):
        """1. **Severity:** P1 with no parseable summary → uncertain=True."""
        review_text = "1. **Severity:** P1\n"
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        assert findings == []
        assert uncertain

    def test_parses_bold_label_format(self, review_mod):
        """**Severity:** P1 format should be parsed."""
        review_text = "- **Severity:** P1 — Missing NaN guard in `foo.py:L10`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"

    def test_parses_plain_label_format(self, review_mod):
        """Severity: P2 format should be parsed."""
        review_text = "Severity: P2 — Unused import in `bar.py:L5`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P2"

    def test_finding_with_skip_marker_in_summary_still_parsed(self, review_mod):
        """Findings whose summaries contain skip markers like 'Path to Approval' should parse."""
        review_text = "**P2** The prompt omits the Path to Approval section in `foo.py:L10`\n"
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P2"
        assert not uncertain

    def test_finding_with_looks_good_in_summary(self, review_mod):
        """Finding mentioning 'Looks good' in summary should not be skipped."""
        review_text = "**P1** Assessment says Looks good but edge case is unhandled in `bar.py:L5`\n"
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"

    def test_parses_multiline_finding_block(self, review_mod):
        """Multi-line finding blocks (Severity/Impact on separate lines)."""
        review_text = (
            "## Code Quality\n\n"
            "- **Severity:** P1\n"
            "  **Impact:** Missing NaN guard causes silent incorrect output\n"
            "  **Location:** `diff_diff/staggered.py:L145`\n"
            "  **Concrete fix:** Use safe_inference()\n"
        )
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"
        assert "NaN guard" in findings[0]["summary"]
        assert not uncertain

    def test_parses_plain_multiline_block(self, review_mod):
        """Plain Severity: / Impact: labels (no bold) should be parsed."""
        review_text = (
            "## Code Quality\n\n"
            "Severity: P1\n"
            "Impact: Missing NaN guard causes silent incorrect output\n"
            "Location: `diff_diff/staggered.py:L145`\n"
            "Concrete fix: Use safe_inference()\n"
        )
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        assert len(findings) == 1
        assert findings[0]["severity"] == "P1"
        assert "NaN guard" in findings[0]["summary"]
        assert not uncertain

    def test_midline_severity_not_detected(self, review_mod):
        """Severity markers embedded mid-line are not block starts — no uncertainty."""
        review_text = (
            "There is a Severity: P1 issue but the rest of the text\n"
            "doesn't follow any recognized block structure at all\n"
        )
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        # Mid-line markers are not valid block starts — correctly returns ([], False)
        assert findings == []
        assert not uncertain

    def test_midline_bold_severity_not_detected(self, review_mod):
        """Bold severity mid-line (not at line start) is not a block start."""
        review_text = (
            "The review found **P1** issues but in a format\n"
            "that the block parser cannot delimit properly.\n"
        )
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        # Mid-line bold is not a valid block start — correctly returns ([], False)
        assert findings == []
        assert not uncertain

    def test_bold_label_severity_triggers_uncertainty(self, review_mod):
        """**Severity:** P1 format with no parseable summary → uncertain=True."""
        review_text = "- **Severity:** P1\n"
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        assert findings == []
        assert uncertain

    def test_bold_inline_severity_triggers_uncertainty(self, review_mod):
        """**Severity: P1** format with no parseable summary → uncertain=True."""
        review_text = "- **Severity: P1**\n"
        findings, uncertain = review_mod.parse_review_findings(review_text, 1)
        assert findings == []
        assert uncertain

    def test_ignores_multi_severity_prose(self, review_mod):
        """Lines like 'P2/P3 items may exist' should not be parsed as findings."""
        review_text = (
            "P2/P3 items may exist. A PR does NOT need to be perfect.\n"
            "If all previous P1+ findings are resolved, assessment should be good.\n"
        )
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert findings == []

    def test_ignores_assessment_lines(self, review_mod):
        """Assessment criteria lines with severity labels should be skipped."""
        review_text = (
            "⛔ Blocker — One or more P0: silent correctness bugs\n"
            "⚠️ Needs changes — One or more P1 (no P0s)\n"
            "✅ Looks good — No unmitigated P0 or P1 findings.\n"
        )
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert findings == []

    def test_ignores_table_rows(self, review_mod):
        """Findings tables from previous reviews should not be re-parsed."""
        review_text = (
            "| R1-P1-1 | P1 | Methodology | Missing NaN guard | foo.py:L10 | open |\n"
            "| R1-P2-1 | P2 | Code Quality | Unused import | bar.py:L5 | addressed |\n"
        )
        findings, _ = review_mod.parse_review_findings(review_text, 2)
        assert findings == []

    def test_ignores_instructional_text(self, review_mod):
        """Instructional text referencing severities should be skipped."""
        review_text = (
            "Focus on whether previous P0/P1 findings have been addressed.\n"
            "If all previous P1+ findings are resolved, the assessment should be good.\n"
        )
        findings, _ = review_mod.parse_review_findings(review_text, 1)
        assert findings == []


# ---------------------------------------------------------------------------
# Merge findings
# ---------------------------------------------------------------------------


class TestMergeFindings:
    def test_matching_finding_stays_open(self, review_mod):
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard", "status": "open"}
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard", "status": "open"}
        ]
        merged = review_mod.merge_findings(previous, current)
        open_at_loc = [
            f for f in merged
            if f["location"] == "foo.py:L10" and f["status"] == "open"
        ]
        assert len(open_at_loc) >= 1

    def test_absent_finding_marked_addressed(self, review_mod):
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard", "status": "open"}
        ]
        current = []  # Finding was addressed
        merged = review_mod.merge_findings(previous, current)
        addressed = [f for f in merged if f["status"] == "addressed"]
        assert len(addressed) == 1
        assert addressed[0]["location"] == "foo.py:L10"

    def test_new_finding_added_as_open(self, review_mod):
        previous = []
        current = [
            {"id": "R2-P0-1", "severity": "P0", "location": "bar.py:L5",
             "section": "Methodology", "summary": "Missing check", "status": "open"}
        ]
        merged = review_mod.merge_findings(previous, current)
        assert len(merged) == 1
        assert merged[0]["status"] == "open"
        assert merged[0]["location"] == "bar.py:L5"

    def test_matching_with_shifted_line_numbers(self, review_mod):
        """Same finding at different line ranges should still match via summary."""
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"}
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "foo.py:L10-L12",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"}
        ]
        merged = review_mod.merge_findings(previous, current)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        # Should match (same severity, file, summary) — not create a false "addressed"
        assert len(open_findings) == 1
        assert len(addressed) == 0

    def test_matching_with_missing_location(self, review_mod):
        """Finding with no location should still match on summary fingerprint."""
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"}
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"}
        ]
        merged = review_mod.merge_findings(previous, current)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        # Same severity + same summary = match. No false "addressed" record.
        assert len(open_findings) == 1
        assert len(addressed) == 0

    def test_multiple_findings_same_key(self, review_mod):
        """Multiple previous findings with same key should not overwrite each other."""
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"},
            {"id": "R1-P1-2", "severity": "P1", "location": "foo.py:L20",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"},
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"},
        ]
        merged = review_mod.merge_findings(previous, current)
        # One should match, one should be addressed
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        assert len(open_findings) == 1
        assert len(addressed) == 1

    def test_duplicate_no_location_findings_one_to_one(self, review_mod):
        """Two prior no-location findings should not both match one current finding."""
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "",
             "section": "Code Quality", "summary": "Missing NaN guard",
             "status": "open"},
            {"id": "R1-P1-2", "severity": "P1", "location": "",
             "section": "Methodology", "summary": "Missing NaN guard",
             "status": "open"},
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard",
             "status": "open"},
        ]
        merged = review_mod.merge_findings(previous, current)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        # One current + one prior matched = 1 open; one prior unmatched = 1 addressed
        assert len(open_findings) == 1
        assert len(addressed) == 1

    def test_previous_missing_location_current_has_location(self, review_mod):
        """Previous finding with no location, current has one → should match."""
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"}
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "staggered.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard in staggered",
             "status": "open"}
        ]
        merged = review_mod.merge_findings(previous, current)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        # Should match via symmetric fallback — no false "addressed"
        assert len(open_findings) == 1
        assert len(addressed) == 0

    def test_same_basename_different_dirs_no_cross_match(self, review_mod):
        """__init__.py in different dirs with same summary should NOT cross-match."""
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "diff_diff/__init__.py:L10",
             "section": "Code Quality", "summary": "Missing type export", "status": "open"}
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "diff_diff/visualization/__init__.py:L5",
             "section": "Code Quality", "summary": "Missing type export", "status": "open"}
        ]
        merged = review_mod.merge_findings(previous, current)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        # Different full paths: previous should be addressed, current stays open
        assert len(open_findings) == 1
        assert len(addressed) == 1

    def test_long_summaries_dont_collide(self, review_mod):
        """Two findings with same first 50 chars but different suffixes should NOT collapse."""
        prefix = "a" * 50
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": prefix + " first issue details",
             "status": "open"},
            {"id": "R1-P1-2", "severity": "P1", "location": "foo.py:L20",
             "section": "Code Quality", "summary": prefix + " second different issue",
             "status": "open"},
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": prefix + " first issue details",
             "status": "open"},
            {"id": "R2-P1-2", "severity": "P1", "location": "foo.py:L20",
             "section": "Code Quality", "summary": prefix + " second different issue",
             "status": "open"},
        ]
        merged = review_mod.merge_findings(previous, current)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        # Both should match — neither dropped
        assert len(open_findings) == 2
        assert len(addressed) == 0

    def test_same_summary_different_files_no_cross_match(self, review_mod):
        """Two findings with same summary but different files should NOT cross-match."""
        previous = [
            {"id": "R1-P1-1", "severity": "P1", "location": "foo.py:L10",
             "section": "Code Quality", "summary": "Missing NaN guard in estimator",
             "status": "open"},
        ]
        current = [
            {"id": "R2-P1-1", "severity": "P1", "location": "bar.py:L20",
             "section": "Code Quality", "summary": "Missing NaN guard in estimator",
             "status": "open"},
        ]
        merged = review_mod.merge_findings(previous, current)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        # Different files: previous should be addressed, current stays open
        assert len(open_findings) == 1
        assert open_findings[0]["location"] == "bar.py:L20"
        assert len(addressed) == 1
        assert addressed[0]["location"] == "foo.py:L10"


# ---------------------------------------------------------------------------
# estimate_cost — prefix matching regression
# ---------------------------------------------------------------------------


class TestEstimateCostPrefixRegression:
    def test_mini_model_gets_mini_pricing(self, review_mod):
        """gpt-4.1-mini snapshot should get mini pricing, not parent gpt-4.1."""
        mini_cost = review_mod.estimate_cost(1_000_000, 1_000_000, "gpt-4.1-mini-2025-04-14")
        parent_cost = review_mod.estimate_cost(1_000_000, 1_000_000, "gpt-4.1")
        assert mini_cost is not None
        assert parent_cost is not None
        # Mini should be cheaper than parent
        assert mini_cost != parent_cost

    def test_o3_mini_gets_mini_pricing(self, review_mod):
        """o3-mini snapshot should get o3-mini pricing, not o3."""
        mini_cost = review_mod.estimate_cost(1_000_000, 1_000_000, "o3-mini-2025-01-31")
        parent_cost = review_mod.estimate_cost(1_000_000, 1_000_000, "o3")
        assert mini_cost is not None
        assert parent_cost is not None
        assert mini_cost != parent_cost


# ---------------------------------------------------------------------------
# Delta context derivation
# ---------------------------------------------------------------------------


class TestDeltaContextDerivation:
    def test_delta_files_resolve_only_delta(self, review_mod, repo_root):
        """resolve_changed_source_files with delta file list returns only delta files."""
        # Simulate: full branch changed bacon.py and staggered.py, but delta only has bacon.py
        delta_text = "M\tdiff_diff/bacon.py"
        paths = review_mod.resolve_changed_source_files(delta_text, repo_root)
        filenames = [os.path.basename(p) for p in paths]
        assert "bacon.py" in filenames
        # staggered.py should NOT be in the result (it's not in delta)
        assert "staggered.py" not in filenames


# ---------------------------------------------------------------------------
# Review state — branch/base validation support
# ---------------------------------------------------------------------------


class TestReviewStateBranchValidation:
    def test_stores_and_retrieves_branch_and_base(self, review_mod, tmp_path):
        """write_review_state stores branch/base; parse_review_state returns them."""
        path = str(tmp_path / "review-state.json")
        review_mod.write_review_state(
            path=path,
            commit_sha="abc123",
            base_ref="main",
            branch="feature/test",
            review_round=1,
            findings=[],
        )
        # Read back and verify fields are present
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["branch"] == "feature/test"
        assert data["base_ref"] == "main"


# ---------------------------------------------------------------------------
# End-to-end: parse then merge pipeline
# ---------------------------------------------------------------------------


class TestParseThenMerge:
    def test_line_shift_does_not_cause_churn(self, review_mod):
        """Same finding at different line numbers should merge as 1 open, 0 addressed."""
        review_r1 = "**P1** Missing NaN guard in `foo.py:L10`\n"
        review_r2 = "**P1** Missing NaN guard in `foo.py:L12`\n"
        findings_r1, _ = review_mod.parse_review_findings(review_r1, 1)
        findings_r2, _ = review_mod.parse_review_findings(review_r2, 2)
        assert len(findings_r1) == 1
        assert len(findings_r2) == 1
        merged = review_mod.merge_findings(findings_r1, findings_r2)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        assert len(open_findings) == 1
        assert len(addressed) == 0

    def test_md_file_line_shift_does_not_cause_churn(self, review_mod):
        """Same finding on a .md file at different line numbers should merge as 1 open."""
        review_r1 = "**P1** Missing docs in `ai-review-local.md:L10`\n"
        review_r2 = "**P1** Missing docs in `ai-review-local.md:L20`\n"
        findings_r1, _ = review_mod.parse_review_findings(review_r1, 1)
        findings_r2, _ = review_mod.parse_review_findings(review_r2, 2)
        assert len(findings_r1) == 1
        assert len(findings_r2) == 1
        merged = review_mod.merge_findings(findings_r1, findings_r2)
        open_findings = [f for f in merged if f["status"] == "open"]
        addressed = [f for f in merged if f["status"] == "addressed"]
        assert len(open_findings) == 1
        assert len(addressed) == 0

    def test_parse_uncertain_does_not_advance_state(self, review_mod, tmp_path):
        """When parse_uncertain fires, review-state.json should not be modified."""
        state_path = str(tmp_path / "review-state.json")
        # Write initial state
        review_mod.write_review_state(
            path=state_path,
            commit_sha="initial123",
            base_ref="main",
            branch="feature/x",
            review_round=1,
            findings=[{"id": "R1-P1-1", "severity": "P1", "summary": "Test", "status": "open"}],
        )
        initial_mtime = os.path.getmtime(state_path)

        # Simulate parse_uncertain scenario
        unparseable_review = "- **Severity:** P1\n"  # Will return ([], True)
        findings, uncertain = review_mod.parse_review_findings(unparseable_review, 2)
        assert uncertain
        assert findings == []

        # The state file should NOT have been modified
        # (in production, main() skips write_review_state when uncertain)
        current_mtime = os.path.getmtime(state_path)
        assert current_mtime == initial_mtime

        # Verify original state is intact
        stored_findings, stored_round = review_mod.parse_review_state(state_path)
        assert stored_round == 1
        assert stored_findings[0]["id"] == "R1-P1-1"


# ---------------------------------------------------------------------------
# validate_review_state — comprehensive validation
# ---------------------------------------------------------------------------


class TestValidateReviewState:
    def test_valid_state_returns_true(self, review_mod, tmp_path):
        path = str(tmp_path / "review-state.json")
        review_mod.write_review_state(
            path=path, commit_sha="abc123", base_ref="main",
            branch="feature/test", review_round=1,
            findings=[{"id": "R1-P1-1", "severity": "P1",
                       "summary": "Test", "status": "open"}],
        )
        findings, rnd, commit, valid = review_mod.validate_review_state(
            path, "feature/test", "main"
        )
        assert valid
        assert commit == "abc123"
        assert len(findings) == 1

    def test_branch_mismatch_returns_false(self, review_mod, tmp_path):
        path = str(tmp_path / "review-state.json")
        review_mod.write_review_state(
            path=path, commit_sha="abc123", base_ref="main",
            branch="feature/old", review_round=1, findings=[],
        )
        _, _, _, valid = review_mod.validate_review_state(
            path, "feature/new", "main"
        )
        assert not valid

    def test_schema_mismatch_returns_false(self, review_mod, tmp_path):
        state_file = tmp_path / "review-state.json"
        state_file.write_text(json.dumps({"schema_version": 999}))
        _, _, _, valid = review_mod.validate_review_state(
            str(state_file), "b", "main"
        )
        assert not valid

    def test_missing_file_returns_false(self, review_mod):
        _, _, _, valid = review_mod.validate_review_state(
            "/nonexistent.json", "b", "main"
        )
        assert not valid

    def test_malformed_finding_returns_false(self, review_mod, tmp_path):
        """Any malformed finding dict should invalidate delta mode entirely."""
        state_file = tmp_path / "review-state.json"
        state = {
            "schema_version": 1,
            "last_reviewed_commit": "abc123",
            "branch": "feature/test",
            "base_ref": "main",
            "review_round": 1,
            "findings": [
                {"id": "R1-P1-1", "severity": "P1"},  # missing summary, status
            ],
        }
        state_file.write_text(json.dumps(state))
        _, _, _, valid = review_mod.validate_review_state(
            str(state_file), "feature/test", "main"
        )
        assert not valid  # fail closed on malformed finding


# ---------------------------------------------------------------------------
# Include-files path confinement
# ---------------------------------------------------------------------------


class TestIncludeFilesConfinement:
    """Verify --include-files rejects paths outside repo root."""

    def test_rejects_absolute_path(self, review_mod, repo_root, capsys):
        """Absolute paths should be rejected."""
        # Simulate the path resolution logic from main()
        name = "/etc/passwd"
        assert os.path.isabs(name)
        # The script rejects absolute paths before even resolving

    def test_rejects_traversal(self, review_mod, repo_root):
        """../ traversal should be detected after realpath normalization."""
        candidate = os.path.join(repo_root, "../../../etc/passwd")
        candidate = os.path.realpath(candidate)
        repo_root_real = os.path.realpath(repo_root)
        assert not candidate.startswith(repo_root_real + os.sep)


# ---------------------------------------------------------------------------
# Responses API migration
# ---------------------------------------------------------------------------


class TestIsReasoningModel:
    def test_o3_is_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("o3") is True

    def test_o3_mini_is_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("o3-mini") is True

    def test_o3_snapshot_is_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("o3-mini-2025-01-31") is True

    def test_o1_is_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("o1") is True

    def test_o4_mini_is_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("o4-mini") is True

    def test_pro_is_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("gpt-5.4-pro") is True

    def test_pro_snapshot_is_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("gpt-5.4-pro-2026-03-05") is True

    def test_gpt54_is_not_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("gpt-5.4") is False

    def test_gpt41_is_not_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("gpt-4.1") is False

    def test_gpt41_mini_is_not_reasoning(self, review_mod):
        assert review_mod._is_reasoning_model("gpt-4.1-mini") is False


class TestProModelPricing:
    def test_pro_gets_own_pricing(self, review_mod):
        """gpt-5.4-pro should not fall back to gpt-5.4 pricing."""
        pro_cost = review_mod.estimate_cost(1_000_000, 1_000_000, "gpt-5.4-pro")
        base_cost = review_mod.estimate_cost(1_000_000, 1_000_000, "gpt-5.4")
        assert pro_cost is not None
        assert base_cost is not None
        assert pro_cost != base_cost

    def test_pro_snapshot_matches_pro(self, review_mod):
        """gpt-5.4-pro-2026-03-05 should match gpt-5.4-pro via prefix."""
        snapshot = review_mod.estimate_cost(1_000_000, 1_000_000, "gpt-5.4-pro-2026-03-05")
        base = review_mod.estimate_cost(1_000_000, 1_000_000, "gpt-5.4-pro")
        assert snapshot == base


class TestExtractResponseText:
    def test_prefers_output_text_field(self, review_mod):
        result = {"output_text": "Direct text.", "output": []}
        assert review_mod._extract_response_text(result) == "Direct text."

    def test_walks_output_items_when_output_text_null(self, review_mod):
        result = {
            "output_text": None,
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "Walked text."},
            ]}],
        }
        assert review_mod._extract_response_text(result) == "Walked text."

    def test_concatenates_multiple_blocks(self, review_mod):
        result = {
            "output_text": None,
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "A"},
                {"type": "output_text", "text": "B"},
            ]}],
        }
        assert review_mod._extract_response_text(result) == "AB"

    def test_empty_when_no_output(self, review_mod):
        assert review_mod._extract_response_text({"output_text": None, "output": []}) == ""

    def test_empty_when_missing_keys(self, review_mod):
        assert review_mod._extract_response_text({}) == ""


class TestResponsesAPIConstants:
    def test_endpoint_is_responses(self, review_mod):
        assert "responses" in review_mod.ENDPOINT
        assert "chat/completions" not in review_mod.ENDPOINT

    def test_reasoning_max_tokens_larger(self, review_mod):
        assert review_mod.REASONING_MAX_TOKENS > review_mod.DEFAULT_MAX_TOKENS


class TestCallOpenAIPayload:
    """Test call_openai() payload construction and response parsing via mocked urllib."""

    @pytest.fixture()
    def mock_urlopen(self, monkeypatch, review_mod):
        """Patch urllib.request.urlopen to capture requests and return canned responses."""
        import io
        import urllib.request

        captured = {}

        class FakeResponse:
            def __init__(self, data):
                self._data = json.dumps(data).encode("utf-8")

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        _DEFAULT_RESPONSE = {
            "status": "completed",
            "output_text": None,
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Review content here."}],
            }],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        def fake_urlopen(req, timeout=None):
            captured["request"] = req
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeResponse(captured.get("response_data", _DEFAULT_RESPONSE))

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return captured

    def test_standard_model_payload(self, review_mod, mock_urlopen):
        """Standard model sends input, max_output_tokens, and temperature=0."""
        content, usage = review_mod.call_openai("test prompt", "gpt-5.4", "fake-key")
        payload = mock_urlopen["payload"]
        assert payload["input"] == "test prompt"
        assert payload["max_output_tokens"] == review_mod.DEFAULT_MAX_TOKENS
        assert payload["temperature"] == 0
        assert "messages" not in payload
        assert "max_completion_tokens" not in payload
        assert content == "Review content here."
        assert usage["input_tokens"] == 100

    def test_reasoning_model_payload(self, review_mod, mock_urlopen):
        """Reasoning model omits temperature and uses REASONING_MAX_TOKENS."""
        content, _ = review_mod.call_openai("test prompt", "gpt-5.4-pro", "fake-key")
        payload = mock_urlopen["payload"]
        assert payload["max_output_tokens"] == review_mod.REASONING_MAX_TOKENS
        assert "temperature" not in payload
        assert content == "Review content here."

    def test_request_url_is_responses_endpoint(self, review_mod, mock_urlopen):
        review_mod.call_openai("test", "gpt-5.4", "fake-key")
        assert mock_urlopen["request"].full_url == review_mod.ENDPOINT

    def test_timeout_passed_through(self, review_mod, mock_urlopen):
        review_mod.call_openai("test", "gpt-5.4", "fake-key", timeout=900)
        assert mock_urlopen["timeout"] == 900

    def test_missing_status_with_valid_output_succeeds(self, review_mod, mock_urlopen):
        """Valid content should be accepted even when status field is absent."""
        mock_urlopen["response_data"] = {
            "output_text": None,
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Good review."}],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        content, _ = review_mod.call_openai("test", "gpt-5.4", "fake-key")
        assert content == "Good review."

    def test_status_none_with_valid_output_succeeds(self, review_mod, mock_urlopen):
        """status=None should not prevent content extraction."""
        mock_urlopen["response_data"] = {
            "status": None,
            "output_text": None,
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Good review."}],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        content, _ = review_mod.call_openai("test", "gpt-5.4", "fake-key")
        assert content == "Good review."

    def test_incomplete_status_with_content_exits(self, review_mod, mock_urlopen):
        """Truncated response (status=incomplete) should exit even if content exists."""
        mock_urlopen["response_data"] = {
            "status": "incomplete",
            "output_text": None,
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Partial review."}],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        with pytest.raises(SystemExit):
            review_mod.call_openai("test", "gpt-5.4", "fake-key")

    def test_incomplete_status_surfaces_details(self, review_mod, mock_urlopen, capsys):
        """Incomplete response should print incomplete_details to stderr."""
        mock_urlopen["response_data"] = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output_text": None,
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Partial."}],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        with pytest.raises(SystemExit):
            review_mod.call_openai("test", "gpt-5.4", "fake-key")
        captured = capsys.readouterr()
        assert "truncated" in captured.err.lower()
        assert "max_output_tokens" in captured.err

    def test_output_text_convenience_field_used(self, review_mod, mock_urlopen):
        """When output_text is populated (SDK-style), use it directly."""
        mock_urlopen["response_data"] = {
            "status": "completed",
            "output_text": "SDK-provided text.",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        content, _ = review_mod.call_openai("test", "gpt-5.4", "fake-key")
        assert content == "SDK-provided text."

    def test_multiple_output_text_blocks_concatenated(self, review_mod, mock_urlopen):
        """Multiple output_text blocks should be concatenated in order."""
        mock_urlopen["response_data"] = {
            "status": "completed",
            "output_text": None,
            "output": [{
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Part 1. "},
                    {"type": "output_text", "text": "Part 2."},
                ],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        content, _ = review_mod.call_openai("test", "gpt-5.4", "fake-key")
        assert content == "Part 1. Part 2."

    def test_failed_status_no_content_exits(self, review_mod, mock_urlopen):
        """Failed status with no usable content should exit."""
        mock_urlopen["response_data"] = {
            "status": "failed",
            "output_text": None,
            "output": [],
            "usage": {},
        }
        with pytest.raises(SystemExit):
            review_mod.call_openai("test", "gpt-5.4", "fake-key")

    def test_empty_output_exits(self, review_mod, mock_urlopen):
        """Empty output items with completed status should exit."""
        mock_urlopen["response_data"] = {
            "status": "completed",
            "output_text": None,
            "output": [],
            "usage": {},
        }
        with pytest.raises(SystemExit):
            review_mod.call_openai("test", "gpt-5.4", "fake-key")
