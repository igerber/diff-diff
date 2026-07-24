"""Contract tests for the shipped `.claude/skills/plan-review/` engine.

Guards that production == the plan-review engine Campaign 1 graded: the bundled
prompt artifacts are byte-identical to the validated `candidates/` copies, and
the shipped `render.py` renders them byte-identically to the harness renderer
(a byte-match of the template FILES is not enough — the RENDERED prompt is what
reaches a reviewer). The SKILL.md invocation pins (codex model/effort/timeout,
Claude subagent model) and command-contract migration are asserted here too.
"""

import importlib.util
import pathlib
import re
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_SKILL = _REPO / ".claude" / "skills" / "plan-review"
_CANDIDATES = _REPO / "tools" / "plan-review-eval" / "candidates"

pytestmark = pytest.mark.skipif(
    not _SKILL.exists(), reason="plan-review skill not present (isolated install)"
)

# harness render() lives under tools/plan-review-eval/plan_adapters/
if str(_REPO / "tools" / "plan-review-eval") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools" / "plan-review-eval"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))


def _load_skill_render():
    spec = importlib.util.spec_from_file_location("plan_review_skill_render", _SKILL / "render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ARTIFACTS = ("criteria.md", "reviewer_prompt.md", "merge_verify.md")


def test_bundled_artifacts_byte_match_validated_candidates():
    """The shipped prompts must be the exact bytes the campaign graded."""
    for name in _ARTIFACTS:
        shipped = (_SKILL / name).read_bytes()
        graded = (_CANDIDATES / name).read_bytes()
        assert shipped == graded, f"{name} drifted from the campaign-validated candidate"
    # extraction_prompt.md is eval-only — must NOT be promoted.
    assert not (_SKILL / "extraction_prompt.md").exists()


def test_render_byte_equivalent_to_harness_on_shipped_templates():
    """render.py output == criteria_source.render output, byte-for-byte, on the
    real reviewer + merge templates with representative token values."""
    skill_render = _load_skill_render().render
    from plan_adapters.criteria_source import render as harness_render

    criteria = (_SKILL / "criteria.md").read_text()
    plan = "# A plan\n\nStep 1: touch __PLAN__ and __CRITERIA__ in prose (must not re-substitute)."
    reviewer_tmpl = (_SKILL / "reviewer_prompt.md").read_text()
    merge_tmpl = (_SKILL / "merge_verify.md").read_text()

    a = skill_render(reviewer_tmpl, criteria=criteria, plan=plan)
    b = harness_render(reviewer_tmpl, criteria=criteria, plan=plan)
    assert a == b

    a2 = skill_render(
        merge_tmpl, criteria=criteria, plan=plan, review_a="R1 findings", review_b="R2 findings"
    )
    b2 = harness_render(
        merge_tmpl, criteria=criteria, plan=plan, review_a="R1 findings", review_b="R2 findings"
    )
    assert a2 == b2
    # single-pass: plan prose mentioning tokens is not re-substituted
    assert "__PLAN__" in a and "__CRITERIA__" in a


def test_render_raises_on_unfilled_template_token():
    """A template token with no provided value must raise, never ship a literal
    placeholder to a reviewer."""
    skill = _load_skill_render()
    with pytest.raises(skill.RenderError):
        skill.render("before __CRITERIA__ __PLAN__ after", criteria="x")  # __PLAN__ missing
    # a surplus kwarg absent from the template is ignored (harness semantics)
    assert skill.render("just __CRITERIA__", criteria="c", unused="ignored") == "just c"


def test_skill_md_present_with_frontmatter():
    text = (_SKILL / "SKILL.md").read_text()
    assert text.startswith("---"), "SKILL.md needs YAML frontmatter"
    fm = text.split("---", 2)[1]
    assert re.search(r"^name:\s*plan-review\s*$", fm, re.M)
    assert re.search(r"^description:\s*\S", fm, re.M)


def test_reviewer_invocations_are_pinned():
    """Identical templates + a wrong model/effort/timeout would pass the
    byte-match silently. The codex pins live at the real `call_codex` site
    (`codex_review.py`); the Claude subagent model pin lives in SKILL.md."""
    codex = (_SKILL / "codex_review.py").read_text()
    assert 'CODEX_MODEL = "gpt-5.6-sol"' in codex
    assert 'CODEX_EFFORT = "xhigh"' in codex
    assert "CODEX_TIMEOUT_S = 600" in codex
    # and codex_review.py actually passes them to call_codex
    assert "model=CODEX_MODEL" in codex and "effort=CODEX_EFFORT" in codex
    assert "timeout_s=CODEX_TIMEOUT_S" in codex

    skill = (_SKILL / "SKILL.md").read_text()
    # Claude reviewer + merge subagents pin model=opus (Task tool takes family
    # aliases; "opus" resolves to the graded Opus 4.8).
    assert re.search(r'model[=:]\s*["\']opus', skill), "Claude subagents must pin model=opus"
    # and SKILL.md drives the codex half through codex_review.py, not free-text
    assert "codex_review.py" in skill
    assert "render.py" in skill


def test_skill_md_routes_through_helper_not_basenames():
    """The skill must use plan_snapshot.py for snapshot/persist/check and never
    derive review paths from basenames (CI round-3 lesson)."""
    text = (_SKILL / "SKILL.md").read_text()
    assert "plan_snapshot.py" in text
    assert "replace the trailing `.md` with `.review.md`" not in text
    assert "<plan-basename>.review.md" not in text


# --------------------------------------------------------------------------- #
# Contracts migrated from test_command_contract.py when /review-plan +
# /revise-plan were retired (they now target SKILL.md + CLAUDE.md).
# --------------------------------------------------------------------------- #

_PLACEHOLDER_ASSIGN = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*="<[^>]+>"')


def test_skill_no_raw_placeholder_assignment():
    """No `VAR="<placeholder>"` shell assignment in SKILL.md — paths flow only
    as quoted literal arguments / through the Write tool (injection class)."""
    text = (_SKILL / "SKILL.md").read_text()
    offenders = [ln for ln in text.splitlines() if _PLACEHOLDER_ASSIGN.match(ln)]
    assert not offenders, f"SKILL.md has raw placeholder assignment(s): {offenders}"


def test_skill_no_prose_shasum_over_plan_paths():
    """SKILL.md hashes plans only via plan_snapshot.py, never a raw shasum over
    an interpolated plan path."""
    text = (_SKILL / "SKILL.md").read_text()
    offenders = [ln for ln in text.splitlines() if "shasum" in ln and "<plan" in ln]
    assert not offenders, f"SKILL.md interpolates a plan path into shasum: {offenders}"


def test_skill_revise_uses_check_and_review_path():
    """The revise phase locates the review via `plan_snapshot.py check` and the
    helper's `review_path`, never a basename-derived name (CI round-3)."""
    text = (_SKILL / "SKILL.md").read_text()
    assert "plan_snapshot.py check" in text
    assert "review_path" in text


def test_skill_initializes_scratch_before_first_write():
    """mkdir -p the scratch dir before the first `plan-path.txt` Write (round-4)."""
    text = (_SKILL / "SKILL.md").read_text()
    first_write = text.index("plan-path.txt")
    mkdir = text.index('mkdir -p "$SCRATCH"')
    assert mkdir < first_write, "mkdir -p must precede the first scratch Write"


def test_ingress_calls_require_plan_path_confirmation():
    """Every `plan_snapshot.py check|snapshot` in SKILL.md AND CLAUDE.md must be
    followed by a plan_path-confirmation instruction (shared per-worktree
    ingress file; CI round-7)."""
    surfaces = {
        "SKILL.md": (_SKILL / "SKILL.md").read_text(),
        "CLAUDE.md": (_REPO / "CLAUDE.md").read_text(),
    }
    invoke = re.compile(r"plan_snapshot\.py (?:check|snapshot)\b")
    confirm = re.compile(r"[Cc]onfirm the printed\s+`plan_path`")
    offenders = []
    for name, text in surfaces.items():
        for m in invoke.finditer(text):
            if not confirm.search(text[m.end() : m.end() + 700]):
                offenders.append((name, text.count("\n", 0, m.start()) + 1))
    assert not offenders, f"ingress call(s) without plan_path confirmation: {offenders}"
