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


def _load_codex_review():
    spec = importlib.util.spec_from_file_location(
        "plan_review_codex_review", _SKILL / "codex_review.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeOpenAIReview:
    """Stand-in for the loaded openai_review module, recording the calls
    codex_review.main makes so the tests assert BEHAVIOR (pins forwarded,
    sensitive-file notice printed) rather than source text."""

    def __init__(self):
        self.notice_calls = []
        self.codex_calls = []

    def _scan_sensitive_files(self, repo_root):
        return [".env"]

    def _print_sensitive_notice(self, repo_root, found):
        self.notice_calls.append((repo_root, tuple(found)))

    def call_codex(self, *, prompt, model, repo_root, effort, timeout_s):
        self.codex_calls.append(
            dict(
                prompt=prompt, model=model, repo_root=repo_root, effort=effort, timeout_s=timeout_s
            )
        )
        return ("CODEX REVIEW BODY", {"meta": 1})


# The DETECTION-critical prompts (what each reviewer looks for) ship byte-
# identical to what the campaign graded. merge_verify.md is production-adapted
# (reviewer naming un-blinded — see test_merge_verify_is_production_adapted);
# its verify LOGIC is unchanged, only the output labeling.
_DETECTION_ARTIFACTS = ("criteria.md", "reviewer_prompt.md")


def test_detection_prompts_byte_match_validated_candidates():
    """The detection-critical prompts must be the exact bytes the campaign graded."""
    for name in _DETECTION_ARTIFACTS:
        shipped = (_SKILL / name).read_bytes()
        graded = (_CANDIDATES / name).read_bytes()
        assert shipped == graded, f"{name} drifted from the campaign-validated candidate"
    # extraction_prompt.md is eval-only — must NOT be promoted.
    assert not (_SKILL / "extraction_prompt.md").exists()


def test_merge_verify_is_production_adapted():
    """merge_verify NAMES reviewers in the persisted review (pre-campaign
    'names in file' decision) while keeping the verify-every-finding logic
    identical to the graded copy — the campaign blinded it only for grading."""
    shipped = (_SKILL / "merge_verify.md").read_text()
    graded = (_CANDIDATES / "merge_verify.md").read_text()
    # verify/detection invariant preserved from the graded copy
    assert "Verify EVERY finding" in shipped and "Verify EVERY finding" in graded
    assert "[consensus]" in shipped
    # named single-reviewer attribution added; the blinding rule removed
    assert "[single reviewer: claude]" in shipped and "[single reviewer: codex]" in shipped
    assert "Never name" not in shipped  # blinding rule gone from production
    assert "Never name" in graded  # the campaign copy WAS blinded ("Never name or ...")


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
    assert "CODEX_TIMEOUT_S = 1200" in codex
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


def test_skill_supports_deliberate_single_and_dual_modes():
    """The skill must support a DELIBERATE single-reviewer mode (not only the
    codex-unavailable fallback) as well as dual."""
    text = (_SKILL / "SKILL.md").read_text()
    assert re.search(r"\bdual\b", text, re.I) and re.search(r"\bsingle\b", text, re.I)
    assert "Single-reviewer mode (deliberate)" in text
    # the deliberate-single note is distinct from the codex-unavailable warning
    assert "deliberate one-reviewer choice" in text


def test_gate_offers_three_way_adaptive_recommendation():
    """CLAUDE.md must ALWAYS offer Dual / Single / Skip with the recommendation
    chosen ADAPTIVELY by plan complexity, not a fixed default (restores the
    pre-campaign 'Adaptive' decision)."""
    claude = (_REPO / "CLAUDE.md").read_text()
    section = claude.split("## Plan Review Before Approval", 1)[1]
    section = section.split("\n## ", 1)[0]
    for opt in ("Dual review", "Single review", "Skip"):
        assert opt in section, f"gate offer missing the {opt!r} option"
    assert re.search(r"ADAPTIVEL?Y|adaptiv", section), "recommendation must be adaptive"
    assert "not a fixed default" in section


def test_modes_do_not_advertise_descoped_flags():
    """`--updated`/`--pr` were descoped from the initial skill (they were
    advertised-but-unimplemented). The Modes section must present them as NOT
    reimplemented / a tracked follow-up, never as active flags."""
    text = (_SKILL / "SKILL.md").read_text()
    modes = text.split("## Modes", 1)[1].split("\n## ", 1)[0]
    # the old active-flag advertisements are gone
    assert "Delta Assessment" not in text  # the `--updated` output section
    assert "= fresh re-review" not in text
    # they are named only to mark them descoped
    assert "--updated" in modes and "--pr" in modes
    assert re.search(r"not reimplemented|descoped|tracked follow-up", modes, re.I)


# --------------------------------------------------------------------------- #
# Behavioral tests for codex_review.py (reviewer 2 half) — exercise main()'s
# control flow, not source text: pins forwarded to call_codex, the sensitive-
# file notice fires before the codex call, and the exit-code contract holds.
# --------------------------------------------------------------------------- #


def test_codex_review_forwards_pins_and_prints_sensitive_notice(tmp_path, monkeypatch):
    cr = _load_codex_review()
    fake = _FakeOpenAIReview()
    monkeypatch.setattr(cr, "_load_openai_review", lambda repo_root: fake)
    monkeypatch.setattr(cr, "_codex_present", lambda mod: True)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("RENDERED PROMPT")
    out = tmp_path / "review_b.md"

    rc = cr.main(["--prompt-file", str(prompt), "--repo-root", str(tmp_path), "-o", str(out)])

    assert rc == 0
    assert out.read_text() == "CODEX REVIEW BODY"
    # S1: the sensitive-file notice ran (with the repo root + scan result)
    assert fake.notice_calls == [(str(tmp_path), (".env",))]
    # the campaign pins reach call_codex (a wrong value here would ship a
    # different, unmeasured engine while the byte-match tests still passed)
    assert len(fake.codex_calls) == 1
    call = fake.codex_calls[0]
    assert call["prompt"] == "RENDERED PROMPT"
    assert call["model"] == cr.CODEX_MODEL == "gpt-5.6-sol"
    assert call["effort"] == cr.CODEX_EFFORT == "xhigh"
    assert call["timeout_s"] == cr.CODEX_TIMEOUT_S == 1200.0


def test_codex_review_absent_returns_2_and_writes_nothing(tmp_path, monkeypatch):
    cr = _load_codex_review()
    monkeypatch.setattr(cr, "_load_openai_review", lambda repo_root: _FakeOpenAIReview())
    monkeypatch.setattr(cr, "_codex_present", lambda mod: False)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x")
    out = tmp_path / "review_b.md"

    rc = cr.main(["--prompt-file", str(prompt), "--repo-root", str(tmp_path), "-o", str(out)])

    assert rc == 2  # SKILL.md routes this to the LOUD single-Claude fallback
    assert not out.exists()


def test_codex_review_error_returns_3_and_writes_nothing(tmp_path, monkeypatch):
    cr = _load_codex_review()
    fake = _FakeOpenAIReview()

    def boom(**_kw):
        raise RuntimeError("codex exec exploded")

    fake.call_codex = boom  # type: ignore[method-assign]
    monkeypatch.setattr(cr, "_load_openai_review", lambda repo_root: fake)
    monkeypatch.setattr(cr, "_codex_present", lambda mod: True)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x")
    out = tmp_path / "review_b.md"

    rc = cr.main(["--prompt-file", str(prompt), "--repo-root", str(tmp_path), "-o", str(out)])

    assert rc == 3  # timeout/error is treated identically to absence (fallback)
    assert not out.exists()
