"""Contract tests for the shipped `.claude/skills/plan-review/` engine.

Guards that production == the plan-review engine Campaign 1 graded: the bundled
prompt artifacts are byte-identical to the campaign-graded `candidates/` copies, and
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
        assert shipped == graded, f"{name} drifted from the campaign-graded candidate"
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
    assert "CODEX_TIMEOUT_S = 2400" in codex
    # and codex_review.py actually passes them to call_codex
    assert "model=CODEX_MODEL" in codex and "effort=CODEX_EFFORT" in codex
    assert "timeout_s=CODEX_TIMEOUT_S" in codex

    # SKILL.md STATES the ceiling to the reader ("caps at Ns and exits 3"), so
    # bumping CODEX_TIMEOUT_S without editing the prose leaves the skill's own
    # instructions lying about its behavior. Derive both and compare rather than
    # hard-coding the number twice, so this cannot drift on the next bump.
    code_cap = re.search(r"^CODEX_TIMEOUT_S\s*=\s*([\d.]+)", codex, re.M)
    assert code_cap, "CODEX_TIMEOUT_S assignment not found in codex_review.py"
    doc_cap = re.search(r"caps at (\d+)s and exits 3", (_SKILL / "SKILL.md").read_text())
    assert doc_cap, "SKILL.md no longer states the codex cap - keep it or drop this pin"
    assert float(doc_cap.group(1)) == float(code_cap.group(1)), (
        f"SKILL.md says {doc_cap.group(1)}s but codex_review.py caps at " f"{code_cap.group(1)}s"
    )

    skill = (_SKILL / "SKILL.md").read_text()
    # BOTH Claude subagents (reviewer 1 AND merge) select model=opus — assert
    # each section separately, not just one match anywhere (round-4: a single
    # global match did not prove both invocations). "opus" is a runtime family
    # alias (Task takes aliases, not exact IDs), documented as such, not an
    # immutable pin.
    reviewer_sec = skill.split("### 3. Reviewer 1", 1)[1].split("### 4.", 1)[0]
    merge_sec = skill.split("### 5. Merge + verify", 1)[1].split("### 6.", 1)[0]
    assert re.search(r'model[=:]\s*["\']opus', reviewer_sec), "reviewer 1 must select model=opus"
    assert re.search(r'model[=:]\s*["\']opus', merge_sec), "merge subagent must select model=opus"
    # and the alias is documented as a runtime alias, not overclaimed as a pin
    norm_skill = " ".join(skill.split())
    assert "runtime alias" in norm_skill and "immutable pin" in norm_skill
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


def test_claude_md_skip_branch_initializes_scratch():
    """The standalone CLAUDE.md Skip branch (bypasses the skill) must also
    `mkdir -p "$SCRATCH"` before writing plan-path.txt — on a fresh worktree
    `.git/plan-review` does not exist yet, so the Write would fail (round-3)."""
    claude = (_REPO / "CLAUDE.md").read_text()
    skip = claude.split("**If skip**", 1)[1].split("**Rollback**", 1)[0]
    assert 'mkdir -p "$SCRATCH"' in skip, "Skip branch must create the scratch dir"
    assert skip.index('mkdir -p "$SCRATCH"') < skip.index(
        "plan-path.txt"
    ), "mkdir -p must precede the Skip branch's plan-path.txt Write"


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
    assert call["timeout_s"] == cr.CODEX_TIMEOUT_S == 2400.0


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


# --------------------------------------------------------------------------- #
# Round-2 AI-review lifecycle contracts (SKILL.md is agent-executed prose, so
# these assert the corrected instructions are present — the code-level parts
# have behavioral tests above).
# --------------------------------------------------------------------------- #


def test_intermediate_files_are_invocation_scoped():
    """P0/P1: concurrent same-worktree reviews must not cross-wire, AND the work
    dir must be HELPER-emitted (safe-charset, HOME-prefixed) rather than derived
    from the repo/worktree path (round-6 path-injection fix). The four
    intermediates live under `<work_dir>`; there is no mktemp-under-git-path and
    no manual `rm -rf` (persist/abort clean it)."""
    text = (_SKILL / "SKILL.md").read_text()
    # work_dir comes from the snapshot helper's printed output, not a repo mktemp
    assert "`work_dir`" in text, "work_dir must be a helper-emitted field"
    assert "mktemp" not in text, "work_dir must NOT be built from the repo/worktree path"
    for name in ("reviewer_prompt.txt", "review_a.md", "review_b.md", "merge_prompt.txt"):
        assert f"<work_dir>/{name}" in text, f"{name} must be under <work_dir>"
        assert f'"$SCRATCH/{name}"' not in text
    # cleanup is via persist/abort (which remove work_dir), not a shell rm -rf of
    # a pasted path literal
    assert 'rm -rf "<work_dir>"' not in text


def test_assessment_is_deterministic():
    """P2: `assessment` must be derived by a deterministic count->label rule from
    the ACTUAL finding lines (not the reviewer's self-reported table, CI round),
    cross-checked against the table; a mismatch/malformed report must abort."""
    norm = " ".join((_SKILL / "SKILL.md").read_text().split())
    # the three-branch mapping is spelled out (label + its trigger)
    assert "any **P0 or P1**" in norm and "Significant issues found" in norm
    assert "else any **P2**" in norm and "Minor revisions recommended" in norm
    assert "Ready to implement" in norm
    # counts come from the finding lines and are cross-checked against the table
    assert "count the finding lines" in norm and "cross-check" in norm
    # a mismatch / malformed report -> abort, not a persisted guess
    assert "malformed" in norm and "do NOT persist" in norm


def test_engine_provenance_is_machine_readable():
    """P2: the persisted body carries a machine-readable engine marker for all
    three modes, and Revise reads THAT (not fragile prose)."""
    text = (_SKILL / "SKILL.md").read_text()
    for mode in ("dual", "single", "single-fallback"):
        assert f"plan-review-engine: {mode}" in text, f"missing {mode} marker"
    # Revise reads the marker
    revise = text.split("## Revise phase", 1)[1]
    assert "plan-review-engine" in revise


def test_codex_read_surface_is_disclosed():
    """Verifies the accepted-surface DISCLOSURE, not a filesystem boundary. Per
    the user's parity+tracking decision the codex read surface is NOT confined
    (same as /ai-review-local); this asserts the skill discloses it in-line and
    cross-links the tracked isolation follow-up. It is not a security control —
    a real boundary would need OS-level isolation (tracked, TODO.md:L55)."""
    text = (_SKILL / "SKILL.md").read_text()
    assert "read-only" in text and "read surface" in text.lower()
    assert "ai-review-local" in text  # names the accepted-parity surface
    assert "TODO.md" in text  # cross-links the tracked isolation follow-up


def test_ingress_paths_are_self_contained():
    """P1 (round-4): ingress must not rely on shell vars persisting across Bash
    tool calls or the Write tool expanding them. The skill dir is a literal (no
    `$SKILL`), no Write target is a `$SCRATCH/...` token, the scratch dir is
    printed for the Write-tool literal, and snapshot/check re-derive it inline."""
    skill_raw = (_SKILL / "SKILL.md").read_text()
    skill = " ".join(skill_raw.split())  # normalize: paths can wrap across lines
    # the $SKILL variable is gone — the skill dir is used literally
    assert "$SKILL/" not in skill_raw, "$SKILL must be the literal .claude/skills/plan-review/"
    assert ".claude/skills/plan-review/render.py" in skill
    # no Write-tool target the tool cannot expand
    assert "$SCRATCH/plan-path.txt" not in skill_raw, "Write target must be a printed literal"
    # scratch is printed for the Write literal, and re-derived inline for snapshot/check
    assert 'echo "$SCRATCH"' in skill
    assert "$(git rev-parse --git-path plan-review)/plan-path.txt" in skill

    # the CLAUDE.md Skip branch (bypasses the skill) meets the same contract
    skip_raw = (
        (_REPO / "CLAUDE.md").read_text().split("**If skip**", 1)[1].split("**Rollback**", 1)[0]
    )
    skip = " ".join(skip_raw.split())
    assert "$SCRATCH/plan-path.txt" not in skip_raw, "Skip Write target must be a printed literal"
    assert 'echo "$SCRATCH"' in skip
    assert "$(git rev-parse --git-path plan-review)/plan-path.txt" in skip


def test_skip_paths_release_snapshot_on_failure():
    """P2 (round-4): both Skip paths (SKILL.md revise-Skipped, CLAUDE.md Skip)
    must `abort` on a pre-persist failure so the snapshot is not retained."""
    skill_revise = " ".join(
        (_SKILL / "SKILL.md").read_text().split("## Revise phase", 1)[1].split()
    )
    assert "abort" in skill_revise and "BEFORE persist" in skill_revise
    skip = " ".join(
        (_REPO / "CLAUDE.md")
        .read_text()
        .split("**If skip**", 1)[1]
        .split("**Rollback**", 1)[0]
        .split()
    )
    assert "abort" in skip and "not retained" in skip


def test_abort_is_pre_persist_only_and_strict():
    """CI round: `persist` self-cleans its own failures, so the skill NEVER aborts
    after persist (that could only hit a wrong/stale token). `--allow-missing` was
    removed entirely; every abort in the skill is a plain pre-persist cleanup."""
    text = (_SKILL / "SKILL.md").read_text()
    assert "--allow-missing" not in text, "the --allow-missing escape hatch is gone"
    # the pre-persist snapshot-lifecycle callout still aborts (plain)
    callout = " ".join(
        text.split("Release the snapshot exactly once", 1)[1].split("### 2.", 1)[0].split()
    )
    assert 'abort --state-file "<state_path>"' in callout, "pre-persist callout must abort"
    # the persist step explicitly says NOT to abort after persist
    persist = " ".join(text.split("### 7. Persist", 1)[1].split("### 8.", 1)[0].split())
    assert "Do NOT abort after persist" in persist
    assert "self-cleans" in persist
