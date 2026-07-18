"""Pure-logic tests for the reviewer-eval engine (no codex, no network).

Covers what survives the carve-back to the minimal comparison harness: the run
store round-trip + content-hash keying, the model JSON round-trip, and the
side-by-side comparison bundle (rendered from each run's case snapshot). These
run in normal CI; skipped only when the harness isn't on disk.
"""

import pathlib
import sys

import pytest

_EVAL_ROOT = pathlib.Path(__file__).resolve().parent.parent / "tools" / "reviewer-eval"

pytestmark = pytest.mark.skipif(
    not _EVAL_ROOT.exists(),
    reason="reviewer-eval eval harness not present (isolated install)",
)

if _EVAL_ROOT.exists() and str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))


# --------------------------------------------------------------------------- #
# Modules import + model JSON round-trip.
# --------------------------------------------------------------------------- #


def test_engine_modules_import():
    import engine.compare  # noqa: F401
    import engine.models  # noqa: F401
    import engine.runner  # noqa: F401
    import engine.store  # noqa: F401


def test_models_json_roundtrip():
    from engine.models import RunResult, run_result_from_dict, to_jsonable

    rr = RunResult(
        case_id="c",
        config_id="A",
        repeat_idx=0,
        review_markdown="hi",
        model="gpt-5.4",
        run_id="c.A.r0.deadbeef",
        case_snapshot={"stratum": "s1_synthetic", "ground_truth": []},
    )
    assert run_result_from_dict(to_jsonable(rr)) == rr


# --------------------------------------------------------------------------- #
# Run store: round-trip + content-hash keying.
# --------------------------------------------------------------------------- #


def test_store_roundtrip(tmp_path):
    from engine.models import RunResult
    from engine.store import RunStore, run_key

    store = RunStore(str(tmp_path / "runs"))
    key = run_key("c", "A", 0, "tag1")
    rr = RunResult(case_id="c", config_id="A", repeat_idx=0, review_markdown="x", run_id=key)
    store.save(key, rr)
    assert store.has(key)
    assert store.load(key) == rr
    assert [r.run_id for r in store.load_all()] == [key]


def test_run_key_distinct_by_experiment_tag():
    from engine.store import run_key

    # Same case/config/repeat but different experiment identity must not collide.
    assert run_key("c", "A", 0, "tag-gpt54") != run_key("c", "A", 0, "tag-gpt55")
    # ...and the key is stable for a fixed identity.
    assert run_key("c", "A", 0, "t") == run_key("c", "A", 0, "t")


def test_run_key_distinct_by_case_tag():
    from engine.store import run_key

    base = ("c", "A", 0, "exp")
    # Same case/config/repeat/experiment but different case identity must NOT collide.
    assert run_key(*base, "casetag1") != run_key(*base, "casetag2")
    # Stable for a fixed case identity.
    assert run_key(*base, "ct") == run_key(*base, "ct")
    # case_tag genuinely participates (a case edit can't alias the no-case-tag key).
    assert run_key(*base) != run_key(*base, "ct")


# --------------------------------------------------------------------------- #
# Comparison bundle: rendered from each run's case_snapshot (not the corpus).
# --------------------------------------------------------------------------- #


def _bug(**kw):
    d = {
        "id": "c1:b1",
        "file": "f.py",
        "line_window": [10, 20],
        "bug_class": "x",
        "expected_severity": "P1",
        "rationale": "removed guard",
    }
    d.update(kw)
    return d


def _snap(stratum="s1_synthetic", **kw):
    d = {
        "title": "T",
        "stratum": stratum,
        "ground_truth": [_bug()],
        "expect_no_blockers": False,
        "allow_severities": ["P2", "P3"],
        "known_fp_topics": [],
    }
    d.update(kw)
    return d


def _run(case_id, config_id, review, snap, **kw):
    from engine.models import RunResult

    return RunResult(
        case_id=case_id,
        config_id=config_id,
        repeat_idx=0,
        review_markdown=review,
        model=kw.pop("model", "m"),
        case_snapshot=snap,
        **kw,
    )


def test_build_bundle_has_ground_truth_and_both_reviews():
    from engine.compare import build_bundle

    snap = _snap()
    runs = [
        _run("c1", "A", "A says: bug at f.py", snap, model="gpt-5.4"),
        _run("c1", "B", "B says: looks fine", snap, model="gpt-5.5"),
    ]
    out = build_bundle(runs)
    assert "c1:b1" in out
    assert "removed guard" in out
    assert "A says: bug at f.py" in out
    assert "B says: looks fine" in out
    # The grading instruction must point readers at the real rubric.
    assert "pr_review.md" in out


def test_build_bundle_marks_negative_control():
    from engine.compare import build_bundle

    snap = _snap(stratum="s3_negative", title="", ground_truth=[], expect_no_blockers=True)
    out = build_bundle([_run("cl", "A", "ok", snap, model="gpt-5.4")])
    assert "NO known bugs" in out


def test_build_bundle_infra_error_surfaced_not_as_review():
    from engine.compare import build_bundle

    snap = _snap(ground_truth=[])
    out = build_bundle([_run("c", "A", "", snap, model="gpt-5.4", infra_error="codex timeout")])
    assert "INFRA_ERROR" in out and "codex timeout" in out


def test_build_bundle_fence_survives_embedded_code_fences():
    """A review containing ``` fences must not break the bundle's own fence."""
    from engine.compare import build_bundle

    review = "Here is code:\n```python\nx = 1\n```\nthat's the bug."
    out = build_bundle([_run("c", "A", review, _snap(), model="gpt-5.4")])
    # The whole review (including its embedded fence) must appear verbatim.
    assert "```python\nx = 1\n```" in out


def test_build_bundle_renders_only_cases_with_runs():
    """A subset run renders only its own cases — no empty placeholder sections."""
    from engine.compare import build_bundle

    out = build_bundle([_run("only", "A", "R", _snap(title="Only"))])
    assert "## only" in out
    assert "_(no runs for this case)_" not in out


# --------------------------------------------------------------------------- #
# Blinded grading support (gpt-5.6 eval): deterministic mapping + sanitization.
# --------------------------------------------------------------------------- #


def test_derive_blind_mapping_deterministic_and_salt_dependent():
    from engine.compare import derive_blind_mapping

    ids = ["A", "B", "C", "D"]
    m1 = derive_blind_mapping(ids, salt="s1")
    m2 = derive_blind_mapping(ids, salt="s1")
    assert m1 == m2, "same (ids, salt) must yield the same permutation"
    assert sorted(m1.keys()) == ids
    assert sorted(m1.values()) == ["M1", "M2", "M3", "M4"], "M* namespace, disjoint from ids"
    # A different experiment (salt) should not be forced onto the same permutation:
    # across a handful of salts at least one must differ (probabilistic but with
    # 4! = 24 permutations and 8 salts, a collision of ALL is astronomically unlikely).
    others = [derive_blind_mapping(ids, salt=f"s{i}") for i in range(2, 10)]
    assert any(o != m1 for o in others), "salt must be able to reshuffle the mapping"


def test_sanitize_model_refs_scrubs_names_tiers_and_versions():
    from engine.compare import sanitize_model_refs

    text = (
        "As GPT-5.6-Sol I disagree with gpt-5.5; Sol and Terra differ, "
        "and 5.5 missed this. But we solved the solution in solver.py."
    )
    out = sanitize_model_refs(text, ["gpt-5.5", "gpt-5.6-sol"])
    low = out.lower()
    assert "gpt" not in low
    assert "[model-redacted]" in out
    # Word-boundary tier scrub: bare Sol/Terra go, but ordinary words survive.
    assert " sol " not in f" {low} ".replace("[model-redacted]", "X")
    assert "solved" in out and "solution" in out and "solver.py" in out
    assert " 5.5 " not in f" {out} "


def test_apply_blinding_strips_identity_and_preserves_originals():
    from engine.compare import apply_blinding

    snap = _snap(notes="gpt-5.5 missed this in PR #600", previous_review="gpt-5.5 said fine")
    rr = _run(
        "c1",
        "B",
        "As gpt-5.6-sol, I found the bug.",
        snap,
        model="gpt-5.6-sol",
        effort="max",
        cli_version="codex-cli 0.144.5",
        latency_s=123.0,
    )
    blinded = apply_blinding([rr], {"B": "M2"}, ["gpt-5.6-sol", "gpt-5.5"])[0]
    assert blinded.config_id == "M2"
    assert blinded.model == "" and blinded.effort == "" and blinded.cli_version == ""
    assert blinded.latency_s == 0.0
    assert "gpt" not in blinded.review_markdown.lower()
    assert "gpt" not in blinded.case_snapshot["notes"].lower()
    assert "gpt" not in blinded.case_snapshot["previous_review"].lower()
    # Original untouched (dataclasses.replace copies).
    assert rr.config_id == "B" and rr.model == "gpt-5.6-sol" and "gpt" in rr.review_markdown


def test_blinded_bundle_has_no_identity_leaks():
    from engine.compare import apply_blinding, build_bundle, derive_blind_mapping

    snap = _snap()
    runs = [
        _run("c1", "A", "A-model output", snap, model="gpt-5.5", effort="xhigh", latency_s=60.0),
        _run("c1", "B", "B-model output", snap, model="gpt-5.6-sol", effort="max", latency_s=600.0),
    ]
    mapping = derive_blind_mapping(["A", "B"], salt="x")
    out = build_bundle(apply_blinding(runs, mapping, ["gpt-5.5", "gpt-5.6-sol"]), redact_meta=True)
    low = out.lower()
    assert "gpt" not in low
    assert "latency" not in low and "cli " not in low
    assert "### A " not in out and "### B " not in out
    for label in mapping.values():
        assert f"### {label} — review" in out


def test_build_bundle_labels_effort_when_recorded():
    """Unblinded bundles label multi-effort arms (e.g. `@ max`) so graders can
    tell B from D; artifacts without a recorded effort render exactly as before."""
    from engine.compare import build_bundle

    snap = _snap()
    with_effort = build_bundle([_run("c", "D", "x", snap, model="gpt-5.6-sol", effort="max")])
    assert "### D (gpt-5.6-sol @ max) — review" in with_effort
    without = build_bundle([_run("c", "A", "x", snap, model="gpt-5.4")])
    assert "### A (gpt-5.4) — review" in without, "legacy artifacts render unchanged"


def test_negative_control_render_matches_allow_severities():
    """The FP rule in the rendered bundle must derive from the case's
    allow_severities — a P3-only control must not read as 'P2 acceptable'
    (local review R1 P3)."""
    from engine.compare import build_bundle

    snap = _snap(
        stratum="s3_negative",
        title="",
        ground_truth=[],
        expect_no_blockers=True,
        allow_severities=["P3"],
    )
    out = build_bundle([_run("c", "A", "ok", snap, model="m")])
    assert "outside the allowed set (P3) is a FALSE POSITIVE" in out
    assert "any P0/P1 finding is a FALSE POSITIVE" not in out
