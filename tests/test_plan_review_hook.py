"""Behavioral tests for the content-hash ExitPlanMode gate.

Drives `.claude/hooks/check-plan-review.py` as a subprocess — JSON payload on
stdin, a temporary HOME holding the plans directory — and asserts the full
allow/deny decision table. Every deny asserts the PreToolUse OUTPUT PROTOCOL,
not just the decision: exit code 0 AND well-formed
``hookSpecificOutput.permissionDecision: "deny"`` JSON on stdout (the old shell
hook's documented failure mode was exit 2 reading as "hook error" instead of a
deliberate block).
"""

import hashlib
import json
import pathlib
import subprocess

import pytest

_HOOK = (
    pathlib.Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "check-plan-review.py"
)

pytestmark = pytest.mark.skipif(
    not _HOOK.exists(), reason="hook not present (installed distribution)"
)


def _run(payload, home):
    """Run the hook with ``payload`` (dict → JSON, str → raw bytes) and HOME=home."""
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        ["python3", str(_HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        timeout=30,
    )


def _assert_allow(cp):
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.strip() == "", f"allow must be silent, got: {cp.stdout!r}"


def _assert_deny(cp, reason_contains):
    """The shared protocol helper: exit 0 + well-formed deny JSON, never exit 2."""
    assert cp.returncode == 0, f"deny must exit 0 (protocol), got {cp.returncode}: {cp.stderr}"
    out = json.loads(cp.stdout)
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    assert reason_contains in hso["permissionDecisionReason"]


def _write_plan(home, name="snazzy-plan.md", content="# The plan\n\ndo things\n"):
    plans = home / ".claude" / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    plan = plans / name
    plan.write_text(content)
    return plan


def _review_name(plan):
    """The collision-free key the hook derives: canonical basename + canonical-
    path digest (mirrors plan_snapshot._review_path)."""
    import os

    real = os.path.realpath(str(plan))
    digest = hashlib.sha256(real.encode()).hexdigest()[:12]
    return f"{plan.name[:-3]}.{digest}.review.md"


def _write_review(plan, plan_field=None, sha=None, extra=""):
    """Review file for ``plan`` with standard frontmatter; sha=None → real hash."""
    if sha is None:
        sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    review = plan.parent / _review_name(plan)
    review.write_text(
        "---\n"
        f"plan: {plan_field if plan_field is not None else plan}\n"
        f"plan_sha256: {sha}\n"
        'reviewed_at: "2026-07-20T12:00:00Z"\n'
        'assessment: "No critical issues found"\n'
        f"{extra}"
        "---\n\n## Overall Assessment\n\nfine\n"
    )
    return review


def _payload(plan):
    return {"tool_input": {"plan": plan.read_text(), "planFilePath": str(plan)}}


# --------------------------------------------------------------------------- #
# Allow paths
# --------------------------------------------------------------------------- #


def test_allow_when_payload_has_neither_field(tmp_path):
    _assert_allow(_run({"tool_input": {}}, tmp_path))


def test_allow_when_hash_matches(tmp_path):
    plan = _write_plan(tmp_path)
    _write_review(plan)
    _assert_allow(_run(_payload(plan), tmp_path))


def test_allow_skipped_marker(tmp_path):
    plan = _write_plan(tmp_path)
    _write_review(plan, extra="flags: []\n")
    review = plan.parent / _review_name(plan)
    review.write_text(review.read_text().replace("No critical issues found", "Skipped"))
    _assert_allow(_run(_payload(plan), tmp_path))


def test_allow_with_tilde_plan_field(tmp_path):
    """A ``~/...`` plan: value must match after normalization on both sides."""
    plan = _write_plan(tmp_path)
    _write_review(plan, plan_field=f"~/.claude/plans/{plan.name}")
    _assert_allow(_run(_payload(plan), tmp_path))


def test_allow_with_quoted_frontmatter_values(tmp_path):
    """The frontmatter contract tolerates optional surrounding quotes."""
    plan = _write_plan(tmp_path)
    sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    review = plan.parent / _review_name(plan)
    review.write_text(f'---\nplan: "{plan}"\nplan_sha256: "{sha}"\n---\nbody\n')
    _assert_allow(_run(_payload(plan), tmp_path))


def test_allow_with_symlinked_plan_path(tmp_path):
    """realpath normalization: a symlinked payload path matches the real one."""
    plan = _write_plan(tmp_path)
    link_dir = tmp_path / "link"
    link_dir.symlink_to(plan.parent)
    _write_review(plan)
    payload = {"tool_input": {"plan": plan.read_text(), "planFilePath": str(link_dir / plan.name)}}
    _assert_allow(_run(payload, tmp_path))


# --------------------------------------------------------------------------- #
# Deny paths (each asserts the full protocol via the shared helper)
# --------------------------------------------------------------------------- #


def test_deny_on_payload_drift_plan_without_path(tmp_path):
    """The fail-closed branch: plan text present, planFilePath absent — the
    signature of a harness payload-shape change must never silently disable
    the gate."""
    _assert_deny(
        _run({"tool_input": {"plan": "# a plan"}}, tmp_path),
        "payload shape has changed",
    )


def test_deny_on_empty_plan_string_without_path(tmp_path):
    """KEY PRESENCE, not truthiness: {"plan": ""} is a plan-shaped payload with
    a missing path — it must hit the drift branch, never allow."""
    _assert_deny(
        _run({"tool_input": {"plan": ""}}, tmp_path),
        "payload shape has changed",
    )


def test_deny_on_non_dict_tool_input(tmp_path):
    _assert_deny(_run({"tool_input": "garbage"}, tmp_path), "not an object")


def test_deny_on_non_object_payload(tmp_path):
    _assert_deny(_run('["a", "list"]', tmp_path), "non-object hook payload")


def test_deny_when_payload_text_disagrees_with_file(tmp_path):
    """Tripwire: the harness injects the plan from disk, so payload text that
    differs from the file means the file changed under us (or payload
    semantics changed) — verify nothing."""
    plan = _write_plan(tmp_path)
    _write_review(plan)
    payload = {"tool_input": {"plan": "ENTIRELY different content", "planFilePath": str(plan)}}
    _assert_deny(_run(payload, tmp_path), "does not match the on-disk plan")


def test_deny_on_truncated_frontmatter(tmp_path):
    """A frontmatter block with no closing --- is malformed and approves
    nothing, even if its keys would otherwise match."""
    import hashlib as _hashlib

    plan = _write_plan(tmp_path)
    sha = _hashlib.sha256(plan.read_bytes()).hexdigest()
    review = plan.parent / _review_name(plan)
    review.write_text(f"---\nplan: {plan}\nplan_sha256: {sha}\n")  # never closed
    _assert_deny(_run(_payload(plan), tmp_path), "different plan")


def test_deny_on_missing_review(tmp_path):
    plan = _write_plan(tmp_path)
    _assert_deny(_run(_payload(plan), tmp_path), "No plan review found")


def test_deny_on_hash_mismatch(tmp_path):
    plan = _write_plan(tmp_path)
    _write_review(plan)
    plan.write_text(plan.read_text() + "\nrevised after review\n")
    _assert_deny(_run(_payload(plan), tmp_path), "stale")


def test_deny_on_plan_path_mismatch(tmp_path):
    """A review whose plan: field points at a DIFFERENT plan must not approve
    this one (the cross-plan aliasing the old sentinel design allowed)."""
    plan = _write_plan(tmp_path)
    other = _write_plan(tmp_path, name="other-plan.md", content="other\n")
    _write_review(plan, plan_field=str(other))
    _assert_deny(_run(_payload(plan), tmp_path), "different plan")


def test_deny_on_missing_sha_field(tmp_path):
    """A pre-hash-gate review file (no plan_sha256) cannot approve anything."""
    plan = _write_plan(tmp_path)
    review = plan.parent / _review_name(plan)
    review.write_text(f"---\nplan: {plan}\nreviewed_at: x\n---\nbody\n")
    _assert_deny(_run(_payload(plan), tmp_path), "no plan_sha256")


def test_deny_on_unreadable_plan_file(tmp_path):
    payload = {"tool_input": {"plan": "x", "planFilePath": str(tmp_path / "does-not-exist.md")}}
    _assert_deny(_run(payload, tmp_path), "unreadable")


def test_deny_on_malformed_json(tmp_path):
    _assert_deny(_run("this is not json{{", tmp_path), "could not parse")


def test_deny_on_review_without_frontmatter(tmp_path):
    plan = _write_plan(tmp_path)
    review = plan.parent / _review_name(plan)
    review.write_text("no frontmatter here\n")
    _assert_deny(_run(_payload(plan), tmp_path), "different plan")


# --------------------------------------------------------------------------- #
# The race the redesign exists to kill
# --------------------------------------------------------------------------- #


def test_two_concurrent_plans_do_not_cross_approve(tmp_path):
    """The old sentinel design let worktree B's review overwrite the pointer and
    approve worktree A's unreviewed plan. Hash-keyed reviews are per-plan: each
    plan approves iff ITS review matches ITS bytes, regardless of order."""
    plan_a = _write_plan(tmp_path, name="worktree-a-plan.md", content="A's plan\n")
    plan_b = _write_plan(tmp_path, name="worktree-b-plan.md", content="B's plan\n")
    _write_review(plan_b)  # only B reviewed
    _assert_deny(_run(_payload(plan_a), tmp_path), "No plan review found")
    _assert_allow(_run(_payload(plan_b), tmp_path))
    # A gets its review; both now pass independently.
    _write_review(plan_a)
    _assert_allow(_run(_payload(plan_a), tmp_path))
    _assert_allow(_run(_payload(plan_b), tmp_path))


# --------------------------------------------------------------------------- #
# Round-6 payload-type discipline (C2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_plan", [None, 123, ["a"], {"x": 1}])
def test_deny_on_non_string_plan_with_valid_path(tmp_path, bad_plan):
    plan = _write_plan(tmp_path)
    _write_review(plan)
    payload = {"tool_input": {"plan": bad_plan, "planFilePath": str(plan)}}
    _assert_deny(_run(payload, tmp_path), "not a string")


def test_deny_on_empty_plan_string_mismatching_file(tmp_path):
    """plan: "" with a NONEMPTY file is a consistency violation, not a skip."""
    plan = _write_plan(tmp_path)
    _write_review(plan)
    payload = {"tool_input": {"plan": "", "planFilePath": str(plan)}}
    _assert_deny(_run(payload, tmp_path), "does not match the on-disk plan")


def test_allow_empty_plan_string_matching_empty_file(tmp_path):
    plan = _write_plan(tmp_path, content="")
    _write_review(plan)
    payload = {"tool_input": {"plan": "", "planFilePath": str(plan)}}
    _assert_allow(_run(payload, tmp_path))


def test_deny_when_tool_input_key_missing_entirely(tmp_path):
    """A payload with NO tool_input field is a shape change, not an empty
    input — it must fail closed (CI round-6: `{}` previously allowed)."""
    _assert_deny(_run({}, tmp_path), "no tool_input field")


def test_allow_when_tool_input_present_but_empty(tmp_path):
    _assert_allow(_run({"tool_input": {}}, tmp_path))
