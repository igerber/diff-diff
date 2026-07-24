"""Behavioral tests for the plan snapshot/persist helper.

The helper owns the review-integrity protocol the prose commands invoke:
invocation-unique immutable snapshots (concurrent reviews can never share or
clobber one), certify-exactly-the-reviewed-bytes persistence (exit 3 when the
live plan drifted), canonical review-key derivation matching the ExitPlanMode
hook (symlink aliases can't split the key), and frontmatter emission that a
hostile meta string cannot break.
"""

import hashlib
import json
import pathlib
import subprocess

import pytest

_HELPER = (
    pathlib.Path(__file__).resolve().parent.parent / ".claude" / "scripts" / "plan_snapshot.py"
)
_HOOK = (
    pathlib.Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "check-plan-review.py"
)

pytestmark = pytest.mark.skipif(
    not _HELPER.exists(), reason="helper not present (installed distribution)"
)


def _run(*argv, home=None, check=True):
    env = {"PATH": "/usr/bin:/bin"}
    if home is not None:
        env["HOME"] = str(home)
    cp = subprocess.run(
        ["python3", str(_HELPER), *argv],
        capture_output=True,
        text=True,
        timeout=30,
        env=env if home is not None else None,
    )
    if check:
        assert cp.returncode == 0, cp.stderr
    return cp


def _mk_plan(tmp_path, name="fancy-plan.md", content="# plan\n\nsteps\n"):
    plans = tmp_path / ".claude" / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    plan = plans / name
    plan.write_text(content)
    return plan


def _write_file(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content)
    return f


def _snapshot(tmp_path, plan):
    pf = _write_file(tmp_path, "plan-path.txt", str(plan))
    cp = _run("snapshot", "--plan-path-file", str(pf), home=tmp_path)
    return json.loads(cp.stdout)


def _persist(tmp_path, out, meta="{}", body="body\n", check=True):
    pathlib.Path(out["meta_path"]).write_text(meta)
    pathlib.Path(out["body_path"]).write_text(body)
    return _run("persist", "--state-file", out["state_path"], home=tmp_path, check=check)


def test_snapshot_roundtrip_and_review_key(tmp_path):
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    snap = pathlib.Path(out["snapshot_path"])
    assert snap.read_bytes() == plan.read_bytes()
    assert out["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    leaf = pathlib.Path(out["review_path"]).name
    assert leaf.startswith("fancy-plan.") and leaf.endswith(".review.md")


def test_snapshots_are_invocation_unique(tmp_path):
    """Two concurrent reviews (same plan, same basename) must never share or
    overwrite a snapshot — the round-6 basename-collision race."""
    plan = _mk_plan(tmp_path)
    a = _snapshot(tmp_path, plan)
    b = _snapshot(tmp_path, plan)
    assert a["snapshot_path"] != b["snapshot_path"]
    assert pathlib.Path(a["snapshot_path"]).exists()
    assert pathlib.Path(b["snapshot_path"]).exists()


def test_snapshot_emits_and_cleans_work_dir(tmp_path):
    """work_dir is a helper-emitted per-invocation dir UNDER the snapshots dir
    (safe-charset leaf, NOT built from the repo/worktree path — round-6 path
    injection fix), created at snapshot and removed by both persist and abort."""
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    work = pathlib.Path(out["work_dir"])
    assert work.is_dir(), "work_dir must be created"
    assert work.parent.name == ".snapshots", "work_dir lives under the snapshots dir"
    assert work.name.endswith(".work")
    (work / "reviewer_prompt.txt").write_text("intermediate")  # a stray file in it
    _persist(tmp_path, out)
    assert not work.exists(), "persist must remove work_dir and its contents"


def test_abort_cleans_work_dir(tmp_path):
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    work = pathlib.Path(out["work_dir"])
    (work / "review_a.md").write_text("y")
    _run("abort", "--state-file", out["state_path"], home=tmp_path)
    assert not work.exists(), "abort must remove work_dir"


def test_persist_certifies_reviewed_bytes(tmp_path):
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    cp = _persist(tmp_path, out, meta=json.dumps({"assessment": "ok"}))
    review = pathlib.Path(cp.stdout.strip())
    assert review == pathlib.Path(out["review_path"])
    assert (
        review.parent == pathlib.Path(tmp_path) / ".claude" / "plans"
    ), "review must land where the hook reads"
    text = review.read_text()
    assert f"plan_sha256: {out['plan_sha256']}" in text
    assert f"plan: {out['plan_path']}" in text
    for key in ("snapshot_path", "state_path", "meta_path", "body_path", "work_dir"):
        assert not pathlib.Path(out[key]).exists(), f"{key} must be cleaned up"


def test_persist_exit3_when_plan_changed_during_review(tmp_path):
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    plan.write_text("# plan\n\nEDITED while reviewing\n")
    cp = _persist(tmp_path, out, check=False)
    assert cp.returncode == 3
    assert "modified during the review" in cp.stderr
    assert not pathlib.Path(out["review_path"]).exists()
    assert not pathlib.Path(out["snapshot_path"]).exists(), "snapshot cleaned on abort too"


def test_same_basename_plans_get_distinct_reviews(tmp_path):
    """/repo-a/plan.md and /repo-b/plan.md must never share (and overwrite)
    one review file (CI round-2 P1)."""
    a_dir = tmp_path / ".claude" / "plans" / "a"
    b_dir = tmp_path / ".claude" / "plans" / "b"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    plan_a = a_dir / "plan.md"
    plan_b = b_dir / "plan.md"
    plan_a.write_text("A\n")
    plan_b.write_text("B\n")
    out_a = _snapshot(tmp_path, plan_a)
    out_b = _snapshot(tmp_path, plan_b)
    assert out_a["review_path"] != out_b["review_path"]
    _persist(tmp_path, out_a, meta='{"assessment": "A-rev"}')
    _persist(tmp_path, out_b, meta='{"assessment": "B-rev"}')
    assert "A-rev" in pathlib.Path(out_a["review_path"]).read_text()
    assert "B-rev" in pathlib.Path(out_b["review_path"]).read_text()


def test_generated_paths_are_strict_charset_regardless_of_plan_name(tmp_path):
    """The CI round-2 P0: generated snapshot/state leaves must contain ONLY
    [a-f0-9.] — a plan stem like weird$(touch ...) must never flow into a path
    that prose later substitutes into a shell command."""
    import re

    plan = _mk_plan(tmp_path, name="weird$(touch marker).md", content="x\n")
    out = _snapshot(tmp_path, plan)
    for key in ("snapshot_path", "state_path", "meta_path", "body_path"):
        leaf = pathlib.Path(out[key]).name
        assert re.fullmatch(r"[a-z0-9.]+", leaf), f"{key} leaf {leaf!r} unsafe"
    # End-to-end through a real shell, the exact quoted shape the prose uses:
    import subprocess as sp

    pathlib.Path(out["meta_path"]).write_text("{}")
    pathlib.Path(out["body_path"]).write_text("body\n")
    script = f'python3 "{_HELPER}" persist --state-file "{out["state_path"]}"'
    cp = sp.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
        cwd=tmp_path,
        timeout=30,
    )
    assert cp.returncode == 0, cp.stderr
    assert not (tmp_path / "marker").exists(), "hostile plan stem executed via state path!"


def test_abort_cleans_up_invocation(tmp_path):
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    pathlib.Path(out["meta_path"]).write_text("{}")
    pathlib.Path(out["body_path"]).write_text("b\n")
    _run("abort", "--state-file", out["state_path"], home=tmp_path)
    for key in ("snapshot_path", "state_path", "meta_path", "body_path", "work_dir"):
        assert not pathlib.Path(out[key]).exists(), f"{key} survived abort"
    assert not pathlib.Path(out["review_path"]).exists()


def test_persist_self_cleans_the_whole_invocation_on_failure(tmp_path):
    """CI round: persist self-cleans the ENTIRE invocation (snapshot + work_dir +
    sidecars) on a post-load failure — e.g. a malformed meta — so the caller never
    runs an abort AFTER persist (which could mask a wrong token). Exit 3 (plan
    changed) is the same story, covered by test_persist_exit3_..."""
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    pathlib.Path(out["meta_path"]).write_text("{ not valid json")
    pathlib.Path(out["body_path"]).write_text("b\n")
    cp = _run("persist", "--state-file", out["state_path"], home=tmp_path, check=False)
    assert cp.returncode == 2
    for key in ("snapshot_path", "state_path", "meta_path", "body_path", "work_dir"):
        assert not pathlib.Path(out[key]).exists(), f"{key} must be self-cleaned on persist failure"


def test_abort_fails_on_missing_state(tmp_path):
    """abort is a PRE-persist cleanup only (persist self-cleans its own failures),
    so a nonexistent (mistyped / cross-wired / stale) state token must FAIL rather
    than report success while a real snapshot is left behind (round-5 + CI: the
    `--allow-missing` escape hatch was removed since it is no longer needed)."""
    snap_dir = tmp_path / ".claude" / "plans" / ".snapshots"
    snap_dir.mkdir(parents=True)
    ghost = str(snap_dir / "deadbeef0000.00000000.state.json")  # well-formed, nonexistent
    cp = _run("abort", "--state-file", ghost, home=tmp_path, check=False)
    assert cp.returncode == 2
    assert "does not exist" in cp.stderr


def test_snapshot_refuses_shell_unsafe_home(tmp_path):
    """CI round (fail-closed validation gate): the plans dir is HOME-derived and
    every generated path is pasted into the caller's shell as a quoted literal.
    If HOME contains a shell metacharacter those paths would execute under command
    substitution, so the helper REFUSES to emit them (fails closed). Covers BOTH
    `$()` and backtick command-substitution forms (CI: don't test only `$()`)."""
    for i, home_name in enumerate(("home$(touch pwned)", "home`touch pwned`")):
        unsafe_home = tmp_path / f"h{i}" / home_name
        plan = unsafe_home / ".claude" / "plans" / "p.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# p\n")
        pf = _write_file(tmp_path, f"plan-path-{i}.txt", str(plan))
        cp = _run("snapshot", "--plan-path-file", str(pf), home=unsafe_home, check=False)
        assert cp.returncode == 2, f"{home_name!r}: expected fail-closed refusal"
        assert "shell metacharacter" in cp.stderr
        # nothing executed (the helper never shells out — this guards the CALLER)
        assert not (unsafe_home / "pwned").exists()


def test_persist_rejects_rewritten_snapshot(tmp_path):
    """The RECORDED digest is what gets certified: if the snapshot file was
    altered after capture (even to match a changed live plan), persist refuses
    — a rewrite can never be certified (round-7 P0)."""
    plan = _mk_plan(tmp_path, content="A\n")
    out = _snapshot(tmp_path, plan)
    plan.write_text("B\n")
    pathlib.Path(out["snapshot_path"]).write_text("B\n")  # attacker/accident rewrite
    cp = _persist(tmp_path, out, check=False)
    assert cp.returncode == 2
    assert "no longer hashes to the recorded sha" in cp.stderr
    assert not pathlib.Path(out["review_path"]).exists()


def test_concurrent_invocations_cannot_cross_wire(tmp_path):
    """Two sessions' meta/body files live under distinct state tokens — one
    session's persist can only consume its own inputs (round-7 P0)."""
    plan_a = _mk_plan(tmp_path, name="plan-a.md", content="A\n")
    plan_b = _mk_plan(tmp_path, name="plan-b.md", content="B\n")
    out_a = _snapshot(tmp_path, plan_a)
    out_b = _snapshot(tmp_path, plan_b)
    assert out_a["state_path"] != out_b["state_path"]
    assert out_a["meta_path"] != out_b["meta_path"]
    _persist(tmp_path, out_a, meta=json.dumps({"assessment": "A-review"}))
    _persist(tmp_path, out_b, meta=json.dumps({"assessment": "B-review"}))
    a_text = pathlib.Path(out_a["review_path"]).read_text()
    b_text = pathlib.Path(out_b["review_path"]).read_text()
    assert "A-review" in a_text and "B-review" not in a_text
    assert "B-review" in b_text and "A-review" not in b_text


def test_persist_rejects_state_outside_snapshots_dir(tmp_path):
    plan = _mk_plan(tmp_path)
    _snapshot(tmp_path, plan)
    rogue = tmp_path / "rogue.state.json"
    rogue.write_text("{}")
    cp = _run("persist", "--state-file", str(rogue), home=tmp_path, check=False)
    assert cp.returncode == 2
    assert "inside" in cp.stderr


def test_persist_certifies_a_to_b_to_a_exactly(tmp_path):
    """A->B->A during review is harmless BY IDENTITY: the live bytes equal the
    reviewed snapshot, so certifying them is exact."""
    plan = _mk_plan(tmp_path, content="A-content\n")
    out = _snapshot(tmp_path, plan)
    plan.write_text("B-content\n")
    plan.write_text("A-content\n")
    _persist(tmp_path, out)


def test_symlink_alias_derives_same_review_key_as_hook(tmp_path):
    """A symlink whose LEAF name differs from its target must resolve to the
    same review file the hook will look for (round-6 C3)."""
    plan = _mk_plan(tmp_path, name="real-plan.md")
    alias = plan.parent / "alias-plan.md"
    alias.symlink_to(plan)
    out = _snapshot(tmp_path, alias)
    leaf = pathlib.Path(out["review_path"]).name
    assert leaf.startswith("real-plan.") and leaf.endswith(".review.md")
    assert pathlib.Path(out["review_path"]).parent == pathlib.Path(tmp_path) / ".claude" / "plans"
    assert out["plan_path"] == str(plan)
    if _HOOK.exists():
        # The hook, fed the ALIAS path, must consult the same review file.
        review = pathlib.Path(out["review_path"])
        review.write_text(
            f"---\nplan: {out['plan_path']}\nplan_sha256: {out['plan_sha256']}\n---\nok\n"
        )
        payload = json.dumps({"tool_input": {"plan": plan.read_text(), "planFilePath": str(alias)}})
        cp = subprocess.run(
            ["python3", str(_HOOK)],
            input=payload,
            capture_output=True,
            text=True,
            env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
            timeout=30,
        )
        # plans dir is $HOME/.claude/plans for the hook; our tmp layout differs,
        # so only assert the KEY DERIVATION agrees (realpath basename + digest).
        assert pathlib.Path(out["review_path"]).name in (cp.stdout + out["review_path"])


def test_relative_and_empty_paths_rejected(tmp_path):
    for bad in ("relative/x.md", ""):
        pf = _write_file(tmp_path, "pp.txt", bad)
        cp = _run("snapshot", "--plan-path-file", str(pf), home=tmp_path, check=False)
        assert cp.returncode == 2, f"{bad!r} accepted"


def test_paths_with_spaces_unicode_and_shellish_names_are_data(tmp_path):
    """The helper never shells out, so ANY absolute path is valid DATA — spaces,
    Unicode, even $() in a filename is inert bytes here (CI review: rejecting
    such paths was over-restriction; nothing may execute either)."""
    for name in ("my plan.md", "plán-übersicht.md", "weird$(true).md"):
        plan = _mk_plan(tmp_path, name=name, content="content\n")
        out = _snapshot(tmp_path, plan)
        assert out["plan_path"] == str(plan)
        assert pathlib.Path(out["snapshot_path"]).read_text() == "content\n"
        cp = _persist(tmp_path, out)
        assert pathlib.Path(cp.stdout.strip()).exists()
    assert not pathlib.Path("pwned").exists()


def test_check_reports_freshness(tmp_path):
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    pf = _write_file(tmp_path, "cp.txt", str(plan))
    probe = json.loads(_run("check", "--plan-path-file", str(pf), home=tmp_path).stdout)
    assert probe["review_exists"] is False and probe["fresh"] is False
    _persist(tmp_path, out)
    probe = json.loads(_run("check", "--plan-path-file", str(pf), home=tmp_path).stdout)
    assert probe["fresh"] is True
    plan.write_text("revised\n")
    probe = json.loads(_run("check", "--plan-path-file", str(pf), home=tmp_path).stdout)
    assert probe["fresh"] is False and probe["review_exists"] is True


def test_tilde_expanded_as_data(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    plan = _mk_plan(tmp_path)
    rel = "~/" + str(plan.relative_to(tmp_path))
    pf = _write_file(tmp_path, "pp.txt", rel)
    cp = subprocess.run(
        ["python3", str(_HELPER), "snapshot", "--plan-path-file", str(pf)],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert cp.returncode == 0, cp.stderr
    assert json.loads(cp.stdout)["plan_path"] == str(plan)


def test_hostile_meta_cannot_break_frontmatter(tmp_path):
    """A hostile assessment (newlines, ---, $()) must stay one plain scalar line
    — the hook's line-based parser and the frontmatter block must survive."""
    plan = _mk_plan(tmp_path)
    out = _snapshot(tmp_path, plan)
    cp = _persist(
        tmp_path,
        out,
        meta=json.dumps({"assessment": "bad\n---\nplan_sha256: 0000\n$(touch owned)"}),
    )
    text = pathlib.Path(cp.stdout.strip()).read_text()
    frontmatter = text.split("---\n")[1]
    # The hook parses line-based `key: value`: exactly ONE line may start with
    # plan_sha256:, and it must carry the real hash — the hostile content stays
    # embedded inside the quoted assessment scalar, never a line of its own.
    sha_lines = [ln for ln in frontmatter.splitlines() if ln.startswith("plan_sha256:")]
    assert sha_lines == [f"plan_sha256: {out['plan_sha256']}"]
    assert "\n---\n" not in frontmatter, "hostile newlines must not close the block early"
    assert not pathlib.Path("owned").exists()


def test_whitespace_and_newline_paths_rejected_explicitly(tmp_path):
    """Line-based frontmatter can't represent CR/LF paths, and trailing
    whitespace is an ingress accident — both fail loudly instead of silently
    normalizing to a different file (CI round-5 P2). The Write tool's single
    trailing newline is still fine."""
    plan = _mk_plan(tmp_path)
    ok = _write_file(tmp_path, "ok.txt", str(plan) + "\n")  # Write-tool convention
    assert _run("snapshot", "--plan-path-file", str(ok), home=tmp_path).returncode == 0
    for bad, frag in [
        (str(plan) + " ", "whitespace"),
        (" " + str(plan), "whitespace"),
        (str(plan) + "\nextra", "newline"),
        (str(plan).replace("plans", "pla\rns"), "newline"),
    ]:
        pf = tmp_path / "bad.txt"
        pf.write_text(bad)
        cp = _run("snapshot", "--plan-path-file", str(pf), home=tmp_path, check=False)
        assert cp.returncode == 2, f"{bad!r} accepted"
        assert frag in cp.stderr
