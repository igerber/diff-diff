"""Runtime tests for the reviewer-eval harness (no codex, no network).

These exercise the real ``CodexReviewer.review()`` return contract and the
experiment-identity keying that resume relies on — the two bugs an earlier local
AI review caught (the ``ReviewOutput`` kwarg mismatch and run-artifact aliasing
across models) live here, so this is where they get a regression test.

``call_codex`` and the git worktree are stubbed.
"""

import functools
import importlib.util
import os
import pathlib
import re
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_EVAL_ROOT = _REPO / "tools" / "reviewer-eval"

pytestmark = pytest.mark.skipif(
    not _EVAL_ROOT.exists(),
    reason="reviewer-eval harness not present (isolated install)",
)

if _EVAL_ROOT.exists() and str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))
# eval_core (the shared engine) lives directly under tools/.
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))


@functools.lru_cache(maxsize=1)
def _run_eval():
    """Load THIS harness's CLI by explicit path under a UNIQUE module name.

    A bare ``import run_eval`` is ambiguous once both eval harnesses are on
    sys.path (plan-review-eval ships its own ``run_eval.py``), and which one
    wins depends on test-collection insert order — CI's alphabetical
    single-process run (the Pure Python Fallback leg) loaded the WRONG
    harness. Mirrors test_plan_review_eval.py's loader."""
    spec = importlib.util.spec_from_file_location(
        "reviewer_eval_run_eval", _EVAL_ROOT / "run_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pinned_cli_version():
    """The cli_version pin from config/configs.json (single source of truth).

    Stub reviewers must report this exact string or run_matrix's CLI-equality
    assert fires before the behavior under test; reading it from the live config
    means a CLI bump never breaks these stubs again.
    """
    import json

    raw = json.loads((_EVAL_ROOT / "config" / "configs.json").read_text())
    return raw["arms"][0]["cli_version"]


_STUB_PROVENANCE = {
    "repo_head_sha": "feedface",
    "repo_tree_sha": "cafebabe",
    "repo_dirty": False,
    "repo_branch": "test-branch",
    "repo_common_dir": "/tmp/x/.git",
    "repo_worktree_paths": [],
    "repo_ref_names": ["test-branch", "origin/main"],
}


class _FakeIsolatedHome:
    """Test double for ``isolated_codex_home()``: no credential copy, but the
    SAME observable contract (sets ``ACTIVE_ISOLATED_HOME`` + ``CODEX_HOME`` to a
    tmp dir while active, restores both on exit, counts entries)."""

    def __init__(self, tmp_path):
        self.home = str(tmp_path / "fake-codex-home")
        self.entries = 0
        self._prev = None
        self._prev_active = None

    def __call__(self):
        return self

    def __enter__(self):
        import os as _os

        from adapters import codex_reviewer as cr

        _os.makedirs(self.home, exist_ok=True)
        self.entries += 1
        self._prev = _os.environ.get("CODEX_HOME")
        self._prev_active = cr.ACTIVE_ISOLATED_HOME
        _os.environ["CODEX_HOME"] = self.home
        cr.ACTIVE_ISOLATED_HOME = self.home
        return self.home

    def __exit__(self, *exc):
        import os as _os

        from adapters import codex_reviewer as cr

        if self._prev is None:
            _os.environ.pop("CODEX_HOME", None)
        else:
            _os.environ["CODEX_HOME"] = self._prev
        cr.ACTIVE_ISOLATED_HOME = self._prev_active
        return False


def _stub_run_git(monkeypatch, tmp_path):
    """Keep cmd_run/cmd_smoke tests off the dev/CI checkout's git state and off
    ``~/.codex/auth.json``: stub the manifest provenance (fixed seven keys), the
    hidden-input guard (no-op) and the isolated codex home (``_FakeIsolatedHome``).
    Returns the fake home so a test can assert it was entered."""
    run_eval = _run_eval()
    fake = _FakeIsolatedHome(tmp_path)
    monkeypatch.setattr(run_eval, "_manifest_provenance", lambda repo_root: dict(_STUB_PROVENANCE))
    monkeypatch.setattr(run_eval, "_assert_no_hidden_inputs", lambda repo_root: None)
    monkeypatch.setattr(run_eval, "isolated_codex_home", fake)
    # The frozen-ref helpers would touch the dev checkout's refs: stub them with
    # an in-memory ref keyed by contrast.
    refs: dict = {}
    monkeypatch.setattr(
        run_eval, "_read_frozen_ref", lambda root, contrast: refs.get(tuple(contrast))
    )
    monkeypatch.setattr(
        run_eval,
        "_anchor_frozen_ref",
        lambda root, contrast, head: refs.setdefault(tuple(contrast), head),
    )
    fake.refs = refs
    return fake


def _make_reviewer(monkeypatch, review_md="## Overall Assessment\n✅ Looks good\n"):
    """A CodexReviewer with call_codex + worktree stubbed (no codex, no git)."""
    from adapters import codex_reviewer as cr
    from adapters import worktree

    r = cr.CodexReviewer(
        repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="BASE PROMPT BODY"
    )

    # Stub the codex call and the worktree materialize/cleanup so review() is
    # exercised end-to-end without spawning codex or touching git.
    monkeypatch.setattr(
        r._mod,
        "call_codex",
        lambda prompt, model, repo_root, effort="xhigh", timeout_s=None: (
            review_md,
            {"backend": "codex"},
        ),
        raising=True,
    )
    monkeypatch.setattr(
        r,
        "build_prompt_for_case",
        lambda case, worktree_key=None: ("PROMPT", "/tmp/reviewer-eval-test/wt", "deadbeef"),
        raising=True,
    )
    monkeypatch.setattr(worktree, "cleanup", lambda *a, **k: None, raising=True)
    # review() samples git provenance through the module; keep the helper's
    # "no codex, no git" contract by stubbing it as one unit.
    monkeypatch.setattr(
        worktree, "git_provenance", lambda root: dict(_STUB_PROVENANCE), raising=True
    )
    return r


def _case():
    from eval_core.models import STRATUM_HISTORICAL, Case

    return Case(id="c1", stratum=STRATUM_HISTORICAL)


# --------------------------------------------------------------------------- #
# Bug 1 (was P1): CodexReviewer.review() must return a valid ReviewOutput.
# --------------------------------------------------------------------------- #


def test_codex_reviewer_review_returns_ok(monkeypatch):
    from eval_core.models import Config, ReviewOutput

    r = _make_reviewer(monkeypatch)
    out = r.review(_case(), Config(id="B", model="gpt-5.5"), 0)
    assert isinstance(out, ReviewOutput)
    assert out.review_markdown.startswith("## Overall Assessment")
    assert out.cli_version  # recorded
    assert out.latency_s >= 0.0


def test_run_matrix_produces_ok_runresult(monkeypatch):
    """A successful review must yield an ok RunResult, not an INFRA_ERROR."""
    from eval_core.models import Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-ok")
    # fresh store each run
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()
    results = run_matrix(
        [_case()],
        [Config(id="B", model="gpt-5.5")],
        r,
        store,
        k=1,
        max_parallel=1,
    )
    assert len(results) == 1
    assert results[0].ok, f"expected ok RunResult, got infra_error={results[0].infra_error}"
    assert results[0].review_markdown


# --------------------------------------------------------------------------- #
# Bug 2 (was P0): experiment identity must not alias across different models
# sharing the same config id.
# --------------------------------------------------------------------------- #


def test_experiment_tag_differs_by_model(monkeypatch):
    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    tag_a = r.experiment_tag(Config(id="B", model="gpt-5.4"))
    tag_b = r.experiment_tag(Config(id="B", model="gpt-5.5"))
    assert tag_a != tag_b, "same config id + different model must yield distinct tags"


def test_run_key_no_alias_across_models(monkeypatch):
    from eval_core.models import Config
    from eval_core.store import run_key

    r = _make_reviewer(monkeypatch)
    k4 = run_key("c1", "B", 0, r.experiment_tag(Config(id="B", model="gpt-5.4")))
    k5 = run_key("c1", "B", 0, r.experiment_tag(Config(id="B", model="gpt-5.5")))
    assert k4 != k5, "run files for different models must not collide under one config id"


def test_runresult_carries_run_id_and_prompt_sha(monkeypatch):
    """The run artifact must record its own identity so compare can key on it."""
    from eval_core.models import Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-id")
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()
    results = run_matrix(
        [_case()],
        [Config(id="B", model="gpt-5.5")],
        r,
        store,
        k=1,
        max_parallel=1,
    )
    rr = results[0]
    assert rr.run_id, "RunResult must carry a stable run_id"
    assert rr.prompt_sha, "RunResult must record the prompt_sha it reviewed"


def test_resume_reruns_when_model_changes(monkeypatch):
    """Changing the model under the same config id must NOT resume stale runs."""
    from eval_core.models import Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch, review_md="## A\n✅ first\n")
    store = RunStore("/tmp/reviewer-eval-test/runs-resume")
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()

    run_matrix(
        [_case()],
        [Config(id="B", model="gpt-5.4")],
        r,
        store,
        k=1,
        max_parallel=1,
    )
    # Now rerun the SAME config id but a DIFFERENT model. Must not reuse the
    # gpt-5.4 artifact; a new run file must appear.
    r2 = _make_reviewer(monkeypatch, review_md="## B\n✅ second\n")
    run_matrix(
        [_case()],
        [Config(id="B", model="gpt-5.5")],
        r2,
        store,
        k=1,
        max_parallel=1,
    )
    files = sorted(pathlib.Path(store.root).glob("*.json"))
    assert len(files) == 2, f"expected 2 distinct run files (one per model), got {len(files)}"


def _ns(**kw):
    import argparse

    return argparse.Namespace(**kw)


# --------------------------------------------------------------------------- #
# Case-aware run identity (P1 #1): editing a case must invalidate its cache.
# --------------------------------------------------------------------------- #


def test_case_tag_changes_with_case_content():
    from adapters.codex_reviewer import CodexReviewer
    from eval_core.models import STRATUM_HISTORICAL, Case

    r = CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="X")
    fx = {"kind": "git_range", "base_sha": "aaa"}
    base = Case(id="c", stratum=STRATUM_HISTORICAL, fixture=dict(fx))
    same = Case(id="c", stratum=STRATUM_HISTORICAL, fixture=dict(fx))
    edited = Case(
        id="c", stratum=STRATUM_HISTORICAL, fixture={"kind": "git_range", "base_sha": "bbb"}
    )
    assert r.case_tag(base) == r.case_tag(same)  # stable; no patch read (git_range)
    assert r.case_tag(base) != r.case_tag(edited)  # base_sha edit -> new tag
    # the machine-local _case_dir must NOT affect the tag
    with_dir = Case(id="c", stratum=STRATUM_HISTORICAL, fixture={**fx, "_case_dir": "/wherever"})
    assert r.case_tag(base) == r.case_tag(with_dir)


def test_case_tag_reads_patch_bytes_and_fails_loud(tmp_path):
    from adapters.codex_reviewer import CodexReviewer
    from eval_core.models import STRATUM_SYNTHETIC, Case

    r = CodexReviewer(repo_root=str(_REPO), runs_root=str(tmp_path / "runs"), prompt_text="X")
    patch = tmp_path / "inject.diff"
    patch.write_text("AAA")
    fx = {
        "kind": "stored_patch",
        "base_sha": "x",
        "patch": "inject.diff",
        "_case_dir": str(tmp_path),
    }
    t1 = r.case_tag(Case(id="c", stratum=STRATUM_SYNTHETIC, fixture=dict(fx)))
    patch.write_text("BBB")  # editing the patch bytes must change the tag
    assert r.case_tag(Case(id="c", stratum=STRATUM_SYNTHETIC, fixture=dict(fx))) != t1
    patch.unlink()  # a declared-but-missing patch must fail loud, not hash-around it
    with pytest.raises(FileNotFoundError):
        r.case_tag(Case(id="c", stratum=STRATUM_SYNTHETIC, fixture=dict(fx)))


def test_resume_reruns_when_case_changes(monkeypatch):
    from eval_core.models import STRATUM_HISTORICAL, Case, Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-case")
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()
    cfg = [Config(id="A", model="gpt-5.4")]
    run_matrix(
        [Case(id="x", stratum=STRATUM_HISTORICAL, fixture={"base_sha": "aaa"})],
        cfg,
        r,
        store,
        k=1,
        max_parallel=1,
    )
    # Same case id, edited content -> must NOT resume the stale run.
    run_matrix(
        [Case(id="x", stratum=STRATUM_HISTORICAL, fixture={"base_sha": "bbb"})],
        cfg,
        r,
        store,
        k=1,
        max_parallel=1,
    )
    files = sorted(pathlib.Path(store.root).glob("*.json"))
    assert len(files) == 2, f"editing the case must rerun, not resume; got {len(files)}"


# --------------------------------------------------------------------------- #
# compare (P1 #2): the per-run manifest isolates one experiment.
# --------------------------------------------------------------------------- #


def test_compare_honors_manifest(tmp_path, monkeypatch):
    run_eval = _run_eval()
    from eval_core.models import RunResult
    from eval_core.store import RunStore, write_json

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    store = RunStore(str(tmp_path / "runs" / "full"))
    cid = "s1-coef-dict-collision"  # a real corpus case so build_bundle renders it
    store.save(
        "keep",
        RunResult(
            case_id=cid,
            config_id="A",
            repeat_idx=0,
            review_markdown="KEEP THIS REVIEW",
            model="gpt-5.5",
            run_id="keep",
        ),
    )
    store.save(
        "drop",
        RunResult(
            case_id=cid,
            config_id="A",
            repeat_idx=0,
            review_markdown="STALE DROP REVIEW",
            model="gpt-5.4",
            run_id="drop",
        ),
    )
    write_json(
        str(tmp_path / "runs" / "full-manifest.json"), {"run_ids": ["keep"], "configs": ["A"]}
    )
    assert run_eval.cmd_compare(_ns(subdir="full")) == 0
    out = (tmp_path / "runs" / "full" / "comparison.md").read_text()
    assert "KEEP THIS REVIEW" in out
    assert "STALE DROP REVIEW" not in out, "manifest must exclude the stale experiment's run"


def test_compare_without_manifest_fails_closed_unless_allow_mixed(tmp_path, monkeypatch, capsys):
    run_eval = _run_eval()
    from eval_core.models import RunResult
    from eval_core.store import RunStore

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    store = RunStore(str(tmp_path / "runs" / "full"))
    store.save(
        "only",
        RunResult(
            case_id="s1-coef-dict-collision",
            config_id="A",
            repeat_idx=0,
            review_markdown="SOLO REVIEW",
            model="gpt-5.4",
            run_id="only",
        ),
    )
    # No manifest -> refuse by default (one run = one experiment).
    assert run_eval.cmd_compare(_ns(subdir="full", allow_mixed=False)) != 0
    assert "no manifest" in capsys.readouterr().err.lower()
    # --allow-mixed is the explicit override: compare ALL runs, with a warning.
    assert run_eval.cmd_compare(_ns(subdir="full", allow_mixed=True)) == 0
    assert "allow-mixed" in capsys.readouterr().err.lower()
    assert "SOLO REVIEW" in (tmp_path / "runs" / "full" / "comparison.md").read_text()


def test_compare_fails_closed_on_rubric_drift(tmp_path, monkeypatch):
    """compare points graders at the live pr_review.md, so it must refuse if that
    rubric changed since the run (stored base_prompt_sha != live)."""
    run_eval = _run_eval()
    from adapters import ci_prompt
    from eval_core.models import RunResult
    from eval_core.store import RunStore, write_json

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    store = RunStore(str(tmp_path / "runs" / "full"))
    store.save(
        "r1",
        RunResult(
            case_id="c", config_id="A", repeat_idx=0, review_markdown="x", model="m", run_id="r1"
        ),
    )
    write_json(
        str(tmp_path / "runs" / "full-manifest.json"),
        {"run_ids": ["r1"], "configs": ["A"], "base_prompt_sha": "deadbeefdeadbeef"},
    )
    # Live rubric hashes to something else -> drift -> refuse.
    monkeypatch.setattr(
        ci_prompt, "read_current_prompt", lambda *a, **k: "A DIFFERENT RUBRIC", raising=True
    )
    assert run_eval.cmd_compare(_ns(subdir="full", allow_mixed=False)) != 0


def test_compare_renders_from_run_snapshot_not_corpus(tmp_path, monkeypatch):
    run_eval = _run_eval()
    from eval_core.models import RunResult
    from eval_core.store import RunStore, write_json

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    store = RunStore(str(tmp_path / "runs" / "full"))
    # Ground truth whose marker exists ONLY in the artifact's snapshot — and a
    # case_id that is NOT in the corpus — so a corpus reload could not produce it.
    snap = {
        "title": "Snapshot Case",
        "stratum": "s2_historical",
        "ground_truth": [
            {
                "id": "snap:b1",
                "file": "z.py",
                "line_window": [1, 2],
                "bug_class": "x",
                "expected_severity": "P1",
                "rationale": "SNAPSHOT-ONLY-MARKER",
            }
        ],
        "expect_no_blockers": False,
        "allow_severities": ["P2", "P3"],
        "known_fp_topics": [],
    }
    store.save(
        "k",
        RunResult(
            case_id="not-a-corpus-case",
            config_id="A",
            repeat_idx=0,
            review_markdown="rev",
            model="gpt-5.4",
            run_id="k",
            case_snapshot=snap,
        ),
    )
    write_json(str(tmp_path / "runs" / "full-manifest.json"), {"run_ids": ["k"], "configs": ["A"]})
    assert run_eval.cmd_compare(_ns(subdir="full")) == 0
    out = (tmp_path / "runs" / "full" / "comparison.md").read_text()
    # Ground truth comes from the run's snapshot — compare never reads the corpus.
    assert "SNAPSHOT-ONLY-MARKER" in out
    assert "snap:b1" in out
    assert "Snapshot Case" in out


def test_run_rejects_unknown_configs(tmp_path, monkeypatch):
    _stub_run_git(monkeypatch, tmp_path)
    run_eval = _run_eval()

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    # A typo'd config id must fail closed BEFORE any codex call (no reviewer built),
    # rather than silently running 0/0 and writing an empty manifest.
    rc = run_eval.cmd_run(_ns(configs="Z", strata=None, subdir="full", k=1, max_parallel=1))
    assert rc == 1
    assert not (tmp_path / "runs" / "full-manifest.json").exists()


def test_case_tag_changes_with_scoring_metadata():
    """A metadata-only case edit (ground truth, NOT the fixture) must bust the cache.

    Regression for PR #510 P1: case_tag previously hashed only the fixture+patch, so
    editing ground_truth/severity/negative-control flags left the run key unchanged
    and `compare` graded against a stale snapshot.
    """
    from adapters.codex_reviewer import CodexReviewer
    from eval_core.models import STRATUM_HISTORICAL, Case, GroundTruthBug

    r = CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="X")
    fx = {"kind": "git_range", "base_sha": "aaa"}  # identical fixture across all three
    base = Case(
        id="c",
        stratum=STRATUM_HISTORICAL,
        fixture=dict(fx),
        ground_truth=[
            GroundTruthBug(
                id="c:b1", file="f.py", line_window=(1, 5), bug_class="x", expected_severity="P1"
            )
        ],
    )
    sev = Case(
        id="c",
        stratum=STRATUM_HISTORICAL,
        fixture=dict(fx),
        ground_truth=[
            GroundTruthBug(
                id="c:b1", file="f.py", line_window=(1, 5), bug_class="x", expected_severity="P0"
            )
        ],
    )
    neg = Case(
        id="c",
        stratum=STRATUM_HISTORICAL,
        fixture=dict(fx),
        ground_truth=[],
        expect_no_blockers=True,
    )
    assert r.case_tag(base) != r.case_tag(sev), "editing expected_severity must bust the cache"
    assert r.case_tag(base) != r.case_tag(neg), "editing expect_no_blockers must bust the cache"


def test_run_and_smoke_fail_closed_on_empty_corpus(tmp_path, monkeypatch):
    """run/smoke must NOT report success (or write a manifest) on zero selected cases."""
    _stub_run_git(monkeypatch, tmp_path)
    run_eval = _run_eval()

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    # A stratum that matches no corpus directory -> zero cases (no codex reached).
    rc_run = run_eval.cmd_run(
        _ns(configs="A,B", strata=["no_such_stratum"], subdir="full", k=1, max_parallel=1)
    )
    assert rc_run == 1
    assert not (tmp_path / "runs" / "full-manifest.json").exists(), "no manifest for a no-op run"
    rc_smoke = run_eval.cmd_smoke(
        _ns(configs="A", strata=["no_such_stratum"], k=1, limit=0, max_parallel=1)
    )
    assert rc_smoke == 1


def test_build_prompt_cleans_worktree_on_build_failure(monkeypatch):
    """A prompt-build failure after materialize (e.g. the notebook guard) must not
    leak a detached worktree."""
    from adapters import ci_prompt, worktree
    from adapters import codex_reviewer as cr
    from eval_core.models import STRATUM_SYNTHETIC, Case

    r = cr.CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="X")

    class _Mat:
        worktree_dir = "/tmp/reviewer-eval-test/wt-leaktest"
        base_sha = "b"
        head_sha = "h"

    def _raise(**_kw):
        raise NotImplementedError("notebook case unsupported")

    cleaned = []
    monkeypatch.setattr(worktree, "materialize", lambda *a, **k: _Mat(), raising=True)
    monkeypatch.setattr(ci_prompt, "build_ci_prompt", _raise, raising=True)
    monkeypatch.setattr(
        worktree, "cleanup", lambda wt, root, *a, **k: cleaned.append(wt), raising=True
    )

    case = Case(id="c", stratum=STRATUM_SYNTHETIC, fixture={"_case_dir": "/x"})
    with pytest.raises(NotImplementedError):
        r.build_prompt_for_case(case, worktree_key="c.A.r0")
    assert cleaned == ["/tmp/reviewer-eval-test/wt-leaktest"], "worktree must be cleaned on failure"


# --------------------------------------------------------------------------- #
# Verify-on-resume (PR #510 round 2): a cached run is reused ONLY if the model
# would see byte-identical input now — covers harness-code edits the run key
# can't fingerprint.
# --------------------------------------------------------------------------- #


def test_resume_reruns_when_built_prompt_changes(monkeypatch):
    from eval_core.models import STRATUM_HISTORICAL, Case, Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    store = RunStore("/tmp/reviewer-eval-test/runs-promptchange")
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()
    cfg = [Config(id="A", model="gpt-5.4")]
    case = Case(id="x", stratum=STRATUM_HISTORICAL)

    # Run 1: the stub builds "PROMPT" (default) and caches review "## V1".
    r1 = _make_reviewer(monkeypatch, review_md="## V1\n")
    run_matrix([case], cfg, r1, store, k=1, max_parallel=1)

    # Same case/config (=> same run key), but the BUILT PROMPT now differs.
    r2 = _make_reviewer(monkeypatch, review_md="## V2\n")
    # Pin the backend-contract term equal to r1's so the run KEY is identical:
    # this test must exercise the prompt re-verify path, not a key miss from the
    # backend-contract term (which _make_reviewer perturbs by stubbing call_codex).
    r2.backend_contract_sha = r1.backend_contract_sha
    monkeypatch.setattr(
        r2,
        "build_prompt_for_case",
        lambda case, worktree_key=None: (
            "PROMPT-CHANGED",
            "/tmp/reviewer-eval-test/wt",
            "deadbeef",
        ),
        raising=True,
    )
    results = run_matrix([case], cfg, r2, store, k=1, max_parallel=1)
    assert results[0].review_markdown.startswith(
        "## V2"
    ), "stale cached review was reused despite the built prompt changing"


# --------------------------------------------------------------------------- #
# Backend-contract identity (PR #510 round 3 P1): experiment identity must also
# cover the codex-invocation wrapper (_build_codex_cmd / call_codex), not just
# the model/prompt/declared-config — else a wrapper edit silently resumes a stale
# artifact run under the OLD wrapper.
# --------------------------------------------------------------------------- #


def _wrapper_v2(model, repo_root, output_path):
    # A _build_codex_cmd whose SOURCE differs from openai_review's (flips the
    # sandbox + drops the effort/-o flags) -> a distinct backend_contract_sha.
    return ["codex", "exec", "--model", model, "--sandbox", "workspace-write"]


def test_experiment_tag_differs_when_backend_wrapper_changes(monkeypatch):
    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    cfg = Config(id="B", model="gpt-5.4")
    tag_before = r.experiment_tag(cfg)

    # Same model/effort/sandbox/prompt/cli, but HOW codex is invoked changed.
    monkeypatch.setattr(r._mod, "_build_codex_cmd", _wrapper_v2, raising=True)
    r.backend_contract_sha = r._backend_contract_sha(r._mod)
    assert (
        r.experiment_tag(cfg) != tag_before
    ), "a codex-wrapper change must yield a distinct experiment tag"


def test_resume_reruns_when_backend_wrapper_changes(monkeypatch):
    """A cached run must NOT be resumed once the codex-invocation wrapper changed,
    even when the case/config and the built prompt are byte-identical.
    """
    from eval_core.models import STRATUM_HISTORICAL, Case, Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    store = RunStore("/tmp/reviewer-eval-test/runs-backendchange")
    for f in pathlib.Path(store.root).glob("*.json"):
        f.unlink()
    cfg = [Config(id="A", model="gpt-5.4")]
    case = Case(id="x", stratum=STRATUM_HISTORICAL)

    # Run 1 caches "## V1" under a fixed backend-contract baseline.
    r1 = _make_reviewer(monkeypatch, review_md="## V1\n")
    r1.backend_contract_sha = "contract-v1"
    run_matrix([case], cfg, r1, store, k=1, max_parallel=1)

    # Control: an arm with the SAME contract (and same prompt) resumes V1 — proves
    # the rerun below is caused by the contract change, not by something else.
    r_same = _make_reviewer(monkeypatch, review_md="## SHOULD-NOT-APPEAR\n")
    r_same.backend_contract_sha = "contract-v1"
    res_same = run_matrix([case], cfg, r_same, store, k=1, max_parallel=1)
    assert res_same[0].review_markdown.startswith(
        "## V1"
    ), "identical backend contract + prompt must resume the cached run"

    # Treatment: the wrapper source changed -> distinct identity -> rerun.
    r2 = _make_reviewer(monkeypatch, review_md="## V2\n")
    monkeypatch.setattr(r2._mod, "_build_codex_cmd", _wrapper_v2, raising=True)
    r2.backend_contract_sha = r2._backend_contract_sha(r2._mod)
    assert r2.backend_contract_sha != "contract-v1", "the wrapper edit must move identity"
    res2 = run_matrix([case], cfg, r2, store, k=1, max_parallel=1)
    assert res2[0].review_markdown.startswith(
        "## V2"
    ), "stale cached review was reused despite the codex invocation wrapper changing"


class _InfraBoom:
    # cli_version must match config/configs.json's pin so the A/B CLI-equality
    # assert doesn't fire before we reach the infra path.
    def cli_version(self):
        return _pinned_cli_version()

    def experiment_tag(self, config):
        return "tag"

    def case_tag(self, case):
        return "ctag"

    def prompt_sha_for(self, case):
        return "psha"

    def review(self, case, config, repeat_idx):
        raise RuntimeError("simulated codex failure")


def test_run_fails_closed_on_infra_error(tmp_path, monkeypatch):
    """cmd_run must exit non-zero and write a FAILURE-MARKER manifest (never a valid
    run_ids manifest) when any run hits INFRA_ERROR, so `compare` can't present a
    partial run as a valid A/B."""
    _stub_run_git(monkeypatch, tmp_path)
    import json as _json

    run_eval = _run_eval()

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)
    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _InfraBoom(), raising=True)
    rc = run_eval.cmd_run(
        _ns(configs="A", strata=["s1_synthetic"], subdir="full", k=1, max_parallel=1)
    )
    assert rc != 0, "run must fail closed on INFRA_ERROR"
    manifest_path = tmp_path / "runs" / "full-manifest.json"
    assert manifest_path.exists(), "infra-failed run must write a failure-marker manifest"
    m = _json.loads(manifest_path.read_text())
    assert m.get("failed") is True, "manifest must be marked failed"
    assert not m.get("run_ids"), "a failed run must not record run_ids"


# --------------------------------------------------------------------------- #
# smoke --limit contract (PR #510 round 3 P2): bare `smoke` must run exactly ONE
# case (matching the README's "1 case, first codex call"), with `--limit 0` as
# the explicit "run the whole selected corpus" escape hatch.
# --------------------------------------------------------------------------- #


def test_smoke_cli_default_limits_to_one_case(tmp_path, monkeypatch):
    _stub_run_git(monkeypatch, tmp_path)
    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))

    # Three fake cases so "1 case" vs "all" is observable without loading the
    # real corpus or spawning codex.
    fake_cases = [Case(id=f"c{i}", stratum=STRATUM_SYNTHETIC) for i in range(3)]
    monkeypatch.setattr(
        run_eval.CorpusLoader, "load_cases", lambda self, strata: list(fake_cases), raising=True
    )
    # Not testing validation here: skip the (git-materializing) verify preflight.
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    class _Rev:
        def cli_version(self):
            return _pinned_cli_version()

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Rev(), raising=True)

    captured = {}

    def _capture_run_matrix(cases, configs, *a, **k):
        captured["n"] = len(cases)
        return []

    # cmd_smoke does `from eval_core.runner import run_matrix` at call time, so patch
    # the source name, not run_eval's namespace.
    monkeypatch.setattr("eval_core.runner.run_matrix", _capture_run_matrix, raising=True)

    # Bare `smoke --configs A` (argparse default --limit) -> exactly one case.
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "smoke", "--configs", "A"])
    assert run_eval.main() == 0
    assert captured["n"] == 1, "bare `smoke` must run exactly one case (README contract)"

    # `--limit 0` is the explicit "run all selected" escape hatch.
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "smoke", "--configs", "A", "--limit", "0"])
    assert run_eval.main() == 0
    assert captured["n"] == 3, "`smoke --limit 0` must run all selected cases"


# --------------------------------------------------------------------------- #
# Non-model confounds + manifest fail-closed (local codex round, PR #510): the
# model must be the ONLY variable across arms, and a failed rerun must not leave
# a prior run's manifest live for `compare` to render.
# --------------------------------------------------------------------------- #


def test_run_matrix_aborts_on_confound_mismatch(monkeypatch):
    """Arms that drift in any held-constant confound (effort/sandbox/action_version)
    must abort up front — the model is the only intended variable."""
    from eval_core.models import Config
    from eval_core.runner import ConfoundMismatch, run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-confound")
    for field, a, b in [
        ("sandbox", "read-only", "workspace-write"),
        ("effort", "xhigh", "high"),
        ("action_version", "v1", "v2"),
    ]:
        cfgs = [
            Config(id="A", model="gpt-5.4", **{field: a}),
            Config(id="B", model="gpt-5.5", **{field: b}),
        ]
        with pytest.raises(ConfoundMismatch):
            run_matrix([_case()], cfgs, r, store, k=1, max_parallel=1)


def test_review_rejects_non_readonly_sandbox(monkeypatch):
    """recorded==executed: _build_codex_cmd hardcodes read-only, so a config asking
    for a different sandbox must fail closed (mirrors the effort guard)."""
    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    with pytest.raises(NotImplementedError):
        r.review(_case(), Config(id="B", model="gpt-5.5", sandbox="workspace-write"), 0)


def test_failed_rerun_invalidates_stale_manifest(tmp_path, monkeypatch):
    """A successful run writes a manifest; a later FAILED rerun into the same subdir
    must invalidate it (mark failed) so `compare` refuses instead of rendering the
    stale experiment."""
    _stub_run_git(monkeypatch, tmp_path)
    import json as _json

    run_eval = _run_eval()
    from eval_core.models import ReviewOutput

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)
    manifest_path = tmp_path / "runs" / "full-manifest.json"

    class _Ok:
        def cli_version(self):
            return _pinned_cli_version()

        def experiment_tag(self, config):
            return "tag"

        def case_tag(self, case):
            return "ctag-v1"

        def prompt_sha_for(self, case):
            return "psha"

        def review(self, case, config, repeat_idx):
            return ReviewOutput(
                review_markdown="## ok",
                cli_version=_pinned_cli_version(),
                latency_s=0.0,
                usage={"prompt_sha": "psha"},
            )

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Ok(), raising=True)
    rc_ok = run_eval.cmd_run(
        _ns(configs="A", strata=["s1_synthetic"], subdir="full", k=1, max_parallel=1)
    )
    assert rc_ok == 0
    assert _json.loads(manifest_path.read_text()).get("run_ids"), "success records run_ids"

    # The case was edited (new case_tag) and now errors -> a failed rerun. The prior
    # valid manifest must be invalidated, not left live for compare.
    class _BoomEdited(_InfraBoom):
        def case_tag(self, case):
            return "ctag-v2"  # edited case -> new key -> not resumed from cache

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _BoomEdited(), raising=True)
    rc_fail = run_eval.cmd_run(
        _ns(configs="A", strata=["s1_synthetic"], subdir="full", k=1, max_parallel=1)
    )
    assert rc_fail != 0
    m = _json.loads(manifest_path.read_text())
    assert m.get("failed") is True and not m.get("run_ids"), "stale manifest must be invalidated"

    # compare must refuse the failed/incomplete experiment, not fall back to compare-all.
    assert run_eval.cmd_compare(_ns(subdir="full")) != 0, "compare must refuse a failed run"


def test_run_aborts_on_invalid_case_before_any_codex_call(tmp_path, monkeypatch):
    """smoke/run must fail closed on a CorpusLoader.verify() failure BEFORE any
    Codex call, so a stale/malformed case is never reviewed/graded against stale
    ground truth."""
    _stub_run_git(monkeypatch, tmp_path)
    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="bad", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "verify",
        lambda self, case: "diff does not touch expected file(s)",
        raising=True,
    )
    called = {"run": False}

    def _no_run(*a, **k):
        called["run"] = True
        return []

    monkeypatch.setattr("eval_core.runner.run_matrix", _no_run, raising=True)
    monkeypatch.setattr(
        run_eval,
        "_build_reviewer",
        lambda repo_root: (_ for _ in ()).throw(AssertionError),
        raising=True,
    )

    rc = run_eval.cmd_run(
        _ns(configs="A", strata=["s1_synthetic"], subdir="full", k=1, max_parallel=1)
    )
    assert rc != 0, "run must fail closed when a case fails validation"
    assert not called["run"], "no review may run when a case fails validation"
    # The up-front failure marker must remain (a preflight abort leaves the subdir in
    # a failed state that compare refuses — never a prior run's live manifest).
    import json as _json

    m = _json.loads((tmp_path / "runs" / "full-manifest.json").read_text())
    assert m.get("failed") is True and not m.get(
        "run_ids"
    ), "preflight abort leaves a failed marker"


def test_run_matrix_fails_closed_on_experiment_tag_error(monkeypatch):
    """If experiment_tag() can't be computed, run_matrix must abort — never fall back
    to an empty tag that could resume a stale experiment under an unchanged prompt."""
    from eval_core.models import Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    monkeypatch.setattr(
        r,
        "experiment_tag",
        lambda config: (_ for _ in ()).throw(RuntimeError("tag boom")),
        raising=True,
    )
    store = RunStore("/tmp/reviewer-eval-test/runs-tagfail")
    with pytest.raises(RuntimeError):
        run_matrix([_case()], [Config(id="A", model="gpt-5.4")], r, store, k=1, max_parallel=1)


def test_compare_fails_closed_on_missing_artifact(tmp_path, monkeypatch):
    """compare must refuse when a manifest-listed run_id has no loadable artifact,
    rather than silently emitting a partial bundle from the surviving subset."""
    run_eval = _run_eval()
    from eval_core.models import RunResult
    from eval_core.store import RunStore, write_json

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    store = RunStore(str(tmp_path / "runs" / "full"))
    store.save(
        "present",
        RunResult(
            case_id="c",
            config_id="A",
            repeat_idx=0,
            review_markdown="r",
            model="m",
            run_id="present",
        ),
    )
    # Manifest promises two runs, but only "present" has an artifact on disk.
    write_json(
        str(tmp_path / "runs" / "full-manifest.json"),
        {"run_ids": ["present", "absent"], "configs": ["A", "B"]},
    )
    assert run_eval.cmd_compare(_ns(subdir="full")) != 0, "compare must refuse a missing artifact"


def test_stored_patch_path_must_be_contained(tmp_path):
    """A stored_patch's patch path must resolve inside its case directory; absolute or
    traversal paths are rejected before any worktree work."""
    from adapters import worktree

    case_dir = str(tmp_path)
    (tmp_path / "inject.diff").write_text("x")
    assert worktree._resolve_patch_path("c", case_dir, "inject.diff").endswith("inject.diff")
    with pytest.raises(worktree.MaterializeError):
        worktree._resolve_patch_path("c", case_dir, "/etc/passwd")
    with pytest.raises(worktree.MaterializeError):
        worktree._resolve_patch_path("c", case_dir, "../../../../etc/passwd")
    with pytest.raises(worktree.MaterializeError):
        worktree._resolve_patch_path("c", case_dir, "")


def test_build_prompt_threads_rerun_state(monkeypatch):
    """A rerun case (fixture.rerun.previous_review) must be built as a CI re-review:
    CodexReviewer threads is_rerun/prev_review into ci_prompt.build_ci_prompt."""
    from adapters import ci_prompt, worktree
    from adapters import codex_reviewer as cr
    from eval_core.models import STRATUM_HISTORICAL, Case

    r = cr.CodexReviewer(
        repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="BASE PROMPT"
    )

    class _Mat:
        worktree_dir = "/tmp/reviewer-eval-test/wt"
        base_sha = "base"
        head_sha = "head"

    monkeypatch.setattr(worktree, "materialize", lambda *a, **k: _Mat(), raising=True)
    monkeypatch.setattr(worktree, "cleanup", lambda *a, **k: None, raising=True)

    captured = {}
    monkeypatch.setattr(
        ci_prompt, "build_ci_prompt", lambda **kw: captured.update(kw) or "PROMPT", raising=True
    )

    rr_case = Case(
        id="rr",
        stratum=STRATUM_HISTORICAL,
        fixture={"_case_dir": "", "rerun": {"previous_review": "## prior P1: foo"}},
    )
    r.build_prompt_for_case(rr_case, worktree_key="rr")
    assert captured.get("is_rerun") is True, "rerun case must build with is_rerun=True"
    assert "prior P1: foo" in captured.get("prev_review", ""), "prior review must be threaded"

    captured.clear()
    fresh = Case(id="nr", stratum=STRATUM_HISTORICAL, fixture={"_case_dir": ""})
    r.build_prompt_for_case(fresh, worktree_key="nr")
    assert captured.get("is_rerun") is False, "a non-rerun case must not set is_rerun"


def test_compare_bundle_header_reflects_configs():
    """The grading table is sized to the configs actually present — a single-arm run
    must not be graded against a hardcoded A/B table."""
    from eval_core.compare import build_bundle
    from eval_core.models import RunResult

    def _rr(cfg):
        return RunResult(
            case_id="c",
            config_id=cfg,
            repeat_idx=0,
            review_markdown="## r",
            model="m",
            case_snapshot={"stratum": "s1_synthetic", "title": "t"},
        )

    single = build_bundle([_rr("A")])
    assert "A caught?" in single and "B caught?" not in single, "single-arm: no B column"

    ab = build_bundle([_rr("A"), _rr("B")])
    assert "A caught?" in ab and "B caught?" in ab, "A/B run: both columns present"


def test_run_early_abort_writes_failure_marker(tmp_path, monkeypatch):
    """An early abort (e.g. ConfoundMismatch) after the manifest is removed must
    leave a {failed:true} marker — NOT a missing manifest, which `compare` would
    treat as 'no manifest -> compare ALL' over the prior run's stale artifacts."""
    _stub_run_git(monkeypatch, tmp_path)
    import json as _json

    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case
    from eval_core.runner import ConfoundMismatch

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    class _Rev:
        def cli_version(self):
            return _pinned_cli_version()

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Rev(), raising=True)

    def _boom(*a, **k):
        raise ConfoundMismatch("configs differ")

    monkeypatch.setattr("eval_core.runner.run_matrix", _boom, raising=True)

    with pytest.raises(ConfoundMismatch):
        run_eval.cmd_run(
            _ns(configs="A", strata=["s1_synthetic"], subdir="full", k=1, max_parallel=1)
        )
    manifest_path = tmp_path / "runs" / "full-manifest.json"
    assert manifest_path.exists(), "early abort must leave a failure marker, not a missing manifest"
    m = _json.loads(manifest_path.read_text())
    assert m.get("failed") is True and not m.get(
        "run_ids"
    ), "marker must be a failed, run_id-less manifest"


# --------------------------------------------------------------------------- #
# action_version confound + single-arm CLI pin + case_tag containment (local
# review round 5): each is a held-constant confound or file-surface contract.
# --------------------------------------------------------------------------- #


def test_experiment_tag_differs_by_action_version(monkeypatch):
    """action_version is a documented confound — changing it must bust experiment
    identity so a stale run can't be resumed/presented as the new experiment."""
    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    t1 = r.experiment_tag(Config(id="A", model="gpt-5.4", action_version="v1"))
    t2 = r.experiment_tag(Config(id="A", model="gpt-5.4", action_version="v2"))
    assert t1 != t2, "changing action_version must change experiment identity"


def test_review_rejects_non_v1_action_version(monkeypatch):
    """The harness runs `codex exec`, not the GH action, so it only models v1 —
    a non-v1 action_version must fail closed (recorded==executed)."""
    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    with pytest.raises(NotImplementedError):
        r.review(_case(), Config(id="A", model="gpt-5.4", action_version="v2"), 0)


def test_run_matrix_enforces_cli_pin_for_single_arm(monkeypatch):
    """A single-arm run must still run under the config's pinned CLI version — the
    pin check is fidelity, not a multi-arm-only concern."""
    from eval_core.models import Config
    from eval_core.runner import CLIVersionMismatch, run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)  # cli_version() reads the live codex (or sentinel)
    cfg = [Config(id="A", model="gpt-5.4", cli_version="codex-cli 0.0.0-nonexistent")]
    store = RunStore("/tmp/reviewer-eval-test/runs-clipin")
    with pytest.raises(CLIVersionMismatch):
        run_matrix([_case()], cfg, r, store, k=1, max_parallel=1)


def test_case_tag_rejects_escaping_patch_path(tmp_path):
    """case_tag must resolve the patch through the same containment check as
    materialization, so the hashing path can't read outside the case directory."""
    from adapters import worktree
    from adapters.codex_reviewer import CodexReviewer
    from eval_core.models import STRATUM_SYNTHETIC, Case

    r = CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="X")
    case = Case(
        id="c",
        stratum=STRATUM_SYNTHETIC,
        fixture={
            "kind": "stored_patch",
            "base_sha": "aaaaaaa",
            "patch": "../../../../etc/passwd",
            "_case_dir": str(tmp_path),
        },
    )
    with pytest.raises(worktree.MaterializeError):
        r.case_tag(case)


def test_load_cases_rejects_duplicate_case_id(tmp_path):
    """case.id is the primary key for caching/artifacts/bundle grouping; two cases
    sharing an id must fail closed at load, before any run."""
    import json as _json

    from adapters.corpus_loader import CorpusLoader

    cases_dir = tmp_path / "corpus" / "cases" / "s1_synthetic"
    for sub in ("a", "b"):
        d = cases_dir / sub
        d.mkdir(parents=True)
        (d / "case.json").write_text(
            _json.dumps(
                {
                    "id": "dup",
                    "stratum": "s1_synthetic",
                    "fixture": {"kind": "git_range", "base_sha": "aaaaaaa", "head_sha": "bbbbbbb"},
                }
            )
        )
    loader = CorpusLoader(str(tmp_path / "corpus"), str(_REPO))
    with pytest.raises(ValueError):
        loader.load_cases()


def test_compare_honors_case_allow_severities():
    """A negative-control case's own allow_severities must drive the grading rubric;
    the bundle must not hardcode a P2/P3 allowance that contradicts a stricter case."""
    from eval_core.compare import build_bundle
    from eval_core.models import RunResult

    snap = {
        "title": "Strict negative control",
        "stratum": "s3_negative",
        "expect_no_blockers": True,
        "allow_severities": ["P3"],
        "known_fp_topics": [],
        "ground_truth": [],
    }
    bundle = build_bundle(
        [
            RunResult(
                case_id="nc",
                config_id="A",
                repeat_idx=0,
                review_markdown="r",
                model="m",
                run_id="nc",
                case_snapshot=snap,
            )
        ]
    )
    assert "Allowed severities: P3" in bundle, "per-case allowance must be rendered"
    assert "P2/P3" not in bundle, "header must not hardcode a P2/P3 allowance"


def test_run_matrix_fails_closed_when_pinned_cli_unavailable(monkeypatch):
    """If a config pins a CLI version but the live version can't be read, abort —
    don't silently run under an unverified CLI (the pin is a recorded confound)."""
    from eval_core.models import Config
    from eval_core.runner import CLIVersionMismatch, run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    monkeypatch.setattr(
        r, "cli_version", lambda: (_ for _ in ()).throw(RuntimeError("codex missing")), raising=True
    )
    cfg = [Config(id="A", model="gpt-5.4", cli_version=_pinned_cli_version())]
    store = RunStore("/tmp/reviewer-eval-test/runs-clidown")
    with pytest.raises(CLIVersionMismatch):
        run_matrix([_case()], cfg, r, store, k=1, max_parallel=1)


def test_run_and_compare_reject_subdir_traversal(tmp_path, monkeypatch):
    """--subdir flows into filesystem paths; a `..` traversal must be rejected so the
    harness can't read/write outside runs/."""
    _stub_run_git(monkeypatch, tmp_path)
    run_eval = _run_eval()

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    rc_run = run_eval.cmd_run(
        _ns(configs="A", strata=["s1_synthetic"], subdir="../../escape", k=1, max_parallel=1)
    )
    assert rc_run != 0, "run must reject a traversing --subdir"
    rc_cmp = run_eval.cmd_compare(_ns(subdir="../../escape", allow_mixed=False))
    assert rc_cmp != 0, "compare must reject a traversing --subdir"


def test_resolve_configs_rejects_duplicate_ids():
    """Duplicate --configs ids (A,A) alias both arms onto one config_id; reject them
    rather than collapse the A/B comparison."""
    run_eval = _run_eval()

    assert run_eval._resolve_configs("A,A") is None
    assert run_eval._resolve_configs("A,B,A") is None
    assert run_eval._resolve_configs("A,B") is not None  # sanity: the valid case still works


def test_run_marks_failed_on_corpus_load_error(tmp_path, monkeypatch):
    """A corpus-load exception (e.g. the duplicate-id guard) must leave a failure
    marker so compare refuses the stale subdir — not leave a prior manifest live."""
    _stub_run_git(monkeypatch, tmp_path)
    import json as _json

    run_eval = _run_eval()
    from eval_core.store import write_json

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    manifest_path = tmp_path / "runs" / "full-manifest.json"
    # A prior successful manifest exists.
    write_json(str(manifest_path), {"run_ids": ["old"], "configs": ["A"]})

    def _boom(self, strata):
        raise ValueError("duplicate case id 'dup'")

    monkeypatch.setattr(run_eval.CorpusLoader, "load_cases", _boom, raising=True)
    with pytest.raises(ValueError):
        run_eval.cmd_run(_ns(configs="A", strata=None, subdir="full", k=1, max_parallel=1))
    m = _json.loads(manifest_path.read_text())
    assert m.get("failed") is True and not m.get(
        "run_ids"
    ), "load error must invalidate the manifest"
    assert run_eval.cmd_compare(_ns(subdir="full", allow_mixed=False)) != 0, "compare must refuse"


def test_run_bad_configs_invalidates_existing_manifest(tmp_path, monkeypatch):
    """A run attempt that exits at input validation (bad --configs) must still
    invalidate a PRIOR successful manifest, so compare doesn't render the stale one."""
    _stub_run_git(monkeypatch, tmp_path)
    import json as _json

    run_eval = _run_eval()
    from eval_core.store import write_json

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    manifest_path = tmp_path / "runs" / "full-manifest.json"
    write_json(str(manifest_path), {"run_ids": ["old"], "configs": ["A"]})

    rc = run_eval.cmd_run(_ns(configs="Z", strata=None, subdir="full", k=1, max_parallel=1))
    assert rc != 0, "bad --configs must fail"
    m = _json.loads(manifest_path.read_text())
    assert m.get("failed") is True and not m.get(
        "run_ids"
    ), "a bad-config rerun must invalidate the prior manifest"
    assert run_eval.cmd_compare(_ns(subdir="full", allow_mixed=False)) != 0


def test_verify_expected_file_matching_is_exact():
    """verify() must match expected files by exact repo-relative path, not suffix —
    duplicate basenames (runner.py, compare.py, ...) must not false-match."""
    from adapters.corpus_loader import _missing_expected

    touched = {"diff_diff/estimators.py", "tools/eval_core/runner.py"}
    assert _missing_expected(touched, {"diff_diff/estimators.py"}) == set()  # exact hit
    # same basename, different path -> must be flagged missing (no suffix false-match)
    assert _missing_expected(touched, {"other/pkg/estimators.py"}) == {"other/pkg/estimators.py"}
    assert _missing_expected(touched, {"runner.py"}) == {"runner.py"}  # basename alone != exact


def test_post_diff_paths_uses_rename_destination_only():
    """ground_truth.file is the POST-diff path, so a rename contributes only its
    destination — a case recording the pre-rename path must NOT verify."""
    from adapters.corpus_loader import _missing_expected, _post_diff_paths

    paths = _post_diff_paths("R100\told.py\tnew.py")
    assert paths == {"new.py"}, "rename must yield the destination only"
    assert _missing_expected(paths, {"new.py"}) == set(), "post-rename path accepted"
    assert _missing_expected(paths, {"old.py"}) == {"old.py"}, "pre-rename path rejected"
    # plain modify keeps its single path; delete contributes nothing.
    assert _post_diff_paths("M\tdiff_diff/estimators.py") == {"diff_diff/estimators.py"}
    assert _post_diff_paths("D\tgone.py") == set()


def test_case_snapshot_preserves_notes_and_rerun():
    """The bundle grades from the snapshot, so it must carry grading context: case
    notes and (for re-review cases) the prior review the reviewer was shown."""
    from eval_core.models import STRATUM_HISTORICAL, Case
    from eval_core.runner import _case_snapshot

    case = Case(
        id="c",
        stratum=STRATUM_HISTORICAL,
        notes="NOTE-ABC",
        fixture={"rerun": {"previous_review": "PRIOR-XYZ"}},
    )
    snap = _case_snapshot(case)
    assert snap["notes"] == "NOTE-ABC"
    assert snap["previous_review"] == "PRIOR-XYZ"


def test_compare_renders_grading_context():
    """build_bundle must render case notes and rerun prior-review so a grader can
    apply mitigation notes / re-review scope from the bundle alone."""
    from eval_core.compare import build_bundle
    from eval_core.models import RunResult

    snap = {
        "title": "t",
        "stratum": "s2_historical",
        "ground_truth": [],
        "expect_no_blockers": False,
        "allow_severities": ["P2", "P3"],
        "known_fp_topics": [],
        "notes": "KNOWN-MITIGATION-XYZ",
        "previous_review": "## prior: PRIOR-P1-ABC",
    }
    bundle = build_bundle(
        [
            RunResult(
                case_id="c",
                config_id="A",
                repeat_idx=0,
                review_markdown="r",
                model="m",
                run_id="c",
                case_snapshot=snap,
            )
        ]
    )
    assert "KNOWN-MITIGATION-XYZ" in bundle, "case notes must render in the bundle"
    assert "PRIOR-P1-ABC" in bundle, "rerun prior review must render in the bundle"
    assert "Re-review case" in bundle


def test_cleanup_removes_orphaned_worktree_dir(tmp_path):
    """An interrupted run can leave an orphaned dir that is no longer a registered
    worktree; cleanup must force-remove it (within the managed .worktrees root) so
    the next `git worktree add` to the same key doesn't fail."""
    from adapters import worktree

    orphan = tmp_path / ".worktrees" / "case.A.r0"
    orphan.mkdir(parents=True)
    (orphan / "leftover.txt").write_text("x")  # non-empty, not a registered worktree
    # repo_root has no .git, so the git worktree commands no-op (check=False).
    worktree.cleanup(str(orphan), str(tmp_path))
    assert not orphan.exists(), "an orphaned dir under .worktrees must be force-removed"


def test_validate_touched_files_negative_control_contract():
    """Negative controls must declare an exact expected_files contract; a clean case
    that silently becomes a code-changing diff must be rejected."""
    from adapters.corpus_loader import _validate_touched_files
    from eval_core.models import STRATUM_NEGATIVE, Case

    # expect_no_blockers without expected_files -> rejected (no file guard).
    nc = Case(id="nc", stratum=STRATUM_NEGATIVE, expect_no_blockers=True)
    assert _validate_touched_files(nc, {"CHANGELOG.md"}) is not None

    nc_ok = Case(
        id="nc", stratum=STRATUM_NEGATIVE, expect_no_blockers=True, expected_files=["CHANGELOG.md"]
    )
    assert _validate_touched_files(nc_ok, {"CHANGELOG.md"}) is None
    # a code file sneaking in -> exact-match contract rejects it.
    assert _validate_touched_files(nc_ok, {"CHANGELOG.md", "diff_diff/x.py"}) is not None
    # declared file not the one actually touched -> rejected.
    nc_wrong = Case(
        id="nc", stratum=STRATUM_NEGATIVE, expect_no_blockers=True, expected_files=["WRONG.md"]
    )
    assert _validate_touched_files(nc_wrong, {"CHANGELOG.md"}) is not None


def test_compare_renders_must_catch_and_fp_metadata():
    """The bundle must expose must_catch (optional bugs) and known-FP file/severity so
    graders can apply the full corpus contract from comparison.md alone."""
    from eval_core.compare import build_bundle
    from eval_core.models import RunResult

    def _bundle(snap):
        return build_bundle(
            [
                RunResult(
                    case_id="c",
                    config_id="A",
                    repeat_idx=0,
                    review_markdown="r",
                    model="m",
                    run_id="c",
                    case_snapshot=snap,
                )
            ]
        )

    gt = _bundle(
        {
            "title": "t",
            "stratum": "s1_synthetic",
            "ground_truth": [
                {
                    "id": "b1",
                    "file": "x.py",
                    "line_window": [1, 2],
                    "bug_class": "c",
                    "expected_severity": "P1",
                    "must_catch": False,
                }
            ],
            "expect_no_blockers": False,
            "allow_severities": ["P2", "P3"],
            "known_fp_topics": [],
        }
    )
    assert "optional" in gt.lower(), "a must_catch=false bug must render as optional"

    nc = _bundle(
        {
            "title": "t",
            "stratum": "s3_negative",
            "ground_truth": [],
            "expect_no_blockers": True,
            "allow_severities": ["P2", "P3"],
            "known_fp_topics": [
                {"topic": "naming nit", "file": "y.py", "would_be_severity_if_flagged": "P3"}
            ],
        }
    )
    assert "y.py" in nc and "P3" in nc, "known-FP file + would-be severity must render"


def test_worktree_leaf_is_collision_resistant_and_path_safe():
    """The worktree leaf must be a path-safe digest: distinct keys never alias (a/b vs
    a_b), and reserved/traversal ids can't escape the worktrees root."""
    import os

    from adapters.worktree import _worktree_leaf

    assert _worktree_leaf("a/b") != _worktree_leaf(
        "a_b"
    ), "lossy slug would collide; digest must not"
    root = "/tmp/x/.worktrees"
    for key in ("..", ".", "a/b", "../../etc", "case.A.r0"):
        leaf = _worktree_leaf(key)
        assert "/" not in leaf and "\\" not in leaf and not leaf.startswith(".")
        full = os.path.realpath(os.path.join(root, leaf))
        assert full.startswith(os.path.realpath(root) + os.sep), "leaf must stay under the root"


def test_cleanup_refuses_to_escape_worktrees_root(tmp_path):
    """cleanup must never rmtree outside a .worktrees leaf — a traversal path that
    resolves to the root's parent (or the root itself) must be refused."""
    from adapters import worktree

    wt_root = tmp_path / ".worktrees"
    wt_root.mkdir(parents=True)
    sentinel = tmp_path / "KEEP.txt"
    sentinel.write_text("keep")

    worktree.cleanup(str(wt_root / ".."), str(tmp_path))  # -> tmp_path
    assert sentinel.exists() and wt_root.exists(), "must not delete the worktrees root's parent"
    worktree.cleanup(str(wt_root / "."), str(tmp_path))  # -> the root itself
    assert wt_root.exists(), "must not delete the .worktrees root itself"


def test_load_cases_rejects_reserved_case_id(tmp_path):
    """A reserved dot-segment case id must be rejected at load (never used as a path)."""
    import json as _json

    from adapters.corpus_loader import CorpusLoader

    d = tmp_path / "corpus" / "cases" / "s1_synthetic" / "dotcase"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        _json.dumps(
            {
                "id": "..",
                "stratum": "s1_synthetic",
                "fixture": {"kind": "git_range", "base_sha": "aaaaaaa", "head_sha": "bbbbbbb"},
            }
        )
    )
    loader = CorpusLoader(str(tmp_path / "corpus"), str(_REPO))
    with pytest.raises(ValueError):
        loader.load_cases()


def test_compare_separates_case_versions():
    """Two versions of the same case_id (distinct snapshots, possible under
    --allow-mixed) must render as SEPARATE sections, not be conflated under one
    heading with the first version's ground truth."""
    from eval_core.compare import build_bundle
    from eval_core.models import RunResult

    def _rr(rid, marker):
        return RunResult(
            case_id="c",
            config_id="A",
            repeat_idx=0,
            review_markdown="r",
            model="m",
            run_id=rid,
            case_snapshot={
                "title": "t",
                "stratum": "s1_synthetic",
                "ground_truth": [],
                "expect_no_blockers": False,
                "allow_severities": ["P2", "P3"],
                "known_fp_topics": [],
                "notes": marker,
            },
        )

    bundle = build_bundle([_rr("r1", "VERSION-ONE"), _rr("r2", "VERSION-TWO")])
    assert "VERSION-ONE" in bundle and "VERSION-TWO" in bundle, "both versions must render"
    assert "variant" in bundle, "differing snapshots under one case_id must be marked variants"
    assert "variant" not in build_bundle([_rr("r1", "ONLY")]), "single version is not a variant"


def test_smoke_clears_cache_for_live_run(tmp_path, monkeypatch):
    """smoke is a live plumbing check, so it must clear cached artifacts and actually
    exercise codex rather than resume a stale success."""
    _stub_run_git(monkeypatch, tmp_path)
    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    smoke_dir = tmp_path / "runs" / "smoke"
    smoke_dir.mkdir(parents=True)
    stale = smoke_dir / "stale.json"
    stale.write_text("{}")
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    class _Rev:
        def cli_version(self):
            return _pinned_cli_version()

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Rev(), raising=True)
    monkeypatch.setattr("eval_core.runner.run_matrix", lambda *a, **k: [], raising=True)
    run_eval.cmd_smoke(_ns(configs="A", strata=None, k=1, limit=1, max_parallel=2))
    assert not stale.exists(), "smoke must clear cached artifacts so it runs live"


def test_load_cases_rejects_stratum_dir_mismatch(tmp_path):
    """A case.json whose declared stratum differs from its cases/<stratum>/ directory
    must be rejected — otherwise --strata X could run a case reported under Y."""
    import json as _json

    from adapters.corpus_loader import CorpusLoader

    d = tmp_path / "corpus" / "cases" / "s1_synthetic" / "misfiled"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        _json.dumps(
            {
                "id": "misfiled",
                "stratum": "s3_negative",  # declared != containing dir
                "fixture": {"kind": "git_range", "base_sha": "aaaaaaa", "head_sha": "bbbbbbb"},
            }
        )
    )
    loader = CorpusLoader(str(tmp_path / "corpus"), str(_REPO))
    with pytest.raises(ValueError):
        loader.load_cases()


def test_materialize_rewraps_post_add_git_failure(tmp_path, monkeypatch):
    """A non-MaterializeError git failure after `worktree add` must clean up the
    detached worktree and surface as a case-scoped MaterializeError (-> INFRA_ERROR),
    not leak the worktree and crash with a raw traceback."""
    import subprocess

    from adapters import worktree

    cleaned = {"n": 0}
    monkeypatch.setattr(
        worktree,
        "cleanup",
        lambda *a, **k: cleaned.__setitem__("n", cleaned["n"] + 1),
        raising=True,
    )

    def _fake_git(repo, args, check=True):
        if args[:2] == ["worktree", "add"] or args[:1] == ["apply"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:1] == ["add"]:
            raise subprocess.CalledProcessError(1, args)  # surprise post-add failure
        # rev-parse / cat-file used by _resolve / _ensure_present
        return subprocess.CompletedProcess(args, 0, "deadbeef" * 5 + "\n", "")

    monkeypatch.setattr(worktree, "_git", _fake_git, raising=True)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "inject.diff").write_text("patch bytes")
    with pytest.raises(worktree.MaterializeError):
        worktree.materialize(
            "c",
            {"kind": "stored_patch", "base_sha": "a" * 40, "patch": "inject.diff"},
            str(tmp_path / "repo"),
            str(tmp_path / "repo" / ".worktrees"),
            case_dir=str(case_dir),
        )
    assert cleaned["n"] >= 1, "a post-add failure must clean up the leaked worktree"


def test_worktrees_namespaced_per_invocation(monkeypatch):
    """Two invocations (CodexReviewer instances) must NOT share a worktree path for the
    same (case, config, repeat) — otherwise a concurrent smoke/run could cleanup() each
    other's live checkout mid-review and corrupt the A/B."""
    from adapters import ci_prompt, worktree
    from adapters.codex_reviewer import CodexReviewer
    from eval_core.models import STRATUM_HISTORICAL, Case

    seen = []

    class _Mat:
        worktree_dir = "/tmp/reviewer-eval-test/wt"
        base_sha = "b"
        head_sha = "h"

    def _fake_mat(case_id, fixture, repo_root, worktrees_root, case_dir="", worktree_key=None):
        seen.append(worktree_key)
        return _Mat()

    monkeypatch.setattr(worktree, "materialize", _fake_mat, raising=True)
    monkeypatch.setattr(worktree, "cleanup", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(ci_prompt, "build_ci_prompt", lambda **kw: "PROMPT", raising=True)

    case = Case(id="x", stratum=STRATUM_HISTORICAL, fixture={"_case_dir": ""})
    r1 = CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="B")
    r2 = CodexReviewer(repo_root=str(_REPO), runs_root="/tmp/reviewer-eval-test", prompt_text="B")
    assert r1._wt_namespace != r2._wt_namespace, "each invocation gets a unique worktree namespace"
    r1.prompt_sha_for(case)
    r2.prompt_sha_for(case)
    assert seen[0] != seen[1], "same case, different invocations -> distinct worktree keys"
    assert r1._wt_namespace in seen[0] and r2._wt_namespace in seen[1]


def test_resolve_configs_rejects_empty_selectors():
    """Malformed comma selectors must fail closed, not silently drop empty segments and
    run a narrower matrix than intended."""
    run_eval = _run_eval()

    for bad in ("A,", ",A", "A,,B", "", ",", "A, ,B"):
        assert run_eval._resolve_configs(bad) is None, f"{bad!r} must fail closed"
    assert run_eval._resolve_configs("A,B") is not None, "valid A,B still resolves"
    assert run_eval._resolve_configs("A") is not None, "valid single arm still resolves"


def test_load_cases_distinguishes_none_from_empty_strata():
    """No --strata (None) loads all; a bare --strata ([]) selects NOTHING (fail closed),
    not the whole corpus."""
    from adapters.corpus_loader import CorpusLoader

    loader = CorpusLoader(str(_EVAL_ROOT / "corpus"), str(_REPO))
    assert len(loader.load_cases(None)) >= 2, "None (no flag) loads all strata"
    assert loader.load_cases([]) == [], "bare --strata ([]) must select nothing"
    s3_ids = {c.id for c in loader.load_cases(["s3_negative"])}
    assert "s3-changelog-prose" in s3_ids, "stratum selection must load the s3 cases"
    assert all(c.startswith("s3-") for c in s3_ids), "stratum selection must not leak other strata"


# --------------------------------------------------------------------------- #
# N-arm configs.json (gpt-5.6 eval): fail-closed loading + declared treatments.
# --------------------------------------------------------------------------- #


def _write_configs(tmp_path, payload):
    import json as _json

    (tmp_path / "configs.json").write_text(_json.dumps(payload))
    return str(tmp_path)


def _arm(id_, model="m", effort="xhigh", role=None, **kw):
    d = {"id": id_, "model": model, "effort": effort}
    if role:
        d["role"] = role
    d.update(kw)
    return d


def test_make_configs_resolves_four_arms_in_order():
    """The live configs.json still carries the historical 2026-07 A-D subset (the
    4-arm gpt-5.6 matrix); ids resolve in the requested order and unknown ids still
    fail closed."""
    run_eval = _run_eval()

    cfgs = run_eval._make_configs(["A", "B", "C", "D"])
    assert [c.id for c in cfgs] == ["A", "B", "C", "D"]
    assert run_eval._resolve_configs("A,B,C,D") is not None
    assert run_eval._resolve_configs("A,Z") is None, "unknown id must still fail closed"
    # Exactly one declared control, and every pairwise contrast the runner will
    # accept is single-field: B-A={model}, C-B={model}, D-B={effort}.
    by_id = {c.id: c for c in cfgs}
    assert by_id["A"].model != by_id["B"].model
    assert by_id["A"].effort == by_id["B"].effort == by_id["C"].effort
    assert by_id["D"].model == by_id["B"].model and by_id["D"].effort != by_id["B"].effort


def test_make_configs_fails_closed_on_malformed(tmp_path, monkeypatch):
    """A malformed configs.json must abort, never quietly run a different
    experiment than the file describes."""
    run_eval = _run_eval()

    bad_payloads = [
        {},  # no arms at all
        {"arms": []},  # empty arms
        {"arms": [_arm("A", role="control"), _arm("A")]},  # duplicate id
        {"arms": [_arm("A", role="control"), _arm("B", extra="nope")]},  # unknown key
        {"arms": [_arm("A", role="control"), _arm("B", model="")]},  # missing model
        {"arms": [_arm("A", role="control"), _arm("B", effort="")]},  # missing effort
        {"arms": [_arm("A"), _arm("B")]},  # zero controls
        {"arms": [_arm("A", role="control"), _arm("B", role="control")]},  # two controls
    ]
    for payload in bad_payloads:
        monkeypatch.setattr(run_eval, "CONFIG_DIR", _write_configs(tmp_path, payload))
        with pytest.raises(ValueError):
            run_eval._make_configs(["A"])


def test_treatment_fields_validated(tmp_path, monkeypatch):
    """treatment_fields must be a clean subset of the contrastable Config fields;
    absent -> the classic model-only default."""
    run_eval = _run_eval()

    assert run_eval._treatment_fields() == ("model", "effort"), "live configs.json declaration"

    ok = {"arms": [_arm("A", role="control"), _arm("B", model="m2")]}
    monkeypatch.setattr(run_eval, "CONFIG_DIR", _write_configs(tmp_path, ok))
    assert run_eval._treatment_fields() == ("model",), "absent -> default model-only"

    for bad in (["model", "typo"], [], "model", ["model", "model"]):
        payload = dict(ok)
        payload["treatment_fields"] = bad
        monkeypatch.setattr(run_eval, "CONFIG_DIR", _write_configs(tmp_path, payload))
        with pytest.raises(ValueError):
            run_eval._treatment_fields()


def test_run_matrix_allows_declared_effort_treatment(monkeypatch):
    """With effort DECLARED as a treatment, a model-identical effort contrast
    (arm B vs D) must run — while sandbox drift still aborts."""
    from eval_core.models import Config
    from eval_core.runner import ConfoundMismatch, run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-effort-treatment")
    cfgs = [
        Config(id="B", model="gpt-5.6-sol", effort="xhigh"),
        Config(id="D", model="gpt-5.6-sol", effort="max"),
    ]
    results = run_matrix(
        [_case()], cfgs, r, store, k=1, max_parallel=1, treatment_fields=("model", "effort")
    )
    assert len(results) == 2 and all(rr.ok for rr in results)
    assert {rr.effort for rr in results} == {"xhigh", "max"}, "effort recorded per arm"

    drift = [
        Config(id="B", model="gpt-5.6-sol", sandbox="read-only"),
        Config(id="X", model="gpt-5.5", sandbox="workspace-write"),
    ]
    with pytest.raises(ConfoundMismatch):
        run_matrix(
            [_case()], drift, r, store, k=1, max_parallel=1, treatment_fields=("model", "effort")
        )


def test_run_matrix_rejects_duplicate_treatment_tuple(monkeypatch):
    from eval_core.models import Config
    from eval_core.runner import ConfoundMismatch, run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-dup-treatment")
    cfgs = [
        Config(id="A", model="gpt-5.6-sol", effort="xhigh"),
        Config(id="B", model="gpt-5.6-sol", effort="xhigh"),
    ]
    with pytest.raises(ConfoundMismatch):
        run_matrix(
            [_case()], cfgs, r, store, k=1, max_parallel=1, treatment_fields=("model", "effort")
        )


def test_run_matrix_rejects_jointly_confounded_pair(monkeypatch):
    """A,D alone differ in model AND effort with no bridging arm -> a confounded
    2-arm contrast the runner must refuse (the full matrix decomposes fine)."""
    from eval_core.models import Config
    from eval_core.runner import ConfoundMismatch, run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-joint-confound")
    cfgs = [
        Config(id="A", model="gpt-5.5", effort="xhigh"),
        Config(id="D", model="gpt-5.6-sol", effort="max"),
    ]
    with pytest.raises(ConfoundMismatch):
        run_matrix(
            [_case()], cfgs, r, store, k=1, max_parallel=1, treatment_fields=("model", "effort")
        )


def test_run_matrix_full_matrix_decomposes(monkeypatch):
    """The 4-arm matrix satisfies the single-field-contrast rule: every arm has a
    one-field neighbor (B-A model, C-B model, D-B effort)."""
    from eval_core.models import Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-full-matrix")
    cfgs = [
        Config(id="A", model="gpt-5.5", effort="xhigh"),
        Config(id="B", model="gpt-5.6-sol", effort="xhigh"),
        Config(id="C", model="gpt-5.6-terra", effort="xhigh"),
        Config(id="D", model="gpt-5.6-sol", effort="max"),
    ]
    results = run_matrix(
        [_case()], cfgs, r, store, k=1, max_parallel=1, treatment_fields=("model", "effort")
    )
    assert len(results) == 4 and all(rr.ok for rr in results)


def test_run_matrix_single_arm_exempt_from_treatment_rules(monkeypatch):
    """One arm can't be confounded: a lone max-effort arm (the D smoke) must run."""
    from eval_core.models import Config
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-single-max")
    results = run_matrix(
        [_case()],
        [Config(id="D", model="gpt-5.6-sol", effort="max")],
        r,
        store,
        k=1,
        max_parallel=1,
        treatment_fields=("model", "effort"),
    )
    assert len(results) == 1 and results[0].ok


def test_experiment_tag_differs_by_effort(monkeypatch):
    """Effort is part of experiment identity: a D run can never alias a B run."""
    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    tag_b = r.experiment_tag(Config(id="X", model="gpt-5.6-sol", effort="xhigh"))
    tag_d = r.experiment_tag(Config(id="X", model="gpt-5.6-sol", effort="max"))
    assert tag_b != tag_d, "same model + different effort must yield distinct tags"


def test_review_passes_effort_and_timeout_to_call_codex(monkeypatch):
    """review() must execute the DECLARED effort (recorded == executed) and pass
    the harness's per-run timeout ceiling."""
    from adapters.codex_reviewer import CodexReviewer
    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    captured = {}

    def _capture(prompt, model, repo_root, effort="xhigh", timeout_s=None):
        captured["effort"] = effort
        captured["timeout_s"] = timeout_s
        return "## ok", {"backend": "codex"}

    monkeypatch.setattr(r._mod, "call_codex", _capture, raising=True)
    out = r.review(_case(), Config(id="D", model="gpt-5.6-sol", effort="max"), 0)
    assert out.review_markdown == "## ok"
    assert captured["effort"] == "max"
    assert captured["timeout_s"] == CodexReviewer.CALL_TIMEOUT_S


def test_review_rejects_unverified_effort(monkeypatch):
    """Levels outside SUPPORTED_EFFORTS fail closed — never silently run a
    different level than recorded."""
    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    with pytest.raises(NotImplementedError, match="SUPPORTED_EFFORTS|supports"):
        r.review(_case(), Config(id="E", model="gpt-5.6-sol", effort="high"), 0)


# --------------------------------------------------------------------------- #
# Per-config repeats (--k-per): k=2 on the primary arms, k=1 probes, ONE
# invocation = one manifest.
# --------------------------------------------------------------------------- #


def test_plan_runs_k_overrides():
    from eval_core.models import STRATUM_SYNTHETIC, Case, Config
    from eval_core.runner import _plan_runs

    cases = [Case(id="c1", stratum=STRATUM_SYNTHETIC), Case(id="c2", stratum=STRATUM_SYNTHETIC)]
    cfgs = [Config(id=i, model="m") for i in ("A", "B", "C")]
    jobs = _plan_runs(cases, cfgs, k=2, k_overrides={"C": 1})
    per_config = {}
    for _case_, cfg, _r in jobs:
        per_config[cfg.id] = per_config.get(cfg.id, 0) + 1
    assert per_config == {"A": 4, "B": 4, "C": 2}, "2 cases x (A2/B2/C1) repeats"


def test_parse_k_per_fail_closed():
    run_eval = _run_eval()

    cfgs = run_eval._make_configs(["A", "B", "C", "D"])
    assert run_eval._parse_k_per("", cfgs) == {}, "absent flag -> no overrides"
    assert run_eval._parse_k_per("C=1,D=1", cfgs) == {"C": 1, "D": 1}
    for bad in ("C", "C=", "=1", "C=0", "C=-1", "C=x", "C=1,C=2", "Z=1", "C=1,,D=1"):
        assert run_eval._parse_k_per(bad, cfgs) is None, f"{bad!r} must fail closed"


def test_cmd_run_k_per_end_to_end(tmp_path, monkeypatch):
    """cmd_run with --k 2 --k-per B=1 must execute A twice and B once per case,
    record k/k_per in the manifest, and refuse a bad --k-per up front."""
    _stub_run_git(monkeypatch, tmp_path)
    import json as _json

    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case, ReviewOutput

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    counts = {}

    class _Counting:
        def cli_version(self):
            return _pinned_cli_version()

        def experiment_tag(self, config):
            return f"tag-{config.id}"

        def case_tag(self, case):
            return "ctag"

        def prompt_sha_for(self, case):
            return "psha"

        def review(self, case, config, repeat_idx):
            counts[config.id] = counts.get(config.id, 0) + 1
            return ReviewOutput(
                review_markdown="## ok",
                cli_version=_pinned_cli_version(),
                latency_s=0.0,
                usage={"prompt_sha": "psha"},
            )

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Counting(), raising=True)
    rc = run_eval.cmd_run(
        _ns(
            configs="A,B",
            strata=["s1_synthetic"],
            subdir="kper",
            k=2,
            k_per="B=1",
            max_parallel=1,
        )
    )
    assert rc == 0
    assert counts == {"A": 2, "B": 1}, "per-config repeat overrides must drive the plan"
    manifest = _json.loads((tmp_path / "runs" / "kper-manifest.json").read_text())
    assert manifest.get("k") == 2 and manifest.get("k_per") == {"B": 1}

    counts.clear()
    rc_bad = run_eval.cmd_run(
        _ns(
            configs="A,B",
            strata=["s1_synthetic"],
            subdir="kper2",
            k=2,
            k_per="Z=1",
            max_parallel=1,
        )
    )
    assert rc_bad != 0 and not counts, "a bad --k-per must abort before any review"


# --------------------------------------------------------------------------- #
# compare --blinded: identity-stripped bundle + sealed mapping.
# --------------------------------------------------------------------------- #


def _run_ok_experiment(tmp_path, monkeypatch, subdir="blind"):
    """Drive a real cmd_run (stubbed reviewer) so cmd_compare sees a valid
    manifest; returns the runs dir."""
    _stub_run_git(monkeypatch, tmp_path)
    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case, ReviewOutput

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    class _Ok:
        def cli_version(self):
            return _pinned_cli_version()

        def experiment_tag(self, config):
            return f"tag-{config.id}"

        def case_tag(self, case):
            return "ctag"

        def prompt_sha_for(self, case):
            return "psha"

        def review(self, case, config, repeat_idx):
            return ReviewOutput(
                review_markdown=f"As {config.model}, I see no issues.",
                cli_version=_pinned_cli_version(),
                latency_s=0.0,
                usage={"prompt_sha": "psha"},
            )

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Ok(), raising=True)
    rc = run_eval.cmd_run(
        _ns(configs="A,B", strata=["s1_synthetic"], subdir=subdir, k=1, max_parallel=1)
    )
    assert rc == 0
    return tmp_path / "runs"


def test_cmd_compare_blinded_writes_sealed_bundle(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    monkeypatch.setattr(run_eval, "_assert_tree_identity", lambda *a, **k: None)

    runs_dir = _run_ok_experiment(tmp_path, monkeypatch)
    rc = run_eval.cmd_compare(_ns(subdir="blind", allow_mixed=False, blinded=True))
    assert rc == 0
    blinded_md = (runs_dir / "blind" / "comparison.blinded.md").read_text()
    blinding = _json.loads((runs_dir / "blind" / "blinding.json").read_text())
    # Neutral labels present; real ids and every model identity absent.
    assert sorted(blinding["mapping"].keys()) == ["A", "B"]
    assert sorted(blinding["mapping"].values()) == ["M1", "M2"]
    for label in blinding["mapping"].values():
        assert f"### {label} — review" in blinded_md
    lowered = blinded_md.lower()
    assert "gpt-" not in lowered, "no model family string may survive blinding"
    assert "gpt-5.5" not in lowered and "sol" not in lowered
    assert "### A " not in blinded_md and "### B " not in blinded_md
    assert "latency" not in lowered, "latency is a side channel and must be redacted"
    # The unblinded bundle is still written and still names models.
    unblinded = (runs_dir / "blind" / "comparison.md").read_text()
    assert "gpt-5.5" in unblinded


def test_cmd_compare_blinded_mapping_stable_across_rerenders(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    monkeypatch.setattr(run_eval, "_assert_tree_identity", lambda *a, **k: None)

    runs_dir = _run_ok_experiment(tmp_path, monkeypatch, subdir="stable")
    assert run_eval.cmd_compare(_ns(subdir="stable", allow_mixed=False, blinded=True)) == 0
    first = _json.loads((runs_dir / "stable" / "blinding.json").read_text())
    assert run_eval.cmd_compare(_ns(subdir="stable", allow_mixed=False, blinded=True)) == 0
    second = _json.loads((runs_dir / "stable" / "blinding.json").read_text())
    assert first["mapping"] == second["mapping"], "same experiment -> same permutation"


def test_cmd_compare_blinded_refusals(tmp_path, monkeypatch):
    """--blinded is manifest-scoped by construction: refuse --allow-mixed and
    refuse when the manifest is missing."""
    run_eval = _run_eval()
    monkeypatch.setattr(run_eval, "_assert_tree_identity", lambda *a, **k: None)

    runs_dir = _run_ok_experiment(tmp_path, monkeypatch, subdir="refuse")
    assert (
        run_eval.cmd_compare(_ns(subdir="refuse", allow_mixed=True, blinded=True)) != 0
    ), "--blinded + --allow-mixed must refuse"
    (runs_dir / "refuse-manifest.json").unlink()
    assert (
        run_eval.cmd_compare(_ns(subdir="refuse", allow_mixed=False, blinded=True)) != 0
    ), "--blinded without a manifest must refuse"


def test_run_matrix_holds_model_constant_when_not_a_treatment(monkeypatch):
    """An effort-only experiment must hold MODEL constant: arms differing in both
    model and effort under treatment_fields=("effort",) are silently confounded
    and must abort (local review R1 P1)."""
    from eval_core.models import Config
    from eval_core.runner import ConfoundMismatch, run_matrix
    from eval_core.store import RunStore

    r = _make_reviewer(monkeypatch)
    store = RunStore("/tmp/reviewer-eval-test/runs-model-confound")
    cfgs = [
        Config(id="A", model="gpt-5.6-sol", effort="xhigh"),
        Config(id="B", model="gpt-5.5", effort="max"),
    ]
    with pytest.raises(ConfoundMismatch):
        run_matrix([_case()], cfgs, r, store, k=1, max_parallel=1, treatment_fields=("effort",))


def test_smoke_default_selects_control_arm(tmp_path, monkeypatch):
    """Bare `smoke` (no --configs) must exercise the sole role=control arm from
    configs.json - a control flip must never leave the default smoking a stale
    arm (local review P2 on the gpt-5.6 swap)."""
    _stub_run_git(monkeypatch, tmp_path)
    import json as _json

    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    class _Rev:
        def cli_version(self):
            return _pinned_cli_version()

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Rev(), raising=True)

    captured = {}

    def _capture_run_matrix(cases, configs, *a, **k):
        captured["config_ids"] = [c.id for c in configs]
        return []

    monkeypatch.setattr("eval_core.runner.run_matrix", _capture_run_matrix, raising=True)
    rc = run_eval.cmd_smoke(_ns(configs=None, strata=None, k=1, limit=1, max_parallel=2))
    assert rc == 0

    live = _json.loads((_EVAL_ROOT / "config" / "configs.json").read_text())
    control = [c["id"] for c in live["arms"] if c.get("role") == "control"]
    assert captured["config_ids"] == control, "bare smoke must run exactly the control arm"
    assert run_eval._control_id() == control[0]


# --------------------------------------------------------------------------- #
# 2026-10 gpt-6 campaign: configs, provenance, isolated home, sealing, guards.
# --------------------------------------------------------------------------- #


def _git(repo, *args):
    import subprocess

    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tmp_repo(tmp_path, name="repo"):
    """A committed tmp git repo (inline identity, no global config needed)."""
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "trunk")
    (repo / "README.md").write_text("hello\n")
    (repo / ".gitignore").write_text("__pycache__/\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "-m", "init")
    return repo


def test_live_configs_resolve_control_and_candidate():
    run_eval = _run_eval()

    cfgs = run_eval._resolve_configs("B,E")
    assert cfgs is not None and [c.id for c in cfgs] == ["B", "E"]
    b, e = cfgs
    assert b.model != e.model and b.effort == e.effort == "xhigh"
    assert b.sandbox == e.sandbox and b.action_version == e.action_version
    assert b.cli_version == e.cli_version == "codex-cli 0.162.1"
    assert run_eval._control_id() == "B"
    assert run_eval._treatment_fields() == ("model", "effort")
    # Bare `run` is the registered campaign shape end to end: the COMPLETE default
    # argument combination passes the candidate-exposure gate (configs B,E at k=2,
    # no --k-per, no --strata), so `python run_eval.py run` is not configured to refuse.
    ns = run_eval._build_parser().parse_args(["run"])
    assert ns.configs == "B,E" and ns.k == 2 and ns.k_per == "" and ns.strata is None
    assert (
        run_eval._candidate_exposure_violations(
            cfgs, ns.k, run_eval._parse_k_per(ns.k_per, cfgs), ns.strata, "run"
        )
        == []
    )


def test_git_provenance_seven_keys_on_tmp_repo(tmp_path):
    from adapters import worktree

    repo = _tmp_repo(tmp_path)
    prov = worktree.git_provenance(str(repo))
    assert sorted(prov) == sorted(
        [
            "repo_head_sha",
            "repo_tree_sha",
            "repo_dirty",
            "repo_branch",
            "repo_common_dir",
            "repo_worktree_paths",
            "repo_ref_names",
        ]
    )
    assert prov["repo_head_sha"] == _git(repo, "rev-parse", "HEAD")
    assert prov["repo_tree_sha"] == _git(repo, "rev-parse", "HEAD^{tree}")
    assert prov["repo_dirty"] is False
    assert prov["repo_branch"] == "trunk"
    assert os.path.isabs(prov["repo_common_dir"])
    assert os.path.realpath(prov["repo_common_dir"]) == os.path.realpath(str(repo / ".git"))
    assert prov["repo_ref_names"] == ["trunk"]
    # Own-harness worktrees are excluded; a linked worktree elsewhere is included.
    own = repo / "tools" / "reviewer-eval" / "runs" / ".worktrees"
    own.mkdir(parents=True)
    _git(repo, "worktree", "add", "--detach", str(own / "x"), "HEAD")
    other = tmp_path / "linked-elsewhere"
    _git(repo, "worktree", "add", "--detach", str(other), "HEAD")
    prov2 = worktree.git_provenance(str(repo))
    assert [os.path.realpath(p) for p in prov2["repo_worktree_paths"]] == [
        os.path.realpath(str(other))
    ]
    assert os.path.realpath(str(repo)) not in [
        os.path.realpath(p) for p in prov2["repo_worktree_paths"]
    ]
    (repo / "stray.txt").write_text("x")
    assert worktree.git_provenance(str(repo))["repo_dirty"] is True
    # Non-git dir: fail-soft values, None (never []) for the list fields.
    empty = tmp_path / "not-a-repo"
    empty.mkdir()
    prov3 = worktree.git_provenance(str(empty))
    assert prov3["repo_head_sha"] == "" and prov3["repo_dirty"] is None
    assert prov3["repo_worktree_paths"] is None and prov3["repo_ref_names"] is None


def test_review_records_provenance_worktree_session_and_home(monkeypatch):
    from adapters import codex_reviewer as cr

    r = _make_reviewer(monkeypatch)
    monkeypatch.setattr(
        r._mod,
        "call_codex",
        lambda prompt, model, repo_root, effort="xhigh", timeout_s=None: (
            "## ok",
            {
                "backend": "codex",
                "session_commands": ["/bin/zsh -lc \"sed -n '1,3p' a.py\" in /tmp/x"],
            },
        ),
        raising=True,
    )
    from eval_core.models import Config

    cfg = Config(id="B", model="m", effort="xhigh", cli_version=_pinned_cli_version())
    out = r.review(_case(), cfg, 0)
    for k, v in _STUB_PROVENANCE.items():
        assert out.usage[k] == v
    assert out.usage["worktree_dir"] == "/tmp/reviewer-eval-test/wt"
    assert out.usage["session_commands"] == ["/bin/zsh -lc \"sed -n '1,3p' a.py\" in /tmp/x"]
    assert out.usage["codex_home_path"] is None and out.usage["codex_home_isolated"] is False
    monkeypatch.setattr(cr, "ACTIVE_ISOLATED_HOME", "/tmp/codex-home-t")
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex-home-t")
    out2 = r.review(_case(), cfg, 1)
    assert out2.usage["codex_home_path"] == "/tmp/codex-home-t"
    assert out2.usage["codex_home_isolated"] is True


def test_review_worktree_key_is_salted_opaque_hash(monkeypatch):
    import hashlib

    from eval_core.models import Config

    r = _make_reviewer(monkeypatch)
    seen = []
    monkeypatch.setattr(
        r,
        "build_prompt_for_case",
        lambda case, worktree_key=None: (
            seen.append(worktree_key) or ("PROMPT", "/tmp/reviewer-eval-test/wt", "deadbeef")
        ),
        raising=True,
    )
    cfg_b = Config(id="B", model="m", effort="xhigh", cli_version=_pinned_cli_version())
    cfg_e = Config(id="E", model="n", effort="xhigh", cli_version=_pinned_cli_version())
    case = _case()
    r.review(case, cfg_b, 0)
    r.review(case, cfg_b, 1)
    r.review(case, cfg_e, 0)
    assert len(seen) == 3 and len(set(seen)) == 3
    for key in seen:
        assert re.fullmatch(r"[0-9a-f]{32}", key), key
    unsalted = hashlib.sha256(f"{case.id}.B.r0".encode()).hexdigest()[:32]
    namespaced = hashlib.sha256(f"{r._wt_namespace}.{case.id}.B.r0".encode()).hexdigest()[:32]
    assert seen[0] not in (unsalted, namespaced), "the salt must be unrecomputable"
    r2 = _make_reviewer(monkeypatch)
    seen2 = []
    monkeypatch.setattr(
        r2,
        "build_prompt_for_case",
        lambda case, worktree_key=None: (
            seen2.append(worktree_key) or ("PROMPT", "/tmp/reviewer-eval-test/wt", "deadbeef")
        ),
        raising=True,
    )
    r2.review(case, cfg_b, 0)
    assert seen2[0] != seen[0], "two instances must not share review worktree keys"


def _fake_real_home(tmp_path, monkeypatch, with_auth=True):
    real = tmp_path / "real-codex"
    real.mkdir()
    if with_auth:
        (real / "auth.json").write_text('{"token": "entry"}')
    (real / "config.toml").write_text('[mcp_servers.x]\ncommand = "node"\n')
    (real / "memories").mkdir()
    monkeypatch.setenv("CODEX_HOME", str(real))
    return real


def test_isolated_codex_home_mirrors_ci_and_restores_env(tmp_path, monkeypatch):
    from adapters import codex_reviewer as cr

    real = _fake_real_home(tmp_path, monkeypatch)
    with cr.isolated_codex_home() as home:
        assert sorted(os.listdir(home)) == ["auth.json", "config.toml"]
        assert (pathlib.Path(home) / "config.toml").read_text() == cr.ISOLATED_CONFIG_TOML
        assert 'sandbox_mode = "read-only"' in cr.ISOLATED_CONFIG_TOML
        assert 'approval_policy = "never"' in cr.ISOLATED_CONFIG_TOML
        assert "[features]" not in cr.ISOLATED_CONFIG_TOML and "mcp" not in cr.ISOLATED_CONFIG_TOML
        assert cr.ACTIVE_ISOLATED_HOME == home and os.environ["CODEX_HOME"] == home
        assert home != str(real)
    assert cr.ACTIVE_ISOLATED_HOME is None
    assert os.environ["CODEX_HOME"] == str(real)
    assert not os.path.exists(home)
    # (a) isolated copy unchanged -> real file untouched, nothing else written.
    assert (real / "auth.json").read_text() == '{"token": "entry"}'
    assert sorted(os.listdir(real)) == ["auth.json", "config.toml", "memories"]


def test_isolated_codex_home_saves_rotated_token_beside_real(tmp_path, monkeypatch):
    from adapters import codex_reviewer as cr

    real = _fake_real_home(tmp_path, monkeypatch)
    with cr.isolated_codex_home() as home:
        (pathlib.Path(home) / "auth.json").write_text('{"token": "rotated"}')
    # (b) real file byte-identical to entry; exactly one campaign copy holds the new bytes.
    assert (real / "auth.json").read_text() == '{"token": "entry"}'
    saved = sorted(real.glob("auth.json.campaign-*"))
    assert len(saved) == 1 and saved[0].read_text() == '{"token": "rotated"}'
    # The rerun seeds from the NEWEST candidate (the rotated copy).
    os.utime(saved[0], None)
    with cr.isolated_codex_home() as home2:
        assert (pathlib.Path(home2) / "auth.json").read_text() == '{"token": "rotated"}'


def test_isolated_codex_home_requires_credential(tmp_path, monkeypatch):
    from adapters import codex_reviewer as cr

    _fake_real_home(tmp_path, monkeypatch, with_auth=False)
    with pytest.raises(RuntimeError, match="no codex credential"):
        with cr.isolated_codex_home():
            pass


def test_cmd_run_enters_isolated_home_once_and_writes_provenance(tmp_path, monkeypatch):
    """The happy path enters the isolated home exactly once around the matrix, and
    the manifest carries the seven provenance keys, codex_home_path, the
    aggregates and the artifacts seal."""
    import json as _json

    run_eval = _run_eval()
    runs_dir = _run_ok_experiment(tmp_path, monkeypatch, subdir="prov")
    fake = run_eval.isolated_codex_home  # the _FakeIsolatedHome the helper installed
    assert fake.entries == 1, "the isolated home is entered exactly once around the matrix"
    manifest = _json.loads((runs_dir / "prov-manifest.json").read_text())
    for k, v in _STUB_PROVENANCE.items():
        assert manifest[k] == v
    assert manifest["codex_home_path"] == fake.home
    assert manifest["artifacts_sha"]
    assert manifest["run_head_shas"] == [] and manifest["run_head_missing"] == 2
    assert manifest["run_dirty_any"] is False
    assert manifest["run_marker_mismatch"] == 2 and manifest["run_not_isolated"] == 2


def _ok_results(usages):
    from eval_core.models import RunResult

    return [
        RunResult(case_id="c", config_id="A", repeat_idx=i, usage=u, run_id=f"k{i}")
        for i, u in enumerate(usages)
    ]


def test_provenance_aggregates_precedence():
    run_eval = _run_eval()
    prov = {"repo_branch": "trunk", "repo_common_dir": "/r/.git"}
    base = {
        "repo_head_sha": "h1",
        "repo_dirty": False,
        "repo_branch": "trunk",
        "repo_common_dir": "/r/.git",
        "codex_home_isolated": True,
        "repo_ref_names": ["trunk"],
        "repo_worktree_paths": [],
    }
    # (i) full provenance collects.
    agg = run_eval._provenance_aggregates(_ok_results([base, base]), prov)
    assert agg == {
        "run_head_shas": ["h1"],
        "run_head_missing": 0,
        "run_dirty_any": False,
        "run_marker_mismatch": 0,
        "run_not_isolated": 0,
    }
    # (ii) legacy prompt_sha-only stubs.
    agg = run_eval._provenance_aggregates(_ok_results([{"prompt_sha": "p"}] * 3), prov)
    assert agg["run_head_shas"] == [] and agg["run_head_missing"] == 3
    assert agg["run_dirty_any"] is False
    # (iii) head sha + repo_dirty None -> None.
    assert (
        run_eval._provenance_aggregates(_ok_results([{**base, "repo_dirty": None}]), prov)[
            "run_dirty_any"
        ]
        is None
    )
    # (iv) any True -> True.
    assert (
        run_eval._provenance_aggregates(_ok_results([base, {**base, "repo_dirty": True}]), prov)[
            "run_dirty_any"
        ]
        is True
    )
    # (v)/(vi) marker mismatch: a different or empty branch counts.
    assert (
        run_eval._provenance_aggregates(
            _ok_results([base, {**base, "repo_branch": "other"}]), prov
        )["run_marker_mismatch"]
        == 1
    )
    assert (
        run_eval._provenance_aggregates(_ok_results([base, {**base, "repo_branch": ""}]), prov)[
            "run_marker_mismatch"
        ]
        == 1
    )
    # not isolated: flag false, or None ref/worktree lists.
    assert (
        run_eval._provenance_aggregates(
            _ok_results([{**base, "codex_home_isolated": False}]), prov
        )["run_not_isolated"]
        == 1
    )
    assert (
        run_eval._provenance_aggregates(_ok_results([{**base, "repo_ref_names": None}]), prov)[
            "run_not_isolated"
        ]
        == 1
    )
    assert (
        run_eval._provenance_aggregates(_ok_results([{**base, "repo_worktree_paths": None}]), prov)[
            "run_not_isolated"
        ]
        == 1
    )
    # (vii) a manifest whose own repo_branch is "" -> the Campaign gate expression prints BAD.
    m = {
        **prov,
        "repo_branch": "",
        "run_head_missing": 0,
        "run_dirty_any": False,
        "run_head_shas": ["h"],
        "repo_head_sha": "h",
        "repo_dirty": False,
        "run_marker_mismatch": 0,
        "run_not_isolated": 0,
        "repo_tree_sha": "t",
        "repo_ref_names": [],
        "repo_worktree_paths": [],
    }
    ok = (
        m.get("run_head_missing") == 0
        and m.get("run_dirty_any") is False
        and (m.get("run_head_shas") or []) == [m.get("repo_head_sha")]
        and m.get("repo_dirty") is False
        and m.get("run_marker_mismatch") == 0
        and m.get("run_not_isolated") == 0
        and bool(m.get("repo_branch"))
        and bool(m.get("repo_common_dir"))
        and bool(m.get("repo_tree_sha"))
        and isinstance(m.get("repo_ref_names"), list)
        and isinstance(m.get("repo_worktree_paths"), list)
    )
    assert ok is False


def test_cmd_run_manifest_carries_real_write_time_provenance(tmp_path, monkeypatch):
    """With the REAL _manifest_provenance (not the stub) and find_repo_root pointed
    at a tmp git repo, the manifest's write-time fields equal git_provenance(tmp)."""
    import json as _json

    from adapters import worktree

    run_eval = _run_eval()
    repo = _tmp_repo(tmp_path)
    fake = _FakeIsolatedHome(tmp_path)
    monkeypatch.setattr(run_eval, "isolated_codex_home", fake)
    monkeypatch.setattr(run_eval, "_assert_no_hidden_inputs", lambda repo_root: None)
    monkeypatch.setattr(run_eval, "find_repo_root", lambda: str(repo))
    runs_dir = _run_ok_experiment_no_stub(tmp_path, monkeypatch, subdir="real")
    manifest = _json.loads((runs_dir / "real-manifest.json").read_text())
    live = worktree.git_provenance(str(repo))
    for k in ("repo_head_sha", "repo_tree_sha", "repo_dirty", "repo_branch", "repo_ref_names"):
        assert manifest[k] == live[k], k


def _run_ok_experiment_no_stub(tmp_path, monkeypatch, subdir):
    """_run_ok_experiment minus _stub_run_git (callers stub what they need)."""
    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case, ReviewOutput

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    class _Ok:
        def cli_version(self):
            return _pinned_cli_version()

        def experiment_tag(self, config):
            return f"tag-{config.id}"

        def case_tag(self, case):
            return "ctag"

        def prompt_sha_for(self, case):
            return "psha"

        def review(self, case, config, repeat_idx):
            return ReviewOutput(
                review_markdown=f"As {config.model}, I see no issues.",
                cli_version=_pinned_cli_version(),
                latency_s=0.0,
                usage={"prompt_sha": "psha"},
            )

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Ok(), raising=True)
    rc = run_eval.cmd_run(
        _ns(configs="A,B", strata=["s1_synthetic"], subdir=subdir, k=1, max_parallel=1)
    )
    assert rc == 0
    return tmp_path / "runs"


def test_cmd_run_refuses_hidden_corpus_input_before_reviewer(tmp_path, monkeypatch):
    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case

    repo = _tmp_repo(tmp_path)
    hidden = repo / "tools" / "reviewer-eval" / "corpus" / "cases" / "s9_x" / "extra"
    hidden.mkdir(parents=True)
    (hidden / "case.json").write_text("{}")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("tools/reviewer-eval/corpus/cases/s9_x/\n")
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    monkeypatch.setattr(run_eval, "find_repo_root", lambda: str(repo))
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(run_eval, "isolated_codex_home", _FakeIsolatedHome(tmp_path))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    def _never(repo_root):
        raise AssertionError("_build_reviewer must not be reached")

    monkeypatch.setattr(run_eval, "_build_reviewer", _never, raising=True)
    with pytest.raises(SystemExit, match="IGNORED FILE"):
        run_eval.cmd_run(_ns(configs="A,B", strata=None, subdir="h", k=1, max_parallel=1))


def test_assert_tree_identity_guard(tmp_path, monkeypatch):
    run_eval = _run_eval()

    repo = _tmp_repo(tmp_path)
    manifest = {
        "repo_head_sha": _git(repo, "rev-parse", "HEAD"),
        "repo_tree_sha": _git(repo, "rev-parse", "HEAD^{tree}"),
    }
    run_eval._assert_tree_identity(str(repo), manifest)  # clean: passes
    # Bytecode FILES under tools/eval_core do not trip it ...
    pyc = repo / "tools" / "eval_core" / "__pycache__"
    pyc.mkdir(parents=True)
    (pyc / "x.cpython-314.pyc").write_bytes(b"\x00")
    (repo / ".DS_Store").write_bytes(b"\x00")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text(".DS_Store\n")
    run_eval._assert_tree_identity(str(repo), manifest)
    # ... but an excluded corpus file does, and so does an excluded __pycache__/case.json dir.
    extra = repo / "tools" / "reviewer-eval" / "corpus"
    extra.mkdir(parents=True)
    (extra / "extra.json").write_text("{}")
    (repo / ".git" / "info" / "exclude").write_text(
        ".DS_Store\ntools/reviewer-eval/corpus/extra.json\n"
    )
    with pytest.raises(SystemExit, match="IGNORED FILE"):
        run_eval._assert_tree_identity(str(repo), manifest)
    (extra / "extra.json").unlink()
    cache_case = repo / "tools" / "reviewer-eval" / "corpus" / "cases" / "s1" / "__pycache__"
    cache_case.mkdir(parents=True)
    (cache_case / "case.json").write_text("{}")
    with pytest.raises(SystemExit, match="IGNORED FILE"):
        run_eval._assert_tree_identity(str(repo), manifest)
    import shutil

    shutil.rmtree(repo / "tools" / "reviewer-eval")
    run_eval._assert_tree_identity(str(repo), manifest)
    # Dirty tree.
    (repo / "stray.txt").write_text("x")
    with pytest.raises(SystemExit, match="not clean"):
        run_eval._assert_tree_identity(str(repo), manifest)
    (repo / "stray.txt").unlink()
    # Tree-sha mismatch.
    with pytest.raises(SystemExit, match="repo_tree_sha"):
        run_eval._assert_tree_identity(str(repo), {**manifest, "repo_tree_sha": "0" * 40})
    # Same tree, amended commit -> HEAD mismatch.
    _git(repo, "commit", "-q", "--amend", "--no-verify", "-m", "amended")
    assert _git(repo, "rev-parse", "HEAD^{tree}") == manifest["repo_tree_sha"]
    with pytest.raises(SystemExit, match="repo_head_sha"):
        run_eval._assert_tree_identity(str(repo), manifest)


def _sealed_experiment_in_tmp_repo(tmp_path, monkeypatch, subdir="seal"):
    """A real cmd_run (stubbed reviewer) whose manifest's HEAD/tree are then
    REWRITTEN to a clean tmp git repo, so the tree-identity guard can pass there."""
    import json as _json

    run_eval = _run_eval()
    repo = _tmp_repo(tmp_path)
    monkeypatch.setattr(run_eval, "find_repo_root", lambda: str(repo))
    runs_dir = _run_ok_experiment(tmp_path, monkeypatch, subdir=subdir)
    mpath = runs_dir / f"{subdir}-manifest.json"
    manifest = _json.loads(mpath.read_text())
    manifest["repo_head_sha"] = _git(repo, "rev-parse", "HEAD")
    manifest["repo_tree_sha"] = _git(repo, "rev-parse", "HEAD^{tree}")
    mpath.write_text(_json.dumps(manifest))
    monkeypatch.setattr(run_eval, "_assert_no_hidden_inputs", lambda repo_root: None)
    return repo, runs_dir, manifest


def test_compare_blinded_refuses_edited_artifact_and_checks_identity_at_exit(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    repo, runs_dir, manifest = _sealed_experiment_in_tmp_repo(tmp_path, monkeypatch)
    # Restore the REAL guard (the ok-experiment helper no-ops it).
    monkeypatch.setattr(
        run_eval, "_assert_tree_identity", _run_eval().__dict__["_assert_tree_identity"]
    )
    assert run_eval.cmd_compare(_ns(subdir="seal", allow_mixed=False, blinded=True)) == 0
    blinding = _json.loads((runs_dir / "seal" / "blinding.json").read_text())
    assert blinding["artifacts_sha"] == manifest["artifacts_sha"]
    assert blinding["run_ids_sha"] == run_eval._run_ids_sha(manifest["run_ids"])
    # Edit one artifact's session_commands after run -> refused with both hashes.
    art = next(p for p in (runs_dir / "seal").glob("*.json") if p.name != "blinding.json")
    d = _json.loads(art.read_text())
    d["usage"]["session_commands"] = ["/bin/zsh -lc 'git log --all' in /x"]
    art.write_text(_json.dumps(d))
    assert run_eval.cmd_compare(_ns(subdir="seal", allow_mixed=False, blinded=True)) == 1
    art.write_text(_json.dumps(_json.loads(art.read_text())))  # leave edited; next tests re-run


def test_stage_exit_identity_recheck_catches_mid_stage_edit(tmp_path, monkeypatch):
    run_eval = _run_eval()
    repo, runs_dir, manifest = _sealed_experiment_in_tmp_repo(
        tmp_path, monkeypatch, subdir="exitck"
    )
    real_guard = run_eval._assert_tree_identity

    def _dirtying_audit(runs, subdir_path, repo_root, markers, run_ids_sha, sealed):
        (repo / "mid-stage.txt").write_text("edit while the stage runs")
        return {
            "run_ids_sha": run_ids_sha,
            "artifacts_sha": sealed,
            "branch": "",
            "hits": [],
            "unaudited": [],
            "no_commands": [],
            "coverage": {"covered": 0, "no_commands": 0, "unaudited": 0, "total": 0},
            "vacuous": True,
        }

    monkeypatch.setattr(run_eval.reviewer_audit, "audit", _dirtying_audit)
    monkeypatch.setattr(run_eval, "_assert_tree_identity", real_guard)
    with pytest.raises(SystemExit, match="stage exit"):
        run_eval.cmd_audit(_ns(subdir="exitck", output=str(tmp_path / "hits.json")))


# --------------------------------------------------------------------------- #
# Local-review fixes: credential mode, candidate-exposure gate, stop conditions.
# --------------------------------------------------------------------------- #


def test_rotated_credential_saved_owner_only(tmp_path, monkeypatch):
    import stat

    from adapters import codex_reviewer as cr

    real = _fake_real_home(tmp_path, monkeypatch)
    with cr.isolated_codex_home() as home:
        (pathlib.Path(home) / "auth.json").write_text('{"token": "rotated"}')
    saved = sorted(real.glob("auth.json.campaign-*"))
    assert len(saved) == 1
    mode = stat.S_IMODE(saved[0].stat().st_mode)
    assert mode & 0o077 == 0, f"credential copy must be owner-only, got {oct(mode)}"


def _gate_setup(tmp_path, monkeypatch):
    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case

    _stub_run_git(monkeypatch, tmp_path)
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    def _never(repo_root):
        raise AssertionError("_build_reviewer must not be reached")

    monkeypatch.setattr(run_eval, "_build_reviewer", _never, raising=True)
    return run_eval


@pytest.mark.parametrize(
    "ns",
    [
        dict(configs="E", strata=None, subdir="x", k=2, max_parallel=1),
        dict(configs="B,E", strata=None, subdir="x", k=1, max_parallel=1),
        dict(configs="B,E", strata=["s1_synthetic"], subdir="x", k=2, max_parallel=1),
        dict(configs="B,E", strata=None, subdir="x", k=2, k_per="E=1", max_parallel=1),
        dict(configs="A,B,E", strata=None, subdir="x", k=2, max_parallel=1),
    ],
)
def test_run_refuses_candidate_exposure_outside_registered_campaign(
    tmp_path, monkeypatch, ns, capsys
):
    run_eval = _gate_setup(tmp_path, monkeypatch)
    assert run_eval.cmd_run(_ns(**ns)) == 1
    assert "pre-registration" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "x-attempts.jsonl").exists()


def test_smoke_refuses_candidate_arm(tmp_path, monkeypatch, capsys):
    run_eval = _gate_setup(tmp_path, monkeypatch)
    assert run_eval.cmd_smoke(_ns(configs="E", strata=None, k=1, limit=1, max_parallel=1)) == 1
    assert "may not be smoked" in capsys.readouterr().err


def test_registered_campaign_shape_passes_the_gate():
    run_eval = _run_eval()
    cfgs = run_eval._resolve_configs("B,E")
    assert run_eval._candidate_exposure_violations(cfgs, 2, {}, None, "run") == []
    assert (
        run_eval._candidate_exposure_violations(
            run_eval._resolve_configs("A,B"), 1, {}, ["s1_synthetic"], "run"
        )
        == []
    )


def test_stop_conditions_from_attempts_log(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    path = tmp_path / "runs" / "g-attempts.jsonl"

    def end(failures):
        return _json.dumps(
            {"kind": "end", "started_at": "t", "ok": 1, "total": 2, "failures": failures}
        )

    start = _json.dumps({"kind": "start", "head": "h", "at": "t", "configs": ["B", "E"], "k": 2})
    # 4 dispatched attempts -> the cap (a START without END does not count).
    path.write_text(
        "\n".join([start, end([]), start, end([]), start, end([]), start, end([]), start]) + "\n"
    )
    stops = run_eval._stop_condition_violations("g")
    assert len(stops) == 1 and "cap" in stops[0]
    # Two consecutive timeouts on the same cell -> stop; on different cells -> no stop.
    t = {"case": "c", "arm": "E", "repeat": 1, "infra_error": "codex exec timed out after 3600s"}
    path.write_text("\n".join([start, end([t]), start, end([t])]) + "\n")
    stops = run_eval._stop_condition_violations("g")
    assert len(stops) == 1 and "2 consecutive" in stops[0]
    t2 = {**t, "repeat": 0}
    path.write_text("\n".join([start, end([t]), start, end([t2])]) + "\n")
    assert run_eval._stop_condition_violations("g") == []
    # Non-timeout failures never trip the consecutive rule.
    f = {**t, "infra_error": "worktree add failed"}
    path.write_text("\n".join([start, end([f]), start, end([f])]) + "\n")
    assert run_eval._stop_condition_violations("g") == []


def test_cmd_run_appends_attempt_records_and_refuses_at_cap(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    _run_ok_experiment(tmp_path, monkeypatch, subdir="cap")  # the one real dispatch
    lines = [
        _json.loads(line)
        for line in (tmp_path / "runs" / "cap-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in lines] == ["start", "end"]
    assert lines[0]["head"] == _STUB_PROVENANCE["repo_head_sha"] and lines[1]["failures"] == []
    assert len(lines[1]["ok_cells"]) == 2 and all(c["sha256"] for c in lines[1]["ok_cells"])
    # Three more dispatched attempts (synthetic END records that RESUMED every OK cell).
    for _ in range(3):
        run_eval._append_attempt("cap", {**lines[0], "at": "later"})
        run_eval._append_attempt("cap", lines[1])
    manifest_before = (tmp_path / "runs" / "cap-manifest.json").read_bytes()
    assert b'"failed"' not in manifest_before
    with pytest.raises(
        AssertionError
    ):  # the 5th attempt is refused (rc 1), the helper asserts rc == 0
        _run_ok_experiment(tmp_path, monkeypatch, subdir="cap")
    assert (
        len((tmp_path / "runs" / "cap-attempts.jsonl").read_text().splitlines()) == 8
    ), "refused attempt appends nothing"
    assert (
        tmp_path / "runs" / "cap-manifest.json"
    ).read_bytes() == manifest_before, "a refused attempt leaves the last valid manifest intact"


def test_aborted_attempt_after_dispatch_counts_toward_cap(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case
    from eval_core.runner import ConfoundMismatch

    _stub_run_git(monkeypatch, tmp_path)
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)
    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: object(), raising=True)

    def _boom(*a, **k):
        k["on_dispatch"]()  # the runner has handed the first review to the pool
        raise RuntimeError("codex died mid-matrix")

    monkeypatch.setattr("eval_core.runner.run_matrix", _boom, raising=True)
    with pytest.raises(RuntimeError):
        run_eval.cmd_run(_ns(configs="A,B", strata=None, subdir="ab", k=1, max_parallel=1))
    recs = [
        _json.loads(x) for x in (tmp_path / "runs" / "ab-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == ["start", "end"] and recs[1]["aborted"] == "RuntimeError"
    assert _json.loads((tmp_path / "runs" / "ab-manifest.json").read_text())["failed"] is True

    def _pre(*a, **k):
        raise ConfoundMismatch("arms differ in sandbox")

    def _planning(*a, **k):
        raise RuntimeError("experiment_tag failed for config B")  # planning: before dispatch

    for raiser, exc in ((_pre, ConfoundMismatch), (_planning, RuntimeError)):
        monkeypatch.setattr("eval_core.runner.run_matrix", raiser, raising=True)
        with pytest.raises(exc):
            run_eval.cmd_run(_ns(configs="A,B", strata=None, subdir="ab", k=1, max_parallel=1))
    recs = [
        _json.loads(x) for x in (tmp_path / "runs" / "ab-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == [
        "start",
        "end",
        "start",
        "start",
    ], "pre-dispatch raise (any type): no END record"
    # Three more aborted attempts hit the cap.
    monkeypatch.setattr("eval_core.runner.run_matrix", _boom, raising=True)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            run_eval.cmd_run(_ns(configs="A,B", strata=None, subdir="ab", k=1, max_parallel=1))
    assert run_eval.cmd_run(_ns(configs="A,B", strata=None, subdir="ab", k=1, max_parallel=1)) == 1


def test_cmd_run_preflight_refuses_dirty_or_incomplete_provenance(tmp_path, monkeypatch, capsys):
    """Before the manifest is touched and before any codex call: a dirty tree or
    an incomplete git sample refuses the attempt (no attempt record, no reviewer)."""
    run_eval = _run_eval()
    _stub_run_git(monkeypatch, tmp_path)
    for bad, why in (
        ({**_STUB_PROVENANCE, "repo_ref_names": None}, "incomplete git provenance"),
        ({**_STUB_PROVENANCE, "repo_dirty": True}, "working tree is dirty"),
    ):
        monkeypatch.setattr(run_eval, "_manifest_provenance", lambda repo_root, bad=bad: dict(bad))
        monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))

        def _never(repo_root):
            raise AssertionError("_build_reviewer must not be reached")

        monkeypatch.setattr(run_eval, "_build_reviewer", _never, raising=True)
        rc = run_eval.cmd_run(_ns(configs="A,B", strata=None, subdir="pre", k=1, max_parallel=1))
        assert rc == 1 and why in capsys.readouterr().err
        assert not (tmp_path / "runs" / "pre-attempts.jsonl").exists()
        assert not (tmp_path / "runs" / "pre-manifest.json").exists()


def test_cmd_run_fails_closed_on_incomplete_write_time_provenance(tmp_path, monkeypatch):
    """The preflight sample is clean but the WRITE-TIME sample fails (a transient git
    error mid-run): failure marker, exit 2, resume reuses the OK artifacts."""
    import json as _json

    run_eval = _run_eval()
    _stub_run_git(monkeypatch, tmp_path)
    calls = {"n": 0}

    def _flaky(repo_root):
        calls["n"] += 1
        if calls["n"] == 1:
            return dict(_STUB_PROVENANCE)
        return {**_STUB_PROVENANCE, "repo_ref_names": None}

    monkeypatch.setattr(run_eval, "_manifest_provenance", _flaky)
    with pytest.raises(AssertionError):  # helper asserts rc == 0; the run returns 2
        _run_ok_experiment_no_stub(tmp_path, monkeypatch, subdir="incomplete")
    m = _json.loads((tmp_path / "runs" / "incomplete-manifest.json").read_text())
    assert m["failed"] is True and "repo_ref_names" in m["error"]
    assert run_eval._provenance_failures({**_STUB_PROVENANCE, "repo_worktree_paths": None}) == [
        "repo_worktree_paths"
    ]
    assert run_eval._provenance_failures({**_STUB_PROVENANCE, "repo_head_sha": ""}) == [
        "repo_head_sha"
    ]
    assert run_eval._provenance_failures({**_STUB_PROVENANCE, "repo_dirty": None}) == ["repo_dirty"]
    assert run_eval._provenance_failures(_STUB_PROVENANCE) == []


def test_run_matrix_calls_on_dispatch_once_before_first_review():
    import tempfile

    from eval_core.models import Config, ReviewOutput
    from eval_core.runner import run_matrix
    from eval_core.store import RunStore

    events = []

    class _Rev:
        def cli_version(self):
            return "codex-cli 0.162.1"

        def experiment_tag(self, config):
            return "t"

        def case_tag(self, case):
            return "c"

        def prompt_sha_for(self, case):
            return "p"

        def review(self, case, config, repeat_idx):
            events.append("review")
            return ReviewOutput(
                review_markdown="ok",
                cli_version="codex-cli 0.162.1",
                latency_s=0.0,
                usage={"prompt_sha": "p"},
            )

    cfg = Config(id="A", model="m", effort="xhigh", cli_version="codex-cli 0.162.1")
    with tempfile.TemporaryDirectory() as d:
        store = RunStore(d)
        run_matrix(
            [_case()],
            [cfg],
            _Rev(),
            store,
            k=2,
            max_parallel=1,
            on_dispatch=lambda: events.append("dispatch"),
        )
        assert events[0] == "dispatch" and events.count("dispatch") == 1
        assert events.count("review") == 2
        events.clear()
        run_matrix(
            [_case()],
            [cfg],
            _Rev(),
            store,
            k=2,
            max_parallel=1,
            on_dispatch=lambda: events.append("dispatch"),
        )
        assert events == [], "a fully cached matrix dispatches nothing"


def test_cmd_run_appends_campaign_ledger_only_for_registered_contrast(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    _stub_run_git(monkeypatch, tmp_path)
    # A,B is not a registered contrast: no ledger record.
    _run_ok_experiment(tmp_path, monkeypatch, subdir="ab")
    assert not (tmp_path / "runs" / "campaign-ledger.jsonl").exists()
    # B,E k=2 IS the registered campaign: one dispatch record on the preflight head.
    from eval_core.models import STRATUM_SYNTHETIC, Case, ReviewOutput

    class _Ok:
        def cli_version(self):
            return _pinned_cli_version()

        def experiment_tag(self, config):
            return f"tag-{config.id}"

        def case_tag(self, case):
            return "ctag"

        def prompt_sha_for(self, case):
            return "psha"

        def review(self, case, config, repeat_idx):
            return ReviewOutput(
                review_markdown="x",
                cli_version=_pinned_cli_version(),
                latency_s=0.0,
                usage={"prompt_sha": "psha"},
            )

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Ok(), raising=True)
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    assert (
        run_eval.cmd_run(_ns(configs="B,E", strata=None, subdir="camp", k=2, max_parallel=1)) == 0
    )
    recs = [
        _json.loads(x)
        for x in (tmp_path / "runs" / "campaign-ledger.jsonl").read_text().splitlines()
    ]
    assert recs == [
        {
            "at": recs[0]["at"],
            "contrast": ["B", "E"],
            "head": "feedface",
            "kind": "dispatch",
            "ledger_reset": False,
            "subdir": "camp",
            "tree": "cafebabe",
        }
    ]


def test_cached_resume_writes_no_end_record(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    _run_ok_experiment(tmp_path, monkeypatch, subdir="cache")  # dispatches
    _run_ok_experiment(tmp_path, monkeypatch, subdir="cache")  # fully cached resume
    recs = [
        _json.loads(x)
        for x in (tmp_path / "runs" / "cache-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == [
        "start",
        "end",
        "start",
    ], "a cached resume is not an attempt"
    assert not run_eval._stop_condition_violations("cache")


def test_ledger_reset_is_recorded_when_frozen_ref_exists(tmp_path, monkeypatch):
    import json as _json

    run_eval = _run_eval()
    fake = _stub_run_git(monkeypatch, tmp_path)
    fake.refs[("B", "E")] = "feedface"  # the ref survived; the ledger did not
    from eval_core.models import STRATUM_SYNTHETIC, Case, ReviewOutput

    class _Ok:
        def cli_version(self):
            return _pinned_cli_version()

        def experiment_tag(self, config):
            return f"tag-{config.id}"

        def case_tag(self, case):
            return "ctag"

        def prompt_sha_for(self, case):
            return "psha"

        def review(self, case, config, repeat_idx):
            return ReviewOutput(
                review_markdown="x",
                cli_version=_pinned_cli_version(),
                latency_s=0.0,
                usage={"prompt_sha": "psha"},
            )

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Ok(), raising=True)
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)
    assert (
        run_eval.cmd_run(_ns(configs="B,E", strata=None, subdir="again", k=2, max_parallel=1)) == 0
    )
    recs = [
        _json.loads(x)
        for x in (tmp_path / "runs" / "campaign-ledger.jsonl").read_text().splitlines()
    ]
    assert recs[0]["ledger_reset"] is True and recs[0]["head"] == "feedface"


def test_frozen_ref_helpers_on_tmp_repo(tmp_path):
    run_eval = _run_eval()
    repo = _tmp_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    assert run_eval._read_frozen_ref(str(repo), ("B", "E")) is None
    assert run_eval._anchor_frozen_ref(str(repo), ("B", "E"), head) == head
    assert run_eval._read_frozen_ref(str(repo), ("B", "E")) == head
    assert _git(repo, "rev-parse", "refs/campaign/B-E/frozen") == head
    # A later anchor call never moves the ref.
    _git(repo, "commit", "-q", "--allow-empty", "--no-verify", "-m", "later")
    assert (
        run_eval._anchor_frozen_ref(str(repo), ("B", "E"), _git(repo, "rev-parse", "HEAD")) == head
    )


def test_cmd_run_refuses_a_concurrent_attempt_into_the_same_subdir(tmp_path, monkeypatch, capsys):
    """Two processes with the same --subdir compute identical run ids and would
    interleave artifacts into one store; the second is refused at the per-subdir
    lock, before the attempts log or the manifest is touched. The holder here is
    a REAL second process; once it exits, the same attempt proceeds."""
    import subprocess
    import sys as _sys

    run_eval = _run_eval()
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(runs))
    lock = run_eval._lock_path("lk")
    assert lock == str(runs / "lk.lock")
    holder = subprocess.Popen(
        [
            _sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            "fh = open(sys.argv[1], 'a')\n"
            "fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "print('held', flush=True)\n"
            "time.sleep(120)\n",
            lock,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(AssertionError):  # helper asserts rc == 0; the run returns 1
            _run_ok_experiment(tmp_path, monkeypatch, subdir="lk")
        err = capsys.readouterr().err
        assert "concurrent run" in err and "lk.lock" in err
        assert not (runs / "lk-attempts.jsonl").exists()
        assert not (runs / "lk-manifest.json").exists()
        assert not (runs / "lk").exists() or not list((runs / "lk").glob("*.json"))
    finally:
        holder.kill()
        holder.wait()
    # The holder is gone: the same attempt proceeds and owns every artifact.
    _run_ok_experiment(tmp_path, monkeypatch, subdir="lk")
    assert len(list((runs / "lk").glob("*.json"))) == 2
    # In-process: a held lock yields False to a second acquirer, True after release.
    with run_eval._subdir_lock("lk") as first:
        assert first is True
        with run_eval._subdir_lock("lk") as second:
            assert second is False
    with run_eval._subdir_lock("lk") as again:
        assert again is True


def test_aborted_attempt_records_the_cells_it_persisted_and_resume_protects_them(
    tmp_path, monkeypatch, capsys
):
    """An attempt whose worker raises immediately AFTER its artifact landed on disk
    (before any bookkeeping) aborts; the aborted END record lists every cell on
    disk (that one AND the sibling the pool finished while shutting down);
    deleting or editing one before the rerun is refused as re-sampling, and a
    rerun that preserves them resumes both without a dispatch."""
    import json as _json

    run_eval = _run_eval()
    real_store = run_eval.RunStore
    calls = {"n": 0}

    class _DiesRightAfterFirstSave(real_store):
        # The narrowest window: the artifact's os.replace has landed, nothing
        # after it ran. The journal must still see it.
        def save(self, key, result):
            calls["n"] += 1
            super().save(key, result)
            if calls["n"] == 1:
                raise RuntimeError("interrupted mid-matrix")

    monkeypatch.setattr(run_eval, "RunStore", _DiesRightAfterFirstSave)
    with pytest.raises(RuntimeError, match="interrupted"):
        _run_ok_experiment(tmp_path, monkeypatch, subdir="ab2")
    recs = [
        _json.loads(x) for x in (tmp_path / "runs" / "ab2-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == ["start", "end"] and recs[1]["aborted"] == "RuntimeError"
    arts = sorted((tmp_path / "runs" / "ab2").glob("*.json"))
    assert len(arts) == 2, "the raising save landed its artifact; the pool finished the sibling"
    assert sorted(c["run_id"] for c in recs[1]["ok_cells"]) == [a.stem for a in arts]
    assert all(c["sha256"] for c in recs[1]["ok_cells"])
    monkeypatch.setattr(run_eval, "RunStore", real_store)
    saved = arts[0].read_bytes()
    arts[0].unlink()  # a persisted cell deleted before the rerun: refuse
    with pytest.raises(AssertionError):
        _run_ok_experiment(tmp_path, monkeypatch, subdir="ab2")
    assert "resume integrity" in capsys.readouterr().err
    d = _json.loads(saved)
    d["review_markdown"] = "edited"
    arts[0].write_text(_json.dumps(d))  # edited: refuse too
    with pytest.raises(AssertionError):
        _run_ok_experiment(tmp_path, monkeypatch, subdir="ab2")
    assert "artifact changed" in capsys.readouterr().err
    arts[0].write_bytes(saved)
    # Preserved: the rerun resumes both cells (fully cached: no dispatch, no END).
    _run_ok_experiment(tmp_path, monkeypatch, subdir="ab2")
    recs = [
        _json.loads(x) for x in (tmp_path / "runs" / "ab2-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == ["start", "end", "start"]
    assert arts[0].read_bytes() == saved


def test_coordinator_interruption_keeps_every_finished_review(tmp_path, monkeypatch):
    """Two workers finish; the coordinator is interrupted while consuming the FIRST
    future (the `done:` progress line raises). Both reviews were persisted by
    their workers, so the aborted END record lists both and the rerun resumes
    both without a dispatch."""
    import json as _json

    run_eval = _run_eval()
    from eval_core.models import STRATUM_SYNTHETIC, Case, ReviewOutput

    _stub_run_git(monkeypatch, tmp_path)
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: [Case(id="c", stratum=STRATUM_SYNTHETIC)],
        raising=True,
    )
    monkeypatch.setattr(run_eval.CorpusLoader, "verify", lambda self, case: None, raising=True)

    class _Ok:
        def cli_version(self):
            return _pinned_cli_version()

        def experiment_tag(self, config):
            return f"tag-{config.id}"

        def case_tag(self, case):
            return "ctag"

        def prompt_sha_for(self, case):
            return "psha"

        def review(self, case, config, repeat_idx):
            return ReviewOutput(
                review_markdown="fine",
                cli_version=_pinned_cli_version(),
                latency_s=0.0,
                usage={"prompt_sha": "psha"},
            )

    monkeypatch.setattr(run_eval, "_build_reviewer", lambda repo_root: _Ok(), raising=True)
    real_print = print

    def _interrupting_print(*a, **k):
        if a and "done:" in str(a[0]):
            raise KeyboardInterrupt("operator hit ^C while the coordinator consumed a future")
        real_print(*a, **k)

    monkeypatch.setattr(run_eval, "print", _interrupting_print, raising=False)
    with pytest.raises(KeyboardInterrupt):
        run_eval.cmd_run(_ns(configs="A,B", strata=None, subdir="ki", k=1, max_parallel=2))
    monkeypatch.setattr(run_eval, "print", real_print, raising=False)
    arts = sorted((tmp_path / "runs" / "ki").glob("*.json"))
    assert len(arts) == 2, "both workers' reviews are on disk"
    recs = [
        _json.loads(x) for x in (tmp_path / "runs" / "ki-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == ["start", "end"]
    assert recs[1]["aborted"] == "KeyboardInterrupt"
    assert sorted(c["run_id"] for c in recs[1]["ok_cells"]) == [a.stem for a in arts]
    # The rerun resumes both (fully cached: no dispatch, no END record) and leaves
    # the artifacts byte-identical.
    before = [a.read_bytes() for a in arts]
    assert run_eval.cmd_run(_ns(configs="A,B", strata=None, subdir="ki", k=1, max_parallel=2)) == 0
    assert [a.read_bytes() for a in arts] == before
    recs = [
        _json.loads(x) for x in (tmp_path / "runs" / "ki-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == ["start", "end", "start"]


def test_resume_integrity_protects_the_cumulative_ok_cell_set(tmp_path, monkeypatch):
    """A cell OK in an EARLIER END record stays protected when a later (aborted)
    END record lists only the cells it persisted itself."""
    import hashlib as _hashlib
    import json as _json

    run_eval = _run_eval()
    runs = tmp_path / "runs"
    (runs / "cum").mkdir(parents=True)
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(runs))
    x = runs / "cum" / "c.A.r0.x.json"
    y = runs / "cum" / "c.B.r0.y.json"
    x.write_text("{}")
    y.write_text("{}")

    def cell(path, arm):
        return {
            "case": "c",
            "arm": arm,
            "repeat": 0,
            "run_id": path.stem,
            "sha256": _hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    start = {"kind": "start", "head": "h", "at": "t", "configs": ["A", "B"], "k": 1}
    end1 = {
        "kind": "end",
        "started_at": "t",
        "ok": 1,
        "total": 2,
        "failures": [],
        "ok_cells": [cell(x, "A")],
    }
    end2 = {
        "kind": "end",
        "started_at": "t2",
        "aborted": "RuntimeError",
        "ok": None,
        "total": None,
        "failures": [],
        "ok_cells": [cell(y, "B")],
    }
    (runs / "cum-attempts.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in (start, end1, start, end2)) + "\n"
    )
    assert run_eval._resume_integrity_violations("cum", str(runs / "cum")) == []
    x.unlink()  # listed only by the FIRST END record
    probs = run_eval._resume_integrity_violations("cum", str(runs / "cum"))
    assert len(probs) == 1 and "c/A/r0" in probs[0] and "missing" in probs[0]


def test_rerun_refuses_when_an_ok_artifact_is_missing_or_changed(tmp_path, monkeypatch, capsys):
    import json as _json

    _run_ok_experiment(tmp_path, monkeypatch, subdir="res")
    art = sorted((tmp_path / "runs" / "res").glob("*.json"))[0]
    saved = art.read_bytes()
    art.unlink()  # a deleted OK artifact would be RE-DISPATCHED: refuse
    with pytest.raises(AssertionError):  # helper asserts rc == 0; the run returns 1
        _run_ok_experiment(tmp_path, monkeypatch, subdir="res")
    assert "resume integrity" in capsys.readouterr().err
    art.write_bytes(saved)
    d = _json.loads(saved)
    d["review_markdown"] = "edited"
    art.write_text(_json.dumps(d))  # a changed OK artifact: refuse too
    with pytest.raises(AssertionError):
        _run_ok_experiment(tmp_path, monkeypatch, subdir="res")
    assert "artifact changed" in capsys.readouterr().err
    art.write_bytes(saved)
    # A genuine infra-only resume that preserves every OK artifact proceeds (cached, no END).
    _run_ok_experiment(tmp_path, monkeypatch, subdir="res")
    recs = [
        _json.loads(x) for x in (tmp_path / "runs" / "res-attempts.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in recs] == ["start", "end", "start"]
