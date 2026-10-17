"""Tests for tools/reviewer-eval/reviewer_verdict.py and the `verdict` CLI.

Isolation pattern copied from test_evals_runtime.py (module-level skip, guarded
sys.path insert, no top-level import of the module under test).
"""

import functools
import importlib.util
import json
import pathlib
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
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))


@functools.lru_cache(maxsize=1)
def _v():
    import reviewer_verdict

    return reviewer_verdict


@functools.lru_cache(maxsize=1)
def _run_eval():
    spec = importlib.util.spec_from_file_location(
        "reviewer_eval_run_eval", _EVAL_ROOT / "run_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Synthetic k=2 tables
# --------------------------------------------------------------------------- #


def _run(
    case, arm, rep, stratum="s1_synthetic", bugs=("b1",), allow=("P3",), must=True, infra=None
):
    return {
        "case_id": case,
        "config_id": arm,
        "repeat_idx": rep,
        "infra_error": infra,
        "run_id": f"{case}.{arm}.r{rep}",
        "case_snapshot": {
            "stratum": stratum,
            "ground_truth": [{"id": b, "must_catch": must} for b in bugs],
            "allow_severities": list(allow),
        },
    }


def _runs(cases=(("c1", ("b1",)),), s3=("s3",), k=2, arms=("B", "E")):
    out = []
    for case, bugs in cases:
        for a in arms:
            for r in range(k):
                out.append(_run(case, a, r, bugs=bugs))
    for case in s3:
        for a in arms:
            for r in range(k):
                out.append(_run(case, a, r, stratum="s3_negative", bugs=()))
    return out


def _row(case, bug, arm, rep, verd, leak="none", flag="none", ev="", notes=""):
    return {
        "case": case,
        "bug_id": bug,
        "arm": arm,
        "repeat": rep,
        "verdict": verd,
        "evidence": "quote" if verd == "caught" else "",
        "leak": leak,
        "grader_flag": flag,
        "leak_evidence": ev,
        "notes": notes,
    }


def _neg(case, arm, rep, findings=(), leak="none", flag="none", ev="", notes=""):
    return {
        "case": case,
        "arm": arm,
        "repeat": rep,
        "findings": list(findings),
        "leak": leak,
        "grader_flag": flag,
        "leak_evidence": ev,
        "notes": notes,
    }


def _table(
    b=("caught", "caught"),
    e=("caught", "caught"),
    case="c1",
    bug="b1",
    s3=("s3",),
    b_fp=(),
    e_fp=(),
):
    rows = [_row(case, bug, "B", i, v) for i, v in enumerate(b)] + [
        _row(case, bug, "E", i, v) for i, v in enumerate(e)
    ]
    negs = []
    for c in s3:
        for r in (0, 1):
            negs.append(_neg(c, "B", r, b_fp if r == 0 else ()))
            negs.append(_neg(c, "E", r, e_fp if r == 0 else ()))
    return {"rows": rows, "negative_assessments": negs}


def _gates(table, runs=None, audit=None):
    return _v().gates(table, runs or _runs(), "B", "E", audit)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def test_labels_and_precedence():
    v = _v()
    assert _gates(_table(b=("caught", "caught"), e=("missed", "missed")))["verdict"] == v.NO_GO
    # Regression on a must_catch=false bug is still NO-GO.
    runs = [dict(r) for r in _runs()]
    for r in runs:
        if r["case_id"] == "c1":
            r["case_snapshot"] = {
                **r["case_snapshot"],
                "ground_truth": [{"id": "b1", "must_catch": False}],
            }
    assert _v().gates(_table(e=("missed", "missed")), runs, "B", "E")["verdict"] == v.NO_GO
    # partial == missed: B partial both + E caught both -> improvement -> GO.
    assert _gates(_table(b=("partial", "partial")))["verdict"] == v.GO
    # B caught both + E partial both -> regression.
    assert _gates(_table(e=("partial", "partial")))["verdict"] == v.NO_GO
    # B caught both + E caught/partial -> one downgrade -> PARITY (no improvement).
    r = _gates(_table(e=("caught", "partial")))
    assert r["verdict"] == v.PARITY and r["downgrades"] == [{"case": "c1", "bug_id": "b1"}]
    # FP excess -> NO-GO; P3 on an allow-P3 case is not an FP, P2 is.
    assert _gates(_table(e_fp=[{"severity": "P3", "summary": "x"}]))["verdict"] == v.PARITY
    assert _gates(_table(e_fp=[{"severity": "P2", "summary": "x"}]))["verdict"] == v.NO_GO
    # GO via fewer FPs.
    assert _gates(_table(b_fp=[{"severity": "P1", "summary": "x"}]))["verdict"] == v.GO
    # 2 downgrades + 1 improvement -> PARITY; regression + 2 downgrades -> NO-GO.
    runs3 = _runs(cases=(("c1", ("b1",)), ("c2", ("b1",)), ("c3", ("b1",))))
    t = {"rows": [], "negative_assessments": _table()["negative_assessments"]}
    t["rows"] += [_row("c1", "b1", "B", i, "caught") for i in (0, 1)] + [
        _row("c1", "b1", "E", 0, "caught"),
        _row("c1", "b1", "E", 1, "missed"),
    ]
    t["rows"] += [_row("c2", "b1", "B", i, "caught") for i in (0, 1)] + [
        _row("c2", "b1", "E", 0, "caught"),
        _row("c2", "b1", "E", 1, "missed"),
    ]
    t["rows"] += [_row("c3", "b1", "B", i, "missed") for i in (0, 1)] + [
        _row("c3", "b1", "E", i, "caught") for i in (0, 1)
    ]
    res = v.gates(t, runs3, "B", "E")
    assert (
        res["verdict"] == v.PARITY and len(res["downgrades"]) == 2 and len(res["improvements"]) == 1
    )
    # Turn c3 into a regression (B caught both, E missed both): regression + 2 downgrades -> NO-GO.
    t["rows"] = [r for r in t["rows"] if r["case"] != "c3"]
    t["rows"] += [_row("c3", "b1", "B", i, "caught") for i in (0, 1)] + [
        _row("c3", "b1", "E", i, "missed") for i in (0, 1)
    ]
    assert v.gates(t, runs3, "B", "E")["verdict"] == v.NO_GO


def test_judgment_flag_b_unstable_e_missed():
    r = _gates(_table(b=("caught", "missed"), e=("missed", "missed")))
    assert r["judgment_flags"] == [{"bug": "c1|b1", "kind": "b_unstable_e_missed"}]
    assert r["verdict"] == _v().PARITY
    assert _gates(_table(b=("caught", "caught"), e=("missed", "missed")))["judgment_flags"] == []


def test_undetermined_on_infra_short_exposure():
    runs = _runs()
    runs[1] = _run("c1", "B", 1, infra="boom")
    t = _table(b=("caught",))
    t["rows"] = [r for r in t["rows"] if not (r["arm"] == "B" and r["repeat"] == 1)]
    assert _v().gates(t, runs, "B", "E")["verdict"] == _v().UNDETERMINED


def test_void_cells_contest_and_statuses():
    v = _v()
    # One void B repeat on a bug case -> B at most UNSTABLE + contested, not UNDETERMINED.
    t = _table(b=("caught", "caught"), e=("missed", "missed"))
    t["rows"][0]["leak"] = "confirmed"
    t["rows"][0]["leak_evidence"] = "git log --all"
    r = _gates(t)
    assert r["catch_status"]["c1|b1|B"] == v.UNSTABLE
    assert r["verdict"] != v.UNDETERMINED
    assert sorted(r["contested_gates"]) == ["downgrade", "improvement", "regression"]
    assert r["voided_cells"] == [{"case": "c1", "arm": "B", "repeat": 0}]
    # Both repeats void -> UNGRADEABLE + contested.
    t["rows"][1]["leak"] = "confirmed"
    t["rows"][1]["leak_evidence"] = "git log --all"
    r = _gates(t)
    assert r["catch_status"]["c1|b1|B"] == v.UNGRADEABLE
    assert any("ungradeable" in c["reason"] for c in r["contested"])
    # Confirmed leak on an S3 repeat -> its FP list dropped + gate 2 contested.
    t = _table(e_fp=[{"severity": "P1", "summary": "bad"}])
    cell = next(c for c in t["negative_assessments"] if c["arm"] == "E" and c["repeat"] == 0)
    cell["leak"] = "confirmed"
    cell["leak_evidence"] = "git show main:x"
    r = _gates(t)
    assert r["fp_candidate"] == 0 and "fp" in r["contested_gates"]
    assert r["informational"]["E"]["leaks"] == 1
    # leak confirmed with grader_flag none + evidence -> accepted and voids.
    t = _table(b=("caught", "caught"), e=("caught", "caught"))
    t["rows"][2]["leak"] = "confirmed"
    t["rows"][2]["leak_evidence"] = "cites the commit message"
    assert _gates(t)["catch_status"]["c1|b1|E"] == v.UNSTABLE


def test_distinct_cell_counting():
    runs = _runs(cases=(("c1", ("b1", "b2")),))
    t = _table()
    t["rows"] = []
    for bug in ("b1", "b2"):
        for arm in ("B", "E"):
            for rep in (0, 1):
                t["rows"].append(
                    _row(
                        "c1",
                        bug,
                        arm,
                        rep,
                        "caught",
                        flag="flag" if (arm == "B" and rep == 0) else "none",
                        ev="x" if (arm == "B" and rep == 0) else "",
                        notes="halluc" if (arm == "E" and rep == 1) else "",
                    )
                )
    r = _v().gates(t, runs, "B", "E")
    assert r["informational"]["B"] == {"leaks": 0, "unconfirmed_flags": 1, "hallucinations": 0}
    assert r["informational"]["E"] == {"leaks": 0, "unconfirmed_flags": 0, "hallucinations": 1}
    # A flagged cell the verifier confirms counts one leak and zero unconfirmed flags.
    t["rows"][0]["leak"] = "confirmed"
    r = _v().gates(t, runs, "B", "E")
    assert (
        r["informational"]["B"]["leaks"] == 1 and r["informational"]["B"]["unconfirmed_flags"] == 0
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda t: t["negative_assessments"].pop(), "missing negative-control assessment"),
        (lambda t: t["rows"][0].__setitem__("leak", "flag"), "leak 'flag'"),
        (lambda t: t["rows"][0].pop("leak"), "leak None"),
        (lambda t: t["rows"][0].pop("grader_flag"), "grader_flag None"),
        (lambda t: t["rows"][0].update(leak="confirmed"), "leak_evidence must be non-empty"),
        (lambda t: t["rows"][0].update(grader_flag="flag"), "leak_evidence must be non-empty"),
        (lambda t: t["rows"][0].update(leak_evidence="stray"), "none/none cell"),
        (lambda t: t["negative_assessments"][0].update(leak="flag"), "leak 'flag'"),
        (
            lambda t: t["negative_assessments"][0].update(
                findings=[
                    {"severity": "P1", "summary": "Same Thing"},
                    {"severity": "P2", "summary": "same  thing"},
                ]
            ),
            "duplicate finding",
        ),
        (
            lambda t: t["negative_assessments"][0].update(
                findings=[{"severity": "blocker", "summary": "x"}]
            ),
            "severity 'blocker'",
        ),
        (lambda t: t["rows"].pop(), "missing grading row"),
        (lambda t: t["rows"].append(dict(t["rows"][0])), "duplicate grading row"),
    ],
)
def test_validation_errors(mutate, msg):
    t = _table()
    mutate(t)
    with pytest.raises(ValueError, match=msg):
        _v().validate_grades(t, _runs())


def test_unblind_rejects_unknown_labels():
    t = _table()
    t["rows"][0]["arm"] = "M9"
    with pytest.raises(ValueError, match="M9"):
        _v().unblind(t, {"B": "M1", "E": "M2"})


# --------------------------------------------------------------------------- #
# Audit attachment (informational)
# --------------------------------------------------------------------------- #


def _audit(hits=(), unaudited=(), no_commands=(), vacuous=False, total=8):
    return {
        "hits": list(hits),
        "unaudited": list(unaudited),
        "no_commands": list(no_commands),
        "coverage": {
            "covered": total - len(unaudited) - len(no_commands),
            "no_commands": len(no_commands),
            "unaudited": len(unaudited),
            "total": total,
        },
        "vacuous": vacuous,
    }


def test_audit_hits_attach_per_gate_without_changing_label():
    v = _v()
    base = _table(
        b=("caught", "missed"), e=("missed", "missed")
    )  # B unstable blocks nothing; label PARITY
    hit_b = {
        "case": "c1",
        "arm": "B",
        "repeat": 0,
        "markers": ["--all"],
        "line": "git log --all",
        "orientation_only": False,
    }
    hit_s3 = {
        "case": "s3",
        "arm": "E",
        "repeat": 1,
        "markers": ["git log -1", "GIT_LOG_RE"],
        "line": "git log -1",
        "orientation_only": True,
    }
    plain = _gates(base)
    assert plain["definitive"] is False and "PROVISIONAL" in plain["label"]
    r = _gates(
        base,
        audit=_audit(hits=[hit_b, hit_s3], unaudited=[{"case": "c1", "arm": "E", "repeat": 0}]),
    )
    assert (
        r["verdict"] == plain["verdict"] and r["definitive"] is True and "DEFINITIVE" in r["label"]
    )
    assert r["contested"] == [] and r["contested_gates"] == []
    reg = {(c["case"], c["arm"], c["repeat"]): c for c in r["gate_cells"]["regression"]}
    assert reg[("c1", "B", 0)]["hits"][0]["markers"] == ["--all"]
    assert reg[("c1", "E", 0)]["coverage"] == "unaudited"
    fp = {(c["case"], c["arm"], c["repeat"]): c for c in r["gate_cells"]["fp"]}
    assert fp[("s3", "E", 1)]["orientation_only"] is True
    assert ("c1", "B", 0) not in fp
    assert (
        r["audit_summary"]["cells_with_hits"] == 2
        and r["audit_summary"]["orientation_only_cells"] == 1
    )
    # Vacuous audit sets the flag, label unchanged, no contests.
    r2 = _gates(
        base,
        audit=_audit(
            unaudited=[
                {"case": c, "arm": a, "repeat": k}
                for c in ("c1", "s3")
                for a in "BE"
                for k in (0, 1)
            ],
            vacuous=True,
        ),
    )
    assert (
        r2["audit_vacuous"] is True and r2["verdict"] == plain["verdict"] and r2["contested"] == []
    )
    # An S3 hit is listed under gate 2 even when the label is already NO-GO on a regression.
    nogo = _gates(_table(e=("missed", "missed")), audit=_audit(hits=[hit_s3]))
    assert nogo["verdict"] == v.NO_GO and any(c["hits"] for c in nogo["gate_cells"]["fp"])


# --------------------------------------------------------------------------- #
# CLI: binding order (a)-(h) with synthetic sealed artifacts
# --------------------------------------------------------------------------- #


def _write_campaign(tmp_path, monkeypatch, cases=None, k=2, arms=("B", "E"), tamper=None):
    """A synthetic sealed campaign under a tmp RUNS_DIR: artifacts, manifest,
    blinding.json and a bundle whose id hashes. Returns (run_eval, paths)."""

    import reviewer_audit
    from eval_core.models import RunResult
    from eval_core.store import RunStore, write_json

    run_eval = _run_eval()
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(runs_dir))
    monkeypatch.setattr(run_eval, "_assert_tree_identity", lambda *a, **kw: None)
    # Campaign ledger: this synthetic campaign is the contrast's first dispatch on
    # head h1 (tests that need another ledger shape rewrite the file afterwards);
    # the registered-surface diff is stubbed (fake heads have no git history).
    monkeypatch.setattr(run_eval, "_registered_surface_diff", lambda root, a, b: ("", False))
    monkeypatch.setattr(run_eval, "_read_frozen_ref", lambda root, contrast: "h1")
    run_eval._append_ledger(
        {
            "kind": "dispatch",
            "contrast": ["B", "E"],
            "head": "h1",
            "tree": "t1",
            "subdir": "camp",
            "at": "t",
            "ledger_reset": False,
        }
    )
    loader_cases = cases if cases is not None else _corpus_cases()
    monkeypatch.setattr(
        run_eval.CorpusLoader, "load_cases", lambda self, strata: loader_cases, raising=True
    )
    store = RunStore(str(runs_dir / "camp"))
    results = []
    for case in loader_cases:
        for arm in arms:
            for rep in range(k):
                snap = {
                    "stratum": case.stratum,
                    "ground_truth": (
                        [{"id": f"{case.id}:b1", "must_catch": True}]
                        if case.stratum != "s3_negative"
                        else []
                    ),
                    "allow_severities": ["P3"],
                    "title": case.id,
                }
                usage = {
                    "repo_head_sha": "h1",
                    "repo_dirty": False,
                    "repo_branch": "trunk",
                    "repo_common_dir": "/r/.git",
                    "codex_home_isolated": True,
                    "repo_ref_names": ["trunk"],
                    "repo_worktree_paths": [],
                    "session_commands": [],
                    "session_commands_start_lines": 0,
                    "session_commands_exec_lines": 0,
                    "session_other_events": 0,
                    "session_commands_truncated": False,
                    "worktree_dir": "/x/wt",
                }
                rr = RunResult(
                    case_id=case.id,
                    config_id=arm,
                    repeat_idx=rep,
                    review_markdown=f"review {case.id} {arm} {rep}",
                    model="m",
                    usage=usage,
                    run_id=f"{case.id}.{arm}.r{rep}",
                    case_snapshot=snap,
                )
                store.save(rr.run_id, rr)
                results.append(rr)
    run_ids = [r.run_id for r in results]
    prov = {
        "repo_head_sha": "h1",
        "repo_tree_sha": "t1",
        "repo_dirty": False,
        "repo_branch": "trunk",
        "repo_common_dir": "/r/.git",
        "repo_worktree_paths": [],
        "repo_ref_names": ["trunk"],
    }
    manifest = {
        "run_ids": run_ids,
        "configs": list(arms),
        "k": k,
        "k_per": {},
        "codex_home_path": "/tmp/codex-home-x",
        "artifacts_sha": reviewer_audit.artifacts_sha(store.root, run_ids),
        **prov,
        **run_eval._provenance_aggregates(results, prov),
    }
    if tamper:
        tamper(manifest, store)
        manifest["artifacts_sha"] = manifest.get("artifacts_sha") or reviewer_audit.artifacts_sha(
            store.root, run_ids
        )
    write_json(str(runs_dir / "camp-manifest.json"), manifest)
    monkeypatch.setattr(run_eval, "find_repo_root", lambda: str(_REPO))
    # Build the blinded bundle through the real compare path.
    rc = run_eval.cmd_compare(
        type("N", (), {"subdir": "camp", "allow_mixed": False, "blinded": True})()
    )
    assert rc == 0
    blinding = json.loads((runs_dir / "camp" / "blinding.json").read_text())
    return run_eval, runs_dir, manifest, blinding, results


def _corpus_cases():
    from eval_core.models import Case

    cases = [Case(id=f"s1-c{i}", stratum="s1_synthetic") for i in range(2)]
    cases += [Case(id=f"s2-c{i}", stratum="s2_historical") for i in range(4)]
    cases += [Case(id=f"s3-c{i}", stratum="s3_negative") for i in range(2)]
    cases += [Case(id=f"s4-c{i}", stratum="s4_missed") for i in range(3)]
    return cases


def _full_grades(
    cases, blinding, results, b_catch=("caught", "caught"), e_catch=("caught", "caught")
):
    rows, negs = [], []
    for case in cases:
        for arm, verds in (("B", b_catch), ("E", e_catch)):
            label = blinding["mapping"][arm]
            for rep in (0, 1):
                if case.stratum == "s3_negative":
                    negs.append(_neg(case.id, label, rep))
                else:
                    rows.append(_row(case.id, f"{case.id}:b1", label, rep, verds[rep]))
    return {"bundle_id": blinding["bundle_id"], "rows": rows, "negative_assessments": negs}


def _ns(**kw):
    base = {
        "subdir": "camp",
        "control": "B",
        "candidate": "E",
        "audit_hits": None,
        "expect_head": None,
    }
    base.update(kw)
    return type("N", (), base)()


def test_verdict_cli_happy_path_and_audit(tmp_path, monkeypatch, capsys):
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    cases = _corpus_cases()
    grades = tmp_path / "g.json"
    grades.write_text(
        json.dumps(_full_grades(cases, blinding, results, b_catch=("missed", "missed")))
    )
    assert run_eval.cmd_verdict(_ns(grades=str(grades))) == 0
    out = capsys.readouterr().out
    assert "PROVISIONAL" in out and "GO" in out
    hits = tmp_path / "hits.json"
    assert run_eval.cmd_audit(type("N", (), {"subdir": "camp", "output": str(hits)})()) == 0
    h = json.loads(hits.read_text())
    assert (
        h["coverage"]["no_commands"] == len(results)
        and h["vacuous"] is False
        and h["branch"] == "trunk"
    )
    assert (
        run_eval.cmd_verdict(_ns(grades=str(grades), audit_hits=str(hits), expect_head="h1")) == 0
    )
    assert "DEFINITIVE" in capsys.readouterr().out


@pytest.mark.parametrize(
    "control, candidate, msg",
    [
        ("E", "B", "registered contrast"),
        ("F", "B", "registered contrast"),  # campaign-2 contrast inverted
        ("E", "F", "registered contrast"),  # two candidates, no control
        ("B", "B", "must differ"),
        ("A", "B", "registered contrast"),
    ],
)
def test_verdict_cli_argument_checks(tmp_path, monkeypatch, control, candidate, msg):
    run_eval = _run_eval()
    with pytest.raises(SystemExit, match=msg):
        run_eval.cmd_verdict(_ns(control=control, candidate=candidate, grades="/nonexistent"))


def test_verdict_cli_binding_refusals(tmp_path, monkeypatch):
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    cases = _corpus_cases()
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(cases, blinding, results)))
    bpath = runs_dir / "camp" / "blinding.json"
    good = bpath.read_text()
    # (c) mapping swapped.
    b = json.loads(good)
    b["mapping"] = {k: v for k, v in zip(b["mapping"], reversed(list(b["mapping"].values())))}
    bpath.write_text(json.dumps(b))
    with pytest.raises(SystemExit, match=r"(?s)\(c\).*mapping"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    bpath.write_text(good)
    # (c) run_ids_sha.
    b = json.loads(good)
    b["run_ids_sha"] = "0" * 64
    bpath.write_text(json.dumps(b))
    with pytest.raises(SystemExit, match=r"(?s)\(c\).*run_ids_sha"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    bpath.write_text(good)
    # (c) rubric sha.
    b = json.loads(good)
    b["rubric_sha256"] = "0" * 64
    bpath.write_text(json.dumps(b))
    with pytest.raises(SystemExit, match=r"(?s)\(c\).*rubric"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    bpath.write_text(good)
    # (c) bundle edited after compare.
    bundle = runs_dir / "camp" / "comparison.blinded.md"
    orig = bundle.read_text()
    bundle.write_text(orig + "x")
    with pytest.raises(SystemExit, match=r"(?s)\(c\).*(bundle_id|regenerated)"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    bundle.write_text(orig)
    # (d) wrong bundle_id in grades.
    g = json.loads(grades.read_text())
    g["bundle_id"] = "deadbeefdeadbeef"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(g))
    with pytest.raises(SystemExit, match=r"(?s)\(d\).*bundle_id"):
        run_eval.cmd_verdict(_ns(grades=str(bad)))
    # (d) validation error (omitted S3 cell).
    g = json.loads(grades.read_text())
    g["negative_assessments"].pop()
    bad.write_text(json.dumps(g))
    with pytest.raises(SystemExit, match=r"(?s)\(d\).*missing negative-control"):
        run_eval.cmd_verdict(_ns(grades=str(bad)))
    # (c) artifact edited after sealing -> both the manifest's and blinding.json's seal mismatch.
    art = runs_dir / "camp" / f"{results[0].run_id}.json"
    d = json.loads(art.read_text())
    d["usage"]["session_commands"] = ["/bin/zsh -lc 'git log --all' in /x"]
    art_orig = art.read_text()
    art.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match=r"(?s)\(c\).*artifacts_sha"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    art.write_text(art_orig)


def test_verdict_cli_audit_hits_refusals(tmp_path, monkeypatch):
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    hits = tmp_path / "hits.json"
    assert run_eval.cmd_audit(type("N", (), {"subdir": "camp", "output": str(hits)})()) == 0
    good = json.loads(hits.read_text())
    for key, val, msg in (
        ("run_ids_sha", "0" * 64, "run_ids_sha"),
        ("artifacts_sha", "0" * 64, "artifacts_sha"),
        ("branch", "other", "branch"),
        ("vacuous", True, "'vacuous'"),
        ("no_commands", [], "'no_commands'"),
    ):
        h = dict(good)
        h[key] = val
        (tmp_path / "h2.json").write_text(json.dumps(h))
        with pytest.raises(SystemExit, match=r"(?s)\(e\).*" + msg):
            run_eval.cmd_verdict(_ns(grades=str(grades), audit_hits=str(tmp_path / "h2.json")))
    # Hashes intact but one hit removed -> recomputed audit differs. (Add a hit first.)
    art = runs_dir / "camp" / f"{results[0].run_id}.json"
    d = json.loads(art.read_text())
    d["usage"]["session_commands"] = ["/bin/zsh -lc 'git log --all' in /x"]
    d["usage"]["session_commands_start_lines"] = 1
    d["usage"]["session_commands_exec_lines"] = 1
    art.write_text(json.dumps(d))
    import reviewer_audit

    manifest_path = runs_dir / "camp-manifest.json"
    m = json.loads(manifest_path.read_text())
    m["artifacts_sha"] = reviewer_audit.artifacts_sha(str(runs_dir / "camp"), m["run_ids"])
    manifest_path.write_text(json.dumps(m))
    assert (
        run_eval.cmd_compare(
            type("N", (), {"subdir": "camp", "allow_mixed": False, "blinded": True})()
        )
        == 0
    )
    blinding = json.loads((runs_dir / "camp" / "blinding.json").read_text())
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    assert run_eval.cmd_audit(type("N", (), {"subdir": "camp", "output": str(hits)})()) == 0
    h = json.loads(hits.read_text())
    assert len(h["hits"]) == 1
    h["hits"] = []
    (tmp_path / "h3.json").write_text(json.dumps(h))
    with pytest.raises(SystemExit, match=r"\(e\).*'hits'"):
        run_eval.cmd_verdict(_ns(grades=str(grades), audit_hits=str(tmp_path / "h3.json")))


def test_verdict_cli_protocol_guards(tmp_path, monkeypatch):
    # k=1 realized grid -> refused at (f) AFTER (c)-(d) passed.
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch, k=1)
    rows, negs = [], []
    for case in _corpus_cases():
        for arm in ("B", "E"):
            label = blinding["mapping"][arm]
            if case.stratum == "s3_negative":
                negs.append(_neg(case.id, label, 0))
            else:
                rows.append(_row(case.id, f"{case.id}:b1", label, 0, "caught"))
    grades = tmp_path / "g.json"
    grades.write_text(
        json.dumps({"bundle_id": blinding["bundle_id"], "rows": rows, "negative_assessments": negs})
    )
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*realized grid"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))


def test_verdict_cli_corpus_floor_and_filter(tmp_path, monkeypatch):
    seven = _corpus_cases()[:7]
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(
        tmp_path, monkeypatch, cases=seven
    )
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(seven, blinding, results)))
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*below the floor"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    # A filtered run (manifest covers fewer cases than the corpus) is refused too.
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_eval.CorpusLoader,
        "load_cases",
        lambda self, strata: _corpus_cases()
        + [_corpus_cases()[0].__class__(id="s4-extra", stratum="s4_missed")],
        raising=True,
    )
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*OK runs cover"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))


def test_verdict_cli_provenance_guards(tmp_path, monkeypatch):
    def dirty(manifest, store):
        manifest["run_dirty_any"] = False  # stored says clean ...
        import glob
        import os as _os

        f = sorted(
            x
            for x in glob.glob(store.root + "/*.json")
            if not _os.path.basename(x).startswith("blinding")
        )[0]
        d = json.loads(open(f).read())
        d["usage"]["repo_dirty"] = True  # ... but an artifact is dirty
        open(f, "w").write(json.dumps(d))
        manifest["artifacts_sha"] = None  # re-sealed by the helper after the tamper

    run_eval, runs_dir, manifest, blinding, results = _write_campaign(
        tmp_path, monkeypatch, tamper=dirty
    )
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*aggregates"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    # wrong --expect-head
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*expect-head"):
        run_eval.cmd_verdict(_ns(grades=str(grades), expect_head="h2"))

    # a second head sha hidden by editing run_head_shas -> recomputed differs
    def two_heads(manifest, store):
        import glob
        import os as _os

        f = sorted(
            x
            for x in glob.glob(store.root + "/*.json")
            if not _os.path.basename(x).startswith("blinding")
        )[0]
        d = json.loads(open(f).read())
        d["usage"]["repo_head_sha"] = "h2"
        open(f, "w").write(json.dumps(d))
        manifest["run_head_shas"] = ["h1"]
        manifest["artifacts_sha"] = None

    run_eval, runs_dir, manifest, blinding, results = _write_campaign(
        tmp_path, monkeypatch, tamper=two_heads
    )
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*aggregates"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))


def test_verdict_cli_order_c_before_f(tmp_path, monkeypatch):
    """A manifest violating (c) AND (f) at once is refused at (c)."""
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch, k=1)
    bpath = runs_dir / "camp" / "blinding.json"
    b = json.loads(bpath.read_text())
    b["run_ids_sha"] = "0" * 64
    bpath.write_text(json.dumps(b))
    grades = tmp_path / "g.json"
    grades.write_text(
        json.dumps({"bundle_id": b["bundle_id"], "rows": [], "negative_assessments": []})
    )
    with pytest.raises(SystemExit, match=r"verdict \(c\)"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))


def test_verdict_refuses_failed_manifest_and_traversal(tmp_path, monkeypatch):
    run_eval = _run_eval()
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "camp-manifest.json").write_text(json.dumps({"failed": True}))
    with pytest.raises(SystemExit, match="failed/incomplete"):
        run_eval.cmd_verdict(_ns(grades="/nonexistent"))
    assert run_eval.cmd_verdict(_ns(subdir="../x", grades="/nonexistent")) == 1


@pytest.mark.parametrize("field", ["repo_ref_names", "repo_worktree_paths"])
def test_verdict_refuses_incomplete_manifest_provenance(tmp_path, monkeypatch, field):
    def blank(manifest, store):
        manifest[field] = None

    run_eval, runs_dir, manifest, blinding, results = _write_campaign(
        tmp_path, monkeypatch, tamper=blank
    )
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*write-time provenance incomplete"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))


def test_campaign_reads_latency_flag_and_cost():
    v = _v()
    runs = _runs()
    for rr in runs:
        rr["latency_s"] = 100.0 if rr["config_id"] == "B" else 250.0
        rr["model"] = "gpt-5.6-sol" if rr["config_id"] == "B" else "gpt-6-astra"
        rr["usage"] = {"prompt_tokens_est": 1000}
    reads = v.campaign_reads(runs, "B", "E", {"gpt-5.6-sol": 4.0, "gpt-6-astra": 10.0})
    assert reads["latency"]["B"]["n"] == 4 and reads["latency"]["B"]["p95"] == 100.0
    assert reads["latency_flag_candidate_p95_over_2x_control"] is True  # 250 > 200
    assert reads["cost_estimate_usd"]["E"]["prompt_tokens_est"] == 4000
    assert reads["cost_estimate_usd"]["E"]["usd"] == pytest.approx(4000 * 10.0 / 1e6)
    assert reads["cost_estimate_usd"]["B"]["usd"] == pytest.approx(4000 * 4.0 / 1e6)
    for rr in runs:
        if rr["config_id"] == "E":
            rr["latency_s"] = 199.0
    assert v.campaign_reads(runs, "B", "E")["latency_flag_candidate_p95_over_2x_control"] is False
    assert (
        v.campaign_reads(runs, "B", "E")["cost_estimate_usd"]["E"]["usd"] is None
    )  # no price given
    assert v.campaign_reads(runs, "B", "E")["vendor_lifecycle"] == "operator_supplied"


def test_verdict_cli_output_carries_campaign_reads(tmp_path, monkeypatch, capsys):
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    capsys.readouterr()  # drop the compare/setup output
    assert run_eval.cmd_verdict(_ns(grades=str(grades))) == 0
    out = capsys.readouterr().out
    body = json.loads(out[out.index("{") : out.index("\nVERDICT:")])
    assert "latency" in body["campaign_reads"] and "cost_estimate_usd" in body["campaign_reads"]
    assert body["campaign_reads"]["cost_estimate_usd"]["B"]["model"] == "m"


def _ledger_write(run_eval, entries):
    import os as _os

    path = run_eval._ledger_path()
    if _os.path.exists(path):
        _os.remove(path)
    for e in entries:
        run_eval._append_ledger(e)


def _dispatch(head, subdir):
    return {
        "kind": "dispatch",
        "contrast": ["B", "E"],
        "head": head,
        "tree": "t",
        "subdir": subdir,
        "at": "t",
    }


def _verdict_body(run_eval, grades, capsys, **kw):
    capsys.readouterr()
    assert run_eval.cmd_verdict(_ns(grades=str(grades), **kw)) == 0
    out = capsys.readouterr().out
    return json.loads(out[out.index("{") : out.index("\nVERDICT:")])


def test_verdict_requires_ledger_record(tmp_path, monkeypatch):
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    _ledger_write(run_eval, [])
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*no campaign-ledger"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))


def test_verdict_exposure_status_cases(tmp_path, monkeypatch, capsys):
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    # First dispatch on the frozen head -> mechanical.
    body = _verdict_body(run_eval, grades, capsys)
    assert body["exposure"]["mechanical"] is True
    assert "RE-REGISTERED" not in body["label"] and "PROVISIONAL" in body["label"]
    # Re-exposure: an earlier subdir on the SAME head -> non-mechanical.
    _ledger_write(run_eval, [_dispatch("h1", "first"), _dispatch("h1", "camp")])
    body = _verdict_body(run_eval, grades, capsys)
    assert body["exposure"]["mechanical"] is False and body["definitive"] is False
    assert "RE-REGISTERED" in body["label"] and "re-exposure" in body["exposure"]["reason"]
    assert body["exposure"]["subdirs"] == ["camp", "first"]
    # The FIRST subdir reused after a HEAD change -> non-mechanical (fresh subdir required).
    _ledger_write(run_eval, [_dispatch("h0", "camp"), _dispatch("h1", "camp")])
    monkeypatch.setattr(run_eval, "_read_frozen_ref", lambda root, contrast: "h0")
    body = _verdict_body(run_eval, grades, capsys)
    assert body["exposure"]["mechanical"] is False and "FRESH subdir" in body["exposure"]["reason"]
    # Re-run on a NEW head (frozen h0, anchored by the ref): registered surfaces untouched -> mechanical.
    _ledger_write(run_eval, [_dispatch("h0", "first"), _dispatch("h1", "camp")])
    monkeypatch.setattr(run_eval, "_read_frozen_ref", lambda root, contrast: "h0")
    monkeypatch.setattr(run_eval, "_registered_surface_diff", lambda root, a, b: ("", False))
    body = _verdict_body(run_eval, grades, capsys)
    assert body["exposure"]["mechanical"] is True and body["exposure"]["head_changed"] is True
    # ... the CALL_TIMEOUT_S-only exception -> mechanical with the note.
    monkeypatch.setattr(
        run_eval,
        "_registered_surface_diff",
        lambda root, a, b: (" tools/reviewer-eval/adapters/codex_reviewer.py | 2 +-", True),
    )
    body = _verdict_body(run_eval, grades, capsys)
    assert body["exposure"]["mechanical"] is True and body["exposure"]["timeout_only"] is True
    # ... any other registered-surface change -> RE-REGISTERED, non-mechanical.
    monkeypatch.setattr(
        run_eval,
        "_registered_surface_diff",
        lambda root, a, b: (" tools/reviewer-eval/reviewer_verdict.py | 4 ++--", False),
    )
    body = _verdict_body(run_eval, grades, capsys)
    assert body["exposure"]["mechanical"] is False and body["definitive"] is False
    assert (
        "re-registered" in body["exposure"]["reason"]
        and "reviewer_verdict.py" in body["exposure"]["surface_diff_stat"]
    )


def test_registered_surface_diff_detects_timeout_only_change(tmp_path):
    """Real git: a CALL_TIMEOUT_S-only commit is the exception; any other change is not."""
    import subprocess

    run_eval = _run_eval()
    repo = tmp_path / "r"
    repo.mkdir()

    def git(*a):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@l", *a],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q", "-b", "trunk")
    adapter = repo / "tools" / "reviewer-eval" / "adapters" / "codex_reviewer.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("class X:\n    CALL_TIMEOUT_S = 3600.0\n")
    git("add", "-A")
    git("commit", "-q", "--no-verify", "-m", "a")
    frozen = git("rev-parse", "HEAD")
    adapter.write_text("class X:\n    CALL_TIMEOUT_S = 7200.0\n")
    git("commit", "-q", "-a", "--no-verify", "-m", "b")
    stat, timeout_only = run_eval._registered_surface_diff(
        str(repo), frozen, git("rev-parse", "HEAD")
    )
    assert "codex_reviewer.py" in stat and timeout_only is True
    (repo / "tools" / "reviewer-eval" / "x.py").write_text("y = 1\n")
    git("add", "-A")
    git("commit", "-q", "--no-verify", "-m", "c")
    stat, timeout_only = run_eval._registered_surface_diff(
        str(repo), frozen, git("rev-parse", "HEAD")
    )
    assert "x.py" in stat and timeout_only is False
    assert run_eval._registered_surface_diff(str(repo), frozen, frozen) == ("", False)


def test_verdict_refuses_when_frozen_ref_absent_or_conflicting(tmp_path, monkeypatch):
    """The three frozen-provenance integrity failures are protocol-guard REFUSALS
    (DECISION_RULE.md), distinct from the documented non-mechanical labels."""
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), blinding, results)))
    monkeypatch.setattr(run_eval, "_read_frozen_ref", lambda root, contrast: None)
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*frozen ref .* is absent"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    monkeypatch.setattr(run_eval, "_read_frozen_ref", lambda root, contrast: "h0")
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*!= frozen ref"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))
    monkeypatch.setattr(run_eval, "_read_frozen_ref", lambda root, contrast: "h1")
    _ledger_write(run_eval, [{**_dispatch("h1", "camp"), "ledger_reset": True}])
    with pytest.raises(SystemExit, match=r"(?s)\(f\).*ledger was reset"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))


def test_integrity_logs_fail_closed_on_malformed_records(tmp_path, monkeypatch):
    run_eval = _run_eval()
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    good = json.dumps(
        {
            "kind": "dispatch",
            "contrast": ["B", "E"],
            "head": "h",
            "tree": "t",
            "subdir": "s",
            "at": "a",
        }
    )
    (tmp_path / "runs" / "campaign-ledger.jsonl").write_text(good + "\n" + good[:20] + "\n")
    with pytest.raises(SystemExit, match=r"campaign-ledger.jsonl:2: malformed"):
        run_eval._read_ledger()
    (tmp_path / "runs" / "campaign-ledger.jsonl").write_text(
        json.dumps({"kind": "dispatch", "head": "h"}) + "\n"
    )
    with pytest.raises(SystemExit, match=r":1: 'dispatch' record missing 'contrast'"):
        run_eval._read_ledger()
    (tmp_path / "runs" / "campaign-ledger.jsonl").write_text(good + "\n\n" + good + "\n")
    assert len(run_eval._read_ledger()) == 2  # blank lines are fine
    (tmp_path / "runs" / "x-attempts.jsonl").write_text(
        json.dumps({"kind": "end", "failures": []}) + "\n"
    )
    with pytest.raises(SystemExit, match=r"x-attempts.jsonl:1: 'end' record missing 'started_at'"):
        run_eval._read_attempts("x")
    (tmp_path / "runs" / "x-attempts.jsonl").write_text(json.dumps({"kind": "weird"}) + "\n")
    with pytest.raises(SystemExit, match=r"unknown record kind"):
        run_eval._read_attempts("x")


def test_verdict_refuses_self_consistent_bundle_tamper(tmp_path, monkeypatch):
    """Editing the bundle, recomputing its id and updating blinding.json + the grades
    stays self-consistent but is NOT the bundle regenerated from the sealed artifacts."""
    run_eval, runs_dir, manifest, blinding, results = _write_campaign(tmp_path, monkeypatch)
    bundle = runs_dir / "camp" / "comparison.blinded.md"
    text = bundle.read_text()
    old_id = blinding["bundle_id"]
    edited_with_token = (
        text.replace(old_id, run_eval._BUNDLE_ID_TOKEN) + "\n(an extra grader-visible line)\n"
    )
    new_id = run_eval._bundle_id_of(edited_with_token)
    bundle.write_text(edited_with_token.replace(run_eval._BUNDLE_ID_TOKEN, new_id))
    b = dict(blinding)
    b["bundle_id"] = new_id
    (runs_dir / "camp" / "blinding.json").write_text(json.dumps(b))
    grades = tmp_path / "g.json"
    grades.write_text(json.dumps(_full_grades(_corpus_cases(), b, results)))
    with pytest.raises(SystemExit, match=r"(?s)\(c\).*regenerated from the sealed artifacts"):
        run_eval.cmd_verdict(_ns(grades=str(grades)))


def test_registered_surface_diff_rejects_timeout_decrease(tmp_path):
    import subprocess

    run_eval = _run_eval()
    repo = tmp_path / "r2"
    repo.mkdir()

    def git(*a):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@l", *a],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q", "-b", "trunk")
    adapter = repo / "tools" / "reviewer-eval" / "adapters" / "codex_reviewer.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("class X:\n    CALL_TIMEOUT_S = 7200.0\n")
    git("add", "-A")
    git("commit", "-q", "--no-verify", "-m", "a")
    frozen = git("rev-parse", "HEAD")
    adapter.write_text("class X:\n    CALL_TIMEOUT_S = 60.0\n")
    git("commit", "-q", "-a", "--no-verify", "-m", "b")
    stat, timeout_only = run_eval._registered_surface_diff(
        str(repo), frozen, git("rev-parse", "HEAD")
    )
    assert (
        "codex_reviewer.py" in stat and timeout_only is False
    ), "lowering the cap is NOT the exception"
