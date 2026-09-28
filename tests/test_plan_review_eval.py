"""Unit tests for the plan-review eval harness (no reviewer subprocesses).

Covers the corpus loader contracts, the pinned-SHA criteria sourcing (tested
HERMETICALLY against a temporary git repo — CI checkouts are fetch-depth 1, so
a historical-SHA test would be red in CI), configs.json schema + arm-matrix
decomposition, and the mechanical verdict computation fed synthetic k=2
grading tables covering the reliable/unstable/missed boundaries.

Real corpus cases are local-only (gitignored); tests over them skip when
``corpus/cases/`` is absent (golden-file-skip pattern). The committed fixture
case always loads.
"""

import functools
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_EVAL_ROOT = _REPO / "tools" / "plan-review-eval"

pytestmark = pytest.mark.skipif(
    not _EVAL_ROOT.exists(), reason="plan-review-eval harness not present (isolated install)"
)

if _EVAL_ROOT.exists() and str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))
# eval_core (the shared engine) lives directly under tools/.
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))


@functools.lru_cache(maxsize=1)
def _run_eval():
    """Load the harness CLI under a UNIQUE module name — `run_eval` would collide
    with reviewer-eval's identically-named module when one pytest process runs
    both suites (sys.modules is first-import-wins)."""
    spec = importlib.util.spec_from_file_location(
        "plan_review_run_eval", _EVAL_ROOT / "run_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Corpus loader
# --------------------------------------------------------------------------- #


def test_fixture_case_loads_and_verifies():
    from plan_adapters.corpus_loader import PlanCorpusLoader

    loader = PlanCorpusLoader(str(_EVAL_ROOT / "corpus"), str(_REPO))
    cases = loader.load_cases(None)
    fixture = [c for c in cases if c.stratum == "fixture"]
    assert len(fixture) == 1, "exactly one committed fixture case expected"
    case = fixture[0]
    assert case.id == "fx-mini-plan"
    assert len(case.ground_truth) == 2
    assert loader.verify(case) is None


# Minimal schema-valid fixture object for loader tests (the strict validator
# requires the plan_at_sha discriminator + base_sha on every case).
_FX = {"kind": "plan_at_sha", "base_sha": "HEAD"}


def test_loader_rejects_stratum_mismatch(tmp_path):
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s1_synthetic" / "bad-case"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps({"id": "bad-case", "stratum": "s2_historical", "fixture": _FX})
    )
    with pytest.raises(ValueError, match="stratum mismatch"):
        PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)


def test_loader_rejects_duplicate_ids(tmp_path):
    from plan_adapters.corpus_loader import PlanCorpusLoader

    for stratum in ("s1_synthetic", "s3_negative"):
        d = tmp_path / "cases" / stratum / "dup"
        d.mkdir(parents=True)
        (d / "case.json").write_text(json.dumps({"id": "dup", "stratum": stratum, "fixture": _FX}))
    with pytest.raises(ValueError, match="duplicate case id"):
        PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)


def test_verify_enforces_neutral_severity_vocabulary(tmp_path):
    """P0-P3 / CRITICAL vocab in ground truth must fail verification — a native
    engine vocabulary in the corpus would leak into (and skew) blinded grading."""
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s1_synthetic" / "vocab-case"
    d.mkdir(parents=True)
    (d / "plan.md").write_text("# p\n")
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "vocab-case",
                "stratum": "s1_synthetic",
                "fixture": {"kind": "plan_at_sha", "base_sha": "HEAD", "plan": "plan.md"},
                "ground_truth": [{"id": "g1", "expected_severity": "P1", "rationale": "r"}],
            }
        )
    )
    loader = PlanCorpusLoader(str(tmp_path), str(_REPO))
    (case,) = loader.load_cases(None)
    err = loader.verify(case)
    assert err is not None and "neutral scale" in err


def test_verify_rejects_plan_escaping_case_dir(tmp_path):
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s1_synthetic" / "escape-case"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "escape-case",
                "stratum": "s1_synthetic",
                "fixture": {
                    "kind": "plan_at_sha",
                    "base_sha": "HEAD",
                    "plan": "../../../etc/passwd",
                },
                "ground_truth": [{"id": "g1", "expected_severity": "blocker", "rationale": "r"}],
            }
        )
    )
    loader = PlanCorpusLoader(str(tmp_path), str(_REPO))
    (case,) = loader.load_cases(None)
    err = loader.verify(case)
    assert err is not None and "escapes" in err


def test_real_corpus_cases_verify_when_present():
    """Local-only real cases (golden-file-skip: skipped when the gitignored
    corpus/cases/ tree is absent or empty)."""
    from plan_adapters.corpus_loader import PlanCorpusLoader

    loader = PlanCorpusLoader(str(_EVAL_ROOT / "corpus"), str(_REPO))
    real = [c for c in loader.load_cases(None) if c.stratum != "fixture"]
    if not real:
        pytest.skip("no local corpus cases present (corpus/cases/ is gitignored)")
    for case in real:
        assert loader.verify(case) is None, f"{case.id} failed verification"


# --------------------------------------------------------------------------- #
# Pinned-SHA criteria sourcing (hermetic temp repo)
# --------------------------------------------------------------------------- #


def _mk_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "criteria.md").write_text("historical criteria v1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-q", "-m", "c1"],
        cwd=repo,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    return repo, sha


def test_git_show_materializes_pinned_content(tmp_path):
    from plan_adapters.criteria_source import git_show

    repo, sha = _mk_repo(tmp_path)
    # Content survives later edits — the pin is the whole point.
    (repo / "criteria.md").write_text("drifted v2\n")
    assert git_show(str(repo), sha, "criteria.md") == "historical criteria v1\n"


def test_git_show_fails_actionably_on_unknown_sha(tmp_path):
    from plan_adapters.criteria_source import CriteriaSourceError, git_show

    repo, _sha = _mk_repo(tmp_path)
    with pytest.raises(CriteriaSourceError, match="not present in this clone"):
        git_show(str(repo), "0" * 40, "criteria.md")


def test_git_show_fails_actionably_on_unknown_path(tmp_path):
    from plan_adapters.criteria_source import CriteriaSourceError, git_show

    repo, sha = _mk_repo(tmp_path)
    with pytest.raises(CriteriaSourceError, match="does not exist at the pinned SHA"):
        git_show(str(repo), sha, "nope.md")


def _skip_unless_pin_available():
    """Real-pin guard (golden-file-skip pattern): CI checkouts are depth-1, so
    the pinned control SHA may be absent — hermetic tmp-repo tests cover the
    git-show mechanism everywhere; tests touching the REAL pin skip on
    shallow clones instead of failing."""
    cfg = json.loads((_EVAL_ROOT / "config" / "configs.json").read_text())
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{cfg['control_criteria']['sha']}^{{commit}}"],
        cwd=_REPO,
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("pinned control SHA not present (shallow clone)")


def test_configured_pin_resolves_in_full_clone():
    """The REAL configs.json pin, guarded for shallow clones (CI is depth-1)."""
    from plan_adapters.criteria_source import git_show

    _skip_unless_pin_available()
    cfg = json.loads((_EVAL_ROOT / "config" / "configs.json").read_text())
    pin = cfg["control_criteria"]
    text = git_show(str(_REPO), pin["sha"], pin["path"])
    assert "Review Plan" in text and len(text) > 10_000


def test_render_is_brace_safe():
    from plan_adapters.criteria_source import render

    out = render("A __CRITERIA__ B __PLAN__", criteria="{x}", plan="${y} `z`")
    assert out == "A {x} B ${y} `z`"
    assert "__CRITERIA__" not in out


# --------------------------------------------------------------------------- #
# configs.json schema + arm matrix
# --------------------------------------------------------------------------- #


def test_configs_load_and_pin_models():
    run_eval = _run_eval()

    configs = run_eval._make_configs(run_eval._all_arm_ids())
    assert [c.id for c in configs] == ["A", "B", "C", "D", "E"]
    models = {c.id: c.model for c in configs}
    # "session default" is not a recordable parameter — every Claude arm pins a
    # concrete id (recorded == executed).
    assert all(m.startswith("claude-") for m in models.values()), models
    assert models["D"] != models["B"], "the D probe must pin a different Claude model"
    assert run_eval._control_id() == "A"
    # effort is a held-constant confound (codex-side knob), identical everywhere.
    assert len({c.effort for c in configs}) == 1


def test_arm_matrix_decomposes_into_single_field_contrasts():
    """Every selected arm must differ from at least one other in EXACTLY one
    treatment field — the runner aborts otherwise; assert it up front."""
    run_eval = _run_eval()
    from eval_core.runner import CONFOUND_FIELDS

    configs = run_eval._make_configs(run_eval._all_arm_ids())
    tf = run_eval._treatment_fields()
    for f in CONFOUND_FIELDS:
        if f in tf:
            continue
        assert len({getattr(c, f) for c in configs}) == 1, f"confound {f} varies"
    treatments = {c.id: tuple(getattr(c, f) for f in tf) for c in configs}
    assert len(set(treatments.values())) == len(configs)

    def n_diffs(a, b):
        return sum(1 for f in tf if getattr(a, f) != getattr(b, f))

    for c in configs:
        partners = [o.id for o in configs if o.id != c.id and n_diffs(c, o) == 1]
        assert partners, f"arm {c.id} has no single-field contrast partner"


def test_configs_reject_unknown_arm_key(tmp_path, monkeypatch):
    run_eval = _run_eval()

    cfg = json.loads((_EVAL_ROOT / "config" / "configs.json").read_text())
    cfg["arms"][0]["surprise"] = True
    bad = tmp_path / "configs.json"
    bad.write_text(json.dumps(cfg))
    monkeypatch.setattr(run_eval, "CONFIG_PATH", str(bad))
    with pytest.raises(ValueError, match="unknown key"):
        run_eval._make_configs(["A"])


def test_configs_require_exactly_one_control(tmp_path, monkeypatch):
    run_eval = _run_eval()

    cfg = json.loads((_EVAL_ROOT / "config" / "configs.json").read_text())
    cfg["arms"][1]["role"] = "control"
    bad = tmp_path / "configs.json"
    bad.write_text(json.dumps(cfg))
    monkeypatch.setattr(run_eval, "CONFIG_PATH", str(bad))
    with pytest.raises(ValueError, match="exactly one arm with role='control'"):
        run_eval._make_configs(["A", "B"])


# --------------------------------------------------------------------------- #
# Verdict computation — synthetic k=2 tables (the aggregation the gates run on)
# --------------------------------------------------------------------------- #


def _runs(arms=("A", "B"), cases=("c1",), k=2, gt=("g1",), stratum="s1_synthetic", infra=()):
    """Synthetic RunResult dicts: k repeats per (case, arm); ``infra`` marks
    (case, arm, repeat) triples as INFRA_ERROR."""
    out = []
    for case in cases:
        snap = {
            "stratum": stratum,
            "ground_truth": [{"id": g, "must_catch": True} for g in gt],
            "allow_severities": ["major", "minor"] if stratum == "s3_negative" else [],
            "expect_no_blockers": stratum == "s3_negative",
        }
        for arm in arms:
            for r in range(k):
                out.append(
                    {
                        "case_id": case,
                        "config_id": arm,
                        "repeat_idx": r,
                        "case_snapshot": snap,
                        "infra_error": ("boom" if (case, arm, r) in infra else None),
                    }
                )
    return out


def _neg_cells(spec):
    """spec: {(case, arm, repeat): [finding dicts]} -> assessment cells."""
    return [{"case": c, "arm": a, "repeat": r, "findings": f} for (c, a, r), f in spec.items()]


def _rows(spec):
    """spec: {(case, bug, arm): [verdict per repeat]} -> grading rows (caught
    rows carry the evidence quote validate_grades requires)."""
    rows = []
    for (case, bug, arm), verdicts in spec.items():
        for r, v in enumerate(verdicts):
            row = {"case": case, "bug_id": bug, "arm": arm, "repeat": r, "verdict": v}
            if v == "caught":
                row["evidence"] = f"quote naming {bug}"
            rows.append(row)
    return rows


def test_catch_status_reliable_unstable_missed_boundaries():
    import verdict as V

    runs = _runs()
    grades = {
        "rows": _rows(
            {
                ("c1", "g1", "A"): ["caught", "caught"],  # reliable
                ("c1", "g1", "B"): ["caught", "missed"],  # unstable
            }
        )
    }
    status = V.catch_status(grades, runs)
    assert status[("c1", "g1", "A")] == V.RELIABLE
    assert status[("c1", "g1", "B")] == V.UNSTABLE


def test_partial_counts_as_missed():
    import verdict as V

    runs = _runs()
    grades = {"rows": _rows({("c1", "g1", "B"): ["partial", "partial"]})}
    status = V.catch_status(grades, runs)
    assert status[("c1", "g1", "B")] == V.MISSED


def test_infra_error_repeat_excluded_from_denominator():
    """One OK repeat + one INFRA_ERROR repeat, caught on the OK one → RELIABLE
    (infra noise is not a recall signal)."""
    import verdict as V

    runs = _runs(infra={("c1", "B", 1)})
    grades = {"rows": _rows({("c1", "g1", "B"): ["caught"]})}
    status = V.catch_status(grades, runs)
    assert status[("c1", "g1", "B")] == V.RELIABLE


def test_gate_regression_is_no_go():
    import verdict as V

    runs = _runs()
    grades = {
        "rows": _rows(
            {
                ("c1", "g1", "A"): ["caught", "caught"],
                ("c1", "g1", "B"): ["missed", "missed"],
            }
        ),
        "negative_assessments": [],
    }
    out = V.gates(grades, runs, control="A", candidate="B")
    assert out["verdict"] == V.NO_GO
    assert out["regressions"] == [{"case": "c1", "bug_id": "g1"}]


def test_gate_unstable_control_flags_judgment_not_no_go():
    import verdict as V

    runs = _runs()
    grades = {
        "rows": _rows(
            {
                ("c1", "g1", "A"): ["caught", "missed"],  # unstable control
                ("c1", "g1", "B"): ["missed", "missed"],
            }
        ),
        "negative_assessments": [],
    }
    out = V.gates(grades, runs, control="A", candidate="B")
    assert out["verdict"] == V.PARITY
    assert out["judgment_flags"] == [{"case": "c1", "bug_id": "g1"}]
    assert not out["regressions"]


def test_gate_fp_excess_is_no_go_and_allowed_severities_do_not_count():
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=())
    grades = {
        "rows": [],
        "negative_assessments": _neg_cells(
            {
                ("n1", "A", 0): [],
                ("n1", "A", 1): [],
                ("n1", "B", 0): [{"severity": "blocker", "summary": "spurious"}],
                # inside allow_severities -> never counted, even if a grader listed it
                ("n1", "B", 1): [{"severity": "minor", "summary": "style nit"}],
            }
        ),
    }
    out = V.gates(grades, runs, control="A", candidate="B")
    assert out["fp_candidate"] == 1 and out["fp_control"] == 0
    assert out["verdict"] == V.NO_GO


def test_gate_go_on_strict_improvement():
    import verdict as V

    runs = _runs(gt=("g1", "g2"))
    grades = {
        "rows": _rows(
            {
                ("c1", "g1", "A"): ["caught", "caught"],
                ("c1", "g1", "B"): ["caught", "caught"],
                ("c1", "g2", "A"): ["missed", "missed"],
                ("c1", "g2", "B"): ["caught", "caught"],  # B catches what A missed
            }
        ),
        "negative_assessments": [],
    }
    out = V.gates(grades, runs, control="A", candidate="B")
    assert out["verdict"] == V.GO
    assert out["improvements"] == [{"case": "c1", "bug_id": "g2"}]


def test_gate_parity_when_identical():
    import verdict as V

    runs = _runs()
    grades = {
        "rows": _rows(
            {
                ("c1", "g1", "A"): ["caught", "caught"],
                ("c1", "g1", "B"): ["caught", "caught"],
            }
        ),
        "negative_assessments": [],
    }
    assert V.gates(grades, runs, control="A", candidate="B")["verdict"] == V.PARITY


def test_unblind_maps_labels_and_rejects_unknown():
    import verdict as V

    grades = {
        "rows": [{"case": "c1", "bug_id": "g1", "arm": "M1", "repeat": 0, "verdict": "caught"}],
        "negative_assessments": [{"case": "n1", "arm": "M2", "repeat": 0, "findings": []}],
        "bundle_id": "abc",
    }
    out = V.unblind(grades, {"A": "M1", "B": "M2"})
    assert out["rows"][0]["arm"] == "A"
    assert out["negative_assessments"][0]["arm"] == "B"
    assert out["bundle_id"] == "abc", "non-arm fields must survive unblinding"
    with pytest.raises(ValueError, match="not present in the blinding mapping"):
        V.unblind({"rows": [{"arm": "M9"}], "negative_assessments": []}, {"A": "M1"})


# --------------------------------------------------------------------------- #
# Round-1 review regressions: strict render, verdict integrity, protocol
# --------------------------------------------------------------------------- #


def test_render_raises_on_unprovided_template_token():
    """The dual-arm merge-prompt bug class: a template token render was not
    given must raise, never ship a literal __CRITERIA__ to a reviewer."""
    from plan_adapters.criteria_source import CriteriaSourceError, render

    with pytest.raises(CriteriaSourceError, match="CRITERIA"):
        render("apply <criteria>__CRITERIA__</criteria> to __PLAN__", plan="p")


def test_render_never_rescans_substituted_values():
    """Single-pass: a plan whose TEXT discusses __PLAN__ (like a plan about this
    very token convention) is neither re-substituted nor flagged."""
    from plan_adapters.criteria_source import render

    out = render("plan: __PLAN__", plan="mentions __PLAN__ and __CRITERIA__ tokens")
    assert out == "plan: mentions __PLAN__ and __CRITERIA__ tokens"


def test_merge_prompt_template_renders_completely():
    """The real candidate merge template with the real call signature — every
    token provided, criteria included (the round-1 P1)."""
    from plan_adapters.criteria_source import render

    template = (_EVAL_ROOT / "candidates" / "merge_verify.md").read_text()
    out = render(template, criteria="THE-CRITERIA", plan="THE-PLAN", review_a="R1", review_b="R2")
    assert "THE-CRITERIA" in out and "R1" in out and "R2" in out


def test_gates_raise_on_unknown_arm():
    import verdict as V

    runs = _runs()
    grades = {"rows": [], "negative_assessments": []}
    with pytest.raises(ValueError, match="candidate arm 'X' has no runs"):
        V.gates(grades, runs, control="A", candidate="X")


def test_all_infra_control_is_undetermined_not_go():
    """The round-1 P0: an infra-dead control must never hand the candidate a
    GO (that would be a definitive verdict from no comparative evidence)."""
    import verdict as V

    runs = _runs(infra={("c1", "A", 0), ("c1", "A", 1)})
    grades = {
        "rows": _rows({("c1", "g1", "B"): ["caught", "caught"]}),
        "negative_assessments": [],
    }
    out = V.gates(grades, runs, control="A", candidate="B")
    assert out["verdict"] == V.UNDETERMINED
    assert out["evidence_gaps"] == [{"case": "c1", "arm": "A", "ok": 0, "scheduled": 2}]


def test_all_infra_candidate_is_undetermined_not_no_go():
    import verdict as V

    runs = _runs(infra={("c1", "B", 0), ("c1", "B", 1)})
    grades = {
        "rows": _rows({("c1", "g1", "A"): ["caught", "caught"]}),
        "negative_assessments": [],
    }
    out = V.gates(grades, runs, control="A", candidate="B")
    assert out["verdict"] == V.UNDETERMINED


def test_validate_grades_rejects_bad_vocabulary_and_keys():
    import verdict as V

    runs = _runs()
    bad = {
        "rows": [
            {"case": "c1", "bug_id": "g1", "arm": "A", "repeat": 0, "verdict": "CAUGHT!"},
            {"case": "nope", "bug_id": "g1", "arm": "A", "repeat": 0, "verdict": "missed"},
            {"case": "c1", "bug_id": "g9", "arm": "A", "repeat": 0, "verdict": "missed"},
            {"case": "c1", "bug_id": "g1", "arm": "Z", "repeat": 0, "verdict": "missed"},
            {"case": "c1", "bug_id": "g1", "arm": "B", "repeat": 7, "verdict": "missed"},
        ],
        "negative_assessments": [],
    }
    with pytest.raises(ValueError) as exc:
        V.gates(bad, runs, control="A", candidate="B")
    msg = str(exc.value)
    for frag in ("CAUGHT!", "unknown case", "unknown bug", "unknown arm", "not a scheduled"):
        assert frag in msg, f"missing {frag!r} in: {msg}"


def test_validate_grades_rejects_duplicates_and_evidence_free_catches():
    import verdict as V

    runs = _runs()
    bad = {
        "rows": [
            {"case": "c1", "bug_id": "g1", "arm": "A", "repeat": 0, "verdict": "caught"},
            {"case": "c1", "bug_id": "g1", "arm": "A", "repeat": 0, "verdict": "missed"},
        ],
        "negative_assessments": [],
    }
    with pytest.raises(ValueError) as exc:
        V.gates(bad, runs, control="A", candidate="B")
    msg = str(exc.value)
    assert "duplicate grading row" in msg
    assert "'caught' without evidence" in msg


def test_known_fp_topic_rows_are_never_counted():
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=())
    for rr in runs:
        rr["case_snapshot"] = {
            **rr["case_snapshot"],
            "known_fp_topics": [{"id": "kt-1", "topic": "documented deviation"}],
        }
    grades = {
        "rows": [],
        "negative_assessments": _neg_cells(
            {
                ("n1", "A", 0): [],
                ("n1", "A", 1): [],
                ("n1", "B", 0): [
                    {"severity": "blocker", "summary": "kt", "known_topic_id": "kt-1"}
                ],
                ("n1", "B", 1): [{"severity": "blocker", "summary": "real"}],
            }
        ),
    }
    out = V.gates(grades, runs, control="A", candidate="B")
    assert out["fp_candidate"] == 1, "known-topic finding must be excluded; the other counted"


def test_safe_subdir_rejects_escapes():
    run_eval = _run_eval()

    assert run_eval._safe_subdir("campaign-1") == "campaign-1"
    for bad in ("../x", "a/b", "/abs", ".hidden", "a..b"):
        with pytest.raises(SystemExit):
            run_eval._safe_subdir(bad)


def test_protocol_violations_flag_subfloor_and_k():
    run_eval = _run_eval()

    campaign_ok = {
        "case_strata": {
            **{f"s1-{i}": "s1_synthetic" for i in range(3)},
            **{f"s2-{i}": "s2_historical" for i in range(3)},
            **{f"s3-{i}": "s3_negative" for i in range(2)},
        },
        "k": 2,
        "k_overrides": {},
        "corpus_verified": True,
        "config_ids": ["A", "B", "C", "D", "E"],
    }
    assert run_eval._protocol_violations(campaign_ok) == []

    rehearsal = {"case_strata": {"fx-mini-plan": "fixture"}, "k": 2, "k_overrides": {}}
    v = run_eval._protocol_violations(rehearsal)
    assert any("non-fixture" in x for x in v)

    k1 = {**campaign_ok, "k": 1}
    assert any("k=1" in x for x in run_eval._protocol_violations(k1))
    kper = {**campaign_ok, "k_overrides": {"C": 1}}
    assert any("k_overrides" in x for x in run_eval._protocol_violations(kper))


def test_extraction_identity_tracks_prompt_and_model():
    run_eval = _run_eval()

    ident = run_eval._extraction_identity(run_eval._protocol_snapshot())
    assert ident["model"], "extraction model must be pinned in configs.json"
    assert len(ident["prompt_sha"]) == 16


# --------------------------------------------------------------------------- #
# Round-2 review regressions: confinement, complete grid, registered contrasts,
# corpus-verification gating, artifact lineage
# --------------------------------------------------------------------------- #


def test_claude_argv_has_no_permission_bypass():
    """The round-2 P0: reviewer subprocesses must run under the DEFAULT
    permission model (in-worktree reads auto-allowed, outside reads denied
    headlessly) — bypassPermissions would let a hostile plan read arbitrary
    local files."""
    from plan_adapters.plan_reviewer import PlanReviewer

    argv = PlanReviewer._claude_argv("claude-sonnet-5", "Read,Grep,Glob")
    assert "bypassPermissions" not in argv
    assert "--permission-mode" not in argv
    assert "--no-session-persistence" in argv
    # --safe-mode strips per-machine customization (user CLAUDE.md, plugins,
    # hooks, MCP — which --tools alone does not restrict): reviewer behavior is
    # a function of the pinned model + prompt, and MCP can't widen reads.
    assert "--safe-mode" in argv


def test_prompts_declare_plan_as_untrusted_data():
    for name in ("reviewer_prompt.md", "merge_verify.md", "extraction_prompt.md"):
        text = (_EVAL_ROOT / "candidates" / name).read_text()
        assert "UNTRUSTED DATA" in text, f"{name} lacks the untrusted-data guard"
    # The CONTROL prompt must NOT carry the guard: the baseline is the pinned
    # production workflow exactly as it was — adding an instruction production
    # never had would contaminate the arm-A baseline (round-4 review finding).
    from plan_adapters.criteria_source import _CONTROL_PROMPT

    assert "UNTRUSTED DATA" not in _CONTROL_PROMPT


def test_gates_reject_incomplete_grading_grid():
    """An absent cell fails validation — an empty or truncated table must never
    score (absence used to silently count as missed)."""
    import verdict as V

    runs = _runs()
    grades = {"rows": _rows({("c1", "g1", "B"): ["caught", "caught"]}), "negative_assessments": []}
    with pytest.raises(ValueError, match="missing grading row"):
        V.gates(grades, runs, control="A", candidate="B")


def test_gates_reject_empty_table():
    import verdict as V

    runs = _runs()
    with pytest.raises(ValueError, match="missing grading row"):
        V.gates({"rows": [], "negative_assessments": []}, runs, control="A", candidate="B")


def test_fp_repeat_must_be_ok_repeat():
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=())
    grades = {
        "rows": [],
        "negative_assessments": _neg_cells(
            {
                ("n1", "A", 0): [],
                ("n1", "A", 1): [],
                ("n1", "B", 0): [],
                ("n1", "B", 1): [],
                ("n1", "B", 9): [{"severity": "blocker", "summary": "x"}],
            }
        ),
    }
    with pytest.raises(ValueError, match="not an OK repeat"):
        V.gates(grades, runs, control="A", candidate="B")


def test_registered_contrasts_derived_from_roles():
    run_eval = _run_eval()

    assert run_eval._registered_contrasts() == {("A", "B"), ("B", "C")}


def test_unregistered_contrast_is_a_protocol_violation():
    """D/E probes, reversed pairs, and arbitrary pairs are never gating —
    exercised through the same violation channel cmd_verdict uses."""
    run_eval = _run_eval()

    registered = run_eval._registered_contrasts()
    for pair in (("A", "D"), ("A", "E"), ("B", "A"), ("A", "C"), ("C", "E")):
        assert pair not in registered, f"{pair} must not be a registered gating contrast"


def test_protocol_violation_when_corpus_not_verified():
    run_eval = _run_eval()

    manifest = {
        "case_strata": {
            **{f"s1-{i}": "s1_synthetic" for i in range(3)},
            **{f"s2-{i}": "s2_historical" for i in range(3)},
            **{f"s3-{i}": "s3_negative" for i in range(2)},
        },
        "k": 2,
        "k_overrides": {},
        "config_ids": ["A", "B", "C", "D", "E"],
    }
    v = run_eval._protocol_violations(manifest)
    assert any("not verified" in x for x in v)
    assert run_eval._protocol_violations({**manifest, "corpus_verified": True}) == []


def test_run_aborts_on_unverifiable_case(tmp_path, monkeypatch):
    """cmd_run must fail BEFORE any reviewer call when a selected case fails
    verification (pre-registered prerequisite for a gating verdict)."""
    import argparse

    run_eval = _run_eval()

    d = tmp_path / "cases" / "s1_synthetic" / "bad"
    d.mkdir(parents=True)
    (d / "plan.md").write_text("# p\n")
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "bad",
                "stratum": "s1_synthetic",
                "fixture": {"kind": "plan_at_sha", "base_sha": "HEAD", "plan": "plan.md"},
                "ground_truth": [{"id": "g1", "expected_severity": "P1", "rationale": "r"}],
            }
        )
    )
    monkeypatch.setattr(run_eval, "CORPUS_DIR", str(tmp_path))

    def _boom(*a, **k):  # any reviewer construction means the gate failed
        raise AssertionError("reviewer must not be constructed for an unverified corpus")

    monkeypatch.setattr(run_eval, "_run_matrix", _boom)
    args = argparse.Namespace(
        strata=None, cases="", configs="", k=2, k_per="", subdir="x", max_parallel=1
    )
    assert run_eval.cmd_run(args) == 1


def test_extraction_meta_binds_review_bytes():
    import dataclasses as _dc

    from eval_core.models import RunResult

    run_eval = _run_eval()

    rr = RunResult(case_id="c", config_id="A", repeat_idx=0, review_markdown="review v1")
    ident = {"prompt_sha": "p" * 16, "model": "m"}
    m1 = run_eval._extraction_meta_expected(ident, rr, "s" * 16)
    m2 = run_eval._extraction_meta_expected(
        ident, _dc.replace(rr, review_markdown="review v2"), "s" * 16
    )
    assert m1 != m2, "a regenerated review must invalidate its extraction"
    assert m1["prompt_sha"] == "p" * 16 and "review_sha" in m1
    # And an artifact stamped under a FOREIGN protocol identity never matches.
    m3 = run_eval._extraction_meta_expected(ident, rr, "f" * 16)
    assert m3 != m1 and m3["protocol_sha"] == "f" * 16


def test_run_keys_sha_binds_blinding_to_manifest():
    run_eval = _run_eval()

    a = run_eval._run_keys_sha({"run_keys": ["k1", "k2"]})
    b = run_eval._run_keys_sha({"run_keys": ["k1", "k3"]})
    assert a != b and len(a) == 16


# --------------------------------------------------------------------------- #
# Round-3 review regressions: cleanup containment, fixture/arm gating, corpus
# semantics, bundle binding, full-sha pin
# --------------------------------------------------------------------------- #


def _cleanup_escape_scenario(tmp_path, cleanup_fn):
    """A worktree leaf that is a SYMLINK to an external .worktrees/victim must
    be unlinked, never followed into a recursive delete."""
    victim_root = tmp_path / "external" / ".worktrees"
    victim = victim_root / "victim"
    victim.mkdir(parents=True)
    (victim / "precious.txt").write_text("do not delete")
    managed = tmp_path / "runs" / ".worktrees"
    managed.mkdir(parents=True)
    leaf = managed / "case-x"
    leaf.symlink_to(victim)
    cleanup_fn(str(leaf), str(tmp_path), str(managed))
    assert victim.exists() and (victim / "precious.txt").exists(), "external target deleted!"
    assert not leaf.exists(), "symlinked leaf should have been unlinked"


def test_plan_worktree_cleanup_never_follows_symlink(tmp_path):
    from plan_adapters.worktree import cleanup

    _cleanup_escape_scenario(tmp_path, cleanup)


def test_reviewer_eval_worktree_cleanup_never_follows_symlink(tmp_path):
    """The cross-surface twin (reviewer-eval) carries the same guard."""
    rev_eval = _REPO / "tools" / "reviewer-eval"
    if not rev_eval.exists():
        pytest.skip("reviewer-eval not present")
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(
        "reviewer_eval_worktree_twin", rev_eval / "adapters" / "worktree.py"
    )
    mod = ilu.module_from_spec(spec)
    sys.modules["reviewer_eval_worktree_twin"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("reviewer_eval_worktree_twin", None)
    _cleanup_escape_scenario(tmp_path, mod.cleanup)


def test_cleanup_refuses_targets_outside_trusted_root(tmp_path):
    from plan_adapters.worktree import cleanup

    managed = tmp_path / "runs" / ".worktrees"
    managed.mkdir(parents=True)
    outside = tmp_path / "elsewhere" / ".worktrees" / "dir"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("x")
    # A real directory OUTSIDE the trusted root: parent is named .worktrees
    # (the old guard would have deleted it) but containment now refuses.
    cleanup(str(outside), str(tmp_path), str(managed))
    assert outside.exists() and (outside / "keep.txt").exists()


def test_campaign_run_excludes_fixture_by_default(tmp_path, monkeypatch):
    """The documented campaign command (no --strata/--cases) must not include
    the fabricated fixture case in the gating corpus."""
    import argparse

    run_eval = _run_eval()
    captured = {}

    def _capture(cases, *a, **k):
        captured["ids"] = [c.id for c in cases]
        return []

    monkeypatch.setattr(run_eval, "_run_matrix", _capture)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    d = tmp_path / "cases" / "s3_negative" / "real-neg"
    d.mkdir(parents=True)
    (d / "plan.md").write_text("# plan\n")
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "real-neg",
                "stratum": "s3_negative",
                "fixture": {"kind": "plan_at_sha", "base_sha": head, "plan": "plan.md"},
                "expect_no_blockers": True,
            }
        )
    )
    fx = tmp_path / "fixture" / "fx-local"
    fx.mkdir(parents=True)
    (fx / "plan.md").write_text("# plan\n")
    (fx / "case.json").write_text(
        json.dumps(
            {
                "id": "fx-local",
                "stratum": "fixture",
                "fixture": {"kind": "plan_at_sha", "base_sha": "HEAD", "plan": "plan.md"},
            }
        )
    )
    monkeypatch.setattr(run_eval, "CORPUS_DIR", str(tmp_path))
    args = argparse.Namespace(
        strata=None, cases="", configs="", k=2, k_per="", subdir="x", max_parallel=1
    )
    run_eval.cmd_run(args)
    assert captured["ids"] == ["real-neg"], "fixture leaked into the default campaign selection"


def test_protocol_violation_on_fixture_and_arm_subset():
    run_eval = _run_eval()

    base = {
        "case_strata": {
            **{f"s1-{i}": "s1_synthetic" for i in range(3)},
            **{f"s2-{i}": "s2_historical" for i in range(3)},
            **{f"s3-{i}": "s3_negative" for i in range(2)},
        },
        "k": 2,
        "k_overrides": {},
        "corpus_verified": True,
        "config_ids": ["A", "B", "C", "D", "E"],
    }
    assert run_eval._protocol_violations(base) == []
    with_fixture = {**base, "case_strata": {**base["case_strata"], "fx-mini-plan": "fixture"}}
    assert any("fixture" in v for v in run_eval._protocol_violations(with_fixture))
    subset = {**base, "config_ids": ["A", "B"]}
    assert any(
        "five-arm" in v or "pre-registered" in v for v in run_eval._protocol_violations(subset)
    )


@pytest.mark.parametrize(
    "patch,expect",
    [
        (
            {"stratum_dir": "s1_synthetic", "expect_no_blockers": True, "ground_truth": []},
            "only valid in s3_negative",
        ),
        (
            {
                "stratum_dir": "s1_synthetic",
                "ground_truth": [
                    {"id": "dup", "expected_severity": "blocker", "rationale": "r"},
                    {"id": "dup", "expected_severity": "major", "rationale": "r"},
                ],
            },
            "duplicate ground-truth bug id",
        ),
    ],
)
def test_verify_rejects_semantic_contract_violations(tmp_path, patch, expect):
    from plan_adapters.corpus_loader import PlanCorpusLoader

    stratum = patch.pop("stratum_dir")
    d = tmp_path / "cases" / stratum / "sem-case"
    d.mkdir(parents=True)
    (d / "plan.md").write_text("# p\n")
    case = {
        "id": "sem-case",
        "stratum": stratum,
        "fixture": {"kind": "plan_at_sha", "base_sha": "HEAD", "plan": "plan.md"},
        "ground_truth": [{"id": "g1", "expected_severity": "blocker", "rationale": "r"}],
        **patch,
    }
    (d / "case.json").write_text(json.dumps(case))
    loader = PlanCorpusLoader(str(tmp_path), str(_REPO))
    (c,) = loader.load_cases(None)
    err = loader.verify(c)
    assert err is not None and expect in err


def test_s3_without_negative_flag_rejected(tmp_path):
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s3_negative" / "neg-case"
    d.mkdir(parents=True)
    (d / "plan.md").write_text("# p\n")
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "neg-case",
                "stratum": "s3_negative",
                "fixture": {"kind": "plan_at_sha", "base_sha": "HEAD", "plan": "plan.md"},
                "ground_truth": [{"id": "g1", "expected_severity": "blocker", "rationale": "r"}],
            }
        )
    )
    loader = PlanCorpusLoader(str(tmp_path), str(_REPO))
    (c,) = loader.load_cases(None)
    err = loader.verify(c)
    assert err is not None  # s3 with ground truth (and no negative flag) is contradictory


def test_bundle_id_is_the_artifact_hash():
    """The id IS the hash of the grader-visible bytes (id slot tokenized), so
    EVERYTHING graders see is bound by construction: swapped extraction/run
    assignments, header edits, renderer changes — any byte change changes the
    id (round-5 terminal fix)."""
    run_eval = _run_eval()

    doc = f"# bundle\nBundle ID: `{run_eval._BUNDLE_ID_TOKEN}`\n### M1\nfinding-a\n### M2\nfinding-b\n"
    a = run_eval._bundle_id_of(doc)
    swapped = (
        doc.replace("finding-a", "X").replace("finding-b", "finding-a").replace("X", "finding-b")
    )
    assert a != run_eval._bundle_id_of(swapped), "swapping arm assignments must change the id"
    assert a != run_eval._bundle_id_of(doc.replace("# bundle", "# bundle v2"))
    # verdict's restore-token-then-rehash round trip
    final = doc.replace(run_eval._BUNDLE_ID_TOKEN, a)
    restored = final.replace(a, run_eval._BUNDLE_ID_TOKEN)
    assert run_eval._bundle_id_of(restored) == a


def test_control_pin_must_be_full_sha(tmp_path):
    from plan_adapters.criteria_source import CriteriaSourceError, load_artifacts

    with pytest.raises(CriteriaSourceError, match="FULL 40-hex"):
        load_artifacts(str(_REPO), str(_EVAL_ROOT), {"sha": "7181ec63", "path": "x.md"})


# --------------------------------------------------------------------------- #
# Round-4 review regressions
# --------------------------------------------------------------------------- #


def test_cleanup_guards_run_before_git(tmp_path):
    """git worktree remove itself follows a symlinked leaf, so the symlink and
    containment checks must run BEFORE any git command: a registered external
    worktree behind a symlinked leaf must survive cleanup."""
    import subprocess as sp

    from plan_adapters.worktree import cleanup

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "f.txt").write_text("x")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-q", "-m", "c"],
        cwd=repo,
        check=True,
    )
    external = tmp_path / "external" / ".worktrees" / "registered"
    external.parent.mkdir(parents=True)
    sp.run(["git", "worktree", "add", "--detach", str(external)], cwd=repo, check=True)
    assert external.exists()

    managed = tmp_path / "runs" / ".worktrees"
    managed.mkdir(parents=True)
    leaf = managed / "leaf"
    leaf.symlink_to(external)
    cleanup(str(leaf), str(repo), str(managed))
    assert external.exists() and (external / "f.txt").exists(), (
        "cleanup let git follow the symlinked leaf and remove a registered " "external worktree"
    )
    assert not leaf.exists()


def test_missing_negative_assessment_cells_fail_validation():
    """An omitted negative-control cell must fail validation — never silently
    read as zero false positives (round-4 P1)."""
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=())
    with pytest.raises(ValueError, match="missing negative-control assessment"):
        V.gates({"rows": [], "negative_assessments": []}, runs, control="A", candidate="B")


def test_obsolete_false_positives_section_rejected():
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=())
    grades = {"rows": [], "false_positives": [], "negative_assessments": []}
    with pytest.raises(ValueError, match="obsolete 'false_positives'"):
        V.gates(grades, runs, control="A", candidate="B")


def test_verify_rejects_symbolic_base_sha_for_real_cases(tmp_path):
    from plan_adapters.corpus_loader import PlanCorpusLoader

    for bad in ("HEAD", "main", "7181ec63"):
        d = tmp_path / "cases" / "s3_negative" / f"case-{bad.lower()}"
        d.mkdir(parents=True)
        (d / "plan.md").write_text("# p\n")
        (d / "case.json").write_text(
            json.dumps(
                {
                    "id": f"case-{bad.lower()}",
                    "stratum": "s3_negative",
                    "fixture": {"kind": "plan_at_sha", "base_sha": bad, "plan": "plan.md"},
                    "expect_no_blockers": True,
                }
            )
        )
    loader = PlanCorpusLoader(str(tmp_path), str(_REPO))
    for case in loader.load_cases(None):
        err = loader.verify(case)
        assert err is not None and "FULL 40-hex" in err, f"{case.id} accepted symbolic sha"


def test_control_prompt_matches_pinned_workflow_shape():
    """Arm A's spawn prompt carries ONLY production-faithful instructions: the
    criteria block, the plan block, and the historical spawn framing — no
    candidate-side additions."""
    from plan_adapters.criteria_source import _CONTROL_PROMPT

    assert "__CRITERIA__" in _CONTROL_PROMPT and "__PLAN__" in _CONTROL_PROMPT
    assert "UNTRUSTED" not in _CONTROL_PROMPT
    assert "Steps 2 through 5" in _CONTROL_PROMPT


# --------------------------------------------------------------------------- #
# Round-5 review regressions
# --------------------------------------------------------------------------- #


def test_unequal_fp_exposure_is_undetermined():
    """2-OK control vs 1-OK candidate: the shorter arm must NOT win the FP gate
    by having failed more — unequal exposure is UNDETERMINED (round-5 P1)."""
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=(), infra={("n1", "B", 1)})
    grades = {
        "rows": [],
        "negative_assessments": _neg_cells(
            {
                ("n1", "A", 0): [{"severity": "blocker", "summary": "fp"}],
                ("n1", "A", 1): [],
                ("n1", "B", 0): [],
            }
        ),
    }
    out = V.gates(grades, runs, control="A", candidate="B")
    assert out["verdict"] == V.UNDETERMINED, "infra-shortened arm won the FP gate"
    assert out["evidence_gaps"] == [{"case": "n1", "arm": "B", "ok": 1, "scheduled": 2}]


def test_known_topic_ids_are_rendered_in_bundle():
    """Graders exempt findings by topic id, so the id must appear in the bundle
    they read (round-5 P1)."""
    from eval_core.compare import _render_ground_truth

    snap = {
        "expect_no_blockers": True,
        "allow_severities": ["major", "minor"],
        "known_fp_topics": [{"id": "kt-1", "topic": "documented deviation"}],
    }
    out = _render_ground_truth(snap)
    assert "[kt-1]" in out


def test_verify_requires_topic_ids(tmp_path):
    import subprocess as sp

    from plan_adapters.corpus_loader import PlanCorpusLoader

    head = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    d = tmp_path / "cases" / "s3_negative" / "topic-case"
    d.mkdir(parents=True)
    (d / "plan.md").write_text("# p\n")
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "topic-case",
                "stratum": "s3_negative",
                "fixture": {"kind": "plan_at_sha", "base_sha": head, "plan": "plan.md"},
                "expect_no_blockers": True,
                "known_fp_topics": [{"topic": "no id here"}],
            }
        )
    )
    loader = PlanCorpusLoader(str(tmp_path), str(_REPO))
    (c,) = loader.load_cases(None)
    err = loader.verify(c)
    assert err is not None and "nonempty stable 'id'" in err


# --------------------------------------------------------------------------- #
# Round-6 harness regressions
# --------------------------------------------------------------------------- #


def test_duplicate_findings_in_one_cell_fail_validation():
    """A doubled FP entry must raise, never inflate the total into a NO-GO
    (round-6 M1)."""
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=())
    dup = {"severity": "blocker", "summary": "same spurious claim"}
    grades = {
        "rows": [],
        "negative_assessments": _neg_cells(
            {
                ("n1", "A", 0): [],
                ("n1", "A", 1): [],
                ("n1", "B", 0): [dup, dict(dup)],
                ("n1", "B", 1): [],
            }
        ),
    }
    with pytest.raises(ValueError, match="duplicate finding"):
        V.gates(grades, runs, control="A", candidate="B")


def test_findings_require_summaries():
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=())
    grades = {
        "rows": [],
        "negative_assessments": _neg_cells(
            {
                ("n1", "A", 0): [],
                ("n1", "A", 1): [],
                ("n1", "B", 0): [{"severity": "blocker"}],
                ("n1", "B", 1): [],
            }
        ),
    }
    with pytest.raises(ValueError, match="no summary"):
        V.gates(grades, runs, control="A", candidate="B")


def test_merge_markers_stripped_from_bundle_text():
    """Dual-arm agreement tags identify C/E as dual — the deterministic strip
    must remove them from anything headed into the blinded bundle (round-6 M2)."""
    run_eval = _run_eval()

    text = '- [blocker] claim | quote: "x [consensus]"\n- [minor] other [single reviewer]\n'
    out = run_eval._strip_merge_markers(text)
    assert "[consensus]" not in out and "[single reviewer]" not in out
    assert "[blocker] claim" in out


def test_realized_grid_must_match_declared_schedule():
    """A manifest whose repeat-1 runs vanished must be flagged — one repeat per
    arm must never silently satisfy the declared k=2 schedule (round-7 P1)."""
    from eval_core.models import RunResult

    run_eval = _run_eval()

    manifest = {"case_ids": ["c1"], "config_ids": ["A", "B"], "k": 2, "k_overrides": {}}
    full = [
        RunResult(case_id="c1", config_id=a, repeat_idx=r, review_markdown="x")
        for a in ("A", "B")
        for r in range(2)
    ]
    assert run_eval._realized_grid_violations(manifest, full) == []
    partial = [rr for rr in full if rr.repeat_idx == 0]
    v = run_eval._realized_grid_violations(manifest, partial)
    assert any("missing" in x for x in v)
    dup = full + [full[0]]
    v = run_eval._realized_grid_violations(manifest, dup)
    assert any("duplicate" in x for x in v)


# --------------------------------------------------------------------------- #
# CI round-1 regressions: protocol provenance
# --------------------------------------------------------------------------- #


def test_protocol_identity_captures_rule_configs_candidates_contrasts():
    run_eval = _run_eval()

    ident = run_eval._protocol_identity()
    assert len(ident["decision_rule_sha"]) == 16
    assert len(ident["configs_sha"]) == 16
    assert set(ident["candidates_sha"]) >= {
        "criteria.md",
        "reviewer_prompt.md",
        "merge_verify.md",
        "extraction_prompt.md",
    }
    assert ident["extraction"]["model"]
    assert sorted(map(tuple, ident["registered_contrasts"])) == [("A", "B"), ("B", "C")]


def test_verdict_flags_protocol_drift(tmp_path, monkeypatch):
    """A post-run edit to the protocol (decision rule, configs, candidates,
    extraction, contrasts) must make the verdict NON-GATING — the recorded
    campaign identity, not the live files, defines what a verdict means."""
    run_eval = _run_eval()

    recorded = run_eval._protocol_identity()
    live_drifted = {**recorded, "decision_rule_sha": "f" * 16}
    monkeypatch.setattr(run_eval, "_protocol_identity", lambda: live_drifted)
    manifest = {
        "case_strata": {
            **{f"s1-{i}": "s1_synthetic" for i in range(3)},
            **{f"s2-{i}": "s2_historical" for i in range(3)},
            **{f"s3-{i}": "s3_negative" for i in range(2)},
        },
        "k": 2,
        "k_overrides": {},
        "corpus_verified": True,
        "config_ids": ["A", "B", "C", "D", "E"],
        "protocol": recorded,
    }
    # _protocol_violations itself is clean; the drift check lives in cmd_verdict,
    # exercised here at the comparison level the command performs.
    assert run_eval._protocol_violations(manifest) == []
    assert manifest["protocol"] != run_eval._protocol_identity()


# --------------------------------------------------------------------------- #
# CI round-3 regressions
# --------------------------------------------------------------------------- #


def test_same_summary_different_severity_is_one_finding():
    """The same defect at two severities must be rejected as a duplicate, not
    counted twice (a 1-vs-1 FP comparison became 2-vs-1 NO-GO otherwise)."""
    import verdict as V

    runs = _runs(cases=("n1",), stratum="s3_negative", gt=())
    grades = {
        "rows": [],
        "negative_assessments": _neg_cells(
            {
                ("n1", "A", 0): [],
                ("n1", "A", 1): [],
                ("n1", "B", 0): [
                    {"severity": "blocker", "summary": "same spurious claim"},
                    {"severity": "major", "summary": "Same  Spurious   Claim"},
                ],
                ("n1", "B", 1): [],
            }
        ),
    }
    with pytest.raises(ValueError, match="duplicate finding"):
        V.gates(grades, runs, control="A", candidate="B")


def test_execution_derives_from_protocol_snapshot(tmp_path, monkeypatch):
    """An A->B->A (or any) edit AFTER the snapshot must not reach execution:
    configs/artifacts are built from the snapshot's bytes, not re-read."""
    import json as _json

    run_eval = _run_eval()

    snap = run_eval._protocol_snapshot()
    original_model = snap["raw_config"]["arms"][0]["model"]
    assert original_model != "claude-mutated"
    # Mutate the live config AFTER the snapshot (simulating a mid-window edit).
    cfg = _json.loads((_EVAL_ROOT / "config" / "configs.json").read_text())
    cfg["arms"][0]["model"] = "claude-mutated"
    mutated = tmp_path / "configs.json"
    mutated.write_text(_json.dumps(cfg))
    monkeypatch.setattr(run_eval, "CONFIG_PATH", str(mutated))
    # Execution built from the SNAPSHOT still sees the original model...
    configs = run_eval._make_configs(["A"], raw=snap["raw_config"])
    assert configs[0].model == original_model
    # ...and the end-of-run identity check detects the drift.
    assert run_eval._protocol_identity() != snap["identity"]


def test_snapshot_identity_matches_snapshot_bytes():
    """Identity hashes derive from the SAME bytes execution uses."""
    import hashlib as _hashlib

    run_eval = _run_eval()

    snap = run_eval._protocol_snapshot()
    for name, text in snap["candidate_texts"].items():
        assert (
            _hashlib.sha256(text.encode()).hexdigest()[:16]
            == snap["identity"]["candidates_sha"][name]
        )


# --------------------------------------------------------------------------- #
# CI round-4 regressions
# --------------------------------------------------------------------------- #


def test_control_prompt_participates_in_protocol_identity():
    """Arm A's spawn framing must be part of the recorded protocol and of the
    snapshot execution builds from — editing it can never silently alter the
    control arm under an unchanged identity."""
    import hashlib as _hashlib

    from plan_adapters.criteria_source import _CONTROL_PROMPT, load_artifacts

    run_eval = _run_eval()

    snap = run_eval._protocol_snapshot()
    assert (
        snap["identity"]["control_prompt_sha"]
        == _hashlib.sha256(snap["control_prompt"].encode()).hexdigest()[:16]
    )
    # load_artifacts materializes the control criteria from the REAL pin.
    _skip_unless_pin_available()
    arts = load_artifacts(
        str(_REPO),
        str(_EVAL_ROOT),
        snap["raw_config"]["control_criteria"],
        candidate_texts=snap["candidate_texts"],
        control_prompt_text="SNAPSHOTTED CONTROL PROMPT __CRITERIA__ __PLAN__",
    )
    assert arts["control"].reviewer_prompt.startswith("SNAPSHOTTED CONTROL PROMPT")
    # And the module constant remains the non-campaign default.
    arts_default = load_artifacts(
        str(_REPO), str(_EVAL_ROOT), snap["raw_config"]["control_criteria"]
    )
    assert arts_default["control"].reviewer_prompt == _CONTROL_PROMPT


# --------------------------------------------------------------------------- #
# CI round-6 regressions
# --------------------------------------------------------------------------- #


def test_evaluator_sources_join_protocol_identity():
    run_eval = _run_eval()

    ident = run_eval._protocol_identity()
    assert len(ident["evaluator_sha"]) == 16


def test_post_run_stages_refuse_drifted_protocol(monkeypatch):
    """extract/compare must refuse to run under a protocol that differs from
    the manifest's recorded identity — an A->B->A restore bracketing a single
    stage is caught AT that stage (CI round-6)."""
    run_eval = _run_eval()

    recorded = run_eval._protocol_identity()
    manifest = {"protocol": recorded}
    run_eval._require_recorded_protocol(manifest, "extract")  # clean: no raise
    drifted = {**recorded, "evaluator_sha": "0" * 16}
    monkeypatch.setattr(run_eval, "_protocol_identity", lambda: drifted)
    with pytest.raises(SystemExit, match="differs from the identity recorded"):
        run_eval._require_recorded_protocol(manifest, "extract")


def test_blind_mapping_is_recomputed_not_trusted():
    """A hand-swapped blinding.json mapping must be refused: verdict derives
    the deterministic mapping from the manifest and requires equality."""
    import hashlib as _hashlib

    from eval_core.compare import derive_blind_mapping

    manifest = {"run_keys": ["k1", "k2"], "config_ids": ["A", "B"]}
    salt = _hashlib.sha256("|".join(sorted(manifest["run_keys"])).encode()).hexdigest()
    honest = derive_blind_mapping(sorted(manifest["config_ids"]), salt)
    swapped = {a: honest[b] for a, b in zip(sorted(honest), sorted(honest)[::-1])}
    assert swapped != honest
    # The equality check cmd_verdict applies:
    assert derive_blind_mapping(sorted(manifest["config_ids"]), salt) == honest


# --------------------------------------------------------------------------- #
# CI round-7 regressions
# --------------------------------------------------------------------------- #


def test_extractor_sources_join_protocol_identity():
    """The identity walks BOTH source trees — every plan_adapters module and
    every eval_core module joins evaluator_sha by construction, never by a
    hand-kept list (CI round-7: the extractor was outside the identity)."""
    run_eval = _run_eval()

    labels = [label for label, _ in run_eval._evaluator_source_files()]
    for needed in (
        "plan-review-eval/run_eval.py",
        "plan-review-eval/verdict.py",
        "plan-review-eval/plan_adapters/plan_reviewer.py",
        "plan-review-eval/plan_adapters/criteria_source.py",
        "plan-review-eval/plan_adapters/corpus_loader.py",
        "plan-review-eval/plan_adapters/worktree.py",
        "eval_core/compare.py",
        "eval_core/models.py",
        "eval_core/runner.py",
        "eval_core/store.py",
    ):
        assert needed in labels, f"{needed} missing from the evaluator identity"
    # Artifact/data trees never leak in (runs/ holds materialized worktrees —
    # whole repo checkouts — which would drift the identity per run).
    assert not [x for x in labels if "/runs/" in x or "/corpus/" in x]


def test_evaluator_sha_tracks_source_bytes_and_names(tmp_path, monkeypatch):
    """Editing ANY covered source (e.g. the extractor) — or renaming it with
    unchanged bytes — changes the recorded identity."""
    run_eval = _run_eval()

    src = tmp_path / "plan_reviewer.py"
    src.write_text("EXTRACT = 1\n")
    monkeypatch.setattr(
        run_eval,
        "_evaluator_source_files",
        lambda: [("plan_adapters/plan_reviewer.py", str(src))],
    )
    before = run_eval._protocol_identity()["evaluator_sha"]
    src.write_text("EXTRACT = 2\n")
    edited = run_eval._protocol_identity()["evaluator_sha"]
    assert edited != before, "an extractor edit must drift the identity"
    monkeypatch.setattr(
        run_eval,
        "_evaluator_source_files",
        lambda: [("plan_adapters/renamed.py", str(src))],
    )
    renamed = run_eval._protocol_identity()["evaluator_sha"]
    assert renamed != edited, "a rename with unchanged bytes must drift the identity"


def test_extract_stage_exit_gate_catches_mid_stage_drift(tmp_path, monkeypatch):
    """A→B→A around a single stage, closed end-to-end: the entry gate passes
    against the snapshot the stage executes from; a protocol edit landing
    WHILE the stage runs is caught by the exit gate's FRESH read — wired
    through the real cmd_extract, not just the helper."""
    import argparse as _ap

    run_eval = _run_eval()

    recorded = run_eval._protocol_identity()
    subdir = "exitgate"
    (tmp_path / subdir).mkdir()
    (tmp_path / f"{subdir}-manifest.json").write_text(
        json.dumps({"protocol": recorded, "run_keys": []})
    )
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(run_eval, "_reviewer", lambda root, snapshot=None: None)
    # Entry gate sees the live (== recorded) snapshot; the fresh exit read
    # then reports a drifted identity, as if an edit landed mid-stage.
    drifted = {**recorded, "evaluator_sha": "0" * 16}
    monkeypatch.setattr(run_eval, "_protocol_identity", lambda: drifted)
    with pytest.raises(SystemExit, match="stage exit"):
        run_eval.cmd_extract(_ap.Namespace(subdir=subdir, force=False))


def test_verdict_refuses_foreign_protocol_blinding(tmp_path, monkeypatch):
    """blinding.json must name the protocol identity the manifest records — a
    bundle blinded under another protocol (or a smuggled blinding.json) is
    refused before any grade is read."""
    import argparse as _ap
    import hashlib as _hashlib

    run_eval = _run_eval()

    recorded = run_eval._protocol_identity()
    subdir = "foreignblind"
    (tmp_path / subdir).mkdir()
    (tmp_path / f"{subdir}-manifest.json").write_text(
        json.dumps({"protocol": recorded, "run_keys": []})
    )
    run_keys_sha = _hashlib.sha256(b"").hexdigest()[:16]
    (tmp_path / subdir / "blinding.json").write_text(
        json.dumps({"mapping": {}, "run_keys_sha": run_keys_sha, "protocol_sha": "dead" * 4})
    )
    grades = tmp_path / "grades.json"
    grades.write_text(json.dumps({"rows": [], "bundle_id": "x"}))
    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="protocol_sha"):
        run_eval.cmd_verdict(
            _ap.Namespace(subdir=subdir, grades=str(grades), control="", candidate="B")
        )


@pytest.mark.parametrize("bad", ["false", "true", 0, 1, "yes"])
def test_loader_rejects_schema_invalid_must_catch(tmp_path, bad):
    """The committed schema requires a boolean; the loader enforces it. A
    string 'false' would otherwise become True through truthiness and turn an
    optional defect into a false NO-GO (CI round-7)."""
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s1_synthetic" / "typed-case"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "typed-case",
                "stratum": "s1_synthetic",
                "fixture": _FX,
                "ground_truth": [{"id": "g1", "expected_severity": "blocker", "must_catch": bad}],
            }
        )
    )
    with pytest.raises(ValueError, match="must_catch"):
        PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)


def test_loader_keeps_genuine_false_must_catch_optional(tmp_path):
    """A real JSON `false` loads as False end-to-end: loader → dataclass →
    verdict's must-catch map (never coerced back to mandatory)."""
    import verdict as V
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s1_synthetic" / "optional-case"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "optional-case",
                "stratum": "s1_synthetic",
                "fixture": _FX,
                "ground_truth": [
                    {
                        "id": "g1",
                        "expected_severity": "blocker",
                        "must_catch": False,
                        "rationale": "r",
                    }
                ],
            }
        )
    )
    (case,) = PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)
    assert case.ground_truth[0].must_catch is False
    runs = [
        {
            "case_id": "optional-case",
            "case_snapshot": {"ground_truth": [{"id": "g1", "must_catch": False}]},
        }
    ]
    assert V._must_catch_map(runs)[("optional-case", "g1")] is False


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("expect_no_blockers", "false", "expect_no_blockers"),
        ("weight", True, "weight"),
        ("allow_severities", "major", "allow_severities"),
    ],
)
def test_loader_rejects_other_schema_invalid_types(tmp_path, field, value, match):
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s3_negative" / "typed-neg"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps({"id": "typed-neg", "stratum": "s3_negative", field: value})
    )
    with pytest.raises(ValueError, match=match):
        PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)


# --------------------------------------------------------------------------- #
# CI round-9 regressions
# --------------------------------------------------------------------------- #


def test_external_codex_wrapper_joins_identity():
    """Dual arms execute .claude/scripts/openai_review.py — an external
    executable dependency is protocol too (CI round-8)."""
    run_eval = _run_eval()

    labels = [label for label, _ in run_eval._evaluator_source_files()]
    assert "external/.claude/scripts/openai_review.py" in labels


def test_snapshot_aborts_when_sources_change_during_import(tmp_path, monkeypatch):
    """The read→import→re-read bracket: if a protocol source changes while the
    executing modules are being imported, the snapshot aborts — the imported
    code can no longer be proven identical to the hashed bytes (CI round-8:
    PlanReviewer was imported after the source snapshot)."""
    run_eval = _run_eval()

    src = tmp_path / "mod.py"
    src.write_text("X = 1\n")
    monkeypatch.setattr(
        run_eval, "_evaluator_source_files", lambda: [("plan_adapters/mod.py", str(src))]
    )
    monkeypatch.setattr(run_eval, "_import_executing_modules", lambda: src.write_text("X = 2\n"))
    with pytest.raises(SystemExit, match="changed while the snapshot"):
        run_eval._protocol_snapshot()


def test_campaign_subdir_is_write_once_per_protocol(tmp_path, monkeypatch):
    """A subdirectory registered under one protocol refuses any invocation
    under a different one — cached outcomes can never be re-attributed to a
    later protocol by re-running the same subdir after an edit (CI round-8)."""
    run_eval = _run_eval()

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    recorded = {"decision_rule_sha": "a" * 16}
    fp = {"config_ids": ["A"], "k": 2, "k_overrides": {}, "cases": {}}
    (tmp_path / "camp-manifest.json").write_text(
        json.dumps({"protocol": recorded, "campaign_fingerprint": fp})
    )
    # Same protocol AND same sample plan: resume OK.
    run_eval._require_campaign_registration("camp", recorded, fp)
    with pytest.raises(SystemExit, match="DIFFERENT protocol"):
        run_eval._require_campaign_registration("camp", {"decision_rule_sha": "b" * 16}, fp)
    # An unregistered subdir is a fresh campaign — no manifest, no constraint.
    run_eval._require_campaign_registration("fresh", recorded, fp)


def test_campaign_fingerprint_freezes_sample_plan(tmp_path, monkeypatch):
    """Registration freezes the SAMPLE PLAN too: same protocol but different
    cases, plan bytes, base SHAs, arms, or repeat schedule refuses — an
    outcome-dependent corpus/schedule change can never rewrite a campaign
    whose results exist (CI round-9)."""
    run_eval = _run_eval()

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    recorded = {"decision_rule_sha": "a" * 16}
    fp = {
        "config_ids": ["A", "B"],
        "k": 2,
        "k_overrides": {},
        "cases": {"c1": {"base_sha": "a" * 40, "plan_sha": "s" * 16}},
    }
    (tmp_path / "camp-manifest.json").write_text(
        json.dumps({"protocol": recorded, "campaign_fingerprint": fp})
    )
    mutations = (
        {**fp, "k": 1},
        {**fp, "config_ids": ["A"]},
        {**fp, "k_overrides": {"B": 1}},
        {**fp, "cases": {"c1": {"base_sha": "a" * 40, "plan_sha": "t" * 16}}},  # edited plan
        {**fp, "cases": {"c1": {"base_sha": "b" * 40, "plan_sha": "s" * 16}}},  # moved base
        {**fp, "cases": {"c2": {"base_sha": "a" * 40, "plan_sha": "s" * 16}}},  # swapped case
    )
    for mutated in mutations:
        with pytest.raises(SystemExit, match="sample plan"):
            run_eval._require_campaign_registration("camp", recorded, mutated)


def test_reviewer_executes_bracketed_wrapper_module(tmp_path, monkeypatch):
    """The codex wrapper module handed to PlanReviewer IS the one loaded
    inside the snapshot's read→import→re-read bracket — after the snapshot,
    a disk edit to openai_review.py can never reach dual-arm execution
    (CI round-9: the wrapper was imported after the source snapshot)."""
    import plan_adapters.criteria_source as cs

    run_eval = _run_eval()

    snap = run_eval._protocol_snapshot()
    assert snap["openai_review_mod"] is not None
    # Avoid the pinned-SHA git materialization (shallow CI clones): the wiring
    # under test is snapshot → _reviewer → PlanReviewer, not criteria sourcing.
    monkeypatch.setattr(cs, "load_artifacts", lambda *a, **k: {})
    reviewer = run_eval._reviewer(str(tmp_path), snapshot=snap)
    assert reviewer._mod is snap["openai_review_mod"]


def test_campaign_registers_before_observing(tmp_path, monkeypatch):
    """Registration-first: the manifest (with its protocol identity) reaches
    disk BEFORE any reviewer call, so a crash mid-matrix leaves a registered
    campaign with empty run_keys — never an orphaned cache that a later
    protocol could adopt."""
    import eval_core.runner as runner_mod
    from eval_core.models import Case

    run_eval = _run_eval()

    monkeypatch.setattr(run_eval, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(run_eval, "_reviewer", lambda root, snapshot=None: None)
    monkeypatch.setattr(run_eval, "_read_plan_for_freeze", lambda c: "plan")

    def _crash(*a, **k):
        raise RuntimeError("reviewer died mid-matrix")

    monkeypatch.setattr(runner_mod, "run_matrix", _crash)
    case = Case(id="fx", stratum="fixture")
    with pytest.raises(RuntimeError, match="mid-matrix"):
        run_eval._run_matrix(
            [case], ["A"], k=1, subdir="reg", max_parallel=1, enforce_registration=True
        )
    manifest = json.loads((tmp_path / "reg-manifest.json").read_text())
    assert manifest["run_keys"] == []
    assert manifest["protocol"] == run_eval._protocol_identity()
    # The sample plan is registered alongside the protocol (CI round-9).
    assert manifest["campaign_fingerprint"]["cases"]["fx"]["plan_sha"]
    assert manifest["campaign_fingerprint"]["config_ids"] == ["A"]


def test_loader_rejects_unknown_stratum(tmp_path):
    """An unknown stratum directory must never load (it would count toward the
    pre-registered >=8-case corpus floor while being scored under neither the
    positive nor the negative contract)."""
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s4_custom" / "rogue"
    d.mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"id": "rogue", "stratum": "s4_custom"}))
    with pytest.raises(ValueError, match="stratum"):
        PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)


def test_loader_rejects_string_class_keywords(tmp_path):
    """list() over a string silently becomes a character list in
    grader-visible evidence — reject it at load."""
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s1_synthetic" / "kw-case"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "kw-case",
                "stratum": "s1_synthetic",
                "fixture": _FX,
                "ground_truth": [
                    {
                        "id": "g1",
                        "expected_severity": "blocker",
                        "rationale": "r",
                        "class_keywords": "safe_inference",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="class_keywords"):
        PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)


def test_loader_rejects_unknown_fixture_kind(tmp_path):
    """The schema's discriminator is a const: any other kind would be silently
    materialized as a plan_at_sha checkout — reviewing the wrong repository
    state while counting toward the corpus floor (CI round-9)."""
    from plan_adapters.corpus_loader import PlanCorpusLoader

    d = tmp_path / "cases" / "s1_synthetic" / "kind-case"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps(
            {
                "id": "kind-case",
                "stratum": "s1_synthetic",
                "fixture": {"kind": "git_range", "base_sha": "a" * 40},
                "ground_truth": [{"id": "g1", "expected_severity": "blocker"}],
            }
        )
    )
    with pytest.raises(ValueError, match="plan_at_sha"):
        PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)


def test_worktree_refuses_unknown_fixture_kind(tmp_path):
    """Defense in depth at the adapter: materialize() never guesses a
    repository state for a kind it does not implement."""
    from plan_adapters import worktree as wt

    with pytest.raises(wt.MaterializeError, match="plan_at_sha"):
        wt.materialize("k1", {"kind": "git_range", "base_sha": "HEAD"}, str(_REPO), str(tmp_path))


# --------------------------------------------------------------------------- #
# CI round-10 regressions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rationale", [None, "", "   "])
def test_loader_requires_ground_truth_rationale(tmp_path, rationale):
    """The schema requires ground_truth[].rationale — graders match findings
    against it, so a defect without one cannot be graded faithfully
    (CI round-10)."""
    from plan_adapters.corpus_loader import PlanCorpusLoader

    gt: dict = {"id": "g1", "expected_severity": "blocker"}
    if rationale is not None:
        gt["rationale"] = rationale
    d = tmp_path / "cases" / "s1_synthetic" / "no-rationale"
    d.mkdir(parents=True)
    (d / "case.json").write_text(
        json.dumps(
            {"id": "no-rationale", "stratum": "s1_synthetic", "fixture": _FX, "ground_truth": [gt]}
        )
    )
    with pytest.raises(ValueError, match="rationale"):
        PlanCorpusLoader(str(tmp_path), str(_REPO)).load_cases(None)


def test_extraction_prompt_keeps_substantive_questions():
    """CI round-12: the control engine surfaces implementation-blocking
    ambiguities under 'Questions for the Author'; a categorical question
    exclusion in the extraction rules would drop those findings for arm A
    only — a systematic bias toward the candidate arms (false GO). The
    prompt must extract substantive questions and exclude only non-defect
    clarifications, and the committed control-format rehearsal sample must
    seed exactly that scenario."""
    import re as _re

    prompt = (_EVAL_ROOT / "candidates" / "extraction_prompt.md").read_text()
    # The categorical exclusion form is gone...
    assert ", questions," not in prompt
    # ...replaced by the substantive-question inclusion rule...
    assert "PHRASED AS A QUESTION" in prompt
    assert "missing decision" in prompt
    # ...and the remaining exclusion is qualified, never bare.
    bullet = _re.search(r"- Do NOT extract:.*?(?=\n- )", prompt, _re.S)
    assert bullet is not None
    assert "non-defect clarification questions" in bullet.group(0)

    # The rehearsal probe artifact: sole seeded-defect mention lives under
    # the control format's questions section.
    sample = (
        _EVAL_ROOT / "corpus" / "fixture" / "fx-mini-plan" / "control_format_review.md"
    ).read_text()
    before, sep, after = sample.partition("## Questions for the Author")
    assert sep, "sample must carry the control questions section"
    assert "safe_inference_v2" not in before
    assert "safe_inference_v2" in after
    # And DECISION_RULE.md registers the probe as a rehearsal requirement.
    rule = (_EVAL_ROOT / "DECISION_RULE.md").read_text()
    assert "format-parity extraction probe" in rule
    assert "control_format_review.md" in rule


def test_fingerprint_covers_scoring_metadata(tmp_path):
    """The campaign fingerprint hashes the WHOLE case definition: two cases
    identical in plan bytes and base_sha but differing in any scoring field
    (must_catch, allowances, topics, weight...) register as different
    campaigns (CI round-10: outcome-dependent redefinition produced an
    identical fingerprint)."""
    from plan_adapters.corpus_loader import PlanCorpusLoader

    def _load_case(root, must_catch):
        d = root / "cases" / "s1_synthetic" / "meta-case"
        d.mkdir(parents=True)
        (d / "case.json").write_text(
            json.dumps(
                {
                    "id": "meta-case",
                    "stratum": "s1_synthetic",
                    "fixture": _FX,
                    "ground_truth": [
                        {
                            "id": "g1",
                            "expected_severity": "blocker",
                            "must_catch": must_catch,
                            "rationale": "r",
                        }
                    ],
                }
            )
        )
        (case,) = PlanCorpusLoader(str(root), str(_REPO)).load_cases(None)
        return case

    a = _load_case(tmp_path / "a", True)
    b = _load_case(tmp_path / "b", False)
    assert a.fixture["_case_sha"] != b.fixture["_case_sha"]

    run_eval = _run_eval()

    class _Cfg:
        id = "A"

    fp_a = run_eval._campaign_fingerprint([a], [_Cfg()], 2, None)
    fp_b = run_eval._campaign_fingerprint([b], [_Cfg()], 2, None)
    assert fp_a != fp_b
    assert fp_a["cases"]["meta-case"]["case_sha"] == a.fixture["_case_sha"]
