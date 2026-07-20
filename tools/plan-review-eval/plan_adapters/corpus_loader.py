"""Load the plan-case JSON corpus into ``eval_core.models.Case`` objects.

A plan case directory holds ``case.json`` plus the plan document (default
``plan.md``). The fixture pins ``base_sha`` (the repo state the plan was
written against) and the plan filename; ground truth is a list of defects a
faithful plan review should surface, in the harness's NEUTRAL severity
vocabulary (``blocker`` / ``major`` / ``minor`` — see DECISION_RULE.md), so
neither engine's native vocabulary (CRITICAL/MEDIUM/LOW vs P0–P3) leaks into
grading.

Real cases live under ``corpus/cases/`` (gitignored — they are the user's
plans and are never committed); the committed fabricated fixture case lives
under ``corpus/fixture/`` so CI tests and ``smoke`` always have one case.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from eval_core.models import Case, GroundTruthBug

from plan_adapters import worktree

NEUTRAL_SEVERITIES = ("blocker", "major", "minor")

# The complete stratum vocabulary (manifest.schema.json enum). An unknown
# stratum directory would otherwise load, verify, and count toward the
# pre-registered corpus floor (n_real counts every non-fixture stratum).
KNOWN_STRATA = ("fixture", "s1_synthetic", "s2_historical", "s3_negative")


def _strict_types_violation(d: dict) -> Optional[str]:
    """Enforce the manifest.schema.json type contract on the verdict-shaping
    fields BEFORE dataclass construction. json.load never coerces, but Python
    truthiness does: bool("false") is True, so a schema-invalid string like
    '"must_catch": "false"' would silently turn an optional defect into a
    mandatory one (a false NO-GO). The committed schema documents the
    contract; this is its executable enforcement (this repo carries no
    jsonschema dependency)."""
    if not isinstance(d.get("id"), str) or not d["id"].strip():
        return "case 'id' must be a non-empty string"
    if d.get("stratum") not in KNOWN_STRATA:
        return (
            f"case 'stratum' must be one of {KNOWN_STRATA}, got {d.get('stratum')!r} "
            f"— an unknown stratum would silently count toward the corpus floor"
        )
    for field in ("title", "notes"):
        if not isinstance(d.get(field, ""), str):
            return f"'{field}' must be a string"
    for field in ("expect_no_blockers",):
        if not isinstance(d.get(field, False), bool):
            return f"'{field}' must be a JSON boolean (true/false), got {d.get(field)!r}"
    weight = d.get("weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        return f"'weight' must be a number, got {weight!r}"
    sevs = d.get("allow_severities", [])
    if not isinstance(sevs, list) or not all(isinstance(s, str) for s in sevs):
        return "'allow_severities' must be a list of strings"
    files = d.get("expected_files", [])
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        return "'expected_files' must be a list of strings"
    fixture = d.get("fixture", {})
    if not isinstance(fixture, dict):
        return f"'fixture' must be an object, got {fixture!r}"
    # Schema discriminator (const): the only materialization this harness
    # implements is a detached checkout at base_sha. Any other kind would be
    # silently checked out AS plan_at_sha — reviewing the wrong repo state —
    # while still counting toward the corpus floor.
    if fixture.get("kind") != "plan_at_sha":
        return (
            f"fixture.kind must be the schema constant 'plan_at_sha', "
            f"got {fixture.get('kind')!r}"
        )
    if not isinstance(fixture.get("base_sha"), str) or not fixture["base_sha"].strip():
        return "fixture.base_sha must be a non-empty string (schema-required)"
    if "plan" in fixture and not isinstance(fixture["plan"], str):
        return f"fixture.plan must be a string, got {fixture['plan']!r}"
    for field in ("known_fp_topics", "ground_truth"):
        if not isinstance(d.get(field, []), list):
            return f"'{field}' must be a list"
    for i, topic in enumerate(d.get("known_fp_topics", [])):
        if not isinstance(topic, dict):
            return f"known_fp_topics[{i}] must be an object"
    for i, b in enumerate(d.get("ground_truth", [])):
        if not isinstance(b, dict):
            return f"ground_truth[{i}] must be an object"
        if not isinstance(b.get("id"), str) or not b["id"].strip():
            return f"ground_truth[{i}] 'id' must be a non-empty string"
        mc = b.get("must_catch", True)
        if not isinstance(mc, bool):
            return (
                f"ground_truth[{i}] 'must_catch' must be a JSON boolean "
                f"(true/false), got {mc!r} — truthiness would read this as "
                f"{bool(mc)} and silently change the campaign gates"
            )
        if not isinstance(b.get("expected_severity", "major"), str):
            return f"ground_truth[{i}] 'expected_severity' must be a string"
        for field in ("file", "bug_class", "anchor_symbol"):
            if not isinstance(b.get(field, ""), str):
                return f"ground_truth[{i}] '{field}' must be a string"
        # Schema-required (non-empty): graders match findings against the
        # rationale — a defect without one cannot be graded faithfully.
        rationale = b.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            return (
                f"ground_truth[{i}] 'rationale' is schema-required and must be a "
                f"non-empty string, got {rationale!r}"
            )
        kw = b.get("class_keywords", [])
        if not isinstance(kw, list) or not all(isinstance(s, str) for s in kw):
            return (
                f"ground_truth[{i}] 'class_keywords' must be a list of strings, "
                f"got {kw!r} — list() over a string would silently become a "
                f"character list in grader-visible evidence"
            )
        if not isinstance(b.get("provenance", {}), dict):
            return f"ground_truth[{i}] 'provenance' must be an object"
        lw = b.get("line_window", [0, 0])
        if (
            not isinstance(lw, (list, tuple))
            or len(lw) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in lw)
        ):
            return f"ground_truth[{i}] 'line_window' must be a pair of integers, got {lw!r}"
    return None


def _bug_from_dict(d: dict) -> GroundTruthBug:
    lw = d.get("line_window", [0, 0])
    return GroundTruthBug(
        id=d["id"],
        file=d.get("file", "plan.md"),
        line_window=(int(lw[0]), int(lw[1])),
        bug_class=d.get("bug_class", ""),
        expected_severity=d.get("expected_severity", "major"),
        must_catch=d.get("must_catch", True),
        anchor_symbol=d.get("anchor_symbol", ""),
        class_keywords=list(d.get("class_keywords", [])),
        rationale=d.get("rationale", ""),
        provenance=d.get("provenance", {}),
    )


def _case_from_dict(d: dict, case_dir: str) -> Case:
    fixture = dict(d.get("fixture", {}))
    fixture["_case_dir"] = case_dir
    return Case(
        id=d["id"],
        stratum=d["stratum"],
        title=d.get("title", ""),
        fixture=fixture,
        ground_truth=[_bug_from_dict(b) for b in d.get("ground_truth", [])],
        expect_no_blockers=d.get("expect_no_blockers", False),
        allow_severities=d.get("allow_severities", ["major", "minor"]),
        known_fp_topics=d.get("known_fp_topics", []),
        expected_files=d.get("expected_files", []),
        weight=float(d.get("weight", 1.0)),
        notes=d.get("notes", ""),
    )


class PlanCorpusLoader:
    """Loads fixture + real cases; verifies each case is materializable."""

    def __init__(self, corpus_dir: str, repo_root: str):
        self.corpus_dir = corpus_dir
        self.repo_root = repo_root

    def _case_dirs(self, strata: Optional[list[str]] = None) -> list[tuple[str, str]]:
        """(stratum, case_dir) pairs: committed fixture/ plus gitignored cases/."""
        out: list[tuple[str, str]] = []
        fixture_root = os.path.join(self.corpus_dir, "fixture")
        if os.path.isdir(fixture_root) and (strata is None or "fixture" in strata):
            for case_id in sorted(os.listdir(fixture_root)):
                d = os.path.join(fixture_root, case_id)
                if os.path.isdir(d):
                    out.append(("fixture", d))
        cases_root = os.path.join(self.corpus_dir, "cases")
        if os.path.isdir(cases_root):
            for stratum in sorted(os.listdir(cases_root)):
                if strata is not None and stratum not in strata:
                    continue
                stratum_dir = os.path.join(cases_root, stratum)
                if not os.path.isdir(stratum_dir):
                    continue
                for case_id in sorted(os.listdir(stratum_dir)):
                    d = os.path.join(stratum_dir, case_id)
                    if os.path.isdir(d):
                        out.append((stratum, d))
        return out

    def load_cases(self, strata: Optional[list[str]] = None) -> list[Case]:
        cases: list[Case] = []
        for stratum, case_dir in self._case_dirs(strata):
            case_json = os.path.join(case_dir, "case.json")
            if not os.path.exists(case_json):
                continue
            with open(case_json, encoding="utf-8") as fh:
                d = json.load(fh)
            violation = _strict_types_violation(d)
            if violation:
                raise ValueError(f"{case_json}: {violation}")
            if d.get("stratum") != stratum:
                raise ValueError(
                    f"stratum mismatch in {case_json}: declared {d.get('stratum')!r} "
                    f"but filed under {stratum}/ — they must match."
                )
            case = _case_from_dict(d, case_dir)
            # Canonical content identity of the WHOLE case definition — every
            # scoring/grading field (ground truth, must_catch, allowances,
            # known-FP topics, weight, notes) — for the campaign fingerprint:
            # editing any of it after observation is a NEW campaign.
            case.fixture["_case_sha"] = hashlib.sha256(
                json.dumps(d, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            cases.append(case)
        # Fail closed on duplicate/reserved ids (primary key for caching,
        # artifacts, and bundle grouping — mirrors the reviewer-eval loader).
        seen: dict[str, str] = {}
        for case in cases:
            if case.id in (".", "..") or not case.id.strip():
                raise ValueError(f"reserved/empty case id {case.id!r}")
            prior = seen.get(case.id)
            if prior is not None:
                raise ValueError(
                    f"duplicate case id {case.id!r}: defined by both {prior} and "
                    f"{case.fixture.get('_case_dir', '?')}"
                )
            seen[case.id] = case.fixture.get("_case_dir", "?")
        return cases

    def verify(self, case: Case) -> Optional[str]:
        """Materialize the worktree, check the plan file and severity contract.

        Returns an error string, or None if OK. Worktree is always cleaned up.
        """
        # Severity-vocabulary contract (neutral scale end-to-end).
        for bug in case.ground_truth:
            if bug.expected_severity not in NEUTRAL_SEVERITIES:
                return (
                    f"ground-truth bug {bug.id!r} uses severity "
                    f"{bug.expected_severity!r}; plan cases use the neutral scale "
                    f"{NEUTRAL_SEVERITIES} (see DECISION_RULE.md)."
                )
        for sev in case.allow_severities:
            if sev not in NEUTRAL_SEVERITIES:
                return (
                    f"allow_severities entry {sev!r} is not in the neutral scale "
                    f"{NEUTRAL_SEVERITIES}."
                )
        if case.expect_no_blockers and case.ground_truth:
            return "negative-control case (expect_no_blockers) must not declare ground_truth"
        if not case.expect_no_blockers and not case.ground_truth and case.stratum != "fixture":
            return "non-negative case has no ground_truth bugs"
        # Stratum ⇔ scoring-contract semantics: negative controls live ONLY in
        # s3_negative (FP counting keys on the stratum), and s3 cases must be
        # negative controls — a mislabeled case would pad a corpus-floor stratum
        # while being scored under the other contract.
        if case.expect_no_blockers and case.stratum != "s3_negative":
            return (
                f"expect_no_blockers=true is only valid in s3_negative "
                f"(case is filed under {case.stratum})"
            )
        if case.stratum == "s3_negative" and not case.expect_no_blockers:
            return "s3_negative cases must declare expect_no_blockers=true"
        # Unique scoring ids: verdict aggregation is keyed on them; duplicates
        # would silently merge cells.
        bug_ids = [b.id for b in case.ground_truth]
        if len(bug_ids) != len(set(bug_ids)):
            return f"duplicate ground-truth bug id(s): {sorted(set(b for b in bug_ids if bug_ids.count(b) > 1))}"
        for i, topic in enumerate(case.known_fp_topics):
            if not isinstance(topic, dict) or not str(topic.get("id", "")).strip():
                return (
                    f"known_fp_topics[{i}] must be an object with a nonempty stable "
                    f"'id' — graders exempt a finding by that id, so an id-less "
                    f"topic can never be exempted and would miscount as an FP"
                )
        topic_ids = [t["id"] for t in case.known_fp_topics]
        if len(topic_ids) != len(set(topic_ids)):
            return "duplicate known_fp_topics id(s)"

        # Plan file exists, is non-empty, and stays inside the case directory.
        case_dir = case.fixture.get("_case_dir", "")
        rel = case.fixture.get("plan", "plan.md")
        if os.path.isabs(rel):
            return f"fixture.plan {rel!r} must be relative to the case directory"
        base = os.path.realpath(case_dir or ".")
        plan_path = os.path.realpath(os.path.join(base, rel))
        if plan_path != base and not plan_path.startswith(base + os.sep):
            return f"fixture.plan {rel!r} escapes its case directory"
        if not os.path.exists(plan_path):
            return f"plan file missing: {plan_path}"
        with open(plan_path, encoding="utf-8") as fh:
            if not fh.read().strip():
                return "plan file is empty"

        # Real (non-fixture) cases must pin an IMMUTABLE repo state: a symbolic
        # revision (HEAD, a branch, a tag, an abbreviation) would drift between
        # invocations while still reporting corpus_verified=true.
        if case.stratum != "fixture":
            import re as _re

            base = str(case.fixture.get("base_sha", ""))
            if not _re.fullmatch(r"[0-9a-f]{40}", base):
                return (
                    f"fixture.base_sha {base!r} must be a FULL 40-hex commit sha for "
                    f"real cases (symbolic/abbreviated revisions drift; only the "
                    f"committed fixture may use HEAD)"
                )

        # Repo state is materializable.
        runs_root = os.path.join(self.corpus_dir, "..", "runs")
        worktrees_root = os.path.join(os.path.abspath(runs_root), ".worktrees")
        try:
            mat = worktree.materialize(case.id, dict(case.fixture), self.repo_root, worktrees_root)
        except worktree.MaterializeError as exc:
            return f"materialize failed: {exc}"
        worktree.cleanup(mat.worktree_dir, self.repo_root, worktrees_root)
        return None


__all__ = ["PlanCorpusLoader", "NEUTRAL_SEVERITIES"]
