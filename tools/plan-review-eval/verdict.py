"""Mechanical verdict computation from a final (reconciled) grading table.

Implements the pre-registered gates in DECISION_RULE.md as pure functions over
plain dicts, so the reliable/unstable/missed aggregation and the gate logic are
unit-testable with synthetic k=2 tables and never depend on a live run. Human
judgment ends at the grading table; everything after it is mechanical.

Inputs (all plain JSON):

* grades — the FINAL grading table, after the two independent graders were
  reconciled. Blind labels (``M*``) as arms::

      {"rows": [{"case": ..., "bug_id": ..., "arm": "M1", "repeat": 0,
                 "verdict": "caught" | "partial" | "missed",
                 "evidence": "<verbatim quote>"}, ...],
       "negative_assessments": [{"case": ..., "arm": "M2", "repeat": 1,
                                 "findings": [{"severity": "blocker",
                                               "summary": ...,
                                               "known_topic_id": ...}, ...]},
                                ...]}

  ``negative_assessments`` must contain ONE cell per (s3_negative case, arm,
  OK repeat) — with ``findings: []`` for a clean read. An omitted cell fails
  validation rather than silently counting as zero false positives.

* blinding — ``derive_blind_mapping``'s ``{config_id: blind_label}``.
* runs — the manifest-scoped RunResults (for OK repeat counts per (case, arm)
  and each case's snapshot: stratum, ground truth, allow_severities).

``partial`` counts as missed for every gate (right file or class, wrong defect
or location). Repeats that ended in INFRA_ERROR are excluded from the
"caught in EVERY repeat" denominator — infra noise is not a recall signal.

A gating verdict requires COMPARATIVE EVIDENCE AT EQUAL EXPOSURE: ``gates``
raises on an arm with no runs at all, and returns ``UNDETERMINED`` (never
GO/NO-GO/PARITY) when either compared arm has FEWER OK repeats than scheduled
on any compared case — an infra-shortened arm would otherwise win the FP gate
simply by having failed more (less exposure, fewer findings), turning
infrastructure failure into apparent precision. Re-run the failed repeats. ``validate_grades`` rejects malformed tables
(unknown vocabulary, unknown case/bug/arm keys, out-of-range or duplicate
repeats, catches without evidence) before anything is scored.
"""

from __future__ import annotations

from typing import Any

RELIABLE = "reliable"
UNSTABLE = "unstable"
MISSED = "missed"

GO = "GO"
NO_GO = "NO-GO"
PARITY = "PARITY"
UNDETERMINED = "UNDETERMINED"

_VERDICT_VOCAB = ("caught", "partial", "missed")
_SEVERITY_VOCAB = ("blocker", "major", "minor")


def unblind(grades: dict, blinding: dict[str, str]) -> dict:
    """Replace blind labels with real arm ids throughout a grading table."""
    label_to_arm = {v: k for k, v in blinding.items()}
    unknown = {
        r["arm"]
        for r in list(grades.get("rows", [])) + list(grades.get("negative_assessments", []))
        if r.get("arm") not in label_to_arm
    }
    if unknown:
        raise ValueError(
            f"grading table uses arm label(s) {sorted(unknown)} not present in the "
            f"blinding mapping {sorted(blinding.values())} — wrong blinding.json?"
        )
    out = {k: v for k, v in grades.items() if k not in ("rows", "negative_assessments")}
    out["rows"] = [{**r, "arm": label_to_arm[r["arm"]]} for r in grades.get("rows", [])]
    out["negative_assessments"] = [
        {**r, "arm": label_to_arm[r["arm"]]} for r in grades.get("negative_assessments", [])
    ]
    return out


def validate_grades(grades: dict, runs: list[dict]) -> None:
    """Reject a malformed grading table BEFORE it can silently shape a verdict.

    Checks (each violation collected; one ValueError lists them all):
    - row verdicts in {caught, partial, missed}; FP severities in the neutral scale
    - every row's (case, bug_id, arm) exists in the runs' snapshots/config ids
    - repeats are integers that exist as scheduled repeats of that (case, arm)
    - no duplicate (case, bug_id, arm, repeat) rows (a reconciliation slip would
      otherwise double-count), and no duplicate FP (case, arm, repeat, summary)
    - every ``caught`` row carries nonempty evidence (the hallucination check
      has nothing to verify otherwise)
    """
    problems: list[str] = []
    if "false_positives" in grades:
        problems.append(
            "obsolete 'false_positives' section — negative-control findings live in "
            "per-cell 'negative_assessments' (a stale table must not score as 0 FPs)"
        )
    arms = {rr["config_id"] for rr in runs}
    reps_all: dict[tuple[str, str], set[int]] = {}
    for rr in runs:
        reps_all.setdefault((rr["case_id"], rr["config_id"]), set()).add(int(rr["repeat_idx"]))
    bugs: dict[str, set[str]] = {}
    strata: dict[str, str] = {}
    for rr in runs:
        snap = rr.get("case_snapshot") or {}
        strata[rr["case_id"]] = snap.get("stratum", "")
        for bug in snap.get("ground_truth") or []:
            bugs.setdefault(rr["case_id"], set()).add(bug.get("id", "?"))

    seen_rows: set[tuple] = set()
    for i, row in enumerate(grades.get("rows", [])):
        where = f"rows[{i}]"
        case, bug, arm = row.get("case"), row.get("bug_id"), row.get("arm")
        if row.get("verdict") not in _VERDICT_VOCAB:
            problems.append(f"{where}: verdict {row.get('verdict')!r} not in {_VERDICT_VOCAB}")
        if case not in bugs:
            problems.append(f"{where}: unknown case {case!r}")
        elif bug not in bugs[case]:
            problems.append(f"{where}: unknown bug {bug!r} for case {case!r}")
        if arm not in arms:
            problems.append(f"{where}: unknown arm {arm!r} (have {sorted(arms)})")
        rep = row.get("repeat", 0)
        if not isinstance(rep, int) or rep not in reps_all.get((case, arm), set()):
            problems.append(f"{where}: repeat {rep!r} is not a scheduled repeat of ({case}, {arm})")
        key = (case, bug, arm, rep)
        if key in seen_rows:
            problems.append(f"{where}: duplicate grading row for {key}")
        seen_rows.add(key)
        if row.get("verdict") == "caught" and not str(row.get("evidence", "")).strip():
            problems.append(f"{where}: 'caught' without evidence quote")

    ok_reps = ok_repeats(runs)
    seen_cells: set[tuple] = set()
    for i, cell in enumerate(grades.get("negative_assessments", [])):
        where = f"negative_assessments[{i}]"
        case, arm = cell.get("case"), cell.get("arm")
        if strata.get(case) != "s3_negative":
            problems.append(f"{where}: case {case!r} is not an s3_negative case")
        if arm not in arms:
            problems.append(f"{where}: unknown arm {arm!r}")
        rep = cell.get("repeat", 0)
        if not isinstance(rep, int) or rep not in ok_reps.get((case, arm), []):
            problems.append(f"{where}: repeat {rep!r} is not an OK repeat of ({case}, {arm})")
        if not isinstance(cell.get("findings"), list):
            problems.append(f"{where}: findings must be a list (may be empty for a clean read)")
        else:
            seen_summaries: set[str] = set()
            for j, f in enumerate(cell["findings"]):
                if f.get("severity") not in _SEVERITY_VOCAB:
                    problems.append(
                        f"{where}.findings[{j}]: severity {f.get('severity')!r} "
                        f"not in {_SEVERITY_VOCAB}"
                    )
                summary = " ".join(str(f.get("summary", "")).split()).lower()
                if not summary:
                    problems.append(
                        f"{where}.findings[{j}]: finding has no summary — every "
                        f"finding needs a stable identity so duplicates are detectable"
                    )
                # Identity is the FINDING, independent of severity: the same
                # defect listed at two severities is still one defect — counting
                # it twice (or once per severity) would inflate the FP total and
                # could flip the gate. Conflicting severities are a
                # reconciliation failure, rejected outright.
                if summary in seen_summaries:
                    problems.append(
                        f"{where}.findings[{j}]: duplicate finding {summary!r} in "
                        f"one cell (same defect, possibly different severity) — "
                        f"reconcile to ONE entry before scoring"
                    )
                seen_summaries.add(summary)
        key = (case, arm, rep)
        if key in seen_cells:
            problems.append(f"{where}: duplicate assessment cell for {key}")
        seen_cells.add(key)

    # COMPLETE NEGATIVE GRID: one assessment cell per (s3 case, arm, OK repeat)
    # — an omitted cell must fail validation, never read as zero FPs.
    missing_neg = [
        (case, arm, rep)
        for case, stratum in strata.items()
        if stratum == "s3_negative"
        for arm in sorted(arms)
        for rep in ok_reps.get((case, arm), [])
        if (case, arm, rep) not in seen_cells
    ]
    for cell in missing_neg[:20]:
        problems.append(
            f"missing negative-control assessment for {cell} (a clean read is "
            f"findings: [], never an omitted cell)"
        )
    if len(missing_neg) > 20:
        problems.append(f"... and {len(missing_neg) - 20} more missing assessment cells")

    # COMPLETE GRID: the grading protocol requires one row per (case,
    # ground-truth bug, arm, OK repeat). An absent cell must fail validation,
    # never silently score as a miss — an empty or truncated table would
    # otherwise produce a definitive verdict.
    graded_cells = {
        (r.get("case"), r.get("bug_id"), r.get("arm"), r.get("repeat"))
        for r in grades.get("rows", [])
    }
    missing_cells = [
        (case, bug, arm, rep)
        for case, bug_ids in bugs.items()
        for bug in sorted(bug_ids)
        for arm in sorted(arms)
        for rep in ok_reps.get((case, arm), [])
        if (case, bug, arm, rep) not in graded_cells
    ]
    for cell in missing_cells[:20]:
        problems.append(f"missing grading row for {cell} (the grid must be complete)")
    if len(missing_cells) > 20:
        problems.append(f"... and {len(missing_cells) - 20} more missing cells")

    if problems:
        raise ValueError(
            "grading table failed validation (fix the table, do not score it):\n  - "
            + "\n  - ".join(problems)
        )


def ok_repeats(runs: list[dict]) -> dict[tuple[str, str], list[int]]:
    """(case_id, config_id) -> sorted OK repeat indices (INFRA_ERROR excluded)."""
    out: dict[tuple[str, str], list[int]] = {}
    for rr in runs:
        if rr.get("infra_error"):
            continue
        out.setdefault((rr["case_id"], rr["config_id"]), []).append(int(rr["repeat_idx"]))
    return {k: sorted(v) for k, v in out.items()}


def catch_status(
    grades: dict,
    runs: list[dict],
) -> dict[tuple[str, str, str], str]:
    """(case, bug_id, arm) -> reliable | unstable | missed.

    A bug is RELIABLE for an arm iff it was graded ``caught`` on EVERY OK repeat
    of that (case, arm); UNSTABLE if caught on some but not all; MISSED if
    caught on none. ``partial`` is missed. A repeat with no grading row for a
    bug counts as missed for that repeat (graders fill one row per
    (case, bug, arm, repeat); an absent row means nothing caught it).
    """
    reps = ok_repeats(runs)
    caught: dict[tuple[str, str, str], set[int]] = {}
    bugs_per_case: dict[str, set[str]] = {}
    for rr in runs:
        snap = rr.get("case_snapshot") or {}
        for bug in snap.get("ground_truth") or []:
            bugs_per_case.setdefault(rr["case_id"], set()).add(bug.get("id", "?"))
    for row in grades.get("rows", []):
        key = (row["case"], row["bug_id"], row["arm"])
        if row.get("verdict") == "caught":
            caught.setdefault(key, set()).add(int(row.get("repeat", 0)))

    out: dict[tuple[str, str, str], str] = {}
    arms = {rr["config_id"] for rr in runs}
    for case_id, bug_ids in bugs_per_case.items():
        for bug_id in bug_ids:
            for arm in arms:
                n_ok = len(reps.get((case_id, arm), []))
                if n_ok == 0:
                    continue  # no gradeable repeats for this (case, arm)
                got = caught.get((case_id, bug_id, arm), set())
                n_caught = len(got & set(reps[(case_id, arm)]))
                if n_caught == n_ok:
                    out[(case_id, bug_id, arm)] = RELIABLE
                elif n_caught > 0:
                    out[(case_id, bug_id, arm)] = UNSTABLE
                else:
                    out[(case_id, bug_id, arm)] = MISSED
    return out


def fp_counts(grades: dict, runs: list[dict]) -> dict[str, int]:
    """Arm -> total false positives across the negative-assessment findings.

    Defensive re-checks (the decision rule is applied mechanically even where a
    grader slipped): a finding whose severity IS inside the case's
    allow_severities is not counted, and a finding matching one of the case's
    ``known_fp_topics`` — by ``known_topic_id`` referencing a topic's ``id`` —
    is never counted (DECISION_RULE.md).
    """
    snap_by_case: dict[str, dict] = {}
    for rr in runs:
        snap_by_case.setdefault(rr["case_id"], rr.get("case_snapshot") or {})
    out: dict[str, int] = {}
    for cell in grades.get("negative_assessments", []):
        snap = snap_by_case.get(cell["case"], {})
        if snap.get("stratum") != "s3_negative":
            continue
        allowed = set(snap.get("allow_severities") or [])
        topic_ids = {
            t.get("id")
            for t in (snap.get("known_fp_topics") or [])
            if isinstance(t, dict) and t.get("id")
        }
        for f in cell.get("findings") or []:
            sev = f.get("severity", "")
            if sev and sev in allowed:
                continue
            if f.get("known_topic_id") and f["known_topic_id"] in topic_ids:
                continue
            out[cell["arm"]] = out.get(cell["arm"], 0) + 1
    return out


def _must_catch_map(runs: list[dict]) -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    for rr in runs:
        snap = rr.get("case_snapshot") or {}
        for bug in snap.get("ground_truth") or []:
            out[(rr["case_id"], bug.get("id", "?"))] = bool(bug.get("must_catch", True))
    return out


def gates(
    grades: dict,
    runs: list[dict],
    control: str,
    candidate: str,
) -> dict[str, Any]:
    """Apply the pre-registered primary gates for ``candidate`` vs ``control``.

    Returns a dict with the verdict and its full evidence trail:

    * ``regressions`` — must_catch bugs control catches RELIABLY and the
      candidate misses entirely (each one is a NO-GO).
    * ``judgment_flags`` — control UNSTABLE + candidate missed (flagged for
      human judgment in the report, not an automatic NO-GO).
    * ``fp_control`` / ``fp_candidate`` — S3 false-positive totals; candidate
      exceeding control is a NO-GO.
    * ``improvements`` — bugs the candidate catches reliably that control
      misses entirely (any one, or strictly fewer FPs, is a strict improvement).
    * ``verdict`` — NO-GO / GO / PARITY per DECISION_RULE.md, or UNDETERMINED
      when either compared arm lacks gradeable evidence (zero OK repeats) on
      any compared case — a definitive verdict from an infra-dead arm would be
      a decision from no comparative evidence.

    Raises ValueError on an arm with no runs at all (unknown/typo'd arm id) and
    on a grading table that fails ``validate_grades``.
    """
    arms_present = {rr["config_id"] for rr in runs}
    for arm, role in ((control, "control"), (candidate, "candidate")):
        if arm not in arms_present:
            raise ValueError(
                f"{role} arm {arm!r} has no runs in this manifest (have "
                f"{sorted(arms_present)}) — cannot compute a verdict."
            )
    validate_grades(grades, runs)
    # Comparative evidence: both compared arms must have their FULL scheduled
    # repeats OK on every case — unequal exposure is not comparable (an arm
    # with one clean OK repeat vs two would win the FP gate simply by having
    # failed more, turning infrastructure failure into apparent precision).
    reps = ok_repeats(runs)
    scheduled: dict[tuple, set] = {}
    for rr in runs:
        scheduled.setdefault((rr["case_id"], rr["config_id"]), set()).add(int(rr["repeat_idx"]))
    cases = sorted({rr["case_id"] for rr in runs})
    evidence_gaps = [
        {
            "case": c,
            "arm": arm,
            "ok": len(reps.get((c, arm), [])),
            "scheduled": len(scheduled.get((c, arm), set())),
        }
        for c in cases
        for arm in (control, candidate)
        if len(reps.get((c, arm), [])) < len(scheduled.get((c, arm), set()))
    ]

    status = catch_status(grades, runs)
    fps = fp_counts(grades, runs)
    must = _must_catch_map(runs)

    def _s(case: str, bug: str, arm: str) -> str:
        return status.get((case, bug, arm), MISSED)

    pairs = sorted({(c, b) for (c, b, _a) in status})
    regressions = [
        {"case": c, "bug_id": b}
        for (c, b) in pairs
        if must.get((c, b), True)
        and _s(c, b, control) == RELIABLE
        and _s(c, b, candidate) == MISSED
    ]
    judgment_flags = [
        {"case": c, "bug_id": b}
        for (c, b) in pairs
        if _s(c, b, control) == UNSTABLE and _s(c, b, candidate) == MISSED
    ]
    improvements = [
        {"case": c, "bug_id": b}
        for (c, b) in pairs
        if _s(c, b, candidate) == RELIABLE and _s(c, b, control) == MISSED
    ]
    fp_control = fps.get(control, 0)
    fp_candidate = fps.get(candidate, 0)

    if evidence_gaps:
        verdict = UNDETERMINED
    elif regressions or fp_candidate > fp_control:
        verdict = NO_GO
    elif improvements or fp_candidate < fp_control:
        verdict = GO
    else:
        verdict = PARITY

    return {
        "control": control,
        "candidate": candidate,
        "verdict": verdict,
        "evidence_gaps": evidence_gaps,
        "regressions": regressions,
        "judgment_flags": judgment_flags,
        "improvements": improvements,
        "fp_control": fp_control,
        "fp_candidate": fp_candidate,
        "catch_status": {f"{c}|{b}|{a}": s for (c, b, a), s in sorted(status.items())},
    }


__all__ = [
    "unblind",
    "validate_grades",
    "ok_repeats",
    "catch_status",
    "fp_counts",
    "gates",
    "RELIABLE",
    "UNSTABLE",
    "MISSED",
    "GO",
    "NO_GO",
    "PARITY",
    "UNDETERMINED",
]
