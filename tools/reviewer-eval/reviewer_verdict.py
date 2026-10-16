"""Mechanical verdict for the reviewer-eval campaign (DECISION_RULE.md gates).

Pure functions over plain dicts (the sibling ``tools/plan-review-eval/verdict.py``
is the model; this module has a UNIQUE basename because both harness roots sit on
``sys.path`` in tests). Differences from the sibling, per the 2026-10 rule:

* severity vocabulary ``P0|P1|P2|P3`` compared against each S3 case's
  ``allow_severities`` (no known-topic exemption: on this corpus the only
  known-FP topic is already inside the allowed severities);
* a per-row / per-cell ``leak`` field (``none|confirmed`` in the RECONCILED
  table; ``flag`` is a validation error there) plus ``grader_flag``
  (``none|flag``) and ``leak_evidence`` (non-empty iff flagged or confirmed);
  any ``confirmed`` row or S3 cell VOIDS that whole (case, arm, repeat) for
  every gate;
* per-bug statuses RELIABLE / UNSTABLE / MISSED / UNGRADEABLE computed from
  the remaining (non-void, OK) repeats; a void or ungradeable cell CONTESTS
  every gate its case feeds (reported, never decided mechanically);
* the regression gate runs over ALL ground-truth bugs (``must_catch`` is a
  label only); reliable-to-unstable downgrades count, and >= 2 of them force
  PARITY; precedence (0)-(7);
* an optional ``audit_hits`` (the session-log audit's ``hits.json``) attaches
  per-gate cell lists with coverage class and hits - informational only,
  never a contest; without it the outcome is PROVISIONAL.

Python 3.9 compatible.
"""

from __future__ import annotations

from typing import Any, Optional

RELIABLE = "reliable"
UNSTABLE = "unstable"
MISSED = "missed"
UNGRADEABLE = "ungradeable"

GO = "GO"
NO_GO = "NO-GO"
PARITY = "PARITY"
UNDETERMINED = "UNDETERMINED"

# The ONE pre-registered gating contrast (control, candidate). A PR-2 swap does
# not change it; a future campaign registers its own pair in its own PR-1.
REGISTERED_CONTRASTS = (("B", "E"),)

_VERDICT_VOCAB = ("caught", "partial", "missed")
_SEVERITY_VOCAB = ("P0", "P1", "P2", "P3")
_LEAK_FINAL_VOCAB = ("none", "confirmed")
_FLAG_VOCAB = ("none", "flag")
GATE_NAMES = ("regression", "fp", "improvement", "downgrade")


def unblind(grades: dict, blinding: "dict[str, str]") -> dict:
    """Replace blind labels with real arm ids throughout a grading table."""
    label_to_arm = {v: k for k, v in blinding.items()}
    unknown = {
        r.get("arm")
        for r in list(grades.get("rows", [])) + list(grades.get("negative_assessments", []))
        if r.get("arm") not in label_to_arm
    }
    if unknown:
        raise ValueError(
            f"grading table uses arm label(s) {sorted(map(str, unknown))} not present in the "
            f"blinding mapping {sorted(blinding.values())} - wrong blinding.json?"
        )
    out = {k: v for k, v in grades.items() if k not in ("rows", "negative_assessments")}
    out["rows"] = [{**r, "arm": label_to_arm[r["arm"]]} for r in grades.get("rows", [])]
    out["negative_assessments"] = [
        {**r, "arm": label_to_arm[r["arm"]]} for r in grades.get("negative_assessments", [])
    ]
    return out


def _runs_index(runs: "list[dict]") -> "tuple[set, dict, dict, dict]":
    arms = {rr["config_id"] for rr in runs}
    bugs: "dict[str, set[str]]" = {}
    strata: "dict[str, str]" = {}
    snaps: "dict[str, dict]" = {}
    for rr in runs:
        snap = rr.get("case_snapshot") or {}
        strata[rr["case_id"]] = snap.get("stratum", "")
        snaps.setdefault(rr["case_id"], snap)
        for bug in snap.get("ground_truth") or []:
            bugs.setdefault(rr["case_id"], set()).add(bug.get("id", "?"))
        bugs.setdefault(rr["case_id"], set())
    return arms, bugs, strata, snaps


def _check_leak_fields(cell: dict, where: str, problems: "list[str]") -> None:
    leak = cell.get("leak")
    flag = cell.get("grader_flag")
    evidence = str(cell.get("leak_evidence", "") or "").strip()
    if leak not in _LEAK_FINAL_VOCAB:
        problems.append(
            f"{where}: leak {leak!r} not in {_LEAK_FINAL_VOCAB} (a 'flag' left in the "
            f"reconciled table means the verifier never ruled on this cell)"
        )
    if flag not in _FLAG_VOCAB:
        problems.append(f"{where}: grader_flag {flag!r} not in {_FLAG_VOCAB}")
    needs_evidence = leak == "confirmed" or flag == "flag"
    if needs_evidence and not evidence:
        problems.append(f"{where}: leak_evidence must be non-empty when flagged or confirmed")
    if not needs_evidence and evidence and leak in _LEAK_FINAL_VOCAB and flag in _FLAG_VOCAB:
        problems.append(f"{where}: leak_evidence present on a none/none cell")


def ok_repeats(runs: "list[dict]") -> "dict[tuple[str, str], list[int]]":
    """(case_id, config_id) -> sorted OK repeat indices (INFRA_ERROR excluded)."""
    out: "dict[tuple[str, str], list[int]]" = {}
    for rr in runs:
        if rr.get("infra_error"):
            continue
        out.setdefault((rr["case_id"], rr["config_id"]), []).append(int(rr["repeat_idx"]))
    return {k: sorted(v) for k, v in out.items()}


def validate_grades(grades: dict, runs: "list[dict]") -> None:
    """Reject a malformed RECONCILED grading table before it can shape a verdict.

    Beyond the sibling's checks (vocabulary, known cells, scheduled repeats, no
    duplicate rows/cells, caught-with-evidence, complete bug grid and complete
    negative grid, duplicate findings in one S3 cell), every row and S3 cell
    must carry ``leak`` in {none, confirmed}, ``grader_flag`` in {none, flag}
    and a ``leak_evidence`` that is non-empty exactly when flagged or confirmed.
    """
    problems: "list[str]" = []
    arms, bugs, strata, snaps = _runs_index(runs)
    reps_all: "dict[tuple[str, str], set[int]]" = {}
    for rr in runs:
        reps_all.setdefault((rr["case_id"], rr["config_id"]), set()).add(int(rr["repeat_idx"]))
    ok_reps = ok_repeats(runs)

    seen_rows: "set[tuple]" = set()
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
        if not isinstance(rep, int) or rep not in ok_reps.get((case, arm), []):
            problems.append(f"{where}: repeat {rep!r} is not an OK repeat of ({case}, {arm})")
        key = (case, bug, arm, rep)
        if key in seen_rows:
            problems.append(f"{where}: duplicate grading row for {key}")
        seen_rows.add(key)
        if row.get("verdict") == "caught" and not str(row.get("evidence", "")).strip():
            problems.append(f"{where}: 'caught' without evidence quote")
        _check_leak_fields(row, where, problems)

    seen_cells: "set[tuple]" = set()
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
            seen_summaries: "set[str]" = set()
            for j, f in enumerate(cell["findings"]):
                if f.get("severity") not in _SEVERITY_VOCAB:
                    problems.append(
                        f"{where}.findings[{j}]: severity {f.get('severity')!r} "
                        f"not in {_SEVERITY_VOCAB}"
                    )
                summary = " ".join(str(f.get("summary", "")).split()).lower()
                if not summary:
                    problems.append(f"{where}.findings[{j}]: finding has no summary")
                if summary in seen_summaries:
                    problems.append(
                        f"{where}.findings[{j}]: duplicate finding {summary!r} in one cell "
                        f"(same defect, possibly different severity) - reconcile to ONE entry"
                    )
                seen_summaries.add(summary)
        key = (case, arm, rep)
        if key in seen_cells:
            problems.append(f"{where}: duplicate assessment cell for {key}")
        seen_cells.add(key)
        _check_leak_fields(cell, where, problems)

    missing_neg = [
        (case, arm, rep)
        for case, stratum in strata.items()
        if stratum == "s3_negative"
        for arm in sorted(arms)
        for rep in ok_reps.get((case, arm), [])
        if (case, arm, rep) not in seen_cells
    ]
    for cell in missing_neg[:20]:
        problems.append(f"missing negative-control assessment for {cell}")
    if len(missing_neg) > 20:
        problems.append(f"... and {len(missing_neg) - 20} more missing assessment cells")

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


def void_cells(grades: dict) -> "set[tuple[str, str, int]]":
    """(case, arm, repeat) cells with ANY ``leak == "confirmed"`` row or S3 cell."""
    out: "set[tuple[str, str, int]]" = set()
    for r in list(grades.get("rows", [])) + list(grades.get("negative_assessments", [])):
        if r.get("leak") == "confirmed":
            out.add((r["case"], r["arm"], int(r.get("repeat", 0))))
    return out


def catch_status(
    grades: dict, runs: "list[dict]", voids: "Optional[set]" = None
) -> "dict[tuple[str, str, str], str]":
    """(case, bug_id, arm) -> RELIABLE | UNSTABLE | MISSED | UNGRADEABLE.

    Computed from the REMAINING (OK, non-void) repeats of that (case, arm):
    RELIABLE = caught in every remaining repeat AND no void repeat; UNSTABLE =
    caught in some remaining repeat, or in all remaining but >= 1 void; MISSED =
    caught in no remaining repeat (>= 1 remaining); UNGRADEABLE = no remaining
    repeat. ``partial`` is missed. A missing row for a repeat counts as missed.
    """
    voids = voids if voids is not None else void_cells(grades)
    reps = ok_repeats(runs)
    _arms, bugs, _strata, _snaps = _runs_index(runs)
    caught: "dict[tuple[str, str, str], set[int]]" = {}
    for row in grades.get("rows", []):
        if row.get("verdict") == "caught":
            caught.setdefault((row["case"], row["bug_id"], row["arm"]), set()).add(
                int(row.get("repeat", 0))
            )
    arms = {rr["config_id"] for rr in runs}
    out: "dict[tuple[str, str, str], str]" = {}
    for case_id, bug_ids in bugs.items():
        for bug_id in bug_ids:
            for arm in arms:
                ok = reps.get((case_id, arm), [])
                if not ok:
                    continue
                void_here = {r for r in ok if (case_id, arm, r) in voids}
                remaining = [r for r in ok if r not in void_here]
                got = caught.get((case_id, bug_id, arm), set())
                if not remaining:
                    out[(case_id, bug_id, arm)] = UNGRADEABLE
                    continue
                n_caught = len(got & set(remaining))
                if n_caught == len(remaining) and not void_here:
                    out[(case_id, bug_id, arm)] = RELIABLE
                elif n_caught > 0:
                    out[(case_id, bug_id, arm)] = UNSTABLE
                else:
                    out[(case_id, bug_id, arm)] = MISSED
    return out


def fp_counts(
    grades: dict, runs: "list[dict]", voids: "Optional[set]" = None
) -> "tuple[dict[str, int], dict[str, list[dict]]]":
    """Arm -> S3 false-positive total over NON-VOID repeats, plus the FP texts.

    An FP is a finding whose severity is NOT inside the case's ``allow_severities``.
    No topic exemption (the engine counts every listed finding above the line).
    """
    voids = voids if voids is not None else void_cells(grades)
    _arms, _bugs, _strata, snaps = _runs_index(runs)
    counts: "dict[str, int]" = {}
    texts: "dict[str, list[dict]]" = {}
    for cell in grades.get("negative_assessments", []):
        snap = snaps.get(cell["case"], {})
        if snap.get("stratum") != "s3_negative":
            continue
        if (cell["case"], cell["arm"], int(cell.get("repeat", 0))) in voids:
            continue
        allowed = set(snap.get("allow_severities") or [])
        for f in cell.get("findings") or []:
            if f.get("severity") in allowed:
                continue
            counts[cell["arm"]] = counts.get(cell["arm"], 0) + 1
            texts.setdefault(cell["arm"], []).append(
                {
                    "case": cell["case"],
                    "repeat": int(cell.get("repeat", 0)),
                    "severity": f.get("severity"),
                    "summary": f.get("summary", ""),
                    "known_topic": f.get("known_topic", ""),
                }
            )
    return counts, texts


def informational_counts(grades: dict, runs: "list[dict]") -> "dict[str, dict[str, int]]":
    """Per arm: distinct leaked cells, distinct unconfirmed-flag cells, and
    distinct cells with hallucination ``notes`` (counted once per cell)."""
    arms = {rr["config_id"] for rr in runs}
    out = {a: {"leaks": 0, "unconfirmed_flags": 0, "hallucinations": 0} for a in sorted(arms)}
    confirmed: "dict[str, set]" = {a: set() for a in arms}
    flagged: "dict[str, set]" = {a: set() for a in arms}
    noted: "dict[str, set]" = {a: set() for a in arms}
    for r in list(grades.get("rows", [])) + list(grades.get("negative_assessments", [])):
        arm = r.get("arm")
        if arm not in arms:
            continue
        cell = (r["case"], int(r.get("repeat", 0)))
        if r.get("leak") == "confirmed":
            confirmed[arm].add(cell)
        if r.get("grader_flag") == "flag":
            flagged[arm].add(cell)
        if str(r.get("notes", "") or "").strip():
            noted[arm].add(cell)
    for arm in arms:
        out[arm]["leaks"] = len(confirmed[arm])
        out[arm]["unconfirmed_flags"] = len(flagged[arm] - confirmed[arm])
        out[arm]["hallucinations"] = len(noted[arm])
    return out


def _cells_of(runs: "list[dict]", case_id: str, arms: "tuple[str, ...]") -> "list[tuple]":
    return sorted(
        {
            (rr["case_id"], rr["config_id"], int(rr["repeat_idx"]))
            for rr in runs
            if rr["case_id"] == case_id and rr["config_id"] in arms and not rr.get("infra_error")
        }
    )


def _percentile(values: "list[float]", q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]) over a non-empty list."""
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return float(xs[idx])


def campaign_reads(
    runs: "list[dict]", control: str, candidate: str, input_prices: "Optional[dict]" = None
) -> "dict[str, Any]":
    """The pre-registered INFORMATIONAL reads computed from the sealed artifacts:
    per-arm latency (n/mean/p95/max over OK runs), the ``E p95 > 2x B p95``
    flag, and the rough CI input cost per arm = sum(prompt_tokens_est) x the
    official input price per 1M tokens (``input_prices`` maps model name ->
    $/1M input; a missing price yields None). Vendor lifecycle is prose the
    verdict file carries (no machine source), named here as ``operator_supplied``.
    """
    out: "dict[str, Any]" = {
        "latency": {},
        "cost_estimate_usd": {},
        "vendor_lifecycle": "operator_supplied",
    }
    prices = input_prices or {}
    for arm in (control, candidate):
        lat = [
            float(rr.get("latency_s") or 0.0)
            for rr in runs
            if rr["config_id"] == arm and not rr.get("infra_error")
        ]
        out["latency"][arm] = (
            {
                "n": len(lat),
                "mean": sum(lat) / len(lat),
                "p95": _percentile(lat, 0.95),
                "max": max(lat),
            }
            if lat
            else {"n": 0, "mean": None, "p95": None, "max": None}
        )
        toks = 0
        model = ""
        for rr in runs:
            if rr["config_id"] == arm and not rr.get("infra_error"):
                toks += int((rr.get("usage") or {}).get("prompt_tokens_est") or 0)
                model = rr.get("model") or model
        price = prices.get(model)
        out["cost_estimate_usd"][arm] = {
            "model": model,
            "prompt_tokens_est": toks,
            "input_price_per_1m": price,
            "usd": (toks * price / 1_000_000.0) if price is not None else None,
            "note": "+/-50% initial-prompt estimate; output/reasoning/tool tokens not recorded",
        }
    b, e = out["latency"][control]["p95"], out["latency"][candidate]["p95"]
    out["latency_flag_candidate_p95_over_2x_control"] = bool(
        b is not None and e is not None and e > 2 * b
    )
    return out


def attach_audit(result: dict, runs: "list[dict]", audit_hits: "Optional[dict]") -> dict:
    """Attach the hit list per gate (informational): for each gate, every cell
    feeding it with its coverage class and hits. Sets ``audit_vacuous`` and
    ``definitive``. Without ``audit_hits`` the result stays PROVISIONAL."""
    if audit_hits is None:
        result["gate_cells"] = None
        result["audit_vacuous"] = None
        result["definitive"] = False
        result["label"] = f"{result['verdict']} (PROVISIONAL, no session-log audit)"
        return result
    control, candidate = result["control"], result["candidate"]
    arms = (control, candidate)
    unaudited = {(c["case"], c["arm"], int(c["repeat"])) for c in audit_hits.get("unaudited", [])}
    no_cmds = {(c["case"], c["arm"], int(c["repeat"])) for c in audit_hits.get("no_commands", [])}
    by_cell: "dict[tuple, list[dict]]" = {}
    for h in audit_hits.get("hits", []):
        by_cell.setdefault((h["case"], h["arm"], int(h["repeat"])), []).append(
            {
                "markers": h.get("markers", []),
                "line": h.get("line", ""),
                "orientation_only": bool(h.get("orientation_only", False)),
            }
        )
    _a, bugs, strata, _s = _runs_index(runs)
    bug_cases = sorted(c for c, b in bugs.items() if b)
    s3_cases = sorted(c for c, st in strata.items() if st == "s3_negative")

    def describe(cell: tuple) -> dict:
        cov = (
            "unaudited" if cell in unaudited else ("no_commands" if cell in no_cmds else "covered")
        )
        hits = by_cell.get(cell, [])
        return {
            "case": cell[0],
            "arm": cell[1],
            "repeat": cell[2],
            "coverage": cov,
            "hits": hits,
            "orientation_only": bool(hits) and all(h["orientation_only"] for h in hits),
        }

    bug_cells = [describe(c) for case in bug_cases for c in _cells_of(runs, case, arms)]
    s3_cells = [describe(c) for case in s3_cases for c in _cells_of(runs, case, arms)]
    result["gate_cells"] = {
        "regression": bug_cells,
        "fp": s3_cells,
        "improvement": bug_cells + s3_cells,
        "downgrade": bug_cells,
    }
    all_cells = {(c["case"], c["arm"], c["repeat"]): c for c in bug_cells + s3_cells}
    with_hits = [c for c in all_cells.values() if c["hits"]]
    result["audit_summary"] = {
        "cells": len(all_cells),
        "cells_with_hits": len(with_hits),
        "orientation_only_cells": sum(1 for c in with_hits if c["orientation_only"]),
        "unaudited_cells": sum(1 for c in all_cells.values() if c["coverage"] == "unaudited"),
        "coverage": audit_hits.get("coverage", {}),
    }
    result["audit_vacuous"] = bool(audit_hits.get("vacuous", False))
    result["definitive"] = True
    result["label"] = f"{result['verdict']} (DEFINITIVE)"
    return result


def gates(
    grades: dict,
    runs: "list[dict]",
    control: str,
    candidate: str,
    audit_hits: "Optional[dict]" = None,
) -> "dict[str, Any]":
    """Apply the pre-registered gates in precedence order (0)-(7).

    (0) validate; (1) UNDETERMINED on any compared (case, arm) with fewer OK
    repeats than scheduled (INFRA exposure; void cells do not count here);
    (2) void/ungradeable marking from ``leak: confirmed``; (3) NO-GO on any
    regression (control RELIABLE, candidate MISSED, ANY bug); (4) NO-GO on FP
    excess; (5) PARITY if >= 2 reliable-to-unstable downgrades; (6) GO if >= 1
    strict improvement (candidate RELIABLE where control MISSED, or strictly
    fewer FPs); (7) PARITY. ``contested`` lists ONLY void/ungradeable cells by
    gate; the audit hit list is informational (``attach_audit``).
    """
    arms_present = {rr["config_id"] for rr in runs}
    for arm, role in ((control, "control"), (candidate, "candidate")):
        if arm not in arms_present:
            raise ValueError(
                f"{role} arm {arm!r} has no runs in this manifest (have {sorted(arms_present)})"
            )
    validate_grades(grades, runs)  # (0)

    reps = ok_repeats(runs)
    scheduled: "dict[tuple, set]" = {}
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

    voids = void_cells(grades)  # (2)
    status = catch_status(grades, runs, voids)
    fps, fp_texts = fp_counts(grades, runs, voids)
    _a, bugs, strata, _s = _runs_index(runs)

    def _s(case: str, bug: str, arm: str) -> str:
        return status.get((case, bug, arm), MISSED)

    pairs = sorted({(c, b) for (c, b, _arm) in status})
    regressions = [
        {"case": c, "bug_id": b}
        for (c, b) in pairs
        if _s(c, b, control) == RELIABLE and _s(c, b, candidate) == MISSED
    ]
    downgrades = [
        {"case": c, "bug_id": b}
        for (c, b) in pairs
        if _s(c, b, control) == RELIABLE and _s(c, b, candidate) == UNSTABLE
    ]
    improvements = [
        {"case": c, "bug_id": b}
        for (c, b) in pairs
        if _s(c, b, candidate) == RELIABLE and _s(c, b, control) == MISSED
    ]
    judgment_flags = [
        {"bug": f"{c}|{b}", "kind": "b_unstable_e_missed"}
        for (c, b) in pairs
        if _s(c, b, control) == UNSTABLE and _s(c, b, candidate) == MISSED
    ]
    fp_control = fps.get(control, 0)
    fp_candidate = fps.get(candidate, 0)

    # Contests: ONLY verifier-confirmed voids / ungradeable cells.
    contested: "list[dict]" = []
    void_by_case: "dict[str, list]" = {}
    for case, arm, rep in sorted(voids):
        if arm in (control, candidate):
            void_by_case.setdefault(case, []).append({"case": case, "arm": arm, "repeat": rep})
    for case, cells in sorted(void_by_case.items()):
        ungradeable = [
            f"{c}|{b}|{a}" for (c, b, a), st in status.items() if c == case and st == UNGRADEABLE
        ]
        reason = "verifier-confirmed leak voids this cell" + (
            f"; ungradeable: {sorted(ungradeable)}" if ungradeable else ""
        )
        if strata.get(case) == "s3_negative":
            contested.append({"gate": "fp", "case": case, "reason": reason, "cells": cells})
        else:
            for gate in ("regression", "improvement", "downgrade"):
                contested.append({"gate": gate, "case": case, "reason": reason, "cells": cells})

    if evidence_gaps:
        verdict = UNDETERMINED  # (1)
    elif regressions:
        verdict = NO_GO  # (3)
    elif fp_candidate > fp_control:
        verdict = NO_GO  # (4)
    elif len(downgrades) >= 2:
        verdict = PARITY  # (5)
    elif improvements or fp_candidate < fp_control:
        verdict = GO  # (6)
    else:
        verdict = PARITY  # (7)

    result: "dict[str, Any]" = {
        "control": control,
        "candidate": candidate,
        "verdict": verdict,
        "contested_gates": sorted({c["gate"] for c in contested}),
        "contested": contested,
        "voided_cells": [{"case": c, "arm": a, "repeat": r} for (c, a, r) in sorted(voids)],
        "evidence_gaps": evidence_gaps,
        "regressions": regressions,
        "downgrades": downgrades,
        "improvements": improvements,
        "judgment_flags": judgment_flags,
        "fp_control": fp_control,
        "fp_candidate": fp_candidate,
        "fp_findings": {control: fp_texts.get(control, []), candidate: fp_texts.get(candidate, [])},
        "informational": informational_counts(grades, runs),
        "catch_status": {f"{c}|{b}|{a}": s for (c, b, a), s in sorted(status.items())},
    }
    return attach_audit(result, runs, audit_hits)


__all__ = [
    "campaign_reads",
    "unblind",
    "validate_grades",
    "ok_repeats",
    "void_cells",
    "catch_status",
    "fp_counts",
    "informational_counts",
    "attach_audit",
    "gates",
    "REGISTERED_CONTRASTS",
    "GATE_NAMES",
    "RELIABLE",
    "UNSTABLE",
    "MISSED",
    "UNGRADEABLE",
    "GO",
    "NO_GO",
    "PARITY",
    "UNDETERMINED",
]
