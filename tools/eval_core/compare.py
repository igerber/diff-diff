"""Build the side-by-side A/B comparison bundle for an LLM (or human) to grade.

This replaces the old programmatic grader + metrics + gates engine. Instead of
parsing free-form review prose with regex, we emit ONE markdown document: per
case, the ground-truth bugs followed by each arm's RAW review output, plus
grading instructions that point at the real `pr_review.md` rubric. An LLM reads
it top-to-bottom and fills the caught/missed/false-positive table. Stdlib-only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Iterable

from eval_core.models import RunResult

# Order strata for a stable, sensible read order; unknown strata sort last.
_STRATUM_ORDER = {
    "s1_synthetic": 0,
    "s2_historical": 1,
    "s3_negative": 2,
    "s4_missed": 3,
}


def _render_header(config_ids: list[str], has_repeats: bool) -> str:
    """Grading header + table, sized to the configs (and repeats) actually present.

    The canonical run is A vs B at k=1, but the CLI also allows single-arm and
    k>1 runs; deriving the columns from the real config set keeps those bundles
    from being graded against a hardcoded two-arm contract."""
    cfgs = ", ".join(f"**{c}**" for c in config_ids) or "(none)"
    caught_cols = " ".join(f"{c} caught? | {c} sev |" for c in config_ids)
    header_row = "| case | bug id | expected sev | " + caught_cols + " notes |"
    # 3 fixed columns + 2 per config (caught?/sev) + 1 notes column.
    sep_row = "|" + "------|" * (3 + 2 * len(config_ids) + 1)
    repeat_note = ""
    if has_repeats:
        repeat_note = (
            "\n- Some (case, config) pairs ran multiple repeats (labeled "
            '"(repeat N)" above); judge each config\'s behavior across its repeats.'
        )
    return f"""# Codex reviewer A/B comparison

Configs reviewed for each case below: {cfgs}. Each case shows its ground-truth
bugs (what a faithful review SHOULD surface) followed by every config's RAW
review output.

## How to grade
Read each case's ground truth, then every config's raw review, and fill this
table — one row per (case, ground-truth bug), plus a row for any false positive
on a negative-control case:

{header_row}
{sep_row}

Rules:
- Severity definitions (P0–P3) and what counts as a "real issue" (P0/P1) live in
  `.github/codex/prompts/pr_review.md` — the exact rubric every config was given.
- A finding "catches" a bug if it names the same defect at the same
  location/symbol, regardless of wording or output format.
- On a **negative-control** case (marked "NO known bugs" below), any finding
  whose severity is OUTSIDE that case's "Allowed severities" line is a FALSE
  POSITIVE (severities inside the line, plus its listed known-FP topics, are
  acceptable — note some cases allow only P3).
- Note any bug that ALL configs miss (a shared blind spot) and any bug only one
  config catches (the signal that matters for the upgrade decision).{repeat_note}
"""


def _fence_for(text: str) -> str:
    """A backtick fence longer than the longest backtick run in ``text``.

    Review markdown routinely contains ``` code fences; wrapping it in a fence of
    (max-run + 1) backticks keeps the embedded fences from terminating ours.
    """
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _render_ground_truth(snap: dict) -> str:
    """Render a case's ground truth from its run-time snapshot (a plain dict)."""
    if snap.get("expect_no_blockers"):
        allowed = ", ".join(snap.get("allow_severities") or []) or "none"
        lines = [
            "**NO known bugs** (negative control) — any finding whose severity is "
            f"outside the allowed set ({allowed}) is a FALSE POSITIVE.",
            f"- Allowed severities: {allowed}",
        ]
        for topic in snap.get("known_fp_topics") or []:
            desc = topic.get("topic") or topic.get("description") or str(topic)
            if topic.get("id"):
                # Graders exempt a matching finding by citing this id — it must
                # be visible in the (blinded) bundle, not only in case.json.
                desc = f"[{topic['id']}] {desc}"
            extra = []
            if topic.get("file"):
                extra.append(f"in `{topic['file']}`")
            if topic.get("would_be_severity_if_flagged"):
                extra.append(f"would be {topic['would_be_severity_if_flagged']} if flagged")
            suffix = f" ({'; '.join(extra)})" if extra else ""
            lines.append(f"- Known-FP topic (do NOT flag): {desc}{suffix}")
        return "\n".join(lines)

    gt = snap.get("ground_truth") or []
    if not gt:
        return "_(no ground-truth bugs recorded)_"

    out = ["**Ground-truth bugs** (a faithful review should surface these):"]
    for bug in gt:
        lw = list(bug.get("line_window") or [0, 0]) + [0, 0]
        lo, hi = lw[0], lw[1]
        loc = f"`{bug.get('file', '?')}`"
        if lo or hi:
            loc += f" lines {lo}–{hi}"
        if bug.get("anchor_symbol"):
            loc += f" (symbol `{bug['anchor_symbol']}`)"
        optional = "" if bug.get("must_catch", True) else " — _optional_ (must_catch=false)"
        out.append(
            f"- **{bug.get('id', '?')}** — expected **{bug.get('expected_severity', '?')}**, "
            f"class `{bug.get('bug_class', '?')}`{optional} — {loc}"
        )
        if bug.get("class_keywords"):
            out.append(f"  - keywords: {', '.join(bug['class_keywords'])}")
        if bug.get("rationale"):
            out.append(f"  - why: {bug['rationale']}")
    return "\n".join(out)


def _render_grading_context(snap: dict) -> str:
    """Case context the grader needs beyond ground truth: the documented case notes
    and, for a re-review case, the prior review the reviewer was shown (so re-review
    scope rules can be applied from the bundle alone)."""
    lines = []
    notes = (snap.get("notes") or "").strip()
    if notes:
        lines.append(f"**Case notes (grading context):** {notes}")
    prev = (snap.get("previous_review") or "").strip()
    if prev:
        lines.append(
            "**Re-review case** — the reviewer was shown this prior review; judge the new "
            "review under re-review scope rules:"
        )
        fence = _fence_for(prev)
        lines.append(f"{fence}markdown\n{prev}\n{fence}")
    return "\n".join(lines)


def _render_review(rr: RunResult, redact_meta: bool = False) -> str:
    if redact_meta:
        # Blinded rendering: no model name, no latency (a real side channel — the
        # max-effort arm is visibly slower), no CLI string.
        label = f"### {rr.config_id} — review"
    else:
        effort = f" @ {rr.effort}" if getattr(rr, "effort", "") else ""
        label = f"### {rr.config_id} ({rr.model or 'model?'}{effort}) — review"
    if rr.repeat_idx:
        label += f" (repeat {rr.repeat_idx})"
    if not rr.ok:
        return f"{label}\n\n> INFRA_ERROR — excluded: `{rr.infra_error}`"
    md = rr.review_markdown.strip() or "_(empty review)_"
    fence = _fence_for(md)
    if redact_meta:
        return f"{label}\n\n{fence}markdown\n{md}\n{fence}"
    meta = f"latency {rr.latency_s:.0f}s · cli {rr.cli_version or '?'}"
    return f"{label}\n\n_{meta}_\n\n{fence}markdown\n{md}\n{fence}"


# --------------------------------------------------------------------------- #
# Blinded grading support
# --------------------------------------------------------------------------- #

# Family patterns scrubbed from blinded text in ADDITION to the configured model
# names: any gpt-5.x-style token, the GPT-5.6 tier names as standalone words, and
# bare major.minor version shorthand ("5.5", "5.6"). Over-redaction is safe (a
# stray [model-redacted] doesn't change whether a defect was named at a location);
# under-redaction breaks the blind.
_MODEL_FAMILY_PATTERNS = (
    r"gpt[\s._-]?5(?:\.\d+)?(?:-[a-z0-9]+)*",
    r"\b(?:sol|terra|luna)\b",
    r"\b5\.[0-9]\b",
    # Claude-family MODEL tokens only: "claude" followed by a version/family
    # tail ("claude-fable-5", "Claude 3.5", "claude_sonnet"). A bare "claude"
    # is deliberately NOT redacted: every plan-review arm runs the claude CLI,
    # so generic mentions carry no arm signal — and a broader pattern mangles
    # the very repository paths evidence quotes must preserve (`.claude/hooks/
    # check-plan-review.py`, `CLAUDE.md`), turning genuine catches into graded
    # misses. Family words alone still redact via the standalone pattern.
    r"\bclaude[\s._-]+(?:\d[a-z0-9.-]*|fable|sonnet|opus|haiku)[a-z0-9.-]*",
    r"\b(?:fable|sonnet|opus|haiku)\b",
)


def derive_blind_mapping(config_ids: list[str], salt: str) -> dict[str, str]:
    """Deterministic config_id -> blind-label permutation (labels ``M1..Mn``).

    Assignment order sorts ids by ``sha256(salt|id)``: stable across re-renders
    of the SAME experiment (the salt derives from the manifest's content-hash
    run_ids, not wall clock), different across experiments — so a grader can
    never learn "B is always M3". The ``M*`` namespace is deliberately disjoint
    from the real arm ids.
    """
    ordered = sorted(
        config_ids,
        key=lambda cid: hashlib.sha256(f"{salt}|{cid}".encode("utf-8")).hexdigest(),
    )
    return {cid: f"M{i + 1}" for i, cid in enumerate(ordered)}


def sanitize_model_refs(text: str, model_names: Iterable[str]) -> str:
    """Replace model identities in ``text`` with ``[model-redacted]``.

    Case-insensitive; configured names first (longest first, so "gpt-5.6-sol"
    wins over any "gpt-5.6" family match), then the family patterns. Reviews
    occasionally self-reference their model, and s4_missed case notes routinely
    name the model that missed the bug — both would unblind a grader.
    """
    out = text
    for name in sorted({n for n in model_names if n}, key=len, reverse=True):
        out = re.sub(re.escape(name), "[model-redacted]", out, flags=re.IGNORECASE)
    for pat in _MODEL_FAMILY_PATTERNS:
        out = re.sub(pat, "[model-redacted]", out, flags=re.IGNORECASE)
    return out


def apply_blinding(
    runs: Iterable[RunResult], mapping: dict[str, str], model_names: Iterable[str]
) -> list[RunResult]:
    """Return blinded copies of ``runs``: config ids swapped for blind labels and
    every identity-bearing field cleared or sanitized.

    Cleared: ``model``, ``effort``, ``cli_version``, ``latency_s`` (the
    max-effort arm is visibly slower — a real side channel). Sanitized:
    ``review_markdown``, ``infra_error``, and the snapshot's ``notes`` /
    ``previous_review`` (the surfaces that can name models). The originals are
    never mutated.
    """
    names = list(model_names)
    blinded: list[RunResult] = []
    for rr in runs:
        snap = dict(rr.case_snapshot or {})
        for field in ("notes", "previous_review"):
            if snap.get(field):
                snap[field] = sanitize_model_refs(str(snap[field]), names)
        blinded.append(
            dataclasses.replace(
                rr,
                config_id=mapping[rr.config_id],
                model="",
                effort="",
                cli_version="",
                latency_s=0.0,
                review_markdown=sanitize_model_refs(rr.review_markdown, names),
                infra_error=(
                    sanitize_model_refs(rr.infra_error, names) if rr.infra_error else None
                ),
                case_snapshot=snap,
            )
        )
    return blinded


def _snapshot_key(snap: dict) -> str:
    """Short hash of a case snapshot, so two VERSIONS of the same case_id (distinct
    snapshots — possible only under ``compare --allow-mixed``) render as SEPARATE
    sections instead of one review being graded against the other version's truth."""
    payload = json.dumps(snap or {}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def build_bundle(
    runs: Iterable[RunResult],
    redact_meta: bool = False,
    header_text: "str | None" = None,
) -> str:
    """Render the full comparison bundle from the stored run artifacts.

    Each run carries its own ``case_snapshot`` (the case AS REVIEWED), so the
    bundle reflects exactly what ran — never the current corpus. Only cases that
    actually have runs are rendered (so a subset run shows only its cases), and a
    later edit to the corpus can't change how an old run is presented. Groups are
    keyed by ``(case_id, snapshot)`` and ordered by stratum: under a normal
    manifest-scoped compare there's one snapshot per case_id, but ``--allow-mixed``
    can surface two versions of the same case_id, which MUST be graded separately.

    ``redact_meta`` drops the per-review model/latency/cli meta (used for blinded
    bundles — pass runs through ``apply_blinding`` first; this flag only controls
    the renderer's own meta line and label).

    ``header_text`` replaces the default (codex-reviewer) grading header, letting
    another harness supply its own grading instructions while reusing the
    grouping/rendering/blinding machinery unchanged. ``None`` keeps the default.
    """
    runs_by_group: dict[tuple[str, str], list[RunResult]] = {}
    for rr in runs:
        gkey = (rr.case_id, _snapshot_key(rr.case_snapshot or {}))
        runs_by_group.setdefault(gkey, []).append(rr)

    # case_ids that appear with >1 distinct snapshot get a variant marker so the
    # bundle never silently conflates two versions under one heading.
    versions_per_case: dict[str, set[str]] = {}
    for cid, skey in runs_by_group:
        versions_per_case.setdefault(cid, set()).add(skey)

    def _sort_key(gkey: tuple[str, str]):
        snap = runs_by_group[gkey][0].case_snapshot or {}
        return (_STRATUM_ORDER.get(snap.get("stratum", ""), 99), gkey[0], gkey[1])

    group_keys = sorted(runs_by_group, key=_sort_key)

    all_runs = [rr for grp in runs_by_group.values() for rr in grp]
    config_ids = sorted({rr.config_id for rr in all_runs})
    has_repeats = any(rr.repeat_idx for rr in all_runs)

    n_cases = len({cid for cid, _ in group_keys})
    header = header_text if header_text is not None else _render_header(config_ids, has_repeats)
    parts = [header, f"\n_{n_cases} cases._\n"]
    for gkey in group_keys:
        cid, skey = gkey
        case_runs = sorted(runs_by_group[gkey], key=lambda r: (r.config_id, r.repeat_idx))
        snap = case_runs[0].case_snapshot or {}
        parts.append("\n---\n")
        title = f" — {snap['title']}" if snap.get("title") else ""
        variant = f" · variant `{skey}`" if len(versions_per_case[cid]) > 1 else ""
        parts.append(f"## {cid}{title}{variant}\n")
        parts.append(f"_stratum: {snap.get('stratum', '?')}_\n")
        parts.append(_render_ground_truth(snap))
        ctx = _render_grading_context(snap)
        if ctx:
            parts.append("")
            parts.append(ctx)
        parts.append("")
        for rr in case_runs:
            parts.append(_render_review(rr, redact_meta=redact_meta))
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


__all__ = ["build_bundle", "derive_blind_mapping", "sanitize_model_refs", "apply_blinding"]
