#!/usr/bin/env python3
"""PreToolUse hook for ExitPlanMode: a plan may be approved only if a review of
EXACTLY its current content exists.

The gate is a content hash: the review file's ``plan_sha256`` frontmatter field
must equal the SHA-256 of the plan file's current bytes. This replaces the old
sentinel (``.last-reviewed``) + mtime design, whose failure modes (a global
sentinel raced by concurrent worktree sessions, ``ls -t`` fallback picking the
wrong plan, 1-second mtime granularity, the ``touch``-after-revision hack) are
all structurally gone: no shared state, no time comparison — the plan bytes
either match the reviewed bytes or they don't.

Payload contract (verified before the shell hook was deleted, two ways: a live
payload capture in a prior session, and the deployed CLI binary's own input
schema — build 2.1.215 defines ExitPlanMode's tool_input as ``plan`` "injected
by normalizeToolInput from disk" and ``planFilePath``, both OPTIONAL): stdin
JSON carries ``tool_input`` with ``plan`` (the plan text) and ``planFilePath``
(absolute path of the plan file). Their optionality in the upstream schema is
why the drift branch below fails closed.

Decision table (KEY PRESENCE, not truthiness — ``{"plan": ""}`` is a
plan-shaped payload whose path is missing, not a non-plan session):
  - neither ``plan`` nor ``planFilePath`` key in tool_input -> ALLOW
    (not a plan-file session; preserves the old behavior of allowing when no
    plans exist)
  - ``plan`` key present but ``planFilePath`` missing/empty -> DENY (fail
    closed: the signature of a harness payload-shape change; the gate must
    never silently disappear)
  - ``planFilePath`` present but unreadable -> DENY
  - nonempty payload ``plan`` text whose hash differs from the on-disk file ->
    DENY (tripwire: the harness injects the plan from disk, so a mismatch
    means the file changed under us or payload semantics changed)
  - review file missing / ``plan:`` path mismatch / ``plan_sha256`` missing or
    mismatched / frontmatter without a closing ``---`` -> DENY with the cause
  - all match -> ALLOW
  - malformed stdin (non-JSON, non-object payload, non-object tool_input) ->
    DENY with a clear reason (fail closed), still exit 0 with JSON per the
    PreToolUse protocol — never exit 2 (exit 2 reads as "hook error", not a
    deliberate block)

Frontmatter contract (stdlib has no YAML parser): review-file writers emit
``plan:`` and ``plan_sha256:`` as plain single-line scalars; this hook parses
line-based ``key: value``, tolerates optional surrounding quotes, and ignores
every other key. Paths are compared after ``os.path.realpath(expanduser(...))``
on BOTH sides (macOS ``/tmp`` -> ``/private/tmp``, mixed ``~``/absolute forms).

The plans directory derives from ``$HOME`` (tests override HOME).

THREAT MODEL (recorded decision — see DEFERRED.md "Decision record"): this
gate prevents ACCIDENTS — stale approvals, concurrent-worktree confusion,
plans edited mid-review — not attacks. A malicious local process needs none of
those races: it can simply write a matching review file itself; nothing here
verifies WHO wrote a review, and no userland hook can. Findings that
presuppose a hostile local actor are out of scope for this mechanism by
decision, not oversight.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys


def _respond_deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def _allow() -> None:
    sys.exit(0)


def _norm(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def _frontmatter_value(review_path: str, key: str) -> str:
    """Line-based ``key: value`` from the YAML frontmatter block (between the
    leading ``---`` and the next ``---``). Optional surrounding quotes stripped;
    "" when absent. A block with NO closing ``---`` is treated as malformed
    (returns "" for every key) — a truncated review header must not approve
    anything. Never raises on content — callers treat "" as missing."""
    try:
        with open(review_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return ""
    if not lines or lines[0].strip() != "---":
        return ""
    found = ""
    closed = False
    for line in lines[1:]:
        if line.strip() == "---":
            closed = True
            break
        if not found and line.startswith(f"{key}:"):
            value = line[len(key) + 1 :].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            found = value
    return found if closed else ""


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        _respond_deny(
            "check-plan-review.py could not parse the hook payload as JSON — "
            "failing closed. Update .claude/hooks/check-plan-review.py or use the "
            "rollback in CLAUDE.md 'Plan Review Before Approval'."
        )
        return
    if not isinstance(payload, dict):
        _respond_deny(
            "check-plan-review.py received a non-object hook payload — failing "
            "closed. Update .claude/hooks/check-plan-review.py or use the rollback "
            "in CLAUDE.md 'Plan Review Before Approval'."
        )
        return
    if "tool_input" not in payload:
        _respond_deny(
            "hook payload has no tool_input field at all — the PreToolUse payload "
            "shape has changed; failing closed rather than treating it as an empty "
            "input. Update .claude/hooks/check-plan-review.py or use the CLAUDE.md "
            "rollback."
        )
        return
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        _respond_deny(
            "ExitPlanMode payload's tool_input is not an object — failing closed "
            "rather than skipping the review gate on a malformed payload."
        )
        return

    # KEY PRESENCE, not truthiness: {"plan": ""} is still a plan-shaped payload
    # whose planFilePath is missing — that must hit the drift branch, not allow.
    has_plan = "plan" in tool_input
    has_path = "planFilePath" in tool_input
    plan_text = tool_input.get("plan")
    plan_file_path = tool_input.get("planFilePath")

    # Type discipline: a present-but-non-string plan (null, list, number) is a
    # malformed payload — fail closed rather than skip the consistency check.
    if has_plan and not isinstance(plan_text, str):
        _respond_deny(
            "ExitPlanMode payload 'plan' is present but not a string — malformed "
            "payload; failing closed rather than skipping the plan/file "
            "consistency check."
        )
        return

    if not has_plan and not has_path:
        _allow()  # not a plan-file session
        return
    if not has_path or not plan_file_path:
        _respond_deny(
            "ExitPlanMode payload carries a plan field but no usable planFilePath — "
            "the payload shape has changed and the review gate cannot verify the "
            "plan. Failing closed: update .claude/hooks/check-plan-review.py for the "
            "new payload, or use the rollback in CLAUDE.md 'Plan Review Before "
            "Approval'."
        )
        return

    plan_path = _norm(str(plan_file_path))
    try:
        with open(plan_path, "rb") as fh:
            plan_bytes = fh.read()
    except OSError:
        _respond_deny(
            f"Plan file {plan_path} is unreadable — cannot verify its review. " f"Failing closed."
        )
        return
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()

    # Tripwire: when the payload carries plan text (a string — even empty or
    # whitespace-only, which must match an equally-empty file), it must be the
    # SAME content as the file (the harness injects it from disk). A mismatch
    # means the file changed under us or payload semantics changed — verify
    # nothing.
    if isinstance(plan_text, str):
        if hashlib.sha256(plan_text.encode("utf-8")).hexdigest() != plan_sha:
            _respond_deny(
                f"ExitPlanMode payload plan text does not match the on-disk plan "
                f"file {plan_path} — the file changed after the payload was built, "
                f"or the payload semantics changed. Failing closed; retry the "
                f"approval (and re-review if the plan really did change)."
            )
            return

    plans_dir = os.path.join(os.path.expanduser("~"), ".claude", "plans")
    basename = os.path.basename(plan_path)
    stem = basename[:-3] if basename.endswith(".md") else basename
    # Collision-free key shared with plan_snapshot.py: canonical basename +
    # canonical-path digest, so same-basename plans in different locations can
    # never overwrite each other's approvals.
    digest = hashlib.sha256(plan_path.encode()).hexdigest()[:12]
    review_path = os.path.join(plans_dir, f"{stem}.{digest}.review.md")

    if not os.path.exists(review_path):
        _respond_deny(
            f"No plan review found for {basename}. Expected {review_path} with a "
            f"plan_sha256 matching the current plan content. Run the plan review "
            f"workflow (see CLAUDE.md 'Plan Review Before Approval') before "
            f"presenting for approval."
        )
        return

    reviewed_plan = _frontmatter_value(review_path, "plan")
    if not reviewed_plan or _norm(reviewed_plan) != plan_path:
        _respond_deny(
            f"Review file {review_path} is for a different plan "
            f"(its plan: field is {reviewed_plan or '<missing>'!r}, expected "
            f"{plan_path}). Re-run the review for this plan."
        )
        return

    reviewed_sha = _frontmatter_value(review_path, "plan_sha256")
    if not reviewed_sha:
        _respond_deny(
            f"Review file {review_path} has no plan_sha256 field — it predates the "
            f"content-hash gate (or the writer omitted it). Re-run the review; the "
            f"review step records plan_sha256 of the reviewed plan bytes."
        )
        return
    if reviewed_sha != plan_sha:
        _respond_deny(
            f"Plan review is stale: {basename} was modified after its review "
            f"(current sha256 {plan_sha[:12]}..., reviewed {reviewed_sha[:12]}...). "
            f"Re-run the review — after an intentional revision, the review step "
            f"recomputes plan_sha256 (there is no touch/mtime bypass)."
        )
        return

    _allow()


if __name__ == "__main__":
    main()
