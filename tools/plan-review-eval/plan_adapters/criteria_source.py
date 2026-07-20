"""Materialize each arm's criteria/prompt artifacts.

Two variants exist (``Config.variant``):

* ``control`` — the production plan-review workflow as it existed BEFORE this
  program changed anything. Its criteria are the historical
  ``.claude/commands/review-plan.md``, sourced via ``git show
  <pinned-sha>:<path>`` — pinned by SHA in ``config/configs.json``, never a
  committed copy, so it cannot drift and survives the file's eventual deletion
  (the blob lives in git history forever).

* ``candidate`` — the engine under test, drafted as lab artifacts in
  ``candidates/`` (criteria, reviewer prompt templates, merge+verify
  instructions, extraction prompt). Step 3 of the program promotes the winning
  configuration into a live skill; the campaign grades it first.

Templates use literal ``__TOKEN__`` placeholders (``__CRITERIA__``,
``__PLAN__``) substituted via ``str.replace`` — never ``str.format`` — so
criteria/plan text containing braces can't break substitution.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


class CriteriaSourceError(RuntimeError):
    """An arm's criteria could not be materialized (bad pin, missing file)."""


@dataclass
class ArmArtifacts:
    """Everything an arm's reviewer invocation needs, resolved to text."""

    variant: str
    criteria: str
    reviewer_prompt: str  # template with __CRITERIA__ / __PLAN__ tokens
    merge_prompt: str = ""  # dual arms only ("" for control/single)
    provenance: dict = field(default_factory=dict)


def git_show(repo_root: str, sha: str, path: str) -> str:
    """``git show <sha>:<path>`` with a clear failure message.

    Campaigns run in full local clones; a shallow checkout that lacks the pinned
    SHA fails here with an actionable message rather than a raw git error.
    """
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise CriteriaSourceError(
            f"pinned control SHA {sha} is not present in this clone "
            f"({probe.stderr.strip() or 'unknown object'}); run from a full clone "
            f"(git fetch --unshallow) or fix the pin in config/configs.json."
        )
    cp = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        raise CriteriaSourceError(
            f"git show {sha}:{path} failed: {cp.stderr.strip()} — the pinned path "
            f"does not exist at the pinned SHA; fix config/configs.json."
        )
    return cp.stdout


def _read(path: str) -> str:
    if not os.path.exists(path):
        raise CriteriaSourceError(f"candidate artifact missing: {path}")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# The control engine's spawn prompt: a faithful transcription of how the
# production workflow (CLAUDE.md "Plan Review Before Approval" + /revise-plan
# Step 2) instructs the review agent, with the criteria inlined because the
# pinned file no longer necessarily exists in the case worktree.
# NOTE: deliberately NO untrusted-data guard here — the control arm must be the
# production workflow EXACTLY as pinned (adding a guard the production prompt
# never had would change the baseline being measured). The guard is part of the
# CANDIDATE engine's treatment, in candidates/.
_CONTROL_PROMPT = """You are reviewing a Claude Code plan file as an independent reviewer.

Follow the review instructions below (Steps 2 through 5): read CLAUDE.md and
CONTRIBUTING.md for project context, read the files the plan references,
evaluate across the review dimensions, and present structured feedback.
Number each issue sequentially within its severity section (CRITICAL #1,
MEDIUM #1, etc.). Return ONLY the structured review output (from
"## Overall Assessment" through "## Summary").

The review instructions:

<review-instructions>
__CRITERIA__
</review-instructions>

The plan file under review:

<plan>
__PLAN__
</plan>
"""


def load_artifacts(
    repo_root: str,
    harness_root: str,
    control_pin: dict,
    candidate_texts: "dict[str, str] | None" = None,
    control_prompt_text: "str | None" = None,
) -> dict[str, ArmArtifacts]:
    """Resolve both variants' artifacts.

    ``control_pin`` is configs.json's ``control_criteria`` object:
    ``{"sha": ..., "path": ..., "rationale": ...}``. When ``candidate_texts``
    is given (a campaign's immutable protocol snapshot: filename -> text), the
    candidate artifacts are built from those exact bytes instead of re-reading
    disk — execution and recorded provenance then derive from ONE read, so an
    A→B→A edit between hashing and loading cannot split them. (The control
    criteria are sha-addressed via ``git show`` and immutable by construction.)
    """
    sha = (control_pin or {}).get("sha", "")
    path = (control_pin or {}).get("path", "")
    if not sha or not path:
        raise CriteriaSourceError(
            "config/configs.json must pin control_criteria.sha and .path "
            "(the pre-program review-plan.md)."
        )
    import re as _re

    if not _re.fullmatch(r"[0-9a-f]{40}", sha):
        raise CriteriaSourceError(
            f"control_criteria.sha {sha!r} must be a FULL 40-hex commit sha — an "
            f"abbreviated pin can become ambiguous as history grows."
        )
    control_criteria = git_show(repo_root, sha, path)

    cdir = os.path.join(harness_root, "candidates")

    def _text(name: str) -> str:
        if candidate_texts is not None:
            if name not in candidate_texts:
                raise CriteriaSourceError(f"protocol snapshot is missing {name}")
            return candidate_texts[name]
        return _read(os.path.join(cdir, name))

    candidate = ArmArtifacts(
        variant="candidate",
        criteria=_text("criteria.md"),
        reviewer_prompt=_text("reviewer_prompt.md"),
        merge_prompt=_text("merge_verify.md"),
        provenance={
            "source": "protocol snapshot" if candidate_texts is not None else "candidates/",
            "harness_root": harness_root,
        },
    )
    control = ArmArtifacts(
        variant="control",
        criteria=control_criteria,
        # A campaign passes the SNAPSHOTTED control-prompt text so arm A's
        # spawn framing participates in protocol identity like every other
        # artifact; None (non-campaign paths) uses the module constant.
        reviewer_prompt=(
            control_prompt_text if control_prompt_text is not None else _CONTROL_PROMPT
        ),
        merge_prompt="",  # the control workflow has no dual/merge stage
        provenance={"source": f"git show {sha}:{path}", "sha": sha, "path": path},
    )
    return {"control": control, "candidate": candidate}


def render(template: str, **tokens: str) -> str:
    """Strict single-pass ``__NAME__`` token substitution (brace-safe).

    Every token present in the TEMPLATE must have a provided value — a template
    token render was not given (the dual-arm merge-prompt bug class) raises
    instead of shipping a literal ``__CRITERIA__`` to a reviewer. Single-pass
    ``re.sub`` means substituted VALUES are never re-scanned: a plan whose text
    discusses ``__PLAN__`` cannot trip the check or be re-substituted.
    """
    import re

    values = {name.upper(): value for name, value in tokens.items()}
    wanted = set(re.findall(r"__([A-Z][A-Z_]*)__", template))
    missing = sorted(wanted - set(values))
    if missing:
        raise CriteriaSourceError(
            f"template token(s) {missing} were not provided to render() — a "
            f"literal placeholder must never reach a reviewer."
        )
    return re.sub(
        r"__([A-Z][A-Z_]*)__",
        lambda m: values.get(m.group(1), m.group(0)),
        template,
    )


__all__ = ["ArmArtifacts", "CriteriaSourceError", "git_show", "load_artifacts", "render"]
