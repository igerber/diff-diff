"""Core data model for the minimal Codex-reviewer comparison harness.

Plain stdlib dataclasses describing one reviewer-comparison experiment: the
arms (``Config``), a corpus case + its ground truth (``Case`` /
``GroundTruthBug``), and the persisted output of one review run (``ReviewOutput``
/ ``RunResult``). Scoring is NOT modeled here — both raw reviews are bundled
side-by-side and read by an LLM into a caught/missed/false-positive table (see
``engine.compare`` and the README).

Stratum strings are module-level constants (not Enums) so JSON round-tripping in
``engine.store`` stays trivial.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Controlled vocabularies (string constants — JSON-friendly)
# --------------------------------------------------------------------------- #

# Corpus strata (the corpus directory names under corpus/cases/). S1 = synthetic
# injection, S2 = historical real-PR replay, S3 = clean negative control, S4 =
# AI-missed bug. They label cases for the side-by-side read; the harness no
# longer weights or gates on them.
STRATUM_SYNTHETIC = "s1_synthetic"
STRATUM_HISTORICAL = "s2_historical"
STRATUM_NEGATIVE = "s3_negative"
STRATUM_MISSED = "s4_missed"
STRATA = (STRATUM_SYNTHETIC, STRATUM_HISTORICAL, STRATUM_NEGATIVE, STRATUM_MISSED)

# A run that could not be executed faithfully (worktree/codex infra failure).
# Such runs are surfaced loudly and excluded from the comparison bundle —
# NEVER presented as a missed bug (an infra failure is not a recall signal).
INFRA_ERROR = "INFRA_ERROR"


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """One reviewer configuration arm (e.g. A = control, B..D = candidates).

    Only the fields configs.json declares as ``treatment_fields`` (default:
    ``model``) may differ between arms; everything else (``sandbox``,
    ``action_version``, ``cli_version``, and ``effort`` when not declared) is a
    confound the experiment pins and the runner asserts identical across arms
    (see ``engine.runner`` for the single-field-contrast rules).
    """

    id: str  # "A", "B", ...
    model: str  # e.g. "gpt-5.5" / "gpt-5.6-sol"
    effort: str = "xhigh"
    sandbox: str = "read-only"
    action_version: str = "v1"  # openai/codex-action@<version>
    # Pinned Codex CLI version string the runner asserts at runtime. None =
    # "record whatever is on PATH but still assert A==B".
    cli_version: Optional[str] = None
    label: str = ""  # human description ("control / current production")


@dataclass
class GroundTruthBug:
    """A confirmed-real defect a faithful review should surface.

    ``class_keywords`` are resolved at corpus-load time (from the corpus's
    bug-class synonym table) and surfaced in the comparison bundle so the LLM
    reader has the vocabulary for "did the review name this class of defect".
    """

    id: str
    file: str
    line_window: tuple[int, int]  # inclusive [start, end] in the POST-diff file
    bug_class: str
    expected_severity: str
    must_catch: bool = True
    anchor_symbol: str = ""  # fallback locator when line parsing is unavailable
    class_keywords: list[str] = field(default_factory=list)
    rationale: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Case:
    """One evaluation case: a materializable diff state + its ground truth.

    The runner/reviewer treat ``fixture`` as an opaque dict — only
    ``adapters/worktree.py`` knows how to materialize it. ``expect_no_blockers``
    marks a clean negative control: any finding OUTSIDE that case's
    ``allow_severities`` (default ``["P2", "P3"]``; a calibration probe may allow
    only ``["P3"]``) is a false positive, and ``known_fp_topics`` documents
    topics that must not be flagged at all.
    """

    id: str
    stratum: str
    title: str = ""
    fixture: dict[str, Any] = field(default_factory=dict)
    ground_truth: list[GroundTruthBug] = field(default_factory=list)
    # Negative-control / precision controls:
    expect_no_blockers: bool = False  # True ⇒ any P0/P1 is a false positive
    allow_severities: list[str] = field(default_factory=lambda: ["P2", "P3"])
    known_fp_topics: list[dict[str, Any]] = field(default_factory=list)
    # Exact post-diff path contract. When set, verify() requires the diff to touch
    # EXACTLY these files — REQUIRED for negative controls (expect_no_blockers), so a
    # mis-pinned clean case can't silently become a code-changing diff that the grader
    # still treats as blocker-free. For ground-truth cases it's optional (the
    # ground_truth.file subset check applies when it's absent).
    expected_files: list[str] = field(default_factory=list)
    weight: float = 1.0  # retained so the corpus JSON loads unchanged
    notes: str = ""


# --------------------------------------------------------------------------- #
# Reviewer output + run results
# --------------------------------------------------------------------------- #


@dataclass
class ReviewOutput:
    """What a Reviewer returns for one (case, config, repeat) invocation.

    We deliberately do NOT carry structured findings: the RAW ``review_markdown``
    is the comparison input. An LLM reads both arms' raw markdown side by side —
    regex parsing of free-form, model-specific review prose is brittle and biased.
    """

    review_markdown: str
    cli_version: str = ""
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """The persisted artifact of one (case, config, repeat) review.

    ``infra_error`` is set (to a human-readable cause) when the run could not
    be executed faithfully; such results are excluded from the comparison
    bundle. The stored ``review_markdown`` is what the LLM reader sees.
    """

    case_id: str
    config_id: str
    repeat_idx: int
    review_markdown: str = ""
    cli_version: str = ""
    model: str = ""
    # The arm's reasoning effort, recorded so multi-effort experiments stay
    # readable in the (unblinded) bundle. "" on pre-effort artifacts — loading
    # old runs stays compatible.
    effort: str = ""
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    prompt_sha: str = ""  # content hash of the exact prompt the reviewer saw
    # Stable identity of this run (case/config/repeat + experiment_tag). Run
    # artifacts are keyed on this so they cannot alias across models/prompts —
    # a rerun under a changed model never silently resumes a stale review.
    run_id: str = ""
    # Snapshot of the case AS REVIEWED (title / stratum / ground_truth / negative-
    # control fields), captured at run time. ``compare`` renders from this — never
    # from the live corpus — so editing a case after a run can't silently re-grade
    # the old review against new ground truth, and subset runs render only their
    # own cases.
    case_snapshot: dict[str, Any] = field(default_factory=dict)
    infra_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.infra_error is None


# --------------------------------------------------------------------------- #
# Lightweight JSON (de)serialization helpers
# --------------------------------------------------------------------------- #


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/tuples to JSON-serializable structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def run_result_from_dict(d: dict[str, Any]) -> RunResult:
    return RunResult(
        case_id=d["case_id"],
        config_id=d["config_id"],
        repeat_idx=d["repeat_idx"],
        review_markdown=d.get("review_markdown", ""),
        cli_version=d.get("cli_version", ""),
        model=d.get("model", ""),
        effort=d.get("effort", ""),
        latency_s=d.get("latency_s", 0.0),
        usage=d.get("usage", {}),
        prompt_sha=d.get("prompt_sha", ""),
        run_id=d.get("run_id", ""),
        case_snapshot=d.get("case_snapshot", {}),
        infra_error=d.get("infra_error"),
    )
