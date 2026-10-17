#!/usr/bin/env python3
"""CLI for the minimal Codex-reviewer comparison harness.

Pipeline:

  verify-corpus   Materialize every case and assert its diff touches the
                  expected files (no codex; cheap; run in CI-lite).
  smoke           Tiny matrix (default: control arm, 1 case, k=1) end-to-end
                  to validate plumbing -- the FIRST command that calls codex.
                  Pass --limit 0 to run the whole selected corpus.
  run             Full comparison matrix over the selected arms (configs.json
                  defines them; 2..N arms); saves each arm's RAW review markdown.
  compare         Emit one side-by-side bundle (ground truth + every arm's raw
                  reviews) for an LLM (or you) to read into a caught/missed/
                  false-positive table. --blinded adds an identity-stripped
                  bundle for blind grading, sealed to the run artifacts
                  (bundle_id, run_ids_sha, artifacts_sha, rubric/prompt sha).
  audit           Post-unblinding session-log audit (reviewer_audit.py):
                  marker hits per (case, arm, repeat) with coverage classes;
                  informational, never a gate.
  verdict         Apply the pre-registered gates (reviewer_verdict.py) to a
                  reconciled grades JSON, after binding it to the sealed
                  artifacts; PROVISIONAL without --audit-hits, DEFINITIVE with.

Usage:
  python tools/reviewer-eval/run_eval.py verify-corpus
  python tools/reviewer-eval/run_eval.py smoke   # smokes the role=control arm
  python tools/reviewer-eval/run_eval.py run --subdir gpt6 --configs B,E --k 2
  python tools/reviewer-eval/run_eval.py compare --subdir gpt6 --blinded
  python tools/reviewer-eval/run_eval.py audit --subdir gpt6 -o <scratch>/hits.json
  python tools/reviewer-eval/run_eval.py verdict --subdir gpt6 --grades <reconciled.json> \\
      --control B --candidate E [--audit-hits <scratch>/hits.json] [--expect-head <sha>]
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
# Make `adapters` importable (the eval dir is the package root) and `eval_core`
# (the shared engine, directly under tools/) importable.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import reviewer_audit  # noqa: E402
from adapters import codex_reviewer as _codex_reviewer_mod  # noqa: E402
from adapters import worktree  # noqa: E402
from adapters.codex_reviewer import isolated_codex_home  # noqa: E402  (module-level: tests stub it)
from adapters.corpus_loader import CorpusLoader  # noqa: E402
from adapters.openai_review_loader import find_repo_root  # noqa: E402
from eval_core.store import RunStore, read_json, write_json  # noqa: E402

CONFIG_DIR = os.path.join(HERE, "config")
CORPUS_DIR = os.path.join(HERE, "corpus")
RUNS_DIR = os.path.join(HERE, "runs")


def _manifest_path(subdir: str) -> str:
    """Per-invocation run manifest — a SIBLING of runs/<subdir>/ so the RunStore's
    ``*.json`` glob in ``load_all`` never parses it as a RunResult."""
    return os.path.join(RUNS_DIR, f"{subdir}-manifest.json")


def _invalidate_manifest_if_exists(subdir: str) -> None:
    """If a prior manifest exists for ``subdir``, replace it with a failure marker.
    Called at the start of a run attempt so a failed/aborted rerun (even one that
    exits at input validation) can never leave a PRIOR run's manifest live for
    ``compare``. A fresh subdir is left manifest-less (a no-op misconfig fabricates
    nothing); ``compare`` already fails closed on a missing manifest."""
    path = _manifest_path(subdir)
    if os.path.exists(path):
        write_json(path, {"failed": True, "error": "superseded by an incomplete rerun"})


def _safe_subdir(subdir: str) -> bool:
    """True iff ``subdir`` resolves to a real child of RUNS_DIR. ``--subdir`` flows
    straight into filesystem paths (store dir, manifest, comparison.md), so reject
    absolute paths and ``..`` traversal that would read/write outside runs/."""
    base = os.path.realpath(RUNS_DIR)
    full = os.path.realpath(os.path.join(RUNS_DIR, subdir or ""))
    return full.startswith(base + os.sep)


# Ignored paths a registered surface legitimately carries: harness output dirs
# (the three ignore rules under tools/), bytecode FILES (never a whole
# __pycache__ dir - CorpusLoader treats any stratum subdir holding a case.json as
# a case) and Finder metadata.
_IGNORED_OK_PREFIXES = (
    "tools/reviewer-eval/runs/",
    "tools/plan-review-eval/runs/",
    "tools/plan-review-eval/corpus/cases/",
)
_IGNORED_OK_RE = __import__("re").compile(r"(__pycache__/[^/]*\.py[cod]$|(^|/)\.DS_Store$)")
_REGISTERED_SURFACES = ("tools", ".claude/scripts", ".github/codex")


def _git_out(repo_root: str, args: list) -> str:
    import subprocess

    cp = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {repo_root}: {cp.stderr.strip()}")
    return cp.stdout


def _hidden_inputs(repo_root: str) -> list:
    out = _git_out(
        repo_root,
        ["ls-files", "--others", "--ignored", "--exclude-standard", *_REGISTERED_SURFACES],
    )
    bad = []
    for line in out.splitlines():
        path = line.strip()
        if not path or path.startswith(_IGNORED_OK_PREFIXES) or _IGNORED_OK_RE.search(path):
            continue
        bad.append(path)
    return sorted(bad)


def _assert_no_hidden_inputs(repo_root: str) -> None:
    """Refuse when an IGNORED (or .git/info/exclude-hidden) file sits under a
    registered surface: the corpus loader enumerates cases by listdir, so an
    ignored extra case dir would run; an ignored tools/json.py would shadow the
    stdlib. Remedy is in the message; nothing is removed automatically."""
    bad = _hidden_inputs(repo_root)
    if bad:
        raise SystemExit(
            "IGNORED FILE(S) UNDER A REGISTERED SURFACE (tree identity cannot pin them):\n  "
            + "\n  ".join(bad)
            + "\nStop, show these paths to the user, remove or move them (never "
            "automatically), then re-run this stage."
        )


def _assert_tree_identity(repo_root: str, manifest: dict, stage: str = "") -> None:
    """Engine identity = tree identity: the working tree must be clean (tracked
    and untracked), carry no hidden inputs, and sit at the manifest's recorded
    HEAD and tree sha (a same-tree amend or force-push is refused too). Called
    at stage ENTRY and again at stage EXIT (a fresh read)."""
    where = f"{stage}: " if stage else ""
    status = _git_out(repo_root, ["status", "--porcelain", "--untracked-files=all"]).strip()
    if status:
        raise SystemExit(
            f"{where}working tree is not clean; compare/audit/verdict run only on the exact "
            f"committed tree the campaign recorded. `git status --porcelain`:\n{status}"
        )
    _assert_no_hidden_inputs(repo_root)
    live_tree = _git_out(repo_root, ["rev-parse", "HEAD^{tree}"]).strip()
    live_head = _git_out(repo_root, ["rev-parse", "HEAD"]).strip()
    if live_tree != manifest.get("repo_tree_sha"):
        raise SystemExit(
            f"{where}live tree {live_tree} != manifest repo_tree_sha "
            f"{manifest.get('repo_tree_sha')}; restore the recorded tree (or re-run the campaign)."
        )
    if live_head != manifest.get("repo_head_sha"):
        raise SystemExit(
            f"{where}live HEAD {live_head} != manifest repo_head_sha "
            f"{manifest.get('repo_head_sha')} (same tree, different commit is refused too)."
        )


def _manifest_provenance(repo_root: str) -> dict:
    """The seven-key provenance contract at manifest-write time (one call)."""
    return worktree.git_provenance(repo_root)


def _provenance_aggregates(results, manifest_prov: dict) -> dict:
    """Aggregates over the OK results' ``usage`` (read with .get, so legacy
    prompt_sha-only artifacts yield []/n/False):

    * ``run_head_shas`` sorted unique non-empty; ``run_head_missing`` count of
      missing/"" head shas;
    * ``run_dirty_any``: True if any repo_dirty is True; else None if any run has
      a head sha and repo_dirty None; else False;
    * ``run_marker_mismatch``: runs whose repo_branch / repo_common_dir differs
      from the manifest's own value or is empty/None;
    * ``run_not_isolated``: runs whose codex_home_isolated is not True OR whose
      repo_ref_names / repo_worktree_paths is None.
    """
    heads = set()
    missing = 0
    dirty_any = False
    dirty_none = False
    marker_mismatch = 0
    not_isolated = 0
    for r in results:
        if not getattr(r, "ok", True):
            continue
        u = getattr(r, "usage", None) or {}
        head = u.get("repo_head_sha")
        if head:
            heads.add(head)
        else:
            missing += 1
        if u.get("repo_dirty") is True:
            dirty_any = True
        elif head and u.get("repo_dirty") is None:
            dirty_none = True
        if (
            not u.get("repo_branch")
            or not u.get("repo_common_dir")
            or u.get("repo_branch") != manifest_prov.get("repo_branch")
            or u.get("repo_common_dir") != manifest_prov.get("repo_common_dir")
        ):
            marker_mismatch += 1
        if (
            u.get("codex_home_isolated") is not True
            or u.get("repo_ref_names") is None
            or u.get("repo_worktree_paths") is None
        ):
            not_isolated += 1
    return {
        "run_head_shas": sorted(heads),
        "run_head_missing": missing,
        "run_dirty_any": True if dirty_any else (None if dirty_none else False),
        "run_marker_mismatch": marker_mismatch,
        "run_not_isolated": not_isolated,
    }


def _provenance_failures(prov: dict) -> list:
    """Names of provenance fields in their documented FAILURE form (fail-soft
    ``""``/``None``): non-empty shas, branch and common dir; ``repo_dirty`` a
    bool; ``repo_ref_names`` and ``repo_worktree_paths`` LISTS (never None)."""
    bad = []
    for key in ("repo_head_sha", "repo_tree_sha", "repo_branch", "repo_common_dir"):
        if not prov.get(key):
            bad.append(key)
    if not isinstance(prov.get("repo_dirty"), bool):
        bad.append("repo_dirty")
    for key in ("repo_ref_names", "repo_worktree_paths"):
        if not isinstance(prov.get(key), list):
            bad.append(key)
    return bad


_PROVENANCE_KEYS = (
    "repo_head_sha",
    "repo_tree_sha",
    "repo_dirty",
    "repo_branch",
    "repo_common_dir",
    "repo_worktree_paths",
    "repo_ref_names",
)
_AGGREGATE_KEYS = (
    "run_head_shas",
    "run_head_missing",
    "run_dirty_any",
    "run_marker_mismatch",
    "run_not_isolated",
)
_MANIFEST_MARKER_KEYS = (
    "repo_branch",
    "repo_common_dir",
    "repo_worktree_paths",
    "repo_ref_names",
    "codex_home_path",
)
_BUNDLE_ID_TOKEN = "__BUNDLE_ID__"


def _render_blinded_bundle(runs, run_ids_sha: str) -> "tuple[dict, str]":
    """DETERMINISTIC blinded bundle from the sealed runs: (mapping, bundle text
    with the Bundle-ID TOKEN still in place). ``compare --blinded`` writes it;
    ``verdict`` regenerates it from the same sealed artifacts and compares, so
    the grader-visible evidence is bound to the artifacts, not only to itself."""
    from eval_core.compare import (
        apply_blinding,
        build_bundle,
        derive_blind_mapping,
        render_default_header,
        sanitize_model_refs,
    )

    config_ids = sorted({r.config_id for r in runs})
    model_names = sorted({r.model for r in runs if r.model})
    mapping = derive_blind_mapping(config_ids, run_ids_sha)
    blind_runs = apply_blinding(runs, mapping, model_names)
    # The default "How to grade" header with the BLIND labels as columns (so the
    # pr_review.md pointer survives), plus the Bundle ID line graders copy.
    has_repeats = any(r.repeat_idx > 0 for r in blind_runs)
    header_b = render_default_header(sorted(mapping.values()), has_repeats) + (
        f"\nBundle ID: `{_BUNDLE_ID_TOKEN}` - copy this verbatim into your grading "
        f"table as `bundle_id` (the verdict stage rejects a table whose bundle_id "
        f"does not match this bundle).\n"
    )
    # Belt-and-suspenders: sanitize the WHOLE rendered bundle once more, so a
    # model name that leaked through any renderer path is still scrubbed. The id
    # is the hash of the SANITIZED bytes with the token still in place.
    with_token = sanitize_model_refs(
        build_bundle(blind_runs, redact_meta=True, header_text=header_b), model_names
    )
    return mapping, with_token


def _bundle_id_of(bundle_text_with_token: str) -> str:
    """Identity of one blinded bundle = sha256 of the exact grader-visible bytes
    with the id slot tokenized (then substituted). First 16 hex."""
    import hashlib

    return hashlib.sha256(bundle_text_with_token.encode("utf-8")).hexdigest()[:16]


def _run_ids_sha(run_ids) -> str:
    import hashlib

    return hashlib.sha256("|".join(sorted(run_ids)).encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> str:
    import hashlib

    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _registered_candidates() -> set:
    """Candidate ids of the pre-registered contrasts (reviewer_verdict)."""
    import reviewer_verdict

    return {cand for _ctrl, cand in reviewer_verdict.REGISTERED_CONTRASTS}


def _candidate_exposure_violations(
    configs, k: int, k_overrides: dict, strata, command: str
) -> list:
    """Pre-registration BY CONSTRUCTION, enforced in code: a registered
    candidate arm may run on the corpus ONLY as the exact registered campaign -
    its registered contrast {control, candidate}, k == 2, no --k-per, no
    --strata filtering - and never under `smoke`. Anything else would expose
    the candidate's corpus output before the decisive run."""
    import reviewer_verdict

    selected = {c.id for c in configs}
    cands = selected & _registered_candidates()
    if not cands:
        return []
    if command == "smoke":
        return [f"candidate arm(s) {sorted(cands)} may not be smoked (first corpus run = campaign)"]
    problems = []
    for ctrl, cand in reviewer_verdict.REGISTERED_CONTRASTS:
        if cand in cands and selected != {ctrl, cand}:
            problems.append(
                f"candidate {cand} runs only as --configs {ctrl},{cand} (got {sorted(selected)})"
            )
    if k != 2:
        problems.append(f"the registered campaign is --k 2 (got --k {k})")
    if k_overrides:
        problems.append(f"no --k-per in the registered campaign (got {k_overrides})")
    if strata:
        problems.append(f"no --strata filtering in the registered campaign (got {strata})")
    return problems


def _attempts_path(subdir: str) -> str:
    """Append-only per-attempt record, a SIBLING of runs/<subdir>/ (never parsed
    as a RunResult). Each `run` attempt appends a START record before dispatch
    and an END record (per-cell infra errors, first 120 chars) after the matrix
    returns; an attempt that raised before dispatch has no END record and does
    not count toward the cap."""
    return os.path.join(RUNS_DIR, f"{subdir}-attempts.jsonl")


MAX_RERUNS = 3  # DECISION_RULE.md: at most 3 reruns after the first attempt


def _read_jsonl_strict(path: str, required: "dict[str, tuple]") -> list:
    """Append-only integrity logs FAIL CLOSED: a malformed line (a truncated or
    corrupt append) or a record missing a required field refuses with the file
    and line number, never a silent skip - a skipped record would undercount the
    rerun cap or erase the first candidate-exposure record."""
    import json as _json

    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"{path}:{lineno}: malformed record ({exc}); the log fails closed")
            if not isinstance(rec, dict):
                raise SystemExit(f"{path}:{lineno}: record is not an object; the log fails closed")
            kind = rec.get("kind")
            for field in required.get(kind, required.get("*", ())):
                if field not in rec:
                    raise SystemExit(
                        f"{path}:{lineno}: {kind!r} record missing {field!r}; the log fails closed"
                    )
            if kind not in required and "*" not in required:
                raise SystemExit(
                    f"{path}:{lineno}: unknown record kind {kind!r}; the log fails closed"
                )
            out.append(rec)
    return out


_ATTEMPT_SCHEMA = {"start": ("at", "head", "configs", "k"), "end": ("started_at", "failures")}
_LEDGER_SCHEMA = {"dispatch": ("contrast", "head", "tree", "subdir", "at")}


def _read_attempts(subdir: str) -> list:
    return _read_jsonl_strict(_attempts_path(subdir), _ATTEMPT_SCHEMA)


def _lock_path(subdir: str) -> str:
    return os.path.join(RUNS_DIR, f"{subdir}.lock")


@contextlib.contextmanager
def _subdir_lock(subdir: str):
    """Exclusive cross-process ownership of one run subdir for the whole of a
    `run` attempt (stop-condition check through manifest write). Two attempts
    into the same subdir compute identical run ids and would interleave their
    artifacts into one store that the last writer then seals; the second
    process is refused instead. `flock` is released by the kernel when the
    holder exits, so a crashed attempt leaves no stale lock. Yields True when
    held, False when another process holds it."""
    import fcntl

    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(_lock_path(subdir), "a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _append_attempt(subdir: str, record: dict) -> None:
    import json as _json

    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(_attempts_path(subdir), "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(record, sort_keys=True) + "\n")


def _ok_cells(results, store_root: str) -> list:
    """Identity + artifact hash of every OK cell of an attempt (cached and fresh
    alike), recorded in the END record so a later attempt can prove it RESUMED
    them rather than re-sampling: a previously successful cell may never be
    re-dispatched (DECISION_RULE.md: reruns reuse OK artifacts)."""
    import hashlib as _hashlib

    out = []
    for r in results:
        if not getattr(r, "ok", False) or not getattr(r, "run_id", ""):
            continue
        path = os.path.join(store_root, f"{r.run_id}.json")
        try:
            with open(path, "rb") as fh:
                digest = _hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            digest = None
        out.append(
            {
                "case": r.case_id,
                "arm": r.config_id,
                "repeat": int(r.repeat_idx),
                "run_id": r.run_id,
                "sha256": digest,
            }
        )
    return sorted(out, key=lambda c: (c["case"], c["arm"], c["repeat"]))


def _resume_integrity_violations(subdir: str, store_root: str) -> list:
    """Before a new attempt: every OK cell ANY previous END record lists (the
    cumulative set, aborted attempts included - an aborted END records the
    cells persisted before the abort) must still be present as the same
    artifact bytes. A missing or changed one means the rerun would RE-DISPATCH
    a previously successful cell (selective re-sampling), which fails closed."""
    import hashlib as _hashlib

    cumulative: dict = {}
    for rec in _read_attempts(subdir):
        if rec.get("kind") != "end":
            continue
        for cell in rec.get("ok_cells") or []:
            cumulative[cell["run_id"]] = cell  # the latest record of a cell wins
    problems = []
    for cell in cumulative.values():
        path = os.path.join(store_root, f"{cell['run_id']}.json")
        if not os.path.isfile(path):
            problems.append(
                f"OK cell {cell['case']}/{cell['arm']}/r{cell['repeat']} artifact missing ({cell['run_id']})"
            )
            continue
        with open(path, "rb") as fh:
            digest = _hashlib.sha256(fh.read()).hexdigest()
        if cell.get("sha256") and digest != cell["sha256"]:
            problems.append(
                f"OK cell {cell['case']}/{cell['arm']}/r{cell['repeat']} artifact changed since the last attempt"
            )
    return problems


def _stop_condition_violations(subdir: str) -> list:
    """The pre-registered stop conditions, checked BEFORE a new attempt runs:
    (1) at most MAX_RERUNS reruns after the first attempt (= 4 dispatched
    attempts in all); (2) a (case, arm, repeat) whose END records show a
    timeout in the last two consecutive dispatched attempts stops the campaign
    (surfaced, never silently excluded)."""
    ends = [a for a in _read_attempts(subdir) if a.get("kind") == "end"]
    problems = []
    if len(ends) >= 1 + MAX_RERUNS:
        problems.append(
            f"{len(ends)} dispatched attempts already recorded for runs/{subdir} "
            f"(1 attempt + {MAX_RERUNS} reruns is the cap); stop and surface to the user"
        )
    if len(ends) >= 2:
        last, prev = ends[-1], ends[-2]

        def _timeouts(rec):
            return {
                (f["case"], f["arm"], f["repeat"])
                for f in rec.get("failures", [])
                if "timed out" in (f.get("infra_error") or "")
            }

        both = _timeouts(last) & _timeouts(prev)
        if both:
            problems.append(
                f"cell(s) {sorted(both)} timed out in 2 consecutive attempts; stop and surface "
                f"(never silently excluded, never graded partial)"
            )
    return problems


# The pre-registered surfaces: everything the campaign executes or is judged by.
REGISTERED_SURFACES = (
    "tools/reviewer-eval",
    "tools/eval_core",
    ".claude/scripts/openai_review.py",
    ".github/codex/prompts/pr_review.md",
    "tools/notebook_md_extract.py",
)
_TIMEOUT_LINE_RE = __import__("re").compile(r"^[-+]\s*CALL_TIMEOUT_S\s*=\s*[\d.]+\s*$")


def _ledger_path() -> str:
    """Append-only CAMPAIGN ledger (campaign-global, not per subdir): one record
    per dispatch of a registered contrast, with the HEAD/tree it ran on and the
    subdir. The FIRST record for a contrast is its frozen candidate-exposure
    revision; `verdict` judges every later run against it."""
    return os.path.join(RUNS_DIR, "campaign-ledger.jsonl")


def _append_ledger(record: dict) -> None:
    import json as _json

    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(_ledger_path(), "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(record, sort_keys=True) + "\n")


def _read_ledger() -> list:
    return _read_jsonl_strict(_ledger_path(), _LEDGER_SCHEMA)


def _registered_surface_diff(repo_root: str, frozen_head: str, head: str) -> "tuple[str, bool]":
    """(``git diff --stat`` over REGISTERED_SURFACES between the two commits,
    whether the FULL content diff is exactly the single CALL_TIMEOUT_S line in
    adapters/codex_reviewer.py - the ONE pre-registered exception)."""
    stat = _git_out(
        repo_root, ["diff", "--stat", frozen_head, head, "--", *REGISTERED_SURFACES]
    ).strip()
    if not stat:
        return "", False
    full = _git_out(repo_root, ["diff", frozen_head, head, "--", *REGISTERED_SURFACES])
    files = {ln[6:] for ln in full.splitlines() if ln.startswith("+++ b/")}
    changed = [
        ln
        for ln in full.splitlines()
        if (ln.startswith("+") or ln.startswith("-"))
        and not ln.startswith("+++")
        and not ln.startswith("---")
    ]
    timeout_only = False
    if (
        files == {"tools/reviewer-eval/adapters/codex_reviewer.py"}
        and len(changed) == 2
        and all(_TIMEOUT_LINE_RE.match(ln) for ln in changed)
    ):
        removed = [float(ln.split("=", 1)[1]) for ln in changed if ln.startswith("-")]
        added = [float(ln.split("=", 1)[1]) for ln in changed if ln.startswith("+")]
        # The rule permits RAISING the ceiling only (a lower cap changes censoring).
        timeout_only = len(removed) == 1 and len(added) == 1 and added[0] > removed[0]
    return stat, timeout_only


def _frozen_ref_name(contrast: "tuple[str, str]") -> str:
    """Local git ref (never pushed; outside refs/heads and refs/tags) that
    anchors the contrast's FIRST candidate exposure to a commit. It lives in
    the repository, not under the disposable runs/ dir, so deleting the ledger
    cannot reset the pre-registration boundary."""
    return f"refs/campaign/{contrast[0]}-{contrast[1]}/frozen"


def _read_frozen_ref(repo_root: str, contrast: "tuple[str, str]") -> "Optional[str]":
    import subprocess

    cp = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", _frozen_ref_name(contrast)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        return None
    return cp.stdout.strip() or None


def _anchor_frozen_ref(repo_root: str, contrast: "tuple[str, str]", head: str) -> "Optional[str]":
    """Create the frozen ref at ``head`` if absent; return the ref's sha."""
    existing = _read_frozen_ref(repo_root, contrast)
    if existing:
        return existing
    _git_out(repo_root, ["update-ref", _frozen_ref_name(contrast), head])
    return head


def _exposure_status(
    manifest: dict, subdir: str, contrast: "tuple[str, str]", repo_root: str
) -> dict:
    """Judge this campaign against the contrast's FIRST dispatch in the ledger.

    Mechanical (pre-registration intact) iff this subdir IS the first dispatch,
    OR the HEAD moved (a voided campaign re-run into a fresh subdir) and the
    registered surfaces are untouched or changed ONLY by the CALL_TIMEOUT_S
    line. A second subdir on the SAME head is a re-exposure of the candidate;
    any other registered-surface change is "re-registered after candidate
    exposure": the verdict is still computed but never a mechanical GO.
    """
    entries = [e for e in _read_ledger() if list(e.get("contrast") or []) == list(contrast)]
    if not entries:
        return {
            "ledger_entries": 0,
            "mechanical": False,
            "protocol_error": "no campaign-ledger dispatch record for this contrast",
            "reason": "no campaign-ledger dispatch record for this contrast",
        }
    first = entries[0]
    anchor = _read_frozen_ref(repo_root, contrast)
    # Integrity of the frozen anchor itself: these are PROTOCOL ERRORS (verdict
    # refuses at (f)), not the documented non-mechanical re-exposure cases.
    protocol_error = None
    if not anchor:
        protocol_error = f"frozen ref {_frozen_ref_name(contrast)} is absent"
    elif first.get("head") != anchor:
        protocol_error = (
            f"ledger first head {first.get('head')} != frozen ref {anchor} "
            f"(ledger rewritten or reset)"
        )
    elif first.get("ledger_reset"):
        protocol_error = "ledger was reset after the frozen ref existed (exposure history lost)"
    if protocol_error:
        return {
            "ledger_entries": len(entries),
            "mechanical": False,
            "protocol_error": protocol_error,
            "reason": protocol_error,
        }
    head = manifest.get("repo_head_sha")
    status = {
        "ledger_entries": len(entries),
        "frozen_head": first.get("head"),
        "first_subdir": first.get("subdir"),
        "subdirs": sorted({e.get("subdir") for e in entries}),
        "this_subdir": subdir,
        "head_changed": head != first.get("head"),
        "surface_diff_stat": "",
        "timeout_only": False,
    }
    if subdir == first.get("subdir") and not status["head_changed"]:
        status.update(mechanical=True, reason="first dispatch of the contrast (frozen revision)")
        return status
    if not status["head_changed"]:
        status.update(mechanical=False, reason="re-exposure: a second subdir on the frozen HEAD")
        return status
    if subdir == first.get("subdir"):
        status.update(
            mechanical=False,
            reason="HEAD changed but the first subdir was reused (a voided campaign re-runs into a FRESH subdir)",
        )
        return status
    stat, timeout_only = _registered_surface_diff(repo_root, first.get("head") or "", head or "")
    status["surface_diff_stat"] = stat
    status["timeout_only"] = timeout_only
    if not stat:
        status.update(mechanical=True, reason="re-run on a new HEAD; registered surfaces untouched")
    elif timeout_only:
        status.update(
            mechanical=True,
            reason="re-run on a new HEAD; the single CALL_TIMEOUT_S exception applies",
        )
    else:
        status.update(
            mechanical=False,
            reason="re-registered after candidate exposure: registered surfaces changed since the frozen HEAD",
        )
    return status


def _load_manifest_runs(subdir: str, stage: str):
    """(manifest, OK runs scoped to the manifest's run_ids) or SystemExit."""
    manifest_path = _manifest_path(subdir)
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        raise SystemExit(f"{stage}: no readable manifest at {manifest_path}; run `run` first.")
    if not isinstance(manifest, dict) or manifest.get("failed") or not manifest.get("run_ids"):
        raise SystemExit(
            f"{stage}: the last `run` into runs/{subdir} failed/incomplete (or has no "
            f"run_ids); no valid experiment. Fix the failing case(s) and re-run `run`."
        )
    store = RunStore(os.path.join(RUNS_DIR, subdir))
    wanted = set(manifest["run_ids"])
    runs = [r for r in store.load_all() if r.run_id in wanted]
    missing = wanted - {r.run_id for r in runs}
    if missing:
        raise SystemExit(
            f"{stage}: manifest lists {len(wanted)} run(s) but {len(missing)} artifact(s) "
            f"are missing/unreadable under runs/{subdir}: {sorted(missing)[:6]}"
        )
    return manifest, runs


def _verify_cases(loader, cases) -> int:
    """Fail-closed preflight: materialize + sanity-check every selected case BEFORE
    any Codex call, so a stale/malformed case can't be reviewed and graded against
    stale ground truth. Returns the count of invalid cases (0 = all OK) and prints
    each failure. (``verify-corpus`` is the same check as a standalone command.)"""
    errors = []
    for c in cases:
        err = loader.verify(c)
        if err:
            errors.append((c.id, err))
    for case_id, err in errors[:10]:
        print(f"  CASE INVALID: {case_id}: {err}", file=sys.stderr)
    if errors:
        print(
            f"{len(errors)} of {len(cases)} selected case(s) failed validation; "
            "aborting before any codex call (run `verify-corpus` to debug).",
            file=sys.stderr,
        )
    return len(errors)


def _load_configs() -> dict:
    return read_json(os.path.join(CONFIG_DIR, "configs.json"))  # type: ignore[return-value]


# Per-arm keys configs.json may carry. Fail-closed on anything else: `effort` is
# now a TREATMENT field, so a typo'd key (e.g. "efort") silently falling back to
# the default would corrupt the experiment, not just a label.
_ARM_KEYS = {"id", "role", "model", "effort", "sandbox", "action_version", "cli_version", "label"}
# treatment_fields may only name Config fields the runner knows how to contrast
# AND the codex adapter can actually execute. sandbox/action_version are
# deliberately absent (2026-07-18): CodexReviewer hard-fails any non-read-only
# sandbox / non-v1 action_version, so declaring them would only defer the
# failure to per-run INFRA_ERRORs — extend this set together with the adapter.
_TREATABLE_FIELDS = {"model", "effort"}


def _make_configs(which: list) -> list:
    """Build Config objects for the requested arm ids from configs.json.

    Fail-closed on a malformed configs.json: missing/empty ``arms``, a duplicate
    arm id (would silently overwrite), an unknown per-arm key, a missing required
    field, or anything but EXACTLY one ``role: control`` arm all raise instead of
    quietly running a different experiment than the file describes.
    """
    from eval_core.models import Config

    raw = _load_configs()
    arms = raw.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("configs.json must define a non-empty 'arms' list")
    by_id: dict = {}
    n_controls = 0
    for c in arms:
        unknown = set(c) - _ARM_KEYS
        if unknown:
            raise ValueError(
                f"configs.json arm {c.get('id', '?')!r} has unknown key(s) {sorted(unknown)}; "
                f"allowed: {sorted(_ARM_KEYS)}"
            )
        for req in ("id", "model", "effort"):
            if not c.get(req):
                raise ValueError(f"configs.json arm {c.get('id', '?')!r} missing {req!r}")
        if c["id"] in by_id:
            raise ValueError(f"configs.json has duplicate arm id {c['id']!r}")
        if c.get("role") == "control":
            n_controls += 1
        by_id[c["id"]] = Config(
            id=c["id"],
            model=c["model"],
            effort=c["effort"],
            sandbox=c.get("sandbox", "read-only"),
            action_version=c.get("action_version", "v1"),
            cli_version=c.get("cli_version"),
            label=c.get("label", ""),
        )
    if n_controls != 1:
        raise ValueError(
            f"configs.json must declare exactly one arm with role='control' (found {n_controls})"
        )
    return [by_id[i] for i in which if i in by_id]


def _control_id() -> str:
    """The id of the sole ``role='control'`` arm (current production config).

    ``smoke`` defaults to this arm, DERIVED at run time so a control flip in
    configs.json can never leave the bare smoke command exercising a stale arm.
    Fail-closed like ``_make_configs`` (exactly one control required).
    """
    raw = _load_configs()
    arms = raw.get("arms") or []
    controls = [c.get("id") for c in arms if isinstance(c, dict) and c.get("role") == "control"]
    if len(controls) != 1 or not controls[0]:
        raise ValueError(
            f"configs.json must declare exactly one arm with role='control' "
            f"(found {len(controls)})"
        )
    return controls[0]


def _treatment_fields() -> tuple:
    """The declared treatment fields from configs.json (default: model only).

    Fail-closed: a typo'd or non-Config field aborts rather than silently
    weakening the runner's confound protection.
    """
    raw = _load_configs()
    fields = raw.get("treatment_fields", ["model"])
    if (
        not isinstance(fields, list)
        or not fields
        or not set(fields) <= _TREATABLE_FIELDS
        or len(set(fields)) != len(fields)
    ):
        raise ValueError(
            f"configs.json treatment_fields must be a non-empty, duplicate-free subset of "
            f"{sorted(_TREATABLE_FIELDS)}; got {fields!r}"
        )
    return tuple(fields)


def _resolve_configs(arg: str):
    """Parse --configs, returning the Config list or None if any id is unknown.

    Fail-closed: a typo like ``--configs Z`` (or ``A,Z``) returns None so the
    caller can abort, rather than silently running a partial/empty matrix.
    """
    requested = (arg or "").split(",")
    # Reject malformed/empty selectors ("", "A,", ",A", "A,,B") rather than silently
    # dropping empty segments and running a narrower matrix than the operator intended.
    if any(c == "" for c in requested):
        return None
    # Reject duplicate ids ("A,A"): run identity, artifacts, and bundle columns key
    # off config_id, so a repeat would alias both arms onto one identity and collapse
    # the comparison rather than running two arms.
    if len(set(requested)) != len(requested):
        return None
    configs = _make_configs(requested)
    if not configs or len(configs) != len(requested):
        return None
    return configs


def _bad_configs_msg(arg: str) -> str:
    return (
        f"invalid --configs {arg!r}: unknown, empty, or duplicate config id(s) "
        "(valid ids come from config/configs.json, e.g. A,B)"
    )


def _parse_k_per(arg, configs):
    """Parse ``--k-per "C=1,D=1"`` into {config_id: k}, or None on any error.

    Fail-closed like ``_resolve_configs``: a malformed entry, duplicate id, an id
    not among the RESOLVED --configs, or k < 1 rejects the whole invocation
    rather than silently running a different matrix than the operator intended.
    """
    if not arg:
        return {}
    valid_ids = {c.id for c in configs}
    overrides: dict = {}
    for part in arg.split(","):
        cid, sep, val = part.partition("=")
        if not sep or not cid or cid in overrides or cid not in valid_ids:
            return None
        try:
            k = int(val)
        except ValueError:
            return None
        if k < 1:
            return None
        overrides[cid] = k
    return overrides


def cmd_verify_corpus(args: argparse.Namespace) -> int:
    repo_root = find_repo_root()
    loader = CorpusLoader(CORPUS_DIR, repo_root)
    cases = loader.load_cases(args.strata)
    if not cases:
        print("no cases found", file=sys.stderr)
        return 1
    failures = 0
    for case in cases:
        err = loader.verify(case)
        status = "OK" if err is None else f"FAIL: {err}"
        if err is not None:
            failures += 1
        print(f"[{status}] {case.id} ({case.stratum})")
    print(f"\n{len(cases) - failures}/{len(cases)} cases verified.")
    return 1 if failures else 0


def _build_reviewer(repo_root: str):
    from adapters.codex_reviewer import CodexReviewer

    return CodexReviewer(repo_root=repo_root, runs_root=RUNS_DIR)


def cmd_smoke(args: argparse.Namespace) -> int:
    from eval_core.runner import run_matrix

    repo_root = find_repo_root()
    loader = CorpusLoader(CORPUS_DIR, repo_root)
    cases = loader.load_cases(args.strata)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(
            f"no cases selected (strata={args.strata}, limit={args.limit}); nothing to run",
            file=sys.stderr,
        )
        return 1
    # Bare `smoke` exercises the CURRENT production arm (the sole role=control
    # entry), derived at run time so a control flip can't drift the default.
    configs_arg = args.configs if args.configs is not None else _control_id()
    configs = _resolve_configs(configs_arg)
    if configs is None:
        print(_bad_configs_msg(configs_arg), file=sys.stderr)
        return 1
    exposure = _candidate_exposure_violations(configs, args.k, {}, args.strata, "smoke")
    if exposure:
        print("refusing (pre-registration): " + "; ".join(exposure), file=sys.stderr)
        return 1
    if _verify_cases(loader, cases):
        return 1
    store = RunStore(os.path.join(RUNS_DIR, "smoke"))
    # smoke is a LIVE plumbing/auth/connectivity check (the README's "first real codex
    # call"), so it must actually exercise codex every time — clear any cached smoke
    # artifacts so run_matrix never resumes a stale success.
    for name in os.listdir(store.root):
        if name.endswith(".json"):
            os.remove(os.path.join(store.root, name))
    # Every harness codex call runs under an isolated CODEX_HOME that mirrors CI's
    # codex-action home (auth + sandbox/approval only; no MCP, hooks, memories).
    with isolated_codex_home():
        reviewer = _build_reviewer(repo_root)
        print(f"codex CLI version: {reviewer.cli_version()}")
        results = run_matrix(
            cases,
            configs,
            reviewer,
            store,
            k=args.k,
            max_parallel=args.max_parallel,
            progress=lambda m: print(f"  {m}"),
            treatment_fields=_treatment_fields(),
        )
    ok = sum(1 for r in results if r.ok)
    print(f"\nsmoke: {ok}/{len(results)} runs ok")
    for r in results:
        tag = "OK" if r.ok else f"INFRA_ERROR ({r.infra_error})"
        print(f"  {r.case_id} {r.config_id} r{r.repeat_idx}: {r.latency_s:.0f}s [{tag}]")
    return 0 if ok == len(results) else 2


def cmd_run(args: argparse.Namespace) -> int:
    if not _safe_subdir(args.subdir):
        print(
            f"invalid --subdir {args.subdir!r}: must stay within the runs directory",
            file=sys.stderr,
        )
        return 1
    # One attempt per subdir at a time, held through the manifest write: a
    # concurrent `run` into the same subdir would write the same run ids into
    # the same store (an artifact set mixed from two invocations that the last
    # one seals and verdicts).
    with _subdir_lock(args.subdir) as held:
        if not held:
            print(
                f"refusing (concurrent run): another `run` holds {_lock_path(args.subdir)}; "
                "one attempt per subdir at a time - wait for it or use a different --subdir",
                file=sys.stderr,
            )
            return 1
        return _cmd_run_locked(args)


def _cmd_run_locked(args: argparse.Namespace) -> int:
    from eval_core.runner import run_matrix

    # Pre-registered stop conditions FIRST, before anything on disk is touched: a
    # refused fifth attempt must leave the fourth attempt's successful manifest
    # (and its sealed artifacts) intact for compare/audit/verdict.
    stops = _stop_condition_violations(args.subdir)
    if stops:
        print("refusing (stop condition): " + "; ".join(stops), file=sys.stderr)
        return 1
    resample = _resume_integrity_violations(args.subdir, os.path.join(RUNS_DIR, args.subdir))
    if resample:
        print(
            "refusing (resume integrity): a rerun must reuse every previously OK artifact; "
            + "; ".join(resample),
            file=sys.stderr,
        )
        return 1
    repo_root = find_repo_root()
    # Provenance PREFLIGHT, also before the manifest is touched and before any
    # codex call: a dirty tree or an incomplete git sample would spend the
    # candidate's first corpus exposure on a campaign the post-run gate rejects
    # (and resume would keep reusing the repo_dirty=True artifacts).
    preflight = _manifest_provenance(repo_root)
    bad_prov = _provenance_failures(preflight)
    if bad_prov or preflight.get("repo_dirty") is not False:
        why = f"incomplete git provenance: {bad_prov}" if bad_prov else "working tree is dirty"
        print(
            f"refusing (provenance preflight): {why}; a campaign runs only on a clean, "
            f"fully sampled tree (`git status --porcelain --untracked-files=all` must be empty).",
            file=sys.stderr,
        )
        return 1
    # A run attempt into <subdir> supersedes any prior experiment there: invalidate an
    # EXISTING manifest now, so even an early exit (bad configs, zero cases, a corpus-
    # load error) can't leave the previous run's manifest live for compare. A fresh
    # subdir stays manifest-less here; compare fails closed on a missing manifest.
    _invalidate_manifest_if_exists(args.subdir)
    loader = CorpusLoader(CORPUS_DIR, repo_root)
    cases = loader.load_cases(args.strata)
    if not cases:
        print(f"no cases selected (strata={args.strata}); nothing to run", file=sys.stderr)
        return 1
    configs = _resolve_configs(args.configs)
    if configs is None:
        print(_bad_configs_msg(args.configs), file=sys.stderr)
        return 1
    k_overrides = _parse_k_per(getattr(args, "k_per", ""), configs)
    if k_overrides is None:
        print(
            f"invalid --k-per {args.k_per!r}: expected 'ID=N[,ID=N...]' with each ID "
            f"among the resolved --configs, no duplicates, and N >= 1",
            file=sys.stderr,
        )
        return 1
    exposure = _candidate_exposure_violations(configs, args.k, k_overrides, args.strata, "run")
    if exposure:
        print("refusing (pre-registration): " + "; ".join(exposure), file=sys.stderr)
        return 1
    # Past input validation, this is a real run attempt: write the failure marker
    # now and overwrite it with run_ids ONLY on success. Every failure from here on
    # (an invalid case, reviewer/runner error, an interrupt, or an infra failure)
    # then leaves the subdir in a failed state that `compare` refuses — a prior
    # run's manifest is never exposed as the current experiment. (An EXISTING
    # manifest was already superseded by _invalidate_manifest_if_exists at the top
    # of this attempt — even input-validation exits above supersede it; only a
    # FRESH subdir stays manifest-less through input validation.)
    write_json(_manifest_path(args.subdir), {"failed": True, "error": "run did not complete"})
    if _verify_cases(loader, cases):
        return 1
    # An ignored/excluded extra case dir under corpus/cases/ would be enumerated
    # by CorpusLoader; refuse BEFORE any codex call (shared with the tree-identity
    # guard the post-run stages apply).
    _assert_no_hidden_inputs(repo_root)
    store = RunStore(os.path.join(RUNS_DIR, args.subdir))
    # If run_matrix raises (ConfoundMismatch, CLIVersionMismatch, an interrupt, ...)
    # the up-front failure marker stays in place and the exception propagates.
    # The reviewer, the matrix AND the manifest write run under ONE isolated
    # CODEX_HOME (CI-faithful; its path is a manifest field and an audit marker).
    import datetime as _dt

    attempt_start = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _append_attempt(
        args.subdir,
        {
            "kind": "start",
            "at": attempt_start,
            "head": preflight.get("repo_head_sha", ""),  # the validated preflight sample
            "configs": [c.id for c in configs],
            "k": args.k,
        },
    )
    dispatched = {"v": False}
    planned: list = []  # the run keys of THIS attempt's matrix (from the runner's plan)
    selected_ids = {c.id for c in configs}
    import reviewer_verdict as _rv

    registered = [c for c in _rv.REGISTERED_CONTRASTS if set(c) == selected_ids]

    def _on_dispatch() -> None:
        dispatched["v"] = True
        if registered:
            # Campaign-global exposure record: the FIRST entry for a contrast is
            # its frozen revision; verdict judges every later run against it. The
            # frozen ref (in .git, not under runs/) anchors that revision; a ledger
            # with no entries while the ref already exists was reset -> recorded.
            contrast = registered[0]
            prior = [e for e in _read_ledger() if list(e.get("contrast") or []) == list(contrast)]
            existing_ref = _read_frozen_ref(repo_root, contrast)
            _anchor_frozen_ref(repo_root, contrast, preflight.get("repo_head_sha", ""))
            _append_ledger(
                {
                    "kind": "dispatch",
                    "contrast": list(contrast),
                    "head": preflight.get("repo_head_sha", ""),
                    "tree": preflight.get("repo_tree_sha", ""),
                    "subdir": args.subdir,
                    "at": attempt_start,
                    "ledger_reset": bool(existing_ref and not prior),
                }
            )

    with isolated_codex_home():
        reviewer = _build_reviewer(repo_root)
        try:
            results = run_matrix(
                cases,
                configs,
                reviewer,
                store,
                k=args.k,
                max_parallel=max(1, args.max_parallel),
                progress=lambda m: print(f"  {m}"),
                treatment_fields=_treatment_fields(),
                k_overrides=k_overrides,
                on_dispatch=_on_dispatch,
                on_plan=planned.extend,
            )
        except BaseException as exc:  # noqa: BLE001 - classify by dispatch, not by type
            if not dispatched["v"]:
                # Raised during PLANNING (CLI pin, confounds, experiment/case tags,
                # resume verification): no codex call was made, so no END record
                # and the attempt does not count toward the cap (DECISION_RULE.md).
                raise
            # Codex calls may have started: record the attempt as ABORTED so it
            # counts toward the rerun cap, WITH every OK cell this attempt holds
            # ON DISK for its planned keys (cached or persisted an instant before
            # the abort - the store, not an in-memory callback, is the truth), so
            # the resume-integrity check protects them like a completed attempt's.
            held = [a for a in (store.load(k) for k in planned) if a is not None and a.ok]
            _append_attempt(
                args.subdir,
                {
                    "kind": "end",
                    "started_at": attempt_start,
                    "aborted": type(exc).__name__,
                    "ok": None,
                    "total": None,
                    "ok_cells": _ok_cells(held, store.root),
                    "failures": [],
                },
            )
            raise
        codex_home_path = _codex_reviewer_mod.ACTIVE_ISOLATED_HOME
        ok = sum(1 for r in results if r.ok)
        infra = [r for r in results if not r.ok]
        if not infra:
            provenance = _manifest_provenance(repo_root)
    # Append the END record (the per-attempt failure record of the stop
    # conditions) BEFORE anything can overwrite the artifacts - but ONLY for a
    # dispatched attempt: a fully cached resume made no codex call and must not
    # consume the rerun cap.
    if dispatched["v"]:
        _append_attempt(
            args.subdir,
            {
                "kind": "end",
                "started_at": attempt_start,
                "ok": ok,
                "total": len(results),
                "ok_cells": _ok_cells(results, store.root),
                "failures": [
                    {
                        "case": r.case_id,
                        "arm": r.config_id,
                        "repeat": r.repeat_idx,
                        "infra_error": (r.infra_error or "")[:120],
                    }
                    for r in infra
                ],
            },
        )
    if infra:
        # Fail closed: an INFRA_ERROR (a notebook case, or a codex/worktree failure)
        # means the experiment is incomplete. Replace the up-front marker with a
        # detailed failure marker (no run_ids) so `compare` refuses, and exit
        # non-zero. The OK runs stay cached, so a re-run resumes them and only
        # retries the failures.
        for r in infra[:10]:
            print(
                f"  INFRA_ERROR: {r.case_id} {r.config_id} r{r.repeat_idx}: {r.infra_error}",
                file=sys.stderr,
            )
        write_json(
            _manifest_path(args.subdir),
            {"failed": True, "n_infra_errors": len(infra), "configs": [c.id for c in configs]},
        )
        print(
            f"\n{ok}/{len(results)} runs ok, {len(infra)} FAILED — manifest marked "
            "incomplete. Fix the failing case(s) and re-run (ok runs resume from cache).",
            file=sys.stderr,
        )
        return 2
    # Manifest scopes `compare` to THIS invocation's runs, so a later rerun into the
    # same subdir can't silently mix experiments into one bundle. It also records the
    # rubric provenance (base_prompt_sha) so compare can refuse if pr_review.md drifts
    # between run and compare (the grader is pointed at the live rubric).
    run_ids = [r.run_id for r in results if r.run_id]
    incomplete = _provenance_failures(provenance)
    if incomplete:
        # A write-time git failure would blank the computed audit markers; fail
        # closed (failure marker, exit 2) so the operator re-invokes the attempt
        # block - resume reuses every OK artifact, no codex call is repeated.
        write_json(
            _manifest_path(args.subdir),
            {"failed": True, "error": f"incomplete write-time provenance: {incomplete}"},
        )
        print(
            f"\n{ok}/{len(results)} runs ok but the write-time git provenance is incomplete "
            f"({', '.join(incomplete)}); manifest marked failed - re-run `run` (resume reuses "
            f"the OK artifacts).",
            file=sys.stderr,
        )
        return 2
    # Seal the artifacts AT RUN TIME: compare --blinded / audit / verdict refuse if
    # the live artifacts no longer hash to this value.
    sealed = reviewer_audit.artifacts_sha(store.root, run_ids)
    manifest = {
        "run_ids": run_ids,
        "configs": [c.id for c in configs],
        "base_prompt_sha": getattr(reviewer, "base_prompt_sha", ""),
        # Provenance for humans reading the manifest (compare derives repeats
        # from the artifacts themselves).
        "k": args.k,
        "k_per": k_overrides,
        "artifacts_sha": sealed,
        "codex_home_path": codex_home_path,
    }
    manifest.update({k: provenance.get(k) for k in _PROVENANCE_KEYS})
    manifest.update(_provenance_aggregates(results, provenance))
    write_json(_manifest_path(args.subdir), manifest)
    print(f"\n{ok}/{len(results)} runs ok. Next: compare --subdir {args.subdir}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Emit the side-by-side bundle for an LLM (or human) to grade.

    Renders ONLY from the stored run artifacts under runs/<subdir> (each carries
    its own as-reviewed case snapshot), scoped to the last `run` invocation via its
    manifest. Writes runs/<subdir>/comparison.md. No corpus reload, no scoring.
    """
    import hashlib

    from adapters import ci_prompt
    from eval_core.compare import build_bundle

    if not _safe_subdir(args.subdir):
        print(
            f"invalid --subdir {args.subdir!r}: must stay within the runs directory",
            file=sys.stderr,
        )
        return 1
    blinded = getattr(args, "blinded", False)
    if blinded and getattr(args, "allow_mixed", False):
        # Blinding's label permutation is salted from the manifest's run_ids;
        # --allow-mixed exists precisely for the no-manifest case, so the combo
        # has no stable identity to blind against. Refuse rather than improvise.
        print(
            "--blinded requires a manifest-scoped experiment; drop --allow-mixed.", file=sys.stderr
        )
        return 1
    runs = RunStore(os.path.join(RUNS_DIR, args.subdir)).load_all()
    if not runs:
        print(f"no runs found under runs/{args.subdir}", file=sys.stderr)
        return 1

    # Scope to the last `run` invocation via its manifest, so a subdir that
    # accumulated runs from multiple experiments yields ONE clean A/B bundle.
    manifest_path = _manifest_path(args.subdir)
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        manifest = None
    if isinstance(manifest, dict) and manifest.get("failed"):
        # The last `run` into this subdir failed/incomplete (its manifest is a
        # failure marker, not a run_ids set). Refuse rather than fall back to the
        # "no manifest -> compare ALL" path, which would silently surface leftover
        # runs as if they were a valid experiment.
        print(
            f"the last `run` into runs/{args.subdir} failed/incomplete — no valid "
            f"experiment to compare. Fix the failing case(s) and re-run `run` first.",
            file=sys.stderr,
        )
        return 1
    # Rubric provenance: the bundle points graders at the LIVE pr_review.md as "the
    # exact rubric every config was given". Fail closed if it drifted since the run,
    # or we'd grade stored reviews under a different standard.
    stored_rubric = manifest.get("base_prompt_sha") if isinstance(manifest, dict) else None
    if stored_rubric:
        live_rubric = hashlib.sha256(
            ci_prompt.read_current_prompt(find_repo_root()).encode("utf-8")
        ).hexdigest()[:16]
        if live_rubric != stored_rubric:
            print(
                f"the review rubric (.github/codex/prompts/pr_review.md) changed since this "
                f"run (stored {stored_rubric}, live {live_rubric}); compare would grade the "
                f"stored reviews under a different rubric. Re-run `run`, or restore the prompt.",
                file=sys.stderr,
            )
            return 1
    wanted = set(manifest.get("run_ids", [])) if isinstance(manifest, dict) else None
    if blinded and wanted is None:
        print(
            "--blinded requires a completed `run` manifest (its run_ids salt the "
            "blind-label permutation); re-run `run` first.",
            file=sys.stderr,
        )
        return 1
    if wanted is None:
        # No manifest: a completed `run` always writes one, so this means the subdir
        # holds legacy/manually-accumulated runs. Fail closed (one run = one
        # experiment) unless the operator explicitly opts into a mixed bundle.
        if not getattr(args, "allow_mixed", False):
            print(
                f"no manifest at {os.path.basename(manifest_path)} for runs/{args.subdir}; "
                f"refusing to compare (one run = one experiment). Re-run `run`, or pass "
                f"--allow-mixed to compare ALL runs in the subdir.",
                file=sys.stderr,
            )
            return 1
        print(
            f"  WARNING: --allow-mixed: comparing ALL runs under runs/{args.subdir} "
            f"(may mix experiments).",
            file=sys.stderr,
        )
    else:
        no_id = [r for r in runs if not r.run_id]
        if no_id:
            print(
                f"  warning: {len(no_id)} run(s) have no run_id; excluded from the "
                "manifest-scoped bundle",
                file=sys.stderr,
            )
        runs = [r for r in runs if r.run_id in wanted]
        # Fail closed on incomplete coverage: load_all silently drops missing/corrupt
        # artifacts, so a bundle built from a subset would bias the A/B read. Every
        # run_id the manifest promised must actually be present.
        missing = wanted - {r.run_id for r in runs}
        if missing:
            print(
                f"manifest lists {len(wanted)} run(s) but {len(missing)} artifact(s) are "
                f"missing/unreadable under runs/{args.subdir}: {sorted(missing)}; re-run `run`.",
                file=sys.stderr,
            )
            return 1
        if not runs:
            print(
                f"manifest matched no runs under runs/{args.subdir}; re-run `run`.",
                file=sys.stderr,
            )
            return 1
    if blinded:
        # Engine identity = tree identity, then the run-time seal (both AFTER the
        # --allow-mixed / missing-manifest refusals above).
        repo_root_live = find_repo_root()
        _assert_tree_identity(repo_root_live, manifest, "compare --blinded (stage entry)")
        sealed = manifest.get("artifacts_sha") if isinstance(manifest, dict) else None
        if sealed:
            live_sha = reviewer_audit.artifacts_sha(
                os.path.join(RUNS_DIR, args.subdir), sorted(wanted or ())
            )
            if live_sha != sealed:
                print(
                    f"run artifacts changed since `run` sealed them: live artifacts_sha "
                    f"{live_sha} != manifest {sealed}; refusing to build a blinded bundle.",
                    file=sys.stderr,
                )
                return 1
    bundle = build_bundle(runs)
    out_dir = os.path.join(RUNS_DIR, args.subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(bundle)
    ok = sum(1 for r in runs if r.ok)
    n_cases = len({r.case_id for r in runs})
    print(f"wrote {out_path} ({ok}/{len(runs)} runs ok across {n_cases} cases)")
    if blinded:

        run_ids_sha = _run_ids_sha(wanted or ())
        mapping, with_token = _render_blinded_bundle(runs, run_ids_sha)
        bundle_id = _bundle_id_of(with_token)
        blind_bundle = with_token.replace(_BUNDLE_ID_TOKEN, bundle_id)
        blind_path = os.path.join(out_dir, "comparison.blinded.md")
        with open(blind_path, "w", encoding="utf-8") as fh:
            fh.write(blind_bundle)
        repo_root_live = find_repo_root()
        rubric_path = os.path.join(HERE, "GRADING_RUBRIC.md")
        prompt_path = os.path.join(repo_root_live, ".github", "codex", "prompts", "pr_review.md")
        # The seal: graders must never read this file (or config/configs.json).
        # Unblind the finished grading table by joining on these labels.
        write_json(
            os.path.join(out_dir, "blinding.json"),
            {
                "mapping": mapping,
                "models": {r.config_id: r.model for r in runs},
                "efforts": {r.config_id: getattr(r, "effort", "") for r in runs},
                "salt_source": "sha256 of the sorted manifest run_ids",
                "bundle_id": bundle_id,
                "run_ids_sha": run_ids_sha,
                "artifacts_sha": (
                    manifest.get("artifacts_sha") if isinstance(manifest, dict) else None
                ),
                "rubric_sha256": _file_sha256(rubric_path) if os.path.exists(rubric_path) else None,
                "prompt_sha256": _file_sha256(prompt_path) if os.path.exists(prompt_path) else None,
            },
        )
        print(
            f"wrote {blind_path} (blind labels: {', '.join(sorted(mapping.values()))}; bundle_id {bundle_id})"
        )
        _assert_tree_identity(repo_root_live, manifest, "compare --blinded (stage exit)")
        print(
            "  BLINDING: give graders ONLY comparison.blinded.md; they must not read "
            "blinding.json or config/configs.json. Unblind finished tables via blinding.json."
        )
    print(
        "Next: have an LLM (a subagent or in-conversation) read it into the "
        "caught/missed/false-positive table, then decide."
    )
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Post-unblinding session-log audit -> hits.json (informational)."""
    if not _safe_subdir(args.subdir):
        print(
            f"invalid --subdir {args.subdir!r}: must stay within the runs directory",
            file=sys.stderr,
        )
        return 1
    manifest, runs = _load_manifest_runs(args.subdir, "audit")
    repo_root = find_repo_root()
    _assert_tree_identity(repo_root, manifest, "audit (stage entry)")
    markers = {k: manifest.get(k) for k in _MANIFEST_MARKER_KEYS}
    try:
        result = reviewer_audit.audit(
            runs,
            os.path.join(RUNS_DIR, args.subdir),
            repo_root,
            markers,
            _run_ids_sha(manifest["run_ids"]),
            manifest.get("artifacts_sha") or "",
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"audit refused: {exc}", file=sys.stderr)
        return 1
    write_json(args.output, result)
    cov = result["coverage"]
    print(
        f"wrote {args.output}: {len(result['hits'])} hit(s); coverage covered={cov['covered']} "
        f"no_commands={cov['no_commands']} unaudited={cov['unaudited']} total={cov['total']}"
        + ("; AUDIT VACUOUS" if result["vacuous"] else "")
    )
    _assert_tree_identity(repo_root, manifest, "audit (stage exit)")
    return 0


def cmd_verdict(args: argparse.Namespace) -> int:
    """Pinned check order: (a) arguments, (b) manifest, (c) blinding.json binding,
    (d) grades + bundle_id + validation, (e) --audit-hits binding, (f) protocol
    guards, (g) tree identity, (h) the gates."""
    import json as _json

    import reviewer_verdict
    from eval_core.compare import derive_blind_mapping
    from eval_core.models import to_jsonable

    # (a)
    if not _safe_subdir(args.subdir):
        print(f"invalid --subdir {args.subdir!r}", file=sys.stderr)
        return 1
    if args.control == args.candidate:
        raise SystemExit("verdict: --control and --candidate must differ")
    if (args.control, args.candidate) not in reviewer_verdict.REGISTERED_CONTRASTS:
        raise SystemExit(
            f"verdict: ({args.control}, {args.candidate}) is not the registered contrast "
            f"{reviewer_verdict.REGISTERED_CONTRASTS}; a future campaign registers its own "
            f"pair in its own PR-1."
        )
    # (b)
    manifest, runs = _load_manifest_runs(args.subdir, "verdict")
    repo_root = find_repo_root()
    subdir_path = os.path.join(RUNS_DIR, args.subdir)
    run_ids_sha = _run_ids_sha(manifest["run_ids"])
    # (c)
    blinding_path = os.path.join(subdir_path, "blinding.json")
    try:
        blinding = read_json(blinding_path)
    except (OSError, ValueError):
        raise SystemExit(f"verdict (c): no readable blinding.json at {blinding_path}")
    assert isinstance(blinding, dict)
    expected_mapping = derive_blind_mapping(sorted({r.config_id for r in runs}), run_ids_sha)
    if blinding.get("mapping") != expected_mapping:
        raise SystemExit(
            "verdict (c): blinding.json mapping != deterministic mapping from the manifest"
        )
    if blinding.get("run_ids_sha") != run_ids_sha:
        raise SystemExit("verdict (c): blinding.json run_ids_sha != manifest run_ids")
    live_artifacts = reviewer_audit.artifacts_sha(subdir_path, manifest["run_ids"])
    if manifest.get("artifacts_sha") != live_artifacts:
        raise SystemExit(
            f"verdict (c): manifest artifacts_sha {manifest.get('artifacts_sha')} != live {live_artifacts}"
        )
    if blinding.get("artifacts_sha") != live_artifacts:
        raise SystemExit(
            f"verdict (c): blinding.json artifacts_sha {blinding.get('artifacts_sha')} != live {live_artifacts}"
        )
    bundle_path = os.path.join(subdir_path, "comparison.blinded.md")
    try:
        with open(bundle_path, encoding="utf-8") as fh:
            bundle_text = fh.read()
    except OSError:
        raise SystemExit(f"verdict (c): blinded bundle missing at {bundle_path}")
    expected_bundle = blinding.get("bundle_id") or ""
    # The grader-visible bundle is bound to the SEALED ARTIFACTS, not only to
    # itself: regenerate it deterministically from the manifest runs and require
    # the file to be exactly that rendering (a coordinated edit of bundle +
    # blinding.json + grades that stays self-consistent is still refused).
    _mapping_regen, regen_with_token = _render_blinded_bundle(runs, run_ids_sha)
    if _bundle_id_of(regen_with_token) != expected_bundle:
        raise SystemExit(
            "verdict (c): blinding.json bundle_id != the bundle regenerated from the sealed artifacts"
        )
    if bundle_text != regen_with_token.replace(_BUNDLE_ID_TOKEN, expected_bundle):
        raise SystemExit(
            "verdict (c): comparison.blinded.md differs from the bundle regenerated from the sealed artifacts"
        )
    if (
        not expected_bundle
        or _bundle_id_of(bundle_text.replace(expected_bundle, _BUNDLE_ID_TOKEN)) != expected_bundle
    ):
        raise SystemExit(
            "verdict (c): comparison.blinded.md does not hash to its recorded bundle_id"
        )
    rubric_path = os.path.join(HERE, "GRADING_RUBRIC.md")
    prompt_path = os.path.join(repo_root, ".github", "codex", "prompts", "pr_review.md")
    if blinding.get("rubric_sha256") != _file_sha256(rubric_path):
        raise SystemExit("verdict (c): blinding.json rubric_sha256 != live GRADING_RUBRIC.md")
    if blinding.get("prompt_sha256") != _file_sha256(prompt_path):
        raise SystemExit("verdict (c): blinding.json prompt_sha256 != live pr_review.md")
    # (d)
    grades = read_json(args.grades)
    assert isinstance(grades, dict)
    if grades.get("bundle_id") != expected_bundle:
        raise SystemExit(
            f"verdict (d): grades bundle_id {grades.get('bundle_id')!r} != blinded bundle {expected_bundle!r}"
        )
    grades = reviewer_verdict.unblind(grades, blinding["mapping"])
    runs_jsonable = [to_jsonable(r) for r in runs]
    try:
        reviewer_verdict.validate_grades(grades, runs_jsonable)
    except ValueError as exc:
        raise SystemExit(f"verdict (d): {exc}")
    # (e)
    audit_hits = None
    if args.audit_hits:
        audit_hits = read_json(args.audit_hits)
        assert isinstance(audit_hits, dict)
        if audit_hits.get("run_ids_sha") != run_ids_sha:
            raise SystemExit("verdict (e): hits.json run_ids_sha != manifest")
        if audit_hits.get("artifacts_sha") != live_artifacts:
            raise SystemExit("verdict (e): hits.json artifacts_sha != live artifacts")
        if audit_hits.get("branch") != manifest.get("repo_branch"):
            raise SystemExit("verdict (e): hits.json branch != manifest repo_branch")
        markers = {k: manifest.get(k) for k in _MANIFEST_MARKER_KEYS}
        recomputed = reviewer_audit.audit(
            runs, subdir_path, repo_root, markers, run_ids_sha, live_artifacts
        )
        for key in ("hits", "unaudited", "no_commands", "coverage", "vacuous"):
            if _json.dumps(audit_hits.get(key), sort_keys=True) != _json.dumps(
                recomputed.get(key), sort_keys=True
            ):
                raise SystemExit(
                    f"verdict (e): hits.json {key!r} differs from the recomputed audit:\n"
                    f"  file:       {_json.dumps(audit_hits.get(key), sort_keys=True)[:400]}\n"
                    f"  recomputed: {_json.dumps(recomputed.get(key), sort_keys=True)[:400]}"
                )
    # (f)
    violations = []
    configs = set(manifest.get("configs") or [])
    if configs != {args.control, args.candidate}:
        violations.append(
            f"manifest configs {sorted(configs)} != {{{args.control}, {args.candidate}}}"
        )
    loader = CorpusLoader(CORPUS_DIR, repo_root)
    corpus = loader.load_cases(None)
    corpus_ids = {c.id for c in corpus}
    run_cases = {r.case_id for r in runs}
    if run_cases != corpus_ids:
        violations.append(
            f"OK runs cover {len(run_cases)} case(s) but the committed corpus has {len(corpus_ids)} "
            f"(no --strata filtering allowed): missing {sorted(corpus_ids - run_cases)[:6]}, "
            f"extra {sorted(run_cases - corpus_ids)[:6]}"
        )
    n_s4 = sum(1 for c in corpus if c.stratum == "s4_missed")
    if len(corpus_ids) < 8 or n_s4 < 2:
        violations.append(
            f"corpus below the floor: {len(corpus_ids)} cases, {n_s4} s4_missed (need >= 8 / >= 2)"
        )
    expected_grid = {
        (c, a, r) for c in corpus_ids for a in (args.control, args.candidate) for r in (0, 1)
    }
    realized = [(r.case_id, r.config_id, int(r.repeat_idx)) for r in runs]
    if len(realized) != len(set(realized)):
        violations.append("duplicate (case, arm, repeat) among the manifest's OK runs")
    if set(realized) != expected_grid:
        violations.append(
            f"realized grid != cases x {{control, candidate}} x {{0, 1}}: missing "
            f"{sorted(expected_grid - set(realized))[:6]}, extra {sorted(set(realized) - expected_grid)[:6]}"
        )
    prov = {k: manifest.get(k) for k in _PROVENANCE_KEYS}
    recomputed_agg = _provenance_aggregates(runs, prov)
    stored_agg = {k: manifest.get(k) for k in _AGGREGATE_KEYS}
    if stored_agg != recomputed_agg:
        violations.append(
            f"manifest aggregates {stored_agg} != recomputed from the sealed artifacts {recomputed_agg}"
        )
    if recomputed_agg["run_head_missing"] != 0:
        violations.append(f"run_head_missing = {recomputed_agg['run_head_missing']}")
    if recomputed_agg["run_dirty_any"] is not False:
        violations.append(f"run_dirty_any = {recomputed_agg['run_dirty_any']!r}")
    if recomputed_agg["run_head_shas"] != [manifest.get("repo_head_sha")]:
        violations.append(
            f"run_head_shas {recomputed_agg['run_head_shas']} != [{manifest.get('repo_head_sha')}]"
        )
    if recomputed_agg["run_marker_mismatch"] != 0 or recomputed_agg["run_not_isolated"] != 0:
        violations.append(
            f"run_marker_mismatch={recomputed_agg['run_marker_mismatch']} "
            f"run_not_isolated={recomputed_agg['run_not_isolated']} (both must be 0)"
        )
    if manifest.get("repo_dirty") is not False:
        violations.append(f"manifest repo_dirty = {manifest.get('repo_dirty')!r}")
    bad_prov = _provenance_failures(prov)
    if bad_prov:
        violations.append(f"manifest write-time provenance incomplete: {bad_prov}")
    if args.expect_head and manifest.get("repo_head_sha") != args.expect_head:
        violations.append(
            f"repo_head_sha {manifest.get('repo_head_sha')} != --expect-head {args.expect_head}"
        )
    exposure = _exposure_status(manifest, args.subdir, (args.control, args.candidate), repo_root)
    if exposure.get("protocol_error"):
        violations.append(f"frozen-provenance integrity: {exposure['protocol_error']}")
    if violations:
        raise SystemExit(
            "verdict (f): protocol guard refused a definitive verdict:\n  - "
            + "\n  - ".join(violations)
        )
    # (g)
    _assert_tree_identity(repo_root, manifest, "verdict (stage entry)")
    # (h)
    result = reviewer_verdict.gates(grades, runs_jsonable, args.control, args.candidate, audit_hits)
    # Pre-registered informational reads (latency, 2x-p95 flag, input cost) from
    # the sealed artifacts; prices from the production script's PRICING table.
    from adapters.openai_review_loader import load_openai_review

    pricing = getattr(load_openai_review(repo_root), "PRICING", {})
    input_prices = {m: p[0] for m, p in pricing.items()}
    result["campaign_reads"] = reviewer_verdict.campaign_reads(
        runs_jsonable, args.control, args.candidate, input_prices
    )
    result["exposure"] = exposure
    if not exposure["mechanical"]:
        # Pre-registration did not survive: computed and reported, never a
        # mechanical GO - the final call is the user's, recorded in the verdict.
        result["definitive"] = False
        result["label"] = (
            f"{result['verdict']} (RE-REGISTERED after candidate exposure - "
            f"non-mechanical, user decision: {exposure['reason']})"
        )
    print(_json.dumps(result, indent=2, sort_keys=True))
    print(f"\nVERDICT: {result['label']}")
    if result["contested_gates"]:
        print(
            f"CONTESTED (verifier-confirmed leaks): {result['contested_gates']} - the final call is the user's"
        )
    _assert_tree_identity(repo_root, manifest, "verdict (stage exit)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Codex reviewer A/B comparison harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify-corpus", help="materialize + check every case")
    pv.add_argument("--strata", nargs="*", default=None)
    pv.set_defaults(func=cmd_verify_corpus)

    ps = sub.add_parser("smoke", help="tiny end-to-end run (1 case; first codex call)")
    ps.add_argument(
        "--configs",
        default=None,
        help="arm ids to smoke (default: the role=control arm from configs.json)",
    )
    ps.add_argument("--strata", nargs="*", default=None)
    ps.add_argument("--k", type=int, default=1)
    ps.add_argument(
        "--limit", type=int, default=1, help="cap selected cases (default 1; pass 0 to run all)"
    )
    ps.add_argument("--max-parallel", type=int, default=2)
    ps.set_defaults(func=cmd_smoke)

    pr = sub.add_parser("run", help="full comparison matrix; saves raw reviews")
    pr.add_argument("--configs", default="B,E")
    pr.add_argument("--strata", nargs="*", default=None)
    pr.add_argument("--subdir", default="full")
    pr.add_argument(
        "--k",
        type=int,
        default=2,
        help="repeats per (case, config); the default B,E contrast is registered at k=2",
    )
    pr.add_argument(
        "--k-per",
        default="",
        help="per-config repeat overrides, e.g. 'C=1,D=1' (others use --k)",
    )
    pr.add_argument("--max-parallel", type=int, default=5)
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare", help="emit the side-by-side bundle to grade")
    pc.add_argument("--subdir", default="full")
    pc.add_argument(
        "--allow-mixed",
        action="store_true",
        help="compare ALL runs in the subdir when no manifest exists (may mix experiments)",
    )
    pc.add_argument(
        "--blinded",
        action="store_true",
        help="also write comparison.blinded.md (arm identities replaced by neutral "
        "labels; mapping sealed in blinding.json) for identity-blind grading",
    )
    pc.set_defaults(func=cmd_compare)

    pa = sub.add_parser("audit", help="post-unblinding session-log audit -> hits.json")
    pa.add_argument("--subdir", default="full")
    pa.add_argument(
        "-o",
        "--output",
        required=True,
        help="where to write hits.json (outside the repo or under runs/)",
    )
    pa.set_defaults(func=cmd_audit)

    pvd = sub.add_parser(
        "verdict", help="apply the pre-registered gates to a reconciled grades JSON"
    )
    pvd.add_argument("--subdir", default="full")
    pvd.add_argument("--grades", required=True)
    pvd.add_argument("--control", required=True)
    pvd.add_argument("--candidate", required=True)
    pvd.add_argument("--audit-hits", default=None)
    pvd.add_argument("--expect-head", default=None)
    pvd.set_defaults(func=cmd_verdict)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
