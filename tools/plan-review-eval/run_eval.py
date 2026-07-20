#!/usr/bin/env python3
"""Plan-review engine comparison harness (second consumer of tools/eval_core).

Subcommands:

  verify-corpus   materialize + contract-check every case (no reviewer calls)
  smoke           tiny end-to-end run (fixture case, one arm, k=1)
  run             the arm matrix; saves raw/merged reviews (resumable)
  extract         neutral findings extraction over stored reviews (resumable)
  compare         emit the grading bundle from EXTRACTIONS (+ --blinded)
  verdict         mechanical gates over a final reconciled grading table

See DECISION_RULE.md for the pre-registered decision rule; README.md for the
flow. Run artifacts land under runs/ (gitignored). Real corpus cases live
under corpus/cases/ (gitignored — the user's plans are never committed).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
# Make `plan_adapters`/`verdict` (harness root) and `eval_core` (under tools/)
# importable. The package is named plan_adapters (not adapters) so one pytest
# process can import both harnesses without sys.modules collisions.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from eval_core.store import RunStore, read_json, write_json  # noqa: E402

CONFIG_PATH = os.path.join(HERE, "config", "configs.json")
CORPUS_DIR = os.path.join(HERE, "corpus")
RUNS_DIR = os.path.join(HERE, "runs")

_ARM_KEYS = {
    "id",
    "role",
    "variant",
    "mode",
    "model",
    "effort",
    "cli_version",
    "label",
}


_SUBDIR_RE = None  # compiled lazily below


def _safe_subdir(subdir: str) -> str:
    """Restrict --subdir to a plain identifier so manifests/extractions/verdicts
    can never land outside runs/ (no separators, no ``..``, no absolute paths)."""
    import re

    global _SUBDIR_RE
    if _SUBDIR_RE is None:
        _SUBDIR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    if not _SUBDIR_RE.match(subdir) or ".." in subdir:
        raise SystemExit(
            f"--subdir {subdir!r} must be a plain identifier "
            f"([A-Za-z0-9._-], no leading dot, no path separators)."
        )
    return subdir


def _repo_root() -> str:
    cp = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise SystemExit(f"cannot locate repo root from {HERE}: {cp.stderr.strip()}")
    return cp.stdout.strip()


def _load_configs() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _make_configs(which: list[str], raw: dict | None = None) -> list:
    """Build Config objects for the requested arm ids (fail-closed validation,
    mirroring reviewer-eval: unknown keys, duplicate ids, missing fields, and
    anything but exactly one role=control arm all raise). ``raw`` is a
    campaign's protocol-snapshot config; None re-reads disk (non-campaign
    paths)."""
    from eval_core.models import Config

    raw = raw if raw is not None else _load_configs()
    arms = raw.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("configs.json must define a non-empty 'arms' list")
    by_id: dict = {}
    n_controls = 0
    for c in arms:
        unknown = set(c) - _ARM_KEYS
        if unknown:
            raise ValueError(
                f"configs.json arm {c.get('id', '?')!r} has unknown key(s) "
                f"{sorted(unknown)}; allowed: {sorted(_ARM_KEYS)}"
            )
        for req in ("id", "variant", "mode", "model", "effort"):
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
            cli_version=c.get("cli_version"),
            label=c.get("label", ""),
            variant=c["variant"],
            mode=c["mode"],
        )
    if n_controls != 1:
        raise ValueError(
            f"configs.json must declare exactly one arm with role='control' "
            f"(found {n_controls})"
        )
    missing = [i for i in which if i not in by_id]
    if missing:
        raise ValueError(f"unknown arm id(s) {missing}; have {sorted(by_id)}")
    return [by_id[i] for i in which]


def _control_id(raw: dict | None = None) -> str:
    raw = raw if raw is not None else _load_configs()
    controls = [c.get("id") for c in raw.get("arms", []) if c.get("role") == "control"]
    if len(controls) != 1 or not controls[0]:
        raise ValueError("configs.json must declare exactly one role='control' arm")
    return controls[0]


def _all_arm_ids(raw: dict | None = None) -> list[str]:
    return [c["id"] for c in (raw if raw is not None else _load_configs()).get("arms", [])]


def _treatment_fields(raw: dict | None = None) -> tuple:
    tf = (raw if raw is not None else _load_configs()).get("treatment_fields")
    if not tf:
        raise ValueError("configs.json must declare treatment_fields")
    return tuple(tf)


def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _evaluator_source_files() -> list[tuple[str, str]]:
    """(label, abspath) for EVERY Python source that can shape a gating
    artifact: this harness's whole tree (run_eval, verdict, plan_adapters) and
    all of eval_core. Enumerated by walking the trees, so a new or renamed
    module joins the identity by construction — never by remembering to add it
    to a hand-kept list. runs/ (artifacts, incl. materialized worktrees under
    runs/.worktrees/) and corpus/ (data) are pruned: they hold repo checkouts
    and case files, not evaluator code."""
    import eval_core

    roots = [
        (os.path.basename(HERE), HERE),
        ("eval_core", os.path.dirname(os.path.abspath(eval_core.__file__))),
    ]
    out: list[tuple[str, str]] = []
    for label, root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in ("runs", "corpus", "__pycache__") and not d.startswith(".")
            ]
            for fn in filenames:
                if fn.endswith(".py"):
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    out.append((f"{label}/{rel}", full))
    # External executable dependency: dual arms run the codex reviewer through
    # .claude/scripts/openai_review.py (importlib-loaded by plan_reviewer), so
    # its bytes are protocol too — an edit there changes what arm C/E execute.
    external = os.path.join(_repo_root(), ".claude", "scripts", "openai_review.py")
    if not os.path.exists(external):
        raise SystemExit(f"protocol source missing: {external}")
    out.append(("external/.claude/scripts/openai_review.py", external))
    return sorted(out)


def _read_source_bytes(files: list[tuple[str, str]]) -> list[bytes]:
    parts = []
    for label, path in files:
        with open(path, "rb") as fh:
            parts.append(label.encode("utf-8") + b"\0" + fh.read() + b"\0")
    return parts


def _import_executing_modules():
    """Eagerly import every module the campaign will execute, INSIDE the
    snapshot's read→import→re-read stability bracket. Python caches imports
    process-wide, so later stages run the code imported here — which the
    bracket proves is byte-identical to what the identity hashed (closing the
    snapshot-then-import A→B→A window). Returns the dynamically loaded codex
    wrapper module, ALSO loaded inside the bracket: the snapshot hands it to
    PlanReviewer, so dual arms execute those exact bytes (a disk edit after
    the snapshot can never reach execution)."""
    import eval_core.compare  # noqa: F401
    import eval_core.models  # noqa: F401
    import eval_core.runner  # noqa: F401
    import eval_core.store  # noqa: F401
    import verdict  # noqa: F401
    from plan_adapters import (  # noqa: F401
        corpus_loader,
        criteria_source,
        plan_reviewer,
        worktree,
    )

    return plan_reviewer._load_openai_review(_repo_root())


def _identity_sha(identity: dict) -> str:
    """One canonical sha over a protocol-identity dict (stable key order).
    Stamped into extraction metadata and blinding.json so every gating
    artifact names the protocol it was produced under."""
    return _sha16(json.dumps(identity, sort_keys=True).encode("utf-8"))


def _protocol_snapshot() -> dict:
    """ONE read of every protocol artifact into memory: the decision rule, the
    raw config bytes (parsed from those same bytes), and every candidate
    artifact. Execution AND the recorded identity both derive from this
    snapshot, so an A→B→A edit between hashing and loading cannot make the
    campaign execute one protocol while recording another. (Control criteria
    are sha-addressed via `git show` — immutable by construction.)"""
    # The EVALUATOR code is protocol too: the gates, the bundle renderer, the
    # extractor, the criteria renderer, the loaders, this orchestrator, and
    # the external codex wrapper all shape what a verdict means, so EVERY
    # source joins the recorded identity (any edit -> drift -> NON-GATING
    # until re-run). The label participates so a rename drifts the identity
    # even with unchanged bytes. The read→import→re-read bracket proves the
    # code Python will execute is byte-identical to what was hashed: sources
    # are read, the executing modules are imported (cached process-wide,
    # BEFORE any other module use in this snapshot), and the sources are read
    # AGAIN — any mismatch aborts the snapshot.
    source_files = _evaluator_source_files()
    evaluator_parts = _read_source_bytes(source_files)
    openai_review_mod = _import_executing_modules()
    if _read_source_bytes(source_files) != evaluator_parts:
        raise SystemExit(
            "protocol sources changed while the snapshot was being taken — "
            "the imported code cannot be proven identical to the hashed bytes; "
            "re-run once the tree is quiescent."
        )
    from plan_adapters.criteria_source import _CONTROL_PROMPT

    cand_dir = os.path.join(HERE, "candidates")
    with open(CONFIG_PATH, "rb") as fh:
        config_bytes = fh.read()
    with open(os.path.join(HERE, "DECISION_RULE.md"), "rb") as fh:
        rule_bytes = fh.read()
    candidates = {}
    for name in sorted(os.listdir(cand_dir)):
        if name.endswith(".md"):
            with open(os.path.join(cand_dir, name), "rb") as fh:
                candidates[name] = fh.read()
    raw = json.loads(config_bytes.decode("utf-8"))
    prompt = candidates.get("extraction_prompt.md", b"")
    extraction = {
        "prompt_sha": _sha16(prompt),
        "model": (raw.get("extraction") or {}).get("model", ""),
    }
    identity = {
        "decision_rule_sha": _sha16(rule_bytes),
        "configs_sha": _sha16(config_bytes),
        "candidates_sha": {name: _sha16(data) for name, data in candidates.items()},
        "control_prompt_sha": _sha16(_CONTROL_PROMPT.encode("utf-8")),
        "evaluator_sha": _sha16(b"".join(evaluator_parts)),
        "extraction": extraction,
        "registered_contrasts": sorted(list(c) for c in _registered_contrasts(raw)),
    }
    return {
        "raw_config": raw,
        "candidate_texts": {n: d.decode("utf-8") for n, d in candidates.items()},
        "control_prompt": _CONTROL_PROMPT,
        "openai_review_mod": openai_review_mod,
        "identity": identity,
    }


def _protocol_identity() -> dict:
    """Identity-only view (verdict-time drift comparison)."""
    return _protocol_snapshot()["identity"]


def _read_plan_for_freeze(case) -> str:
    """One canonical read of a case's plan text (same containment rules the
    reviewer applies), used to freeze bytes before the matrix and to detect
    drift after it."""
    case_dir = case.fixture.get("_case_dir", "")
    rel = case.fixture.get("plan", "plan.md")
    base = os.path.realpath(case_dir or ".")
    full = os.path.realpath(os.path.join(base, rel))
    if full != base and not full.startswith(base + os.sep):
        raise SystemExit(f"{case.id}: fixture.plan escapes its case directory")
    with open(full, encoding="utf-8") as fh:
        return fh.read()


def _loader():
    from plan_adapters.corpus_loader import PlanCorpusLoader

    return PlanCorpusLoader(CORPUS_DIR, _repo_root())


def _reviewer(runs_root: str, snapshot: dict | None = None):
    from plan_adapters.criteria_source import load_artifacts
    from plan_adapters.plan_reviewer import PlanReviewer

    raw = snapshot["raw_config"] if snapshot else _load_configs()
    repo = _repo_root()
    artifacts = load_artifacts(
        repo,
        HERE,
        raw.get("control_criteria", {}),
        candidate_texts=snapshot["candidate_texts"] if snapshot else None,
        control_prompt_text=snapshot["control_prompt"] if snapshot else None,
    )
    extraction_model = (raw.get("extraction") or {}).get("model", "")
    return PlanReviewer(
        repo,
        runs_root,
        artifacts,
        extraction_model=extraction_model,
        openai_mod=(snapshot or {}).get("openai_review_mod"),
    )


# --------------------------------------------------------------------------- #
# verify-corpus
# --------------------------------------------------------------------------- #


def cmd_verify_corpus(args: argparse.Namespace) -> int:
    loader = _loader()
    cases = loader.load_cases(args.strata)
    if not cases:
        print("no cases found (corpus/cases/ is local-only; the committed fixture should load)")
        return 1
    failures = 0
    for case in cases:
        err = loader.verify(case)
        if err:
            failures += 1
            print(f"[FAIL] {case.id} ({case.stratum}): {err}")
        else:
            print(f"[OK] {case.id} ({case.stratum})")
    print(f"\n{len(cases) - failures}/{len(cases)} cases verified.")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# smoke / run
# --------------------------------------------------------------------------- #


def _campaign_fingerprint(cases, configs, k: int, k_overrides) -> dict:
    """The registered SAMPLE PLAN: which observations this campaign will make
    AND how they will be scored — arms, repeat schedule, and every case's
    identity, content, and complete scoring metadata (declared base_sha,
    frozen plan bytes, and the canonical hash of the whole case definition:
    ground truth, must_catch, FP allowances, known-FP topics, weight, notes).
    Fixed with the protocol at registration: an outcome-dependent change to
    any of it after results exist is a NEW campaign, never a rewrite."""
    return {
        "config_ids": sorted(c.id for c in configs),
        "k": k,
        "k_overrides": dict(k_overrides or {}),
        "cases": {
            c.id: {
                "base_sha": str(c.fixture.get("base_sha", "")),
                "plan_sha": _sha16(str(c.fixture.get("_plan_text", "")).encode("utf-8")),
                "case_sha": str(c.fixture.get("_case_sha", "")),
            }
            for c in cases
        },
    }


def _require_campaign_registration(subdir: str, identity: dict, fingerprint: dict) -> None:
    """PRE-REGISTRATION IS WRITE-ONCE: a subdirectory IS a campaign, and its
    protocol identity AND sample plan are fixed by the first registration. An
    invocation differing in either may never touch it — the store's cached
    outcomes can never be re-attributed to a later protocol, and the schedule
    can never be reshaped after outcomes were observed (a changed protocol or
    sample plan is a NEW campaign in a fresh subdir)."""
    path = os.path.join(RUNS_DIR, f"{subdir}-manifest.json")
    if not os.path.exists(path):
        return
    prior = read_json(path)
    if (prior.get("protocol") or {}) != identity:  # type: ignore[union-attr]
        raise SystemExit(
            f"runs/{subdir} is registered under a DIFFERENT protocol identity — "
            f"a changed protocol is a NEW campaign; use a fresh --subdir (or "
            f"restore the registered protocol to resume this one)."
        )
    if (prior.get("campaign_fingerprint") or {}) != fingerprint:  # type: ignore[union-attr]
        raise SystemExit(
            f"runs/{subdir} registered a DIFFERENT sample plan (cases, plan "
            f"bytes, base SHAs, arms, k, or overrides) — the schedule cannot "
            f"be reshaped after observation; use a fresh --subdir for a "
            f"changed campaign."
        )


def _run_matrix(
    cases,
    config_ids: list[str],
    k: int,
    subdir: str,
    max_parallel: int,
    k_overrides=None,
    verified: bool = False,
    enforce_registration: bool = False,
):
    from eval_core.runner import run_matrix

    # ONE immutable protocol snapshot: the same in-memory bytes produce both
    # the recorded identity and the executing configs/artifacts (an A→B→A edit
    # between hashing and loading cannot split them), and a post-run re-read
    # detects anything that changed while the matrix ran.
    snapshot = _protocol_snapshot()
    protocol_before = snapshot["identity"]
    manifest_path = os.path.join(RUNS_DIR, f"{subdir}-manifest.json")
    configs = _make_configs(config_ids, raw=snapshot["raw_config"])
    runs_root = os.path.join(RUNS_DIR, subdir)
    store = RunStore(runs_root)
    reviewer = _reviewer(runs_root, snapshot=snapshot)
    # Freeze every case's plan bytes for the WHOLE matrix: each job renders
    # from this in-memory copy, so an edit to a corpus plan mid-campaign can
    # never expose later arms/repeats to different content under one run key.
    for case in cases:
        case.fixture["_plan_text"] = _read_plan_for_freeze(case)
    fingerprint = _campaign_fingerprint(cases, configs, k, k_overrides)
    if enforce_registration:
        _require_campaign_registration(subdir, protocol_before, fingerprint)
    registration = {
        "subdir": subdir,
        "config_ids": [c.id for c in configs],
        "case_ids": sorted({c.id for c in cases}),
        "case_strata": {c.id: c.stratum for c in cases},
        "k": k,
        "k_overrides": dict(k_overrides or {}),
        "treatment_fields": list(_treatment_fields(snapshot["raw_config"])),
        # Realized after the matrix; empty at registration time.
        "run_keys": [],
        # cmd_run verified every selected case before any reviewer call; smoke
        # runs the (already CI-verified) fixture without the full corpus gate.
        "corpus_verified": verified,
        "protocol": protocol_before,
        "campaign_fingerprint": fingerprint,
        "protocol_drift_during_run": False,
        "corpus_drift_during_run": False,
    }
    if enforce_registration:
        # REGISTER FIRST, OBSERVE SECOND: the protocol identity reaches disk
        # before any reviewer call, so a crash mid-matrix leaves a registered
        # campaign (empty run_keys — never gating), not an orphaned cache a
        # later protocol could silently adopt.
        write_json(manifest_path, registration)
    results = run_matrix(
        cases,
        configs,
        reviewer,
        store,
        k=k,
        max_parallel=max_parallel,
        progress=lambda m: print(f"  {m}"),
        treatment_fields=_treatment_fields(snapshot["raw_config"]),
        k_overrides=k_overrides,
    )
    manifest = {
        **registration,
        "run_keys": sorted(rr.run_id for rr in results),
        # Post-run drift checks: True means the protocol files or a corpus plan
        # changed WHILE the matrix ran — verdict treats either as a violation.
        "protocol_drift_during_run": _protocol_identity() != protocol_before,
        "corpus_drift_during_run": any(
            _read_plan_for_freeze(c) != c.fixture.get("_plan_text") for c in cases
        ),
    }
    write_json(manifest_path, manifest)
    n_err = sum(1 for rr in results if not rr.ok)
    print(f"\n{len(results)} runs ({n_err} INFRA_ERROR) -> runs/{subdir}/")
    if n_err:
        for rr in results:
            if not rr.ok:
                print(
                    f"  INFRA_ERROR {rr.case_id} {rr.config_id} r{rr.repeat_idx}: {rr.infra_error}"
                )
    return results


def cmd_smoke(args: argparse.Namespace) -> int:
    loader = _loader()
    cases = [c for c in loader.load_cases(None) if c.stratum == "fixture"]
    if not cases:
        print("no committed fixture case found under corpus/fixture/")
        return 1
    config_ids = args.configs.split(",") if args.configs else [_control_id()]
    results = _run_matrix(cases, config_ids, k=1, subdir="smoke", max_parallel=2)
    bad = [rr for rr in results if not rr.ok]
    for rr in results:
        if rr.ok:
            head = rr.review_markdown.strip().splitlines()[:8]
            print(f"\n--- {rr.case_id} {rr.config_id} ({rr.latency_s:.0f}s) ---")
            print("\n".join(head))
    return 1 if bad else 0


def cmd_run(args: argparse.Namespace) -> int:
    loader = _loader()
    cases = loader.load_cases(args.strata)
    # The fabricated fixture is CI/smoke/dress-rehearsal data — it must never
    # slip into a campaign implicitly. Include it only when explicitly selected
    # via --strata fixture or --cases.
    if args.strata is None and not args.cases:
        cases = [c for c in cases if c.stratum != "fixture"]
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.id in wanted]
        missing = wanted - {c.id for c in cases}
        if missing:
            print(f"unknown case id(s): {sorted(missing)}")
            return 1
    if not cases:
        print("no cases selected")
        return 1
    # Verified corpus is a PRE-RUN gate (and a pre-registered prerequisite for a
    # gating verdict): a malformed or non-materializable case must fail here,
    # before any reviewer spend, and the manifest records that it did not.
    failures = [(c.id, err) for c in cases if (err := loader.verify(c))]
    if failures:
        for cid, err in failures:
            print(f"[VERIFY-FAIL] {cid}: {err}")
        print("aborting before any reviewer call — fix the corpus first.")
        return 1
    config_ids = args.configs.split(",") if args.configs else _all_arm_ids()
    k_overrides = None
    if args.k_per:
        k_overrides = {}
        for part in args.k_per.split(","):
            cid, _, n = part.partition("=")
            k_overrides[cid.strip()] = int(n)
    results = _run_matrix(
        cases,
        config_ids,
        k=args.k,
        subdir=args.subdir,
        max_parallel=args.max_parallel,
        k_overrides=k_overrides,
        verified=True,
        # Campaign subdirs are write-once per protocol; smoke re-registers its
        # fixture subdir freely (never gating, never graded).
        enforce_registration=True,
    )
    return 1 if any(not rr.ok for rr in results) else 0


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def _load_manifest(subdir: str) -> dict:
    path = os.path.join(RUNS_DIR, f"{subdir}-manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"no manifest at {path}; run `run --subdir {subdir}` first")
    return read_json(path)  # type: ignore[return-value]


def _manifest_results(subdir: str):
    manifest = _load_manifest(subdir)
    store = RunStore(os.path.join(RUNS_DIR, subdir))
    results = []
    missing = []
    for key in manifest["run_keys"]:
        rr = store.load(key)
        if rr is None:
            missing.append(key)
        else:
            results.append(rr)
    if missing:
        raise SystemExit(
            f"manifest lists {len(missing)} run(s) with no stored artifact: {missing[:5]}"
        )
    return manifest, results


def _extraction_path(subdir: str, run_id: str) -> str:
    return os.path.join(RUNS_DIR, subdir, "extractions", f"{run_id}.md")


def _run_keys_sha(manifest: dict) -> str:
    return hashlib.sha256("|".join(sorted(manifest["run_keys"])).encode("utf-8")).hexdigest()[:16]


_BUNDLE_ID_TOKEN = "__BUNDLE_ID__"


def _bundle_id_of(bundle_text_with_token: str) -> str:
    """Identity of one blinded grading bundle = the hash of the EXACT bytes the
    graders read (rendered with a fixed placeholder where the id itself goes,
    then substituted). Because the id IS the artifact hash, everything grader-
    visible is bound by construction: per-run extraction assignment, snapshots,
    the header, the sanitizer's output. `verdict` restores the placeholder,
    re-hashes, and requires the match — grades from any other bundle (swapped
    extractions, edited header, re-render) can never silently score this one."""
    return hashlib.sha256(bundle_text_with_token.encode("utf-8")).hexdigest()[:16]


_MERGE_MARKERS = ("[consensus]", "[single reviewer]")


def _strip_merge_markers(text: str) -> str:
    """Deterministic backstop for the extraction prompt: dual-arm merge metadata
    (agreement tags) must never reach the blinded bundle — they identify C/E as
    dual arms, unblinding the very contrast being judged."""
    for marker in _MERGE_MARKERS:
        text = text.replace(marker, "")
    return text


def _extraction_identity(snapshot: dict) -> dict:
    """Cache identity of the extraction stage: prompt hash + pinned model,
    derived from the stage's ONE protocol snapshot — never a separate disk
    read (a mid-stage edit could otherwise split identity from execution).

    Stored beside every extraction; a prompt or model change invalidates prior
    extractions (they are re-run, or `compare` refuses a mixed set) — one
    blinded comparison must never mix extraction methodologies.
    """
    prompt = snapshot["candidate_texts"].get("extraction_prompt.md", "")
    if not prompt:
        raise SystemExit("candidates/extraction_prompt.md is missing or empty")
    model = (snapshot["raw_config"].get("extraction") or {}).get("model", "")
    return {
        "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "model": model,
    }


def _extraction_meta_path(subdir: str, run_id: str) -> str:
    return os.path.join(RUNS_DIR, subdir, "extractions", f"{run_id}.meta.json")


def _extraction_meta_expected(identity: dict, rr, protocol_sha: str) -> dict:
    """The lineage an extraction must match to be reused: the extraction
    prompt+model identity, the sha of the raw review it extracted from, AND
    the campaign protocol identity it was produced under — an artifact carried
    over from a foreign protocol or campaign can never be silently reused."""
    return {
        **identity,
        "review_sha": hashlib.sha256(rr.review_markdown.encode("utf-8")).hexdigest()[:16],
        "protocol_sha": protocol_sha,
    }


def _require_recorded_protocol(manifest: dict, stage: str, live: dict | None = None) -> None:
    """Every post-run stage runs under the SAME protocol the campaign recorded
    — an A→B→A restore around a single stage is caught at that stage, not at
    verdict time. Stages call this twice: at entry with `live` = the snapshot
    identity they will execute from, and at exit with no `live` (a FRESH disk
    read), so an edit landing WHILE the stage runs is surfaced before its
    artifacts are trusted."""
    recorded = manifest.get("protocol")
    if recorded and recorded != (live if live is not None else _protocol_identity()):
        raise SystemExit(
            f"{stage}: the live protocol differs from the identity recorded in the "
            f"manifest — post-run protocol edits cannot re-define this campaign; "
            f"restore the recorded protocol (or re-run the campaign) first."
        )


def cmd_extract(args: argparse.Namespace) -> int:
    # ONE snapshot feeds the entry gate AND every stage input (prompt, model,
    # reviewer artifacts) — there is no window where a disk edit can split
    # what the gate checked from what the stage executes.
    snap = _protocol_snapshot()
    _manifest, results = _manifest_results(args.subdir)
    _require_recorded_protocol(_manifest, "extract", live=snap["identity"])
    reviewer = _reviewer(os.path.join(RUNS_DIR, args.subdir), snapshot=snap)
    identity = _extraction_identity(snap)
    prompt = snap["candidate_texts"]["extraction_prompt.md"]
    protocol_sha = _identity_sha(_manifest.get("protocol") or {})
    ok_runs = [rr for rr in results if rr.ok]
    done = skipped = 0
    for rr in ok_runs:
        out_path = _extraction_path(args.subdir, rr.run_id)
        meta_path = _extraction_meta_path(args.subdir, rr.run_id)
        expected = _extraction_meta_expected(identity, rr, protocol_sha)
        # Reuse ONLY when the stored identity matches the current prompt+model
        # AND the raw review bytes it extracted from; anything else re-extracts
        # (a regenerated run or edited prompt never silently mixes).
        if os.path.exists(out_path) and not args.force:
            try:
                stored = read_json(meta_path)
                if {k: stored.get(k) for k in expected} == expected:
                    skipped += 1
                    continue
            except (OSError, ValueError, AttributeError):
                pass
        text, models_used = reviewer.extract(rr.review_markdown, prompt)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = f"{out_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, out_path)
        write_json(meta_path, {**expected, "models_used": models_used})
        done += 1
        print(f"  extracted {rr.case_id} {rr.config_id} r{rr.repeat_idx}")
    # Stage-exit recheck (FRESH read): a protocol edit that landed while
    # extraction ran is surfaced now, never first at verdict time.
    _require_recorded_protocol(_manifest, "extract (stage exit)")
    print(
        f"\n{done} extracted, {skipped} already present, {len(results) - len(ok_runs)} INFRA_ERROR skipped."
    )
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

_GRADING_HEADER = """# Plan-review engine comparison — grading bundle

Each case below shows its ground-truth defects (what a faithful plan review
SHOULD surface) followed by every arm's EXTRACTED findings (a uniform,
format-neutral reduction of the arm's review; see DECISION_RULE.md).

## How to grade

Fill one row per (case, ground-truth defect, arm label, repeat):
`caught` / `partial` / `missed`, with the extraction's verbatim evidence quote
for every `caught`. Plus ONE `negative_assessments` cell per (negative-control
case, arm label, repeat) with a `findings` list — `findings: []` for a clean
read; an omitted cell is a validation error, never "zero FPs".

Rules:
- Severities use the neutral scale: blocker / major / minor.
- A finding "catches" a defect if it names the same defect at the same
  location/symbol, regardless of wording. `partial` (right file or class,
  wrong defect or location) counts as MISSED for the gates.
- On a negative-control case (marked "NO known bugs"), any finding whose
  severity is outside the allowed set is a FALSE POSITIVE — EXCEPT findings
  matching one of the case's listed known-FP topics, which are never counted:
  mark such a row with the topic's id in a `known_topic_id` field instead of
  omitting it, so the exemption is auditable and applied mechanically.
- Hallucination check: a `caught` row's evidence quote must actually name the
  defect; a finding asserting verification the plan/repo cannot support is an
  FP-class note, never a catch.
- Grade on content only — never on guessed arm identity.
"""


def cmd_compare(args: argparse.Namespace) -> int:
    from eval_core.compare import apply_blinding, build_bundle, derive_blind_mapping

    # ONE snapshot feeds the entry gate AND every stage input (see cmd_extract).
    snap = _protocol_snapshot()
    manifest, results = _manifest_results(args.subdir)
    _require_recorded_protocol(manifest, "compare", live=snap["identity"])
    ok_runs = [rr for rr in results if rr.ok]
    infra = [rr for rr in results if not rr.ok]
    if infra:
        print(f"NOTE: {len(infra)} INFRA_ERROR run(s) excluded from the bundle:")
        for rr in infra:
            print(f"  {rr.case_id} {rr.config_id} r{rr.repeat_idx}: {rr.infra_error}")

    identity = _extraction_identity(snap)
    protocol_sha = _identity_sha(manifest.get("protocol") or {})
    missing, stale = [], []
    for rr in ok_runs:
        if not os.path.exists(_extraction_path(args.subdir, rr.run_id)):
            missing.append(rr)
            continue
        expected = _extraction_meta_expected(identity, rr, protocol_sha)
        try:
            stored = read_json(_extraction_meta_path(args.subdir, rr.run_id))
            if {k: stored.get(k) for k in expected} != expected:
                stale.append(rr)
        except (OSError, ValueError, AttributeError):
            stale.append(rr)
    if missing or stale:
        ids = [f"{rr.case_id}/{rr.config_id}/r{rr.repeat_idx}" for rr in (missing + stale)[:8]]
        raise SystemExit(
            f"{len(missing)} run(s) have no extraction and {len(stale)} have a "
            f"stale/foreign extraction identity ({ids}...); run `extract --subdir "
            f"{args.subdir}` first — the bundle grades extractions of ONE "
            f"homogeneous prompt+model identity, never raw reviews (DECISION_RULE.md)."
        )

    # The graded artifact is the extraction; swap it in for rendering. Raw
    # reviews stay untouched in the store.
    ext_runs = []
    for rr in ok_runs:
        with open(_extraction_path(args.subdir, rr.run_id), encoding="utf-8") as fh:
            ext_text = _strip_merge_markers(fh.read())
        ext_runs.append(dataclasses.replace(rr, review_markdown=ext_text))

    out_dir = os.path.join(RUNS_DIR, args.subdir)
    bundle = build_bundle(ext_runs, header_text=_GRADING_HEADER)
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write(bundle)
    print(f"wrote runs/{args.subdir}/comparison.md")

    if args.blinded:
        raw_cfg = snap["raw_config"]
        model_names = {c.get("model", "") for c in raw_cfg.get("arms", [])}
        model_names |= {
            c["mode"].split(":", 1)[1]
            for c in raw_cfg.get("arms", [])
            if c.get("mode", "").startswith("dual:")
        }
        model_names.add((raw_cfg.get("extraction") or {}).get("model", ""))
        salt = hashlib.sha256("|".join(sorted(manifest["run_keys"])).encode("utf-8")).hexdigest()
        mapping = derive_blind_mapping(sorted({rr.config_id for rr in ext_runs}), salt)
        blinded = apply_blinding(ext_runs, mapping, model_names)
        header_b = _GRADING_HEADER + (
            f"\nBundle ID: `{_BUNDLE_ID_TOKEN}` — copy this verbatim into your "
            f"grading table as `bundle_id` (the verdict stage rejects a table whose "
            f"bundle_id does not match this bundle).\n"
        )
        bundle_with_token = build_bundle(blinded, redact_meta=True, header_text=header_b)
        bundle_id = _bundle_id_of(bundle_with_token)
        bundle_b = bundle_with_token.replace(_BUNDLE_ID_TOKEN, bundle_id)
        with open(os.path.join(out_dir, "comparison.blinded.md"), "w", encoding="utf-8") as fh:
            fh.write(bundle_b)
        write_json(
            os.path.join(out_dir, "blinding.json"),
            {
                "mapping": mapping,
                "run_keys_sha": _run_keys_sha(manifest),
                "bundle_id": bundle_id,
                "protocol_sha": protocol_sha,
            },
        )
        print(
            f"wrote runs/{args.subdir}/comparison.blinded.md (+ sealed blinding.json — "
            f"graders read ONLY the blinded bundle)"
        )
    # Stage-exit recheck (FRESH read): see cmd_extract.
    _require_recorded_protocol(manifest, "compare (stage exit)")
    return 0


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #


def _registered_contrasts(raw: dict | None = None) -> set:
    """The pre-registered gating pairs, derived from arm roles: (control,
    primary) and (primary, dual-primary). Everything else — probes, reversed or
    arbitrary pairs — is non-gating by construction (DECISION_RULE.md)."""
    arms = (raw if raw is not None else _load_configs()).get("arms", [])
    by_role = {}
    for c in arms:
        by_role.setdefault(c.get("role", ""), []).append(c.get("id"))
    pairs = set()
    if by_role.get("control") and by_role.get("primary"):
        pairs.add((by_role["control"][0], by_role["primary"][0]))
    if by_role.get("primary") and by_role.get("dual-primary"):
        pairs.add((by_role["primary"][0], by_role["dual-primary"][0]))
    return pairs


def _protocol_violations(manifest: dict, raw: dict | None = None) -> list[str]:
    """Pre-registered campaign protocol checks (DECISION_RULE.md): the corpus
    floor and the k=2 design. A verdict computed under violations is labeled
    NON-GATING — a rehearsal or subset run must never be mistaken for a
    campaign-grade decision."""
    violations: list[str] = []
    strata = manifest.get("case_strata") or {}
    n_real = sum(1 for s in strata.values() if s != "fixture")
    n_s2 = sum(1 for s in strata.values() if s == "s2_historical")
    n_s3 = sum(1 for s in strata.values() if s == "s3_negative")
    if n_real < 8:
        violations.append(f"corpus floor: {n_real} non-fixture cases < 8")
    if n_s2 < 3:
        violations.append(f"corpus floor: {n_s2} s2_historical cases < 3")
    if n_s3 < 2:
        violations.append(f"corpus floor: {n_s3} s3_negative cases < 2")
    if manifest.get("k") != 2:
        violations.append(f"k={manifest.get('k')} != 2 (pre-registered design)")
    if manifest.get("k_overrides"):
        violations.append(f"k_overrides present: {manifest['k_overrides']} (all arms run k=2)")
    if not manifest.get("corpus_verified"):
        violations.append(
            "corpus was not verified by this run (manifest.corpus_verified is not "
            "set — pre-registration requires verified cases)"
        )
    if manifest.get("protocol_drift_during_run"):
        violations.append(
            "protocol files changed WHILE the matrix ran — results cannot be "
            "attributed to the recorded protocol identity"
        )
    if manifest.get("corpus_drift_during_run"):
        violations.append(
            "a corpus plan changed WHILE the matrix ran — arms may have reviewed "
            "different bytes under one run key"
        )
    if any(s == "fixture" for s in strata.values()):
        violations.append(
            "manifest contains the fabricated fixture case (CI/smoke/rehearsal "
            "data must never shape a campaign verdict)"
        )
    if sorted(manifest.get("config_ids") or []) != sorted(_all_arm_ids(raw)):
        violations.append(
            f"manifest arms {sorted(manifest.get('config_ids') or [])} != the "
            f"pre-registered five-arm set {sorted(_all_arm_ids())} (subset runs "
            f"are never campaign-grade)"
        )
    return violations


def _realized_grid_violations(manifest: dict, results) -> list[str]:
    """The manifest's DECLARED schedule must equal the REALIZED run grid: every
    (case, arm, repeat) in case_ids × config_ids × range(k) present exactly
    once. A manifest whose repeat-1 keys simply vanished would otherwise treat
    one repeat per arm as the whole schedule and still gate (round-7 P1)."""
    k = int(manifest.get("k") or 0)
    overrides = manifest.get("k_overrides") or {}
    expected = {
        (case, arm, rep)
        for case in manifest.get("case_ids") or []
        for arm in manifest.get("config_ids") or []
        for rep in range(int(overrides.get(arm, k)))
    }
    realized = [(rr.case_id, rr.config_id, rr.repeat_idx) for rr in results]
    violations = []
    if len(realized) != len(set(realized)):
        violations.append("duplicate (case, arm, repeat) results in the manifest run set")
    missing = expected - set(realized)
    extra = set(realized) - expected
    if missing:
        violations.append(f"realized run grid is missing {sorted(missing)[:6]}")
    if extra:
        violations.append(f"realized run grid has unscheduled entries {sorted(extra)[:6]}")
    return violations


def cmd_verdict(args: argparse.Namespace) -> int:
    import verdict as verdict_mod
    from eval_core.models import to_jsonable

    # ONE snapshot: the drift comparison, the control-arm resolution, and the
    # five-arm/contrast checks all read the same bytes (see cmd_extract).
    snap = _protocol_snapshot()
    manifest, results = _manifest_results(args.subdir)
    grades = read_json(args.grades)
    blinding_path = os.path.join(RUNS_DIR, args.subdir, "blinding.json")
    blinded_grading = os.path.exists(blinding_path)
    if blinded_grading:
        blinding = read_json(blinding_path)
        stored_sha = blinding.get("run_keys_sha")  # type: ignore[union-attr]
        if stored_sha != _run_keys_sha(manifest):
            raise SystemExit(
                "blinding.json run_keys_sha is missing or was generated for a "
                "DIFFERENT run set than this manifest (stale subdirectory reuse "
                "would silently swap arm attribution); re-run `compare --blinded`."
            )
        if blinding.get("protocol_sha") != _identity_sha(  # type: ignore[union-attr]
            manifest.get("protocol") or {}
        ):
            raise SystemExit(
                "blinding.json protocol_sha is missing or names a DIFFERENT "
                "protocol identity than this manifest records — the bundle was "
                "blinded under another protocol; re-run `compare --blinded`."
            )
        # The mapping is DERIVED, not trusted: recompute it from the manifest
        # exactly as compare did; a hand-swapped blinding.json (silently
        # re-attributing grades between engines) is refused.
        from eval_core.compare import derive_blind_mapping

        salt = hashlib.sha256("|".join(sorted(manifest["run_keys"])).encode("utf-8")).hexdigest()
        expected_mapping = derive_blind_mapping(sorted(manifest.get("config_ids") or []), salt)
        if blinding.get("mapping") != expected_mapping:  # type: ignore[union-attr]
            raise SystemExit(
                "blinding.json mapping does not match the deterministic mapping "
                "derived from this manifest — arm attribution cannot be trusted; "
                "re-run `compare --blinded`."
            )
        expected_bundle = blinding.get("bundle_id")  # type: ignore[union-attr]
        if not expected_bundle or grades.get("bundle_id") != expected_bundle:
            raise SystemExit(
                f"grading table bundle_id {grades.get('bundle_id')!r} does not "
                f"match this subdirectory's blinded bundle {expected_bundle!r} — "
                f"the grades were produced from a different (or unidentified) "
                f"bundle; re-grade the current comparison.blinded.md."
            )
        # And the bundle file itself must still hash to that id (the id IS the
        # hash of the grader-visible bytes with the id slot tokenized).
        bundle_path = os.path.join(RUNS_DIR, args.subdir, "comparison.blinded.md")
        try:
            with open(bundle_path, encoding="utf-8") as fh:
                bundle_text = fh.read()
        except OSError:
            raise SystemExit(f"blinded bundle missing at {bundle_path}; re-run compare --blinded")
        restored = bundle_text.replace(expected_bundle, _BUNDLE_ID_TOKEN)
        if _bundle_id_of(restored) != expected_bundle:
            raise SystemExit(
                "comparison.blinded.md does not hash to its recorded bundle_id — "
                "the bundle was modified after blinding.json was written; re-run "
                "compare --blinded and re-grade."
            )
        grades = verdict_mod.unblind(grades, blinding["mapping"])  # type: ignore[index]
    control = args.control or _control_id(snap["raw_config"])
    runs_jsonable = [to_jsonable(rr) for rr in results]
    out = verdict_mod.gates(grades, runs_jsonable, control=control, candidate=args.candidate)
    violations = _protocol_violations(manifest, raw=snap["raw_config"])
    violations.extend(_realized_grid_violations(manifest, results))
    recorded_protocol = manifest.get("protocol") or {}
    if not recorded_protocol:
        violations.append("manifest records no protocol identity (pre-provenance run)")
    elif recorded_protocol != snap["identity"]:
        violations.append(
            "the LIVE protocol (decision rule / configs / candidates / extraction "
            "identity / contrasts) differs from the one recorded at run time — a "
            "post-run protocol edit cannot re-define what this campaign's verdict "
            "means; re-run under the current protocol or check out the recorded one"
        )
    if not blinded_grading:
        violations.append(
            "no blinded bundle (blinding.json absent) — the pre-registered protocol "
            "grades ONLY the blinded extraction bundle; unblinded grades are "
            "informational, never campaign-grade"
        )
    recorded_contrasts = {
        tuple(c) for c in (recorded_protocol.get("registered_contrasts") or [])
    } or _registered_contrasts(snap["raw_config"])
    contrast = (control, args.candidate)
    if contrast not in recorded_contrasts:
        violations.append(
            f"contrast {contrast} is not a pre-registered gating pair "
            f"({sorted(recorded_contrasts)}); probes and arbitrary pairs are "
            f"informational only"
        )
    out["gating"] = not violations and out["verdict"] != verdict_mod.UNDETERMINED
    out["protocol_violations"] = violations
    out_path = os.path.join(RUNS_DIR, f"{args.subdir}-verdict-{args.candidate}.json")
    write_json(out_path, out)
    label = "" if out["gating"] else " [NON-GATING]"
    print(f"verdict ({args.candidate} vs {control}): {out['verdict']}{label}")
    if violations:
        print("  protocol violations (rehearsal/subset run — NOT campaign-grade):")
        for v in violations:
            print(f"    - {v}")
    if out.get("evidence_gaps"):
        print(f"  evidence gaps (zero OK repeats): {out['evidence_gaps']}")
    if out["regressions"]:
        print(f"  regressions: {out['regressions']}")
    if out["judgment_flags"]:
        print(f"  flagged for judgment: {out['judgment_flags']}")
    if out["improvements"]:
        print(f"  strict improvements: {out['improvements']}")
    print(f"  FPs: candidate={out['fp_candidate']} control={out['fp_control']}")
    print(f"wrote {os.path.relpath(out_path, HERE)}")
    return 0


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify-corpus", help="materialize + contract-check every case")
    pv.add_argument("--strata", nargs="*", default=None)
    pv.set_defaults(func=cmd_verify_corpus)

    ps = sub.add_parser("smoke", help="tiny end-to-end run (fixture case, k=1)")
    ps.add_argument("--configs", default="", help="CSV arm ids (default: the control arm)")
    ps.set_defaults(func=cmd_smoke)

    pr = sub.add_parser("run", help="the arm matrix; saves reviews (resumable)")
    pr.add_argument("--subdir", required=True)
    pr.add_argument("--configs", default="", help="CSV arm ids (default: all)")
    pr.add_argument("--k", type=int, default=2)
    pr.add_argument("--k-per", default="", help="per-arm overrides, e.g. C=1,D=1")
    pr.add_argument("--strata", nargs="*", default=None)
    pr.add_argument("--cases", default="", help="CSV case ids (default: all selected strata)")
    pr.add_argument("--max-parallel", type=int, default=3)
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("extract", help="neutral findings extraction over stored reviews")
    pe.add_argument("--subdir", required=True)
    pe.add_argument("--force", action="store_true", help="re-extract even if present")
    pe.set_defaults(func=cmd_extract)

    pc = sub.add_parser("compare", help="emit the grading bundle from extractions")
    pc.add_argument("--subdir", required=True)
    pc.add_argument("--blinded", action="store_true")
    pc.set_defaults(func=cmd_compare)

    pj = sub.add_parser("verdict", help="mechanical gates over a final grading table")
    pj.add_argument("--subdir", required=True)
    pj.add_argument("--grades", required=True, help="path to the reconciled grading table JSON")
    pj.add_argument("--candidate", required=True, help="candidate arm id (e.g. B)")
    pj.add_argument("--control", default="", help="control arm id (default: role=control)")
    pj.set_defaults(func=cmd_verdict)

    args = ap.parse_args()
    # One choke point: every subcommand that takes --subdir gets it validated
    # before any path is built from it.
    if getattr(args, "subdir", None) is not None:
        _safe_subdir(args.subdir)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
