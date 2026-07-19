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
                  bundle for blind grading.

Usage:
  python tools/reviewer-eval/run_eval.py verify-corpus
  python tools/reviewer-eval/run_eval.py smoke   # smokes the role=control arm
  python tools/reviewer-eval/run_eval.py run --configs A,B,C,D --k 2 --k-per C=1,D=1
  python tools/reviewer-eval/run_eval.py compare --subdir full --blinded
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Make `engine` and `adapters` importable (the eval dir is the package root).
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from adapters.corpus_loader import CorpusLoader  # noqa: E402
from adapters.openai_review_loader import find_repo_root  # noqa: E402
from engine.store import RunStore, read_json, write_json  # noqa: E402

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
    from engine.models import Config

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
    from engine.runner import run_matrix

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
    if _verify_cases(loader, cases):
        return 1
    reviewer = _build_reviewer(repo_root)
    print(f"codex CLI version: {reviewer.cli_version()}")
    store = RunStore(os.path.join(RUNS_DIR, "smoke"))
    # smoke is a LIVE plumbing/auth/connectivity check (the README's "first real codex
    # call"), so it must actually exercise codex every time — clear any cached smoke
    # artifacts so run_matrix never resumes a stale success.
    for name in os.listdir(store.root):
        if name.endswith(".json"):
            os.remove(os.path.join(store.root, name))
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
    from engine.runner import run_matrix

    if not _safe_subdir(args.subdir):
        print(
            f"invalid --subdir {args.subdir!r}: must stay within the runs directory",
            file=sys.stderr,
        )
        return 1
    # A run attempt into <subdir> supersedes any prior experiment there: invalidate an
    # EXISTING manifest now, so even an early exit (bad configs, zero cases, a corpus-
    # load error) can't leave the previous run's manifest live for compare. A fresh
    # subdir stays manifest-less here; compare fails closed on a missing manifest.
    _invalidate_manifest_if_exists(args.subdir)
    repo_root = find_repo_root()
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
    reviewer = _build_reviewer(repo_root)
    store = RunStore(os.path.join(RUNS_DIR, args.subdir))
    # If run_matrix raises (ConfoundMismatch, CLIVersionMismatch, an interrupt, ...)
    # the up-front failure marker stays in place and the exception propagates.
    results = run_matrix(
        cases,
        configs,
        reviewer,
        store,
        k=args.k,
        max_parallel=args.max_parallel,
        progress=lambda m: print(f"  {m}"),
        treatment_fields=_treatment_fields(),
        k_overrides=k_overrides,
    )
    ok = sum(1 for r in results if r.ok)
    infra = [r for r in results if not r.ok]
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
    write_json(
        _manifest_path(args.subdir),
        {
            "run_ids": run_ids,
            "configs": [c.id for c in configs],
            "base_prompt_sha": getattr(reviewer, "base_prompt_sha", ""),
            # Provenance for humans reading the manifest (compare derives repeats
            # from the artifacts themselves).
            "k": args.k,
            "k_per": k_overrides,
        },
    )
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
    from engine.compare import build_bundle

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
        from engine.compare import apply_blinding, derive_blind_mapping, sanitize_model_refs

        config_ids = sorted({r.config_id for r in runs})
        model_names = sorted({r.model for r in runs if r.model})
        salt = hashlib.sha256("|".join(sorted(wanted or ())).encode("utf-8")).hexdigest()
        mapping = derive_blind_mapping(config_ids, salt)
        blind_runs = apply_blinding(runs, mapping, model_names)
        # Belt-and-suspenders: sanitize the WHOLE rendered bundle once more, so a
        # model name that leaked through any renderer path is still scrubbed.
        blind_bundle = sanitize_model_refs(build_bundle(blind_runs, redact_meta=True), model_names)
        blind_path = os.path.join(out_dir, "comparison.blinded.md")
        with open(blind_path, "w", encoding="utf-8") as fh:
            fh.write(blind_bundle)
        # The seal: graders must never read this file (or config/configs.json).
        # Unblind the finished grading table by joining on these labels.
        write_json(
            os.path.join(out_dir, "blinding.json"),
            {
                "mapping": mapping,
                "models": {r.config_id: r.model for r in runs},
                "efforts": {r.config_id: getattr(r, "effort", "") for r in runs},
                "salt_source": "sha256 of the sorted manifest run_ids",
            },
        )
        print(f"wrote {blind_path} (blind labels: {', '.join(sorted(mapping.values()))})")
        print(
            "  BLINDING: give graders ONLY comparison.blinded.md; they must not read "
            "blinding.json or config/configs.json. Unblind finished tables via blinding.json."
        )
    print(
        "Next: have an LLM (a subagent or in-conversation) read it into the "
        "caught/missed/false-positive table, then decide."
    )
    return 0


def main() -> int:
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
    pr.add_argument("--configs", default="A,B")
    pr.add_argument("--strata", nargs="*", default=None)
    pr.add_argument("--subdir", default="full")
    pr.add_argument("--k", type=int, default=1, help="repeats per (case, config)")
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

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
