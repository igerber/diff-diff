"""Run-matrix executor: case × config × repeat, with resume + bounded parallelism.

Stdlib-only (``concurrent.futures`` thread pool; reviewer calls are
subprocess/IO-bound so threads give real parallelism). The runner never spawns
codex itself — it calls the injected ``reviewer`` (duck-typed: must provide
``review(case, config, repeat) -> ReviewOutput``, ``cli_version() -> str``, and
``experiment_tag(config) -> str``). A reviewer exception becomes an INFRA_ERROR
RunResult (never a missed bug). Completed runs are skipped on resume via the
content-hash key in ``eval_core.store``.

The runner ASSERTS the Codex CLI version is identical across arms, and that the
arms differ ONLY in the declared ``treatment_fields`` (default: model), each in
single-field contrasts — any other drift would confound the comparison and must
abort the run.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from eval_core.models import INFRA_ERROR, Case, Config, RunResult, to_jsonable
from eval_core.store import RunStore, run_key

# Config fields that are held-constant confounds unless declared as treatments.
# `model` IS in this list: an effort-only experiment must hold model constant, or
# the arms are silently confounded — the exact failure the harness exists to stop.
# `variant` (criteria/prompt source) and `mode` (reviewer composition) are the
# generic multi-dimension arm fields; "" everywhere when a harness doesn't use them.
CONFOUND_FIELDS = ("model", "effort", "sandbox", "action_version", "variant", "mode")


class CLIVersionMismatch(RuntimeError):
    """Raised when arms would run under different reviewer CLI versions."""


class ConfoundMismatch(RuntimeError):
    """Raised when the selected arms don't form a clean experiment: they drift in
    a held-constant confound (any of ``CONFOUND_FIELDS`` not declared as a
    treatment), duplicate a treatment tuple, or differ from every other arm in
    more than one treatment field at once (a confounded contrast). Aborted up
    front — a confounded comparison is exactly what the harness exists to avoid."""


def _plan_runs(
    cases: list[Case],
    configs: list[Config],
    k: int,
    k_overrides: Optional[dict] = None,
) -> list[tuple[Case, Config, int]]:
    """Enumerate (case, config, repeat) triples — ``k`` per case, with optional
    per-config overrides (e.g. full repeats on the primary arms, k=1 probes)."""
    jobs: list[tuple[Case, Config, int]] = []
    for case in cases:
        for config in configs:
            reps = max(1, (k_overrides or {}).get(config.id, k))
            for r in range(reps):
                jobs.append((case, config, r))
    return jobs


def _case_snapshot(case: Case) -> dict:
    """Capture the case AS REVIEWED for the comparison bundle (stdlib dict).

    Stored on every RunResult so ``compare`` renders ground truth from what the
    run actually saw, not from the (possibly later-edited) live corpus.
    """
    rerun = case.fixture.get("rerun", {}) if isinstance(case.fixture, dict) else {}
    return {
        "title": case.title,
        "stratum": case.stratum,
        "ground_truth": [to_jsonable(b) for b in case.ground_truth],
        "expect_no_blockers": case.expect_no_blockers,
        "allow_severities": case.allow_severities,
        "known_fp_topics": case.known_fp_topics,
        # Grading context the bundle must show, or rerun/notes-dependent cases can't
        # be graded faithfully: the documented case notes and, for a re-review case,
        # the prior review the reviewer was actually given.
        "notes": case.notes,
        "previous_review": (rerun or {}).get("previous_review", ""),
    }


def run_matrix(
    cases: list[Case],
    configs: list[Config],
    reviewer,
    store: RunStore,
    k: int = 1,
    max_parallel: int = 5,
    progress: Optional[Callable[[str], None]] = None,
    treatment_fields: tuple = ("model",),
    k_overrides: Optional[dict] = None,
) -> list[RunResult]:
    """Execute the full matrix, resuming completed runs, returning all results.

    ``treatment_fields`` declares which Config fields are the experimental
    treatment (default: model only — the classic A/B); everything else in
    ``CONFOUND_FIELDS`` stays a held-constant confound. ``k_overrides`` maps
    config ids to per-config repeat counts (others use ``k``). ``progress`` (if
    given) is called with a short status string per completed run. Reviewer
    errors are captured as INFRA_ERROR results, not raised.
    """

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    # Pin/assert the reviewer CLI version up front so a drift aborts before we
    # spend any compute. This is a FIDELITY check (live CLI must match each config's
    # pin), so it runs for every config — including a single-arm smoke run — not only
    # multi-arm comparisons. A config that doesn't pin a version (cli_version=None)
    # is skipped.
    cli_version = ""
    cli_error: Optional[Exception] = None
    try:
        cli_version = reviewer.cli_version()
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator
        cli_error = exc
        log(f"WARNING: could not read reviewer CLI version: {exc}")
    # Fail closed: if any config PINS a CLI version but we couldn't read the live one,
    # abort rather than run under an unverified CLI (the pin is a recorded==executed
    # confound). Previously a cli_version() failure left cli_version="" and silently
    # skipped the pin check.
    if any(c.cli_version for c in configs) and not cli_version:
        raise CLIVersionMismatch(
            f"configs pin cli_version but the live reviewer CLI version is unavailable"
            f"{f' ({cli_error})' if cli_error else ''}; aborting rather than run under "
            f"an unverified CLI."
        )
    for config in configs:
        if config.cli_version and config.cli_version != cli_version:
            raise CLIVersionMismatch(
                f"config {config.id} pins cli_version={config.cli_version!r} "
                f"but reviewer reports {cli_version!r}; aborting to run under the "
                f"pinned CLI (fidelity / unconfounded A/B)."
            )

    # Only the DECLARED treatment fields may vary across arms. All three checks
    # apply to multi-arm comparisons only (one arm can't be confounded — the
    # single-arm exemption the per-arm smokes rely on):
    #   1. any confound not declared as a treatment must be identical across arms;
    #   2. no two arms may share the same treatment tuple (they'd alias);
    #   3. every arm must differ from at least one other selected arm in EXACTLY
    #      one treatment field — so the selection decomposes into clean
    #      single-factor contrasts, and a jointly-confounded pair (e.g. model AND
    #      effort both changed, with no bridging arm) is refused.
    if len(configs) >= 2:
        for field in CONFOUND_FIELDS:
            if field in treatment_fields:
                continue
            values = {getattr(c, field) for c in configs}
            if len(values) > 1:
                raise ConfoundMismatch(
                    f"configs differ in {field!r} ({sorted(values)}), which is not a "
                    f"declared treatment field ({sorted(treatment_fields)}) — aborting "
                    f"to avoid a confounded comparison."
                )
        treatments = {c.id: tuple(getattr(c, f) for f in treatment_fields) for c in configs}
        if len(set(treatments.values())) != len(configs):
            raise ConfoundMismatch(
                f"two or more configs share the same treatment tuple over "
                f"{sorted(treatment_fields)}; arms must be distinct experiments."
            )

        def _n_diffs(a: Config, b: Config) -> int:
            return sum(1 for f in treatment_fields if getattr(a, f) != getattr(b, f))

        for c in configs:
            if not any(_n_diffs(c, other) == 1 for other in configs if other.id != c.id):
                raise ConfoundMismatch(
                    f"config {c.id} differs from every other selected arm in more than "
                    f"one treatment field ({sorted(treatment_fields)}); each arm needs a "
                    f"single-field contrast partner — select a bridging arm (e.g. the "
                    f"full matrix) instead of a jointly-confounded subset."
                )

    # Experiment identity per config: folds in model/effort/prompt/cli so a
    # rerun with a changed model (under the same config id) gets a distinct key
    # and never resumes a stale run for a different experiment.
    def _tag(config: Config) -> str:
        # Fail closed: the experiment tag is load-bearing for resume. A fallback
        # tag (e.g. "") would weaken the run key so a stale artifact from a
        # different model/backend could be resumed under an unchanged prompt. The
        # reviewer interface REQUIRES experiment_tag; if it can't be computed, abort
        # rather than silently risk a confounded resume.
        try:
            return reviewer.experiment_tag(config)
        except Exception as exc:
            raise RuntimeError(
                f"experiment_tag failed for config {config.id}: {exc}; cannot key "
                f"runs safely — aborting (a fallback tag risks resuming a stale "
                f"experiment)."
            ) from exc

    config_tags = {config.id: _tag(config) for config in configs}

    # Per-case content identity (mirrors _tag): editing a case under the same id
    # must invalidate its cache. Degrade to "" only if the reviewer lacks a
    # case_tag method (a minimal stub) — a real error (e.g. a vanished pinned
    # patch) propagates rather than silently weakening the key.
    def _case_tag(case: Case) -> str:
        try:
            return reviewer.case_tag(case)
        except AttributeError:
            return ""

    case_tags = {case.id: _case_tag(case) for case in cases}

    jobs = _plan_runs(cases, configs, k, k_overrides)
    results: list[RunResult] = []
    pending: list[tuple[Case, Config, int]] = []

    # prompt_sha_for(case) rematerializes the worktree; it's case-scoped for a fixed
    # reviewer, so memoize per case.id to avoid recomputing it for every cached
    # (config, repeat) of the same case on a resumed run.
    _prompt_sha_cache: dict[str, str] = {}

    def _prompt_matches(case, cached) -> bool:
        # Reuse a cached run ONLY if the model would see byte-identical input now —
        # so an edit to the prompt builder / materializer / case busts the cache,
        # regardless of whether the run-key fingerprint happened to capture it.
        # Dissolves the "is the cache identity complete?" question into one check.
        if not cached.prompt_sha:
            return False  # legacy artifact w/o a recorded prompt: rerun to be safe
        try:
            sha = _prompt_sha_cache.get(case.id)
            if sha is None:
                sha = reviewer.prompt_sha_for(case)
                _prompt_sha_cache[case.id] = sha
            return sha == cached.prompt_sha
        except AttributeError:
            return True  # minimal reviewer w/o prompt_sha_for: fall back to key match
        except Exception as exc:  # noqa: BLE001 - can't verify -> rerun
            log(f"WARNING: prompt re-verify failed for {case.id}: {exc}; will rerun")
            return False

    # Resume: load any already-completed runs, keyed on the experiment identity,
    # AND verified to match the prompt the reviewer would build now.
    for case, config, r in jobs:
        key = run_key(case.id, config.id, r, config_tags[config.id], case_tags[case.id])
        cached = store.load(key)
        if cached is not None and cached.ok and _prompt_matches(case, cached):
            results.append(cached)
            log(f"resume: {case.id} {config.id} r{r} (cached)")
        else:
            pending.append((case, config, r))

    log(
        f"{len(results)} cached, {len(pending)} to run "
        f"({len(cases)} cases × {len(configs)} configs)"
    )

    def _execute(job: tuple[Case, Config, int]) -> RunResult:
        case, config, r = job
        key = run_key(case.id, config.id, r, config_tags[config.id], case_tags[case.id])
        snap = _case_snapshot(case)
        t0 = time.monotonic()
        try:
            out = reviewer.review(case, config, r)
            rr = RunResult(
                case_id=case.id,
                config_id=config.id,
                repeat_idx=r,
                review_markdown=out.review_markdown,
                cli_version=out.cli_version or cli_version,
                model=config.model,
                effort=config.effort,
                latency_s=out.latency_s or (time.monotonic() - t0),
                usage=out.usage,
                prompt_sha=str((out.usage or {}).get("prompt_sha", "")),
                run_id=key,
                case_snapshot=snap,
            )
        except Exception as exc:  # noqa: BLE001 - infra failures are data
            rr = RunResult(
                case_id=case.id,
                config_id=config.id,
                repeat_idx=r,
                cli_version=cli_version,
                model=config.model,
                effort=config.effort,
                latency_s=time.monotonic() - t0,
                run_id=key,
                case_snapshot=snap,
                infra_error=f"{type(exc).__name__}: {exc}",
            )
        return rr

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
            futures = {pool.submit(_execute, job): job for job in pending}
            for fut in as_completed(futures):
                case, config, r = futures[fut]
                rr = fut.result()
                key = run_key(case.id, config.id, r, config_tags[config.id], case_tags[case.id])
                # Persist successes and infra-errors alike (infra-errors are
                # surfaced but excluded from the bundle; persisting avoids
                # re-run churn while still being visibly non-ok).
                store.save(key, rr)
                results.append(rr)
                status = INFRA_ERROR if not rr.ok else f"{rr.latency_s:.0f}s"
                log(f"done: {case.id} {config.id} r{r} [{status}]")

    return results


__all__ = ["run_matrix", "CLIVersionMismatch", "ConfoundMismatch"]
