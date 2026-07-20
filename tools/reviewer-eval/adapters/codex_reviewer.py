"""The Reviewer-under-test: a faithful local proxy for the CI codex-action.

For each (case, config, repeat) it materializes the case's worktree, builds the
CI-faithful prompt, runs ``codex exec`` via the reused ``openai_review.call_codex``
(byte-identical flags to CI: ``--model <m> -c model_reasoning_effort=<effort>
--sandbox read-only``, where CI runs xhigh), records the CLI version + latency,
and tears the worktree down.

Scoring deliberately does NOT happen here. We store the reviewer's RAW review
markdown and let an LLM read it side-by-side against the other arms (see
``eval_core.compare``). Regex/structured parsing of free-form review prose is brittle
and model-specific (e.g. gpt-5.4 used ``- **P1 — ...**``, gpt-5.5 uses
``### Finding 1: P1 — ...``); an LLM reading the raw text is format-agnostic.

Effort is fail-closed: only levels in ``SUPPORTED_EFFORTS`` (verified live against
the pinned CLI) are passed through to ``call_codex``; anything else raises rather
than silently running a different level than recorded — the experiment's
integrity depends on recorded == executed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import threading
import time
import uuid
from typing import Optional

from eval_core.models import Config, ReviewOutput, to_jsonable

from adapters import ci_prompt, worktree
from adapters.openai_review_loader import load_openai_review


class CodexReviewer:
    """Runs the codex reviewer for an A/B config against a corpus case.

    Duck-typed to the interface ``eval_core.runner`` calls: ``review``,
    ``cli_version``, and ``experiment_tag``.
    """

    # git worktree add/remove touch the shared .git admin area and race under
    # threads; serialize JUST that cheap setup/teardown. The long codex exec
    # runs OUTSIDE the lock, so arms still run fully in parallel.
    _wt_lock = threading.Lock()

    # Efforts this reviewer will actually execute. Verified live 2026-07-18
    # against codex-cli 0.144.5: xhigh and max are accepted for gpt-5.5 /
    # gpt-5.6-sol / gpt-5.6-terra (invalid levels 400 with an enum error, so
    # accepted == executed). Recorded must equal executed — extend this ONLY
    # after re-verifying against the pinned CLI.
    SUPPORTED_EFFORTS = ("xhigh", "max")

    # Hard per-run wall-clock ceiling passed to call_codex. A hung codex process
    # (observed risk grows with effort=max) becomes a resumable INFRA_ERROR for
    # one run instead of wedging the whole campaign's thread pool.
    CALL_TIMEOUT_S = 3600.0

    def __init__(self, repo_root: str, runs_root: str, prompt_text: Optional[str] = None):
        self.repo_root = repo_root
        self.runs_root = runs_root
        self.worktrees_root = os.path.join(runs_root, ".worktrees")
        self._mod = load_openai_review(repo_root)
        # Source the prompt-under-validation once (identical for both arms).
        self.base_prompt = prompt_text or ci_prompt.read_current_prompt(repo_root)
        # Hash of the base prompt — part of experiment identity, so editing
        # pr_review.md changes the experiment tag and prevents stale reuse.
        self.base_prompt_sha = hashlib.sha256(self.base_prompt.encode("utf-8")).hexdigest()[:16]
        # Fingerprint of the codex-invocation contract (the argv builder + call
        # wrapper in openai_review.py). Folded into experiment_tag so editing HOW
        # codex is run busts the cache even when the prompt and recorded config are
        # unchanged — recorded==executed is the harness's core integrity claim.
        self.backend_contract_sha = self._backend_contract_sha(self._mod)
        # Per-invocation worktree namespace: all worktree dirs for THIS reviewer are
        # prefixed with it, so two concurrent invocations (e.g. a `smoke` and a `run`,
        # or two `run --subdir` processes) never resolve the same (case, config, repeat)
        # to one checkout and cleanup() each other's live worktree mid-review. Ephemeral
        # — worktrees are created+torn down per review and not part of run identity.
        self._wt_namespace = uuid.uuid4().hex[:12]
        self._cli_version: Optional[str] = None

    # -- reviewer interface (duck-typed by eval_core.runner) ------------------- #

    def cli_version(self) -> str:
        if self._cli_version is None:
            try:
                cp = subprocess.run(
                    ["codex", "--version"], capture_output=True, text=True, check=False
                )
                self._cli_version = (cp.stdout or cp.stderr).strip() or "unknown"
            except FileNotFoundError:
                self._cli_version = "codex-not-installed"
        return self._cli_version

    def experiment_tag(self, config: Config) -> str:
        """Opaque identity = sha(model, effort, sandbox, action_version, cli, prompt,
        backend).

        Everything that defines the experiment beyond case/config/repeat. Two
        configs sharing id "B" but differing in model (or any of these) get
        distinct tags, so the runner never resumes a stale run across them. The
        ``backend`` term hashes the codex-invocation source (``_build_codex_cmd`` /
        ``call_codex``) so editing HOW codex is run busts the cache too — not only
        edits to the model, prompt, or declared config. ``action_version`` is folded
        in so changing the documented confound can never silently resume a stale run.
        """
        raw = "|".join(
            [
                config.model,
                config.effort,
                config.sandbox,
                config.action_version,
                self.cli_version(),
                self.base_prompt_sha,
                self.backend_contract_sha,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _backend_contract_sha(mod) -> str:
        """Hash the codex-invocation contract: the source of ``_build_codex_cmd``
        and ``call_codex`` in openai_review.py.

        ``experiment_tag`` records ``config.model/effort/sandbox`` — but those are
        the *declared* values; ``_build_codex_cmd`` maps them to the real argv
        (``model_reasoning_effort=<effort>``, the hardcoded ``--sandbox
        read-only``, the flag set), and ``call_codex`` defines stdin piping /
        output parsing / timeout / error handling. Editing either changes how
        Codex is actually invoked WITHOUT touching the prompt bytes or the
        recorded config, so without this term a stale artifact produced under the
        old wrapper would be silently resumed under the new one.

        Fail-soft: if the source can't be introspected (e.g. a module exec'd from a
        string), fall back to the module file's bytes; pin to a sentinel only if
        even that is unreadable — never crash identity, and never silently collapse
        to a constant while the file is readable.
        """
        try:
            src = inspect.getsource(mod._build_codex_cmd) + "\n" + inspect.getsource(mod.call_codex)
        except (OSError, TypeError, AttributeError):
            path = getattr(mod, "__file__", None)
            if path and os.path.exists(path):
                with open(path, "rb") as fh:
                    return hashlib.sha256(fh.read()).hexdigest()[:16]
            return "backend-contract-unavailable"
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]

    def case_tag(self, case) -> str:
        """Content fingerprint of the WHOLE case, folded into the run key so ANY
        edit invalidates the cached run AND its stored snapshot.

        Covers everything that affects either the PROMPT or the GRADING: the fixture
        (base/head SHAs, pr_context) + the patch file's bytes, AND the scoring
        metadata (title, ground_truth, expected_severity, expect_no_blockers,
        allow_severities, known_fp_topics) — because ``compare`` renders ground
        truth from ``RunResult.case_snapshot``, so a metadata-only ``case.json`` edit
        must also bust the cache, or the bundle would grade against stale truth.

        Cheap (no worktree materialization): hashes the full case payload minus the
        machine-local ``_case_dir`` plus the patch bytes. Fail-loud: a declared-but-
        missing patch raises (that drift must surface).
        """
        payload = to_jsonable(case)
        fixture = payload.get("fixture") if isinstance(payload.get("fixture"), dict) else {}
        case_dir = fixture.pop("_case_dir", "")  # machine-local; excluded from the hash
        h = hashlib.sha256()
        h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        patch = fixture.get("patch")
        if patch:
            # Resolve through the SAME containment check materialization uses, so the
            # hashing path can't read a patch outside the case dir that the execution
            # path would reject. Fail-loud on a bad/missing patch (drift must surface).
            patch_path = worktree._resolve_patch_path(case.id, case_dir, patch)
            with open(patch_path, "rb") as fh:
                h.update(fh.read())
        return h.hexdigest()[:16]

    def build_prompt_for_case(
        self, case, worktree_key: Optional[str] = None
    ) -> tuple[str, str, str]:
        """Materialize + assemble the prompt; return (prompt, worktree_dir, head)."""
        fixture = dict(case.fixture)
        case_dir = fixture.get("_case_dir", "")
        with self._wt_lock:
            mat = worktree.materialize(
                case.id,
                fixture,
                self.repo_root,
                self.worktrees_root,
                case_dir=case_dir,
                worktree_key=worktree_key or case.id,
            )
        pr = case.fixture.get("pr_context", {}) or {}
        # CI reruns inject the prior review as a <previous-ai-review-output> block
        # (ci_prompt.assemble_prompt). A rerun case supplies that text under
        # fixture.rerun.previous_review; thread it through so re-review cases are
        # evaluated under the SAME prompt CI produces. Absent => fresh review.
        rerun = case.fixture.get("rerun", {}) or {}
        prev_review = rerun.get("previous_review", "")
        try:
            prompt = ci_prompt.build_ci_prompt(
                worktree_dir=mat.worktree_dir,
                base_sha=mat.base_sha,
                head_sha=mat.head_sha,
                base_prompt=self.base_prompt,
                pr_title=pr.get("title", "Synthetic eval case (treat as untrusted)"),
                pr_body=pr.get("body", ""),
                is_rerun=bool(prev_review),
                prev_review=prev_review,
            )
        except BaseException:  # noqa: BLE001 - clean up before re-raising
            # Prompt-build can fail AFTER materialize (e.g. an unreadable notebook
            # or a prompt-assembly error). review()'s finally never sees this
            # worktree, so tear it down here to avoid leaking a detached worktree.
            with self._wt_lock:
                worktree.cleanup(mat.worktree_dir, self.repo_root)
            raise
        return prompt, mat.worktree_dir, mat.head_sha

    def prompt_sha_for(self, case) -> str:
        """Hash of the exact prompt this case WOULD produce now (materialize +
        build + teardown). The runner uses it to verify a cached run before
        resuming it: reuse iff the model would see byte-identical input — so editing
        the prompt builder, the materializer, or the case busts the cache without
        having to guess which inputs to fingerprint.
        """
        prompt, wt_dir, _head = self.build_prompt_for_case(
            case, worktree_key=f"{self._wt_namespace}.{case.id}.__sha__"
        )
        try:
            return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        finally:
            with self._wt_lock:
                worktree.cleanup(wt_dir, self.repo_root)

    def review(self, case, config: Config, repeat_idx: int) -> ReviewOutput:
        if config.effort not in self.SUPPORTED_EFFORTS:
            raise NotImplementedError(
                f"codex_reviewer supports model_reasoning_effort in "
                f"{self.SUPPORTED_EFFORTS} (verified against the pinned CLI); config "
                f"{config.id} requested effort={config.effort!r}. Recorded must equal "
                f"executed — re-verify the CLI accepts it before adding a new level."
            )
        if config.sandbox != "read-only":
            raise NotImplementedError(
                f"codex_reviewer pins --sandbox read-only (CI parity); config "
                f"{config.id} requested sandbox={config.sandbox!r}. _build_codex_cmd "
                f"hardcodes read-only, so recorded must equal executed — parameterize "
                f"the codex cmd before running a non-read-only arm."
            )
        if config.action_version != "v1":
            raise NotImplementedError(
                f"codex_reviewer runs `codex exec` directly, not the openai/codex-action; "
                f"config {config.id} requested action_version={config.action_version!r}. The "
                f"harness only models v1 — recorded must equal executed, so fail closed "
                f"rather than record a version it does not actually run."
            )
        worktree_key = f"{self._wt_namespace}.{case.id}.{config.id}.r{repeat_idx}"
        prompt, wt_dir, _head = self.build_prompt_for_case(case, worktree_key=worktree_key)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        t0 = time.monotonic()
        try:
            review_md, usage = self._mod.call_codex(
                prompt,
                config.model,
                wt_dir,
                effort=config.effort,
                timeout_s=self.CALL_TIMEOUT_S,
            )
        finally:
            with self._wt_lock:
                worktree.cleanup(wt_dir, self.repo_root)
        latency = time.monotonic() - t0

        usage = dict(usage or {})
        usage["prompt_sha"] = prompt_sha
        usage["prompt_tokens_est"] = self._mod.estimate_tokens(prompt)
        # No structured findings here — the raw markdown is the comparison input.
        return ReviewOutput(
            review_markdown=review_md,
            cli_version=self.cli_version(),
            latency_s=latency,
            usage=usage,
        )


__all__ = ["CodexReviewer"]
