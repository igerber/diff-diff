"""The plan-reviewer-under-test: runs one arm's engine over one plan case.

For each (case, config, repeat) it materializes the case's repo state at
``base_sha`` in a detached worktree, renders the arm's reviewer prompt (the
variant's criteria + the plan text), and invokes the reviewer(s):

* ``mode == "single"`` — one headless ``claude -p`` subprocess (read-only
  tools, pinned ``--model``), cwd = the case worktree so the reviewer can
  verify the plan's claims against the repo as it was.
* ``mode == "dual:<codex-model>"`` — the ``claude -p`` reviewer AND a codex
  reviewer (reusing the production ``openai_review.call_codex``) run in
  parallel over the same rendered prompt, each blind to the other; then one
  additional ``claude -p`` invocation executes the candidate merge+verify
  prompt in the same worktree. The MERGED report is the graded artifact (it is
  what the production engine emits); both raw reviews are kept in ``usage``
  for diagnostics.

Recorded == executed: every Claude arm pins a concrete ``--model`` id, the
composite CLI version (claude + codex) is pinned/asserted by the runner, and
``experiment_tag`` folds in the variant's criteria/prompt hashes plus this
module's own invocation source — editing HOW reviewers are invoked busts the
run cache even when prompts and configs are unchanged.

``prompt_sha_for`` is deliberately NOT implemented: unlike reviewer-eval
(where one prompt-under-validation is shared by all arms), plan-review prompts
differ per arm by design (the criteria ARE the treatment), and the runner's
prompt re-verification memoizes per case only. Cache identity rests on
``experiment_tag`` (criteria + prompts + invocation source + CLI) and
``case_tag`` (case payload + plan bytes) instead; the runner falls back to key
matching, which those tags make sound.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from eval_core.models import Config, ReviewOutput, to_jsonable

from plan_adapters import worktree
from plan_adapters.criteria_source import ArmArtifacts, render


def _load_openai_review(repo_root: str):
    """Import .claude/scripts/openai_review.py (same-dir importlib pattern)."""
    import importlib.util

    path = os.path.join(repo_root, ".claude", "scripts", "openai_review.py")
    spec = importlib.util.spec_from_file_location("openai_review_for_plan_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load openai_review.py from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ClaudeInvocationError(RuntimeError):
    """The claude -p subprocess failed or returned an unusable payload."""


class PlanReviewer:
    """Duck-typed to the ``eval_core.runner`` reviewer interface:
    ``review``, ``cli_version``, ``experiment_tag``, ``case_tag``."""

    _wt_lock = threading.Lock()

    # Hard per-invocation wall-clock ceilings. A hung subprocess becomes a
    # resumable INFRA_ERROR for one run instead of wedging the campaign.
    CLAUDE_TIMEOUT_S = 2400.0
    CODEX_TIMEOUT_S = 3600.0

    # Read-only built-in tool surface for reviewer subprocesses. No Write/Edit/
    # Bash. Confinement comes from the DEFAULT permission model (see
    # _claude_argv): in-worktree reads are auto-allowed, outside reads are
    # denied headlessly — confidentiality, not just mutation, is in scope.
    CLAUDE_TOOLS = "Read,Grep,Glob"

    def __init__(
        self,
        repo_root: str,
        runs_root: str,
        artifacts: dict[str, ArmArtifacts],
        extraction_model: str = "",
        openai_mod=None,
    ):
        self.repo_root = repo_root
        self.runs_root = runs_root
        self.worktrees_root = os.path.join(runs_root, ".worktrees")
        self.artifacts = artifacts
        self.extraction_model = extraction_model
        # Prefer the wrapper module loaded inside the protocol snapshot's
        # read→import→re-read bracket: dual arms then provably execute the
        # exact bytes the recorded identity hashed (a disk edit after the
        # snapshot can never reach execution). The fallback load exists only
        # for direct/non-campaign construction.
        self._mod = openai_mod if openai_mod is not None else _load_openai_review(repo_root)
        self._cli_version: Optional[str] = None
        # Fingerprint of the invocation contract: this module's claude/codex
        # call sites plus the production codex wrapper it reuses.
        self.invocation_sha = self._invocation_sha()
        # Per-variant artifact fingerprints (criteria + prompts are the treatment).
        self.artifact_shas = {
            name: hashlib.sha256(
                "|".join([a.criteria, a.reviewer_prompt, a.merge_prompt]).encode("utf-8")
            ).hexdigest()[:16]
            for name, a in artifacts.items()
        }
        import uuid

        self._wt_namespace = uuid.uuid4().hex[:12]

    # -- reviewer interface (duck-typed by eval_core.runner) ----------------- #

    def cli_version(self) -> str:
        """Composite pinned-CLI string: the Claude CLI and (for dual arms) the
        codex CLI both shape execution, so both are part of experiment identity
        and both are asserted against the configs' pin."""
        if self._cli_version is None:
            self._cli_version = (
                f"{self._one_version(['claude', '--version'])} | "
                f"{self._one_version(['codex', '--version'])}"
            )
        return self._cli_version

    @staticmethod
    def _one_version(argv: list[str]) -> str:
        try:
            cp = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=60)
            return (cp.stdout or cp.stderr).strip().splitlines()[0] or "unknown"
        except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
            return f"{argv[0]}-not-installed"

    def experiment_tag(self, config: Config) -> str:
        art_sha = self.artifact_shas.get(config.variant)
        if art_sha is None:
            raise RuntimeError(
                f"config {config.id} declares variant={config.variant!r} but no "
                f"artifacts were loaded for it (have: {sorted(self.artifact_shas)})."
            )
        raw = "|".join(
            [
                config.variant,
                config.mode,
                config.model,
                config.effort,
                self.cli_version(),
                art_sha,
                self.invocation_sha,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _invocation_sha(self) -> str:
        """Hash the FULL invocation contract: this module's file bytes plus the
        reused production codex wrapper's file bytes.

        Whole-file (not selected ``getsource`` methods) deliberately: helpers
        like ``_plan_text``, the render path, and worktree calls all shape what
        a reviewer actually sees, so any edit to either module busts the run
        cache. Coarser than necessary is the safe direction for cache identity.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        paths = [
            __file__,
            os.path.join(here, "criteria_source.py"),
            os.path.join(here, "worktree.py"),
            getattr(self._mod, "__file__", None),
        ]
        h = hashlib.sha256()
        for path in paths:
            if path and os.path.exists(path):
                with open(path, "rb") as fh:
                    h.update(fh.read())
            else:
                h.update(b"module-file-unavailable")
        return h.hexdigest()[:16]

    def case_tag(self, case) -> str:
        """Fingerprint of the whole case INCLUDING the plan file's bytes, so any
        edit (metadata or plan text) invalidates cached runs and their stored
        snapshots. A symbolic ``base_sha`` (the fixture case's ``HEAD``) is
        RESOLVED to its commit before hashing — otherwise a moved HEAD would
        silently resume a run against a different repo state. Fail-loud on a
        declared-but-missing plan file."""
        payload = to_jsonable(case)
        fixture = payload.get("fixture") if isinstance(payload.get("fixture"), dict) else {}
        fixture.pop("_case_dir", "")  # machine-local; excluded from the hash
        base = fixture.get("base_sha", "")
        if base:
            cp = subprocess.run(
                ["git", "rev-parse", "--verify", f"{base}^{{commit}}"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                raise ClaudeInvocationError(
                    f"{case.id}: base_sha {base!r} does not resolve: {cp.stderr.strip()}"
                )
            fixture["base_sha"] = cp.stdout.strip()
        h = hashlib.sha256()
        h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        h.update(self._plan_text(case).encode("utf-8"))
        return h.hexdigest()[:16]

    # -- internals ----------------------------------------------------------- #

    @staticmethod
    def _plan_text(case) -> str:
        # A campaign freezes each plan's bytes before scheduling (run_eval
        # stores them on the fixture); every arm/repeat renders from that
        # immutable copy so a mid-campaign edit can never split the arms.
        frozen = case.fixture.get("_plan_text")
        if isinstance(frozen, str):
            return frozen
        case_dir = case.fixture.get("_case_dir", "")
        rel = case.fixture.get("plan", "plan.md")
        if os.path.isabs(rel):
            raise ClaudeInvocationError(
                f"{case.id}: fixture.plan must be relative to the case directory"
            )
        base = os.path.realpath(case_dir or ".")
        full = os.path.realpath(os.path.join(base, rel))
        if full != base and not full.startswith(base + os.sep):
            raise ClaudeInvocationError(
                f"{case.id}: fixture.plan {rel!r} escapes its case directory"
            )
        if not os.path.exists(full):
            raise ClaudeInvocationError(f"{case.id}: plan file not found at {full}")
        with open(full, encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def _claude_argv(model: str, tools: str) -> list[str]:
        """The exact argv shape for every headless claude invocation.

        DELIBERATELY no ``--permission-mode bypassPermissions``: under the
        default permission model, reads INSIDE the working directory (the
        detached case worktree) are auto-allowed, while reads outside it
        require an approval that a headless ``-p`` session cannot grant — so
        the filesystem read surface is confined to the worktree. Bypass would
        let a hostile plan/review instruct the reviewer to read arbitrary
        local files (confidentiality, not just mutation, is in scope).

        ``--safe-mode`` strips ALL per-machine customization (user CLAUDE.md,
        plugins, hooks, MCP servers — which ``--tools`` alone does not
        restrict): reviewer behavior must be a function of the pinned model +
        prompt, not whatever is configured on the operator's machine, and no
        MCP tool may widen the read surface. Contract-tested in
        tests/test_plan_review_eval.py.
        """
        return [
            "claude",
            "-p",
            "--safe-mode",
            "--model",
            model,
            "--tools",
            tools,
            "--no-session-persistence",
            "--output-format",
            "json",
        ]

    def _call_claude(self, prompt: str, model: str, cwd: str, timeout_s: float) -> tuple[str, dict]:
        """One headless ``claude -p`` invocation; returns (text, usage-ish dict).

        ``--output-format json`` gives a structured result including the models
        actually used; when that field is present, a pinned model that does not
        appear there is a recorded!=executed violation and raises.
        """
        argv = self._claude_argv(model, self.CLAUDE_TOOLS)
        t0 = time.monotonic()
        try:
            cp = subprocess.run(
                argv,
                input=prompt,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeInvocationError(f"claude -p timed out after {timeout_s:.0f}s") from exc
        except FileNotFoundError as exc:
            raise ClaudeInvocationError("claude CLI not installed / not on PATH") from exc
        latency = time.monotonic() - t0
        if cp.returncode != 0:
            tail = (cp.stderr or cp.stdout or "").strip()[-2000:]
            raise ClaudeInvocationError(f"claude -p exited {cp.returncode}: {tail}")
        try:
            obj = json.loads(cp.stdout)
        except ValueError as exc:
            raise ClaudeInvocationError(
                f"claude -p emitted non-JSON output despite --output-format json: "
                f"{cp.stdout[:500]!r}"
            ) from exc
        text = obj.get("result")
        if not isinstance(text, str) or not text.strip():
            raise ClaudeInvocationError(
                f"claude -p JSON payload has no usable 'result' field: {sorted(obj)}"
            )
        # recorded == executed: assert the pinned model actually served the call
        # whenever the payload discloses the models used.
        used = obj.get("modelUsage")
        if isinstance(used, dict) and used and not any(model in k for k in used):
            raise ClaudeInvocationError(
                f"pinned --model {model} but the response reports models "
                f"{sorted(used)} — recorded != executed; fix the pin."
            )
        usage = {
            "claude_latency_s": round(latency, 3),
            "claude_models_used": sorted(used) if isinstance(used, dict) else [],
            "total_cost_usd": obj.get("total_cost_usd"),
        }
        return text, usage

    # -- the reviewer-under-test --------------------------------------------- #

    def review(self, case, config: Config, repeat_idx: int) -> ReviewOutput:
        art = self.artifacts.get(config.variant)
        if art is None:
            raise RuntimeError(f"no artifacts for variant {config.variant!r}")
        mode = config.mode or "single"
        if mode != "single" and not mode.startswith("dual:"):
            raise NotImplementedError(
                f"config {config.id} requested mode={mode!r}; supported: "
                f"'single' or 'dual:<codex-model>'. Recorded must equal executed."
            )
        if mode.startswith("dual:") and not art.merge_prompt:
            raise NotImplementedError(
                f"config {config.id} is a dual arm but variant {config.variant!r} "
                f"has no merge prompt — the control engine has no dual mode."
            )

        plan = self._plan_text(case)
        # render() is strict: any template token without a provided value raises,
        # so a literal __CRITERIA__ can never reach a reviewer.
        prompt = render(art.reviewer_prompt, criteria=art.criteria, plan=plan)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        worktree_key = f"{self._wt_namespace}.{case.id}.{config.id}.r{repeat_idx}"
        with self._wt_lock:
            mat = worktree.materialize(
                case.id,
                dict(case.fixture),
                self.repo_root,
                self.worktrees_root,
                worktree_key=worktree_key,
            )
        t0 = time.monotonic()
        usage: dict = {"prompt_sha": prompt_sha, "mode": mode}
        try:
            if mode == "single":
                review_md, cu = self._call_claude(
                    prompt, config.model, mat.worktree_dir, self.CLAUDE_TIMEOUT_S
                )
                usage.update(cu)
            else:
                codex_model = mode.split(":", 1)[1]
                # Both reviewers run in parallel over the same rendered prompt,
                # each blind to the other.
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_claude = pool.submit(
                        self._call_claude,
                        prompt,
                        config.model,
                        mat.worktree_dir,
                        self.CLAUDE_TIMEOUT_S,
                    )
                    f_codex = pool.submit(
                        self._mod.call_codex,
                        prompt,
                        codex_model,
                        mat.worktree_dir,
                        effort=config.effort,
                        timeout_s=self.CODEX_TIMEOUT_S,
                    )
                    raw_claude, cu = f_claude.result()
                    raw_codex, codex_usage = f_codex.result()
                usage.update(cu)
                usage["codex_usage"] = dict(codex_usage or {})
                usage["raw_claude_review"] = raw_claude
                usage["raw_codex_review"] = raw_codex
                # Merge + verify stage: one more claude -p in the same worktree
                # (verification needs repo access). The merged report is the
                # graded artifact — it is what the production engine emits.
                merge_prompt = render(
                    art.merge_prompt,
                    criteria=art.criteria,
                    plan=plan,
                    review_a=raw_claude,
                    review_b=raw_codex,
                )
                review_md, mu = self._call_claude(
                    merge_prompt, config.model, mat.worktree_dir, self.CLAUDE_TIMEOUT_S
                )
                usage["merge_latency_s"] = mu.get("claude_latency_s")
        finally:
            with self._wt_lock:
                worktree.cleanup(mat.worktree_dir, self.repo_root, self.worktrees_root)

        return ReviewOutput(
            review_markdown=review_md,
            cli_version=self.cli_version(),
            latency_s=time.monotonic() - t0,
            usage=usage,
        )

    # -- extraction stage (post-run; not part of run identity) --------------- #

    def extract(self, review_markdown: str, extraction_prompt: str) -> tuple[str, list[str]]:
        """Neutral findings extraction over one raw/merged review.

        Runs OUTSIDE the run matrix (a post-processing stage over stored
        artifacts) with the pinned extraction model; no repo tools (``--tools
        ""``) — extraction is a pure text transformation. Returns
        ``(extraction_text, models_used)``; the same recorded==executed model
        check as the reviewer path applies (the extraction pin grades every
        arm, so a silent fallback would change methodology mid-campaign).
        """
        if not self.extraction_model:
            raise RuntimeError("configs.json must pin extraction.model")
        prompt = render(extraction_prompt, review=review_markdown)
        argv = self._claude_argv(self.extraction_model, "")
        try:
            cp = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeInvocationError("extraction claude -p timed out") from exc
        if cp.returncode != 0:
            tail = (cp.stderr or cp.stdout or "").strip()[-2000:]
            raise ClaudeInvocationError(f"extraction claude -p exited {cp.returncode}: {tail}")
        try:
            obj = json.loads(cp.stdout)
        except ValueError as exc:
            raise ClaudeInvocationError(
                f"extraction emitted non-JSON: {cp.stdout[:500]!r}"
            ) from exc
        text = obj.get("result")
        if not isinstance(text, str) or not text.strip():
            raise ClaudeInvocationError("extraction JSON payload has no usable 'result'")
        used = obj.get("modelUsage")
        if isinstance(used, dict) and used and not any(self.extraction_model in k for k in used):
            raise ClaudeInvocationError(
                f"pinned extraction model {self.extraction_model} but the response "
                f"reports models {sorted(used)} — recorded != executed; fix the pin."
            )
        return text, (sorted(used) if isinstance(used, dict) else [])


__all__ = ["PlanReviewer", "ClaudeInvocationError"]
