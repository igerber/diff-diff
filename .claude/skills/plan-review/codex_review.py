"""Run the codex (reviewer 2) half of the plan-review dual engine.

Called by SKILL.md with an already-rendered reviewer prompt. Detects codex,
invokes `openai_review.call_codex` with the campaign-graded model/effort
pins (gpt-6.1-sol @ xhigh; the production code-review pin) under a read-only sandbox, writes the codex
review. Before invoking, it runs openai_review's own sensitive-file scan +
stderr notice (the codex `--cd` read surface is the whole repo — the same
notice the `/ai-review-local` codex path prints). Exit codes let SKILL.md fall
through to the LOUD single-Claude fallback without wedging the gate:

  0  codex review written to --output
  2  codex unavailable (not installed / not logged in)
  3  codex timed out or errored

Both non-zero exits are treated identically by SKILL.md (loud fallback).
"""

import argparse
import importlib.util
import os
import shutil
import sys

# Codex pins follow the PRODUCTION code-review model (lockstep-tested against the
# CI workflow and openai_review.DEFAULT_MODEL). gpt-5.6-sol was the engine
# Campaign 1 graded (arm C, as-run; exploratory / non-gating); gpt-6.1-sol was
# carried over by the 2026-10 code-review campaign 2
# (tools/reviewer-eval/campaigns/2026-10-gpt61sol-VERDICT.md), NOT re-run through
# plan-review-eval. To re-validate: first re-pin
# tools/plan-review-eval/config/configs.json cli_version (the composite
# "<claude-code version> | codex-cli <version>" string on all five arms) to the
# live Claude Code + codex CLI pair, add a gpt-6.1-sol arm, run that harness
# against the sol baseline (TODO.md row). tests/test_plan_review_skill.py asserts
# these pins.
CODEX_MODEL = "gpt-6.1-sol"
CODEX_EFFORT = "xhigh"
# The timeout equals the code-review harness's CALL_TIMEOUT_S
# (tools/reviewer-eval/adapters/codex_reviewer.py; tests/test_plan_review_skill.py
# pins the equality). It is still an interactive-gate ceiling, not a campaign
# setting: high enough to clear real plan-review runtimes with room to spare,
# low enough to bound the interactive wait. A timeout is treated as codex being unavailable (exit 3 →
# LOUD single-Claude fallback), so err generous.
#
# Raised 1200 → 2400 on 2026-08-08. The old value was calibrated to the campaign's
# observed runtimes (up to ~430s), whose plans were far smaller than production
# ones: a dense ~500-line plan whose reviewer must verify claims against 15+ repo
# files ran 3-4x that, and codex timed out on 6 of 9 rounds of a single v4 plan
# (every failure a genuine `codex exec timed out after 1200.0s`, never an auth or
# availability error). Retrying is the wrong remedy — a failed attempt burns the
# full ceiling before the retry starts, and a round that retried once timed out
# twice for 40 minutes total. Plan review is simply the slow surface here: its
# reviewer reads the plan AND cross-checks every claim against the live tree.
#
# Changing this does NOT require engine re-validation: the "do not change without
# re-validation" note above binds CODEX_MODEL and CODEX_EFFORT (what Campaign 1
# graded). The ceiling is an operational knob and is graded by nothing.
#
# Raised 2400 -> 3600 on 2026-10-17 with the gpt-6.1-sol swap, to equal the
# code-review harness ceiling (the 2026-10 campaigns measured 6.1-sol at
# ~180 s mean / 277 s p95 per code review, faster than 5.6-sol, but plan review
# is the slower surface and the two pins are kept in lockstep); still a hard
# cap, exit 3 = loud single-Claude fallback.
CODEX_TIMEOUT_S = 3600.0


def _load_openai_review(repo_root: str):
    path = os.path.join(repo_root, ".claude", "scripts", "openai_review.py")
    spec = importlib.util.spec_from_file_location("openai_review_for_plan_review", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load openai_review.py from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _codex_present(mod) -> bool:
    # The two conditions inside openai_review._detect_backend (which itself
    # never signals absence — it returns "api" when codex is missing).
    return bool(shutil.which("codex")) and os.path.exists(mod.CODEX_AUTH_PATH)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-file", required=True, help="rendered reviewer prompt")
    ap.add_argument("--repo-root", required=True, help="repo the plan targets (codex --cd)")
    ap.add_argument("-o", "--output", required=True, help="write the codex review here")
    args = ap.parse_args(argv)

    try:
        mod = _load_openai_review(args.repo_root)
    except Exception as exc:  # pragma: no cover - env-specific
        print(f"codex: cannot load openai_review.py: {exc}", file=sys.stderr)
        return 2
    if not _codex_present(mod):
        print("codex: not installed or not logged in (run `codex login`)", file=sys.stderr)
        return 2

    # codex runs with `--cd <repo-root>` read access to the whole repo. Print
    # openai_review's own sensitive-file notice before invoking, matching the
    # `/ai-review-local` codex path (the direct call_codex entry bypasses main()
    # where that path prints it). Non-blocking, like the CLI: a notice, not a gate.
    try:
        mod._print_sensitive_notice(args.repo_root, mod._scan_sensitive_files(args.repo_root))
    except Exception as exc:  # pragma: no cover - never let the notice block review
        print(f"codex: sensitive-file scan skipped ({type(exc).__name__}: {exc})", file=sys.stderr)

    with open(args.prompt_file, encoding="utf-8") as fh:
        prompt = fh.read()
    try:
        review, _ = mod.call_codex(
            prompt=prompt,
            model=CODEX_MODEL,
            repo_root=args.repo_root,
            effort=CODEX_EFFORT,
            timeout_s=CODEX_TIMEOUT_S,
        )
    except Exception as exc:
        print(f"codex: review failed or timed out ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 3
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
