"""Run the codex (reviewer 2) half of the plan-review dual engine.

Called by SKILL.md with an already-rendered reviewer prompt. Detects codex,
invokes `openai_review.call_codex` with the campaign-validated pins
(gpt-5.6-sol @ xhigh, read-only sandbox, 600s ceiling), and writes the codex
review. Exit codes let SKILL.md fall through to the LOUD single-Claude fallback
without wedging the gate:

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

# Campaign-validated codex pins (Campaign 1: arm C). Do not change without
# re-validation — tests/test_plan_review_skill.py asserts these.
CODEX_MODEL = "gpt-5.6-sol"
CODEX_EFFORT = "xhigh"
CODEX_TIMEOUT_S = 600.0


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
