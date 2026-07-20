"""Import ``.claude/scripts/openai_review.py`` as a module, without running it.

The script is not part of an installable package; it has a ``__main__`` guard
(so import is side-effect-free) and stdlib-only imports. We load it via
``importlib.util.spec_from_file_location`` — the exact pattern the existing
``tests/test_openai_review.py`` ``review_mod`` fixture uses.

We reuse from it (do NOT reimplement): ``call_codex`` + ``_build_codex_cmd`` (the
faithful CI codex invocation; their source is also hashed into experiment identity
via ``inspect.getsource``) and ``estimate_tokens`` (the prompt-size estimate
recorded in each run's usage). The carved-back, bundle-first harness does NOT
score, so it does not use ``parse_review_findings`` / ``estimate_cost`` /
``PRICING``; and it deliberately does NOT use ``compile_prompt`` (it inlines
REGISTRY — the API-backend path; CI does not). This module lives in ``adapters/``
precisely because it couples to diff-diff internals; the generic eval_core engine never
imports it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import Optional

_CACHE: Optional[ModuleType] = None


def find_repo_root(start: Optional[str] = None) -> str:
    """Walk up from ``start`` (default: this file) to the git repo root."""
    here = os.path.abspath(start or __file__)
    cur = here if os.path.isdir(here) else os.path.dirname(here)
    while True:
        if os.path.isdir(os.path.join(cur, ".git")) or os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # Fallback: three levels up (tools/reviewer-eval/adapters/..).
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cur = parent


def script_path(repo_root: Optional[str] = None) -> str:
    root = repo_root or find_repo_root()
    return os.path.join(root, ".claude", "scripts", "openai_review.py")


def load_openai_review(repo_root: Optional[str] = None) -> ModuleType:
    """Return the imported ``openai_review`` module (cached)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = script_path(repo_root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"openai_review.py not found at {path}")
    spec = importlib.util.spec_from_file_location("openai_review", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["openai_review"] = mod
    spec.loader.exec_module(mod)  # __main__-guarded ⇒ no side effects
    _CACHE = mod
    return mod


__all__ = ["load_openai_review", "find_repo_root", "script_path"]
