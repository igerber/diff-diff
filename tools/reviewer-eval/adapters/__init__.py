"""diff-diff-specific bindings for the reviewer-eval harness.

These adapters wire the real diff-diff corpus, the production ``openai_review``
codex invocation, and the git worktree machinery into the generic ``eval_core``
engine (shared across eval harnesses, under ``tools/eval_core/``).
"""
