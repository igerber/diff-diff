"""Shared engine for the workflow eval harnesses (generic glue, no bindings).

Plain data (``eval_core.models``), a resumable run store (``eval_core.store``), a
run-matrix executor (``eval_core.runner``), and the side-by-side bundle builder
(``eval_core.compare``). Surface-specific bindings live in each harness's own
``adapters/`` (first consumer: ``tools/reviewer-eval/``; see its README for the
flow).
"""
