"""Shared helpers for tutorial-drift tests (T20, T21, ...).

The HAD tutorial drift tests pin numbers / verdict strings against the
locked DGP + seed. Without these helpers each drift test re-derived
numbers but never verified that the rendered notebook surface (markdown
prose + executed output cells) actually quotes those values. Because
``nbsphinx_execute = "never"`` in ``docs/conf.py``, CI cannot detect
drift between the pinned constants and the committed tutorial via
notebook re-execution; the constants and the notebook can diverge
silently. These helpers parse the .ipynb JSON directly so each
tutorial-drift test file can cross-check its pins against the
rendered surface it claims to protect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _read_notebook(nb_relpath: str) -> dict:
    """Load a notebook by repo-relative path (e.g. ``docs/tutorials/X.ipynb``).

    Skips the calling test via ``pytest.skip(...)`` when the notebook file
    is not present. The Rust-test CI job (and the isolated-install job)
    copies only ``tests/`` to ``/tmp/tests`` and runs from there, without
    ``docs/`` available. The repo convention is to skip cleanly when
    artifacts are absent rather than fail (see e.g.
    ``tests/test_notebook_md_extract.py`` and ``tests/test_nprobust_port.py``).
    """
    import pytest

    nb_path = Path(__file__).resolve().parents[1] / nb_relpath
    if not nb_path.exists():
        pytest.skip(
            f"Notebook {nb_relpath!r} not available in this CI environment "
            "(isolated-install job copies only tests/, not docs/); "
            "rendered-surface cross-check requires a full repo checkout."
        )
    return json.loads(nb_path.read_text())


def notebook_markdown(nb_relpath: str) -> str:
    """Return all markdown cells concatenated into one string."""
    nb = _read_notebook(nb_relpath)
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        parts.append(src)
    return "\n".join(parts)


def notebook_output_text(nb_relpath: str) -> str:
    """Return all executed-output text (``stream`` and ``execute_result``
    text/plain) from every code cell, concatenated.

    Covers the rendered numeric surface that markdown alone misses —
    e.g. printed verdict strings, formatted summary tables, p-values.
    """
    nb = _read_notebook(nb_relpath)
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            # stream-style outputs (print / stdout / stderr)
            text = out.get("text")
            if text is not None:
                parts.append("".join(text) if isinstance(text, list) else text)
            # execute_result / display_data with text/plain
            data = out.get("data") or {}
            plain = data.get("text/plain")
            if plain is not None:
                parts.append("".join(plain) if isinstance(plain, list) else plain)
    return "\n".join(parts)


def notebook_rendered_text(nb_relpath: str) -> str:
    """Return markdown + executed-output text together — the full
    rendered surface a reader sees on RTD."""
    return notebook_markdown(nb_relpath) + "\n" + notebook_output_text(nb_relpath)


def assert_quotes_in_rendered(
    nb_relpath: str,
    expected_quotes: Iterable[str],
    *,
    surface: str = "rendered",
) -> None:
    """Assert each expected substring appears in the chosen rendered surface.

    Parameters
    ----------
    nb_relpath
        Notebook path relative to repo root (e.g.
        ``"docs/tutorials/21_had_pretest_workflow.ipynb"``).
    expected_quotes
        Iterable of substrings that MUST appear in the chosen rendered
        surface. Each is checked independently; the assertion message
        lists every missing quote so a single failure surfaces all of
        them.
    surface
        Which slice of the notebook to check: ``"markdown"`` (prose
        only), ``"output"`` (executed output cells only), or
        ``"rendered"`` (both — default; matches what a reader sees
        on RTD).
    """
    if surface == "markdown":
        text = notebook_markdown(nb_relpath)
    elif surface == "output":
        text = notebook_output_text(nb_relpath)
    elif surface == "rendered":
        text = notebook_rendered_text(nb_relpath)
    else:
        raise ValueError(f"surface must be 'markdown' / 'output' / 'rendered'; got {surface!r}")
    missing = [q for q in expected_quotes if q not in text]
    assert not missing, (
        f"Tutorial {nb_relpath!r} ({surface=}) is missing load-bearing "
        f"quoted values that the pinned drift constants assume are "
        f"rendered verbatim. Either the notebook drifted from the "
        f"locked DGP output (rerun the tutorial against the pinned "
        f"seed) or the drift-test constants were updated without "
        f"updating the tutorial. Missing: {missing}"
    )
