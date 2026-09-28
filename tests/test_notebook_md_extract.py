"""Smoke tests for `tools/notebook_md_extract.py`.

The CI workflow at `.github/workflows/ai_pr_review.yml` invokes this script
to render tutorial notebooks as Markdown for the AI PR reviewer. The test
uses an inline-fixture pattern so it has no I/O dependency on
`docs/tutorials/` and runs cleanly in `rust-test.yml`'s isolated-install
matrix (which copies only `tests/` to `/tmp/tests`).

The skip-guard on `tools/notebook_md_extract.py` existence covers that
isolated matrix where `tools/` is not present.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "notebook_md_extract.py"

if not SCRIPT_PATH.exists():
    pytest.skip(
        "tools/notebook_md_extract.py not present in working tree",
        allow_module_level=True,
    )


def _load_extractor():
    spec = importlib.util.spec_from_file_location("notebook_md_extract", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_fixture_notebook() -> dict:
    """Build a minimal nbformat-v4 notebook covering the list+string source
    cases AND the stream-text list coercion case (the empirical 88%/100%
    list-form rates documented in the extractor's module docstring)."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Header from list-of-strings\n", "\n", "Body line.\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Header from plain string\n\nMore body.",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": ["print('hello')\n", "x = 42\n"],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["hello\n"],
                    },
                    {
                        "output_type": "execute_result",
                        "execution_count": 1,
                        "data": {"text/plain": ["42"]},
                        "metadata": {},
                    },
                ],
            },
        ],
    }


def _minimal_nb_with_cells(cells: list[dict]) -> dict:
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}


def test_extractor_handles_list_and_string_sources(tmp_path):
    nb_path = tmp_path / "fixture.ipynb"
    nb_path.write_text(json.dumps(_build_fixture_notebook()))

    extractor = _load_extractor()
    rendered = extractor.extract(nb_path)

    assert "# Header from list-of-strings" in rendered
    assert "Body line." in rendered
    assert "## Header from plain string" in rendered
    assert "More body." in rendered
    assert "```python" in rendered
    assert "print('hello')" in rendered
    assert "**Output:**" in rendered
    assert "hello" in rendered
    assert "42" in rendered


def test_to_str_helper_coerces_list_and_string():
    extractor = _load_extractor()
    assert extractor._to_str(["a", "b", "c"]) == "abc"
    assert extractor._to_str("plain") == "plain"
    assert extractor._to_str(None) == ""
    assert extractor._to_str([]) == ""


def test_html_only_display_data_is_omitted(tmp_path):
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "render_html()",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"text/html": "<b>bold</b>"},
                        "metadata": {},
                    }
                ],
            }
        ]
    )
    nb_path = tmp_path / "html_only.ipynb"
    nb_path.write_text(json.dumps(nb))

    rendered = _load_extractor().extract(nb_path)
    assert "<b>bold</b>" not in rendered
    assert "**Output:**" not in rendered


def test_html_plus_text_plain_display_data_renders_text_plain(tmp_path):
    """Real-world `display(df)` from pandas emits BOTH `text/html` (the rich
    table) and `text/plain` (the repr-style fallback). The extractor's
    shared `execute_result | display_data` branch reads `text/plain`
    specifically, so the plain-text repr must render even when html is
    co-emitted. Locks the positive-render gap that the html-only and
    image-only omission tests don't cover (a regression in the shared
    branch could silently drop display_data plain-text rendering without
    failing the omission tests)."""
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "display(df)",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "text/html": "<table><tr><td>1</td></tr></table>",
                            "text/plain": "   col1\n0     1",
                        },
                        "metadata": {},
                    }
                ],
            }
        ]
    )
    nb_path = tmp_path / "html_plus_plain.ipynb"
    nb_path.write_text(json.dumps(nb))

    rendered = _load_extractor().extract(nb_path)
    # Plain-text repr renders.
    assert "**Output:**" in rendered
    assert "col1" in rendered
    assert "0     1" in rendered
    # HTML form is still dropped (we never render rich text/html).
    assert "<table>" not in rendered
    assert "<td>" not in rendered


def test_image_plus_text_plain_display_data_renders_text_plain(tmp_path):
    """Real-world matplotlib `display()` (or seaborn-style) emits BOTH
    `image/png` (the chart) and `text/plain` (e.g., a `<Figure ...>` repr).
    The image is intentionally dropped (base64 noise has no review value),
    but the plain-text repr must still render via the shared branch.
    Locks the positive-render path for the image-with-fallback case."""
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "fig, ax = plt.subplots(); ax.plot([1,2,3]); display(fig)",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "image/png": "iVBORw0KGgoAAAANSUhEUg==",
                            "text/plain": "<Figure size 640x480 with 1 Axes>",
                        },
                        "metadata": {},
                    }
                ],
            }
        ]
    )
    nb_path = tmp_path / "image_plus_plain.ipynb"
    nb_path.write_text(json.dumps(nb))

    rendered = _load_extractor().extract(nb_path)
    # Plain-text repr renders.
    assert "**Output:**" in rendered
    assert "<Figure size 640x480 with 1 Axes>" in rendered
    # Image base64 is still dropped.
    assert "iVBORw0KGgo" not in rendered


def test_image_png_display_data_is_omitted(tmp_path):
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "plt.show()",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"image/png": "iVBORw0KGgoAAAANSUhEUg=="},
                        "metadata": {},
                    }
                ],
            }
        ]
    )
    nb_path = tmp_path / "image_only.ipynb"
    nb_path.write_text(json.dumps(nb))

    rendered = _load_extractor().extract(nb_path)
    assert "iVBORw0KGgo" not in rendered
    assert "**Output:**" not in rendered


def test_error_output_renders_ename_and_evalue(tmp_path):
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "1/0",
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": "ZeroDivisionError",
                        "evalue": "division by zero",
                        "traceback": ["..."],
                    }
                ],
            }
        ]
    )
    nb_path = tmp_path / "err.ipynb"
    nb_path.write_text(json.dumps(nb))

    rendered = _load_extractor().extract(nb_path)
    assert "**Output:**" in rendered
    assert "ZeroDivisionError: division by zero" in rendered


def test_raw_cells_are_omitted(tmp_path):
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "raw",
                "metadata": {},
                "source": ".. raw:: latex\n\n   \\newpage\n",
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# After raw\n",
            },
        ]
    )
    nb_path = tmp_path / "raw.ipynb"
    nb_path.write_text(json.dumps(nb))

    rendered = _load_extractor().extract(nb_path)
    assert ".. raw:: latex" not in rendered
    assert "newpage" not in rendered
    assert "# After raw" in rendered


def test_max_total_chars_truncates_whole_notebook(tmp_path):
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "y" * 5000,
            }
        ]
    )
    nb_path = tmp_path / "long_total.ipynb"
    nb_path.write_text(json.dumps(nb))

    rendered = _load_extractor().extract(nb_path, max_total_chars=500)
    assert "[notebook extract truncated after 500 chars]" in rendered
    assert len(rendered) < 700


def test_max_output_chars_truncates_long_streams(tmp_path):
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "print('x' * 1000)",
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": "x" * 1000,
                    }
                ],
            }
        ]
    )
    nb_path = tmp_path / "long.ipynb"
    nb_path.write_text(json.dumps(nb))

    rendered = _load_extractor().extract(nb_path, max_output_chars=100)
    assert "[truncated after 100 chars]" in rendered
    assert rendered.count("x") <= 200
    rendered_uncapped = _load_extractor().extract(nb_path)
    assert "[truncated" not in rendered_uncapped
    assert rendered_uncapped.count("x") >= 1000


def test_main_cli_writes_to_output_file(tmp_path):
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# CLI smoke",
            }
        ]
    )
    nb_path = tmp_path / "cli.ipynb"
    nb_path.write_text(json.dumps(nb))
    out_path = tmp_path / "cli.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(nb_path),
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert out_path.exists()
    assert "# CLI smoke" in out_path.read_text()


def test_main_cli_writes_to_stdout_when_no_output(tmp_path):
    nb = _minimal_nb_with_cells(
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# stdout smoke",
            }
        ]
    )
    nb_path = tmp_path / "stdout.ipynb"
    nb_path.write_text(json.dumps(nb))

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--input", str(nb_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "# stdout smoke" in result.stdout
