"""Extract a Jupyter notebook's narrative (markdown + code + executed outputs)
as a single Markdown document.

Used by `.github/workflows/ai_pr_review.yml` to give the CI AI reviewer
visibility into tutorial notebook prose. The unified diff sent to the
reviewer excludes `.ipynb` files (notebook JSON is huge and noisy); this
extractor substitutes a much smaller markdown-only view that the reviewer
can read.

Limitations (intentional, documented as policy):
- `text/html` outputs without a `text/plain` co-emit are dropped silently.
  Pandas DataFrames always co-emit both per Jupyter convention, so today's
  coverage is complete; tutorials using `IPython.display.HTML(...)` without
  a plain fallback would be silently truncated.
- `image/png` / `image/jpeg` `display_data` is dropped. Base64 PNG noise has
  no review value (~198KB combined across the project's 22 tutorials).
- `raw` cells (used for nbsphinx directives, latex preamble) are dropped.
  None of the project's current tutorials use raw cells.

The extractor uses stdlib `json` rather than `nbformat` so the CI workflow
needs no `pip install` step. Tradeoff: nbformat normalizes `cell.source`
and stream `text` to strings; with raw json those fields are lists of
strings ~88% of the time, so we coerce via `_to_str()` at every read site.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _to_str(value: Any) -> str:
    """Coerce nbformat raw JSON `source`/`text` (list-or-string) to string."""
    if isinstance(value, list):
        return "".join(value)
    return value or ""


def _truncate(text: str, max_chars: int | None) -> str:
    """Cap `text` at `max_chars` characters with a truncation marker."""
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated after {max_chars} chars]"


def _format_output(out: dict, max_chars: int | None = None) -> str:
    """Render a single notebook cell output as a fenced code block.

    Streams (stdout/stderr) come back as plain text. `execute_result` and
    `display_data` carry `text/plain` reprs. `image/*` and `text/html`-only
    outputs are dropped (see module docstring). Long outputs are capped at
    `max_chars` when provided.
    """
    ot = out.get("output_type", "")
    if ot == "stream":
        text = _truncate(_to_str(out.get("text")).rstrip(), max_chars)
        return f"```\n{text}\n```" if text else ""
    if ot in ("execute_result", "display_data"):
        data = out.get("data", {})
        text = _truncate(_to_str(data.get("text/plain")).rstrip(), max_chars)
        return f"```\n{text}\n```" if text else ""
    if ot == "error":
        ename = out.get("ename", "")
        evalue = out.get("evalue", "")
        return f"```\n{ename}: {evalue}\n```"
    return ""


def extract(
    notebook_path: Path,
    max_output_chars: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    """Render the notebook at `notebook_path` as a Markdown string.

    `max_output_chars`, when set, caps each individual `text/plain` and
    `stream` output to that length with a truncation marker. `max_total_chars`,
    when set, caps the entire rendered notebook with a truncation marker
    appended at the end. `None` (the default) preserves the relevant scope
    verbatim.
    """
    with open(notebook_path) as f:
        nb = json.load(f)

    parts: list[str] = []
    for cell in nb.get("cells", []):
        ct = cell.get("cell_type", "")
        if ct == "markdown":
            src = _to_str(cell.get("source")).rstrip()
            if src:
                parts.append(src)
                parts.append("")
        elif ct == "code":
            src = _to_str(cell.get("source")).rstrip()
            if src:
                parts.append("```python")
                parts.append(src)
                parts.append("```")
                parts.append("")
            for out in cell.get("outputs", []):
                rendered = _format_output(out, max_chars=max_output_chars)
                if rendered:
                    parts.append("**Output:**")
                    parts.append("")
                    parts.append(rendered)
                    parts.append("")

    rendered = "\n".join(parts)
    if max_total_chars is not None and len(rendered) > max_total_chars:
        rendered = (
            rendered[:max_total_chars]
            + f"\n\n... [notebook extract truncated after {max_total_chars} chars]"
        )
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a Jupyter notebook's narrative (markdown + code + outputs) "
            "as Markdown. text/html-only, image/*, and raw cells are dropped — "
            "see module docstring for details."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the .ipynb file to extract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the extracted Markdown. Defaults to stdout.",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=None,
        help=(
            "Cap each text/plain or stream output at this many characters with a "
            "truncation marker. Default: no cap."
        ),
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=None,
        help=(
            "Cap the entire rendered notebook at this many characters with a "
            "truncation marker. Default: no cap."
        ),
    )
    args = parser.parse_args()

    rendered = extract(
        args.input,
        max_output_chars=args.max_output_chars,
        max_total_chars=args.max_total_chars,
    )

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
