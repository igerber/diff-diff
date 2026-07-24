"""Strict single-pass ``__TOKEN__`` renderer for the plan-review skill.

Ships WITH the skill so production renders the campaign-graded prompt templates
byte-identically to what Campaign 1 graded — decoupled from the eval harness
under ``tools/``. Mirrors
``tools/plan-review-eval/plan_adapters/criteria_source.render``;
``tests/test_plan_review_skill.py`` asserts byte-equivalence on the shipped
templates. Do not free-text-substitute the templates anywhere else — that
would be a different, unmeasured engine.
"""

import re

_TOKEN = re.compile(r"__([A-Z][A-Z_]*)__")


class RenderError(ValueError):
    """A ``__TOKEN__`` present in the template had no provided value."""


def render(template: str, **tokens: str) -> str:
    """Strict single-pass ``__NAME__`` token substitution (brace-safe).

    Every ``__TOKEN__`` present in the TEMPLATE must have a provided value — a
    template token render was not given (the dual-arm merge-prompt bug class)
    raises rather than shipping a literal ``__CRITERIA__`` to a reviewer. A
    surplus kwarg absent from the template is ignored (matches the harness
    ``missing = wanted - set(values)`` semantics). Single-pass ``re.sub`` means
    substituted VALUES are never re-scanned: a plan whose text discusses
    ``__PLAN__`` cannot trip the check or be re-substituted.
    """
    values = {name.upper(): value for name, value in tokens.items()}
    wanted = set(_TOKEN.findall(template))
    missing = sorted(wanted - set(values))
    if missing:
        raise RenderError(
            f"template token(s) {missing} were not provided to render() — a "
            f"literal placeholder must never reach a reviewer."
        )
    return _TOKEN.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def _main(argv=None) -> int:
    """CLI so SKILL.md renders prompts in tested Python, never free-text.

    `render.py <template> --token NAME=<file> [...] -o <out>` — each token's
    VALUE is read from its file (criteria, the snapshotted plan, raw reviews).
    """
    import argparse

    ap = argparse.ArgumentParser(description="Strict __TOKEN__ prompt renderer.")
    ap.add_argument("template", help="path to the .md template")
    ap.add_argument(
        "--token",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="token value read from FILE (repeatable)",
    )
    ap.add_argument("-o", "--output", required=True, help="write the rendered prompt here")
    args = ap.parse_args(argv)

    tokens = {}
    for spec in args.token:
        if "=" not in spec:
            ap.error(f"--token {spec!r} must be NAME=FILE")
        name, _, path = spec.partition("=")
        with open(path, encoding="utf-8") as fh:
            tokens[name] = fh.read()
    with open(args.template, encoding="utf-8") as fh:
        template = fh.read()
    try:
        out = render(template, **tokens)
    except RenderError as exc:
        print(f"render error: {exc}", file=__import__("sys").stderr)
        return 2
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
