#!/usr/bin/env python3
"""Argv-safe pre-merge scanning over git-controlled filenames.

Why this exists
---------------
`/pre-merge-check` (and the pattern checks `/submit-pr` and `/push-pr-update` share)
run methodology greps, test resolution, and pytest/nbmake over the set of *changed
files*. Git permits ``$()``, backticks, quotes, spaces, and newlines in a path, so any
prose that pastes a changed filename into a shell command — ``grep pattern <files>``,
``pytest <files>`` — executes a payload like ``diff_diff/$(touch x).py``. A prose
"screen" is a blocklist and leaks (untracked paths, other commands, argument vs
assignment forms).

This helper closes the class structurally:

- **Discovery** uses ``git … -z`` through ``subprocess.run([...])`` (an argv array, no
  shell) and splits on NUL, so a filename is only ever *data*.
- **Pattern checks** run as pure-Python regex / AST over file *content*. Opening a file
  by path never invokes a shell, so a hostile filename cannot execute.
- **Test resolution** matches names in Python; nothing is globbed through a shell.
- Every changed path is **screened** for shell metacharacters; unsafe paths are
  reported and excluded from the emitted run-lists, so the caller's ``pytest``/``nbmake``
  only ever receives validated-safe paths.
- Git failures **fail closed**: a nonzero git exit stops the scan with a nonzero status
  rather than being reported as "0 findings".

Modes
-----
Default scans the working tree (tracked-modified + staged + untracked). ``--range
A..B`` scans an already-committed range instead — used by ``/push-pr-update`` when the
tree is clean but commits are ahead.

The caller runs the helper, reads its report, and runs pytest/nbmake over the
NUL-delimited safe lists it writes, portably: ``xargs -0 pytest < run-tests.z``.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess

_SHELL_META = re.compile(r"""[`$;|&<>(){}\[\]"'*?\s\\]""")

_CHECK_A = re.compile(r"t_stat\s*=\s*[^#]*/\s*se")
_CHECK_B = re.compile(r"if.*(?:se|SE).*>.*0.*else\s+(?:0\.0|0)")
_CHECK_D_GUARD = re.compile(r"if.*se.*>")
_SELF_ASSIGN = re.compile(r"self\.(\w+)\s*=\s*\w")


class GitError(RuntimeError):
    """A git command failed — the scan must fail closed, not report empty."""


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — the unit-test surface
# ---------------------------------------------------------------------------


def is_safe_path(path: str) -> bool:
    """True iff ``path`` has no character that could execute or reparse in a shell."""
    return bool(path) and _SHELL_META.search(path) is None


def categorize(paths: "list[str]") -> "dict[str, list[str]]":
    """Bucket changed paths. Methodology = ``diff_diff/**/*.py`` minus ``__init__``."""
    out: "dict[str, list[str]]" = {"methodology": [], "tests": [], "notebooks": [], "docs": []}
    for p in paths:
        base = os.path.basename(p)
        if p.startswith("diff_diff/") and p.endswith(".py") and base != "__init__.py":
            out["methodology"].append(p)
        elif p.startswith("tests/") and p.endswith(".py"):
            out["tests"].append(p)
        elif p.startswith("docs/tutorials/") and p.endswith(".ipynb"):
            out["notebooks"].append(p)
        elif p.endswith((".md", ".rst")) or p.startswith("docs/"):
            out["docs"].append(p)
    return out


def check_content_patterns(lines: "list[str]") -> "list[tuple[int, str]]":
    """Run content checks A/B/D over a file's lines. Returns (line_no, message)."""
    findings = []
    for i, line in enumerate(lines, 1):
        if _CHECK_A.search(line) and "safe_inference" not in line:
            findings.append((i, "Check A: use safe_inference() instead of inline t_stat"))
        if _CHECK_B.search(line):
            findings.append((i, "Check B: SE=0 should produce NaN, not 0.0"))
        if "compute_confidence_interval" in line and not (
            "safe_inference" in line or "isfinite" in line or _CHECK_D_GUARD.search(line)
        ):
            findings.append((i, "Check D: guard compute_confidence_interval for non-finite SE"))
    return findings


def _init_self_assigns(cls: ast.ClassDef) -> "set[str]":
    names: "set[str]" = set()
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if (
                            isinstance(tgt, ast.Attribute)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == "self"
                        ):
                            names.add(tgt.attr)
    return names


def _get_params_refs(cls: ast.ClassDef) -> "set[str]":
    refs: "set[str]" = set()
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_params":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute):
                    refs.add(sub.attr)
                elif isinstance(sub, ast.Name):
                    refs.add(sub.id)
                elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    refs.add(sub.value)
    return refs


def new_params_missing_from_get_params(
    added_param_names: "list[str]", file_text: str
) -> "list[str]":
    """Check C (AST-based): of the ``added_param_names`` (new ``self.X`` from the diff),
    return those that are assigned in some class ``__init__`` but never referenced in
    that same class's ``get_params()``. Restricting to ``__init__`` and to the class's
    own ``get_params`` avoids the old substring heuristic's false negatives."""
    try:
        tree = ast.parse(file_text)
    except SyntaxError:
        return []
    added = set(added_param_names)
    missing: "set[str]" = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        assigned = _init_self_assigns(cls) & added
        if not assigned:
            continue
        refs = _get_params_refs(cls)
        missing |= assigned - refs
    return sorted(missing)


def param_names_from_added(added_lines: "list[str]") -> "list[str]":
    return sorted({m.group(1) for ln in added_lines for m in _SELF_ASSIGN.finditer(ln)})


def stem_of(path: str) -> str:
    """Test-discovery stem: the module name (or package dir), leading underscore dropped."""
    if path.startswith("diff_diff/"):
        rest = path[len("diff_diff/") :]
        first = rest.split("/", 1)[0]
        stem = first[:-3] if first.endswith(".py") else first
    else:
        stem = os.path.basename(path)
        stem = stem[:-3] if stem.endswith(".py") else stem
    return stem[1:] if stem.startswith("_") else stem


def resolve_tests(stem: str, test_files: "list[str]", group: "str | None" = None) -> "list[str]":
    """Every test file whose name contains the (underscore-stripped) stem or group."""
    keys = [k for k in (stem, group) if k]
    return sorted({t for t in test_files for k in keys if k in os.path.basename(t)})


def load_groups(yaml_text: str) -> "dict[str, str]":
    """Parse the ``groups:`` block of doc-deps.yaml into a member-path → group-name map.
    Regex-parsed (PyYAML is not a project dependency)."""
    member2group: "dict[str, str]" = {}
    m = re.search(r"^groups:\n((?:  \S.*\n|    - .*\n|\n)*)", yaml_text, re.M)
    if not m:
        return member2group
    cur = None
    for line in m.group(1).split("\n"):
        gm = re.match(r"^  (\w[\w_]*):\s*$", line)
        im = re.match(r"^\s+- (diff_diff/\S+)", line)
        if gm:
            cur = gm.group(1)
        elif im and cur:
            member2group[im.group(1)] = cur
    return member2group


# ---------------------------------------------------------------------------
# Thin git wiring (argv, NUL-delimited) and I/O — fail closed on git errors
# ---------------------------------------------------------------------------


def _git_lines_z(*args: str) -> "list[str]":
    r = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed (rc={r.returncode}): {r.stderr.strip()}")
    return [p for p in r.stdout.split("\0") if p]


def parse_name_status_z(tokens: "list[str]") -> "dict[str, str]":
    """Parse `git diff --name-status -z` tokens → {path: 'A'|'M'|'D'}. Renames/copies
    (``R``/``C``) carry two paths; the destination is treated as modified."""
    status: "dict[str, str]" = {}
    i = 0
    while i < len(tokens):
        code = tokens[i]
        i += 1
        if code and code[0] in ("R", "C"):
            # old, new
            if i + 1 < len(tokens):
                status[tokens[i + 1]] = "M"
            i += 2
        else:
            if i < len(tokens):
                status[tokens[i]] = {"A": "A", "D": "D"}.get(code[:1], "M")
            i += 1
    return status


def _name_status_z(*args: str) -> "dict[str, str]":
    return parse_name_status_z(_git_lines_z(*args))


def _changed_with_status(range_spec: "str | None") -> "dict[str, str]":
    """{path: status} for the working tree (default) or a committed range. Status is
    'A' (added/untracked), 'M' (modified), or 'D' (deleted)."""
    status: "dict[str, str]" = {}
    if range_spec:
        status.update(_name_status_z("diff", "--name-status", "-z", range_spec))
    else:
        status.update(_name_status_z("diff", "--name-status", "-z", "HEAD"))
        status.update(_name_status_z("diff", "--cached", "--name-status", "-z"))
        for p in _git_lines_z("ls-files", "--others", "--exclude-standard", "-z"):
            status.setdefault(p, "A")
    return status


def _added_lines_for(path: str, range_spec: "str | None") -> "list[str]":
    diff_arg = range_spec if range_spec else "HEAD"
    r = subprocess.run(
        ["git", "diff", diff_arg, "--", path], capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        raise GitError(f"git diff for {path!r} failed (rc={r.returncode})")
    return [
        ln[1:] for ln in r.stdout.splitlines() if ln.startswith("+") and not ln.startswith("+++")
    ]


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="Argv-safe pre-merge scan over changed files.")
    p.add_argument("--scratch", required=True, help="Directory for the safe run-lists")
    p.add_argument("--range", dest="range_spec", default=None, help="Scan a committed range A..B")
    p.add_argument("--groups-file", default="docs/doc-deps.yaml", help="doc-deps.yaml for groups")
    p.add_argument("--test-files-list", default=None, help="Optional NUL file of test paths")
    args = p.parse_args(argv)
    os.makedirs(args.scratch, exist_ok=True)

    # Truncate the run-lists up front. The scratch dir is deterministic and reused, and
    # an early failure (exit 3/4) returns before the lists are (re)written — so clear
    # them now, or a prior run's stale lists could drive pytest/nbmake over unrelated
    # files. They are repopulated only on the success path below.
    run_tests_path = os.path.join(args.scratch, "run-tests.z")
    run_notebooks_path = os.path.join(args.scratch, "run-notebooks.z")
    open(run_tests_path, "w").close()
    open(run_notebooks_path, "w").close()

    try:
        status = _changed_with_status(args.range_spec)
    except GitError as exc:
        print(f"SCAN FAILED (git error): {exc}")
        return 4  # fail closed — do NOT report "0 findings" on a broken scan

    unsafe = [p for p in status if not is_safe_path(p)]
    safe = {p: s for p, s in status.items() if is_safe_path(p)}
    cats = categorize(list(safe))

    print("== Pre-merge scan ==")
    if unsafe:
        print("UNSAFE PATHS (shell metacharacters) — excluded from checks, review by hand:")
        for u in unsafe:
            print(f"  !! {u!r}")

    findings = []
    try:
        for mpath in cats["methodology"]:
            if safe[mpath] == "D":
                continue  # deleted — nothing to read or run
            text = _read_text(mpath)
            for line_no, msg in check_content_patterns(text.splitlines()):
                findings.append(f"{mpath}:{line_no}: {msg}")
            if safe[mpath] == "A":
                # Untracked/added: `git diff HEAD` shows no added lines, so treat the
                # whole file as new when extracting self.X params for Check C.
                added_params = param_names_from_added(text.splitlines())
            else:
                added_params = param_names_from_added(_added_lines_for(mpath, args.range_spec))
            for miss in new_params_missing_from_get_params(added_params, text):
                findings.append(f"{mpath}: Check C: self.{miss} not found in get_params()")
    except (GitError, OSError) as exc:
        print(f"SCAN FAILED during pattern checks: {exc}")
        return 4

    print(f"\nPattern checks: {len(findings)} finding(s)")
    for f in findings:
        print(f"  {f}")

    # Test resolution. Seed with changed *collectable* tests (test_*.py only — a
    # conftest.py/helper change is not a pytest target), then add module-resolved suites
    # (name stem + doc-deps group). Deleted files are excluded from run-lists.
    try:
        test_files = (
            [p for p in _read_text(args.test_files_list).split("\0") if p]
            if args.test_files_list and os.path.isfile(args.test_files_list)
            else _git_lines_z("ls-files", "-z", "tests/test_*.py")
        )
    except GitError as exc:
        print(f"SCAN FAILED listing tests: {exc}")
        return 4
    groups = load_groups(_read_text(args.groups_file)) if os.path.isfile(args.groups_file) else {}

    def collectable(t: str) -> bool:
        return os.path.basename(t).startswith("test_") and safe.get(t) != "D"

    to_run: "set[str]" = {t for t in cats["tests"] if collectable(t)}
    non_test_changes = [t for t in cats["tests"] if not os.path.basename(t).startswith("test_")]
    for mpath in cats["methodology"]:
        # A deleted module still resolves surviving dependent tests to run.
        hits = [
            t
            for t in resolve_tests(stem_of(mpath), test_files, groups.get(mpath))
            if safe.get(t) != "D"
        ]
        if not hits and safe[mpath] != "D":
            print(f"\n  no test file resolved for {mpath} — confirm coverage manually")
        to_run.update(hits)
    if non_test_changes:
        print(f"\n  changed support files (not pytest targets): {non_test_changes} —")
        print("  targeted resolution is insufficient; consider the full suite.")
    ordered = sorted(to_run)

    notebooks = [n for n in cats["notebooks"] if safe.get(n) != "D"]
    with open(run_tests_path, "w", encoding="utf-8") as fh:
        fh.write("\0".join(ordered))
    with open(run_notebooks_path, "w", encoding="utf-8") as fh:
        fh.write("\0".join(notebooks))

    print(f"\nResolved {len(ordered)} test file(s) and {len(notebooks)} notebook(s).")
    print("Run portably:  xargs -0 pytest <", os.path.join(args.scratch, "run-tests.z"))
    return 3 if unsafe else 0


if __name__ == "__main__":
    raise SystemExit(main())
