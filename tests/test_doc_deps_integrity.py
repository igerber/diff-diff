"""
Integrity checks for the documentation dependency map (``docs/doc-deps.yaml``).

The map is consumed by the ``/docs-impact`` and ``/pre-merge-check`` skills and is referenced
in CLAUDE.md to decide which docs need updating when a source file changes. Nothing else guards
that the map stays honest, so this module enforces two properties:

1. **No stale paths** — every file the map references (doc ``path:`` values, ``sources:`` keys,
   and ``groups:`` members) exists on disk.
2. **No unmapped public source** — every public ``diff_diff/**/*.py`` module appears in the map
   (as a ``sources:`` key or a ``groups:`` member). Private modules (leading underscore, which
   includes ``__init__.py``) are exempt by convention — a private helper feeds a public module
   whose own docs cover it. Intentional public exceptions go in ``_ALLOWLIST``.
3. **Resolvable groups** — every ``groups:`` block is non-empty and its primary (first) member
   has a ``sources:`` entry. Group members resolve to the primary module for doc lookup, so a
   group whose primary lacks a ``sources:`` entry dead-ends — yielding no docs for any member
   even though each member is nominally "mapped".

Parsing is a tiny purpose-built line scanner rather than a full YAML load: the project ships
only numpy/pandas/scipy (no PyYAML), and the check needs just paths/keys/members from a
controlled, consistently-formatted file. If that formatting ever changes shape, these tests
fail loudly (see ``test_doc_deps_parsed_nonempty``).

NOTE: the test *logic* is pure-stdlib, but ``tests/conftest.py`` imports ``diff_diff`` + numpy
at collection, so the CI step runs with ``PYTHONPATH=. DIFF_DIFF_BACKEND=python`` like the
sibling doc-snippet job (see ``.github/workflows/docs-tests.yml``).
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_DEPS = REPO_ROOT / "docs" / "doc-deps.yaml"

# Repo-integrity test: reads docs/doc-deps.yaml and globs the diff_diff/ source tree, both
# relative to the repo root. When the suite runs against an INSTALLED package with the tests
# copied to a temp dir (the "Python Tests" CI matrix runs from /tmp/tests against site-packages),
# the repo's docs/ tree is absent — there is nothing to validate, so skip the whole module rather
# than error. It still runs from the source tree: the docs-tests CI job and local `pytest`.
if not DOC_DEPS.exists():
    pytest.skip(
        f"{DOC_DEPS} not found — doc-deps integrity runs only from the source tree, not against "
        "an installed package copied to a temp dir.",
        allow_module_level=True,
    )

# Public modules that intentionally have no doc-deps mapping. Keep empty unless a public module
# genuinely needs no documentation entry; prefer adding a real mapping in docs/doc-deps.yaml.
_ALLOWLIST: set = set()

# ---------------------------------------------------------------------------
# Line scanners — doc-deps.yaml is a controlled, consistently-formatted file:
#   sources keys are 2-space indented and end with ':'  (e.g. "  diff_diff/estimators.py:")
#   group members are 4-space indented list items       (e.g. "    - diff_diff/staggered.py")
#   doc paths are "path: <value>" list items            (e.g. "      - path: docs/...rst")
# ---------------------------------------------------------------------------
_PATH_RE = re.compile(r"^\s*-?\s*path:\s*(.+?)\s*$")
_SOURCE_KEY_RE = re.compile(r"^  (diff_diff/[^\s:]+\.py):\s*$")
_GROUP_MEMBER_RE = re.compile(r"^    - (diff_diff/\S+\.py)\s*$")
# group header: a 2-space-indented bare name ending in ':' (e.g. "  staggered:"). Cannot match a
# `sources:` key, which contains "diff_diff/.../*.py" (slash/dot are outside [\w-]).
_GROUP_HEADER_RE = re.compile(r"^  ([A-Za-z][\w-]*):\s*$")


def _parse_doc_deps():
    """Return (doc_paths, source_keys, group_members) as sets of repo-relative path strings."""
    text = DOC_DEPS.read_text()
    doc_paths: set = set()
    source_keys: set = set()
    group_members: set = set()
    for line in text.splitlines():
        m = _PATH_RE.match(line)
        if m:
            doc_paths.add(m.group(1).strip().strip('"').strip("'"))
            continue
        m = _SOURCE_KEY_RE.match(line)
        if m:
            source_keys.add(m.group(1))
            continue
        m = _GROUP_MEMBER_RE.match(line)
        if m:
            group_members.add(m.group(1))
    return doc_paths, source_keys, group_members


def _parse_groups():
    """Return {group_name: [member, ...]} (member order preserved) from the ``groups:`` block.

    Member order is load-bearing: a group member resolves to the group's FIRST entry (the
    primary module) for doc lookup (see the doc-deps.yaml header).
    """
    text = DOC_DEPS.read_text()
    groups: dict = {}
    in_groups = False
    current = None
    for line in text.splitlines():
        if line.rstrip() == "groups:":
            in_groups = True
            continue
        if in_groups and line and not line[0].isspace():
            # Only a real top-level key (e.g. "sources:") ends the groups block. Skip top-level
            # comments so a comment inserted between group definitions cannot truncate the parse
            # — truncation would silently drop later groups and reopen the false-green this guards.
            if line.lstrip().startswith("#"):
                continue
            break
        if not in_groups:
            continue
        m = _GROUP_HEADER_RE.match(line)
        if m:
            current = m.group(1)
            groups[current] = []
            continue
        m = _GROUP_MEMBER_RE.match(line)
        if m and current is not None:
            groups[current].append(m.group(1))
    return groups


_DOC_PATHS, _SOURCE_KEYS, _GROUP_MEMBERS = _parse_doc_deps()
_GROUPS = _parse_groups()
_ALL_REFERENCED = sorted(_DOC_PATHS | _SOURCE_KEYS | _GROUP_MEMBERS)


def test_doc_deps_parsed_nonempty():
    """Guard: a parser/format break must not silently empty the checks below."""
    assert _DOC_PATHS, "no doc `path:` values parsed from doc-deps.yaml — parser/format drift?"
    assert _SOURCE_KEYS, "no `sources:` keys parsed from doc-deps.yaml — parser/format drift?"
    assert _GROUP_MEMBERS, "no `groups:` members parsed from doc-deps.yaml — parser/format drift?"
    assert _GROUPS, "no `groups:` blocks parsed from doc-deps.yaml — parser/format drift?"
    # Cross-check: the structural group parse must capture exactly the same members as the flat
    # whole-file scan. A mismatch means _parse_groups() truncated (e.g. a top-level comment
    # between groups), which would let test_group_primaries_have_sources_entry skip groups.
    _flat_members = {m for members in _GROUPS.values() for m in members}
    assert _flat_members == _GROUP_MEMBERS, (
        "structural _parse_groups() members != flat scan — groups block truncated?\n"
        f"  missing from structural: {sorted(_GROUP_MEMBERS - _flat_members)}\n"
        f"  extra in structural: {sorted(_flat_members - _GROUP_MEMBERS)}"
    )


@pytest.mark.parametrize("ref_path", _ALL_REFERENCED, ids=lambda p: p)
def test_referenced_paths_exist(ref_path):
    """Every file referenced by doc-deps.yaml must exist on disk (no stale paths)."""
    assert (REPO_ROOT / ref_path).exists(), (
        f"docs/doc-deps.yaml references '{ref_path}', which does not exist on disk. "
        f"Update or remove the entry."
    )


def test_all_public_sources_mapped():
    """Every public diff_diff source module must be mapped (sources key or group member)."""
    mapped = _SOURCE_KEYS | _GROUP_MEMBERS
    actual = {p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "diff_diff").rglob("*.py")}
    offenders = sorted(
        rel
        for rel in actual - mapped
        # Private modules (leading underscore, which includes __init__.py) are exempt.
        if not Path(rel).name.startswith("_") and rel not in _ALLOWLIST
    )
    assert not offenders, (
        "Public diff_diff source modules are missing from docs/doc-deps.yaml:\n  "
        + "\n  ".join(offenders)
        + "\n\nAdd a `sources:` entry (mapping each to its docs) in docs/doc-deps.yaml, or add "
        "the module to _ALLOWLIST in this test if it intentionally needs no doc mapping."
    )


def test_group_primaries_have_sources_entry():
    """Each ``groups:`` block must be non-empty and its primary (first) member must have a
    ``sources:`` entry. Group members resolve to the primary module for doc lookup, so a group
    whose primary lacks a ``sources:`` entry dead-ends — /docs-impact would yield no docs for the
    whole group even though every member is nominally 'mapped' by ``test_all_public_sources_mapped``.
    """
    broken = []
    for name, members in _GROUPS.items():
        if not members:
            broken.append(f"{name!r}: empty group")
        elif members[0] not in _SOURCE_KEYS:
            broken.append(f"{name!r}: primary '{members[0]}' has no `sources:` entry")
    assert not broken, (
        "groups: whose primary member lacks a `sources:` entry (doc lookup dead-ends):\n  "
        + "\n  ".join(broken)
        + "\n\nGroup members resolve to the group's first entry; add a `sources:` entry for that "
        "primary module in docs/doc-deps.yaml."
    )
