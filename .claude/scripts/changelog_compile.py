#!/usr/bin/env python3
"""Changelog-fragment validator and release-time compiler.

PRs write per-PR entry files under ``changelog.d/`` instead of editing
``CHANGELOG.md``'s ``## [Unreleased]`` section (which stays pointer-only
between releases; the invariant is CI-enforced by
``tests/test_changelog_fragments.py``). At release time ``compile`` merges
the fragments into a new ``## [X.Y.Z] - DATE`` section and deletes them.

Subcommands
-----------
check
    Validate repo state: ``changelog.d/README.md`` exists; the directory
    holds only README.md, valid fragments, and dotfiles; every fragment
    parses under the body grammar; CHANGELOG.md's Unreleased section is
    pointer-only. Exit 0 clean, exit 1 with findings otherwise.
compile --version X.Y.Z --date YYYY-MM-DD [--allow-dirty]
    Assemble the release section. Exit 0 = compiled; exit 4 = the target
    header already exists with no fragments left (idempotent re-run; the
    header's date is printed for RELEASE_DATE reuse); anything else is an
    error (exit 2 usage / exit 1 state).

Stdlib-only, Python 3.9-compatible (annotations quoted), no dependence on
cwd: the repo root defaults to this file's grandparent directory and can be
overridden with ``--root`` (used by the tmp-dir tests).
"""

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

# Single source of the category vocabulary AND the compile-time section
# order. changelog.d/README.md mirrors this list; the mirror is pinned by
# tests/test_changelog_fragments.py (README<->compiler parity test).
CATEGORIES = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Performance",
    "Documentation",
    "Testing",
    "Breaking Changes",
    "Behavioral Changes",
    "Internal",
)

POINTER_COMMENT = (
    "<!-- entries live in changelog.d/*.md; "
    "compiled at release by .claude/scripts/changelog_compile.py -->"
)

FRAGMENT_NAME_RE = re.compile(r"^(\d{8})-[a-z0-9][a-z0-9-]*\.md$")
VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2
EXIT_ALREADY_COMPILED = 4


def default_root():
    # .claude/scripts/changelog_compile.py -> repo root is parents[2].
    return Path(__file__).resolve().parents[2]


def _parse_fragment(text):
    """Return (blocks, errors). blocks = list of (category, body_lines)
    where body_lines excludes the header line and preserves bytes."""
    errors = []
    blocks = []
    current = None  # (category, [lines])
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            errors.append(f"line {lineno}: '## ' header not allowed in a fragment")
            continue
        if line.startswith("### "):
            cat = line[4:].strip()
            if cat not in CATEGORIES:
                errors.append(
                    f"line {lineno}: unknown category {cat!r} "
                    f"(allowed: {', '.join(CATEGORIES)})"
                )
            current = (cat, [])
            blocks.append(current)
            continue
        if not line.strip():
            if current is not None:
                current[1].append(line)
            continue
        if current is None:
            errors.append(f"line {lineno}: content outside any '### <Category>' block")
            continue
        if line.startswith("- ") or line[0] in (" ", "\t"):
            current[1].append(line)
        else:
            errors.append(
                f"line {lineno}: not a top-level bullet ('- ') or indented " "continuation line"
            )
    if not blocks:
        errors.append("no '### <Category>' block found")
    for cat, lines in blocks:
        if not any(ln.startswith("- ") for ln in lines):
            errors.append(f"category {cat!r}: no top-level bullet")
    return blocks, errors


def _valid_fragment_name(name):
    m = FRAGMENT_NAME_RE.match(name)
    if not m:
        return False
    raw = m.group(1)
    try:
        datetime.date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return False
    return True


def _unreleased_slice(changelog_text):
    """Return (start, end) character offsets of the Unreleased section body
    (after the header line, up to the next '## ' line), or None."""
    m = re.search(r"^## \[Unreleased\]\s*$", changelog_text, flags=re.MULTILINE)
    if not m:
        return None
    body_start = changelog_text.index("\n", m.start()) + 1
    nxt = re.compile(r"^## ", flags=re.MULTILINE).search(changelog_text, body_start)
    body_end = nxt.start() if nxt else len(changelog_text)
    return body_start, body_end


def run_check(root):
    findings = []
    frag_dir = root / "changelog.d"
    readme = frag_dir / "README.md"
    fragments = []
    if not frag_dir.is_dir():
        findings.append("changelog.d/ directory is missing")
    else:
        if not readme.is_file():
            findings.append("changelog.d/README.md (the format spec) is missing")
        for p in sorted(frag_dir.iterdir()):
            if p.name.startswith("."):
                continue  # dotfiles (.DS_Store etc.) are ignored
            if p.name == "README.md":
                continue
            if p.is_dir():
                findings.append(f"changelog.d/{p.name}: subdirectories are not allowed")
                continue
            if not _valid_fragment_name(p.name):
                findings.append(
                    f"changelog.d/{p.name}: name must match "
                    "YYYYMMDD-<kebab-slug>.md with a real calendar date"
                )
                continue
            fragments.append(p)
            _, errors = _parse_fragment(p.read_text())
            findings.extend(f"changelog.d/{p.name}: {e}" for e in errors)

    changelog = root / "CHANGELOG.md"
    if not changelog.is_file():
        findings.append("CHANGELOG.md is missing")
    else:
        sl = _unreleased_slice(changelog.read_text())
        if sl is None:
            findings.append("CHANGELOG.md: '## [Unreleased]' header is missing")
        else:
            body = changelog.read_text()[sl[0] : sl[1]]
            nonblank = [ln for ln in body.splitlines() if ln.strip()]
            if nonblank != [POINTER_COMMENT]:
                findings.append(
                    "CHANGELOG.md: the Unreleased section must contain exactly "
                    "the pointer comment (write entries as changelog.d/ "
                    "fragments instead):\n    " + POINTER_COMMENT
                )
    return findings, fragments


def _semver_tuple(s):
    return tuple(int(x) for x in s.split("."))


def _existing_headers(changelog_text):
    """[(version, date_or_None, match_start)] in file order."""
    out = []
    for m in re.finditer(
        r"^## \[(\d+\.\d+\.\d+)\](?: - (\S+))?\s*$", changelog_text, flags=re.MULTILINE
    ):
        out.append((m.group(1), m.group(2), m.start()))
    return out


def _section_nonempty(changelog_text, header_start):
    body_start = changelog_text.index("\n", header_start) + 1
    nxt = re.compile(r"^## ", flags=re.MULTILINE).search(changelog_text, body_start)
    body = changelog_text[body_start : nxt.start() if nxt else len(changelog_text)]
    has_cat = any(
        ln.startswith("### ") and ln[4:].strip() in CATEGORIES for ln in body.splitlines()
    )
    has_bullet = any(ln.startswith("- ") for ln in body.splitlines())
    return has_cat and has_bullet


def _canonical_date(s):
    """True iff s is exactly YYYY-MM-DD for a real calendar date (rejects
    3.11+/3.14 fromisoformat leniency toward compact/partial spellings)."""
    try:
        parsed = datetime.date.fromisoformat(s)
    except ValueError:
        return False
    return parsed.isoformat() == s


def run_compile(root, version, date_s, allow_dirty):
    if not VERSION_RE.match(version):
        print(
            f"error: --version {version!r} is not a canonical SemVer " "X.Y.Z (no leading zeros)",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if not _canonical_date(date_s):
        print(
            f"error: --date {date_s!r} is not a canonical YYYY-MM-DD date",
            file=sys.stderr,
        )
        return EXIT_USAGE

    findings, fragments = run_check(root)
    if findings:
        for f in findings:
            print(f"check: {f}", file=sys.stderr)
        return EXIT_FINDINGS

    changelog_path = root / "CHANGELOG.md"
    text = changelog_path.read_text()
    headers = _existing_headers(text)
    versions = [h[0] for h in headers]
    prev = max(versions, key=_semver_tuple) if versions else None

    # Existing-header detection FIRST, so the idempotent re-run path is
    # reachable before any monotonicity check.
    for hv, hdate, hstart in headers:
        if hv == version:
            if prev != version:
                print(
                    f"error: '## [{version}]' exists but is older than the "
                    f"latest release {prev} — refusing a downgrade re-run",
                    file=sys.stderr,
                )
                return EXIT_FINDINGS
            if not _section_nonempty(text, hstart):
                print(
                    f"error: '## [{version}]' exists but its section is "
                    "empty — not a completed compile; fix the header",
                    file=sys.stderr,
                )
                return EXIT_FINDINGS
            if fragments:
                print(
                    f"error: '## [{version}]' already exists but "
                    f"{len(fragments)} fragment(s) remain in changelog.d/ — "
                    "partial or conflicting state; resolve manually",
                    file=sys.stderr,
                )
                return EXIT_FINDINGS
            if hdate is None or not _canonical_date(hdate):
                print(
                    f"error: '## [{version}]' has a missing or non-canonical "
                    f"date ({hdate!r}); fix the header before reusing it",
                    file=sys.stderr,
                )
                return EXIT_FINDINGS
            below = [v for v in versions if _semver_tuple(v) < _semver_tuple(version)]
            predecessor = max(below, key=_semver_tuple) if below else None
            if predecessor is None:
                # compile always anchors its comparison link to an existing
                # release, so a sole-header state cannot be its output.
                print(
                    f"error: '## [{version}]' is the only release header — "
                    "not a completed compile (no preceding release to anchor "
                    "the comparison link); fix CHANGELOG.md",
                    file=sys.stderr,
                )
                return EXIT_FINDINGS
            if predecessor is not None:
                link_re = re.compile(
                    r"^\["
                    + re.escape(version)
                    + r"\]: \S+/compare/v"
                    + re.escape(predecessor)
                    + r"\.\.\.v"
                    + re.escape(version)
                    + r"$",
                    flags=re.MULTILINE,
                )
                if not link_re.search(text):
                    print(
                        f"error: '## [{version}]' exists but its comparison "
                        f"link '[{version}]: .../compare/v{predecessor}...v"
                        f"{version}' is missing or wrongly sourced — not a "
                        "completed compile; fix the link block",
                        file=sys.stderr,
                    )
                    return EXIT_FINDINGS
            print(f"already-compiled: version={version} date={hdate}")
            return EXIT_ALREADY_COMPILED

    if prev is None:
        print(
            "error: CHANGELOG.md has no existing '## [X.Y.Z]' release header "
            "to anchor the comparison link",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    if _semver_tuple(version) <= _semver_tuple(prev):
        print(
            f"error: --version {version} must exceed the latest release " f"{prev}",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    if not fragments:
        print(
            "error: changelog.d/ has no fragments — nothing to release "
            "(releases require fragments; for a fragment-free cycle write a "
            "minimal '### Internal' stub fragment)",
            file=sys.stderr,
        )
        return EXIT_FINDINGS

    if (root / ".git").exists():
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "changelog.d/"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if status and not allow_dirty:
            print(
                "error: changelog.d/ has uncommitted changes (they would be "
                "swept into the release and deleted):\n"
                + status
                + "\ncommit them first, or pass --allow-dirty",
                file=sys.stderr,
            )
            return EXIT_FINDINGS
    elif not allow_dirty:
        print(
            "error: no .git at the resolved root — cannot run the "
            "dirty-fragment guard; pass --allow-dirty for scratch copies",
            file=sys.stderr,
        )
        return EXIT_FINDINGS

    # Assemble: category order = CATEGORIES; within a category, fragments in
    # ascending filename order, block bodies preserved byte-for-byte with
    # trailing blank lines stripped and contributions joined contiguously.
    per_category = {}
    for p in fragments:
        blocks, _ = _parse_fragment(p.read_text())
        for cat, lines in blocks:
            body = "\n".join(lines).rstrip("\n")
            per_category.setdefault(cat, []).append(body)
    parts = []
    for cat in CATEGORIES:
        if cat in per_category:
            parts.append(f"### {cat}\n" + "\n".join(per_category[cat]))
    new_section = f"## [{version}] - {date_s}\n\n" + "\n\n".join(parts) + "\n\n"

    sl = _unreleased_slice(text)
    assert sl is not None  # run_check guaranteed the header + pointer
    insert_at = sl[1]
    text = text[:insert_at] + new_section + text[insert_at:]

    prev_link_re = re.compile(
        r"^\[" + re.escape(prev) + r"\]: (\S+?)/compare/\S+$", flags=re.MULTILINE
    )
    m = prev_link_re.search(text)
    if not m:
        print(
            f"error: comparison link '[{prev}]: .../compare/...' not found "
            "at the bottom of CHANGELOG.md",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    base = m.group(1)
    new_link = f"[{version}]: {base}/compare/v{prev}...v{version}\n"
    text = text[: m.start()] + new_link + text[m.start() :]

    changelog_path.write_text(text)
    for p in fragments:
        p.unlink()
    print(f"compiled: version={version} date={date_s} fragments={len(fragments)}")
    return EXIT_OK


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="repo root override")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    comp = sub.add_parser("compile")
    comp.add_argument("--version", required=True)
    comp.add_argument("--date", required=True)
    comp.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    root = (args.root or default_root()).resolve()
    if args.cmd == "check":
        findings, _ = run_check(root)
        if findings:
            for f in findings:
                print(f"check: {f}", file=sys.stderr)
            return EXIT_FINDINGS
        print("check: OK")
        return EXIT_OK
    return run_compile(root, args.version, args.date, args.allow_dirty)


if __name__ == "__main__":
    sys.exit(main())
