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

FRAGMENT_NAME_RE = re.compile(r"^(\d{8})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
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
        if re.match(r" {1,3}#{1,6}(?:[ \t]|$)", line):
            # CommonMark recognizes ATX headings indented up to 3 spaces, so
            # an indented '## ...' would slip past the column-zero checks
            # above as a "continuation" yet render as a real heading.
            errors.append(f"line {lineno}: indented Markdown heading not allowed in a fragment")
            continue
        if _VERSION_LINK_SHAPE.search(line):
            # A '- [1.3.0]:url' bullet would register a CommonMark link
            # definition ONCE COMPILED — and the first definition wins,
            # outranking the canonical one the compiler appends.
            errors.append(
                f"line {lineno}: version-link-like construct ('[X.Y.Z]:') "
                "not allowed in a fragment"
            )
            continue
        if not line.strip():
            if current is not None:
                current[1].append(line)
            continue
        if current is None:
            errors.append(f"line {lineno}: content outside any '### <Category>' block")
            continue
        if line.startswith("- ") and not line[2:].strip():
            errors.append(f"line {lineno}: empty top-level bullet ('- ' with no content)")
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
        if not any(ln.startswith("- ") and ln[2:].strip() for ln in lines):
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


_UNRELEASED_HEADER_RE = re.compile(r"^## \[Unreleased\]\s*$", flags=re.MULTILINE)


def _unreleased_headers(changelog_text):
    """All '## [Unreleased]' header matches, in file order."""
    return list(_UNRELEASED_HEADER_RE.finditer(changelog_text))


def _unreleased_slice(changelog_text):
    """Return (start, end) character offsets of the FIRST Unreleased section
    body (after the header line, up to the next '## ' line), or None.

    Callers must separately enforce that exactly one Unreleased header
    exists (run_check does) — a duplicate header would otherwise hide its
    body from this slice.
    """
    headers = _unreleased_headers(changelog_text)
    if not headers:
        return None
    m = headers[0]
    nl = changelog_text.find("\n", m.start())
    if nl == -1:
        # Header is the last line with no trailing newline: empty body.
        return len(changelog_text), len(changelog_text)
    body_start = nl + 1
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
            # lstat-level guard BEFORE any read: a symlink's content is
            # mutable out-of-band, and a FIFO/device would hang or crash the
            # read; neither can be certified.
            if p.is_symlink() or not p.is_file():
                findings.append(f"changelog.d/{p.name}: not a regular file (symlink or special)")
                continue
            try:
                text = p.read_text()
            except (OSError, UnicodeDecodeError) as exc:
                findings.append(f"changelog.d/{p.name}: unreadable ({exc})")
                continue
            fragments.append(p)
            _, errors = _parse_fragment(text)
            findings.extend(f"changelog.d/{p.name}: {e}" for e in errors)
            # check↔compile parity: everything the assembled-output guards
            # would refuse at release time must already fail HERE, at PR
            # time — a fragment that passes CI but blocks the release is a
            # fail-late trap. The whole-text scans also cover what the
            # line-based grammar cannot see (a multi-line link label, an
            # indented fence in a continuation line).
            blk = _block_context_violation(text)
            if blk:
                findings.append(
                    f"changelog.d/{p.name}: line {blk[0]}: Markdown block "
                    f"context ({blk[1]!r}) not allowed in a fragment — "
                    "fenced code / raw-HTML blocks are refused in the "
                    "compiled changelog; use inline code instead"
                )
            link = _noncanonical_version_link(text)
            if link:
                findings.append(
                    f"changelog.d/{p.name}: line {link[0]}: version-link-"
                    f"like construct ({link[1]!r}) not allowed in a fragment"
                )

    changelog = root / "CHANGELOG.md"
    if not changelog.is_file():
        findings.append("CHANGELOG.md is missing")
    else:
        changelog_text = changelog.read_text()
        headers = _unreleased_headers(changelog_text)
        if not headers:
            findings.append("CHANGELOG.md: '## [Unreleased]' header is missing")
        elif len(headers) > 1:
            findings.append(
                f"CHANGELOG.md: {len(headers)} '## [Unreleased]' headers found "
                "— exactly one is allowed (a duplicate section would hide "
                "direct edits from the pointer-only guard)"
            )
        else:
            sl = _unreleased_slice(changelog_text)
            assert sl is not None
            body = changelog_text[sl[0] : sl[1]]
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


_VERSION_LINK_SHAPE = re.compile(r"\[\s*\d+\.\d+\.\d+\s*\]:")
_CANONICAL_LINK_SHAPE = re.compile(r"\[\d+\.\d+\.\d+\]:")


def _noncanonical_version_link(text):
    """(lineno, snippet) of the first version-link-LIKE construct that is
    not a canonical column-zero '[X.Y.Z]:' definition, else None.

    CommonMark registers reference definitions inside containers (list
    items, blockquotes) and whitespace-normalizes labels ('[ 1.3.0 ]'
    resolves as '1.3.0'), and the FIRST definition wins — so any
    version-label-plus-colon anywhere outside the canonical link block
    could silently outrank the definitions this compiler writes and
    scans. Rather than chase every container form, refuse them all."""
    for m in _VERSION_LINK_SHAPE.finditer(text):
        at_col0 = m.start() == 0 or text[m.start() - 1] == "\n"
        if at_col0 and _CANONICAL_LINK_SHAPE.match(text, m.start()):
            continue  # canonical: exactly what link_versions collects
        lineno = text.count("\n", 0, m.start()) + 1
        return lineno, m.group(0).replace("\n", "\\n")
    return None


# CommonMark block contexts that persist through blank lines (and through
# EOF when unterminated): fenced code (```/~~~ at 0-3 indent) and raw-HTML
# block types 1-5 (<pre/script/style/textarea>, processing instructions,
# declarations, CDATA). Any of these could swallow subsequent lines —
# including the terminal link block — so headers/definitions inside them
# would not render while still matching the line-based scans. Type-6/7
# HTML blocks end at the first blank line and cannot reach the terminal
# block. Comments (type 2) are allowed only when closed on the same line
# (the Unreleased pointer's form).
_BLOCK_CONTEXT_STARTER = re.compile(
    r"^ {0,3}(?:```|~~~|<(?:pre|script|style|textarea)\b|<\?|<!\[CDATA\[|<![A-Za-z])",
    flags=re.IGNORECASE,
)


def _block_context_violation(text):
    """(lineno, snippet) of the first persistent block-context starter, or
    an unclosed same-line HTML comment, else None. The changelog (and
    compiler output — the fragment grammar cannot produce a column-0-3
    fence) must contain none, so no scan ever needs Markdown block
    context."""
    for i, ln in enumerate(text.split("\n"), 1):
        if _BLOCK_CONTEXT_STARTER.match(ln):
            return i, ln[:60]
        if "<!--" in ln and "-->" not in ln[ln.index("<!--") :]:
            return i, ln[:60]
    return None


def _link_block_violation(text):
    """(lineno, snippet) of the first canonical-LOOKING definition that is
    not part of the terminal contiguous link block, else None.

    The inverse hazard of _noncanonical_version_link: a column-zero
    '[X.Y.Z]: url' inside a fenced code or raw-HTML block LOOKS canonical
    to the line-based scans but does not render as a definition — the
    anchor search could then insert the new link inside that block.
    Requiring every definition to live in ONE contiguous run at the end
    of the file (where no fence/HTML context can exist without breaking
    the run) removes Markdown block context from the problem entirely."""
    lines = text.split("\n")
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    start = end
    while start > 0 and _CANONICAL_LINK_SHAPE.match(lines[start - 1]):
        start -= 1
    terminal = set(range(start, end))
    # The run must be immediately preceded by a BLANK line: type-6/7 raw-
    # HTML blocks (e.g. a bare '<div>') persist only until a blank line,
    # so a blank boundary guarantees no such block spans into the run.
    if terminal and start > 0 and lines[start - 1].strip():
        return start, lines[start - 1][:60]
    for i, ln in enumerate(lines):
        if _CANONICAL_LINK_SHAPE.match(ln) and i not in terminal:
            return i + 1, ln[:60]
    return None


def _section_body(changelog_text, header_start):
    nl = changelog_text.find("\n", header_start)
    if nl == -1:
        return ""
    body_start = nl + 1
    nxt = re.compile(r"^## ", flags=re.MULTILINE).search(changelog_text, body_start)
    return changelog_text[body_start : nxt.start() if nxt else len(changelog_text)]


def _section_grammar_errors(changelog_text, header_start):
    """Errors from validating a release section's body with the fragment
    block grammar. A compiled section is exactly a concatenation of
    fragment category blocks, so compiler OUTPUT always passes; an orphan
    bullet, an empty category, or an unknown-category bullet is a state
    compile could not have produced."""
    body = _section_body(changelog_text, header_start)
    if not body.strip():
        return ["section is empty"]
    _, errors = _parse_fragment(body)
    return errors


def _canonical_date(s):
    """True iff s is exactly YYYY-MM-DD for a real calendar date (rejects
    3.11+/3.14 fromisoformat leniency toward compact/partial spellings)."""
    try:
        parsed = datetime.date.fromisoformat(s)
    except ValueError:
        return False
    return parsed.isoformat() == s


def run_compile(root, version, date_s, allow_dirty, previous_version=None):
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
    # Reject release-LIKE headers the canonical grammar does not match
    # (e.g. '## [1.3.0] - 2026-08-30 draft'): _existing_headers sees only
    # canonical shapes, so a malformed section would otherwise be invisible
    # to the duplicate/date safeguards and compile could add a SECOND
    # section for the same version.
    # The scan accepts ANY CommonMark heading spacing ('##  [x]', a tab,
    # any hash depth) so a rendering-but-noncanonical heading cannot hide
    # from the duplicate/date safeguards; only exact-canonical release and
    # Unreleased headers are exempt.
    canonical_starts = {m_start for _, _, m_start in _existing_headers(text)}
    canonical_starts |= {m.start() for m in _unreleased_headers(text)}
    for m in re.finditer(r"^#{1,6}[ \t]+\[[^\]]*\][^\n]*$", text, flags=re.MULTILINE):
        if m.start() not in canonical_starts:
            print(
                "error: malformed release header in CHANGELOG.md: "
                f"{m.group(0)!r} does not match '## [X.Y.Z] - YYYY-MM-DD' — "
                "fix the file before compiling",
                file=sys.stderr,
            )
            return EXIT_FINDINGS
    # CommonMark renders ATX headings and link-reference definitions
    # indented up to 3 spaces, but every scan in this function is anchored
    # at column zero — an indented ' ## [1.3.0]' or ' [1.3.0]: ...' would
    # be invisible to the duplicate/date safeguards yet render as a real
    # heading or definition. Refuse them outright.
    indented = re.search(
        r"^ {1,3}(?:#{1,6}(?:[ \t]|$)|\[\d+\.\d+\.\d+\]:)",
        text,
        flags=re.MULTILINE,
    )
    if indented:
        lineno = text.count("\n", 0, indented.start()) + 1
        print(
            f"error: CHANGELOG.md line {lineno}: indented heading or "
            "version-link definition (CommonMark renders these, but the "
            "compiler's column-zero scans cannot see them) — fix the file "
            "before compiling",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    # Version-link-like constructs (any container form, any label spacing)
    # must not exist outside the canonical column-zero definitions — see
    # _noncanonical_version_link.
    bad_link = _noncanonical_version_link(text)
    if bad_link:
        print(
            f"error: CHANGELOG.md line {bad_link[0]}: version-link-like "
            f"construct {bad_link[1]!r} outside the canonical column-zero "
            "link block (CommonMark would register it as a definition, "
            "invisible to the duplicate/target scans) — fix the file "
            "before compiling",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    # No persistent Markdown block context may exist at all — an
    # unterminated fence/<pre> would swallow everything after it
    # (the terminal link block included) without breaking any run.
    blk = _block_context_violation(text)
    if blk:
        print(
            f"error: CHANGELOG.md line {blk[0]}: persistent Markdown block "
            f"context starter {blk[1]!r} — fenced code / raw-HTML blocks "
            "can swallow the sections and link definitions after them, so "
            "the compiler refuses them anywhere in the file",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    # And the inverse: canonical-LOOKING definitions must all live in the
    # terminal contiguous link block — see _link_block_violation.
    stray = _link_block_violation(text)
    if stray:
        print(
            f"error: CHANGELOG.md line {stray[0]}: definition-like line "
            f"{stray[1]!r} outside the terminal link block (it may sit in a "
            "fenced/HTML block and not render, yet the scans would treat it "
            "as canonical) — move it into the contiguous block at the end "
            "of the file",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    headers = _existing_headers(text)
    versions = [h[0] for h in headers]
    duplicated = sorted({v for v in versions if versions.count(v) > 1})
    if duplicated:
        # A duplicated release header is a corrupt changelog whichever
        # version it is: the loop below inspects only the first match, so
        # exit 4 could otherwise certify a file that still carries a second
        # '## [X.Y.Z]' section.
        print(
            "error: duplicated release header(s) in CHANGELOG.md: "
            + ", ".join(f"'## [{v}]'" for v in duplicated)
            + " — fix the file before compiling",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    # Link-definition duplicate check runs BEFORE the existing-target path:
    # exit 4 must never certify a changelog carrying two definitions for
    # any version (one correct + one wrong would otherwise pass).
    # Label-only match: CommonMark permits '[1.3.0]:https://...' with NO
    # whitespace after the colon (and even a next-line destination), so
    # requiring '\s+<dest>' here would let such a definition hide from the
    # duplicate and already-defined-target checks while still controlling
    # the rendered link (CommonMark resolves a label to its FIRST
    # definition).
    link_versions = re.findall(r"^\[(\d+\.\d+\.\d+)\]:", text, flags=re.MULTILINE)
    dup_links = sorted({v for v in link_versions if link_versions.count(v) > 1})
    if dup_links:
        print(
            "error: duplicated comparison-link definition(s) in "
            "CHANGELOG.md: " + ", ".join(f"'[{v}]:'" for v in dup_links),
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    prev = max(versions, key=_semver_tuple) if versions else None
    if prev is not None:
        # The anchor header is the compiler's own output from the previous
        # cycle, so its date must be canonical (a fresh compile atop
        # '## [X.Y.Z]' with no/impossible date would silently extend a
        # corrupt tip). OLDER legacy headers are deliberately exempt - the
        # real CHANGELOG carries dateless pre-convention releases
        # (## [0.6.0] and earlier) that the compiler never touches.
        prev_date = next(d for v, d, _ in headers if v == prev)
        if prev_date is None or not _canonical_date(prev_date):
            print(
                f"error: the latest release header '## [{prev}]' has a "
                f"missing or non-canonical date ({prev_date!r}) — fix it "
                "before compiling a new release on top",
                file=sys.stderr,
            )
            return EXIT_FINDINGS
    if previous_version is not None and prev is not None and previous_version != prev:
        # Cross-check against the caller's package-metadata version
        # (bump-version's OLD_VERSION): a drifted CHANGELOG would otherwise
        # silently anchor the comparison link to the wrong ancestor and
        # delete the fragments before anyone noticed. EXCEPTION: when the
        # latest release IS the compile target, this is the interrupted-bump
        # recovery shape (compile finished, package metadata not yet
        # updated) — fall through to the existing-target path, which itself
        # validates previous_version against the target's real predecessor.
        if prev != version:
            print(
                f"error: --previous-version {previous_version!r} does not match "
                f"the latest CHANGELOG.md release {prev!r} — package metadata "
                "and changelog have drifted; reconcile before compiling",
                file=sys.stderr,
            )
            return EXIT_FINDINGS

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
            section_errors = _section_grammar_errors(text, hstart)
            if section_errors:
                print(
                    f"error: '## [{version}]' exists but its section is not "
                    "a state compile could have produced — not a completed "
                    "compile; fix the section:\n  " + "\n  ".join(section_errors),
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
            if previous_version is not None and previous_version not in (
                version,
                predecessor,
            ):
                # Recovery accepts package metadata at the target (bump
                # completed) or its immediate predecessor (interrupted
                # before the metadata update); anything else is drift.
                print(
                    f"error: --previous-version {previous_version!r} matches "
                    f"neither '## [{version}]' nor its predecessor "
                    f"{predecessor!r} — package metadata and changelog have "
                    "drifted; reconcile before re-running",
                    file=sys.stderr,
                )
                return EXIT_FINDINGS
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
                # The target link must use the SAME base URL as the
                # predecessor's link (the base the compiler itself would
                # have written) - a correct-version link pointing at an
                # unrelated repository is not a completed compile.
                pred_m = re.search(
                    r"^\[" + re.escape(predecessor) + r"\]:\s+(\S+?)/(?:compare|releases)/\S+$",
                    text,
                    flags=re.MULTILINE,
                )
                if pred_m is None:
                    # The compiler's own output always links the
                    # predecessor (compare/, or releases/tag/ for the very
                    # first release) - no link means this is not a state
                    # compile produced; never fall back to accepting any
                    # base.
                    print(
                        f"error: no link definition found for the "
                        f"predecessor '[{predecessor}]' — not a completed "
                        "compile; fix the link block",
                        file=sys.stderr,
                    )
                    return EXIT_FINDINGS
                pred_base = re.escape(pred_m.group(1))
                link_re = re.compile(
                    r"^\["
                    + re.escape(version)
                    + r"\]: "
                    + pred_base
                    + r"/compare/v"
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

    if not allow_dirty:
        # Committed-content guard: every fragment about to be compiled and
        # DELETED must be a regular file whose bytes match its HEAD blob.
        # This is deliberately not a `git status` parse — status output
        # depends on configuration (status.showUntrackedFiles=no, ignore
        # rules via .git/info/exclude or a global gitignore) and does not
        # see through symlinks, all of which could let unreviewed content
        # be swept into the release and destroyed.
        if not (root / ".git").exists():
            print(
                "error: no .git at the resolved root — cannot verify the "
                "fragments are committed; pass --allow-dirty for scratch "
                "copies",
                file=sys.stderr,
            )
            return EXIT_FINDINGS
        problems = []
        for p in fragments:
            rel = f"changelog.d/{p.name}"
            if p.is_symlink() or not p.is_file():
                problems.append(f"{rel}: not a regular file (symlink or special)")
                continue
            tree_res = subprocess.run(
                ["git", "-C", str(root), "ls-tree", "HEAD", "--", rel],
                capture_output=True,
                text=True,
            )
            tree = tree_res.stdout.strip()
            if tree_res.returncode != 0 or not tree:
                # Missing from HEAD — including an unborn HEAD (init with no
                # commits), where ls-tree itself fails.
                problems.append(f"{rel}: not committed in HEAD")
                continue
            mode, _, sha = tree.split()[0], tree.split()[1], tree.split()[2]
            if mode not in ("100644", "100755"):
                problems.append(f"{rel}: committed as a non-regular entry (mode {mode})")
                continue
            blob = subprocess.run(
                ["git", "-C", str(root), "cat-file", "blob", sha],
                capture_output=True,
                check=True,
            ).stdout
            if blob != p.read_bytes():
                problems.append(f"{rel}: worktree bytes differ from the HEAD blob")
        # Set equality with HEAD: a committed fragment DELETED from the
        # worktree would otherwise be silently dropped from the release
        # (the compile removes every fragment file, so the git diff shows
        # both deletions while only the surviving one was compiled).
        head_res = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD", "--", "changelog.d"],
            capture_output=True,
            text=True,
        )
        if head_res.returncode == 0:
            head_frags = {
                Path(line).name
                for line in head_res.stdout.splitlines()
                if FRAGMENT_NAME_RE.match(Path(line).name)
            }
            worktree_frags = {p.name for p in fragments}
            for missing in sorted(head_frags - worktree_frags):
                problems.append(
                    f"changelog.d/{missing}: committed in HEAD but absent "
                    "from the worktree - its release note would be "
                    "silently lost"
                )
        else:
            # Fail CLOSED: an unenumerable HEAD means the deleted-fragment
            # guard cannot run, so the dirty state is unverifiable. (An
            # unborn HEAD is already refused above — every present fragment
            # was flagged "not committed in HEAD".)
            problems.append(
                "changelog.d: cannot enumerate HEAD's fragments "
                f"(git ls-tree failed: {head_res.stderr.strip() or 'unknown error'}) "
                "- deleted-fragment check is unverifiable"
            )
        if problems:
            print(
                "error: changelog.d/ fragments must be committed unchanged "
                "before a release (they are compiled into CHANGELOG.md and "
                "deleted):\n  "
                + "\n  ".join(problems)
                + "\ncommit them first, or pass --allow-dirty",
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

    # A definition for the TARGET version must not already exist on the
    # fresh-compile path (compiling would emit a duplicate '[X.Y.Z]:' line;
    # duplicates generally were rejected before the existing-target path).
    if version in link_versions:
        print(
            f"error: a comparison-link definition '[{version}]:' already "
            "exists in CHANGELOG.md but the release header does not — "
            "inconsistent state; fix the link block before compiling",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    # Same predecessor-link form the exit-4 path accepts: compare/ links,
    # or releases/tag/ for a sole first release (its second release could
    # not compile otherwise); whitespace-tolerant after ':'.
    prev_link_re = re.compile(
        r"^\[" + re.escape(prev) + r"\]:\s+(\S+?)/(?:compare|releases)/\S+$",
        flags=re.MULTILINE,
    )
    m = prev_link_re.search(text)
    if not m:
        print(
            f"error: comparison link '[{prev}]: .../compare/...' (or the "
            "first release's .../releases/tag/... form) not found at the "
            "bottom of CHANGELOG.md",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    base = m.group(1)
    new_link = f"[{version}]: {base}/compare/v{prev}...v{version}\n"
    text = text[: m.start()] + new_link + text[m.start() :]

    # Backstop on the fully assembled text (multi-line labels can span the
    # fragment-line checks): the only version-link definitions in the
    # output must be the canonical column-zero ones.
    bad_link = _noncanonical_version_link(text)
    if bad_link:
        print(
            f"error: assembled CHANGELOG line {bad_link[0]}: version-link-"
            f"like construct {bad_link[1]!r} would be written outside the "
            "canonical link block — refusing to compile",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    stray = _link_block_violation(text)
    if stray:
        print(
            f"error: assembled CHANGELOG line {stray[0]}: definition-like "
            f"line {stray[1]!r} would sit outside the terminal link block — "
            "refusing to compile",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
    blk = _block_context_violation(text)
    if blk:
        # A fragment continuation line indented a single space could land a
        # 1-space fence at top level once compiled — catch it here.
        print(
            f"error: assembled CHANGELOG line {blk[0]}: persistent Markdown "
            f"block context starter {blk[1]!r} would be written — refusing "
            "to compile",
            file=sys.stderr,
        )
        return EXIT_FINDINGS

    original_text = changelog_path.read_text()
    fragment_bytes = {p: p.read_bytes() for p in fragments}
    changelog_path.write_text(text)
    try:
        for p in fragments:
            p.unlink()
    except OSError as exc:
        # Roll back to the pre-compile state: a half-deleted fragment set
        # beside an already-written release section would need manual
        # recovery (the next run deliberately refuses that shape).
        changelog_path.write_text(original_text)
        for p, data in fragment_bytes.items():
            if not p.exists():
                p.write_bytes(data)
        print(
            f"error: fragment deletion failed ({exc}); CHANGELOG.md and "
            "all fragments restored - nothing was compiled",
            file=sys.stderr,
        )
        return EXIT_FINDINGS
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
    comp.add_argument(
        "--previous-version",
        default=None,
        help="cross-check: must equal the latest CHANGELOG.md release "
        "(bump-version passes its package-metadata OLD_VERSION)",
    )
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
    return run_compile(root, args.version, args.date, args.allow_dirty, args.previous_version)


if __name__ == "__main__":
    sys.exit(main())
