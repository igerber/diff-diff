"""Tracking-file contract guards for TODO.md / DEFERRED.md.

Origin: tracking-split local review R2 (TODO row). The split's contract:

- TODO.md is the ACTIONABLE backlog; deferred/blocked work lives in
  DEFERRED.md. A TODO table row that points its work at DEFERRED.md is a
  misfiled deferral (prose links — the header's "Related tracking surfaces"
  block and past-tense "moved to" notes — are legal and are not table rows).
- Lifecycle state (deprecated_in / removed_in / status) for `M-xxx` ledger
  ids lives ONLY in the CI-enforced ``docs/v4-deprecations.yaml``; tracking
  rows cross-link ids but must not restate ledger fields (which would go
  stale silently — the ledger tests probe the yaml, not markdown prose).
- The table shapes are documented contract (CLAUDE.md "Tracking-file map"):
  TODO Actionable rows carry an Effort column, DEFERRED blocker rows a PR
  column, and the DEFERRED Decision record has its own three-column shape.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TODO = REPO_ROOT / "TODO.md"
DEFERRED = REPO_ROOT / "DEFERRED.md"

# Isolated-install CI runs from a wheel with no tracking files; these guards
# only make sense against a full checkout (the test_docs_ia.py pattern).
pytestmark = pytest.mark.skipif(
    not TODO.exists() or not DEFERRED.exists(),
    reason="tracking files not present (isolated-install CI)",
)

TODO_HEADER = "| Issue | Location | Origin | Effort | Priority |"
DEFERRED_BLOCKER_HEADER = "| Issue | Location | PR | Priority |"
DEFERRED_DECISION_HEADER = "| Decision | Location | Verified |"

_HEADER_SET = {TODO_HEADER, DEFERRED_BLOCKER_HEADER, DEFERRED_DECISION_HEADER}

# Ledger-lifecycle restatements forbidden in rows that cross-link an M-id.
# Field-name tokens (the yaml spellings: status/deprecated_in/removed_in)
# plus "<lifecycle word> in <version>" prose forms. Deliberately NARROW:
# "the M-011 removal" (naming a transition, no version) and "Lifecycle
# tracked in docs/v4-deprecations.yaml (M-008)" (a pointer) are legal;
# DEFERRED's Version-gated section may say "at v4" (a section-level frame,
# not a per-row field restatement).
# ``status:`` keeps the colon OUTSIDE a trailing word boundary (":" is not a
# word character, so ``status:\b`` would never match "status: planned");
# the version form accepts an optional ``v`` and Markdown delimiters
# (``removed in v4``, ``deprecated in **3.9**``).
# Field-name forms: bare, backticked/emphasized, and colon/equals
# assignment spellings (``status: planned``, `` `status` = planned``).
_LEDGER_TOKENS = re.compile(
    # ``status`` needs an assignment marker (the bare word is common prose);
    # the underscore field names are ledger vocabulary on their own.
    r"[`*_]*\bstatus\b[`*_]*\s*[:=]|\b(?:deprecated_in|removed_in)\b",
    re.I,
)
# "<lifecycle word> <prep> <version>" prose (in/at/for/by/with), the
# version-before-noun form ("its 4.0 removal", "the v4 deprecation"), and
# the noun-<prep>-version form ("removal scheduled for v4"). The (?<![-\w])
# guard keeps M-id digits ("M-011 removal" names a transition, not a
# version target) from matching as a version. This detector is a
# BEST-EFFORT TRIPWIRE over prose, not a parser: it is calibrated to catch
# every restatement form found in the repo plus the common rephrasings
# below, while "default flip at v4" (no lifecycle stem) and bare
# transition names stay legal.
_LIFECYCLE_VERSION = re.compile(
    r"\b(?:deprecated|removed|shimmed|planned)\b[^|]{0,40}?"
    r"\b(?:in|at|for|by|with)\s+[\"'*_`]*v?\d"
    r"|(?<![-\w])v?\d[\d.]*\s+(?:removal|deprecation)\b"
    r"|\b(?:removal|deprecation)\b[^|]{0,30}?\b(?:at|for|by|with|in)\s+[\"'*_`]*v?\d",
    re.I,
)
_M_ID = re.compile(r"\bM-\d{3}\b")


def _restates_lifecycle(line: str) -> bool:
    """True when a row restates ledger lifecycle state (forbidden beside
    an M-xxx cross-link)."""
    return bool(_LEDGER_TOKENS.search(line) or _LIFECYCLE_VERSION.search(line))


def _split_unescaped_pipes(s: str):
    r"""Split on pipes preceded by an EVEN run of backslashes.

    Markdown escaping is parity-based: ``\|`` is a literal pipe, ``\\|``
    is a literal backslash followed by a cell separator, ``\\\|`` a
    literal backslash + literal pipe. A fixed-width ``(?<!\\)`` lookbehind
    gets the even cases wrong.
    """
    cells: list = []
    cur: list = []
    run = 0
    for ch in s:
        if ch == "\\":
            run += 1
            cur.append(ch)
            continue
        if ch == "|" and run % 2 == 0:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        run = 0
    cells.append("".join(cur))
    return cells


def _cells(row: str):
    r"""Cell texts of a pipe row, honoring escaped ``\|`` inside cells."""
    parts = _split_unescaped_pipes(row.strip())
    if parts and parts[0] == "":  # leading pipe
        parts = parts[1:]
    if parts and parts[-1] == "":  # trailing unescaped pipe
        parts = parts[:-1]
    return [c.strip() for c in parts]


def _normalize(row: str) -> str:
    """Canonical pipe-row form: single spaces around cell text.

    Markdown accepts ``|Issue|Location|`` as readily as ``| Issue | Location |``;
    normalizing keeps the guard from being bypassed by spacing.
    """
    return "| " + " | ".join(_cells(row)) + " |"


_DIVIDER_CELL = re.compile(r"^:?-+:?$")


def _is_divider(row: str) -> bool:
    """GFM delimiter-row grammar: every cell is ``:?-+:?`` (an empty or
    colon-only cell makes the whole table not render)."""
    if not row or "-" not in row:
        return False
    cells = _cells(row)
    return bool(cells) and all(_DIVIDER_CELL.match(c) for c in cells)


def _tables(path: Path):
    """Yield (header_lineno, normalized_header, [(lineno, normalized_row)])
    for every contiguous table block.

    A block is any run of ``|``-prefixed lines, OR a run of pipe-containing
    lines whose second line is a divider (GFM also accepts tables WITHOUT
    leading pipes - ``Issue | Location`` over ``---|---`` - and those must
    not bypass the guard; the divider requirement keeps prose that merely
    contains a pipe from being misread as a table).
    """
    block: list = []
    for i, line in enumerate(path.read_text().split("\n") + [""], 1):
        stripped = line.strip()
        if stripped.startswith("|") or ("|" in stripped and (block or _looks_tabular(path, i))):
            block.append((i, stripped))
            continue
        if block:
            yield from _flush_block(block)
            block = []
    if block:
        yield from _flush_block(block)


def _looks_tabular(path: Path, lineno: int) -> bool:
    """True when the line AFTER ``lineno`` is a divider row (the GFM
    no-leading-pipe table shape's signature)."""
    lines = path.read_text().split("\n")
    return lineno < len(lines) and _is_divider(lines[lineno])


def _flush_block(block):
    header_i, header = block[0]
    # A no-leading-pipe block only counts as a table when its second line
    # is a divider; a leading-pipe block always does (current convention).
    if not header.startswith("|") and not (len(block) >= 2 and _is_divider(block[1][1])):
        return
    # A recognized table must be WELL-FORMED: GFM only renders a table when
    # line 2 is a divider whose cell count equals the header's. A pipe-shaped
    # block failing that renders as prose, so its rows would silently escape
    # every row check below - fail loudly instead of skipping.
    if len(block) < 2 or not _is_divider(block[1][1]):
        raise AssertionError(
            f"line {header_i}: pipe-shaped block has no divider on its second "
            f"line - not a rendering table, rows would escape the guard: "
            f"{header[:80]!r}"
        )
    if len(_cells(block[1][1])) != len(_cells(header)):
        raise AssertionError(
            f"line {header_i + 1}: table divider has {len(_cells(block[1][1]))} "
            f"cells but the header has {len(_cells(header))} - GFM will not "
            f"render this as a table: {header[:80]!r}"
        )
    rows = [(j, _normalize(ln)) for j, ln in block[1:] if not _is_divider(ln)]
    yield header_i, _normalize(header), rows


def _table_rows(path: Path):
    """Yield (lineno, normalized line) for table DATA rows of every block."""
    for _, _, rows in _tables(path):
        yield from rows


class TestTodoNoDeferredPointers:
    def test_no_table_row_points_work_at_deferred(self):
        offenders = [(i, ln[:100]) for i, ln in _table_rows(TODO) if "DEFERRED.md" in ln]
        assert not offenders, (
            "TODO.md table rows must not point work at DEFERRED.md - a "
            "deferred/blocked item belongs in DEFERRED.md itself (prose "
            f"links outside table rows are fine): {offenders}"
        )


class TestNoLedgerRestatement:
    @pytest.mark.parametrize("path", [TODO, DEFERRED], ids=lambda p: p.name)
    def test_m_id_rows_do_not_restate_lifecycle(self, path):
        offenders = []
        for i, ln in _table_rows(path):
            if not _M_ID.search(ln):
                continue
            if _restates_lifecycle(ln):
                offenders.append((i, ln[:100]))
        assert not offenders, (
            f"{path.name} rows cross-linking an M-xxx id must not restate "
            "ledger lifecycle fields (status/deprecated_in/removed_in or "
            "'<lifecycle word> in <version>') - the CI-enforced "
            f"docs/v4-deprecations.yaml is the only authority: {offenders}"
        )


class TestTableShapes:
    def test_todo_headers_are_the_actionable_shape(self):
        headers = [(i, h) for i, h, _ in _tables(TODO)]
        assert headers, "TODO.md has no tables"
        bad = [(i, h) for i, h in headers if h != TODO_HEADER]
        assert not bad, f"undocumented TODO.md table header shape(s): {bad}"

    def test_deferred_headers_are_the_documented_shapes(self):
        headers = [(i, h) for i, h, _ in _tables(DEFERRED)]
        assert headers, "DEFERRED.md has no tables"
        bad = [
            (i, h)
            for i, h in headers
            if h not in (DEFERRED_BLOCKER_HEADER, DEFERRED_DECISION_HEADER)
        ]
        assert not bad, f"undocumented DEFERRED.md table header shape(s): {bad}"


class TestLifecycleRegexCalibration:
    """The guard's own detector fixtures: prohibited spellings are caught,
    the legitimate current-row shapes are not."""

    @pytest.mark.parametrize(
        "text",
        [
            "| M-001 status: planned | x | y |",
            "| M-001 removed in v4 | x | y |",
            "| M-001 deprecated in **3.9** | x | y |",
            "| M-001 removed in 4.0 | x | y |",
            "| M-001 deprecated_in restated | x | y |",
            "| M-001 shimmed in '3.9' | x | y |",
            "| M-001 until its 4.0 removal | x | y |",
            "| M-001 `status`: planned | x | y |",
            "| M-001 status = planned | x | y |",
            "| M-001 the v4 deprecation | x | y |",
            "| M-010 will be removed with v4 | x | y |",
            "| M-010 removal scheduled for v4 | x | y |",
            "| M-010 planned for 4.0 | x | y |",
        ],
    )
    def test_prohibited_spellings_are_caught(self, text):
        assert _restates_lifecycle(text)

    @pytest.mark.parametrize(
        "text",
        [
            "| interplay with the M-011 removal | x | y |",
            "| Lifecycle tracked in docs/v4-deprecations.yaml (M-008) | x | y |",
            "| default flip at v4 (M-004) | x | y |",
            "| the M-024 renamed key | x | y |",
            "| the M-011 removal interplay | x | y |",
        ],
    )
    def test_legitimate_shapes_pass(self, text):
        assert not _restates_lifecycle(text)


class TestTableParserCalibration:
    """The parser recognizes unspaced pipe rows and unknown headers."""

    def test_unspaced_rows_are_normalized_and_seen(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text(
            "|Issue|Location|Origin|Effort|Priority|\n|---|---|---|---|---|\n|work DEFERRED.md|x|y|z|w|\n"
        )
        tables = list(_tables(f))
        assert tables[0][1] == TODO_HEADER  # unspaced header normalizes
        rows = list(_table_rows(f))
        assert rows and "DEFERRED.md" in rows[0][1]

    def test_unknown_header_is_a_distinct_shape(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text("| Task | Who |\n|------|-----|\n| a | b |\n")
        ((_, header, _),) = _tables(f)
        assert header == "| Task | Who |"
        assert header not in _HEADER_SET


class TestRowShapes:
    r"""Every data row carries exactly its table header's column count
    (escaped ``\|`` inside cells honored)."""

    @pytest.mark.parametrize("path", [TODO, DEFERRED], ids=lambda p: p.name)
    def test_data_rows_match_header_column_count(self, path):
        offenders = []
        for _, header, rows in _tables(path):
            n = len(_cells(header))
            for i, ln in rows:
                if len(_cells(ln)) != n:
                    offenders.append((i, len(_cells(ln)), n, ln[:80]))
        assert not offenders, (
            f"{path.name} rows with a cell count differing from their "
            f"header (line, got, want, row): {offenders}"
        )

    def test_missing_and_extra_cells_are_detected(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text(
            "| Issue | Location | Origin | Effort | Priority |\n"
            "|---|---|---|---|---|\n"
            "| short row | only two |\n"
            "| a | b | c | d | e | extra |\n"
            "| ok `a \\| b` cell | x | y | z | w |\n"
        )
        ((_, header, rows),) = _tables(f)
        n = len(_cells(header))
        counts = [len(_cells(ln)) for _, ln in rows]
        assert counts == [2, 6, n]  # short, extra, escaped-pipe row intact


class TestBackslashParity:
    r"""Markdown pipe escaping is parity-based: ``\|`` literal pipe,
    ``\\|`` literal backslash + SEPARATOR, ``\\\|`` backslash + literal
    pipe. A fixed-width lookbehind gets the even runs wrong."""

    @pytest.mark.parametrize(
        "row,expected",
        [
            (r"| a \| b | x |", 2),  # odd: escaped pipe stays in-cell
            (r"| a \\| b | x |", 3),  # even: separator after literal backslash
            (r"| a \\\| b | x |", 2),  # odd again
            (r"| a \\\\| b | x |", 3),  # even again
        ],
    )
    def test_backslash_run_parity(self, row, expected):
        assert len(_cells(row)) == expected

    def test_trailing_backslash_pipe_parity(self):
        # '\\|' at row end: the pipe is a real (trailing) delimiter.
        assert _cells(r"| a | b \\|") == ["a", r"b \\"]
        # '\|' at row end: escaped pipe belongs to the cell.
        assert _cells(r"| a | b \|") == ["a", r"b \|"]


class TestLifecycleCaseInsensitivity:
    @pytest.mark.parametrize(
        "text",
        [
            "| M-001 Status: planned | x |",
            "| M-001 REMOVED_IN: 4.0 | x |",
            "| M-001 Deprecated In 3.9 | x |",
        ],
    )
    def test_mixed_case_restatements_are_caught(self, text):
        assert _restates_lifecycle(text)


class TestNoLeadingPipeTables:
    """GFM tables written without leading pipes are recognized too."""

    def test_no_leading_pipe_table_is_parsed(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text(
            "Issue | Location | Origin | Effort | Priority\n"
            "------|----------|--------|--------|----------\n"
            "work DEFERRED.md | x | y | z | w\n"
        )
        tables = list(_tables(f))
        assert tables and tables[0][1] == TODO_HEADER
        rows = list(_table_rows(f))
        assert rows and "DEFERRED.md" in rows[0][1]

    def test_prose_with_a_pipe_is_not_a_table(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text("some prose with a | pipe in it\nand a second line\n")
        assert list(_tables(f)) == []


class TestMalformedTablesFailLoudly:
    """A recognized (pipe-shaped) table missing a well-formed divider does
    not render in GFM, so its rows would escape every row check - the
    parser fails loudly instead of silently skipping the block."""

    def test_missing_divider_fails(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text(
            "| Issue | Location | Origin | Effort | Priority |\n"
            "| a row with no divider above it | x | y | z | w |\n"
        )
        with pytest.raises(AssertionError, match="no divider"):
            list(_tables(f))

    def test_short_divider_fails(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text(
            "| Issue | Location | Origin | Effort | Priority |\n"
            "|---|---|---|\n"
            "| a | b | c | d | e |\n"
        )
        with pytest.raises(AssertionError, match="3 cells but the header has 5"):
            list(_tables(f))

    def test_no_leading_pipe_short_divider_fails(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text("Issue | Location | Origin\n" "------|----------\n" "a | b | c\n")
        with pytest.raises(AssertionError, match="2 cells but the header has 3"):
            list(_tables(f))

    @pytest.mark.parametrize(
        "divider",
        ["|---||---|", "|:|---|", "| |---|"],
        ids=["empty-cell", "colon-only-cell", "space-cell"],
    )
    def test_invalid_divider_cell_fails(self, tmp_path, divider):
        # GFM requires every delimiter cell to be ':?-+:?'; an empty or
        # colon-only cell stops the whole table from rendering.
        assert not _is_divider(divider)
        f = tmp_path / "t.md"
        f.write_text(f"| a | b |\n{divider}\n| x | y |\n")
        with pytest.raises(AssertionError, match="no divider"):
            list(_tables(f))

    @pytest.mark.parametrize("path", [TODO, DEFERRED], ids=lambda p: p.name)
    def test_real_tracking_files_are_well_formed(self, path):
        # The strict parser raises on a malformed table; consuming every
        # block is itself the assertion.
        assert list(_tables(path))


class TestWorkflowPins:
    """The docs-tests lane is this guard's ONLY CI lane: pin its path
    filters and step so a workflow edit cannot silently de-wire it (the
    changelog-fragment guard's TestWorkflowPins precedent)."""

    _ENTRIES = [
        "tests/test_tracking_files.py",
        "TODO.md",
        "DEFERRED.md",
        ".github/workflows/lwdid-data-canary.yml",
    ]

    @pytest.fixture(scope="class")
    def docs_tests_text(self):
        path = REPO_ROOT / ".github" / "workflows" / "docs-tests.yml"
        if not path.exists():
            pytest.skip("docs-tests.yml not present")
        return path.read_text()

    @pytest.mark.parametrize("trigger", ["push:", "pull_request:"])
    def test_path_filters_cover_tracking_surfaces(self, docs_tests_text, trigger):
        on_block = docs_tests_text[: docs_tests_text.index("\njobs:")]
        tstart = on_block.index(trigger)
        others = [
            on_block.index(t)
            for t in ("push:", "pull_request:", "schedule:", "workflow_dispatch:")
            if t != trigger and t in on_block and on_block.index(t) > tstart
        ]
        tblock = on_block[tstart : min(others)] if others else on_block[tstart:]
        for entry in self._ENTRIES:
            assert f"'{entry}'" in tblock, (
                f"docs-tests.yml {trigger} paths filter is missing {entry!r} - "
                "a tracking-file-only PR would run no CI guard"
            )

    def test_step_invokes_guard(self, docs_tests_text):
        jobs = docs_tests_text[docs_tests_text.index("\njobs:") :]
        assert (
            "pytest tests/test_tracking_files.py" in jobs
        ), "docs-tests.yml no longer runs the tracking-file guard"


class TestCanaryWorkflowPins:
    """Pin the LWDiD data-canary lane's load-bearing pieces: it has no PR
    trigger, so without these pins a canary-only edit could merge with the
    loaders/provenance/replication wiring silently broken and the first
    signal would be the next weekly run."""

    @pytest.fixture(scope="class")
    def canary_text(self):
        path = REPO_ROOT / ".github" / "workflows" / "lwdid-data-canary.yml"
        if not path.exists():
            pytest.skip("lwdid-data-canary.yml not present")
        return path.read_text()

    def test_triggers(self, canary_text):
        assert "workflow_dispatch:" in canary_text
        assert "schedule:" in canary_text and "cron:" in canary_text

    def test_canary_step_wiring(self, canary_text):
        for needle in (
            "load_prop99",
            "load_walmart",
            "lwdid_ssc_ancillary",
            "simplefilter('error')",
        ):
            assert needle in canary_text, f"canary step lost {needle!r}"

    def test_replication_step_and_backend(self, canary_text):
        assert "pytest tests/test_methodology_lwdid.py" in canary_text
        assert "DIFF_DIFF_BACKEND: python" in canary_text
