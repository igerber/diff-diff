"""Structural guards for the documentation information architecture.

The RTD site's navigation contract (PR #703: docs/index.rst -> exactly five
section landing pages; pydata-sphinx-theme flattens every root-toctree entry
into the header navbar) is NOT enforced by the Sphinx -W build: a sixth root
entry, a tutorial missing its landing-page card, a homepage catalog row
omitted for a new estimator, or a class documented only under ``:no-index:``
(dead cross-references) all build clean and silently erode the IA.

These tests pin those invariants by parsing the RST sources directly - no
Sphinx build required. Conventions live in CLAUDE.md ("README discipline")
and CONTRIBUTING.md ("Documentation Requirements").

Scope: these guards are drift tripwires for the RST conventions this
repository actually uses (hand-written directives at column zero or nested
one level, autosummary blocks with ``:toctree:``, grid cards carrying their
options in the contiguous option block). They are deliberately NOT a general
RST validator - exotic-but-valid RST the repo does not use is out of scope
by design, and hardening beyond the conventions in use adds maintenance
surface without catching real drift.
"""

import importlib
import re
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# Isolated-install CI (Python Tests workflow) runs from a wheel install with
# no docs/ tree; these guards only make sense against a full checkout.
pytestmark = pytest.mark.skipif(
    not (DOCS / "index.rst").exists(),
    reason="docs tree not present (isolated-install CI)",
)

# The five section landing pages, in navbar order. A change here is a
# deliberate IA decision: update CLAUDE.md + CONTRIBUTING.md in the same PR.
ROOT_SECTIONS = [
    "getting_started",
    "practitioners",
    "tutorials/index",
    "user_guide",
    "api/index",
]

# Sidebar labels beyond this read as multi-line clutter in the primary
# sidebar (the pre-#703 mobile menu wrapped labels to 3-4 lines each).
MAX_TUTORIAL_LABEL_LEN = 40


def _directive_bodies(text, name):
    """Indented body of each ``.. <name>::`` directive (header may carry
    trailing whitespace; argument-less directives only, which covers
    ``toctree`` and the bare ``autosummary`` blocks used in this repo)."""
    bodies = []
    for match in re.finditer(
        rf"^\.\. {re.escape(name)}::[ \t]*\n((?:[ \t]+.*\n|[ \t]*\n)*)", text, re.M
    ):
        bodies.append(match.group(1))
    return bodies


def _toctree_blocks(text):
    """Entry-lists for each ``.. toctree::`` directive in an RST source."""
    blocks = []
    for body in _directive_bodies(text, "toctree"):
        entries = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            entries.append(stripped)
        blocks.append(entries)
    return blocks


def _autosummary_entries(text):
    """``diff_diff.*`` qualnames from stub-generating ``autosummary`` blocks.

    Scoped to autosummary directive bodies so an indented example line
    elsewhere (e.g. inside a code-block) cannot register as a catalog entry,
    and gated on the ``:toctree:`` option - an autosummary block without it
    renders a table but generates NO stub page, so its entries are not
    canonical documentation targets.
    """
    entries = []
    for body in _directive_bodies(text, "autosummary"):
        if not re.search(r"^[ \t]+:toctree:", body, re.M):
            continue
        entries.extend(re.findall(r"^[ \t]+diff_diff\.((?:\w+\.)*\w+)[ \t]*$", body, re.M))
    return entries


def _docname(entry):
    """Docname of a toctree entry, unwrapping ``Label <docname>`` form."""
    match = re.match(r".*<(.+)>\s*$", entry)
    return match.group(1) if match else entry


def _section_body(text, title):
    """Body of an underlined RST section, from its header to the next one."""
    pattern = rf"^{re.escape(title)}\n[-=~]+\n(.*?)(?=^\S[^\n]*\n[-=~]+\n|\Z)"
    match = re.search(pattern, text, re.M | re.S)
    assert match is not None, f"section {title!r} not found"
    return match.group(1)


def _h2_sections(text):
    """(title, body) pairs for every dash-underlined H2 section."""
    matches = list(re.finditer(r"^(\S[^\n]*)\n-{3,}\n", text, re.M))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(1), text[match.end() : end]))
    return sections


def _card_links(text):
    """Doc targets of ``grid-item-card`` directives with ``:link-type: doc``.

    Only the contiguous option block directly under the directive header is
    inspected, so card BODY content that merely mentions ``:link:`` cannot
    register a notebook, and a stray ``:link:`` in some other directive
    never counts.
    """
    links = []
    for match in re.finditer(
        r"^([ \t]+)\.\. grid-item-card::[^\n]*\n((?:\1[ \t]+:[^\n]*\n)*)", text, re.M
    ):
        options = match.group(2)
        link = re.search(r":link:[ \t]+(\S+)", options)
        if link and re.search(r":link-type:[ \t]+doc\b", options):
            links.append(link.group(1))
    return links


def _estimator_first_cells(section_body):
    """First-column cell text of every list-table row, continuations joined.

    Walks complete row blocks so a first cell that is empty on the marker
    line or continues on deeper-indented lines is still captured as one
    cell rather than silently skipped.
    """
    cells = []
    lines = section_body.splitlines()
    i = 0
    while i < len(lines):
        marker = re.match(r"^([ \t]+)\* -[ \t]*(.*)$", lines[i])
        if not marker:
            i += 1
            continue
        indent = marker.group(1)
        parts = [marker.group(2).strip()]
        i += 1
        # In a list-table row, the second-cell marker sits two columns in
        # from the row marker ('   * - a' then '     - b'); continuation
        # lines of the first cell are indented deeper still.
        next_row = re.compile(rf"^{re.escape(indent)}\* -")
        second_cell = re.compile(rf"^{re.escape(indent)}  -[ \t]")
        while i < len(lines):
            line = lines[i]
            if next_row.match(line) or second_cell.match(line):
                break
            if not line.strip() or not line.startswith(indent + "  "):
                break  # blank or dedent ends the cell
            parts.append(line.strip())
            i += 1
        cells.append(" ".join(part for part in parts if part))
    return cells


def _estimator_table_counts(section_body):
    """Counter of ``:class:`` targets, one per estimator-table row.

    Every first-column cell of the list-table (except the literal
    ``Estimator`` header) must carry exactly one ``:class:`` reference -
    empty or markup-less cells fail loudly, so no row can silently drop
    out of the comparison.
    """
    counts = Counter()
    for cell in _estimator_first_cells(section_body):
        if cell == "Estimator":  # list-table header row
            continue
        targets = re.findall(r":class:`~diff_diff\.(\w+)`", cell)
        assert len(targets) == 1, (
            f"estimator table row {cell!r} must contain exactly one "
            f":class:`~diff_diff.X` reference (found {len(targets)})"
        )
        counts[targets[0]] += 1
    return counts


def _no_index_autoclasses(text):
    """Qualname tails of ``:no-index:``-marked autoclass directives.

    Matches directives at ANY indentation (nesting under a container
    directive is valid RST) and tolerates trailing whitespace after the
    qualname. Only the contiguous option block is inspected - RST ends an
    option field list at the first blank line, so this is exact.
    """
    found = []
    for match in re.finditer(
        r"^([ \t]*)\.\. autoclass:: diff_diff\.(\S+?)[ \t]*\n((?:\1[ \t]+.*\n)*)",
        text,
        re.M,
    ):
        if ":no-index:" in match.group(3):
            found.append(match.group(2))
    return found


def _stub_no_index_autoclasses(text):
    """Bare-name autoclass directives carrying ``:no-index:`` in a stub.

    Stub pages use ``.. currentmodule:: diff_diff`` + a bare class name,
    so this matcher is unprefixed (unlike ``_no_index_autoclasses``).
    """
    found = []
    for match in re.finditer(
        r"^([ \t]*)\.\. autoclass:: (\S+?)[ \t]*\n((?:\1[ \t]+.*\n)*)", text, re.M
    ):
        if ":no-index:" in match.group(3):
            found.append(match.group(2))
    return found


def _scan_no_index(api_root):
    """(path, qualname_tail) for :no-index: autoclasses under ``api_root``.

    Recursive, so module pages in nested API subdirectories are covered
    (including any future nested index pages); only the TOP-LEVEL
    ``index.rst`` (the canonical catalog itself) and generated
    ``_autosummary`` stubs are excluded.
    """
    hits = []
    for path in sorted(api_root.rglob("*.rst")):
        if path == api_root / "index.rst" or "_autosummary" in path.parts:
            continue
        for qualname_tail in _no_index_autoclasses(path.read_text()):
            hits.append((path, qualname_tail))
    return hits


def _resolve(qualname_tail):
    """Resolve ``diff_diff.<tail>`` to the actual Python object.

    Object identity (not name matching) is what decides whether two
    documented paths refer to the same class - module pages use submodule
    paths (``diff_diff.bacon.BaconDecompositionResults``) while the API
    index uses the package re-export (``diff_diff.BaconDecompositionResults``).
    """
    obj = importlib.import_module("diff_diff")
    prefix = "diff_diff"
    for part in qualname_tail.split("."):
        prefix = f"{prefix}.{part}"
        # Prefer the real submodule at this dotted path: a package attribute
        # can shadow a same-named submodule (diff_diff.trop is a function,
        # while the diff_diff.trop MODULE is what holds the TROP class).
        try:
            obj = importlib.import_module(prefix)
            continue
        except ModuleNotFoundError:
            pass
        obj = getattr(obj, part)
    return obj


def test_root_toctree_is_exactly_the_five_sections():
    """docs/index.rst must hold ONE toctree listing exactly the 5 sections.

    Every additional root-toctree entry becomes a navbar item; past 5 the
    theme regrows the unusable "More" dropdown that #703 removed. New pages
    belong inside a section landing page, not at the root.
    """
    blocks = _toctree_blocks((DOCS / "index.rst").read_text())
    assert len(blocks) == 1, (
        f"docs/index.rst has {len(blocks)} toctree directives; the IA "
        "contract is a single root toctree of section landing pages"
    )
    assert [_docname(e) for e in blocks[0]] == ROOT_SECTIONS


def test_every_tutorial_has_labeled_toctree_entry_and_card():
    """Each notebook: exactly one short-labeled toctree entry + one card.

    The toctree entry drives the sidebar; the grid card is the visible
    navigation on the Tutorials landing page. A notebook with only one of
    the two is half-invisible, and -W cannot detect the missing card.
    """
    notebooks = sorted(p.stem for p in (DOCS / "tutorials").glob("*.ipynb"))
    assert notebooks, "no tutorial notebooks found"
    text = (DOCS / "tutorials" / "index.rst").read_text()

    # Exactly-once accounting runs over the ENTIRE file, so a duplicate in
    # the preamble above the first H2 cannot escape the count. The per-group
    # maps additionally pin every registration inside its H2 group section.
    entries = [e for block in _toctree_blocks(text) for e in block]
    toctree_counts = Counter(_docname(e) for e in entries)
    card_counts = Counter(_card_links(text))

    toctree_sections = {}  # docname -> [section titles]
    card_sections = {}  # docname -> [section titles]
    for title, body in _h2_sections(text):
        for block in _toctree_blocks(body):
            for entry in block:
                toctree_sections.setdefault(_docname(entry), []).append(title)
        for link in _card_links(body):
            card_sections.setdefault(link, []).append(title)

    problems = []
    for nb in notebooks:
        in_toctrees = toctree_sections.get(nb, [])
        in_cards = card_sections.get(nb, [])
        if toctree_counts.get(nb, 0) != 1 or len(in_toctrees) != 1:
            problems.append(
                f"{nb}: {toctree_counts.get(nb, 0)} toctree entries, "
                f"{len(in_toctrees)} inside group sections (want exactly 1 in a group)"
            )
        if card_counts.get(nb, 0) != 1 or len(in_cards) != 1:
            problems.append(
                f"{nb}: {card_counts.get(nb, 0)} cards, "
                f"{len(in_cards)} inside group sections (want exactly 1 in a group)"
            )
        if len(in_toctrees) == 1 and len(in_cards) == 1 and in_toctrees != in_cards:
            problems.append(
                f"{nb}: card is under {in_cards[0]!r} but toctree entry is "
                f"under {in_toctrees[0]!r} - both must live in the same group"
            )
    for name in list(toctree_counts) + list(card_counts):
        if name not in notebooks:
            problems.append(f"{name}: registered but no such notebook")
    assert not problems, "tutorials/index.rst registration drift:\n" + "\n".join(problems)

    for entry in entries:
        match = re.match(r"(.+?)\s*<.+>\s*$", entry)
        assert match, (
            f"toctree entry {entry!r} must use an explicit 'Short Label <docname>' "
            "form - bare docnames inherit the notebook's long H1 as the sidebar label"
        )
        label = match.group(1)
        assert (
            len(label) <= MAX_TUTORIAL_LABEL_LEN
        ), f"label {label!r} is {len(label)} chars (max {MAX_TUTORIAL_LABEL_LEN})"


def test_homepage_estimator_table_matches_api_catalog():
    """Homepage "Supported Estimators" rows == api/index.rst Estimators list.

    Keeps the landing-page catalog honest when estimators are added or
    removed (the drift this guards against: QDiD and five other estimators
    were missing from the homepage before #703's drift sync).
    """
    home_body = _section_body((DOCS / "index.rst").read_text(), "Supported Estimators")
    table_counts = _estimator_table_counts(home_body)
    assert table_counts, "homepage Supported Estimators table is empty"

    api_body = _section_body((DOCS / "api" / "index.rst").read_text(), "Estimators")
    api_qualnames = _autosummary_entries(api_body)
    assert api_qualnames, "api/index.rst Estimators autosummary is empty"

    # Compare by OBJECT IDENTITY (like the :no-index: guard): a same-named
    # but distinct class under another module must not satisfy parity, while
    # a submodule path and its package re-export are the same entry.
    names_by_id = {}

    def _identity_counter(qualname_tails):
        counter = Counter()
        for tail in qualname_tails:
            obj = _resolve(tail)
            counter[id(obj)] += 1
            names_by_id.setdefault(id(obj), f"diff_diff.{tail}")
        return counter

    table_ids = _identity_counter(name for name in table_counts.elements())
    api_ids = _identity_counter(api_qualnames)

    assert table_ids == api_ids, (
        f"homepage table vs API Estimators autosummary drift: "
        f"missing on homepage="
        f"{sorted(names_by_id[i] for i in (api_ids - table_ids))}, "
        f"missing or duplicated in API list="
        f"{sorted(names_by_id[i] for i in (table_ids - api_ids))}"
    )


def test_no_index_autoclass_has_canonical_autosummary_page():
    """Every ``:no-index:`` autoclass needs a canonical autosummary entry.

    Module pages document classes with ``:no-index:`` by convention; the
    indexed object lives on the generated ``api/_autosummary`` page. A class
    that is no-index'd everywhere has NO indexed object, so every
    ``:class:`` cross-reference to it renders as dead text (this happened
    to the four RDD classes before #703's drift sync). Comparison is by
    object identity, so an unrelated same-named class cannot satisfy it.
    """
    api_index = (DOCS / "api" / "index.rst").read_text()
    canonical_ids = {id(_resolve(q)) for q in _autosummary_entries(api_index)}
    assert canonical_ids, "api/index.rst has no autosummary entries"

    offenders = []
    for path, qualname_tail in _scan_no_index(DOCS / "api"):
        try:
            documented = _resolve(qualname_tail)
        except (AttributeError, ImportError) as exc:
            offenders.append(f"{path.name}: diff_diff.{qualname_tail} ({exc})")
            continue
        if id(documented) not in canonical_ids:
            offenders.append(f"{path.name}: diff_diff.{qualname_tail}")
    assert not offenders, (
        "classes documented only with :no-index: (dead :class: xrefs) - add them "
        "to the matching api/index.rst autosummary section:\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Negative fixtures: the parsers must detect the drift patterns they guard.
# ---------------------------------------------------------------------------

_SWAPPED_GROUPS_RST = """\
Group A
-------

.. grid:: 1 2 2 3

   .. grid-item-card:: Alpha
      :link: 01_alpha
      :link-type: doc

      Text.

.. toctree::
   :hidden:

   Beta <02_beta>

Group B
-------

.. grid:: 1 2 2 3

   .. grid-item-card:: Beta
      :link: 02_beta
      :link-type: doc

      Text.

.. toctree::
   :hidden:

   Alpha <01_alpha>
"""


def test_parser_detects_cross_group_swap():
    """A card and toctree entry in different groups must be attributable."""
    toctree_sections = {}
    card_sections = {}
    for title, body in _h2_sections(_SWAPPED_GROUPS_RST):
        for block in _toctree_blocks(body):
            for entry in block:
                toctree_sections.setdefault(_docname(entry), []).append(title)
        for link in _card_links(body):
            card_sections.setdefault(link, []).append(title)
    assert toctree_sections["01_alpha"] == ["Group B"]
    assert card_sections["01_alpha"] == ["Group A"]
    assert toctree_sections["01_alpha"] != card_sections["01_alpha"]


def test_parser_counts_toctree_with_trailing_whitespace():
    """A ``.. toctree::`` header with trailing spaces still counts."""
    assert _toctree_blocks(".. toctree:: \n   :hidden:\n\n   quickstart\n") == [["quickstart"]]


def test_parser_ignores_link_outside_card_option_block():
    """Only the card's contiguous option block registers a notebook."""
    other_directive = ".. button-link:: somewhere\n" "   :link: 01_alpha\n" "   :link-type: doc\n"
    assert _card_links(other_directive) == []

    body_mention = (
        "   .. grid-item-card:: Alpha\n"
        "      :link-type: doc\n"
        "\n"
        "      Body text mentioning :link: 01_alpha and :link-type: doc.\n"
    )
    assert _card_links(body_mention) == []


def test_parser_ignores_qualnames_outside_autosummary():
    """An indented diff_diff.* line outside autosummary is not an entry."""
    rst = (
        ".. code-block:: text\n"
        "\n"
        "   diff_diff.NotARealEntry\n"
        "\n"
        ".. autosummary::\n"
        "   :toctree: _autosummary\n"
        "\n"
        "   diff_diff.TROP\n"
    )
    assert _autosummary_entries(rst) == ["TROP"]


def test_resolve_identifies_reexport_aliases():
    """Submodule path and package re-export resolve to the same object."""
    assert _resolve("bacon.BaconDecompositionResults") is _resolve("BaconDecompositionResults")
    assert _resolve("TROP") is not _resolve("BaconDecomposition")


def test_estimator_table_counts_flags_duplicates_and_bare_rows():
    """Duplicated rows are counted, and markup-less rows fail loudly."""
    dup = (
        "   * - Estimator\n"
        "     - Description\n"
        "   * - :class:`~diff_diff.TROP`\n"
        "     - Once.\n"
        "   * - :class:`~diff_diff.TROP`\n"
        "     - Twice.\n"
    )
    assert _estimator_table_counts(dup)["TROP"] == 2

    with pytest.raises(AssertionError, match="exactly one"):
        _estimator_table_counts("   * - TROP without markup\n     - Text.\n")


def test_no_index_parser_handles_indent_and_trailing_whitespace():
    """Nested and whitespace-suffixed autoclass directives are still seen."""
    nested = (
        ".. some-container::\n" "\n" "   .. autoclass:: diff_diff.Example   \n" "      :no-index:\n"
    )
    assert _no_index_autoclasses(nested) == ["Example"]
    assert _no_index_autoclasses(".. autoclass:: diff_diff.Other\n   :members:\n") == []


def test_parser_counts_preamble_duplicates_globally():
    """A duplicate registration above the first H2 is still counted."""
    rst = (
        ".. toctree::\n"
        "   :hidden:\n"
        "\n"
        "   Alpha <01_alpha>\n"
        "\n"
        "Group A\n"
        "-------\n"
        "\n"
        ".. grid:: 1 2 2 3\n"
        "\n"
        "   .. grid-item-card:: Alpha\n"
        "      :link: 01_alpha\n"
        "      :link-type: doc\n"
        "\n"
        "      Text.\n"
        "\n"
        ".. toctree::\n"
        "   :hidden:\n"
        "\n"
        "   Alpha <01_alpha>\n"
    )
    global_counts = Counter(_docname(e) for block in _toctree_blocks(rst) for e in block)
    assert global_counts["01_alpha"] == 2  # the guard rejects != 1


def test_no_index_scan_recurses_into_subdirectories(tmp_path):
    """Nested pages (even nested index.rst) are inspected; only the
    top-level index.rst and _autosummary stubs are excluded."""
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.rst").write_text(".. autoclass:: diff_diff.Example\n   :no-index:\n")
    (nested / "index.rst").write_text(".. autoclass:: diff_diff.NestedIdx\n   :no-index:\n")
    stubs = tmp_path / "_autosummary"
    stubs.mkdir()
    (stubs / "diff_diff.Ignored.rst").write_text(
        ".. autoclass:: diff_diff.Ignored\n   :no-index:\n"
    )
    (tmp_path / "index.rst").write_text(".. autoclass:: diff_diff.AlsoIgnored\n   :no-index:\n")
    assert [(p.name, q) for p, q in _scan_no_index(tmp_path)] == [
        ("deep.rst", "Example"),
        ("index.rst", "NestedIdx"),
    ]


def test_autosummary_without_toctree_is_not_canonical():
    """An autosummary block lacking :toctree: generates no stub pages."""
    rst = ".. autosummary::\n   :nosignatures:\n\n   diff_diff.TROP\n"
    assert _autosummary_entries(rst) == []


def test_identity_parity_distinguishes_same_named_classes():
    """Alias paths compare equal; distinct classes never do."""
    assert id(_resolve("trop.TROP")) == id(_resolve("TROP"))
    assert id(_resolve("TROP")) != id(_resolve("BaconDecomposition"))


def test_estimator_table_handles_multiline_and_empty_first_cells():
    """Continuation-line cells are captured; empty cells fail loudly."""
    multiline = "   * -\n" "       :class:`~diff_diff.TROP`\n" "     - Written across two lines.\n"
    assert _estimator_table_counts(multiline)["TROP"] == 1

    with pytest.raises(AssertionError, match="exactly one"):
        _estimator_table_counts("   * -\n     - Empty first cell.\n")


def test_committed_stub_autoclasses_are_not_no_indexed():
    """No committed _autosummary CLASS stub may carry ``:no-index:``.

    Belt-and-suspenders for the canonical-target invariant: stubs are
    regenerated from docs/_templates/autosummary/class.rst at build time
    (the template never emits :no-index:), so a hand-edited stub self-heals
    in any real build - but the committed tree should never even
    transiently encode a dead canonical target. Function stubs are exempt:
    their template intentionally uses :no-index: (functions index on their
    module pages).
    """
    offenders = []
    for path in sorted((DOCS / "api" / "_autosummary").glob("*.rst")):
        for name in _stub_no_index_autoclasses(path.read_text()):
            offenders.append(f"{path.name}: {name}")
    assert not offenders, (
        "committed autosummary class stubs carrying :no-index: (suppresses "
        "the canonical link target; regenerate with a docs build instead of "
        "hand-editing):\n" + "\n".join(offenders)
    )


def test_stub_parser_flags_no_index_classes_but_not_functions():
    """The stub scan catches no-indexed autoclass, ignores autofunction."""
    bad_class = "diff\\_diff.Example\n" "==================\n\n" ".. currentmodule:: diff_diff\n\n"
    assert _stub_no_index_autoclasses(
        bad_class + ".. autoclass:: Example\n   :no-members:\n   :no-index:\n"
    ) == ["Example"]
    assert (
        _stub_no_index_autoclasses(bad_class + ".. autofunction:: example\n   :no-index:\n") == []
    )
    assert _stub_no_index_autoclasses(bad_class + ".. autoclass:: Example\n   :no-members:\n") == []
