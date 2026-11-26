"""Item 9 of the 4.0 cut - the estimators / diagnostics roster split (ledger row M-090).

This file is the ``test_ref`` for ledger row M-090 (spec ``docs/v4-design.md``
section 3.5 and the item-9 line of section 9). It pins the 4.0 public-surface
claim that the library ships ESTIMATORS plus a DIAGNOSTIC FAMILY, on every
catalogue that used to file ``BaconDecomposition`` as an estimator:

1. ``docs/api/index.rst`` - the Estimators autosummary contains the 23
   spec-3.1 estimator classes plus experimental ChainedDiD; ``BaconDecomposition``, ``RDPlot`` and
   ``RDDensityTest`` are catalogued under Diagnostics; the ``bacon`` module
   page sits in the Diagnostics & Inference toctree; no stub is referenced
   from two ``:toctree:`` autosummary blocks (the ``-W`` build would warn).
2. ``docs/index.rst`` - the Supported Estimators table carries no Bacon row,
   and the Diagnostics & Sensitivity table lists exactly the class-backed
   entry points of the family.
3. ``README.md`` and ``llms.txt`` - Bacon lives in the Diagnostics section,
   and the ``## Estimators`` catalogue has one bullet per API estimator.
4. ``llms-full.txt`` - the four diagnostic ``###`` blocks sit under
   ``## Diagnostics``, and the alias line no longer files ``Bacon`` as an
   estimator alias.
5. ``docs/methodology/REGISTRY.md`` - the archive framing for the classes
   removed at 4.0 (MultiPeriodDiD, StaggeredTripleDifference, the QDiD class
   spelling) is in place and the "lands with [M-090]" placeholders are gone.
6. The ledger row itself.

Import discipline: numpy/pandas/scipy/pytest/diff_diff only - the CI test legs
do not install pyyaml, so the ledger is read through ``tests.test_v4_matrix``
(imported INSIDE the ledger test), and the RST/Markdown surfaces are parsed with
``re``. ``docs/doc-deps.yaml`` is deliberately not parsed here; its integrity is
``tests/test_doc_deps_integrity.py``'s job.
"""

import re
from collections import Counter

import pytest

from diff_diff import get_llm_guide
from tests.test_docs_ia import (
    DOCS,
    REPO_ROOT,
    _autosummary_entries,
    _docname,
    _estimator_table_counts,
    _resolve,
    _section_body,
    _toctree_blocks,
)

# Isolated-install CI (Python Tests workflow) runs from a wheel install with
# no docs/ tree; the repo-surface pins only make sense against a checkout.
pytestmark = pytest.mark.skipif(
    not (DOCS / "index.rst").exists(),
    reason="docs tree not present (isolated-install CI)",
)

THIS_TEST_REF = "tests/test_v4_diagnostic_family_docs.py"

# The 23 cut estimators of spec section 3.1, plus experimental ChainedDiD.
ESTIMATOR_ROSTER = {
    "DifferenceInDifferences",
    "TwoWayFixedEffects",
    "SyntheticDiD",
    "CallawaySantAnna",
    "ChainedDiD",
    "ChaisemartinDHaultfoeuille",
    "SunAbraham",
    "ImputationDiD",
    "StackedDiD",
    "TripleDifference",
    "TROP",
    "SyntheticControl",
    "ContinuousDiD",
    "HeterogeneousAdoptionDiD",
    "EfficientDiD",
    "TwoStageDiD",
    "SpilloverDiD",
    "WooldridgeDiD",
    "LPDiD",
    "ChangesInChanges",
    "LWDiD",
    "DMLDiD",
    "DurationDiD",
    "RegressionDiscontinuity",
}

# Class-backed entry points of the diagnostic family (spec section 3.5) that
# the docs homepage catalogues. Function-shaped members (parallel-trends,
# placebo, TWFE weights, the HAD pretest workflow) are docs-family rows too but
# are not classes, so they are not part of this identity pin.
DIAGNOSTIC_ENTRY_POINTS = {
    "BaconDecomposition",
    "RDPlot",
    "RDDensityTest",
    "HonestDiD",
    "PreTrendsPower",
    "PowerAnalysis",
    "DiagnosticReport",
}

# The llms-full.txt ``###`` blocks that moved out of ``## Estimators``.
FULL_GUIDE_DIAGNOSTIC_BLOCKS = (
    "### BaconDecomposition",
    "### TWFE Weight Diagnostics",
    "### RDPlot",
    "### RDDensityTest",
)

API_INDEX = DOCS / "api" / "index.rst"
HOMEPAGE = DOCS / "index.rst"
README = REPO_ROOT / "README.md"
REGISTRY = DOCS / "methodology" / "REGISTRY.md"


# ---------------------------------------------------------------------------
# Helpers (the test_docs_ia ones do not fit these shapes)
# ---------------------------------------------------------------------------


def _h3_body(text, title):
    """Body of a TILDE-underlined RST H3, from its header to the next header.

    ``_section_body`` matches the FIRST header of that title at any level, so
    for ``docs/api/index.rst`` it returns the H2 "Estimators" autosummary
    section, never the ``~~~`` "Estimators" heading under Module Documentation.
    """
    pattern = rf"^{re.escape(title)}\n~+\n(.*?)(?=^\S[^\n]*\n[-=~]+\n|\Z)"
    match = re.search(pattern, text, re.M | re.S)
    assert match is not None, f"H3 section {title!r} not found"
    return match.group(1)


def _first_cells(table_body):
    """First-column cells of a list-table body (the ``* - `` lines)."""
    return [m.group(1).strip() for m in re.finditer(r"^[ \t]+\* -[ \t]*(.*)$", table_body, re.M)]


def _class_targets(table_body):
    """``:class:`` targets of a list-table whose rows may also be ``:func:`` rows.

    ``_estimator_table_counts`` demands exactly one ``:class:`` per cell and
    skips only a literal ``Estimator`` header, so it rejects the homepage
    Diagnostics table (``Diagnostic`` header, ``:func:`` rows). Every first
    cell here must be the header, a ``:class:`` cell or a ``:func:`` cell.
    """
    targets = Counter()
    for cell in _first_cells(table_body):
        if cell == "Diagnostic":
            continue
        classes = re.findall(r":class:`~diff_diff\.(\w+)`", cell)
        funcs = re.findall(r":func:`~diff_diff\.(\w+)`", cell)
        assert classes or funcs, f"diagnostics table row {cell!r} carries no :class:/:func: target"
        if classes:
            assert len(classes) == 1, f"diagnostics table row {cell!r} must name one class"
            targets[classes[0]] += 1
    return targets


def _md_section(text, heading):
    """Body of a Markdown section up to the next heading of the same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.rstrip() == heading]
    assert len(starts) == 1, f"heading {heading!r} found {len(starts)} times"
    start = starts[0]
    end = start + 1
    in_fence = False
    while end < len(lines):
        line = lines[end]
        if line.startswith("```"):
            # A ``# comment`` inside a fenced code block is not a heading.
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            if 0 < hashes <= level and line[hashes : hashes + 1] == " ":
                break
        end += 1
    return "\n".join(lines[start + 1 : end])


def _identity_counter(qualname_tails):
    counter = Counter()
    for tail in qualname_tails:
        counter[id(_resolve(tail))] += 1
    return counter


def _ids(names):
    return {id(_resolve(name)) for name in names}


# ---------------------------------------------------------------------------
# 1. API reference
# ---------------------------------------------------------------------------


def test_api_estimators_autosummary_matches_the_current_roster():
    body = _section_body(API_INDEX.read_text(), "Estimators")
    entries = _autosummary_entries(body)
    counts = _identity_counter(entries)
    assert max(counts.values()) == 1, "duplicate entry in the Estimators autosummary"
    assert set(counts) == _ids(
        ESTIMATOR_ROSTER
    ), f"Estimators autosummary drift: {sorted(entries)} vs the current estimator roster"
    assert not (set(counts) & _ids(DIAGNOSTIC_ENTRY_POINTS))
    assert len(entries) == len(ESTIMATOR_ROSTER)


def test_api_diagnostics_autosummary_catalogues_bacon_rdplot_and_density_test():
    text = API_INDEX.read_text()
    diagnostics = set(_autosummary_entries(_section_body(text, "Diagnostics")))
    assert {"BaconDecomposition", "RDPlot", "RDDensityTest"} <= diagnostics
    visualization = set(_autosummary_entries(_section_body(text, "Visualization")))
    assert "RDPlot" not in visualization
    assert "plot_bacon" in visualization


def test_api_index_references_no_stub_from_two_toctree_autosummaries():
    """A stub listed in two ``:toctree:`` blocks is a duplicate toctree reference."""
    all_entries = _autosummary_entries(API_INDEX.read_text())
    dupes = sorted(name for name, n in Counter(all_entries).items() if n > 1)
    assert dupes == [], f"autosummary entries listed twice: {dupes}"


def test_api_module_toctree_files_bacon_under_diagnostics_and_inference():
    text = API_INDEX.read_text()
    estimators_pages = [
        _docname(e) for block in _toctree_blocks(_h3_body(text, "Estimators")) for e in block
    ]
    diagnostics_pages = [
        _docname(e)
        for block in _toctree_blocks(_h3_body(text, "Diagnostics & Inference"))
        for e in block
    ]
    assert "bacon" not in estimators_pages
    assert "bacon" in diagnostics_pages
    assert "regression_discontinuity" in estimators_pages


def test_api_estimators_intro_no_longer_names_the_bacon_diagnostic():
    body = _section_body(API_INDEX.read_text(), "Estimators")
    intro = body.split(".. autosummary::")[0]
    assert "Goodman-Bacon decomposition diagnostic:" not in intro
    assert "Diagnostics" in intro


# ---------------------------------------------------------------------------
# 2. Docs homepage
# ---------------------------------------------------------------------------


def test_homepage_estimator_table_has_no_bacon_row():
    counts = _estimator_table_counts(_section_body(HOMEPAGE.read_text(), "Supported Estimators"))
    assert "BaconDecomposition" not in counts
    assert sum(counts.values()) == len(ESTIMATOR_ROSTER)


def test_homepage_diagnostics_table_lists_the_family_entry_points():
    text = HOMEPAGE.read_text()
    body = _section_body(text, "Diagnostics & Sensitivity")
    targets = _class_targets(body)
    assert max(targets.values()) == 1, "duplicate class row in the Diagnostics & Sensitivity table"
    assert set(targets) == DIAGNOSTIC_ENTRY_POINTS, sorted(targets)
    api_text = API_INDEX.read_text()
    all_api = _ids(_autosummary_entries(api_text))
    estimators = _ids(_autosummary_entries(_section_body(api_text, "Estimators")))
    for name in DIAGNOSTIC_ENTRY_POINTS:
        obj_id = id(_resolve(name))
        assert obj_id in all_api, f"{name} has no canonical autosummary entry"
        assert obj_id not in estimators, f"{name} is still in the Estimators autosummary"
    # Function-shaped family members are listed as :func: rows.
    for func in (
        "check_parallel_trends",
        "run_placebo_test",
        "attgt_weights",
        "did_had_pretest_workflow",
    ):
        assert f":func:`~diff_diff.{func}`" in body, f"missing :func: row for {func}"


def test_homepage_key_features_estimator_bullet_no_longer_mentions_bacon():
    lines = [
        line for line in HOMEPAGE.read_text().split("\n") if line.startswith("- **20+ Estimators**")
    ]
    assert len(lines) == 1
    assert "Bacon" not in lines[0]
    assert lines[0].rstrip().endswith("Regression Discontinuity")


# ---------------------------------------------------------------------------
# 3. README + llms.txt
# ---------------------------------------------------------------------------


def _bullet_count(section_body):
    return len([line for line in section_body.split("\n") if line.startswith("- [")])


def test_readme_files_bacon_under_diagnostics_not_estimators():
    text = README.read_text()
    estimators = _md_section(text, "## Estimators")
    diagnostics = _md_section(text, "## Diagnostics & Sensitivity")
    assert "BaconDecomposition" not in estimators
    assert "api/bacon.html" not in estimators
    assert "api/bacon.html" in diagnostics
    assert "BaconDecomposition" in diagnostics
    api_count = len(_autosummary_entries(_section_body(API_INDEX.read_text(), "Estimators")))
    assert _bullet_count(estimators) == api_count == len(ESTIMATOR_ROSTER)


def test_llms_txt_files_bacon_under_diagnostics_not_estimators():
    text = get_llm_guide("concise")
    estimators = _md_section(text, "## Estimators")
    diagnostics = _md_section(text, "## Diagnostics and Sensitivity Analysis")
    assert "BaconDecomposition" not in estimators
    assert "api/bacon.html" not in estimators
    assert "api/bacon.html" in diagnostics
    assert "BaconDecomposition" in diagnostics
    assert _bullet_count(estimators) == len(ESTIMATOR_ROSTER)


# ---------------------------------------------------------------------------
# 4. llms-full.txt
# ---------------------------------------------------------------------------


def test_llms_full_diagnostic_blocks_live_under_diagnostics():
    text = get_llm_guide("full")
    estimators = _md_section(text, "## Estimators")
    diagnostics = _md_section(text, "## Diagnostics")
    for heading in FULL_GUIDE_DIAGNOSTIC_BLOCKS:
        assert heading not in estimators, f"{heading!r} is still filed under ## Estimators"
        assert heading in diagnostics, f"{heading!r} is missing from ## Diagnostics"
    # The estimator blocks that neighboured the moved ones stayed put.
    assert "### WooldridgeDiD" in estimators
    assert "### TROP" in estimators
    assert "### StackedDiD" in estimators


def test_llms_full_alias_line_does_not_file_bacon_as_an_estimator_alias():
    lines = [
        line
        for line in get_llm_guide("full").split("\n")
        if line.startswith("- **Estimator aliases**")
    ]
    assert len(lines) == 1
    match = re.search(r"\((e\.g\., [^)]*)\)", lines[0])
    assert match is not None, lines[0]
    alias_list = match.group(1)
    assert "`Bacon`" not in alias_list
    assert "`BaconDecomposition` keeps its `Bacon` alias" in lines[0]


# ---------------------------------------------------------------------------
# 5. REGISTRY archive framing
# ---------------------------------------------------------------------------

ARCHIVED_SECTIONS = (
    "## MultiPeriodDiD",
    "## StaggeredTripleDifference",
    "### QDiD (quantile DiD comparison estimator)",
)


def test_registry_archive_placeholders_are_gone():
    text = REGISTRY.read_text()
    # The old sentence is line-wrapped in both sections, so the match must be
    # whitespace-tolerant (a literal substring check passed before any edit).
    assert re.search(r"lands\s+with\s+\[M-090\]", text) is None
    assert "20-estimator" not in text


@pytest.mark.parametrize("heading", ARCHIVED_SECTIONS)
def test_registry_archived_section_opens_with_an_archive_note(heading):
    body = _md_section(REGISTRY.read_text(), heading)
    first = next(line for line in body.split("\n") if line.strip())
    assert first.startswith("- **Note (archived"), first
    if heading.startswith("## "):
        # The two removed-class sections cite the archive record; the QDiD
        # class-spelling Note keeps its 3.9 text apart from the relabel.
        assert "([M-090], 4.0 cut item 9)" in body


def test_registry_toc_marks_the_archived_sections_and_states_the_convention():
    text = REGISTRY.read_text()
    toc = _md_section(text, "## Table of Contents")
    for anchor in ("(#multiperioddid)", "(#staggeredtripledifference)"):
        line = next(line for line in toc.split("\n") if anchor in line)
        assert "(archived" in line, line
    cic = next(line for line in toc.split("\n") if "(#changesinchanges-cic)" in line)
    assert "QDiD class spelling archived" in cic
    assert "**Archived sections.**" in toc


def test_registry_cross_reference_tables_have_no_standalone_qdid_row():
    text = REGISTRY.read_text()
    assert re.search(r"^\| QDiD \|", text, re.M) is None
    assert text.count('| ChangesInChanges(method="qdid") |') == 2


# ---------------------------------------------------------------------------
# 6. Ledger
# ---------------------------------------------------------------------------


def test_ledger_row_m090_is_done_and_points_here():
    from tests.test_v4_matrix import ROWS as LEDGER_ROWS

    row = {r["id"]: r for r in LEDGER_ROWS}["M-090"]
    assert row["status"] == "done"
    assert row["kind"] == "behavior"
    assert row["test_ref"] == THIS_TEST_REF
    assert "item 9" in row["notes"]
