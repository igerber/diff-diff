"""Static guard for the tutorial conventions.

The conventions are written down in CONTRIBUTING.md ("Tutorial conventions").
This file enforces the mechanical ones over every notebook in
``docs/tutorials/`` by parsing the ``.ipynb`` JSON directly: nothing is
executed, so it is cheap and runs on notebook-only diffs (docs-tests.yml).

Ratchet: ``KNOWN_VIOLATIONS`` lists, per notebook, the rules it violated when
the guard was introduced. The guard fails on any violation that is NOT listed
and ALSO fails when a listed violation has been fixed, so a PR that repairs a
notebook must delete its entries. The dict is meant to end empty; never add to
it to get a new or edited notebook past the guard - fix the notebook.

Rules scan cell SOURCES only. Committed outputs are not scanned: library
output legitimately contains characters and names the rules ban in prose.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = REPO_ROOT / "docs" / "tutorials"
TESTS_DIR = REPO_ROOT / "tests"

if not TUTORIALS_DIR.is_dir():
    # The isolated-install CI jobs copy only tests/ and run without docs/.
    pytest.skip("docs/tutorials not present in this environment", allow_module_level=True)

# --------------------------------------------------------------------------
# Rule ids
# --------------------------------------------------------------------------
EXECUTED = "executed"
ERROR_OUTPUT = "error-output"
REMOVED_NAME = "removed-name"
REMOVED_IDENTIFIER = "removed-identifier"
JARGON = "jargon"
BLANKET_WARNING_FILTER = "blanket-warning-filter"
EM_DASH = "em-dash"
MATPLOTLIB_GUARD = "matplotlib-guard"
DRIFT_TEST = "drift-test"

ALL_RULES = (
    EXECUTED,
    ERROR_OUTPUT,
    REMOVED_NAME,
    REMOVED_IDENTIFIER,
    JARGON,
    BLANKET_WARNING_FILTER,
    EM_DASH,
    MATPLOTLIB_GUARD,
    DRIFT_TEST,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Notebooks exempt from every rule: tutorials that are in flight on other
# branches and were written before this guard existed. An entry is inert
# while its file does not exist. Conform the notebook, then remove it here.
PENDING: FrozenSet[str] = frozenset(
    {
        "34_chained_did",
        "34_did_ovb",
    }
)

# Explicit notebook -> drift-test file. REQUIRED for every notebook whose
# number is shared with another notebook (the NN glob is ambiguous there);
# optional otherwise.
DRIFT_TEST_FILES: Dict[str, str] = {
    "16_survey_did": "test_t16_survey_did_drift.py",
    "16_wooldridge_etwfe": "test_t16_wooldridge_etwfe_drift.py",
    "34_design_diagnosis": "test_t34_design_diagnosis_drift.py",
}

# Names removed at 4.0 that must not appear anywhere in a cell source.
# ``overall_att`` is matched as an exact name, because the bootstrap containers
# legitimately keep ``overall_att_se`` / ``overall_att_ci`` / ...
_REMOVED_NAME_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("overall_att", re.compile(r"(?<!\w)overall_att(?!\w)")),
    ("MultiPeriodDiD", re.compile(r"(?<!\w)MultiPeriodDiD(?!\w)")),
    ("StaggeredTripleDifference", re.compile(r"(?<!\w)StaggeredTripleDifference(?!\w)")),
)

# Removed classes and aliases, matched as Python identifier tokens in code
# cells only (prose may say "QDiD" or "Stacked": those are method names).
_REMOVED_IDENTIFIERS: FrozenSet[str] = frozenset(
    {"QDiD", "QDiDResults", "EventStudy", "SDDD", "CDiD", "Stacked", "Gardner"}
)

# Internal vocabulary that has no meaning to a reader of the docs.
_JARGON_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("deprecation-ledger id", re.compile(r"(?<![\w-])M-\d{3}(?!\d)")),
    ("PR number", re.compile(r"\bPR #\d+")),
    ("phase name", re.compile(r"\bPhase \d")),
    ("wave name", re.compile(r"\bWave [A-Z]\b")),
    ("scratch path", re.compile(r"_scratch/")),
    ("file:line reference", re.compile(r"\.py:\d+")),
)

# Fallback only, for a code cell that does not parse: any ignore filter.
_IGNORE_FILTER_TEXT = re.compile(
    r"\b(?:filterwarnings|simplefilter)\(\s*(?:action\s*=\s*)?[\"']ignore[\"']"
)

# Rules that may never be allowlisted: a violation is a broken notebook, not
# a style debt to pay down later.
NEVER_ALLOWED: FrozenSet[str] = frozenset({ERROR_OUTPUT})

_EM_DASH = "—"

# Rules each notebook violated when the guard was introduced. See the module
# docstring: entries are deleted as notebooks are repaired, never added.
KNOWN_VIOLATIONS: Dict[str, FrozenSet[str]] = {
    "01_basic_did": frozenset({EXECUTED, EM_DASH, DRIFT_TEST}),
    "02_staggered_did": frozenset({REMOVED_NAME, EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "03_synthetic_did": frozenset({EXECUTED, EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "04_parallel_trends": frozenset({EXECUTED, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "05_honest_did": frozenset({EXECUTED, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "06_power_analysis": frozenset({EXECUTED, EM_DASH, DRIFT_TEST}),
    "07_pretrends_power": frozenset({EXECUTED, EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "08_triple_diff": frozenset({REMOVED_NAME, DRIFT_TEST}),
    "09_real_world_examples": frozenset({MATPLOTLIB_GUARD, DRIFT_TEST}),
    "10_trop": frozenset({EXECUTED, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "11_imputation_did": frozenset(
        {EXECUTED, JARGON, BLANKET_WARNING_FILTER, EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}
    ),
    "12_two_stage_did": frozenset(
        {EXECUTED, JARGON, BLANKET_WARNING_FILTER, EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}
    ),
    "13_stacked_did": frozenset({EXECUTED, JARGON, EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "14_continuous_did": frozenset(
        {EXECUTED, REMOVED_NAME, JARGON, EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}
    ),
    "15_efficient_did": frozenset({EXECUTED, JARGON, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "16_survey_did": frozenset({REMOVED_NAME, JARGON, EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "16_wooldridge_etwfe": frozenset({EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "17_brand_awareness_survey": frozenset({EM_DASH, MATPLOTLIB_GUARD, DRIFT_TEST}),
    "18_geo_experiments": frozenset({EM_DASH, DRIFT_TEST}),
    "19_dcdh_marketing_pulse": frozenset({EXECUTED, JARGON}),
    "20_had_brand_campaign": frozenset({EXECUTED, MATPLOTLIB_GUARD}),
    "21_had_pretest_workflow": frozenset({JARGON, EM_DASH}),
    "22_had_survey_design": frozenset({EXECUTED, JARGON, EM_DASH, MATPLOTLIB_GUARD}),
    "23_spillover_tva": frozenset({EXECUTED, JARGON, EM_DASH}),
    "24_staggered_vs_collapsed_power": frozenset({REMOVED_NAME, BLANKET_WARNING_FILTER, EM_DASH}),
    "25_synthetic_control_policy": frozenset({EM_DASH}),
    "26_composition_drift_calibration": frozenset({MATPLOTLIB_GUARD}),
    "32_dml_did": frozenset({REMOVED_NAME, EM_DASH}),
    "33_bad_controls": frozenset({BLANKET_WARNING_FILTER}),
}


# --------------------------------------------------------------------------
# Notebook access
# --------------------------------------------------------------------------
def _stems() -> List[str]:
    return sorted(p.stem for p in TUTORIALS_DIR.glob("*.ipynb"))


def _load(stem: str) -> dict:
    return json.loads((TUTORIALS_DIR / f"{stem}.ipynb").read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _number(stem: str) -> str:
    return stem.split("_", 1)[0]


# --------------------------------------------------------------------------
# Rule implementations. Each returns a list of human-readable findings
# (empty = the rule holds).
# --------------------------------------------------------------------------
def check_executed(nb: dict) -> List[str]:
    """Code cells were executed top to bottom in one pass.

    Counts must be exactly 1..n over the non-empty code cells in cell order
    (a cell that produced no output is fine).
    """
    findings: List[str] = []
    counts = [
        cell.get("execution_count")
        for cell in nb["cells"]
        if cell["cell_type"] == "code" and _source(cell).strip()
    ]
    if counts != list(range(1, len(counts) + 1)):
        n_missing = sum(c is None for c in counts)
        if n_missing == len(counts):
            findings.append("no code cell has been executed (no committed outputs)")
        elif n_missing:
            findings.append(f"{n_missing} of {len(counts)} code cells have no execution_count")
        else:
            findings.append(f"execution counts are not 1..{len(counts)} in cell order")
    return findings


def check_error_output(nb: dict) -> List[str]:
    """No committed traceback. Separate from ``check_executed`` so that a
    notebook allowlisted for missing outputs still fails on a new error."""
    return [
        f"cell {idx}: error output ({out.get('ename')})"
        for idx, cell in enumerate(nb["cells"])
        if cell["cell_type"] == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]


def check_removed_names(nb: dict) -> List[str]:
    findings = []
    for idx, cell in enumerate(nb["cells"]):
        text = _source(cell)
        for label, pattern in _REMOVED_NAME_PATTERNS:
            if pattern.search(text):
                findings.append(f"cell {idx}: removed 4.0 name {label!r}")
    return findings


def _strip_magics(code: str) -> str:
    """Drop IPython magics / shell escapes so the cell is plain Python."""
    lines = [ln for ln in code.splitlines() if not ln.lstrip().startswith(("%", "!"))]
    return "\n".join(lines) + "\n"


def _identifier_tokens(code: str) -> FrozenSet[str]:
    """Python NAME tokens in a code cell (strings and comments excluded).

    IPython magics / shell escapes are dropped first. If the cell still does
    not tokenize, fall back to a word scan of the lines with string literals
    crudely stripped - conservative, so a broken cell cannot hide a name.
    """
    cleaned = _strip_magics(code)
    try:
        return frozenset(
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(cleaned).readline)
            if tok.type == tokenize.NAME
        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        no_strings = re.sub(r"(\"[^\"\n]*\"|'[^'\n]*')", "", cleaned)
        no_comments = re.sub(r"#.*", "", no_strings)
        return frozenset(re.findall(r"[A-Za-z_]\w*", no_comments))


def check_removed_identifiers(nb: dict) -> List[str]:
    findings = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        hits = sorted(_identifier_tokens(_source(cell)) & _REMOVED_IDENTIFIERS)
        if hits:
            findings.append(f"cell {idx}: removed class/alias used in code: {hits}")
    return findings


def check_jargon(nb: dict) -> List[str]:
    findings = []
    for idx, cell in enumerate(nb["cells"]):
        text = _source(cell)
        for label, pattern in _JARGON_PATTERNS:
            m = pattern.search(text)
            if m:
                findings.append(f"cell {idx}: {label} ({m.group(0)!r})")
    return findings


def _is_str(node, value=None) -> bool:
    ok = isinstance(node, ast.Constant) and isinstance(node.value, str)
    return ok and (value is None or node.value == value)


# Literal ``message=`` / ``module=`` regexes that match every warning. The
# warnings module applies both anchored at the START of the text
# (``re.match``), so each of these is equivalent to passing no pattern.
_BLANKET_PATTERNS: FrozenSet[str] = frozenset({"", ".*", "^", "^.*", ".*$", "^.*$", ".*?", "(.*)"})


def _narrows(node) -> bool:
    """Does this ``message=`` / ``module=`` argument restrict the filter?

    Deliberately conservative: only the known blanket literals in
    ``_BLANKET_PATTERNS`` count as "restricts nothing". Whether an arbitrary
    regex matches every possible warning cannot be established statically, so
    any other pattern - and any non-literal argument - is taken as narrowing.
    The rule exists to stop "ignore everything" filters, not to grade regexes.
    """
    return not (_is_str(node) and node.value in _BLANKET_PATTERNS)


def _unqualified_ignore_calls(code: str) -> List[str]:
    """``filterwarnings`` / ``simplefilter`` calls that ignore EVERY warning.

    A call is qualified - and allowed - only if it narrows what it ignores:
    a ``message`` / ``module`` pattern that does not match everything
    (filterwarnings; see ``_narrows``), or a ``category`` other than the base
    ``Warning``. Extra arguments that do not narrow it
    (``append=``, ``lineno=``) do not count, and neither does spelling the
    action as a keyword.
    """
    try:
        tree = ast.parse(_strip_magics(code))
    except SyntaxError:
        # Cannot inspect arguments: be conservative and flag any ignore filter.
        return [m.group(0) for m in _IGNORE_FILTER_TEXT.finditer(code)]
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in ("filterwarnings", "simplefilter"):
            continue
        # signatures: filterwarnings(action, message, category, module, lineno, append)
        #             simplefilter(action, category, lineno, append)
        order = (
            ("action", "message", "category", "module")
            if name == "filterwarnings"
            else ("action", "category")
        )
        args = dict(zip(order, node.args))
        args.update({kw.arg: kw.value for kw in node.keywords if kw.arg in order})
        if not _is_str(args.get("action"), "ignore"):
            continue
        narrowed = False
        for key in ("message", "module"):
            value = args.get(key)
            if value is not None and _narrows(value):
                narrowed = True
        category = args.get("category")
        if category is not None:
            base = category.attr if isinstance(category, ast.Attribute) else None
            base = category.id if isinstance(category, ast.Name) else base
            if base != "Warning":
                narrowed = True
        if not narrowed:
            hits.append(ast.unparse(node) if hasattr(ast, "unparse") else f"{name}('ignore', ...)")
    return hits


def check_blanket_warning_filter(nb: dict) -> List[str]:
    findings = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for call in _unqualified_ignore_calls(_source(cell)):
            findings.append(f"cell {idx}: unqualified warning suppression ({call})")
    return findings


def check_em_dash(nb: dict) -> List[str]:
    cells = [idx for idx, cell in enumerate(nb["cells"]) if _EM_DASH in _source(cell)]
    return [f"em dash in cell sources: cells {cells}"] if cells else []


def check_matplotlib_guard(nb: dict) -> List[str]:
    cells = [idx for idx, cell in enumerate(nb["cells"]) if "HAS_MATPLOTLIB" in _source(cell)]
    return [f"HAS_MATPLOTLIB guard: cells {cells}"] if cells else []


def check_drift_test(stem: str, all_stems: List[str]) -> List[str]:
    """The notebook has exactly one drift test.

    ``tests/test_t<NN>_*_drift.py`` must match exactly one file, unless the
    notebook has an explicit ``DRIFT_TEST_FILES`` entry - which is mandatory
    when another notebook shares its number.
    """
    explicit = DRIFT_TEST_FILES.get(stem)
    if explicit is not None:
        if (TESTS_DIR / explicit).is_file():
            return []
        return [f"drift test tests/{explicit} does not exist"]
    number = _number(stem)
    if sum(_number(s) == number for s in all_stems) > 1:
        return [f"number {number} is shared with another notebook: add a DRIFT_TEST_FILES entry"]
    matches = sorted(p.name for p in TESTS_DIR.glob(f"test_t{number}_*_drift.py"))
    if len(matches) == 1:
        return []
    if not matches:
        return [f"no drift test matches tests/test_t{number}_*_drift.py"]
    return [f"several drift tests match tests/test_t{number}_*_drift.py: {matches}"]


def violations(stem: str, all_stems: List[str]) -> Dict[str, List[str]]:
    """Every rule the notebook violates, with its findings."""
    nb = _load(stem)
    results = {
        EXECUTED: check_executed(nb),
        ERROR_OUTPUT: check_error_output(nb),
        REMOVED_NAME: check_removed_names(nb),
        REMOVED_IDENTIFIER: check_removed_identifiers(nb),
        JARGON: check_jargon(nb),
        BLANKET_WARNING_FILTER: check_blanket_warning_filter(nb),
        EM_DASH: check_em_dash(nb),
        MATPLOTLIB_GUARD: check_matplotlib_guard(nb),
        DRIFT_TEST: check_drift_test(stem, all_stems),
    }
    return {rule: found for rule, found in results.items() if found}


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------
_ALL_STEMS = _stems()
_GUARDED_STEMS = [s for s in _ALL_STEMS if s not in PENDING]


@pytest.mark.parametrize("stem", _GUARDED_STEMS)
def test_notebook_follows_conventions(stem):
    found = violations(stem, _ALL_STEMS)
    allowed = KNOWN_VIOLATIONS.get(stem, frozenset())

    new = {rule: detail for rule, detail in found.items() if rule not in allowed}
    assert not new, (
        f"docs/tutorials/{stem}.ipynb breaks the tutorial conventions "
        f"(CONTRIBUTING.md, 'Tutorial conventions'). Fix the notebook; do not "
        f"add to KNOWN_VIOLATIONS.\n"
        + "\n".join(f"  [{rule}] {line}" for rule, lines in new.items() for line in lines)
    )

    fixed = sorted(allowed - set(found))
    assert not fixed, (
        f"docs/tutorials/{stem}.ipynb no longer violates {fixed}: delete those "
        f"entries from KNOWN_VIOLATIONS in tests/test_tutorial_conventions.py "
        f"so the fix cannot regress."
    )


def test_configuration_is_consistent():
    stems = set(_ALL_STEMS)
    unknown_rules = {
        stem: sorted(set(rules) - set(ALL_RULES))
        for stem, rules in KNOWN_VIOLATIONS.items()
        if set(rules) - set(ALL_RULES)
    }
    assert not unknown_rules, f"KNOWN_VIOLATIONS names unknown rules: {unknown_rules}"
    forbidden = {
        stem: sorted(set(rules) & NEVER_ALLOWED)
        for stem, rules in KNOWN_VIOLATIONS.items()
        if set(rules) & NEVER_ALLOWED
    }
    assert not forbidden, f"these rules can never be allowlisted: {forbidden}"
    assert not [
        s for s, rules in KNOWN_VIOLATIONS.items() if not rules
    ], "KNOWN_VIOLATIONS has an empty entry: delete the notebook's line instead"
    stale = sorted(set(KNOWN_VIOLATIONS) - stems)
    assert not stale, f"KNOWN_VIOLATIONS lists notebooks that do not exist: {stale}"
    assert not (
        PENDING & set(KNOWN_VIOLATIONS)
    ), "a notebook is in both PENDING and KNOWN_VIOLATIONS"
    stale_map = sorted(set(DRIFT_TEST_FILES) - stems - PENDING)
    assert not stale_map, f"DRIFT_TEST_FILES lists notebooks that do not exist: {stale_map}"


# --------------------------------------------------------------------------
# The rules themselves: each fires on a violating fixture and stays quiet on
# a conforming one (so an edit cannot silently turn a rule into a no-op).
# --------------------------------------------------------------------------
def _nb(*cells) -> dict:
    return {"cells": list(cells)}


def _code(src, count=None, outputs=()):
    return {
        "cell_type": "code",
        "source": src,
        "execution_count": count,
        "outputs": list(outputs),
    }


def _md(src):
    return {"cell_type": "markdown", "source": src}


class TestRules:
    def test_executed_accepts_sequential_counts_and_empty_outputs(self):
        nb = _nb(_md("# t"), _code("import x", 1), _code("", None), _code("print(1)", 2))
        assert check_executed(nb) == []

    @pytest.mark.parametrize(
        "cells, needle",
        [
            ([_code("a", None), _code("b", None)], "no code cell has been executed"),
            ([_code("a", 1), _code("b", None)], "1 of 2 code cells"),
            ([_code("a", 2), _code("b", 1)], "not 1..2 in cell order"),
            ([_code("a", 1), _code("b", 3)], "not 1..2 in cell order"),
        ],
    )
    def test_executed_rejects(self, cells, needle):
        found = check_executed(_nb(*cells))
        assert any(needle in f for f in found), found

    def test_error_output_is_its_own_rule(self):
        err = {"output_type": "error", "ename": "ValueError", "evalue": "x"}
        # an executed notebook with a traceback: only the error rule fires
        executed = _nb(_code("boom()", 1, [err]))
        assert check_executed(executed) == []
        assert check_error_output(executed) == ["cell 0: error output (ValueError)"]
        # a notebook allowlisted for missing execution counts still fails on it
        partial = _nb(_code("a", None), _code("boom()", 1, [err]))
        assert check_executed(partial)
        assert check_error_output(partial) == ["cell 1: error output (ValueError)"]
        assert check_error_output(_nb(_code("a", 1, [{"output_type": "stream"}]))) == []
        assert ERROR_OUTPUT in NEVER_ALLOWED

    def test_removed_names(self):
        assert check_removed_names(_nb(_md("see `results.att`"), _code("r.att", 1))) == []
        # live bootstrap-container fields are not the removed attribute
        ok = _nb(_code("b.overall_att_se; b.overall_att_ci; b.overall_se", 1))
        assert check_removed_names(ok) == []
        assert check_removed_names(_nb(_md("reported as `overall_att`.")))
        assert check_removed_names(_nb(_code("# results.overall_att", 1)))
        assert check_removed_names(_nb(_md("the former MultiPeriodDiD class")))
        assert check_removed_names(_nb(_code("StaggeredTripleDifference()", 1)))

    def test_removed_identifiers_only_match_code_identifiers(self):
        quiet = _nb(
            _md("QDiD and Stacked DiD are methods; Gardner (2022)."),
            _code('print("Stacked data shape")  # Gardner two-stage\nStackedDiD()', 1),
            _code("%matplotlib inline\n!pip list\nChangesInChanges(method='qdid')", 2),
        )
        assert check_removed_identifiers(quiet) == []
        assert check_removed_identifiers(_nb(_code("from diff_diff import Stacked", 1)))
        assert check_removed_identifiers(_nb(_code("est = QDiD()", 1)))
        assert check_removed_identifiers(_nb(_code("x: QDiDResults = fit()", 1)))
        # a cell that does not tokenize still cannot hide a removed name
        assert check_removed_identifiers(_nb(_code("est = Gardner(\n", 1)))

    @pytest.mark.parametrize(
        "text",
        [
            "(3.9, row M-021)",
            "shipped in PR #397",
            "the Phase 1 placebo",
            "added in Wave C",
            "see _scratch/dcdh_tutorial/",
            "documented at had.py:1450",
        ],
    )
    def test_jargon_fires(self, text):
        assert check_jargon(_nb(_md(text)))

    @pytest.mark.parametrize(
        "text",
        [
            "the M-estimator and ATT-012 are fine",
            "a two-phase rollout; a wave of adoption",
            "run `python script.py` and read model.py for details",
            "PR is short for parallel trends here",
        ],
    )
    def test_jargon_quiet(self, text):
        assert check_jargon(_nb(_md(text))) == []

    @pytest.mark.parametrize(
        "call",
        [
            "warnings.filterwarnings('ignore')",
            'warnings.simplefilter("ignore")',
            "warnings.filterwarnings('ignore', append=True)",
            "warnings.simplefilter(action='ignore')",
            "warnings.filterwarnings('ignore', category=Warning)",
            "warnings.simplefilter('ignore', Warning)",
            "warnings.filterwarnings('ignore', message='')",
            "warnings.filterwarnings('ignore', message='.*')",
            "warnings.filterwarnings('ignore', module='.*')",
            "warnings.filterwarnings('ignore', '^')",
            "warnings.filterwarnings('ignore', message='^.*$')",
            "warnings.filterwarnings('ignore', message='.*', category=Warning, module='')",
            "filterwarnings('ignore')",
            "with warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n    fit()",
            # a cell that does not parse cannot hide one
            "warnings.simplefilter('ignore')\nest = fit(",
        ],
    )
    def test_blanket_warning_filter_fires(self, call):
        assert check_blanket_warning_filter(_nb(_code(call, 1)))

    @pytest.mark.parametrize(
        "call",
        [
            "warnings.filterwarnings('ignore', category=FutureWarning)",
            "warnings.simplefilter('ignore', DeprecationWarning)",
            "warnings.simplefilter('ignore', category=np.exceptions.RankWarning)",
            'warnings.filterwarnings("ignore", message=".*matmul")',
            "warnings.filterwarnings('ignore', '.*matmul', RuntimeWarning)",
            "warnings.filterwarnings('ignore', module='pymc')",
            "warnings.filterwarnings('ignore', message=r'^$|.*matmul')",
            "warnings.filterwarnings('ignore', module=r'^$|pymc')",
            "warnings.filterwarnings('ignore', message=r'^$|^[a-z]')",
            "warnings.filterwarnings('ignore', message='.*', category=FutureWarning)",
            "warnings.filterwarnings('ignore', message=PATTERN)",
            "warnings.filterwarnings('error')",
            "warnings.simplefilter('always')",
            "%matplotlib inline\nwarnings.filterwarnings('ignore', category=UserWarning)",
        ],
    )
    def test_blanket_warning_filter_quiet(self, call):
        assert check_blanket_warning_filter(_nb(_code(call, 1))) == []

    def test_blanket_warning_filter_ignores_prose(self):
        prose = _nb(_md("never call `warnings.filterwarnings('ignore')` globally"))
        assert check_blanket_warning_filter(prose) == []

    def test_em_dash_and_matplotlib_guard(self):
        assert check_em_dash(_nb(_md("a — b")))
        assert check_em_dash(_nb(_code("# a — b", 1)))
        assert check_em_dash(_nb(_md("a - b, 2019–2021"))) == []  # en dash is fine
        assert check_matplotlib_guard(_nb(_code("if HAS_MATPLOTLIB:\n    plot()", 1)))
        assert check_matplotlib_guard(_nb(_code("import matplotlib.pyplot as plt", 1))) == []

    def test_drift_test_resolution(self):
        # explicit entry
        assert check_drift_test("34_design_diagnosis", ["34_design_diagnosis"]) == []
        # unique number, glob matches exactly one existing file
        assert check_drift_test("19_dcdh_marketing_pulse", ["19_dcdh_marketing_pulse"]) == []
        # shared number without an explicit entry
        shared = check_drift_test("19_dcdh_marketing_pulse", ["19_dcdh_marketing_pulse", "19_x"])
        assert shared and "shared" in shared[0]
        # no drift test at all
        missing = check_drift_test("99_nothing", ["99_nothing"])
        assert missing and "no drift test" in missing[0]


class TestWorkflowPins:
    """Notebook-only diffs must run this guard and the drift tests, and
    neither lane fires on ``docs/**`` by accident: pin the wiring so a
    workflow edit cannot silently de-wire it (the tracking-file guard's
    TestWorkflowPins precedent)."""

    @staticmethod
    def _text(name):
        path = REPO_ROOT / ".github" / "workflows" / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _trigger_block(text, trigger):
        on_block = text[: text.index("\njobs:")]
        start = on_block.index(trigger)
        later = [
            on_block.index(t)
            for t in ("push:", "pull_request:", "schedule:", "workflow_dispatch:")
            if t != trigger and t in on_block and on_block.index(t) > start
        ]
        return on_block[start : min(later)] if later else on_block[start:]

    @pytest.mark.parametrize("trigger", ["push:", "pull_request:"])
    def test_docs_tests_triggers_on_notebooks_and_this_file(self, trigger):
        block = self._trigger_block(self._text("docs-tests.yml"), trigger)
        for entry in ("docs/**", "tests/test_tutorial_conventions.py"):
            assert f"'{entry}'" in block, (
                f"docs-tests.yml {trigger} paths filter is missing {entry!r} - "
                "a diff touching only it would not run the tutorial conventions guard"
            )

    def test_docs_tests_runs_this_guard(self):
        text = self._text("docs-tests.yml")
        jobs = text[text.index("\njobs:") :]
        assert (
            "pytest tests/test_tutorial_conventions.py" in jobs
        ), "docs-tests.yml no longer runs the tutorial conventions guard"

    @pytest.mark.parametrize("trigger", ["push:", "pull_request:"])
    def test_notebooks_workflow_triggers_on_drift_tests(self, trigger):
        block = self._trigger_block(self._text("notebooks.yml"), trigger)
        for entry in ("docs/tutorials/**", "tests/test_t*_drift.py", "tests/_tutorial_drift.py"):
            assert f"'{entry}'" in block, (
                f"notebooks.yml {trigger} paths filter is missing {entry!r} - "
                "a notebook-only or drift-test-only diff would not run the drift tests"
            )

    def test_notebooks_workflow_runs_every_drift_test(self):
        text = self._text("notebooks.yml")
        jobs = text[text.index("\njobs:") :]
        assert (
            "pytest tests/test_t*_drift.py" in jobs
        ), "notebooks.yml no longer runs the tutorial drift tests on notebook diffs"
