"""Naming-completeness guard (spec: docs/v4-design.md sections 8 + 9).

Three duties, each previously enforced only by hand and each missed across
successive review rounds (this module replaced the TODO.md guard row, whose
amended 2026-07-31 spec it carries):

A. PUBLIC-SURFACE SWEEP - every ``diff_diff.__all__`` export's signatures,
   dataclass fields, and property names are swept for the section-8
   contract-rename vocabulary (the ``_PATTERN_TOKENS`` predicate below); every
   hit must be covered by a ``docs/v4-deprecations.yaml`` row, structurally
   exempt (wrapper functions and classes that die wholesale at 4.0), or carry a
   ``SURFACE_ALLOWLIST`` entry with a stated reason.

B. PHASE-TABLE AGREEMENT with the section-9 phase table.
   Direction 1: every non-terminal ledger row's id appears - after citation
   expansion - in the cell of its current ``phase``.
   Direction 2: every id cited in the cell of phase P resolves to a live row
   that is terminal (``done``/``removed`` - exempt both directions) or has P's
   ship-version (phases 2/3/4 -> "3.9", 5 -> "4.0", 6 -> "4.1") among its
   ``introduced_in``/``deprecated_in``/``removed_in`` - or ``decision_due`` for
   ``env-default`` rows (M-008, scheduled solely via ``decision_due: "4.0"``,
   is the named positive fixture).
   Citation grammar is BRACKETED-ONLY, three forms: single tokens
   (``[M-122]``), compound single brackets with ``..`` ranges, comma elements
   and trailing prose (``[M-030, M-032..M-047 old names]``), and
   endpoint-bracket ranges (``[M-132]..[M-134]``). Bare ids in cell prose
   (``M-020's``, ``M-031's``) are NOT citations - M-031, named in the phase-5
   cell's prose while deliberately absent from its roster, is the named
   negative fixture.
   Accepted limitation: phases 2, 3 and 4 all ship 3.9, so a row cited in
   the wrong 3.9 phase is undetectable by the version predicate.

C. CONSUMER COVERAGE (section 8 rule 11) - for each enforceable rename row,
   readers of the old name in ``diff_diff/`` source (plus the packaged
   ``diff_diff/guides/*.txt``) and ``docs/methodology/`` must be named in the
   token family's ``code_refs`` union (rule 11's token-family clause) or carry
   a ``CONSUMER_ALLOWLIST`` entry with a stated reason.
   Matching lanes are tiered by row kind and token ambiguity: AMBIGUOUS tokens
   (legal canonical vocabulary elsewhere - rule 1 makes ``time`` THE calendar
   column) get AST call-site matching PLUS literal-name state reads
   (``getattr``/``hasattr`` - rule 11's exact silent form) on PARAM rows
   (receiver typing is statically unknowable; the 3.9 FutureWarnings net
   direct calls) and attr/quoted lanes on FIELD rows (no warning covers a
   field read - the ``getattr(obj, "old", default)`` silent-degradation
   hazard rule 11 names). Terminal rows KEEP their lanes with a stricter
   predicate (their own historical ``code_refs`` no longer count), so
   enforcement does not switch off at the removal commit.
   FUNCTION rows additionally get a bare-call AST lane (an unqualified
   ``bacon_decompose(...)`` call has no dot, no quotes, no ``=``).
   Accepted limitations: rows whose old token equals the new token (the
   API-move family M-020..M-027, M-084, M-117..M-120) are skipped - token
   lanes cannot distinguish old surface from new; their known readers are
   recorded in ``code_refs`` anyway. ``param-value`` rows (M-086) are excluded the same
   way. Rows whose ``deprecated_in`` window has not opened defer via the
   version-aware lifecycle gate (``_NEXT_RELEASE``) and arm automatically at
   the version bump. Ambiguous-param readers that build kwargs INDIRECTLY -
   ``dict(time=...)`` / ``{"time": ...}`` fed through ``**fit_kwargs`` (the
   ``power.py`` fit-kwargs builders) - are outside the AST call-site lane:
   extending it to dict keys measured 12 hit files of which only ``power.py``
   is a genuine reader (the rest build CANONICAL calendar-time payloads), an
   11:1 noise ratio, so those known readers are recorded in the ``time``
   family's ``code_refs`` by hand instead; the 3.9 FutureWarnings fire
   through ``**kwargs`` expansion at runtime and the removal turns them into
   loud TypeErrors under test coverage, so the silent getattr class is not
   implicated. The same trade-off DECLINES raw attr / quoted-dict-key lanes
   for ambiguous tokens (``self.robust`` reads, ``{"robust": self.robust}``
   in ``get_params``/``to_dict`` bodies): those sites live almost entirely in
   the defining estimators' own modules, which every rename row's
   ``code_refs`` already names, while the lane family measures 24-35 files
   per ambiguous token repo-wide - rule 11's pre-terminal repo-wide grep
   remains the manual backstop for that residue.
"""

import ast
import dataclasses
import inspect
import re

import pytest

import diff_diff
from tests.test_v4_matrix import (
    _LOCATOR_RE,
    REPO_ROOT,
    ROWS,
    SPEC,
    _import_module_hard,
    _version_tuple,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ROW_INDEX = {row["id"]: row for row in ROWS}
_TERMINAL_STATUSES = {"done", "removed"}

# Phase -> version that phase ships in (phase 1 predates the ledger and ships
# nothing; the table's own "Ships in" column is cross-checked in
# test_ship_version_map_matches_table).
_SHIP_VERSION = {2: "3.9", 3: "3.9", 4: "3.9", 5: "4.0", 6: "4.1"}

# Release ladder (v4-design section 2): maps the CURRENT (major, minor) to the
# NEXT release. Keyed off diff_diff.__version__ so the Duty C lifecycle gate
# re-arms mechanically at each version bump; a missing key fails loudly.
_NEXT_RELEASE = {(3, 8): "3.9", (3, 9): "4.0", (4, 0): "4.1"}

# Section-8 vocabulary predicate (Duty A). Every param/field rename token in
# the ledger - current and lifecycle-gated - must be matched here
# (test_predicate_binds_to_ledger_tokens), so a new rename row cannot be
# silently unswept.
_PATTERN_TOKENS = {
    "time",
    "controls",
    "cohort",
    "aggregation",
    "aggregate",
    "robust",
    "clean_control",
    "group",
    "groups",
    "overall_att",
    "lambda_reg",
    "zeta",
    "placebo_effects",
    "period_effects",
}

# Tokens that are LEGAL canonical vocabulary on other surfaces (rule 1 `time`,
# rule 3 `group`, CS-notation `groups`, ...). Duty C matches these through the
# precise lanes only (AST call-site / field attr reads), never a bare grep.
_AMBIGUOUS_TOKENS = {
    "time",
    "group",
    "groups",
    "cohort",
    "controls",
    "robust",
    "aggregation",
    # M-139's old token: `.aggregate` is ALSO the canonical post-fit method
    # name library-wide, so only the precise lanes may match it - never a
    # bare grep (docs/guides-lane hits on canonical successor vocabulary
    # are covered by CONSUMER_ALLOWLIST entries below).
    "aggregate",
}


def _pattern_hit(name):
    return name.endswith("_col") or name in _PATTERN_TOKENS


# ---------------------------------------------------------------------------
# Duty B helpers: section-9 phase-table parsing + the citation predicate
# ---------------------------------------------------------------------------

_PHASE_ROW_RE = re.compile(r"^\|\s*(\d)\b")
_ENDPOINT_RANGE_RE = re.compile(r"\[M-(\d{3})\]\.\.\[M-(\d{3})\]")
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_ELEMENT_RE = re.compile(r"^M-(\d{3})(?:\.\.M-(\d{3}))?\b")


def extract_phase_cells(text):
    """Return {phase: (ships_in_cell, pr_cell)} from section 9's table ONLY.

    Scope toggles on ``## `` headers, so other sections' tables and the
    citation-semantic paragraph after the table never leak in.
    """
    cells = {}
    in_section_9 = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_section_9 = line.startswith("## 9.")
            continue
        if not in_section_9:
            continue
        m = _PHASE_ROW_RE.match(line)
        if not m:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        cells[int(m.group(1))] = (parts[2].strip(), parts[3].strip())
    return cells


def expand_citations(cell):
    """Expand a cell's bracketed citations to the full id set.

    Handles all three citation forms; bare (unbracketed) ids never match.
    """
    ids = set()

    def _add_range(lo, hi):
        for n in range(int(lo), int(hi) + 1):
            ids.add(f"M-{n:03d}")

    def _consume_endpoint(m):
        _add_range(m.group(1), m.group(2))
        return " "

    remainder = _ENDPOINT_RANGE_RE.sub(_consume_endpoint, cell)
    for bracket in _BRACKET_RE.findall(remainder):
        for element in bracket.split(","):
            m = _ELEMENT_RE.match(element.strip())
            if m:
                _add_range(m.group(1), m.group(2) or m.group(1))
    return ids


def direction2_ok(row, ship_version):
    """The Duty B direction-2 citation predicate (terminal rows exempt)."""
    if row.get("status") in _TERMINAL_STATUSES:
        return True
    versions = {
        row.get("introduced_in"),
        row.get("deprecated_in"),
        row.get("removed_in"),
    }
    if row.get("kind") == "env-default":
        versions.add(row.get("decision_due"))
    return ship_version in versions


def _phase_citations():
    cells = extract_phase_cells(SPEC.read_text())
    return {phase: expand_citations(pr_cell) for phase, (_, pr_cell) in cells.items()}


# ---------------------------------------------------------------------------
# Locator helpers (Duty A + C)
# ---------------------------------------------------------------------------


def _parse_locator(locator, rid):
    m = _LOCATOR_RE.match(locator)
    if m is None:
        pytest.fail(f"{rid}: locator '{locator}' does not match the grammar")
    return m.group("mod"), m.group("attrs"), m.group("param")


def _token_from_locator(locator, rid):
    """Old-name token: the [param] group, else the last attr, else nothing.

    ``locator`` may be None (``new: null`` drops) - the token is None, which
    never equals an old token, so null-new rows are always enforced.
    """
    if locator is None:
        return None
    mod, attrs, param = _parse_locator(locator, rid)
    if param:
        return param
    if attrs:
        return attrs.split(".")[-1]
    return mod.split(".")[-1]


def _resolve_old_surface(locator, rid):
    """Resolve a param/field locator to its defining object.

    Returns ``("param", defining_function, param_name)`` for ``[param]``
    locators (the function actually carrying the signature, base class's for
    inherited params - matching the sweep's dedup key) or
    ``("attr", class_or_module_obj, attr_name)`` for dotted-attr locators.
    Unlike ``test_v4_matrix.resolve_locator`` this returns the OBJECT (that
    helper returns only ``(resolved, detail)``), so a mis-resolution here
    would silently mis-key the rowed index - see its dedicated self-test.
    """
    mod, attrs, param = _parse_locator(locator, rid)
    module = _import_module_hard(mod, rid)
    target = module
    attr_chain = attrs.split(".") if attrs else []
    walked = []
    for attr in attr_chain if param else attr_chain[:-1]:
        try:
            nxt = inspect.getattr_static(target, attr) if walked else getattr(target, attr)
        except AttributeError:
            pytest.fail(f"{rid}: locator '{locator}' - '{attr}' absent on {target!r}")
        target = nxt
        walked.append(attr)
    if param is None:
        return ("attr", target, attr_chain[-1] if attr_chain else None)
    func = target.__init__ if inspect.isclass(target) else target
    func = _unwrap_callable(func)
    if func is None:
        pytest.fail(f"{rid}: locator '{locator}' target is not callable")
    return ("param", func, param)


def _unwrap_callable(obj):
    if isinstance(obj, (staticmethod, classmethod)):
        return obj.__func__
    if isinstance(obj, property):
        return obj.fget
    if inspect.isfunction(obj) or inspect.ismethod(obj) or callable(obj):
        return getattr(obj, "__func__", obj)
    return None


# ---------------------------------------------------------------------------
# Duty A: the public-surface sweep
# ---------------------------------------------------------------------------


def _declaring_class(cls, field_name):
    for base in cls.__mro__:
        if field_name in vars(base).get("__annotations__", {}):
            return base
    return cls


def _sweep_public_surface():
    """Yield pattern hits over every ``diff_diff.__all__`` export.

    Hit shapes:
      ("param", owner_label, defining_func, param)  - owner_label like
          "DifferenceInDifferences.fit" or "trim_weights"
      ("field", class_qualname, field_name)
      ("prop",  class_qualname, prop_name)
    Dedup is by id() of the defining function object (params) / declaring
    class (fields), so aliases and inherited surfaces collapse.
    """
    hits = []
    seen_funcs = set()
    seen_classes = set()

    def _sweep_callable(owner_label, func):
        if id(func) in seen_funcs:
            return
        seen_funcs.add(id(func))
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return
        for pname in sig.parameters:
            if pname in ("self", "cls"):
                continue
            if _pattern_hit(pname):
                hits.append(("param", owner_label, func, pname))

    for name in diff_diff.__all__:
        obj = getattr(diff_diff, name)
        if inspect.isclass(obj):
            if id(obj) in seen_classes:
                continue
            seen_classes.add(id(obj))
            is_dc = dataclasses.is_dataclass(obj)
            if is_dc:
                for fname in obj.__dataclass_fields__:
                    decl = _declaring_class(obj, fname)
                    if id(decl) in seen_classes and decl is not obj:
                        pass  # fields attributed once per declaring class below
                    if _pattern_hit(fname):
                        hits.append(("field", decl.__qualname__, fname))
                # A CUSTOM dataclass __init__ (init=False + hand-written) can
                # carry params that are NOT fields - sweep exactly those, so
                # the mirror assumption cannot hide a prohibited constructor
                # param.
                try:
                    init_sig = inspect.signature(obj.__init__)
                except (TypeError, ValueError):
                    init_sig = None
                if init_sig is not None:
                    for pname in init_sig.parameters:
                        if pname in ("self", "cls") or pname in obj.__dataclass_fields__:
                            continue
                        if _pattern_hit(pname):
                            hits.append(
                                (
                                    "param",
                                    f"{obj.__qualname__}.__init__",
                                    _unwrap_callable(obj.__init__),
                                    pname,
                                )
                            )
            for base in obj.__mro__:
                if base is object:
                    continue
                for mname, member in vars(base).items():
                    if mname.startswith("_") and mname != "__init__":
                        continue
                    if mname == "__init__" and is_dc:
                        continue  # generated; params mirror the fields
                    if isinstance(member, property):
                        if _pattern_hit(mname):
                            hits.append(("prop", base.__qualname__, mname))
                        continue
                    func = _unwrap_callable(member)
                    if func is None or not inspect.isfunction(func):
                        continue
                    label = f"{base.__qualname__}.{mname}"
                    _sweep_callable(label, func)
        elif callable(obj):
            _sweep_callable(getattr(obj, "__name__", name), obj)
    # field hits can repeat via subclass __all__ entries; dedup by key
    deduped = []
    seen_keys = set()
    for hit in hits:
        key = (hit[0], hit[1], hit[-1]) if hit[0] != "param" else (hit[0], id(hit[2]), hit[3])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(hit)
    return deduped


def _build_rowed_index():
    """Ledger coverage index + structural-exemption sets from the rows."""
    param_keys = set()
    attr_keys = set()
    wrapper_names = set()
    exempt_classes = set()
    for row in ROWS:
        rid = row["id"]
        kind = row.get("kind")
        status = row.get("status")
        if kind == "function" and row.get("removed_in"):
            wrapper_names.add(_token_from_locator(row["old"], rid))
            continue
        if kind == "class" and row.get("removed_in"):
            mod, attrs, _ = _parse_locator(row["old"], rid)
            module = _import_module_hard(mod, rid)
            target = module
            resolved = True
            for attr in (attrs or "").split("."):
                target = getattr(target, attr, None)
                if target is None:
                    resolved = False
                    break
            if resolved and inspect.isclass(target):
                exempt_classes.add(target)
            continue
        if kind not in ("param", "field") or status in _TERMINAL_STATUSES:
            continue
        shape = _resolve_old_surface(row["old"], rid)
        if shape[0] == "param":
            param_keys.add((id(shape[1]), shape[2]))
        else:
            owner = shape[1]
            attr_keys.add((getattr(owner, "__qualname__", str(owner)), shape[2]))
    return param_keys, attr_keys, wrapper_names, exempt_classes


# Duty A allowlist: surfaces the predicate flags that are documented domain
# vocabulary (v4-design :588-601) or rule-1 canonical calendar columns. Every
# entry must be an ACTUAL current sweep hit (test_allowlists_are_reachable).
_CS_COHORT = "ATT(g,t) cohort in Callaway-Sant'Anna's own notation (v4-design section 8 carve-out)"
_RULE1_TIME = "rule-1 canonical calendar column (v4-design section 8 rule 1)"

# The 8 surviving staggered-family results containers whose `groups` field
# names the ATT(g,t) cohort (dCDH's `groups` means unit ids and IS rowed, M-114).
_CS_GROUPS_CLASSES = (
    "CallawaySantAnnaResults",
    "ContinuousDiDResults",
    "EfficientDiDResults",
    "ImputationDiDResults",
    "StackedDiDResults",
    "SunAbrahamResults",
    "TwoStageDiDResults",
    "WooldridgeDiDResults",
)

# Rule-1 canonical calendar `time` surfaces (the sweep's inventory the
# 2(c)/3(a) rename PRs work from - these are the LEGAL `time`s; the two
# 0/1-post overloads are rowed as M-030/M-082).
_RULE1_TIME_SURFACES = (
    "BaconDecomposition.fit[time]",
    "BusinessReport.__init__[time]",
    "CallawaySantAnna.diagnose_propensity[time]",
    "CallawaySantAnna.fit[time]",
    "ChaisemartinDHaultfoeuille.fit[time]",
    "ChangesInChanges.fit[time]",
    "ContinuousDiD.fit[time]",
    "DiagnosticReport.__init__[time]",
    "EfficientDiD.fit[time]",
    "EfficientDiD.hausman_pretest[time]",
    "GroupTimeEffect.time",
    "HeterogeneousAdoptionDiD.fit[time]",
    "ImputationDiD.fit[time]",
    "LPDiD.fit[time]",
    "SpilloverDiD.fit[time]",
    "StackedDiD.fit[time]",
    "SunAbraham.fit[time]",
    "SyntheticControl.fit[time]",
    "SyntheticDiD.fit[time]",
    "TROP.fit[time]",
    "TwoStageDiD.fit[time]",
    "TwoWayFixedEffects.decompose[time]",
    "WooldridgeDiD.fit[time]",
    "agent_workflow[time]",
    "check_parallel_trends[time]",
    "check_parallel_trends_robust[time]",
    "did_had_pretest_workflow[time]",
    "equivalence_test_trends[time]",
    "joint_homogeneity_test[time]",
    "joint_pretrends_test[time]",
    "placebo_group_test[time]",
    "placebo_timing_test[time]",
    "plot_staircase[time]",
    "profile_panel[time]",
    "summarize_did_data[time]",
    "twowayfeweights[time]",
    "validate_did_data[time]",
)

SURFACE_ALLOWLIST = {
    **{f"{cls}.groups": _CS_COHORT for cls in _CS_GROUPS_CLASSES},
    "GroupTimeEffect.group": _CS_COHORT,
    "HADPretestReport.aggregate": (
        "honest OUTPUT metadata, not the deprecated param (M-139 kills only "
        "did_had_pretest_workflow's aggregate= INPUT): the field records "
        "which pretest battery the workflow RAN, and the overall/event_study "
        "modes survive 4.0 - only the routing param dies"
    ),
    "plot_group_effects[groups]": _CS_COHORT + " - cohort selector on the plotting surface",
    "TripleDifference.fit[group]": (
        "rule-3 reserved treated-group 0/1 indicator (v4-design section 8 rule 3)"
    ),
    "run_placebo_test[time]": (
        "OVERLOADED pass-through, redesign pending (TODO.md): forwarded as "
        "the calendar column to placebo_timing_test/placebo_group_test AND "
        "as the 0/1 post dummy to permutation_test/leave_one_out_test "
        "(M-137/M-138) - workable only because a two-period 0/1 calendar "
        "column is both; a rename cannot fix the dual semantics"
    ),
    "run_all_placebo_tests[time]": (
        "OVERLOADED pass-through, redesign pending (TODO.md) - see " "run_placebo_test[time]"
    ),
    "StackedDiD.clean_control": (
        "M-043's deprecated estimator-attribute alias PROPERTY (warns and "
        "returns control_group; dies with the param at 4.0) - the param row "
        "covers the __init__ surface, this entry covers the property lane"
    ),
    **{key: _RULE1_TIME for key in _RULE1_TIME_SURFACES},
    "rank_control_units[lambda_reg]": (
        "prep helper's own independent regularization param (prep.py) - a "
        "different defining function from M-001's SyntheticDiD[lambda_reg]"
    ),
}


def _surface_key(hit):
    if hit[0] == "param":
        return f"{hit[1]}[{hit[3]}]"
    return f"{hit[1]}.{hit[2]}"


# ---------------------------------------------------------------------------
# Duty C: consumer-coverage lanes
# ---------------------------------------------------------------------------

_SOURCE_FILES = sorted(
    p for p in (REPO_ROOT / "diff_diff").rglob("*.py") if "__pycache__" not in p.parts
)
_DOCS_FILES = sorted((REPO_ROOT / "docs" / "methodology").rglob("*.md"))
_GUIDES_FILES = sorted((REPO_ROOT / "diff_diff" / "guides").glob("*.txt"))


def _relpath(path):
    return str(path.relative_to(REPO_ROOT))


def _read(path, _cache={}):
    if path not in _cache:
        _cache[path] = path.read_text()
    return _cache[path]


def _tree(path, _cache={}):
    if path not in _cache:
        _cache[path] = ast.parse(_read(path))
    return _cache[path]


def quoted_hits(tok, text):
    return re.search(r"[\"']" + re.escape(tok) + r"[\"']", text) is not None


def kwarg_hits(tok, text):
    return re.search(r"(?<![\w.])" + re.escape(tok) + r"\s*=(?!=)", text) is not None


def attr_hits(tok, text):
    pat = re.compile(r"\." + re.escape(tok) + r"\b")
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("from ", "import ")):
            continue
        if pat.search(line):
            return True
    return False


def backtick_hits(tok, text):
    return "`" + tok + "`" in text


def guides_hits(tok, text, bare_call=False):
    if kwarg_hits(tok, text) or backtick_hits(tok, text):
        return True
    return bare_call and re.search(r"(?<![\w.])" + re.escape(tok) + r"\(", text) is not None


def bare_call_text_hits(tok, text, exempt_spans=()):
    """Raw-text bare-call lane for FUNCTION rows: ``tok(`` anywhere the AST
    cannot see - docstring examples especially (``_diagnostic.py``'s ``>>> results = bacon_decompose(...)`` example goes
    stale at removal). Definition lines and imports are excluded (a removed
    symbol's import breaks loudly on its own), and so are lines inside
    ``exempt_spans`` - the wrapper-body (lineno, end_lineno) spans, mirroring
    the AST lane's exemption (the OR of the two lanes would otherwise
    nullify it)."""
    pat = re.compile(r"(?<![\w.])" + re.escape(tok) + r"\(")
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(lo <= lineno <= hi for lo, hi in exempt_spans):
            continue
        stripped = line.lstrip()
        if stripped.startswith(("def ", "from ", "import ", "async def ")):
            continue
        if pat.search(line):
            return True
    return False


def _wrapper_spans(tree, wrapper_names):
    return [
        (node.lineno, node.end_lineno)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wrapper_names
    ]


def _callee_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def ast_state_read_hits(tok, tree):
    """Literal-name state reads: ``getattr(x, "tok", ...)`` / ``hasattr`` /
    mapping ``x.get("tok", ...)``.

    THE silent-degradation forms rule 11 names - after removal each returns
    its default instead of raising. Precise even for
    ambiguous tokens: measured 3 getattr/hasattr + 4 ``.get`` (token, file)
    pairs repo-wide vs 24-32 files for a quoted-string lane.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "hasattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == tok
        ):
            return True
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == tok
        ):
            return True
    return False


def ast_call_hits(tok, callees, tree, wrapper_spans, bare_call=False):
    """AST lane: kwarg ``tok`` at a callee named in ``callees`` (param rows),
    or - with ``bare_call`` - a call whose callee IS ``tok`` (function rows).
    Hits inside a top-level wrapper FunctionDef are exempt."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        lineno = node.lineno
        if any(lo <= lineno <= hi for lo, hi in wrapper_spans):
            continue
        name = _callee_name(node)
        if bare_call and name == tok:
            return True
        if name in callees and any(kw.arg == tok for kw in node.keywords):
            return True
    return False


def _enforceable_rename_rows():
    """Duty C rows + their (token, lanes) - see module docstring for tiering."""
    vt = _version_tuple(diff_diff.__version__)
    current = (vt[0], vt[1])
    if current not in _NEXT_RELEASE:
        pytest.fail(
            f"__version__ {diff_diff.__version__} has no _NEXT_RELEASE entry - "
            "extend the ladder map so the Duty C lifecycle gate re-arms"
        )
    horizon = _version_tuple(_NEXT_RELEASE[current])
    rows = []
    for row in ROWS:
        rid = row["id"]
        if row.get("kind") not in ("param", "field", "function"):
            continue
        # A rename is live once EITHER lifecycle field is set: M-031/M-082
        # (fit time->post) carry deprecated_in "3.9" with removed_in null -
        # their removal is folded into the 4.0 merge enforcement - and their
        # readers must be tracked all the same.
        if not (row.get("removed_in") or row.get("deprecated_in")):
            continue
        if row.get("status") in _TERMINAL_STATUSES:
            continue
        old_tok = _token_from_locator(row.get("old"), rid)
        new_tok = _token_from_locator(row.get("new"), rid)
        if old_tok is None or old_tok == new_tok:
            continue  # API-move rows: readers recorded in code_refs, not laned
        dep = row.get("deprecated_in")
        if dep is not None and _version_tuple(dep) > horizon:
            continue  # window not open; arms at the version bump
        rows.append(row)
    return rows


def _terminal_rename_rows(rows=None):
    """Rename rows whose lifecycle has COMPLETED (``done``/``removed``).

    Enforcement does not switch off at the removal commit:
    a stale or newly introduced reader of a removed old name is exactly when
    ``getattr(obj, "old", default)`` starts silently returning the default.
    Terminal rows keep their lanes, but their own historical ``code_refs`` no
    longer count as coverage - a hit must be allowlisted or covered by a
    still-LIVE row of the same token family. Vacuous today (zero terminal
    rename rows) - this arms the post-removal era for free.
    """
    out = []
    for row in ROWS if rows is None else rows:
        rid = row.get("id", "?")
        if row.get("kind") not in ("param", "field", "function"):
            continue
        if row.get("status") not in _TERMINAL_STATUSES:
            continue
        old_tok = _token_from_locator(row.get("old"), rid)
        new_tok = _token_from_locator(row.get("new"), rid)
        if old_tok is None or old_tok == new_tok:
            continue
        out.append(row)
    return out


def _safe_family_token(row):
    """Row's old-name token for family grouping, or None when inapplicable.

    The ONE token parser for both family-union functions (two hand-rolled
    extractions drifted - `param-value` locators carry an
    ``=value`` suffix outside ``_LOCATOR_RE``'s grammar and crashed the
    terminal path). ``param-value`` rows group by their PARAM name so their
    ``code_refs`` join the family union; unparseable/absent locators return
    None instead of failing.
    """
    old = row.get("old")
    if old is None:
        return None
    if "=" in old:  # param-value grammar: Class.method[param]=value
        head = old.split("=", 1)[0]
        return head.split("[")[1].rstrip("]") if "[" in head else None
    m = _LOCATOR_RE.match(old)
    if m is None:
        return None
    return m.group("param") or (m.group("attrs") or m.group("mod")).split(".")[-1]


def _live_family_code_refs(tok):
    """code_refs union over NON-terminal rows sharing the token only."""
    refs = set()
    for row in ROWS:
        if row.get("status") in _TERMINAL_STATUSES:
            continue
        if row.get("kind") not in ("param", "field", "function", "param-value"):
            continue
        if _safe_family_token(row) == tok:
            refs.update(row.get("code_refs") or [])
    return refs


# Deprecation shims whose __init__ FORWARDS verbatim (*args/**kwargs +
# super().__init__) to the named base with a mirrored __signature__ (the
# 3.9 class merges, v4-design section 4.1). Their effective __init__ is no
# longer the base's function object, but calls under the shim's name still
# read every base constructor param - so they stay in the base's
# init-sharing group. Forward home for the phase-3 siblings (SDDD, QDiD).
_FORWARDING_INIT_SHIMS = {
    "MultiPeriodDiD": "DifferenceInDifferences",
}


def _init_sharing_class_names(_cache={}):
    """Exported class names grouped by the id() of their EFFECTIVE __init__.

    A subclass that inherits its constructor (``TwoWayFixedEffects`` from
    ``DifferenceInDifferences``) is callable under its own name with the
    base's params - ``TwoWayFixedEffects(robust=True)`` reads M-045's dying
    param under a callee name the base-class form cannot see. Forwarding
    deprecation shims (``_FORWARDING_INIT_SHIMS``) join their base's group
    by declaration - identity grouping cannot see through the wrapper."""
    if not _cache:
        groups = {}
        for name in diff_diff.__all__:
            obj = getattr(diff_diff, name)
            if not inspect.isclass(obj):
                continue
            func = _unwrap_callable(inspect.getattr_static(obj, "__init__", None))
            if func is not None:
                groups.setdefault(id(func), set()).add(name)
        for shim_name, base_name in _FORWARDING_INIT_SHIMS.items():
            base = getattr(diff_diff, base_name, None)
            if base is None or not inspect.isclass(base):
                continue
            base_func = _unwrap_callable(inspect.getattr_static(base, "__init__", None))
            if base_func is not None:
                groups.setdefault(id(base_func), set()).add(shim_name)
        _cache["groups"] = groups
    return _cache["groups"]


def _ast_callees(row):
    """Callee names whose keyword args count as readers of a param row.

    ``Class.method[param]`` -> the method name (a subclass's
    ``super().method(...)`` delegation has the same callee name, so it is
    covered too). ``Class[param]`` (constructor) -> the class name PLUS
    ``__init__`` (a delegating ``super().__init__(tok=...)`` is a genuine
    reader - ``SyntheticDiD``'s ``robust=True``) PLUS every
    exported class whose effective __init__ IS the same function (inherited
    constructors called under the subclass name).
    """
    _, attrs, param = _parse_locator(row["old"], row["id"])
    if param is None or not attrs:
        return set()
    parts = attrs.split(".")
    if len(parts) > 1:
        return {parts[-1]}
    callees = {parts[0], "__init__"}
    cls = getattr(diff_diff, parts[0], None)
    if inspect.isclass(cls):
        func = _unwrap_callable(inspect.getattr_static(cls, "__init__", None))
        if func is not None:
            callees |= _init_sharing_class_names().get(id(func), set())
    return callees


def _consumer_hit_files(row):
    """All files the row's lanes flag, as repo-relative paths."""
    rid = row["id"]
    kind = row["kind"]
    tok = _token_from_locator(row["old"], rid)
    ambiguous = tok in _AMBIGUOUS_TOKENS
    _, _, wrapper_names, _ = _rowed_index()
    files = set()

    for path in _SOURCE_FILES:
        text = _read(path)
        hit = False
        if kind == "function":
            spans = _wrapper_spans(_tree(path), wrapper_names)
            hit = (
                quoted_hits(tok, text)
                or kwarg_hits(tok, text)
                or attr_hits(tok, text)
                or bare_call_text_hits(tok, text, exempt_spans=spans)
                or ast_call_hits(tok, set(), _tree(path), spans, bare_call=True)
            )
        elif kind == "param":
            if ambiguous:
                hit = ast_call_hits(
                    tok, _ast_callees(row), _tree(path), _wrapper_spans(_tree(path), wrapper_names)
                ) or ast_state_read_hits(tok, _tree(path))
            else:
                hit = quoted_hits(tok, text) or kwarg_hits(tok, text) or attr_hits(tok, text)
        else:  # field
            hit = quoted_hits(tok, text) or attr_hits(tok, text)
            if not ambiguous:
                hit = hit or kwarg_hits(tok, text)
        if hit:
            files.add(_relpath(path))

    # Docs + guides: exact-backtick plus the QUALIFIED forms (
    # `StackedDiDResults.clean_control` / `results.clean_control` /
    # `fit(tok=...)` in fenced examples carry no bare-backtick token). The
    # anchored forms measure 0-4 docs files per token - unlike bare
    # word-in-span matching, which balloons to 10+ for ambiguous words.
    for path in _DOCS_FILES:
        text = _read(path)
        if (
            backtick_hits(tok, text)
            or kwarg_hits(tok, text)
            or attr_hits(tok, text)
            or (kind == "function" and bare_call_text_hits(tok, text))
        ):
            files.add(_relpath(path))

    for path in _GUIDES_FILES:
        text = _read(path)
        if guides_hits(tok, text, bare_call=(kind == "function")) or attr_hits(tok, text):
            files.add(_relpath(path))

    return files


def _token_family_code_refs(tok):
    """Union of code_refs across ALL rows sharing the old token (rule 11's
    token-family clause) - record-anyway entries on gated rows count."""
    refs = set()
    for row in ROWS:
        if row.get("kind") not in ("param", "field", "function", "param-value"):
            continue
        if _safe_family_token(row) == tok:
            refs.update(row.get("code_refs") or [])
    return refs


# Duty C allowlist: (token, repo-relative path) -> reason. Entries are hits of
# the ACTIVE lanes that are NOT readers of the dying surface (independent
# same-named surfaces, canonical-vocabulary usage, matcher noise). Every entry
# must remain an actual lane hit (test_allowlists_are_reachable).
_CS_GROUPS_READER = (
    "legal reader of the SURVIVING CS-family groups fields (ATT(g,t) cohorts), "
    "not dCDH's unit-id groups (M-114)"
)

CONSUMER_ALLOWLIST = {
    ("lambda_reg", "diff_diff/prep.py"): (
        "rank_control_units' own independent regularization param - not the "
        "removed SyntheticDiD kwarg (M-001)"
    ),
    # M-139's `aggregate` token doubles as the library's canonical post-fit
    # METHOD name, so every doc that teaches `results.aggregate(...)` hits
    # the docs/guides attr lane. Each entry below was verified clean of
    # stale fit-time `aggregate=` teaching before allowlisting (the
    # migrate-first rule; stale sites were migrated in the same diff that
    # added these entries). Dated phase-closure records in the paper-review
    # doc describe the API as-shipped at closure time (the released-record
    # convention) and are not regenerated for a deprecation sweep.
    ("aggregate", "diff_diff/guides/llms-autonomous.txt"): (
        "canonical post-fit results.aggregate() vocabulary; the HAD "
        "mode-kwarg teachings were migrated to panel-shape wording with "
        "M-027/M-139"
    ),
    ("aggregate", "docs/methodology/REGISTRY.md"): (
        "the aggregate-postfit register Notes (M-020..M-027) teach the "
        "canonical successor method; HAD-section fit-time mode mentions "
        "were migrated with M-027"
    ),
    ("aggregate", "docs/methodology/REPORTING.md"): (
        "canonical post-fit aggregate() mention only (zero fit-time "
        "kwarg sites; the fit-time-population clause was reworded with "
        "M-027)"
    ),
    ("aggregate", "docs/methodology/papers/dechaisemartin-2026-review.md"): (
        "dated phase-closure records describing the API as-shipped at "
        "closure time (paper-review docs source from the paper + the "
        "then-current implementation; the live surface is the REGISTRY "
        "HAD Note)"
    ),
    ("aggregate", "docs/methodology/papers/roth-2022-review.md"): (
        "canonical post-fit results.aggregate('event_study') pointer "
        "(the invalid `aggregate=event` recommendation was corrected "
        "with M-027)"
    ),
    ("aggregate", "docs/methodology/variance-conventions.md"): (
        "canonical post-fit aggregate() vocabulary only (zero fit-time " "kwarg sites)"
    ),
    **{
        (
            "zeta",
            f,
        ): "internal solver vocabulary, independent of the removed SyntheticDiD kwarg (M-002)"
        for f in (
            "diff_diff/conformal.py",
            "diff_diff/prep.py",
            "diff_diff/synthetic_control.py",
            "diff_diff/utils.py",
        )
    },
    ("time", "diff_diff/guides/llms-autonomous.txt"): (
        "canonical calendar-time kwargs on rule-1 surfaces (agent_workflow/"
        "profile_panel) - no post-overload usage in this guide"
    ),
    ("time", "docs/methodology/papers/goodman-bacon-2021-review.md"): (
        "canonical calendar column prose (bacon period remapping), not the M-030 overload"
    ),
    ("time", "docs/methodology/papers/wooldridge-2023-review.md"): (
        "canonical calendar column prose in the shipped-API description, not the M-030 overload"
    ),
    ("cohort", "docs/methodology/papers/borusyak-jaravel-spiess-2024-review.md"): (
        "ImputationDiD partition-value prose, not the Wooldridge fit[cohort] kwarg"
    ),
    ("group", "diff_diff/guides/llms-autonomous.txt"): (
        "DDD treated-group covariate prose - rule-3's reserved meaning, not dCDH's unit id"
    ),
    ("group", "docs/methodology/REGISTRY.md"): (
        "aggregation-type values ('group' cell-count weighting) and the rule-3 "
        "TripleDifference notes, not dCDH's unit-id kwarg"
    ),
    ("group", "docs/methodology/papers/wooldridge-2025-review.md"): (
        "aggregation-type value in the weight table, not dCDH's unit-id kwarg"
    ),
    ("controls", "docs/methodology/papers/dechaisemartin-dhaultfoeuille-2022-review.md"): (
        "the R package's own option name (did_multiplegt), not diff-diff's fit[controls]"
    ),
    ("outcome_col", "diff_diff/profile.py"): "local-variable assignment noise, not an API reader",
    ("unit_col", "diff_diff/power.py"): "local-variable assignment noise, not an API reader",
    ("time_col", "diff_diff/chaisemartin_dhaultfoeuille.py"): (
        "an internal helper's own time_col parameter, not the HAD/pretest API"
    ),
    ("weight_col", "diff_diff/practitioner.py"): (
        "column-name example string in guidance prose, not the trim_weights kwarg"
    ),
    ("trop", "diff_diff/prep_dgp.py"): "local-variable assignment noise, not the wrapper",
    ("synthetic_control", "diff_diff/estimators.py"): (
        "module-path mentions in docstring/comments, not calls to the wrapper function"
    ),
    ("stacked_did", "docs/methodology/papers/goodman-bacon-2021-review.md"): (
        "names the method family in prose, not the wrapper function"
    ),
    ("zeta", "docs/methodology/REGISTRY.md"): (
        "internal solver call spelling (zeta=0) in the SC/SDiD math notes, not "
        "the removed SyntheticDiD kwarg"
    ),
    ("time", "docs/methodology/papers/dube-2025-review.md"): (
        "Stata factor-variable notation (i.time) in the paper's RA syntax, not "
        "the fit[time] overload"
    ),
    ("time", "diff_diff/diagnostic_report.py"): (
        "getattr reads of GroupTimeEffect's SURVIVING CS-notation time field, "
        "not the fit[time] overload"
    ),
    ("group", "diff_diff/diagnostic_report.py"): (
        "getattr reads of GroupTimeEffect's SURVIVING CS-notation group field, "
        "not the dCDH unit-id kwarg"
    ),
    ("group", "docs/methodology/papers/borusyak-jaravel-spiess-2024-review.md"): (
        "partition/estimand math prose, not the dCDH unit-id kwarg"
    ),
    ("group", "docs/methodology/papers/chen-santanna-xie-2025-review.md"): (
        "parallel-trends assumption math (G=g conditioning), not the dCDH kwarg"
    ),
    ("group", "docs/methodology/papers/gardner-2022-review.md"): (
        "treatment-group index math prose, not the dCDH kwarg"
    ),
    ("controls", "docs/methodology/papers/dube-2025-review.md"): (
        "clean-controls estimand prose (Equation 13), not the fit[controls] kwarg"
    ),
    ("aggregation", "docs/methodology/continuous-did.md"): (
        "the R contdid package's own cont_did(aggregation=) option in its "
        "documented API surface, not our to_dataframe/summary param"
    ),
    ("aggregation", "diff_diff/business_report.py"): (
        "reporting-block dict key ('aggregation' in the serialized report "
        "schema), an independent concept from the Wooldridge param"
    ),
    ("aggregation", "diff_diff/diagnostic_report.py"): (
        "reporting-block dict key ('aggregation' in the serialized report "
        "schema), an independent concept from the Wooldridge param"
    ),
    ("groups", "diff_diff/guides/llms-autonomous.txt"): _CS_GROUPS_READER,
    **{
        ("groups", f): _CS_GROUPS_READER
        for f in (
            "diff_diff/continuous_did_results.py",
            "diff_diff/efficient_did.py",
            "diff_diff/efficient_did_results.py",
            "diff_diff/guides/llms-full.txt",
            "diff_diff/imputation_results.py",
            "diff_diff/results_base.py",
            "diff_diff/stacked_did_results.py",
            "diff_diff/staggered_results.py",
            "diff_diff/staggered_triple_diff_results.py",
            "diff_diff/sun_abraham.py",
            "diff_diff/two_stage_results.py",
            "diff_diff/visualization/_staggered.py",
            "diff_diff/wooldridge.py",
            "diff_diff/wooldridge_results.py",
        )
    },
}


# ===========================================================================
# Guard tests
# ===========================================================================


def test_phase_table_direction1_every_live_row_cited_in_its_phase_cell():
    citations = _phase_citations()
    missing = []
    for row in ROWS:
        if row.get("status") in _TERMINAL_STATUSES:
            continue
        phase = int(row["phase"])
        if row["id"] not in citations.get(phase, set()):
            missing.append(f"{row['id']} (phase {phase}, status {row.get('status')})")
    assert not missing, (
        "live ledger rows missing from their section-9 phase cell " f"(direction 1): {missing}"
    )


def test_phase_table_direction2_every_citation_has_lifecycle_at_ship_version():
    citations = _phase_citations()
    bad = []
    for phase, cited in citations.items():
        ship = _SHIP_VERSION.get(phase)
        for rid in sorted(cited):
            row = _ROW_INDEX.get(rid)
            if row is None:
                bad.append(f"{rid} cited in phase {phase} but not a ledger row")
                continue
            if ship is None:
                bad.append(f"{rid} cited in phase {phase}, which ships nothing")
                continue
            if not direction2_ok(row, ship):
                bad.append(
                    f"{rid} cited in phase {phase} (ships {ship}) but its lifecycle "
                    f"is intro={row.get('introduced_in')} dep={row.get('deprecated_in')} "
                    f"rem={row.get('removed_in')}"
                )
    assert not bad, f"section-9 citations failing direction 2: {bad}"


def test_named_fixtures_m008_passes_and_m031_prose_is_not_a_citation():
    cells = extract_phase_cells(SPEC.read_text())
    phase5_raw = cells[5][1]
    citations = expand_citations(phase5_raw)
    # Positive fixture: M-008 is scheduled SOLELY via decision_due.
    m008 = _ROW_INDEX["M-008"]
    assert m008.get("kind") == "env-default"
    assert "M-008" in citations
    assert direction2_ok(m008, "4.0")
    stripped = dict(m008)
    stripped["kind"] = "param"
    assert not direction2_ok(
        stripped, "4.0"
    ), "M-008 must pass ONLY through the env-default decision_due channel"
    # Negative fixture: bare prose id is not a citation.
    assert "M-031" in phase5_raw, "phase-5 cell should still carve out M-031 in prose"
    assert "M-031" not in citations, "bare prose M-031 must not be tokenized as a citation"


def test_ship_version_map_matches_table():
    cells = extract_phase_cells(SPEC.read_text())
    assert set(cells) == {1, 2, 3, 4, 5, 6}
    for phase, (ships, _) in cells.items():
        normalized = ships.replace(" cut", "").strip()
        if normalized == "-":
            assert phase not in _SHIP_VERSION
        else:
            assert _SHIP_VERSION[phase] == normalized, (
                f"phase {phase}: hardcoded map says {_SHIP_VERSION.get(phase)}, "
                f"table says {ships!r}"
            )


def _rowed_index(_cache=[]):
    if not _cache:
        _cache.append(_build_rowed_index())
    return _cache[0]


def test_public_surface_pattern_hits_are_rowed_or_allowlisted():
    param_keys, attr_keys, wrapper_names, exempt_classes = _rowed_index()
    exempt_qualnames = {cls.__qualname__ for cls in exempt_classes}
    violations = []
    for hit in _sweep_public_surface():
        key = _surface_key(hit)
        if hit[0] == "param":
            owner_label, func, pname = hit[1], hit[2], hit[3]
            if (id(func), pname) in param_keys:
                continue
            top = owner_label.split(".")[0]
            if top in wrapper_names or top in exempt_qualnames:
                continue
        else:
            qualname, aname = hit[1], hit[2]
            if (qualname, aname) in attr_keys:
                continue
            if qualname in exempt_qualnames:
                continue
        if key in SURFACE_ALLOWLIST:
            continue
        violations.append(key)
    assert not violations, (
        "public surfaces matching section-8 rename vocabulary with no ledger "
        f"row and no allowlist reason: {sorted(violations)}"
    )


def test_allowlists_are_reachable():
    """Anti-drift: every allowlist entry must be an ACTUAL current hit."""
    param_keys, attr_keys, wrapper_names, exempt_classes = _rowed_index()
    exempt_qualnames = {cls.__qualname__ for cls in exempt_classes}
    live_surface_keys = set()
    for hit in _sweep_public_surface():
        if hit[0] == "param":
            if (id(hit[2]), hit[3]) in param_keys:
                continue
            top = hit[1].split(".")[0]
            if top in wrapper_names or top in exempt_qualnames:
                continue
        else:
            if (hit[1], hit[2]) in attr_keys or hit[1] in exempt_qualnames:
                continue
        live_surface_keys.add(_surface_key(hit))
    dead_surface = set(SURFACE_ALLOWLIST) - live_surface_keys
    assert (
        not dead_surface
    ), f"SURFACE_ALLOWLIST entries no longer hit by the sweep: {sorted(dead_surface)}"

    # Terminal rows KEEP producing hits (their stale-reader check needs the
    # same allowlist entries) - excluding them here would declare those
    # entries dead while the terminal test still requires them, making the
    # post-removal state unsatisfiable.
    live_consumer_keys = set()
    for row in _enforceable_rename_rows() + _terminal_rename_rows():
        tok = _token_from_locator(row["old"], row["id"])
        for f in _consumer_hit_files(row):
            live_consumer_keys.add((tok, f))
    dead_consumer = set(CONSUMER_ALLOWLIST) - live_consumer_keys
    assert (
        not dead_consumer
    ), f"CONSUMER_ALLOWLIST entries no longer hit by any lane: {sorted(dead_consumer)}"
    for key, reason in list(SURFACE_ALLOWLIST.items()) + list(CONSUMER_ALLOWLIST.items()):
        assert isinstance(reason, str) and reason.strip(), f"allowlist entry {key} needs a reason"


def test_rename_consumer_files_covered_by_code_refs_union_or_allowlist():
    uncovered = []
    for row in _enforceable_rename_rows():
        rid = row["id"]
        tok = _token_from_locator(row["old"], rid)
        refs = _token_family_code_refs(tok)
        for f in sorted(_consumer_hit_files(row)):
            if f in refs:
                continue
            if (tok, f) in CONSUMER_ALLOWLIST:
                continue
            uncovered.append(f"{rid} token={tok}: {f}")
    assert not uncovered, (
        "old-name readers not in the token family's code_refs union and not "
        f"allowlisted (section 8 rule 11): {uncovered}"
    )


def test_terminal_rename_rows_have_no_unaccounted_readers():
    """Post-removal stale readers fail: a terminal row's hits must be
    allowlisted or covered by a still-live same-token row - its own historical
    code_refs are no longer proof."""
    stale = []
    for row in _terminal_rename_rows():
        rid = row["id"]
        tok = _token_from_locator(row["old"], rid)
        live_refs = _live_family_code_refs(tok)
        for f in sorted(_consumer_hit_files(row)):
            if f in live_refs or (tok, f) in CONSUMER_ALLOWLIST:
                continue
            stale.append(f"{rid} token={tok}: {f}")
    assert not stale, f"readers of REMOVED old names survive (silent-getattr hazard): {stale}"


def _predicate_bound_tokens(rows):
    """(rid, old_token) for every live param/field rename the predicate must
    match - EITHER lifecycle field arms a row (deprecated-only renames like
    M-031/M-082 included; same condition as ``_enforceable_rename_rows``),
    gated rows included so a version bump cannot break the
    binding."""
    out = []
    for row in rows:
        if row.get("kind") not in ("param", "field"):
            continue
        if not (row.get("removed_in") or row.get("deprecated_in")):
            continue
        if row.get("status") in _TERMINAL_STATUSES:
            continue
        rid = row["id"]
        old_tok = _token_from_locator(row.get("old"), rid)
        new_tok = _token_from_locator(row.get("new"), rid)
        if old_tok is None or old_tok == new_tok:
            continue  # API moves keep the name; nothing for the sweep to catch
        out.append((rid, old_tok))
    return out


def test_predicate_binds_to_ledger_tokens():
    """Every param/field rename token - gated rows INCLUDED - matches the
    Duty A predicate, so a new rename row cannot be silently unswept."""
    unmatched = [
        f"{rid}: {tok}" for rid, tok in _predicate_bound_tokens(ROWS) if not _pattern_hit(tok)
    ]
    assert (
        not unmatched
    ), f"rename tokens outside the Duty A predicate (extend _PATTERN_TOKENS): {unmatched}"


# ===========================================================================
# Parser / matcher self-tests (synthetic fixtures)
# ===========================================================================

_SYNTH_SPEC = """\
## 3. Decoy

| Phase | Ships in | PRs |
|---|---|---|
| 2 decoy | 9.9 | [M-900] must not leak |

## 9. Real

| Phase | Ships in | PRs |
|---|---|---|
| 2: foo | 3.9 | single [M-122] + compound [M-030..M-032, M-040 old names] + M-020's bare id |
| 5: bar | 4.0 | endpoint [M-132]..[M-134]; M-031's old name persists |

Trailing paragraph citing [M-901] must not leak either.
"""


def test_expander_handles_all_three_bracket_forms_and_ignores_bare_ids():
    cells = extract_phase_cells(_SYNTH_SPEC)
    assert set(cells) == {2, 5}, "decoy table and trailing prose must not leak"
    p2 = expand_citations(cells[2][1])
    assert p2 == {"M-122", "M-030", "M-031", "M-032", "M-040"}
    assert "M-020" not in p2, "bare possessive ids are not citations"
    p5 = expand_citations(cells[5][1])
    assert p5 == {"M-132", "M-133", "M-134"}, "endpoint range expands its interior"
    assert "M-031" not in p5


def test_direction2_predicate_on_synthetic_rows():
    env = {"kind": "env-default", "status": "planned", "decision_due": "4.0"}
    terminal = {"kind": "param", "status": "done"}
    wrong = {"kind": "param", "status": "planned", "deprecated_in": "3.9", "removed_in": None}
    assert direction2_ok(env, "4.0")
    assert not direction2_ok(dict(env, kind="param"), "4.0")
    assert direction2_ok(terminal, "4.0"), "terminal rows are exempt"
    assert not direction2_ok(wrong, "4.0"), "no 4.0 lifecycle work -> illegitimate citation"
    assert direction2_ok(wrong, "3.9")


def test_custom_dataclass_init_extra_params_are_swept():
    """Regression: a hand-written dataclass __init__ with a non-field
    param in the rename vocabulary must be swept (the generated-init mirror
    assumption cannot hide it)."""

    @dataclasses.dataclass(init=False)
    class _CustomInit:
        value: float

        def __init__(self, value=0.0, foo_col=None):
            self.value = value

    extra = [
        p
        for p in inspect.signature(_CustomInit.__init__).parameters
        if p not in ("self",) and p not in _CustomInit.__dataclass_fields__ and _pattern_hit(p)
    ]
    assert extra == ["foo_col"]


def test_sweep_flags_synthetic_unrowed_col_param():
    class _Synthetic:
        def fit(self, outcome, foo_col=None):
            return self

    hits = []
    for pname in inspect.signature(_Synthetic.fit).parameters:
        if pname != "self" and _pattern_hit(pname):
            hits.append(pname)
    assert hits == ["foo_col"]


def test_matcher_forms_and_negatives():
    assert quoted_hits("zeta", 'getattr(x, "zeta", None)')
    assert kwarg_hits("zeta", "fit(zeta=1)")
    assert not kwarg_hits("zeta", "fit(zeta_omega=1)")
    assert not kwarg_hits("zeta", "x.zeta == 1")
    assert attr_hits("groups", "value = res.groups")
    assert not attr_hits("stacked_did", "from diff_diff.stacked_did import StackedDiD")
    assert backtick_hits("robust", "the `robust` parameter")
    assert not backtick_hits("robust", "a robustness check")
    # qualified docs forms: dot-qualified and kwarg-in-code
    # references are matched via the attr/kwarg lanes over docs text
    assert attr_hits("clean_control", "reads `StackedDiDResults.clean_control` at")
    assert attr_hits("clean_control", "then results.clean_control is checked")
    assert kwarg_hits("clean_control", 'configured with `clean_control="strict"`')
    assert not attr_hits("clean_control", "the clean control design")
    assert guides_hits("trop", "results = trop(data)", bare_call=True)
    assert not guides_hits("trop", "the TROP estimator", bare_call=True)
    # raw-text bare-call lane (function rows): docstring examples count,
    # definition/import lines do not (regression pin)
    assert bare_call_text_hits(
        "bacon_decompose", '    """Example:\n    >>> results = bacon_decompose(data)\n    """'
    )
    assert not bare_call_text_hits("bacon_decompose", "def bacon_decompose(data):")
    assert not bare_call_text_hits("bacon_decompose", "from diff_diff.bacon import bacon_decompose")


_SYNTH_MODULE = """\
import diff_diff

def stacked_did(data):
    return bacon_decompose(data)

def user_code(data, est):
    est.fit(data, time="period")
    other.transform(time="period")
    return bacon_decompose(data)

class Sub(Base):
    def __init__(self, alpha=0.05):
        super().__init__(robust=True, alpha=alpha)
"""


def test_ast_lane_kwarg_callee_bare_call_and_wrapper_exemption():
    tree = ast.parse(_SYNTH_MODULE)
    spans = _wrapper_spans(tree, {"stacked_did"})
    assert spans, "wrapper FunctionDef span must be detected"
    # kwarg at matching callee
    assert ast_call_hits("time", {"fit"}, tree, spans)
    # same kwarg, non-matching callee only
    assert not ast_call_hits("time", {"predict"}, tree, spans)
    # bare Name call (function rows) - found outside the wrapper
    assert ast_call_hits("bacon_decompose", set(), tree, spans, bare_call=True)
    # constructor rows include __init__ so super().__init__(tok=...) delegation
    # is a reader (regression pin: SyntheticDiD's robust=True)
    assert ast_call_hits("robust", {"Base", "__init__"}, tree, spans)
    assert not ast_call_hits("robust", {"Base"}, tree, spans), (
        "the class-name form alone must NOT see super().__init__ delegation - "
        "that blindness is why __init__ joins the constructor callee set"
    )
    # Regression: an inherited constructor called under the SUBCLASS name
    # is a reader once the subclass joins the callee set
    child_tree = ast.parse("est = Child(robust=True)")
    assert not ast_call_hits("robust", {"Base", "__init__"}, child_tree, [])
    assert ast_call_hits("robust", {"Base", "Child", "__init__"}, child_tree, [])
    # and the real ledger case: TwoWayFixedEffects/MultiPeriodDiD inherit
    # DifferenceInDifferences.__init__, so M-045's callee set names them
    m045_callees = _ast_callees(_ROW_INDEX["M-045"])
    assert {"DifferenceInDifferences", "TwoWayFixedEffects", "MultiPeriodDiD"} <= m045_callees
    # a module holding ONLY the wrapper: its interior call is exempt
    wrapper_only_src = "def stacked_did(data):\n    return bacon_decompose(data)\n"
    wtree = ast.parse(wrapper_only_src)
    wspans = _wrapper_spans(wtree, {"stacked_did"})
    assert not ast_call_hits(
        "bacon_decompose", set(), wtree, wspans, bare_call=True
    ), "calls inside a wrapper's own body are exempt"


def test_lifecycle_gate_windows_and_loud_keyerror():
    horizon_38 = _version_tuple(_NEXT_RELEASE[(3, 8)])
    assert _version_tuple("3.9") <= horizon_38, "3.9-window rows enforce at 3.8.x"
    assert _version_tuple("4.0") > horizon_38, "4.0-window rows defer at 3.8.x"
    horizon_39 = _version_tuple(_NEXT_RELEASE[(3, 9)])
    assert _version_tuple("4.0") <= horizon_39, "4.0-window rows arm at the 3.9 bump"
    assert (99, 99) not in _NEXT_RELEASE, "unknown versions must fail the gate loudly"
    current = _version_tuple(diff_diff.__version__)[:2]
    assert current in _NEXT_RELEASE, f"extend _NEXT_RELEASE for __version__ {diff_diff.__version__}"


def test_resolver_self_test():
    shape = _resolve_old_surface("diff_diff:DifferenceInDifferences.fit[time]", "SELF")
    assert shape[0] == "param" and shape[2] == "time"
    assert shape[1] is _unwrap_callable(
        inspect.getattr_static(diff_diff.DifferenceInDifferences, "fit")
    )
    # inherited constructor param resolves to the BASE's function (the sweep's key)
    twfe = _resolve_old_surface("diff_diff:TwoWayFixedEffects[robust]", "SELF")
    base = _resolve_old_surface("diff_diff:DifferenceInDifferences[robust]", "SELF")
    assert twfe[1] is base[1], "inherited __init__ must dedup onto the base"
    # field locator resolves to (class, attr)
    fshape = _resolve_old_surface("diff_diff:CallawaySantAnnaResults.overall_att", "SELF")
    assert fshape[0] == "attr" and fshape[2] == "overall_att"
    assert fshape[1] is diff_diff.CallawaySantAnnaResults


def test_state_read_lane_and_terminal_row_filter():
    """Regressions: getattr/hasattr literal reads are detected, and
    terminal rename rows stay in scope with their own refs discounted."""
    tree = ast.parse(
        'x = getattr(estimator, "robust", False)\n'
        'y = hasattr(results, "robust")\n'
        "z = getattr(estimator, name_var, False)\n"
    )
    assert ast_state_read_hits("robust", tree)
    assert not ast_state_read_hits("groups", tree), "non-matching literal must miss"
    assert not ast_state_read_hits(
        "robust", ast.parse("w = getattr(estimator, name_var, False)")
    ), "non-literal attribute names are unknowable and must not match"
    # mapping .get() reads are the same silent-default form
    assert ast_state_read_hits("robust", ast.parse('v = params.get("robust", False)'))
    assert not ast_state_read_hits("robust", ast.parse("v = params.get(key, False)"))
    synthetic = [
        {
            "id": "M-902",
            "kind": "param",
            "status": "removed",
            "old": "diff_diff:Foo.fit[dead_tok]",
            "new": "diff_diff:Foo.fit[live_tok]",
        },
        {
            "id": "M-903",
            "kind": "param",
            "status": "planned",
            "old": "diff_diff:Foo.fit[other_tok]",
            "new": "diff_diff:Foo.fit[other_new]",
        },
    ]
    terminal_ids = [r["id"] for r in _terminal_rename_rows(synthetic)]
    assert terminal_ids == ["M-902"], "removed rows stay in Duty C scope"
    # Regressions: the family-token parser handles param-value grammar
    # (M-086's `[type]=event` crashed _token_from_locator), and the live
    # union executes against the REAL ledger - the terminal path's exact
    # call - without failing.
    assert _safe_family_token(_ROW_INDEX["M-086"]) == "type"
    assert _safe_family_token({"old": None}) is None
    assert "diff_diff/linalg.py" in _live_family_code_refs("robust")
    # Regression: the raw-text bare-call lane honors wrapper spans
    wrapper_src = "def stacked_did(data):\n    return bacon_decompose(data)\n"
    assert bare_call_text_hits("bacon_decompose", wrapper_src)
    assert not bare_call_text_hits("bacon_decompose", wrapper_src, exempt_spans=[(1, 2)])


def test_predicate_binding_includes_deprecated_only_rows():
    """A deprecated-only rename (removed_in null) with a unique token must
    reach the binding check."""
    synthetic = [
        {
            "id": "M-900",
            "kind": "param",
            "status": "planned",
            "old": "diff_diff:Foo.fit[unique_dep_only_tok]",
            "new": "diff_diff:Foo.fit[whatever]",
            "deprecated_in": "3.9",
            "removed_in": None,
        },
        {
            "id": "M-901",
            "kind": "param",
            "status": "planned",
            "old": "diff_diff:Foo.fit[no_lifecycle_tok]",
            "new": "diff_diff:Foo.fit[other]",
            "deprecated_in": None,
            "removed_in": None,
        },
    ]
    bound = dict(_predicate_bound_tokens(synthetic))
    assert bound.get("M-900") == "unique_dep_only_tok"
    assert "M-901" not in bound, "rows with no lifecycle at all stay unbound"


def test_token_extraction_rule():
    assert _token_from_locator("diff_diff:StackedDiD[clean_control]", "T") == "clean_control"
    assert _token_from_locator("diff_diff:StackedDiDResults.clean_control", "T") == "clean_control"
    assert _token_from_locator("diff_diff:bacon_decompose", "T") == "bacon_decompose"
    assert _token_from_locator("diff_diff:CallawaySantAnnaResults.aggregate", "T") == "aggregate"
    assert _token_from_locator(None, "T") is None
    # the API-move family extracts to identical tokens and is skipped
    m020 = _ROW_INDEX["M-020"]
    assert _token_from_locator(m020["old"], "M-020") == _token_from_locator(m020["new"], "M-020")
