"""
Enforcement for the 4.0 deprecation ledger (``docs/v4-deprecations.yaml``).

Every ledger row records the lifecycle of one API surface in the 4.0 program
(``docs/v4-design.md`` section 11 is the normative schema). This module asserts,
on every run, that each row's ``status`` matches reality at HEAD:

- ``planned`` symbol rows: the old surface resolves and the new surface does
  NOT - so a PR that ships new surface without flipping its row fails CI (the
  tripwire that keeps the ledger honest through Phases 2-5).
- ``shimmed`` rows: both surfaces resolve and the row's dedicated behavioral
  test file exists (warning-emission itself is asserted THERE, not here - this
  module is pure introspection, no fits, no data).
- ``removed``/``done`` rows keep asserting forever (anti-resurrection guard).

FORWARD-REFERENCE NOTE: ``new`` locators on ``planned`` rows name 3.9/4.0
surface that intentionally does not exist yet; this module asserts its ABSENCE.

The release-cut sweep is AUTOMATED: ``test_due_rows_are_terminal`` fails any
version bump that reaches a row's scheduled version (``removed_in``, flip
version, ``decision_due``, or ``introduced_in``) while the row has not flipped
(spec section 9) - due rows must flip or be re-scheduled in the release PR.

Parsing is a purpose-built line scanner (restricted-YAML format contract in the
ledger header) because the project ships no PyYAML - same precedent as
``tests/test_doc_deps_integrity.py``. A dev-side dual-parse test keeps the file
honest as real YAML wherever PyYAML happens to be installed.
"""

import importlib
import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX = REPO_ROOT / "docs" / "v4-deprecations.yaml"
SPEC = REPO_ROOT / "docs" / "v4-design.md"

# Repo-integrity test: reads docs/ artifacts relative to the repo root. The installed-package CI
# matrix copies tests/ to a temp dir where docs/ is absent - nothing to validate there, so skip
# the whole module (same guard as tests/test_doc_deps_integrity.py).
if not MATRIX.exists():
    pytest.skip(
        f"{MATRIX} not found - v4-matrix enforcement runs only from the source tree, not against "
        "an installed package copied to a temp dir.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Schema vocabulary (normative home: docs/v4-design.md section 11)
# ---------------------------------------------------------------------------
SYMBOL_KINDS = {"param", "class", "field", "function"}
NONSYMBOL_KINDS = {
    "param-value",
    "alias",
    "default-flip",
    "env-default",
    "warning-retirement",
    "behavior",
}
KINDS = SYMBOL_KINDS | NONSYMBOL_KINDS
# param-value rows share the symbol LIFECYCLE (planned -> shimmed -> removed, test_ref at
# shim/removal, due-gated) but skip reality probes - accepted values are not introspectable;
# their behavioral enforcement lives in the row's test_ref suite.
LIFECYCLE_KINDS = SYMBOL_KINDS | {"param-value"}
SYMBOL_STATUSES = {"planned", "shimmed", "removed"}
NONSYMBOL_STATUSES = {"planned", "evaluate", "done"}
WARNING_VALUES = {"FutureWarning", "DeprecationWarning"}
KNOWN_FIELDS = {
    "id",
    "kind",
    "group",
    "old",
    "new",
    "introduced_in",
    "deprecated_in",
    "removed_in",
    "status",
    "phase",
    "warning",
    "test_ref",
    "code_refs",
    "notes",
    "old_default",
    "new_default",
    "snippet",
    "old_target",
    "new_target",
    "env_var",
    "decision_due",
    "decided_default",
}
REQUIRED_FIELDS = {
    "id",
    "kind",
    "group",
    "old",
    "new",
    "deprecated_in",
    "removed_in",
    "status",
    "phase",
}
_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")
_ID_RE = re.compile(r"^M-\d{3}$")
_ROW_START_RE = re.compile(r"^  - id:\s*(\S+)\s*$")
_FIELD_RE = re.compile(r"^    ([a-z_]+):\s*(.*?)\s*$")
_MD_TOKEN_RE = re.compile(r"\[(M-\d{3})\]")

# Row-count floor: 75 rows from the Phase 1 spec + diagnostic-family amendment,
# plus the 2 Phase 2a results-contract rows (M-092/M-093) = 77, plus the 3
# gating-completeness amendment rows (M-094..M-096) = 80, plus the 19-row
# completeness sweep over public FUNCTIONS and the dCDH results mirror
# (M-097..M-115) = 99. Ids are never reused and terminal rows are never
# deleted, so the ledger only grows - raise the floor when rows are added; a
# lower parse count means scanner/format drift or an illegal row deletion.
ROW_COUNT_FLOOR = 99

# Committed snapshot of the shipped id set ("ids are never deleted or reused"
# contract - a delete-one-add-one edit keeps the count above the floor but trips
# this). Extend with a NEW tuple, never edit an existing one. Ranges:
# (1,8)/(10,16)/(20,27)/(30,47)/(50,58)/(60,64)/(70,77)/(80,91) = Phase 1 + the
# diagnostic-family amendment; (92,93) = Phase 2a results-contract rows;
# (94,96) = the gating-completeness amendment (two results-field mirrors of
# existing param rows, plus the wild-bootstrap exposure policy);
# (97,115) = its completeness sweep over module-level public functions
# (twowayfeweights, three HAD pretest entry points, trim_weights) plus the
# dCDH results mirror and the fourth `robust` site.
_INITIAL_ID_RANGES = [
    (1, 8),
    (10, 16),
    (20, 27),
    (30, 47),
    (50, 58),
    (60, 64),
    (70, 77),
    (80, 91),
    (92, 93),
    (94, 96),
    (97, 115),
]
EXPECTED_INITIAL_IDS = frozenset(
    f"M-{n:03d}" for lo, hi in _INITIAL_ID_RANGES for n in range(lo, hi + 1)
)


def _parse_scalar(raw):
    """Decode one restricted-YAML scalar: quotes stripped, ``null`` -> None, flow list -> list."""
    if raw == "null" or raw == "":
        return None
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        return [] if not inner else [item.strip().strip("\"'") for item in inner.split(",")]
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def parse_matrix(text):
    """Parse the restricted-YAML ledger; returns (rows, errors).

    Format errors (unknown field, content outside a row, no ``rows:`` header) are collected
    rather than raised so schema tests can report them all at once.
    """
    rows, errors = [], []
    in_rows = False
    current = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not in_rows:
            if stripped == "rows:":
                in_rows = True
            continue
        m = _ROW_START_RE.match(line)
        if m:
            current = {"id": m.group(1)}
            rows.append(current)
            continue
        m = _FIELD_RE.match(line)
        if m and current is not None:
            key, raw = m.group(1), m.group(2)
            if key not in KNOWN_FIELDS:
                errors.append(f"line {lineno}: unknown field '{key}' (row {current['id']})")
                continue
            if key in current and key != "id":
                errors.append(f"line {lineno}: duplicate field '{key}' in row {current['id']}")
                continue
            current[key] = _parse_scalar(raw)
            continue
        errors.append(f"line {lineno}: unparseable line inside rows block: {stripped!r}")
    if not in_rows:
        errors.append("no top-level 'rows:' key found")
    return rows, errors


def validate_schema(rows):
    """Return a list of schema-violation strings (empty = valid)."""
    errors = []
    seen_ids = set()
    for row in rows:
        rid = row.get("id", "<missing>")
        if not _ID_RE.match(rid):
            errors.append(f"{rid}: id does not match M-###")
        if rid in seen_ids:
            errors.append(f"{rid}: duplicate id")
        seen_ids.add(rid)
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{rid}: missing required field(s) {sorted(missing)}")
            continue
        kind, status = row["kind"], row["status"]
        if kind not in KINDS:
            errors.append(f"{rid}: unknown kind '{kind}'")
            continue
        legal = SYMBOL_STATUSES if kind in LIFECYCLE_KINDS else NONSYMBOL_STATUSES
        if status not in legal:
            errors.append(
                f"{rid}: status '{status}' illegal for kind '{kind}' (legal: {sorted(legal)})"
            )
        for vfield in ("introduced_in", "deprecated_in", "removed_in", "decision_due"):
            val = row.get(vfield)
            if val is not None and not _VERSION_RE.match(str(val)):
                errors.append(f"{rid}: {vfield}={val!r} does not match \\d+.\\d+(.\\d+)?")
        if row.get("warning") is not None and row["warning"] not in WARNING_VALUES:
            errors.append(f"{rid}: warning={row['warning']!r} not in {sorted(WARNING_VALUES)}")
        if kind in LIFECYCLE_KINDS and status in ("shimmed", "removed") and not row.get("test_ref"):
            errors.append(
                f"{rid}: status '{status}' requires a test_ref (dedicated behavioral test)"
            )
        if not row.get("code_refs"):
            errors.append(f"{rid}: code_refs must be a non-empty list")
        if kind == "default-flip" and not (row.get("old_default") and row.get("new_default")):
            errors.append(f"{rid}: default-flip requires old_default and new_default")
        if kind == "warning-retirement" and not row.get("snippet"):
            errors.append(f"{rid}: warning-retirement requires a snippet")
        if kind == "env-default" and not row.get("env_var"):
            errors.append(f"{rid}: env-default requires env_var")
        if (
            kind == "env-default"
            and status == "done"
            and row.get("decided_default") not in ("on", "off")
        ):
            errors.append(
                f"{rid}: env-default at 'done' requires decided_default: on|off "
                "(evaluated-kept-off is a first-class outcome)"
            )
        if (
            kind in ("behavior", "default-flip", "env-default")
            and status == "done"
            and not row.get("test_ref")
        ):
            errors.append(
                f"{rid}: {kind} at 'done' requires a test_ref (semantic flips need "
                "ledger-linked behavioral evidence)"
            )
        if kind == "alias" and ("old_target" not in row or "new_target" not in row):
            errors.append(f"{rid}: alias requires old_target and new_target (nullable)")
        if kind == "alias" and row.get("warning") is not None:
            errors.append(
                f"{rid}: alias rows must not declare 'warning' - an alias is the same object "
                "as its target, so the deprecation warning rides the parent class row"
            )
        if (
            kind in ("class", "function")
            and row.get("removed_in") is not None
            and not str(row["old"]).startswith("diff_diff:")
        ):
            errors.append(
                f"{rid}: removable {kind} rows must use a top-level 'diff_diff:' locator "
                f"(survives module deletion), got {row['old']!r}"
            )
    return errors


# ---------------------------------------------------------------------------
# Locator resolution
# ---------------------------------------------------------------------------
_LOCATOR_RE = re.compile(r"^(?P<mod>[\w.]+)(?::(?P<attrs>[\w.]+))?(?:\[(?P<param>\w+)\])?$")


def _import_module_hard(modname, rid):
    """Module import is the typo guard: failure is ALWAYS a test failure, never a legal absence."""
    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover - exercised only on locator typos
        pytest.fail(f"{rid}: locator module '{modname}' failed to import ({exc}) - typo'd locator?")


def resolve_locator(locator, rid):
    """Resolve ``module[:Attr.chain][ [param] ]``; return (resolved: bool, detail: str).

    The module part must import (hard failure otherwise). Attribute-chain or parameter absence is
    the legal, assertable signal and returns (False, why).
    """
    m = _LOCATOR_RE.match(locator)
    if m is None or (m.group("param") and not m.group("attrs")):
        pytest.fail(f"{rid}: locator {locator!r} does not match the grammar (spec section 11)")
    obj = _import_module_hard(m.group("mod"), rid)
    attrs = (m.group("attrs") or "").split(".") if m.group("attrs") else []
    for i, attr in enumerate(attrs):
        nxt = (
            inspect.getattr_static(obj, attr, None)
            if not inspect.ismodule(obj) or i > 0
            else getattr(obj, attr, None)
        )
        if nxt is None:
            nxt = getattr(obj, attr, None)
        if nxt is None:
            return False, f"attribute '{attr}' absent on {'.'.join([m.group('mod')] + attrs[:i]) }"
        obj = nxt
    param = m.group("param")
    if param is None:
        return True, "resolved"
    target = obj
    if inspect.isclass(target):
        target = target.__init__
    if isinstance(target, (staticmethod, classmethod, property)):
        target = getattr(target, "__func__", getattr(target, "fget", target))
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"{rid}: cannot take signature of {locator!r} ({exc})")
    if param in sig.parameters:
        return True, "param present"
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.fail(
            f"{rid}: param '{param}' not found but {locator!r} has **kwargs - absence is "
            "unprovable; use a precise locator or a dedicated row kind (spec section 11)"
        )
    return False, f"param '{param}' absent"


def _top_level_export_name(locator):
    """Return the export name for a ``diff_diff:Name`` locator (no dots, no [param]); else None."""
    m = _LOCATOR_RE.match(locator)
    if m and m.group("mod") == "diff_diff" and m.group("attrs") and not m.group("param"):
        if "." not in m.group("attrs"):
            return m.group("attrs")
    return None


def _check_all_membership(all_names, name, expect_present):
    """Pure check: is ``name``'s presence in ``__all__`` consistent with expectation?"""
    return (name in all_names) is expect_present


def _locator_class_and_attr(locator, rid):
    """For field locators ``diff_diff:Class.attr``: return (class object, attr name)."""
    m = _LOCATOR_RE.match(locator)
    if m is None or not m.group("attrs") or m.group("param"):
        pytest.fail(f"{rid}: field locator {locator!r} must be 'module:Class.attr'")
    mod = _import_module_hard(m.group("mod"), rid)
    attrs = m.group("attrs").split(".")
    obj = mod
    for attr in attrs[:-1]:
        obj = getattr(obj, attr, None)
        if obj is None:
            pytest.fail(f"{rid}: class part of field locator {locator!r} does not resolve")
    return obj, attrs[-1]


def _signature_default(locator, rid):
    """Return repr(default) for a ``Class[param]`` / ``Class.method[param]`` locator."""
    m = _LOCATOR_RE.match(locator)
    if m is None or not m.group("param"):
        pytest.fail(f"{rid}: default-flip locator {locator!r} must carry a [param]")
    resolved, _ = resolve_locator(locator, rid)
    if not resolved:
        pytest.fail(f"{rid}: default-flip param in {locator!r} does not exist")
    mod = _import_module_hard(m.group("mod"), rid)
    obj = mod
    for attr in m.group("attrs").split("."):
        obj = getattr(obj, attr)
    if inspect.isclass(obj):
        obj = obj.__init__
    return repr(inspect.signature(obj).parameters[m.group("param")].default)


# ---------------------------------------------------------------------------
# Load the real ledger once
# ---------------------------------------------------------------------------
ROWS, _PARSE_ERRORS = parse_matrix(MATRIX.read_text())
_ROW_IDS = [r.get("id", "?") for r in ROWS]


def test_matrix_parses_cleanly():
    assert not _PARSE_ERRORS, "format errors in docs/v4-deprecations.yaml:\n" + "\n".join(
        _PARSE_ERRORS
    )


def test_matrix_parsed_nonempty():
    """Guard: a scanner/format break must not silently green the reality checks below."""
    assert len(ROWS) >= ROW_COUNT_FLOOR, (
        f"only {len(ROWS)} rows parsed (floor {ROW_COUNT_FLOOR}) - scanner/format drift? "
        "Rows are never deleted, so the ledger only grows."
    )


def test_matrix_schema_valid():
    errors = validate_schema(ROWS)
    assert not errors, "schema violations in docs/v4-deprecations.yaml:\n" + "\n".join(errors)


def test_spec_tokens_resolve():
    """Every [M-###] token in docs/v4-design.md must name a ledger row (anti-drift crossref)."""
    assert SPEC.exists(), "docs/v4-design.md missing while the ledger exists"
    tokens = set(_MD_TOKEN_RE.findall(SPEC.read_text()))
    known = set(_ROW_IDS)
    dangling = sorted(tokens - known)
    assert not dangling, f"spec cites ledger rows that do not exist: {dangling}"


@pytest.mark.parametrize("row", ROWS, ids=_ROW_IDS)
def test_code_refs_exist(row):
    """code_refs must exist for non-terminal rows; test_ref must exist at ANY status.

    Terminal rows skip only the code_refs check (the referenced code may legitimately be
    deleted); their removal-pin test file must survive forever, so a declared test_ref is
    always validated (verify-paths discipline).
    """
    tref = row.get("test_ref")
    if tref:
        assert (REPO_ROOT / tref).exists(), f"{row['id']}: test_ref '{tref}' does not exist"
    if row.get("status") in ("removed", "done"):
        return
    for ref in row.get("code_refs") or []:
        assert (REPO_ROOT / ref).exists(), f"{row['id']}: code_ref '{ref}' does not exist"


def _version_tuple(version):
    """Parse '3.9' / '4.0.0' (ignoring any local suffix) into a comparable int 3-tuple.

    Padded to exactly three components so '4.0' and '4.0.0' compare EQUAL - unpadded
    tuples would make (4, 0) < (4, 0, 0) and silently skip a due gate.
    """
    core = re.split(r"[+a-zA-Z]", str(version), maxsplit=1)[0].rstrip(".")
    parts = [int(p) for p in core.split(".")]
    return tuple(parts + [0] * (3 - len(parts)))[:3]


def collect_due_problems(rows, current):
    """Pure due-row sweep over parsed rows at version tuple ``current`` (unit-testable).

    Rules (spec section 11 release gate):
    - symbol rows past ``removed_in`` must be ``removed``; alias rows must be ``done``;
    - default-flip / warning-retirement / behavior rows past their flip version
      (``deprecated_in``) must be ``done``;
    - env-default rows past ``decision_due`` must be ``done`` (go/no-go recorded either way);
    - ANY row past ``introduced_in`` may no longer be ``planned`` OR ``evaluate`` (the new
      surface must have shipped - this is what gates introduce-only aliases and the Phase 2
      marker; evaluate cannot satisfy an introduction);
    - symbol rows that declare a ``warning`` and are past ``deprecated_in`` must no longer
      be ``planned`` (their shim must have shipped);
    - EARLY-REMOVAL GUARD: a row must NOT be terminal while its scheduled version
      (``removed_in``, or the flip version for non-symbol kinds) is still in the future -
      the shim window is a promise to users, not a suggestion.
    """
    problems = []
    for row in rows:
        rid, kind, status = row["id"], row["kind"], row["status"]
        removed_in = row.get("removed_in")
        if status in ("removed", "done"):
            early_version = removed_in
            if kind in ("default-flip", "warning-retirement", "behavior"):
                early_version = row.get("deprecated_in")
            elif kind == "env-default":
                early_version = row.get("decision_due")
            if early_version and current < _version_tuple(early_version):
                problems.append(
                    f"{rid}: status '{status}' but its scheduled version {early_version} is "
                    f"still in the future - early removal/flip breaks the shim window promise"
                )
        if removed_in and current >= _version_tuple(removed_in):
            if kind in LIFECYCLE_KINDS and status != "removed":
                problems.append(f"{rid}: removed_in {removed_in} is due but status is '{status}'")
            if kind == "alias" and status != "done":
                problems.append(f"{rid}: removed_in {removed_in} is due but status is '{status}'")
        if kind in ("default-flip", "warning-retirement", "behavior"):
            flip = row.get("deprecated_in")
            if flip and current >= _version_tuple(flip) and status != "done":
                problems.append(f"{rid}: flip version {flip} is due but status is '{status}'")
        decision_due = row.get("decision_due")
        if kind == "env-default" and decision_due and current >= _version_tuple(decision_due):
            if status != "done":
                problems.append(
                    f"{rid}: decision_due {decision_due} is due but status is '{status}' "
                    "(record the go/no-go either way)"
                )
        introduced_in = row.get("introduced_in")
        if (
            introduced_in
            and current >= _version_tuple(introduced_in)
            and status
            in (
                "planned",
                "evaluate",
            )
        ):
            problems.append(
                f"{rid}: introduced_in {introduced_in} is due but status is '{status}' - the "
                "new surface must have shipped (evaluate cannot satisfy an introduction)"
            )
        if (
            kind in LIFECYCLE_KINDS
            and row.get("warning")
            and row.get("deprecated_in")
            and current >= _version_tuple(row["deprecated_in"])
            and status == "planned"
        ):
            problems.append(
                f"{rid}: deprecated_in {row['deprecated_in']} is due but status is still "
                "'planned' (shim not shipped?)"
            )
    return problems


def test_due_rows_are_terminal():
    """Release gate: the automated form of the 4.0-cut sweep (spec section 9). A release
    bump whose version reaches a row's scheduled version fails CI unless the row flipped
    or was explicitly re-scheduled in the release PR."""
    import diff_diff

    problems = collect_due_problems(ROWS, _version_tuple(diff_diff.__version__))
    assert not problems, (
        f"rows due at version {diff_diff.__version__} have not flipped "
        "(flip them or re-schedule in the release PR):\n" + "\n".join(problems)
    )


def test_initial_ids_never_deleted():
    """The shipped id set is immutable: ids are never deleted or reused (spec section 11).

    ROW_COUNT_FLOOR alone would let a delete-one-add-one edit pass; this snapshot cannot.
    Extends as rows ship (99 as of the gating-completeness amendment: Phase 1 +
    diagnostic-family + M-092/M-093 + M-094..M-096 + the M-097..M-115
    public-function completeness sweep)."""
    missing = sorted(EXPECTED_INITIAL_IDS - set(_ROW_IDS))
    assert not missing, f"ledger rows deleted (ids are permanent): {missing}"
    assert len(EXPECTED_INITIAL_IDS) == 99


def test_version_tuple_pads_to_three_components():
    """'4.0' and '4.0.0' must compare EQUAL - unpadded tuples would skip due gates."""
    assert _version_tuple("4.0") == _version_tuple("4.0.0") == (4, 0, 0)
    assert _version_tuple("3.9") < _version_tuple("4.0.0")
    overdue = {
        "id": "M-905",
        "kind": "param",
        "old": "diff_diff:SyntheticDiD[lambda_reg]",
        "new": None,
        "deprecated_in": "3.0.0",
        "removed_in": "4.0.0",
        "status": "shimmed",
        "phase": 5,
    }
    # current version stated with two components, row with three: still due
    assert collect_due_problems([overdue], _version_tuple("4.0"))


def test_all_membership_helper_semantics():
    """Pure semantics of the __all__ consistency check used by the reality tests."""
    assert _check_all_membership(["A", "B"], "A", True)
    assert _check_all_membership(["A", "B"], "C", False)
    assert not _check_all_membership(["A", "B"], "Stale", True)  # missing but expected present
    assert not _check_all_membership(["A", "Stale"], "Stale", False)  # stale entry not removed


@pytest.mark.parametrize(
    ("row_overrides", "current", "expected_substring", "clean_at"),
    [
        # overdue introduce-only alias: introduced_in due but still planned
        (
            {
                "id": "M-901",
                "kind": "alias",
                "old": "SCM",
                "new": None,
                "old_target": None,
                "new_target": "diff_diff:SyntheticControl",
                "introduced_in": "3.9",
                "deprecated_in": None,
                "removed_in": None,
                "status": "planned",
                "phase": 2,
            },
            (3, 9, 0),
            "introduced_in 3.9 is due",
            (3, 8, 0),
        ),
        # overdue env-default evaluation: decision_due due but not done
        (
            {
                "id": "M-902",
                "kind": "env-default",
                "old": "diff_diff.linalg:_resolve_solve_ols_fastpath",
                "new": None,
                "deprecated_in": None,
                "removed_in": None,
                "decision_due": "4.0",
                "status": "evaluate",
                "phase": 5,
            },
            (4, 0, 0),
            "decision_due 4.0 is due",
            (3, 9, 0),
        ),
        # overdue removal: removed_in due but still shimmed
        (
            {
                "id": "M-903",
                "kind": "param",
                "old": "diff_diff:SyntheticDiD[lambda_reg]",
                "new": None,
                "deprecated_in": "3.0.0",
                "removed_in": "4.0",
                "status": "shimmed",
                "phase": 5,
            },
            (4, 0, 0),
            "removed_in 4.0 is due",
            (3, 9, 0),
        ),
        # introduced_in cannot be dodged via 'evaluate' (behavior-row bypass regression)
        (
            {
                "id": "M-907",
                "kind": "behavior",
                "old": "diff_diff:BaconDecompositionResults",
                "new": None,
                "introduced_in": "3.9",
                "deprecated_in": None,
                "removed_in": None,
                "status": "evaluate",
                "phase": 2,
            },
            (3, 9, 0),
            "evaluate cannot satisfy an introduction",
            (3, 8, 0),
        ),
        # overdue param-value removal: value migrations are due-gated like symbol rows
        (
            {
                "id": "M-906",
                "kind": "param-value",
                "old": "diff_diff:WooldridgeDiDResults.aggregate[type]=event",
                "new": "diff_diff:WooldridgeDiDResults.aggregate[type]=event_study",
                "deprecated_in": "3.9",
                "removed_in": "4.0",
                "status": "shimmed",
                "phase": 5,
            },
            (4, 0, 0),
            "removed_in 4.0 is due",
            (3, 9, 0),
        ),
        # EARLY removal: row flipped to removed while removed_in is still in the future
        (
            {
                "id": "M-904",
                "kind": "param",
                "old": "diff_diff:SyntheticDiD[lambda_reg]",
                "new": None,
                "deprecated_in": "3.0.0",
                "removed_in": "5.0",
                "status": "removed",
                "phase": 5,
            },
            (4, 0, 0),
            "still in the future",
            (5, 0, 0),
        ),
    ],
    ids=[
        "overdue-introduce-only-alias",
        "overdue-env-default-decision",
        "overdue-removal",
        "introduced-in-evaluate-bypass",
        "overdue-param-value-removal",
        "early-removal-before-schedule",
    ],
)
def test_due_gate_catches_overdue_rows(row_overrides, current, expected_substring, clean_at):
    """Negative coverage for the release gate: each bad shape must be reported at ``current``
    and clean at ``clean_at`` (the gate is version-driven, not static)."""
    problems = collect_due_problems([row_overrides], current)
    assert any(
        expected_substring in p for p in problems
    ), f"expected a due-gate problem containing {expected_substring!r}, got: {problems}"
    assert not collect_due_problems([row_overrides], clean_at)


@pytest.mark.parametrize("row", ROWS, ids=_ROW_IDS)
def test_row_matches_reality(row, monkeypatch):
    """The core enforcement: each row's status must match reality at HEAD (spec section 11)."""
    rid, kind, status = row["id"], row["kind"], row["status"]
    old, new = row["old"], row["new"]

    if kind == "param-value":
        pytest.skip(
            "no reality probe (accepted values are not introspectable); lifecycle is "
            "schema+due-gate enforced and value behavior lives in the row's test_ref suite"
        )
    if kind == "behavior":
        pytest.skip("schema-checked kind; flipped manually, swept at the cut")

    if kind in ("param", "class", "function"):
        if status == "planned":
            resolved, detail = resolve_locator(old, rid)
            assert resolved, f"{rid} planned but old {old!r} does not resolve ({detail})"
            if new is not None:
                resolved, detail = resolve_locator(new, rid)
                assert not resolved, (
                    f"{rid} planned but new {new!r} ALREADY resolves - the owning PR must flip "
                    f"this row to 'shimmed' in the same diff (docs/v4-design.md section 11)"
                )
        elif status == "shimmed":
            resolved, detail = resolve_locator(old, rid)
            assert resolved, f"{rid} shimmed but old {old!r} does not resolve ({detail})"
            if new is not None:
                resolved, detail = resolve_locator(new, rid)
                assert resolved, f"{rid} shimmed but new {new!r} does not resolve ({detail})"
        elif status == "removed":
            resolved, _ = resolve_locator(old, rid)
            assert not resolved, f"{rid} removed but old {old!r} still resolves (resurrection?)"
            if new is not None:
                resolved, detail = resolve_locator(new, rid)
                assert resolved, f"{rid} removed but new {new!r} does not resolve ({detail})"
        if _top_level_export_name(old) is not None:
            import diff_diff

            name = _top_level_export_name(old)
            expect = status != "removed"
            assert _check_all_membership(diff_diff.__all__, name, expect), (
                f"{rid}: '{name}' should {'be' if expect else 'NOT be'} in diff_diff.__all__ at "
                f"status '{status}' (stale __all__ entries break `from diff_diff import *`)"
            )

    elif kind == "field":
        cls, old_attr = _locator_class_and_attr(old, rid)
        fields = getattr(cls, "__dataclass_fields__", {})
        if status == "planned":
            assert old_attr in fields, f"{rid} planned but '{old_attr}' is not a dataclass field"
            if new is not None:
                _, new_attr = _locator_class_and_attr(new, rid)
                assert new_attr not in fields, (
                    f"{rid} planned but '{new_attr}' is ALREADY a dataclass field - flip the row "
                    "in the storage-flip PR (a property is expected and fine at this status)"
                )
        elif status == "shimmed":
            assert (
                old_attr not in fields
            ), f"{rid} shimmed but '{old_attr}' is still a dataclass field (flip not shipped?)"
            assert (
                inspect.getattr_static(cls, old_attr, None) is not None
            ), f"{rid} shimmed but '{old_attr}' does not resolve as a descriptor"
            if new is not None:
                _, new_attr = _locator_class_and_attr(new, rid)
                assert new_attr in fields, f"{rid} shimmed but '{new_attr}' is not the native field"
        elif status == "removed":
            assert (
                inspect.getattr_static(cls, old_attr, None) is None and old_attr not in fields
            ), f"{rid} removed but '{old_attr}' still present (resurrection?)"
        # Family checks are ADDITIVE to the status branches above (a plain `if`, never part of
        # the status elif chain - review round 6 caught the base assertions being skipped).
        if row.get("group") == "field-flip":
            # The flip is a FAMILY move (spec section 5): the headline pair is the assertable
            # anchor, but the whole canonical quintet and every overall_* sibling flip together.
            quintet = ("att", "se", "t_stat", "p_value", "conf_int")
            if status == "planned":
                for name in quintet:
                    assert inspect.getattr_static(cls, name, None) is not None, (
                        f"{rid}: canonical '{name}' no longer resolves on {cls.__name__} - the "
                        "3.x property surface regressed"
                    )
            elif status in ("shimmed", "removed"):
                # Keeps asserting at 'removed' too (post-5.0): terminal rows guard forever
                # against quintet regressions or deprecated siblings resurrecting as native.
                for name in quintet:
                    assert name in fields, (
                        f"{rid} {status} but canonical '{name}' is not a native dataclass field "
                        f"on {cls.__name__} - partial quintet migration"
                    )
                # Row-specific deprecated family, NOT a broad overall_* prefix match - other
                # overall_* estimand families (e.g. ContinuousDiD's overall_acrt*) are separate
                # estimands governed by their own rows, if any. Both sibling conventions are
                # covered: CS-style (overall_att -> overall_se) and ContinuousDiD-style
                # (overall_att -> overall_att_se).
                _, new_attr = _locator_class_and_attr(new, rid)
                family = {old_attr}
                for canonical in quintet[1:]:
                    family.add(old_attr.replace(new_attr, canonical, 1))
                    family.add(f"{old_attr}_{canonical}")
                leftovers = sorted(f for f in fields if f in family)
                assert not leftovers, (
                    f"{rid} {status} but deprecated siblings remain native fields on "
                    f"{cls.__name__}: {leftovers} - the family flips together"
                )

    elif kind == "alias":
        import diff_diff

        if status in ("planned", "evaluate"):
            if row["old_target"] is None:
                assert not hasattr(diff_diff, old), (
                    f"{rid} introduce-only alias '{old}' ALREADY exists - flip the row in the "
                    "introducing PR"
                )
                alias_in_all = False
            else:
                target_resolved = _resolve_alias_target(row["old_target"], rid)
                assert (
                    getattr(diff_diff, old, None) is target_resolved
                ), f"{rid}: alias '{old}' is not identical to {row['old_target']}"
                alias_in_all = True
        else:  # done
            if row["new_target"] is None:
                assert not hasattr(diff_diff, old), f"{rid} done but alias '{old}' still exists"
                alias_in_all = False
            else:
                target_resolved = _resolve_alias_target(row["new_target"], rid)
                assert (
                    getattr(diff_diff, old, None) is target_resolved
                ), f"{rid} done but alias '{old}' does not point at {row['new_target']}"
                alias_in_all = True
        assert _check_all_membership(diff_diff.__all__, old, alias_in_all), (
            f"{rid}: alias '{old}' should {'be' if alias_in_all else 'NOT be'} in "
            f"diff_diff.__all__ at status '{status}'"
        )

    elif kind == "default-flip":
        expected = row["old_default"] if status in ("planned", "evaluate") else row["new_default"]
        actual = _signature_default(old, rid)
        assert (
            actual == expected
        ), f"{rid}: {old!r} default is {actual}, expected {expected} at status '{status}'"

    elif kind == "env-default":
        monkeypatch.delenv(row["env_var"], raising=False)
        m = _LOCATOR_RE.match(old)
        mod = _import_module_hard(m.group("mod"), rid)
        resolver = getattr(mod, m.group("attrs"), None)
        assert resolver is not None, f"{rid}: resolver {old!r} does not resolve"
        result = bool(resolver())
        if status == "done":
            # decided_default records the go/no-go outcome ("on" = flipped, "off" =
            # evaluated and kept off - both are first-class terminal states).
            expected = row.get("decided_default") == "on"
        else:
            expected = False
        assert result is expected, (
            f"{rid}: {old!r}() with {row['env_var']} unset returned {result}, expected {expected} "
            f"at status '{status}' (decided_default={row.get('decided_default')!r})"
        )

    elif kind == "warning-retirement":
        text = (REPO_ROOT / row["code_refs"][0]).read_text()
        present = row["snippet"] in text
        if status in ("planned", "evaluate"):
            assert present, (
                f"{rid}: snippet {row['snippet']!r} not found in {row['code_refs'][0]} - if the "
                "message was reworded, update this row's snippet in the same PR"
            )
        else:
            assert not present, f"{rid} done but snippet still present in {row['code_refs'][0]}"


def _resolve_alias_target(locator, rid):
    m = _LOCATOR_RE.match(locator)
    if m is None or not m.group("attrs") or m.group("param"):
        pytest.fail(f"{rid}: alias target {locator!r} must be 'module:Name'")
    mod = _import_module_hard(m.group("mod"), rid)
    obj = mod
    for attr in m.group("attrs").split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            pytest.fail(f"{rid}: alias target {locator!r} does not resolve")
    return obj


# ---------------------------------------------------------------------------
# Scanner/validator self-tests (committed negative coverage - a regression in
# the scanner must not silently green the whole matrix)
# ---------------------------------------------------------------------------
_BASE_ROW = (
    "rows:\n"
    "  - id: M-900\n"
    "    kind: param\n"
    "    group: fixture\n"
    '    old: "diff_diff:DifferenceInDifferences.fit[time]"\n'
    "    new: null\n"
    '    deprecated_in: "3.9"\n'
    '    removed_in: "4.0"\n'
    "    status: planned\n"
    "    phase: 2\n"
    "    code_refs: [diff_diff/estimators.py]\n"
)


def _schema_errors_for(text):
    rows, parse_errors = parse_matrix(text)
    return parse_errors + validate_schema(rows)


@pytest.mark.parametrize(
    ("mutation", "expected_substring"),
    [
        (lambda t: t.replace("    kind: param\n", "    kindd: param\n"), "unknown field"),
        (lambda t: t + _BASE_ROW.split("rows:\n")[1], "duplicate id"),
        (lambda t: t.replace("    status: planned\n", "    status: done\n"), "illegal for kind"),
        (lambda t: t.replace("    status: planned\n", ""), "missing required field"),
        (lambda t: t.replace('"4.0"', '"four"'), "does not match"),
        (
            lambda t: t.replace("    status: planned\n", "    status: shimmed\n"),
            "requires a test_ref",
        ),
        (
            lambda t: t.replace("    kind: param\n", "    kind: class\n").replace(
                '"diff_diff:DifferenceInDifferences.fit[time]"', '"diff_diff.estimators:Foo"'
            ),
            "top-level 'diff_diff:' locator",
        ),
        (
            lambda t: t.replace("    code_refs: [diff_diff/estimators.py]\n", ""),
            "code_refs must be a non-empty list",
        ),
        (lambda t: t.replace("    group: fixture\n", ""), "missing required field"),
        (
            lambda t: t.replace("    kind: param\n", "    kind: param-value\n").replace(
                "    status: planned\n", "    status: shimmed\n"
            ),
            "requires a test_ref",
        ),
        (
            lambda t: t.replace("    kind: param\n", "    kind: behavior\n").replace(
                "    status: planned\n", "    status: done\n"
            ),
            "ledger-linked behavioral evidence",
        ),
    ],
    ids=[
        "unknown-field",
        "duplicate-id",
        "status-illegal-for-kind",
        "missing-required",
        "bad-version",
        "shimmed-without-test-ref",
        "dotted-locator-on-removable",
        "empty-code-refs",
        "missing-group",
        "param-value-shimmed-without-test-ref",
        "behavior-done-without-test-ref",
    ],
)
def test_schema_rejects_bad_rows(mutation, expected_substring):
    errors = _schema_errors_for(mutation(_BASE_ROW))
    assert any(
        expected_substring in e for e in errors
    ), f"expected a schema error containing {expected_substring!r}, got: {errors}"


def test_base_fixture_row_is_clean():
    """The mutation baseline itself must be schema-clean, or the negative tests prove nothing."""
    assert _schema_errors_for(_BASE_ROW) == []


def test_dual_parse_against_real_yaml_when_available():
    """Dev-only: wherever PyYAML happens to be installed, the purpose-built scanner must agree
    with a real YAML load (CI does not install PyYAML - skipped there by design)."""
    yaml = pytest.importorskip("yaml")
    loaded = yaml.safe_load(MATRIX.read_text())
    assert isinstance(loaded, dict) and "rows" in loaded, "ledger is not valid YAML"
    real_ids = [r["id"] for r in loaded["rows"]]
    assert real_ids == _ROW_IDS, "scanner row ids diverge from yaml.safe_load - format drift"
