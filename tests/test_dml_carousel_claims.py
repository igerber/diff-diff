"""Claim-sync guard for the DMLDiD launch carousel.

The deck (``carousel/generate_dml_carousel.py``) hardcodes committed,
seed-locked numbers from the executed tutorial-32 notebook as module
constants. The synchronization chain is:

- library <-> notebook: tutorial 32 is numbers-locked and executed against
  the committed library (``tests/test_t32_dml_did_drift.py`` re-derives the
  estimates and hash-pins the code cells);
- notebook <-> carousel: THIS file parses the deck's constants (via ``ast``
  - the generator imports ``fpdf2``, a carousel-only dependency deliberately
  absent from CI) and locates each on the COMMITTED NOTEBOOK SURFACE, not
  restated here. If the tutorial is ever retuned and the carousel is
  forgotten, the notebook surface changes and this file fails.
- REGISTRY <-> carousel: the DoubleML parity strip renders the diff the
  REGISTRY DR-score families note records for the committed spike.

Skips cleanly when ``carousel/`` or ``docs/`` is absent (the
isolated-install CI jobs copy only ``tests/``).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tests._tutorial_drift import notebook_markdown, notebook_output_text

GENERATOR = Path(__file__).resolve().parents[1] / "carousel" / "generate_dml_carousel.py"
REGISTRY = Path(__file__).resolve().parents[1] / "docs" / "methodology" / "REGISTRY.md"
NB = "docs/tutorials/32_dml_did.ipynb"

TOTAL_SLIDES = 11

# Constant values the code slide renders via f-strings / str() - resolved
# here so the snippet reconstruction stays a pure-AST operation.
DECK_LITERALS = {"CODE_LEARNER": "sieve", "CODE_FOLDS": 5, "CODE_SEED": 42}


@pytest.fixture(scope="module")
def deck_constants():
    if not GENERATOR.exists():
        pytest.skip(f"{GENERATOR} not available in this CI environment.")
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    consts[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(target, ast.Tuple) and all(
                isinstance(el, ast.Name) for el in target.elts
            ):
                try:
                    values = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                if isinstance(values, tuple) and len(values) == len(target.elts):
                    for el, val in zip(target.elts, values):
                        consts[el.id] = val
    return consts


@pytest.fixture(scope="module")
def visible_strings():
    """Slide copy: every string literal in the generator EXCEPT docstrings.

    f-strings contribute their constant fragments (``ast.Constant`` nodes
    inside ``ast.JoinedStr``), so phrase pins must not span a ``{...}``
    substitution.
    """
    if not GENERATOR.exists():
        pytest.skip(f"{GENERATOR} not available in this CI environment.")
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))
    return " ".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    )


# Chart helpers attributed to the slide that places them.
_SLIDE_HELPERS = {
    "slide_01_cover": ("_render_cover_motif",),
    "slide_06_math": ("_render_score_equation",),
    "slide_07_payoff": ("_render_payoff_chart",),
}


def _per_slide(collect):
    if not GENERATOR.exists():
        pytest.skip(f"{GENERATOR} not available in this CI environment.")
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    per_func = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            per_func[node.name] = collect(node)
    slides = {}
    for name, value in per_func.items():
        if name.startswith("slide_"):
            for helper in _SLIDE_HELPERS.get(name, ()):
                value = value + per_func.get(helper, type(value)())
            slides[name] = value
    assert (
        len(slides) == TOTAL_SLIDES
    ), f"expected {TOTAL_SLIDES} slide_* methods, found {sorted(slides)}"
    return slides


@pytest.fixture(scope="module")
def slide_strings():
    """Per-slide visible strings (a qualifier must live ON the slide that
    makes the claim, not merely somewhere in the module)."""

    def collect(node):
        return " ".join(
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        )

    return _per_slide(collect)


@pytest.fixture(scope="module")
def slide_constant_refs():
    """Per-slide referenced module-level names (each pinned statistical
    constant must actually be RENDERED by its intended slide - a hardcoded
    literal bypassing the constant would otherwise pass)."""

    def collect(node):
        return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]

    return _per_slide(collect)


@pytest.fixture(scope="module")
def nb_output():
    return notebook_output_text(NB)


def _notebook_code(rel_path: str) -> str:
    nb_path = Path(__file__).resolve().parents[1] / rel_path
    if not nb_path.exists():
        pytest.skip(f"Notebook not found at {nb_path}.")
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in json.loads(nb_path.read_text())["cells"]
        if c["cell_type"] == "code"
    )


class TestEstimatesMatchNotebook:
    """Every estimate on the deck is on the committed tutorial-32 surface."""

    def test_truth_and_sieve_lines(self, deck_constants, nb_output):
        c = deck_constants
        att, se = c["SIEVE"]
        for needle in (
            f"DGP-implied overall ATT: {c['TRUTH']:.4f}",
            f"DMLDiD (sieve) estimate: {att:.4f} +/- {se:.4f}",
        ):
            assert needle in nb_output, f"deck value not on tutorial-32 surface: {needle!r}"

    def test_learner_table_values(self, deck_constants, nb_output):
        # The notebook renders the comparison as a pandas table; pin each
        # deck (ATT, SE) pair to ITS OWN ROW (label + ATT + SE in order on
        # one line), so swapped labels or a mislabeled constant cannot pass
        # on mere presence of the numbers (review round 1 P2).
        import re

        rows = {
            "LINEAR": "linear",
            "RIDGE": "ridge",
            "SIEVE": "sieve",
            "POLY": r"PolynomialRidge \(custom\)",
        }
        for key, label in rows.items():
            att, se = deck_constants[key]
            pattern = rf"{label}\s+{att:.4f}\s+{se:.4f}"
            assert re.search(
                pattern, nb_output
            ), f"{key}: no notebook table row matches {pattern!r}"

    def test_dgp_shape_values(self, deck_constants, nb_output):
        c = deck_constants
        md = notebook_markdown(NB)
        assert f"{c['N_NEVER']} never-treated units" in md
        # cohort sizes render in the DGP cell's value_counts output
        assert str(c["N_NEVER"]) in nb_output
        code = _notebook_code(NB)
        assert f"n_units, periods = {c['N_UNITS']}," in code
        # exact cohort-construction expression (review round: a bare
        # str(g)-in-code fallback was vacuous - single digits occur
        # everywhere)
        assert f"np.where(early, {c['COHORTS'][0]}, {c['COHORTS'][1]})" in code

    def test_survey_provenance_line(self, deck_constants, nb_output):
        c = deck_constants
        needle = (
            f"design df:  {c['SURVEY_DF']} ({c['SURVEY_PSU']} PSUs - "
            f"{c['SURVEY_STRATA']} strata)"
        )
        assert needle in nb_output, f"deck value not on tutorial-32 surface: {needle!r}"

    def test_five_se_claim_is_arithmetic(self, deck_constants):
        # "Almost five standard errors off" must follow from the pinned
        # values (a distance in reported-SE units, mirroring the tutorial's
        # own wording - never presented as valid coverage).
        c = deck_constants
        att, se = c["LINEAR"]
        ratio = abs(att - c["TRUTH"]) / se
        assert 4.0 < ratio < 5.0, ratio

    def test_recovery_claim_is_arithmetic(self, deck_constants):
        # The payoff headline ("Flexible outcome model: truth.") rests on
        # sieve/custom landing within ~1 reported SE of the DGP truth.
        c = deck_constants
        for key in ("SIEVE", "POLY"):
            att, se = c[key]
            assert abs(att - c["TRUTH"]) / se < 1.05, key


class TestRegistrySync:
    def test_doubleml_diff_matches_registry(self, deck_constants):
        if not REGISTRY.exists():
            pytest.skip("REGISTRY.md not available in this CI environment.")
        text = REGISTRY.read_text(encoding="utf-8")
        assert (
            f"ATT diff {deck_constants['DOUBLEML_ATT_DIFF']}" in text
        ), "the deck's DoubleML parity figure is not the REGISTRY's"
        assert "chang_case1_parity.py" in text


class TestGuardrailPhrases:
    def test_qualifiers_are_slide_local(self, slide_strings):
        expectations = {
            # the cover motif is stylized art, labeled on-slide
            "slide_01_cover": ("illustrative",),
            # the receipt states simulation + known truth WITH the numbers,
            # labels uncertainty as reported/nominal, and carries the
            # rate-condition caveat locally (review round 1 P1)
            "slide_03_receipt": (
                "truth known by construction",
                "simulated",
                "reported SE ",
                "narrow nominal interval",
                "nominal, not ",
            ),
            # the twist repeats the nominal qualifier locally
            "slide_04_twist": ("nominal precision", "nominal - illustrative"),
            # the math slide scopes orthogonality to the learned nuisances
            # and distinguishes the summand from the centered score
            "slide_06_math": (
                "LEARNED nuisances g and l",
                "psi = s - ATT",
                "own variance correction",
            ),
            # the payoff carries the DR beat + the illustrative-inference
            # qualifier (the tutorial's rate-condition caveat) + simulation
            "slide_07_payoff": (
                "misspecified in every arm",
                "illustrative",
                "known\nby construction",
                "Seed-locked simulated example",
            ),
            # the parity strip scopes the DoubleML anchor to the committed
            # Case 1 spike and labels staggered/survey as extensions
            "slide_10_production": (
                "committed parity spike",
                "documented extensions of the paper",
                "Chang Case 1 score",
                # A2.3: fresh samples must come from the SAME target
                # population (review round 1 P1)
                "SAME\ntarget population each wave",
            ),
        }
        for slide, phrases in expectations.items():
            for phrase in phrases:
                assert phrase in slide_strings[slide], (
                    f"qualifier {phrase!r} is not on {slide} - it must ride "
                    "with the claim it scopes"
                )

    def test_statistical_constants_reach_their_slides(self, slide_constant_refs):
        expectations = {
            "slide_03_receipt": ("LINEAR", "TRUTH", "N_UNITS", "N_PERIODS", "N_NEVER", "COHORTS"),
            "slide_04_twist": ("RIDGE",),
            "slide_07_payoff": ("LINEAR", "RIDGE", "SIEVE", "POLY", "TRUTH"),
            "slide_09_code": ("CODE_LEARNER", "CODE_FOLDS", "CODE_SEED"),
            "slide_10_production": (
                "SURVEY_PSU",
                "SURVEY_STRATA",
                "SURVEY_DF",
                "DOUBLEML_ATT_DIFF",
            ),
        }
        for slide, names in expectations.items():
            for name in names:
                assert name in slide_constant_refs[slide], (
                    f"constant {name} is not rendered by {slide} - a "
                    "hardcoded literal may have replaced the synced value"
                )

    def test_no_absolutes_or_competitor_claims(self, visible_strings):
        # CS is named neutrally/honestly (slide 8 RECOMMENDS it for the
        # parametric case); DoubleML appears only as the parity anchor. No
        # superlatives, no external-competitor comparisons, no coverage
        # overclaims for the deliberately misspecified demo.
        lowered = visible_strings.lower()
        for banned in (
            "the only ",
            "first and only",
            "no other estimator",
            "nothing else",
            "better than",
            "state of the art",
            "state-of-the-art",
            "game changer",
            "revolutionary",
            "econml",
            "geolift",
            "causalimpact",
            "csdid",
            "unbiased",  # the deck shows recovery, never claims unbiasedness
            "valid coverage",
            "guarantees",
            "in each nuisance",  # orthogonality covers g and l, not p
            "confident ci",
            "same confidence",
        ):
            assert banned not in lowered, f"banned absolute/competitive claim: {banned!r}"

    def test_no_verbatim_pull_quote(self, slide_strings):
        # The Chang paper review is pinned to the arXiv layout and the
        # published PDF has not been cross-checked word-for-word, so the
        # paper slide must carry NO quotation-marked pull quote (module
        # docstring records the decision).
        paper = slide_strings["slide_05_paper"]
        for ch in ('"', "“", "”"):
            assert ch not in paper, "the paper slide must not carry a quoted pull quote"

    def test_cs_is_a_real_exported_estimator(self):
        init = Path(__file__).resolve().parents[1] / "diff_diff" / "__init__.py"
        if not init.exists():
            pytest.skip("diff_diff/__init__.py not available in this CI environment.")
        assert "CallawaySantAnna" in init.read_text(encoding="utf-8")

    def test_code_slide_fragments_match_notebook_invocation(self, deck_constants, visible_strings):
        # Slide 9's fit is the tutorial's own configuration; the displayed
        # learner/folds/seed render from the deck constants, so
        # reconstructing the fragments pins deck <-> notebook.
        code = _notebook_code(NB)
        c = deck_constants
        for fragment in (
            'base_period="universal"',
            f'outcome_learner="{c["CODE_LEARNER"]}"',
            f"n_folds={c['CODE_FOLDS']},",
            f"seed={c['CODE_SEED']},",
            'covariates=["x1", "x2"]',
            'aggregate("event_study")',
        ):
            assert fragment in code, f"tutorial 32 lost the displayed invocation: {fragment!r}"
        assert "outcome_learner=" in visible_strings
        # the bring-your-own swap line renders a real sklearn constructor
        assert "GradientBoostingRegressor()" in visible_strings

    def test_builtin_learner_claim_is_dependency_free(self):
        # Slide 8 says the four built-in learners need "no extra installs":
        # diff_diff/_learners.py must import no external ML package.
        learners = Path(__file__).resolve().parents[1] / "diff_diff" / "_learners.py"
        if not learners.exists():
            pytest.skip("diff_diff/_learners.py not available in this CI environment.")
        tree = ast.parse(learners.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"sklearn", "statsmodels", "xgboost", "lightgbm", "torch"}, imported

    def test_sklearn_contract_claim_is_documented(self):
        # Slide 8/9 say scikit-learn estimators fit the duck-typed
        # contract - the API docs must actually document that claim.
        rst = Path(__file__).resolve().parents[1] / "docs" / "api" / "dml_did.rst"
        if not rst.exists():
            pytest.skip("docs/api/dml_did.rst not available in this CI environment.")
        assert "scikit-learn" in rst.read_text(encoding="utf-8")

    def test_module_docstring_states_both_contracts(self):
        # The review-round P1: the guard below excludes docstrings, so the
        # module docstring itself must also carry the classifier contract
        # (predict_proba), never the bare fit()/predict() shorthand.
        if not GENERATOR.exists():
            pytest.skip(f"{GENERATOR} not available in this CI environment.")
        tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree) or ""
        assert "predict_proba()" in doc
        assert "CLASSIFIERS" in doc or "classifier" in doc.lower()

    def test_flexibility_band_is_slide_local(self, slide_strings):
        # The user-emphasized flexibility beats (2026-08-29) live on their
        # slides: built-ins + no-extra-installs on slide 8, the sklearn
        # contract line on the code slide.
        assert "no extra installs" in slide_strings["slide_08_when"]
        assert "scikit-learn regressors already fit the" in slide_strings["slide_08_when"]
        # the propensity contract is predict_proba, not predict - both
        # contracts must be stated (review round 2 P2)
        assert "predict_proba()" in slide_strings["slide_08_when"]
        assert "classifiers" in slide_strings["slide_08_when"]
        assert "sklearn fits the contract" in slide_strings["slide_09_code"]

    def test_code_slide_snippet_is_valid_python(self):
        # Review round 3 P2: the displayed fit call was invalid Python
        # (`fit(df, outcome="y", ..., covariates=...)` - positional
        # ellipsis after keyword arguments) under a caption claiming the
        # shown call reproduces the shown numbers. Reconstruct the snippet
        # from the slide's token lines and require it to parse.
        if not GENERATOR.exists():
            pytest.skip(f"{GENERATOR} not available in this CI environment.")
        tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
        code_list = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "slide_09_code":
                for stmt in ast.walk(node):
                    if (
                        isinstance(stmt, ast.Assign)
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == "code"
                    ):
                        code_list = stmt.value
        assert code_list is not None, "slide_09_code lost its `code` token-line list"

        def _frag(nodeval):
            # token text: plain Constant or an f-string of Constants/Names
            if isinstance(nodeval, ast.Constant):
                return str(nodeval.value)
            if isinstance(nodeval, ast.JoinedStr):
                out = []
                for v in nodeval.values:
                    if isinstance(v, ast.Constant):
                        out.append(str(v.value))
                    elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name):
                        out.append(str(DECK_LITERALS[v.value.id]))
                return "".join(out)
            if isinstance(nodeval, ast.Call):  # str(CONST)
                arg = nodeval.args[0]
                if isinstance(arg, ast.Name):
                    return str(DECK_LITERALS[arg.id])
            raise AssertionError(f"unrecognized token node: {ast.dump(nodeval)[:80]}")

        lines = []
        for line in code_list.elts:
            assert isinstance(line, ast.List)
            lines.append("".join(_frag(tok.elts[0]) for tok in line.elts))
        snippet = "\n".join(lines)
        ast.parse(snippet)  # must be syntactically valid Python
        assert 'base_period="universal"' in snippet  # the tutorial fit's config
        assert 'unit="unit"' in snippet and 'first_treat="first_treat"' in snippet

    def test_tutorial_referenced_once_on_cta(self, visible_strings):
        assert visible_strings.count("Tutorial 32") == 1
        assert "tutorials/32_dml_did.html" in visible_strings
        nb_path = Path(__file__).resolve().parents[1] / NB
        assert nb_path.exists(), "CTA points to a tutorial that does not exist"
