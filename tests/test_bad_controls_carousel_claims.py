"""Claim-sync guard for the bad-controls launch carousel.

The deck (``carousel/generate_bad_controls_carousel.py``) hardcodes
committed, seed-locked numbers from the executed tutorial-33 notebook, the
paper's application numbers as transcribed in the paper review, and two
validation anchors from the REGISTRY / CHANGELOG as module constants. The
synchronization chain is:

- library <-> notebook: tutorial 33 is numbers-locked and executed against
  the committed library (``tests/test_t33_bad_controls_drift.py`` re-derives
  the estimates and hash-pins the code cells);
- notebook <-> carousel: THIS file parses the deck's constants (via ``ast``
  - the generator imports ``fpdf2``, a carousel-only dependency deliberately
  absent from CI) and locates each on the COMMITTED NOTEBOOK SURFACE, not
  restated here. If the tutorial is ever retuned and the carousel is
  forgotten, the notebook surface changes and this file fails;
- paper review <-> carousel: the NLSY79 application numbers on slide 10 are
  the authors' estimates as the review transcribes them (never reproduced
  by the library; the slide says so);
- CHANGELOG / REGISTRY <-> carousel: the cover's "New in 3.12" names the
  release whose CHANGELOG section ships the lane; the validation strip's
  oracle tolerance and the R package name come from the REGISTRY block.

Skips cleanly when ``carousel/`` or ``docs/`` is absent (the
isolated-install CI jobs copy only ``tests/``).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from tests._tutorial_drift import notebook_markdown, notebook_output_text

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "carousel" / "generate_bad_controls_carousel.py"
REGISTRY = ROOT / "docs" / "methodology" / "REGISTRY.md"
REVIEW = ROOT / "docs" / "methodology" / "papers" / "caetano-2026-review.md"
CHANGELOG = ROOT / "CHANGELOG.md"
NB = "docs/tutorials/33_bad_controls.ipynb"

TOTAL_SLIDES = 11

# Constant values the code slide renders via f-strings / str() - resolved
# here so the snippet reconstruction stays a pure-AST operation.
DECK_LITERALS = {
    "CODE_SEED": 0,
    "CODE_BAD_CONTROL": "x",
    "CODE_W": "w",
    "CODE_Z": "z",
    "CODE_CONTROL_GROUP": "not_yet_treated",
}


def _skip_unless(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not available in this CI environment.")


@pytest.fixture(scope="module")
def deck_constants():
    _skip_unless(GENERATOR)
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
    _skip_unless(GENERATOR)
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


# Chart / table helpers attributed to the slide that places them.
_SLIDE_HELPERS = {
    "slide_01_cover": ("_render_cover_motif",),
    "slide_07_route2": ("_render_payoff_chart",),
    "slide_08_pretest": ("_draw_att_x_table",),
}


def _per_slide(collect):
    _skip_unless(GENERATOR)
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


@pytest.fixture(scope="module")
def nb_markdown():
    return notebook_markdown(NB)


@pytest.fixture(scope="module")
def review_text():
    _skip_unless(REVIEW)
    return REVIEW.read_text(encoding="utf-8")


def _notebook_code(rel_path: str) -> str:
    nb_path = ROOT / rel_path
    if not nb_path.exists():
        pytest.skip(f"Notebook not found at {nb_path}.")
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in json.loads(nb_path.read_text())["cells"]
        if c["cell_type"] == "code"
    )


class TestEstimatesMatchNotebook:
    """Every simulated estimate on the deck is on the committed tutorial-33
    surface (prose AND rendered outputs)."""

    @pytest.mark.parametrize("name", ["TWFE_INCLUDE", "CS_A1", "CS_A1_W", "LANE_W"])
    def test_headline_estimates(self, deck_constants, nb_markdown, nb_output, name):
        att, se = deck_constants[name]
        # The tutorial quotes each as "**att ± se**" in prose ...
        assert f"**{att:.4f} ± {se:.4f}**" in nb_markdown, f"{name} not quoted in tutorial prose"
        # ... and renders both in the running results table.
        assert f"{att:.4f}" in nb_output and f"{se:.4f}" in nb_output

    def test_att_x_rows(self, deck_constants, nb_output, nb_markdown):
        rows = deck_constants["ATT_X_ROWS"]
        assert len(rows) == 6
        for g, t, att_x, se_x in rows:
            # bad_control_summary() prints one row per (g, t) with both columns
            # columns: group, time, post, att_x, se_x, ...
            pattern = (
                rf"\b{g}\s+{t}\s+(?:True|False)\s+"
                rf"{re.escape(f'{att_x:.4f}')}\s+{re.escape(f'{se_x:.4f}')}"
            )
            assert re.search(pattern, nb_output), f"ATT_X row ({g},{t}) not in the summary output"
        pre = [r for r in rows if r[1] < r[0]]
        post = [r for r in rows if r[1] >= r[0]]
        assert len(pre) == 3 and len(post) == 3
        for _g, _t, att_x, _se in pre + post:
            assert f"**{att_x:.4f}**" in nb_markdown, f"{att_x:.4f} not quoted in tutorial prose"

    def test_truth_values(self, deck_constants, nb_markdown):
        assert deck_constants["TRUTH"] == 1.0 and deck_constants["TRUTH_X"] == 0.5
        assert r"\mathbf{1.00}" in nb_markdown and r"\mathbf{0.50}" in nb_markdown

    def test_dgp_shape_values(self, deck_constants, nb_markdown):
        c = deck_constants
        code = _notebook_code(NB)
        assert f"n, T = {c['N_UNITS']}, {c['N_PERIODS']}" in code
        assert f"**{c['N_NEVER']}**" in nb_markdown
        for g, size in c["COHORT_SIZES"]:
            assert f"**{size}** ($g = {g}$)" in nb_markdown

    def test_paper_s2_bias_is_in_tutorial_and_review(
        self, deck_constants, nb_markdown, review_text
    ):
        bias = deck_constants["PAPER_S2_BIAS"]
        assert f"Table S2 bias of ${bias:.3f}$" in nb_markdown
        # The review's Table S2 row: TWFE Include BC at n = 2000
        assert re.search(
            rf"\| TWFE Include BC \| 2000 \| {re.escape(f'{bias:.3f}')} \|", review_text
        )


class TestApplicationNumbersMatchReview:
    """Slide 10 shows the AUTHORS' NLSY79 estimates as the review records them."""

    def test_att_x_and_att(self, deck_constants, review_text):
        ax, sx = deck_constants["NLSY_ATT_X"]
        a, s = deck_constants["NLSY_ATT"]
        assert f"`ATT_X = {ax:.4f} (SE {sx:.4f})`" in review_text
        assert f"`ATT = {a:.4f} (SE {s:.4f})`" in review_text

    def test_sample_sizes(self, deck_constants, review_text):
        assert f"displaced `N = {deck_constants['NLSY_N_DISPLACED']}`" in review_text
        assert f"non-displaced `N = {deck_constants['NLSY_N_NOT']:,}`" in review_text

    def test_percent_readings_are_the_papers(self, deck_constants, review_text, slide_strings):
        ax = deck_constants["NLSY_ATT_X"][0]
        assert f"{ax * 100:+.1f}%" == "-2.6%"
        assert "2.6% lower wages" in review_text
        assert "about 7%" in review_text
        assert deck_constants["NLSY_TWFE_TEXT"] == "almost -10%"
        assert '"almost 10%"' in review_text  # the paper's words for the naive arms
        assert "roughly 40% larger in magnitude" in review_text
        assert "roughly 40% larger in magnitude" in slide_strings["slide_10_wild"]


class TestAnchors:
    def test_feature_release_is_the_changelog_section_that_shipped_the_lane(self, deck_constants):
        _skip_unless(CHANGELOG)
        text = CHANGELOG.read_text(encoding="utf-8")
        release = deck_constants["FEATURE_RELEASE"]
        m = re.search(
            rf"^## \[{re.escape(release)}\.\d+\].*?(?=^## \[)", text, re.MULTILINE | re.DOTALL
        )
        assert m is not None, f"no CHANGELOG section for {release}.x"
        assert "bad-control lane" in m.group(0)

    def test_release_label_appears_only_as_the_pinned_constant(
        self, visible_strings, deck_constants
    ):
        # The footer derives from pyproject; the cover's "New in" renders
        # FEATURE_RELEASE. No other copy may carry a release number.
        hits = re.findall(r"\b3\.1\d(?:\.\d+)?\b", visible_strings)
        assert hits == [deck_constants["FEATURE_RELEASE"]], hits

    def test_validation_anchors_match_registry(self, deck_constants):
        _skip_unless(REGISTRY)
        text = REGISTRY.read_text(encoding="utf-8")
        start = text.find("**Bad-control extension (CCPS 2026): `fit(")
        assert start > 0, "REGISTRY lost the DMLDiD bad-control block"
        block = text[start : start + 30000]
        assert f"rtol={deck_constants['ORACLE_TOL']}" in block
        assert (
            f"`{deck_constants['R_PACKAGE']}` is used ONLY as an executed black-box oracle" in block
        )
        assert 'CS "simple" post-treatment average' in block and "Remark 4 overall" in block


class TestArithmeticBehindTheCopy:
    def test_half_the_effect(self, deck_constants):
        c = deck_constants
        gap = c["TRUTH"] - c["TWFE_INCLUDE"][0]
        assert abs(gap - c["TRUTH_X"]) < 0.02, "'half the effect' no longer describes the gap"

    def test_route1_se_distances(self, deck_constants):
        c = deck_constants
        att, se = c["CS_A1"]
        assert 2.0 <= (att - c["TRUTH"]) / se < 3.0, "'two SE above' is stale"
        att, se = c["CS_A1_W"]
        assert abs(att - c["TRUTH"]) < se, "'within one SE' is stale (Approach 1 + W)"
        att, se = c["LANE_W"]
        assert abs(att - c["TRUTH"]) < se, "'within one SE' is stale (lane)"

    def test_pre_rows_near_zero_post_rows_near_truth(self, deck_constants):
        c = deck_constants
        for g, t, att_x, se_x in c["ATT_X_ROWS"]:
            target = 0.0 if t < g else c["TRUTH_X"]
            assert abs(att_x - target) < 2 * se_x, f"ATT_X({g},{t}) is no longer 'near' {target}"


class TestClaimDiscipline:
    def test_qualifiers_are_slide_local(self, slide_strings):
        expectations = {
            # the cover motif is stylized art, labeled on-slide
            "slide_01_cover": ("illustrative",),
            # the receipt states simulation + known truth WITH the numbers
            "slide_03_receipt": ("truth known by construction", "simulated", "reported SE "),
            # the paper slide states preprint status and the author disambiguation
            "slide_05_paper": ("unpublished preprint",),
            # Route 1: which conditioning set is right is an identification call
            "slide_06_route1": ("identification call", "known by construction"),
            # Route 2: simulation + no "bigger number" overclaim
            "slide_07_route2": (
                "Seed-locked simulated example",
                "known\nby construction",
                "Both routes are valid",
            ),
            # the pre-test slide carries BOTH Remark 6 caveats
            "slide_08_pretest": ("not evidence of a bad control", "does not prove"),
            # the code slide states the exclusivity rule
            "slide_09_code": ("never also through covariates=",),
            # the application numbers are the authors', the oracle is black-box,
            # the lane scope and the aggregation weighting are stated
            "slide_10_wild": (
                "The authors' estimates, not reproduced here",
                "black-box",
                "simple weighting",
                "panel data only",
            ),
        }
        for slide, phrases in expectations.items():
            for phrase in phrases:
                assert phrase in slide_strings[slide], (
                    f"qualifier {phrase!r} is not on {slide} - it must ride "
                    "with the claim it scopes"
                )

    def test_paper_is_introduced_before_it_is_referenced(self, slide_strings):
        assert "Caetano, Callaway, Payne & Sant'Anna (2026)" in slide_strings["slide_02_dilemma"]
        for later in ("slide_03_receipt", "slide_04_twist"):
            assert "the paper's" in slide_strings[later]

    def test_statistical_constants_reach_their_slides(self, slide_constant_refs):
        expectations = {
            "slide_01_cover": ("FEATURE_RELEASE",),
            "slide_03_receipt": (
                "TWFE_INCLUDE",
                "TRUTH",
                "TRUTH_X",
                "N_UNITS",
                "N_PERIODS",
                "N_NEVER",
                "COHORT_SIZES",
                "PAPER_S2_BIAS",
            ),
            "slide_06_route1": ("CS_A1", "CS_A1_W"),
            "slide_07_route2": ("TWFE_INCLUDE", "CS_A1_W", "LANE_W", "TRUTH"),
            "slide_08_pretest": ("ATT_X_ROWS", "TRUTH_X"),
            "slide_09_code": (
                "CODE_SEED",
                "CODE_BAD_CONTROL",
                "CODE_W",
                "CODE_Z",
                "CODE_CONTROL_GROUP",
            ),
            "slide_10_wild": (
                "NLSY_ATT_X",
                "NLSY_ATT",
                "NLSY_N_DISPLACED",
                "NLSY_N_NOT",
                "NLSY_TWFE_TEXT",
                "ORACLE_TOL",
                "R_PACKAGE",
            ),
        }
        for slide, names in expectations.items():
            for name in names:
                assert name in slide_constant_refs[slide], (
                    f"constant {name} is not rendered by {slide} - a "
                    "hardcoded literal may have replaced the synced value"
                )

    def test_no_absolutes_or_competitor_claims(self, visible_strings):
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
            "csdid",
            "ptetools",
            "unbiased",  # the deck shows recovery, never claims unbiasedness
            "valid coverage",
            "guarantees",
            "detects bad control",  # ATT_X is not a bad-control detector
            "detect a bad control",
            "pedro",  # author disambiguation: Hugo Sant'Anna
            "reproduced the paper",
        ):
            assert banned not in lowered, f"banned absolute/competitive claim: {banned!r}"

    def test_no_double_quoted_pull_quote_on_paper_slide(self, slide_strings):
        # Preprint, arXiv-v2 pinned: the paper slide carries an attributed
        # paraphrase, never a double-quoted pull quote (the title's own
        # single-quoted 'bad controls' is the paper's typography).
        paper = slide_strings["slide_05_paper"]
        for ch in ('"', "“", "”"):
            assert ch not in paper, "the paper slide must not carry a quoted pull quote"

    def test_r_package_named_once_as_oracle(self, visible_strings, deck_constants):
        # The authors' GPL-3 package is a black-box parity oracle, named once.
        # (the constant's own literal is the single occurrence; the slide
        # renders it through the constant - see slide_constant_refs)
        assert visible_strings.count(deck_constants["R_PACKAGE"]) == 1
        assert visible_strings.count("authors' R ") == 1

    def test_cs_and_dml_are_real_exported_estimators(self):
        init = ROOT / "diff_diff" / "__init__.py"
        _skip_unless(init)
        text = init.read_text(encoding="utf-8")
        assert "CallawaySantAnna" in text and "DMLDiD" in text

    def test_bad_control_summary_is_a_real_method(self):
        results = ROOT / "diff_diff" / "dml_did_results.py"
        if not results.exists():
            results = next((ROOT / "diff_diff").glob("*dml*results*.py"), None) or results
        _skip_unless(results)
        assert "def bad_control_summary" in results.read_text(encoding="utf-8")

    def test_code_slide_fragments_match_notebook_invocation(self, deck_constants):
        code = _notebook_code(NB)
        c = deck_constants
        for fragment in (
            f'control_group="{c["CODE_CONTROL_GROUP"]}"',
            f"seed={c['CODE_SEED']}",
            f'covariates=["{c["CODE_Z"]}"]',
            f'bad_control="{c["CODE_BAD_CONTROL"]}"',
            f'bad_control_covariates=["{c["CODE_W"]}"]',
            "bad_control_summary()",
            'aggregate("event_study")',
            f'covariates=["{c["CODE_BAD_CONTROL"]}", "{c["CODE_Z"]}", "{c["CODE_W"]}"]',
        ):
            assert fragment in code, f"tutorial 33 lost the displayed invocation: {fragment!r}"

    def test_code_slide_snippet_is_valid_python(self):
        _skip_unless(GENERATOR)
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
        assert 'bad_control="x"' in snippet and 'bad_control_covariates=["w"]' in snippet
        assert 'unit="unit"' in snippet and 'first_treat="first_treat"' in snippet

    def test_tutorial_referenced_once_on_cta(self, visible_strings, slide_strings):
        assert visible_strings.count("Tutorial 33") == 1
        assert "Tutorial 33" in slide_strings["slide_11_cta"]
        assert "tutorials/33_bad_controls.html" in visible_strings
        assert (ROOT / NB).exists(), "CTA points to a tutorial that does not exist"

    def test_module_docstring_records_the_discipline(self):
        _skip_unless(GENERATOR)
        doc = ast.get_docstring(ast.parse(GENERATOR.read_text(encoding="utf-8"))) or ""
        for phrase in ("Hugo", "AUTHORS' estimates", "no number", "illustrative"):
            assert phrase.lower() in doc.lower(), f"docstring lost the {phrase!r} discipline note"
