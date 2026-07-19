"""Claim-sync guard for the CiC launch carousel.

The deck (``carousel/generate_cic_carousel.py``) hardcodes Tutorial 27's
seed-locked numbers as module constants. The synchronization chain is:

- library <-> notebook: ``tests/test_t27_cic_distributional_effects_drift.py``
  re-derives the tutorial's numbers from the public API and cross-checks the
  committed notebook surface;
- notebook <-> carousel: THIS file parses the deck's constants (via ``ast`` -
  the generator imports ``fpdf2``, a carousel-only dependency deliberately
  absent from CI) and compares them against values PARSED FROM THE COMMITTED
  NOTEBOOK SURFACE, not restated here. If Tutorial 27 is ever retuned and the
  carousel is forgotten, the notebook surface changes and this file fails.

Skips cleanly when ``carousel/`` or ``docs/`` is absent (the isolated-install
CI jobs copy only ``tests/``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests._tutorial_drift import notebook_markdown, notebook_output_text

GENERATOR = Path(__file__).resolve().parents[1] / "carousel" / "generate_cic_carousel.py"
NB = "docs/tutorials/27_cic_distributional_effects.ipynb"


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
                # e.g. ``_MU_LOG, _SIGMA_LOG = 3.4, 0.75``
                try:
                    values = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                if isinstance(values, tuple) and len(values) == len(target.elts):
                    for el, val in zip(target.elts, values):
                        consts[el.id] = val
    return consts


@pytest.fixture(scope="module")
def nb_output():
    return notebook_output_text(NB)


@pytest.fixture(scope="module")
def nb_markdown():
    return notebook_markdown(NB)


@pytest.fixture(scope="module")
def nb_code():
    import json

    nb_path = Path(__file__).resolve().parents[1] / NB
    if not nb_path.exists():
        pytest.skip(f"Notebook not found at {nb_path}.")
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in json.loads(nb_path.read_text())["cells"]
        if c["cell_type"] == "code"
    )


class TestDeckMatchesNotebookSurface:
    """Every visible deck number is derived from the committed notebook."""

    def test_headline_effects_quoted_in_notebook(self, deck_constants, nb_markdown, nb_output):
        c = deck_constants
        surface = nb_markdown + nb_output
        # Needles are BUILT from the deck constants, then located on the
        # notebook surface - never restated as independent literals here.
        for needle in (
            f"{c['DID_ATT']:.2f}, p = {c['DID_P']:.2f}",  # "$0.22, p = 0.90" (md)
            f"[-\\${abs(c['DID_CI'][0]):.2f}, \\${c['DID_CI'][1]:.2f}]",  # md, escaped $
            f"{c['TRUE_MEAN_EFFECT']:.2f} mean effect",  # "$3.01 mean effect" (md)
            f"+{c['LOG_DID_PCT']:.1f}%",
            f"(p = {c['LOG_DID_P']:.3f})",
            f"p = {c['BLIP_P']:.3f}",  # the tau=0.55 pointwise blip
        ):
            assert needle in surface, f"deck value not on notebook surface: {needle!r}"
        # SEED_LABEL / N_CELL_LABEL live on the notebook's CODE surface and
        # are checked in test_dgp_motif_constants_match_notebook_code.

    def test_cic_att_matches_summary_row(self, deck_constants, nb_output):
        m = re.search(
            r"^\s+ATT\s+(\d+\.\d{4})\s+\d+\.\d{4}\s+[\d.]+\s+(0\.\d{3})",
            nb_output,
            re.MULTILINE,
        )
        assert m, "CiC ATT summary row not found in notebook output"
        assert round(float(m.group(1)), 2) == deck_constants["CIC_ATT"]
        assert float(m.group(2)) == deck_constants["CIC_ATT_P"]

    def test_qte_table_matches_summary_verbatim(self, deck_constants, nb_output):
        # Parse the 19-row quantile table from the committed summary()
        # output: tau, qte, se, [conf_low, conf_high].
        rows = re.findall(
            r"^\s+(0\.\d{2})\s+(-?\d+\.\d{4})\s+(\d+\.\d{4})\s+-?[\d.]+\s+[\d.]+"
            r"\s+\[\s*(-?\d+\.\d{4}),\s+(-?\d+\.\d{4})\]",
            nb_output,
            re.MULTILINE,
        )
        assert len(rows) == 19, f"expected 19 QTE rows in notebook output, found {len(rows)}"
        nb_rows = [(float(t), float(q), float(lo), float(hi)) for t, q, _s, lo, hi in rows]
        nb_ses = [float(s) for _t, _q, s, _lo, _hi in rows]
        assert deck_constants["QTE_ROWS"] == nb_rows
        assert deck_constants["QTE_SES"] == nb_ses

    def test_sup_t_crit_matches_notebook(self, deck_constants, nb_output):
        assert f"sup-t critical value: {deck_constants['SUP_T_CRIT']:.3f}" in nb_output

    def test_uniform_band_split_reproduced(self, deck_constants):
        # Self-contained reproduction of the headline joint claim from the
        # deck's own constants: bands (qte +/- crit*se) exclude zero for
        # EXACTLY tau = 0.05..0.50.
        rows = deck_constants["QTE_ROWS"]
        ses = deck_constants["QTE_SES"]
        crit = deck_constants["SUP_T_CRIT"]
        assert len(ses) == len(rows)
        for (tau, qte, _lo, _hi), se in zip(rows, ses):
            excluded = (qte - crit * se) > 0 or (qte + crit * se) < 0
            assert excluded == (tau <= 0.50), f"band split breaks at tau={tau}"

    def test_covariate_section_matches_notebook(self, deck_constants, nb_output):
        c = deck_constants
        assert f"true effect:        {c['COV_TRUTH']:.1f} points" in nb_output
        unc, cov = c["COV_UNC"], c["COV_COND"]
        assert f"ATT = {unc[0]:.2f}  CI [{unc[1]:.2f}, {unc[2]:.2f}]" in nb_output
        assert f"ATT = {cov[0]:.2f}  CI [{cov[1]:.2f}, {cov[2]:.2f}]" in nb_output
        # Structural facts of the story: unconditional CI excludes the
        # truth, conditional CI covers it.
        assert unc[1] > c["COV_TRUTH"]
        assert cov[1] < c["COV_TRUTH"] < cov[2]

    def test_receipt_max_rel_diff_label_matches_notebook(self, deck_constants, nb_output):
        # The "0.00e+00" label is a visible numeric claim - sync it to the
        # committed notebook output line it quotes.
        needle = (
            "max relative difference across all 19 quantiles: "
            f"{deck_constants['MAX_REL_DIFF_LABEL']}"
        )
        assert needle in nb_output

    def test_receipt_rows_match_notebook(self, deck_constants, nb_output):
        rows = re.findall(
            r"^\s*(0\.\d{2})\s+(\d+\.\d{6})\s+(\d+\.\d{6})\s*$", nb_output, re.MULTILINE
        )
        assert rows, "receipt comparison table not found in notebook output"
        nb_receipt = []
        for t, lvl, log in rows:
            assert lvl == log, f"receipt columns differ at tau={t} in the notebook itself"
            nb_receipt.append((float(t), round(float(lvl), 4)))
        assert deck_constants["RECEIPT_ROWS"] == nb_receipt

    def test_dgp_motif_constants_match_notebook_code(self, deck_constants, nb_code):
        # The cover-density motif reuses the tutorial's locked DGP shape -
        # parse the notebook's own constant lines and compare numerically.
        c = deck_constants
        patterns = {
            r"MU_LOG, SIGMA_LOG = ([\d.]+), ([\d.]+)": ("_MU_LOG", "_SIGMA_LOG"),
            r"GAMMA, ALPHA = ([\d.]+), (-[\d.]+)": ("_GAMMA", "_ALPHA"),
            r"U_LO, U_HI = ([\d.]+), ([\d.]+)": ("_U_LO", "_U_HI"),
            r"BETA_A, BETA_B = ([\d.]+), ([\d.]+)": ("_BETA_A", "_BETA_B"),
            r"LIFT_MAX, LIFT_MID, LIFT_SCALE = ([\d.]+), ([\d.]+), ([\d.]+)": (
                "_LIFT_MAX",
                "_LIFT_MID",
                "_LIFT_SCALE",
            ),
            r"SEED = (\d+)\n": ("SEED_LABEL",),
            r"N_CELL = (\d+)": ("N_CELL_LABEL",),
        }
        for pattern, names in patterns.items():
            m = re.search(pattern, nb_code)
            assert m, f"notebook DGP constant line not found: {pattern!r}"
            for group, name in zip(m.groups(), names):
                assert float(group) == float(c[name]), f"{name} drifted from the notebook"

    def test_guardrail_phrases_on_visible_slide_surface(self):
        # The qualifiers ARE the claim (LP-DiD lesson): a rewording that
        # drops one would ship an overclaim even with every number intact.
        # Checked against the deck's USER-VISIBLE surface: every string
        # literal in the generator EXCEPT docstrings (module / class /
        # function first-statement strings). Slide copy lives entirely in
        # such literals, so this is the rendered text without needing the
        # carousel-only PDF toolchain in CI.
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
        visible = " ".join(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        )
        for phrase in (
            "(unconditional fits",  # equivariance scope (slide-4 receipt)
            "scale-specific for every estimator",  # ATT/means caveat (slide 4)
            "fixed 95%",  # sup-t band level, slide-7 caption
            "fixed-95% bands (qte parity)",  # sup-t band level, slide-10 card
            "silent, not exonerating",  # bands above the median (slide 7)
            "power-law",  # trend framing (slide 3)
            "to 1e-10",  # unconditional golden-parity wording (slide 10)
            "bit-exactly (atol=0)",  # reserved for type-1 arithmetic (slide 10)
            "conventions verbatim",  # covariate xformla wording (slide 9)
        ):
            assert phrase in visible, f"load-bearing phrase missing from slides: {phrase!r}"
        # Negative lock on the trend wording: a purely multiplicative trend
        # WOULD be log-parallel, so the slides must never call it that.
        # (The module docstring discusses the word; it is excluded above.)
        assert "multiplicative" not in visible.lower()
