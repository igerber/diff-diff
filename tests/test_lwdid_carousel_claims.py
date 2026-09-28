"""Claim-sync guard for the LWDiD launch carousel.

The deck (``carousel/generate_lwdid_carousel.py``) hardcodes committed,
seed-locked numbers from the executed tutorial-31 notebook as module
constants. The synchronization chain is:

- library <-> notebook: tutorial 31 is numbers-locked and executed against
  the committed library (its real-data cells assert their
  ``lwdid_ssc_ancillary`` provenance in the notebook itself);
- notebook <-> carousel: THIS file parses the deck's constants (via ``ast``
  - the generator imports ``fpdf2``, a carousel-only dependency deliberately
  absent from CI) and locates each on the COMMITTED NOTEBOOK SURFACE, not
  restated here. If the tutorial is ever retuned and the carousel is
  forgotten, the notebook surface changes and this file fails.

Skips cleanly when ``carousel/`` or ``docs/`` is absent (the
isolated-install CI jobs copy only ``tests/``).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from tests._tutorial_drift import notebook_markdown, notebook_output_text

GENERATOR = Path(__file__).resolve().parents[1] / "carousel" / "generate_lwdid_carousel.py"
NB = "docs/tutorials/31_lwdid.ipynb"


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
                # e.g. ``WM_COUNTIES, WM_YEARS, WM_COHORTS, WM_NEVER = ...``
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


# Chart helpers attributed to the slide that places them (shared by the
# per-slide string and per-slide constant-reference fixtures).
_SLIDE_HELPERS = {
    "slide_01_cover": ("_render_cover_motif",),
    "slide_03_trick": ("_render_transform_schematic",),
    "slide_05_walmart": ("_render_walmart_event_study",),
}


def _per_slide(collect):
    """Map ``slide_*`` method name -> ``collect(function node)`` output,
    concatenated with the output for the slide's chart helpers."""
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
    assert len(slides) == 8, f"expected 8 slide_* methods, found {sorted(slides)}"
    return slides


@pytest.fixture(scope="module")
def slide_strings():
    """Per-slide visible strings (review R3: a qualifier must live ON the
    slide that makes the claim, not merely somewhere in the module)."""

    def collect(node):
        return " ".join(
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        )

    return _per_slide(collect)


@pytest.fixture(scope="module")
def slide_constant_refs():
    """Per-slide referenced module-level names (review R7: each pinned
    statistical constant must actually be RENDERED by its intended slide -
    a hardcoded literal bypassing the constant would otherwise pass)."""

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


class TestRealDataMatchesNotebook:
    """Every real-data number on the deck is on the committed surface."""

    def test_walmart_lines(self, deck_constants, nb_output):
        c = deck_constants
        att, se = c["WM_ATT"]
        span = c["WM_COHORT_SPAN"]
        needles = [
            f"overall ATT: {att:.4f} (SE {se:.4f})",
            f"{c['WM_COUNTIES']} counties x {c['WM_YEARS']} years; "
            f"{c['WM_COHORTS']} cohorts ({span[0]}-{span[1]}); "
            f"{c['WM_NEVER']} never-treated",
            f"near leads (r = {c['WM_NEAR_LEADS_WINDOW'][0]}..{c['WM_NEAR_LEADS_WINDOW'][1]}):"
            f"  max |WATT(r)| = {c['WM_NEAR_LEADS_MAX']:.4f}",
            f"all leads (r = {c['WM_ALL_LEADS_AT']}..-3):  max |WATT(r)| = "
            f"{c['WM_ALL_LEADS_MAX']:.4f} (at r = {c['WM_ALL_LEADS_AT']})",
        ]
        for r, (eff, wse) in c["WM_WATT"].items():
            needles.append(f"WATT({r}):  {eff:.4f} (SE {wse:.4f})")
        for needle in needles:
            assert needle in nb_output, f"deck value not on tutorial-31 surface: {needle!r}"

    def test_prop99_lines(self, deck_constants, nb_output):
        c = deck_constants
        det, dem = c["P99_DETREND"], c["P99_DEMEAN"]
        span = c["P99_SPAN"]
        for needle in (
            f"{c['P99_STATES']} states x {c['P99_YEARS']} years ({span[0]}-{span[1]})",
            f"detrend : ATT {det[0]:.4f} (SE {det[1]:.4f})   "
            f"exact p = {c['P99_EXACT_P']:.4f} (df = {c['P99_DF']})",
            f"demean  : ATT {dem[0]:.4f} (SE {dem[1]:.4f})",
            f"randomization inference p-value: {c['P99_RI_P']:.4f}",
        ):
            assert needle in nb_output, f"deck value not on tutorial-31 surface: {needle!r}"

    def test_ri_ran_on_the_detrended_fit(self):
        # Slide 4 pairs the exact p and the RI p as two roads to inference on
        # the SAME (detrended) fit; the notebook must keep that provenance.
        code = _notebook_code(NB)
        assert 'det = p99["detrend"]' in code
        assert "det.randomization_test(n_reps=9999" in code

    def test_about_20_percent_phrase_is_notebook_backed(self, deck_constants, visible_strings):
        # "about a 20% reduction" is the committed tutorial markdown's own
        # characterization of the detrended log-point estimate; re-derive the
        # implied percentage from the deck's constant as well.
        assert "about a 20% reduction" in visible_strings
        assert "about a 20% reduction" in notebook_markdown(NB)
        implied = 1.0 - math.exp(deck_constants["P99_DETREND"][0])
        assert 0.18 <= implied <= 0.22, "log-point estimate no longer implies ~20%"


class TestSimulationsMatchNotebook:
    """The simulated-DGP numbers on the deck are on the committed surface."""

    def test_drifting_units_lines(self, deck_constants, nb_output):
        c = deck_constants
        for label, (att, se) in (
            ("demean  ", c["TREND_DEMEAN"]),
            ("detrend ", c["TREND_DETREND"]),
        ):
            needle = f"{label}: ATT {att:.3f} (SE {se:.3f})   truth: {c['TREND_TRUTH']:.1f}"
            assert needle in nb_output, f"deck value not on tutorial-31 surface: {needle!r}"

    def test_clustering_lines(self, deck_constants, nb_output):
        c = deck_constants
        for needle in (
            f"hc1, clustering ignored: ATT {c['CL_ATT']:.3f} (SE {c['CL_NAIVE_SE']:.3f})",
            f"region-clustered:        ATT {c['CL_ATT']:.3f} "
            f"(CR1 SE {c['CL_CR1_SE']:.3f}), G = {c['CL_G']}   truth: {c['CL_TRUTH']:.1f}",
            f"wild cluster bootstrap: p = {c['WCB_P']:.4f}, "
            f"95% CI [{c['WCB_LO']:.3f}, {c['WCB_HI']:.3f}]",
        ):
            assert needle in nb_output, f"deck value not on tutorial-31 surface: {needle!r}"

    def test_staggered_sim_line(self, deck_constants, nb_output):
        c = deck_constants
        att, se, truth = c["SIM_STAG"]
        needle = f"LWDiD overall ATT: {att:.3f} (SE {se:.3f})   (truth: {truth:.3f})"
        assert needle in nb_output, f"deck value not on tutorial-31 surface: {needle!r}"

    def test_barely_half_claim(self, deck_constants):
        # Slide 6: "The naive SE was barely half the honest one." Recompute
        # from the committed SEs; the wording requires a ratio <= 0.6.
        c = deck_constants
        assert c["CL_NAIVE_SE"] / c["CL_CR1_SE"] <= 0.6

    def test_factor_of_two_claim(self, deck_constants):
        # Slide 6: Prop 99's two transformations "disagree by nearly
        # a factor of two". Recompute from the committed estimates.
        c = deck_constants
        ratio = c["P99_DEMEAN"][0] / c["P99_DETREND"][0]
        assert 1.5 <= ratio <= 2.05, f"'nearly a factor of two' no longer holds: {ratio:.2f}"


class TestGuardrailPhrases:
    """The qualifiers ARE the claim - a rewording that drops one ships an
    overclaim even with every number intact (CiC/MMM-deck precedent)."""

    def test_load_bearing_phrases_on_visible_surface(self, visible_strings):
        for phrase in (
            "(simulated)",  # simulated-DGP truth labels (slides 5, 6)
            "illustrative",  # cover motif is art, not a data claim
            "schematic",  # slide-3 transformation figure label
            # The exact-t assumptions travel WITH the claim (slide 3);
            # REGISTRY "Small-sample (exact) inference layer" + review R1.
            "no anticipation",
            "mean-zero, conditionally normal, homoskedastic",
            # RI's justification is the assignment mechanism (review R1).
            "complete-randomization assignment",
            # Slide 5's lead-flatness claim is scoped to the printed window
            # AND discloses the far lead (review R1) - never generalized.
            "near leads (r = ",
            "Near leads stay below ",
            "distant leads reach |WATT| = ",
            # CHT is a heterogeneous LINEAR trends model (review R1).
            "unit-specific linear drift",
            # Slide 4's replication claim is scoped to ATT/SE/exact p; the
            # RI card follows the authors' package convention (REGISTRY).
            "reproduce LW 2026 Table 3",
            "authors' package convention",
            "conditional on the detrending (CHT)",
            # Slide 2's single, neutral SyntheticDiD mention: estimation is
            # covered; the gap claim is scoped to the exact p-value.
            "SyntheticDiD does it.",
            "An exact p-value with",
            # Slide 4 presents exact t and RI side by side without claiming
            # they agree (0.021 and 0.054 straddle 0.05).
            "Two roads to inference with a ",
            # Slide 7's df comment renders from the pinned Prop 99 df.
            "# exact t, df = ",
            # Cover subtitle names the release (label derived, never typed).
            "New in diff-diff ",
            # Review R2: the exact-inference claim is about ONE TREATED unit
            # (N1 = 1); LW 2026 requires total N >= 3, so the deck must say
            # "one treated unit", never total-sample notation.
            "one treated unit",
        ):
            assert phrase in visible_strings, f"load-bearing phrase missing: {phrase!r}"

    def test_qualifiers_are_slide_local(self, slide_strings):
        # Review R3: each qualifier must appear on the SLIDE whose claim it
        # scopes - a qualifier drifting to an unrelated slide keeps the
        # global test green while the claim ships unqualified.
        expectations = {
            "slide_01_cover": ("New in diff-diff ", "illustrative"),
            "slide_02_gap": (
                "SyntheticDiD does it.",
                "An exact p-value with",
                # Review R6: the regression tagline is scoped to the
                # classical path (ipw/dr/psm exist; exact t is reg-only).
                "classical path",
            ),
            "slide_03_trick": (
                "schematic",
                # Review R5: the identification stack rides with the claim.
                "no anticipation",
                "overlap",
                "mean-zero, conditionally normal, homoskedastic",
                # Review R3: the sample-size guard rides with the claim -
                # one treated unit is valid only with total N >= 3.
                "at least two controls (N >= 3)",
                "one treated unit",
            ),
            "slide_04_prop99": (
                "reproduce LW 2026 Table 3",
                "authors' package convention",
                "complete-randomization assignment",
                "conditional on the detrending (CHT)",
                # Review R8: the proof slide repeats the classical-error
                # qualifier locally (it circulates on its own).
                "conditionally normal, homoskedastic collapsed errors",
                "Two roads to inference with a ",
            ),
            "slide_05_walmart": (
                "near leads (r = ",
                "Near leads stay below ",
                "(magnitude envelope)",
                # Review R8: the far-lead disclosure is a magnitude - the
                # committed surface pins max |WATT(r)|, never the sign.
                "distant leads reach |WATT| = ",
                "(simulated)",
            ),
            "slide_06_use_it_well": (
                "unit-specific linear drift",
                # Reviews R5/R6: the diagnostics call covers the
                # transformation (not clustering) and is descriptive - the
                # tutorial's own "not assumption tests" framing.
                "descriptive, not an assumption test",
                "(simulated)",
                # Review R3: WCB is a clustered common-timing capability.
                "common-timing reg",
            ),
            "slide_07_code": (
                "# exact t, df = ",
                ", seed=",
                # Review R3/R4: the RT is common-timing only, but the
                # staggered scope note must keep the exact-composite
                # exception (REGISTRY per-surface reference distributions).
                "common-timing reg only",
                "classical composite keeps exact t",
            ),
        }
        for slide, phrases in expectations.items():
            for phrase in phrases:
                assert phrase in slide_strings[slide], (
                    f"qualifier {phrase!r} is not on {slide} - it must ride "
                    "with the claim it scopes"
                )

    def test_statistical_constants_reach_their_slides(self, slide_constant_refs):
        # Review R7: each notebook-pinned constant must be referenced by the
        # slide that displays it, so a hardcoded literal cannot silently
        # replace the synced value.
        expectations = {
            "slide_04_prop99": (
                "P99_EXACT_P",
                "P99_RI_P",
                "P99_DF",
                "P99_DETREND",
                "P99_STATES",
                "P99_YEARS",
                "P99_SPAN",
            ),
            "slide_05_walmart": (
                "WM_ATT",
                "WM_WATT",
                "WM_COUNTIES",
                "WM_COHORTS",
                "WM_NEVER",
                "WM_COHORT_SPAN",
                "WM_NEAR_LEADS_MAX",
                "WM_NEAR_LEADS_WINDOW",
                "WM_ALL_LEADS_MAX",
                "WM_ALL_LEADS_AT",
                "SIM_STAG",
            ),
            "slide_06_use_it_well": (
                "TREND_DEMEAN",
                "TREND_DETREND",
                "TREND_TRUTH",
                "P99_DEMEAN",
                "CL_NAIVE_SE",
                "CL_CR1_SE",
                "CL_G",
                "WCB_P",
                "WCB_LO",
                "WCB_HI",
            ),
            "slide_07_code": ("P99_DF", "RI_REPS", "RI_SEED"),
        }
        for slide, names in expectations.items():
            for name in names:
                assert name in slide_constant_refs[slide], (
                    f"constant {name} is not rendered by {slide} - a "
                    "hardcoded literal may have replaced the synced value"
                )

    def test_no_absolutes_or_competitor_claims(self, visible_strings):
        # The deck positions LWDiD as complementary (user decision,
        # 2026-08-22): no superlatives about our own estimators and no
        # external-package comparisons (not click-through verifiable).
        lowered = visible_strings.lower()
        for banned in (
            "the only ",
            "first and only",
            "no other estimator",
            "nothing else",
            "unlike synthetic",
            "better than",
            "synthdid",
            "csdid",
            "fixest",
            "geolift",
            "causalimpact",
            # Review R1: the categorical post-only claim suppressed the
            # r = -22 far lead; the unqualified drift headline overstated
            # CHT (a linear-trends model). Neither may return.
            "only after entry",
            "detrending allows drift",
            # Review R2: "N = 1" is total-sample notation - the method
            # requires N >= 3; the valid claim is N1 = 1 (one treated unit).
            "n = 1",
            # Review R9: "flat" is inferential wording - the chart shows a
            # magnitude envelope, not lead-level uncertainty.
            "flat near leads",
            "flat leads",
        ):
            assert banned not in lowered, f"banned absolute/competitive claim: {banned!r}"

    def test_syntheticdid_is_a_real_exported_estimator(self):
        # Slide 2 says "SyntheticDiD does it" - the named estimator must
        # exist on the library's public surface.
        init = Path(__file__).resolve().parents[1] / "diff_diff" / "__init__.py"
        if not init.exists():
            pytest.skip("diff_diff/__init__.py not available in this CI environment.")
        assert "SyntheticDiD" in init.read_text(encoding="utf-8")

    def test_code_slide_fragments_match_notebook_invocation(self, deck_constants, visible_strings):
        # Slide 7's code is the tutorial's own common-timing classical fit -
        # the only mode where randomization_test() is defined - so each
        # displayed fragment must appear in the notebook's code cells. The
        # displayed reps/seed render from the deck constants (review R4), so
        # reconstructing the call from those constants pins deck <-> notebook.
        code = _notebook_code(NB)
        c = deck_constants
        for fragment in (
            'rolling="detrend"',
            'vcov_type="classical"',
            'outcome="lcigsale"',
            'unit="state"',
            'time="year"',
            'treatment="treated"',
            # The seed is DISPLAYED (review R1) so the shown call reproduces
            # the shown p-value; the notebook must keep the same call.
            f"randomization_test(n_reps={c['RI_REPS']}, seed={c['RI_SEED']}",
        ):
            assert fragment in code, f"tutorial 31 lost the displayed invocation: {fragment!r}"
        assert "randomization_test(n_reps=" in visible_strings
        assert ", seed=" in visible_strings, "the displayed RI call lost its seed"

    def test_tutorial_referenced_once_on_cta(self, visible_strings):
        # House precedent: the tutorial teaser appears ONCE, on the CTA.
        assert visible_strings.count("Tutorial 31") == 1
        assert "tutorials/31_lwdid.html" in visible_strings
        nb_path = Path(__file__).resolve().parents[1] / NB
        assert nb_path.exists(), "CTA points to a tutorial that does not exist"
