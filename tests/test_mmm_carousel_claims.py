"""Claim-sync guard for the MMM calibration interop carousel.

The deck (``carousel/generate_mmm_carousel.py``) hardcodes the committed
executed tutorials' seed-locked numbers as module constants. The
synchronization chain is:

- library <-> notebook: ``tests/test_t29_mmm_calibration_pymc_drift.py`` and
  ``tests/test_t30_mmm_calibration_meridian_drift.py`` re-derive the DiD-side
  numbers from the public API and pin the committed notebook surfaces;
- notebook <-> carousel: THIS file parses the deck's constants (via ``ast`` -
  the generator imports ``fpdf2``, a carousel-only dependency deliberately
  absent from CI) and locates each on the COMMITTED NOTEBOOK SURFACE, not
  restated here. If either tutorial is ever retuned and the carousel is
  forgotten, the notebook surface changes and this file fails.

Skips cleanly when ``carousel/`` or ``docs/`` is absent (the isolated-install
CI jobs copy only ``tests/``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests._tutorial_drift import notebook_output_text

GENERATOR = Path(__file__).resolve().parents[1] / "carousel" / "generate_mmm_carousel.py"
NB_MER = "docs/tutorials/30_mmm_calibration_meridian.ipynb"
NB_PM = "docs/tutorials/29_mmm_calibration_pymc.ipynb"


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
                # e.g. ``PRIOR_MU, PRIOR_SIGMA = 0.9109, 0.0195``
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
    """Slide copy: every string literal in the generator EXCEPT docstrings."""
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


def _notebook_code(rel_path: str) -> str:
    import json

    nb_path = Path(__file__).resolve().parents[1] / rel_path
    if not nb_path.exists():
        pytest.skip(f"Notebook not found at {nb_path}.")
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in json.loads(nb_path.read_text())["cells"]
        if c["cell_type"] == "code"
    )


@pytest.fixture(scope="module")
def mer_output():
    return notebook_output_text(NB_MER)


@pytest.fixture(scope="module")
def pm_output():
    return notebook_output_text(NB_PM)


class TestMeridianSpineMatchesNotebook:
    """Every tutorial-30 number on the deck is on the committed surface."""

    def test_posterior_summary_lines(self, deck_constants, mer_output):
        c = deck_constants
        d, cal = c["MER_DEFAULT"], c["MER_CAL"]
        for needle in (
            f"default prior: ROI mean {d[0]:.2f}, 90% interval ({d[1]:.2f}, {d[2]:.2f})",
            f"calibrated prior: ROI mean {cal[0]:.2f}, 90% interval ({cal[1]:.2f}, {cal[2]:.2f})",
            f"truth: {c['MER_TRUTH']:.2f}",
            f"DiD measurement: {c['MER_DID'][0]:.2f} ± {c['MER_DID'][1]:.2f}",
            f"error: {c['MER_ERR'][0]:.2f} -> {c['MER_ERR'][1]:.2f}; "
            f"90% width: {c['MER_WIDTH'][0]:.2f} -> {c['MER_WIDTH'][1]:.2f}",
        ):
            assert needle in mer_output, f"deck value not on tutorial-30 surface: {needle!r}"

    def test_confidently_wrong_structure(self, deck_constants):
        # The load-bearing story beats, reproduced from the deck's own
        # constants: the default interval EXCLUDES the truth, the calibrated
        # interval covers it and is strictly narrower.
        c = deck_constants
        d, cal, truth = c["MER_DEFAULT"], c["MER_CAL"], c["MER_TRUTH"]
        assert d[1] > truth, "slide-5 claim requires truth below the default interval"
        assert cal[1] < truth < cal[2], "calibrated interval must cover the truth"
        assert (cal[2] - cal[1]) < (d[2] - d[1])

    def test_totals_lines(self, deck_constants, mer_output):
        # The slide-3 code-panel comments quote the CS total and the spend;
        # both must match the labeled lines in the committed output.
        c = deck_constants
        for label_re, key in (
            (r"CS total incremental sales:\s+", "CS_TOTAL_LABEL"),
            (r"total experiment spend:\s+", "SPEND_LABEL"),
        ):
            pattern = label_re + re.escape(c[key])
            assert re.search(pattern, mer_output), f"totals line not found: {pattern!r}"

    def test_prior_and_mask_lines(self, deck_constants, mer_output):
        c = deck_constants
        roi = c["ROI_MEAS"]
        assert f"experiment ROI: {roi[0]:.3f} ± {roi[1]:.3f}" in mer_output
        assert f"mu={c['PRIOR_MU']:.4f}, sigma={c['PRIOR_SIGMA']:.4f}" in mer_output
        assert f"search weeks in window: {c['MASK_WEEKS']}" in mer_output

    def test_snippet_lines_verbatim_from_committed_to_code_output(self, deck_constants, mer_output):
        # Slide 4 shows the generated snippet; every displayed line must be
        # byte-verbatim from the committed to_code() output cell.
        for line in deck_constants["SNIPPET_LINES"]:
            assert line in mer_output, f"snippet line not verbatim in tutorial 30: {line!r}"

    def test_slide4_displays_scoped_to_code_invocation(self, visible_strings):
        # CI review R1: to_code() fails closed without channel + time scope,
        # so the "Paste and Run" slide must display the tutorial's scoped
        # invocation - a bare prior.to_code() raises ValueError. Pin the
        # scoping kwargs on the displayed call and confirm the same kwargs
        # appear in the tutorial's own to_code() cell.
        assert 'prior.to_code(channel="search", media_channels=["search", "tv"],' in visible_strings
        assert "roi_calibration_period=mask)" in visible_strings
        assert (
            "prior.to_code()" not in visible_strings
        ), "bare prior.to_code() is back on the deck - it raises ValueError"
        code = _notebook_code(NB_MER)
        for kwarg in (
            'channel="search"',
            'media_channels=["search", "tv"]',
            "roi_calibration_period=mask",
        ):
            assert kwarg in code, f"tutorial 30's to_code() call lost {kwarg!r}"

    def test_design_facts_match_notebook_code(self, deck_constants):
        c = deck_constants
        code = _notebook_code(NB_MER)
        assert f"N_GEOS, N_WEEKS = {c['N_GEOS_MER']}, {c['N_WEEKS']}" in code
        m = re.search(r"LAUNCH_COHORTS = (\{[^}]*\})", code)
        assert m, "LAUNCH_COHORTS literal not found in tutorial 30"
        cohorts = ast.literal_eval(m.group(1))
        assert set(cohorts) == set(c["WAVE_WEEKS"])
        assert sum(len(g) for g in cohorts.values()) == c["N_LAUNCHED"]


class TestPymcBeatMatchesNotebook:
    """The tutorial-29 numbers on slide 8 are on the committed surface."""

    def test_posterior_summary_lines(self, deck_constants, pm_output):
        c = deck_constants
        p, cal = c["PM_PLAIN"], c["PM_CAL"]
        for needle in (
            f"without lift test: ROI mean {p[0]:.2f}, 90% interval ({p[1]:.2f}, {p[2]:.2f})",
            f"with lift test: ROI mean {cal[0]:.2f}, 90% interval ({cal[1]:.2f}, {cal[2]:.2f})",
            f"truth: {c['PM_TRUTH']:.2f}",
            f"error: {c['PM_ERR'][0]:.2f} -> {c['PM_ERR'][1]:.2f}; "
            f"90% width: {c['PM_WIDTH'][0]:.2f} -> {c['PM_WIDTH'][1]:.2f}",
        ):
            assert needle in pm_output, f"deck value not on tutorial-29 surface: {needle!r}"

    def test_six_x_narrower_claim(self, deck_constants):
        # "the 90% interval is 6x narrower" - recompute from the committed
        # widths; the wording rounds DOWN, so the true ratio must be >= 6.
        w = deck_constants["PM_WIDTH"]
        assert w[0] / w[1] >= 6.0

    def test_design_facts_match_notebook_code(self, deck_constants):
        c = deck_constants
        code = _notebook_code(NB_PM)
        assert f"N_GEOS, N_WEEKS = {c['N_GEOS_PM']}, {c['N_WEEKS']}" in code
        m = re.search(r"TREAT_COHORTS = (\{[^}]*\})", code)
        assert m, "TREAT_COHORTS literal not found in tutorial 29"
        cohorts = ast.literal_eval(m.group(1))
        assert sum(len(g) for g in cohorts.values()) == c["N_BOOSTED"]

    def test_slide8_call_arguments_rederived_from_notebook_dgp(self, deck_constants):
        # Round-2 review: the slide-8 x/delta_x/scale literals render from
        # deck constants; re-derive each from tutorial 29's own DGP lines so
        # a notebook retune cannot leave the deck's call arguments stale.
        c = deck_constants
        code = _notebook_code(NB_PM)
        base = re.search(r"SEARCH_BASE = ([\d.]+)", code)
        boost = re.search(r"BOOST = ([\d.]+)", code)
        assert base and boost, "tutorial 29 DGP spend constants not found"
        assert c["PM_X"] == float(base.group(1)) * c["N_GEOS_PM"]
        assert c["PM_DELTA_X"] == float(boost.group(1)) * c["N_BOOSTED"]
        assert c["PM_SCALE"] == c["N_BOOSTED"]


class TestGuardrailPhrases:
    """The qualifiers ARE the claim - a rewording that drops one ships an
    overclaim even with every number intact (CiC-deck precedent)."""

    def test_load_bearing_phrases_on_visible_surface(self, visible_strings):
        for phrase in (
            # Truth provenance is deliberately CONCENTRATED (2026-08-20):
            # the chart's "(simulated)" label + the CTA's full statement are
            # the only sites, so losing either ships a real-market
            # ground-truth implication.
            "(simulated)",  # slide-5 truth-line label
            "the true ROI (simulated)",  # cover truth-line label (round-4
            # review: the cover circulates alone as the post thumbnail)
            "illustrative",  # cover motif shape label (curves are drawn from
            # summaries with inflated widths, not posterior draws)
            "A simulated market, so the truth is known",  # CTA teaser line
            "abridged",  # slide-4 snippet honesty
            "schematic",  # slide-6 figure label
            "For 'search', the calibration mask",  # slide-4 mask scoping
            # Guardrail scoping (round-1 review): the deck may only claim the
            # checks the exporters actually run - "the easy mistakes" scopes
            # the loud-failure claim, and the ownership line assigns the
            # semantic-alignment responsibility (estimand, outcome scale,
            # population, window) to the caller without a compliance clause.
            "the easy mistakes fail loudly",  # slide-7 setup-check scope
            "you own the design, it owns the math",  # slide-9 guardrail scope
        ):
            assert phrase in visible_strings, f"load-bearing phrase missing: {phrase!r}"

    def test_no_unqualified_guardrail_absolutes(self, visible_strings):
        # Round-1 review: these absolutes are falsifiable (a log-scale or
        # window-mismatched estimate passes numeric validation silently), so
        # they may never return in any rewording.
        lowered = visible_strings.lower()
        for banned in (
            "no silent mis-calibration",
            "never silently",
            "anything with an estimate + se",
            "any effect + se",
            "any estimate exports",
            "catches every",
            # aggregate()/export are NOT universal estimator methods
            # (round-3 review): the five totals adopters own aggregate();
            # the exporters are module functions.
            "one pattern across every estimator",
        ):
            assert banned not in lowered, f"unqualified guardrail absolute on deck: {banned!r}"

    def test_deck_comment_literals_match_constants(self, deck_constants, visible_strings):
        # The slide-3 code-panel comments are hardcoded strings; tie them to
        # the notebook-synced constants so they cannot drift independently.
        c = deck_constants
        roi = c["ROI_MEAS"]
        assert f"# incremental sales: {c['CS_TOTAL_LABEL']}" in visible_strings
        assert f"# {c['SPEND_LABEL']}" in visible_strings
        assert f"# experiment ROI: {roi[0]:.3f} +/- {roi[1]:.3f}" in visible_strings
        assert f"LogNormal(mu={c['PRIOR_MU']:.4f}, sigma={c['PRIOR_SIGMA']:.4f})" in visible_strings
        # Slide 8's scale= argument is the boosted-geo count.
        assert f"# scale = {c['N_BOOSTED']} boosted geos" in visible_strings

    def test_totals_adopter_card_matches_source(self, visible_strings):
        # Slide 9 names the five aggregate('total') adopters; each named
        # estimator's results module must actually list "total" in its
        # _AGGREGATE_SUPPORTED tuple, so the card cannot outlive a de-adoption
        # (or miss staying in sync if the deck's list is ever edited).
        assert (
            "aggregate('total') on Callaway-Sant'Anna, DML DiD, EfficientDiD, ImputationDiD, TwoStageDiD"
            in visible_strings
        ), "slide-9 totals-adopter card wording changed - re-sync this test"
        repo = Path(__file__).resolve().parents[1]
        for module in (
            "diff_diff/staggered_results.py",
            "diff_diff/efficient_did_results.py",
            "diff_diff/imputation_results.py",
            "diff_diff/two_stage_results.py",
        ):
            path = repo / module
            if not path.exists():
                pytest.skip(f"{module} not available in this CI environment.")
            src = path.read_text(encoding="utf-8")
            m = re.search(r"_AGGREGATE_SUPPORTED = \(([^)]*)\)", src)
            assert m and '"total"' in m.group(
                1
            ), f"{module} no longer advertises aggregate('total') - slide 9 overclaims"
        # DMLDiDResults INHERITS _AGGREGATE_SUPPORTED (its module carries no
        # literal assignment for the source grep) — pin it by import, behind
        # the suite's availability convention.
        if not (repo / "diff_diff" / "dml_did_results.py").exists():
            pytest.skip("dml_did_results.py not available in this CI environment.")
        from diff_diff.dml_did_results import DMLDiDResults

        assert (
            "total" in DMLDiDResults._AGGREGATE_SUPPORTED
        ), "DMLDiDResults no longer advertises aggregate('total') - slide 9 overclaims"

    FORBIDDEN_FRAMEWORKS = (
        "meridian",
        "pymc_marketing",
        "pymc",
        "tensorflow_probability",
        "tensorflow",
    )

    def test_zero_new_dependencies_claim_imports(self, visible_strings):
        # Slide 9's "Zero new dependencies" strip: diff_diff/mmm.py must not
        # import either MMM framework (or its substrate) ANYWHERE - stricter
        # than module level, so a lazy in-function import also fails - the
        # to_code() templates embed those imports as TEXT only.
        assert "Zero new dependencies" in visible_strings
        repo = Path(__file__).resolve().parents[1]
        path = repo / "diff_diff" / "mmm.py"
        if not path.exists():
            pytest.skip("diff_diff/mmm.py not available in this CI environment.")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = set(self.FORBIDDEN_FRAMEWORKS)
        assert not (imported & forbidden), (
            f"mmm.py imports MMM frameworks {sorted(imported & forbidden)} - "
            "the 'Zero new dependencies' claim on slide 9 is false"
        )

    def test_zero_new_dependencies_claim_pyproject(self):
        # Round-1 review: an import scan alone would miss a framework added
        # to the install requirements without being imported. Scan every
        # quoted requirement string in pyproject.toml for the forbidden
        # distributions (regex, not tomllib - the test must run on the 3.9
        # CI floor).
        repo = Path(__file__).resolve().parents[1]
        path = repo / "pyproject.toml"
        if not path.exists():
            pytest.skip("pyproject.toml not available in this CI environment.")
        text = path.read_text(encoding="utf-8")
        # The terminator set includes the closing quote so a BARE requirement
        # ("pymc-marketing", no version/extras marker) is also caught
        # (round-2 review).
        spec_names = {
            m.group(1).lower().replace("_", "-")
            for m in re.finditer(r'"([A-Za-z0-9][A-Za-z0-9._-]*)\s*[><=~!;\["]', text)
        }
        forbidden = {
            "google-meridian",
            "meridian",
            "pymc-marketing",
            "pymc",
            "tensorflow",
            "tensorflow-probability",
            "tfp-nightly",
        }
        assert not (spec_names & forbidden), (
            f"pyproject.toml declares MMM framework dependencies {sorted(spec_names & forbidden)} - "
            "the 'Zero new dependencies' claim on slide 9 is false"
        )

    def test_framework_version_claim_matches_notebook_requirements(self, visible_strings):
        # Round-4 review: the slide-9 strip's "Built against ..." versions
        # must be DERIVED from the tutorials' own stated requirements
        # (each notebook's intro markdown pins the exact framework version),
        # so a tutorial version bump cannot leave the deck claim stale.
        from tests._tutorial_drift import notebook_markdown

        pm = re.search(r"pymc-marketing==(\d+\.\d+)", notebook_markdown(NB_PM))
        mer = re.search(r"google-meridian==(\d+\.\d+)", notebook_markdown(NB_MER))
        assert pm and mer, "tutorial intro requirement pins not found"
        expected = f"Built against pymc-marketing {pm.group(1)} + google-meridian {mer.group(1)}"
        assert (
            expected in visible_strings
        ), f"slide-9 version strip does not match the tutorials' requirements: {expected!r}"

    def test_no_competitive_claims(self, visible_strings):
        # User decision (2026-08-19): no landscape/competitor claims - they
        # are not click-through verifiable.
        lowered = visible_strings.lower()
        for banned in (
            "only frequentist",
            "the only ",
            "robyn",
            "causalpy",
            "geolift",
            "causalimpact",
            "first and only",
        ):
            assert banned not in lowered, f"banned competitive claim on deck: {banned!r}"
