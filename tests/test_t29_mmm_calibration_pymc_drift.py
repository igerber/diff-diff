"""Drift guards for docs/tutorials/29_mmm_calibration_pymc.ipynb.

The notebook is CI-excluded from execution (it needs pymc-marketing, which
conflicts with the Meridian tutorial's environment), so these tests protect its
committed surface with diff-diff-only recomputation:

- DGP constant + formula-line sync (the t26 pattern): the notebook's DGP must
  equal the copy recomputed here, or stale committed outputs would stay green;
- sampler-configuration needles: sampling cannot be weakened or de-seeded
  silently;
- integration-content presence: the framework fits and acceptance asserts cannot
  be deleted while CI stays green;
- rendered-number cross-check: the DiD-side numbers the notebook prints are
  recomputed with diff-diff alone and matched against the committed outputs
  (loose bands per the t27 normal-suite convention - never bit-exact floats).
"""

import re

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna
from diff_diff.mmm import to_pymc_marketing_lift_test
from tests._tutorial_drift import (
    _read_notebook,
    assert_quotes_in_rendered,
    notebook_output_text,
)

NB = "docs/tutorials/29_mmm_calibration_pymc.ipynb"

# ---- locked DGP constants (must equal the notebook's copy verbatim) ----
SEED = 2026
N_GEOS, N_WEEKS = 15, 104
BETA_SEARCH, SEARCH_BASE, BOOST = 2.0, 100.0, 50.0
FREEZE_WEEK = 52
TREAT_COHORTS = {60: [0, 1, 2], 68: [3, 4, 5], 76: [6, 7, 8]}
DEMAND_AMP, CHASE = 300.0, 0.5
TV_BASE, TV_BETA, TV_ADSTOCK, TV_SAT = 200.0, 600.0, 0.5, 0.004
BASE_SALES, GEO_SD, NOISE_SD = 3000.0, 100.0, 15.0

DGP_CONSTANT_NEEDLES = [
    "SEED = 2026",
    "N_GEOS, N_WEEKS = 15, 104",
    "BETA_SEARCH = 2.0",
    "SEARCH_BASE = 100.0",
    "BOOST = 50.0",
    "FREEZE_WEEK = 52",
    "TREAT_COHORTS = {60: [0, 1, 2], 68: [3, 4, 5], 76: [6, 7, 8]}",
    "DEMAND_AMP = 300.0",
    "CHASE = 0.5",
    "TV_BASE, TV_BETA, TV_ADSTOCK, TV_SAT = 200.0, 600.0, 0.5, 0.004",
    "BASE_SALES, GEO_SD, NOISE_SD = 3000.0, 100.0, 15.0",
]

# Mechanism lines: linear zero-carryover experiment channel, tv adstock, the
# demand-chasing history, and the frozen baseline + constant boost.
DGP_FORMULA_NEEDLES = [
    "+ BETA_SEARCH * search",
    "carry = tv[t] + TV_ADSTOCK * carry",
    "search = SEARCH_BASE * (1 + CHASE * season[t])",
    "search = SEARCH_BASE + (BOOST if boosted else 0.0)",
    "DEMAND_AMP * season[t]",
]

SAMPLER_NEEDLES = [
    "chains=4, draws=500, tune=3000",
    "target_accept=0.999",
    'nuts_sampler_kwargs={"max_treedepth": 14}',
    "random_seed=SEED",
]

INTEGRATION_NEEDLES = [
    "add_lift_test_measurements(",
    "mmm_plain = fit_mmm()",
    "mmm_cal = fit_mmm(df_lift)",
    "aggregation_result=simple,\n    scale=G_TREATED,",
    "assert div == 0",
    "assert mtd == 0",
    "assert max_rhat_model < 1.01",
    "assert rhat_roi < 1.01",
    "assert ess_bulk >= ess_floors[name]",
    "assert ess_tail >= 300",
    'ess_floors = {"plain": 300, "calibrated": 800}',
    "assert err_cal < err_plain",
    "assert width_cal < 0.8 * width_plain",
]


def _notebook_source() -> str:
    nb = _read_notebook(NB)
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )


def _build_panel() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    geo_start = {g: w for w, gg in TREAT_COHORTS.items() for g in gg}
    season = np.sin(2 * np.pi * np.arange(N_WEEKS) / 52.0)
    rows = []
    for g in range(N_GEOS):
        tv = TV_BASE * (1 + 0.1 * rng.random(N_WEEKS))
        ad = np.zeros(N_WEEKS)
        carry = 0.0
        for t in range(N_WEEKS):
            carry = tv[t] + TV_ADSTOCK * carry
            ad[t] = carry
        tvc = TV_BETA * (1 - np.exp(-TV_SAT * ad))
        geo_level = rng.normal(0, GEO_SD)
        for t in range(N_WEEKS):
            if t < FREEZE_WEEK:
                search = SEARCH_BASE * (1 + CHASE * season[t])
            else:
                boosted = g in geo_start and t >= geo_start[g]
                search = SEARCH_BASE + (BOOST if boosted else 0.0)
            sales = (
                BASE_SALES
                + geo_level
                + DEMAND_AMP * season[t]
                + BETA_SEARCH * search
                + tvc[t]
                + rng.normal(0, NOISE_SD)
            )
            rows.append((g, t, sales, search, tv[t], geo_start.get(g, 0)))
    return pd.DataFrame(
        rows, columns=["geo", "week", "sales", "search_spend", "tv_spend", "first_treat"]
    )


def _parse_number(output: str, label_regex: str) -> float:
    match = re.search(label_regex, output)
    assert match, f"notebook output lost the line matching {label_regex!r}"
    return float(match.group(1).replace(",", ""))


class TestNotebookSourceSync:
    def test_dgp_constants_match(self):
        src = _notebook_source()
        missing = [n for n in DGP_CONSTANT_NEEDLES if n not in src]
        assert not missing, f"notebook DGP constants drifted from the test copy: {missing}"

    def test_dgp_formula_lines_present(self):
        src = _notebook_source()
        missing = [n for n in DGP_FORMULA_NEEDLES if n not in src]
        assert not missing, f"notebook DGP mechanism lines drifted: {missing}"

    def test_sampler_config_pinned(self):
        src = _notebook_source()
        missing = [n for n in SAMPLER_NEEDLES if n not in src]
        assert not missing, f"notebook sampler configuration drifted: {missing}"

    def test_integration_cells_present(self):
        src = _notebook_source()
        missing = [n for n in INTEGRATION_NEEDLES if n not in src]
        assert not missing, f"notebook integration/acceptance cells missing: {missing}"

    def test_no_kernelspec_metadata(self):
        nb = _read_notebook(NB)
        assert "kernelspec" not in nb["metadata"], (
            "committed notebooks stay kernelspec-free; local execution names the "
            "venv kernel explicitly"
        )


class TestRenderedOutputs:
    def test_acceptance_checks_passed_in_committed_run(self):
        assert_quotes_in_rendered(
            NB, ["all acceptance checks passed", "both fits done"], surface="output"
        )

    def test_money_plot_output_present(self):
        nb = _read_notebook(NB)
        has_image = any(
            "image/png" in (out.get("data") or {})
            for c in nb["cells"]
            if c["cell_type"] == "code"
            for out in c.get("outputs", [])
        )
        assert has_image, "the committed money-plot image output is missing"


@pytest.fixture(scope="module")
def fitted():
    panel = _build_panel()
    res = CallawaySantAnna().fit(
        panel, outcome="sales", unit="geo", time="week", first_treat="first_treat"
    )
    return panel, res


class TestDidNumbersRecompute:
    """Recompute the DiD/export side with diff-diff only; match printed numbers."""

    def test_total_and_att_match_notebook(self, fitted):
        panel, res = fitted
        total = res.aggregate("total")
        out = notebook_output_text(NB)
        printed_att = _parse_number(out, r"per-geo-week ATT: ([\d.,-]+)")
        printed_total = _parse_number(out, r"CS total incremental sales:\s+([\d.,-]+)")
        np.testing.assert_allclose(printed_att, res.overall_att, rtol=2e-3)
        np.testing.assert_allclose(printed_total, total.att[0], rtol=2e-3)
        mass = int(((panel.first_treat > 0) & (panel.week >= panel.first_treat)).sum())
        assert total.n[0] == mass

    def test_lift_row_matches_notebook(self, fitted):
        # Every lift-row FIELD is pinned individually (not just the ROI ratio,
        # where the treated-geo multiplier would cancel): x/delta_x against the
        # frozen-baseline national-weekly grain, delta_y/sigma against the
        # aggregate('simple') container scaled by the treated-geo count.
        panel, res = fitted
        simple = res.aggregate("simple")
        g_treated = int((panel.first_treat > 0).sum() / N_WEEKS)
        assert g_treated == 9
        df = to_pymc_marketing_lift_test(
            channel="search",
            x=SEARCH_BASE * N_GEOS,
            delta_x=BOOST * g_treated,
            aggregation_result=simple,
            scale=g_treated,
        )
        assert float(df.x[0]) == SEARCH_BASE * N_GEOS == 1500.0
        assert float(df.delta_x[0]) == BOOST * g_treated == 450.0
        np.testing.assert_allclose(float(df.delta_y[0]), res.overall_att * g_treated, rtol=1e-12)
        np.testing.assert_allclose(float(df.sigma[0]), res.overall_se * g_treated, rtol=1e-12)
        out = notebook_output_text(NB)
        printed_roi = _parse_number(out, r"DiD measurement:\s+([\d.,-]+)")
        np.testing.assert_allclose(printed_roi, df.delta_y[0] / df.delta_x[0], atol=5e-3)

    def test_notebook_states_linearity_scope_of_lift_row(self):
        # The staggered-average -> single-transition construction is exact only
        # under this simulation's linear homogeneous response; the notebook must
        # keep saying so (round-4 review: an unscoped "generalizes" claim would
        # transport a mixed-regime average onto one spend step of a saturating
        # curve).
        from tests._tutorial_drift import notebook_markdown

        md = notebook_markdown(NB)
        assert "linear and homogeneous" in md
        assert "boost all test geos simultaneously" in md
        # The nonlinear-case guidance must keep the estimator fit on the FULL
        # panel (a post-only refit strips every cohort's pre-treatment baseline
        # and leaves Callaway-Sant'Anna unidentified - round-5 review) and
        # restrict only the post-fit aggregation.
        assert "keep the FULL panel in the estimator fit" in md
        assert "re-fit the DiD restricted" not in md
