"""Drift guards for docs/tutorials/30_mmm_calibration_meridian.ipynb.

Same protection pattern as the t29 twin (see its module docstring): the notebook
is CI-excluded from execution (google-meridian conflicts with the PyMC tutorial's
environment), so its committed surface is guarded here with diff-diff-only
recomputation - DGP constant/formula sync, sampler-config needles,
integration-content presence, and rendered-number cross-checks with loose bands.
"""

import re

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna
from diff_diff.mmm import meridian_calibration_mask, to_meridian_roi_prior
from tests._tutorial_drift import (
    _read_notebook,
    assert_quotes_in_rendered,
    notebook_output_text,
)

NB = "docs/tutorials/30_mmm_calibration_meridian.ipynb"

# ---- locked DGP constants (must equal the notebook's copy verbatim) ----
SEED = 30301
N_GEOS, N_WEEKS = 12, 104
BETA_SEARCH, LAUNCH_SPEND = 2.5, 150.0
LAUNCH_COHORTS = {30: [0, 1, 2], 44: [3, 4, 5], 58: [6, 7]}
DEMAND_AMP = 80.0
TV_BASE, TV_BETA, TV_ADSTOCK, TV_SAT = 250.0, 800.0, 0.4, 0.003
BASE_SALES, GEO_SD, NOISE_SD, SHOCK_SD = 5000.0, 150.0, 20.0, 60.0

DGP_CONSTANT_NEEDLES = [
    "SEED = 30301",
    "N_GEOS, N_WEEKS = 12, 104",
    "BETA_SEARCH = 2.5",
    "LAUNCH_SPEND = 150.0",
    "LAUNCH_COHORTS = {30: [0, 1, 2], 44: [3, 4, 5], 58: [6, 7]}",
    "DEMAND_AMP = 80.0",
    "TV_BASE, TV_BETA, TV_ADSTOCK, TV_SAT = 250.0, 800.0, 0.4, 0.003",
    "BASE_SALES, GEO_SD, NOISE_SD, SHOCK_SD = 5000.0, 150.0, 20.0, 60.0",
]

# Mechanism lines: linear zero-carryover experiment channel, launched-geos-only
# spend, the demand ramp, and the common national shocks.
DGP_FORMULA_NEEDLES = [
    "+ BETA_SEARCH * search",
    "carry = tv[t] + TV_ADSTOCK * carry",
    "search = LAUNCH_SPEND if launched else 0.0",
    "DEMAND_AMP * ramp[t] + shock[t]",
    "shock = rng.normal(0, SHOCK_SD, N_WEEKS)",
]

SAMPLER_NEEDLES = [
    "n_chains=4, n_adapt=500, n_burnin=500, n_keep=500, seed=SEED",
]

INTEGRATION_NEEDLES = [
    "exec(code, namespace)",
    "mer_default = fit_meridian(m_spec.ModelSpec())",
    "mer_calibrated = fit_meridian(model_spec_calibrated)",
    "to_meridian_roi_prior(aggregation_result=total, spend=total_spend)",
    "assert np.isfinite(max_rhat_model) and max_rhat_model < 1.2",
    "assert np.isfinite(rhat_roi) and rhat_roi < 1.05",
    "assert err_calibrated < err_default",
    "assert width_calibrated < 0.8 * width_default",
    "assert total.n[0] == spend_positive",
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
    geo_start = {g: w for w, gg in LAUNCH_COHORTS.items() for g in gg}
    ramp = np.clip((np.arange(N_WEEKS) - 26) / 52.0, 0, None)
    shock = rng.normal(0, SHOCK_SD, N_WEEKS)
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
            launched = g in geo_start and t >= geo_start[g]
            search = LAUNCH_SPEND if launched else 0.0
            sales = (
                BASE_SALES
                + geo_level
                + DEMAND_AMP * ramp[t]
                + shock[t]
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

    def test_generated_snippet_shown(self):
        assert_quotes_in_rendered(
            NB,
            ["Generated by diff_diff.mmm.to_meridian_roi_prior"],
            surface="output",
        )


@pytest.fixture(scope="module")
def fitted():
    panel = _build_panel()
    res = CallawaySantAnna().fit(
        panel, outcome="sales", unit="geo", time="week", first_treat="first_treat"
    )
    return panel, res


class TestDidNumbersRecompute:
    """Recompute the DiD/export side with diff-diff only; match printed numbers."""

    def test_total_and_prior_match_notebook(self, fitted):
        panel, res = fitted
        total = res.aggregate("total")
        total_spend = float(panel.search_spend.sum())
        prior = to_meridian_roi_prior(aggregation_result=total, spend=total_spend)
        out = notebook_output_text(NB)
        printed_total = _parse_number(out, r"CS total incremental sales:\s+([\d.,-]+)")
        printed_roi = _parse_number(out, r"experiment ROI: ([\d.,-]+)")
        np.testing.assert_allclose(printed_total, total.att[0], rtol=2e-3)
        np.testing.assert_allclose(printed_roi, prior.roi_mean, atol=5e-3)
        # launch-design alignment: the total's mass is the spend-positive count
        assert total.n[0] == int((panel.search_spend > 0).sum())

    def test_mask_window_matches_notebook(self, fitted):
        panel, _ = fitted
        geo_start = {g: w for w, gg in LAUNCH_COHORTS.items() for g in gg}
        first_launch = min(geo_start.values())
        times = [str(d.date()) for d in pd.date_range("2024-01-01", periods=N_WEEKS, freq="W-MON")]
        mask = meridian_calibration_mask(
            media_times=times,
            media_channels=["search", "tv"],
            channel="search",
            window=(times[first_launch], times[-1]),
        )
        out = notebook_output_text(NB)
        printed_true = _parse_number(out, r"search weeks in window: (\d+)")
        assert int(printed_true) == int(mask[:, 0].sum())
        # every spend-positive week is inside the window (the time-axis
        # containment the TIME x CHANNEL mask expresses)
        assert panel.loc[panel.search_spend > 0, "week"].min() >= first_launch
