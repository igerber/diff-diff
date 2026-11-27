"""Drift detection for Tutorial 12 (``docs/tutorials/12_two_stage_did.ipynb``).

The tutorial fits ``TwoStageDiD`` on the same panel as Tutorial 11 (constant
true effect 2.0). Its claims:

1. **Truth check.** ATT 1.98, 95% CI [1.91, 2.05], which contains 2.0.
2. **Identical point estimates to Imputation DiD**, overall and at every
   post-treatment horizon; the treatment-effect standard errors agree within
   2% (GMM sandwich against the conservative estimator).
3. **Pre-treatment rows are a different construction**: coefficients differ
   and the standard errors are about a factor of two apart.
4. **Observation-level effects** average to the ATT and are individually
   noisy (SD 0.56 with a true effect of exactly 2.0 everywhere).
5. ``pretrends=True`` is what puts pre-treatment coefficients in the event
   study; ``TwoStageDiD`` has no ``pretrend_test()``.

Everything is deterministic given the seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diff_diff import (
    CallawaySantAnna,
    ImputationDiD,
    TwoStageDiD,
    generate_staggered_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/12_two_stage_did.ipynb"

TRUE_ATT = 2.0
FIT = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


@pytest.fixture(scope="module")
def data():
    return generate_staggered_data(
        n_units=300, n_periods=10, treatment_effect=TRUE_ATT, dynamic_effects=False, seed=42
    )


@pytest.fixture(scope="module")
def results(data):
    return TwoStageDiD(pretrends=True).fit(data, **FIT)


@pytest.fixture(scope="module")
def imputation(data):
    return ImputationDiD(pretrends=True).fit(data, **FIT)


class TestFit:
    def test_estimate_covers_truth(self, results):
        assert results.att == pytest.approx(1.9817, abs=5e-4)
        assert results.se == pytest.approx(0.0370, abs=2e-4)
        assert results.conf_int[0] <= TRUE_ATT <= results.conf_int[1]
        assert_quotes_in_rendered(
            NB, ["Estimate: 1.982  (SE 0.037)", "95% CI:   [1.909, 2.054]"], surface="output"
        )

    def test_pretrends_flag_controls_the_pre_period_rows(self, results, data):
        with_pre = results.aggregate("event_study").to_dataframe()
        without = TwoStageDiD().fit(data, **FIT).aggregate("event_study").to_dataframe()
        assert with_pre["event_time"].min() == -7
        assert without["event_time"].min() == -1  # starts at the reference period
        assert not hasattr(results, "pretrend_test")


class TestAgainstImputation:
    def test_point_estimates_are_identical(self, results, imputation):
        assert results.att == pytest.approx(imputation.att, abs=1e-10)
        ts = results.aggregate("event_study").to_dataframe().set_index("event_time")
        im = imputation.aggregate("event_study").to_dataframe().set_index("event_time")
        np.testing.assert_allclose(
            ts.loc[0:6, "att"].to_numpy(), im.loc[0:6, "att"].to_numpy(), atol=1e-10
        )
        assert_quotes_in_rendered(
            NB, ["Two-stage minus imputation estimate: 0.0e+00"], surface="output"
        )

    def test_treatment_effect_standard_errors_agree_within_two_percent(self, results, imputation):
        assert results.se / imputation.se == pytest.approx(1.0, abs=0.02)
        ts = results.aggregate("event_study").to_dataframe().set_index("event_time")
        im = imputation.aggregate("event_study").to_dataframe().set_index("event_time")
        ratio = (ts.loc[0:6, "se"] / im.loc[0:6, "se"]).to_numpy()
        assert ratio.min() >= 0.999 and ratio.max() <= 1.02

    def test_pre_treatment_rows_differ(self, results, imputation):
        ts = results.aggregate("event_study").to_dataframe().set_index("event_time")
        im = imputation.aggregate("event_study").to_dataframe().set_index("event_time")
        ratio = (ts.loc[-7:-2, "se"] / im.loc[-7:-2, "se"]).to_numpy()
        assert ratio.max() < 0.6 and ratio.min() > 0.35  # "a factor of two apart"
        assert (ts.loc[-7:-2, "att"] - im.loc[-7:-2, "att"]).abs().max() > 0.03

    def test_callaway_is_about_forty_percent_wider(self, results, data):
        cs = CallawaySantAnna().fit(data, **FIT)
        assert cs.se / results.se == pytest.approx(1.43, abs=0.03)
        assert cs.conf_int[0] <= TRUE_ATT <= cs.conf_int[1]


class TestObservationLevelEffects:
    def test_effects_average_to_the_att_and_are_noisy(self, results, data):
        effects = results.treatment_effects
        assert len(effects) == 1020
        assert list(effects.columns) == ["unit", "period", "tau_hat", "rel_time", "weight"]
        assert (effects["tau_hat"] * effects["weight"]).sum() == pytest.approx(
            results.att, abs=1e-10
        )
        # each tau_hat carries that observation's own error (noise SD 0.5 in the data)
        assert effects["tau_hat"].std() == pytest.approx(0.563, abs=5e-3)
        np.testing.assert_allclose(data.loc[data["treated"] == 1, "true_effect"], TRUE_ATT)
        assert_quotes_in_rendered(
            NB, ["SD of tau_hat across observations: 0.563"], surface="output"
        )

    def test_cohorts_and_anticipation(self, results, data):
        groups = results.aggregate("group").to_dataframe().set_index("label")
        np.testing.assert_allclose(
            groups.loc[[3, 5, 7], "att"].to_numpy(), [2.005, 1.999, 1.908], atol=1e-3
        )
        fit = TwoStageDiD(anticipation=1).fit(data, **FIT)
        assert fit.att == pytest.approx(1.674, abs=1e-3)


class TestNotebookSurface:
    def test_prose(self):
        md = notebook_markdown(NB)
        for needle in (
            "it is not an\nindependent check on the estimate",
            "the two estimators report different\ncoefficients there",
            "Do not read the spread of `tau_hat` as heterogeneity in the effect.",
            "`TwoStageDiD` has no `pretrend_test()`",
            "The row at event time -1 is a zero marker for the plot",
            "*The R\n  Journal*, 14(3), 162-173.",
        ):
            assert needle in md, needle

    def test_code(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "TRUE_ATT = 2.0",
            "dynamic_effects=False,",
            "TwoStageDiD(pretrends=True)",
            'results.aggregate("group")',
            "plot_event_study(es,",
        ):
            assert needle in code, needle


def test_comparison_table_and_summary_are_scoped_to_treatment_effects():
    """The Imputation DiD column mixes two variance estimators (Theorem 3 for the
    treatment effects, the lead regression's cluster-robust one before
    treatment), so it is not labelled "conservative"; and the equality claim is
    for the overall and post-treatment estimates only."""
    assert_quotes_in_rendered(NB, ["effect  GMM SE  Imputation SE   ratio"], surface="output")
    md = notebook_markdown(NB)
    assert "Its overall estimate and post-treatment coefficients are identical to Imputation" in md
    assert "Its point estimates are identical to Imputation DiD's." not in md
    path = Path(__file__).resolve().parents[1] / NB
    code = "\n".join(
        "".join(c["source"])
        for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    )
    assert '"conservative SE"' not in code
