"""Drift detection for Tutorial 11 (``docs/tutorials/11_imputation_did.ipynb``).

The tutorial fits ``ImputationDiD`` on a staggered panel with a constant true
effect of 2.0. Its claims:

1. **Truth check.** ATT 1.98, 95% CI [1.91, 2.05], which contains 2.0; the
   generator is called with ``dynamic_effects=False`` so 2.0 is the truth.
2. **Pre-treatment evidence is not a verdict.** Pre-period coefficients are
   small (largest 1.4 SE from zero); the joint test has p = 0.285.
3. **Precision, and its condition.** The SE is 70% of Callaway-Sant'Anna's. The
   prose attributes the gain to spherical errors (BJS Theorem 1), not to
   homogeneous treatment effects.
4. **Anticipation is not a robustness switch.** Declaring one period of
   anticipation on data without any lowers the ATT to 1.67, by averaging 210
   zero-effect unit-periods in with 1,020 treated ones.
5. **aux_partition** changes the SE in the fourth decimal and never the ATT.

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
    SunAbraham,
    generate_staggered_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/11_imputation_did.ipynb"

TRUE_ATT = 2.0
FIT = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


@pytest.fixture(scope="module")
def data():
    return generate_staggered_data(
        n_units=300, n_periods=10, treatment_effect=TRUE_ATT, dynamic_effects=False, seed=42
    )


@pytest.fixture(scope="module")
def results(data):
    return ImputationDiD(pretrends=True).fit(data, **FIT)


class TestFit:
    def test_truth_is_what_the_notebook_says(self, data):
        treated = data[data["treated"] == 1]
        assert len(treated) == 1020
        np.testing.assert_allclose(treated["true_effect"].to_numpy(), TRUE_ATT)
        sizes = data.groupby("unit")["first_treat"].first().value_counts().to_dict()
        assert sizes == {0: 90, 3: 60, 5: 75, 7: 75}

    def test_estimate_covers_truth(self, results, data):
        assert results.att == pytest.approx(1.9817, abs=5e-4)
        assert results.se == pytest.approx(0.0369, abs=2e-4)
        assert results.conf_int[0] == pytest.approx(1.909, abs=1e-3)
        assert results.conf_int[1] == pytest.approx(2.054, abs=1e-3)
        assert results.conf_int[0] <= TRUE_ATT <= results.conf_int[1]
        # pretrends=True adds pre-period coefficients and leaves the overall fit alone
        plain = ImputationDiD().fit(data, **FIT)
        assert plain.att == pytest.approx(results.att, rel=1e-12)
        assert plain.se == pytest.approx(results.se, rel=1e-12)
        # "Control units: 300": every unit has at least one untreated observation
        assert results.n_control_units == 300
        assert_quotes_in_rendered(
            NB, ["Estimate: 1.982  (SE 0.037)", "95% CI:   [1.909, 2.054]"], surface="output"
        )


class TestPreTreatmentEvidence:
    def test_event_study(self, results):
        es = results.aggregate("event_study").to_dataframe().set_index("event_time")
        assert bool(es.loc[-1, "is_reference"]) and es.loc[-1, "att"] == 0.0
        post = es.loc[0:6]
        assert (post["att"] - TRUE_ATT).abs().max() < 0.07
        # standard errors grow with the horizon as cohorts drop out
        assert post.loc[5, "se"] > 1.5 * post.loc[0, "se"]
        pre = es.loc[-7:-2]
        t_stats = (pre["att"] / pre["se"]).abs()
        assert t_stats.idxmax() == -5
        assert t_stats.max() == pytest.approx(1.41, abs=0.02)

    def test_joint_pretrend_test(self, results):
        test = results.pretrend_test(n_leads=3)
        assert test["f_stat"] == pytest.approx(1.269, abs=2e-3)
        assert test["p_value"] == pytest.approx(0.285, abs=2e-3)
        assert test["n_leads"] == 3
        assert_quotes_in_rendered(
            NB, ["F statistic: 1.269", "p-value:     0.285"], surface="output"
        )


class TestCohortsAndComparison:
    def test_cohort_effects(self, results):
        groups = results.aggregate("group").to_dataframe().set_index("label")
        np.testing.assert_allclose(
            groups.loc[[3, 5, 7], "att"].to_numpy(), [2.005, 1.999, 1.908], atol=1e-3
        )
        assert (
            (groups["conf_int_lower"] <= TRUE_ATT) & (TRUE_ATT <= groups["conf_int_upper"])
        ).all()

    def test_precision_against_callaway_and_sun_abraham(self, results, data):
        cs = CallawaySantAnna().fit(data, **FIT)
        sa = SunAbraham().fit(data, **FIT)
        assert cs.att == pytest.approx(1.9754, abs=5e-4)
        assert cs.se == pytest.approx(0.0528, abs=2e-4)
        # CS and SA make the same comparisons here
        assert sa.att == pytest.approx(cs.att, abs=1e-3)
        assert sa.se == pytest.approx(cs.se, abs=5e-4)
        for fit in (cs, sa):
            assert fit.conf_int[0] <= TRUE_ATT <= fit.conf_int[1]
        assert results.se / cs.se == pytest.approx(0.70, abs=0.01)
        assert_quotes_in_rendered(
            NB,
            ["Imputation SE as a share of the Callaway-Sant'Anna SE: 70%"],
            surface="output",
        )


class TestOptions:
    def test_anticipation_averages_in_zero_effect_periods(self, results, data):
        fit = ImputationDiD(anticipation=1).fit(data, **FIT)
        assert fit.att == pytest.approx(1.674, abs=1e-3)
        # 210 treated units each contribute one extra, zero-effect, unit-period
        assert TRUE_ATT * 1020 / (1020 + 210) == pytest.approx(1.66, abs=5e-3)
        assert abs(fit.att - TRUE_ATT * 1020 / 1230) < 0.03
        assert fit.att < results.att - 0.25

    def test_aux_partition(self, data):
        fits = {
            p: ImputationDiD(aux_partition=p).fit(data, **FIT)
            for p in ("cohort_horizon", "cohort", "horizon")
        }
        atts = [f.att for f in fits.values()]
        assert max(atts) - min(atts) < 1e-12
        assert fits["cohort"].se == pytest.approx(fits["cohort_horizon"].se, rel=1e-9)
        assert abs(fits["horizon"].se - fits["cohort_horizon"].se) < 5e-4


class TestNotebookSurface:
    def test_prose(self):
        md = notebook_markdown(NB)
        for needle in (
            # the efficiency condition is on the errors, not on the effects
            "It is **not** a condition on the treatment effects",
            "equal variance and are uncorrelated across units and periods (Theorem 1)",
            "Nothing is assumed about the treatment effects themselves.",
            # anticipation
            "it is the second number that is wrong here",
            "2.0 x 1020 / 1230 = 1.66",
            # pre-trends are not pass/fail
            "not as confirmation of it",
        ):
            assert needle in md, needle
        for stale in ("homogeneous", "~50% shorter"):
            assert stale not in md, stale

    def test_code(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "TRUE_ATT = 2.0",
            "treatment_effect=TRUE_ATT,",
            "dynamic_effects=False,",
            "ImputationDiD(pretrends=True)",
            'results.aggregate("group")',
            "plot_event_study(es,",
        ):
            assert needle in code, needle
