"""Drift detection for Tutorial 14 (``docs/tutorials/14_continuous_did.ipynb``).

The tutorial fits ``ContinuousDiD`` on a panel where the true dose-response is
ATT(d) = 1 + 2d. Its claims:

1. **Truth check.** Overall ATT 4.55 [4.24, 4.86] against a true 4.68; overall
   ACRT 2.15 [1.96, 2.34] against a true 2.0.
2. **The curve is evaluated on a bounded grid**, percentiles 10 to 99 of the
   dose (0.90 to 4.80), although the largest dose is 7.08.
3. **The overall ATT does not depend on the spline**; the overall ACRT does,
   and more flexibility is not automatically closer (0.67 / 0.91 / 0.97
   against a true 0.77 on log-shaped data).
4. **The binary analysis returns the same overall ATT with one cohort**,
   standard error included, and nothing else. With two cohorts the two overall
   numbers differ (4.7070 against 4.6741): they weight the same cohort-period
   effects differently.
5. The prose states strong parallel trends as an assumption about outcomes
   under treatment (no selection into dose on gains), and names the result
   attribute ``results.att``.

Analytical quantities are deterministic. The bootstrap is pinned loosely
because its draws are backend-specific.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from diff_diff import (
    CallawaySantAnna,
    ContinuousDiD,
    compute_honest_did,
    generate_continuous_did_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/14_continuous_did.ipynb"

ATT_INTERCEPT = 1.0
ATT_SLOPE = 2.0
FIT = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat", dose="dose")


def _panel(att_function="linear", n_periods=4, cohort_periods=(2,)):
    return generate_continuous_did_data(
        n_units=500,
        n_periods=n_periods,
        cohort_periods=list(cohort_periods),
        never_treated_frac=0.3,
        att_function=att_function,
        att_intercept=ATT_INTERCEPT,
        att_slope=ATT_SLOPE,
        seed=42,
    )


@pytest.fixture(scope="module")
def data():
    return _panel()


@pytest.fixture(scope="module")
def doses(data):
    return data.loc[data["first_treat"] > 0].groupby("unit")["dose"].first()


@pytest.fixture(scope="module")
def results(data):
    return ContinuousDiD(seed=42).fit(data, **FIT)


class TestFit:
    def test_overall_parameters_cover_truth(self, results, doses):
        true_att = ATT_INTERCEPT + ATT_SLOPE * doses.mean()
        assert true_att == pytest.approx(4.675, abs=2e-3)
        assert results.att == pytest.approx(4.5468, abs=5e-4)
        assert results.se == pytest.approx(0.1575, abs=5e-4)
        assert results.conf_int[0] <= true_att <= results.conf_int[1]
        assert results.overall_acrt == pytest.approx(2.1487, abs=5e-4)
        lower, upper = results.overall_acrt_conf_int
        assert lower == pytest.approx(1.958, abs=2e-3) and upper == pytest.approx(2.339, abs=2e-3)
        assert lower <= ATT_SLOPE <= upper
        assert not hasattr(results, "overall_att")
        assert_quotes_in_rendered(
            NB,
            [
                "Overall ATT        4.547       [4.238, 4.855]    4.675",
                "Overall ACRT       2.149       [1.958, 2.339]    2.000",
            ],
            surface="output",
        )

    def test_curve_grid_is_bounded(self, results, doses):
        curve = results.dose_response_att.to_dataframe()
        assert len(curve) == 90
        assert curve["dose"].min() == pytest.approx(doses.quantile(0.10), abs=1e-6)
        assert curve["dose"].max() == pytest.approx(doses.quantile(0.99), abs=1e-6)
        # the grid endpoint is not the limit of the observed doses
        assert doses.max() > curve["dose"].max()
        md = notebook_markdown(NB)
        assert "where the default grid stops evaluating, not where the data end" in md
        assert "the curve is not supported by data beyond it" not in md
        assert curve["dose"].min() == pytest.approx(0.897, abs=2e-3)
        assert curve["dose"].max() == pytest.approx(4.796, abs=2e-3)
        assert doses.max() == pytest.approx(7.08, abs=5e-3)
        # the curve tracks 1 + 2d over the grid
        truth = ATT_INTERCEPT + ATT_SLOPE * curve["dose"]
        assert (curve["effect"] - truth).abs().max() < 0.3
        # ACRT band is wider at the low end than in the middle
        acrt = results.dose_response_acrt
        assert acrt.se[0] > 2 * np.min(acrt.se)


class TestEventStudyAndOptions:
    def test_event_study(self):
        fit = ContinuousDiD(seed=42).fit(_panel(n_periods=6, cohort_periods=(3, 5)), **FIT)
        es = fit.aggregate("event_study").to_dataframe().set_index("event_time")
        assert list(es.index) == [-3, -2, -1, 0, 1, 2, 3]
        assert not es["is_reference"].any()  # varying base: no omitted reference row
        pre = es.loc[-3:-1]
        assert ((pre["att"] / pre["se"]).abs() < 1.2).all()
        assert es.loc[0:3, "att"].between(4.5, 4.8).all()

    def test_spline_changes_the_slope_not_the_overall_att(self):
        log_data = _panel(att_function="log")
        log_doses = log_data.loc[log_data["first_treat"] > 0].groupby("unit")["dose"].first()
        assert (1.0 + 2.0 * np.log1p(log_doses)).mean() == pytest.approx(2.990, abs=2e-3)
        true_acrt = (2.0 / (1.0 + log_doses)).mean()
        assert true_acrt == pytest.approx(0.771, abs=2e-3)
        fits = [
            ContinuousDiD(seed=42, degree=d, num_knots=k).fit(log_data, **FIT)
            for d, k in ((1, 0), (3, 0), (3, 2))
        ]
        atts = [f.att for f in fits]
        assert max(atts) - min(atts) < 1e-8 and atts[0] == pytest.approx(2.862, abs=1e-3)
        acrts = [f.overall_acrt for f in fits]
        np.testing.assert_allclose(acrts, [0.672, 0.910, 0.965], atol=2e-3)
        # linear understates, both cubic fits overstate, each by 1 to 2 standard errors
        assert acrts[0] < true_acrt < acrts[1] < acrts[2]
        for fit in fits:
            assert 1.0 < abs(fit.overall_acrt - true_acrt) / fit.overall_acrt_se < 2.0
        assert fits[0].overall_acrt_se < fits[1].overall_acrt_se < fits[2].overall_acrt_se

    def test_control_groups_agree_here(self):
        panel = _panel(n_periods=6, cohort_periods=(3, 5))
        never = ContinuousDiD(control_group="never_treated", seed=42).fit(panel, **FIT)
        not_yet = ContinuousDiD(control_group="not_yet_treated", seed=42).fit(panel, **FIT)
        assert never.att == pytest.approx(4.707, abs=2e-3)
        assert not_yet.att == pytest.approx(4.709, abs=2e-3)

    def test_bootstrap_agrees_with_analytical(self, results, data):
        boot = ContinuousDiD(n_bootstrap=999, seed=42).fit(data, **FIT)
        assert boot.att == pytest.approx(results.att, rel=1e-9)
        assert boot.se == pytest.approx(results.se, rel=0.12)
        assert boot.overall_acrt_se == pytest.approx(results.overall_acrt_se, rel=0.12)

    def test_binary_analysis_returns_the_same_overall_att_with_one_cohort(self, results, data):
        binary = CallawaySantAnna(control_group="never_treated").fit(
            data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert binary.att == pytest.approx(results.att, abs=1e-8)
        assert binary.se == pytest.approx(results.se, abs=1e-8)
        assert_quotes_in_rendered(
            NB,
            [
                "one cohort    ContinuousDiD            4.5468   0.1575",
                "one cohort    CallawaySantAnna         4.5468   0.1575",
            ],
            surface="output",
        )

    def test_binary_analysis_differs_with_staggered_cohorts(self):
        """Same cohort-period effects, different weights: ContinuousDiD averages
        each cohort over its post-treatment periods and weights cohorts by size;
        Callaway-Sant'Anna weights every cohort-period cell by cohort size."""
        two_cohorts = _panel(n_periods=6, cohort_periods=(3, 5))
        keys = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
        continuous = ContinuousDiD(seed=42).fit(two_cohorts, dose="dose", **keys)
        binary = CallawaySantAnna(control_group="never_treated").fit(two_cohorts, **keys)
        assert continuous.att == pytest.approx(4.7070, abs=5e-4)
        assert binary.att == pytest.approx(4.6741, abs=5e-4)

        sizes = two_cohorts.drop_duplicates("unit")["first_treat"].value_counts()
        post = {k: v["effect"] for k, v in binary.group_time_effects.items() if k[1] >= k[0]}
        cohorts = sorted({g for g, _ in post})
        per_cohort = {g: np.mean([e for (g2, _), e in post.items() if g2 == g]) for g in cohorts}
        n_treated = sum(sizes[g] for g in cohorts)
        by_cohort = sum(sizes[g] / n_treated * per_cohort[g] for g in cohorts)
        by_cell = sum(sizes[g] * e for (g, _), e in post.items()) / sum(sizes[g] for g, _ in post)
        assert by_cohort == pytest.approx(continuous.att, abs=1e-6)
        assert by_cell == pytest.approx(binary.att, abs=1e-6)
        assert_quotes_in_rendered(
            NB,
            [
                "two cohorts   ContinuousDiD            4.7070   0.1370",
                "two cohorts   CallawaySantAnna         4.6741   0.1413",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert "On the two-cohort panel they differ, 4.71 against 4.67" in md
        assert "the\nevent study of a `ContinuousDiD` fit is not accepted there" in md
        assert 'fit `CallawaySantAnna(base_period="universal")` on\nthe binary treatment' in md
        with pytest.raises(TypeError):
            compute_honest_did(
                ContinuousDiD(base_period="universal", seed=42).fit(
                    two_cohorts, dose="dose", **keys
                )
            )
        # the workflow the prose recommends: a universal-base binary fit gives
        # bounds without the base-period warning; the default base warns
        for base_period, warns in (("universal", False), ("varying", True)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = CallawaySantAnna(control_group="never_treated", base_period=base_period).fit(
                    two_cohorts, **keys
                )
                bounds = compute_honest_did(
                    fit.aggregate("event_study"), method="relative_magnitude", M=1.0
                )
            assert np.isfinite(bounds.lb) and np.isfinite(bounds.ub) and bounds.lb < bounds.ub
            assert any("base_period" in str(w.message) for w in caught) is warns


class TestNotebookSurface:
    def test_prose(self):
        md = notebook_markdown(NB)
        for needle in (
            "workers do not sort into doses according to\n  how much they would gain",
            "It is an assumption about outcomes *under treatment*",
            "in code they are `results.att` and\n`results.overall_acrt`",
            "The **overall ATT does not depend on the spline**",
            "**not automatically closer to the truth**",
            "different assumptions, not a weaker and a stronger version of one",
            "keep them inside the range of\ndoses you observe",
            # a lowest-dose comparison is relative to that group, with its own conditions
            "It requires a balanced panel, a single treatment date, no never-treated units and at\n  least two units at the minimum dose.",
            "Reading it as the effect of\nmoving from the lowest dose to dose *d* needs strong parallel trends.",
            # parallel trends alone identifies own-dose effects, not only the overall ATT
            "| ATT(d\\|d) |",
            "the\nheight of the left curve at *d* read as ATT(d|d)",
            "parallel trends identifies the overall ATT and each dose group's own\n  effect, ATT(d|d)",
            "The estimator requires a balanced panel.",
        ):
            assert needle in md, needle
        for stale in (
            "overall_att",
            "since 4.0",
            "removed at 4.0",
            "M-025",
            "only the overall ATT has a causal\nreading",
        ):
            assert stale not in md, stale

    def test_code(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "ATT_INTERCEPT = 1.0",
            "ATT_SLOPE = 2.0",
            "att_intercept=ATT_INTERCEPT,",
            'es = results_two_cohorts.aggregate("event_study")',
            "plot_event_study(es,",
            "n_bootstrap=999",
        ):
            assert needle in code, needle
        assert "errorbar" not in code  # the event study uses the shared plotting idiom
