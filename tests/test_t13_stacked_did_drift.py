"""Drift detection for Tutorial 13 (``docs/tutorials/13_stacked_did.ipynb``).

The tutorial fits ``StackedDiD`` on a staggered panel with a constant true
effect of 2.0. Its claims:

1. **Truth check.** ATT 1.92, 95% CI [1.82, 2.02], which contains 2.0.
2. **The stack is inspectable.** 2,775 stacked rows; control Q-weights 0.60
   for the cohort-3 sub-experiment and 1.37 for the others, equal to (cohort
   share of treated units) / (sub-experiment share of control appearances).
3. **Trimming.** The window [-3, 3] drops cohort 7 and [-4, 4] also drops
   cohort 3; the asymmetric [-3, 2] keeps all three and returns the [-2, 2]
   overall estimate.
4. **Control definitions.** ``strict`` and ``never_treated`` coincide here
   (90 control units); the default uses 165.
5. **Anticipation** lowers the ATT to 1.54, the mean of four event-time
   coefficients of which one is a zero-effect period.
6. **Clustering.** ``unit_subexp`` gives the larger standard error here, not a
   smaller one.

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
    StackedDiD,
    generate_staggered_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/13_stacked_did.ipynb"

TRUE_ATT = 2.0
FIT = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


@pytest.fixture(scope="module")
def data():
    return generate_staggered_data(
        n_units=300, n_periods=10, treatment_effect=TRUE_ATT, dynamic_effects=False, seed=42
    )


@pytest.fixture(scope="module")
def results(data):
    return StackedDiD(kappa_pre=2, kappa_post=2).fit(data, **FIT)


class TestFit:
    def test_estimate_covers_truth(self, results):
        assert results.att == pytest.approx(1.9218, abs=5e-4)
        assert results.se == pytest.approx(0.0509, abs=2e-4)
        assert results.conf_int[0] == pytest.approx(1.822, abs=1e-3)
        assert results.conf_int[1] == pytest.approx(2.022, abs=1e-3)
        assert results.conf_int[0] <= TRUE_ATT <= results.conf_int[1]
        # 90 never-treated units plus the 75 of the last cohort
        assert results.n_control_units == 165
        assert_quotes_in_rendered(
            NB, ["Estimate: 1.922  (SE 0.051)", "95% CI:   [1.822, 2.022]"], surface="output"
        )

    def test_event_study(self, results):
        es = results.aggregate("event_study").to_dataframe().set_index("event_time")
        assert list(es.index) == [-2, -1, 0, 1, 2]
        assert bool(es.loc[-1, "is_reference"])
        np.testing.assert_allclose(
            es.loc[[-2, 0, 1, 2], "att"].to_numpy(), [-0.103, 1.945, 1.936, 1.885], atol=1e-3
        )
        post = es.loc[0:2]
        assert ((post["conf_int_lower"] <= TRUE_ATT) & (TRUE_ATT <= post["conf_int_upper"])).all()
        assert es.loc[-2, "conf_int_lower"] < 0 < es.loc[-2, "conf_int_upper"]

    def test_no_cohort_effects(self, results):
        with pytest.raises(ValueError, match="group"):
            results.aggregate("group")


class TestStack:
    def test_rows_and_q_weights(self, results, data):
        stacked = results.stacked_data
        assert len(data) == 3000 and len(stacked) == 2775
        rows = stacked.groupby("_sub_exp").size().to_dict()
        assert rows == {3.0: 1125, 5.0: 825, 7.0: 825}
        treated = {3: 60, 5: 75, 7: 75}
        controls = {3: 165, 5: 90, 7: 90}
        for cohort in (3, 5, 7):
            sub = stacked[stacked["_sub_exp"] == cohort]
            assert sub.loc[sub["_D_sa"] == 1, "unit"].nunique() == treated[cohort]
            assert sub.loc[sub["_D_sa"] == 0, "unit"].nunique() == controls[cohort]
            # (cohort share of treated) / (sub-experiment share of control appearances)
            expected = (treated[cohort] / 210) / (controls[cohort] / 345)
            q = sub.loc[sub["_D_sa"] == 0, "_Q_weight"]
            np.testing.assert_allclose(q.to_numpy(), expected, rtol=1e-9)
        np.testing.assert_allclose(stacked.loc[stacked["_D_sa"] == 1, "_Q_weight"], 1.0)
        assert (60 / 210) / (165 / 345) == pytest.approx(0.597, abs=1e-3)
        assert (75 / 210) / (90 / 345) == pytest.approx(1.369, abs=1e-3)


class TestWindowAndTrimming:
    def test_wider_windows_trim_cohorts(self, data):
        expected = {
            (1, 1): ([3, 5, 7], [], 1.928, 0.054),
            (2, 2): ([3, 5, 7], [], 1.922, 0.051),
            (3, 3): ([3, 5], [7], 2.005, 0.061),
            (4, 4): ([5], [3, 7], 1.986, 0.087),
        }
        for (pre, post), (kept, trimmed, att, se) in expected.items():
            estimator = StackedDiD(kappa_pre=pre, kappa_post=post)
            if trimmed:
                with pytest.warns(UserWarning, match="Trimmed"):
                    fit = estimator.fit(data, **FIT)
            else:
                fit = estimator.fit(data, **FIT)
            assert [int(g) for g in fit.groups] == kept
            assert [int(g) for g in fit.trimmed_groups] == trimmed
            assert fit.att == pytest.approx(att, abs=1e-3)
            assert fit.se == pytest.approx(se, abs=1e-3)
            assert fit.conf_int[0] <= TRUE_ATT <= fit.conf_int[1]

    def test_asymmetric_window_keeps_every_cohort(self, results, data):
        fit = StackedDiD(kappa_pre=3, kappa_post=2).fit(data, **FIT)
        assert [int(g) for g in fit.groups] == [3, 5, 7] and not list(fit.trimmed_groups)
        assert fit.att == pytest.approx(results.att, abs=1e-3)
        assert fit.se == pytest.approx(results.se, abs=1e-3)
        assert fit.aggregate("event_study").to_dataframe()["event_time"].min() == -3
        # identical post-treatment estimates: same units, controls, weights and reference
        both = [
            f.aggregate("event_study").to_dataframe().set_index("event_time")["att"]
            for f in (results, fit)
        ]
        np.testing.assert_allclose(both[0].loc[[0, 1, 2]], both[1].loc[[0, 1, 2]], atol=1e-10)
        md = notebook_markdown(NB)
        assert "That is not automatic." in md
        assert 'under `control_group="strict"`\nit moves the clean-control cutoff' in md

    def test_anticipation_lengthens_the_window_and_can_trim_a_cohort(self, data):
        with pytest.warns(UserWarning, match="Trimmed 1 adoption event"):
            fit = StackedDiD(kappa_pre=3, kappa_post=2, anticipation=1).fit(data, **FIT)
        assert [int(g) for g in fit.trimmed_groups] == [3]
        assert fit.aggregate("event_study").to_dataframe()["event_time"].min() == -4
        md = notebook_markdown(NB)
        assert "to `[-kappa_pre - k, kappa_post]`" in md
        assert "drops the cohort adopting in period 3" in md


class TestChoices:
    def test_weighting(self, data):
        with_population = data.copy()
        with_population["population"] = (
            with_population["first_treat"].map({3: 1000, 5: 2000, 7: 500}).fillna(0).astype(int)
        )
        atts = {}
        for scheme in ("aggregate", "population", "sample_share"):
            extra = {"population": "population"} if scheme == "population" else {}
            atts[scheme] = (
                StackedDiD(kappa_pre=2, kappa_post=2, weighting=scheme)
                .fit(with_population, **FIT, **extra)
                .att
            )
        assert atts["aggregate"] == pytest.approx(1.922, abs=1e-3)
        assert atts["population"] == pytest.approx(1.960, abs=1e-3)
        assert atts["sample_share"] == pytest.approx(1.939, abs=1e-3)
        assert_quotes_in_rendered(
            NB,
            ["population       1.960   0.058", "sample_share     1.939   0.051"],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert "the treated observations are reweighted as well as the\ncontrols" in md
        assert "Treated observations always have" not in md
        # the population column is per unit: cohort mass = units x per-unit value
        treated = with_population.drop_duplicates("unit")
        treated = treated[treated["first_treat"] > 0]
        mass = treated.groupby("first_treat")["population"].sum()
        assert mass.to_dict() == {3: 60_000, 5: 150_000, 7: 37_500}
        np.testing.assert_allclose((mass / mass.sum()).to_numpy(), [0.24, 0.61, 0.15], atol=0.006)
        counts = treated["first_treat"].value_counts().sort_index()
        np.testing.assert_allclose(
            (counts / counts.sum()).to_numpy(), [0.29, 0.36, 0.36], atol=0.006
        )
        assert (
            "they enter with weights of 24%,\n61% and 15%, against 29%, 36% and 36% under the default"
            in md
        )
        code = "\n".join(
            "".join(c["source"])
            for c in json.loads(
                (Path(__file__).resolve().parents[1] / NB).read_text(encoding="utf-8")
            )["cells"]
            if c["cell_type"] == "code"
        )
        assert "an external size for each cohort" not in code
        # clean controls under anticipation
        assert (
            "With `anticipation=k`, both cutoffs for later adopters move out by `k` periods" in md
        )

    def test_control_definitions(self, data):
        fits = {
            d: StackedDiD(kappa_pre=2, kappa_post=2, control_group=d).fit(data, **FIT)
            for d in ("not_yet_treated", "strict", "never_treated")
        }
        assert fits["not_yet_treated"].n_control_units == 165
        assert fits["strict"].n_control_units == fits["never_treated"].n_control_units == 90
        assert fits["strict"].att == pytest.approx(fits["never_treated"].att, rel=1e-9)
        assert fits["strict"].att == pytest.approx(1.937, abs=1e-3)
        for fit in fits.values():
            assert fit.conf_int[0] <= TRUE_ATT <= fit.conf_int[1]
            assert fit.se == pytest.approx(0.051, abs=1e-3)

    def test_other_estimators(self, results, data):
        cs = CallawaySantAnna().fit(data, **FIT)
        imputation = ImputationDiD().fit(data, **FIT)
        for fit in (results, cs, imputation):
            assert fit.conf_int[0] <= TRUE_ATT <= fit.conf_int[1]
        assert abs(results.se - cs.se) < 0.005
        assert imputation.se < 0.8 * results.se

    def test_anticipation(self, results, data):
        fit = StackedDiD(kappa_pre=2, kappa_post=2, anticipation=1).fit(data, **FIT)
        assert fit.att == pytest.approx(1.544, abs=1e-3)
        es = fit.aggregate("event_study").to_dataframe().set_index("event_time")
        assert bool(es.loc[-2, "is_reference"])
        # the overall ATT is the mean over event times -1..2, the first of them a non-effect
        assert es.loc[-1:2, "att"].mean() == pytest.approx(fit.att, abs=1e-6)
        assert es.loc[-1, "att"] == pytest.approx(0.103, abs=1e-3)

    def test_clustering(self, data):
        unit = StackedDiD(kappa_pre=2, kappa_post=2, cluster="unit").fit(data, **FIT)
        subexp = StackedDiD(kappa_pre=2, kappa_post=2, cluster="unit_subexp").fit(data, **FIT)
        assert unit.att == pytest.approx(subexp.att, rel=1e-12)
        assert unit.se == pytest.approx(0.0509, abs=2e-4)
        assert subexp.se == pytest.approx(0.0539, abs=2e-4)
        assert unit.se < subexp.se  # not "narrower" for unit_subexp


class TestNotebookSurface:
    def test_prose(self):
        md = notebook_markdown(NB)
        for needle in (
            "The *trimmed aggregate ATT*",
            "a longer `kappa_post` is a different target, not a better estimate",
            "0.29 / 0.48 = 0.60",
            '`"strict"` and `"never_treated"` give the same result here.',
            "(0.10 + 2.05 + 2.04 + 1.99) / 4 = 1.54",
            "here the unit-level one is the smaller of the two",
            "None is uniformly the safer choice",
        ):
            assert needle in md, needle
        for stale in ("strongest identification", "narrower SEs", "homogeneous"):
            assert stale not in md, stale

    def test_code(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "TRUE_ATT = 2.0",
            "dynamic_effects=False,",
            "StackedDiD(kappa_pre=2, kappa_post=2)",
            'es = results.aggregate("event_study")',
            "plot_event_study(es,",
        ):
            assert needle in code, needle
