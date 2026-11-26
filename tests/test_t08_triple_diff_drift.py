"""Drift detection for Tutorial 08 (``docs/tutorials/08_triple_diff.ipynb``).

The tutorial's narrative rests on these claims:

1. **The triple difference recovers the truth where each plain DiD does not.**
   ``generate_ddd_data`` builds in a group trend (+1.0) and an eligibility
   trend (+0.5). The DiD on eligible units across groups (expected 3.0) and the
   DiD within treated groups across eligibility (expected 2.5) both have
   intervals that exclude the true ATT of 2.0; the triple difference (about
   1.95) covers it.
2. **Without covariates the estimate is the cell-mean DDD**, and the three
   estimation methods coincide; with covariates they differ and all cover.
3. **Staggered mode** (true effect 3.0 in every treated period): the overall
   ATT and every post-enabling event-study interval cover the truth, the fit
   warns about the two period-1 cells it cannot compute, and the two overall
   summaries (cohort-size-weighted cells vs the plain average of event-study
   estimates) are close.

The staggered example uses ``seed=7``. Across seeds the interval covers the
truth at its nominal rate (60 seeds: mean estimate 3.02, coverage 0.95), but
individual draws miss, as the notebook's previous seed did by 2.5 standard
errors; the tutorial's seed is a typical draw, not a guarantee.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from diff_diff import (
    DifferenceInDifferences,
    TripleDifference,
    generate_ddd_data,
    generate_staggered_ddd_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/08_triple_diff.ipynb"

TRUE_ATT = 2.0
TRUE_ATT_STAGGERED = 3.0
FIT = dict(outcome="outcome", group="group", partition="partition", post="post")
STAGGERED_FIT = dict(
    outcome="outcome",
    partition="eligibility",
    unit="unit",
    time="period",
    first_treat="first_treat",
)


@pytest.fixture(scope="module")
def data():
    return generate_ddd_data(
        n_per_cell=2000,
        treatment_effect=TRUE_ATT,
        group_effect=5.0,
        partition_effect=3.0,
        time_effect=2.0,
        noise_sd=3.0,
        add_covariates=True,
        seed=42,
    ).rename(columns={"time": "post"})


@pytest.fixture(scope="module")
def results(data):
    return TripleDifference(estimation_method="dr").fit(data, **FIT)


@pytest.fixture(scope="module")
def staggered_fit():
    panel = generate_staggered_ddd_data(
        n_units=120,
        n_periods=6,
        cohort_periods=[3, 4],
        treatment_effect=TRUE_ATT_STAGGERED,
        dynamic_effects=False,
        seed=7,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = TripleDifference(estimation_method="dr", control_group="not_yet_treated").fit(
            panel, **STAGGERED_FIT
        )
    return panel, res, [str(w.message) for w in caught]


class TestTwoByTwoByTwo:
    def test_ddd_covers_truth(self, results):
        assert results.att == pytest.approx(1.9530, abs=5e-4)
        assert results.se == pytest.approx(0.2101, abs=5e-4)
        lower, upper = results.conf_int
        assert (lower, upper) == pytest.approx((1.5413, 2.3648), abs=5e-4)
        assert lower <= TRUE_ATT <= upper
        assert_quotes_in_rendered(
            NB,
            ["Estimated ATT: 1.9530  (SE 0.2101)", "CI covers the true ATT: True"],
            surface="output",
        )

    def test_each_plain_did_misses(self, data):
        across = DifferenceInDifferences().fit(
            data[data["partition"] == 1], outcome="outcome", treatment="group", post="post"
        )
        within = DifferenceInDifferences().fit(
            data[data["group"] == 1], outcome="outcome", treatment="partition", post="post"
        )
        assert across.att == pytest.approx(2.8867, abs=5e-4)
        assert within.att == pytest.approx(2.4647, abs=5e-4)
        for res, expected in ((across, TRUE_ATT + 1.0), (within, TRUE_ATT + 0.5)):
            lower, upper = res.conf_int
            assert not (lower <= TRUE_ATT <= upper)  # misses the truth ...
            assert lower <= expected <= upper  # ... and covers its biased expectation
        assert_quotes_in_rendered(
            NB,
            [
                "DiD: eligible units, treated vs control groups    2.8867    2.5938    3.1796",
                "DiD: treated groups, eligible vs ineligible       2.4647    2.1711    2.7583",
            ],
            surface="output",
        )
        assert_quotes_in_rendered(
            NB, ["(expected value 2.0 + 1.0)", "(expected value 2.0 + 0.5)"], surface="markdown"
        )

    def test_generator_builds_in_both_trends(self):
        """The prose states the two trend differences; pin them on a noise-free draw."""
        clean = generate_ddd_data(n_per_cell=50, treatment_effect=TRUE_ATT, noise_sd=0.0, seed=1)
        means = clean.groupby(["group", "partition", "time"])["outcome"].mean().unstack("time")
        change = means[1] - means[0]
        assert change[1, 0] - change[0, 0] == pytest.approx(1.0)  # group trend
        assert change[0, 1] - change[0, 0] == pytest.approx(0.5)  # eligibility trend
        ddd = (change[1, 1] - change[1, 0]) - (change[0, 1] - change[0, 0])
        assert ddd == pytest.approx(TRUE_ATT)

    def test_estimate_is_the_cell_mean_ddd(self, data, results):
        means = data.groupby(["group", "partition", "post"])["outcome"].mean().unstack("post")
        change = means[1] - means[0]
        manual = (change[1, 1] - change[1, 0]) - (change[0, 1] - change[0, 0])
        np.testing.assert_allclose(results.att, manual, rtol=1e-10)

    def test_methods_coincide_without_covariates(self, data):
        plain = {
            m: TripleDifference(estimation_method=m).fit(data, **FIT) for m in ("reg", "ipw", "dr")
        }
        for m in ("ipw", "dr"):
            np.testing.assert_allclose(plain[m].att, plain["reg"].att, rtol=1e-8)
            np.testing.assert_allclose(plain[m].se, plain["reg"].se, rtol=1e-6)
        adjusted = {
            m: TripleDifference(estimation_method=m).fit(
                data, covariates=["age", "education"], **FIT
            )
            for m in ("reg", "ipw", "dr")
        }
        atts = [adjusted[m].att for m in ("reg", "ipw", "dr")]
        np.testing.assert_allclose(atts, [1.9190, 1.9520, 1.9659], atol=5e-4)
        assert len({round(a, 3) for a in atts}) == 3  # "the three estimates differ slightly"
        for res in adjusted.values():
            assert res.conf_int[0] <= TRUE_ATT <= res.conf_int[1]


class TestStaggered:
    def test_overall_att_covers_truth(self, staggered_fit):
        _, res, _ = staggered_fit
        assert res.att == pytest.approx(2.9879, abs=5e-4)
        lower, upper = res.conf_int
        assert (lower, upper) == pytest.approx((2.6257, 3.3501), abs=5e-4)
        assert lower <= TRUE_ATT_STAGGERED <= upper
        assert_quotes_in_rendered(
            NB,
            ["Estimated ATT: 2.9879  (SE 0.1848)", "95% CI:        [2.6257, 3.3501]"],
            surface="output",
        )

    def test_warns_about_the_first_period_cells(self, staggered_fit):
        _, _, messages = staggered_fit
        skipped = [m for m in messages if "is outside the observed panel" in m]
        assert len(skipped) == 2  # "skipped two cells"
        assert all("t=1" in m for m in skipped)

    def test_event_study_and_cohorts_cover_truth(self, staggered_fit):
        _, res, _ = staggered_fit
        es = res.aggregate("event_study").to_dataframe()
        assert es["event_time"].tolist() == [-2, -1, 0, 1, 2, 3]
        post = es[es["event_time"] >= 0]
        covers = (post["conf_int_lower"] <= TRUE_ATT_STAGGERED) & (
            TRUE_ATT_STAGGERED <= post["conf_int_upper"]
        )
        assert covers.all()
        pre = es[es["event_time"] < 0]
        assert ((pre["conf_int_lower"] < 0) & (pre["conf_int_upper"] > 0)).all()
        by_cohort = res.aggregate("group").to_dataframe()
        np.testing.assert_allclose(by_cohort["att"], [3.0626, 2.8883], atol=5e-4)

    def test_two_overall_summaries(self, staggered_fit):
        _, res, _ = staggered_fit
        by_event_time = res.aggregate("simple", weights="event_time").to_dataframe()["att"].iloc[0]
        assert by_event_time == pytest.approx(2.9651, abs=5e-4)
        # the plain average of the post-enabling event-study estimates
        es = res.aggregate("event_study").to_dataframe()
        np.testing.assert_allclose(
            by_event_time, es.loc[es["event_time"] >= 0, "att"].mean(), rtol=1e-8
        )
        assert abs(by_event_time - res.att) < 0.1  # "they are close"
        assert_quotes_in_rendered(
            NB,
            [
                "Cohort-size-weighted average of ATT(g,t): 2.9879",
                "Average of the event-study estimates:      2.9651",
            ],
            surface="output",
        )

    def test_cluster_is_rejected_in_staggered_mode(self, staggered_fit):
        """The prose says ``cluster=`` is not accepted in this mode."""
        panel, _, _ = staggered_fit
        with pytest.raises(ValueError, match="n_bootstrap"):
            TripleDifference(cluster="unit").fit(panel, **STAGGERED_FIT)


class TestNotebookSurface:
    def test_states_the_assumption(self):
        md = notebook_markdown(NB)
        for needle in (
            "the DDD version of **parallel trends**",
            "**no anticipation**",
            "not a\npass/fail test",
        ):
            assert needle in md, needle

    def test_notebook_constants_match(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "TRUE_ATT = 2.0",
            "n_per_cell=2000,",
            "TRUE_ATT_STAGGERED = 3.0",
            "dynamic_effects=False,",
            "seed=7,",
            'control_group="not_yet_treated"',
        ):
            assert needle in source, f"notebook code no longer contains {needle!r}"
