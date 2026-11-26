"""Drift detection for Tutorial 04 (``docs/tutorials/04_parallel_trends.ipynb``).

The tutorial runs every diagnostic on two panels that differ only in a built-in
trend difference of 1.0 per period for the treated group. Its claims:

1. **Slope and distribution tests** separate the two panels: slope difference
   about 0.007 (SE 0.19) vs about 1.0.
2. **Equivalence testing depends on the margin.** On the parallel panel the
   TOST establishes equivalence at margins 0.5 and 0.25 per period but not at
   0.1; the margin translates into a bias bound of margin x 4 periods.
3. **Placebos.** The timing placebo is near zero on the parallel panel and
   about 2 on the violated one; the group placebo needs ``treatment=`` so the
   really-treated units are dropped (without it they sit in the placebo
   control group and the "placebo" is large on valid data).
4. **Permutation and leave-one-out are about the effect**: ``original_effect``
   is the estimate (4.90); ``placebo_effect`` of a permutation result is the
   mean of the null distribution, not the estimate.
5. **A violation biases DiD by trend x distance**: 8.90 vs a true 5.0, an error
   of 3.9 against a predicted 4.0, with an interval that excludes the truth.

Everything here is deterministic given the seeds except the permutation draws,
which are numpy-generated on both backends.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diff_diff import (
    DifferenceInDifferences,
    TwoWayFixedEffects,
    check_parallel_trends,
    check_parallel_trends_robust,
    equivalence_test_trends,
    generate_panel_data,
    leave_one_out_test,
    permutation_test,
    placebo_group_test,
    placebo_timing_test,
    run_all_placebo_tests,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/04_parallel_trends.ipynb"

TRUE_ATT = 5.0
TREND_VIOLATION = 1.0
PRE_PERIODS = [0, 1, 2, 3]
POST_PERIODS = [4, 5, 6, 7]
COMMON = dict(
    n_units=100,
    n_periods=8,
    treatment_period=4,
    treatment_fraction=0.5,
    treatment_effect=TRUE_ATT,
    unit_fe_sd=2.0,
    noise_sd=0.5,
    seed=42,
)
TREND = dict(outcome="outcome", time="period", treatment_group="treated", pre_periods=PRE_PERIODS)


@pytest.fixture(scope="module")
def df_parallel():
    return generate_panel_data(parallel_trends=True, **COMMON)


@pytest.fixture(scope="module")
def df_violated():
    return generate_panel_data(parallel_trends=False, trend_violation=TREND_VIOLATION, **COMMON)


class TestTrendTests:
    def test_slopes(self, df_parallel, df_violated):
        ok = check_parallel_trends(df_parallel, **TREND)
        bad = check_parallel_trends(df_violated, **TREND)
        assert ok["trend_difference"] == pytest.approx(0.0073, abs=5e-4)
        assert ok["trend_difference_se"] == pytest.approx(0.1905, abs=5e-4)  # "near 0.19"
        assert ok["p_value"] > 0.9
        assert bad["trend_difference"] == pytest.approx(1.0073, abs=5e-4)
        assert bad["p_value"] < 1e-4
        assert_quotes_in_rendered(
            NB,
            [
                "parallel trends holds            1.0108            1.0035      0.0073  0.1905",
                "parallel trends violated         2.0108            1.0035      1.0073  0.1905",
            ],
            surface="output",
        )

    def test_distribution_test(self, df_parallel, df_violated):
        kwargs = dict(unit="unit", n_permutations=999, seed=42, **TREND)
        ok = check_parallel_trends_robust(df_parallel, **kwargs)
        bad = check_parallel_trends_robust(df_violated, **kwargs)
        assert ok["wasserstein_p_value"] > 0.05 and ok["ks_p_value"] > 0.05
        assert bad["wasserstein_p_value"] < 0.01 and bad["ks_p_value"] < 0.01
        assert bad["mean_difference"] == pytest.approx(0.9961, abs=5e-4)  # "shifted by about one"
        # the caveat in the prose: unit-period changes are pooled (50 units x 3 changes)
        assert len(ok["treated_changes"]) == 150 and ok["n_treated"] == 150

    def test_equivalence_depends_on_the_margin(self, df_parallel):
        periods_apart = np.mean(POST_PERIODS) - np.mean(PRE_PERIODS)
        assert periods_apart == 4.0  # "which is 4 periods here"
        verdicts = {}
        for margin in (0.5, 0.25, 0.1):
            res = equivalence_test_trends(
                df_parallel, unit="unit", equivalence_margin=margin, **TREND
            )
            verdicts[margin] = bool(res["equivalent"])
            assert res["mean_difference"] == pytest.approx(-0.0039, abs=5e-4)
            assert res["se_difference"] == pytest.approx(0.0824, abs=5e-4)
        assert verdicts == {0.5: True, 0.25: True, 0.1: False}
        assert_quotes_in_rendered(
            NB,
            [
                "0.10                 0.4                    -0.0039 0.0824        0.1222             False"
            ],
            surface="output",
        )
        assert_quotes_in_rendered(
            NB,
            ["equivalence at margins of\n0.5 and 0.25 per period, but not at 0.1"],
            surface="markdown",
        )


class TestPlacebos:
    def test_timing_placebo(self, df_parallel, df_violated):
        kwargs = dict(
            outcome="outcome",
            treatment="treated",
            time="period",
            fake_treatment_period=2,
            post_periods=POST_PERIODS,
            cluster="unit",
        )
        ok = placebo_timing_test(df_parallel, **kwargs)
        bad = placebo_timing_test(df_violated, **kwargs)
        assert ok.placebo_effect == pytest.approx(0.0483, abs=5e-4)
        assert ok.conf_int[0] < 0 < ok.conf_int[1]
        assert bad.placebo_effect == pytest.approx(2.0483, abs=5e-4)  # "it is about 2"
        assert bad.conf_int[0] > 1.5

    def test_group_placebo_needs_the_treatment_column(self, df_parallel):
        comparison_units = df_parallel.loc[df_parallel["treated"] == 0, "unit"].unique()
        rng = np.random.default_rng(7)
        fake = rng.choice(comparison_units, size=len(comparison_units) // 2, replace=False).tolist()
        kwargs = dict(
            outcome="outcome",
            time="period",
            unit="unit",
            fake_treated_units=fake,
            post_periods=POST_PERIODS,
            cluster="unit",
        )
        with_treatment = placebo_group_test(df_parallel, treatment="treated", **kwargs)
        assert with_treatment.placebo_effect == pytest.approx(0.0909, abs=5e-4)
        assert with_treatment.conf_int[0] < 0 < with_treatment.conf_int[1]
        # the mistake the notebook used to make: without treatment=, on the full frame,
        # the really-treated units are in the placebo control group
        without = placebo_group_test(df_parallel, **kwargs)
        assert abs(without.placebo_effect) > 2.0
        assert_quotes_in_rendered(NB, ["Placebo estimate: 0.0909  (SE 0.1002)"], surface="output")

    def test_permutation_reports_the_real_effect_separately(self, df_parallel):
        res = permutation_test(
            df_parallel,
            outcome="outcome",
            treatment="treated",
            post="post",
            unit="unit",
            n_permutations=999,
            seed=42,
        )
        assert res.original_effect == pytest.approx(4.9000, abs=5e-4)
        assert abs(res.placebo_effect) < 0.1  # the mean of the null distribution
        draws = np.asarray(res.permutation_distribution)
        assert len(draws) == 999
        extreme = int(np.sum(np.abs(draws) >= abs(res.original_effect)))
        assert extreme == 0
        assert res.p_value == pytest.approx((1 + extreme) / (999 + 1), rel=1e-9)
        assert_quotes_in_rendered(
            NB,
            [
                "Estimated ATT:                    4.9000",
                "Permutation p-value:              0.0010",
            ],
            surface="output",
        )

    def test_leave_one_out(self, df_parallel):
        loo = leave_one_out_test(
            df_parallel, outcome="outcome", treatment="treated", post="post", unit="unit"
        )
        effects = np.array(list(loo.leave_one_out_effects.values()))
        assert len(effects) == 50
        assert effects.min() == pytest.approx(4.8857, abs=5e-4)
        assert effects.max() == pytest.approx(4.9193, abs=5e-4)
        assert np.abs(effects - loo.original_effect).max() < 0.05  # "a few hundredths"

    def test_run_all(self, df_parallel):
        results = run_all_placebo_tests(
            df_parallel,
            outcome="outcome",
            treatment="treated",
            time="period",
            unit="unit",
            pre_periods=PRE_PERIODS,
            post_periods=POST_PERIODS,
            n_permutations=499,
            seed=42,
            cluster="unit",
        )
        assert list(results) == [
            "fake_timing_1",
            "fake_timing_2",
            "fake_timing_3",
            "permutation",
            "leave_one_out",
        ]
        for name in ("fake_timing_1", "fake_timing_2", "fake_timing_3"):
            assert abs(results[name].placebo_effect) < 0.1
            assert not results[name].is_significant
        for name in ("permutation", "leave_one_out"):
            assert results[name].original_effect == pytest.approx(4.9000, abs=5e-4)
            assert results[name].is_significant  # about the real effect, not a failed placebo


class TestEventStudyAndBias:
    def test_event_study_pre_periods(self, df_parallel, df_violated):
        def pre(df):
            es = TwoWayFixedEffects().fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="period",
                event_study=True,
                post_periods=POST_PERIODS,
                reference_period=3,
            )
            table = es.to_dataframe().set_index("event_time")
            assert bool(table.loc[3, "is_reference"])
            return table

        ok, bad = pre(df_parallel), pre(df_violated)
        assert ok.loc[[0, 1, 2], "att"].abs().max() < 0.1
        np.testing.assert_allclose(bad.loc[[0, 1, 2], "att"], [-2.988, -2.019, -0.910], atol=1e-3)
        np.testing.assert_allclose(np.diff(bad.loc[[0, 1, 2, 3], "att"]), 1.0, atol=0.15)
        # "the post-treatment estimates keep climbing from 6 to 9"
        np.testing.assert_allclose(bad.loc[[4, 7], "att"], [6.071, 8.977], atol=1e-3)
        np.testing.assert_allclose(ok.loc[POST_PERIODS, "att"], TRUE_ATT, atol=0.25)

    def test_did_bias_matches_the_trend(self, df_parallel, df_violated):
        fit = dict(outcome="outcome", treatment="treated", post="post")
        ok = DifferenceInDifferences(cluster="unit").fit(df_parallel, **fit)
        bad = DifferenceInDifferences(cluster="unit").fit(df_violated, **fit)
        assert ok.att == pytest.approx(4.9000, abs=5e-4)
        assert ok.conf_int[0] <= TRUE_ATT <= ok.conf_int[1]
        assert bad.att == pytest.approx(8.9000, abs=5e-4)
        assert not (bad.conf_int[0] <= TRUE_ATT <= bad.conf_int[1])
        assert bad.att - ok.att == pytest.approx(TREND_VIOLATION * 4.0, abs=1e-6)
        assert_quotes_in_rendered(
            NB,
            [
                "Estimated ATT: 8.9000  (SE 0.0681)",
                "Error:         3.9000   (trend difference x periods apart = 4.0)",
            ],
            surface="output",
        )


class TestNotebookSurface:
    def test_no_pass_fail_language(self):
        md = notebook_markdown(NB)
        for needle in (
            "**Not rejecting is not confirming.**",
            "it does not say the slopes are equal",
            "not proof of it",
            "drop periods or units until a test stops rejecting",
        ):
            assert needle in md, needle
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        # the library's verdict-style key and Pass/Fail prints are not shown
        assert "parallel_trends_plausible" not in code
        assert "Pass" not in code
        for needle in (
            "TRUE_ATT = 5.0",
            "TREND_VIOLATION = 1.0",
            'treatment="treated",',
            "seed=42",
        ):
            assert needle in code, needle
