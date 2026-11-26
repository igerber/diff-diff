"""Drift detection for Tutorial 05 (``docs/tutorials/05_honest_did.ipynb``).

The tutorial fits an event study on a panel whose treated group drifts by an
extra 0.2 per period (true effect 5.0), then relaxes parallel trends. Its
claims:

1. **The conventional interval misses the truth.** The average post-treatment
   coefficient is 5.54 (built-in bias 0.6); the interval under exact parallel
   trends, [5.39, 5.70], excludes 5.0.
2. **Relative magnitudes bounds first differences, not levels.** At Mbar = 1 the
   identified set is the estimate +/- (largest pre-treatment step) x 3, where 3
   is the average number of steps over five treated periods; the robust interval
   widens that by 1.96 SE and contains 5.0.
3. **Breakdown value.** About 5.7 for relative magnitudes; a grid that stops
   short of it reports ``breakdown_M = None``.
4. **Smoothness at M = 0** extrapolates the linear pre-trend: [4.82, 5.29]
   contains 5.0 with an empty (NaN) estimated identified set; breakdown about
   0.58.
5. **Target parameter.** ``l_vec`` on the first treated period gives a tighter
   relative-magnitudes set (one step, not an average of 1..5).
6. **Staggered route.** ``compute_honest_did`` on a universal-base
   CallawaySantAnna event study contains the true 2.0.

Everything is deterministic given the seeds. The smoothness interval comes from
a numerical optimizer, so it is pinned more loosely than the closed-form
relative-magnitudes numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diff_diff import (
    CallawaySantAnna,
    HonestDiD,
    TwoWayFixedEffects,
    compute_honest_did,
    generate_panel_data,
    generate_staggered_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/05_honest_did.ipynb"

TRUE_ATT = 5.0
TREND_VIOLATION = 0.2
STAGGERED_TRUE_ATT = 2.0


@pytest.fixture(scope="module")
def event_study():
    data = generate_panel_data(
        n_units=200,
        n_periods=10,
        treatment_period=5,
        treatment_fraction=0.5,
        treatment_effect=TRUE_ATT,
        parallel_trends=False,
        trend_violation=TREND_VIOLATION,
        unit_fe_sd=2.0,
        noise_sd=0.5,
        seed=42,
    )
    return TwoWayFixedEffects().fit(
        data,
        outcome="outcome",
        treatment="treated",
        unit="unit",
        time="period",
        event_study=True,
        post_periods=[5, 6, 7, 8, 9],
    )


@pytest.fixture(scope="module")
def es_table(event_study):
    return event_study.to_dataframe()


class TestConventionalEventStudy:
    def test_coefficients_track_the_built_in_drift(self, es_table):
        att = es_table.set_index("event_time")["att"]
        np.testing.assert_allclose(
            att.loc[[0, 1, 2, 3]].to_numpy(), [-0.7416, -0.6596, -0.3569, -0.3152], atol=5e-4
        )
        assert att.loc[4] == 0.0  # the reference period
        expected = TREND_VIOLATION * (np.arange(10) - 4) + np.where(np.arange(10) >= 5, TRUE_ATT, 0)
        # every coefficient is within sampling noise of effect + violation
        assert np.abs(att.to_numpy() - expected).max() < 0.25
        assert att.loc[[5, 6, 7, 8, 9]].mean() == pytest.approx(5.5433, abs=5e-4)
        assert_quotes_in_rendered(
            NB,
            [
                "Average post-treatment coefficient: 5.543",
                "Average true violation after treatment: 0.600",
            ],
            surface="output",
        )

    def test_exact_parallel_trends_interval_misses_the_truth(self, event_study):
        conventional = HonestDiD(method="relative_magnitude", M=0.0).fit(event_study)
        assert conventional.lb == pytest.approx(conventional.ub)
        assert conventional.original_estimate == pytest.approx(5.5433, abs=5e-4)
        assert conventional.ci_lb == pytest.approx(5.3906, abs=5e-4)
        assert conventional.ci_ub == pytest.approx(5.6959, abs=5e-4)
        assert not (conventional.ci_lb <= TRUE_ATT <= conventional.ci_ub)
        assert_quotes_in_rendered(
            NB,
            ["95% CI [ 5.391,  5.696]   covers 5.0: False"],
            surface="output",
        )
        assert "[5.39, 5.70] and **misses the true effect of" in notebook_markdown(NB)


class TestRelativeMagnitudes:
    def test_identified_set_is_built_from_first_differences(self, event_study, es_table):
        honest = HonestDiD(method="relative_magnitude", M=1.0).fit(event_study)
        pre = es_table.loc[es_table["event_time"] <= 4, "att"].to_numpy()
        max_pre_step = np.abs(np.diff(pre)).max()
        assert max_pre_step == pytest.approx(0.3152, abs=5e-4)
        half_width = max_pre_step * np.mean([1, 2, 3, 4, 5])
        assert honest.lb == pytest.approx(honest.original_estimate - half_width, abs=1e-6)
        assert honest.ub == pytest.approx(honest.original_estimate + half_width, abs=1e-6)
        # a bound on LEVELS (max |pre coefficient|) would give a different set
        level_half_width = np.abs(pre).max()
        assert abs(level_half_width - half_width) > 0.1
        assert honest.lb == pytest.approx(4.5978, abs=5e-4)
        assert honest.ub == pytest.approx(6.4888, abs=5e-4)
        assert_quotes_in_rendered(
            NB,
            [
                "Largest pre-treatment step:     0.3152",
                "By hand:  [4.5978, 6.4888]",
                "HonestDiD: [4.5978, 6.4888]",
            ],
            surface="output",
        )

    def test_robust_interval_covers_the_truth(self, event_study):
        honest = HonestDiD(method="relative_magnitude", M=1.0).fit(event_study)
        conventional = HonestDiD(method="relative_magnitude", M=0.0).fit(event_study)
        assert honest.ci_lb == pytest.approx(4.4451, abs=5e-4)
        assert honest.ci_ub == pytest.approx(6.6414, abs=5e-4)
        assert honest.ci_lb <= TRUE_ATT <= honest.ci_ub
        # "the identified set widened by 1.96 standard errors on each side"
        assert honest.lb - honest.ci_lb == pytest.approx(1.96 * honest.original_se, abs=1e-3)
        assert honest.ci_ub - honest.ub == pytest.approx(1.96 * honest.original_se, abs=1e-3)
        # "about seven times as wide"
        assert 6.5 < honest.ci_width / conventional.ci_width < 7.5
        assert_quotes_in_rendered(
            NB,
            ["95% CI [ 4.445,  6.641]   covers 5.0: True"],
            surface="output",
        )

    def test_breakdown_value(self, event_study):
        rm = HonestDiD(method="relative_magnitude")
        full = rm.sensitivity_analysis(event_study, M_grid=[0, 0.5, 1, 2, 3, 4, 5, 6, 7])
        short = rm.sensitivity_analysis(event_study, M_grid=[0, 1, 2, 3])
        assert short.breakdown_M is None  # not reached on the short grid
        assert full.breakdown_M == pytest.approx(5.70, abs=0.02)
        assert rm.breakdown_value(event_study, tol=0.01) == pytest.approx(5.70, abs=0.02)
        table = full.to_dataframe().set_index("M")
        assert bool(table.loc[5.0, "is_significant"]) and not bool(table.loc[6.0, "is_significant"])
        assert_quotes_in_rendered(
            NB,
            [
                "Grid up to 3: breakdown_M = None",
                "Grid up to 7: breakdown_M = 5.701",
                "Direct search: 5.698",
            ],
            surface="output",
        )


class TestSmoothness:
    def test_linear_extrapolation_covers_the_truth(self, event_study):
        honest = HonestDiD(method="smoothness", M=0.0).fit(event_study)
        assert honest.ci_lb == pytest.approx(4.824, abs=5e-3)
        assert honest.ci_ub == pytest.approx(5.289, abs=5e-3)
        assert honest.ci_lb <= TRUE_ATT <= honest.ci_ub
        # the estimated identified set is empty: noisy pre-coefficients are not exactly linear
        assert np.isnan(honest.lb) and np.isnan(honest.ub)
        rm = HonestDiD(method="relative_magnitude", M=1.0).fit(event_study)
        assert honest.ci_width < rm.ci_width / 4  # "much narrower"
        assert_quotes_in_rendered(
            NB,
            ["Smoothness (M = 0)                 set [   nan,    nan]   95% CI [ 4.824,  5.289]"],
            surface="output",
        )

    def test_sensitivity(self, event_study):
        sd = HonestDiD(method="smoothness")
        sens = sd.sensitivity_analysis(event_study, M_grid=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75])
        table = sens.to_dataframe().set_index("M")
        assert table.loc[0.1, "ci_lb"] == pytest.approx(3.723, abs=5e-3)
        assert table.loc[0.1, "ci_ub"] == pytest.approx(6.086, abs=5e-3)
        # empty estimated set below 0.3, non-empty from there
        assert table.loc[[0.0, 0.05, 0.1, 0.2], "lb"].isna().all()
        assert table.loc[[0.3, 0.5, 0.75], "lb"].notna().all()
        assert sens.breakdown_M == pytest.approx(0.58, abs=0.01)
        assert_quotes_in_rendered(NB, ["Breakdown value: M = 0.577"], surface="output")


class TestTargetAndStaggered:
    def test_first_period_target_is_tighter(self, event_study):
        first = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        avg = HonestDiD(method="relative_magnitude", M=1.0).fit(event_study)
        rm_first = HonestDiD(method="relative_magnitude", M=1.0, l_vec=first).fit(event_study)
        sd_first = HonestDiD(method="smoothness", M=0.0, l_vec=first).fit(event_study)
        assert rm_first.original_estimate == pytest.approx(5.2111, abs=5e-4)
        # one step after the reference: half-width is one pre-treatment step, not three.
        # This is the boundary condition of the restriction: the step from the reference
        # period into the first treated period is itself bounded.
        assert (rm_first.ub - rm_first.lb) / 2 == pytest.approx(0.3152, abs=5e-4)
        assert (rm_first.ub - rm_first.lb) < (avg.ub - avg.lb) / 2.5
        assert rm_first.ci_lb <= TRUE_ATT <= rm_first.ci_ub
        assert sd_first.ci_lb <= TRUE_ATT <= sd_first.ci_ub
        assert_quotes_in_rendered(
            NB,
            ["Conventional estimate, first treated period: 5.211"],
            surface="output",
        )

    def test_staggered_route(self):
        staggered = generate_staggered_data(
            n_units=300,
            n_periods=10,
            cohort_periods=[4, 6],
            never_treated_frac=0.4,
            treatment_effect=STAGGERED_TRUE_ATT,
            dynamic_effects=False,
            noise_sd=0.5,
            seed=42,
        )
        cs = CallawaySantAnna(base_period="universal").fit(
            staggered, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        honest = compute_honest_did(cs.aggregate("event_study"), method="relative_magnitude", M=1.0)
        assert honest.original_estimate == pytest.approx(1.999, abs=2e-3)
        assert honest.lb == pytest.approx(1.730, abs=2e-3)
        assert honest.ub == pytest.approx(2.268, abs=2e-3)
        assert honest.ci_lb == pytest.approx(1.619, abs=2e-3)
        assert honest.ci_ub == pytest.approx(2.379, abs=2e-3)
        assert honest.ci_lb <= STAGGERED_TRUE_ATT <= honest.ci_ub
        assert_quotes_in_rendered(
            NB,
            [
                "Identified set (Mbar = 1): [1.730, 2.268]",
                "Robust 95% CI:             [1.619, 2.379]",
            ],
            surface="output",
        )


class TestNotebookSurface:
    def test_prose_states_the_restrictions_correctly(self):
        md = notebook_markdown(NB)
        for needle in (
            # relative magnitudes is a bound on first differences
            r"|\delta_{t+1} - \delta_t| \le \bar{M} \times \max_{s<0} |\delta_{s+1} - \delta_s|",
            "The bound is on **changes** in the violation, not on its level",
            # ... starting with the step out of the reference period (t = 0)
            r"\quad \text{for } t = 0, 1, 2, \ldots",
            "is the one from the reference period into the first",
            # smoothness at M = 0 is linear extrapolation, not parallel trends
            "It is *not* parallel trends.",
            # the two scales are not comparable
            "have nothing in common",
            # the breakdown value is about the sign
            "That is a robust *sign*.",
            '"not reached on this grid", not "no breakdown exists"',
            "least 95% of the time *provided the violation is in*",
        ):
            assert needle in md, needle

    def test_code_passes_the_truth_determining_arguments(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "TRUE_ATT = 5.0",
            "TREND_VIOLATION = 0.2",
            "treatment_effect=TRUE_ATT,",
            "trend_violation=TREND_VIOLATION,",
            "dynamic_effects=False,",
            'CallawaySantAnna(base_period="universal")',
            "seed=42",
        ):
            assert needle in code, needle
