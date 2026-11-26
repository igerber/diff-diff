"""Drift detection for Tutorial 06 (``docs/tutorials/06_power_analysis.ipynb``).

The tutorial covers closed-form and simulated power. Its claims:

1. **Closed form.** MDE 7.92 for 50+50 units at SD 10; 252 units to detect an
   effect of 5; 41% power for an effect of 4 with 75 per group. More periods
   and a higher within-unit correlation both lower the MDE.
2. **sigma and rho are not double counted.** A unit component with SD 2 plus
   noise with SD 5 gives the same MDE described as (noise SD, rho 0) or as
   (total SD, rho 4/29); mixing the two understates the MDE.
3. **Closed form and simulation agree for matching designs.** Power 85.1%
   against a simulated 83.0%; MDE 2.80 against 2.81; 32 units against a
   searched 35, with 79.5% simulated power at 32. The closed-form default of
   one pre and one post period needs 63 units, a different design.
4. **Clustering.** Unclustered standard errors on the simulated panel average
   1.08, the value that counts the unit effect as noise; clustered ones 1.00.

The closed-form numbers are deterministic. Simulation re-derivations run the
same seeds as the notebook and are marked ``slow``; the Synthetic DiD and
triple-difference runs are pinned on the rendered surface only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diff_diff import (
    DifferenceInDifferences,
    PowerAnalysis,
    compute_mde,
    compute_power,
    compute_sample_size,
    simulate_mde,
    simulate_power,
    simulate_sample_size,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/06_power_analysis.ipynb"

SIGMA = 10.0
UNIT_SD = 2.0
NOISE_SD = 5.0
TRUE_EFFECT = 3.0
DESIGN = dict(n_treated=50, n_control=50, n_pre=2, n_post=2)
SIM_DESIGN = dict(
    n_units=100, n_periods=4, treatment_period=2, treatment_fraction=0.5, sigma=NOISE_SD
)


@pytest.fixture(scope="module")
def pa():
    return PowerAnalysis(alpha=0.05, power=0.80)


class TestClosedForm:
    def test_mde(self, pa):
        mdes = {n: pa.mde(n_treated=n, n_control=n, sigma=SIGMA).mde for n in (25, 50, 100, 200)}
        assert mdes[50] == pytest.approx(7.9241, abs=5e-4)
        # "four times the units halves it"
        assert mdes[25] / mdes[100] == pytest.approx(2.0, rel=1e-9)
        assert mdes[50] / mdes[200] == pytest.approx(2.0, rel=1e-9)
        assert "detects effects of 7.9 or more" in notebook_markdown(NB)
        assert_quotes_in_rendered(
            NB, ["          50        100       7.92       0.79"], surface="output"
        )

    def test_sample_size(self, pa):
        sizes = {e: pa.sample_size(effect_size=e, sigma=SIGMA).required_n for e in (5.0, 10.0)}
        assert sizes == {5.0: 252, 10.0: 63}
        # "halving the effect you want to detect quadruples the units needed"
        assert sizes[5.0] / sizes[10.0] == pytest.approx(4.0, abs=0.05)
        by_power = [
            PowerAnalysis(power=p).sample_size(effect_size=5.0, sigma=SIGMA).required_n
            for p in (0.70, 0.80, 0.90, 0.95)
        ]
        assert by_power == [198, 252, 337, 416]

    def test_power(self):
        result = PowerAnalysis(alpha=0.05).power(
            effect_size=4.0, n_treated=75, n_control=75, sigma=SIGMA
        )
        assert result.power == pytest.approx(0.410, abs=1e-3)
        assert result.power < 0.5  # "would miss a true effect of 4.0 more often than find it"
        assert_quotes_in_rendered(
            NB,
            ["Power to detect an effect of 4.0 with 75 units per group: 41.0%"],
            surface="output",
        )

    def test_periods_and_correlation_lower_the_mde(self, pa):
        by_periods = [
            pa.mde(n_treated=50, n_control=50, sigma=SIGMA, n_pre=k, n_post=k).mde
            for k in (1, 2, 3, 5)
        ]
        np.testing.assert_allclose(by_periods, [7.92, 5.60, 4.57, 3.54], atol=5e-3)
        by_rho = [
            pa.mde(n_treated=50, n_control=50, sigma=SIGMA, n_pre=3, n_post=3, rho=r).mde
            for r in (0.0, 0.1, 0.3, 0.5, 0.7)
        ]
        np.testing.assert_allclose(by_rho, [4.57, 4.34, 3.83, 3.23, 2.51], atol=5e-3)
        assert all(a > b for a, b in zip(by_rho, by_rho[1:]))  # higher rho, smaller MDE
        # variance scales with (1 - rho)
        assert (by_rho[3] / by_rho[0]) ** 2 == pytest.approx(0.5, rel=1e-9)

    def test_sigma_and_rho_are_two_descriptions_of_one_design(self, pa):
        total_sd = np.sqrt(UNIT_SD**2 + NOISE_SD**2)
        rho = UNIT_SD**2 / total_sd**2
        by_noise = pa.mde(sigma=NOISE_SD, rho=0.0, **DESIGN).mde
        by_total = pa.mde(sigma=total_sd, rho=rho, **DESIGN).mde
        double_counted = pa.mde(sigma=NOISE_SD, rho=rho, **DESIGN).mde
        assert by_noise == pytest.approx(2.8016, abs=5e-4)
        assert by_total == pytest.approx(by_noise, rel=1e-12)
        assert double_counted == pytest.approx(2.6012, abs=5e-4)
        assert double_counted < by_noise  # the mix makes the design look more powerful
        assert_quotes_in_rendered(
            NB,
            [
                "Within-unit SD 5.0, rho 0:                  MDE = 2.802",
                "Total SD 5.39, rho 0.14:               MDE = 2.802",
                "Within-unit SD 5.0 with rho 0.14 (wrong):  MDE = 2.601",
            ],
            surface="output",
        )

    def test_matching_and_default_designs(self, pa):
        assert pa.sample_size(effect_size=5.0, sigma=NOISE_SD, n_pre=2, n_post=2).required_n == 32
        assert pa.sample_size(effect_size=5.0, sigma=NOISE_SD).required_n == 63
        closed = PowerAnalysis(alpha=0.05).power(effect_size=TRUE_EFFECT, sigma=NOISE_SD, **DESIGN)
        assert closed.power == pytest.approx(0.851, abs=1e-3)
        # standard error implied by the design: 5 * sqrt((1/50 + 1/50) * (1/2 + 1/2)) = 1.0
        assert NOISE_SD * 0.2 == pytest.approx(1.0)
        assert np.sqrt(UNIT_SD**2 + NOISE_SD**2) * 0.2 == pytest.approx(1.077, abs=5e-4)

    def test_shortcuts(self):
        assert compute_mde(n_treated=100, n_control=100, sigma=SIGMA, power=0.80) == pytest.approx(
            5.60, abs=5e-3
        )
        assert compute_power(
            effect_size=5.0, n_treated=100, n_control=100, sigma=SIGMA
        ) == pytest.approx(0.705, abs=1e-3)
        assert compute_sample_size(effect_size=5.0, sigma=SIGMA, power=0.80) == 252


@pytest.mark.slow
class TestSimulation:
    """Re-run the basic-DiD simulations at the notebook's seeds."""

    def test_power_matches_the_closed_form(self):
        sim = simulate_power(
            estimator=DifferenceInDifferences(cluster="unit"),
            treatment_effect=TRUE_EFFECT,
            n_simulations=500,
            seed=42,
            progress=False,
            **SIM_DESIGN,
        )
        closed = PowerAnalysis(alpha=0.05).power(effect_size=TRUE_EFFECT, sigma=NOISE_SD, **DESIGN)
        assert sim.power == pytest.approx(0.830, abs=0.01)
        assert sim.power_ci[0] <= closed.power <= sim.power_ci[1]
        assert sim.bias == pytest.approx(-0.032, abs=5e-3)
        assert sim.coverage == pytest.approx(0.934, abs=0.01)

    def test_unclustered_se_counts_the_unit_effect(self):
        runs = {
            label: simulate_power(
                estimator=est,
                treatment_effect=TRUE_EFFECT,
                n_simulations=500,
                seed=42,
                progress=False,
                **SIM_DESIGN,
            )
            for label, est in (
                ("clustered", DifferenceInDifferences(cluster="unit")),
                ("unclustered", DifferenceInDifferences()),
            )
        }
        # same estimates, different standard errors
        assert runs["clustered"].std_estimate == pytest.approx(
            runs["unclustered"].std_estimate, rel=1e-9
        )
        assert runs["clustered"].mean_se == pytest.approx(0.999, abs=5e-3)
        assert runs["unclustered"].mean_se == pytest.approx(1.077, abs=5e-3)
        assert runs["unclustered"].power < runs["clustered"].power  # conservative here

    def test_mde_search(self, pa):
        mde_sim = simulate_mde(
            DifferenceInDifferences(cluster="unit"),
            n_simulations=300,
            seed=42,
            progress=False,
            **SIM_DESIGN,
        )
        closed = pa.mde(sigma=NOISE_SD, **DESIGN).mde
        assert mde_sim.mde == pytest.approx(2.81, abs=0.07)
        assert abs(mde_sim.mde - closed) < 0.15

    def test_sample_size_search(self, pa):
        n_sim = simulate_sample_size(
            DifferenceInDifferences(cluster="unit"),
            treatment_effect=5.0,
            n_periods=4,
            treatment_period=2,
            treatment_fraction=0.5,
            sigma=NOISE_SD,
            n_simulations=300,
            seed=42,
            progress=False,
        )
        assert 32 <= n_sim.required_n <= 37  # notebook: 35
        at_32 = simulate_power(
            estimator=DifferenceInDifferences(cluster="unit"),
            n_units=32,
            n_periods=4,
            treatment_period=2,
            treatment_fraction=0.5,
            treatment_effect=5.0,
            sigma=NOISE_SD,
            n_simulations=1000,
            seed=42,
            progress=False,
        )
        assert at_32.power == pytest.approx(0.795, abs=0.01)
        assert at_32.power_ci[0] <= 0.80 <= at_32.power_ci[1]


class TestNotebookSurface:
    def test_rendered_simulation_numbers(self):
        assert_quotes_in_rendered(
            NB,
            [
                "Simulated power:       83.0%  (95% CI 79.7% to 86.3%)",
                "Closed-form power:     85.1%",
                "clustered          0.999             1.052    83.0%      93.4%",
                "unclustered        1.077             1.052    80.8%      95.8%",
                "MDE with 100 units                         2.80         2.81",
                "Units needed for an effect of 5              32           35",
                "  (closed form, 1 pre and 1 post)            63",
                "Simulated power with 32 units: 79.5% (95% CI 77.0% to 82.0%)",
                "Callaway-Sant'Anna MDE: 3.28",
                "Effective N: None",
            ],
            surface="output",
        )

    def test_prose(self):
        md = notebook_markdown(NB)
        for needle in (
            "**`sigma`** is the standard deviation of the outcome across units within a group and",
            "removes the unit component\ntwice",
            "**`sigma` is the SD of the period-to-period noise**",
            "The MDEs agree: 2.80 and 2.81. The sample sizes are 32 and 35.",
            "a different design, not a disagreement between the methods",
            "it reports 1.08 where the design implies 1.00",
            "a large-sample normal approximation",
            "it says nothing about parallel\ntrends",
        ):
            assert needle in md, needle

    def test_code(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "TRUE_EFFECT = 3.0",
            'DifferenceInDifferences(cluster="unit")',
            'TripleDifference(cluster="unit")',
            "SyntheticDiD(seed=42)",
            'message="Pre-treatment fit is poor", category=UserWarning',
            "n_pre=2, n_post=2",
        ):
            assert needle in code, needle
