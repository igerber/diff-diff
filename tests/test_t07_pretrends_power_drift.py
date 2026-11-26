"""Drift detection for Tutorial 07 (``docs/tutorials/07_pretrends_power.ipynb``).

The tutorial characterises the pre-trend test of an event study on a panel
where parallel trends holds (true effect 5.0). Its claims:

1. **The pre-test passes** and the conventional interval covers the truth.
2. **The linear MDV is a slope**, 0.208 per period. It implies a gap of 0.83 at
   the earliest pre-period and, continued after treatment, a bias of 0.62 in
   the average effect: 12% of the truth, about 2.5 standard errors.
3. **Power is low where it matters**: 35% against a slope of 0.10, 56.5%
   against 0.15; and the default test rejects 15% of the time at slope zero
   where the Wald form holds 5%.
4. **Shapes have different units.** The largest pre-period gap at the MDV is
   0.65 to 0.9 for every shape.
5. **Conditioning on a pass can make a missed violation worse** (Roth 2022,
   Proposition 2: guaranteed under equal variances, a common positive
   covariance and a monotone trend; ambiguous otherwise). For this event
   study's covariance and a linear violation: at a slope of 0.15 the bias is
   0.45 over all samples and 0.61 among those that pass, with coverage 57%
   against 30%.
6. **The MDV is not a smoothness M.** A linear trend satisfies the smoothness
   restriction at M = 0, whose interval contains the truth.

The default (NIS) power is a multivariate-normal box probability computed by a
randomised integration routine, so power and MDV move at about 1e-6 between
identical calls; nothing is pinned tighter than 1e-3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from diff_diff import (
    CallawaySantAnna,
    HonestDiD,
    PreTrendsPower,
    TwoWayFixedEffects,
    compute_mdv,
    compute_pretrends_power,
    generate_event_study_data,
    generate_staggered_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/07_pretrends_power.ipynb"

TRUE_ATT = 5.0
SLOPE = 0.10
Z = stats.norm.ppf(0.975)


@pytest.fixture(scope="module")
def event_study():
    data = generate_event_study_data(
        n_units=300,
        n_pre=5,
        n_post=5,
        treatment_fraction=0.5,
        treatment_effect=TRUE_ATT,
        unit_fe_sd=2.0,
        noise_sd=2.0,
        seed=7,
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
def power_results(event_study):
    return PreTrendsPower(alpha=0.05, power=0.80, violation_type="linear").fit(event_study, M=SLOPE)


@pytest.fixture(scope="module")
def conventional(event_study):
    return HonestDiD(method="relative_magnitude", M=0.0).fit(event_study)


class TestEventStudy:
    def test_pretest_passes_and_interval_covers_truth(self, event_study, conventional):
        table = event_study.to_dataframe()
        pre = table[table["event_time"] < 4]
        t_stats = (pre["att"] / pre["se"]).to_numpy()
        np.testing.assert_allclose(t_stats, [0.44, -1.35, -0.63, 0.42], atol=5e-3)
        assert not (np.abs(t_stats) > Z).any()
        assert conventional.original_estimate == pytest.approx(5.116, abs=1e-3)
        assert conventional.ci_lb == pytest.approx(4.620, abs=1e-3)
        assert conventional.ci_ub == pytest.approx(5.613, abs=1e-3)
        assert conventional.ci_lb <= TRUE_ATT <= conventional.ci_ub
        assert_quotes_in_rendered(
            NB,
            [
                "Any individually significant at 5%: False",
                "95% CI under parallel trends:    [4.620, 5.613]",
            ],
            surface="output",
        )


class TestPowerAndMDV:
    def test_mdv_is_a_slope_and_what_it_implies(self, power_results, conventional):
        mdv = power_results.mdv
        assert mdv == pytest.approx(0.208, abs=1e-3)
        np.testing.assert_allclose(power_results.violation_weights, [4.0, 3.0, 2.0, 1.0])
        # gap at the earliest pre-period: slope x 4 periods from the reference
        assert power_results.max_abs_pre_violation == pytest.approx(4 * mdv, rel=1e-9)
        assert 4 * mdv == pytest.approx(0.833, abs=2e-3)
        # bias in the average of five post periods: slope x mean(1..5)
        bias = 3 * mdv
        assert bias == pytest.approx(0.624, abs=2e-3)
        assert bias / TRUE_ATT == pytest.approx(0.12, abs=0.01)
        assert bias / conventional.original_se == pytest.approx(2.5, abs=0.1)
        assert_quotes_in_rendered(
            NB,
            [
                "MDV (slope per period):                     0.208",
                "Gap it implies at the earliest pre-period:  0.833",
                "Bias in the average post-treatment effect:  0.624",
                "  as a share of the true effect:            12%",
                "  in standard errors of the estimate:       2.5",
            ],
            surface="output",
        )

    def test_power_at_the_named_slope(self, power_results):
        assert power_results.violation_magnitude == SLOPE
        assert power_results.power == pytest.approx(0.352, abs=2e-3)
        assert not power_results.power_adequate
        assert "the pre-test rejects 35% of the time" in notebook_markdown(NB)
        assert_quotes_in_rendered(
            NB, ["Power to detect this violation:     35.2%"], surface="output"
        )

    def test_power_grid(self, power_results):
        powers = [power_results.power_at(s) for s in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)]
        np.testing.assert_allclose(
            powers, [0.154, 0.205, 0.352, 0.565, 0.772, 0.910, 0.975], atol=2e-3
        )
        assert all(a < b for a, b in zip(powers, powers[1:]))
        # the grid straddles the 80% target instead of saturating
        assert powers[0] < 0.2 and powers[3] < 0.8 < powers[5]

    def test_informativeness_rule(self, power_results):
        max_se = max(power_results.pre_period_ses)
        assert max_se == pytest.approx(0.336, abs=1e-3)
        assert power_results.max_abs_pre_violation > 2 * max_se
        assert power_results.is_informative is False

    def test_test_form(self, event_study):
        nis = PreTrendsPower(violation_type="linear", pretest_form="nis").fit(event_study)
        wald = PreTrendsPower(violation_type="linear", pretest_form="wald").fit(event_study)
        assert nis.mdv == pytest.approx(0.208, abs=1e-3)
        assert wald.mdv == pytest.approx(0.249, abs=1e-3)
        # four separate 5% tests reject about 15% of the time under the null
        assert nis.power_at(0.0) == pytest.approx(0.154, abs=2e-3)
        assert wald.power_at(0.0) == pytest.approx(0.05, abs=1e-6)

    def test_shapes_compared_on_the_largest_gap(self, event_study):
        gaps = {}
        for shape, mdv in (("linear", 0.208), ("constant", 0.652), ("last_period", 0.892)):
            result = PreTrendsPower(violation_type=shape).fit(event_study)
            assert result.mdv == pytest.approx(mdv, abs=2e-3)
            gaps[shape] = result.max_abs_pre_violation
        late = PreTrendsPower(
            violation_type="custom", violation_weights=np.array([0.0, 0.0, 1.0, 1.0])
        ).fit(event_study)
        # custom weights are normalised, so the MDV is the multiplier and the gap is smaller
        assert late.mdv == pytest.approx(1.045, abs=2e-3)
        gaps["custom"] = late.max_abs_pre_violation
        assert gaps["custom"] == pytest.approx(0.739, abs=2e-3)
        assert all(0.6 < g < 0.95 for g in gaps.values()), gaps

    def test_shortcuts(self, event_study):
        assert compute_mdv(
            event_study, target_power=0.80, violation_type="linear"
        ) == pytest.approx(0.208, abs=1e-3)
        assert compute_pretrends_power(event_study, M=0.15).power == pytest.approx(0.565, abs=2e-3)


class TestConditionalOnPassing:
    def test_bias_and_coverage_after_a_pass(self, event_study):
        vcov = np.asarray(event_study.vcov)
        distance = np.asarray(event_study.vcov_index) - 4
        is_post = distance > 0
        pre_se = np.sqrt(np.diag(vcov))[~is_post]
        weights = np.where(is_post, 1.0 / is_post.sum(), 0.0)
        estimate_se = float(np.sqrt(weights @ vcov @ weights))
        rng = np.random.default_rng(0)
        out = {}
        for slope in (0.0, 0.10, 0.15, 0.20):
            centre = slope * distance + np.where(is_post, TRUE_ATT, 0.0)
            draws = rng.multivariate_normal(centre, vcov, size=200_000)
            passed = (np.abs(draws[:, ~is_post]) <= Z * pre_se).all(axis=1)
            estimate = draws @ weights
            covered = np.abs(estimate - TRUE_ATT) <= Z * estimate_se
            out[slope] = (
                passed.mean(),
                estimate.mean() - TRUE_ATT,
                estimate[passed].mean() - TRUE_ATT,
                covered.mean(),
                covered[passed].mean(),
            )
        # no violation: conditioning is harmless
        assert abs(out[0.0][2]) < 0.01 and out[0.0][4] > 0.95
        # the pass rate is one minus the pre-test's power
        assert out[0.15][0] == pytest.approx(0.435, abs=0.01)
        # unconditional bias is slope x 3; conditional on passing it is larger
        assert out[0.15][1] == pytest.approx(0.45, abs=0.01)
        assert out[0.15][2] == pytest.approx(0.605, abs=0.01)
        assert out[0.15][3] == pytest.approx(0.571, abs=0.01)
        assert out[0.15][4] == pytest.approx(0.295, abs=0.01)
        for slope in (0.10, 0.15, 0.20):
            assert out[slope][2] > out[slope][1] and out[slope][4] < out[slope][3]
        assert out[0.20][4] == pytest.approx(0.036, abs=0.01)
        assert_quotes_in_rendered(
            NB,
            ["  0.15      0.435              0.451         0.605          0.571             0.295"],
            surface="output",
        )


class TestHonestDiDLinkAndStaggered:
    def test_linear_trend_is_allowed_at_smoothness_zero(self, event_study, conventional):
        honest = HonestDiD(method="smoothness", M=0.0).fit(event_study)
        assert honest.ci_lb == pytest.approx(4.297, abs=5e-3)
        assert honest.ci_ub == pytest.approx(5.819, abs=5e-3)
        assert honest.ci_lb <= TRUE_ATT <= honest.ci_ub
        # "about half as wide again"
        assert honest.ci_width / conventional.ci_width == pytest.approx(1.5, abs=0.1)
        link = PreTrendsPower().sensitivity_to_honest_did(event_study)
        assert link["max_abs_pre_violation"] == pytest.approx(0.833, abs=2e-3)
        assert link["mdv_in_ses"] == pytest.approx(2.48, abs=0.02)
        assert_quotes_in_rendered(
            NB,
            [
                "Max level deviation / max(SE): 2.48",
                "Allowing any linear trend (M = 0): [4.297, 5.819]",
            ],
            surface="output",
        )

    def test_staggered_route(self):
        staggered = generate_staggered_data(
            n_units=300,
            n_periods=10,
            cohort_periods=[4, 6],
            never_treated_frac=0.4,
            treatment_effect=2.0,
            dynamic_effects=False,
            noise_sd=0.5,
            seed=42,
        )
        cs = CallawaySantAnna(base_period="universal").fit(
            staggered, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        result = PreTrendsPower().fit(cs.aggregate("event_study"))
        assert result.n_pre_periods == 5
        assert result.mdv == pytest.approx(0.044, abs=1e-3)


class TestNotebookSurface:
    def test_prose(self):
        md = notebook_markdown(NB)
        for needle in (
            "so the MDV is a **slope**",
            "reject parallel trends if *any*",
            "The MDV is therefore not a value for $M$",
            "the samples that **pass**, it is 0.61, a third larger",
            "a passed test cannot tell you which row you are in",
            # the direction of the post-pass bias is conditional, and said to be
            "## Passing the pre-test can make a missed violation worse",
            "With other covariance structures the bias\nafter a pass can be larger or smaller.",
            "an example of what can happen, not a law",
            # coverage: reported as a result of this simulation, with no general guarantee
            "there is no general rule\nfor its direction",
            "The effect on coverage has no general guarantee.",
            "not a better pre-test but a sensitivity analysis",
        ):
            assert needle in md, needle

    def test_code(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "TRUE_ATT = 5.0",
            "treatment_effect=TRUE_ATT,",
            "seed=7,",
            "pretrends.fit(event_study, M=SLOPE)",
            'HonestDiD(method="smoothness", M=0.0)',
        ):
            assert needle in code, needle
        # the MDV (a slope) is never passed as a smoothness bound, and the
        # event study's own pre-period classification is used
        assert "M=power_results.mdv" not in code and "M=mdv" not in code
        assert "pre_periods=" not in code
