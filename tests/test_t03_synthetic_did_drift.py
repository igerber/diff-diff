"""Drift detection for Tutorial 03 (``docs/tutorials/03_synthetic_did.ipynb``).

The tutorial's narrative rests on these claims:

1. **Plain DiD is biased by construction** on the panel (the average control
   trends more slowly than the treated units): its unit-clustered estimate is
   more than twice the true ATT of 5.0 and its interval excludes it.
2. **Synthetic DiD recovers the truth**: about 4.98, interval covering 5.0,
   with the weight on the ten "similar" controls.
3. **Time weights are fit on controls and are degenerate here**: all weight on
   the last pre-treatment period.
4. **Fit is judged by shape**: a constant level gap of about -1.5 (synthetic
   above treated) with a small shape RMSE and a placebo fit p-value above 0.05.
5. **p-values are method-specific**: the placebo p-value is a count at its
   floor 1/(199 + 1); the jackknife interval is wider; the jackknife is NaN
   with a single treated unit.
6. **Regularization sensitivity**: stable near the default, drifting up at 4x.

Structure (the T24/T25/T27 split). The point estimate, both sets of weights,
the fit diagnostics and the regularization sweep do not depend on the variance
method, so the unmarked tests read them from a deterministic jackknife fit
(seconds on either backend). The placebo fits cost about 0.2 s per replication
in pure Python and must reproduce the notebook's seed and replication count,
so they are ``@pytest.mark.slow`` and run in the ``-m ''`` CI legs. Placebo
replications are drawn by the active backend: their SE and interval are pinned
loosely and the digits are checked on the rendered surface.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import DifferenceInDifferences, SyntheticDiD
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/03_synthetic_did.ipynb"

TRUE_ATT = 5.0
N_TREATED, N_CONTROL = 3, 40
N_PRE, N_POST = 10, 4
N_PERIODS = N_PRE + N_POST
N_PLACEBO = 199
FIT = dict(
    outcome="outcome",
    treatment="treated",
    unit="unit",
    time="period",
    post_periods=list(range(N_PRE, N_PERIODS)),
)


def make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    control_intercepts = np.concatenate([np.linspace(45, 55, 10), np.linspace(30, 70, 30)])
    control_trends = np.concatenate([np.linspace(1.7, 2.3, 10), np.linspace(0.3, 1.0, 30)])
    treated_intercepts = [48, 50, 52]
    treated_trends = [1.8, 2.0, 2.2]
    rows = []
    for i in range(N_CONTROL):
        for period in range(N_PERIODS):
            outcome = control_intercepts[i] + control_trends[i] * period + rng.normal(0, 0.5)
            rows.append({"unit": i, "period": period, "treated": 0, "outcome": outcome})
    for i in range(N_TREATED):
        for period in range(N_PERIODS):
            outcome = (
                treated_intercepts[i]
                + treated_trends[i] * period
                + TRUE_ATT * (period >= N_PRE)
                + rng.normal(0, 0.5)
            )
            rows.append({"unit": N_CONTROL + i, "period": period, "treated": 1, "outcome": outcome})
    df = pd.DataFrame(rows)
    df["post"] = (df["period"] >= N_PRE).astype(int)
    return df


@pytest.fixture(scope="module")
def df():
    return make_panel()


@pytest.fixture(scope="module")
def results(df):
    """Deterministic fit: the same ATT, weights and fit diagnostics as the
    notebook's placebo fit (none of them depends on the variance method)."""
    return SyntheticDiD(variance_method="jackknife").fit(df, **FIT)


@pytest.fixture(scope="module")
def placebo(df):
    """The notebook's headline fit. Slow in pure Python: slow tests only."""
    return SyntheticDiD(n_bootstrap=N_PLACEBO, seed=42).fit(df, **FIT)


class TestPlainDiD:
    def test_plain_did_excludes_the_truth(self, df):
        res = DifferenceInDifferences(cluster="unit").fit(
            df, outcome="outcome", treatment="treated", post="post"
        )
        assert res.att == pytest.approx(12.0480, abs=5e-4)
        assert res.att > 2 * TRUE_ATT  # "more than twice the true effect"
        lower, upper = res.conf_int
        assert not (lower <= TRUE_ATT <= upper)
        assert_quotes_in_rendered(
            NB,
            ["Plain DiD ATT: 12.0480  (SE 1.0102)", "CI covers the true ATT: False"],
            surface="output",
        )


class TestSyntheticDiD:
    def test_point_estimate(self, results):
        assert results.att == pytest.approx(4.9766, abs=5e-4)
        assert abs(results.att - TRUE_ATT) < 0.1
        assert_quotes_in_rendered(
            NB,
            [
                "Synthetic DiD ATT: 4.9766  (SE 0.3600)",
                "95% CI:            [4.2709, 5.6822]",
                "CI covers the true ATT: True",
            ],
            surface="output",
        )

    @pytest.mark.slow
    def test_placebo_interval_covers_truth(self, placebo, results):
        assert placebo.variance_method == "placebo"
        np.testing.assert_allclose(placebo.att, results.att, rtol=1e-10)
        lower, upper = placebo.conf_int
        assert lower <= TRUE_ATT <= upper
        # placebo draws are backend-specific: loose numeric pin, digits on the surface
        assert placebo.se == pytest.approx(0.36, abs=0.08)

    def test_weight_on_similar_controls(self, results):
        weights = results.get_unit_weights_df().set_index("unit")["weight"]
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert (weights >= 0).all()
        # "Units 0 to 9 are the similar controls, and they carry most of the weight"
        assert weights.loc[range(10)].sum() > 0.75
        assert set(weights.sort_values(ascending=False).head(5).index) <= set(range(10))
        assert int((weights > 0.001).sum()) == 18
        conc = results.get_weight_concentration()
        assert conc["effective_n"] == pytest.approx(11.38, abs=0.01)
        assert conc["top_k"] == 5
        assert conc["top_k_share"] == pytest.approx(0.520, abs=0.001)
        assert_quotes_in_rendered(
            NB,
            [
                "Control units with weight above 0.001: 18 of 40",
                "Effective number of controls:           11.4",
                "Share of weight on the top 5 units:      52.0%",
            ],
            surface="output",
        )

    def test_time_weights_sit_on_the_last_pre_period(self, results):
        tw = results.get_time_weights_df().set_index("period")["weight"]
        assert list(tw.index) == list(range(N_PRE))
        assert tw.loc[N_PRE - 1] == pytest.approx(1.0, abs=1e-6)
        assert tw.drop(N_PRE - 1).abs().max() < 1e-6
        assert_quotes_in_rendered(
            NB,
            ["all of the weight lands on the last pre-treatment period", "**control units only**"],
            surface="markdown",
        )

    def test_fit_diagnostics(self, results):
        assert results.pre_treatment_fit == pytest.approx(0.2946, abs=5e-4)
        # treated minus synthetic: negative = the synthetic sits above the treated units
        assert results.pre_treatment_level_gap == pytest.approx(-1.5018, abs=5e-4)
        assert results.pre_fit_placebo_pvalue > 0.05
        assert len(results.treated_pre_trajectory) == N_PRE
        gap = np.asarray(results.treated_pre_trajectory) - np.asarray(
            results.synthetic_pre_trajectory
        )
        assert gap.mean() == pytest.approx(results.pre_treatment_level_gap, abs=1e-8)
        assert_quotes_in_rendered(
            NB,
            ["Pre-treatment level gap:              -1.5018", "Placebo fit p-value:"],
            surface="output",
        )


class TestInference:
    @pytest.mark.slow
    def test_placebo_p_value_is_the_floor(self, placebo):
        draws = np.asarray(placebo.variance_effects)
        assert len(draws) == N_PLACEBO
        # "None of the 199 placebo estimates is as large as the actual one"
        assert int(np.sum(np.abs(draws) >= abs(placebo.att))) == 0
        assert placebo.p_value == pytest.approx(1.0 / (N_PLACEBO + 1), rel=1e-9)

    def test_placebo_floor_is_what_the_notebook_says(self):
        assert_quotes_in_rendered(
            NB,
            ["at its floor of 1/(199 + 1) = 0.005", "None of the 199 placebo estimates"],
            surface="markdown",
        )
        assert_quotes_in_rendered(
            NB,
            ["placebo (199 replications)  0.3600    0.005    4.2709    5.6822"],
            surface="output",
        )

    def test_jackknife_interval(self, results):
        assert results.variance_method == "jackknife"
        assert results.se == pytest.approx(0.5995, abs=5e-4)
        lower, upper = results.conf_int
        assert (lower, upper) == pytest.approx((3.8015, 6.1516), abs=5e-4)
        assert lower <= TRUE_ATT <= upper
        assert_quotes_in_rendered(
            NB,
            ["jackknife                   0.5995    0.000    3.8015    6.1516"],
            surface="output",
        )

    @pytest.mark.slow
    def test_jackknife_is_wider_than_placebo(self, placebo, results):
        assert results.se > placebo.se

    def test_single_treated_unit_jackknife_is_nan(self, df):
        """The jackknife is not defined with a single treated unit and returns NaN."""
        single = df[(df["treated"] == 0) | (df["unit"] == N_CONTROL)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jk = SyntheticDiD(variance_method="jackknife").fit(single, **FIT)
        assert jk.att == pytest.approx(4.6370, abs=5e-4)
        assert np.isnan(jk.se) and np.isnan(jk.p_value)
        assert np.isnan(jk.conf_int[0]) and np.isnan(jk.conf_int[1])
        assert any("more than 1 treated unit" in str(w.message) for w in caught)
        assert_quotes_in_rendered(NB, ["ATT:    4.6370  (SE 0.6448)"], surface="output")

    @pytest.mark.slow
    def test_single_treated_unit_placebo(self, df, placebo):
        single = df[(df["treated"] == 0) | (df["unit"] == N_CONTROL)]
        res = SyntheticDiD(n_bootstrap=N_PLACEBO, seed=42).fit(single, **FIT)
        assert res.se > placebo.se  # "the standard error is larger"
        lower, upper = res.conf_int
        assert lower <= TRUE_ATT <= upper


class TestRegularizationAndCovariates:
    def test_sensitivity_table(self, results):
        table = results.sensitivity_to_zeta_omega()
        np.testing.assert_allclose(
            table["zeta_omega"], np.array([0.25, 0.5, 1.0, 2.0, 4.0]) * 1.6919, rtol=1e-3
        )
        np.testing.assert_allclose(
            table["att"], [5.1743, 5.0330, 4.9766, 5.1177, 5.5585], atol=5e-4
        )
        # more regularization: weight spreads out, fit loosens
        assert table["effective_n"].is_monotonic_increasing
        assert table["pre_fit_rmse"].is_monotonic_increasing
        # "stable around the default", "at four times the default it starts to drift upward"
        assert table["att"].iloc[:4].sub(TRUE_ATT).abs().max() < 0.2
        assert table["att"].iloc[4] - TRUE_ATT > 0.5

    def test_default_zeta_formula(self, df, results):
        control = df[df["treated"] == 0].pivot(index="period", columns="unit", values="outcome")
        noise = float(np.std(np.diff(control.loc[: N_PRE - 1].to_numpy(), axis=0), ddof=1))
        assert results.noise_level == pytest.approx(noise, rel=1e-8)
        assert results.zeta_omega == pytest.approx((N_TREATED * N_POST) ** 0.25 * noise, rel=1e-8)
        assert results.zeta_lambda == pytest.approx(1e-6 * noise, rel=1e-8)

    def test_noise_covariates_hardly_change_the_estimate(self, df, results):
        data = df.copy()
        rng_cov = np.random.default_rng(2024)
        data["size"] = rng_cov.normal(100, 20, len(data))
        data["age"] = rng_cov.normal(10, 3, len(data))
        # the point estimate does not depend on the variance method
        res = SyntheticDiD(variance_method="jackknife").fit(data, covariates=["size", "age"], **FIT)
        assert res.att == pytest.approx(4.9318, abs=5e-4)
        assert abs(res.att - results.att) < 0.1
        assert_quotes_in_rendered(
            NB, ["With covariates:    ATT = 4.9318 (SE 0.3585)"], surface="output"
        )


class TestNotebookSurface:
    def test_states_requirements(self):
        md = notebook_markdown(NB)
        for needle in (
            "**some weighted average of the control\nunits**",
            "**no anticipation**",
            "(block\ntreatment), and the panel is balanced",
            "never reported below 1/(replications + 1)",
        ):
            assert needle in md, needle

    def test_notebook_constants_match(self):
        """The DGP constants and replication count duplicated here are the notebook's."""
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "N_PLACEBO = 199",
            "TRUE_ATT = 5.0",
            "np.random.default_rng(42)",
            "np.random.default_rng(2024)",
            "SyntheticDiD(n_bootstrap=N_PLACEBO, seed=42)",
        ):
            assert needle in source, f"notebook code no longer contains {needle!r}"
