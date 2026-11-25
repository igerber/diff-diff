"""Drift detection for Tutorial 02 (``docs/tutorials/02_staggered_did.ipynb``).

The tutorial's narrative rests on these claims:

1. **The truth is stated and recovered.** With ``dynamic_effects=True`` the
   true effect at event time e is ``treatment_effect * (1 + effect_growth * e)``
   = 2, 3, 4, 5, 6, and the true ATT over treated observations is 3.5769.
   Callaway-Sant'Anna estimates 3.4763 with an interval that covers it; every
   post-treatment event-study interval covers its true value.
2. **TWFE is off, and the decomposition says why.** The TWFE coefficient is
   3.0682, outside the CS interval and several CS standard errors below the
   truth; about a ninth (11.4%) of its weight is on later-vs-earlier
   comparisons.
3. **Simultaneous bands differ from pointwise intervals.** On the bootstrapped
   event study the sup-t band is wider than the pointwise interval in every
   row. On the universal-base fit several pointwise pre-treatment intervals
   exclude zero although parallel trends holds by construction, while the
   simultaneous band contains zero in every row.
4. **Agreement between CS and Sun-Abraham here is arithmetic.** With
   never-treated controls, no covariates and a balanced panel their point
   estimates are identical; not-yet-treated CS and the imputation estimator
   differ.
5. **Anticipation changes the estimand**: ``anticipation=1`` lowers the overall
   ATT because the zero-effect period before adoption enters the average.

Analytical numbers are pinned tightly. Multiplier-bootstrap numbers depend on
the backend's random draws (outputs are committed from the Rust backend, CI
also runs pure Python), so for those this file asserts the STRUCTURAL claims
the prose makes and checks the digits only on the rendered surface.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    ImputationDiD,
    SunAbraham,
    TwoWayFixedEffects,
    generate_did_data,
    generate_staggered_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/02_staggered_did.ipynb"

TREATMENT_EFFECT = 2.0
EFFECT_GROWTH = 0.5
FIT = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


@pytest.fixture(scope="module")
def df():
    return generate_staggered_data(
        n_units=100,
        n_periods=8,
        cohort_periods=[3, 5],
        never_treated_frac=0.4,
        treatment_effect=TREATMENT_EFFECT,
        dynamic_effects=True,
        effect_growth=EFFECT_GROWTH,
        unit_fe_sd=2.0,
        noise_sd=0.5,
        seed=42,
    )


@pytest.fixture(scope="module")
def true_att(df):
    return float(df.loc[df["treated"] == 1, "true_effect"].mean())


@pytest.fixture(scope="module")
def cs(df):
    return CallawaySantAnna(control_group="never_treated").fit(df, **FIT)


class TestTruth:
    def test_generator_effect_path(self, df, true_att):
        treated = df[df["treated"] == 1]
        by_e = treated.assign(e=treated["period"] - treated["first_treat"]).groupby("e")
        path = by_e["true_effect"].mean().to_dict()
        assert path == {
            e: pytest.approx(TREATMENT_EFFECT * (1 + EFFECT_GROWTH * e)) for e in range(5)
        }
        assert true_att == pytest.approx(3.5769, abs=5e-5)
        by_cohort = treated.groupby("first_treat")["true_effect"].mean().to_dict()
        assert by_cohort == {3: pytest.approx(4.0), 5: pytest.approx(3.0)}
        sizes = df.groupby("unit")["first_treat"].first().value_counts().to_dict()
        assert sizes == {0: 40, 3: 27, 5: 33}
        assert_quotes_in_rendered(
            NB,
            [
                "True effect by event time: {0: 2.0, 1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0}",
                "True ATT (average over treated observations): 3.5769",
            ],
            surface="output",
        )
        assert_quotes_in_rendered(
            NB,
            ["it is 2.0 in the\nadoption period and increases by 1.0 per period"],
            surface="markdown",
        )


class TestTwfeAndBacon:
    def test_twfe_is_outside_the_cs_interval(self, df, cs, true_att):
        bacon = TwoWayFixedEffects().decompose(df, **FIT)
        assert bacon.twfe_estimate == pytest.approx(3.0682, abs=5e-4)
        lower, upper = cs.conf_int
        assert not (lower <= bacon.twfe_estimate <= upper)
        # "several times the standard error" below the truth
        assert (true_att - bacon.twfe_estimate) / cs.se > 3.0
        # "about a ninth of the TWFE weight"
        assert bacon.total_weight_later_vs_earlier == pytest.approx(0.1145, abs=5e-4)
        assert abs(bacon.total_weight_later_vs_earlier - 1 / 9) < 0.01
        assert_quotes_in_rendered(
            NB,
            [
                "TWFE estimate: 3.0682",
                "Weight on later-vs-earlier comparisons: 11.4%",
                "TWFE estimate inside this CI: False",
            ],
            surface="output",
        )


class TestCallawaySantAnna:
    def test_overall_att_covers_truth(self, cs, true_att):
        assert cs.att == pytest.approx(3.4763, abs=5e-4)
        assert cs.se == pytest.approx(0.1164, abs=5e-4)
        lower, upper = cs.conf_int
        assert (lower, upper) == pytest.approx((3.2483, 3.7044), abs=5e-4)
        assert lower <= true_att <= upper
        assert_quotes_in_rendered(
            NB,
            [
                "Estimated ATT: 3.4763  (SE 0.1164)",
                "95% CI:        [3.2483, 3.7044]",
                "CI covers the true ATT: True",
            ],
            surface="output",
        )

    def test_event_study_covers_the_true_path(self, cs):
        es = cs.aggregate("event_study").to_dataframe()
        post = es[es["event_time"] >= 0]
        assert post["event_time"].tolist() == [0, 1, 2, 3, 4]
        np.testing.assert_allclose(post["att"], [1.8625, 2.8985, 4.0353, 4.7591, 5.8218], atol=5e-4)
        truth = TREATMENT_EFFECT * (1 + EFFECT_GROWTH * post["event_time"])
        assert ((post["conf_int_lower"] <= truth) & (truth <= post["conf_int_upper"])).all()

    def test_cohort_aggregation(self, cs):
        by_cohort = cs.aggregate("group").to_dataframe().set_index("label")
        assert by_cohort.loc[3, "att"] == pytest.approx(3.9681, abs=5e-4)
        assert by_cohort.loc[5, "att"] == pytest.approx(2.8058, abs=5e-4)
        for cohort, truth in ((3, 4.0), (5, 3.0)):
            row = by_cohort.loc[cohort]
            assert row["conf_int_lower"] <= truth <= row["conf_int_upper"]


class TestBootstrap:
    """Structural claims only: the draws differ by backend."""

    def test_band_is_wider_and_p_value_is_the_floor(self, df):
        boot = CallawaySantAnna(
            control_group="never_treated", n_bootstrap=999, bootstrap_weights="rademacher", seed=42
        ).fit(df, **FIT)
        # share of draws, floored at 1 / (B + 1)
        assert boot.p_value == pytest.approx(1.0 / 1000.0, rel=1e-9)
        es = boot.aggregate("event_study").to_dataframe()
        assert (es["cband_lower"] < es["conf_int_lower"]).all()
        assert (es["cband_upper"] > es["conf_int_upper"]).all()
        assert_quotes_in_rendered(NB, ["Bootstrap p-value: 0.0010"], surface="output")
        assert_quotes_in_rendered(
            NB, ["cannot be smaller than 1/(999 + 1) = 0.001"], surface="markdown"
        )

    def test_universal_base_pre_periods(self, df):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = CallawaySantAnna(
                control_group="never_treated", base_period="universal", n_bootstrap=999, seed=42
            ).fit(df, **FIT)
            es = res.aggregate("event_study").to_dataframe()
        # the warning the notebook explains: reference cells have zero bootstrap SE
        messages = [str(w.message) for w in caught]
        assert any("non-finite or zero bootstrap SE" in m for m in messages)

        pre = es[es["event_time"] < 0]
        assert pre["event_time"].tolist() == [-5, -4, -3, -2, -1]
        reference = pre[pre["is_reference"]]
        assert reference["event_time"].tolist() == [-1]
        assert reference["att"].iloc[0] == 0.0
        assert reference[["conf_int_lower", "cband_lower"]].isna().all(axis=None)

        estimated = pre[~pre["is_reference"]]
        np.testing.assert_allclose(
            estimated["att"], [-0.3391, -0.3913, -0.2356, -0.1720], atol=5e-4
        )
        # the simultaneous band contains zero in every row ...
        assert ((estimated["cband_lower"] < 0) & (estimated["cband_upper"] > 0)).all()
        # ... while several pointwise intervals do not ("Three of the pointwise
        # intervals exclude zero" is the committed Rust-backend count)
        excludes_zero = (estimated["conf_int_upper"] < 0) | (estimated["conf_int_lower"] > 0)
        assert excludes_zero.sum() >= 2
        assert_quotes_in_rendered(
            NB, ["Three of the pointwise intervals exclude zero"], surface="markdown"
        )
        assert_quotes_in_rendered(
            NB, ["-5 -0.3391         -0.6736         -0.0138"], surface="output"
        )


class TestComparisons:
    def test_control_groups_and_anticipation(self, df, cs):
        nyt = CallawaySantAnna(control_group="not_yet_treated").fit(df, **FIT)
        assert nyt.att == pytest.approx(3.4637, abs=5e-4)
        assert nyt.se < cs.se  # "slightly smaller with the larger comparison group"
        antic = CallawaySantAnna(control_group="never_treated", anticipation=1).fit(df, **FIT)
        assert antic.att == pytest.approx(2.9387, abs=5e-4)
        assert antic.att < cs.att
        assert_quotes_in_rendered(
            NB,
            ["not_yet_treated  3.4637  0.1140", "anticipation=1: ATT = 2.9387"],
            surface="output",
        )

    def test_cs_and_sun_abraham_coincide(self, df, cs, true_att):
        sa = SunAbraham(control_group="never_treated").fit(df, **FIT)
        np.testing.assert_allclose(sa.att, cs.att, rtol=1e-10)
        assert sa.se != pytest.approx(cs.se, rel=0.05)  # different variance estimators
        bjs = ImputationDiD().fit(df, **FIT)
        assert bjs.att == pytest.approx(3.6174, abs=5e-4)
        assert not np.isclose(bjs.att, cs.att, atol=0.01)
        for res in (sa, bjs):
            lower, upper = res.conf_int
            assert lower <= true_att <= upper
        assert_quotes_in_rendered(
            NB,
            [
                "CS and SA point estimates identical: True",
                "Imputation (Borusyak-Jaravel-Spiess)  3.6174  0.0687",
            ],
            surface="output",
        )

    def test_covariates_move_within_sampling_noise(self, df, cs):
        data = df.copy()
        rng = np.random.default_rng(2024)
        unit_ids = data["unit"].unique()
        data["size"] = data["unit"].map(
            pd.Series(rng.normal(100, 20, len(unit_ids)), index=unit_ids)
        )
        data["age"] = data["unit"].map(pd.Series(rng.normal(10, 3, len(unit_ids)), index=unit_ids))
        cov = CallawaySantAnna(control_group="never_treated").fit(
            data, covariates=["size", "age"], **FIT
        )
        assert cov.att == pytest.approx(3.4255, abs=5e-4)
        assert abs(cov.att - cs.att) < cs.se
        assert_quotes_in_rendered(
            NB, ["With covariates:    ATT = 3.4255 (SE 0.1207)"], surface="output"
        )


class TestSingleDateEventStudy:
    def test_twfe_event_study_covers_truth(self):
        data = generate_did_data(
            n_units=100,
            n_periods=8,
            treatment_period=4,
            treatment_fraction=0.5,
            treatment_effect=2.5,
            seed=42,
        )
        res = TwoWayFixedEffects().fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            event_study=True,
            post_periods=[4, 5, 6, 7],
        )
        table = res.to_dataframe()
        assert table.loc[table["is_reference"], "event_time"].tolist() == [3]
        post = table[table["event_time"] >= 4]
        np.testing.assert_allclose(post["att"], [2.5355, 2.2599, 2.6983, 2.7879], atol=5e-4)
        assert ((post["conf_int_lower"] <= 2.5) & (2.5 <= post["conf_int_upper"])).all()


class TestNotebookSurface:
    def test_states_assumptions_and_cautions(self):
        md = notebook_markdown(NB)
        for needle in (
            "**Parallel trends.**",
            "**No anticipation.**",
            "two different assumptions, not a stronger and a weaker version of one",
            "do not treat pre-treatment estimates as a pass/fail test",
            "Their agreement here is arithmetic, so it is not a robustness check.",
        ):
            assert needle in md, needle

    def test_no_removed_aggregation_name(self):
        """The 3.x value ``aggregate("event")`` raises in 4.0."""
        md = notebook_markdown(NB)
        assert '"event"' not in md
        assert '"event_study"' in md
