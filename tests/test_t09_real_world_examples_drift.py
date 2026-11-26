"""Drift detection for Tutorial 09 (``docs/tutorials/09_real_world_examples.ipynb``).

The tutorial's narrative rests on these claims:

1. **Card-Krueger: the significance conclusion depends on clustering.** On the
   384 restaurants observed in both waves the DiD is +2.75. Clustered by
   restaurant the interval lies above zero; with rows treated as independent it
   includes zero.
2. **Castle doctrine: TWFE and Callaway-Sant'Anna disagree because of the
   baseline, not forbidden comparisons.** TWFE is 0.1346 with about 3% of its
   weight on later-vs-earlier comparisons and about 91% on treated-vs-never;
   CS is 0.4725. For the 2006 cohort the TWFE 2x2 building block is 0.0069 and
   the CS cohort effect 0.4991.
3. **Castle doctrine pre-treatment estimates are not reassuring.** Under the
   universal base, the estimates six to two years before adoption are all
   positive (about 0.2 to 0.6), and the earliest row comes from one cohort with
   a simultaneous band that excludes zero.
4. **Divorce panel (simulated by design).** 21 group-time cells of the cohort
   treated in the first panel year cannot be estimated, so 12 of 13 cohorts
   have estimates; near adoption every simultaneous band includes zero.

Real data: the Card-Krueger and Castle loaders fall back to a generated panel
when neither the source nor a verified cache is reachable, and the canonical
numbers cannot be re-derived from that, so those tests ``pytest.skip`` on
``attrs["source"] == "synthetic_fallback"``. ``load_divorce_laws`` is synthetic
by design (fixed seed, always warns), so its pins are exact and never skipped.

Bootstrap draws are backend-specific (outputs are committed from the Rust
backend; CI also runs pure Python): point estimates and analytical quantities
are pinned tightly, bootstrap intervals and bands structurally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from diff_diff import (
    CallawaySantAnna,
    DifferenceInDifferences,
    TwoWayFixedEffects,
    load_card_krueger,
    load_castle_doctrine,
    load_divorce_laws,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/09_real_world_examples.ipynb"

CASTLE_FIT = dict(outcome="homicide_rate", unit="state", time="year", first_treat="first_treat")
DIVORCE_FIT = dict(outcome="divorce_rate", unit="state", time="year", first_treat="first_treat")


def _canonical_or_skip(loader, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = loader()
    if df.attrs.get("source") == "synthetic_fallback":
        pytest.skip(
            f"{name}: canonical data unavailable (synthetic fallback); "
            "the replication numbers cannot be re-derived offline"
        )
    return df


@pytest.fixture(scope="module")
def ck_long():
    ck = _canonical_or_skip(load_card_krueger, "card_krueger")
    both = ck.dropna(subset=["emp_pre", "emp_post"])
    assert (len(both), len(ck)) == (384, 410)
    long = both.melt(
        id_vars=["store_id", "state", "chain", "treated"],
        value_vars=["emp_pre", "emp_post"],
        var_name="wave",
        value_name="employment",
    )
    long["post"] = (long["wave"] == "emp_post").astype(int)
    return long


@pytest.fixture(scope="module")
def castle():
    return _canonical_or_skip(load_castle_doctrine, "castle_doctrine")


@pytest.fixture(scope="module")
def divorce():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_divorce_laws()
    assert any("SYNTHETIC" in str(w.message) for w in caught)
    assert df.attrs["source"] == "synthetic_fallback"
    return df


class TestCardKrueger:
    def test_clustering_changes_the_conclusion(self, ck_long):
        fit = dict(outcome="employment", treatment="treated", post="post")
        clustered = DifferenceInDifferences(cluster="store_id").fit(ck_long, **fit)
        unclustered = DifferenceInDifferences().fit(ck_long, **fit)
        np.testing.assert_allclose(clustered.att, unclustered.att, rtol=1e-10)
        assert clustered.att == pytest.approx(2.7500, abs=5e-4)
        assert clustered.se == pytest.approx(1.3386, abs=5e-4)
        assert unclustered.se == pytest.approx(1.8428, abs=5e-4)
        assert clustered.conf_int[0] > 0  # "the interval lies above zero"
        assert unclustered.conf_int[0] < 0 < unclustered.conf_int[1]  # "includes zero"
        assert clustered.n_clusters == 384
        assert_quotes_in_rendered(
            NB,
            [
                "Restaurants observed in both waves: 384 of 410",
                "Difference-in-differences by hand: 2.7500",
                "clustered by restaurant      1.3386   0.0406    0.1181    5.3819",
                "rows treated as independent  1.8428   0.1360   -0.8676    6.3676",
            ],
            surface="output",
        )

    def test_state_changes(self, ck_long):
        means = ck_long.groupby(["state", "post"])["employment"].mean().unstack()
        change = means[1] - means[0]
        # "rose slightly while it fell in Pennsylvania"
        assert change["NJ"] == pytest.approx(0.467, abs=1e-3)
        assert change["PA"] == pytest.approx(-2.283, abs=1e-3)
        assert change["NJ"] - change["PA"] == pytest.approx(2.7500, abs=5e-4)


class TestCastleDoctrine:
    def test_cohorts(self, castle):
        cohorts = castle.drop_duplicates("state").groupby("first_treat").size().to_dict()
        assert cohorts == {0: 29, 2005: 1, 2006: 13, 2007: 4, 2008: 2, 2009: 1}
        assert sum(cohorts.values()) == 50  # "29 of the 50 states never adopt"

    def test_twfe_decomposition(self, castle):
        bacon = TwoWayFixedEffects().decompose(castle, **CASTLE_FIT)
        assert bacon.twfe_estimate == pytest.approx(0.1346, abs=5e-4)
        assert bacon.total_weight_later_vs_earlier == pytest.approx(0.032, abs=1e-3)  # "about 3%"
        assert bacon.total_weight_treated_vs_never > 0.90  # "more than 90%"
        table = bacon.to_dataframe()
        row = table[
            (table["treated_group"] == 2006) & (table["comparison_type"] == "treated_vs_never")
        ].iloc[0]
        assert row["estimate"] == pytest.approx(0.0069, abs=5e-4)
        assert row["weight"] == pytest.approx(0.592, abs=1e-3)
        assert_quotes_in_rendered(
            NB,
            [
                "TWFE estimate: 0.1346",
                "later vs earlier treated:  3.2%",
                "TWFE building block (2x2 DiD): 0.0069   weight in TWFE: 59.2%",
            ],
            surface="output",
        )

    def test_cs_estimate_and_baseline_gap(self, castle):
        cs = CallawaySantAnna(control_group="never_treated").fit(castle, **CASTLE_FIT)
        assert cs.att == pytest.approx(0.4725, abs=5e-4)
        bacon = TwoWayFixedEffects().decompose(castle, **CASTLE_FIT)
        assert cs.att / bacon.twfe_estimate > 3  # "a factor of more than three"
        by_cohort = cs.aggregate("group").to_dataframe().set_index("label")
        assert by_cohort.loc[2006, "att"] == pytest.approx(0.4991, abs=5e-4)
        assert_quotes_in_rendered(NB, ["Callaway-Sant'Anna:            0.4991"], surface="output")

    def test_pre_treatment_path(self, castle):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = CallawaySantAnna(
                control_group="never_treated", base_period="universal", n_bootstrap=999, seed=42
            ).fit(castle, **CASTLE_FIT)
            es = res.aggregate("event_study").to_dataframe().set_index("event_time")
        assert any("non-finite or zero bootstrap SE" in str(w.message) for w in caught)
        assert es.index.tolist() == list(range(-9, 6))
        assert bool(es.loc[-1, "is_reference"]) and es.loc[-1, "att"] == 0.0
        # "From six years to two years before adoption they are all positive,
        # between about 0.2 and 0.6"
        np.testing.assert_allclose(
            es.loc[[-6, -5, -4, -3, -2], "att"], [0.5619, 0.5564, 0.1663, 0.4702, 0.3231], atol=5e-4
        )
        # the earliest row: one cohort, band excludes zero
        assert es.loc[-9, "n"] == 1
        assert es.loc[-9, "att"] == pytest.approx(-1.1818, abs=5e-4)
        assert es.loc[-9, "cband_upper"] < 0
        # post-adoption: positive; pointwise intervals exclude zero in the first two years
        np.testing.assert_allclose(es.loc[[0, 1], "att"], [0.4655, 0.7781], atol=5e-4)
        assert (es.loc[0:5, "att"] > 0).all()
        assert (es.loc[[0, 1], "conf_int_lower"] > 0).all()
        # ... "the simultaneous band is much less clear-cut": wider, lower end near zero
        assert (es.loc[[0, 1], "cband_lower"] < es.loc[[0, 1], "conf_int_lower"]).all()
        assert es.loc[0, "cband_lower"] < 0.15

    def test_not_yet_treated_agrees(self, castle):
        nt = CallawaySantAnna(control_group="never_treated").fit(castle, **CASTLE_FIT)
        nyt = CallawaySantAnna(control_group="not_yet_treated").fit(castle, **CASTLE_FIT)
        assert nyt.att == pytest.approx(0.4748, abs=5e-4)
        assert abs(nyt.att - nt.att) < 0.01  # "The two agree closely"


class TestDivorcePanel:
    """Synthetic by design: exact, never skipped."""

    def test_first_period_cohort_cannot_be_estimated(self, divorce):
        cohorts = divorce.drop_duplicates("state").groupby("first_treat").size()
        assert cohorts.loc[1968] == 1 and divorce["year"].min() == 1968
        assert (cohorts.index > 0).sum() == 13  # "13 adoption years appear in the data"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = CallawaySantAnna(control_group="never_treated").fit(divorce, **DIVORCE_FIT)
        assert any(
            "21 (group, time) cell(s) could not be estimated" in str(w.message) for w in caught
        )
        gt = res.to_dataframe()
        skipped = gt[gt["skip_reason"].notna()]
        assert len(skipped) == 21 and set(skipped["group"]) == {1968}
        by_cohort = res.aggregate("group").to_dataframe()
        assert len(by_cohort) == 12 and 1968 not in set(by_cohort["label"])
        assert res.att == pytest.approx(0.1917, abs=5e-4)
        assert_quotes_in_rendered(NB, ["Cohorts with an estimate: 12"], surface="output")

    def test_event_study_shape(self, divorce):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(control_group="never_treated", n_bootstrap=999, seed=42).fit(
                divorce, **DIVORCE_FIT
            )
            es = res.aggregate("event_study").to_dataframe().set_index("event_time")
        # jump at adoption, elevated for about five years, then back toward zero
        np.testing.assert_allclose(
            es.loc[0:5, "att"], [0.560, 0.503, 0.460, 0.277, 0.360, 0.391], atol=1e-3
        )
        assert (es.loc[0:5, "att"] > 0.25).all()
        assert (es.loc[6:10, "att"].abs() < 0.2).all()
        # near adoption: small estimates, every simultaneous band includes zero
        near = es.loc[-5:-1]
        assert near["att"].abs().max() < 0.2
        assert ((near["cband_lower"] < 0) & (near["cband_upper"] > 0)).all()
        # the far ends rest on one to three cohorts
        assert es.loc[[-18, -17, 19], "n"].max() <= 3


class TestNotebookSurface:
    def test_cautions_are_stated(self):
        md = notebook_markdown(NB)
        for needle in (
            "The clustered version is the appropriate one.",
            "no support for a sizeable\n  *negative* employment effect",
            "The reason is the *baseline*.",
            "should not be read as the causal\neffect of the laws",
            "not proof of it",
            "**Simulated data.**",
        ):
            assert needle in md, needle
        # the removed claims
        assert "not statistically significant" not in md.replace(
            'one would say "positive\nbut not statistically significant"', ""
        )
        assert "Robustness Check" not in md
