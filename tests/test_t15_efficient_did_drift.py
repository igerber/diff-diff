"""Drift detection for Tutorial 15 (``docs/tutorials/15_efficient_did.ipynb``).

The tutorial fits ``EfficientDiD`` on a staggered panel with a constant true
effect of 2.0 and parallel trends in every period. Its claims:

1. **Truth check.** ATT 1.98, 95% CI [1.91, 2.05], which contains 2.0.
2. **PT-Post reproduces Callaway-Sant'Anna exactly**; under PT-All the SE is
   69% of that, and across ten panels the ratio is 0.64 to 0.75 (mean 0.69).
3. **The Hausman pretest** does not reject (p = 0.216) and warns that its
   variance-difference matrix is not positive definite in this sample.
4. **One pre-treatment interval excludes zero** (event time -5) although
   parallel trends holds by construction.
5. **Diagnostics.** ATT(5, 6) combines 21 comparisons (9 + 2 + 4 + 6) with
   weights that sum to one and include negative ones; condition numbers run
   from about 80 to 160, with no threshold applied.
6. **Against Imputation DiD** the standard errors agree within 1%.

Analytical quantities are deterministic. Bootstrap standard errors are pinned
loosely because the draws are backend-specific.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from diff_diff import (
    CallawaySantAnna,
    EfficientDiD,
    ImputationDiD,
    PreTrendsPower,
    compute_honest_did,
    generate_staggered_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/15_efficient_did.ipynb"

TRUE_ATT = 2.0
FIT = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


@pytest.fixture(scope="module")
def data():
    return generate_staggered_data(
        n_units=300, n_periods=10, treatment_effect=TRUE_ATT, dynamic_effects=False, seed=42
    )


@pytest.fixture(scope="module")
def results(data):
    return EfficientDiD(pt_assumption="all").fit(data, **FIT)


@pytest.fixture(scope="module")
def callaway(data):
    return CallawaySantAnna().fit(data, **FIT)


class TestFitAndAssumptions:
    def test_estimate_covers_truth(self, results):
        assert results.att == pytest.approx(1.9808, abs=5e-4)
        assert results.se == pytest.approx(0.0366, abs=2e-4)
        assert results.conf_int[0] <= TRUE_ATT <= results.conf_int[1]
        assert_quotes_in_rendered(
            NB, ["Estimate: 1.981  (SE 0.0366)", "95% CI:   [1.909, 2.053]"], surface="output"
        )

    def test_pt_post_is_callaway_santanna(self, results, callaway, data):
        post = EfficientDiD(pt_assumption="post").fit(data, **FIT)
        assert post.att == pytest.approx(callaway.att, abs=1e-9)
        assert post.se == pytest.approx(callaway.se, abs=1e-9)
        assert callaway.att == pytest.approx(1.9754, abs=5e-4)
        assert results.se / post.se == pytest.approx(0.69, abs=0.01)
        assert_quotes_in_rendered(
            NB, ["PT-All SE as a share of the PT-Post SE: 69%"], surface="output"
        )

    def test_gain_is_systematic_across_panels(self):
        ratios = []
        for seed in range(10):
            panel = generate_staggered_data(
                n_units=200,
                n_periods=8,
                treatment_effect=TRUE_ATT,
                dynamic_effects=False,
                seed=seed,
            )
            efficient = EfficientDiD(pt_assumption="all").fit(panel, **FIT)
            baseline = CallawaySantAnna().fit(panel, **FIT)
            ratios.append(efficient.se / baseline.se)
        assert np.mean(ratios) == pytest.approx(0.692, abs=5e-3)
        assert min(ratios) == pytest.approx(0.640, abs=5e-3)
        assert max(ratios) == pytest.approx(0.752, abs=5e-3)
        assert_quotes_in_rendered(
            NB, ["Mean ratio: 0.692   (range 0.640 to 0.752)"], surface="output"
        )

    def test_hausman_pretest(self, data):
        with pytest.warns(UserWarning, match="negative eigenvalue"):
            hausman = EfficientDiD.hausman_pretest(data, **FIT)
        assert hausman.statistic == pytest.approx(8.318, abs=5e-3)
        assert hausman.df == 6
        assert hausman.p_value == pytest.approx(0.216, abs=2e-3)
        assert hausman.recommendation == "pt_all"
        assert_quotes_in_rendered(
            NB, ["Statistic: 8.318  (df 6)", "p-value:   0.216"], surface="output"
        )


class TestSummaries:
    def test_event_study_has_one_spurious_pre_period(self, results):
        es = results.aggregate("event_study").to_dataframe().set_index("event_time")
        assert list(es.index) == list(range(-6, 7))
        assert not es["is_reference"].any()
        post = es.loc[0:6]
        assert (post["att"] - TRUE_ATT).abs().max() < 0.07
        pre = es.loc[-6:-1]
        excludes_zero = (pre["conf_int_lower"] > 0) | (pre["conf_int_upper"] < 0)
        assert list(pre.index[excludes_zero]) == [-5]
        # under independence, at least one of six 95% intervals misses zero 26% of the time
        assert 1 - 0.95**6 == pytest.approx(0.26, abs=0.01)

    def test_group_time_and_cohorts(self, results):
        group_time = results.to_dataframe(level="group_time")
        assert len(group_time) == 27
        assert group_time["time"].min() == 1  # period 0 is the common first period
        groups = results.aggregate("group").to_dataframe().set_index("label")
        np.testing.assert_allclose(
            groups.loc[[3, 5, 7], "att"].to_numpy(), [1.996, 1.993, 1.932], atol=1e-3
        )

    def test_bootstrap_agrees_with_analytical(self, results, data):
        for weights in ("rademacher", "mammen", "webb"):
            fit = EfficientDiD(
                pt_assumption="all", n_bootstrap=999, bootstrap_weights=weights, seed=42
            ).fit(data, **FIT)
            assert fit.att == pytest.approx(results.att, rel=1e-9)
            assert fit.se == pytest.approx(results.se, rel=0.10)


class TestDiagnosticsAndComparison:
    def test_weights_and_conditioning(self, results):
        weights = results.efficient_weights[(5, 6)]
        # never-treated with baselines 1..9, cohorts 3, 5, 7 with their own pre-periods
        assert len(weights) == 9 + 2 + 4 + 6 == 21
        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert weights.min() < 0 < weights.max()
        counts = {len(w) for w in results.efficient_weights.values()}
        assert counts == {20, 21}
        conditioning = np.array(list(results.omega_condition_numbers.values()))
        assert 70 < conditioning.min() < 90 and 150 < conditioning.max() < 175

    def test_anticipation(self, results, data):
        fit = EfficientDiD(pt_assumption="all", anticipation=1).fit(data, **FIT)
        assert fit.att == pytest.approx(1.665, abs=2e-3)
        assert fit.att < results.att - 0.25

    def test_imputation_is_equally_precise_here(self, results, callaway, data):
        imputation = ImputationDiD().fit(data, **FIT)
        assert results.se / imputation.se == pytest.approx(1.0, abs=0.01)
        assert results.se / callaway.se == pytest.approx(0.69, abs=0.01)
        for fit in (results, imputation, callaway):
            assert fit.conf_int[0] <= TRUE_ATT <= fit.conf_int[1]


class TestNotebookSurface:
    def test_diagnostics_take_a_callaway_santanna_fit_not_this_one(self, results, data):
        """The notebook sends readers to Pre-Trends Power and Honest DiD; both
        reject this estimator's event study and accept a universal-base
        Callaway-Sant'Anna one, which is the fit the notebook names."""
        own = results.aggregate("event_study")
        with pytest.raises(TypeError, match="PreTrendsPower accepts"):
            PreTrendsPower().fit(own)
        with pytest.raises(TypeError, match="HonestDiD accepts"):
            compute_honest_did(own, method="relative_magnitude", M=1.0)
        supported = (
            CallawaySantAnna(base_period="universal").fit(data, **FIT).aggregate("event_study")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PreTrendsPower().fit(supported)
            compute_honest_did(supported, method="relative_magnitude", M=1.0)
        md = notebook_markdown(NB)
        assert "Neither takes this estimator's event study." in md
        assert '`CallawaySantAnna(base_period="universal")` fit of the same data' in md

    def test_prose(self):
        md = notebook_markdown(NB)
        for needle in (
            "The efficiency is not free: it is paid for with the assumption.",
            "this is a pre-test",
            "A single individually significant pre-treatment coefficient is weak evidence",
            "it is not the negative-weights problem of two-way fixed\neffects",
            "there is no universal threshold",
            "Neither dominates",
            "pre-treatment period *for the\ncomparison group*",
            "difference is positive semidefinite when PT-All holds",
        ):
            assert needle in md, needle
        for stale in ("(3.9", "(3.10)", "M-023", "REGISTRY", "<< HIGH", " --- "):
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
            'EfficientDiD(pt_assumption="all")',
            "EfficientDiD.hausman_pretest(data, **fit_args)",
            'es = results.aggregate("event_study")',
            "plot_event_study(es,",
        ):
            assert needle in code, needle
        # no rename back to removed column names, no hard-coded conditioning threshold
        assert "relative_period" not in code and "> 100" not in code


def test_pt_all_comparisons_are_described_as_two_changes():
    """Eq. 3.9: the comparison group supplies the change up to the baseline (and
    may be the target cohort itself); the never-treated units supply the change
    from the baseline to the target period."""
    md = notebook_markdown(NB)
    assert "the never-treated units, another cohort, or the target cohort\n  itself" in md
    assert "The change from the baseline to period t comes from the never-treated units." in md
    assert "Cohort 5 is among its own\ncomparison groups." in md
    assert "(the never-treated units, or another cohort) paired with any" not in md
