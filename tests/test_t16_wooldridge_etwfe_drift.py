"""Drift detection for Tutorial 16 (``docs/tutorials/16_wooldridge_etwfe.ipynb``).

The tutorial fits ``WooldridgeDiD`` on a staggered panel with a constant true
effect of 2.0, on transformed count and binary outcomes, on the ``mpdta``
benchmark, and on an all-eventually-treated panel. Its claims:

1. **Truth check.** Default fit 1.98, 95% CI [1.91, 2.05], which contains 2.0.
2. **The control group changes the regression.** The default has no
   pre-treatment cells; ``never_treated`` has twelve placebo cells and omits
   one reference cell per cohort (g - 1).
3. **With never-treated controls ETWFE reproduces Callaway-Sant'Anna**; the
   default is 30% more precise and equals Imputation DiD's estimate.
4. **Poisson and logit recover the truth on the outcome's scale**: 222.6
   against a true 223.1 in levels, 0.246 against a true 0.231 in probability.
5. **All-eventually-treated panel**: periods 8 and 9 are dropped with a
   warning, cohort 8 is the reference, and the estimate covers the true 1.5.
6. **mpdta**: overall log-employment effect -0.048 (SE 0.013).

Everything is deterministic. The ``mpdta`` pins are skipped when the loader
returns its synthetic fallback (no network and no cache).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    ImputationDiD,
    WooldridgeDiD,
    generate_staggered_data,
    load_mpdta,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/16_wooldridge_etwfe.ipynb"

TRUE_ATT = 2.0
FIT = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


@pytest.fixture(scope="module")
def data():
    return generate_staggered_data(
        n_units=300, n_periods=10, treatment_effect=TRUE_ATT, dynamic_effects=False, seed=42
    )


@pytest.fixture(scope="module")
def results(data):
    return WooldridgeDiD().fit(data, **FIT)


@pytest.fixture(scope="module")
def results_never(data):
    return WooldridgeDiD(control_group="never_treated").fit(data, **FIT)


class TestLinear:
    def test_estimate_covers_truth(self, results):
        assert results.att == pytest.approx(1.9817, abs=5e-4)
        assert results.se == pytest.approx(0.0371, abs=2e-4)
        assert results.conf_int[0] <= TRUE_ATT <= results.conf_int[1]
        assert_quotes_in_rendered(
            NB, ["Estimate: 1.982  (SE 0.0371)", "95% CI:   [1.909, 2.055]"], surface="output"
        )

    def test_control_group_changes_the_cells(self, results, results_never):
        default_cells = set(results.group_time_effects)
        assert len(default_cells) == 15 and not any(t < g for g, t in default_cells)
        placebo = {
            (int(g), int(t)): v for (g, t), v in results_never.group_time_effects.items() if t < g
        }
        assert len(placebo) == 12
        for cohort in (3, 5, 7):
            estimated = {t for (g, t) in results_never.group_time_effects if g == cohort}
            assert sorted(set(range(10)) - estimated) == [cohort - 1]
        # the one nominally significant placebo, and cohort 7's common sign
        assert placebo[(7, 2)]["p_value"] == pytest.approx(0.018, abs=2e-3)
        assert sum(v["p_value"] < 0.05 for v in placebo.values()) == 1
        assert all(v["att"] < 0 for (g, _), v in placebo.items() if g == 7)

    def test_event_study_and_cohorts(self, results, results_never):
        results_never.aggregate("event_study")
        es = results_never.to_dataframe(level="event_study").set_index("event_time")
        assert -1 not in es.index  # the omitted reference period is not a row
        assert list(es.index) == [-7, -6, -5, -4, -3, -2, 0, 1, 2, 3, 4, 5, 6]
        assert ((es.loc[-7:-2, "att"] / es.loc[-7:-2, "se"]).abs() < 2).all()
        assert (es.loc[0:6, "att"] - TRUE_ATT).abs().max() < 0.11
        results.aggregate("group")
        groups = results.to_dataframe(level="group").set_index("cohort")
        np.testing.assert_allclose(groups["att"].to_numpy(), [2.005, 1.999, 1.908], atol=1e-3)
        assert {"conf_int_lo", "conf_int_hi"} <= set(groups.columns)
        assert ((groups["conf_int_lo"] <= TRUE_ATT) & (TRUE_ATT <= groups["conf_int_hi"])).all()

    def test_event_study_matches_callaway_santanna_on_a_shared_reference(self, results_never, data):
        """Leads and lags: with never-treated controls ETWFE measures every period
        against the last one before treatment, so it matches Callaway-Sant'Anna
        at every event time only under ``base_period="universal"``. Under the
        default (each lead against the period before it) the leads differ."""
        results_never.aggregate("event_study")
        etwfe = results_never.to_dataframe(level="event_study").set_index("event_time")["att"]

        def cs_event_study(**kwargs):
            frame = (
                CallawaySantAnna(**kwargs).fit(data, **FIT).aggregate("event_study").to_dataframe()
            )
            return frame[~frame["is_reference"]].set_index("event_time")["att"]

        universal = cs_event_study(base_period="universal")
        assert list(etwfe.index) == list(universal.index)
        assert (etwfe.index < 0).sum() >= 2 and (etwfe.index >= 0).sum() >= 2
        np.testing.assert_allclose(etwfe.to_numpy(), universal.to_numpy(), atol=1e-8)

        varying = cs_event_study()
        shared = etwfe.index.intersection(varying.index)
        lags, leads = shared[shared >= 0], shared[shared < 0]
        np.testing.assert_allclose(etwfe[lags].to_numpy(), varying[lags].to_numpy(), atol=1e-8)
        assert (etwfe[leads] - varying[leads]).abs().max() > 0.05

        assert_quotes_in_rendered(
            NB,
            [
                "Event times compared: -7 to 6",
                "Largest difference between the two estimates at any event time: 0.0000",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert 'With `base_period="universal"` it uses ETWFE\'s reference' in md
        assert "agreement period by period" not in md

    def test_matches_callaway_santanna_with_the_same_controls(self, results, results_never, data):
        callaway = CallawaySantAnna().fit(data, **FIT)
        assert results_never.att == pytest.approx(callaway.att, abs=1e-4)
        assert results_never.se == pytest.approx(0.0527, abs=2e-4)
        assert callaway.se == pytest.approx(0.0528, abs=2e-4)
        assert results.se / callaway.se == pytest.approx(0.70, abs=0.01)
        # the default fit is the imputation estimate
        assert results.att == pytest.approx(ImputationDiD().fit(data, **FIT).att, abs=1e-8)
        assert_quotes_in_rendered(
            NB,
            [
                "ETWFE, never-treated                 1.9754   0.0527",
                "Callaway-Sant'Anna (default)         1.9754   0.0528",
            ],
            surface="output",
        )


class TestNonlinear:
    def test_poisson_recovers_the_effect_in_levels(self, data):
        count_data = data.copy()
        count_data["emp"] = np.exp(count_data["outcome"] / 4 + 3)
        treated = count_data["treated"] == 1
        untreated_emp = np.exp((count_data["outcome"] - count_data["true_effect"]) / 4 + 3)
        truth = (count_data.loc[treated, "emp"] - untreated_emp[treated]).mean()
        assert truth == pytest.approx(223.1, abs=0.1)
        fit = WooldridgeDiD(method="poisson").fit(
            count_data, outcome="emp", unit="unit", time="period", first_treat="first_treat"
        )
        assert fit.att == pytest.approx(222.6, abs=0.2)
        assert fit.conf_int[0] <= truth <= fit.conf_int[1]
        fit.aggregate("event_study")
        es = fit.to_dataframe(level="event_study").set_index("event_time")
        assert es.loc[6, "att"] > es.loc[0, "att"] + 40  # constant in logs, growing in levels

    def test_logit_recovers_the_effect_in_probability(self, data):
        binary = data.copy()
        threshold = binary.loc[binary["period"] == 0, "outcome"].median()
        binary["high"] = (binary["outcome"] > threshold).astype(int)
        treated = binary["treated"] == 1
        untreated = ((binary["outcome"] - binary["true_effect"]) > threshold).astype(int)
        truth = (binary.loc[treated, "high"] - untreated[treated]).mean()
        assert truth == pytest.approx(0.231, abs=1e-3)
        fit = WooldridgeDiD(method="logit").fit(
            binary, outcome="high", unit="unit", time="period", first_treat="first_treat"
        )
        assert fit.att == pytest.approx(0.246, abs=2e-3)
        assert fit.conf_int[0] <= truth <= fit.conf_int[1]


class TestAllEventuallyTreated:
    def test_last_cohort_is_the_reference(self):
        rng = np.random.default_rng(7)
        unit = np.repeat(np.arange(200), 9)
        time = np.tile(np.arange(1, 10), 200)
        cohort = np.where(unit < 70, 3, np.where(unit < 140, 5, 8))
        panel = pd.DataFrame(
            {
                "unit": unit,
                "time": time,
                "cohort": cohort,
                "y": rng.standard_normal(len(unit)) + 1.5 * (time >= cohort),
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = WooldridgeDiD(control_group="not_yet_treated").fit(
                panel, outcome="y", unit="unit", time="time", first_treat="cohort"
            )
        messages = [str(w.message) for w in caught]
        assert any("Dropped 400 of 1800 observations" in m for m in messages)
        assert any("Cohort(s) 8 have NO estimated cells" in m for m in messages)
        assert sorted(int(g) for g in fit.groups) == [3, 5]
        assert sorted({int(t) for _, t in fit.group_time_effects}) == [3, 4, 5, 6, 7]
        assert fit.att == pytest.approx(1.550, abs=2e-3)
        assert fit.conf_int[0] <= 1.5 <= fit.conf_int[1]
        with pytest.raises(ValueError):
            WooldridgeDiD(unsupported_period_action="error").fit(
                panel, outcome="y", unit="unit", time="time", first_treat="cohort"
            )


class TestMpdta:
    def test_overall_effect_on_log_employment(self):
        mpdta = load_mpdta()
        if mpdta.attrs.get("source") == "synthetic_fallback":
            pytest.skip("mpdta not available offline; canonical numbers cannot be re-derived")
        fit = WooldridgeDiD().fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        assert fit.att == pytest.approx(-0.0477, abs=5e-4)
        assert fit.se == pytest.approx(0.0133, abs=3e-4)
        fit.aggregate("event_study")
        es = fit.to_dataframe(level="event_study").set_index("event_time")
        np.testing.assert_allclose(
            es["att"].to_numpy(), [-0.0311, -0.0522, -0.1361, -0.1047], atol=5e-4
        )
        mpdta = mpdta.copy()
        mpdta["emp"] = np.exp(mpdta["lemp"])
        poisson = WooldridgeDiD(method="poisson").fit(
            mpdta, outcome="emp", unit="countyreal", time="year", first_treat="first_treat"
        )
        assert poisson.att == pytest.approx(-28.9, abs=0.3)
        assert poisson.conf_int[0] < 0 < poisson.conf_int[1]


class TestNotebookSurface:
    def test_prose(self):
        md = notebook_markdown(NB)
        for needle in (
            "are different regressions",
            "Read placebo cells for their size\nand pattern, not as a pass/fail test",
            "It is the control group",
            "check the control groups before\nanything else",
            "parallel trends is assumed on the index\nscale",
            "analytical from the influence function; multiplier bootstrap optional",
            "Rios-Avila, F. (2021).",
        ):
            assert needle in md, needle
        for stale in (
            "mirrors ETWFE's idiom",
            "Tracked as follow-up work",
            "Friosavila",
            "the only valid",
        ):
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
            'WooldridgeDiD(control_group="never_treated")',
            'es = results_never.aggregate("event_study")',
            "plot_event_study(es,",
            'results_never.to_dataframe(level="event_study")',
        ):
            assert needle in code, needle
        assert "event_study_effects" not in code  # no raw event-study dicts


def test_never_treated_advice_is_scoped_to_the_linear_model():
    """Poisson and logit emit post-treatment cells only under either control
    group, so the prose must not offer ``"never_treated"`` as their remedy for
    suspect pre-treatment periods."""
    md = notebook_markdown(NB)
    assert 'with a linear\noutcome use `"never_treated"`, or Callaway-Sant\'Anna' in md
    assert "A Poisson or logit fit gives coefficients to post-treatment cells only" in md
    # nonlinear fits use cohort and period effects (jwdid's i.gvar i.tvar), not unit effects
    assert (
        "cohort and period effects in place of the unit and period effects of the linear model"
        in md
    )
    assert 'The linear model (`method="ols"`) includes unit and period fixed effects.' in md
    # the reference is the latest SUPPORTED pre-period, not merely the latest observed one
    assert "the\n  reference is the latest pre-treatment period in which both are observed" in md
    panel = generate_staggered_data(n_units=120, n_periods=8, seed=3, dynamic_effects=False)
    panel["count"] = np.exp(panel["outcome"] / 4 + 3)
    fit = WooldridgeDiD(method="poisson", control_group="never_treated").fit(
        panel, outcome="count", unit="unit", time="period", first_treat="first_treat"
    )
    assert all(t >= g for (g, t) in fit.group_time_effects)


def test_reference_period_on_an_unbalanced_panel():
    """The reference is the latest pre-treatment period in which the cohort and
    the never-treated units are both observed. The fit warns either way, but
    only names the move when the never-treated units lack the period."""
    md = notebook_markdown(NB)
    assert "it names the new reference when the never-treated units lack the\n  period" in md
    panel = generate_staggered_data(
        n_units=150,
        n_periods=9,
        cohort_periods=[4, 6],
        never_treated_frac=0.4,
        treatment_effect=2.0,
        dynamic_effects=False,
        seed=5,
    )
    gaps = {
        "cohort": (panel["first_treat"] == 6) & (panel["period"] == 5),
        "never_treated": (panel["first_treat"] == 0) & (panel["period"] == 5),
    }
    for missing, drop in gaps.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = WooldridgeDiD(control_group="never_treated").fit(panel[~drop], **FIT)
        messages = " ".join(str(w.message) for w in caught)
        # period 4 is the reference in both cases: no (6, 4) or (6, 5) cell
        assert sorted(int(t) for g, t in fit.group_time_effects if g == 6) == [0, 1, 2, 3, 6, 7, 8]
        if missing == "never_treated":
            assert "reference period moved from 5 to 4" in messages
        else:
            assert "reference period moved" not in messages
            assert "no observations: (6, 5)" in messages
