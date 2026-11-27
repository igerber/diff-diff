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

import hashlib
import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    ImputationDiD,
    PreTrendsPower,
    WooldridgeDiD,
    compute_honest_did,
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

    def test_diagnostics_take_a_callaway_santanna_fit_not_this_one(self, results_never, data):
        """Pre-Trends Power and Honest DiD reject an ETWFE event study and accept the
        universal-base Callaway-Sant'Anna fit the notebook names."""
        own = results_never.aggregate("event_study")
        rejected = "Unsupported results type: .*WooldridgeDiDResults"
        with pytest.raises(TypeError, match=rejected):
            PreTrendsPower().fit(own)
        with pytest.raises(TypeError, match=rejected):
            compute_honest_did(own, method="relative_magnitude", M=1.0)
        supported = (
            CallawaySantAnna(base_period="universal").fit(data, **FIT).aggregate("event_study")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            power = PreTrendsPower().fit(supported)
            bounds = compute_honest_did(supported, method="relative_magnitude", M=1.0)
        assert np.isfinite(power.mdv) and power.mdv > 0
        assert np.isfinite(bounds.lb) and np.isfinite(bounds.ub) and bounds.lb < bounds.ub

    def test_equality_with_callaway_santanna_needs_a_balanced_panel(self, data):
        """The regression uses every observation; each Callaway-Sant'Anna
        comparison uses the units observed in both of its periods. Dropping one
        post-treatment period for ten units of one cohort separates them."""
        units = data.loc[data["first_treat"] == 5, "unit"].unique()[:10]
        unbalanced = data[~(data["unit"].isin(units) & (data["period"] == 6))]
        etwfe_fit = WooldridgeDiD(control_group="never_treated").fit(unbalanced, **FIT)
        etwfe_fit.aggregate("event_study")
        etwfe = etwfe_fit.to_dataframe(level="event_study").set_index("event_time")["att"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unbalanced panel", category=UserWarning)
            frame = (
                CallawaySantAnna(base_period="universal")
                .fit(unbalanced, **FIT)
                .aggregate("event_study")
                .to_dataframe()
            )
        callaway = frame[~frame["is_reference"]].set_index("event_time")["att"]
        assert list(etwfe.index) == list(callaway.index)
        gap = (etwfe - callaway).abs()
        assert gap.idxmax() == 1
        assert gap.max() == pytest.approx(0.0036, abs=5e-4)
        assert gap.drop(1).max() < 1e-8
        md = notebook_markdown(NB)
        assert "on a balanced\npanel with a linear outcome and no covariates" in md
        assert "moves the estimates at event time\n1 apart by 0.004" in md

    def test_reference_period_reading_needs_a_balanced_panel_without_covariates(self):
        """On a balanced panel without covariates an early shock stays out of the
        post-treatment coefficient, and a time-constant covariate keeps it out.
        Gaps, or a time-varying covariate, can let early outcomes reach it (not
        every gap does), which is all the notebook claims."""

        def post(missing=(), early=None, late=None, covariate=None, n_treated=2, n_control=3):
            early, late = early or {}, late or {}
            rows = []
            units = [(f"T{i}", 3) for i in range(n_treated)] + [
                (f"C{i}", 0) for i in range(n_control)
            ]
            for index, (unit, cohort) in enumerate(units):
                for time in (1, 2, 3):
                    if (unit, time) in missing:
                        continue
                    y = 2.0 if (cohort == 3 and time == 3) else 0.0
                    if time == 1:
                        y += early.get(unit, 0.0)
                    if time == 3:
                        y += late.get(unit, 0.0)
                    varying = float(np.sin(1.7 * index + 2.3 * time))
                    constant = float(np.sin(1.7 * index))
                    rows.append(
                        {
                            "unit": unit,
                            "time": time,
                            "cohort": cohort,
                            "y": y,
                            "varying": varying,
                            "constant": constant,
                        }
                    )
            keys = dict(outcome="y", unit="unit", time="time", first_treat="cohort")
            if covariate == "time_varying":
                keys["xtvar"] = ["varying"]
            elif covariate == "time_constant":
                keys["exovar"] = ["constant"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = WooldridgeDiD(control_group="never_treated").fit(pd.DataFrame(rows), **keys)
            return fit.group_time_effects[(3, 3)]["att"]

        # balanced, no covariates: early shocks leave the effect at 2
        assert post() == pytest.approx(2.0, abs=1e-8)
        assert post(early={"T0": 10.0}) == pytest.approx(2.0, abs=1e-8)
        assert post(early={"T0": 10.0, "T1": 10.0}) == pytest.approx(2.0, abs=1e-8)
        # a gap, in the reference period or the target period, lets an early shock
        # reach the effect; a gap in the shocked period itself does not
        assert post(missing={("T1", 2)}) == pytest.approx(2.0, abs=1e-8)
        assert post(missing={("T1", 2)}, early={"T1": 10.0}) == pytest.approx(-0.5, abs=1e-8)
        assert post(missing={("T1", 3)}) == pytest.approx(2.0, abs=1e-8)
        assert post(missing={("T1", 3)}, early={"T0": 10.0}) == pytest.approx(-0.5, abs=1e-8)
        assert post(missing={("T1", 1)}, early={"T0": 10.0}) == pytest.approx(2.0, abs=1e-8)
        # a time-varying covariate on a balanced panel does it too
        wide = dict(covariate="time_varying", n_treated=5, n_control=7)
        assert post(**wide) == pytest.approx(2.0, abs=1e-8)
        assert abs(post(early={"T0": 10.0}, **wide) - 2.0) > 0.1
        # a time-constant covariate does not: the same shock leaves the effect at 2
        fixed = dict(covariate="time_constant", n_treated=5, n_control=7)
        assert post(**fixed) == pytest.approx(2.0, abs=1e-8)
        assert post(early={"T0": 10.0}, **fixed) == pytest.approx(2.0, abs=1e-8)
        # missingness related to the idiosyncratic shocks biases the fit: offsetting
        # target-period shocks average out until the unit with the low one is dropped
        shocks = {"T0": 10.0, "T1": -10.0}
        assert post(late=shocks) == pytest.approx(2.0, abs=1e-8)
        assert post(late=shocks, missing={("T1", 3)}) == pytest.approx(12.0, abs=1e-8)
        md = notebook_markdown(NB)
        for needle in (
            "With no covariates on a balanced panel, the case this tutorial uses, each coefficient\n"
            "  is then a difference-in-differences against that period",
            "Gaps in the panel, or time-varying covariates, can\n  change that",
            "outcomes in other periods, early ones included, can then move the post-treatment\n"
            "  coefficients",
            "to be unrelated to the idiosyncratic shocks to the outcome",
            "a\n  cohort treated in period 6 with no rows in period 5 is measured against period 4",
            'For the linear model (`method="ols"`) with `"never_treated"`, a balanced panel and\n'
            "no covariates, only those from the reference period on.",
            "for Poisson and logit under either, any of\nthem",
            'With gaps in the panel or time-varying covariates `"never_treated"`\n'
            "may no longer confine the assumption to the reference period",
        ):
            assert needle in md, needle
        for stale in (
            "observations being unrelated to the outcome",
            "cohort's placebo coefficient, not into the post-treatment ones",
            "protects against early shifts a cohort shares",
            "use Callaway-Sant'Anna there",
            "With\n  missing observations or with covariates",
        ):
            assert stale not in md, stale

    def test_comparison_cell_refuses_unmatched_event_times(self):
        """The notebook's own alignment code, run on two event studies of which one
        lacks a horizon: it must raise, not report the gap over the rows that match."""
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        source = next(
            "".join(c["source"])
            for c in cells
            if c["cell_type"] == "code" and "both = pd.concat(" in "".join(c["source"])
        )
        start = source.index("both = pd.concat(")
        stop = source.index("\n", source.index("gap = ")) + 1
        snippet = source[start:stop]

        def run(etwfe_times, callaway_values):
            scope = {
                "pd": pd,
                "np": np,
                "etwfe_es": pd.DataFrame({"event_time": etwfe_times, "att": 1.0}),
                "callaway_es": pd.DataFrame(
                    {"event_time": list(callaway_values), "att": list(callaway_values.values())}
                ),
            }
            exec(snippet, scope)  # noqa: S102 - the notebook's own cell source
            return scope["gap"]

        assert run([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.5}) == pytest.approx(0.5)
        with pytest.raises(ValueError, match=r"\[2\]"):
            run([0, 1, 2], {0: 1.0, 1: 1.0})
        with pytest.raises(ValueError, match=r"\[1\]"):
            run([0, 1, 2], {0: 1.0, 1: float("nan"), 2: 1.0})

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


@pytest.fixture(scope="module")
def mpdta(tmp_path_factory):
    """The canonical panel, loaded once for the module. A full checkout carries
    the checksum-matching source file, so the loader is fed those bytes and the
    numbers are re-derived offline; without it the loader is asked once, and a
    fallback skips. A copy whose bytes do not match the pin (a checkout that
    rewrote its line endings) is treated as absent. Tests copy the frame before
    changing it."""
    import diff_diff.datasets as datasets_mod

    fixture = Path(__file__).resolve().parents[1] / "benchmarks/data/real/mpdta.csv"
    payload = fixture.read_bytes() if fixture.exists() else None
    if (
        payload is not None
        and hashlib.sha256(payload).hexdigest() == datasets_mod._MPDTA_SOURCE_SHA256
    ):
        response = MagicMock()
        response.__enter__ = lambda self: self
        response.__exit__ = lambda self, *a: False
        response.read.return_value = payload
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(datasets_mod, "_CACHE_DIR", tmp_path_factory.mktemp("mpdta"))
            with patch("diff_diff.datasets.urlopen", return_value=response):
                frame = load_mpdta(force_download=True)
        # with the source file in the checkout a fallback is a failure, not a skip
        assert frame.attrs.get("source") == "callaway_santanna_mpdta"
        return frame
    frame = load_mpdta()
    if frame.attrs.get("source") == "synthetic_fallback":
        pytest.skip("mpdta not available offline; canonical numbers cannot be re-derived")
    return frame


class TestMpdta:
    def test_overall_effect_on_log_employment(self, mpdta):
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
        # the notebook prints where the data came from, so a fallback run is visible
        assert mpdta.attrs["source"] == "callaway_santanna_mpdta"
        assert_quotes_in_rendered(NB, ["Provenance: callaway_santanna_mpdta"], surface="output")

    def test_state_clustering_widens_the_interval(self, mpdta):
        """Minimum wages are set by states. The county-clustered fit is the `jwdid`
        benchmark; clustering by state gives the same estimate and a larger SE."""
        keys = dict(outcome="lemp", unit="countyreal", time="year", first_treat="first_treat")
        mpdta = mpdta.copy()
        mpdta["state"] = mpdta["countyreal"] // 1000
        assert mpdta["state"].nunique() == 29
        assert mpdta.groupby("state")["first_treat"].nunique().max() == 1  # set by state
        assert mpdta.groupby("first_treat")["state"].nunique().to_dict() == {
            0: 16,
            2004: 1,
            2006: 3,
            2007: 9,
        }
        by_county = WooldridgeDiD().fit(mpdta, **keys)
        by_state = WooldridgeDiD(cluster="state").fit(mpdta, **keys)
        assert by_state.att == pytest.approx(by_county.att, abs=1e-12)
        assert by_state.se == pytest.approx(0.0190, abs=3e-4)
        assert by_state.se > 1.3 * by_county.se
        assert by_state.conf_int[0] == pytest.approx(-0.0867, abs=5e-4)
        assert by_state.conf_int[1] == pytest.approx(-0.0087, abs=5e-4)
        assert_quotes_in_rendered(
            NB,
            [
                "county          -0.0477   0.0133",
                "state           -0.0477   0.0190   [-0.0867, -0.0087]",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert "the state is the level to cluster at" in md
        assert "the 2004 cohort is a single state" in md

    def test_late_horizons_are_the_2004_cohort_alone(self, mpdta):
        """The data end in 2007, so the event-study horizons average different
        cohorts; the notebook says which, and that the late ones are one state."""
        fit = WooldridgeDiD().fit(
            mpdta, outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )
        by_horizon = {}
        for g, t in fit.group_time_effects:
            by_horizon.setdefault(int(t - g), set()).add(int(g))
        assert by_horizon == {
            0: {2004, 2006, 2007},
            1: {2004, 2006},
            2: {2004},
            3: {2004},
        }
        first = mpdta[mpdta["first_treat"] == 2004]
        assert first["countyreal"].nunique() == 20
        assert (first["countyreal"] // 1000).nunique() == 1
        md = notebook_markdown(NB)
        assert (
            "event times 2 and 3 only the 2004 cohort, which is 20\ncounties in a single state"
            in md
        )
        assert "Do not read that growth as the effect of time alone." in md

    def test_notebook_cells_print_the_state_clustered_row(self, mpdta, capsys):
        """The notebook's own load-and-fit cell and state-clustering cell, run on the
        canonical panel: the live output must carry the rows the prose quotes."""
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        load_cell = next(s for s in code if "canonical = mpdta.attrs.get" in s)
        state_cell = next(s for s in code if "mpdta_by_state = " in s)
        scope = {"load_mpdta": lambda: mpdta.copy(), "WooldridgeDiD": WooldridgeDiD, "np": np}
        exec(load_cell, scope)  # noqa: S102 - the notebook's own cell source
        capsys.readouterr()
        exec(state_cell, scope)  # noqa: S102
        printed = capsys.readouterr().out
        assert "States: 29  (by cohort: {0: 16, 2004: 1, 2006: 3, 2007: 9})" in printed
        assert "county          -0.0477   0.0133   [-0.0738, -0.0216]" in printed
        assert "state           -0.0477   0.0190   [-0.0867, -0.0087]" in printed
        assert scope["mpdta_by_state"].att == pytest.approx(scope["mpdta_fit"].att, abs=1e-12)

    def test_section_skips_its_fits_on_the_synthetic_fallback(self, capsys):
        """The section's own cells, run with a loader that returns the simulated
        panel: they must say so and fit nothing, because the prose below them
        discusses the canonical numbers."""
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        section = [
            "".join(c["source"])
            for c in cells
            if c["cell_type"] == "code"
            and "mpdta" in "".join(c["source"])
            and "import" not in "".join(c["source"])
        ]
        assert len(section) == 4

        def fallback_loader():
            frame = pd.DataFrame({"countyreal": [1], "year": [2003], "lemp": [0.0]})
            frame.attrs["source"] = "synthetic_fallback"
            return frame

        def no_fit(*args, **kwargs):
            raise AssertionError("a fit ran on the synthetic fallback")

        scope = {"load_mpdta": fallback_loader, "WooldridgeDiD": no_fit, "np": np}
        for source in section:
            exec(source, scope)  # noqa: S102 - the notebook's own cell source
        printed = capsys.readouterr().out
        assert "Provenance: synthetic_fallback" in printed
        assert "section are skipped" in printed
        assert not {"mpdta_fit", "mpdta_poisson", "mpdta_by_state"} & set(scope)


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
            # parity with Callaway-Sant'Anna is stated with its conditions everywhere
            "On this balanced panel without covariates, ETWFE with never-treated controls reproduces",
            "That diagnostic and Honest DiD do not take an ETWFE event study",
            "it reproduces Callaway-Sant'Anna when the panel is\n  balanced",
        ):
            assert needle in md, needle
        for stale in (
            "mirrors ETWFE's idiom",
            "Tracked as follow-up work",
            "Friosavila",
            "the only valid",
            "it reproduces Callaway-Sant'Anna on a linear outcome",
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
    assert (
        'with a linear\noutcome, a balanced panel and no covariates, use `"never_treated"`, or\n'
        "Callaway-Sant'Anna" in md
    )
    assert 'The\n`"never_treated"` remedy does not carry over to\nPoisson and logit' in md
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
    # a period no unit is observed in is not on the panel's time grid: period 4 is
    # the reference again, and nothing is reported
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = WooldridgeDiD(control_group="never_treated").fit(panel[panel["period"] != 5], **FIT)
    assert sorted(int(t) for g, t in fit.group_time_effects if g == 6) == [0, 1, 2, 3, 6, 7, 8]
    assert not [w for w in caught if "(6, 5)" in str(w.message) or "moved" in str(w.message)]
    assert "the fit moves past it without\n  a warning" in md
    assert "provided it and the never-treated units both have rows in period 4" in md
