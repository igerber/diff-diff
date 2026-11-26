"""Placebo event studies in ``diagnose_design`` and planned pre-trend power.

``diagnose_design(..., event_study=True)`` records, per candidate, the mean and
the covariance across the no-effect placebo draws of its event-study
coefficients. ``PreTrendsPower`` reads that record to plan pre-trend power for
a design that has not been run.

Numerical note: the default NIS pre-test integrates a multivariate normal with
a randomized routine, so two calls agree only to about 1e-5. Exact checks are
made on the Wald form and on deterministic inputs; NIS values are compared at
``NIS_ATOL``.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    BusinessReport,
    CallawaySantAnna,
    DiagnosticReport,
    HonestDiD,
    ImputationDiD,
    PreTrendsPower,
    StackedDiD,
    TwoWayFixedEffects,
    compute_mdv,
    compute_pretrends_power,
    diagnose_design,
)
from diff_diff import design as design_module
from diff_diff.results_base import EventStudyResults
from diff_diff.utils import safe_inference

NIS_ATOL = 1e-3


def _panel(n_units=40, periods=range(1, 9), seed=0, rho=0.5):
    """An untreated panel with unit effects, a common trend and AR(1) errors."""
    rng = np.random.default_rng(seed)
    periods = list(periods)
    rows = []
    for i in range(n_units):
        unit_effect = rng.normal()
        error = rng.normal()
        for t in periods:
            error = rho * error + rng.normal()
            rows.append((i, t, unit_effect + 0.1 * t + error))
    return pd.DataFrame(rows, columns=["id", "t", "y"])


def _diagnose(panel, estimators, cohorts=(6,), n_simulations=40, seed=1, quiet=True, **kwargs):
    kwargs.setdefault("event_study", True)
    if "n_treated" not in kwargs:
        kwargs.setdefault("treatment_fraction", 0.25)
    call = dict(
        outcome="y",
        unit="id",
        time="t",
        estimators=estimators,
        cohort_periods=list(cohorts),
        n_simulations=n_simulations,
        seed=seed,
        progress=False,
        **kwargs,
    )
    if not quiet:
        return diagnose_design(panel, **call)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diagnose_design(panel, **call)


def _messages(recorded):
    return [str(w.message) for w in recorded]


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def cs_diagnosis(panel):
    return _diagnose(panel, [CallawaySantAnna(base_period="universal")])


@pytest.fixture(scope="module")
def cs_surface(cs_diagnosis):
    return cs_diagnosis.event_study_results["CallawaySantAnna"]


def _independent_surface(n_pre=4, se=0.5, inflate=(), n_draws=4000, seed=0):
    """A hand-built placebo surface with independent pre-period coefficients.

    Its per-draw record holds standard normal ratios, so the pre-trend test
    over any ``K`` of them rejects at ``1 - 0.95^K`` as the formula assumes.
    Columns listed in ``inflate`` (positions among the pre-periods, earliest
    first) get ratios 1.5 times too large: tests that include them reject
    more often than assumed.
    """
    event_time = np.arange(-n_pre - 1, 2)
    is_reference = event_time == -1
    k = int((~is_reference).sum())
    ses = np.where(is_reference, np.nan, se)
    surface = EventStudyResults(
        event_time=event_time,
        att=np.zeros(len(event_time)),
        se=ses,
        t_stat=np.where(is_reference, np.nan, 0.0),
        p_value=np.where(is_reference, np.nan, 1.0),
        conf_int_lower=np.where(is_reference, np.nan, -1.96 * se),
        conf_int_upper=np.where(is_reference, np.nan, 1.96 * se),
        is_reference=is_reference,
        n=np.full(len(event_time), np.nan),
        vcov=np.eye(k) * se**2,
        vcov_index=event_time[~is_reference],
        source="CallawaySantAnnaResults",
        base_period="universal",
        anticipation=0,
        reference_event_times=(-1,),
        estimand=design_module._PLACEBO_ESTIMAND,
    )
    ratios = np.abs(np.random.default_rng(seed).normal(size=(n_draws, n_pre)))
    for column in inflate:
        ratios[:, column] *= 1.5
    labels = event_time[event_time < -1].tolist()
    setattr(surface, design_module._PRETEST_RECORD, {"labels": labels, "abs_t": ratios})
    return surface


# ---------------------------------------------------------------------------
# The recorded surface
# ---------------------------------------------------------------------------


class TestSurface:
    """The surface is the mean and sample covariance of the per-draw event studies."""

    @pytest.mark.parametrize(
        "estimator, cohorts",
        [
            (CallawaySantAnna(base_period="universal"), (6,)),
            (CallawaySantAnna(), (6,)),
            (CallawaySantAnna(base_period="universal"), (5, 7)),
            (StackedDiD(kappa_pre=3, kappa_post=1), (6,)),
        ],
        ids=["cs-universal", "cs-varying", "cs-staggered", "stacked"],
    )
    def test_equals_hand_loop(self, panel, estimator, cohorts):
        n_simulations, seed = 40, 7
        diagnosis = _diagnose(panel, [estimator], cohorts, n_simulations, seed)
        label = type(estimator).__name__
        surface = diagnosis.event_study_results[label]
        assert diagnosis._event_study_checks[label]["n_excluded"] == 0

        generator = design_module._make_placebo_generator(
            panel, "y", "id", "t", 10, list(cohorts), True
        )
        fit_kwargs = design_module._fit_kwargs(estimator, "y", "id", "t", {})
        draws, reported, first = [], [], None
        for replicate_seed in design_module._replicate_seeds(seed, n_simulations):
            frame = generator(treatment_effect=0.0, seed=replicate_seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = design_module._clone(estimator).fit(frame, **fit_kwargs)
                one = fitted.aggregate("event_study")
            first = one if first is None else first
            np.testing.assert_array_equal(one.event_time, first.event_time)
            draws.append(one.att[~one.is_reference])
            reported.append(one.se[~one.is_reference])
        draws = np.vstack(draws)

        np.testing.assert_array_equal(surface.event_time, first.event_time)
        np.testing.assert_array_equal(surface.is_reference, first.is_reference)
        estimated = ~surface.is_reference
        np.testing.assert_allclose(surface.att[estimated], draws.mean(axis=0), rtol=1e-12)
        expected_vcov = np.atleast_2d(np.cov(draws, rowvar=False, ddof=1))
        np.testing.assert_allclose(surface.vcov, expected_vcov, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            surface.se[estimated], np.sqrt(np.diag(expected_vcov)), rtol=1e-12
        )
        np.testing.assert_array_equal(surface.vcov_index, first.event_time[~first.is_reference])
        assert surface.source == first.source
        assert surface.base_period == first.base_period
        assert surface.event_time_convention == first.event_time_convention
        expected_refs = (
            None
            if first.reference_event_times is None
            else tuple(int(v) for v in first.reference_event_times)
        )
        assert surface.reference_event_times == expected_refs
        assert surface.estimand == "placebo"
        assert surface.anticipation == 0
        assert surface.n_kind is None
        assert np.all(np.isnan(surface.n))
        assert surface.alpha == diagnosis.alpha

        # The calibration record: mean reported SE over the placebo spread.
        pre = first.event_time[~first.is_reference] < 0
        expected_ratio = (
            np.vstack(reported).mean(axis=0)[pre] / np.sqrt(np.diag(expected_vcov))[pre]
        )
        ratios = diagnosis._event_study_checks[label]["se_ratio"]
        np.testing.assert_allclose(list(ratios.values()), expected_ratio, rtol=1e-12)
        assert list(ratios) == first.event_time[~first.is_reference][pre].tolist()

        # The pre-trend test counted in the draws, with each draw's own SEs.
        reported = np.vstack(reported)
        rejected = np.any(np.abs(draws[:, pre] / reported[:, pre]) > 1.959963984540054, axis=1)
        checks = diagnosis._event_study_checks[label]
        assert checks["pretest_rejection_rate"] == pytest.approx(rejected.mean(), abs=1e-12)
        pre_labels = first.event_time[~first.is_reference][pre].tolist()
        record = getattr(surface, design_module._PRETEST_RECORD)
        assert record["labels"] == pre_labels
        np.testing.assert_allclose(
            record["abs_t"], np.abs(draws[:, pre] / reported[:, pre]), rtol=1e-12
        )
        expected_rate = design_module._formula_pretest_rate(expected_vcov[np.ix_(pre, pre)], 0.05)
        assert checks["pretest_expected_rate"] == pytest.approx(expected_rate, abs=NIS_ATOL)
        assert checks["pretest_agrees"] is design_module._pretest_size_ok(
            checks["pretest_rejection_rate"], checks["pretest_expected_rate"], n_simulations
        )
        assert (checks["n_pre"], checks["n_post"]) == (int(pre.sum()), int((~pre).sum()))

    def test_inference_columns_follow_from_mean_and_spread(self, cs_surface):
        for i in range(len(cs_surface.event_time)):
            if cs_surface.is_reference[i]:
                assert cs_surface.att[i] == 0.0
                assert np.isnan(cs_surface.se[i]) and np.isnan(cs_surface.p_value[i])
                continue
            t_stat, p_value, conf_int = safe_inference(
                float(cs_surface.att[i]), float(cs_surface.se[i]), alpha=cs_surface.alpha
            )
            assert cs_surface.t_stat[i] == pytest.approx(t_stat)
            assert cs_surface.p_value[i] == pytest.approx(p_value)
            assert cs_surface.conf_int_lower[i] == pytest.approx(conf_int[0])
            assert cs_surface.conf_int_upper[i] == pytest.approx(conf_int[1])

    def test_summary_labels_the_column_placebo(self, cs_surface):
        text = cs_surface.summary()
        assert "estimand: placebo" in text
        assert "placebo" in text.splitlines()[4]
        assert set(cs_surface.to_dataframe()["estimand"]) == {"placebo"}

    def test_keyword_does_not_change_any_cell(self, panel):
        estimators = [CallawaySantAnna(base_period="universal"), StackedDiD(kappa_pre=3)]
        off = _diagnose(panel, estimators, seed=11, event_study=False, effect_sizes=[0.5])
        on = _diagnose(panel, estimators, seed=11, effect_sizes=[0.5])
        pd.testing.assert_frame_equal(off.to_dataframe(), on.to_dataframe())
        assert off.event_study is False and off.event_study_results == {}
        assert on.event_study is True and set(on.event_study_results) == {
            "CallawaySantAnna",
            "StackedDiD",
        }
        assert "Placebo event studies" not in off.summary()
        assert "Placebo event studies" in on.summary()
        assert off.to_dict()["event_study"] is False
        assert on.to_dict()["event_study"] is True
        stripped = {k: v for k, v in on.to_dict().items() if k != "event_study"}
        assert set(stripped) == set(off.to_dict()) - {"event_study"}

    def test_only_the_no_effect_cell_is_captured(self, panel):
        estimator = CallawaySantAnna(base_period="universal")
        plain = _diagnose(panel, [estimator])
        searched = _diagnose(panel, [estimator], effect_sizes=[0.4, 0.8], power=0.8)
        label = "CallawaySantAnna"
        assert searched.mde_results, "the MDE search must have run for this check to bind"
        checks = searched._event_study_checks[label]
        assert checks["n_draws"] == searched.power_results[label][0].n_simulations
        reference = plain._event_study_checks[label]
        for key in ("n_draws", "n_excluded", "se_ratio", "pretest_rejection_rate"):
            assert checks[key] == reference[key], key
        np.testing.assert_array_equal(
            searched.event_study_results[label].vcov, plain.event_study_results[label].vcov
        )
        np.testing.assert_array_equal(
            searched.event_study_results[label].att, plain.event_study_results[label].att
        )

    def test_summary_block(self, cs_diagnosis):
        text = cs_diagnosis.summary()
        assert "Placebo event studies (no-effect runs)" in text
        row = [line for line in text.splitlines() if line.startswith("CallawaySantAnna")][-1]
        checks = cs_diagnosis._event_study_checks["CallawaySantAnna"]
        verdict = "yes" if checks["pretest_agrees"] else "NO"
        assert str(checks["n_draws"]) in row and row.rstrip().endswith(verdict)
        assert f"{checks['pretest_rejection_rate']:.1%}" in row
        assert f"{checks['pretest_expected_rate']:.1%}" in row


# ---------------------------------------------------------------------------
# Reduction rules
# ---------------------------------------------------------------------------


def _record(event_times, references, values, ses=None, refs=(-1,), base="universal"):
    rows = tuple((int(e), bool(r)) for e, r in zip(event_times, references))
    provenance = ("CallawaySantAnnaResults", base, refs, "e0_first_treated")
    values = np.asarray(values, dtype=float)
    ses = np.ones_like(values) if ses is None else np.asarray(ses, dtype=float)
    return (rows, provenance), values, ses


class TestReduction:
    """``_placebo_event_study``: which draws are averaged and when nothing is reported."""

    def _draws(self, n, k=3, seed=0):
        return np.random.default_rng(seed).normal(size=(n, k))

    def test_provenance_comes_from_the_retained_draws(self):
        """A draw with another layout never lends its provenance, even when first."""
        odd = _record([-2, -1, 0], [False, True, False], [0.1, 0.2], refs=(-1,))
        main = [
            _record([-3, -2, -1, 0], [False, False, True, False], row, refs=(-2, -1))
            for row in self._draws(30)
        ]
        surface, checks, reason = design_module._placebo_event_study(
            CallawaySantAnna(), [odd] + main, 0.05
        )
        assert reason is None
        assert surface.reference_event_times == (-2, -1)
        assert surface.event_time.tolist() == [-3, -2, -1, 0]
        assert checks["n_draws"] == 30 and checks["n_excluded"] == 1

    def test_failed_aggregations_count_as_excluded(self):
        main = [_record([-2, -1, 0], [False, True, False], row) for row in self._draws(30, k=2)]
        surface, checks, _ = design_module._placebo_event_study(
            CallawaySantAnna(), main + [None, None], 0.05
        )
        assert surface is not None and checks["n_excluded"] == 2

    @pytest.mark.parametrize("records", [[], [None, None, None]], ids=["empty", "all-failed"])
    def test_nothing_captured(self, records):
        surface, checks, reason = design_module._placebo_event_study(
            CallawaySantAnna(), records, 0.05
        )
        assert surface is None and checks is None
        assert "no no-effect replicate produced an event study" in reason

    def test_more_than_ten_percent_excluded(self):
        main = [_record([-2, -1, 0], [False, True, False], row) for row in self._draws(30, k=2)]
        other = [_record([-1, 0], [True, False], [0.1]) for _ in range(5)]
        surface, _, reason = design_module._placebo_event_study(
            CallawaySantAnna(), main + other, 0.05
        )
        assert surface is None and "5 of 35" in reason

    def test_tied_layouts_are_not_reported(self):
        a = [_record([-2, -1, 0], [False, True, False], row) for row in self._draws(10, k=2)]
        b = [_record([-3, -1, 0], [False, True, False], row) for row in self._draws(10, k=2)]
        first = design_module._placebo_event_study(CallawaySantAnna(), a + b, 0.05)
        second = design_module._placebo_event_study(CallawaySantAnna(), b + a, 0.05)
        assert first[0] is None and second[0] is None
        assert first[2] == second[2]

    def test_too_few_draws(self):
        records = [
            _record([-3, -2, -1, 0], [False, False, True, False], row) for row in self._draws(6)
        ]
        surface, _, reason = design_module._placebo_event_study(CallawaySantAnna(), records, 0.05)
        assert surface is None
        assert "6 replicates are too few" in reason and "at least 7" in reason

    def test_no_pre_period(self):
        records = [_record([-1, 0, 1], [True, False, False], row) for row in self._draws(30, k=2)]
        for estimator, hint in ((StackedDiD(), True), (CallawaySantAnna(), False)):
            surface, _, reason = design_module._placebo_event_study(estimator, records, 0.05)
            assert surface is None and "no estimated pre-treatment period" in reason
            assert ("kappa_pre" in reason) is hint

    def test_rank_deficient_covariance(self):
        """Few distinct draws: positive variances, rank below the number of rows."""
        distinct = self._draws(3, k=5)
        distinct = np.vstack([distinct, -distinct])
        rows = distinct[np.random.default_rng(1).integers(0, 6, size=100)]
        assert np.all(rows.var(axis=0) > 0)
        assert np.linalg.matrix_rank(np.cov(rows, rowvar=False)) < 5
        records = [
            _record([-4, -3, -2, -1, 0, 1], [False, False, False, True, False, False], row)
            for row in rows
        ]
        surface, _, reason = design_module._placebo_event_study(CallawaySantAnna(), records, 0.05)
        assert surface is None and "rank-deficient" in reason

    def test_pretest_record_and_rates(self):
        """Counted rate = share of draws with any |coefficient / own SE| above z."""
        rng = np.random.default_rng(3)
        values = rng.normal(size=(200, 3))
        records = [
            _record([-3, -2, -1, 0], [False, False, True, False], v, ses=np.ones(3)) for v in values
        ]
        surface, checks, reason = design_module._placebo_event_study(
            CallawaySantAnna(), records, 0.05
        )
        assert surface is not None and reason is None
        z = 1.959963984540054
        assert checks["pretest_rejection_rate"] == pytest.approx(
            np.mean(np.any(np.abs(values[:, :2]) > z, axis=1)), abs=1e-12
        )
        assert checks["pretest_expected_rate"] == pytest.approx(
            design_module._formula_pretest_rate(surface.vcov[:2, :2], 0.05), abs=NIS_ATOL
        )
        # The same count for a subset, and at another level.
        counted, n_draws = design_module._placebo_pretest_rate(surface, [-2], 0.10)
        assert n_draws == 200
        assert counted == pytest.approx(np.mean(np.abs(values[:, 1]) > 1.6448536269514722))
        with pytest.raises(ValueError, match="not in this placebo event study"):
            design_module._placebo_pretest_rate(surface, [-9], 0.05)

    @pytest.mark.parametrize("scale, direction", [(0.6, "more"), (1.8, "less")])
    def test_full_set_verdict_in_both_directions(self, scale, direction):
        """Reported SEs too small (test over-rejects) or too large (under-rejects)."""
        values = np.random.default_rng(4).normal(size=(400, 3))
        records = [
            _record([-3, -2, -1, 0], [False, False, True, False], v, ses=np.full(3, scale))
            for v in values
        ]
        surface, checks, reason = design_module._placebo_event_study(
            CallawaySantAnna(), records, 0.05
        )
        assert surface is not None and reason is None
        assert checks["pretest_agrees"] is False
        gap = checks["pretest_rejection_rate"] - checks["pretest_expected_rate"]
        assert (gap > 0) is (direction == "more")

    def test_formula_pretest_rate(self):
        """Independent coefficients: 1 - 0.95^K; correlated ones reject less often."""
        for k in (1, 3, 6):
            rate = design_module._formula_pretest_rate(np.eye(k) * 0.3, 0.05)
            assert rate == pytest.approx(1.0 - 0.95**k, abs=NIS_ATOL)
        correlated = np.full((4, 4), 0.9) + 0.1 * np.eye(4)
        assert design_module._formula_pretest_rate(correlated, 0.05) < 1.0 - 0.95**4 - 0.03

    def test_pretest_check_tolerance(self):
        """Agreement within 1.96 Monte Carlo standard errors at the expected rate."""
        tolerance = 1.96 * np.sqrt(0.2 * 0.8 / 500)
        assert design_module._pretest_size_ok(0.2 + 0.99 * tolerance, 0.2, 500)
        assert not design_module._pretest_size_ok(0.2 + 1.01 * tolerance, 0.2, 500)
        assert design_module._pretest_size_ok(0.2 - 0.99 * tolerance, 0.2, 500)
        assert not design_module._pretest_size_ok(0.2 - 1.01 * tolerance, 0.2, 500)
        assert not design_module._pretest_size_ok(float("nan"), 0.2, 500)
        assert not design_module._pretest_size_ok(0.2, 0.2, 0)


class TestNotRecorded:
    """End-to-end: the cases in which a candidate gets no surface, and says why."""

    def _run(self, panel, estimators, **kwargs):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            diagnosis = _diagnose(panel, estimators, quiet=False, **kwargs)
        return diagnosis, _messages(recorded)

    def test_non_bool_keyword(self, panel):
        with pytest.raises(ValueError, match="event_study must be True or False"):
            _diagnose(panel, [CallawaySantAnna()], event_study="yes")

    def test_unsupported_class(self, panel):
        diagnosis, messages = self._run(
            panel, [ImputationDiD(), CallawaySantAnna(base_period="universal")]
        )
        assert list(diagnosis.event_study_results) == ["CallawaySantAnna"]
        assert any(
            "[ImputationDiD] no placebo event study is recorded" in m
            and "not an input PreTrendsPower accepts" in m
            for m in messages
        )

    def test_jackknife_candidate(self):
        diagnosis, messages = self._run(
            _panel(20), [CallawaySantAnna(vcov_type="jackknife")], n_simulations=6
        )
        assert diagnosis.event_study_results == {}
        assert any("cluster-jackknife" in m and "Bonferroni" in m for m in messages)

    def test_stacked_without_pre_window(self, panel):
        diagnosis, messages = self._run(panel, [StackedDiD()])
        assert diagnosis.event_study_results == {}
        assert any("kappa_pre" in m for m in messages)

    def test_too_few_replicates(self, panel):
        diagnosis, messages = self._run(
            panel, [CallawaySantAnna(base_period="universal")], n_simulations=8
        )
        assert diagnosis.event_study_results == {}
        assert any("too few for the covariance" in m for m in messages)

    def test_rank_deficient_on_a_small_fixed_panel(self):
        """Four units, two treated, no resampling: six distinct placebo assignments."""
        diagnosis, messages = self._run(
            _panel(4),
            [CallawaySantAnna(base_period="universal")],
            n_simulations=100,
            n_treated=2,
            resample_units=False,
        )
        assert diagnosis.event_study_results == {}
        assert any("rank-deficient" in m and "resample_units=False" in m for m in messages)

    def test_rare_cohort_excludes_too_many_draws(self, panel):
        """Three treated units over two cohorts: a quarter of the draws lose a cohort."""
        diagnosis, messages = self._run(
            panel,
            [CallawaySantAnna(base_period="universal")],
            cohorts=(5, 7),
            n_simulations=60,
            n_treated=3,
        )
        assert diagnosis.event_study_results == {}
        assert any("different event-study layout" in m for m in messages)

    def test_gapped_grid_keeps_its_own_provenance(self):
        """A universal-base design on a gapped grid: the surface records both bases.

        The first scored draw loses a cohort (a single base). The surface is
        built from the draws with both cohorts, records their two bases, and
        ``PreTrendsPower`` refuses it with its common-reference guard.
        """
        gapped = _panel(40, periods=[1, 2, 3, 4, 6])
        seed = 33
        generator = design_module._make_placebo_generator(gapped, "y", "id", "t", 6, [3, 6], True)
        first = generator(treatment_effect=0.0, seed=design_module._replicate_seeds(seed, 1)[0])
        cohorts_in_first = first.loc[first["_placebo_first_treat"] > 0, "_placebo_first_treat"]
        assert cohorts_in_first.nunique() == 1, "precondition: the first draw loses a cohort"

        diagnosis, _ = self._run(
            gapped,
            [CallawaySantAnna(base_period="universal")],
            cohorts=(3, 6),
            n_simulations=60,
            seed=seed,
            n_treated=6,
        )
        surface = diagnosis.event_study_results["CallawaySantAnna"]
        assert diagnosis._event_study_checks["CallawaySantAnna"]["n_excluded"] >= 1
        assert surface.reference_event_times == (-2, -1)
        with pytest.raises(ValueError, match="one common reference period"):
            PreTrendsPower().fit(surface)

    def test_full_pretest_that_over_rejects_warns(self):
        """Four treated units of forty: the test over all pre-periods rejects too often."""
        diagnosis, messages = self._run(
            _panel(40),
            [CallawaySantAnna(base_period="universal")],
            n_simulations=200,
            seed=3,
            n_treated=4,
        )
        checks = diagnosis._event_study_checks["CallawaySantAnna"]
        assert "CallawaySantAnna" in diagnosis.event_study_results
        assert checks["pretest_agrees"] is False
        assert checks["pretest_rejection_rate"] > checks["pretest_expected_rate"]
        assert any(
            "a pre-trend test over all 4 pre-periods" in m
            and f"{checks['pretest_rejection_rate']:.1%}" in m
            and f"{checks['pretest_expected_rate']:.1%}" in m
            for m in messages
        )
        row = [
            line for line in diagnosis.summary().splitlines() if line.startswith("CallawaySantAnna")
        ][-1]
        assert row.rstrip().endswith("NO")
        with pytest.raises(ValueError, match="Planned pre-trend power is not reported"):
            PreTrendsPower().fit(diagnosis)

    def test_size_not_controlled_warns_at_diagnosis(self, panel, monkeypatch):
        monkeypatch.setattr(design_module, "_size_controlled", lambda *args: False)
        diagnosis, messages = self._run(panel, [CallawaySantAnna(base_period="universal")])
        assert "CallawaySantAnna" in diagnosis.event_study_results
        assert any("did not pass the size check" in m for m in messages)

    def test_twfe_fit_keyword_of_the_same_name_still_works(self, panel):
        """``estimator_kwargs={"event_study": False}`` stays a TWFE fit keyword."""
        for flag in (False, True):
            diagnosis = _diagnose(
                panel,
                [TwoWayFixedEffects()],
                event_study=flag,
                estimator_kwargs={"event_study": False},
            )
            assert diagnosis.power_results["TwoWayFixedEffects"][0].n_simulations == 40
            assert diagnosis.event_study_results == {}


# ---------------------------------------------------------------------------
# PreTrendsPower on a planned design
# ---------------------------------------------------------------------------


class TestPlannedPower:
    """``PreTrendsPower`` on a diagnosis or one of its placebo event studies (NIS form)."""

    def test_diagnosis_and_surface_give_the_same_result(self, cs_diagnosis, cs_surface):
        planner = PreTrendsPower()
        from_diagnosis = planner.fit(cs_diagnosis)
        from_surface = planner.fit(cs_surface)
        assert from_diagnosis.mdv == pytest.approx(from_surface.mdv, rel=5e-3)
        assert from_diagnosis.covariance_source == "placebo_draws"
        assert from_diagnosis.original_results is cs_surface
        assert from_diagnosis.n_pre_periods == 4
        np.testing.assert_array_equal(from_diagnosis.vcov, from_surface.vcov)

    def test_equals_hand_computation(self, cs_surface):
        pre = [-5, -4, -3, -2]
        index = cs_surface.vcov_index.tolist()
        rows = [index.index(label) for label in pre]
        block = cs_surface.vcov[np.ix_(rows, rows)]
        weights = np.abs(np.asarray(pre, dtype=float) - (-1.0))

        planner = PreTrendsPower()
        fitted = planner.fit(cs_surface)
        np.testing.assert_array_equal(fitted.violation_weights, weights)
        np.testing.assert_allclose(fitted.vcov, block, rtol=1e-12)
        assert fitted.mdv == pytest.approx(planner._compute_mdv_nis(weights, block), rel=5e-3)
        assert fitted.power == pytest.approx(0.80, abs=NIS_ATOL)
        assert fitted.pretest_form == "nis"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"violation_type": "linear"},
            {"violation_type": "constant"},
            {"violation_type": "last_period"},
            {"violation_type": "custom", "violation_weights": np.array([4.0, 3.0, 2.0, 1.0])},
        ],
        ids=["linear", "constant", "last_period", "custom"],
    )
    def test_violation_types(self, cs_diagnosis, kwargs):
        fitted = PreTrendsPower(**kwargs).fit(cs_diagnosis)
        assert np.isfinite(fitted.mdv) and fitted.mdv > 0
        assert fitted.covariance_source == "placebo_draws"

    def test_other_entry_points_accept_the_diagnosis(self, cs_diagnosis, cs_surface):
        planner = PreTrendsPower()
        reference = planner.fit(cs_surface)
        assert compute_mdv(cs_diagnosis) == pytest.approx(reference.mdv, rel=5e-3)
        at_m = compute_pretrends_power(cs_diagnosis, M=0.3)
        assert at_m.power == pytest.approx(planner.fit(cs_surface, M=0.3).power, abs=NIS_ATOL)
        assert planner.power_at(cs_diagnosis, 0.3) == pytest.approx(at_m.power, abs=NIS_ATOL)
        curve = planner.power_curve(cs_diagnosis, M_grid=[0.0, 0.2, 0.4])
        expected = planner.power_curve(cs_surface, M_grid=[0.0, 0.2, 0.4])
        np.testing.assert_allclose(curve.powers, expected.powers, atol=NIS_ATOL)
        assert curve.mdv == pytest.approx(reference.mdv, rel=5e-3)

    def test_pre_period_subset(self, cs_diagnosis):
        planner = PreTrendsPower()
        subset = planner.fit(cs_diagnosis, pre_periods=[-3, -2])
        assert subset.n_pre_periods == 2
        np.testing.assert_array_equal(subset.violation_weights, [2.0, 1.0])
        with pytest.raises(ValueError, match="not eligible"):
            planner.fit(cs_diagnosis, pre_periods=[-9])

    def test_wald_form_is_refused_for_a_planned_input(self, cs_diagnosis, cs_surface):
        wald = PreTrendsPower(pretest_form="wald")
        for planned in (cs_diagnosis, cs_surface):
            with pytest.raises(ValueError, match="pretest_form='nis' only"):
                wald.fit(planned)
        with pytest.raises(ValueError, match="pretest_form='nis' only"):
            wald.power_curve(cs_diagnosis)
        with pytest.raises(ValueError, match="pretest_form='nis' only"):
            wald.power_by_pre_periods(cs_diagnosis)

    def test_check_is_made_on_the_tested_set(self):
        """Power is reported for a set that agrees and refused for one that does not."""
        surface = _independent_surface(n_pre=4, inflate=(0, 1))
        planner = PreTrendsPower()
        near = planner.fit(surface, pre_periods=[-3, -2])
        assert np.isfinite(near.mdv)
        for call in (
            lambda: planner.fit(surface),
            lambda: planner.fit(surface, pre_periods=[-5, -2]),
            lambda: planner.power_at(surface, 0.3),
            lambda: planner.power_curve(surface, M_grid=[0.1]),
        ):
            with pytest.raises(ValueError, match="Planned pre-trend power is not reported") as info:
                call()
            assert "more often than assumed" in str(info.value)

    def test_test_that_rejects_too_rarely_is_refused_too(self):
        surface = _independent_surface(n_pre=3)
        record = getattr(surface, design_module._PRETEST_RECORD)
        record["abs_t"] = record["abs_t"] * 0.6
        with pytest.raises(ValueError, match="less often than assumed"):
            PreTrendsPower().fit(surface)

    def test_surface_without_its_record_is_refused(self, cs_surface):
        """A copy rebuilt field by field cannot be checked, so nothing is planned from it."""
        import dataclasses

        copied = dataclasses.replace(cs_surface)
        assert copied.estimand == "placebo"
        with pytest.raises(ValueError, match="carries no per-draw record"):
            PreTrendsPower().fit(copied)

    def test_check_uses_the_planners_alpha(self):
        surface = _independent_surface(n_pre=3)
        planner = PreTrendsPower(alpha=0.10)
        counted, expected, n_draws, agrees = planner._planned_pretest_check(
            surface, [-4, -3, -2], np.array([3.0, 2.0, 1.0]), surface.vcov[:3, :3]
        )
        assert expected == pytest.approx(1.0 - 0.90**3, abs=NIS_ATOL)
        assert counted == pytest.approx(1.0 - 0.90**3, abs=0.03)
        assert n_draws == 4000 and agrees

    def test_diagnosis_without_the_keyword(self, panel):
        diagnosis = _diagnose(panel, [CallawaySantAnna()], event_study=False)
        with pytest.raises(ValueError, match="event_study=True"):
            PreTrendsPower().fit(diagnosis)

    def test_diagnosis_with_no_surface(self, panel):
        diagnosis = _diagnose(panel, [ImputationDiD()])
        with pytest.raises(ValueError, match="recorded no placebo event study"):
            PreTrendsPower().fit(diagnosis)

    def test_diagnosis_with_several_surfaces(self, panel):
        diagnosis = _diagnose(
            panel, {"a": CallawaySantAnna(base_period="universal"), "b": StackedDiD(kappa_pre=3)}
        )
        with pytest.raises(ValueError, match=r"event_study_results\[label\]"):
            PreTrendsPower().fit(diagnosis)
        assert np.isfinite(PreTrendsPower().fit(diagnosis.event_study_results["b"]).mdv)

    def test_varying_base_warns_on_a_planned_surface(self, panel):
        diagnosis = _diagnose(panel, [CallawaySantAnna()])
        with pytest.warns(UserWarning, match="without a universal"):
            PreTrendsPower().fit(diagnosis, pre_periods=[-2, -1])

    def test_size_warning_once_per_public_call(self, panel):
        diagnosis = _diagnose(panel, [CallawaySantAnna(base_period="universal")])
        diagnosis.size_controlled["CallawaySantAnna"] = False
        planner = PreTrendsPower()
        calls = {
            "fit": lambda: planner.fit(diagnosis),
            "power_at": lambda: planner.power_at(diagnosis, 0.2),
            "power_curve": lambda: planner.power_curve(diagnosis, M_grid=[0.1]),
            "power_by_pre_periods": lambda: planner.power_by_pre_periods(diagnosis),
        }
        for name, call in calls.items():
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                call()
            hits = [m for m in _messages(recorded) if "did not pass the size check" in m]
            assert len(hits) == 1, name


class TestPowerByPrePeriods:
    COLUMNS = [
        "n_pre_periods",
        "pre_periods",
        "mdv",
        "max_abs_pre_violation",
        "power_no_violation",
        "placebo_rejection_rate",
        "power",
    ]

    def test_rows_equal_fit_calls(self, cs_diagnosis, cs_surface):
        planner = PreTrendsPower()
        table = planner.power_by_pre_periods(cs_diagnosis, M=0.3)
        assert list(table.columns) == self.COLUMNS
        assert table["n_pre_periods"].tolist() == [1, 2, 3, 4]
        assert table["pre_periods"].tolist() == [[-2], [-3, -2], [-4, -3, -2], [-5, -4, -3, -2]]
        assert table["mdv"].notna().all(), "every tested set agrees on this fixture"
        for _, row in table.iterrows():
            fitted = planner.fit(cs_surface, M=0.3, pre_periods=row["pre_periods"])
            assert row["mdv"] == pytest.approx(fitted.mdv, rel=5e-3)
            assert row["max_abs_pre_violation"] == pytest.approx(
                fitted.max_abs_pre_violation, rel=5e-3
            )
            assert row["power"] == pytest.approx(fitted.power, abs=NIS_ATOL)
            assert row["power_no_violation"] == pytest.approx(fitted.power_at(0.0), abs=NIS_ATOL)
            counted, _ = design_module._placebo_pretest_rate(cs_surface, row["pre_periods"], 0.05)
            assert row["placebo_rejection_rate"] == counted
        assert np.all(np.isnan(planner.power_by_pre_periods(cs_diagnosis)["power"]))

    def test_no_violation_rate_grows_with_k(self):
        """Independent coefficients: the any-significant test rejects at 1 - 0.95^K."""
        table = PreTrendsPower().power_by_pre_periods(_independent_surface(n_pre=4))
        expected = 1.0 - 0.95 ** np.arange(1, 5)
        np.testing.assert_allclose(table["power_no_violation"], expected, atol=NIS_ATOL)
        np.testing.assert_allclose(table["placebo_rejection_rate"], expected, atol=0.02)
        assert np.all(np.diff(table["power_no_violation"]) > 10 * NIS_ATOL)
        assert table["mdv"].notna().all()

    def test_rows_that_disagree_are_blank(self):
        """Tests reaching the two earliest pre-periods reject too often: no number there."""
        surface = _independent_surface(n_pre=4, inflate=(0, 1))
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            table = PreTrendsPower().power_by_pre_periods(surface, M=0.5)
        assert table["mdv"].notna().tolist() == [True, True, False, False]
        blank = table.iloc[2:]
        assert blank[["mdv", "max_abs_pre_violation", "power"]].isna().all().all()
        assert (blank["placebo_rejection_rate"] > blank["power_no_violation"]).all()
        assert table.iloc[:2]["power"].notna().all()
        hits = [m for m in _messages(recorded) if "is not reported for tests of [3, 4]" in m]
        assert len(hits) == 1

    def test_refuses_a_fitted_model(self, panel):
        frame = design_module._make_placebo_generator(panel, "y", "id", "t", 10, [6], True)(
            treatment_effect=0.0, seed=1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = CallawaySantAnna(base_period="universal").fit(
                frame, outcome="y", unit="id", time="t", first_treat="_placebo_first_treat"
            )
        for realized in (fitted, fitted.aggregate("event_study")):
            with pytest.raises(TypeError, match="has not been run"):
                PreTrendsPower().power_by_pre_periods(realized)

    @pytest.mark.parametrize("magnitude", [float("nan"), float("inf")])
    def test_non_finite_magnitude_is_refused(self, cs_diagnosis, cs_surface, magnitude):
        """A NaN magnitude must not be reported as a rejection probability of one."""
        planner = PreTrendsPower()
        for planned in (cs_diagnosis, cs_surface):
            with pytest.raises(ValueError, match="must be finite"):
                planner.power_by_pre_periods(planned, M=magnitude)
            with pytest.raises(ValueError, match="must be finite"):
                planner.fit(planned, M=magnitude)
            with pytest.raises(ValueError, match="must be finite"):
                planner.power_at(planned, magnitude)
            with pytest.raises(ValueError, match="must be finite"):
                planner.power_curve(planned, M_grid=[0.1, magnitude])
        with pytest.raises(ValueError, match="must be finite"):
            planner.fit(cs_surface).power_at(magnitude)

    def test_refuses_custom_weights(self, cs_diagnosis):
        planner = PreTrendsPower(violation_type="custom", violation_weights=np.ones(4))
        with pytest.raises(ValueError, match="violation_type='custom'"):
            planner.power_by_pre_periods(cs_diagnosis)

    def test_repeated_warning_is_issued_once(self, panel):
        diagnosis = _diagnose(panel, [CallawaySantAnna()])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            table = PreTrendsPower().power_by_pre_periods(diagnosis)
        assert len(table) > 1
        hits = [m for m in _messages(recorded) if "without a universal" in m]
        assert len(hits) == 1


# ---------------------------------------------------------------------------
# Consumers that must not read a placebo surface as an estimate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_cs(panel):
    frame = design_module._make_placebo_generator(panel, "y", "id", "t", 10, [6], True)(
        treatment_effect=0.5, seed=2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CallawaySantAnna(base_period="universal").fit(
            frame, outcome="y", unit="id", time="t", first_treat="_placebo_first_treat"
        )


class TestConsumers:
    def test_honest_did_refuses(self, cs_surface):
        with pytest.raises(ValueError, match="placebo event study from diagnose_design"):
            HonestDiD().fit(cs_surface)
        with pytest.raises(ValueError, match="placebo event study from diagnose_design"):
            HonestDiD().sensitivity_analysis(cs_surface)

    def test_sensitivity_to_honest_did_refuses(self, cs_diagnosis, cs_surface):
        for planned in (cs_diagnosis, cs_surface):
            with pytest.raises(ValueError, match="describes a realized event study"):
                PreTrendsPower().sensitivity_to_honest_did(planned)

    def test_reports_refuse_planned_power(self, cs_diagnosis, fitted_cs):
        planned = PreTrendsPower().fit(cs_diagnosis)
        precomputed = {"pretrends_power": planned}
        with pytest.raises(ValueError, match="DiagnosticReport: precomputed"):
            DiagnosticReport(fitted_cs, precomputed=precomputed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ready = DiagnosticReport(fitted_cs)
        paths = (
            {},
            {"auto_diagnostics": False},
            {"diagnostics": ready},
        )
        for extra in paths:
            with pytest.raises(ValueError, match="BusinessReport: precomputed"):
                BusinessReport(fitted_cs, precomputed=precomputed, **extra)

    def test_reports_still_accept_fitted_power(self, fitted_cs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            realized = compute_pretrends_power(fitted_cs.aggregate("event_study"))
            assert realized.covariance_source != "placebo_draws"
            DiagnosticReport(fitted_cs, precomputed={"pretrends_power": realized})
            BusinessReport(fitted_cs, precomputed={"pretrends_power": realized})

    def test_plot_event_study_accepts(self, cs_surface):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from diff_diff import plot_event_study

        ax = plot_event_study(cs_surface, ylabel="Placebo estimate", show=False)
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Results contract
# ---------------------------------------------------------------------------


class TestResultsContract:
    def test_pickle_round_trip(self, cs_diagnosis):
        loaded = pickle.loads(pickle.dumps(cs_diagnosis))
        assert loaded.event_study is True
        np.testing.assert_array_equal(
            loaded.event_study_results["CallawaySantAnna"].vcov,
            cs_diagnosis.event_study_results["CallawaySantAnna"].vcov,
        )
        assert loaded._event_study_checks == cs_diagnosis._event_study_checks
        record = getattr(
            loaded.event_study_results["CallawaySantAnna"], design_module._PRETEST_RECORD
        )
        original = getattr(
            cs_diagnosis.event_study_results["CallawaySantAnna"], design_module._PRETEST_RECORD
        )
        np.testing.assert_array_equal(record["abs_t"], original["abs_t"])
        planner = PreTrendsPower()
        assert planner.fit(loaded).mdv == pytest.approx(planner.fit(cs_diagnosis).mdv, rel=5e-3)

    def test_pickle_from_before_the_fields_existed(self, panel):
        """A state without the three new attributes loads with their defaults."""
        diagnosis = _diagnose(panel, [CallawaySantAnna()], event_study=False)
        state = dict(diagnosis.__dict__)
        for name in ("event_study", "event_study_results", "_event_study_checks"):
            del state[name]
        old = object.__new__(type(diagnosis))
        old.__dict__.update(state)
        loaded = pickle.loads(pickle.dumps(old))
        assert loaded.event_study is False
        assert loaded.event_study_results == {}
        assert loaded._event_study_checks == {}
        assert loaded.summary() == diagnosis.summary()
        assert loaded.to_dict()["event_study"] is False
        assert isinstance(loaded == diagnosis, bool)
        assert loaded.lock(progress=False).to_json() == diagnosis.lock(progress=False).to_json()

    def test_equality_does_not_raise_with_surfaces(self, panel):
        """Surfaces hold arrays; ``==`` must still return a bool."""
        estimator = [CallawaySantAnna(base_period="universal")]
        first = _diagnose(panel, estimator, seed=3)
        second = _diagnose(panel, estimator, seed=3)
        assert first.event_study_results and second.event_study_results
        assert isinstance(first == second, bool)
        assert first == first

    def test_lock_is_unchanged_by_the_keyword(self, panel):
        estimators = {
            "cs": CallawaySantAnna(base_period="universal"),
            "st": StackedDiD(kappa_pre=3),
        }
        locks = []
        for flag in (False, True):
            diagnosis = _diagnose(panel, estimators, seed=9, event_study=flag)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                locks.append(diagnosis.lock("cs", data=panel, progress=False).to_json())
        assert locks[0] == locks[1]


# ---------------------------------------------------------------------------
# Simulation evidence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planning_evidence():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks/data/pretrend_planning_check.json"
    if not path.exists():
        pytest.skip("benchmarks/data/pretrend_planning_check.json is not available")
    return json.loads(path.read_text())


class TestPlanningEvidence:
    """The numbers the REGISTRY ``## DesignDiagnosis`` placebo event-study Notes quote."""

    @staticmethod
    def _cells(evidence, passed):
        return [
            (point, run, cell)
            for point in evidence["design_points"]
            for run in point["runs"]
            for cell in run["cells"]
            if cell["check_passed"] is passed
        ]

    def test_grid(self, planning_evidence):
        meta = planning_evidence["meta"]
        assert (meta["n_diagnosis"], meta["n_direct"], meta["n_cells"]) == (500, 1000, 48)
        assert len(planning_evidence["design_points"]) == 12
        assert (meta["n_cells_check_passed"], meta["n_cells_check_failed"]) == (28, 20)

    def test_which_cells_pass_the_pretest_check(self, planning_evidence):
        def by_units(passed):
            units = [point["n_units"] for point, _, _ in self._cells(planning_evidence, passed)]
            return units.count(40), units.count(200)

        assert by_units(True) == (6, 22)
        assert by_units(False) == (18, 2)

    def test_formula_holds_where_the_check_agrees(self, planning_evidence):
        cells = [cell for _, _, cell in self._cells(planning_evidence, True)]
        assert not any(cell["flag_power"] or cell["flag_no_violation"] for cell in cells)
        powers = [cell["direct_power_at_mdv"] for cell in cells]
        assert 0.725 <= min(powers) and max(powers) <= 0.845
        assert np.mean(powers) == pytest.approx(0.79, abs=0.005)
        gaps = [cell["direct_no_violation"] - cell["planned_no_violation"] for cell in cells]
        assert -0.0045 <= min(gaps) and max(gaps) <= 0.0525
        counted = [cell["placebo_rejection_rate"] - cell["planned_no_violation"] for cell in cells]
        assert -0.0195 <= min(counted) and max(counted) <= 0.0245

    def test_formula_fails_where_the_check_disagrees(self, planning_evidence):
        cells = [cell for _, _, cell in self._cells(planning_evidence, False)]
        flagged = [cell for cell in cells if cell["flag_power"] or cell["flag_no_violation"]]
        assert len(flagged) == 15
        unflagged_units = [
            point["n_units"]
            for point, _, cell in self._cells(planning_evidence, False)
            if not (cell["flag_power"] or cell["flag_no_violation"])
        ]
        assert sorted(unflagged_units) == [40, 40, 40, 200, 200]
        powers = [cell["direct_power_at_mdv"] for cell in cells]
        assert 0.775 <= min(powers) and max(powers) <= 0.905
        assert np.mean(powers) == pytest.approx(0.83, abs=0.005)
        gaps = [cell["direct_no_violation"] - cell["planned_no_violation"] for cell in cells]
        assert 0.0225 <= min(gaps) and max(gaps) <= 0.2155
        counted = [cell["placebo_rejection_rate"] - cell["planned_no_violation"] for cell in cells]
        assert 0.0285 <= min(counted) and max(counted) <= 0.2165

    def test_monte_carlo_spread_of_the_planned_mdv(self, planning_evidence):
        spread = {
            (row["n_units"], row["n_simulations"]): row["relative_spread"]
            for row in planning_evidence["stability"]
        }
        assert all(0.04 <= spread[(n, 200)] <= 0.05 for n in (40, 200))
        assert all(0.03 <= spread[(n, 500)] <= 0.04 for n in (40, 200))
        assert all(0.02 <= spread[(n, 1000)] <= 0.025 for n in (40, 200))

    def test_selection_effect_is_small(self, planning_evidence):
        rows = {row["n_units"]: row for row in planning_evidence["selection"]}
        assert all(row["measurable"] and not row["flag"] for row in rows.values())
        assert rows[40]["relative_difference_mean"] == pytest.approx(-0.017, abs=6e-4)
        assert rows[40]["relative_difference_se"] == pytest.approx(0.007, abs=6e-4)
        assert rows[200]["relative_difference_mean"] == pytest.approx(0.014, abs=6e-4)
        assert rows[200]["relative_difference_se"] == pytest.approx(0.006, abs=6e-4)


@pytest.fixture(scope="module")
def planning_benchmark():
    """The benchmark script as a module (skipped where benchmarks/ is not checked out)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks/python/pretrend_planning_check.py"
    if not path.exists():
        pytest.skip("benchmarks/python/pretrend_planning_check.py is not available")
    spec = importlib.util.spec_from_file_location("pretrend_planning_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBenchmarkScoring:
    """The benchmark scores every tested set of a run on the same fresh draws."""

    @staticmethod
    def _surface(event_time, ratios):
        event_time = np.asarray(event_time)
        is_reference = event_time == -1
        ratios = np.asarray(ratios, dtype=float)
        return EventStudyResults(
            event_time=event_time,
            att=np.where(is_reference, 0.0, ratios),
            se=np.where(is_reference, np.nan, 1.0),
            t_stat=np.where(is_reference, np.nan, ratios),
            p_value=np.where(is_reference, np.nan, np.where(np.abs(ratios) > 1.96, 0.01, 0.5)),
            conf_int_lower=np.where(is_reference, np.nan, ratios - 1.96),
            conf_int_upper=np.where(is_reference, np.nan, ratios + 1.96),
            is_reference=is_reference,
            n=np.full(len(event_time), np.nan),
        )

    def test_a_draw_missing_a_planned_period_is_not_scored_at_all(self, planning_benchmark):
        """It rejects on the nearest set but lacks the earliest period: no partial count."""
        required = [-4, -3, -2]
        tested_sets = [[-2], [-3, -2], [-4, -3, -2]]
        short = self._surface([-3, -2, -1, 0], [0.1, 5.0, 0.0, 0.0])
        assert planning_benchmark.score_event_study(short, required, tested_sets, 1.96) is None
        full = self._surface([-4, -3, -2, -1, 0], [3.0, 0.1, 0.2, 0.0, 0.0])
        assert planning_benchmark.score_event_study(full, required, tested_sets, 1.96) == [
            (False, False),
            (False, False),
            (True, True),
        ]

    def test_a_draw_with_a_non_finite_ratio_is_not_scored(self, planning_benchmark):
        surface = self._surface([-4, -3, -2, -1, 0], [np.nan, 0.1, 5.0, 0.0, 0.0])
        assert planning_benchmark.score_event_study(surface, [-4, -3, -2], [[-2]], 1.96) is None
