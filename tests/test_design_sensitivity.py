"""Sensitivity planning at the design stage: ``HonestDiD.power_by_M``.

``diagnose_design(..., event_study=True)`` keeps the event-study container of
every no-effect placebo draw. ``HonestDiD.power_by_M`` runs the sensitivity
analysis on each of them and counts how often the robust interval would detect
an effect of a given size.

The counting rests on one identity, checked here against real injection: an
effect that is the same across treated units moves a draw's robust interval by
the value of the target parameter and changes nothing else.
"""

import dataclasses
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    HonestDiD,
    PreTrendsPower,
    StackedDiD,
    diagnose_design,
)
from diff_diff import design as design_module
from diff_diff import honest_did as honest_module
from diff_diff.honest_did import _score_planned_intervals

COLUMNS = [
    "M",
    "placebo_rejection_rate",
    "ci_width",
    "mde",
    "power",
    "n_simulations",
    "n_simulation_failures",
]


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


def _diagnose(panel, estimators, cohorts=(6,), n_simulations=30, seed=1, **kwargs):
    kwargs.setdefault("event_study", True)
    if "n_treated" not in kwargs:
        kwargs.setdefault("treatment_fraction", 0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diagnose_design(
            panel,
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


def _table(honest, results, **kwargs):
    """``power_by_M`` without its progress lines or warnings."""
    kwargs.setdefault("progress", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return honest.power_by_M(results, **kwargs)


def _intervals(honest, containers, M):
    """The robust interval ``honest.fit`` gives on each container."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for container in containers:
            fitted = honest.fit(container, M=M)
            out.append((fitted.ci_lb, fitted.ci_ub))
    return np.asarray(out, dtype=float)


def _replay(panel, estimator, cohorts, n_simulations, seed, inject, n_treated=10):
    """Refit every replicate of a diagnosis by hand, with ``inject`` applied.

    ``inject(frame)`` returns the outcome column to use (the frame is the
    no-effect placebo draw). Returns the event-study containers.
    """
    generator = design_module._make_placebo_generator(
        panel, "y", "id", "t", n_treated, list(cohorts), True
    )
    fit_kwargs = design_module._fit_kwargs(estimator, "y", "id", "t", {})
    containers = []
    for replicate_seed in design_module._replicate_seeds(seed, n_simulations):
        frame = generator(treatment_effect=0.0, seed=replicate_seed)
        frame["y"] = inject(frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = design_module._clone(estimator).fit(frame, **fit_kwargs)
            containers.append(fitted.aggregate("event_study"))
    return containers


def _constant(effect):
    def inject(frame):
        first = frame[design_module._FIRST_TREAT_COLUMN].to_numpy()
        active = (first > 0) & (frame["t"].to_numpy() >= first)
        return frame["y"].to_numpy(dtype=float) + effect * active

    return inject


def _by_event_time(effects):
    def inject(frame):
        first = frame[design_module._FIRST_TREAT_COLUMN].to_numpy()
        time = frame["t"].to_numpy()
        added = np.zeros(len(frame))
        for event_time, effect in effects.items():
            added[(first > 0) & (time - first == event_time)] = effect
        return frame["y"].to_numpy(dtype=float) + added

    return inject


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def cs():
    return CallawaySantAnna(base_period="universal")


@pytest.fixture(scope="module")
def cs_diagnosis(panel, cs):
    return _diagnose(panel, [cs])


@pytest.fixture(scope="module")
def cs_surface(cs_diagnosis):
    return cs_diagnosis.event_study_results["CallawaySantAnna"]


# ---------------------------------------------------------------------------
# The shift identity, against real injection
# ---------------------------------------------------------------------------


class TestShiftIdentity:
    """Counting from the no-effect draws equals refitting with the effect injected."""

    @pytest.mark.parametrize(
        "estimator, cohorts",
        [
            (CallawaySantAnna(base_period="universal"), (6,)),
            (CallawaySantAnna(base_period="universal", control_group="not_yet_treated"), (5, 7)),
            (CallawaySantAnna(base_period="universal"), (5, 7)),
            (StackedDiD(kappa_pre=3, kappa_post=1), (6,)),
        ],
        ids=["cs-common", "cs-nyt-staggered", "cs-staggered", "stacked"],
    )
    @pytest.mark.parametrize(
        "method, M",
        [("relative_magnitude", 1.0), ("smoothness", 0.0)],
        ids=["rm", "sd0"],
    )
    def test_injected_effect_equals_shifted_interval(self, panel, estimator, cohorts, method, M):
        n_simulations, seed, effect = 30, 7, 1.5
        diagnosis = _diagnose(panel, [estimator], cohorts, n_simulations, seed)
        label = type(estimator).__name__
        surface = diagnosis.event_study_results[label]
        draws, n_lost = design_module._placebo_draw_record(surface)
        assert len(draws) == n_simulations and n_lost == 0

        honest = HonestDiD(method=method)
        no_effect = _intervals(honest, draws, M)
        injected = _intervals(
            honest, _replay(panel, estimator, cohorts, n_simulations, seed, _constant(effect)), M
        )
        np.testing.assert_allclose(injected, no_effect + effect, rtol=0, atol=1e-8)

        table = _table(honest, diagnosis, M_grid=[M], effect_size=effect)
        row = table.iloc[0]
        assert row["n_simulations"] == n_simulations
        assert row["n_simulation_failures"] == 0
        counted = np.mean((injected[:, 0] > 0) | (injected[:, 1] < 0))
        detected = np.mean(injected[:, 0] > 0)
        by_hand_rate = np.mean((no_effect[:, 0] > 0) | (no_effect[:, 1] < 0))
        assert row["placebo_rejection_rate"] == pytest.approx(by_hand_rate, abs=1e-12)
        np.testing.assert_allclose(
            row["ci_width"], np.mean(no_effect[:, 1] - no_effect[:, 0]), rtol=1e-12
        )
        if np.isnan(row["mde"]):
            # Withheld by the size rule: the planned numbers are blank together.
            assert np.isnan(row["power"])
            assert not design_module._size_controlled(by_hand_rate, n_simulations, 0, 0.05)
        else:
            assert row["power"] == pytest.approx(detected, abs=1e-12)
            # No draw excludes zero on the wrong side of a positive effect
            # here, so the two-sided count agrees as well.
            assert counted == pytest.approx(detected, abs=1e-12)

    def test_relative_magnitude_rows_are_reported(self, cs_diagnosis):
        """The identity test above is not vacuous: these rows pass the size rule."""
        table = _table(HonestDiD(), cs_diagnosis, M_grid=[0.5, 1.0], effect_size=1.5)
        assert np.all(np.isfinite(table["mde"])) and np.all(np.isfinite(table["power"]))

    def test_smoothness_with_positive_M(self, panel, cs):
        """The optimized interval (M > 0) shifts by the injected effect too."""
        n_simulations, seed, effect, M = 20, 3, 1.0, 0.05
        diagnosis = _diagnose(panel, [cs], (6,), n_simulations, seed)
        draws, _ = design_module._placebo_draw_record(
            diagnosis.event_study_results["CallawaySantAnna"]
        )
        honest = HonestDiD(method="smoothness")
        no_effect = _intervals(honest, draws, M)
        injected = _intervals(
            honest, _replay(panel, cs, (6,), n_simulations, seed, _constant(effect)), M
        )
        np.testing.assert_allclose(injected, no_effect + effect, rtol=0, atol=1e-6)
        row = _table(honest, diagnosis, M_grid=[M], effect_size=effect).iloc[0]
        np.testing.assert_allclose(
            row["ci_width"], np.mean(no_effect[:, 1] - no_effect[:, 0]), rtol=1e-9
        )

    @pytest.mark.parametrize(
        "estimator, estimator_kwargs",
        [
            (CallawaySantAnna(base_period="universal", n_bootstrap=29, seed=5), {}),
            (CallawaySantAnna(base_period="universal"), {"covariates": ["x"]}),
        ],
        ids=["bootstrap", "covariate"],
    )
    def test_bootstrap_and_covariate_candidates(self, panel, estimator, estimator_kwargs):
        """The identity does not depend on the candidate's configuration."""
        n_simulations, seed, effect = 20, 4, 1.5
        data = panel.copy()
        data["x"] = np.random.default_rng(11).normal(size=len(data))
        diagnosis = _diagnose(
            data, [estimator], (6,), n_simulations, seed, estimator_kwargs=estimator_kwargs
        )
        draws, _ = design_module._placebo_draw_record(
            diagnosis.event_study_results["CallawaySantAnna"]
        )
        generator = design_module._make_placebo_generator(data, "y", "id", "t", 10, [6], True)
        fit_kwargs = design_module._fit_kwargs(estimator, "y", "id", "t", estimator_kwargs)
        injected = []
        for replicate_seed in design_module._replicate_seeds(seed, n_simulations):
            frame = generator(treatment_effect=effect, seed=replicate_seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = design_module._clone(estimator).fit(frame, **fit_kwargs)
                injected.append(fitted.aggregate("event_study"))
        honest = HonestDiD()
        np.testing.assert_allclose(
            _intervals(honest, injected, 1.0),
            _intervals(honest, draws, 1.0) + effect,
            rtol=0,
            atol=1e-8,
        )

    @pytest.mark.parametrize("method, M", [("relative_magnitude", 1.0), ("smoothness", 0.0)])
    def test_effect_that_differs_by_event_time(self, panel, cs, cs_diagnosis, method, M):
        """``effect_size`` is the value of ``l_vec' tau``, whatever the profile."""
        effects = {0: 0.4, 1: 1.3, 2: -0.2}
        l_vec = np.array([0.5, 0.5, 0.0])
        target = 0.5 * 0.4 + 0.5 * 1.3
        draws, _ = design_module._placebo_draw_record(
            cs_diagnosis.event_study_results["CallawaySantAnna"]
        )
        honest = HonestDiD(method=method, l_vec=l_vec)
        injected = _intervals(honest, _replay(panel, cs, (6,), 30, 1, _by_event_time(effects)), M)
        np.testing.assert_allclose(
            injected, _intervals(honest, draws, M) + target, rtol=0, atol=1e-8
        )

    def test_constant_effect_under_another_l_vec(self, panel, cs, cs_diagnosis):
        """A constant effect ``e`` moves the target by ``e * sum(l_vec)``."""
        effect = 1.5
        draws, _ = design_module._placebo_draw_record(
            cs_diagnosis.event_study_results["CallawaySantAnna"]
        )
        injected = _replay(panel, cs, (6,), 30, 1, _constant(effect))
        for l_vec in (np.array([1.0, 1.0, 0.0]), np.array([1.0, -1.0, 0.0])):
            honest = HonestDiD(l_vec=l_vec)
            np.testing.assert_allclose(
                _intervals(honest, injected, 1.0),
                _intervals(honest, draws, 1.0) + effect * l_vec.sum(),
                rtol=0,
                atol=1e-8,
            )

    def test_linear_trend_does_not_move_the_smoothness_interval(self, panel, cs):
        """Backs the docstring: a linear trend difference leaves it unchanged."""
        generator = design_module._make_placebo_generator(panel, "y", "id", "t", 10, [6], True)
        fit_kwargs = design_module._fit_kwargs(cs, "y", "id", "t", {})
        frame = generator(treatment_effect=0.0, seed=17)
        trended = frame.copy()
        treated = trended[design_module._FIRST_TREAT_COLUMN].to_numpy() > 0
        trended["y"] = trended["y"].to_numpy(dtype=float) + 0.3 * treated * trended["t"].to_numpy()
        containers = []
        for data in (frame, trended):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                containers.append(
                    design_module._clone(cs).fit(data, **fit_kwargs).aggregate("event_study")
                )
        honest = HonestDiD(method="smoothness")
        for M in (0.0, 0.05):
            both = _intervals(honest, containers, M)
            np.testing.assert_allclose(both[1], both[0], rtol=0, atol=1e-6)
        # The relative-magnitudes interval does move: the trend enters its bound.
        both = _intervals(HonestDiD(), containers, 1.0)
        assert abs(both[1, 1] - both[0, 1]) > 0.1


# ---------------------------------------------------------------------------
# Scoring one row from the draws' intervals
# ---------------------------------------------------------------------------


def _score(lower, upper, n_lost=0, alpha=0.05, power=0.8, effect_size=None):
    return _score_planned_intervals(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        n_lost,
        alpha,
        power,
        effect_size,
    )


class TestScoring:
    def test_symmetric_intervals(self):
        lower = -np.arange(1.0, 101.0)
        upper = np.arange(1.0, 101.0)
        row = _score(lower, upper)
        assert row["placebo_rejection_rate"] == 0.0
        assert row["ci_width"] == pytest.approx(101.0)
        assert row["mde"] == 80.0
        assert row["n_simulations"] == 100 and row["n_simulation_failures"] == 0

    def test_power_reaches_the_target_above_mde_only(self):
        lower = -np.arange(1.0, 101.0)
        upper = np.arange(1.0, 101.0)
        mde = _score(lower, upper)["mde"]
        for sign in (1.0, -1.0):
            assert _score(lower, upper, effect_size=sign * (mde + 1e-9))["power"] >= 0.8
            # At mde itself an endpoint sits on zero, which is not an exclusion.
            assert _score(lower, upper, effect_size=sign * mde)["power"] < 0.8

    def test_mde_is_the_larger_of_the_two_signs(self):
        # Lower endpoints near zero (a positive effect is detected early),
        # upper endpoints far from it (a negative effect is detected late).
        lower = -np.linspace(0.1, 1.0, 10)
        upper = np.linspace(2.0, 11.0, 10)
        row = _score(lower, upper, alpha=0.5)
        assert row["mde"] == pytest.approx(9.0)
        assert _score(lower, upper, alpha=0.5, effect_size=-9.0 - 1e-9)["power"] >= 0.8
        assert _score(lower, upper, alpha=0.5, effect_size=-9.0 + 1e-9)["power"] < 0.8
        # The positive side needed far less.
        assert _score(lower, upper, alpha=0.5, effect_size=0.9)["power"] >= 0.8

    def test_wrong_side_exclusion_is_not_a_detection(self):
        # One interval lies entirely below zero. With no effect it is a
        # rejection; for a small positive effect it is not a detection.
        lower = np.array([-3.0] + [-1.0] * 99)
        upper = np.array([-2.0] + [1.0] * 99)
        assert _score(lower, upper)["placebo_rejection_rate"] == pytest.approx(0.01)
        assert _score(lower, upper, effect_size=0.5)["power"] == 0.0
        assert _score(lower, upper, effect_size=-0.5)["power"] == pytest.approx(0.01)

    def test_power_at_zero_effect_is_the_placebo_rate(self):
        lower = np.array([1.0, -3.0] + [-1.0] * 98)
        upper = np.array([2.0, -2.0] + [1.0] * 98)
        row = _score(lower, upper, effect_size=0.0)
        assert row["power"] == row["placebo_rejection_rate"] == pytest.approx(0.02)

    def test_mde_is_never_negative(self):
        # A target below the share already excluding zero on each side.
        lower = np.array([1.0, -2.0] + [-1.0] * 98)
        upper = np.array([2.0, -1.0] + [1.0] * 98)
        row = _score(lower, upper, power=0.005)
        assert row["placebo_rejection_rate"] == pytest.approx(0.02)
        assert row["mde"] == 0.0

    def test_power_without_effect_size_is_nan(self):
        row = _score([-1.0] * 50, [1.0] * 50)
        assert np.isnan(row["power"]) and row["mde"] == pytest.approx(1.0)

    def test_over_rejection_withholds_mde_and_power(self):
        lower = np.array([0.5] * 20 + [-1.0] * 80)
        upper = np.array([1.5] * 20 + [1.0] * 80)
        row = _score(lower, upper, effect_size=2.0)
        assert row["placebo_rejection_rate"] == pytest.approx(0.2)
        assert row["ci_width"] == pytest.approx(0.2 * 1.0 + 0.8 * 2.0)
        assert np.isnan(row["mde"]) and np.isnan(row["power"])
        assert not row["size_controlled"]

    def test_the_bound_is_the_one_of_the_diagnosis(self):
        # 100 draws at alpha = 0.05: the bound is 0.05 + 1.96 * sqrt(.05 * .95 / 100).
        bound = 0.05 + 1.96 * np.sqrt(0.05 * 0.95 / 100)
        assert 0.09 < bound < 0.10
        for rejections, reported in ((9, True), (10, False)):
            lower = np.array([0.5] * rejections + [-1.0] * (100 - rejections))
            upper = np.array([1.5] * rejections + [1.0] * (100 - rejections))
            assert bool(np.isfinite(_score(lower, upper)["mde"])) is reported

    def test_draws_without_a_finite_interval_are_lost(self):
        lower = np.array([np.nan, -np.inf] + [-1.0] * 98)
        upper = np.array([np.nan, np.inf] + [1.0] * 98)
        row = _score(lower, upper, n_lost=3)
        assert row["n_simulations"] == 98 and row["n_simulation_failures"] == 5
        assert np.isfinite(row["mde"])

    def test_more_than_ten_percent_lost_withholds(self):
        lower, upper = np.full(90, -1.0), np.full(90, 1.0)
        assert np.isfinite(_score(lower, upper, n_lost=10)["mde"])
        row = _score(lower, upper, n_lost=11)
        assert np.isnan(row["mde"]) and row["placebo_rejection_rate"] == 0.0

    def test_no_scored_draw(self):
        row = _score([np.nan] * 5, [np.nan] * 5, n_lost=2)
        assert row["n_simulations"] == 0 and row["n_simulation_failures"] == 7
        assert np.isnan(row["placebo_rejection_rate"]) and np.isnan(row["ci_width"])
        assert np.isnan(row["mde"]) and not row["size_controlled"]

    def test_target_count_is_not_inflated_by_rounding(self):
        # 0.7 * 10 is 7.000000000000001 in floating point: k must stay 7.
        lower = -np.arange(1.0, 11.0)
        upper = np.arange(1.0, 11.0)
        assert _score(lower, upper, alpha=0.5, power=0.7)["mde"] == 7.0


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TestTable:
    def test_columns_and_default_grid(self, cs_diagnosis):
        table = _table(HonestDiD(), cs_diagnosis)
        assert list(table.columns) == COLUMNS
        assert table["M"].tolist() == [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        assert np.all(np.isnan(table["power"]))
        # A looser restriction never makes the interval narrower.
        assert np.all(np.diff(table["ci_width"]) >= -1e-12)
        assert np.all(np.diff(table["mde"]) >= -1e-12)

    def test_default_grid_is_the_one_of_sensitivity_analysis(self, panel, cs):
        assert honest_module._default_M_grid("relative_magnitude") == [
            0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
        ]
        assert honest_module._default_M_grid("smoothness") == [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
        generator = design_module._make_placebo_generator(panel, "y", "id", "t", 10, [6], True)
        frame = generator(treatment_effect=1.0, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = cs.fit(frame, **design_module._fit_kwargs(cs, "y", "id", "t", {}))
            sensitivity = HonestDiD().sensitivity_analysis(fitted.aggregate("event_study"))
        assert sensitivity.M_values.tolist() == honest_module._default_M_grid("relative_magnitude")

    def test_surface_and_diagnosis_give_the_same_table(self, cs_diagnosis, cs_surface):
        kwargs = dict(M_grid=[0.5, 1.0], effect_size=2.0)
        pd.testing.assert_frame_equal(
            _table(HonestDiD(), cs_diagnosis, **kwargs), _table(HonestDiD(), cs_surface, **kwargs)
        )

    def test_uses_the_instance_alpha_and_l_vec(self, cs_surface):
        draws, _ = design_module._placebo_draw_record(cs_surface)
        honest = HonestDiD(alpha=0.2, l_vec=np.array([1.0, 0.0, 0.0]))
        expected = _intervals(honest, draws, 1.0)
        row = _table(honest, cs_surface, M_grid=[1.0]).iloc[0]
        np.testing.assert_allclose(
            row["ci_width"], np.mean(expected[:, 1] - expected[:, 0]), rtol=1e-12
        )
        default = _table(HonestDiD(), cs_surface, M_grid=[1.0]).iloc[0]
        assert row["ci_width"] != pytest.approx(default["ci_width"])

    def test_power_is_higher_for_a_larger_effect(self, cs_diagnosis):
        small = _table(HonestDiD(), cs_diagnosis, M_grid=[0.5], effect_size=1.0)["power"][0]
        large = _table(HonestDiD(), cs_diagnosis, M_grid=[0.5], effect_size=4.0)["power"][0]
        assert 0.0 <= small < large <= 1.0

    def test_negative_effect_size(self, cs_surface):
        draws, _ = design_module._placebo_draw_record(cs_surface)
        intervals = _intervals(HonestDiD(), draws, 0.5)
        row = _table(HonestDiD(), cs_surface, M_grid=[0.5], effect_size=-3.0).iloc[0]
        assert row["power"] == pytest.approx(np.mean(intervals[:, 1] - 3.0 < 0), abs=1e-12)

    def test_progress_lines(self, cs_diagnosis, capsys):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HonestDiD().power_by_M(cs_diagnosis, M_grid=[0.5, 1.0])
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 3
        assert "30 placebo draws" in lines[0] and "2 values of M" in lines[0]
        assert "M=0.5" in lines[1] and "30 scored, 0 lost" in lines[1]
        _table(HonestDiD(), cs_diagnosis, M_grid=[0.5])
        assert capsys.readouterr().out == ""


class TestSizeRule:
    def test_over_rejecting_interval_is_withheld(self, panel, cs):
        """Four treated units: the smoothness interval excludes zero far too often."""
        diagnosis = _diagnose(panel, [cs], n_simulations=60, n_treated=4)
        with pytest.warns(UserWarning, match="fails the size check") as recorded:
            table = HonestDiD(method="smoothness").power_by_M(
                diagnosis, M_grid=[0.0], effect_size=2.0, progress=False
            )
        row = table.iloc[0]
        assert row["placebo_rejection_rate"] > 0.15
        assert np.isfinite(row["ci_width"])
        assert np.isnan(row["mde"]) and np.isnan(row["power"])
        withheld = [w for w in recorded if "fails the size check" in str(w.message)]
        assert len(withheld) == 1
        assert "M=0 (excludes zero with no effect in" in str(withheld[0].message)

    def test_interval_that_holds_its_level_is_not_flagged(self, cs_diagnosis):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            table = HonestDiD().power_by_M(cs_diagnosis, M_grid=[1.0], progress=False)
        assert np.isfinite(table["mde"][0])

    def test_only_the_failing_rows_are_blank(self, panel, cs):
        diagnosis = _diagnose(panel, [cs], n_simulations=60, n_treated=4)
        surface = diagnosis.event_study_results["CallawaySantAnna"]
        draws, _ = design_module._placebo_draw_record(surface)
        honest = HonestDiD(method="smoothness")
        intervals = _intervals(honest, draws, 0.0)
        rate = np.mean((intervals[:, 0] > 0) | (intervals[:, 1] < 0))
        assert not design_module._size_controlled(rate, 60, 0, 0.05)
        # Relative magnitudes on the same draws passes and is reported.
        assert np.isfinite(_table(HonestDiD(), diagnosis, M_grid=[1.0])["mde"][0])


class TestLostReplicates:
    def test_fit_failures_before_retention_are_counted(self, panel, cs, monkeypatch):
        """A replicate the estimator failed on never becomes a draw, but it counts."""
        original = CallawaySantAnna.fit
        calls = {"n": 0}

        def flaky(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] % 4 == 0:
                raise ValueError("constructed failure")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(CallawaySantAnna, "fit", flaky)
        diagnosis = _diagnose(panel, [cs], n_simulations=40)
        monkeypatch.undo()
        null = diagnosis.results["CallawaySantAnna"][0]
        assert null.n_simulation_failures > 0.1 * 40
        surface = diagnosis.event_study_results["CallawaySantAnna"]
        draws, n_lost = design_module._placebo_draw_record(surface)
        assert n_lost == null.n_simulation_failures
        assert len(draws) == null.n_simulations

        with pytest.warns(UserWarning, match="fails the size check"):
            table = HonestDiD().power_by_M(diagnosis, M_grid=[1.0], progress=False)
        row = table.iloc[0]
        # Every retained draw is scored, and the row is still withheld.
        assert row["n_simulations"] == len(draws)
        assert row["n_simulation_failures"] == n_lost
        assert row["placebo_rejection_rate"] == 0.0
        assert np.isnan(row["mde"])

    def test_reduction_carries_the_lost_count(self):
        rows = ((-2, False), (-1, True), (0, False))
        provenance = ("CallawaySantAnnaResults", "universal", (-1,), "e0_first_treated")
        rng = np.random.default_rng(0)
        records = [
            ((rows, provenance), rng.normal(size=2), np.ones(2), object()) for _ in range(40)
        ]
        other = ((((-1, True), (0, False)), provenance), rng.normal(size=1), np.ones(1), object())
        surface, _checks, reason = design_module._placebo_event_study(
            CallawaySantAnna(), records + [other, None], 0.05, n_lost=3
        )
        assert reason is None
        draws, n_lost = design_module._placebo_draw_record(surface)
        assert len(draws) == 40
        assert all(draw is record[3] for draw, record in zip(draws, records))
        # 3 fits lost, one draw with another layout, one without an event study.
        assert n_lost == 5

    def test_records_without_a_container_keep_no_draws(self):
        rows = ((-2, False), (-1, True), (0, False))
        provenance = ("CallawaySantAnnaResults", "universal", (-1,), "e0_first_treated")
        rng = np.random.default_rng(0)
        records = [((rows, provenance), rng.normal(size=2), np.ones(2)) for _ in range(40)]
        surface, _checks, reason = design_module._placebo_event_study(
            CallawaySantAnna(), records, 0.05
        )
        assert reason is None
        with pytest.raises(ValueError, match="does not carry its placebo draws"):
            design_module._placebo_draw_record(surface)

    def _with_draws(self, surface, draws, n_lost=0):
        clone = dataclasses.replace(surface)
        setattr(clone, design_module._PLACEBO_DRAWS, {"draws": draws, "n_lost": n_lost})
        return clone

    def test_draws_the_analysis_fails_on_are_lost(self, cs_surface):
        draws, _ = design_module._placebo_draw_record(cs_surface)
        broken = dataclasses.replace(draws[0], vcov=np.zeros_like(draws[0].vcov))
        with pytest.raises(ValueError):
            HonestDiD().fit(broken)

        few = self._with_draws(cs_surface, [broken, broken] + draws[2:])
        row = _table(HonestDiD(), few, M_grid=[1.0]).iloc[0]
        assert row["n_simulations"] == 28 and row["n_simulation_failures"] == 2
        assert np.isfinite(row["mde"])

        many = self._with_draws(cs_surface, [broken] * 4 + draws[4:])
        with pytest.warns(UserWarning, match="4 of 30 replicates lost"):
            table = HonestDiD().power_by_M(many, M_grid=[1.0], progress=False)
        assert table["n_simulation_failures"][0] == 4 and np.isnan(table["mde"][0])

    def test_no_draw_scored_raises_with_the_draws_error(self, cs_surface):
        draws, _ = design_module._placebo_draw_record(cs_surface)
        broken = dataclasses.replace(draws[0], vcov=np.zeros_like(draws[0].vcov))
        none = self._with_draws(cs_surface, [broken] * 5)
        with pytest.raises(ValueError, match="could not be computed on any placebo draw"):
            HonestDiD().power_by_M(none, M_grid=[1.0], progress=False)

    def test_wrong_l_vec_length_raises_once(self, cs_diagnosis):
        honest = HonestDiD(l_vec=np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="l_vec must have length 3, got 2"):
            honest.power_by_M(cs_diagnosis, M_grid=[1.0], progress=False)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_cs(panel, cs):
    generator = design_module._make_placebo_generator(panel, "y", "id", "t", 10, [6], True)
    frame = generator(treatment_effect=1.0, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cs.fit(frame, **design_module._fit_kwargs(cs, "y", "id", "t", {}))


class TestRefusals:
    def test_fitted_model_is_refused(self, fitted_cs):
        for results in (fitted_cs, fitted_cs.aggregate("event_study")):
            with pytest.raises(TypeError, match="use sensitivity_analysis"):
                HonestDiD().power_by_M(results, progress=False)

    def test_combined_is_refused(self, cs_diagnosis):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            honest = HonestDiD(method="combined")
        with pytest.raises(ValueError, match="does not plan method='combined'"):
            honest.power_by_M(cs_diagnosis, progress=False)

    def test_diagnosis_without_the_keyword(self, panel, cs):
        diagnosis = _diagnose(panel, [cs], n_simulations=20, event_study=False)
        with pytest.raises(ValueError, match="event_study=True"):
            HonestDiD().power_by_M(diagnosis, progress=False)

    def test_diagnosis_with_several_surfaces(self, panel, cs):
        diagnosis = _diagnose(panel, [cs, StackedDiD(kappa_pre=3, kappa_post=1)])
        with pytest.raises(ValueError, match="several estimators"):
            HonestDiD().power_by_M(diagnosis, progress=False)
        table = _table(HonestDiD(), diagnosis.event_study_results["StackedDiD"], M_grid=[1.0])
        assert table["n_simulations"][0] == 30

    def test_surface_without_its_draws(self, cs_surface):
        rebuilt = dataclasses.replace(cs_surface)
        with pytest.raises(ValueError, match="does not carry its placebo draws"):
            HonestDiD().power_by_M(rebuilt, progress=False)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(M_grid=[]), "M_grid must be a non-empty list"),
            (dict(M_grid=[0.5, -1.0]), "M_grid must be a non-empty list"),
            (dict(M_grid=[np.nan]), "M_grid must be a non-empty list"),
            (dict(M_grid=[np.inf]), "M_grid must be a non-empty list"),
            (dict(M_grid=["a"]), "M_grid must be a list of numbers"),
            (dict(power=0.0), "power must be between 0 and 1"),
            (dict(power=1.0), "power must be between 0 and 1"),
            (dict(power=np.nan), "power must be between 0 and 1"),
            (dict(power="high"), "power must be a number"),
            (dict(power=True), "power must be a number"),
            (dict(effect_size=np.nan), "effect_size must be a finite number"),
            (dict(effect_size=np.inf), "effect_size must be a finite number"),
            (dict(effect_size="big"), "effect_size must be a finite number"),
            (dict(effect_size=True), "effect_size must be a finite number"),
        ],
    )
    def test_invalid_arguments(self, cs_diagnosis, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HonestDiD().power_by_M(cs_diagnosis, progress=False, **kwargs)

    def test_fitted_model_entry_points_still_refuse_a_placebo_input(self, cs_diagnosis, cs_surface):
        honest = HonestDiD()
        for call in (
            lambda: honest.fit(cs_surface),
            lambda: honest.sensitivity_analysis(cs_surface),
            lambda: honest.breakdown_value(cs_surface),
        ):
            with pytest.raises(ValueError, match="HonestDiD.power_by_M"):
                call()
        for results in (cs_diagnosis, cs_surface):
            with pytest.raises(ValueError, match="HonestDiD.power_by_M"):
                PreTrendsPower().sensitivity_to_honest_did(results)


# ---------------------------------------------------------------------------
# Warnings, and what retaining the draws leaves unchanged
# ---------------------------------------------------------------------------


class TestWarningsAndContract:
    def test_draw_warnings_are_issued_once(self, panel):
        """A varying base makes ``HonestDiD.fit`` warn on every draw."""
        diagnosis = _diagnose(panel, [CallawaySantAnna()])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            HonestDiD().power_by_M(diagnosis, M_grid=[0.5, 1.0], progress=False)
        base = [w for w in recorded if "universal" in str(w.message)]
        assert len(base) == 1
        message = str(base[0].message)
        assert message.startswith("[HonestDiD.power_by_M] ")
        assert "raised 60 times" in message

    def test_pickle_round_trip_keeps_the_draws(self, cs_diagnosis):
        restored = pickle.loads(pickle.dumps(cs_diagnosis))
        kwargs = dict(M_grid=[0.5, 1.0], effect_size=2.0)
        pd.testing.assert_frame_equal(
            _table(HonestDiD(), restored, **kwargs), _table(HonestDiD(), cs_diagnosis, **kwargs)
        )

    def test_surface_pickled_without_the_draws(self, cs_surface):
        """A surface from before the draws were kept gives a clear error."""
        state = dict(cs_surface.__dict__)
        del state[design_module._PLACEBO_DRAWS]
        old = dataclasses.replace(cs_surface)
        old.__dict__.clear()
        old.__dict__.update(state)
        restored = pickle.loads(pickle.dumps(old))
        with pytest.raises(ValueError, match="does not carry its placebo draws"):
            HonestDiD().power_by_M(restored, progress=False)
        # The pre-trend record is untouched, so pre-trend planning still works.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            planned = PreTrendsPower().power_by_pre_periods(restored)
        assert len(planned) == 4 and "placebo_rejection_rate" in planned.columns

    def test_draws_are_the_containers_of_the_placebo_fits(self, panel, cs, cs_surface):
        draws, n_lost = design_module._placebo_draw_record(cs_surface)
        by_hand = _replay(panel, cs, (6,), 30, 1, _constant(0.0))
        assert n_lost == 0 and len(draws) == len(by_hand)
        for kept, expected in zip(draws, by_hand):
            np.testing.assert_array_equal(kept.event_time, expected.event_time)
            np.testing.assert_allclose(kept.att, expected.att, rtol=1e-12, equal_nan=True)
            np.testing.assert_allclose(kept.vcov, expected.vcov, rtol=1e-12, atol=1e-15)
            assert kept.estimand != "placebo"

    def test_the_diagnosis_itself_is_unchanged(self, panel, cs, cs_diagnosis):
        plain = _diagnose(panel, [cs], event_study=False)
        assert plain.summary() != ""
        for with_draws, without in zip(
            cs_diagnosis.results["CallawaySantAnna"], plain.results["CallawaySantAnna"]
        ):
            assert with_draws.power == without.power
            assert with_draws.mean_estimate == without.mean_estimate
            assert with_draws.n_simulation_failures == without.n_simulation_failures
        assert cs_diagnosis.lock("CallawaySantAnna", panel, progress=False).to_dict() == (
            plain.lock("CallawaySantAnna", panel, progress=False).to_dict()
        )
        assert (cs_diagnosis == cs_diagnosis) is True

    def test_get_params_is_unchanged(self):
        assert set(HonestDiD().get_params()) == {"method", "M", "alpha", "l_vec"}


# ---------------------------------------------------------------------------
# The simulation numbers the registry quotes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sensitivity_evidence():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks/data/sensitivity_planning_check.json"
    if not path.exists():
        pytest.skip("benchmarks/data/sensitivity_planning_check.json is not available")
    return json.loads(path.read_text())


class TestSensitivityEvidence:
    """The numbers the REGISTRY ``## HonestDiD`` design-stage Notes quote."""

    @staticmethod
    def _rows(evidence, method=None, positive=None, units=None, reported=None):
        out = []
        for run in evidence["runs"]:
            for row in run["rows"]:
                if method is not None and row["method"] != method:
                    continue
                if positive is not None and (row["M"] > 0) is not positive:
                    continue
                if units is not None and run["n_units"] != units:
                    continue
                if reported is not None and row["reported"] is not reported:
                    continue
                out.append(row)
        return out

    def test_grid(self, sensitivity_evidence):
        meta = sensitivity_evidence["meta"]
        assert (meta["n_diagnosis"], meta["n_direct"]) == (500, 1000)
        assert (meta["n_runs"], meta["n_rows"]) == (18, 72)
        assert (meta["n_rows_reported"], meta["n_rows_withheld"]) == (60, 12)
        assert all(run["recorded"] for run in sensitivity_evidence["runs"])
        rows = self._rows(sensitivity_evidence)
        assert {row["fresh_no_effect_n"] for row in rows} == {1000}
        assert {row["observed_power_n"] for row in rows} == {1000}

    @pytest.mark.parametrize(
        "method, positive, units, n_reported, n_rows, in_sample, fresh",
        [
            ("smoothness", False, 40, 0, 9, (0.140, 0.452), (0.119, 0.435)),
            ("smoothness", False, 200, 6, 9, (0.046, 0.110), (0.047, 0.107)),
            ("smoothness", True, 40, 9, 9, (0.016, 0.042), (0.010, 0.040)),
            ("smoothness", True, 200, 9, 9, (0.000, 0.010), (0.002, 0.009)),
            ("relative_magnitude", None, 40, 18, 18, (0.000, 0.002), (0.000, 0.004)),
            ("relative_magnitude", None, 200, 18, 18, (0.000, 0.000), (0.000, 0.002)),
        ],
    )
    def test_level_table(
        self, sensitivity_evidence, method, positive, units, n_reported, n_rows, in_sample, fresh
    ):
        rows = self._rows(sensitivity_evidence, method, positive, units)
        assert len(rows) == n_rows
        assert sum(row["reported"] for row in rows) == n_reported
        counted = [row["placebo_rejection_rate"] for row in rows]
        assert (min(counted), max(counted)) == pytest.approx(in_sample, abs=5e-4)
        out_of_sample = [row["fresh_no_effect_rate"] for row in rows]
        assert (min(out_of_sample), max(out_of_sample)) == pytest.approx(fresh, abs=5e-4)

    def test_reported_smoothness_rows_at_200_units(self, sensitivity_evidence):
        rows = self._rows(sensitivity_evidence, "smoothness", False, 200, reported=True)
        counted = [row["placebo_rejection_rate"] for row in rows]
        fresh = [row["fresh_no_effect_rate"] for row in rows]
        assert (min(counted), max(counted)) == pytest.approx((0.046, 0.064), abs=5e-4)
        assert (min(fresh), max(fresh)) == pytest.approx((0.047, 0.094), abs=5e-4)
        # The one reported row whose fresh rate crosses the level flag.
        assert sum(row["flag_level"] for row in rows) == 1
        meta = sensitivity_evidence["meta"]
        assert meta["n_flag_level_reported"] == 1 and meta["n_flag_level_withheld"] == 11

    def test_planned_mde_against_real_injection(self, sensitivity_evidence):
        rows = self._rows(sensitivity_evidence)
        assert not any(row["flag_power"] for row in rows)
        powers = [row["observed_power"] for row in rows]
        assert (min(powers), max(powers)) == pytest.approx((0.764, 0.848), abs=5e-4)
        assert np.mean(powers) == pytest.approx(0.803, abs=1e-3)
        assert {row["injected_sign"] for row in rows} == {-1, 1}

    def test_monte_carlo_spread_of_the_planned_mde(self, sensitivity_evidence):
        spread = {200: [], 500: []}
        for row in sensitivity_evidence["stability"]:
            assert row["n_seeds"] == 10
            spread[row["n_simulations"]].append(row["relative_spread"])
        assert len(spread[200]) == len(spread[500]) == 6
        assert (min(spread[200]), max(spread[200])) == pytest.approx((0.015, 0.059), abs=5e-4)
        assert (min(spread[500]), max(spread[500])) == pytest.approx((0.014, 0.022), abs=6e-4)

    def test_seconds_per_row(self, sensitivity_evidence):
        slow = [row["seconds"] for row in self._rows(sensitivity_evidence, "smoothness", True)]
        assert 102 <= min(slow) and max(slow) <= 181
        fast = [
            row["seconds"]
            for row in self._rows(sensitivity_evidence)
            if not (row["method"] == "smoothness" and row["M"] > 0)
        ]
        assert max(fast) < 1.0
