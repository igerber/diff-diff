"""Behavioural tests for diff_diff.design (diagnose_design)."""

import dataclasses
import itertools
import json
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DESIGN_LOCK_SCHEMA_VERSION,
    CallawaySantAnna,
    DesignDiagnosisResults,
    DesignLock,
    DesignReconciliationResults,
    Diagnostic,
    DifferenceInDifferences,
    EfficientDiD,
    ImputationDiD,
    StackedDiD,
    SunAbraham,
    SyntheticDiD,
    TwoStageDiD,
    TwoWayFixedEffects,
    diagnose_design,
)
from diff_diff.design import (
    _data_fingerprint,
    _encode_cell,
    _failed_cell,
    _json_value,
    _lock_digest,
    _make_placebo_generator,
    _size_controlled,
)
from diff_diff.power import SimulationPowerResults

POWER_COLUMNS = list(
    SimulationPowerResults(
        power=0.5,
        power_se=0.1,
        power_ci=(0.3, 0.7),
        rejection_rate=0.5,
        mean_estimate=0.0,
        std_estimate=1.0,
        mean_se=1.0,
        coverage=0.9,
        n_simulations=10,
        effect_sizes=[0.0],
        powers=[0.5],
        true_effect=0.0,
        alpha=0.05,
        estimator_name="X",
    )
    .to_dict()
    .keys()
)


def make_panel(n_units=20, n_periods=8, noise=1.0, seed=0, first_period=1):
    """Additive unit and period effects plus iid noise."""
    rng = np.random.default_rng(seed)
    unit_fe = rng.normal(size=n_units)
    time_fe = np.linspace(0.0, 1.0, n_periods)
    y = unit_fe[:, None] + time_fe[None, :] + noise * rng.normal(size=(n_units, n_periods))
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), n_periods),
            "period": np.tile(np.arange(first_period, first_period + n_periods), n_units),
            "y": y.ravel(),
        }
    )


def run(panel, estimators, **kwargs):
    kwargs.setdefault("cohort_periods", [6])
    if "n_treated" not in kwargs:
        kwargs.setdefault("treatment_fraction", 0.5)
    kwargs.setdefault("n_simulations", 20)
    kwargs.setdefault("seed", 11)
    kwargs.setdefault("progress", False)
    # Held-out periods are tested in TestHoldout; everywhere else the
    # comparison runs on the whole panel.
    kwargs.setdefault("n_holdout", 0)
    return diagnose_design(panel, "y", "unit", "period", estimators, **kwargs)


def estimates(result, label, cell=0):
    return np.array([r["estimate"] for r in result.results[label][cell].simulation_results])


ALL_EIGHT = [
    DifferenceInDifferences,
    TwoWayFixedEffects,
    CallawaySantAnna,
    SunAbraham,
    ImputationDiD,
    TwoStageDiD,
    StackedDiD,
    EfficientDiD,
]
STAGGERED_SIX = ALL_EIGHT[2:]


class TestRecovery:
    def test_all_eight_recover_injected_effect_common_timing(self):
        panel = make_panel(n_units=24, n_periods=8, noise=1e-3, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run(panel, [cls() for cls in ALL_EIGHT], effect_sizes=[2.0], n_simulations=4)
        assert list(result.results) == [cls.__name__ for cls in ALL_EIGHT]
        for label, cells in result.results.items():
            assert cells[0].n_simulations == 4, label
            assert abs(cells[0].mean_estimate) < 1e-2, label
            assert abs(cells[1].mean_estimate - 2.0) < 1e-2, label
            assert abs(cells[1].bias) < 1e-2, label

    def test_staggered_six_recover_injected_effect(self):
        panel = make_panel(n_units=30, n_periods=9, noise=1e-3, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run(
                panel,
                [cls() for cls in STAGGERED_SIX],
                cohort_periods=[5, 7],
                n_treated=14,
                effect_sizes=[1.5],
                n_simulations=4,
            )
        for label, cells in result.results.items():
            assert cells[1].n_simulations == 4, label
            assert abs(cells[1].mean_estimate - 1.5) < 1e-2, label

    def test_negative_effect_shifts_estimates_down(self):
        panel = make_panel(seed=3)
        result = run(panel, [DifferenceInDifferences()], effect_sizes=[-0.7])
        shift = estimates(result, "DifferenceInDifferences", 1) - estimates(
            result, "DifferenceInDifferences", 0
        )
        np.testing.assert_allclose(shift, -0.7, atol=1e-10)


class TestPlaceboDraws:
    def test_seed_reproducible(self):
        panel = make_panel(seed=4)
        first = run(panel, [DifferenceInDifferences(), CallawaySantAnna()], effect_sizes=[0.5])
        second = run(panel, [DifferenceInDifferences(), CallawaySantAnna()], effect_sizes=[0.5])
        pd.testing.assert_frame_equal(first.to_dataframe(), second.to_dataframe())
        assert first.seed == 11

    def test_seed_none_is_resolved_and_shared_across_estimators(self):
        panel = make_panel(seed=5)
        result = run(
            panel,
            {"a": DifferenceInDifferences(), "b": DifferenceInDifferences(cluster="unit")},
            seed=None,
        )
        assert isinstance(result.seed, int)
        # Same placebo draws for both labels: identical point estimates.
        np.testing.assert_allclose(estimates(result, "a"), estimates(result, "b"), atol=1e-12)
        again = run(
            panel,
            {"a": DifferenceInDifferences(), "b": DifferenceInDifferences(cluster="unit")},
            seed=result.seed,
        )
        np.testing.assert_allclose(estimates(result, "a"), estimates(again, "a"), atol=1e-12)

    def test_same_draws_across_effect_sizes(self):
        panel = make_panel(seed=6)
        result = run(panel, [DifferenceInDifferences()], effect_sizes=[0.25, 1.0])
        base = estimates(result, "DifferenceInDifferences", 0)
        np.testing.assert_allclose(
            estimates(result, "DifferenceInDifferences", 1) - base, 0.25, atol=1e-10
        )
        np.testing.assert_allclose(
            estimates(result, "DifferenceInDifferences", 2) - base, 1.0, atol=1e-10
        )

    def test_row_order_invariant(self):
        panel = make_panel(seed=7)
        shuffled = panel.sample(frac=1.0, random_state=0).reset_index(drop=True)
        first = run(panel, [TwoWayFixedEffects()])
        second = run(shuffled, [TwoWayFixedEffects()])
        np.testing.assert_allclose(
            estimates(first, "TwoWayFixedEffects"),
            estimates(second, "TwoWayFixedEffects"),
            atol=1e-10,
        )
        assert first.data_sha256 == second.data_sha256

    def test_fixed_panel_matches_exact_enumeration(self):
        """N=6, one date, no resampling: 20 possible assignments, all enumerable."""
        panel = make_panel(n_units=6, n_periods=6, noise=1.0, seed=8)
        estimator = DifferenceInDifferences(vcov_type="classical")
        enumerated = {}
        for treated in itertools.combinations(range(6), 3):
            frame = panel.copy()
            frame["g"] = frame["unit"].isin(treated).astype(int)
            frame["p"] = (frame["period"] >= 4).astype(int)
            fit = DifferenceInDifferences(vcov_type="classical").fit(
                frame, outcome="y", treatment="g", post="p"
            )
            enumerated[round(float(fit.att), 10)] = bool(fit.p_value < 0.05)
        exact_rate = float(np.mean(list(enumerated.values())))
        result = run(
            panel,
            [estimator],
            cohort_periods=[4],
            n_treated=3,
            resample_units=False,
            n_simulations=400,
        )
        cell = result.results["DifferenceInDifferences"][0]
        drawn = {round(float(e), 10) for e in estimates(result, "DifferenceInDifferences")}
        assert drawn <= set(enumerated)
        assert len(drawn) > 10
        rate_from_draws = np.mean(
            [enumerated[round(float(e), 10)] for e in estimates(result, "DifferenceInDifferences")]
        )
        assert cell.rejection_rate == pytest.approx(rate_from_draws)
        assert abs(cell.rejection_rate - exact_rate) < 0.1


class TestPlaceboGenerator:
    def test_resampled_duplicates_are_distinct_units(self):
        panel = make_panel(n_units=10, n_periods=6, seed=9)
        generator = _make_placebo_generator(panel, "y", "unit", "period", 4, [4], True)
        frame = generator(treatment_effect=0.0, seed=3)
        assert len(frame) == len(panel)
        assert not frame.duplicated(["unit", "period"]).any()
        assert frame["unit"].nunique() == 10
        # With replacement, some original series repeat.
        first_period = frame[frame["period"] == 1]["y"].round(12)
        assert first_period.nunique() < 10

    def test_common_timing_columns(self):
        panel = make_panel(n_units=10, n_periods=6, seed=10)
        generator = _make_placebo_generator(panel, "y", "unit", "period", 4, [4], False)
        null = generator(treatment_effect=0.0, seed=5)
        shifted = generator(treatment_effect=2.0, seed=5)
        by_unit = null.groupby("unit")["_placebo_treated"].agg(["min", "max"])
        assert (by_unit["min"] == by_unit["max"]).all()  # time-invariant group indicator
        assert by_unit["max"].sum() == 4
        np.testing.assert_array_equal(null["_placebo_post"], (null["period"] >= 4).astype(int))
        active = (null["_placebo_treated"] == 1) & (null["_placebo_post"] == 1)
        np.testing.assert_allclose((shifted["y"] - null["y"])[active], 2.0)
        np.testing.assert_allclose((shifted["y"] - null["y"])[~active], 0.0)
        np.testing.assert_allclose(null["y"], panel["y"])  # fixed panel left as observed

    def test_staggered_columns(self):
        panel = make_panel(n_units=40, n_periods=8, seed=11)
        generator = _make_placebo_generator(panel, "y", "unit", "period", 15, [4, 6, 7], True)
        null = generator(treatment_effect=0.0, seed=2)
        shifted = generator(treatment_effect=1.0, seed=2)
        assert "_placebo_post" not in null.columns
        first_treat = null.groupby("unit")["_placebo_first_treat"].agg(["min", "max"])
        assert (first_treat["min"] == first_treat["max"]).all()
        per_unit = first_treat["max"]
        assert (per_unit > 0).sum() == 15
        assert set(per_unit[per_unit > 0]) <= {4, 6, 7}
        assert len(set(per_unit[per_unit > 0])) > 1
        active = (null["_placebo_first_treat"] > 0) & (
            null["period"] >= null["_placebo_first_treat"]
        )
        np.testing.assert_allclose((shifted["y"] - null["y"])[active], 1.0)
        np.testing.assert_allclose((shifted["y"] - null["y"])[~active], 0.0)


class TestOutcomeBlindGuard:
    def test_truncates_at_earliest_real_adoption(self):
        panel = make_panel(n_units=20, n_periods=10, seed=12)
        panel["ft"] = np.where(panel["unit"] < 5, 8, np.where(panel["unit"] < 8, 9, 0))
        clean = panel[panel["period"] < 8]
        result = run(panel, [DifferenceInDifferences()], cohort_periods=[5], first_treat="ft")
        assert result.n_dropped == 20 * 3
        assert result.n_periods == 7
        assert result.first_treat == "ft"
        # Rows from the first real adoption on are never read.
        poisoned = panel.copy()
        poisoned.loc[poisoned["period"] >= 8, "y"] = np.nan
        again = run(poisoned, [DifferenceInDifferences()], cohort_periods=[5], first_treat="ft")
        np.testing.assert_allclose(
            estimates(result, "DifferenceInDifferences"),
            estimates(again, "DifferenceInDifferences"),
        )
        assert result.data_sha256 == again.data_sha256
        assert result.data_sha256 == _data_fingerprint(clean, "unit", "period")[0]

    @pytest.mark.parametrize("never", [0, np.inf, np.nan])
    def test_never_treated_codes_drop_nothing(self, never):
        panel = make_panel(seed=13)
        panel["ft"] = float(never)
        result = run(panel, [DifferenceInDifferences()], first_treat="ft")
        assert result.n_dropped == 0
        assert result.n_periods == 8

    @pytest.mark.parametrize("dtype", ["Int64", "Float64"])
    def test_nullable_first_treat_with_pd_na(self, dtype):
        """pd.NA marks never-treated units in a nullable column."""
        panel = make_panel(n_units=20, n_periods=10, seed=64)
        plain = panel.assign(ft=np.where(panel["unit"] < 5, 8.0, np.nan))
        nullable = panel.assign(
            ft=pd.array([8 if u < 5 else pd.NA for u in panel["unit"]], dtype=dtype)
        )
        assert nullable["ft"].isna().any()
        expected = run(plain, [DifferenceInDifferences()], cohort_periods=[5], first_treat="ft")
        result = run(nullable, [DifferenceInDifferences()], cohort_periods=[5], first_treat="ft")
        assert result.n_periods == 7  # periods 8 onward are dropped
        assert result.n_dropped == expected.n_dropped == 20 * 3
        np.testing.assert_allclose(
            estimates(result, "DifferenceInDifferences"),
            estimates(expected, "DifferenceInDifferences"),
        )

    def test_nullable_integer_calendar_column(self):
        panel = make_panel(seed=65)
        expected = run(panel, [DifferenceInDifferences(), CallawaySantAnna()])
        nullable = panel.assign(period=panel["period"].astype("Int64"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run(nullable, [DifferenceInDifferences(), CallawaySantAnna()])
        for label in ("DifferenceInDifferences", "CallawaySantAnna"):
            np.testing.assert_allclose(estimates(result, label), estimates(expected, label))

    def test_always_treated_unit_is_named(self):
        panel = make_panel(seed=14)
        panel["ft"] = np.where(panel["unit"] == 3, 1, 0)
        with pytest.raises(ValueError, match="always-treated"):
            run(panel, [DifferenceInDifferences()], first_treat="ft")

    def test_declared_untreated_is_reported_in_summary(self):
        result = run(make_panel(seed=15), [DifferenceInDifferences()])
        assert result.first_treat is None
        assert "declared by caller" in result.summary()

    def test_first_treat_must_be_constant_within_unit(self):
        panel = make_panel(seed=16)
        panel["ft"] = np.where(panel["period"] > 4, 7, 0)
        with pytest.raises(ValueError, match="constant within each unit"):
            run(panel, [DifferenceInDifferences()], first_treat="ft")


class TestRefusals:
    def test_unbalanced_panel(self):
        panel = make_panel(seed=17).iloc[1:]
        with pytest.raises(ValueError, match="balanced panel"):
            run(panel, [DifferenceInDifferences()])

    def test_missing_outcome(self):
        panel = make_panel(seed=18)
        panel.loc[3, "y"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            run(panel, [DifferenceInDifferences()])

    def test_non_numeric_time(self):
        panel = make_panel(seed=19)
        panel["period"] = panel["period"].astype(str)
        with pytest.raises(ValueError, match="must be numeric"):
            run(panel, [DifferenceInDifferences()], cohort_periods=["6"])

    def test_reserved_column(self):
        panel = make_panel(seed=20)
        panel["_placebo_post"] = 0
        with pytest.raises(ValueError, match="_placebo_"):
            run(panel, [DifferenceInDifferences()])

    @pytest.mark.parametrize(
        "cohort_periods, match",
        [
            ([], "empty"),
            ([6, 6], "duplicates"),
            ([99], "not a period"),
            ([2], "Too few periods"),
        ],
    )
    def test_cohort_periods(self, cohort_periods, match):
        with pytest.raises(ValueError, match=match):
            run(make_panel(seed=21), [CallawaySantAnna()], cohort_periods=cohort_periods)

    def test_non_positive_adoption_period(self):
        panel = make_panel(seed=22, first_period=-5)  # periods -5..2
        with pytest.raises(ValueError, match="not positive"):
            run(panel, [CallawaySantAnna()], cohort_periods=[0])

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"treatment_fraction": None, "n_treated": None}, "exactly one"),
            ({"treatment_fraction": 0.5, "n_treated": 3}, "exactly one"),
            ({"treatment_fraction": 0.01, "n_treated": None}, "at least one treated"),
            ({"treatment_fraction": None, "n_treated": 20}, "at least one treated"),
            ({"treatment_fraction": 1.5, "n_treated": None}, r"in \(0, 1\)"),
        ],
    )
    def test_treated_count(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            run(make_panel(seed=23), [DifferenceInDifferences()], **kwargs)

    def test_unsupported_estimator(self):
        with pytest.raises(ValueError, match="does not support"):
            run(make_panel(seed=24), [SyntheticDiD()])

    def test_duplicate_class_in_list_asks_for_mapping(self):
        with pytest.raises(ValueError, match="Pass a mapping"):
            run(
                make_panel(seed=25),
                [DifferenceInDifferences(), DifferenceInDifferences(cluster="unit")],
            )

    def test_alpha_mismatch(self):
        with pytest.raises(ValueError, match="alpha"):
            run(make_panel(seed=26), [DifferenceInDifferences(alpha=0.10)])

    def test_anticipation_refused(self):
        with pytest.raises(ValueError, match="anticipation"):
            run(make_panel(seed=27), [CallawaySantAnna(anticipation=1)])

    def test_other_cluster_column_refused_when_resampling(self):
        panel = make_panel(seed=60)
        panel["unit_copy"] = panel["unit"]
        with pytest.raises(ValueError, match="clusters on column 'unit_copy'"):
            run(panel, [DifferenceInDifferences(cluster="unit_copy")])
        with pytest.raises(ValueError, match="clusters on column 'unit_copy'"):
            run(panel, [CallawaySantAnna(cluster="unit_copy")])

    def test_other_cluster_column_allowed_on_the_fixed_panel(self):
        """Without resampling the panel's clusters are as observed, so an
        equivalent cluster column gives identical inference."""
        panel = make_panel(seed=61)
        panel["unit_copy"] = panel["unit"]
        result = run(
            panel,
            {
                "by_unit": DifferenceInDifferences(cluster="unit"),
                "by_copy": DifferenceInDifferences(cluster="unit_copy"),
            },
            resample_units=False,
        )
        a, b = result.results["by_unit"][0], result.results["by_copy"][0]
        assert a.rejection_rate == b.rejection_rate
        assert a.mean_se == pytest.approx(b.mean_se, rel=1e-12)

    def test_unit_cluster_allowed_when_resampling_whatever_the_column_is_called(self):
        panel = make_panel(seed=62).rename(columns={"unit": "store"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = diagnose_design(
                panel,
                "y",
                "store",
                "period",
                # StackedDiD's cluster="unit" is a mode, not a column name.
                {"did": DifferenceInDifferences(cluster="store"), "stacked": StackedDiD()},
                cohort_periods=[6],
                n_holdout=0,
                treatment_fraction=0.5,
                n_simulations=5,
                seed=1,
                progress=False,
            )
        for cells in result.results.values():
            assert cells[0].n_simulations == 5

    def test_common_timing_estimator_under_staggered_declaration(self):
        with pytest.raises(ValueError, match="one common adoption period"):
            run(make_panel(seed=28), [TwoWayFixedEffects()], cohort_periods=[5, 6])

    @pytest.mark.parametrize(
        "key",
        ["data", "outcome", "unit", "time", "first_treat", "treatment", "post", "formula"]
        + ["post_periods", "survey_design"],
    )
    def test_refused_fit_keywords(self, key):
        with pytest.raises(ValueError, match="may not contain"):
            run(make_panel(seed=29), [DifferenceInDifferences()], estimator_kwargs={key: "x"})

    def test_fit_keyword_not_accepted_by_one_estimator(self):
        with pytest.raises(ValueError, match="not accepted by TwoWayFixedEffects.fit"):
            run(
                make_panel(seed=30),
                [DifferenceInDifferences(), TwoWayFixedEffects()],
                estimator_kwargs={"absorb": ["unit"]},
            )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_simulations": 0}, "positive integer"),
            ({"n_simulations": 2.5}, "positive integer"),
            ({"alpha": 0.0}, "alpha"),
            ({"power": 0.05}, "alpha < power < 1"),
            ({"power": 1.0}, "alpha < power < 1"),
            ({"effect_sizes": [np.inf]}, "finite"),
        ],
    )
    def test_scalar_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            run(make_panel(seed=31), [DifferenceInDifferences()], **kwargs)


def _stub_fit(monkeypatch, behaviour):
    """Replace TwoWayFixedEffects.fit with a call-counted wrapper.

    ``behaviour(call_index, result_factory)`` returns the fit result (or
    raises). Call order is deterministic: the pre-flight fit(s), then
    n_simulations fits per cell in effect order, then the MDE search.
    """
    original = TwoWayFixedEffects.fit
    calls = {"n": 0}

    def fit(self, data, **kwargs):
        calls["n"] += 1
        return behaviour(calls["n"], lambda: original(self, data, **kwargs))

    monkeypatch.setattr(TwoWayFixedEffects, "fit", fit)
    return calls


def _with_nan_inference(result):
    result.se = np.nan
    result.p_value = np.nan
    result.conf_int = (np.nan, np.nan)
    return result


class TestFailurePolicy:
    estimators = staticmethod(
        lambda: {"did": DifferenceInDifferences(), "twfe": TwoWayFixedEffects()}
    )

    def test_nan_inference_after_preflight_is_excluded_and_counted(self, monkeypatch):
        _stub_fit(monkeypatch, lambda i, make: make() if i == 1 else _with_nan_inference(make()))
        with pytest.warns(UserWarning):
            result = run(make_panel(seed=32), self.estimators(), n_simulations=15, power=0.8)
        cell = result.results["twfe"][0]
        assert cell.n_simulations == 0
        assert cell.n_simulation_failures == 15
        assert np.isnan(cell.rejection_rate)
        assert result.n_nan_inference["twfe"] == [15]
        assert result.size_controlled["twfe"] is False
        assert np.isnan(result.mde["twfe"])
        assert "twfe" not in result.mde_results
        assert result.results["did"][0].n_simulations == 15  # the other estimator completes

    def test_some_nan_replicates_leave_rates_over_the_rest(self, monkeypatch):
        # Calls 2..6 are the first five replicates of the effect-0 cell.
        _stub_fit(
            monkeypatch,
            lambda i, make: _with_nan_inference(make()) if 2 <= i <= 6 else make(),
        )
        with pytest.warns(UserWarning):
            result = run(make_panel(seed=33), self.estimators(), n_simulations=20)
        cell = result.results["twfe"][0]
        assert cell.n_simulations == 15
        assert cell.n_simulation_failures == 5
        assert result.n_nan_inference["twfe"] == [5]
        assert np.isfinite(cell.mean_estimate)
        assert result.size_controlled["twfe"] is False  # 25% lost > 10%

    def test_value_error_after_preflight_is_a_failed_row(self, monkeypatch):
        def behaviour(i, make):
            if i == 1:
                return make()
            raise ValueError("boom on this draw")

        _stub_fit(monkeypatch, behaviour)
        with pytest.warns(UserWarning):
            result = run(make_panel(seed=34), self.estimators(), n_simulations=12)
        cell = result.results["twfe"][0]
        assert cell.n_simulations == 0
        assert cell.n_simulation_failures == 12
        assert result.n_nan_inference["twfe"] == [0]
        assert result.size_controlled["twfe"] is False
        assert result.results["did"][0].n_simulations == 12

    def test_value_error_on_every_call_is_refused_by_preflight(self, monkeypatch):
        def behaviour(i, make):
            raise ValueError("cannot fit this design")

        calls = _stub_fit(monkeypatch, behaviour)
        with pytest.raises(ValueError, match="'twfe'.*cannot fit this design"):
            run(make_panel(seed=35), self.estimators())
        assert calls["n"] == 5  # all pre-flight attempts, nothing else

    def test_nan_inference_on_every_call_is_refused_by_preflight(self, monkeypatch):
        _stub_fit(monkeypatch, lambda i, make: _with_nan_inference(make()))
        with pytest.raises(ValueError, match="non-finite inference"):
            run(make_panel(seed=36), self.estimators())

    def test_type_error_propagates(self, monkeypatch):
        def behaviour(i, make):
            if i == 1:
                return make()
            raise TypeError("a bug, not a data problem")

        _stub_fit(monkeypatch, behaviour)
        with pytest.raises(TypeError, match="a bug"):
            run(make_panel(seed=37), self.estimators())

    def test_failures_after_the_null_cell_leave_size_intact(self, monkeypatch):
        n = 12

        def behaviour(i, make):
            if i > 1 + n:  # pre-flight + effect-0 cell succeed
                raise ValueError("fails once an effect is injected")
            return make()

        _stub_fit(monkeypatch, behaviour)
        with pytest.warns(UserWarning):
            result = run(
                make_panel(seed=38), self.estimators(), n_simulations=n, effect_sizes=[1.0]
            )
        null, shifted = result.results["twfe"]
        assert null.n_simulations == n and null.n_simulation_failures == 0
        assert shifted.n_simulations == 0 and shifted.n_simulation_failures == n
        assert np.isfinite(null.rejection_rate)

    def test_lost_fits_during_mde_search_give_nan_mde(self, monkeypatch):
        n = 30

        def behaviour(i, make):
            # After the pre-flight and the effect-0 cell, lose every third fit.
            if i > 1 + n and i % 3 == 0:
                raise ValueError("lost during the search")
            return make()

        _stub_fit(monkeypatch, behaviour)
        panel = make_panel(n_units=30, n_periods=8, seed=39)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run(panel, self.estimators(), n_simulations=n, power=0.8)
        messages = [str(w.message) for w in caught]
        assert any(m.startswith("[twfe] MDE not reported") for m in messages)
        assert result.results["twfe"][0].n_simulation_failures == 0
        assert result.size_controlled["twfe"]  # the gate looks at effect 0 only
        assert np.isnan(result.mde["twfe"])
        assert "twfe" in result.mde_results  # the search ran; its value is withheld
        assert result.size_controlled["did"] and np.isfinite(result.mde["did"])

    def test_event_study_keyword_is_refused_by_preflight(self):
        with pytest.raises(ValueError, match="could not be fitted"):
            run(
                make_panel(seed=40),
                [TwoWayFixedEffects()],
                estimator_kwargs={"event_study": True},
            )

    def test_stacked_did_on_gapped_calendar_is_refused(self):
        panel = make_panel(n_units=20, n_periods=8, seed=41)
        panel["period"] = panel["period"] * 2  # 2, 4, ..., 16: no period before adoption
        with pytest.raises(ValueError, match="'StackedDiD'.*could not be fitted"):
            run(panel, [StackedDiD()], cohort_periods=[10])


class TestSizeGate:
    def test_bound(self):
        alpha, n = 0.05, 400
        bound = alpha + 1.96 * np.sqrt(alpha * (1 - alpha) / n)
        assert _size_controlled(bound - 1e-9, n, 0, alpha) is True
        assert _size_controlled(bound, n, 0, alpha) is True
        assert _size_controlled(bound + 1e-9, n, 0, alpha) is False
        assert _size_controlled(0.0, n, 0, alpha) is True

    def test_bound_uses_scored_count(self):
        # 0.09 passes with 100 scored replicates but not with 10,000.
        assert _size_controlled(0.09, 100, 0, 0.05) is True
        assert _size_controlled(0.09, 10_000, 0, 0.05) is False

    def test_lost_share_boundary(self):
        assert _size_controlled(0.05, 90, 10, 0.05) is True  # exactly 10% lost
        assert _size_controlled(0.05, 89, 11, 0.05) is False  # 11% lost
        assert _size_controlled(0.0, 80, 20, 0.05) is False

    def test_nothing_scored(self):
        assert _size_controlled(float("nan"), 0, 50, 0.05) is False
        assert _size_controlled(0.0, 0, 0, 0.05) is False


class TestFingerprint:
    def frame(self):
        return pd.DataFrame(
            {
                "unit": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "y": [0.1, 0.2, 0.3, 0.4],
                "tag": ["a", "b", "c", "d"],
            }
        )

    def test_deterministic_and_order_invariant(self):
        frame = self.frame()
        digest, columns = _data_fingerprint(frame, "unit", "period")
        assert columns == ["period", "tag", "unit", "y"]
        assert _data_fingerprint(frame, "unit", "period")[0] == digest
        reordered = frame.iloc[[3, 0, 2, 1]][["tag", "y", "unit", "period"]]
        assert _data_fingerprint(reordered, "unit", "period")[0] == digest

    def test_changes_with_any_value(self):
        frame = self.frame()
        digest = _data_fingerprint(frame, "unit", "period")[0]
        changed = frame.copy()
        changed.loc[2, "y"] = 0.3000000001
        assert _data_fingerprint(changed, "unit", "period")[0] != digest
        changed = frame.copy()
        changed.loc[1, "tag"] = "B"
        assert _data_fingerprint(changed, "unit", "period")[0] != digest

    def test_distinguishes_types(self):
        def digest(values):
            frame = self.frame()
            frame["tag"] = pd.Series(values, dtype=object)
            return _data_fingerprint(frame, "unit", "period")[0]

        base = ["a", "b", "c", "d"]
        assert digest(["", "b", "c", "d"]) != digest([None, "b", "c", "d"])
        assert digest(["", "b", "c", "d"]) != digest([np.nan, "b", "c", "d"])
        assert digest([True, "b", "c", "d"]) != digest([1, "b", "c", "d"])
        # Numbers are encoded by value, not by storage type.
        assert digest([1, "b", "c", "d"]) == digest([1.0, "b", "c", "d"])
        assert digest([1, "b", "c", "d"]) != digest([1.5, "b", "c", "d"])
        assert digest([1, "b", "c", "d"]) != digest(["1", "b", "c", "d"])
        assert digest([(1, 2), "b", "c", "d"]) != digest(["(1, 2)", "b", "c", "d"])
        assert digest(base) == digest(list(base))

    def test_column_labels_equal_as_strings_are_refused(self):
        frame = self.frame()
        frame[1] = [1.0, 2.0, 3.0, 4.0]
        frame["1"] = [5.0, 6.0, 7.0, 8.0]
        with pytest.raises(ValueError, match="equal as strings"):
            _data_fingerprint(frame, "unit", "period")
        panel = make_panel(seed=63)
        panel[1] = 0.0
        panel["1"] = 1.0
        with pytest.raises(ValueError, match="equal as strings"):
            run(panel, [DifferenceInDifferences()])

    def test_not_fooled_by_punctuation_inside_values(self):
        def digest(first, second):
            frame = self.frame().iloc[:2].copy()
            frame["tag"] = [first, second]
            frame["tag2"] = ["x", "y"]
            return _data_fingerprint(frame, "unit", "period")[0]

        assert digest('a"],["s","b', "c") != digest("a", 'b"],["s","c')
        assert digest("a\nb", "c") != digest("a", "b\nc")


class _Chatty:
    """Stub behaviour: warn on every fit, with a different message each time."""

    def __call__(self, i, make):
        warnings.warn(f"chatty estimator, call {i}", UserWarning)
        return make()


class TestWarnings:
    estimators = staticmethod(
        lambda: {"did": DifferenceInDifferences(), "twfe": TwoWayFixedEffects()}
    )

    def test_aggregated_and_labelled(self, monkeypatch):
        _stub_fit(monkeypatch, _Chatty())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run(make_panel(seed=42), self.estimators(), n_simulations=10)
        messages = [str(w.message) for w in caught if "chatty" in str(w.message)]
        assert messages and all(m.startswith("[twfe] ") for m in messages)
        # One per block (pre-flight, the single cell), not one per fit.
        assert len(messages) == 2
        assert any("raised 10 times" in m for m in messages)
        assert result.results["twfe"][0].n_simulation_failures == 0

    def test_caller_ignore_filter_does_not_lose_replicates(self, monkeypatch):
        _stub_fit(monkeypatch, _Chatty())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore")
            result = run(make_panel(seed=43), self.estimators(), n_simulations=10)
        assert caught == []
        assert result.results["twfe"][0].n_simulations == 10

    def test_caller_error_filter_raises_the_labelled_warning(self, monkeypatch):
        _stub_fit(monkeypatch, _Chatty())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Raised at re-emission (labelled), not from inside a fit, and the
            # estimator is not refused by the pre-flight because of it.
            with pytest.raises(UserWarning, match=r"^\[twfe\] chatty estimator, call 1$"):
                run(make_panel(seed=44), self.estimators(), n_simulations=10)

    def test_unseeded_bootstrap_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run(
                make_panel(seed=45),
                {"cs": CallawaySantAnna(n_bootstrap=19)},
                n_simulations=3,
            )
        assert [w for w in caught if "[cs] uses bootstrap inference" in str(w.message)]

    def test_seeded_bootstrap_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run(
                make_panel(seed=46),
                {"cs": CallawaySantAnna(n_bootstrap=19, seed=1)},
                n_simulations=3,
            )
        assert not [w for w in caught if "not reproducible" in str(w.message)]


@pytest.fixture(scope="module")
def contract_result():
    return run(
        make_panel(seed=47),
        {
            "did": DifferenceInDifferences(),
            "did_clustered": DifferenceInDifferences(cluster="unit"),
        },
        effect_sizes=[0.5, 0.5, 0.0, 1.0],
        n_simulations=15,
    )


class TestResultsContract:
    @pytest.fixture
    def result(self, contract_result):
        return contract_result

    def test_is_a_diagnostic_without_the_estimator_quintet(self, result):
        assert isinstance(result, DesignDiagnosisResults)
        assert isinstance(result, Diagnostic)
        for name in ("att", "se", "t_stat", "p_value", "conf_int"):
            assert not hasattr(result, name)

    def test_effect_sizes_are_the_ones_run(self, result):
        assert result.effect_sizes == [0.0, 0.5, 1.0]
        for cells in result.results.values():
            assert [cell.true_effect for cell in cells] == [0.0, 0.5, 1.0]

    def test_dataframe_layout(self, result):
        frame = result.to_dataframe()
        assert list(frame.columns) == (
            ["label"] + POWER_COLUMNS + ["size_controlled", "n_nan_inference", "mde"]
        )
        assert list(frame["label"]) == ["did"] * 3 + ["did_clustered"] * 3
        assert set(frame["estimator_name"]) == {"DifferenceInDifferences"}
        assert list(frame["true_effect"]) == [0.0, 0.5, 1.0] * 2
        # The requested count is scored + failed on every row.
        assert ((frame["n_simulations"] + frame["n_simulation_failures"]) == 15).all()

    def test_rejection_rate_at_zero_is_the_size(self, result):
        null = result.results["did"][0]
        assert null.true_effect == 0.0
        assert null.rejection_rate == null.power
        assert 0.0 <= null.rejection_rate <= 1.0

    def test_to_dict(self, result):
        payload = result.to_dict()
        assert payload["estimators"]["did_clustered"] == {
            "estimator_name": "DifferenceInDifferences",
            "params": DifferenceInDifferences(cluster="unit").get_params(),
        }
        assert payload["n_treated"] == 10
        assert payload["cohort_periods"] == [6]
        assert len(payload["results"]) == 6
        assert payload["data_sha256"] == result.data_sha256

    def test_estimators_field_holds_unfitted_copies(self, result):
        copy = result.estimators["did_clustered"]
        assert isinstance(copy, DifferenceInDifferences)
        assert copy.get_params()["cluster"] == "unit"
        assert not getattr(copy, "is_fitted_", False)

    def test_summary_lists_every_estimator_in_input_order(self, result):
        text = result.summary()
        assert text.index("did ") < text.index("did_clustered")
        assert "not ranked" in text

    def test_progress_prints_one_line_per_cell(self, capsys):
        run(
            make_panel(seed=48),
            [DifferenceInDifferences()],
            effect_sizes=[0.5],
            n_simulations=5,
            progress=True,
        )
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "diagnose_design" in ln]
        assert len(lines) == 2


class TestMDE:
    def test_mde_reported_only_for_size_controlled_estimators(self, monkeypatch):
        import diff_diff.design as design

        monkeypatch.setattr(
            design,
            "_size_controlled",
            lambda rate, n_scored, n_lost, alpha: False,
        )
        result = run(make_panel(seed=49), [DifferenceInDifferences()], power=0.8)
        assert np.isnan(result.mde["DifferenceInDifferences"])
        assert result.mde_results == {}
        assert result.target_power == 0.8

    def test_mde_is_on_the_outcome_scale(self):
        # Same panel in two units: the MDE scales with the outcome.
        panel = make_panel(n_units=30, n_periods=8, seed=50)
        scaled = panel.assign(y=panel["y"] * 1000.0)
        kwargs = dict(
            estimators=[DifferenceInDifferences(cluster="unit")],
            n_simulations=60,
            power=0.8,
        )
        small = run(panel, **kwargs)
        large = run(scaled, **kwargs)
        label = "DifferenceInDifferences"
        assert small.size_controlled[label]
        assert np.isfinite(small.mde[label]) and small.mde[label] > 0
        assert large.mde[label] == pytest.approx(1000.0 * small.mde[label], rel=1e-6)
        found = small.mde_results[label]
        assert found.estimator_name == label
        assert found.power_at_mde >= 0.8

    def test_no_mde_without_power(self):
        result = run(make_panel(seed=51), [DifferenceInDifferences()])
        assert result.target_power is None
        assert np.isnan(result.mde["DifferenceInDifferences"])


class TestConfigurations:
    @pytest.mark.parametrize("estimator_class", [DifferenceInDifferences, TwoWayFixedEffects])
    @pytest.mark.parametrize("lag_cutoff", [0, 1])
    @pytest.mark.parametrize("resample_units", [True, False])
    def test_conley_candidates_are_refused(self, estimator_class, lag_cutoff, resample_units):
        """Their Conley calendar would be the 0/1 post indicator, not the period."""
        panel = make_panel(n_units=20, n_periods=8, seed=52)
        rng = np.random.default_rng(0)
        coords = pd.DataFrame(
            {
                "unit": np.arange(20),
                "lat": rng.uniform(30, 45, size=20),
                "lon": rng.uniform(-120, -80, size=20),
            }
        )
        panel = panel.merge(coords, on="unit")
        estimator = estimator_class(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=500.0,
            conley_lag_cutoff=lag_cutoff,
        )
        with pytest.raises(ValueError, match="vcov_type='conley'"):
            run(panel, [estimator], resample_units=resample_units)

    def test_fit_keyword_reaches_every_estimator(self):
        panel = make_panel(seed=53)
        panel["x"] = np.random.default_rng(1).normal(size=len(panel))
        result = run(
            panel,
            [DifferenceInDifferences(), TwoWayFixedEffects()],
            estimator_kwargs={"covariates": ["x"]},
            n_simulations=5,
        )
        assert result.estimator_kwargs == {"covariates": ["x"]}
        for cells in result.results.values():
            assert cells[0].n_simulations == 5


# ---------------------------------------------------------------------------
# Held-out periods, the design lock and reconcile
# ---------------------------------------------------------------------------

NUMERIC = ["power", "rejection_rate", "mean_estimate", "bias", "rmse", "mean_se", "coverage"]


def lock_seed(seed):
    return int(np.random.SeedSequence([seed, 1]).generate_state(1)[0])


def relock(lock, **changes):
    """A lock with some fields replaced and its sha256 recomputed."""
    fields = {k: v for k, v in lock.to_dict().items() if k != "sha256"}
    fields.update(changes)
    return DesignLock(**fields, sha256=_lock_digest(fields))


def deviating(reconciliation):
    return {c["item"] for c in reconciliation.checks if c["status"] == "deviates"}


def statuses(reconciliation):
    return {c["item"]: c["status"] for c in reconciliation.checks}


def analysis_frame(panel, n_new=3, cohorts=((9, range(10)),)):
    """The diagnosed panel plus ``n_new`` later periods and a real adoption column."""
    last = int(panel["period"].max())
    units = np.sort(panel["unit"].unique())
    rng = np.random.default_rng(5)
    later = pd.DataFrame(
        {
            "unit": np.repeat(units, n_new),
            "period": np.tile(np.arange(last + 1, last + 1 + n_new), len(units)),
            "y": rng.normal(size=len(units) * n_new),
        }
    )
    frame = pd.concat([panel[["unit", "period", "y"]], later], ignore_index=True)
    frame["g"] = 0
    for period, members in cohorts:
        frame.loc[frame["unit"].isin(list(members)), "g"] = period
    return frame


class TestHoldout:
    def test_default_is_zero_for_one_estimator(self):
        result = run(make_panel(seed=70), [DifferenceInDifferences()], n_holdout=None)
        assert result.n_holdout == 0
        assert result.n_periods == 8
        assert result.time_periods == list(range(1, 9))

    def test_default_is_half_the_post_window_for_several(self):
        panel = make_panel(n_periods=12, seed=71)
        result = run(
            panel,
            [DifferenceInDifferences(), TwoWayFixedEffects()],
            cohort_periods=[7],
            n_holdout=None,
            n_simulations=5,
        )
        # Periods 7..12 are at or after adoption: 6 // 2 = 3 held out.
        assert result.n_holdout == 3
        assert result.n_periods == 9
        assert result.n_obs == 20 * 9
        assert result.time_periods == list(range(1, 10))
        assert result.n_dropped == 0
        assert result.to_dict()["n_holdout"] == 3
        assert result.to_dict()["time_periods"] == list(range(1, 10))
        assert "Held-out periods:" in result.summary()

    def test_default_raises_when_adoption_is_the_final_period(self):
        with pytest.raises(ValueError, match="n_holdout=0"):
            run(
                make_panel(seed=72),
                [DifferenceInDifferences(), TwoWayFixedEffects()],
                cohort_periods=[8],
                n_holdout=None,
            )

    @pytest.mark.parametrize("value", [3, -1])
    def test_out_of_range_is_refused(self, value):
        # Adoption at 6 on periods 1..8: at most 2 can be held out.
        with pytest.raises(ValueError, match="out of range"):
            run(make_panel(seed=73), [DifferenceInDifferences()], n_holdout=value)

    @pytest.mark.parametrize("value", [True, 1.0, "1"])
    def test_non_integer_is_refused(self, value):
        with pytest.raises(ValueError, match="non-negative integer"):
            run(make_panel(seed=73), [DifferenceInDifferences()], n_holdout=value)

    def test_numpy_integer_and_single_estimator_are_honoured(self):
        result = run(make_panel(seed=74), [DifferenceInDifferences()], n_holdout=np.int64(2))
        assert result.n_holdout == 2
        assert result.time_periods == list(range(1, 7))

    @pytest.mark.parametrize("integer_outcome", [False, True])
    def test_held_out_rows_are_not_read(self, integer_outcome):
        panel = make_panel(n_periods=10, seed=75)
        if integer_outcome:
            panel["y"] = np.round(panel["y"] * 10).astype(int)
        estimators = {"did": DifferenceInDifferences(), "twfe": TwoWayFixedEffects()}
        kwargs = dict(cohort_periods=[6], n_holdout=2, effect_sizes=[1.0], n_simulations=8)
        base = run(panel, estimators, **kwargs)

        changed = panel.copy()
        held_out = changed["period"] > 8
        changed.loc[held_out, "y"] = changed.loc[held_out, "y"] + 1000
        missing = panel.copy()
        missing["y"] = missing["y"].astype(float)
        missing.loc[held_out, "y"] = np.nan  # an integer column is now stored as float

        for other_panel in (changed, missing):
            other = run(other_panel, estimators, **kwargs)
            assert other.data_sha256 == base.data_sha256
            pd.testing.assert_frame_equal(other.to_dataframe(), base.to_dataframe())

    def test_numbers_are_fingerprinted_by_value(self):
        assert _encode_cell(1) == _encode_cell(1.0) == _encode_cell(np.int64(1))
        assert _encode_cell(1.5) != _encode_cell(1)
        assert _encode_cell(True) != _encode_cell(1)
        assert _encode_cell("1") != _encode_cell(1)
        assert _encode_cell(np.inf) != _encode_cell(-np.inf)

    def test_truncation_happens_before_the_held_out_split(self):
        panel = make_panel(n_periods=10, seed=76)
        panel["ft"] = np.where(panel["unit"] < 5, 9, 0)
        estimators = {"did": DifferenceInDifferences(), "twfe": TwoWayFixedEffects()}
        result = run(
            panel,
            estimators,
            cohort_periods=[5],
            first_treat="ft",
            n_holdout=None,
            n_simulations=6,
        )
        # Truncated to periods 1..8; periods 5..8 follow adoption: 2 held out.
        assert result.n_dropped == 2 * 20
        assert result.n_holdout == 2
        assert result.time_periods == list(range(1, 7))
        lock = result.lock("did", data=panel, progress=False)
        assert lock.time_periods == list(range(1, 9))
        assert lock.n_periods == 8
        assert lock.first_treat == "ft"

    def test_unlockable_parameter_warns_before_the_run(self):
        estimator = CallawaySantAnna(epv_threshold=np.inf)
        with pytest.warns(UserWarning, match="cannot be recorded in a design lock"):
            result = run(make_panel(seed=77), {"cs": estimator}, n_simulations=3)
        with pytest.raises(ValueError, match=r"params\.epv_threshold"):
            result.lock()


def diagnosed(seed=80, **kwargs):
    panel = make_panel(n_periods=10, seed=seed)
    kwargs.setdefault("cohort_periods", [6])
    kwargs.setdefault("n_holdout", None)
    kwargs.setdefault("effect_sizes", [1.0])
    kwargs.setdefault("n_simulations", 12)
    estimators = kwargs.pop(
        "estimators", {"did": DifferenceInDifferences(), "twfe": TwoWayFixedEffects()}
    )
    return panel, run(panel, estimators, **kwargs)


class TestLock:
    def test_held_out_run_needs_data(self):
        _, result = diagnosed()
        assert result.n_holdout == 2
        with pytest.raises(ValueError, match="pass data="):
            result.lock("did")

    def test_label_rules(self):
        panel, result = diagnosed()
        with pytest.raises(ValueError, match="pass the label"):
            result.lock(data=panel)
        with pytest.raises(ValueError, match="Unknown label"):
            result.lock("nope", data=panel)
        single = run(make_panel(seed=81), [DifferenceInDifferences()], n_simulations=5)
        assert single.lock().label == "DifferenceInDifferences"

    def test_refuses_a_panel_that_was_not_diagnosed(self):
        panel, result = diagnosed()
        changed = panel.copy()
        changed.loc[(changed["unit"] == 3) & (changed["period"] == 2), "y"] += 1.0
        with pytest.raises(ValueError, match="not the panel that was diagnosed"):
            result.lock("did", data=changed)

    def test_refuses_extra_or_missing_periods(self):
        panel, result = diagnosed()
        extra = pd.concat(
            [panel, panel[panel["period"] == 10].assign(period=11)], ignore_index=True
        )
        with pytest.raises(ValueError, match=r"other periods found: \[9, 10, 11\]"):
            result.lock("did", data=extra)
        with pytest.raises(ValueError, match="periods of the diagnosed panel"):
            result.lock("did", data=panel[panel["period"] != 10])
        with pytest.raises(ValueError, match=r"Compared periods missing: \[4\]"):
            result.lock("did", data=panel[panel["period"] != 4])

    def test_held_out_run_is_a_plain_run_on_the_full_panel(self):
        panel, result = diagnosed()
        lock = result.lock("did", data=panel, progress=False)
        expected = run(
            panel,
            {"did": DifferenceInDifferences()},
            cohort_periods=[6],
            effect_sizes=[1.0],
            n_simulations=12,
            seed=lock_seed(result.seed),
        )
        np.testing.assert_allclose(
            pd.DataFrame(lock.holdout_results)[NUMERIC].to_numpy(dtype=float),
            expected.to_dataframe()[NUMERIC].to_numpy(dtype=float),
            rtol=1e-12,
        )
        assert lock.n_holdout == 2
        assert lock.n_periods == 10 and lock.n_obs == 200
        assert lock.time_periods == list(range(1, 11))
        assert lock.data_sha256 == expected.data_sha256
        assert lock.data_sha256 != result.data_sha256
        assert lock.size_controlled == expected.size_controlled["did"]
        # The comparison rows of every estimator are kept alongside.
        assert [row["label"] for row in lock.comparison_results] == ["did", "did", "twfe", "twfe"]
        pd.testing.assert_frame_equal(lock.to_dataframe(), pd.DataFrame(lock.holdout_results))
        # The locked rows are a different run from the comparison rows.
        compared = result.to_dataframe().query("label == 'did'")[NUMERIC].to_numpy(dtype=float)
        assert not np.allclose(compared, lock.to_dataframe()[NUMERIC].to_numpy(dtype=float))

    def test_only_the_chosen_estimator_is_fitted(self, monkeypatch):
        panel, result = diagnosed()
        calls = {"twfe": 0}
        original = TwoWayFixedEffects.fit

        def counting(self, *args, **kwargs):
            calls["twfe"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(TwoWayFixedEffects, "fit", counting)
        result.lock("did", data=panel, progress=False)
        assert calls["twfe"] == 0

    def test_relocking_is_reproducible_and_a_second_label_warns(self):
        panel, result = diagnosed()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = result.lock("did", data=panel, progress=False)
            second = result.lock("did", data=panel, progress=False)
        assert first == second
        assert not [w for w in caught if "already used" in str(w.message)]
        with pytest.warns(UserWarning, match="already used for \\['did'\\]"):
            result.lock("twfe", data=panel, progress=False)

    def test_progress_prints_one_line_per_effect(self, capsys):
        panel, result = diagnosed()
        result.lock("did", data=panel)
        lines = [line for line in capsys.readouterr().out.splitlines() if "lock: did" in line]
        assert len(lines) == 2

    def test_size_failing_estimator_warns_and_is_recorded(self):
        _, result = diagnosed(n_holdout=0)
        result.size_controlled["did"] = False
        with pytest.warns(UserWarning, match="did not pass the size check"):
            lock = result.lock("did")
        assert lock.size_controlled is False

    def test_estimator_changed_after_the_diagnosis_is_refused(self):
        _, result = diagnosed(n_holdout=0)
        result.estimators["did"].set_params(cluster="unit")
        with pytest.raises(ValueError, match="changed after the diagnosis"):
            result.lock("did")

    def test_without_held_out_periods_no_data_is_needed(self):
        panel = make_panel(seed=82)
        # A calendar with gaps cannot be rebuilt from a period count.
        panel["period"] = panel["period"].map(dict(zip(range(1, 9), [1, 2, 4, 5, 7, 8, 10, 11])))
        result = run(panel, [DifferenceInDifferences()], cohort_periods=[8], n_simulations=6)
        lock = result.lock(progress=False)
        assert lock.n_holdout == 0
        assert lock.holdout_results == []
        assert lock.time_periods == [1, 2, 4, 5, 7, 8, 10, 11]
        assert lock.data_sha256 == result.data_sha256
        assert "not confirmed on held-out periods" in lock.summary()
        pd.testing.assert_frame_equal(
            lock.to_dataframe()[NUMERIC], result.to_dataframe()[NUMERIC], check_dtype=False
        )
        # data is optional here, but it is still verified when given.
        assert result.lock(data=panel) == lock
        changed = panel.copy()
        changed.loc[0, "y"] += 1.0
        with pytest.raises(ValueError, match="not the panel that was diagnosed"):
            result.lock(data=changed)

    def test_non_string_column_label_locks_and_reconciles(self):
        panel, _ = diagnosed()
        panel[1] = 0.5
        result = run(
            panel,
            {"did": DifferenceInDifferences(), "twfe": TwoWayFixedEffects()},
            cohort_periods=[6],
            n_holdout=2,
            n_simulations=5,
        )
        assert result.data_columns == ["1", "period", "unit", "y"]
        lock = result.lock("did", data=panel, progress=False)
        reconciliation = lock.reconcile(
            DifferenceInDifferences(), outcome="y", unit="unit", time="period", data=panel
        )
        assert statuses(reconciliation)["data_sha256"] == "match"

    def test_contract(self):
        panel, result = diagnosed()
        lock = result.lock("did", data=panel, progress=False)
        assert isinstance(lock, Diagnostic)
        for name in ("att", "se", "t_stat", "p_value", "conf_int"):
            assert not hasattr(lock, name)
        with pytest.raises(dataclasses.FrozenInstanceError):
            lock.label = "twfe"
        assert lock.label == "did" and lock.estimator_name == "DifferenceInDifferences"
        assert lock.params == _json_value(DifferenceInDifferences().get_params(), "params")
        assert lock.schema_version == DESIGN_LOCK_SCHEMA_VERSION == "1.0"
        text = lock.summary()
        assert "did (DifferenceInDifferences)" in text and lock.sha256 in text
        assert "held-out run" in text
        assert repr(lock).startswith("DesignLock(label='did'")


class TestLockSerialization:
    def lock(self, **kwargs):
        panel, result = diagnosed(**kwargs)
        return result.lock("did", data=panel, progress=False)

    def test_json_round_trip(self):
        lock = self.lock()
        text = lock.to_json()
        assert DesignLock.from_json(text) == lock
        assert DesignLock.from_dict(lock.to_dict()) == lock
        payload = json.loads(text)
        assert payload["schema_version"] == DESIGN_LOCK_SCHEMA_VERSION
        assert set(payload) == {f.name for f in dataclasses.fields(DesignLock)}
        # to_dict returns copies.
        lock.to_dict()["params"]["alpha"] = 0.5
        assert lock.params["alpha"] == 0.05

    def test_edited_file_fails_the_hash(self):
        payload = self.lock().to_dict()
        payload["n_treated"] += 1
        with pytest.raises(ValueError, match="does not match its sha256"):
            DesignLock.from_dict(payload)
        with pytest.raises(ValueError, match="does not match its sha256"):
            DesignLock.from_json(json.dumps(payload))

    def test_unknown_missing_and_unsupported(self):
        payload = self.lock().to_dict()
        with pytest.raises(ValueError, match=r"unknown key\(s\) \['extra'\]"):
            DesignLock.from_dict({**payload, "extra": 1})
        with pytest.raises(ValueError, match=r"missing key\(s\) \['seed'\]"):
            DesignLock.from_dict({k: v for k, v in payload.items() if k != "seed"})
        with pytest.raises(ValueError, match="schema version '2.0' is not supported"):
            DesignLock.from_dict({**payload, "schema_version": "2.0"})
        with pytest.raises(TypeError, match="dictionary"):
            DesignLock.from_dict([1, 2])

    def test_unreported_mde_is_none(self):
        lock = self.lock()  # no power= target, so no MDE search ran
        assert lock.mde is None
        assert DesignLock.from_json(lock.to_json()).mde is None
        assert all(row["mde"] is None for row in lock.comparison_results)

    def test_reported_mde_round_trips(self):
        panel = make_panel(n_units=40, n_periods=10, noise=0.2, seed=83)
        result = run(
            panel,
            {"did": DifferenceInDifferences(cluster="unit"), "twfe": TwoWayFixedEffects()},
            cohort_periods=[6],
            n_holdout=None,
            n_simulations=30,
            power=0.8,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lock = result.lock("twfe", data=panel, progress=False)
        if lock.size_controlled:
            assert lock.mde is not None and lock.mde > 0
        assert DesignLock.from_json(lock.to_json()) == lock

    def test_cell_that_could_not_be_scored_round_trips(self):
        _, result = diagnosed(n_holdout=0)
        result.results["did"][1] = _failed_cell(1.0, 0.05, "DifferenceInDifferences", 12)
        lock = result.lock("did")
        failed = lock.comparison_results[1]
        for name in ("power", "power_ci_lower", "power_ci_upper", "bias", "rmse", "coverage"):
            assert failed[name] is None
        assert failed["n_simulations"] == 0 and failed["n_simulation_failures"] == 12
        assert DesignLock.from_json(lock.to_json()) == lock
        assert "-" in lock.summary()

    def test_callable_parameter_is_refused(self):
        estimator = DifferenceInDifferences(conley_metric=lambda a, b: np.zeros((len(a), len(b))))
        with pytest.warns(UserWarning, match="cannot be recorded in a design lock"):
            result = run(make_panel(seed=84), {"did": estimator}, n_simulations=3)
        with pytest.raises(ValueError, match=r"params\.conley_metric holds a function"):
            result.lock()

    def test_editing_a_lock_in_place_is_caught(self):
        lock = self.lock()
        lock.params["cluster"] = "unit"
        with pytest.raises(ValueError, match="edited after it was made"):
            lock.to_estimator()
        with pytest.raises(ValueError, match="edited after it was made"):
            lock.reconcile(DifferenceInDifferences(), outcome="y", unit="unit", time="period")


class TestToEstimator:
    @pytest.mark.parametrize(
        "estimator",
        [
            DifferenceInDifferences(cluster="unit"),
            TwoWayFixedEffects(),
            CallawaySantAnna(control_group="not_yet_treated"),
            SunAbraham(),
            ImputationDiD(),
            TwoStageDiD(),
            StackedDiD(kappa_pre=2),
            EfficientDiD(),
        ],
        ids=lambda est: type(est).__name__,
    )
    def test_rebuilds_the_locked_estimator(self, estimator):
        _, result = diagnosed(n_holdout=0, n_simulations=3)
        params = _json_value(estimator.get_params(), "params")
        lock = relock(
            result.lock("did", progress=False),
            estimator_name=type(estimator).__name__,
            params=params,
        )
        built = lock.to_estimator()
        assert type(built) is type(estimator)
        assert not getattr(built, "is_fitted_", False)
        assert _json_value(built.get_params(), "params") == params
        reconciliation = lock.reconcile(built, outcome="y", unit="unit", time="period")
        assert reconciliation.status == "pre-specified"
        assert not deviating(reconciliation)

    def test_unsupported_class_name_is_refused(self):
        _, result = diagnosed(n_holdout=0, n_simulations=3)
        lock = relock(result.lock("did", progress=False), estimator_name="SyntheticDiD")
        with pytest.raises(ValueError, match="not one of"):
            lock.to_estimator()


@pytest.fixture(scope="module")
def staggered():
    """A diagnosed panel and the lock of a staggered-API estimator on it."""
    panel = make_panel(seed=90)
    lock = run(
        panel, {"cs": CallawaySantAnna()}, cohort_periods=[6], n_treated=10, n_simulations=4
    ).lock(progress=False)
    return panel, lock


@pytest.fixture(scope="module")
def two_group():
    """An analysis frame with 0/1 group and post columns, and a DiD lock."""
    panel = make_panel(seed=91)
    lock = run(
        panel,
        {"did": DifferenceInDifferences()},
        cohort_periods=[6],
        n_treated=10,
        n_simulations=4,
    ).lock(progress=False)
    frame = analysis_frame(panel)
    frame["treated"] = (frame["unit"] < 10).astype(int)
    frame["post"] = (frame["period"] >= 9).astype(int)
    return frame, lock


class TestReconcile:
    DATA = ("data_sha256", "n_units", "is_balanced", "n_treated", "n_cohorts", "n_post_by_cohort")

    def reconcile(self, lock, estimator=None, **kwargs):
        estimator = CallawaySantAnna() if estimator is None else estimator
        kwargs.setdefault("outcome", "y")
        kwargs.setdefault("unit", "unit")
        kwargs.setdefault("time", "period")
        return lock.reconcile(estimator, **kwargs)

    def test_matching_analysis_is_pre_specified(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel)
        estimator = lock.to_estimator()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = estimator.fit(frame, outcome="y", unit="unit", time="period", first_treat="g")
        reconciliation = self.reconcile(lock, estimator, first_treat="g", data=frame, results=fit)
        assert reconciliation.status == "pre-specified"
        assert set(statuses(reconciliation).values()) == {"match"}
        assert reconciliation.label == "cs"
        assert reconciliation.lock_sha256 == lock.sha256
        assert isinstance(reconciliation, DesignReconciliationResults)
        assert isinstance(reconciliation, Diagnostic)

    def test_never_treated_coded_inf_is_not_a_cohort(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel)
        frame["g"] = frame["g"].astype(float).replace(0.0, np.inf)
        reconciliation = self.reconcile(lock, first_treat="g", data=frame)
        assert reconciliation.status == "pre-specified"

    def test_estimator_class(self, staggered):
        _, lock = staggered
        reconciliation = self.reconcile(lock, SunAbraham())
        assert reconciliation.status == "deviates"
        assert "estimator_name" in deviating(reconciliation)

    def test_one_parameter(self, staggered):
        _, lock = staggered
        reconciliation = self.reconcile(lock, CallawaySantAnna(control_group="not_yet_treated"))
        assert deviating(reconciliation) == {"params.control_group"}
        assert reconciliation.status == "deviates"

    def test_fit_keyword(self, staggered):
        _, lock = staggered
        reconciliation = self.reconcile(lock, estimator_kwargs={"covariates": ["x"]})
        assert deviating(reconciliation) == {"estimator_kwargs.covariates"}

    def test_column_name(self, staggered):
        _, lock = staggered
        assert deviating(self.reconcile(lock, outcome="sales")) == {"outcome"}

    def test_alpha_on_results(self, staggered):
        _, lock = staggered
        estimator = CallawaySantAnna()
        estimator.results_ = types.SimpleNamespace(alpha=0.10)
        reconciliation = self.reconcile(lock, estimator, results=estimator.results_)
        assert deviating(reconciliation) == {"alpha"}
        assert statuses(self.reconcile(lock))["alpha"] == "not checked"

    def test_results_of_another_estimator_are_refused(self, staggered):
        _, lock = staggered
        with pytest.raises(ValueError, match="not this estimator's results_"):
            self.reconcile(lock, results=types.SimpleNamespace(alpha=0.05))

    def test_changed_pre_period_data(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel)
        frame.loc[(frame["unit"] == 4) & (frame["period"] == 3), "y"] += 1.0
        reconciliation = self.reconcile(lock, first_treat="g", data=frame)
        assert deviating(reconciliation) == {"data_sha256"}

    def test_unit_count(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel)
        frame = frame[frame["unit"] != 19]
        reconciliation = self.reconcile(lock, first_treat="g", data=frame)
        # The dropped unit's diagnosed rows are gone too.
        assert deviating(reconciliation) == {"n_units", "data_sha256"}

    def test_treated_count(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel, cohorts=((9, range(9)),))
        assert deviating(self.reconcile(lock, first_treat="g", data=frame)) == {"n_treated"}

    def test_cohort_count(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel, cohorts=((9, range(5)), (10, range(5, 10))))
        reconciliation = self.reconcile(lock, first_treat="g", data=frame)
        # One more cohort also means one more post window.
        assert deviating(reconciliation) == {"n_cohorts", "n_post_by_cohort"}
        rows = {c["item"]: c for c in reconciliation.checks}
        assert rows["n_post_by_cohort"]["locked"] == [3]
        assert rows["n_post_by_cohort"]["realized"] == [3, 2]

    def test_post_window_length(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel, n_new=4)
        reconciliation = self.reconcile(lock, first_treat="g", data=frame)
        assert deviating(reconciliation) == {"n_post_by_cohort"}

    def test_lost_follow_up_shows_as_unbalanced(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel)
        frame = frame[~((frame["unit"] < 10) & (frame["period"] == 11))]
        reconciliation = self.reconcile(lock, first_treat="g", data=frame)
        assert deviating(reconciliation) == {"is_balanced"}

    def test_adoption_varying_within_unit(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel)
        frame.loc[(frame["unit"] == 0) & (frame["period"] < 5), "g"] = 0
        reconciliation = self.reconcile(lock, first_treat="g", data=frame)
        assert deviating(reconciliation) == {"n_treated", "n_cohorts", "n_post_by_cohort"}
        rows = {c["item"]: c for c in reconciliation.checks}
        assert rows["n_treated"]["realized"] == "varies within unit"

    def test_missing_columns_deviate_and_do_not_raise(self, staggered):
        panel, lock = staggered
        frame = analysis_frame(panel)
        reconciliation = self.reconcile(lock, first_treat="nope", data=frame)
        assert deviating(reconciliation) == {"n_treated", "n_cohorts", "n_post_by_cohort"}
        reconciliation = self.reconcile(lock, data=frame.drop(columns="y"))
        assert deviating(reconciliation) == {"data_sha256"}
        reconciliation = self.reconcile(lock, unit="store", data=frame)
        assert deviating(reconciliation) == {"unit", "n_units", "is_balanced"}

    def test_what_is_not_checked(self, staggered):
        panel, lock = staggered
        without_data = statuses(self.reconcile(lock))
        assert all(without_data[item] == "not checked" for item in self.DATA)
        assert self.reconcile(lock).status == "pre-specified"
        with_data = statuses(self.reconcile(lock, data=analysis_frame(panel)))
        for item in ("n_treated", "n_cohorts", "n_post_by_cohort"):
            assert with_data[item] == "not checked"
        for item in ("data_sha256", "n_units", "is_balanced"):
            assert with_data[item] == "match"

    def test_integer_outcome_stored_as_float_still_matches(self):
        panel = make_panel(seed=92)
        panel["y"] = np.round(panel["y"] * 10).astype(int)
        lock = run(panel, [DifferenceInDifferences()], n_simulations=3).lock(progress=False)
        frame = analysis_frame(panel)
        frame.loc[frame["period"] > 8, "y"] = np.nan
        assert frame["y"].dtype == float
        reconciliation = self.reconcile(lock, DifferenceInDifferences(), data=frame)
        assert statuses(reconciliation)["data_sha256"] == "match"

    def test_unserializable_realized_parameter_deviates(self, staggered, two_group):
        _, lock = staggered
        reconciliation = self.reconcile(lock, CallawaySantAnna(epv_threshold=np.inf))
        assert deviating(reconciliation) == {"params.epv_threshold"}
        rows = {c["item"]: c for c in reconciliation.checks}
        assert rows["params.epv_threshold"]["realized"] == "inf"
        _, did_lock = two_group
        estimator = DifferenceInDifferences(conley_metric=lambda a, b: None)
        assert deviating(self.reconcile(did_lock, estimator)) == {"params.conley_metric"}

    def test_two_group_design(self, two_group):
        frame, lock = two_group
        estimator = DifferenceInDifferences()
        reconciliation = self.reconcile(
            lock, estimator, treatment="treated", post="post", data=frame
        )
        assert reconciliation.status == "pre-specified"
        assert not [s for s in statuses(reconciliation).values() if s == "deviates"]
        without_post = statuses(self.reconcile(lock, estimator, treatment="treated", data=frame))
        assert without_post["n_post_by_cohort"] == "not checked"
        assert without_post["n_treated"] == "match"

    def test_two_group_structure(self, two_group):
        frame, lock = two_group
        estimator = DifferenceInDifferences()
        varying = frame.copy()
        varying.loc[(varying["unit"] == 0) & (varying["period"] < 9), "treated"] = 0
        reconciliation = self.reconcile(
            lock, estimator, treatment="treated", post="post", data=varying
        )
        assert deviating(reconciliation) == {"n_treated", "n_cohorts"}
        staggered_post = frame.copy()
        staggered_post.loc[
            (staggered_post["unit"] == 0) & (staggered_post["period"] == 9), "post"
        ] = 0
        reconciliation = self.reconcile(
            lock, estimator, treatment="treated", post="post", data=staggered_post
        )
        assert deviating(reconciliation) == {"n_post_by_cohort"}
        rows = {c["item"]: c for c in reconciliation.checks}
        assert rows["n_post_by_cohort"]["realized"] == "varies within period"

    def test_argument_rules(self, two_group):
        frame, lock = two_group
        with pytest.raises(ValueError, match="not both"):
            self.reconcile(lock, first_treat="g", treatment="treated", data=frame)
        with pytest.raises(TypeError, match="estimator instance"):
            self.reconcile(lock, "DifferenceInDifferences")
        with pytest.raises(TypeError, match="DataFrame"):
            self.reconcile(lock, DifferenceInDifferences(), data=[1, 2])

    def test_results_contract(self, staggered):
        _, lock = staggered
        reconciliation = self.reconcile(lock, CallawaySantAnna(control_group="not_yet_treated"))
        frame = reconciliation.to_dataframe()
        assert list(frame.columns) == ["item", "locked", "realized", "status"]
        assert len(frame) == len(reconciliation.checks)
        assert frame.loc[0, "item"] == "estimator_name"
        as_dict = reconciliation.to_dict()
        assert as_dict["status"] == "deviates" and as_dict["lock_sha256"] == lock.sha256
        json.dumps(as_dict)
        text = reconciliation.summary()
        assert "deviates" in text and "params.control_group" in text
        assert "1 deviate" in text
        assert repr(reconciliation).startswith("DesignReconciliationResults(status='deviates'")
        for name in ("att", "se", "t_stat", "p_value", "conf_int"):
            assert not hasattr(reconciliation, name)
