"""Contract, guard and edge-case tests for the TWFE weight diagnostics.

R output parity lives in ``tests/test_twfe_weights_parity.py``; this module
covers the behaviour that is ours rather than R's - the input guards, the
result-object surface, and the design restrictions we enforce as errors.
"""

import numpy as np
import pandas as pd
import pytest

import diff_diff
from diff_diff.twfe_weights import attgt_weights


def _panel(seed=11, n_per_cohort=40, n_periods=5, cohorts=(0, 3, 4)):
    """Balanced staggered panel with a never-treated group."""
    rng = np.random.RandomState(seed)
    first_treat = np.repeat(np.array(cohorts), n_per_cohort)
    n_units = len(first_treat)
    unit_fe = rng.normal(size=n_units)
    rows = []
    for t in range(1, n_periods + 1):
        treated = (first_treat != 0) & (t >= first_treat)
        rows.append(
            pd.DataFrame(
                {
                    "unit": np.arange(n_units),
                    "period": t,
                    "first_treat": first_treat,
                    "outcome": (
                        unit_fe
                        + 0.5 * t
                        + 1.0 * treated * (t - first_treat + 1)
                        + rng.normal(scale=0.3, size=n_units)
                    ),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["unit", "period"])


def _fit(df, **kwargs):
    params = {"control_group": "never_treated", "base_period": "universal"}
    params.update(kwargs)
    return diff_diff.CallawaySantAnna(**params).fit(
        df, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
    )


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def fitted(panel):
    return _fit(panel)


class TestPublicSurface:
    def test_exported_from_package_root(self):
        assert diff_diff.attgt_weights is attgt_weights
        for name in ("attgt_weights", "ATTGTWeightsResult", "TWFEDecompositionResult"):
            assert name in diff_diff.__all__

    def test_name_is_distinct_from_the_dcdh_surface(self):
        """The two weight surfaces must stay separately addressable."""
        assert diff_diff.attgt_weights is not diff_diff.twowayfeweights
        assert diff_diff.ATTGTWeightsResult is not diff_diff.TWFEWeightsResult

    def test_result_is_a_diagnostic_without_the_quintet(self, fitted):
        result = attgt_weights(fitted)
        assert isinstance(result, diff_diff.Diagnostic)
        for banned in ("att", "se", "t_stat", "p_value", "conf_int"):
            assert not hasattr(result, banned)

    def test_result_renders(self, fitted):
        result = attgt_weights(fitted)
        text = result.summary()
        assert "Implicit Weights on ATT(g, t)" in text
        assert "TWFE regression" in text
        frame = result.to_dataframe()
        assert list(frame.columns) == ["group", "time", "post", "weight", "att"]
        # to_dataframe hands back a copy, not the live table
        frame.loc[0, "weight"] = 999.0
        assert result.weights.loc[0, "weight"] != 999.0
        assert set(result.to_dict()) >= {"aggregation", "implied_att", "weights"}
        assert "aggregation='twfe'" in repr(result)


class TestAggregationBehaviour:
    @pytest.mark.parametrize("aggregation", ["twfe", "overall", "simple"])
    def test_implied_att_is_the_weighted_sum(self, fitted, aggregation):
        result = attgt_weights(fitted, aggregation=aggregation)
        expected = (result.weights["weight"] * result.weights["att"]).sum()
        assert result.implied_att == pytest.approx(expected, abs=1e-14)

    @pytest.mark.parametrize("aggregation", ["overall", "simple"])
    def test_target_estimands_are_convex(self, fitted, aggregation):
        """ATT^O / ATT^simple weights are non-negative and sum to one."""
        weights = attgt_weights(fitted, aggregation=aggregation).weights["weight"]
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)

    def test_twfe_weights_can_be_negative(self, fitted):
        """The whole point of the diagnostic: staggered TWFE is not convex."""
        result = attgt_weights(fitted, aggregation="twfe")
        assert result.n_negative > 0
        assert 0.0 < result.negative_weight_share < 1.0
        assert "Negative-weight cells:" in result.summary()

    def test_pre_treatment_cells_carry_weight_under_twfe(self, fitted):
        """TWFE loads on pre-treatment cells; the CS estimands do not."""
        twfe = attgt_weights(fitted, aggregation="twfe").weights
        assert (twfe.loc[twfe["post"] == 0, "weight"].abs() > 0).any()
        for aggregation in ("overall", "simple"):
            benign = attgt_weights(fitted, aggregation=aggregation).weights
            assert (benign.loc[benign["post"] == 0, "weight"] == 0).all()

    def test_rejects_unknown_aggregation(self, fitted):
        with pytest.raises(ValueError, match="aggregation must be one of"):
            attgt_weights(fitted, aggregation="everything")


class TestDesignGuards:
    def test_rejects_non_universal_base_period_for_twfe(self, panel):
        fit = _fit(panel, base_period="varying")
        with pytest.raises(ValueError, match="base_period='universal'"):
            attgt_weights(fit, aggregation="twfe")

    def test_varying_base_is_fine_for_the_cs_estimands(self, panel):
        """Only the TWFE formula needs the complete grid."""
        fit = _fit(panel, base_period="varying")
        for aggregation in ("overall", "simple"):
            result = attgt_weights(fit, aggregation=aggregation)
            assert result.weights["weight"].sum() == pytest.approx(1.0, abs=1e-12)

    def test_rejects_not_yet_treated_control_for_twfe(self, panel):
        fit = _fit(panel, control_group="not_yet_treated")
        with pytest.raises(ValueError, match="control_group='never_treated'"):
            attgt_weights(fit, aggregation="twfe")

    def test_rejects_repeated_cross_sections(self, panel):
        # A true RCS needs one observation per unit id, so re-key the rows
        # rather than just flipping the flag (panel=False rejects duplicates).
        rcs = panel.copy().reset_index(drop=True)
        rcs["unit"] = np.arange(len(rcs))
        fit = _fit(rcs, panel=False)
        with pytest.raises(ValueError, match="requires a panel fit"):
            attgt_weights(fit)


class TestDataFrameFallback:
    def test_requires_the_full_panel_spec(self, fitted, panel):
        frame = fitted.to_dataframe("group_time")
        with pytest.raises(ValueError, match="missing"):
            attgt_weights(frame, data=panel, unit="unit")

    def test_rejects_panel_args_alongside_a_fitted_result(self, fitted, panel):
        with pytest.raises(ValueError, match="only for the DataFrame fallback"):
            attgt_weights(
                fitted,
                data=panel,
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_accepts_an_att_column_as_well_as_effect(self, fitted, panel):
        frame = fitted.to_dataframe("group_time")[["group", "time", "effect"]]
        via_effect = attgt_weights(
            frame, data=panel, unit="unit", time="period", first_treat="first_treat"
        )
        via_att = attgt_weights(
            frame.rename(columns={"effect": "att"}),
            data=panel,
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        np.testing.assert_allclose(
            via_effect.weights["weight"], via_att.weights["weight"], atol=1e-15
        )

    def test_rejects_a_frame_without_an_effect_column(self, panel):
        frame = pd.DataFrame({"group": [3], "time": [3]})
        with pytest.raises(ValueError, match="'effect' or 'att'"):
            attgt_weights(
                frame,
                data=panel,
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_rejects_time_varying_cohort_labels(self, fitted, panel):
        broken = panel.copy()
        broken.loc[broken.index[0], "first_treat"] = 99
        with pytest.raises(ValueError, match="varies within unit"):
            attgt_weights(
                fitted.to_dataframe("group_time"),
                data=broken,
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_source_is_recorded(self, fitted, panel):
        assert attgt_weights(fitted).source == "CallawaySantAnnaResults"
        from_frame = attgt_weights(
            fitted.to_dataframe("group_time"),
            data=panel,
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert from_frame.source == "DataFrame"


class TestNonConsecutiveTimeLabels:
    """Positional rescaling: gapped period labels must not change the weights."""

    def test_gapped_periods_match_consecutive_ones(self, panel):
        consecutive = attgt_weights(_fit(panel), aggregation="twfe")

        gapped = panel.copy()
        remap = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
        gapped["period"] = gapped["period"].map(remap)
        gapped["first_treat"] = gapped["first_treat"].map(lambda g: remap.get(g, 0))
        result = attgt_weights(_fit(gapped), aggregation="twfe")

        np.testing.assert_allclose(
            result.weights["weight"].to_numpy(),
            consecutive.weights["weight"].to_numpy(),
            atol=1e-14,
        )
        assert result.implied_att == pytest.approx(consecutive.implied_att, abs=1e-14)


class TestSamplingWeights:
    def test_uniform_weights_are_a_no_op(self, fitted, panel):
        baseline = attgt_weights(
            fitted.to_dataframe("group_time"),
            aggregation="overall",
            data=panel,
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        weighted_panel = panel.assign(w=1.0)
        weighted = attgt_weights(
            fitted.to_dataframe("group_time"),
            aggregation="overall",
            data=weighted_panel,
            unit="unit",
            time="period",
            first_treat="first_treat",
            weights="w",
        )
        np.testing.assert_allclose(
            baseline.weights["weight"], weighted.weights["weight"], atol=1e-15
        )

    def test_reweighting_a_cohort_shifts_its_weight(self, fitted, panel):
        """Doubling a cohort's sampling weight raises its share of ATT^O."""
        baseline = attgt_weights(
            fitted.to_dataframe("group_time"),
            aggregation="overall",
            data=panel,
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        tilted_panel = panel.assign(w=np.where(panel["first_treat"] == 3, 2.0, 1.0))
        tilted = attgt_weights(
            fitted.to_dataframe("group_time"),
            aggregation="overall",
            data=tilted_panel,
            unit="unit",
            time="period",
            first_treat="first_treat",
            weights="w",
        )
        mass_3_before = baseline.weights.query("group == 3")["weight"].sum()
        mass_3_after = tilted.weights.query("group == 3")["weight"].sum()
        assert mass_3_after > mass_3_before
        assert tilted.weights["weight"].sum() == pytest.approx(1.0, abs=1e-12)

    def test_rejects_time_varying_sampling_weights(self, fitted, panel):
        broken = panel.copy()
        broken["w"] = np.arange(len(broken), dtype=float)
        with pytest.raises(ValueError, match="must be time-invariant"):
            attgt_weights(
                fitted.to_dataframe("group_time"),
                data=broken,
                unit="unit",
                time="period",
                first_treat="first_treat",
                weights="w",
            )

    def test_rejects_a_weights_column_name_on_the_fitted_path(self, fitted):
        with pytest.raises(ValueError, match="only name a column"):
            attgt_weights(fitted, weights="w")


class TestDegenerateInputs:
    def test_rejects_a_panel_with_no_treated_units(self, panel):
        frame = pd.DataFrame({"group": [3.0], "time": [3.0], "effect": [1.0]})
        never = panel.assign(first_treat=0)
        with pytest.raises(ValueError, match="no ever-treated units"):
            attgt_weights(
                frame,
                data=never,
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_rejects_a_frame_with_no_finite_effects(self, panel):
        frame = pd.DataFrame({"group": [3.0, 3.0], "time": [3.0, 4.0], "effect": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="no finite effects"):
            attgt_weights(
                frame,
                data=panel,
                unit="unit",
                time="period",
                first_treat="first_treat",
            )

    def test_warns_and_renormalizes_when_cells_are_dropped(self, fitted, panel):
        frame = fitted.to_dataframe("group_time").copy()
        frame.loc[frame.index[0], "effect"] = np.nan
        with pytest.warns(UserWarning, match="had no estimable ATT"):
            result = attgt_weights(
                frame,
                aggregation="overall",
                data=panel,
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        assert result.n_dropped_cells == 1
        assert len(result.weights) == len(frame) - 1
        assert "Non-estimable cells dropped:" in result.summary()

    def test_rejects_a_cohort_label_off_the_period_grid(self, panel):
        frame = pd.DataFrame({"group": [3.0], "time": [3.0], "effect": [1.0]})
        broken = panel.copy()
        broken.loc[broken["first_treat"] == 4, "first_treat"] = 99
        with pytest.raises(ValueError, match="not one of the observed time periods"):
            attgt_weights(
                frame,
                data=broken,
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
