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
        assert "Negative POST-period cells:" in result.summary()
        assert result.n_negative_post <= result.n_negative

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


class TestWeightedRegressionPin:
    """Frozen-numbers pin for the WEIGHTED branches.

    No parity fixture passes ``weights=`` and R ``twfe_weights`` has no ``w=``,
    so the weighted code paths (weighted two-way demeaning, weighted FWL solve,
    weighted cohort masses) have no external oracle. These literals were
    captured from the implementation BEFORE the linear algebra was routed
    through ``diff_diff.linalg.solve_ols`` / ``diff_diff.utils.within_transform``
    and pin that behaviour: any refactor must leave them green at 1e-12.
    """

    @staticmethod
    def _weighted_panel():
        rng = np.random.default_rng(20260907)
        n_per, n_periods = 12, 5
        cohorts = [0] * n_per + [3] * n_per + [4] * n_per
        rows = []
        for i, g in enumerate(cohorts):
            w = float(rng.choice([0.5, 1.0, 1.5, 2.5]))
            alpha = rng.normal()
            for t in range(1, n_periods + 1):
                x = rng.normal() + 0.3 * t
                effect = 1.0 * (t - g + 1) if (g and t >= g) else 0.0
                y = alpha + 0.2 * t + 0.5 * x + effect + rng.normal(scale=0.3)
                rows.append({"id": i, "t": t, "g": g, "y": y, "x": x, "w": w})
        return pd.DataFrame(rows)

    _DEC = {
        "nocov": dict(
            kwargs={},
            estimate=1.4221735897240102,
            pretrend_bias=0.5127724996498023,
            post_only=0.9094010900742079,
            ess=59.999999999999986,
            weight=[
                -0.26383763837638374,
                -0.26383763837638374,
                0.3726937269372693,
                0.07749077490774903,
                0.07749077490774903,
                -0.059040590405904064,
                -0.059040590405904064,
                -0.3542435424354244,
                0.23616236162361626,
                0.23616236162361626,
            ],
            att=[
                0.0,
                -1.4436560985156102,
                0.2605332041282682,
                1.5423748763069138,
                2.020457826078349,
                0.0,
                -0.8817595960499504,
                -0.22533107108782402,
                1.0544057969164409,
                1.2161310006314463,
            ],
        ),
        "cov": dict(
            kwargs={"covariates": ["x"]},
            estimate=1.3930847792561663,
            pretrend_bias=0.5397671816632773,
            post_only=0.8533175975928889,
            ess=58.816020983744,
            weight=[
                -0.26642873140338846,
                -0.26217358219812914,
                0.3731356281678529,
                0.07703988099301252,
                0.07842680444065203,
                -0.05851961726129342,
                -0.05862391113360511,
                -0.35425415800358373,
                0.23433448557874187,
                0.23706320081974044,
            ],
            att=[
                0.0,
                -1.4333600946359863,
                0.1470326297519402,
                1.502553090163623,
                1.9646825183908976,
                0.0,
                -0.8405505765908995,
                -0.32378354968013,
                1.0394622134176261,
                1.2023475510925157,
            ],
        ),
        "gmin1": dict(
            kwargs={"base_period": "gmin1"},
            estimate=1.4221735897240106,
            pretrend_bias=-0.3554385674608017,
            post_only=1.777612157184812,
            ess=59.999999999999986,
            weight=None,  # identical to nocov (weights do not depend on the base period)
            att=[
                1.4436560985156102,
                0.0,
                1.7041893026438784,
                2.986030974822525,
                3.4641139245939594,
                0.2253310710878238,
                -0.6564285249621264,
                0.0,
                1.279736868004265,
                1.4414620717192705,
            ],
        ),
    }

    @pytest.mark.parametrize("key", ["nocov", "cov", "gmin1"])
    def test_decomposition_weighted_branches(self, key):
        spec = self._DEC[key]
        df = self._weighted_panel()
        result = diff_diff.decompose_twfe_weights(
            df, outcome="y", unit="id", time="t", first_treat="g", weights="w", **spec["kwargs"]
        )
        assert result.estimate == pytest.approx(spec["estimate"], abs=1e-12)
        assert result.pretrend_bias == pytest.approx(spec["pretrend_bias"], abs=1e-12)
        assert result.post_only == pytest.approx(spec["post_only"], abs=1e-12)
        assert result.effective_sample_size == pytest.approx(spec["ess"], abs=1e-9)
        expected_w = spec["weight"] if spec["weight"] is not None else self._DEC["nocov"]["weight"]
        np.testing.assert_allclose(result.cells["weight"].to_numpy(), expected_w, atol=1e-12)
        np.testing.assert_allclose(result.cells["att"].to_numpy(), spec["att"], atol=1e-12)

    _AGG = {
        "twfe": (
            1.4348838554104435,
            [
                -0.2638376383763837,
                -0.2638376383763837,
                0.3726937269372694,
                0.0774907749077491,
                0.0774907749077491,
                -0.059040590405904085,
                -0.059040590405904085,
                -0.3542435424354242,
                0.2361623616236162,
                0.2361623616236162,
            ],
        ),
        "overall": (
            2.101822892918353,
            [
                0,
                0,
                0.17874396135265702,
                0.17874396135265702,
                0.17874396135265702,
                0,
                0,
                0,
                0.2318840579710145,
                0.2318840579710145,
            ],
        ),
        "simple": (
            2.239531384413449,
            [
                0,
                0,
                0.21142857142857147,
                0.21142857142857147,
                0.21142857142857147,
                0,
                0,
                0,
                0.18285714285714288,
                0.18285714285714288,
            ],
        ),
    }

    @pytest.mark.parametrize("aggregation", ["twfe", "overall", "simple"])
    def test_attgt_weighted_branches(self, aggregation):
        df = self._weighted_panel()
        cs = diff_diff.CallawaySantAnna(base_period="universal", control_group="never_treated").fit(
            df, outcome="y", unit="id", time="t", first_treat="g"
        )
        unit_w = df.groupby("id", sort=True)["w"].first().to_numpy()
        result = attgt_weights(cs, aggregation=aggregation, weights=unit_w)
        implied, weight = self._AGG[aggregation]
        assert result.implied_att == pytest.approx(implied, abs=1e-12)
        np.testing.assert_allclose(result.weights["weight"].to_numpy(), weight, atol=1e-12)
        assert list(zip(result.weights["group"], result.weights["time"])) == [
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
        ]


# ---------------------------------------------------------------------------
# Review-response regression tests (PR #812 items 1-5, 9, 17, 19)
# ---------------------------------------------------------------------------


def _gt_frame(fitted):
    return fitted.to_dataframe("group_time")


def _frame_call(frame, panel, **kw):
    return attgt_weights(
        frame, data=panel, unit="unit", time="period", first_treat="first_treat", **kw
    )


class TestCohortLabelValidation:
    """Item 1: NaN / -inf labels are errors, never a silent never-treated unit."""

    @pytest.mark.parametrize("bad", [np.nan, -np.inf])
    def test_decompose_rejects_non_finite_labels(self, panel, bad):
        df = panel.copy()
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[df["unit"] == 45, "first_treat"] = bad
        with pytest.raises(ValueError, match="NaN or -inf cohort label"):
            diff_diff.decompose_twfe_weights(
                df, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
            )

    @pytest.mark.parametrize("bad", [np.nan, -np.inf])
    def test_frame_path_rejects_non_finite_labels(self, fitted, panel, bad):
        df = panel.copy()
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[df["unit"] == 45, "first_treat"] = bad
        with pytest.raises(ValueError, match="NaN or -inf cohort label"):
            _frame_call(_gt_frame(fitted), df)

    def test_plus_inf_is_never_treated(self, fitted, panel):
        df = panel.copy()
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[df["first_treat"] == 0, "first_treat"] = np.inf
        with_inf = _frame_call(_gt_frame(fitted), df)
        with_zero = _frame_call(_gt_frame(fitted), panel)
        np.testing.assert_allclose(
            with_inf.weights["weight"].to_numpy(),
            with_zero.weights["weight"].to_numpy(),
            atol=1e-15,
        )

    def test_nan_in_one_period_fails_invariance(self, fitted, panel):
        df = panel.copy()
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[(df["unit"] == 45) & (df["period"] == 2), "first_treat"] = np.nan
        with pytest.raises(ValueError, match="varies within unit"):
            _frame_call(_gt_frame(fitted), df)
        with pytest.raises(ValueError, match="varies within unit"):
            diff_diff.decompose_twfe_weights(
                df, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
            )

    def test_nan_period_label_is_rejected_up_front(self, fitted, panel):
        df = panel.copy()
        df["period"] = df["period"].astype(float)
        df.loc[(df["unit"] == 45) & (df["period"] == 2), "period"] = np.nan
        with pytest.raises(ValueError, match="non-finite or non-numeric period"):
            _frame_call(_gt_frame(fitted), df)
        with pytest.raises(ValueError, match="non-finite or non-numeric period"):
            diff_diff.decompose_twfe_weights(
                df, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
            )


class TestBalanceNaNPropagation:
    """Item 2: frac_extreme's NA for <3 distinct values survives the summary roll-up."""

    @pytest.fixture(scope="class")
    def decomposed(self):
        df = _panel()
        rng = np.random.RandomState(3)
        df["binary"] = rng.binomial(1, 0.4, size=len(df)).astype(float)
        df["const"] = 1.0
        df["cont"] = rng.normal(size=len(df))
        # Make the binary / constant columns unit-invariant so the unit mean
        # keeps them at <3 distinct values.
        df["binary"] = df.groupby("unit")["binary"].transform("first")
        return diff_diff.decompose_twfe_weights(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["cont"],
            balance_covariates=["binary", "const", "cont"],
        )

    def test_cell_level_is_nan_for_degenerate_covariates(self, decomposed):
        cells = decomposed.covariate_balance(level="cell")
        for cov in ("binary", "const"):
            sub = cells[cells["covariate"] == cov]
            assert sub["unweighted_frac_extreme"].isna().all()
            assert sub["weighted_frac_extreme"].isna().all()
        cont = cells[cells["covariate"] == "cont"]
        assert np.isfinite(cont["unweighted_frac_extreme"]).all()

    def test_summary_level_propagates_nan_not_zero(self, decomposed):
        summary = decomposed.covariate_balance(level="summary").set_index("covariate")
        for cov in ("binary", "const"):
            assert np.isnan(summary.loc[cov, "unweighted_frac_extreme"])
            assert np.isnan(summary.loc[cov, "weighted_frac_extreme"])
            # The other statistics are ordinary sums and stay finite.
            assert np.isfinite(summary.loc[cov, "unweighted_diff"])
        assert np.isfinite(summary.loc["cont", "weighted_frac_extreme"])


class TestCovariateGuard:
    """Item 3: a covariate-adjusted CS fit is not a TWFE regression."""

    def test_covariate_adjusted_fit_is_rejected_under_twfe(self, panel):
        df = panel.copy()
        df["x"] = np.random.RandomState(5).normal(size=len(df))
        fit = diff_diff.CallawaySantAnna(
            control_group="never_treated", base_period="universal"
        ).fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["x"],
        )
        assert fit._aggregation_kit.bookkeeping["covariates"] == ("x",)
        with pytest.raises(ValueError, match="requires a fit without covariates"):
            attgt_weights(fit, aggregation="twfe")
        # The CS estimands do not depend on the regression specification.
        assert attgt_weights(fit, aggregation="overall").n_negative_post == 0

    def test_unadjusted_fit_records_empty_covariates(self, fitted):
        assert fitted._aggregation_kit.bookkeeping["covariates"] == ()

    def test_legacy_kit_without_the_key_warns(self, fitted):
        kit = fitted._aggregation_kit
        saved = kit.bookkeeping.pop("covariates")
        try:
            with pytest.warns(UserWarning, match="predates covariate bookkeeping"):
                attgt_weights(fitted, aggregation="twfe")
        finally:
            kit.bookkeeping["covariates"] = saved

    def test_wrong_result_type_is_a_type_error(self, panel):
        dec = diff_diff.decompose_twfe_weights(
            panel, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        with pytest.raises(TypeError, match="CallawaySantAnna"):
            attgt_weights(dec)  # type: ignore[arg-type]


class TestFrameValidationAndGrid:
    """Item 4: duplicates, non-finite cells, and incomplete grids fail closed."""

    def test_duplicate_cells_are_rejected(self, fitted, panel):
        frame = _gt_frame(fitted)
        dup = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicated \\(group, time\\)"):
            _frame_call(dup, panel)

    def test_non_finite_group_label_is_rejected(self, fitted, panel):
        frame = _gt_frame(fitted)
        frame.loc[0, "group"] = np.nan
        with pytest.raises(ValueError, match="NaN or -inf cohort label"):
            _frame_call(frame, panel)

    @pytest.mark.parametrize("aggregation", ["twfe", "overall", "simple"])
    def test_inf_att_on_a_post_cell_is_an_incomplete_grid(self, fitted, panel, aggregation):
        frame = _gt_frame(fitted)
        idx = frame.index[(frame["group"] == 3) & (frame["time"] == 3)][0]
        frame.loc[idx, "effect"] = np.inf
        with pytest.raises(ValueError, match="complete .* grid"):
            _frame_call(frame, panel, aggregation=aggregation)

    @pytest.mark.parametrize("aggregation", ["twfe", "overall", "simple"])
    def test_missing_post_cell_raises_for_every_aggregation(self, fitted, panel, aggregation):
        frame = _gt_frame(fitted)
        frame = frame[~((frame["group"] == 3) & (frame["time"] == 3))]
        with pytest.raises(ValueError, match="required cell\\(s\\) are missing"):
            _frame_call(frame, panel, aggregation=aggregation)

    def test_missing_pre_cell_raises_for_twfe_but_warns_for_cs_estimands(self, fitted, panel):
        frame = _gt_frame(fitted)
        frame = frame[~((frame["group"] == 4) & (frame["time"] == 1))]
        with pytest.raises(ValueError, match="complete cohort x period grid"):
            _frame_call(frame, panel, aggregation="twfe")
        complete = _frame_call(_gt_frame(fitted), panel, aggregation="overall")
        for aggregation in ("overall", "simple"):
            # An ABSENT pre row is not a drop: nothing to warn about, weights unchanged.
            partial = _frame_call(frame, panel, aggregation=aggregation)
            assert partial.n_dropped_cells == 0
            assert len(partial.weights) == len(frame)
            ref = _frame_call(_gt_frame(fitted), panel, aggregation=aggregation)
            assert partial.implied_att == pytest.approx(ref.implied_att, abs=1e-15)
        # A NaN pre cell (present but non-estimable) is what n_dropped_cells counts.
        frame = _gt_frame(fitted)
        frame.loc[frame.index[(frame["group"] == 4) & (frame["time"] == 1)][0], "effect"] = np.nan
        with pytest.warns(UserWarning, match="pre-treatment group-time cell"):
            dropped = _frame_call(frame, panel, aggregation="overall")
        assert dropped.n_dropped_cells == 1
        assert dropped.implied_att == pytest.approx(complete.implied_att, abs=1e-15)

    def test_first_period_cohort_is_dropped_like_r_did(self):
        df = _panel(cohorts=(0, 1, 3, 4), n_periods=5)
        fit = _fit(df)
        with pytest.warns(UserWarning, match="no estimable post-treatment cell"):
            result = attgt_weights(fit, aggregation="overall")
        assert 1 not in set(result.weights["group"])
        assert result.weights["weight"].sum() == pytest.approx(1.0, abs=1e-12)
        # Same numbers as fitting on the panel with those units removed up front.
        pre_filtered = _fit(df[df["first_treat"] != 1])
        reference = attgt_weights(pre_filtered, aggregation="overall")
        np.testing.assert_allclose(
            result.weights["weight"].to_numpy(), reference.weights["weight"].to_numpy(), atol=1e-12
        )

    def test_bare_frame_missing_a_whole_non_first_cohort_raises(self, fitted, panel):
        frame = _gt_frame(fitted).drop(columns=["skip_reason"], errors="ignore")
        frame = frame[frame["group"] != 4]
        with pytest.raises(ValueError, match="no post-treatment cell in the ATT"):
            _frame_call(frame, panel, aggregation="overall")

    def test_not_yet_treated_carve_out_mirrors_aggte(self):
        df = _panel(cohorts=(3, 4, 5), n_periods=6)
        fit = _fit(df, control_group="not_yet_treated")
        with pytest.warns(UserWarning) as record:
            result = attgt_weights(fit, aggregation="overall")
        messages = " | ".join(str(w.message) for w in record)
        assert "structurally absent" in messages  # (3,5),(3,6),(4,5),(4,6)
        assert "no estimable post-treatment cell" in messages  # cohort 5
        assert set(result.weights["group"]) == {3, 4}
        assert result.weights["weight"].sum() == pytest.approx(1.0, abs=1e-12)
        post = result.weights[result.weights["post"] == 1]
        # cohort 3 keeps (3,3),(3,4): divisor 2; cohort 4 keeps (4,4): divisor 1
        assert post[post["group"] == 3]["time"].tolist() == [3, 4]
        assert post[post["group"] == 4]["time"].tolist() == [4]
        w3 = post[post["group"] == 3]["weight"].to_numpy()
        w4 = post[post["group"] == 4]["weight"].to_numpy()
        assert w3[0] == pytest.approx(w3[1])
        assert w4[0] == pytest.approx(2 * w3[0])  # equal cohorts: pbar_3 == pbar_4
        expected = float((post["weight"] * post["att"]).sum())
        assert result.implied_att == pytest.approx(expected, abs=1e-12)
        with pytest.raises(ValueError, match="control_group='never_treated'"):
            attgt_weights(fit, aggregation="twfe")


class TestWeightValidation:
    """Item 5: finite, non-negative, positive treated (and control) mass."""

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda w: np.where(np.arange(len(w)) < 50, -1.0, w), "non-negative"),
            (lambda w: np.where(np.arange(len(w)) == 0, np.nan, w), "must be finite"),
            (lambda w: np.where(np.arange(len(w)) == 0, np.inf, w), "must be finite"),
            (lambda w: np.zeros_like(w), "sum to zero"),
        ],
    )
    def test_bad_unit_weights_are_rejected(self, fitted, mutate, match):
        w = mutate(np.ones(len(fitted._aggregation_kit.bookkeeping["unit_cohorts"])))
        with pytest.raises(ValueError, match=match):
            attgt_weights(fitted, aggregation="overall", weights=w)

    def test_zero_control_mass_only_matters_where_controls_enter(self, fitted, panel):
        cohorts = np.asarray(fitted._aggregation_kit.bookkeeping["unit_cohorts"], dtype=float)
        w = np.where(cohorts == 0, 0.0, 1.0)
        with pytest.raises(ValueError, match="never-treated comparison group carries zero"):
            attgt_weights(fitted, aggregation="twfe", weights=w)
        for aggregation in ("overall", "simple"):
            assert attgt_weights(fitted, aggregation=aggregation, weights=w).n_cells > 0

    def test_decompose_reports_a_nan_weight_as_non_finite(self, panel):
        df = panel.copy()
        df["w"] = 1.0
        df.loc[(df["unit"] == 3) & (df["period"] == 2), "w"] = np.nan
        with pytest.raises(ValueError, match="must be finite"):
            diff_diff.decompose_twfe_weights(
                df,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                weights="w",
            )


class TestNegativePostWeights:
    """Item 9: the pathology is negative weight on POST cells."""

    @pytest.mark.parametrize("aggregation", ["overall", "simple"])
    def test_cs_estimands_have_no_negative_post_weight(self, fitted, aggregation):
        result = attgt_weights(fitted, aggregation=aggregation)
        assert result.n_negative_post == 0
        assert result.negative_post_weight_share == 0.0

    def test_twfe_reports_both_labelled(self, fitted):
        result = attgt_weights(fitted, aggregation="twfe")
        assert result.n_negative_post <= result.n_negative
        assert 0.0 <= result.negative_post_weight_share <= 1.0
        d = result.to_dict()
        assert {"n_negative_post", "negative_post_weight_share"} <= set(d)
        assert "Negative POST-period cells:" in result.summary()


class TestWeightedTwfeExtension:
    """Item 17: weighted aggregation="twfe" has no R counterpart; tie it to the decomposition."""

    def test_weighted_twfe_weights_match_the_weighted_decomposition(self, panel):
        df = panel.copy()
        rng = np.random.RandomState(9)
        unit_w = pd.Series(
            rng.choice([0.5, 1.0, 2.0], size=df["unit"].nunique()),
            index=sorted(df["unit"].unique()),
        )
        df["w"] = df["unit"].map(unit_w)
        fit = _fit(df)
        weighted = attgt_weights(fit, aggregation="twfe", weights=unit_w.to_numpy())
        decomposed = diff_diff.decompose_twfe_weights(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            weights="w",
        )
        np.testing.assert_allclose(
            weighted.weights["weight"].to_numpy(), decomposed.cells["weight"].to_numpy(), atol=1e-12
        )


class TestHandComputedWeights:
    """Item 19: assert against numbers computed OUTSIDE the module."""

    def test_implied_att_equals_hand_computed_overall(self):
        # 2 cohorts (g=2,3) x 3 periods, 10 + 30 treated units + 20 never-treated.
        frame = pd.DataFrame(
            {
                "group": [2, 2, 2, 3, 3, 3],
                "time": [1, 2, 3, 1, 2, 3],
                "effect": [0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
            }
        )
        rows = []
        for u, g in enumerate([0] * 20 + [2] * 10 + [3] * 30):
            for t in (1, 2, 3):
                rows.append({"unit": u, "period": t, "first_treat": g, "outcome": 0.0})
        panel = pd.DataFrame(rows)
        result = _frame_call(frame, panel, aggregation="overall")
        # pbar_2 = 10/40, pbar_3 = 30/40; cohort 2 has 2 post periods, cohort 3 has 1.
        expected = (10 / 40) / 2 * (1.0 + 2.0) + (30 / 40) / 1 * 4.0
        assert result.implied_att == pytest.approx(expected, abs=1e-15)
        assert result.weights["weight"].sum() == pytest.approx(1.0, abs=1e-15)

    def test_all_five_public_names_are_exported(self):
        for name in (
            "attgt_weights",
            "decompose_twfe_weights",
            "ATTGTWeightsResult",
            "TWFEDecompositionResult",
            "plot_twfe_weights",
        ):
            assert name in diff_diff.__all__, name
            assert hasattr(diff_diff, name)


class TestCollinearCovariates:
    """Item 7: rank-deficient designs go through solve_ols' R-style NaN handling."""

    def test_exactly_collinear_pair_warns_once_and_leaves_the_estimate_unchanged(self, panel):
        df = panel.copy()
        rng = np.random.RandomState(21)
        df["x1"] = rng.normal(size=len(df))
        df["x2"] = 2.0 * df["x1"]  # exactly collinear twin
        common = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
        with pytest.warns(UserWarning, match="dropped collinear covariate") as record:
            both = diff_diff.decompose_twfe_weights(df, covariates=["x1", "x2"], **common)
        collinear = [w for w in record if "dropped collinear" in str(w.message)]
        assert len(collinear) == 1
        message = str(collinear[0].message)
        assert ("'x1'" in message) != ("'x2'" in message)  # exactly one of the pair
        alone = diff_diff.decompose_twfe_weights(df, covariates=["x1"], **common)
        assert both.estimate == pytest.approx(alone.estimate, abs=1e-12)
        np.testing.assert_allclose(
            both.cells["weight"].to_numpy(), alone.cells["weight"].to_numpy(), atol=1e-12
        )


class TestDecompositionEdgeCases:
    """Item 8: the REGISTRY edge cases for decompose_twfe_weights, asserted."""

    COMMON = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")

    def test_unbalanced_panel_is_rejected(self, panel):
        df = panel.drop(panel.index[(panel["unit"] == 7) & (panel["period"] == 3)])
        with pytest.raises(ValueError, match="balanced panel"):
            diff_diff.decompose_twfe_weights(df, **self.COMMON)

    def test_no_never_treated_group_is_rejected(self):
        df = _panel(cohorts=(3, 4))
        with pytest.raises(ValueError, match="never-treated units"):
            diff_diff.decompose_twfe_weights(df, **self.COMMON)

    def test_time_varying_cohort_is_rejected(self, panel):
        df = panel.copy()
        df.loc[(df["unit"] == 50) & (df["period"] == 5), "first_treat"] = 4
        with pytest.raises(ValueError, match="varies within unit"):
            diff_diff.decompose_twfe_weights(df, **self.COMMON)

    def test_gmin1_with_a_first_period_cohort_is_rejected(self):
        df = _panel(cohorts=(0, 1, 3))
        with pytest.raises(ValueError, match="gmin1"):
            diff_diff.decompose_twfe_weights(df, base_period="gmin1", **self.COMMON)

    def test_balance_without_request_and_bad_level(self, panel):
        result = diff_diff.decompose_twfe_weights(panel, **self.COMMON)
        with pytest.raises(ValueError, match="balance_covariates="):
            result.covariate_balance()
        df = panel.copy()
        df["x"] = np.random.RandomState(4).normal(size=len(df))
        with_balance = diff_diff.decompose_twfe_weights(df, balance_covariates=["x"], **self.COMMON)
        with pytest.raises(ValueError, match="level must be"):
            with_balance.covariate_balance(level="cohort")

    def test_bad_method_and_base_period(self, panel):
        with pytest.raises(ValueError, match="method must be"):
            diff_diff.decompose_twfe_weights(panel, method="aipw", **self.COMMON)
        with_bad = dict(self.COMMON)
        with pytest.raises(ValueError, match="base_period must be"):
            diff_diff.decompose_twfe_weights(panel, base_period="universal", **with_bad)

    def test_identities_hold(self, panel, fitted):
        result = diff_diff.decompose_twfe_weights(panel, **self.COMMON)
        assert result.estimate == pytest.approx(result.decomposition + result.remainder, abs=1e-12)
        assert result.pretrend_bias + result.post_only == pytest.approx(
            result.decomposition, abs=1e-12
        )
        assert result.remainder == 0.0
        assert attgt_weights(fitted, aggregation="twfe").implied_att == pytest.approx(
            result.estimate, abs=1e-6
        )
        gmin1 = diff_diff.decompose_twfe_weights(panel, base_period="gmin1", **self.COMMON)
        assert gmin1.estimate == pytest.approx(result.estimate, abs=1e-10)
        assert gmin1.estimate == pytest.approx(gmin1.decomposition + gmin1.remainder, abs=1e-12)


class TestPlotTWFEWeights:
    """Item 8 / 13: matplotlib behaviour of plot_twfe_weights."""

    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        yield
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_weights_view(self, fitted):
        result = attgt_weights(fitted, aggregation="twfe")
        ax = diff_diff.plot_twfe_weights(result, show=False)
        assert ax.get_xlabel() == "Implicit weight"
        assert "negative" in ax.get_title()
        assert len(ax.collections) == 2  # post + pre scatters

    def test_ax_reuse_and_annotate(self, fitted):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        out = diff_diff.plot_twfe_weights(
            attgt_weights(fitted, aggregation="overall"), ax=ax, annotate=True, show=False
        )
        assert out is ax
        assert len(ax.texts) == 10

    def test_balance_view_and_auto(self, panel):
        df = panel.copy()
        df["x"] = np.random.RandomState(8).normal(size=len(df))
        dec = diff_diff.decompose_twfe_weights(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["x"],
            balance_covariates=["x"],
        )
        ax = diff_diff.plot_twfe_weights(dec, show=False)  # auto -> balance
        assert "balance" in ax.get_title().lower()
        ax2 = diff_diff.plot_twfe_weights(dec, kind="weights", show=False)
        assert ax2.get_ylabel() == "ATT(g, t)"

    def test_balance_requested_without_table_raises(self, panel):
        dec = diff_diff.decompose_twfe_weights(
            panel, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        with pytest.raises(ValueError, match="no covariate balance table"):
            diff_diff.plot_twfe_weights(dec, kind="balance", show=False)

    def test_all_nan_balance_table_is_a_clear_error(self, panel):
        df = panel.copy()
        df["const"] = 1.0  # zero pooled SD -> standardized diffs are all NaN
        dec = diff_diff.decompose_twfe_weights(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            balance_covariates=["const"],
        )
        with pytest.raises(ValueError, match="no finite differences"):
            diff_diff.plot_twfe_weights(dec, kind="balance", show=False)

    def test_bad_kind_and_backend(self, fitted):
        result = attgt_weights(fitted)
        with pytest.raises(ValueError, match="kind must be"):
            diff_diff.plot_twfe_weights(result, kind="heat", show=False)
        with pytest.raises(ValueError, match="backend must be"):
            diff_diff.plot_twfe_weights(result, backend="bokeh", show=False)
