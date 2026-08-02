"""Tests for diff_diff.mmm - MMM calibration input assembly (interop).

The exporters are explicit-in / validated-out: the caller supplies the already-scoped
incremental outcome and its SE, and the module assembles the target schema, enforces
each consumer's guards, converts to the lognormal parameterization, and pools. These
tests cover the schema/guards/math directly, plus a realistic workflow test that reads
att/se off fitted estimators and feeds them in (the module never introspects results).
"""

import math
import warnings

import pytest
from scipy import stats

from diff_diff import (
    CallawaySantAnna,
    DifferenceInDifferences,
    to_meridian_roi_prior,
    to_pymc_marketing_lift_test,
)
from diff_diff.mmm import MeridianROIPrior
from diff_diff.prep import generate_did_data, generate_staggered_data


class TestLiftTestFrame:
    def test_columns_and_passthrough(self):
        df = to_pymc_marketing_lift_test(
            channel="tv", x=50_000.0, delta_x=20_000.0, delta_y=120.0, sigma=30.0
        )
        assert list(df.columns) == ["channel", "x", "delta_x", "delta_y", "sigma"]
        row = df.iloc[0]
        assert (row["channel"], row["x"], row["delta_x"], row["delta_y"], row["sigma"]) == (
            "tv",
            50_000.0,
            20_000.0,
            120.0,
            30.0,
        )

    def test_multiple_experiments_broadcast(self):
        df = to_pymc_marketing_lift_test(
            channel="tv",
            x=[100.0, 200.0],
            delta_x=50.0,
            delta_y=[10.0, 20.0],
            sigma=[2.0, 4.0],
        )
        assert list(df["x"]) == [100.0, 200.0]
        assert list(df["delta_x"]) == [50.0, 50.0]
        assert list(df["delta_y"]) == [10.0, 20.0]

    def test_dims_columns_between_channel_and_x(self):
        df = to_pymc_marketing_lift_test(
            channel="search",
            x=[100.0, 200.0],
            delta_x=50.0,
            delta_y=[10.0, 20.0],
            sigma=[2.0, 4.0],
            dims=[{"geo": "US-CA"}, {"geo": "US-NY"}],
        )
        assert list(df.columns) == ["channel", "geo", "x", "delta_x", "delta_y", "sigma"]
        assert list(df["geo"]) == ["US-CA", "US-NY"]

    def test_single_dims_mapping_broadcasts(self):
        df = to_pymc_marketing_lift_test(
            channel="tv",
            x=[1.0, 2.0],
            delta_x=1.0,
            delta_y=[1.0, 1.0],
            sigma=1.0,
            dims={"geo": "US-CA"},
        )
        assert list(df["geo"]) == ["US-CA", "US-CA"]

    def test_dims_non_mapping_raises(self):
        with pytest.raises(TypeError, match="sequence of mappings"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, delta_y=1.0, sigma=1.0, dims=["geo"]
            )

    def test_dims_reserved_collision_raises(self):
        with pytest.raises(ValueError, match="reserved lift-test columns"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=1.0,
                delta_x=1.0,
                delta_y=1.0,
                sigma=1.0,
                dims={"sigma": "s"},
            )

    def test_heterogeneous_dims_keys_raise(self):
        with pytest.raises(ValueError, match="share one key set"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=1.0,
                delta_x=1.0,
                delta_y=[1.0, 1.0],
                sigma=1.0,
                dims=[{"geo": "US-CA"}, {"region": "NY"}],
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="delta_y has length 2 but 3"):
            to_pymc_marketing_lift_test(
                channel="tv", x=[1.0, 2.0, 3.0], delta_x=1.0, delta_y=[1.0, 2.0], sigma=1.0
            )

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="empty sequence"):
            to_pymc_marketing_lift_test(channel="tv", x=[], delta_x=1.0, delta_y=1.0, sigma=1.0)

    def test_sigma_must_be_positive(self):
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="sigma must be finite and > 0"):
                to_pymc_marketing_lift_test(
                    channel="tv", x=1.0, delta_x=1.0, delta_y=1.0, sigma=bad
                )

    def test_delta_x_zero_raises(self):
        with pytest.raises(ValueError, match="delta_x must be finite and nonzero"):
            to_pymc_marketing_lift_test(channel="tv", x=1.0, delta_x=0.0, delta_y=1.0, sigma=1.0)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x must be finite"):
            to_pymc_marketing_lift_test(channel="tv", x=-5.0, delta_x=1.0, delta_y=1.0, sigma=1.0)

    def test_negative_post_test_spend_raises(self):
        with pytest.raises(ValueError, match="x \\+ delta_x must be finite and >= 0"):
            to_pymc_marketing_lift_test(
                channel="tv", x=100.0, delta_x=-101.0, delta_y=-40.0, sigma=10.0
            )

    def test_non_finite_delta_y_raises(self):
        with pytest.raises(ValueError, match="delta_y must be finite"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, delta_y=float("nan"), sigma=1.0
            )

    def test_go_dark_is_valid(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = to_pymc_marketing_lift_test(
                channel="tv", x=1_000.0, delta_x=-500.0, delta_y=-40.0, sigma=10.0
            )
        assert df.loc[0, "delta_y"] == -40.0

    def test_go_dark_to_zero_spend_is_valid(self):
        df = to_pymc_marketing_lift_test(
            channel="tv", x=1_000.0, delta_x=-1_000.0, delta_y=-40.0, sigma=10.0
        )
        assert df.loc[0, "delta_x"] == -1_000.0

    def test_wrong_sign_raises_by_default(self):
        with pytest.raises(ValueError, match="NonMonotonicError"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1_000.0, delta_x=500.0, delta_y=-40.0, sigma=10.0
            )

    def test_wrong_sign_drop_and_keep(self):
        with pytest.warns(UserWarning, match="Dropping invalid lift-test"):
            df = to_pymc_marketing_lift_test(
                channel="tv",
                x=1_000.0,
                delta_x=500.0,
                delta_y=[10.0, -40.0],
                sigma=[2.0, 10.0],
                on_wrong_sign="drop",
            )
        assert len(df) == 1 and df.loc[0, "delta_y"] == 10.0
        with pytest.warns(UserWarning, match="Keeping invalid lift-test"):
            df2 = to_pymc_marketing_lift_test(
                channel="tv",
                x=1_000.0,
                delta_x=500.0,
                delta_y=-40.0,
                sigma=10.0,
                on_wrong_sign="keep",
            )
        assert df2.loc[0, "delta_y"] == -40.0

    def test_wrong_sign_drop_all_raises(self):
        with pytest.raises(ValueError, match="would remove every row"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=1_000.0,
                delta_x=500.0,
                delta_y=-40.0,
                sigma=10.0,
                on_wrong_sign="drop",
            )

    def test_tiny_wrong_sign_detected_despite_underflow(self):
        # delta_x * delta_y underflows to -0.0 here; a direct sign comparison
        # still catches the wrong sign.
        with pytest.raises(ValueError, match="NonMonotonicError"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1e-200, delta_y=-1e-200, sigma=1.0
            )

    def test_zero_lift_names_gamma(self):
        with pytest.raises(ValueError, match="Gamma lift likelihood"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1_000.0, delta_x=500.0, delta_y=0.0, sigma=10.0
            )

    def test_zero_lift_drop_removes_only_zero(self):
        with pytest.warns(UserWarning, match="delta_y == 0"):
            df = to_pymc_marketing_lift_test(
                channel="tv",
                x=1_000.0,
                delta_x=500.0,
                delta_y=[10.0, 0.0],
                sigma=[2.0, 2.0],
                on_wrong_sign="drop",
            )
        assert len(df) == 1 and df.loc[0, "delta_y"] == 10.0

    def test_invalid_policy_value(self):
        with pytest.raises(ValueError, match="on_wrong_sign must be"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, delta_y=1.0, sigma=1.0, on_wrong_sign="x"
            )

    def test_post_spend_overflow_raises(self):
        # x + delta_x must be finite, not just non-negative.
        with pytest.raises(ValueError, match="finite and >= 0"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1e308, delta_x=1e308, delta_y=1.0, sigma=1.0
            )


class TestMeridianROIPrior:
    def test_roi_math_and_lognormal_roundtrip(self):
        prior = to_meridian_roi_prior(
            incremental_outcome=9_600.0, incremental_outcome_se=2_400.0, spend=20_000.0
        )
        assert prior.roi_mean == pytest.approx(9_600.0 / 20_000.0)
        assert prior.roi_sd == pytest.approx(2_400.0 / 20_000.0)
        assert prior.parameter == "roi_m"
        dist = stats.lognorm(s=prior.sigma, scale=math.exp(prior.mu))
        assert dist.mean() == pytest.approx(prior.roi_mean, rel=1e-12)
        assert dist.std() == pytest.approx(prior.roi_sd, rel=1e-12)

    def test_spend_weighted_pooling(self):
        prior = to_meridian_roi_prior(
            incremental_outcome=[9_600.0, 9_600.0],
            incremental_outcome_se=[2_400.0, 2_400.0],
            spend=[20_000.0, 30_000.0],
        )
        roi_1, roi_2 = 9_600.0 / 20_000.0, 9_600.0 / 30_000.0
        w_1, w_2 = 0.4, 0.6
        assert prior.roi_mean == pytest.approx(w_1 * roi_1 + w_2 * roi_2)
        sd_1, sd_2 = 2_400.0 / 20_000.0, 2_400.0 / 30_000.0
        assert prior.roi_sd == pytest.approx(math.hypot(w_1 * sd_1, w_2 * sd_2))
        assert [e.weight for e in prior.per_experiment] == pytest.approx([w_1, w_2])

    def test_se_widening_scales_sd_only(self):
        base = to_meridian_roi_prior(
            incremental_outcome=9_600.0, incremental_outcome_se=2_400.0, spend=20_000.0
        )
        wide = to_meridian_roi_prior(
            incremental_outcome=9_600.0,
            incremental_outcome_se=2_400.0,
            spend=20_000.0,
            se_widening=2.0,
        )
        assert wide.roi_mean == pytest.approx(base.roi_mean)
        assert wide.roi_sd == pytest.approx(2.0 * base.roi_sd)

    def test_mroi_m_parameter(self):
        prior = to_meridian_roi_prior(
            incremental_outcome=500.0,
            incremental_outcome_se=100.0,
            spend=1_000.0,
            parameter="mroi_m",
        )
        assert prior.parameter == "mroi_m"
        assert prior.to_dict()["parameter"] == "mroi_m"

    def test_invalid_parameter_raises(self):
        with pytest.raises(ValueError, match="parameter must be one of"):
            to_meridian_roi_prior(
                incremental_outcome=1.0,
                incremental_outcome_se=1.0,
                spend=1.0,
                parameter="contribution",
            )

    def test_spend_and_se_must_be_positive(self):
        with pytest.raises(ValueError, match="spend must be finite and > 0"):
            to_meridian_roi_prior(incremental_outcome=1.0, incremental_outcome_se=1.0, spend=0.0)
        with pytest.raises(ValueError, match="incremental_outcome_se must be finite and > 0"):
            to_meridian_roi_prior(incremental_outcome=1.0, incremental_outcome_se=-1.0, spend=1.0)

    def test_se_widening_invalid_raises(self):
        for bad in (0.0, -1.0, float("inf")):
            with pytest.raises(ValueError, match="se_widening"):
                to_meridian_roi_prior(
                    incremental_outcome=1.0,
                    incremental_outcome_se=1.0,
                    spend=1.0,
                    se_widening=bad,
                )

    def test_non_positive_pooled_roi_raises(self):
        with pytest.raises(ValueError, match="positive support"):
            to_meridian_roi_prior(
                incremental_outcome=-500.0, incremental_outcome_se=100.0, spend=1_000.0
            )

    def test_overflow_raises_not_inf(self):
        with pytest.raises(ValueError, match="finite-positive"):
            to_meridian_roi_prior(
                incremental_outcome=1e308, incremental_outcome_se=1e307, spend=1e-10
            )

    def test_small_relative_se_keeps_positive_sigma(self):
        prior = to_meridian_roi_prior(
            incremental_outcome=1.0, incremental_outcome_se=1e-8, spend=1.0
        )
        assert prior.sigma > 0 and math.isfinite(prior.mu)

    def test_extreme_coefficient_of_variation_no_overflow(self):
        # A huge roi_sd/roi_mean ratio would overflow (sd/mean)**2; the log-domain
        # log1p keeps mu/sigma finite instead of raising a raw OverflowError.
        prior = to_meridian_roi_prior(
            incremental_outcome=1e-300, incremental_outcome_se=1.0, spend=1.0
        )
        assert math.isfinite(prior.mu) and math.isfinite(prior.sigma) and prior.sigma > 0

    def test_pooled_sd_uses_scaled_norm_no_overflow(self):
        # Per-experiment SDs whose naive sum-of-squares would overflow (>1.8e308)
        # but whose true weighted norm is finite: math.hypot keeps it finite
        # instead of raising a raw OverflowError.
        prior = to_meridian_roi_prior(
            incremental_outcome=[1e160, 1e160],
            incremental_outcome_se=[1e160, 1e160],
            spend=[1.0, 1.0],
        )
        assert math.isfinite(prior.roi_sd) and prior.roi_sd > 0

    def test_empty_and_length_mismatch(self):
        with pytest.raises(ValueError, match="empty sequence"):
            to_meridian_roi_prior(incremental_outcome=[], incremental_outcome_se=1.0, spend=1.0)
        with pytest.raises(ValueError, match="length 3"):
            to_meridian_roi_prior(
                incremental_outcome=[1.0, 2.0],
                incremental_outcome_se=[1.0, 1.0, 1.0],
                spend=1.0,
            )


class TestToCode:
    def _prior(self, parameter="roi_m"):
        return to_meridian_roi_prior(
            incremental_outcome=9_600.0,
            incremental_outcome_se=2_400.0,
            spend=20_000.0,
            parameter=parameter,
        )

    def test_requires_channel_scope(self):
        with pytest.raises(ValueError, match="broadcasts to"):
            self._prior().to_code(full_model_window=True)

    def test_requires_time_scope(self):
        with pytest.raises(ValueError, match="time scope"):
            self._prior().to_code(single_channel=True)

    def test_single_channel_snippet(self):
        prior = self._prior()
        code = prior.to_code(single_channel=True, full_model_window=True)
        assert "SINGLE-CHANNEL MODEL ONLY" in code
        assert 'name="roi_m"' in code
        assert 'media_prior_type="roi"' in code
        assert repr(prior.mu) in code

    def test_multi_channel_vector_order_and_defaults(self):
        prior = self._prior()
        code = prior.to_code(
            channel="tv", media_channels=["search", "tv", "social"], full_model_window=True
        )
        assert repr([0.2, prior.mu, 0.2]) in code
        assert repr([0.9, prior.sigma, 0.9]) in code

    def test_mroi_uses_its_default_and_prior_type(self):
        prior = self._prior(parameter="mroi_m")
        code = prior.to_code(channel="tv", media_channels=["search", "tv"], full_model_window=True)
        assert 'name="mroi_m"' in code
        assert 'media_prior_type="mroi"' in code
        assert repr([0.0, prior.mu]) in code
        assert repr([0.5, prior.sigma]) in code

    def test_calibration_mask_emitted(self):
        code = self._prior().to_code(single_channel=True, roi_calibration_period="my_mask")
        assert "roi_calibration_period=my_mask" in code

    def test_blank_or_unparseable_calibration_mask_raises(self):
        for bad in ("", "   ", "mask ="):
            with pytest.raises(ValueError, match="valid Python expression"):
                self._prior().to_code(single_channel=True, roi_calibration_period=bad)

    def test_scope_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            self._prior().to_code(
                single_channel=True, roi_calibration_period="m", full_model_window=True
            )
        with pytest.raises(ValueError, match="not both"):
            self._prior().to_code(
                channel="tv",
                media_channels=["tv"],
                single_channel=True,
                full_model_window=True,
            )


class TestRealisticWorkflow:
    """The documented use: fit an estimator, read att/se off it, scope, and export.
    The module never introspects the result - the caller passes the numbers."""

    def test_did_workflow(self):
        data = generate_did_data(
            n_units=60, n_periods=2, treatment_effect=5.0, treatment_period=1, seed=7
        )
        result = DifferenceInDifferences().fit(
            data, outcome="outcome", treatment="treated", post="post"
        )
        df = to_pymc_marketing_lift_test(
            channel="tv",
            x=1_000.0,
            delta_x=500.0,
            delta_y=result.att,
            sigma=result.se,
            on_wrong_sign="keep",
        )
        assert df.loc[0, "delta_y"] == pytest.approx(result.att)
        total, total_se = result.att * 30, result.se * 30
        prior = to_meridian_roi_prior(
            incremental_outcome=total, incremental_outcome_se=total_se, spend=100_000.0
        )
        assert prior.roi_mean == pytest.approx(total / 100_000.0)

    def test_callaway_santanna_workflow(self):
        # CS gives a headline ATT; the caller is responsible for turning it into a
        # total incremental outcome using the estimator's own aggregation (not a
        # naive att x count, which need not reproduce CS's cohort weights or
        # variance - that estimator-owned aggregation is the post-4.0 follow-up).
        # Here we simply confirm the fitted numbers flow through the exporter once
        # the caller has supplied a total and its SE.
        data = generate_staggered_data(n_units=80, n_periods=8, seed=7)
        cs = CallawaySantAnna().fit(
            data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        assert cs.att is not None and cs.se > 0  # fit produced usable inference
        # A caller-derived total incremental outcome + SE (illustrative values).
        prior = to_meridian_roi_prior(
            incremental_outcome=150_000.0,
            incremental_outcome_se=40_000.0,
            spend=200_000.0,
        )
        assert prior.roi_mean == pytest.approx(150_000.0 / 200_000.0)


def test_public_exports():
    import diff_diff

    assert diff_diff.to_pymc_marketing_lift_test is to_pymc_marketing_lift_test
    assert diff_diff.to_meridian_roi_prior is to_meridian_roi_prior
    assert diff_diff.MeridianROIPrior is MeridianROIPrior
