"""Tests for diff_diff.mmm - MMM calibration input assembly (interop).

The exporters take either explicit already-scoped numbers or the pinned
``AggregationResult`` container (``aggregation_result=`` + ``scale=``, with
``scale="auto"`` honored only for the audited ImputationDiD/TwoStageDiD
producers); the module assembles the target schema, enforces each consumer's
guards, converts to the lognormal parameterization, and pools. These tests cover
the schema/guards/math directly, the container-mode extraction/routing matrix,
end-to-end workflows on fitted estimators, and a contract pin of exactly the
container surface the module consumes.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff import (
    AggregationResult,
    CallawaySantAnna,
    DifferenceInDifferences,
    ImputationDiD,
    TwoStageDiD,
    meridian_calibration_mask,
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
        with pytest.raises(ValueError, match="not both"):
            self._prior().to_code(
                single_channel=True,
                roi_calibration_period=np.ones((3, 1), dtype=bool),
                full_model_window=True,
            )


class TestMeridianCalibrationMask:
    """meridian_calibration_mask: window resolution + fail-closed validation."""

    _TIMES = pd.date_range("2024-01-01", periods=5, freq="W-MON")
    _CHANNELS = ["search", "tv", "radio"]

    def _expected(self, rows, cols):
        # Meridian's convention: non-experiment channels use ALL periods, so
        # the expected mask is all-True except the experiment columns, which
        # carry only the window rows.
        expected = np.ones((5, 3), dtype=bool)
        expected[:, cols] = False
        expected[np.ix_(rows, cols)] = True
        return expected

    def _assert_mask(self, mask, rows, cols):
        assert mask.dtype == np.bool_
        assert mask.shape == (5, 3)
        assert np.array_equal(mask, self._expected(rows, cols))

    def test_bounds_window_datetime_media_times_str_bounds(self):
        mask = meridian_calibration_mask(
            media_times=self._TIMES,
            media_channels=self._CHANNELS,
            channel="tv",
            window=("2024-01-15", "2024-01-29"),  # inclusive both ends
        )
        self._assert_mask(mask, [2, 3, 4], [1])

    def test_bounds_window_numeric_labels(self):
        mask = meridian_calibration_mask(
            media_times=[1, 2, 3, 4, 5],
            media_channels=self._CHANNELS,
            channel="tv",
            window=(2, 4),
        )
        self._assert_mask(mask, [1, 2, 3], [1])

    def test_bounds_window_iso_string_labels(self):
        labels = [str(t.date()) for t in self._TIMES]
        mask = meridian_calibration_mask(
            media_times=labels,
            media_channels=self._CHANNELS,
            channel="search",
            window=("2024-01-15", "2024-01-29"),
        )
        self._assert_mask(mask, [2, 3, 4], [0])

    def test_explicit_labels_window_noncontiguous_multichannel(self):
        mask = meridian_calibration_mask(
            media_times=[1, 2, 3, 4, 5],
            media_channels=self._CHANNELS,
            channel=["tv", "radio"],
            window=[1, 3, 5],
        )
        self._assert_mask(mask, [0, 2, 4], [1, 2])

    def test_media_times_accepts_index_and_series(self):
        from_index = meridian_calibration_mask(
            media_times=pd.Index([1, 2, 3, 4, 5]),
            media_channels=self._CHANNELS,
            channel="tv",
            window=(2, 4),
        )
        from_series = meridian_calibration_mask(
            media_times=pd.Series([1, 2, 3, 4, 5], index=[9, 8, 7, 6, 5]),
            media_channels=self._CHANNELS,
            channel="tv",
            window=(2, 4),
        )
        assert np.array_equal(from_index, from_series)
        self._assert_mask(from_index, [1, 2, 3], [1])

    def test_channel_string_vs_sequence_equivalent(self):
        a = meridian_calibration_mask(
            media_times=[1, 2, 3, 4, 5],
            media_channels=self._CHANNELS,
            channel="tv",
            window=(1, 5),
        )
        b = meridian_calibration_mask(
            media_times=[1, 2, 3, 4, 5],
            media_channels=self._CHANNELS,
            channel=["tv"],
            window=(1, 5),
        )
        assert np.array_equal(a, b)

    def test_tz_aware_success(self):
        times = pd.date_range("2024-01-01", periods=5, freq="W-MON", tz="UTC")
        mask = meridian_calibration_mask(
            media_times=times,
            media_channels=self._CHANNELS,
            channel="tv",
            window=(pd.Timestamp("2024-01-15", tz="UTC"), pd.Timestamp("2024-01-29", tz="UTC")),
        )
        self._assert_mask(mask, [2, 3, 4], [1])

    def test_datetime_coercion_on_labels_path(self):
        mask = meridian_calibration_mask(
            media_times=self._TIMES,
            media_channels=self._CHANNELS,
            channel="tv",
            window=["2024-01-01", "2024-01-29"],  # str labels vs DatetimeIndex
        )
        self._assert_mask(mask, [0, 4], [1])

    def test_duplicate_window_labels_ok(self):
        mask = meridian_calibration_mask(
            media_times=[1, 2, 3, 4, 5],
            media_channels=self._CHANNELS,
            channel="tv",
            window=[3, 3],
        )
        self._assert_mask(mask, [2], [1])

    def test_missing_channel_raises(self):
        with pytest.raises(ValueError, match="not in media_channels"):
            meridian_calibration_mask(
                media_times=[1, 2],
                media_channels=["search"],
                channel="tv",
                window=(1, 2),
            )

    def test_missing_window_label_raises(self):
        with pytest.raises(ValueError, match="not in media_times"):
            meridian_calibration_mask(
                media_times=[1, 2],
                media_channels=["tv"],
                channel="tv",
                window=[1, 7],
            )

    def test_empty_and_bad_media_times(self):
        with pytest.raises(ValueError, match="media_times must be non-empty"):
            meridian_calibration_mask(
                media_times=[], media_channels=["tv"], channel="tv", window=(1, 2)
            )
        with pytest.raises(ValueError, match="duplicate label"):
            meridian_calibration_mask(
                media_times=[1, 1, 2], media_channels=["tv"], channel="tv", window=(1, 2)
            )
        with pytest.raises(ValueError, match="NaN/NaT"):
            meridian_calibration_mask(
                media_times=[1.0, math.nan],
                media_channels=["tv"],
                channel="tv",
                window=(1, 2),
            )

    def test_missing_channel_names_raise(self):
        # pd.NA would otherwise raise a raw ambiguous-truth TypeError in the
        # duplicate check; None would silently become a mask column.
        for bad in (None, math.nan, pd.NA):
            with pytest.raises(ValueError, match="must not contain missing names"):
                meridian_calibration_mask(
                    media_times=[1, 2],
                    media_channels=["tv", bad],
                    channel="tv",
                    window=(1, 2),
                )
            with pytest.raises(ValueError, match="must not contain missing names"):
                meridian_calibration_mask(
                    media_times=[1, 2],
                    media_channels=["tv", "search"],
                    channel=["tv", bad],
                    window=(1, 2),
                )

    def test_duplicate_channels_raise(self):
        with pytest.raises(ValueError, match="duplicate channel"):
            meridian_calibration_mask(
                media_times=[1, 2],
                media_channels=["tv", "tv"],
                channel="tv",
                window=(1, 2),
            )
        with pytest.raises(ValueError, match="duplicate name"):
            meridian_calibration_mask(
                media_times=[1, 2],
                media_channels=["tv", "search"],
                channel=["tv", "tv"],
                window=(1, 2),
            )

    def test_empty_channel_and_media_channels(self):
        with pytest.raises(ValueError, match="media_channels must be non-empty"):
            meridian_calibration_mask(
                media_times=[1, 2], media_channels=[], channel="tv", window=(1, 2)
            )
        with pytest.raises(ValueError, match="at least one experiment channel"):
            meridian_calibration_mask(
                media_times=[1, 2], media_channels=["tv"], channel=[], window=(1, 2)
            )

    def test_window_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly"):
            meridian_calibration_mask(
                media_times=[1, 2, 3],
                media_channels=["tv"],
                channel="tv",
                window=(1, 2, 3),
            )

    def test_window_empty_labels_raises(self):
        with pytest.raises(ValueError, match="at least one time label"):
            meridian_calibration_mask(
                media_times=[1, 2], media_channels=["tv"], channel="tv", window=[]
            )

    def test_reversed_bounds_raises(self):
        with pytest.raises(ValueError, match="after window end"):
            meridian_calibration_mask(
                media_times=[1, 2, 3],
                media_channels=["tv"],
                channel="tv",
                window=(3, 1),
            )

    def test_bounds_select_nothing_raises(self):
        with pytest.raises(ValueError, match="selects no"):
            meridian_calibration_mask(
                media_times=[1, 2, 3],
                media_channels=["tv"],
                channel="tv",
                window=(7, 9),
            )

    def test_unorderable_bounds_raise(self):
        with pytest.raises(ValueError, match="order-compared"):
            meridian_calibration_mask(
                media_times=["a", "b", "c"],
                media_channels=["tv"],
                channel="tv",
                window=(1, 2),
            )

    def test_unparseable_datetime_bound_raises(self):
        with pytest.raises(ValueError, match="coerced"):
            meridian_calibration_mask(
                media_times=self._TIMES,
                media_channels=["tv"],
                channel="tv",
                window=("not-a-date", "2024-01-29"),
            )

    def test_tz_mismatch_fails_closed_both_ways(self):
        aware = pd.date_range("2024-01-01", periods=3, freq="W-MON", tz="UTC")
        with pytest.raises(ValueError, match="timezone"):
            meridian_calibration_mask(
                media_times=aware,
                media_channels=["tv"],
                channel="tv",
                window=("2024-01-01", "2024-01-15"),
            )
        naive = pd.date_range("2024-01-01", periods=3, freq="W-MON")
        with pytest.raises(ValueError, match="timezone"):
            meridian_calibration_mask(
                media_times=naive,
                media_channels=["tv"],
                channel="tv",
                window=(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-15", tz="UTC")),
            )

    def test_missing_bound_raises(self):
        for bad in (None, math.nan, pd.NaT, pd.NA):
            with pytest.raises(ValueError, match="must not be missing"):
                meridian_calibration_mask(
                    media_times=[1, 2, 3],
                    media_channels=["tv"],
                    channel="tv",
                    window=(bad, 2),
                )

    def test_missing_window_label_raises_named(self):
        # Missing labels fail closed on BOTH coordinate kinds, naming the
        # actual input (not its NaT coercion).
        times = pd.date_range("2024-01-01", periods=3, freq="W-MON")
        for bad in (None, math.nan, pd.NaT, pd.NA):
            with pytest.raises(ValueError, match="must not be missing"):
                meridian_calibration_mask(
                    media_times=times,
                    media_channels=["tv"],
                    channel="tv",
                    window=["2024-01-01", bad],
                )
        with pytest.raises(ValueError, match="must not be missing"):
            meridian_calibration_mask(
                media_times=[1, 2, 3],
                media_channels=["tv"],
                channel="tv",
                window=[1, None],
            )

    def test_wrong_typed_media_times(self):
        multi = pd.MultiIndex.from_tuples([(1, 2), (3, 4)])
        for bad in ("abc", {"a": 1}, 5, multi, np.zeros((2, 2)), np.array(1.0)):
            with pytest.raises(TypeError):
                meridian_calibration_mask(
                    media_times=bad, media_channels=["tv"], channel="tv", window=(1, 2)
                )
        # Tuple-valued labels pass the outer gate but pd.Index promotes them to
        # a MultiIndex - rejected after construction.
        with pytest.raises(TypeError, match="MultiIndex"):
            meridian_calibration_mask(
                media_times=[(1, 2), (3, 4)],
                media_channels=["tv"],
                channel="tv",
                window=(1, 2),
            )

    def test_wrong_typed_media_channels(self):
        for bad in ("abc", {"a": 1}, 5, np.zeros((2, 2))):
            with pytest.raises(TypeError):
                meridian_calibration_mask(
                    media_times=[1, 2], media_channels=bad, channel="tv", window=(1, 2)
                )

    def test_wrong_typed_channel_and_window(self):
        with pytest.raises(TypeError):
            meridian_calibration_mask(
                media_times=[1, 2],
                media_channels=["tv"],
                channel={"a": 1},
                window=(1, 2),
            )
        with pytest.raises(TypeError):
            meridian_calibration_mask(
                media_times=[1, 2],
                media_channels=["tv"],
                channel=np.array(1.0),
                window=(1, 2),
            )
        for bad_window in (5, "2024-01-01", {"a": 1}):
            with pytest.raises(TypeError, match=r"\(start, end\) tuple"):
                meridian_calibration_mask(
                    media_times=[1, 2],
                    media_channels=["tv"],
                    channel="tv",
                    window=bad_window,
                )
        with pytest.raises(TypeError, match="must be scalar labels"):
            meridian_calibration_mask(
                media_times=[1, 2, 3],
                media_channels=["tv"],
                channel="tv",
                window=(np.array([1, 2]), 3),
            )


class TestToCodeArrayMask:
    """to_code() ndarray route: validation, serialization, round-trips."""

    def _prior(self):
        return to_meridian_roi_prior(
            incremental_outcome=9_600.0,
            incremental_outcome_se=2_400.0,
            spend=20_000.0,
        )

    def _builder_mask(self):
        return meridian_calibration_mask(
            media_times=pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            media_channels=["search", "tv"],
            channel="tv",
            window=("2024-01-15", "2024-01-29"),
        )

    @staticmethod
    def _exec_prelude(code, stop_marker):
        ns = {}
        prelude = code[code.index("import numpy as np") : code.index(stop_marker)]
        exec(compile(prelude, "<prelude>", "exec"), ns)
        return ns["roi_calibration_period"]

    def test_array_mask_prelude_and_slot(self):
        code = self._prior().to_code(
            channel="tv",
            media_channels=["search", "tv"],
            roi_calibration_period=self._builder_mask(),
        )
        assert "import numpy as np" in code
        assert "np.ones((5, 2), dtype=bool)" in code
        assert "roi_calibration_period[:, [1]] = False" in code
        assert "np.ix_([2, 3, 4], [1])" in code
        assert "roi_calibration_period=roi_calibration_period" in code

    def test_array_mask_round_trip(self):
        from diff_diff.mmm import _mask_prelude

        mask = self._builder_mask()
        # Helper-level round-trip.
        ns = {}
        exec(compile(_mask_prelude(mask), "<prelude>", "exec"), ns)
        assert np.array_equal(ns["roi_calibration_period"], mask)
        assert ns["roi_calibration_period"].dtype == np.bool_
        assert ns["roi_calibration_period"].shape == mask.shape
        # Snippet-slice round-trip (multi-channel snippet: prelude ends at mu =).
        code = self._prior().to_code(
            channel="tv", media_channels=["search", "tv"], roi_calibration_period=mask
        )
        rebuilt = self._exec_prelude(code, "mu = ")
        assert np.array_equal(rebuilt, mask)

    def test_array_mask_per_channel_windows_round_trip(self):
        # Different window per channel: two column groups in the prelude, each
        # cleared and re-set, must reproduce the mask exactly.
        mask = np.zeros((4, 2), dtype=bool)
        mask[0:2, 0] = True
        mask[2:4, 1] = True
        from diff_diff.mmm import _mask_prelude

        prelude = _mask_prelude(mask)
        assert prelude.count("= False") == 2  # one clear per column group
        code = self._prior().to_code(
            channel="tv", media_channels=["search", "tv"], roi_calibration_period=mask
        )
        rebuilt = self._exec_prelude(code, "mu = ")
        assert np.array_equal(rebuilt, mask)
        assert rebuilt.dtype == np.bool_

    def test_array_mask_all_true_round_trip(self):
        mask = np.ones((3, 2), dtype=bool)
        from diff_diff.mmm import _mask_prelude

        prelude = _mask_prelude(mask)
        assert "= False" not in prelude  # all-True needs only the ones init
        ns = {}
        exec(compile(prelude, "<prelude>", "exec"), ns)
        assert np.array_equal(ns["roi_calibration_period"], mask)

    def test_array_mask_channel_count_mismatch(self):
        mask = np.ones((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="channel column"):
            self._prior().to_code(
                channel="tv", media_channels=["search", "tv"], roi_calibration_period=mask
            )

    def test_array_mask_single_channel_column_check(self):
        with pytest.raises(ValueError, match="exactly 1 column"):
            self._prior().to_code(
                single_channel=True, roi_calibration_period=np.ones((3, 2), dtype=bool)
            )
        code = self._prior().to_code(
            single_channel=True, roi_calibration_period=np.ones((3, 1), dtype=bool)
        )
        rebuilt = self._exec_prelude(code, "roi_prior = ")
        assert rebuilt.shape == (3, 1)

    def test_array_mask_all_false_rejected(self):
        with pytest.raises(ValueError, match="all False"):
            self._prior().to_code(
                single_channel=True, roi_calibration_period=np.zeros((3, 1), dtype=bool)
            )

    def test_array_mask_all_false_column_rejected(self):
        # Meridian aggregates each channel's calibration spend through its mask
        # column, so ANY all-False column (here 'tv') is rejected even when
        # other columns carry Trues - Google's convention gives non-experiment
        # channels ALL periods, never none.
        mask = np.zeros((3, 2), dtype=bool)
        mask[:, 0] = True  # 'search' all periods; 'tv' entirely False
        with pytest.raises(ValueError, match="entirely False"):
            self._prior().to_code(
                channel="tv", media_channels=["search", "tv"], roi_calibration_period=mask
            )
        # The mirror direction: every column has a True -> accepted.
        mask[1, 1] = True
        code = self._prior().to_code(
            channel="tv", media_channels=["search", "tv"], roi_calibration_period=mask
        )
        assert "roi_calibration_period=roi_calibration_period" in code

    def test_mroi_priors_cannot_be_time_scoped(self):
        # Meridian 1.7.0's ModelSpec rejects roi_calibration_period unless the
        # media prior type is 'roi' - both the ndarray and expression routes
        # must fail closed for mroi_m; full_model_window stays valid.
        mroi = to_meridian_roi_prior(
            incremental_outcome=9_600.0,
            incremental_outcome_se=2_400.0,
            spend=20_000.0,
            parameter="mroi_m",
        )
        with pytest.raises(ValueError, match="unless the media prior type"):
            mroi.to_code(
                single_channel=True,
                roi_calibration_period=np.ones((3, 1), dtype=bool),
            )
        with pytest.raises(ValueError, match="unless the media prior type"):
            mroi.to_code(single_channel=True, roi_calibration_period="my_mask")
        code = mroi.to_code(single_channel=True, full_model_window=True)
        assert "roi_calibration_period=None" in code
        # The no-time-scope error is parameter-aware: for mroi_m it recommends
        # ONLY the route that Meridian accepts, never the mask routes the next
        # validation would reject.
        with pytest.raises(ValueError, match="exactly one route"):
            mroi.to_code(single_channel=True)

    def test_array_mask_bad_shape_and_values(self):
        with pytest.raises(ValueError, match="2-D"):
            self._prior().to_code(
                single_channel=True, roi_calibration_period=np.ones(3, dtype=bool)
            )
        with pytest.raises(ValueError, match="non-empty"):
            self._prior().to_code(
                single_channel=True, roi_calibration_period=np.ones((0, 2), dtype=bool)
            )
        bad = np.zeros((3, 1))
        bad[0, 0] = 0.5
        with pytest.raises(ValueError, match="only 0/1"):
            self._prior().to_code(single_channel=True, roi_calibration_period=bad)
        with pytest.raises(ValueError, match="only 0/1"):
            self._prior().to_code(
                single_channel=True,
                roi_calibration_period=np.array([["a"]], dtype=object),
            )

    def test_array_mask_float01_google_parity(self):
        # Google's configure-model example builds the mask with float np.zeros.
        mask = np.zeros((3, 1))
        mask[1, 0] = 1.0
        code = self._prior().to_code(single_channel=True, roi_calibration_period=mask)
        rebuilt = self._exec_prelude(code, "roi_prior = ")
        assert rebuilt.dtype == np.bool_
        assert np.array_equal(rebuilt, mask.astype(bool))

    def test_array_mask_masked_array_rejected(self):
        masked = np.ma.masked_array(np.ones((2, 1)), mask=[[True], [False]])
        with pytest.raises(TypeError, match="masked arrays are not accepted"):
            self._prior().to_code(single_channel=True, roi_calibration_period=masked)

    def test_non_str_non_array_rejected(self):
        with pytest.raises(TypeError, match="expression string or a boolean numpy array"):
            self._prior().to_code(single_channel=True, roi_calibration_period=[[True, False]])

    def test_time_scope_error_mentions_builder(self):
        with pytest.raises(ValueError, match="meridian_calibration_mask"):
            self._prior().to_code(single_channel=True)

    def test_empty_prelude_routes_stay_clean(self):
        # Both empty-{mask_prelude} routes emit no numpy artifacts.
        full_window = self._prior().to_code(single_channel=True, full_model_window=True)
        str_expr = self._prior().to_code(single_channel=True, roi_calibration_period="my_mask")
        for code in (full_window, str_expr):
            assert "import numpy" not in code
            assert "np.zeros(" not in code


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
        # CS gives a headline ATT; the container route now exists
        # (aggregation_result= + a NUMERIC scale=, see TestAggregationWorkflow),
        # but CS's own estimator-owned aggregation producing a total incremental
        # outcome remains undelivered (DEFERRED.md remainder row) - a naive
        # att x count need not reproduce CS's cohort weights or variance. Here we
        # simply confirm the fitted numbers flow through the exporter once the
        # caller has supplied a total and its SE.
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


def _make_agg(
    level="simple",
    estimator="ImputationDiD",
    n_kind="obs",
    labels=("overall",),
    atts=(2.0,),
    ses=(0.5,),
    ns=(30.0,),
    targets=None,
):
    """Hand-built AggregationResult mirroring the producers' field conventions."""
    k = len(labels)
    return AggregationResult(
        level=level,
        label=np.array(labels, dtype=object),
        target=np.array(targets if targets is not None else ["att"] * k, dtype=object),
        att=np.array(atts, dtype=float),
        se=np.array(ses, dtype=float),
        t_stat=np.full(k, 1.0),
        p_value=np.full(k, 0.5),
        conf_int_lower=np.full(k, 0.0),
        conf_int_upper=np.full(k, 1.0),
        n=np.array(ns, dtype=float),
        df=np.full(k, np.nan),
        n_kind=n_kind,
        estimator=estimator,
    )


class TestAggregationExtraction:
    """Container-mode routing/rejection matrix, on hand-built containers."""

    def test_rejects_results_object_with_remedy(self):
        result = DifferenceInDifferences().fit(
            generate_did_data(n_units=20, n_periods=2, treatment_period=1, seed=3),
            outcome="outcome",
            treatment="treated",
            post="post",
        )
        with pytest.raises(TypeError, match="have no container-mode route"):
            to_pymc_marketing_lift_test(channel="tv", x=1.0, delta_x=1.0, aggregation_result=result)

    def test_event_study_container_gets_results_object_message(self):
        from diff_diff import EventStudyResults

        es = EventStudyResults(
            event_time=np.array([-1, 0, 1]),
            att=np.array([0.0, 1.0, 1.5]),
            se=np.array([np.nan, 0.2, 0.3]),
            t_stat=np.array([np.nan, 5.0, 5.0]),
            p_value=np.array([np.nan, 0.01, 0.01]),
            conf_int_lower=np.array([np.nan, 0.6, 0.9]),
            conf_int_upper=np.array([np.nan, 1.4, 2.1]),
            is_reference=np.array([True, False, False]),
            n=np.array([np.nan, 10.0, 10.0]),
        )
        # EventStudyResults subclasses BaseResults, so it must get the 1a
        # results-object message, never the generic 1b one.
        with pytest.raises(TypeError, match="EventStudyResults and estimators"):
            to_pymc_marketing_lift_test(channel="tv", x=1.0, delta_x=1.0, aggregation_result=es)

    def test_rejects_non_container_type(self):
        from diff_diff.results_base import Diagnostic

        class _SomeDiagnostic(Diagnostic):
            """Stand-in for the Diagnostic-marker roster (not BaseResults)."""

        for bad in (object(), _SomeDiagnostic()):
            with pytest.raises(TypeError, match="Only AggregationResult is supported"):
                to_pymc_marketing_lift_test(
                    channel="tv", x=1.0, delta_x=1.0, aggregation_result=bad
                )

    def test_results_object_without_scale_still_type_error(self):
        # The type check runs before the scale-required check.
        result = DifferenceInDifferences().fit(
            generate_did_data(n_units=20, n_periods=2, treatment_period=1, seed=3),
            outcome="outcome",
            treatment="treated",
            post="post",
        )
        with pytest.raises(TypeError, match="must be an AggregationResult"):
            to_meridian_roi_prior(aggregation_result=result, spend=10.0)

    def test_rejects_unsupported_level(self):
        agg = _make_agg(level="calendar")
        with pytest.raises(ValueError, match="level must be 'simple' or 'group'"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale=1.0
            )

    def test_rejects_empty_container(self):
        agg = _make_agg(level="group", labels=(), atts=(), ses=(), ns=())
        with pytest.raises(ValueError, match="has no rows"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale=1.0
            )

    def test_rejects_multi_row_simple_container(self):
        agg = _make_agg(
            level="simple",
            labels=("overall", "overall2"),
            atts=(1.0, 2.0),
            ses=(0.1, 0.1),
            ns=(5.0, 5.0),
        )
        with pytest.raises(ValueError, match="out of contract"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale=1.0
            )

    def test_rejects_non_att_target_whole_container(self):
        # ContinuousDiD's simple container carries att + acrt rows; the acrt row
        # is a dose derivative, so the WHOLE container is rejected (no filtering).
        agg = _make_agg(
            level="simple",
            estimator="ContinuousDiD",
            n_kind="units",
            labels=("overall", "overall"),
            atts=(1.0, 0.2),
            ses=(0.1, 0.05),
            ns=(40.0, 40.0),
            targets=("att", "acrt"),
        )
        with pytest.raises(ValueError, match="rejected whole"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale=1.0
            )

    def test_rejects_unusable_inference(self):
        for atts, ses in (
            ((math.nan,), (0.5,)),
            ((math.inf,), (0.5,)),
            ((1.0,), (0.0,)),
            ((1.0,), (-1.0,)),
            ((1.0,), (math.nan,)),
            ((1.0,), (math.inf,)),
        ):
            agg = _make_agg(atts=atts, ses=ses)
            with pytest.raises(ValueError, match="no usable point estimate/SE"):
                to_pymc_marketing_lift_test(
                    channel="tv",
                    x=1.0,
                    delta_x=1.0,
                    aggregation_result=agg,
                    scale=1.0,
                )

    def test_auto_scale_allowlisted_estimators(self):
        for estimator in ("ImputationDiD", "TwoStageDiD"):
            agg = _make_agg(estimator=estimator, atts=(2.0,), ses=(0.5,), ns=(30.0,))
            df = to_pymc_marketing_lift_test(
                channel="tv",
                x=100.0,
                delta_x=50.0,
                aggregation_result=agg,
                scale="auto",
            )
            assert df.loc[0, "delta_y"] == 2.0 * 30.0
            assert df.loc[0, "sigma"] == 0.5 * 30.0

    def test_auto_scale_rejected_off_allowlist(self):
        cases = [
            ("CallawaySantAnna", "treated\\+control"),
            ("EfficientDiD", "disjoint treated\\+control"),
            ("StackedDiD", "deduplicated distinct-treated-unit"),
            ("SomethingNew", "not audited"),
            (None, "not audited"),
        ]
        for estimator, fragment in cases:
            agg = _make_agg(estimator=estimator)
            with pytest.raises(ValueError, match=fragment):
                to_pymc_marketing_lift_test(
                    channel="tv",
                    x=1.0,
                    delta_x=1.0,
                    aggregation_result=agg,
                    scale="auto",
                )

    def test_auto_scale_n_kind_drift_guard(self):
        agg = _make_agg(estimator="ImputationDiD", n_kind="units")
        with pytest.raises(ValueError, match="schema has drifted"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale="auto"
            )

    def test_auto_scale_rejects_bad_n(self):
        for n in (math.nan, 0.0, -3.0, math.inf, -math.inf):
            agg = _make_agg(ns=(n,))
            with pytest.raises(ValueError, match="cannot auto-derive scale"):
                to_pymc_marketing_lift_test(
                    channel="tv",
                    x=1.0,
                    delta_x=1.0,
                    aggregation_result=agg,
                    scale="auto",
                )

    def test_missing_scale_names_both_routes(self):
        agg = _make_agg()
        with pytest.raises(ValueError, match="scale is required with aggregation_result"):
            to_pymc_marketing_lift_test(channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg)

    def test_boolean_scale_rejected(self):
        # float(True) == 1.0 would silently scale by one - a plausible typo
        # for scale="auto" - so booleans fail closed, scalar and per-row.
        agg = _make_agg()
        for bad in (True, np.bool_(True), [True]):
            with pytest.raises(ValueError, match="got a boolean"):
                to_pymc_marketing_lift_test(
                    channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale=bad
                )
        with pytest.raises(ValueError, match="got a boolean"):
            to_meridian_roi_prior(aggregation_result=agg, scale=True, spend=10.0)

    def test_non_auto_string_scale_rejected(self):
        agg = _make_agg()
        with pytest.raises(ValueError, match="or the string 'auto'"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale="AUTO"
            )

    def test_explicit_scale_scalar_and_per_row(self):
        agg = _make_agg(
            level="group",
            estimator="CallawaySantAnna",
            n_kind="cells",
            labels=(2001, 2002),
            atts=(1.0, 2.0),
            ses=(0.1, 0.2),
            ns=(4.0, 5.0),
        )
        df = to_pymc_marketing_lift_test(
            channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale=10.0
        )
        assert df["delta_y"].tolist() == [10.0, 20.0]
        df2 = to_pymc_marketing_lift_test(
            channel="tv", x=1.0, delta_x=1.0, aggregation_result=agg, scale=[10.0, 100.0]
        )
        assert df2["delta_y"].tolist() == [10.0, 200.0]
        assert df2["sigma"].tolist() == [pytest.approx(1.0), pytest.approx(20.0)]

    def test_scale_validation_reuses_existing_messages(self):
        agg = _make_agg(
            level="group",
            labels=(1, 2),
            atts=(1.0, 2.0),
            ses=(0.1, 0.2),
            ns=(4.0, 5.0),
        )
        with pytest.raises(ValueError, match="scale has length 3 but 2"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=1.0,
                delta_x=1.0,
                aggregation_result=agg,
                scale=[1.0, 2.0, 3.0],
            )
        for bad in (0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="scale must be finite and > 0"):
                to_pymc_marketing_lift_test(
                    channel="tv",
                    x=1.0,
                    delta_x=1.0,
                    aggregation_result=agg,
                    scale=bad,
                )

    def test_row_order_is_to_dataframe_order(self):
        # Producer order 2003, 2001; to_dataframe() sorts sortable labels, so the
        # 2001 row comes first and per-row scale aligns to the SORTED order.
        agg = _make_agg(
            level="group",
            labels=(2003, 2001),
            atts=(3.0, 1.0),
            ses=(0.3, 0.1),
            ns=(7.0, 5.0),
        )
        df = to_pymc_marketing_lift_test(
            channel="tv",
            x=1.0,
            delta_x=1.0,
            aggregation_result=agg,
            scale=[10.0, 100.0],
        )
        # Sorted order: label 2001 (att=1.0) gets scale 10, label 2003 gets 100.
        assert df["delta_y"].tolist() == [10.0, 300.0]


class TestLiftTestFromAggregation:
    def test_simple_container_one_row(self):
        agg = _make_agg(atts=(2.5,), ses=(0.4,), ns=(12.0,))
        df = to_pymc_marketing_lift_test(
            channel="tv",
            x=500.0,
            delta_x=250.0,
            aggregation_result=agg,
            scale="auto",
            dims={"geo": "US"},
        )
        assert list(df.columns) == ["channel", "geo", "x", "delta_x", "delta_y", "sigma"]
        assert df.loc[0, "delta_y"] == 2.5 * 12.0
        assert df.loc[0, "sigma"] == pytest.approx(0.4 * 12.0)

    def test_group_container_broadcast_interplay(self):
        agg = _make_agg(
            level="group",
            labels=(1, 2, 3),
            atts=(1.0, 2.0, 3.0),
            ses=(0.1, 0.2, 0.3),
            ns=(10.0, 10.0, 10.0),
        )
        df = to_pymc_marketing_lift_test(
            channel="tv",
            x=[100.0, 200.0, 300.0],
            delta_x=50.0,
            aggregation_result=agg,
            scale="auto",
        )
        assert len(df) == 3
        assert df["x"].tolist() == [100.0, 200.0, 300.0]
        assert df["delta_x"].tolist() == [50.0, 50.0, 50.0]
        with pytest.raises(ValueError, match="x has length 2 but 3"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=[100.0, 200.0],
                delta_x=50.0,
                aggregation_result=agg,
                scale="auto",
            )

    def test_mode_exclusivity(self):
        agg = _make_agg()
        with pytest.raises(ValueError, match="not both"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=1.0,
                delta_x=1.0,
                delta_y=5.0,
                sigma=1.0,
                aggregation_result=agg,
                scale=1.0,
            )
        # Container plus exactly ONE lone member is also "not both" (or-joined).
        with pytest.raises(ValueError, match="not both"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=1.0,
                delta_x=1.0,
                delta_y=5.0,
                aggregation_result=agg,
                scale=1.0,
            )
        with pytest.raises(ValueError, match="delta_y and sigma are required"):
            to_pymc_marketing_lift_test(channel="tv", x=1.0, delta_x=1.0)
        with pytest.raises(ValueError, match="sigma is required"):
            to_pymc_marketing_lift_test(channel="tv", x=1.0, delta_x=1.0, delta_y=5.0)
        with pytest.raises(ValueError, match="delta_y is required"):
            to_pymc_marketing_lift_test(channel="tv", x=1.0, delta_x=1.0, sigma=1.0)
        with pytest.raises(ValueError, match="scale only applies with"):
            to_pymc_marketing_lift_test(
                channel="tv", x=1.0, delta_x=1.0, delta_y=5.0, sigma=1.0, scale=2.0
            )

    def test_sign_policy_applies_to_derived_rows(self):
        agg = _make_agg(atts=(-2.0,), ses=(0.5,), ns=(10.0,))
        with pytest.raises(ValueError, match="NonMonotonicError"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=100.0,
                delta_x=50.0,
                aggregation_result=agg,
                scale="auto",
            )
        with pytest.warns(UserWarning, match="Keeping invalid"):
            df = to_pymc_marketing_lift_test(
                channel="tv",
                x=100.0,
                delta_x=50.0,
                aggregation_result=agg,
                scale="auto",
                on_wrong_sign="keep",
            )
        assert df.loc[0, "delta_y"] == -20.0


class TestMeridianFromAggregation:
    def test_simple_container_prior_math(self):
        agg = _make_agg(atts=(2.0,), ses=(0.5,), ns=(30.0,))
        prior = to_meridian_roi_prior(aggregation_result=agg, scale="auto", spend=1_000.0)
        total, total_se = 2.0 * 30.0, 0.5 * 30.0
        assert prior.roi_mean == pytest.approx(total / 1_000.0)
        assert prior.roi_sd == pytest.approx(total_se / 1_000.0)
        # Lognormal closed-form roundtrip (Google's parameterization).
        m, s = prior.roi_mean, prior.roi_sd
        assert prior.sigma == pytest.approx(math.sqrt(math.log(1 + (s / m) ** 2)))
        assert prior.mu == pytest.approx(math.log(m) - 0.5 * math.log(1 + (s / m) ** 2))

    def test_group_container_per_cohort_spend_pooling(self):
        agg = _make_agg(
            level="group",
            estimator="TwoStageDiD",
            labels=(2001, 2002),
            atts=(1.0, 2.0),
            ses=(0.1, 0.2),
            ns=(10.0, 20.0),
        )
        spends = [500.0, 1_500.0]
        prior = to_meridian_roi_prior(aggregation_result=agg, scale="auto", spend=spends)
        rois = [1.0 * 10.0 / 500.0, 2.0 * 20.0 / 1_500.0]
        sds = [0.1 * 10.0 / 500.0, 0.2 * 20.0 / 1_500.0]
        weights = [500.0 / 2_000.0, 1_500.0 / 2_000.0]
        assert prior.roi_mean == pytest.approx(sum(w * r for w, r in zip(weights, rois)))
        assert prior.roi_sd == pytest.approx(math.hypot(*(w * s for w, s in zip(weights, sds))))
        assert [e.spend for e in prior.per_experiment] == spends

    def test_mode_exclusivity(self):
        agg = _make_agg()
        with pytest.raises(ValueError, match="not both"):
            to_meridian_roi_prior(
                incremental_outcome=10.0,
                incremental_outcome_se=1.0,
                aggregation_result=agg,
                scale=1.0,
                spend=100.0,
            )
        with pytest.raises(ValueError, match="not both"):
            to_meridian_roi_prior(
                incremental_outcome_se=1.0,
                aggregation_result=agg,
                scale=1.0,
                spend=100.0,
            )
        with pytest.raises(ValueError, match="incremental_outcome and incremental_outcome_se are"):
            to_meridian_roi_prior(spend=100.0)
        with pytest.raises(ValueError, match="incremental_outcome_se is required"):
            to_meridian_roi_prior(incremental_outcome=10.0, spend=100.0)
        with pytest.raises(ValueError, match="incremental_outcome is required"):
            to_meridian_roi_prior(incremental_outcome_se=1.0, spend=100.0)
        with pytest.raises(ValueError, match="scale only applies with"):
            to_meridian_roi_prior(
                incremental_outcome=10.0,
                incremental_outcome_se=1.0,
                scale=2.0,
                spend=100.0,
            )

    def test_negative_pooled_att_hits_positivity_guard(self):
        agg = _make_agg(atts=(-2.0,), ses=(0.5,), ns=(30.0,))
        with pytest.raises(ValueError, match="positive support"):
            to_meridian_roi_prior(aggregation_result=agg, scale="auto", spend=1_000.0)


class TestAggregationWorkflow:
    """End-to-end: fit -> aggregate -> export, with container-independent oracles
    (treated unit-period counts computed from the input frame, never agg.n)."""

    def test_imputation_auto_scale(self):
        data = generate_staggered_data(n_units=60, n_periods=6, seed=11)
        res = ImputationDiD().fit(
            data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        agg = res.aggregate("simple")
        assert agg.estimator == "ImputationDiD"
        # Frame-derived treated-obs count (never-treated coded first_treat == 0).
        treated_obs = int(
            ((data["first_treat"] > 0) & (data["period"] >= data["first_treat"])).sum()
        )
        spend = 50_000.0
        prior = to_meridian_roi_prior(aggregation_result=agg, scale="auto", spend=spend)
        manual = to_meridian_roi_prior(
            incremental_outcome=float(agg.att[0]) * treated_obs,
            incremental_outcome_se=float(agg.se[0]) * treated_obs,
            spend=spend,
        )
        assert prior.roi_mean == pytest.approx(manual.roi_mean)
        assert prior.roi_sd == pytest.approx(manual.roi_sd)

    def test_two_stage_group_auto_scale(self):
        data = generate_staggered_data(n_units=60, n_periods=6, seed=11)
        res = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        agg = res.aggregate("group")
        frame = agg.to_dataframe()
        df = to_pymc_marketing_lift_test(
            channel="tv",
            x=100.0,
            delta_x=50.0,
            aggregation_result=agg,
            scale="auto",
            on_wrong_sign="keep",
        )
        assert len(df) == len(frame)
        for i, (_, row) in enumerate(frame.iterrows()):
            cohort = row["label"]
            cohort_obs = int(
                ((data["first_treat"] == cohort) & (data["period"] >= data["first_treat"])).sum()
            )
            assert df.loc[i, "delta_y"] == pytest.approx(row["att"] * cohort_obs)

    def test_callaway_explicit_scale_path(self):
        data = generate_staggered_data(n_units=60, n_periods=6, seed=11)
        cs = CallawaySantAnna().fit(
            data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        agg = cs.aggregate("simple")
        with pytest.raises(ValueError, match="treated\\+control"):
            to_meridian_roi_prior(aggregation_result=agg, scale="auto", spend=1_000.0)
        scale = 40.0  # caller-derived treated unit-periods for THEIR scoping
        prior = to_meridian_roi_prior(aggregation_result=agg, scale=scale, spend=1_000.0)
        assert prior.roi_mean == pytest.approx(float(agg.att[0]) * scale / 1_000.0)

    def test_imputation_nonfinite_tau_divergence(self):
        # tests/test_imputation.py recipe: drop never-treated obs at t=5 so that
        # period's time FE is unidentified -> non-finite tau-hat there, while the
        # container's n stays the raw |Omega_1| count. This is the documented
        # scale="auto" overcount scenario, pinned behaviorally.
        rng = np.random.default_rng(42)
        rows = []
        for i in range(40):
            ft = 2 if i < 20 else 99
            for t in range(6):
                if ft == 99 and t == 5:
                    continue
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0
                rows.append({"unit": i, "time": t, "outcome": y, "first_treat": ft})
        data = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ImputationDiD().fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )
        agg = res.aggregate("simple")
        full_omega_1 = 20 * 4  # 20 treated units, periods 2..5
        identified_obs = 20 * 3  # (cohort, t=5) cell is unidentified
        assert float(agg.n[0]) == full_omega_1
        assert float(agg.n[0]) > identified_obs  # auto-scale would overcount

    def test_two_stage_nonfinite_ytilde_divergence(self):
        # Analogous construction: with no untreated observations at t=5, stage 1
        # (fit on untreated obs only) cannot identify that period's time FE, so
        # treated rows there get non-finite y_tilde and drop from the ATT's
        # support - while the simple container's n stays the pre-filter |Omega_1|.
        rng = np.random.default_rng(42)
        rows = []
        for i in range(40):
            ft = 2 if i < 20 else 99
            for t in range(6):
                if ft == 99 and t == 5:
                    continue
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0
                rows.append({"unit": i, "time": t, "outcome": y, "first_treat": ft})
        data = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD().fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )
        agg = res.aggregate("simple")
        full_omega_1 = 20 * 4
        post_filter_support = 20 * 3  # treated rows at t=5 lose their y_tilde
        assert float(agg.n[0]) == full_omega_1
        assert float(agg.n[0]) > post_filter_support

    def test_overflow_surfaces_downstream_messages(self):
        # Only the effect product overflows (huge att, tiny se), so the message
        # names the effect, per each exporter's own validation order.
        agg = _make_agg(atts=(1e300,), ses=(1e-300,), ns=(30.0,))
        with pytest.raises(ValueError, match="delta_y must be finite"):
            to_pymc_marketing_lift_test(
                channel="tv",
                x=1.0,
                delta_x=1.0,
                aggregation_result=agg,
                scale=1e10,
            )
        with pytest.raises(ValueError, match="incremental_outcome must be finite"):
            to_meridian_roi_prior(aggregation_result=agg, scale=1e10, spend=100.0)


class TestAggregationContractPin:
    """Pin exactly the AggregationResult surface mmm consumes, so container schema
    drift fails loudly here (independent of tests/test_aggregate_contract.py,
    which compares producer frames AGAINST the constant, not the constant itself).
    Containers at any alpha behave identically: alpha/CI/df/p are not consumed."""

    _CONSUMED_COLUMNS = {"label", "target", "att", "se", "n"}
    _NOT_CONSUMED_COLUMNS = {
        "level",  # consumed as the ATTRIBUTE, not the frame column
        "t_stat",
        "p_value",
        "conf_int_lower",
        "conf_int_upper",
        "weight",
        "df",
    }

    def test_aggregation_schema_literal_pin(self):
        from diff_diff.aggregation import AGGREGATION_SCHEMA

        # mmm consumes to_dataframe() rows; a schema change must be consciously
        # synced with _extract_aggregation_rows.
        assert AGGREGATION_SCHEMA == (
            "level",
            "label",
            "target",
            "att",
            "se",
            "t_stat",
            "p_value",
            "conf_int_lower",
            "conf_int_upper",
            "n",
            "weight",
            "df",
        )

    def test_consumed_surface_is_closed(self):
        from diff_diff.aggregation import AGGREGATION_SCHEMA

        # The consumed + deliberately-not-consumed sets jointly cover the schema,
        # so this pin is a closed definition of mmm's read surface.
        assert self._CONSUMED_COLUMNS | self._NOT_CONSUMED_COLUMNS == set(AGGREGATION_SCHEMA)
        agg = _make_agg()
        for attr in ("level", "n_kind", "estimator"):
            assert hasattr(agg, attr)
        assert callable(agg.to_dataframe)
        frame = agg.to_dataframe()
        for col in self._CONSUMED_COLUMNS:
            assert col in frame.columns

    def test_public_annotations_runtime_resolvable(self):
        # AggregationResult is imported at module runtime (not TYPE_CHECKING),
        # so annotation consumers can resolve the exporters' signatures.
        import typing

        for fn in (to_pymc_marketing_lift_test, to_meridian_roi_prior):
            hints = typing.get_type_hints(fn)
            assert "aggregation_result" in hints and "scale" in hints

    def test_allowlist_strings_match_provenance_derivation(self):
        from diff_diff.imputation_results import ImputationDiDResults
        from diff_diff.mmm import _SCALE_AUTO_ESTIMATORS
        from diff_diff.two_stage_results import TwoStageDiDResults

        derived = {
            cls.__name__.replace("Results", "")
            for cls in (ImputationDiDResults, TwoStageDiDResults)
        }
        assert derived == set(_SCALE_AUTO_ESTIMATORS)


def test_public_exports():
    import diff_diff

    assert diff_diff.to_pymc_marketing_lift_test is to_pymc_marketing_lift_test
    assert diff_diff.to_meridian_roi_prior is to_meridian_roi_prior
    assert diff_diff.meridian_calibration_mask is meridian_calibration_mask
    assert diff_diff.MeridianROIPrior is MeridianROIPrior
