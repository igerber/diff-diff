"""Meridian interop smoke tests (schema/defaults-drift canary).

These tests exercise :func:`diff_diff.mmm.to_meridian_roi_prior`,
:meth:`MeridianROIPrior.to_code`, and :func:`diff_diff.mmm.meridian_calibration_mask`
against the REAL installed google-meridian API - no sampling anywhere. Skipped
unless google-meridian >= 1.8 is installed (NOT a diff-diff dependency); the
dedicated ``mmm-interop.yml`` CI job installs it and runs this file, with an
import-canary step so an install failure cannot be silently skipped into a green
run.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "meridian",
    minversion="1.8",
    reason="google-meridian>=1.8 required (installed only in the mmm-interop CI job)",
)

import tensorflow_probability as tfp  # noqa: E402
from meridian.data import data_frame_input_data_builder as m_dfb  # noqa: E402
from meridian.model import model as m_model  # noqa: E402
from meridian.model import prior_distribution, spec  # noqa: E402

from diff_diff.mmm import (  # noqa: E402
    _MERIDIAN_PARAM_DEFAULTS,
    meridian_calibration_mask,
    to_meridian_roi_prior,
)

TIMES = [str(d.date()) for d in pd.date_range("2024-01-01", periods=8, freq="W-MON")]
CHANNELS = ["search", "tv"]


def _prior():
    return to_meridian_roi_prior(
        incremental_outcome=5000.0, incremental_outcome_se=800.0, spend=2000.0
    )


def _mask():
    return meridian_calibration_mask(
        media_times=TIMES,
        media_channels=CHANNELS,
        channel="search",
        window=(TIMES[4], TIMES[-1]),
    )


class TestMeridianDefaultsPin:
    def test_param_defaults_match_live_meridian(self):
        # The multi-channel to_code() snippet keeps non-experiment channels on what
        # it BELIEVES are Meridian's default LogNormal params - if Meridian changes
        # its defaults, this canary must fail rather than silently rot the snippet.
        # Meridian stores float32 tensors: compare after float32 rounding.
        live = prior_distribution.PriorDistribution()
        for param, (mu, sigma) in _MERIDIAN_PARAM_DEFAULTS.items():
            dist = getattr(live, param)
            live_mu = np.float32(np.asarray(dist.parameters["loc"]))
            live_sigma = np.float32(np.asarray(dist.parameters["scale"]))
            assert (live_mu, live_sigma) == (np.float32(mu), np.float32(sigma)), (
                f"Meridian default drift for {param}: "
                f"ours=({mu}, {sigma}) live=({live_mu}, {live_sigma})"
            )


class TestToCodeExecutes:
    def test_multi_channel_snippet_execs_with_value_retention(self):
        prior = _prior()
        mask = _mask()
        code = prior.to_code(channel="search", media_channels=CHANNELS, roi_calibration_period=mask)
        ns: dict = {}
        exec(code, ns)  # noqa: S102 - executing our own generated snippet is the test
        model_spec = ns["model_spec"]
        assert isinstance(model_spec, spec.ModelSpec)
        # Value retention: the exec'd prior's roi_m LogNormal carries our mu/sigma
        # for the experiment channel and Meridian's defaults for the other channel.
        dist = model_spec.prior.roi_m
        loc = np.asarray(dist.parameters["loc"], dtype=float)
        scale = np.asarray(dist.parameters["scale"], dtype=float)
        default_mu, default_sigma = _MERIDIAN_PARAM_DEFAULTS["roi_m"]
        np.testing.assert_allclose(loc, [prior.mu, default_mu], rtol=1e-6)
        np.testing.assert_allclose(scale, [prior.sigma, default_sigma], rtol=1e-6)
        # And the serialized mask round-trips exactly.
        assert np.array_equal(np.asarray(model_spec.roi_calibration_period), mask)

    def test_single_channel_snippet_execs(self):
        prior = _prior()
        code = prior.to_code(single_channel=True, full_model_window=True)
        ns: dict = {}
        exec(code, ns)  # noqa: S102
        model_spec = ns["model_spec"]
        assert isinstance(model_spec, spec.ModelSpec)
        dist = model_spec.prior.roi_m
        assert float(np.asarray(dist.parameters["loc"])) == pytest.approx(prior.mu)
        assert float(np.asarray(dist.parameters["scale"])) == pytest.approx(prior.sigma)
        assert model_spec.roi_calibration_period is None


class TestModelAcceptsExports:
    def test_mask_and_prior_accepted_by_meridian(self):
        prior = _prior()
        mask = _mask()
        roi_prior = tfp.distributions.LogNormal(
            [prior.mu, _MERIDIAN_PARAM_DEFAULTS["roi_m"][0]],
            [prior.sigma, _MERIDIAN_PARAM_DEFAULTS["roi_m"][1]],
            name="roi_m",
        )
        model_spec = spec.ModelSpec(
            prior=prior_distribution.PriorDistribution(roi_m=roi_prior),
            media_prior_type="roi",
            roi_calibration_period=mask,
        )
        # Build InputData through the PUBLIC DataFrameInputDataBuilder (the
        # route notebook 30 teaches) rather than Meridian's internal test-utils
        # fixtures, so upstream fixture churn cannot break this canary.
        rng = np.random.default_rng(1)
        rows = []
        for geo in ["g0", "g1", "g2", "g3"]:
            for t in TIMES:
                rows.append(
                    {
                        "geo": geo,
                        "time": t,
                        "sales": 1000.0 + 50.0 * rng.random(),
                        "population": 1_000_000.0,
                        "search_spend": 100.0 + 10.0 * rng.random(),
                        "tv_spend": 200.0 + 10.0 * rng.random(),
                    }
                )
        frame = pd.DataFrame(rows)
        input_data = (
            m_dfb.DataFrameInputDataBuilder(kpi_type="revenue")
            .with_kpi(frame, kpi_col="sales")
            .with_population(frame)
            .with_media(
                frame,
                media_cols=["search_spend", "tv_spend"],
                media_spend_cols=["search_spend", "tv_spend"],
                media_channels=CHANNELS,
            )
            .build()
        )
        mer = m_model.Meridian(input_data=input_data, model_spec=model_spec)
        assert mer is not None
        assert np.array_equal(np.asarray(mer.model_spec.roi_calibration_period), mask)
