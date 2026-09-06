"""Independent Deaner--Ku equations / Algorithms 1--2 verification."""

import numpy as np
import pytest

from diff_diff import DurationDiD
from diff_diff.duration_did import _inference
from tests.conftest import assert_nan_inference
from tests.test_duration_did import duration_panel, fit_duration


def test_eq_216_ph_ratio_direction_and_mean_of_ratios():
    data = duration_panel(control=(100, 80, 60, 40), treated=(100, 64, 36, 10), scale=1)
    r = DurationDiD(method="proportional_hazards", n_bootstrap=10, seed=4).fit(
        data, "absorbed", "group", "id", "date", post_periods=[3]
    )
    assert r.coefficient == pytest.approx(2)
    assert r.att == pytest.approx(0.4**2 - 0.1)
    # Noisy non-proportional pre-increments distinguish mean ratios from slopes.
    data = duration_panel(control=(100, 80, 60, 40), treated=(100, 70, 40, 10), scale=1)
    r = DurationDiD(method="proportional_hazards", n_bootstrap=10, seed=4).fit(
        data, "absorbed", "group", "id", "date", post_periods=[3]
    )
    d1, d0 = -np.log([0.7, 0.4]), -np.log([0.8, 0.6])
    assert r.coefficient == pytest.approx(np.mean(d1 / d0))
    assert abs(r.coefficient - np.dot(d0, d1) / np.dot(d0, d0)) > 0.01


def test_eq_32_cd_whole_population_denominators_and_baseline():
    r = fit_duration(n_bootstrap=5, seed=5)
    assert r.survival_curve.treated_survival.iloc[0] == 0.9
    assert r.survival_curve.control_survival.iloc[0] == 0.96
    assert r.coefficient == pytest.approx(0, abs=1e-15)
    np.testing.assert_allclose(r.effects.att, [0.1, 0.07])
    np.testing.assert_allclose(r.survival_curve.raw_counterfactual_survival.iloc[4:], [0.3, 0.15])


def test_algorithm_1_manual_pooled_history_bootstrap():
    data = duration_panel()
    b = 13
    seed = 81
    r = fit_duration(data, n_bootstrap=b, seed=seed)
    histories = data.pivot(index="id", columns="date", values="absorbed").to_numpy()
    group = data.groupby("id").group.first().to_numpy()
    rng = np.random.default_rng(seed)
    expected = []
    expected_contrasts = []
    for _ in range(b):
        idx = rng.integers(len(histories), size=len(histories))
        sampled, g = histories[idx], group[idx]
        s0, s1 = [1 - sampled[g == k].mean(axis=0) for k in (0, 1)]
        d0, d1 = -np.log(s0 / s0[0]), -np.log(s1 / s1[0])
        gap = (d1[1:4] - d0[1:4]) / np.arange(1, 4)
        c = gap.mean()
        expected.append(np.exp(np.log(s1[0]) - d0[4:] - np.arange(4, 6) * c) - s1[4:])
        expected_contrasts.append(gap[:2] - gap[2])
    expected = np.array(expected)
    np.testing.assert_allclose(r.bootstrap_effects, expected, atol=1e-15)
    np.testing.assert_allclose(
        r.pretrend_results.bootstrap_contrasts, expected_contrasts, atol=1e-15
    )
    np.testing.assert_allclose(r.vcov, np.cov(expected.T, ddof=1))
    assert r.se**2 == pytest.approx(np.ones(2) @ r.vcov @ np.ones(2) / 4)
    sd = expected.std(axis=0, ddof=1)
    qindex = int(np.ceil((1 - r.alpha) * b)) - 1
    centered = np.abs(expected - r.effects.att.to_numpy())
    widths = np.sort(centered, axis=0)[qindex]
    np.testing.assert_allclose(r.effects.conf_int_upper - r.effects.att, widths)
    q = np.sort((centered / sd).max(axis=1))[qindex]
    assert r.cband_crit_value == pytest.approx(q)
    np.testing.assert_allclose(r.effects.cband_upper - r.effects.att, q * sd)
    np.testing.assert_allclose(
        r.effects.p_value, (centered >= np.abs(r.effects.att.to_numpy())).mean(axis=0)
    )


def test_algorithm_2_fixed_anchor_independent_of_calibration():
    r = fit_duration(n_bootstrap=13, seed=11)
    d = r.pretrend_results
    sd = d.bootstrap_contrasts.std(axis=0, ddof=1)
    v = d.contrasts.contrast.to_numpy()
    maxima = np.max(abs(d.bootstrap_contrasts - v) / sd, axis=1)
    q = np.sort(maxima)[int(np.ceil((1 - d.alpha) * d.n_bootstrap)) - 1]
    np.testing.assert_allclose(d.contrasts.cband_upper, v + q * sd)
    assert d.statistic == pytest.approx(max(abs(v / sd)))
    assert d.p_value == np.mean(maxima >= d.statistic)
    assert d.reject == bool(
        np.any(d.contrasts.cband_lower > 0) or np.any(d.contrasts.cband_upper < 0)
    )


def test_centered_ties_and_zero_variance_companions():
    theta = np.array([1.0, 2.0, 3.0])
    draws = np.array([[0.0, 2.0, 2.0], [2.0, 2.0, 4.0]])
    r = _inference(theta, draws, 0.5)
    assert r["p_value"][0] == 1  # Equality counted conservatively.
    assert r["se"][1] == 0
    for key in ("t_stat", "p_value", "conf_int_lower", "conf_int_upper"):
        assert np.isnan(r[key][1])
    assert np.isnan(r["cband_crit_value"])
    opposite = _inference(-theta, -draws, 0.5)
    np.testing.assert_array_equal(r["p_value"], opposite["p_value"])


def test_treated_post_extinction_valid_and_required_control_extinction_error():
    data = duration_panel(treated=(90, 75, 60, 45, 0, 0))
    r = fit_duration(data, n_bootstrap=10, seed=3)
    np.testing.assert_allclose(r.effects.att, [0.3, 0.15])
    assert any("treated survivors at 4: 0" in w for w in r.support_warnings)
    data = duration_panel(control=(96, 80, 64, 48, 0, 0))
    with pytest.raises(ValueError, match="control survival"):
        fit_duration(data, n_bootstrap=2)


def test_ph_eligibility_and_diagnostic_failure_are_independent():
    data = duration_panel(control=(100, 100, 80, 60, 40, 20), treated=(100, 100, 70, 50, 20, 10))
    r = fit_duration(data, method="proportional_hazards", n_bootstrap=12, seed=3)
    assert r.fit_periods == [2, 3] and 1 in r.excluded_fit_periods
    assert np.isfinite(r.att)
    assert r.pretrend_results.status == "unavailable"
    assert r.pretrend_results.reject is None
    assert r.pretrend_results.n_bootstrap_attempted == 0
    with pytest.raises(ValueError, match="calibration"):
        DurationDiD(method="proportional_hazards", n_bootstrap=2).fit(
            data, "absorbed", "group", "id", "date", post_periods=[4, 5], fit_periods=[1, 2]
        )
    z = DurationDiD(method="proportional_hazards", n_bootstrap=2, seed=1).fit(
        data,
        "absorbed",
        "group",
        "id",
        "date",
        post_periods=[4, 5],
        fit_periods=[1, 2],
        time_weights={1: 0, 2: 1},
    )
    assert z.fit_periods == [2]


def test_two_pre_dates_unavailable_diagnostic():
    data = duration_panel(control=(100, 80, 60), treated=(100, 80, 40))
    r = DurationDiD(n_bootstrap=10, seed=3).fit(
        data, "absorbed", "group", "id", "date", post_periods=[2]
    )
    assert r.pretrend_results.contrasts.empty
    assert r.pretrend_results.bootstrap_contrasts.shape == (10, 0)
    assert r.pretrend_results.reject is None
    assert r.att == pytest.approx(0.2)


def test_invalid_counterfactual_preserves_raw_and_suppresses_causal_output():
    data = duration_panel(control=(100, 70, 40, 20, 20, 20), treated=(100, 99, 98, 97, 96, 95))
    r = fit_duration(data, n_bootstrap=10, seed=5)
    assert r.estimation_status == "invalid_counterfactual"
    assert np.isnan(r.att) and r.effects.att.isna().all()
    assert r.vcov is None
    assert r.survival_curve.raw_att.iloc[4:].notna().all()
    assert (r.survival_curve.counterfactual_status == "invalid").any()


def test_failed_draws_do_not_filter_or_retry():
    data = duration_panel(control=(3, 2, 1, 1, 1, 1), treated=(3, 2, 1, 1, 1, 1), n=3, scale=1)
    r = fit_duration(data, n_bootstrap=100, seed=2)
    assert 0 < r.n_bootstrap_valid < 100
    assert r.bootstrap_effects.shape == (100, 2)
    assert np.isfinite(r.att) and np.isnan(r.se)
    assert r.vcov is None
    assert r.inference_status["simple"] == "unavailable"
    mask = np.isnan(r.bootstrap_effects).all(axis=1)
    assert mask.sum() == 100 - r.n_bootstrap_valid
    assert r.bootstrap_failures.query("family == 'effects'").draw.nunique() == mask.sum()
    assert r.pretrend_results.status == "unavailable"


def test_diagnostic_bootstrap_failure_preserves_valid_effect_family():
    data = duration_panel(
        control=(100, 99, 70, 50, 30, 10), treated=(100, 99, 70, 50, 20, 5), scale=1
    )
    r = DurationDiD(method="proportional_hazards", n_bootstrap=40, seed=33).fit(
        data, "absorbed", "group", "id", "date", post_periods=[4, 5], fit_periods=[2, 3]
    )
    assert r.fit_periods == [2, 3]
    assert r.n_bootstrap_valid == 40
    assert r.inference_status["simple"] == "available"
    assert r.pretrend_results.n_bootstrap_valid < 40
    assert r.pretrend_results.status == "unavailable"
    assert r.pretrend_results.reject is None
    assert set(r.bootstrap_failures.family) == {"diagnostics"}


def test_complete_survival_zero_se_and_unidentified_ph():
    data = duration_panel(control=(100,) * 6, treated=(100,) * 6)
    r = fit_duration(data, n_bootstrap=9, seed=4)
    assert r.att == 0 and r.se == 0
    assert_nan_inference(
        {name: getattr(r, name) for name in ("se", "t_stat", "p_value", "conf_int")}
    )
    assert r.n_bootstrap_valid == 9
    assert r.pretrend_results.reject is None
    with pytest.raises(ValueError, match="calibration"):
        fit_duration(data, method="proportional_hazards", n_bootstrap=2)


def test_support_warning_thresholds():
    data = duration_panel(control=(6, 5, 4, 3, 2, 1), treated=(6, 5, 4, 3, 2, 1), n=6, scale=1)
    r = fit_duration(data, n_bootstrap=2, seed=1)
    assert not any("at 0:" in text or "at 1:" in text for text in r.support_warnings)
    assert any("at 2: 4" in text for text in r.support_warnings)
