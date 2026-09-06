"""Duration panel, calibration, bootstrap and estimator API contracts."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import DurationDiD


def duration_panel(
    control=(96, 80, 64, 48, 32, 16), treated=(90, 75, 60, 45, 20, 8), n=100, scale=4
):
    """Independent fixture: nested survivor sets define complete absorbing histories."""
    rows = []
    for g, counts in enumerate((control, treated)):
        for i in range(n * scale):
            rows.extend(
                (g * n * scale + i, t, g, int(i >= count * scale)) for t, count in enumerate(counts)
            )
    return pd.DataFrame(rows, columns=["id", "date", "group", "absorbed"])


def fit_duration(data=None, **kwargs):
    return DurationDiD(**kwargs).fit(
        duration_panel() if data is None else data,
        "absorbed",
        "group",
        "id",
        "date",
        post_periods=[4, 5],
    )


@pytest.mark.parametrize("method", ["common_dynamics", "proportional_hazards"])
def test_happy_path(method, ci_params):
    result = fit_duration(method=method, n_bootstrap=ci_params.bootstrap(99), seed=12)
    assert result.estimation_status == "ok"
    assert result.att == pytest.approx((0.30 - 0.20 + 0.15 - 0.08) / 2)
    assert result.n_treated == result.n_control == 400
    assert result.n_units == 800 and result.n_obs == 4800
    assert result.n_bootstrap_valid == result.n_bootstrap
    assert result.inference_status == dict(
        pointwise="available", simultaneous="available", simple="available"
    )
    assert result.pretrend_results.status == "available"
    assert result.pretrend_results.anchor_period == 3
    assert list(result.pretrend_results.contrasts.period) == [1, 2]


@pytest.mark.parametrize("bad", [True, -1, 0, 1, 1.5, None])
def test_bootstrap_validation(bad):
    with pytest.raises(ValueError, match="n_bootstrap"):
        DurationDiD(n_bootstrap=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "other"},
        {"alpha": 0},
        {"alpha": np.nan},
        {"alpha": True},
        {"seed": True},
        {"seed": -1},
        {"seed": 1.5},
    ],
)
def test_constructor_transactions(kwargs):
    est = DurationDiD(n_bootstrap=5, seed=2)
    old = est.get_params()
    with pytest.raises(ValueError):
        est.set_params(**kwargs)
    assert old == est.get_params()


def test_refits_and_shuffle_reproducibility():
    data = duration_panel()
    estimator = DurationDiD(n_bootstrap=15, seed=23)
    r = estimator.fit(data, "absorbed", "group", "id", "date", post_periods=[5, 4])
    assert estimator.is_fitted_ and estimator.results_ is r
    again = estimator.fit(data, "absorbed", "group", "id", "date", post_periods=[4, 5])
    np.testing.assert_array_equal(r.bootstrap_effects, again.bootstrap_effects)
    shuffled = estimator.fit(
        data.sample(frac=1, random_state=1), "absorbed", "group", "id", "date", post_periods=[4, 5]
    )
    assert r.att == pytest.approx(shuffled.att)
    assert r.coefficient == pytest.approx(shuffled.coefficient)
    assert estimator.results_ is shuffled


def test_calibration_selection_zero_weights_and_diagnostic_window():
    data = duration_panel()
    kwargs = dict(
        data=data,
        outcome="absorbed",
        treatment="group",
        unit="id",
        time="date",
        post_periods=[4, 5],
    )
    a = DurationDiD(n_bootstrap=10, seed=1).fit(
        **kwargs, fit_periods=[3, 2, 1], time_weights={1: 0, 2: 2, 3: 2}
    )
    b = DurationDiD(n_bootstrap=10, seed=1).fit(**kwargs, fit_periods=[2, 3])
    assert a.fit_periods == [2, 3]
    assert a.time_weights == {2: 0.5, 3: 0.5}
    assert a.excluded_fit_periods == {1: "zero calibration weight"}
    np.testing.assert_array_equal(a.bootstrap_effects, b.bootstrap_effects)
    np.testing.assert_array_equal(
        a.pretrend_results.bootstrap_contrasts, b.pretrend_results.bootstrap_contrasts
    )
    assert a.pre_periods == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "options",
    [
        {"post_periods": []},
        {"post_periods": [3, 5]},
        {"post_periods": [4, 4, 5]},
        {"post_periods": [10]},
        {"post_periods": [1, 2, 3, 4, 5]},
        {"fit_periods": [0]},
        {"fit_periods": [4]},
        {"fit_periods": [1, 1]},
        {"fit_periods": []},
        {"time_weights": {1: 1}},
        {"time_weights": {1: 0, 2: 0, 3: 0}},
        {"time_weights": {1: -1, 2: 1, 3: 1}},
        {"time_weights": {1: np.inf, 2: 1, 3: 1}},
    ],
)
def test_period_validation(options):
    kwargs = dict(post_periods=[4, 5])
    kwargs.update(options)
    with pytest.raises(ValueError):
        DurationDiD(n_bootstrap=2).fit(
            duration_panel(), "absorbed", "group", "id", "date", **kwargs
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing_cell",
        "missing_value",
        "nonbinary",
        "reversal",
        "group_change",
        "one_group",
    ],
)
def test_invalid_panels(mutation):
    data = duration_panel()
    if mutation == "duplicate":
        data = pd.concat([data, data.iloc[[0]]])
    elif mutation == "missing_cell":
        data = data.iloc[1:]
    elif mutation == "missing_value":
        data.loc[0, "absorbed"] = np.nan
    elif mutation == "nonbinary":
        data.loc[0, "absorbed"] = 2
    elif mutation == "reversal":
        data.loc[0, "absorbed"] = 1
    elif mutation == "group_change":
        data.loc[0, "group"] = 1
    else:
        data = data[data.group == 0]
    with pytest.raises(ValueError):
        fit_duration(data, n_bootstrap=2)


@pytest.mark.parametrize(
    "labels",
    [
        np.arange(6) * 0.1,
        np.arange(6) * 7 + 1920,
        pd.date_range("2020-01-01", periods=6, freq="2D"),
        pd.timedelta_range("0 days", periods=6, freq="3h"),
    ],
)
def test_clock_labels(labels):
    data = duration_panel()
    data["date"] = data.date.map(dict(enumerate(labels)))
    r = DurationDiD(method="proportional_hazards", n_bootstrap=5, seed=1).fit(
        data, "absorbed", "group", "id", "date", post_periods=list(labels[4:])
    )
    assert r.coefficient == pytest.approx(1)
    assert r.att == pytest.approx(0.085)
    assert r.periods == list(labels)


@pytest.mark.parametrize(
    "labels", [[0, 1, 2, 3, 4, 6], list("abcdef"), [0, 0.1, 0.2, 0.3, 0.4, 0.51]]
)
def test_irregular_or_unsupported_clock(labels):
    data = duration_panel()
    data["date"] = data.date.map(dict(enumerate(labels)))
    with pytest.raises(ValueError):
        DurationDiD(n_bootstrap=2).fit(
            data, "absorbed", "group", "id", "date", post_periods=labels[4:]
        )


@pytest.mark.parametrize(
    "keyword", ["survey_design", "weights", "cluster", "covariates", "first_treat"]
)
def test_unsupported_extensions(keyword):
    with pytest.raises(TypeError):
        DurationDiD(n_bootstrap=2).fit(
            duration_panel(),
            "absorbed",
            "group",
            "id",
            "date",
            post_periods=[4, 5],
            **{keyword: None},
        )


def test_extreme_weights_preserve_support_or_raise():
    data = duration_panel()
    est = DurationDiD(n_bootstrap=3, seed=1)
    r = est.fit(
        data,
        "absorbed",
        "group",
        "id",
        "date",
        post_periods=[4, 5],
        time_weights={1: 1e308, 2: 1e308, 3: 1e308},
    )
    assert r.time_weights == {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
    with pytest.raises(ValueError, match="underflows"):
        est.fit(
            data,
            "absorbed",
            "group",
            "id",
            "date",
            post_periods=[4, 5],
            time_weights={1: 1e-300, 2: 1e300, 3: 1e300},
        )
