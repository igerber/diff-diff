import pandas as pd
import pytest

from pathlib import Path

from diff_diff import LPDiD, LPDiDResults


FIXTURE_DIR = Path(__file__).resolve().parent / "data"


def _make_linear_panel(units):
    rows = []
    for spec in units:
        for time in spec["times"]:
            rows.append(
                {
                    "unit": spec["unit"],
                    "time": time,
                    "y": spec["y"][time],
                    "treat": int(time >= spec["treat_start"]),
                    **{k: v for k, v in spec.items() if k not in {"unit", "times", "y", "treat_start"}},
                }
            )
    return pd.DataFrame(rows)


def _event_coef(results, horizon=0):
    return results.event_study.loc[results.event_study["horizon"] == horizon, "coefficient"].iloc[0]


def test_lpdid_get_params_round_trip():
    est = LPDiD(pre_window=4, post_window=6, reweight=True, no_composition=True)
    params = est.get_params()
    assert params["pre_window"] == 4
    assert params["post_window"] == 6
    assert params["reweight"] is True
    assert params["no_composition"] is True


def test_lpdid_set_params_updates_attributes():
    est = LPDiD()
    returned = est.set_params(pre_window=5, control_group="never_treated")
    assert returned is est
    assert est.pre_window == 5
    assert est.control_group == "never_treated"


def test_lpdid_rejects_invalid_control_group():
    with pytest.raises(ValueError, match="control_group"):
        LPDiD(control_group="bad")


def test_lpdid_rejects_invalid_rank_deficient_action():
    with pytest.raises(ValueError, match="rank_deficient_action"):
        LPDiD(rank_deficient_action="bad")


def test_lpdid_set_params_rejects_unknown_key():
    est = LPDiD()
    with pytest.raises(ValueError, match="Unknown parameter"):
        est.set_params(nonexistent_param=1)


def test_lpdid_set_params_rejects_invalid_control_group():
    est = LPDiD()
    with pytest.raises(ValueError, match="control_group"):
        est.set_params(control_group="bad")


def test_lpdid_set_params_rejects_invalid_rank_deficient_action():
    est = LPDiD()
    with pytest.raises(ValueError, match="rank_deficient_action"):
        est.set_params(rank_deficient_action="bad")


def test_lpdid_results_to_dataframe_and_repr():
    df = pd.DataFrame({"effect": [1.0], "se": [0.1]})
    results = LPDiDResults(
        event_study=df,
        pooled=None,
        n_obs=10,
        n_treated_units=4,
        n_control_units=6,
        pre_window=2,
        post_window=0,
        control_group="clean",
        reweight=False,
        no_composition=False,
        pmd=None,
    )

    assert results.to_dataframe(level="event").equals(df)
    with pytest.raises(ValueError, match="not computed"):
        results.to_dataframe(level="pooled")
    with pytest.raises(ValueError, match="level must be"):
        results.to_dataframe(level="bad")

    rep = repr(results)
    assert "LPDiDResults" in rep
    assert "effect" not in rep


def test_lpdid_requires_core_columns():
    df = pd.DataFrame({"y": [1.0], "id": [1], "t": [0]})
    est = LPDiD()
    with pytest.raises(ValueError, match="Missing columns"):
        est.fit(df, outcome="y", unit="id", time="t", treatment="treat")


def test_lpdid_rejects_only_event_and_only_pooled_together():
    df = pd.DataFrame({"y": [1.0], "id": [1], "t": [0], "treat": [0]})
    est = LPDiD()
    with pytest.raises(ValueError, match="only_event"):
        est.fit(
            df,
            outcome="y",
            unit="id",
            time="t",
            treatment="treat",
            only_event=True,
            only_pooled=True,
        )


def test_lpdid_rejects_invalid_pmd_value():
    with pytest.raises(ValueError, match="pmd"):
        LPDiD(pmd="bad")


def test_lpdid_event_dataframe_contains_reference_and_requested_horizons():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "y": [1, 2, 4, 1, 1, 1, 2, 2, 2],
            "treat": [0, 1, 1, 0, 0, 0, 0, 0, 0],
        }
    )
    res = LPDiD(pre_window=2, post_window=1).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )
    assert list(res.event_study["horizon"]) == [-2, -1, 0, 1]


def test_lpdid_pooled_dataframe_has_pre_and_post_rows():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 2, 2, 2, 2],
            "time": [0, 1, 2, 3] * 2,
            "y": [1, 2, 4, 6, 1, 1, 1, 1],
            "treat": [0, 0, 1, 1, 0, 0, 0, 0],
        }
    )
    res = LPDiD(pre_window=2, post_window=1).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat"
    )
    assert set(res.pooled["window"]) == {"pre", "post"}


def test_lpdid_detects_positive_post_effect_on_simple_panel():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "time": [0, 1, 2] * 4,
            "y": [1, 3, 5, 1, 1, 1, 2, 4, 6, 2, 2, 2],
            "treat": [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        }
    )

    res = LPDiD(pre_window=2, post_window=1).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )

    post0 = res.event_study.loc[res.event_study["horizon"] == 0, "coefficient"].iloc[0]
    assert post0 > 0.5


def test_lpdid_event_estimation_controls_for_calendar_time():
    df = pd.DataFrame(
        {
            "unit": [
                "c1",
                "c1",
                "c1",
                "c2",
                "c2",
                "c2",
                "t1",
                "t1",
                "t1",
                "t2a",
                "t2a",
                "t2a",
                "t2b",
                "t2b",
                "t2b",
                "t2c",
                "t2c",
                "t2c",
            ],
            "time": [0, 1, 2] * 6,
            "y": [
                0,
                0,
                100,
                0,
                0,
                100,
                0,
                5,
                105,
                0,
                0,
                105,
                0,
                0,
                105,
                0,
                0,
                105,
            ],
            "treat": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
            ],
        }
    )

    res = LPDiD(pre_window=2, post_window=0).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )

    post0 = res.event_study.loc[res.event_study["horizon"] == 0, "coefficient"].iloc[0]
    assert post0 == pytest.approx(5.0, abs=1e-8)


def test_lpdid_rejects_pooled_horizon_outside_supported_window():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2],
            "time": [0, 1, 2, 0, 1, 2],
            "y": [1, 3, 5, 1, 1, 1],
            "treat": [0, 1, 1, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="outside the supported pre window"):
        LPDiD(pre_window=2, post_window=1).fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            pre_pooled=(-3, -1),
        )


def test_lpdid_rejects_pooled_window_with_unidentified_horizon():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [0, 1, 2] * 3,
            "y": [1, 3, 5, 1, 1, 1, 2, 2, 2],
            "treat": [0, 1, 1, 0, 0, 0, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="unidentified pooled pre horizons"):
        LPDiD(pre_window=2, post_window=1).fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            pre_pooled=(-2, -1),
        )


def test_lpdid_returns_nan_inference_when_cluster_vcov_is_undefined():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "time": [0, 1, 2] * 4,
            "y": [1, 3, 5, 1, 1, 1, 2, 4, 6, 2, 2, 2],
            "treat": [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            "cluster_id": [1] * 12,
        }
    )

    res = LPDiD(pre_window=2, post_window=1, cluster="cluster_id").fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )

    post0 = res.event_study.loc[res.event_study["horizon"] == 0].iloc[0]
    assert post0["coefficient"] > 0.5
    assert pd.isna(post0["se"])
    assert pd.isna(post0["t_stat"])
    assert pd.isna(post0["p_value"])
    assert pd.isna(post0["conf_low"])
    assert pd.isna(post0["conf_high"])


def test_lpdid_rejects_non_numeric_treatment_values():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "time": [0, 1, 0, 1],
            "y": [1, 2, 1, 1],
            "treat": [0, "treated", 0, 0],
        }
    )

    with pytest.raises(ValueError, match="binary numeric"):
        LPDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")


def test_lpdid_rejects_non_binary_treatment_values():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "time": [0, 1, 0, 1],
            "y": [1, 2, 1, 1],
            "treat": [0, 2, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="binary numeric"):
        LPDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")


def test_lpdid_clears_fitted_state_after_failed_refit():
    good = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2],
            "time": [0, 1, 2, 0, 1, 2],
            "y": [1, 3, 5, 1, 1, 1],
            "treat": [0, 1, 1, 0, 0, 0],
        }
    )
    bad = good.copy()
    bad["treat"] = [0, "bad", 1, 0, 0, 0]

    est = LPDiD(pre_window=2, post_window=1)
    est.fit(good, outcome="y", unit="unit", time="time", treatment="treat", only_event=True)

    with pytest.raises(ValueError, match="binary numeric"):
        est.fit(bad, outcome="y", unit="unit", time="time", treatment="treat")

    assert est.is_fitted_ is False
    assert est.results_ is None


def test_lpdid_covariates_and_absorb_remove_confounding_bias():
    df = _make_linear_panel(
        [
            {"unit": "tA1", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: 6, 2: 12}, "x1": 2, "region": "A"},
            {"unit": "tA2", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: 5, 2: 10}, "x1": 1, "region": "A"},
            {"unit": "tB1", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: -1, 2: -2}, "x1": 1, "region": "B"},
            {"unit": "cA1", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 4, 2: 8}, "x1": 0, "region": "A"},
            {"unit": "cB1", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: -1, 2: -2}, "x1": 1, "region": "B"},
            {"unit": "cB2", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: -2, 2: -4}, "x1": 0, "region": "B"},
        ]
    )

    uncontrolled = LPDiD(pre_window=2, post_window=0).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )
    controlled = LPDiD(pre_window=2, post_window=0).fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
        covariates=["x1"],
        absorb=["region"],
        only_event=True,
    )

    assert _event_coef(uncontrolled) > 2.5
    assert abs(_event_coef(controlled)) < 1e-8


def test_lpdid_pmd_max_uses_mean_of_all_available_pre_periods():
    df = _make_linear_panel(
        [
            {"unit": "t1", "times": [0, 1, 2, 3], "treat_start": 3, "y": {0: 0, 1: 2, 2: 4, 3: 6}},
            {"unit": "t2", "times": [0, 1, 2, 3], "treat_start": 3, "y": {0: 0, 1: 2, 2: 4, 3: 6}},
            {"unit": "c1", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 1, 2: 2, 3: 3}},
            {"unit": "c2", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 1, 2: 2, 3: 3}},
        ]
    )

    standard = LPDiD(pre_window=2, post_window=0).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )
    pmd_max = LPDiD(pre_window=2, post_window=0, pmd="max").fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )

    assert _event_coef(standard) == pytest.approx(1.0, abs=1e-8)
    assert _event_coef(pmd_max) == pytest.approx(2.0, abs=1e-8)


def test_lpdid_pmd_integer_uses_last_k_pre_periods():
    df = _make_linear_panel(
        [
            {"unit": "t1", "times": [0, 1, 2, 3], "treat_start": 3, "y": {0: 0, 1: 2, 2: 4, 3: 6}},
            {"unit": "t2", "times": [0, 1, 2, 3], "treat_start": 3, "y": {0: 0, 1: 2, 2: 4, 3: 6}},
            {"unit": "c1", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 1, 2: 2, 3: 3}},
            {"unit": "c2", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 1, 2: 2, 3: 3}},
        ]
    )

    pmd_two = LPDiD(pre_window=2, post_window=0, pmd=2).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )

    assert _event_coef(pmd_two) == pytest.approx(1.5, abs=1e-8)


def test_lpdid_ylags_remove_lagged_outcome_bias():
    df = _make_linear_panel(
        [
            {"unit": "t1", "times": [0, 1, 2, 3], "treat_start": 2, "y": {0: 8, 1: 4, 2: 2, 3: 1}},
            {"unit": "t2", "times": [0, 1, 2, 3], "treat_start": 2, "y": {0: 6, 1: 3, 2: 1.5, 3: 0.75}},
            {"unit": "c1", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 4, 1: 2, 2: 1, 3: 0.5}},
            {"unit": "c2", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 2, 1: 1, 2: 0.5, 3: 0.25}},
        ]
    )

    uncontrolled = LPDiD(pre_window=2, post_window=0).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )
    with_ylag = LPDiD(pre_window=2, post_window=0).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", ylags=1, only_event=True
    )

    assert _event_coef(uncontrolled) < -0.5
    assert abs(_event_coef(with_ylag)) < 1e-8


def test_lpdid_dylags_remove_lagged_difference_bias():
    df = _make_linear_panel(
        [
            {"unit": "t1", "times": [0, 1, 2, 3], "treat_start": 3, "y": {0: 0, 1: 0, 2: 4, 3: 8}},
            {"unit": "t2", "times": [0, 1, 2, 3], "treat_start": 3, "y": {0: 0, 1: 0, 2: 3, 3: 6}},
            {"unit": "c1", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 0, 2: 2, 3: 4}},
            {"unit": "c2", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 0, 2: 1, 3: 2}},
        ]
    )

    uncontrolled = LPDiD(pre_window=2, post_window=0).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )
    with_dylag = LPDiD(pre_window=2, post_window=0).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", dylags=1, only_event=True
    )

    assert _event_coef(uncontrolled) > 1.5
    assert abs(_event_coef(with_dylag)) < 1e-8


def test_lpdid_no_composition_drops_controls_that_fail_later_clean_control_checks():
    df = _make_linear_panel(
        [
            {"unit": "t1", "times": [0, 1, 2, 3], "treat_start": 1, "y": {0: 0, 1: 1, 2: 2, 3: 3}},
            {"unit": "t2", "times": [0, 1, 2, 3], "treat_start": 2, "y": {0: 0, 1: 0, 2: 2, 3: 4}},
            {"unit": "c1", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 0, 2: 0, 3: 0}},
        ]
    )

    unrestricted = LPDiD(pre_window=2, post_window=2).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )
    common_sample = LPDiD(pre_window=2, post_window=2, no_composition=True).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )

    unrestricted_h0 = unrestricted.event_study.loc[unrestricted.event_study["horizon"] == 0, "n_obs"].iloc[0]
    common_h0 = common_sample.event_study.loc[common_sample.event_study["horizon"] == 0, "n_obs"].iloc[0]

    assert common_h0 < unrestricted_h0


def test_lpdid_reweight_changes_event_target_when_treatment_timing_mix_differs():
    df = _make_linear_panel(
        [
            {"unit": "early1", "times": [0, 1, 2, 3], "treat_start": 1, "y": {0: 0, 1: 10, 2: 20, 3: 30}},
            {"unit": "early2", "times": [0, 1, 2, 3], "treat_start": 1, "y": {0: 0, 1: 10, 2: 20, 3: 30}},
            {"unit": "late1", "times": [0, 1, 2, 3], "treat_start": 2, "y": {0: 0, 1: 0, 2: 2, 3: 4}},
            {"unit": "c1", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 0, 2: 0, 3: 0}},
            {"unit": "c2", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 0, 2: 0, 3: 0}},
        ]
    )

    variance_weighted = LPDiD(pre_window=2, post_window=0).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )
    equally_weighted = LPDiD(pre_window=2, post_window=0, reweight=True).fit(
        df, outcome="y", unit="unit", time="time", treatment="treat", only_event=True
    )

    assert _event_coef(variance_weighted) != pytest.approx(_event_coef(equally_weighted), abs=1e-8)


def test_lpdid_reweight_with_covariates_uses_regression_adjustment_att():
    df = _make_linear_panel(
        [
            {"unit": "u0", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 4, 2: 8}, "x": 2},
            {"unit": "u1", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 0, 2: 0}, "x": 1},
            {"unit": "u2", "times": [0, 1, 2], "treat_start": 2, "y": {0: 0, 1: 2, 2: 5}, "x": 1},
            {"unit": "u3", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 2, 2: 4}, "x": 2},
            {"unit": "u4", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: 0, 2: 0}, "x": 2},
            {"unit": "u5", "times": [0, 1, 2], "treat_start": 2, "y": {0: 0, 1: 1, 2: 2}, "x": 1},
            {"unit": "u6", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 2, 2: 4}, "x": 2},
            {"unit": "u7", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: -3, 2: -6}, "x": 0},
        ]
    )

    res = LPDiD(pre_window=2, post_window=0, reweight=True).fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
        covariates=["x"],
        only_event=True,
    )

    assert _event_coef(res) == pytest.approx(-5 / 12, abs=1e-8)
    post = res.event_study.loc[res.event_study["horizon"] == 0].iloc[0]
    assert post["se"] == pytest.approx(0.9343482687107618, abs=1e-8)


def test_lpdid_reweight_with_absorb_matches_stata_ra_path():
    df = _make_linear_panel(
        [
            {"unit": "tA1", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: 3, 2: 6}, "region": "A"},
            {"unit": "tA2", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: 2, 2: 4}, "region": "A"},
            {"unit": "cA1", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 2, 2: 4}, "region": "A"},
            {"unit": "cA2", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 1, 2: 2}, "region": "A"},
            {"unit": "tB1", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: -2, 2: -4}, "region": "B"},
            {"unit": "cB1", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: -1, 2: -2}, "region": "B"},
            {"unit": "cB2", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: -2, 2: -4}, "region": "B"},
        ]
    )

    res = LPDiD(pre_window=2, post_window=1, reweight=True).fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
        absorb=["region"],
        only_event=True,
    )

    tau0 = res.event_study.loc[res.event_study["horizon"] == 0].iloc[0]
    tau1 = res.event_study.loc[res.event_study["horizon"] == 1].iloc[0]
    assert tau0["coefficient"] == pytest.approx(0.5, abs=1e-8)
    assert tau0["se"] == pytest.approx(0.5400617248673215, abs=1e-8)
    assert tau1["coefficient"] == pytest.approx(1.0, abs=1e-8)
    assert tau1["se"] == pytest.approx(1.080123449734643, abs=1e-8)


def test_lpdid_reweight_with_ylags_matches_stata_ra_path():
    df = _make_linear_panel(
        [
            {"unit": "t1", "times": [0, 1, 2, 3], "treat_start": 2, "y": {0: 8, 1: 4, 2: 2, 3: 1}},
            {"unit": "t2", "times": [0, 1, 2, 3], "treat_start": 2, "y": {0: 6, 1: 3, 2: 1.5, 3: 0.75}},
            {"unit": "c1", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 4, 1: 2, 2: 1, 3: 0.5}},
            {"unit": "c2", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 2, 1: 1, 2: 0.5, 3: 0.25}},
            {"unit": "c3", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 10, 1: 5, 2: 2.5, 3: 1.25}},
        ]
    )

    res = LPDiD(pre_window=2, post_window=0, reweight=True).fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
        ylags=1,
        only_event=True,
    )

    pre2 = res.event_study.loc[res.event_study["horizon"] == -2].iloc[0]
    tau0 = res.event_study.loc[res.event_study["horizon"] == 0].iloc[0]
    assert pre2["coefficient"] == pytest.approx(0.0, abs=1e-12)
    assert pre2["se"] == pytest.approx(0.0, abs=1e-12)
    assert tau0["coefficient"] == pytest.approx(0.0, abs=1e-12)
    assert tau0["se"] == pytest.approx(0.0, abs=1e-12)


def test_lpdid_reweight_with_dylags_matches_stata_ra_path():
    df = _make_linear_panel(
        [
            {"unit": "t1", "times": [0, 1, 2, 3], "treat_start": 3, "y": {0: 0, 1: 0, 2: 4, 3: 8}},
            {"unit": "t2", "times": [0, 1, 2, 3], "treat_start": 3, "y": {0: 0, 1: 0, 2: 3, 3: 6}},
            {"unit": "c1", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 0, 2: 2, 3: 4}},
            {"unit": "c2", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 0, 2: 1, 3: 2}},
            {"unit": "c3", "times": [0, 1, 2, 3], "treat_start": 99, "y": {0: 0, 1: 0, 2: 5, 3: 10}},
        ]
    )

    res = LPDiD(pre_window=2, post_window=0, reweight=True).fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
        dylags=1,
        only_event=True,
    )

    pre2 = res.event_study.loc[res.event_study["horizon"] == -2].iloc[0]
    tau0 = res.event_study.loc[res.event_study["horizon"] == 0].iloc[0]
    assert pre2["coefficient"] == pytest.approx(0.0, abs=1e-12)
    assert pre2["se"] == pytest.approx(0.0, abs=1e-12)
    assert tau0["coefficient"] == pytest.approx(0.0, abs=1e-12)
    assert tau0["se"] == pytest.approx(0.0, abs=1e-12)


def test_lpdid_reweight_with_covariates_matches_stata_ra_pooled_path():
    df = _make_linear_panel(
        [
            {"unit": "u0", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 4, 2: 8}, "x": 2},
            {"unit": "u1", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 0, 2: 0}, "x": 1},
            {"unit": "u2", "times": [0, 1, 2], "treat_start": 2, "y": {0: 0, 1: 2, 2: 5}, "x": 1},
            {"unit": "u3", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 2, 2: 4}, "x": 2},
            {"unit": "u4", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: 0, 2: 0}, "x": 2},
            {"unit": "u5", "times": [0, 1, 2], "treat_start": 2, "y": {0: 0, 1: 1, 2: 2}, "x": 1},
            {"unit": "u6", "times": [0, 1, 2], "treat_start": 99, "y": {0: 0, 1: 2, 2: 4}, "x": 2},
            {"unit": "u7", "times": [0, 1, 2], "treat_start": 1, "y": {0: 0, 1: -3, 2: -6}, "x": 0},
        ]
    )

    res = LPDiD(pre_window=2, post_window=0, reweight=True).fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
        covariates=["x"],
    )

    pre = res.pooled.loc[res.pooled["window"] == "pre"].iloc[0]
    post = res.pooled.loc[res.pooled["window"] == "post"].iloc[0]
    assert pre["coefficient"] == pytest.approx(-1.5, abs=1e-8)
    assert pre["se"] == pytest.approx(0.3535533905932737, abs=1e-8)
    assert post["coefficient"] == pytest.approx(-5 / 12, abs=1e-8)
    assert post["se"] == pytest.approx(0.9343482687107618, abs=1e-8)


def test_lpdid_matches_stata_official_absorbing_example_event_study():
    data = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_core.csv")
    expected = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_event_stata.csv")

    res = LPDiD(pre_window=5, post_window=10).fit(
        data,
        outcome="Y",
        unit="unit",
        time="time",
        treatment="treat",
        only_event=True,
    )

    actual = res.event_study[
        ["horizon", "coefficient", "se", "conf_low", "conf_high", "n_obs"]
    ].copy()
    actual = actual.rename(
        columns={"conf_low": "ci_low", "conf_high": "ci_high", "n_obs": "obs"}
    ).reset_index(drop=True)
    expected = expected[
        ["horizon", "coefficient", "se", "ci_low", "ci_high", "obs"]
    ].copy()

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        atol=1e-6,
        rtol=1e-6,
    )


def test_lpdid_matches_stata_official_absorbing_example_pooled_results():
    data = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_core.csv")
    expected = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_pooled_stata.csv")

    res = LPDiD(pre_window=5, post_window=10).fit(
        data,
        outcome="Y",
        unit="unit",
        time="time",
        treatment="treat",
    )

    actual = res.pooled[
        ["window", "coefficient", "se", "conf_low", "conf_high", "n_obs"]
    ].copy()
    actual = actual.rename(
        columns={"conf_low": "ci_low", "conf_high": "ci_high", "n_obs": "obs"}
    ).reset_index(drop=True)
    expected = expected[
        ["window", "coefficient", "se", "ci_low", "ci_high", "obs"]
    ].copy()

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        atol=1e-6,
        rtol=1e-6,
    )


def test_lpdid_matches_stata_official_absorbing_example_no_composition_results():
    data = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_core.csv")
    expected_event = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_nocomp_event_stata.csv")
    expected_pooled = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_nocomp_pooled_stata.csv")

    res = LPDiD(pre_window=5, post_window=10, no_composition=True).fit(
        data,
        outcome="Y",
        unit="unit",
        time="time",
        treatment="treat",
    )

    actual_event = res.event_study[
        ["horizon", "coefficient", "se", "conf_low", "conf_high", "n_obs"]
    ].copy()
    actual_event = actual_event.rename(
        columns={"conf_low": "ci_low", "conf_high": "ci_high", "n_obs": "obs"}
    ).reset_index(drop=True)
    expected_event = expected_event[
        ["horizon", "coefficient", "se", "ci_low", "ci_high", "obs"]
    ].copy()

    actual_pooled = res.pooled[
        ["window", "coefficient", "se", "conf_low", "conf_high", "n_obs"]
    ].copy()
    actual_pooled = actual_pooled.rename(
        columns={"conf_low": "ci_low", "conf_high": "ci_high", "n_obs": "obs"}
    ).reset_index(drop=True)
    expected_pooled = expected_pooled[
        ["window", "coefficient", "se", "ci_low", "ci_high", "obs"]
    ].copy()

    pd.testing.assert_frame_equal(
        actual_event,
        expected_event,
        check_exact=False,
        atol=2e-4,
        rtol=2e-4,
    )
    pd.testing.assert_frame_equal(
        actual_pooled,
        expected_pooled,
        check_exact=False,
        atol=2e-4,
        rtol=2e-4,
    )


def test_lpdid_matches_stata_official_absorbing_example_reweighted_results():
    data = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_core.csv")
    expected_event = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_rw_event_stata.csv")
    expected_pooled = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_rw_pooled_stata.csv")

    res = LPDiD(pre_window=5, post_window=10, reweight=True).fit(
        data,
        outcome="Y",
        unit="unit",
        time="time",
        treatment="treat",
    )

    actual_event = res.event_study[
        ["horizon", "coefficient", "se", "conf_low", "conf_high", "n_obs"]
    ].copy()
    actual_event = actual_event.rename(
        columns={"conf_low": "ci_low", "conf_high": "ci_high", "n_obs": "obs"}
    ).reset_index(drop=True)
    expected_event = expected_event[
        ["horizon", "coefficient", "se", "ci_low", "ci_high", "obs"]
    ].copy()

    actual_pooled = res.pooled[
        ["window", "coefficient", "se", "conf_low", "conf_high", "n_obs"]
    ].copy()
    actual_pooled = actual_pooled.rename(
        columns={"conf_low": "ci_low", "conf_high": "ci_high", "n_obs": "obs"}
    ).reset_index(drop=True)
    expected_pooled = expected_pooled[
        ["window", "coefficient", "se", "ci_low", "ci_high", "obs"]
    ].copy()

    pd.testing.assert_frame_equal(
        actual_event,
        expected_event,
        check_exact=False,
        atol=5e-5,
        rtol=5e-5,
    )
    pd.testing.assert_frame_equal(
        actual_pooled,
        expected_pooled,
        check_exact=False,
        atol=5e-5,
        rtol=5e-5,
    )


def test_lpdid_matches_stata_official_absorbing_example_reweighted_no_composition_results():
    data = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_core.csv")
    expected_event = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_rw_nocomp_event_stata.csv")
    expected_pooled = pd.read_csv(FIXTURE_DIR / "lpdidtestdata1_rw_nocomp_pooled_stata.csv")

    res = LPDiD(pre_window=5, post_window=10, reweight=True, no_composition=True).fit(
        data,
        outcome="Y",
        unit="unit",
        time="time",
        treatment="treat",
    )

    actual_event = res.event_study[
        ["horizon", "coefficient", "se", "conf_low", "conf_high", "n_obs"]
    ].copy()
    actual_event = actual_event.rename(
        columns={"conf_low": "ci_low", "conf_high": "ci_high", "n_obs": "obs"}
    ).reset_index(drop=True)
    expected_event = expected_event[
        ["horizon", "coefficient", "se", "ci_low", "ci_high", "obs"]
    ].copy()

    actual_pooled = res.pooled[
        ["window", "coefficient", "se", "conf_low", "conf_high", "n_obs"]
    ].copy()
    actual_pooled = actual_pooled.rename(
        columns={"conf_low": "ci_low", "conf_high": "ci_high", "n_obs": "obs"}
    ).reset_index(drop=True)
    expected_pooled = expected_pooled[
        ["window", "coefficient", "se", "ci_low", "ci_high", "obs"]
    ].copy()

    pd.testing.assert_frame_equal(
        actual_event,
        expected_event,
        check_exact=False,
        atol=5e-5,
        rtol=5e-5,
    )
    pd.testing.assert_frame_equal(
        actual_pooled,
        expected_pooled,
        check_exact=False,
        atol=5e-5,
        rtol=5e-5,
    )
