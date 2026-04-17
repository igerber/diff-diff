import pytest
import pandas as pd

from diff_diff import LPDiD, LPDiDResults


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
