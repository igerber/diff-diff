import numpy as np
import pandas as pd

from diff_diff import didbc, extract_att


def _bad_control_panel():
    rows = []
    for unit in range(20):
        treated = unit >= 10
        x_pre = (unit % 5) / 5.0
        for period in (0, 1):
            noise = 0.01 * ((unit * 7) % 5) if period == 1 else 0.0
            x = x_pre + (1.5 if treated else 0.5) * period + noise
            y = 2.0 * x + (3.0 if treated and period == 1 else 0.0)
            rows.append({"id": unit, "period": period, "G": 1 if treated else 0, "Y": y, "X": x})
    return pd.DataFrame(rows)


def test_imputation_recovers_effect_through_a_bad_control():
    result = didbc(
        _bad_control_panel(),
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
    )
    assert np.isclose(result.att, 5.0)
    assert np.isfinite(result.se)
    assert extract_att(result) == {"att": result.att, "se": result.se}


def test_unknown_dr_nuisance_method_fails_closed():
    try:
        didbc(
            _bad_control_panel(),
            yname="Y",
            gname="G",
            tname="period",
            idname="id",
            bad_control="X",
            est_method="dr_ml",
            nuisance_method="unknown",
        )
    except NotImplementedError as exc:
        assert "dr_ml" in str(exc)
    else:
        raise AssertionError("unknown nuisance methods must fail closed")


def test_parametric_dr_returns_finite_att_and_influence_function():
    result = didbc(
        _bad_control_panel(),
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        est_method="dr_ml",
        nuisance_method="parametric",
    )
    assert result.method == "dr_ml-parametric"
    assert np.isfinite(result.att)
    assert np.isfinite(result.se)
    assert np.isclose(result.influence_function.mean(), 0.0)


def test_random_forest_dr_cross_fits_and_returns_finite_result():
    result = didbc(
        _bad_control_panel(),
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        est_method="dr_ml",
        nuisance_method="ml",
        n_folds=2,
    )
    assert result.method == "dr_ml"
    assert np.isfinite(result.att)
    assert np.isfinite(result.se)


def test_dr_small_treated_group_falls_back_to_imputation():
    panel = _bad_control_panel().query("id < 14").copy()
    panel["G"] = (panel["id"] >= 10).astype(int)
    panel.loc[panel["G"].eq(1), "G"] = 1
    result = didbc(
        panel,
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        est_method="dr_ml",
        nuisance_method="parametric",
        min_group_size=5,
    )
    assert result.method == "imputation"
