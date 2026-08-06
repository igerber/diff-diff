import numpy as np

from diff_diff import didbc, simulate_bad_controls


def test_staggered_imputation_returns_group_time_cells():
    simulated = simulate_bad_controls(n=120, T_max=4, seed=7)
    result = didbc(
        simulated["data"],
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
    )
    assert result.method == "imputation-staggered"
    assert {"group", "time", "attgt", "se"}.issubset(result.att_gt.columns)
    assert len(result.att_gt) > 0
    assert np.isfinite(result.att)


def test_staggered_parametric_dr_returns_group_time_cells():
    simulated = simulate_bad_controls(n=120, T_max=4, seed=7)
    result = didbc(
        simulated["data"],
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        est_method="dr_ml",
        nuisance_method="parametric",
    )
    assert result.method == "dr_ml-parametric-staggered"
    assert len(result.att_gt) > 0
    assert np.isfinite(result.att)
