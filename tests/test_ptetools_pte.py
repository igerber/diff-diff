import numpy as np

from diff_diff import pte


def _panel():
    import pandas as pd

    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(4), 3),
            "period": np.tile([1, 2, 3], 4),
            "G": np.repeat([0, 0, 2, 3], 3),
            "Y": [0, 1, 2, 0, 0, 1, 0, 2, 4, 0, 0, 3],
        }
    )


def test_pte_runs_group_time_loop_and_returns_results():
    result = pte(_panel(), yname="Y", gname="G", tname="period", idname="id")
    assert set(result.att_gt.columns) == {"group", "time", "attgt", "se"}
    assert len(result.att_gt) == 4
    assert np.isfinite(result.overall_att)
    assert result.to_dataframe().equals(result.att_gt)


def test_pte_accepts_pre_period_covariates():
    panel = _panel()
    panel["Z"] = np.repeat([0.0, 1.0, 0.5, 1.5], 3)
    result = pte(
        panel,
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        covariates=["Z"],
    )
    assert np.isfinite(result.att_gt["attgt"].dropna()).all()


def test_pte_empirical_bootstrap_is_seed_reproducible():
    kwargs = {
        "yname": "Y",
        "gname": "G",
        "tname": "period",
        "idname": "id",
        "bstrap": True,
        "biters": 9,
        "seed": 42,
    }
    first = pte(_panel(), **kwargs)
    second = pte(_panel(), **kwargs)
    assert np.isfinite(first.overall_se)
    assert np.isclose(first.overall_se, second.overall_se)
    assert first.bootstrap_distribution is not None
    assert len(first.bootstrap_distribution) == 9
    assert "overall_att" in first.to_dict()
    assert first.overall_conf_int[0] <= first.overall_conf_int[1]
    assert "PTEResults" in first.summary()
