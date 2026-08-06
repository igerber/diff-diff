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
