import numpy as np
import pandas as pd

from diff_diff import implicit_twfe_weights


def test_implicit_twfe_weights_returns_group_time_decomposition():
    rows = []
    for unit, group in enumerate([0, 0, 2, 2, 3, 3]):
        for period in (1, 2, 3):
            treatment = group > 0 and period >= group
            rows.append(
                {"id": unit, "period": period, "G": group, "Y": unit + period + 2 * treatment}
            )
    result = implicit_twfe_weights(
        pd.DataFrame(rows), yname="Y", tname="period", idname="id", gname="G"
    )
    assert {"group", "time", "alpha_weight", "attgt"}.issubset(result.twfe_gt.columns)
    assert np.isfinite(result.est)
    assert np.isfinite(result.pre_trends_bias)
