import numpy as np
import pandas as pd

from diff_diff import combine_twfe_weights_gt, implicit_twfe_weights_gt


def test_group_time_twfe_weight_surface_matches_parent_decomposition():
    rows = []
    for unit, group in enumerate([0, 0, 2, 2, 3, 3]):
        for period in (1, 2, 3):
            treated = group > 0 and period >= group
            rows.append(
                {"id": unit, "period": period, "G": group, "Y": unit + period + 2 * treated}
            )
    data = pd.DataFrame(rows)
    local = implicit_twfe_weights_gt(
        data, g=2, tp=2, yname="Y", tname="period", idname="id", gname="G"
    )
    assert np.isfinite(local.weighted_outcome_diff)
    assert np.isclose(
        local.alpha_weight,
        combine_twfe_weights_gt(data, g=2, tp=2, yname="Y", tname="period", idname="id", gname="G"),
    )
