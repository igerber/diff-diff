import numpy as np
import pandas as pd

from diff_diff import ImplicitAIPWResult, implicit_aipw_weights


def test_implicit_aipw_weights_returns_group_time_aggregation():
    rows = []
    for unit, group in enumerate([0, 0, 2, 2, 3, 3]):
        for period in (1, 2, 3):
            treated = group > 0 and period >= group
            rows.append(
                {
                    "id": unit,
                    "period": period,
                    "G": group,
                    "Y": unit + period + 2 * treated,
                    "X": unit / 10,
                }
            )
    result = implicit_aipw_weights(
        pd.DataFrame(rows), yname="Y", tname="period", idname="id", gname="G", covariates=["X"]
    )
    assert isinstance(result, ImplicitAIPWResult)
    assert len(result.aipw_gt) > 0
    assert np.isfinite(result.est)
    assert np.isclose(result.aipw_gt["att_weight"].sum(), 1.0)
