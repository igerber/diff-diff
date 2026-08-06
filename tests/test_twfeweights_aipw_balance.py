import numpy as np
import pandas as pd

from diff_diff import aipw_cov_bal, aipw_cov_bal_gt


def _data():
    rows = []
    for unit, group in enumerate([0, 0, 2, 2]):
        for period in (1, 2):
            rows.append(
                {
                    "id": unit,
                    "period": period,
                    "G": group,
                    "Y": unit + period,
                    "X": unit + 0.1 * period,
                }
            )
    return pd.DataFrame(rows)


def test_aipw_balance_reports_weighted_differences():
    data = _data()
    local = aipw_cov_bal_gt(
        data, covariates=["X"], yname="Y", tname="period", idname="id", gname="G"
    )
    full = aipw_cov_bal(data, covariates=["X"], yname="Y", tname="period", idname="id", gname="G")
    assert len(local) == 1
    assert len(full) == 1
    assert np.isfinite(local.loc[0, "weighted_standardized_diff"])
