import numpy as np
import pandas as pd

from diff_diff import twfe_cov_bal, twfe_cov_bal_gt


def _data():
    rows = []
    for unit, group in enumerate([0, 0, 2, 2, 3, 3]):
        for period in (1, 2, 3):
            rows.append(
                {
                    "id": unit,
                    "period": period,
                    "G": group,
                    "Y": unit + period,
                    "X": unit + 0.2 * period,
                }
            )
    return pd.DataFrame(rows)


def test_twfe_covariate_balance_reports_standardized_differences():
    data = _data()
    local = twfe_cov_bal_gt(
        data, g=2, tp=3, covariates=["X"], tname="period", idname="id", gname="G"
    )
    full = twfe_cov_bal(data, covariates=["X"], tname="period", idname="id", gname="G")
    assert len(local) == 1
    assert len(full) > 0
    assert np.isfinite(local.loc[0, "sd"])
    assert np.isfinite(local.loc[0, "weighted_standardized_diff"])
