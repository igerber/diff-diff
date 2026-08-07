import numpy as np
import pandas as pd

from diff_diff import did_rcs_attgt, two_by_two_rcs_subset


def test_rcs_subset_and_attgt_use_period_specific_cross_sections():
    data = pd.DataFrame(
        {
            "period": [1, 1, 2, 2],
            "G": [0, 2, 0, 2],
            "Y": [0.0, 1.0, 1.0, 4.0],
        }
    )
    subset = two_by_two_rcs_subset(data, 2, 2)
    result = did_rcs_attgt(subset.gt_data)
    assert subset.n1 == 2
    assert np.isclose(result.attgt, 2.0)
    assert len(result.inf_func) == 4


def test_rcs_covariate_adjustment_uses_drdid_rc_core():
    rows = []
    for period in (1, 2):
        for i in range(40):
            treated = i < 20
            x = float(i % 10) / 10
            y = 0.5 * x + period + (1.0 if treated and period == 2 else 0.0)
            rows.append({"period": period, "G": 2 if treated else 0, "Y": y, "x": x})
    subset = two_by_two_rcs_subset(pd.DataFrame(rows), 2, 2, covariates=["x"])
    result = did_rcs_attgt(subset.gt_data, covariates=["x"])

    assert np.isfinite(result.attgt)
    assert result.inf_func is not None
    assert result.inf_func.shape == (80,)
