import numpy as np
import pandas as pd

from diff_diff import mp_covariate_bal_summary_helper


def test_balance_summary_aggregates_group_time_rows():
    balance = pd.DataFrame(
        {
            "group": [2, 2, 3],
            "time": [2, 3, 3],
            "covariate": ["X", "X", "X"],
            "unweighted_diff": [1.0, 2.0, 3.0],
            "weighted_diff": [0.5, 1.0, 1.5],
            "sd": [2.0, 2.0, 2.0],
        }
    )
    result = mp_covariate_bal_summary_helper(balance)
    assert len(result) == 1
    assert np.isclose(result.loc[0, "weighted_standardized_diff"], 0.5)
