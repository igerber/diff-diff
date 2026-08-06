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
