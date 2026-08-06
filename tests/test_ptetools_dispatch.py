import numpy as np
import pandas as pd

from diff_diff import attgt_pte_aggregations, process_att_gt


def test_ptetools_aggregation_dispatch_aliases():
    att_gt = process_att_gt(pd.DataFrame({"group": [2], "time": [2], "attgt": [1.0]}))
    result = attgt_pte_aggregations(att_gt)
    assert np.isclose(result.estimate, 1.0)
