import numpy as np
import pandas as pd

from diff_diff import pte_emp_boot, pte_params, pte_results


def test_ptetools_r_style_constructor_aliases():
    data = pd.DataFrame(
        {"id": [0, 0, 1, 1], "period": [1, 2, 1, 2], "G": [0, 0, 2, 2], "Y": [0.0, 1.0, 0.0, 3.0]}
    )
    params = pte_params(data, "Y", "G", "period", "id")
    att_gt = pd.DataFrame({"group": [2], "time": [2], "attgt": [2.0]})
    result = pte_results(att_gt, 2.0)
    boot = pte_emp_boot(data, yname="Y", gname="G", tname="period", idname="id", biters=3, seed=1)
    assert params.groups == [2]
    assert np.isclose(result.overall_att, 2.0)
    assert np.isfinite(boot.overall_se)
