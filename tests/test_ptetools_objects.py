import numpy as np
import pandas as pd

from diff_diff import aggte_obj, group_time_att


def test_ptetools_result_object_factories():
    att_gt = group_time_att(pd.DataFrame({"group": [2], "time": [2], "attgt": [1.0]}))
    result = aggte_obj(1.0, att_gt, type="group", standard_error=0.2, conf_int=(0.6, 1.4))
    assert np.isclose(result.estimate, 1.0)
    assert result.to_dict()["type"] == "group"
