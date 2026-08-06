import numpy as np
import pandas as pd

from diff_diff import DoseResult, pte_dose_results


def test_dose_result_container_preserves_att_curve():
    curve = pd.DataFrame({"dose": [1.0, 2.0], "att": [0.5, 1.0]})
    result = pte_dose_results([1.0, 2.0], curve, overall_att=0.75, overall_att_se=0.1)
    assert isinstance(result, DoseResult)
    assert np.isclose(result.overall_att, 0.75)
    assert result.summary().equals(curve)
    assert result.to_dict()["att_d"] is not None
