import numpy as np
import pandas as pd

from diff_diff import two_period_reg_weights


def test_two_period_reg_weights_matches_fwl_identity():
    data = pd.DataFrame(
        {
            "id": [0, 0, 1, 1, 2, 2, 3, 3],
            "period": [1, 2] * 4,
            "G": [0, 0, 0, 0, 2, 2, 2, 2],
            "Y": [0.0, 1.0, 1.0, 1.5, 0.0, 3.0, 1.0, 4.0],
        }
    )
    result = two_period_reg_weights(data, yname="Y", tname="period", idname="id", gname="G")
    assert np.isclose(result.est, 2.25)
    assert np.isclose(result.weights[result.treatment == 1].mean(), 2.0)
    assert np.isfinite(result.ess)
