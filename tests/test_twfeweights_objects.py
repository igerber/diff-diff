import numpy as np
import pandas as pd

from diff_diff import (
    GTWeightsResult,
    ggtwfeweights,
    gt_weights,
    mp_weights_obj,
    two_period_covs_obj,
)


def test_twfeweights_object_factories_preserve_numeric_payloads():
    local = gt_weights(
        g=2,
        tp=3,
        treated=[1.0],
        comparison=[0.0],
        weights_treated=[1.0],
        weights_comparison=[1.0],
        weighted_outcome_diff=2.0,
        alpha_weight=0.5,
        ess=1.0,
    )
    two_period = two_period_covs_obj(1.0, [1.0, -1.0], [2.0, 1.0], [1, 0], ess=2.0)
    assert isinstance(local, GTWeightsResult)
    assert np.isclose(local.weighted_outcome_diff, 2.0)
    assert np.isclose(two_period.ess, 2.0)


def test_mp_weights_obj_matches_r_post_flag_contract():
    result = mp_weights_obj(
        pd.DataFrame(
            {
                "group": [0, 2, 2],
                "time.period": [1, 2, 1],
                "weight": [0.0, 0.5, 0.5],
                "attgt": [0.0, 1.0, 0.0],
            }
        )
    )
    assert result.weights_df["post"].tolist() == [False, True, False]
    assert ggtwfeweights(result).get_xlabel() == "weight"
