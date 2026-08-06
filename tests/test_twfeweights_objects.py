import numpy as np

from diff_diff import GTWeightsResult, gt_weights, two_period_covs_obj


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
