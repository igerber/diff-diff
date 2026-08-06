import numpy as np

from diff_diff import effective_sample_size, frac_treated_extreme, log_ratio_sd, pooled_sd


def test_twfeweights_helpers_match_basic_r_definitions():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = np.array([1, 1, 0, 0])
    assert np.isclose(effective_sample_size(np.ones(4)), 4.0)
    assert np.isclose(pooled_sd(x, treatment), 0.5)
    assert np.isclose(log_ratio_sd(x, treatment), 0.0)
    assert 0.0 <= frac_treated_extreme(x, treatment) <= 1.0
