import numpy as np

from diff_diff import PostLassoResult, did_post_lasso


def test_post_lasso_returns_finite_att_and_selection_surfaces():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 4))
    treatment = np.r_[np.zeros(40), np.ones(40)]
    y0 = rng.normal(size=80)
    y1 = y0 + 1.5 * treatment + 0.5 * x[:, 0]
    result = did_post_lasso(y1, y0, treatment, x, random_state=3)
    assert isinstance(result, PostLassoResult)
    assert np.isfinite(result.att)
    assert np.isfinite(result.se)
    assert result.influence_function.shape == (80,)
