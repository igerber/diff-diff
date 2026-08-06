import numpy as np

from diff_diff import did_post_lasso_ra


def test_post_lasso_ra_returns_finite_regression_adjustment():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 3))
    d = np.r_[np.zeros(40), np.ones(40)]
    y0 = rng.normal(size=80)
    y1 = y0 + 1.25 * d + x[:, 0]
    result = did_post_lasso_ra(y1, y0, d, x, random_state=4)
    assert np.isfinite(result.att)
    assert np.isfinite(result.se)
    assert result.selected_vars_propensity.size == 0
