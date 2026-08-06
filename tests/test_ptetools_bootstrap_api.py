import numpy as np
import pandas as pd

from diff_diff import mboot2, panel_empirical_bootstrap


def _panel():
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(4), 3),
            "period": np.tile([1, 2, 3], 4),
            "G": np.repeat([0, 0, 2, 3], 3),
            "Y": [0, 1, 2, 0, 0, 1, 0, 2, 4, 0, 0, 3],
        }
    )


def test_mboot2_is_seed_reproducible():
    influence = np.arange(12, dtype=float).reshape(4, 3)
    first = mboot2(influence, biters=5, seed=4)
    second = mboot2(influence, biters=5, seed=4)
    assert first.shape == (5, 3)
    assert np.array_equal(first, second)


def test_panel_empirical_bootstrap_returns_pte_results():
    result = panel_empirical_bootstrap(
        _panel(), yname="Y", gname="G", tname="period", idname="id", biters=5, seed=4
    )
    assert np.isfinite(result.overall_se)
