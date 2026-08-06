import numpy as np

from diff_diff import simulate_bad_controls


def test_simulate_bad_controls_returns_reproducible_panel_and_truth():
    first = simulate_bad_controls(n=40, T_max=4, seed=42)
    second = simulate_bad_controls(n=40, T_max=4, seed=42)
    assert first["data"].equals(second["data"])
    assert first["true_att_gt"].equals(second["true_att_gt"])
    assert len(first["data"]) == 160
    assert np.isfinite(first["true_att_overall"])
