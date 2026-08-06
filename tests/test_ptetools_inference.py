import numpy as np

from diff_diff import crit_val_checks


def test_crit_val_checks_falls_back_to_pointwise_normal_value():
    value, cband = crit_val_checks(float("nan"))
    assert not cband
    assert np.isclose(value, 1.959963984540054)


def test_crit_val_checks_preserves_valid_simultaneous_value():
    value, cband = crit_val_checks(2.5)
    assert cband
    assert value == 2.5
