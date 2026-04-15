import pytest

from diff_diff import LPDiD


def test_lpdid_get_params_round_trip():
    est = LPDiD(pre_window=4, post_window=6, reweight=True, no_composition=True)
    params = est.get_params()
    assert params["pre_window"] == 4
    assert params["post_window"] == 6
    assert params["reweight"] is True
    assert params["no_composition"] is True


def test_lpdid_set_params_updates_attributes():
    est = LPDiD()
    returned = est.set_params(pre_window=5, control_group="never_treated")
    assert returned is est
    assert est.pre_window == 5
    assert est.control_group == "never_treated"


def test_lpdid_rejects_invalid_control_group():
    with pytest.raises(ValueError, match="control_group"):
        LPDiD(control_group="bad")
