import numpy as np
import pandas as pd

from diff_diff import att_simple_weights, attO_weights, twfe_weights


def _fixture():
    panel = pd.DataFrame(
        {
            "id": np.repeat(np.arange(6), 3),
            "period": np.tile([1, 2, 3], 6),
            "G": np.repeat([0, 0, 2, 2, 3, 3], 3),
        }
    )
    rows = [
        (0, 1, 0.0),
        (0, 2, 0.0),
        (0, 3, 0.0),
        (2, 1, 0.0),
        (2, 2, 1.0),
        (2, 3, 2.0),
        (3, 1, 0.0),
        (3, 2, 0.0),
        (3, 3, 3.0),
    ]
    effects = pd.DataFrame(rows, columns=["group", "time", "attgt"])
    return panel, effects


def test_twfeweights_preserves_r_output_columns_and_normalization():
    panel, effects = _fixture()
    result = twfe_weights(effects, panel, treatment_group="G", keep_untreated=True)
    frame = result.to_dataframe()

    assert list(frame.columns) == ["group", "time.period", "weight", "attgt", "post"]
    assert np.isclose(frame.loc[frame.post, "weight"].sum(), 1.0)
    assert np.isclose(frame.loc[~frame.post, "weight"].sum(), -1.0)
    assert np.isclose(result.att, 2.0)


def test_att_weights_are_nonnegative_and_sum_to_one():
    panel, effects = _fixture()
    for fn in (attO_weights, att_simple_weights):
        frame = fn(effects, panel, treatment_group="G", keep_untreated=True).to_dataframe()
        assert np.isclose(frame.weight.sum(), 1.0)
        assert np.all(frame.loc[frame.post, "weight"] >= 0)
        assert np.isclose(frame.loc[~frame.post, "weight"].sum(), 0.0)
