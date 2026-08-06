import numpy as np
import pandas as pd

from diff_diff import (
    did_attgt,
    overall_weights,
    pte_aggte,
    setup_pte,
    two_by_two_subset,
)


def _panel():
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(4), 3),
            "period": np.tile([1, 2, 3], 4),
            "G": np.repeat([0, 0, 2, 3], 3),
            "Y": [0, 1, 2, 0, 0, 1, 0, 2, 4, 0, 0, 3],
        }
    )


def test_setup_and_two_by_two_subset_match_ptetools_contract():
    panel = _panel()
    params = setup_pte(panel, "Y", "G", "period", "id")
    assert params.groups == [2, 3]
    subset = two_by_two_subset(panel, 2, 2, gname="G", tname="period", idname="id")
    result = did_attgt(subset.gt_data)
    assert subset.n1 == 1
    assert np.isclose(result.attgt, 5.0 / 3.0)
    assert result.inf_func is not None
    assert np.isclose(result.inf_func.mean(), 0.0)


def test_ptetools_aggregation_weights_and_att():
    effects = pd.DataFrame(
        {
            "group": [0, 0, 0, 2, 2, 3, 3],
            "time": [1, 2, 3, 2, 3, 2, 3],
            "attgt": [0, 0, 0, 1, 2, 3, 4],
        }
    )
    weights = overall_weights(effects)
    assert np.isclose(weights["overall_weight"].sum(), 1.0)
    result = pte_aggte(effects, type="group")
    assert np.isclose(result.estimate, 2.75)


def test_dynamic_aggregation_normalizes_cohort_weights_by_event_time():
    effects = pd.DataFrame(
        {
            "group": [2, 2, 3, 3],
            "time": [2, 3, 3, 4],
            "attgt": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = pte_aggte(effects, type="dynamic", cohort_weights={2: 0.75, 3: 0.25})
    assert np.isclose(
        result.weights.groupby(result.weights["time"] - result.weights["group"])["overall_weight"]
        .sum()
        .min(),
        1.0,
    )
