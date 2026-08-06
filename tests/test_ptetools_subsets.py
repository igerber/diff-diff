from diff_diff import keep_all_pretreatment_subset, keep_all_untreated_subset


def _data():
    return {
        "id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "period": [1, 2, 3] * 3,
        "G": [0, 0, 0, 2, 2, 2, 3, 3, 3],
        "Y": list(range(9)),
    }


def test_keep_all_untreated_subset_drops_cohort_post_history():
    import pandas as pd

    result = keep_all_untreated_subset(pd.DataFrame(_data()), 2, 2)
    assert set(result.gt_data.data["id"]) == {0, 1, 2}
    assert result.gt_data.data.loc[result.gt_data.data["id"] == 1, "period"].max() == 2


def test_keep_all_pretreatment_subset_keeps_not_yet_treated_units():
    import pandas as pd

    result = keep_all_pretreatment_subset(pd.DataFrame(_data()), 2, 2)
    assert set(result.gt_data.data["id"]) == {0, 1, 2}
    assert result.gt_data.data["period"].max() == 2
