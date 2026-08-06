from diff_diff import pte_attgt, pte_default, setup_pte_basic, two_by_two_subset


def test_ptetools_wrappers_share_core_behavior():
    import pandas as pd

    data = pd.DataFrame(
        {
            "id": [0, 0, 1, 1],
            "period": [1, 2, 1, 2],
            "G": [0, 0, 2, 2],
            "Y": [0.0, 1.0, 0.0, 3.0],
        }
    )
    params = setup_pte_basic(data, "Y", "G", "period", "id")
    result = pte_default(data, yname="Y", gname="G", tname="period", idname="id")
    assert params.groups == [2]
    subset = two_by_two_subset(data, 2, 2)
    assert pte_attgt(subset.gt_data).attgt == result.att_gt.iloc[0].attgt
