import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from diff_diff import DoseResult, ggpte_cont, pte_dose_results


def test_dose_result_container_preserves_att_curve():
    curve = pd.DataFrame({"dose": [1.0, 2.0], "att": [0.5, 1.0]})
    result = pte_dose_results([1.0, 2.0], curve, overall_att=0.75, overall_att_se=0.1)
    assert isinstance(result, DoseResult)
    assert np.isclose(result.overall_att, 0.75)
    assert result.summary().equals(curve)
    assert result.to_dict()["att_d"] is not None


def test_ggpte_cont_plots_att_and_acrt_curves():
    result = DoseResult(
        dose=[0.0, 1.0],
        att_d=pd.DataFrame(
            {"dose": [0.0, 1.0], "att": [0.5, 1.0], "se": [0.1, 0.2], "crit": [2.0, 2.0]}
        ),
        acrt_d=pd.DataFrame(
            {"dose": [0.0, 1.0], "acrt": [0.4, 0.9], "se": [0.1, 0.2], "crit": [2.0, 2.0]}
        ),
    )

    att_ax = ggpte_cont(result, show=False)
    acrt_ax = ggpte_cont(result, type="acrt", show=False)

    assert att_ax.get_ylabel() == "Treatment Effect"
    assert acrt_ax.get_ylabel() == "Treatment Effect"
