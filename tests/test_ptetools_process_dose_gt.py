"""Tests for ``ptetools.process_dose_gt`` and its B-spline basis helper.

The ``bspline_basis`` helper is pinned against golden values from
``splines2::bSpline`` / ``splines2::dbs``; ``process_dose_gt`` is exercised
end-to-end on a synthetic, self-consistent ``gt_results`` dict whose per-cell
``att.overall`` matches what the generic ``pte`` loop reports.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.ptetools import bspline_basis, process_dose_gt, pte


def _panel() -> pd.DataFrame:
    ids = ["c0", "c1", "t0", "t1", "t2", "t3"]
    panel = []
    for unit, group in zip(ids, [0, 0, 2, 2, 2, 2]):
        for period in (1, 2):
            panel.append({"id": unit, "period": period, "G": group, "Y": float(period)})
    return pd.DataFrame(panel)


def _params(data: pd.DataFrame) -> dict[str, object]:
    return {
        "data": data,
        "yname": "Y",
        "gname": "G",
        "tname": "period",
        "idname": "id",
        "panel": True,
        "control_group": "notyettreated",
        "anticipation": 0,
        "base_period": "varying",
        "dvals": np.array([0.5, 1.5]),
        "degree": 1,
        "knots": np.array([]),
        "biters": 200,
        "alp": 0.05,
        "cband": True,
        "bstrap": True,
    }


def test_bspline_basis_level_matches_splines2_bSpline():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    basis = bspline_basis(x, degree=2, knots=[2.5, 3.5])
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.6222222, 0.2666667, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.0, 0.2666667, 0.6222222, 0.1111111],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(basis, expected, atol=1e-7)


def test_bspline_derivative_matches_splines2_dbs():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    deriv = bspline_basis(x, degree=2, knots=[2.5, 3.5], derivative=1)
    expected = np.array(
        [
            [1.33333333, 0.0, 0.0, 0.0],
            [-0.0888889, 0.5333333, 0.0, 0.0],
            [-0.4, 0.0, 0.4, 0.0],
            [0.0, -0.5333333, 0.0888889, 0.4444444],
            [0.0, 0.0, -1.3333333, 1.3333333],
        ]
    )
    assert np.allclose(deriv, expected, atol=1e-7)


def test_bspline_rejects_bad_knots():
    x = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        bspline_basis(x, degree=2, knots=[0.0])  # on the boundary
    with pytest.raises(ValueError):
        bspline_basis(x, degree=2, knots=[0.5, 0.5])  # not strictly increasing


def _build_gt_results(data: pd.DataFrame) -> tuple[dict, float]:
    res = pte(data, yname="Y", gname="G", tname="period", idname="id", bstrap=False)
    influence = np.nan_to_num(np.asarray(res.influence_functions, dtype=float))
    acrt_inffunc = np.zeros_like(influence)
    acrt_inffunc[2:6, 0] = 1.0
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    bread = np.array([[0.5, 0.0], [0.0, 1.0]])
    inner = {
        "att.d": np.array([0.1, 0.2]),
        "acrt.d": np.array([0.05, 0.06]),
        "att.overall": float(res.overall_att),
        "acrt.overall": 0.5,
        "bet": np.array([0.1, 0.1]),
        "bread": bread,
        "Xe": X,
    }
    gt = {
        "inffunc": acrt_inffunc,
        "attgt_list": [{"group": 2, "time.period": 2, "att": float(res.overall_att)}],
        "extra_gt_returns": [{"group": 2, "time.period": 2, "extra_gt_returns": inner}],
    }
    return gt, float(res.overall_att)


def test_process_dose_gt_matches_r_point_estimates():
    data = _panel()
    gt, overall = _build_gt_results(data)
    result = process_dose_gt(gt, _params(data), seed=7)

    assert np.allclose(result.att_d["att"], [0.1, 0.2])
    assert np.allclose(result.acrt_d["att"], [0.05, 0.06])
    assert np.isclose(result.overall_att, overall)
    assert np.isclose(result.overall_acrt, 0.5)
    assert np.isfinite(result.overall_att_se)
    assert np.isfinite(result.overall_acrt_se)
    assert result.att_d_se.shape == (2,)
    assert result.acrt_d_se.shape == (2,)
    assert isinstance(result.simultaneous, bool)

    for table in (result.att_d, result.acrt_d):
        assert set(table.columns) == {"dose", "att", "se", "crit"}


def test_process_dose_gt_seed_reproducible():
    data = _panel()
    gt, _ = _build_gt_results(data)
    first = process_dose_gt(gt, _params(data), seed=11)
    second = process_dose_gt(gt, _params(data), seed=11)
    assert np.allclose(first.att_d_se, second.att_d_se)
    assert np.isclose(first.overall_att_se, second.overall_att_se)


def test_process_dose_gt_rejects_mismatched_cell_order():
    data = _panel()
    gt, _ = _build_gt_results(data)
    gt = dict(gt)
    gt["extra_gt_returns"] = [
        {
            "group": 3,
            "time.period": 2,
            "extra_gt_returns": gt["extra_gt_returns"][0]["extra_gt_returns"],
        }
    ]
    with pytest.raises(ValueError):
        process_dose_gt(gt, _params(data), seed=1)


def test_process_dose_gt_rejects_missing_cell_fields():
    data = _panel()
    gt, _ = _build_gt_results(data)
    gt = dict(gt)
    inner = dict(gt["extra_gt_returns"][0]["extra_gt_returns"])
    del inner["bread"]
    gt["extra_gt_returns"] = [{"group": 2, "time.period": 2, "extra_gt_returns": inner}]
    with pytest.raises(ValueError):
        process_dose_gt(gt, _params(data), seed=1)
