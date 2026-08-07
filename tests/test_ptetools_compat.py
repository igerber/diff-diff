import json
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    ATTGTResult,
    attgt_noif,
    covid_attgt,
    did_attgt,
    gt_data_frame,
    overall_weights,
    pte_aggte,
    setup_pte,
    two_by_two_subset,
)


def test_attgt_noif_matches_r_result_shape():
    extra = {"method": "example"}
    result = attgt_noif(1.25, extra)

    assert isinstance(result, ATTGTResult)
    assert result.attgt == 1.25
    assert result.inf_func is None
    assert result.extra_gt_returns == extra


def test_covid_attgt_reuses_drdid_panel_score_for_levels_and_changes():
    panel = pd.DataFrame(
        {
            "id": np.repeat(np.arange(8), 2),
            "name": np.tile(["pre", "post"], 8),
            "period": np.tile([1, 2], 8),
            "G": np.repeat([2, 2, 2, 2, 0, 0, 0, 0], 2),
            "D": np.repeat([1, 1, 1, 1, 0, 0, 0, 0], 2),
            "Y": [2, 4, 3, 6, 4, 7, 5, 9, 1, 2, 2, 3, 3, 4, 4, 5],
            "x": np.repeat([0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5], 2),
        }
    )
    levels = covid_attgt(gt_data_frame(panel), covariates=["x"])
    changes = covid_attgt(gt_data_frame(panel), covariates=["x"], d_outcome=True)

    assert levels.inf_func is not None
    assert levels.inf_func.shape == (8,)
    assert changes.inf_func is not None
    assert np.isfinite([levels.attgt, changes.attgt]).all()


def test_covid_attgt_matches_r_drdid_when_available():
    if subprocess.run(
        ["Rscript", "-e", "quit(status=!requireNamespace('ptetools', quietly=TRUE))"],
        capture_output=True,
    ).returncode:
        pytest.skip("R ptetools is not installed")

    rng = np.random.default_rng(42)
    n = 80
    x = rng.normal(size=n)
    treated = np.arange(n) < n // 2
    pre_y = 0.5 * x + rng.normal(scale=0.2, size=n)
    post_y = 1.0 + 0.5 * x + 1.5 * treated + rng.normal(scale=0.2, size=n)
    z_pre = rng.normal(size=n)
    z_post = z_pre + rng.normal(scale=0.2, size=n)
    panel = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n), 2),
            "period": np.tile([1, 2], n),
            "G": np.repeat(np.where(treated, 2, 0), 2),
            "Y": np.column_stack([pre_y, post_y]).ravel(),
            "x": np.repeat(x, 2),
            "z": np.column_stack([z_pre, z_post]).ravel(),
            "name": np.tile(["pre", "post"], n),
            "D": np.repeat(treated.astype(int), 2),
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        input_path = f"{tmp}/panel.csv"
        panel.to_csv(input_path, index=False)
        script = (
            "suppressPackageStartupMessages({library(ptetools); library(jsonlite)}); "
            f"d <- read.csv('{input_path}'); "
            "o <- covid_attgt(d, xformla=~x, d_covs_formula=~z); "
            "od <- covid_attgt(d, xformla=~x, d_covs_formula=~z, d_outcome=TRUE); "
            "cat(toJSON(list(level=list(att=o$attgt, inf=o$inf_func), "
            "difference=list(att=od$attgt, inf=od$inf_func)), auto_unbox=TRUE, digits=16))"
        )
        r = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        reference = json.loads(r.stdout)

    result = covid_attgt(gt_data_frame(panel), covariates=["x"], d_covariates=["z"])
    result_diff = covid_attgt(
        gt_data_frame(panel), covariates=["x"], d_covariates=["z"], d_outcome=True
    )
    assert np.isclose(result.attgt, reference["level"]["att"], atol=1e-6)
    assert result.inf_func is not None
    assert result_diff.inf_func is not None
    # R exposes the unnormalized psi; ptetools' Python result contract stores
    # phi = psi / n, matching the aggregation/bootstrap implementation.
    assert np.allclose(
        result.inf_func, np.asarray(reference["level"]["inf"], float).ravel() / n, atol=1e-6
    )
    assert np.isclose(result_diff.attgt, reference["difference"]["att"], atol=1e-6)
    assert np.allclose(
        result_diff.inf_func,
        np.asarray(reference["difference"]["inf"], float).ravel() / n,
        atol=1e-6,
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
