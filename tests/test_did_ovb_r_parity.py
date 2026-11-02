import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import DIDOVBSensitivity


@pytest.mark.skipif(
    __import__("shutil").which("Rscript") is None,
    reason="Rscript is required for the DiD OVB parity fixture",
)
def test_python_matches_independent_r_application_fixture():
    root = Path(__file__).parents[1]
    fixture = json.loads((root / "benchmarks/data/did_ovb_r_results.json").read_text())
    data = pd.read_csv(root / "benchmarks/data/real/mpdta.csv")
    data = data[data["year"].isin([2006, 2007]) & data["first.treat"].isin([0, 2007])].copy()
    data["treated"] = (data["first.treat"] == 2007).astype(int)
    data = data.sort_values(["countyreal", "year"], kind="stable").reset_index(drop=True)
    fold_ids = np.arange(fixture["n_obs"], dtype=int) % fixture["settings"]["n_folds"]

    result = DIDOVBSensitivity(n_folds=2, seed=0).fit(
        data,
        outcome="lemp",
        treatment="treated",
        time="year",
        unit="countyreal",
        covariates=["lpop"],
        fold_ids=fold_ids,
    )
    expected = fixture

    for name in ("short_att", "short_se", "sigma2_control", "nu2_selection", "scale"):
        assert getattr(result, name) == pytest.approx(expected[name], rel=5e-5, abs=5e-7)

    bounds = result.bounds(trend_r2=1.0, selection_r2=0.5)
    for name in ("lower", "upper", "radius", "lower_se", "upper_se", "lower_ci", "upper_ci"):
        assert getattr(bounds, name) == pytest.approx(expected["bounds"][name], rel=5e-5, abs=5e-7)

    robustness = result.robustness_value(
        null_value=expected["settings"]["null_value"], alpha=expected["settings"]["alpha"]
    )
    assert robustness.rv == pytest.approx(expected["robustness"]["rv"], rel=5e-5, abs=5e-7)
    assert robustness.xrv == pytest.approx(expected["robustness"]["xrv"], rel=5e-5, abs=5e-7)
