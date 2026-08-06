import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import did_attgt, didbc, simulate_bad_controls, twfe_weights, two_by_two_subset

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "r_parity_reference.R"


def _panel():
    rows = []
    for unit, group in enumerate([0] * 6 + [2] * 3 + [3] * 3):
        for period in (1, 2, 3):
            outcome = 0.2 * unit + 0.5 * period + (1.5 if group and period >= group else 0.0)
            rows.append({"id": unit, "period": period, "G": group, "Y": outcome})
    return pd.DataFrame(rows)


def _bad_control_panel():
    rows = []
    for unit in range(20):
        treated = unit >= 10
        x_pre = unit / 10.0
        for period in (1, 2):
            period_noise = 0.01 * ((unit * 7) % 5) if period == 2 else 0.0
            x = x_pre + (1.5 if treated else 0.5) * (period - 1) + period_noise
            y = 2.0 * x + (3.0 if treated and period == 2 else 0.0)
            rows.append({"id": unit, "period": period, "G": 2 if treated else 0, "Y": y, "X": x})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def rscript():
    executable = shutil.which("Rscript")
    if executable is None:
        pytest.skip("Rscript is not installed")
    required = subprocess.run(
        [
            executable,
            "-e",
            "quit(status=ifelse(all(vapply(c('did','twfeweights','ptetools','badcontrols'), requireNamespace, logical(1), quietly=TRUE)), 0, 1))",
        ],
        check=False,
    )
    if required.returncode != 0:
        pytest.skip("R parity packages are not installed")
    return executable


def _run_r(rscript, mode, data, tmp_path):
    input_path = tmp_path / f"{mode}-input.csv"
    output_path = tmp_path / f"{mode}-output.csv"
    data.to_csv(input_path, index=False)
    completed = subprocess.run(
        [rscript, str(REFERENCE), mode, str(input_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"R parity command failed for {mode}:\n{completed.stderr}")
    return pd.read_csv(output_path)


def test_twfeweights_matches_r(rscript, tmp_path):
    panel = _panel()
    r_frame = _run_r(rscript, "twfeweights", panel, tmp_path)
    effects = r_frame[["group", "time.period", "attgt"]].rename(columns={"time.period": "time"})
    py_frame = twfe_weights(effects, panel, treatment_group="G", keep_untreated=True).to_dataframe()
    py_frame = py_frame.sort_values(["group", "time.period"]).reset_index(drop=True)
    r_frame = r_frame.sort_values(["group", "time.period"]).reset_index(drop=True)
    np.testing.assert_allclose(py_frame["weight"], r_frame["weight"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(py_frame["attgt"], r_frame["attgt"], rtol=1e-8, atol=1e-8)


def test_ptetools_did_attgt_matches_r(rscript, tmp_path):
    panel = _panel()
    r_att = float(_run_r(rscript, "ptetools", panel, tmp_path).loc[0, "att"])
    subset = two_by_two_subset(panel, 2, 2, gname="G", tname="period", idname="id")
    py_att = did_attgt(subset.gt_data).attgt
    np.testing.assert_allclose(py_att, r_att, rtol=1e-8, atol=1e-8)


def test_badcontrols_imputation_matches_r(rscript, tmp_path):
    panel = _bad_control_panel()
    r_result = _run_r(rscript, "badcontrols", panel, tmp_path).loc[0]
    py_result = didbc(
        panel,
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
    )
    np.testing.assert_allclose(py_result.att, r_result["att"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(py_result.se, r_result["se"], rtol=1e-8, atol=1e-8)


def test_binary_badcontrols_imputation_matches_r(rscript, tmp_path):
    simulated = simulate_bad_controls(
        n=200,
        T_max=2,
        groups=[2],
        binary_bad_control=True,
        seed=42,
    )
    panel = simulated["data"]
    r_result = _run_r(rscript, "badcontrols", panel, tmp_path).loc[0]
    py_result = didbc(
        panel,
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
    )
    np.testing.assert_allclose(py_result.att, r_result["att"], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(py_result.se, r_result["se"], rtol=1e-7, atol=1e-7)
