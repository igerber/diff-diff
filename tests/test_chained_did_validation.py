"""Small checks for the independent qualification harness, not coverage evidence."""

import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest


def harness():
    name = "benchmarks.python.validate_chained_did"
    assert importlib.util.find_spec(name) is not None, "Qualification harness must exist"
    return importlib.import_module(name)


def test_dgp_keeps_original_registry_and_has_reproducible_analytic_targets():
    data, truth = harness().generate_data(200, "rotating", 42)
    again, second_truth = harness().generate_data(200, "rotating", 42)
    pd.testing.assert_frame_equal(data, again)
    assert truth == second_truth
    assert data.id.nunique() == 200
    assert data.groupby("id").y.count().eq(2).all()
    assert 1 < truth["overall"] < 4
    assert 2 < truth["g2_t3"] < 3
    assert 3 < truth["g3_t4"] < 4
    _, null_truth = harness().generate_data(200, "rotating", 42, null=True)
    assert set(null_truth.values()) == {0}


def test_summary_counts_failures_in_coverage_and_reports_monte_carlo_uncertainty():
    result = harness().summarize(np.array([0.0, 0.0, np.nan]), np.array([1.0, 1.0, np.nan]), 0)
    assert result["replications"] == 3
    assert result["failures"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["coverage_given_valid"] == 1
    assert result["coverage_mcse"] == pytest.approx(np.sqrt((2 / 3) * (1 / 3) / 3))
    assert result["rejection_mcse"] == 0
    assert result["nominal_binomial_mcse"] == pytest.approx(np.sqrt(0.95 * 0.05 / 3))


def test_unknown_observation_design_is_rejected():
    with pytest.raises(ValueError):
        harness().generate_data(200, "unknown", 42)


@pytest.mark.parametrize("sparse_treated", [0, 1, 2, 5, 20, 50])
def test_sparse_design_preserves_counts_and_analytic_target(sparse_treated):
    data, truth = harness().generate_sparse_data(400, sparse_treated, 123)
    other, _ = harness().generate_sparse_data(400, sparse_treated, 123)
    pd.testing.assert_frame_equal(data, other)
    assert truth == {"g3_t3": 1.0}
    assert data.id.nunique() == 400
    at_one = data.loc[data.t == 1].groupby("g").id.nunique()
    assert at_one.get(3, 0) == sparse_treated
    assert at_one[0] == 19
    assert data.loc[data.t == 2].id.nunique() == 400
    assert data.loc[data.t == 3].id.nunique() == 400


def test_sparse_runner_records_pair_exclusions_and_unconditional_inference_metrics():
    case = dict(n=100, sparse_treated=1, sparse_controls=19, violation="none")
    record = harness()._run_case(case, 3, 123, 0)
    assert record["failures"] == {}
    assert record["targets"]["g3_t3"]["replications"] == 3
    assert record["targets"]["g3_t3"]["truth"] == 1
    assert record["pair_diagnostics"]["replications_with_singleton_exclusions"] == 3
    assert record["pair_diagnostics"]["replications_with_low_support"] == 0


@pytest.mark.parametrize(
    "kwargs", [{"n": 3}, {"sparse_treated": -1}, {"sparse_treated": 51}, {"sparse_controls": 51}]
)
def test_sparse_design_rejects_invalid_arm_counts(kwargs):
    params = dict(n=100, sparse_treated=1, sparse_controls=19, seed=123)
    params.update(kwargs)
    with pytest.raises(ValueError, match="counts"):
        harness().generate_sparse_data(**params)


def test_sparse_cli_records_completed_manifest_and_refuses_overwrite(tmp_path, monkeypatch):
    import json
    import shutil
    import sys
    from pathlib import Path

    repo = Path(harness().__file__).resolve().parents[2]
    for name in (
        "diff_diff/chained_did.py",
        "diff_diff/chained_did_results.py",
        "benchmarks/python/validate_chained_did.py",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "sparse.json"
    args = [
        "validate_chained_did",
        "--design",
        "sparse",
        "--replications",
        "2",
        "--workers",
        "1",
        "--output",
        str(destination),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert harness().main() == 0
    record = json.loads(destination.read_text())
    assert record["status"] == "complete" and record["scope"] == "pilot"
    assert record["design"] == "sparse"
    assert record["source_commit"] is None
    assert record["source_uncommitted"] is None
    assert len(record["results"]) == 6
    before = destination.read_bytes()
    with pytest.raises(SystemExit) as error:
        harness().main()
    assert error.value.code == 2
    assert destination.read_bytes() == before


def test_qualification_cli_protects_frozen_evidence(monkeypatch):
    import sys
    from pathlib import Path

    frozen = Path(harness().__file__).resolve().parents[1] / "data/chained_did/qualification.json"
    before = frozen.read_bytes()
    monkeypatch.setattr(sys, "argv", ["validate_chained_did", "--output", str(frozen)])
    with pytest.raises(SystemExit) as error:
        harness().main()
    assert error.value.code == 2
    assert frozen.read_bytes() == before
