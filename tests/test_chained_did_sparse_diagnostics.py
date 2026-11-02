"""Exact Gaussian benchmark geometry; these checks do not qualify inference."""

import importlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from benchmarks.python.validate_chained_did import generate_sparse_data


def study():
    name = "benchmarks.python.diagnose_chained_did_sparse"
    assert importlib.util.find_spec(name) is not None, "Sparse oracle diagnostic must exist"
    return importlib.import_module(name)


def balanced_dates(dates):
    return pd.DataFrame(
        [
            (i, t, 3 if i < 2 else 0, (i + 1) * t + (i < 2 and t >= 3))
            for i in range(4)
            for t in dates
        ],
        columns=["id", "t", "g", "y"],
    )


def test_two_date_oracle_covariance_is_known_difference_of_means_variance():
    system = study().sparse_system(balanced_dates((2, 3)))
    assert system["pairs"] == [(2, 3)]
    np.testing.assert_allclose(system["omega"], [[2]])
    np.testing.assert_allclose(system["loadings"], [[1], [1]])
    assert np.allclose(system["error_loadings"].reshape(4, 2, 1).sum(axis=1), 0)


def test_oracle_covariance_keeps_shared_endpoint_signs_and_redundant_links():
    system = study().sparse_system(balanced_dates((1, 2, 3)))
    np.testing.assert_allclose(system["omega"], [[2, 1, -1], [1, 2, 1], [-1, 1, 2]])
    assert np.linalg.matrix_rank(system["omega"]) == 2
    np.testing.assert_allclose(system["oracle_variances"], [2, 2])


@pytest.mark.parametrize("n,k,c", [(400, 0, 19), (400, 1, 19), (400, 2, 19), (4000, 50, 190)])
def test_independent_linear_maps_match_native_points_and_hc0_with_sparse_support(n, k, c):
    data, _ = generate_sparse_data(n, k, 123, c)
    system = study().sparse_system(data)
    out = study().evaluate(data, system)
    assert out["native_discrepancy"] < 1e-11
    assert out["native_variance_discrepancy"] < 1e-11
    assert out["estimates"][0] == out["estimates"][1]
    assert out["estimates"][2] == out["estimates"][3]
    assert np.isfinite(out["ses"]).all()
    np.testing.assert_allclose(system["metrics"][1], 2 / np.diag(system["omega"]))
    other = data.assign(y=data.y * 3 + 100)
    np.testing.assert_allclose(study().sparse_system(other)["loadings"], system["loadings"])


def test_unsupported_target_is_not_repaired_by_pseudoinverse():
    data = balanced_dates((1, 2, 3))
    data.loc[data.t == 3, "y"] = np.nan
    with pytest.raises(ValueError, match="unsupported"):
        study().sparse_system(data)


def test_grid_separates_fixed_counts_from_proportional_support():
    grid = study().cases()
    assert len(grid) == 10
    assert {
        (c["n"], c["sparse_treated"], c["sparse_controls"])
        for c in grid
        if c["regime"] == "growing"
    } == {(4000, 20, 190), (4000, 50, 190)}


def test_small_run_preserves_pairing_and_reports_native_replay_and_failures():
    report = study().run_case(study().cases()[2], 3, 123, 2)
    assert report["native_max_abs_error"] < 1e-11
    assert report["native_variance_max_abs_error"] < 1e-11
    assert report["failures"] == {}
    assert len(report["variants"]) == 4
    assert all(v["replications"] == 3 and v["failures"] == 0 for v in report["variants"].values())
    assert report["oracle_variance_ratio_design_to_identity"] < 1


def test_cli_protects_existing_and_historical_evidence(tmp_path, monkeypatch):
    output = tmp_path / "evidence.json"
    output.write_text('{"keep": true}')
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        study().main()
    assert exc.value.code == 2
    assert json.loads(output.read_text()) == {"keep": True}


def test_unsupported_replications_remain_in_unconditional_coverage():
    report = study().run_case(dict(n=2, sparse_treated=0, sparse_controls=0), 3, 123, 0)
    assert sum(report["failures"].values()) == 3
    assert report["oracle_variance_ratio_design_to_identity"] is None
    assert all(v["failures"] == 3 and v["coverage"] == 0 for v in report["variants"].values())


def test_cli_completes_small_study_and_binds_exact_source_hashes(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    output = tmp_path / "diagnostic.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["diagnostic", "--replications", "2", "--workers", "1", "--output", str(output)],
    )
    assert study().main() == 0
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert not report["statistically_qualified"]
    assert len(report["results"]) == 10
    root = Path(study().__file__).resolve().parents[2]
    assert all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in report["source_hashes"].items()
    )


def test_design_weighting_cannot_change_balanced_equal_count_recovery():
    data = balanced_dates((1, 2, 3, 4))
    system = study().sparse_system(data)
    assert np.ptp(system["metrics"][1]) == 0
    np.testing.assert_allclose(system["loadings"][0], system["loadings"][1], atol=1e-14)
    result = study().evaluate(data, system)
    np.testing.assert_allclose(result["estimates"][:2], result["estimates"][2:], atol=1e-14)
    np.testing.assert_allclose(result["ses"][:2], result["ses"][2:], atol=1e-14)


def test_native_replay_cannot_accept_unavailable_target_as_a_nan_comparison():
    data = balanced_dates((1, 2, 3, 4))
    system = study().sparse_system(data)
    changed = data.copy()
    changed.loc[changed.t == 3, "y"] = np.nan
    with pytest.raises(ValueError, match="native estimate or covariance"):
        study().evaluate(changed, system)
