"""Schema and mismatch checks; these synthetic records do not prove R parity."""

import copy
import importlib
import importlib.util

import numpy as np
import pytest

from diff_diff import ChainedDiD
from tests.test_chained_did import rotating_panel


def comparator():
    name = "benchmarks.python.compare_chained_did"
    assert importlib.util.find_spec(name) is not None, "Reference comparator must exist"
    return importlib.import_module(name)


def reference(result):
    event = result.aggregate("event_study")
    return {
        "metadata": {
            "n": result.n_units,
            "covariance_scale": "estimator",
            "cluster_correction": "none",
        },
        "cells": [
            {"group": g, "time": t, "att": cell.effect, "available": cell.skip_reason is None}
            for (g, t), cell in result.group_time_effects.items()
        ],
        "vcov": result.vcov.tolist(),
        "overall": {"att": result.att, "se": result.se},
        "events": {
            "event_time": event.event_time.tolist(),
            "att": event.att.tolist(),
            "se": np.where(event.is_reference, 0, event.se).tolist(),
            "is_reference": event.is_reference.tolist(),
            "vcov": event.vcov.tolist(),
        },
    }


def test_reference_comparison_matches_by_keys_and_rejects_estimate_and_scale_mismatches():
    fit = ChainedDiD().fit(rotating_panel(), "y", "id", "t", "g")
    record = reference(fit)
    order = np.arange(len(record["cells"]))[::-1]
    record["cells"] = [record["cells"][j] for j in order]
    record["vcov"] = np.asarray(record["vcov"])[np.ix_(order, order)].tolist()
    assert comparator().compare(fit, record, outcome_sd=1)["passed"]
    bad = copy.deepcopy(record)
    bad["cells"][0]["att"] += 0.1
    assert not comparator().compare(fit, bad, outcome_sd=1)["passed"]
    bad = copy.deepcopy(record)
    bad["metadata"]["covariance_scale"] = "sqrt_n"
    with pytest.raises(ValueError, match="scale"):
        comparator().compare(fit, bad, outcome_sd=1)


def test_declared_cluster_cr1_reconciliation_is_applied_once_to_all_inference():
    data = rotating_panel()
    data["cluster"] = np.arange(len(data)) // 4
    fit = ChainedDiD(cluster="cluster").fit(data, "y", "id", "t", "g")
    record = reference(fit)
    k = data.cluster.nunique()
    correction = k / (k - 1)
    record["vcov"] = (fit.vcov / correction).tolist()
    record["overall"]["se"] /= np.sqrt(correction)
    record["events"]["vcov"] = (np.asarray(record["events"]["vcov"]) / correction).tolist()
    record["events"]["se"] = (np.asarray(record["events"]["se"]) / np.sqrt(correction)).tolist()
    assert comparator().compare(fit, record, outcome_sd=1, clusters=k)["passed"]
    assert not comparator().compare(fit, record, outcome_sd=1)["passed"]


def test_unavailable_cells_and_registry_masses_cannot_be_silently_discarded():
    fit = ChainedDiD().fit(rotating_panel(windows=((0, 1), (0, 3))), "y", "id", "t", "g")
    record = reference(fit)
    record["cells"][2]["available"] = True
    assert not comparator().compare(fit, record, outcome_sd=1)["passed"]
    record = reference(fit)
    record["metadata"]["n"] += 1
    assert not comparator().compare(fit, record, outcome_sd=1)["passed"]


def test_reference_row_reporting_is_reconciled_but_estimated_event_inference_is_checked():
    native = ChainedDiD().fit(rotating_panel(), "y", "id", "t", "g")
    record = reference(native)
    event = native.aggregate("event_study")
    record["events"] = {
        "event_time": event.event_time.tolist(),
        "att": event.att.tolist(),
        "se": np.where(event.is_reference, 0, event.se).tolist(),
        "is_reference": event.is_reference.tolist(),
        "vcov": event.vcov.tolist(),
    }
    assert comparator().compare(native, record, outcome_sd=1)["passed"]
    record["events"]["se"][-1] *= 1.1
    assert not comparator().compare(native, record, outcome_sd=1)["passed"]


def test_calibrated_score_diagnostic_reproduces_external_continuous_pair_points():
    data = comparator().fixtures()["incomplete_continuous_x"]
    points = comparator().calibrated_points(data)
    # Pinned public R output for this seeded fixture; no R implementation code.
    assert points[2, 2] == pytest.approx(3.0406236096388795, abs=1e-8)
    assert points[2, 3] == pytest.approx(2.8900418764849487, abs=1e-8)
    native = ChainedDiD().fit(data, "y", "id", "t", "g", covariates=["x"])
    assert abs(native.group_time_effects[2, 2].effect - points[2, 2]) > 1e-4


def test_missing_aggregate_evidence_cannot_pass_the_reference_gate():
    native = ChainedDiD().fit(rotating_panel(), "y", "id", "t", "g")
    record = reference(native)
    del record["events"]
    assert not comparator().compare(native, record, outcome_sd=1)["passed"]


def test_reference_environment_rejects_changed_missing_and_extra_packages():
    expected = {
        "R": "R 4.5.2",
        "platform": "linux",
        "packages": [
            {"Package": "cdid", "Version": "1.1.0.0", "Built": "4.5.2", "_row": "cdid"},
            {"Package": "dependency", "Version": "1", "Built": "4.5.2"},
        ],
    }
    actual = copy.deepcopy(expected)
    actual["packages"].reverse()
    comparator().validate_environment(expected, actual)
    for field in ("Version", "Built"):
        changed = copy.deepcopy(actual)
        changed["packages"][0][field] = "changed"
        with pytest.raises(ValueError, match="inventory"):
            comparator().validate_environment(expected, changed)
    for packages in (actual["packages"][:1], actual["packages"] + [actual["packages"][0]]):
        with pytest.raises(ValueError, match="inventory"):
            comparator().validate_environment(expected, dict(actual, packages=packages))
    with pytest.raises(ValueError, match="inventory"):
        comparator().validate_environment(expected, dict(actual, R="changed"))


@pytest.mark.parametrize("name", sorted(comparator().fixtures()))
def test_frozen_public_r_outputs_detect_native_numerical_drift(name):
    import hashlib
    import json
    from pathlib import Path

    tool = comparator()
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads(
        (root / "benchmarks/data/chained_did/reference-validation.json").read_text()
    )
    case = frozen["cases"][name]
    data = tool.fixtures()[name]
    assert (
        hashlib.sha256(data.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
        == case["input_sha256"]
    )
    native = ChainedDiD(cluster="cluster" if "cluster" in data else None).fit(
        data, "y", "id", "t", "g", covariates=["x"] if "x" in data else None
    )
    clusters = data.drop_duplicates("id").cluster.nunique() if "cluster" in data else None
    outcome = tool.compare(
        native, case["reference"], outcome_sd=float(data.y.std()), clusters=clusters
    )
    # Pin every check, including the known criterion mismatch, without relaxing tolerances.
    assert outcome["checks"] == case["checks"]
    assert outcome["passed"] == (name != "incomplete_continuous_x")
