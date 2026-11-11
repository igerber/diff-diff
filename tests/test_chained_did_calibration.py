"""Independent checks for validation-only oracle and follow-up experiments."""

import importlib
import importlib.util

import numpy as np
import pytest

from diff_diff import ChainedDiD
from tests.test_methodology_chained_did import binary_covariate_panel


def study():
    name = "benchmarks.python.diagnose_chained_did"
    assert importlib.util.find_spec(name) is not None, "Calibration study must exist"
    return importlib.import_module(name)


def test_known_propensity_link_omits_estimation_uncertainty_and_matches_ratio_influence():
    data, delta, treated, x = binary_covariate_panel()
    odds = np.where(x == 0, 1.5, 2.5)
    effects, scores = study().oracle_cells(data, {1: odds})
    wt = (~treated) * odds
    mt, mc = delta[treated].mean(), wt @ delta / wt.sum()
    expected = treated * (delta - mt) / treated.sum() - wt * (delta - mc) / wt.sum()
    assert effects[1, 1] == pytest.approx(mt - mc)
    np.testing.assert_allclose(scores[1, 1], expected, atol=1e-12)


def test_known_population_weights_change_point_and_remove_share_uncertainty_only():
    memberships = np.array([1, 1, 1, 2, 0, 0])
    cells = {(1, 1): 2.0, (2, 2): 10.0}
    q = {k: np.zeros(6) for k in cells}
    empirical, empirical_q = study().combine(cells, q, memberships)
    known, known_q = study().combine(cells, q, memberships, masses={1: 0.5, 2: 0.5})
    assert empirical == 4
    assert known == 6
    np.testing.assert_allclose(empirical_q, [-0.5, -0.5, -0.5, 1.5, 0, 0])
    np.testing.assert_array_equal(known_q, np.zeros(6))


def test_fitted_contribution_aggregation_matches_production_including_cluster_scaling():
    base = importlib.import_module("benchmarks.python.validate_chained_did")
    data, _ = base.generate_data(300, "intermittent", 42, clusters=30)
    native = ChainedDiD(cluster="cluster").fit(data, "y", "id", "t", "g", covariates=["x"])
    cells = {k: v.effect for k, v in native.group_time_effects.items() if k[1] >= k[0]}
    q = dict(zip(native.vcov_index, native.influence_functions.T))
    registry = data.drop_duplicates("id")
    estimate, contribution = study().combine(cells, q, registry.g.to_numpy())
    assert estimate == pytest.approx(native.att)
    assert study().standard_error(contribution, registry.cluster.to_numpy()) == pytest.approx(
        native.se
    )


def test_standardised_error_diagnostics_keep_failed_replications_and_use_observed_rates():
    report = study().summarize(np.array([-2.0, 0.0, 2.0, np.nan]), np.ones(4), 0)
    assert report["failures"] == 1
    assert report["coverage"] == 0.25
    assert report["z_sd"] == 2
    assert report["abs_z_95"] > 1.96
    assert report["z_quantiles"][1] == 0


def test_scenario_grid_crosses_sample_and_cluster_counts_and_uses_fresh_seeds():
    cases = study().diagnostic_cases()
    clustered = {(c["n"], c["clusters"]) for c in cases if c["clusters"]}
    assert clustered == {(1000, 50), (1000, 100), (4000, 50), (4000, 100)}
    assert study().replication_seed(20261104, 0, 0) == study().replication_seed(20261104, 0, 0)
    seeds = [study().replication_seed(20261104, i, j) for i in range(8) for j in range(50)]
    assert len(set(seeds)) == len(seeds)
    assert max(seeds) > 2**32
