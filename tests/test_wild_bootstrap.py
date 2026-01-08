"""
Tests for Wild Cluster Bootstrap functionality.

Tests the wild_bootstrap_se() function and its integration with DiD estimators.
"""


import numpy as np
import pandas as pd
import pytest

from diff_diff import DifferenceInDifferences, TwoWayFixedEffects
from diff_diff.utils import (
    WildBootstrapResults,
    _generate_mammen_weights,
    _generate_rademacher_weights,
    _generate_webb_weights,
    wild_bootstrap_se,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clustered_did_data():
    """Create DiD data with cluster structure (10 clusters)."""
    np.random.seed(42)

    n_clusters = 10
    obs_per_cluster = 20

    data = []
    for cluster in range(n_clusters):
        # Treatment at cluster level
        is_treated = cluster < 5

        # Cluster-specific effect
        cluster_effect = np.random.normal(0, 2)

        for obs in range(obs_per_cluster):
            for period in [0, 1]:
                y = 10.0
                y += cluster_effect  # Cluster effect
                if period == 1:
                    y += 5.0  # Time effect
                if is_treated and period == 1:
                    y += 3.0  # True ATT = 3.0
                y += np.random.normal(0, 1)  # Idiosyncratic error

                data.append({
                    "cluster": cluster,
                    "unit": cluster * obs_per_cluster + obs,
                    "period": period,
                    "treated": int(is_treated),
                    "post": period,
                    "outcome": y,
                })

    return pd.DataFrame(data)


@pytest.fixture
def few_cluster_data():
    """Create DiD data with very few clusters (4 clusters)."""
    np.random.seed(42)

    n_clusters = 4
    obs_per_cluster = 50

    data = []
    for cluster in range(n_clusters):
        is_treated = cluster < 2
        cluster_effect = np.random.normal(0, 3)

        for obs in range(obs_per_cluster):
            for period in [0, 1]:
                y = 10.0
                y += cluster_effect
                if period == 1:
                    y += 5.0
                if is_treated and period == 1:
                    y += 4.0  # True ATT = 4.0
                y += np.random.normal(0, 1)

                data.append({
                    "cluster": cluster,
                    "unit": cluster * obs_per_cluster + obs,
                    "period": period,
                    "treated": int(is_treated),
                    "post": period,
                    "outcome": y,
                })

    return pd.DataFrame(data)


@pytest.fixture
def ols_components(clustered_did_data):
    """Extract OLS components needed for wild_bootstrap_se."""
    data = clustered_did_data

    y = data["outcome"].values.astype(float)
    d = data["treated"].values.astype(float)
    t = data["post"].values.astype(float)
    dt = d * t

    X = np.column_stack([np.ones(len(y)), d, t, dt])

    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ coefficients
    cluster_ids = data["cluster"].values

    return X, y, residuals, cluster_ids


# =============================================================================
# Weight Generation Tests
# =============================================================================


class TestWeightGeneration:
    """Tests for bootstrap weight generation functions."""

    def test_rademacher_weights_values(self):
        """Test that Rademacher weights are +/-1."""
        rng = np.random.default_rng(42)
        weights = _generate_rademacher_weights(1000, rng)

        unique_values = set(weights)
        assert unique_values == {-1.0, 1.0}

    def test_rademacher_weights_distribution(self):
        """Test Rademacher weights are approximately 50/50."""
        rng = np.random.default_rng(42)
        weights = _generate_rademacher_weights(10000, rng)

        prop_positive = np.mean(weights > 0)
        assert abs(prop_positive - 0.5) < 0.02  # Within 2%

    def test_webb_weights_values(self):
        """Test Webb weights have correct values."""
        rng = np.random.default_rng(42)
        weights = _generate_webb_weights(10000, rng)

        expected_values = np.array([
            -np.sqrt(3/2), -np.sqrt(2/2), -np.sqrt(1/2),
            np.sqrt(1/2), np.sqrt(2/2), np.sqrt(3/2)
        ])

        # Check all observed values are in expected set
        for w in weights:
            assert any(np.isclose(w, ev) for ev in expected_values)

    def test_webb_weights_mean_near_zero(self):
        """Test Webb weights have approximately zero mean."""
        rng = np.random.default_rng(42)
        weights = _generate_webb_weights(50000, rng)

        assert abs(np.mean(weights)) < 0.02

    def test_mammen_weights_values(self):
        """Test Mammen weights have correct values."""
        rng = np.random.default_rng(42)
        weights = _generate_mammen_weights(10000, rng)

        sqrt5 = np.sqrt(5)
        val1 = -(sqrt5 - 1) / 2
        val2 = (sqrt5 + 1) / 2

        # Check all observed values are one of the two Mammen values
        for w in weights:
            assert np.isclose(w, val1) or np.isclose(w, val2)

    def test_mammen_weights_moments(self):
        """Test Mammen weights have E[v]=0, E[v^2]=1, E[v^3]=1."""
        rng = np.random.default_rng(42)
        weights = _generate_mammen_weights(100000, rng)

        # E[v] ≈ 0
        assert abs(np.mean(weights)) < 0.02
        # E[v^2] ≈ 1
        assert abs(np.mean(weights**2) - 1.0) < 0.02
        # E[v^3] ≈ 1
        assert abs(np.mean(weights**3) - 1.0) < 0.05


# =============================================================================
# Wild Bootstrap SE Function Tests
# =============================================================================


class TestWildBootstrapSE:
    """Tests for wild_bootstrap_se function."""

    def test_returns_wild_bootstrap_results(self, ols_components):
        """Test that function returns WildBootstrapResults."""
        X, y, residuals, cluster_ids = ols_components

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        assert isinstance(results, WildBootstrapResults)

    def test_se_is_positive(self, ols_components):
        """Test bootstrap SE is positive."""
        X, y, residuals, cluster_ids = ols_components

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        assert results.se > 0

    def test_p_value_in_valid_range(self, ols_components):
        """Test p-value is in [0, 1]."""
        X, y, residuals, cluster_ids = ols_components

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        assert 0 <= results.p_value <= 1

    def test_ci_contains_reasonable_values(self, ols_components):
        """Test CI bounds are ordered correctly."""
        X, y, residuals, cluster_ids = ols_components

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=199,
            seed=42
        )

        assert results.ci_lower < results.ci_upper

    def test_reproducibility_with_seed(self, ols_components):
        """Test same seed gives same results."""
        X, y, residuals, cluster_ids = ols_components

        results1 = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        results2 = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        assert results1.se == results2.se
        assert results1.p_value == results2.p_value
        assert results1.ci_lower == results2.ci_lower

    def test_different_seeds_different_results(self, ols_components):
        """Test different seeds give different results."""
        X, y, residuals, cluster_ids = ols_components

        results1 = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        results2 = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=123
        )

        # Should be different (not exactly equal)
        assert results1.se != results2.se

    def test_different_weight_types(self, ols_components):
        """Test all weight types produce valid results."""
        X, y, residuals, cluster_ids = ols_components

        for weight_type in ["rademacher", "webb", "mammen"]:
            results = wild_bootstrap_se(
                X, y, residuals, cluster_ids,
                coefficient_index=3,
                n_bootstrap=99,
                weight_type=weight_type,
                seed=42
            )

            assert results.se > 0
            assert 0 <= results.p_value <= 1
            assert results.weight_type == weight_type

    def test_invalid_weight_type_raises(self, ols_components):
        """Test invalid weight type raises ValueError."""
        X, y, residuals, cluster_ids = ols_components

        with pytest.raises(ValueError, match="weight_type must be one of"):
            wild_bootstrap_se(
                X, y, residuals, cluster_ids,
                coefficient_index=3,
                weight_type="invalid"
            )

    def test_few_clusters_warning(self, few_cluster_data):
        """Test warning when clusters < 5."""
        data = few_cluster_data

        y = data["outcome"].values.astype(float)
        d = data["treated"].values.astype(float)
        t = data["post"].values.astype(float)
        dt = d * t
        X = np.column_stack([np.ones(len(y)), d, t, dt])

        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coefficients
        cluster_ids = data["cluster"].values

        with pytest.warns(UserWarning, match="Only 4 clusters detected"):
            wild_bootstrap_se(
                X, y, residuals, cluster_ids,
                coefficient_index=3,
                n_bootstrap=99,
                seed=42
            )

    def test_too_few_clusters_raises(self, ols_components):
        """Test error when clusters < 2."""
        X, y, residuals, _ = ols_components

        # Create single cluster
        single_cluster = np.zeros(len(y))

        with pytest.raises(ValueError, match="at least 2 clusters"):
            wild_bootstrap_se(
                X, y, residuals, single_cluster,
                coefficient_index=3
            )

    def test_n_clusters_reported_correctly(self, ols_components):
        """Test n_clusters is reported correctly."""
        X, y, residuals, cluster_ids = ols_components

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        assert results.n_clusters == 10

    def test_n_bootstrap_reported_correctly(self, ols_components):
        """Test n_bootstrap is reported correctly."""
        X, y, residuals, cluster_ids = ols_components

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=199,
            seed=42
        )

        assert results.n_bootstrap == 199


# =============================================================================
# Integration with Estimators
# =============================================================================


class TestEstimatorIntegration:
    """Tests for wild bootstrap integration with DiD estimators."""

    def test_did_with_wild_bootstrap(self, clustered_did_data):
        """Test DifferenceInDifferences with wild bootstrap."""
        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            seed=42
        )

        results = did.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        assert results.inference_method == "wild_bootstrap"
        assert results.n_bootstrap == 99
        assert results.n_clusters == 10
        assert results.se > 0

    def test_did_wild_bootstrap_reproducibility(self, clustered_did_data):
        """Test wild bootstrap results are reproducible with seed."""
        did1 = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            seed=42
        )

        did2 = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            seed=42
        )

        results1 = did1.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        results2 = did2.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        assert results1.se == results2.se
        assert results1.p_value == results2.p_value

    def test_did_analytical_vs_bootstrap_att_same(self, clustered_did_data):
        """Test that ATT is the same regardless of inference method."""
        did_analytical = DifferenceInDifferences(cluster="cluster")
        did_bootstrap = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            seed=42
        )

        results_analytical = did_analytical.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        results_bootstrap = did_bootstrap.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        # ATT should be identical
        assert results_analytical.att == results_bootstrap.att

    def test_did_wild_bootstrap_with_webb_weights(self, clustered_did_data):
        """Test wild bootstrap with Webb weights."""
        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            bootstrap_weights="webb",
            seed=42
        )

        results = did.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        assert results.inference_method == "wild_bootstrap"
        assert results.se > 0

    def test_did_wild_bootstrap_requires_cluster(self, clustered_did_data):
        """Test that wild bootstrap is only used when cluster is specified."""
        did = DifferenceInDifferences(
            inference="wild_bootstrap",  # No cluster specified
            n_bootstrap=99,
            seed=42
        )

        results = did.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        # Should fall back to analytical since no cluster specified
        assert results.inference_method == "analytical"

    def test_twfe_with_wild_bootstrap(self, clustered_did_data):
        """Test TwoWayFixedEffects with wild bootstrap."""
        twfe = TwoWayFixedEffects(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            seed=42
        )

        results = twfe.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            unit="unit"
        )

        assert results.inference_method == "wild_bootstrap"
        assert results.n_bootstrap == 99
        assert results.se > 0

    def test_summary_shows_bootstrap_info(self, clustered_did_data):
        """Test that summary shows bootstrap info."""
        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            seed=42
        )

        results = did.fit(
            clustered_did_data,
            outcome="outcome",
            treatment="treated",
            time="post"
        )

        summary = results.summary()

        assert "wild_bootstrap" in summary
        assert "99" in summary  # n_bootstrap
        assert "10" in summary  # n_clusters

    def test_get_params_includes_bootstrap_params(self):
        """Test get_params includes bootstrap parameters."""
        did = DifferenceInDifferences(
            inference="wild_bootstrap",
            n_bootstrap=499,
            bootstrap_weights="webb",
            seed=123
        )

        params = did.get_params()

        assert params["inference"] == "wild_bootstrap"
        assert params["n_bootstrap"] == 499
        assert params["bootstrap_weights"] == "webb"
        assert params["seed"] == 123

    def test_set_params_for_bootstrap(self):
        """Test set_params works for bootstrap parameters."""
        did = DifferenceInDifferences()

        did.set_params(
            inference="wild_bootstrap",
            n_bootstrap=499,
            bootstrap_weights="mammen"
        )

        assert did.inference == "wild_bootstrap"
        assert did.n_bootstrap == 499
        assert did.bootstrap_weights == "mammen"


# =============================================================================
# WildBootstrapResults Tests
# =============================================================================


class TestWildBootstrapResults:
    """Tests for WildBootstrapResults dataclass."""

    def test_summary_format(self, ols_components):
        """Test summary method produces readable output."""
        X, y, residuals, cluster_ids = ols_components

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        summary = results.summary()

        assert "Wild Cluster Bootstrap Results" in summary
        assert "Bootstrap SE:" in summary
        assert "Bootstrap p-value:" in summary
        assert "Number of clusters:" in summary

    def test_print_summary(self, ols_components, capsys):
        """Test print_summary outputs to stdout."""
        X, y, residuals, cluster_ids = ols_components

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=3,
            n_bootstrap=99,
            seed=42
        )

        results.print_summary()

        captured = capsys.readouterr()
        assert "Wild Cluster Bootstrap Results" in captured.out


# =============================================================================
# Edge Case Tests: Few Clusters (< 5)
# =============================================================================


class TestFewClustersEdgeCases:
    """Tests for wild bootstrap behavior with very few clusters."""

    def test_three_clusters_still_works(self):
        """Test wild bootstrap works with 3 clusters (minimum viable)."""
        np.random.seed(42)

        n_clusters = 3
        obs_per_cluster = 40

        data = []
        for cluster in range(n_clusters):
            is_treated = cluster < 2  # 2 treated, 1 control cluster
            cluster_effect = np.random.normal(0, 2)

            for obs in range(obs_per_cluster):
                for period in [0, 1]:
                    y = 10.0 + cluster_effect
                    if period == 1:
                        y += 5.0
                    if is_treated and period == 1:
                        y += 3.0
                    y += np.random.normal(0, 1)

                    data.append({
                        "cluster": cluster,
                        "unit": cluster * obs_per_cluster + obs,
                        "period": period,
                        "treated": int(is_treated),
                        "post": period,
                        "outcome": y,
                    })

        df = pd.DataFrame(data)

        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            bootstrap_weights="webb",  # Webb recommended for few clusters
            seed=42
        )

        # Should warn about few clusters but still produce valid results
        with pytest.warns(UserWarning, match="Only 3 clusters"):
            results = did.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="post"
            )

        assert results.se > 0
        assert results.inference_method == "wild_bootstrap"
        assert results.n_clusters == 3

    def test_two_clusters_minimum(self):
        """Test wild bootstrap works with exactly 2 clusters (absolute minimum)."""
        np.random.seed(42)

        n_clusters = 2
        obs_per_cluster = 50

        data = []
        for cluster in range(n_clusters):
            is_treated = cluster == 0
            cluster_effect = np.random.normal(0, 2)

            for obs in range(obs_per_cluster):
                for period in [0, 1]:
                    y = 10.0 + cluster_effect
                    if period == 1:
                        y += 5.0
                    if is_treated and period == 1:
                        y += 3.0
                    y += np.random.normal(0, 1)

                    data.append({
                        "cluster": cluster,
                        "unit": cluster * obs_per_cluster + obs,
                        "period": period,
                        "treated": int(is_treated),
                        "post": period,
                        "outcome": y,
                    })

        df = pd.DataFrame(data)

        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=99,
            bootstrap_weights="webb",
            seed=42
        )

        # Should warn about few clusters
        with pytest.warns(UserWarning, match="Only 2 clusters"):
            results = did.fit(
                df,
                outcome="outcome",
                treatment="treated",
                time="post"
            )

        # Results should still be valid (though may have high variance)
        assert results.se > 0
        assert np.isfinite(results.att)
        assert results.n_clusters == 2

    def test_few_clusters_webb_vs_rademacher(self, few_cluster_data):
        """Test that Webb weights produce different (often more conservative) SEs than Rademacher with few clusters."""
        did_webb = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=199,
            bootstrap_weights="webb",
            seed=42
        )

        did_rademacher = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=199,
            bootstrap_weights="rademacher",
            seed=42
        )

        with pytest.warns(UserWarning):
            results_webb = did_webb.fit(
                few_cluster_data,
                outcome="outcome",
                treatment="treated",
                time="post"
            )

        with pytest.warns(UserWarning):
            results_rademacher = did_rademacher.fit(
                few_cluster_data,
                outcome="outcome",
                treatment="treated",
                time="post"
            )

        # Both should produce valid results
        assert results_webb.se > 0
        assert results_rademacher.se > 0
        # ATT should be identical (same point estimate)
        assert results_webb.att == results_rademacher.att
        # SEs will differ due to different weight distributions
        # (This is expected, not necessarily one > other)

    def test_few_clusters_confidence_intervals_valid(self, few_cluster_data):
        """Test that CIs are valid even with few clusters."""
        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=199,
            bootstrap_weights="webb",
            seed=42
        )

        with pytest.warns(UserWarning):
            results = did.fit(
                few_cluster_data,
                outcome="outcome",
                treatment="treated",
                time="post"
            )

        lower, upper = results.conf_int
        assert lower < upper
        # CI should contain the point estimate
        assert lower < results.att < upper
