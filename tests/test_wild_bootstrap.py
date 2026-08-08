"""
Tests for Wild Cluster Bootstrap functionality.

Tests the wild_bootstrap_se() function and its integration with DiD estimators.
"""

import json
import re
from pathlib import Path

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

                data.append(
                    {
                        "cluster": cluster,
                        "unit": cluster * obs_per_cluster + obs,
                        "period": period,
                        "treated": int(is_treated),
                        "post": period,
                        "outcome": y,
                    }
                )

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

                data.append(
                    {
                        "cluster": cluster,
                        "unit": cluster * obs_per_cluster + obs,
                        "period": period,
                        "treated": int(is_treated),
                        "post": period,
                        "outcome": y,
                    }
                )

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

        expected_values = np.array(
            [
                -np.sqrt(3 / 2),
                -np.sqrt(2 / 2),
                -np.sqrt(1 / 2),
                np.sqrt(1 / 2),
                np.sqrt(2 / 2),
                np.sqrt(3 / 2),
            ]
        )

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

    def test_returns_wild_bootstrap_results(self, ols_components, ci_params):
        """Test that function returns WildBootstrapResults."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        assert isinstance(results, WildBootstrapResults)

    def test_se_is_positive(self, ols_components, ci_params):
        """Test bootstrap SE is positive."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        assert results.se > 0

    def test_p_value_in_valid_range(self, ols_components, ci_params):
        """Test p-value is in [0, 1]."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        assert 0 <= results.p_value <= 1

    def test_ci_contains_reasonable_values(self, ols_components, ci_params):
        """Test CI bounds are ordered correctly."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(199)

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        assert results.ci_lower < results.ci_upper

    def test_reproducibility_with_seed(self, ols_components, ci_params):
        """Test same seed gives same results."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        results1 = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        results2 = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        # The reported `se` is the analytical cluster-robust (CR1) SE and the CI
        # is from test inversion; both are reproducible only up to the
        # bit-reproducibility of the underlying (possibly threaded BLAS / Rust)
        # cluster-vcov solve (~1e-13). The p-value is count-based and exact.
        assert results1.se == pytest.approx(results2.se, rel=1e-9)
        assert results1.p_value == results2.p_value
        assert results1.ci_lower == pytest.approx(results2.ci_lower, rel=1e-9)

    def test_different_seeds_different_results(self, ols_components, ci_params):
        """Test different seeds give different results."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        results1 = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        results2 = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=123
        )

        # With 10 clusters and n_boot < 2**9, signs are sampled (not enumerated),
        # so different seeds draw different bootstrap samples. The reported `se`
        # is the analytical CR1 SE (seed-independent by construction); it is the
        # inverted CI that reflects the random draws, so assert the CI differs.
        assert (results1.ci_lower, results1.ci_upper) != (results2.ci_lower, results2.ci_upper)

    def test_different_weight_types(self, ols_components, ci_params):
        """Test all weight types produce valid results."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        for weight_type in ["rademacher", "webb", "mammen"]:
            results = wild_bootstrap_se(
                X,
                y,
                residuals,
                cluster_ids,
                coefficient_index=3,
                n_bootstrap=n_boot,
                weight_type=weight_type,
                seed=42,
            )

            assert results.se > 0
            assert 0 <= results.p_value <= 1
            assert results.weight_type == weight_type

    def test_invalid_weight_type_raises(self, ols_components):
        """Test invalid weight type raises ValueError."""
        X, y, residuals, cluster_ids = ols_components

        with pytest.raises(ValueError, match="weight_type must be one of"):
            wild_bootstrap_se(
                X, y, residuals, cluster_ids, coefficient_index=3, weight_type="invalid"
            )

    def test_few_clusters_warning(self, few_cluster_data, ci_params):
        """Test warning when clusters < 5."""
        data = few_cluster_data
        n_boot = ci_params.bootstrap(99)

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
                X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
            )

    def test_too_few_clusters_raises(self, ols_components):
        """Test error when clusters < 2."""
        X, y, residuals, _ = ols_components

        # Create single cluster
        single_cluster = np.zeros(len(y))

        with pytest.raises(ValueError, match="at least 2 clusters"):
            wild_bootstrap_se(X, y, residuals, single_cluster, coefficient_index=3)

    def test_n_clusters_reported_correctly(self, ols_components, ci_params):
        """Test n_clusters is reported correctly."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        assert results.n_clusters == 10

    def test_n_bootstrap_reported_correctly(self, ols_components, ci_params):
        """Test n_bootstrap is reported correctly."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(199)

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        assert results.n_bootstrap == n_boot


# =============================================================================
# Integration with Estimators
# =============================================================================


class TestEstimatorIntegration:
    """Tests for wild bootstrap integration with DiD estimators."""

    def test_did_with_wild_bootstrap(self, clustered_did_data, ci_params):
        """Test DifferenceInDifferences with wild bootstrap."""
        n_boot = ci_params.bootstrap(99)
        did = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        )

        results = did.fit(clustered_did_data, outcome="outcome", treatment="treated", post="post")

        assert results.inference_method == "wild_bootstrap"
        assert results.n_bootstrap == n_boot
        assert results.n_clusters == 10
        assert results.se > 0

    def test_did_wild_bootstrap_reproducibility(self, clustered_did_data, ci_params):
        """Test wild bootstrap results are reproducible with seed."""
        n_boot = ci_params.bootstrap(99)
        did1 = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        )

        did2 = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        )

        results1 = did1.fit(clustered_did_data, outcome="outcome", treatment="treated", post="post")

        results2 = did2.fit(clustered_did_data, outcome="outcome", treatment="treated", post="post")

        # se = analytical CR1 SE (reproducible up to the cluster-vcov solve's
        # bit-reproducibility, ~1e-13 on threaded BLAS / Rust); p-value is exact.
        assert results1.se == pytest.approx(results2.se, rel=1e-9)
        assert results1.p_value == results2.p_value

    def test_did_analytical_vs_bootstrap_att_same(self, clustered_did_data, ci_params):
        """Test that ATT is the same regardless of inference method."""
        n_boot = ci_params.bootstrap(99)
        did_analytical = DifferenceInDifferences(cluster="cluster")
        did_bootstrap = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        )

        results_analytical = did_analytical.fit(
            clustered_did_data, outcome="outcome", treatment="treated", post="post"
        )

        results_bootstrap = did_bootstrap.fit(
            clustered_did_data, outcome="outcome", treatment="treated", post="post"
        )

        # ATT should be identical
        assert results_analytical.att == results_bootstrap.att

    def test_did_wild_bootstrap_with_webb_weights(self, clustered_did_data, ci_params):
        """Test wild bootstrap with Webb weights."""
        n_boot = ci_params.bootstrap(99)
        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            bootstrap_weights="webb",
            seed=42,
        )

        results = did.fit(clustered_did_data, outcome="outcome", treatment="treated", post="post")

        assert results.inference_method == "wild_bootstrap"
        assert results.se > 0

    def test_did_wild_bootstrap_requires_cluster(self, clustered_did_data, ci_params):
        """Wild bootstrap without cluster= raises (fail-closed, M-096).

        Flipped BY DESIGN in 3.9: this previously pinned a SILENT fallback
        to analytical inference — a no-silent-failures violation the
        selector contract closes. The name is now literally true.
        """
        n_boot = ci_params.bootstrap(99)
        did = DifferenceInDifferences(
            inference="wild_bootstrap", n_bootstrap=n_boot, seed=42  # No cluster specified
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "inference='wild_bootstrap' requires cluster=. The wild cluster "
                "bootstrap resamples at the cluster level; pass cluster= or use "
                "inference='analytical'."
            ),
        ):
            did.fit(clustered_did_data, outcome="outcome", treatment="treated", post="post")

    def test_twfe_with_wild_bootstrap(self, clustered_did_data, ci_params):
        """Test TwoWayFixedEffects with wild bootstrap."""
        n_boot = ci_params.bootstrap(99)
        twfe = TwoWayFixedEffects(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        )

        results = twfe.fit(
            clustered_did_data, outcome="outcome", treatment="treated", post="period", unit="unit"
        )

        assert results.inference_method == "wild_bootstrap"
        assert results.n_bootstrap == n_boot
        assert results.se > 0

    def test_summary_shows_bootstrap_info(self, clustered_did_data, ci_params):
        """Test that summary shows bootstrap info."""
        n_boot = ci_params.bootstrap(99)
        did = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        )

        results = did.fit(clustered_did_data, outcome="outcome", treatment="treated", post="post")

        summary = results.summary()

        assert "wild_bootstrap" in summary
        assert str(n_boot) in summary  # n_bootstrap
        assert "10" in summary  # n_clusters

    def test_get_params_includes_bootstrap_params(self):
        """Test get_params includes bootstrap parameters."""
        did = DifferenceInDifferences(
            inference="wild_bootstrap", n_bootstrap=499, bootstrap_weights="webb", seed=123
        )

        params = did.get_params()

        assert params["inference"] == "wild_bootstrap"
        assert params["n_bootstrap"] == 499
        assert params["bootstrap_weights"] == "webb"
        assert params["seed"] == 123

    def test_set_params_for_bootstrap(self):
        """Test set_params works for bootstrap parameters."""
        did = DifferenceInDifferences()

        did.set_params(inference="wild_bootstrap", n_bootstrap=499, bootstrap_weights="mammen")

        assert did.inference == "wild_bootstrap"
        assert did.n_bootstrap == 499
        assert did.bootstrap_weights == "mammen"


# =============================================================================
# WildBootstrapResults Tests
# =============================================================================


class TestWildBootstrapResults:
    """Tests for WildBootstrapResults dataclass."""

    def test_summary_format(self, ols_components, ci_params):
        """Test summary method produces readable output."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        summary = results.summary()

        assert "Wild Cluster Bootstrap Results" in summary
        assert "Cluster-robust SE:" in summary
        assert "Bootstrap p-value:" in summary
        assert "Number of clusters:" in summary

    def test_print_summary(self, ols_components, capsys, ci_params):
        """Test print_summary outputs to stdout."""
        X, y, residuals, cluster_ids = ols_components
        n_boot = ci_params.bootstrap(99)

        results = wild_bootstrap_se(
            X, y, residuals, cluster_ids, coefficient_index=3, n_bootstrap=n_boot, seed=42
        )

        results.print_summary()

        captured = capsys.readouterr()
        assert "Wild Cluster Bootstrap Results" in captured.out


# =============================================================================
# Edge Case Tests: Few Clusters (< 5)
# =============================================================================


class TestFewClustersEdgeCases:
    """Tests for wild bootstrap behavior with very few clusters."""

    def test_three_clusters_still_works(self, ci_params):
        """Test wild bootstrap works with 3 clusters (minimum viable)."""
        np.random.seed(42)
        n_boot = ci_params.bootstrap(99)

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

                    data.append(
                        {
                            "cluster": cluster,
                            "unit": cluster * obs_per_cluster + obs,
                            "period": period,
                            "treated": int(is_treated),
                            "post": period,
                            "outcome": y,
                        }
                    )

        df = pd.DataFrame(data)

        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            bootstrap_weights="webb",  # Webb recommended for few clusters
            seed=42,
        )

        # Should warn about few clusters but still produce valid results
        with pytest.warns(UserWarning, match="Only 3 clusters"):
            results = did.fit(df, outcome="outcome", treatment="treated", post="post")

        assert results.se > 0
        assert results.inference_method == "wild_bootstrap"
        assert results.n_clusters == 3

    def test_two_clusters_minimum(self, ci_params):
        """Test wild bootstrap works with exactly 2 clusters (absolute minimum)."""
        np.random.seed(42)
        n_boot = ci_params.bootstrap(99)

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

                    data.append(
                        {
                            "cluster": cluster,
                            "unit": cluster * obs_per_cluster + obs,
                            "period": period,
                            "treated": int(is_treated),
                            "post": period,
                            "outcome": y,
                        }
                    )

        df = pd.DataFrame(data)

        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            bootstrap_weights="webb",
            seed=42,
        )

        # Should warn about few clusters
        with pytest.warns(UserWarning, match="Only 2 clusters"):
            results = did.fit(df, outcome="outcome", treatment="treated", post="post")

        # Results should still be valid (though may have high variance)
        assert results.se > 0
        assert np.isfinite(results.att)
        assert results.n_clusters == 2

    def test_few_clusters_webb_vs_rademacher(self, few_cluster_data, ci_params):
        """Test that Webb weights produce different (often more conservative) SEs than Rademacher with few clusters."""
        n_boot = ci_params.bootstrap(199)
        did_webb = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            bootstrap_weights="webb",
            seed=42,
        )

        did_rademacher = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            bootstrap_weights="rademacher",
            seed=42,
        )

        with pytest.warns(UserWarning):
            results_webb = did_webb.fit(
                few_cluster_data, outcome="outcome", treatment="treated", post="post"
            )

        with pytest.warns(UserWarning):
            results_rademacher = did_rademacher.fit(
                few_cluster_data, outcome="outcome", treatment="treated", post="post"
            )

        # Both should produce valid results
        assert results_webb.se > 0
        assert results_rademacher.se > 0
        # ATT should be identical (same point estimate)
        assert results_webb.att == results_rademacher.att
        # SEs will differ due to different weight distributions
        # (This is expected, not necessarily one > other)

    def test_few_clusters_confidence_intervals_valid(self, few_cluster_data, ci_params):
        """Test that CIs are valid even with few clusters."""
        n_boot = ci_params.bootstrap(199)
        did = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            bootstrap_weights="webb",
            seed=42,
        )

        with pytest.warns(UserWarning):
            results = did.fit(few_cluster_data, outcome="outcome", treatment="treated", post="post")

        lower, upper = results.conf_int
        assert lower < upper
        # The CI is obtained by inverting the bootstrap test, so it need NOT be
        # centered on (or even contain) the point estimate under inversion with
        # very few clusters. The load-bearing guarantee is internal consistency
        # between the test and the interval: 0 lies outside the CI iff the
        # bootstrap p-value rejects at alpha.
        zero_in_ci = lower <= 0.0 <= upper
        rejects = results.p_value < did.alpha
        assert zero_in_ci != rejects


# =============================================================================
# Degenerate bootstrap: all-or-nothing NaN inference contract
# =============================================================================


class TestWildBootstrapDegenerateAllNaN:
    """Verify wild_bootstrap_se() returns the full NaN inference tuple when
    the bootstrap is degenerate (fewer than 2 valid coefficient draws),
    per feedback_bootstrap_nan_on_invalid_contract.md.

    Mocks the internal solve_ols path so we can force `se_star <= 0` on
    every draw (n_valid == 0) and exactly-one-valid (n_valid == 1) without
    relying on a pathological numerical design. These two branches are not
    exercised by the analytical-design tests above.
    """

    def _make_ols_components(self, n: int = 40):
        rng = np.random.default_rng(0)
        cluster_ids = np.repeat(np.arange(8), 5)
        X = np.column_stack(
            [
                np.ones(n),
                rng.normal(size=n),
            ]
        )
        y = X @ np.array([1.0, 0.5]) + rng.normal(scale=0.1, size=n)
        return X, y, cluster_ids

    def test_degenerate_nan_restricted_fit_returns_all_nan(self, monkeypatch):
        """When the restricted residualization yields non-finite values (so
        every bootstrap statistic is NaN, n_valid == 0), se / t_stat / p_value /
        CI are all NaN together — the full inference quadruple moves as a unit
        (feedback_bootstrap_nan_on_invalid_contract).
        """
        from diff_diff import utils as utils_mod
        from tests.conftest import assert_nan_inference

        X, y, cluster_ids = self._make_ols_components()
        orig_solve = utils_mod._solve_ols_linalg

        def fake_solve(X_, y_, cluster_ids=None, return_vcov=True, return_fitted=False, **kw):
            # The original cluster-robust fit (return_fitted=False) passes
            # through. The two restricted residualization fits ask for fitted
            # values (return_fitted=True) — poison those with NaN so the
            # restricted residuals, and hence every bootstrap t*, become
            # non-finite (n_valid == 0).
            if return_fitted:
                n_, k_ = X_.shape[0], X_.shape[1]
                return np.zeros(k_), np.zeros(n_), np.full(n_, np.nan), None
            return orig_solve(X_, y_, cluster_ids=cluster_ids, return_vcov=return_vcov, **kw)

        monkeypatch.setattr(utils_mod, "_solve_ols_linalg", fake_solve)
        results = utils_mod.wild_bootstrap_se(
            X=X,
            y=y,
            residuals=y - y.mean(),
            cluster_ids=cluster_ids,
            coefficient_index=1,
            n_bootstrap=20,
            seed=1,
        )
        assert_nan_inference(
            {
                "se": results.se,
                "t_stat": results.t_stat_original,
                "p_value": results.p_value,
                "conf_int": (results.ci_lower, results.ci_upper),
            }
        )

    def test_degenerate_nonfinite_analytical_se_returns_all_nan(self, monkeypatch):
        """When the analytical cluster-robust SE of the coefficient is
        non-finite (e.g. an unidentified / rank-deficient estimand), no
        bootstrap is attempted and the full inference quadruple is NaN.
        """
        from diff_diff import utils as utils_mod
        from tests.conftest import assert_nan_inference

        X, y, cluster_ids = self._make_ols_components()
        orig_solve = utils_mod._solve_ols_linalg

        def fake_solve(X_, y_, cluster_ids=None, return_vcov=True, return_fitted=False, **kw):
            # Poison the variance of the coefficient of interest on the original
            # cluster-robust fit (cluster_ids set, vcov requested, no fitted).
            if cluster_ids is not None and return_vcov and not return_fitted:
                coefs, residuals, vcov = orig_solve(
                    X_, y_, cluster_ids=cluster_ids, return_vcov=True, **kw
                )
                vcov = np.array(vcov, dtype=float)
                vcov[1, 1] = np.nan
                return coefs, residuals, vcov
            return orig_solve(
                X_,
                y_,
                cluster_ids=cluster_ids,
                return_vcov=return_vcov,
                return_fitted=return_fitted,
                **kw,
            )

        monkeypatch.setattr(utils_mod, "_solve_ols_linalg", fake_solve)
        results = utils_mod.wild_bootstrap_se(
            X=X,
            y=y,
            residuals=y - y.mean(),
            cluster_ids=cluster_ids,
            coefficient_index=1,
            n_bootstrap=20,
            seed=1,
        )
        assert_nan_inference(
            {
                "se": results.se,
                "t_stat": results.t_stat_original,
                "p_value": results.p_value,
                "conf_int": (results.ci_lower, results.ci_upper),
            }
        )


# =============================================================================
# Correctness & consistency (regression tests for issue #543)
# =============================================================================


def _make_clustered(n_clusters, att, seed, obs_per_cluster=10):
    """Clustered 2x2 DiD data with a known true ATT (cluster-level treatment)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        is_treated = c < n_clusters // 2
        cluster_effect = rng.normal(0, 2)
        for _ in range(obs_per_cluster):
            for period in (0, 1):
                y = 10.0 + cluster_effect
                if period == 1:
                    y += 3.0
                if is_treated and period == 1:
                    y += att
                y += rng.normal(0, 0.5)
                rows.append(
                    {"cluster": c, "treated": int(is_treated), "post": period, "outcome": y}
                )
    return pd.DataFrame(rows)


class TestWildBootstrapCorrectness:
    """Regression tests for the WCR null-imposition fix (issue #543).

    The original bug never imposed the null, so the bootstrap t* distribution
    was centered on the estimate instead of 0 and the p-value was ~0.5-0.86
    regardless of significance, contradicting a CI that (coincidentally)
    excluded 0. These tests pin the corrected behaviour: a strong true effect
    is significant, and the p-value and CI are always mutually consistent.
    """

    def test_strong_effect_is_significant(self, clustered_did_data, ci_params):
        """A strong true effect (ATT=3, 10 clusters) must be significant.

        On the buggy implementation this returned p ~= 0.85 (non-significant)
        while the CI excluded 0 — the exact #543 contradiction.
        """
        n_boot = ci_params.bootstrap(999, min_n=99)
        did = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        )
        res = did.fit(clustered_did_data, outcome="outcome", treatment="treated", post="post")
        lower, upper = res.conf_int
        assert res.p_value < 0.05, f"strong effect should be significant, got p={res.p_value}"
        assert not (lower <= 0.0 <= upper), "CI should exclude 0 for a strong effect"

    def test_p_value_ci_consistency(self, ci_params):
        """0 outside the CI iff the bootstrap p-value rejects at alpha, across a
        range of effect sizes / seeds (the property the bug violated)."""
        n_boot = ci_params.bootstrap(999, min_n=99)
        for att, seed in [(0.0, 1), (0.2, 2), (0.5, 3), (2.5, 4)]:
            df = _make_clustered(20, att, seed)
            did = DifferenceInDifferences(
                cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=seed
            )
            res = did.fit(df, outcome="outcome", treatment="treated", post="post")
            lower, upper = res.conf_int
            zero_in_ci = lower <= 0.0 <= upper
            rejects = res.p_value < did.alpha
            assert zero_in_ci != rejects, (
                f"inconsistent p/CI at att={att}, seed={seed}: "
                f"p={res.p_value}, CI=[{lower}, {upper}]"
            )

    def test_true_null_not_significant(self, ci_params):
        """Under a true null (ATT=0) the test should not reject (p not tiny,
        0 inside the CI)."""
        n_boot = ci_params.bootstrap(999, min_n=99)
        df = _make_clustered(20, 0.0, seed=7)
        did = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=7
        )
        res = did.fit(df, outcome="outcome", treatment="treated", post="post")
        lower, upper = res.conf_int
        assert res.p_value > 0.05
        assert lower <= 0.0 <= upper

    def test_enumeration_is_deterministic(self):
        """With few clusters (Rademacher), the full sign-vector set is
        enumerated, so results are independent of the seed and n_bootstrap is
        reported as 2**n_clusters."""
        df = _make_clustered(6, 2.5, seed=1)  # 6 clusters -> 2**5 = 32 <= 999 -> enumerate
        r1 = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=999, seed=1
        ).fit(df, outcome="outcome", treatment="treated", post="post")
        r2 = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=999, seed=999
        ).fit(df, outcome="outcome", treatment="treated", post="post")
        assert r1.n_bootstrap == 2**6
        assert r1.p_value == r2.p_value
        # CI is reproducible up to the cluster-vcov solve's bit-reproducibility
        # (~1e-13 on threaded BLAS / Rust); seed-independence is the point.
        assert r1.conf_int == pytest.approx(r2.conf_int, rel=1e-9)

    def test_se_matches_analytical_cluster_robust(self, clustered_did_data, ci_params):
        """The reported wild-bootstrap SE is the analytical cluster-robust (CR1)
        SE — identical to the analytical cluster-robust fit."""
        n_boot = ci_params.bootstrap(199, min_n=49)
        analytical = DifferenceInDifferences(cluster="cluster").fit(
            clustered_did_data, outcome="outcome", treatment="treated", post="post"
        )
        boot = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        ).fit(clustered_did_data, outcome="outcome", treatment="treated", post="post")
        assert boot.se == pytest.approx(analytical.se, rel=1e-10)

    def test_many_clusters_ci_comparable_to_analytical(self, ci_params):
        """With many clusters the inverted wild CI is comparable to the
        analytical cluster-robust CI (not wildly different, as the old
        percentile-of-coefficients CI could be)."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        df = _make_clustered(30, 1.0, seed=11)
        analytical = DifferenceInDifferences(cluster="cluster").fit(
            df, outcome="outcome", treatment="treated", post="post"
        )
        boot = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_boot, seed=11
        ).fit(df, outcome="outcome", treatment="treated", post="post")
        a_half = (analytical.conf_int[1] - analytical.conf_int[0]) / 2
        b_half = (boot.conf_int[1] - boot.conf_int[0]) / 2
        assert 0.7 < (b_half / a_half) < 1.5

    def test_equal_tailed_consistent(self, ci_params):
        """Equal-tailed p_val_type yields a valid, internally consistent CI."""
        n_boot = ci_params.bootstrap(999, min_n=99)
        df = _make_clustered(20, 0.6, seed=5)
        res = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            seed=5,
            p_val_type="equal-tailed",
        ).fit(df, outcome="outcome", treatment="treated", post="post")
        lower, upper = res.conf_int
        assert lower < upper
        assert (not (lower <= 0.0 <= upper)) == (res.p_value < 0.05)

    def test_p_val_type_round_trips_in_params(self):
        """p_val_type is exposed via get_params / set_params and round-trips."""
        est = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", p_val_type="equal-tailed"
        )
        params = est.get_params()
        assert params["p_val_type"] == "equal-tailed"
        clone = DifferenceInDifferences(**params)
        assert clone.p_val_type == "equal-tailed"
        clone.set_params(p_val_type="two-tailed")
        assert clone.p_val_type == "two-tailed"

    def test_invalid_p_val_type_raises(self, ols_components):
        """An unrecognized p_val_type raises ValueError."""
        X, y, residuals, cluster_ids = ols_components
        with pytest.raises(ValueError, match="p_val_type must be one of"):
            wild_bootstrap_se(X, y, residuals, cluster_ids, coefficient_index=3, p_val_type="bogus")


# =============================================================================
# Cross-check: independent brute-force WCR == production (== fwildclusterboot)
# =============================================================================


def _brute_force_wcr_refit(X, y, cluster_ids, coef_index, null=0.0):
    """Independent textbook WCR bootstrap, computed separately from the
    production fast-form path.

    Imposes the null by dropping the coefficient's column, enumerates ALL
    ``2**G`` Rademacher sign-vectors, does a FULL OLS refit per draw, forms the
    CR1 cluster-robust SE, and counts strict exceedances ``|t*| > |t0|``. This
    is the same WCR statistic ``fwildclusterboot::boottest`` computes — the two
    were verified to agree on the bootstrap t-distribution to ~6e-14 — so a
    match here certifies R-parity without requiring R at test time.

    Returns ``(t0, se_a, p_raw)`` where ``p_raw(r)`` is the two-tailed bootstrap
    p-value at null ``r`` (unfloored).
    """
    from itertools import product

    uniq = np.unique(cluster_ids)
    G = len(uniq)
    n, k = X.shape
    cidx = [np.where(cluster_ids == c)[0] for c in uniq]
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    XtX_inv = np.linalg.inv(X.T @ X)
    a = X @ XtX_inv[:, coef_index]

    def cr1(resid):
        return np.sqrt(corr * sum((a[idx] @ resid[idx]) ** 2 for idx in cidx))

    b = XtX_inv @ X.T @ y
    se_a = cr1(y - X @ b)
    t0 = (b[coef_index] - null) / se_a
    Xr = np.delete(X, coef_index, axis=1)
    Pr = Xr @ np.linalg.inv(Xr.T @ Xr) @ Xr.T
    cl_pos = {c: i for i, c in enumerate(uniq)}
    cl_of = np.array([cl_pos[c] for c in cluster_ids])
    signs = np.array(list(product([-1.0, 1.0], repeat=G)))

    def p_raw(r):
        yr = y - X[:, coef_index] * r
        u = yr - Pr @ yr
        fit = (Pr @ yr) + X[:, coef_index] * r  # restricted fitted in original-y space
        tb = []
        for w in signs:
            ystar = fit + u * w[cl_of]
            bs = XtX_inv @ X.T @ ystar
            tb.append((bs[coef_index] - r) / cr1(ystar - X @ bs))
        tb = np.array(tb)
        t0r = (b[coef_index] - r) / se_a
        return float(np.mean(np.abs(tb) > abs(t0r) + 1e-9 * max(1.0, abs(t0r))))

    return t0, se_a, p_raw


def test_wcr_matches_independent_bruteforce():
    """Production WCR == independent full-refit enumeration (== boottest).

    A 6-cluster, noisy design gives an interior (non-floored) bootstrap p-value
    so the exact p, SE, t-stat, and inverted CI can all be checked against the
    independent reference.
    """
    rng = np.random.default_rng(3)
    rows = []
    for c in range(6):
        is_treated = c < 3
        cluster_effect = rng.normal(0, 1.5)
        for _ in range(8):
            for period in (0, 1):
                y = 4.0 + cluster_effect + 1.0 * period
                if is_treated and period == 1:
                    y += 0.3  # weak effect + heavy noise -> interior p
                y += rng.normal(0, 4.0)
                rows.append(
                    {"cluster": c, "treated": int(is_treated), "post": period, "outcome": y}
                )
    df = pd.DataFrame(rows)
    X = np.column_stack([np.ones(len(df)), df.treated, df.post, df.treated * df.post])
    y = df.outcome.to_numpy()
    cl = df.cluster.to_numpy()
    j = 3
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    res = wild_bootstrap_se(X, y, y - X @ beta, cl, coefficient_index=j, n_bootstrap=999, seed=3)
    t0, se_a, p_raw = _brute_force_wcr_refit(X, y, cl, j)

    # Core statistic: exact agreement with the independent reference.
    assert res.t_stat_original == pytest.approx(t0, rel=1e-9)
    assert res.se == pytest.approx(se_a, rel=1e-9)
    assert res.p_value == pytest.approx(p_raw(0.0), abs=1e-9)  # interior p -> floor inactive

    # CI by test inversion: the brute-force p-value at the production endpoints
    # sits at the alpha crossing (granular to ~1/2**G with full enumeration).
    assert abs(p_raw(res.ci_lower) - 0.05) <= 2.0 / 2**6
    assert abs(p_raw(res.ci_upper) - 0.05) <= 2.0 / 2**6


# =============================================================================
# R parity — fwildclusterboot::boottest (skip-guarded golden)
# =============================================================================
#
# Pins wild_bootstrap_se against R `fwildclusterboot::boottest()` on a fixed
# few-cluster golden (G=6, fully enumerated -> deterministic both sides).
# Regenerate with `Rscript benchmarks/R/generate_wild_cluster_boot_golden.R`.
# R is NOT required at test time; the golden JSON is checked in.

_WCB_GOLDEN = Path(__file__).parent.parent / "benchmarks" / "data" / "wild_cluster_boot_golden.json"
_WCB_DATA = Path(__file__).parent.parent / "benchmarks" / "data" / "wild_cluster_boot_test_data.csv"
_WCB_AVAILABLE = _WCB_GOLDEN.is_file() and _WCB_DATA.is_file()


@pytest.mark.skipif(not _WCB_AVAILABLE, reason="fwildclusterboot golden fixture not present")
class TestWildBootstrapParityR:
    """Cross-language parity with R fwildclusterboot::boottest (WCR defaults)."""

    @pytest.fixture
    def golden(self):
        with _WCB_GOLDEN.open() as f:
            return json.load(f)

    @pytest.fixture
    def design(self):
        df = pd.read_csv(_WCB_DATA)
        X = np.column_stack([np.ones(len(df)), df.treated, df.post, df.treated * df.post])
        return X, df.y.to_numpy(), df.cluster.to_numpy()

    def _fit(self, design, p_val_type):
        X, y, cl = design
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return wild_bootstrap_se(
            X,
            y,
            y - X @ beta,
            cl,
            coefficient_index=3,
            n_bootstrap=99999,
            weight_type="rademacher",
            seed=1,
            p_val_type=p_val_type,
        )

    def test_se_matches_r(self, golden, design):
        # Reported SE is the analytical CR1 clustered SE (== feols se()).
        assert self._fit(design, "two-tailed").se == pytest.approx(golden["se_cr1"], abs=1e-6)

    def test_tstat_matches_r(self, golden, design):
        assert self._fit(design, "two-tailed").t_stat_original == pytest.approx(
            golden["two_tailed"]["t_stat"], abs=1e-6
        )

    def test_two_tailed_p_value_matches_r(self, golden, design):
        # Interior p (by construction) -> the 1/(B+1) floor is inactive, so the
        # WCR p-value matches boottest's strict-exceedance count exactly.
        assert self._fit(design, "two-tailed").p_value == pytest.approx(
            golden["two_tailed"]["p_val"], abs=1e-9
        )

    def test_two_tailed_ci_matches_r(self, golden, design):
        res = self._fit(design, "two-tailed")
        lo, hi = golden["two_tailed"]["conf_int"]
        # Inversion convention differs (bisection vs boottest's grid); agree ~1e-4.
        assert res.ci_lower == pytest.approx(lo, abs=5e-4)
        assert res.ci_upper == pytest.approx(hi, abs=5e-4)

    def test_equal_tailed_matches_r(self, golden, design):
        res = self._fit(design, "equal-tailed")
        lo, hi = golden["equal_tailed"]["conf_int"]
        assert res.p_value == pytest.approx(golden["equal_tailed"]["p_val"], abs=1e-9)
        assert res.ci_lower == pytest.approx(lo, abs=5e-4)
        assert res.ci_upper == pytest.approx(hi, abs=5e-4)


# =============================================================================
# Floor / low-effective-draw consistency + p_val_type result propagation
# =============================================================================


@pytest.mark.parametrize(
    "n_clusters, n_bootstrap, att, seed",
    [
        (6, 9, 3.0, 1),  # RNG path, 9 draws -> floor 1/10 = 0.10 > alpha
        (4, 999, 4.0, 2),  # enumerated 2**4 = 16 draws -> floor 1/17 ~= 0.059 > alpha
    ],
)
def test_low_draw_floor_preserves_consistency(n_clusters, n_bootstrap, att, seed):
    """With very few effective draws the p-value floor 1/(n_valid+1) can exceed
    alpha; the floor must NOT then flip a bootstrap-significant result (0 outside
    the inverted CI) to non-significant. The verdict must always match the CI.
    """
    import warnings

    df = _make_clustered(n_clusters, att, seed)
    did = DifferenceInDifferences(
        cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_bootstrap, seed=seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # few-cluster warning
        res = did.fit(df, outcome="outcome", treatment="treated", post="post")
    lower, upper = res.conf_int
    zero_in_ci = lower <= 0.0 <= upper
    rejects = res.p_value < did.alpha
    # The exact bug the reviewer flagged: p >= alpha while the CI excludes 0.
    assert zero_in_ci != rejects, (
        f"p/CI inconsistent at G={n_clusters}, B={n_bootstrap}: "
        f"p={res.p_value}, CI=[{lower}, {upper}]"
    )


def test_p_val_type_surfaced_on_results():
    """p_val_type is carried on WildBootstrapResults and the high-level DiD
    result (None for analytical inference), and appears in summary()/to_dict()."""
    df = _make_clustered(20, 0.6, seed=5)
    for ptype in ("two-tailed", "equal-tailed"):
        res = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=999, seed=5, p_val_type=ptype
        ).fit(df, outcome="outcome", treatment="treated", post="post")
        assert res.p_val_type == ptype
        assert res.to_dict()["p_val_type"] == ptype
        assert "Test type:" in res.summary()
    # Analytical inference does not set p_val_type.
    res_a = DifferenceInDifferences(cluster="cluster").fit(
        df, outcome="outcome", treatment="treated", post="post"
    )
    assert res_a.p_val_type is None


def test_wild_bootstrap_results_carries_p_val_type():
    """The low-level WildBootstrapResults dataclass exposes p_val_type."""
    df = _make_clustered(20, 0.6, seed=5)
    X = np.column_stack([np.ones(len(df)), df.treated, df.post, df.treated * df.post])
    y = df.outcome.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = wild_bootstrap_se(
        X,
        y,
        y - X @ beta,
        df.cluster.to_numpy(),
        coefficient_index=3,
        n_bootstrap=999,
        seed=5,
        p_val_type="equal-tailed",
    )
    assert res.p_val_type == "equal-tailed"
    assert "equal-tailed" in res.summary()


def test_twfe_wild_bootstrap_p_val_type_propagates():
    """TwoWayFixedEffects (inherits the WCR path) surfaces p_val_type on its
    result and stays p/CI consistent for the equal-tailed test."""
    rng = np.random.default_rng(7)
    rows = []
    for c in range(20):
        is_treated = c < 10
        ce = rng.normal(0, 2)
        for o in range(10):
            for period in (0, 1):
                y = 10 + ce + 1.0 * period + (0.6 if (is_treated and period == 1) else 0)
                y += rng.normal(0, 0.5)
                rows.append(
                    {
                        "cluster": c,
                        "unit": c * 10 + o,
                        "treated": int(is_treated),
                        "post": period,
                        "outcome": y,
                    }
                )
    df = pd.DataFrame(rows)
    res = TwoWayFixedEffects(
        cluster="cluster",
        inference="wild_bootstrap",
        n_bootstrap=999,
        seed=7,
        p_val_type="equal-tailed",
    ).fit(df, outcome="outcome", treatment="treated", post="post", unit="unit")
    assert res.p_val_type == "equal-tailed"
    assert res.to_dict()["p_val_type"] == "equal-tailed"
    lower, upper = res.conf_int
    assert lower < upper
    assert (not (lower <= 0.0 <= upper)) == (res.p_value < 0.05)


# =============================================================================
# Degenerate-design robustness: no crash, no mixed finite/NaN inference
# =============================================================================


def test_saturated_design_returns_degenerate_not_crash():
    """A saturated 2x2 DiD (G=2, one obs per cluster-period -> n == k, no
    residual DOF) must NOT raise ZeroDivisionError from the CR1 small-sample
    adjustment; it returns the all-or-nothing NaN inference contract."""
    import warnings

    df = pd.DataFrame(
        [
            {
                "cluster": c,
                "treated": c,
                "post": p,
                "outcome": 5.0 + 2 * p + (1.0 if (c and p) else 0),
            }
            for c in range(2)
            for p in (0, 1)
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=99, seed=1
        ).fit(df, outcome="outcome", treatment="treated", post="post")
    # Full inference family is NaN together (no raw exception, no mixed output).
    assert np.isnan(res.se)
    assert np.isnan(res.p_value)
    assert np.isnan(res.conf_int[0]) and np.isnan(res.conf_int[1])


def test_near_zero_se_no_mixed_finite_nan_ci():
    """A near-degenerate two-cluster design (tiny CR1 SE) must not return finite
    se/p with NaN CI endpoints. An unbounded inverted CI is represented with
    +/-inf (NOT NaN), keeping 0 in CI <=> p >= alpha."""
    import warnings

    rng = np.random.default_rng(0)
    rows_X, y, cl = [], [], []
    for c in range(2):
        tr = 1.0 if c < 1 else 0.0
        ce = rng.normal(0, 1)
        for _ in range(2):
            for p in (0.0, 1.0):
                rows_X.append([1.0, tr, p, tr * p])
                y.append(5 + ce + 2 * p + (1.0 if (tr and p) else 0) + rng.normal(0, 0.5))
                cl.append(c)
    X = np.array(rows_X)
    y = np.array(y)
    cl = np.array(cl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wild_bootstrap_se(
            X,
            y,
            y - X @ np.linalg.lstsq(X, y, rcond=None)[0],
            cl,
            coefficient_index=3,
            n_bootstrap=999,
            seed=1,
        )
    finite_point = np.isfinite(res.se) or np.isfinite(res.p_value)
    nan_ci = np.isnan(res.ci_lower) or np.isnan(res.ci_upper)
    assert not (finite_point and nan_ci), "mixed finite point estimate with NaN CI endpoints"
    # Consistency holds whether the result is degenerate or an unbounded interval.
    if np.isfinite(res.p_value):
        zero_in_ci = res.ci_lower <= 0.0 <= res.ci_upper
        assert zero_in_ci == (res.p_value >= 0.05)


# =============================================================================
# Enumeration trigger parity with fwildclusterboot (B >= 2**G)
# =============================================================================


def _make_clustered_g10(n_bootstrap, seed):
    """G=10 design + a wild-bootstrap fit, for the enumeration-boundary tests."""
    rng = np.random.default_rng(99)  # data fixed; only the bootstrap seed varies
    rows = []
    for c in range(10):
        is_treated = c < 5
        ce = rng.normal(0, 1.5)
        for _ in range(6):
            for p in (0, 1):
                y = 4 + ce + 1.0 * p + (0.5 if (is_treated and p == 1) else 0) + rng.normal(0, 2.0)
                rows.append({"cluster": c, "treated": int(is_treated), "post": p, "outcome": y})
    df = pd.DataFrame(rows)
    return DifferenceInDifferences(
        cluster="cluster", inference="wild_bootstrap", n_bootstrap=n_bootstrap, seed=seed
    ).fit(df, outcome="outcome", treatment="treated", post="post")


def test_enumeration_trigger_matches_boottest_boundary():
    """Enumeration fires only when n_bootstrap >= 2**n_clusters, matching
    fwildclusterboot::boottest (verified: G=10 samples at B=1023, enumerates at
    B=1024). Below the threshold the result is seed-dependent (sampled); at/above
    it is deterministic with n_bootstrap == 2**n_clusters."""
    # Below 2**10 = 1024: sampled -> seed-dependent, reported n_bootstrap == B.
    below_a = _make_clustered_g10(999, seed=1)
    below_b = _make_clustered_g10(999, seed=7)
    assert below_a.n_bootstrap == 999
    assert below_a.conf_int != below_b.conf_int  # different seeds -> different draws

    # At/above 2**10: enumerated -> deterministic, reported n_bootstrap == 1024.
    at_a = _make_clustered_g10(1024, seed=1)
    at_b = _make_clustered_g10(1024, seed=7)
    assert at_a.n_bootstrap == 2**10
    assert at_a.p_value == at_b.p_value
    # Seed-independent up to the cluster-vcov solve's bit-reproducibility
    # (~1e-13 on threaded BLAS / Rust); far below the sampled-draw scale (~1e-2).
    assert at_a.conf_int == pytest.approx(at_b.conf_int, rel=1e-9)


def test_single_regressor_design_does_not_crash():
    """A degenerate single-regressor design (the reduced model has zero columns)
    must not raise IndexError; the restricted residuals are the variables
    themselves."""
    import warnings

    rng = np.random.default_rng(3)
    cluster_ids = np.repeat(np.arange(6), 8)
    X = rng.normal(size=(48, 1))  # a single regressor, no intercept
    y = X[:, 0] * 0.5 + rng.normal(scale=0.5, size=48)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wild_bootstrap_se(
            X,
            y,
            y - X @ np.linalg.lstsq(X, y, rcond=None)[0],
            cluster_ids,
            coefficient_index=0,
            n_bootstrap=99,
            seed=1,
        )
    # No exception; whatever inference is returned is self-consistent.
    assert isinstance(res, WildBootstrapResults)
    if np.isfinite(res.p_value):
        assert (res.ci_lower <= 0.0 <= res.ci_upper) == (res.p_value >= 0.05)


def test_wild_bootstrap_rank_deficient_storage_vcov_does_not_crash():
    """The estimator's stored cluster-robust vcov is computed through the
    rank-aware solve_ols path, so a wild-bootstrap fit on a rank-deficient
    full-dummy design (here a fixed-effect dummy that EXACTLY duplicates the
    treatment indicator) does not crash, and the stored vcov is NaN-expanded for
    the dropped column rather than raising on the singular X'X. Regression for
    the storage-vcov gap in `_run_wild_bootstrap_inference` (the bootstrap helper
    already handled rank deficiency internally).
    """
    import warnings

    rng = np.random.default_rng(0)
    rows = []
    for u in range(16):
        treated = int(u < 8)
        fe = "T" if treated else "C"  # the 'T' dummy == treated exactly -> singular X'X
        for period in (0, 1):
            y = 5 + 2 * period + (1.5 if (treated and period) else 0) + rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": u,
                    "fe": fe,
                    "cluster": u % 8,
                    "treated": treated,
                    "post": period,
                    "outcome": y,
                }
            )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected rank-deficient drop warning
        res = DifferenceInDifferences(
            cluster="cluster", inference="wild_bootstrap", n_bootstrap=99, seed=1
        ).fit(df, outcome="outcome", treatment="treated", post="post", fixed_effects=["fe"])
    # ATT identified, bootstrap inference finite, no exception.
    assert np.isfinite(res.att)
    assert np.isfinite(res.se) and res.se > 0
    assert np.isfinite(res.p_value)
    assert np.isfinite(res.conf_int[0]) and np.isfinite(res.conf_int[1])
    # Stored vcov is rank-aware (NaN-expanded for the dropped column), not +/-inf.
    assert res.vcov is not None
    assert np.any(np.isnan(res.vcov))
    assert not np.any(np.isinf(res.vcov))


class TestPrecomputeChunking:
    """The r-independent precompute pass must be chunk-count invariant: forcing
    many draw-chunks (tiny byte budget) reproduces the single-chunk outputs
    (each chunk computes its own rows independently, so the p-value is exact;
    se/CI carry documented ambient-backend tolerances, see in-test notes)."""

    def test_multi_chunk_count_invariant(self, monkeypatch):
        import diff_diff.utils as du

        rng = np.random.default_rng(3)
        G, n_per = 9, 30
        n = G * n_per
        cl = np.repeat(np.arange(G), n_per)
        x = rng.normal(size=n)
        d = (np.arange(n) % 2).astype(float)
        X = np.column_stack([np.ones(n), d, x])
        y = 1.0 + 0.4 * d + 0.2 * x + rng.normal(size=n) + rng.normal(size=G)[cl]
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        one = du.wild_bootstrap_se(X, y, resid, cl, 1, n_bootstrap=499, seed=11)
        # ~3 rows per chunk -> hundreds of chunks.
        monkeypatch.setattr(du, "_WILD_PRECOMPUTE_CHUNK_BYTES", 3 * 8 * 8 * n)
        many = du.wild_bootstrap_se(X, y, resid, cl, 1, n_bootstrap=499, seed=11)

        # p-value: strict-count statistic with a 1e-9 relative tie guard —
        # exact equality expected. se / CI endpoints keep small tolerances
        # for historical cross-run robustness: the Rust clustered-vcov
        # run-to-run wobble was fixed in #653 (first-appearance cluster
        # aggregation; see the REGISTRY determinism note and
        # tests/test_rust_backend.py::TestClusterVcovDeterminism) — the
        # chunked precompute itself is deterministic numpy.
        assert many.p_value == one.p_value
        np.testing.assert_allclose(many.se, one.se, rtol=1e-12)
        np.testing.assert_allclose(many.ci_lower, one.ci_lower, rtol=1e-6)
        np.testing.assert_allclose(many.ci_upper, one.ci_upper, rtol=1e-6)
