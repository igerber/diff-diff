"""Tests for lwdid_wild_bootstrap module."""

import numpy as np
import pandas as pd
import pytest

from diff_diff.lwdid_wild_bootstrap import (
    WildClusterBootstrapResult,
    wild_cluster_bootstrap,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cross_section_data():
    rng = np.random.default_rng(42)
    n = 100
    y = np.concatenate([rng.normal(2, 0.5, 30), rng.normal(0, 0.5, 70)])
    treatment = np.array([1.0] * 30 + [0.0] * 70)
    cluster_ids = np.repeat(np.arange(20), 5)
    controls = rng.normal(0, 1, (n, 2))
    return y, treatment, cluster_ids, controls


# ---------------------------------------------------------------------------
# Result dataclass fields
# ---------------------------------------------------------------------------


class TestWildClusterBootstrapResultFields:
    """Test that result has all expected fields."""

    def test_result_has_att(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        assert hasattr(r, "att")
        assert isinstance(r.att, float)

    def test_result_has_se_bootstrap(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        assert hasattr(r, "se_bootstrap")

    def test_result_has_ci(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        assert hasattr(r, "ci_lower")
        assert hasattr(r, "ci_upper")
        assert r.ci_lower <= r.ci_upper

    def test_result_has_pvalue(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        assert hasattr(r, "pvalue")

    def test_result_has_weight_type(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        assert hasattr(r, "weight_type")
        assert r.weight_type == "rademacher"

    def test_result_has_n_reps(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        assert hasattr(r, "n_reps")

    def test_result_has_n_clusters(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        assert hasattr(r, "n_clusters")
        assert r.n_clusters == 20

    def test_result_has_t_stats(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        assert hasattr(r, "t_stats")
        assert isinstance(r.t_stats, np.ndarray)

    def test_summary_returns_string(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        s = r.summary()
        assert isinstance(s, str)
        assert "ATT" in s


# ---------------------------------------------------------------------------
# Weight types
# ---------------------------------------------------------------------------


class TestWeightTypes:
    """Test that all 3 weight types work correctly."""

    def test_rademacher(self, cross_section_data):
        y, treatment, cluster_ids, _ = cross_section_data
        r = wild_cluster_bootstrap(
            y, treatment, cluster_ids, weight_type="rademacher", seed=1, n_reps=199
        )
        assert r.weight_type == "rademacher"
        assert 0.0 <= r.pvalue <= 1.0

    def test_mammen(self, cross_section_data):
        y, treatment, cluster_ids, _ = cross_section_data
        r = wild_cluster_bootstrap(
            y, treatment, cluster_ids, weight_type="mammen", seed=1, n_reps=199
        )
        assert r.weight_type == "mammen"
        assert 0.0 <= r.pvalue <= 1.0

    def test_webb(self, cross_section_data):
        y, treatment, cluster_ids, _ = cross_section_data
        r = wild_cluster_bootstrap(
            y, treatment, cluster_ids, weight_type="webb", seed=1, n_reps=199
        )
        assert r.weight_type == "webb"
        assert 0.0 <= r.pvalue <= 1.0

    def test_invalid_weight_type_raises(self, cross_section_data):
        y, treatment, cluster_ids, _ = cross_section_data
        with pytest.raises(ValueError, match="Unknown weight_type"):
            wild_cluster_bootstrap(y, treatment, cluster_ids, weight_type="invalid")


# ---------------------------------------------------------------------------
# P-value and SE properties
# ---------------------------------------------------------------------------


class TestStatisticalProperties:
    """Test p-value range and SE positivity."""

    def test_pvalue_in_0_1(self, cross_section_data):
        y, treatment, cluster_ids, _ = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=42, n_reps=499)
        assert 0.0 <= r.pvalue <= 1.0

    def test_se_positive(self, cross_section_data):
        y, treatment, cluster_ids, _ = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=42, n_reps=499)
        assert r.se_bootstrap > 0

    def test_with_controls(self, cross_section_data):
        y, treatment, cluster_ids, controls = cross_section_data
        r = wild_cluster_bootstrap(
            y, treatment, cluster_ids, controls=controls, seed=42, n_reps=199
        )
        assert 0.0 <= r.pvalue <= 1.0
        assert r.se_bootstrap > 0


# ---------------------------------------------------------------------------
# Full enumeration
# ---------------------------------------------------------------------------


class TestFullEnumeration:
    """Test full enumeration with few clusters (G=5)."""

    def test_full_enumeration_g5(self):
        """With G=5, full enumeration should use 2^5=32 reps."""
        rng = np.random.default_rng(99)
        y = np.concatenate([rng.normal(3, 0.5, 10), rng.normal(0, 0.5, 40)])
        treatment = np.array([1.0] * 10 + [0.0] * 40)
        cluster_ids = np.repeat(np.arange(5), 10)

        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, full_enumeration=True)
        assert r.n_reps == 2**5
        assert r.n_clusters == 5

    def test_full_enumeration_deterministic(self):
        """Full enumeration should give same result every time."""
        rng = np.random.default_rng(99)
        y = np.concatenate([rng.normal(3, 0.5, 10), rng.normal(0, 0.5, 40)])
        treatment = np.array([1.0] * 10 + [0.0] * 40)
        cluster_ids = np.repeat(np.arange(5), 10)

        r1 = wild_cluster_bootstrap(y, treatment, cluster_ids, full_enumeration=True)
        r2 = wild_cluster_bootstrap(y, treatment, cluster_ids, full_enumeration=True)
        assert r1.pvalue == r2.pvalue


# ---------------------------------------------------------------------------
# n_reps matches t_stats length
# ---------------------------------------------------------------------------


class TestNRepsConsistency:
    """Test that n_reps matches the t_stats array length."""

    def test_n_reps_matches_t_stats_length(self, cross_section_data):
        y, treatment, cluster_ids, _ = cross_section_data
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=199)
        assert len(r.t_stats) == r.n_reps

    def test_full_enum_n_reps_matches(self):
        rng = np.random.default_rng(10)
        y = np.concatenate([rng.normal(2, 1, 10), rng.normal(0, 1, 40)])
        treatment = np.array([1.0] * 10 + [0.0] * 40)
        cluster_ids = np.repeat(np.arange(5), 10)
        r = wild_cluster_bootstrap(y, treatment, cluster_ids, full_enumeration=True)
        assert len(r.t_stats) == r.n_reps


# ---------------------------------------------------------------------------
# Numerical stability with extreme data
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Test behaviour with extreme data."""

    def test_extreme_large_values(self):
        """Bootstrap should handle very large outcome values."""
        rng = np.random.default_rng(7)
        y = np.concatenate([rng.normal(1e6, 1e4, 15), rng.normal(0, 1e4, 45)])
        treatment = np.array([1.0] * 15 + [0.0] * 45)
        cluster_ids = np.repeat(np.arange(12), 5)

        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=199)
        assert np.isfinite(r.att)
        assert 0.0 <= r.pvalue <= 1.0

    def test_near_zero_variation(self):
        """If outcome has near-zero variation within groups, should still return."""
        rng = np.random.default_rng(3)
        # Very tight distribution
        y = np.concatenate(
            [
                rng.normal(5, 1e-8, 15),
                rng.normal(0, 1e-8, 45),
            ]
        )
        treatment = np.array([1.0] * 15 + [0.0] * 45)
        cluster_ids = np.repeat(np.arange(12), 5)

        r = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=0, n_reps=99)
        # Should produce a result without raising
        assert isinstance(r, WildClusterBootstrapResult)


class TestResultsConvenienceMethods:
    """Test LWDiDResults.wild_cluster_bootstrap() and .randomization_test() wrappers."""

    def test_results_wild_cluster_bootstrap(self):
        """Convenience method delegates correctly."""
        import numpy as np

        from diff_diff import LWDiD

        rng = np.random.default_rng(42)
        n = 60
        records = []
        for i in range(n):
            d = int(i < 20)
            for t in range(1, 7):
                y = 1.0 + 0.1 * t + rng.normal(0, 0.3)
                if d and t > 3:
                    y += 2.0
                records.append({"unit": i, "time": t, "y": y, "treat": d * int(t > 3)})
        df = pd.DataFrame(records)

        res = LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

        # Build cross-section for bootstrap test
        y_cs = rng.normal(2, 0.5, 20).tolist() + rng.normal(0, 0.5, 40).tolist()
        y_arr = np.array(y_cs)
        d_arr = np.array([1.0] * 20 + [0.0] * 40)
        c_arr = np.repeat(np.arange(12), 5)

        wcb = res.wild_cluster_bootstrap(y_arr, d_arr, c_arr, n_reps=99, seed=42)
        assert np.isfinite(wcb.att)
        assert np.isfinite(wcb.pvalue)
        assert 0 <= wcb.pvalue <= 1

    def test_results_randomization_test(self):
        """Convenience method delegates correctly."""
        import numpy as np

        from diff_diff import LWDiD

        rng = np.random.default_rng(42)
        n = 60
        records = []
        for i in range(n):
            d = int(i < 20)
            for t in range(1, 7):
                y = 1.0 + 0.1 * t + rng.normal(0, 0.3)
                if d and t > 3:
                    y += 2.0
                records.append({"unit": i, "time": t, "y": y, "treat": d * int(t > 3)})
        df = pd.DataFrame(records)

        res = LWDiD().fit(df, outcome="y", unit="unit", time="time", treatment="treat")

        y_arr = np.concatenate([rng.normal(2, 0.5, 20), rng.normal(0, 0.5, 40)])
        d_arr = np.array([1.0] * 20 + [0.0] * 40)

        ri = res.randomization_test(y_arr, d_arr, n_reps=199, seed=42)
        assert np.isfinite(ri.pvalue)
        assert 0 <= ri.pvalue <= 1
