"""Tests for lwdid_randomization module."""

import numpy as np
import pytest

from diff_diff.lwdid_randomization import (
    _compute_pvalue,
    randomization_inference,
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
# Result fields
# ---------------------------------------------------------------------------


class TestRandomizationResultFields:
    """Test that RandomizationResult has all expected fields."""

    def test_result_fields_present(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, n_reps=200, seed=0)
        assert hasattr(r, "pvalue")
        assert hasattr(r, "att_observed")
        assert hasattr(r, "att_distribution")
        assert hasattr(r, "n_reps")
        assert hasattr(r, "n_valid")
        assert hasattr(r, "n_failed")
        assert hasattr(r, "failure_rate")
        assert hasattr(r, "method")
        assert hasattr(r, "seed")

    def test_result_types(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, n_reps=200, seed=0)
        assert isinstance(r.pvalue, float)
        assert isinstance(r.att_observed, float)
        assert isinstance(r.att_distribution, np.ndarray)
        assert isinstance(r.n_reps, int)
        assert isinstance(r.n_valid, int)
        assert isinstance(r.n_failed, int)
        assert isinstance(r.failure_rate, float)
        assert isinstance(r.method, str)


# ---------------------------------------------------------------------------
# Permutation preserves N_treated
# ---------------------------------------------------------------------------


class TestPermutationPreservation:
    """Permutation should preserve number of treated units."""

    def test_permutation_preserves_n_treated(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, method="permutation", n_reps=500, seed=0)
        # With permutation, no draws are degenerate
        assert r.n_failed == 0
        assert r.failure_rate == 0.0

    def test_bootstrap_method_removed(self, cross_section_data):
        # Fix-wave review finding: label resampling WITH replacement is not
        # Fisher randomization inference; the mode is removed.
        y, treatment, *_ = cross_section_data
        with pytest.raises(ValueError, match="method='bootstrap' has been removed"):
            randomization_inference(y, treatment, method="bootstrap", n_reps=100, seed=0)

    def test_pvalue_in_0_1_permutation(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, method="permutation", n_reps=500, seed=42)
        assert 0.0 <= r.pvalue <= 1.0

    def test_clear_treatment_effect_detected(self, cross_section_data):
        """With a clear treatment effect, p-value should be small."""
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, method="permutation", n_reps=999, seed=0)
        assert r.pvalue < 0.05


# ---------------------------------------------------------------------------
# With and without controls
# ---------------------------------------------------------------------------


class TestControls:
    """Test with and without control variables."""

    def test_without_controls(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r = randomization_inference(y, treatment, n_reps=200, seed=0)
        assert np.isfinite(r.att_observed)
        assert r.n_valid > 0

    def test_with_controls(self, cross_section_data):
        y, treatment, _, controls = cross_section_data
        r = randomization_inference(y, treatment, controls=controls, n_reps=200, seed=0)
        assert np.isfinite(r.att_observed)
        assert r.n_valid > 0


# ---------------------------------------------------------------------------
# Degenerate data handling
# ---------------------------------------------------------------------------


class TestDegenerateData:
    """Test handling of degenerate inputs."""

    def test_all_treated_raises(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([1.0, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            randomization_inference(y, treatment, n_reps=100)

    def test_all_control_raises(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            randomization_inference(y, treatment, n_reps=100)

    def test_too_small_sample_raises(self):
        y = np.array([1.0, 2.0])
        treatment = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            randomization_inference(y, treatment, n_reps=100)

    def test_invalid_method_raises(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        with pytest.raises(ValueError):
            randomization_inference(y, treatment, method="invalid", n_reps=100)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Test that seed produces reproducible results."""

    def test_same_seed_same_result(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r1 = randomization_inference(y, treatment, n_reps=200, seed=123)
        r2 = randomization_inference(y, treatment, n_reps=200, seed=123)
        assert r1.pvalue == r2.pvalue
        np.testing.assert_array_equal(r1.att_distribution, r2.att_distribution)

    def test_different_seed_different_result(self, cross_section_data):
        y, treatment, _, _ = cross_section_data
        r1 = randomization_inference(y, treatment, n_reps=200, seed=1)
        r2 = randomization_inference(y, treatment, n_reps=200, seed=2)
        # Distributions should differ (extremely unlikely to be equal)
        assert not np.array_equal(r1.att_distribution, r2.att_distribution)


# ---------------------------------------------------------------------------
# Tie handling ('at least as extreme' convention)
# ---------------------------------------------------------------------------


class TestTieHandling:
    """Ties must count as 'at least as extreme' (>=), not strictly greater."""

    def test_constant_outcome_all_ties_pvalue_is_one(self):
        """Constant outcome: every permutation ATT ties with the observed
        ATT (all zero), so the two-sided p-value must be exactly 1.0."""
        y = np.full(40, 3.0)
        treatment = np.array([1.0] * 15 + [0.0] * 25)
        r = randomization_inference(y, treatment, method="permutation", n_reps=999, seed=0)
        assert r.pvalue == 1.0

    def test_compute_pvalue_full_tie_distribution(self):
        """All replications tied with the observed statistic -> p == 1.0."""
        att_dist = np.zeros(999)
        pvalue, n_valid, n_failed = _compute_pvalue(att_dist, att_obs=0.0)
        assert pvalue == 1.0
        assert n_valid == 999
        assert n_failed == 0

    def test_compute_pvalue_half_tie_distribution(self):
        """Half the replications tie in absolute value, the rest are less
        extreme: p = (n_tied + 1) / (n_valid + 1) under the >= rule."""
        att_dist = np.concatenate([np.full(50, 1.0), np.full(49, 0.0)])
        pvalue, n_valid, _ = _compute_pvalue(att_dist, att_obs=-1.0)
        assert n_valid == 99
        assert pvalue == pytest.approx((50 + 1) / (99 + 1))

    def test_discrete_outcome_pvalue_near_theoretical(self):
        """Binary outcome with a coarse permutation distribution: the exact
        randomization p-value is 1/3 (2 of 6 assignments are at least as
        extreme), so the Monte Carlo p should be close to that."""
        y = np.array([1.0, 1.0, 0.0, 0.0])
        treatment = np.array([1.0, 1.0, 0.0, 0.0])
        r = randomization_inference(y, treatment, method="permutation", n_reps=999, seed=42)
        assert abs(r.pvalue - 1.0 / 3.0) < 0.05
