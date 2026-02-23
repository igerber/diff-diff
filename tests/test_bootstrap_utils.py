"""Tests for bootstrap utility edge cases (NaN propagation)."""

import numpy as np
import pytest

from diff_diff.bootstrap_utils import compute_effect_bootstrap_stats


class TestBootstrapStatsNaNPropagation:
    """Regression tests for compute_effect_bootstrap_stats NaN guard."""

    def test_bootstrap_stats_single_valid_sample(self):
        """Single valid sample: ddof=1 produces NaN SE -> all NaN."""
        boot_dist = np.array([1.5])
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=1.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_all_nonfinite(self):
        """All non-finite samples: fails 50% validity check -> all NaN."""
        boot_dist = np.array([np.nan, np.nan, np.inf])
        with pytest.warns(RuntimeWarning):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=1.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_identical_values(self):
        """All identical values: se=0 -> all NaN."""
        boot_dist = np.array([2.0] * 100)
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=2.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_mostly_valid_but_identical(self):
        """67% valid (passes 50% check) but identical values: se=0 -> all NaN."""
        boot_dist = np.array([2.0, 2.0, np.nan])
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=2.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_original_effect_with_finite_boot_dist(self, bad_value):
        """Non-finite original_effect must return all-NaN even with finite boot_dist."""
        boot_dist = np.arange(100.0)
        se, ci, p_value = compute_effect_bootstrap_stats(
            original_effect=bad_value, boot_dist=boot_dist
        )
        assert np.isnan(se)
        assert np.isnan(ci[0]) and np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_normal_case(self):
        """Normal case with varied values: all fields finite."""
        boot_dist = np.arange(100.0)
        se, ci, p_value = compute_effect_bootstrap_stats(
            original_effect=50.0, boot_dist=boot_dist
        )
        assert np.isfinite(se)
        assert se > 0
        assert np.isfinite(ci[0])
        assert np.isfinite(ci[1])
        assert ci[0] < ci[1]
        assert np.isfinite(p_value)
        assert 0 < p_value <= 1
