"""
Tests for diagnostics module (placebo tests).

Tests the placebo test functions for validating DiD assumptions.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.diagnostics import (
    PlaceboTestResults,
    leave_one_out_test,
    permutation_test,
    placebo_group_test,
    placebo_timing_test,
    run_all_placebo_tests,
    run_placebo_test,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def panel_data_parallel_trends():
    """Create panel data where parallel trends holds."""
    np.random.seed(42)

    n_units = 50
    n_treated = 25
    n_periods = 6  # Periods 0-2 are pre-treatment, 3-5 are post-treatment

    data = []
    for unit in range(n_units):
        is_treated = unit < n_treated
        unit_effect = np.random.normal(0, 1)

        for period in range(n_periods):
            y = 10.0
            y += unit_effect
            y += period * 0.5  # Common time trend

            # Treatment effect only in post periods
            if is_treated and period >= 3:
                y += 3.0  # True ATT = 3.0

            y += np.random.normal(0, 0.5)

            data.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": int(period >= 3),
                "outcome": y,
            })

    return pd.DataFrame(data)


@pytest.fixture
def panel_data_violated_trends():
    """Create panel data where parallel trends is violated."""
    np.random.seed(42)

    n_units = 50
    n_treated = 25
    n_periods = 6

    data = []
    for unit in range(n_units):
        is_treated = unit < n_treated
        unit_effect = np.random.normal(0, 1)

        for period in range(n_periods):
            y = 10.0
            y += unit_effect
            y += period * 0.5  # Common time trend

            # Differential pre-trend for treated (violation!)
            if is_treated:
                y += period * 0.8  # Treated growing faster

            # Treatment effect only in post periods
            if is_treated and period >= 3:
                y += 3.0

            y += np.random.normal(0, 0.5)

            data.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": int(period >= 3),
                "outcome": y,
            })

    return pd.DataFrame(data)


@pytest.fixture
def simple_panel_data():
    """Simple panel data for basic tests."""
    np.random.seed(42)

    data = []
    for unit in range(20):
        is_treated = unit < 10

        for period in [0, 1, 2, 3]:
            y = 10.0 + np.random.normal(0, 1)

            if period >= 2:
                y += 2.0  # Time effect

            if is_treated and period >= 2:
                y += 5.0  # Treatment effect

            data.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": int(period >= 2),
                "outcome": y,
            })

    return pd.DataFrame(data)


# =============================================================================
# PlaceboTestResults Tests
# =============================================================================


class TestPlaceboTestResults:
    """Tests for PlaceboTestResults dataclass."""

    def test_summary_format(self):
        """Test summary produces readable output."""
        results = PlaceboTestResults(
            test_type="fake_timing",
            placebo_effect=0.5,
            se=0.2,
            t_stat=2.5,
            p_value=0.02,
            conf_int=(0.1, 0.9),
            n_obs=100,
            is_significant=True,
        )

        summary = results.summary()

        assert "Placebo Test Results" in summary
        assert "fake_timing" in summary
        assert "0.5" in summary
        assert "WARNING" in summary  # Because significant

    def test_summary_not_significant(self):
        """Test summary for non-significant result."""
        results = PlaceboTestResults(
            test_type="permutation",
            placebo_effect=0.1,
            se=0.5,
            t_stat=0.2,
            p_value=0.85,
            conf_int=(-0.9, 1.1),
            n_obs=100,
            is_significant=False,
        )

        summary = results.summary()

        assert "No significant placebo effect" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = PlaceboTestResults(
            test_type="fake_timing",
            placebo_effect=0.5,
            se=0.2,
            t_stat=2.5,
            p_value=0.02,
            conf_int=(0.1, 0.9),
            n_obs=100,
            is_significant=True,
            original_effect=3.0,
        )

        d = results.to_dict()

        assert d["test_type"] == "fake_timing"
        assert d["placebo_effect"] == 0.5
        assert d["original_effect"] == 3.0

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        results = PlaceboTestResults(
            test_type="fake_timing",
            placebo_effect=0.5,
            se=0.2,
            t_stat=2.5,
            p_value=0.02,
            conf_int=(0.1, 0.9),
            n_obs=100,
            is_significant=True,
        )

        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_significance_stars(self):
        """Test significance stars property."""
        # Very significant
        r1 = PlaceboTestResults(
            test_type="test", placebo_effect=0, se=1, t_stat=0,
            p_value=0.0001, conf_int=(0, 0), n_obs=100, is_significant=True
        )
        assert r1.significance_stars == "***"

        # Not significant
        r2 = PlaceboTestResults(
            test_type="test", placebo_effect=0, se=1, t_stat=0,
            p_value=0.5, conf_int=(0, 0), n_obs=100, is_significant=False
        )
        assert r2.significance_stars == ""


# =============================================================================
# Fake Timing Test
# =============================================================================


class TestPlaceboTimingTest:
    """Tests for fake timing placebo test."""

    def test_basic_fake_timing(self, simple_panel_data):
        """Test basic fake timing test."""
        results = placebo_timing_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            fake_treatment_period=1,
            post_periods=[2, 3],
        )

        assert isinstance(results, PlaceboTestResults)
        assert results.test_type == "fake_timing"
        assert results.fake_period == 1

    def test_fake_timing_no_effect_when_parallel(self, panel_data_parallel_trends):
        """Should find no significant effect when parallel trends holds."""
        results = placebo_timing_test(
            panel_data_parallel_trends,
            outcome="outcome",
            treatment="treated",
            time="period",
            fake_treatment_period=1,
            post_periods=[3, 4, 5],
        )

        # With parallel trends, placebo effect should not be significant
        # Using a lenient threshold since this is statistical
        assert results.p_value > 0.01 or abs(results.placebo_effect) < 1.0

    def test_fake_timing_detects_violation(self, panel_data_violated_trends):
        """Should detect effect when parallel trends violated."""
        results = placebo_timing_test(
            panel_data_violated_trends,
            outcome="outcome",
            treatment="treated",
            time="period",
            fake_treatment_period=1,
            post_periods=[3, 4, 5],
        )

        # With violated trends, should see larger placebo effect
        # The differential trend should create a detectable effect
        assert abs(results.placebo_effect) > 0.3

    def test_fake_timing_invalid_period_raises(self, simple_panel_data):
        """Test error when fake period is a post period."""
        with pytest.raises(ValueError, match="must be a pre-treatment period"):
            placebo_timing_test(
                simple_panel_data,
                outcome="outcome",
                treatment="treated",
                time="period",
                fake_treatment_period=3,  # This is a post period
                post_periods=[2, 3],
            )

    def test_fake_timing_returns_original_effect(self, simple_panel_data):
        """Test that original effect is included."""
        results = placebo_timing_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            fake_treatment_period=1,
            post_periods=[2, 3],
        )

        assert results.original_effect is not None
        assert results.original_se is not None


# =============================================================================
# Fake Group Test
# =============================================================================


class TestPlaceboGroupTest:
    """Tests for fake group placebo test."""

    def test_basic_fake_group(self, simple_panel_data):
        """Test basic fake group test."""
        # Use some control units as fake treated
        control_units = simple_panel_data[
            simple_panel_data["treated"] == 0
        ]["unit"].unique()
        fake_treated = list(control_units[:3])

        results = placebo_group_test(
            simple_panel_data,
            outcome="outcome",
            time="period",
            unit="unit",
            fake_treated_units=fake_treated,
            post_periods=[2, 3],
        )

        assert isinstance(results, PlaceboTestResults)
        assert results.test_type == "fake_group"
        assert results.fake_group == fake_treated

    def test_fake_group_empty_raises(self, simple_panel_data):
        """Test error when fake group is empty."""
        with pytest.raises(ValueError, match="non-empty list"):
            placebo_group_test(
                simple_panel_data,
                outcome="outcome",
                time="period",
                unit="unit",
                fake_treated_units=[],
                post_periods=[2, 3],
            )


# =============================================================================
# Permutation Test
# =============================================================================


class TestPermutationTest:
    """Tests for permutation inference."""

    def test_basic_permutation(self, simple_panel_data):
        """Test basic permutation test."""
        results = permutation_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            n_permutations=50,
            seed=42,
        )

        assert isinstance(results, PlaceboTestResults)
        assert results.test_type == "permutation"
        assert results.n_permutations == 50

    def test_permutation_p_value_range(self, simple_panel_data):
        """P-value should be in (0, 1]."""
        results = permutation_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            n_permutations=50,
            seed=42,
        )

        assert 0 < results.p_value <= 1

    def test_permutation_null_distribution(self, simple_panel_data):
        """Null distribution should be approximately centered at zero."""
        results = permutation_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            n_permutations=100,
            seed=42,
        )

        # Mean of null distribution should be close to zero
        # (but not exactly, due to finite permutations)
        assert abs(np.mean(results.permutation_distribution)) < 2.0

    def test_permutation_reproducibility(self, simple_panel_data):
        """Same seed gives same results."""
        results1 = permutation_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            n_permutations=50,
            seed=42,
        )

        results2 = permutation_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            n_permutations=50,
            seed=42,
        )

        assert results1.p_value == results2.p_value
        np.testing.assert_array_equal(
            results1.permutation_distribution,
            results2.permutation_distribution
        )

    def test_permutation_detects_true_effect(self, simple_panel_data):
        """Should detect when true effect exists."""
        results = permutation_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            n_permutations=100,
            seed=42,
        )

        # Original effect should be outside the bulk of null distribution
        assert results.original_effect is not None
        assert abs(results.original_effect) > 3.0  # True effect is 5.0


# =============================================================================
# Leave-One-Out Test
# =============================================================================


class TestLeaveOneOutTest:
    """Tests for leave-one-out sensitivity."""

    def test_basic_leave_one_out(self, simple_panel_data):
        """Test basic LOO test."""
        results = leave_one_out_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
        )

        assert isinstance(results, PlaceboTestResults)
        assert results.test_type == "leave_one_out"
        assert results.leave_one_out_effects is not None

    def test_loo_returns_all_treated_units(self, simple_panel_data):
        """Should have estimate for each treated unit."""
        results = leave_one_out_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
        )

        # Get number of treated units
        n_treated = simple_panel_data[
            simple_panel_data["treated"] == 1
        ]["unit"].nunique()

        assert len(results.leave_one_out_effects) == n_treated

    def test_loo_detects_influential_unit(self):
        """Should detect when one unit drives results."""
        np.random.seed(42)

        # Create data where one unit has extreme effect
        data = []
        for unit in range(10):
            is_treated = unit < 5

            for period in [0, 1]:
                y = 10.0 + np.random.normal(0, 0.5)

                if period == 1:
                    y += 1.0

                if is_treated and period == 1:
                    if unit == 0:
                        y += 20.0  # Extreme influential unit
                    else:
                        y += 2.0  # Normal treatment effect

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": period,
                    "outcome": y,
                })

        df = pd.DataFrame(data)

        results = leave_one_out_test(
            df,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
        )

        effects = list(results.leave_one_out_effects.values())

        # Dropping unit 0 should significantly change the estimate
        # High variance in LOO effects indicates influential unit
        assert np.std(effects) > 1.0

    def test_loo_summary_shows_stats(self, simple_panel_data):
        """Test that summary shows LOO statistics."""
        results = leave_one_out_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
        )

        summary = results.summary()

        assert "Leave-One-Out Summary" in summary
        assert "Units analyzed" in summary
        assert "Mean effect" in summary


# =============================================================================
# run_placebo_test dispatcher
# =============================================================================


class TestRunPlaceboTest:
    """Tests for run_placebo_test dispatcher."""

    def test_dispatch_fake_timing(self, simple_panel_data):
        """Test dispatch to fake timing test."""
        results = run_placebo_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            test_type="fake_timing",
            fake_treatment_period=1,
            post_periods=[2, 3],
        )

        assert results.test_type == "fake_timing"

    def test_dispatch_permutation(self, simple_panel_data):
        """Test dispatch to permutation test."""
        results = run_placebo_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            test_type="permutation",
            n_permutations=50,
            seed=42,
        )

        assert results.test_type == "permutation"

    def test_dispatch_leave_one_out(self, simple_panel_data):
        """Test dispatch to LOO test."""
        results = run_placebo_test(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            test_type="leave_one_out",
        )

        assert results.test_type == "leave_one_out"

    def test_invalid_test_type_raises(self, simple_panel_data):
        """Test error on invalid test type."""
        with pytest.raises(ValueError, match="test_type must be one of"):
            run_placebo_test(
                simple_panel_data,
                outcome="outcome",
                treatment="treated",
                time="post",
                test_type="invalid_type",
            )

    def test_permutation_requires_unit(self, simple_panel_data):
        """Test that permutation requires unit."""
        with pytest.raises(ValueError, match="unit is required"):
            run_placebo_test(
                simple_panel_data,
                outcome="outcome",
                treatment="treated",
                time="post",
                test_type="permutation",
            )


# =============================================================================
# run_all_placebo_tests
# =============================================================================


class TestRunAllPlaceboTests:
    """Tests for comprehensive placebo suite."""

    def test_runs_all_tests(self, panel_data_parallel_trends):
        """Should run fake timing, permutation, and LOO."""
        results = run_all_placebo_tests(
            panel_data_parallel_trends,
            outcome="outcome",
            treatment="treated",
            time="period",
            unit="unit",
            pre_periods=[0, 1, 2],
            post_periods=[3, 4, 5],
            n_permutations=50,
            seed=42,
        )

        # Should have fake timing tests for periods 1 and 2
        assert "fake_timing_1" in results
        assert "fake_timing_2" in results

        # Should have permutation and LOO tests
        assert "permutation" in results
        assert "leave_one_out" in results

    def test_returns_dict_structure(self, simple_panel_data):
        """Should return properly structured dict."""
        results = run_all_placebo_tests(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            unit="unit",
            pre_periods=[0, 1],
            post_periods=[2, 3],
            n_permutations=20,
            seed=42,
        )

        assert isinstance(results, dict)

        # Check that each result is either PlaceboTestResults or error dict
        for key, value in results.items():
            assert isinstance(value, (PlaceboTestResults, dict))
