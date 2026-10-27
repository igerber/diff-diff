"""
Tests for Honest DiD sensitivity analysis module.

Tests the implementation of Rambachan & Roth (2023) methods for
robust inference in difference-in-differences under violations
of parallel trends.
"""

import numpy as np
import pandas as pd
import pytest
from event_study_surface_builders import calendar_surface
from legacy_surface import legacy_standin

from diff_diff import TwoWayFixedEffects
from diff_diff.honest_did import (
    DeltaRM,
    DeltaSD,
    DeltaSDRM,
    HonestDiD,
    HonestDiDResults,
    SensitivityResults,
    _compute_flci,
    _construct_A_sd,
    _construct_constraints_rm_component,
    _construct_constraints_sd,
    _extract_event_study_params,
    compute_honest_did,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_panel_data():
    """Generate simple panel data for testing."""
    np.random.seed(42)
    n_units = 100
    n_periods = 8
    treatment_time = 4
    true_att = 5.0

    data = []
    for unit in range(n_units):
        is_treated = unit < n_units // 2
        unit_effect = np.random.normal(0, 2)

        for period in range(n_periods):
            time_effect = period * 1.0
            y = 10.0 + unit_effect + time_effect

            post = period >= treatment_time
            if is_treated and post:
                y += true_att

            y += np.random.normal(0, 0.5)

            data.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": int(post),
                    "outcome": y,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def mock_event_study_results():
    """The same event study as a hand-built calendar ``EventStudyResults`` - what
    ``TwoWayFixedEffects().fit(..., event_study=True)`` emits (4.0 cut, PR 4c-i):
    pre 0/1/2, reference 3, post 4-7; the event-study vcov is the 7x7 diagonal
    sub-block (the former MPD mock spread it into a 20x20 full vcov)."""
    rows = {
        0: (0.05, 0.4, 0.125, 0.90, (-0.73, 0.83)),
        1: (-0.02, 0.35, -0.057, 0.95, (-0.71, 0.67)),
        2: (0.08, 0.3, 0.267, 0.79, (-0.51, 0.67)),
        4: (5.0, 0.5, 10.0, 0.0001, (4.02, 5.98)),
        5: (5.2, 0.5, 10.4, 0.0001, (4.22, 6.18)),
        6: (4.8, 0.5, 9.6, 0.0001, (3.82, 5.78)),
        7: (5.0, 0.5, 10.0, 0.0001, (4.02, 5.98)),
    }
    vcov_diag = [0.4**2, 0.35**2, 0.3**2, 0.5**2, 0.5**2, 0.5**2, 0.5**2]
    return calendar_surface(rows, reference=3, post_periods=[4, 5, 6, 7], vcov=np.diag(vcov_diag))


# =============================================================================
# Tests for Delta Restriction Classes
# =============================================================================


class TestDeltaClasses:
    """Tests for Delta restriction dataclasses."""

    def test_delta_sd_creation(self):
        """Test DeltaSD creation."""
        delta = DeltaSD(M=0.5)
        assert delta.M == 0.5
        assert repr(delta) == "DeltaSD(M=0.5)"

    def test_delta_sd_default(self):
        """Test DeltaSD default value."""
        delta = DeltaSD()
        assert delta.M == 0.0

    def test_delta_sd_negative_raises(self):
        """Test that negative M raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            DeltaSD(M=-1.0)

    def test_delta_rm_creation(self):
        """Test DeltaRM creation."""
        delta = DeltaRM(Mbar=1.5)
        assert delta.Mbar == 1.5
        assert repr(delta) == "DeltaRM(Mbar=1.5)"

    def test_delta_rm_default(self):
        """Test DeltaRM default value."""
        delta = DeltaRM()
        assert delta.Mbar == 1.0

    def test_delta_rm_negative_raises(self):
        """Test that negative Mbar raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            DeltaRM(Mbar=-0.5)

    def test_delta_sdrm_creation(self):
        """Test DeltaSDRM (combined) creation."""
        delta = DeltaSDRM(M=0.3, Mbar=1.2)
        assert delta.M == 0.3
        assert delta.Mbar == 1.2
        assert repr(delta) == "DeltaSDRM(M=0.3, Mbar=1.2)"


# =============================================================================
# Tests for Constraint Matrix Construction
# =============================================================================


class TestConstraintConstruction:
    """Tests for constraint matrix construction."""

    def test_construct_A_sd_basic(self):
        """Test smoothness constraint matrix with delta_0=0 boundary."""
        # 3 pre + 2 post: should have T+Tbar-1 = 4 rows
        A = _construct_A_sd(3, 2)
        assert A.shape == (4, 5)

        # First row: pure pre second difference [1, -2, 1, 0, 0]
        np.testing.assert_array_equal(A[0], [1, -2, 1, 0, 0])
        # Boundary at t=-1: [0, 1, -2, 0, 0] (uses delta_0=0)
        np.testing.assert_array_equal(A[1], [0, 1, -2, 0, 0])
        # Bridge at t=0: [0, 0, 1, 1, 0] (delta_{-1} + delta_1)
        np.testing.assert_array_equal(A[2], [0, 0, 1, 1, 0])
        # Boundary at t=1: [0, 0, 0, -2, 1] (uses delta_0=0)
        np.testing.assert_array_equal(A[3], [0, 0, 0, -2, 1])

    def test_construct_A_sd_small(self):
        """Test that 1+1 returns bridge row only."""
        A = _construct_A_sd(1, 1)
        assert A.shape == (1, 2)
        np.testing.assert_array_equal(A[0], [1, 1])

    def test_construct_constraints_sd(self):
        """Test smoothness constraints."""
        A_ineq, b_ineq = _construct_constraints_sd(num_pre_periods=3, num_post_periods=4, M=0.5)

        # Should have 2 * (T+Tbar-1) = 2 * 6 = 12 constraints
        assert A_ineq.shape[0] == 12
        assert A_ineq.shape[1] == 7
        assert np.all(b_ineq == 0.5)

    def test_construct_constraints_rm_component(self):
        """Test relative magnitudes first-difference constraints."""
        A_ineq, b_ineq = _construct_constraints_rm_component(
            num_pre_periods=3, num_post_periods=4, Mbar=1.5, max_pre_first_diff=0.2
        )

        # Should have 2 * 4 = 8 constraints (pos/neg for each post first-diff)
        assert A_ineq.shape[0] == 8
        assert A_ineq.shape[1] == 7
        assert np.all(b_ineq == 1.5 * 0.2)


# =============================================================================
# Tests for Confidence Interval Methods
# =============================================================================


class TestCIMethods:
    """Tests for confidence interval computation methods."""

    def test_flci_symmetric(self):
        """Test FLCI produces symmetric extension of bounds."""
        lb, ub = 1.0, 2.0
        se = 0.5
        alpha = 0.05

        ci_lb, ci_ub = _compute_flci(lb, ub, se, alpha)

        # FLCI extends each side by z * se
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        expected_ci_lb = lb - z * se
        expected_ci_ub = ub + z * se

        assert ci_lb == pytest.approx(expected_ci_lb)
        assert ci_ub == pytest.approx(expected_ci_ub)

    def test_flci_point_identified(self):
        """Test FLCI when lb == ub (point identified)."""
        point = 5.0
        se = 0.5
        alpha = 0.05

        ci_lb, ci_ub = _compute_flci(point, point, se, alpha)

        # Should be standard CI
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        assert ci_lb == pytest.approx(point - z * se)
        assert ci_ub == pytest.approx(point + z * se)


# =============================================================================
# Tests for Parameter Extraction
# =============================================================================


class TestParameterExtraction:
    """Tests for extracting parameters from results objects."""

    def test_extract_unsupported_type_raises(self):
        """Test that unsupported types raise TypeError, naming the container route."""
        with pytest.raises(TypeError, match="Unsupported results type") as exc_info:
            _extract_event_study_params("not a results object")
        # The expected-types list names the post-fit container route too.
        assert "EventStudyResults" in str(exc_info.value)


# =============================================================================
# Tests for HonestDiD Main Class
# =============================================================================


class TestHonestDiD:
    """Tests for the main HonestDiD class."""

    def test_init_defaults(self):
        """Test default initialization."""
        honest = HonestDiD()
        assert honest.method == "relative_magnitude"
        assert honest.M == 1.0
        assert honest.alpha == 0.05
        assert honest.l_vec is None

    def test_init_smoothness(self):
        """Test initialization with smoothness method."""
        honest = HonestDiD(method="smoothness")
        assert honest.method == "smoothness"
        assert honest.M == 0.0  # Default for smoothness

    def test_init_custom_M(self):
        """Test initialization with custom M."""
        honest = HonestDiD(method="relative_magnitude", M=2.0)
        assert honest.M == 2.0

    def test_init_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            HonestDiD(method="invalid")

    def test_init_negative_M_raises(self):
        """Test that negative M raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            HonestDiD(M=-1.0)

    def test_init_invalid_alpha_raises(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            HonestDiD(alpha=1.5)

    def test_get_params(self):
        """Test get_params method."""
        honest = HonestDiD(method="smoothness", M=0.5, alpha=0.1)
        params = honest.get_params()

        assert params["method"] == "smoothness"
        assert params["M"] == 0.5
        assert params["alpha"] == 0.1

    def test_set_params(self):
        """Test set_params method."""
        honest = HonestDiD()
        honest.set_params(M=2.0, alpha=0.1)

        assert honest.M == 2.0
        assert honest.alpha == 0.1

    def test_fit_returns_results(self, mock_event_study_results):
        """Test that fit returns HonestDiDResults."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_event_study_results)

        assert isinstance(results, HonestDiDResults)
        assert results.M == 1.0
        assert results.method == "relative_magnitude"

    def test_fit_negative_M_override_raises(self, mock_event_study_results):
        """A negative M passed to fit() (bypassing constructor validation) must
        raise, not be silently treated as +|M| by the FLCI cv abs()."""
        honest = HonestDiD(method="smoothness", M=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            honest.fit(mock_event_study_results, M=-0.1)

    def test_sensitivity_analysis_negative_M_grid_raises(self, mock_event_study_results):
        """A negative value in M_grid must raise (bypasses constructor validation)."""
        honest = HonestDiD(method="smoothness")
        with pytest.raises(ValueError, match="non-negative"):
            honest.sensitivity_analysis(mock_event_study_results, M_grid=[0.0, -0.1])

    def test_fit_smoothness(self, mock_event_study_results):
        """Test fit with smoothness method."""
        honest = HonestDiD(method="smoothness", M=0.0)
        results = honest.fit(mock_event_study_results)

        assert isinstance(results, HonestDiDResults)
        assert results.method == "smoothness"
        assert results.ci_method == "FLCI"

    def test_fit_combined(self, mock_event_study_results):
        """Test fit with combined method."""
        honest = HonestDiD(method="combined", M=1.0)
        results = honest.fit(mock_event_study_results)

        assert isinstance(results, HonestDiDResults)
        assert results.method == "combined"

    def test_fit_override_M(self, mock_event_study_results):
        """Test that M can be overridden in fit."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_event_study_results, M=2.0)

        assert results.M == 2.0

    def test_bounds_widen_with_M(self, mock_event_study_results):
        """Test that bounds widen as M increases."""
        honest = HonestDiD(method="relative_magnitude")

        results_small = honest.fit(mock_event_study_results, M=0.5)
        results_large = honest.fit(mock_event_study_results, M=2.0)

        # Larger M should give wider bounds
        assert (
            results_large.ci_ub - results_large.ci_lb >= results_small.ci_ub - results_small.ci_lb
        )


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis functionality."""

    def test_sensitivity_analysis_returns_results(self, mock_event_study_results):
        """Test that sensitivity_analysis returns SensitivityResults."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_event_study_results)

        assert isinstance(sensitivity, SensitivityResults)

    def test_sensitivity_analysis_custom_grid(self, mock_event_study_results):
        """Test sensitivity analysis with custom M grid."""
        honest = HonestDiD(method="relative_magnitude")
        M_grid = [0, 0.5, 1.0, 1.5]
        sensitivity = honest.sensitivity_analysis(mock_event_study_results, M_grid=M_grid)

        assert len(sensitivity.M_values) == 4
        np.testing.assert_array_equal(sensitivity.M_values, M_grid)

    def test_sensitivity_analysis_bounds_list(self, mock_event_study_results):
        """Test that bounds list has correct length."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_event_study_results)

        assert len(sensitivity.bounds) == len(sensitivity.M_values)
        assert len(sensitivity.robust_cis) == len(sensitivity.M_values)


class TestBreakdownValue:
    """Tests for breakdown value computation."""

    def test_breakdown_value_type(self, mock_event_study_results):
        """Test that breakdown_value returns float or None."""
        honest = HonestDiD(method="relative_magnitude")
        breakdown = honest.breakdown_value(mock_event_study_results)

        assert breakdown is None or isinstance(breakdown, float)

    def test_breakdown_value_monotonic(self, mock_event_study_results):
        """Test breakdown value properties."""
        honest = HonestDiD(method="relative_magnitude")
        breakdown = honest.breakdown_value(mock_event_study_results)

        if breakdown is not None and breakdown > 0:
            # Before breakdown, should be significant
            result_before = honest.fit(mock_event_study_results, M=breakdown * 0.9)
            assert result_before.is_significant

            # At/after breakdown, should not be significant
            # We call fit but don't assert on significance since
            # the binary search tolerance may not match exactly
            honest.fit(mock_event_study_results, M=breakdown * 1.1)


# =============================================================================
# Tests for Results Classes
# =============================================================================


class TestHonestDiDResults:
    """Tests for HonestDiDResults dataclass."""

    def test_results_properties(self, mock_event_study_results):
        """Test HonestDiDResults properties."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_event_study_results)

        assert hasattr(results, "lb")
        assert hasattr(results, "ub")
        assert hasattr(results, "ci_lb")
        assert hasattr(results, "ci_ub")
        assert hasattr(results, "is_significant")
        assert hasattr(results, "identified_set_width")
        assert hasattr(results, "ci_width")

    def test_results_is_significant(self, mock_event_study_results):
        """Test is_significant property."""
        honest = HonestDiD(method="relative_magnitude", M=0.0)
        results = honest.fit(mock_event_study_results)

        # With M=0 (parallel trends), should be significant for our data
        # Check that property is a bool
        assert isinstance(results.is_significant, bool)

    def test_results_width_properties(self, mock_event_study_results):
        """Test width properties are non-negative."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_event_study_results)

        assert results.identified_set_width >= 0
        assert results.ci_width >= 0

    def test_results_summary(self, mock_event_study_results):
        """Test summary method produces string."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_event_study_results)

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Honest DiD" in summary
        assert "Rambachan" in summary

    def test_results_to_dict(self, mock_event_study_results):
        """Test to_dict method."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_event_study_results)

        d = results.to_dict()
        assert isinstance(d, dict)
        assert "lb" in d
        assert "ub" in d
        assert "M" in d
        assert "method" in d

    def test_results_to_dataframe(self, mock_event_study_results):
        """Test to_dataframe method."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_event_study_results)

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


class TestSensitivityResults:
    """Tests for SensitivityResults dataclass."""

    def test_sensitivity_results_to_dataframe(self, mock_event_study_results):
        """Test to_dataframe method."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_event_study_results)

        df = sensitivity.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "M" in df.columns
        assert "lb" in df.columns
        assert "ci_lb" in df.columns

    def test_sensitivity_results_summary(self, mock_event_study_results):
        """Test summary method."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_event_study_results)

        summary = sensitivity.summary()
        assert isinstance(summary, str)


# =============================================================================
# Tests for Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_honest_did(self, mock_event_study_results):
        """Test compute_honest_did function."""
        results = compute_honest_did(mock_event_study_results, method="relative_magnitude", M=1.0)

        assert isinstance(results, HonestDiDResults)
        assert results.M == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with real estimators."""

    def test_with_twfe_event_study(self, simple_panel_data):
        """Test full pipeline with the TWFE event-study surface."""
        # Fit event study (TWFE event-study mode, pooled design = the former MPD fit)
        event_results = TwoWayFixedEffects().fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            event_study=True,
            spec="pooled",
            post_periods=[4, 5, 6, 7],
            reference_period=3,
        )

        # Run Honest DiD
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        bounds = honest.fit(event_results)

        # Check results are reasonable
        assert bounds.original_estimate > 0  # True effect is positive
        assert bounds.ci_ub > bounds.ci_lb
        assert bounds.M == 1.0

    def test_sensitivity_analysis_integration(self, simple_panel_data):
        """Test sensitivity analysis with real data."""
        event_results = TwoWayFixedEffects().fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            event_study=True,
            spec="pooled",
            post_periods=[4, 5, 6, 7],
            reference_period=3,
        )

        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(event_results, M_grid=[0, 0.5, 1.0, 2.0])

        # Bounds should widen as M increases
        widths = [ub - lb for lb, ub in sensitivity.bounds]
        assert widths[-1] >= widths[0]

    def test_smoothness_method_integration(self, simple_panel_data):
        """Test smoothness method with real data."""
        event_results = TwoWayFixedEffects().fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            event_study=True,
            spec="pooled",
            post_periods=[4, 5, 6, 7],
            reference_period=3,
        )

        honest = HonestDiD(method="smoothness", M=0.5)
        bounds = honest.fit(event_results)

        assert isinstance(bounds, HonestDiDResults)
        assert bounds.method == "smoothness"


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_post_period_no_pre_effects_raises(self):
        """Test with single post-period and no pre-period effects raises ValueError.

        HonestDiD requires pre-period coefficients for sensitivity analysis.
        A results object with only post-period effects is not usable.
        """
        # Only the post period is estimated; the reference (3) is the last pre.
        results = calendar_surface(
            {4: (5.0, 0.5, 10.0, 0.0001, (4.02, 5.98))},
            reference=3,
            post_periods=[4],
            vcov=np.array([[0.25]]),
        )

        honest = HonestDiD(method="relative_magnitude", M=1.0)
        with pytest.raises(ValueError, match="No pre-period effects with finite"):
            honest.fit(results)

    def test_m_zero_recovers_standard(self, mock_event_study_results):
        """Test that M=0 gives tighter bounds."""
        honest = HonestDiD(method="relative_magnitude")

        results_0 = honest.fit(mock_event_study_results, M=0)
        results_1 = honest.fit(mock_event_study_results, M=1)

        # M=0 should give tighter or equal bounds
        assert results_0.ci_width <= results_1.ci_width + 0.01  # Small tolerance

    def test_very_large_M(self, mock_event_study_results):
        """Test with very large M value."""
        honest = HonestDiD(method="relative_magnitude", M=100)
        results = honest.fit(mock_event_study_results)

        # Should still return valid results
        assert isinstance(results, HonestDiDResults)
        assert results.ci_width > 0

    def test_callaway_santanna_universal_base_period(self):
        """Test that reference period (e=-1) is correctly filtered out with universal base period.

        The reference period has n_groups=0 and se=NaN, so it should be excluded
        from HonestDiD analysis to avoid contaminating the vcov matrix.
        """
        from diff_diff import CallawaySantAnna, generate_staggered_data

        # Generate data and fit with universal base period
        data = generate_staggered_data(n_units=200, n_periods=10, seed=42)
        cs = CallawaySantAnna(base_period="universal")
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).aggregate("event_study")

        # Verify reference period exists with NaN SE
        times = results.event_time.tolist()
        assert -1 in times
        assert bool(results.is_reference[times.index(-1)])
        assert np.isnan(results.se[times.index(-1)])

        # HonestDiD should work without errors (reference period filtered out)
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        bounds = honest.fit(results)

        # Should have valid (non-NaN) results
        assert isinstance(bounds, HonestDiDResults)
        assert np.isfinite(bounds.ci_lb)
        assert np.isfinite(bounds.ci_ub)

    def test_max_pre_violation_excludes_reference_period(self):
        """Test that reference period (effect=0, n_groups=0) is excluded from max pre-violation.

        With universal base period, the reference period e=-1 is a normalization constraint
        with n_groups=0. It should not be used in _estimate_max_pre_violation because
        its effect is artificially set to 0, which would collapse RM bounds incorrectly.
        """
        from diff_diff import CallawaySantAnna, generate_staggered_data

        # Generate data with universal base period
        data = generate_staggered_data(n_units=200, n_periods=10, seed=42)
        cs = CallawaySantAnna(base_period="universal")
        fit = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        es = fit.aggregate("event_study")

        # Verify the reference period exists and is marked on the container
        times = es.event_time.tolist()
        assert -1 in times
        assert bool(es.is_reference[times.index(-1)])
        assert np.isnan(es.n[times.index(-1)])

        # The max pre-violation calculation should exclude the reference period
        honest = HonestDiD(method="relative_magnitude", M=1.0)

        # Get pre_periods excluding the reference row
        real_pre_periods = [
            int(t) for t, ref in zip(es.event_time, es.is_reference) if t < 0 and not ref
        ]

        # ``_estimate_max_pre_violation`` reads the CS raw ``event_study_effects``
        # dict (the 3.x fit-time shape, now the legacy-pickle route): feed it a
        # stand-in carrying the container's surface. Its reference row is the
        # zero-count sentinel the method excludes.
        results = legacy_standin(fit)
        assert results.event_study_effects[-1]["n_groups"] == 0

        # If there are real pre-periods, max_violation should be > 0
        # (based on actual pre-period effects, not the reference period's effect=0)
        if real_pre_periods:
            max_violation = honest._estimate_max_pre_violation(results, real_pre_periods)
            assert max_violation == max(abs(es.att[times.index(t)]) for t in real_pre_periods)
            # Max violation should reflect actual pre-period coefficients, not 0
            # The actual effects are non-zero due to sampling variation
            assert max_violation > 0, "max_pre_violation should be > 0 when real pre-periods exist"

    def test_calendar_surface_filters_nan_pre_period_effects(self):
        """A NaN pre-period row (rank-deficient design) is filtered by the
        calendar route's `_extract_event_study_params`, not propagated (the TWFE
        event-study surface twin of the former MPD-branch test, 4.0 cut, PR 4c-i).

        Only the extraction half is pinned: `_estimate_max_pre_violation` has
        no surface branch (it falls through to a constant).
        """
        nan_row = (np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan))
        vcov_with_nan = np.full((4, 4), 0.0)
        vcov_with_nan[0, :] = np.nan
        vcov_with_nan[:, 0] = np.nan
        vcov_with_nan[1, 1] = 0.09
        vcov_with_nan[2, 2] = 0.16
        vcov_with_nan[3, 3] = 0.16
        surface = calendar_surface(
            {
                0: nan_row,
                1: (0.1, 0.3, 0.33, 0.74, (-0.49, 0.69)),
                3: (2.5, 0.4, 6.25, 0.0001, (1.72, 3.28)),
                4: (2.8, 0.4, 7.0, 0.0001, (2.02, 3.58)),
            },
            reference=2,
            post_periods=[3, 4],
            vcov=vcov_with_nan,
        )
        beta_hat, sigma, num_pre, num_post, pre_p, post_p, _df = _extract_event_study_params(
            surface
        )
        assert len(beta_hat) == 3  # periods 1, 3, 4 (period 0 filtered)
        assert num_pre == 1
        assert num_post == 2
        assert np.all(np.isfinite(beta_hat))
        assert np.all(np.isfinite(sigma))

    def test_honest_did_all_pre_nan_raises(self):
        """HonestDiD should raise ValueError when all pre-period effects are NaN."""
        nan_row = (np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan))
        vcov = np.full((2, 2), np.nan)
        vcov[1, 1] = 0.16
        results = calendar_surface(
            {0: nan_row, 2: (2.5, 0.4, 6.25, 0.0001, (1.72, 3.28))},
            reference=1,
            post_periods=[2],
            vcov=vcov,
        )
        with pytest.raises(ValueError, match="No pre-period effects with finite"):
            _extract_event_study_params(results)

    def test_honest_did_all_post_nan_raises(self):
        """HonestDiD should raise ValueError when all post-period effects are NaN.

        When the event-study fit produces NaN for all post-period effects (e.g.
        from severe rank deficiency), HonestDiD.fit() should raise rather than
        silently computing with an empty weight vector.
        """
        nan_row = (np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan))
        vcov = np.full((3, 3), 0.0)
        vcov[0, 0] = 0.09
        vcov[1, :] = np.nan
        vcov[:, 1] = np.nan
        vcov[2, :] = np.nan
        vcov[:, 2] = np.nan
        results = calendar_surface(
            {0: (0.1, 0.3, 0.33, 0.74, (-0.49, 0.69)), 2: nan_row, 3: nan_row},
            reference=1,
            post_periods=[2, 3],
            vcov=vcov,
        )
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        with pytest.raises(ValueError, match="No post-period effects with finite"):
            honest.fit(results)

    def test_honest_did_cs_all_post_nan_raises(self):
        """HonestDiD should raise ValueError when all CS post-period effects have NaN SEs.

        When CallawaySantAnnaResults has non-finite SEs for all t>=0 event-study
        effects, HonestDiD.fit() should raise rather than computing with empty
        post-period arrays.
        """
        from diff_diff.staggered_results import CallawaySantAnnaResults

        # Create CS results with valid pre-periods but NaN SEs in post-periods
        cs_results = CallawaySantAnnaResults(
            group_time_effects={},
            att=np.nan,
            se=np.nan,
            t_stat=np.nan,
            p_value=np.nan,
            conf_int=(np.nan, np.nan),
            groups=[2004],
            time_periods=[2000, 2001, 2002, 2003],
            n_obs=400,
            n_treated_units=200,
            n_control_units=200,
        )
        cs_results.event_study_effects = {
            -2: {"effect": 0.1, "se": 0.3, "n_groups": 2},
            -1: {"effect": 0.05, "se": 0.25, "n_groups": 2},
            0: {"effect": 2.0, "se": np.nan, "n_groups": 0},
            1: {"effect": 2.5, "se": np.nan, "n_groups": 0},
        }

        honest = HonestDiD(method="relative_magnitude", M=1.0)
        with pytest.raises(ValueError, match="No post-period effects with finite"):
            honest.fit(cs_results)


# =============================================================================
# Tests for Phase 7d: Survey variance support
# =============================================================================


class TestSurveyVariance:
    """Tests for HonestDiD survey variance support (Phase 7d)."""

    def test_df_survey_extracted_from_cs_results(self):
        """df_survey is extracted from CallawaySantAnna survey_metadata."""
        from diff_diff import CallawaySantAnna, SurveyDesign, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)
        unit_ids = data["unit"].unique()
        n_units = len(unit_ids)
        unit_map = {uid: i for i, uid in enumerate(unit_ids)}
        idx = data["unit"].map(unit_map).values

        data["weight"] = (1.0 + 0.5 * (np.arange(n_units) % 4))[idx]
        data["stratum"] = (np.arange(n_units) // 25)[idx]
        data["psu"] = (np.arange(n_units) // 5)[idx]

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        cs_result = CallawaySantAnna(base_period="universal").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        es = cs_result.aggregate("event_study")

        honest = HonestDiD(method="smoothness", M=0.0)
        h_result = honest.fit(es)

        # df_survey should be propagated through the container's df_survey
        assert h_result.df_survey is not None
        assert h_result.df_survey > 0
        assert h_result.df_survey == es.df_survey
        # The container carries no survey-metadata OBJECT (documented
        # divergence from the raw-field route); df_survey is its replacement.
        assert h_result.survey_metadata is None
        # Single-source oracle: the container's df_survey resolves via the
        # shared helper (aggregation.resolve_inference_df) - value-equal,
        # float-typed.
        from diff_diff.aggregation import resolve_inference_df

        assert h_result.df_survey == resolve_inference_df(cs_result)
        assert isinstance(h_result.df_survey, float)

    def test_event_study_vcov_computed(self):
        """The CallawaySantAnna event-study container carries the full vcov HonestDiD uses."""
        from diff_diff import CallawaySantAnna, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=6, seed=42)
        cs_result = CallawaySantAnna(base_period="universal").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        # 4.0 invariant: no CS fit stores the fit-time surface or its vcov.
        assert cs_result.event_study_effects is None
        assert cs_result.event_study_vcov is None
        assert cs_result.event_study_vcov_index is None

        es = cs_result.aggregate("event_study")
        # VCV should be computed on the container
        assert es.vcov is not None
        estimated = ~es.is_reference & np.isfinite(es.se)
        n_effects = int(estimated.sum())
        assert es.vcov.shape == (n_effects, n_effects)
        np.testing.assert_array_equal(es.vcov_index, es.event_time[estimated])

        # Diagonal should be positive and equal the stored SEs squared
        for i in range(n_effects):
            assert es.vcov[i, i] > 0
        np.testing.assert_allclose(np.diag(es.vcov), es.se[estimated] ** 2, rtol=1e-12)

        # HonestDiD consumes it without the diagonal fallback
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            HonestDiD(method="smoothness", M=0.0).fit(es)
        assert not [w for w in caught if "covariance" in str(w.message)]

    def test_survey_df_widens_bounds(self):
        """Survey df (t-distribution) should give wider CIs than normal."""
        from diff_diff import CallawaySantAnna, SurveyDesign, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)
        unit_ids = data["unit"].unique()
        n_units = len(unit_ids)
        unit_map = {uid: i for i, uid in enumerate(unit_ids)}
        idx = data["unit"].map(unit_map).values

        data["weight"] = (1.0 + 0.3 * (np.arange(n_units) % 3))[idx]
        # 2 strata x 5 nested PSUs -> df = 10 - 2 = 8 (low df). The
        # stratified survey event-study VCV has rank <= df, and HonestDiD
        # validates the native-route matrix with allow_singular=False, so
        # df must be >= the 6 estimated horizons for a non-singular block
        # (4 PSUs / df = 2 would be singular and raise).
        data["stratum"] = (np.arange(n_units) // 50)[idx]
        data["psu"] = ((np.arange(n_units) // 50) * 5 + (np.arange(n_units) % 5))[idx]

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        cs_result = CallawaySantAnna(base_period="universal").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )

        # Use RM method (naive FLCI) which honors df via _get_critical_value.
        # The optimal FLCI (smoothness) uses folded normal which doesn't use df.
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        h_result = honest.fit(cs_result.aggregate("event_study"))

        # With df=8, t critical value (~2.31) > z critical value (1.96)
        # So CI width should be wider than 2*1.96*SE
        ci_width = h_result.ci_ub - h_result.ci_lb
        # Lower bound: normal-based CI width
        normal_width = 2 * 1.96 * h_result.original_se
        assert np.isfinite(ci_width), "CI should be finite with M=1.0"
        assert ci_width > normal_width

    def test_no_survey_gives_none_df(self):
        """Without survey, df_survey should be None."""
        from diff_diff import CallawaySantAnna, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)
        cs_result = CallawaySantAnna(base_period="universal").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )

        honest = HonestDiD(method="smoothness", M=0.0)
        h_result = honest.fit(cs_result.aggregate("event_study"))

        assert h_result.df_survey is None
        assert h_result.survey_metadata is None

    def test_replicate_weight_uses_diagonal_fallback(self):
        """Replicate-weight designs should NOT produce a full event-study vcov."""
        from diff_diff import CallawaySantAnna, SurveyDesign, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)
        unit_ids = data["unit"].unique()
        n_units = len(unit_ids)
        unit_map = {uid: i for i, uid in enumerate(unit_ids)}
        idx = data["unit"].map(unit_map).values

        # Create replicate weights (4 replicates)
        rng = np.random.default_rng(42)
        data["weight"] = (1.0 + 0.3 * (np.arange(n_units) % 3))[idx]
        for k in range(4):
            data[f"repwt_{k}"] = data["weight"] * rng.uniform(0.8, 1.2, len(data))
            # Make constant within unit
            unit_rw = data.groupby("unit")[f"repwt_{k}"].first()
            data[f"repwt_{k}"] = data["unit"].map(unit_rw)

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=[f"repwt_{k}" for k in range(4)],
            replicate_method="JK1",
        )
        cs_result = CallawaySantAnna().fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )

        # The container's vcov should be None (diagonal fallback for replicate designs)
        es = cs_result.aggregate("event_study")
        assert es.vcov is None and es.vcov_index is None
        assert np.isfinite(es.se[~es.is_reference]).any()

    def test_bootstrap_fit_carries_bootstrap_vcov(self):
        """A bootstrapped CS event-study container carries the covariance of
        the bootstrap draws (never the analytical vcov): its diagonal is the
        bootstrap SEs squared, and HonestDiD uses it without the diagonal
        fallback."""
        from diff_diff import CallawaySantAnna, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)
        cs_result = CallawaySantAnna(n_bootstrap=49, seed=42, base_period="universal").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
        )
        es = cs_result.aggregate("event_study")

        vcov = es.vcov
        index = es.vcov_index
        assert vcov is not None and vcov.shape == (len(index), len(index))
        times = es.event_time.tolist()
        ses = np.array([es.se[times.index(e)] for e in index])
        np.testing.assert_allclose(np.diag(vcov), ses**2, rtol=1e-12, atol=0.0)

        honest = HonestDiD(method="relative_magnitude", M=1.0)
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            h_result = honest.fit(es)
        fallback_warnings = [
            w
            for w in caught
            if "diagonal covariance" in str(w.message) or "no full covariance" in str(w.message)
        ]
        assert not fallback_warnings
        assert np.isfinite(h_result.original_se)
        assert h_result.original_se > 0

    def test_bootstrap_vcov_same_on_native_and_container_routes(self):
        """On a bootstrapped fit the replayed aggregate('event_study')
        container's vcov diagonal is the stored bootstrap SEs squared, and
        HonestDiD reads that covariance identically on the container route
        and on the native (raw-field, legacy-pickle) route - here a stand-in
        carrying the same surface in the 3.x fields."""
        import warnings as _w

        from diff_diff import CallawaySantAnna, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            cs_result = CallawaySantAnna(n_bootstrap=49, seed=42, base_period="universal").fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
            )
            es = cs_result.aggregate("event_study")
            standin = legacy_standin(cs_result)
            native = _extract_event_study_params(standin)
            container = _extract_event_study_params(es)
            h_native = HonestDiD(method="smoothness", M=0.0).fit(standin)
            h_container = HonestDiD(method="smoothness", M=0.0).fit(es)

        # The fit itself stores no surface; the container's vcov is the
        # bootstrap-draw covariance whose diagonal is se**2.
        assert cs_result.event_study_vcov is None
        in_vcov = np.isin(es.event_time, es.vcov_index)
        np.testing.assert_allclose(np.diag(es.vcov), es.se[in_vcov] ** 2, rtol=1e-12, atol=0.0)
        np.testing.assert_array_equal(native[1], standin.event_study_vcov)
        np.testing.assert_array_equal(container[0], native[0])
        np.testing.assert_allclose(container[1], native[1], rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(h_container.original_se, h_native.original_se, rtol=1e-12)
        np.testing.assert_allclose(
            [h_container.ci_lb, h_container.ci_ub], [h_native.ci_lb, h_native.ci_ub], rtol=1e-9
        )

    def test_bootstrap_fit_without_vcov_falls_back_to_diagonal(self):
        """With n_bootstrap at or below the number of event times no
        bootstrap covariance is stored (it would be singular), and HonestDiD
        keeps the diagonal fallback with its warning on both routes."""
        import warnings as _w

        from diff_diff import CallawaySantAnna, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)

        def fit(n_bootstrap):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = CallawaySantAnna(
                    n_bootstrap=n_bootstrap, seed=42, base_period="universal"
                ).fit(data, "outcome", "unit", "period", "first_treat")
                return res, res.aggregate("event_study")

        n_horizons = len(fit(49)[1].vcov_index)
        cs_result, es = fit(n_horizons)
        assert es.vcov is None and es.vcov_index is None
        # Native (raw-field) route on the same surface: a stand-in with the
        # dict but no vcov takes the diagonal fallback too.
        with pytest.warns(UserWarning, match="diagonal covariance"):
            HonestDiD(method="relative_magnitude", M=1.0).fit(legacy_standin(cs_result))
        with pytest.warns(UserWarning, match="no full covariance"):
            HonestDiD(method="relative_magnitude", M=1.0).fit(es)

    @pytest.mark.parametrize("n_bootstrap", [0, 199])
    def test_singular_vcov_few_psus_raises_on_both_routes(self, n_bootstrap):
        """Fewer clusters (PSUs) than event-study horizons: the stored
        event-study covariance (analytical or bootstrap) is singular, and
        HonestDiD raises the same singular-covariance ValueError on the
        native results route and on the aggregate('event_study') container
        route instead of returning bounds from a rank-deficient matrix."""
        import warnings as _w

        from diff_diff import CallawaySantAnna, generate_staggered_data

        data = generate_staggered_data(n_units=80, n_periods=10, cohort_periods=[4, 6, 8], seed=5)
        data["psu"] = data["unit"] % 7  # 7 PSUs << 13 horizons
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            cs_result = CallawaySantAnna(
                n_bootstrap=n_bootstrap, seed=1, base_period="universal", cluster="psu"
            ).fit(data, "outcome", "unit", "period", "first_treat")
            es = cs_result.aggregate("event_study")
            standin = legacy_standin(cs_result)
        vcov = standin.event_study_vcov
        assert vcov is not None and vcov.shape[0] > 7
        # Precondition: both PSU-built matrices are genuinely rank-deficient.
        assert np.linalg.matrix_rank(vcov) < vcov.shape[0]
        assert np.linalg.matrix_rank(es.vcov) < es.vcov.shape[0]
        with pytest.raises(ValueError, match="singular"):
            HonestDiD(method="relative_magnitude", M=1.0).fit(standin)
        with pytest.raises(ValueError, match="singular"):
            HonestDiD(method="relative_magnitude", M=1.0).fit(es)


# =============================================================================
# Tests for Visualization (without matplotlib)
# =============================================================================


class TestVisualizationNoMatplotlib:
    """Tests for visualization that don't require rendering."""

    def test_sensitivity_results_has_plot_method(self, mock_event_study_results):
        """Test that SensitivityResults has plot method."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_event_study_results)

        assert hasattr(sensitivity, "plot")
        assert callable(sensitivity.plot)


# =============================================================================
# dCDH Integration Tests
# =============================================================================


class TestDCDHIntegration:
    """HonestDiD integration with ChaisemartinDHaultfoeuille results."""

    @staticmethod
    def _fit_dcdh(n_groups=40, n_periods=6, seed=42, L_max=2):
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille
        from diff_diff.prep import generate_reversible_did_data

        df = generate_reversible_did_data(n_groups=n_groups, n_periods=n_periods, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ChaisemartinDHaultfoeuille(seed=1).fit(
                df,
                "outcome",
                "group",
                "period",
                "treatment",
                L_max=L_max,
            )

    def test_dcdh_integration(self):
        """compute_honest_did works on dCDH results (mirrors CS pattern)."""
        results = self._fit_dcdh()
        bounds = compute_honest_did(results, method="relative_magnitude", M=1.0)
        assert isinstance(bounds, HonestDiDResults)
        assert np.isfinite(bounds.ci_lb)
        assert np.isfinite(bounds.ci_ub)
        assert bounds.method == "relative_magnitude"

    def test_dcdh_extraction(self):
        """_extract_event_study_params returns correct shapes for dCDH."""
        results = self._fit_dcdh()
        beta_hat, sigma, n_pre, n_post, pre_t, post_t, df_s = _extract_event_study_params(results)
        assert n_pre >= 1
        assert n_post >= 1
        assert beta_hat.shape == (n_pre + n_post,)
        assert sigma.shape == (n_pre + n_post, n_pre + n_post)
        assert all(t < 0 for t in pre_t)
        assert all(t > 0 for t in post_t)
        assert df_s is None  # non-survey fixture → df_survey is None
        # Single-source oracle: the dCDH branch resolves via the shared
        # helper (both None on this non-survey fixture).
        from diff_diff.aggregation import resolve_inference_df

        assert df_s == resolve_inference_df(results)

    def test_dcdh_survey_df_matches_shared_resolver(self):
        """Survey dCDH fit: the extracted df equals the shared resolver's
        value (float-typed), pinning the CS/dCDH single-source
        consolidation on a branch with a FINITE df."""
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille, SurveyDesign
        from diff_diff.aggregation import resolve_inference_df
        from diff_diff.prep import generate_reversible_did_data

        data = generate_reversible_did_data(n_groups=40, n_periods=6, seed=42)
        units = data["group"].unique()
        wmap = {u: 1.0 + 0.3 * (i % 3) for i, u in enumerate(units)}
        pmap = {u: i // 5 for i, u in enumerate(units)}
        data = data.assign(weight=data["group"].map(wmap), psu=data["group"].map(pmap))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ChaisemartinDHaultfoeuille(n_bootstrap=49, seed=7).fit(
                data,
                "outcome",
                "group",
                "period",
                "treatment",
                L_max=2,
                survey_design=SurveyDesign(weights="weight", psu="psu"),
            )
        *_, df_s = _extract_event_study_params(res)
        expected = resolve_inference_df(res)
        assert df_s == expected
        if df_s is not None:
            assert isinstance(df_s, float)

    def test_dcdh_no_placebos_raises(self):
        """dCDH results without placebos raise ValueError."""
        import warnings

        from diff_diff import ChaisemartinDHaultfoeuille
        from diff_diff.prep import generate_reversible_did_data

        df = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1, placebo=False).fit(
                df,
                "outcome",
                "group",
                "period",
                "treatment",
            )
        with pytest.raises(ValueError, match="placebo_event_study"):
            compute_honest_did(r)

    def test_dcdh_emits_placebo_warning(self):
        """compute_honest_did on dCDH emits warning about placebo-based pre-periods."""
        import warnings

        results = self._fit_dcdh()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_honest_did(results)
        placebo_warnings = [
            x
            for x in w
            if "placebo" in str(x.message).lower() and "pre-period" in str(x.message).lower()
        ]
        assert (
            len(placebo_warnings) >= 1
        ), "Expected a UserWarning about placebo-based pre-period inputs"

    def test_dcdh_empty_consecutive_block_raises(self):
        """ValueError when all placebos have NaN SE (no valid pre-periods)."""
        import warnings

        # Fit real results, then corrupt placebo SEs to NaN
        results = self._fit_dcdh()
        for h in results.placebo_event_study:
            results.placebo_event_study[h]["se"] = float("nan")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No placebo horizons with finite SEs"):
                compute_honest_did(results)

    def test_dcdh_standalone_surfaces_target_metadata(self):
        """Standalone HonestDiDResults summary/to_dict include target metadata."""
        results = self._fit_dcdh()
        bounds = compute_honest_did(results, l_vec=np.array([1.0, 0.0]))
        # summary() includes target and period metadata
        text = bounds.summary()
        assert "on-impact" in text.lower()
        assert "Post horizons used:" in text
        assert "Pre horizons used:" in text
        # to_dict() includes the fields
        d = bounds.to_dict()
        assert "target_label" in d
        assert "pre_periods_used" in d
        assert "post_periods_used" in d
        assert d["post_periods_used"] == [1, 2]

    def test_dcdh_interior_gap_triggers_trimming_warning(self):
        """Non-consecutive horizons after SE filtering emit trimming warning."""
        import warnings

        # L_max=3 gives horizons [-3,-2,-1,1,2,3]. Corrupt h=-2 to create
        # interior gap [-3, -1], which triggers consecutive-block trimming
        # that drops -3 and keeps only [-1].
        results = self._fit_dcdh(n_periods=8, L_max=3)
        results.placebo_event_study[-2]["se"] = float("nan")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bounds = compute_honest_did(results)
        trim_warns = [x for x in w if "dropping non-consecutive" in str(x.message).lower()]
        assert len(trim_warns) >= 1, "Expected a warning about dropping non-consecutive horizons"
        # Retained pre should be [-1] only (h=-3 dropped due to gap at -2)
        assert bounds.pre_periods_used == [-1]

    def test_dcdh_zero_se_treated_like_nan(self):
        """se == 0 placebo rows drop exactly like NaN-SE rows (undefined
        inference; a zero row would enter Sigma with zero variance)."""
        import warnings

        results = self._fit_dcdh(n_periods=8, L_max=3)
        results.placebo_event_study[-2]["se"] = 0.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bounds = compute_honest_did(results)
        trim_warns = [x for x in w if "dropping non-consecutive" in str(x.message).lower()]
        assert len(trim_warns) >= 1
        # Same outcome as the NaN-SE interior-gap test: only [-1] retained.
        assert bounds.pre_periods_used == [-1]

    def test_dcdh_missing_boundary_minus1_raises(self):
        """ValueError when horizon -1 has NaN SE (boundary required)."""
        import warnings

        results = self._fit_dcdh()
        # Corrupt only horizon -1 SE; leave -2 intact
        results.placebo_event_study[-1]["se"] = float("nan")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="requires horizon -1"):
                compute_honest_did(results)

    def test_dcdh_missing_boundary_plus1_raises(self):
        """ValueError when horizon +1 has NaN SE (boundary required)."""
        import warnings

        results = self._fit_dcdh()
        # Corrupt only horizon +1 SE
        results.event_study_effects[1]["se"] = float("nan")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="requires horizon 1"):
                compute_honest_did(results)
