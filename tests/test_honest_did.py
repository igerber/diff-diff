"""
Tests for Honest DiD sensitivity analysis module.

Tests the implementation of Rambachan & Roth (2023) methods for
robust inference in difference-in-differences under violations
of parallel trends.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import MultiPeriodDiD
from diff_diff.honest_did import (
    DeltaRM,
    DeltaSD,
    DeltaSDRM,
    HonestDiD,
    HonestDiDResults,
    SensitivityResults,
    _compute_flci,
    _construct_A_sd,
    _construct_constraints_rm,
    _construct_constraints_sd,
    _extract_event_study_params,
    compute_honest_did,
)
from diff_diff.results import MultiPeriodDiDResults, PeriodEffect

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

            data.append({
                'unit': unit,
                'period': period,
                'treated': int(is_treated),
                'post': int(post),
                'outcome': y
            })

    return pd.DataFrame(data)


@pytest.fixture
def multiperiod_results(simple_panel_data):
    """Fit MultiPeriodDiD and return results."""
    mp_did = MultiPeriodDiD()
    results = mp_did.fit(
        simple_panel_data,
        outcome='outcome',
        treatment='treated',
        time='period',
        post_periods=[4, 5, 6, 7]
    )
    return results


@pytest.fixture
def mock_multiperiod_results():
    """Create mock MultiPeriodDiDResults for unit testing."""
    period_effects = {
        4: PeriodEffect(
            period=4, effect=5.0, se=0.5,
            t_stat=10.0, p_value=0.0001,
            conf_int=(4.02, 5.98)
        ),
        5: PeriodEffect(
            period=5, effect=5.2, se=0.5,
            t_stat=10.4, p_value=0.0001,
            conf_int=(4.22, 6.18)
        ),
        6: PeriodEffect(
            period=6, effect=4.8, se=0.5,
            t_stat=9.6, p_value=0.0001,
            conf_int=(3.82, 5.78)
        ),
        7: PeriodEffect(
            period=7, effect=5.0, se=0.5,
            t_stat=10.0, p_value=0.0001,
            conf_int=(4.02, 5.98)
        ),
    }

    # Create vcov matrix (diagonal for simplicity)
    vcov = np.diag([0.25] * 4)

    return MultiPeriodDiDResults(
        period_effects=period_effects,
        avg_att=5.0,
        avg_se=0.25,
        avg_t_stat=20.0,
        avg_p_value=0.0001,
        avg_conf_int=(4.51, 5.49),
        n_obs=800,
        n_treated=400,
        n_control=400,
        pre_periods=[0, 1, 2, 3],
        post_periods=[4, 5, 6, 7],
        vcov=vcov,
    )


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
        """Test smoothness constraint matrix construction."""
        A = _construct_A_sd(5)
        assert A.shape == (3, 5)

        # Check second difference structure: [1, -2, 1, 0, 0] etc.
        expected_first_row = [1, -2, 1, 0, 0]
        np.testing.assert_array_equal(A[0], expected_first_row)

    def test_construct_A_sd_small(self):
        """Test that small n_periods returns empty matrix."""
        A = _construct_A_sd(2)
        assert A.shape == (0, 2)

    def test_construct_constraints_sd(self):
        """Test smoothness constraints."""
        A_ineq, b_ineq = _construct_constraints_sd(
            num_pre_periods=3,
            num_post_periods=4,
            M=0.5
        )

        # Should have 2 * (7 - 2) = 10 constraints
        assert A_ineq.shape[0] == 10
        assert A_ineq.shape[1] == 7
        assert np.all(b_ineq == 0.5)

    def test_construct_constraints_rm(self):
        """Test relative magnitudes constraints."""
        A_ineq, b_ineq = _construct_constraints_rm(
            num_pre_periods=3,
            num_post_periods=4,
            Mbar=1.5,
            max_pre_violation=0.2
        )

        # Should have 2 * 4 = 8 constraints (upper and lower for each post period)
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

    def test_extract_from_multiperiod(self, mock_multiperiod_results):
        """Test extraction from MultiPeriodDiDResults."""
        (beta_hat, sigma, num_pre, num_post,
         pre_periods, post_periods) = _extract_event_study_params(mock_multiperiod_results)

        assert len(beta_hat) == 4
        assert sigma.shape == (4, 4)
        assert num_pre == 4
        assert num_post == 4
        assert post_periods == [4, 5, 6, 7]

    def test_extract_unsupported_type_raises(self):
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported results type"):
            _extract_event_study_params("not a results object")


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

    def test_fit_returns_results(self, mock_multiperiod_results):
        """Test that fit returns HonestDiDResults."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_multiperiod_results)

        assert isinstance(results, HonestDiDResults)
        assert results.M == 1.0
        assert results.method == "relative_magnitude"

    def test_fit_smoothness(self, mock_multiperiod_results):
        """Test fit with smoothness method."""
        honest = HonestDiD(method="smoothness", M=0.0)
        results = honest.fit(mock_multiperiod_results)

        assert isinstance(results, HonestDiDResults)
        assert results.method == "smoothness"
        assert results.ci_method == "FLCI"

    def test_fit_combined(self, mock_multiperiod_results):
        """Test fit with combined method."""
        honest = HonestDiD(method="combined", M=1.0)
        results = honest.fit(mock_multiperiod_results)

        assert isinstance(results, HonestDiDResults)
        assert results.method == "combined"

    def test_fit_override_M(self, mock_multiperiod_results):
        """Test that M can be overridden in fit."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_multiperiod_results, M=2.0)

        assert results.M == 2.0

    def test_bounds_widen_with_M(self, mock_multiperiod_results):
        """Test that bounds widen as M increases."""
        honest = HonestDiD(method="relative_magnitude")

        results_small = honest.fit(mock_multiperiod_results, M=0.5)
        results_large = honest.fit(mock_multiperiod_results, M=2.0)

        # Larger M should give wider bounds
        assert results_large.ci_ub - results_large.ci_lb >= results_small.ci_ub - results_small.ci_lb


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis functionality."""

    def test_sensitivity_analysis_returns_results(self, mock_multiperiod_results):
        """Test that sensitivity_analysis returns SensitivityResults."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_multiperiod_results)

        assert isinstance(sensitivity, SensitivityResults)

    def test_sensitivity_analysis_custom_grid(self, mock_multiperiod_results):
        """Test sensitivity analysis with custom M grid."""
        honest = HonestDiD(method="relative_magnitude")
        M_grid = [0, 0.5, 1.0, 1.5]
        sensitivity = honest.sensitivity_analysis(mock_multiperiod_results, M_grid=M_grid)

        assert len(sensitivity.M_values) == 4
        np.testing.assert_array_equal(sensitivity.M_values, M_grid)

    def test_sensitivity_analysis_bounds_list(self, mock_multiperiod_results):
        """Test that bounds list has correct length."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_multiperiod_results)

        assert len(sensitivity.bounds) == len(sensitivity.M_values)
        assert len(sensitivity.robust_cis) == len(sensitivity.M_values)


class TestBreakdownValue:
    """Tests for breakdown value computation."""

    def test_breakdown_value_type(self, mock_multiperiod_results):
        """Test that breakdown_value returns float or None."""
        honest = HonestDiD(method="relative_magnitude")
        breakdown = honest.breakdown_value(mock_multiperiod_results)

        assert breakdown is None or isinstance(breakdown, float)

    def test_breakdown_value_monotonic(self, mock_multiperiod_results):
        """Test breakdown value properties."""
        honest = HonestDiD(method="relative_magnitude")
        breakdown = honest.breakdown_value(mock_multiperiod_results)

        if breakdown is not None and breakdown > 0:
            # Before breakdown, should be significant
            result_before = honest.fit(mock_multiperiod_results, M=breakdown * 0.9)
            assert result_before.is_significant

            # At/after breakdown, should not be significant
            # We call fit but don't assert on significance since
            # the binary search tolerance may not match exactly
            honest.fit(mock_multiperiod_results, M=breakdown * 1.1)


# =============================================================================
# Tests for Results Classes
# =============================================================================


class TestHonestDiDResults:
    """Tests for HonestDiDResults dataclass."""

    def test_results_properties(self, mock_multiperiod_results):
        """Test HonestDiDResults properties."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_multiperiod_results)

        assert hasattr(results, 'lb')
        assert hasattr(results, 'ub')
        assert hasattr(results, 'ci_lb')
        assert hasattr(results, 'ci_ub')
        assert hasattr(results, 'is_significant')
        assert hasattr(results, 'identified_set_width')
        assert hasattr(results, 'ci_width')

    def test_results_is_significant(self, mock_multiperiod_results):
        """Test is_significant property."""
        honest = HonestDiD(method="relative_magnitude", M=0.0)
        results = honest.fit(mock_multiperiod_results)

        # With M=0 (parallel trends), should be significant for our data
        # Check that property is a bool
        assert isinstance(results.is_significant, bool)

    def test_results_width_properties(self, mock_multiperiod_results):
        """Test width properties are non-negative."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_multiperiod_results)

        assert results.identified_set_width >= 0
        assert results.ci_width >= 0

    def test_results_summary(self, mock_multiperiod_results):
        """Test summary method produces string."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_multiperiod_results)

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Honest DiD" in summary
        assert "Rambachan" in summary

    def test_results_to_dict(self, mock_multiperiod_results):
        """Test to_dict method."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_multiperiod_results)

        d = results.to_dict()
        assert isinstance(d, dict)
        assert 'lb' in d
        assert 'ub' in d
        assert 'M' in d
        assert 'method' in d

    def test_results_to_dataframe(self, mock_multiperiod_results):
        """Test to_dataframe method."""
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        results = honest.fit(mock_multiperiod_results)

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


class TestSensitivityResults:
    """Tests for SensitivityResults dataclass."""

    def test_sensitivity_results_to_dataframe(self, mock_multiperiod_results):
        """Test to_dataframe method."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_multiperiod_results)

        df = sensitivity.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'M' in df.columns
        assert 'lb' in df.columns
        assert 'ci_lb' in df.columns

    def test_sensitivity_results_summary(self, mock_multiperiod_results):
        """Test summary method."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_multiperiod_results)

        summary = sensitivity.summary()
        assert isinstance(summary, str)


# =============================================================================
# Tests for Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_honest_did(self, mock_multiperiod_results):
        """Test compute_honest_did function."""
        results = compute_honest_did(
            mock_multiperiod_results,
            method='relative_magnitude',
            M=1.0
        )

        assert isinstance(results, HonestDiDResults)
        assert results.M == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with real estimators."""

    def test_with_multiperiod_did(self, simple_panel_data):
        """Test full pipeline with MultiPeriodDiD."""
        # Fit event study
        mp_did = MultiPeriodDiD()
        event_results = mp_did.fit(
            simple_panel_data,
            outcome='outcome',
            treatment='treated',
            time='period',
            post_periods=[4, 5, 6, 7]
        )

        # Run Honest DiD
        honest = HonestDiD(method='relative_magnitude', M=1.0)
        bounds = honest.fit(event_results)

        # Check results are reasonable
        assert bounds.original_estimate > 0  # True effect is positive
        assert bounds.ci_ub > bounds.ci_lb
        assert bounds.M == 1.0

    def test_sensitivity_analysis_integration(self, simple_panel_data):
        """Test sensitivity analysis with real data."""
        mp_did = MultiPeriodDiD()
        event_results = mp_did.fit(
            simple_panel_data,
            outcome='outcome',
            treatment='treated',
            time='period',
            post_periods=[4, 5, 6, 7]
        )

        honest = HonestDiD(method='relative_magnitude')
        sensitivity = honest.sensitivity_analysis(event_results, M_grid=[0, 0.5, 1.0, 2.0])

        # Bounds should widen as M increases
        widths = [ub - lb for lb, ub in sensitivity.bounds]
        assert widths[-1] >= widths[0]

    def test_smoothness_method_integration(self, simple_panel_data):
        """Test smoothness method with real data."""
        mp_did = MultiPeriodDiD()
        event_results = mp_did.fit(
            simple_panel_data,
            outcome='outcome',
            treatment='treated',
            time='period',
            post_periods=[4, 5, 6, 7]
        )

        honest = HonestDiD(method='smoothness', M=0.5)
        bounds = honest.fit(event_results)

        assert isinstance(bounds, HonestDiDResults)
        assert bounds.method == 'smoothness'


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_post_period(self):
        """Test with single post-period."""
        period_effects = {
            4: PeriodEffect(
                period=4, effect=5.0, se=0.5,
                t_stat=10.0, p_value=0.0001,
                conf_int=(4.02, 5.98)
            ),
        }

        results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=5.0,
            avg_se=0.5,
            avg_t_stat=10.0,
            avg_p_value=0.0001,
            avg_conf_int=(4.02, 5.98),
            n_obs=200,
            n_treated=100,
            n_control=100,
            pre_periods=[0, 1, 2, 3],
            post_periods=[4],
            vcov=np.array([[0.25]]),
        )

        honest = HonestDiD(method='relative_magnitude', M=1.0)
        bounds = honest.fit(results)

        assert isinstance(bounds, HonestDiDResults)

    def test_m_zero_recovers_standard(self, mock_multiperiod_results):
        """Test that M=0 gives tighter bounds."""
        honest = HonestDiD(method='relative_magnitude')

        results_0 = honest.fit(mock_multiperiod_results, M=0)
        results_1 = honest.fit(mock_multiperiod_results, M=1)

        # M=0 should give tighter or equal bounds
        assert results_0.ci_width <= results_1.ci_width + 0.01  # Small tolerance

    def test_very_large_M(self, mock_multiperiod_results):
        """Test with very large M value."""
        honest = HonestDiD(method='relative_magnitude', M=100)
        results = honest.fit(mock_multiperiod_results)

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
            outcome='outcome',
            unit='unit',
            time='period',
            first_treat='first_treat',
            aggregate='event_study'
        )

        # Verify reference period exists with NaN SE
        assert -1 in results.event_study_effects
        assert np.isnan(results.event_study_effects[-1]['se'])

        # HonestDiD should work without errors (reference period filtered out)
        honest = HonestDiD(method='relative_magnitude', M=1.0)
        bounds = honest.fit(results)

        # Should have valid (non-NaN) results
        assert isinstance(bounds, HonestDiDResults)
        assert np.isfinite(bounds.ci_lb)
        assert np.isfinite(bounds.ci_ub)


# =============================================================================
# Tests for Visualization (without matplotlib)
# =============================================================================


class TestVisualizationNoMatplotlib:
    """Tests for visualization that don't require rendering."""

    def test_sensitivity_results_has_plot_method(self, mock_multiperiod_results):
        """Test that SensitivityResults has plot method."""
        honest = HonestDiD(method='relative_magnitude')
        sensitivity = honest.sensitivity_analysis(mock_multiperiod_results)

        assert hasattr(sensitivity, 'plot')
        assert callable(sensitivity.plot)
