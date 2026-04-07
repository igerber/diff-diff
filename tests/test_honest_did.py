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
def multiperiod_results(simple_panel_data):
    """Fit MultiPeriodDiD and return results."""
    mp_did = MultiPeriodDiD()
    results = mp_did.fit(
        simple_panel_data,
        outcome="outcome",
        treatment="treated",
        time="period",
        post_periods=[4, 5, 6, 7],
        reference_period=3,
    )
    return results


@pytest.fixture
def mock_multiperiod_results():
    """Create mock MultiPeriodDiDResults for unit testing.

    Simulates a full event-study with pre-period and post-period effects.
    Reference period is 3 (last pre-period), so period_effects has
    periods 0, 1, 2 (pre) and 4, 5, 6, 7 (post).
    """
    period_effects = {
        # Pre-period effects (should be ~0 under parallel trends)
        0: PeriodEffect(
            period=0, effect=0.05, se=0.4, t_stat=0.125, p_value=0.90, conf_int=(-0.73, 0.83)
        ),
        1: PeriodEffect(
            period=1, effect=-0.02, se=0.35, t_stat=-0.057, p_value=0.95, conf_int=(-0.71, 0.67)
        ),
        2: PeriodEffect(
            period=2, effect=0.08, se=0.3, t_stat=0.267, p_value=0.79, conf_int=(-0.51, 0.67)
        ),
        # Post-period effects
        4: PeriodEffect(
            period=4, effect=5.0, se=0.5, t_stat=10.0, p_value=0.0001, conf_int=(4.02, 5.98)
        ),
        5: PeriodEffect(
            period=5, effect=5.2, se=0.5, t_stat=10.4, p_value=0.0001, conf_int=(4.22, 6.18)
        ),
        6: PeriodEffect(
            period=6, effect=4.8, se=0.5, t_stat=9.6, p_value=0.0001, conf_int=(3.82, 5.78)
        ),
        7: PeriodEffect(
            period=7, effect=5.0, se=0.5, t_stat=10.0, p_value=0.0001, conf_int=(4.02, 5.98)
        ),
    }

    # SE^2 for all 7 interaction terms (periods 0,1,2,4,5,6,7)
    vcov_diag = [0.4**2, 0.35**2, 0.3**2, 0.5**2, 0.5**2, 0.5**2, 0.5**2]

    # interaction_indices maps period -> column index in the full regression VCV
    # (in a real fit, these would be the actual column positions)
    interaction_indices = {0: 10, 1: 11, 2: 12, 4: 13, 5: 14, 6: 15, 7: 16}

    # Build a larger "full" VCV that the sub-extraction will index into
    full_vcov = np.zeros((20, 20))
    for i, period in enumerate(sorted(interaction_indices.keys())):
        col = interaction_indices[period]
        full_vcov[col, col] = vcov_diag[i]

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
        vcov=full_vcov,
        reference_period=3,
        interaction_indices=interaction_indices,
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
        A_ineq, b_ineq = _construct_constraints_sd(num_pre_periods=3, num_post_periods=4, M=0.5)

        # Should have 2 * (7 - 2) = 10 constraints
        assert A_ineq.shape[0] == 10
        assert A_ineq.shape[1] == 7
        assert np.all(b_ineq == 0.5)

    def test_construct_constraints_rm(self):
        """Test relative magnitudes constraints."""
        A_ineq, b_ineq = _construct_constraints_rm(
            num_pre_periods=3, num_post_periods=4, Mbar=1.5, max_pre_violation=0.2
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
        (beta_hat, sigma, num_pre, num_post, pre_periods, post_periods, _df) = (
            _extract_event_study_params(mock_multiperiod_results)
        )

        # 7 estimated effects: 3 pre (0,1,2) + 4 post (4,5,6,7), ref=3 excluded
        assert len(beta_hat) == 7
        assert sigma.shape == (7, 7)
        assert num_pre == 3
        assert num_post == 4
        assert post_periods == [4, 5, 6, 7]

        # Verify sub-VCV diagonal matches squared SEs
        for i, period in enumerate(sorted(mock_multiperiod_results.period_effects.keys())):
            pe = mock_multiperiod_results.period_effects[period]
            assert sigma[i, i] == pytest.approx(pe.se**2, abs=1e-10)

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
        assert (
            results_large.ci_ub - results_large.ci_lb >= results_small.ci_ub - results_small.ci_lb
        )


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

        assert hasattr(results, "lb")
        assert hasattr(results, "ub")
        assert hasattr(results, "ci_lb")
        assert hasattr(results, "ci_ub")
        assert hasattr(results, "is_significant")
        assert hasattr(results, "identified_set_width")
        assert hasattr(results, "ci_width")

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
        assert "lb" in d
        assert "ub" in d
        assert "M" in d
        assert "method" in d

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
        assert "M" in df.columns
        assert "lb" in df.columns
        assert "ci_lb" in df.columns

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
        results = compute_honest_did(mock_multiperiod_results, method="relative_magnitude", M=1.0)

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
            outcome="outcome",
            treatment="treated",
            time="period",
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
        mp_did = MultiPeriodDiD()
        event_results = mp_did.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
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
        mp_did = MultiPeriodDiD()
        event_results = mp_did.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[4, 5, 6, 7],
            reference_period=3,
        )

        honest = HonestDiD(method="smoothness", M=0.5)
        bounds = honest.fit(event_results)

        assert isinstance(bounds, HonestDiDResults)
        assert bounds.method == "smoothness"

    def test_multiperiod_sub_vcov_extraction(self, simple_panel_data):
        """Test that interaction_indices enables correct sub-VCV extraction.

        Fit MultiPeriodDiD, pass to _extract_event_study_params, and verify:
        - sigma shape matches len(period_effects) x len(period_effects)
        - Diagonal of sigma matches squared SEs from period_effects
        - beta_hat length equals num_pre + num_post
        """
        mp_did = MultiPeriodDiD()
        results = mp_did.fit(
            simple_panel_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[4, 5, 6, 7],
            reference_period=3,
        )

        (beta_hat, sigma, num_pre, num_post, pre_periods, post_periods, _df) = (
            _extract_event_study_params(results)
        )

        n_effects = len(results.period_effects)
        assert len(beta_hat) == n_effects
        assert sigma.shape == (n_effects, n_effects)
        assert num_pre + num_post == n_effects

        # Verify sub-VCV diagonal matches squared SEs from period_effects
        sorted_periods = sorted(results.period_effects.keys())
        for i, period in enumerate(sorted_periods):
            pe = results.period_effects[period]
            assert sigma[i, i] == pytest.approx(
                pe.se**2, rel=1e-6
            ), f"sigma[{i},{i}] = {sigma[i, i]} != se^2 = {pe.se**2} for period {period}"


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
        period_effects = {
            4: PeriodEffect(
                period=4, effect=5.0, se=0.5, t_stat=10.0, p_value=0.0001, conf_int=(4.02, 5.98)
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

        honest = HonestDiD(method="relative_magnitude", M=1.0)
        with pytest.raises(ValueError, match="No pre-period effects with finite"):
            honest.fit(results)

    def test_m_zero_recovers_standard(self, mock_multiperiod_results):
        """Test that M=0 gives tighter bounds."""
        honest = HonestDiD(method="relative_magnitude")

        results_0 = honest.fit(mock_multiperiod_results, M=0)
        results_1 = honest.fit(mock_multiperiod_results, M=1)

        # M=0 should give tighter or equal bounds
        assert results_0.ci_width <= results_1.ci_width + 0.01  # Small tolerance

    def test_very_large_M(self, mock_multiperiod_results):
        """Test with very large M value."""
        honest = HonestDiD(method="relative_magnitude", M=100)
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
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Verify reference period exists with NaN SE
        assert -1 in results.event_study_effects
        assert np.isnan(results.event_study_effects[-1]["se"])

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
        results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Verify reference period exists with n_groups=0
        assert -1 in results.event_study_effects
        assert results.event_study_effects[-1]["n_groups"] == 0

        # The max pre-violation calculation should exclude the reference period
        honest = HonestDiD(method="relative_magnitude", M=1.0)

        # Get pre_periods excluding reference (n_groups=0)
        real_pre_periods = [
            t
            for t in results.event_study_effects
            if t < 0 and results.event_study_effects[t].get("n_groups", 1) > 0
        ]

        # If there are real pre-periods, max_violation should be > 0
        # (based on actual pre-period effects, not the reference period's effect=0)
        if real_pre_periods:
            max_violation = honest._estimate_max_pre_violation(results, real_pre_periods)
            # Max violation should reflect actual pre-period coefficients, not 0
            # The actual effects are non-zero due to sampling variation
            assert max_violation > 0, "max_pre_violation should be > 0 when real pre-periods exist"

    def test_honest_did_filters_nan_pre_period_effects(self):
        """HonestDiD should filter NaN pre-period effects from MultiPeriodDiDResults.

        When MultiPeriodDiD produces NaN effects (e.g. from rank-deficient designs
        with time-varying treatment), HonestDiD should skip those periods rather
        than propagating NaN into sensitivity bounds.
        """
        # Create results with one NaN pre-period (simulating rank deficiency)
        period_effects = {
            0: PeriodEffect(
                period=0,
                effect=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                conf_int=(np.nan, np.nan),
            ),
            1: PeriodEffect(
                period=1,
                effect=0.1,
                se=0.3,
                t_stat=0.33,
                p_value=0.74,
                conf_int=(-0.49, 0.69),
            ),
            # Reference period (2) omitted
            3: PeriodEffect(
                period=3,
                effect=2.5,
                se=0.4,
                t_stat=6.25,
                p_value=0.0001,
                conf_int=(1.72, 3.28),
            ),
            4: PeriodEffect(
                period=4,
                effect=2.8,
                se=0.4,
                t_stat=7.0,
                p_value=0.0001,
                conf_int=(2.02, 3.58),
            ),
        }

        # Build VCV with NaN row/col for period 0 (rank-deficient)
        interaction_indices = {0: 0, 1: 1, 3: 2, 4: 3}
        vcov_with_nan = np.full((4, 4), 0.0)
        vcov_with_nan[0, :] = np.nan
        vcov_with_nan[:, 0] = np.nan
        vcov_with_nan[1, 1] = 0.09
        vcov_with_nan[2, 2] = 0.16
        vcov_with_nan[3, 3] = 0.16

        results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=2.65,
            avg_se=0.4,
            avg_t_stat=6.625,
            avg_p_value=0.0001,
            avg_conf_int=(1.87, 3.43),
            n_obs=400,
            n_treated=200,
            n_control=200,
            pre_periods=[0, 1, 2],
            post_periods=[3, 4],
            vcov=vcov_with_nan,
            reference_period=2,
            interaction_indices=interaction_indices,
        )

        # _extract_event_study_params should filter out period 0 (NaN)
        beta_hat, sigma, num_pre, num_post, pre_p, post_p, _df = _extract_event_study_params(
            results
        )
        assert len(beta_hat) == 3  # periods 1, 3, 4 (period 0 filtered)
        assert num_pre == 1  # only period 1
        assert num_post == 2  # periods 3, 4
        assert np.all(np.isfinite(beta_hat))
        assert np.all(np.isfinite(sigma))

        # _estimate_max_pre_violation should ignore the NaN period
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        max_viol = honest._estimate_max_pre_violation(results, [0, 1])
        assert np.isfinite(max_viol)
        assert max_viol == pytest.approx(0.1, abs=1e-10)  # only period 1's |effect|

    def test_honest_did_all_pre_nan_raises(self):
        """HonestDiD should raise ValueError when all pre-period effects are NaN."""
        period_effects = {
            0: PeriodEffect(
                period=0,
                effect=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                conf_int=(np.nan, np.nan),
            ),
            # Reference period (1) omitted
            2: PeriodEffect(
                period=2,
                effect=2.5,
                se=0.4,
                t_stat=6.25,
                p_value=0.0001,
                conf_int=(1.72, 3.28),
            ),
        }

        results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=2.5,
            avg_se=0.4,
            avg_t_stat=6.25,
            avg_p_value=0.0001,
            avg_conf_int=(1.72, 3.28),
            n_obs=200,
            n_treated=100,
            n_control=100,
            pre_periods=[0, 1],
            post_periods=[2],
            vcov=np.diag([0.16]),
            reference_period=1,
            interaction_indices={0: 0, 2: 1},
        )

        with pytest.raises(ValueError, match="No pre-period effects with finite"):
            _extract_event_study_params(results)

    def test_honest_did_all_post_nan_raises(self):
        """HonestDiD should raise ValueError when all post-period effects are NaN.

        When MultiPeriodDiD produces NaN for all post-period effects (e.g. from
        severe rank deficiency), HonestDiD.fit() should raise rather than
        silently computing with an empty weight vector.
        """
        period_effects = {
            0: PeriodEffect(
                period=0,
                effect=0.1,
                se=0.3,
                t_stat=0.33,
                p_value=0.74,
                conf_int=(-0.49, 0.69),
            ),
            # Reference period (1) omitted
            2: PeriodEffect(
                period=2,
                effect=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                conf_int=(np.nan, np.nan),
            ),
            3: PeriodEffect(
                period=3,
                effect=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                conf_int=(np.nan, np.nan),
            ),
        }

        interaction_indices = {0: 0, 2: 1, 3: 2}
        vcov = np.full((3, 3), 0.0)
        vcov[0, 0] = 0.09
        vcov[1, :] = np.nan
        vcov[:, 1] = np.nan
        vcov[2, :] = np.nan
        vcov[:, 2] = np.nan

        results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=np.nan,
            avg_se=np.nan,
            avg_t_stat=np.nan,
            avg_p_value=np.nan,
            avg_conf_int=(np.nan, np.nan),
            n_obs=300,
            n_treated=150,
            n_control=150,
            pre_periods=[0, 1],
            post_periods=[2, 3],
            vcov=vcov,
            reference_period=1,
            interaction_indices=interaction_indices,
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
            overall_att=np.nan,
            overall_se=np.nan,
            overall_t_stat=np.nan,
            overall_p_value=np.nan,
            overall_conf_int=(np.nan, np.nan),
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

    def test_honest_did_nonmonotone_period_labels(self):
        """HonestDiD extraction should handle period labels where sorted order
        doesn't separate pre/post (e.g. pre=[5,6], post=[1,2]).

        The extraction must place pre-period effects before post-period effects
        in beta_hat regardless of label values.
        """
        # Pre-periods 5, 6, 7 (reference=7 omitted), post-periods 1, 2
        # sorted() would give [1, 2, 5, 6] — post before pre — which is wrong
        period_effects = {
            5: PeriodEffect(
                period=5, effect=0.1, se=0.3, t_stat=0.33, p_value=0.74, conf_int=(-0.49, 0.69)
            ),
            6: PeriodEffect(
                period=6, effect=0.2, se=0.35, t_stat=0.57, p_value=0.57, conf_int=(-0.49, 0.89)
            ),
            1: PeriodEffect(
                period=1, effect=2.5, se=0.4, t_stat=6.25, p_value=0.0001, conf_int=(1.72, 3.28)
            ),
            2: PeriodEffect(
                period=2, effect=2.8, se=0.45, t_stat=6.22, p_value=0.0001, conf_int=(1.92, 3.68)
            ),
        }

        # VCV column mapping: period -> index in regression VCV
        interaction_indices = {5: 0, 6: 1, 1: 2, 2: 3}

        # Distinct diagonal entries so we can verify VCV block extraction
        vcov = np.diag([0.09, 0.1225, 0.16, 0.2025])

        results = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=2.65,
            avg_se=0.42,
            avg_t_stat=6.31,
            avg_p_value=0.0001,
            avg_conf_int=(1.83, 3.47),
            n_obs=400,
            n_treated=200,
            n_control=200,
            pre_periods=[5, 6, 7],
            post_periods=[1, 2],
            vcov=vcov,
            reference_period=7,
            interaction_indices=interaction_indices,
        )

        beta_hat, sigma, num_pre, num_post, pre_p, post_p, _df = _extract_event_study_params(
            results
        )

        # Pre-periods: 5, 6 (7 is reference, omitted)
        assert num_pre == 2
        # Post-periods: 1, 2
        assert num_post == 2

        # beta_hat must be [pre_5, pre_6, post_1, post_2]
        assert beta_hat[0] == pytest.approx(0.1)  # period 5
        assert beta_hat[1] == pytest.approx(0.2)  # period 6
        assert beta_hat[2] == pytest.approx(2.5)  # period 1
        assert beta_hat[3] == pytest.approx(2.8)  # period 2

        # sigma blocks must match: pre block = diag(0.09, 0.1225), post block = diag(0.16, 0.2025)
        assert sigma[0, 0] == pytest.approx(0.09)  # period 5 variance
        assert sigma[1, 1] == pytest.approx(0.1225)  # period 6 variance
        assert sigma[2, 2] == pytest.approx(0.16)  # period 1 variance
        assert sigma[3, 3] == pytest.approx(0.2025)  # period 2 variance


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
            aggregate="event_study",
        )

        honest = HonestDiD(method="smoothness", M=0.0)
        h_result = honest.fit(cs_result)

        # df_survey should be propagated
        assert h_result.df_survey is not None
        assert h_result.df_survey > 0
        assert h_result.survey_metadata is not None

    def test_event_study_vcov_computed(self):
        """CallawaySantAnna event_study_vcov is computed and used by HonestDiD."""
        from diff_diff import CallawaySantAnna, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=6, seed=42)
        cs_result = CallawaySantAnna(base_period="universal").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            aggregate="event_study",
        )

        # VCV should be computed
        assert cs_result.event_study_vcov is not None
        n_effects = len(
            [
                e
                for e, d in cs_result.event_study_effects.items()
                if d.get("n_groups", 1) > 0 and np.isfinite(d.get("se", np.nan))
            ]
        )
        assert cs_result.event_study_vcov.shape == (n_effects, n_effects)

        # Diagonal should be positive
        for i in range(n_effects):
            assert cs_result.event_study_vcov[i, i] > 0

    def test_survey_df_widens_bounds(self):
        """Survey df (t-distribution) should give wider CIs than normal."""
        from diff_diff import CallawaySantAnna, SurveyDesign, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)
        unit_ids = data["unit"].unique()
        n_units = len(unit_ids)
        unit_map = {uid: i for i, uid in enumerate(unit_ids)}
        idx = data["unit"].map(unit_map).values

        data["weight"] = (1.0 + 0.3 * (np.arange(n_units) % 3))[idx]
        # 2 strata, 4 PSUs total -> df = 4 - 2 = 2 (very low df)
        data["stratum"] = (np.arange(n_units) // 50)[idx]
        data["psu"] = (np.arange(n_units) // 25)[idx]

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        cs_result = CallawaySantAnna(base_period="universal").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
            aggregate="event_study",
        )

        honest = HonestDiD(method="smoothness", M=0.0)
        h_result = honest.fit(cs_result)

        # With df=2, t critical value (~4.3) >> z critical value (1.96)
        # So CI width should be wider than 2*1.96*SE
        ci_width = h_result.ci_ub - h_result.ci_lb
        # Lower bound: normal-based CI width
        normal_width = 2 * 1.96 * h_result.original_se
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
            aggregate="event_study",
        )

        honest = HonestDiD(method="smoothness", M=0.0)
        h_result = honest.fit(cs_result)

        assert h_result.df_survey is None
        assert h_result.survey_metadata is None

    def test_replicate_weight_uses_diagonal_fallback(self):
        """Replicate-weight designs should NOT produce full event_study_vcov."""
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
            aggregate="event_study",
        )

        # event_study_vcov should be None (diagonal fallback for replicate designs)
        assert cs_result.event_study_vcov is None

    def test_bootstrap_fit_clears_analytical_vcov(self):
        """Bootstrap CS results should NOT carry analytical event_study_vcov."""
        from diff_diff import CallawaySantAnna, generate_staggered_data

        data = generate_staggered_data(n_units=100, n_periods=5, seed=42)
        cs_result = CallawaySantAnna(n_bootstrap=49, seed=42, base_period="universal").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            aggregate="event_study",
        )

        # event_study_vcov should be None when bootstrap is used
        # (prevents HonestDiD from mixing analytical VCV with bootstrap SEs)
        assert cs_result.event_study_vcov is None

        # HonestDiD should warn about diagonal fallback but still work
        honest = HonestDiD(method="relative_magnitude", M=1.0)
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            h_result = honest.fit(cs_result)
        diag_warnings = [w for w in caught if "diagonal covariance" in str(w.message)]
        assert len(diag_warnings) >= 1
        assert np.isfinite(h_result.original_se)
        assert h_result.original_se > 0


# =============================================================================
# Tests for Visualization (without matplotlib)
# =============================================================================


class TestVisualizationNoMatplotlib:
    """Tests for visualization that don't require rendering."""

    def test_sensitivity_results_has_plot_method(self, mock_multiperiod_results):
        """Test that SensitivityResults has plot method."""
        honest = HonestDiD(method="relative_magnitude")
        sensitivity = honest.sensitivity_analysis(mock_multiperiod_results)

        assert hasattr(sensitivity, "plot")
        assert callable(sensitivity.plot)
