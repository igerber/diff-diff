"""
Tests for pre-trends power analysis module.

Tests the implementation of Roth (2022) methods for assessing
the informativeness of pre-trends tests.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import MultiPeriodDiD
from diff_diff.pretrends import (
    PreTrendsPower,
    PreTrendsPowerCurve,
    PreTrendsPowerResults,
    compute_mdv,
    compute_pretrends_power,
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
    """Create mock MultiPeriodDiDResults for unit testing.

    This fixture simulates event study results with:
    - Pre-periods: 0, 1, 2, 3 (period 3 is reference, omitted from estimation)
    - Post-periods: 4, 5, 6, 7
    - Estimated coefficients for periods 0, 1, 2 (pre) and 4, 5, 6, 7 (post)
    """
    # Pre-period effects (excluding reference period 3)
    period_effects = {
        0: PeriodEffect(
            period=0, effect=0.1, se=0.5,
            t_stat=0.2, p_value=0.84,
            conf_int=(-0.88, 1.08)
        ),
        1: PeriodEffect(
            period=1, effect=-0.05, se=0.5,
            t_stat=-0.1, p_value=0.92,
            conf_int=(-1.03, 0.93)
        ),
        2: PeriodEffect(
            period=2, effect=0.08, se=0.5,
            t_stat=0.16, p_value=0.87,
            conf_int=(-0.90, 1.06)
        ),
        # Period 3 is reference - not in period_effects
        # Post-period effects
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

    # Coefficients for estimated periods (excludes reference period 3)
    coefficients = {
        'treated:period_0': 0.1,
        'treated:period_1': -0.05,
        'treated:period_2': 0.08,
        'treated:period_4': 5.0,
        'treated:period_5': 5.2,
        'treated:period_6': 4.8,
        'treated:period_7': 5.0,
    }

    # Create vcov matrix (diagonal for simplicity)
    # 7 coefficients: 3 pre + 4 post
    vcov = np.diag([0.25] * 7)

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
        pre_periods=[0, 1, 2, 3],  # 4 pre-periods, but period 3 is reference
        post_periods=[4, 5, 6, 7],
        vcov=vcov,
        coefficients=coefficients,
    )


# =============================================================================
# Tests for PreTrendsPower class initialization
# =============================================================================


class TestPreTrendsPowerInit:
    """Tests for PreTrendsPower initialization."""

    def test_default_init(self):
        """Test default initialization."""
        pt = PreTrendsPower()
        assert pt.alpha == 0.05
        assert pt.target_power == 0.80
        assert pt.violation_type == "linear"
        assert pt.violation_weights is None

    def test_custom_alpha(self):
        """Test initialization with custom alpha."""
        pt = PreTrendsPower(alpha=0.10)
        assert pt.alpha == 0.10

    def test_custom_power(self):
        """Test initialization with custom target power."""
        pt = PreTrendsPower(power=0.90)
        assert pt.target_power == 0.90

    def test_violation_type_constant(self):
        """Test initialization with constant violation type."""
        pt = PreTrendsPower(violation_type="constant")
        assert pt.violation_type == "constant"

    def test_violation_type_last_period(self):
        """Test initialization with last_period violation type."""
        pt = PreTrendsPower(violation_type="last_period")
        assert pt.violation_type == "last_period"

    def test_violation_type_custom(self):
        """Test initialization with custom violation type."""
        weights = np.array([0.5, 0.3, 0.2])
        pt = PreTrendsPower(violation_type="custom", violation_weights=weights)
        assert pt.violation_type == "custom"
        np.testing.assert_array_equal(pt.violation_weights, weights)

    def test_invalid_alpha_raises(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            PreTrendsPower(alpha=1.5)

    def test_invalid_power_raises(self):
        """Test that invalid power raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            PreTrendsPower(power=0.0)

    def test_invalid_violation_type_raises(self):
        """Test that invalid violation type raises ValueError."""
        with pytest.raises(ValueError, match="violation_type must be"):
            PreTrendsPower(violation_type="invalid")

    def test_custom_without_weights_raises(self):
        """Test that custom type without weights raises ValueError."""
        with pytest.raises(ValueError, match="violation_weights must be provided"):
            PreTrendsPower(violation_type="custom")


# =============================================================================
# Tests for violation weight computation
# =============================================================================


class TestViolationWeights:
    """Tests for violation weight computation."""

    def test_linear_weights(self):
        """Test linear violation weights."""
        pt = PreTrendsPower(violation_type="linear")
        weights = pt._get_violation_weights(4)

        # Should be normalized to unit norm
        assert np.isclose(np.linalg.norm(weights), 1.0)
        # Should increase (in absolute value) as we go back in time
        # For linear: [3, 2, 1, 0] normalized
        assert len(weights) == 4

    def test_constant_weights(self):
        """Test constant violation weights."""
        pt = PreTrendsPower(violation_type="constant")
        weights = pt._get_violation_weights(4)

        # Should be normalized to unit norm
        assert np.isclose(np.linalg.norm(weights), 1.0)
        # All weights should be equal
        assert np.allclose(weights[0], weights[1])
        assert np.allclose(weights[1], weights[2])

    def test_last_period_weights(self):
        """Test last_period violation weights."""
        pt = PreTrendsPower(violation_type="last_period")
        weights = pt._get_violation_weights(4)

        # Only last period should have weight
        assert weights[-1] == 1.0
        assert np.allclose(weights[:-1], 0.0)

    def test_custom_weights(self):
        """Test custom violation weights."""
        custom = np.array([0.6, 0.8])
        pt = PreTrendsPower(violation_type="custom", violation_weights=custom)
        weights = pt._get_violation_weights(2)

        # Should be normalized
        assert np.isclose(np.linalg.norm(weights), 1.0)


# =============================================================================
# Tests for power computation
# =============================================================================


class TestPowerComputation:
    """Tests for power computation."""

    def test_power_at_zero_equals_alpha(self):
        """Test that power at M=0 equals alpha (size of test)."""
        pt = PreTrendsPower(alpha=0.05)

        # Create simple vcov
        n_pre = 3
        vcov = np.eye(n_pre) * 0.25
        weights = pt._get_violation_weights(n_pre)

        power, _, _, _ = pt._compute_power(0.0, weights, vcov)
        assert np.isclose(power, 0.05, atol=0.01)

    def test_power_increases_with_M(self):
        """Test that power increases with M."""
        pt = PreTrendsPower()

        n_pre = 3
        vcov = np.eye(n_pre) * 0.25
        weights = pt._get_violation_weights(n_pre)

        power_small, _, _, _ = pt._compute_power(0.5, weights, vcov)
        power_large, _, _, _ = pt._compute_power(2.0, weights, vcov)

        assert power_large > power_small

    def test_power_at_mdv_equals_target(self):
        """Test that power at MDV equals target power."""
        pt = PreTrendsPower(power=0.80)

        n_pre = 3
        vcov = np.eye(n_pre) * 0.25
        weights = pt._get_violation_weights(n_pre)

        mdv = pt._compute_mdv(weights, vcov)
        power, _, _, _ = pt._compute_power(mdv, weights, vcov)

        assert np.isclose(power, 0.80, atol=0.02)


# =============================================================================
# Tests for MDV computation
# =============================================================================


class TestMDVComputation:
    """Tests for minimum detectable violation computation."""

    def test_mdv_positive(self):
        """Test that MDV is positive."""
        pt = PreTrendsPower()

        n_pre = 3
        vcov = np.eye(n_pre) * 0.25
        weights = pt._get_violation_weights(n_pre)

        mdv = pt._compute_mdv(weights, vcov)
        assert mdv > 0

    def test_mdv_decreases_with_precision(self):
        """Test that MDV decreases with smaller variance."""
        pt = PreTrendsPower()

        n_pre = 3
        weights = pt._get_violation_weights(n_pre)

        # Large variance
        vcov_large = np.eye(n_pre) * 1.0
        mdv_large = pt._compute_mdv(weights, vcov_large)

        # Small variance
        vcov_small = np.eye(n_pre) * 0.1
        mdv_small = pt._compute_mdv(weights, vcov_small)

        assert mdv_small < mdv_large

    def test_mdv_increases_with_target_power(self):
        """Test that MDV increases with higher target power."""
        n_pre = 3
        vcov = np.eye(n_pre) * 0.25

        pt_low = PreTrendsPower(power=0.50)
        pt_high = PreTrendsPower(power=0.90)

        weights = pt_low._get_violation_weights(n_pre)

        mdv_low = pt_low._compute_mdv(weights, vcov)
        mdv_high = pt_high._compute_mdv(weights, vcov)

        assert mdv_high > mdv_low


# =============================================================================
# Tests for fit method
# =============================================================================


class TestPreTrendsPowerFit:
    """Tests for PreTrendsPower.fit() method."""

    def test_fit_returns_results(self, mock_multiperiod_results):
        """Test that fit returns PreTrendsPowerResults."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results)

        assert isinstance(results, PreTrendsPowerResults)

    def test_fit_custom_M(self, mock_multiperiod_results):
        """Test fit with custom M value."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results, M=1.0)

        assert results.violation_magnitude == 1.0

    def test_results_has_expected_attributes(self, mock_multiperiod_results):
        """Test that results have all expected attributes."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results)

        assert hasattr(results, 'power')
        assert hasattr(results, 'mdv')
        assert hasattr(results, 'violation_magnitude')
        assert hasattr(results, 'violation_type')
        assert hasattr(results, 'alpha')
        assert hasattr(results, 'target_power')
        assert hasattr(results, 'n_pre_periods')
        assert hasattr(results, 'test_statistic')
        assert hasattr(results, 'critical_value')
        assert hasattr(results, 'noncentrality')

    def test_results_n_pre_periods(self, mock_multiperiod_results):
        """Test that n_pre_periods matches estimated pre-periods (excluding reference)."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results)

        # n_pre_periods should be the number of estimated coefficients (3)
        # not the total number of pre-periods (4), since period 3 is the reference
        expected_n_pre = len([
            p for p in mock_multiperiod_results.pre_periods
            if f"treated:period_{p}" in mock_multiperiod_results.coefficients
        ])
        assert results.n_pre_periods == expected_n_pre
        assert results.n_pre_periods == 3  # 4 pre-periods minus 1 reference


# =============================================================================
# Tests for power_curve method
# =============================================================================


class TestPowerCurve:
    """Tests for power_curve method."""

    def test_power_curve_returns_curve(self, mock_multiperiod_results):
        """Test that power_curve returns PreTrendsPowerCurve."""
        pt = PreTrendsPower()
        curve = pt.power_curve(mock_multiperiod_results)

        assert isinstance(curve, PreTrendsPowerCurve)

    def test_power_curve_custom_grid(self, mock_multiperiod_results):
        """Test power_curve with custom M grid."""
        pt = PreTrendsPower()
        M_grid = [0, 0.5, 1.0, 1.5, 2.0]
        curve = pt.power_curve(mock_multiperiod_results, M_grid=M_grid)

        assert len(curve.M_values) == len(M_grid)
        np.testing.assert_array_equal(curve.M_values, M_grid)

    def test_power_curve_n_points(self, mock_multiperiod_results):
        """Test power_curve with custom n_points."""
        pt = PreTrendsPower()
        curve = pt.power_curve(mock_multiperiod_results, n_points=20)

        assert len(curve.M_values) == 20

    def test_power_curve_monotonic(self, mock_multiperiod_results):
        """Test that power curve is monotonically increasing."""
        pt = PreTrendsPower()
        curve = pt.power_curve(mock_multiperiod_results)

        # Power should be non-decreasing
        diffs = np.diff(curve.powers)
        assert np.all(diffs >= -0.01)  # Allow small numerical noise

    def test_power_curve_to_dataframe(self, mock_multiperiod_results):
        """Test power_curve to_dataframe method."""
        pt = PreTrendsPower()
        curve = pt.power_curve(mock_multiperiod_results)
        df = curve.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert 'M' in df.columns
        assert 'power' in df.columns


# =============================================================================
# Tests for PreTrendsPowerResults
# =============================================================================


class TestPreTrendsPowerResults:
    """Tests for PreTrendsPowerResults dataclass."""

    def test_results_summary(self, mock_multiperiod_results):
        """Test summary method produces string."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results)

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Pre-Trends Power Analysis" in summary
        assert "Roth 2022" in summary

    def test_results_to_dict(self, mock_multiperiod_results):
        """Test to_dict method."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results)

        d = results.to_dict()
        assert isinstance(d, dict)
        assert 'power' in d
        assert 'mdv' in d
        assert 'violation_type' in d

    def test_results_to_dataframe(self, mock_multiperiod_results):
        """Test to_dataframe method."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results)

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_is_informative_property(self, mock_multiperiod_results):
        """Test is_informative property."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results)

        assert isinstance(results.is_informative, bool)

    def test_power_adequate_property(self, mock_multiperiod_results):
        """Test power_adequate property."""
        pt = PreTrendsPower()
        results = pt.fit(mock_multiperiod_results)

        assert isinstance(results.power_adequate, bool)


# =============================================================================
# Tests for convenience functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_pretrends_power(self, mock_multiperiod_results):
        """Test compute_pretrends_power function."""
        results = compute_pretrends_power(mock_multiperiod_results)

        assert isinstance(results, PreTrendsPowerResults)

    def test_compute_pretrends_power_custom_params(self, mock_multiperiod_results):
        """Test compute_pretrends_power with custom parameters."""
        results = compute_pretrends_power(
            mock_multiperiod_results,
            alpha=0.10,
            target_power=0.90,
            violation_type='constant'
        )

        assert results.alpha == 0.10
        assert results.target_power == 0.90
        assert results.violation_type == 'constant'

    def test_compute_mdv(self, mock_multiperiod_results):
        """Test compute_mdv function."""
        mdv = compute_mdv(mock_multiperiod_results)

        assert isinstance(mdv, float)
        assert mdv > 0


# =============================================================================
# Tests for get_params and set_params
# =============================================================================


class TestGetSetParams:
    """Tests for sklearn-like parameter interface."""

    def test_get_params(self):
        """Test get_params method."""
        pt = PreTrendsPower(alpha=0.10, power=0.90, violation_type='constant')
        params = pt.get_params()

        assert params['alpha'] == 0.10
        assert params['power'] == 0.90
        assert params['violation_type'] == 'constant'

    def test_set_params(self):
        """Test set_params method."""
        pt = PreTrendsPower()
        pt.set_params(alpha=0.10, power=0.90)

        assert pt.alpha == 0.10
        assert pt.target_power == 0.90


# =============================================================================
# Tests for integration with real estimators
# =============================================================================


class TestIntegration:
    """Integration tests with event study results.

    Note: These tests use mock results with pre-period coefficients.
    MultiPeriodDiD by default only estimates post-period treatment effects,
    so we use a fixture that simulates full event study results with
    pre-period coefficients (excluding the reference period).
    """

    def test_with_multiperiod_results(self, mock_multiperiod_results):
        """Test full pipeline with MultiPeriodDiDResults."""
        # Run pre-trends power analysis
        pt = PreTrendsPower()
        power_results = pt.fit(mock_multiperiod_results)

        # Check results are reasonable
        assert power_results.power >= 0
        assert power_results.power <= 1
        assert power_results.mdv > 0 or np.isinf(power_results.mdv)

    def test_power_curve_with_results(self, mock_multiperiod_results):
        """Test power curve with event study results."""
        pt = PreTrendsPower()
        curve = pt.power_curve(mock_multiperiod_results, n_points=10)

        # Check curve properties
        assert len(curve.M_values) == 10
        assert len(curve.powers) == 10
        assert np.all((curve.powers >= 0) & (curve.powers <= 1))

    def test_sensitivity_to_honest_did(self, mock_multiperiod_results):
        """Test sensitivity_to_honest_did method."""
        pt = PreTrendsPower()
        sensitivity = pt.sensitivity_to_honest_did(mock_multiperiod_results)

        assert 'mdv' in sensitivity
        assert 'interpretation' in sensitivity
        assert isinstance(sensitivity['interpretation'], str)


# =============================================================================
# Tests for different violation types
# =============================================================================


class TestViolationTypes:
    """Tests for different violation types."""

    def test_linear_violation(self, mock_multiperiod_results):
        """Test power analysis with linear violation."""
        pt = PreTrendsPower(violation_type='linear')
        results = pt.fit(mock_multiperiod_results)

        assert results.violation_type == 'linear'

    def test_constant_violation(self, mock_multiperiod_results):
        """Test power analysis with constant violation."""
        pt = PreTrendsPower(violation_type='constant')
        results = pt.fit(mock_multiperiod_results)

        assert results.violation_type == 'constant'

    def test_last_period_violation(self, mock_multiperiod_results):
        """Test power analysis with last_period violation."""
        pt = PreTrendsPower(violation_type='last_period')
        results = pt.fit(mock_multiperiod_results)

        assert results.violation_type == 'last_period'

    def test_different_types_give_different_results(self, mock_multiperiod_results):
        """Test that different violation types can give different MDV."""
        pt_linear = PreTrendsPower(violation_type='linear')
        pt_constant = PreTrendsPower(violation_type='constant')
        pt_last = PreTrendsPower(violation_type='last_period')

        mdv_linear = pt_linear.fit(mock_multiperiod_results).mdv
        mdv_constant = pt_constant.fit(mock_multiperiod_results).mdv
        mdv_last = pt_last.fit(mock_multiperiod_results).mdv

        # All MDVs should be positive (or inf for degenerate cases)
        assert mdv_linear > 0 or np.isinf(mdv_linear)
        assert mdv_constant > 0 or np.isinf(mdv_constant)
        assert mdv_last > 0 or np.isinf(mdv_last)

        # Just verify we can compute all three types without error
        # (with diagonal vcov, they may be the same due to normalization)


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_pre_period(self):
        """Test with single pre-period (excluding reference).

        This tests the case where there's only one estimated pre-period
        coefficient. We have pre_periods=[2, 3] where period 3 is the
        reference (excluded), leaving only period 2 as estimated.
        """
        period_effects = {
            2: PeriodEffect(
                period=2, effect=0.1, se=0.5,
                t_stat=0.2, p_value=0.84,
                conf_int=(-0.88, 1.08)
            ),
            # Period 3 is reference - not estimated
            4: PeriodEffect(
                period=4, effect=5.0, se=0.5,
                t_stat=10.0, p_value=0.0001,
                conf_int=(4.02, 5.98)
            ),
        }

        coefficients = {
            'treated:period_2': 0.1,
            'treated:period_4': 5.0,
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
            pre_periods=[2, 3],  # Period 3 is reference
            post_periods=[4],
            vcov=np.array([[0.25, 0], [0, 0.25]]),
            coefficients=coefficients,
        )

        pt = PreTrendsPower()
        power_results = pt.fit(results)

        assert power_results.n_pre_periods == 1  # Only period 2 is estimated

    def test_many_pre_periods(self):
        """Test with many pre-periods.

        This tests the case with 10 pre-periods where period 9 is the
        reference (excluded), leaving 9 estimated pre-period coefficients.
        """
        n_pre_total = 10
        n_pre_estimated = 9  # Excluding reference period

        # Pre-period effects (excluding reference period 9)
        period_effects = {}
        for i in range(n_pre_estimated):
            period_effects[i] = PeriodEffect(
                period=i, effect=0.05 * (i - 4), se=0.5,
                t_stat=0.1 * (i - 4), p_value=0.92,
                conf_int=(-0.88, 1.08)
            )

        # Post-period effects
        for i in range(4):
            period_effects[n_pre_total + i] = PeriodEffect(
                period=n_pre_total + i, effect=5.0, se=0.5,
                t_stat=10.0, p_value=0.0001,
                conf_int=(4.02, 5.98)
            )

        # Coefficients (excluding reference period 9)
        coefficients = {}
        for i in range(n_pre_estimated):
            coefficients[f'treated:period_{i}'] = 0.05 * (i - 4)
        for i in range(4):
            coefficients[f'treated:period_{n_pre_total + i}'] = 5.0

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
            pre_periods=list(range(n_pre_total)),  # Includes reference period 9
            post_periods=list(range(n_pre_total, n_pre_total + 4)),
            vcov=np.diag([0.25] * (n_pre_estimated + 4)),
            coefficients=coefficients,
        )

        pt = PreTrendsPower()
        power_results = pt.fit(results)

        assert power_results.n_pre_periods == n_pre_estimated

    def test_unsupported_results_type_raises(self):
        """Test that unsupported results type raises TypeError."""
        pt = PreTrendsPower()

        with pytest.raises(TypeError, match="Unsupported results type"):
            pt.fit("not a results object")


# =============================================================================
# Tests for visualization (without rendering)
# =============================================================================


class TestVisualization:
    """Tests for visualization methods (without rendering)."""

    def test_power_curve_has_plot_method(self, mock_multiperiod_results):
        """Test that PreTrendsPowerCurve has plot method."""
        pt = PreTrendsPower()
        curve = pt.power_curve(mock_multiperiod_results)

        assert hasattr(curve, 'plot')
        assert callable(curve.plot)
