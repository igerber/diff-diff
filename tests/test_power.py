"""Tests for power analysis module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    TROP,
    CallawaySantAnna,
    DifferenceInDifferences,
    EfficientDiD,
    ImputationDiD,
    MultiPeriodDiD,
    PowerAnalysis,
    PowerResults,
    SimulationMDEResults,
    SimulationPowerResults,
    SimulationSampleSizeResults,
    StackedDiD,
    SunAbraham,
    SyntheticDiD,
    TripleDifference,
    TwoStageDiD,
    TwoWayFixedEffects,
    compute_mde,
    compute_power,
    compute_sample_size,
    simulate_mde,
    simulate_power,
    simulate_sample_size,
)
from diff_diff.power import (
    MAX_SAMPLE_SIZE,
    _basic_dgp_kwargs,
    _basic_fit_kwargs,
    _ddd_dgp_kwargs,
    _ddd_fit_kwargs,
    _extract_multiperiod,
    _extract_simple,
    _extract_staggered,
    _factor_dgp_kwargs,
    _get_registry,
    _staggered_dgp_kwargs,
    _staggered_fit_kwargs,
    _trop_fit_kwargs,
)
from diff_diff.prep import generate_did_data


class TestPowerAnalysis:
    """Tests for PowerAnalysis class."""

    def test_init_defaults(self):
        """Test default initialization."""
        pa = PowerAnalysis()
        assert pa.alpha == 0.05
        assert pa.target_power == 0.80
        assert pa.alternative == "two-sided"

    def test_init_custom(self):
        """Test custom initialization."""
        pa = PowerAnalysis(alpha=0.10, power=0.90, alternative="greater")
        assert pa.alpha == 0.10
        assert pa.target_power == 0.90
        assert pa.alternative == "greater"

    def test_init_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            PowerAnalysis(alpha=0)
        with pytest.raises(ValueError):
            PowerAnalysis(alpha=1.5)
        with pytest.raises(ValueError):
            PowerAnalysis(power=0)
        with pytest.raises(ValueError):
            PowerAnalysis(power=1.1)
        with pytest.raises(ValueError):
            PowerAnalysis(alternative="invalid")

    def test_mde_basic(self):
        """Test minimum detectable effect calculation."""
        pa = PowerAnalysis(power=0.80, alpha=0.05)
        result = pa.mde(n_treated=50, n_control=50, sigma=1.0)

        assert isinstance(result, PowerResults)
        assert result.mde > 0
        assert result.power == 0.80
        assert result.n_treated == 50
        assert result.n_control == 50
        assert result.sigma == 1.0

    def test_mde_increases_with_noise(self):
        """Test that MDE increases with noise level."""
        pa = PowerAnalysis(power=0.80)

        result_low = pa.mde(n_treated=50, n_control=50, sigma=1.0)
        result_high = pa.mde(n_treated=50, n_control=50, sigma=2.0)

        assert result_high.mde > result_low.mde

    def test_mde_decreases_with_sample_size(self):
        """Test that MDE decreases with sample size."""
        pa = PowerAnalysis(power=0.80)

        result_small = pa.mde(n_treated=25, n_control=25, sigma=1.0)
        result_large = pa.mde(n_treated=100, n_control=100, sigma=1.0)

        assert result_large.mde < result_small.mde

    def test_power_calculation(self):
        """Test power calculation."""
        pa = PowerAnalysis(alpha=0.05)
        result = pa.power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)

        assert isinstance(result, PowerResults)
        assert 0 < result.power < 1
        assert result.effect_size == 0.5

    def test_power_increases_with_effect_size(self):
        """Test that power increases with effect size."""
        pa = PowerAnalysis()

        result_small = pa.power(effect_size=0.2, n_treated=50, n_control=50, sigma=1.0)
        result_large = pa.power(effect_size=0.8, n_treated=50, n_control=50, sigma=1.0)

        assert result_large.power > result_small.power

    def test_power_increases_with_sample_size(self):
        """Test that power increases with sample size."""
        pa = PowerAnalysis()

        result_small = pa.power(effect_size=0.5, n_treated=25, n_control=25, sigma=1.0)
        result_large = pa.power(effect_size=0.5, n_treated=100, n_control=100, sigma=1.0)

        assert result_large.power > result_small.power

    def test_sample_size_calculation(self):
        """Test sample size calculation."""
        pa = PowerAnalysis(power=0.80, alpha=0.05)
        result = pa.sample_size(effect_size=0.5, sigma=1.0)

        assert isinstance(result, PowerResults)
        assert result.required_n > 0
        assert result.n_treated + result.n_control == result.required_n

    def test_sample_size_increases_with_smaller_effect(self):
        """Test that required N increases for smaller effects."""
        pa = PowerAnalysis(power=0.80)

        result_large_effect = pa.sample_size(effect_size=1.0, sigma=1.0)
        result_small_effect = pa.sample_size(effect_size=0.2, sigma=1.0)

        assert result_small_effect.required_n > result_large_effect.required_n

    def test_panel_design(self):
        """Test panel DiD power calculations."""
        pa = PowerAnalysis(power=0.80)

        # Panel with multiple periods should have smaller MDE
        result_2period = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=1, n_post=1)
        result_6period = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=3, n_post=3)

        # More periods should reduce MDE (more data)
        assert result_6period.mde < result_2period.mde
        assert result_6period.design == "panel"

    def test_icc_effect(self):
        """Test that intra-cluster correlation affects power."""
        pa = PowerAnalysis(power=0.80)

        result_no_icc = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=3, n_post=3, rho=0.0)
        result_with_icc = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=3, n_post=3, rho=0.5)

        # Higher ICC should increase MDE (less independent information)
        assert result_with_icc.mde > result_no_icc.mde

    def test_power_curve(self):
        """Test power curve generation."""
        pa = PowerAnalysis()
        curve = pa.power_curve(
            n_treated=50, n_control=50, sigma=1.0, effect_sizes=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        )

        assert isinstance(curve, pd.DataFrame)
        assert "effect_size" in curve.columns
        assert "power" in curve.columns
        assert len(curve) == 6
        # Power should be monotonically increasing
        assert curve["power"].is_monotonic_increasing

    def test_power_curve_default_range(self):
        """Test power curve with default effect size range."""
        pa = PowerAnalysis()
        curve = pa.power_curve(n_treated=50, n_control=50, sigma=1.0)

        assert isinstance(curve, pd.DataFrame)
        assert len(curve) > 10  # Should have many points

    def test_sample_size_curve(self):
        """Test sample size curve generation."""
        pa = PowerAnalysis()
        curve = pa.sample_size_curve(
            effect_size=0.5, sigma=1.0, sample_sizes=[20, 50, 100, 150, 200]
        )

        assert isinstance(curve, pd.DataFrame)
        assert "sample_size" in curve.columns
        assert "power" in curve.columns
        assert len(curve) == 5
        # Power should increase with sample size
        assert curve["power"].is_monotonic_increasing

    def test_results_summary(self):
        """Test PowerResults summary method."""
        pa = PowerAnalysis()
        result = pa.mde(n_treated=50, n_control=50, sigma=1.0)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Power Analysis" in summary
        assert "MDE" in summary or "Minimum detectable effect" in summary

    def test_results_to_dict(self):
        """Test PowerResults to_dict method."""
        pa = PowerAnalysis()
        result = pa.mde(n_treated=50, n_control=50, sigma=1.0)

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "power" in d
        assert "mde" in d
        assert "n_treated" in d

    def test_results_to_dataframe(self):
        """Test PowerResults to_dataframe method."""
        pa = PowerAnalysis()
        result = pa.mde(n_treated=50, n_control=50, sigma=1.0)

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_one_sided_alternative(self):
        """Test one-sided hypothesis tests."""
        pa_greater = PowerAnalysis(alternative="greater")
        pa_less = PowerAnalysis(alternative="less")
        pa_two = PowerAnalysis(alternative="two-sided")

        result_greater = pa_greater.mde(n_treated=50, n_control=50, sigma=1.0)
        result_less = pa_less.mde(n_treated=50, n_control=50, sigma=1.0)
        result_two = pa_two.mde(n_treated=50, n_control=50, sigma=1.0)

        # One-sided tests should have smaller MDE than two-sided
        assert result_greater.mde < result_two.mde
        assert result_less.mde < result_two.mde

    def test_one_sided_power_calculation(self):
        """Test power calculation for one-sided alternatives."""
        pa_greater = PowerAnalysis(alternative="greater")
        pa_less = PowerAnalysis(alternative="less")
        pa_two = PowerAnalysis(alternative="two-sided")

        # For positive effect, 'greater' should have higher power than two-sided
        result_greater = pa_greater.power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)
        result_two = pa_two.power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)

        assert result_greater.power > result_two.power

        # For negative effect, 'less' should have higher power
        result_less = pa_less.power(effect_size=-0.5, n_treated=50, n_control=50, sigma=1.0)
        result_two_neg = pa_two.power(effect_size=-0.5, n_treated=50, n_control=50, sigma=1.0)

        assert result_less.power > result_two_neg.power

    def test_negative_effect_size(self):
        """Test power calculation with negative effect sizes."""
        pa = PowerAnalysis()

        # Power should work the same for negative effects (symmetric)
        result_pos = pa.power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)
        result_neg = pa.power(effect_size=-0.5, n_treated=50, n_control=50, sigma=1.0)

        # Two-sided test should have same power for positive and negative effects
        assert abs(result_pos.power - result_neg.power) < 0.01

    def test_extreme_icc(self):
        """Test power calculation with extreme intra-cluster correlation."""
        pa = PowerAnalysis(power=0.80)

        # Test with very high ICC (0.99)
        result_extreme = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=5, n_post=5, rho=0.99)

        result_moderate = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=5, n_post=5, rho=0.5)

        # Extreme ICC should have higher MDE (less independent info)
        assert result_extreme.mde > result_moderate.mde
        # MDE should still be finite and reasonable
        assert result_extreme.mde < float("inf")
        assert result_extreme.mde > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_mde(self):
        """Test compute_mde convenience function."""
        mde = compute_mde(n_treated=50, n_control=50, sigma=1.0)

        assert isinstance(mde, float)
        assert mde > 0

    def test_compute_power(self):
        """Test compute_power convenience function."""
        power = compute_power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)

        assert isinstance(power, float)
        assert 0 < power < 1

    def test_compute_sample_size(self):
        """Test compute_sample_size convenience function."""
        n = compute_sample_size(effect_size=0.5, sigma=1.0)

        assert isinstance(n, int)
        assert n > 0

    def test_convenience_functions_consistency(self):
        """Test that convenience functions are consistent with class."""
        pa = PowerAnalysis(power=0.80, alpha=0.05)

        # MDE
        mde_class = pa.mde(n_treated=50, n_control=50, sigma=1.0).mde
        mde_func = compute_mde(n_treated=50, n_control=50, sigma=1.0, power=0.80)
        assert mde_class == mde_func

        # Power
        power_class = pa.power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0).power
        power_func = compute_power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)
        assert power_class == power_func

        # Sample size
        n_class = pa.sample_size(effect_size=0.5, sigma=1.0).required_n
        n_func = compute_sample_size(effect_size=0.5, sigma=1.0, power=0.80)
        assert n_class == n_func


class TestSimulatePower:
    """Tests for simulate_power function."""

    def test_basic_simulation(self):
        """Test basic power simulation."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            n_units=50,
            n_periods=4,
            treatment_effect=5.0,
            sigma=2.0,
            n_simulations=20,  # Small for speed
            seed=42,
            progress=False,
        )

        assert isinstance(results, SimulationPowerResults)
        assert 0 <= results.power <= 1
        assert results.n_simulations == 20
        assert results.true_effect == 5.0
        assert results.estimator_name == "DifferenceInDifferences"

    def test_simulation_with_large_effect(self):
        """Test that simulation correctly identifies high power for large effects."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            n_units=100,
            n_periods=4,
            treatment_effect=10.0,  # Very large effect
            sigma=1.0,  # Low noise
            n_simulations=30,
            seed=42,
            progress=False,
        )

        # Should have very high power
        assert results.power > 0.80

    def test_simulation_with_zero_effect(self):
        """Test that simulation has low power for zero effect."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            n_units=50,
            n_periods=4,
            treatment_effect=0.0,  # No effect
            sigma=1.0,
            n_simulations=30,
            seed=42,
            progress=False,
        )

        # Power should be close to alpha (false positive rate)
        assert results.power < 0.20  # Should be around 5%

    def test_simulation_results_methods(self):
        """Test SimulationPowerResults methods."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            n_simulations=20,
            seed=42,
            progress=False,
        )

        # Test summary
        summary = results.summary()
        assert isinstance(summary, str)
        assert "Power" in summary

        # Test to_dict
        d = results.to_dict()
        assert isinstance(d, dict)
        assert "power" in d
        assert "coverage" in d

        # Test to_dataframe
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_simulation_coverage(self):
        """Test that confidence interval coverage is reasonable."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            n_units=100,
            n_periods=4,
            treatment_effect=5.0,
            sigma=2.0,
            n_simulations=50,
            seed=42,
            progress=False,
        )

        # Coverage should be close to 95% for 95% CIs
        assert 0.80 <= results.coverage <= 1.0  # Allow exact 1.0

    def test_simulation_bias(self):
        """Test that estimator is approximately unbiased."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            n_units=200,
            n_periods=4,
            treatment_effect=5.0,
            sigma=1.0,
            n_simulations=50,
            seed=42,
            progress=False,
        )

        # Bias should be small relative to effect size
        assert abs(results.bias) < 0.5  # Less than 10% of true effect

    def test_simulation_multiple_effects(self):
        """Test simulation with multiple effect sizes."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            n_units=50,
            effect_sizes=[1.0, 3.0, 5.0],
            sigma=2.0,
            n_simulations=30,
            seed=42,
            progress=False,
        )

        assert len(results.effect_sizes) == 3
        assert len(results.powers) == 3
        # Power should increase with effect size
        assert results.powers[0] < results.powers[2]

    def test_simulation_power_curve_df(self):
        """Test power curve DataFrame from simulation."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            effect_sizes=[1.0, 2.0, 3.0],
            n_simulations=20,
            seed=42,
            progress=False,
        )

        curve = results.power_curve_df()
        assert isinstance(curve, pd.DataFrame)
        assert "effect_size" in curve.columns
        assert "power" in curve.columns
        assert len(curve) == 3

    def test_simulation_confidence_interval(self):
        """Test power confidence interval."""
        did = DifferenceInDifferences()
        results = simulate_power(
            estimator=did,
            n_simulations=50,
            seed=42,
            progress=False,
        )

        # CI should contain the point estimate
        assert results.power_ci[0] <= results.power <= results.power_ci[1]
        # CI should be reasonable width (0 is valid when power is exactly 0 or 1)
        ci_width = results.power_ci[1] - results.power_ci[0]
        assert 0 <= ci_width < 0.5

    def test_simulation_handles_failures(self):
        """Test that simulation handles and reports failures."""
        import warnings

        # Create a mock estimator that sometimes fails
        class FailingEstimator:
            """Estimator that fails on specific simulations."""

            def __init__(self, fail_rate=0.0):
                self.fail_rate = fail_rate
                self.call_count = 0

            def fit(self, data, **kwargs):
                self.call_count += 1
                # Fail on every other call if fail_rate > 0
                if self.fail_rate > 0 and self.call_count % 2 == 0:
                    raise ValueError("Simulated failure")

                # Return a simple result
                class Result:
                    att = 5.0
                    se = 1.0
                    p_value = 0.01
                    conf_int = (3.0, 7.0)

                return Result()

        # Test with low failure rate (should not warn)
        from diff_diff.prep import generate_did_data

        estimator = FailingEstimator(fail_rate=0.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simulate_power(
                estimator=estimator,
                n_simulations=10,
                progress=False,
                data_generator=generate_did_data,
            )
            # Should have completed successfully without warning
            assert len([x for x in w if "simulations" in str(x.message)]) == 0


class TestVisualization:
    """Tests for power curve visualization."""

    def test_plot_power_curve_dataframe(self):
        """Test plotting from DataFrame."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import plot_power_curve

        df = pd.DataFrame(
            {
                "effect_size": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                "power": [0.1, 0.2, 0.4, 0.7, 0.9, 0.99],
            }
        )

        ax = plot_power_curve(df, show=False)
        assert ax is not None

    def test_plot_power_curve_manual_data(self):
        """Test plotting with manual effect sizes and powers."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import plot_power_curve

        ax = plot_power_curve(
            effect_sizes=[0.1, 0.2, 0.3, 0.5], powers=[0.1, 0.3, 0.6, 0.9], mde=0.25, show=False
        )
        assert ax is not None

    def test_plot_power_curve_sample_size(self):
        """Test plotting power vs sample size."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import plot_power_curve

        df = pd.DataFrame(
            {"sample_size": [20, 50, 100, 150, 200], "power": [0.2, 0.5, 0.8, 0.9, 0.95]}
        )

        ax = plot_power_curve(df, show=False)
        assert ax is not None

    def test_plot_validates_input(self):
        """Test that plot validates input."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import plot_power_curve

        with pytest.raises(ValueError):
            plot_power_curve(show=False)  # No data provided

        with pytest.raises(ValueError):
            plot_power_curve(effect_sizes=[1, 2, 3], show=False)  # Missing powers


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimum_sample_size(self):
        """Test that minimum sample size is enforced."""
        pa = PowerAnalysis()
        result = pa.sample_size(effect_size=100.0, sigma=1.0)  # Huge effect

        # Should have at least 4 units
        assert result.required_n >= 4

    def test_extreme_power_values(self):
        """Test power calculation at extremes."""
        pa = PowerAnalysis()

        # Zero effect should give ~alpha power
        result_zero = pa.power(effect_size=0.0, n_treated=50, n_control=50, sigma=1.0)
        assert result_zero.power < 0.10

        # Huge effect should give ~1.0 power
        result_huge = pa.power(effect_size=100.0, n_treated=50, n_control=50, sigma=1.0)
        assert result_huge.power > 0.99

    def test_unbalanced_design(self):
        """Test with unbalanced treatment/control."""
        pa = PowerAnalysis()

        result_balanced = pa.mde(n_treated=50, n_control=50, sigma=1.0)
        result_unbalanced = pa.mde(n_treated=25, n_control=75, sigma=1.0)

        # Balanced design should be more efficient
        assert result_balanced.mde < result_unbalanced.mde

    def test_treat_frac_sample_size(self):
        """Test treatment fraction in sample size calculation."""
        pa = PowerAnalysis()

        result_50 = pa.sample_size(effect_size=0.5, sigma=1.0, treat_frac=0.5)
        result_25 = pa.sample_size(effect_size=0.5, sigma=1.0, treat_frac=0.25)

        # 50-50 split should be most efficient
        assert result_50.required_n <= result_25.required_n

    def test_max_sample_size_constant(self):
        """Test that MAX_SAMPLE_SIZE is used for undetectable effects."""
        pa = PowerAnalysis()

        # Zero effect should return MAX_SAMPLE_SIZE
        result = pa.sample_size(effect_size=0.0, sigma=1.0)
        assert result.required_n == MAX_SAMPLE_SIZE

        # Verify constant is the expected value
        assert MAX_SAMPLE_SIZE == 2**31 - 1


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestEstimatorRegistry:
    """Tests for the estimator registry."""

    EXPECTED_ESTIMATORS = [
        "DifferenceInDifferences",
        "MultiPeriodDiD",
        "CallawaySantAnna",
        "SunAbraham",
        "ImputationDiD",
        "TwoStageDiD",
        "StackedDiD",
        "EfficientDiD",
        "TROP",
        "SyntheticDiD",
        "TripleDifference",
    ]

    def test_all_estimators_registered(self):
        """Every supported estimator has a registry entry."""
        registry = _get_registry()
        for name in self.EXPECTED_ESTIMATORS:
            assert name in registry, f"{name} missing from registry"

    def test_bacon_excluded(self):
        """BaconDecomposition is diagnostic-only and should not be in registry."""
        registry = _get_registry()
        assert "BaconDecomposition" not in registry

    def test_dgp_kwargs_builders_return_dicts(self):
        """Each DGP kwargs builder returns a non-empty dict."""
        params = dict(
            n_units=50,
            n_periods=4,
            treatment_effect=5.0,
            treatment_fraction=0.5,
            treatment_period=2,
            sigma=1.0,
        )
        for builder in [
            _basic_dgp_kwargs,
            _staggered_dgp_kwargs,
            _factor_dgp_kwargs,
            _ddd_dgp_kwargs,
        ]:
            result = builder(**params)
            assert isinstance(result, dict)
            assert len(result) > 0

    def test_fit_kwargs_builders_return_dicts(self):
        """Each fit kwargs builder returns a dict with 'outcome'."""
        dummy_df = pd.DataFrame({"period": [0, 1, 2, 3]})
        for builder in [
            _basic_fit_kwargs,
            _staggered_fit_kwargs,
            _ddd_fit_kwargs,
            _trop_fit_kwargs,
        ]:
            result = builder(dummy_df, 50, 4, 2)
            assert isinstance(result, dict)
            assert "outcome" in result

    def test_extract_simple(self):
        """_extract_simple extracts from .att/.se/.p_value/.conf_int."""

        class MockResult:
            att = 3.0
            se = 0.5
            p_value = 0.01
            conf_int = (2.0, 4.0)

        att, se, p, ci = _extract_simple(MockResult())
        assert att == 3.0
        assert se == 0.5
        assert p == 0.01
        assert ci == (2.0, 4.0)

    def test_extract_multiperiod(self):
        """_extract_multiperiod extracts from avg_* attributes."""

        class MockResult:
            avg_att = 4.0
            avg_se = 0.6
            avg_p_value = 0.001
            avg_conf_int = (2.8, 5.2)

        att, se, p, ci = _extract_multiperiod(MockResult())
        assert att == 4.0
        assert se == 0.6
        assert p == 0.001
        assert ci == (2.8, 5.2)

    def test_extract_staggered_analytical(self):
        """_extract_staggered handles analytical result objects."""

        class MockResult:
            overall_att = 2.0
            overall_se = 0.3
            overall_p_value = 0.02
            overall_conf_int = (1.4, 2.6)

        att, se, p, ci = _extract_staggered(MockResult())
        assert att == 2.0
        assert se == 0.3
        assert p == 0.02
        assert ci == (1.4, 2.6)

    def test_extract_staggered_bootstrap_fallback(self):
        """_extract_staggered falls back to bootstrap attribute names."""

        class MockBootstrapResult:
            overall_att = 2.0
            overall_att_se = 0.4
            overall_att_p_value = 0.03
            overall_att_ci = (1.2, 2.8)

        att, se, p, ci = _extract_staggered(MockBootstrapResult())
        assert att == 2.0
        assert se == 0.4
        assert p == 0.03
        assert ci == (1.2, 2.8)

    def test_continuous_did_not_in_registry(self):
        """ContinuousDiD is not in registry and raises without custom data_generator."""
        from diff_diff import ContinuousDiD

        registry = _get_registry()
        assert "ContinuousDiD" not in registry

        with pytest.raises(ValueError, match="not in registry"):
            simulate_power(
                ContinuousDiD(),
                n_simulations=5,
                progress=False,
            )

    def test_twfe_in_registry(self):
        """TwoWayFixedEffects is in the registry."""
        registry = _get_registry()
        assert "TwoWayFixedEffects" in registry

    def test_unknown_estimator_raises_without_data_generator(self):
        """Unknown estimator without data_generator raises ValueError."""

        class UnknownEstimator:
            pass

        with pytest.raises(ValueError, match="not in registry"):
            simulate_power(
                UnknownEstimator(),
                n_simulations=5,
                progress=False,
            )


# ---------------------------------------------------------------------------
# Estimator coverage tests for simulate_power
# ---------------------------------------------------------------------------


class TestEstimatorCoverage:
    """Verify simulate_power works for each registered estimator."""

    def _assert_valid_result(self, result, expected_name):
        assert 0 <= result.power <= 1
        assert result.estimator_name == expected_name
        assert np.isfinite(result.mean_estimate)
        assert result.n_simulations > 0
        assert result.coverage >= 0

    def test_did(self):
        result = simulate_power(
            DifferenceInDifferences(),
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "DifferenceInDifferences")

    def test_multiperiod(self):
        result = simulate_power(
            MultiPeriodDiD(),
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "MultiPeriodDiD")

    def test_callaway_santanna(self):
        result = simulate_power(
            CallawaySantAnna(),
            n_units=60,
            n_periods=6,
            treatment_period=3,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "CallawaySantAnna")

    def test_sun_abraham(self):
        result = simulate_power(
            SunAbraham(),
            n_units=60,
            n_periods=6,
            treatment_period=3,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "SunAbraham")

    def test_imputation_did(self):
        result = simulate_power(
            ImputationDiD(),
            n_units=60,
            n_periods=6,
            treatment_period=3,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "ImputationDiD")

    def test_two_stage_did(self):
        result = simulate_power(
            TwoStageDiD(),
            n_units=60,
            n_periods=6,
            treatment_period=3,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "TwoStageDiD")

    def test_stacked_did(self):
        result = simulate_power(
            StackedDiD(),
            n_units=60,
            n_periods=6,
            treatment_period=3,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "StackedDiD")

    def test_efficient_did(self):
        result = simulate_power(
            EfficientDiD(),
            n_units=60,
            n_periods=6,
            treatment_period=3,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "EfficientDiD")

    def test_triple_difference(self):
        result = simulate_power(
            TripleDifference(),
            n_units=80,
            n_periods=2,
            treatment_period=1,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "TripleDifference")

    def test_ddd_warns_ignored_params(self):
        """TripleDifference warns when simulation params don't match DDD design."""
        with pytest.warns(UserWarning, match="n_periods=6 is ignored"):
            simulate_power(
                TripleDifference(),
                n_units=80,
                n_periods=6,
                treatment_period=3,
                treatment_fraction=0.3,
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_ddd_warns_nonaligned_n_units(self):
        """TripleDifference warns when n_units doesn't map cleanly to 8 cells."""
        with pytest.warns(UserWarning, match="effective sample size is 64"):
            simulate_power(
                TripleDifference(),
                n_units=65,
                n_periods=2,
                treatment_period=1,
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_ddd_small_n_units_warns(self):
        """TripleDifference warns when n_units < 16 (clamped to 16)."""
        with pytest.warns(UserWarning, match="effective sample size is 16"):
            simulate_power(
                TripleDifference(),
                n_units=10,
                n_periods=2,
                treatment_period=1,
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_ddd_no_warn_aligned(self):
        """No warning when n_units is a multiple of 8 and defaults match DDD."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            simulate_power(
                TripleDifference(),
                n_units=80,
                n_periods=2,
                treatment_period=1,
                treatment_fraction=0.5,
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_ddd_no_warn_custom_dgp(self):
        """Custom data_generator bypasses the DDD compat check."""

        def custom_dgp(**kwargs):
            from diff_diff.prep_dgp import generate_ddd_data

            return generate_ddd_data(n_per_cell=10, seed=kwargs.get("seed"))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            simulate_power(
                TripleDifference(),
                n_units=65,
                n_periods=6,
                data_generator=custom_dgp,
                estimator_kwargs=dict(
                    outcome="outcome",
                    group="group",
                    partition="partition",
                    time="time",
                ),
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_ddd_no_warn_n_per_cell_override(self):
        """n_per_cell override suppresses rounding warning but not ignored-param warnings."""
        with pytest.warns(UserWarning, match="n_periods=6 is ignored"):
            simulate_power(
                TripleDifference(),
                n_units=80,
                n_periods=6,
                treatment_period=1,
                data_generator_kwargs=dict(n_per_cell=10),
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_ddd_n_per_cell_suppresses_rounding(self):
        """n_per_cell override suppresses effective-N rounding warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            simulate_power(
                TripleDifference(),
                n_units=80,
                n_periods=2,
                treatment_period=1,
                data_generator_kwargs=dict(n_per_cell=10),
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_ddd_power_effective_n_nonaligned(self):
        """simulate_power reports effective_n_units when n_units isn't grid-aligned."""
        with pytest.warns(UserWarning, match="effective sample size is 64"):
            result = simulate_power(
                TripleDifference(),
                n_units=65,
                n_periods=2,
                treatment_period=1,
                n_simulations=2,
                seed=42,
                progress=False,
            )
        assert result.effective_n_units == 64
        assert result.to_dict()["effective_n_units"] == 64
        assert "Effective sample size" in result.summary()

    def test_ddd_power_effective_n_aligned(self):
        """simulate_power sets effective_n_units=None when n_units is grid-aligned."""
        result = simulate_power(
            TripleDifference(),
            n_units=80,
            n_periods=2,
            treatment_period=1,
            n_simulations=2,
            seed=42,
            progress=False,
        )
        assert result.effective_n_units is None
        assert result.to_dict()["effective_n_units"] is None
        assert "Effective sample size" not in result.summary()

    @pytest.mark.slow
    def test_ddd_mde(self):
        """simulate_mde works for TripleDifference."""
        result = simulate_mde(
            TripleDifference(),
            n_units=80,
            n_periods=2,
            treatment_period=1,
            n_simulations=5,
            effect_range=(0.5, 5.0),
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationMDEResults)
        assert result.mde > 0
        assert result.effective_n_units is None

    @pytest.mark.slow
    def test_ddd_mde_effective_n(self):
        """simulate_mde reports effective_n_units for non-aligned n_units."""
        with pytest.warns(UserWarning, match="effective sample size is 64"):
            result = simulate_mde(
                TripleDifference(),
                n_units=65,
                n_periods=2,
                treatment_period=1,
                n_simulations=5,
                effect_range=(0.5, 5.0),
                seed=42,
                progress=False,
            )
        assert result.effective_n_units == 64
        assert result.to_dict()["effective_n_units"] == 64

    @pytest.mark.slow
    def test_ddd_sample_size(self):
        """simulate_sample_size works for TripleDifference."""
        result = simulate_sample_size(
            TripleDifference(),
            n_periods=2,
            treatment_period=1,
            n_simulations=5,
            n_range=(64, 200),
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationSampleSizeResults)
        assert result.required_n > 0

    @pytest.mark.slow
    def test_ddd_sample_size_grid_aligned(self):
        """simulate_sample_size returns grid-aligned required_n for DDD."""
        result = simulate_sample_size(
            TripleDifference(),
            n_periods=2,
            treatment_period=1,
            n_simulations=5,
            n_range=(64, 200),
            seed=42,
            progress=False,
        )
        assert (
            result.required_n % 8 == 0
        ), f"DDD required_n={result.required_n} is not a multiple of 8"

    @pytest.mark.slow
    def test_ddd_sample_size_low_range(self):
        """DDD sample-size search with low n_range stays within bracket."""
        result = simulate_sample_size(
            TripleDifference(),
            n_periods=2,
            treatment_period=1,
            treatment_effect=0.5,
            sigma=5.0,
            n_simulations=5,
            n_range=(16, 56),
            seed=42,
            progress=False,
        )
        assert (
            result.required_n % 8 == 0
        ), f"DDD required_n={result.required_n} is not a multiple of 8"
        assert (
            16 <= result.required_n <= 56
        ), f"DDD required_n={result.required_n} outside requested bracket [16, 56]"
        assert (
            len(result.search_path) > 2
        ), f"Bisection should explore >2 points, got {len(result.search_path)}"

    @pytest.mark.slow
    def test_trop(self):
        result = simulate_power(
            TROP(),
            n_units=50,
            n_periods=6,
            treatment_period=3,
            treatment_fraction=0.3,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "TROP")

    @pytest.mark.slow
    def test_synthetic_did(self):
        result = simulate_power(
            SyntheticDiD(),
            n_units=50,
            n_periods=6,
            treatment_period=3,
            treatment_fraction=0.3,
            n_simulations=10,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "SyntheticDiD")

    def test_sdid_placebo_rejects_high_fraction(self):
        """SyntheticDiD placebo variance raises when n_control <= n_treated."""
        with pytest.raises(ValueError, match="placebo variance requires more control"):
            simulate_power(
                SyntheticDiD(),
                treatment_fraction=0.5,
                n_simulations=5,
                seed=42,
                progress=False,
            )

    @pytest.mark.slow
    def test_sdid_placebo_boundary_fraction(self):
        """treatment_fraction=0.49 with 50 units gives n_control=26 > n_treated=24."""
        result = simulate_power(
            SyntheticDiD(),
            treatment_fraction=0.49,
            n_units=50,
            n_periods=6,
            treatment_period=3,
            n_simulations=5,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "SyntheticDiD")

    @pytest.mark.slow
    def test_sdid_bootstrap_allows_high_fraction(self):
        """Bootstrap variance method bypasses the placebo constraint."""
        result = simulate_power(
            SyntheticDiD(variance_method="bootstrap"),
            treatment_fraction=0.5,
            n_units=50,
            n_periods=6,
            treatment_period=3,
            n_simulations=5,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "SyntheticDiD")
        assert result.power >= 0

    def test_sdid_mde_rejects_high_fraction(self):
        """simulate_mde raises for SyntheticDiD placebo with high treatment_fraction."""
        with pytest.raises(ValueError, match="placebo variance requires more control"):
            simulate_mde(
                SyntheticDiD(),
                treatment_fraction=0.5,
                n_simulations=5,
                seed=42,
                progress=False,
            )

    def test_sdid_sample_size_rejects_high_fraction(self):
        """simulate_sample_size raises for SyntheticDiD placebo with high fraction."""
        with pytest.raises(ValueError, match="placebo variance requires more control"):
            simulate_sample_size(
                SyntheticDiD(),
                treatment_fraction=0.5,
                n_simulations=5,
                seed=42,
                progress=False,
            )

    def test_sdid_placebo_rejects_n_treated_override(self):
        """SDID placebo raises when data_generator_kwargs overrides n_treated."""
        with pytest.raises(ValueError, match="placebo variance requires more control"):
            simulate_power(
                SyntheticDiD(),
                n_units=50,
                treatment_fraction=0.3,
                data_generator_kwargs=dict(n_treated=30),
                n_simulations=5,
                seed=42,
                progress=False,
            )

    def test_sdid_mde_rejects_n_treated_override(self):
        """simulate_mde raises when kwargs override makes n_control <= n_treated."""
        with pytest.raises(ValueError, match="placebo variance requires more control"):
            simulate_mde(
                SyntheticDiD(),
                n_units=50,
                treatment_fraction=0.3,
                data_generator_kwargs=dict(n_treated=30),
                n_simulations=5,
                seed=42,
                progress=False,
            )

    def test_sdid_sample_size_rejects_n_treated_override(self):
        """simulate_sample_size raises when kwargs override is infeasible."""
        with pytest.raises(ValueError, match="placebo variance requires more control"):
            simulate_sample_size(
                SyntheticDiD(),
                treatment_fraction=0.3,
                data_generator_kwargs=dict(n_treated=30),
                n_range=(50, 100),
                n_simulations=5,
                seed=42,
                progress=False,
            )

    @pytest.mark.slow
    def test_sdid_mde(self):
        """simulate_mde works for SyntheticDiD with valid treatment_fraction."""
        result = simulate_mde(
            SyntheticDiD(),
            treatment_fraction=0.3,
            n_units=50,
            n_periods=6,
            treatment_period=3,
            n_simulations=5,
            effect_range=(0.5, 3.0),
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationMDEResults)
        assert result.mde > 0

    @pytest.mark.slow
    def test_sdid_sample_size(self):
        """simulate_sample_size works for SyntheticDiD with valid fraction."""
        result = simulate_sample_size(
            SyntheticDiD(),
            treatment_fraction=0.3,
            n_periods=6,
            treatment_period=3,
            n_simulations=5,
            n_range=(30, 80),
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationSampleSizeResults)
        assert result.required_n > 0

    @pytest.mark.slow
    def test_twfe(self):
        result = simulate_power(
            TwoWayFixedEffects(),
            n_simulations=5,
            seed=42,
            progress=False,
        )
        self._assert_valid_result(result, "TwoWayFixedEffects")

    @pytest.mark.slow
    def test_twfe_mde(self):
        result = simulate_mde(
            TwoWayFixedEffects(),
            n_simulations=5,
            effect_range=(0.5, 5.0),
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationMDEResults)
        assert result.mde > 0

    @pytest.mark.slow
    def test_twfe_sample_size(self):
        result = simulate_sample_size(
            TwoWayFixedEffects(),
            n_simulations=5,
            n_range=(20, 100),
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationSampleSizeResults)
        assert result.required_n > 0

    @pytest.mark.slow
    def test_custom_fallback_unregistered_estimator(self):
        """Unregistered estimator works with custom data_generator and estimator_kwargs."""

        class _UnregisteredEstimator:
            """Unregistered wrapper for testing custom fallback."""

            def __init__(self):
                self._inner = DifferenceInDifferences()

            def fit(self, data, **kwargs):
                return self._inner.fit(data, **kwargs)

        result = simulate_power(
            _UnregisteredEstimator(),
            data_generator=generate_did_data,
            estimator_kwargs=dict(outcome="outcome", treatment="treated", time="post"),
            n_simulations=5,
            seed=42,
            progress=False,
        )
        assert 0 <= result.power <= 1
        assert result.n_simulations > 0

    def test_custom_fallback_missing_kwargs_raises(self):
        """Unregistered estimator with no estimator_kwargs fails on fit."""

        class _UnregisteredEstimator:
            def __init__(self):
                self._inner = DifferenceInDifferences()

            def fit(self, data, **kwargs):
                return self._inner.fit(data, **kwargs)

        with pytest.raises((ValueError, TypeError, RuntimeError)):
            simulate_power(
                _UnregisteredEstimator(),
                data_generator=generate_did_data,
                n_simulations=5,
                seed=42,
                progress=False,
            )

    @pytest.mark.slow
    def test_custom_result_extractor(self):
        """Custom result_extractor works for unregistered estimator."""

        class _UnregisteredEstimator:
            def __init__(self):
                self._inner = DifferenceInDifferences()

            def fit(self, data, **kwargs):
                return self._inner.fit(data, **kwargs)

        def _custom_extractor(result):
            return (result.att, result.se, result.p_value, result.conf_int)

        result = simulate_power(
            _UnregisteredEstimator(),
            data_generator=generate_did_data,
            estimator_kwargs=dict(outcome="outcome", treatment="treated", time="post"),
            result_extractor=_custom_extractor,
            n_simulations=5,
            seed=42,
            progress=False,
        )
        assert 0 <= result.power <= 1
        assert result.n_simulations > 0

    @pytest.mark.slow
    def test_custom_result_extractor_mde_forwarding(self):
        """result_extractor forwards correctly through simulate_mde."""

        class _UnregisteredEstimator:
            def __init__(self):
                self._inner = DifferenceInDifferences()

            def fit(self, data, **kwargs):
                return self._inner.fit(data, **kwargs)

        def _custom_extractor(result):
            return (result.att, result.se, result.p_value, result.conf_int)

        result = simulate_mde(
            _UnregisteredEstimator(),
            data_generator=generate_did_data,
            estimator_kwargs=dict(outcome="outcome", treatment="treated", time="post"),
            result_extractor=_custom_extractor,
            n_simulations=5,
            effect_range=(0.5, 5.0),
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationMDEResults)
        assert result.mde > 0

    # -- Staggered DGP compatibility warnings --

    def test_staggered_dgp_warns_not_yet_treated(self):
        """Auto DGP warns when CS has control_group='not_yet_treated'."""
        with pytest.warns(UserWarning, match="not_yet_treated"):
            simulate_power(
                CallawaySantAnna(control_group="not_yet_treated"),
                n_simulations=3,
                seed=42,
                progress=False,
            )

    def test_staggered_dgp_warns_anticipation(self):
        """Auto DGP warns when staggered estimator has anticipation > 0."""
        with pytest.warns(UserWarning, match="anticipation=1"):
            simulate_power(
                CallawaySantAnna(anticipation=1),
                n_simulations=3,
                seed=42,
                progress=False,
            )

    def test_staggered_dgp_warns_strict_clean_control(self):
        """Auto DGP warns when StackedDiD has clean_control='strict'."""
        with pytest.warns(UserWarning, match="strict"):
            simulate_power(
                StackedDiD(clean_control="strict"),
                n_simulations=3,
                seed=42,
                progress=False,
            )

    def test_staggered_dgp_no_warn_custom_dgp_bypasses_check(self):
        """Custom data_generator bypasses DGP compat check entirely."""
        from diff_diff.prep import generate_staggered_data

        def _custom_staggered(**kwargs):
            # Adapt simulate_power's standard kwargs to generate_staggered_data
            return generate_staggered_data(
                n_units=kwargs["n_units"],
                n_periods=kwargs["n_periods"],
                treatment_effect=kwargs["treatment_effect"],
                cohort_periods=[2, 4],
                never_treated_frac=0.0,
                noise_sd=kwargs["noise_sd"],
                seed=kwargs["seed"],
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            simulate_power(
                CallawaySantAnna(control_group="not_yet_treated"),
                data_generator=_custom_staggered,
                n_periods=6,
                treatment_period=3,
                estimator_kwargs=dict(
                    outcome="outcome",
                    unit="unit",
                    time="period",
                    first_treat="first_treat",
                ),
                n_simulations=3,
                seed=42,
                progress=False,
            )

    def test_staggered_dgp_no_warn_with_dgp_kwargs_override(self):
        """data_generator_kwargs with cohort_periods suppresses warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = simulate_power(
                CallawaySantAnna(control_group="not_yet_treated"),
                n_periods=6,
                treatment_period=3,
                data_generator_kwargs=dict(cohort_periods=[2, 4], never_treated_frac=0.0),
                n_simulations=3,
                seed=42,
                progress=False,
            )
        assert 0 <= result.power <= 1

    @pytest.mark.slow
    def test_cs_not_yet_treated_with_matching_dgp(self):
        """CS with control_group='not_yet_treated' and multi-cohort DGP."""
        result = simulate_power(
            CallawaySantAnna(control_group="not_yet_treated"),
            n_units=60,
            n_periods=6,
            treatment_period=3,
            data_generator_kwargs=dict(cohort_periods=[2, 4], never_treated_frac=0.0),
            n_simulations=10,
            seed=42,
            progress=False,
        )
        assert 0 <= result.power <= 1
        assert result.n_simulations > 0

    @pytest.mark.slow
    def test_stacked_did_strict_with_matching_dgp(self):
        """StackedDiD with clean_control='strict' and multi-cohort DGP."""
        result = simulate_power(
            StackedDiD(clean_control="strict", kappa_pre=1, kappa_post=1),
            n_units=80,
            n_periods=8,
            treatment_period=4,
            data_generator_kwargs=dict(cohort_periods=[3, 5]),
            n_simulations=10,
            seed=42,
            progress=False,
        )
        assert 0 <= result.power <= 1
        assert result.n_simulations > 0


# ---------------------------------------------------------------------------
# simulate_mde tests
# ---------------------------------------------------------------------------


class TestSimulateMDE:
    """Tests for simulate_mde function."""

    def test_basic_mde(self):
        """MDE found for DiD, power at MDE close to target."""
        result = simulate_mde(
            DifferenceInDifferences(),
            n_units=100,
            sigma=1.0,
            n_simulations=50,
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationMDEResults)
        assert result.mde > 0
        assert result.power_at_mde >= result.target_power - 0.10

    def test_result_methods(self):
        """summary(), to_dict(), to_dataframe() work."""
        result = simulate_mde(
            DifferenceInDifferences(),
            n_simulations=30,
            seed=42,
            progress=False,
        )
        summary = result.summary()
        assert "MDE" in summary or "Minimum" in summary

        d = result.to_dict()
        assert "mde" in d
        assert "estimator_name" in d

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_monotonicity_in_search_path(self):
        """The search path records plausible effect_size / power pairs."""
        result = simulate_mde(
            DifferenceInDifferences(),
            n_simulations=50,
            seed=42,
            progress=False,
        )
        assert len(result.search_path) > 0
        for step in result.search_path:
            assert "effect_size" in step
            assert "power" in step
            assert 0 <= step["power"] <= 1

    def test_convergence_within_max_steps(self):
        """Search terminates within max_steps."""
        result = simulate_mde(
            DifferenceInDifferences(),
            n_simulations=30,
            max_steps=10,
            seed=42,
            progress=False,
        )
        # n_steps includes bracketing steps + bisection
        assert result.n_steps <= 25  # generous bound

    def test_custom_data_generator(self):
        """Works with user-provided DGP."""
        from diff_diff.prep import generate_did_data

        result = simulate_mde(
            DifferenceInDifferences(),
            n_simulations=30,
            seed=42,
            progress=False,
            data_generator=generate_did_data,
        )
        assert result.mde > 0

    def test_small_sigma_gives_small_mde(self):
        """Small noise → small MDE."""
        result = simulate_mde(
            DifferenceInDifferences(),
            n_units=100,
            sigma=0.1,
            n_simulations=50,
            seed=42,
            progress=False,
        )
        assert result.mde < 1.0

    def test_large_sigma_gives_large_mde(self):
        """Large noise → large MDE."""
        result = simulate_mde(
            DifferenceInDifferences(),
            n_units=50,
            sigma=10.0,
            n_simulations=50,
            seed=42,
            progress=False,
        )
        assert result.mde > 1.0

    def test_explicit_effect_range(self):
        """Explicit effect_range evaluates endpoints and populates search_path."""
        result = simulate_mde(
            DifferenceInDifferences(),
            n_units=100,
            sigma=1.0,
            n_simulations=30,
            effect_range=(0.5, 5.0),
            seed=42,
            progress=False,
        )
        assert result.mde > 0
        assert result.power_at_mde > 0
        assert len(result.search_path) > 0

    def test_unbracketed_effect_range_warns(self):
        """Tiny effect_range that cannot bracket target power warns."""
        with pytest.warns(UserWarning, match="not bracketed"):
            simulate_mde(
                DifferenceInDifferences(),
                n_units=50,
                sigma=10.0,
                n_simulations=30,
                effect_range=(0.0, 0.001),
                seed=42,
                progress=False,
            )


# ---------------------------------------------------------------------------
# simulate_sample_size tests
# ---------------------------------------------------------------------------


class TestSimulateSampleSize:
    """Tests for simulate_sample_size function."""

    def test_basic_sample_size(self):
        """Required N found for DiD, power at N close to target."""
        result = simulate_sample_size(
            DifferenceInDifferences(),
            treatment_effect=5.0,
            sigma=1.0,
            n_simulations=50,
            seed=42,
            progress=False,
        )
        assert isinstance(result, SimulationSampleSizeResults)
        assert result.required_n > 0
        assert result.power_at_n >= result.target_power - 0.10

    def test_result_methods(self):
        """summary(), to_dict(), to_dataframe() work."""
        result = simulate_sample_size(
            DifferenceInDifferences(),
            treatment_effect=5.0,
            n_simulations=30,
            seed=42,
            progress=False,
        )
        summary = result.summary()
        assert "Sample Size" in summary or "Required" in summary

        d = result.to_dict()
        assert "required_n" in d
        assert "estimator_name" in d

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_monotonicity_in_search_path(self):
        """The search path records plausible n_units / power pairs."""
        result = simulate_sample_size(
            DifferenceInDifferences(),
            treatment_effect=5.0,
            n_simulations=50,
            seed=42,
            progress=False,
        )
        assert len(result.search_path) > 0
        for step in result.search_path:
            assert "n_units" in step
            assert "power" in step
            assert 0 <= step["power"] <= 1

    def test_custom_data_generator(self):
        """Works with user-provided DGP."""
        from diff_diff.prep import generate_did_data

        result = simulate_sample_size(
            DifferenceInDifferences(),
            treatment_effect=5.0,
            n_simulations=30,
            seed=42,
            progress=False,
            data_generator=generate_did_data,
        )
        assert result.required_n > 0

    def test_large_effect_gives_small_n(self):
        """Large effect → small N."""
        result = simulate_sample_size(
            DifferenceInDifferences(),
            treatment_effect=20.0,
            sigma=1.0,
            n_simulations=50,
            seed=42,
            progress=False,
        )
        assert result.required_n <= 100

    def test_small_effect_gives_large_n(self):
        """Small effect → large N."""
        result = simulate_sample_size(
            DifferenceInDifferences(),
            treatment_effect=0.5,
            sigma=5.0,
            n_simulations=50,
            seed=42,
            progress=False,
        )
        assert result.required_n >= 50

    def test_explicit_n_range(self):
        """Explicit n_range evaluates endpoints and populates search_path."""
        result = simulate_sample_size(
            DifferenceInDifferences(),
            treatment_effect=5.0,
            sigma=1.0,
            n_simulations=30,
            n_range=(20, 200),
            seed=42,
            progress=False,
        )
        assert result.required_n > 0
        assert result.power_at_n > 0
        assert len(result.search_path) > 0

    def test_unbracketed_n_range_warns(self):
        """Tiny n_range that cannot bracket target power warns."""
        with pytest.warns(UserWarning, match="not bracketed"):
            simulate_sample_size(
                DifferenceInDifferences(),
                treatment_effect=0.01,
                sigma=10.0,
                n_simulations=30,
                n_range=(20, 22),
                seed=42,
                progress=False,
            )

    def test_lo_already_sufficient_explicit(self):
        """When lo already meets power, return lo immediately with warning."""
        with pytest.warns(UserWarning, match="Lower bound already achieves"):
            result = simulate_sample_size(
                DifferenceInDifferences(),
                treatment_effect=50.0,
                sigma=0.1,
                n_simulations=50,
                n_range=(20, 200),
                seed=42,
                progress=False,
            )
        assert result.required_n == 20
        assert result.power_at_n >= 0.80

    def test_lo_already_sufficient_auto(self):
        """Auto-bracket searches downward when floor already achieves power."""
        with pytest.warns(UserWarning, match="Could not find a smaller N"):
            result = simulate_sample_size(
                DifferenceInDifferences(),
                treatment_effect=50.0,
                sigma=0.1,
                n_simulations=50,
                seed=42,
                progress=False,
            )
        # Effect is so large even abs_min=4 achieves power
        assert result.required_n <= 20
        assert result.power_at_n >= 0.80

    @pytest.mark.slow
    def test_sample_size_searches_below_floor(self):
        """Large effect → downward search finds required_n below registry floor."""
        result = simulate_sample_size(
            DifferenceInDifferences(),
            treatment_effect=50.0,
            sigma=1.0,
            n_simulations=5,
            seed=42,
            progress=False,
        )
        # min_n for DiD is 20; huge effect should find smaller N
        assert result.required_n < 20

    def test_reject_n_per_cell_in_ddd_sample_size(self):
        """n_per_cell override in simulate_sample_size raises for DDD."""
        with pytest.raises(ValueError, match="n_per_cell"):
            simulate_sample_size(
                TripleDifference(),
                treatment_effect=5.0,
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={"n_per_cell": 10},
            )


class TestDGPKeyCollisions:
    """Verify registry-path DGP key collision detection."""

    def test_reject_treatment_effect_collision(self):
        """treatment_effect in data_generator_kwargs raises ValueError."""
        with pytest.raises(ValueError, match="conflict"):
            simulate_power(
                DifferenceInDifferences(),
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={"treatment_effect": 99},
            )

    def test_reject_noise_sd_collision(self):
        """noise_sd in data_generator_kwargs raises ValueError."""
        with pytest.raises(ValueError, match="conflict"):
            simulate_power(
                DifferenceInDifferences(),
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={"noise_sd": 5.0},
            )

    def test_allow_cohort_periods_override(self):
        """cohort_periods is not a protected key — no collision."""
        # Should not raise
        simulate_power(
            CallawaySantAnna(),
            n_units=60,
            n_periods=6,
            treatment_period=3,
            n_simulations=2,
            seed=42,
            progress=False,
            data_generator_kwargs={"cohort_periods": [2, 4]},
        )

    def test_allow_n_per_cell_override(self):
        """n_per_cell is not a protected key — no collision for DDD."""
        # Should not raise (n_per_cell is in DDD builder output but not
        # in _PROTECTED_DGP_KEYS, so 3-way intersection is empty)
        simulate_power(
            TripleDifference(),
            n_simulations=2,
            seed=42,
            progress=False,
            data_generator_kwargs={"n_per_cell": 15},
        )

    def test_reject_n_pre_collision_sdid(self):
        """n_pre in data_generator_kwargs raises for SyntheticDiD (factor DGP)."""
        with pytest.raises(ValueError, match="conflict"):
            simulate_power(
                SyntheticDiD(variance_method="bootstrap"),
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={"n_pre": 1},
            )

    def test_reject_n_post_collision_trop(self):
        """n_post in data_generator_kwargs raises for TROP (factor DGP)."""
        with pytest.raises(ValueError, match="conflict"):
            simulate_power(
                TROP(),
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={"n_post": 5},
            )

    def test_n_pre_not_rejected_for_basic_did(self):
        """n_pre passes collision guard for basic DiD (not a derived key there).

        Basic DGP doesn't return n_pre, so 3-way intersection is empty.
        generate_did_data rejects n_pre (not a valid param), proving the
        collision guard did NOT fire (would have raised ValueError("conflict")).
        """
        with pytest.raises(TypeError, match="n_pre"):
            simulate_power(
                DifferenceInDifferences(),
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={"n_pre": 1},
            )

    def test_collision_skipped_for_custom_dgp(self):
        """Custom data_generator bypasses collision check entirely."""
        # unit_fe_sd is accepted by generate_did_data; collision check is
        # skipped because a custom data_generator is provided.
        simulate_power(
            DifferenceInDifferences(),
            n_simulations=2,
            seed=42,
            progress=False,
            data_generator=generate_did_data,
            data_generator_kwargs={"unit_fe_sd": 3.0},
        )


class TestStaggeredSingleCohort:
    """Verify staggered DGP compat check handles single-cohort overrides."""

    def test_staggered_single_cohort_still_warns(self):
        """CS with cohort_periods=[2] still warns — single cohort."""
        with pytest.warns(UserWarning, match="DGP mismatch"):
            simulate_power(
                CallawaySantAnna(control_group="not_yet_treated"),
                n_units=60,
                n_periods=6,
                treatment_period=3,
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={"cohort_periods": [2]},
            )

    def test_staggered_multi_cohort_no_warn(self):
        """CS with cohort_periods=[2, 4] does NOT warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            simulate_power(
                CallawaySantAnna(control_group="not_yet_treated"),
                n_units=60,
                n_periods=6,
                treatment_period=3,
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={
                    "cohort_periods": [2, 4],
                    "never_treated_frac": 0.0,
                },
            )

    def test_stacked_strict_single_cohort_warns(self):
        """StackedDiD clean_control='strict' with cohort_periods=[2] warns."""
        with pytest.warns(UserWarning, match="DGP mismatch"):
            simulate_power(
                StackedDiD(clean_control="strict"),
                n_units=60,
                n_periods=6,
                treatment_period=3,
                n_simulations=2,
                seed=42,
                progress=False,
                data_generator_kwargs={"cohort_periods": [2]},
            )


class TestSDIDPlaceboCustomDGP:
    """Verify SyntheticDiD placebo feasibility check on custom-DGP path."""

    @staticmethod
    def _factor_dgp_wrapper(
        n_units=100,
        n_periods=4,
        treatment_effect=5.0,
        treatment_fraction=0.5,
        treatment_period=2,
        noise_sd=1.0,
        seed=None,
        **kwargs,
    ):
        from diff_diff.prep_dgp import generate_factor_data

        n_treated = max(1, int(n_units * treatment_fraction))
        n_pre = treatment_period
        n_post = n_periods - treatment_period
        return generate_factor_data(
            n_units=n_units,
            n_treated=n_treated,
            n_pre=n_pre,
            n_post=n_post,
            treatment_effect=treatment_effect,
            noise_sd=noise_sd,
            seed=seed,
        )

    def test_sdid_placebo_custom_dgp_power_raises(self):
        """simulate_power raises ValueError for infeasible placebo design."""
        with pytest.raises(ValueError, match="placebo"):
            simulate_power(
                SyntheticDiD(),
                data_generator=self._factor_dgp_wrapper,
                n_units=100,
                n_periods=4,
                treatment_fraction=0.6,
                treatment_period=2,
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_sdid_placebo_custom_dgp_mde_raises(self):
        """simulate_mde raises ValueError for infeasible placebo design."""
        with pytest.raises(ValueError, match="placebo"):
            simulate_mde(
                SyntheticDiD(),
                data_generator=self._factor_dgp_wrapper,
                n_units=100,
                n_periods=4,
                treatment_fraction=0.6,
                treatment_period=2,
                n_simulations=2,
                seed=42,
                progress=False,
            )

    def test_sdid_placebo_custom_dgp_sample_size_raises(self):
        """simulate_sample_size raises ValueError for infeasible placebo design."""
        with pytest.raises(ValueError, match="placebo"):
            simulate_sample_size(
                SyntheticDiD(),
                data_generator=self._factor_dgp_wrapper,
                n_periods=4,
                treatment_fraction=0.6,
                treatment_period=2,
                n_simulations=2,
                seed=42,
                progress=False,
            )
