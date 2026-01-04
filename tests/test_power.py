"""Tests for power analysis module."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DifferenceInDifferences,
    PowerAnalysis,
    PowerResults,
    SimulationPowerResults,
    compute_mde,
    compute_power,
    compute_sample_size,
    simulate_power,
)


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
        result = pa.power(
            effect_size=0.5,
            n_treated=50,
            n_control=50,
            sigma=1.0
        )

        assert isinstance(result, PowerResults)
        assert 0 < result.power < 1
        assert result.effect_size == 0.5

    def test_power_increases_with_effect_size(self):
        """Test that power increases with effect size."""
        pa = PowerAnalysis()

        result_small = pa.power(
            effect_size=0.2, n_treated=50, n_control=50, sigma=1.0
        )
        result_large = pa.power(
            effect_size=0.8, n_treated=50, n_control=50, sigma=1.0
        )

        assert result_large.power > result_small.power

    def test_power_increases_with_sample_size(self):
        """Test that power increases with sample size."""
        pa = PowerAnalysis()

        result_small = pa.power(
            effect_size=0.5, n_treated=25, n_control=25, sigma=1.0
        )
        result_large = pa.power(
            effect_size=0.5, n_treated=100, n_control=100, sigma=1.0
        )

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
        result_2period = pa.mde(
            n_treated=50, n_control=50, sigma=1.0,
            n_pre=1, n_post=1
        )
        result_6period = pa.mde(
            n_treated=50, n_control=50, sigma=1.0,
            n_pre=3, n_post=3
        )

        # More periods should reduce MDE (more data)
        assert result_6period.mde < result_2period.mde
        assert result_6period.design == "panel"

    def test_icc_effect(self):
        """Test that intra-cluster correlation affects power."""
        pa = PowerAnalysis(power=0.80)

        result_no_icc = pa.mde(
            n_treated=50, n_control=50, sigma=1.0,
            n_pre=3, n_post=3, rho=0.0
        )
        result_with_icc = pa.mde(
            n_treated=50, n_control=50, sigma=1.0,
            n_pre=3, n_post=3, rho=0.5
        )

        # Higher ICC should increase MDE (less independent information)
        assert result_with_icc.mde > result_no_icc.mde

    def test_power_curve(self):
        """Test power curve generation."""
        pa = PowerAnalysis()
        curve = pa.power_curve(
            n_treated=50, n_control=50, sigma=1.0,
            effect_sizes=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
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
            effect_size=0.5, sigma=1.0,
            sample_sizes=[20, 50, 100, 150, 200]
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


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_mde(self):
        """Test compute_mde convenience function."""
        mde = compute_mde(n_treated=50, n_control=50, sigma=1.0)

        assert isinstance(mde, float)
        assert mde > 0

    def test_compute_power(self):
        """Test compute_power convenience function."""
        power = compute_power(
            effect_size=0.5,
            n_treated=50,
            n_control=50,
            sigma=1.0
        )

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
        power_class = pa.power(
            effect_size=0.5, n_treated=50, n_control=50, sigma=1.0
        ).power
        power_func = compute_power(
            effect_size=0.5, n_treated=50, n_control=50, sigma=1.0
        )
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
            n_simulations=50,
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
            n_simulations=50,
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
            n_simulations=100,
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
            n_simulations=100,
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
            n_simulations=100,
            seed=42,
            progress=False,
        )

        # CI should contain the point estimate
        assert results.power_ci[0] <= results.power <= results.power_ci[1]
        # CI should be reasonable width (0 is valid when power is exactly 0 or 1)
        ci_width = results.power_ci[1] - results.power_ci[0]
        assert 0 <= ci_width < 0.5


class TestVisualization:
    """Tests for power curve visualization."""

    def test_plot_power_curve_dataframe(self):
        """Test plotting from DataFrame."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import plot_power_curve

        df = pd.DataFrame({
            "effect_size": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            "power": [0.1, 0.2, 0.4, 0.7, 0.9, 0.99]
        })

        ax = plot_power_curve(df, show=False)
        assert ax is not None

    def test_plot_power_curve_manual_data(self):
        """Test plotting with manual effect sizes and powers."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import plot_power_curve

        ax = plot_power_curve(
            effect_sizes=[0.1, 0.2, 0.3, 0.5],
            powers=[0.1, 0.3, 0.6, 0.9],
            mde=0.25,
            show=False
        )
        assert ax is not None

    def test_plot_power_curve_sample_size(self):
        """Test plotting power vs sample size."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import plot_power_curve

        df = pd.DataFrame({
            "sample_size": [20, 50, 100, 150, 200],
            "power": [0.2, 0.5, 0.8, 0.9, 0.95]
        })

        ax = plot_power_curve(df, show=False)
        assert ax is not None

    def test_plot_validates_input(self):
        """Test that plot validates input."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import plot_power_curve

        with pytest.raises(ValueError):
            plot_power_curve(show=False)  # No data provided

        with pytest.raises(ValueError):
            plot_power_curve(
                effect_sizes=[1, 2, 3],
                show=False  # Missing powers
            )


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
        result_zero = pa.power(
            effect_size=0.0, n_treated=50, n_control=50, sigma=1.0
        )
        assert result_zero.power < 0.10

        # Huge effect should give ~1.0 power
        result_huge = pa.power(
            effect_size=100.0, n_treated=50, n_control=50, sigma=1.0
        )
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
