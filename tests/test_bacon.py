"""
Tests for Goodman-Bacon decomposition.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    BaconDecomposition,
    BaconDecompositionResults,
    Comparison2x2,
    TwoWayFixedEffects,
    bacon_decompose,
)


def generate_staggered_data(
    n_units: int = 100,
    n_periods: int = 10,
    n_cohorts: int = 3,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    dynamic_effect: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic staggered adoption data for testing."""
    np.random.seed(seed)

    # Generate unit and time identifiers
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    # Assign treatment cohorts
    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    # Treatment periods start from period 3 onwards
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)

    first_treat = np.zeros(n_units)
    if n_treated > 0:
        cohort_assignments = np.random.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = cohort_periods[cohort_assignments]

    first_treat_expanded = np.repeat(first_treat, n_periods)

    # Generate outcomes
    unit_fe = np.random.randn(n_units) * 2
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    # Treatment indicator
    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)

    # Treatment effect (can be dynamic)
    if dynamic_effect:
        relative_time = times - first_treat_expanded
        effect = treatment_effect * (1 + 0.2 * np.maximum(relative_time, 0))
    else:
        effect = np.full(len(units), treatment_effect)

    outcomes = (
        unit_fe_expanded +
        time_fe_expanded +
        effect * post +
        np.random.randn(len(units)) * 0.5
    )

    df = pd.DataFrame({
        'unit': units,
        'time': times,
        'outcome': outcomes,
        'first_treat': first_treat_expanded.astype(int),
        'treated': post.astype(int),
    })

    return df


class TestBaconDecomposition:
    """Tests for BaconDecomposition class."""

    def test_basic_fit(self):
        """Test basic decomposition fitting."""
        data = generate_staggered_data()

        decomp = BaconDecomposition()
        results = decomp.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert decomp.is_fitted_
        assert isinstance(results, BaconDecompositionResults)
        assert results.twfe_estimate is not None
        assert len(results.comparisons) > 0

    def test_weights_sum_to_one(self):
        """Test that decomposition weights sum to approximately 1."""
        data = generate_staggered_data(seed=123)

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        total_weight = sum(c.weight for c in results.comparisons)
        assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, not 1.0"

    def test_weighted_sum_equals_twfe(self):
        """Test that weighted sum of 2x2 estimates equals TWFE."""
        data = generate_staggered_data(seed=456)

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        weighted_sum = sum(c.weight * c.estimate for c in results.comparisons)

        # Allow for small numerical error
        assert abs(results.twfe_estimate - weighted_sum) < 0.1, (
            f"TWFE ({results.twfe_estimate:.4f}) != weighted sum ({weighted_sum:.4f})"
        )

    def test_comparison_types(self):
        """Test that all three comparison types are identified."""
        data = generate_staggered_data(n_cohorts=3, never_treated_frac=0.3)

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        comp_types = set(c.comparison_type for c in results.comparisons)

        # With never-treated and multiple cohorts, should have all types
        assert "treated_vs_never" in comp_types
        assert "earlier_vs_later" in comp_types
        assert "later_vs_earlier" in comp_types

    def test_no_never_treated(self):
        """Test decomposition with no never-treated units."""
        data = generate_staggered_data(never_treated_frac=0.0)

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Should still work
        assert len(results.comparisons) > 0
        assert results.total_weight_treated_vs_never == 0.0

    def test_single_cohort(self):
        """Test with single treatment cohort."""
        data = generate_staggered_data(n_cohorts=1, never_treated_frac=0.3)

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # With single cohort, should only have treated vs never
        assert results.total_weight_earlier_vs_later == 0.0
        assert results.total_weight_later_vs_earlier == 0.0
        assert results.total_weight_treated_vs_never > 0.0

    def test_weight_by_type(self):
        """Test weight_by_type method."""
        data = generate_staggered_data()

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        weights = results.weight_by_type()

        assert "treated_vs_never" in weights
        assert "earlier_vs_later" in weights
        assert "later_vs_earlier" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_effect_by_type(self):
        """Test effect_by_type method."""
        data = generate_staggered_data()

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        effects = results.effect_by_type()

        assert "treated_vs_never" in effects
        assert "earlier_vs_later" in effects
        assert "later_vs_earlier" in effects

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        data = generate_staggered_data()

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(results.comparisons)
        assert "treated_group" in df.columns
        assert "control_group" in df.columns
        assert "comparison_type" in df.columns
        assert "estimate" in df.columns
        assert "weight" in df.columns

    def test_summary(self):
        """Test summary generation."""
        data = generate_staggered_data()

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        summary = results.summary()

        assert isinstance(summary, str)
        assert "Goodman-Bacon" in summary
        assert "TWFE" in summary

    def test_missing_column_error(self):
        """Test that missing columns raise appropriate error."""
        data = generate_staggered_data()

        with pytest.raises(ValueError, match="Missing columns"):
            bacon_decompose(
                data,
                outcome='nonexistent',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )


class TestComparison2x2:
    """Tests for Comparison2x2 dataclass."""

    def test_comparison_repr(self):
        """Test string representation."""
        comp = Comparison2x2(
            treated_group=3,
            control_group="never_treated",
            comparison_type="treated_vs_never",
            estimate=2.5,
            weight=0.25,
            n_treated=50,
            n_control=30,
            time_window=(0, 9),
        )

        repr_str = repr(comp)
        assert "3 vs never_treated" in repr_str
        assert "2.5" in repr_str or "2.50" in repr_str


class TestTWFEIntegration:
    """Tests for TWFE integration with Bacon decomposition."""

    def test_twfe_decompose_method(self):
        """Test that TwoWayFixedEffects.decompose() works."""
        data = generate_staggered_data()

        twfe = TwoWayFixedEffects()
        decomp = twfe.decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert isinstance(decomp, BaconDecompositionResults)
        assert len(decomp.comparisons) > 0

    def test_twfe_staggered_warning(self):
        """Test that TWFE warns about staggered treatment."""
        data = generate_staggered_data(n_cohorts=3)

        twfe = TwoWayFixedEffects()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            twfe.fit(
                data,
                outcome='outcome',
                treatment='treated',
                time='time',
                unit='unit'
            )

            # Should have emitted a warning about staggered treatment
            staggered_warnings = [
                x for x in w
                if "staggered" in str(x.message).lower()
            ]
            assert len(staggered_warnings) > 0


class TestBaconDecomposeFunction:
    """Tests for bacon_decompose convenience function."""

    def test_convenience_function(self):
        """Test that convenience function works."""
        data = generate_staggered_data()

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert isinstance(results, BaconDecompositionResults)


class TestVisualization:
    """Tests for Bacon decomposition visualization."""

    def test_plot_bacon_scatter(self):
        """Test scatter plot creation."""
        pytest.importorskip("matplotlib")
        from diff_diff import plot_bacon

        data = generate_staggered_data()
        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Should not raise
        ax = plot_bacon(results, plot_type='scatter', show=False)
        assert ax is not None

    def test_plot_bacon_bar(self):
        """Test bar chart creation."""
        pytest.importorskip("matplotlib")
        from diff_diff import plot_bacon

        data = generate_staggered_data()
        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Should not raise
        ax = plot_bacon(results, plot_type='bar', show=False)
        assert ax is not None

    def test_plot_bacon_invalid_type(self):
        """Test that invalid plot type raises error."""
        pytest.importorskip("matplotlib")
        from diff_diff import plot_bacon

        data = generate_staggered_data()
        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        with pytest.raises(ValueError, match="Unknown plot_type"):
            plot_bacon(results, plot_type='invalid', show=False)


class TestWeightsParameter:
    """Tests for configurable weights parameter."""

    def test_approximate_weights_default(self):
        """Test that approximate weights are used by default."""
        data = generate_staggered_data(seed=789)

        decomp = BaconDecomposition()
        assert decomp.weights == "approximate"

        results = decomp.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Weights should sum to 1
        total_weight = sum(c.weight for c in results.comparisons)
        assert abs(total_weight - 1.0) < 0.01

    def test_exact_weights(self):
        """Test exact weight calculation."""
        data = generate_staggered_data(seed=789)

        decomp = BaconDecomposition(weights="exact")
        assert decomp.weights == "exact"

        results = decomp.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Weights should still sum to 1
        total_weight = sum(c.weight for c in results.comparisons)
        assert abs(total_weight - 1.0) < 0.01

    def test_exact_vs_approximate_different(self):
        """Test that exact and approximate weights can differ."""
        data = generate_staggered_data(seed=123, n_cohorts=3)

        results_approx = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            weights="approximate"
        )

        results_exact = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            weights="exact"
        )

        # TWFE estimates should be the same
        assert abs(results_approx.twfe_estimate - results_exact.twfe_estimate) < 0.0001

        # Same number of comparisons
        assert len(results_approx.comparisons) == len(results_exact.comparisons)

    def test_exact_weights_lower_decomposition_error(self):
        """Test that exact weights generally have lower decomposition error."""
        data = generate_staggered_data(seed=456, n_cohorts=3)

        results_approx = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            weights="approximate"
        )

        results_exact = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            weights="exact"
        )

        # Exact weights should have equal or lower decomposition error
        # Allow some margin for numerical precision
        assert results_exact.decomposition_error <= results_approx.decomposition_error + 0.1

    def test_invalid_weights_raises(self):
        """Test that invalid weights parameter raises error."""
        with pytest.raises(ValueError, match="weights must be"):
            BaconDecomposition(weights="invalid")

    def test_convenience_function_weights_param(self):
        """Test that convenience function accepts weights parameter."""
        data = generate_staggered_data()

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            weights="exact"
        )

        assert isinstance(results, BaconDecompositionResults)

    def test_twfe_decompose_weights_param(self):
        """Test that TWFE.decompose() accepts weights parameter."""
        data = generate_staggered_data()

        twfe = TwoWayFixedEffects()

        # Test with approximate
        decomp_approx = twfe.decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            weights="approximate"
        )
        assert isinstance(decomp_approx, BaconDecompositionResults)

        # Test with exact
        decomp_exact = twfe.decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            weights="exact"
        )
        assert isinstance(decomp_exact, BaconDecompositionResults)

    def test_get_set_params(self):
        """Test get_params and set_params with weights."""
        decomp = BaconDecomposition(weights="approximate")
        assert decomp.get_params()["weights"] == "approximate"

        decomp.set_params(weights="exact")
        assert decomp.get_params()["weights"] == "exact"

        with pytest.raises(ValueError):
            decomp.set_params(weights="invalid")


class TestBalancedPanelWarning:
    """Tests for balanced panel warning."""

    def test_unbalanced_panel_warning(self):
        """Test that unbalanced panel triggers warning."""
        # Create an unbalanced panel manually
        data = generate_staggered_data(n_units=50, n_periods=10)

        # Remove some observations from specific units to make it unbalanced
        # This ensures different units have different numbers of periods
        mask = ~((data['unit'] == 0) & (data['time'] == 0))  # Remove one period from unit 0
        data = data[mask].reset_index(drop=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bacon_decompose(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

            # Should have warning about unbalanced panel
            unbalanced_warnings = [
                x for x in w
                if "unbalanced" in str(x.message).lower()
            ]
            assert len(unbalanced_warnings) > 0

    def test_balanced_panel_no_warning(self):
        """Test that balanced panel does not trigger warning."""
        data = generate_staggered_data()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bacon_decompose(
                data,
                outcome='outcome',
                unit='unit',
                time='time',
                first_treat='first_treat'
            )

            # Should NOT have warning about unbalanced panel
            unbalanced_warnings = [
                x for x in w
                if "unbalanced" in str(x.message).lower()
            ]
            assert len(unbalanced_warnings) == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_sample(self):
        """Test with small sample size."""
        data = generate_staggered_data(n_units=20, n_periods=5, n_cohorts=2)

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert len(results.comparisons) > 0

    def test_many_cohorts(self):
        """Test with many treatment cohorts."""
        data = generate_staggered_data(
            n_units=200, n_periods=15, n_cohorts=5, never_treated_frac=0.2
        )

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        # Should have comparisons from all cohort pairs
        assert len(results.comparisons) > 5

    def test_inf_for_never_treated(self):
        """Test using np.inf for never-treated units."""
        data = generate_staggered_data()

        # Replace 0 with inf for never-treated
        data['first_treat'] = data['first_treat'].replace(0, np.inf)

        results = bacon_decompose(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat'
        )

        assert results.n_never_treated > 0
        assert len(results.comparisons) > 0
