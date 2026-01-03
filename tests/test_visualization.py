"""
Tests for visualization functions.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    MultiPeriodDiD,
    CallawaySantAnna,
    plot_event_study,
)


def generate_multi_period_data(n_obs: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate data for multi-period DiD testing."""
    np.random.seed(seed)

    n_per_group = n_obs // 8

    data = []
    for treated in [0, 1]:
        for period in range(4):  # 4 time periods
            for _ in range(n_per_group):
                # Base outcome
                y = 10 + period * 0.5

                # Treatment effect (only in post-treatment periods 2, 3)
                if treated == 1 and period >= 2:
                    y += 2.0 + 0.5 * (period - 2)

                y += np.random.randn() * 0.5

                data.append({
                    'outcome': y,
                    'treated': treated,
                    'period': period,
                })

    return pd.DataFrame(data)


def generate_staggered_data(
    n_units: int = 50,
    n_periods: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate staggered adoption data."""
    np.random.seed(seed)

    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    # First 15 units never treated, rest treated at period 3 or 5
    first_treat = np.zeros(n_units)
    first_treat[15:30] = 3
    first_treat[30:] = 5

    first_treat_expanded = np.repeat(first_treat, n_periods)

    # Outcomes
    unit_fe = np.random.randn(n_units)
    unit_fe_expanded = np.repeat(unit_fe, n_periods)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)

    outcomes = unit_fe_expanded + 0.5 * times + 2.0 * post + np.random.randn(len(units)) * 0.3

    return pd.DataFrame({
        'unit': units,
        'time': times,
        'outcome': outcomes,
        'first_treat': first_treat_expanded.astype(int),
    })


class TestPlotEventStudy:
    """Tests for plot_event_study function."""

    @pytest.fixture
    def multi_period_results(self):
        """Fixture for MultiPeriodDiD results."""
        data = generate_multi_period_data()
        did = MultiPeriodDiD()
        return did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='period',
            post_periods=[2, 3]
        )

    @pytest.fixture
    def cs_results(self):
        """Fixture for CallawaySantAnna results with event study."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        return cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

    def test_plot_from_multi_period_results(self, multi_period_results):
        """Test plotting from MultiPeriodDiD results."""
        pytest.importorskip("matplotlib")

        ax = plot_event_study(multi_period_results, show=False)
        assert ax is not None

    def test_plot_from_cs_results(self, cs_results):
        """Test plotting from CallawaySantAnna results."""
        pytest.importorskip("matplotlib")

        ax = plot_event_study(cs_results, show=False)
        assert ax is not None

    def test_plot_from_dataframe(self):
        """Test plotting from DataFrame."""
        pytest.importorskip("matplotlib")

        df = pd.DataFrame({
            'period': [-2, -1, 0, 1, 2],
            'effect': [0.1, 0.05, 0.0, 0.5, 0.6],
            'se': [0.1, 0.1, 0.0, 0.15, 0.15]
        })

        ax = plot_event_study(df, reference_period=0, show=False)
        assert ax is not None

    def test_plot_from_dict(self):
        """Test plotting from dictionaries."""
        pytest.importorskip("matplotlib")

        effects = {-2: 0.1, -1: 0.05, 0: 0.0, 1: 0.5, 2: 0.6}
        se = {-2: 0.1, -1: 0.1, 0: 0.0, 1: 0.15, 2: 0.15}

        ax = plot_event_study(
            effects=effects,
            se=se,
            reference_period=0,
            show=False
        )
        assert ax is not None

    def test_plot_customization(self, multi_period_results):
        """Test plot customization options."""
        pytest.importorskip("matplotlib")

        ax = plot_event_study(
            multi_period_results,
            title="Custom Title",
            xlabel="Custom X",
            ylabel="Custom Y",
            color="red",
            marker="s",
            markersize=10,
            show=False
        )

        assert ax.get_title() == "Custom Title"
        assert ax.get_xlabel() == "Custom X"
        assert ax.get_ylabel() == "Custom Y"

    def test_plot_no_zero_line(self, multi_period_results):
        """Test disabling zero line."""
        pytest.importorskip("matplotlib")

        ax = plot_event_study(
            multi_period_results,
            show_zero_line=False,
            show=False
        )
        assert ax is not None

    def test_plot_with_existing_axes(self, multi_period_results):
        """Test plotting on existing axes."""
        matplotlib = pytest.importorskip("matplotlib")
        plt = matplotlib.pyplot

        fig, ax = plt.subplots()
        returned_ax = plot_event_study(multi_period_results, ax=ax, show=False)
        assert returned_ax is ax
        plt.close()

    def test_error_no_inputs(self):
        """Test error when no inputs provided."""
        pytest.importorskip("matplotlib")
        with pytest.raises(ValueError, match="Must provide either"):
            plot_event_study()

    def test_error_invalid_effects_type(self):
        """Test error with invalid effects type."""
        pytest.importorskip("matplotlib")
        with pytest.raises(TypeError, match="effects must be a dictionary"):
            plot_event_study(effects=[1, 2, 3], se={1: 0.1})

    def test_error_invalid_se_type(self):
        """Test error with invalid se type."""
        pytest.importorskip("matplotlib")
        with pytest.raises(TypeError, match="se must be a dictionary"):
            plot_event_study(effects={1: 0.5}, se=[0.1])

    def test_error_missing_dataframe_columns(self):
        """Test error with missing DataFrame columns."""
        pytest.importorskip("matplotlib")
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [0.1, 0.2, 0.3]
        })

        with pytest.raises(ValueError, match="must have 'period' column"):
            plot_event_study(df)

    def test_error_invalid_results_type(self):
        """Test error with invalid results type."""
        pytest.importorskip("matplotlib")
        with pytest.raises(TypeError, match="Cannot extract plot data"):
            plot_event_study("invalid")


class TestPlotEventStudyIntegration:
    """Integration tests for event study plotting."""

    def test_full_workflow_multi_period(self):
        """Test full workflow with MultiPeriodDiD."""
        matplotlib = pytest.importorskip("matplotlib")
        plt = matplotlib.pyplot

        # Generate data
        data = generate_multi_period_data()

        # Fit model
        did = MultiPeriodDiD()
        results = did.fit(
            data,
            outcome='outcome',
            treatment='treated',
            time='period',
            post_periods=[2, 3]
        )

        # Plot
        ax = plot_event_study(
            results,
            title="Treatment Effects Over Time",
            show=False
        )

        assert ax is not None
        plt.close()

    def test_full_workflow_callaway_santanna(self):
        """Test full workflow with CallawaySantAnna."""
        matplotlib = pytest.importorskip("matplotlib")
        plt = matplotlib.pyplot

        # Generate data
        data = generate_staggered_data()

        # Fit model with event study
        cs = CallawaySantAnna()
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='time',
            first_treat='first_treat',
            aggregate='event_study'
        )

        # Plot
        ax = plot_event_study(
            results,
            title="Staggered DiD Event Study",
            show=False
        )

        assert ax is not None
        plt.close()
