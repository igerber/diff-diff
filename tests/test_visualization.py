"""
Tests for visualization functions.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    MultiPeriodDiD,
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

    def test_plot_with_nan_se_reference_period(self):
        """Test that reference period with NaN SE is plotted without error bars.

        With universal base period, the reference period (e=-1) has SE=NaN.
        The plot should show the point estimate without error bars rather than
        skipping it entirely.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # Create data with NaN SE for reference period
        effects = {-2: 0.1, -1: 0.0, 0: 0.5, 1: 0.6}
        se = {-2: 0.1, -1: np.nan, 0: 0.15, 1: 0.15}  # NaN SE at reference period

        ax = plot_event_study(
            effects=effects,
            se=se,
            reference_period=-1,
            show=False
        )

        # Verify the plot was created successfully
        assert ax is not None

        # Verify all 4 periods are plotted (including reference with NaN SE)
        # The x-axis should have 4 tick labels
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(xtick_labels) == 4
        assert '-1' in xtick_labels

        plt.close()

    def test_plot_cs_universal_base_period(self):
        """Test plotting CallawaySantAnna results with universal base period.

        The reference period (e=-1) should appear in the plot even though
        it has SE=NaN.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from diff_diff import generate_staggered_data

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

        # Should not raise even with NaN SE in reference period
        ax = plot_event_study(results, show=False)
        assert ax is not None

        # Verify reference period (-1) is in the plot
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert '-1' in xtick_labels

        plt.close()

    def test_plot_cs_with_anticipation(self):
        """Test plotting CallawaySantAnna results with anticipation > 0.

        When anticipation=1, the reference period should be e=-2, not e=-1.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from diff_diff import generate_staggered_data

        data = generate_staggered_data(n_units=200, n_periods=10, seed=42)
        cs = CallawaySantAnna(base_period="universal", anticipation=1)
        results = cs.fit(
            data,
            outcome='outcome',
            unit='unit',
            time='period',
            first_treat='first_treat',
            aggregate='event_study'
        )

        # Reference period should be at e=-2 (not e=-1) with anticipation=1
        assert -2 in results.event_study_effects
        assert results.event_study_effects[-2]['n_groups'] == 0

        ax = plot_event_study(results, show=False)
        assert ax is not None

        # Verify -2 is in the plot (the true reference period)
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert '-2' in xtick_labels

        plt.close()

    def test_plot_event_study_reference_period_normalization(self):
        """Test that reference_period normalizes effects and sets reference SE to NaN.

        When reference_period is specified:
        1. The effect at that period is subtracted from all effects (ref period = 0)
        2. The SE at the reference period is set to NaN (it's a constraint, not an estimate)
        3. Other periods retain their original SEs and have error bars

        This follows the fixest (R) convention where the omitted/reference category
        has no associated uncertainty (it's an identifying constraint).
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # Create data where reference period (period=0) has effect=0.3
        df = pd.DataFrame({
            'period': [-2, -1, 0, 1, 2],
            'effect': [0.1, 0.2, 0.3, 0.5, 0.6],  # ref at 0 has effect 0.3
            'se': [0.1, 0.1, 0.1, 0.1, 0.1]
        })

        ax = plot_event_study(df, reference_period=0, show=False)

        # Find plotted y-values by extracting data from Line2D objects
        # The point estimates are plotted as individual markers
        y_values = []
        for child in ax.get_children():
            # Line2D objects with single points are our markers
            if hasattr(child, 'get_ydata'):
                ydata = child.get_ydata()
                if len(ydata) == 1:
                    y_values.append(float(ydata[0]))

        # After normalization:
        # - Original effects: [0.1, 0.2, 0.3, 0.5, 0.6]
        # - Reference effect: 0.3
        # - Normalized: [-0.2, -0.1, 0.0, 0.2, 0.3]
        expected_normalized = [-0.2, -0.1, 0.0, 0.2, 0.3]

        # Check that reference period (0) is at y=0
        assert 0.0 in y_values or any(abs(y) < 0.01 for y in y_values), \
            f"Reference period should be at y=0, got y_values={y_values}"

        # Verify all expected normalized values are present
        for expected in expected_normalized:
            assert any(abs(y - expected) < 0.01 for y in y_values), \
                f"Expected normalized value {expected} not found in {y_values}"

        # Verify error bars: reference period (y=0) should have NO error bars
        # while other periods should have error bars
        # Error bars are drawn via ax.errorbar, which creates ErrorbarContainer or Line2D
        # The error bar x-coordinates tell us which periods have error bars

        # Find the errorbar data (the line segments that form error bars)
        errorbar_x_coords = set()
        for child in ax.get_children():
            # ErrorbarContainer's children include LineCollection for the caps/stems
            if hasattr(child, 'get_segments'):
                segments = child.get_segments()
                for seg in segments:
                    # Each segment is [[x1, y1], [x2, y2]]
                    if len(seg) >= 2:
                        # x-coordinate of error bar (both points have same x)
                        errorbar_x_coords.add(round(seg[0][0], 1))

        # x-coordinates: period -2 -> x=0, -1 -> x=1, 0 -> x=2, 1 -> x=3, 2 -> x=4
        # The reference period (period=0) is at x=2
        reference_x = 2  # period 0 is at x-coordinate 2

        # Reference period should NOT have error bars (x=2 should not be in errorbar_x_coords)
        assert reference_x not in errorbar_x_coords, \
            f"Reference period should have no error bars but found error bar at x={reference_x}"

        # Other periods SHOULD have error bars
        # At least some of x=0, x=1, x=3, x=4 should have error bars
        non_ref_x_coords = {0, 1, 3, 4}
        assert len(errorbar_x_coords & non_ref_x_coords) >= 2, \
            f"Non-reference periods should have error bars, found: {errorbar_x_coords}"

        plt.close()

    def test_plot_event_study_no_normalization_without_reference(self):
        """Test that effects are NOT normalized when reference_period is None."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = pd.DataFrame({
            'period': [-1, 0, 1],
            'effect': [0.1, 0.3, 0.5],
            'se': [0.1, 0.1, 0.1]
        })

        ax = plot_event_study(df, reference_period=None, show=False)

        # Extract y-values
        y_values = []
        for child in ax.get_children():
            if hasattr(child, 'get_ydata'):
                ydata = child.get_ydata()
                if len(ydata) == 1:
                    y_values.append(float(ydata[0]))

        # Without normalization, original values should be preserved
        for expected in [0.1, 0.3, 0.5]:
            assert any(abs(y - expected) < 0.01 for y in y_values), \
                f"Original value {expected} not found in {y_values}"

        plt.close()

    def test_plot_event_study_normalization_with_nan_reference(self):
        """Test that normalization is skipped when reference effect is NaN."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = pd.DataFrame({
            'period': [-1, 0, 1],
            'effect': [0.1, np.nan, 0.5],  # Reference period has NaN effect
            'se': [0.1, 0.1, 0.1]
        })

        # This should not raise and should skip normalization
        ax = plot_event_study(df, reference_period=0, show=False)

        # Extract y-values (NaN effect is skipped in plotting)
        y_values = []
        for child in ax.get_children():
            if hasattr(child, 'get_ydata'):
                ydata = child.get_ydata()
                if len(ydata) == 1:
                    y_values.append(float(ydata[0]))

        # Original non-NaN values should be preserved (not normalized)
        for expected in [0.1, 0.5]:
            assert any(abs(y - expected) < 0.01 for y in y_values), \
                f"Original value {expected} not found in {y_values}"

        plt.close()


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
