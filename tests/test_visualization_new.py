"""Tests for new visualization functions and plotly backend.

Tests cover:
- Import compatibility after subpackage refactoring
- plot_synth_weights
- plot_staircase
- plot_dose_response
- plot_group_time_heatmap
- Plotly backend for all functions
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Skip all tests if matplotlib is not available
mpl = pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def synth_results():
    """Mock SyntheticDiDResults."""
    results = MagicMock()
    results.unit_weights = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.05, "E": 0.0005}
    results.time_weights = {2000: 0.1, 2001: 0.3, 2002: 0.6}
    return results


@pytest.fixture
def cs_results():
    """Mock CallawaySantAnnaResults with group_time_effects."""
    results = MagicMock()
    results.groups = [2004, 2006]
    results.time_periods = [2003, 2004, 2005, 2006, 2007]
    results.group_time_effects = {
        (2004, 2003): {
            "effect": 0.02,
            "se": 0.1,
            "p_value": 0.84,
            "n_treated": 50,
            "n_control": 100,
        },
        (2004, 2004): {
            "effect": 0.5,
            "se": 0.12,
            "p_value": 0.001,
            "n_treated": 50,
            "n_control": 100,
        },
        (2004, 2005): {
            "effect": 0.6,
            "se": 0.13,
            "p_value": 0.001,
            "n_treated": 50,
            "n_control": 100,
        },
        (2004, 2006): {
            "effect": 0.7,
            "se": 0.14,
            "p_value": 0.001,
            "n_treated": 50,
            "n_control": 100,
        },
        (2004, 2007): {
            "effect": 0.75,
            "se": 0.15,
            "p_value": 0.001,
            "n_treated": 50,
            "n_control": 100,
        },
        (2006, 2003): {
            "effect": -0.01,
            "se": 0.1,
            "p_value": 0.92,
            "n_treated": 30,
            "n_control": 100,
        },
        (2006, 2004): {
            "effect": 0.03,
            "se": 0.11,
            "p_value": 0.78,
            "n_treated": 30,
            "n_control": 100,
        },
        (2006, 2005): {
            "effect": 0.01,
            "se": 0.1,
            "p_value": 0.92,
            "n_treated": 30,
            "n_control": 100,
        },
        (2006, 2006): {
            "effect": 0.4,
            "se": 0.12,
            "p_value": 0.001,
            "n_treated": 30,
            "n_control": 100,
        },
        (2006, 2007): {
            "effect": 0.45,
            "se": 0.13,
            "p_value": 0.001,
            "n_treated": 30,
            "n_control": 100,
        },
    }
    return results


@pytest.fixture
def dose_response_curve():
    """Mock DoseResponseCurve."""
    curve = MagicMock()
    curve.dose_grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    curve.effects = np.array([0.1, 0.3, 0.5, 0.4, 0.3])
    curve.se = np.array([0.05, 0.06, 0.07, 0.08, 0.09])
    curve.conf_int_lower = np.array([0.0, 0.18, 0.36, 0.24, 0.12])
    curve.conf_int_upper = np.array([0.2, 0.42, 0.64, 0.56, 0.48])
    curve.target = "att"
    return curve


@pytest.fixture
def continuous_results(dose_response_curve):
    """Mock ContinuousDiDResults."""
    results = MagicMock()
    results.dose_response_att = dose_response_curve
    acrt = MagicMock()
    acrt.dose_grid = np.array([1.0, 2.0, 3.0])
    acrt.effects = np.array([0.05, 0.15, 0.25])
    acrt.se = np.array([0.03, 0.04, 0.05])
    acrt.conf_int_lower = np.array([-0.01, 0.07, 0.15])
    acrt.conf_int_upper = np.array([0.11, 0.23, 0.35])
    acrt.target = "acrt"
    results.dose_response_acrt = acrt
    return results


# ── TestImportCompatibility ───────────────────────────────────────────────────


class TestImportCompatibility:
    """Verify all import paths work after subpackage refactoring."""

    def test_import_extract_plot_data(self):
        from diff_diff.visualization import _extract_plot_data

        assert callable(_extract_plot_data)

    def test_import_plot_event_study_from_visualization(self):
        from diff_diff.visualization import plot_event_study

        assert callable(plot_event_study)

    def test_import_plot_event_study_from_main(self):
        from diff_diff import plot_event_study

        assert callable(plot_event_study)

    def test_import_new_internal_path(self):
        from diff_diff.visualization._event_study import plot_event_study

        assert callable(plot_event_study)

    def test_import_all_new_functions(self):
        from diff_diff import (
            plot_dose_response,
            plot_group_time_heatmap,
            plot_staircase,
            plot_synth_weights,
        )

        assert callable(plot_synth_weights)
        assert callable(plot_staircase)
        assert callable(plot_dose_response)
        assert callable(plot_group_time_heatmap)

    def test_import_power_from_visualization(self):
        from diff_diff.visualization import plot_power_curve

        assert callable(plot_power_curve)

    def test_import_sensitivity_from_visualization(self):
        from diff_diff.visualization import plot_sensitivity

        assert callable(plot_sensitivity)

    def test_import_bacon_from_visualization(self):
        from diff_diff.visualization import plot_bacon

        assert callable(plot_bacon)


# ── TestPlotSynthWeights ──────────────────────────────────────────────────────


class TestPlotSynthWeights:
    """Tests for plot_synth_weights."""

    def test_basic_from_results(self, synth_results):
        from diff_diff import plot_synth_weights

        ax = plot_synth_weights(synth_results, show=False)
        assert ax is not None
        assert ax.get_title() == "Synthetic Control Unit Weights"

    def test_time_weights(self, synth_results):
        from diff_diff import plot_synth_weights

        ax = plot_synth_weights(synth_results, weight_type="time", show=False)
        assert ax.get_title() == "Synthetic Control Time Weights"

    def test_from_dict(self):
        from diff_diff import plot_synth_weights

        weights = {"unit_1": 0.5, "unit_2": 0.3, "unit_3": 0.2}
        ax = plot_synth_weights(weights=weights, show=False)
        assert ax is not None

    def test_top_n(self, synth_results):
        from diff_diff import plot_synth_weights

        ax = plot_synth_weights(synth_results, top_n=2, show=False)
        # Should only show 2 bars
        patches = [p for p in ax.patches if hasattr(p, "get_width")]
        assert len(patches) == 2

    def test_min_weight_filter(self, synth_results):
        from diff_diff import plot_synth_weights

        ax = plot_synth_weights(synth_results, min_weight=0.1, show=False)
        # E (0.0005) and D (0.05) filtered out, leaving A, B, C
        patches = [p for p in ax.patches if hasattr(p, "get_width")]
        assert len(patches) == 3

    def test_empty_weights_raises(self):
        from diff_diff import plot_synth_weights

        with pytest.raises(ValueError, match="No weights available"):
            plot_synth_weights(weights={}, show=False)

    def test_both_inputs_raises(self, synth_results):
        from diff_diff import plot_synth_weights

        with pytest.raises(ValueError, match="not both"):
            plot_synth_weights(synth_results, weights={"a": 1}, show=False)

    def test_custom_title_and_color(self, synth_results):
        from diff_diff import plot_synth_weights

        ax = plot_synth_weights(synth_results, title="Custom", color="#ff0000", show=False)
        assert ax.get_title() == "Custom"


# ── TestPlotStaircase ─────────────────────────────────────────────────────────


class TestPlotStaircase:
    """Tests for plot_staircase."""

    def test_from_cs_results(self, cs_results):
        from diff_diff import plot_staircase

        ax = plot_staircase(cs_results, show=False)
        assert ax is not None
        assert ax.get_title() == "Treatment Adoption Over Time"

    def test_from_dataframe(self):
        from diff_diff import plot_staircase

        df = pd.DataFrame(
            {
                "state": [1, 1, 2, 2, 3, 3, 4, 4],
                "year": [2000, 2001, 2000, 2001, 2000, 2001, 2000, 2001],
                "first_treat_year": [2000, 2000, 2001, 2001, 2001, 2001, 2000, 2000],
            }
        )
        ax = plot_staircase(
            data=df, unit="state", time="year", first_treat="first_treat_year", show=False
        )
        assert ax is not None

    def test_show_counts_toggle(self, cs_results):
        from diff_diff import plot_staircase

        ax = plot_staircase(cs_results, show_counts=False, show=False)
        assert ax is not None

    def test_missing_data_raises(self):
        from diff_diff import plot_staircase

        with pytest.raises(ValueError, match="Must provide"):
            plot_staircase(show=False)

    def test_both_inputs_raises(self, cs_results):
        from diff_diff import plot_staircase

        df = pd.DataFrame({"state": [1], "year": [2000], "first_treat_year": [2000]})
        with pytest.raises(ValueError, match="not both"):
            plot_staircase(
                cs_results,
                data=df,
                unit="state",
                time="year",
                first_treat="first_treat_year",
                show=False,
            )

    def test_dataframe_missing_columns(self):
        from diff_diff import plot_staircase

        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="must provide"):
            plot_staircase(data=df, show=False)


# ── TestPlotDoseResponse ──────────────────────────────────────────────────────


class TestPlotDoseResponse:
    """Tests for plot_dose_response."""

    def test_from_results_att(self, continuous_results):
        from diff_diff import plot_dose_response

        ax = plot_dose_response(continuous_results, target="att", show=False)
        assert ax is not None
        assert "ATT" in ax.get_title()

    def test_from_results_acrt(self, continuous_results):
        from diff_diff import plot_dose_response

        ax = plot_dose_response(continuous_results, target="acrt", show=False)
        assert ax is not None
        assert "ACRT" in ax.get_title()

    def test_from_curve_directly(self, dose_response_curve):
        from diff_diff import plot_dose_response

        ax = plot_dose_response(curve=dose_response_curve, show=False)
        assert ax is not None

    def test_from_dataframe(self):
        from diff_diff import plot_dose_response

        df = pd.DataFrame(
            {
                "dose": [1, 2, 3, 4],
                "effect": [0.1, 0.3, 0.5, 0.4],
                "se": [0.05, 0.06, 0.07, 0.08],
            }
        )
        ax = plot_dose_response(data=df, show=False)
        assert ax is not None

    def test_dataframe_with_ci(self):
        from diff_diff import plot_dose_response

        df = pd.DataFrame(
            {
                "dose": [1, 2, 3],
                "effect": [0.1, 0.3, 0.5],
                "conf_int_lower": [0.0, 0.2, 0.4],
                "conf_int_upper": [0.2, 0.4, 0.6],
            }
        )
        ax = plot_dose_response(data=df, show=False)
        assert ax is not None

    def test_multiple_inputs_raises(self, continuous_results, dose_response_curve):
        from diff_diff import plot_dose_response

        with pytest.raises(ValueError, match="exactly one"):
            plot_dose_response(continuous_results, curve=dose_response_curve, show=False)

    def test_no_input_raises(self):
        from diff_diff import plot_dose_response

        with pytest.raises(ValueError, match="exactly one"):
            plot_dose_response(show=False)


# ── TestPlotGroupTimeHeatmap ──────────────────────────────────────────────────


class TestPlotGroupTimeHeatmap:
    """Tests for plot_group_time_heatmap."""

    def test_from_cs_results(self, cs_results):
        from diff_diff import plot_group_time_heatmap

        ax = plot_group_time_heatmap(cs_results, show=False)
        assert ax is not None
        assert ax.get_title() == "Group-Time Treatment Effects"

    def test_from_dataframe(self):
        from diff_diff import plot_group_time_heatmap

        df = pd.DataFrame(
            {
                "group": [2004, 2004, 2006, 2006],
                "time": [2004, 2005, 2006, 2007],
                "effect": [0.5, 0.6, 0.4, 0.45],
            }
        )
        ax = plot_group_time_heatmap(data=df, show=False)
        assert ax is not None

    def test_annotate_toggle(self, cs_results):
        from diff_diff import plot_group_time_heatmap

        ax = plot_group_time_heatmap(cs_results, annotate=False, show=False)
        assert ax is not None

    def test_mask_insignificant(self, cs_results):
        from diff_diff import plot_group_time_heatmap

        ax = plot_group_time_heatmap(cs_results, mask_insignificant=True, show=False)
        assert ax is not None

    def test_empty_results_raises(self):
        from diff_diff import plot_group_time_heatmap

        results = MagicMock()
        results.group_time_effects = {}
        with pytest.raises(ValueError, match="empty"):
            plot_group_time_heatmap(results, show=False)

    def test_both_inputs_raises(self, cs_results):
        from diff_diff import plot_group_time_heatmap

        df = pd.DataFrame({"group": [1], "time": [1], "effect": [0.1]})
        with pytest.raises(ValueError, match="not both"):
            plot_group_time_heatmap(cs_results, data=df, show=False)


# ── TestPlotlyBackend ─────────────────────────────────────────────────────────


class TestPlotlyBackend:
    """Tests for plotly backend across all plot functions."""

    @pytest.fixture(autouse=True)
    def _require_plotly(self):
        pytest.importorskip("plotly")

    def test_event_study_plotly(self):
        import plotly.graph_objects as go

        from diff_diff import plot_event_study

        effects = {-2: 0.1, -1: 0.0, 0: 0.5, 1: 0.6}
        se = {-2: 0.1, -1: 0.0, 0: 0.15, 1: 0.15}
        fig = plot_event_study(
            effects=effects, se=se, reference_period=-1, backend="plotly", show=False
        )
        assert isinstance(fig, go.Figure)

    def test_synth_weights_plotly(self, synth_results):
        import plotly.graph_objects as go

        from diff_diff import plot_synth_weights

        fig = plot_synth_weights(synth_results, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_staircase_plotly(self, cs_results):
        import plotly.graph_objects as go

        from diff_diff import plot_staircase

        fig = plot_staircase(cs_results, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_dose_response_plotly(self, dose_response_curve):
        import plotly.graph_objects as go

        from diff_diff import plot_dose_response

        fig = plot_dose_response(curve=dose_response_curve, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_heatmap_plotly(self, cs_results):
        import plotly.graph_objects as go

        from diff_diff import plot_group_time_heatmap

        fig = plot_group_time_heatmap(cs_results, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_group_effects_plotly(self, cs_results):
        import plotly.graph_objects as go

        from diff_diff.visualization import plot_group_effects

        fig = plot_group_effects(cs_results, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_power_curve_plotly(self):
        import plotly.graph_objects as go

        from diff_diff.visualization import plot_power_curve

        fig = plot_power_curve(
            effect_sizes=[1, 2, 3, 4, 5],
            powers=[0.2, 0.5, 0.75, 0.90, 0.97],
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)

    def test_pretrends_power_plotly(self):
        import plotly.graph_objects as go

        from diff_diff.visualization import plot_pretrends_power

        fig = plot_pretrends_power(
            M_values=[0, 0.5, 1, 1.5, 2],
            powers=[0.05, 0.3, 0.6, 0.85, 0.95],
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)

    def test_matplotlib_default_returns_axes(self):
        """Ensure default backend still returns matplotlib axes."""
        from diff_diff import plot_event_study

        effects = {-2: 0.1, -1: 0.0, 0: 0.5, 1: 0.6}
        se = {-2: 0.1, -1: 0.0, 0: 0.15, 1: 0.15}
        ax = plot_event_study(effects=effects, se=se, reference_period=-1, show=False)
        assert isinstance(ax, matplotlib.axes.Axes)


# ── Regression Tests ──────────────────────────────────────────────────────────


class TestPlotlyColorHandling:
    """Regression: named colors must not crash plotly backend (PR #222 P1)."""

    @pytest.fixture(autouse=True)
    def _require_plotly(self):
        pytest.importorskip("plotly")

    def test_event_study_named_colors(self):
        import plotly.graph_objects as go

        from diff_diff import plot_event_study

        effects = {-2: 0.1, -1: 0.0, 0: 0.5, 1: 0.6}
        se = {-2: 0.1, -1: 0.0, 0: 0.15, 1: 0.15}
        fig = plot_event_study(
            effects=effects,
            se=se,
            reference_period=-1,
            color="red",
            shade_color="lightgray",
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)

    def test_dose_response_named_color(self, dose_response_curve):
        import plotly.graph_objects as go

        from diff_diff import plot_dose_response

        fig = plot_dose_response(
            curve=dose_response_curve, color="blue", backend="plotly", show=False
        )
        assert isinstance(fig, go.Figure)

    def test_staircase_named_color(self, cs_results):
        import plotly.graph_objects as go

        from diff_diff import plot_staircase

        fig = plot_staircase(cs_results, color="teal", backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_event_study_string_periods(self):
        """Regression: plotly event study must handle string period labels."""
        import plotly.graph_objects as go

        from diff_diff import plot_event_study

        effects = {"pre2": 0.1, "pre1": 0.0, "post1": 0.5, "post2": 0.6}
        se = {"pre2": 0.1, "pre1": 0.0, "post1": 0.15, "post2": 0.15}
        fig = plot_event_study(
            effects=effects,
            se=se,
            reference_period="pre1",
            pre_periods=["pre2", "pre1"],
            post_periods=["post1", "post2"],
            shade_pre=True,
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)

    def test_event_study_timestamp_periods(self):
        """Regression: plotly event study must handle pd.Timestamp periods."""
        import plotly.graph_objects as go

        from diff_diff import plot_event_study

        p1 = pd.Timestamp("2020-01-01")
        p2 = pd.Timestamp("2020-02-01")
        p3 = pd.Timestamp("2020-03-01")
        p4 = pd.Timestamp("2020-04-01")
        effects = {p1: 0.1, p2: 0.0, p3: 0.5, p4: 0.6}
        se = {p1: 0.1, p2: 0.0, p3: 0.15, p4: 0.15}
        fig = plot_event_study(
            effects=effects,
            se=se,
            reference_period=p2,
            pre_periods=[p1, p2],
            post_periods=[p3, p4],
            shade_pre=True,
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)

    def test_three_digit_hex(self):
        from diff_diff.visualization._common import _color_to_rgba

        result = _color_to_rgba("#abc", 0.5)
        assert result == "rgba(170, 187, 204, 0.5)"


class TestStaircaseCohortCounts:
    """Regression: varying n_treated across cells (PR #222 P1)."""

    def test_varying_n_treated_uses_max(self):
        from diff_diff import plot_staircase

        results = MagicMock()
        results.groups = [2004]
        results.group_time_effects = {
            (2004, 2003): {"effect": 0.0, "se": 0.1, "n_treated": 48},
            (2004, 2004): {"effect": 0.5, "se": 0.1, "n_treated": 50},
        }
        with pytest.warns(UserWarning, match="n_treated varies"):
            ax = plot_staircase(results, show=False)
        assert ax is not None

    def test_consistent_n_treated_no_warning(self, cs_results):
        """No warning when n_treated is consistent within each cohort."""
        # cs_results fixture has consistent n_treated per cohort
        import warnings

        from diff_diff import plot_staircase

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ax = plot_staircase(cs_results, show=False)
        assert ax is not None


class TestDoseResponseTargetInference:
    """Regression: curve.target should drive auto-title (PR #222 P2)."""

    def test_acrt_curve_gets_acrt_title(self):
        from diff_diff import plot_dose_response

        curve = MagicMock()
        curve.target = "acrt"
        curve.dose_grid = np.array([1, 2, 3])
        curve.effects = np.array([0.1, 0.2, 0.3])
        curve.conf_int_lower = np.array([0.0, 0.1, 0.2])
        curve.conf_int_upper = np.array([0.2, 0.3, 0.4])
        ax = plot_dose_response(curve=curve, show=False)
        assert "ACRT" in ax.get_title()

    def test_att_curve_gets_att_title(self, dose_response_curve):
        from diff_diff import plot_dose_response

        ax = plot_dose_response(curve=dose_response_curve, show=False)
        assert "ATT" in ax.get_title()


class TestBaconPlotlyWeightedAvg:
    """Regression: plotly scatter must show weighted avg lines (PR #222 P2)."""

    @pytest.fixture(autouse=True)
    def _require_plotly(self):
        pytest.importorskip("plotly")

    def test_show_weighted_avg_adds_shapes(self):
        import plotly.graph_objects as go

        from diff_diff.visualization import plot_bacon

        results = MagicMock()
        results.comparisons = [
            MagicMock(comparison_type="treated_vs_never", estimate=1.0, weight=0.5),
            MagicMock(comparison_type="earlier_vs_later", estimate=0.8, weight=0.3),
            MagicMock(comparison_type="later_vs_earlier", estimate=0.6, weight=0.2),
        ]
        results.twfe_estimate = 0.85
        results.effect_by_type.return_value = {
            "treated_vs_never": 1.0,
            "earlier_vs_later": 0.8,
            "later_vs_earlier": 0.6,
        }

        fig = plot_bacon(
            results,
            show_weighted_avg=True,
            show_twfe_line=True,
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)
        # Should have vertical line shapes (weighted avg + TWFE + zero line)
        shapes = fig.layout.shapes
        assert len(shapes) >= 4  # 3 weighted avg + 1 TWFE + zero line
