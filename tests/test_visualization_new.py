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
    results.alpha = 0.05  # real numeric fit alpha (drives the band label)
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

    @staticmethod
    def _band_vertices(ax):
        import numpy as np

        polys = [c for c in ax.collections if hasattr(c, "get_paths")]
        if not polys:
            return np.empty((0, 2))
        return np.vstack([p.vertices for c in polys for p in c.get_paths()])

    def test_invalid_se_rows_masked_with_warning(self):
        """Zero/negative/non-finite se rows lose their band (not zero-width)
        while the effect line keeps every row."""
        from diff_diff import plot_dose_response

        df = pd.DataFrame(
            {
                "dose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "effect": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "se": [0.05, 0.0, np.nan, -0.1, np.inf, 0.06],
            }
        )
        with pytest.warns(UserWarning, match="4 row\\(s\\) with non-positive or non-finite 'se'"):
            ax = plot_dose_response(data=df, show=False)
        verts = self._band_vertices(ax)
        finite = verts[np.isfinite(verts).all(axis=1)]
        for masked_x in (2.0, 3.0, 4.0, 5.0):
            at_x = finite[np.isclose(finite[:, 0], masked_x)]
            # No band interval may survive at a masked dose (degenerate
            # single-y fill endpoints are tolerated; a real interval is not).
            assert at_x.size == 0 or np.allclose(at_x[:, 1], at_x[0, 1])
        # Effects untouched: the effect line still carries all six rows.
        assert len(ax.lines[-1].get_xdata()) == 6

    def test_valid_se_no_warning_and_alpha_bounds(self):
        import warnings

        from diff_diff import plot_dose_response

        df = pd.DataFrame({"dose": [1.0, 2.0], "effect": [0.1, 0.2], "se": [0.05, 0.06]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot_dose_response(data=df, show=False)
        for bad in (1.0, 0.0):
            with pytest.raises(ValueError, match="strictly between 0 and 1"):
                plot_dose_response(data=df, alpha=bad, show=False)

    def test_band_labels_state_level_only_where_knowable(
        self, continuous_results, dose_response_curve
    ):
        from diff_diff import plot_dose_response

        def label_of(ax):
            return [t.get_text() for t in ax.get_legend().get_texts()]

        df_se = pd.DataFrame({"dose": [1.0, 2.0], "effect": [0.1, 0.2], "se": [0.05, 0.06]})
        assert "90% CI" in label_of(plot_dose_response(data=df_se, alpha=0.10, show=False))
        assert "95% CI" in label_of(plot_dose_response(data=df_se, show=False))
        # results= carries a knowable level (results.alpha).
        assert "95% CI" in label_of(plot_dose_response(continuous_results, show=False))
        # Bare curve and explicit-CI input: level-free.
        labels_curve = label_of(plot_dose_response(curve=dose_response_curve, show=False))
        assert "CI" in labels_curve and "95% CI" not in labels_curve
        df_ci = pd.DataFrame(
            {
                "dose": [1.0, 2.0],
                "effect": [0.1, 0.2],
                "conf_int_lower": [0.0, 0.1],
                "conf_int_upper": [0.2, 0.3],
            }
        )
        labels_ci = label_of(plot_dose_response(data=df_ci, show=False))
        assert "CI" in labels_ci and "95% CI" not in labels_ci

    def test_results_numpy_scalar_alpha_labels_level(self, continuous_results):
        # np.float32 is a real numeric fit alpha (not a float subclass) -
        # the level is knowable and must be labeled.
        from diff_diff import plot_dose_response

        continuous_results.alpha = np.float32(0.05)
        ax = plot_dose_response(continuous_results, show=False)
        assert "95% CI" in [t.get_text() for t in ax.get_legend().get_texts()]

    def test_fractional_coverage_labeled_exactly(self, continuous_results):
        # alpha=0.025 is a 97.5% interval - the label must not truncate or
        # round it to a whole percentage.
        from diff_diff import plot_dose_response

        df = pd.DataFrame({"dose": [1.0, 2.0], "effect": [0.1, 0.2], "se": [0.05, 0.06]})
        ax = plot_dose_response(data=df, alpha=0.025, show=False)
        assert "97.5% CI" in [t.get_text() for t in ax.get_legend().get_texts()]
        continuous_results.alpha = 0.025
        ax_r = plot_dose_response(continuous_results, show=False)
        assert "97.5% CI" in [t.get_text() for t in ax_r.get_legend().get_texts()]

    def test_explicit_alpha_warns_on_every_no_op_path(
        self, continuous_results, dose_response_curve
    ):
        from diff_diff import plot_dose_response

        df_ci = pd.DataFrame(
            {
                "dose": [1.0, 2.0],
                "effect": [0.1, 0.2],
                "conf_int_lower": [0.0, 0.1],
                "conf_int_upper": [0.2, 0.3],
            }
        )
        df_bare = pd.DataFrame({"dose": [1.0, 2.0], "effect": [0.1, 0.2]})
        for kwargs in (
            {"results": continuous_results},
            {"curve": dose_response_curve},
            {"data": df_ci},
            {"data": df_bare},
        ):
            with pytest.warns(UserWarning, match="only applies to DataFrame input"):
                plot_dose_response(alpha=0.10, show=False, **kwargs)

    def test_all_masked_se_no_band_no_stray_legend(self):
        from diff_diff import plot_dose_response

        df = pd.DataFrame(
            {"dose": [1.0, 2.0, 3.0], "effect": [0.1, 0.2, 0.3], "se": [0.0, np.nan, -1.0]}
        )
        with pytest.warns(UserWarning, match="3 row"):
            ax = plot_dose_response(data=df, show=False)
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Effect"]

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


class TestPlotEventStudyZeroSE:
    """Zero-SE rows draw no finite zero-width pointwise interval, while an
    auto-inferred reference row keeps its degenerate constraint bar (the
    plot_group_effects twin gate + the REGISTRY reference-retention
    contract)."""

    @staticmethod
    def _fake(ref_conf_int):
        nan = float("nan")

        class _Fake:
            anticipation = 0

        f = _Fake()
        f.event_study_effects = {
            -1: {
                "effect": 0.0,
                "se": 0.0,
                "t_stat": nan,
                "p_value": nan,
                "conf_int": ref_conf_int,
                "n_obs": 0,
            },
            0: {
                "effect": 1.5,
                "se": 0.0,
                "t_stat": nan,
                "p_value": nan,
                "conf_int": (nan, nan),
                "n_obs": 8,
            },
            1: {
                "effect": 1.0,
                "se": 0.5,
                "t_stat": 2.0,
                "p_value": 0.045,
                "conf_int": (0.02, 1.98),
                "n_obs": 8,
            },
        }
        return f

    @staticmethod
    def _yerr_segments(ax):
        segs = []
        for coll in ax.collections:
            if hasattr(coll, "get_segments"):
                segs.extend(coll.get_segments())
        return segs

    @pytest.mark.parametrize(
        "ref_conf_int",
        [(0.0, 0.0), (float("nan"), float("nan"))],  # Imputation vs StackedDiD shapes
    )
    def test_zero_se_gate_and_reference_retention(self, ref_conf_int):
        from diff_diff.visualization import plot_event_study

        ax = plot_event_study(self._fake(ref_conf_int), show=False)
        # x positions are ordinal: -1 -> 0, 0 -> 1, 1 -> 2
        saw_reference_bar = False
        for seg in self._yerr_segments(ax):
            if len(seg) != 2:
                continue
            x, lo_y, hi_y = float(seg[0][0]), float(seg[0][1]), float(seg[1][1])
            if abs(x - 1.0) < 1e-9:  # the zero-SE NON-reference row
                assert not (
                    abs(lo_y - 1.5) < 1e-12 and abs(hi_y - 1.5) < 1e-12
                ), "zero-SE non-reference row drawn with a finite zero-width CI"
            if abs(x - 0.0) < 1e-9 and abs(lo_y) < 1e-12 and abs(hi_y) < 1e-12:
                saw_reference_bar = True
        assert saw_reference_bar, (
            "auto-inferred reference row lost its degenerate (0, 0) bar "
            "(REGISTRY: retained for auto-inferred)"
        )
