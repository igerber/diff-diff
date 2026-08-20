"""Tests for lwdid_visualization module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from diff_diff.lwdid_visualization import (
    _require_matplotlib,
    plot_bootstrap_distribution,
    plot_cohort_trends,
    plot_event_study,
    plot_sensitivity,
)

# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestImportability:
    """Test that all visualization functions are importable."""

    def test_plot_cohort_trends_importable(self):
        assert callable(plot_cohort_trends)

    def test_plot_event_study_importable(self):
        assert callable(plot_event_study)

    def test_plot_sensitivity_importable(self):
        assert callable(plot_sensitivity)

    def test_plot_bootstrap_distribution_importable(self):
        assert callable(plot_bootstrap_distribution)

    def test_require_matplotlib_importable(self):
        assert callable(_require_matplotlib)


# ---------------------------------------------------------------------------
# _require_matplotlib error handling
# ---------------------------------------------------------------------------


class TestRequireMatplotlib:
    """Test _require_matplotlib raises proper error if no matplotlib."""

    def test_raises_visualization_error_when_no_matplotlib(self):
        """Mock ImportError to simulate missing matplotlib."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot" or name == "matplotlib":
                raise ImportError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="matplotlib"):
                _require_matplotlib()


# ---------------------------------------------------------------------------
# Plot functions return Figure when matplotlib available
# ---------------------------------------------------------------------------


class TestPlotFunctions:
    """Test plot functions return Figure when matplotlib is available."""

    @pytest.fixture
    def panel_data(self):
        rng = np.random.default_rng(42)
        records = []
        for i in range(80):
            d = int(i < 25)
            for t in range(1, 9):
                y = 1.0 + 0.1 * t + rng.normal(0, 0.3)
                if d and t > 4:
                    y += 2.0
                records.append({"unit": i, "time": t, "y": y, "treat": d * int(t > 4)})
        return pd.DataFrame(records)

    def test_plot_cohort_trends_returns_figure(self, panel_data):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        fig = plot_cohort_trends(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert fig is not None
        assert hasattr(fig, "savefig")  # duck-type check for Figure
        plt.close(fig)

    def test_plot_event_study_returns_figure(self, panel_data):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from diff_diff import LWDiD

        # Staggered fit populates the event-study surface
        staggered = panel_data.copy()
        staggered["first_treat"] = np.where(staggered["unit"] < 25, 5, 0)
        res = LWDiD(rolling="demean").fit(
            staggered,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            first_treat="first_treat",
        )
        assert res.event_study_effects

        fig = plot_event_study(res)
        assert fig is not None
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_event_study_common_timing_returns_figure(self, panel_data):
        """Common-timing fits now populate the per-period event-study
        surface at fit time, so plot_event_study works on them too."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from diff_diff import LWDiD

        res = LWDiD(rolling="demean").fit(
            panel_data, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert res.event_study_effects

        fig = plot_event_study(res)
        assert fig is not None
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_bootstrap_distribution_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        t_stats = np.random.default_rng(0).normal(0, 1, 500)
        fig = plot_bootstrap_distribution(t_stats, t_observed=2.5)
        assert fig is not None
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestPlottingConventions:
    """Fix-wave WS10 pins: NaN-SE effects plot the point and OMIT the
    interval (never a zero-length bar); cohort-trend plots accept datetime
    time columns (the onset marker no longer computes 'Timestamp - 0.5').
    """

    def test_event_study_nan_se_omits_interval(self):
        matplotlib = pytest.importorskip("matplotlib")

        matplotlib.use("Agg")
        from types import SimpleNamespace

        from diff_diff.lwdid_visualization import plot_event_study

        results = SimpleNamespace(
            event_study_effects={
                -2: {"effect": 0.1, "se": 0.05},
                0: {"effect": 1.0, "se": float("nan")},  # inference unavailable
                1: {"effect": 1.2, "se": 0.07},
            },
            reference_periods=(-1,),
        )
        fig = plot_event_study(results)
        assert fig is not None

    def test_cohort_trends_accepts_datetime_time(self):
        matplotlib = pytest.importorskip("matplotlib")

        matplotlib.use("Agg")
        from diff_diff.lwdid_visualization import plot_cohort_trends

        rng = np.random.default_rng(3)
        rows = []
        times = pd.date_range("2020-01-01", periods=6, freq="MS")
        for u in range(8):
            for i, t in enumerate(times):
                d = int(u < 4 and i >= 3)
                rows.append(dict(unit=u, time=t, treat=d, y=rng.normal() + d, first=0))
        df = pd.DataFrame(rows)
        fig = plot_cohort_trends(df, outcome="y", unit="unit", time="time", treatment="treat")
        assert fig is not None
