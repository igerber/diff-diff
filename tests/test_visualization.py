"""
Tests for visualization functions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from legacy_surface import es_dict_from_container, legacy_standin

from diff_diff import (
    CallawaySantAnna,
    TwoWayFixedEffects,
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

                data.append(
                    {
                        "outcome": y,
                        "treated": treated,
                        "period": period,
                    }
                )

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

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcomes,
            "first_treat": first_treat_expanded.astype(int),
        }
    )


@pytest.fixture
def twfe_es_results():
    """TWFE event-study surface (pooled design) on the multi-period panel - the
    successor of the former MultiPeriodDiD fixture for the plot-option tests."""
    data = generate_multi_period_data()
    return TwoWayFixedEffects().fit(
        data,
        outcome="outcome",
        treatment="treated",
        time="period",
        event_study=True,
        spec="pooled",
        post_periods=[2, 3],
        reference_period=1,
    )


class TestPlotEventStudy:
    """Tests for plot_event_study function."""

    @pytest.fixture
    def cs_fit(self):
        """A plain CallawaySantAnna fit (4.0: no fit-time event-study surface)."""
        data = generate_staggered_data()
        cs = CallawaySantAnna()
        return cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

    @pytest.fixture
    def cs_results(self, cs_fit):
        """The post-fit event-study container - what plot_event_study consumes."""
        return cs_fit.aggregate("event_study")

    def test_plot_from_twfe_event_study_results(self, twfe_es_results):
        """Test plotting from the TWFE event-study surface."""
        pytest.importorskip("matplotlib")

        ax = plot_event_study(twfe_es_results, show=False)
        assert ax is not None

    def test_plot_from_cs_results(self, cs_results):
        """Test plotting from the CallawaySantAnna event-study container."""
        pytest.importorskip("matplotlib")

        ax = plot_event_study(cs_results, show=False)
        assert ax is not None

    def test_cs_results_object_has_no_fit_time_surface(self, cs_fit):
        """4.0 invariant: a CS results object carries no event-study surface.

        Every CallawaySantAnna fit leaves the legacy field None; plotting the
        results object itself fails closed and points at the container.
        """
        assert cs_fit.event_study_effects is None
        with pytest.raises(TypeError, match=r"aggregate\(\s*'event_study'\)"):
            plot_event_study(cs_fit, show=False)

    def test_plot_from_legacy_raw_field_results(self, cs_fit):
        """Raw-field branch pin: a results object whose ``event_study_effects``
        dict is populated by hand (the legacy-pickle shape) still plots, and
        the plotted effects equal the container's.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from diff_diff.visualization import _extract_plot_data

        standin = legacy_standin(cs_fit)
        assert standin.event_study_effects is not None
        es = cs_fit.aggregate("event_study")

        effects, se, periods, _, _, ref, inferred, *_ = _extract_plot_data(
            standin, None, None, None, None
        )
        assert sorted(periods) == sorted(es.event_time.tolist())
        for i, e in enumerate(es.event_time.tolist()):
            assert effects[e] == pytest.approx(float(es.att[i]), nan_ok=True)
            assert se[e] == pytest.approx(float(es.se[i]), nan_ok=True)
        # Varying-base CS marks no reference row on the container; the raw-field
        # branch keeps its legacy ``-1`` fallback for that case.
        assert ref == (es.reference_period if es.reference_period is not None else -1)
        assert inferred is True

        ax = plot_event_study(standin, show=False)
        assert ax is not None
        plt.close()

    def test_plot_from_dataframe(self):
        """Test plotting from DataFrame."""
        pytest.importorskip("matplotlib")

        df = pd.DataFrame(
            {
                "period": [-2, -1, 0, 1, 2],
                "effect": [0.1, 0.05, 0.0, 0.5, 0.6],
                "se": [0.1, 0.1, 0.0, 0.15, 0.15],
            }
        )

        ax = plot_event_study(df, reference_period=0, show=False)
        assert ax is not None

    def test_plot_from_dataframe_with_cband(self):
        """Test that cband_lower/cband_upper columns are used as CI overrides."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import _extract_plot_data

        effects = [0.1, 0.05, 0.0, 0.5, 0.6]
        se_vals = [0.1, 0.1, 0.0, 0.15, 0.15]
        periods = [-2, -1, 0, 1, 2]
        # cband bounds wider than pointwise 95% CI (effect ± 1.96*se)
        cband_lower = [e - 3 * s for e, s in zip(effects, se_vals)]
        cband_upper = [e + 3 * s for e, s in zip(effects, se_vals)]

        df = pd.DataFrame(
            {
                "period": periods,
                "effect": effects,
                "se": se_vals,
                "cband_lower": cband_lower,
                "cband_upper": cband_upper,
            }
        )

        # Verify _extract_plot_data returns cband overrides
        result = _extract_plot_data(
            df, periods=None, pre_periods=None, post_periods=None, reference_period=0
        )
        ci_lo = result[7]
        ci_hi = result[8]
        assert ci_lo is not None, "ci_lower_override should not be None with cband columns"
        assert ci_hi is not None, "ci_upper_override should not be None with cband columns"
        for p, el, eu in zip(periods, cband_lower, cband_upper):
            assert ci_lo[p] == el
            assert ci_hi[p] == eu

        # Verify plot renders successfully
        ax = plot_event_study(df, reference_period=0, show=False)
        assert ax is not None

    def test_plot_from_dataframe_with_nan_cband(self):
        """Test that NaN cband columns fall back to pointwise CIs."""
        pytest.importorskip("matplotlib")
        from diff_diff.visualization import _extract_plot_data

        df = pd.DataFrame(
            {
                "period": [-2, -1, 0, 1, 2],
                "effect": [0.1, 0.05, 0.0, 0.5, 0.6],
                "se": [0.1, 0.1, 0.0, 0.15, 0.15],
                "cband_lower": [np.nan] * 5,
                "cband_upper": [np.nan] * 5,
            }
        )

        # All-NaN cband columns should not produce overrides
        result = _extract_plot_data(
            df, periods=None, pre_periods=None, post_periods=None, reference_period=0
        )
        ci_lo = result[7]
        ci_hi = result[8]
        assert ci_lo is None, "ci_lower_override should be None when cband is all-NaN"
        assert ci_hi is None, "ci_upper_override should be None when cband is all-NaN"

        # Plot should still work (falls back to pointwise CIs)
        ax = plot_event_study(df, reference_period=0, show=False)
        assert ax is not None

    def test_plot_from_dict(self):
        """Test plotting from dictionaries."""
        pytest.importorskip("matplotlib")

        effects = {-2: 0.1, -1: 0.05, 0: 0.0, 1: 0.5, 2: 0.6}
        se = {-2: 0.1, -1: 0.1, 0: 0.0, 1: 0.15, 2: 0.15}

        ax = plot_event_study(effects=effects, se=se, reference_period=0, show=False)
        assert ax is not None

    def test_plot_customization(self, twfe_es_results):
        """Test plot customization options."""
        pytest.importorskip("matplotlib")

        ax = plot_event_study(
            twfe_es_results,
            title="Custom Title",
            xlabel="Custom X",
            ylabel="Custom Y",
            color="red",
            marker="s",
            markersize=10,
            show=False,
        )

        assert ax.get_title() == "Custom Title"
        assert ax.get_xlabel() == "Custom X"
        assert ax.get_ylabel() == "Custom Y"

    def test_plot_no_zero_line(self, twfe_es_results):
        """Test disabling zero line."""
        pytest.importorskip("matplotlib")

        ax = plot_event_study(twfe_es_results, show_zero_line=False, show=False)
        assert ax is not None

    def test_plot_with_existing_axes(self, twfe_es_results):
        """Test plotting on existing axes."""
        matplotlib = pytest.importorskip("matplotlib")
        plt = matplotlib.pyplot

        fig, ax = plt.subplots()
        returned_ax = plot_event_study(twfe_es_results, ax=ax, show=False)
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
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]})

        with pytest.raises(ValueError, match="must have 'period' column"):
            plot_event_study(df)

    def test_error_invalid_results_type(self):
        """Test error with invalid results type."""
        pytest.importorskip("matplotlib")
        with pytest.raises(TypeError, match="Cannot extract plot data") as exc_info:
            plot_event_study("invalid")
        # The expected-types list names the unified container too.
        assert "EventStudyResults" in str(exc_info.value)

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

        ax = plot_event_study(effects=effects, se=se, reference_period=-1, show=False)

        # Verify the plot was created successfully
        assert ax is not None

        # Verify all 4 periods are plotted (including reference with NaN SE)
        # The x-axis should have 4 tick labels
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(xtick_labels) == 4
        assert "-1" in xtick_labels

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
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).aggregate("event_study")

        # Should not raise even with NaN SE in reference period
        ax = plot_event_study(results, show=False)
        assert ax is not None

        # Verify reference period (-1) is in the plot
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "-1" in xtick_labels

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
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).aggregate("event_study")

        # Reference period should be at e=-2 (not e=-1) with anticipation=1
        times = results.event_time.tolist()
        assert -2 in times
        assert bool(results.is_reference[times.index(-2)])
        assert results.reference_period == -2

        ax = plot_event_study(results, show=False)
        assert ax is not None

        # Verify -2 is in the plot (the true reference period)
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "-2" in xtick_labels

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
        df = pd.DataFrame(
            {
                "period": [-2, -1, 0, 1, 2],
                "effect": [0.1, 0.2, 0.3, 0.5, 0.6],  # ref at 0 has effect 0.3
                "se": [0.1, 0.1, 0.1, 0.1, 0.1],
            }
        )

        ax = plot_event_study(df, reference_period=0, show=False)

        # Find plotted y-values by extracting data from Line2D objects
        # The point estimates are plotted as individual markers
        y_values = []
        for child in ax.get_children():
            # Line2D objects with single points are our markers
            if hasattr(child, "get_ydata"):
                ydata = child.get_ydata()
                if len(ydata) == 1:
                    y_values.append(float(ydata[0]))

        # After normalization:
        # - Original effects: [0.1, 0.2, 0.3, 0.5, 0.6]
        # - Reference effect: 0.3
        # - Normalized: [-0.2, -0.1, 0.0, 0.2, 0.3]
        expected_normalized = [-0.2, -0.1, 0.0, 0.2, 0.3]

        # Check that reference period (0) is at y=0
        assert 0.0 in y_values or any(
            abs(y) < 0.01 for y in y_values
        ), f"Reference period should be at y=0, got y_values={y_values}"

        # Verify all expected normalized values are present
        for expected in expected_normalized:
            assert any(
                abs(y - expected) < 0.01 for y in y_values
            ), f"Expected normalized value {expected} not found in {y_values}"

        # Verify error bars: reference period (y=0) should have NO error bars
        # while other periods should have error bars
        # Error bars are drawn via ax.errorbar, which creates ErrorbarContainer or Line2D
        # The error bar x-coordinates tell us which periods have error bars

        # Find the errorbar data (the line segments that form error bars)
        errorbar_x_coords = set()
        for child in ax.get_children():
            # ErrorbarContainer's children include LineCollection for the caps/stems
            if hasattr(child, "get_segments"):
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
        assert (
            reference_x not in errorbar_x_coords
        ), f"Reference period should have no error bars but found error bar at x={reference_x}"

        # Other periods SHOULD have error bars
        # At least some of x=0, x=1, x=3, x=4 should have error bars
        non_ref_x_coords = {0, 1, 3, 4}
        assert (
            len(errorbar_x_coords & non_ref_x_coords) >= 2
        ), f"Non-reference periods should have error bars, found: {errorbar_x_coords}"

        plt.close()

    def test_plot_event_study_no_normalization_without_reference(self):
        """Test that effects are NOT normalized when reference_period is None."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = pd.DataFrame({"period": [-1, 0, 1], "effect": [0.1, 0.3, 0.5], "se": [0.1, 0.1, 0.1]})

        ax = plot_event_study(df, reference_period=None, show=False)

        # Extract y-values
        y_values = []
        for child in ax.get_children():
            if hasattr(child, "get_ydata"):
                ydata = child.get_ydata()
                if len(ydata) == 1:
                    y_values.append(float(ydata[0]))

        # Without normalization, original values should be preserved
        for expected in [0.1, 0.3, 0.5]:
            assert any(
                abs(y - expected) < 0.01 for y in y_values
            ), f"Original value {expected} not found in {y_values}"

        plt.close()

    def test_plot_event_study_normalization_with_nan_reference(self):
        """Test that normalization is skipped when reference effect is NaN."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "period": [-1, 0, 1],
                "effect": [0.1, np.nan, 0.5],  # Reference period has NaN effect
                "se": [0.1, 0.1, 0.1],
            }
        )

        # This should not raise and should skip normalization
        ax = plot_event_study(df, reference_period=0, show=False)

        # Extract y-values (NaN effect is skipped in plotting)
        y_values = []
        for child in ax.get_children():
            if hasattr(child, "get_ydata"):
                ydata = child.get_ydata()
                if len(ydata) == 1:
                    y_values.append(float(ydata[0]))

        # Original non-NaN values should be preserved (not normalized)
        for expected in [0.1, 0.5]:
            assert any(
                abs(y - expected) < 0.01 for y in y_values
            ), f"Original value {expected} not found in {y_values}"

        plt.close()

    def test_plot_cs_results_no_auto_normalization(self, cs_results):
        """Test that auto-inferred reference period does NOT normalize effects.

        When CallawaySantAnna results auto-infer reference_period=-1 (or from n_groups=0),
        effects should NOT be normalized (just hollow marker styling).
        Only explicit reference_period=X should trigger normalization.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # Use fixture instead of re-fitting
        results = cs_results

        # Get original effects from results (before any normalization)
        original_effects = {
            period: effect_data["effect"]
            for period, effect_data in es_dict_from_container(results).items()
        }

        # Plot WITHOUT explicitly specifying reference_period
        # This should auto-infer reference but NOT normalize
        ax = plot_event_study(results, show=False)

        # Extract plotted y-values
        y_values = []
        for child in ax.get_children():
            if hasattr(child, "get_ydata"):
                ydata = child.get_ydata()
                if len(ydata) == 1:
                    y_values.append(float(ydata[0]))

        # Verify that the original (non-normalized) effects are plotted
        # Check that at least some non-zero effects are preserved
        non_zero_originals = [e for e in original_effects.values() if abs(e) > 0.01]
        assert len(non_zero_originals) > 0, "Should have non-zero original effects"

        # The key check: effects should NOT all be relative to some reference
        # If normalized, reference would be at 0 and others shifted accordingly
        # Since NOT normalized, we should see the original effect values
        for period, orig_effect in original_effects.items():
            if np.isfinite(orig_effect):
                # Check that original value is present (not normalized)
                assert any(abs(y - orig_effect) < 0.05 for y in y_values), (
                    f"Original effect {orig_effect:.3f} for period {period} "
                    f"should be plotted without normalization. Found y_values: {y_values}"
                )

        plt.close()

    def test_plot_cs_results_explicit_reference_normalizes(self, cs_results):
        """Test that explicit reference_period normalizes CallawaySantAnna results.

        When user explicitly passes reference_period=X to plot_event_study,
        it should normalize effects (subtract ref effect) and set ref SE to NaN.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # Use fixture instead of re-fitting
        results = cs_results

        # Get original effects from results
        original_effects = {
            period: effect_data["effect"]
            for period, effect_data in es_dict_from_container(results).items()
        }

        # Choose reference period (typically -1)
        ref_period = -1
        ref_effect = original_effects.get(ref_period, 0.0)

        # Compute expected normalized effects
        expected_normalized = {
            period: effect - ref_effect for period, effect in original_effects.items()
        }

        # Plot WITH explicit reference_period - this SHOULD normalize
        ax = plot_event_study(results, reference_period=ref_period, show=False)

        # Extract plotted y-values
        y_values = []
        for child in ax.get_children():
            if hasattr(child, "get_ydata"):
                ydata = child.get_ydata()
                if len(ydata) == 1:
                    y_values.append(float(ydata[0]))

        # The reference period should now be at y=0 (normalized)
        assert any(
            abs(y) < 0.01 for y in y_values
        ), f"Reference period should be normalized to y=0, got y_values={y_values}"

        # Verify normalized values are present
        for period, norm_effect in expected_normalized.items():
            if np.isfinite(norm_effect):
                assert any(abs(y - norm_effect) < 0.05 for y in y_values), (
                    f"Normalized effect {norm_effect:.3f} for period {period} "
                    f"not found in {y_values}"
                )

        # Verify reference period has no error bars (SE was set to NaN)
        # Find error bar x-coordinates
        periods_in_plot = sorted(original_effects.keys())
        ref_x_idx = periods_in_plot.index(ref_period) if ref_period in periods_in_plot else None

        if ref_x_idx is not None:
            errorbar_x_coords = set()
            for child in ax.get_children():
                if hasattr(child, "get_segments"):
                    segments = child.get_segments()
                    for seg in segments:
                        if len(seg) >= 2:
                            errorbar_x_coords.add(round(seg[0][0], 1))

            # Reference period should NOT have error bars
            assert (
                ref_x_idx not in errorbar_x_coords
            ), f"Reference period at x={ref_x_idx} should have no error bars"

        plt.close()


class TestPlotEventStudyCband:
    """Tests for simultaneous confidence band (cband) support in plot_event_study."""

    @pytest.fixture
    def cs_cband_results(self):
        """Fixture for CallawaySantAnna results with cband enabled."""
        data = generate_staggered_data(n_units=50, n_periods=8, seed=42)
        cs = CallawaySantAnna(n_bootstrap=199, seed=42, cband=True)
        return cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        ).aggregate("event_study")

    def test_plot_uses_cband_cis_by_default(self, cs_cband_results):
        """Test that cband CIs are used by default when available."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from diff_diff.visualization import _extract_plot_data

        # The container carries the simultaneous band.
        assert cs_cband_results.cband_lower is not None
        assert cs_cband_results.cband_upper is not None
        assert np.isfinite(cs_cband_results.cband_crit_value)
        finite = np.isfinite(cs_cband_results.cband_lower) & np.isfinite(
            cs_cband_results.cband_upper
        )
        assert finite.any()
        # The band is at least as wide as the pointwise interval on every finite row.
        assert np.all(
            cs_cband_results.cband_lower[finite] <= cs_cband_results.conf_int_lower[finite] + 1e-12
        )
        assert np.all(
            cs_cband_results.cband_upper[finite] >= cs_cband_results.conf_int_upper[finite] - 1e-12
        )

        # Verify plot succeeds
        ax = plot_event_study(cs_cband_results, show=False)
        assert ax is not None

        # Verify cband CIs are extracted and equal the container's band
        _, _, _, _, _, _, _, ci_lower_override, ci_upper_override, _, _, _ = _extract_plot_data(
            cs_cband_results, None, None, None, None
        )
        assert ci_lower_override is not None
        assert ci_upper_override is not None
        for i, e in enumerate(cs_cband_results.event_time.tolist()):
            if finite[i]:
                assert ci_lower_override[e] == float(cs_cband_results.cband_lower[i])
                assert ci_upper_override[e] == float(cs_cband_results.cband_upper[i])

        plt.close()

    def test_plot_use_cband_false(self, cs_cband_results):
        """Test that use_cband=False suppresses cband CIs."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # Should succeed and use pointwise CIs instead
        ax = plot_event_study(cs_cband_results, use_cband=False, show=False)
        assert ax is not None

        plt.close()


class TestPlotEventStudyIntegration:
    """Integration tests for event study plotting."""

    def test_full_workflow_twfe_event_study(self):
        """Test full workflow with the TWFE event-study mode."""
        matplotlib = pytest.importorskip("matplotlib")
        plt = matplotlib.pyplot

        # Generate data
        data = generate_multi_period_data()

        # Fit model
        results = TwoWayFixedEffects().fit(
            data,
            outcome="outcome",
            treatment="treated",
            time="period",
            event_study=True,
            spec="pooled",
            post_periods=[2, 3],
            reference_period=1,
        )

        # Plot
        ax = plot_event_study(results, title="Treatment Effects Over Time", show=False)

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
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Plot the post-fit container
        ax = plot_event_study(
            results.aggregate("event_study"), title="Staggered DiD Event Study", show=False
        )

        assert ax is not None
        plt.close()


class TestEfficientDiDPlotReference:
    """M-023: the membership-gated ``reference_period`` property corrects the
    plotted reference on PT-Post EfficientDiD fits.

    Before M-023 the native extraction fell back to ``-1`` for EDiD; under
    ``pt_assumption="post"`` with ``anticipation=1`` the materialized
    mechanical anchor sits at ``e = -2``, and the property now feeds the
    extractor the true value.  PT-All fits have no anchor (the property is
    None) and keep the legacy ``-1`` fallback unchanged.
    """

    def test_pt_post_anticipation_reference_shifts(self):
        pytest.importorskip("matplotlib")

        from diff_diff import EfficientDiD
        from diff_diff.prep_dgp import generate_staggered_data
        from diff_diff.visualization._event_study import _extract_plot_data

        d = generate_staggered_data(n_units=80, n_periods=8, cohort_periods=[4, 6], seed=9)
        res = EfficientDiD(pt_assumption="post", anticipation=1).fit(
            d,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # 4.0: EfficientDiD stores no fit-time surface, so the MEMBERSHIP-GATED
        # property is None on the results object itself ...
        assert res.event_study_effects is None
        assert res.reference_period is None
        # ... and reads -2 on a surface-bearing stand-in (the legacy-pickle shape),
        # while the container carries the anchor as its single is_reference row.
        assert legacy_standin(res).reference_period == -2
        es = res.aggregate("event_study")
        assert es.reference_period == -2
        extracted = _extract_plot_data(es, None, None, None, None)
        assert extracted[5] == -2  # reference_period slot
        assert extracted[6] is True  # inferred, not caller-supplied


def _plotted_points(ax):
    """Point estimates drawn by plot_event_study: [(x, y, is_hollow)] in x order."""
    pts = []
    for line in ax.lines:
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size != 1 or line.get_marker() in (None, "None", ""):
            continue
        hollow = line.get_markerfacecolor() in ("white", "w")
        pts.append((float(np.asarray(line.get_xdata())[0]), float(ydata[0]), hollow))
    return sorted(pts)


def _plotted_intervals(ax):
    """Vertical error-bar segments drawn by plot_event_study: {x: (lower, upper)}."""
    out = {}
    for coll in ax.collections:
        if not hasattr(coll, "get_segments"):
            continue
        for seg in coll.get_segments():
            seg = np.asarray(seg, dtype=float)
            if seg.shape == (2, 2) and seg[0, 0] == seg[1, 0]:
                out[float(seg[0, 0])] = (float(seg[:, 1].min()), float(seg[:, 1].max()))
    return out


def _zero_df_stacked_fit(zero_df=True):
    """StackedDiD survey fit on six units. With one unit per stratum under
    lonely_psu="adjust" the design has zero degrees of freedom: finite SEs,
    NaN stored intervals. With three strata of two units inference is defined."""
    from diff_diff import StackedDiD, SurveyDesign

    cohorts = [3, 3, np.inf, np.inf, np.inf, np.inf]
    strata = list(range(6)) if zero_df else [0, 1, 2, 0, 1, 2]
    rows = []
    for unit, g in enumerate(cohorts):
        for t in range(9):
            effect = 2.0 + 0.5 * unit if (np.isfinite(g) and t >= g) else 0.0
            y = 0.5 * t + unit + effect + 0.3 * ((unit * 7 + t * 3) % 5)
            rows.append((unit, t, g, y, strata[unit], 1.0))
    data = pd.DataFrame(rows, columns=["unit", "period", "first_treat", "outcome", "stratum", "w"])
    design = SurveyDesign(weights="w", strata="stratum", lonely_psu="adjust")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StackedDiD(kappa_pre=2, kappa_post=2).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            survey_design=design,
        )


class TestPlotEventStudyUndefinedStoredInference:
    """A results object passed directly has its intervals recomputed from the
    SE. A row whose STORED interval is NaN while its SE is finite (inference
    failed closed, e.g. a survey design with zero degrees of freedom) must not
    get one back."""

    def test_fit_has_finite_se_and_nan_intervals(self):
        results = _zero_df_stacked_fit()
        rows = [e for e in results.event_study_effects.values() if not e.get("is_reference")]
        assert rows and all(np.isfinite(e["se"]) and e["se"] > 0 for e in rows)
        assert all(np.isnan(e["conf_int"][0]) and np.isnan(e["conf_int"][1]) for e in rows)

    @pytest.mark.parametrize("kwargs", [{}, {"use_cband": False}, {"reference_period": 0}])
    def test_no_estimated_row_gets_an_interval(self, kwargs):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        results = _zero_df_stacked_fit()
        # Without an explicit reference the auto-inferred reference row keeps
        # its zero-width constraint bar (REGISTRY Event Study Plotting); no
        # estimated row may carry an interval on either route.
        for source in (results, results.aggregate("event_study")):
            ax = plot_event_study(source, show=False, **kwargs)
            assert len({p[0] for p in _plotted_points(ax)}) == len(results.event_study_effects)
            drawn = _plotted_intervals(ax)
            assert all(lower == upper == 0.0 for lower, upper in drawn.values())
            assert len(drawn) <= 1
            plt.close()

    @pytest.mark.parametrize("kwargs", [{}, {"use_cband": False}, {"reference_period": 0}])
    def test_plotly_draws_no_band(self, kwargs):
        pytest.importorskip("plotly")

        def band_heights(fig):
            return [
                float(y)
                for trace in fig.data
                if getattr(trace, "fill", None) == "toself"
                for y in trace.y
                if y is not None and np.isfinite(y)
            ]

        fig = plot_event_study(_zero_df_stacked_fit(), backend="plotly", show=False, **kwargs)
        assert all(y == 0.0 for y in band_heights(fig))
        # positive control: defined inference draws a band on the same route
        fig = plot_event_study(
            _zero_df_stacked_fit(zero_df=False), backend="plotly", show=False, **kwargs
        )
        assert any(y != 0.0 for y in band_heights(fig))

    @pytest.mark.parametrize("kwargs", [{}, {"use_cband": False}, {"reference_period": -1}])
    def test_one_undefined_row_among_defined_rows(self, kwargs):
        """Only the row with undefined inference loses its interval."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        results = _zero_df_stacked_fit(zero_df=False)
        row = results.event_study_effects[1]
        assert np.isfinite(row["se"]) and np.isfinite(row["conf_int"][0])
        row.update(t_stat=np.nan, p_value=np.nan, conf_int=(np.nan, np.nan))

        ax = plot_event_study(results, show=False, **kwargs)
        horizons = sorted(results.event_study_effects)
        drawn = _plotted_intervals(ax)
        assert float(horizons.index(1)) not in drawn
        assert all(float(horizons.index(h)) in drawn for h in horizons if h not in (1, -1))
        assert float(horizons.index(1)) in {p[0] for p in _plotted_points(ax)}
        plt.close()

    @pytest.mark.parametrize(
        "stored", [(float("nan"), 3.1), (0.4, float("nan")), (None, 3.1), (0.4, None)]
    )
    def test_a_partly_missing_stored_interval_is_not_rebuilt(self, stored):
        """One finite endpoint is not an interval: the row keeps its marker and
        gets no bar, as it does through the container."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        results = _zero_df_stacked_fit(zero_df=False)
        results.event_study_effects[1].update(p_value=np.nan, conf_int=stored)
        horizons = sorted(results.event_study_effects)
        ax = plot_event_study(results, show=False)
        drawn = _plotted_intervals(ax)
        assert float(horizons.index(1)) not in drawn
        assert float(horizons.index(0)) in drawn and float(horizons.index(2)) in drawn
        plt.close()

    def test_a_row_that_stores_no_interval_keeps_the_recomputed_one(self):
        """Legacy rows without a ``conf_int`` key are still drawn from the SE."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        results = _zero_df_stacked_fit(zero_df=False)
        del results.event_study_effects[1]["conf_int"]
        horizons = sorted(results.event_study_effects)
        ax = plot_event_study(results, show=False)
        assert float(horizons.index(1)) in _plotted_intervals(ax)
        plt.close()


class TestPlotEventStudyRawDCDHUndefinedInference:
    """The raw dCDH route recomputes intervals from the SE too. An effect or
    placebo horizon whose stored interval is NaN plots with no interval."""

    @staticmethod
    def _dcdh(conf_int, placebo=None):
        from types import SimpleNamespace

        entry = {
            "effect": 1.4,
            "se": 0.3,
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "conf_int": conf_int,
            "n_obs": 40,
        }
        return SimpleNamespace(L_max=1, placebo_event_study=placebo, event_study_effects={1: entry})

    def test_undefined_effect_and_placebo_get_no_interval(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        nan_ci = (float("nan"), float("nan"))
        placebo = {-1: {"effect": 0.1, "se": 0.2, "conf_int": nan_ci, "p_value": float("nan")}}
        ax = plot_event_study(self._dcdh(nan_ci, placebo=placebo), show=False)
        assert _plotted_intervals(ax) == {}
        assert len({p[0] for p in _plotted_points(ax)}) == 3
        plt.close()

    def test_defined_effect_keeps_its_interval(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        ax = plot_event_study(self._dcdh((0.8, 2.0)), show=False)
        (interval,) = _plotted_intervals(ax).values()
        half = 1.959963984540054 * 0.3
        assert interval == pytest.approx((1.4 - half, 1.4 + half))
        plt.close()

    def test_plotly_draws_no_band_for_an_undefined_horizon(self):
        pytest.importorskip("plotly")

        fig = plot_event_study(
            self._dcdh((float("nan"), float("nan"))), backend="plotly", show=False
        )
        assert not [t for t in fig.data if getattr(t, "fill", None) == "toself"]


class TestPlotEventStudyFittedUndefinedRow:
    """End to end on a fitted model: SunAbraham with ``vcov_type="hc2_bm"``.
    Calendar period 3.5 is observed only for the one-unit cohort adopting at
    3.5 and for the never-treated units, so only that unit identifies event
    time 0: the fit stores a finite SE and NaN inference there."""

    @staticmethod
    def _fit():
        from diff_diff import SunAbraham

        rows, unit = [], 0
        for g, n, has_half_period in ((2.5, 10, False), (3.5, 1, True), (np.inf, 12, True)):
            for _ in range(n):
                for t in [0, 1, 2, 3, 4, 5, 6, 7] + ([3.5] if has_half_period else []):
                    effect = 2.0 if (np.isfinite(g) and t >= g) else 0.0
                    y = 0.3 * t + (unit % 4) + effect + 0.2 * ((unit * 5 + int(t * 2) * 3) % 7)
                    rows.append((unit, t, g, y))
                unit += 1
        data = pd.DataFrame(rows, columns=["unit", "period", "first_treat", "outcome"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SunAbraham(vcov_type="hc2_bm").fit(
                data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
            )

    @pytest.mark.parametrize("kwargs", [{}, {"use_cband": False}])
    def test_event_time_zero_gets_no_interval(self, kwargs):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        results = self._fit()
        surface = results.event_study_effects
        row = surface[0.0]
        assert np.isfinite(row["se"]) and row["se"] > 0 and np.isnan(row["conf_int"][0])
        horizons = sorted(surface)
        ax = plot_event_study(results, show=False, **kwargs)
        drawn = _plotted_intervals(ax)
        assert float(horizons.index(0.0)) not in drawn
        for h in (-1.5, 0.5):
            assert np.isfinite(surface[h]["conf_int"][0])
            assert float(horizons.index(h)) in drawn
        plt.close()


class TestPlotEventStudyWooldridge:
    """plot_event_study accepts a WooldridgeDiDResults by converting it to the
    unified container, so it takes the EventStudyResults route end to end."""

    @staticmethod
    def _fit(**kwargs):
        from diff_diff import WooldridgeDiD
        from diff_diff.prep_dgp import generate_staggered_data as _gen

        data = _gen(n_units=120, n_periods=8, seed=1)
        return WooldridgeDiD(**kwargs).fit(
            data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )

    def test_plots_the_stored_surface(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        res = self._fit()
        res.aggregate("event_study")
        frame = res.to_dataframe(level="event_study")
        # A Wooldridge event-study surface carries no reference row.
        assert not frame["is_reference"].any()

        ax = plot_event_study(res, show=False)

        points = _plotted_points(ax)
        assert len(points) == len(frame)
        np.testing.assert_allclose([p[1] for p in points], frame["att"].to_numpy(), rtol=1e-12)
        assert not any(p[2] for p in points)  # nothing drawn as a reference marker
        intervals = _plotted_intervals(ax)
        assert len(intervals) == len(frame)
        for x, (_, row) in enumerate(frame.iterrows()):
            lo, hi = intervals[float(x)]
            np.testing.assert_allclose(
                [lo, hi], [row["conf_int_lower"], row["conf_int_upper"]], rtol=1e-10
            )
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            str(e) for e in frame["event_time"].tolist()
        ]
        plt.close()

    def test_matches_the_container_route(self):
        """The raw result and build_event_study_surface(result) plot identically."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from diff_diff.results_base import build_event_study_surface

        res = self._fit()
        res.aggregate("event_study")
        ax_raw = plot_event_study(res, show=False)
        pts_raw, ci_raw = _plotted_points(ax_raw), _plotted_intervals(ax_raw)
        plt.close()
        ax_cont = plot_event_study(build_event_study_surface(res), show=False)
        assert _plotted_points(ax_cont) == pts_raw
        assert _plotted_intervals(ax_cont) == ci_raw
        plt.close()

    def test_unaggregated_fit_raises(self):
        pytest.importorskip("matplotlib")

        res = self._fit()
        assert res.event_study_effects is None
        with pytest.raises(ValueError, match=r"aggregate\('event_study'\)"):
            plot_event_study(res, show=False)

    def test_stored_alpha_mismatch_warns(self):
        """Stored intervals are at the fit's level; a different plot alpha warns."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        res = self._fit(alpha=0.10)
        res.aggregate("event_study")
        with pytest.warns(UserWarning, match="does not apply to an EventStudyResults container"):
            plot_event_study(res, show=False)  # default alpha=0.05
        plt.close()

    def test_cohort_share_draws_no_error_bars(self):
        """The fail-closed cohort-share inference (NaN intervals) is preserved:
        no interval is recomputed from the conditional-on-shares SE."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        res = self._fit()
        with pytest.warns(UserWarning, match="conditional-on-shares"):
            res.aggregate("event_study", weights="cohort_share")
        frame = res.to_dataframe(level="event_study")
        assert np.isfinite(frame["se"]).all()
        assert frame["conf_int_lower"].isna().all()

        ax = plot_event_study(res, show=False)
        assert len(_plotted_points(ax)) == len(frame)
        assert _plotted_intervals(ax) == {}
        plt.close()

    @pytest.mark.parametrize("use_cband", [True, False])
    def test_cohort_share_stays_undefined_under_explicit_reference(self, use_cband):
        """Explicit reference_period= discards the stored intervals and recomputes
        from the SE; that recomputation must not revive the inference the
        cohort-share aggregation disabled (finite SE, NaN t/p/CI)."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        res = self._fit()
        with pytest.warns(UserWarning, match="conditional-on-shares"):
            res.aggregate("event_study", weights="cohort_share")
        frame = res.to_dataframe(level="event_study")

        ax = plot_event_study(res, reference_period=0, use_cband=use_cband, show=False)
        points = _plotted_points(ax)
        # the normalized point estimates still plot ...
        np.testing.assert_allclose(
            [p[1] for p in points], (frame["att"] - frame["att"].iloc[0]).to_numpy(), atol=1e-12
        )
        # ... with no interval on any row
        assert _plotted_intervals(ax) == {}
        plt.close()

    @pytest.mark.parametrize("use_cband", [True, False])
    def test_cohort_share_explicit_reference_plotly_draws_no_band(self, use_cband):
        pytest.importorskip("plotly")

        res = self._fit()
        with pytest.warns(UserWarning, match="conditional-on-shares"):
            res.aggregate("event_study", weights="cohort_share")
        for kwargs in ({}, {"reference_period": 0}):
            fig = plot_event_study(res, backend="plotly", use_cband=use_cband, show=False, **kwargs)
            assert not [t for t in fig.data if getattr(t, "fill", None) == "toself"]
        # positive control: the same call on defined (cell-weighted) inference bands
        res.aggregate("event_study")
        fig = plot_event_study(
            res, backend="plotly", reference_period=0, use_cband=use_cband, show=False
        )
        assert [t for t in fig.data if getattr(t, "fill", None) == "toself"]

    @pytest.mark.parametrize("use_cband", [True, False])
    def test_cell_weighted_explicit_reference_keeps_intervals(self, use_cband):
        """The undefined-inference mask is row-level: a cell-weighted surface
        (defined inference) still gets recomputed intervals on every
        non-reference row after explicit normalization."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from scipy import stats

        res = self._fit()
        res.aggregate("event_study")
        frame = res.to_dataframe(level="event_study")
        ref = frame["event_time"].tolist().index(0)

        ax = plot_event_study(res, reference_period=0, use_cband=use_cband, show=False)
        intervals = _plotted_intervals(ax)
        assert sorted(intervals) == [float(i) for i in range(len(frame)) if i != ref]
        z = stats.norm.ppf(0.975)
        shifted = frame["att"] - frame["att"].iloc[ref]
        for x, (lo, hi) in intervals.items():
            i = int(x)
            np.testing.assert_allclose(
                [lo, hi],
                [
                    shifted.iloc[i] - z * frame["se"].iloc[i],
                    shifted.iloc[i] + z * frame["se"].iloc[i],
                ],
                rtol=1e-10,
            )
        plt.close()


def _idiom_fit(name):
    """A small analytical fit for each event-study producer the tutorials plot."""
    import warnings

    import diff_diff as dd
    from diff_diff import prep_dgp

    stag = prep_dgp.generate_staggered_data(n_units=120, n_periods=8, seed=1)
    kw = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "CallawaySantAnna":
            return dd.CallawaySantAnna().fit(stag, **kw)
        if name == "SunAbraham":
            return dd.SunAbraham().fit(stag, **kw)
        if name == "ImputationDiD":
            return dd.ImputationDiD().fit(stag, **kw)
        if name == "TwoStageDiD":
            return dd.TwoStageDiD().fit(stag, **kw)
        if name == "StackedDiD":
            return dd.StackedDiD(kappa_pre=2, kappa_post=2).fit(stag, **kw)
        if name == "EfficientDiD":
            return dd.EfficientDiD().fit(stag, **kw)
        if name == "WooldridgeDiD":
            return dd.WooldridgeDiD().fit(stag, **kw)
        if name == "LWDiD":
            return dd.LWDiD().fit(stag, treatment="treated", **kw)
        if name == "DMLDiD":
            cov = stag.copy()
            unit_x = np.random.default_rng(0).normal(size=cov["unit"].nunique())
            cov["x1"] = unit_x[cov["unit"].astype("category").cat.codes.to_numpy()]
            return dd.DMLDiD(outcome_learner="linear", n_folds=2, seed=1).fit(
                cov, covariates=["x1"], **kw
            )
        if name == "ChaisemartinDHaultfoeuille":
            rev = prep_dgp.generate_reversible_did_data(
                n_groups=40, n_periods=6, pattern="single_switch", seed=1
            )
            return dd.ChaisemartinDHaultfoeuille(twfe_diagnostic=False).fit(
                rev, outcome="outcome", unit="group", time="period", treatment="treatment", L_max=2
            )
        if name == "ContinuousDiD":
            cont = prep_dgp.generate_continuous_did_data(
                n_units=150, n_periods=6, cohort_periods=[3, 5], never_treated_frac=0.3, seed=1
            )
            return dd.ContinuousDiD(seed=1).fit(cont, dose="dose", **kw)
        if name == "HeterogeneousAdoptionDiD":
            had = prep_dgp.generate_continuous_did_data(
                n_units=60,
                n_periods=8,
                cohort_periods=[5],
                never_treated_frac=0.0,
                dose_distribution="uniform",
                dose_params={"low": 5.0, "high": 50.0},
                seed=1,
            )
            had.loc[had["period"] < had["first_treat"], "dose"] = 0.0
            return dd.HeterogeneousAdoptionDiD(design="auto").fit(had, dose="dose", **kw)
        if name == "TwoWayFixedEffects":
            return dd.TwoWayFixedEffects().fit(
                generate_multi_period_data(),
                outcome="outcome",
                treatment="treated",
                time="period",
                event_study=True,
                spec="pooled",
                post_periods=[2, 3],
                reference_period=1,
            )
    raise AssertionError(name)


_IDIOM_PRODUCERS = [
    "CallawaySantAnna",
    "SunAbraham",
    "ImputationDiD",
    "TwoStageDiD",
    "StackedDiD",
    "EfficientDiD",
    "WooldridgeDiD",
    "LWDiD",
    "DMLDiD",
    "ChaisemartinDHaultfoeuille",
    "ContinuousDiD",
    "HeterogeneousAdoptionDiD",
    "TwoWayFixedEffects",
]


class TestEventStudyIdiomContract:
    """The one event-study idiom the tutorials teach (CONTRIBUTING.md, "Tutorial
    conventions") works on every producer they plot:

        es = results.aggregate("event_study") if hasattr(results, "aggregate") else results
        plot_event_study(es)
        table = es.to_dataframe() if isinstance(es, EventStudyResults) \\
            else results.to_dataframe(level="event_study")

    A producer that stops satisfying it must be fixed in the plotter (convert to
    the container at the top of plot_event_study), not excepted in the docs.
    """

    @pytest.mark.parametrize("name", _IDIOM_PRODUCERS)
    def test_surface_plot_and_table(self, name):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        from diff_diff.results_base import EVENT_STUDY_SCHEMA, EventStudyResults

        results = _idiom_fit(name)
        es = results.aggregate("event_study") if hasattr(results, "aggregate") else results
        table = (
            es.to_dataframe()
            if isinstance(es, EventStudyResults)
            else results.to_dataframe(level="event_study")
        )
        assert list(table.columns[: len(EVENT_STUDY_SCHEMA)]) == list(EVENT_STUDY_SCHEMA)
        assert len(table) > 0

        ax = plot_event_study(es, show=False)
        points = _plotted_points(ax)
        # Every estimated row is drawn. A reference row is drawn (hollow) when the
        # surface materializes it; a producer whose baseline is omitted from its
        # native dict (SunAbraham's raw-result route) plots the estimated rows only.
        if len(points) != len(table):
            table = table[~table["is_reference"].astype(bool)]
        assert len(points) == len(table)
        estimates = table["att"].to_numpy(dtype=float)
        plotted = np.array([p[1] for p in points])
        finite = np.isfinite(estimates)
        assert finite.any()
        np.testing.assert_allclose(plotted[finite], estimates[finite], rtol=1e-12, atol=1e-12)
        plt.close()

    def test_container_route_classes(self):
        """Pin which producers take which route, so the convention text stays true:
        only SunAbraham and the TWFE event-study result lack aggregate(), and only
        Wooldridge's aggregate() returns the result itself."""
        from diff_diff.results_base import EventStudyResults

        no_aggregate, self_returning = [], []
        for name in _IDIOM_PRODUCERS:
            results = _idiom_fit(name)
            if not hasattr(results, "aggregate"):
                no_aggregate.append(name)
                continue
            es = results.aggregate("event_study")
            if not isinstance(es, EventStudyResults):
                assert es is results
                self_returning.append(name)
        assert no_aggregate == ["SunAbraham", "TwoWayFixedEffects"]
        assert self_returning == ["WooldridgeDiD"]
