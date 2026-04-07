"""Event study visualization functions."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from diff_diff.honest_did import HonestDiDResults
    from diff_diff.imputation import ImputationDiDResults
    from diff_diff.results import MultiPeriodDiDResults
    from diff_diff.stacked_did import StackedDiDResults
    from diff_diff.staggered import CallawaySantAnnaResults
    from diff_diff.sun_abraham import SunAbrahamResults
    from diff_diff.two_stage import TwoStageDiDResults

# Type alias for results that can be plotted
PlottableResults = Union[
    "MultiPeriodDiDResults",
    "CallawaySantAnnaResults",
    "SunAbrahamResults",
    "ImputationDiDResults",
    "TwoStageDiDResults",
    "StackedDiDResults",
    pd.DataFrame,
]


def plot_event_study(
    results: Optional[PlottableResults] = None,
    *,
    effects: Optional[Dict[Any, float]] = None,
    se: Optional[Dict[Any, float]] = None,
    periods: Optional[List[Any]] = None,
    reference_period: Optional[Any] = None,
    pre_periods: Optional[List[Any]] = None,
    post_periods: Optional[List[Any]] = None,
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Event Study",
    xlabel: str = "Period Relative to Treatment",
    ylabel: str = "Treatment Effect",
    color: str = "#2563eb",
    marker: str = "o",
    markersize: int = 8,
    linewidth: float = 1.5,
    capsize: int = 4,
    show_zero_line: bool = True,
    show_reference_line: bool = True,
    shade_pre: bool = True,
    shade_color: str = "#f0f0f0",
    ax: Optional[Any] = None,
    show: bool = True,
    use_cband: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Create an event study plot showing treatment effects over time.

    This function creates a coefficient plot with point estimates and
    confidence intervals for each time period, commonly used to visualize
    dynamic treatment effects and assess pre-trends.

    Parameters
    ----------
    results : MultiPeriodDiDResults, CallawaySantAnnaResults, or DataFrame, optional
        Results object from MultiPeriodDiD, CallawaySantAnna, or a DataFrame
        with columns 'period', 'effect', 'se' (and optionally 'conf_int_lower',
        'conf_int_upper'). If None, must provide effects and se directly.
    effects : dict, optional
        Dictionary mapping periods to effect estimates. Used if results is None.
    se : dict, optional
        Dictionary mapping periods to standard errors. Used if results is None.
    periods : list, optional
        List of periods to plot. If None, uses all periods from results.
    reference_period : any, optional
        The reference period to highlight. When explicitly provided, effects
        are normalized (ref effect subtracted) and ref SE is set to NaN.
        When None and auto-inferred from results, only hollow marker styling
        is applied (no normalization). If None, tries to infer from results.
    pre_periods : list, optional
        List of pre-treatment periods. Used for shading.
    post_periods : list, optional
        List of post-treatment periods. Used for shading.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    title : str, default="Event Study"
        Plot title.
    xlabel : str, default="Period Relative to Treatment"
        X-axis label.
    ylabel : str, default="Treatment Effect"
        Y-axis label.
    color : str, default="#2563eb"
        Color for points and error bars.
    marker : str, default="o"
        Marker style for point estimates.
    markersize : int, default=8
        Size of markers.
    linewidth : float, default=1.5
        Width of error bar lines.
    capsize : int, default=4
        Size of error bar caps.
    show_zero_line : bool, default=True
        Whether to show a horizontal line at y=0.
    show_reference_line : bool, default=True
        Whether to show a vertical line at the reference period.
    shade_pre : bool, default=True
        Whether to shade the pre-treatment region.
    shade_color : str, default="#f0f0f0"
        Color for pre-treatment shading.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool, default=True
        Whether to call plt.show() at the end.
    use_cband : bool, default=True
        Whether to use simultaneous confidence band CIs when available
        from CallawaySantAnna results. When False, pointwise CIs from
        ``alpha`` are used regardless.
    backend : str, default="matplotlib"
        Plotting backend: ``"matplotlib"`` for static plots or
        ``"plotly"`` for interactive plots.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The axes object (matplotlib) or figure (plotly) containing the plot.

    Examples
    --------
    Using with MultiPeriodDiD results:

    >>> from diff_diff import MultiPeriodDiD, plot_event_study
    >>> did = MultiPeriodDiD()
    >>> results = did.fit(data, outcome='y', treatment='treated',
    ...                   time='period', post_periods=[3, 4, 5])
    >>> plot_event_study(results)

    Using with a DataFrame:

    >>> df = pd.DataFrame({
    ...     'period': [-2, -1, 0, 1, 2],
    ...     'effect': [0.1, 0.05, 0.0, 0.5, 0.6],
    ...     'se': [0.1, 0.1, 0.0, 0.15, 0.15]
    ... })
    >>> plot_event_study(df, reference_period=0)

    Using with manual effects:

    >>> effects = {-2: 0.1, -1: 0.05, 0: 0.0, 1: 0.5, 2: 0.6}
    >>> se = {-2: 0.1, -1: 0.1, 0: 0.0, 1: 0.15, 2: 0.15}
    >>> plot_event_study(effects=effects, se=se, reference_period=0)

    Notes
    -----
    Event study plots are a standard visualization in difference-in-differences
    analysis. They show:

    1. **Pre-treatment periods**: Effects should be close to zero if parallel
       trends holds. Large pre-treatment effects suggest the assumption may
       be violated.

    2. **Reference period**: Usually the last pre-treatment period (t=-1).
       When explicitly specified via ``reference_period``, effects are normalized
       to zero at this period. When auto-inferred, shown with hollow marker only.

    3. **Post-treatment periods**: The treatment effects of interest. These
       show how the outcome evolved after treatment.

    The confidence intervals help assess statistical significance. Effects
    whose CIs don't include zero are typically considered significant.
    """
    from scipy import stats as scipy_stats

    # Track if reference_period was explicitly provided by user
    reference_period_explicit = reference_period is not None

    # Extract data from results if provided
    ci_lower_override = None
    ci_upper_override = None
    if results is not None:
        (
            effects,
            se,
            periods,
            pre_periods,
            post_periods,
            reference_period,
            reference_inferred,
            ci_lower_override,
            ci_upper_override,
        ) = _extract_plot_data(results, periods, pre_periods, post_periods, reference_period)
        # If reference was inferred from results, it was NOT explicitly provided
        if reference_inferred:
            reference_period_explicit = False
        # Suppress simultaneous confidence band overrides when user opts out
        if not use_cband:
            ci_lower_override = None
            ci_upper_override = None
    elif effects is None or se is None:
        raise ValueError("Must provide either 'results' or both 'effects' and 'se'")

    # Ensure effects and se are dicts
    if not isinstance(effects, dict):
        raise TypeError("effects must be a dictionary mapping periods to values")
    if not isinstance(se, dict):
        raise TypeError("se must be a dictionary mapping periods to values")

    # Get periods to plot
    if periods is None:
        periods = sorted(effects.keys())

    # Compute confidence intervals
    critical_value = scipy_stats.norm.ppf(1 - alpha / 2)

    # Normalize effects to reference period ONLY if explicitly specified by user
    # Auto-inferred reference periods (from CallawaySantAnna) just get hollow marker styling,
    # NO normalization. This prevents unintended normalization when the reference period
    # isn't a true identifying constraint (e.g., CallawaySantAnna with base_period="varying").
    if reference_period is not None and reference_period in effects and reference_period_explicit:
        ref_effect = effects[reference_period]
        if np.isfinite(ref_effect):
            effects = {p: e - ref_effect for p, e in effects.items()}
            # Set reference SE to NaN (it's now a constraint, not an estimate)
            # This follows fixest convention where the omitted category has no SE/CI
            se = {p: (np.nan if p == reference_period else s) for p, s in se.items()}

    plot_data = []
    for period in periods:
        effect = effects.get(period, np.nan)
        std_err = se.get(period, np.nan)

        # Skip entries with NaN effect, but allow NaN SE (will plot without error bars)
        if np.isnan(effect):
            continue

        # Use cband CI overrides when available, otherwise compute pointwise
        if ci_lower_override is not None and period in ci_lower_override:
            ci_lower = ci_lower_override[period]
            assert ci_upper_override is not None
            ci_upper = ci_upper_override[period]
        elif np.isfinite(std_err):
            ci_lower = effect - critical_value * std_err
            ci_upper = effect + critical_value * std_err
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        plot_data.append(
            {
                "period": period,
                "effect": effect,
                "se": std_err,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "is_reference": period == reference_period,
            }
        )

    if not plot_data:
        raise ValueError("No valid data to plot")

    df = pd.DataFrame(plot_data)

    if backend == "plotly":
        return _render_event_study_plotly(
            df,
            reference_period=reference_period,
            pre_periods=pre_periods,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            marker=marker,
            markersize=markersize,
            shade_pre=shade_pre,
            shade_color=shade_color,
            show_zero_line=show_zero_line,
            show_reference_line=show_reference_line,
            show=show,
        )

    return _render_event_study_mpl(
        df,
        reference_period=reference_period,
        pre_periods=pre_periods,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        marker=marker,
        markersize=markersize,
        linewidth=linewidth,
        capsize=capsize,
        shade_pre=shade_pre,
        shade_color=shade_color,
        show_zero_line=show_zero_line,
        show_reference_line=show_reference_line,
        ax=ax,
        show=show,
    )


def _render_event_study_mpl(
    df,
    *,
    reference_period,
    pre_periods,
    figsize,
    title,
    xlabel,
    ylabel,
    color,
    marker,
    markersize,
    linewidth,
    capsize,
    shade_pre,
    shade_color,
    show_zero_line,
    show_reference_line,
    ax,
    show,
):
    """Render event study plot with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Convert periods to numeric for plotting
    period_to_x = {p: i for i, p in enumerate(df["period"])}
    x_vals = [period_to_x[p] for p in df["period"]]

    # Shade pre-treatment region
    if shade_pre and pre_periods is not None:
        pre_x = [period_to_x[p] for p in pre_periods if p in period_to_x]
        if pre_x:
            ax.axvspan(min(pre_x) - 0.5, max(pre_x) + 0.5, color=shade_color, alpha=0.5, zorder=0)

    # Draw horizontal zero line
    if show_zero_line:
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, zorder=1)

    # Draw vertical reference line
    if show_reference_line and reference_period is not None:
        if reference_period in period_to_x:
            ref_x = period_to_x[reference_period]
            ax.axvline(x=ref_x, color="gray", linestyle=":", linewidth=1, zorder=1)

    # Plot error bars (only for entries with finite CI)
    has_ci = df["ci_lower"].notna() & df["ci_upper"].notna()
    if has_ci.any():
        df_with_ci = df[has_ci]
        x_with_ci = [period_to_x[p] for p in df_with_ci["period"]]
        yerr = [
            df_with_ci["effect"] - df_with_ci["ci_lower"],
            df_with_ci["ci_upper"] - df_with_ci["effect"],
        ]
        ax.errorbar(
            x_with_ci,
            df_with_ci["effect"],
            yerr=yerr,
            fmt="none",
            color=color,
            capsize=capsize,
            linewidth=linewidth,
            capthick=linewidth,
            zorder=2,
        )

    # Plot point estimates
    for i, row in df.iterrows():
        x = period_to_x[row["period"]]
        if row["is_reference"]:
            # Hollow marker for reference period
            ax.plot(
                x,
                row["effect"],
                marker=marker,
                markersize=markersize,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2,
                zorder=3,
            )
        else:
            ax.plot(x, row["effect"], marker=marker, markersize=markersize, color=color, zorder=3)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set x-axis ticks
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(p) for p in df["period"]])

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    # Tight layout
    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_event_study_plotly(
    df,
    *,
    reference_period,
    pre_periods,
    title,
    xlabel,
    ylabel,
    color,
    marker,
    markersize,
    shade_pre,
    shade_color,
    show_zero_line,
    show_reference_line,
    show,
):
    """Render event study plot with plotly."""
    from diff_diff.visualization._common import (
        _color_to_rgba,
        _mpl_marker_to_plotly_symbol,
        _plotly_default_layout,
        _require_plotly,
    )

    go = _require_plotly()

    fig = go.Figure()

    periods = df["period"].tolist()
    effects = df["effect"].tolist()
    ci_lower = df["ci_lower"].tolist()
    ci_upper = df["ci_upper"].tolist()
    is_ref = df["is_reference"].tolist()

    # Map periods to ordinal x positions (matching matplotlib renderer).
    # This ensures string, timestamp, and other non-numeric periods work correctly.
    period_to_x = {p: i for i, p in enumerate(periods)}
    x_vals = list(range(len(periods)))
    tick_labels = [str(p) for p in periods]

    # Shade pre-treatment region
    if shade_pre and pre_periods is not None:
        pre_x = [period_to_x[p] for p in pre_periods if p in period_to_x]
        if pre_x:
            fig.add_vrect(
                x0=min(pre_x) - 0.5,
                x1=max(pre_x) + 0.5,
                fillcolor=_color_to_rgba(shade_color, 0.5),
                line_width=0,
                layer="below",
            )

    # Zero line
    if show_zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    # Reference line
    if show_reference_line and reference_period is not None and reference_period in period_to_x:
        fig.add_vline(
            x=period_to_x[reference_period], line_dash="dot", line_color="gray", line_width=1
        )

    # CI band (filled area)
    has_ci = [not (np.isnan(lo) or np.isnan(hi)) for lo, hi in zip(ci_lower, ci_upper)]
    ci_x = [period_to_x[p] for p, h in zip(periods, has_ci) if h]
    ci_lo = [lo for lo, h in zip(ci_lower, has_ci) if h]
    ci_hi = [hi for hi, h in zip(ci_upper, has_ci) if h]

    if ci_x:
        fig.add_trace(
            go.Scatter(
                x=ci_x + ci_x[::-1],
                y=ci_hi + ci_lo[::-1],
                fill="toself",
                fillcolor=_color_to_rgba(color, 0.15),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Point estimates — separate reference vs non-reference.
    # Attach original period labels via customdata + hovertemplate so hover
    # shows real periods instead of ordinal positions.
    non_ref_x = [period_to_x[p] for p, r in zip(periods, is_ref) if not r]
    non_ref_e = [e for e, r in zip(effects, is_ref) if not r]
    non_ref_labels = [str(p) for p, r in zip(periods, is_ref) if not r]
    ref_x = [period_to_x[p] for p, r in zip(periods, is_ref) if r]
    ref_e = [e for e, r in zip(effects, is_ref) if r]
    ref_labels = [str(p) for p, r in zip(periods, is_ref) if r]

    hover_tpl = "Period: %{customdata}<br>Effect: %{y:.4f}<extra></extra>"

    symbol = _mpl_marker_to_plotly_symbol(marker)

    if non_ref_x:
        fig.add_trace(
            go.Scatter(
                x=non_ref_x,
                y=non_ref_e,
                mode="markers",
                marker=dict(color=color, size=markersize, symbol=symbol),
                name="Effect",
                customdata=non_ref_labels,
                hovertemplate=hover_tpl,
            )
        )

    if ref_x:
        fig.add_trace(
            go.Scatter(
                x=ref_x,
                y=ref_e,
                mode="markers",
                marker=dict(
                    color="white",
                    size=markersize,
                    symbol=symbol,
                    line=dict(color=color, width=2),
                ),
                name="Reference",
                customdata=ref_labels,
                hovertemplate=hover_tpl,
            )
        )

    # Set tick labels to show original period values
    fig.update_xaxes(tickvals=x_vals, ticktext=tick_labels)

    _plotly_default_layout(fig, title=title, xlabel=xlabel, ylabel=ylabel)

    if show:
        fig.show()

    return fig


def _extract_plot_data(
    results: PlottableResults,
    periods: Optional[List[Any]],
    pre_periods: Optional[List[Any]],
    post_periods: Optional[List[Any]],
    reference_period: Optional[Any],
) -> Tuple[Dict, Dict, List, List, List, Any, bool, Optional[Dict], Optional[Dict]]:
    """
    Extract plotting data from various result types.

    Returns
    -------
    effects : dict
        Mapping of period to effect estimate.
    se : dict
        Mapping of period to standard error.
    periods : list
        Ordered list of periods to plot.
    pre_periods : list
        Pre-treatment periods.
    post_periods : list
        Post-treatment periods.
    reference_period : any
        The reference period (explicit or inferred).
    reference_inferred : bool
        True if reference_period was auto-detected from results
        rather than explicitly provided by the user.
    ci_lower_override : dict or None
        Simultaneous confidence band lower bounds, if available.
    ci_upper_override : dict or None
        Simultaneous confidence band upper bounds, if available.
    """
    # Handle DataFrame input
    if isinstance(results, pd.DataFrame):
        if "period" not in results.columns:
            raise ValueError("DataFrame must have 'period' column")
        if "effect" not in results.columns:
            raise ValueError("DataFrame must have 'effect' column")
        if "se" not in results.columns:
            raise ValueError("DataFrame must have 'se' column")

        effects = dict(zip(results["period"], results["effect"]))
        se = dict(zip(results["period"], results["se"]))

        if periods is None:
            periods = list(results["period"])

        # Extract simultaneous confidence bands if present and finite
        ci_lower_override = None
        ci_upper_override = None
        if "cband_lower" in results.columns and "cband_upper" in results.columns:
            finite_mask = results["cband_lower"].notna() & results["cband_upper"].notna()
            if finite_mask.any():
                finite_rows = results[finite_mask]
                ci_lower_override = dict(zip(finite_rows["period"], finite_rows["cband_lower"]))
                ci_upper_override = dict(zip(finite_rows["period"], finite_rows["cband_upper"]))

        # DataFrame input: reference_period was already set by caller, never inferred here
        return (
            effects,
            se,
            periods,
            pre_periods,
            post_periods,
            reference_period,
            False,
            ci_lower_override,
            ci_upper_override,
        )

    # Handle MultiPeriodDiDResults
    if hasattr(results, "period_effects"):
        effects = {}
        se = {}

        for period, pe in results.period_effects.items():
            effects[period] = pe.effect
            se[period] = pe.se

        if pre_periods is None and hasattr(results, "pre_periods"):
            pre_periods = results.pre_periods

        if post_periods is None and hasattr(results, "post_periods"):
            post_periods = results.post_periods

        if periods is None:
            periods = sorted(results.period_effects.keys())

        # Auto-detect reference period from results if not explicitly provided
        ref_inferred = False
        if (
            reference_period is None
            and hasattr(results, "reference_period")
            and results.reference_period is not None
        ):
            reference_period = results.reference_period
            ref_inferred = True

        return (
            effects,
            se,
            periods,
            pre_periods,
            post_periods,
            reference_period,
            ref_inferred,
            None,
            None,
        )

    # Handle CallawaySantAnnaResults (event study aggregation)
    if hasattr(results, "event_study_effects") and results.event_study_effects is not None:
        effects = {}
        se = {}
        ci_lower_override = {}
        ci_upper_override = {}
        has_cband = False

        for rel_period, effect_data in results.event_study_effects.items():
            effects[rel_period] = effect_data["effect"]
            se[rel_period] = effect_data["se"]
            # Use simultaneous CIs when available
            if "cband_conf_int" in effect_data:
                cband_ci = effect_data["cband_conf_int"]
                ci_lower_override[rel_period] = cband_ci[0]
                ci_upper_override[rel_period] = cband_ci[1]
                has_cband = True

        if periods is None:
            periods = sorted(effects.keys())

        # Track if reference_period was explicitly provided vs auto-inferred
        reference_inferred = False

        # Reference period is typically -1 for event study
        if reference_period is None:
            reference_inferred = True  # We're about to infer it
            # Detect reference period from n_groups=0 marker (normalization constraint)
            # This handles anticipation > 0 where reference is at e = -1 - anticipation
            for period, effect_data in results.event_study_effects.items():
                if effect_data.get("n_groups", 1) == 0 or effect_data.get("n_obs", 1) == 0:
                    reference_period = period
                    break
            # Fallback to -1 if no marker found (backward compatibility)
            if reference_period is None:
                reference_period = -1

        if pre_periods is None:
            pre_periods = [p for p in periods if p < 0]

        if post_periods is None:
            post_periods = [p for p in periods if p >= 0]

        return (
            effects,
            se,
            periods,
            pre_periods,
            post_periods,
            reference_period,
            reference_inferred,
            ci_lower_override if has_cband else None,
            ci_upper_override if has_cband else None,
        )

    raise TypeError(
        f"Cannot extract plot data from {type(results).__name__}. "
        "Expected MultiPeriodDiDResults, CallawaySantAnnaResults, "
        "SunAbrahamResults, ImputationDiDResults, or DataFrame."
    )


def plot_honest_event_study(
    honest_results: "HonestDiDResults",
    *,
    periods: Optional[List[Any]] = None,
    reference_period: Optional[Any] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Event Study with Honest Confidence Intervals",
    xlabel: str = "Period Relative to Treatment",
    ylabel: str = "Treatment Effect",
    original_color: str = "#6b7280",
    honest_color: str = "#2563eb",
    marker: str = "o",
    markersize: int = 8,
    capsize: int = 4,
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Create event study plot with Honest DiD confidence intervals.

    Shows both the original confidence intervals (assuming parallel trends)
    and the robust confidence intervals that allow for bounded violations.

    Parameters
    ----------
    honest_results : HonestDiDResults
        Results from HonestDiD.fit() that include event_study_bounds.
    periods : list, optional
        Periods to plot. If None, uses all available periods.
    reference_period : any, optional
        Reference period to show as hollow marker.
    figsize : tuple, default=(10, 6)
        Figure size.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    original_color : str
        Color for original (standard) confidence intervals.
    honest_color : str
        Color for honest (robust) confidence intervals.
    marker : str
        Marker style.
    markersize : int
        Marker size.
    capsize : int
        Error bar cap size.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    show : bool, default=True
        Whether to call plt.show().
    backend : str, default="matplotlib"
        Plotting backend: ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The axes object (matplotlib) or figure (plotly).

    Notes
    -----
    This function requires the HonestDiDResults to have been computed
    with event_study_bounds. If only a scalar bound was computed,
    use plot_sensitivity() instead.
    """
    from scipy import stats as scipy_stats

    # Get original results for standard CIs
    original_results = honest_results.original_results
    if original_results is None:
        raise ValueError("HonestDiDResults must have original_results to plot event study")

    # Extract data from original results
    if hasattr(original_results, "period_effects"):
        # MultiPeriodDiDResults
        effects_dict = {p: pe.effect for p, pe in original_results.period_effects.items()}
        se_dict = {p: pe.se for p, pe in original_results.period_effects.items()}
        if periods is None:
            periods = list(original_results.period_effects.keys())
    elif hasattr(original_results, "event_study_effects"):
        # CallawaySantAnnaResults
        effects_dict = {
            t: data["effect"] for t, data in original_results.event_study_effects.items()
        }
        se_dict = {t: data["se"] for t, data in original_results.event_study_effects.items()}
        if periods is None:
            periods = sorted(original_results.event_study_effects.keys())
    else:
        raise TypeError("Cannot extract event study data from original_results")

    # Compute CIs
    alpha_val = honest_results.alpha
    z = scipy_stats.norm.ppf(1 - alpha_val / 2)

    effects = [effects_dict[p] for p in periods]
    original_ci_lower = [effects_dict[p] - z * se_dict[p] for p in periods]
    original_ci_upper = [effects_dict[p] + z * se_dict[p] for p in periods]

    # Get honest bounds if available for each period
    if honest_results.event_study_bounds:
        honest_ci_lower = [honest_results.event_study_bounds[p]["ci_lb"] for p in periods]
        honest_ci_upper = [honest_results.event_study_bounds[p]["ci_ub"] for p in periods]
    else:
        # Use scalar bounds applied to all periods
        honest_ci_lower = [honest_results.ci_lb] * len(periods)
        honest_ci_upper = [honest_results.ci_ub] * len(periods)

    if backend == "plotly":
        return _render_honest_event_study_plotly(
            periods=periods,
            effects=effects,
            original_ci_lower=original_ci_lower,
            original_ci_upper=original_ci_upper,
            honest_ci_lower=honest_ci_lower,
            honest_ci_upper=honest_ci_upper,
            honest_M=honest_results.M,
            reference_period=reference_period,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            original_color=original_color,
            honest_color=honest_color,
            marker=marker,
            markersize=markersize,
            show=show,
        )

    return _render_honest_event_study_mpl(
        periods=periods,
        effects=effects,
        original_ci_lower=original_ci_lower,
        original_ci_upper=original_ci_upper,
        honest_ci_lower=honest_ci_lower,
        honest_ci_upper=honest_ci_upper,
        honest_M=honest_results.M,
        reference_period=reference_period,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        original_color=original_color,
        honest_color=honest_color,
        marker=marker,
        markersize=markersize,
        capsize=capsize,
        ax=ax,
        show=show,
    )


def _render_honest_event_study_mpl(
    *,
    periods,
    effects,
    original_ci_lower,
    original_ci_upper,
    honest_ci_lower,
    honest_ci_upper,
    honest_M,
    reference_period,
    figsize,
    title,
    xlabel,
    ylabel,
    original_color,
    honest_color,
    marker,
    markersize,
    capsize,
    ax,
    show,
):
    """Render honest event study plot with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x_vals = list(range(len(periods)))

    # Zero line
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Plot original CIs (thinner, background)
    yerr_orig = [
        [e - lower for e, lower in zip(effects, original_ci_lower)],
        [u - e for e, u in zip(effects, original_ci_upper)],
    ]
    ax.errorbar(
        x_vals,
        effects,
        yerr=yerr_orig,
        fmt="none",
        color=original_color,
        capsize=capsize - 1,
        linewidth=1,
        alpha=0.6,
        label="Standard CI",
    )

    # Plot honest CIs (thicker, foreground)
    yerr_honest = [
        [e - lower for e, lower in zip(effects, honest_ci_lower)],
        [u - e for e, u in zip(effects, honest_ci_upper)],
    ]
    ax.errorbar(
        x_vals,
        effects,
        yerr=yerr_honest,
        fmt="none",
        color=honest_color,
        capsize=capsize,
        linewidth=2,
        label=f"Honest CI (M={honest_M:.2f})",
    )

    # Plot point estimates
    for i, (x, effect, period) in enumerate(zip(x_vals, effects, periods)):
        is_ref = period == reference_period
        if is_ref:
            ax.plot(
                x,
                effect,
                marker=marker,
                markersize=markersize,
                markerfacecolor="white",
                markeredgecolor=honest_color,
                markeredgewidth=2,
                zorder=3,
            )
        else:
            ax.plot(x, effect, marker=marker, markersize=markersize, color=honest_color, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(p) for p in periods])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_honest_event_study_plotly(
    *,
    periods,
    effects,
    original_ci_lower,
    original_ci_upper,
    honest_ci_lower,
    honest_ci_upper,
    honest_M,
    reference_period,
    title,
    xlabel,
    ylabel,
    original_color,
    honest_color,
    marker,
    markersize,
    show,
):
    """Render honest event study plot with plotly."""
    from diff_diff.visualization._common import (
        _color_to_rgba,
        _mpl_marker_to_plotly_symbol,
        _plotly_default_layout,
        _require_plotly,
    )

    go = _require_plotly()

    fig = go.Figure()

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)

    # Original CI band
    fig.add_trace(
        go.Scatter(
            x=list(periods) + list(periods)[::-1],
            y=list(original_ci_upper) + list(original_ci_lower)[::-1],
            fill="toself",
            fillcolor=_color_to_rgba(original_color, 0.15),
            line=dict(color="rgba(0,0,0,0)"),
            name="Standard CI",
            hoverinfo="skip",
        )
    )

    # Honest CI band
    fig.add_trace(
        go.Scatter(
            x=list(periods) + list(periods)[::-1],
            y=list(honest_ci_upper) + list(honest_ci_lower)[::-1],
            fill="toself",
            fillcolor=_color_to_rgba(honest_color, 0.15),
            line=dict(color="rgba(0,0,0,0)"),
            name=f"Honest CI (M={honest_M:.2f})",
            hoverinfo="skip",
        )
    )

    # Point estimates
    is_ref = [p == reference_period for p in periods]
    non_ref_p = [p for p, r in zip(periods, is_ref) if not r]
    non_ref_e = [e for e, r in zip(effects, is_ref) if not r]
    ref_p = [p for p, r in zip(periods, is_ref) if r]
    ref_e = [e for e, r in zip(effects, is_ref) if r]

    symbol = _mpl_marker_to_plotly_symbol(marker)

    if non_ref_p:
        fig.add_trace(
            go.Scatter(
                x=non_ref_p,
                y=non_ref_e,
                mode="markers",
                marker=dict(color=honest_color, size=markersize, symbol=symbol),
                name="Effect",
            )
        )

    if ref_p:
        fig.add_trace(
            go.Scatter(
                x=ref_p,
                y=ref_e,
                mode="markers",
                marker=dict(
                    color="white",
                    size=markersize,
                    symbol=symbol,
                    line=dict(color=honest_color, width=2),
                ),
                name="Reference",
            )
        )

    _plotly_default_layout(fig, title=title, xlabel=xlabel, ylabel=ylabel)

    if show:
        fig.show()

    return fig
