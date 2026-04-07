"""Diagnostic visualization functions (sensitivity, Bacon decomposition)."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from diff_diff.bacon import BaconDecompositionResults
    from diff_diff.honest_did import SensitivityResults


def plot_sensitivity(
    sensitivity_results: "SensitivityResults",
    *,
    show_bounds: bool = True,
    show_ci: bool = True,
    breakdown_line: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Honest DiD Sensitivity Analysis",
    xlabel: str = "M (restriction parameter)",
    ylabel: str = "Treatment Effect",
    bounds_color: str = "#2563eb",
    bounds_alpha: float = 0.3,
    ci_color: str = "#2563eb",
    ci_linewidth: float = 1.5,
    breakdown_color: str = "#dc2626",
    original_color: str = "#1f2937",
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot sensitivity analysis results from Honest DiD.

    Shows how treatment effect bounds and confidence intervals
    change as the restriction parameter M varies.

    Parameters
    ----------
    sensitivity_results : SensitivityResults
        Results from HonestDiD.sensitivity_analysis().
    show_bounds : bool, default=True
        Whether to show the identified set bounds as shaded region.
    show_ci : bool, default=True
        Whether to show robust confidence interval lines.
    breakdown_line : bool, default=True
        Whether to show vertical line at breakdown value.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    bounds_color : str
        Color for identified set shading.
    bounds_alpha : float
        Transparency for identified set shading.
    ci_color : str
        Color for confidence interval lines.
    ci_linewidth : float
        Line width for CI lines.
    breakdown_color : str
        Color for breakdown value line.
    original_color : str
        Color for original estimate line.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool, default=True
        Whether to call plt.show().
    backend : str, default="matplotlib"
        Plotting backend: ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The axes object (matplotlib) or figure (plotly).

    Examples
    --------
    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.honest_did import HonestDiD
    >>> from diff_diff.visualization import plot_sensitivity
    >>>
    >>> # Fit event study and run sensitivity analysis
    >>> results = MultiPeriodDiD().fit(data, ...)
    >>> honest = HonestDiD(method='relative_magnitude')
    >>> sensitivity = honest.sensitivity_analysis(results)
    >>>
    >>> # Create sensitivity plot
    >>> plot_sensitivity(sensitivity)
    """
    M = sensitivity_results.M_values
    bounds_arr = np.array(sensitivity_results.bounds)
    ci_arr = np.array(sensitivity_results.robust_cis)

    if backend == "plotly":
        return _render_sensitivity_plotly(
            M=M,
            bounds_arr=bounds_arr,
            ci_arr=ci_arr,
            original_estimate=sensitivity_results.original_estimate,
            breakdown_M=sensitivity_results.breakdown_M,
            show_bounds=show_bounds,
            show_ci=show_ci,
            breakdown_line=breakdown_line,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            bounds_color=bounds_color,
            bounds_alpha=bounds_alpha,
            ci_color=ci_color,
            ci_linewidth=ci_linewidth,
            breakdown_color=breakdown_color,
            original_color=original_color,
            show=show,
        )

    return _render_sensitivity_mpl(
        M=M,
        bounds_arr=bounds_arr,
        ci_arr=ci_arr,
        original_estimate=sensitivity_results.original_estimate,
        breakdown_M=sensitivity_results.breakdown_M,
        show_bounds=show_bounds,
        show_ci=show_ci,
        breakdown_line=breakdown_line,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        bounds_color=bounds_color,
        bounds_alpha=bounds_alpha,
        ci_color=ci_color,
        ci_linewidth=ci_linewidth,
        breakdown_color=breakdown_color,
        original_color=original_color,
        ax=ax,
        show=show,
    )


def _render_sensitivity_mpl(
    *,
    M,
    bounds_arr,
    ci_arr,
    original_estimate,
    breakdown_M,
    show_bounds,
    show_ci,
    breakdown_line,
    figsize,
    title,
    xlabel,
    ylabel,
    bounds_color,
    bounds_alpha,
    ci_color,
    ci_linewidth,
    breakdown_color,
    original_color,
    ax,
    show,
):
    """Render sensitivity plot with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot original estimate
    ax.axhline(
        y=original_estimate,
        color=original_color,
        linestyle="-",
        linewidth=1.5,
        label="Original estimate",
        alpha=0.7,
    )

    # Plot zero line
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Plot identified set bounds
    if show_bounds:
        ax.fill_between(
            M,
            bounds_arr[:, 0],
            bounds_arr[:, 1],
            alpha=bounds_alpha,
            color=bounds_color,
            label="Identified set",
        )

    # Plot confidence intervals
    if show_ci:
        ax.plot(M, ci_arr[:, 0], color=ci_color, linewidth=ci_linewidth, label="Robust CI")
        ax.plot(M, ci_arr[:, 1], color=ci_color, linewidth=ci_linewidth)

    # Plot breakdown line
    if breakdown_line and breakdown_M is not None:
        ax.axvline(
            x=breakdown_M,
            color=breakdown_color,
            linestyle=":",
            linewidth=2,
            label=f"Breakdown (M={breakdown_M:.2f})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_sensitivity_plotly(
    *,
    M,
    bounds_arr,
    ci_arr,
    original_estimate,
    breakdown_M,
    show_bounds,
    show_ci,
    breakdown_line,
    title,
    xlabel,
    ylabel,
    bounds_color,
    bounds_alpha,
    ci_color,
    ci_linewidth,
    breakdown_color,
    original_color,
    show,
):
    """Render sensitivity plot with plotly."""
    from diff_diff.visualization._common import (
        _color_to_rgba,
        _plotly_default_layout,
        _require_plotly,
    )

    go = _require_plotly()

    fig = go.Figure()

    M_list = list(M) if not isinstance(M, list) else M

    # Original estimate line
    fig.add_hline(
        y=original_estimate,
        line_color=original_color,
        line_width=1.5,
        opacity=0.7,
        annotation_text="Original estimate",
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)

    # Identified set bounds
    if show_bounds:
        fig.add_trace(
            go.Scatter(
                x=M_list + M_list[::-1],
                y=list(bounds_arr[:, 1]) + list(bounds_arr[:, 0])[::-1],
                fill="toself",
                fillcolor=_color_to_rgba(bounds_color, bounds_alpha),
                line=dict(color="rgba(0,0,0,0)"),
                name="Identified set",
            )
        )

    # Confidence intervals
    if show_ci:
        fig.add_trace(
            go.Scatter(
                x=M_list,
                y=list(ci_arr[:, 0]),
                mode="lines",
                line=dict(color=ci_color, width=ci_linewidth),
                name="Robust CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=M_list,
                y=list(ci_arr[:, 1]),
                mode="lines",
                line=dict(color=ci_color, width=ci_linewidth),
                showlegend=False,
            )
        )

    # Breakdown line
    if breakdown_line and breakdown_M is not None:
        fig.add_vline(
            x=breakdown_M,
            line_dash="dot",
            line_color=breakdown_color,
            line_width=2,
            annotation_text=f"Breakdown (M={breakdown_M:.2f})",
        )

    _plotly_default_layout(fig, title=title, xlabel=xlabel, ylabel=ylabel)

    if show:
        fig.show()

    return fig


def plot_bacon(
    results: "BaconDecompositionResults",
    *,
    plot_type: str = "scatter",
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    xlabel: str = "2x2 DiD Estimate",
    ylabel: str = "Weight",
    colors: Optional[Dict[str, str]] = None,
    marker: str = "o",
    markersize: int = 80,
    alpha: float = 0.7,
    show_weighted_avg: bool = True,
    show_twfe_line: bool = True,
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Visualize Goodman-Bacon decomposition results.

    Creates either a scatter plot showing the weight and estimate for each
    2x2 comparison, or a stacked bar chart showing total weight by comparison
    type.

    Parameters
    ----------
    results : BaconDecompositionResults
        Results from BaconDecomposition.fit() or bacon_decompose().
    plot_type : str, default="scatter"
        Type of plot to create:
        - "scatter": Scatter plot with estimates on x-axis, weights on y-axis
        - "bar": Stacked bar chart of weights by comparison type
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, uses a default based on plot_type.
    xlabel : str, default="2x2 DiD Estimate"
        X-axis label (scatter plot only).
    ylabel : str, default="Weight"
        Y-axis label.
    colors : dict, optional
        Dictionary mapping comparison types to colors. Keys are:
        "treated_vs_never", "earlier_vs_later", "later_vs_earlier".
        If None, uses default colors.
    marker : str, default="o"
        Marker style for scatter plot.
    markersize : int, default=80
        Marker size for scatter plot.
    alpha : float, default=0.7
        Transparency for markers/bars.
    show_weighted_avg : bool, default=True
        Whether to show weighted average lines for each comparison type
        (scatter plot only).
    show_twfe_line : bool, default=True
        Whether to show a vertical line at the TWFE estimate (scatter plot only).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool, default=True
        Whether to call plt.show() at the end.
    backend : str, default="matplotlib"
        Plotting backend: ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The axes object (matplotlib) or figure (plotly).

    Examples
    --------
    Scatter plot (default):

    >>> from diff_diff import bacon_decompose, plot_bacon
    >>> results = bacon_decompose(data, outcome='y', unit='id',
    ...                           time='t', first_treat='first_treat')
    >>> plot_bacon(results)

    Bar chart of weights by type:

    >>> plot_bacon(results, plot_type='bar')

    Notes
    -----
    The scatter plot is particularly useful for understanding:

    1. **Distribution of estimates**: Are 2x2 estimates clustered or spread?
       Wide spread suggests heterogeneous treatment effects.

    2. **Weight concentration**: Do a few comparisons dominate the TWFE?
       Points with high weights have more influence.

    3. **Forbidden comparison problem**: Red points (later_vs_earlier) show
       comparisons using already-treated units as controls. If these have
       different estimates than clean comparisons, TWFE may be biased.

    See Also
    --------
    bacon_decompose : Perform the decomposition
    BaconDecomposition : Class-based interface
    """
    # Default colors
    if colors is None:
        colors = {
            "treated_vs_never": "#22c55e",  # Green - clean comparison
            "earlier_vs_later": "#3b82f6",  # Blue - valid comparison
            "later_vs_earlier": "#ef4444",  # Red - forbidden comparison
        }

    # Default titles
    if title is None:
        if plot_type == "scatter":
            title = "Goodman-Bacon Decomposition"
        else:
            title = "TWFE Weight by Comparison Type"

    if plot_type not in ("scatter", "bar"):
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'scatter' or 'bar'.")

    if backend == "plotly":
        return _render_bacon_plotly(
            results=results,
            plot_type=plot_type,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            colors=colors,
            marker=marker,
            markersize=markersize,
            alpha=alpha,
            show_weighted_avg=show_weighted_avg,
            show_twfe_line=show_twfe_line,
            show=show,
        )

    return _render_bacon_mpl(
        results=results,
        plot_type=plot_type,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        marker=marker,
        markersize=markersize,
        alpha=alpha,
        show_weighted_avg=show_weighted_avg,
        show_twfe_line=show_twfe_line,
        ax=ax,
        show=show,
    )


def _render_bacon_mpl(
    *,
    results,
    plot_type,
    figsize,
    title,
    xlabel,
    ylabel,
    colors,
    marker,
    markersize,
    alpha,
    show_weighted_avg,
    show_twfe_line,
    ax,
    show,
):
    """Render Bacon decomposition plot with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if plot_type == "scatter":
        _plot_bacon_scatter(
            ax,
            results,
            colors,
            marker,
            markersize,
            alpha,
            show_weighted_avg,
            show_twfe_line,
            xlabel,
            ylabel,
            title,
        )
    else:
        _plot_bacon_bar(ax, results, colors, alpha, ylabel, title)

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _plot_bacon_scatter(
    ax: Any,
    results: "BaconDecompositionResults",
    colors: Dict[str, str],
    marker: str,
    markersize: int,
    alpha: float,
    show_weighted_avg: bool,
    show_twfe_line: bool,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Create scatter plot of Bacon decomposition."""
    # Separate comparisons by type
    by_type: Dict[str, List[Tuple[float, float]]] = {
        "treated_vs_never": [],
        "earlier_vs_later": [],
        "later_vs_earlier": [],
    }

    for comp in results.comparisons:
        by_type[comp.comparison_type].append((comp.estimate, comp.weight))

    # Plot each type
    labels = {
        "treated_vs_never": "Treated vs Never-treated",
        "earlier_vs_later": "Earlier vs Later treated",
        "later_vs_earlier": "Later vs Earlier (forbidden)",
    }

    for ctype, points in by_type.items():
        if not points:
            continue
        estimates = [p[0] for p in points]
        weights = [p[1] for p in points]
        ax.scatter(
            estimates,
            weights,
            c=colors[ctype],
            label=labels[ctype],
            marker=marker,
            s=markersize,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.5,
        )

    # Show weighted average lines
    if show_weighted_avg:
        effect_by_type = results.effect_by_type()
        for ctype, avg_effect in effect_by_type.items():
            if avg_effect is not None and by_type[ctype]:
                ax.axvline(
                    x=avg_effect,
                    color=colors[ctype],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.5,
                )

    # Show TWFE estimate line
    if show_twfe_line:
        ax.axvline(
            x=results.twfe_estimate,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"TWFE = {results.twfe_estimate:.4f}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Add zero line
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)


def _plot_bacon_bar(
    ax: Any,
    results: "BaconDecompositionResults",
    colors: Dict[str, str],
    alpha: float,
    ylabel: str,
    title: str,
) -> None:
    """Create stacked bar chart of weights by comparison type."""
    # Get weights
    weights = results.weight_by_type()

    # Labels and colors
    type_order = ["treated_vs_never", "earlier_vs_later", "later_vs_earlier"]
    labels = {
        "treated_vs_never": "Treated vs Never-treated",
        "earlier_vs_later": "Earlier vs Later",
        "later_vs_earlier": "Later vs Earlier\n(forbidden)",
    }

    # Create bar data
    bar_labels = [labels[t] for t in type_order]
    bar_weights = [weights[t] for t in type_order]
    bar_colors = [colors[t] for t in type_order]

    # Create bars
    bars = ax.bar(
        bar_labels,
        bar_weights,
        color=bar_colors,
        alpha=alpha,
        edgecolor="white",
        linewidth=1,
    )

    # Add percentage labels on bars
    for bar, weight in zip(bars, bar_weights):
        if weight > 0.01:  # Only label if > 1%
            height = bar.get_height()
            ax.annotate(
                f"{weight:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Add weighted average effect annotations
    effects = results.effect_by_type()
    for bar, ctype in zip(bars, type_order):
        effect = effects[ctype]
        if effect is not None and weights[ctype] > 0.01:
            ax.annotate(
                f"β = {effect:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)

    # Add horizontal line at total weight = 1
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    # Add TWFE estimate as text
    ax.text(
        0.98,
        0.98,
        f"TWFE = {results.twfe_estimate:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def _render_bacon_plotly(
    *,
    results,
    plot_type,
    title,
    xlabel,
    ylabel,
    colors,
    marker,
    markersize,
    alpha,
    show_weighted_avg,
    show_twfe_line,
    show,
):
    """Render Bacon decomposition plot with plotly."""
    from diff_diff.visualization._common import (
        _mpl_marker_to_plotly_symbol,
        _plotly_default_layout,
        _require_plotly,
    )

    go = _require_plotly()

    fig = go.Figure()

    if plot_type == "scatter":
        # Separate comparisons by type
        by_type = {
            "treated_vs_never": [],
            "earlier_vs_later": [],
            "later_vs_earlier": [],
        }
        for comp in results.comparisons:
            by_type[comp.comparison_type].append((comp.estimate, comp.weight))

        labels = {
            "treated_vs_never": "Treated vs Never-treated",
            "earlier_vs_later": "Earlier vs Later treated",
            "later_vs_earlier": "Later vs Earlier (forbidden)",
        }

        # Convert matplotlib scatter area (points^2) to plotly diameter (px)
        plotly_size = max(1, int(round(markersize**0.5)))
        symbol = _mpl_marker_to_plotly_symbol(marker)

        for ctype, points in by_type.items():
            if not points:
                continue
            estimates = [p[0] for p in points]
            weights = [p[1] for p in points]
            fig.add_trace(
                go.Scatter(
                    x=estimates,
                    y=weights,
                    mode="markers",
                    marker=dict(
                        color=colors[ctype],
                        size=plotly_size,
                        symbol=symbol,
                        opacity=alpha,
                    ),
                    name=labels[ctype],
                )
            )

        # Weighted average lines
        if show_weighted_avg:
            effect_by_type = results.effect_by_type()
            for ctype, avg_effect in effect_by_type.items():
                if avg_effect is not None and by_type[ctype]:
                    fig.add_vline(
                        x=avg_effect,
                        line_dash="dash",
                        line_color=colors[ctype],
                        opacity=0.5,
                        line_width=1.5,
                    )

        # TWFE line
        if show_twfe_line:
            fig.add_vline(
                x=results.twfe_estimate,
                line_color="black",
                line_width=2,
                annotation_text=f"TWFE = {results.twfe_estimate:.4f}",
            )

        # Zero line
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

        _plotly_default_layout(fig, title=title, xlabel=xlabel, ylabel=ylabel)

    else:  # bar
        weights = results.weight_by_type()
        type_order = ["treated_vs_never", "earlier_vs_later", "later_vs_earlier"]
        labels = {
            "treated_vs_never": "Treated vs Never-treated",
            "earlier_vs_later": "Earlier vs Later",
            "later_vs_earlier": "Later vs Earlier (forbidden)",
        }

        fig.add_trace(
            go.Bar(
                x=[labels[t] for t in type_order],
                y=[weights[t] for t in type_order],
                marker_color=[colors[t] for t in type_order],
                opacity=alpha,
                text=[f"{weights[t]:.1%}" for t in type_order],
                textposition="outside",
            )
        )

        fig.update_layout(yaxis_range=[0, 1.1])
        _plotly_default_layout(fig, title=title, xlabel=None, ylabel=ylabel, show_legend=False)

    if show:
        fig.show()

    return fig
