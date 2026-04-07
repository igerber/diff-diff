"""Power analysis visualization functions."""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from diff_diff.power import PowerResults, SimulationPowerResults
    from diff_diff.pretrends import PreTrendsPowerCurve, PreTrendsPowerResults


def plot_power_curve(
    results: Optional[Union["PowerResults", "SimulationPowerResults", pd.DataFrame]] = None,
    *,
    effect_sizes: Optional[List[float]] = None,
    powers: Optional[List[float]] = None,
    mde: Optional[float] = None,
    target_power: float = 0.80,
    plot_type: str = "effect",
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Power",
    color: str = "#2563eb",
    mde_color: str = "#dc2626",
    target_color: str = "#22c55e",
    linewidth: float = 2.0,
    show_mde_line: bool = True,
    show_target_line: bool = True,
    show_grid: bool = True,
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Create a power curve visualization.

    Shows how statistical power changes with effect size or sample size,
    helping researchers understand the trade-offs in study design.

    Parameters
    ----------
    results : PowerResults, SimulationPowerResults, or DataFrame, optional
        Results object from PowerAnalysis or simulate_power(), or a DataFrame
        with columns 'effect_size' and 'power' (or 'sample_size' and 'power').
        If None, must provide effect_sizes and powers directly.
    effect_sizes : list of float, optional
        Effect sizes (x-axis values). Required if results is None.
    powers : list of float, optional
        Power values (y-axis values). Required if results is None.
    mde : float, optional
        Minimum detectable effect to mark on the plot.
    target_power : float, default=0.80
        Target power level to show as horizontal line.
    plot_type : str, default="effect"
        Type of power curve: "effect" (power vs effect size) or
        "sample" (power vs sample size).
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, uses a sensible default.
    xlabel : str, optional
        X-axis label. If None, uses a sensible default.
    ylabel : str, default="Power"
        Y-axis label.
    color : str, default="#2563eb"
        Color for the power curve line.
    mde_color : str, default="#dc2626"
        Color for the MDE vertical line.
    target_color : str, default="#22c55e"
        Color for the target power horizontal line.
    linewidth : float, default=2.0
        Line width for the power curve.
    show_mde_line : bool, default=True
        Whether to show vertical line at MDE.
    show_target_line : bool, default=True
        Whether to show horizontal line at target power.
    show_grid : bool, default=True
        Whether to show grid lines.
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
    From PowerAnalysis results:

    >>> from diff_diff import PowerAnalysis, plot_power_curve
    >>> pa = PowerAnalysis(power=0.80)
    >>> curve_df = pa.power_curve(n_treated=50, n_control=50, sigma=5.0)
    >>> mde_result = pa.mde(n_treated=50, n_control=50, sigma=5.0)
    >>> plot_power_curve(curve_df, mde=mde_result.mde)

    From simulation results:

    >>> from diff_diff import simulate_power, DifferenceInDifferences
    >>> results = simulate_power(
    ...     DifferenceInDifferences(),
    ...     effect_sizes=[1, 2, 3, 5, 7, 10],
    ...     n_simulations=200
    ... )
    >>> plot_power_curve(results)

    Manual data:

    >>> plot_power_curve(
    ...     effect_sizes=[1, 2, 3, 4, 5],
    ...     powers=[0.2, 0.5, 0.75, 0.90, 0.97],
    ...     mde=2.5,
    ...     target_power=0.80
    ... )
    """
    # Extract data from results if provided
    if results is not None:
        if isinstance(results, pd.DataFrame):
            if "effect_size" in results.columns:
                effect_sizes = results["effect_size"].tolist()
                plot_type = "effect"
            elif "sample_size" in results.columns:
                effect_sizes = results["sample_size"].tolist()
                plot_type = "sample"
            else:
                raise ValueError("DataFrame must have 'effect_size' or 'sample_size' column")
            powers = results["power"].tolist()
        elif hasattr(results, "effect_sizes") and hasattr(results, "powers"):
            # SimulationPowerResults
            effect_sizes = results.effect_sizes
            powers = results.powers
            if mde is None and hasattr(results, "true_effect"):
                mde = results.true_effect
        elif hasattr(results, "mde"):
            raise ValueError(
                "PowerResults should be used to get mde value, not as direct input. "
                "Use PowerAnalysis.power_curve() to generate curve data."
            )
        else:
            raise TypeError(f"Cannot extract power curve data from {type(results).__name__}")
    elif effect_sizes is None or powers is None:
        raise ValueError("Must provide either 'results' or both 'effect_sizes' and 'powers'")

    # Default titles and labels
    if title is None:
        title = "Power Curve" if plot_type == "effect" else "Power vs Sample Size"
    if xlabel is None:
        xlabel = "Effect Size" if plot_type == "effect" else "Sample Size"

    if backend == "plotly":
        return _render_power_curve_plotly(
            effect_sizes=effect_sizes,
            powers=powers,
            mde=mde,
            target_power=target_power,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            mde_color=mde_color,
            target_color=target_color,
            linewidth=linewidth,
            show_mde_line=show_mde_line,
            show_target_line=show_target_line,
            show_grid=show_grid,
            show=show,
        )

    return _render_power_curve_mpl(
        effect_sizes=effect_sizes,
        powers=powers,
        mde=mde,
        target_power=target_power,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        mde_color=mde_color,
        target_color=target_color,
        linewidth=linewidth,
        show_mde_line=show_mde_line,
        show_target_line=show_target_line,
        show_grid=show_grid,
        ax=ax,
        show=show,
    )


def _render_power_curve_mpl(
    *,
    effect_sizes,
    powers,
    mde,
    target_power,
    figsize,
    title,
    xlabel,
    ylabel,
    color,
    mde_color,
    target_color,
    linewidth,
    show_mde_line,
    show_target_line,
    show_grid,
    ax,
    show,
):
    """Render power curve with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot power curve
    ax.plot(effect_sizes, powers, color=color, linewidth=linewidth, label="Power")

    # Add target power line
    if show_target_line:
        ax.axhline(
            y=target_power,
            color=target_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Target power ({target_power:.0%})",
        )

    # Add MDE line
    if show_mde_line and mde is not None:
        ax.axvline(
            x=mde,
            color=mde_color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"MDE = {mde:.3f}",
        )

        # Mark intersection point
        if mde in effect_sizes:
            idx = effect_sizes.index(mde)
            power_at_mde = powers[idx]
        else:
            effect_arr = np.array(effect_sizes)
            power_arr = np.array(powers)
            if effect_arr.min() <= mde <= effect_arr.max():
                power_at_mde = np.interp(mde, effect_arr, power_arr)
            else:
                power_at_mde = None

        if power_at_mde is not None:
            ax.scatter([mde], [power_at_mde], color=mde_color, s=50, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    if show_grid:
        ax.grid(True, alpha=0.3)

    ax.legend(loc="lower right")
    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_power_curve_plotly(
    *,
    effect_sizes,
    powers,
    mde,
    target_power,
    title,
    xlabel,
    ylabel,
    color,
    mde_color,
    target_color,
    linewidth,
    show_mde_line,
    show_target_line,
    show_grid,
    show,
):
    """Render power curve with plotly."""
    from diff_diff.visualization._common import _plotly_default_layout, _require_plotly

    go = _require_plotly()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=effect_sizes,
            y=powers,
            mode="lines",
            line=dict(color=color, width=linewidth),
            name="Power",
        )
    )

    if show_target_line:
        fig.add_hline(
            y=target_power,
            line_dash="dash",
            line_color=target_color,
            opacity=0.7,
            annotation_text=f"Target ({target_power:.0%})",
        )

    if show_mde_line and mde is not None:
        fig.add_vline(
            x=mde,
            line_dash="dot",
            line_color=mde_color,
            opacity=0.7,
            annotation_text=f"MDE = {mde:.3f}",
        )

    _plotly_default_layout(fig, title=title, xlabel=xlabel, ylabel=ylabel)
    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", showgrid=show_grid)

    if show:
        fig.show()

    return fig


def plot_pretrends_power(
    results: Optional[Union["PreTrendsPowerResults", "PreTrendsPowerCurve", pd.DataFrame]] = None,
    *,
    M_values: Optional[List[float]] = None,
    powers: Optional[List[float]] = None,
    mdv: Optional[float] = None,
    target_power: float = 0.80,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Pre-Trends Test Power Curve",
    xlabel: str = "Violation Magnitude (M)",
    ylabel: str = "Power",
    color: str = "#2563eb",
    mdv_color: str = "#dc2626",
    target_color: str = "#22c55e",
    linewidth: float = 2.0,
    show_mdv_line: bool = True,
    show_target_line: bool = True,
    show_grid: bool = True,
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot pre-trends test power curve.

    Visualizes how the power to detect parallel trends violations changes
    with the violation magnitude (M). This helps understand what violations
    your pre-trends test is capable of detecting.

    Parameters
    ----------
    results : PreTrendsPowerResults, PreTrendsPowerCurve, or DataFrame, optional
        Results from PreTrendsPower.fit() or power_curve(), or a DataFrame
        with columns 'M' and 'power'. If None, must provide M_values and powers.
    M_values : list of float, optional
        Violation magnitudes (x-axis). Required if results is None.
    powers : list of float, optional
        Power values (y-axis). Required if results is None.
    mdv : float, optional
        Minimum detectable violation to mark on the plot.
    target_power : float, default=0.80
        Target power level to show as horizontal line.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    color : str, default="#2563eb"
        Color for the power curve line.
    mdv_color : str, default="#dc2626"
        Color for the MDV vertical line.
    target_color : str, default="#22c55e"
        Color for the target power horizontal line.
    linewidth : float, default=2.0
        Line width for the power curve.
    show_mdv_line : bool, default=True
        Whether to show vertical line at MDV.
    show_target_line : bool, default=True
        Whether to show horizontal line at target power.
    show_grid : bool, default=True
        Whether to show grid lines.
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
    From PreTrendsPower results:

    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.pretrends import PreTrendsPower
    >>> from diff_diff.visualization import plot_pretrends_power
    >>>
    >>> mp_did = MultiPeriodDiD()
    >>> event_results = mp_did.fit(data, outcome='y', treatment='treated',
    ...                            time='period', post_periods=[4, 5, 6, 7])
    >>>
    >>> pt = PreTrendsPower()
    >>> curve = pt.power_curve(event_results)
    >>> plot_pretrends_power(curve)

    Notes
    -----
    The power curve shows how likely you are to reject the null hypothesis
    of parallel trends given a true violation of magnitude M.

    See Also
    --------
    PreTrendsPower : Main class for pre-trends power analysis
    plot_sensitivity : Plot HonestDiD sensitivity analysis
    """
    # Extract data from results if provided
    if results is not None:
        if isinstance(results, pd.DataFrame):
            if "M" not in results.columns or "power" not in results.columns:
                raise ValueError("DataFrame must have 'M' and 'power' columns")
            M_values = results["M"].tolist()
            powers = results["power"].tolist()
        elif hasattr(results, "M_values") and hasattr(results, "powers"):
            # PreTrendsPowerCurve
            M_values = results.M_values.tolist()
            powers = results.powers.tolist()
            if mdv is None:
                mdv = results.mdv
            if target_power is None:
                target_power = results.target_power
        elif hasattr(results, "mdv") and hasattr(results, "power"):
            # Single PreTrendsPowerResults
            if mdv is None:
                mdv = results.mdv
            if np.isfinite(mdv):
                M_values = [0, mdv * 0.5, mdv, mdv * 1.5, mdv * 2]
            else:
                M_values = [0, 1, 2, 3, 4]
            powers = None
        else:
            raise TypeError(f"Cannot extract power curve data from {type(results).__name__}")
    elif M_values is None or powers is None:
        raise ValueError("Must provide either 'results' or both 'M_values' and 'powers'")

    if backend == "plotly":
        return _render_pretrends_power_plotly(
            M_values=M_values,
            powers=powers,
            mdv=mdv,
            target_power=target_power,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            mdv_color=mdv_color,
            target_color=target_color,
            linewidth=linewidth,
            show_mdv_line=show_mdv_line,
            show_target_line=show_target_line,
            show_grid=show_grid,
            show=show,
        )

    return _render_pretrends_power_mpl(
        M_values=M_values,
        powers=powers,
        mdv=mdv,
        target_power=target_power,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        mdv_color=mdv_color,
        target_color=target_color,
        linewidth=linewidth,
        show_mdv_line=show_mdv_line,
        show_target_line=show_target_line,
        show_grid=show_grid,
        ax=ax,
        show=show,
    )


def _render_pretrends_power_mpl(
    *,
    M_values,
    powers,
    mdv,
    target_power,
    figsize,
    title,
    xlabel,
    ylabel,
    color,
    mdv_color,
    target_color,
    linewidth,
    show_mdv_line,
    show_target_line,
    show_grid,
    ax,
    show,
):
    """Render pre-trends power curve with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot power curve if we have powers
    if powers is not None:
        ax.plot(M_values, powers, color=color, linewidth=linewidth, label="Power")

    # Add target power line
    if show_target_line:
        ax.axhline(
            y=target_power,
            color=target_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Target power ({target_power:.0%})",
        )

    # Add MDV line
    if show_mdv_line and mdv is not None and np.isfinite(mdv):
        ax.axvline(
            x=mdv,
            color=mdv_color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"MDV = {mdv:.3f}",
        )

        # Mark intersection point if we have powers
        if powers is not None:
            M_arr = np.array(M_values)
            power_arr = np.array(powers)
            if M_arr.min() <= mdv <= M_arr.max():
                power_at_mdv = np.interp(mdv, M_arr, power_arr)
                ax.scatter([mdv], [power_at_mdv], color=mdv_color, s=50, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    if show_grid:
        ax.grid(True, alpha=0.3)

    ax.legend(loc="lower right")
    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_pretrends_power_plotly(
    *,
    M_values,
    powers,
    mdv,
    target_power,
    title,
    xlabel,
    ylabel,
    color,
    mdv_color,
    target_color,
    linewidth,
    show_mdv_line,
    show_target_line,
    show_grid,
    show,
):
    """Render pre-trends power curve with plotly."""
    from diff_diff.visualization._common import _plotly_default_layout, _require_plotly

    go = _require_plotly()

    fig = go.Figure()

    if powers is not None:
        fig.add_trace(
            go.Scatter(
                x=M_values,
                y=powers,
                mode="lines",
                line=dict(color=color, width=linewidth),
                name="Power",
            )
        )

    if show_target_line:
        fig.add_hline(
            y=target_power,
            line_dash="dash",
            line_color=target_color,
            opacity=0.7,
            annotation_text=f"Target ({target_power:.0%})",
        )

    if show_mdv_line and mdv is not None and np.isfinite(mdv):
        fig.add_vline(
            x=mdv,
            line_dash="dot",
            line_color=mdv_color,
            opacity=0.7,
            annotation_text=f"MDV = {mdv:.3f}",
        )

    _plotly_default_layout(fig, title=title, xlabel=xlabel, ylabel=ylabel)
    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", showgrid=show_grid)

    if show:
        fig.show()

    return fig
