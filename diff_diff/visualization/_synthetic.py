"""Synthetic control visualization functions."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from diff_diff.results import SyntheticDiDResults


def plot_synth_weights(
    results: Optional["SyntheticDiDResults"] = None,
    *,
    weights: Optional[Dict[Any, float]] = None,
    weight_type: str = "unit",
    top_n: Optional[int] = None,
    min_weight: float = 0.001,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    color: str = "#2563eb",
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot synthetic control weights as a bar chart.

    Visualizes the unit weights or time weights from a Synthetic
    Difference-in-Differences estimation.

    Parameters
    ----------
    results : SyntheticDiDResults, optional
        Results from SyntheticDiD estimator. Extracts weights based on
        ``weight_type``.
    weights : dict, optional
        Dictionary mapping unit/period IDs to weights. Used if results
        is None.
    weight_type : str, default="unit"
        Which weights to plot: ``"unit"`` for control unit weights or
        ``"time"`` for pre-treatment time weights.
    top_n : int, optional
        Show only the top N weights by magnitude. Useful when there
        are many control units.
    min_weight : float, default=0.001
        Minimum weight threshold for display.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, auto-generated based on ``weight_type``.
    color : str, default="#2563eb"
        Bar color.
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
    """
    # Extract weights
    if results is not None and weights is not None:
        raise ValueError("Provide either 'results' or 'weights', not both.")

    if results is not None:
        if weight_type == "unit":
            weights = results.unit_weights
        elif weight_type == "time":
            weights = results.time_weights
        else:
            raise ValueError(f"weight_type must be 'unit' or 'time', got '{weight_type}'")

    if weights is None:
        raise ValueError("Must provide either 'results' or 'weights'.")

    if not weights:
        raise ValueError("No weights available to plot.")

    # Filter by min_weight
    filtered = {k: v for k, v in weights.items() if abs(v) >= min_weight}
    if not filtered:
        raise ValueError(f"No weights >= {min_weight} to plot.")

    # Sort by weight descending
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    # Apply top_n limit
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    labels = [str(k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Auto-generate title
    if title is None:
        if weight_type == "unit":
            title = "Synthetic Control Unit Weights"
        else:
            title = "Synthetic Control Time Weights"

    if backend == "plotly":
        return _render_synth_weights_plotly(
            labels=labels,
            values=values,
            title=title,
            color=color,
            weight_type=weight_type,
            show=show,
        )

    return _render_synth_weights_mpl(
        labels=labels,
        values=values,
        figsize=figsize,
        title=title,
        color=color,
        weight_type=weight_type,
        ax=ax,
        show=show,
    )


def _render_synth_weights_mpl(*, labels, values, figsize, title, color, weight_type, ax, show):
    """Render synthetic control weights with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Horizontal bar chart
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Highest weight at top

    xlabel = "Weight"
    ylabel = "Control Unit" if weight_type == "unit" else "Time Period"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_synth_weights_plotly(*, labels, values, title, color, weight_type, show):
    """Render synthetic control weights with plotly."""
    from diff_diff.visualization._common import _plotly_default_layout, _require_plotly

    go = _require_plotly()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=color,
            opacity=0.8,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        )
    )

    ylabel = "Control Unit" if weight_type == "unit" else "Time Period"
    _plotly_default_layout(
        fig,
        title=title,
        xlabel="Weight",
        ylabel=ylabel,
        show_legend=False,
    )
    # Reverse y-axis so highest weight is at top
    fig.update_yaxes(autorange="reversed")

    if show:
        fig.show()

    return fig
