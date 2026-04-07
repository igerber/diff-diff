"""Continuous DiD visualization functions (dose-response curves)."""

from typing import TYPE_CHECKING, Any, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from diff_diff.continuous_did_results import ContinuousDiDResults, DoseResponseCurve


def plot_dose_response(
    results: Optional["ContinuousDiDResults"] = None,
    *,
    curve: Optional["DoseResponseCurve"] = None,
    data: Optional[pd.DataFrame] = None,
    target: str = "att",
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    xlabel: str = "Dose",
    ylabel: str = "Treatment Effect",
    color: str = "#2563eb",
    ci_color: Optional[str] = None,
    show_zero_line: bool = True,
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot dose-response curve from Continuous DiD estimation.

    Visualizes how the treatment effect varies with the treatment dose
    (intensity), with confidence bands.

    Parameters
    ----------
    results : ContinuousDiDResults, optional
        Results from ContinuousDiD estimator. Extracts the dose-response
        curve based on ``target``.
    curve : DoseResponseCurve, optional
        A DoseResponseCurve object directly.
    data : pd.DataFrame, optional
        DataFrame with columns ``dose``, ``effect``, ``se`` (and optionally
        ``conf_int_lower``, ``conf_int_upper``).
    target : str, default="att"
        Which dose-response curve: ``"att"`` or ``"acrt"``.
    alpha : float, default=0.05
        Significance level for confidence intervals (used with DataFrame input).
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. Auto-generated if None.
    xlabel : str, default="Dose"
        X-axis label.
    ylabel : str, default="Treatment Effect"
        Y-axis label.
    color : str, default="#2563eb"
        Color for the line.
    ci_color : str, optional
        Color for confidence band. Defaults to ``color`` with transparency.
    show_zero_line : bool, default=True
        Whether to show a horizontal line at y=0.
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
    from scipy import stats as scipy_stats

    # Extract dose-response data
    if sum(x is not None for x in (results, curve, data)) != 1:
        raise ValueError("Provide exactly one of 'results', 'curve', or 'data'.")

    if results is not None:
        if target == "att":
            curve = results.dose_response_att
        elif target == "acrt":
            curve = results.dose_response_acrt
        else:
            raise ValueError(f"target must be 'att' or 'acrt', got '{target}'")

    if curve is not None:
        # Infer target from curve when passed directly (not via results)
        if results is None and hasattr(curve, "target") and curve.target:
            target = curve.target
        dose_grid = curve.dose_grid
        effects = curve.effects
        ci_lower = curve.conf_int_lower
        ci_upper = curve.conf_int_upper
    elif data is not None:
        if "dose" not in data.columns or "effect" not in data.columns:
            raise ValueError("DataFrame must have 'dose' and 'effect' columns")
        dose_grid = data["dose"].values
        effects = data["effect"].values
        if "conf_int_lower" in data.columns and "conf_int_upper" in data.columns:
            ci_lower = data["conf_int_lower"].values
            ci_upper = data["conf_int_upper"].values
        elif "se" in data.columns:
            z = scipy_stats.norm.ppf(1 - alpha / 2)
            ci_lower = effects - z * data["se"].values
            ci_upper = effects + z * data["se"].values
        else:
            ci_lower = None
            ci_upper = None
    else:
        raise ValueError("Must provide 'results', 'curve', or 'data'.")

    # Auto-generate title
    if title is None:
        if target == "att":
            title = "ATT Dose-Response Curve"
        else:
            title = "ACRT Dose-Response Curve"

    if backend == "plotly":
        return _render_dose_response_plotly(
            dose_grid=dose_grid,
            effects=effects,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            ci_color=ci_color,
            show_zero_line=show_zero_line,
            show=show,
        )

    return _render_dose_response_mpl(
        dose_grid=dose_grid,
        effects=effects,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        ci_color=ci_color,
        show_zero_line=show_zero_line,
        ax=ax,
        show=show,
    )


def _render_dose_response_mpl(
    *,
    dose_grid,
    effects,
    ci_lower,
    ci_upper,
    figsize,
    title,
    xlabel,
    ylabel,
    color,
    ci_color,
    show_zero_line,
    ax,
    show,
):
    """Render dose-response curve with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Zero line
    if show_zero_line:
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Confidence band
    if ci_lower is not None and ci_upper is not None:
        band_color = ci_color or color
        ax.fill_between(
            dose_grid,
            ci_lower,
            ci_upper,
            alpha=0.15,
            color=band_color,
            label="95% CI",
        )

    # Effect line
    ax.plot(dose_grid, effects, color=color, linewidth=2, label="Effect")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_dose_response_plotly(
    *,
    dose_grid,
    effects,
    ci_lower,
    ci_upper,
    title,
    xlabel,
    ylabel,
    color,
    ci_color,
    show_zero_line,
    show,
):
    """Render dose-response curve with plotly."""
    from diff_diff.visualization._common import (
        _color_to_rgba,
        _plotly_default_layout,
        _require_plotly,
    )

    go = _require_plotly()

    fig = go.Figure()

    # Zero line
    if show_zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)

    # Confidence band
    if ci_lower is not None and ci_upper is not None:
        band_color = ci_color or color
        dose_list = list(dose_grid)
        fig.add_trace(
            go.Scatter(
                x=dose_list + dose_list[::-1],
                y=list(ci_upper) + list(ci_lower)[::-1],
                fill="toself",
                fillcolor=_color_to_rgba(band_color, 0.15),
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                hoverinfo="skip",
            )
        )

    # Effect line
    fig.add_trace(
        go.Scatter(
            x=list(dose_grid),
            y=list(effects),
            mode="lines",
            line=dict(color=color, width=2),
            name="Effect",
        )
    )

    _plotly_default_layout(fig, title=title, xlabel=xlabel, ylabel=ylabel)

    if show:
        fig.show()

    return fig
