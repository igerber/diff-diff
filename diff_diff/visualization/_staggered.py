"""Staggered DiD visualization functions (group effects, staircase, heatmap)."""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from diff_diff.continuous_did_results import ContinuousDiDResults
    from diff_diff.efficient_did_results import EfficientDiDResults
    from diff_diff.staggered import CallawaySantAnnaResults


def plot_group_effects(
    results: "CallawaySantAnnaResults",
    *,
    groups: Optional[List[Any]] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Treatment Effects by Cohort",
    xlabel: str = "Time Period",
    ylabel: str = "Treatment Effect",
    alpha: float = 0.05,
    show: bool = True,
    ax: Optional[Any] = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot treatment effects by treatment cohort (group).

    Parameters
    ----------
    results : CallawaySantAnnaResults
        Results from CallawaySantAnna estimator.
    groups : list, optional
        List of groups (cohorts) to plot. If None, plots all groups.
    figsize : tuple, default=(10, 6)
        Figure size.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    show : bool, default=True
        Whether to call plt.show().
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    backend : str, default="matplotlib"
        Plotting backend: ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The axes object (matplotlib) or figure (plotly).
    """
    from scipy import stats as scipy_stats

    if not hasattr(results, "group_time_effects"):
        raise TypeError("results must be a CallawaySantAnnaResults object")

    # Get groups to plot
    if groups is None:
        groups = sorted(set(g for g, t in results.group_time_effects.keys()))

    critical_value = scipy_stats.norm.ppf(1 - alpha / 2)

    # Build data per group
    group_data = {}
    for group in groups:
        group_effects = [
            (t, data) for (g, t), data in results.group_time_effects.items() if g == group
        ]
        group_effects.sort(key=lambda x: x[0])
        if not group_effects:
            continue
        group_data[group] = group_effects

    if backend == "plotly":
        return _render_group_effects_plotly(
            group_data=group_data,
            groups=groups,
            critical_value=critical_value,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            show=show,
        )

    return _render_group_effects_mpl(
        group_data=group_data,
        groups=groups,
        critical_value=critical_value,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        show=show,
    )


def _render_group_effects_mpl(
    *, group_data, groups, critical_value, figsize, title, xlabel, ylabel, ax, show
):
    """Render group effects plot with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cmap = getattr(plt.cm, "tab10", None) or plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(groups)))

    for i, group in enumerate(groups):
        if group not in group_data:
            continue
        group_effects = group_data[group]
        times = [t for t, _ in group_effects]
        effects = [data["effect"] for _, data in group_effects]
        ses = [data["se"] for _, data in group_effects]

        yerr = [
            [e - (e - critical_value * s) for e, s in zip(effects, ses)],
            [(e + critical_value * s) - e for e, s in zip(effects, ses)],
        ]

        ax.errorbar(
            times,
            effects,
            yerr=yerr,
            label=f"Cohort {group}",
            color=colors[i],
            marker="o",
            capsize=3,
            linewidth=1.5,
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_group_effects_plotly(
    *, group_data, groups, critical_value, title, xlabel, ylabel, show
):
    """Render group effects plot with plotly."""
    from diff_diff.visualization._common import _plotly_default_layout, _require_plotly

    go = _require_plotly()

    fig = go.Figure()

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    for group in groups:
        if group not in group_data:
            continue
        group_effects = group_data[group]
        times = [t for t, _ in group_effects]
        effects = [data["effect"] for _, data in group_effects]
        ses = [data["se"] for _, data in group_effects]

        ci_lo = [e - critical_value * s for e, s in zip(effects, ses)]
        ci_hi = [e + critical_value * s for e, s in zip(effects, ses)]

        fig.add_trace(
            go.Scatter(
                x=times,
                y=effects,
                mode="lines+markers",
                name=f"Cohort {group}",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[h - e for e, h in zip(effects, ci_hi)],
                    arrayminus=[e - lo for e, lo in zip(effects, ci_lo)],
                ),
            )
        )

    _plotly_default_layout(fig, title=title, xlabel=xlabel, ylabel=ylabel)

    if show:
        fig.show()

    return fig


def plot_staircase(
    results: Optional["CallawaySantAnnaResults"] = None,
    *,
    data: Optional[pd.DataFrame] = None,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    first_treat: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Treatment Adoption Over Time",
    color: str = "#2563eb",
    show_counts: bool = True,
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot treatment adoption "staircase" for staggered designs.

    Shows how many units enter treatment over time, creating a step-function
    pattern that illustrates the staggered adoption of treatment.

    Parameters
    ----------
    results : CallawaySantAnnaResults, optional
        Results from CallawaySantAnna estimator. Extracts groups and cohort
        sizes from ``group_time_effects``.
    data : pd.DataFrame, optional
        Raw panel data. Must provide ``unit``, ``time``, and ``first_treat``
        column names.
    unit : str, optional
        Column name for unit identifier (required with ``data``).
    time : str, optional
        Column name for time period (required with ``data``).
    first_treat : str, optional
        Column name for first treatment period (required with ``data``).
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    title : str, default="Treatment Adoption Over Time"
        Plot title.
    color : str, default="#2563eb"
        Base color for the staircase.
    show_counts : bool, default=True
        Whether to annotate each step with the cohort size.
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
    # Extract cohort data
    cohort_counts = _extract_staircase_data(results, data, unit, time, first_treat)

    if backend == "plotly":
        return _render_staircase_plotly(
            cohort_counts=cohort_counts,
            title=title,
            color=color,
            show_counts=show_counts,
            show=show,
        )

    return _render_staircase_mpl(
        cohort_counts=cohort_counts,
        figsize=figsize,
        title=title,
        color=color,
        show_counts=show_counts,
        ax=ax,
        show=show,
    )


def _extract_staircase_data(results, data, unit, time, first_treat):
    """Extract cohort periods and counts for the staircase plot.

    Returns
    -------
    list of (period, count) tuples, sorted by period.
    """
    if results is not None and data is not None:
        raise ValueError("Provide either 'results' or 'data', not both.")

    if results is not None:
        if not hasattr(results, "group_time_effects") or not hasattr(results, "groups"):
            raise TypeError("results must be a CallawaySantAnnaResults object")

        groups = sorted(results.groups)
        cohort_counts = []
        for g in groups:
            # Collect n_treated across all (g, t) cells for this cohort.
            # n_treated is a per-cell observation count that can vary with
            # missingness, so we use the max as the best cohort size estimate.
            cell_counts = []
            for (gg, _tt), eff in results.group_time_effects.items():
                if gg == g:
                    n = eff.get("n_treated", eff.get("n_obs", None))
                    if n is not None:
                        cell_counts.append(int(n))
            if not cell_counts:
                cohort_counts.append((g, 0))
                continue
            max_count = max(cell_counts)
            if min(cell_counts) != max_count:
                import warnings

                warnings.warn(
                    f"Cohort {g}: n_treated varies across cells "
                    f"({min(cell_counts)}-{max_count}). "
                    f"Using max as cohort size; pass data= for exact counts.",
                    stacklevel=3,
                )
            cohort_counts.append((g, max_count))

        return cohort_counts

    if data is not None:
        if unit is None or time is None or first_treat is None:
            raise ValueError(
                "When using 'data', must provide 'unit', 'time', and 'first_treat' column names."
            )
        # Count unique units per first_treat cohort
        cohort_df = data.groupby(first_treat)[unit].nunique().reset_index()
        cohort_df.columns = ["period", "count"]
        cohort_df = cohort_df.sort_values("period")
        # Exclude never-treated (inf, NaN, or 0 conventions)
        cohort_df = cohort_df[
            cohort_df["period"].notna()
            & np.isfinite(cohort_df["period"])
            & (cohort_df["period"] > 0)
        ]
        return list(zip(cohort_df["period"], cohort_df["count"]))

    raise ValueError("Must provide either 'results' or 'data'.")


def _render_staircase_mpl(*, cohort_counts, figsize, title, color, show_counts, ax, show):
    """Render staircase plot with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if not cohort_counts:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No treatment cohorts", ha="center", va="center", transform=ax.transAxes)
        if show:
            plt.show()
        return ax

    periods = [p for p, _ in cohort_counts]
    counts = [c for _, c in cohort_counts]
    cumulative = np.cumsum(counts)

    # Create step plot
    ax.step(periods, cumulative, where="post", color=color, linewidth=2, label="Cumulative treated")
    ax.fill_between(periods, cumulative, step="post", alpha=0.15, color=color)

    # Annotate cohort sizes
    if show_counts:
        for i, (period, count) in enumerate(cohort_counts):
            cum = cumulative[i]
            ax.annotate(
                f"+{count}",
                xy=(period, cum),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Cumulative Treated Units")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Set y to start at 0
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_staircase_plotly(*, cohort_counts, title, color, show_counts, show):
    """Render staircase plot with plotly."""
    from diff_diff.visualization._common import (
        _color_to_rgba,
        _plotly_default_layout,
        _require_plotly,
    )

    go = _require_plotly()

    fig = go.Figure()

    if not cohort_counts:
        fig.add_annotation(text="No treatment cohorts", x=0.5, y=0.5, showarrow=False)
        _plotly_default_layout(fig, title=title)
        if show:
            fig.show()
        return fig

    periods = [p for p, _ in cohort_counts]
    counts = [c for _, c in cohort_counts]
    cumulative = list(np.cumsum(counts))

    # Step line
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=cumulative,
            mode="lines",
            line=dict(color=color, width=2, shape="hv"),
            fill="tozeroy",
            fillcolor=_color_to_rgba(color, 0.15),
            name="Cumulative treated",
        )
    )

    # Annotations for cohort sizes
    if show_counts:
        for period, count, cum in zip(periods, counts, cumulative):
            fig.add_annotation(
                x=period,
                y=cum,
                text=f"+{count}",
                showarrow=False,
                yshift=15,
                font=dict(color=color, size=11),
            )

    _plotly_default_layout(
        fig,
        title=title,
        xlabel="Time Period",
        ylabel="Cumulative Treated Units",
    )
    fig.update_yaxes(rangemode="tozero")

    if show:
        fig.show()

    return fig


def plot_group_time_heatmap(
    results: Optional[
        Union["CallawaySantAnnaResults", "EfficientDiDResults", "ContinuousDiDResults"]
    ] = None,
    *,
    data: Optional[pd.DataFrame] = None,
    figsize: Tuple[float, float] = (10, 8),
    title: str = "Group-Time Treatment Effects",
    cmap: str = "RdBu_r",
    center: float = 0.0,
    annotate: bool = True,
    fmt: str = ".3f",
    mask_insignificant: bool = False,
    alpha: float = 0.05,
    ax: Optional[Any] = None,
    show: bool = True,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot heatmap of group-time treatment effects ATT(g,t).

    Displays treatment effects as a colored matrix with treatment cohorts
    (groups) on the y-axis and calendar time periods on the x-axis.

    Parameters
    ----------
    results : CallawaySantAnnaResults, EfficientDiDResults, or ContinuousDiDResults, optional
        Results object with ``group_time_effects`` dict.
    data : pd.DataFrame, optional
        DataFrame with columns ``group``, ``time``, ``effect``
        (and optionally ``p_value``).
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches.
    title : str, default="Group-Time Treatment Effects"
        Plot title.
    cmap : str, default="RdBu_r"
        Colormap name. Diverging colormaps centered at zero work best.
    center : float, default=0.0
        Value to center the colormap at.
    annotate : bool, default=True
        Whether to show effect values in each cell.
    fmt : str, default=".3f"
        Format string for cell annotations.
    mask_insignificant : bool, default=False
        Whether to grey out cells with non-significant effects.
    alpha : float, default=0.05
        Significance level for masking (when ``mask_insignificant=True``).
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
    # Extract data into matrix form
    effect_matrix, p_matrix, group_labels, time_labels = _extract_heatmap_data(results, data)

    if backend == "plotly":
        return _render_group_time_heatmap_plotly(
            effect_matrix=effect_matrix,
            p_matrix=p_matrix,
            group_labels=group_labels,
            time_labels=time_labels,
            title=title,
            cmap=cmap,
            center=center,
            annotate=annotate,
            fmt=fmt,
            mask_insignificant=mask_insignificant,
            alpha=alpha,
            show=show,
        )

    return _render_group_time_heatmap_mpl(
        effect_matrix=effect_matrix,
        p_matrix=p_matrix,
        group_labels=group_labels,
        time_labels=time_labels,
        figsize=figsize,
        title=title,
        cmap=cmap,
        center=center,
        annotate=annotate,
        fmt=fmt,
        mask_insignificant=mask_insignificant,
        alpha=alpha,
        ax=ax,
        show=show,
    )


def _extract_heatmap_data(results, data):
    """Extract group-time effects into a 2D matrix.

    Returns
    -------
    effect_matrix : np.ndarray
        2D array of effects (groups x time).
    p_matrix : np.ndarray or None
        2D array of p-values, or None if unavailable.
    group_labels : list
        Sorted group labels.
    time_labels : list
        Sorted time labels.
    """
    if results is not None and data is not None:
        raise ValueError("Provide either 'results' or 'data', not both.")

    if results is not None:
        if not hasattr(results, "group_time_effects"):
            raise TypeError(f"{type(results).__name__} does not have group_time_effects attribute")
        gte = results.group_time_effects
        if not gte:
            raise ValueError("group_time_effects is empty — nothing to plot.")

        groups = sorted(set(g for g, t in gte.keys()))
        times = sorted(set(t for g, t in gte.keys()))

        effect_matrix = np.full((len(groups), len(times)), np.nan)
        p_matrix = np.full((len(groups), len(times)), np.nan)

        group_idx = {g: i for i, g in enumerate(groups)}
        time_idx = {t: j for j, t in enumerate(times)}

        for (g, t), eff_data in gte.items():
            i, j = group_idx[g], time_idx[t]
            # Handle different result type structures
            if "effect" in eff_data:
                effect_matrix[i, j] = eff_data["effect"]
            elif "att_glob" in eff_data:
                effect_matrix[i, j] = eff_data["att_glob"]
            if "p_value" in eff_data:
                p_matrix[i, j] = eff_data["p_value"]

        has_p = np.any(np.isfinite(p_matrix))
        return effect_matrix, p_matrix if has_p else None, groups, times

    if data is not None:
        required = {"group", "time", "effect"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        pivot = data.pivot(index="group", columns="time", values="effect")
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

        p_matrix = None
        if "p_value" in data.columns:
            p_pivot = data.pivot(index="group", columns="time", values="p_value")
            p_pivot = p_pivot.sort_index(axis=0).sort_index(axis=1)
            p_matrix = p_pivot.values

        return pivot.values, p_matrix, list(pivot.index), list(pivot.columns)

    raise ValueError("Must provide either 'results' or 'data'.")


def _render_group_time_heatmap_mpl(
    *,
    effect_matrix,
    p_matrix,
    group_labels,
    time_labels,
    figsize,
    title,
    cmap,
    center,
    annotate,
    fmt,
    mask_insignificant,
    alpha,
    ax,
    show,
):
    """Render group-time heatmap with matplotlib."""
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()
    from matplotlib.colors import TwoSlopeNorm

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    display_matrix = effect_matrix.copy()

    # Build significance mask
    sig_mask = None
    if mask_insignificant and p_matrix is not None:
        sig_mask = p_matrix > alpha

    # Compute color normalization centered at `center`
    finite_vals = effect_matrix[np.isfinite(effect_matrix)]
    vmin = center - 0.01
    vmax = center + 0.01
    if len(finite_vals) > 0:
        vmin = np.nanmin(finite_vals)
        vmax = np.nanmax(finite_vals)
        # Ensure center is between vmin and vmax
        if vmin >= center:
            vmin = center - 0.01
        if vmax <= center:
            vmax = center + 0.01
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    im = ax.imshow(display_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Add colorbar
    fig.colorbar(im, ax=ax, label="Treatment Effect")

    # Set ticks
    ax.set_xticks(range(len(time_labels)))
    ax.set_xticklabels([str(t) for t in time_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(group_labels)))
    ax.set_yticklabels([str(g) for g in group_labels])

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Treatment Cohort")
    ax.set_title(title)

    # Annotate cells
    if annotate:
        for i in range(len(group_labels)):
            for j in range(len(time_labels)):
                val = effect_matrix[i, j]
                if np.isnan(val):
                    continue
                is_masked = sig_mask is not None and sig_mask[i, j]
                text_color = (
                    "gray"
                    if is_masked
                    else ("white" if abs(val - center) > (vmax - vmin) * 0.3 else "black")
                )
                ax.text(
                    j,
                    i,
                    f"{val:{fmt}}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    # Grey out insignificant cells
    if sig_mask is not None:
        for i in range(sig_mask.shape[0]):
            for j in range(sig_mask.shape[1]):
                if sig_mask[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            facecolor="white",
                            alpha=0.6,
                            edgecolor="none",
                        )
                    )

    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _render_group_time_heatmap_plotly(
    *,
    effect_matrix,
    p_matrix,
    group_labels,
    time_labels,
    title,
    cmap,
    center,
    annotate,
    fmt,
    mask_insignificant,
    alpha,
    show,
):
    """Render group-time heatmap with plotly."""
    from diff_diff.visualization._common import _plotly_default_layout, _require_plotly

    go = _require_plotly()

    # Pass cmap name through to plotly unchanged — plotly supports the same
    # diverging colorscale names as matplotlib (RdBu, RdBu_r, etc.)
    plotly_cmap = cmap

    # Build text annotations
    text = None
    if annotate:
        text = []
        for i in range(effect_matrix.shape[0]):
            row = []
            for j in range(effect_matrix.shape[1]):
                val = effect_matrix[i, j]
                if np.isnan(val):
                    row.append("")
                else:
                    row.append(f"{val:{fmt}}")
            text.append(row)

    # Build significance mask for overlay (do NOT replace with NaN — that
    # conflates "insignificant" with "missing cell")
    sig_mask = None
    if mask_insignificant and p_matrix is not None:
        sig_mask = p_matrix > alpha

    # Center the colorscale
    finite_vals = effect_matrix[np.isfinite(effect_matrix)]
    if len(finite_vals) > 0:
        abs_max = max(abs(np.nanmin(finite_vals) - center), abs(np.nanmax(finite_vals) - center))
        zmin = center - abs_max
        zmax = center + abs_max
    else:
        zmin, zmax = -1, 1

    # Main heatmap — always shows all values (insignificant cells greyed via opacity)
    fig = go.Figure(
        data=go.Heatmap(
            z=effect_matrix,
            x=[str(t) for t in time_labels],
            y=[str(g) for g in group_labels],
            colorscale=plotly_cmap,
            zmin=zmin,
            zmax=zmax,
            text=text,
            texttemplate="%{text}" if annotate else None,
            colorbar=dict(title="Effect"),
        )
    )

    # Grey overlay for insignificant cells (preserves underlying value)
    if sig_mask is not None and np.any(sig_mask):
        grey_z = np.where(sig_mask, 1.0, np.nan)
        fig.add_trace(
            go.Heatmap(
                z=grey_z,
                x=[str(t) for t in time_labels],
                y=[str(g) for g in group_labels],
                colorscale=[[0, "rgba(255,255,255,0.6)"], [1, "rgba(255,255,255,0.6)"]],
                showscale=False,
                hoverinfo="skip",
            )
        )

    _plotly_default_layout(
        fig,
        title=title,
        xlabel="Time Period",
        ylabel="Treatment Cohort",
        show_legend=False,
    )

    if show:
        fig.show()

    return fig
