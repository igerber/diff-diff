"""
Visualization functions for difference-in-differences analysis.

Provides event study plots and other diagnostic visualizations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Type alias for results that can be plotted
PlottableResults = Union[
    "MultiPeriodDiDResults",
    "CallawaySantAnnaResults",
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
        The reference period (normalized to effect=0). Will be shown as a
        hollow marker. If None, tries to infer from results.
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

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

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

    2. **Reference period**: Usually the last pre-treatment period (t=-1),
       normalized to zero. This is the omitted category.

    3. **Post-treatment periods**: The treatment effects of interest. These
       show how the outcome evolved after treatment.

    The confidence intervals help assess statistical significance. Effects
    whose CIs don't include zero are typically considered significant.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    from scipy import stats as scipy_stats

    # Extract data from results if provided
    if results is not None:
        effects, se, periods, pre_periods, post_periods, reference_period = \
            _extract_plot_data(results, periods, pre_periods, post_periods, reference_period)
    elif effects is None or se is None:
        raise ValueError(
            "Must provide either 'results' or both 'effects' and 'se'"
        )

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

    plot_data = []
    for period in periods:
        effect = effects.get(period, np.nan)
        std_err = se.get(period, np.nan)

        if np.isnan(effect) or np.isnan(std_err):
            continue

        ci_lower = effect - critical_value * std_err
        ci_upper = effect + critical_value * std_err

        plot_data.append({
            'period': period,
            'effect': effect,
            'se': std_err,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_reference': period == reference_period,
        })

    if not plot_data:
        raise ValueError("No valid data to plot")

    df = pd.DataFrame(plot_data)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Convert periods to numeric for plotting
    period_to_x = {p: i for i, p in enumerate(df['period'])}
    x_vals = [period_to_x[p] for p in df['period']]

    # Shade pre-treatment region
    if shade_pre and pre_periods is not None:
        pre_x = [period_to_x[p] for p in pre_periods if p in period_to_x]
        if pre_x:
            ax.axvspan(min(pre_x) - 0.5, max(pre_x) + 0.5,
                       color=shade_color, alpha=0.5, zorder=0)

    # Draw horizontal zero line
    if show_zero_line:
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=1)

    # Draw vertical reference line
    if show_reference_line and reference_period is not None:
        if reference_period in period_to_x:
            ref_x = period_to_x[reference_period]
            ax.axvline(x=ref_x, color='gray', linestyle=':', linewidth=1, zorder=1)

    # Plot error bars
    yerr = [df['effect'] - df['ci_lower'], df['ci_upper'] - df['effect']]
    ax.errorbar(
        x_vals, df['effect'], yerr=yerr,
        fmt='none', color=color, capsize=capsize, linewidth=linewidth,
        capthick=linewidth, zorder=2
    )

    # Plot point estimates
    for i, row in df.iterrows():
        x = period_to_x[row['period']]
        if row['is_reference']:
            # Hollow marker for reference period
            ax.plot(x, row['effect'], marker=marker, markersize=markersize,
                    markerfacecolor='white', markeredgecolor=color,
                    markeredgewidth=2, zorder=3)
        else:
            ax.plot(x, row['effect'], marker=marker, markersize=markersize,
                    color=color, zorder=3)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set x-axis ticks
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(p) for p in df['period']])

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Tight layout
    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _extract_plot_data(
    results: PlottableResults,
    periods: Optional[List[Any]],
    pre_periods: Optional[List[Any]],
    post_periods: Optional[List[Any]],
    reference_period: Optional[Any],
) -> Tuple[Dict, Dict, List, List, List, Any]:
    """
    Extract plotting data from various result types.

    Returns
    -------
    tuple
        (effects, se, periods, pre_periods, post_periods, reference_period)
    """
    # Handle DataFrame input
    if isinstance(results, pd.DataFrame):
        if 'period' not in results.columns:
            raise ValueError("DataFrame must have 'period' column")
        if 'effect' not in results.columns:
            raise ValueError("DataFrame must have 'effect' column")
        if 'se' not in results.columns:
            raise ValueError("DataFrame must have 'se' column")

        effects = dict(zip(results['period'], results['effect']))
        se = dict(zip(results['period'], results['se']))

        if periods is None:
            periods = list(results['period'])

        return effects, se, periods, pre_periods, post_periods, reference_period

    # Handle MultiPeriodDiDResults
    if hasattr(results, 'period_effects'):
        effects = {}
        se = {}

        for period, pe in results.period_effects.items():
            effects[period] = pe.effect
            se[period] = pe.se

        if pre_periods is None and hasattr(results, 'pre_periods'):
            pre_periods = results.pre_periods

        if post_periods is None and hasattr(results, 'post_periods'):
            post_periods = results.post_periods

        if periods is None:
            periods = post_periods

        return effects, se, periods, pre_periods, post_periods, reference_period

    # Handle CallawaySantAnnaResults (event study aggregation)
    if hasattr(results, 'event_study_effects'):
        effects = {}
        se = {}

        for rel_period, effect_data in results.event_study_effects.items():
            effects[rel_period] = effect_data['effect']
            se[rel_period] = effect_data['se']

        if periods is None:
            periods = sorted(effects.keys())

        # Reference period is typically -1 for event study
        if reference_period is None:
            reference_period = -1

        if pre_periods is None:
            pre_periods = [p for p in periods if p < 0]

        if post_periods is None:
            post_periods = [p for p in periods if p >= 0]

        return effects, se, periods, pre_periods, post_periods, reference_period

    raise TypeError(
        f"Cannot extract plot data from {type(results).__name__}. "
        "Expected MultiPeriodDiDResults, CallawaySantAnnaResults, or DataFrame."
    )


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

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    from scipy import stats as scipy_stats

    if not hasattr(results, 'group_time_effects'):
        raise TypeError("results must be a CallawaySantAnnaResults object")

    # Get groups to plot
    if groups is None:
        groups = sorted(set(g for g, t in results.group_time_effects.keys()))

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    critical_value = scipy_stats.norm.ppf(1 - alpha / 2)

    for i, group in enumerate(groups):
        # Get effects for this group
        group_effects = [
            (t, data) for (g, t), data in results.group_time_effects.items()
            if g == group
        ]
        group_effects.sort(key=lambda x: x[0])

        if not group_effects:
            continue

        times = [t for t, _ in group_effects]
        effects = [data['effect'] for _, data in group_effects]
        ses = [data['se'] for _, data in group_effects]

        yerr = [
            [e - (e - critical_value * s) for e, s in zip(effects, ses)],
            [(e + critical_value * s) - e for e, s in zip(effects, ses)]
        ]

        ax.errorbar(
            times, effects, yerr=yerr,
            label=f'Cohort {group}', color=colors[i],
            marker='o', capsize=3, linewidth=1.5
        )

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if show:
        plt.show()

    return ax
