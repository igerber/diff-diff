"""
Visualization functions for difference-in-differences analysis.

Provides event study plots and other diagnostic visualizations.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from diff_diff.bacon import BaconDecompositionResults
    from diff_diff.honest_did import HonestDiDResults, SensitivityResults
    from diff_diff.power import PowerResults, SimulationPowerResults
    from diff_diff.pretrends import PreTrendsPowerCurve, PreTrendsPowerResults
    from diff_diff.results import MultiPeriodDiDResults
    from diff_diff.staggered import CallawaySantAnnaResults
    from diff_diff.sun_abraham import SunAbrahamResults

# Type alias for results that can be plotted
PlottableResults = Union[
    "MultiPeriodDiDResults",
    "CallawaySantAnnaResults",
    "SunAbrahamResults",
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

    # Normalize effects to reference period if specified
    if reference_period is not None and reference_period in effects:
        ref_effect = effects[reference_period]
        if np.isfinite(ref_effect):
            effects = {p: e - ref_effect for p, e in effects.items()}

    plot_data = []
    for period in periods:
        effect = effects.get(period, np.nan)
        std_err = se.get(period, np.nan)

        # Skip entries with NaN effect, but allow NaN SE (will plot without error bars)
        if np.isnan(effect):
            continue

        # Compute CI only if SE is finite
        if np.isfinite(std_err):
            ci_lower = effect - critical_value * std_err
            ci_upper = effect + critical_value * std_err
        else:
            ci_lower = np.nan
            ci_upper = np.nan

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

    # Plot error bars (only for entries with finite CI)
    has_ci = df['ci_lower'].notna() & df['ci_upper'].notna()
    if has_ci.any():
        df_with_ci = df[has_ci]
        x_with_ci = [period_to_x[p] for p in df_with_ci['period']]
        yerr = [
            df_with_ci['effect'] - df_with_ci['ci_lower'],
            df_with_ci['ci_upper'] - df_with_ci['effect']
        ]
        ax.errorbar(
            x_with_ci, df_with_ci['effect'], yerr=yerr,
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
    if hasattr(results, 'event_study_effects') and results.event_study_effects is not None:
        effects = {}
        se = {}

        for rel_period, effect_data in results.event_study_effects.items():
            effects[rel_period] = effect_data['effect']
            se[rel_period] = effect_data['se']

        if periods is None:
            periods = sorted(effects.keys())

        # Reference period is typically -1 for event study
        if reference_period is None:
            # Detect reference period from n_groups=0 marker (normalization constraint)
            # This handles anticipation > 0 where reference is at e = -1 - anticipation
            for period, effect_data in results.event_study_effects.items():
                if effect_data.get('n_groups', 1) == 0:
                    reference_period = period
                    break
            # Fallback to -1 if no marker found (backward compatibility)
            if reference_period is None:
                reference_period = -1

        if pre_periods is None:
            pre_periods = [p for p in periods if p < 0]

        if post_periods is None:
            post_periods = [p for p in periods if p >= 0]

        return effects, se, periods, pre_periods, post_periods, reference_period

    raise TypeError(
        f"Cannot extract plot data from {type(results).__name__}. "
        "Expected MultiPeriodDiDResults, CallawaySantAnnaResults, "
        "SunAbrahamResults, or DataFrame."
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

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    M = sensitivity_results.M_values
    bounds_arr = np.array(sensitivity_results.bounds)
    ci_arr = np.array(sensitivity_results.robust_cis)

    # Plot original estimate
    ax.axhline(
        y=sensitivity_results.original_estimate,
        color=original_color,
        linestyle='-',
        linewidth=1.5,
        label='Original estimate',
        alpha=0.7
    )

    # Plot zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Plot identified set bounds
    if show_bounds:
        ax.fill_between(
            M, bounds_arr[:, 0], bounds_arr[:, 1],
            alpha=bounds_alpha,
            color=bounds_color,
            label='Identified set'
        )

    # Plot confidence intervals
    if show_ci:
        ax.plot(
            M, ci_arr[:, 0],
            color=ci_color,
            linewidth=ci_linewidth,
            label='Robust CI'
        )
        ax.plot(
            M, ci_arr[:, 1],
            color=ci_color,
            linewidth=ci_linewidth
        )

    # Plot breakdown line
    if breakdown_line and sensitivity_results.breakdown_M is not None:
        ax.axvline(
            x=sensitivity_results.breakdown_M,
            color=breakdown_color,
            linestyle=':',
            linewidth=2,
            label=f'Breakdown (M={sensitivity_results.breakdown_M:.2f})'
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return ax


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

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.

    Notes
    -----
    This function requires the HonestDiDResults to have been computed
    with event_study_bounds. If only a scalar bound was computed,
    use plot_sensitivity() instead.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    from scipy import stats as scipy_stats

    # Get original results for standard CIs
    original_results = honest_results.original_results
    if original_results is None:
        raise ValueError(
            "HonestDiDResults must have original_results to plot event study"
        )

    # Extract data from original results
    if hasattr(original_results, 'period_effects'):
        # MultiPeriodDiDResults
        effects_dict = {
            p: pe.effect for p, pe in original_results.period_effects.items()
        }
        se_dict = {
            p: pe.se for p, pe in original_results.period_effects.items()
        }
        if periods is None:
            periods = list(original_results.period_effects.keys())
    elif hasattr(original_results, 'event_study_effects'):
        # CallawaySantAnnaResults
        effects_dict = {
            t: data['effect']
            for t, data in original_results.event_study_effects.items()
        }
        se_dict = {
            t: data['se']
            for t, data in original_results.event_study_effects.items()
        }
        if periods is None:
            periods = sorted(original_results.event_study_effects.keys())
    else:
        raise TypeError("Cannot extract event study data from original_results")

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute CIs
    alpha = honest_results.alpha
    z = scipy_stats.norm.ppf(1 - alpha / 2)

    x_vals = list(range(len(periods)))

    effects = [effects_dict[p] for p in periods]
    original_ci_lower = [effects_dict[p] - z * se_dict[p] for p in periods]
    original_ci_upper = [effects_dict[p] + z * se_dict[p] for p in periods]

    # Get honest bounds if available for each period
    if honest_results.event_study_bounds:
        honest_ci_lower = [
            honest_results.event_study_bounds[p]['ci_lb']
            for p in periods
        ]
        honest_ci_upper = [
            honest_results.event_study_bounds[p]['ci_ub']
            for p in periods
        ]
    else:
        # Use scalar bounds applied to all periods
        honest_ci_lower = [honest_results.ci_lb] * len(periods)
        honest_ci_upper = [honest_results.ci_ub] * len(periods)

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Plot original CIs (thinner, background)
    yerr_orig = [
        [e - lower for e, lower in zip(effects, original_ci_lower)],
        [u - e for e, u in zip(effects, original_ci_upper)]
    ]
    ax.errorbar(
        x_vals, effects, yerr=yerr_orig,
        fmt='none', color=original_color, capsize=capsize - 1,
        linewidth=1, alpha=0.6, label='Standard CI'
    )

    # Plot honest CIs (thicker, foreground)
    yerr_honest = [
        [e - lower for e, lower in zip(effects, honest_ci_lower)],
        [u - e for e, u in zip(effects, honest_ci_upper)]
    ]
    ax.errorbar(
        x_vals, effects, yerr=yerr_honest,
        fmt='none', color=honest_color, capsize=capsize,
        linewidth=2, label=f'Honest CI (M={honest_results.M:.2f})'
    )

    # Plot point estimates
    for i, (x, effect, period) in enumerate(zip(x_vals, effects, periods)):
        is_ref = period == reference_period
        if is_ref:
            ax.plot(x, effect, marker=marker, markersize=markersize,
                    markerfacecolor='white', markeredgecolor=honest_color,
                    markeredgewidth=2, zorder=3)
        else:
            ax.plot(x, effect, marker=marker, markersize=markersize,
                    color=honest_color, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(p) for p in periods])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if show:
        plt.show()

    return ax


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

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

    Examples
    --------
    Scatter plot (default):

    >>> from diff_diff import bacon_decompose, plot_bacon
    >>> results = bacon_decompose(data, outcome='y', unit='id',
    ...                           time='t', first_treat='first_treat')
    >>> plot_bacon(results)

    Bar chart of weights by type:

    >>> plot_bacon(results, plot_type='bar')

    Customized scatter plot:

    >>> plot_bacon(results,
    ...            colors={'treated_vs_never': 'green',
    ...                    'earlier_vs_later': 'blue',
    ...                    'later_vs_earlier': 'red'},
    ...            title='My Bacon Decomposition')

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

    The bar chart provides a quick summary of how much weight falls on
    each comparison type, which is useful for assessing the severity
    of potential TWFE bias.

    See Also
    --------
    bacon_decompose : Perform the decomposition
    BaconDecomposition : Class-based interface
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    # Default colors
    if colors is None:
        colors = {
            "treated_vs_never": "#22c55e",    # Green - clean comparison
            "earlier_vs_later": "#3b82f6",    # Blue - valid comparison
            "later_vs_earlier": "#ef4444",    # Red - forbidden comparison
        }

    # Default titles
    if title is None:
        if plot_type == "scatter":
            title = "Goodman-Bacon Decomposition"
        else:
            title = "TWFE Weight by Comparison Type"

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if plot_type == "scatter":
        _plot_bacon_scatter(
            ax, results, colors, marker, markersize, alpha,
            show_weighted_avg, show_twfe_line, xlabel, ylabel, title
        )
    elif plot_type == "bar":
        _plot_bacon_bar(ax, results, colors, alpha, ylabel, title)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'scatter' or 'bar'.")

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
            estimates, weights,
            c=colors[ctype],
            label=labels[ctype],
            marker=marker,
            s=markersize,
            alpha=alpha,
            edgecolors='white',
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
                    linestyle='--',
                    alpha=0.5,
                    linewidth=1.5,
                )

    # Show TWFE estimate line
    if show_twfe_line:
        ax.axvline(
            x=results.twfe_estimate,
            color='black',
            linestyle='-',
            linewidth=2,
            label=f'TWFE = {results.twfe_estimate:.4f}',
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add zero line
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)


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
        edgecolor='white',
        linewidth=1,
    )

    # Add percentage labels on bars
    for bar, weight in zip(bars, bar_weights):
        if weight > 0.01:  # Only label if > 1%
            height = bar.get_height()
            ax.annotate(
                f'{weight:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
            )

    # Add weighted average effect annotations
    effects = results.effect_by_type()
    for bar, ctype in zip(bars, type_order):
        effect = effects[ctype]
        if effect is not None and weights[ctype] > 0.01:
            ax.annotate(
                f'β = {effect:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                ha='center',
                va='center',
                fontsize=9,
                color='white',
                fontweight='bold',
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)

    # Add horizontal line at total weight = 1
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Add TWFE estimate as text
    ax.text(
        0.98, 0.98,
        f'TWFE = {results.twfe_estimate:.4f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )


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

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    # Extract data from results if provided
    if results is not None:
        if isinstance(results, pd.DataFrame):
            # DataFrame input
            if "effect_size" in results.columns:
                effect_sizes = results["effect_size"].tolist()
                plot_type = "effect"
            elif "sample_size" in results.columns:
                effect_sizes = results["sample_size"].tolist()
                plot_type = "sample"
            else:
                raise ValueError(
                    "DataFrame must have 'effect_size' or 'sample_size' column"
                )
            powers = results["power"].tolist()
        elif hasattr(results, "effect_sizes") and hasattr(results, "powers"):
            # SimulationPowerResults
            effect_sizes = results.effect_sizes
            powers = results.powers
            if mde is None and hasattr(results, "true_effect"):
                # Mark true effect on plot
                mde = results.true_effect
        elif hasattr(results, "mde"):
            # PowerResults - create curve data
            raise ValueError(
                "PowerResults should be used to get mde value, not as direct input. "
                "Use PowerAnalysis.power_curve() to generate curve data."
            )
        else:
            raise TypeError(
                f"Cannot extract power curve data from {type(results).__name__}"
            )
    elif effect_sizes is None or powers is None:
        raise ValueError(
            "Must provide either 'results' or both 'effect_sizes' and 'powers'"
        )

    # Default titles and labels
    if title is None:
        if plot_type == "effect":
            title = "Power Curve"
        else:
            title = "Power vs Sample Size"

    if xlabel is None:
        if plot_type == "effect":
            xlabel = "Effect Size"
        else:
            xlabel = "Sample Size"

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot power curve
    ax.plot(
        effect_sizes, powers,
        color=color,
        linewidth=linewidth,
        label="Power"
    )

    # Add target power line
    if show_target_line:
        ax.axhline(
            y=target_power,
            color=target_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Target power ({target_power:.0%})"
        )

    # Add MDE line
    if show_mde_line and mde is not None:
        ax.axvline(
            x=mde,
            color=mde_color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"MDE = {mde:.3f}"
        )

        # Mark intersection point
        # Find power at MDE
        if mde in effect_sizes:
            idx = effect_sizes.index(mde)
            power_at_mde = powers[idx]
        else:
            # Interpolate
            effect_arr = np.array(effect_sizes)
            power_arr = np.array(powers)
            if effect_arr.min() <= mde <= effect_arr.max():
                power_at_mde = np.interp(mde, effect_arr, power_arr)
            else:
                power_at_mde = None

        if power_at_mde is not None:
            ax.scatter(
                [mde], [power_at_mde],
                color=mde_color,
                s=50,
                zorder=5
            )

    # Configure axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Y-axis from 0 to 1
    ax.set_ylim(0, 1.05)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    if show_grid:
        ax.grid(True, alpha=0.3)

    ax.legend(loc="lower right")

    fig.tight_layout()

    if show:
        plt.show()

    return ax


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

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

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

    From manual data:

    >>> plot_pretrends_power(
    ...     M_values=[0, 0.5, 1, 1.5, 2],
    ...     powers=[0.05, 0.3, 0.6, 0.85, 0.95],
    ...     mdv=1.2,
    ...     target_power=0.80
    ... )

    Notes
    -----
    The power curve shows how likely you are to reject the null hypothesis
    of parallel trends given a true violation of magnitude M. Key points:

    1. **At M=0**: Power equals alpha (size of the test).
    2. **At MDV**: Power equals target power (default 80%).
    3. **Beyond MDV**: Power increases toward 100%.

    A steep power curve indicates a sensitive pre-trends test. A flat curve
    indicates the test has limited ability to detect violations, suggesting
    you should use HonestDiD sensitivity analysis for robust inference.

    See Also
    --------
    PreTrendsPower : Main class for pre-trends power analysis
    plot_sensitivity : Plot HonestDiD sensitivity analysis
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

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
            # Single PreTrendsPowerResults - create a simple plot
            if mdv is None:
                mdv = results.mdv
            # Create minimal curve around MDV
            if np.isfinite(mdv):
                M_values = [0, mdv * 0.5, mdv, mdv * 1.5, mdv * 2]
            else:
                M_values = [0, 1, 2, 3, 4]
            # We don't have the actual powers, so we need to create a placeholder
            # Just show MDV marker
            powers = None
        else:
            raise TypeError(
                f"Cannot extract power curve data from {type(results).__name__}"
            )
    elif M_values is None or powers is None:
        raise ValueError(
            "Must provide either 'results' or both 'M_values' and 'powers'"
        )

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot power curve if we have powers
    if powers is not None:
        ax.plot(
            M_values, powers,
            color=color,
            linewidth=linewidth,
            label="Power"
        )

    # Add target power line
    if show_target_line:
        ax.axhline(
            y=target_power,
            color=target_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Target power ({target_power:.0%})"
        )

    # Add MDV line
    if show_mdv_line and mdv is not None and np.isfinite(mdv):
        ax.axvline(
            x=mdv,
            color=mdv_color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"MDV = {mdv:.3f}"
        )

        # Mark intersection point if we have powers
        if powers is not None:
            # Find power at MDV (interpolate)
            M_arr = np.array(M_values)
            power_arr = np.array(powers)
            if M_arr.min() <= mdv <= M_arr.max():
                power_at_mdv = np.interp(mdv, M_arr, power_arr)
                ax.scatter(
                    [mdv], [power_at_mdv],
                    color=mdv_color,
                    s=50,
                    zorder=5
                )

    # Configure axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Y-axis from 0 to 1
    ax.set_ylim(0, 1.05)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    if show_grid:
        ax.grid(True, alpha=0.3)

    ax.legend(loc="lower right")

    fig.tight_layout()

    if show:
        plt.show()

    return ax
