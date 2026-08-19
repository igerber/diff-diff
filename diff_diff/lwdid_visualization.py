"""Visualization methods for LWDiD results.

Provides plotting functions for cohort trends, event studies,
sensitivity analysis, and bootstrap distributions.

Requires matplotlib (optional dependency). If not installed,
raises ImportError with installation instructions.

Note
----
All plot functions return a matplotlib Figure object without closing it.
In batch/loop usage, call ``plt.close(fig)`` after saving or displaying
each figure to avoid memory accumulation.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for LWDiD visualization. "
            "Install with: pip install matplotlib"
        )


def plot_cohort_trends(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    cohort: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    show_ci: bool = True,
    ax=None,
):
    """Plot pre/post outcome trajectories by treatment group (or cohort).

    Shows average outcomes over time for treated vs control groups,
    with optional confidence intervals.
    """
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute group means by time
    # Identify ever-treated units
    treated_units = data.loc[data[treatment] == 1, unit].unique()
    data = data.copy()
    data["_ever_treated"] = data[unit].isin(treated_units).astype(int)

    # Group averages
    group_means = (
        data.groupby([time, "_ever_treated"])[outcome].agg(["mean", "std", "count"]).reset_index()
    )
    group_means["se"] = group_means["std"] / np.sqrt(group_means["count"])

    for grp, label, color in [(1, "Treated", "steelblue"), (0, "Control", "coral")]:
        gdf = group_means[group_means["_ever_treated"] == grp]
        ax.plot(gdf[time], gdf["mean"], "o-", label=label, color=color)
        if show_ci:
            ax.fill_between(
                gdf[time],
                gdf["mean"] - 1.96 * gdf["se"],
                gdf["mean"] + 1.96 * gdf["se"],
                alpha=0.15,
                color=color,
            )

    # Mark treatment onset
    treated_times = data.loc[data[treatment] == 1, time]
    if len(treated_times) > 0:
        first_treat = treated_times.min()
        # Datetime/Period/string time columns cannot take `- 0.5` (raw
        # TypeError pre-fix); draw the marker AT the onset for
        # non-numeric scales, offset by half a period for numeric ones.
        if pd.api.types.is_numeric_dtype(data[time]):
            onset_x = first_treat - 0.5
        else:
            onset_x = first_treat
        ax.axvline(onset_x, color="gray", linestyle="--", alpha=0.7, label="Treatment onset")

    ax.set_xlabel("Time")
    ax.set_ylabel(outcome)
    ax.set_title(title or "LWDiD: Cohort Trends")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_event_study(
    results: Any,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    ax=None,
):
    """Plot event-study estimates from a fitted LWDiD result.

    Consumes the unified post-fit event-study surface
    (``results.event_study_effects``, keyed by event time relative to
    first treatment, with ``reference_periods`` anchored at zero).
    Both staggered and common-timing fits populate this surface at fit
    time; a result without one raises a clear error instead of plotting.

    Parameters
    ----------
    results : LWDiDResults
        Fitted result whose event-study surface is populated.
    title : str or None
        Plot title.
    figsize : tuple
        Figure size (ignored when ``ax`` is supplied).
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Raises
    ------
    ValueError
        If ``results`` carries no populated event-study surface (e.g. a
        degenerate fit with no estimable post period).
    """
    effects = getattr(results, "event_study_effects", None)
    if not effects:
        raise ValueError(
            "plot_event_study requires a fitted result with a populated "
            "event-study surface (results.event_study_effects); this fit "
            "does not carry one (no estimable post-period effects)."
        )

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    reference_periods = set(getattr(results, "reference_periods", ()) or ())
    event_times = sorted(set(effects) | reference_periods)
    atts = []
    err_lo = []
    err_hi = []
    for r in event_times:
        if r in reference_periods and r not in effects:
            atts.append(0.0)
            err_lo.append(0.0)
            err_hi.append(0.0)
            continue
        row = effects[r]
        att_r = row.get("effect", np.nan)
        atts.append(att_r)
        # Round-21 review: render the FITTED interval endpoints (t-based
        # per-row df, fitted alpha, cband/bootstrap intervals), not a
        # fabricated normal-theory +/-1.96*SE. Preference: simultaneous
        # cband when present, else the stored conf_int. House rule: OMIT
        # the interval when unavailable/non-finite (a zero-length bar
        # would render an inference-unavailable effect as infinitely
        # precise); reference periods keep their deliberate zero bars.
        interval = row.get("cband_conf_int") or row.get("conf_int")
        if interval is not None and np.all(np.isfinite(interval)) and np.isfinite(att_r):
            err_lo.append(att_r - float(interval[0]))
            err_hi.append(float(interval[1]) - att_r)
        else:
            err_lo.append(np.nan)
            err_hi.append(np.nan)

    yerr = np.vstack([err_lo, err_hi])
    ax.errorbar(event_times, atts, yerr=yerr, fmt="o-", capsize=3, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Event time")
    ax.set_ylabel("ATT")
    ax.set_title(title or "LWDiD: Event-Study Effects")
    ax.grid(True, alpha=0.3)

    return fig


def plot_sensitivity(
    sensitivity_result,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    ax=None,
):
    """Plot sensitivity analysis results.

    Shows ATT estimates across different specifications with
    confidence bands, highlighting the baseline estimate.
    """
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    specs = sensitivity_result.specifications
    x = range(len(specs))
    atts = [s.att for s in specs]
    labels = [s.label for s in specs]

    # Round-21 review: render each specification's FITTED interval
    # endpoints when stored; failed/legacy specs show the point only
    # (no fabricated normal-theory interval).
    err_lo = []
    err_hi = []
    for s_ in specs:
        interval = getattr(s_, "conf_int", None)
        if interval is not None and np.all(np.isfinite(interval)) and np.isfinite(s_.att):
            err_lo.append(s_.att - float(interval[0]))
            err_hi.append(float(interval[1]) - s_.att)
        else:
            err_lo.append(np.nan)
            err_hi.append(np.nan)
    yerr = np.vstack([err_lo, err_hi])
    ax.errorbar(x, atts, yerr=yerr, fmt="o", capsize=3, color="steelblue")
    ax.axhline(
        sensitivity_result.baseline_att,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Baseline ATT",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ATT")
    ax.set_title(
        title or f"Sensitivity Analysis (robustness: {sensitivity_result.robustness_level})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_bootstrap_distribution(
    t_stats: np.ndarray,
    t_observed: float,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
    ax=None,
):
    """Plot bootstrap t-statistic distribution with observed value."""
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.hist(t_stats, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(t_observed, color="red", linewidth=2, label=f"t_obs = {t_observed:.3f}")
    ax.axvline(-t_observed, color="red", linewidth=2, linestyle="--", alpha=0.5)
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    ax.set_title(title or "Wild Cluster Bootstrap Distribution")
    ax.legend()

    return fig
