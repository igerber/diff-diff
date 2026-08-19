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
    """Plot pre/post outcome trajectories by treatment group or cohort.

    Without ``cohort=``, shows average outcomes over time for the
    ever-treated vs control groups. With ``cohort=`` (round-23 review:
    the parameter was previously accepted but silently ignored), one
    trajectory is drawn PER treated cohort (never-treated encodings
    0/NaN form the control line) with a per-cohort onset marker.
    Optional confidence bands in both modes.
    """
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    data = data.copy()
    if cohort is not None:
        cohort_by_unit = data.drop_duplicates(subset=[unit], keep="first").set_index(unit)[cohort]
        never_mask = cohort_by_unit.isna() | (cohort_by_unit == 0)
        data["_plot_group"] = data[unit].map(
            {
                u: ("Control" if never_mask[u] else f"Cohort {cohort_by_unit[u]}")
                for u in cohort_by_unit.index
            }
        )
        group_order = sorted({g for g in data["_plot_group"].unique() if g != "Control"}) + (
            ["Control"] if (data["_plot_group"] == "Control").any() else []
        )
    else:
        treated_units = data.loc[data[treatment] == 1, unit].unique()
        data["_plot_group"] = np.where(data[unit].isin(treated_units), "Treated", "Control")
        group_order = ["Treated", "Control"]

    group_means = (
        data.groupby([time, "_plot_group"])[outcome].agg(["mean", "std", "count"]).reset_index()
    )
    group_means["se"] = group_means["std"] / np.sqrt(group_means["count"])

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["steelblue", "coral"])
    for i, label in enumerate(group_order):
        gdf = group_means[group_means["_plot_group"] == label]
        color = "coral" if label == "Control" else colors[i % len(colors)]
        ax.plot(gdf[time], gdf["mean"], "o-", label=label, color=color)
        if show_ci:
            ax.fill_between(
                gdf[time],
                gdf["mean"] - 1.96 * gdf["se"],
                gdf["mean"] + 1.96 * gdf["se"],
                alpha=0.15,
                color=color,
            )

    # Onset markers: one per cohort when cohort= is given, else the
    # common onset. Datetime/Period/string time columns cannot take
    # `- 0.5` (raw TypeError pre-fix); draw AT the onset for non-numeric
    # scales, offset by half a period for numeric ones.
    def _onset_x(value):
        return value - 0.5 if pd.api.types.is_numeric_dtype(data[time]) else value

    if cohort is not None:
        onsets = sorted({v for v in cohort_by_unit.dropna().unique() if not (pd.isna(v) or v == 0)})
        for i, g in enumerate(onsets):
            ax.axvline(
                _onset_x(g),
                color="gray",
                linestyle="--",
                alpha=0.7,
                label="Cohort onsets" if i == 0 else None,
            )
    else:
        treated_times = data.loc[data[treatment] == 1, time]
        if len(treated_times) > 0:
            ax.axvline(
                _onset_x(treated_times.min()),
                color="gray",
                linestyle="--",
                alpha=0.7,
                label="Treatment onset",
            )

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
