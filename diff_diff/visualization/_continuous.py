"""Continuous DiD visualization functions (dose-response curves)."""

import numbers
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from diff_diff.continuous_did_results import ContinuousDiDResults, DoseResponseCurve


def _coverage_label(alpha: float) -> str:
    """Exact coverage label: '97.5% CI' for alpha=0.025, '95% CI' for 0.05.

    Rounds to 6 decimals first so a float32-noised alpha (0.0500000007...)
    still reads 95 rather than 94.9999999255, then trims trailing zeros.
    """
    level = round(100.0 * (1.0 - float(alpha)), 6)
    return f"{level:g}% CI"


def plot_dose_response(
    results: Optional["ContinuousDiDResults"] = None,
    *,
    curve: Optional["DoseResponseCurve"] = None,
    data: Optional[pd.DataFrame] = None,
    target: str = "att",
    alpha: Optional[float] = None,
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
        ``conf_int_lower``, ``conf_int_upper``). Rows whose ``se`` is
        non-positive or non-finite carry no defined interval: their band is
        masked (with a warning) rather than drawn zero-width.
    target : str, default="att"
        Which dose-response curve: ``"att"`` or ``"acrt"``.
    alpha : float, optional
        Significance level for the reconstructed band on DataFrame-``se``
        input ONLY (default 0.05 there; must be strictly inside (0, 1)).
        Stored (``results=``/``curve=``) and explicit-CI intervals keep the
        level they were built at — an explicitly passed ``alpha`` warns and
        is ignored on those inputs. The band legend states the level where
        it is knowable (the requested alpha on the ``se`` branch,
        ``results.alpha`` on ``results=`` input) and the level-free "CI"
        otherwise.
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

    # ``alpha`` constructs an interval ONLY on the DataFrame-``se`` branch;
    # everywhere else the band's level is fixed by how it was built, so an
    # explicitly passed alpha is a no-op — warn instead of silently ignoring.
    se_branch = (
        data is not None
        and "se" in data.columns
        and not ("conf_int_lower" in data.columns and "conf_int_upper" in data.columns)
    )
    if alpha is not None and not se_branch:
        warnings.warn(
            "alpha= only applies to DataFrame input with an 'se' column; the "
            "displayed band keeps its stored/unknown confidence level.",
            UserWarning,
            stacklevel=2,
        )

    # Band legend: state the level only where it is knowable.
    resolved_alpha = 0.05 if alpha is None else alpha
    if se_branch:
        if not 0.0 < resolved_alpha < 1.0:
            raise ValueError(f"alpha must be strictly between 0 and 1, got {resolved_alpha}")
        ci_label = _coverage_label(resolved_alpha)
    elif results is not None:
        fit_alpha = getattr(results, "alpha", None)
        # numbers.Real (not (int, float)): np.float32 etc. are real numeric
        # fit alphas too; the (0, 1) range check excludes bools.
        if isinstance(fit_alpha, numbers.Real) and 0.0 < float(fit_alpha) < 1.0:
            ci_label = _coverage_label(fit_alpha)
        else:
            ci_label = "CI"
    else:
        # Bare curve= (DoseResponseCurve carries no alpha) or explicit-CI /
        # CI-less DataFrame input: no knowable level.
        ci_label = "CI"

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
            se = np.asarray(data["se"].values, dtype=float)
            invalid = ~(np.isfinite(se) & (se > 0))
            z = scipy_stats.norm.ppf(1 - resolved_alpha / 2)
            ci_lower = np.asarray(effects - z * se, dtype=float)
            ci_upper = np.asarray(effects + z * se, dtype=float)
            if invalid.any():
                # A zero/negative/non-finite SE carries no defined interval:
                # masking beats drawing a zero-width band asserting certainty.
                warnings.warn(
                    f"{int(invalid.sum())} row(s) with non-positive or "
                    "non-finite 'se' have no confidence band.",
                    UserWarning,
                    stacklevel=2,
                )
                ci_lower[invalid] = np.nan
                ci_upper[invalid] = np.nan
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
            ci_label=ci_label,
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
        ci_label=ci_label,
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
    ci_label,
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

    # Confidence band. Skip entirely when no CI row is finite - an
    # all-masked band would still register a stray legend entry.
    if ci_lower is not None and ci_upper is not None:
        finite_band = np.isfinite(np.asarray(ci_lower, dtype=float)) & np.isfinite(
            np.asarray(ci_upper, dtype=float)
        )
        if finite_band.any():
            band_color = ci_color or color
            ax.fill_between(
                dose_grid,
                ci_lower,
                ci_upper,
                alpha=0.15,
                color=band_color,
                label=ci_label,
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
    ci_label,
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

    # Confidence band. A NaN vertex splits a fill="toself" polygon into
    # independently-closed sub-polygons, so filter non-finite CI rows out
    # before building the trace (the _event_study.py precedent - CI VALUES
    # only, never the dose, which may be non-numeric) and skip the trace
    # entirely when nothing survives (else an empty stray legend entry).
    if ci_lower is not None and ci_upper is not None:
        lo = np.asarray(ci_lower, dtype=float)
        hi = np.asarray(ci_upper, dtype=float)
        keep = np.isfinite(lo) & np.isfinite(hi)
        if keep.any():
            band_color = ci_color or color
            dose_list = [d for d, k in zip(dose_grid, keep) if k]
            hi_list = list(hi[keep])
            lo_list = list(lo[keep])
            fig.add_trace(
                go.Scatter(
                    x=dose_list + dose_list[::-1],
                    y=hi_list + lo_list[::-1],
                    fill="toself",
                    fillcolor=_color_to_rgba(band_color, 0.15),
                    line=dict(color="rgba(0,0,0,0)"),
                    name=ci_label,
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
