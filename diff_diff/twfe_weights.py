"""Implicit TWFE weights on group-time average treatment effects.

A two-way fixed effects regression run on staggered-adoption data does not
estimate a simple average of the underlying ATT(g, t). It estimates a
*weighted* average, and some of those weights can be negative - so the
coefficient need not lie in the convex hull of the effects it summarizes.
:func:`attgt_weights` reports those weights, next to the weights the target
estimands ATT^O and ATT^simple would use. :func:`decompose_twfe_weights`
re-derives the regression from its building blocks and separates the part
driven by pre-treatment parallel-trends violations.

Distinct from :func:`diff_diff.twowayfeweights`, which implements the de
Chaisemartin & D'Haultfoeuille (2020) Theorem 1 decomposition: that one
weights ``(unit, time)`` cells, this one weights ATT(g, t) *parameters*.
Distinct also from :class:`diff_diff.BaconDecomposition`, which decomposes
TWFE into 2x2 DiD comparisons rather than into group-time effects.

Ported from the R package ``twfeweights`` (version 0.9.0) by Brantly
Callaway, released under the MIT License. The upstream notice is reproduced
in full, as its terms require::

    MIT License

    Copyright (c) 2023 Brantly Callaway

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Methodology: Baker, Callaway, Cunningham, Goodman-Bacon & Sant'Anna (2025),
"Difference-in-Differences Designs: A Practitioner's Guide"
(arXiv:2503.13323); Callaway & Sant'Anna (2021) for the ATT^O / ATT^simple
weights.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from diff_diff.twfe_weights_results import ATTGTWeightsResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from diff_diff.staggered_results import CallawaySantAnnaResults

__all__ = ["attgt_weights"]

_AGGREGATIONS = ("twfe", "overall", "simple")


def _is_never(values: np.ndarray) -> np.ndarray:
    """Boolean mask for never-treated cohort labels.

    diff-diff and R ``did`` have both used ``0`` and ``+inf`` as the
    never-treated sentinel over time; accept either and normalize to ``0``.
    """
    arr = np.asarray(values, dtype=float)
    return ~np.isfinite(arr) | (arr == 0)


def _positional_grid(
    time_periods: Sequence[Any],
) -> Dict[float, int]:
    """Map ordered period labels onto ``1..T``.

    R computes ``(maxT - g + 1) / length(tlist)`` directly on the raw period
    labels, which is only correct when those labels are consecutive integers.
    Working in positional time makes the same expression correct on gapped or
    non-integer grids, and is bit-identical when the grid IS consecutive
    (mpdta's 2003..2007 maps to 1..5 and both give 4/5 for g = 2004).
    Recorded as a deviation in the methodology registry.
    """
    ordered = sorted({float(t) for t in time_periods})
    return {t: i + 1 for i, t in enumerate(ordered)}


def _to_positional_cohort(cohorts: np.ndarray, grid: Dict[float, int]) -> np.ndarray:
    """Cohort labels -> positional time; never-treated stays 0.

    Mirrors ``BMisc::orig2t``, which leaves the never-treated sentinel alone
    under positional rescaling.
    """
    out = np.zeros(len(cohorts), dtype=float)
    never = _is_never(cohorts)
    for i, (g, is_never) in enumerate(zip(cohorts, never)):
        if is_never:
            continue
        key = float(g)
        if key not in grid:
            raise ValueError(
                f"cohort label {g!r} is not one of the observed time periods "
                f"{sorted(grid)!r}; cannot place it on the period grid"
            )
        out[i] = grid[key]
    return out


def _cohort_masses(
    unit_cohorts: np.ndarray,
    grid: Dict[float, int],
    weights: Optional[np.ndarray],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], float]:
    """Cohort shares and treated-share-by-period, all in positional time.

    Returns
    -------
    p_all : {positional g: share of ALL units in cohort g}
        R's ``pg2`` - the denominator is every unit, never-treated included.
        Used by the TWFE weights.
    p_treated : {positional g: share of EVER-TREATED units in cohort g}
        R's ``pg``. Used by the ATT^O / ATT^simple weights.
    e_dt : {positional t: weighted share of units treated by t}
        R's ``Edt(t)``.
    mean_e_dt : float
        R's ``mEdt`` - the average of ``e_dt`` over the period grid.
    """
    g_pos = _to_positional_cohort(unit_cohorts, grid)
    w = np.ones(len(g_pos)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(g_pos):
        raise ValueError(f"weights has length {len(w)} but there are {len(g_pos)} units")
    total = w.sum()
    if total <= 0:
        raise ValueError("unit weights sum to zero; cannot form cohort shares")

    treated = g_pos != 0
    treated_mass = w[treated].sum()

    cohorts = sorted({int(g) for g in g_pos if g != 0})
    p_all = {g: float(w[g_pos == g].sum() / total) for g in cohorts}
    if treated_mass > 0:
        p_treated = {g: float(w[g_pos == g].sum() / treated_mass) for g in cohorts}
    else:  # pragma: no cover - guarded upstream by the never-treated check
        p_treated = {g: 0.0 for g in cohorts}

    periods = sorted(grid.values())
    e_dt = {t: float(w[treated & (g_pos <= t)].sum() / total) for t in periods}
    mean_e_dt = float(np.mean([e_dt[t] for t in periods]))
    return p_all, p_treated, e_dt, mean_e_dt


def _twfe_weight_vector(
    groups: np.ndarray,
    times: np.ndarray,
    n_periods: int,
    p_all: Dict[int, float],
    e_dt: Dict[int, float],
    mean_e_dt: float,
) -> np.ndarray:
    """Weights a static TWFE regression places on each ATT(g, t).

    ``h(g,t) = 1[t >= g] - (maxT - g + 1)/T - E_t[D] + mean_t E_t[D]``
    ``num(g,t) = h(g,t) * p_g``, normalized by the sum over post cells.

    All arguments are in positional time, so ``maxT == n_periods``.
    """
    h = (
        (times >= groups).astype(float)
        - (n_periods - groups + 1.0) / n_periods
        - np.array([e_dt[int(t)] for t in times])
        + mean_e_dt
    )
    num = h * np.array([p_all[int(g)] for g in groups])
    post = times >= groups
    denom = num[post].sum()
    if denom == 0:
        raise ValueError(
            "TWFE weight normalization is degenerate (post-treatment weights "
            "sum to zero); the regression has no identifying variation"
        )
    return num / denom


def _overall_weight_vector(
    groups: np.ndarray,
    times: np.ndarray,
    n_periods: int,
    p_treated: Dict[int, float],
) -> np.ndarray:
    """ATT^O weights: ``1[t >= g] * pbar_g / (maxT - g + 1)``.

    Not renormalized - the ``(maxT - g + 1)`` divisor already makes them sum
    to one over a complete post-treatment grid.
    """
    return (
        (times >= groups).astype(float)
        * np.array([p_treated[int(g)] for g in groups])
        / (n_periods - groups + 1.0)
    )


def _simple_weight_vector(
    groups: np.ndarray,
    times: np.ndarray,
    p_treated: Dict[int, float],
) -> np.ndarray:
    """ATT^simple weights: ``1[t >= g] * pbar_g``, normalized to sum to one."""
    raw = (times >= groups).astype(float) * np.array([p_treated[int(g)] for g in groups])
    total = raw.sum()
    if total == 0:
        raise ValueError(
            "ATT^simple weight normalization is degenerate (no post-treatment "
            "cells carry weight)"
        )
    return raw / total


def _attgt_from_cs(
    results: "CallawaySantAnnaResults",
) -> Tuple[pd.DataFrame, int]:
    """Extract the ``(g, t, att)`` table from a fitted CS result.

    Non-estimable cells (``skip_reason`` set, NaN effect) are dropped and
    counted, so a partially-estimable fit still produces weights over the
    cells that exist rather than propagating NaN through every aggregate.
    """
    rows: List[Dict[str, Any]] = []
    dropped = 0
    for (g, t), cell in results.group_time_effects.items():
        effect = cell.get("effect", np.nan)
        if cell.get("skip_reason") is not None or not np.isfinite(effect):
            dropped += 1
            continue
        rows.append({"group": g, "time": t, "att": float(effect)})
    if not rows:
        raise ValueError(
            "the fitted result has no estimable group-time cells; there is " "nothing to weight"
        )
    table = pd.DataFrame(rows).sort_values(["group", "time"]).reset_index(drop=True)
    return table, dropped


def _attgt_from_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Extract ``(g, t, att)`` from a user-supplied ATT(g, t) frame.

    ``effect`` is preferred over ``att`` because that is the column
    ``CallawaySantAnnaResults.to_dataframe("group_time")`` emits - so the
    fallback consumes our own frame verbatim.
    """
    missing = {"group", "time"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"ATT(g,t) frame is missing required column(s) {sorted(missing)!r}; "
            "expected 'group', 'time', and one of 'effect' / 'att'"
        )
    for candidate in ("effect", "att"):
        if candidate in frame.columns:
            value_col = candidate
            break
    else:
        raise ValueError(
            "ATT(g,t) frame must carry an 'effect' or 'att' column; got " f"{list(frame.columns)!r}"
        )
    table = pd.DataFrame(
        {
            "group": frame["group"].to_numpy(),
            "time": frame["time"].to_numpy(),
            "att": pd.to_numeric(frame[value_col], errors="coerce").to_numpy(),
        }
    )
    dropped = int(table["att"].isna().sum())
    table = table.dropna(subset=["att"])
    if table.empty:
        raise ValueError("ATT(g,t) frame has no finite effects to weight")
    return table.sort_values(["group", "time"]).reset_index(drop=True), dropped


def _unit_cohorts_from_frame(
    data: pd.DataFrame, unit: str, time: str, first_treat: str
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Collapse a long panel to one cohort label per unit."""
    for col in (unit, time, first_treat):
        if col not in data.columns:
            raise ValueError(f"column {col!r} not found in data")
    per_unit = data.groupby(unit, sort=True)[first_treat].nunique()
    if (per_unit > 1).any():
        offenders = per_unit[per_unit > 1].index.tolist()[:5]
        raise ValueError(
            f"{first_treat!r} varies within unit(s) {offenders!r}; cohort "
            "membership must be time-invariant"
        )
    cohorts = data.groupby(unit, sort=True)[first_treat].first().to_numpy()
    periods = np.asarray(sorted(data[time].unique()))
    return cohorts, periods, None


def _resolve_cs_inputs(
    results: "CallawaySantAnnaResults",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read cohort labels (and survey weights) off a fitted CS result.

    The aggregation kit is package-internal, but it is the same channel
    ``CallawaySantAnnaResults._aggregate_compute`` already uses - so this is
    an established in-package coupling rather than a new one. When the kit is
    absent (an old pickle), the caller is pointed at the ``data=`` fallback.
    """
    kit = getattr(results, "_aggregation_kit", None)
    if kit is None:
        raise ValueError(
            "this CallawaySantAnnaResults carries no aggregation bookkeeping "
            "(it may have been unpickled from an older version), so cohort "
            "shares cannot be recovered from it. Pass the panel explicitly:\n"
            "    attgt_weights(result.to_dataframe('group_time'), data=panel,\n"
            "                  unit=..., time=..., first_treat=...)"
        )
    bookkeeping = getattr(kit, "bookkeeping", {}) or {}
    cohorts = bookkeeping.get("unit_cohorts")
    if cohorts is None:
        raise ValueError(
            "aggregation bookkeeping does not carry 'unit_cohorts'; pass the "
            "panel explicitly via data=/unit=/time=/first_treat="
        )
    weights = bookkeeping.get("survey_weights")
    return np.asarray(cohorts), (None if weights is None else np.asarray(weights, dtype=float))


def _guard_cs_design(results: "CallawaySantAnnaResults", aggregation: str) -> None:
    """Reject fits whose design breaks the weight formulas.

    These are hard errors rather than warnings: a silently wrong weight table
    is worse than no weight table, and every one of these has a concrete fix.
    """
    if not getattr(results, "panel", True):
        raise ValueError(
            "attgt_weights requires a panel fit: E_t[D] and the cohort shares "
            "average over a fixed set of units, which repeated cross-sections "
            "do not provide. Refit with panel=True."
        )
    if getattr(results, "used_rc_on_unbalanced_panel", False):
        raise ValueError(
            "this fit fell back to repeated-cross-section estimation on an "
            "unbalanced panel, so the cohort shares are not comparable across "
            "periods. Balance the panel (diff_diff.balance_panel) and refit."
        )
    if aggregation != "twfe":
        return
    control_group = getattr(results, "control_group", None)
    if control_group not in (None, "never_treated"):
        raise ValueError(
            f"aggregation='twfe' requires control_group='never_treated', got "
            f"{control_group!r}. The TWFE weight formula is derived against a "
            "never-treated comparison group (matching R's twfe_weights, which "
            "raises the same restriction)."
        )
    base_period = getattr(results, "base_period", None)
    if base_period not in (None, "universal"):
        raise ValueError(
            f"aggregation='twfe' requires base_period='universal', got "
            f"{base_period!r}. The formula needs the complete cohort x period "
            "grid, including the pre-treatment cells that a varying base does "
            "not report. Refit with base_period='universal'."
        )


def attgt_weights(
    results: Union["CallawaySantAnnaResults", pd.DataFrame],
    *,
    aggregation: str = "twfe",
    data: Optional[pd.DataFrame] = None,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    first_treat: Optional[str] = None,
    weights: Optional[Union[str, np.ndarray]] = None,
) -> ATTGTWeightsResult:
    """Weights an estimand places on each group-time effect ATT(g, t).

    Three estimands are available. ``"twfe"`` gives the weights implied by a
    static two-way fixed effects regression - the ones that can go negative.
    ``"overall"`` and ``"simple"`` give the weights of the Callaway &
    Sant'Anna (2021) target parameters ATT^O and ATT^simple, which are
    non-negative by construction. Comparing them shows how far the regression
    is from the estimand you meant to report.

    Parameters
    ----------
    results : CallawaySantAnnaResults or pd.DataFrame
        A fitted Callaway & Sant'Anna result (preferred), or a frame with
        ``group`` / ``time`` / ``effect`` (or ``att``) columns. On the frame
        path, ``data``, ``unit``, ``time`` and ``first_treat`` are required
        so cohort shares can be formed.
    aggregation : {"twfe", "overall", "simple"}, default "twfe"
        Which estimand's weights to report.
    data : pd.DataFrame, optional
        Balanced panel backing the ATT(g, t) frame. Only for the fallback
        path; passing it alongside a fitted result raises.
    unit, time, first_treat : str, optional
        Column names in ``data``. Required together with ``data``.
    weights : str or array-like, optional
        Unit-level sampling weights (R's ``w=``): a column name in ``data``,
        or one value per unit. Rejected when the fit already carries survey
        weights, which take precedence.

    Returns
    -------
    ATTGTWeightsResult
        Per-cell weights plus the negative-weight roll-up.

    Raises
    ------
    ValueError
        On an unknown ``aggregation``; on a design the formula does not
        support (repeated cross-sections, unbalanced fallback, and - for
        ``aggregation="twfe"`` - a non-never-treated control group or a
        non-universal base period); or on an incomplete fallback spec.

    Notes
    -----
    R's ``keep_untreated=TRUE`` is not exposed. It synthesizes ``G = 0`` rows
    with ``attgt = 0`` to mirror an internal vector layout; those rows are
    excluded from every normalization and contribute exactly zero, so the
    argument does not affect any number.

    Examples
    --------
    >>> import diff_diff  # doctest: +SKIP
    >>> cs = diff_diff.CallawaySantAnna(base_period="universal")  # doctest: +SKIP
    >>> res = cs.fit(df, outcome="y", unit="id", time="t",
    ...              first_treat="g")  # doctest: +SKIP
    >>> w = diff_diff.attgt_weights(res, aggregation="twfe")  # doctest: +SKIP
    >>> print(w.summary())  # doctest: +SKIP
    """
    if aggregation not in _AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {list(_AGGREGATIONS)!r}, got " f"{aggregation!r}"
        )

    frame_path = isinstance(results, pd.DataFrame)
    fallback_args = {"data": data, "unit": unit, "time": time, "first_treat": first_treat}
    supplied = {k: v for k, v in fallback_args.items() if v is not None}

    if frame_path:
        if len(supplied) != 4:
            missing = sorted(set(fallback_args) - set(supplied))
            raise ValueError(
                "the DataFrame path needs the panel too, so cohort shares can "
                f"be formed; missing {missing!r}. Call it as:\n"
                "    attgt_weights(gt_frame, data=panel, unit='id', "
                "time='t', first_treat='g')"
            )
        assert data is not None and unit is not None
        assert time is not None and first_treat is not None
        table, dropped = _attgt_from_frame(results)
        cohorts, periods, _ = _unit_cohorts_from_frame(data, unit, time, first_treat)
        unit_weights = _resolve_frame_weights(weights, data, unit)
        source = "DataFrame"
        control_group = None
        base_period = None
    else:
        if supplied:
            raise ValueError(
                f"{sorted(supplied)!r} are only for the DataFrame fallback. A "
                "fitted CallawaySantAnnaResults already carries the cohort "
                "bookkeeping - drop them, or pass "
                "result.to_dataframe('group_time') as the first argument."
            )
        _guard_cs_design(results, aggregation)
        table, dropped = _attgt_from_cs(results)
        cohorts, survey_weights = _resolve_cs_inputs(results)
        if survey_weights is not None and weights is not None:
            raise ValueError(
                "this fit already carries survey weights; passing weights= as "
                "well is ambiguous. Drop weights= to use the fit's own."
            )
        if weights is not None and not isinstance(weights, str):
            unit_weights = np.asarray(weights, dtype=float)
        elif isinstance(weights, str):
            raise ValueError(
                "weights= may only name a column on the DataFrame path; pass "
                "an array of per-unit weights instead"
            )
        else:
            unit_weights = survey_weights
        periods = np.asarray(results.time_periods)
        source = "CallawaySantAnnaResults"
        control_group = getattr(results, "control_group", None)
        base_period = getattr(results, "base_period", None)

    if dropped:
        warnings.warn(
            f"{dropped} group-time cell(s) had no estimable ATT(g,t) and were "
            "excluded from the weight table; the reported weights renormalize "
            "over the remaining cells",
            UserWarning,
            stacklevel=2,
        )

    grid = _positional_grid(periods)
    n_periods = len(grid)
    p_all, p_treated, e_dt, mean_e_dt = _cohort_masses(cohorts, grid, unit_weights)
    if not p_treated:
        raise ValueError(
            "no ever-treated units found; cohort labels are all never-treated "
            "sentinels (0 or inf)"
        )

    g_pos = _to_positional_cohort(table["group"].to_numpy(), grid)
    t_pos = np.array([grid[float(t)] for t in table["time"].to_numpy()])

    if aggregation == "twfe":
        weight_vec = _twfe_weight_vector(g_pos, t_pos, n_periods, p_all, e_dt, mean_e_dt)
    elif aggregation == "overall":
        weight_vec = _overall_weight_vector(g_pos, t_pos, n_periods, p_treated)
    else:
        weight_vec = _simple_weight_vector(g_pos, t_pos, p_treated)

    out = pd.DataFrame(
        {
            "group": table["group"].to_numpy(),
            "time": table["time"].to_numpy(),
            "post": (t_pos >= g_pos).astype(int),
            "weight": weight_vec,
            "att": table["att"].to_numpy(),
        }
    )

    negative = weight_vec < 0
    abs_total = float(np.abs(weight_vec).sum())
    return ATTGTWeightsResult(
        weights=out,
        aggregation=aggregation,
        implied_att=float((weight_vec * table["att"].to_numpy()).sum()),
        n_negative=int(negative.sum()),
        negative_weight_share=(
            float(np.abs(weight_vec[negative]).sum() / abs_total) if abs_total > 0 else 0.0
        ),
        n_cells=len(out),
        source=source,
        control_group=control_group,
        base_period=base_period,
        n_dropped_cells=dropped,
    )


def _resolve_frame_weights(
    weights: Optional[Union[str, np.ndarray]],
    data: pd.DataFrame,
    unit: str,
) -> Optional[np.ndarray]:
    """Turn ``weights=`` into one value per unit, or None."""
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights not in data.columns:
            raise ValueError(f"weights column {weights!r} not found in data")
        per_unit = data.groupby(unit, sort=True)[weights].nunique()
        if (per_unit > 1).any():
            offenders = per_unit[per_unit > 1].index.tolist()[:5]
            raise ValueError(
                f"weights column {weights!r} varies within unit(s) "
                f"{offenders!r}; sampling weights must be time-invariant"
            )
        return data.groupby(unit, sort=True)[weights].first().to_numpy(dtype=float)
    return np.asarray(weights, dtype=float)
