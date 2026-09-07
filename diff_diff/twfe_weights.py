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

from diff_diff.linalg import solve_ols
from diff_diff.twfe_weights_results import (
    ATTGTWeightsResult,
    TWFEDecompositionResult,
)
from diff_diff.utils import within_transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from diff_diff.staggered_results import CallawaySantAnnaResults

__all__ = ["attgt_weights", "decompose_twfe_weights"]

_AGGREGATIONS = ("twfe", "overall", "simple")


def _is_never(values: np.ndarray) -> np.ndarray:
    """Boolean mask for never-treated cohort labels.

    diff-diff and R ``did`` have both used ``0`` and ``+inf`` as the
    never-treated sentinel over time; accept exactly those two and normalize
    to ``0``. Every OTHER non-finite label (NaN, ``-inf``) is an input error -
    see :func:`_validate_cohort_labels` - not a never-treated unit.
    """
    arr = np.asarray(values, dtype=float)
    return (arr == 0) | (arr == np.inf)


def _validate_cohort_labels(
    values: np.ndarray, *, unit_ids: Optional[np.ndarray] = None, what: str = "first_treat"
) -> None:
    """Reject NaN / ``-inf`` cohort labels instead of silently treating them as never-treated."""
    arr = np.asarray(values, dtype=float)
    bad = np.isnan(arr) | (arr == -np.inf)
    if bad.any():
        idx = np.flatnonzero(bad)[:5]
        who = [unit_ids[i] for i in idx] if unit_ids is not None else idx.tolist()
        raise ValueError(
            f"{what!r} contains NaN or -inf cohort label(s) for unit(s) {who!r}; "
            "never-treated units must be coded exactly 0 or +inf, and every "
            "other unit needs a finite first-treatment period"
        )


def _validate_time_labels(values: np.ndarray, *, what: str = "time") -> None:
    """Reject NaN / non-finite period labels before any grid is formed."""
    arr = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise ValueError(
            f"{what!r} contains {int(bad.sum())} non-finite or non-numeric period "
            f"label(s) (first at row {int(np.flatnonzero(bad)[0])}); every observation "
            "must carry a finite period"
        )


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


def _validate_unit_weights(
    w: np.ndarray, is_never: np.ndarray, *, require_control_mass: bool
) -> None:
    """Shared contract for unit-level sampling weights.

    Finite, non-negative, positive total, positive TREATED mass; positive
    never-treated mass only where the never-treated group enters the formula
    (``aggregation="twfe"`` and the decomposition) - ATT^O / ATT^simple are
    defined without one.
    """
    if not np.all(np.isfinite(w)):
        raise ValueError("unit weights must be finite; got NaN or infinite weight(s)")
    if (w < 0).any():
        idx = np.flatnonzero(w < 0)[:5].tolist()
        raise ValueError(
            f"unit weights must be non-negative; negative weight(s) at unit index {idx!r}"
        )
    if w.sum() <= 0:
        raise ValueError("unit weights sum to zero; cannot form cohort shares")
    if w[~is_never].sum() <= 0:
        raise ValueError(
            "the ever-treated units carry zero total weight; cannot form cohort shares"
        )
    if require_control_mass and w[is_never].sum() <= 0:
        raise ValueError(
            "the never-treated comparison group carries zero total weight, so "
            "every group-time contrast is undefined"
        )


def _cohort_masses(
    unit_cohorts: np.ndarray,
    grid: Dict[float, int],
    weights: Optional[np.ndarray],
    *,
    require_control_mass: bool = False,
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
    _validate_unit_weights(w, g_pos == 0, require_control_mass=require_control_mass)
    total = w.sum()

    treated = g_pos != 0
    treated_mass = w[treated].sum()

    cohorts = sorted({int(g) for g in g_pos if g != 0})
    p_all = {g: float(w[g_pos == g].sum() / total) for g in cohorts}
    p_treated = {g: float(w[g_pos == g].sum() / treated_mass) for g in cohorts}

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
    n_post_available: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """ATT^O weights: ``1[t >= g] * pbar_g / (maxT - g + 1)``.

    Not renormalized - the ``(maxT - g + 1)`` divisor already makes them sum
    to one over a complete post-treatment grid. ``n_post_available`` replaces
    that divisor with each cohort's number of AVAILABLE post periods when
    some post cells are structurally absent (``control_group="not_yet_treated"``
    runs out of comparison units) - what R ``aggte(type="group")`` averages
    over on such a fit.
    """
    if n_post_available is None:
        divisor = n_periods - groups + 1.0
    else:
        divisor = np.array([float(n_post_available[int(g)]) for g in groups])
    return (times >= groups).astype(float) * np.array([p_treated[int(g)] for g in groups]) / divisor


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
) -> Tuple[pd.DataFrame, Dict[Tuple[Any, Any], Optional[str]]]:
    """Extract the ``(g, t, att)`` table from a fitted CS result.

    Non-estimable cells (``skip_reason`` set, NaN effect) are left out of the
    table and reported in the returned ``{(g, t): skip_reason}`` map, so the
    caller can decide - per aggregation - whether the gap is structural, a
    harmless pre-period drop, or a hard error.
    """
    rows: List[Dict[str, Any]] = []
    skipped: Dict[Tuple[Any, Any], Optional[str]] = {}
    for (g, t), cell in results.group_time_effects.items():
        effect = cell.get("effect", np.nan)
        if cell.get("skip_reason") is not None or not np.isfinite(effect):
            skipped[(g, t)] = cell.get("skip_reason")
            continue
        rows.append({"group": g, "time": t, "att": float(effect)})
    if not rows:
        raise ValueError(
            "the fitted result has no estimable group-time cells; there is nothing to weight"
        )
    table = pd.DataFrame(rows).sort_values(["group", "time"]).reset_index(drop=True)
    return table, skipped


def _attgt_from_frame(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[Tuple[Any, Any], Optional[str]]]:
    """Extract ``(g, t, att)`` from a user-supplied ATT(g, t) frame.

    ``effect`` is preferred over ``att`` because that is the column
    ``CallawaySantAnnaResults.to_dataframe("group_time")`` emits - so the
    fallback consumes our own frame verbatim, including its ``skip_reason``
    column when present. Duplicate cells and non-finite ``group`` / ``time``
    labels are rejected; a non-finite effect is reported in the skip map, not
    silently kept (an ``inf`` ATT would otherwise propagate into
    ``implied_att``).
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
    groups = pd.to_numeric(frame["group"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    _validate_cohort_labels(groups, what="group")
    _validate_time_labels(frame["time"].to_numpy(), what="time")
    key = pd.MultiIndex.from_arrays([frame["group"].to_numpy(), frame["time"].to_numpy()])
    if key.duplicated().any():
        dupes = sorted({tuple(k) for k in key[key.duplicated()].tolist()})[:5]
        raise ValueError(
            f"ATT(g,t) frame has duplicated (group, time) cell(s) {dupes!r}; each "
            "cell must appear exactly once"
        )
    att = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    reasons = (
        frame["skip_reason"].tolist() if "skip_reason" in frame.columns else [None] * len(frame)
    )
    table = pd.DataFrame(
        {"group": frame["group"].to_numpy(), "time": frame["time"].to_numpy(), "att": att}
    )
    finite = np.isfinite(att)
    skipped: Dict[Tuple[Any, Any], Optional[str]] = {}
    for i in np.flatnonzero(~finite):
        reason = reasons[i]
        skipped[(table["group"].iat[i], table["time"].iat[i])] = (
            None
            if reason is None or (isinstance(reason, float) and np.isnan(reason))
            else str(reason)
        )
    table = table[finite]
    if table.empty:
        raise ValueError("ATT(g,t) frame has no finite effects to weight")
    _ = groups, times  # validated above; positional mapping happens in the caller
    return table.sort_values(["group", "time"]).reset_index(drop=True), skipped


def _unit_cohorts_from_frame(
    data: pd.DataFrame, unit: str, time: str, first_treat: str
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Collapse a long panel to one cohort label per unit."""
    for col in (unit, time, first_treat):
        if col not in data.columns:
            raise ValueError(f"column {col!r} not found in data")
    _validate_time_labels(data[time].to_numpy(), what=time)
    # dropna=False: a unit whose label is NaN in one period must fail the
    # invariance check, not slip through because nunique() skipped the NaN.
    per_unit = data.groupby(unit, sort=True)[first_treat].nunique(dropna=False)
    if (per_unit > 1).any():
        offenders = per_unit[per_unit > 1].index.tolist()[:5]
        raise ValueError(
            f"{first_treat!r} varies within unit(s) {offenders!r}; cohort "
            "membership must be time-invariant"
        )
    firsts = data.groupby(unit, sort=True)[first_treat].first()
    cohorts = firsts.to_numpy()
    _validate_cohort_labels(cohorts, unit_ids=firsts.index.to_numpy(), what=first_treat)
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
    from diff_diff.staggered_results import CallawaySantAnnaResults

    if not isinstance(results, CallawaySantAnnaResults):
        raise TypeError(
            "attgt_weights takes a CallawaySantAnna (or DMLDiD) fitted result, or an "
            f"ATT(g,t) DataFrame; got {type(results).__name__}"
        )
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
    # R's third restriction: xformla == ~1. The fit records its covariate
    # column names on the aggregation kit; a kit without the key predates that
    # bookkeeping (an old pickle) and can only be warned about. A missing kit
    # is left to _resolve_cs_inputs, whose error is the useful one.
    kit = getattr(results, "_aggregation_kit", None)
    if kit is None:
        return
    bookkeeping = getattr(kit, "bookkeeping", {}) or {}
    if "covariates" not in bookkeeping:
        warnings.warn(
            "this fit predates covariate bookkeeping, so attgt_weights cannot "
            "verify it used no covariates; the TWFE weight formula assumes an "
            "unadjusted regression (R twfe_weights requires xformla == ~1)",
            UserWarning,
            stacklevel=3,
        )
    elif bookkeeping["covariates"]:
        raise ValueError(
            f"aggregation='twfe' requires a fit without covariates, but this one "
            f"adjusted for {list(bookkeeping['covariates'])!r}. The TWFE weight "
            "formula describes the unadjusted regression (R's twfe_weights stops "
            "unless xformla == ~1); refit with covariates=None, or use "
            "decompose_twfe_weights(covariates=...) for the covariate-adjusted "
            "decomposition."
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
        ``group`` / ``time`` / ``effect`` (or ``att``) columns - the output of
        ``result.to_dataframe("group_time")`` is consumed verbatim, including
        its ``skip_reason`` column. On the frame path, ``data``, ``unit``,
        ``time`` and ``first_treat`` are required so cohort shares can be
        formed, and the caller is responsible for the fit having used no
        covariates under ``aggregation="twfe"`` (a frame carries no record of
        that; the fitted path checks it).
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
        weights, which take precedence. Must be finite and non-negative with
        positive treated mass (and positive never-treated mass for ``"twfe"``).

    Returns
    -------
    ATTGTWeightsResult
        Per-cell weights plus the negative-weight roll-ups.

    Raises
    ------
    ValueError
        On an unknown ``aggregation``; on a design the formula does not
        support (repeated cross-sections, unbalanced fallback, and - for
        ``aggregation="twfe"`` - a non-never-treated control group, a
        non-universal base period, or a covariate-adjusted fit); on NaN /
        ``-inf`` cohort labels, invalid weights, duplicated or non-finite
        cells; or on an INCOMPLETE grid: ``"twfe"`` needs every cohort x period
        cell, ``"overall"`` / ``"simple"`` every post-treatment cell.
    TypeError
        When ``results`` is neither a CallawaySantAnna-family result nor a
        DataFrame.

    Notes
    -----
    Two structural gaps are handled rather than raised, mirroring R:

    * A cohort with NO estimable post-treatment cell (typically one treated in
      the first observed period, which has no base period) is dropped from the
      table AND from the cohort masses with a warning - what
      ``did::pre_process_did`` does when it drops units already treated in the
      first period.
    * Under ``control_group="not_yet_treated"`` the last cohorts run out of
      comparison units, and CS marks those post cells ``zero_treated_control``.
      For ``"overall"`` / ``"simple"`` they are treated as structurally absent:
      ``"overall"`` divides each cohort by its number of AVAILABLE post periods
      and ``"simple"`` renormalizes over the available post cells - what
      R ``aggte()`` computes on such a fit. A warning names the cells.
      (``"twfe"`` requires a never-treated control group and never reaches
      this branch.)

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
    frame = results if isinstance(results, pd.DataFrame) else None
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
        assert frame is not None
        table, skipped = _attgt_from_frame(frame)
        cohorts, periods, _ = _unit_cohorts_from_frame(data, unit, time, first_treat)
        unit_weights = _resolve_frame_weights(weights, data, unit)
        source = "DataFrame"
        control_group = None
        base_period = None
        has_skip_reasons = "skip_reason" in frame.columns
    else:
        if supplied:
            raise ValueError(
                f"{sorted(supplied)!r} are only for the DataFrame fallback. A "
                "fitted CallawaySantAnnaResults already carries the cohort "
                "bookkeeping - drop them, or pass "
                "result.to_dataframe('group_time') as the first argument."
            )
        _guard_cs_design(results, aggregation)
        table, skipped = _attgt_from_cs(results)
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
        has_skip_reasons = True

    _validate_cohort_labels(cohorts, what="first_treat")
    grid = _positional_grid(periods)
    n_periods = len(grid)
    first_period_pos = 1

    # Positional mapping FIRST: the cohort universe the masses are formed over
    # must be known before the masses are formed.
    unit_g_pos = _to_positional_cohort(cohorts, grid)
    if not (unit_g_pos != 0).any():
        raise ValueError(
            "no ever-treated units found; cohort labels are all never-treated "
            "sentinels (0 or inf)"
        )
    g_pos = _to_positional_cohort(table["group"].to_numpy(), grid)
    t_pos = np.array([grid[float(t)] for t in table["time"].to_numpy()])
    post_mask = t_pos >= g_pos

    # --- whole-cohort exclusion (R did drops units treated in the first period)
    panel_cohorts = sorted({int(g) for g in unit_g_pos if g != 0})
    cohorts_with_post = {int(g) for g in g_pos[post_mask]}
    excluded = [g for g in panel_cohorts if g not in cohorts_with_post]
    if excluded:
        if not has_skip_reasons:
            not_structural = [g for g in excluded if g != first_period_pos]
            if not_structural:
                labels = [_label_for(grid, g) for g in not_structural]
                raise ValueError(
                    f"cohort(s) {labels!r} are present in data= but have no "
                    "post-treatment cell in the ATT(g,t) frame. A bare frame "
                    "cannot say why; pass result.to_dataframe('group_time') "
                    "verbatim (it carries skip_reason) or the fitted result itself."
                )
        n_units_excl = int(np.isin(unit_g_pos, excluded).sum())
        warnings.warn(
            f"cohort(s) {[_label_for(grid, g) for g in excluded]!r} ({n_units_excl} "
            "unit(s)) have no estimable post-treatment cell and were dropped from "
            "the weight table and the cohort shares, matching R did's drop of units "
            "already treated in the first observed period",
            UserWarning,
            stacklevel=2,
        )
        keep_units = ~np.isin(unit_g_pos, excluded)
        cohorts = cohorts[keep_units]
        unit_g_pos = unit_g_pos[keep_units]
        if unit_weights is not None:
            unit_weights = np.asarray(unit_weights, dtype=float)[keep_units]
        keep_rows = ~np.isin(g_pos, excluded)
        table = table[keep_rows].reset_index(drop=True)
        g_pos, t_pos, post_mask = g_pos[keep_rows], t_pos[keep_rows], post_mask[keep_rows]
        skipped = {k: v for k, v in skipped.items() if _pos_of(grid, k[0]) not in excluded}

    p_all, p_treated, e_dt, mean_e_dt = _cohort_masses(
        cohorts, grid, unit_weights, require_control_mass=(aggregation == "twfe")
    )

    # --- grid completeness
    present = set(zip(g_pos.tolist(), t_pos.tolist()))
    surviving = sorted(cohorts_with_post)
    if aggregation == "twfe":
        required = {(g, t) for g in surviving for t in range(1, n_periods + 1)}
    else:
        required = {(g, t) for g in surviving for t in range(g, n_periods + 1)}
    missing_cells = sorted(required - present)
    structurally_absent: List[Tuple[Any, Any]] = []
    if missing_cells:
        carve_out_ok = aggregation != "twfe" and control_group == "not_yet_treated"
        hard: List[Tuple[Tuple[Any, Any], Optional[str]]] = []
        for g, t in missing_cells:
            label = (_label_for(grid, g), _label_for(grid, t))
            reason = skipped.get(label)
            if carve_out_ok and reason == "zero_treated_control":
                structurally_absent.append(label)
            else:
                hard.append((label, reason))
        if hard:
            what = "cohort x period" if aggregation == "twfe" else "post-treatment"
            detail = ", ".join(
                f"{lab} [{reason or 'not in source table'}]" for lab, reason in hard[:6]
            )
            raise ValueError(
                f"aggregation={aggregation!r} needs the complete {what} grid, but "
                f"{len(hard)} required cell(s) are missing: {detail}. A weight table "
                "over a partial grid is not the named estimand. Fix the source fit "
                "(or pass the complete to_dataframe('group_time') output)."
            )
        warnings.warn(
            f"{len(structurally_absent)} post-treatment cell(s) {structurally_absent[:6]!r} "
            "have no not-yet-treated comparison units (skip_reason "
            "'zero_treated_control') and are treated as structurally absent: "
            f"aggregation={aggregation!r} averages over each cohort's AVAILABLE "
            "post periods, as R aggte() does on a not-yet-treated fit",
            UserWarning,
            stacklevel=2,
        )

    # Non-estimable PRE cells of surviving cohorts are the only drops left;
    # the CS estimands ignore pre cells, so they change nothing.
    dropped = sum(
        1
        for (g_lab, t_lab) in skipped
        if _pos_of(grid, g_lab) in cohorts_with_post and _pos_of(grid, t_lab) < _pos_of(grid, g_lab)
    )
    if dropped and aggregation != "twfe":
        warnings.warn(
            f"{dropped} pre-treatment group-time cell(s) had no estimable ATT(g,t) "
            f"and were excluded; aggregation={aggregation!r} places no weight on "
            "pre-treatment cells, so the weights are unaffected",
            UserWarning,
            stacklevel=2,
        )

    if aggregation == "twfe":
        weight_vec = _twfe_weight_vector(g_pos, t_pos, n_periods, p_all, e_dt, mean_e_dt)
    elif aggregation == "overall":
        n_post_available = None
        if structurally_absent:
            n_post_available = {g: int(((g_pos == g) & post_mask).sum()) for g in surviving}
        weight_vec = _overall_weight_vector(g_pos, t_pos, n_periods, p_treated, n_post_available)
    else:
        weight_vec = _simple_weight_vector(g_pos, t_pos, p_treated)

    out = pd.DataFrame(
        {
            "group": table["group"].to_numpy(),
            "time": table["time"].to_numpy(),
            "post": post_mask.astype(int),
            "weight": weight_vec,
            "att": table["att"].to_numpy(),
        }
    )

    negative = weight_vec < 0
    abs_total = float(np.abs(weight_vec).sum())
    negative_post = negative & post_mask
    abs_post_total = float(np.abs(weight_vec[post_mask]).sum())
    return ATTGTWeightsResult(
        weights=out,
        aggregation=aggregation,
        implied_att=float((weight_vec * table["att"].to_numpy()).sum()),
        n_negative=int(negative.sum()),
        negative_weight_share=(
            float(np.abs(weight_vec[negative]).sum() / abs_total) if abs_total > 0 else 0.0
        ),
        n_negative_post=int(negative_post.sum()),
        negative_post_weight_share=(
            float(np.abs(weight_vec[negative_post]).sum() / abs_post_total)
            if abs_post_total > 0
            else 0.0
        ),
        n_cells=len(out),
        source=source,
        control_group=control_group,
        base_period=base_period,
        n_dropped_cells=dropped,
    )


def _label_for(grid: Dict[float, int], pos: int) -> Any:
    """Positional period -> original label (inverse of ``_positional_grid``)."""
    for label, p in grid.items():
        if p == pos:
            return int(label) if float(label).is_integer() else label
    return pos


def _pos_of(grid: Dict[float, int], label: Any) -> int:
    """Original label -> positional period; never-treated sentinel stays 0."""
    try:
        value = float(label)
    except (TypeError, ValueError):
        return -1
    if value == 0 or value == np.inf:
        return 0
    return grid.get(value, -1)


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
        per_unit = data.groupby(unit, sort=True)[weights].nunique(dropna=False)
        if (per_unit > 1).any():
            offenders = per_unit[per_unit > 1].index.tolist()[:5]
            raise ValueError(
                f"weights column {weights!r} varies within unit(s) "
                f"{offenders!r}; sampling weights must be time-invariant"
            )
        return data.groupby(unit, sort=True)[weights].first().to_numpy(dtype=float)
    return np.asarray(weights, dtype=float)


# ---------------------------------------------------------------------------
# Panel plumbing for the decomposition
# ---------------------------------------------------------------------------


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """``stats::weighted.mean`` on flat arrays."""
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float((values * weights).sum() / total)


def _effective_sample_size(est_weights: np.ndarray, sampling_weights: np.ndarray) -> float:
    """``sum(w)^2 / sum(w^2)`` after normalizing both weight vectors."""
    sw = sampling_weights / sampling_weights.mean()
    ew = est_weights / _weighted_mean(est_weights, sw)
    denom = float((ew**2).sum())
    if denom == 0:
        return float("nan")
    return float(ew.sum() ** 2 / denom)


class _Panel:
    """Balanced panel reshaped to ``(n_units, n_periods)`` with positional time.

    Sorting by ``(unit, period)`` and reshaping means every ``(g, t)`` slice
    is a plain boolean row mask plus a column index, instead of repeated
    boolean scans over the long frame.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Sequence[str],
        weights: Optional[str],
    ) -> None:
        for col in (outcome, unit, time, first_treat, *covariates):
            if col not in data.columns:
                raise ValueError(f"column {col!r} not found in data")
        if weights is not None and weights not in data.columns:
            raise ValueError(f"weights column {weights!r} not found in data")

        _validate_time_labels(data[time].to_numpy(), what=time)
        frame = data.sort_values([unit, time]).reset_index(drop=True)
        units = frame[unit].to_numpy()
        periods = frame[time].to_numpy()
        self.unit_ids = np.asarray(sorted(pd.unique(units)))
        self.period_labels = np.asarray(sorted(pd.unique(periods)))
        n_units = len(self.unit_ids)
        n_periods = len(self.period_labels)
        if len(frame) != n_units * n_periods:
            raise ValueError(
                f"decompose_twfe_weights requires a balanced panel: got "
                f"{len(frame)} rows for {n_units} units x {n_periods} periods. "
                "Balance it first, e.g. diff_diff.balance_panel(data, unit=..., "
                "time=...)."
            )
        counts = frame.groupby(unit, sort=True)[time].nunique().to_numpy()
        if not np.all(counts == n_periods):
            raise ValueError(
                "decompose_twfe_weights requires a balanced panel: some units "
                "are missing periods"
            )

        self.grid = _positional_grid(self.period_labels)
        self.n_units = n_units
        self.n_periods = n_periods

        cohort_long = frame[first_treat].to_numpy()
        # dropna=False: a NaN label in one period must fail invariance, not
        # be skipped by nunique().
        per_unit = frame.groupby(unit, sort=True)[first_treat].nunique(dropna=False)
        if (per_unit > 1).any():
            offenders = per_unit[per_unit > 1].index.tolist()[:5]
            raise ValueError(
                f"{first_treat!r} varies within unit(s) {offenders!r}; cohort "
                "membership must be time-invariant"
            )
        raw_cohorts = cohort_long.reshape(n_units, n_periods)[:, 0]
        _validate_cohort_labels(raw_cohorts, unit_ids=self.unit_ids, what=first_treat)
        self.cohorts = _to_positional_cohort(raw_cohorts, self.grid)
        if not (self.cohorts == 0).any():
            raise ValueError(
                "decompose_twfe_weights needs never-treated units as the "
                "comparison group; none were found (matching R's twfeweights, "
                "which supports only a never-treated comparison)"
            )
        self.outcome = frame[outcome].to_numpy(dtype=float).reshape(n_units, n_periods)
        if weights is None:
            self.weights = np.ones((n_units, n_periods))
        else:
            block = frame[weights].to_numpy(dtype=float).reshape(n_units, n_periods)
            # Finite check FIRST: np.allclose is False on any NaN, which would
            # otherwise be misreported as "varies within unit".
            if not np.all(np.isfinite(block)):
                raise ValueError(
                    f"weights column {weights!r} must be finite; got NaN or infinite weight(s)"
                )
            if not np.allclose(block, block[:, :1]):
                raise ValueError(
                    f"weights column {weights!r} varies within unit; sampling "
                    "weights must be time-invariant"
                )
            _validate_unit_weights(block[:, 0], self.cohorts == 0, require_control_mass=True)
            self.weights = block
        self.covariates = tuple(covariates)
        if covariates:
            self.design = (
                frame[list(covariates)]
                .to_numpy(dtype=float)
                .reshape(n_units, n_periods, len(covariates))
            )
        else:
            self.design = np.zeros((n_units, n_periods, 0))

        periods_positional = np.arange(1, n_periods + 1)
        self.treated = (
            (periods_positional[None, :] >= self.cohorts[:, None]) & (self.cohorts[:, None] != 0)
        ).astype(float)

        # Two-way demeaning through the house helper (the same alternating
        # projections fixest::demean runs), on the sorted long frame so the
        # (unit, period) reshape afterwards is a plain view. The treatment
        # indicator is DERIVED from cohorts x positional periods, not an input
        # column, so it is synthesized here before the call. Both the RAW and
        # the demeaned covariate blocks are kept: the annihilation filter in
        # _fwl_residuals compares one against the other.
        demean_frame = pd.DataFrame(
            {"_unit": frame[unit].to_numpy(), "_time": frame[time].to_numpy()}
        )
        demean_frame["_treated"] = self.treated.reshape(-1)
        for j, name in enumerate(self.covariates):
            demean_frame[f"_x{j}"] = self.design[:, :, j].reshape(-1)
        row_weights = None if weights is None else self.weights.reshape(-1)
        demeaned = within_transform(
            demean_frame,
            ["_treated", *(f"_x{j}" for j in range(len(self.covariates)))],
            "_unit",
            "_time",
            weights=row_weights,
            suffix="_dm",
            tol=1e-12,
        )
        self.treated_demeaned = (
            demeaned["_treated_dm"].to_numpy(dtype=float).reshape(n_units, n_periods)
        )
        if self.covariates:
            self.design_demeaned = np.stack(
                [
                    demeaned[f"_x{j}_dm"].to_numpy(dtype=float).reshape(n_units, n_periods)
                    for j in range(len(self.covariates))
                ],
                axis=2,
            )
        else:
            self.design_demeaned = np.zeros((n_units, n_periods, 0))

    def covariate_block(
        self, names: Sequence[str], data: pd.DataFrame, unit: str, time: str
    ) -> np.ndarray:
        """Unit-mean-collapsed covariates, one column per name.

        R's ``twfe_cov_bal`` averages each balance covariate over ALL periods
        within a unit before comparing groups, so a time-varying covariate is
        summarized by its unit mean.
        """
        frame = data.sort_values([unit, time]).reset_index(drop=True)
        block = (
            frame[list(names)]
            .to_numpy(dtype=float)
            .reshape(self.n_units, self.n_periods, len(names))
        )
        return block.mean(axis=1)


def _fwl_residuals(panel: _Panel) -> Tuple[np.ndarray, float]:
    """Frisch-Waugh-Lovell residual of treatment on covariates, plus its scale.

    Double-demeans ``D`` and ``X``, projects the demeaned treatment on the
    demeaned covariates, and returns the residual. That residual IS the
    implicit weight the regression applies to each observation; ``alpha_den``
    is the normalization ``E[resid * Ddot]`` from R's
    ``combine_twfe_weights_gt``.

    With no covariates the projection is empty and the residual is just the
    double-demeaned treatment - which is exactly the branch R cannot run,
    because ``fixest::demean`` segfaults on the zero-column model matrix it
    builds for ``xformula = ~1``.
    """
    weights = panel.weights
    d_dot = panel.treated_demeaned
    x_dot = panel.design_demeaned

    flat_d = d_dot.reshape(-1)
    flat_w = weights.reshape(-1)
    # Explicit row count: with zero covariates the trailing axis is 0 and
    # numpy cannot infer a -1 against it. This is the same no-covariate branch
    # on which fixest::demean segfaults; here it simply has to be spelled out.
    flat_x = x_dot.reshape(panel.n_units * panel.n_periods, x_dot.shape[2])

    # Numerical hygiene: drop covariates that double-demeaning ANNIHILATED
    # before anything is projected on them. A time-invariant regressor leaves a
    # column of pure rounding noise (~1e-16 against a raw scale of ~1). Keeping
    # it is not catastrophic - the column lies in the FE span and is orthogonal
    # to the treatment residual, so on mpdta's `lpop` it moves the FWL residual
    # by ~2e-18 - but regressing on an exactly-zero column is meaningless, and
    # dropping it is what makes covariates=None and covariates=[<invariant>]
    # agree exactly. The test is scale-relative: a column counts as having no
    # within-variation when its demeaned norm is negligible NEXT TO ITS OWN raw
    # norm, which a rank test on the demeaned matrix alone cannot see (there,
    # 1e-16 is simply the largest pivot). The 1e-10 relative threshold is a
    # blunt instrument: a covariate with a large level and genuinely small
    # within-variation can trip it, which is why the warning says so.
    raw_scale = np.linalg.norm(
        panel.design.reshape(panel.n_units * panel.n_periods, x_dot.shape[2]),
        axis=0,
    )
    demeaned_scale = np.linalg.norm(flat_x, axis=0)
    annihilated = demeaned_scale <= 1e-10 * np.maximum(raw_scale, 1.0)
    if annihilated.any():
        names = [panel.covariates[j] for j in np.flatnonzero(annihilated)]
        warnings.warn(
            f"covariate(s) {names!r} have no within-unit-and-period variation "
            "(or within-variation below 1e-10 of their own level) and were "
            "dropped: two-way demeaning annihilates them, so they cannot affect "
            "a two-way fixed effects regression. If that is not intended, "
            "centre or rescale the covariate so its within-variation is not "
            "negligible next to its level",
            UserWarning,
            stacklevel=3,
        )
    flat_x = flat_x[:, ~annihilated]
    surviving = [name for name, drop in zip(panel.covariates, annihilated) if not drop]

    if flat_x.shape[1]:
        # House solver: WLS through the origin (R's lm(y ~ -1 + X, w)). On a
        # rank-deficient design it fits the maximal independent set, sets the
        # dropped coefficients to NaN (R-style) and computes the residual from
        # the identified ones - so the residual is the FWL residual we need
        # and the NaN positions name the collinear columns.
        gamma, resid, _ = solve_ols(
            flat_x,
            flat_d,
            weights=flat_w,
            return_vcov=False,
            rank_deficient_action="silent",
            column_names=list(surviving),
        )
        dropped = np.flatnonzero(np.isnan(gamma))
        if dropped.size:
            names = [surviving[j] for j in dropped]
            warnings.warn(
                f"dropped collinear covariate column(s) {names!r} after "
                "double-demeaning; they carry no within-variation independent of "
                "the others",
                UserWarning,
                stacklevel=3,
            )
        resid = np.asarray(resid, dtype=float)
    else:
        resid = flat_d
    alpha_den = _weighted_mean(resid * flat_d, flat_w)
    if not np.isfinite(alpha_den) or alpha_den == 0:
        raise ValueError(
            "the treatment indicator has no within-variation left after "
            "double-demeaning and covariate adjustment, so the TWFE "
            "coefficient is not identified"
        )
    return resid.reshape(panel.n_units, panel.n_periods), alpha_den


def _normalize_cell_weights(
    resid: np.ndarray, sampling_weights: np.ndarray, scale: float
) -> Tuple[np.ndarray, bool]:
    """Scale a cell's residuals to mean one, handling the 0/0 case.

    The implicit weights within a cell are ``resid / mean(resid)``. For the
    never-treated comparison group the residual is CONSTANT within a period
    (their treatment indicator is identically zero, so the double-demeaned
    value is ``-E_t[D] + mean_t E_t[D]``, the same for every control unit) -
    and for some cohort structures that constant is analytically ZERO. On
    sim_staggered (three equal cohorts at g in {0,3,4}, T=5) it vanishes
    exactly at t=3: ``-1/3 + 1/3``.

    That makes the ratio 0/0. The limit is unambiguous - a constant divided
    by its own mean is one - so return exactly one rather than dividing two
    rounding errors. R divides anyway, which is why its per-cell ATT(g,t) at
    such a cell carries ~1e-4 of noise; the aggregate is unaffected because
    the weights on the affected cells cancel exactly.

    Returns the weights and whether the degenerate branch was taken.
    """
    mean = _weighted_mean(resid, sampling_weights)
    spread = float(np.max(resid) - np.min(resid)) if resid.size else 0.0
    tol = 1e-12 * max(scale, 1.0)
    if abs(mean) <= tol:
        if spread <= tol:
            return np.ones_like(resid), True
        raise ValueError(
            "a group-time cell has comparison-group implicit weights that "
            "average to zero but are not constant, so the cell's ATT(g,t) is "
            "not identified. This usually means the panel has too little "
            "variation in treatment timing."
        )
    return resid / mean, False


def _decompose_fwl(
    panel: _Panel,
    base_period: str,
    balance_covariates: Sequence[str],
    balance_block: Optional[np.ndarray],
) -> Dict[str, Any]:
    """R ``implicit_twfe_weights``: TWFE as weighted ATT(g, t) + a remainder."""
    resid, alpha_den = _fwl_residuals(panel)
    weights = panel.weights
    flat_w = weights.reshape(-1)
    cohorts = panel.cohorts
    treated_cohorts = sorted({int(g) for g in cohorts if g != 0})
    if not treated_cohorts:
        raise ValueError("no ever-treated units found; nothing to decompose")
    control_mask = cohorts == 0
    if not control_mask.any():
        raise ValueError(
            "decompose_twfe_weights needs never-treated units as the "
            "comparison group; none were found (matching R's twfeweights, "
            "which supports only a never-treated comparison)"
        )
    if base_period == "gmin1" and 1 in treated_cohorts:
        raise ValueError(
            "base_period='gmin1' needs a period before each cohort's "
            "treatment, but a cohort is treated in the first period. Use "
            "base_period='first_period', or drop that cohort."
        )

    resid_scale = float(np.abs(resid).max())
    cells: List[Dict[str, Any]] = []
    balance_rows: List[Dict[str, Any]] = []
    degenerate_cells: List[Tuple[Any, Any]] = []
    for g in treated_cohorts:
        treated_mask = cohorts == g
        for t_pos in range(1, panel.n_periods + 1):
            col = t_pos - 1
            w_treated = weights[treated_mask, col]
            w_control = weights[control_mask, col]

            r_treated = resid[treated_mask, col]
            r_control = resid[control_mask, col]
            gpart_w, _ = _normalize_cell_weights(r_treated, w_treated, resid_scale)
            upart_w, degenerate = _normalize_cell_weights(r_control, w_control, resid_scale)
            if degenerate:
                degenerate_cells.append((panel.period_labels[g - 1], panel.period_labels[col]))

            y_t = panel.outcome[:, col]
            if base_period == "first_period":
                base = panel.outcome[:, 0]
            else:
                base = panel.outcome[:, g - 2]
            adjusted = y_t - base

            gpart = _weighted_mean(gpart_w * adjusted[treated_mask], w_treated)
            upart = _weighted_mean(upart_w * adjusted[control_mask], w_control)

            p_g = _weighted_mean(
                (cohorts == g).astype(float)[:, None].repeat(panel.n_periods, axis=1).reshape(-1),
                flat_w,
            )
            alpha_weight = (
                _weighted_mean(r_treated, w_treated) * p_g / (alpha_den * panel.n_periods)
            )

            remainder = 0.0
            if base_period == "gmin1":
                y_gmin1 = panel.outcome[:, g - 2]
                remainder = -_weighted_mean(upart_w * y_gmin1[control_mask], w_control)

            cells.append(
                {
                    "group": panel.period_labels[g - 1],
                    "time": panel.period_labels[col],
                    "post": int(t_pos >= g),
                    "att": gpart - upart,
                    "weight": alpha_weight,
                    "ess": _effective_sample_size(upart_w, w_control),
                    "remainder": remainder,
                }
            )
            if balance_block is not None:
                balance_rows.extend(
                    _balance_cell(
                        balance_block,
                        balance_covariates,
                        treated_mask,
                        control_mask,
                        gpart_w,
                        upart_w,
                        w_treated,
                        w_control,
                        group=panel.period_labels[g - 1],
                        time=panel.period_labels[col],
                        post=int(t_pos >= g),
                    )
                )

    if degenerate_cells:
        warnings.warn(
            f"{len(degenerate_cells)} group-time cell(s) {degenerate_cells[:4]!r}"
            " have comparison-group implicit weights that are constant and "
            "average to zero, so their ATT(g,t) is a 0/0 limit (taken as the "
            "unweighted contrast). The weights on these cells cancel in the "
            "aggregate, so `estimate` is unaffected; read the individual "
            "ATT(g,t) there with caution",
            UserWarning,
            stacklevel=3,
        )

    frame = pd.DataFrame(cells)
    weight_vec = frame["weight"].to_numpy()
    att_col = frame["att"].to_numpy()
    post_col = frame["post"].to_numpy().astype(bool)
    decomposition = float((weight_vec * att_col).sum())
    remainder_total = float((frame["remainder"].to_numpy() * weight_vec).sum())
    ess_col = frame["ess"].to_numpy()
    return {
        "cells": frame,
        "estimate": decomposition + remainder_total,
        "decomposition": decomposition,
        "remainder": remainder_total,
        "pretrend_bias": float((weight_vec[~post_col] * att_col[~post_col]).sum()),
        "post_only": float((weight_vec[post_col] * att_col[post_col]).sum()),
        # summary.decomposed_twfe: post cells only, on both factors
        "effective_sample_size": float(
            post_col.sum() * (weight_vec[post_col] * ess_col[post_col]).sum()
        ),
        "balance": pd.DataFrame(balance_rows) if balance_block is not None else None,
    }


# ---------------------------------------------------------------------------
# Balance statistics (Imbens & Rubin 2015, as implemented upstream)
# ---------------------------------------------------------------------------


def _weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``BMisc::weighted_ecdf``: knots and CDF heights.

    ``weights`` are normalized by their mean, the knots are the sorted unique
    values, and ``F(knot_j) = mean(w * (y <= knot_j))``.
    """
    w = weights / weights.mean()
    # Sort once and read cumulative mass at each unique-value boundary, rather
    # than rescanning the full vector per knot (the naive form is O(n * k), and
    # this runs per covariate x cohort x period). `np.unique` returns the
    # sorted knots, so a single searchsorted locates each boundary.
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(w[order])
    knots = np.unique(values)
    last = np.searchsorted(sorted_values, knots, side="right") - 1
    heights = cumulative[last] / len(values)
    return knots, heights


def _ecdf_eval(knots: np.ndarray, heights: np.ndarray, at: float) -> float:
    """Evaluate the step function from ``BMisc::make_dist``.

    ``approxfun(method="constant", yleft=0, yright=1, f=0)``: the value on
    ``[knot_i, knot_{i+1})`` is ``heights[i]``, zero below the first knot and
    one above the last.
    """
    if at < knots[0]:
        return 0.0
    if at > knots[-1]:
        return 1.0
    idx = int(np.searchsorted(knots, at, side="right") - 1)
    return float(heights[idx])


def _ecdf_quantile(knots: np.ndarray, heights: np.ndarray, prob: float) -> float:
    """``stats:::quantile.ecdf``: type-7 quantile of a reconstructed sample.

    R does NOT invert the step function directly. It rebuilds a pseudo-sample
    by repeating each knot ``diff(c(0, round(nobs * F)))`` times - where
    ``nobs`` is the number of KNOTS, not the number of observations - and then
    takes an ordinary type-7 quantile of that. Reproduced exactly, because the
    rounding makes the result differ from a direct inversion.
    """
    nobs = len(knots)
    counts = np.diff(np.concatenate([[0.0], np.round(nobs * heights)]))
    counts = np.maximum(counts, 0).astype(int)
    sample = np.repeat(knots, counts)
    if sample.size == 0:
        return float("nan")
    # R's default type-7 quantile.
    sample = np.sort(sample)
    h = (len(sample) - 1) * prob
    lo = int(np.floor(h))
    hi = min(lo + 1, len(sample) - 1)
    return float(sample[lo] + (h - lo) * (sample[hi] - sample[lo]))


def _pooled_sd(x: np.ndarray, treated: np.ndarray, sampling_weights: np.ndarray) -> float:
    """Pooled standard deviation across the treated and comparison groups."""
    sw = sampling_weights / sampling_weights.mean()

    def wvar(values: np.ndarray, w: np.ndarray) -> float:
        return _weighted_mean((values - _weighted_mean(values, w)) ** 2, w)

    var1 = wvar(x[treated == 1], sw[treated == 1])
    var0 = wvar(x[treated == 0], sw[treated == 0])
    n1 = sw[treated == 1].sum()
    n0 = sw[treated == 0].sum()
    if n1 + n0 - 2 <= 0:
        return float("nan")
    return float(np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)))


def _normalize_est_weights(
    est_weights: np.ndarray, treated: np.ndarray, sw: np.ndarray
) -> np.ndarray:
    """Scale estimation weights to mean one WITHIN each group, as R does."""
    out = np.array(est_weights, dtype=float, copy=True)
    for group in (0, 1):
        mask = treated == group
        if mask.any():
            out[mask] = out[mask] / _weighted_mean(out[mask], sw[mask])
    return out


def _log_ratio_sd(
    x: np.ndarray,
    treated: np.ndarray,
    est_weights: np.ndarray,
    sampling_weights: np.ndarray,
) -> float:
    """Log ratio of treated to comparison spread.

    Note: upstream scales each group's SD by ``sqrt(n - 1)`` before taking
    the ratio, which is not a conventional standard deviation. Preserved
    verbatim for parity - the quantity is only ever read as a relative
    balance statistic, and the extra factor largely cancels in the ratio.
    """
    sw = sampling_weights / sampling_weights.mean()
    ew = _normalize_est_weights(est_weights, treated, sw)

    def wvar(values: np.ndarray, e: np.ndarray, w: np.ndarray) -> float:
        scaled = values * e
        return _weighted_mean((scaled - _weighted_mean(scaled, w)) ** 2, w)

    var1 = wvar(x[treated == 1], ew[treated == 1], sw[treated == 1])
    var0 = wvar(x[treated == 0], ew[treated == 0], sw[treated == 0])
    n1 = sw[treated == 1].sum()
    n0 = sw[treated == 0].sum()
    sd1 = np.sqrt(max(n1 - 1, 0)) * np.sqrt(var1)
    sd0 = np.sqrt(max(n0 - 1, 0)) * np.sqrt(var0)
    if sd1 <= 0 or sd0 <= 0:
        return float("nan")
    return float(np.log(sd1) - np.log(sd0))


def _frac_treated_extreme(
    x: np.ndarray,
    treated: np.ndarray,
    est_weights: np.ndarray,
    sampling_weights: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Share of treated mass outside the comparison group's central range.

    A step function of a weighted empirical CDF, so a perturbation of order
    1e-12 can move one unit across a knot and shift the value by 1/n. Tests
    gate it with an absolute tolerance of ``1 / n_control`` rather than a
    relative one.
    """
    if len(np.unique(x)) < 3:
        return float("nan")
    sw = sampling_weights / sampling_weights.mean()
    ew = _normalize_est_weights(est_weights, treated, sw)

    control = treated == 0
    treat = treated == 1
    knots_u, heights_u = _weighted_ecdf(ew[control] * x[control], sw[control])
    upper = _ecdf_quantile(knots_u, heights_u, 1 - alpha / 2)
    lower = _ecdf_quantile(knots_u, heights_u, alpha / 2)
    knots_t, heights_t = _weighted_ecdf(ew[treat] * x[treat], sw[treat])
    return float(
        1.0 - _ecdf_eval(knots_t, heights_t, upper) + _ecdf_eval(knots_t, heights_t, lower)
    )


def _balance_cell(
    block: np.ndarray,
    names: Sequence[str],
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
    weights_treated: np.ndarray,
    weights_control: np.ndarray,
    sw_treated: np.ndarray,
    sw_control: np.ndarray,
    *,
    group: Any,
    time: Any,
    post: int,
) -> List[Dict[str, Any]]:
    """Per-covariate implicit-weight balance for one ``(g, t)`` cell."""
    both = treated_mask | control_mask
    indicator = np.where(treated_mask[both], 1, 0)
    est = np.empty(int(both.sum()))
    est[indicator == 1] = weights_treated
    est[indicator == 0] = weights_control
    sw_both = np.empty_like(est)
    sw_both[indicator == 1] = sw_treated
    sw_both[indicator == 0] = sw_control
    ones = np.ones_like(est)

    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(names):
        col = block[:, j]
        x_t = col[treated_mask]
        x_c = col[control_mask]
        x_both = col[both]
        unweighted_treated = _weighted_mean(x_t, sw_treated)
        unweighted_control = _weighted_mean(x_c, sw_control)
        weighted_treated = _weighted_mean(x_t * weights_treated, sw_treated)
        weighted_control = _weighted_mean(x_c * weights_control, sw_control)
        rows.append(
            {
                "group": group,
                "time": time,
                "post": post,
                "covariate": name,
                "unweighted_treated": unweighted_treated,
                "unweighted_control": unweighted_control,
                "unweighted_diff": unweighted_treated - unweighted_control,
                "weighted_treated": weighted_treated,
                "weighted_control": weighted_control,
                "weighted_diff": weighted_treated - weighted_control,
                "sd": _pooled_sd(x_both, indicator, sw_both),
                "unweighted_log_ratio_sd": _log_ratio_sd(x_both, indicator, ones, sw_both),
                "weighted_log_ratio_sd": _log_ratio_sd(x_both, indicator, est, sw_both),
                "unweighted_frac_extreme": _frac_treated_extreme(x_both, indicator, ones, sw_both),
                "weighted_frac_extreme": _frac_treated_extreme(x_both, indicator, est, sw_both),
            }
        )
    return rows


_METHODS = ("fwl",)
_BASE_PERIODS = ("first_period", "gmin1")


def decompose_twfe_weights(
    data: pd.DataFrame,
    *,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    method: str = "fwl",
    covariates: Optional[Sequence[str]] = None,
    base_period: str = "first_period",
    balance_covariates: Optional[Sequence[str]] = None,
    weights: Optional[str] = None,
) -> TWFEDecompositionResult:
    """Decompose a TWFE estimate into weighted group-time effects.

    Runs the regression, recovers the implicit weight it places on each
    ATT(g, t), and separates the part of the estimate that comes from
    PRE-treatment cells - i.e. from parallel-trends violations rather than
    from treatment.

    Takes the raw panel rather than a fitted result, because it re-estimates:
    it double-demeans treatment and covariates and forms its own group-time
    contrasts, so there is no ATT(g, t) table it could consume. Its companion
    :func:`attgt_weights` is the fitted-result surface, and the two are tied
    by an identity that holds when the fit used ``base_period="universal"``,
    ``control_group="never_treated"`` and no covariates::

        sum(attgt_weights(cs, aggregation="twfe").weights.eval("weight * att"))
            == decompose_twfe_weights(panel, ...).estimate

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel in long form.
    outcome, unit, time, first_treat : str
        Column names, matching :meth:`CallawaySantAnna.fit`. Never-treated
        units carry ``first_treat`` of ``0`` (or ``inf``).
    method : {"fwl"}, default "fwl"
        ``"fwl"`` recovers the Frisch-Waugh-Lovell implicit weights from the
        TWFE regression.
    covariates : sequence of str, optional
        Covariates the regression adjusts for. ``None`` runs the
        no-covariate decomposition.
    base_period : {"first_period", "gmin1"}, default "first_period"
        Which pre-period each cell is measured against. ``"gmin1"`` (the
        period before treatment) generates a non-zero ``remainder``.
    balance_covariates : sequence of str, optional
        Covariates to report implicit-weight balance for, readable afterwards
        via :meth:`TWFEDecompositionResult.covariate_balance`. Each is
        averaged over periods within unit before groups are compared, as
        upstream does.
    weights : str, optional
        Time-invariant sampling-weight column.

    Returns
    -------
    TWFEDecompositionResult

    Raises
    ------
    ValueError
        On an unknown ``method`` or ``base_period``; on an unbalanced panel,
        a missing never-treated group, or time-varying cohort labels; or when
        the treatment has no within-variation left after demeaning.

    Examples
    --------
    >>> import diff_diff  # doctest: +SKIP
    >>> dec = diff_diff.decompose_twfe_weights(  # doctest: +SKIP
    ...     panel, outcome="y", unit="id", time="t", first_treat="g",
    ...     covariates=["x"], balance_covariates=["x"],
    ... )
    >>> dec.pretrend_bias  # doctest: +SKIP
    >>> dec.covariate_balance()  # doctest: +SKIP
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)!r}, got {method!r}")
    if base_period not in _BASE_PERIODS:
        raise ValueError(
            f"base_period must be one of {list(_BASE_PERIODS)!r}, got " f"{base_period!r}"
        )

    covariate_names = tuple(covariates or ())
    balance_names = tuple(balance_covariates or ())
    panel = _Panel(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        covariates=covariate_names,
        weights=weights,
    )
    balance_block = (
        panel.covariate_block(balance_names, data, unit, time) if balance_names else None
    )

    payload = _decompose_fwl(panel, base_period, balance_names, balance_block)
    return TWFEDecompositionResult(
        cells=payload["cells"],
        method=method,
        estimate=payload["estimate"],
        decomposition=payload["decomposition"],
        remainder=payload["remainder"],
        pretrend_bias=payload["pretrend_bias"],
        post_only=payload["post_only"],
        base_period=base_period,
        covariates=covariate_names,
        effective_sample_size=payload["effective_sample_size"],
        n_units=panel.n_units,
        n_periods=panel.n_periods,
        balance=payload["balance"],
    )
