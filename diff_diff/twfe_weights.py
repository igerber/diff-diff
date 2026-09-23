"""Implicit TWFE weights on group-time average treatment effects.

A two-way fixed effects regression run on staggered-adoption data does not
estimate a simple average of the underlying ATT(g, t). It estimates a
*weighted* average, and some of those weights can be negative - so the
coefficient need not lie in the convex hull of the effects it summarizes.
:func:`attgt_weights` reports those weights, next to the weights the target
estimands ATT^O and ATT^simple would use. :func:`decompose_twfe_weights`
re-derives the regression from its building blocks and separates the sample
contribution of the pre-treatment cells, which can reflect differential
pre-trends or sampling variation.

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

_TYPES = ATTGTWeightsResult.LEVELS

# ``skip_reason`` values CS / DMLDiD emit for cells they structurally could not
# form: no base period under the anticipation window (``missing_period``), no
# comparison units (``zero_treated_control``), or zero survey mass
# (``zero_weight_mass``). A cohort whose only gaps carry these reasons is
# handled the way ``aggregate()`` handles it - dropped when it has no estimable
# post cell, otherwise averaged over the cells it has; any other gap fails closed.
_STRUCTURAL_REASONS = frozenset({"missing_period", "zero_treated_control", "zero_weight_mass"})


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


def _validate_time_labels(values: np.ndarray, *, what: str = "time") -> np.ndarray:
    """Validate period labels and return them as the canonical numeric key.

    Every downstream step - sorting, reshaping, the positional grid, cohort
    mapping - MUST use this one key. Using the raw column instead lets a
    numeric-string label ("10" sorts before "2" lexicographically) desynchronize
    the sort order from the grid, silently rebuilding a different panel.
    """
    arr = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise ValueError(
            f"{what!r} contains {int(bad.sum())} non-finite or non-numeric period "
            f"label(s) (first at row {int(np.flatnonzero(bad)[0])}); every observation "
            "must carry a finite period"
        )
    return arr


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
    (``type="twfe"`` and the decomposition) - ATT^O / ATT^simple are
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
    p_treated: Dict[int, float],
    post_mask: np.ndarray,
    n_post_available: Dict[int, int],
) -> np.ndarray:
    """ATT^O weights: ``1[label(t) >= label(g) - a] * pbar_g / n_post(g)``.

    Not renormalized - the per-cohort divisor ``n_post(g)`` is each cohort's
    number of AVAILABLE post cells under the anticipation window ``a`` (raw
    time units), so the weights already sum to one (the ``pbar_g`` sum to one
    over cohorts). On a complete grid that is ``maxT - post_start(g) + 1``,
    which reduces to R's ``(maxT - g + 1)`` only when ``a == 0``; counting the
    available cells rather than writing it analytically is what makes the
    window and the structurally-absent cells (no comparison units, no base
    period, zero survey mass) come out right.
    """
    divisor = np.array([float(n_post_available[int(g)]) for g in groups])
    return post_mask.astype(float) * np.array([p_treated[int(g)] for g in groups]) / divisor


def _simple_weight_vector(
    groups: np.ndarray,
    p_treated: Dict[int, float],
    post_mask: np.ndarray,
) -> np.ndarray:
    """ATT^simple weights: ``1[label(t) >= label(g) - a] * pbar_g``, normalized.

    ``a`` is the anticipation window in raw time units; the vector
    self-normalizes, so no per-cohort divisor is needed.
    """
    raw = post_mask.astype(float) * np.array([p_treated[int(g)] for g in groups])
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
    caller can decide - per estimand type - whether the gap is structural, a
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
    labels are rejected - duplicates are detected on the NUMERIC keys, so two
    spellings of one period (``3`` and ``"3"``) are the same cell, exactly as
    the positional grid treats them; a non-finite effect is reported in the
    skip map, not silently kept (an ``inf`` ATT would otherwise propagate
    into ``implied_att``).
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
    key = pd.MultiIndex.from_arrays([groups, times])
    if key.duplicated().any():
        dupes = sorted({tuple(k) for k in key[key.duplicated()].tolist()})[:5]
        raise ValueError(
            f"ATT(g,t) frame has duplicated (group, time) cell(s) {dupes!r}; each "
            "cell must appear exactly once (labels compare as numbers, so '3' and "
            "3.0 name the same cell)"
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
    key = _validate_time_labels(data[time].to_numpy(), what=time)
    # Exactly one observation per unit-period, on a rectangular grid: the cohort
    # shares and E_t[D] assume the same units in every period, exactly as the
    # fitted path does.
    cells = pd.MultiIndex.from_arrays([data[unit].to_numpy(), key])
    if cells.duplicated().any():
        raise ValueError(
            "data has duplicate (unit, period) row(s); the frame path needs "
            "exactly one observation per unit-period"
        )
    n_periods = len(pd.unique(key))
    observed = pd.Series(key).groupby(data[unit].to_numpy(), sort=True).nunique()
    if not (observed == n_periods).all():
        raise ValueError(
            "data is not a balanced panel: some units are missing periods, so "
            "the cohort shares are not comparable across periods. The frame path "
            "needs exactly one observation per unit-period (a balanced panel)."
        )
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
    periods = np.asarray(sorted(pd.unique(key)))
    return cohorts, periods, None


def _resolve_cs_inputs(
    results: "CallawaySantAnnaResults",
) -> Tuple[np.ndarray, Optional[np.ndarray], int, bool]:
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
    anticipation = int(getattr(kit, "anticipation", 0) or 0)
    # Fail closed on a kit without the completeness record: a result pickled
    # by diff-diff <= 3.12.0 cannot say whether its cohort masses are those of
    # a complete panel, and a silently wrong weight table is worse than a
    # refit. (Same policy as the missing-`covariates` key in _guard_cs_design.)
    if "is_balanced" not in bookkeeping:
        raise ValueError(_LEGACY_KIT_MESSAGE)
    is_balanced = bool(bookkeeping["is_balanced"])
    return (
        np.asarray(cohorts),
        (None if weights is None else np.asarray(weights, dtype=float)),
        anticipation,
        is_balanced,
    )


_LEGACY_KIT_MESSAGE = (
    "this result was fitted by diff-diff <= 3.12.0 and does not record the "
    "bookkeeping attgt_weights needs (panel completeness and covariate usage); "
    "refit it to use attgt_weights"
)


def _guard_cs_design(results: "CallawaySantAnnaResults", estimand: str) -> None:
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
    if estimand != "twfe":
        return
    control_group = getattr(results, "control_group", None)
    if control_group not in (None, "never_treated"):
        raise ValueError(
            f"type='twfe' requires control_group='never_treated', got "
            f"{control_group!r}. The TWFE weight formula is derived against a "
            "never-treated comparison group (matching R's twfe_weights, which "
            "raises the same restriction)."
        )
    base_period = getattr(results, "base_period", None)
    if base_period not in (None, "universal"):
        raise ValueError(
            f"type='twfe' requires base_period='universal', got "
            f"{base_period!r}. The formula needs the complete cohort x period "
            "grid, including the pre-treatment cells that a varying base does "
            "not report. Refit with base_period='universal'."
        )
    # R's third restriction: xformla == ~1. The fit records its covariate
    # column names on the aggregation kit; a kit without the key predates that
    # bookkeeping (a result pickled by <= 3.12.0) and fails closed, the same
    # policy as the missing completeness record in _resolve_cs_inputs. A
    # missing kit is left to _resolve_cs_inputs, whose error is the useful one.
    kit = getattr(results, "_aggregation_kit", None)
    if kit is None:
        return
    bookkeeping = getattr(kit, "bookkeeping", {}) or {}
    if "covariates" not in bookkeeping:
        raise ValueError(_LEGACY_KIT_MESSAGE)
    if bookkeeping["covariates"]:
        raise ValueError(
            f"type='twfe' requires a fit without covariates, but this one "
            f"adjusted for {list(bookkeeping['covariates'])!r}. The TWFE weight "
            "formula describes the unadjusted regression (R's twfe_weights stops "
            "unless xformla == ~1); refit with covariates=None, or use "
            "decompose_twfe_weights(covariates=...) for the covariate-adjusted "
            "decomposition."
        )


def attgt_weights(
    results: Union["CallawaySantAnnaResults", pd.DataFrame],
    *,
    type: str = "twfe",  # noqa: A002 - matches results.aggregate(type=) vocabulary
    data: Optional[pd.DataFrame] = None,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    first_treat: Optional[str] = None,
    weights: Optional[Union[str, np.ndarray]] = None,
    anticipation: Optional[int] = None,
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
        formed. A frame carries no record of the producing fit, so on this
        path the caller is responsible for what the fitted path checks: the
        fit used no covariates under ``type="twfe"``; it dropped no unit from
        any cell by its own complete-case rules; ``anticipation=`` is the
        window the fit used; and no cell was hand-built for a cohort the
        estimator would not have estimated. None of these can be detected
        from the frame.
    type : {"twfe", "overall", "simple"}, default "twfe"
        Which estimand's weights to report (the same keyword as
        ``results.aggregate(type=...)``; the accepted values are
        ``ATTGTWeightsResult.LEVELS``). ``"overall"`` is ATT^O, R ``did``'s
        ``attO``. The result records the choice as ``.level``.
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
    anticipation : int, optional
        Anticipation window for the CS estimands, in the calendar's own time
        units exactly as ``CallawaySantAnna(anticipation=)`` counts them: a
        cell counts as post-treatment when ``time >= group - anticipation``
        on the raw labels (on a calendar 10, 20, 30 an ``anticipation=10``
        shifts the window by one period; ``anticipation=1`` shifts it by
        none). Only accepted on the DataFrame path, where it must be the value
        the producing fit used; the fitted path reads it off the fit and
        rejects an explicit ``anticipation=``. It does not change
        ``type="twfe"``'s post window, which stays ``1[t >= g]`` (the
        regression's own indicator; R's ``twfe_weights`` has no anticipation
        argument) - but a cohort the estimator could not estimate at all under
        the window is dropped for every ``type``, and the remaining cohorts'
        masses and weights are those of the panel without it.

    Returns
    -------
    ATTGTWeightsResult
        Per-cell weights plus the negative-weight roll-ups.

    Raises
    ------
    ValueError
        On an unknown ``type``; on a design the formula does not
        support (repeated cross-sections, an unbalanced panel, and - for
        ``type="twfe"`` - a non-never-treated control group, a
        non-universal base period, or a covariate-adjusted fit); on NaN /
        ``-inf`` cohort labels, invalid weights, duplicated or non-finite
        cells; on a fitted result that carries no completeness / covariate
        bookkeeping (pickled by diff-diff <= 3.12.0); or on an INCOMPLETE
        grid: ``"twfe"`` needs every cohort x period cell, ``"overall"`` /
        ``"simple"`` every cell in the anticipation window that the estimator
        did not itself mark structurally absent (see Notes).
    TypeError
        When ``results`` is neither a CallawaySantAnna-family result nor a
        DataFrame.

    Notes
    -----
    Gaps the estimator itself could not fill are handled rather than raised,
    the way ``results.aggregate()`` handles them. A cell is *structurally
    absent* when its ``skip_reason`` is one of ``missing_period`` (no base
    period under the anticipation window - R ``did``'s first-period drop),
    ``zero_treated_control`` (no comparison units, as under
    ``control_group="not_yet_treated"`` for the last cohorts) or
    ``zero_weight_mass`` (zero survey mass in the cell). Then:

    * A cohort with NO estimable post-treatment cell, every missing post cell
      of which is structurally absent, is dropped from the table AND from the
      cohort masses with a warning, for every ``type`` - what
      ``did::pre_process_did`` does when it drops units already treated in the
      first period (or within the anticipation window). The remaining cohorts'
      masses and weights are those of the panel without it. A cohort blanked
      out any other way (NaN effects with no reason) raises rather than
      disappearing.
    * A cohort that keeps at least one estimable post cell is kept, and for
      ``"overall"`` / ``"simple"`` its structurally absent post cells are left
      out of the grid: ``"overall"`` divides the cohort by its number of
      AVAILABLE post cells and ``"simple"`` renormalizes over the available
      cells - what R ``aggte()`` computes on a not-yet-treated fit, and what
      ``aggregate()`` computes on a varying-base fit that keeps a cohort
      through a single estimable cell. A warning names the cells and the
      reasons. (``"twfe"`` requires a never-treated, universal-base fit and
      never reaches this branch.)

    The classification keys on the ``skip_reason`` values, so a
    ``to_dataframe("group_time")`` frame behaves exactly like the fitted
    result. A bare frame WITHOUT ``skip_reason`` has only one structural
    route: a cohort whose window starts at or before the first observed
    period (``group - anticipation <= first period``, R's
    ``g <= first.period + anticipation``) may be dropped; every other gap
    fails closed, being indistinguishable from user truncation.

    Cohort shares assume a COMPLETE panel - the same units in every period -
    so a fitted result is rejected unless every unit-period outcome cell is
    present and finite and the estimator dropped no unit from any cell by its
    own complete-case rules (DMLDiD records its drops; CallawaySantAnna's
    NaN-covariate fallback keeps the cohort masses intact and is not a
    rejection cause). :func:`decompose_twfe_weights` rejects an unbalanced
    panel the same way.

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
    >>> w = diff_diff.attgt_weights(res, type="twfe")  # doctest: +SKIP
    >>> print(w.summary())  # doctest: +SKIP
    """
    if type not in _TYPES:
        raise ValueError(f"type must be one of {list(_TYPES)!r}, got " f"{type!r}")
    if anticipation is not None:
        if isinstance(anticipation, bool) or not isinstance(anticipation, (int, np.integer)):
            raise ValueError(f"anticipation must be a non-negative integer, got {anticipation!r}")
        if int(anticipation) < 0:
            raise ValueError(f"anticipation must be non-negative, got {anticipation!r}")

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
        window = 0 if anticipation is None else int(anticipation)
    else:
        if supplied:
            raise ValueError(
                f"{sorted(supplied)!r} are only for the DataFrame fallback. A "
                "fitted CallawaySantAnnaResults already carries the cohort "
                "bookkeeping - drop them, or pass "
                "result.to_dataframe('group_time') as the first argument."
            )
        if anticipation is not None:
            raise ValueError(
                "anticipation= is only for the DataFrame fallback; a fitted "
                "CallawaySantAnnaResults already carries its own anticipation. "
                "Drop anticipation=, or pass result.to_dataframe('group_time') "
                "as the first argument."
            )
        _guard_cs_design(results, type)
        table, skipped = _attgt_from_cs(results)
        cohorts, survey_weights, window, is_balanced = _resolve_cs_inputs(results)
        if not is_balanced:
            raise ValueError(
                "attgt_weights requires a balanced panel: every unit-period "
                "outcome cell present and finite, and no unit dropped by the "
                "producing estimator's per-cell complete-case rules, because the "
                "cohort shares and E_t[D] assume the same units in every period. "
                "Balance the panel (diff_diff.balance_panel), clean the "
                "non-finite cells, and refit."
            )
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

    # A wrong-length weights= must fail HERE, before the excluded-cohort
    # boolean slice below, which would otherwise raise a raw IndexError.
    if unit_weights is not None and len(np.asarray(unit_weights)) != len(unit_g_pos):
        raise ValueError(
            f"weights has length {len(np.asarray(unit_weights))} but the panel "
            f"has {len(unit_g_pos)} units"
        )

    g_pos = _to_positional_cohort(table["group"].to_numpy(), grid)
    t_pos = np.array([grid[float(t)] for t in table["time"].to_numpy()])
    # Raw label of each position, 1-based (index 0 unused): the anticipation
    # window is defined in the calendar's own units, as CallawaySantAnna
    # applies it (``t < g - anticipation`` on labels), NOT in positions - on a
    # gapped calendar the two differ.
    label_of = np.array([np.nan] + sorted(grid))
    g_int = g_pos.astype(int)

    # Post-treatment mask. The TWFE regression's own indicator is 1[t >= g]
    # regardless of the CS anticipation window; the CS target estimands shift
    # it to 1[label(t) >= label(g) - anticipation] in raw time units.
    if type == "twfe":
        post_mask = t_pos >= g_pos
    else:
        post_mask = label_of[t_pos] >= label_of[g_int] - window

    def _cs_post_start(g: int) -> int:
        """First position whose raw label is ``>= label(g) - window``; never below 1."""
        return int(np.searchsorted(label_of[1:], label_of[g] - window, side="left")) + 1

    def _post_start(g: int) -> int:
        return g if type == "twfe" else _cs_post_start(g)

    # --- whole-cohort exclusion (R did drops cohorts it cannot estimate at all)
    present = set(zip(g_pos.tolist(), t_pos.tolist()))
    panel_cohorts = sorted({int(g) for g in unit_g_pos if g != 0})
    cohorts_with_post = {int(g) for g in g_pos[post_mask]}
    excluded = [g for g in panel_cohorts if g not in cohorts_with_post]
    if excluded:
        # A cohort may be dropped ONLY when the drop is structural. With
        # skip_reason available (fitted result, or its to_dataframe frame):
        # every one of its missing post cells carries a reason the estimator
        # itself emits for a cell it could not form (no base period under the
        # window, no comparison units, zero survey mass). On a bare frame the
        # only route is R did's first-period rule in raw time, g - anticipation
        # <= first period. Any other blanked-out cohort fails closed.
        structural: List[int] = []
        for g in excluded:
            if has_skip_reasons:
                missing_post = [
                    (_label_for(grid, g), _label_for(grid, t))
                    for t in range(_post_start(g), n_periods + 1)
                    if (g, t) not in present
                ]
                if missing_post and all(
                    skipped.get(lab) in _STRUCTURAL_REASONS for lab in missing_post
                ):
                    structural.append(g)
            elif _cs_post_start(g) <= first_period_pos:
                structural.append(g)
        not_structural = [g for g in excluded if g not in structural]
        if not_structural:
            labels = [_label_for(grid, g) for g in not_structural]
            if not has_skip_reasons:
                raise ValueError(
                    f"cohort(s) {labels!r} are present in data= but have no "
                    "post-treatment cell in the ATT(g,t) frame. A bare frame "
                    "cannot say why; pass result.to_dataframe('group_time') "
                    "verbatim (it carries skip_reason) or the fitted result itself."
                )
            raise ValueError(
                f"cohort(s) {labels!r} are present in data= but have no estimable "
                "post-treatment cell, and their missing post cell(s) do not all "
                "carry a structural skip_reason (one of "
                f"{sorted(_STRUCTURAL_REASONS)!r}: no base period under the "
                "anticipation window, no comparison units, or zero survey mass). "
                "A cohort is only dropped when the estimator itself could not "
                "form any of its post cells; blanking a mid cohort's effects is "
                "not that, so it fails closed instead of silently leaving the "
                "estimand."
            )
        n_units_excl = int(np.isin(unit_g_pos, excluded).sum())
        reasons_seen = sorted(
            {
                str(skipped[lab])
                for lab in skipped
                if _pos_of(grid, lab[0]) in excluded and skipped[lab] in _STRUCTURAL_REASONS
            }
        )
        why = (
            f"skip_reason {reasons_seen!r}"
            if reasons_seen
            else "treated at or before the first observed period plus the anticipation window"
        )
        warnings.warn(
            f"cohort(s) {[_label_for(grid, g) for g in excluded]!r} ({n_units_excl} "
            "unit(s)) have no estimable post-treatment cell under the anticipation "
            f"window ({why}) and were dropped from the weight table and the cohort "
            "shares, as R did drops cohorts it cannot estimate; the remaining "
            "cohorts' masses and weights are those of the panel without them",
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
        cohorts, grid, unit_weights, require_control_mass=(type == "twfe")
    )

    # --- grid completeness
    present = set(zip(g_pos.tolist(), t_pos.tolist()))
    surviving = sorted(cohorts_with_post)
    if type == "twfe":
        required = {(g, t) for g in surviving for t in range(1, n_periods + 1)}
    else:
        required = {(g, t) for g in surviving for t in range(_post_start(g), n_periods + 1)}
    missing_cells = sorted(required - present)
    structurally_absent: List[Tuple[Any, Any]] = []
    if missing_cells:
        # The carve-out is keyed on the skip_reason VALUE, not on control_group
        # or base_period: the estimator emits these reasons only for cells it
        # structurally could not form, and the frame path has no design
        # metadata to read, so keying on the reason is what makes the two
        # paths agree. `aggregate()` finite-masks the same cells.
        carve_out_ok = type != "twfe"
        hard: List[Tuple[Tuple[Any, Any], Optional[str]]] = []
        absent_reasons: set = set()
        for g, t in missing_cells:
            label = (_label_for(grid, g), _label_for(grid, t))
            reason = skipped.get(label)
            if carve_out_ok and reason in _STRUCTURAL_REASONS:
                structurally_absent.append(label)
                absent_reasons.add(str(reason))
            else:
                hard.append((label, reason))
        if hard:
            what = "cohort x period" if type == "twfe" else "post-treatment"
            detail = ", ".join(
                f"{lab} [{reason or 'not in source table'}]" for lab, reason in hard[:6]
            )
            raise ValueError(
                f"type={type!r} needs the complete {what} grid, but "
                f"{len(hard)} required cell(s) are missing: {detail}. A weight table "
                "over a partial grid is not the named estimand. Fix the source fit "
                "(or pass the complete to_dataframe('group_time') output)."
            )
        warnings.warn(
            f"{len(structurally_absent)} post-treatment cell(s) {structurally_absent[:6]!r} "
            f"could not be estimated (skip_reason {sorted(absent_reasons)!r}: no "
            "comparison units, no base period under the anticipation window, or "
            "zero survey mass) and are treated as structurally absent: "
            f"type={type!r} averages over each cohort's AVAILABLE post cells, as "
            "R aggte() and results.aggregate() do",
            UserWarning,
            stacklevel=2,
        )

    # Non-estimable PRE cells of surviving cohorts are the only drops left;
    # the CS estimands ignore pre cells, so they change nothing.
    dropped = 0
    surviving_positional = {int(g) for g in cohorts_with_post}
    for g_lab, t_lab in skipped:
        gp, tp = _pos_of(grid, g_lab), _pos_of(grid, t_lab)
        if gp in surviving_positional and tp < _post_start(gp):
            dropped += 1
    if dropped and type != "twfe":
        warnings.warn(
            f"{dropped} pre-treatment group-time cell(s) had no estimable ATT(g,t) "
            f"and were excluded; type={type!r} places no weight on "
            "pre-treatment cells, so the weights are unaffected",
            UserWarning,
            stacklevel=2,
        )

    if type == "twfe":
        weight_vec = _twfe_weight_vector(g_pos, t_pos, n_periods, p_all, e_dt, mean_e_dt)
    elif type == "overall":
        n_post_available = {g: int(((g_pos == g) & post_mask).sum()) for g in surviving}
        weight_vec = _overall_weight_vector(g_pos, p_treated, post_mask, n_post_available)
    else:
        weight_vec = _simple_weight_vector(g_pos, p_treated, post_mask)

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
        level=type,
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


def _require_finite(block: np.ndarray, name: str, *, what: str) -> np.ndarray:
    """Fail closed on NaN / inf in an estimation input block.

    A NaN outcome otherwise propagates silently: the demeaned residual becomes
    NaN and every ``(g, t)`` ATT(g, t) is NaN, so the decomposition returns an
    all-NaN result with no error. Complete-case handling is a policy choice we
    do not make here, so the caller must clean the input.
    """
    if not np.all(np.isfinite(block)):
        n_bad = int((~np.isfinite(block)).sum())
        raise ValueError(
            f"{what} {name!r} contains {n_bad} non-finite value(s) (NaN or inf); "
            "decompose_twfe_weights does not drop incomplete cases, so clean the "
            f"panel first (e.g. drop or impute rows with a missing {what})"
        )
    return block


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

        time_key = _validate_time_labels(data[time].to_numpy(), what=time)
        frame = data.assign(_twfe_time_key=time_key)
        frame = frame.sort_values([unit, "_twfe_time_key"]).reset_index(drop=True)
        units = frame[unit].to_numpy()
        periods = frame["_twfe_time_key"].to_numpy(dtype=float)
        self.unit_ids = np.asarray(sorted(pd.unique(units)))
        # The numeric key orders, reshapes and maps cohorts; the ORIGINAL
        # labels are what every reporting surface (cells, summary(), balance
        # rows, plots) shows, so a string-labelled panel reports strings.
        self.period_keys = np.asarray(sorted(pd.unique(periods)))
        n_units = len(self.unit_ids)
        n_periods = len(self.period_keys)
        if len(frame) != n_units * n_periods:
            raise ValueError(
                f"decompose_twfe_weights requires a balanced panel: got "
                f"{len(frame)} rows for {n_units} units x {n_periods} periods. "
                "Balance it first, e.g. diff_diff.balance_panel(data, unit=..., "
                "time=...)."
            )
        counts = frame.groupby(unit, sort=True)["_twfe_time_key"].nunique().to_numpy()
        if not np.all(counts == n_periods):
            raise ValueError(
                "decompose_twfe_weights requires a balanced panel: some units "
                "are missing periods"
            )

        self.grid = _positional_grid(self.period_keys)
        self.n_units = n_units
        self.n_periods = n_periods
        labels = frame.groupby("_twfe_time_key")[time]
        if (labels.nunique() > 1).any():
            raise ValueError(
                f"{time!r} mixes representations of the same period (e.g. '2' and "
                "2.0); use one label per period"
            )
        self.period_labels = labels.first().loc[self.period_keys].to_numpy()

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
        self.outcome = _require_finite(
            frame[outcome].to_numpy(dtype=float).reshape(n_units, n_periods),
            outcome,
            what="outcome",
        )
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
            self.design = _require_finite(
                frame[list(covariates)]
                .to_numpy(dtype=float)
                .reshape(n_units, n_periods, len(covariates)),
                ", ".join(covariates),
                what="covariate",
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
            {"_unit": frame[unit].to_numpy(), "_time": frame["_twfe_time_key"].to_numpy()}
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
        frame = data.assign(_twfe_time_key=_validate_time_labels(data[time].to_numpy(), what=time))
        frame = frame.sort_values([unit, "_twfe_time_key"]).reset_index(drop=True)
        block = (
            frame[list(names)]
            .to_numpy(dtype=float)
            .reshape(self.n_units, self.n_periods, len(names))
        )
        for j, name in enumerate(names):
            _require_finite(block[:, :, j], name, what="balance covariate")
        return block.mean(axis=1)


def _fwl_residuals(panel: _Panel) -> Tuple[np.ndarray, float, List[str]]:
    """Frisch-Waugh-Lovell residual of treatment on covariates, plus its scale.

    Double-demeans ``D`` and ``X``, projects the demeaned treatment on the
    demeaned covariates, and returns the residual. That residual IS the
    implicit weight the regression applies to each observation; ``alpha_den``
    is the normalization ``E[resid * Ddot]`` from R's
    ``combine_twfe_weights_gt``. The third element is the names of the
    covariates that SURVIVED the annihilation filter, so the result reports the
    columns the regression actually used rather than the user's input list.

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
    n_obs = panel.n_units * panel.n_periods
    flat_x = x_dot.reshape(n_obs, x_dot.shape[2])

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
    # 1e-16 is simply the largest pivot).
    #
    # The threshold is the accumulated ROUNDING-NOISE scale, not a fixed
    # relative constant: demeaning an ``n_obs``-row column accumulates
    # ``O(sqrt(n) * eps)`` of relative error, times a safety factor of 64.
    # A fixed 1e-10 was five orders too loose - it discarded a covariate with
    # level 1e6 and genuine within-sd 1e-4 (ratio 1e-10) - while still
    # annihilating mpdta's `lpop`, whose true within-variation is many orders
    # above the noise floor.
    raw_scale = np.linalg.norm(
        panel.design.reshape(n_obs, x_dot.shape[2]),
        axis=0,
    )
    demeaned_scale = np.linalg.norm(flat_x, axis=0)
    noise_floor = np.sqrt(n_obs) * 64.0 * np.finfo(float).eps
    annihilated = demeaned_scale <= noise_floor * np.maximum(raw_scale, 1.0)
    if annihilated.any():
        names = [panel.covariates[j] for j in np.flatnonzero(annihilated)]
        warnings.warn(
            f"covariate(s) {names!r} have no within-unit-and-period variation "
            "(or within-variation at the floating-point noise floor of their own "
            "level) and were dropped: two-way demeaning annihilates them, so they "
            "cannot affect a two-way fixed effects regression. If that is not "
            "intended, centre or rescale the covariate so its within-variation is "
            "not negligible next to its level",
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
    return resid.reshape(panel.n_units, panel.n_periods), alpha_den, surviving


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
    resid, alpha_den, surviving_covariates = _fwl_residuals(panel)
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
        "pre_period_contribution": float((weight_vec[~post_col] * att_col[~post_col]).sum()),
        "post_only": float((weight_vec[post_col] * att_col[post_col]).sum()),
        # summary.decomposed_twfe: post cells only, on both factors
        "effective_sample_size": float(
            post_col.sum() * (weight_vec[post_col] * ess_col[post_col]).sum()
        ),
        "balance": pd.DataFrame(balance_rows) if balance_block is not None else None,
        "covariates": tuple(surviving_covariates),
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
    ATT(g, t), and separates out the part of the estimate that comes from
    PRE-treatment cells (``pre_period_contribution``). That component can
    reflect differential pre-trends OR plain sampling variation - the
    diagnostic carries no inference - so it is diagnostic evidence about the
    earlier-period restrictions, not proof that the identifying assumption
    fails in the post-treatment counterfactual.

    Takes the raw panel rather than a fitted result, because it re-estimates:
    it double-demeans treatment and covariates and forms its own group-time
    contrasts, so there is no ATT(g, t) table it could consume. Its companion
    :func:`attgt_weights` is the fitted-result surface, and the two are tied
    by an identity that holds when the fit used ``base_period="universal"``,
    ``control_group="never_treated"`` and no covariates, no cohort was
    dropped by :func:`attgt_weights`' structural rule (every cohort has at
    least one estimable post cell under the fit's anticipation window - a
    cohort treated in the first observed period is dropped there but kept
    here under ``base_period="first_period"``), and both sides use the same
    unit weights::

        sum(attgt_weights(cs, type="twfe").weights.eval("weight * att"))
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
    >>> dec.pre_period_contribution  # doctest: +SKIP
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
        pre_period_contribution=payload["pre_period_contribution"],
        post_only=payload["post_only"],
        base_period=base_period,
        covariates=payload["covariates"],
        effective_sample_size=payload["effective_sample_size"],
        n_units=panel.n_units,
        n_periods=panel.n_periods,
        balance=payload["balance"],
    )
