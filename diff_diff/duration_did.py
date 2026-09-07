"""Duration Difference-in-Differences (Deaner & Ku 2026).

Two-group, common-timing causal duration analysis for a binary ABSORBING
outcome (``Y_it = 1`` once the spell has ended). Under a restriction on the
UNTREATED hazards of the two groups — a constant additive gap (common
dynamics, ``method="cd"``) or a constant ratio (proportional hazards,
``method="ph"``) — the treated group's counterfactual survival is imputed
from the control group's cumulative hazard and the treated baseline, and
the absorption ATT ``E[Y_it - Y_it(0) | treated]`` is reported for every
post-treatment date (Theorem 1; Equations 3.1-3.4 with the mean-of-ratios PH
estimator). Inference is the whole-individual pooled bootstrap of Appendix B
Algorithm 1 (centered absolute-deviation pointwise and simultaneous bands);
the Algorithm 2 fixed-anchor pre-treatment specification test is reported
separately. See ``docs/methodology/REGISTRY.md`` (DurationDiD) and
``docs/methodology/papers/deaner-ku-2026-review.md``.

Notation (review lines 94-108): ``S_kt`` group survival, ``R_kt = -log S_kt``,
``D_kt = R_kt - R_k1``, ``H_kt = D_kt / e_t`` with the actual elapsed time
``e_t = time_t - time_1``; ``tstar`` is the last untreated date.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff.bootstrap_chunking import compute_block_size
from diff_diff.duration_did_results import (
    DurationDiDPretestResults,
    DurationDiDResults,
    invalid_curve_message,
)
from diff_diff.utils import (
    safe_inference,
    safe_inference_batch,
    validate_binary,
    validate_n_bootstrap,
)

_VALID_METHODS = ("cd", "ph")
#: A used log-survival moment backed by fewer than this many survivors (but at
#: least one) is reported as weak numerical support — a warning only, no
#: behavior changes (review lines 917-918). A count rule: a survival
#: PROPORTION is never below ``1/n_group``, so a proportion threshold would be
#: unreachable at any realistic sample size.
_WEAK_SUPPORT_MIN_SURVIVORS = 5
#: Relative tolerance of the equal-spacing check on the time grid.
_SPACING_RTOL = 1e-8
#: Baseline + last pre-treatment date + one post-treatment date.
_MIN_PERIODS = 3
#: Cap on bootstrap rows per chunk (the count matrix is ``(rows, n)`` float64).
_MAX_CHUNK_ROWS = 256


def _errstate() -> Any:
    """Silence every floating-point warning class (CONTRIBUTING: protect all arithmetic)."""
    return np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")


# Per-family draw-failure reasons, in FIRST-MATCH precedence order.
_POST_FAILURE_ORDER = (
    "group_empty",
    "zero_survival_baseline",
    "zero_survival_last_pre",
    "zero_control_increment",
    "control_survival_zero",
    "nonfinite_counterfactual",
    "nonfinite_effect",
)
_PRETEST_FAILURE_ORDER = (
    "group_empty",
    "zero_survival_baseline",
    "zero_survival_last_pre",
    "zero_control_increment",
    "nonfinite_contrast",
)


# =============================================================================
# Constructor validation
# =============================================================================


def _validate_method(method: Any) -> None:
    if not isinstance(method, str) or method not in _VALID_METHODS:
        raise ValueError(f"method must be 'cd' or 'ph', got {method!r}")


def _validate_alpha(alpha: Any) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)):
        raise ValueError(f"alpha must be a float strictly between 0 and 1, got {alpha!r}")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be a float strictly between 0 and 1, got {alpha!r}")


def _validate_seed(seed: Any) -> None:
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be None or a non-negative integer, got {seed!r}")


def _validate_draws(n_bootstrap: Any) -> None:
    validate_n_bootstrap(n_bootstrap)
    if int(n_bootstrap) == 1:
        raise ValueError(
            "n_bootstrap must be 0 (point estimates only, no inference) or at "
            "least 2 (a bootstrap SD needs two draws); got 1"
        )


# =============================================================================
# Numerical core (pure numpy; a leading draw axis where noted)
# =============================================================================


def _validate_and_arrange(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    last_pre_period: Any,
) -> Dict[str, Any]:
    """Validate the balanced absorbing panel and arrange it as arrays.

    Returns ``Y`` (n x T float 0/1, treated units first), ``n_treated``,
    ``grid`` (the sorted common dates in their native numeric dtype),
    ``elapsed`` (float, ``grid - grid[0]``) and ``tstar_idx``.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    for name, col in (
        ("outcome", outcome),
        ("unit", unit),
        ("time", time),
        ("treatment", treatment),
    ):
        if col not in data.columns:
            raise ValueError(f"{name} column {col!r} not found in data")

    # Identifier checks BEFORE any grouping/pivot (no silent groupby drop, no
    # phantom NaN unit).
    if data[unit].isna().any():
        raise ValueError(
            f"unit column {unit!r} contains missing values; every row needs a "
            "unit identifier (no silent groupby drop)"
        )

    t_col = data[time]
    if pd.api.types.is_datetime64_any_dtype(t_col) or pd.api.types.is_timedelta64_dtype(t_col):
        raise ValueError(
            f"time column {time!r} must be numeric; convert datetime/timedelta "
            "values to a numeric elapsed scale (e.g. days since the spell "
            "start) before fitting"
        )
    if pd.api.types.is_bool_dtype(t_col) or not pd.api.types.is_numeric_dtype(t_col):
        raise ValueError(f"time column {time!r} must be numeric (got dtype {t_col.dtype})")
    if t_col.isna().any() or not np.all(np.isfinite(t_col.to_numpy(dtype=float))):
        raise ValueError(f"time column {time!r} contains missing or non-finite values")

    grid = np.unique(t_col.to_numpy())
    if len(data) == 0 or len(grid) < _MIN_PERIODS:
        raise ValueError(
            "DurationDiD requires at least three distinct time periods (baseline, "
            f"last_pre_period, and one post-period); found {len(grid)}"
        )

    if data.duplicated(subset=[unit, time]).any():
        raise ValueError(
            f"DurationDiD requires exactly one row per (unit, period); found duplicate "
            f"({unit!r}, {time!r}) combinations"
        )
    counts = data.groupby(unit, sort=True)[time].size()
    n_periods = len(grid)
    incomplete = counts[counts != n_periods]
    if len(incomplete) > 0:
        bad = incomplete.index.tolist()[:5]
        raise ValueError(
            "Unbalanced panel: every individual must be observed at every date of the "
            f"common time grid ({n_periods} periods); {len(incomplete)} unit(s) are not "
            f"(e.g. {bad}). Late entry, dropout, and missing cells are not supported; "
            "an administrative end of a complete window is fine."
        )

    grid_f = grid.astype(float)
    diffs = np.diff(grid_f)
    step = diffs[0]
    if not (np.isfinite(step) and step > 0):
        raise ValueError("time grid must have a positive finite common spacing")
    if not np.allclose(diffs, step, rtol=_SPACING_RTOL, atol=0.0):
        raise ValueError(
            "DurationDiD requires an equally spaced time grid (relative tolerance "
            f"{_SPACING_RTOL:g}); found spacings {np.unique(diffs).tolist()[:5]}"
        )

    # Binary columns: explicit float coercion, then missing/non-finite, then
    # the 0/1 domain (validate_binary strips NaN before its membership test).
    coerced: Dict[str, np.ndarray] = {}
    for name, col in (("outcome", outcome), ("treatment", treatment)):
        if not (pd.api.types.is_numeric_dtype(data[col]) or pd.api.types.is_bool_dtype(data[col])):
            bad_vals = pd.unique(data[col].astype(object))[:5].tolist()
            raise ValueError(
                f"{name} column {col!r} must be a numeric 0/1 column (got dtype "
                f"{data[col].dtype}; values such as {bad_vals}); numeric strings are not "
                "coerced"
            )
        try:
            arr = data[col].to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            bad_vals = pd.unique(data[col].astype(object))[:5].tolist()
            raise ValueError(
                f"{name} column {col!r} must be numeric 0/1; could not convert values "
                f"such as {bad_vals} ({exc})"
            ) from None
        nonfinite = ~np.isfinite(arr)
        if nonfinite.any():
            rows = data.loc[nonfinite, [unit, time]].head(5).values.tolist()
            raise ValueError(
                f"{name} column {col!r} contains {int(nonfinite.sum())} missing or "
                f"non-finite value(s); first offending (unit, period) pairs: {rows}"
            )
        validate_binary(arr, name)
        coerced[name] = arr

    # Internal frame with fixed names, so a user column named like a temporary
    # (or a role column named "unit"/"time") can never collide.
    frame = pd.DataFrame(
        {
            "unit": data[unit].to_numpy(),
            "time": data[time].to_numpy(),
            "y": coerced["outcome"],
            "g": coerced["treatment"],
        }
    )

    g_nunique = frame.groupby("unit")["g"].nunique()
    if (g_nunique > 1).any():
        bad = g_nunique[g_nunique > 1].index.tolist()[:5]
        raise ValueError(
            f"treatment column {treatment!r} must be a fixed 0/1 group indicator "
            f"(constant within unit), not a time-varying received-treatment variable; "
            f"units with varying values include {bad}"
        )

    y_wide = frame.pivot(index="unit", columns="time", values="y").reindex(columns=grid)
    g_units = frame.groupby("unit")["g"].first().reindex(y_wide.index)
    Y = y_wide.to_numpy(dtype=float)
    G = g_units.to_numpy(dtype=float)
    n_treated = int(np.sum(G == 1.0))
    n_control = int(np.sum(G == 0.0))
    if n_treated == 0 or n_control == 0:
        raise ValueError(
            "both groups are required: found "
            f"{n_treated} treated and {n_control} control individual(s)"
        )

    reversal = np.diff(Y, axis=1) < 0
    if reversal.any():
        bad_units = y_wide.index[reversal.any(axis=1)].tolist()[:5]
        raise ValueError(
            "outcome must be absorbing (once 1, always 1 within each individual); "
            f"found reversals (1 -> 0) for {int(reversal.any(axis=1).sum())} unit(s), "
            f"e.g. {bad_units}"
        )

    if isinstance(last_pre_period, (bool, str)) or not isinstance(
        last_pre_period, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"last_pre_period must be a numeric value of the time column, got {last_pre_period!r}"
        )
    tstar_val = float(last_pre_period)
    matches = np.nonzero(grid_f == tstar_val)[0]
    if len(matches) != 1:
        raise ValueError(
            f"last_pre_period {last_pre_period!r} is not a value of the time column "
            f"(grid: {grid.tolist()[:8]}{'...' if len(grid) > 8 else ''})"
        )
    tstar_idx = int(matches[0])
    if tstar_idx == 0:
        raise ValueError(
            "last_pre_period equals the first date; at least two pre-treatment dates "
            "(the baseline and last_pre_period) are required"
        )
    if tstar_idx == n_periods - 1:
        raise ValueError(
            "last_pre_period equals the last date; at least one post-treatment date is required"
        )

    order = np.argsort(-G, kind="stable")  # treated first, stable within group
    return {
        "Y": np.ascontiguousarray(Y[order]),
        "n_treated": n_treated,
        "grid": grid,
        "elapsed": grid_f - grid_f[0],
        "tstar_idx": tstar_idx,
    }


def _group_survival(
    Y: np.ndarray, n_treated: int, W: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group survival curves for count-weight rows ``W`` (C x n).

    Returns ``S`` (C x 2 x T; treated row 0, control row 1), plus the
    resampled group sizes ``n1``, ``n2`` (C,). Integer-valued sums below
    ``2**53`` make the GEMM bit-identical to a per-draw loop. An empty group
    yields NaN survival (caught by the draw-failure predicates).
    """
    W = np.asarray(W, dtype=float)
    W1, W2 = W[:, :n_treated], W[:, n_treated:]
    n1, n2 = W1.sum(axis=1), W2.sum(axis=1)
    with _errstate():
        S1 = 1.0 - (W1 @ Y[:n_treated]) / n1[:, None]
        S2 = 1.0 - (W2 @ Y[n_treated:]) / n2[:, None]
    return np.stack([S1, S2], axis=1), n1, n2


def _log_survival_moments(
    S: np.ndarray, elapsed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``R = -log S``, ``D = R - R[..., :1]``, ``H = D / elapsed`` (baseline NaN)."""
    with _errstate():
        R = -np.log(S)
        D = R - R[..., :1]
        H = D / elapsed
    return R, D, H


def _estimate_from_survival(
    S: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    elapsed: np.ndarray,
    fit_idx: np.ndarray,
    fit_weights: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fitted coefficient and imputed counterfactual, vectorized over draws.

    Returns ``c`` (C,), ``R0`` (C x T), ``S0`` (C x T) and ``tau = S0 - S1``
    (C x T, every date; the caller slices the post dates). Under
    ``method="cd"``: ``c = sum_t alpha_t (H_1t - H_2t)`` and
    ``R0 = R_11 + D_2 + e*c`` (Equations 3.3-3.4); under ``method="ph"``:
    ``c = sum_t alpha_t D_1t / D_2t`` (mean of ratios, Theorem 1) and
    ``R0 = R_11 + c * D_2`` (Equation 2.16, the treated baseline outside the
    exponent).
    """
    S1 = S[:, 0, :]
    R1, D1, D2 = R[:, 0, :], D[:, 0, :], D[:, 1, :]
    w = np.asarray(fit_weights, dtype=float)
    with _errstate():
        if method == "cd":
            e_fit = elapsed[fit_idx]
            H1 = D1[:, fit_idx] / e_fit
            H2 = D2[:, fit_idx] / e_fit
            c = (H1 - H2) @ w
            R0 = R1[:, :1] + D2 + elapsed[None, :] * c[:, None]
        else:
            ratio = D1[:, fit_idx] / D2[:, fit_idx]
            c = ratio @ w
            R0 = R1[:, :1] + c[:, None] * D2
        S0 = np.exp(-R0)
        tau = S0 - S1
    return c, R0, S0, tau


#: Relative roundoff tolerance of the curve-validity gate (scaled by the
#: magnitudes actually summed into ``R0``).
_CURVE_GATE_RTOL = 1e-12


def _curve_tolerance(
    R: np.ndarray, D: np.ndarray, elapsed: np.ndarray, c: np.ndarray, method: str
) -> np.ndarray:
    """Per-date absolute tolerance for the curve-validity gate (leading draw axis).

    ``R0`` is a sum of ``R_11``, ``D_2t`` and ``e_t c`` (CD) or ``c D_2t``
    (PH); at a mathematical boundary (e.g. an exactly-zero imputed cumulative
    hazard when the treated group has no pre-treatment exits) the sum can
    land a few ulps on either side of the exact value, and which side
    depends on the time labelling. The tolerance is ``1e-12`` times the
    largest magnitude summed, floored at ``1e-12``.
    """
    with _errstate():
        term = elapsed[None, :] * c[:, None] if method == "cd" else c[:, None] * D[:, 1, :]
        scale = np.maximum.reduce(
            [
                np.ones_like(term),
                np.abs(np.broadcast_to(R[:, 0, :1], term.shape)),
                np.abs(D[:, 1, :]),
                np.abs(term),
            ]
        )
    return _CURVE_GATE_RTOL * scale


def _curve_status(
    R0: np.ndarray, S2: np.ndarray, tstar_idx: int, tol: Optional[np.ndarray] = None
) -> List[str]:
    """Validity of the imputed curve at every date (first match wins).

    Review lines 920-924: negative or decreasing imputed cumulative hazards
    and survival outside [0, 1] are invalid. The gate is on the imputed
    curve itself over the WHOLE path (fitted pre-dates included), never on
    the observed treated hazard at ``tstar`` (that residual is Algorithm 2's
    object). ``tol`` (per date, from :func:`_curve_tolerance`) keeps an
    exact mathematical boundary — ``R0 == 0`` or a zero step — from being
    flagged on roundoff; ``None`` means exact comparisons.
    """
    n_periods = len(R0)
    tol_arr = np.zeros(n_periods) if tol is None else np.asarray(tol, dtype=float)
    status = ["ok"] * n_periods
    for t in range(1, n_periods):
        if t > tstar_idx and S2[t] <= 0:
            status[t] = "control_survival_zero"
        elif not np.isfinite(R0[t]):
            status[t] = "counterfactual_nonfinite"
        elif R0[t] < -tol_arr[t]:
            status[t] = "counterfactual_survival_above_one"
        elif np.isfinite(R0[t - 1]) and R0[t] < R0[t - 1] - tol_arr[t]:
            status[t] = "counterfactual_nonmonotone"
    return status


def _resolve_fit_periods(
    S: np.ndarray,
    D: np.ndarray,
    method: str,
    grid: np.ndarray,
    tstar_idx: int,
    pre_periods: Optional[Sequence[Any]],
    pre_period_weights: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, Dict[Any, str], List[str]]:
    """Validate the fitting selectors and resolve the eligible fitting set.

    Returns the sorted fitting date indices ``F``, their normalized weights,
    an exclusion dict ``{period: reason}`` and warning messages. Zero-weight
    candidates are excluded first, then eligibility (positive survival in
    both groups; under PH a positive control increment); surviving weights
    are renormalized to sum to one. ``F`` and the weights are frozen for
    every bootstrap draw.
    """
    grid_f = grid.astype(float)
    if pre_period_weights is not None and pre_periods is None:
        raise ValueError("pre_period_weights requires pre_periods (the dates the weights refer to)")

    if pre_periods is None:
        cand_idx = np.arange(1, tstar_idx + 1)
        requested = False
    else:
        req = _selector_array(pre_periods, "pre_periods", "time values")
        if req.size == 0:
            raise ValueError("pre_periods must name at least one pre-treatment date")
        idx_list: List[int] = []
        for p in req:
            hit = np.nonzero(grid_f == p)[0]
            if len(hit) != 1:
                raise ValueError(f"Pre-period '{_fmt(p)}' not found in time column")
            k = int(hit[0])
            if k < 1 or k > tstar_idx:
                raise ValueError(
                    f"pre_periods value {_fmt(p)} must lie strictly after the baseline "
                    f"date {_label(grid, 0)!r} and at or before last_pre_period {_label(grid, tstar_idx)!r}"
                )
            idx_list.append(k)
        if len(set(idx_list)) != len(idx_list):
            raise ValueError("pre_periods contains duplicate dates")
        cand_idx = np.asarray(idx_list, dtype=int)
        requested = True

    if pre_period_weights is None:
        cand_w = np.ones(len(cand_idx), dtype=float)
    else:
        cand_w = _selector_array(pre_period_weights, "pre_period_weights", "nonnegative weights")
        if cand_w.shape != (len(cand_idx),):
            raise ValueError(
                f"pre_period_weights must have one entry per pre_periods date "
                f"({len(cand_idx)}), got {cand_w.shape[0]}"
            )
        if not np.all(np.isfinite(cand_w)):
            raise ValueError("pre_period_weights must be finite")
        if np.any(cand_w < 0):
            raise ValueError("pre_period_weights must be nonnegative")
        if np.all(cand_w == 0):
            raise ValueError("pre_period_weights must not all be zero")

    excluded: Dict[Any, str] = {}
    keep_idx: List[int] = []
    keep_w: List[float] = []
    S1, S2, D2 = S[0], S[1], D[1]
    for k, w in zip(cand_idx.tolist(), cand_w.tolist()):
        label = grid[k].item() if hasattr(grid[k], "item") else grid[k]
        if w == 0:
            excluded[label] = "zero_weight"
        elif S1[k] <= 0:
            excluded[label] = "zero_treated_survival"
        elif S2[k] <= 0:
            excluded[label] = "zero_control_survival"
        elif method == "ph" and D2[k] <= 0:
            excluded[label] = "zero_control_increment"
        else:
            keep_idx.append(k)
            keep_w.append(w)
    if not keep_idx:
        raise ValueError(
            "no eligible fitting period: every candidate pre-treatment date was "
            f"excluded ({excluded}). PH needs a positive control cumulative-hazard "
            "increment at some pre-date after the baseline; both groups need positive "
            "survival at the fitting dates."
        )
    order = np.argsort(keep_idx)
    fit_idx = np.asarray(keep_idx, dtype=int)[order]
    weights = np.asarray(keep_w, dtype=float)[order]
    # Scale-invariant normalization: the kept weights are finite and strictly
    # positive, so dividing by the maximum first keeps the sum finite even for
    # weights near the float64 limit (a bare sum could overflow to inf and
    # silently normalize to zeros).
    weights = weights / weights.max()
    weights = weights / weights.sum()

    messages: List[str] = []
    elig_excluded = {k: v for k, v in excluded.items() if v != "zero_weight"}
    if elig_excluded:
        who = "requested" if requested else "default"
        messages.append(
            f"Excluded {len(elig_excluded)} {who} fitting period(s) as ineligible "
            f"{elig_excluded}; the remaining fitting weights were renormalized to sum to one."
        )
    if tstar_idx not in set(fit_idx.tolist()):
        messages.append(
            f"The fitting periods omit last_pre_period {_label(grid, tstar_idx)!r}; the "
            "pre-treatment diagnostic keeps that date as its fixed anchor regardless."
        )
    return fit_idx, weights, excluded, messages


def _selector_array(value: Any, name: str, kind: str) -> np.ndarray:
    """Coerce a fit-time selector to a 1-d float array with a typed guard.

    Scalars, strings and bytes are rejected explicitly: ``list("34")`` would
    otherwise split a numeric string into two different dates and silently
    change the fitting set. Sets are rejected because ``pre_periods`` and
    ``pre_period_weights`` are paired by position.
    """
    if isinstance(value, (str, bytes)) or np.isscalar(value) or not hasattr(value, "__iter__"):
        raise ValueError(f"{name} must be a list of {kind} (e.g. [...]), got {value!r}")
    if isinstance(value, (set, frozenset)):
        # Unordered: pre_periods and pre_period_weights are paired positionally.
        raise ValueError(
            f"{name} must be an ordered list of {kind} (a set has no positional order "
            f"to align with its companion selector), got {value!r}"
        )
    try:
        arr = np.asarray(list(value), dtype=float).ravel()
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a list of numeric {kind}, got {value!r}") from None
    return arr


def _label(grid: np.ndarray, k: int) -> Any:
    """Native Python scalar for a grid date (never a numpy repr in messages)."""
    v = grid[k]
    return v.item() if hasattr(v, "item") else v


def _fmt(p: float) -> Any:
    return int(p) if float(p).is_integer() else p


def _pretest_contrasts(
    D: np.ndarray, H: np.ndarray, J: np.ndarray, tstar_idx: int, method: str
) -> np.ndarray:
    """Algorithm 2 fixed-anchor contrasts over ``J`` (leading draw axis)."""
    with _errstate():
        if method == "cd":
            gap = H[:, 0, J] - H[:, 1, J]
            anchor = H[:, 0, tstar_idx] - H[:, 1, tstar_idx]
        else:
            gap = D[:, 0, J] / D[:, 1, J]
            anchor = D[:, 0, tstar_idx] / D[:, 1, tstar_idx]
        return gap - anchor[:, None]


def _quantile_inverted_cdf(x: np.ndarray, p: float) -> float:
    """Inverse empirical CDF quantile: the ``ceil(p*B)``-th order statistic.

    ``p*B`` is evaluated with a ``1e-9`` tie guard (``ceil(p*B - 1e-9)``) so a
    product that lands within floating-point noise of an integer (e.g.
    ``0.95 * 20 = 19.000000000000004``) selects that integer's order statistic,
    matching the documented rule for the alphas users actually pass (the same
    guard magnitude as ``utils._frac_gt``).
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.shape[0]
    if n == 0:
        return float("nan")
    k = int(math.ceil(p * n - 1e-9))
    k = min(max(k, 1), n)
    return float(xs[k - 1])


def _centered_bootstrap_summary(
    point: np.ndarray, draws: np.ndarray, alpha: float
) -> Dict[str, Any]:
    """Centered absolute-deviation bootstrap summary on COMPLETE draws.

    ``se`` is derived from the diagonal of the covariance (never a separate
    ``np.std``) so ``se == sqrt(diag(vcov))`` holds exactly. Pointwise
    and simultaneous critical values are inverse-empirical-CDF quantiles of
    ``|draw - point| / se`` and of its per-draw maximum; p-values are the
    empirical tail fractions (equality counted). ``|point/se| > crit`` is
    exactly ``p <= alpha``.
    """
    point = np.asarray(point, dtype=float).ravel()
    draws = np.asarray(draws, dtype=float).reshape(draws.shape[0], -1)
    with _errstate():
        vcov = np.atleast_2d(np.cov(draws, rowvar=False, ddof=1))
        se = np.sqrt(np.diag(vcov))
        z = np.abs(draws - point[None, :]) / se[None, :]
        crit = np.array([_quantile_inverted_cdf(z[:, k], 1.0 - alpha) for k in range(z.shape[1])])
        t_abs = np.abs(point / se)
        p = np.mean(z >= t_abs[None, :], axis=0)
        m = np.max(z, axis=1)
        crit_sim = _quantile_inverted_cdf(m, 1.0 - alpha)
        p_joint = float(np.mean(m >= np.max(t_abs)))
    return {
        "vcov": vcov,
        "se": se,
        "crit": crit,
        "ci_lower": point - crit * se,
        "ci_upper": point + crit * se,
        "p": p,
        "crit_sim": crit_sim,
        "band_lower": point - crit_sim * se,
        "band_upper": point + crit_sim * se,
        "p_joint": p_joint,
        "statistic": float(np.max(t_abs)),
    }


def _draw_indices(rng: np.random.Generator, n: int, size: int) -> np.ndarray:
    """Pooled whole-individual resample indices (``size`` x ``n``)."""
    return rng.integers(0, n, size=(size, n))


def _first_match(masks: Dict[str, np.ndarray], order: Tuple[str, ...], n: int) -> np.ndarray:
    """Per-draw reason string: the first predicate in ``order`` that fires."""
    reason = np.full(n, "", dtype=object)
    for name in order:
        reason = np.where((reason == "") & masks[name], name, reason)
    return reason


def _run_bootstrap(
    Y: np.ndarray,
    n_treated: int,
    elapsed: np.ndarray,
    tstar_idx: int,
    fit_idx: np.ndarray,
    fit_weights: np.ndarray,
    method: str,
    J: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Algorithm 1: ``n_bootstrap`` pooled whole-history resamples.

    Every draw recomputes survival, moments, the coefficient, the
    counterfactual, the post effects, the headline and the pretest
    contrasts with the frozen fitting set. Failures are recorded per family
    (first-match reason); failed draws are NaN rows.
    """
    n, n_periods = Y.shape
    post_idx = np.arange(tstar_idx + 1, n_periods)
    n_post = len(post_idx)
    tau_star = np.full((n_bootstrap, n_post), np.nan)
    head_star = np.full(n_bootstrap, np.nan)
    delta_star = np.full((n_bootstrap, len(J)), np.nan)
    ok_post = np.zeros(n_bootstrap, dtype=bool)
    ok_pretest = np.zeros(n_bootstrap, dtype=bool)
    reason_post = np.full(n_bootstrap, "", dtype=object)
    reason_pretest = np.full(n_bootstrap, "", dtype=object)
    n_invalid_curve = 0

    chunk = int(min(compute_block_size(n, n_bootstrap), _MAX_CHUNK_ROWS))
    pretest_dates = np.concatenate([J, [tstar_idx]]).astype(int)
    for start in range(0, n_bootstrap, chunk):
        size = min(chunk, n_bootstrap - start)
        idx = _draw_indices(rng, n, size)
        W = np.zeros((size, n), dtype=float)
        for r in range(size):
            W[r] = np.bincount(idx[r], minlength=n)
        S, n1, n2 = _group_survival(Y, n_treated, W)
        R, D, H = _log_survival_moments(S, elapsed)
        c, R0, S0, tau = _estimate_from_survival(S, R, D, elapsed, fit_idx, fit_weights, method)
        tau_post = tau[:, post_idx]
        with _errstate():
            head = tau_post.mean(axis=1)
            delta = (
                _pretest_contrasts(D, H, J, tstar_idx, method) if len(J) else np.zeros((size, 0))
            )

        S1, S2 = S[:, 0, :], S[:, 1, :]
        group_empty = (n1 == 0) | (n2 == 0)
        base_zero = ~(np.nan_to_num(S1[:, 0], nan=0.0) > 0) | ~(
            np.nan_to_num(S2[:, 0], nan=0.0) > 0
        )
        tstar_zero = ~(np.nan_to_num(S1[:, tstar_idx], nan=0.0) > 0) | ~(
            np.nan_to_num(S2[:, tstar_idx], nan=0.0) > 0
        )
        with _errstate():
            if method == "ph":
                zero_inc_post = ~np.all(D[:, 1, fit_idx] > 0, axis=1)
                zero_inc_pre = ~np.all(D[:, 1, pretest_dates] > 0, axis=1)
            else:
                zero_inc_post = np.zeros(size, dtype=bool)
                zero_inc_pre = np.zeros(size, dtype=bool)
            ctrl_zero = ~np.all(np.nan_to_num(S2[:, post_idx], nan=0.0) > 0, axis=1)
            nonfinite_cf = (
                ~np.isfinite(c)
                | ~np.all(np.isfinite(R0[:, post_idx]), axis=1)
                | ~np.all(np.isfinite(S0[:, post_idx]), axis=1)
            )
            nonfinite_eff = ~np.all(np.isfinite(tau_post), axis=1)
            if method == "cd":
                inputs_ok = np.all(np.isfinite(H[:, :, pretest_dates]), axis=(1, 2))
            else:
                inputs_ok = np.all(np.isfinite(D[:, :, pretest_dates]), axis=(1, 2))
            nonfinite_delta = ~(inputs_ok & np.all(np.isfinite(delta), axis=1))

        r_post = _first_match(
            {
                "group_empty": group_empty,
                "zero_survival_baseline": base_zero,
                "zero_survival_last_pre": tstar_zero,
                "zero_control_increment": zero_inc_post,
                "control_survival_zero": ctrl_zero,
                "nonfinite_counterfactual": nonfinite_cf,
                "nonfinite_effect": nonfinite_eff,
            },
            _POST_FAILURE_ORDER,
            size,
        )
        r_pre = _first_match(
            {
                "group_empty": group_empty,
                "zero_survival_baseline": base_zero,
                "zero_survival_last_pre": tstar_zero,
                "zero_control_increment": zero_inc_pre,
                "nonfinite_contrast": nonfinite_delta,
            },
            _PRETEST_FAILURE_ORDER,
            size,
        )
        okp = r_post == ""
        okq = r_pre == ""
        sl = slice(start, start + size)
        ok_post[sl] = okp
        ok_pretest[sl] = okq
        reason_post[sl] = r_post
        reason_pretest[sl] = r_pre
        tau_star[sl][okp] = tau_post[okp]
        head_star[sl][okp] = head[okp]
        if len(J):
            delta_star[sl][okq] = delta[okq]
        # Diagnostic count: complete draws whose imputed curve leaves the
        # domain (finite S0 > 1 or a decreasing step) — not failures.
        if okp.any():
            R0_ok = R0[okp]
            tol_ok = _curve_tolerance(R, D, elapsed, c, method)[okp]
            with _errstate():
                bad = (R0_ok < -tol_ok).any(axis=1) | (np.diff(R0_ok, axis=1) < -tol_ok[:, 1:]).any(
                    axis=1
                )
            n_invalid_curve += int(bad.sum())

    return {
        "tau_star": tau_star,
        "head_star": head_star,
        "delta_star": delta_star,
        "ok_post": ok_post,
        "ok_pretest": ok_pretest,
        "reason_post": reason_post,
        "reason_pretest": reason_pretest,
        "n_draws_invalid_counterfactual": n_invalid_curve,
    }


def _count_reasons(reasons: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in reasons.tolist():
        if r:
            out[r] = out.get(r, 0) + 1
    return out


# =============================================================================
# Estimator
# =============================================================================


class DurationDiD(BaseEstimator):
    """Duration difference-in-differences (Deaner & Ku 2026) for absorbing outcomes.

    Two-group, common-timing design: ``treatment`` is a FIXED 0/1 group
    indicator, ``outcome`` is a binary absorbing spell-ended indicator on a
    balanced, equally spaced numeric time grid, and ``last_pre_period`` is
    the last untreated date. The counterfactual treated survival is imputed
    from the control group under a restriction on the untreated hazards:

    - ``method="cd"`` (common dynamics): the untreated hazards differ by a
      constant additive gap ``c`` (Equation 2.3); fitted as the weighted mean
      of the pre-treatment average-hazard gaps (Equations 3.2-3.4).
    - ``method="ph"`` (proportional hazards): the untreated hazards are
      proportional, ratio ``c`` (Equation 2.4); fitted as the weighted mean of
      the pre-treatment cumulative-increment ratios (Theorem 1).

    The reported effect at each post-treatment date is the absorption ATT
    ``E[Y_it - Y_it(0) | treated]`` (positive = more cumulative exit); the
    headline ``att`` is its uniform average over the post-treatment dates.
    Inference is the Appendix B whole-individual pooled bootstrap with
    centered absolute-deviation pointwise and simultaneous (max-|t|) bands,
    plus the Algorithm 2 fixed-anchor pre-treatment specification test.

    Parameters
    ----------
    method : {"cd", "ph"}, default="cd"
        Untreated-hazard restriction.
    n_bootstrap : int, default=1000
        Whole-individual bootstrap draws. ``0`` returns point estimates with
        NaN inference; otherwise at least ``2``.
    alpha : float, default=0.05
        Significance level for every band and the pretest.
    seed : int, optional
        Seed for ``numpy.random.default_rng``.

    Notes
    -----
    Identification requires binary absorbing outcomes, a fixed population,
    no anticipation before the common intervention, unaffected controls, and
    the chosen hazard restriction on the UNTREATED hazards (not on outcome
    levels). Bootstrap validity additionally assumes independence across
    individuals with arbitrary serial dependence within each history.
    Covariates, staggered adoption, censoring, survey weights and cluster
    dependence are not supported in this version. Fitting dates default to
    every eligible pre-treatment date after the baseline with equal weights;
    ``fit(pre_periods=..., pre_period_weights=...)`` selects a subset and
    nonnegative weights (see :meth:`fit`).
    """

    def __init__(
        self,
        method: str = "cd",
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ):
        _validate_method(method)
        _validate_draws(n_bootstrap)
        _validate_alpha(alpha)
        _validate_seed(seed)
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.seed = seed
        self.is_fitted_ = False
        self.results_: Optional[DurationDiDResults] = None

    # get_params/set_params come from BaseEstimator.

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        *,
        last_pre_period: Any,
        pre_periods: Optional[Sequence[Any]] = None,
        pre_period_weights: Optional[Sequence[float]] = None,
    ) -> DurationDiDResults:
        """Fit the estimator on a balanced long individual panel.

        Parameters
        ----------
        data : pd.DataFrame
            Long panel with exactly one row per (individual, date).
        outcome : str
            Binary absorbing spell-ended indicator column (0/1; once 1,
            always 1 within an individual). Baseline absorption is allowed.
        unit : str
            Individual identifier column.
        time : str
            Numeric calendar or elapsed-duration column; every individual
            must be observed at the same equally spaced dates.
        treatment : str
            Fixed 0/1 group indicator (constant within individual).
        last_pre_period : value of ``time``
            The last untreated date (``tstar``); the intervention occurs
            strictly afterwards. Never inferred from the data.
        pre_periods : list of ``time`` values, optional
            Pre-treatment dates used to fit the hazard relationship (strictly
            after the baseline, at or before ``last_pre_period``). Default:
            every eligible pre-treatment date after the baseline.
        pre_period_weights : array-like, optional
            Nonnegative weights aligned with ``pre_periods`` (normalized to
            sum to one; a zero weight drops that date). Requires
            ``pre_periods``. Default: equal weights over the eligible set.

        Returns
        -------
        DurationDiDResults
        """
        method = self.method
        arranged = _validate_and_arrange(data, outcome, unit, time, treatment, last_pre_period)
        Y: np.ndarray = arranged["Y"]
        n_treated: int = arranged["n_treated"]
        grid: np.ndarray = arranged["grid"]
        elapsed: np.ndarray = arranged["elapsed"]
        tstar_idx: int = arranged["tstar_idx"]
        n, n_periods = Y.shape
        n_control = n - n_treated
        post_idx = np.arange(tstar_idx + 1, n_periods)
        n_post = len(post_idx)
        J = np.arange(1, tstar_idx)
        messages: List[str] = []

        # ---- survival and moments (original sample) --------------------
        S_all, _, _ = _group_survival(Y, n_treated, np.ones((1, n)))
        S = S_all[0]
        for k, label in ((0, "treated"), (1, "control")):
            if S[k, 0] <= 0:
                raise ValueError(
                    f"the {label} group has zero survival at the baseline date {_label(grid, 0)!r}; "
                    "the initial survival level is unidentified"
                )
        if S[0, tstar_idx] <= 0:
            raise ValueError(
                f"the treated group is fully absorbed by last_pre_period {_label(grid, tstar_idx)!r}; "
                "the absorption ATT is identically zero and no hazard relationship can be fitted"
            )
        if S[1, tstar_idx] <= 0:
            raise ValueError(
                f"the control group is fully absorbed by last_pre_period {_label(grid, tstar_idx)!r}; "
                "no post-treatment date has control survival"
            )
        R_all, D_all, H_all = _log_survival_moments(S_all, elapsed)
        D = D_all[0]

        # ---- fitting set (frozen for every draw) -----------------------
        fit_idx, fit_w, excluded, fit_msgs = _resolve_fit_periods(
            S, D, method, grid, tstar_idx, pre_periods, pre_period_weights
        )
        messages.extend(fit_msgs)

        # ---- core estimate --------------------------------------------
        c_arr, R0_arr, S0_arr, tau_arr = _estimate_from_survival(
            S_all, R_all, D_all, elapsed, fit_idx, fit_w, method
        )
        c = float(c_arr[0])
        R0, S0, tau_all = R0_arr[0], S0_arr[0], tau_arr[0]
        ph_boundary = bool(method == "ph" and np.isfinite(c) and c == 0.0)
        if ph_boundary:
            messages.append(
                "The fitted PH ratio is exactly zero (no treated exits over the fitting "
                "dates): a boundary case outside the strict positive-hazard interpretation "
                "and the interior regularity argument behind the bootstrap."
            )

        # ---- curve validity and retained values -------------------------
        tol = _curve_tolerance(R_all, D_all, elapsed, c_arr, method)[0]
        curve_status = _curve_status(R0, S[1], tstar_idx, tol)
        period_status = curve_status[tstar_idx + 1 :]
        counterfactual = S0.astype(float).copy()
        att_by_period = tau_all[post_idx].astype(float).copy()
        for t, st in enumerate(curve_status):
            if st in ("control_survival_zero", "counterfactual_nonfinite") or not np.isfinite(
                counterfactual[t]
            ):
                counterfactual[t] = np.nan
        for j, t in enumerate(post_idx.tolist()):
            st = curve_status[t]
            if st in ("control_survival_zero", "counterfactual_nonfinite") or not np.isfinite(
                att_by_period[j]
            ):
                att_by_period[j] = np.nan
        curve_ok = all(s == "ok" for s in curve_status)
        if not curve_ok:
            messages.append(invalid_curve_message(grid, curve_status, grid[tstar_idx]))
        headline = float(np.mean(att_by_period)) if curve_ok else float("nan")

        # ---- weak support ---------------------------------------------
        used_counts = np.concatenate(
            [
                np.round(S[0, np.r_[0, tstar_idx, fit_idx]] * n_treated),
                np.round(S[1, np.r_[0, tstar_idx, fit_idx, post_idx]] * n_control),
            ]
        )
        weak = used_counts[(used_counts > 0) & (used_counts < _WEAK_SUPPORT_MIN_SURVIVORS)]
        if weak.size:
            messages.append(
                f"A used group survival is backed by fewer than {_WEAK_SUPPORT_MIN_SURVIVORS} "
                f"survivors (minimum {int(weak.min())}): weak numerical support for the "
                "log-survival moments. No cutoff-based adjustment is applied."
            )

        # ---- pretest contrasts (original sample) ----------------------
        pretest_status: Optional[str] = None
        if len(J) == 0:
            delta = np.zeros(0)
            pretest_status = "unavailable_insufficient_pre_periods"
            messages.append(
                "Only two pre-treatment dates (the baseline and last_pre_period): the "
                "hazard relationship is fitted from one moment and the Algorithm 2 "
                "pre-treatment diagnostic is unavailable (it needs an interior pre-date)."
            )
        else:
            pretest_dates = np.concatenate([J, [tstar_idx]])
            if method == "ph" and not np.all(D[1, pretest_dates] > 0):
                delta = np.full(len(J), np.nan)
                pretest_status = "unavailable_ph_support"
            else:
                delta = _pretest_contrasts(D_all, H_all, J, tstar_idx, method)[0]
                if not np.all(np.isfinite(delta)):
                    pretest_status = "unavailable_nonfinite_moments"

        # ---- bootstrap ------------------------------------------------
        n_boot = int(self.n_bootstrap)
        boot: Optional[Dict[str, Any]] = None
        if n_boot >= 2:
            rng = np.random.default_rng(self.seed)
            boot = _run_bootstrap(
                Y, n_treated, elapsed, tstar_idx, fit_idx, fit_w, method, J, n_boot, rng
            )

        # ---- post-family inference ------------------------------------
        nan_p = np.full(n_post, np.nan)
        se_by_period = nan_p.copy()
        crit_pw = nan_p.copy()
        p_pw = nan_p.copy()
        ci_pw = np.full((n_post, 2), np.nan)
        cband_lo = nan_p.copy()
        cband_hi = nan_p.copy()
        cband_crit = float("nan")
        joint_p = float("nan")
        vcov: Optional[np.ndarray] = None
        se_head = float("nan")
        p_head = float("nan")
        ci_head: Tuple[float, float] = (float("nan"), float("nan"))
        n_valid_post = int(boot["ok_post"].sum()) if boot is not None else 0
        n_valid_pre = int(boot["ok_pretest"].sum()) if boot is not None else 0

        if n_boot == 0:
            inference_status = "disabled"
        elif not curve_ok:
            inference_status = "unavailable_invalid_periods"
        elif boot is not None and not bool(boot["ok_post"].all()):
            inference_status = "unavailable_failed_draws"
        else:
            assert boot is not None
            summ = _centered_bootstrap_summary(att_by_period, boot["tau_star"], self.alpha)
            head_summ = _centered_bootstrap_summary(
                np.array([headline]), boot["head_star"][:, None], self.alpha
            )
            se_cand = np.concatenate([summ["se"], head_summ["se"]])
            if not np.all(np.isfinite(se_cand)) or np.any(se_cand <= 0):
                inference_status = "unavailable_zero_se"
            else:
                inference_status = "ok"
                se_by_period = summ["se"]
                crit_pw = summ["crit"]
                p_pw = summ["p"]
                ci_pw = np.column_stack([summ["ci_lower"], summ["ci_upper"]])
                cband_lo = summ["band_lower"]
                cband_hi = summ["band_upper"]
                cband_crit = float(summ["crit_sim"])
                joint_p = float(summ["p_joint"])
                vcov = summ["vcov"]
                se_head = float(head_summ["se"][0])
                p_head = float(head_summ["p"][0])
                ci_head = (float(head_summ["ci_lower"][0]), float(head_summ["ci_upper"][0]))

        # One safe_inference call (headline) and one safe_inference_batch call
        # (per-period): the joint-NaN gate, then the centered-bootstrap p/CI
        # override on an available family. A withheld family has NaN SEs, so
        # every column is NaN through the gate.
        t_head, p_gate, ci_gate = safe_inference(headline, se_head, alpha=self.alpha)
        t_pw, p_gate_pw, _, _ = safe_inference_batch(att_by_period, se_by_period, alpha=self.alpha)
        if inference_status == "ok":
            p_value = p_head
            conf_int = ci_head
            p_value_by_period = p_pw
            conf_int_by_period = ci_pw
        else:
            p_value = float(p_gate)
            conf_int = (float(ci_gate[0]), float(ci_gate[1]))
            p_value_by_period = np.asarray(p_gate_pw, dtype=float)
            conf_int_by_period = np.full((n_post, 2), np.nan)
            se_by_period = nan_p.copy()
            se_head = float("nan")

        # ---- pretest inference ----------------------------------------
        n_J = len(J)
        pre_se = np.full(n_J, np.nan)
        pre_lo = np.full(n_J, np.nan)
        pre_hi = np.full(n_J, np.nan)
        pre_crit = float("nan")
        pre_stat = float("nan")
        pre_p = float("nan")
        pre_reject: Optional[bool] = None
        # Precedence (first applicable label wins): disabled >
        # insufficient_pre_periods > ph_support > nonfinite_moments >
        # failed_draws > zero_se > ok.
        if n_boot == 0:
            pretest_status = "disabled"
        elif pretest_status is None:
            assert boot is not None
            if not bool(boot["ok_pretest"].all()):
                pretest_status = "unavailable_failed_draws"
            else:
                psumm = _centered_bootstrap_summary(delta, boot["delta_star"], self.alpha)
                if not np.all(np.isfinite(psumm["se"])) or np.any(psumm["se"] <= 0):
                    pretest_status = "unavailable_zero_se"
                else:
                    pretest_status = "ok"
                    pre_se = psumm["se"]
                    pre_lo = psumm["band_lower"]
                    pre_hi = psumm["band_upper"]
                    pre_crit = float(psumm["crit_sim"])
                    pre_stat = float(psumm["statistic"])
                    pre_p = float(psumm["p_joint"])
                    pre_reject = bool(pre_stat > pre_crit)

        pretest = DurationDiDPretestResults(
            method=method,
            periods=grid[J] if n_J else grid[:0],
            anchor_period=(
                grid[tstar_idx].item() if hasattr(grid[tstar_idx], "item") else grid[tstar_idx]
            ),
            contrast=np.asarray(delta, dtype=float),
            se=pre_se,
            band_lower=pre_lo,
            band_upper=pre_hi,
            crit_value=pre_crit,
            statistic=pre_stat,
            p_value=pre_p,
            reject=pre_reject,
            alpha=self.alpha,
            n_bootstrap=n_boot,
            n_bootstrap_valid=n_valid_pre,
            status=pretest_status,
        )

        # ---- failure warnings -----------------------------------------
        failure_reasons: Dict[str, Dict[str, int]] = {"post": {}, "pretest": {}}
        n_invalid_cf = 0
        boot_effects: Optional[np.ndarray] = None
        if boot is not None:
            failure_reasons = {
                "post": _count_reasons(boot["reason_post"]),
                "pretest": _count_reasons(boot["reason_pretest"]),
            }
            n_invalid_cf = int(boot["n_draws_invalid_counterfactual"])
            boot_effects = boot["tau_star"]
            if failure_reasons["post"] or failure_reasons["pretest"]:
                messages.append(
                    f"Bootstrap draws failed (post family: {failure_reasons['post']}; "
                    f"pretest family: {failure_reasons['pretest']}) out of {n_boot}. Fixed "
                    "draws without retries: any failed draw marks that family's inference "
                    "unavailable. Remedies: a larger sample, or an explicitly shorter horizon "
                    "(subset the data) when control survivors at the horizon are scarce."
                )

        for msg in messages:
            warnings.warn(msg, UserWarning, stacklevel=2)

        def unit_label(k: int) -> Any:
            return grid[k].item() if hasattr(grid[k], "item") else grid[k]

        results = DurationDiDResults(
            att=headline,
            se=se_head,
            t_stat=float(t_head),
            p_value=float(p_value),
            conf_int=conf_int,
            method=method,
            alpha=self.alpha,
            n_bootstrap=n_boot,
            n_bootstrap_valid=n_valid_post,
            n_bootstrap_valid_pretest=n_valid_pre,
            seed=self.seed,
            n_units=int(n),
            n_obs=int(n * n_periods),
            n_treated=int(n_treated),
            n_control=int(n_control),
            n_periods=int(n_periods),
            periods=grid.copy(),
            last_pre_period=unit_label(tstar_idx),
            post_periods=grid[post_idx].copy(),
            pre_periods=grid[fit_idx].copy(),
            pre_period_weights=fit_w.copy(),
            excluded_pre_periods=excluded,
            coefficient=c,
            ph_ratio_boundary=ph_boundary,
            n_treated_survivors_at_last_pre=int(round(S[0, tstar_idx] * n_treated)),
            n_control_survivors_at_horizon=int(round(S[1, -1] * n_control)),
            survival_treated=S[0].copy(),
            survival_control=S[1].copy(),
            counterfactual_survival=counterfactual,
            att_by_period=att_by_period,
            se_by_period=np.asarray(se_by_period, dtype=float),
            t_stat_by_period=np.asarray(t_pw, dtype=float),
            p_value_by_period=np.asarray(p_value_by_period, dtype=float),
            conf_int_by_period=np.asarray(conf_int_by_period, dtype=float),
            pointwise_crit_values=np.asarray(crit_pw, dtype=float),
            cband_lower=np.asarray(cband_lo, dtype=float),
            cband_upper=np.asarray(cband_hi, dtype=float),
            cband_crit_value=cband_crit,
            joint_p_value=joint_p,
            vcov=vcov,
            curve_status=list(curve_status),
            period_status=list(period_status),
            inference_status=inference_status,
            bootstrap_effects=boot_effects,
            bootstrap_failure_reasons=failure_reasons,
            n_draws_invalid_counterfactual=n_invalid_cf,
            pretest=pretest,
        )
        self.results_ = results
        self.is_fitted_ = True
        return results
