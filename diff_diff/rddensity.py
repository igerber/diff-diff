"""
Manipulation testing via local polynomial density estimation (RDDensityTest) -
parity-targeting R ``rddensity`` 3.0.

Implements the density-discontinuity (manipulation) test of Cattaneo, Jansson
& Ma (2020): the running variable's density is estimated just below and just
above the cutoff with the boundary-adaptive local polynomial density estimator
(a kernel-weighted local polynomial regression of the empirical distribution
function; the density is the slope coefficient), and the two-sided difference
is t-tested. The reported test is ROBUST BIAS-CORRECTED: the density
estimators run at order ``q = p + 1`` while the bandwidths are selected as
MSE-optimal for the order-``p`` estimator, following the CCT 2014 / CCF 2018
robust bias-correction logic. A significant discontinuity is evidence of
sorting/manipulation around the cutoff; a null result supports (but cannot
prove) the RD continuity assumption.

This is a DIAGNOSTIC (like :class:`RDPlot`), not a treatment-effect estimator:
the result carries ``t_stat``/``p_value`` and per-side density estimates, and
deliberately no ``att``/``se``/``conf_int`` - matching R ``rddensity``'s
output surface (confidence intervals live in R's ``rdplotdensity``, which is
out of scope here).

rddensity equivalents
---------------------
======================  ==========================================
diff-diff               R rddensity
======================  ==========================================
``cutoff``              ``c``
``p``, ``q``            ``p``, ``q`` (``q=None`` resolves to p+1;
                        R uses the ``q=0`` sentinel)
``fitselect``           ``fitselect`` ("unrestricted"/"restricted")
``kernel``              ``kernel`` (accepts "tri"/"epa"/"uni" too)
``vcov_type``           ``vce`` ("jackknife" default, "plugin")
``h``                   ``h`` (scalar or (h_left, h_right))
``bwselect``            ``bwselect`` (each/diff/sum/comb)
``masspoints``          ``massPoints`` - STRING surface mapping the
                        R boolean: "adjust" = TRUE, "off" = FALSE,
                        "check" = FALSE + a detection warning
``regularize``          ``regularize``
``n_local_min``         ``nLocalMin`` (None -> 20+p+1)
``n_unique_min``        ``nUniqueMin`` (None -> 20+p+1)
``report_all``          ``all``
======================  ==========================================

Not in v1 (documented seams, see REGISTRY.md): the binomial windows test
(R's ``bino=`` block; sourced from Cattaneo, Frandsen & Titiunik 2015, not
the reviewed CJM 2020 paper), ``rdplotdensity`` plots/confidence bands
(need an lpdensity port), and a public bandwidth-selector helper.

References
----------
- Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple Local Polynomial
  Density Estimators. *Journal of the American Statistical Association*,
  115(531), 1449-1455.
- Cattaneo, M. D., Jansson, M., & Ma, X. (2018). Manipulation Testing based
  on Density Discontinuity. *Stata Journal*, 18(1), 234-261.
- McCrary, J. (2008). Manipulation of the running variable in the regression
  discontinuity design: A density test. *Journal of Econometrics*, 142(2),
  698-714.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff._base import BaseEstimator
from diff_diff._rdrobust_port import _normalize_kernel
from diff_diff.results_base import Diagnostic
from diff_diff.utils import safe_inference

__all__ = [
    "RDDensityTest",
    "RDDensityTestResult",
]


# --------------------------------------------------------------------------- #
# Closed-form moment matrices (R: Sgenerate / Cgenerate / Ggenerate over
# (0, 1), plus the (p+2)-dimensional restricted "plus" embeddings and the
# Psi reflection matrix).  All production integrals run over [0, 1], where
# every supported kernel is a plain polynomial - so the integrals are exact
# polynomial integrals, not quadrature.
# --------------------------------------------------------------------------- #

_KERNEL_POLY_01: Dict[str, np.ndarray] = {
    # kernel restricted to [0, 1], ascending coefficients
    "uniform": np.array([0.5]),
    "triangular": np.array([1.0, -1.0]),  # 1 - x
    "epanechnikov": np.array([0.75, 0.0, -0.75]),  # 0.75 (1 - x^2)
}


def _poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(a, b)


def _poly_antiderivative(a: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a) + 1)
    out[1:] = a / np.arange(1, len(a) + 1)
    return out


def _poly_eval(a: np.ndarray, x: float) -> float:
    return float(np.polyval(a[::-1], x))


def _poly_integral_01(a: np.ndarray) -> float:
    anti = _poly_antiderivative(a)
    return _poly_eval(anti, 1.0) - _poly_eval(anti, 0.0)


def _monomial(power: int) -> np.ndarray:
    out = np.zeros(power + 1)
    out[power] = 1.0
    return out


def _moment_s(p: int, kernel: str) -> np.ndarray:
    """R ``Sgenerate(p, low=0, up=1)``: S[i, j] = int_0^1 x^(i+j) k(x) dx
    (0-indexed i, j)."""
    k = _KERNEL_POLY_01[kernel]
    out = np.empty((p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            out[i, j] = _poly_integral_01(_poly_mul(_monomial(i + j), k))
    return out


def _moment_c(k_power: int, p: int, kernel: str) -> np.ndarray:
    """R ``Cgenerate(k, p, low=0, up=1)``: C[i] = int_0^1 x^(i+k) k(x) dx."""
    kpoly = _KERNEL_POLY_01[kernel]
    out = np.empty(p + 1)
    for i in range(p + 1):
        out[i] = _poly_integral_01(_poly_mul(_monomial(i + k_power), kpoly))
    return out


def _moment_g(p: int, kernel: str) -> np.ndarray:
    """R ``Ggenerate(p, low=0, up=1)``:

    G[i, j] = int_0^1 k(y) y^j     [int_0^y x^(i+1) k(x) dx] dy
            + int_0^1 k(y) y^(j+1) [int_y^1 x^i     k(x) dx] dy

    (0-indexed translation of R's 1-indexed powers).  Exact polynomial
    integration: the inner integrals are polynomials in ``y``.
    """
    kpoly = _KERNEL_POLY_01[kernel]
    out = np.empty((p + 1, p + 1))
    for i in range(p + 1):
        inner_low = _poly_antiderivative(_poly_mul(_monomial(i + 1), kpoly))
        inner_low = inner_low - _poly_eval(inner_low, 0.0)  # int_0^y
        inner_up_anti = _poly_antiderivative(_poly_mul(_monomial(i), kpoly))
        # int_y^1 = F(1) - F(y): constant minus polynomial in y
        inner_up = -inner_up_anti
        inner_up[0] += _poly_eval(inner_up_anti, 1.0)
        for j in range(p + 1):
            term1 = _poly_mul(_poly_mul(_monomial(j), kpoly), inner_low)
            term2 = _poly_mul(_poly_mul(_monomial(j + 1), kpoly), inner_up)
            out[i, j] = _poly_integral_01(term1) + _poly_integral_01(term2)
    return out


def _embed_plus(mat: np.ndarray, p: int) -> np.ndarray:
    """R ``Splusgenerate``/``Gplusgenerate``: embed a (p+1)x(p+1) matrix over
    (0,1) into the (p+2)-dimensional restricted basis (slope slot 2 is the
    minus-side slot and stays zero)."""
    out = np.zeros((p + 2, p + 2))
    out[0, 0] = mat[0, 0]
    out[0, 2:] = mat[0, 1:]
    out[2:, 0] = mat[1:, 0]
    out[2:, 2:] = mat[1:, 1:]
    return out


def _embed_plus_vec(vec: np.ndarray, p: int) -> np.ndarray:
    """R ``Cplusgenerate``: vector analogue of :func:`_embed_plus`."""
    out = np.zeros(p + 2)
    out[0] = vec[0]
    out[2:] = vec[1:]
    return out


def _psi_matrix(p: int) -> np.ndarray:
    """R ``Psigenerate(p)``: the reflection matrix mapping plus-side
    restricted-basis matrices to the minus side."""
    diag = np.zeros(p + 2)
    diag[0] = 1.0
    if p > 1:
        diag[3:] = (-1.0) ** np.arange(2, p + 1)
    out = np.diag(diag)
    out[1, 2] = out[2, 1] = -1.0
    return out


# --------------------------------------------------------------------------- #
# Unique-value bookkeeping (R: rddensityUnique / rddensityHasRepeated)
# --------------------------------------------------------------------------- #


def _rddensity_unique(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Unique values of an ASCENDING-sorted vector with frequencies and the
    first/last occurrence indices (0-based), matching R's helper."""
    n = len(x)
    if n == 0:
        return {
            "unique": x,
            "freq": np.array([], dtype=int),
            "index_first": np.array([], dtype=int),
            "index_last": np.array([], dtype=int),
        }
    is_last = np.append(x[1:] != x[:-1], True)
    index_last = np.flatnonzero(is_last)
    index_first = np.concatenate(([0], index_last[:-1] + 1))
    freq = np.diff(np.concatenate(([-1], index_last)))
    return {
        "unique": x[is_last],
        "freq": freq,
        "index_first": index_first,
        "index_last": index_last,
    }


def _has_repeated(x: np.ndarray) -> bool:
    return len(x) > 1 and bool(np.any(x[1:] == x[:-1]))


# --------------------------------------------------------------------------- #
# Kernel constants for the normal-reference preliminary bandwidths
# (R: hard-coded Cb / Cc vectors, indexed by p = 1..7) and the Hermite
# polynomials H_p used by the normal-reference derivative plug-ins.
# --------------------------------------------------------------------------- #

_CB = np.array(
    [
        25884.444444494150957,
        3430865.4551236177795,
        845007948.04262602329,
        330631733667.03808594,
        187774809656037.3125,
        145729502641999264.0,
        146013502974449876992.0,
    ]
)
_CC = np.array(
    [
        4.8000000000000246914,
        548.57142857155463389,
        100800.00000020420703,
        29558225.458100609481,
        12896196859.612621307,
        7890871468221.609375,
        6467911284037581.0,
    ]
)

_HERMITE = {
    0: lambda x: 1.0,
    1: lambda x: x,
    2: lambda x: x**2 - 1,
    3: lambda x: x**3 - 3 * x,
    4: lambda x: x**4 - 6 * x**2 + 3,
    5: lambda x: x**5 - 10 * x**3 + 15 * x,
    6: lambda x: x**6 - 15 * x**4 + 45 * x**2 - 15,
    7: lambda x: x**7 - 21 * x**5 + 105 * x**3 - 105 * x,
    8: lambda x: x**8 - 28 * x**6 + 210 * x**4 - 420 * x**2 + 105,
    9: lambda x: x**9 - 36 * x**7 + 378 * x**5 - 1260 * x**3 + 945 * x,
}


# --------------------------------------------------------------------------- #
# Core estimation (R: rddensity_fV)
# --------------------------------------------------------------------------- #


def _check_unique_support(x_window: np.ndarray, order: int, w: np.ndarray) -> None:
    """Fail-loud design guard: each side of the (already cutoff-centered)
    window must carry at least ``order + 1`` UNIQUE running-variable values
    with POSITIVE kernel weight - a polynomial basis on that many distinct
    nodes is full rank, so this is a complete (and threshold-free) rank
    precondition.  Counting zero-weight values would over-count: the
    triangular and epanechnikov kernels assign weight exactly 0 at the
    bandwidth boundary ``|x| = h`` (which the regularization range clamp
    hits exactly), and such points contribute no identifying information.
    R instead runs ``solve(..., tol=0)`` and silently returns numerically
    meaningless output on such designs (documented Deviation from R)."""
    for side, mask in (("left", x_window < 0), ("right", x_window >= 0)):
        n_unique = len(np.unique(x_window[mask & (w > 0)]))
        if n_unique < order + 1:
            raise ValueError(
                f"Local design is degenerate: the {side} side of the "
                f"estimation window contains {n_unique} unique running-"
                f"variable value(s) with positive kernel weight, fewer than "
                f"the {order + 1} required by the order-{order} polynomial "
                "basis. Likely causes: a side of (almost) only repeated "
                "values, a bandwidth too small for the local support, or "
                "support points sitting exactly at the bandwidth boundary "
                "(zero kernel weight). R rddensity silently returns "
                "numerically meaningless estimates here; diff-diff raises "
                "instead."
            )


def _restricted_plugin_variance(
    f_l: float, f_r: float, p: int, kernel: str, n: int, h: float
) -> Tuple[float, float, float, float]:
    """Restricted-model plugin variance (R's Psi-coupled sandwich).

    Returns ``(v_left, v_right, v_diff, v_sum)``.  On a singular combined
    Gram (exactly reachable at ``f_side == 0``) or non-finite output, all
    four are NaN and a warning is emitted (degraded-inference contract:
    point estimates stand, inference degrades).
    """
    s_plus = _embed_plus(_moment_s(p, kernel), p)
    g_plus = _embed_plus(_moment_g(p, kernel), p)
    psi = _psi_matrix(p)
    s_minus = psi @ s_plus @ psi
    g_minus = psi @ g_plus @ psi
    try:
        sandwich_inv = np.linalg.inv(f_l * s_minus + f_r * s_plus)
        v = sandwich_inv @ (f_l**3 * g_minus + f_r**3 * g_plus) @ sandwich_inv
        v_left = v[1, 1] / (n * h)
        v_right = v[2, 2] / (n * h)
        v_diff = (v[1, 1] + v[2, 2] - 2 * v[1, 2]) / (n * h)
        v_sum = (v[1, 1] + v[2, 2] + 2 * v[1, 2]) / (n * h)
        values = (v_left, v_right, v_diff, v_sum)
        if not all(np.isfinite(val) for val in values):
            raise np.linalg.LinAlgError("non-finite plugin variance")
        return values
    except np.linalg.LinAlgError:
        warnings.warn(
            "Restricted-model plugin variance is degenerate (the density-"
            "weighted combined Gram matrix is singular or produced non-"
            "finite output, e.g. a (near-)zero one-sided density). Point "
            "estimates are reported; the affected standard errors and "
            "inference are NaN.",
            UserWarning,
            stacklevel=2,
        )
        return (np.nan, np.nan, np.nan, np.nan)


def _rddensity_fv(
    y: np.ndarray,
    x: np.ndarray,
    n: int,
    n_lh: int,
    n_rh: int,
    hl: float,
    hr: float,
    p: int,
    s: int,
    kernel: str,
    fitselect: str,
    vcov_type: str,
    masspoints_flag: bool,
) -> np.ndarray:
    """Port of R ``rddensity_fV``: point estimates plus jackknife/plugin
    variances for (left, right, diff, sum).

    Returns a 4x4 array with rows ``l, r, diff, sum`` and columns
    ``hat, jackknife, plugin, s`` (NaN where not computed), exactly R's
    layout.  ``x`` is the cutoff-centered, ascending window sample; ``y``
    the matching EDF values; ``n`` the FULL sample size.
    """
    n_h = n_lh + n_rh

    # kernel weights (per side, already 1/h scaled as in R)
    w = np.empty(n_h)
    left = np.arange(n_h) < n_lh
    if kernel == "uniform":
        w[left] = 1.0 / (2.0 * hl)
        w[~left] = 1.0 / (2.0 * hr)
    elif kernel == "triangular":
        w[left] = (1.0 + x[left] / hl) / hl
        w[~left] = (1.0 - x[~left] / hr) / hr
    else:  # epanechnikov
        w[left] = 0.75 * (1.0 - (x[left] / hl) ** 2) / hl
        w[~left] = 0.75 * (1.0 - (x[~left] / hr) ** 2) / hr

    # design guard AFTER the weights exist: the rank precondition counts
    # only positive-weight support points
    _check_unique_support(x, p, w)

    # design matrix + bandwidth scaling matrix
    if fitselect == "restricted":
        xp = np.zeros((n_h, p + 2))
        xp[:, 0] = 1.0
        xp[left, 1] = x[left] / hl
        xp[~left, 2] = x[~left] / hr
        if p > 1:
            for j in range(3, p + 2):  # R columns 4..p+2 -> powers 2..p
                power = j - 1
                xp[left, j] = (x[left] / hl) ** power
                xp[~left, j] = (x[~left] / hr) ** power
            v_exp = np.concatenate(([0, 1, 1], np.arange(2, p + 1)))
        else:
            v_exp = np.array([0, 1, 1])
        hp_inv = np.diag(1.0 / hl**v_exp)
    else:
        xp = np.zeros((n_h, 2 * p + 2))
        hp_diag = np.empty(2 * p + 2)
        for j in range(1, 2 * p + 3):  # R's 1-based j
            if j % 2 == 1:
                power = (j - 1) // 2
                xp[left, j - 1] = (x[left] / hl) ** power
                hp_diag[j - 1] = hl**power
            else:
                power = (j - 2) // 2
                xp[~left, j - 1] = (x[~left] / hr) ** power
                hp_diag[j - 1] = hr**power
        hp_inv = np.diag(1.0 / hp_diag)

    out = np.full((4, 4), np.nan)

    xpw = xp * w[:, None]
    try:
        s_inv = np.linalg.inv(xpw.T @ xp)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Local design matrix is numerically singular; the estimation "
            "window cannot identify the local polynomial fit. Try a larger "
            "bandwidth (h=) or inspect the running variable for degenerate "
            "local support."
        ) from exc

    b = hp_inv @ s_inv @ (xpw.T @ y)
    if not np.all(np.isfinite(b)):
        raise ValueError(
            "Local polynomial solve produced non-finite estimates; the "
            "estimation window is numerically degenerate."
        )

    if fitselect == "restricted":
        out[0, 0] = b[1]
        out[1, 0] = b[2]
        out[2, 0] = b[2] - b[1]
        out[3, 0] = b[2] + b[1]
        out[0, 3] = out[1, 3] = b[s + 1]
        out[2, 3] = 0.0
        out[3, 3] = 2.0 * out[0, 3]
    else:
        out[0, 0] = b[2]
        out[1, 0] = b[3]
        out[2, 0] = b[3] - b[2]
        out[3, 0] = b[3] + b[2]
        out[0, 3] = b[2 * s]
        out[1, 3] = b[2 * s + 1]
        out[2, 3] = out[1, 3] - out[0, 3]
        out[3, 3] = out[1, 3] + out[0, 3]

    if vcov_type == "jackknife":
        # leave-one-out projection: L[i, :] = sum_{k > i} xpw[k, :] / (n - 1)
        # (window rows only - R's construction; differs from the SA literal
        # double sum, documented Note)
        csum = np.vstack([np.zeros(xp.shape[1]), np.cumsum(xpw[::-1], axis=0)])
        base_l = csum[n_h - 1 :: -1] / (n - 1)
        if masspoints_flag:
            uinfo = _rddensity_unique(x)
            l_mat = np.repeat(base_l[uinfo["index_first"]], uinfo["freq"], axis=0)
        else:
            l_mat = base_l
        v = hp_inv @ s_inv @ (l_mat.T @ l_mat) @ s_inv @ hp_inv
        if fitselect == "restricted":
            out[0, 1] = v[1, 1]
            out[1, 1] = v[2, 2]
            out[2, 1] = v[1, 1] + v[2, 2] - 2 * v[1, 2]
            out[3, 1] = v[1, 1] + v[2, 2] + 2 * v[1, 2]
        else:
            out[0, 1] = v[2, 2]
            out[1, 1] = v[3, 3]
            out[2, 1] = v[2, 2] + v[3, 3] - 2 * v[2, 3]
            out[3, 1] = v[2, 2] + v[3, 3] + 2 * v[2, 3]

    if vcov_type == "plugin":
        if fitselect == "unrestricted":
            s_mat = _moment_s(p, kernel)
            g_mat = _moment_g(p, kernel)
            s_mat_inv = np.linalg.inv(s_mat)
            v = s_mat_inv @ g_mat @ s_mat_inv
            out[0, 2] = out[0, 0] * v[1, 1] / (n * hl)
            out[1, 2] = out[1, 0] * v[1, 1] / (n * hr)
            out[2, 2] = out[3, 2] = out[0, 2] + out[1, 2]
        else:
            out[0, 2], out[1, 2], out[2, 2], out[3, 2] = _restricted_plugin_variance(
                float(out[0, 0]), float(out[1, 0]), p, kernel, n, hl
            )

    # R: negative variances -> NA
    for i in range(4):
        for j in (1, 2):
            if np.isfinite(out[i, j]) and out[i, j] < 0:
                out[i, j] = np.nan

    return out


# --------------------------------------------------------------------------- #
# Bandwidth selection (R: rdbwdensity)
# --------------------------------------------------------------------------- #


def _rdbwdensity(
    x_sorted: np.ndarray,
    cutoff: float,
    p: int,
    kernel: str,
    fitselect: str,
    vcov_type: str,
    regularize: bool,
    n_local_min: int,
    n_unique_min: int,
    masspoints_flag: bool,
    unique_info: Dict[str, np.ndarray],
) -> np.ndarray:
    """Port of R ``rdbwdensity``: the 4x3 h-table (rows l/r/diff/sum,
    columns bw/variance/bias_sq). ``x_sorted`` is the ascending FULL
    sample (uncentered); ``unique_info`` its :func:`_rddensity_unique`."""
    n = len(x_sorted)
    n_left = int(np.sum(x_sorted < cutoff))
    n_right = n - n_left

    x = x_sorted - cutoff
    x_mu = float(np.mean(x))
    x_sd = float(np.std(x, ddof=1))
    x_unique = unique_info["unique"] - cutoff
    n_unique_left = int(np.sum(x_unique < 0))
    n_unique_right = len(x_unique) - n_unique_left

    # normal-reference preliminary bandwidths
    z = x_mu / x_sd
    fhatb = 1.0 / (_HERMITE[p + 2](z) ** 2 * stats.norm.pdf(z))
    fhatc = 1.0 / (_HERMITE[p](z) ** 2 * stats.norm.pdf(z))
    bn = ((2 * p + 1) / 4.0 * fhatb * _CB[p - 1] / n) ** (1.0 / (2 * p + 5))
    cn = (1.0 / (2 * p) * fhatc * _CC[p - 1] / n) ** (1.0 / (2 * p + 1))
    bn *= x_sd
    cn *= x_sd

    abs_left_sorted = np.sort(np.abs(x[x < 0]))
    right_vals = x[x >= 0]
    abs_left_unique_sorted = np.sort(np.abs(x_unique[x_unique < 0]))
    right_unique_vals = x_unique[x_unique >= 0]

    if regularize:
        bn = min(bn, float(np.max(np.abs(x_unique))))
        cn = min(cn, float(np.max(np.abs(x_unique))))
        # preliminary-stage floors: HARD-CODED counts (20+p+3 for bn,
        # 20+p+1 for cn) inside each gate; the nLocalMin gate quantiles the
        # full sample, the nUniqueMin gate the unique values (R quirk,
        # replicated exactly)
        if n_local_min > 0:
            bn = max(
                bn,
                abs_left_sorted[min(20 + p + 3, n_left) - 1],
                right_vals[min(20 + p + 3, n_right) - 1],
            )
            cn = max(
                cn,
                abs_left_sorted[min(20 + p + 1, n_left) - 1],
                right_vals[min(20 + p + 1, n_right) - 1],
            )
        if n_unique_min > 0:
            bn = max(
                bn,
                abs_left_unique_sorted[min(20 + p + 3, n_unique_left) - 1],
                right_unique_vals[min(20 + p + 3, n_unique_right) - 1],
            )
            cn = max(
                cn,
                abs_left_unique_sorted[min(20 + p + 1, n_unique_left) - 1],
                right_unique_vals[min(20 + p + 1, n_unique_right) - 1],
            )

    # EDF (rank-based, mass-point adjusted)
    y = np.arange(n) / (n - 1)
    if masspoints_flag:
        y = np.repeat(y[unique_info["index_last"]], unique_info["freq"])

    win_b = np.abs(x) <= bn
    win_c = np.abs(x) <= cn
    yb, xb = y[win_b], x[win_b]
    yc, xc = y[win_c], x[win_c]
    n_lb = int(np.sum(xb < 0))
    n_lc = int(np.sum(xc < 0))

    fv_b = _rddensity_fv(
        yb,
        xb,
        n,
        n_lb,
        len(xb) - n_lb,
        bn,
        bn,
        p + 2,
        p + 1,
        kernel,
        fitselect,
        vcov_type,
        masspoints_flag,
    )
    fv_c = _rddensity_fv(
        yc,
        xc,
        n,
        n_lc,
        len(xc) - n_lc,
        cn,
        cn,
        p,
        1,
        kernel,
        fitselect,
        vcov_type,
        masspoints_flag,
    )

    hn = np.full((4, 3), np.nan)
    var_col = 2 if vcov_type == "plugin" else 1
    hn[:, 1] = n * cn * fv_c[:, var_col]

    # R's cleanup loop then evaluates `if (hn[i,2] < 0)` with NA for ANY
    # NaN-variance row and errors opaquely; the port raises descriptively
    # HERE, before the NaN->0 cleanup and the regularization floors could
    # rescue such a row into a positive bandwidth whose MSE objective is
    # undefined (fail-loud deviation, REGISTRY)
    if not np.all(np.isfinite(hn[:, 1])):
        bad = [
            name
            for name, v in zip(("left", "right", "diff", "sum"), hn[:, 1])
            if not np.isfinite(v)
        ]
        raise ValueError(
            "Bandwidth selection failed: the pilot variance estimate is "
            f"undefined (negative or non-finite) for the {', '.join(bad)} "
            "row(s) of the selector objective; the local pilot fit is "
            "degenerate near the cutoff. Supply manual bandwidths via h= "
            "to bypass data-driven selection."
        )

    if fitselect == "unrestricted":
        s_mat = _moment_s(p, kernel)
        c_vec = _moment_c(p + 1, p, kernel)
        sc = np.linalg.solve(s_mat, c_vec)
        hn[0, 2] = fv_b[0, 3] * sc[1] * (-1.0) ** p
        hn[1, 2] = fv_b[1, 3] * sc[1]
        hn[2, 2] = hn[1, 2] - hn[0, 2]
        hn[3, 2] = hn[1, 2] + hn[0, 2]
    else:
        s_plus = _embed_plus(_moment_s(p, kernel), p)
        c_plus = _embed_plus_vec(_moment_c(p + 1, p, kernel), p)
        psi = _psi_matrix(p)
        s_inv = np.linalg.inv(fv_c[1, 0] * s_plus + fv_c[0, 0] * psi @ s_plus @ psi)
        c_comb = fv_b[0, 3] * (fv_c[1, 0] * c_plus + (-1.0) ** (p + 1) * fv_c[0, 0] * psi @ c_plus)
        temp = s_inv @ c_comb
        hn[0, 2] = temp[1]
        hn[1, 2] = temp[2]
        hn[2, 2] = hn[1, 2] - hn[0, 2]
        hn[3, 2] = hn[1, 2] + hn[0, 2]

    hn[:, 2] = hn[:, 2] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        hn[:, 0] = (1.0 / (2 * p) * hn[:, 1] / hn[:, 2] / n) ** (1.0 / (2 * p + 1))

    # R's cleanup loop: ONLY NaN bandwidths -> 0 (`is.na(hn[i,1])`); an
    # infinite bandwidth (bias-squared exactly 0, e.g. the restricted sum
    # row under exact per-side cancellation) passes through, exactly as in
    # R, and is resolved by the range clamp below when regularize=True or
    # by the caller's finite-and-positive guard otherwise. (R's
    # negative-variance branch is dead code because _rddensity_fv already
    # NaNs negatives.)
    for i in range(4):
        if np.isnan(hn[i, 0]):
            hn[i, 0] = 0.0

    if regularize:
        hn[0, 0] = min(hn[0, 0], abs(x_unique[0]))
        hn[1, 0] = min(hn[1, 0], x_unique[-1])
        hn[2, 0] = min(hn[2, 0], max(abs(x_unique[0]), x_unique[-1]))
        hn[3, 0] = min(hn[3, 0], max(abs(x_unique[0]), x_unique[-1]))
        if n_local_min > 0:
            hl_min = abs_left_sorted[min(n_left, n_local_min) - 1]
            hr_min = right_vals[min(n_right, n_local_min) - 1]
            hn[0, 0] = max(hn[0, 0], hl_min)
            hn[1, 0] = max(hn[1, 0], hr_min)
            hn[2, 0] = max(hn[2, 0], hl_min, hr_min)
            hn[3, 0] = max(hn[3, 0], hl_min, hr_min)
        if n_unique_min > 0:
            hl_min = abs_left_unique_sorted[min(n_unique_left, n_unique_min) - 1]
            hr_min = right_unique_vals[min(n_unique_right, n_unique_min) - 1]
            hn[0, 0] = max(hn[0, 0], hl_min)
            hn[1, 0] = max(hn[1, 0], hr_min)
            hn[2, 0] = max(hn[2, 0], hl_min, hr_min)
            hn[3, 0] = max(hn[3, 0], hl_min, hr_min)

    return hn


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #


@dataclass
class RDDensityTestResult(Diagnostic):
    """Results of the CJM 2020 manipulation (density-discontinuity) test.

    The headline ``t_stat``/``p_value`` are the ROBUST BIAS-CORRECTED test
    (order-``q`` density estimators at order-``p``-optimal bandwidths), with
    the standard error selected by ``vcov_type`` - when ``q > p``; a fit
    with ``q == p`` makes the headline the CONVENTIONAL (uncorrected) test,
    which over-rejects at data-driven bandwidths (see the estimator class
    docstring).  The ``*_conventional`` fields carry the same caveat.  The ``f_*`` fields are
    WHOLE-SAMPLE-scale densities exactly as R's ``hat$left/right/diff``
    report them (joint-basis estimates).  Per-side CONDITIONAL densities are
    these times ``(N-1)/(n_left-1)`` and ``(N-1)/(n_right-1)`` - the exact
    conversion under the rank-based EDF ``(0:(N-1))/(N-1)`` that R (and this
    port) implement; the CJM 2020 paper's ``n/n_side`` factors apply to the
    paper's ``(1/n)*sum`` EDF, which is not what R computes.

    There are deliberately no ``att``/``se``/``conf_int`` fields: R
    ``rddensity`` reports no confidence interval anywhere in its output
    (intervals belong to ``rdplotdensity``), and the estimand vocabulary of
    the treatment-effect estimators does not apply to a density test.
    """

    # headline (robust bias-corrected) block
    t_stat: float
    p_value: float
    f_left: float
    f_right: float
    f_diff: float
    se_left: float
    se_right: float
    se_diff: float
    # conventional (order-p) block, populated when report_all=True
    f_left_conventional: Optional[float]
    f_right_conventional: Optional[float]
    f_diff_conventional: Optional[float]
    se_left_conventional: Optional[float]
    se_right_conventional: Optional[float]
    se_diff_conventional: Optional[float]
    t_stat_conventional: Optional[float]
    p_value_conventional: Optional[float]
    # samples and bandwidths
    n: int
    n_left: int
    n_right: int
    n_eff_left: int
    n_eff_right: int
    h_left: float
    h_right: float
    bandwidths: Optional[pd.DataFrame]
    # configuration echo
    cutoff: float
    p: int
    q: int
    fitselect: str
    kernel: str
    vcov_type: str
    bwselect: str
    bandwidth_method: str
    masspoints: str
    masspoints_adjusted: bool
    regularize: bool
    n_local_min: int
    n_unique_min: int
    report_all: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy per-side table: rows ``left``/``right``/``diff`` (plus the
        ``*_conventional`` rows when ``report_all=True``); columns
        ``estimate``/``se``/``t_stat``/``p_value`` (inference populated on
        the ``diff`` rows only)."""
        rows: Dict[
            str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
        ] = {
            "left": (self.f_left, self.se_left, np.nan, np.nan),
            "right": (self.f_right, self.se_right, np.nan, np.nan),
            "diff": (self.f_diff, self.se_diff, self.t_stat, self.p_value),
        }
        if self.report_all:
            rows["left_conventional"] = (
                self.f_left_conventional,
                self.se_left_conventional,
                np.nan,
                np.nan,
            )
            rows["right_conventional"] = (
                self.f_right_conventional,
                self.se_right_conventional,
                np.nan,
                np.nan,
            )
            rows["diff_conventional"] = (
                self.f_diff_conventional,
                self.se_diff_conventional,
                self.t_stat_conventional,
                self.p_value_conventional,
            )
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=["estimate", "se", "t_stat", "p_value"]
        )

    def summary(self) -> str:
        """R-style summary block."""
        width = 72
        lines = [
            "=" * width,
            "Manipulation Test - Local Polynomial Density (rddensity parity)".center(width),
            "=" * width,
            f"Number of obs:        {self.n}",
            f"Model:                {self.fitselect}",
            f"Kernel:               {self.kernel}",
            "Bandwidth method:     "
            + (self.bandwidth_method if self.bandwidth_method == "manual" else self.bwselect),
            f"VCE method:           {self.vcov_type}",
            "",
            f"Cutoff c = {self.cutoff:g}" f"{'Left of c':>25}{'Right of c':>20}",
            f"{'Number of obs':<22}{self.n_left:>14}{self.n_right:>20}",
            f"{'Eff. number of obs':<22}{self.n_eff_left:>14}{self.n_eff_right:>20}",
            f"{'Order est. (p)':<22}{self.p:>14}{self.p:>20}",
            f"{'Order bias (q)':<22}{self.q:>14}{self.q:>20}",
            f"{'BW est. (h)':<22}{self.h_left:>14.4f}{self.h_right:>20.4f}",
            "",
            f"{'Method':<22}{'T':>14}{'P > |T|':>20}",
            "-" * width,
        ]
        # R labels the headline row "Robust" unconditionally, even when
        # q == p makes it the conventional (uncorrected) test, and with
        # all=TRUE prints the identical conventional statistic twice; the
        # port labels by the actual procedure and suppresses the duplicate
        # row (documented Deviation from R - display only)
        headline_label = "Robust" if self.q > self.p else "Conventional"
        if self.report_all and self.t_stat_conventional is not None and self.q > self.p:
            lines.append(
                f"{'Conventional':<22}{self.t_stat_conventional:>14.4f}"
                f"{self.p_value_conventional:>20.4f}"
            )
        lines.append(f"{headline_label:<22}{self.t_stat:>14.4f}{self.p_value:>20.4f}")
        lines.append("=" * width)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public estimator
# --------------------------------------------------------------------------- #


class RDDensityTest(BaseEstimator):
    """Manipulation testing via local polynomial density estimation
    (rddensity 3.0 parity).

    Tests the null of NO discontinuity in the running variable's density at
    the cutoff (McCrary 2008's question) with the boundary-adaptive local
    polynomial density estimator of Cattaneo, Jansson & Ma (2020) and robust
    bias-corrected inference: density estimators of order ``q = p + 1`` are
    evaluated at bandwidths selected as MSE-optimal for the order-``p``
    estimator.  Defaults reproduce R ``rddensity(X)``: ``p=2``, triangular
    kernel, jackknife variance, unrestricted model, ``bwselect="comb"``,
    mass-point adjustment on.

    Assumptions (CJM 2020 Assumption 1, held separately on each side of
    the cutoff): the observations are an i.i.d. random sample, and the
    running variable's distribution function is smooth with a POSITIVE
    density near the cutoff on each side.  The jackknife/plugin variances
    assume cross-observation independence - clustered, survey-weighted, or
    repeated-observation dependence is NOT supported by the reviewed
    theory and there is no ``cluster=`` surface here.  Interpret
    non-rejection as "no density discontinuity detected", which supports
    but never proves the RD continuity assumptions; a significant test is
    evidence of sorting around the cutoff.

    Parameters
    ----------
    cutoff : float, default 0.0
        The threshold ``c`` in the support of the running variable.
    p : int, default 2
        Local polynomial order for the density point estimators (1..7,
        R's accepted surface).
    q : int or None, default None
        Order for the bias-corrected estimators; ``None`` resolves to
        ``p + 1`` at fit time (R uses a ``q=0`` sentinel).  An explicit
        ``q`` must satisfy ``q >= p``; ``q == p`` gives the conventional
        (uncorrected) test, which OVER-REJECTS at data-driven MSE-optimal
        bandwidths (CJM 2020 Section 4) - valid conventional inference
        requires a manually undersmoothed ``h``, the caller's
        responsibility (a fit-time warning fires when ``h`` is data-driven).
    fitselect : str, default "unrestricted"
        ``"unrestricted"``: separate model on each side.  ``"restricted"``:
        equal distribution function and higher-order derivatives across the
        cutoff (only the density may jump); requires equal bandwidths and
        forbids ``bwselect="each"``.
    kernel : str, default "triangular"
        "triangular", "epanechnikov", or "uniform" (R spellings
        "tri"/"epa"/"uni" accepted).
    vcov_type : str, default "jackknife"
        Variance estimator: "jackknife" (R's default; the SA Section 5.2
        leave-one-out construction, window-restricted as in R) or "plugin"
        (asymptotic plug-in).
    h : float or (float, float) or None, default None
        Manual bandwidth(s).  A scalar applies to both sides; a pair is
        ``(h_left, h_right)``.  ``None`` selects data-driven bandwidths.
    bwselect : str, default "comb"
        Bandwidth selection: "each" (per-side MSE), "diff" (MSE of the
        difference), "sum" (MSE of the sum), "comb" (median of each/diff/sum
        per side for the unrestricted model; min of diff/sum for the
        restricted model).  Ignored when ``h`` is supplied.
    masspoints : str, default "adjust"
        Mass-point handling: "adjust" (R ``massPoints=TRUE`` - the EDF is
        computed on unique values and replicated back, with a warning),
        "check" (detect repeated values and warn WITHOUT adjusting -
        estimation matches R ``massPoints=FALSE``), "off" (no detection).
        String surface matching the RD-family ``masspoints`` domain
        (documented Deviation from R's boolean).
    regularize : bool, default True
        Local sample-size regularization of the data-driven bandwidths.
    n_local_min : real or None, default None -> 20 + p + 1
        Minimum observations in each local neighborhood (final-stage
        floor).  ``ceil()`` is applied as in R; ``ceil(value) >= 0``
        required, 0 disables the gate.  Must be finite (Deviation from R,
        which accepts Inf).
    n_unique_min : real or None, default None -> 20 + p + 1
        Minimum unique observations in each local neighborhood; same
        contract as ``n_local_min``.
    report_all : bool, default False
        Also compute and report the conventional (order-``p``) test
        alongside the robust one (R's ``all=TRUE``).  The conventional
        test's inference carries the same over-rejection caveat as
        ``q == p`` at data-driven bandwidths (see ``q`` above).

    Examples
    --------
    >>> test = RDDensityTest(cutoff=0.0)
    >>> result = test.fit(df, running="score")
    >>> result.t_stat, result.p_value  # robust bias-corrected test
    """

    def __init__(
        self,
        cutoff: float = 0.0,
        p: int = 2,
        q: Optional[int] = None,
        fitselect: str = "unrestricted",
        kernel: str = "triangular",
        vcov_type: str = "jackknife",
        h: Union[None, float, Tuple[float, float]] = None,
        bwselect: str = "comb",
        masspoints: str = "adjust",
        regularize: bool = True,
        n_local_min: Optional[float] = None,
        n_unique_min: Optional[float] = None,
        report_all: bool = False,
    ):
        self.cutoff = cutoff
        self.p = p
        self.q = q
        self.fitselect = fitselect
        self.kernel = kernel
        self.vcov_type = vcov_type
        self.h = h
        self.bwselect = bwselect
        self.masspoints = masspoints
        self.regularize = regularize
        self.n_local_min = n_local_min
        self.n_unique_min = n_unique_min
        self.report_all = report_all
        self._validate_constructor_args()

    # get_params/set_params come from BaseEstimator.

    @staticmethod
    def _is_real_scalar(value: Any) -> bool:
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and np.isfinite(value)
        )

    @classmethod
    def _check_bandwidth_pair(
        cls, value: Any, name: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Scalar-or-pair bandwidth surface (RDPlot's `_check_pair`
        pattern): returns (h_left, h_right) or (None, None)."""
        if value is None:
            return (None, None)
        if cls._is_real_scalar(value):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}.")
            return (float(value), float(value))
        if isinstance(value, (tuple, list)) and len(value) == 2:
            lo, hi = value
            if not (cls._is_real_scalar(lo) and cls._is_real_scalar(hi)):
                raise ValueError(f"{name} entries must be finite real scalars; got {value!r}.")
            if lo <= 0 or hi <= 0:
                raise ValueError(f"{name} entries must be positive; got {value!r}.")
            return (float(lo), float(hi))
        raise ValueError(
            f"{name} must be None, a positive scalar, or a length-2 " f"sequence; got {value!r}."
        )

    @classmethod
    def _resolve_floor(cls, value: Any, name: str) -> Optional[int]:
        """R's nLocalMin/nUniqueMin acceptance: any finite real with
        ceil(value) >= 0 (R rejects only ceiling(v) < 0; -0.5 resolves to
        0).  Non-finite values are rejected (Deviation from R)."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number or None; got {value!r}.")
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite (R accepts Inf; diff-diff requires a "
                f"finite floor - documented deviation); got {value!r}."
            )
        ceiled = int(np.ceil(value))
        if ceiled < 0:
            raise ValueError(f"{name} must satisfy ceil(value) >= 0; got {value!r}.")
        return ceiled

    def _validate_constructor_args(self) -> None:
        if not self._is_real_scalar(self.cutoff):
            raise ValueError(f"cutoff must be a finite real scalar; got {self.cutoff!r}.")
        if isinstance(self.p, bool) or not isinstance(self.p, (int, np.integer)):
            raise ValueError(f"p must be an integer in 1..7; got {self.p!r}.")
        if not 1 <= int(self.p) <= 7:
            raise ValueError(f"p must be an integer in 1..7; got {self.p!r}.")
        if self.q is not None:
            if isinstance(self.q, bool) or not isinstance(self.q, (int, np.integer)):
                raise ValueError(f"q must be an integer >= p or None; got {self.q!r}.")
            if int(self.q) < int(self.p):
                raise ValueError(f"q cannot be smaller than p; got q={self.q!r}, p={self.p!r}.")
        if self.fitselect not in ("unrestricted", "restricted"):
            raise ValueError(
                "fitselect must be 'unrestricted' or 'restricted'; " f"got {self.fitselect!r}."
            )
        _normalize_kernel(self.kernel)
        if self.vcov_type not in ("jackknife", "plugin"):
            raise ValueError(f"vcov_type must be 'jackknife' or 'plugin'; got {self.vcov_type!r}.")
        hl, hr = self._check_bandwidth_pair(self.h, "h")
        if self.fitselect == "restricted" and hl is not None and hl != hr:
            raise ValueError(
                "Bandwidths must be equal in the restricted model; " f"got h={self.h!r}."
            )
        if self.bwselect not in ("each", "diff", "sum", "comb"):
            raise ValueError(
                "bwselect must be one of 'each', 'diff', 'sum', 'comb'; " f"got {self.bwselect!r}."
            )
        if self.fitselect == "restricted" and self.bwselect == "each":
            raise ValueError("bwselect='each' is not available in the restricted model.")
        if self.masspoints not in ("adjust", "check", "off"):
            raise ValueError(
                "masspoints must be one of 'adjust', 'check', 'off'; " f"got {self.masspoints!r}."
            )
        if not isinstance(self.regularize, bool):
            raise ValueError(f"regularize must be a bool; got {self.regularize!r}.")
        self._resolve_floor(self.n_local_min, "n_local_min")
        self._resolve_floor(self.n_unique_min, "n_unique_min")
        if not isinstance(self.report_all, bool):
            raise ValueError(f"report_all must be a bool; got {self.report_all!r}.")

    def fit(self, data: pd.DataFrame, running: str) -> RDDensityTestResult:
        """Run the manipulation test on ``data[running]``.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        running : str
            Column name of the running variable.
        """
        if running not in data.columns:
            raise ValueError(f"Column {running!r} not found in data.")

        # input coercion (RDPlot precedent): coerce, drop non-finite with a
        # counting warning (Deviation from R for the +-inf/non-numeric part;
        # R itself warns on NA drops)
        raw = np.asarray(pd.to_numeric(data[running], errors="coerce"), dtype=float)
        finite = np.isfinite(raw)
        n_dropped = int((~finite).sum())
        if n_dropped:
            warnings.warn(
                f"Dropped {n_dropped} observation(s) with missing or "
                f"non-finite {running!r} values.",
                UserWarning,
                stacklevel=2,
            )
        x = np.sort(raw[finite])
        n = len(x)
        if n == 0:
            raise ValueError("No complete observations in the running variable.")

        cutoff = float(self.cutoff)
        if cutoff <= x[0] or cutoff >= x[-1]:
            raise ValueError("The cutoff should be set within the range of the data.")
        n_left = int(np.sum(x < cutoff))
        n_right = n - n_left
        if n_left < 2 or n_right < 2:
            raise ValueError(
                f"Each side of the cutoff needs at least 2 observations; got "
                f"{n_left} left / {n_right} right. (House guard; R proceeds "
                "and fails downstream.)"
            )

        p = int(self.p)
        q = int(self.q) if self.q is not None else p + 1
        kernel = _normalize_kernel(self.kernel)
        n_local_min = self._resolve_floor(self.n_local_min, "n_local_min")
        if n_local_min is None:
            n_local_min = 20 + p + 1
        n_unique_min = self._resolve_floor(self.n_unique_min, "n_unique_min")
        if n_unique_min is None:
            n_unique_min = 20 + p + 1

        unique_info = _rddensity_unique(x)
        has_repeated = _has_repeated(x)
        masspoints_flag = has_repeated and self.masspoints == "adjust"
        if has_repeated and self.masspoints == "adjust":
            warnings.warn(
                "Repeated running-variable values detected; point estimates "
                "and standard errors have been mass-point adjusted (matching "
                "R massPoints=TRUE). Use masspoints='off' to disable.",
                UserWarning,
                stacklevel=2,
            )
        elif has_repeated and self.masspoints == "check":
            warnings.warn(
                "Repeated running-variable values detected; estimation "
                "proceeds WITHOUT mass-point adjustment (masspoints='check'). "
                "Use masspoints='adjust' for the R-default adjustment.",
                UserWarning,
                stacklevel=2,
            )

        # bandwidths
        hl_manual, hr_manual = self._check_bandwidth_pair(self.h, "h")
        bandwidths_table: Optional[pd.DataFrame] = None
        if hl_manual is not None and hr_manual is not None:
            hl, hr = hl_manual, hr_manual
            bandwidth_method = "manual"
        else:
            bandwidth_method = "estimated"
            hn = _rdbwdensity(
                x,
                cutoff,
                p,
                kernel,
                self.fitselect,
                self.vcov_type,
                self.regularize,
                n_local_min,
                n_unique_min,
                masspoints_flag,
                unique_info,
            )
            bandwidths_table = pd.DataFrame(
                hn,
                index=["left", "right", "diff", "sum"],
                columns=["bw", "variance", "bias_sq"],
            )
            if self.fitselect == "unrestricted":
                if self.bwselect == "each":
                    hl, hr = hn[0, 0], hn[1, 0]
                elif self.bwselect == "diff":
                    hl = hr = hn[2, 0]
                elif self.bwselect == "sum":
                    hl = hr = hn[3, 0]
                else:  # comb
                    hl = float(np.median([hn[0, 0], hn[2, 0], hn[3, 0]]))
                    hr = float(np.median([hn[1, 0], hn[2, 0], hn[3, 0]]))
            else:
                if self.bwselect == "diff":
                    hl = hr = hn[2, 0]
                elif self.bwselect == "sum":
                    hl = hr = hn[3, 0]
                else:  # comb
                    hl = hr = float(min(hn[2, 0], hn[3, 0]))
            for side, h_val in (("left", hl), ("right", hr)):
                if not np.isfinite(h_val) or h_val <= 0:
                    raise ValueError(
                        f"The data-driven bandwidth selector returned a "
                        f"degenerate {side} bandwidth ({h_val!r}); the local "
                        "variance or bias estimate is unusable on this "
                        "sample. Supply a manual bandwidth via h= instead."
                    )
        hl = float(hl)
        hr = float(hr)

        # CJM 2020 Section 4: the conventional (non-bias-corrected) order-p
        # test over-rejects at the MSE-optimal bandwidth; valid conventional
        # inference needs an undersmoothed bandwidth. R reports the
        # conventional numbers without comment; the port keeps the R-parity
        # values but warns (REGISTRY Note)
        if bandwidth_method == "estimated" and (q == p or self.report_all):
            conv_surface = (
                "the headline test (q == p is the conventional, non-bias-" "corrected fit)"
                if q == p
                else "the conventional block (report_all=True)"
            )
            warnings.warn(
                f"Conventional (order-p) inference in {conv_surface} is "
                "evaluated at a data-driven MSE-optimal bandwidth, where the "
                "uncorrected test over-rejects (Cattaneo, Jansson & Ma 2020, "
                "Section 4). Valid conventional inference requires a manually "
                "undersmoothed bandwidth (h=), which the library cannot "
                "verify; the robust bias-corrected test (default q = p + 1) "
                "is the recommended inference.",
                UserWarning,
                stacklevel=2,
            )

        # bandwidth-vs-range warnings (R warns at summary time)
        if hl > cutoff - x[0]:
            warnings.warn(
                "Bandwidth h_left is greater than the range of the data " "below the cutoff.",
                UserWarning,
                stacklevel=2,
            )
        if hr > x[-1] - cutoff:
            warnings.warn(
                "Bandwidth h_right is greater than the range of the data " "above the cutoff.",
                UserWarning,
                stacklevel=2,
            )

        # EDF + window
        y = np.arange(n) / (n - 1)
        if masspoints_flag:
            y = np.repeat(y[unique_info["index_last"]], unique_info["freq"])
        xc = x - cutoff
        window = (xc >= -hl) & (xc <= hr)
        xh = xc[window]
        yh = y[window]
        n_lh = int(np.sum(xh < 0))
        n_rh = len(xh) - n_lh
        if n_lh < 20 or n_rh < 20:
            warnings.warn(
                "Fewer than 20 effective observations on at least one side "
                "of the cutoff; the bandwidth may be too small.",
                UserWarning,
                stacklevel=2,
            )

        fv_q = _rddensity_fv(
            yh,
            xh,
            n,
            n_lh,
            n_rh,
            hl,
            hr,
            q,
            1,
            kernel,
            self.fitselect,
            self.vcov_type,
            masspoints_flag,
        )
        var_col = 2 if self.vcov_type == "plugin" else 1

        def _block(fv: np.ndarray) -> Tuple[float, ...]:
            with np.errstate(invalid="ignore"):
                raw_ses = [float(np.sqrt(fv[i, var_col])) for i in range(3)]
            # degraded-inference contract: a negative, zero, or non-finite
            # variance degrades the SE (and hence the inference) to NaN
            se_l, se_r, se_d = (
                se if np.isfinite(se) and se > 0 else float("nan") for se in raw_ses
            )
            # R composes the difference variance from the raw one-sided
            # contributions BEFORE clearing negatives, so an invalid side can
            # leave a finite (understated) difference variance. The composed
            # inference is only as valid as its components: an invalid
            # marginal SE degrades the difference SE and the test inference
            # too (REGISTRY degraded-inference note).
            if np.isnan(se_l) or np.isnan(se_r):
                se_d = float("nan")
            if any(np.isnan(se) for se in (se_l, se_r, se_d)):
                warnings.warn(
                    "A variance estimate is negative, zero, or non-finite; "
                    "the affected standard errors and test inference are NaN.",
                    UserWarning,
                    stacklevel=3,
                )
            t, pval, _ = safe_inference(float(fv[2, 0]), se_d)
            return (
                float(fv[0, 0]),
                float(fv[1, 0]),
                float(fv[2, 0]),
                se_l,
                se_r,
                se_d,
                t,
                pval,
            )

        # safe_inference with df=None uses the standard normal, matching R's
        # 2*pnorm(-|T|) p-value - use its p directly (single inference path)
        (
            f_left,
            f_right,
            f_diff,
            se_left,
            se_right,
            se_diff,
            t_stat,
            p_value,
        ) = _block(fv_q)

        conv: Dict[str, Optional[float]] = dict.fromkeys(
            [
                "f_left_conventional",
                "f_right_conventional",
                "f_diff_conventional",
                "se_left_conventional",
                "se_right_conventional",
                "se_diff_conventional",
                "t_stat_conventional",
                "p_value_conventional",
            ]
        )
        if self.report_all:
            fv_p = _rddensity_fv(
                yh,
                xh,
                n,
                n_lh,
                n_rh,
                hl,
                hr,
                p,
                1,
                kernel,
                self.fitselect,
                self.vcov_type,
                masspoints_flag,
            )
            (
                conv["f_left_conventional"],
                conv["f_right_conventional"],
                conv["f_diff_conventional"],
                conv["se_left_conventional"],
                conv["se_right_conventional"],
                conv["se_diff_conventional"],
                conv["t_stat_conventional"],
                conv["p_value_conventional"],
            ) = _block(fv_p)

        # R silently reports finite negative side densities on
        # solvable-but-degenerate data; the port keeps the R-parity numbers
        # but warns (REGISTRY deviation: finite negative densities warn)
        negative_sides = [
            f"{label}={val:.6g}"
            for label, val in (
                ("f_left", f_left),
                ("f_right", f_right),
                ("f_left_conventional", conv["f_left_conventional"]),
                ("f_right_conventional", conv["f_right_conventional"]),
            )
            if val is not None and np.isfinite(val) and val < 0
        ]
        if negative_sides:
            warnings.warn(
                "Estimated density is negative at the cutoff ("
                + ", ".join(negative_sides)
                + "); the local polynomial fit is degenerate there "
                "(bandwidth too large for the local support, or too few "
                "observations near the cutoff). Interpret the test with "
                "caution.",
                UserWarning,
                stacklevel=2,
            )

        return RDDensityTestResult(
            t_stat=t_stat,
            p_value=p_value,
            f_left=f_left,
            f_right=f_right,
            f_diff=f_diff,
            se_left=se_left,
            se_right=se_right,
            se_diff=se_diff,
            f_left_conventional=conv["f_left_conventional"],
            f_right_conventional=conv["f_right_conventional"],
            f_diff_conventional=conv["f_diff_conventional"],
            se_left_conventional=conv["se_left_conventional"],
            se_right_conventional=conv["se_right_conventional"],
            se_diff_conventional=conv["se_diff_conventional"],
            t_stat_conventional=conv["t_stat_conventional"],
            p_value_conventional=conv["p_value_conventional"],
            n=n,
            n_left=n_left,
            n_right=n_right,
            n_eff_left=n_lh,
            n_eff_right=n_rh,
            h_left=hl,
            h_right=hr,
            bandwidths=bandwidths_table,
            cutoff=cutoff,
            p=p,
            q=q,
            fitselect=self.fitselect,
            kernel=kernel,
            vcov_type=self.vcov_type,
            bwselect=self.bwselect,
            bandwidth_method=bandwidth_method,
            masspoints=self.masspoints,
            masspoints_adjusted=bool(masspoints_flag),
            regularize=self.regularize,
            n_local_min=n_local_min,
            n_unique_min=n_unique_min,
            report_all=self.report_all,
        )
