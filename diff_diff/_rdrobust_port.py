"""In-house port of rdrobust's sharp-RD bandwidth-selection machinery.

Faithful Python translation of the sharp-RD branch (no fuzzy / covariates /
cluster / weights) of ``rdbwselect`` from the R package ``rdrobust`` 4.0.0,
ported from the CRAN source tarball (sha256 below), cross-checked against
``deparse(getFromNamespace(<fn>, "rdrobust"))`` of the installed 4.0.0
package. The unreleased GitHub development tree (4.1.0-dev) differs from
4.0.0 in three load-bearing ways and was NOT used as the parity source:
4.0.0 compares nearest-neighbor distances EXACTLY (no ``nn_tol``
tolerance), defaults ``stdvars=FALSE``, and adds ``+1e-8`` to the bwcheck
floor.

Source mapping (every public function here pairs with one R function):

==========================================  ===================================
Python                                      R (rdrobust 4.0.0)
==========================================  ===================================
``rdrobust_kweight(x, c, h, kernel)``       ``rdrobust_kweight`` (functions.R:134-144)
``qrXXinv(x)``                              ``qrXXinv`` (functions.R:128-132)
``rdrobust_vander(u, p)``                   ``.rdrobust_vander`` (functions.R:85-94)
``compute_dups_dupsid(x_sorted)``           rle blocks (rdbwselect.R:322-327)
``rdrobust_res_nn(...)``                    ``rdrobust_res`` vce="nn" branch
                                            (functions.R:146-181)
``rdrobust_vce_sharp(RX, res)``             ``rdrobust_vce`` null-cluster d==0
                                            branch (functions.R:374-378)
``rdrobust_bw(...)``                        ``rdrobust_bw`` sharp path
                                            (functions.R:207-355)
``rdbwselect_sharp(...)``                   ``rdbwselect`` main flow
                                            (rdbwselect.R; anchors inline)
==========================================  ===================================

Deviations from rdrobust (documented; see REGISTRY.md RegressionDiscontinuity
section):

* Inputs must be complete-case 1-D float arrays: this port REJECTS
  non-finite values with ``ValueError`` instead of silently dropping rows
  the way R's ``complete.cases`` filter does (rdbwselect.R:72-95). The
  public estimator (PR-2) warns-and-drops before calling in.
* ``N < 20`` raises ``ValueError`` here, mirroring rdbwselect.R:237-239's
  warn-and-abort (``exit = 1``). The estimator-level full-range-bandwidth
  fallback lives in rdrobust.R and is ported with the estimator (PR-2).
* Empty sides and zero running-variable variance raise targeted
  ``ValueError`` rather than propagating R's opaque downstream errors.
* Only ``vce="nn"`` is implemented; ``hc0``-``hc3`` and cluster modes raise
  ``NotImplementedError`` (documented v1 seam).
* ``deriv`` is machinery-supported for any ``0 <= deriv <= p`` but golden-
  covered only for ``deriv in {0, 1}``; the public estimator does not expose
  it (sharp levels only).
* ``qrXXinv``'s Cholesky-failure fallback maps ``MASS::ginv(G)`` to
  ``numpy.linalg.pinv(G, rcond=sqrt(eps))`` - both are Moore-Penrose
  pseudo-inverses with the same default singular-value cutoff. Reachable
  only on degenerate (rank-deficient) kernel windows.

Nothing in this module is shared with ``diff_diff._nprobust_port``:
the corresponding nprobust primitives differ in kernel scaling (``/h``),
inversion fallback behavior, and NN tie handling, and the two ports carry
independent upstream version pins. Parity trumps DRY.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import linalg as _scipy_linalg

__all__ = [
    "RDROBUST_VERSION",
    "RDROBUST_TARBALL_SHA256",
    "KERNEL_C_C",
    "BWSELECT_OPTIONS",
    "RdBwselectResult",
    "rdrobust_kweight",
    "qrXXinv",
    "rdrobust_vander",
    "compute_dups_dupsid",
    "rdrobust_res_nn",
    "rdrobust_vce_sharp",
    "rdrobust_bw",
    "rdbwselect_sharp",
    "quantile_type2",
    "RdFitSharpResult",
    "rdrobust_fit_sharp",
]

# Upstream pin: CRAN source tarball of record. rdrobust has no git SHA on
# CRAN; the tarball hash is the provenance anchor (see module docstring for
# why the GitHub tree is NOT the parity source).
RDROBUST_VERSION = "4.0.0"
RDROBUST_TARBALL_SHA256 = "78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f"

# Pilot-bandwidth kernel constants C_c (rdbwselect.R:263-274). Keyed by the
# normalized kernel name; never collapse to the triangular value alone.
KERNEL_C_C = {
    "epanechnikov": 2.34,
    "uniform": 1.843,
    "triangular": 2.576,
}

_KERNEL_ALIASES = {
    "tri": "triangular",
    "triangular": "triangular",
    "epa": "epanechnikov",
    "epanechnikov": "epanechnikov",
    "uni": "uniform",
    "uniform": "uniform",
}

# Canonical selector order, matching R's all=TRUE row order
# (rdbwselect.R:548-560). Imported by the estimator and the parity tests.
BWSELECT_OPTIONS = (
    "mserd",
    "msetwo",
    "msesum",
    "msecomb1",
    "msecomb2",
    "cerrd",
    "certwo",
    "cersum",
    "cercomb1",
    "cercomb2",
)


def _normalize_kernel(kernel: str) -> str:
    key = str(kernel).lower()
    if key not in _KERNEL_ALIASES:
        raise ValueError(
            f"kernel must be one of {sorted(set(_KERNEL_ALIASES.values()))} "
            f"(or R spellings 'tri'/'epa'/'uni'); got {kernel!r}."
        )
    return _KERNEL_ALIASES[key]


def rdrobust_kweight(x: np.ndarray, c: float, h: float, kernel: str) -> np.ndarray:
    """Kernel weights ``w = k((x - c)/h)/h`` (functions.R:134-144).

    Triangular is R's else-branch default. Weights are exactly zero outside
    ``|u| <= 1``; callers rely on ``w > 0`` to define the effective sample.
    """
    kernel = _normalize_kernel(kernel)
    u = (x - c) / h
    inside = np.abs(u) <= 1
    if kernel == "epanechnikov":
        w = (0.75 * (1 - u**2) * inside) / h
    elif kernel == "uniform":
        w = (0.5 * inside) / h
    else:  # triangular (functions.R:141)
        w = ((1 - np.abs(u)) * inside) / h
    return w


def qrXXinv(x: np.ndarray) -> np.ndarray:
    """``(X'X)^{-1}`` via Cholesky, pseudo-inverse on failure
    (functions.R:128-132).

    R computes ``chol2inv(chol(crossprod(x)))`` and falls back to
    ``MASS::ginv(G)`` when the Cholesky factorization fails (non-PD Gram
    matrix from a degenerate kernel window). ``MASS::ginv``'s default
    tolerance is ``sqrt(.Machine$double.eps)`` applied to the largest
    singular value, which is exactly ``numpy.linalg.pinv``'s ``rcond``
    semantics.
    """
    G = x.T @ x
    try:
        cf = _scipy_linalg.cho_factor(G, lower=False)
        return _scipy_linalg.cho_solve(cf, np.eye(G.shape[0]))
    except _scipy_linalg.LinAlgError:
        return np.linalg.pinv(G, rcond=float(np.sqrt(np.finfo(np.float64).eps)))


def rdrobust_vander(u: np.ndarray, p: int) -> np.ndarray:
    """Vandermonde matrix ``[1, u, ..., u^p]`` by successive multiplication
    (functions.R:85-94). The successive-product construction is kept (rather
    than ``u[:, None] ** arange``) so float rounding matches R exactly."""
    n = u.shape[0]
    if p < 1:
        return np.ones((n, 1))
    out = np.ones((n, p + 1))
    for j in range(1, p + 1):
        out[:, j] = out[:, j - 1] * u
    return out


def compute_dups_dupsid(x_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tie-block sizes and within-block 1-based indices (rdbwselect.R:322-327).

    R: ``runs = rle(x); dups = rep(runs$lengths, runs$lengths);
    dupsid = sequence(runs$lengths)``. Input must be sorted ascending (the
    caller sorts with a STABLE sort first); consecutive-equal runs then
    coincide with value groups.
    """
    n = x_sorted.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = x_sorted[1:] != x_sorted[:-1]
    run_starts = np.flatnonzero(change)
    run_lengths = np.diff(np.append(run_starts, n))
    dups = np.repeat(run_lengths, run_lengths)
    dupsid = np.concatenate([np.arange(1, L + 1) for L in run_lengths])
    return dups.astype(np.int64), dupsid.astype(np.int64)


def rdrobust_res_nn(
    x: np.ndarray,
    y: np.ndarray,
    matches: int,
    dups: np.ndarray,
    dupsid: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor variance residuals, sharp outcome-only case
    (functions.R:146-181, ``vce == "nn"`` branch).

    Abadie-Imbens NN sigma via same-side neighbors on the SORTED ``x``.
    Ties are matched as whole ``dups``/``dupsid`` blocks; distances compare
    EXACTLY (4.0.0 semantics - the 4.1.0-dev ``nn_tol`` tolerance is
    deliberately absent). Equal left/right distances expand BOTH directions
    (functions.R:162-165). Returns the (n,) residual vector
    ``sqrt(J/(J+1)) * (y_i - mean(y_neighbors))``.
    """
    n = y.shape[0]
    res = np.empty(n, dtype=np.float64)
    limit = min(matches, n - 1)
    for pos in range(n):  # R pos is 1-based; comments track R indices
        rpos = int(dups[pos] - dupsid[pos])
        lpos = int(dupsid[pos] - 1)
        while lpos + rpos < limit:
            # R indices: pos1 = pos+1 (1-based). Left probe = pos1-lpos-1,
            # right probe = pos1+rpos+1; Python offsets below are 0-based.
            left_probe = pos - lpos - 1
            right_probe = pos + rpos + 1
            if left_probe < 0:  # R: pos-lpos-1 <= 0 (functions.R:158)
                rpos += int(dups[right_probe])
            elif right_probe > n - 1:  # R: pos+rpos+1 > n (functions.R:159)
                lpos += int(dups[left_probe])
            else:
                dleft = x[pos] - x[left_probe]
                dright = x[right_probe] - x[pos]
                if dleft > dright:  # functions.R:160 (exact comparison)
                    rpos += int(dups[right_probe])
                elif dleft < dright:  # functions.R:161
                    lpos += int(dups[left_probe])
                else:  # exact tie: expand both sides (functions.R:162-165)
                    rpos += int(dups[right_probe])
                    lpos += int(dups[left_probe])
        lo = max(0, pos - lpos)
        hi = min(n - 1, pos + rpos)
        # R: ind_J = max(0, pos-lpos):min(n, pos+rpos) on 1-based indices,
        # where a 0 lower bound is out-of-range and drops silently on
        # subset; the effective window is [lo, hi] inclusive around pos.
        y_J = float(np.sum(y[lo : hi + 1])) - float(y[pos])
        Ji = (hi - lo + 1) - 1
        res[pos] = np.sqrt(Ji / (Ji + 1)) * (y[pos] - y_J / Ji)
    return res


def rdrobust_vce_sharp(RX: np.ndarray, res: np.ndarray) -> np.ndarray:
    """Variance meat, sharp no-cluster case (functions.R:374-378, d==0):
    ``M = crossprod(res * RX)``."""
    scaled = res[:, None] * RX
    return scaled.T @ scaled


@dataclass
class _BwPilot:
    """Per-side pilot block returned by :func:`rdrobust_bw`
    (functions.R:349-353): V, B, R (regularization), rate = 1/(2o+3)."""

    V: float
    B: float
    R: float
    rate: float


def rdrobust_bw(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    o: int,
    nu: int,
    o_B: int,
    h_V: float,
    h_B: float,
    scale: float,
    vce: str,
    nnmatch: int,
    kernel: str,
    dups: np.ndarray,
    dupsid: np.ndarray,
    vcache: Optional[Dict[str, Tuple[float, float]]] = None,
) -> _BwPilot:
    """Per-side pilot V/B(/R) block (functions.R:207-355, sharp path).

    Sharp specialization: T = Z = C = W = NULL so the fuzzy/covariate
    combination vector ``s`` is the scalar 1 (functions.R:234) and the
    response matrix is the outcome column alone. ``vcache`` shares the
    fixed-``h_V`` V-fit across pilot calls keyed on ``(o, nu)``
    (functions.R:216-222), matching R's per-side environment cache.
    """
    if vce != "nn":
        raise NotImplementedError(
            "Only vce='nn' is ported in v1; rdrobust's hc0-hc3 and cluster "
            "variance modes are a documented seam."
        )
    key = f"{o}_{nu}"
    if vcache is not None and key in vcache:
        V_V, BConst = vcache[key]  # functions.R:218-222
    else:
        # --- V-fit at (o, nu), bandwidth h_V (functions.R:226-299) ---
        w = rdrobust_kweight(x, c, h_V, kernel)
        ind_V = w > 0
        eY = y[ind_V]
        eX = x[ind_V]
        eW = w[ind_V]
        R_V = rdrobust_vander(eX - c, o)
        invG_V = qrXXinv(R_V * np.sqrt(eW)[:, None])
        # R computes beta_V here (functions.R:263) but the sharp/nn path
        # never consumes it (it feeds the fuzzy ratio and hc predictions);
        # omitted - no numeric effect on V, B, or R.
        res_V = rdrobust_res_nn(eX, eY, nnmatch, dups[ind_V], dupsid[ind_V])  # functions.R:293
        aux = rdrobust_vce_sharp(R_V * eW[:, None], res_V)  # functions.R:294
        V_V = float((invG_V @ aux @ invG_V)[nu, nu])  # functions.R:295
        v = (R_V * eW[:, None]).T @ ((eX - c) / h_V) ** (o + 1)  # :296
        Hp = h_V ** np.arange(o + 1, dtype=np.float64)  # functions.R:297-298
        BConst = float((Hp * (invG_V @ v))[nu])  # functions.R:299
        if vcache is not None:
            vcache[key] = (V_V, BConst)
    # --- B-fit at o_B, bandwidth h_B (functions.R:306-348) ---
    w = rdrobust_kweight(x, c, h_B, kernel)
    ind = w > 0
    eY = y[ind]
    eX = x[ind]
    eW = w[ind]
    R_B = rdrobust_vander(eX - c, o_B)
    invG_B = qrXXinv(R_B * np.sqrt(eW)[:, None])
    beta_B = invG_B @ (R_B * eW[:, None]).T @ eY  # functions.R:326
    BWreg = 0.0
    if scale > 0:  # functions.R:328-348
        res_B = rdrobust_res_nn(eX, eY, nnmatch, dups[ind], dupsid[ind])
        V_B = float(
            (invG_B @ rdrobust_vce_sharp(R_B * eW[:, None], res_B) @ invG_B)[o + 1, o + 1]
        )  # functions.R:346 - R row/col o+2 is 0-based (o+1, o+1)
        BWreg = 3.0 * BConst**2 * V_B  # functions.R:347
    # functions.R:349-353. R row o+2 (1-based) of beta_B is 0-based o+1;
    # sharp s == 1 so t(s) %*% beta_B[o+2,] is the scalar coefficient.
    B = float(np.sqrt(2.0 * (o + 1 - nu)) * BConst * beta_B[o + 1])
    V = float((2.0 * nu + 1.0) * h_V ** (2 * nu + 1) * V_V)
    R_reg = float(scale * (2.0 * (o + 1 - nu)) * BWreg)
    return _BwPilot(V=V, B=B, R=R_reg, rate=1.0 / (2.0 * o + 3.0))


def quantile_type2(x: np.ndarray, prob: float) -> float:
    """R ``quantile(x, prob, type=2)`` (Hyndman-Fan type 2): inverse of the
    empirical CDF with averaging at discontinuities. rdbwselect.R:117 uses
    type=2 for the IQR entering ``BWp``; numpy's default (type-7 linear
    interpolation) does NOT match, so the definition is implemented
    directly and unit-tested against R values in the golden file."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = xs.shape[0]
    np_p = n * prob
    j = int(np.floor(np_p))
    g = np_p - j
    if j <= 0:
        return float(xs[0])
    if j >= n:
        return float(xs[-1])
    if g == 0.0:
        return float((xs[j - 1] + xs[j]) / 2.0)
    return float(xs[j])


@dataclass
class RdBwselectResult:
    """All-selector bandwidth output plus parity diagnostics.

    ``bws`` maps each of the 10 selector names to
    ``(h_left, h_right, b_left, b_right)`` on the ORIGINAL x scale
    (rdbwselect.R:520-546 layout). ``diagnostics`` carries the pilot
    intermediates the golden files pin per stage and side.
    """

    bws: Dict[str, Tuple[float, float, float, float]]
    N: int
    N_l: int
    N_r: int
    M_l: int
    M_r: int
    c_bw: float
    kernel: str
    masspoints: str
    bwcheck_effective: Optional[int]
    diagnostics: Dict[str, float] = field(default_factory=dict)


def rdbwselect_sharp(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 0.0,
    p: int = 1,
    q: int = 2,
    deriv: int = 0,
    kernel: str = "triangular",
    vce: str = "nn",
    nnmatch: int = 3,
    masspoints: str = "adjust",
    bwcheck: Optional[int] = None,
    bwrestrict: bool = True,
    scaleregul: float = 1.0,
    stdvars: bool = False,
    warn_masspoints: bool = True,
) -> RdBwselectResult:
    """Sharp-RD data-driven bandwidth selection, all 10 selectors
    (rdbwselect.R main flow at the anchors cited inline).

    Always computes the full selector matrix (R's ``all=TRUE``): the ten
    selectors share the same six per-side pilot blocks, so the marginal
    cost is a handful of scalar operations. Inputs must be complete-case
    1-D arrays (see module docstring for the deviation from R's silent
    ``complete.cases`` drop).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    for name, arr in (("y", y), ("x", x)):
        # Accept true 1-D vectors or explicit (n, 1) columns; reject any
        # other shape rather than silently flattening a 2-D array onto
        # unintended (y, x) pairings.
        if not (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)):
            raise ValueError(
                f"{name} must be a 1-D vector or an (n, 1) column; got shape {arr.shape}."
            )
    y = y.reshape(-1)
    x = x.reshape(-1)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"y and x must have equal length; got {y.shape[0]} vs {x.shape[0]}.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError(
            "y and x must be finite and complete-case; drop or impute "
            "missing values before bandwidth selection (the public "
            "estimator warns-and-drops; R's rdbwselect drops silently)."
        )
    kernel = _normalize_kernel(kernel)
    if vce != "nn":
        raise NotImplementedError(
            "Only vce='nn' is ported in v1 (rdrobust default); hc0-hc3 and "
            "cluster modes are a documented seam."
        )
    if masspoints not in ("adjust", "check", "off"):
        raise ValueError(f"masspoints must be 'adjust', 'check', or 'off'; got {masspoints!r}.")
    for name, val in (("p", p), ("q", q), ("deriv", deriv)):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer; got {val!r}.")
    if not (0 <= deriv <= p < q):
        raise ValueError(
            f"Orders must satisfy 0 <= deriv <= p < q; got deriv={deriv}, " f"p={p}, q={q}."
        )
    # Fail-closed numeric-knob validation (library policy; R reaches opaque
    # indexing/division errors on these instead).
    if not (isinstance(nnmatch, (int, np.integer)) and nnmatch >= 1):
        raise ValueError(f"nnmatch must be an integer >= 1; got {nnmatch!r}.")
    if bwcheck is not None and not (isinstance(bwcheck, (int, np.integer)) and bwcheck >= 1):
        raise ValueError(f"bwcheck must be None or an integer >= 1; got {bwcheck!r}.")
    if not (np.isfinite(scaleregul) and scaleregul >= 0):
        raise ValueError(f"scaleregul must be a finite value >= 0; got {scaleregul!r}.")
    N = y.shape[0]
    if N < 20:
        # rdbwselect.R:237-239 warns and aborts (exit = 1). The estimator's
        # full-range-bandwidth fallback is rdrobust.R behavior (PR-2).
        raise ValueError(
            "Not enough observations to perform bandwidth calculations "
            "(N < 20); supply manual bandwidths instead."
        )

    # --- Sort (rdbwselect.R:106-114). STABLE sort: NN tie blocks and the
    # rle-based dups/dupsid depend on within-tie order preservation. ---
    if vce == "nn" or masspoints in ("check", "adjust"):
        order_x = np.argsort(x, kind="stable")
        x = x[order_x]
        y = y[order_x]

    # --- Degeneracy guards BEFORE any standardization division: a constant
    # running variable must surface as the assumption failure it is, not as
    # a divide-by-zero under stdvars or a misleading one-sided-data error
    # downstream (all-equal x always lands on a single side of the cutoff).
    x_sd_data = float(np.std(x, ddof=1))
    if x_sd_data == 0.0:
        raise ValueError("The running variable has zero variance.")
    if stdvars and float(np.std(y, ddof=1)) == 0.0:
        raise ValueError(
            "The outcome has zero variance; stdvars standardization would divide by zero."
        )

    # --- Rescaling constants (rdbwselect.R:116-129) ---
    x_iq = quantile_type2(x, 0.75) - quantile_type2(x, 0.25)
    BWp = min(x_sd_data, x_iq / 1.349)
    # Diagnostic contract: BWp_data is the PRE-standardization data property
    # (what the golden's head-intermediates helper records); the working BWp
    # below is overwritten under stdvars per rdbwselect.R:128.
    BWp_data = BWp
    x_sd = 1.0
    if stdvars:  # 4.0.0 default FALSE; ported for completeness
        y_sd = float(np.std(y, ddof=1))
        x_sd = x_sd_data
        y = y / y_sd
        x = x / x_sd
        c = c / x_sd
        BWp = min(1.0, (x_iq / x_sd) / 1.349)

    # --- Side split (rdbwselect.R:131-137). X == c is treated (right). ---
    ind_l = x < c
    ind_r = x >= c
    X_l, X_r = x[ind_l], x[ind_r]
    Y_l, Y_r = y[ind_l], y[ind_r]
    N_l, N_r = int(X_l.shape[0]), int(X_r.shape[0])
    if N_l == 0 or N_r == 0:
        raise ValueError(
            "All observations fall on one side of the cutoff; sharp RD "
            "requires data on both sides."
        )
    x_min, x_max = float(np.min(x)), float(np.max(x))
    range_l = abs(c - x_min)
    range_r = abs(c - x_max)

    # --- Mass points (rdbwselect.R:139-159) ---
    M_l, M_r = N_l, N_r
    X_uniq_l = X_uniq_r = None
    if masspoints in ("check", "adjust") or bwcheck is not None:
        X_uniq_l = np.unique(X_l)[::-1]  # sorted DECREASING (:145)
        X_uniq_r = np.unique(X_r)  # ascending = closest-first (:146)
        M_l = int(X_uniq_l.shape[0])
        M_r = int(X_uniq_r.shape[0])
    bwcheck_effective = bwcheck
    if masspoints in ("check", "adjust"):
        mass_l = 1.0 - M_l / N_l
        mass_r = 1.0 - M_r / N_r
        if mass_l >= 0.2 or mass_r >= 0.2:
            # warn_masspoints=False silences ONLY the warnings (the
            # bwcheck floor below still applies): R's rdrobust() runs the
            # detection itself before its inline selection (rdrobust.R:
            # 365-380), so the estimator warns once at fit() level and
            # passes False here to avoid a stacked duplicate. Direct
            # callers keep rdbwselect.R:139-159's own warning behavior.
            if warn_masspoints:
                warnings.warn(
                    "Mass points detected in the running variable.",
                    UserWarning,
                    stacklevel=2,
                )
                if masspoints == "check":
                    warnings.warn(
                        "Try using option masspoints='adjust'.",
                        UserWarning,
                        stacklevel=2,
                    )
            if bwcheck is None and masspoints == "adjust":
                bwcheck_effective = 10  # rdbwselect.R:157

    # --- Kernel constant + pilot bandwidth (rdbwselect.R:263-274, 360-377) ---
    C_c = KERNEL_C_C[kernel]
    M = M_l + M_r
    c_bw = C_c * BWp * N ** (-1.0 / 5.0)
    if masspoints == "adjust":
        c_bw = C_c * BWp * M ** (-1.0 / 5.0)  # rdbwselect.R:361
    bw_max_l = range_l
    bw_max_r = range_r
    bw_max = max(bw_max_l, bw_max_r)
    if bwrestrict:
        c_bw = min(c_bw, bw_max)  # rdbwselect.R:363-368
    bw_min_l = bw_min_r = None
    if bwcheck_effective is not None:
        # rdbwselect.R:371-377; the +1e-8 is 4.0.0-specific. R's 1-based
        # [bwcheck_l] indexing maps to 0-based [bwcheck_l - 1]. X_uniq_* are
        # always populated on this path (computed under check/adjust or an
        # explicit bwcheck, the only routes to a non-None bwcheck_effective).
        assert X_uniq_l is not None and X_uniq_r is not None
        bwcheck_l = min(bwcheck_effective, M_l)
        bwcheck_r = min(bwcheck_effective, M_r)
        bw_min_l = float(np.abs(X_uniq_l - c)[bwcheck_l - 1]) + 1e-8
        bw_min_r = float(np.abs(X_uniq_r - c)[bwcheck_r - 1]) + 1e-8
        c_bw = max(c_bw, bw_min_l, bw_min_r)

    # --- NN tie blocks (rdbwselect.R:322-327) ---
    dups_l, dupsid_l = compute_dups_dupsid(X_l)
    dups_r, dupsid_r = compute_dups_dupsid(X_r)

    vcache_l: Dict[str, Tuple[float, float]] = {}
    vcache_r: Dict[str, Tuple[float, float]] = {}

    def _bw(side: str, o: int, nu: int, o_B: int, h_B: float, scale: float) -> _BwPilot:
        if side == "l":
            return rdrobust_bw(
                Y_l,
                X_l,
                c,
                o,
                nu,
                o_B,
                c_bw,
                h_B,
                scale,
                vce,
                nnmatch,
                kernel,
                dups_l,
                dupsid_l,
                vcache_l,
            )
        return rdrobust_bw(
            Y_r,
            X_r,
            c,
            o,
            nu,
            o_B,
            c_bw,
            h_B,
            scale,
            vce,
            nnmatch,
            kernel,
            dups_r,
            dupsid_r,
            vcache_r,
        )

    def _stage_bw(num, den, rate, clamp_max=None, floors=()):
        """One MSE-stage bandwidth: (num/den)^rate with R division
        semantics (0/0 -> NaN, x/0 -> inf), bwrestrict clamp, bwcheck
        floor, and a fail-closed finiteness check. R propagates NaN into
        the next pilot fit and fails opaquely; we raise a targeted error
        (documented deviation)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            val = float(np.float64(num) / np.float64(den)) ** rate
        if not np.isnan(val):
            if clamp_max is not None:
                val = min(val, clamp_max)  # absorbs +inf under bwrestrict
            if floors:
                val = max(val, *floors)
        if not np.isfinite(val):
            raise ValueError(
                "Bandwidth selection produced a non-finite pilot bandwidth "
                "(degenerate MSE objective, e.g. a constant outcome or no "
                "curvature signal); supply manual bandwidths h=/b= instead."
            )
        return val

    # --- d-stage pilots (rdbwselect.R:386-387): o=q+1, nu=q+1, o_B=q+2,
    # h_B = full per-side range, UNregularized (scale=0). ---
    C_d_l = _bw("l", q + 1, q + 1, q + 2, range_l, 0.0)
    C_d_r = _bw("r", q + 1, q + 1, q + 2, range_r, 0.0)

    rate_d = C_d_l.rate  # 1/(2(q+1)+3)
    diagnostics: Dict[str, float] = {
        "x_iq": x_iq,
        "BWp": BWp_data,
        "BWp_working": BWp,
        "C_d_l_V": C_d_l.V,
        "C_d_l_B": C_d_l.B,
        "C_d_r_V": C_d_r.V,
        "C_d_r_B": C_d_r.B,
    }

    _global_max = bw_max if bwrestrict else None
    _d_floors = (bw_min_l, bw_min_r) if bwcheck_effective is not None else ()

    # ============ mserd chain (rdbwselect.R:444-462) ============
    d_bw_d = _stage_bw(
        C_d_l.V + C_d_r.V,
        (C_d_r.B - C_d_l.B) ** 2,
        rate_d,
        clamp_max=_global_max,
        floors=_d_floors,
    )
    C_b_l = _bw("l", q, p + 1, q + 1, d_bw_d, scaleregul)
    C_b_r = _bw("r", q, p + 1, q + 1, d_bw_d, scaleregul)
    b_bw_d = _stage_bw(
        C_b_l.V + C_b_r.V,
        (C_b_r.B - C_b_l.B) ** 2 + scaleregul * (C_b_r.R + C_b_l.R),
        C_b_l.rate,
        clamp_max=_global_max,
    )
    C_h_l = _bw("l", p, deriv, q, b_bw_d, scaleregul)
    C_h_r = _bw("r", p, deriv, q, b_bw_d, scaleregul)
    h_bw_d = _stage_bw(
        C_h_l.V + C_h_r.V,
        (C_h_r.B - C_h_l.B) ** 2 + scaleregul * (C_h_r.R + C_h_l.R),
        C_h_l.rate,
        clamp_max=_global_max,
    )
    diagnostics.update(
        d_bw_d=d_bw_d,
        b_bw_d=b_bw_d,
        h_bw_d=h_bw_d,
        C_b_l_V=C_b_l.V,
        C_b_l_B=C_b_l.B,
        C_b_l_R=C_b_l.R,
        C_b_r_V=C_b_r.V,
        C_b_r_B=C_b_r.B,
        C_b_r_R=C_b_r.R,
        C_h_l_V=C_h_l.V,
        C_h_l_B=C_h_l.B,
        C_h_l_R=C_h_l.R,
        C_h_r_V=C_h_r.V,
        C_h_r_B=C_h_r.B,
        C_h_r_R=C_h_r.R,
    )

    # ============ msetwo chain (rdbwselect.R:389-420) ============
    # Per-side clamps throughout; bwcheck floor applies at the d-stage only
    # (rdbwselect.R:396-399); b/h stages clamp with bwrestrict only.
    _max_l = bw_max_l if bwrestrict else None
    _max_r = bw_max_r if bwrestrict else None
    _fl_l = (bw_min_l,) if bwcheck_effective is not None else ()
    _fl_r = (bw_min_r,) if bwcheck_effective is not None else ()
    d_bw_l = _stage_bw(C_d_l.V, C_d_l.B**2, rate_d, clamp_max=_max_l, floors=_fl_l)
    d_bw_r = _stage_bw(C_d_r.V, C_d_r.B**2, rate_d, clamp_max=_max_r, floors=_fl_r)
    C_b_l2 = _bw("l", q, p + 1, q + 1, d_bw_l, scaleregul)
    b_bw_l = _stage_bw(C_b_l2.V, C_b_l2.B**2 + scaleregul * C_b_l2.R, C_b_l2.rate, clamp_max=_max_l)
    C_b_r2 = _bw("r", q, p + 1, q + 1, d_bw_r, scaleregul)
    b_bw_r = _stage_bw(C_b_r2.V, C_b_r2.B**2 + scaleregul * C_b_r2.R, C_b_r2.rate, clamp_max=_max_r)
    C_h_l2 = _bw("l", p, deriv, q, b_bw_l, scaleregul)
    h_bw_l = _stage_bw(C_h_l2.V, C_h_l2.B**2 + scaleregul * C_h_l2.R, C_h_l2.rate, clamp_max=_max_l)
    C_h_r2 = _bw("r", p, deriv, q, b_bw_r, scaleregul)
    h_bw_r = _stage_bw(C_h_r2.V, C_h_r2.B**2 + scaleregul * C_h_r2.R, C_h_r2.rate, clamp_max=_max_r)

    # ============ msesum chain (rdbwselect.R:422-441) ============
    # Sum expansion: (B_r + B_l)^2 in the denominator; global clamps.
    d_bw_s = _stage_bw(
        C_d_l.V + C_d_r.V,
        (C_d_r.B + C_d_l.B) ** 2,
        rate_d,
        clamp_max=_global_max,
        floors=_d_floors,
    )
    C_b_ls = _bw("l", q, p + 1, q + 1, d_bw_s, scaleregul)
    C_b_rs = _bw("r", q, p + 1, q + 1, d_bw_s, scaleregul)
    b_bw_s = _stage_bw(
        C_b_ls.V + C_b_rs.V,
        (C_b_rs.B + C_b_ls.B) ** 2 + scaleregul * (C_b_rs.R + C_b_ls.R),
        C_b_ls.rate,
        clamp_max=_global_max,
    )
    C_h_ls = _bw("l", p, deriv, q, b_bw_s, scaleregul)
    C_h_rs = _bw("r", p, deriv, q, b_bw_s, scaleregul)
    h_bw_s = _stage_bw(
        C_h_ls.V + C_h_rs.V,
        (C_h_rs.B + C_h_ls.B) ** 2 + scaleregul * (C_h_rs.R + C_h_ls.R),
        C_h_ls.rate,
        clamp_max=_global_max,
    )

    # ============ Rescale + assemble (rdbwselect.R:464-546) ============
    h_mserd, b_mserd = x_sd * h_bw_d, x_sd * b_bw_d
    h_msesum, b_msesum = x_sd * h_bw_s, x_sd * b_bw_s
    h_msetwo_l, h_msetwo_r = x_sd * h_bw_l, x_sd * h_bw_r
    b_msetwo_l, b_msetwo_r = x_sd * b_bw_l, x_sd * b_bw_r
    h_msecomb1 = min(h_mserd, h_msesum)
    b_msecomb1 = min(b_mserd, b_msesum)
    h_msecomb2_l = float(np.median([h_mserd, h_msesum, h_msetwo_l]))
    h_msecomb2_r = float(np.median([h_mserd, h_msesum, h_msetwo_r]))
    b_msecomb2_l = float(np.median([b_mserd, b_msesum, b_msetwo_l]))
    b_msecomb2_r = float(np.median([b_mserd, b_msesum, b_msetwo_r]))
    # CER shrinkage (rdbwselect.R:489-496): h only; cer_b = 1 so b is
    # inherited from the matching MSE selector unchanged.
    cer_h = N ** (-(p / ((3.0 + p) * (3.0 + 2.0 * p))))
    bws: Dict[str, Tuple[float, float, float, float]] = {
        "mserd": (h_mserd, h_mserd, b_mserd, b_mserd),
        "msetwo": (h_msetwo_l, h_msetwo_r, b_msetwo_l, b_msetwo_r),
        "msesum": (h_msesum, h_msesum, b_msesum, b_msesum),
        "msecomb1": (h_msecomb1, h_msecomb1, b_msecomb1, b_msecomb1),
        "msecomb2": (h_msecomb2_l, h_msecomb2_r, b_msecomb2_l, b_msecomb2_r),
        "cerrd": (h_mserd * cer_h, h_mserd * cer_h, b_mserd, b_mserd),
        "certwo": (
            h_msetwo_l * cer_h,
            h_msetwo_r * cer_h,
            b_msetwo_l,
            b_msetwo_r,
        ),
        "cersum": (h_msesum * cer_h, h_msesum * cer_h, b_msesum, b_msesum),
        "cercomb1": (
            h_msecomb1 * cer_h,
            h_msecomb1 * cer_h,
            b_msecomb1,
            b_msecomb1,
        ),
        "cercomb2": (
            h_msecomb2_l * cer_h,
            h_msecomb2_r * cer_h,
            b_msecomb2_l,
            b_msecomb2_r,
        ),
    }
    diagnostics["cer_h"] = cer_h
    return RdBwselectResult(
        bws=bws,
        N=N,
        N_l=N_l,
        N_r=N_r,
        M_l=M_l,
        M_r=M_r,
        c_bw=float(c_bw),
        kernel=kernel,
        masspoints=masspoints,
        bwcheck_effective=bwcheck_effective,
        diagnostics=diagnostics,
    )


@dataclass
class RdFitSharpResult:
    """Sharp-RD point estimates and variances (rdrobust.R estimation body).

    ``tau_cl`` is the conventional local-polynomial RD estimate,
    ``tau_bc`` the bias-corrected estimate; ``se_cl``/``se_rb`` are the
    conventional and robust bias-corrected standard errors. rdrobust's
    three output rows map as Conventional = (tau_cl, se_cl),
    Bias-Corrected = (tau_bc, se_cl), Robust = (tau_bc, se_rb)
    (rdrobust.R:854-863). ``beta_p_l``/``beta_p_r`` are the per-side
    order-p coefficient vectors (rdplot seam); ``bias_l``/``bias_r`` the
    per-side estimated biases (rdrobust.R:629-630).
    """

    tau_cl: float
    tau_bc: float
    se_cl: float
    se_rb: float
    bias_l: float
    bias_r: float
    beta_p_l: np.ndarray
    beta_p_r: np.ndarray
    N_h_l: int
    N_h_r: int
    N_b_l: int
    N_b_r: int


def rdrobust_fit_sharp(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    h_l: float,
    h_r: float,
    b_l: float,
    b_r: float,
    p: int = 1,
    q: int = 2,
    deriv: int = 0,
    kernel: str = "triangular",
    vce: str = "nn",
    nnmatch: int = 3,
) -> RdFitSharpResult:
    """Sharp-RD estimation at known bandwidths (rdrobust.R:533-800, sharp
    no-covariate/no-cluster path with ``scalepar = 1``).

    Inputs must be complete-case 1-D arrays (same contract as
    :func:`rdbwselect_sharp`); sorting, side-splitting, and NN tie blocks
    are handled internally. Per-side bandwidths follow rdrobust's
    ``bws = [[h_l, b_l], [h_r, b_r]]`` layout. Steps:

    1. Effective window per side = observations with positive kernel weight
       inside the WIDER of h and b (rdrobust.R:541-549); observations inside
       the b-window but outside the h-window carry ``W_h = 0`` and drop out
       of the p-regression through the weights.
    2. Point estimates: order-p WLS per side at h (``beta_p``); the
       bias-corrected coefficient vector uses the ``Q_q`` score matrix
       (rdrobust.R:577-578, 609-618).
    3. Variances: conventional sandwiches ``R_p * W_h`` with same-side NN
       residuals; robust sandwiches ``Q_q`` with the SAME residuals
       (``res_b = res_h`` for vce="nn", rdrobust.R:753-754; the h==b
       special branches at rdrobust.R:773-786 are cluster-only and never
       taken on this path).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    for name, arr in (("y", y), ("x", x)):
        # Same input contract as rdbwselect_sharp: 1-D vectors or explicit
        # (n, 1) columns only.
        if not (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)):
            raise ValueError(
                f"{name} must be a 1-D vector or an (n, 1) column; got shape {arr.shape}."
            )
    y = y.reshape(-1)
    x = x.reshape(-1)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"y and x must have equal length; got {y.shape[0]} vs {x.shape[0]}.")
    if vce != "nn":
        raise NotImplementedError(
            "Only vce='nn' is ported in v1 (rdrobust default); hc0-hc3 and "
            "cluster variance modes are a documented seam."
        )
    kernel = _normalize_kernel(kernel)
    for name, val in (("p", p), ("q", q), ("deriv", deriv)):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer; got {val!r}.")
    if not (0 <= deriv <= p < q):
        raise ValueError(
            f"Orders must satisfy 0 <= deriv <= p < q; got deriv={deriv}, " f"p={p}, q={q}."
        )
    if not (isinstance(nnmatch, (int, np.integer)) and nnmatch >= 1):
        raise ValueError(f"nnmatch must be an integer >= 1; got {nnmatch!r}.")
    for name, val in (("h_l", h_l), ("h_r", h_r), ("b_l", b_l), ("b_r", b_r)):
        # Fail-closed: a non-finite or non-positive bandwidth (e.g. from a
        # degenerate selector run) must not reach the kernel weights.
        if not (np.isfinite(val) and val > 0):
            raise ValueError(f"{name} must be finite and positive; got {val!r}.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError(
            "y and x must be finite and complete-case; drop or impute "
            "missing values before estimation."
        )

    # Sort + side split, mirroring the rdbwselect preamble (X == c treated).
    order_x = np.argsort(x, kind="stable")
    x = x[order_x]
    y = y[order_x]
    ind_l = x < c
    ind_r = x >= c
    X_l, X_r = x[ind_l], x[ind_r]
    Y_l, Y_r = y[ind_l], y[ind_r]
    if X_l.shape[0] == 0 or X_r.shape[0] == 0:
        raise ValueError(
            "All observations fall on one side of the cutoff; sharp RD "
            "requires data on both sides."
        )
    dups_l, dupsid_l = compute_dups_dupsid(X_l)
    dups_r, dupsid_r = compute_dups_dupsid(X_r)

    def _side(
        X: np.ndarray,
        Y: np.ndarray,
        h: float,
        b: float,
        dups: np.ndarray,
        dupsid: np.ndarray,
        side: str,
    ):
        # Weights + effective window (rdrobust.R:533-549)
        w_h = rdrobust_kweight(X, c, h, kernel)
        w_b = rdrobust_kweight(X, c, b, kernel)
        ind_h = w_h > 0
        ind_b = w_b > 0
        N_h = int(np.sum(ind_h))
        N_b = int(np.sum(ind_b))
        ind = ind_b
        if h > b:
            ind = ind_h
        eY = Y[ind]
        eX = X[ind]
        W_h = w_h[ind]
        W_b = w_b[ind]
        edups = dups[ind]
        edupsid = dupsid[ind]
        # Per-window identification guards (documented deviation: fail closed
        # where 4.0.0's ginv fallback would silently return a degenerate fit,
        # e.g. an empty b-window collapsing Q_q back to the conventional
        # score). A weighted Vandermonde Gram matrix is nonsingular iff its
        # positive-weight window holds at least order+1 distinct
        # running-variable values, so the distinct count IS the exact rank
        # condition - no numerical rank estimate is needed.
        for w_label, w_vec, order_needed in (
            ("main", W_h, p),
            ("bias", W_b, q),
        ):
            support = eX[w_vec > 0]
            n_distinct = int(np.unique(support).size)
            if n_distinct <= order_needed:
                raise ValueError(
                    f"Too few observations inside the {side} {w_label} "
                    f"bandwidth window to identify the order-{order_needed} "
                    f"local polynomial fit ({support.size} observations, "
                    f"{n_distinct} distinct running-variable values; need "
                    f">= {order_needed + 1} distinct)."
                )
        # Design matrices (rdrobust.R:569-578)
        u = (eX - c) / h
        R_q = rdrobust_vander(eX - c, q)
        R_p = R_q[:, : p + 1]
        L = (R_p * W_h[:, None]).T @ (u ** (p + 1))  # (p+1,)
        invG_q = qrXXinv(np.sqrt(W_b)[:, None] * R_q)
        invG_p = qrXXinv(np.sqrt(W_h)[:, None] * R_p)
        # Q_q = R_p*W_h - h^(p+1) * outer(M[:, p+1], L) with
        # M = (R_q @ invG_q) * W_b  (rdrobust.R:577, algebraically identical
        # to the nested-transpose R expression; e_p1 selects column p+2
        # 1-based = p+1 0-based).
        M = (R_q @ invG_q) * W_b[:, None]
        Q_q = R_p * W_h[:, None] - h ** (p + 1) * np.outer(M[:, p + 1], L)
        # Point estimates (rdrobust.R:609-614)
        beta_p = invG_p @ (R_p * W_h[:, None]).T @ eY
        beta_bc = invG_p @ Q_q.T @ eY
        # NN residuals shared by both variances (rdrobust.R:750-754)
        res_h = rdrobust_res_nn(eX, eY, nnmatch, edups, edupsid)
        # Conventional / robust meats (rdrobust.R:762-764, 789-798)
        V_cl = invG_p @ rdrobust_vce_sharp(R_p * W_h[:, None], res_h) @ invG_p
        V_rb = invG_p @ rdrobust_vce_sharp(Q_q, res_h) @ invG_p
        return beta_p, beta_bc, V_cl, V_rb, N_h, N_b

    beta_p_l, beta_bc_l, V_cl_l, V_rb_l, N_h_l, N_b_l = _side(
        X_l, Y_l, h_l, b_l, dups_l, dupsid_l, "left"
    )
    beta_p_r, beta_bc_r, V_cl_r, V_rb_r, N_h_r, N_b_r = _side(
        X_r, Y_r, h_r, b_r, dups_r, dupsid_r, "right"
    )

    # factorial(deriv) scaling per rdrobust.R:621-622 (deriv=0 -> 1).
    fact = float(math.factorial(deriv))
    tau_cl = fact * float(beta_p_r[deriv] - beta_p_l[deriv])
    tau_bc = fact * float(beta_bc_r[deriv] - beta_bc_l[deriv])
    bias_l = fact * float(beta_p_l[deriv]) - fact * float(beta_bc_l[deriv])
    bias_r = fact * float(beta_p_r[deriv]) - fact * float(beta_bc_r[deriv])
    V_tau_cl = fact**2 * float((V_cl_l + V_cl_r)[deriv, deriv])
    V_tau_rb = fact**2 * float((V_rb_l + V_rb_r)[deriv, deriv])
    return RdFitSharpResult(
        tau_cl=tau_cl,
        tau_bc=tau_bc,
        se_cl=float(np.sqrt(V_tau_cl)),
        se_rb=float(np.sqrt(V_tau_rb)),
        bias_l=bias_l,
        bias_r=bias_r,
        beta_p_l=beta_p_l,
        beta_p_r=beta_p_r,
        N_h_l=N_h_l,
        N_h_r=N_h_r,
        N_b_l=N_b_l,
        N_b_r=N_b_r,
    )
