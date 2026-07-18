"""In-house port of rdrobust's RD bandwidth-selection and estimation
machinery - sharp, fuzzy, and covariate-adjusted paths.

Faithful Python translation of the sharp, fuzzy, and covariate-adjusted
no-cluster ``nn`` branches of ``rdbwselect`` and ``rdrobust`` from the R
package ``rdrobust`` 4.0.0,
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
``rdrobust_vce(RX, res)``             ``rdrobust_vce`` null-cluster
                                            branches (functions.R:374-385)
``rdrobust_bw(...)``                        ``rdrobust_bw``
                                            (functions.R:207-355)
``rdbwselect(...)``                   ``rdbwselect`` main flow
                                            (rdbwselect.R; anchors inline)
``covs_drop_fun(z)``                        ``covs_drop_fun``
                                            (functions.R:683-688) via
                                            LINPACK dqrdc2 rank/pivot
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
* Degenerate covariate adjustment is GUARDED instead of reproduced: R's
  ``ginv(ZWZ, tol=1e-20)`` inverts a float-noise singular value on
  exactly-degenerate partialled systems (constant covariate, full dummy
  set), making its output platform-noise. See :func:`_covs_gamma` for the
  guard (per-column exclusion + scale-invariant stabilized cut + warning);
  well-posed systems reproduce R exactly. Rank-0 covariate matrices fail
  closed with a clear error.

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
    "rdrobust_vce",
    "covs_drop_fun",
    "rdrobust_bw",
    "rdbwselect",
    "quantile_type2",
    "RdFitResult",
    "rdrobust_fit",
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
    t: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Nearest-neighbor variance residuals (functions.R:146-181,
    ``vce == "nn"`` branch).

    Abadie-Imbens NN sigma via same-side neighbors on the SORTED ``x``.
    Ties are matched as whole ``dups``/``dupsid`` blocks; distances compare
    EXACTLY (4.0.0 semantics - the 4.1.0-dev ``nn_tol`` tolerance is
    deliberately absent). Equal left/right distances expand BOTH directions
    (functions.R:162-165). Returns the (n,) residual vector
    ``sqrt(J/(J+1)) * (y_i - mean(y_neighbors))`` for the sharp
    outcome-only case, or the (n, 1+dT+dZ) residual matrix with the fuzzy
    take-up column (functions.R:171-174) and covariate columns
    (functions.R:175-180) appended when ``t`` / ``z`` are supplied - the
    extra responses share Y's neighbor sets exactly (all depend only on
    ``x``). Column order matches R's response stack: [Y, T, Z...].
    """
    n = y.shape[0]
    fuzzy = t is not None
    dZ = 0 if z is None else z.shape[1]
    ncol = 1 + (1 if fuzzy else 0) + dZ
    res = np.empty((n, ncol) if ncol > 1 else n, dtype=np.float64)
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
        r_y = np.sqrt(Ji / (Ji + 1)) * (y[pos] - y_J / Ji)
        if ncol == 1:
            res[pos] = r_y
            continue
        res[pos, 0] = r_y
        col = 1
        if fuzzy:
            assert t is not None
            t_J = float(np.sum(t[lo : hi + 1])) - float(t[pos])  # functions.R:172
            res[pos, col] = np.sqrt(Ji / (Ji + 1)) * (t[pos] - t_J / Ji)
            col += 1
        for i in range(dZ):  # functions.R:175-180
            assert z is not None
            z_J = float(np.sum(z[lo : hi + 1, i])) - float(z[pos, i])
            res[pos, col + i] = np.sqrt(Ji / (Ji + 1)) * (z[pos, i] - z_J / Ji)
    return res


def rdrobust_vce(RX: np.ndarray, res: np.ndarray, s: Optional[np.ndarray] = None) -> np.ndarray:
    """Variance meat, no-cluster case (functions.R:374-385).

    Sharp (``s=None``, d==0): ``M = crossprod(res * RX)`` with the (n,)
    residual vector. Fuzzy (d>0): the (n, 1+d) residual matrix is collapsed
    by the linear-combination vector first, ``r_comb = res %*% s``, then
    ``M = crossprod(r_comb * RX)`` - the Y-T covariance materializes in the
    ``2*s[0]*s[1]*res_Y*res_T`` cross term (functions.R:379-385). ``s`` is
    the delta-method vector ``s_Y`` for the ratio variance or the selector
    ``sV_T = [0, 1]`` for the first-stage variance.
    """
    if s is None:
        scaled = res[:, None] * RX
    else:
        r_comb = res @ s
        scaled = r_comb[:, None] * RX
    return scaled.T @ scaled


def _var0(v: np.ndarray) -> bool:
    """R's exact ``var(T_side) == 0`` check (rdrobust.R:179). A
    single-element side is treated as no-variation: R's ``var()`` would be
    NA there and crash the ``if()`` opaquely; zero variation is the
    semantically correct fail-closed reading and feeds the same
    perf_comp/identification logic. Implemented as exact constancy rather
    than ``np.var(...) == 0.0``: R's two-pass ``mean()`` makes its var of
    a constant vector exactly zero, while numpy's single-pass mean leaves
    ~1e-32 roundoff for constants like 0.7 - the exact-constancy test is
    the faithful translation (var == 0 iff constant)."""
    return v.shape[0] < 2 or bool(np.all(v == v[0]))


def _fuzzy_identification_stop(t_l: np.ndarray, t_r: np.ndarray) -> None:
    """Fuzzy identification guard, R-exact condition and message
    (rdrobust.R:175-177 == rdbwselect.R:339-341): reject only the FULLY
    degenerate first stage - zero variance on BOTH sides AND no mean jump
    at the cutoff. One-sided zero variance is a legitimate design
    (one-sided perfect compliance -> perf_comp bandwidth switch)."""
    if t_l.shape[0] == 0 or t_r.shape[0] == 0:
        # An empty side is the one-sided-data failure, not a first-stage
        # identification failure - defer to the targeted one-sided error
        # downstream instead of np.mean-ing an empty array here.
        return
    if (
        _var0(t_l)
        and _var0(t_r)
        and abs(float(np.mean(t_l)) - float(np.mean(t_r)))
        < float(np.sqrt(np.finfo(np.float64).eps))
    ):
        raise ValueError(
            "Fuzzy RD: first-stage variable has no variation and no jump "
            "at the cutoff. The fuzzy estimator is not identified."
        )


def covs_drop_fun(z: np.ndarray, tol: float = 1e-7) -> Tuple[np.ndarray, int]:
    """Redundant-covariate detection: R's ``covs_drop_fun``
    (functions.R:683-688) = ``qr(z, tol=1e-7)`` rank/pivot, keep
    ``sort(pivot[1:rank])``.

    R's default ``qr()`` is LINPACK ``dqrdc2``, whose limited pivoting
    cycles a column to the right edge when its REDUCED norm falls below
    ``tol`` times that column's OWN original norm (zero-norm columns take
    an original norm of 1.0, dqrdc2.f:8) - a per-column relative rule, so
    small-but-independent covariates are never dropped. LAPACK's ``geqp3``
    pivots differently (greedy by current norm), so the dqrdc2 loop is
    ported directly rather than approximated; pivot order decides WHICH of
    a collinear set survives. Returns ``(keep, rank)`` with ``keep`` the
    sorted 0-based indices of retained columns.
    """
    x = np.array(z, dtype=np.float64, copy=True)
    n, p = x.shape
    jpvt = np.arange(p)
    qraux = np.sqrt((x * x).sum(axis=0))
    work1 = qraux.copy()  # dqrdc2 work(j,1): recompute reference norm
    work2 = qraux.copy()  # dqrdc2 work(j,2): original norm for the tol test
    work2[work2 == 0.0] = 1.0  # dqrdc2.f:8 zero-norm fixup
    k = p + 1
    rank = 0
    for ll in range(min(n, p)):
        # Cycle negligible columns to the right edge (dqrdc2.f:80-120);
        # the ll < k-1 guard prevents infinite cycling.
        while ll < k - 1 and qraux[ll] < work2[ll] * tol:
            x[:, ll:p] = np.roll(x[:, ll:p], -1, axis=1)
            jpvt[ll:p] = np.roll(jpvt[ll:p], -1)
            qraux[ll:p] = np.roll(qraux[ll:p], -1)
            work1[ll:p] = np.roll(work1[ll:p], -1)
            work2[ll:p] = np.roll(work2[ll:p], -1)
            k -= 1
        rank = ll + 1
        # Householder for column ll + LINPACK norm downdate with the
        # 0.05-heuristic recompute (dqrdc2.f main loop).
        nrmxl = float(np.sqrt((x[ll:, ll] ** 2).sum()))
        if nrmxl == 0.0:
            continue
        if x[ll, ll] != 0.0:
            nrmxl = float(np.copysign(nrmxl, x[ll, ll]))
        x[ll:, ll] /= nrmxl
        x[ll, ll] += 1.0
        for j in range(ll + 1, p):
            tval = -(x[ll:, ll] @ x[ll:, j]) / x[ll, ll]
            x[ll:, j] += tval * x[ll:, ll]
            if qraux[j] != 0.0:
                tt = 1.0 - (abs(x[ll, j]) / qraux[j]) ** 2
                tt = max(tt, 0.0)
                t_keep = tt
                tt = 1.0 + 0.05 * tt * (qraux[j] / work1[j]) ** 2
                if tt != 1.0:
                    qraux[j] *= float(np.sqrt(t_keep))
                else:
                    qraux[j] = float(np.sqrt((x[ll + 1 :, j] ** 2).sum()))
                    work1[j] = qraux[j]
        qraux[ll] = 0.0
    rank = min(rank, k - 1)
    keep = np.sort(jpvt[:rank])
    return keep, int(rank)


def _covs_gamma(
    ZWZ: np.ndarray,
    ZWY: np.ndarray,
    diag_pre: np.ndarray,
    covs_drop: bool,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Covariate-projection solve ``gamma`` from the partialled normal
    equations (rdrobust.R:659-671 / functions.R:246-257).

    ``covs_drop=False``: R's ``chol2inv(chol(ZWZ))`` - a strict Cholesky
    solve that fails hard on a collinear system (clear ``ValueError``
    here instead of R's opaque ``chol()`` error).

    ``covs_drop=True``: R uses ``MASS::ginv(ZWZ, tol=1e-20)``. On a
    well-posed system that equals ``np.linalg.pinv(rcond=1e-20)`` and is
    reproduced exactly. On an EXACTLY-degenerate system (covariates
    collinear with the local polynomial design after partialling: a
    constant covariate, or a full dummy set - both pass the intercept-free
    ``covs_drop_fun`` QR check) R inverts a FLOAT-NOISE singular value
    (~1e-16 * sv_max > 1e-20 * sv_max), making its gamma platform-noise
    and shifting tau silently. Documented deviation from R - guarded
    instead of reproduced:

    * per-column: ``diag(ZWZ)_j / (z_j' W z_j) < 1e-14`` means column j is
      numerically fully explained by the design -> excluded (gamma row 0;
      a constant covariate then contributes exactly nothing and the fit
      matches the one without it to floating-point roundoff - not
      bit-for-bit, because the response matrix still carries the excluded
      column and BLAS matmul kernels differ with matrix SHAPE on some
      platforms);
    * set-level: equilibrated (scale-invariant) singular values of the
      remaining block with ``sv_min < 1e-12 * sv_max`` -> stabilized
      equilibrated pseudo-inverse with the noise directions cut
      (``rcond=1e-12``); tau then equals any identified reparametrization
      of the same covariate span (e.g. dropping one dummy category);
    * otherwise the raw ``pinv(rcond=1e-20)`` solve, R-identical
      (tiny-SCALED independent covariates stay on this path - the
      equilibration makes the check scale-invariant).

    Returns ``(gamma, excluded_mask, set_degenerate)``.
    """
    dZ = ZWZ.shape[0]
    n_rhs = ZWY.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.diag(ZWZ) / diag_pre
    excluded = ratio < 1e-14  # NaN (0/0) compares False -> handled below
    excluded |= ~np.isfinite(ratio)
    keep = np.flatnonzero(~excluded)
    set_degenerate = False
    zwz_k = dvec = eq = None
    if keep.size > 0:
        zwz_k = ZWZ[np.ix_(keep, keep)]
        dvec = np.sqrt(np.diag(zwz_k))
        eq = zwz_k / np.outer(dvec, dvec)
        sv = np.linalg.svd(eq, compute_uv=False)
        set_degenerate = bool(sv[-1] < 1e-12 * sv[0])
    if not covs_drop:
        # Strict mode: fail DETERMINISTICALLY on any degeneracy. (R's
        # covs_drop=FALSE relies on chol() erroring, which on an
        # exactly-singular float matrix is roundoff-dependent - it can
        # "succeed" through a tiny positive pivot and return noise; the
        # explicit check makes the strict contract reliable.)
        if excluded.any() or set_degenerate or keep.size == 0:
            raise ValueError(
                "Covariates are collinear with each other or with the "
                "local polynomial design (the partialled covariate Gram "
                "matrix is singular) and covs_drop=False requests a "
                "strict solve. Remove the redundant covariates or use "
                "covs_drop=True."
            )
        try:
            cf = _scipy_linalg.cho_factor(ZWZ, lower=False)
            gamma = _scipy_linalg.cho_solve(cf, ZWY)
        except _scipy_linalg.LinAlgError:
            raise ValueError(
                "Covariates are collinear (the partialled covariate Gram "
                "matrix is not positive definite) and covs_drop=False "
                "requests a strict solve. Remove the redundant covariates "
                "or use covs_drop=True."
            ) from None
        return gamma, np.zeros(dZ, dtype=bool), False
    gamma = np.zeros((dZ, n_rhs))
    if keep.size == 0:
        return gamma, excluded, True
    assert zwz_k is not None and dvec is not None and eq is not None
    zwy = ZWY[keep]
    if set_degenerate:
        gamma_k = np.linalg.pinv(eq, rcond=1e-12) @ (zwy / dvec[:, None])
        gamma_k /= dvec[:, None]
    else:
        gamma_k = np.linalg.pinv(zwz_k, rcond=1e-20) @ zwy
    gamma[keep] = gamma_k
    return gamma, excluded, set_degenerate


def _covs_entry_drop(
    covs: np.ndarray, covs_drop: bool, warn: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Entry-point redundant-covariate drop, shared by :func:`rdbwselect`
    and :func:`rdrobust_fit` (rdbwselect.R:164-181 == rdrobust.R:121-140,
    minus the name-length column sort - the port takes an unnamed matrix,
    for which R's ``order(nchar(...))`` sort is a stable no-op; the
    estimator applies the name sort before building the matrix).

    Returns ``(reduced_covs, dropped_indices)``. Rank 0 fails closed with
    a clear error (R would index a nonexistent column downstream).
    """
    dZ = covs.shape[1]
    if not covs_drop:
        return covs, np.array([], dtype=np.int64)
    keep, rank = covs_drop_fun(covs)
    if rank == 0:
        raise ValueError(
            "All covariates are numerically zero (rank-0 covariate "
            "matrix); remove the covariates instead."
        )
    if rank < dZ:
        dropped = np.setdiff1d(np.arange(dZ), keep)
        if warn:
            # R's message (rdrobust.R:138) with the dropped 0-based column
            # indices appended; the estimator maps indices to column names
            # and warns itself instead.
            warnings.warn(
                "Multicollinearity issue detected in covs. Redundant "
                f"covariates dropped (column indices {dropped.tolist()}).",
                UserWarning,
                stacklevel=3,
            )
        return covs[:, keep], dropped
    return covs, np.array([], dtype=np.int64)


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
    t: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    covs_drop: bool = True,
    vcache: Optional[Dict[str, Tuple[float, float, Optional[np.ndarray]]]] = None,
) -> _BwPilot:
    """Per-side pilot V/B(/R) block (functions.R:207-355, sharp, fuzzy,
    and covariate-adjusted paths).

    Sharp (``t=None, z=None``): C = W = NULL so the combination vector
    ``s`` is the scalar 1 (functions.R:234) and the response is the
    outcome column alone. Fuzzy: T is stacked as a second response column
    into BOTH the V-fit and B-fit designs (functions.R:236-240, 315-318)
    and the pilot ratio + delta vector ``s = [1/tau_T, -tau_Y/tau_T^2]``
    is computed from the V-fit coefficients (functions.R:264-268).
    Covariates: Z stacks after T; a PER-PILOT gamma comes from the
    partialled normal equations inside the V-window (functions.R:241-258;
    degenerate systems take the silent stabilized solve documented at
    :func:`_covs_gamma`), giving ``s = [1, -gamma[,1]]`` (sharp+covs) or
    the length-(2+dZ) fuzzy+covs vector of functions.R:269-274. ``s``
    threads into the V/B variance meats and the bias constant
    ``t(s) %*% beta_B[o+2,]`` (functions.R:294, 346, 349). A pilot window
    with no take-up variation makes ``tau_T == 0``; the division follows
    R's Inf/NaN flow-on (numpy float under ``errstate``) and the
    downstream stage assembly fails closed on the non-finite bandwidth.
    ``vcache`` shares the fixed-``h_V`` V-fit across pilot calls keyed on
    ``(o, nu)`` (functions.R:216-222) and stores ``(V_V, BConst, s)`` -
    the cached ``V_V`` embeds the fuzzy/covariate ``s``, so ``s`` must be
    reused on cache hits exactly as R's environment cache does.
    """
    if vce != "nn":
        raise NotImplementedError(
            "Only vce='nn' is ported in v1; rdrobust's hc0-hc3 and cluster "
            "variance modes are a documented seam."
        )
    key = f"{o}_{nu}"
    s: Optional[np.ndarray]
    if vcache is not None and key in vcache:
        V_V, BConst, s = vcache[key]  # functions.R:218-222
    else:
        # --- V-fit at (o, nu), bandwidth h_V (functions.R:226-299) ---
        w = rdrobust_kweight(x, c, h_V, kernel)
        ind_V = w > 0
        eY = y[ind_V]
        eX = x[ind_V]
        eW = w[ind_V]
        R_V = rdrobust_vander(eX - c, o)
        invG_V = qrXXinv(R_V * np.sqrt(eW)[:, None])
        eT = t[ind_V] if t is not None else None  # functions.R:236-240
        eZ = z[ind_V] if z is not None else None  # functions.R:241-244
        if eT is None and eZ is None:
            # R computes beta_V here (functions.R:263) but the sharp/nn
            # path never consumes it (it feeds the fuzzy ratio and hc
            # predictions); omitted - no numeric effect on V, B, or R.
            s = None
            res_V = rdrobust_res_nn(eX, eY, nnmatch, dups[ind_V], dupsid[ind_V])  # functions.R:293
        else:
            dT = 0 if eT is None else 1
            resp = [eY] + ([eT] if eT is not None else [])
            D_V = np.column_stack(resp + ([eZ] if eZ is not None else []))
            s = None
            gamma = None
            if eZ is not None:
                # Per-pilot partialled gamma (functions.R:245-257).
                U = (R_V * eW[:, None]).T @ D_V
                ZWD = (eZ * eW[:, None]).T @ D_V
                colsZ = slice(1 + dT, D_V.shape[1])
                UiGU = U[:, colsZ].T @ (invG_V @ U)
                gamma, _, _ = _covs_gamma(
                    ZWD[:, colsZ] - UiGU[:, colsZ],
                    ZWD[:, : 1 + dT] - UiGU[:, : 1 + dT],
                    np.diag(ZWD[:, colsZ]).copy(),
                    covs_drop,
                )
                s = np.concatenate([[1.0], -gamma[:, 0]])  # functions.R:257
            beta_V = invG_V @ (R_V * eW[:, None]).T @ D_V  # functions.R:263
            if eT is not None and eZ is None:
                # Fuzzy pilot ratio + delta vector (functions.R:264-268); R
                # row nu+1 (1-based) is 0-based nu.
                tau_Y = float(math.factorial(nu)) * float(beta_V[nu, 0])
                tau_T = float(math.factorial(nu)) * float(beta_V[nu, 1])
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = np.array(
                        [
                            float(np.float64(1.0) / np.float64(tau_T)),
                            float(-(np.float64(tau_Y) / np.float64(tau_T) ** 2)),
                        ]
                    )
            elif eT is not None and eZ is not None:
                # Fuzzy + covariates (functions.R:269-274): adjusted ratio
                # from the covariate-combined coefficients, then the
                # extended delta vector.
                assert gamma is not None and s is not None
                s_T = np.concatenate([[1.0], -gamma[:, 1]])
                colsZ = slice(2, D_V.shape[1])
                tau_Y = float(math.factorial(nu)) * float(
                    s @ np.concatenate([[beta_V[nu, 0]], beta_V[nu, colsZ]])
                )
                tau_T = float(math.factorial(nu)) * float(
                    s_T @ np.concatenate([[beta_V[nu, 1]], beta_V[nu, colsZ]])
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv_tT = float(np.float64(1.0) / np.float64(tau_T))
                    ratio2 = float(np.float64(tau_Y) / np.float64(tau_T) ** 2)
                    s = np.concatenate(
                        [
                            [inv_tT, -ratio2],
                            -inv_tT * gamma[:, 0] + ratio2 * gamma[:, 1],
                        ]
                    )
            res_V = rdrobust_res_nn(eX, eY, nnmatch, dups[ind_V], dupsid[ind_V], t=eT, z=eZ)
        aux = rdrobust_vce(R_V * eW[:, None], res_V, s)  # functions.R:294
        V_V = float((invG_V @ aux @ invG_V)[nu, nu])  # functions.R:295
        v = (R_V * eW[:, None]).T @ ((eX - c) / h_V) ** (o + 1)  # :296
        Hp = h_V ** np.arange(o + 1, dtype=np.float64)  # functions.R:297-298
        BConst = float((Hp * (invG_V @ v))[nu])  # functions.R:299
        if vcache is not None:
            vcache[key] = (V_V, BConst, s)
    # --- B-fit at o_B, bandwidth h_B (functions.R:306-348) ---
    w = rdrobust_kweight(x, c, h_B, kernel)
    ind = w > 0
    eY = y[ind]
    eX = x[ind]
    eW = w[ind]
    R_B = rdrobust_vander(eX - c, o_B)
    invG_B = qrXXinv(R_B * np.sqrt(eW)[:, None])
    eT_B = t[ind] if t is not None else None  # functions.R:315-318
    eZ_B = z[ind] if z is not None else None  # functions.R:319-322
    if eT_B is None and eZ_B is None:
        beta_B = invG_B @ (R_B * eW[:, None]).T @ eY  # functions.R:326
        # functions.R:349-353 with sharp s == 1: t(s) %*% beta_B[o+2,] is
        # the scalar coefficient (R row o+2 1-based = 0-based o+1).
        beta_B_comb = float(beta_B[o + 1])
    else:
        resp_B = [eY] + ([eT_B] if eT_B is not None else [])
        D_B = np.column_stack(resp_B + ([eZ_B] if eZ_B is not None else []))
        beta_B = invG_B @ (R_B * eW[:, None]).T @ D_B  # functions.R:326
        assert s is not None
        beta_B_comb = float(s @ beta_B[o + 1, :])  # functions.R:349
    BWreg = 0.0
    if scale > 0:  # functions.R:328-348
        res_B = rdrobust_res_nn(eX, eY, nnmatch, dups[ind], dupsid[ind], t=eT_B, z=eZ_B)
        V_B = float(
            (invG_B @ rdrobust_vce(R_B * eW[:, None], res_B, s) @ invG_B)[o + 1, o + 1]
        )  # functions.R:346 - R row/col o+2 is 0-based (o+1, o+1)
        BWreg = 3.0 * BConst**2 * V_B  # functions.R:347
    B = float(np.sqrt(2.0 * (o + 1 - nu)) * BConst * beta_B_comb)
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


def rdbwselect(
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
    fuzzy: Optional[np.ndarray] = None,
    sharpbw: bool = False,
    covs: Optional[np.ndarray] = None,
    covs_drop: bool = True,
) -> RdBwselectResult:
    """RD data-driven bandwidth selection, all 10 selectors
    (rdbwselect.R main flow at the anchors cited inline; sharp, fuzzy,
    and covariate-adjusted paths).

    Always computes the full selector matrix (R's ``all=TRUE``): the ten
    selectors share the same six per-side pilot blocks, so the marginal
    cost is a handful of scalar operations. Inputs must be complete-case
    1-D arrays (see module docstring for the deviation from R's silent
    ``complete.cases`` drop).

    Fuzzy (``fuzzy`` = observed take-up variable): bandwidths are selected
    on the FUZZY RATIO objective - T is threaded into every pilot call of
    all three chains so the pilot V/B constants are those of the ratio
    estimator (rdbwselect.R:386-457; the port computes the mserd, msetwo,
    AND msesum chains unconditionally, so T must reach all 14
    ``rdrobust_bw`` call sites, not just R's default single chain). When
    ``sharpbw=True`` OR either side has zero take-up variance
    (``perf_comp``, one-sided perfect compliance), T is nulled for
    SELECTION ONLY and the sharp reduced-form objective on Y is used
    (rdbwselect.R:334-346); estimation always remains fuzzy.

    Covariates (``covs`` = (n, dZ) matrix): bandwidths are
    COVARIATE-AWARE - after the entry-point redundant-column drop
    (rdbwselect.R:164-181, ``covs_drop``), Z is threaded into every pilot
    of all three chains alongside T (rdbwselect.R:330-332, 386-457), so
    the pilot V/B constants are those of the covariate-adjusted estimator.
    ``perf_comp``/``sharpbw`` null ONLY T - Z always stays in selection
    (rdbwselect.R:343-345). Standardize note: R's ``stdvars`` scales y and
    x only - the fuzzy and covariate columns are never standardized
    (rdbwselect.R:120-129).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    arrs = [("y", y), ("x", x)]
    if fuzzy is not None:
        fuzzy = np.asarray(fuzzy, dtype=np.float64)
        arrs.append(("fuzzy", fuzzy))
    for name, arr in arrs:
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
    if fuzzy is not None:
        fuzzy = fuzzy.reshape(-1)
        if fuzzy.shape[0] != x.shape[0]:
            raise ValueError(
                f"fuzzy must have length equal to x; got {fuzzy.shape[0]} vs {x.shape[0]}."
            )
        if not np.all(np.isfinite(fuzzy)):
            raise ValueError(
                "fuzzy must be finite and complete-case; drop or impute "
                "missing values before bandwidth selection (the public "
                "estimator warns-and-drops; R's complete.cases filter "
                "includes the fuzzy column)."
            )
    if not isinstance(covs_drop, (bool, np.bool_)):
        raise ValueError(f"covs_drop must be a bool; got {covs_drop!r}.")
    if covs is not None:
        covs = np.asarray(covs, dtype=np.float64)
        if covs.ndim == 1:
            covs = covs.reshape(-1, 1)
        if covs.ndim != 2:
            raise ValueError(
                f"covs must be a 1-D vector or (n, dZ) matrix; got shape {covs.shape}."
            )
        if covs.shape[0] != x.shape[0]:
            raise ValueError(f"covs must have {x.shape[0]} rows to match x; got {covs.shape[0]}.")
        if not np.all(np.isfinite(covs)):
            raise ValueError(
                "covs must be finite and complete-case; drop or impute "
                "missing values before bandwidth selection (the public "
                "estimator warns-and-drops; R's complete.cases filter "
                "includes the covariate columns)."
            )
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
    if not isinstance(sharpbw, (bool, np.bool_)):
        # Same strict-bool contract as the estimator: a truthy non-bool
        # must not silently flip the bandwidth objective.
        raise ValueError(f"sharpbw must be a bool; got {sharpbw!r}.")
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
        if fuzzy is not None:
            fuzzy = fuzzy[order_x]  # rdbwselect.R:112 (fuzzy = fuzzy[order_x,])
        if covs is not None:
            covs = covs[order_x]  # rdbwselect.R:110

    # --- Entry-point redundant-covariate drop (rdbwselect.R:164-181),
    # after the row sort so near-threshold QR rank decisions see the same
    # row order as R (and as rdrobust_fit). ---
    if covs is not None:
        covs, _ = _covs_entry_drop(covs, covs_drop)

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

    # --- Fuzzy first-stage split + identification + perf_comp
    # (rdbwselect.R:334-346; runs after the masspoints block, mirroring
    # R's rdbwselect ordering - R's rdrobust() checks identification
    # FIRST, which the estimator mirrors at fit() level). ---
    T_sel_l: Optional[np.ndarray] = None
    T_sel_r: Optional[np.ndarray] = None
    if fuzzy is not None:
        T_l_full = fuzzy[ind_l]
        T_r_full = fuzzy[ind_r]
        _fuzzy_identification_stop(T_l_full, T_r_full)
        perf_comp = _var0(T_l_full) or _var0(T_r_full)  # rdbwselect.R:343
        if not (perf_comp or sharpbw):  # rdbwselect.R:344-346 null-out
            T_sel_l, T_sel_r = T_l_full, T_r_full

    # --- Covariate split (rdbwselect.R:330-332). Z is NEVER nulled by
    # perf_comp/sharpbw - those switches drop only T (rdbwselect.R:344),
    # so sharpbw-with-covariates selects on the covariate-adjusted sharp
    # objective. ---
    Z_sel_l: Optional[np.ndarray] = None
    Z_sel_r: Optional[np.ndarray] = None
    if covs is not None:
        Z_sel_l = covs[ind_l]
        Z_sel_r = covs[ind_r]

    # --- NN tie blocks (rdbwselect.R:322-327) ---
    dups_l, dupsid_l = compute_dups_dupsid(X_l)
    dups_r, dupsid_r = compute_dups_dupsid(X_r)

    vcache_l: Dict[str, Tuple[float, float, Optional[np.ndarray]]] = {}
    vcache_r: Dict[str, Tuple[float, float, Optional[np.ndarray]]] = {}

    def _bw(side: str, o: int, nu: int, o_B: int, h_B: float, scale: float) -> _BwPilot:
        # Single funnel for ALL 14 pilot calls across the mserd, msetwo,
        # and msesum chains: threading T and Z here guarantees every
        # chain's pilots receive the fuzzy and covariate columns (R passes
        # T_l/T_r, Z_l/Z_r into each chain's calls individually,
        # rdbwselect.R:386-457).
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
                t=T_sel_l,
                z=Z_sel_l,
                covs_drop=bool(covs_drop),
                vcache=vcache_l,
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
            t=T_sel_r,
            z=Z_sel_r,
            covs_drop=bool(covs_drop),
            vcache=vcache_r,
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
class RdFitResult:
    """RD point estimates and variances (rdrobust.R estimation body,
    sharp, fuzzy, and covariate-adjusted paths).

    ``tau_cl`` is the conventional RD estimate (the fuzzy ratio
    ``tau_Y_cl/tau_T_cl`` on fuzzy fits), ``tau_bc`` the bias-corrected
    estimate (linearized on fuzzy fits); ``se_cl``/``se_rb`` are the
    conventional and robust bias-corrected standard errors. rdrobust's
    three output rows map as Conventional = (tau_cl, se_cl),
    Bias-Corrected = (tau_bc, se_cl), Robust = (tau_bc, se_rb)
    (rdrobust.R:854-863). ``beta_p_l``/``beta_p_r`` are the per-side
    order-p outcome coefficient vectors (rdplot seam; on
    covariate-adjusted fits these are the ADJUSTED vectors ``s_Y``
    applied across the response columns, matching R's ``beta_Y_p_*``,
    rdrobust.R:685-686/706-709); ``bias_l``/``bias_r`` the per-side
    estimated biases (sharp: rdrobust.R:629-630; fuzzy: the LINEARIZED
    ``s_Y . B_F_side``, rdrobust.R:649-652 - a different formula, not the
    per-component difference). Fuzzy-only fields (None on sharp fits):
    the first-stage ``tau_T_cl/tau_T_bc/se_T_cl/se_T_rb``
    (rdrobust.R:637-638, 800-822) and per-side take-up coefficient
    vectors ``beta_t_p_l/beta_t_p_r`` (raw, like ``beta_p_*``; R applies
    ``scalepar*factorial(deriv)`` to both - identical at the public
    deriv=0/scalepar=1 surface). Covariate-only fields (None otherwise):
    ``gamma_p`` = R's ``coef_covs``, the (dZ, 1+dT) common projection
    coefficients over the covariates KEPT by the entry-point drop
    (column 0 = outcome equation, column 1 = first-stage equation on
    fuzzy fits, rdrobust.R:907); ``covs_excluded`` = per-kept-column
    bool mask of covariates excluded by the degeneracy guard (see
    :func:`_covs_gamma`); ``covs_set_degenerate`` = True when the
    set-level stabilized cut engaged.
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
    tau_T_cl: Optional[float] = None
    tau_T_bc: Optional[float] = None
    se_T_cl: Optional[float] = None
    se_T_rb: Optional[float] = None
    beta_t_p_l: Optional[np.ndarray] = None
    beta_t_p_r: Optional[np.ndarray] = None
    gamma_p: Optional[np.ndarray] = None
    covs_excluded: Optional[np.ndarray] = None
    covs_set_degenerate: bool = False


def rdrobust_fit(
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
    t: Optional[np.ndarray] = None,
    covs: Optional[np.ndarray] = None,
    covs_drop: bool = True,
    warn_covs_degenerate: bool = True,
) -> RdFitResult:
    """RD estimation at known bandwidths (rdrobust.R:533-822, sharp,
    fuzzy, and covariate-adjusted no-cluster paths with ``scalepar = 1``).

    Inputs must be complete-case 1-D arrays (same contract as
    :func:`rdbwselect`); sorting, side-splitting, and NN tie blocks
    are handled internally. Per-side bandwidths follow rdrobust's
    ``bws = [[h_l, b_l], [h_r, b_r]]`` layout. Steps:

    1. Effective window per side = observations with positive kernel weight
       inside the WIDER of h and b (rdrobust.R:541-549); observations inside
       the b-window but outside the h-window carry ``W_h = 0`` and drop out
       of the p-regression through the weights.
    2. Point estimates: order-p WLS per side at h (``beta_p``); the
       bias-corrected coefficient vector uses the ``Q_q`` score matrix
       (rdrobust.R:577-578, 609-618). Fuzzy (``t`` = observed take-up):
       T is stacked as a second response column through the SAME fits
       (rdrobust.R:581-591), the point estimate is the ratio
       ``tau_Y_cl/tau_T_cl`` and the bias correction is LINEARIZED via
       the delta vector ``s_Y = [1/tau_T_cl, -tau_Y_cl/tau_T_cl^2]``:
       ``tau_bc = tau_cl - s_Y . B_F`` (rdrobust.R:636-657). The
       identification guard (both-sides-constant T with no jump) raises
       here too, covering manual-bandwidth fits that skip selection.
       Covariates (``covs`` = (n, dZ) matrix): after the entry-point
       redundant-column drop (rdrobust.R:121-140, ``covs_drop``), Z
       stacks after T through the SAME fits (rdrobust.R:593-598); a
       common POOLED-ACROSS-SIDES gamma comes from the per-side
       partialled normal equations summed (rdrobust.R:659-671; degenerate
       systems take the guarded solve documented at :func:`_covs_gamma` -
       ``warn_covs_degenerate=False`` lets the estimator own that warning
       with column names, the masspoints pattern), and the adjusted
       estimates apply ``s_Y = [1, -gamma[,1]]`` across the response
       columns (rdrobust.R:672-686; the R branch omits
       ``factorial(deriv)`` present in the no-covariate branch -
       identical at the fixed deriv=0 surface, replicated verbatim).
       Fuzzy + covariates composes both: adjusted Y and T jumps, their
       ratio, and the extended delta vectors of rdrobust.R:688-723.
    3. Variances: conventional sandwiches ``R_p * W_h`` with same-side NN
       residuals; robust sandwiches ``Q_q`` with the SAME residuals
       (``res_b = res_h`` for vce="nn", rdrobust.R:753-754; the h==b
       special branches at rdrobust.R:773-786 are cluster-only and never
       taken on this path). Fuzzy/covariates: the (n, 1+dT+dZ) residual
       matrix is collapsed by the delta vector for the main variance and
       by ``sV_T`` (``[0, 1]``, or ``[0, 1, -gamma[,2]]`` with
       covariates) for the first-stage variance (rdrobust.R:769-822); a
       zero first-stage jump follows R's Inf/NaN flow-on (numpy float
       under ``errstate``).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    arrs = [("y", y), ("x", x)]
    if t is not None:
        t = np.asarray(t, dtype=np.float64)
        arrs.append(("t", t))
    for name, arr in arrs:
        # Same input contract as rdbwselect: 1-D vectors or explicit
        # (n, 1) columns only.
        if not (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)):
            raise ValueError(
                f"{name} must be a 1-D vector or an (n, 1) column; got shape {arr.shape}."
            )
    y = y.reshape(-1)
    x = x.reshape(-1)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"y and x must have equal length; got {y.shape[0]} vs {x.shape[0]}.")
    if t is not None:
        t = t.reshape(-1)
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"t must have length equal to x; got {t.shape[0]} vs {x.shape[0]}.")
        if not np.all(np.isfinite(t)):
            raise ValueError(
                "t must be finite and complete-case; drop or impute missing "
                "values before estimation."
            )
    if not isinstance(covs_drop, (bool, np.bool_)):
        raise ValueError(f"covs_drop must be a bool; got {covs_drop!r}.")
    if covs is not None:
        covs = np.asarray(covs, dtype=np.float64)
        if covs.ndim == 1:
            covs = covs.reshape(-1, 1)
        if covs.ndim != 2:
            raise ValueError(
                f"covs must be a 1-D vector or (n, dZ) matrix; got shape {covs.shape}."
            )
        if covs.shape[0] != x.shape[0]:
            raise ValueError(f"covs must have {x.shape[0]} rows to match x; got {covs.shape[0]}.")
        if not np.all(np.isfinite(covs)):
            raise ValueError(
                "covs must be finite and complete-case; drop or impute "
                "missing values before estimation."
            )
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
    if t is not None:
        t = t[order_x]  # rdrobust.R:115
    if covs is not None:
        covs = covs[order_x]  # rdrobust.R:114
        # Entry-point redundant-covariate drop (rdrobust.R:121-140), after
        # the row sort so near-threshold QR rank decisions see the same
        # row order as R (and as rdbwselect).
        covs, _ = _covs_entry_drop(covs, covs_drop)
    ind_l = x < c
    ind_r = x >= c
    X_l, X_r = x[ind_l], x[ind_r]
    Y_l, Y_r = y[ind_l], y[ind_r]
    if X_l.shape[0] == 0 or X_r.shape[0] == 0:
        raise ValueError(
            "All observations fall on one side of the cutoff; sharp RD "
            "requires data on both sides."
        )
    T_l = t[ind_l] if t is not None else None
    T_r = t[ind_r] if t is not None else None
    if T_l is not None and T_r is not None:
        # rdrobust.R:164-185: the identification guard lives in the
        # estimation entry point too, so manual-bandwidth fuzzy fits that
        # never touch bandwidth selection still fail closed.
        _fuzzy_identification_stop(T_l, T_r)
    Z_l = covs[ind_l] if covs is not None else None  # rdrobust.R:190-193
    Z_r = covs[ind_r] if covs is not None else None
    dups_l, dupsid_l = compute_dups_dupsid(X_l)
    dups_r, dupsid_r = compute_dups_dupsid(X_r)

    def _side(
        X: np.ndarray,
        Y: np.ndarray,
        T: Optional[np.ndarray],
        Z: Optional[np.ndarray],
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
        eT = T[ind] if T is not None else None  # rdrobust.R:588-590
        eZ = Z[ind] if Z is not None else None  # rdrobust.R:594-596
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
        # Point estimates (rdrobust.R:609-614). Fuzzy stacks T as the
        # second response column (rdrobust.R:588-591), covariates stack
        # after T (rdrobust.R:593-598); the sharp branch keeps the
        # original vector products verbatim (bit-identity).
        if eT is None and eZ is None:
            beta_p = invG_p @ (R_p * W_h[:, None]).T @ eY
            beta_bc = invG_p @ Q_q.T @ eY
            res_h = rdrobust_res_nn(eX, eY, nnmatch, edups, edupsid)
            zblocks = None
        else:
            resp = [eY] + ([eT] if eT is not None else [])
            eD = np.column_stack(resp + ([eZ] if eZ is not None else []))
            beta_p = invG_p @ (R_p * W_h[:, None]).T @ eD
            beta_bc = invG_p @ Q_q.T @ eD
            # NN residual matrix, T and Z sharing Y's neighbor sets
            # (rdrobust.R:750-754; functions.R:171-180).
            res_h = rdrobust_res_nn(eX, eY, nnmatch, edups, edupsid, t=eT, z=eZ)
            zblocks = None
            if eZ is not None:
                # Per-side partialled normal-equation blocks
                # (rdrobust.R:597, 659-667); summed across sides by the
                # caller for the POOLED gamma.
                dT_loc = 0 if eT is None else 1
                U_p = (R_p * W_h[:, None]).T @ eD
                ZWD_p = (eZ * W_h[:, None]).T @ eD
                colsZ = slice(1 + dT_loc, eD.shape[1])
                UiGU = U_p[:, colsZ].T @ (invG_p @ U_p)
                zblocks = (
                    ZWD_p[:, colsZ] - UiGU[:, colsZ],
                    ZWD_p[:, : 1 + dT_loc] - UiGU[:, : 1 + dT_loc],
                    np.diag(ZWD_p[:, colsZ]).copy(),
                )
        return beta_p, beta_bc, invG_p, R_p * W_h[:, None], Q_q, res_h, N_h, N_b, zblocks

    (
        beta_p_l,
        beta_bc_l,
        invG_p_l,
        RX_cl_l,
        Q_q_l,
        res_l,
        N_h_l,
        N_b_l,
        zblocks_l,
    ) = _side(X_l, Y_l, T_l, Z_l, h_l, b_l, dups_l, dupsid_l, "left")
    (
        beta_p_r,
        beta_bc_r,
        invG_p_r,
        RX_cl_r,
        Q_q_r,
        res_r,
        N_h_r,
        N_b_r,
        zblocks_r,
    ) = _side(X_r, Y_r, T_r, Z_r, h_r, b_r, dups_r, dupsid_r, "right")

    # ---- Pooled covariate projection gamma (rdrobust.R:659-671) ----
    gamma_p: Optional[np.ndarray] = None
    covs_excluded: Optional[np.ndarray] = None
    covs_set_degenerate = False
    if covs is not None:
        assert zblocks_l is not None and zblocks_r is not None
        gamma_p, covs_excluded, covs_set_degenerate = _covs_gamma(
            zblocks_l[0] + zblocks_r[0],
            zblocks_l[1] + zblocks_r[1],
            zblocks_l[2] + zblocks_r[2],
            covs_drop,
        )
        if warn_covs_degenerate and (covs_excluded.any() or covs_set_degenerate):
            # Deviation from R (which silently inverts a noise singular
            # value here, making the result platform-dependent); the
            # estimator passes warn_covs_degenerate=False and re-warns
            # with column names.
            parts = []
            if covs_excluded.any():
                parts.append(
                    "covariate column(s) at index "
                    f"{np.flatnonzero(covs_excluded).tolist()} are "
                    "numerically collinear with the local polynomial "
                    "design (e.g. constant near the cutoff) and were "
                    "excluded from the adjustment"
                )
            if covs_set_degenerate:
                parts.append(
                    "the covariate set is numerically rank-deficient "
                    "after partialling (e.g. a full dummy set); a "
                    "stabilized pseudo-inverse cut was used - consider "
                    "dropping a reference category"
                )
            warnings.warn(
                "Degenerate covariate adjustment: " + "; ".join(parts) + ".",
                UserWarning,
                stacklevel=2,
            )

    # factorial(deriv) scaling per rdrobust.R:621-622 (deriv=0 -> 1).
    fact = float(math.factorial(deriv))

    def _v(invG_p, RX, res, s):
        # Conventional / robust meats (rdrobust.R:762-764, 789-798); the
        # fuzzy s collapses the residual matrix (functions.R:379-385).
        return invG_p @ rdrobust_vce(RX, res, s) @ invG_p

    if t is None and covs is None:
        tau_cl = fact * float(beta_p_r[deriv] - beta_p_l[deriv])
        tau_bc = fact * float(beta_bc_r[deriv] - beta_bc_l[deriv])
        bias_l = fact * float(beta_p_l[deriv]) - fact * float(beta_bc_l[deriv])
        bias_r = fact * float(beta_p_r[deriv]) - fact * float(beta_bc_r[deriv])
        V_cl_l = _v(invG_p_l, RX_cl_l, res_l, None)
        V_cl_r = _v(invG_p_r, RX_cl_r, res_r, None)
        V_rb_l = _v(invG_p_l, Q_q_l, res_l, None)
        V_rb_r = _v(invG_p_r, Q_q_r, res_r, None)
        V_tau_cl = fact**2 * float((V_cl_l + V_cl_r)[deriv, deriv])
        V_tau_rb = fact**2 * float((V_rb_l + V_rb_r)[deriv, deriv])
        return RdFitResult(
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

    if t is None:
        # ---- Covariate-adjusted sharp assembly (rdrobust.R:672-686,
        # scalepar = 1). NOTE: R's covariate branch applies NO
        # factorial(deriv) to the point estimates/biases (unlike the
        # no-covariate branch at rdrobust.R:621-630) while the VARIANCES
        # keep factorial^2 (rdrobust.R:796-797) - identical at the fixed
        # deriv=0 surface; replicated verbatim, not "fixed". ----
        assert gamma_p is not None
        s_Y = np.concatenate([[1.0], -gamma_p[:, 0]])  # rdrobust.R:672
        tau_cl = float(s_Y @ (beta_p_r[deriv, :] - beta_p_l[deriv, :]))
        tau_bc = float(s_Y @ (beta_bc_r[deriv, :] - beta_bc_l[deriv, :]))
        # Per-side adjusted taus -> biases (rdrobust.R:678-683).
        bias_l = float(s_Y @ beta_p_l[deriv, :]) - float(s_Y @ beta_bc_l[deriv, :])
        bias_r = float(s_Y @ beta_p_r[deriv, :]) - float(s_Y @ beta_bc_r[deriv, :])
        # Adjusted per-side coefficient vectors (rdrobust.R:685-686).
        beta_Y_p_l = s_Y @ beta_p_l.T
        beta_Y_p_r = s_Y @ beta_p_r.T
        V_tau_cl = fact**2 * float(
            (_v(invG_p_l, RX_cl_l, res_l, s_Y) + _v(invG_p_r, RX_cl_r, res_r, s_Y))[deriv, deriv]
        )
        V_tau_rb = fact**2 * float(
            (_v(invG_p_l, Q_q_l, res_l, s_Y) + _v(invG_p_r, Q_q_r, res_r, s_Y))[deriv, deriv]
        )
        return RdFitResult(
            tau_cl=tau_cl,
            tau_bc=tau_bc,
            se_cl=float(np.sqrt(V_tau_cl)),
            se_rb=float(np.sqrt(V_tau_rb)),
            bias_l=bias_l,
            bias_r=bias_r,
            beta_p_l=beta_Y_p_l,
            beta_p_r=beta_Y_p_r,
            N_h_l=N_h_l,
            N_h_r=N_h_r,
            N_b_l=N_b_l,
            N_b_r=N_b_r,
            gamma_p=gamma_p,
            covs_excluded=covs_excluded,
            covs_set_degenerate=covs_set_degenerate,
        )

    if covs is not None:
        # ---- Fuzzy + covariates assembly (rdrobust.R:688-723, 769-822;
        # scalepar = 1): covariate-adjusted Y and T jumps via
        # s_Y = [1, -gamma[,1]] / s_T = [1, -gamma[,2]], their ratio,
        # the linearized bias correction, and the EXTENDED delta vector
        # for the variance collapse. ----
        assert gamma_p is not None
        s_Y0 = np.concatenate([[1.0], -gamma_p[:, 0]])  # rdrobust.R:672
        s_T0 = np.concatenate([[1.0], -gamma_p[:, 1]])  # rdrobust.R:689
        colsZ = slice(2, beta_p_l.shape[1])

        def _adj(bmat: np.ndarray, srow: np.ndarray, col: int) -> float:
            # rdrobust.R:691-704: s applied to [response col, covariate
            # cols] of the (deriv+1) coefficient row.
            return float(srow @ np.concatenate([[bmat[deriv, col]], bmat[deriv, colsZ]]))

        tau_Y_cl = fact * float(_adj(beta_p_r, s_Y0, 0) - _adj(beta_p_l, s_Y0, 0))
        tau_Y_bc = fact * float(_adj(beta_bc_r, s_Y0, 0) - _adj(beta_bc_l, s_Y0, 0))
        tau_T_cl = fact * float(_adj(beta_p_r, s_T0, 1) - _adj(beta_p_l, s_T0, 1))
        tau_T_bc = fact * float(_adj(beta_bc_r, s_T0, 1) - _adj(beta_bc_l, s_T0, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_cl = float(np.float64(tau_Y_cl) / np.float64(tau_T_cl))
            inv_tT = float(np.float64(1.0) / np.float64(tau_T_cl))
            ratio2 = float(np.float64(tau_Y_cl) / np.float64(tau_T_cl) ** 2)
            s_ratio = np.array([inv_tT, -ratio2])  # rdrobust.R:716
            # Extended variance delta vector (rdrobust.R:722).
            s_V = np.concatenate([s_ratio, -inv_tT * gamma_p[:, 0] + ratio2 * gamma_p[:, 1]])
        B_F = np.array([tau_Y_cl - tau_Y_bc, tau_T_cl - tau_T_bc])  # :712
        tau_bc = float(tau_cl - s_ratio @ B_F)  # rdrobust.R:717
        sV_T = np.concatenate([[0.0, 1.0], -gamma_p[:, 1]])  # rdrobust.R:690
        # Per-side linearized biases from the ADJUSTED per-side taus
        # (rdrobust.R:696-704, 713-720).
        B_F_l = np.array(
            [
                fact * (_adj(beta_p_l, s_Y0, 0) - _adj(beta_bc_l, s_Y0, 0)),
                fact * (_adj(beta_p_l, s_T0, 1) - _adj(beta_bc_l, s_T0, 1)),
            ]
        )
        B_F_r = np.array(
            [
                fact * (_adj(beta_p_r, s_Y0, 0) - _adj(beta_bc_r, s_Y0, 0)),
                fact * (_adj(beta_p_r, s_T0, 1) - _adj(beta_bc_r, s_T0, 1)),
            ]
        )
        bias_l = float(s_ratio @ B_F_l)
        bias_r = float(s_ratio @ B_F_r)
        # Adjusted per-side coefficient vectors (rdrobust.R:706-709; the
        # fuzzy-covariate branch DOES carry factorial(deriv), unlike the
        # sharp-covariate one - deriv=0 either way).

        def _adj_vec(bmat: np.ndarray, srow: np.ndarray, col: int) -> np.ndarray:
            return fact * (srow @ np.vstack([bmat[:, col][None, :], bmat[:, colsZ].T]))

        beta_Y_p_l = _adj_vec(beta_p_l, s_Y0, 0)
        beta_Y_p_r = _adj_vec(beta_p_r, s_Y0, 0)
        beta_T_p_l = _adj_vec(beta_p_l, s_T0, 1)
        beta_T_p_r = _adj_vec(beta_p_r, s_T0, 1)
        V_tau_cl = fact**2 * float(
            (_v(invG_p_l, RX_cl_l, res_l, s_V) + _v(invG_p_r, RX_cl_r, res_r, s_V))[deriv, deriv]
        )
        V_tau_rb = fact**2 * float(
            (_v(invG_p_l, Q_q_l, res_l, s_V) + _v(invG_p_r, Q_q_r, res_r, s_V))[deriv, deriv]
        )
        V_T_cl = fact**2 * float(
            (_v(invG_p_l, RX_cl_l, res_l, sV_T) + _v(invG_p_r, RX_cl_r, res_r, sV_T))[deriv, deriv]
        )
        V_T_rb = fact**2 * float(
            (_v(invG_p_l, Q_q_l, res_l, sV_T) + _v(invG_p_r, Q_q_r, res_r, sV_T))[deriv, deriv]
        )
        return RdFitResult(
            tau_cl=tau_cl,
            tau_bc=tau_bc,
            se_cl=float(np.sqrt(V_tau_cl)),
            se_rb=float(np.sqrt(V_tau_rb)),
            bias_l=bias_l,
            bias_r=bias_r,
            beta_p_l=beta_Y_p_l,
            beta_p_r=beta_Y_p_r,
            N_h_l=N_h_l,
            N_h_r=N_h_r,
            N_b_l=N_b_l,
            N_b_r=N_b_r,
            tau_T_cl=tau_T_cl,
            tau_T_bc=tau_T_bc,
            se_T_cl=float(np.sqrt(V_T_cl)),
            se_T_rb=float(np.sqrt(V_T_rb)),
            beta_t_p_l=beta_T_p_l,
            beta_t_p_r=beta_T_p_r,
            gamma_p=gamma_p,
            covs_excluded=covs_excluded,
            covs_set_degenerate=covs_set_degenerate,
        )

    # ---- Fuzzy assembly (rdrobust.R:636-657, 769-822; scalepar = 1) ----
    tau_Y_cl = fact * float(beta_p_r[deriv, 0] - beta_p_l[deriv, 0])
    tau_Y_bc = fact * float(beta_bc_r[deriv, 0] - beta_bc_l[deriv, 0])
    tau_T_cl = fact * float(beta_p_r[deriv, 1] - beta_p_l[deriv, 1])
    tau_T_bc = fact * float(beta_bc_r[deriv, 1] - beta_bc_l[deriv, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        # R flows Inf/NaN through a zero first-stage jump (no guard at
        # rdrobust.R:639-642); numpy-float division mirrors that instead
        # of raising ZeroDivisionError. Non-finite results NaN-gate the
        # downstream inference (estimator contract).
        tau_cl = float(np.float64(tau_Y_cl) / np.float64(tau_T_cl))
        s_Y = np.array(
            [
                float(np.float64(1.0) / np.float64(tau_T_cl)),
                float(-(np.float64(tau_Y_cl) / np.float64(tau_T_cl) ** 2)),
            ]
        )
    B_F = np.array([tau_Y_cl - tau_Y_bc, tau_T_cl - tau_T_bc])  # rdrobust.R:641
    tau_bc = float(tau_cl - s_Y @ B_F)  # rdrobust.R:642 (linearized)
    sV_T = np.array([0.0, 1.0])  # rdrobust.R:643
    # Fuzzy per-side biases are the LINEARIZED s_Y . B_F_side
    # (rdrobust.R:645-652), not the sharp per-component differences.
    B_F_l = np.array(
        [
            fact * float(beta_p_l[deriv, 0] - beta_bc_l[deriv, 0]),
            fact * float(beta_p_l[deriv, 1] - beta_bc_l[deriv, 1]),
        ]
    )
    B_F_r = np.array(
        [
            fact * float(beta_p_r[deriv, 0] - beta_bc_r[deriv, 0]),
            fact * float(beta_p_r[deriv, 1] - beta_bc_r[deriv, 1]),
        ]
    )
    bias_l = float(s_Y @ B_F_l)
    bias_r = float(s_Y @ B_F_r)
    # Ratio variance with s_Y; first-stage variance with sV_T
    # (rdrobust.R:769-798 and 800-822, no-cluster branches).
    V_tau_cl = fact**2 * float(
        (_v(invG_p_l, RX_cl_l, res_l, s_Y) + _v(invG_p_r, RX_cl_r, res_r, s_Y))[deriv, deriv]
    )
    V_tau_rb = fact**2 * float(
        (_v(invG_p_l, Q_q_l, res_l, s_Y) + _v(invG_p_r, Q_q_r, res_r, s_Y))[deriv, deriv]
    )
    V_T_cl = fact**2 * float(
        (_v(invG_p_l, RX_cl_l, res_l, sV_T) + _v(invG_p_r, RX_cl_r, res_r, sV_T))[deriv, deriv]
    )
    V_T_rb = fact**2 * float(
        (_v(invG_p_l, Q_q_l, res_l, sV_T) + _v(invG_p_r, Q_q_r, res_r, sV_T))[deriv, deriv]
    )
    return RdFitResult(
        tau_cl=tau_cl,
        tau_bc=tau_bc,
        se_cl=float(np.sqrt(V_tau_cl)),
        se_rb=float(np.sqrt(V_tau_rb)),
        bias_l=bias_l,
        bias_r=bias_r,
        beta_p_l=beta_p_l[:, 0],
        beta_p_r=beta_p_r[:, 0],
        N_h_l=N_h_l,
        N_h_r=N_h_r,
        N_b_l=N_b_l,
        N_b_r=N_b_r,
        tau_T_cl=tau_T_cl,
        tau_T_bc=tau_T_bc,
        se_T_cl=float(np.sqrt(V_T_cl)),
        se_T_rb=float(np.sqrt(V_T_rb)),
        beta_t_p_l=beta_p_l[:, 1],
        beta_t_p_r=beta_p_r[:, 1],
    )
