"""In-house port of nprobust's MSE-DPI bandwidth selector.

Faithful Python translation of the mse-dpi branch of
``nprobust::lpbwselect`` from the R package
``nprobust`` 0.5.0 (SHA ``36e4e532d2f7d23d4dc6e162575cca79e0927cda``,
``github.com/nppackages/nprobust``). This module implements Calonico,
Cattaneo, and Farrell (2018, JASA 113(522)) plug-in bandwidth selection
exactly as the reference R code does, so that per-stage bandwidths and
per-stage bias/variance components parity-check to 1% on deterministic
seeds (see ``benchmarks/R/generate_nprobust_golden.R`` and
``tests/test_nprobust_port.py``).

Source mapping (every public function here pairs with one R function):

========================================  =========================================
Python                                    R (npfunctions.R)
========================================  =========================================
``kernel_W(u, kernel)``                   ``W.fun`` (npfunctions.R:1-7)
``qrXXinv(x)``                            ``qrXXinv`` (npfunctions.R:89-93)
``_precompute_nn_duplicates(x)``          npfunctions.R:518-529 (inline in mse.dpi)
``lprobust_res(...)``                     ``lprobust.res`` (npfunctions.R:131-162)
``lprobust_vce(...)``                     ``lprobust.vce`` (npfunctions.R:165-185)
``LprobustBwResult``                      return list of ``lprobust.bw``
``lprobust_bw(...)``                      ``lprobust.bw`` (npfunctions.R:187-288)
``lpbwselect_mse_dpi(...)``               ``lpbwselect.mse.dpi`` (npfunctions.R:498-607)
========================================  =========================================

This module is nprobust-internal logic; the public wrapper is
``diff_diff.local_linear.mse_optimal_bandwidth``. Phase 1c's robust
variance and bias estimator will also compose these helpers.

Deviations from nprobust (documented):

* ``weights=`` in ``lprobust`` (Phase 4.5 survey support): supported.
  User weights multiply into the kernel weights pointwise
  (``W_combined = k((x-c)/h) · w``) and propagate through design
  matrices, Q.q, and variance matrices. When ``weights=np.ones(N)`` the
  function is bit-identical to the unweighted path (regression-tested
  at atol=1e-14). ``return_influence=True`` surfaces the per-obs IF of
  the BIAS-CORRECTED point estimate (aligned with V_Y_bc) for survey-
  composed variance at the estimator level. The bandwidth selector
  ``lpbwselect_mse_dpi`` and its public wrapper ``mse_optimal_bandwidth``
  remain unweighted in this release (no DPI-selector weight derivation
  shipped); pass user-specified ``h``/``b`` for weight-aware bandwidths.
* ``vce="nn"`` is the default and is fully ported. ``vce in
  {"hc0", "hc1", "hc2", "hc3"}`` is implemented in ``lprobust_res`` /
  ``lprobust_vce`` but has not been separately golden-tested; use at
  your own risk until Phase 1c.
* ``cluster=`` is supported in ``lprobust_vce`` and the ``lprobust_bw``
  wrapper but is only exercised by the HAD estimator via Phase 2.
* **Missing-data policy for cluster IDs:** nprobust's ``lpbwselect``
  complete-case-filters ``(x, y, cluster)`` before dispatch, dropping
  rows where any of the three is missing. This port deliberately
  rejects missing cluster IDs with a targeted ``ValueError`` instead
  so callers see the missingness rather than silently losing rows.
  (``x`` and ``y`` finiteness is also rejected up front for the same
  reason; they could not be silently dropped in the nprobust way
  without ambiguity.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize
from scipy.stats import norm as _scipy_norm

__all__ = [
    "NPROBUST_VERSION",
    "NPROBUST_SHA",
    "LprobustBwResult",
    "LprobustResult",
    "kernel_W",
    "qrXXinv",
    "lprobust_res",
    "lprobust_vce",
    "lprobust_bw",
    "lprobust",
    "lpbwselect_mse_dpi",
]

NPROBUST_VERSION = "0.5.0"
NPROBUST_SHA = "36e4e532d2f7d23d4dc6e162575cca79e0927cda"

_VALID_KERNELS = ("epa", "uni", "tri", "gau")
_VALID_VCE = ("nn", "hc0", "hc1", "hc2", "hc3")


def _cluster_has_missing(cluster: np.ndarray) -> bool:
    """Detect missing cluster IDs across float / object / string dtypes.

    nprobust::lpbwselect complete-case-filters (x, y, cluster) before
    dispatch. This port deliberately rejects missingness instead so
    callers see it rather than silently losing rows. Used by
    ``lpbwselect_mse_dpi`` and ``lprobust`` (and the public
    ``bias_corrected_local_linear`` wrapper) so all three surfaces
    honor the same contract.
    """
    if cluster.dtype.kind in ("f", "c"):
        return bool(np.any(~np.isfinite(cluster)))
    # Object / string / None-containing arrays: treat None and NaN-like
    # sentinels as missing.
    try:
        if bool(np.any([v is None for v in cluster])):
            return True
    except TypeError:
        pass
    try:
        # np.nan comparisons are False; cast to float and check finiteness.
        cluster_f = cluster.astype(np.float64, copy=False)
        return bool(np.any(~np.isfinite(cluster_f)))
    except (TypeError, ValueError):
        return False


# =============================================================================
# Kernel (W.fun, npfunctions.R:1-7)
# =============================================================================


def kernel_W(u: np.ndarray, kernel: str) -> np.ndarray:
    """Symmetric kernel evaluation matching ``nprobust::W.fun``.

    Parameters
    ----------
    u : np.ndarray
        Scaled argument ``(x - c) / h``.
    kernel : str
        One of "epa", "uni", "tri", "gau".

    Returns
    -------
    np.ndarray
        Kernel values, same shape as ``u``. Zero where ``|u| > 1`` for
        compact-support kernels.
    """
    u = np.asarray(u, dtype=np.float64)
    if kernel == "epa":
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)
    if kernel == "uni":
        return np.where(np.abs(u) <= 1.0, 0.5, 0.0)
    if kernel == "tri":
        return np.where(np.abs(u) <= 1.0, 1.0 - np.abs(u), 0.0)
    if kernel == "gau":
        return _scipy_norm.pdf(u)
    raise ValueError(f"Unknown kernel {kernel!r}. Expected one of {_VALID_KERNELS}.")


# =============================================================================
# qrXXinv (npfunctions.R:89-93)
# =============================================================================


def qrXXinv(x: np.ndarray) -> np.ndarray:
    """Cholesky-based inverse of ``x.T @ x``.

    Mirrors ``chol2inv(chol(crossprod(x)))`` in R. ``x`` typically
    represents a design matrix already scaled by ``sqrt(weights)``.

    Parameters
    ----------
    x : np.ndarray, shape (n, k)

    Returns
    -------
    np.ndarray, shape (k, k)
        Inverse of ``x.T @ x``.

    Raises
    ------
    ValueError
        If ``x.T @ x`` is rank-deficient (Cholesky fails). Converts
        the raw ``np.linalg.LinAlgError`` into a targeted message so
        callers (``lprobust_bw``) can surface a clear failure reason
        instead of an opaque linear-algebra error.
    """
    xtx = x.T @ x
    k = xtx.shape[0]
    # Cholesky solve for the inverse. Matches R's chol2inv(chol(.)).
    try:
        L = np.linalg.cholesky(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"qrXXinv: Cholesky decomposition of X'X ({k}x{k}) failed. "
            f"The weighted design matrix is rank-deficient, likely "
            f"because the in-window support has fewer than {k} distinct "
            f"points. Increase sample size, widen the bandwidth, or pick "
            f"a boundary with more distinct values nearby. "
            f"(LinAlgError: {exc})"
        ) from exc
    Linv = np.linalg.solve(L, np.eye(k))
    return Linv.T @ Linv


# =============================================================================
# Nearest-neighbor duplicate precomputation (inlined in mse.dpi, R:518-529)
# =============================================================================


def _precompute_nn_duplicates(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``dups`` and ``dupsid`` arrays for NN residuals.

    ``x`` must already be sorted ascending.

    Mirrors npfunctions.R:518-529:

        for (j in 1:N) dups[j] = sum(x == x[j])
        j = 1
        while (j <= N) {
          dupsid[j:(j + dups[j] - 1)] = 1:dups[j]
          j = j + dups[j]
        }
    """
    n = x.shape[0]
    dups = np.empty(n, dtype=np.int64)
    for j in range(n):
        dups[j] = int(np.sum(x == x[j]))
    dupsid = np.empty(n, dtype=np.int64)
    j = 0
    while j < n:
        k = int(dups[j])
        # 1-indexed in R: dupsid[j:(j+dups[j]-1)] = 1:dups[j]
        dupsid[j : j + k] = np.arange(1, k + 1)
        j += k
    return dups, dupsid


# =============================================================================
# lprobust.res (npfunctions.R:131-162)
# =============================================================================


def lprobust_res(
    X: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    hii: Optional[np.ndarray],
    vce: str,
    matches: int,
    dups: Optional[np.ndarray],
    dupsid: Optional[np.ndarray],
    d: int,
) -> np.ndarray:
    """Port of ``lprobust.res``.

    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Regressor (one-column; the R code treats ``X`` as a scalar
        series for NN distance comparisons).
    y : np.ndarray, shape (n,)
        Outcome.
    m : np.ndarray, shape (n, 1) or (n,)
        Fitted values from the local-polynomial regression. Unused
        when ``vce="nn"``.
    hii : np.ndarray or None
        Hat-matrix diagonal, used by "hc2" / "hc3". Ignored for "nn",
        "hc0", "hc1".
    vce : str
        One of "nn", "hc0", "hc1", "hc2", "hc3".
    matches : int
        ``nnmatch``, target number of nearest neighbors per observation.
    dups, dupsid : np.ndarray or None
        Precomputed (see ``_precompute_nn_duplicates``). Required when
        ``vce="nn"``.
    d : int
        Polynomial-order-plus-one (``o + 1`` in lprobust.bw). Used for
        the HC1 degrees-of-freedom correction.

    Returns
    -------
    np.ndarray, shape (n, 1)
        Column vector of residuals.
    """
    if vce not in _VALID_VCE:
        raise ValueError(f"Unknown vce {vce!r}. Expected one of {_VALID_VCE}.")

    n = y.shape[0]
    res = np.empty((n, 1), dtype=np.float64)

    if vce == "nn":
        if dups is None or dupsid is None:
            raise ValueError("vce='nn' requires precomputed dups / dupsid")
        # Port of npfunctions.R:134-153. R uses 1-based indexing; Python is
        # 0-based, so every `pos`, `pos-lpos-1`, and `pos+rpos+1` translates
        # to subtracting one from the R expression to land in [0, n).
        for pos in range(n):
            rpos = int(dups[pos] - dupsid[pos])
            lpos = int(dupsid[pos] - 1)
            lim = min(matches, n - 1)
            while lpos + rpos < lim:
                # Guard conditions mirror R exactly; "pos-lpos-1" in R means
                # 1-indexed position, so "pos-lpos-1 <= 0" in R corresponds to
                # Python "(pos) - lpos - 1 < 0" where pos is 0-indexed.
                left_idx = pos - lpos - 1  # R: pos-lpos-1 (1-indexed)
                right_idx = pos + rpos + 1  # R: pos+rpos+1 (1-indexed)
                # In Python 0-indexed, "pos-lpos-1 <= 0" means no more room left.
                # Equivalent check: left_idx < 0 means index 0 already past.
                if left_idx < 0:
                    rpos = rpos + int(dups[right_idx])
                elif right_idx > n - 1:
                    lpos = lpos + int(dups[left_idx])
                elif (X[pos] - X[left_idx]) > (X[right_idx] - X[pos]):
                    rpos = rpos + int(dups[right_idx])
                elif (X[pos] - X[left_idx]) < (X[right_idx] - X[pos]):
                    lpos = lpos + int(dups[left_idx])
                else:
                    rpos = rpos + int(dups[right_idx])
                    lpos = lpos + int(dups[left_idx])
            # Indices of the neighbor group (inclusive bounds in R).
            ind_J_start = pos - lpos
            ind_J_end = min(n - 1, pos + rpos)  # inclusive
            y_J = float(np.sum(y[ind_J_start : ind_J_end + 1]) - y[pos])
            J_i = (ind_J_end - ind_J_start + 1) - 1
            res[pos, 0] = np.sqrt(J_i / (J_i + 1.0)) * (y[pos] - y_J / J_i)
        return res

    # HC variants (vce != "nn")
    m1 = m.reshape(-1) if m.ndim == 2 else m
    if vce == "hc0":
        w = np.ones(n, dtype=np.float64)
    elif vce == "hc1":
        w = np.full(n, np.sqrt(n / (n - d)), dtype=np.float64)
    elif vce == "hc2":
        if hii is None:
            raise ValueError("vce='hc2' requires hii")
        w = np.sqrt(1.0 / (1.0 - hii.reshape(-1)))
    else:  # hc3
        if hii is None:
            raise ValueError("vce='hc3' requires hii")
        w = 1.0 / (1.0 - hii.reshape(-1))
    res[:, 0] = w * (y - m1)
    return res


# =============================================================================
# lprobust.vce (npfunctions.R:165-185)
# =============================================================================


def lprobust_vce(
    RX: np.ndarray,
    res: np.ndarray,
    cluster: Optional[np.ndarray],
) -> np.ndarray:
    """Port of ``lprobust.vce``. Meat of the sandwich.

    Parameters
    ----------
    RX : np.ndarray, shape (n, k)
        Weighted design matrix ``R * eW`` from the caller.
    res : np.ndarray, shape (n, 1) or (n,)
        Residuals from ``lprobust_res``.
    cluster : np.ndarray or None
        Cluster identifier, same length as ``res``. ``None`` for
        unclustered.

    Returns
    -------
    np.ndarray, shape (k, k)
        Meat matrix.
    """
    k = RX.shape[1]
    r = res.reshape(-1)

    if cluster is None:
        rRX = (r[:, None]) * RX
        return rRX.T @ rRX

    clusters = np.unique(cluster)
    M = np.zeros((k, k), dtype=np.float64)
    for c in clusters:
        ind = cluster == c
        Xi = RX[ind, :]
        ri = r[ind]
        # R: M = M + crossprod(t(crossprod(Xi,ri)),t(crossprod(Xi,ri)))
        # crossprod(Xi, ri) is a (k,) vector = Xi.T @ ri
        v = Xi.T @ ri
        M = M + np.outer(v, v)
    # nprobust's lprobust.vce computes w = ((n-1)/(n-k))*(g/(g-1))
    # but does NOT apply it to the returned M (npfunctions.R:183;
    # w is dead code in the R source). Match the R return exactly.
    return M


# =============================================================================
# lprobust.bw (npfunctions.R:187-288)
# =============================================================================


@dataclass
class LprobustBwResult:
    """Return value of ``lprobust_bw``. Mirrors the R list.

    Attributes
    ----------
    V, B1, B2, R, r, rB, rV, bw : float
        See npfunctions.R:276-287 for the exact formulas.
    """

    V: float
    B1: float
    B2: float
    R: float
    r: float
    rB: float
    rV: float
    bw: float


def lprobust_bw(
    Y: np.ndarray,
    X: np.ndarray,
    cluster: Optional[np.ndarray],
    c: float,
    o: int,
    nu: int,
    o_B: int,
    h_V: float,
    h_B1: float,
    h_B2: float,
    scale: float,
    vce: str,
    nnmatch: int,
    kernel: str,
    dups: Optional[np.ndarray],
    dupsid: Optional[np.ndarray],
) -> LprobustBwResult:
    """Port of ``lprobust.bw`` (npfunctions.R:187-288).

    The heart of the 3-stage DPI: one call produces one stage's
    ``(V, B1, B2, R, bw)``. Called four times from ``lpbwselect_mse_dpi``.

    Parameters match the R signature argument-for-argument.

    Raises
    ------
    ValueError
        If any of the three local-polynomial fits has fewer in-window
        observations than its required column count. Catches opaque
        ``LinAlgError`` failures from downstream Cholesky inversion in
        tiny-sample or mispositioned-boundary settings and surfaces a
        targeted error naming the failing stage.
    """
    N = X.shape[0]
    eC: Optional[np.ndarray] = None

    # === Variance: local-poly fit of order o at bandwidth h_V ===
    u_V = (X - c) / h_V
    w = kernel_W(u_V, kernel) / h_V
    ind_V = w > 0.0
    eY = Y[ind_V]
    eX = X[ind_V]
    eW = w[ind_V]
    n_V = int(np.sum(ind_V))
    if n_V < o + 1:
        raise ValueError(
            f"lprobust_bw: variance stage has n_V={n_V} in-window "
            f"observations at h_V={h_V:.6g} (boundary={c}, kernel={kernel!r}), "
            f"but needs at least o+1={o + 1}. Increase sample size, choose "
            f"a valid lower boundary with sufficient data to the right, "
            f"or disable bwcheck=None-driven narrow windows."
        )
    # Design matrix R.V in R; rename to R_V to avoid Python builtin conflict.
    R_V = np.empty((n_V, o + 1), dtype=np.float64)
    for j in range(o + 1):
        R_V[:, j] = (eX - c) ** j
    sqrtW = np.sqrt(eW)
    invG_V = qrXXinv(R_V * sqrtW[:, None])
    beta_V = invG_V @ (R_V.T @ (eW * eY))

    if cluster is not None:
        eC = cluster[ind_V]

    dups_V = dupsid_V = None
    if vce == "nn":
        if dups is None or dupsid is None:
            raise ValueError("vce='nn' requires precomputed dups/dupsid")
        dups_V = dups[ind_V]
        dupsid_V = dupsid[ind_V]

    predicts_V = np.zeros(n_V, dtype=np.float64)
    hii: Optional[np.ndarray] = None
    if vce in ("hc0", "hc1", "hc2", "hc3"):
        predicts_V = R_V @ beta_V
        if vce in ("hc2", "hc3"):
            hii = np.empty(n_V, dtype=np.float64)
            RW = R_V * eW[:, None]
            for i in range(n_V):
                hii[i] = R_V[i, :] @ invG_V @ RW[i, :]

    res_V = lprobust_res(
        eX,
        eY,
        predicts_V.reshape(-1, 1),
        hii,
        vce,
        nnmatch,
        dups_V,
        dupsid_V,
        o + 1,
    )
    meat_V = lprobust_vce(R_V * eW[:, None], res_V, eC)
    V_V = float((invG_V @ meat_V @ invG_V)[nu, nu])

    # === Bias coefficient BConst1 / BConst2 ===
    # Hp (diag scaling in R). Hp[j] = h.V^((j-1)) for j=1..o+1, so at Python
    # index i in [0, o], Hp[i] = h_V ** i.
    Hp = np.array([h_V**j for j in range(o + 1)], dtype=np.float64)
    v1 = (R_V * eW[:, None]).T @ ((eX - c) / h_V) ** (o + 1)
    v2 = (R_V * eW[:, None]).T @ ((eX - c) / h_V) ** (o + 2)
    # (Hp * (invG.V %*% v1))[nu+1] in R == Python index nu.
    BConst1 = float((Hp * (invG_V @ v1))[nu])
    BConst2 = float((Hp * (invG_V @ v2))[nu])

    # === B1 via a separate fit at h.B1, order o.B ===
    u_B1 = (X - c) / h_B1
    w1 = kernel_W(u_B1, kernel)
    ind1 = w1 > 0.0
    eY1 = Y[ind1]
    eX1 = X[ind1]
    eW1 = w1[ind1]
    n_B1 = int(np.sum(ind1))
    if n_B1 < o_B + 1:
        raise ValueError(
            f"lprobust_bw: B1 stage has n_B1={n_B1} in-window observations "
            f"at h_B1={h_B1:.6g} (boundary={c}, kernel={kernel!r}), but "
            f"needs at least o_B+1={o_B + 1}. Increase sample size or "
            f"widen the pilot bandwidth."
        )
    R_B1 = np.empty((n_B1, o_B + 1), dtype=np.float64)
    for j in range(o_B + 1):
        R_B1[:, j] = (eX1 - c) ** j
    sqrtW1 = np.sqrt(eW1)
    invG_B1 = qrXXinv(R_B1 * sqrtW1[:, None])
    beta_B1 = invG_B1 @ (R_B1.T @ (eW1 * eY1))

    # === BWreg (only when scale > 0) ===
    BWreg = 0.0
    if scale > 0:
        eC1: Optional[np.ndarray] = None
        if cluster is not None:
            eC1 = cluster[ind1]
        dups_B = dupsid_B = None
        hii_B = None
        predicts_B = np.zeros(n_B1, dtype=np.float64)
        if vce == "nn":
            dups_B = dups[ind1] if dups is not None else None
            dupsid_B = dupsid[ind1] if dupsid is not None else None
        if vce in ("hc0", "hc1", "hc2", "hc3"):
            # Suppress spurious BLAS FPE warnings (numpy issue #21432
            # pattern); matmul on some platforms (Accelerate / OpenBLAS)
            # sets divide/overflow flags on SIMD intermediates even when
            # input and output are finite.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
                predicts_B = R_B1 @ beta_B1
                if vce in ("hc2", "hc3"):
                    hii_B = np.empty(n_B1, dtype=np.float64)
                    RW1 = R_B1 * eW1[:, None]
                    for i in range(n_B1):
                        hii_B[i] = R_B1[i, :] @ invG_B1 @ RW1[i, :]
        res_B = lprobust_res(
            eX1,
            eY1,
            predicts_B.reshape(-1, 1),
            hii_B,
            vce,
            nnmatch,
            dups_B,
            dupsid_B,
            o_B + 1,
        )
        V_B = float(
            (invG_B1 @ lprobust_vce(R_B1 * eW1[:, None], res_B, eC1) @ invG_B1)[o + 1, o + 1]
        )
        BWreg = 3.0 * BConst1 * BConst1 * V_B

    # === B2 via a separate fit at h.B2, order o.B+1 ===
    u_B2 = (X - c) / h_B2
    w2 = kernel_W(u_B2, kernel)
    ind2 = w2 > 0.0
    eY2 = Y[ind2]
    eX2 = X[ind2]
    eW2 = w2[ind2]
    n_B2 = int(np.sum(ind2))
    if n_B2 < o_B + 2:
        raise ValueError(
            f"lprobust_bw: B2 stage has n_B2={n_B2} in-window observations "
            f"at h_B2={h_B2:.6g} (boundary={c}, kernel={kernel!r}), but "
            f"needs at least o_B+2={o_B + 2}. Increase sample size or "
            f"widen the pilot bandwidth."
        )
    R_B2 = np.empty((n_B2, o_B + 2), dtype=np.float64)
    for j in range(o_B + 2):
        R_B2[:, j] = (eX2 - c) ** j
    sqrtW2 = np.sqrt(eW2)
    invG_B2 = qrXXinv(R_B2 * sqrtW2[:, None])
    beta_B2 = invG_B2 @ (R_B2.T @ (eW2 * eY2))

    # === Compose final scalars (npfunctions.R:276-287) ===
    # R: B1 = BConst1 * beta.B1[o+2]  (1-indexed); Python: beta_B1[o+1]
    B1_val = float(BConst1 * beta_B1[o + 1])
    # R: B2 = BConst2 * beta.B2[o+3]  (1-indexed); Python: beta_B2[o + 2]
    B2_val = float(BConst2 * beta_B2[o + 2])
    V = float(N * h_V ** (2 * nu + 1) * V_V)
    R_reg = float(BWreg)
    r_val = 1.0 / (2.0 * o + 3.0)
    rB = float(2 * (o + 1 - nu))
    rV = float(2 * nu + 1)
    bw = (rV * V / (N * rB * (B1_val**2 + scale * R_reg))) ** r_val

    return LprobustBwResult(
        V=V,
        B1=B1_val,
        B2=B2_val,
        R=R_reg,
        r=r_val,
        rB=rB,
        rV=rV,
        bw=float(bw),
    )


# =============================================================================
# lpbwselect.mse.dpi (npfunctions.R:498-607)
# =============================================================================


@dataclass
class MseDpiStages:
    """Return value of ``lpbwselect_mse_dpi``. Mirrors the R list plus
    exposes per-stage diagnostics."""

    h_mse_dpi: float
    b_mse_dpi: float
    c_bw: float
    bw_mp2: float
    bw_mp3: float
    bw_max: float
    bw_min: Optional[float]
    stage_d1: LprobustBwResult
    stage_d2: LprobustBwResult
    stage_b: LprobustBwResult
    stage_h: LprobustBwResult


def lpbwselect_mse_dpi(
    y: np.ndarray,
    x: np.ndarray,
    cluster: Optional[np.ndarray] = None,
    eval_point: float = 0.0,
    p: int = 1,
    q: Optional[int] = None,
    deriv: int = 0,
    kernel: str = "epa",
    bwcheck: Optional[int] = 21,
    bwregul: float = 1.0,
    vce: str = "nn",
    nnmatch: int = 3,
    interior: bool = False,
) -> MseDpiStages:
    """Port of ``lpbwselect.mse.dpi`` (npfunctions.R:498-607).

    The R source computes ``even = (p - deriv) %% 2 == 0`` and dispatches
    each stage via ``if (even == FALSE | interior == TRUE) bw <- C$bw
    else bw <- optimize(...)$minimum``. For the HAD use case
    (``p=1, deriv=0``), ``(p - deriv) %% 2 == 1``, so ``even == FALSE`` and
    every stage bandwidth comes from the closed-form ``C$bw`` expression
    inside ``lprobust.bw``; the ``optimize()`` branch is taken only when
    ``(p - deriv)`` is even AND ``interior == FALSE``.

    Parameters match the R signature. See the R source comments for
    semantics.

    Raises
    ------
    ValueError
        If ``bwcheck`` is supplied and falls outside the valid range
        ``[1, len(x)]``.
    """
    # Front-door input contract (shape / emptiness / finiteness).
    # Must run BEFORE the bwcheck range check so empty-array or
    # non-finite inputs get targeted messages instead of "bwcheck
    # exceeds sample size".
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same 1-D shape; got " f"{x.shape} and {y.shape}")
    if x.size == 0:
        raise ValueError(
            "x and y must be non-empty; lpbwselect_mse_dpi cannot "
            "estimate a bandwidth from zero observations."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf)")
    if not np.isfinite(eval_point):
        raise ValueError(f"eval_point must be finite; got {eval_point}")
    if cluster is not None:
        cluster = np.asarray(cluster).ravel()
        if cluster.shape != x.shape:
            raise ValueError(
                f"cluster must have the same shape as x; got " f"{cluster.shape} and {x.shape}"
            )
        # Missing cluster IDs must be rejected, not silently dropped.
        # nprobust::lpbwselect complete-case-filters (x, y, cluster)
        # before dispatch; this port deliberately rejects instead so
        # callers see the missingness rather than lose rows silently.
        # The "reject" vs "filter" choice is documented in the module
        # docstring deviations list. Dtype-agnostic via
        # `_cluster_has_missing`.
        if _cluster_has_missing(cluster):
            raise ValueError(
                "cluster contains missing values (NaN / None). Unlike "
                "nprobust::lpbwselect which complete-case-filters "
                "(x, y, cluster), this port rejects missing cluster "
                "IDs so the caller sees the missingness rather than "
                "silently losing rows. Filter your data before the "
                "call or drop missing observations explicitly."
            )

    N = x.shape[0]
    if bwcheck is not None:
        if bwcheck < 1:
            raise ValueError(f"bwcheck must be a positive integer (>= 1); got {bwcheck}")
        if bwcheck > N:
            raise ValueError(
                f"bwcheck={bwcheck} exceeds sample size N={N}. Either "
                f"reduce bwcheck or increase sample size; pass "
                f"bwcheck=None to skip the nearest-neighbor floor."
            )
    if kernel not in _VALID_KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}. Expected one of {_VALID_KERNELS}.")
    if vce not in _VALID_VCE:
        raise ValueError(f"Unknown vce {vce!r}. Expected one of {_VALID_VCE}.")

    if q is None:
        q = p + 1

    even = (p - deriv) % 2 == 0
    x_min = float(x.min())
    x_max = float(x.max())
    range_ = x_max - x_min
    x_iq = float(np.percentile(x, 75) - np.percentile(x, 25))

    C_c_table = {"epa": 2.34, "uni": 1.843, "tri": 2.576, "gau": 1.06}
    C_c = C_c_table[kernel]
    sd_x = float(np.std(x, ddof=1))
    c_bw = C_c * min(sd_x, x_iq / 1.349) * N ** (-1.0 / 5.0)
    bw_max = max(abs(eval_point - x_min), abs(eval_point - x_max))
    c_bw = min(c_bw, bw_max)

    # Sort and precompute NN structure (npfunctions.R:518-529).
    dups: Optional[np.ndarray] = None
    dupsid: Optional[np.ndarray] = None
    if vce == "nn":
        order_x = np.argsort(x)
        x = x[order_x]
        y = y[order_x]
        if cluster is not None:
            cluster = cluster[order_x]
        dups, dupsid = _precompute_nn_duplicates(x)

    bw_min: Optional[float] = None
    if bwcheck is not None:
        sorted_abs = np.sort(np.abs(x - eval_point))
        # R: bw.min <- sort(abs(x-eval))[bwcheck]. R is 1-indexed, so bwcheck-1 in Python.
        bw_min = float(sorted_abs[bwcheck - 1])
        c_bw = max(c_bw, bw_min)

    def _optimize_amse(
        C: LprobustBwResult,
        exp_bias: float,
        exp_var: float,
        scale: float,
    ) -> float:
        """Minimize ``|H^exp_bias * (B1 + H*B2 + scale*R)^2 + V / (N * H^exp_var)|``
        over ``H`` in ``(eps, range)``. Matches ``optimize()`` in R.

        Only called on the even-and-boundary branch of the original R
        conditional; for the HAD case (p=1, deriv=0) we take the
        closed-form ``C.bw`` branch instead.
        """

        def _fun(H: float) -> float:
            return abs(H**exp_bias * (C.B1 + H * C.B2 + scale * C.R) ** 2 + C.V / (N * H**exp_var))

        res = optimize.minimize_scalar(
            _fun,
            bounds=(np.finfo(float).eps, range_),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(res.x)

    # Stage 2: C.d1 -> bw.mp2
    C_d1 = lprobust_bw(
        y,
        x,
        cluster,
        eval_point,
        o=q + 1,
        nu=q + 1,
        o_B=q + 2,
        h_V=c_bw,
        h_B1=range_,
        h_B2=range_,
        scale=0.0,
        vce=vce,
        nnmatch=nnmatch,
        kernel=kernel,
        dups=dups,
        dupsid=dupsid,
    )
    if (not even) or interior:
        bw_mp2 = C_d1.bw
    else:
        bw_mp2 = _optimize_amse(
            C_d1,
            exp_bias=2 * (q + 1) + 2 - 2 * (q + 1),  # = 2
            exp_var=1 + 2 * (q + 1),
            scale=0.0,
        )

    # Stage 2: C.d2 -> bw.mp3
    C_d2 = lprobust_bw(
        y,
        x,
        cluster,
        eval_point,
        o=q + 2,
        nu=q + 2,
        o_B=q + 3,
        h_V=c_bw,
        h_B1=range_,
        h_B2=range_,
        scale=0.0,
        vce=vce,
        nnmatch=nnmatch,
        kernel=kernel,
        dups=dups,
        dupsid=dupsid,
    )
    if (not even) or interior:
        bw_mp3 = C_d2.bw
    else:
        bw_mp3 = _optimize_amse(
            C_d2,
            exp_bias=2 * (q + 2) + 2 - 2 * (q + 2),  # = 2
            exp_var=1 + 2 * (q + 2),
            scale=0.0,
        )

    # Clipping (npfunctions.R:559-565)
    bw_mp2 = min(bw_mp2, bw_max)
    bw_mp3 = min(bw_mp3, bw_max)
    if bw_min is not None:
        bw_mp2 = max(bw_mp2, bw_min)
        bw_mp3 = max(bw_mp3, bw_min)

    # Stage 3: C.b -> b.mse.dpi
    C_b = lprobust_bw(
        y,
        x,
        cluster,
        eval_point,
        o=q,
        nu=p + 1,
        o_B=q + 1,
        h_V=c_bw,
        h_B1=bw_mp2,
        h_B2=bw_mp3,
        scale=bwregul,
        vce=vce,
        nnmatch=nnmatch,
        kernel=kernel,
        dups=dups,
        dupsid=dupsid,
    )
    if (not even) or interior:
        b_mse_dpi = C_b.bw
    else:
        b_mse_dpi = _optimize_amse(
            C_b,
            exp_bias=2 * q + 2 - 2 * (p + 1),
            exp_var=1 + 2 * (p + 1),
            scale=bwregul,
        )
    b_mse_dpi = min(b_mse_dpi, bw_max)
    if bw_min is not None:
        b_mse_dpi = max(b_mse_dpi, bw_min)

    # Stage 3 final: C.h -> h.mse.dpi
    C_h = lprobust_bw(
        y,
        x,
        cluster,
        eval_point,
        o=p,
        nu=deriv,
        o_B=q,
        h_V=c_bw,
        h_B1=b_mse_dpi,
        h_B2=bw_mp2,
        scale=bwregul,
        vce=vce,
        nnmatch=nnmatch,
        kernel=kernel,
        dups=dups,
        dupsid=dupsid,
    )
    if (not even) or interior:
        h_mse_dpi = C_h.bw
    else:
        h_mse_dpi = _optimize_amse(
            C_h,
            exp_bias=2 * p + 2 - 2 * deriv,
            exp_var=1 + 2 * deriv,
            scale=bwregul,
        )
    h_mse_dpi = min(h_mse_dpi, bw_max)
    if bw_min is not None:
        h_mse_dpi = max(h_mse_dpi, bw_min)

    return MseDpiStages(
        h_mse_dpi=float(h_mse_dpi),
        b_mse_dpi=float(b_mse_dpi),
        c_bw=float(c_bw),
        bw_mp2=float(bw_mp2),
        bw_mp3=float(bw_mp3),
        bw_max=float(bw_max),
        bw_min=bw_min,
        stage_d1=C_d1,
        stage_d2=C_d2,
        stage_b=C_b,
        stage_h=C_h,
    )


# =============================================================================
# lprobust single-eval-point path (lprobust.R:177-248) — Phase 1c
# =============================================================================
#
# Port of the body of nprobust::lprobust's per-eval-point loop iteration from
# lprobust.R (version pinned by NPROBUST_VERSION / NPROBUST_SHA above). The
# single-eval path produces the classical (no bias correction) and Calonico-
# Cattaneo-Titiunik (2014) bias-corrected point estimates plus their naive
# and robust standard errors. The multi-eval grid and the covgrid=TRUE
# cross-covariance branch (lprobust.R:253-378) are intentionally out of scope
# for Phase 1c.
#
# Active-window rule (lprobust.R:181-182, single-eval only): ``ind = ind.b``
# by default, overwritten to ``ind.h`` only when ``h > b``. This is a
# CONDITIONAL REPLACEMENT, not a union. The union ``ind.h | ind.b`` is only
# used in the covgrid branch.


@dataclass
class LprobustResult:
    """Single-eval-point result of ``lprobust`` (CCT 2014 bias correction).

    Mirrors the per-eval-point row of nprobust's ``Estimate`` matrix
    (lprobust.R:163-164, 248) plus the full intermediate ``(p+1)x(p+1)``
    variance matrices (kept so Phase 2 diagnostics can inspect them).

    Attributes
    ----------
    eval_point : float
        Evaluation point (``c`` in nprobust's notation; the boundary in HAD).
    h, b : float
        Main and bias-correction bandwidths as actually used after the
        ``bwcheck`` clip (never below the ``bwcheck``-nearest-neighbor floor).
    n_used : int
        Observations in the SELECTED single kernel window (line 181-182 of
        lprobust.R): ``sum(ind.b)`` when ``h <= b`` and ``sum(ind.h)`` when
        ``h > b``. With ``rho=1`` default (``h == b``), the two windows
        coincide.
    tau_cl : float
        Classical point estimate ``factorial(deriv) * beta.p[deriv+1]``
        (lprobust.R:226).
    tau_bc : float
        Bias-corrected point estimate ``factorial(deriv) *
        beta.bc[deriv+1]`` (lprobust.R:227), equal to ``mu_hat + M_hat`` in
        the Equation 8 notation of de Chaisemartin et al. (2026).
    se_cl : float
        Naive plug-in standard error ``sqrt(factorial(deriv)^2 *
        V.Y.cl[deriv+1, deriv+1])`` (lprobust.R:245).
    se_rb : float
        Robust standard error under the CCT (2014) bias-corrected
        asymptotics ``sqrt(factorial(deriv)^2 * V.Y.bc[deriv+1, deriv+1])``
        (lprobust.R:246).
    V_Y_cl, V_Y_bc : np.ndarray, shape (p+1, p+1)
        Full classical and robust variance matrices. CI bounds are not
        produced here; the public wrapper ``bias_corrected_local_linear``
        adds ``ci_low``/``ci_high`` from ``tau.bc +/- z_{1-alpha/2} * se.rb``.
    """

    eval_point: float
    h: float
    b: float
    n_used: int
    tau_cl: float
    tau_bc: float
    se_cl: float
    se_rb: float
    V_Y_cl: np.ndarray
    V_Y_bc: np.ndarray
    influence_function: Optional[np.ndarray] = None
    """Per-observation influence function of the BIAS-CORRECTED point
    estimate at ``deriv`` (``tau_bc`` for the deriv=0 case), aligned
    with the ORIGINAL x ordering. Shape ``(N,)``. Set only when
    ``return_influence=True``; ``None`` otherwise. Observations outside
    the active kernel window have IF=0.

    Derived from ``Q_q`` + ``res_b``, so the variance self-check is
    ``sum(IF^2) == V_Y_bc[deriv, deriv]`` (up to BLAS ordering) under
    unclustered HC0. Used by estimator-level Binder (1983) TSL
    composition for survey-design variance on HAD's continuous path — the
    bias-corrected scale matches the ATT which itself uses ``tau_bc``.
    Using the classical IF here would silently under-estimate survey SE
    by ignoring the bias-correction variance inflation."""


def lprobust(
    y: np.ndarray,
    x: np.ndarray,
    eval_point: float,
    h: float,
    b: float,
    p: int = 1,
    q: Optional[int] = None,
    deriv: int = 0,
    kernel: str = "epa",
    vce: str = "nn",
    cluster: Optional[np.ndarray] = None,
    nnmatch: int = 3,
    bwcheck: Optional[int] = 21,
    weights: Optional[np.ndarray] = None,
    return_influence: bool = False,
) -> LprobustResult:
    """Local-polynomial point estimate with CCT (2014) bias correction.

    Single-eval port of ``nprobust::lprobust`` (lprobust.R:177-246) from
    ``nprobust`` version ``NPROBUST_VERSION`` (SHA ``NPROBUST_SHA``).
    Produces both the classical and bias-corrected point estimates along
    with their naive and robust standard errors.

    Computation (source-mapped to lprobust.R:177-246):

        w.h = kernel_W((x - eval_point) / h, kernel) / h   # line 177
        w.b = kernel_W((x - eval_point) / b, kernel) / b   # line 178
        ind.h = w.h > 0;  ind.b = w.b > 0                  # line 179
        ind = ind.b                                         # line 181 (default)
        if h > b: ind = ind.h                               # line 182 (conditional
                                                            #   replacement, NOT union)
        eN = sum(ind); eY = y[ind]; eX = x[ind]            # line 189-191
        W.h = w.h[ind]; W.b = w.b[ind]                     # line 192-193
        u_h = (eX - eval_point) / h                        # line 212
        R.q[:, j] = (eX - eval_point)^j, j = 0..q          # line 213-214
        R.p = R.q[:, :p+1]                                 # line 215
        L = (R.p * W.h).T @ u_h^(p+1)                      # line 218
        invG.q = qrXXinv(sqrt(W.b) * R.q)                  # line 219
        invG.p = qrXXinv(sqrt(W.h) * R.p)                  # line 220
        e_{p+2}: unit vector with 1 at position p+2         # line 221 (1-indexed)
        Q.q = (R.p * W.h) - h^(p+1) * L @ e_{p+2}.T
              @ (invG.q @ R.q.T * W.b).T                    # line 223 (eN, p+1)
        beta.p  = invG.p @ (R.p * W.h).T @ eY              # line 224
        beta.bc = invG.p @ Q.q.T @ eY                      # line 224
        tau.cl = deriv! * beta.p[deriv+1]                  # line 226
        tau.bc = deriv! * beta.bc[deriv+1]                 # line 227
        res.h = lprobust_res(..., j=p+1, vce=vce)          # line 239
        res.b = res.h  if vce=="nn"  else
                lprobust_res(..., j=q+1)                   # line 240-241
        V.Y.cl = invG.p @ lprobust_vce(R.p*W.h, res.h, eC) @ invG.p   # line 243
        V.Y.bc = invG.p @ lprobust_vce(Q.q,     res.b, eC) @ invG.p   # line 244
        se.cl  = sqrt(deriv!^2 * V.Y.cl[deriv+1, deriv+1])            # line 245
        se.rb  = sqrt(deriv!^2 * V.Y.bc[deriv+1, deriv+1])            # line 246

    Parameters
    ----------
    y, x : np.ndarray, shape (N,)
        Outcome and regressor.
    eval_point : float
        Evaluation point ``c``.
    h, b : float
        Main and bias-correction bandwidths. Must both be positive.
        With nprobust's ``rho=1`` default, the caller passes ``h == b``.
    p : int, default=1
        Main polynomial order.
    q : int or None, default=None
        Bias-correction polynomial order. ``None`` resolves to ``p + 1``.
    deriv : int, default=0
        Derivative order to estimate. Must satisfy ``deriv <= p``.
    kernel : {"epa", "uni", "tri", "gau"}, default="epa"
    vce : {"nn", "hc0", "hc1", "hc2", "hc3"}, default="nn"
    cluster : np.ndarray or None
        Per-observation cluster IDs for cluster-robust variance. Length
        must match ``x``. ``None`` for unclustered.
    nnmatch : int, default=3
        Number of nearest neighbors for ``vce="nn"`` residuals.
    bwcheck : int or None, default=21
        Floor ``h`` and ``b`` at the distance to the ``bwcheck``-th nearest
        neighbor of ``eval_point``. Matches nprobust's default.
    weights : np.ndarray or None, default=None
        Per-observation non-negative weights (e.g., survey sampling
        weights). When provided, they multiply the kernel weights
        pointwise (``W_h = w_h * weights``; ``W_b = w_b * weights``) and
        propagate through all downstream computations — design matrices,
        ``Q.q`` bias-correction, variance matrices, ``hii`` for
        ``vce in {hc2, hc3}``. When ``weights=None`` the function is
        bit-identical to the unweighted path (regression-tested at
        ``atol=1e-14, rtol=1e-14``). Observations with ``weights[i] == 0``
        drop out of the active kernel window via the ``w > 0`` selector.

        **Parity gap**: no public weighted-CCF reference exists
        (nprobust has no weight argument). The uniform-weights bit-parity
        test is the only bit-parity anchor; validation under informative
        weights relies on MC oracle consistency + ``np::npreg`` partial
        parity on the raw local linear (no CCT bias correction).

    Returns
    -------
    LprobustResult

    Raises
    ------
    ValueError
        On shape mismatch, empty/non-finite inputs, unknown ``kernel``/
        ``vce``, ``deriv > p``, or a rank-deficient design (surfaced from
        ``qrXXinv``).
    """
    # --- input coercion + validation ---
    if kernel not in _VALID_KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}. Expected one of {_VALID_KERNELS}.")
    if vce not in _VALID_VCE:
        raise ValueError(f"Unknown vce {vce!r}. Expected one of {_VALID_VCE}.")
    if p < 0 or deriv < 0 or nnmatch <= 0:
        raise ValueError("p, deriv, nnmatch must be nonneg integers; nnmatch > 0.")
    if deriv > p:
        raise ValueError(f"deriv ({deriv}) cannot exceed p ({p}).")
    if q is None:
        q = p + 1
    if q < p:
        raise ValueError(f"q ({q}) cannot be less than p ({p}).")
    if not (np.isfinite(h) and h > 0):
        raise ValueError(f"h must be finite and positive; got {h!r}.")
    if not (np.isfinite(b) and b > 0):
        raise ValueError(f"b must be finite and positive; got {b!r}.")
    if not np.isfinite(eval_point):
        raise ValueError(f"eval_point must be finite; got {eval_point!r}.")

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape; got {x.shape} and {y.shape}.")
    N = x.shape[0]
    if N == 0:
        raise ValueError("x and y must be non-empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values (NaN or Inf).")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf).")

    if cluster is not None:
        cluster = np.asarray(cluster).ravel()
        if cluster.shape[0] != N:
            raise ValueError(f"cluster length ({cluster.shape[0]}) does not match x/y ({N}).")
        # Dtype-agnostic missingness check. Float NaN/Inf, object None,
        # and object np.nan all get rejected here (shared with
        # `lpbwselect_mse_dpi` via `_cluster_has_missing`) so the
        # downstream `lprobust_vce` cluster grouping on `np.unique`
        # cannot silently treat a missing sentinel as a real cluster.
        if _cluster_has_missing(cluster):
            raise ValueError(
                "cluster contains missing values (NaN / None). "
                "Filter your data before the call or drop missing "
                "observations explicitly."
            )

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.shape[0] != N:
            raise ValueError(f"weights length ({weights.shape[0]}) does not match x/y ({N}).")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contains non-finite values (NaN or Inf).")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if np.sum(weights) <= 0:
            raise ValueError("weights sum to zero — no observations have positive weight.")

    # --- vce="nn" setup: sort ascending, precompute dups ---
    dups: Optional[np.ndarray] = None
    dupsid: Optional[np.ndarray] = None
    # Track the sort permutation so that when ``return_influence=True``
    # we can unsort the IF back to the caller-supplied ordering below.
    order_x: Optional[np.ndarray] = None
    if vce == "nn":
        order_x = np.argsort(x, kind="mergesort")  # stable, matches R order()
        x = x[order_x]
        y = y[order_x]
        if cluster is not None:
            cluster = cluster[order_x]
        if weights is not None:
            weights = weights[order_x]
        dups, dupsid = _precompute_nn_duplicates(x)

    # --- bwcheck floor (lprobust.R:169-175) ---
    if bwcheck is not None:
        if bwcheck < 1 or bwcheck > N:
            raise ValueError(f"bwcheck must be in [1, {N}]; got {bwcheck}.")
        bw_min = float(np.sort(np.abs(x - eval_point))[bwcheck - 1])
        h = max(h, bw_min)
        b = max(b, bw_min)

    # --- kernel weights and active-window selection (lprobust.R:177-182) ---
    w_h = kernel_W((x - eval_point) / h, kernel) / h
    w_b = kernel_W((x - eval_point) / b, kernel) / b
    # Compose user weights into kernel weights (Phase 4.5 survey support).
    # Design matrices, Q.q, and variance propagation all source from
    # W_h/W_b, so multiplying here threads weights through the entire
    # downstream pipeline. Observations with weights[i]==0 get w_h[i]=0
    # and drop out of the active window via the `w>0` selector.
    if weights is not None:
        w_h = w_h * weights
        w_b = w_b * weights
    ind_h = w_h > 0
    ind_b = w_b > 0

    # Single-eval CONDITIONAL REPLACEMENT: default to the b-window, swap to
    # the h-window only when h > b. This is NOT a union; the union form
    # appears only in the out-of-scope covgrid branch (lprobust.R:275).
    ind = ind_b.copy()
    if h > b:
        ind = ind_h.copy()

    eN = int(np.sum(ind))
    if eN < (p + 1):
        raise ValueError(
            f"Active kernel window retains only {eN} observations; need at "
            f"least p+1={p+1} for the local-polynomial fit. Widen h/b or "
            f"pick a boundary with more distinct values nearby."
        )

    eY = y[ind]
    eX = x[ind]
    W_h = w_h[ind]
    W_b = w_b[ind]

    eC: Optional[np.ndarray] = None
    if cluster is not None:
        eC = cluster[ind]

    edups: Optional[np.ndarray] = None
    edupsid: Optional[np.ndarray] = None
    if vce == "nn":
        assert dups is not None and dupsid is not None  # set above
        edups = dups[ind]
        edupsid = dupsid[ind]

    # --- design matrices (lprobust.R:212-215) ---
    u_h = (eX - eval_point) / h
    R_q = np.empty((eN, q + 1), dtype=np.float64)
    for j in range(q + 1):
        R_q[:, j] = (eX - eval_point) ** j
    R_p = R_q[:, : p + 1]

    # --- L vector (lprobust.R:218): L = crossprod(R.p*W.h, u^(p+1)) ---
    L = (R_p * W_h[:, None]).T @ (u_h ** (p + 1))  # shape (p+1,)

    # --- Inverses (lprobust.R:219-220) ---
    # qrXXinv expects an already-row-scaled design; sqrt(W) * R.
    invG_q = qrXXinv(R_q * np.sqrt(W_b)[:, None])
    invG_p = qrXXinv(R_p * np.sqrt(W_h)[:, None])

    # --- Q.q combined design matrix (lprobust.R:223) ---
    # e.p1 has 1 at R-index p+2 (1-indexed) = Python index p+1 (0-indexed).
    e_p1 = np.zeros(q + 1, dtype=np.float64)
    e_p1[p + 1] = 1.0

    # R: Q.q <- t(t(R.p*W.h) - h^(p+1)*(L%*%t(e.p1))%*%t(t(invG.q%*%t(R.q))*W.b))
    # Unpacking in Python (see block comment above for derivation).
    # BLAS matmul on some platforms (Accelerate / OpenBLAS) can raise
    # spurious divide/overflow/invalid FPE warnings on SIMD intermediates
    # even when the input and output are finite and bounded. These are
    # known benign (numpy issue #21432 pattern); suppress locally.
    R_p_W_h = R_p * W_h[:, None]  # (eN, p+1)
    L_outer = np.outer(L, e_p1)  # (p+1, q+1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        C = invG_q @ R_q.T  # (q+1, eN)
        CW = C * W_b[None, :]  # (q+1, eN), column j scaled by W_b[j]
        F = (h ** (p + 1)) * (L_outer @ CW)  # (p+1, eN)
    Q_q = (R_p_W_h.T - F).T  # (eN, p+1)
    if not np.all(np.isfinite(Q_q)):
        raise ValueError(
            "Q.q bias-combined design matrix contains non-finite values. "
            "This typically signals a degenerate bandwidth pair (h or b "
            "near zero) or a rank-deficient window."
        )

    # --- beta.p and beta.bc (lprobust.R:224) ---
    beta_p = invG_p @ (R_p_W_h.T @ eY)  # (p+1,)
    beta_bc = invG_p @ (Q_q.T @ eY)  # (p+1,)

    deriv_fact = float(math.factorial(deriv))
    tau_cl = deriv_fact * float(beta_p[deriv])
    tau_bc = deriv_fact * float(beta_bc[deriv])

    # --- Residuals (lprobust.R:239-241) ---
    # hc2/hc3 need the hat-matrix diagonal hii. For vce="nn", hii is unused.
    hii: Optional[np.ndarray] = None
    predicts_p: Optional[np.ndarray] = None
    predicts_q: Optional[np.ndarray] = None
    if vce in ("hc0", "hc1", "hc2", "hc3"):
        predicts_p = (R_p @ beta_p).reshape(-1, 1)
        # R.q @ beta.q where beta.q is the q-polynomial fit. But the R code
        # computes beta.q only for hc2/hc3 and reuses it for predicts.q. We
        # compute it on demand.
        beta_q = invG_q @ ((R_q * W_b[:, None]).T @ eY)  # (q+1,)
        predicts_q = (R_q @ beta_q).reshape(-1, 1)
        if vce in ("hc2", "hc3"):
            # hii[j] = R.p[j] @ invG.p @ (R.p * W.h)[j]
            #       = W.h[j] * R.p[j] @ invG.p @ R.p[j]
            RpG = R_p @ invG_p  # (eN, p+1)
            hii = (np.sum(RpG * R_p, axis=1) * W_h).reshape(-1, 1)

    res_h = lprobust_res(
        eX,
        eY,
        predicts_p if predicts_p is not None else np.zeros((eN, 1)),
        hii,
        vce,
        nnmatch,
        edups,
        edupsid,
        p + 1,
    )
    if vce == "nn":
        res_b = res_h
    else:
        res_b = lprobust_res(
            eX,
            eY,
            predicts_q if predicts_q is not None else np.zeros((eN, 1)),
            hii,
            vce,
            nnmatch,
            edups,
            edupsid,
            q + 1,
        )

    # --- Variance matrices (lprobust.R:243-244) ---
    V_Y_cl = invG_p @ lprobust_vce(R_p_W_h, res_h, eC) @ invG_p
    V_Y_bc = invG_p @ lprobust_vce(Q_q, res_b, eC) @ invG_p

    # --- Standard errors (lprobust.R:245-246) ---
    se_cl = float(np.sqrt((deriv_fact**2) * V_Y_cl[deriv, deriv]))
    se_rb = float(np.sqrt((deriv_fact**2) * V_Y_bc[deriv, deriv]))

    # Cluster-robust variance is unidentified when fewer than two clusters
    # contribute to the ACTIVE kernel window (``eC = cluster[ind]``): the
    # between-cluster meat is degenerate, so a finite ``se`` here would report
    # unidentified clustered inference as if identified. NaN both SEs so any
    # downstream inference (the ``safe_inference`` gate in
    # ``bias_corrected_local_linear``; HAD's beta-scale rescale) is NaN-coupled.
    # Unclustered fits (``eC is None``) are unaffected, and a clustered window
    # with >= 2 distinct clusters is bit-identical, so the DGP-4 golden parity
    # is preserved.
    if eC is not None and len(np.unique(eC)) < 2:
        se_cl = float("nan")
        se_rb = float("nan")

    # --- Per-observation influence function for the BIAS-CORRECTED point
    # estimate at ``deriv`` (Phase 4.5 survey composition).
    # Aligned with ``V_Y_bc`` (NOT ``V_Y_cl``) so survey-composed variance
    # through ``compute_survey_if_variance`` targets the same estimator
    # scale that HAD's beta-scale ATT uses (``tau_bc``-based). Using the
    # classical IF here would under-estimate the variance by ignoring the
    # bias-correction inflation, producing a silently wrong survey SE.
    #
    # Bias-corrected WLS IF decomposition (mirrors the ``V_Y_bc`` sandwich
    # inner at lprobust.R:244): beta_bc - beta  =  invG_p @ sum_g [ Q_q[g] · res_b[g] ],
    # so psi_g = deriv_fact · invG_p[deriv, :] · Q_q[g, :] · res_b[g].
    # The self-check ``sum(psi^2) == V_Y_bc[deriv, deriv]`` holds under
    # unclustered HC0; under clustering, compute_survey_if_variance
    # aggregates by PSU, which is what the survey path wants.
    # Observations outside the active window have Q_q[g, :]=0 row and
    # contribute IF=0. Length-N, aligned with ORIGINAL x ordering (inverse-
    # permuted when vce="nn" sorts). ---
    influence_function: Optional[np.ndarray] = None
    if return_influence:
        # Bias-corrected IF using Q_q + res_b (active-window only).
        row_deriv_bc = invG_p[deriv, :]  # (p+1,)
        coeff_active_bc = (Q_q @ row_deriv_bc).ravel()  # (eN,)
        res_b_flat = np.asarray(res_b).ravel()
        if_active = deriv_fact * coeff_active_bc * res_b_flat  # (eN,)
        # Map back to full N; zeros for obs outside window.
        if_full_sorted = np.zeros(N, dtype=np.float64)
        if_full_sorted[ind] = if_active
        if vce == "nn" and order_x is not None:
            # x was sorted ascending by order_x; invert the permutation
            # so IF aligns with the ORIGINAL caller-supplied x ordering.
            inv_order = np.empty(N, dtype=np.int64)
            inv_order[order_x] = np.arange(N)
            influence_function = if_full_sorted[inv_order]
        else:
            influence_function = if_full_sorted

    return LprobustResult(
        eval_point=float(eval_point),
        h=float(h),
        b=float(b),
        n_used=eN,
        tau_cl=tau_cl,
        tau_bc=tau_bc,
        se_cl=se_cl,
        se_rb=se_rb,
        V_Y_cl=V_Y_cl,
        V_Y_bc=V_Y_bc,
        influence_function=influence_function,
    )
