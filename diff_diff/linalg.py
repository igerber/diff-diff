"""
Unified linear algebra backend for diff-diff.

This module provides optimized OLS and variance estimation with an optional
Rust backend for maximum performance.

The key optimizations are:
1. scipy.linalg.lstsq with 'gelsd' driver (SVD-based, handles rank-deficient matrices)
2. Vectorized cluster-robust SE via groupby (eliminates O(n*clusters) loop)
3. Single interface for all estimators (reduces code duplication)
4. Optional Rust backend for additional speedup (when available)
5. R-style rank deficiency handling: detect, warn, and set NA for dropped columns

The Rust backend is automatically used when available, with transparent
fallback to NumPy/SciPy implementations.

Rank Deficiency Handling
------------------------
When a design matrix is rank-deficient (has linearly dependent columns), the OLS
solution is not unique. This module follows R's `lm()` approach:

1. Detect rank deficiency using pivoted QR decomposition
2. Identify which columns are linearly dependent
3. Drop redundant columns from the solve
4. Set NA (NaN) for coefficients of dropped columns
5. Warn with clear message listing dropped columns
6. Compute valid SEs for remaining (identified) coefficients

This is controlled by the `rank_deficient_action` parameter:
- "warn" (default): Emit warning, set NA for dropped coefficients
- "error": Raise ValueError with dropped column information
- "silent": No warning, but still set NA for dropped coefficients
"""

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.linalg import lstsq as scipy_lstsq
from scipy.linalg.lapack import dpocon

# Import Rust backend if available (from _backend to avoid circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_compute_robust_vcov,
    _rust_compute_robust_vcov_hc2,
    _rust_solve_ols,
    _rust_solve_ols_chol,
)

# Conley (1999) spatial HAC helpers live in diff_diff.conley to keep this
# module focused on linear-algebra primitives. Imported at the top so the
# `ConleyMetric` type alias is in scope for the public function signatures
# below (which advertise `conley_metric: ConleyMetric`).
from diff_diff.conley import (
    ConleyMetric,
    _compute_conley_vcov,
    _validate_conley_kwargs,
)

# =============================================================================
# Utility Functions
# =============================================================================


def _factorize_cluster_ids(cluster_ids: np.ndarray) -> np.ndarray:
    """
    Convert cluster IDs to contiguous integer codes for Rust backend.

    Handles string, categorical, or non-contiguous integer cluster IDs by
    mapping them to contiguous integers starting from 0.

    Parameters
    ----------
    cluster_ids : np.ndarray
        Cluster identifiers (can be strings, integers, or categorical).

    Returns
    -------
    np.ndarray
        Integer cluster codes (dtype int64) suitable for Rust backend.
    """
    # Use pandas factorize for efficient conversion of any dtype
    codes, _ = pd.factorize(cluster_ids)
    return codes.astype(np.int64)


# =============================================================================
# Rank Deficiency Detection and Handling
# =============================================================================


def _detect_rank_deficiency(
    X: np.ndarray,
    rcond: Optional[float] = None,
    *,
    _cert_out: Optional[dict] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Detect rank deficiency using pivoted QR decomposition.

    This follows R's lm() approach of using pivoted QR to detect which columns
    are linearly dependent. The pivoting ensures we drop the "least important"
    columns (those with smallest contribution to the column space).

    Rank detection is scale-invariant. A raw pivoted QR runs first; if the design
    is genuinely rank-deficient (the deficiency persists under equilibration) its
    raw pivot selects which columns to drop, preserving the established drop
    order. But if the raw deficiency DISAPPEARS once columns are equilibrated to
    unit 2-norm (a scale artifact), the higher equilibrated rank and its pivot
    selection are used instead. This repairs the case where a column on a large
    raw scale (e.g. an unstandardized covariate ~1e8) inflates the rank threshold
    — which is anchored to the largest pivot diagonal — and false-drops
    well-scaled columns (intercept / treatment / interaction) on an otherwise
    full-rank design. In both branches the retained columns are guaranteed full
    rank (pivoted QR has strictly decreasing |R[i,i]|). For a genuinely collinear
    design with no scale disparity the dropped column is unchanged.

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Design matrix.
    rcond : float, optional
        Relative condition number threshold for determining rank.
        Diagonal elements of R smaller than rcond * max(|R_ii|) are treated
        as zero. If None, uses 1e-07 to match R's qr() default tolerance.

    Returns
    -------
    rank : int
        Numerical rank of the matrix.
    dropped_cols : ndarray of int
        Indices of columns that are linearly dependent (should be dropped).
        Empty if matrix is full rank.
    pivot : ndarray of int
        Column permutation from QR decomposition. For a full-rank result the
        pivot carries no information (no caller consumes it; the stage-0
        certification below returns a trivial ``arange`` pivot).

    Other Parameters
    ----------------
    _cert_out : dict, optional
        Private out-parameter for the opt-in solve_ols Cholesky fast path.
        When a dict is passed and stage-0 certification runs, it is populated
        with the stage-0 artifacts (``gram``, ``scales``, ``gram_eq``,
        ``eig_min``, ``eig_max`` and, on the certified branch, ``certified``)
        so the caller can reuse the Gram work instead of rebuilding it. Pure
        out-parameter: the return value and every rank decision are unchanged.
    """
    n, k = X.shape
    if k == 0:
        return 0, np.array([], dtype=int), np.array([], dtype=int)

    # R's qr() uses tol = 1e-07 by default (sqrt(eps) ≈ 1.49e-08); we use 1e-07.
    if rcond is None:
        rcond = 1e-07

    # Stage 0 — Gram full-rank CERTIFICATION (perf fast path; decisions never
    # change). The Gram is built directly from X (no equilibrated copy of the
    # tall matrix); diag(G) holds the squared column 2-norms, so equilibrating
    # G symmetrically by sqrt(diag(G)) is exactly column equilibration of X —
    # the `_rank_guarded_inv` convention. Certification threshold 1e-10 is the
    # documented Gram constant from `_rank_guarded_inv` (a Gram squares the
    # condition number of X, so 1e-10 on eigenvalues ~ cond(X_eq) < 1e5, two
    # orders STRICTER than the 1e-7 QR full-rank boundary): stage 0 never
    # reports full rank where the two-stage QR below would report a
    # deficiency, it only declines and falls through. (A pathological
    # Kahan-type matrix could in principle make pivoted-QR R-diagonals
    # undershoot the singular values enough to open a gap; real DiD designs —
    # dummies plus covariates — do not have that structure, and the
    # characterization test documents it.) Skipped when n < k (always
    # deficient — also keeps the sole pivot consumer, staggered.py's
    # underdetermined pair solve, structurally on the QR path) and when a
    # caller passes a LOOSER-than-default rcond (no caller does today; the
    # stricter-than-QR guarantee above assumes rcond <= 1e-7). Non-finite
    # entries poison diag(G), decline certification, and fall through so
    # scipy's qr raises ValueError on NaN/Inf exactly as before.
    if n >= k and rcond <= 1e-07:
        gram = X.T @ X
        diag = np.diag(gram)
        if np.all(np.isfinite(diag)) and np.all(diag > 0):
            scales = np.sqrt(diag)
            gram_eq = gram / scales[:, None] / scales[None, :]
            eigvals = np.linalg.eigvalsh(gram_eq)
            eig_min, eig_max = eigvals[0], eigvals[-1]
            if _cert_out is not None:
                _cert_out.update(
                    gram=gram,
                    scales=scales,
                    gram_eq=gram_eq,
                    eig_min=eig_min,
                    eig_max=eig_max,
                )
            if (
                np.isfinite(eig_min)
                and np.isfinite(eig_max)
                and eig_max > 0.0
                and eig_min > 1e-10 * eig_max
            ):
                if _cert_out is not None:
                    _cert_out["certified"] = True
                return k, np.array([], dtype=int), np.arange(k, dtype=int)

    def _rank_and_pivot(M: np.ndarray) -> Tuple[int, np.ndarray]:
        # Pivoted QR: M @ P = Q @ R. The rank threshold is anchored to the
        # largest pivot diagonal |R[0,0]| (decreasing after pivoting).
        # mode="r" skips forming the (unused) Q: R and the pivot come from
        # the same dgeqp3 factorization either way, so rank decisions are
        # bit-identical; only the dorgqr Q-formation work is saved.
        R, piv = qr(M, mode="r", pivoting=True)
        r_diag = np.abs(np.diag(R))
        if r_diag[0] == 0:
            return 0, piv
        return int(np.sum(r_diag > rcond * r_diag[0])), piv

    # Stage 1 — raw pivoted QR. The common full-rank case exits here with zero
    # added cost, and (when the design IS genuinely rank-deficient) this raw
    # pivot is what selects which columns to drop, so the established
    # drop-column selection for collinear designs is preserved exactly.
    rank_raw, pivot_raw = _rank_and_pivot(X)
    if rank_raw == k:
        return k, np.array([], dtype=int), pivot_raw

    # Stage 2 — raw QR reported a deficiency. That can be GENUINE collinearity
    # or a SCALE artifact: the threshold is anchored to the largest pivot
    # diagonal, so a column on a large raw scale (e.g. an unstandardized
    # covariate ~1e8) inflates the threshold and false-drops well-scaled columns
    # (intercept / treatment / interaction) on an otherwise full-rank design.
    # Equilibrate each column to unit 2-norm and re-detect the rank COUNT only;
    # if it is higher, the raw drop was scale-induced. Zero-norm columns keep
    # scale 1.0 (no divide-by-zero; still pivot last).
    col_norms = np.sqrt(np.einsum("ij,ij->j", X, X))
    safe_norms = np.where(col_norms > 0, col_norms, 1.0)
    rank_eq, pivot_eq = _rank_and_pivot(X / safe_norms)

    if rank_eq > rank_raw:
        # SCALE-INDUCED under-count: the raw threshold was inflated by a
        # large-scale column. Trust the equilibrated rank AND its pivot — its
        # first `rank_eq` columns are independent under the scale-corrected
        # criterion (pivoted QR has strictly decreasing |R[i,i]|, so the kept
        # set is guaranteed full rank and the retained design is identified).
        # The raw pivot's tail is NOT reused here because, with a scale-corrupted
        # ordering, its first `rank_eq` columns need not coincide with a
        # scale-corrected independent subset.
        rank, pivot = rank_eq, pivot_eq
    else:
        # GENUINE collinearity (no scale disparity): preserve the established
        # raw pivot drop selection so the dropped column is unchanged.
        rank, pivot = rank_raw, pivot_raw

    # Columns after the rank position (in pivot order) are linearly dependent.
    # The pivot indexes the ORIGINAL columns, so no remapping is needed.
    if rank < k:
        dropped_cols = np.sort(pivot[rank:])
    else:
        dropped_cols = np.array([], dtype=int)

    return rank, dropped_cols, pivot


def _format_dropped_columns(
    dropped_cols: np.ndarray,
    column_names: Optional[List[str]] = None,
) -> str:
    """
    Format dropped column information for error/warning messages.

    Parameters
    ----------
    dropped_cols : ndarray of int
        Indices of dropped columns.
    column_names : list of str, optional
        Names for the columns. If None, uses indices.

    Returns
    -------
    str
        Formatted string describing dropped columns.
    """
    if len(dropped_cols) == 0:
        return ""

    if column_names is not None:
        names = [column_names[i] if i < len(column_names) else f"column {i}" for i in dropped_cols]
        if len(names) == 1:
            return f"'{names[0]}'"
        elif len(names) <= 5:
            return ", ".join(f"'{n}'" for n in names)
        else:
            shown = ", ".join(f"'{n}'" for n in names[:5])
            return f"{shown}, ... and {len(names) - 5} more"
    else:
        if len(dropped_cols) == 1:
            return f"column {dropped_cols[0]}"
        elif len(dropped_cols) <= 5:
            return ", ".join(f"column {i}" for i in dropped_cols)
        else:
            shown = ", ".join(f"column {i}" for i in dropped_cols[:5])
            return f"{shown}, ... and {len(dropped_cols) - 5} more"


def _expand_coefficients_with_nan(
    coef_reduced: np.ndarray,
    k_full: int,
    kept_cols: np.ndarray,
) -> np.ndarray:
    """
    Expand reduced coefficients to full size, filling dropped columns with NaN.

    Parameters
    ----------
    coef_reduced : ndarray of shape (rank,)
        Coefficients for kept columns only.
    k_full : int
        Total number of columns in original design matrix.
    kept_cols : ndarray of int
        Indices of columns that were kept.

    Returns
    -------
    ndarray of shape (k_full,)
        Full coefficient vector with NaN for dropped columns.
    """
    coef_full = np.full(k_full, np.nan)
    coef_full[kept_cols] = coef_reduced
    return coef_full


def _absorbed_fe_vcov_scale(n_eff: float, k_eff: int, df_adjustment: int) -> float:
    """Finite-sample vcov rescale for absorbed fixed effects (fixest full-K).

    Within-transform (``absorb=``) fits residualize the FE out of the design, so
    the classical ``sse/(n-k)`` and HC1 ``n/(n-k)`` variance factors use only the
    *visible* regressor count ``k_eff = k_visible``. fixest (and the reported
    t-``df``) count the absorbed FE too, i.e. ``K_full = k_visible +
    df_adjustment``. This returns the scalar that maps the ``k_visible`` vcov to
    the ``K_full`` one::

        vcov_full = vcov_visible * (n_eff - k_eff) / (n_eff - k_eff - df_adjustment)

    which is algebraically exact for both the classical and HC1 factors (both are
    pure scalars in ``1/(n-k)``; the single-coefficient robust variance is
    FWL-invariant between the demeaned and full-dummy designs, so only the
    ``1/(n-k)`` scalar differs).

    Returns:
    - ``1.0`` when ``df_adjustment <= 0`` (no absorbed FE -- a no-op).
    - the finite scale when the full-K residual dof
      ``(n_eff - k_eff - df_adjustment)`` is positive.
    - ``nan`` (fail-closed) when that full-K residual dof (or the visible dof) is
      non-positive: the full-K variance is undefined for such a saturated
      within-transform design, so callers void the vcov to NaN -> NaN inference
      (per the non-finite-df fail-closed contract) rather than leaving a
      misleading ``k_visible`` SE in place.

    Callers must gate on non-clustered ``classical``/``hc1``: clustered SEs
    follow fixest's ``ssc`` nested-FE convention (FE nested in the cluster are
    not counted, so ``k_visible`` already matches for the nested case) and
    ``hc2``/``hc2_bm`` use leverage / Satterthwaite DOF -- none must be rescaled.
    """
    denom_visible = n_eff - k_eff
    denom_full = n_eff - k_eff - df_adjustment
    if df_adjustment <= 0:
        return 1.0
    if denom_full <= 0 or denom_visible <= 0:
        return float("nan")
    return denom_visible / denom_full


def _expand_vcov_with_nan(
    vcov_reduced: np.ndarray,
    k_full: int,
    kept_cols: np.ndarray,
) -> np.ndarray:
    """
    Expand reduced vcov matrix to full size, filling dropped entries with NaN.

    Parameters
    ----------
    vcov_reduced : ndarray of shape (rank, rank)
        Variance-covariance matrix for kept columns only.
    k_full : int
        Total number of columns in original design matrix.
    kept_cols : ndarray of int
        Indices of columns that were kept.

    Returns
    -------
    ndarray of shape (k_full, k_full)
        Full vcov matrix with NaN for dropped rows/columns.
    """
    vcov_full = np.full((k_full, k_full), np.nan)
    # Use advanced indexing to fill in the kept entries
    ix = np.ix_(kept_cols, kept_cols)
    vcov_full[ix] = vcov_reduced
    return vcov_full


def _equilibrated_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve OLS via column-equilibrated lstsq for scale robustness.

    ``scipy_lstsq``'s ``cond=1e-7`` truncates singular values relative to the
    largest, so a column on a large scale truncates the genuine small-scale
    direction and returns finite-but-wrong coefficients (e.g. a covariate ~1e8
    alongside a 0/1 dummy). Equilibrating each column to unit 2-norm before the
    solve and unscaling the result (``beta = beta_scaled / norm``) is
    algebraically exact for the raw system and numerically scale-invariant.

    Zero-norm columns use scale 1.0 (their coefficient is unidentified anyway and
    is handled by the rank-deficiency path upstream).
    """
    col_norms = np.sqrt(np.einsum("ij,ij->j", X, X))
    safe_norms = np.where(col_norms > 0, col_norms, 1.0)
    # Materialize the scaled temporary F-order and let gelsd consume it
    # in place: scipy honors overwrite_a only for F-contiguous input (it
    # silently re-copies C-order), and LAPACK wants F-order anyway, so this
    # skips the one internal copy lstsq would otherwise make. The temporary
    # is exclusively ours (never reused after the call), and the VALUES are
    # identical to the previous `X / safe_norms`, so results are bit-equal.
    x_scaled = np.divide(X, safe_norms, out=np.empty(X.shape, order="F"))
    coef_scaled = scipy_lstsq(
        x_scaled,
        y,
        lapack_driver="gelsd",
        check_finite=False,
        cond=1e-07,
        overwrite_a=True,
    )[0]
    return coef_scaled / safe_norms


# Reciprocal-condition guard for the opt-in solve_ols normal-equations Cholesky
# fast path. Same 1e-6 bound and rationale as _IRLS_CHOL_RCOND_GUARD (see the
# solve_logit IRLS inner solve): cond(G_eq) <= 1e6 bounds the Cholesky forward
# error at ~eps*cond ~ 2e-10 relative in the equilibrated basis. Kept as a
# separate constant so the two paths can be tuned independently.
_SOLVE_OLS_CHOL_RCOND_GUARD = 1e-6

# Module default for the opt-in fast path (OFF = byte-identical legacy
# behavior). The env var is read PER CALL (see resolver below) so benchmarks
# and tests can A/B within one process; this constant is the monkeypatch seam
# and the fallback for unset/invalid env values.
_SOLVE_OLS_FASTPATH: bool = False


def _resolve_solve_ols_fastpath() -> bool:
    """Resolve the DIFF_DIFF_SOLVE_OLS_FASTPATH opt-in knob.

    Set to a positive integer (``1``) to enable the certification-gated
    normal-equations Cholesky fast path in ``solve_ols``. Read PER CALL, not
    at import, so a single process can A/B the two paths. Unset or invalid
    values (non-integer strings, zero, negatives) fall back silently to the
    module default, mirroring ``_resolve_demean_chunk_cols``'s convention.
    """
    raw = os.environ.get("DIFF_DIFF_SOLVE_OLS_FASTPATH")
    if raw is None:
        return _SOLVE_OLS_FASTPATH
    try:
        value = int(raw)
    except ValueError:
        return _SOLVE_OLS_FASTPATH
    return True if value > 0 else _SOLVE_OLS_FASTPATH


def _solve_ols_chol_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cert_info: Optional[dict] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Certified equilibrated normal-equations Cholesky solve (opt-in path).

    Returns ``(coefficients, gram_raw)`` on success, or ``None`` when
    certification declines — the caller then falls back VERBATIM to the gelsd
    path, so a decline is always output-identical to the fast path being off.
    ``gram_raw`` is exactly the ``X.T @ X`` expression on the same array, so
    the caller may reuse it as the robust-vcov bread matrix bit-for-bit.

    Guard chain mirrors the solve_logit IRLS inner solve: symmetric
    equilibration of the Gram (== column equilibration of X), ``cho_factor``,
    then an explicit ``dpocon`` reciprocal-condition estimate gated at
    ``_SOLVE_OLS_CHOL_RCOND_GUARD`` — factorization success alone is NOT a
    certificate (cho_factor can succeed with a garbage solution at
    cond ~ 1e10+). Stage-0 artifacts from ``_detect_rank_deficiency`` are
    reused when supplied (``cert_info``); the self-build branch exists for the
    ``skip_rank_check`` route where no stage-0 certification ran. Artifacts
    present but uncertified mean the design already failed the (looser)
    stage-0 rank gate, so the solve gate cannot pass: decline immediately.
    """
    n, k = X.shape
    if k == 0 or n < k:
        return None

    if cert_info is not None:
        if "gram_eq" not in cert_info or not cert_info.get("certified", False):
            return None
        gram = cert_info["gram"]
        scales = cert_info["scales"]
        gram_eq = cert_info["gram_eq"]
    else:
        # Self-build (skip_rank_check route). Same expressions and guards as
        # stage-0: decline unless diag is finite AND strictly positive (zero
        # diag = zero/underflowed column; non-finite diag = NaN/Inf in X —
        # any non-finite entry poisons its own column's diagonal).
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = X.T @ X
        diag = np.diag(gram)
        if not (np.all(np.isfinite(diag)) and np.all(diag > 0)):
            return None
        scales = np.sqrt(diag)
        gram_eq = gram / scales[:, None] / scales[None, :]

    # 1-norm BEFORE factorization (dpocon contract).
    anorm = float(np.max(np.sum(np.abs(gram_eq), axis=0)))
    if not np.isfinite(anorm):
        return None
    try:
        chol = cho_factor(gram_eq)
    except (np.linalg.LinAlgError, ValueError):
        return None
    rcond_gram, pocon_info = dpocon(chol[0], anorm)
    if not (
        pocon_info == 0 and np.isfinite(rcond_gram) and rcond_gram > _SOLVE_OLS_CHOL_RCOND_GUARD
    ):
        return None

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        xty = X.T @ y
    coefficients = cho_solve(chol, xty / scales) / scales
    if not np.all(np.isfinite(coefficients)):
        return None
    return coefficients, gram


@overload
def _rank_guarded_inv(
    A: np.ndarray,
    *,
    rcond: float = ...,
    tracker: Optional[list] = ...,
    return_dropped: Literal[False] = ...,
) -> Tuple[np.ndarray, int, int]: ...


@overload
def _rank_guarded_inv(
    A: np.ndarray,
    *,
    rcond: float = ...,
    tracker: Optional[list] = ...,
    return_dropped: Literal[True],
) -> Tuple[np.ndarray, int, int, np.ndarray]: ...


def _rank_guarded_inv(
    A: np.ndarray,
    *,
    rcond: float = 1e-10,
    tracker: Optional[list] = None,
    return_dropped: bool = False,
) -> Union[Tuple[np.ndarray, int, int], Tuple[np.ndarray, int, int, np.ndarray]]:
    """Rank-guarded (generalized) inverse of a symmetric PSD Gram matrix.

    Influence-function standard errors invert a covariate Gram matrix
    ``A = X'WX`` (or a propensity-score Hessian). A constant or collinear
    covariate makes ``A`` *near*-singular, but ``np.linalg.solve`` / ``inv``
    only raise ``LinAlgError`` on an *exactly* singular matrix, so they return a
    garbage inverse (entries ~1e13) that flows straight into the SE. This helper
    detects near-singularity in a scale-invariant way and returns a finite
    generalized inverse on the identified subspace, truncating redundant
    directions. It returns an all-NaN matrix only when the design collapses to
    rank 0.

    Parameters
    ----------
    A : np.ndarray
        Square, symmetric, positive-semidefinite Gram matrix (k x k).
    rcond : float, default 1e-10
        Relative eigenvalue threshold applied to the *symmetrically
        equilibrated* matrix ``D^{-1/2} A D^{-1/2}`` (``D = diag(A)``): a
        direction is truncated when its equilibrated eigenvalue is
        ``<= rcond * max_eigenvalue``. ``1e-10`` (not the design-side ``1e-7``
        of :func:`_detect_rank_deficiency`) because a Gram matrix squares the
        condition number of ``X``; matches EfficientDiD's ``tol / max_eigval``
        relative threshold.
    tracker : list, optional
        When provided and at least one direction is truncated, ONE
        condition-number sample of ``A`` is appended (under ``np.errstate``).
        Callers pass the per-fit fallback tracker and must NOT append
        themselves: this helper is the sole owner, so the aggregate fallback
        warning counts each deficient inversion exactly once.

    Returns
    -------
    (A_ginv, n_dropped, rank) : Tuple[np.ndarray, int, int]
        ``A_ginv`` is the generalized inverse (all-NaN when ``rank == 0``).
        ``n_dropped`` is the number of truncated directions, ``rank`` is
        ``k - n_dropped``.

    Notes
    -----
    Column-drop (not minimum-norm) generalized inverse: when ``A`` is
    rank-deficient the guarded path keeps the ``rank`` most-independent columns
    (pivoted QR on the equilibrated Gram) and inverts that principal submatrix,
    zero-filling the dropped rows/cols. This is a column-drop in the SAME FAMILY
    as the point estimate (``_detect_rank_deficiency`` / R's ``lm()``; drop
    redundant columns, not a minimum-norm pseudo-inverse), but the column
    selection is computed on the equilibrated Gram and is scale-invariant — so it
    may drop a different *member* of a collinear set than the point estimate's
    raw pivot under mixed-scale *exact* collinearity (a documented deviation that
    leaves the SE unchanged, since the identified subspace is the same whichever
    redundant member is dropped; see ``docs/methodology/REGISTRY.md``). It is
    still a column-drop, so the influence-function SE equals the well-conditioned
    (near-collinear) limit: replacing the exactly-collinear covariate with a
    near-collinear (full-rank) one yields the same SE to working precision. A
    minimum-norm pseudo-inverse would instead diverge from column-drop whenever
    the IF multiplier leaves ``range(A)`` — e.g. an outcome-regression bread fit
    on the *control* (or a treated sub-cell) sample multiplied by a mean from a
    cell where the covariate is NOT collinear — so it is rejected here. With
    column-drop there is no such divergence; a covariate that is rank-deficient
    only within one cell still legitimately enters the other cells' full-rank
    fits, so the ATT and SE reflect that (poor) covariate specification.

    The fast (well-conditioned) path returns ``np.linalg.solve(A, I)``
    unchanged, so well-conditioned fits are numerically unaffected.
    """
    k = A.shape[0]

    def _ret(inv, n_dropped, n_keep, dropped):
        # ``dropped`` is a length-k boolean mask of the truncated (unidentified)
        # coordinates. Callers that report a PER-COEFFICIENT SE from ``vcov``
        # diagonals must NaN the dropped coordinates in the FINAL vcov — the
        # zero-filled inverse would otherwise report ``se=0`` for an unidentified
        # coefficient. (Linear-combination consumers — e.g. an ATT or dose
        # prediction — keep the default 3-tuple: the dropped direction correctly
        # contributes 0 to an identified linear combination.)
        return (inv, n_dropped, n_keep, dropped) if return_dropped else (inv, n_dropped, n_keep)

    if k == 0:
        return _ret(np.zeros((0, 0), dtype=float), 0, 0, np.zeros(0, dtype=bool))

    # Symmetric equilibration: scale row/col i by sqrt(A[i, i]) so the
    # eigenvalue threshold is scale-invariant. Zero/negative diagonal -> 1.0
    # (such a direction has no information and pivots into the truncated tail).
    diag = np.diag(A).astype(float)
    scales = np.sqrt(np.where(diag > 0.0, diag, 1.0))
    inv_scales = 1.0 / scales
    A_eq = A * inv_scales[:, None] * inv_scales[None, :]

    # eigvalsh reads one triangle, so asymmetric round-off in A_eq is ignored.
    # Eigenvalues give the scale-invariant rank; pivoted QR (below) selects which
    # columns to keep when the design is deficient.
    eigvals = np.linalg.eigvalsh(A_eq)
    max_eig = float(eigvals[-1]) if eigvals.size else 0.0
    thresh = rcond * max_eig if max_eig > 0.0 else 0.0
    keep = eigvals > thresh
    n_keep = int(np.count_nonzero(keep))

    # Fast path: full rank -> exact solve (bit-identical to the prior code).
    if max_eig > 0.0 and n_keep == k:
        return _ret(np.linalg.solve(A, np.eye(k)), 0, k, np.zeros(k, dtype=bool))

    # Rank-deficient: record one condition-number sample for the aggregate
    # fallback warning (the helper is the sole owner of this append).
    if tracker is not None:
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            tracker.append(float(np.linalg.cond(A)))

    if n_keep == 0:
        return _ret(np.full((k, k), np.nan), k, 0, np.ones(k, dtype=bool))

    # Column-drop generalized inverse: keep the n_keep most-independent columns
    # (pivoted QR on the equilibrated Gram), invert that principal submatrix, and
    # zero-fill the dropped rows/cols before un-scaling. This is the same
    # generalized-inverse family the point estimate uses (drop redundant columns,
    # not a minimum-norm pseudo-inverse, which diverges from column-drop when the
    # IF multiplier leaves range(A) — e.g. a treated-cell mean). Equilibrating the
    # selection (rather than a raw pivot) keeps the cell-only-aliasing SE equal to
    # the well-conditioned near-collinear limit (se_ratio ~ 1).
    _R, piv = qr(A_eq, mode="r", pivoting=True)  # only the pivot is used
    kept = np.sort(piv[:n_keep])
    A_eq_ginv = np.zeros((k, k), dtype=float)
    A_eq_ginv[np.ix_(kept, kept)] = np.linalg.inv(A_eq[np.ix_(kept, kept)])
    A_ginv = A_eq_ginv * inv_scales[:, None] * inv_scales[None, :]
    n_dropped = k - n_keep
    dropped_mask = np.ones(k, dtype=bool)
    dropped_mask[kept] = False
    return _ret(A_ginv, n_dropped, n_keep, dropped_mask)


def _solve_ols_rust(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
) -> Optional[
    Union[
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
    ]
]:
    """
    Rust backend implementation of solve_ols for full-rank matrices.

    This is only called when:
    1. The Rust backend is available
    2. The design matrix is full rank (no rank deficiency handling needed)

    For rank-deficient matrices, the Python backend is used instead to
    properly handle R-style NA coefficients for dropped columns.

    Why the backends differ (by design):
    - Rust uses SVD-based solve (minimum-norm solution for rank-deficient)
    - Python uses pivoted QR to identify and drop linearly dependent columns
    - ndarray-linalg doesn't support QR with pivoting, so Rust can't identify
      which specific columns to drop
    - For full-rank matrices, both approaches give identical results
    - For rank-deficient matrices, only Python can provide R-style NA handling

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k), must be full rank.
    y : np.ndarray
        Response vector of shape (n,).
    cluster_ids : np.ndarray, optional
        Cluster identifiers for cluster-robust SEs.
    return_vcov : bool
        Whether to compute variance-covariance matrix.
    return_fitted : bool
        Whether to return fitted values.

    Returns
    -------
    coefficients : np.ndarray
        OLS coefficients of shape (k,).
    residuals : np.ndarray
        Residuals of shape (n,).
    fitted : np.ndarray, optional
        Fitted values if return_fitted=True.
    vcov : np.ndarray, optional
        Variance-covariance matrix if return_vcov=True.
    None
        If Rust backend detects numerical instability and caller should
        fall back to Python backend.
    """
    # Convert cluster_ids to int64 for Rust (handles string/categorical IDs)
    if cluster_ids is not None:
        _validate_cluster_ids(cluster_ids)
        cluster_ids = _factorize_cluster_ids(cluster_ids)

    # Call Rust backend with fallback on numerical instability
    try:
        coefficients, residuals, vcov = _rust_solve_ols(
            X, y, cluster_ids=cluster_ids, return_vcov=return_vcov
        )
    except ValueError as e:
        error_msg = str(e).lower()
        if "numerically unstable" in error_msg or "singular" in error_msg:
            warnings.warn(
                f"Rust backend detected numerical instability: {e}. "
                "Falling back to Python backend.",
                UserWarning,
                stacklevel=3,
            )
            return None  # Signal caller to use Python fallback
        raise

    # Convert to numpy arrays
    coefficients = np.asarray(coefficients)
    residuals = np.asarray(residuals)
    if vcov is not None:
        vcov = np.asarray(vcov)

    # Return with optional fitted values
    if return_fitted:
        fitted = np.dot(X, coefficients)
        return coefficients, residuals, fitted, vcov
    else:
        return coefficients, residuals, vcov


def _solve_ols_chol_rust(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
) -> Optional[
    Union[
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
    ]
]:
    """Rust certified normal-equations Cholesky solve (opt-in fast path).

    Mirrors ``_solve_ols_rust``'s wrapper contract. The kernel is
    self-certifying and returns Python ``None`` on a certification decline
    (zero/non-finite column norm, non-PD Gram, exact 1-norm rcond <= 1e-6,
    n < k, non-finite coefficient); this wrapper propagates the ``None`` so
    the dispatcher falls through VERBATIM to the SVD kernel and then the
    numpy path. The "Need at least 2 clusters" ValueError from clustered
    vcov re-raises exactly as on the SVD path (its message names neither
    instability nor singularity).
    """
    if cluster_ids is not None:
        _validate_cluster_ids(cluster_ids)
        cluster_ids = _factorize_cluster_ids(cluster_ids)

    try:
        result = _rust_solve_ols_chol(X, y, cluster_ids=cluster_ids, return_vcov=return_vcov)
    except ValueError as e:
        error_msg = str(e).lower()
        if "numerically unstable" in error_msg or "singular" in error_msg:
            warnings.warn(
                f"Rust backend detected numerical instability: {e}. "
                "Falling back to Python backend.",
                UserWarning,
                stacklevel=3,
            )
            return None
        raise

    if result is None:
        return None  # certification declined — caller falls through verbatim

    coefficients, residuals, vcov = result
    coefficients = np.asarray(coefficients)
    residuals = np.asarray(residuals)
    if vcov is not None:
        vcov = np.asarray(vcov)

    if return_fitted:
        fitted = np.dot(X, coefficients)
        return coefficients, residuals, fitted, vcov
    return coefficients, residuals, vcov


def _nonfinite_vcov_needs_python_rerun(
    vcov: Optional[np.ndarray], *, nan_is_sentinel: bool
) -> bool:
    """Shared guard: should a Rust-backend vcov be rejected in favor of the
    canonical numpy re-run?

    On the rank-checked route (``nan_is_sentinel=False``) any non-finite
    entry (NaN or Inf) reroutes: the design was certified full rank, so the
    numpy re-run safely applies the canonical contracts (rank handling,
    saturated all-NaN guard).

    On the ``skip_rank_check`` route (``nan_is_sentinel=True``) an all-NaN
    vcov is the DOCUMENTED sentinel for what the caller asserted away
    (rank deficiency / saturation) and passes through unchanged — rerouting
    it to a numpy path that assumes full rank could raise on a singular
    bread where users previously received the safe NaN answer. Only Inf
    (e.g. numerical overflow, the silent-corruption class) reroutes there.
    """
    if vcov is None:
        return False
    if nan_is_sentinel:
        return bool(np.any(np.isinf(vcov)))
    return not bool(np.all(np.isfinite(vcov)))


@overload
def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: Literal[False] = ...,
    check_finite: bool = ...,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    skip_rank_check: bool = ...,
    weights: Optional[np.ndarray] = ...,
    weight_type: str = ...,
    vcov_type: str = ...,
    conley_coords: Optional[np.ndarray] = ...,
    conley_cutoff_km: Optional[float] = ...,
    conley_metric: ConleyMetric = ...,
    conley_kernel: str = ...,
    conley_time: Optional[np.ndarray] = ...,
    conley_unit: Optional[np.ndarray] = ...,
    conley_lag_cutoff: Optional[int] = ...,
    diagnostics_out: Optional[dict] = ...,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: ...


@overload
def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: Literal[True],
    check_finite: bool = ...,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    skip_rank_check: bool = ...,
    weights: Optional[np.ndarray] = ...,
    weight_type: str = ...,
    vcov_type: str = ...,
    conley_coords: Optional[np.ndarray] = ...,
    conley_cutoff_km: Optional[float] = ...,
    conley_metric: ConleyMetric = ...,
    conley_kernel: str = ...,
    conley_time: Optional[np.ndarray] = ...,
    conley_unit: Optional[np.ndarray] = ...,
    conley_lag_cutoff: Optional[int] = ...,
    diagnostics_out: Optional[dict] = ...,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]: ...


@overload
def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: bool,
    check_finite: bool = ...,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    skip_rank_check: bool = ...,
    weights: Optional[np.ndarray] = ...,
    weight_type: str = ...,
    vcov_type: str = ...,
    conley_coords: Optional[np.ndarray] = ...,
    conley_cutoff_km: Optional[float] = ...,
    conley_metric: ConleyMetric = ...,
    conley_kernel: str = ...,
    conley_time: Optional[np.ndarray] = ...,
    conley_unit: Optional[np.ndarray] = ...,
    conley_lag_cutoff: Optional[int] = ...,
    diagnostics_out: Optional[dict] = ...,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
]: ...


_VALID_WEIGHT_TYPES = {"pweight", "fweight", "aweight"}


def _validate_weights(weights, weight_type, n):
    """Validate weights array and weight_type for solve_ols/LinearRegression."""
    if weight_type not in _VALID_WEIGHT_TYPES:
        raise ValueError(
            f"weight_type must be one of {_VALID_WEIGHT_TYPES}, " f"got '{weight_type}'"
        )
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != n:
            raise ValueError(f"weights length ({weights.shape[0]}) must match " f"X rows ({n})")
        if np.any(np.isnan(weights)):
            raise ValueError("Weights contain NaN values")
        if np.any(np.isinf(weights)):
            raise ValueError("Weights contain Inf values")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        if np.sum(weights) <= 0:
            raise ValueError(
                "Weights sum to zero — no observations have positive weight. "
                "Cannot fit a model on an empty effective sample."
            )
        if weight_type == "fweight":
            fractional = weights - np.round(weights)
            if np.any(np.abs(fractional) > 1e-10):
                raise ValueError(
                    "Frequency weights (fweight) must be non-negative integers. "
                    "Fractional values detected. Use pweight for non-integer weights."
                )
    return weights


def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
    check_finite: bool = True,
    rank_deficient_action: str = "warn",
    column_names: Optional[List[str]] = None,
    skip_rank_check: bool = False,
    weights: Optional[np.ndarray] = None,
    weight_type: str = "pweight",
    vcov_type: str = "hc1",
    conley_coords: Optional[np.ndarray] = None,
    conley_cutoff_km: Optional[float] = None,
    conley_metric: ConleyMetric = "haversine",
    conley_kernel: str = "bartlett",
    conley_time: Optional[np.ndarray] = None,
    conley_unit: Optional[np.ndarray] = None,
    conley_lag_cutoff: Optional[int] = None,
    diagnostics_out: Optional[dict] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
]:
    """
    Solve OLS regression with optional clustered standard errors.

    This is the unified OLS solver for all diff-diff estimators. It uses
    scipy's optimized LAPACK routines and vectorized variance estimation.

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Design matrix (should include intercept if desired).
    y : ndarray of shape (n,)
        Response vector.
    cluster_ids : ndarray of shape (n,), optional
        Cluster identifiers for cluster-robust standard errors.
        If None, HC1 (heteroskedasticity-robust) SEs are computed.
    return_vcov : bool, default True
        Whether to compute and return the variance-covariance matrix.
        Set to False for faster computation when SEs are not needed.
    return_fitted : bool, default False
        Whether to return fitted values in addition to residuals.
    check_finite : bool, default True
        Whether to check that X and y contain only finite values (no NaN/Inf).
        Set to False for faster computation if you are certain your data is clean.
    rank_deficient_action : str, default "warn"
        How to handle rank-deficient design matrices:
        - "warn": Emit warning and set NaN for dropped coefficients (R-style)
        - "error": Raise ValueError with dropped column information
        - "silent": No warning, but still set NaN for dropped coefficients
    column_names : list of str, optional
        Names for the columns (used in warning/error messages).
        If None, columns are referred to by their indices.
    skip_rank_check : bool, default False
        If True, skip the pivoted QR rank check and use Rust backend directly
        (when available). This saves O(nk²) computation but will not detect
        rank-deficient matrices. Use only when you know the design matrix is
        full rank. If the matrix is actually rank-deficient, results may be
        incorrect (minimum-norm solution instead of R-style NA handling).
    weights : ndarray of shape (n,), optional
        Observation weights for Weighted Least Squares. When provided,
        minimizes sum(w_i * (y_i - X_i @ beta)^2). Weights should be
        pre-normalized (e.g., mean=1 for pweights).
    weight_type : str, default "pweight"
        Type of weights: "pweight" (inverse selection probability),
        "fweight" (frequency), or "aweight" (inverse variance).
        Affects variance estimation but not coefficient computation.
    vcov_type : {"classical", "hc1", "hc2", "hc2_bm", "conley"}, default "hc1"
        Variance-covariance family forwarded to :func:`compute_robust_vcov`:

        - ``"classical"``: non-robust OLS SE, ``sigma_hat^2 * (X'X)^{-1}``.
          One-way only; raises if ``cluster_ids`` is also passed.
        - ``"hc1"``: heteroskedasticity-robust HC1 with ``n/(n-k)`` adjustment
          (default). With ``cluster_ids``, dispatches to CR1 (Liang-Zeger).
        - ``"hc2"``: leverage-corrected meat. One-way only; raises with
          ``cluster_ids`` (use ``"hc2_bm"`` for clustered Bell-McCaffrey).
        - ``"hc2_bm"``: HC2 + Imbens-Kolesar (2016) Satterthwaite DOF one-way;
          Pustejovsky-Tipton (2018) CR2 Bell-McCaffrey with ``cluster_ids``.
          With ``weights``, dispatches to the clubSandwich WLS-CR2 port —
          supported for ``weight_type="pweight"`` only. ``aweight`` and
          ``fweight`` raise ``NotImplementedError`` (port matches the
          ``pweight`` convention only; aweight/fweight derivations are a
          separate methodology task).
        - ``"conley"``: Conley (1999) spatial-HAC sandwich. Requires
          ``conley_coords`` (n × 2 array) and ``conley_cutoff_km`` (positive
          bandwidth, no default per Conley 1999 Section 5's sensitivity-grid
          recommendation). Two operating modes: cross-sectional (single-period
          design or pooled cross-section) and panel block-decomposed (matches
          R ``conleyreg`` with ``lag_cutoff > 0``); switch by passing the
          three co-required kwargs ``conley_time`` / ``conley_unit`` /
          ``conley_lag_cutoff``. Combining with ``cluster_ids`` applies the
          combined spatial + cluster product kernel (a diff-diff convention;
          see :func:`compute_robust_vcov` for details). Combining with
          ``weights`` raises ``NotImplementedError`` regardless of
          ``weight_type``: weighted Conley is not implemented on the generic
          linalg surface. For probability-sampling weights (``pweight`` /
          ``survey_design``) the deferral additionally reflects an open
          methodological question — no canonical extension of Conley (1999)
          exists for weighted spatial-HAC under probability sampling.
    conley_coords : ndarray of shape (n, 2), optional
        Required when ``vcov_type="conley"``. Two-column array of
        ``[lat, lon]`` (degrees, for ``conley_metric="haversine"``) or
        projected coordinates (for ``conley_metric="euclidean"`` / callable
        metric).
    conley_cutoff_km : float, optional
        Required when ``vcov_type="conley"``. Positive finite bandwidth in
        km (haversine) or coord units (euclidean / callable).
    conley_metric : {"haversine", "euclidean", callable}, default "haversine"
        Distance metric. Haversine uses Earth's mean radius 6371.01 km
        (matching R ``conleyreg``). Euclidean treats coords as already
        projected. Callable signature ``(coords1, coords2) -> n×n``.
    conley_kernel : {"bartlett", "uniform"}, default "bartlett"
        Kernel evaluated on pairwise distance ``d_ij/h``. Both kernels emit
        a ``UserWarning`` if the resulting meat is materially indefinite;
        the radial 1-D Bartlett (matching R ``conleyreg``) is not formally
        PSD-guaranteed — see :func:`compute_robust_vcov`.
    diagnostics_out : dict, optional
        Observability sink (zero-cost when None). Populated with
        ``"solve_ols_fastpath"`` ∈ {``"off"``, ``"chol_numpy"``,
        ``"chol_rust"``, ``"fallback_declined"``} recording which solve
        branch produced the coefficients under the opt-in
        ``DIFF_DIFF_SOLVE_OLS_FASTPATH`` normal-equations Cholesky fast
        path (``"off"`` when the knob is unset; ``"fallback_declined"``
        when the knob is on but the fast path did not produce the result —
        certification declined, the design was rank-deficient/structurally
        ineligible, or the Rust fast-path symbol was unavailable — and the
        verbatim legacy chain ran instead).

    Returns
    -------
    coefficients : ndarray of shape (k,)
        OLS coefficient estimates. For rank-deficient matrices, coefficients
        of linearly dependent columns are set to NaN.
    residuals : ndarray of shape (n,)
        Residuals (y - fitted). For rank-deficient matrices, uses only
        identified coefficients to compute fitted values.
    fitted : ndarray of shape (n,), optional
        Fitted values. For full-rank matrices, this is X @ coefficients.
        For rank-deficient matrices, uses only identified coefficients
        (X_reduced @ coefficients_reduced). Only returned if return_fitted=True.
    vcov : ndarray of shape (k, k) or None
        Variance-covariance matrix (HC1 or cluster-robust).
        For rank-deficient matrices, rows/columns for dropped coefficients
        are filled with NaN. None if return_vcov=False.

    Notes
    -----
    This function detects rank-deficient matrices using pivoted QR decomposition
    and handles them following R's lm() approach:

    1. Detect linearly dependent columns via pivoted QR
    2. Drop redundant columns and solve the reduced system
    3. Set NaN for coefficients of dropped columns
    4. Compute valid SEs for identified coefficients only
    5. Expand vcov matrix with NaN for dropped rows/columns

    The cluster-robust standard errors use the sandwich estimator with the
    standard small-sample adjustment: (G/(G-1)) * ((n-1)/(n-k)).

    Examples
    --------
    >>> import numpy as np
    >>> from diff_diff.linalg import solve_ols
    >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
    >>> y = 2 + 3 * X[:, 1] + np.random.randn(100)
    >>> coef, resid, vcov = solve_ols(X, y)
    >>> print(f"Intercept: {coef[0]:.2f}, Slope: {coef[1]:.2f}")

    For rank-deficient matrices with collinear columns:

    >>> X = np.random.randn(100, 3)
    >>> X[:, 2] = X[:, 0] + X[:, 1]  # Perfect collinearity
    >>> y = np.random.randn(100)
    >>> coef, resid, vcov = solve_ols(X, y)  # Emits warning
    >>> print(np.isnan(coef[2]))  # Dropped column has NaN coefficient
    True
    """
    # Validate inputs
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of observations: " f"{X.shape[0]} vs {y.shape[0]}"
        )

    n, k = X.shape
    if n < k:
        raise ValueError(
            f"Fewer observations ({n}) than parameters ({k}). "
            "Cannot solve underdetermined system."
        )

    # Validate rank_deficient_action
    valid_actions = {"warn", "error", "silent"}
    if rank_deficient_action not in valid_actions:
        raise ValueError(
            f"rank_deficient_action must be one of {valid_actions}, "
            f"got '{rank_deficient_action}'"
        )

    # Check for NaN/Inf values if requested
    if check_finite:
        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or Inf values. "
                "Clean your data or set check_finite=False to skip this check."
            )
        if not np.isfinite(y).all():
            raise ValueError(
                "y contains NaN or Inf values. "
                "Clean your data or set check_finite=False to skip this check."
            )

    # Front-door Conley-specific validation. Runs BEFORE the routing/backend
    # branching so `return_vcov=False` cannot bypass the conley+weights /
    # conley+cluster_ids guards. Conley needs this because survey-replicate
    # paths in MultiPeriodDiD pass `return_vcov=False` + `weights=survey_w`
    # and would otherwise silently return survey SEs under a Conley request.
    # The full `_validate_vcov_args` is NOT called here because other vcov
    # types (e.g. `hc2_bm` + replicate weights) intentionally fall through
    # to the survey-vcov path with the analytical validator skipped — that
    # legacy contract is preserved.
    if vcov_type == "conley":
        if cluster_ids is not None or weights is not None:
            _validate_vcov_args(vcov_type, cluster_ids, weights)

    # WLS transformation: apply sqrt(w) scaling to X and y
    # This happens BEFORE routing to Rust or NumPy backends — they receive
    # pre-transformed X_w, y_w and solve standard OLS.
    # Residuals are back-transformed to original scale afterward.
    _original_X = None
    _original_y = None
    if weights is not None:
        weights = _validate_weights(weights, weight_type, n)
        _original_X = X
        _original_y = y
        sqrt_w = np.sqrt(weights)
        X = X * sqrt_w[:, np.newaxis]
        y = y * sqrt_w

    # When weights are present, compute vcov separately on original-scale data
    # to avoid double-weighting. The backend only computes point estimates.
    _weighted_vcov_external = weights is not None
    _backend_return_vcov = return_vcov and not _weighted_vcov_external

    # Opt-in normal-equations Cholesky fast path (DIFF_DIFF_SOLVE_OLS_FASTPATH).
    # Resolved per call so a single process can A/B; OFF (the default) leaves
    # every line below byte-identical to the legacy path. On a certification
    # decline the fast path returns None and execution falls through the
    # UNCHANGED legacy chain (Rust SVD, then gelsd), so knob-on-decline output
    # equals knob-off output exactly.
    fastpath = _resolve_solve_ols_fastpath()
    if diagnostics_out is not None:
        # Pessimistic default; overwritten with the branch actually taken.
        diagnostics_out["solve_ols_fastpath"] = "fallback_declined" if fastpath else "off"

    # Fast path: skip rank check and use Rust directly when requested
    # This saves O(nk²) QR overhead but won't detect rank-deficient matrices
    result = None  # Will hold the tuple from backend functions

    if skip_rank_check:
        if (
            fastpath
            and HAS_RUST_BACKEND
            and _rust_solve_ols_chol is not None
            and weights is None
            and vcov_type == "hc1"
        ):
            # Self-certifying Rust Cholesky kernel (no stage-0 cert exists
            # on this route). None = certification declined; fall through
            # to the UNCHANGED legacy chain (Rust SVD, then numpy).
            result = _solve_ols_chol_rust(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
            )
            if result is not None and diagnostics_out is not None:
                diagnostics_out["solve_ols_fastpath"] = "chol_rust"
        if result is None and (
            HAS_RUST_BACKEND
            and _rust_solve_ols is not None
            and weights is None
            and vcov_type == "hc1"
        ):
            result = _solve_ols_rust(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
            )
            # result is None on numerical instability → fall through
            if result is not None and _nonfinite_vcov_needs_python_rerun(
                result[-1] if _backend_return_vcov else None,
                nan_is_sentinel=True,
            ):
                warnings.warn(
                    "Rust backend detected ill-conditioned matrix (non-finite "
                    "variance-covariance). Re-running with Python backend for "
                    "proper rank detection.",
                    UserWarning,
                    stacklevel=2,
                )
                result = None  # Force Python fallback below
        if result is None:
            result = _solve_ols_numpy(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
                rank_deficient_action=rank_deficient_action,
                column_names=column_names,
                _skip_rank_check=True,
                _fastpath=fastpath,
                _diagnostics_out=diagnostics_out,
                vcov_type=vcov_type,
                conley_coords=conley_coords,
                conley_cutoff_km=conley_cutoff_km,
                conley_metric=conley_metric,
                conley_kernel=conley_kernel,
                conley_time=conley_time,
                conley_unit=conley_unit,
                conley_lag_cutoff=conley_lag_cutoff,
            )
    else:
        # Check for rank deficiency using fast pivoted QR decomposition.
        # Rank detection operates on (possibly weighted) X since collinearity
        # depends on the weighted column space. When the fast path is on,
        # collect the stage-0 certification artifacts (Gram, scales,
        # eigenvalues) so the Cholesky solve can reuse them instead of
        # rebuilding the Gram — the artifacts are of the (possibly weighted)
        # X, which is exactly the matrix being solved.
        cert_out: Optional[dict] = {} if fastpath else None
        rank, dropped_cols, pivot = _detect_rank_deficiency(X, _cert_out=cert_out)
        is_rank_deficient = len(dropped_cols) > 0

        # Routing strategy:
        # - Full-rank + Rust available + no weights + HC1 vcov_type → fast Rust
        # - Weighted or rank-deficient or non-HC1 vcov_type → Python backend
        # - Rust numerical instability → Python fallback (via None return)
        if (
            fastpath
            and HAS_RUST_BACKEND
            and _rust_solve_ols_chol is not None
            and not is_rank_deficient
            and weights is None
            and vcov_type == "hc1"
        ):
            # Self-certifying Rust Cholesky kernel (rebuilds its own
            # equilibrated Gram; the stage-0 artifacts stay with the numpy
            # twin). None = certification declined; fall through to the
            # UNCHANGED legacy chain below.
            result = _solve_ols_chol_rust(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
            )
            if result is not None and diagnostics_out is not None:
                diagnostics_out["solve_ols_fastpath"] = "chol_rust"
        if result is None and (
            HAS_RUST_BACKEND
            and _rust_solve_ols is not None
            and not is_rank_deficient
            and weights is None
            and vcov_type == "hc1"
        ):
            result = _solve_ols_rust(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
            )

            if result is not None:
                vcov_check = result[-1] if _backend_return_vcov else None
                if _nonfinite_vcov_needs_python_rerun(vcov_check, nan_is_sentinel=False):
                    # Non-finite covers both the NaN rank-deficiency sentinel
                    # and Inf leakage (e.g. numerical overflow); the Python
                    # backend re-run applies the canonical numpy contracts.
                    warnings.warn(
                        "Rust backend detected ill-conditioned matrix (non-finite "
                        "variance-covariance). Re-running with Python backend for "
                        "proper rank detection.",
                        UserWarning,
                        stacklevel=2,
                    )
                    result = None  # Force Python fallback below

        if result is None:
            result = _solve_ols_numpy(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
                rank_deficient_action=rank_deficient_action,
                column_names=column_names,
                _precomputed_rank_info=(rank, dropped_cols, pivot),
                _fastpath=fastpath,
                _cert_info=cert_out,
                _diagnostics_out=diagnostics_out,
                vcov_type=vcov_type,
                conley_coords=conley_coords,
                conley_cutoff_km=conley_cutoff_km,
                conley_metric=conley_metric,
                conley_kernel=conley_kernel,
                conley_time=conley_time,
                conley_unit=conley_unit,
                conley_lag_cutoff=conley_lag_cutoff,
            )

    # Back-transform residuals and compute weighted vcov on original-scale data.
    # The WLS transform (sqrt(w) scaling) is for point estimates only. Vcov must
    # be computed on original X and residuals with weights applied exactly once.
    if _original_X is not None and _original_y is not None:
        if return_fitted:
            coefficients, _resid_w, _fitted_w, vcov_out = result
        else:
            coefficients, _resid_w, vcov_out = result

        # Handle rank-deficient case: use only identified columns for fitted values
        # to avoid NaN propagation from dropped coefficients
        nan_mask = np.isnan(coefficients)
        if np.any(nan_mask):
            kept_cols = np.where(~nan_mask)[0]
            fitted_orig = np.dot(_original_X[:, kept_cols], coefficients[kept_cols])
        else:
            fitted_orig = np.dot(_original_X, coefficients)
        residuals_orig = _original_y - fitted_orig

        if return_vcov:
            if np.any(nan_mask):
                kept_cols = np.where(~nan_mask)[0]
                if len(kept_cols) > 0:
                    vcov_reduced = _compute_robust_vcov_numpy(
                        _original_X[:, kept_cols],
                        residuals_orig,
                        cluster_ids,
                        weights=weights,
                        weight_type=weight_type,
                        vcov_type=vcov_type,
                        conley_coords=conley_coords,
                        conley_cutoff_km=conley_cutoff_km,
                        conley_metric=conley_metric,
                        conley_kernel=conley_kernel,
                        conley_time=conley_time,
                        conley_unit=conley_unit,
                        conley_lag_cutoff=conley_lag_cutoff,
                    )
                    vcov_out = _expand_vcov_with_nan(vcov_reduced, _original_X.shape[1], kept_cols)
                else:
                    vcov_out = np.full((_original_X.shape[1], _original_X.shape[1]), np.nan)
            else:
                vcov_out = _compute_robust_vcov_numpy(
                    _original_X,
                    residuals_orig,
                    cluster_ids,
                    weights=weights,
                    weight_type=weight_type,
                    vcov_type=vcov_type,
                    conley_coords=conley_coords,
                    conley_cutoff_km=conley_cutoff_km,
                    conley_metric=conley_metric,
                    conley_kernel=conley_kernel,
                    conley_time=conley_time,
                    conley_unit=conley_unit,
                    conley_lag_cutoff=conley_lag_cutoff,
                )

        if return_fitted:
            result = (coefficients, residuals_orig, fitted_orig, vcov_out)
        else:
            result = (coefficients, residuals_orig, vcov_out)

    return result


@overload
def _solve_ols_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: Literal[False] = ...,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    _precomputed_rank_info: Optional[Tuple[int, np.ndarray, np.ndarray]] = ...,
    _skip_rank_check: bool = ...,
    _fastpath: bool = ...,
    _cert_info: Optional[dict] = ...,
    _diagnostics_out: Optional[dict] = ...,
    vcov_type: str = ...,
    conley_coords: Optional[np.ndarray] = ...,
    conley_cutoff_km: Optional[float] = ...,
    conley_metric: ConleyMetric = ...,
    conley_kernel: str = ...,
    conley_time: Optional[np.ndarray] = ...,
    conley_unit: Optional[np.ndarray] = ...,
    conley_lag_cutoff: Optional[int] = ...,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: ...


@overload
def _solve_ols_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: Literal[True],
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    _precomputed_rank_info: Optional[Tuple[int, np.ndarray, np.ndarray]] = ...,
    _skip_rank_check: bool = ...,
    _fastpath: bool = ...,
    _cert_info: Optional[dict] = ...,
    _diagnostics_out: Optional[dict] = ...,
    vcov_type: str = ...,
    conley_coords: Optional[np.ndarray] = ...,
    conley_cutoff_km: Optional[float] = ...,
    conley_metric: ConleyMetric = ...,
    conley_kernel: str = ...,
    conley_time: Optional[np.ndarray] = ...,
    conley_unit: Optional[np.ndarray] = ...,
    conley_lag_cutoff: Optional[int] = ...,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]: ...


@overload
def _solve_ols_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: bool,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    _precomputed_rank_info: Optional[Tuple[int, np.ndarray, np.ndarray]] = ...,
    _skip_rank_check: bool = ...,
    _fastpath: bool = ...,
    _cert_info: Optional[dict] = ...,
    _diagnostics_out: Optional[dict] = ...,
    vcov_type: str = ...,
    conley_coords: Optional[np.ndarray] = ...,
    conley_cutoff_km: Optional[float] = ...,
    conley_metric: ConleyMetric = ...,
    conley_kernel: str = ...,
    conley_time: Optional[np.ndarray] = ...,
    conley_unit: Optional[np.ndarray] = ...,
    conley_lag_cutoff: Optional[int] = ...,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
]: ...


def _solve_ols_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
    rank_deficient_action: str = "warn",
    column_names: Optional[List[str]] = None,
    _precomputed_rank_info: Optional[Tuple[int, np.ndarray, np.ndarray]] = None,
    _skip_rank_check: bool = False,
    _fastpath: bool = False,
    _cert_info: Optional[dict] = None,
    _diagnostics_out: Optional[dict] = None,
    vcov_type: str = "hc1",
    conley_coords: Optional[np.ndarray] = None,
    conley_cutoff_km: Optional[float] = None,
    conley_metric: ConleyMetric = "haversine",
    conley_kernel: str = "bartlett",
    conley_time: Optional[np.ndarray] = None,
    conley_unit: Optional[np.ndarray] = None,
    conley_lag_cutoff: Optional[int] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
]:
    """
    NumPy/SciPy implementation of solve_ols with R-style rank deficiency handling.

    Detects rank-deficient matrices using pivoted QR decomposition and handles
    them following R's lm() approach: drop redundant columns, set NA (NaN) for
    their coefficients, and compute valid SEs for identified coefficients only.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    y : np.ndarray
        Response vector of shape (n,).
    cluster_ids : np.ndarray, optional
        Cluster identifiers for cluster-robust SEs.
    return_vcov : bool
        Whether to compute variance-covariance matrix.
    return_fitted : bool
        Whether to return fitted values.
    rank_deficient_action : str
        How to handle rank deficiency: "warn", "error", or "silent".
    column_names : list of str, optional
        Names for the columns (used in warning/error messages).
    _precomputed_rank_info : tuple, optional
        Pre-computed (rank, dropped_cols, pivot) from _detect_rank_deficiency.
        Used internally to avoid redundant computation when called from solve_ols.
    _skip_rank_check : bool, default False
        If True, skip rank detection entirely and assume full rank.
        Used when caller has already determined matrix is full rank.
    _fastpath : bool, default False
        Opt-in normal-equations Cholesky fast path (resolved from
        DIFF_DIFF_SOLVE_OLS_FASTPATH by solve_ols). Only the full-rank
        branch is affected; a certification decline falls back verbatim
        to the gelsd solve.
    _cert_info : dict, optional
        Stage-0 certification artifacts from _detect_rank_deficiency
        (Gram reuse). None on the _skip_rank_check route — the fast path
        then self-builds and self-certifies its Gram.
    _diagnostics_out : dict, optional
        solve_ols's diagnostics sink; records which solve branch ran.

    Returns
    -------
    coefficients : np.ndarray
        OLS coefficients of shape (k,). NaN for dropped columns.
    residuals : np.ndarray
        Residuals of shape (n,).
    fitted : np.ndarray, optional
        Fitted values if return_fitted=True.
    vcov : np.ndarray, optional
        Variance-covariance matrix if return_vcov=True. NaN for dropped rows/cols.
    """
    n, k = X.shape

    # Determine rank deficiency status
    if _skip_rank_check:
        # Caller guarantees full rank - skip expensive QR decomposition
        is_rank_deficient = False
        dropped_cols = np.array([], dtype=int)
    elif _precomputed_rank_info is not None:
        # Use pre-computed rank info
        rank, dropped_cols, pivot = _precomputed_rank_info
        is_rank_deficient = len(dropped_cols) > 0
    else:
        # Compute rank via pivoted QR
        rank, dropped_cols, pivot = _detect_rank_deficiency(X)
        is_rank_deficient = len(dropped_cols) > 0

    if is_rank_deficient:
        # Format dropped column information for messages
        dropped_str = _format_dropped_columns(dropped_cols, column_names)

        if rank_deficient_action == "error":
            raise ValueError(
                f"Design matrix is rank-deficient. {k - rank} of {k} columns are "
                f"linearly dependent and cannot be uniquely estimated: {dropped_str}. "
                "This indicates multicollinearity in your model specification."
            )
        elif rank_deficient_action == "warn":
            warnings.warn(
                f"Rank-deficient design matrix: dropping {k - rank} of {k} columns "
                f"({dropped_str}). Coefficients for these columns are set to NA. "
                "This may indicate multicollinearity in your model specification.",
                UserWarning,
                stacklevel=3,  # Point to user code that called solve_ols
            )
        # else: "silent" - no warning

        # Extract kept columns for the reduced solve. dtype=int so an EMPTY
        # comprehension (rank 0, every column dropped) yields an int index array,
        # not the float64 default that raised "arrays used as indices must be of
        # integer (or boolean) type" on X[:, kept_cols].
        kept_cols = np.array([i for i in range(k) if i not in dropped_cols], dtype=int)

        # Rank-0 design: every column dropped (e.g. a constant covariate that
        # collapses to all-zero after FE demeaning). Nothing is identifiable; the
        # warn/error branch above already fired, so return all-NaN cleanly here.
        if kept_cols.size == 0:
            coefficients = np.full(k, np.nan)
            fitted = np.zeros_like(y)
            residuals = y - fitted
            vcov = np.full((k, k), np.nan) if return_vcov else None
            if return_fitted:
                return coefficients, residuals, fitted, vcov
            return coefficients, residuals, vcov

        X_reduced = X[:, kept_cols]

        # Solve the reduced system (now full-rank), equilibrated for scale
        # robustness (see _equilibrated_lstsq).
        coefficients_reduced = _equilibrated_lstsq(X_reduced, y)

        # Expand coefficients to full size with NaN for dropped columns
        coefficients = _expand_coefficients_with_nan(coefficients_reduced, k, kept_cols)

        # Compute residuals using only the identified coefficients
        # Note: Dropped coefficients are NaN, so we use the reduced form
        fitted = np.dot(X_reduced, coefficients_reduced)
        residuals = y - fitted

        # Compute variance-covariance matrix for reduced system, then expand
        vcov = None
        if return_vcov:
            vcov_reduced = _compute_robust_vcov_numpy(
                X_reduced,
                residuals,
                cluster_ids,
                vcov_type=vcov_type,
                conley_coords=conley_coords,
                conley_cutoff_km=conley_cutoff_km,
                conley_metric=conley_metric,
                conley_kernel=conley_kernel,
                conley_time=conley_time,
                conley_unit=conley_unit,
                conley_lag_cutoff=conley_lag_cutoff,
            )
            vcov = _expand_vcov_with_nan(vcov_reduced, k, kept_cols)
    else:
        # Full-rank case: proceed normally. Equilibrate columns before the lstsq
        # so a large-scale column cannot truncate the genuine small-scale
        # direction (cond=1e-07 is relative to the largest singular value).
        #
        # Opt-in fast path (DIFF_DIFF_SOLVE_OLS_FASTPATH): try the certified
        # normal-equations Cholesky solve first; a None return (certification
        # declined) falls through to the verbatim gelsd line below, so the
        # decline case is output-identical to the fast path being off.
        coefficients = None
        gram_bread = None
        if _fastpath:
            fast = _solve_ols_chol_numpy(X, y, cert_info=_cert_info)
            if fast is not None:
                coefficients, gram_bread = fast
                if _diagnostics_out is not None:
                    _diagnostics_out["solve_ols_fastpath"] = "chol_numpy"
        if coefficients is None:
            coefficients = _equilibrated_lstsq(X, y)

        # Compute residuals and fitted values
        fitted = np.dot(X, coefficients)
        residuals = y - fitted

        # Compute variance-covariance matrix if requested
        vcov = None
        if return_vcov:
            vcov = _compute_robust_vcov_numpy(
                X,
                residuals,
                cluster_ids,
                vcov_type=vcov_type,
                conley_coords=conley_coords,
                conley_cutoff_km=conley_cutoff_km,
                conley_metric=conley_metric,
                conley_kernel=conley_kernel,
                conley_time=conley_time,
                conley_unit=conley_unit,
                conley_lag_cutoff=conley_lag_cutoff,
                _bread_matrix=gram_bread,
            )

    if return_fitted:
        return coefficients, residuals, fitted, vcov
    else:
        return coefficients, residuals, vcov


_VALID_VCOV_TYPES = frozenset({"classical", "hc1", "hc2", "hc2_bm", "conley"})


def _validate_vcov_args(
    vcov_type: str,
    cluster_ids: Optional[np.ndarray],
    weights: Optional[np.ndarray],
) -> None:
    """Shared validation for ``vcov_type`` / ``cluster_ids`` / ``weights`` combinations.

    Called from both the public :func:`compute_robust_vcov` and the internal
    :func:`_compute_robust_vcov_numpy` so that any call path reaches the same
    raise. Validation was previously only in the public wrapper, which meant
    direct calls from ``solve_ols`` / ``_solve_ols_numpy`` could silently
    reach an unsupported code path with one-way formulas or drop weights.
    Reviewer P0: prevent that class of silent wrong inference.

    Raises
    ------
    ValueError
        If ``vcov_type`` is not in the allowed set, or if ``cluster_ids`` is
        combined with a ``vcov_type`` that is one-way only (``classical``,
        ``hc2``).
    NotImplementedError
        If ``vcov_type == "conley"`` is combined with ``weights`` (regardless
        of ``weight_type``: weighted Conley is not implemented on the
        generic linalg surface). For ``pweight`` / probability-sampling
        designs the deferral additionally reflects an open methodological
        question — no canonical extension of Conley (1999) exists for
        weighted spatial-HAC under probability sampling. NOT raised here for
        ``hc2_bm + weights``: that weight-type contract is enforced
        downstream in ``_compute_robust_vcov_numpy`` (which has access to
        ``weight_type`` and rejects ``aweight`` / ``fweight`` while routing
        ``pweight`` through the clubSandwich WLS-CR2 port).
    """
    if vcov_type not in _VALID_VCOV_TYPES:
        raise ValueError(
            f"vcov_type must be one of {sorted(_VALID_VCOV_TYPES)}; " f"got {vcov_type!r}"
        )
    if vcov_type in ("classical", "hc2") and cluster_ids is not None:
        msg = {
            "classical": (
                "classical SEs are one-way only; pass vcov_type='hc1' or "
                "'hc2_bm' for cluster-robust."
            ),
            "hc2": (
                "hc2 is one-way only. Use vcov_type='hc2_bm' for " "cluster-robust Bell-McCaffrey."
            ),
        }[vcov_type]
        raise ValueError(msg)
    # Weighted Bell-McCaffrey (both one-way and cluster) is now supported via
    # the clubSandwich WLS-CR2 port. See `_compute_cr2_bm` and
    # `_compute_bm_dof_from_contrasts` for the algorithm: clubSandwich uses an
    # asymmetric weighted hat `H_gg = X_g M_U X_g' W_g` (W, not sqrt(W)) with
    # a W² bias-correction term `S_W = sum_g X_g' W_g² X_g` in the
    # Satterthwaite numerator. Matches `clubSandwich::vcovCR(lm(weights=w),
    # type="CR2") + coef_test(test="Satterthwaite")$df_Satt` at atol=1e-10.
    # Unweighted CR2-BM behavior is unchanged (regression-safe).
    if vcov_type == "conley":
        # Conley + cluster_ids is now supported (combined spatial + cluster
        # product kernel; see ``docs/methodology/REGISTRY.md`` § ConleySpatialHAC
        # → "Combined spatial + cluster product kernel"). Conley + weights
        # remains deferred regardless of weight_type — weighted Conley is
        # not implemented on the generic linalg surface; for probability-
        # sampling weights the deferral additionally reflects an open
        # methodological question with no canonical extension of Conley
        # (1999) for the combination.
        if weights is not None:
            raise NotImplementedError(
                "vcov_type='conley' with weights is not implemented on the "
                "generic linalg surface (any weight_type — pweight, "
                "aweight, or fweight). For probability-sampling weights "
                "(pweight / survey_design), the deferral additionally "
                "reflects an open methodological question: no canonical "
                "extension of Conley (1999) exists for weighted spatial-"
                "HAC under probability sampling. Drop weights for "
                "unweighted Conley (cross-sectional or panel block-"
                "decomposed via conley_lag_cutoff > 0), or use "
                "vcov_type='hc1' for weighted HC1."
            )


def _validate_cluster_ids(cluster_ids: np.ndarray) -> None:
    """Front-door check shared by every clustered vcov backend: missing
    (NaN/None) cluster labels are rejected outright. The pandas groupby
    aggregating cluster scores drops NaN-labelled rows while np.unique /
    factorize-based counts keep (or sentinel) them, so no count can agree
    with the meat's partition — silently wrong CR1 SEs. Matches R fixest /
    Stata, which error on missing cluster values. Called by
    ``compute_robust_vcov``, ``_solve_ols_rust``, and the CR2-BM shared
    core so NumPy, Rust, and CR2 routes all fail closed identically."""
    if pd.isna(np.asarray(cluster_ids)).any():
        raise ValueError(
            "cluster_ids contain missing values (NaN/None). Drop or "
            "impute those rows before requesting cluster-robust SEs."
        )


def effective_cluster_count(cluster_ids: np.ndarray, weights: Optional[np.ndarray] = None) -> int:
    """Effective cluster count for clustered inference metadata.

    Unweighted: the number of unique cluster labels. Weighted: only clusters
    with positive total weight count (zero-weight rows are inert per the
    linalg contract). The positive-total-weight definition applies to ALL
    weight types (pweight/aweight/fweight alike — an all-zero-weight cluster
    contributes nothing to any weighted sandwich), which can be STRICTER
    than the raw-unique count some vcov validators use; consumers that need
    a defined cluster df must fail closed when this count is < 2 (see
    ``get_inference``'s df_convention="cluster" guard). Grouped reduction
    via factorize + bincount, O(n).

    Missing (NaN/None) cluster labels raise ``ValueError`` — the CR1 meat
    aggregation (a pandas groupby) drops NaN-labelled rows, so no count
    can agree with the sandwich's partition; the vcov validation rejects
    them for the same reason (matching R fixest / Stata, which error on
    missing cluster values).
    """
    arr = np.asarray(cluster_ids)
    _validate_cluster_ids(arr)
    codes, uniques = pd.factorize(arr, sort=False)
    if weights is None:
        return int(len(uniques))
    sums = np.bincount(codes, weights=np.asarray(weights, dtype=np.float64))
    return int(np.sum(sums > 0))


def resolve_vcov_type(
    robust: bool = True,
    vcov_type: Optional[str] = None,
) -> str:
    """Resolve the effective ``vcov_type`` from the ``robust``/``vcov_type`` pair.

    Single source of truth for the alias and conflict rules shared by
    :class:`LinearRegression` and :class:`~diff_diff.estimators.DifferenceInDifferences`
    (and any future caller that needs to validate the pair). Keeping the resolution
    in one place prevents ``__init__``/``set_params`` drift.

    Rules (per the Phase 1a plan):

    - If ``vcov_type`` is ``None``: map ``robust=True`` to ``"hc1"`` and
      ``robust=False`` to ``"classical"``.
    - If ``vcov_type`` is supplied: it must be one of the values in the
      module-level ``_VALID_VCOV_TYPES`` set, namely
      ``{"classical", "hc1", "hc2", "hc2_bm", "conley"}``.
    - If ``robust=False`` is supplied together with a non-``"classical"`` ``vcov_type``,
      raise ``ValueError`` - the combination is ambiguous.

    Parameters
    ----------
    robust : bool, default True
        Legacy alias. ``True`` == HC1; ``False`` == classical OLS SEs.
    vcov_type : str, optional
        Explicit variance family. Overrides ``robust`` unless the pair is contradictory.

    Returns
    -------
    str
        One of ``"classical"``, ``"hc1"``, ``"hc2"``, ``"hc2_bm"``, ``"conley"``.

    Raises
    ------
    ValueError
        If ``vcov_type`` is not one of the allowed values, or if
        ``robust=False`` conflicts with an explicit non-classical ``vcov_type``.
    """
    if vcov_type is None:
        return "hc1" if robust else "classical"
    if vcov_type not in _VALID_VCOV_TYPES:
        raise ValueError(
            f"vcov_type must be one of {sorted(_VALID_VCOV_TYPES)}; " f"got {vcov_type!r}"
        )
    if robust is False and vcov_type != "classical":
        raise ValueError(
            f"robust=False conflicts with vcov_type={vcov_type!r}. "
            "Pass vcov_type='classical' for non-robust SEs, or drop "
            "`robust=` and rely on vcov_type alone."
        )
    return vcov_type


# Conley helpers are imported at module top — see the from-import near the
# header of this file.


def compute_robust_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    weight_type: str = "pweight",
    vcov_type: str = "hc1",
    return_dof: bool = False,
    *,
    conley_coords: Optional[np.ndarray] = None,
    conley_cutoff_km: Optional[float] = None,
    conley_metric: ConleyMetric = "haversine",
    conley_kernel: str = "bartlett",
    conley_time: Optional[np.ndarray] = None,
    conley_unit: Optional[np.ndarray] = None,
    conley_lag_cutoff: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute variance-covariance matrix under one of five `vcov_type` variants.

    Uses the sandwich estimator: (X'X)^{-1} * meat * (X'X)^{-1}, with the meat
    matrix determined by the ``vcov_type`` dispatch:

    - ``"classical"``: non-robust OLS SE. ``vcov = sigma_hat^2 * (X'X)^{-1}``
      with ``sigma_hat^2 = sum(u_i^2) / (n - k)``. Useful as a baseline and for
      backward compatibility with ``robust=False``.
    - ``"hc1"`` (default): heteroskedasticity-robust HC1, meat
      ``sum_i (u_i^2) x_i x_i'`` with DOF factor ``n / (n - k)``. With
      ``cluster_ids``, switches to CR1 (Liang-Zeger) cluster-robust.
    - ``"hc2"``: leverage-corrected meat
      ``sum_i (u_i^2 / (1 - h_ii)) x_i x_i'`` where ``h_ii`` are hat-matrix
      diagonals. No DOF adjustment beyond ``n - k``. One-way only; errors with
      ``cluster_ids``.
    - ``"hc2_bm"``: one-way HC2 meat plus Imbens-Kolesar (2016) Bell-McCaffrey
      Satterthwaite degrees of freedom per coefficient when ``cluster_ids`` is
      ``None``. When ``cluster_ids`` is supplied, dispatches to the
      Pustejovsky-Tipton (2018) CR2 Bell-McCaffrey cluster-robust estimator
      (matches R ``clubSandwich::vcovCR(..., type="CR2")``). Required by the
      Pierce-Schott (2016) TWFE application in de Chaisemartin et al. (2026)
      with ``G=103``. **Weighted hc2_bm** (both one-way and clustered) is
      supported for ``weight_type="pweight"`` only via the clubSandwich
      WLS-CR2 port; ``aweight`` and ``fweight`` raise ``NotImplementedError``.
    - ``"conley"``: spatial HAC sandwich (Conley 1999 Eq 4.2). Requires
      ``conley_coords`` (n×2 array) and ``conley_cutoff_km`` (positive
      bandwidth). Two operating modes: cross-sectional (default) and panel
      block-decomposed (pass the three co-required kwargs ``conley_time`` /
      ``conley_unit`` / ``conley_lag_cutoff``, matching R ``conleyreg`` with
      ``lag_cutoff > 0``). Combining with ``cluster_ids`` applies the
      combined spatial + cluster product kernel
      ``K_total[i, j] = K_space(d_ij/h) · 1{c_i = c_j}`` (Wave A #119; on
      the panel path the cluster must be constant within each unit across
      periods). Combining with ``weights`` still raises
      ``NotImplementedError`` regardless of ``weight_type`` (weighted
      Conley is not implemented on the generic linalg surface); for
      probability-sampling weights (``pweight`` / ``survey_design``) the
      deferral additionally reflects an open methodological question, with
      no canonical extension of Conley (1999) for weighted spatial-HAC
      under probability sampling.

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Design matrix.
    residuals : ndarray of shape (n,)
        OLS residuals.
    cluster_ids : ndarray of shape (n,), optional
        Cluster identifiers. Valid with ``vcov_type="hc1"`` (dispatches to CR1)
        and ``vcov_type="hc2_bm"`` (dispatches to CR2 Bell-McCaffrey, including
        the weighted-CR2 clubSandwich port for ``weight_type="pweight"``).
        Combining with ``classical`` or ``hc2`` raises ``ValueError``.
        Combining ``hc2_bm`` with ``weights`` and ``weight_type ∈ {"aweight",
        "fweight"}`` raises ``NotImplementedError`` — the port matches the
        ``pweight`` convention only.
    weights : ndarray of shape (n,), optional
        Observation weights. If provided, computes weighted sandwich estimator.
    weight_type : str, default "pweight"
        Weight type: "pweight", "fweight", or "aweight".
    vcov_type : str, default "hc1"
        One of ``"classical"``, ``"hc1"``, ``"hc2"``, ``"hc2_bm"``,
        ``"conley"`` (see top-level docstring above for the dispatch
        contract).
    conley_coords : ndarray of shape (n, 2), optional, keyword-only
        Required when ``vcov_type="conley"``. Two-column array of
        ``[lat, lon]`` (degrees, for ``conley_metric="haversine"``) or
        projected coordinates (for ``conley_metric="euclidean"`` or a
        callable metric). Raises ``ValueError`` when missing under Conley.
    conley_cutoff_km : float, optional, keyword-only
        Required when ``vcov_type="conley"``. Positive finite bandwidth in
        km (haversine) or coord units (euclidean / callable). No default
        per Conley 1999 Section 5's sensitivity-grid recommendation;
        raises ``ValueError`` when missing under Conley.
    conley_metric : str, default "haversine", keyword-only
        Distance metric for Conley. ``"haversine"`` (lat/lon → km, Earth
        radius 6371.01 matching R ``conleyreg``), ``"euclidean"`` (any
        units), or a callable ``f(coords1, coords2) -> n×n``.
    conley_kernel : str, default "bartlett", keyword-only
        Conley kernel on pairwise distance ``d_ij/h``. ``"bartlett"`` is
        the radial 1-D specialization (matching R ``conleyreg``);
        ``"uniform"`` is the truncated indicator. Both kernels emit a
        ``UserWarning`` if the resulting meat is materially indefinite —
        neither is formally PSD-guaranteed in the radial form (see
        ``docs/methodology/REGISTRY.md`` § ConleySpatialHAC for details).
    return_dof : bool, default False
        When True, returns ``(vcov, dof_vec)`` tuple. ``dof_vec`` is a length-k
        array of per-coefficient degrees of freedom. For ``classical``,
        ``hc1``, ``hc2``: every element is ``n_eff - k``. For ``hc2_bm``
        one-way: Imbens-Kolesar (2016) Satterthwaite DOF per contrast.

    Returns
    -------
    vcov : ndarray of shape (k, k)
        Variance-covariance matrix.
    dof_vec : ndarray of shape (k,), optional
        Only returned when ``return_dof=True``.

    Notes
    -----
    For HC1 (no clustering):
        pweight: meat = Σ s_i s_i' where s_i = w_i x_i u_i (w² in meat)
        fweight: meat = X' diag(w u²) X (matches frequency-expanded HC1)
        aweight/unweighted: meat = X' diag(u²) X
        adjustment = n / (n - k)  (fweight uses n_eff = sum(w))

    For cluster-robust (CR1, Liang-Zeger):
        meat = sum_g (X_g' u_g)(X_g' u_g)'
        adjustment = (G / (G-1)) * ((n-1) / (n-k))

    For HC2 one-way (weighted per review MEDIUM #3):
        h_ii = w_i * x_i' * (X'WX)^{-1} * x_i  (unweighted: w_i = 1)
        meat = sum_i (u_i^2 / (1 - h_ii)) x_i x_i'
        Guards against h_ii > 1 - eps with a fall-back to HC1 plus warning.

    For HC2 + Bell-McCaffrey one-way DOF (per Imbens-Kolesar 2016):
        For each coefficient j, let q_j = X (X'X)^{-1} e_j, let M = I - H.
        DOF_j = (sum_i q_j_i^2)^2 / (a_j' (M^2) a_j) where
        a_j(i) = q_j_i^2 / (1 - h_ii) and M^2 denotes elementwise square.

    The cluster-robust CR1 computation is vectorized using pandas groupby.
    """
    _validate_vcov_args(vcov_type, cluster_ids, weights)

    # Validate weights before dispatching to backend
    if weights is not None:
        weights = _validate_weights(weights, weight_type, X.shape[0])

    # Rust HC2 (one-way, unweighted, no DOF): mirrors the NumPy hc2 branch
    # exactly (leverage meat, no n/(n-k) factor). The near-singular
    # hat-diagonal guard stays Python-side: the kernel returns a sentinel
    # error and the documented warn-and-fall-back-to-HC1 fires here,
    # identical to the NumPy branch's behavior. Imported independently
    # (mixed-version safe) — None on a stale extension.
    if (
        HAS_RUST_BACKEND
        and _rust_compute_robust_vcov_hc2 is not None
        and weights is None
        and vcov_type == "hc2"
        and cluster_ids is None
        and not return_dof
    ):
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        residuals_c = np.ascontiguousarray(residuals, dtype=np.float64)
        try:
            return _rust_compute_robust_vcov_hc2(X_c, residuals_c)
        except ValueError as e:
            error_msg = str(e)
            if "Hat-matrix diagonal exceeds 1" in error_msg:
                warnings.warn(
                    f"{error_msg} Falling back to HC1.",
                    UserWarning,
                    stacklevel=2,
                )
                return _compute_robust_vcov_numpy(
                    X,
                    residuals,
                    cluster_ids=None,
                    weights=None,
                    weight_type=weight_type,
                    vcov_type="hc1",
                    return_dof=return_dof,
                )
            if "Matrix inversion failed" in error_msg:
                raise ValueError(
                    "Design matrix is rank-deficient (singular X'X matrix). "
                    "This indicates perfect multicollinearity. Check your fixed effects "
                    "and covariates for linear dependencies."
                ) from e
            if "numerically unstable" in error_msg.lower():
                # Mirror the HC1 dispatch: fall back to the NumPy HC2 branch
                # (which applies its own hat-diagonal guard semantics) rather
                # than hard-erroring where the pre-kernel path would not.
                warnings.warn(
                    f"Rust backend detected numerical instability: {e}. "
                    "Falling back to Python backend for variance computation.",
                    UserWarning,
                    stacklevel=2,
                )
                return _compute_robust_vcov_numpy(
                    X,
                    residuals,
                    cluster_ids=None,
                    weights=None,
                    weight_type=weight_type,
                    vcov_type="hc2",
                    return_dof=return_dof,
                )
            raise

    # Use Rust backend if available AND no weights AND the requested path is
    # the unchanged HC1/CR1 dispatch AND the caller does not need DOF. Any
    # other combination falls through to the NumPy implementation below.
    if HAS_RUST_BACKEND and weights is None and vcov_type == "hc1" and not return_dof:
        X = np.ascontiguousarray(X, dtype=np.float64)
        residuals = np.ascontiguousarray(residuals, dtype=np.float64)

        cluster_ids_int = None
        if cluster_ids is not None:
            _validate_cluster_ids(cluster_ids)
            cluster_ids_int = pd.factorize(cluster_ids)[0].astype(np.int64)

        try:
            return _rust_compute_robust_vcov(X, residuals, cluster_ids_int)
        except ValueError as e:
            # Translate Rust errors to consistent Python error messages or fallback
            error_msg = str(e)
            if "Matrix inversion failed" in error_msg:
                raise ValueError(
                    "Design matrix is rank-deficient (singular X'X matrix). "
                    "This indicates perfect multicollinearity. Check your fixed effects "
                    "and covariates for linear dependencies."
                ) from e
            if "numerically unstable" in error_msg.lower():
                # Fall back to NumPy on numerical instability (with warning)
                warnings.warn(
                    f"Rust backend detected numerical instability: {e}. "
                    "Falling back to Python backend for variance computation.",
                    UserWarning,
                    stacklevel=2,
                )
                return _compute_robust_vcov_numpy(
                    X,
                    residuals,
                    cluster_ids,
                    weights=weights,
                    weight_type=weight_type,
                    vcov_type=vcov_type,
                    return_dof=return_dof,
                    conley_coords=conley_coords,
                    conley_cutoff_km=conley_cutoff_km,
                    conley_metric=conley_metric,
                    conley_kernel=conley_kernel,
                    conley_time=conley_time,
                    conley_unit=conley_unit,
                    conley_lag_cutoff=conley_lag_cutoff,
                )
            raise

    # Fallback to NumPy implementation
    return _compute_robust_vcov_numpy(
        X,
        residuals,
        cluster_ids,
        weights=weights,
        weight_type=weight_type,
        vcov_type=vcov_type,
        return_dof=return_dof,
        conley_coords=conley_coords,
        conley_cutoff_km=conley_cutoff_km,
        conley_metric=conley_metric,
        conley_kernel=conley_kernel,
        conley_time=conley_time,
        conley_unit=conley_unit,
        conley_lag_cutoff=conley_lag_cutoff,
    )


def _compute_hat_diagonals(
    X: np.ndarray,
    bread_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute hat-matrix diagonals ``h_ii`` for HC2 leverage correction.

    For unweighted OLS: ``h_ii = x_i' (X'X)^{-1} x_i``.
    For weighted OLS (``W = diag(w_i)``): the weighted hat matrix is
    ``H = W^{1/2} X (X'WX)^{-1} X' W^{1/2}``, so the diagonals are
    ``h_ii = w_i * x_i' (X'WX)^{-1} x_i``. This is the same convention as
    ``sandwich::vcovHC(..., type="HC2")`` in R and matches the per-observation
    effective leverage under WLS.

    Returns an ``(n,)`` array. Values are clamped to ``[0, 1 - 1e-10]`` to
    guard against numerical `` h_ii > 1`` from near-singular designs.
    """
    # Compute x_i' (X'WX)^{-1} x_i via a single solve rather than per-row.
    # np.linalg.solve(bread, X.T) has shape (k, n); multiplying element-wise by
    # X.T and summing over k gives the per-observation quadratic form.
    try:
        proj = np.linalg.solve(bread_matrix, X.T)
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'X matrix). "
                "This indicates perfect multicollinearity. Check your fixed effects "
                "and covariates for linear dependencies."
            ) from e
        raise
    h_diag = np.einsum("ij,ji->i", X, proj)
    if weights is not None:
        h_diag = weights * h_diag
    # Numerical guard. Do not silently clip values materially exceeding 1 — that
    # indicates a real design pathology; the caller warns and falls back.
    return np.asarray(h_diag, dtype=np.float64)


def _cr2_adjustment_matrix(G_g: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Symmetric matrix square root of ``G_g^{-1}`` via eigendecomposition.

    For the unweighted case, ``G_g = I - H_gg``. For the WLS case (clubSandwich
    convention), ``G_g = I - H_gg - H_gg' + X_g M_U S_W M_U X_g'`` where
    ``H_gg = X_g M_U X_g' W_g`` (asymmetric weighted hat) and
    ``S_W = sum_g X_g' W_g² X_g``. The algebra (symmetric eigendecomp +
    pseudoinverse) is identical in both cases.

    For real symmetric positive-semidefinite ``G_g``, eigendecompose as
    ``U diag(s) U'`` and return ``U diag(s^{-1/2}) U'`` with pseudoinverse
    handling: eigenvalues below ``tol`` are treated as zero (Moore-Penrose).
    Handles singleton clusters, absorbed cluster FEs (``H_gg`` has eigenvalue
    1), and general rank-deficient cluster blocks. Matches the convention of
    R ``clubSandwich::vcovCR(..., type="CR2")``.
    """
    # Ensure symmetric — the bread_inv arithmetic can leave tiny asymmetry.
    sym = 0.5 * (G_g + G_g.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    inv_sqrt = np.where(eigvals > tol, 1.0 / np.sqrt(np.maximum(eigvals, tol)), 0.0)
    return (eigvecs * inv_sqrt) @ eigvecs.T


def _compute_cr2_bm_vcov_and_dof(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    bread_matrix: np.ndarray,
    contrasts: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Shared CR2 Bell-McCaffrey core — build precomputes once, return ``(vcov, dof)``.

    Single source of truth for the CR2 sandwich and the Satterthwaite DOF.
    Both :func:`_compute_cr2_bm` (per-coefficient vcov + DOF) and
    :func:`_compute_cr2_bm_contrast_dof` (DOF-only for arbitrary contrasts) are
    thin wrappers over this function, so the expensive precomputes
    (``bread_inv``; the per-cluster low-rank ``A_g`` factors on the
    unweighted path; ``S_W``, ``MUWTWUM``, and the per-cluster dense ``A_g``
    eigendecompositions on the weighted path) are defined in exactly one
    place.
    Consolidating the two formerly-duplicated precompute blocks lets a caller
    that needs both vcov and contrast DOF (e.g. :class:`MultiPeriodDiD` under
    ``cluster + hc2_bm``) build them once instead of twice.

    Parameters
    ----------
    X : ndarray of shape (n, k)
    cluster_ids : ndarray of shape (n,)
    bread_matrix : ndarray of shape (k, k)
        ``X'WX`` if weighted, ``X'X`` if unweighted.
    contrasts : ndarray of shape (k, m)
        Each column is a contrast vector for the Satterthwaite DOF. The
        per-coefficient case is recovered with ``contrasts=np.eye(k)``.
    residuals : ndarray of shape (n,), optional
        Raw residuals ``y - X beta_hat`` from the (weighted) fit. When ``None``
        the meat / vcov is skipped and ``vcov`` is returned as ``None``
        (DOF-only callers); the per-cluster precomputes and DOF are unaffected.
    weights : ndarray of shape (n,), optional
        Original (un-normalized) weights. ``None`` for unweighted.

    Returns
    -------
    vcov : ndarray of shape (k, k) or None
        ``None`` when ``residuals is None``.
    dof_vec : ndarray of shape (m,)
        Satterthwaite DOF per contrast column.
    """
    n, k = X.shape
    cluster_ids_arr = np.asarray(cluster_ids)
    _validate_cluster_ids(cluster_ids_arr)
    unique_clusters = np.unique(cluster_ids_arr)
    # When weights are provided, enforce subpopulation invariance: zero-weight
    # rows must contribute nothing to the sandwich. The earlier "drop zero-
    # total-weight clusters only" guard handled all-zero clusters but missed
    # mixed-zero clusters (positive total weight, some zero-weight rows
    # inside). In mixed-zero clusters, the zero-weight rows still entered the
    # CR2 adjustment matrices (H_gg, G_g, A_g, bias_term) on the row side,
    # silently changing SE/DOF — contradicting the linalg contract that
    # zero-weight rows are inert. Fix: physically filter to `weights > 0`
    # rows before all per-cluster computations. CI codex flagged this as P0
    # on PR #475 round 2.
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        positive_mask = weights_arr > 0
        if not np.all(positive_mask):
            X = X[positive_mask]
            # DOF-only callers pass `residuals=None`; guard the subscript so the
            # shared filter does not blow up on them (e.g. StackedDiD's weighted
            # contrast-DOF path and the weighted singleton-cluster dispatch).
            if residuals is not None:
                residuals = residuals[positive_mask]
            cluster_ids_arr = cluster_ids_arr[positive_mask]
            weights_arr = weights_arr[positive_mask]
            weights = weights_arr  # Rebind for downstream w_scale/W_norm logic
            n = X.shape[0]
            # bread_matrix is invariant to zero-weight row removal: the caller
            # computes `X.T @ (X * w[:, None])`, and zero-weight rows contribute
            # zero to that sum. So bread_matrix passed in is already equivalent
            # to bread_matrix on the filtered design. No rebuild needed.
            # Recount unique clusters after filtering.
            unique_clusters = np.unique(cluster_ids_arr)
        eff_clusters = np.array(
            [g for g in unique_clusters if float(np.sum(weights_arr[cluster_ids_arr == g])) > 0]
        )
        if len(eff_clusters) < 2:
            raise ValueError(
                f"Need at least 2 clusters with positive total weight for "
                f"cluster-robust SEs, got {len(eff_clusters)} effective "
                f"clusters out of {len(unique_clusters)} unique."
            )
        unique_clusters = eff_clusters
    G = len(unique_clusters)
    if G < 2:
        raise ValueError(f"Need at least 2 clusters for cluster-robust SEs, got {G}")
    if contrasts.ndim != 2 or contrasts.shape[0] != k:
        raise ValueError(f"contrasts must have shape (k={k}, m); got {contrasts.shape}")

    try:
        bread_inv = np.linalg.solve(bread_matrix, np.eye(k))
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'X matrix). "
                "Cannot compute CR2 Bell-McCaffrey variance."
            ) from e
        raise

    # Normalize weights: w_scale = mean(w[w>0]), W_norm = w / w_scale. Following
    # clubSandwich convention, all internal algebra uses M_U = (X' W_norm X)^{-1}
    # = w_scale * bread_inv. For unweighted (w_scale=1), M_U == bread_inv and the
    # algorithm reduces bit-equally to the prior unweighted form.
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        pos = weights_arr > 0
        w_scale = float(np.mean(weights_arr[pos])) if np.any(pos) else 1.0
        W_norm = weights_arr / w_scale
        M_U = w_scale * bread_inv  # = (X' W_norm X)^{-1}
    else:
        w_scale = 1.0
        W_norm = np.ones(n, dtype=np.float64)
        M_U = bread_inv

    # Per-cluster indices
    cluster_idx = {g: np.where(cluster_ids_arr == g)[0] for g in unique_clusters}

    vcov: Optional[np.ndarray]
    if weights is None:
        # LOW-RANK FACTORED A_g (2026-07, evaluation change — algebraically
        # identical, parity ~1e-15): unweighted G_g = I - H_gg with
        # H_gg = X_g M_U X_g' of rank <= k, so with U_g = X_g M_U^{1/2} and
        # U_g' U_g = Q diag(lam) Q', the adjustment operator is
        #
        #   A_g = (I - U_g U_g')^{-1/2}
        #       = I + (U_g Q) diag(gamma) (U_g Q)',
        #   gamma_i = ((1 - lam_i)^{-1/2} - 1) / lam_i          (regular)
        #   gamma_i = -1 / lam_i                                 (1 - lam_i <= tol:
        #                                                        Moore-Penrose zeroing,
        #                                                        e.g. absorbed cluster FE)
        #
        # matching `_cr2_adjustment_matrix`'s pseudoinverse convention (its
        # tol=1e-10 acts on G_g eigenvalues, which on span(U_g) are exactly
        # `1 - lam_i`; the orthogonal complement carries eigenvalue 1 and is
        # untouched by construction). Consumers only ever need A_g applied to
        # skinny matrices (the residual vector for the meat; X_g bread_inv
        # for the DOF omegas), so the dense (n_g, n_g) A_g — previously an
        # O(n_g^3) eigh per cluster, ~85% of CR2-BM runtime at n=100k — is
        # never materialized: per-cluster work is O(n_g k min(n_g, k) +
        # min(n_g, k)^3) — the eigenproblem is solved on the SMALLER Gram
        # side (k x k when n_g > k; the tiny n_g x n_g dense construction
        # when n_g <= k, so small/singleton clusters never regress vs the
        # prior dense path). gamma is evaluated via expm1/log1p, stable as
        # lam -> 0 (limit 1/2).
        # (The unweighted S_W collapse also applies: S_W = X'X = bread_matrix
        # and MUWTWUM = M_U, so the bias-term build is skipped entirely.)
        wM, VM = np.linalg.eigh(0.5 * (M_U + M_U.T))
        M_U_half = (VM * np.sqrt(np.maximum(wM, 0.0))) @ VM.T
        cluster_scores = np.zeros((G, k)) if residuals is not None else None
        A_g_Xbi: Dict[Any, np.ndarray] = {}
        for gi, g in enumerate(unique_clusters):
            idx_g = cluster_idx[g]
            X_g = X[idx_g]
            U_g = X_g @ M_U_half
            B_g = X_g @ bread_inv
            n_g = len(idx_g)
            if n_g <= k:
                # Small cluster (n_g <= k): the smaller Gram side is the
                # n_g x n_g one, so the k x k eigenproblem would REGRESS vs
                # the dense per-cluster construction (e.g. paired designs or
                # singleton-heavy clusterings with many covariates). Build
                # G_g = I - U_g U_g' directly — it is tiny — and reuse the
                # dense pseudoinverse convention verbatim.
                A_g_small = _cr2_adjustment_matrix(np.eye(n_g) - U_g @ U_g.T)
                if residuals is not None:
                    cluster_scores[gi] = X_g.T @ (A_g_small @ residuals[idx_g])
                A_g_Xbi[g] = A_g_small @ B_g
                continue
            lam, Q_g = np.linalg.eigh(U_g.T @ U_g)
            lam = np.maximum(lam, 0.0)
            s_vals = 1.0 - lam
            gamma = np.zeros(k)
            regular = (lam > 0) & (s_vals > 1e-10)
            gamma[regular] = np.expm1(-0.5 * np.log1p(-lam[regular])) / lam[regular]
            pseudo = (lam > 0) & (s_vals <= 1e-10)
            gamma[pseudo] = -1.0 / lam[pseudo]
            UQ = U_g @ Q_g
            if residuals is not None:
                u_g = residuals[idx_g]
                # s_g = X_g' A_g u_g = X_g'u_g + (X_g'UQ) diag(gamma) (UQ'u_g)
                cluster_scores[gi] = X_g.T @ u_g + (X_g.T @ UQ) @ (gamma * (UQ.T @ u_g))
            A_g_Xbi[g] = B_g + UQ @ (gamma[:, np.newaxis] * (UQ.T @ B_g))
        if residuals is not None:
            meat = cluster_scores.T @ cluster_scores
            vcov = M_U @ meat @ M_U
        else:
            vcov = None
        dof_vec = _cr2_bm_dof_inner(X, A_g_Xbi, cluster_idx, M_U, contrasts)
        return vcov, dof_vec

    # --- WEIGHTED PATH (clubSandwich WLS-CR2; dense per-cluster A_g) ---
    # S_W = sum_g X_g' diag(W_norm_g²) X_g (used for both vcov and DOF).
    S_W = np.zeros((k, k))
    for g in unique_clusters:
        idx_g = cluster_idx[g]
        X_g = X[idx_g]
        S_W += X_g.T @ (X_g * (W_norm[idx_g] ** 2)[:, np.newaxis])
    MUWTWUM = M_U @ S_W @ M_U

    # Per-cluster adjustment matrices A_g via G_g.
    A_g_matrices: Dict[Any, np.ndarray] = {}
    for g in unique_clusters:
        idx_g = cluster_idx[g]
        X_g = X[idx_g]
        # Asymmetric weighted hat block.
        H_gg = (X_g @ M_U @ X_g.T) * W_norm[idx_g][np.newaxis, :]
        I_g = np.eye(len(idx_g))
        bias_term = X_g @ MUWTWUM @ X_g.T
        G_g = I_g - H_gg - H_gg.T + bias_term
        A_g_matrices[g] = _cr2_adjustment_matrix(G_g)

    # --- VCOV (meat) --- only when residuals are supplied (DOF-only callers
    # pass residuals=None and skip this).
    # Per-cluster score: s_g = X_g' diag(W_norm_g) A_g u_g.
    if residuals is not None:
        cluster_scores = np.zeros((G, k))
        for gi, g in enumerate(unique_clusters):
            idx_g = cluster_idx[g]
            u_g = residuals[idx_g]
            A_g = A_g_matrices[g]
            adjusted = A_g @ u_g
            cluster_scores[gi] = X[idx_g].T @ (W_norm[idx_g] * adjusted)
        meat = cluster_scores.T @ cluster_scores
        vcov = M_U @ meat @ M_U
    else:
        vcov = None

    # --- Per-contrast Bell-McCaffrey cluster DOF (weighted: the full
    # clubSandwich P_array construction) ---
    dof_vec = _cr2_bm_dof_inner_weighted(
        X,
        A_g_matrices,
        cluster_idx,
        M_U,
        MUWTWUM,
        W_norm,
        contrasts,
        w_scale=w_scale,
    )

    return vcov, dof_vec


def _compute_cr2_bm(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
    bread_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """CR2 Bell-McCaffrey cluster-robust variance with per-coefficient DOF.

    Implements ``clubSandwich::vcovCR(..., type="CR2") + coef_test(test=
    "Satterthwaite")`` for both unweighted and weighted ``lm`` fits. The
    weighted form uses clubSandwich's specific WLS-CR2 algebra (which is
    NOT a textbook PT2018 §3.3 transform-once derivation; see
    docs/methodology/REGISTRY.md for the algorithm details).

    For each cluster ``g`` (with normalized weights ``W_norm = w / mean(w)``):
      - ``H_gg = X_g M_U X_g' W_g`` (asymmetric weighted hat; W not sqrt(W))
      - ``S_W = sum_g X_g' W_g² X_g`` (W² in the bias-correction term)
      - ``G_g = I - H_gg - H_gg' + X_g M_U S_W M_U X_g'``
      - ``A_g = G_g^{-1/2}`` via symmetric eigendecomposition with
        pseudoinverse handling (see :func:`_cr2_adjustment_matrix`).
      - Per-cluster score ``s_g = X_g' W_g A_g u_g`` (u_g = raw residual)

    Unweighted special case (``weights=None``): ``W_norm=1``, ``S_W=X'X``,
    ``M_U @ S_W @ M_U = M_U``, so ``G_g`` collapses to ``I - H_gg`` (the
    symmetric form). The vcov is bit-equal to the prior unweighted behavior
    at machine precision (atol=1e-14 regression-safety); the per-contrast
    DOF uses the scores-based evaluation (algebraically identical, parity
    within floating-point tolerance — see `_cr2_bm_dof_inner`).

    Meat = ``sum_g s_g s_g'``; VCOV = ``M_U meat M_U`` (where ``M_U`` is the
    normalized bread inverse; ``w_scale`` cancels in the final vcov).

    Per-coefficient Satterthwaite DOF: see :func:`_cr2_bm_dof_inner` for the
    unweighted simple formula and the weighted full P_array construction.

    Parameters
    ----------
    X : ndarray of shape (n, k)
    residuals : ndarray of shape (n,)
        Raw residuals ``y - X beta_hat`` from the (weighted) fit.
    cluster_ids : ndarray of shape (n,)
    bread_matrix : ndarray of shape (k, k)
        ``X'WX`` if weighted, ``X'X`` if unweighted.
    weights : ndarray of shape (n,), optional
        Original (un-normalized) weights. ``None`` for unweighted.

    Returns
    -------
    vcov : ndarray of shape (k, k)
    dof_vec : ndarray of shape (k,)
    """
    # Thin wrapper: per-coefficient vcov + DOF is the shared core with
    # `contrasts = I_k`. See :func:`_compute_cr2_bm_vcov_and_dof` for the
    # single-source-of-truth implementation.
    vcov, dof_vec = _compute_cr2_bm_vcov_and_dof(
        X,
        cluster_ids,
        bread_matrix,
        np.eye(X.shape[1]),
        residuals=residuals,
        weights=weights,
    )
    assert vcov is not None  # residuals provided ⇒ vcov computed
    return vcov, dof_vec


# Contrast-chunk byte budget for the scores-based CR2-BM DOF pass: bounds the
# (G, k, chunk) per-cluster product buffer so a batched per-coefficient sweep
# (contrasts=eye(k), i.e. m == k) cannot allocate O(G*k*m) at once on
# full-dummy / absorbed-FE designs with many clusters and coefficients.
# Each contrast's B matrix is computed independently, so chunking never
# reassociates a contrast's OWN sums — but the per-cluster GEMM
# `X_g' omega_g[:, c0:c1]` runs over a width-c slice, and BLAS kernels
# (GEMV vs GEMM, platform-dependent) may accumulate a column differently at
# different widths: chunk-count invariance holds to ~1 ULP (observed exact
# on Accelerate, 1-ULP drift on OpenBLAS/arm + Windows), NOT bit-for-bit.
# Module-level so tests can monkeypatch it to force the multi-chunk path.
_CR2_BM_CONTRAST_CHUNK_BYTES = 64 * 1024 * 1024


def _cr2_bm_dof_inner(
    X: np.ndarray,
    A_g_Xbi: Dict[Any, np.ndarray],
    cluster_idx: Dict[Any, np.ndarray],
    bread_inv: np.ndarray,
    contrasts: np.ndarray,
) -> np.ndarray:
    """Inner DOF loop, parameterized by an arbitrary contrast matrix.

    Computes the CR2 Bell-McCaffrey Satterthwaite DOF for each column of
    ``contrasts`` (shape ``(k, m)``), using the per-cluster precomputed
    ``A_g_Xbi[g] = A_g @ X_g @ bread_inv`` blocks (built by the shared core
    via the low-rank factored ``A_g`` apply — the dense ``A_g`` is never
    materialized), cluster index map ``cluster_idx``, and
    ``bread_inv``. The per-coefficient case is recovered with
    ``contrasts=np.eye(k)``; compound contrasts (e.g., a
    post-period-average ATT) are handled by the same algebra without
    duplication.

    Per-contrast formula (Pustejovsky-Tipton 2018 Section 4 / Appendix A):

      q       = X @ bread_inv @ c                       (length n)
      omega_g = A_g @ X_g @ bread_inv @ c               (length n_g)
      trace_B = sum_i q_i**2
      trace_B2 = sum_{g, h} (omega_g' M_{g, h} omega_h)**2
      DOF(c)  = trace_B**2 / trace_B2

    SCORES-BASED EVALUATION (algebraic identity; the DOF itself is
    Pustejovsky-Tipton 2018's scalar Satterthwaite t-test — the one-row
    case of their HTZ small-sample correction, §3.1): the
    cluster-pair contraction is never evaluated against an explicit
    residual-maker. With ``Omega`` the ``(n, G)`` matrix stacking the
    ``omega_g`` on their (disjoint) cluster supports and
    ``M = I - X bread_inv X'``, the pairwise matrix
    ``B[g, h] = omega_g' M_{g, h} omega_h`` collapses to

      B = Omega' M Omega = diag(||omega_g||^2) - P' bread_inv P,
      P = X' Omega  (k, G; column g is X_g' omega_g)

    so ``trace_B2 = ||B||_F^2`` costs ``O(n k + G^2 k)`` per contrast. Peak
    memory = two ``O(n k)`` score precomputes (``X_bi`` and the per-cluster
    ``A_g_Xbi`` blocks — input-scale, same order as ``X`` itself) plus
    working buffers capped by ``_CR2_BM_CONTRAST_CHUNK_BYTES`` subject to a
    one-contrast lower bound (a single contrast intrinsically needs the
    ``O(n)`` q vector and ``O(G k)`` P_j/PB buffers): the q vectors,
    per-cluster omegas, and product buffers are contrast-chunked with all
    width-scaled buffers counted in the chunk denominator, and the
    ``(G, G)`` pairwise matrix is row-chunked (its Frobenius sum and max
    are row-separable), so none of ``O(n m)``, ``O(G k m)``, or ``O(G^2)``
    is ever held at once (chunk-count invariant to ~1 ULP, BLAS
    kernels may accumulate a GEMM column differently at different slice
    widths) — the previous form
    materialized the dense ``n x n``
    ``M`` and looped cluster pairs at ``O(n^2)`` per contrast, the exact
    large-``n`` blowup the TODO row tracked. The two evaluations are
    algebraically identical; floating-point agreement is ~1e-12 relative
    (different accumulation order), locked by the frozen-oracle parity test.

    Returns
    -------
    dof_vec : ndarray of shape (m,)
        DOF per contrast column. NaN entries indicate degenerate contrasts
        (trace_B2 ≈ 0 — typically high-collinearity nuisance columns).
    """
    m = contrasts.shape[1]
    unique_clusters = list(cluster_idx.keys())
    n_g_clusters = len(unique_clusters)
    # Precompute once: X_bi (the A_g_Xbi blocks arrive precomputed from the
    # shared core's factored A_g apply). For unit-contrast inputs
    # (contrasts=I_k): q[:, j] == X_bi[:, j] == X @ bread_inv @ e_j.
    X_bi = X @ bread_inv  # (n, k)
    # The q vectors (X_bi @ contrasts) and per-cluster omegas
    # (A_g_Xbi[g] @ contrasts) are computed per contrast chunk below, never
    # at full width m, so no O(n*m) array is ever held.

    # Scores-based precomputes (see docstring): per cluster g,
    # normsq[g, j] = ||omega_g^{(j)}||^2 and P_all[g, :, j] = X_g' omega_g^{(j)}.
    k_X = X.shape[1]
    dof_vec = np.empty(m)
    # Retain max|B_{g,h}| per contrast so we can NaN-guard noise-floor
    # degeneracies in a second pass (mirrors `_cr2_bm_dof_inner_weighted`).
    max_abs_B_arr = np.zeros(m)
    # Chunk the contrasts so every width-scaled working buffer stays under
    # _CR2_BM_CONTRAST_CHUNK_BYTES: per unit of contrast width we allocate a
    # Q_chunk column (n), a transient per-cluster omega (largest cluster
    # n_g_max), a P_all slab (G*k), and a normsq row (G) — a full-m sweep
    # would be O((n + G*k)*m). The cap is subject to a one-contrast lower
    # bound: a single contrast intrinsically needs the O(n) q vector and the
    # O(G*k) P_j / PB buffers. Each contrast's B is computed independently;
    # chunk-count invariance holds to ~1 ULP (BLAS width-dependent column
    # accumulation), not bit-for-bit.
    n = X.shape[0]
    n_g_max = max((idx.size for idx in cluster_idx.values()), default=0)
    per_width_bytes = (n + n_g_max + n_g_clusters * k_X + n_g_clusters) * 8
    chunk = max(1, int(_CR2_BM_CONTRAST_CHUNK_BYTES // max(per_width_bytes, 1)))
    for c0 in range(0, m, chunk):
        c1 = min(c0 + chunk, m)
        width = c1 - c0
        contrasts_c = contrasts[:, c0:c1]
        Q_chunk = X_bi @ contrasts_c  # (n, width) — q vectors as columns
        normsq = np.zeros((n_g_clusters, width))
        P_all = np.zeros((n_g_clusters, k_X, width))
        for gi, g in enumerate(unique_clusters):
            om = A_g_Xbi[g] @ contrasts_c  # (n_g, width)
            normsq[gi] = np.einsum("ij,ij->j", om, om)
            P_all[gi] = X[cluster_idx[g]].T @ om
        # Row-chunk the (G, G) pairwise matrix B_j under the same byte cap:
        # a full B_j is O(G^2) per contrast, which can dominate on
        # many-cluster designs (G in the tens of thousands). trace_B2 is a
        # row-separable Frobenius sum and max|B| a row-separable max, so
        # row blocks accumulate both without ever holding all of B_j.
        row_chunk = max(1, int(_CR2_BM_CONTRAST_CHUNK_BYTES // max(n_g_clusters * 8, 1)))
        for jj in range(width):
            j = c0 + jj
            q = Q_chunk[:, jj]
            trace_B = float(np.sum(q * q))
            P_j = P_all[:, :, jj]  # (G, k)
            PB = P_j @ bread_inv  # (G, k)
            trace_B2 = 0.0
            max_abs_B = 0.0
            for r0 in range(0, n_g_clusters, row_chunk):
                r1 = min(r0 + row_chunk, n_g_clusters)
                B_rows = -(PB[r0:r1] @ P_j.T)  # (rows, G)
                B_rows[np.arange(r0, r1) - r0, np.arange(r0, r1)] += normsq[r0:r1, jj]
                trace_B2 += float(np.sum(B_rows * B_rows))
                if B_rows.size:
                    max_abs_B = max(max_abs_B, float(np.max(np.abs(B_rows))))
            max_abs_B_arr[j] = max_abs_B
            dof_vec[j] = (trace_B * trace_B) / trace_B2 if trace_B2 > 0 else np.nan

    # Noise-floor NaN-guard (unweighted analogue of the guard in
    # `_cr2_bm_dof_inner_weighted`). For a high-leverage FE-dummy / collinear
    # nuisance column, `trace_B2 = sum_{g,h} B_{g,h}²` collapses to float64
    # accumulation noise while `trace_B` stays O(1), so `(trace_B)²/trace_B2`
    # blows up to a non-physical DOF (observed up to ~1e61 on the absorbed-FE
    # golden). `trace_B2 > 0` alone does not catch this because the roundoff is
    # positive. The Satterthwaite DOF is scale-invariant, so a contrast whose
    # `max|B_{g,h}|` sits at the accumulation floor is unreliable and
    # BLAS-order-dependent; we return NaN with a warning rather than ship it.
    # Two union-wise criteria:
    #   1. Batch-relative: a contrast's max|B| is 1e-10× below the largest
    #      contrast's, i.e. it sits at the accumulation floor relative to a
    #      well-conditioned column (per-coefficient sweeps, `contrasts=eye(k)`).
    #      `B_{g,h}` scales as ‖c‖² while the Satterthwaite DOF is scale-invariant,
    #      so the comparison is done on `max|B| / ‖c‖²` - otherwise the same
    #      contrast passed at two scales in one batch would spuriously flag the
    #      smaller copy (‖c‖² differs by the scale²). For `contrasts=eye(k)` every
    #      `‖c‖²=1`, so this is a no-op on the per-coefficient path.
    #   2. Absolute (single-contrast safe): max|B| < (EPS·n·k·bread_scale)².
    #      Calibrated for the O(1)-scale contrasts estimators pass (per-coef unit
    #      vectors, the averaging-weight compound contrast).
    # The treatment / event-study / compound-average contrasts that estimators
    # actually consume are well-conditioned and unaffected.
    _EPS = np.finfo(np.float64).eps
    n_obs, k_X = X.shape
    bread_scale = float(np.max(np.abs(bread_inv))) if bread_inv.size else 1.0
    abs_noise_floor = (_EPS * n_obs * k_X * max(bread_scale, 1.0)) ** 2
    abs_degenerate = max_abs_B_arr < abs_noise_floor
    if m > 1:
        contrast_sq_norm = np.einsum("ij,ij->j", contrasts, contrasts)
        contrast_sq_norm = np.where(contrast_sq_norm > 0, contrast_sq_norm, 1.0)
        scaled_max_B = max_abs_B_arr / contrast_sq_norm
        max_scaled_overall = float(np.max(scaled_max_B))
        rel_degenerate = (
            scaled_max_B < 1e-10 * max_scaled_overall
            if max_scaled_overall > 0
            else np.zeros(m, dtype=bool)
        )
    else:
        rel_degenerate = np.zeros(m, dtype=bool)
    at_noise_floor = abs_degenerate | rel_degenerate
    # Physical-bound guard. The Bell-McCaffrey Satterthwaite DOF is
    # `(tr B)²/tr(B²)` with `B` PSD and cluster-structured, so it is bounded by
    # `rank(B) <= G` (the number of clusters) - it can approach G when clusters
    # are small (few obs each), so the bound is G, NOT the residual dof `n-k`
    # (which can be smaller than G and would wrongly flag legitimate near-G
    # DOFs on short panels). A value above G is non-physical. The simple
    # unweighted `(tr B)²/tr(B²)` form is numerically less faithful than
    # clubSandwich's P-array form (used on the weighted path) for high-leverage
    # FE-dummy columns and can return a finite-but-inflated DOF there (observed
    # ~32.7 and ~16.3 vs R's 6 and 3 on the absorbed-FE golden, G=8) that is NOT
    # at the noise floor. Rather than ship an impossible DOF we NaN it; exact
    # clubSandwich reproduction of these non-user-facing nuisance DOFs would
    # require porting the P-array form and is out of scope (estimators consume
    # only the well-conditioned treatment / event-study / compound-average
    # contrasts, which are unaffected).
    n_clusters = len(unique_clusters)
    non_physical = np.isfinite(dof_vec) & (dof_vec > n_clusters + 1e-6)
    degenerate = at_noise_floor | non_physical
    n_degenerate = int(np.sum(degenerate))
    if n_degenerate > 0:
        dof_vec[degenerate] = np.nan
        warnings.warn(
            f"Satterthwaite DOF for {n_degenerate} of {m} contrast(s) is "
            f"unreliable (at the float64 noise floor, or above the cluster-count "
            f"bound G={n_clusters}); reporting NaN. This affects high-leverage "
            f"FE-dummy / collinear nuisance coefficients; the resulting DOF is "
            f"BLAS-order-dependent / non-physical. The coefficient SEs remain "
            f"valid — only the Satterthwaite DOF (and any t-test / CI depending "
            f"on it) is suppressed.",
            UserWarning,
            stacklevel=3,
        )

    return dof_vec


def _cr2_bm_dof_inner_weighted(
    X: np.ndarray,
    A_g_matrices: Dict[Any, np.ndarray],
    cluster_idx: Dict[Any, np.ndarray],
    bread_inv: np.ndarray,
    MUWTWUM: np.ndarray,
    W_norm: np.ndarray,
    contrasts: np.ndarray,
    w_scale: float = 1.0,
) -> np.ndarray:
    """Per-contrast Satterthwaite DOF for WLS-CR2 via clubSandwich's P_array form.

    Implements ``clubSandwich::vcovCR(..., type="CR2") + coef_test(test=
    "Satterthwaite")`` for the weighted case. Source: jepusto/clubSandwich
    `R/get_arrays.R::get_GH` (inverse_var=FALSE branch) +
    `R/coef_test.R::Satterthwaite_df`.

    For each cluster ``g``:
      ``E_g  = X_g' diag(W_norm_g) A_g``                  (k × n_g)
      ``ME_g = bread_inv @ E_g``                          (k × n_g)
      ``MEU_g = ME_g @ X_g``                              (k × k)
      ``MEF_g = ME_g @ diag(W_norm_g) @ X_g``             (k × k)
      ``H1_g = MEU_g @ M_U_ct``,
      ``H2_g = MEF_g @ M_U_ct``,
      ``H3_g = MEU_g @ Omega_ct``
    where ``M_U_ct = chol(bread_inv).T``, ``Omega = MUWTWUM``,
    ``Omega_ct = chol(Omega).T``.

    For each contrast ``c`` (column of ``contrasts``), build a ``(J × J)``
    matrix ``P`` indexed by cluster pairs:
      ``P[g, h] = (c' H3_g)(H3_h' c) - (c' H1_g)(H2_h' c) - (c' H2_g)(H1_h' c)``
      ``P[g, g] += sum_n (c' ME_g[:, n])²``      (diagonal correction)

    ``df_satt(c) = (tr P)² / sum(P²)``.

    Note: the ``w_scale`` normalization cancels in this ratio (verified Step 0).
    All intermediate matrices use ``bread_inv = (X' W_norm X)^{-1}`` and
    ``W_norm = w / mean(w)``.
    """
    X.shape[1]
    m = contrasts.shape[1]
    unique_clusters = list(cluster_idx.keys())
    len(unique_clusters)

    # "Square-root" factors L such that L @ L.T equals the target matrix.
    # Try Cholesky first (matches clubSandwich's `t(chol(...))` exactly when
    # the matrix is PD — important for full-dummy FE designs where eigh-based
    # symmetric square roots disagree with chol on off-diagonal H-array terms
    # by enough to materially shift Satterthwaite DOF on high-leverage
    # coefficients). Fall back to a symmetric-eigendecomposition pseudo-
    # square-root only on rank-deficient designs (where chol raises), so the
    # MultiPeriodDiD-style full-dummy designs that exposed the original
    # singular-bread case still proceed.
    _TOL = 1e-10

    def _factor_psd(M):
        sym = 0.5 * (M + M.T)
        try:
            return np.linalg.cholesky(sym)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(sym)
            sqrt_eig = np.where(eigvals > _TOL, np.sqrt(np.maximum(eigvals, _TOL)), 0.0)
            return eigvecs * sqrt_eig[np.newaxis, :]

    M_U_ct = _factor_psd(bread_inv)
    Omega_ct = _factor_psd(MUWTWUM)

    # Per-cluster (k × k) H-array slices and (k × n_g) G slices.
    # Use clubSandwich's exact operation ordering: ME_g = M @ E_g where M is
    # the UN-normalized bread inverse `(X' W_orig X)^{-1} = bread_inv / w_scale`,
    # NOT `M_U_norm = bread_inv` (the normalized form). w_scale cancels in
    # the final DOF ratio mathematically, but using the wrong M-convention
    # shifts the float64 roundoff floor on high-leverage contrasts (e.g.,
    # full-dummy FE coefficients) enough to produce 15-30% DOF discrepancies
    # vs `clubSandwich::vcovCR + coef_test()$df_Satt`. Match R's exact
    # operation ordering to reproduce R's roundoff structure.
    M_unnorm = bread_inv / w_scale  # R's "M" = (X' W_orig X)^{-1}
    H1_list: List[np.ndarray] = []
    H2_list: List[np.ndarray] = []
    H3_list: List[np.ndarray] = []
    G_list: List[np.ndarray] = []
    for g in unique_clusters:
        idx_g = cluster_idx[g]
        X_g = X[idx_g]
        W_g_diag = W_norm[idx_g]  # length n_g
        A_g = A_g_matrices[g]

        # E_g = X_g.T @ diag(W_g) @ A_g  (k × n_g)
        E_g = (X_g.T * W_g_diag[np.newaxis, :]) @ A_g
        # R's ME_g uses M = bread_inv_unnormalized = bread_inv / w_scale.
        ME_g = M_unnorm @ E_g  # (k × n_g)

        MEU_g = ME_g @ X_g  # (k × k)
        MEF_g = ME_g @ (W_g_diag[:, np.newaxis] * X_g)  # (k × k)

        H1_list.append(MEU_g @ M_U_ct)
        H2_list.append(MEF_g @ M_U_ct)
        H3_list.append(MEU_g @ Omega_ct)
        G_list.append(ME_g)  # (k × n_g)

    # For each contrast c, build P (J × J) and compute df_satt.
    # Also retain (tr P, ||P||²_F, max(|P|)) per contrast so we can detect
    # noise-floor degeneracies and NaN-guard them in a second pass.
    dof_vec = np.empty(m)
    tr_P_arr = np.empty(m)
    sum_P_sq_arr = np.empty(m)
    max_abs_P_arr = np.empty(m)
    for j in range(m):
        c = contrasts[:, j]  # (k,)
        # Precompute c-projections: (J,)-indexed length-k vectors
        c_H1 = np.array([c @ H1 for H1 in H1_list])  # (J, k)
        c_H2 = np.array([c @ H2 for H2 in H2_list])  # (J, k)
        c_H3 = np.array([c @ H3 for H3 in H3_list])  # (J, k)

        # P[g, h] = c_H3[g] · c_H3[h] - c_H1[g] · c_H2[h] - c_H2[g] · c_H1[h]
        P = c_H3 @ c_H3.T - c_H1 @ c_H2.T - c_H2 @ c_H1.T  # (J, J)

        # Diagonal correction: P[g, g] += sum_n (c' ME_g[:, n])²
        for gi, G_g in enumerate(G_list):
            c_ME_g = c @ G_g  # (n_g,)
            P[gi, gi] += float(c_ME_g @ c_ME_g)

        tr_P = float(np.trace(P))
        sum_P_sq = float(np.sum(P * P))
        max_abs_P = float(np.max(np.abs(P))) if P.size else 0.0
        tr_P_arr[j] = tr_P
        sum_P_sq_arr[j] = sum_P_sq
        max_abs_P_arr[j] = max_abs_P
        dof_vec[j] = (tr_P * tr_P) / sum_P_sq if sum_P_sq > 0 else np.nan

    # Noise-floor NaN-guard. The Satterthwaite DOF formula `(tr P)² / sum(P²)`
    # is scale-invariant, so if a coefficient's P matrix is dominated by
    # float64 accumulation noise (`max(|P|)` is many orders of magnitude
    # smaller than the largest contrast's `max(|P|)`), the DOF computed from
    # noise/noise is unreliable and varies across BLAS implementations by
    # 15-30%. Detection: a contrast's max(|P|) below `1e-10 ×` the largest
    # contrast's max(|P|) signals the noise regime. R's clubSandwich gives
    # different specific values in this regime due to its own BLAS reduction
    # order; we return NaN with a warning rather than ship arbitrarily-
    # different small-sample DOF on the user-facing surface.
    # The noise-floor detector has two criteria, applied union-wise:
    #   1. Batch-relative: a contrast's max(|P|) < 1e-10 × the largest contrast's
    #      max(|P|). Useful when computing per-coefficient DOF for an entire design
    #      (`contrasts=np.eye(k)`) — a single noise-floor coefficient stands out.
    #   2. Absolute (single-contrast safe): max(|P|) < eps_floor based on the
    #      bread matrix scale. Necessary because batch-relative reduces to
    #      max|P| < 1e-10 × max|P| (i.e. False) for a single contrast, leaving
    #      direct `_compute_cr2_bm_contrast_dof(...)` callers (e.g., MPD avg_att)
    #      unprotected. Threshold: `(EPS × n × k × bread_inv_scale)²` covers the
    #      worst-case dgemm accumulation roundoff floor for `H1/H2/H3 @ contrast`
    #      products. CI codex flagged this as P2 (R8 round); regression test in
    #      tests/test_methodology_wls_cr2.py::TestWLSCR2FEDoFNoiseGuard.
    _EPS = np.finfo(np.float64).eps
    n_obs, k_X = X.shape
    bread_inv_scale = float(np.max(np.abs(bread_inv))) if bread_inv.size else 1.0
    abs_noise_floor = (_EPS * n_obs * k_X * max(bread_inv_scale, 1.0)) ** 2

    if m > 1:
        max_P_overall = float(np.max(max_abs_P_arr))
        relative_floor = 1e-10 * max_P_overall if max_P_overall > 0 else 0.0
    else:
        relative_floor = 0.0
    noise_floor = max(relative_floor, abs_noise_floor)
    degenerate = max_abs_P_arr < noise_floor
    n_degenerate = int(np.sum(degenerate))
    if n_degenerate > 0:
        dof_vec[degenerate] = np.nan
        warnings.warn(
            f"Satterthwaite DOF for {n_degenerate} of {m} contrast(s) "
            f"is at the float64 noise floor (max|P| < noise_floor = "
            f"max({relative_floor:.3e}, {abs_noise_floor:.3e})); reporting "
            f"NaN. This typically affects high-leverage FE-dummy "
            f"coefficients whose contrast vector projects to near-zero on "
            f"the design — the resulting DOF varies across BLAS "
            f"implementations and is unreliable. The coefficient SEs "
            f"remain valid; only the Satterthwaite DOF (and any t-test or "
            f"CI that depends on it) is suppressed.",
            UserWarning,
            stacklevel=3,
        )

    return dof_vec


def _compute_cr2_bm_contrast_dof(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    bread_matrix: np.ndarray,
    contrasts: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-contrast CR2 Bell-McCaffrey Satterthwaite DOF.

    Generalizes the per-coefficient DOF from :func:`_compute_cr2_bm` to
    arbitrary linear combinations ``c = sum_j a_j * beta_j``. Used by
    :class:`MultiPeriodDiD` to compute the Satterthwaite DOF for the
    post-period-average ATT contrast under cluster-robust CR2 inference.

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Design matrix (post-rank-deficient-column-drop if applicable).
    cluster_ids : ndarray of shape (n,)
        Per-observation cluster identifiers. NOT subscripted by any
        column mask — cluster IDs are unchanged by column drops.
    bread_matrix : ndarray of shape (k, k)
        ``X.T @ X`` (unweighted) or ``X.T @ W @ X`` (weighted).
    contrasts : ndarray of shape (k, m)
        Each column is a contrast vector ``c`` for the linear combination
        ``c' beta``. The per-coefficient case is recovered with
        ``contrasts=np.eye(k)``.
    weights : ndarray of shape (n,), optional
        Original (un-normalized) weights. ``None`` for unweighted; routes
        through the simple ``(tr B)² / tr(B²)`` formula (algebraically
        identical to the prior evaluation, parity within floating-point
        tolerance). When provided, routes through the clubSandwich WLS-CR2
        P_array form.

    Returns
    -------
    dof_vec : ndarray of shape (m,)
        Satterthwaite DOF per contrast.

    See Also
    --------
    _compute_cr2_bm : per-coefficient DOF (calls this helper internally
        with ``contrasts=np.eye(k)``).
    """
    # Thin wrapper: DOF-only is the shared core with `residuals=None` (the meat
    # / vcov is skipped). See :func:`_compute_cr2_bm_vcov_and_dof` for the
    # single-source-of-truth implementation of the per-cluster precomputes.
    _, dof_vec = _compute_cr2_bm_vcov_and_dof(
        X,
        cluster_ids,
        bread_matrix,
        contrasts,
        residuals=None,
        weights=weights,
    )
    return dof_vec


def _compute_bm_dof_from_contrasts(
    X: np.ndarray,
    bread_matrix: np.ndarray,
    h_diag: np.ndarray,
    contrasts: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-contrast Bell-McCaffrey (Imbens-Kolesar 2016) Satterthwaite DOF.

    Two code paths depending on ``weights``:

    **Unweighted** (``weights is None``): uses the simple Pustejovsky-Tipton
    (2018) Theorem 1 form

        DOF(c) = (sum_i q(i)^2)^2 / sum_{i, k} a(i) a(k) M_{ik}^2

    where ``q = X (X'X)^{-1} c``, ``M = I - H``, ``a(i) = q(i)^2 / (1 - h_ii)``.
    Using the idempotent identity ``M^2 = M``, ``trace(B) = sum_i q(i)^2``
    matches the numerator. The denominator ``a'(M∘M)a`` is evaluated via the
    Schur-product expansion (see the inline derivation) at ``O(n k^2 + k^3)``
    per contrast with NO dense ``n×n`` residual-maker — the prior form's
    ``O(n^2 k)`` hat build limited it to ``n < 10_000``. A noise-floor
    cancellation guard NaNs extreme-leverage contrasts whose expanded
    denominator collapses below float precision (mirrors the clustered
    scores path's guard).

    **Weighted** (``weights is not None``): dispatches to the clubSandwich
    singleton-cluster CR2 reduction (each observation is its own cluster)
    via :func:`_compute_cr2_bm_contrast_dof`. The simple formula above is
    only correct in the unweighted case — empirically it diverges from
    ``clubSandwich::vcovCR(cluster=1:n, type="CR2") + coef_test(test=
    "Satterthwaite")$df_Satt`` by ~6% on heteroskedastic weights. The
    P_array form matches clubSandwich at ``atol=1e-10`` (see the WLS-CR2
    section in ``docs/methodology/REGISTRY.md``). Only ``weight_type=
    "pweight"`` is supported; ``aweight`` / ``fweight`` are rejected by
    ``_compute_robust_vcov_numpy`` upstream of this helper.

    Parameters
    ----------
    X : ndarray of shape (n, k)
    bread_matrix : ndarray of shape (k, k) == (X'WX) or (X'X)
    h_diag : ndarray of shape (n,), hat-matrix diagonals (already weighted).
        Used only on the unweighted path; the weighted dispatch builds its
        own per-cluster `H_gg` blocks via :func:`_compute_cr2_bm`.
    contrasts : ndarray of shape (k, m). Pass ``np.eye(k)`` for per-coefficient DOF.
    weights : optional weights (shape ``(n,)``). When ``None``, uses the
        simple ``(tr B)^2 / tr(B^2)`` formula. When provided, dispatches to
        the clubSandwich singleton-cluster CR2 P_array form.

    Returns
    -------
    ndarray of shape (m,) of Satterthwaite DOF per contrast column. NaN when
    the denominator is non-positive or at/below the cancellation noise
    floor (degenerate / extreme-leverage case; see the inline guard note).
    """
    n, k = X.shape
    if contrasts.ndim != 2 or contrasts.shape[0] != k:
        raise ValueError(f"contrasts must have shape (k={k}, m); got {contrasts.shape}")
    if weights is not None:
        # Weighted one-way HC2-BM uses clubSandwich's singleton-cluster CR2
        # reduction (each obs is its own cluster). The simple (tr B)² / tr(B²)
        # formula is only correct in the unweighted case — empirically it
        # diverges from `clubSandwich::vcovCR(cluster=1:n, type="CR2") +
        # coef_test(test="Satterthwaite")$df_Satt` by ~6% on heteroskedastic
        # weights. The P_array form (via _cr2_bm_dof_inner_weighted) matches
        # at atol=1e-10.
        cluster_ids_singleton = np.arange(n)
        return _compute_cr2_bm_contrast_dof(
            X, cluster_ids_singleton, bread_matrix, contrasts, weights=weights
        )

    # Unweighted: keep the simple (tr B)² / tr(B²) formula — algebraically
    # identical backward compatibility with prior unweighted Bell-McCaffrey
    # output (dense prior evaluation; floating-point-tolerance parity).
    try:
        bread_inv_c = np.linalg.solve(bread_matrix, contrasts)
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'X matrix). "
                "Cannot compute Bell-McCaffrey DOF."
            ) from e
        raise
    # q has shape (n, m); column j is X @ (bread_inv @ contrasts[:, j]).
    q = X @ bread_inv_c
    one_minus_h = np.maximum(1.0 - h_diag, 1e-10)
    one_minus_2h = 1.0 - 2.0 * h_diag
    m = contrasts.shape[1]
    dof = np.empty(m)
    for j in range(m):
        qj_sq = q[:, j] * q[:, j]
        num = qj_sq.sum() ** 2
        a_j = qj_sq / one_minus_h
        # SCORES-BASED EVALUATION (2026-07, evaluation change — exact
        # algebra): den = a'(M∘M)a with M = I − XBX' expands via the Schur
        # product as
        #
        #   den = sum_i a_i^2 (1 − 2 h_ii) + tr((B S_a)^2),
        #   S_a = X' diag(a) X   (k x k),
        #
        # because (M∘M)_{il} = δ_{il}(1 − 2 h_ii) + (x_i'B x_l)^2 and
        # sum_{i,l} a_i a_l (x_i'B x_l)^2 = tr(B S_a B S_a). O(n k^2 + k^3)
        # per contrast, never materializing the dense n×n residual-maker
        # (the prior form's O(n^2 k) hat build + O(n^2) per contrast).
        term_diag = float(np.sum(a_j * a_j * one_minus_2h))
        S_a = X.T @ (X * a_j[:, np.newaxis])
        BS = np.linalg.solve(bread_matrix, S_a)
        term_tr = float(np.sum(BS * BS.T))
        den = term_diag + term_tr
        # Cancellation guard: the dense den = a'(M∘M)a is >= 0 exactly
        # (M∘M is PSD — Schur product of the PSD M with itself), but the
        # expanded difference of same-magnitude terms can collapse to the
        # float64 noise floor for extreme-leverage contrasts. A den at or
        # below the floor would inflate the DOF arbitrarily (the same
        # failure mode the clustered scores path NaN-guards); report NaN
        # instead of a non-physical DOF. The prior dense path's `den > 0`
        # kept such noise-floor denominators; on well-conditioned contrasts
        # the two evaluations agree to ~1e-12 (frozen-oracle parity test).
        noise_floor = np.finfo(float).eps * (abs(term_diag) + abs(term_tr)) * n
        dof[j] = num / den if den > noise_floor else np.nan
    return dof


def _compute_bm_dof_oneway(
    X: np.ndarray,
    bread_matrix: np.ndarray,
    h_diag: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-coefficient Bell-McCaffrey DOF vector (Imbens-Kolesar 2016).

    Thin wrapper over :func:`_compute_bm_dof_from_contrasts` with
    ``contrasts = I_k``, so each column picks out one coefficient.
    """
    k = X.shape[1]
    return _compute_bm_dof_from_contrasts(X, bread_matrix, h_diag, np.eye(k), weights=weights)


def _compute_robust_vcov_numpy(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    weight_type: str = "pweight",
    vcov_type: str = "hc1",
    return_dof: bool = False,
    *,
    conley_coords: Optional[np.ndarray] = None,
    conley_cutoff_km: Optional[float] = None,
    conley_metric: ConleyMetric = "haversine",
    conley_kernel: str = "bartlett",
    conley_time: Optional[np.ndarray] = None,
    conley_unit: Optional[np.ndarray] = None,
    conley_lag_cutoff: Optional[int] = None,
    _bread_matrix: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    NumPy fallback implementation of compute_robust_vcov.

    See :func:`compute_robust_vcov` for parameter and return semantics.

    ``_bread_matrix`` is a private reuse seam for the opt-in solve_ols
    Cholesky fast path: the caller passes its already-built ``X.T @ X``
    (the exact same expression on the same array, so the bread is
    byte-identical to building it here). Honored only for unweighted calls;
    every vcov formula downstream is unchanged.
    """
    # Re-run the shared validation here too. The public wrapper validates
    # before dispatch, but solve_ols / _solve_ols_numpy call this function
    # directly and previously bypassed the raise, letting unsupported
    # combinations (cluster + classical, cluster + hc2, cluster + weights +
    # hc2_bm) silently produce wrong inference. Reviewer P0 fix.
    _validate_vcov_args(vcov_type, cluster_ids, weights)

    n, k = X.shape

    # Bread: (X'WX) or (X'X) depending on whether weights present.
    # Suppress spurious BLAS-level subnormal warnings on macOS Accelerate
    # for sparse-X designs (e.g., MultiPeriodDiD's event-study dummies).
    # Non-finite bread is caught at the downstream np.linalg.solve.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if weights is not None:
            XtWX = X.T @ (X * weights[:, np.newaxis])
            bread_matrix = XtWX
        elif _bread_matrix is not None:
            bread_matrix = _bread_matrix
        else:
            bread_matrix = X.T @ X

    # Effective n for df computation
    # fweights: sum(w) (frequency expansion)
    # pweight/aweight with zeros: positive-weight count (zero-weight rows
    # contribute nothing to the sandwich and should not inflate df)
    n_eff = n
    if weights is not None:
        if weight_type == "fweight":
            n_eff = int(round(np.sum(weights)))
        elif np.any(weights == 0):
            n_eff = int(np.count_nonzero(weights > 0))

    # ------------------------------------------------------------------
    # Conley (1999) spatial HAC. Cross-sectional (Phase 1) when no time/unit
    # is supplied; panel block-decomposed form (matches R conleyreg with
    # lag_cutoff > 0) when all three of conley_time / conley_unit /
    # conley_lag_cutoff are supplied. The validator above already raised
    # on conley + weights. ``cluster_ids`` is threaded through when set,
    # combining the spatial kernel with the cluster indicator (combined
    # spatial + cluster product kernel; see REGISTRY.md § ConleySpatialHAC).
    # ------------------------------------------------------------------
    if vcov_type == "conley":
        _validate_conley_kwargs(
            conley_coords,
            conley_cutoff_km,
            conley_metric,
            conley_kernel,
            n,
            time=conley_time,
            unit=conley_unit,
            lag_cutoff=conley_lag_cutoff,
            cluster_ids=cluster_ids,
        )
        vcov = _compute_conley_vcov(
            X,
            residuals,
            np.asarray(conley_coords, dtype=np.float64),
            float(conley_cutoff_km),  # type: ignore[arg-type]
            conley_metric,
            conley_kernel,
            bread_matrix,
            time=conley_time,
            unit=conley_unit,
            lag_cutoff=conley_lag_cutoff,
            cluster_ids=cluster_ids,
        )
        if return_dof:
            return vcov, np.full(k, n_eff - k, dtype=np.float64)
        return vcov

    # ------------------------------------------------------------------
    # Classical (non-robust) OLS SE.
    # ------------------------------------------------------------------
    if vcov_type == "classical":
        # sigma_hat^2 = sum(w * u^2) / (n_eff - k) for pweight/aweight; for
        # fweight, divide by (sum_w - k).
        if weights is not None:
            if weight_type == "fweight":
                sse = float(np.sum(weights * residuals**2))
            elif weight_type == "pweight":
                sse = float(np.sum(weights * residuals**2))
            else:  # aweight
                sse = float(np.sum(weights * residuals**2))
        else:
            sse = float(np.sum(residuals**2))
        sigma2 = sse / (n_eff - k)
        try:
            bread_inv = np.linalg.solve(bread_matrix, np.eye(k))
        except np.linalg.LinAlgError as e:
            if "Singular" in str(e):
                raise ValueError(
                    "Design matrix is rank-deficient (singular X'X matrix). "
                    "This indicates perfect multicollinearity. Check your fixed effects "
                    "and covariates for linear dependencies."
                ) from e
            raise
        vcov = sigma2 * bread_inv
        if return_dof:
            dof_vec = np.full(k, n_eff - k, dtype=np.float64)
            return vcov, dof_vec
        return vcov

    # ------------------------------------------------------------------
    # CR2 Bell-McCaffrey cluster-robust (vcov_type="hc2_bm" + cluster).
    # ------------------------------------------------------------------
    if vcov_type == "hc2_bm" and cluster_ids is not None:
        # The clubSandwich WLS-CR2 port matches the `pweight` (sampling-weight)
        # convention only. clubSandwich's algebra puts `w` directly into the
        # score (s_g = X_g' diag(W) A_g u_g), producing meat = sum w_g² ...
        # — exactly diff-diff's `pweight` convention. `aweight` (analytical /
        # inverse-variance) and `fweight` (frequency-expanded) require
        # different CR2 derivations that are not in this port. Reject loudly.
        if weights is not None and weight_type != "pweight":
            raise NotImplementedError(
                f"vcov_type='hc2_bm' with weight_type={weight_type!r} is not "
                "supported. The clubSandwich WLS-CR2 port matches the "
                "'pweight' (sampling-weight) convention only. For analytical "
                "('aweight') or frequency ('fweight') weights with CR2 "
                "Bell-McCaffrey, the derivation is a separate methodology "
                "task. Use weight_type='pweight' or vcov_type='hc1' "
                "(CR1 supports all three weight types)."
            )
        vcov_cr2, dof_cr2 = _compute_cr2_bm(
            X, residuals, cluster_ids, bread_matrix, weights=weights
        )
        if return_dof:
            return vcov_cr2, dof_cr2
        return vcov_cr2

    # ------------------------------------------------------------------
    # HC2 / HC2+BM one-way (no cluster).
    # ------------------------------------------------------------------
    if vcov_type in ("hc2", "hc2_bm"):
        # cluster path handled above; here cluster_ids is None by construction.
        # **Weighted hc2_bm one-way**: clubSandwich's CR2 with singleton clusters
        # uses the bias-corrected adjustment `A_i = 1 / sqrt(G_i)` where
        # `G_i = 1 - 2*h_ii + X_i' M_U S_W M_U X_i`, NOT the simple HC2
        # leverage `1/sqrt(1-h_ii)`. The two agree unweighted (S_W = X'X, so
        # the bias term collapses to h_ii) but diverge on weighted designs
        # because S_W = sum_j w_j² X_j X_j' ≠ X'WX. To match
        # `clubSandwich::vcovCR(cluster=1:n, type="CR2")`, route weighted
        # hc2_bm through the singleton-cluster CR2 path so vcov and DOF are
        # consistent (both use the clubSandwich algebra).
        if vcov_type == "hc2_bm" and weights is not None:
            # Same `weight_type` gating as the cluster-CR2 branch above:
            # only pweight matches clubSandwich's WLS-CR2 algebra.
            if weight_type != "pweight":
                raise NotImplementedError(
                    f"vcov_type='hc2_bm' with weight_type={weight_type!r} is "
                    "not supported. The clubSandwich WLS-CR2 port matches "
                    "the 'pweight' (sampling-weight) convention only. Use "
                    "weight_type='pweight' or vcov_type='hc1' (CR1 supports "
                    "all three weight types)."
                )
            vcov_cr2, dof_cr2 = _compute_cr2_bm(
                X, residuals, np.arange(n), bread_matrix, weights=weights
            )
            if return_dof:
                return vcov_cr2, dof_cr2
            return vcov_cr2
        h_diag = _compute_hat_diagonals(X, bread_matrix, weights=weights)
        if np.any(h_diag > 1.0 + 1e-6):
            warnings.warn(
                f"Hat-matrix diagonal exceeds 1 (max={h_diag.max():.6f}); "
                "the design is near-singular. Falling back to HC1.",
                UserWarning,
                stacklevel=3,
            )
            return _compute_robust_vcov_numpy(
                X,
                residuals,
                cluster_ids=None,
                weights=weights,
                weight_type=weight_type,
                vcov_type="hc1",
                return_dof=return_dof,
            )
        one_minus_h = np.maximum(1.0 - h_diag, 1e-10)
        # HC2 meat: sum_i (u_i^2 / (1 - h_ii)) x_i x_i', with pweight scaling
        # matching the HC1 convention (w_i * u_i / sqrt(1 - h_ii) as score).
        if weights is not None and weight_type == "fweight":
            factor = weights * (residuals**2) / one_minus_h
            meat = X.T @ (X * factor[:, np.newaxis])
        elif weights is not None and weight_type == "pweight":
            # pweight scores carry w in the score, so meat = sum (w u / sqrt(1-h))^2 x x'
            scaled = weights * residuals / np.sqrt(one_minus_h)
            scores_hc2 = X * scaled[:, np.newaxis]
            meat = scores_hc2.T @ scores_hc2
        else:
            # aweight / unweighted: meat = sum_i (u_i^2 / (1 - h_ii)) x_i x_i'
            factor = (residuals**2) / one_minus_h
            # Zero out zero-weight rows under aweight (subpopulation invariance)
            if weights is not None and np.any(weights == 0):
                factor = factor * (weights > 0)
            meat = X.T @ (X * factor[:, np.newaxis])

        # Sandwich without DOF adjustment for HC2 (matches sandwich::vcovHC
        # type="HC2" convention: no (n/(n-k)) factor).
        try:
            temp = np.linalg.solve(bread_matrix, meat)
            vcov = np.linalg.solve(bread_matrix, temp.T).T
        except np.linalg.LinAlgError as e:
            if "Singular" in str(e):
                raise ValueError(
                    "Design matrix is rank-deficient (singular X'X matrix). "
                    "This indicates perfect multicollinearity. Check your fixed effects "
                    "and covariates for linear dependencies."
                ) from e
            raise

        if not return_dof:
            return vcov
        if vcov_type == "hc2":
            dof_vec = np.full(k, n_eff - k, dtype=np.float64)
        else:  # hc2_bm
            dof_vec = _compute_bm_dof_oneway(X, bread_matrix, h_diag, weights=weights)
        return vcov, dof_vec

    # ------------------------------------------------------------------
    # HC1 / CR1 (original behavior).
    # ------------------------------------------------------------------
    assert vcov_type == "hc1"

    # Cluster-robust validity check FIRST: a cluster-robust request with fewer
    # than 2 clusters is invalid and must raise the documented error — this has
    # to precede the saturated-design guard below so a 1-cluster *saturated* fit
    # still raises rather than being masked by the NaN return. (Mirrors the
    # effective-cluster count computed in the cluster branch, including the
    # zero-total-weight exclusion.)
    if cluster_ids is not None:
        # Normalize to an array first (as the cluster branch does) so the
        # weighted groupby below cannot index-align a pandas Series grouper
        # against the freshly-created Series(weights) and miscount clusters.
        cluster_ids_arr = np.asarray(cluster_ids)
        _validate_cluster_ids(cluster_ids_arr)
        n_clusters_check = len(np.unique(cluster_ids_arr))
        # Zero-total-weight clusters are inert for ALL weight types (a
        # zero-frequency fweight cluster contributes nothing to the sandwich,
        # exactly like a subpopulation-zeroed pweight cluster) — the prior
        # fweight carve-out let a one-effective-cluster fweight fit through
        # to a degenerate CR1 SE.
        if weights is not None and np.any(weights == 0):
            cluster_weight_sums = pd.Series(weights).groupby(cluster_ids_arr).sum()
            n_clusters_check = int((cluster_weight_sums > 0).sum())
        if n_clusters_check < 2:
            raise ValueError(
                f"Need at least 2 clusters for cluster-robust SEs, got {n_clusters_check}"
            )

    # Saturated design (no residual degrees of freedom): both the HC1 adjustment
    # n_eff/(n_eff-k) and the CR1 adjustment (n_eff-1)/(n_eff-k) divide by
    # (n_eff - k), which is zero when the design exactly determines y. Return a
    # NaN vcov so downstream inference is degenerate (NaN) rather than raising
    # ZeroDivisionError — consistent with the library's all-or-nothing NaN
    # convention for undefined inference.
    if n_eff - k <= 0:
        nan_vcov = np.full((k, k), np.nan)
        if return_dof:
            return nan_vcov, np.full(k, np.nan, dtype=np.float64)
        return nan_vcov

    # Compute weighted scores for cluster-robust meat (outer product of sums).
    # pweight/fweight multiply by w; aweight and unweighted use raw residuals.
    _use_weighted_scores = weights is not None and weight_type not in ("aweight",)
    if _use_weighted_scores:
        scores = X * (weights * residuals)[:, np.newaxis]
    else:
        scores = X * residuals[:, np.newaxis]
        # Zero out scores for zero-weight aweight rows (subpopulation invariance)
        if weights is not None and np.any(weights == 0):
            scores[weights == 0] = 0.0

    if cluster_ids is None:
        # HC1 (heteroskedasticity-robust) standard errors
        adjustment = n_eff / (n_eff - k)
        if weights is not None and weight_type == "fweight":
            # fweight: frequency-expanded HC1, meat = Σ w_i x_i x_i' u_i²
            meat = np.dot(X.T, X * (weights * residuals**2)[:, np.newaxis])
        else:
            # pweight: WLS score outer product, meat = Σ w_i² x_i x_i' u_i²
            # aweight/unweighted: meat = Σ x_i x_i' u_i² (scores have no w)
            meat = scores.T @ scores
    else:
        # Cluster-robust standard errors (vectorized via groupby)
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        # Exclude clusters with zero total weight (subpopulation-zeroed
        # pweight/aweight AND zero-frequency fweight alike — inert either way)
        if weights is not None and np.any(weights == 0):
            cluster_weights = pd.Series(weights).groupby(cluster_ids).sum()
            n_clusters = int((cluster_weights > 0).sum())

        if n_clusters < 2:
            raise ValueError(f"Need at least 2 clusters for cluster-robust SEs, got {n_clusters}")

        # Small-sample adjustment
        adjustment = (n_clusters / (n_clusters - 1)) * ((n_eff - 1) / (n_eff - k))

        # Sum scores within each cluster using pandas groupby (vectorized)
        cluster_scores = pd.DataFrame(scores).groupby(cluster_ids).sum().values

        # Meat is the outer product sum: sum_g (score_g)(score_g)'
        meat = cluster_scores.T @ cluster_scores

    # Sandwich estimator: bread^{-1} meat bread^{-1}
    # Solve bread * temp = meat, then solve bread * vcov' = temp'
    try:
        temp = np.linalg.solve(bread_matrix, meat)
        vcov = adjustment * np.linalg.solve(bread_matrix, temp.T).T
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'X matrix). "
                "This indicates perfect multicollinearity. Check your fixed effects "
                "and covariates for linear dependencies."
            ) from e
        raise

    if return_dof:
        dof_vec = np.full(k, n_eff - k, dtype=np.float64)
        return vcov, dof_vec
    return vcov


# Empirical threshold: coefficients above this magnitude suggest near-separation
# in the logistic model (predicted probabilities collapse to 0/1).
_LOGIT_SEPARATION_COEF_THRESHOLD = 10
_LOGIT_SEPARATION_PROB_THRESHOLD = 1e-5
_DEFAULT_EPV_THRESHOLD = 10

# Reciprocal-condition guard for the IRLS normal-equations Cholesky fast path
# (solve_logit): cond(G) <= 1e6 bounds the Cholesky forward error at
# ~eps * cond ~ 2e-10 relative, consistent with the tol-bounded parity budget;
# anything worse falls back to the legacy tall-matrix lstsq for that iteration.
_IRLS_CHOL_RCOND_GUARD = 1e-6


def solve_logit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
    check_separation: bool = True,
    rank_deficient_action: str = "warn",
    weights: Optional[np.ndarray] = None,
    epv_threshold: float = _DEFAULT_EPV_THRESHOLD,
    context_label: str = "",
    diagnostics_out: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit logistic regression via IRLS (Fisher scoring).

    Matches R's ``glm(family=binomial)`` algorithm: iteratively reweighted
    least squares with working weights ``mu*(1-mu)`` and working response
    ``eta + (y-mu)/(mu*(1-mu))``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features). Intercept added automatically.
    y : np.ndarray
        Binary outcome (0/1).
    max_iter : int, default 25
        Maximum IRLS iterations (R's ``glm`` default).
    tol : float, default 1e-8
        Convergence tolerance on coefficient change (R's ``glm`` default).
    check_separation : bool, default True
        Whether to check for near-separation and emit warnings.
    rank_deficient_action : str, default "warn"
        How to handle rank-deficient design matrices:
        - "warn": Emit warning and drop columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently
    weights : np.ndarray, optional
        Survey/observation weights of shape (n_samples,). When provided,
        the IRLS working weights become ``weights * mu * (1 - mu)``
        instead of ``mu * (1 - mu)``. This produces the survey-weighted
        maximum likelihood estimator, matching R's ``svyglm(family=binomial)``.
        When None (default), behavior is identical to unweighted logistic
        regression.
    epv_threshold : float, default 10
        Events Per Variable threshold. When the ratio of minority-class
        observations to predictor variables (excluding intercept) falls
        below this value, a warning is
        emitted (or ValueError raised if ``rank_deficient_action="error"``).
        Based on Peduzzi et al. (1996).
    context_label : str, default ""
        Optional label for warning messages (e.g., "cohort g=4") to help
        users identify which logit estimation triggered the warning.
    diagnostics_out : dict, optional
        If provided, populated with EPV diagnostic info:
        ``{"epv": float, "n_events": int, "k": int, "is_low": bool}``, plus
        ``"irls_chol_fallback_iters"`` - the number of IRLS iterations whose
        normal-equations Cholesky fast path fell back to the legacy lstsq
        solve (0 on well-conditioned fits).

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients (including intercept as element 0).
    probs : np.ndarray
        Predicted probabilities.
    """
    n, p = X.shape
    X_with_intercept = np.column_stack([np.ones(n), X])
    k = p + 1  # number of parameters including intercept

    # Validate weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights.shape}")
        if np.any(np.isnan(weights)):
            raise ValueError("weights contain NaN values")
        if np.any(~np.isfinite(weights)):
            raise ValueError("weights contain Inf values")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if np.sum(weights) <= 0:
            raise ValueError("weights sum to zero — no observations have positive weight")

    # Validate rank_deficient_action
    valid_actions = {"warn", "error", "silent"}
    if rank_deficient_action not in valid_actions:
        raise ValueError(
            f"rank_deficient_action must be one of {valid_actions}, "
            f"got '{rank_deficient_action}'"
        )

    # Track original column count for coefficient expansion at the end
    k_original = X_with_intercept.shape[1]
    eff_dropped_original: list = []  # indices in original column space

    # Validate effective weighted sample when weights have zeros
    if weights is not None and np.any(weights == 0):
        pos_mask = weights > 0
        n_pos = int(np.sum(pos_mask))
        y_pos = y[pos_mask]
        # Need both outcome classes in the positive-weight subset
        unique_y = np.unique(y_pos)
        if len(unique_y) < 2:
            raise ValueError(
                f"Positive-weight observations have only {len(unique_y)} "
                f"outcome class(es). Logistic regression requires both 0 and 1 "
                f"in the effective (positive-weight) sample."
            )
        # Check rank deficiency on positive-weight rows FIRST — full design
        # may be full rank due to zero-weight padding. Drop columns before
        # checking sample-size identification.
        X_eff = X_with_intercept[pos_mask]
        eff_rank_info = _detect_rank_deficiency(X_eff)
        if len(eff_rank_info[1]) > 0:
            n_dropped_eff = len(eff_rank_info[1])
            if rank_deficient_action == "error":
                raise ValueError(
                    f"Effective (positive-weight) sample is rank-deficient: "
                    f"{n_dropped_eff} linearly dependent column(s). "
                    f"Cannot identify logistic model on this subpopulation."
                )
            elif rank_deficient_action == "warn":
                warnings.warn(
                    f"Effective (positive-weight) sample is rank-deficient: "
                    f"dropping {n_dropped_eff} column(s). Propensity estimates "
                    f"may be unreliable on this subpopulation.",
                    UserWarning,
                    stacklevel=2,
                )
            # Drop columns and track original indices for final expansion
            eff_dropped_original = list(eff_rank_info[1])
            X_with_intercept = np.delete(X_with_intercept, eff_rank_info[1], axis=1)
            k = X_with_intercept.shape[1]
        # Check sample-size identification AFTER column dropping
        if n_pos <= k:
            raise ValueError(
                f"Only {n_pos} positive-weight observation(s) for "
                f"{k} parameters (after rank reduction). "
                f"Cannot identify logistic model."
            )

    # Check rank deficiency once before iterating (on possibly-shrunk matrix)
    rank_info = _detect_rank_deficiency(X_with_intercept)
    rank, dropped_cols, _ = rank_info
    if len(dropped_cols) > 0:
        col_desc = _format_dropped_columns(dropped_cols)
        if rank_deficient_action == "error":
            raise ValueError(
                f"Rank-deficient design matrix in logistic regression: "
                f"dropping {col_desc}. Propensity score estimates may be unreliable."
            )
        elif rank_deficient_action == "warn":
            warnings.warn(
                f"Rank-deficient design matrix in logistic regression: "
                f"dropping {col_desc}. Propensity score estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        # dtype=int for consistency with the other rank-deficiency sites. The
        # prepended intercept column guarantees rank >= 1, so kept_cols is never
        # empty here (no rank-0 / empty-index hazard in the logistic path).
        kept_cols = np.array([i for i in range(k) if i not in dropped_cols], dtype=int)
        X_solve = X_with_intercept[:, kept_cols]
    else:
        kept_cols = np.arange(k)
        X_solve = X_with_intercept

    # Events Per Variable (EPV) check — Peduzzi et al. (1996)
    # Use effective (positive-weight) sample when weights have zeros,
    # since zero-weight rows don't contribute to the likelihood.
    k_solve = X_solve.shape[1]
    if weights is not None and np.any(weights == 0):
        y_eff = y[weights > 0]
        n_eff = len(y_eff)
    else:
        y_eff = y
        n_eff = n
    n_pos_y = int(np.sum(y_eff))
    n_neg_y = n_eff - n_pos_y
    n_events = min(n_pos_y, n_neg_y)
    # Peduzzi et al. (1996) define EPV using predictor variables, excluding
    # the intercept. k_solve includes the intercept column, so use k_solve - 1.
    n_predictors = k_solve - 1  # exclude intercept
    epv = n_events / n_predictors if n_predictors > 0 else float("inf")

    if diagnostics_out is not None:
        diagnostics_out["epv"] = epv
        diagnostics_out["n_events"] = n_events
        diagnostics_out["k"] = n_predictors
        diagnostics_out["is_low"] = epv < epv_threshold

    if epv < epv_threshold:
        ctx = f" for {context_label}" if context_label else ""
        msg = (
            f"Low Events Per Variable (EPV = {epv:.1f}) in propensity score "
            f"model{ctx}. {n_events} minority-class observations for "
            f"{n_predictors} predictor variable(s). "
            f"Peduzzi et al. (1996) recommend EPV >= {epv_threshold:.0f}. "
            f"Estimates may be unreliable (overfitting, biased coefficients, "
            f"inflated standard errors). "
            f"Consider estimation_method='reg' to avoid propensity scores."
        )
        if rank_deficient_action == "error":
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # IRLS (Fisher scoring). Each weighted-least-squares step is solved via
    # EQUILIBRATED normal equations + Cholesky with an explicit condition
    # guard, falling back to the legacy tall-matrix lstsq (gelsd SVD) for any
    # iteration whose normal matrix cannot be certified well-conditioned.
    # Context: the OR path deliberately REMOVED a cho_solve(X'X) fast path
    # because it was NOT scale-equilibrated (see the covariate-reg notes in
    # staggered.py around `_equilibrated_lstsq`); this path differs on
    # exactly that axis - (1) columns are equilibrated to unit 2-norm ONCE
    # (a fixed reparameterization: X_eq @ beta_eq == X @ beta algebraically,
    # so probabilities are unchanged and beta is unscaled per iteration);
    # (2) cho_factor alone can SUCCEED with a garbage solution when cond(G)
    # exceeds ~1e10, so the guard is a dpocon reciprocal-condition estimate,
    # not just the factorization succeeding - working weights can crush a
    # column's effective scale (a dummy on a near-separated subgroup has
    # w_irls ~ 1e-10 on its support), so full column rank pre-loop does NOT
    # imply a well-conditioned G; (3) the fallback reproduces the legacy
    # computation for that iteration on the raw basis. IRLS state (beta,
    # convergence tol, warnings) stays in the RAW basis: a scaled-basis tol
    # would be ~sqrt(n)x tighter for every fit (the intercept column alone
    # has 2-norm sqrt(n)), and the separation check below reads raw beta.
    irls_col_norms = np.sqrt(np.einsum("ij,ij->j", X_solve, X_solve))
    irls_safe_norms = np.where(irls_col_norms > 0, irls_col_norms, 1.0)
    X_eq = X_solve / irls_safe_norms
    chol_fallback_iters = 0

    beta_solve = np.zeros(X_solve.shape[1])
    converged = False

    for iteration in range(max_iter):
        eta = X_solve @ beta_solve
        # Clip to prevent overflow in exp
        eta = np.clip(eta, -500, 500)
        mu = 1.0 / (1.0 + np.exp(-eta))
        # Clip mu to prevent zero working weights
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        # Working weights and working response
        w_irls = mu * (1.0 - mu)
        z = eta + (y - mu) / w_irls

        if weights is not None:
            w_total = weights * w_irls
        else:
            w_total = w_irls

        # Weighted least squares: solve (X'WX) beta = X'Wz
        sqrt_w = np.sqrt(w_total)
        zw = z * sqrt_w
        beta_new = None
        Xw_eq = X_eq * sqrt_w[:, None]
        gram = Xw_eq.T @ Xw_eq
        # 1-norm BEFORE factorization (dpocon contract); G is symmetric so
        # the max absolute column sum is the 1-norm.
        anorm = float(np.max(np.sum(np.abs(gram), axis=0)))
        if np.isfinite(anorm):
            try:
                chol = cho_factor(gram)
            except np.linalg.LinAlgError:
                chol = None
            if chol is not None:
                # cho_factor default lower=False pairs with dpocon's default
                # uplo='U' (dpocon has no `lower=` kwarg).
                rcond_gram, pocon_info = dpocon(chol[0], anorm)
                if (
                    pocon_info == 0
                    and np.isfinite(rcond_gram)
                    and rcond_gram > _IRLS_CHOL_RCOND_GUARD
                ):
                    beta_new = cho_solve(chol, Xw_eq.T @ zw) / irls_safe_norms
        if beta_new is None:
            # Guarded fallback: byte-identical to the pre-fast-path solve.
            chol_fallback_iters += 1
            Xw = X_solve * sqrt_w[:, None]
            beta_new, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)

        # Check convergence
        if np.max(np.abs(beta_new - beta_solve)) < tol:
            beta_solve = beta_new
            converged = True
            break
        beta_solve = beta_new

    if diagnostics_out is not None:
        diagnostics_out["irls_chol_fallback_iters"] = chol_fallback_iters

    # Final predicted probabilities
    eta_final = X_solve @ beta_solve
    eta_final = np.clip(eta_final, -500, 500)
    probs = 1.0 / (1.0 + np.exp(-eta_final))

    # Warnings
    if not converged:
        warnings.warn(
            f"Logistic regression did not converge in {max_iter} iterations. "
            f"Propensity score estimates may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    if check_separation:
        if np.max(np.abs(beta_solve)) > _LOGIT_SEPARATION_COEF_THRESHOLD:
            warnings.warn(
                "Large coefficients detected in propensity score model "
                f"(max|beta| > {_LOGIT_SEPARATION_COEF_THRESHOLD}), "
                "suggesting potential separation.",
                UserWarning,
                stacklevel=2,
            )
        n_extreme = int(
            np.sum(
                (probs < _LOGIT_SEPARATION_PROB_THRESHOLD)
                | (probs > 1 - _LOGIT_SEPARATION_PROB_THRESHOLD)
            )
        )
        if n_extreme > 0:
            warnings.warn(
                f"Near-separation detected in propensity score model: "
                f"{n_extreme} of {n} observations have predicted probabilities "
                f"within {_LOGIT_SEPARATION_PROB_THRESHOLD} of 0 or 1. ATT estimates may be sensitive to "
                f"model specification.",
                UserWarning,
                stacklevel=2,
            )

    # Expand beta back to original column count, accounting for columns
    # dropped in both the effective-sample check and the full-sample check
    if len(dropped_cols) > 0 or len(eff_dropped_original) > 0:
        # First expand from X_solve columns back to post-eff-drop columns
        # Use NaN for dropped coefficients (R convention: not estimable)
        beta_post_eff = np.full(k, np.nan)
        beta_post_eff[kept_cols] = beta_solve

        # Then expand from post-eff-drop columns back to original columns
        if len(eff_dropped_original) > 0:
            beta_full = np.full(k_original, np.nan)
            kept_original = [i for i in range(k_original) if i not in eff_dropped_original]
            beta_full[kept_original] = beta_post_eff
        else:
            beta_full = beta_post_eff
    else:
        beta_full = beta_solve

    return beta_full, probs


def _check_propensity_diagnostics(
    pscore: np.ndarray,
    trim_bound: float = 0.01,
) -> None:
    """
    Warn if propensity scores are extreme.

    Parameters
    ----------
    pscore : np.ndarray
        Predicted probabilities.
    trim_bound : float, default 0.01
        Trimming threshold.
    """
    n_extreme = int(np.sum((pscore < trim_bound) | (pscore > 1 - trim_bound)))
    if n_extreme > 0:
        n_total = len(pscore)
        pct = 100.0 * n_extreme / n_total
        warnings.warn(
            f"Propensity scores for {n_extreme} of {n_total} observations "
            f"({pct:.1f}%) were outside [{trim_bound}, {1 - trim_bound}] "
            f"and will be trimmed. This may indicate near-separation in "
            f"the propensity score model.",
            UserWarning,
            stacklevel=2,
        )


def compute_r_squared(
    y: np.ndarray,
    residuals: np.ndarray,
    adjusted: bool = False,
    n_params: int = 0,
) -> float:
    """
    Compute R-squared or adjusted R-squared.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    residuals : ndarray of shape (n,)
        OLS residuals.
    adjusted : bool, default False
        If True, compute adjusted R-squared.
    n_params : int, default 0
        Number of parameters (including intercept). Required if adjusted=True.

    Returns
    -------
    r_squared : float
        R-squared or adjusted R-squared.
    """
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return 0.0

    r_squared = 1 - (ss_res / ss_tot)

    if adjusted:
        n = len(y)
        if n <= n_params:
            return r_squared
        r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params)

    return r_squared


# =============================================================================
# LinearRegression Helper Class
# =============================================================================


@dataclass
class InferenceResult:
    """
    Container for inference results on a single coefficient.

    This dataclass provides a unified way to access coefficient estimates
    and their associated inference statistics.

    Attributes
    ----------
    coefficient : float
        The point estimate of the coefficient.
    se : float
        Standard error of the coefficient.
    t_stat : float
        T-statistic (coefficient / se).
    p_value : float
        Two-sided p-value for the t-statistic.
    conf_int : tuple of (float, float)
        Confidence interval (lower, upper).
    df : int or None
        Degrees of freedom used for inference. None if using normal distribution.
    alpha : float
        Significance level used for confidence interval.

    Examples
    --------
    >>> result = InferenceResult(
    ...     coefficient=2.5, se=0.5, t_stat=5.0, p_value=0.001,
    ...     conf_int=(1.52, 3.48), df=100, alpha=0.05
    ... )
    >>> result.is_significant()
    True
    >>> result.significance_stars()
    '***'
    """

    coefficient: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    df: Optional[int] = None
    alpha: float = 0.05

    def is_significant(self, alpha: Optional[float] = None) -> bool:
        """Check if the coefficient is statistically significant.

        Returns False for NaN p-values (unidentified coefficients).
        """
        if np.isnan(self.p_value):
            return False
        threshold = alpha if alpha is not None else self.alpha
        return self.p_value < threshold

    def significance_stars(self) -> str:
        """Return significance stars based on p-value.

        Returns empty string for NaN p-values (unidentified coefficients).
        """
        if np.isnan(self.p_value):
            return ""
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        elif self.p_value < 0.1:
            return "."
        return ""

    def to_dict(self) -> Dict[str, Union[float, Tuple[float, float], int, None]]:
        """Convert to dictionary representation."""
        return {
            "coefficient": self.coefficient,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int": self.conf_int,
            "df": self.df,
            "alpha": self.alpha,
        }


class LinearRegression:
    """
    OLS regression helper with unified coefficient extraction and inference.

    This class wraps the low-level `solve_ols` function and provides a clean
    interface for fitting regressions and extracting coefficient-level inference.
    It eliminates code duplication across estimators by centralizing the common
    pattern of: fit OLS -> extract coefficient -> compute SE -> compute t-stat
    -> compute p-value -> compute CI.

    Parameters
    ----------
    include_intercept : bool, default True
        Whether to automatically add an intercept column to the design matrix.
    robust : bool, default True
        Whether to use heteroskedasticity-robust (HC1) standard errors.
        If False and cluster_ids is None, uses classical OLS standard errors.
    cluster_ids : array-like, optional
        Cluster identifiers for cluster-robust standard errors.
        Overrides the `robust` parameter if provided.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    rank_deficient_action : str, default "warn"
        Action when design matrix is rank-deficient (linearly dependent columns):
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    weights : array-like, optional
        Observation weights. When survey_design is provided, weights are
        automatically derived from it (explicit weights are overridden).
    weight_type : str, default "pweight"
        Weight type: "pweight", "fweight", or "aweight".
    survey_design : ResolvedSurveyDesign, optional
        Resolved survey design for Taylor Series Linearization variance
        estimation. When provided, weights and weight_type are canonicalized
        from this object.
    vcov_type : {"classical", "hc1", "hc2", "hc2_bm", "conley"}, optional
        Variance-covariance family. Defaults to the ``robust`` alias
        (``robust=True`` -> ``"hc1"``, ``robust=False`` -> ``"classical"``).
        Passing an explicit ``vcov_type`` overrides ``robust`` unless the
        two conflict (e.g. ``robust=False, vcov_type="hc2"``), in which
        case ``__init__`` raises. See :func:`solve_ols` for the per-family
        semantics and unsupported combinations. For ``"hc2_bm"``: when
        ``cluster_ids`` is provided, dispatches to CR2 Bell-McCaffrey;
        with ``weights`` (one-way or clustered) dispatches to the
        clubSandwich WLS-CR2 port — supported for
        ``weight_type="pweight"`` only (``aweight`` / ``fweight`` raise
        ``NotImplementedError``). On top of the sandwich, the class
        stores per-coefficient BM Satterthwaite DOF (``self._bm_dof``)
        and threads it into ``get_inference``.

        For ``"conley"`` (Conley 1999 spatial-HAC) two operating modes are
        supported on the `LinearRegression` / `compute_robust_vcov` surface:
        cross-sectional (single-period or pooled cross-section) and panel
        block-decomposed (matches R ``conleyreg`` with ``lag_cutoff > 0``;
        pass the three co-required kwargs ``conley_time`` / ``conley_unit`` /
        ``conley_lag_cutoff``). Requires ``conley_coords`` (n × 2 array) and
        a positive ``conley_cutoff_km``. Combining ``vcov_type="conley"``
        with ``cluster_ids`` applies the combined spatial + cluster product
        kernel (Wave A #119; cluster must be constant within each unit on
        the panel path). Combining with ``weights`` raises
        ``NotImplementedError`` regardless of ``weight_type`` (weighted
        Conley is not implemented on the generic linalg surface);
        combining with ``survey_design`` (``LinearRegression`` only;
        ``compute_robust_vcov`` has no survey-design surface) likewise
        raises ``NotImplementedError``. For probability-sampling weights
        (``pweight`` / ``survey_design``) the deferral additionally
        reflects an open methodological question, with no canonical
        extension of Conley (1999) for weighted spatial-HAC under
        probability sampling. The DiD / MPD / TWFE
        estimators all support panel Conley by passing ``unit`` at fit-time
        (DiD as a fit-time kwarg; MPD/TWFE via the existing ``unit``
        argument), threading ``conley_time`` / ``conley_unit`` into the
        block-decomposed sandwich.
    conley_coords : ndarray of shape (n, 2), optional
        Required when ``vcov_type="conley"``. Two-column array of
        ``[lat, lon]`` (degrees, for ``conley_metric="haversine"``) or
        projected coordinates (for ``conley_metric="euclidean"`` / callable
        metric). Raises ``ValueError`` when missing under Conley.
    conley_cutoff_km : float, optional
        Required when ``vcov_type="conley"``. Positive finite bandwidth in
        km (haversine) or coord units (euclidean / callable). No default
        per Conley 1999 Section 5's sensitivity-grid recommendation.
    conley_metric : {"haversine", "euclidean", callable}, default "haversine"
        Distance metric. Haversine uses Earth's mean radius 6371.01 km
        matching R ``conleyreg``. Euclidean treats the coords as already
        projected. Callable signature ``(coords1, coords2) -> n×n``.
    conley_kernel : {"bartlett", "uniform"}, default "bartlett"
        Kernel evaluated on pairwise distance ``d_ij/h``. ``"bartlett"`` is
        the radial 1-D specialization (matching R ``conleyreg``);
        ``"uniform"`` is the truncated indicator. Both kernels emit a
        ``UserWarning`` if the resulting meat is materially indefinite —
        neither is formally PSD-guaranteed in the radial pairwise form
        (Conley 1999's explicit PSD Bartlett formula is the 2-D separable
        product window, Eq 3.14, not the 1-D radial pairwise form).
    df_convention : {"residual", "cluster"}, default "residual"
        Degrees-of-freedom convention for ``get_inference`` t/p/CI on
        clustered fits. ``"residual"`` uses the fitted residual df;
        ``"cluster"`` uses the Stata/fixest cluster df ``G − 1`` (from
        ``n_clusters_``). Fallback-level only: survey df and per-coefficient
        Bell-McCaffrey DOF always take precedence. No effect on
        coefficients, SEs, unclustered fits, or ``vcov_type="conley"`` (no
        documented ``G − 1`` reference for the Conley+cluster product
        kernel). Default flips at v4.

    Attributes
    ----------
    coefficients_ : ndarray
        Fitted coefficient values (available after fit).
    vcov_ : ndarray
        Variance-covariance matrix (available after fit).
    residuals_ : ndarray
        Residuals from the fit (available after fit).
    fitted_values_ : ndarray
        Fitted values from the fit (available after fit).
    n_obs_ : int
        Number of observations (available after fit).
    n_params_ : int
        Number of parameters including intercept (available after fit).
    n_params_effective_ : int
        Effective number of parameters after dropping linearly dependent columns.
        Equals n_params_ for full-rank matrices (available after fit).
    n_clusters_ : int or None
        Effective cluster count on a clustered fit (positive-weight clusters
        only when weighted); None on unclustered fits. Feeds the
        ``df_convention="cluster"`` inference df (available after fit).
    df_ : int
        Degrees of freedom (n - n_params_effective) (available after fit).

    Examples
    --------
    Basic usage with automatic intercept:

    >>> import numpy as np
    >>> from diff_diff.linalg import LinearRegression
    >>> X = np.random.randn(100, 2)
    >>> y = 1 + 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)
    >>> reg = LinearRegression().fit(X, y)
    >>> print(f"Intercept: {reg.coefficients_[0]:.2f}")
    >>> inference = reg.get_inference(1)  # inference for first predictor
    >>> print(f"Coef: {inference.coefficient:.2f}, SE: {inference.se:.2f}")

    Using with cluster-robust standard errors:

    >>> cluster_ids = np.repeat(np.arange(20), 5)  # 20 clusters of 5
    >>> reg = LinearRegression(cluster_ids=cluster_ids).fit(X, y)
    >>> inference = reg.get_inference(1)
    >>> print(f"Cluster-robust SE: {inference.se:.2f}")

    Extracting multiple coefficients at once:

    >>> results = reg.get_inference_batch([1, 2])
    >>> for idx, inf in results.items():
    ...     print(f"Coef {idx}: {inf.coefficient:.2f} ({inf.significance_stars()})")
    """

    def __init__(
        self,
        include_intercept: bool = True,
        robust: bool = True,
        cluster_ids: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        rank_deficient_action: str = "warn",
        weights: Optional[np.ndarray] = None,
        weight_type: str = "pweight",
        survey_design: object = None,
        vcov_type: Optional[str] = None,
        conley_coords: Optional[np.ndarray] = None,
        conley_cutoff_km: Optional[float] = None,
        conley_metric: ConleyMetric = "haversine",
        conley_kernel: str = "bartlett",
        conley_time: Optional[np.ndarray] = None,
        conley_unit: Optional[np.ndarray] = None,
        conley_lag_cutoff: Optional[int] = None,
        df_convention: str = "residual",
    ):
        if df_convention not in ("residual", "cluster"):
            raise ValueError(
                f"df_convention must be 'residual' or 'cluster', got {df_convention!r}"
            )
        self.include_intercept = include_intercept
        self.robust = robust
        self.cluster_ids = cluster_ids
        self.alpha = alpha
        self.rank_deficient_action = rank_deficient_action
        self.weights = weights
        self.weight_type = weight_type
        self.survey_design = survey_design  # ResolvedSurveyDesign or None
        self.conley_coords = conley_coords
        self.conley_cutoff_km = conley_cutoff_km
        self.conley_metric = conley_metric
        self.conley_kernel = conley_kernel
        # Phase 2 panel block-decomposed Conley kwargs. All three are
        # three-way co-required at fit-time when supplied; all None preserves
        # the Phase 1 cross-sectional path.
        self.conley_time = conley_time
        self.conley_unit = conley_unit
        self.conley_lag_cutoff = conley_lag_cutoff
        # Inference df convention for clustered analytical fits:
        # "residual" (default) keeps the fitted residual df `n - K_full` for
        # t/p/CI; "cluster" uses the Stata/fixest cluster df `G - 1` instead.
        # Applies ONLY at the fallback level of the `get_inference` df
        # resolution: survey df and per-coefficient Bell-McCaffrey DOF (which
        # are more refined small-sample corrections) always take precedence.
        # The default flips to "cluster" at v4 (see REGISTRY clustered-CR1
        # inference-df deviation note).
        self.df_convention = df_convention
        # Resolve vcov_type from the legacy `robust` alias via the shared helper.
        self.vcov_type = resolve_vcov_type(robust, vcov_type)
        # Preserve the raw constructor arg (possibly None) so `fit()` can
        # distinguish "alias-derived classical" from "explicit classical".
        # This is the single source of truth for backward-compat remap
        # decisions (robust=False + cluster -> CR1). `fit()` treats the
        # configured state as IMMUTABLE and computes all effective fit-
        # time values as locals, so repeat fits with different cluster
        # or survey context produce the correct result without state
        # drift between calls.
        self._vcov_type_arg = vcov_type
        self._vcov_type_explicit = vcov_type is not None

        # Fitted attributes (set by fit())
        self.coefficients_: Optional[np.ndarray] = None
        self.vcov_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self.n_obs_: Optional[int] = None
        self.n_params_: Optional[int] = None
        self.n_params_effective_: Optional[int] = None
        self.df_: Optional[int] = None
        self.survey_df_: Optional[int] = None
        # Effective cluster count on a clustered fit (None otherwise);
        # feeds the df_convention="cluster" G-1 inference df.
        self.n_clusters_: Optional[int] = None
        # Per-coefficient Bell-McCaffrey DOF vector when vcov_type="hc2_bm".
        # None for all other vcov_types; preserves df_ as the fallback.
        self._bm_dof: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cluster_ids: Optional[np.ndarray] = None,
        df_adjustment: int = 0,
    ) -> "LinearRegression":
        """
        Fit OLS regression.

        Parameters
        ----------
        X : ndarray of shape (n, k)
            Design matrix. An intercept column will be added if include_intercept=True.
        y : ndarray of shape (n,)
            Response vector.
        cluster_ids : ndarray, optional
            Cluster identifiers for this fit. Overrides the instance-level
            cluster_ids if provided.
        df_adjustment : int, default 0
            Additional degrees of freedom adjustment (e.g., for absorbed fixed effects).
            The effective df will be n - k - df_adjustment.

        Returns
        -------
        self : LinearRegression
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Reset replicate df from any previous fit
        self._replicate_df = None

        # Add intercept if requested
        if self.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        # Use provided cluster_ids or fall back to instance-level
        effective_cluster_ids = cluster_ids if cluster_ids is not None else self.cluster_ids

        # Resolve the effective fit-time vcov_type WITHOUT mutating self.
        # Legacy-alias backward compat: when the user supplied
        # ``robust=False`` without an explicit ``vcov_type`` and a cluster
        # structure is present at fit time, remap the implicit
        # ``"classical"`` to ``"hc1"`` so the call dispatches to CR1
        # instead of raising. Per-fit local only; the configured
        # ``self.vcov_type`` is left untouched so a subsequent unclustered
        # fit continues to use classical SEs.
        _fit_vcov_type = self.vcov_type
        if (
            not self._vcov_type_explicit
            and _fit_vcov_type == "classical"
            and effective_cluster_ids is not None
        ):
            warnings.warn(
                "LinearRegression(robust=False) with clustered fit "
                "(cluster_ids=...) historically produced CR1 cluster-"
                "robust SEs. To preserve that behavior, vcov_type has "
                "been remapped from 'classical' to 'hc1' for THIS fit "
                "only (configured state on `self` is preserved). Pass "
                "vcov_type='hc1' explicitly to silence this warning, or "
                "vcov_type='classical' (with cluster_ids=None) for "
                "non-robust SEs.",
                UserWarning,
                stacklevel=2,
            )
            _fit_vcov_type = "hc1"

        # Determine if survey vcov should be used
        _use_survey_vcov = False
        if self.survey_design is not None:
            from diff_diff.survey import ResolvedSurveyDesign

            if isinstance(self.survey_design, ResolvedSurveyDesign):
                _use_survey_vcov = self.survey_design.needs_survey_vcov
                # Canonicalize weights from survey_design to ensure consistency
                # between coefficient estimation and survey vcov computation.
                # Locals only — configured self.weights / self.weight_type
                # are preserved.
                if self.weights is not None and self.weights is not self.survey_design.weights:
                    warnings.warn(
                        "Explicit weights= differ from survey_design.weights. "
                        "Using survey_design weights for both coefficient "
                        "estimation and variance computation to ensure "
                        "consistency.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Reject vcov_type='conley' + survey_design at LinearRegression entry.
        # The downstream `_validate_vcov_args` rejects this combination inside
        # `compute_robust_vcov`, but `LinearRegression.fit()` skips that path
        # entirely when the survey design needs survey variance (return_vcov
        # is set to False on the solve_ols call), and the survey vcov path
        # would silently overwrite the result with a non-Conley variance
        # under a Conley request. Front-door the rejection here so the
        # contract is enforced uniformly. Weighted spatial-HAC under
        # probability sampling is an open methodological question (no
        # canonical extension of Conley (1999) exists for the combination);
        # the generic LinearRegression / compute_robust_vcov path supports
        # unweighted Conley only (cross-sectional or panel block-decomposed
        # via conley_lag_cutoff > 0).
        if _fit_vcov_type == "conley" and _use_survey_vcov:
            raise NotImplementedError(
                "LinearRegression(vcov_type='conley', survey_design=...) "
                "is deferred — weighted spatial-HAC under probability "
                "sampling is an open methodological question; no "
                "canonical extension of Conley (1999) exists for the "
                "combination. The generic LinearRegression / "
                "compute_robust_vcov path supports unweighted Conley only "
                "(cross-sectional or panel block-decomposed via "
                "conley_lag_cutoff > 0) without a survey design."
            )

        # Resolve effective fit-time weights/weight_type WITHOUT mutating
        # self. When a survey design is present, canonicalize weights from
        # the design so coefficient estimation and survey vcov agree.
        # Otherwise use what the user configured.
        _fit_weights = self.weights
        _fit_weight_type = self.weight_type
        if self.survey_design is not None:
            from diff_diff.survey import ResolvedSurveyDesign as _RSD2

            if isinstance(self.survey_design, _RSD2):
                _fit_weights = self.survey_design.weights
                _fit_weight_type = self.survey_design.weight_type
        if _fit_weights is not None:
            _fit_weights = _validate_weights(_fit_weights, _fit_weight_type, X.shape[0])

        # Inject cluster as PSU for survey variance when no PSU specified.
        # Use a local variable to avoid mutating self.survey_design, which
        # would cause stale PSU on repeated fit() calls with different clusters.
        _effective_survey_design = self.survey_design
        if (
            effective_cluster_ids is not None
            and _effective_survey_design is not None
            and _use_survey_vcov
        ):
            from diff_diff.survey import ResolvedSurveyDesign as _RSD
            from diff_diff.survey import _inject_cluster_as_psu

            if isinstance(_effective_survey_design, _RSD) and _effective_survey_design.psu is None:
                _effective_survey_design = _inject_cluster_as_psu(
                    _effective_survey_design, effective_cluster_ids
                )

        if _fit_vcov_type != "classical" or effective_cluster_ids is not None:
            # Use solve_ols with robust/cluster SEs.
            # When survey vcov will be used, skip standard vcov computation.
            # For hc2_bm (non-survey), ALSO skip solve_ols's vcov: the CR2
            # sandwich produces (vcov, dof) together, so the block below gets
            # BOTH from a single compute_robust_vcov(return_dof=True) call
            # instead of computing the vcov here and recomputing the whole
            # sandwich a second time just for the dof (#475).
            _is_bm_path = _fit_vcov_type == "hc2_bm" and not _use_survey_vcov
            coefficients, residuals, fitted, vcov = solve_ols(
                X,
                y,
                cluster_ids=effective_cluster_ids,
                return_fitted=True,
                return_vcov=(not _use_survey_vcov) and not _is_bm_path,
                rank_deficient_action=self.rank_deficient_action,
                weights=_fit_weights,
                weight_type=_fit_weight_type,
                vcov_type=_fit_vcov_type,
                conley_coords=self.conley_coords,
                conley_cutoff_km=self.conley_cutoff_km,
                conley_metric=self.conley_metric,
                conley_kernel=self.conley_kernel,
                conley_time=self.conley_time,
                conley_unit=self.conley_unit,
                conley_lag_cutoff=self.conley_lag_cutoff,
            )
            # For hc2_bm (non-survey), compute the CR2 vcov AND per-coefficient
            # Bell-McCaffrey DOF together in a SINGLE compute_robust_vcov call
            # (solve_ols skipped its vcov above via `_is_bm_path`). Both the
            # one-way HC2+BM case and the (weighted) clustered CR2 case route
            # through the same `_compute_robust_vcov_numpy`/`_compute_cr2_bm`, so
            # this is bit-identical to the prior two-call form while computing the
            # O(n^2 k) sandwich once instead of twice (#475). The dispatcher
            # already rejects non-pweight weight types for hc2_bm + weights.
            if _is_bm_path:
                # Rank-deficient solves set NaN coefficients for dropped columns.
                nan_mask = np.isnan(coefficients)
                if np.all(nan_mask):
                    # All columns dropped: NaN vcov (matches solve_ols's
                    # rank-deficient all-drop) and no BM DOF (n-k fallback).
                    vcov = np.full((X.shape[1], X.shape[1]), np.nan)
                    self._bm_dof = None
                elif not np.any(nan_mask):
                    vcov, self._bm_dof = compute_robust_vcov(
                        X,
                        residuals,
                        cluster_ids=effective_cluster_ids,
                        weights=_fit_weights,
                        weight_type=_fit_weight_type,
                        vcov_type="hc2_bm",
                        return_dof=True,
                    )
                else:
                    # Rank-deficient: compute on identified columns only, then
                    # expand BOTH vcov and per-coef DOF with NaN for dropped
                    # columns (mirrors solve_ols's `_expand_vcov_with_nan` path).
                    kept = np.where(~nan_mask)[0]
                    vcov_reduced, dof_kept = compute_robust_vcov(
                        X[:, kept],
                        residuals,
                        cluster_ids=effective_cluster_ids,
                        weights=_fit_weights,
                        weight_type=_fit_weight_type,
                        vcov_type="hc2_bm",
                        return_dof=True,
                    )
                    vcov = _expand_vcov_with_nan(vcov_reduced, X.shape[1], kept)
                    full = np.full(X.shape[1], np.nan)
                    full[kept] = dof_kept
                    self._bm_dof = full
            else:
                self._bm_dof = None
        else:
            # Classical OLS - compute vcov separately
            coefficients, residuals, fitted, _ = solve_ols(
                X,
                y,
                return_fitted=True,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                weights=_fit_weights,
                weight_type=_fit_weight_type,
            )
            # Compute classical OLS variance-covariance matrix
            # Handle rank-deficient case: use effective rank for df
            n, k = X.shape
            nan_mask = np.isnan(coefficients)
            k_effective = k - np.sum(nan_mask)  # Number of identified coefficients

            # Effective n for df: fweights use sum(w), pweight/aweight with
            # zeros use positive-weight count (zero-weight rows don't contribute)
            n_eff_df = n
            if _fit_weights is not None:
                if _fit_weight_type == "fweight":
                    n_eff_df = int(round(np.sum(_fit_weights)))
                elif np.any(_fit_weights == 0):
                    n_eff_df = int(np.count_nonzero(_fit_weights > 0))

            if k_effective == 0:
                # All coefficients dropped - no valid inference
                vcov = np.full((k, k), np.nan)
            elif np.any(nan_mask):
                # Rank-deficient: compute vcov for identified coefficients only
                kept_cols = np.where(~nan_mask)[0]
                X_reduced = X[:, kept_cols]
                if _fit_weights is not None:
                    # Weighted classical vcov: use weighted RSS and X'WX
                    w = _fit_weights
                    mse = np.sum(w * residuals**2) / (n_eff_df - k_effective)
                    XtWX_reduced = X_reduced.T @ (X_reduced * w[:, np.newaxis])
                    try:
                        vcov_reduced = np.linalg.solve(XtWX_reduced, mse * np.eye(k_effective))
                    except np.linalg.LinAlgError:
                        vcov_reduced = np.linalg.pinv(XtWX_reduced) * mse
                else:
                    mse = np.sum(residuals**2) / (n_eff_df - k_effective)
                    try:
                        vcov_reduced = np.linalg.solve(
                            X_reduced.T @ X_reduced, mse * np.eye(k_effective)
                        )
                    except np.linalg.LinAlgError:
                        vcov_reduced = np.linalg.pinv(X_reduced.T @ X_reduced) * mse
                # Expand to full size with NaN for dropped columns
                vcov = _expand_vcov_with_nan(vcov_reduced, k, kept_cols)
            else:
                # Full rank: standard computation
                if _fit_weights is not None:
                    # Weighted classical vcov: use weighted RSS and X'WX
                    w = _fit_weights
                    mse = np.sum(w * residuals**2) / (n_eff_df - k)
                    XtWX = X.T @ (X * w[:, np.newaxis])
                    try:
                        vcov = np.linalg.solve(XtWX, mse * np.eye(k))
                    except np.linalg.LinAlgError:
                        vcov = np.linalg.pinv(XtWX) * mse
                else:
                    mse = np.sum(residuals**2) / (n_eff_df - k)
                    try:
                        vcov = np.linalg.solve(X.T @ X, mse * np.eye(k))
                    except np.linalg.LinAlgError:
                        vcov = np.linalg.pinv(X.T @ X) * mse

        # Compute survey vcov if applicable
        if _use_survey_vcov:
            from diff_diff.survey import ResolvedSurveyDesign as _RSD

            _uses_rep = (
                isinstance(_effective_survey_design, _RSD)
                and _effective_survey_design.uses_replicate_variance
            )

            if _uses_rep:
                from diff_diff.survey import compute_replicate_vcov

                nan_mask = np.isnan(coefficients)
                if np.any(nan_mask):
                    kept_cols = np.where(~nan_mask)[0]
                    if len(kept_cols) > 0:
                        vcov_reduced, _n_valid_rep = compute_replicate_vcov(
                            X[:, kept_cols],
                            y,
                            coefficients[kept_cols],
                            _effective_survey_design,
                            weight_type=_fit_weight_type,
                        )
                        vcov = _expand_vcov_with_nan(vcov_reduced, X.shape[1], kept_cols)
                    else:
                        vcov = np.full((X.shape[1], X.shape[1]), np.nan)
                        _n_valid_rep = 0
                else:
                    vcov, _n_valid_rep = compute_replicate_vcov(
                        X,
                        y,
                        coefficients,
                        _effective_survey_design,
                        weight_type=_fit_weight_type,
                    )
                # Store effective replicate df only when replicates were dropped
                if _n_valid_rep < _effective_survey_design.n_replicates:
                    self._replicate_df = _n_valid_rep - 1 if _n_valid_rep > 1 else None
                else:
                    self._replicate_df = None  # use rank-based df from design
            else:
                from diff_diff.survey import compute_survey_vcov

                nan_mask = np.isnan(coefficients)
                if np.any(nan_mask):
                    kept_cols = np.where(~nan_mask)[0]
                    if len(kept_cols) > 0:
                        vcov_reduced = compute_survey_vcov(
                            X[:, kept_cols], residuals, _effective_survey_design
                        )
                        vcov = _expand_vcov_with_nan(vcov_reduced, X.shape[1], kept_cols)
                    else:
                        vcov = np.full((X.shape[1], X.shape[1]), np.nan)
                else:
                    vcov = compute_survey_vcov(X, residuals, _effective_survey_design)

        # Store fitted attributes
        self.coefficients_ = coefficients
        self.vcov_ = vcov
        self.residuals_ = residuals
        self.fitted_values_ = fitted
        self._y = y
        self._X = X
        self.n_obs_ = X.shape[0]
        self.n_params_ = X.shape[1]
        # Preserve the effective fit-time weights / weight_type / vcov_type
        # as fitted attributes so downstream helpers (e.g., compute_deff)
        # can read what was actually used without needing to re-derive
        # from the configured state. These are per-fit values; a repeat
        # fit overwrites them. Sklearn convention: fitted attrs end in
        # `_` (so they are distinguishable from config).
        self._fit_weights_ = _fit_weights
        self._fit_weight_type_ = _fit_weight_type
        self._fit_vcov_type_ = _fit_vcov_type

        # Compute effective number of parameters (excluding dropped columns)
        # This is needed for correct degrees of freedom in inference
        nan_mask = np.isnan(coefficients)
        self.n_params_effective_ = int(self.n_params_ - np.sum(nan_mask))
        # Effective n for df: fweights use sum(w), pweight/aweight with
        # zeros use positive-weight count (matches compute_robust_vcov)
        n_eff_df = self.n_obs_
        if _fit_weights is not None:
            if _fit_weight_type == "fweight":
                n_eff_df = int(round(np.sum(_fit_weights)))
            elif np.any(_fit_weights == 0):
                n_eff_df = int(np.count_nonzero(_fit_weights > 0))
        # Absorbed-FE variance scale (fixest full-K convention): the classical
        # sse/(n-k) and HC1 n/(n-k) factors computed above use k_visible, but
        # with absorbed FE the correct finite-sample count is
        # K_full = n_params_effective_ + df_adjustment (the t-df below already
        # uses it). Rescale the NON-CLUSTERED iid/hetero vcov so the SE's k
        # agrees with the t-df's and with fixest feols(vcov="iid"/"hetero").
        # Clustered SEs keep k_visible (fixest ssc nested-FE convention already
        # matches); hc2/hc2_bm use leverage/Satterthwaite DOF; survey has its
        # own df; full-dummy fits carry df_adjustment == 0. When the full-K
        # residual dof is non-positive the helper returns NaN and we void the
        # vcov -> NaN inference (fail-closed, per the non-finite-df contract).
        if (
            df_adjustment > 0
            and effective_cluster_ids is None
            and not _use_survey_vcov
            and _fit_vcov_type in ("classical", "hc1")
        ):
            _fe_scale = _absorbed_fe_vcov_scale(n_eff_df, self.n_params_effective_, df_adjustment)
            if np.isnan(_fe_scale):
                self.vcov_ = np.full_like(self.vcov_, np.nan)
            elif _fe_scale != 1.0:
                self.vcov_ = self.vcov_ * _fe_scale

        self.df_ = n_eff_df - self.n_params_effective_ - df_adjustment

        # Effective cluster count for the df_convention="cluster" inference
        # df (G - 1). Mirrors the vcov path's convention: on a weighted fit
        # only clusters with positive total weight count (zero-weight rows
        # are inert per the linalg contract).
        self.n_clusters_ = None
        if effective_cluster_ids is not None:
            self.n_clusters_ = effective_cluster_count(effective_cluster_ids, _fit_weights)

        # Survey degrees of freedom: n_PSU - n_strata (overrides standard df)
        self.survey_df_ = None
        if _effective_survey_design is not None:
            from diff_diff.survey import ResolvedSurveyDesign

            if isinstance(_effective_survey_design, ResolvedSurveyDesign):
                self.survey_df_ = _effective_survey_design.df_survey
                # Override with effective replicate df if available
                if hasattr(self, "_replicate_df") and self._replicate_df is not None:
                    self.survey_df_ = self._replicate_df

        return self

    def compute_deff(self, coefficient_names=None):
        """Compute per-coefficient design effect diagnostics.

        Compares the survey vcov to an SRS (HC1) baseline.  Must be called
        after ``fit()`` with a survey design.

        Returns
        -------
        DEFFDiagnostics
        """
        self._check_fitted()
        if not (hasattr(self, "survey_design") and self.survey_design is not None):
            raise ValueError(
                "compute_deff() requires a survey design. " "Fit with survey_design= first."
            )
        from diff_diff.survey import compute_deff_diagnostics

        # Handle rank-deficient fits: compute DEFF only on kept columns,
        # then expand back with NaN for dropped columns
        nan_mask = np.isnan(self.coefficients_)
        if np.any(nan_mask):
            kept = np.where(~nan_mask)[0]
            if len(kept) == 0:
                k = len(self.coefficients_)
                nan_arr = np.full(k, np.nan)
                from diff_diff.survey import DEFFDiagnostics

                return DEFFDiagnostics(
                    deff=nan_arr,
                    effective_n=nan_arr.copy(),
                    srs_se=nan_arr.copy(),
                    survey_se=nan_arr.copy(),
                    coefficient_names=coefficient_names,
                )
            # Compute on kept columns only. Use fit-time effective weights
            # (captured in `self._fit_weights_`) so survey-canonicalized
            # weights are used for the DEFF computation, not the
            # user-configured state.
            X_kept = self._X[:, kept]
            vcov_kept = self.vcov_[np.ix_(kept, kept)]
            _deff_weights = getattr(self, "_fit_weights_", self.weights)
            _deff_weight_type = getattr(self, "_fit_weight_type_", self.weight_type)
            deff_kept = compute_deff_diagnostics(
                X_kept,
                self.residuals_,
                vcov_kept,
                _deff_weights,
                weight_type=_deff_weight_type,
            )
            # Expand back to full size with NaN for dropped
            k = len(self.coefficients_)
            full_deff = np.full(k, np.nan)
            full_eff_n = np.full(k, np.nan)
            full_srs_se = np.full(k, np.nan)
            full_survey_se = np.full(k, np.nan)
            full_deff[kept] = deff_kept.deff
            full_eff_n[kept] = deff_kept.effective_n
            full_srs_se[kept] = deff_kept.srs_se
            full_survey_se[kept] = deff_kept.survey_se
            from diff_diff.survey import DEFFDiagnostics

            return DEFFDiagnostics(
                deff=full_deff,
                effective_n=full_eff_n,
                srs_se=full_srs_se,
                survey_se=full_survey_se,
                coefficient_names=coefficient_names,
            )

        _deff_weights = getattr(self, "_fit_weights_", self.weights)
        _deff_weight_type = getattr(self, "_fit_weight_type_", self.weight_type)
        return compute_deff_diagnostics(
            self._X,
            self.residuals_,
            self.vcov_,
            _deff_weights,
            weight_type=_deff_weight_type,
            coefficient_names=coefficient_names,
        )

    def _check_fitted(self) -> None:
        """Raise error if model has not been fitted."""
        if self.coefficients_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

    def get_coefficient(self, index: int) -> float:
        """
        Get the coefficient value at a specific index.

        Parameters
        ----------
        index : int
            Index of the coefficient in the coefficient array.

        Returns
        -------
        float
            Coefficient value.
        """
        self._check_fitted()
        assert self.coefficients_ is not None
        return float(self.coefficients_[index])

    def get_se(self, index: int) -> float:
        """
        Get the standard error for a coefficient.

        Parameters
        ----------
        index : int
            Index of the coefficient.

        Returns
        -------
        float
            Standard error.
        """
        self._check_fitted()
        assert self.vcov_ is not None
        # Clamp a tiny-negative variance artifact at 0 before sqrt. A high-leverage
        # / degenerate coefficient (e.g. an absorbed-FE dummy near-collinear with the
        # treatment) can have a CR2/HC variance of ~0 that lands just below zero under
        # BLAS-dependent float rounding; without the clamp `np.sqrt` returns NaN
        # nondeterministically (passes single-threaded, fails under parallel test
        # load). The SE is then finite — 0 for a genuinely-zero variance.
        return float(np.sqrt(max(float(self.vcov_[index, index]), 0.0)))

    def get_inference(
        self,
        index: int,
        alpha: Optional[float] = None,
        df: Optional[int] = None,
    ) -> InferenceResult:
        """
        Get full inference results for a coefficient.

        This is the primary method for extracting coefficient-level inference,
        returning all statistics in a single call.

        Parameters
        ----------
        index : int
            Index of the coefficient in the coefficient array.
        alpha : float, optional
            Significance level for CI. Defaults to instance-level alpha.
        df : int, optional
            Degrees of freedom. Defaults to fitted df (n - k - df_adjustment).
            Set to None explicitly to use normal distribution instead of t.

        Returns
        -------
        InferenceResult
            Dataclass containing coefficient, se, t_stat, p_value, conf_int.

        Examples
        --------
        >>> reg = LinearRegression().fit(X, y)
        >>> result = reg.get_inference(1)
        >>> print(f"Effect: {result.coefficient:.3f} (SE: {result.se:.3f})")
        >>> print(f"95% CI: [{result.conf_int[0]:.3f}, {result.conf_int[1]:.3f}]")
        >>> if result.is_significant():
        ...     print("Statistically significant!")
        """
        self._check_fitted()
        assert self.coefficients_ is not None
        assert self.vcov_ is not None

        coef = float(self.coefficients_[index])
        # See get_se: clamp a tiny-negative variance artifact at 0 so SE is finite, not NaN.
        se = float(np.sqrt(max(float(self.vcov_[index, index]), 0.0)))

        # Use instance alpha if not provided
        effective_alpha = alpha if alpha is not None else self.alpha

        # Use survey df if available, otherwise per-coef BM DOF (hc2_bm), then fitted df.
        # Note: df=None means use normal distribution
        if df is not None:
            effective_df = df
        elif self.survey_df_ is not None:
            effective_df = self.survey_df_
        elif self._bm_dof is not None and 0 <= index < len(self._bm_dof):
            bm_val = self._bm_dof[index]
            if not np.isfinite(bm_val):
                # NaN BM DOF means the noise-floor guard fired (typically a
                # high-leverage FE-dummy contrast on weighted CR2-BM). Falling
                # through to df=None would silently use the normal distribution
                # and produce misleading p≈0 / zero-width CIs. Instead, return
                # NaN inference fields for the affected coefficient.
                return InferenceResult(
                    coefficient=coef,
                    se=se,
                    t_stat=float("nan"),
                    p_value=float("nan"),
                    conf_int=(float("nan"), float("nan")),
                    df=None,
                    alpha=effective_alpha,
                )
            effective_df = float(bm_val)
        elif (
            hasattr(self, "survey_design")
            and self.survey_design is not None
            and hasattr(self.survey_design, "uses_replicate_variance")
            and self.survey_design.uses_replicate_variance
        ):
            # Replicate design with undefined df (rank <= 1) — NaN inference
            warnings.warn(
                "Replicate design has undefined survey d.f. (rank <= 1). "
                "Inference fields will be NaN.",
                UserWarning,
                stacklevel=2,
            )
            effective_df = 0  # Forces NaN from t-distribution
        elif (
            self.df_convention == "cluster"
            and self.n_clusters_ is not None
            and self.vcov_type != "conley"
        ):
            # Opt-in Stata/fixest cluster-df convention for clustered
            # analytical fits: t/p/CI at df = G - 1 instead of the residual
            # df. conley is excluded: the combined Conley+cluster product
            # kernel is a diff-diff convention with no documented G-1 df
            # reference (REGISTRY Conley section); it keeps the residual df.
            # Deliberately the LAST branch before the residual fallback:
            # survey df and per-coefficient Bell-McCaffrey DOF are more
            # refined small-sample corrections and always win. Default flips
            # at v4 (REGISTRY clustered-CR1 inference-df deviation note).
            if self.n_clusters_ <= 1:
                # Cluster df G - 1 is undefined for an effectively
                # one-cluster fit (e.g. weighted fit where only one cluster
                # carries positive weight). Fail closed with NaN inference
                # (df=0 forces NaN through safe_inference) instead of
                # silently degrading to normal-theory inference.
                warnings.warn(
                    "df_convention='cluster' requires at least 2 effective "
                    f"clusters; got {self.n_clusters_}. Inference fields "
                    "will be NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                return InferenceResult(
                    coefficient=coef,
                    se=se,
                    t_stat=float("nan"),
                    p_value=float("nan"),
                    conf_int=(float("nan"), float("nan")),
                    df=None,
                    alpha=effective_alpha,
                )
            effective_df = self.n_clusters_ - 1
        else:
            effective_df = self.df_

        # Warn if df is non-positive and fall back to normal distribution
        # (skip for replicate designs — df=0 is intentional for NaN inference)
        _is_replicate = (
            hasattr(self, "survey_design")
            and self.survey_design is not None
            and hasattr(self.survey_design, "uses_replicate_variance")
            and self.survey_design.uses_replicate_variance
        )
        if effective_df is not None and effective_df <= 0 and not _is_replicate:
            warnings.warn(
                f"Degrees of freedom is non-positive (df={effective_df}). "
                "Using normal distribution instead of t-distribution for inference.",
                UserWarning,
            )
            effective_df = None

        # Use project-standard NaN-safe inference (returns all-NaN when SE <= 0)
        from diff_diff.utils import safe_inference

        t_stat, p_value, conf_int = safe_inference(coef, se, alpha=effective_alpha, df=effective_df)

        return InferenceResult(
            coefficient=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            df=effective_df,
            alpha=effective_alpha,
        )

    def get_inference_batch(
        self,
        indices: List[int],
        alpha: Optional[float] = None,
        df: Optional[int] = None,
    ) -> Dict[int, InferenceResult]:
        """
        Get inference results for multiple coefficients.

        Parameters
        ----------
        indices : list of int
            Indices of coefficients to extract.
        alpha : float, optional
            Significance level for CIs. Defaults to instance-level alpha.
        df : int, optional
            Degrees of freedom. Defaults to fitted df.

        Returns
        -------
        dict
            Dictionary mapping index -> InferenceResult.

        Examples
        --------
        >>> reg = LinearRegression().fit(X, y)
        >>> results = reg.get_inference_batch([1, 2, 3])
        >>> for idx, inf in results.items():
        ...     print(f"Coef {idx}: {inf.coefficient:.3f} {inf.significance_stars()}")
        """
        self._check_fitted()
        return {idx: self.get_inference(idx, alpha=alpha, df=df) for idx in indices}

    def get_all_inference(
        self,
        alpha: Optional[float] = None,
        df: Optional[int] = None,
    ) -> List[InferenceResult]:
        """
        Get inference results for all coefficients.

        Parameters
        ----------
        alpha : float, optional
            Significance level for CIs. Defaults to instance-level alpha.
        df : int, optional
            Degrees of freedom. Defaults to fitted df.

        Returns
        -------
        list of InferenceResult
            Inference results for each coefficient in order.
        """
        self._check_fitted()
        return [self.get_inference(i, alpha=alpha, df=df) for i in range(len(self.coefficients_))]

    def r_squared(self, adjusted: bool = False) -> float:
        """
        Compute R-squared or adjusted R-squared.

        Parameters
        ----------
        adjusted : bool, default False
            If True, return adjusted R-squared.

        Returns
        -------
        float
            R-squared value.

        Notes
        -----
        For rank-deficient fits, adjusted R² uses the effective number of
        parameters (excluding dropped columns) for consistency with the
        corrected degrees of freedom.
        """
        self._check_fitted()
        assert self._y is not None
        assert self.residuals_ is not None
        # Use effective params for adjusted R² to match df correction
        n_params = self.n_params_effective_ if adjusted else self.n_params_
        return compute_r_squared(self._y, self.residuals_, adjusted=adjusted, n_params=n_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n, k)
            Design matrix for prediction. Should have same number of columns
            as the original X (excluding intercept if include_intercept=True).

        Returns
        -------
        ndarray
            Predicted values.

        Notes
        -----
        For rank-deficient fits where some coefficients are NaN, predictions
        use only the identified (non-NaN) coefficients. This is equivalent to
        treating dropped columns as having zero coefficients.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        if self.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        # Handle rank-deficient case: use only identified coefficients
        # Replace NaN with 0 so they don't contribute to prediction
        assert self.coefficients_ is not None
        coef = self.coefficients_.copy()
        coef[np.isnan(coef)] = 0.0

        return np.dot(X, coef)


# =============================================================================
# Internal helpers for inference (used by LinearRegression)
# =============================================================================


def _compute_p_value(
    t_stat: float,
    df: Optional[int] = None,
    two_sided: bool = True,
) -> float:
    """
    Compute p-value for a t-statistic.

    Parameters
    ----------
    t_stat : float
        T-statistic.
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.
    two_sided : bool, default True
        Whether to compute two-sided p-value.

    Returns
    -------
    float
        P-value.
    """
    if df is not None and df > 0:
        p_value = stats.t.sf(np.abs(t_stat), df)
    else:
        p_value = stats.norm.sf(np.abs(t_stat))

    if two_sided:
        p_value *= 2

    return float(p_value)


def _compute_confidence_interval(
    estimate: float,
    se: float,
    alpha: float = 0.05,
    df: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute confidence interval for an estimate.

    Parameters
    ----------
    estimate : float
        Point estimate.
    se : float
        Standard error.
    alpha : float, default 0.05
        Significance level (0.05 for 95% CI).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound) of confidence interval.
    """
    if df is not None and df > 0:
        critical_value = stats.t.ppf(1 - alpha / 2, df)
    else:
        critical_value = stats.norm.ppf(1 - alpha / 2)

    lower = estimate - critical_value * se
    upper = estimate + critical_value * se

    return (lower, upper)


def solve_poisson(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
    init_beta: Optional[np.ndarray] = None,
    rank_deficient_action: str = "warn",
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Poisson IRLS (Newton-Raphson with log link).

    Does NOT prepend an intercept — caller must include one if needed.
    Returns (beta, W_final) where W_final = mu_hat (used for sandwich vcov).

    Parameters
    ----------
    X : (n, k) design matrix (caller provides intercept / group FE dummies)
    y : (n,) non-negative count outcomes
    max_iter : maximum IRLS iterations
    tol : convergence threshold on sup-norm of coefficient change
    init_beta : optional starting coefficient vector; if None, zeros are used
        with the first column treated as the intercept and initialized to
        log(mean(y)) to improve convergence for large-scale outcomes.
    rank_deficient_action : {"warn", "error", "silent"}
        How to handle rank-deficient design matrices. Mirrors solve_ols/solve_logit.
    weights : (n,) optional observation weights (e.g. survey sampling weights).
        When provided, the weighted pseudo-log-likelihood is maximised:
        score = X'(w*(y - mu)), Hessian = X'diag(w*mu)X.

    Returns
    -------
    beta : (k,) coefficient vector (NaN for dropped columns if rank-deficient)
    W : (n,) final fitted means mu_hat (weights for sandwich vcov)
    """
    n, k_orig = X.shape

    # Validate weights (mirrors solve_logit validation)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights.shape}")
        if np.any(np.isnan(weights)):
            raise ValueError("weights contain NaN values")
        if np.any(~np.isfinite(weights)):
            raise ValueError("weights contain Inf values")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if np.sum(weights) <= 0:
            raise ValueError("weights sum to zero — no observations have positive weight")

    # Validate rank_deficient_action (same as solve_logit/solve_ols)
    valid_actions = ("warn", "error", "silent")
    if rank_deficient_action not in valid_actions:
        raise ValueError(
            f"rank_deficient_action must be one of {valid_actions}, "
            f"got {rank_deficient_action!r}"
        )

    # Rank-deficiency detection (same pattern as solve_logit/solve_ols)
    kept_cols = np.arange(k_orig)
    rank, dropped_cols, _pivot = _detect_rank_deficiency(X)
    if len(dropped_cols) > 0:
        if rank_deficient_action == "error":
            raise ValueError(
                f"Rank-deficient design matrix: {len(dropped_cols)} collinear columns detected."
            )
        if rank_deficient_action == "warn":
            warnings.warn(
                f"Rank-deficient design matrix: dropping {len(dropped_cols)} of {k_orig} columns. "
                f"Coefficients for these columns are set to NA.",
                UserWarning,
                stacklevel=2,
            )
        dropped_set = set(int(d) for d in dropped_cols)
        kept_cols = np.array([i for i in range(k_orig) if i not in dropped_set], dtype=int)
        if kept_cols.size == 0:
            raise ValueError(
                "Rank-deficient design matrix in Poisson regression collapsed to "
                "rank 0 (no identifiable columns). Cannot fit Poisson model. "
                "Check for constant or fully-collinear covariates."
            )
        X = X[:, kept_cols]

    n, k = X.shape

    # Validate effective weighted sample when weights have zeros
    # (mirrors solve_logit's positive-weight safeguards)
    if weights is not None and np.any(weights == 0):
        pos_mask = weights > 0
        n_pos = int(np.sum(pos_mask))
        X_eff = X[pos_mask]
        eff_rank_info = _detect_rank_deficiency(X_eff)
        if len(eff_rank_info[1]) > 0:
            n_dropped_eff = len(eff_rank_info[1])
            if rank_deficient_action == "error":
                raise ValueError(
                    f"Effective (positive-weight) sample is rank-deficient: "
                    f"{n_dropped_eff} linearly dependent column(s). "
                    f"Cannot identify Poisson model on this subpopulation."
                )
            elif rank_deficient_action == "warn":
                warnings.warn(
                    f"Effective (positive-weight) sample is rank-deficient: "
                    f"dropping {n_dropped_eff} column(s). Poisson estimates "
                    f"may be unreliable on this subpopulation.",
                    UserWarning,
                    stacklevel=2,
                )
            eff_dropped = set(int(d) for d in eff_rank_info[1])
            eff_kept = np.array([i for i in range(k) if i not in eff_dropped], dtype=int)
            if eff_kept.size == 0:
                raise ValueError(
                    "Effective (positive-weight) sample collapsed to rank 0 (no "
                    "identifiable columns). Cannot fit Poisson model on this "
                    "subpopulation. Check for constant or fully-collinear covariates."
                )
            X = X[:, eff_kept]
            if len(dropped_cols) > 0:
                kept_cols = kept_cols[eff_kept]
            else:
                kept_cols = eff_kept
                dropped_cols = list(eff_dropped)
            n, k = X.shape
        if n_pos <= k:
            raise ValueError(
                f"Only {n_pos} positive-weight observation(s) for "
                f"{k} parameters (after rank reduction). "
                f"Cannot identify Poisson model."
            )

    if init_beta is not None:
        beta = init_beta[kept_cols].copy() if len(dropped_cols) > 0 else init_beta.copy()
    else:
        beta = np.zeros(k)
        # Initialise the intercept to log(mean(y)) so the first IRLS step
        # starts near the unconditional mean rather than exp(0)=1, which
        # causes overflow when y is large (e.g. employment levels).
        mean_y = float(np.mean(y))
        if mean_y > 0:
            beta[0] = np.log(mean_y)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -500, 500)
        mu = np.exp(eta)
        if weights is not None:
            score = X.T @ (weights * (y - mu))
            hess = X.T @ ((weights * mu)[:, None] * X)
        else:
            score = X.T @ (y - mu)
            hess = X.T @ (mu[:, None] * X)
        try:
            delta = np.linalg.solve(hess + 1e-12 * np.eye(k), score)
        except np.linalg.LinAlgError:
            warnings.warn(
                "solve_poisson: Hessian is singular at iteration. "
                "Design matrix may be rank-deficient.",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        # Damped step: cap the maximum coefficient change to avoid overshooting
        max_step = np.max(np.abs(delta))
        if max_step > 1.0:
            delta = delta / max_step
        beta_new = beta + delta
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    else:
        warnings.warn(
            "solve_poisson did not converge in {} iterations".format(max_iter),
            RuntimeWarning,
            stacklevel=2,
        )
    mu_final = np.exp(np.clip(X @ beta, -500, 500))

    # Expand back to full size if columns were dropped
    if len(dropped_cols) > 0:
        beta_full = np.full(k_orig, np.nan)
        beta_full[kept_cols] = beta
        beta = beta_full

    return beta, mu_final
