"""Conley (1999) spatial HAC helpers for diff-diff.

This module contains the geographic-distance and kernel helpers that
implement the Conley (1999) spatial heteroskedasticity-and-autocorrelation-
consistent variance estimator. Two operating modes:

**Cross-sectional (single-period or any pooled cross-section):**

    Var̂(β) = (X'X)^{-1} · ( Σ_{i,j} K(d_ij/h) · X_i ε_i ε_j X_j' ) · (X'X)^{-1}

**Panel block-decomposed (matches R conleyreg lag_cutoff > 0):**

    XeeX_spatial = Σ_t  Σ_{i,j in units}   K_space(d_ij/h) · X_{i,t} ε_{i,t} ε_{j,t} X_{j,t}'
    XeeX_serial  = Σ_u  Σ_{|t-s|<=L, t!=s} (1 - |t-s|/(L+1)) · X_{u,t} ε_{u,t} ε_{u,s} X_{u,s}'
    Var̂(β) = (X'X)^{-1} · (XeeX_spatial + XeeX_serial) · (X'X)^{-1}

The block decomposition (NOT a multiplicative product kernel) matches R
``conleyreg`` (Düsterhöft 2021, CRAN v0.1.9) at ~1e-14 on the parity
fixtures. The temporal kernel is hardcoded Bartlett-style regardless of
the ``conley_kernel`` argument, matching ``conleyreg::time_dist``.

The public dispatch (``compute_robust_vcov`` in :mod:`diff_diff.linalg`)
imports :func:`_validate_conley_kwargs` and :func:`_compute_conley_vcov` and
calls them from the ``vcov_type="conley"`` branch. Tests exercise the
inner helpers directly.

Earth radius constant is 6371.01 km (mean radius), matching R
``conleyreg::haversine_dist`` (Düsterhöft 2021, CRAN v0.1.9). See
``benchmarks/R/README.md`` for the cross-language parity convention.

A sparse k-d-tree fast path auto-activates for the spatial component when
``n > _CONLEY_SPARSE_N_THRESHOLD`` (5_000), ``conley_metric`` is one of
``{"haversine", "euclidean"}``, and ``conley_kernel`` is ``"bartlett"``.
The bartlett-only gate is for boundary correctness — bartlett at
``u = 1.0`` returns exactly ``0.0``, so pairs at exactly the cutoff
contribute zero and can be safely dropped by the sparse path. The
uniform kernel returns ``1.0`` at the cutoff and is methodologically
incompatible with the chord-projection roundoff in the haversine query;
it falls back to the dense path. The serial component (within-unit
Bartlett HAC over time) is always dense regardless of n. The
``_CONLEY_DENSE_OOM_WARN_N`` constant (20_000) is a separate
memory-exhaustion warning, retained even when the sparse path activates
because dense fallback still applies for non-bartlett kernels and for
callable metrics.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal, Optional, Union, cast

import numpy as np

# Public type alias for the ``conley_metric`` parameter accepted by
# ``compute_robust_vcov``, ``solve_ols``, and ``LinearRegression``. The
# implementation accepts the two named strings as well as a user-supplied
# callable ``(coords1, coords2) -> n×n distance matrix`` for custom
# (e.g. network) distance metrics. Exported so the public signatures
# in :mod:`diff_diff.linalg` can advertise the full accepted type to
# static checkers and IDEs.
ConleyMetric = Union[
    Literal["haversine", "euclidean"],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
]

# Earth's mean radius (km), matching R conleyreg's haversine convention
# (Düsterhöft 2021, conleyreg::haversine_dist in src/distance_functions.cpp,
# CRAN v0.1.9). WGS-84 equatorial radius is 6378.137 km; the 0.01 km delta
# vs 6371.0 is methodologically negligible (Earth mean radius is approximate
# at many more digits) but matters for the 1e-6 cross-language parity bound.
_CONLEY_EARTH_RADIUS_KM = 6371.01

# Empirical threshold above which the dense O(n²) distance matrix is expected
# to risk OOM on typical commodity workstations (64 GB float64 at n=89_443).
# Renamed from the original ``_CONLEY_DENSE_WARN_N`` to disambiguate it from
# ``_CONLEY_SPARSE_N_THRESHOLD``: this constant is about memory exhaustion;
# the sparse-path threshold is about compute. The two are independent because
# the sparse fast path requires ``conley_kernel='bartlett'`` and a non-callable
# metric — callable metrics and uniform kernels still take the dense path
# at any n.
_CONLEY_DENSE_OOM_WARN_N = 20_000

# Total-n threshold above which the sparse k-d-tree fast path auto-activates
# for the spatial Bartlett meat. Subject to two additional gates: the metric
# must be ``"haversine"`` or ``"euclidean"`` (not callable), and the kernel
# must be ``"bartlett"`` (not ``"uniform"``). Crossed in either direction,
# the dense path is used regardless of n.
_CONLEY_SPARSE_N_THRESHOLD = 5_000

# Density gate: fraction of n*n pairs within cutoff above which the sparse
# CSR matrix's storage overhead (~12 bytes/nnz: data + indices + indptr)
# loses its memory advantage over a dense float64 (8 bytes/cell). The
# break-even is at ~67% density; we gate at 30% (well below break-even)
# to give a comfortable safety margin: at 30% nnz, CSR uses ~45% of
# dense memory. Above 30%, fall back to dense + emit a UserWarning so
# users with large cutoffs aren't surprised by the "sparse" path
# materializing a near-dense matrix. Computed exactly via
# ``cKDTree.count_neighbors`` (O(n log n + nnz), shares tree traversal
# with the subsequent ``query_ball_tree``).
_CONLEY_SPARSE_DENSITY_THRESHOLD = 0.3


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance in km between two sets of points.

    Inputs are in DEGREES. NumPy broadcasting applies, so passing
    ``lat1=lats[:, None]`` and ``lat2=lats[None, :]`` (with matching
    ``lon1``, ``lon2``) yields the full pairwise n×n distance matrix.

    Earth radius is 6371.01 km (mean radius), matching R ``conleyreg``.
    """
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    # Clip for numerical robustness — antipodal pairs can produce a >1 by ~eps
    a = np.clip(a, 0.0, 1.0)
    return _CONLEY_EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


def _validate_callable_metric_result(result: object, n: int) -> np.ndarray:
    """Validate the output of a user-supplied callable ``conley_metric``.

    A user-supplied distance callable must return an ``(n, n)`` matrix of
    finite, non-negative, symmetric values with **zero diagonal**;
    otherwise downstream code produces opaque BLAS errors or silently-
    wrong vcov estimates. This helper performs all six checks at the
    boundary and produces a targeted :class:`ValueError` for each
    failure.

    The zero-diagonal invariant is load-bearing for the Conley sandwich:
    the ``i = j`` term contributes ``K(d_ii / h) · X_i ε_i² X_i'``, which
    must reduce to the HC0 diagonal ``X_i ε_i² X_i'`` (i.e., ``K(0) = 1``).
    A callable with positive self-distance would attenuate the HC0
    contribution by ``K(d_ii / h) < 1`` and silently misstate Conley SEs.
    Built-in metrics (``"haversine"``, ``"euclidean"``) satisfy this by
    construction.

    Returns
    -------
    arr : ndarray of shape (n, n), float64
        The validated distance matrix, ready for kernel evaluation.

    Raises
    ------
    ValueError
        Result cannot cast to ``float64``; shape is not ``(n, n)``;
        contains NaN/inf; contains negative entries; is not symmetric
        within ``atol=1e-10``; or has any nonzero diagonal entry
        ``|d(i, i)| > 1e-10``.
    """
    try:
        arr = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "conley_metric callable returned a value that cannot be cast to "
            f"a float64 array: {exc}."
        ) from exc
    if arr.shape != (n, n):
        raise ValueError(
            "conley_metric callable must return a (n, n) distance matrix; "
            f"got shape {arr.shape}, expected ({n}, {n})."
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            "conley_metric callable returned non-finite entries (NaN or inf); "
            "all pairwise distances must be finite."
        )
    if (arr < 0.0).any():
        raise ValueError(
            "conley_metric callable returned negative entries; all pairwise "
            "distances must be non-negative."
        )
    asymmetry = float(np.max(np.abs(arr - arr.T))) if arr.size else 0.0
    if asymmetry > 1e-10:
        raise ValueError(
            "conley_metric callable returned an asymmetric matrix; the "
            "distance matrix must satisfy d(i, j) = d(j, i). Max |D - D.T| = "
            f"{asymmetry:.2e}, tolerance 1e-10."
        )
    diag_max = float(np.max(np.abs(np.diag(arr)))) if arr.size else 0.0
    if diag_max > 1e-10:
        raise ValueError(
            "conley_metric callable returned a matrix with nonzero self-"
            "distance on the diagonal; the Conley sandwich requires "
            "d(i, i) = 0 so the kernel reduces to K(0) = 1 on the HC0 "
            f"diagonal contribution. Max |diag(D)| = {diag_max:.2e}, "
            "tolerance 1e-10."
        )
    return arr


def _validate_conley_estimator_inputs(
    *,
    estimator_name: str,
    data: "Any",
    unit: Optional[str],
    conley_coords: object,
    conley_cutoff_km: Optional[float],
    conley_lag_cutoff: Optional[int],
    survey_design: object,
    inference: str,
    cluster: Optional[str] = None,
) -> None:
    """Shared front-door validation for ``vcov_type='conley'`` on the
    estimator entry points (``DifferenceInDifferences``, ``MultiPeriodDiD``,
    ``TwoWayFixedEffects``, ``SunAbraham``, and ``WooldridgeDiD``-OLS).

    Each estimator's ``fit()`` calls this BEFORE building Conley arrays or
    threading them into the variance computation. The eight checks below
    are the union of what each estimator needs; estimator-specific bits
    (e.g. building the array from a column name) remain inline at the
    caller. Centralizing this in one place prevents the validation-class
    drift that surfaced repeatedly across Wave A CI rounds (front-door
    guards for `unit`, `conley_coords`, and `cluster` were each missing
    on at least one estimator surface and required separate fixes).

    Parameters
    ----------
    estimator_name : str
        Class name surfaced in error messages (e.g.
        ``"DifferenceInDifferences"``).
    data : pandas.DataFrame
        The dataset passed to ``fit()``. Used to check column existence
        for ``unit``, ``conley_coords``, and ``cluster``. Typed as
        ``Any`` here to avoid importing pandas at module load time.
    unit : str or None
        Name of the unit identifier column (required for Conley).
    conley_coords : tuple/list/None
        Pair of column names ``(lat_col, lon_col)``.
    conley_cutoff_km : float or None
        Positive finite bandwidth.
    conley_lag_cutoff : int or None
        Non-negative integer lag for the within-unit Bartlett serial sum.
    survey_design : object
        ``SurveyDesign`` instance or ``None``. Survey + Conley is deferred.
    inference : str
        Estimator inference mode. ``"wild_bootstrap"`` + Conley is rejected.
    cluster : str or None, default None
        Name of the cluster column if the user opted into the combined
        spatial + cluster product kernel (Wave A #119). When non-None,
        the column must exist in ``data``; otherwise the downstream
        ``data[cluster]`` access would raise an opaque pandas
        ``KeyError``. Pass ``None`` (the default) to skip this check.

    Raises
    ------
    ValueError
        Missing/malformed conley_coords, missing conley_cutoff_km,
        missing/unknown unit, missing conley_lag_cutoff, coord column
        not in ``data``, or ``cluster`` set to a name not in ``data``.
    NotImplementedError
        ``survey_design`` is non-None, or ``inference == "wild_bootstrap"``.
    """
    if conley_coords is None or conley_cutoff_km is None:
        raise ValueError(
            f"{estimator_name}(vcov_type='conley') requires "
            "conley_coords=(lat_col, lon_col) and conley_cutoff_km "
            "on the constructor."
        )
    if (
        not isinstance(conley_coords, (tuple, list))
        or len(conley_coords) != 2
        or not all(isinstance(c, str) for c in conley_coords)
    ):
        raise ValueError(
            "conley_coords must be a 2-element tuple/list of column "
            f"names (lat_col, lon_col); got {conley_coords!r}."
        )
    for _coord_col in conley_coords:
        if _coord_col not in data.columns:
            raise ValueError(f"conley_coords column '{_coord_col}' not found in data.")
    if unit is None:
        raise ValueError(
            f"{estimator_name}(vcov_type='conley') requires `unit=<column_name>` "
            "— the panel block-decomposed Conley sandwich needs the unit "
            "identifier to compute the per-unit serial sum."
        )
    if unit not in data.columns:
        raise ValueError(f"Unit column '{unit}' not found in data")
    if conley_lag_cutoff is None:
        raise ValueError(
            f"{estimator_name}(vcov_type='conley') requires conley_lag_cutoff "
            "(non-negative int; 0 means spatial-within-period only, no serial "
            "component). See R conleyreg's `lag_cutoff` argument for the convention."
        )
    if cluster is not None and cluster not in data.columns:
        raise ValueError(f"Cluster column '{cluster}' not found in data")
    if survey_design is not None:
        raise NotImplementedError(
            f"{estimator_name}(vcov_type='conley') + survey_design is "
            "deferred — weighted spatial-HAC under probability sampling "
            "is an open methodological question; no canonical extension "
            "of Conley (1999) exists for the combination. Drop "
            "survey_design for unweighted Conley (cross-sectional or panel "
            "block-decomposed via conley_lag_cutoff > 0), or use "
            "vcov_type='hc1' for survey-aware cluster-robust without "
            "spatial HAC."
        )
    if inference == "wild_bootstrap":
        raise NotImplementedError(
            f"{estimator_name}(vcov_type='conley', inference='wild_bootstrap') "
            "is not supported: the wild bootstrap is a separate inference path "
            "that does not consume the analytical Conley sandwich. Use "
            "inference='analytical' for Conley SEs."
        )


def _pairwise_distance_matrix(coords: np.ndarray, metric: ConleyMetric) -> np.ndarray:
    """Build the dense n×n pairwise distance matrix.

    ``metric`` is one of ``"haversine"`` (lat/lon in degrees, distance in km),
    ``"euclidean"`` (any units), or a callable ``f(coords1, coords2) -> n×n``.

    For the callable branch, the result is validated via
    :func:`_validate_callable_metric_result`: shape ``(n, n)``, finite,
    non-negative, symmetric within ``atol=1e-10``. Each failure raises
    :class:`ValueError` naming the violated invariant.
    """
    if metric == "haversine":
        lats = coords[:, 0]
        lons = coords[:, 1]
        return _haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
    if metric == "euclidean":
        # Vectorized via squared-distance identity; avoids scipy import path
        # while matching scipy.spatial.distance.cdist to ~1e-14
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if callable(metric):
        return _validate_callable_metric_result(metric(coords, coords), coords.shape[0])
    raise ValueError(
        f"conley_metric must be 'haversine', 'euclidean', or callable; got {metric!r}."
    )


def _bartlett_kernel(u: np.ndarray) -> np.ndarray:
    """Bartlett (linear taper) kernel on pairwise distance: K(u) = max(0, 1 - |u|).

    This is the radial 1-D specialization of Conley (1999)'s Bartlett window
    that R ``conleyreg`` (Düsterhöft 2021), Stata ``acreg`` (Colella et al.
    2019), and Hsiang (2010) all use as their Bartlett path. Conley's
    explicit PSD formula (Eq 3.14, page 12) is the **2-D separable product
    window** ``K(j, k) = (1 - |j|/L_M)(1 - |k|/L_N)`` indexed on a lattice;
    the 1-D radial form on pairwise distance is a practitioner specialization
    that is not explicitly written in the paper and is therefore **not
    PSD-guaranteed**. The caller checks the resulting meat for indefiniteness
    and emits a ``UserWarning`` if the smallest eigenvalue is materially
    negative (regardless of kernel).
    """
    return np.maximum(0.0, 1.0 - np.abs(u))


def _uniform_kernel(u: np.ndarray) -> np.ndarray:
    """Uniform (truncated) kernel: K(u) = 1 if |u| <= 1 else 0.

    Cited as White (1980) truncated estimator; Conley (1999) page 11. Easier
    to interpret than Bartlett but the spectral window is negative in regions
    (Conley 1999 footnote 11), so the resulting meat is not guaranteed PSD.
    Caller emits ``UserWarning`` if any meat eigenvalue is materially negative.
    """
    return (np.abs(u) <= 1.0).astype(np.float64)


def _serial_bartlett_kernel_matrix(t_codes: np.ndarray, L: int) -> np.ndarray:
    """Within-unit Newey-West (1987) Bartlett HAC kernel matrix for serial
    correlation in panel data, indexed by panel-wide dense time codes.

    Returns the K matrix with ``K[i, j] = 1 - |t_i - t_j| / (L + 1)`` for
    ``0 < |t_i - t_j| <= L``, else 0. The lag-0 diagonal is excluded so
    callers can add this to a spatial within-period meat without
    double-counting the diagonal.

    Uses the 1-D radial pairwise form (matches conleyreg::time_dist), NOT
    Conley 1999 Eq 3.14's 2-D separable product window — see the methodology
    lock at :func:`_compute_conley_meat` for context.
    """
    t = t_codes.astype(np.float64, copy=False)
    lag_mat = np.abs(t[:, None] - t[None, :])
    return ((lag_mat <= L) & (lag_mat != 0)).astype(np.float64) * (1.0 - lag_mat / (L + 1.0))


def _validate_meat_psd(
    M: np.ndarray,
    *,
    error_msg: str,
    warning_template: str,
    stacklevel: int = 3,
) -> None:
    """Finite + PSD guard for sandwich meat matrices. Raises ``ValueError``
    on non-finite entries; warns ``UserWarning`` when ``min(eigvalsh(M)) <
    -1e-12``.

    Parameters
    ----------
    error_msg
        Message passed to ``ValueError`` on non-finite entries.
    warning_template
        Format string for the negative-eigenvalue warning. May contain an
        ``{eigval}`` placeholder; the caller embeds ``{eigval:.2e}`` directly
        in the template so the helper formats the minimum eigenvalue with
        scientific notation.
    stacklevel
        Frame count from inside ``_validate_meat_psd``: ``stacklevel=N``
        attributes the warning to the Nth caller above the helper itself.
        Default 3 covers a single intermediate frame (the helper's direct
        caller's caller); pass an explicit value matching call-site depth.
    """
    if not np.all(np.isfinite(M)):
        raise ValueError(error_msg)
    eigvals = np.linalg.eigvalsh(M)
    if eigvals.size and eigvals.min() < -1e-12:
        warnings.warn(
            warning_template.format(eigval=eigvals.min()),
            UserWarning,
            stacklevel=stacklevel,
        )


def _compute_spatial_bartlett_meat_sparse(
    S: np.ndarray,
    coords: np.ndarray,
    cutoff: float,
    metric: str,
    *,
    cluster_codes: Optional[np.ndarray] = None,
    density_threshold: float = _CONLEY_SPARSE_DENSITY_THRESHOLD,
) -> Optional[np.ndarray]:
    """Sparse k-d-tree-based spatial Bartlett meat: ``S.T @ K_bartlett @ S``.

    Used by :func:`_compute_conley_vcov` when ``_conley_sparse`` is True or
    when the auto-toggle fires (n above the threshold, bartlett kernel,
    non-callable metric). Avoids materializing the full n×n distance matrix
    by using ``scipy.spatial.cKDTree.query_ball_tree`` to find in-range
    neighbor pairs and assembling a CSR sparse kernel matrix on top of those.

    Parameters
    ----------
    S : (n, k) array
        Score matrix ``X * residuals[:, None]``.
    coords : (n, 2) array of float64
        ``[lat, lon]`` (haversine) or arbitrary 2-D coordinates (euclidean).
    cutoff : float
        Conley bandwidth (km for haversine; same units as ``coords`` for
        euclidean).
    metric : {"haversine", "euclidean"}
        Distance metric. Callable metrics fall back to the dense path and
        are not supported here — the caller is expected to enforce that.
    cluster_codes : (n,) int array, optional, keyword-only
        Integer cluster codes (one per row of ``S``). When supplied, the
        combined kernel ``K_total[i, j] = K_space(d_ij/h) · 1{cluster_i =
        cluster_j}`` is applied: neighbors are filtered to same-cluster
        pairs before the kernel evaluation.

    Returns
    -------
    meat : (k, k) array

    Notes
    -----
    For haversine, lat/lon is projected to a 3-D unit-sphere Cartesian
    point cloud (``x = cos(lat) cos(lon), y = cos(lat) sin(lon), z = sin(lat)``)
    so that the kd-tree's euclidean radius query corresponds to a chord
    distance on the unit sphere. The chord radius is
    ``2 sin(cutoff_km / (2 R_earth))`` (the chord on the unit sphere
    matching the great-circle arc of length ``cutoff_km``). A small
    relative epsilon (``1 + 1e-12``) is added to the query radius to
    absorb chord-projection roundoff; after the query, the exact
    great-circle distance is recomputed via :func:`_haversine_km` and
    the Bartlett kernel is applied. Pairs at exactly the cutoff distance
    contribute zero to the kernel (bartlett at ``u = 1.0`` is exactly
    ``0.0``) and are filtered out, which is why the sparse path is
    bartlett-only — uniform at ``u = 1.0`` returns ``1.0`` and would
    require querying with a strict superset and applying the closed-
    interval kernel by hand.

    Numerical tolerance vs the dense path on the same inputs is
    typically ~1e-12 in absolute terms; the haversine→chord projection
    introduces sub-eps error that propagates through the matmul.
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    n = coords.shape[0]
    k = S.shape[1]

    if metric == "haversine":
        lat_r = np.radians(coords[:, 0])
        lon_r = np.radians(coords[:, 1])
        xyz = np.column_stack(
            [
                np.cos(lat_r) * np.cos(lon_r),
                np.cos(lat_r) * np.sin(lon_r),
                np.sin(lat_r),
            ]
        )
        # Chord on the unit sphere matching arc-length cutoff_km. Clamp the
        # arc to π radians (half-Earth circumference) before computing the
        # chord: the function `arc -> 2 sin(arc/2)` is monotonically
        # increasing only on [0, π] and reaches its global max of 2 (the
        # unit-sphere diameter) at arc = π. Without the clamp, cutoffs
        # above π·R (~20,015 km) would compute a SMALLER chord radius than
        # at cutoff = π·R and silently drop antipodal pairs that still have
        # positive Bartlett weight. The dense path saturates at π·R via
        # `_haversine_km`'s `np.clip(a, 0, 1)` (arcsin caps at π/2), so the
        # clamp mirrors that saturation. The 1+1e-12 epsilon then absorbs
        # chord-projection float roundoff at sub-cutoff distances.
        arc_radians = min(cutoff / _CONLEY_EARTH_RADIUS_KM, np.pi)
        query_r = 2.0 * np.sin(arc_radians / 2.0)
        query_r *= 1.0 + 1e-12
        tree = cKDTree(xyz)
        tree_coords = xyz  # used for per-cluster sub-trees in the density gate
    elif metric == "euclidean":
        # Small relative epsilon for symmetry with the haversine branch.
        # Bartlett's <=-vs-< boundary is moot since kernel is exactly 0 at u=1.
        query_r = cutoff * (1.0 + 1e-12)
        tree = cKDTree(coords)
        tree_coords = coords
    else:
        raise ValueError(
            "sparse Conley path requires metric in {'haversine', 'euclidean'}; "
            f"got {metric!r}. (Callable metrics fall back to the dense path.)"
        )

    # Density gate: count in-range pairs cheaply via the same tree (shares
    # traversal cost with query_ball_tree, no extra allocation). If density
    # exceeds the threshold, return None so the caller falls back to dense
    # (avoids materializing a near-dense CSR matrix that would use MORE
    # memory than dense float64). Codex CI R6 P2.
    n_pairs_in_range = int(tree.count_neighbors(tree, r=query_r, p=2.0))
    density = n_pairs_in_range / float(n * n)
    if density > density_threshold and cluster_codes is not None:
        # Cluster-aware refinement: the actual CSR nnz reflects within-
        # cluster pairs only (the inner loop drops cross-cluster neighbors
        # before adding to the sparse matrix). On large clustered panels,
        # spatial density can be ~100% while within-cluster density is
        # tiny — without refinement we'd spuriously fall back to dense
        # exactly when sparse helps most. Refine by counting per-cluster.
        # Codex CI R8 P1: the unrefined density check was cluster-blind.
        refined_pairs = 0
        for c in np.unique(cluster_codes):
            in_c = cluster_codes == c
            n_c = int(in_c.sum())
            if n_c <= 1:
                # Singleton cluster contributes only the diagonal self-pair.
                refined_pairs += n_c
                continue
            sub_tree = cKDTree(tree_coords[in_c])
            refined_pairs += int(sub_tree.count_neighbors(sub_tree, r=query_r, p=2.0))
        density = refined_pairs / float(n * n)
    if density > density_threshold:
        warnings.warn(
            f"Conley sparse path: neighbor density {density:.1%} exceeds "
            f"threshold {density_threshold:.1%}; falling back to dense "
            "(CSR storage would use more memory than the dense float64 "
            "matrix at this density). Consider a smaller conley_cutoff_km "
            "for genuine memory savings.",
            UserWarning,
            stacklevel=3,
        )
        return None

    neighbors = tree.query_ball_tree(tree, r=query_r, p=2.0)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []
    for i, neigh in enumerate(neighbors):
        if not neigh:
            # Should not happen — i is always in its own ball — but guard.
            continue
        neigh_arr = np.asarray(neigh, dtype=np.intp)
        # Combined spatial + cluster product kernel: drop neighbors that
        # don't share i's cluster. This is the sparse-path analog of the
        # dense path's Hadamard-with-cluster-mask step.
        if cluster_codes is not None:
            same_cluster = cluster_codes[neigh_arr] == cluster_codes[i]
            neigh_arr = neigh_arr[same_cluster]
            if neigh_arr.size == 0:
                continue
        # Recompute exact distance for the in-range neighbors (NOT chord
        # for haversine — we apply the kernel to actual great-circle km).
        if metric == "haversine":
            dist = _haversine_km(
                np.asarray(coords[i, 0]),
                np.asarray(coords[i, 1]),
                coords[neigh_arr, 0],
                coords[neigh_arr, 1],
            )
        else:
            diff = coords[neigh_arr] - coords[i]
            dist = np.sqrt(np.sum(diff * diff, axis=-1))
        kernel_vals = _bartlett_kernel(dist / cutoff)
        nonzero_mask = kernel_vals > 0.0
        if not nonzero_mask.any():
            continue
        nonzero_neighbors = neigh_arr[nonzero_mask]
        nonzero_kernels = kernel_vals[nonzero_mask]
        rows_list.append(np.full(nonzero_neighbors.shape, i, dtype=np.intp))
        cols_list.append(nonzero_neighbors)
        data_list.append(nonzero_kernels)

    if not data_list:
        return np.zeros((k, k))

    K = csr_matrix(
        (np.concatenate(data_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(n, n),
    )
    # scipy sparse @ dense returns dense; do K @ S first then S.T @ that.
    return S.T @ (K @ S)


def _validate_conley_kwargs(
    coords: Optional[np.ndarray],
    cutoff: Optional[float],
    metric: ConleyMetric,
    kernel: str,
    n: int,
    *,
    time: Optional[np.ndarray] = None,
    unit: Optional[np.ndarray] = None,
    lag_cutoff: Optional[int] = None,
    cluster_ids: Optional[np.ndarray] = None,
) -> None:
    """Validate the Conley kwargs against the design's row count.

    The first five positional args define the cross-sectional contract (Phase 1).
    The three keyword-only ``time`` / ``unit`` / ``lag_cutoff`` args define the
    panel block-decomposed contract (Phase 2); they are three-way co-required.
    The ``cluster_ids`` keyword-only arg enables the combined spatial +
    cluster product kernel (Wave A item #119); on the panel block-decomposed
    path it must be **constant within each unit across periods** (otherwise
    the within-unit serial mask would zero out adjacent-time pairs and
    produce a methodologically-muddled meat).

    Raises
    ------
    ValueError
        Missing/malformed coords or cutoff; lat/lon out of range under
        haversine; unknown kernel/metric; non-finite or non-positive cutoff;
        partial panel arg set (must pass all three of time/unit/lag_cutoff or none);
        ``cluster_ids`` with wrong shape or NaN; ``cluster_ids`` not constant
        within each unit on the panel path.

    Warnings
    --------
    UserWarning
        Emitted when ``n > 20_000`` to flag the dense O(n²) memory cost.
    """
    if coords is None:
        raise ValueError(
            "vcov_type='conley' requires conley_coords (n×2 array of [lat, lon] "
            "or projected coords). Pass via LinearRegression(conley_coords=...) "
            "or compute_robust_vcov(conley_coords=...)."
        )
    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError(f"conley_coords must be a 2-D (n, 2) array; got shape {coords_arr.shape}.")
    if coords_arr.shape[0] != n:
        raise ValueError(f"conley_coords has {coords_arr.shape[0]} rows but X has {n} rows.")
    if not np.isfinite(coords_arr).all():
        raise ValueError("conley_coords contains NaN or inf values.")

    if cutoff is None:
        raise ValueError(
            "vcov_type='conley' requires conley_cutoff_km (a positive finite "
            "bandwidth). No defensible default — see Conley 1999 Section 5 "
            "for the sensitivity-grid recommendation."
        )
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"conley_cutoff_km must be a positive finite number; got {cutoff!r}.")

    if not (metric in ("haversine", "euclidean") or callable(metric)):
        raise ValueError(
            f"conley_metric must be 'haversine', 'euclidean', or callable; got {metric!r}."
        )

    # Lat/lon range checks under haversine. Skipped for euclidean (user's
    # responsibility to pass projected coords with consistent units) and
    # for callables (user supplies their own distance function).
    if metric == "haversine":
        if not ((coords_arr[:, 0] >= -90.0) & (coords_arr[:, 0] <= 90.0)).all():
            raise ValueError(
                "conley_metric='haversine' requires latitude in [-90, 90]; "
                f"got min={coords_arr[:, 0].min()}, max={coords_arr[:, 0].max()}."
            )
        if not ((coords_arr[:, 1] >= -180.0) & (coords_arr[:, 1] <= 180.0)).all():
            raise ValueError(
                "conley_metric='haversine' requires longitude in [-180, 180]; "
                f"got min={coords_arr[:, 1].min()}, max={coords_arr[:, 1].max()}."
            )

    if kernel not in ("bartlett", "uniform"):
        raise ValueError(f"conley_kernel must be 'bartlett' or 'uniform'; got {kernel!r}.")

    # Panel block-decomposed contract: time / unit / lag_cutoff are three-way co-required.
    panel_flags = (time is not None, unit is not None, lag_cutoff is not None)
    n_panel_set = sum(panel_flags)
    if n_panel_set not in (0, 3):
        raise ValueError(
            "conley_time, conley_unit, and conley_lag_cutoff must all be passed "
            "together (all None for cross-sectional Conley; all set for the "
            "panel block-decomposed form). Got "
            f"conley_time set={panel_flags[0]}, conley_unit set={panel_flags[1]}, "
            f"conley_lag_cutoff set={panel_flags[2]}."
        )
    if n_panel_set == 3:
        time_arr = np.asarray(time)
        if time_arr.ndim != 1 or time_arr.shape[0] != n:
            raise ValueError(
                f"conley_time must be a 1-D array of length {n}; got shape {time_arr.shape}."
            )
        # Use pandas.isna for object/categorical dtypes; np.isfinite catches
        # NaN/inf for numeric dtypes only.
        import pandas as _pd

        if _pd.isna(time_arr).any():
            raise ValueError("conley_time contains NaN or missing values.")
        if time_arr.dtype.kind in "fc" and not np.isfinite(time_arr).all():
            raise ValueError("conley_time contains NaN or inf values.")
        unit_arr = np.asarray(unit)
        if unit_arr.ndim != 1 or unit_arr.shape[0] != n:
            raise ValueError(
                f"conley_unit must be a 1-D array of length {n}; got shape {unit_arr.shape}."
            )
        # conley_unit may be any hashable (int, string, categorical). Use
        # pandas.isna for cross-dtype NA detection. NaN unit IDs would silently
        # drop those rows from the per-unit serial HAC sum at
        # `np.unique(unit_arr) + mask_u = unit_arr == u_val`.
        if _pd.isna(unit_arr).any():
            raise ValueError("conley_unit contains NaN or missing values.")
        if not (isinstance(lag_cutoff, (int, np.integer)) and int(lag_cutoff) >= 0):
            raise ValueError(
                f"conley_lag_cutoff must be a non-negative integer; got {lag_cutoff!r}."
            )

    # Combined spatial + cluster product kernel (Wave A #119). The validator
    # checks shape, missing values, and — on the panel block-decomposed path
    # only — that cluster membership is constant within each unit across
    # periods. Cross-sectional path has no time dimension, so no
    # invariance constraint applies.
    if cluster_ids is not None:
        cluster_arr = np.asarray(cluster_ids)
        if cluster_arr.ndim != 1 or cluster_arr.shape[0] != n:
            raise ValueError(
                f"cluster_ids must be a 1-D array of length {n} when combined "
                f"with vcov_type='conley'; got shape {cluster_arr.shape}."
            )
        import pandas as _pd

        if _pd.isna(cluster_arr).any():
            raise ValueError(
                "cluster_ids contains NaN or missing values when combined "
                "with vcov_type='conley'."
            )
        if n_panel_set == 3:
            # Panel path: enforce time-invariance within unit. If a unit's
            # cluster changes across periods, the within-unit serial mask
            # would zero out adjacent-time pairs that should contribute.
            cluster_codes_v, _ = _pd.factorize(cluster_arr)
            unit_arr_local = np.asarray(unit)
            unit_codes_v, _ = _pd.factorize(unit_arr_local)
            # Count distinct cluster codes per unit. >1 means the cluster
            # assignment varies across time within at least one unit.
            df_check = _pd.DataFrame({"u": unit_codes_v, "c": cluster_codes_v})
            unit_n_clusters = df_check.groupby("u")["c"].nunique()
            violator_mask = unit_n_clusters.to_numpy() > 1
            if bool(violator_mask.any()):
                # Report up to 5 violating unit labels to help debugging.
                violator_codes = np.asarray(unit_n_clusters.index)[violator_mask]
                uniques_unit = _pd.unique(unit_arr_local)
                shown = [str(uniques_unit[int(c)]) for c in violator_codes[:5]]
                total = int(violator_mask.sum())
                more = "" if total <= 5 else f" (+{total - 5} more)"
                raise ValueError(
                    "cluster_ids combined with the panel block-decomposed "
                    "Conley path (conley_lag_cutoff set) must be constant "
                    "within each unit across periods; the within-unit serial "
                    "Bartlett HAC's mask is trivially all-ones only under "
                    "this contract. Violating units: "
                    f"{', '.join(shown)}{more}. Either pass a "
                    "time-invariant cluster (e.g. unit-level region) or "
                    "drop cluster_ids."
                )

    if n > _CONLEY_DENSE_OOM_WARN_N:
        memory_gb = (n * n * 8) / 1e9
        # The sparse k-d-tree fast path will auto-activate for the spatial
        # meat when n > _CONLEY_SPARSE_N_THRESHOLD AND metric is haversine
        # or euclidean AND kernel is bartlett. Above the OOM warning
        # threshold the dense fallback (callable metric, uniform kernel)
        # is at material memory risk.
        if (
            isinstance(metric, str)
            and metric in ("haversine", "euclidean")
            and kernel == "bartlett"
        ):
            warnings.warn(
                f"vcov_type='conley': the sparse k-d-tree fast path will be "
                f"used for the spatial Bartlett meat (n={n}, kernel='bartlett', "
                f"metric={metric!r}); the per-unit serial Bartlett HAC remains "
                f"dense. The dense {n}x{n} fallback would be ~{memory_gb:.1f} GB "
                "float64.",
                UserWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"vcov_type='conley' builds a dense {n}x{n} distance matrix "
                f"(~{memory_gb:.1f} GB float64). The sparse k-d-tree fast path "
                "requires conley_kernel='bartlett' and conley_metric in "
                "{'haversine', 'euclidean'}; "
                f"current kernel={kernel!r}, metric={metric!r}.",
                UserWarning,
                stacklevel=3,
            )


def _compute_conley_meat(
    scores: np.ndarray,
    coords: np.ndarray,
    cutoff: float,
    metric: ConleyMetric,
    kernel: str,
    *,
    time: Optional[np.ndarray] = None,
    unit: Optional[np.ndarray] = None,
    lag_cutoff: Optional[int] = None,
    cluster_ids: Optional[np.ndarray] = None,
    _conley_sparse: Optional[bool] = None,
) -> np.ndarray:
    """Conley (1999) spatial HAC meat — ``scores' K scores`` for the product kernel.

    Factors the kernel-construction-and-application step out of
    :func:`_compute_conley_vcov` so a second caller (SpilloverDiD's Wave D
    Gardner GMM first-stage correction) can reuse the cross-sectional /
    panel-block / sparse k-d-tree / cluster-product code path with an
    arbitrary score matrix substituted for the canonical ``X * residuals``.

    Two operating modes:

    **Cross-sectional** (``time`` / ``unit`` / ``lag_cutoff`` all None):

        meat = Σ_{i,j} K(d_ij/h) · scores_i scores_j'

    Implemented via ``meat = scores' K scores``.

    **Panel block-decomposed** (all three keyword-only args set):

        meat_spatial = Σ_t  scores_t' · K_space_t · scores_t   (within-period sum)
        meat_serial  = Σ_u  scores_u' · K_time_u · scores_u    if lag_cutoff > 0
        meat = meat_spatial + meat_serial

    The serial Bartlett kernel ``K_time_u[i, j] = 1{|t_i-t_j| <= L, i != j} ·
    (1 - |t_i-t_j|/(L+1))`` is hardcoded regardless of the user-supplied
    ``kernel`` argument, matching R ``conleyreg::time_dist``. The ``lag != 0``
    exclusion avoids double-counting the diagonal already covered by the
    spatial component.

    **Combined spatial + cluster product kernel** (``cluster_ids`` supplied):

        K_total[i, j] = K_space(d_ij/h) · 1{cluster_i = cluster_j}

    The cluster indicator multiplies the spatial kernel on the within-period
    (or full cross-sectional) sandwich. On the panel path, the validator
    enforces that ``cluster_ids`` is constant within each unit across
    periods, which guarantees the within-unit serial mask is trivially
    all-ones — no explicit per-unit-time mask needed in the serial loop.

    Inputs are assumed already validated by :func:`_validate_conley_kwargs`;
    the helper only does the math. Caller is responsible for the validator.

    Emits a ``UserWarning`` if the smallest meat eigenvalue is materially
    negative (< -1e-12) — radial 1-D Bartlett and uniform kernels are
    practitioner specializations of Conley 1999 and are not formally
    PSD-guaranteed.

    Returns
    -------
    meat : ndarray of shape (p, p)
    """
    coords_arr = np.asarray(coords, dtype=np.float64)
    n = scores.shape[0]

    # Factorize cluster_ids once if supplied, so per-slice mask construction
    # below can use integer comparisons rather than re-factorizing per call.
    cluster_codes: Optional[np.ndarray] = None
    if cluster_ids is not None:
        import pandas as _pd

        cluster_codes = np.asarray(_pd.factorize(np.asarray(cluster_ids))[0], dtype=np.int64)

    def _kernel_fn(u: np.ndarray) -> np.ndarray:
        if kernel == "bartlett":
            return _bartlett_kernel(u)
        if kernel == "uniform":
            return _uniform_kernel(u)
        raise ValueError(f"conley_kernel must be 'bartlett' or 'uniform'; got {kernel!r}.")

    # Decide whether to use the sparse k-d-tree path for the spatial component.
    # Three gates: a supported metric (haversine/euclidean, NOT callable), a
    # supported kernel (bartlett — see module docstring for the boundary-
    # semantics rationale), and either the auto-toggle threshold or an
    # explicit override via the private _conley_sparse kwarg. When explicit
    # forcing requests sparse on an unsupported metric/kernel, we raise
    # rather than silently fall back so the caller sees the mismatch.
    # Use direct equality (NOT isinstance(metric, str)) so pyright can keep
    # the Literal narrowing on the dense fallback path below — checking
    # `isinstance(metric, str)` widens metric from ConleyMetric to
    # str | Callable and breaks downstream typing on `_pairwise_distance_matrix`.
    metric_supports_sparse = metric == "haversine" or metric == "euclidean"
    kernel_supports_sparse = kernel == "bartlett"
    if _conley_sparse is True:
        if not (metric_supports_sparse and kernel_supports_sparse):
            raise ValueError(
                "_conley_sparse=True requires conley_metric in "
                "{'haversine', 'euclidean'} and conley_kernel='bartlett'; "
                f"got metric={metric!r}, kernel={kernel!r}."
            )
        use_sparse = True
    elif _conley_sparse is False:
        use_sparse = False
    else:
        use_sparse = (
            n > _CONLEY_SPARSE_N_THRESHOLD and metric_supports_sparse and kernel_supports_sparse
        )

    def _spatial_meat_for_mask(mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the spatial meat ``scores' K scores`` for the given subset.

        ``mask`` may be ``None`` (use all rows) or a boolean array of length n.
        Dispatches to the sparse helper when ``use_sparse`` is True, otherwise
        builds the dense n×n distance matrix. When ``cluster_codes`` is set,
        the cluster indicator multiplies the spatial kernel on this slice
        (per-slice mask, NOT full n×n — saves memory for panel paths).
        """
        if mask is None:
            scores_sub = scores
            coords_sub = coords_arr
            cluster_sub = cluster_codes
        else:
            scores_sub = scores[mask]
            coords_sub = coords_arr[mask]
            cluster_sub = cluster_codes[mask] if cluster_codes is not None else None
        if use_sparse:
            # The auto-toggle and explicit-True gate above both guarantee
            # metric is a str (haversine/euclidean), so this cast is safe;
            # using cast() avoids leaking the narrowing into the dense
            # fallback below. The sparse helper returns None when the
            # neighbor density exceeds the threshold (sparse CSR storage
            # would use more memory than dense); fall through to dense in
            # that case (the warning is already emitted by the helper).
            sparse_meat = _compute_spatial_bartlett_meat_sparse(
                scores_sub, coords_sub, cutoff, cast(str, metric), cluster_codes=cluster_sub
            )
            if sparse_meat is not None:
                return sparse_meat
            # Density-gated fallback: continue to the dense path below.
        D = _pairwise_distance_matrix(coords_sub, metric)
        K = _kernel_fn(D / cutoff)
        if cluster_sub is not None:
            cluster_mask = cluster_sub[:, None] == cluster_sub[None, :]
            K = K * cluster_mask
        return scores_sub.T @ K @ scores_sub

    # Suppress spurious BLAS-level "divide by zero / overflow" warnings on
    # macOS Accelerate when K is sparse-ish (most off-diagonals are exactly
    # 0 outside the cutoff). The matmul result is mathematically correct;
    # the warning is a subnormal-handling false-positive in the AVX path.
    # We verify finiteness immediately after.
    if time is None:
        # Cross-sectional path: full n×n spatial sandwich.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            meat = _spatial_meat_for_mask(None)
    else:
        # Panel block-decomposed path (matches R conleyreg).
        time_arr = np.asarray(time)
        unit_arr = np.asarray(unit)
        # Normalize time labels to dense panel-period codes (0..T-1) so that
        # `conley_lag_cutoff` always refers to a count of panel periods, not
        # a raw-label difference. Works for any orderable encoding (year ints
        # like 2020/2021, YYYYMM like 202012/202101, datetime64, pd.Period,
        # strings). np.unique returns sorted unique values and `return_inverse`
        # gives the position of each row in that sort.
        # **Note (deviation from R-conleyreg literal):** R conleyreg uses raw
        # `time` values for the lag computation directly, which silently
        # mishandles non-dense encodings (e.g. lag 1 between 202012 and 202101
        # is 89 in raw integer differences). diff-diff normalizes first so
        # `conley_lag_cutoff` is meaningfully a "number of observed panel
        # periods" regardless of label scale.
        _, time_codes = np.unique(time_arr, return_inverse=True)
        p = scores.shape[1]
        meat = np.zeros((p, p))
        # Spatial component: within-period sandwich, summed across periods.
        # _spatial_meat_for_mask dispatches to sparse or dense per the toggle.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for t_code in np.unique(time_codes):
                mask_t = time_codes == t_code
                meat += _spatial_meat_for_mask(mask_t)
            # Serial component: within-unit Bartlett HAC for lag in {1..L},
            # excluding lag=0 to avoid double-counting the spatial diagonal.
            # Bartlett form hardcoded (matches conleyreg::time_dist).
            # The validator guarantees lag_cutoff is not None when time is not None.
            assert lag_cutoff is not None
            L = int(lag_cutoff)
            if L > 0:
                for u_val in np.unique(unit_arr):
                    mask_u = unit_arr == u_val
                    scores_u = scores[mask_u]
                    # Use dense panel-period codes (NOT raw labels) for lag math.
                    K_u = _serial_bartlett_kernel_matrix(time_codes[mask_u], L)
                    meat += scores_u.T @ K_u @ scores_u
    # PSD guard. Neither the uniform kernel (Conley 1999 fn 11) nor the
    # radial 1-D Bartlett specialization is formally PSD-guaranteed —
    # Conley's explicit PSD Bartlett formula (Eq 3.14) is the 2-D separable
    # product window, not the 1-D radial pairwise form that R `conleyreg`,
    # Stata `acreg`, and this implementation use. Check both kernels.
    # ``{eigval:.2e}`` is a literal placeholder for ``_validate_meat_psd``;
    # only ``{kernel!r}`` is interpolated by the f-string here.
    _validate_meat_psd(
        meat,
        error_msg=(
            "Conley meat contains non-finite values; check residuals and "
            "score matrix for NaN/Inf."
        ),
        warning_template=(
            f"Conley meat with conley_kernel={kernel!r} has a materially "
            "negative eigenvalue ({eigval:.2e}); the variance "
            "estimator is not guaranteed PSD on this design. Both "
            "supported kernels (radial bartlett and uniform) are "
            "practitioner specializations of Conley 1999 and are not "
            "formally PSD-guaranteed; consider varying conley_cutoff_km "
            "or reviewing the design for collinearity / degenerate "
            "residual structure."
        ),
        stacklevel=4,
    )

    return meat


def _compute_conley_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    coords: np.ndarray,
    cutoff: float,
    metric: ConleyMetric,
    kernel: str,
    bread_matrix: np.ndarray,
    *,
    time: Optional[np.ndarray] = None,
    unit: Optional[np.ndarray] = None,
    lag_cutoff: Optional[int] = None,
    cluster_ids: Optional[np.ndarray] = None,
    _conley_sparse: Optional[bool] = None,
) -> np.ndarray:
    """Conley (1999) spatial HAC sandwich variance.

    Thin wrapper around :func:`_compute_conley_meat`: builds
    ``scores = X * residuals[:, None]``, delegates the meat construction,
    then wraps with the supplied bread inverse.

    Two operating modes (both delegated to :func:`_compute_conley_meat`):

    **Cross-sectional** (``time`` / ``unit`` / ``lag_cutoff`` all None):

        Var̂(β) = bread_inv · (Σ_{i,j} K(d_ij/h) · X_i ε_i ε_j X_j') · bread_inv

    **Panel block-decomposed** (all three keyword-only args set):

        XeeX_spatial = Σ_t  S_t' · K_space_t · S_t                    (within-period sum)
        XeeX_serial  = Σ_u  S_u' · K_time_u · S_u   if lag_cutoff > 0  (within-unit sum)
        Var̂(β) = bread_inv · (XeeX_spatial + XeeX_serial) · bread_inv

    See :func:`_compute_conley_meat` for the kernel choice, sparse
    k-d-tree fallback, cluster-product kernel, and PSD-warning details.

    Inputs are assumed already validated by :func:`_validate_conley_kwargs`;
    this wrapper only does the math. Caller is responsible for the validator.

    Returns
    -------
    vcov : ndarray of shape (k, k)
    """
    scores = X * residuals[:, np.newaxis]
    meat = _compute_conley_meat(
        scores,
        coords,
        cutoff,
        metric,
        kernel,
        time=time,
        unit=unit,
        lag_cutoff=lag_cutoff,
        cluster_ids=cluster_ids,
        _conley_sparse=_conley_sparse,
    )

    # Sandwich via the shared rank-guarded inverse of the design Gram.
    # np.linalg.solve only raises on an *exactly* singular bread, so a *near*-
    # singular X'WX would otherwise flow a garbage inverse (~1e13) straight into
    # the spatial-HAC variance. `_rank_guarded_inv` truncates redundant
    # directions on the equilibrated Gram -> a finite SE on the identified
    # subspace (NaN only at rank 0), matching the covariate IF rank-guard and the
    # other structural bread inversions (ContinuousDiD / TwoStageDiD /
    # SpilloverDiD). Lazy import: `linalg` imports this module, so a top-level
    # `from diff_diff.linalg import ...` would be circular; resolving at call time
    # is safe (linalg is already loaded by the time this runs).
    from diff_diff.linalg import _rank_guarded_inv

    bread_inv, n_dropped, _, dropped = _rank_guarded_inv(bread_matrix, return_dropped=True)
    if n_dropped:
        warnings.warn(
            "Conley spatial HAC variance: the design Gram (X'WX) is "
            f"rank-deficient ({n_dropped} redundant direction(s) dropped); "
            "rank-reducing to a finite SE on the identified subspace "
            "(NaN if rank 0). This usually indicates collinear regressors.",
            UserWarning,
            stacklevel=2,
        )
    # vcov = bread^{-1} @ meat @ bread^{-1}; algebraically identical to the prior
    # two symmetric solves given `bread` symmetric (holds for any meat).
    vcov = bread_inv @ meat @ bread_inv
    # A dropped (unidentified) coefficient is zero-filled in bread_inv, which would
    # otherwise report se=0 for that named coefficient. NaN its row/col in the
    # FINAL vcov so per-coefficient SE extraction yields NaN (not 0) for the
    # unidentified directions, while the identified coefficients stay finite.
    if dropped.any():
        vcov[dropped, :] = np.nan
        vcov[:, dropped] = np.nan

    return vcov
