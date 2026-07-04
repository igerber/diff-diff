"""
SpilloverDiD — Butts (2021) ring-indicator spillover-aware DiD.

Augments a two-stage Gardner (2022) DiD with ring-indicator covariates that
identify the spillover effect on near-control units alongside the direct
effect on treated units. Handles both panel non-staggered and Section 5
staggered timing in a single estimator.

References
----------
Butts, K. (2023). Difference-in-Differences with Spatial Spillovers.
    arXiv:2105.03737v3 (originally posted 2021).
Gardner, J. (2022). Two-stage differences in differences. arXiv:2207.05943.

Notes
-----
The paper's notation in Equation 5/6 is ``(1 - D_it) * Ring_{ij}`` with
``S_i`` unit-static. Reading that literally under a two-way fixed effects
specification yields a rank-deficient design (``(1 - D_it) * S_i = S_i -
D_it``; ``S_i`` is absorbed by ``mu_i``, leaving ``-D_it``). The paper
defines ``S_it = S_i * 1{t >= t_treat}`` (page 12, just above Equation 5)
and Section 5's Table 2 makes the time-varying form explicit
(``S^k_{it}``, ``Ring^k_{it,j}``). This implementation uses the
time-varying form, which is the spec that supports the paper's
identification argument (Proposition 2.3 + Section 3.1 subsample logic).
"""

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

from diff_diff.conley import (
    _CONLEY_EARTH_RADIUS_KM,
    _CONLEY_SPARSE_N_THRESHOLD,
    _haversine_km,
    _validate_callable_metric_result,
)
from diff_diff.linalg import _rank_guarded_inv, solve_ols
from diff_diff.results import SpilloverDiDResults
from diff_diff.two_stage import _compute_gmm_corrected_meat
from diff_diff.utils import safe_inference

# Type alias mirroring diff_diff.conley.ConleyMetric so callers can supply
# any of the built-in identifiers or a user callable returning a pairwise
# distance matrix.
SpilloverMetric = Union[
    Literal["haversine", "euclidean"],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
]


# =============================================================================
# Ring construction helpers (Step 1)
# =============================================================================


def _haversine_km_pairwise(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> np.ndarray:
    """Vectorized pairwise great-circle distance (km) between two coord sets.

    Parameters
    ----------
    coords_a : ndarray of shape (n_a, 2)
        ``(lat, lon)`` in DEGREES for the first set of points.
    coords_b : ndarray of shape (n_b, 2)
        ``(lat, lon)`` in DEGREES for the second set of points.

    Returns
    -------
    ndarray of shape (n_a, n_b)
        Great-circle distances in km. Matches the ``_haversine_km`` Earth
        radius convention (6371.01 km, mirroring R ``conleyreg``).
    """
    lat_a = coords_a[:, 0][:, None]
    lon_a = coords_a[:, 1][:, None]
    lat_b = coords_b[:, 0][None, :]
    lon_b = coords_b[:, 1][None, :]
    return _haversine_km(lat_a, lon_a, lat_b, lon_b)


def _euclidean_pairwise(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> np.ndarray:
    """Vectorized pairwise Euclidean distance between two coord sets.

    Coordinates are treated as planar; no unit conversion. Matches the
    ``_pairwise_distance_matrix`` Euclidean branch of ``conley.py``.
    """
    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs))


def _apply_callable_metric_pairwise(
    metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> np.ndarray:
    """Apply a user-supplied callable metric to two coord sets.

    Unlike :func:`_validate_callable_metric_result` which checks square
    ``(n, n)`` symmetry on a single coord set, this helper accepts a
    rectangular ``(n_a, n_b)`` result. The validator is therefore relaxed:
    we only require finiteness, non-negativity, and correct shape. The
    zero-diagonal / symmetry checks apply only when the same coord set is
    passed on both sides; ring-construction usage passes a treated-only
    subset on side B, so the diagonal of the rectangular result is not
    meaningful.
    """
    result = metric(coords_a, coords_b)
    arr = np.asarray(result, dtype=np.float64)
    expected_shape = (coords_a.shape[0], coords_b.shape[0])
    if arr.shape != expected_shape:
        raise ValueError(
            "conley_metric callable returned shape "
            f"{arr.shape} for pairwise ring distance, expected {expected_shape}."
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            "conley_metric callable returned non-finite entries for pairwise "
            "ring distance; all distances must be finite."
        )
    if (arr < 0.0).any():
        raise ValueError(
            "conley_metric callable returned negative entries for pairwise "
            "ring distance; all distances must be non-negative."
        )
    return arr


def _pairwise_ring_distances(
    coords_units: np.ndarray,
    coords_treated: np.ndarray,
    metric: SpilloverMetric,
) -> np.ndarray:
    """Compute (n_units, n_treated) pairwise distances under the chosen metric."""
    if callable(metric):
        return _apply_callable_metric_pairwise(metric, coords_units, coords_treated)
    if metric == "haversine":
        return _haversine_km_pairwise(coords_units, coords_treated)
    if metric == "euclidean":
        return _euclidean_pairwise(coords_units, coords_treated)
    raise ValueError(
        f"Unknown conley_metric: {metric!r}. Expected 'haversine', 'euclidean', "
        "or a callable returning a pairwise distance matrix."
    )


def _compute_nearest_treated_distance_static(
    data: pd.DataFrame,
    *,
    unit: str,
    coords: Tuple[str, str],
    metric: SpilloverMetric,
    treated_unit_ids: np.ndarray,
    cutoff_km: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-unit nearest-treated distance for the non-staggered case.

    The set of treated units is fixed (ever-treated), so distances are
    unit-level constants and don't vary across periods. Caller broadcasts
    to per-row when assembling ring covariates.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with one row per (unit, period). Used to extract
        per-unit coords via :meth:`DataFrame.drop_duplicates` on ``unit``.
    unit : str
        Column name of the unit identifier.
    coords : tuple of (str, str)
        ``(lat_col, lon_col)``.
    metric : "haversine" | "euclidean" | callable
        Distance metric. For ``"haversine"``, ``coords`` is interpreted as
        ``(lat, lon)`` in degrees. For ``"euclidean"``, ``coords`` is
        planar. Callable receives two ``(n, 2)`` arrays and must return an
        ``(n_a, n_b)`` finite non-negative distance matrix.
    treated_unit_ids : ndarray
        IDs of ever-treated units (used as side B of pairwise distance).
    cutoff_km : float, optional
        If set and ``len(unit_index) > _CONLEY_SPARSE_N_THRESHOLD``, the
        sparse cKDTree path is used to find treated neighbors within
        ``cutoff_km`` per unit; otherwise the dense (n_units × n_treated)
        matrix is built. Units with no treated neighbor within ``cutoff_km``
        receive ``d_i = inf`` (they fall outside any ring and into the
        far-away control group, identical to dense-path behavior with
        infinite distance to the nearest reached treated unit).

    Returns
    -------
    d_i : ndarray of shape (n_unique_units,)
        ``d_i = min_{k in treated_unit_ids} d(i, k)`` per unique unit.
    unit_index : ndarray of shape (n_unique_units,)
        Unit identifiers in the same order as ``d_i``.
    """
    unit_coords_df = (
        data[[unit, coords[0], coords[1]]]
        .drop_duplicates(subset=[unit])
        .set_index(unit)
        .sort_index()
    )
    unit_index = np.asarray(unit_coords_df.index.values)
    all_coords = np.asarray(unit_coords_df[[coords[0], coords[1]]].values, dtype=np.float64)
    treated_set = set(treated_unit_ids.tolist())
    treated_mask = np.array([uid in treated_set for uid in unit_index], dtype=bool)
    treated_coords = all_coords[treated_mask]
    if treated_coords.shape[0] == 0:
        raise ValueError(
            "_compute_nearest_treated_distance_static: no treated units present "
            "in `data` matching `treated_unit_ids`."
        )

    n_units = all_coords.shape[0]
    is_builtin_metric = metric in ("haversine", "euclidean")
    if cutoff_km is not None and n_units > _CONLEY_SPARSE_N_THRESHOLD and is_builtin_metric:
        d_i = _compute_nearest_treated_distance_sparse(
            all_coords=all_coords,
            treated_coords=treated_coords,
            metric=metric,  # type: ignore[arg-type]
            cutoff_km=float(cutoff_km),
        )
    else:
        # Dense path: full pairwise matrix, then row-min.
        dists = _pairwise_ring_distances(all_coords, treated_coords, metric)
        d_i = dists.min(axis=1)
    return d_i.astype(np.float64), unit_index


def _compute_nearest_treated_distance_sparse(
    *,
    all_coords: np.ndarray,
    treated_coords: np.ndarray,
    metric: Literal["haversine", "euclidean"],
    cutoff_km: float,
) -> np.ndarray:
    """Sparse cKDTree path for nearest-treated-distance computation.

    Used when ``n_units > _CONLEY_SPARSE_N_THRESHOLD`` AND the metric is a
    built-in string. The tree is built on the treated subset (small) and
    queried with each unit row. Units with no treated neighbor inside
    ``cutoff_km`` get ``d_i = inf``, which places them in the far-away
    control group on the downstream ring-membership step.

    For haversine: lat/lon are projected to 3-D unit-sphere Cartesian
    coordinates; the chord-distance query radius is
    ``2 * sin(arc / (2 * R_earth))`` with arc clamped at ``pi * R_earth``
    so cutoffs beyond a hemisphere don't shrink. Exact great-circle
    distances are then recomputed via :func:`_haversine_km` for the in-
    range matches and the per-row minimum is taken.

    For euclidean: planar L2 directly in cKDTree.

    Parameters
    ----------
    all_coords : ndarray of shape (n_units, 2)
        Coordinates for all units.
    treated_coords : ndarray of shape (n_treated, 2)
        Coordinates for ever-treated units.
    metric : 'haversine' or 'euclidean'
        Built-in metric only; callables fall back to the dense path.
    cutoff_km : float
        Maximum considered distance. Units beyond this get ``d_i = inf``.

    Returns
    -------
    ndarray of shape (n_units,)
        Nearest-treated distance per unit (inf when no neighbor in range).
    """
    # Imported lazily to mirror conley.py's lazy-scipy pattern and keep
    # module import cheap when the sparse path isn't exercised.
    from scipy.spatial import cKDTree  # noqa: WPS433  (deferred import)

    n_units = all_coords.shape[0]
    if metric == "haversine":
        # Project lat/lon (degrees) to 3-D unit-sphere Cartesian.
        lat_rad_all = np.radians(all_coords[:, 0])
        lon_rad_all = np.radians(all_coords[:, 1])
        unit_xyz = np.column_stack(
            [
                np.cos(lat_rad_all) * np.cos(lon_rad_all),
                np.cos(lat_rad_all) * np.sin(lon_rad_all),
                np.sin(lat_rad_all),
            ]
        )
        lat_rad_tr = np.radians(treated_coords[:, 0])
        lon_rad_tr = np.radians(treated_coords[:, 1])
        tree_xyz = np.column_stack(
            [
                np.cos(lat_rad_tr) * np.cos(lon_rad_tr),
                np.cos(lat_rad_tr) * np.sin(lon_rad_tr),
                np.sin(lat_rad_tr),
            ]
        )
        # Chord-distance radius for the query; clamp arc at pi (a half-revolution)
        # so cutoffs > pi * R_earth do not shrink chord radius below the true reach.
        arc_radians = min(cutoff_km / _CONLEY_EARTH_RADIUS_KM, np.pi)
        query_r = 2.0 * np.sin(arc_radians / 2.0)
        query_r *= 1.0 + 1e-12  # numerical safety margin
        tree = cKDTree(tree_xyz)
        # Query in chord space, recompute exact great-circle distance for matches.
        neighbors = tree.query_ball_point(unit_xyz, r=query_r, p=2.0)
        d_i = np.full(n_units, np.inf, dtype=np.float64)
        for i, idxs in enumerate(neighbors):
            if not idxs:
                continue
            # Exact great-circle distance for the in-range treated neighbors.
            arr_idxs = np.asarray(idxs, dtype=np.intp)
            d_subset = _haversine_km(
                all_coords[i, 0],
                all_coords[i, 1],
                treated_coords[arr_idxs, 0],
                treated_coords[arr_idxs, 1],
            )
            d_i[i] = float(d_subset.min())
        return d_i

    # Euclidean: cKDTree handles directly.
    tree = cKDTree(treated_coords)
    d_i = np.full(n_units, np.inf, dtype=np.float64)
    neighbors = tree.query_ball_point(all_coords, r=cutoff_km, p=2.0)
    for i, idxs in enumerate(neighbors):
        if not idxs:
            continue
        arr_idxs = np.asarray(idxs, dtype=np.intp)
        d_subset = _euclidean_pairwise(all_coords[i : i + 1], treated_coords[arr_idxs])
        d_i[i] = float(d_subset.min())
    return d_i


def _compute_nearest_treated_distance_staggered(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    coords: Tuple[str, str],
    metric: SpilloverMetric,
    first_treat_by_unit: Dict[Any, Any],
    d_bar: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return per-row nearest-treated distance for the staggered case.

    For each (unit, period) observation, find the minimum distance to any
    unit that is treated BY THE END of that period (``first_treat_k <=
    t``). Ring membership in the staggered case is therefore unit-time
    varying.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data (one row per unit-period).
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    coords : tuple of (str, str)
        ``(lat_col, lon_col)``.
    metric : "haversine" | "euclidean" | callable
        Distance metric.
    first_treat_by_unit : dict
        Mapping from unit identifier to onset time (or ``np.inf`` for
        never-treated). Generated by :func:`_extract_treatment_onsets`.
    d_bar : float, optional
        When supplied, the function additionally computes the per-row
        **spillover-trigger onset** (earliest cohort onset whose treated
        units fall within ``d_bar`` of unit ``i``) reusing the cohort
        loop. Used by :func:`_compute_event_time_per_row` to avoid a
        duplicate cohort pass on the event-study path
        (PR #456 R6 performance fix).

    Notes
    -----
    The staggered helper currently always uses dense pairwise distance per
    cohort. A sparse cKDTree branch (mirroring the static helper) is queued
    as a follow-up — see TODO.md.

    Returns
    -------
    d_it : ndarray of shape (n_rows,)
        Per-row nearest-treated distance, with ``inf`` for rows where no
        unit has been treated yet by time t (early periods).
    row_unit : ndarray of shape (n_rows,)
        Aligned unit identifier per row (for downstream broadcasting).
    row_time : ndarray of shape (n_rows,)
        Aligned time identifier per row.
    trigger_onset_per_row : ndarray of shape (n_rows,) or None
        ``None`` when ``d_bar`` is None. Otherwise: per-row earliest
        cohort onset whose treated units fall within ``d_bar`` of the
        row's unit, broadcast from per-unit. NaN for rows whose unit is
        never within ``d_bar`` of any cohort.
    """
    unit_coords_df = (
        data[[unit, coords[0], coords[1]]].drop_duplicates(subset=[unit]).set_index(unit)
    )
    unit_index = np.asarray(unit_coords_df.index.values)
    all_coords = np.asarray(unit_coords_df[[coords[0], coords[1]]].values, dtype=np.float64)
    unit_to_pos = {uid: pos for pos, uid in enumerate(unit_index)}

    row_unit = np.asarray(data[unit].values)
    row_time = np.asarray(data[time].values)
    n_rows = len(row_unit)
    d_it = np.full(n_rows, np.inf, dtype=np.float64)
    trigger_onset_per_unit_pos: Optional[np.ndarray] = (
        np.full(len(unit_index), np.nan, dtype=np.float64) if d_bar is not None else None
    )

    # Determine the cohort onset times that exist in the data (excluding never-treated).
    unique_onsets = sorted({ft for ft in first_treat_by_unit.values() if np.isfinite(ft)})
    if not unique_onsets:
        # Degenerate: no treated units. Caller should have rejected this
        # in `_validate_spillover_inputs`, but defensively return inf.
        return d_it, row_unit, row_time, None

    # Row's unit position. Invariant across cohort iterations — compute
    # once outside the loop.
    row_pos = np.array([unit_to_pos[uid] for uid in row_unit], dtype=np.intp)

    # For each unique onset time, compute (n_units, n_treated_by_then) pairwise
    # distances ONCE, then assign to rows whose t >= that onset (carrying forward
    # the minimum across cohorts).
    for onset in unique_onsets:
        treated_by_onset_ids = [uid for uid, ft in first_treat_by_unit.items() if ft <= onset]
        treated_positions = np.array(
            [unit_to_pos[uid] for uid in treated_by_onset_ids if uid in unit_to_pos],
            dtype=np.intp,
        )
        if treated_positions.size == 0:
            continue
        treated_coords = all_coords[treated_positions]
        # Compute per-unit nearest distance to this cohort's treated set.
        dists_to_cohort = _pairwise_ring_distances(all_coords, treated_coords, metric).min(axis=1)
        # Update rows whose period t >= onset: take min of current d_it and the
        # newly-available cohort distance.
        affected_rows = row_time >= onset
        if not affected_rows.any():
            continue
        row_cohort_dist = dists_to_cohort[row_pos]
        # Only update rows where this cohort's distance is smaller than the
        # current d_it (carries the running minimum across cohorts).
        update_mask = affected_rows & (row_cohort_dist < d_it)
        d_it[update_mask] = row_cohort_dist[update_mask]

        # Reuse this same cohort distance computation for the per-unit
        # spillover-trigger onset when d_bar is supplied. The trigger is
        # the FIRST cohort whose treated units fall within d_bar of unit
        # i — once locked it persists for later cohort iterations. Using
        # cumulative-treated distances here is fine: if a unit is in
        # range of cohort c1, dists_to_cohort at onset=c1 already detects
        # it; later iterations with extra treated units only shrink the
        # distance, never grow it back above d_bar.
        if trigger_onset_per_unit_pos is not None:
            in_range_for_cohort = dists_to_cohort <= d_bar
            not_yet_triggered = np.isnan(trigger_onset_per_unit_pos)
            trigger_onset_per_unit_pos[in_range_for_cohort & not_yet_triggered] = onset

    # Broadcast per-unit trigger to rows when computed.
    if trigger_onset_per_unit_pos is not None:
        trigger_onset_per_row = trigger_onset_per_unit_pos[row_pos]
    else:
        trigger_onset_per_row = None
    return d_it, row_unit, row_time, trigger_onset_per_row


def _compute_event_time_per_row(
    *,
    data: pd.DataFrame,
    unit: str,
    row_unit: np.ndarray,
    row_time: np.ndarray,
    effective_onsets: Dict[Any, float],
    coords: Tuple[str, str],
    metric: SpilloverMetric,
    d_bar: float,
    precomputed_trigger_onset_per_row: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute two event-time clocks per row for Wave C event-study mode.

    Butts (2021) Section 5 / Table 2 uses one symbol ``K_it`` but operationally
    there are TWO event-time clocks — one for the direct-effect series and one
    for the spillover-exposure series. This helper returns both.

    - ``K_direct[r] = row_time[r] - effective_onsets[row_unit[r]]`` for rows of
      ever-treated units (any t, including pre-treatment k < 0 for placebo
      coefficients). NaN for never-treated units.
    - ``K_spill[r] = row_time[r] - trigger_onset[row_unit[r]]`` for rows where
      the spillover-trigger cohort has activated by ``row_time[r]``. NaN
      otherwise. ``trigger_onset[i]`` is the EARLIEST effective onset among
      cohorts whose treated units fall within ``d_bar`` of unit ``i``.

    Cohort onsets are iterated in ascending order so the trigger is the first
    cohort that puts unit ``i`` in any ring — matching the running-min logic
    used by :func:`_compute_nearest_treated_distance_staggered` for ``d_it``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data; used to extract one (lat, lon) coordinate per unit.
    unit, coords, metric, d_bar
        Mirror :func:`_compute_nearest_treated_distance_staggered`.
    row_unit, row_time : ndarray of shape (n_rows,)
        Per-row identifiers (anticipation-adjusted onsets are baked into
        ``effective_onsets``; row_time is the raw period).
    effective_onsets : dict
        Mapping from unit identifier to anticipation-shifted first_treat
        (``first_treat - anticipation``). ``np.inf`` for never-treated units.

    Returns
    -------
    K_direct : ndarray of shape (n_rows,), float64 with NaN where undefined.
    K_spill : ndarray of shape (n_rows,), float64 with NaN where undefined.

    Notes
    -----
    PR #456 R6 performance fix: when ``precomputed_trigger_onset_per_row``
    is supplied (as :func:`_compute_nearest_treated_distance_staggered`
    now optionally returns when called with ``d_bar=...``), the cohort
    loop is skipped — K_spill is derived directly from the precomputed
    trigger. The fallback (compute trigger inline) is kept for unit-test
    callers and other code paths that don't have access to the staggered
    distance helper's output.
    """
    n_rows = len(row_unit)
    row_time_arr = np.asarray(row_time, dtype=np.float64)

    # K_direct: per-row, derived from row_unit -> own effective_onset.
    K_direct = np.full(n_rows, np.nan, dtype=np.float64)
    own_onsets = np.array([effective_onsets.get(uid, np.inf) for uid in row_unit], dtype=np.float64)
    direct_defined = np.isfinite(own_onsets)
    K_direct[direct_defined] = row_time_arr[direct_defined] - own_onsets[direct_defined]

    if precomputed_trigger_onset_per_row is not None:
        # Fast path: reuse trigger onsets already computed by the staggered
        # distance helper. Avoids a duplicate cohort loop.
        row_trigger = np.asarray(precomputed_trigger_onset_per_row, dtype=np.float64)
        K_spill = np.full(n_rows, np.nan, dtype=np.float64)
        triggered = np.isfinite(row_trigger)
        post_trigger = triggered & (row_time_arr >= row_trigger)
        K_spill[post_trigger] = row_time_arr[post_trigger] - row_trigger[post_trigger]
        return K_direct, K_spill

    # Fallback path (test callers, etc.): compute trigger inline via own
    # cohort loop. trigger_onset[i] = first effective_onset among cohorts
    # whose treated units have d(i, treated_in_cohort) <= d_bar.
    unit_coords_df = (
        data[[unit, coords[0], coords[1]]].drop_duplicates(subset=[unit]).set_index(unit)
    )
    unit_index = np.asarray(unit_coords_df.index.values)
    all_coords = np.asarray(unit_coords_df[[coords[0], coords[1]]].values, dtype=np.float64)
    unit_to_pos = {uid: pos for pos, uid in enumerate(unit_index)}

    unique_onsets = sorted({eff_ft for eff_ft in effective_onsets.values() if np.isfinite(eff_ft)})
    trigger_onset_per_unit_pos = np.full(len(unit_index), np.nan, dtype=np.float64)

    for onset in unique_onsets:
        # Units treated AT THIS ONSET (not by/before; we want the cohort's
        # own treated set so we can compute the per-onset distance front).
        treated_at_onset_ids = [uid for uid, ft in effective_onsets.items() if ft == onset]
        treated_positions = np.array(
            [unit_to_pos[uid] for uid in treated_at_onset_ids if uid in unit_to_pos],
            dtype=np.intp,
        )
        if treated_positions.size == 0:
            continue
        treated_coords = all_coords[treated_positions]
        dists_to_cohort = _pairwise_ring_distances(all_coords, treated_coords, metric).min(axis=1)
        in_range_for_cohort = dists_to_cohort <= d_bar
        not_yet_triggered = np.isnan(trigger_onset_per_unit_pos)
        trigger_onset_per_unit_pos[in_range_for_cohort & not_yet_triggered] = onset

    # Broadcast trigger onset to rows; K_spill = t - trigger when t >= trigger.
    row_pos = np.array([unit_to_pos.get(uid, -1) for uid in row_unit], dtype=np.intp)
    K_spill = np.full(n_rows, np.nan, dtype=np.float64)
    valid_pos = row_pos >= 0
    row_trigger = np.where(
        valid_pos, trigger_onset_per_unit_pos[np.where(valid_pos, row_pos, 0)], np.nan
    )
    triggered = np.isfinite(row_trigger)
    post_trigger = triggered & (row_time_arr >= row_trigger)
    K_spill[post_trigger] = row_time_arr[post_trigger] - row_trigger[post_trigger]

    return K_direct, K_spill


def _apply_horizon_binning(
    K_arr: np.ndarray,
    horizon_max: Optional[int],
) -> np.ndarray:
    """Clip per-row event-time values into ``[-horizon_max, +horizon_max]`` bins.

    Wave C event-study path uses bin-into-endpoint-pools semantics: rows with
    event-time ``k < -H`` aggregate into a single ``k = -H`` dummy; rows with
    ``k > +H`` aggregate into a single ``k = +H`` dummy. No observations are
    dropped (cf. TwoStageDiD's ``horizon_max`` which filters rows).

    NaN values in ``K_arr`` propagate through (``np.clip`` preserves NaN by
    default). Omega_0 / never-treated rows carry NaN K values, which cause
    ``1{K_binned = k}`` to evaluate False at every k — so they contribute 0
    to all event-time dummies (correct identification: those rows enter
    stage 1 only, not the event-time decomposition).

    Parameters
    ----------
    K_arr : ndarray
        Per-row event-time values. NaN entries are passed through unchanged.
    horizon_max : int or None
        Bin width; if ``None``, no clipping (used for auto-detect path where
        ``H = max(|K_it|)`` provides the natural bound).

    Returns
    -------
    ndarray of same shape and dtype as input, with NaN-preserving clamp applied.
    """
    if horizon_max is None:
        return K_arr.astype(np.float64, copy=False)
    if not isinstance(horizon_max, (int, np.integer)) or horizon_max < 0:
        raise ValueError(
            f"horizon_max must be a non-negative integer or None; "
            f"got {horizon_max!r} (type {type(horizon_max).__name__})."
        )
    return np.clip(K_arr.astype(np.float64, copy=False), -float(horizon_max), float(horizon_max))


def _build_event_study_design(
    *,
    D_it: np.ndarray,
    ring_masks: np.ndarray,
    ring_labels: List[str],
    K_direct_binned: np.ndarray,
    K_spill_binned: np.ndarray,
    event_time_grid: List[int],
    ref_period: int,
) -> Tuple[
    np.ndarray,
    List[str],
    List[Tuple[str, Optional[str], int]],
    List[Tuple[str, Optional[str], int]],
    np.ndarray,
]:
    """Build per-event-time × ring stage-2 design matrix for Wave C event-study.

    The design has two series of dummies:

    - **Direct effect**: ``D^k_{it} := 1{K_direct_{it} = k AND row is ever-treated}``
      for each ``k ∈ event_time_grid \\ {ref_period}``. NaN entries in
      ``K_direct_binned`` (never-treated rows) cause the indicator to evaluate
      False, naturally yielding zero contribution.
    - **Spillover**: ``Ring^k_{it,j} := (1 - D_it) * ring_masks[:, j] * 1{K_spill_{it} = k}``
      for each ring ``j`` and each ``k ∈ event_time_grid \\ {ref_period}``.

    All-zero columns are pre-filtered (one summary warning instead of many),
    but the FULL rectangular grid of (series, ring, k) tuples is also returned
    so downstream code can emit the MultiIndex ``spillover_effects`` schema
    with NaN coefficients for empty cells (per Wave C plan: rectangular).

    Parameters
    ----------
    D_it : ndarray of shape (n_rows,), float
        Per-row binary indicator (treated AND post-treatment).
    ring_masks : ndarray of shape (n_rows, K), bool
        Per-row ring-membership indicators (from :func:`_build_ring_indicators`).
    ring_labels : list of K strings
        Human-readable labels for each ring band.
    K_direct_binned, K_spill_binned : ndarray of shape (n_rows,), float64
        Per-row event-time clocks (NaN where undefined). Already passed through
        :func:`_apply_horizon_binning` if applicable.
    event_time_grid : list of int
        The full event-time bin set (e.g. ``[-3, -2, -1, 0, 1, 2, 3]`` for
        ``horizon_max=3``). Reference period is dropped from this list inside
        the helper.
    ref_period : int
        The event-time integer to drop from BOTH series.

    Returns
    -------
    X_2 : ndarray of shape (n_rows, n_kept_cols)
        Stage-2 design matrix (only non-empty columns kept).
    kept_col_names : list of str
        Column labels matching X_2 columns. Convention: ``"D^k=+0"``,
        ``"_spillover_[0, 50)^k=-2"``, with signed integer suffix.
    kept_col_meta : list of (series, ring_label_or_None, k)
        Tuple metadata per kept column (``series ∈ {"direct", "spillover"}``).
    rectangular_grid : list of (series, ring_label_or_None, k)
        FULL grid of (series, ring, k) entries including those dropped because
        the column was all zeros. Used for rectangular MultiIndex emission.
        Order matches the design layout (direct first, then per-ring spillover).
    n_obs_per_col : ndarray of shape (n_kept_cols,), int64
        Count of rows with a non-zero contribution to each kept column.
    """
    if not isinstance(ref_period, (int, np.integer)):
        raise TypeError(
            f"ref_period must be an integer; got {ref_period!r} "
            f"(type {type(ref_period).__name__})."
        )
    K = ring_masks.shape[1]
    if len(ring_labels) != K:
        raise ValueError(
            f"ring_labels length ({len(ring_labels)}) must match number of " f"rings ({K})."
        )

    # The grid of event-times to emit dummies for, with the reference dropped.
    k_grid = [int(k) for k in event_time_grid if int(k) != int(ref_period)]

    one_minus_D = 1.0 - D_it.astype(np.float64)
    ring_masks_f = ring_masks.astype(np.float64)
    K_direct_f = np.asarray(K_direct_binned, dtype=np.float64)
    K_spill_f = np.asarray(K_spill_binned, dtype=np.float64)

    def _signed(k: int) -> str:
        return f"{k:+d}"

    # Build candidate columns in canonical order:
    #   1) all direct-effect dummies, ascending k
    #   2) per-ring spillover dummies (ascending ring, ascending k within)
    candidate_cols: List[Tuple[str, Optional[str], int, np.ndarray]] = []
    rectangular_grid: List[Tuple[str, Optional[str], int]] = []

    for k in k_grid:
        # Direct-effect dummy: D_i (implicit via NaN-on-never-treated) * 1{K_direct = k}.
        col = (K_direct_f == float(k)).astype(np.float64)
        candidate_cols.append(("direct", None, k, col))
        rectangular_grid.append(("direct", None, k))

    for j in range(K):
        ring_lab = ring_labels[j]
        for k in k_grid:
            # Spillover dummy: (1 - D_it) * Ring_j * 1{K_spill = k}.
            col = one_minus_D * ring_masks_f[:, j] * (K_spill_f == float(k)).astype(np.float64)
            candidate_cols.append(("spillover", ring_lab, k, col))
            rectangular_grid.append(("spillover", ring_lab, k))

    # Pre-filter all-zero columns to keep solve_ols's rank-deficient warning
    # noise low. Track the kept set.
    kept_indices: List[int] = []
    kept_cols_list: List[np.ndarray] = []
    kept_col_names: List[str] = []
    kept_col_meta: List[Tuple[str, Optional[str], int]] = []
    n_obs_list: List[int] = []
    n_dropped = 0

    for idx, (series, ring_lab, k, col) in enumerate(candidate_cols):
        n_nonzero = int(np.count_nonzero(col))
        if n_nonzero == 0:
            n_dropped += 1
            continue
        kept_indices.append(idx)
        kept_cols_list.append(col)
        if series == "direct":
            kept_col_names.append(f"D^k={_signed(k)}")
        else:
            kept_col_names.append(f"_spillover_{ring_lab}^k={_signed(k)}")
        kept_col_meta.append((series, ring_lab, k))
        n_obs_list.append(n_nonzero)

    if n_dropped > 0:
        warnings.warn(
            f"SpilloverDiD event-study: {n_dropped} of "
            f"{len(candidate_cols)} stage-2 design column(s) were "
            "all-zero (no rows contribute) and dropped before fitting. "
            "Empty (series, ring, event_time) cells appear in the result "
            "with coef=NaN and n_obs=0 (rectangular schema). To shrink the "
            "emitted grid, reduce horizon_max or use horizon_max=None for "
            "auto-detection.",
            UserWarning,
            stacklevel=2,
        )

    if not kept_cols_list:
        # All columns dropped — degenerate. Return empty design; caller
        # handles via downstream df_resid check + safe_inference NaN
        # propagation.
        X_2 = np.zeros((len(D_it), 0), dtype=np.float64)
    else:
        X_2 = np.column_stack(kept_cols_list)

    n_obs_per_col = np.asarray(n_obs_list, dtype=np.int64)
    return X_2, kept_col_names, kept_col_meta, rectangular_grid, n_obs_per_col


def _extract_event_study_results(
    *,
    coef: np.ndarray,
    vcov: Optional[np.ndarray],
    col_names_all: List[str],
    kept_col_meta: List[Tuple[str, Optional[str], int]],
    rectangular_grid: List[Tuple[str, Optional[str], int]],
    n_obs_per_col: np.ndarray,
    ref_period: int,
    df_resid: int,
    alpha: float,
    ring_labels: List[str],
    weight_sum_per_col: Optional[np.ndarray] = None,
) -> Tuple[
    float,
    float,
    float,
    float,
    Tuple[float, float],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[Dict[int, Dict[str, Any]]],
    Dict[str, float],
]:
    """Extract per-event-time inference and the share-weighted scalar ``att``.

    Builds three output surfaces from a single stage-2 fit:

    - ``att_dynamic`` : per-event-time direct-effect DataFrame indexed by ``k``.
      Includes the reference period row with ``coef=0.0, se=0.0, n_obs=0``.
      Rectangular emission across the full direct-effect event-time grid.
    - ``spillover_effects`` : MultiIndex ``(ring_label, event_time)`` DataFrame
      with the same columns. Rectangular over the full spillover grid.
    - ``event_study_effects`` : TwoStageDiD-compatible alias matching
      ``two_stage.py:1355-1389`` schema (``conf_int`` as ``(low, high)`` tuple,
      reference period as ``(0.0, 0.0)``).

    Scalar ``att`` uses share-weighted aggregation on post-treatment
    ``tau_k`` with SE from linear-combination inference on the
    corresponding vcov submatrix. When ``weight_sum_per_col`` is supplied
    (Wave E.1 survey path), the per-horizon shares are SURVEY-WEIGHT
    TOTALS (consistent with the WLS horizon coefficients); otherwise
    shares are raw observation counts (Wave C sample-share rule).
    """
    # Per-coefficient inference dict keyed by (series, ring_label, k).
    per_coef: Dict[Tuple[str, Optional[str], int], Dict[str, Any]] = {}
    for i, (series, ring_label, k) in enumerate(kept_col_meta):
        coef_i = float(coef[i]) if np.isfinite(coef[i]) else float("nan")
        if vcov is not None and np.isfinite(vcov[i, i]):
            se_i = float(np.sqrt(max(vcov[i, i], 0.0)))
        else:
            se_i = float("nan")
        t_i, p_i, ci_i = safe_inference(coef_i, se_i, alpha=alpha, df=df_resid)
        per_coef[(series, ring_label, k)] = {
            "coef": coef_i,
            "se": se_i,
            "t_stat": t_i,
            "p_value": p_i,
            "ci_low": ci_i[0],
            "ci_high": ci_i[1],
            "n_obs": int(n_obs_per_col[i]),
        }

    direct_k_set = sorted({k for (s, _, k) in rectangular_grid if s == "direct"})
    spillover_k_set = sorted({k for (s, _, k) in rectangular_grid if s == "spillover"})

    # Build att_dynamic: rectangular over direct event-time grid + reference row.
    all_direct_ks = sorted(set(direct_k_set) | {ref_period})
    direct_rows: List[Dict[str, Any]] = []
    for k in all_direct_ks:
        if k == ref_period:
            direct_rows.append(
                {
                    "k": k,
                    "coef": 0.0,
                    "se": 0.0,
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n_obs": 0,
                }
            )
        elif ("direct", None, k) in per_coef:
            r = per_coef[("direct", None, k)]
            direct_rows.append({"k": k, **r})
        else:
            direct_rows.append(
                {
                    "k": k,
                    "coef": float("nan"),
                    "se": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n_obs": 0,
                }
            )
    att_dynamic_df = pd.DataFrame(direct_rows).set_index("k").sort_index() if direct_rows else None

    # Build spillover_effects: rectangular over (ring_label, k) grid.
    #
    # PR #456 R1 fix (P3): the spillover grid must INCLUDE the reference
    # period row per ring. The pre-filter in _build_event_study_design drops
    # `ref_period` from the fitted column set, but the rectangular schema
    # for spillover must still emit (ring, ref_period) with `coef=0.0,
    # se=0.0, n_obs=0` for symmetry with the direct-effect series (which
    # emits its reference row at k=ref_period). Without this, consumers
    # iterating `[-H, ..., +H]` would hit a missing (ring, ref_period)
    # slice — the registry promises rectangular emission over the full
    # event-time grid.
    all_spillover_ks = sorted(set(spillover_k_set) | {ref_period})
    spillover_rows: List[Dict[str, Any]] = []
    for ring_lab in ring_labels:
        for k in all_spillover_ks:
            if k == ref_period:
                # Reference-period spillover row: 0-anchored (mirrors direct).
                spillover_rows.append(
                    {
                        "ring": ring_lab,
                        "k": k,
                        "coef": 0.0,
                        "se": 0.0,
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "n_obs": 0,
                    }
                )
                continue
            key = ("spillover", ring_lab, k)
            if key in per_coef:
                r = per_coef[key]
                spillover_rows.append({"ring": ring_lab, "k": k, **r})
            else:
                spillover_rows.append(
                    {
                        "ring": ring_lab,
                        "k": k,
                        "coef": float("nan"),
                        "se": float("nan"),
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n_obs": 0,
                    }
                )
    spillover_df = (
        pd.DataFrame(spillover_rows).set_index(["ring", "k"]).sort_index()
        if spillover_rows
        else None
    )

    # Build event_study_effects dict (TwoStageDiD-compatible).
    event_study_effects: Dict[int, Dict[str, Any]] = {}
    for k in all_direct_ks:
        if k == ref_period:
            event_study_effects[k] = {
                "effect": 0.0,
                "se": 0.0,
                "n_obs": 0,
                "t_stat": float("nan"),
                "p_value": float("nan"),
                "conf_int": (0.0, 0.0),
            }
        elif ("direct", None, k) in per_coef:
            r = per_coef[("direct", None, k)]
            event_study_effects[k] = {
                "effect": r["coef"],
                "se": r["se"],
                "n_obs": r["n_obs"],
                "t_stat": r["t_stat"],
                "p_value": r["p_value"],
                "conf_int": (r["ci_low"], r["ci_high"]),
            }
        else:
            event_study_effects[k] = {
                "effect": float("nan"),
                "se": float("nan"),
                "n_obs": 0,
                "t_stat": float("nan"),
                "p_value": float("nan"),
                "conf_int": (float("nan"), float("nan")),
            }

    # Scalar att via share-weighted average over post-treatment direct
    # coefficients (k >= 0). SE via linear-combination on the vcov submatrix
    # of those kept columns.
    #
    # Wave E.1: when `weight_sum_per_col` is provided (survey-design path),
    # the per-horizon share weights are SURVEY-WEIGHT TOTALS rather than
    # raw observation counts. This keeps the aggregation consistent with
    # the WLS horizon coefficients themselves (which are weighted) — using
    # raw n_obs_per_col would mix unweighted shares with weighted horizons
    # and target the wrong estimand on weighted event-study fits.
    #
    # Fail-closed contract (PR #456 R1 fix): if ANY post-treatment direct
    # coefficient is NaN (solve_ols dropped the column as rank-deficient),
    # the aggregate is structurally unidentified. Set att = NaN with a
    # warning rather than silently zeroing the dropped column's contribution
    # via np.nansum (which would change the point estimate without
    # renormalizing weights). Matches the library-wide
    # `feedback_no_silent_failures` invariant.
    post_direct_indices = [
        i for i, (s, _, k) in enumerate(kept_col_meta) if s == "direct" and k >= 0
    ]
    if post_direct_indices and vcov is not None:
        share_source = weight_sum_per_col if weight_sum_per_col is not None else n_obs_per_col
        n_obs_post = np.array([share_source[i] for i in post_direct_indices], dtype=np.float64)
        total_post_obs = n_obs_post.sum()
        coefs_post = np.array([coef[i] for i in post_direct_indices], dtype=np.float64)
        has_nan_post = bool(np.any(~np.isfinite(coefs_post)))
        if has_nan_post:
            warnings.warn(
                "SpilloverDiD event-study: scalar `att` is NaN because at "
                "least one post-treatment direct-effect coefficient was "
                "dropped as rank-deficient (or otherwise non-finite). The "
                "aggregate is unidentified under this design; inspect "
                "`att_dynamic` for the per-event-time coefficients and "
                "re-aggregate manually if appropriate.",
                UserWarning,
                stacklevel=2,
            )
            att = float("nan")
            att_se = float("nan")
        elif total_post_obs > 0:
            weights = n_obs_post / total_post_obs
            att = float(np.sum(weights * coefs_post))
            vcov_subset = vcov[np.ix_(post_direct_indices, post_direct_indices)]
            var_att = float(weights @ vcov_subset @ weights)
            att_se = float(np.sqrt(max(var_att, 0.0))) if np.isfinite(var_att) else float("nan")
        else:
            att = float("nan")
            att_se = float("nan")
    else:
        att = float("nan")
        att_se = float("nan")
    att_t, att_p, att_ci = safe_inference(att, att_se, alpha=alpha, df=df_resid)

    # Coefficients dict — name → value for every kept stage-2 coefficient.
    coefficients_full: Dict[str, float] = {}
    for i, name in enumerate(col_names_all):
        val = float(coef[i]) if np.isfinite(coef[i]) else float("nan")
        coefficients_full[name] = val
    coefficients_full["ATT"] = att

    return (
        att,
        att_se,
        att_t,
        att_p,
        att_ci,
        spillover_df,
        att_dynamic_df,
        event_study_effects,
        coefficients_full,
    )


def _build_ring_indicators(
    d_values: np.ndarray,
    rings: List[float],
) -> np.ndarray:
    """Build K boolean ring masks from distances and breakpoints.

    Convention (per Butts Equation 6 + plan Risks #2): half-open at the
    top of each interior ring, CLOSED at the outermost upper edge so units
    exactly at ``d_bar`` belong to the last ring (not the far-away group).
    Far-away controls use a strict ``d_i > d_bar`` check (handled
    elsewhere). Treated units have ``d_i = 0`` and fall in Ring_1 by
    construction; their ring contribution is later zeroed by the
    ``(1 - D_i)`` factor.

    Parameters
    ----------
    d_values : ndarray
        Distances (per-unit for non-staggered or per-row for staggered).
    rings : list of float
        Sorted breakpoints with ``len(rings) >= 2``. ``K = len(rings) - 1``
        rings are constructed.

    Returns
    -------
    masks : ndarray of shape (len(d_values), K), bool
        ``masks[i, j] = True`` if ``d_values[i]`` falls in ring ``j``.

    Raises
    ------
    ValueError
        ``rings`` has fewer than 2 elements, or is not strictly increasing.
    """
    rings_arr = np.asarray(rings, dtype=np.float64)
    if rings_arr.ndim != 1 or rings_arr.size < 2:
        raise ValueError(
            "rings must be a sorted list of at least 2 breakpoints "
            f"(got shape {rings_arr.shape})."
        )
    if (np.diff(rings_arr) <= 0).any():
        raise ValueError("rings must be strictly increasing; got " f"{rings_arr.tolist()}.")
    if (rings_arr < 0).any():
        raise ValueError("rings must be non-negative; got " f"{rings_arr.tolist()}.")

    n = d_values.shape[0]
    K = rings_arr.size - 1
    masks = np.zeros((n, K), dtype=bool)
    for j in range(K):
        lo = rings_arr[j]
        hi = rings_arr[j + 1]
        if j == K - 1:
            # Outermost ring: closed at d_bar so units at the boundary
            # belong to this ring (not the far-away group).
            masks[:, j] = (d_values >= lo) & (d_values <= hi)
        else:
            # Interior rings: half-open at top so the breakpoint between
            # adjacent rings unambiguously falls in the next ring.
            masks[:, j] = (d_values >= lo) & (d_values < hi)
    return masks


def _ring_label(rings: List[float], j: int) -> str:
    """Render the human-readable ring label for index ``j``.

    Convention matches :func:`_build_ring_indicators`: half-open at the
    top of interior rings, closed at the outermost upper edge.
    """
    K = len(rings) - 1
    lo = rings[j]
    hi = rings[j + 1]
    if j == K - 1:
        return f"[{lo:g}, {hi:g}]"
    return f"[{lo:g}, {hi:g})"


# =============================================================================
# Treatment-timing helpers (Step 2)
# =============================================================================


def _extract_treatment_onsets(
    data: pd.DataFrame,
    first_treat_col: str,
    unit_col: str,
    *,
    treat_zero_as_never_treated: bool = True,
) -> Dict[Any, float]:
    """Return a dict mapping each unit to its treatment onset time.

    Parameters
    ----------
    treat_zero_as_never_treated : bool, default True
        When True (default, matching Gardner / TwoStageDiD user convention),
        ``first_treat = 0`` is treated as a never-treated sentinel
        equivalent to ``np.inf``. Set to False for INTERNAL onset columns
        produced by :func:`_convert_treatment_to_first_treat` from a
        binary ``D`` column — there, ``0`` may legitimately be the
        onset time on 0-indexed panels (a unit treated at the first
        observed period gets ``first_treat = 0``). The auto-generated
        column writes ``np.inf`` for never-treated, so the 0-as-sentinel
        collision is avoided.

    Notes
    -----
    If a unit has non-constant ``first_treat`` values across its rows,
    ``ValueError`` is raised — SpilloverDiD requires the
    absorbing-treatment assumption (one onset per unit). Mirrors
    :class:`TwoStageDiD`'s warning behaviour, but escalates to a hard error
    because the spillover identification math depends on each unit having a
    single well-defined ``S_it`` trajectory.
    """

    def _normalize(v: float) -> float:
        if np.isinf(v):
            return np.inf
        if treat_zero_as_never_treated and v == 0:
            return np.inf
        return float(v)

    onsets: Dict[Any, float] = {}
    non_constant_units: List[Any] = []
    for unit_id, group in data.groupby(unit_col):
        ft_unique = group[first_treat_col].dropna().unique().tolist()
        normalized = {_normalize(v) for v in ft_unique}
        if len(normalized) > 1:
            non_constant_units.append(unit_id)
            continue
        if not normalized:
            # All rows are NaN → treat as never-treated.
            onsets[unit_id] = np.inf
            continue
        # Use the unique value, not iloc[0], to avoid being fooled by a
        # leading-NaN row when the rest of the unit is consistently treated.
        ft = next(iter(normalized))
        onsets[unit_id] = ft  # already normalized: np.inf for never-treated, float otherwise
    if non_constant_units:
        sample = non_constant_units[:5]
        suffix = f" (and {len(non_constant_units) - 5} more)" if len(non_constant_units) > 5 else ""
        raise ValueError(
            f"{len(non_constant_units)} unit(s) have non-constant "
            f"'{first_treat_col}' values across rows (e.g. {sample}{suffix}). "
            "SpilloverDiD requires the absorbing-treatment assumption "
            "(one onset per unit, treatment never reverses). For "
            "non-absorbing / reversible treatments, see "
            "ChaisemartinDHaultfoeuille."
        )
    return onsets


def _convert_treatment_to_first_treat(
    data: pd.DataFrame,
    treatment: str,
    time: str,
    unit: str,
) -> Tuple[pd.DataFrame, str]:
    """Auto-convert a binary ``D_it`` column to a per-unit ``first_treat`` column.

    Returns a defensive-copy frame augmented with a new
    ``"_spillover_first_treat"`` column whose value per unit is
    ``min{t : D_it = 1}`` for ever-treated units and ``np.inf`` for
    never-treated. The original ``treatment`` column is preserved.

    **Absorbing-treatment validation:** after extracting ``first_treat``,
    each ever-treated unit's ``D_it`` is verified to be 1 at all rows with
    ``t >= first_treat[unit]`` (treatment never reverses). Non-absorbing
    patterns like ``[0, 1, 0]`` raise ``ValueError`` rather than being
    silently coerced into ``first_treat = min(t | D_it = 1)``.

    Raises
    ------
    ValueError
        ``data`` does not contain a numeric ``treatment`` column or
        ``time`` / ``unit`` columns; ``treatment`` has values outside
        ``{0, 1}``; or treatment is non-absorbing for some unit.
    """
    if treatment not in data.columns:
        raise ValueError(f"treatment column '{treatment}' not in data.")
    # NaN in treatment is not silently coerced — it would later be rebuilt
    # from `first_treat` and could flip a row from "unknown" to "treated"
    # or "control" with no warning.
    nan_mask = data[treatment].isna()
    if bool(nan_mask.any()):
        n_nan = int(nan_mask.sum())
        raise ValueError(
            f"treatment column '{treatment}' contains {n_nan} NaN value(s). "
            "SpilloverDiD requires explicit 0/1 status on every row; "
            "missing-treatment rows must be either imputed or dropped "
            "before fitting (the auto-conversion path cannot silently "
            "reclassify them since that would change tau_total and "
            "delta_j without warning)."
        )
    treat_vals = data[treatment].unique()
    # Exact binary check — NOT `int(v) in (0, 1)` (which would accept 0.9,
    # 1.1, etc. by rounding-down semantics and silently misclassify
    # fractional rows into the control group).
    if not all(v in (0, 0.0, 1, 1.0) for v in treat_vals):
        raise ValueError(
            f"treatment column '{treatment}' must contain only exact 0/1 "
            f"values; got unique values: {sorted(treat_vals)}. Fractional "
            "values (e.g. 0.9 or 1.1) are NOT silently coerced — fix the "
            "data or thresholding upstream before passing to SpilloverDiD."
        )

    out = data.copy(deep=False)
    treated_rows = out[out[treatment] == 1]
    if treated_rows.empty:
        out["_spillover_first_treat"] = np.inf
        return out, "_spillover_first_treat"

    onset_by_unit = treated_rows.groupby(unit)[time].min()

    # Verify absorbing: for each ever-treated unit, D_it must be EXACTLY 1
    # at every row with t >= first_treat[unit]. NOT merely "not equal to 0"
    # — that would silently accept e.g. NaN or other non-binary values that
    # slipped past the binary check above (defense in depth).
    reversing_units: List[Any] = []
    for u in onset_by_unit.index:
        onset = onset_by_unit.loc[u]
        unit_rows = out[(out[unit] == u) & (out[time] >= onset)]
        if (unit_rows[treatment] != 1).any():
            reversing_units.append(u)
    if reversing_units:
        sample = reversing_units[:5]
        suffix = f" (and {len(reversing_units) - 5} more)" if len(reversing_units) > 5 else ""
        raise ValueError(
            f"{len(reversing_units)} unit(s) have non-absorbing treatment "
            f"patterns (treatment reverses to 0 after onset; e.g. units "
            f"{sample}{suffix}). SpilloverDiD requires the absorbing-"
            "treatment assumption — once a unit is treated, it stays "
            "treated. For non-absorbing / reversible treatments, see "
            "ChaisemartinDHaultfoeuille."
        )

    onset_lookup: Dict[Any, float] = {
        uid: float(onset_by_unit.loc[uid]) if uid in onset_by_unit.index else np.inf
        for uid in out[unit].unique()
    }
    out["_spillover_first_treat"] = out[unit].map(onset_lookup).astype(np.float64).values
    return out, "_spillover_first_treat"


# =============================================================================
# Two-stage Gardner inline (Step 3)
# =============================================================================

# Convergence tolerance for the iterative alternating-projection FE solver
# (Gauss-Seidel style; same recursion as the shared
# `diff_diff.utils._iterative_fe_solve` used by ImputationDiD/TwoStageDiD).
_FE_ITER_MAX = 100
_FE_ITER_TOL = 1e-10


def _check_omega_0_connectivity(
    *,
    omega_0_mask: np.ndarray,
    unit_codes_arr: np.ndarray,
    time_codes_arr: np.ndarray,
    units_in_omega_0: set,
    n_times: int,
    unit_uniques: List[Any],
) -> None:
    """Raise ``ValueError`` if the Omega_0 bipartite graph is disconnected.

    Stage 1's iterative FE solver identifies ``(mu_i, lambda_t)`` only up to
    component-specific constants per connected component of the bipartite
    graph (supported units on one side, periods on the other; edge =
    Omega_0 row at that (unit, period) cell). If the graph splits into
    K > 1 unit-bearing components, residualization later combines
    ``mu_i`` from one component with ``lambda_t`` from another, silently
    corrupting ``y_tilde`` and downstream ``tau_total`` / ``delta_j``.

    Balanced panel + per-unit/per-period Omega_0 coverage is NECESSARY
    but not SUFFICIENT — connectivity is the load-bearing identification
    condition. Under the current absorbing-treatment + period-strict +
    unit-warn-drop regime this case may be unreachable in practice (we
    were unable to construct an example that survives the upstream
    validators), but the check is defense-in-depth and future-proofs
    Wave B extensions (event-study, survey-design integration, possible
    reversible-treatment relaxations).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    supported_units_sorted = sorted(units_in_omega_0)
    n_supp = len(supported_units_sorted)
    if n_supp <= 1:
        # No multi-component case possible with 0 or 1 supported units.
        return

    supp_unit_to_idx = {code: i for i, code in enumerate(supported_units_sorted)}

    omega_unit_codes = unit_codes_arr[omega_0_mask]
    omega_time_codes = time_codes_arr[omega_0_mask]

    # Every Omega_0 row's unit is by definition in `units_in_omega_0`, so
    # all rows contribute edges to the supported subgraph.
    edge_unit_idx = np.array(
        [supp_unit_to_idx[int(c)] for c in omega_unit_codes],
        dtype=np.int64,
    )
    edge_time_offset = n_supp + np.asarray(omega_time_codes, dtype=np.int64)

    # Symmetric adjacency: nodes 0..n_supp-1 are units, n_supp..n_supp+n_times-1
    # are periods. Edge weights are 1 (presence only).
    rows = np.concatenate([edge_unit_idx, edge_time_offset])
    cols = np.concatenate([edge_time_offset, edge_unit_idx])
    data_ones = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix(
        (data_ones, (rows, cols)),
        shape=(n_supp + n_times, n_supp + n_times),
    )

    _, component_labels = connected_components(adj, directed=False)

    # Count components that contain at least one supported UNIT node.
    # (Period nodes unreachable from any unit form trivial singletons but
    # those would already be caught by the period-level Omega_0 check
    # upstream; here we only fail when there are 2+ unit-bearing
    # components.)
    unit_component_ids = set(int(c) for c in component_labels[:n_supp])
    n_unit_components = len(unit_component_ids)

    if n_unit_components <= 1:
        return

    # Build informative error: name first few units per component.
    component_units: Dict[int, List[Any]] = {}
    for unit_pos, unit_code in enumerate(supported_units_sorted):
        comp = int(component_labels[unit_pos])
        component_units.setdefault(comp, []).append(unit_uniques[unit_code])
    component_summary = "; ".join(
        f"component {comp_id}: {list(units[:3])}"
        + (f" (+{len(units) - 3} more)" if len(units) > 3 else "")
        for comp_id, units in list(sorted(component_units.items()))[:3]
    )
    raise ValueError(
        f"Stage-1 fixed effects unidentified: the Omega_0 bipartite "
        f"graph (supported units linked by shared untreated-and-"
        f"unexposed periods) splits into {n_unit_components} "
        f"disconnected components. Balanced panel and per-unit/per-"
        f"period Omega_0 coverage are NECESSARY but not SUFFICIENT for "
        f"joint identification — the iterative FE solver returns FE only "
        f"up to component-specific constants, and residualization "
        f"combines mu from one component with lambda from another, "
        f"silently corrupting tau_total and delta_j. Examples: "
        f"{component_summary}. To fix, ensure all supported units share "
        f"at least one common Omega_0 period (e.g., add a far-away "
        f"never-treated unit that observes the full time range)."
    )


def _iterative_fe_subset(
    y_full: np.ndarray,
    unit_codes_full: np.ndarray,
    time_codes_full: np.ndarray,
    omega_0_mask: np.ndarray,
    *,
    max_iter: int = _FE_ITER_MAX,
    tol: float = _FE_ITER_TOL,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Stage-1 iterative-alternating-projection FE solver on the Butts subsample.

    Fits ``y[Omega_0] = mu_i + lambda_t + u`` on the untreated-and-unexposed
    rows (``Omega_0_mask`` True). Returns FE arrays indexed by code, with
    ``NaN`` at positions whose unit / time is not represented in the
    subsample (rank-deficient cells).

    Same Gauss-Seidel-on-integer-codes recursion as the shared
    ``diff_diff.utils._iterative_fe_solve`` (which ImputationDiD/TwoStageDiD
    now route through), specialized here to a masked Butts subsample.

    **Wave E.1 weighted path** — when ``weights`` is supplied, the solver
    minimizes ``sum_i w_i * (y_i - mu_i - lambda_t)^2`` (WLS-FE under
    positive weights converges to the same fixed point as the unweighted
    iteration for w == 1). The per-period mean becomes
    ``sum_{i in t} w_i * resid_i / sum_{i in t} w_i`` (weighted bincount
    numerator over weighted bincount denominator). The ``weights is None``
    branch is bit-identical to the pre-Wave-E.1 path so the Wave B/C/D
    no-survey contract is unchanged.

    Parameters
    ----------
    y_full : ndarray of shape (n_rows,)
        Outcome vector for ALL observations (Omega_0 + treated/exposed).
    unit_codes_full : ndarray of shape (n_rows,)
        Integer factor codes per row in ``[0, n_units)``.
    time_codes_full : ndarray of shape (n_rows,)
        Integer factor codes per row in ``[0, n_times)``.
    omega_0_mask : ndarray of shape (n_rows,), bool
        True for rows in the stage-1 fit subsample (D_it=0 AND S_it=0).
    weights : ndarray of shape (n_rows,), optional
        Hájek-normalized survey weights (``sum_i w_i = n``). When provided,
        switches the iteration to WLS-FE; when None, the original unweighted
        bincount path applies.

    Returns
    -------
    unit_fe_arr : ndarray of shape (n_units,)
        Unit FE indexed by code. ``NaN`` for units absent from Omega_0.
    time_fe_arr : ndarray of shape (n_times,)
        Time FE indexed by code. ``NaN`` for periods absent from Omega_0.
    converged : bool
        Whether the iterative solver reached ``tol`` within ``max_iter``.
    """
    if omega_0_mask.sum() == 0:
        raise ValueError(
            "_iterative_fe_subset: Omega_0 (untreated-and-unexposed subsample) "
            "is empty. Cannot fit stage-1 fixed effects. Check that some "
            "control units have d_it > d_bar (Butts Assumption 5(ii))."
        )

    # Wave E.1: when survey weights are supplied, identification support
    # for FE is the POSITIVE-WEIGHT portion of Omega_0. Zero-weight rows
    # are outside the WLS estimating sample (per the registry contract at
    # `docs/methodology/REGISTRY.md` SpilloverDiD "Variance (Wave E.1)"),
    # so any unit / period whose Omega_0 rows all have weight 0 has no
    # identifying contribution and must surface as `NaN` FE (which the
    # downstream `finite_mask` excludes from stage-2). Without this gate,
    # the weighted-bincount denominator collapses to 0 for those groups
    # and `np.where(denom > 0, ...)` writes finite `0.0`, silently
    # corrupting point estimates.
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        omega_0_effective = omega_0_mask & (weights_arr > 0)
        if omega_0_effective.sum() == 0:
            raise ValueError(
                "_iterative_fe_subset: positive-weight Omega_0 is empty "
                "(all untreated-and-unexposed rows have survey_weights == 0). "
                "Stage-1 FE estimation requires at least one Omega_0 row "
                "with strictly positive survey weight."
            )
    else:
        omega_0_effective = omega_0_mask

    n_units = int(unit_codes_full.max()) + 1
    n_times = int(time_codes_full.max()) + 1

    # Operate on the subset only (faster than masking each iteration).
    y_sub = y_full[omega_0_effective]
    unit_sub = unit_codes_full[omega_0_effective]
    time_sub = time_codes_full[omega_0_effective]
    n_sub = len(y_sub)

    # Wave E.1: extract weights subset once outside the iterative loop
    # (mirrors TwoStageDiD's `w_0 = weights[omega_0_mask.values]` cache
    # pattern in `_compute_gmm_variance`).
    w_sub: Optional[np.ndarray] = None
    if weights is not None:
        w_sub = np.asarray(weights, dtype=np.float64)[omega_0_effective]

    alpha = np.zeros(n_sub)
    beta = np.zeros(n_sub)
    converged = False
    for _ in range(max_iter):
        # beta[t] = (weighted) mean over rows in time-group t of (y - alpha)
        resid = y_sub - alpha
        if w_sub is None:
            time_sums = np.bincount(time_sub, weights=resid, minlength=n_times)
            time_denoms = np.bincount(time_sub, minlength=n_times).astype(np.float64)
        else:
            time_sums = np.bincount(time_sub, weights=w_sub * resid, minlength=n_times)
            time_denoms = np.bincount(time_sub, weights=w_sub, minlength=n_times)
        time_means = np.where(time_denoms > 0, time_sums / np.maximum(time_denoms, 1e-300), 0.0)
        beta_new = time_means[time_sub]

        # alpha[i] = (weighted) mean over rows in unit-group i of (y - beta_new)
        resid = y_sub - beta_new
        if w_sub is None:
            unit_sums = np.bincount(unit_sub, weights=resid, minlength=n_units)
            unit_denoms = np.bincount(unit_sub, minlength=n_units).astype(np.float64)
        else:
            unit_sums = np.bincount(unit_sub, weights=w_sub * resid, minlength=n_units)
            unit_denoms = np.bincount(unit_sub, weights=w_sub, minlength=n_units)
        unit_means = np.where(unit_denoms > 0, unit_sums / np.maximum(unit_denoms, 1e-300), 0.0)
        alpha_new = unit_means[unit_sub]

        max_change = max(
            float(np.max(np.abs(alpha_new - alpha))) if n_sub > 0 else 0.0,
            float(np.max(np.abs(beta_new - beta))) if n_sub > 0 else 0.0,
        )
        alpha = alpha_new
        beta = beta_new
        if max_change < tol:
            converged = True
            break

    # Build FE arrays indexed by code; NaN for unseen units/periods.
    unit_fe_arr = np.full(n_units, np.nan, dtype=np.float64)
    time_fe_arr = np.full(n_times, np.nan, dtype=np.float64)
    # For each code present in the subset, take any row's converged value
    # (constant within group at convergence). Sort-by-code to make access
    # deterministic.
    seen_unit_codes = np.unique(unit_sub)
    for u_code in seen_unit_codes:
        idx = np.flatnonzero(unit_sub == u_code)[0]
        unit_fe_arr[u_code] = alpha[idx]
    seen_time_codes = np.unique(time_sub)
    for t_code in seen_time_codes:
        idx = np.flatnonzero(time_sub == t_code)[0]
        time_fe_arr[t_code] = beta[idx]
    return unit_fe_arr, time_fe_arr, converged


def _residualize_butts(
    y_full: np.ndarray,
    unit_codes_full: np.ndarray,
    time_codes_full: np.ndarray,
    unit_fe_arr: np.ndarray,
    time_fe_arr: np.ndarray,
) -> np.ndarray:
    """Compute ``y_tilde = y - mu_hat[i] - lambda_hat[t]`` for ALL rows.

    Rows whose unit or period has ``NaN`` FE (rank-deficient cells from
    stage 1) get ``NaN`` y_tilde and are masked out of stage 2.
    """
    mu_per_row = unit_fe_arr[unit_codes_full]
    lambda_per_row = time_fe_arr[time_codes_full]
    return y_full - mu_per_row - lambda_per_row


def _build_butts_fe_design_csr(
    unit_codes: np.ndarray,
    time_codes: np.ndarray,
    omega_0_mask: np.ndarray,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build sparse FE design matrices for Wave D Gardner GMM correction.

    Column layout: ``[unit_1, ..., unit_{U-1}, time_1, ..., time_{T-1}]``.
    Drops the first unit dummy AND the first time dummy for identification
    (mirrors ``TwoStageDiD._build_fe_design`` at ``two_stage.py:2046``).

    Parameters
    ----------
    unit_codes : np.ndarray of shape (n,)
        Integer codes 0..U-1 (from ``pd.factorize``).
    time_codes : np.ndarray of shape (n,)
        Integer codes 0..T-1 (from ``pd.factorize``).
    omega_0_mask : np.ndarray of shape (n,)
        Boolean mask. ``X_10`` rows where this is False are zeroed out
        (treated AND exposed rows). ``X_1`` keeps all rows.

    Returns
    -------
    X_1 : sparse.csr_matrix, shape (n, (U-1) + (T-1))
        Full-sample FE design with identification dropping.
    X_10 : sparse.csr_matrix, shape (n, (U-1) + (T-1))
        Same column space as ``X_1`` but with ``~omega_0_mask`` rows zeroed.
        Sharing column space is required for the Gardner cross-moment
        ``gamma_hat = (X_10' X_10)^{-1} (X_1' X_2)``.

    Notes
    -----
    Rank-deficient ``X_10' X_10`` (e.g. warn-and-drop units with no
    Omega_0 rows) is detected downstream by ``_compute_gmm_corrected_meat``
    via ``sparse_factorized`` failure → ``np.linalg.lstsq`` fallback with
    a documented ``UserWarning``.

    **Re-factorization on entry:** when callers pass pre-mask integer
    codes that have had interior values dropped via ``finite_mask`` (a
    supported warn-and-drop fit), the input code arrays can be sparse —
    e.g. ``unit_codes = [0, 1, 3, 4]`` with code 2 dropped. Building
    ``X_10`` on the raw codes would materialize an all-zero FE column at
    index 2, forcing ``sparse_factorized`` onto the dense
    ``lstsq``/``XtX_10.toarray()`` fallback unnecessarily (large-memory
    path on big panels). To avoid this, re-factorize via
    :func:`pd.factorize` on entry to compact the code space to
    ``0..n_unique-1`` (no-op when codes are already contiguous; mirrors
    the column-space convention of ``TwoStageDiD._build_fe_design``).
    """
    # Compact the code space before column construction — see Notes.
    unit_codes = pd.factorize(unit_codes)[0]
    time_codes = pd.factorize(time_codes)[0]

    n = unit_codes.shape[0]
    n_units = int(unit_codes.max()) + 1 if n > 0 else 0
    n_times = int(time_codes.max()) + 1 if n > 0 else 0
    n_fe_cols = max(n_units - 1, 0) + max(n_times - 1, 0)

    def _build(mask: Optional[np.ndarray]) -> sparse.csr_matrix:
        # Unit dummies (drop unit_code == 0 for identification).
        u_keep = unit_codes > 0
        if mask is not None:
            u_keep = u_keep & mask
        u_rows = np.flatnonzero(u_keep)
        u_cols = unit_codes[u_keep] - 1

        # Time dummies (drop time_code == 0 for identification).
        t_keep = time_codes > 0
        if mask is not None:
            t_keep = t_keep & mask
        t_rows = np.flatnonzero(t_keep)
        t_cols = (max(n_units - 1, 0)) + (time_codes[t_keep] - 1)

        rows = np.concatenate([u_rows, t_rows])
        cols = np.concatenate([u_cols, t_cols])
        data = np.ones(len(rows), dtype=np.float64)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_fe_cols))

    X_1 = _build(mask=None)
    X_10 = _build(mask=omega_0_mask)
    return X_1, X_10


# =============================================================================
# Public estimator (skeleton — fit() implemented in Step 3)
# =============================================================================


class SpilloverDiD:
    """Ring-indicator spillover-aware DiD (Butts 2021).

    Standalone estimator implementing two-stage Gardner (2022) methodology
    with ring-indicator covariates that identify the direct effect on
    treated units (``tau_total``) alongside per-ring spillover effects on
    near-control units (``delta_j``). Supports both panel non-staggered
    timing and Section 5 staggered timing in a single ``fit()`` entry
    point — non-staggered is the special case where all treated units
    share an onset time.

    Parameters
    ----------
    rings : list of float
        Sorted distance breakpoints with at least 2 elements. ``K =
        len(rings) - 1`` rings are constructed.
    d_bar : float, optional
        Far-away cutoff (Butts Assumption 5). Defaults to ``max(rings)``;
        if explicitly set, must equal ``max(rings)``. Wave B MVP does not
        support a ``d_bar`` strictly larger than the outermost ring edge
        (a "dead zone" where units satisfy ``rings[-1] < d_i <= d_bar``
        but are in neither a ring nor the far-away group has no clean
        methodological interpretation). To use a smaller spillover
        bandwidth, shrink the outermost ring edge instead.
    vcov_type : str, default="hc1"
        Variance estimator. Set to ``"conley"`` and supply
        ``conley_coords``/``conley_cutoff_km``/``conley_lag_cutoff`` to
        enable Conley spatial-HAC at stage 2 (recommended per paper
        Section 3.1).
    conley_coords : tuple of (str, str), optional
        ``(lat_col, lon_col)`` column names. Used for ring construction
        AND for the Conley vcov spatial kernel.
    conley_metric : str or callable, default="haversine"
        Distance metric used for both ring construction and the Conley
        spatial kernel. See :mod:`diff_diff.conley` for callable contract.
    conley_cutoff_km : float, optional
        Conley spatial-HAC bandwidth. Required when ``vcov_type="conley"``.
    conley_lag_cutoff : int, optional
        Within-unit Bartlett max lag. Required when ``vcov_type="conley"``.
        Use ``0`` to suppress the serial-component sandwich.
    cluster : str, optional
        Column name for cluster-robust variance, or the combined Conley
        cluster product kernel when paired with ``vcov_type="conley"``.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    anticipation : int, default=0
        Number of pre-treatment periods where effects may occur. Treatment
        and ring-membership clocks both shift by ``-anticipation`` so the
        stage-1 untreated-and-unexposed subsample correctly excludes
        anticipation rows.
    event_study : bool, default=False
        If ``True``, emit per-event-time × ring coefficients (Butts Table
        2 staggered specification). The result's ``spillover_effects``
        DataFrame uses a ``MultiIndex`` over ``(ring, event_time)``.
    horizon_max : int, optional
        Maximum absolute event-study horizon. Used only when
        ``event_study=True``. Event-times outside ``[-horizon_max,
        +horizon_max]`` are **binned into endpoint pools** (``k <= -H``
        aggregated into a single pre-bin coefficient; ``k >= +H`` into a
        single post-bin coefficient), so no observations are dropped.
        This intentionally **diverges** from
        :class:`diff_diff.two_stage.TwoStageDiD`, which filters rows with
        ``|K| > horizon_max`` out of the stage-2 sample. The endpoint-pool
        semantic honors the library's no-silent-data-drop policy
        (``feedback_no_silent_failures``). When ``None``, the helper
        auto-detects the bin set from observed K values. If ``ref_period
        = -1 - anticipation`` falls outside ``[-horizon_max, +horizon_max]``
        the fit raises ``ValueError``.
    rank_deficient_action : {"warn", "error", "silent"}, default="warn"
        Action when the stage-2 design is rank-deficient.

    Attributes
    ----------
    results_ : SpilloverDiDResults
        Populated after :meth:`fit` completes.
    is_fitted_ : bool

    Notes
    -----
    The implementation uses two-stage Gardner methodology with the
    time-varying ``S_it = S_i * 1{t >= t_treat}`` form (paper page 12,
    just above Equation 5). Reading the literal unit-static ``(1 - D_it) *
    S_i`` from Equation 5 yields a rank-deficient design under TWFE;
    Section 5's Table 2 makes the time-varying form explicit. The
    diff-diff implementation matches the paper's identification argument
    once the ``S_it`` notation is read correctly.

    For non-staggered timing, Gardner identity → stage-2 point estimates
    equal a single-stage TWFE with the time-varying spillover regressor.
    """

    def __init__(
        self,
        *,
        rings: List[float],
        d_bar: Optional[float] = None,
        vcov_type: str = "hc1",
        conley_coords: Optional[Tuple[str, str]] = None,
        conley_metric: SpilloverMetric = "haversine",
        conley_cutoff_km: Optional[float] = None,
        conley_lag_cutoff: Optional[int] = None,
        cluster: Optional[str] = None,
        alpha: float = 0.05,
        anticipation: int = 0,
        event_study: bool = False,
        horizon_max: Optional[int] = None,
        rank_deficient_action: str = "warn",
    ):
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        self.rings = rings
        self.d_bar = d_bar
        self.vcov_type = vcov_type
        self.conley_coords = conley_coords
        self.conley_metric = conley_metric
        self.conley_cutoff_km = conley_cutoff_km
        self.conley_lag_cutoff = conley_lag_cutoff
        self.cluster = cluster
        self.alpha = alpha
        self.anticipation = anticipation
        self.event_study = event_study
        self.horizon_max = horizon_max
        self.rank_deficient_action = rank_deficient_action
        self.is_fitted_ = False
        self.results_: Optional[Any] = None

    def get_params(self) -> Dict[str, Any]:
        return {
            "rings": self.rings,
            "d_bar": self.d_bar,
            "vcov_type": self.vcov_type,
            "conley_coords": self.conley_coords,
            "conley_metric": self.conley_metric,
            "conley_cutoff_km": self.conley_cutoff_km,
            "conley_lag_cutoff": self.conley_lag_cutoff,
            "cluster": self.cluster,
            "alpha": self.alpha,
            "anticipation": self.anticipation,
            "event_study": self.event_study,
            "horizon_max": self.horizon_max,
            "rank_deficient_action": self.rank_deficient_action,
        }

    def set_params(self, **params: Any) -> "SpilloverDiD":
        valid = set(self.get_params().keys())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Unknown parameter: {key!r}. Valid parameters: " f"{sorted(valid)}."
                )
            setattr(self, key, value)
        return self

    # -------------------------------------------------------------------------
    # Fit-time validators (Step 2)
    # -------------------------------------------------------------------------

    def _validate_spillover_inputs(
        self,
        data: pd.DataFrame,
        treatment: Optional[str],
        first_treat: Optional[str],
        time: str,
        unit: str,
        outcome: str,
    ) -> None:
        """Front-door validation for SpilloverDiD.fit().

        Runs BEFORE any stage-1 work. Catches malformed estimator state
        (rings, d_bar), missing/conflicting timing kwargs (treatment XOR
        first_treat), missing required columns, and Conley-specific
        prerequisites. Resolves ``self._effective_d_bar`` as a side
        effect so subsequent helpers can read it directly.

        Raises
        ------
        ValueError
            Any malformed input. Error messages name the offending kwarg
            and (where applicable) the offending row count.
        """
        # 1. rings: sorted list of >= 2 elements, non-negative, strictly increasing.
        if not isinstance(self.rings, (list, tuple, np.ndarray)):
            raise ValueError(
                f"rings must be a list/tuple/array of distance breakpoints; "
                f"got {type(self.rings).__name__}."
            )
        rings_arr = np.asarray(self.rings, dtype=np.float64)
        if rings_arr.ndim != 1 or rings_arr.size < 2:
            raise ValueError(
                "rings must contain at least 2 breakpoints; "
                f"got {len(self.rings)} ({list(self.rings)})."
            )
        if (rings_arr < 0).any():
            raise ValueError(f"rings must be non-negative; got {list(self.rings)}.")
        if (np.diff(rings_arr) <= 0).any():
            raise ValueError(f"rings must be strictly increasing; got {list(self.rings)}.")
        if rings_arr[0] != 0:
            raise ValueError(
                f"rings[0] must equal 0 to cover treated locations "
                f"(d_it = 0 must belong to Ring 1); got rings[0] = "
                f"{rings_arr[0]}. Rows with 0 <= d_it < rings[0] would "
                "be flagged as exposed (S_it = 1) but receive zero "
                "spillover regressors at stage 2, silently biasing the "
                "estimator. To exclude very-close pairs, model that with "
                "an explicit innermost ring covering [0, rings[0])."
            )

        # 2. d_bar: defaults to rings[-1]; if set explicitly must equal rings[-1]
        #    (avoid the dead zone where d_i in (rings[-1], d_bar] is neither
        #    in any ring nor far-away).
        if self.d_bar is None:
            self._effective_d_bar = float(rings_arr[-1])
        else:
            if not np.isfinite(self.d_bar) or self.d_bar <= 0:
                raise ValueError(f"d_bar must be positive and finite; got {self.d_bar}.")
            if not np.isclose(self.d_bar, rings_arr[-1]):
                raise ValueError(
                    f"d_bar ({self.d_bar}) must equal max(rings) ({rings_arr[-1]}); "
                    "to vary d_bar, vary the rings breakpoints (the outermost "
                    "edge is implicitly the spillover cutoff). Setting d_bar "
                    "different from rings[-1] would create a 'dead zone' "
                    "where units in (rings[-1], d_bar] are neither in any "
                    "ring nor in the far-away control group."
                )
            self._effective_d_bar = float(self.d_bar)

        # 3. Exactly ONE of treatment / first_treat must be supplied.
        if treatment is None and first_treat is None:
            raise ValueError(
                "Exactly one of `treatment` (binary D_it column) or "
                "`first_treat` (per-unit onset-time column) must be supplied."
            )
        if treatment is not None and first_treat is not None:
            raise ValueError(
                "Provide either `treatment` or `first_treat`, not both. "
                "`treatment` is auto-converted to `first_treat` internally."
            )

        # 4. Required columns exist in data (treat outcome the same way as
        # other required columns — front-door error rather than late
        # KeyError when `data[outcome]` is dereferenced).
        required = [time, unit, outcome]
        if treatment is not None:
            required.append(treatment)
        if first_treat is not None:
            required.append(first_treat)
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns in data: {missing}.")

        # 4a-bis. Outcome must be finite per-row. Non-finite outcomes
        # propagate into stage-1 FE estimation and surface as non-
        # convergence warnings + late solver failures rather than a
        # targeted input error. Reject up front.
        outcome_arr = np.asarray(data[outcome].values, dtype=np.float64)
        if not np.isfinite(outcome_arr).all():
            n_bad = int((~np.isfinite(outcome_arr)).sum())
            raise ValueError(
                f"outcome column '{outcome}' contains {n_bad} non-finite "
                "value(s) (NaN / Inf). SpilloverDiD requires finite outcomes "
                "for stage-1 FE estimation; impute or drop missing rows "
                "before fitting."
            )

        # 4a-ter. Identifier columns (unit, time, optionally first_treat
        # when user-supplied) must not contain NaN. Missing identifiers
        # would fall through to opaque numpy / pandas errors (e.g.
        # "negative elements" from np.bincount) rather than a targeted
        # ValueError. Reject up front.
        for id_col in (unit, time):
            id_nan_mask = data[id_col].isna()
            if bool(id_nan_mask.any()):
                n_nan = int(id_nan_mask.sum())
                raise ValueError(
                    f"identifier column '{id_col}' contains {n_nan} "
                    "missing value(s). SpilloverDiD requires valid "
                    "unit / time identifiers on every row; drop or "
                    "impute missing-identifier rows before fitting."
                )
        # `first_treat` is checked only when user-supplied; the auto-
        # generated path produces a clean column.
        if first_treat is not None and first_treat in data.columns:
            ft_nan_mask = data[first_treat].isna()
            if bool(ft_nan_mask.any()):
                n_nan = int(ft_nan_mask.sum())
                raise ValueError(
                    f"first_treat column '{first_treat}' contains {n_nan} "
                    "missing value(s). Use np.inf (or 0) for never-treated "
                    "units; do not leave NaN."
                )

        # 4b. One-row-per-(unit, time) cell panel contract. Duplicate cells
        # would silently re-weight stage-1 FE estimation AND stage-2 OLS
        # without any warning. Reject up front.
        cell_counts = data.groupby([unit, time]).size()
        dups = cell_counts[cell_counts > 1]
        if len(dups) > 0:
            sample = list(dups.index[:5])
            suffix = f" (and {len(dups) - 5} more)" if len(dups) > 5 else ""
            raise ValueError(
                f"{len(dups)} duplicate (unit, time) cell(s) detected "
                f"(e.g. {sample}{suffix}). SpilloverDiD requires "
                "one-row-per-(unit, time) panel data — duplicate cells "
                "would silently re-weight both the stage-1 FE fit and the "
                "stage-2 OLS. Aggregate to unique cells before fitting."
            )

        # 4c. Balanced-panel contract for the Wave B MVP. An unbalanced
        # panel where the stage-1 (unit, time) FE bipartite graph induced
        # by Omega_0 isn't connected produces unidentified residuals on
        # treated rows. The exact-graph-connectivity check is queued as
        # a follow-up; the MVP simply rejects panels where some unit
        # doesn't observe every period.
        n_unique_times = data[time].nunique()
        unit_period_counts = data.groupby(unit)[time].nunique()
        underbalanced = unit_period_counts[unit_period_counts < n_unique_times]
        if len(underbalanced) > 0:
            sample = list(underbalanced.index[:5])
            suffix = f" (and {len(underbalanced) - 5} more)" if len(underbalanced) > 5 else ""
            raise ValueError(
                f"Unbalanced panel: {len(underbalanced)} unit(s) do not "
                f"observe every period (panel has {n_unique_times} unique "
                f"periods, affected units e.g. {sample}{suffix}). Wave B "
                "MVP requires a balanced panel — an unbalanced (unit, time) "
                "Omega_0 bipartite graph can produce unidentified residuals "
                "for some treated rows even when every unit and every "
                "period has at least one Omega_0 row. Balance the panel "
                "(impute missing cells or drop affected units) before "
                "fitting. Graph-connectivity-based identification is "
                "queued as a follow-up extension."
            )

        # 5a. conley_coords is required ALWAYS — ring construction dereferences
        # it on every fit() path, regardless of vcov_type. Validate up front
        # rather than letting downstream code fail with AssertionError/KeyError.
        if self.conley_coords is None:
            raise ValueError(
                "SpilloverDiD requires `conley_coords=(lat_col, lon_col)` "
                "for ring construction, regardless of vcov_type."
            )
        if not isinstance(self.conley_coords, (list, tuple)) or len(self.conley_coords) != 2:
            raise ValueError(
                "conley_coords must be a 2-tuple (lat_col, lon_col); "
                f"got {self.conley_coords!r}."
            )
        # Within-unit coord constancy: ring construction collapses coords to
        # one row per unit via drop_duplicates(subset=[unit]). If a unit's
        # lat/lon varies across rows the first observed value is silently
        # used; reject up front rather than silently misclassify spillover
        # exposure.
        coord_cols = list(self.conley_coords)
        if unit in data.columns and all(c in data.columns for c in coord_cols):
            per_unit_unique = data.groupby(unit)[coord_cols].nunique()
            non_constant = per_unit_unique[(per_unit_unique > 1).any(axis=1)]
            if len(non_constant) > 0:
                sample = non_constant.index.tolist()[:5]
                suffix = f" (and {len(non_constant) - 5} more)" if len(non_constant) > 5 else ""
                raise ValueError(
                    f"{len(non_constant)} unit(s) have non-constant "
                    f"conley_coords ({coord_cols}) across rows (e.g. {sample}"
                    f"{suffix}). SpilloverDiD requires within-unit-constant "
                    "coordinates — ring construction collapses coords per "
                    "unit via drop_duplicates. Aggregate to a single (lat, "
                    "lon) per unit (e.g. via the unit's geographic centroid) "
                    "before fitting, or fix the data so coords are constant."
                )
        for c in self.conley_coords:
            if c not in data.columns:
                raise ValueError(f"conley_coords column '{c}' not in data.")
        # Coord finiteness check (per-row).
        coord_vals = data[list(self.conley_coords)].values
        coord_arr = np.asarray(coord_vals, dtype=np.float64)
        if not np.isfinite(coord_arr).all():
            n_nonfinite = int((~np.isfinite(coord_arr)).any(axis=1).sum())
            raise ValueError(
                f"conley_coords contain non-finite values in {n_nonfinite} row(s); "
                "coordinates must be finite for distance computation."
            )
        # Haversine lat/lon domain check: applies on EVERY vcov path (not just
        # vcov_type='conley') because ring construction always uses
        # conley_metric for distance computation. Out-of-range coords silently
        # produce wrong ring assignment otherwise.
        if self.conley_metric == "haversine":
            lat_arr = coord_arr[:, 0]
            lon_arr = coord_arr[:, 1]
            if (lat_arr < -90.0).any() or (lat_arr > 90.0).any():
                bad_rows = int(((lat_arr < -90.0) | (lat_arr > 90.0)).sum())
                raise ValueError(
                    f"conley_coords latitude column '{coord_cols[0]}' contains "
                    f"{bad_rows} row(s) outside [-90, 90] degrees. Haversine "
                    "metric requires geographic lat/lon coords; if your coords "
                    "are already projected (planar), pass conley_metric='euclidean'."
                )
            if (lon_arr < -180.0).any() or (lon_arr > 180.0).any():
                bad_rows = int(((lon_arr < -180.0) | (lon_arr > 180.0)).sum())
                raise ValueError(
                    f"conley_coords longitude column '{coord_cols[1]}' contains "
                    f"{bad_rows} row(s) outside [-180, 180] degrees. Haversine "
                    "metric requires geographic lat/lon coords; if your coords "
                    "are already projected (planar), pass conley_metric='euclidean'."
                )

        # 5b. cluster column existence + NaN check — applies on every vcov
        # path, not just conley. Missing cluster ids would produce wrong
        # SEs (NaN counted as its own cluster by np.unique() but dropped
        # by pandas groupby() in the cluster meat).
        if self.cluster is not None:
            if self.cluster not in data.columns:
                raise ValueError(f"cluster column '{self.cluster}' not in data.")
            cluster_nan_mask = data[self.cluster].isna()
            if bool(cluster_nan_mask.any()):
                n_nan = int(cluster_nan_mask.sum())
                raise ValueError(
                    f"cluster column '{self.cluster}' contains {n_nan} "
                    "missing value(s). NaN cluster ids would silently "
                    "produce wrong clustered SEs (np.unique counts NaN as "
                    "its own cluster but pandas groupby drops it from the "
                    "cluster meat). Drop or impute missing cluster rows "
                    "before fitting."
                )

        # 5c. Conley-specific kwargs (only required when vcov_type='conley').
        if self.vcov_type == "conley":
            if self.conley_cutoff_km is None or not (
                np.isfinite(self.conley_cutoff_km) and self.conley_cutoff_km > 0
            ):
                raise ValueError(
                    "vcov_type='conley' requires conley_cutoff_km > 0 (finite); "
                    f"got {self.conley_cutoff_km}."
                )
            if self.conley_lag_cutoff is None or self.conley_lag_cutoff < 0:
                raise ValueError(
                    "vcov_type='conley' requires conley_lag_cutoff >= 0 (integer); "
                    f"got {self.conley_lag_cutoff}."
                )

        # 6. At least one treated unit must exist.
        if treatment is not None:
            n_treated_obs = int((data[treatment] == 1).sum())
            if n_treated_obs == 0:
                raise ValueError(
                    f"No treated observations found (column '{treatment}' "
                    "is all 0/NaN). SpilloverDiD requires at least one treated unit."
                )
        else:
            ft_finite = np.isfinite(data[first_treat].astype(float).values)  # type: ignore[arg-type]
            n_treated_units = int(
                pd.Series(ft_finite & (data[first_treat].astype(float).values != 0)).any()  # type: ignore[index]
            )
            if not n_treated_units:
                raise ValueError(
                    f"No treated units found (column '{first_treat}' is "
                    "all 0 / inf / NaN). SpilloverDiD requires at least one "
                    "treated unit."
                )

    def _validate_far_away_exists(
        self,
        d_array: np.ndarray,
        is_control_array: np.ndarray,
    ) -> int:
        """Verify Butts Assumption 5(ii): at least one (D=0, d > d_bar) observation.

        Parameters
        ----------
        d_array : ndarray
            Per-unit or per-row distances (caller chooses; the check is
            count-based, not granularity-sensitive).
        is_control_array : ndarray, bool
            Aligned mask: True where the observation belongs to a control
            unit (D_i = 0 for static, D_it = 0 for staggered).

        Returns
        -------
        n_far_away : int
            Number of far-away control observations.

        Raises
        ------
        ValueError
            No (D=0, d > d_bar) observations exist; Assumption 5(ii) fails.
        """
        d_bar = self._effective_d_bar
        far_away_mask = (d_array > d_bar) & is_control_array
        n_far_away = int(far_away_mask.sum())
        if n_far_away < 1:
            raise ValueError(
                "No far-away control observations: every control unit has "
                f"d_i <= d_bar = {d_bar}. Butts (2021) Assumption 5(ii) "
                "requires the sample to contain control units strictly "
                "further than d_bar from any treated unit. Either reduce "
                "d_bar (via the outermost ring breakpoint), expand the sample, "
                "or verify the coords/metric configuration."
            )
        return n_far_away

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        unit: str,
        time: str,
        treatment: Optional[str] = None,
        first_treat: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        survey_design: object = None,
    ) -> SpilloverDiDResults:
        """Fit the two-stage Gardner DiD with ring-indicator covariates.

        Methodology (Butts 2021 Section 5 + Gardner 2022):

        1. Compute per-row spillover indicators from ``conley_coords``.
        2. Build stage-1 subsample ``Omega_0 = {D_it=0 AND S_it=0}``
           (untreated AND unexposed) — Butts' clean control group.
        3. Stage 1: fit ``Y_it = mu_i + lambda_t + u`` on ``Omega_0``.
        4. Residualize: ``Y_tilde = Y - mu_hat - lambda_hat`` for ALL rows.
        5. Stage 2: regress ``Y_tilde`` on ``[D_it, (1-D_it)*Ring_{it,j}]`` via
           :func:`solve_ols`, threading the configured ``vcov_type``.
        6. Wrap as :class:`SpilloverDiDResults`.

        Notes
        -----
        Stage-2 variance applies the Wave D Gardner (2022) GMM first-stage
        uncertainty correction across all supported ``vcov_type`` paths
        (``"hc1"``, ``"conley"``, ``"cluster"`` via ``cluster=<col>``). The
        unified IF outer-product formula is
        ``psi_i = gamma_hat' * X_{10,i} * eps_{10,i} - X_{2,i} * eps_{2,i}``
        with ``meat = Psi' K Psi`` where ``K`` is path-dependent (identity
        for HC1, block-indicator for cluster, spatial kernel for Conley).
        Documented synthesis of Butts (2021) §3.1 + Gardner (2022) §4 +
        Conley (1999); no reference software combines all three.
        ``vcov_type="classical"`` raises ``NotImplementedError`` because
        the Wave D synthesis has not been derived for the homoskedastic
        meat structure ``sigma_hat^2 * (X_10' X_10)``; use ``"hc1"`` for
        heteroskedasticity-robust SE with the GMM correction.
        """
        # Wave E.1: lift the Wave B/C/D upfront survey_design rejection.
        # Wave E.2 (this PR): conley × survey is now supported via a
        # stratified-Conley sandwich on PSU totals (composition of Conley
        # 1999 + Gerber 2026 Prop 1 Binder TSL + Wave D Gardner GMM). The
        # full resolution block (pweight gate, replicate gate, unit-constant
        # check, cluster-vs-PSU warn) runs AFTER `_validate_spillover_inputs`
        # below so it sees the panel columns the validator guarantees.
        #
        # Wave E.2 follow-up (shipped): `vcov_type='conley' + conley_lag_cutoff > 0
        # + survey_design=` is supported via panel-block stratified-Conley
        # sandwich (spatial Wave E.2 term + within-PSU serial Bartlett HAC)
        # WHEN there is an effective PSU (explicit `survey_design.psu` OR
        # injected via `cluster=<col>` per Wave E.1's `_inject_cluster_as_psu`
        # routing). The orchestrator at
        # `two_stage.py::_compute_stratified_conley_meat` sums the two terms
        # with disjoint index sets — matches the no-survey panel-block
        # decomposition at `conley.py::_compute_conley_meat` (Conley 1999
        # spatial + Newey-West 1987 serial Bartlett; separable form, NOT
        # Driscoll-Kraay 2D-HAC). FPC convention: per-period FPC on spatial,
        # panel-wide stratum-level FPC on serial. The no-effective-PSU
        # fail-closed gate is downstream at the post-resolution check (see
        # the `resolved_survey_fit.psu is None` block below the cluster
        # injection); the gate cannot live up here because at this point
        # the user-supplied `cluster=<col>` has not yet been injected into
        # the survey design as the effective PSU.
        # Validate `anticipation` up front: must be a non-negative integer.
        # Accepting fractional or negative values would silently shift
        # treatment timing and ring exposure beyond what the estimator's
        # identification contract supports. Validated BEFORE the
        # event_study / horizon_max checks because the ref_period
        # compatibility check below computes `-1 - self.anticipation` and
        # would otherwise raise a raw TypeError on non-numeric input
        # (PR #456 R2 fix).
        if not isinstance(self.anticipation, (int, np.integer)) or self.anticipation < 0:
            raise ValueError(
                f"anticipation must be a non-negative integer; got "
                f"{self.anticipation!r} (type {type(self.anticipation).__name__})."
            )
        # Wave C: event-study path is now supported. Validate horizon_max
        # up front (fail-fast before any stage-1 work).
        if self.horizon_max is not None:
            if not isinstance(self.horizon_max, (int, np.integer)) or self.horizon_max < 0:
                raise ValueError(
                    f"horizon_max must be a non-negative integer or None; "
                    f"got {self.horizon_max!r} "
                    f"(type {type(self.horizon_max).__name__})."
                )
            # Reject horizon_max=0 under event_study=True (PR #456 R4 fix).
            # H=0 puts the entire panel into a single k=0 bin and the
            # reference period -1-anticipation always falls outside [-0, +0],
            # so the ref_period guard below would reject it anyway. We
            # surface a clearer error explaining the right alternative:
            # users wanting "one aggregate effect" should use
            # event_study=False (Wave B static spec); event-study mode
            # requires at least one event-time bin pair so a reference
            # period can be anchored.
            if self.event_study and self.horizon_max == 0:
                raise ValueError(
                    "horizon_max=0 is not supported when event_study=True: "
                    "the single bin k=0 leaves no event-time pair to anchor "
                    "the reference period against. For a single aggregate "
                    "direct effect, use event_study=False (Wave B static "
                    "spec); for the event-study decomposition, use "
                    "horizon_max>=1 or horizon_max=None (auto-detect)."
                )
            if not self.event_study and self.horizon_max is not None:
                # horizon_max is only meaningful in event-study mode.
                warnings.warn(
                    "horizon_max is ignored when event_study=False (it controls "
                    "event-time binning in the per-event-time design). Set "
                    "event_study=True to use horizon_max.",
                    UserWarning,
                    stacklevel=2,
                )
        # Lock the ref_period × horizon_max compatibility: the reference period
        # must fall inside the binning window or silently floor would change
        # identification (rejected per `feedback_no_silent_failures`).
        if self.event_study and self.horizon_max is not None:
            ref_period_check = -1 - self.anticipation
            if ref_period_check < -self.horizon_max:
                raise ValueError(
                    f"Reference period (-1 - anticipation = {ref_period_check}) "
                    f"falls outside the binning window [-{self.horizon_max}, "
                    f"+{self.horizon_max}]. Either reduce anticipation "
                    f"(currently {self.anticipation}) or increase horizon_max "
                    f"(currently {self.horizon_max}) so the reference period "
                    f"falls inside the window. Silently shifting the reference "
                    f"to -horizon_max would change identification."
                )
        if covariates is not None and len(covariates) > 0:
            raise NotImplementedError(
                "SpilloverDiD does not yet support covariates= in Wave B MVP. "
                "The Gardner-style two-stage pattern requires covariate "
                "effects to be estimated on the untreated-and-unexposed "
                "subsample at stage 1 and subtracted from Y before stage 2 — "
                "appending them only at stage 2 (without stage-1 "
                "residualization) would silently bias tau_total / delta_j on "
                "panels with time-varying covariates. The full covariate "
                "path mirroring TwoStageDiD._fit_untreated_model is queued as "
                "a follow-up extension. See TODO.md."
            )
        if self.vcov_type in ("hc2", "hc2_bm"):
            raise NotImplementedError(
                f"SpilloverDiD does not yet support vcov_type='{self.vcov_type}'. "
                "The current stage-2 inference uses a generic residual df "
                "(n - effective_rank) for t-distribution lookups, but "
                "hc2 / hc2_bm require per-coefficient Bell-McCaffrey / CR2 "
                "degrees of freedom for correct p-values and CIs. Routing "
                "stage 2 through LinearRegression (which supplies the "
                "per-coefficient DOF metadata) is queued as a follow-up "
                "extension. Use vcov_type='hc1' or 'conley', or "
                "leave default; combine with cluster=<col> for CR1."
            )
        if self.vcov_type == "classical":
            # Wave D scope (user-confirmed 2026-05-17): the Gardner GMM
            # first-stage uncertainty correction is implemented for HC1,
            # Conley, and CR1 only. The classical (homoskedastic) variance
            # has not been derived for the IF outer-product form in this
            # PR — under classical assumptions the meat structure changes
            # (`sigma_hat^2 * (X_10' X_10)` rather than `Psi' Psi`) and
            # the Wave D synthesis (Butts §3.1 + Gardner §4 + Conley 1999)
            # does not carry through directly. Reject upfront with a clear
            # remediation message rather than silently HC1-ifying the
            # request (per `feedback_no_silent_failures`).
            raise NotImplementedError(
                "SpilloverDiD does not support vcov_type='classical' under "
                "the Wave D Gardner GMM first-stage uncertainty correction. "
                "Wave D applies the GMM correction unconditionally and the "
                "classical homoskedastic variance does not have a derived "
                "IF outer-product form in the Wave D synthesis (Butts §3.1 "
                "+ Gardner §4 + Conley 1999). Use vcov_type='hc1' for "
                "heteroskedasticity-robust SE with the GMM correction, or "
                "combine with cluster=<col> for CR1 with the GMM correction. "
                "Future PR may extend Wave D to the classical path."
            )

        # Step 0: defensive copy so the caller's DataFrame is never mutated.
        data = data.copy(deep=False)

        # Step 0b: coerce `time` to numeric BEFORE any structural validation.
        # The validator's duplicate-cell and balanced-panel checks depend on
        # period IDENTITY; mixed raw encodings like ['0', 0, '1', 1] would
        # pass validation but collapse to duplicate periods after coercion.
        # Coercing first ensures validation sees the actual numeric labels.
        if time in data.columns:
            try:
                data = data.assign(**{time: pd.to_numeric(data[time])})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"time column '{time}' must be numeric (or string-coercible "
                    f"to numeric). Got: {exc}. Encode periods as integers / "
                    "floats before passing to SpilloverDiD."
                ) from exc
        # User-supplied first_treat must also be coerced BEFORE validation
        # so the NaN check and identity-based checks see the actual labels.
        # Auto-generated `_spillover_first_treat` (from binary D) doesn't
        # exist yet — it's created later by `_convert_treatment_to_first_treat`.
        if first_treat is not None and first_treat in data.columns:
            try:
                data = data.assign(**{first_treat: pd.to_numeric(data[first_treat])})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"first_treat column '{first_treat}' must be numeric (or "
                    f"string-coercible to numeric). Got: {exc}. Encode onset "
                    "times as integers / floats (or np.inf for never-treated) "
                    "before passing to SpilloverDiD."
                ) from exc

        # Step 1: front-door validation (rings, d_bar, timing-kwargs XOR,
        # coords, panel structure — all on COERCED time/first_treat labels).
        self._validate_spillover_inputs(data, treatment, first_treat, time, unit, outcome)

        # Step 1b (Wave E.1): survey-design resolution + validation.
        #
        # Mirrors TwoStageDiD's resolution block at `two_stage.py:485-511`.
        # Returns (resolved_survey, survey_weights, weight_type, survey_metadata)
        # 4-tuple, all None when `survey_design is None`. Weights are
        # Hájek-normalized (sum_i w_i = n) so the downstream gamma_hat solve
        # + Psi construction + bread inversion produce design-consistent
        # variance per Gerber (2026) Proposition 1.
        from diff_diff.survey import (
            _inject_cluster_as_psu,
            _resolve_effective_cluster,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            # Wave E.1 supports pweight only — fweight / aweight semantics
            # do not match Gerber (2026) Proposition 1's stratified-cluster
            # Taylor linearization.
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"SpilloverDiD survey support requires weight_type='pweight', "
                    f"got '{resolved_survey.weight_type}'. The Wave E.1 Binder "
                    f"TSL variance assumes probability weights; see "
                    f"docs/methodology/REGISTRY.md SpilloverDiD section."
                )
            # Wave E.1: replicate-weight variance is deferred as separable
            # follow-up scope. Per Gerber (2026) Appendix A, the IF-reweighting
            # shortcut does NOT apply to TwoStageDiD-class estimators because
            # gamma_hat is weight-sensitive — replicate path requires per-
            # replicate full re-fit.
            if resolved_survey.uses_replicate_variance:
                raise NotImplementedError(
                    "SpilloverDiD does not yet support replicate-weight variance "
                    "(BRR / Fay / JK1 / JKn / SDR). Per Gerber (2026) Appendix A, "
                    "the IF-reweighting shortcut does not apply because gamma_hat "
                    "is weight-sensitive; correct support requires per-replicate "
                    "full re-fit of stage 1 and stage 2. Queued as a follow-up. "
                    "See TODO.md."
                )

        # Step 2: convert binary treatment to per-unit first_treat if needed.
        # Track whether `first_treat` was AUTO-GENERATED (from a binary D
        # column) vs USER-SUPPLIED (Gardner convention). The auto-generated
        # column uses ONLY np.inf for never-treated (no 0-as-never-treated
        # sentinel); preserving this distinction avoids silently
        # reclassifying baseline-treated units (D=1 at t=0) as never-treated.
        treatment_auto_converted = treatment is not None
        if treatment is not None:
            data, first_treat = _convert_treatment_to_first_treat(data, treatment, time, unit)
        assert first_treat is not None  # validator guarantees this

        # Step 3: factorize unit/time → integer codes (mirrors TwoStageDiD).
        unit_vals = data[unit].values
        time_vals = data[time].values
        unit_codes_full, unit_uniques = pd.factorize(pd.Series(unit_vals), sort=True)
        time_codes_full, time_uniques = pd.factorize(pd.Series(time_vals), sort=True)

        # Step 4: extract treatment onsets per unit; detect staggered.
        first_treat_by_unit = _extract_treatment_onsets(
            data,
            first_treat,
            unit,
            treat_zero_as_never_treated=not treatment_auto_converted,
        )
        finite_onsets = {ft for ft in first_treat_by_unit.values() if np.isfinite(ft)}
        if not finite_onsets:
            raise ValueError(
                "No treated units found (all first_treat values are inf or 0). "
                "SpilloverDiD requires at least one treated unit."
            )
        is_staggered = len(finite_onsets) > 1

        # Apply anticipation shift to onsets used for ring construction AND
        # for the D_it indicator (treatment-effective onset).
        effective_onsets = {
            uid: (ft - self.anticipation if np.isfinite(ft) else ft)
            for uid, ft in first_treat_by_unit.items()
        }

        # Step 5: compute per-row d_it. For non-staggered (single common
        # onset), use the cheaper static helper that builds the pairwise
        # distance matrix once; for staggered, use the per-cohort helper
        # that handles time-varying ring membership.
        assert self.conley_coords is not None  # validator-guaranteed

        # If conley_metric is a user callable, validate it against the full
        # 6-check contract (shape / finite / non-negative / symmetric /
        # zero-diagonal) on the per-unit (n, n) self-call BEFORE using it
        # for ring construction. Without this, a callable with positive
        # self-distance silently corrupts ring assignment (treated units
        # at their own location should have d=0 → fall in Ring_1; positive
        # self-distance pushes them out into a different ring).
        if callable(self.conley_metric):
            unit_coords_for_validation = (
                data[list(self.conley_coords)].drop_duplicates().values.astype(np.float64)
            )
            _validate_callable_metric_result(
                self.conley_metric(unit_coords_for_validation, unit_coords_for_validation),
                unit_coords_for_validation.shape[0],
            )

        # Capture the spillover-trigger onsets alongside d_it on the
        # staggered path so the event-study branch below can reuse them
        # without redoing the cohort distance loop (PR #456 R6 perf fix).
        trigger_onset_per_row_cached: Optional[np.ndarray] = None
        if is_staggered:
            d_it_per_row, _, _, trigger_onset_per_row_cached = (
                _compute_nearest_treated_distance_staggered(
                    data,
                    unit=unit,
                    time=time,
                    coords=self.conley_coords,
                    metric=self.conley_metric,
                    first_treat_by_unit=effective_onsets,
                    d_bar=self._effective_d_bar if self.event_study else None,
                )
            )
        else:
            # Non-staggered: single common onset. Build d_i per unit once,
            # then broadcast to per-row AND zero out pre-treatment rows
            # (matching the staggered helper's inf-at-pre-treatment
            # convention so downstream ring + Omega_0 logic is timing-
            # agnostic).
            ever_treated_ids = np.array(
                [uid for uid, ft in first_treat_by_unit.items() if np.isfinite(ft)],
                dtype=object,
            )
            d_i_per_unit, unit_index_static = _compute_nearest_treated_distance_static(
                data,
                unit=unit,
                coords=self.conley_coords,
                metric=self.conley_metric,
                treated_unit_ids=ever_treated_ids,
                # Pass `d_bar` as the cutoff so the cKDTree sparse path
                # auto-activates when n_units > _CONLEY_SPARSE_N_THRESHOLD
                # for built-in metrics. Units beyond d_bar get d_i = inf,
                # which the downstream ring builder treats as far-away
                # controls — same as the dense-path semantics.
                cutoff_km=self._effective_d_bar,
            )
            unit_to_d = {uid: float(d_i_per_unit[idx]) for idx, uid in enumerate(unit_index_static)}
            d_it_per_row = np.array([unit_to_d.get(u, np.inf) for u in unit_vals])
            # Pre-treatment rows have d_it=inf (no unit treated yet).
            shared_onset = next(iter(finite_onsets))
            shared_effective_onset = shared_onset - self.anticipation
            d_it_per_row = np.where(
                np.asarray(time_vals, dtype=np.float64) < shared_effective_onset,
                np.inf,
                d_it_per_row,
            )
            # PR #456 R7 perf fix: derive trigger_onset_per_row directly
            # from the static distance result for the event-study path. In
            # the non-staggered case there's only one cohort onset, so the
            # trigger collapses to the shared effective onset for any unit
            # within d_bar (NaN for far-away units). Avoids hitting
            # `_compute_event_time_per_row`'s dense-fallback cohort loop.
            if self.event_study:
                d_per_unit_inrange = np.array(
                    [
                        (
                            shared_effective_onset
                            if unit_to_d.get(u, np.inf) <= self._effective_d_bar
                            else np.nan
                        )
                        for u in unit_vals
                    ],
                    dtype=np.float64,
                )
                trigger_onset_per_row_cached = d_per_unit_inrange

        # Step 6: build ring indicators per row (Butts Eq 6 time-varying form).
        ring_masks = _build_ring_indicators(d_it_per_row, list(self.rings))
        K = ring_masks.shape[1]

        # Step 7: compute D_it per row (with anticipation shift).
        D_it = np.zeros(len(data), dtype=np.float64)
        for u_id, eff_ft in effective_onsets.items():
            if np.isfinite(eff_ft):
                rows = (unit_vals == u_id) & (np.asarray(time_vals) >= eff_ft)
                D_it[rows] = 1.0

        # Step 7b: verify at least one observation is treated AFTER applying
        # the anticipation shift. If all first_treat values are > max(time)
        # in the panel (e.g. an "anticipation" of treatment that hasn't
        # arrived yet), D_it is all zeros and the stage-2 design has no
        # treatment variation. Fail fast with a clear identification error
        # rather than crashing inside solve_ols.
        if D_it.sum() == 0:
            max_time = float(np.max(np.asarray(time_vals, dtype=np.float64)))
            raise ValueError(
                "No observation is treated in-sample after applying "
                f"anticipation shift of {self.anticipation}. The earliest "
                "effective onset is later than the latest observed period "
                f"({max_time}), so D_it = 0 everywhere and tau_total is "
                "unidentified. Either include post-onset periods in the "
                "panel, reduce the anticipation lead, or verify the "
                "first_treat column."
            )

        # Step 8: compute S_it = 1{d_it <= d_bar}. Treated-self rows have
        # d_it=0 → S_it=1 (Omega_0 excludes them; they're treated anyway).
        S_it = (d_it_per_row <= self._effective_d_bar).astype(np.float64)

        # Step 9: validate far-away controls (Butts Assumption 5(ii)).
        # Use CURRENT-period untreated status, not never-treated-only. The
        # paper defines Omega_0 row-wise as {D_it = 0 AND S_it = 0}, so
        # not-yet-treated observations of eventually-treated units can also
        # contribute to the far-away identifying group. This matters for
        # all-eventually-treated staggered designs (no never-treated units).
        is_control_row_now = D_it == 0
        # Validate far-away rows exist (Assumption 5(ii)). Discard the
        # full-domain count return — Wave E.3 (codex R11 P2 fix)
        # recomputes the REPORTED `n_far_away_obs` on the effective
        # estimation sample (`count_mask`) at result-assembly time so
        # the reported metadata matches n_obs / n_treated / n_control
        # under SurveyDesign.subpopulation().
        self._validate_far_away_exists(d_it_per_row, is_control_row_now)

        # Step 10: Butts Omega_0 mask = (D_it=0 AND S_it=0).
        omega_0_mask = (D_it == 0) & (S_it == 0)

        # Wave E.1: under survey_design, identification support is the
        # POSITIVE-WEIGHT portion of Omega_0. Zero-weight rows are outside
        # the WLS estimating sample (per the registry contract); using raw
        # Omega_0 for unsupported / connectivity checks would let zero-
        # weight rows masquerade as identifying support — silently wrong
        # `att` / ring effects / vcov when a raw Omega_0 bridge has zero
        # weight (positive-weight Omega_0 subgraph disconnected) or when
        # a period's only Omega_0 rows all have weight 0 (time FE
        # unidentified despite passing raw-membership checks).
        if survey_weights is not None:
            omega_0_effective = omega_0_mask & (np.asarray(survey_weights) > 0)
        else:
            omega_0_effective = omega_0_mask

        # Step 10b: row-level Omega_0 identification check.
        #
        # Two regimes (round-16 codex review split):
        #   - PERIOD-level unsupported (no Omega_0 row at some t): time FE
        #     structurally unidentified. Dropping the period would remove
        #     ALL units' observations at that t, including the far-away
        #     rows needed for identification. Hard error.
        #   - UNIT-level unsupported (no Omega_0 row for some i): warn-
        #     and-drop. Unit FE for that i is NaN, residualization writes
        #     NaN on those rows, and the downstream finite_mask path at
        #     Step 14 excludes them from stage 2. Mirrors `TwoStageDiD`'s
        #     always-treated unit handling (`two_stage.py:294-336`) and
        #     Gardner's framework, which identifies effects from supported
        #     observations rather than requiring every unit estimable.
        unit_codes_arr = np.asarray(unit_codes_full)
        time_codes_arr = np.asarray(time_codes_full)
        units_in_omega_0 = set(unit_codes_arr[omega_0_effective].tolist())
        times_in_omega_0 = set(time_codes_arr[omega_0_effective].tolist())
        all_unit_codes = set(unit_codes_arr.tolist())
        all_time_codes = set(time_codes_arr.tolist())
        unsupported_units = sorted(all_unit_codes - units_in_omega_0)
        unsupported_periods = sorted(all_time_codes - times_in_omega_0)

        if unsupported_periods:
            affected = [time_uniques[c] for c in unsupported_periods[:5]]
            suffix = (
                f" (and {len(unsupported_periods) - 5} more)"
                if len(unsupported_periods) > 5
                else ""
            )
            raise ValueError(
                f"Stage-1 fixed effects unidentified: "
                f"{len(unsupported_periods)} period(s) have NO untreated-and-"
                f"unexposed (Omega_0) rows — their time FE is unidentified. "
                f"Examples: {affected}{suffix}. The Butts subsample "
                "Omega_0 = {D_it = 0 AND S_it = 0} must contain at least one "
                "row per period that appears in the data. Consider "
                "tightening d_bar (so fewer rows are flagged as exposed "
                "S_it = 1) or expanding the sample to include never-treated "
                "or pre-treatment observations for the affected periods."
            )

        if unsupported_units:
            affected = [unit_uniques[c] for c in unsupported_units[:5]]
            suffix = (
                f" (and {len(unsupported_units) - 5} more)" if len(unsupported_units) > 5 else ""
            )
            warnings.warn(
                f"SpilloverDiD: {len(unsupported_units)} unit(s) have NO "
                f"untreated-and-unexposed (Omega_0) rows — their unit FE "
                f"is unidentified and their rows will be excluded from "
                f"stage 2 estimation. Examples: {affected}{suffix}. To "
                f"include these units, expand the sample to provide pre-"
                f"treatment or untreated observations for them, or tighten "
                f"d_bar so fewer rows are flagged as exposed (S_it = 1).",
                UserWarning,
                stacklevel=2,
            )

        # Step 10c: connected-component check on the Omega_0 bipartite graph.
        #
        # Stage 1's iterative FE solver identifies (mu_i, lambda_t) only up
        # to component-specific constants per connected component of the
        # bipartite graph (supported units ↔ periods, edge = Omega_0 row).
        # If the graph splits into K > 1 components, _residualize_butts then
        # combines mu_i from one component with lambda_t from another,
        # silently corrupting y_tilde and downstream tau_total / delta_j.
        # Balanced panel + per-unit/per-period Omega_0 coverage is NECESSARY
        # but not SUFFICIENT — connectivity is the load-bearing
        # identification condition for stage 1.
        _check_omega_0_connectivity(
            omega_0_mask=omega_0_effective,
            unit_codes_arr=unit_codes_arr,
            time_codes_arr=time_codes_arr,
            units_in_omega_0=units_in_omega_0,
            n_times=len(time_uniques),
            unit_uniques=unit_uniques,
        )

        # Step 11: stage 1 — fit FE on Omega_0. Wave E.1 threads Hájek-
        # normalized survey weights when survey_design was supplied.
        y_full = np.asarray(data[outcome].values, dtype=np.float64)
        unit_fe_arr, time_fe_arr, converged = _iterative_fe_subset(
            y_full,
            np.asarray(unit_codes_full),
            np.asarray(time_codes_full),
            omega_0_mask,
            weights=survey_weights,
        )
        if not converged:
            warnings.warn(
                "SpilloverDiD stage-1 iterative FE solver did not converge "
                f"within {_FE_ITER_MAX} iterations (tol={_FE_ITER_TOL}). "
                "Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        stage1_n_obs = int(omega_0_effective.sum())

        # Step 12: residualize ALL observations.
        y_tilde = _residualize_butts(
            y_full,
            np.asarray(unit_codes_full),
            np.asarray(time_codes_full),
            unit_fe_arr,
            time_fe_arr,
        )

        # Mask rank-deficient (NaN y_tilde) rows: rather than zero them out
        # (which leaves them in the sample for HC1/CR1 n/(n-k) corrections),
        # we SUBSET stage-2 arrays to the finite rows before solve_ols. This
        # ensures the SE formulas use the actual estimation sample size.
        finite_mask = np.isfinite(y_tilde)
        n_nan = int((~finite_mask).sum())
        if n_nan > 0:
            warnings.warn(
                f"SpilloverDiD: {n_nan} observation(s) excluded from stage 2 "
                "due to rank-deficient stage-1 FE estimates (unit or period "
                "absent from the untreated-and-unexposed subsample).",
                UserWarning,
                stacklevel=2,
            )

        # Wave E.3 (codex R6 P1 fix): survey_finite_mask is the effective
        # estimation mask under the survey path — it filters out BOTH
        # warn-and-dropped rows (~finite_mask, NaN y_tilde) AND zero-
        # weight subpop rows (~survey_weights > 0). Used downstream by:
        #   - the gamma_hat / Psi construction sample (so the FE drop-
        #     first basis is invariant to zero-weight subpop rows)
        #   - score_pad_mask threaded into _compute_gmm_corrected_meat
        #   - n_obs / n_treated / n_control / event_study_meta n_obs_per_col
        #     metadata (so reported counts match the actual weighted sample)
        # On the no-survey path, survey_finite_mask == finite_mask.
        if survey_weights is not None:
            survey_finite_mask = finite_mask & (survey_weights > 0)
        else:
            survey_finite_mask = finite_mask
        n_nan_or_zero = int((~survey_finite_mask).sum())

        # Wave E.3 (CI codex R1 P1 fix): the front-door D_it.sum() == 0 gate
        # at L2556 runs on the FULL DOMAIN. Under SurveyDesign.subpopulation()
        # the user can zero-out all treated rows (e.g. mask excludes every
        # ever-treated unit), and the full-domain check still passes — but
        # the effective estimating sample (survey_finite_mask) has zero
        # treated observations and tau_total is unidentified. The downstream
        # OLS solve would land on a rank-deficient stage-2 design and either
        # NaN-fail silently or surface a generic rank-deficiency warning.
        # Add an active-sample treatment-support check immediately after
        # survey_finite_mask is built so users get a clear assumption-violation
        # error on this edge case (matches the documented R svyrecvar(subset())
        # convention: domain estimation requires the domain to contain
        # identifying variation).
        if resolved_survey is not None and int(D_it[survey_finite_mask].sum()) == 0:
            raise ValueError(
                "SurveyDesign.subpopulation() (or zero-weight survey design) "
                "removes EVERY treated observation from the effective "
                "estimating sample (survey_finite_mask = finite_mask & "
                "survey_weights > 0). The Wave E.3 active-sample identification "
                "support for tau_total requires at least one treated row to "
                "remain in the weighted sample after the subpopulation filter. "
                "Either expand the subpopulation mask to include treated units "
                "or verify the survey weight column."
            )

        # Step 13: build stage-2 design.
        ring_labels = [_ring_label(list(self.rings), j) for j in range(K)]

        # Wave C: when event_study=True, compute per-row event-time clocks AND
        # build the per-event-time × ring design instead of the aggregate design.
        # ``event_study_meta`` carries the rectangular-grid metadata + binned K
        # arrays needed downstream for rectangular MultiIndex emission. None in
        # the aggregate path.
        event_study_meta: Optional[Dict[str, Any]] = None

        if self.event_study:
            K_direct_raw, K_spill_raw = _compute_event_time_per_row(
                data=data,
                unit=unit,
                row_unit=np.asarray(unit_vals),
                row_time=np.asarray(time_vals),
                effective_onsets=effective_onsets,
                coords=(
                    self.conley_coords if self.conley_coords is not None else ("__lat__", "__lon__")
                ),
                metric=self.conley_metric,
                d_bar=self._effective_d_bar,
                # PR #456 R6 perf fix: on the staggered path, reuse the
                # trigger onsets computed during the d_it cohort loop
                # instead of redoing the dense pairwise pass.
                precomputed_trigger_onset_per_row=trigger_onset_per_row_cached,
            )
            # event_study=True without conley_coords requires fallback coords for
            # ring-trigger computation. The validator already requires either
            # conley_coords or none; for now require conley_coords when
            # event_study=True (we read coords from `self.conley_coords` which
            # was validated). Defensive guard:
            if self.conley_coords is None:
                raise ValueError(
                    "event_study=True requires conley_coords to be set so the "
                    "spillover-trigger cohort onset can be computed per row. "
                    "Set conley_coords=(lat_col, lon_col) on the estimator."
                )
            # Apply horizon binning (NaN-preserving).
            K_direct_binned = _apply_horizon_binning(K_direct_raw, self.horizon_max)
            K_spill_binned = _apply_horizon_binning(K_spill_raw, self.horizon_max)
            # Reference period: mirror TwoStageDiD's convention.
            ref_period = -1 - int(self.anticipation)
            # Event-time grid:
            #   - With horizon_max: [-H, ..., +H].
            #   - With None: auto-detect from observed finite K values across
            #     BOTH clocks. The grid is the union (excluding NaN).
            if self.horizon_max is not None:
                H = int(self.horizon_max)
                event_time_grid = list(range(-H, H + 1))
            else:
                observed_k_direct = K_direct_binned[np.isfinite(K_direct_binned)]
                observed_k_spill = K_spill_binned[np.isfinite(K_spill_binned)]
                if observed_k_direct.size == 0 and observed_k_spill.size == 0:
                    raise ValueError(
                        "event_study=True but no rows have a defined K_direct "
                        "or K_spill (the panel has no ever-treated unit AND no "
                        "spillover-exposed unit). Cannot fit event-study design."
                    )
                k_union: set = set()
                if observed_k_direct.size:
                    k_union.update(int(k) for k in np.unique(observed_k_direct))
                if observed_k_spill.size:
                    k_union.update(int(k) for k in np.unique(observed_k_spill))
                # Ensure ref_period is in the grid (so the helper drops it cleanly
                # rather than emitting it as a fitted dummy when it doesn't appear
                # in the observed K set).
                k_union.add(ref_period)
                event_time_grid = sorted(k_union)
            # Build stage-2 design (all-zero columns pre-filtered with summary
            # warning; rectangular_grid retains the full (series, ring, k) tuples).
            X_2, kept_col_names, kept_col_meta, rectangular_grid, n_obs_per_col = (
                _build_event_study_design(
                    D_it=D_it,
                    ring_masks=ring_masks,
                    ring_labels=ring_labels,
                    K_direct_binned=K_direct_binned,
                    K_spill_binned=K_spill_binned,
                    event_time_grid=event_time_grid,
                    ref_period=ref_period,
                )
            )
            col_names_all = kept_col_names
            event_study_meta = {
                "kept_col_meta": kept_col_meta,
                "rectangular_grid": rectangular_grid,
                "n_obs_per_col": n_obs_per_col,
                "ref_period": ref_period,
                "K_direct_binned": K_direct_binned,
                "K_spill_binned": K_spill_binned,
                "event_time_grid": event_time_grid,
            }
        else:
            ring_covariates = np.zeros((len(data), K), dtype=np.float64)
            for j in range(K):
                ring_covariates[:, j] = (1.0 - D_it) * ring_masks[:, j].astype(np.float64)
            X_2 = np.column_stack([D_it.reshape(-1, 1), ring_covariates])
            col_names_all = ["treatment"] + [f"_spillover_{lab}" for lab in ring_labels]

        # Step 14: subset arrays to the estimation sample (finite y_tilde rows).
        # Apply to design, outcome, cluster ids, AND the Conley spatial/temporal
        # auxiliary arrays so the HC1/CR1/Conley sample-size adjustments use the
        # correct n on the NO-SURVEY path.
        #
        # Wave E.3 (this PR): under the survey path, cluster_ids stays at FULL
        # length so `_resolve_effective_cluster` / `_inject_cluster_as_psu`
        # operate on the full-domain design and the meat-helper boundary sees
        # full-length arrays (zero-pad invariant per R `survey::svyrecvar` +
        # `imputation.py:2175-2183` precedent). Under no-survey, keep the
        # historic finite_mask subset so downstream CR1 sample-size matches
        # X_2_fit.
        cluster_ids_full = (
            np.asarray(data[self.cluster].values) if self.cluster is not None else None
        )
        if n_nan > 0:
            X_2_fit = X_2[finite_mask]
            y_tilde_fit = y_tilde[finite_mask]
            if resolved_survey is not None:
                # Wave E.3: keep full-length cluster_ids for the survey path.
                cluster_ids_fit = cluster_ids_full
            else:
                cluster_ids_fit = (
                    cluster_ids_full[finite_mask] if cluster_ids_full is not None else None
                )
            time_vals_fit = np.asarray(time_vals)[finite_mask]
            unit_vals_fit = np.asarray(unit_vals)[finite_mask]
        else:
            X_2_fit = X_2
            y_tilde_fit = y_tilde
            cluster_ids_fit = cluster_ids_full
            time_vals_fit = np.asarray(time_vals)
            unit_vals_fit = np.asarray(unit_vals)

        # Wave E.3 (this PR): the resolved survey DESIGN is NOT subsetted via
        # `finite_mask`. Per R `survey::svyrecvar(subset())` convention and the
        # in-library precedents at `imputation.py:2175-2183` (PreTrendsImputation)
        # and `prep.py:1401-1432` (DCDH cell variance), zero-weight rows from
        # `SurveyDesign.subpopulation()` AND warn-and-dropped rows are kept in
        # the design at full length. The resolved survey design retains full-
        # panel length and full-design `n_psu` / `n_strata` / `df_survey` /
        # Binder centering throughout, so the meat helpers see the full-domain
        # PSU / strata geometry. The full-domain zero-pad invariant on the
        # scores themselves is delivered downstream at the
        # `_compute_gmm_corrected_meat` call site by passing
        # `score_pad_mask=survey_finite_mask` (= finite_mask AND
        # survey_weights > 0 under the survey path; see R6 P1 fix at
        # L3033-L3083 below) — the helper builds Psi on the survey-finite-
        # mask subset of inputs and zero-pads it to full panel length
        # AFTER construction but BEFORE kernel dispatch. The R6 filter is
        # critical for FE-basis invariance: `_build_butts_fe_design_csr`'s
        # `pd.factorize` compaction would otherwise include zero-weight
        # subpop rows in the first-appearance ordering and shift the
        # drop-first column (matches the canonical R svyrecvar(subset())
        # form exactly).
        #
        # `survey_weights_fit` IS finite_mask-subsetted because it is consumed
        # by the stage-2 OLS solve (`solve_ols(X_2_fit, ..., weights=
        # survey_weights_fit)`) which operates on the active sample (zero-
        # weight rows are present here but contribute W=0 to the OLS cross-
        # products, so the OLS coef is bit-equivalent to the survey-finite-
        # mask path; preserves the pre-E.3 OLS contract). The meat helper
        # receives `survey_weights_fit_gamma` (a further projection of
        # survey_weights_fit onto the survey-finite-mask frame) for the
        # gamma_hat / Psi build.
        #
        # Replaces the Wave E.1 design-subset block that mirrored the
        # `two_stage.py:567-601` pattern. TwoStageDiD parity is a deferred
        # follow-up (TODO.md).
        if n_nan > 0:
            survey_weights_fit = survey_weights[finite_mask] if survey_weights is not None else None
        else:
            survey_weights_fit = survey_weights
        resolved_survey_fit = resolved_survey
        # `survey_metadata` was computed upstream by `_resolve_survey_for_fit`
        # on the full-domain design and remains the value returned in
        # `SpilloverDiDResults`. The cluster-injection branch below recomputes
        # post-injection when `cluster=<col>` synthesizes the effective PSU.

        # Wave E.1 cluster-vs-PSU resolution (AFTER `_resolve_survey_for_fit`
        # so the warning text can reference actual PSU count). Two cases:
        #
        # 1. Both `cluster=<col>` and `survey_design.psu` provided:
        #    `_resolve_effective_cluster` warns + prefers PSU (TwoStageDiD
        #    parity — see `survey.py:1253-1275`). SpilloverDiD's
        #    `cluster=<col>` is most often a spatial / unit-level label;
        #    PSU is the design-relevant cluster.
        # 2. `cluster=<col>` provided without `survey_design.psu`:
        #    `_inject_cluster_as_psu` substitutes the cluster column for
        #    the missing PSU so the survey path becomes proper CR1 +
        #    Binder TSL (matches the documented contract for `cluster=<col>`
        #    under survey_design — see REGISTRY "Variance (Wave E.1)").
        if resolved_survey_fit is not None:
            effective_cluster_ids = _resolve_effective_cluster(
                resolved_survey_fit,
                cluster_ids_fit,
                self.cluster if self.cluster is not None else None,
            )
            if effective_cluster_ids is not None:
                # Wave E.1 R11 fix: when `cluster=<col>` becomes the effective
                # PSU (because survey_design.psu is absent), the cluster
                # column must satisfy the same panel-survey constancy
                # contract that `_validate_unit_constant_survey` enforces on
                # explicit `survey_design.psu`. Without this check, a
                # time-varying cluster column silently becomes the PSU labels
                # used for Binder TSL aggregation — producing wrong `n_psu`,
                # `df_survey`, and meat — even though the same labels passed
                # via `survey_design.psu=` would be rejected by the panel-
                # survey validator at `survey.py:1015`.
                if self.cluster is not None and resolved_survey.psu is None:
                    cluster_arr = np.asarray(effective_cluster_ids)
                    unit_arr_full = np.asarray(data[unit].values)
                    # Wave E.3: cluster_arr and the validation unit array
                    # are both full-length under the zero-pad invariant. The
                    # within-unit-constancy contract is "cluster column does
                    # not vary across periods for any unit" — validating on
                    # the full panel surfaces violations even when the row
                    # would later be warn-and-dropped (a stricter, safer
                    # contract than the prior fit-sample-only check).
                    unit_arr_for_check = unit_arr_full
                    # Validate within-unit constancy on the cluster column.
                    constancy_df = pd.DataFrame(
                        {"unit": unit_arr_for_check, "cluster": cluster_arr}
                    )
                    n_vals_per_unit = constancy_df.groupby("unit")["cluster"].nunique()
                    nonconstant = n_vals_per_unit[n_vals_per_unit > 1]
                    if len(nonconstant) > 0:
                        bad_units = list(nonconstant.index[:5])
                        raise ValueError(
                            f"`cluster='{self.cluster}'` is being used as the "
                            f"effective PSU under survey_design= (no explicit "
                            f"survey_design.psu provided), but the cluster "
                            f"column varies within unit for "
                            f"{len(nonconstant)} unit(s) "
                            f"(examples: {bad_units}). Panel-survey TSL "
                            f"requires PSU labels to be constant within unit "
                            f"across periods (matches the explicit-PSU "
                            f"contract enforced at "
                            f"`_validate_unit_constant_survey`). Either "
                            f"collapse the cluster column to be unit-constant, "
                            f"or pass an explicit unit-constant column via "
                            f"`survey_design=SurveyDesign(..., psu=<col>)`."
                        )
                resolved_survey_fit = _inject_cluster_as_psu(
                    resolved_survey_fit, effective_cluster_ids
                )
                # The Binder TSL meat reads PSU labels directly from
                # `resolved_survey_fit.psu`; the cluster_ids_fit array is
                # kept in sync so the downstream non-survey dispatch +
                # n_clusters reporting see consistent labels.
                cluster_ids_fit = resolved_survey_fit.psu
                # Recompute survey_metadata so summary() / to_dict() reflect
                # the post-injection design (df_survey / n_psu were computed
                # on the pre-injection state before `_inject_cluster_as_psu`
                # synthesized PSU from cluster=<col>). Without this,
                # cluster=<col>+survey-without-PSU fits would report
                # df_survey=0 / n_psu=0 despite the inference using the
                # injected cluster labels.
                from diff_diff.survey import compute_survey_metadata as _csm

                # Wave E.3: full-length raw weights (no finite_mask subset).
                # Matches the post-injection resolved_survey_fit length.
                raw_w_for_meta = (
                    np.asarray(data[survey_design.weights].values, dtype=np.float64)
                    if (survey_design is not None and getattr(survey_design, "weights", None))
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = _csm(resolved_survey_fit, raw_w_for_meta)

        # Wave C P1 fix (PR #456 R1): for event_study=True, recompute
        # n_obs_per_col on the POST-finite-mask sample. The original
        # n_obs_per_col from _build_event_study_design counted rows on the
        # pre-mask design — using those stale counts for `att_dynamic`,
        # `event_study_effects[k]["n_obs"]`, and the scalar `att` share
        # weights would mix two samples and change the point estimate on
        # warn-and-drop fits. The post-mask counts reflect the actual
        # stage-2 estimation sample that solve_ols sees.
        if self.event_study and event_study_meta is not None:
            # Wave E.3 (codex R8 P2 fix): on the survey path the effective
            # sample for n_obs_per_col EXCLUDES zero-weight subpop rows
            # (matches the count_mask used for n_obs / n_treated /
            # n_control below). On no-survey path this is bit-identical
            # to pre-E.3 since survey_weights_fit is None.
            if survey_weights_fit is not None:
                # Project survey_finite_mask back into the fit-sample (finite_mask) frame
                survey_finite_in_fit = (
                    survey_finite_mask[finite_mask] if n_nan > 0 else (survey_weights > 0)
                )
                X_2_fit_active = X_2_fit[survey_finite_in_fit]
                event_study_meta["n_obs_per_col"] = (
                    (X_2_fit_active != 0).sum(axis=0).astype(np.int64)
                )
            else:
                event_study_meta["n_obs_per_col"] = (X_2_fit != 0).sum(axis=0).astype(np.int64)
            # Wave E.1: when survey weights are present, also compute per-column
            # survey-weight totals for the event-study scalar `att` lincom
            # aggregation. Using raw `n_obs_per_col` shares on weighted WLS
            # horizon coefficients targets the wrong estimand; the audited
            # composition is survey-weighted-totals as the lincom weights.
            # Zero-weight rows contribute zero to the dot product so this
            # is automatically consistent with the n_obs_per_col fix above.
            if survey_weights_fit is not None:
                indicator_fit = (X_2_fit != 0).astype(np.float64)
                event_study_meta["weight_sum_per_col"] = indicator_fit.T @ survey_weights_fit
            else:
                event_study_meta["weight_sum_per_col"] = None

        # Step 15: stage-2 OLS (or WLS under Wave E.1 survey path) —
        # coef + residuals only. Wave D computes the vcov below via the
        # Gardner GMM first-stage uncertainty correction (documented
        # synthesis of Butts §3.1 + Gardner §4 + Conley 1999); Wave E.1
        # additionally composes Gerber (2026) Prop 1 Binder TSL when
        # survey_design is supplied.
        # `solve_ols` returns vcov=None when return_vcov=False.
        solve_kwargs: Dict[str, Any] = {
            "return_vcov": False,
            "rank_deficient_action": self.rank_deficient_action,
            "column_names": col_names_all,
        }
        if survey_weights_fit is not None:
            solve_kwargs["weights"] = survey_weights_fit
            solve_kwargs["weight_type"] = "pweight"
        coef, residuals, _ = solve_ols(X_2_fit, y_tilde_fit, **solve_kwargs)  # type: ignore[misc]

        # Wave D: Gardner GMM first-stage uncertainty correction.
        #
        # Reconstruct the stage-1 residual `eps_10` on the FULL sample:
        #   - On Omega_0 rows: eps_10 = y - mu_hat[i] - lambda_hat[t]
        #   - On ~Omega_0 rows: eps_10 = y (since X_10[i, :] = 0 collapses
        #     the IF product to just the stage-2 term; matches the Gardner
        #     formula at `two_stage.py:1633-1637`).
        # unit_fe_arr / time_fe_arr may have NaN at warn-and-drop units;
        # the downstream `finite_mask` subset drops those rows BEFORE the
        # GMM helper builds Psi (NaN in eps_10 is intentionally tolerated
        # at this stage — it is masked out before any matrix operation).
        alpha_full = unit_fe_arr[np.asarray(unit_codes_full)]
        beta_full = time_fe_arr[np.asarray(time_codes_full)]
        eps_10_full = np.where(omega_0_mask, y_full - alpha_full - beta_full, y_full)

        # Subset stage-1 inputs to the fit sample for the gamma_hat/Psi
        # build. The fit sample for gamma_hat construction is:
        #   - finite_mask only (no NaN y_tilde rows) on the no-survey path
        #   - finite_mask AND survey_weights > 0 on the survey path
        #
        # Wave E.3 (codex R6 P1 fix): subset the gamma_hat-construction
        # fit-sample inputs by `survey_finite_mask` (= `finite_mask &
        # (survey_weights > 0)` under the survey path; defined earlier
        # alongside `finite_mask`). This excludes zero-weight subpop
        # rows from `unit_codes_fit` / `time_codes_fit`. Once inside
        # `_build_butts_fe_design_csr` the per-call `pd.factorize`
        # compacts codes by first-appearance order — so whether a domain-
        # excluded unit sorts first or last changes which column gets
        # dropped under drop-first identification, and the resulting
        # `gamma_hat` would no longer be invariant to subpop-excluded
        # rows if we used `finite_mask` here. The cross-product
        # `X_10' W X_10` would give those rows ZERO numeric contribution
        # because W=0, but the COLUMN SPACE shifts and `gamma_hat`'s
        # coefficient indexing shifts with it, perturbing `Psi` (and
        # hence the SE) for reasons other than the documented full-
        # design Binder/FPC bookkeeping. score_pad_mask is set to
        # survey_finite_mask below so zero-weight rows are explicitly
        # zero-padded back into the meat at the full-domain bookkeeping
        # step (Wave E.3 contract — R svyrecvar(subset()) treats zero-
        # weight rows as zero-score domain padding).
        if n_nan_or_zero > 0:
            eps_10_fit = eps_10_full[survey_finite_mask]
            unit_codes_fit = np.asarray(unit_codes_full)[survey_finite_mask]
            time_codes_fit = np.asarray(time_codes_full)[survey_finite_mask]
            omega_0_mask_fit = omega_0_mask[survey_finite_mask]
        else:
            eps_10_fit = eps_10_full
            unit_codes_fit = np.asarray(unit_codes_full)
            time_codes_fit = np.asarray(time_codes_full)
            omega_0_mask_fit = omega_0_mask

        # Handle rank-deficient column drops from solve_ols (NaN coefs).
        # Subset to kept columns before building Psi; re-inflate vcov with
        # NaN at dropped positions at the end so downstream indexing
        # (vcov[0, 0] for tau_se, etc.) behaves like the pre-Wave-D path.
        kept_col_mask = np.isfinite(coef)
        n_kept = int(kept_col_mask.sum())
        if n_kept < len(coef):
            X_2_kept = X_2_fit[:, kept_col_mask]
            coef_kept = coef[kept_col_mask]
        else:
            X_2_kept = X_2_fit
            coef_kept = coef
        eps_2_fit = y_tilde_fit - X_2_kept @ coef_kept

        # Wave E.3 (codex R6 P1 fix): subset the gamma_hat-construction
        # arrays from finite_mask length down to survey_finite_mask length
        # too. This excludes zero-weight subpop rows (which have W=0 so
        # they contribute zero to the cross-products numerically, but
        # without the explicit subset they would change the drop-first
        # FE basis via `_build_butts_fe_design_csr`'s `pd.factorize`
        # compaction).
        if survey_weights is not None and n_nan_or_zero > n_nan:
            # Project survey_finite_mask into the fit-sample (finite_mask)
            # frame: True for fit-sample rows that ALSO have weight > 0.
            survey_finite_in_fit = survey_finite_mask[finite_mask]
            X_2_kept_gamma = X_2_kept[survey_finite_in_fit]
            eps_2_fit_gamma = eps_2_fit[survey_finite_in_fit]
            survey_weights_fit_gamma = survey_weights_fit[survey_finite_in_fit]
        else:
            X_2_kept_gamma = X_2_kept
            eps_2_fit_gamma = eps_2_fit
            survey_weights_fit_gamma = survey_weights_fit

        # Build stage-1 FE designs on the fit sample. Column space:
        # [unit_1, ..., unit_{U-1}, time_1, ..., time_{T-1}] (drop-first
        # identification, matches `TwoStageDiD._build_fe_design`).
        #
        # Wave E.3 (this PR): the stage-1 FE design + gamma_hat solve + Psi
        # construction stays on the FIT SAMPLE (post-finite_mask) to keep
        # the drop-first identification stable. `_build_butts_fe_design_csr`
        # re-factorizes inputs via `pd.factorize` and drops the first unit
        # / time code; if the dropped unit sorts first, the fit-length and
        # full-length builds produce DIFFERENT column spaces (an all-zero
        # X_10 column for the dropped unit in the full-length build →
        # rank-deficient `X_10' W X_10` → lstsq fallback → different
        # `gamma_hat`). The zero-pad invariant is preserved by zero-padding
        # the constructed Psi inside `_compute_gmm_corrected_meat` AFTER
        # the fit-sample gamma_hat / Psi build, NOT by rebuilding the FE
        # design at full length. Mirrors the canonical R
        # `survey::svyrecvar(subset())` / `imputation.py:2175-2183` pattern
        # exactly (construct scores on the active sample first; zero-pad to
        # full design at the variance step).
        X_1_sparse_fit, X_10_sparse_fit = _build_butts_fe_design_csr(
            unit_codes_fit,
            time_codes_fit,
            omega_0_mask_fit,
        )

        # Conley spatial kwargs only when vcov_type == "conley".
        if self.vcov_type == "conley":
            coord_array_full = np.asarray(data[list(self.conley_coords)].values, dtype=np.float64)
            coord_array_fit = coord_array_full[finite_mask] if n_nan > 0 else coord_array_full
            _conley_coords_arg = coord_array_fit
            _conley_cutoff_arg = self.conley_cutoff_km
            _conley_metric_arg = self.conley_metric
            _conley_time_arg = time_vals_fit
            _conley_unit_arg = unit_vals_fit
            _conley_lag_arg = self.conley_lag_cutoff
        else:
            coord_array_full = None
            _conley_coords_arg = None
            _conley_cutoff_arg = None
            _conley_metric_arg = None
            _conley_time_arg = None
            _conley_unit_arg = None
            _conley_lag_arg = None

        # Wave E.2 follow-up gate (post-resolution, post-injection):
        # fail-closed for `vcov_type="conley" + conley_lag_cutoff > 0` when
        # the EFFECTIVE PSU is still absent after `_inject_cluster_as_psu`.
        # Under no-effective-PSU survey designs (weights-only / strata-only
        # WITHOUT a cluster fallback) the orchestrator falls back to
        # pseudo-PSU = obs-index in `_compute_stratified_conley_meat`, but
        # each pseudo-PSU appears in exactly one period, so the per-PSU
        # serial cross-period loop never contributes anything (silent zero
        # serial term). Routing the serial loop to `conley_unit` (the panel
        # unit) instead of pseudo-PSU would mix IF allocators (PSU spatial
        # vs unit serial), which violates the single-IF-allocator design
        # pinned by the user-confirmed methodology in the Wave E.2 follow-up
        # plan. Fail-closed per `feedback_no_silent_failures` until a
        # no-effective-PSU-specific derivation is queued. Note: this fires
        # AFTER `_inject_cluster_as_psu` (which runs upstream) so the
        # documented `cluster=<col> + survey_design(without psu)` surface
        # — which becomes an effective-PSU layout via injection — passes
        # through unscathed. R2 P1 fix: original front-door gate at
        # `spillover.py:2210-2242` (now removed) fired before injection
        # and broke the cluster-as-PSU survey-Conley surface.
        if (
            resolved_survey_fit is not None
            and resolved_survey_fit.psu is None
            and self.vcov_type == "conley"
            and self.conley_lag_cutoff is not None
            and self.conley_lag_cutoff > 0
        ):
            raise NotImplementedError(
                "SpilloverDiD(vcov_type='conley', conley_lag_cutoff > 0) "
                "combined with a no-effective-PSU survey_design "
                "(weights-only / strata-only WITHOUT a cluster fallback) "
                "is not supported in Wave E.2 follow-up. Under no-effective-"
                "PSU survey designs the panel-block serial Bartlett HAC "
                "would silently contribute zero (each pseudo-PSU = "
                "obs-index appears in exactly one period, so the within-PSU "
                "temporal sum has no cross-period pairs to accumulate). "
                "Routing the serial loop to `conley_unit` would mix IF "
                "allocators with the spatial term and is not derived in "
                "this PR. Supply either an explicit `survey_design.psu`, "
                "or `cluster=<col>` (which is injected as the effective "
                "PSU per Wave E.1's `_inject_cluster_as_psu` routing), "
                "or use `conley_lag_cutoff=0` (cross-sectional Wave E.2)."
            )

        # Derive the Wave D variance mode from the PUBLIC contract:
        #   - vcov_type="conley"          → "conley" (Conley spatial-HAC + GMM)
        #   - cluster=<col> supplied      → "cluster" (CR1 + GMM)
        #   - vcov_type="hc1" (default)   → "hc1"
        # `self.vcov_type` can be "hc1" / "classical" / "conley"; the public
        # `cluster=<col>` kwarg ORTHOGONALLY selects CR1. Pre-Wave-D the
        # routing happened inside solve_ols; Wave D bypasses that path, so
        # the dispatch must be reconstructed here. (Round 1 codex P0 fix:
        # without this derivation, a user-supplied `cluster=<col>` was
        # silently ignored on the default hc1 path, yielding HC1 SEs when
        # CR1 was requested.)
        #
        # Wave E.1 amendment: when `resolved_survey_fit.psu` is set,
        # `cluster_ids_fit` was overwritten with the PSU labels above
        # (TwoStageDiD warn-and-use-PSU pattern). The PSU IS the cluster,
        # so the dispatch naturally lands on "cluster" — which the meat
        # helper then routes into the Binder TSL branch because
        # `resolved_survey_fit is not None`.
        if self.vcov_type == "conley":
            _wave_d_vcov_mode: "Literal['hc1', 'conley', 'cluster']" = "conley"
        elif cluster_ids_fit is not None:
            _wave_d_vcov_mode = "cluster"
        else:
            _wave_d_vcov_mode = "hc1"

        # Wave E.3 (this PR — revised post codex R2 P1 + R6 P1): on the
        # survey path, the gamma_hat / Psi construction runs on
        # SURVEY-FINITE-MASK length (finite_mask AND survey_weights > 0)
        # so the drop-first FE column space + stage-1 sparse factorization
        # is INVARIANT to zero-weight subpop rows (codex R6 P1 fix). The
        # full-domain zero-pad invariant is delivered by:
        #   (1) passing the kernel-dispatch arrays (cluster_ids, conley_*,
        #       resolved_survey) at FULL LENGTH so the meat helpers
        #       (Binder TSL / stratified-Conley / serial Bartlett) see the
        #       full-domain PSU / strata / centroid / time geometry, and
        #   (2) threading `score_pad_mask=survey_finite_mask` so
        #       `_compute_gmm_corrected_meat` zero-pads the
        #       survey-finite-mask Psi to full panel length AFTER
        #       construction but BEFORE kernel dispatch.
        # Zero-weight rows (subpop-excluded) are zero-padded back at the
        # meat boundary alongside warn-and-dropped rows — both are
        # "domain padding" per R `survey::svyrecvar(subset())` semantics.
        # This matches the canonical R svyrecvar(subset()) and
        # `imputation.py:2175-2183` pattern exactly — Psi computed on the
        # active sample, zero-padded for the variance step, full design
        # retained for bookkeeping.
        if resolved_survey_fit is not None:
            # Kernel-dispatch arrays at FULL length under the survey path.
            cluster_ids_for_meat = cluster_ids_fit  # full-length under Wave E.3
            if self.vcov_type == "conley":
                conley_coords_for_meat = coord_array_full  # full-length, never subsetted
                conley_time_for_meat = np.asarray(time_vals)  # full panel
                conley_unit_for_meat = np.asarray(unit_vals)  # full panel
            else:
                conley_coords_for_meat = None
                conley_time_for_meat = None
                conley_unit_for_meat = None
            score_pad_mask_arg: Optional[np.ndarray] = survey_finite_mask
        else:
            # No-survey path: bit-identical to pre-E.3 (no zero-padding).
            cluster_ids_for_meat = cluster_ids_fit
            conley_coords_for_meat = _conley_coords_arg
            conley_time_for_meat = _conley_time_arg
            conley_unit_for_meat = _conley_unit_arg
            score_pad_mask_arg = None

        # Compute the GMM-corrected meat (Psi' K Psi). Caller-side bread
        # sandwich below mirrors `TwoStageDiD._compute_gmm_variance`
        # at `two_stage.py:1763-1791`. Wave E.1 passes survey_weights +
        # resolved_survey kwargs; the helper routes to Binder TSL meat
        # when both are non-None (hc1 / cluster modes). Wave E.3 adds
        # `score_pad_mask` on the survey path so Psi is zero-padded inside
        # the helper after construction (the gamma_hat / Psi build runs on
        # the `X_2_kept_gamma` / `eps_2_fit_gamma` / `survey_weights_fit_gamma`
        # arrays — survey-finite-mask subset of fit-sample inputs — plus
        # `X_*_sparse_fit` / `eps_10_fit` which are already built on
        # survey_finite_mask above).
        meat_kept = _compute_gmm_corrected_meat(
            X_1_sparse=X_1_sparse_fit,
            X_10_sparse=X_10_sparse_fit,
            eps_10=eps_10_fit,
            X_2=X_2_kept_gamma,
            eps_2=eps_2_fit_gamma,
            vcov_type=_wave_d_vcov_mode,
            cluster_ids=cluster_ids_for_meat,
            conley_coords=conley_coords_for_meat,
            conley_cutoff_km=_conley_cutoff_arg,
            conley_metric=_conley_metric_arg,
            conley_kernel="bartlett",
            conley_time=conley_time_for_meat,
            conley_unit=conley_unit_for_meat,
            conley_lag_cutoff=_conley_lag_arg,
            survey_weights=survey_weights_fit_gamma,
            resolved_survey=resolved_survey_fit,
            score_pad_mask=score_pad_mask_arg,
        )

        # Bread sandwich: A_22^{-1} = (X_2' W X_2)^{-1} via the shared rank-guarded
        # generalized inverse `_rank_guarded_inv` (column-drop on a near-singular
        # Gram + UserWarning; dropped coordinates are NaN'd in the vcov below).
        # Wave E.1 adds the W diagonal under the survey path so the bread aligns
        # with the WLS gamma / weighted Psi construction in the meat helper.
        if survey_weights_fit is not None:
            A_22_kept = X_2_kept.T @ (X_2_kept * survey_weights_fit[:, None])
        else:
            A_22_kept = X_2_kept.T @ X_2_kept
        # np.linalg.solve only raises on an *exactly* singular Gram; a *near*-
        # singular A_22 would otherwise flow a garbage inverse (~1e13) into the
        # SE. `_rank_guarded_inv` truncates redundant directions on the
        # equilibrated Gram -> finite SE on the identified subspace (NaN at
        # rank 0), matching the covariate IF rank-guard. A_22_kept is already
        # column-dropped upstream; this is the within-kept near-singular guard
        # (its rank-0 all-NaN return composes with the (k,k) re-inflation below).
        bread_kept, n_dropped, _, dropped = _rank_guarded_inv(A_22_kept, return_dropped=True)
        if n_dropped:
            warnings.warn(
                "SpilloverDiD Wave D bread: A_22 = X_2' W X_2 is rank-deficient; "
                "rank-reducing to a finite SE on the identified subspace "
                f"({n_dropped} redundant direction(s) dropped, NaN if rank 0).",
                UserWarning,
                stacklevel=2,
            )
        vcov_kept = bread_kept @ meat_kept @ bread_kept
        # A within-kept dropped (unidentified) coefficient is zero-filled in
        # bread_kept, which would report se=0; NaN its row/col so per-coef SE is
        # NaN, not 0. These ride along the (k, k) re-inflation below.
        if dropped.any():
            vcov_kept[dropped, :] = np.nan
            vcov_kept[:, dropped] = np.nan

        # Re-inflate to (k, k) with NaN at rank-deficient column positions
        # so downstream code (which indexes vcov[i, i] for per-coef SE) sees
        # NaN for dropped columns — matches the pre-Wave-D solve_ols
        # behavior at `linalg.py` (rank-deficient drops produce NaN coefs +
        # NaN vcov entries).
        if n_kept < len(coef):
            vcov = np.full((len(coef), len(coef)), np.nan)
            kept_idx = np.flatnonzero(kept_col_mask)
            vcov[np.ix_(kept_idx, kept_idx)] = vcov_kept
        else:
            vcov = vcov_kept

        # Step 16a: shared df_for_inference computation.
        #
        # Wave D (non-survey): df = n_obs - effective_rank (OLS residual df).
        # Wave E.1 (survey): df = resolved_survey_fit.df_survey, which
        # encodes the standard survey-statistics DOF
        # (PSU + strata → n_PSU - n_strata; PSU only → n_PSU - 1;
        #  strata only → n_obs - n_strata; neither → n_obs - 1; see
        #  `ResolvedSurveyDesign.df_survey` at survey.py:619-627). The
        # Binder TSL meat is design-consistent; the OLS residual df is
        # no longer the right t-distribution DOF.
        # Wave E.3 (codex R8 P2 fix): under the survey path, the effective
        # estimation sample EXCLUDES zero-weight subpop rows because they
        # are filtered out of the gamma_hat / Psi construction sample by
        # the survey_finite_mask above. Report n_obs / n_treated / n_control
        # / df_resid on that tighter sample so the metadata matches the
        # actual weighted sample seen by the variance computation. On the
        # no-survey path survey_finite_mask == finite_mask (bit-identical
        # to pre-E.3).
        count_mask = survey_finite_mask if resolved_survey_fit is not None else finite_mask
        n_obs_eff = int(count_mask.sum())
        k_effective = int(np.isfinite(coef).sum())
        df_resid = n_obs_eff - k_effective
        if resolved_survey_fit is not None:
            df_survey_val = resolved_survey_fit.df_survey
            df_for_inference: int = int(df_survey_val) if df_survey_val is not None else 0
        else:
            df_for_inference = df_resid
        if df_for_inference <= 0:
            # Saturated. Either OLS-saturated (n - k <= 0) or
            # survey-saturated (df_survey = 0; lonely_psu='remove' may
            # have removed all strata). Force NaN inference by setting
            # df = 0 (safe_inference treats df = 0 as no usable degrees
            # of freedom). Distinct from df = None which would fall
            # through to a normal-distribution approximation —
            # misleading on a degenerate sample.
            survey_note = (
                " (survey-saturated: df_survey = "
                f"{int(df_survey_val) if df_survey_val is not None else 0}; "
                "lonely_psu='remove' may have removed all strata)"
                if resolved_survey_fit is not None
                else ""
            )
            warnings.warn(
                f"SpilloverDiD inference df = {df_for_inference} (n_obs="
                f"{n_obs_eff}, effective_rank={k_effective}{survey_note}). "
                "Inference (t-stat, p-value, CI) will be NaN.",
                UserWarning,
                stacklevel=2,
            )
            df_for_inference = 0

        # Step 16b: branch on event_study mode for result extraction.
        att_dynamic_df: Optional[pd.DataFrame] = None
        event_study_effects_dict: Optional[Dict[int, Dict[str, Any]]] = None
        reference_period_used: Optional[int] = None

        if self.event_study:
            assert event_study_meta is not None  # set in Step 13 above
            (
                tau_total,
                tau_se,
                tau_t,
                tau_p,
                tau_ci,
                spillover_df,
                att_dynamic_df,
                event_study_effects_dict,
                coefficients_full,
            ) = _extract_event_study_results(
                coef=coef,
                vcov=vcov,
                col_names_all=col_names_all,
                kept_col_meta=event_study_meta["kept_col_meta"],
                rectangular_grid=event_study_meta["rectangular_grid"],
                n_obs_per_col=event_study_meta["n_obs_per_col"],
                ref_period=event_study_meta["ref_period"],
                df_resid=df_for_inference,
                alpha=self.alpha,
                ring_labels=ring_labels,
                weight_sum_per_col=event_study_meta.get("weight_sum_per_col"),
            )
            reference_period_used = event_study_meta["ref_period"]
        else:
            # Wave B aggregate path: extract treatment coef + per-ring inference.
            tau_total = float(coef[0])
            # Clamp negative diagonals to 0 before sqrt: indefinite Conley or
            # near-singular sandwich variances can produce numerically tiny
            # negative values that would otherwise NaN the entire inference
            # row. Matches the sibling-estimator convention
            # (two_stage.py:1183, estimators.py:606, stacked_did.py:515).
            tau_se = (
                float(np.sqrt(max(vcov[0, 0], 0.0)))
                if vcov is not None and np.isfinite(vcov[0, 0])
                else float("nan")
            )
            tau_t, tau_p, tau_ci = safe_inference(
                tau_total, tau_se, alpha=self.alpha, df=df_for_inference
            )

            # Per-ring inference.
            ring_rows = []
            for j in range(K):
                idx = 1 + j  # 0 is treatment; rings follow.
                coef_j = float(coef[idx])
                se_j = (
                    float(np.sqrt(max(vcov[idx, idx], 0.0)))
                    if vcov is not None and np.isfinite(vcov[idx, idx])
                    else float("nan")
                )
                t_j, p_j, ci_j = safe_inference(coef_j, se_j, alpha=self.alpha, df=df_for_inference)
                ring_rows.append(
                    {
                        "ring": ring_labels[j],
                        "coef": coef_j,
                        "se": se_j,
                        "t_stat": t_j,
                        "p_value": p_j,
                        "ci_low": ci_j[0],
                        "ci_high": ci_j[1],
                    }
                )
            spillover_df = pd.DataFrame(ring_rows).set_index("ring") if ring_rows else None

            # Coefficients dict — Wave B name → value layout. "ATT" alias points
            # at the treatment slot (sibling-estimator convention).
            coefficients_full = {}
            for i, name in enumerate(col_names_all):
                val = float(coef[i]) if np.isfinite(coef[i]) else float("nan")
                coefficients_full[name] = val
            coefficients_full["ATT"] = tau_total

        # Step 16c: counts for the result class.
        n_units_ever_in_ring: Dict[str, int] = {}
        for j in range(K):
            in_ring_units = data.loc[ring_masks[:, j], unit].nunique()
            n_units_ever_in_ring[ring_labels[j]] = int(in_ring_units)

        # Step 17: assemble SpilloverDiDResults. n_obs / n_treated / n_control
        # reflect the actual stage-2 estimation sample (after dropping NaN
        # y_tilde rows AND, on the survey path, zero-weight subpop rows that
        # were filtered from the gamma_hat / Psi construction per Wave E.3
        # R6 P1 fix), matching solve_ols's HC1/CR1 sample-size adjustments
        # AND the meat-helper's effective sample.
        D_it_fit = D_it[count_mask] if int((~count_mask).sum()) > 0 else D_it
        # Wave E.3 (codex R11 P2 fix): recompute n_far_away_obs on the
        # effective estimation sample so it doesn't count zero-weight far-
        # away controls from `SurveyDesign.subpopulation()`. The original
        # `n_far_away_obs` (computed at L2579 on the full domain) is used
        # to validate that at least one far-away identifying row exists
        # — that gate already fired upstream. Under Wave E.3 the reported
        # count should reflect the active weighted sample, matching the
        # Wave E.3 contract for n_obs / n_treated / n_control / event-
        # study n_obs_per_col above.
        n_far_away_obs_reported = int(
            (is_control_row_now & (d_it_per_row > self._effective_d_bar) & count_mask).sum()
        )

        result = SpilloverDiDResults(
            att=tau_total,
            se=tau_se,
            t_stat=tau_t,
            p_value=tau_p,
            conf_int=tau_ci,
            n_obs=n_obs_eff,
            n_treated=int(D_it_fit.sum()),
            n_control=int(len(D_it_fit) - D_it_fit.sum()),
            alpha=self.alpha,
            coefficients=coefficients_full,
            vcov=vcov,
            residuals=residuals,
            r_squared=None,
            inference_method="analytical",
            n_bootstrap=None,
            n_clusters=(
                int(len(np.unique(cluster_ids_fit))) if cluster_ids_fit is not None else None
            ),
            vcov_type=self.vcov_type,
            # Wave E.1: when survey.psu wins the warn-and-use-PSU override
            # (`_resolve_effective_cluster`), the EFFECTIVE clustering label
            # is `survey_design.psu`, not `self.cluster`. Report that so
            # `DiDResults.to_dict()` machine-readable metadata stays
            # consistent with the variance numbers.
            cluster_name=(
                survey_design.psu
                if (
                    survey_design is not None
                    and getattr(survey_design, "psu", None)
                    and resolved_survey_fit is not None
                    and resolved_survey_fit.psu is not None
                )
                else self.cluster
            ),
            conley_lag_cutoff=(self.conley_lag_cutoff if self.vcov_type == "conley" else None),
            spillover_effects=spillover_df,
            ring_breakpoints=list(self.rings),
            d_bar=self._effective_d_bar,
            n_units_ever_in_ring=n_units_ever_in_ring,
            n_far_away_obs=n_far_away_obs_reported,
            is_staggered=is_staggered,
            event_study=self.event_study,
            stage1_n_obs=stage1_n_obs,
            anticipation=self.anticipation,
            att_dynamic=att_dynamic_df,
            event_study_effects=event_study_effects_dict,
            horizon_max=self.horizon_max if self.event_study else None,
            reference_period=reference_period_used,
            # Wave E.1 survey-design metadata. Populated only when
            # survey_design was supplied (otherwise all None for
            # backward-compat with the Wave B/C/D no-survey contract).
            #
            # `n_psu` follows the implicit-PSU convention from
            # `ResolvedSurveyDesign.df_survey`: when `psu is None` after
            # all injection steps (no `cluster=<col>` and no
            # `survey_design.psu`), each observation is its own singleton
            # PSU and the reported count is `n_obs`.
            #
            # Wave E.3: under the zero-pad invariant the implicit-PSU
            # count reflects the FULL domain (length of the resolved
            # survey design's weights array), NOT the post-`finite_mask`
            # fit sample. This keeps top-level `n_psu` consistent with
            # `survey_metadata.n_psu` / `survey_metadata.df_survey` —
            # which both reflect the full domain under Wave E.3 —
            # avoiding the cross-surface inconsistency that previously
            # surfaced on weights-only / strata-only survey fits with
            # warn-and-drop (top-level n_psu would track the fit sample
            # while df_survey tracked the full domain).
            survey_metadata=survey_metadata,
            n_psu=(
                (
                    resolved_survey_fit.n_psu
                    if resolved_survey_fit.psu is not None
                    else len(resolved_survey_fit.weights)
                )
                if resolved_survey_fit is not None
                else None
            ),
            n_strata=resolved_survey_fit.n_strata if resolved_survey_fit is not None else None,
        )
        self.results_ = result
        self.is_fitted_ = True
        return result
