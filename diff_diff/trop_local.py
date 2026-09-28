"""
Local (observation-specific) estimation method for the TROP estimator.

Contains the TROPLocalMixin class with all methods for the local
estimation pathway, including preprocessing, distance computation,
per-observation weight computation, model fitting, LOOCV scoring,
and bootstrap variance estimation.

This module is used via mixin inheritance — see trop.py for the
main TROP class definition.
"""

import logging
import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_bootstrap_trop_variance,
    _rust_unit_distance_matrix,
)
from diff_diff.bootstrap_utils import (
    stratified_bootstrap_indices,
    warn_bootstrap_failure_rate,
)
from diff_diff.trop_results import _PrecomputedStructures
from diff_diff.utils import warn_if_not_converged


def _treated_cell_is_estimable(
    control_mask: np.ndarray,
    Y: np.ndarray,
    weight_matrix: np.ndarray,
    i: int,
    t: int,
) -> bool:
    """True iff treated cell (i, t)'s counterfactual is identified by the control fit.

    The working model fits unregularized unit and time fixed effects
    ``alpha_j`` / ``beta_s`` on the weighted observed control cells, then sets
    ``tau_it = Y_it - alpha_i - beta_t - L_it``. For that difference to be a valid
    counterfactual rather than a fixed-effect-contaminated raw outcome, the sum
    ``alpha_i + beta_t`` must be identified by the two-way-FE control fit.

    In a two-way fixed-effect model the effects are pinned only **within each
    connected component** of the bipartite graph whose nodes are units and
    periods and whose edges are the positively-weighted observed control cells
    (``usable = (D==0) & finite(Y) & weight>0``); across components there is a
    free per-component offset. Hence ``alpha_i + beta_t`` is identified iff the
    **target unit node and target period node lie in the same component**.

    A marginal "the target unit has some usable control AND the target period has
    some usable control" test is necessary but NOT sufficient: e.g. usable cells
    at ``(unitA, t0)`` and ``(unitB, t1)`` with target ``(unitA, t1)`` pass it,
    yet ``alpha_A + beta_1`` spans two disconnected components and is unidentified.
    This connected-component check subsumes the simpler degeneracies it replaced:
    an always-treated unit (empty unit column) or a fully-treated period (empty
    period row) leaves the corresponding node isolated, hence non-estimable.

    This is a **general correctness guard applied to every local fit** (absorbing
    and non-absorbing): it NaNs exactly the cells whose ``alpha_i + beta_t`` is
    unidentified. On balanced panels (and absorbing panels with an observed
    never-treated unit, which connects every period to every unit) the whole
    control graph is one component, so the predicate is a no-op (no behavior
    change). Shared by the final point fit and the bootstrap fixed-lambda refit.

    Cost: a bipartite BFS bounded by the usable-cell count, run per treated cell
    only when both the unit column and period row are non-empty (the cheap
    fast-path rejects the common degeneracies first). non_absorbing is opt-in and
    correctness-first, so the extra work is acceptable.
    """
    usable = control_mask & np.isfinite(Y) & (weight_matrix > 0)
    # Fast path: an empty target column (alpha_i) or row (beta_t) is isolated.
    if not bool(np.any(usable[:, i])) or not bool(np.any(usable[t, :])):
        return False
    # Bipartite reachability from period-node t; estimable iff unit-node i reached.
    reached_periods = np.zeros(usable.shape[0], dtype=bool)
    reached_units = np.zeros(usable.shape[1], dtype=bool)
    reached_periods[t] = True
    while True:
        # Units adjacent to any reached period, then periods adjacent to any
        # reached unit; iterate the bipartite expansion to a fixpoint.
        new_units = reached_units | np.any(usable[reached_periods, :], axis=0)
        if np.any(new_units):
            new_periods = reached_periods | np.any(usable[:, new_units], axis=1)
        else:
            new_periods = reached_periods
        if np.array_equal(new_units, reached_units) and np.array_equal(
            new_periods, reached_periods
        ):
            break
        reached_units, reached_periods = new_units, new_periods
    return bool(reached_units[i])


def _validate_and_pivot_treatment(data, time, unit, treatment, all_periods, all_units):
    """Validate treatment column and create D matrix with missing mask.

    Rejects observed rows with missing treatment values (data quality error),
    then pivots to (time x unit) matrix. Structural gaps from unbalanced panels
    are filled with 0 (assumed untreated) and flagged with a warning.

    Returns
    -------
    D : ndarray
        Treatment matrix (n_periods x n_units), int.
    missing_mask : ndarray
        Boolean mask of structurally absent cells (n_periods x n_units).
    """
    n_nan_observed = int(data[treatment].isna().sum())
    if n_nan_observed > 0:
        raise ValueError(
            f"{n_nan_observed} observation(s) have missing treatment values. "
            f"TROP requires non-missing treatment indicators for all observed "
            f"rows. Remove or impute missing values before fitting."
        )

    D_raw = data.pivot(index=time, columns=unit, values=treatment).reindex(
        index=all_periods, columns=all_units
    )
    missing_mask = pd.isna(D_raw).values
    n_missing_structural = int(missing_mask.sum())
    if n_missing_structural > 0:
        warnings.warn(
            f"{n_missing_structural} missing treatment indicator(s) in the "
            f"(time x unit) panel matrix filled with 0 (assumed "
            f"untreated). This typically occurs in unbalanced panels.",
            UserWarning,
            stacklevel=3,
        )
    D = D_raw.fillna(0).astype(int).values
    return D, missing_mask


def _setup_trop_data(
    data,
    outcome,
    treatment,
    unit,
    time,
    resolved_survey,
    survey_design,
    non_absorbing: bool = False,
):
    """Shared data setup for TROP local and global fit paths.

    Performs panel pivoting (long → wide), absorbing-state validation,
    treated/control unit identification, first-treatment-period detection,
    and pre/post period counting. Returns a dict so both callers can
    unpack only the fields they need.

    When ``non_absorbing`` is False (default) the treatment indicator must be an
    absorbing state (monotonic non-decreasing per unit) and a non-monotonic
    indicator raises ``ValueError``; there must be at least one never-treated
    unit and at least 2 leading pre-treatment periods. When ``non_absorbing`` is
    True these absorbing-specific guards are relaxed to support general (on/off)
    assignment patterns (Athey et al. 2025 Eq. 12 / Algorithm 2): the
    monotonicity check is skipped, identification falls back to untreated *cells*
    (rather than requiring whole never-treated units), and the pre-period guard
    becomes a weaker "at least 2 periods contain untreated cells" check. The
    global fit path always calls with ``non_absorbing=False`` (it additionally
    requires simultaneous block adoption).

    The global-method-specific staggered-adoption check stays in
    `_fit_global` as a post-helper validation because it depends on
    estimator semantics (global method requires simultaneous treatment),
    not data preparation.

    Note: the same dict-shaped contract is also relied on by the bootstrap
    paths (`_bootstrap_variance` / `_bootstrap_variance_global`) which
    currently rebuild similar state per draw. Future contract changes to
    this helper must be checked against both `_fit_*` and `_bootstrap_*`
    call sites for cross-file sync.

    Returns
    -------
    dict
        With keys:
        ``all_units, all_periods, n_units, n_periods, unit_to_idx,
        period_to_idx, idx_to_unit, idx_to_period, unit_weight_arr,
        Y, D, missing_mask, treated_mask, n_treated_obs,
        treated_unit_idx, control_unit_idx, first_treat_period,
        n_pre_periods, n_post_periods``.
    """
    all_units = sorted(data[unit].unique())
    all_periods = sorted(data[time].unique())

    if resolved_survey is not None:
        from diff_diff.survey import _extract_unit_survey_weights

        unit_weight_arr = _extract_unit_survey_weights(data, unit, survey_design, all_units)
    else:
        unit_weight_arr = None

    n_units = len(all_units)
    n_periods = len(all_periods)

    unit_to_idx = {u: i for i, u in enumerate(all_units)}
    period_to_idx = {p: i for i, p in enumerate(all_periods)}
    idx_to_unit = {i: u for u, i in unit_to_idx.items()}
    idx_to_period = {i: p for p, i in period_to_idx.items()}

    Y = (
        data.pivot(index=time, columns=unit, values=outcome)
        .reindex(index=all_periods, columns=all_units)
        .values
    )

    D, missing_mask = _validate_and_pivot_treatment(
        data, time, unit, treatment, all_periods, all_units
    )

    # Absorbing-state (monotonic non-decreasing) validation. Skipped when the
    # caller opts into general (on/off) assignment via non_absorbing=True.
    if not non_absorbing:
        violating_units = []
        for unit_idx in range(n_units):
            observed_mask = ~missing_mask[:, unit_idx]
            observed_d = D[observed_mask, unit_idx]
            if len(observed_d) > 1 and np.any(np.diff(observed_d) < 0):
                violating_units.append(all_units[unit_idx])

        if violating_units:
            raise ValueError(
                f"Treatment indicator is not an absorbing state for units: {violating_units}. "
                f"D[t, unit] must be monotonic non-decreasing (once treated, always treated). "
                f"If this is event-study style data with absorbing treatment, convert to "
                f"absorbing state: D[t, i] = 1 for all t >= first treatment period. "
                f"If treatment genuinely turns on and off (non-absorbing), pass "
                f"non_absorbing=True (method='local' only; assumes no dynamic effects)."
            )

    treated_mask = D == 1
    n_treated_obs = int(np.sum(treated_mask))

    if n_treated_obs == 0:
        raise ValueError("No treated observations found")

    unit_ever_treated = np.any(D == 1, axis=0)
    treated_unit_idx = np.where(unit_ever_treated)[0]
    control_unit_idx = np.where(~unit_ever_treated)[0]

    # Observed untreated cells. Structural panel gaps are filled with D=0
    # (_validate_and_pivot_treatment), so identification checks under
    # non_absorbing must exclude those filled cells (and non-finite outcomes):
    # only an OBSERVED D=0 cell can serve as a control for the (1-W)
    # counterfactual fit. A raw `D == 0` count would let an all-observed-treated
    # unbalanced panel pass with no real control outcomes.
    valid_control_mask = (D == 0) & (~missing_mask) & np.isfinite(Y)

    if non_absorbing:
        # General assignment identifies off untreated *cells* (the per-(i,t)
        # estimator masks treated cells via (1-W) and fits the rest), so a fully
        # toggling panel with no never-treated unit is still identified. Require
        # at least one observed untreated cell.
        if not np.any(valid_control_mask):
            raise ValueError(
                "No observed untreated (control) observations found; non_absorbing "
                "TROP needs observed cells with D=0 (not structural panel gaps) to "
                "impute the counterfactual."
            )
    else:
        if len(control_unit_idx) == 0:
            raise ValueError("No control units found")

    first_treat_period = None
    for t in range(n_periods):
        if np.any(D[t, :] == 1):
            first_treat_period = t
            break

    if first_treat_period is None:
        raise ValueError("Could not infer post-treatment periods from D matrix")

    n_pre_periods = first_treat_period
    n_post_periods = int(np.sum(np.any(D[first_treat_period:, :] == 1, axis=1)))

    if non_absorbing:
        # "Leading all-control block" is ill-defined when treatment toggles, so
        # the absorbing n_pre_periods>=2 guard does not apply. Require instead
        # that at least 2 periods contain an OBSERVED untreated cell (a weak
        # factor-model identifiability floor); finer donor-pool degeneracy is
        # handled downstream by the LOOCV empty-control (Q=inf) and inf-distance
        # guards.
        n_periods_with_controls = int(np.sum(np.any(valid_control_mask, axis=1)))
        if n_periods_with_controls < 2:
            raise ValueError(
                "Need at least 2 periods containing observed untreated "
                "observations for non_absorbing TROP."
            )
    elif n_pre_periods < 2:
        raise ValueError("Need at least 2 pre-treatment periods")

    return {
        "all_units": all_units,
        "all_periods": all_periods,
        "n_units": n_units,
        "n_periods": n_periods,
        "unit_to_idx": unit_to_idx,
        "period_to_idx": period_to_idx,
        "idx_to_unit": idx_to_unit,
        "idx_to_period": idx_to_period,
        "unit_weight_arr": unit_weight_arr,
        "Y": Y,
        "D": D,
        "missing_mask": missing_mask,
        "treated_mask": treated_mask,
        "n_treated_obs": n_treated_obs,
        "treated_unit_idx": treated_unit_idx,
        "control_unit_idx": control_unit_idx,
        "first_treat_period": first_treat_period,
        "n_pre_periods": n_pre_periods,
        "n_post_periods": n_post_periods,
    }


def _run_trop_bootstrap_loop(
    data: pd.DataFrame,
    unit: str,
    control_units: np.ndarray,
    treated_units: np.ndarray,
    control_idx: np.ndarray,
    treated_idx: np.ndarray,
    n_control_units: int,
    n_treated_units: int,
    n_bootstrap: int,
    fit_callable: Callable[[pd.DataFrame, List[int]], float],
) -> Tuple[List[float], List[int]]:
    """Shared per-draw resample-and-refit loop for the TROP pairs bootstrap.

    Used by both ``TROP._bootstrap_variance`` (local) and
    ``TROP._bootstrap_variance_global``, whose Python-fallback loops were
    byte-identical apart from the refit call. RNG-free: ``control_idx`` /
    ``treated_idx`` are pre-generated by the caller via
    :func:`stratified_bootstrap_indices`, so the Rust and Python paths consume the
    identical draw sequence and this loop is deterministic. ``fit_callable(boot_data,
    nonconverg_tracker) -> float`` performs the fixed-lambda refit -- the only thing
    that differs between the local and global methods. Returns the list of finite
    per-draw estimates and the shared non-convergence tracker; the caller keeps its
    own (method-specific) warnings and ``np.std(ddof=1)`` SE computation.
    """
    bootstrap_estimates_list: List[float] = []
    nonconverg_tracker: List[int] = []

    for b in range(n_bootstrap):
        sampled_control = (
            control_units[control_idx[b]]
            if n_control_units > 0
            else np.array([], dtype=control_units.dtype)
        )
        sampled_treated = (
            treated_units[treated_idx[b]]
            if n_treated_units > 0
            else np.array([], dtype=treated_units.dtype)
        )
        sampled_units = np.concatenate([sampled_control, sampled_treated])

        # Create bootstrap sample with unique unit IDs
        boot_data = pd.concat(
            [
                data[data[unit] == u].assign(**{unit: f"{u}_{idx}"})
                for idx, u in enumerate(sampled_units)
            ],
            ignore_index=True,
        )

        try:
            est = fit_callable(boot_data, nonconverg_tracker)
            if np.isfinite(est):
                bootstrap_estimates_list.append(est)
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue

    return bootstrap_estimates_list, nonconverg_tracker


# Module-level convergence tolerance for SVD singular value truncation.
# Singular values below this threshold after soft-thresholding are treated
# as zero to improve numerical stability.
_CONVERGENCE_TOL_SVD: float = 1e-10


def _soft_threshold_svd(
    M: np.ndarray,
    threshold: float,
    convergence_tol: float = _CONVERGENCE_TOL_SVD,
) -> np.ndarray:
    """
    Apply soft-thresholding to singular values (proximal operator for nuclear norm).

    Parameters
    ----------
    M : np.ndarray
        Input matrix.
    threshold : float
        Soft-thresholding parameter.
    convergence_tol : float, default=1e-10
        Singular values below this after thresholding are treated as zero.

    Returns
    -------
    np.ndarray
        Matrix with soft-thresholded singular values.
    """
    if threshold <= 0:
        return M

    # Handle NaN/Inf values in input
    if not np.isfinite(M).all():
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD failed, return zero matrix
        return np.zeros_like(M)

    # Check for numerical issues in SVD output
    if not (np.isfinite(U).all() and np.isfinite(s).all() and np.isfinite(Vt).all()):
        # SVD produced non-finite values, return zero matrix
        return np.zeros_like(M)

    s_thresh = np.maximum(s - threshold, 0)

    # Use truncated reconstruction with only non-zero singular values
    nonzero_mask = s_thresh > convergence_tol
    if not np.any(nonzero_mask):
        return np.zeros_like(M)

    # Truncate to non-zero components for numerical stability
    U_trunc = U[:, nonzero_mask]
    s_trunc = s_thresh[nonzero_mask]
    Vt_trunc = Vt[nonzero_mask, :]

    # Compute result, suppressing expected numerical warnings from
    # ill-conditioned matrices during alternating minimization
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = (U_trunc * s_trunc) @ Vt_trunc

    # Replace any NaN/Inf in result with zeros
    if not np.isfinite(result).all():
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


class TROPLocalMixin:
    """Mixin providing local (observation-specific) estimation for TROP.

    Methods in this mixin access the following attributes from the main
    TROP class via ``self``:

    - Solver params: ``max_iter``, ``tol``
    - Inference params: ``n_bootstrap``, ``seed``
    - State: ``_precomputed``
    """

    # Type hints for attributes accessed from the main TROP class
    max_iter: int
    tol: float
    n_bootstrap: int
    seed: Optional[int]
    _precomputed: Optional[_PrecomputedStructures]

    # Convergence tolerance for SVD singular value truncation
    CONVERGENCE_TOL_SVD: float = 1e-10

    # =========================================================================
    # Preprocessing and distance computation
    # =========================================================================

    def _precompute_structures(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        control_unit_idx: np.ndarray,
        n_units: int,
        n_periods: int,
    ) -> _PrecomputedStructures:
        """
        Pre-compute data structures that are reused across LOOCV and estimation.

        This method computes once what would otherwise be computed repeatedly:
        - Pairwise unit distance matrix
        - Time distance vectors
        - Masks and indices

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        control_unit_idx : np.ndarray
            Indices of control units.
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.

        Returns
        -------
        _PrecomputedStructures
            Pre-computed structures for efficient reuse.
        """
        # Compute pairwise unit distances (for all observation-specific weights)
        # Following Equation 3 (page 7): RMSE between units over pre-treatment
        if HAS_RUST_BACKEND and _rust_unit_distance_matrix is not None:
            # Use Rust backend for parallel distance computation (4-8x speedup)
            unit_dist_matrix = _rust_unit_distance_matrix(Y, D.astype(np.float64))
        else:
            unit_dist_matrix = self._compute_all_unit_distances(Y, D, n_units, n_periods)

        # Pre-compute time distance vectors for each target period
        # Time distance: |t - s| for all s and each target t
        time_dist_matrix = np.abs(
            np.arange(n_periods)[:, np.newaxis] - np.arange(n_periods)[np.newaxis, :]
        )  # (n_periods, n_periods) where [t, s] = |t - s|

        # Control and treatment masks
        control_mask = D == 0
        treated_mask = D == 1

        # Identify treated observations
        treated_observations = list(zip(*np.where(treated_mask)))

        # Control observations for LOOCV
        control_obs = [
            (t, i)
            for t in range(n_periods)
            for i in range(n_units)
            if control_mask[t, i] and not np.isnan(Y[t, i])
        ]

        return {
            "unit_dist_matrix": unit_dist_matrix,
            "time_dist_matrix": time_dist_matrix,
            "control_mask": control_mask,
            "treated_mask": treated_mask,
            "treated_observations": treated_observations,
            "control_obs": control_obs,
            "control_unit_idx": control_unit_idx,
            "D": D,
            "Y": Y,
            "n_units": n_units,
            "n_periods": n_periods,
        }

    def _compute_all_unit_distances(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        n_units: int,
        n_periods: int,
    ) -> np.ndarray:
        """
        Compute pairwise unit distance matrix using vectorized operations.

        Following Equation 3 (page 7):
        dist_unit_{-t}(j, i) = sqrt(sum_u (Y_{iu} - Y_{ju})^2 / n_valid)

        For efficiency, we compute a base distance matrix excluding all treated
        observations, which provides a good approximation. The exact per-observation
        distances are refined when needed.

        Uses vectorized numpy operations with masked arrays for O(n^2) complexity
        but with highly optimized inner loops via numpy/BLAS.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.

        Returns
        -------
        np.ndarray
            Pairwise distance matrix (n_units x n_units).
        """
        # Mask for valid observations: control periods only (D=0), non-NaN
        valid_mask = (D == 0) & ~np.isnan(Y)

        # Replace invalid values with NaN for masked computation
        Y_masked = np.where(valid_mask, Y, np.nan)

        # Transpose to (n_units, n_periods) for easier broadcasting
        Y_T = Y_masked.T  # (n_units, n_periods)

        # Compute pairwise squared differences using broadcasting
        # Y_T[:, np.newaxis, :] has shape (n_units, 1, n_periods)
        # Y_T[np.newaxis, :, :] has shape (1, n_units, n_periods)
        # diff has shape (n_units, n_units, n_periods)
        diff = Y_T[:, np.newaxis, :] - Y_T[np.newaxis, :, :]
        sq_diff = diff**2

        # Count valid (non-NaN) observations per pair
        # A difference is valid only if both units have valid observations
        valid_diff = ~np.isnan(sq_diff)
        n_valid = np.sum(valid_diff, axis=2)  # (n_units, n_units)

        # Compute sum of squared differences (treating NaN as 0)
        sq_diff_sum = np.nansum(sq_diff, axis=2)  # (n_units, n_units)

        # Compute RMSE distance: sqrt(sum / n_valid)
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            dist_matrix = np.sqrt(sq_diff_sum / n_valid)

        # Set pairs with no valid observations to inf
        dist_matrix = np.where(n_valid > 0, dist_matrix, np.inf)

        # Ensure diagonal is 0 (same unit distance)
        np.fill_diagonal(dist_matrix, 0.0)

        return dist_matrix

    def _compute_unit_distance_for_obs(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        j: int,
        i: int,
        target_period: int,
    ) -> float:
        """
        Compute observation-specific pairwise distance from unit j to unit i.

        This is the exact computation from Equation 3, excluding the target period.
        Used when the base distance matrix approximation is insufficient.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix.
        j : int
            Control unit index.
        i : int
            Treated unit index.
        target_period : int
            Target period to exclude.

        Returns
        -------
        float
            Pairwise RMSE distance.
        """
        n_periods = Y.shape[0]

        # Mask: exclude target period, both units must be untreated, non-NaN
        valid = np.ones(n_periods, dtype=bool)
        valid[target_period] = False
        valid &= (D[:, i] == 0) & (D[:, j] == 0)
        valid &= ~np.isnan(Y[:, i]) & ~np.isnan(Y[:, j])

        if np.any(valid):
            sq_diffs = (Y[valid, i] - Y[valid, j]) ** 2
            return np.sqrt(np.mean(sq_diffs))
        else:
            return np.inf

    # =========================================================================
    # Observation-specific estimation
    # =========================================================================

    def _compute_observation_weights(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        i: int,
        t: int,
        lambda_time: float,
        lambda_unit: float,
        control_unit_idx: np.ndarray,
        n_units: int,
        n_periods: int,
    ) -> np.ndarray:
        """
        Compute observation-specific weight matrix for treated observation (i, t).

        Following the paper's Algorithm 2 (page 27) and Equation 2 (page 7):
        - Time weights theta_s^{i,t} = exp(-lambda_time * |t - s|)
        - Unit weights omega_j^{i,t} = exp(-lambda_unit * dist_unit_{-t}(j, i))

        Weights are assigned for every unit ``j != i`` (distance-based, per
        Eq. 2/3). Treated-cell exclusion is handled by the `(1 - W_{js})`
        factor applied inside ``_estimate_model`` via the control mask, not
        by gating ``ω_j`` on ``D[t, j]``. Same-cohort donors therefore
        contribute via their pre-treatment rows, and future-cohort donors
        contribute via rows where both units are still untreated.

        Always computes from the function-argument ``Y, D``; does not read
        ``self._precomputed``. Under bootstrap the caller passes resampled
        ``Y, D``, and a prior version of this method silently fell through to
        the original-panel cache via a ``_precomputed`` branch, producing
        stale unit distances.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        i : int
            Treated unit index.
        t : int
            Treatment period index.
        lambda_time : float
            Time weight decay parameter.
        lambda_unit : float
            Unit weight decay parameter.
        control_unit_idx : np.ndarray
            Indices of never-treated units (for backward compatibility, but not
            used for weight computation - we use D matrix directly).
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.

        Returns
        -------
        np.ndarray
            Weight matrix (n_periods x n_units) for observation (i, t).
        """
        # Time distance: |t - s| following paper's Equation 3 (page 7)
        dist_time = np.abs(np.arange(n_periods) - t)
        time_weights = np.exp(-lambda_time * dist_time)

        # Unit weights ω_j = exp(-λ_unit × dist(j, i)) for all j ≠ i per Eq. 2/3.
        # No target-period gate: same-cohort donors enter with distance-based
        # weight (their pre-treatment rows contribute via theta_s * omega_j;
        # their post-treatment cells are zeroed by the control mask (1-D_{js})
        # applied inside ``_estimate_model``). Matches Rust's compute_weight_matrix.
        unit_weights = np.zeros(n_units)

        if lambda_unit == 0:
            # Uniform weights when lambda_unit = 0 — all units get 1.
            # Control masking in _estimate_model handles treated-cell exclusion.
            unit_weights[:] = 1.0
        else:
            for j in range(n_units):
                if j != i:
                    # Compute distance excluding target period t (Issue B fix)
                    dist = self._compute_unit_distance_for_obs(Y, D, j, i, t)
                    if np.isinf(dist):
                        unit_weights[j] = 0.0
                    else:
                        unit_weights[j] = np.exp(-lambda_unit * dist)

        # Target unit gets weight 1 (will be masked out in estimation via
        # the control mask, matching the paper's (1-W_{js}) factor).
        unit_weights[i] = 1.0

        # Weight matrix: outer product (n_periods x n_units)
        W = np.outer(time_weights, unit_weights)

        return W

    def _soft_threshold_svd(
        self,
        M: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Delegate to module-level ``_soft_threshold_svd``."""
        return _soft_threshold_svd(M, threshold, self.CONVERGENCE_TOL_SVD)

    def _weighted_nuclear_norm_solve(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        L_init: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        lambda_nn: float,
        max_inner_iter: int = 20,
    ) -> np.ndarray:
        """
        Solve weighted nuclear norm problem using iterative weighted soft-impute.

        Issue C fix: Implements the weighted nuclear norm optimization from the
        paper's Equation 2 (page 7). The full objective is:
            min_L sum W_{ti}(R_{ti} - L_{ti})^2 + lambda_nn||L||_*

        This uses proximal gradient descent (Mazumder et al. 2010) with
        FISTA/Nesterov acceleration. Lipschitz constant L_f = 2*max(W),
        step size eta = 1/(2*max(W)), proximal threshold eta*lambda_nn:
            G_k = L_k + (W/max(W)) * (R - L_k)
            L_{k+1} = prox_{eta*lambda_nn*||*||_*}(G_k)

        IMPORTANT: For observations with W=0 (treated observations), we keep
        L values from the previous iteration rather than setting L = R, which
        would absorb the treatment effect.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        W : np.ndarray
            Weight matrix (n_periods x n_units), non-negative. W=0 indicates
            observations that should not be used for fitting (treated obs).
        L_init : np.ndarray
            Initial estimate of L matrix.
        alpha : np.ndarray
            Current unit fixed effects estimate.
        beta : np.ndarray
            Current time fixed effects estimate.
        lambda_nn : float
            Nuclear norm regularization parameter.
        max_inner_iter : int, default=20
            Maximum inner iterations for the proximal algorithm.

        Returns
        -------
        np.ndarray
            Updated L matrix estimate.
        """
        # Compute target residual R = Y - alpha - beta
        R = Y - alpha[np.newaxis, :] - beta[:, np.newaxis]

        # Handle invalid values
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

        # For observations with W=0 (treated obs), keep L_init instead of R
        # This prevents L from absorbing the treatment effect
        valid_obs_mask = W > 0
        R_masked = np.where(valid_obs_mask, R, L_init)

        if lambda_nn <= 0:
            # No regularization - just return masked residual
            # Use soft-thresholding with threshold=0 which returns the input
            return R_masked

        # Normalize weights so max is 1 (for step size stability)
        W_max = np.max(W)
        if W_max > 0:
            W_norm = W / W_max
        else:
            W_norm = W

        # Initialize L
        L = L_init.copy()
        L_prev = L.copy()
        t_fista = 1.0

        # Proximal gradient iteration with FISTA/Nesterov acceleration
        # This solves: min_L ||W^{1/2} * (R - L)||_F^2 + lambda||L||_*
        # Lipschitz constant L_f = 2*max(W), so eta = 1/(2*max(W))
        # Threshold = eta*lambda_nn = lambda_nn/(2*max(W))
        for _ in range(max_inner_iter):
            L_old = L.copy()

            # FISTA momentum
            t_fista_new = (1.0 + np.sqrt(1.0 + 4.0 * t_fista**2)) / 2.0
            momentum = (t_fista - 1.0) / t_fista_new
            L_momentum = L + momentum * (L - L_prev)

            # Gradient step from momentum point: L_m + W * (R - L_m)
            # For W=0 observations, this keeps L_m unchanged
            gradient_step = L_momentum + W_norm * (R_masked - L_momentum)

            # Proximal step: soft-threshold singular values
            L_prev = L.copy()
            threshold = lambda_nn / (2.0 * W_max) if W_max > 0 else lambda_nn / 2.0
            L = self._soft_threshold_svd(gradient_step, threshold)
            t_fista = t_fista_new

            # Check convergence
            if np.max(np.abs(L - L_old)) < self.tol:
                break

        return L

    def _estimate_model(
        self,
        Y: np.ndarray,
        control_mask: np.ndarray,
        weight_matrix: np.ndarray,
        lambda_nn: float,
        n_units: int,
        n_periods: int,
        exclude_obs: Optional[Tuple[int, int]] = None,
        _nonconvergence_tracker: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the model: Y = alpha + beta + L + tau*D + eps with nuclear norm penalty on L.

        Uses alternating minimization with vectorized operations:
        1. Fix L, solve for alpha, beta via weighted means
        2. Fix alpha, beta, solve for L via soft-thresholding

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        control_mask : np.ndarray
            Boolean mask for control observations.
        weight_matrix : np.ndarray
            Pre-computed global weight matrix (n_periods x n_units).
        lambda_nn : float
            Nuclear norm regularization parameter.
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.
        exclude_obs : tuple, optional
            (t, i) observation to exclude (for LOOCV).

        Returns
        -------
        tuple
            (alpha, beta, L) estimated parameters.
        """
        W = weight_matrix

        # Mask for estimation (control obs only, excluding LOOCV obs if specified)
        est_mask = control_mask.copy()
        if exclude_obs is not None:
            t_ex, i_ex = exclude_obs
            est_mask[t_ex, i_ex] = False

        # Handle missing values
        valid_mask = ~np.isnan(Y) & est_mask

        # Initialize
        alpha = np.zeros(n_units)
        beta = np.zeros(n_periods)
        L = np.zeros((n_periods, n_units))

        # Pre-compute masked weights for vectorized operations
        # Set weights to 0 where not valid
        W_masked = W * valid_mask

        # Pre-compute weight sums per unit and per time (for denominator)
        # shape: (n_units,) and (n_periods,)
        weight_sum_per_unit = np.sum(W_masked, axis=0)  # sum over periods
        weight_sum_per_time = np.sum(W_masked, axis=1)  # sum over units

        # Handle units/periods with zero weight sum
        unit_has_obs = weight_sum_per_unit > 0
        time_has_obs = weight_sum_per_time > 0

        # Create safe denominators (avoid division by zero)
        safe_unit_denom = np.where(unit_has_obs, weight_sum_per_unit, 1.0)
        safe_time_denom = np.where(time_has_obs, weight_sum_per_time, 1.0)

        # Replace NaN in Y with 0 for computation (mask handles exclusion)
        Y_safe = np.where(np.isnan(Y), 0.0, Y)

        # Alternating minimization following Algorithm 1 (page 9)
        # Minimize: sum W_{ti}(Y_{ti} - alpha_i - beta_t - L_{ti})^2 + lambda_nn||L||_*
        converged = False
        for _ in range(self.max_iter):
            alpha_old = alpha.copy()
            beta_old = beta.copy()
            L_old = L.copy()

            # Step 1: Update alpha and beta (weighted least squares)
            # Following Equation 2 (page 7), fix L and solve for alpha, beta
            # R = Y - L (residual without fixed effects)
            R = Y_safe - L

            # Alpha update (unit fixed effects):
            # alpha_i = argmin_alpha sum_t W_{ti}(R_{ti} - alpha - beta_t)^2
            # Solution: alpha_i = sum_t W_{ti}(R_{ti} - beta_t) / sum_t W_{ti}
            R_minus_beta = R - beta[:, np.newaxis]  # (n_periods, n_units)
            weighted_R_minus_beta = W_masked * R_minus_beta
            alpha_numerator = np.sum(weighted_R_minus_beta, axis=0)  # (n_units,)
            alpha = np.where(unit_has_obs, alpha_numerator / safe_unit_denom, 0.0)

            # Beta update (time fixed effects):
            # beta_t = argmin_beta sum_i W_{ti}(R_{ti} - alpha_i - beta)^2
            # Solution: beta_t = sum_i W_{ti}(R_{ti} - alpha_i) / sum_i W_{ti}
            R_minus_alpha = R - alpha[np.newaxis, :]  # (n_periods, n_units)
            weighted_R_minus_alpha = W_masked * R_minus_alpha
            beta_numerator = np.sum(weighted_R_minus_alpha, axis=1)  # (n_periods,)
            beta = np.where(time_has_obs, beta_numerator / safe_time_denom, 0.0)

            # Step 2: Update L with weighted nuclear norm penalty
            # Issue C fix: Use weighted soft-impute to properly account for
            # observation weights in the nuclear norm optimization.
            # Following Equation 2 (page 7): min_L sum W_{ti}(Y - alpha - beta - L)^2 + lambda||L||_*
            L = self._weighted_nuclear_norm_solve(
                Y_safe, W_masked, L, alpha, beta, lambda_nn, max_inner_iter=10
            )

            # Check convergence
            alpha_diff = np.max(np.abs(alpha - alpha_old))
            beta_diff = np.max(np.abs(beta - beta_old))
            L_diff = np.max(np.abs(L - L_old))

            if max(alpha_diff, beta_diff, L_diff) < self.tol:
                converged = True
                break
        if not converged:
            if _nonconvergence_tracker is not None:
                _nonconvergence_tracker.append(1)
            else:
                warn_if_not_converged(
                    converged,
                    "TROP local alternating minimization",
                    self.max_iter,
                    self.tol,
                )

        return alpha, beta, L

    def _loocv_score_obs_specific(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        control_mask: np.ndarray,
        control_unit_idx: np.ndarray,
        lambda_time: float,
        lambda_unit: float,
        lambda_nn: float,
        n_units: int,
        n_periods: int,
    ) -> float:
        """
        Compute leave-one-out cross-validation score with observation-specific weights.

        Following the paper's Equation 5 (page 8):
        Q(lambda) = sum_{j,s: D_js=0} [tau_js^loocv(lambda)]^2

        For each control observation (j, s), treat it as pseudo-treated,
        compute observation-specific weights, fit model excluding (j, s),
        and sum squared pseudo-treatment effects.

        Uses pre-computed structures when available for efficiency.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        control_mask : np.ndarray
            Boolean mask for control observations.
        control_unit_idx : np.ndarray
            Indices of control units.
        lambda_time : float
            Time weight decay parameter.
        lambda_unit : float
            Unit weight decay parameter.
        lambda_nn : float
            Nuclear norm regularization parameter.
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.

        Returns
        -------
        float
            LOOCV score (lower is better).
        """
        # Use pre-computed control observations if available
        if self._precomputed is not None:
            control_obs = self._precomputed["control_obs"]
        else:
            # Get all control observations
            control_obs = [
                (t, i)
                for t in range(n_periods)
                for i in range(n_units)
                if control_mask[t, i] and not np.isnan(Y[t, i])
            ]

        # Empty control set check: if no control observations, return infinity
        # A score of 0.0 would incorrectly "win" over legitimate parameters
        if len(control_obs) == 0:
            warnings.warn(
                f"LOOCV: No valid control observations for "
                f"\u03bb=({lambda_time}, {lambda_unit}, {lambda_nn}). "
                "Returning infinite score.",
                UserWarning,
            )
            return np.inf

        tau_squared_sum = 0.0
        n_valid = 0
        nonconverg_tracker: List[int] = []

        for t, i in control_obs:
            try:
                # Compute observation-specific weights for pseudo-treated (i, t)
                # Uses pre-computed distance matrices when available
                weight_matrix = self._compute_observation_weights(
                    Y, D, i, t, lambda_time, lambda_unit, control_unit_idx, n_units, n_periods
                )

                # Estimate model excluding observation (t, i)
                alpha, beta, L = self._estimate_model(
                    Y,
                    control_mask,
                    weight_matrix,
                    lambda_nn,
                    n_units,
                    n_periods,
                    exclude_obs=(t, i),
                    _nonconvergence_tracker=nonconverg_tracker,
                )

                # Pseudo treatment effect
                tau_ti = Y[t, i] - alpha[i] - beta[t] - L[t, i]
                tau_squared_sum += tau_ti**2
                n_valid += 1

            except (np.linalg.LinAlgError, ValueError):
                # Per Equation 5: Q(lambda) must sum over ALL D==0 cells
                # Any failure means this lambda cannot produce valid estimates for all cells
                warnings.warn(
                    f"LOOCV: Fit failed for observation ({t}, {i}) with "
                    f"\u03bb=({lambda_time}, {lambda_unit}, {lambda_nn}). "
                    "Returning infinite score per Equation 5.",
                    UserWarning,
                )
                return np.inf

        if nonconverg_tracker:
            warn_if_not_converged(
                False,
                f"TROP local LOOCV: {len(nonconverg_tracker)} of "
                f"{len(control_obs)} per-observation fits did not converge "
                f"(\u03bb=({lambda_time}, {lambda_unit}, {lambda_nn}))",
                self.max_iter,
                self.tol,
            )

        # Return SUM of squared pseudo-treatment effects per Equation 5 (page 8):
        # Q(lambda) = sum_{j,s: D_js=0} [tau_js^loocv(lambda)]^2
        return tau_squared_sum

    def _bootstrap_variance(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        optimal_lambda: Tuple[float, float, float],
        Y: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        control_unit_idx: Optional[np.ndarray] = None,
        survey_design=None,
        unit_weight_arr: Optional[np.ndarray] = None,
        resolved_survey=None,
        force_python: bool = False,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute bootstrap standard error using unit-level block bootstrap.

        ``force_python=True`` skips the Rust happy path so the cell-specific
        estimability guard in ``_fit_with_fixed_lambda`` is applied per draw. The
        point fit sets this whenever it trimmed any non-estimable treated cell
        (the Rust per-cell tau path lacks the guard), keeping the bootstrap SE
        and the point ATT on the same estimable-cell set.

        When the optional Rust backend is available and the matrix parameters
        (Y, D, control_unit_idx) are provided, uses parallelized Rust
        implementation for 5-15x speedup. Falls back to Python implementation
        if Rust is unavailable or if matrix parameters are not provided.

        When a full survey design (strata/PSU/FPC) is present, uses Rao-Wu
        rescaled bootstrap instead, which skips the Rust path.

        Parameters
        ----------
        data : pd.DataFrame
            Original data in long format with unit, time, outcome, and treatment.
        outcome : str
            Name of the outcome column in data.
        treatment : str
            Name of the treatment indicator column in data.
        unit : str
            Name of the unit identifier column in data.
        time : str
            Name of the time period column in data.
        optimal_lambda : tuple of float
            Optimal tuning parameters (lambda_time, lambda_unit, lambda_nn)
            from cross-validation. Used for model estimation in each bootstrap.
        Y : np.ndarray, optional
            Outcome matrix of shape (n_periods, n_units). Required for Rust
            backend acceleration. If None, falls back to Python implementation.
        D : np.ndarray, optional
            Treatment indicator matrix of shape (n_periods, n_units) where
            D[t,i]=1 indicates unit i is treated at time t. Required for Rust
            backend acceleration.
        control_unit_idx : np.ndarray, optional
            Array of indices for control units (never-treated). Required for
            Rust backend acceleration.
        survey_design : SurveyDesign, optional
            Survey design specification.
        unit_weight_arr : np.ndarray, optional
            Unit-level survey weights.
        resolved_survey : ResolvedSurveyDesign, optional
            Resolved survey design (observation-level).

        Returns
        -------
        se : float
            Bootstrap standard error of the ATT estimate.
        bootstrap_estimates : np.ndarray
            Array of ATT estimates from each bootstrap iteration. Length may
            be less than n_bootstrap if some iterations failed.

        Notes
        -----
        Uses unit-level block bootstrap where entire unit time series are
        resampled with replacement. This preserves within-unit correlation
        structure and is appropriate for panel data.
        """
        lambda_time, lambda_unit, lambda_nn = optimal_lambda

        # Check for full survey design (strata/PSU/FPC present)
        _has_full_design = resolved_survey is not None and (
            resolved_survey.strata is not None
            or resolved_survey.psu is not None
            or resolved_survey.fpc is not None
        )

        # Full survey design: use Python Rao-Wu rescaled bootstrap
        if _has_full_design:
            return self._bootstrap_rao_wu_local(
                data,
                outcome,
                treatment,
                unit,
                time,
                optimal_lambda,
                resolved_survey,
                survey_design,
            )

        # Stratified bootstrap pools (shared by Rust and Python paths).
        # Paper's Algorithm 3 (page 27) specifies sampling N_0 control rows
        # and N_1 treated rows separately to preserve treatment ratio.
        unit_ever_treated = data.groupby(unit)[treatment].max()
        treated_units = np.array(unit_ever_treated[unit_ever_treated == 1].index)
        control_units = np.array(unit_ever_treated[unit_ever_treated == 0].index)
        n_treated_units = len(treated_units)
        n_control_units = len(control_units)

        # Pre-generate stratified bootstrap indices via numpy (Python-canonical RNG).
        # Aligns the RNG layer between backends. Combined with the Rust weight-
        # matrix de-normalization and the Python `_compute_observation_weights`
        # cache-fallthrough removal (also shipped with this parity work), local-
        # method Rust and Python produce matching bootstrap SE up to solver-path
        # roundoff (~1e-7); asserted at atol=1e-5 in the parity regression guard.
        rng = np.random.default_rng(self.seed)
        control_idx, treated_idx = stratified_bootstrap_indices(
            rng, n_control_units, n_treated_units, self.n_bootstrap
        )

        # Try Rust backend for parallel bootstrap (5-15x speedup)
        # Only used for pweight-only designs (no strata/PSU/FPC).
        # Routed to the Python loop when the cell-specific estimability contract
        # could diverge from Rust: (a) non_absorbing fits -- a fully non-absorbing
        # panel can have zero never-treated units (empty control stratum -> Rust
        # can return a degenerate ~0 SE), and the Rust per-cell tau path lacks the
        # estimability guard; (b) force_python -- the point fit trimmed at least
        # one non-estimable treated cell (e.g. an unbalanced absorbing panel), so
        # the Rust path (no guard) would compute SE over a different cell set than
        # the point ATT. The Python `_fit_with_fixed_lambda` enforces the guard
        # per draw in both cases.
        if (
            HAS_RUST_BACKEND
            and _rust_bootstrap_trop_variance is not None
            and self._precomputed is not None
            and Y is not None
            and D is not None
            and n_control_units > 0
            and not getattr(self, "non_absorbing", False)
            and not force_python
        ):
            try:
                control_mask = self._precomputed["control_mask"]
                time_dist_matrix = self._precomputed["time_dist_matrix"].astype(np.int64)

                bootstrap_estimates, se = _rust_bootstrap_trop_variance(
                    Y,
                    D.astype(np.float64),
                    control_mask.astype(np.uint8),
                    time_dist_matrix,
                    lambda_time,
                    lambda_unit,
                    lambda_nn,
                    self.n_bootstrap,
                    self.max_iter,
                    self.tol,
                    control_idx,
                    treated_idx,
                    unit_weight_arr,
                )

                if len(bootstrap_estimates) > 0:
                    warn_bootstrap_failure_rate(
                        n_success=len(bootstrap_estimates),
                        n_attempted=self.n_bootstrap,
                        context="TROP local bootstrap (Rust)",
                    )
                    return float(se), bootstrap_estimates
                logger.debug("Rust bootstrap returned 0 samples, falling back to Python")
            except Exception as e:
                logger.debug("Rust bootstrap variance failed, falling back to Python: %s", e)
                warnings.warn(
                    f"Rust backend failed for bootstrap variance; "
                    f"falling back to Python. Performance may be reduced. "
                    f"Error: {e}",
                    UserWarning,
                    stacklevel=2,
                )

        # Python fallback: consume the same indices the Rust branch would have used.
        bootstrap_estimates_list, nonconverg_tracker = _run_trop_bootstrap_loop(
            data,
            unit,
            control_units,
            treated_units,
            control_idx,
            treated_idx,
            n_control_units,
            n_treated_units,
            self.n_bootstrap,
            # Fit with fixed lambda (skip LOOCV for speed)
            lambda boot_data, tracker: self._fit_with_fixed_lambda(
                boot_data,
                outcome,
                treatment,
                unit,
                time,
                optimal_lambda,
                survey_design=survey_design,
                _nonconvergence_tracker=tracker,
            ),
        )

        bootstrap_estimates = np.array(bootstrap_estimates_list)

        if nonconverg_tracker:
            warn_if_not_converged(
                False,
                f"TROP local bootstrap: {len(nonconverg_tracker)} non-converged "
                f"per-observation fits across {self.n_bootstrap} bootstrap replicates",
                self.max_iter,
                self.tol,
            )

        warn_bootstrap_failure_rate(
            n_success=len(bootstrap_estimates),
            n_attempted=self.n_bootstrap,
            context="TROP local bootstrap",
        )
        if len(bootstrap_estimates) == 0:
            return np.nan, np.array([])

        se = np.std(bootstrap_estimates, ddof=1)
        return float(se), bootstrap_estimates

    def _bootstrap_rao_wu_local(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        optimal_lambda: Tuple[float, float, float],
        resolved_survey,
        survey_design,
    ) -> Tuple[float, np.ndarray]:
        """
        Rao-Wu rescaled bootstrap for local method with full survey design.

        Instead of physically resampling units, each iteration generates
        rescaled observation weights via Rao-Wu (1988) weight perturbation.
        Cross-classifies survey strata with treatment group to preserve
        the stratified resampling structure.

        Parameters
        ----------
        data : pd.DataFrame
            Original data.
        outcome, treatment, unit, time : str
            Column names.
        optimal_lambda : tuple
            Optimal tuning parameters (lambda_time, lambda_unit, lambda_nn).
        resolved_survey : ResolvedSurveyDesign
            Resolved survey design (observation-level).
        survey_design : SurveyDesign
            Original survey design specification.

        Returns
        -------
        Tuple[float, np.ndarray]
            (se, bootstrap_estimates).
        """

        from diff_diff.bootstrap_utils import generate_rao_wu_weights
        from diff_diff.linalg import _factorize_cluster_ids
        from diff_diff.survey import ResolvedSurveyDesign

        rng = np.random.default_rng(self.seed)

        # Build unit-level resolved survey with cross-classified strata
        all_units = sorted(data[unit].unique())
        n_units = len(all_units)

        # Determine treatment status per unit
        unit_ever_treated = data.groupby(unit)[treatment].max()
        treatment_group = np.array([int(unit_ever_treated[u]) for u in all_units], dtype=np.int64)

        # Extract unit-level survey design fields
        first_rows = data.groupby(unit).first().loc[all_units]

        # Weights (unit-level)
        if survey_design.weights is not None:
            unit_weights = first_rows[survey_design.weights].values.astype(np.float64)
        else:
            unit_weights = np.ones(n_units, dtype=np.float64)

        # Strata: cross-classify survey strata x treatment group
        if survey_design.strata is not None:
            survey_strata = first_rows[survey_design.strata].values
            cross_labels = np.array([f"{s}_{g}" for s, g in zip(survey_strata, treatment_group)])
            cross_strata = _factorize_cluster_ids(cross_labels)
        else:
            # No survey strata: use treatment group as strata
            cross_strata = treatment_group.copy()
        n_strata = len(np.unique(cross_strata))

        # PSU (unit-level)
        psu_arr = None
        n_psu = 0
        if survey_design.psu is not None:
            psu_raw = first_rows[survey_design.psu].values
            if survey_design.nest and survey_design.strata is not None:
                combined = np.array([f"{s}_{p}" for s, p in zip(cross_strata, psu_raw)])
                psu_arr = _factorize_cluster_ids(combined)
            else:
                psu_arr = _factorize_cluster_ids(psu_raw)
            n_psu = len(np.unique(psu_arr))
        else:
            # Implicit PSU: each unit is its own PSU
            psu_arr = np.arange(n_units, dtype=np.int64)
            n_psu = n_units

        # FPC (unit-level)
        fpc_arr = None
        if survey_design.fpc is not None:
            fpc_arr = first_rows[survey_design.fpc].values.astype(np.float64)

        unit_resolved = ResolvedSurveyDesign(
            weights=unit_weights,
            weight_type=resolved_survey.weight_type,
            strata=cross_strata,
            psu=psu_arr,
            fpc=fpc_arr,
            n_strata=n_strata,
            n_psu=n_psu,
            lonely_psu=resolved_survey.lonely_psu,
        )

        # Check for unidentified variance (single unstratified PSU)
        if (
            survey_design.psu is not None
            and unit_resolved.n_psu < 2
            and survey_design.strata is None
        ):
            return np.nan, np.array([])

        # Bootstrap loop: refit the full model per draw with Rao-Wu rescaled
        # weights, mirroring the physical-resampling bootstrap but using weight
        # perturbation instead of unit resampling.
        bootstrap_estimates_list = []
        nonconverg_tracker: List[int] = []

        for _ in range(self.n_bootstrap):
            try:
                # Generate Rao-Wu rescaled unit weights
                boot_weights = generate_rao_wu_weights(unit_resolved, rng)

                # Skip if all weights are zero
                if boot_weights.sum() == 0:
                    continue

                # Refit the full local model with rescaled weights
                att = self._fit_with_fixed_lambda(
                    data,
                    outcome,
                    treatment,
                    unit,
                    time,
                    optimal_lambda,
                    survey_design=survey_design,
                    unit_weight_arr=boot_weights,
                    _nonconvergence_tracker=nonconverg_tracker,
                )

                if np.isfinite(att):
                    bootstrap_estimates_list.append(att)
            except (ValueError, np.linalg.LinAlgError, KeyError):
                continue

        bootstrap_estimates = np.array(bootstrap_estimates_list)

        if nonconverg_tracker:
            warn_if_not_converged(
                False,
                f"TROP local Rao-Wu bootstrap: {len(nonconverg_tracker)} non-converged "
                f"per-observation fits across {self.n_bootstrap} bootstrap replicates",
                self.max_iter,
                self.tol,
            )

        warn_bootstrap_failure_rate(
            n_success=len(bootstrap_estimates),
            n_attempted=self.n_bootstrap,
            context="TROP local Rao-Wu bootstrap",
        )
        if len(bootstrap_estimates) == 0:
            return np.nan, np.array([])

        se = np.std(bootstrap_estimates, ddof=1)
        return float(se), bootstrap_estimates

    def _fit_with_fixed_lambda(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        fixed_lambda: Tuple[float, float, float],
        survey_design=None,
        unit_weight_arr: Optional[np.ndarray] = None,
        _nonconvergence_tracker: Optional[List[int]] = None,
    ) -> float:
        """
        Fit model with fixed tuning parameters (for bootstrap).

        Uses observation-specific weights following Algorithm 2.
        Returns only the ATT estimate.

        Parameters
        ----------
        unit_weight_arr : np.ndarray, optional
            Pre-computed unit-level weights (e.g. Rao-Wu rescaled weights).
            When provided, overrides weights extracted from survey_design.
        """
        lambda_time, lambda_unit, lambda_nn = fixed_lambda

        # Use pre-computed weights if provided (e.g. Rao-Wu bootstrap),
        # otherwise extract from survey_design.
        if unit_weight_arr is not None:
            local_weight_arr = unit_weight_arr
        elif survey_design is not None and survey_design.weights is not None:
            from diff_diff.survey import _extract_unit_survey_weights

            local_all_units = sorted(data[unit].unique())
            local_weight_arr = _extract_unit_survey_weights(
                data, unit, survey_design, local_all_units
            )
        else:
            local_weight_arr = None

        # Setup matrices
        all_units = sorted(data[unit].unique())
        all_periods = sorted(data[time].unique())

        n_units = len(all_units)
        n_periods = len(all_periods)

        # Vectorized: use pivot for O(1) reshaping instead of O(n) iterrows loop
        Y = (
            data.pivot(index=time, columns=unit, values=outcome)
            .reindex(index=all_periods, columns=all_units)
            .values
        )
        D = (
            data.pivot(index=time, columns=unit, values=treatment)
            .reindex(index=all_periods, columns=all_units)
            .fillna(0)
            .astype(int)
            .values
        )

        control_mask = D == 0

        # Get control unit indices
        unit_ever_treated = np.any(D == 1, axis=0)
        control_unit_idx = np.where(~unit_ever_treated)[0]

        # Get list of treated observations
        treated_observations = [
            (t, i) for t in range(n_periods) for i in range(n_units) if D[t, i] == 1
        ]

        if not treated_observations:
            raise ValueError("No treated observations")

        # Compute ATT using observation-specific weights (Algorithm 2)
        tau_values = []
        tau_weights = []
        for t, i in treated_observations:
            # Skip non-finite outcomes (match main fit NaN contract)
            if not np.isfinite(Y[t, i]):
                continue

            # Compute observation-specific weights for this (i, t)
            weight_matrix = self._compute_observation_weights(
                Y, D, i, t, lambda_time, lambda_unit, control_unit_idx, n_units, n_periods
            )

            # Skip non-estimable cells (same predicate as the main fit): if the
            # target unit or target period has no weighted observed control cell,
            # alpha_i / beta_t are unidentified and tau leaks the fixed effect,
            # silently biasing the draw's ATT. A draw with no estimable cell
            # returns NaN and is counted as a failed replicate.
            if not _treated_cell_is_estimable(control_mask, Y, weight_matrix, i, t):
                continue

            # Fit model with these weights
            alpha, beta, L = self._estimate_model(
                Y,
                control_mask,
                weight_matrix,
                lambda_nn,
                n_units,
                n_periods,
                _nonconvergence_tracker=_nonconvergence_tracker,
            )

            # Compute treatment effect: tau_{it} = Y_{it} - alpha_i - beta_t - L_{it}
            tau = Y[t, i] - alpha[i] - beta[t] - L[t, i]
            tau_values.append(tau)
            if local_weight_arr is not None:
                tau_weights.append(local_weight_arr[i])

        if not tau_values:
            return float("nan")
        if local_weight_arr is not None:
            # Guard against a degenerate weighted draw: after non-estimable cells
            # are skipped, the remaining estimable cells can all carry zero
            # (rescaled survey / Rao-Wu) weight, which would make np.average raise
            # ZeroDivisionError. Treat such a draw as failed (NaN) per the
            # bootstrap NaN-on-degenerate contract.
            weight_sum = float(np.sum(tau_weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                return float("nan")
            return float(np.average(tau_values, weights=tau_weights))
        return float(np.mean(tau_values))
