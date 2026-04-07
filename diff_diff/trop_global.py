"""
Global estimation method for the TROP estimator.

Contains the TROPGlobalMixin class with all methods for the global
(joint) estimation pathway. The global method fits a single weighted
model on control observations and extracts per-observation treatment
effects as post-hoc residuals.

This module is used via mixin inheritance — see trop.py for the
main TROP class definition.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_bootstrap_trop_variance_global,
    _rust_loocv_grid_search_global,
)
from diff_diff.trop_local import _soft_threshold_svd
from diff_diff.trop_results import TROPResults
from diff_diff.utils import safe_inference


class TROPGlobalMixin:
    """Mixin providing global estimation method for TROP.

    Methods in this mixin access the following attributes from the main
    TROP class via ``self``:

    - Tuning grids: ``lambda_time_grid``, ``lambda_unit_grid``, ``lambda_nn_grid``
    - Solver params: ``max_iter``, ``tol``
    - Inference params: ``alpha``, ``n_bootstrap``, ``seed``
    - State: ``results_``, ``is_fitted_``

    """

    # Type hints for attributes accessed from the main TROP class
    lambda_time_grid: List[float]
    lambda_unit_grid: List[float]
    lambda_nn_grid: List[float]
    max_iter: int
    tol: float
    alpha: float
    n_bootstrap: int
    seed: Optional[int]
    results_: Any
    is_fitted_: bool

    def _compute_global_weights(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        lambda_time: float,
        lambda_unit: float,
        treated_periods: int,
        n_units: int,
        n_periods: int,
    ) -> np.ndarray:
        """
        Compute distance-based weights for global estimation.

        Following the reference implementation, weights are computed based on:
        - Time distance: distance to center of treated block
        - Unit distance: RMSE to average treated trajectory over pre-periods

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        lambda_time : float
            Time weight decay parameter.
        lambda_unit : float
            Unit weight decay parameter.
        treated_periods : int
            Number of post-treatment periods.
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.

        Returns
        -------
        np.ndarray
            Weight matrix (n_periods x n_units).
        """
        # Identify treated units (ever treated)
        treated_mask = np.any(D == 1, axis=0)
        treated_unit_idx = np.where(treated_mask)[0]

        if len(treated_unit_idx) == 0:
            raise ValueError("No treated units found")

        # Time weights: distance to center of treated block
        # Following reference: center = T - treated_periods/2
        center = n_periods - treated_periods / 2.0
        dist_time = np.abs(np.arange(n_periods, dtype=float) - center)
        delta_time = np.exp(-lambda_time * dist_time)

        # Unit weights: RMSE to average treated trajectory over pre-periods
        # Compute average treated trajectory (use nanmean to handle NaN)
        average_treated = np.nanmean(Y[:, treated_unit_idx], axis=1)

        # Pre-period mask: 1 in pre, 0 in post
        pre_mask = np.ones(n_periods, dtype=float)
        pre_mask[-treated_periods:] = 0.0

        # Compute RMS distance for each unit
        # dist_unit[i] = sqrt(sum_pre(avg_tr - Y_i)^2 / n_pre)
        # Use NaN-safe operations: treat NaN differences as 0 (excluded)
        diff = average_treated[:, np.newaxis] - Y
        diff_sq = np.where(np.isfinite(diff), diff**2, 0.0) * pre_mask[:, np.newaxis]

        # Count valid observations per unit in pre-period
        # Must check diff is finite (both Y and average_treated finite)
        # to match the periods contributing to diff_sq
        valid_count = np.sum(np.isfinite(diff) * pre_mask[:, np.newaxis], axis=0)
        sum_sq = np.sum(diff_sq, axis=0)
        n_pre = np.sum(pre_mask)

        if n_pre == 0:
            raise ValueError("No pre-treatment periods")

        # Track units with no valid pre-period data
        no_valid_pre = valid_count == 0

        # Use valid count per unit (avoid division by zero for calculation)
        valid_count_safe = np.maximum(valid_count, 1)
        dist_unit = np.sqrt(sum_sq / valid_count_safe)

        # Units with no valid pre-period data get zero weight
        # (dist is undefined, so we set it to inf -> delta_unit = exp(-inf) = 0)
        delta_unit = np.exp(-lambda_unit * dist_unit)
        delta_unit[no_valid_pre] = 0.0

        # Outer product: (n_periods x n_units)
        delta = np.outer(delta_time, delta_unit)

        # (1-W) masking: zero out treated observations per paper Eq. 2
        # Model is fit on control data only; tau extracted post-hoc
        delta = delta * (1 - D)

        return delta

    def _solve_global_model(
        self,
        Y: np.ndarray,
        delta: np.ndarray,
        lambda_nn: float,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Dispatch to no-lowrank or with-lowrank solver based on lambda_nn.

        Returns (mu, alpha, beta, L) in all cases.
        """
        n_periods, n_units = Y.shape
        if lambda_nn >= 1e10:
            mu, alpha, beta = self._solve_global_no_lowrank(Y, delta)
            L = np.zeros((n_periods, n_units))
        else:
            mu, alpha, beta, L = self._solve_global_with_lowrank(
                Y, delta, lambda_nn, self.max_iter, self.tol
            )
        return mu, alpha, beta, L

    @staticmethod
    def _extract_posthoc_tau(
        Y: np.ndarray,
        D: np.ndarray,
        mu: float,
        alpha: np.ndarray,
        beta: np.ndarray,
        L: np.ndarray,
        idx_to_unit: Optional[Dict] = None,
        idx_to_period: Optional[Dict] = None,
        unit_weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict, List[float]]:
        """
        Extract post-hoc treatment effects: tau_it = Y - mu - alpha - beta - L.

        Returns (att, treatment_effects_dict, tau_values_list).
        When idx_to_unit/idx_to_period are None, treatment_effects uses raw indices.
        """
        counterfactual = mu + alpha[np.newaxis, :] + beta[:, np.newaxis] + L
        tau_matrix = Y - counterfactual

        treated_mask = D == 1
        finite_mask = np.isfinite(Y)
        valid_treated = treated_mask & finite_mask

        tau_values = tau_matrix[valid_treated].tolist()
        if unit_weights is not None and tau_values:
            obs_weights = unit_weights[np.where(valid_treated)[1]]
            att = float(np.average(tau_values, weights=obs_weights))
        else:
            att = float(np.mean(tau_values)) if tau_values else np.nan

        # Build treatment effects dict
        treatment_effects: Dict = {}
        n_periods, n_units = D.shape
        for t in range(n_periods):
            for i in range(n_units):
                if D[t, i] == 1:
                    uid = idx_to_unit[i] if idx_to_unit is not None else i
                    tid = idx_to_period[t] if idx_to_period is not None else t
                    if finite_mask[t, i]:
                        treatment_effects[(uid, tid)] = tau_matrix[t, i]
                    else:
                        treatment_effects[(uid, tid)] = np.nan

        return att, treatment_effects, tau_values

    def _loocv_score_global(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        control_obs: List[Tuple[int, int]],
        lambda_time: float,
        lambda_unit: float,
        lambda_nn: float,
        treated_periods: int,
        n_units: int,
        n_periods: int,
    ) -> float:
        """
        Compute LOOCV score for global method with specific parameter combination.

        Following paper's Equation 5:
        Q(lambda) = sum_{j,s: D_js=0} [tau_js^loocv(lambda)]^2

        For global method, we exclude each control observation, fit the global model
        on remaining data, and compute the pseudo-treatment effect at the excluded obs.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        control_obs : List[Tuple[int, int]]
            List of (t, i) control observations for LOOCV.
        lambda_time : float
            Time weight decay parameter.
        lambda_unit : float
            Unit weight decay parameter.
        lambda_nn : float
            Nuclear norm regularization parameter.
        treated_periods : int
            Number of post-treatment periods.
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.

        Returns
        -------
        float
            LOOCV score (sum of squared pseudo-treatment effects).
        """
        # Compute global weights (same for all LOOCV iterations)
        delta = self._compute_global_weights(
            Y, D, lambda_time, lambda_unit, treated_periods, n_units, n_periods
        )

        tau_sq_sum = 0.0
        n_valid = 0

        for t_ex, i_ex in control_obs:
            # Create modified delta with excluded observation zeroed out
            delta_ex = delta.copy()
            delta_ex[t_ex, i_ex] = 0.0

            try:
                mu, alpha, beta, L = self._solve_global_model(Y, delta_ex, lambda_nn)

                # Pseudo treatment effect: tau = Y - mu - alpha - beta - L
                if np.isfinite(Y[t_ex, i_ex]):
                    tau_loocv = Y[t_ex, i_ex] - mu - alpha[i_ex] - beta[t_ex] - L[t_ex, i_ex]
                    tau_sq_sum += tau_loocv**2
                    n_valid += 1

            except (np.linalg.LinAlgError, ValueError):
                # Any failure means this lambda combination is invalid per Equation 5
                return np.inf

        if n_valid == 0:
            return np.inf

        return tau_sq_sum

    def _solve_global_no_lowrank(
        self,
        Y: np.ndarray,
        delta: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve TWFE via weighted least squares on control data (no low-rank).

        Solves: min sum (1-W)*delta_{it}(Y_{it} - mu - alpha_i - beta_t)^2

        The (1-W) masking is already applied to delta by _compute_global_weights,
        so treated observations have zero weight and do not affect the fit.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        delta : np.ndarray
            Weight matrix (n_periods x n_units), already (1-W) masked.

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray]
            (mu, alpha, beta) estimated parameters.
        """
        n_periods, n_units = Y.shape

        # Flatten matrices for regression
        y = Y.flatten()  # length n_periods * n_units
        weights = delta.flatten()

        # Handle NaN values: zero weight for NaN outcomes/weights, impute with 0
        # This ensures NaN observations don't contribute to estimation
        valid_y = np.isfinite(y)
        valid_w = np.isfinite(weights)
        valid_mask = valid_y & valid_w
        weights = np.where(valid_mask, weights, 0.0)
        y = np.where(valid_mask, y, 0.0)

        sqrt_weights = np.sqrt(np.maximum(weights, 0))

        # Check for all-zero weights (matches Rust's sum_w < 1e-10 check)
        sum_w = np.sum(weights)
        if sum_w < 1e-10:
            raise ValueError("All weights are zero - cannot estimate")

        # Build design matrix: [intercept, unit_dummies, time_dummies]
        # Drop first unit (unit 0) and first time (time 0) for identification
        n_obs = n_periods * n_units
        n_params = 1 + (n_units - 1) + (n_periods - 1)

        X = np.zeros((n_obs, n_params))
        X[:, 0] = 1.0  # intercept

        # Unit dummies (skip unit 0)
        for i in range(1, n_units):
            for t in range(n_periods):
                X[t * n_units + i, i] = 1.0

        # Time dummies (skip time 0)
        for t in range(1, n_periods):
            for i in range(n_units):
                X[t * n_units + i, (n_units - 1) + t] = 1.0

        # Apply weights
        X_weighted = X * sqrt_weights[:, np.newaxis]
        y_weighted = y * sqrt_weights

        # Solve weighted least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            coeffs = np.dot(np.linalg.pinv(X_weighted), y_weighted)

        # Extract parameters
        mu = coeffs[0]
        alpha = np.zeros(n_units)
        alpha[1:] = coeffs[1:n_units]
        beta = np.zeros(n_periods)
        beta[1:] = coeffs[n_units : (n_units + n_periods - 1)]

        return float(mu), alpha, beta

    def _solve_global_with_lowrank(
        self,
        Y: np.ndarray,
        delta: np.ndarray,
        lambda_nn: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve TWFE + low-rank on control data via alternating minimization.

        Solves: min sum (1-W)*delta_{it}(Y_{it} - mu - alpha_i - beta_t - L_{it})^2 + lambda_nn||L||_*

        The (1-W) masking is already applied to delta by _compute_global_weights,
        so treated observations have zero weight and do not affect the fit.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        delta : np.ndarray
            Weight matrix (n_periods x n_units), already (1-W) masked.
        lambda_nn : float
            Nuclear norm regularization parameter.
        max_iter : int, default=100
            Maximum iterations for alternating minimization.
        tol : float, default=1e-6
            Convergence tolerance.

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray, np.ndarray]
            (mu, alpha, beta, L) estimated parameters.
        """
        n_periods, n_units = Y.shape

        # Handle NaN values: impute with 0 for computations
        # The solver will also zero weights for NaN observations
        Y_safe = np.where(np.isfinite(Y), Y, 0.0)

        # Mask delta to exclude NaN outcomes from estimation
        # This ensures NaN observations don't contribute to the gradient step
        nan_mask = ~np.isfinite(Y)
        delta_masked = delta.copy()
        delta_masked[nan_mask] = 0.0

        # Precompute normalized weights and threshold (constant across iterations)
        delta_max = np.max(delta_masked)
        if delta_max > 0:
            delta_norm = delta_masked / delta_max
        else:
            delta_norm = delta_masked
        threshold = lambda_nn / (2.0 * delta_max) if delta_max > 0 else lambda_nn / 2.0

        # Initialize L = 0
        L = np.zeros((n_periods, n_units))

        for iteration in range(max_iter):
            L_old = L.copy()

            # Step 1: Fix L, solve for (mu, alpha, beta)
            Y_adj = Y_safe - L
            mu, alpha, beta = self._solve_global_no_lowrank(Y_adj, delta_masked)

            # Step 2: Fix (mu, alpha, beta), update L with FISTA acceleration
            R = Y_safe - mu - alpha[np.newaxis, :] - beta[:, np.newaxis]

            # For delta=0 observations (treated/NaN), keep L rather than R
            R_masked = np.where(delta_masked > 0, R, L)

            # Inner FISTA loop for L update
            L_inner = L.copy()
            L_inner_prev = L_inner  # share reference initially (no copy needed)
            t_fista = 1.0

            for _ in range(20):
                # FISTA momentum
                t_fista_new = (1.0 + np.sqrt(1.0 + 4.0 * t_fista**2)) / 2.0
                momentum = (t_fista - 1.0) / t_fista_new
                L_momentum = L_inner + momentum * (L_inner - L_inner_prev)

                # Gradient step from momentum point
                gradient_step = L_momentum + delta_norm * (R_masked - L_momentum)

                # Proximal step: soft-threshold singular values
                L_inner_prev = L_inner
                L_inner = _soft_threshold_svd(gradient_step, threshold)
                t_fista = t_fista_new

                # Convergence check (L_inner_prev holds the pre-SVD value)
                if np.max(np.abs(L_inner - L_inner_prev)) < tol:
                    break

            L = L_inner

            # Outer convergence check
            if np.max(np.abs(L - L_old)) < tol:
                break

        # Final re-solve with converged L (match Rust behavior)
        Y_adj = Y_safe - L
        mu, alpha, beta = self._solve_global_no_lowrank(Y_adj, delta_masked)

        return mu, alpha, beta, L

    def _fit_global(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        resolved_survey=None,
        survey_metadata=None,
        survey_design=None,
    ) -> TROPResults:
        """
        Fit TROP using global weighted least squares method.

        Fits a single model on control observations using (1-W) masked weights,
        then extracts per-observation treatment effects as post-hoc residuals.
        ATT is the mean of these heterogeneous effects.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        outcome : str
            Outcome variable column name.
        treatment : str
            Treatment indicator column name.
        unit : str
            Unit identifier column name.
        time : str
            Time period column name.

        Returns
        -------
        TROPResults
            Estimation results.

        Notes
        -----
        Bootstrap variance estimation assumes simultaneous treatment adoption
        (fixed `treated_periods` across resamples). The treatment timing is
        inferred from the data once and held constant for all bootstrap
        iterations. For staggered adoption designs where treatment timing varies
        across units, use `method="local"` which computes observation-specific
        weights that naturally handle heterogeneous timing.
        """
        # Data setup (same as local method)
        all_units = sorted(data[unit].unique())
        all_periods = sorted(data[time].unique())

        # Extract per-unit survey weights for weighted ATT aggregation
        if resolved_survey is not None:
            from diff_diff.survey import _extract_unit_survey_weights

            unit_weight_arr = _extract_unit_survey_weights(data, unit, survey_design, all_units)
        else:
            unit_weight_arr = None

        n_units = len(all_units)
        n_periods = len(all_periods)

        idx_to_unit = {i: u for i, u in enumerate(all_units)}
        idx_to_period = {i: p for i, p in enumerate(all_periods)}

        # Create matrices
        Y = (
            data.pivot(index=time, columns=unit, values=outcome)
            .reindex(index=all_periods, columns=all_units)
            .values
        )

        D_raw = data.pivot(index=time, columns=unit, values=treatment).reindex(
            index=all_periods, columns=all_units
        )
        missing_mask = pd.isna(D_raw).values
        D = D_raw.fillna(0).astype(int).values

        # Validate absorbing state
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
                f"If this is event-study style data, convert to absorbing state: "
                f"D[t, i] = 1 for all t >= first treatment period."
            )

        # Identify treated observations
        treated_mask = D == 1
        n_treated_obs = np.sum(treated_mask)

        if n_treated_obs == 0:
            raise ValueError("No treated observations found")

        # Identify treated and control units
        unit_ever_treated = np.any(D == 1, axis=0)
        treated_unit_idx = np.where(unit_ever_treated)[0]
        control_unit_idx = np.where(~unit_ever_treated)[0]

        if len(control_unit_idx) == 0:
            raise ValueError("No control units found")

        # Determine pre/post periods
        first_treat_period = None
        for t in range(n_periods):
            if np.any(D[t, :] == 1):
                first_treat_period = t
                break

        if first_treat_period is None:
            raise ValueError("Could not infer post-treatment periods from D matrix")

        n_pre_periods = first_treat_period
        treated_periods = n_periods - first_treat_period
        n_post_periods = int(np.sum(np.any(D[first_treat_period:, :] == 1, axis=1)))

        if n_pre_periods < 2:
            raise ValueError("Need at least 2 pre-treatment periods")

        # Check for staggered adoption (global method requires simultaneous treatment)
        # Use only observed periods (skip missing) to avoid false positives on unbalanced panels
        first_treat_by_unit = []
        for i in treated_unit_idx:
            observed_mask = ~missing_mask[:, i]
            # Get D values for observed periods only
            observed_d = D[observed_mask, i]
            observed_periods = np.where(observed_mask)[0]
            # Find first treatment among observed periods
            treated_idx = np.where(observed_d == 1)[0]
            if len(treated_idx) > 0:
                first_treat_by_unit.append(observed_periods[treated_idx[0]])

        unique_starts = sorted(set(first_treat_by_unit))
        if len(unique_starts) > 1:
            raise ValueError(
                f"method='global' requires simultaneous treatment adoption, but your data "
                f"shows staggered adoption (units first treated at periods {unique_starts}). "
                f"Use method='local' which properly handles staggered adoption designs."
            )

        # LOOCV grid search for tuning parameters
        # Use Rust backend when available for parallel LOOCV (5-10x speedup)
        best_lambda = None
        best_score = np.inf
        control_mask = D == 0

        if HAS_RUST_BACKEND and _rust_loocv_grid_search_global is not None:
            try:
                # Prepare inputs for Rust function
                control_mask_u8 = control_mask.astype(np.uint8)

                lambda_time_arr = np.array(self.lambda_time_grid, dtype=np.float64)
                lambda_unit_arr = np.array(self.lambda_unit_grid, dtype=np.float64)
                lambda_nn_arr = np.array(self.lambda_nn_grid, dtype=np.float64)

                result = _rust_loocv_grid_search_global(
                    Y,
                    D.astype(np.float64),
                    control_mask_u8,
                    lambda_time_arr,
                    lambda_unit_arr,
                    lambda_nn_arr,
                    self.max_iter,
                    self.tol,
                )
                # Unpack result - 7 values including optional first_failed_obs
                best_lt, best_lu, best_ln, best_score, n_valid, n_attempted, first_failed_obs = (
                    result
                )
                # Only accept finite scores - infinite means all fits failed
                if np.isfinite(best_score):
                    best_lambda = (best_lt, best_lu, best_ln)
                # Emit warnings consistent with Python implementation
                if n_valid == 0:
                    obs_info = ""
                    if first_failed_obs is not None:
                        t_idx, i_idx = first_failed_obs
                        obs_info = f" First failure at observation ({t_idx}, {i_idx})."
                    warnings.warn(
                        f"LOOCV: All {n_attempted} fits failed for "
                        f"\u03bb=({best_lt}, {best_lu}, {best_ln}). "
                        f"Returning infinite score.{obs_info}",
                        UserWarning,
                    )
                elif n_attempted > 0 and (n_attempted - n_valid) > 0.1 * n_attempted:
                    n_failed = n_attempted - n_valid
                    obs_info = ""
                    if first_failed_obs is not None:
                        t_idx, i_idx = first_failed_obs
                        obs_info = f" First failure at observation ({t_idx}, {i_idx})."
                    warnings.warn(
                        f"LOOCV: {n_failed}/{n_attempted} fits failed for "
                        f"\u03bb=({best_lt}, {best_lu}, {best_ln}). "
                        f"This may indicate numerical instability.{obs_info}",
                        UserWarning,
                    )
            except Exception as e:
                # Fall back to Python implementation on error
                logger.debug(
                    "Rust LOOCV grid search (global) failed, falling back to Python: %s", e
                )
                best_lambda = None
                best_score = np.inf

        # Fall back to Python implementation if Rust unavailable or failed
        if best_lambda is None:
            # Get control observations for LOOCV
            control_obs = [
                (t, i)
                for t in range(n_periods)
                for i in range(n_units)
                if control_mask[t, i] and not np.isnan(Y[t, i])
            ]

            # Grid search with true LOOCV
            for lambda_time_val in self.lambda_time_grid:
                for lambda_unit_val in self.lambda_unit_grid:
                    for lambda_nn_val in self.lambda_nn_grid:
                        # Convert lambda_nn=inf -> large finite value (factor model disabled)
                        lt = lambda_time_val
                        lu = lambda_unit_val
                        ln = 1e10 if np.isinf(lambda_nn_val) else lambda_nn_val

                        try:
                            score = self._loocv_score_global(
                                Y, D, control_obs, lt, lu, ln, treated_periods, n_units, n_periods
                            )

                            if score < best_score:
                                best_score = score
                                best_lambda = (lambda_time_val, lambda_unit_val, lambda_nn_val)

                        except (np.linalg.LinAlgError, ValueError):
                            continue

        if best_lambda is None:
            warnings.warn("All tuning parameter combinations failed. Using defaults.", UserWarning)
            best_lambda = (1.0, 1.0, 0.1)
            best_score = np.nan

        # Final estimation with best parameters
        lambda_time, lambda_unit, lambda_nn = best_lambda
        original_lambda_nn = lambda_nn

        # Convert lambda_nn=inf -> large finite value (factor model disabled, L~0)
        # lambda_time and lambda_unit use 0.0 for uniform weights directly (no conversion needed)
        if np.isinf(lambda_nn):
            lambda_nn = 1e10

        # Compute final weights and fit
        delta = self._compute_global_weights(
            Y, D, lambda_time, lambda_unit, treated_periods, n_units, n_periods
        )

        mu, alpha, beta, L = self._solve_global_model(Y, delta, lambda_nn)

        # Post-hoc tau extraction (per paper Eq. 2)
        att, treatment_effects, tau_values = self._extract_posthoc_tau(
            Y,
            D,
            mu,
            alpha,
            beta,
            L,
            idx_to_unit,
            idx_to_period,
            unit_weights=unit_weight_arr,
        )

        # Use count of valid (finite) treated outcomes for df and metadata
        n_valid_treated = len(tau_values)
        if n_valid_treated == 0:
            warnings.warn(
                "All treated outcomes are NaN/missing. Cannot estimate ATT.",
                UserWarning,
            )
        elif n_valid_treated < n_treated_obs:
            warnings.warn(
                f"Only {n_valid_treated} of {n_treated_obs} treated outcomes are finite. "
                "df and n_treated_obs reflect valid observations only.",
                UserWarning,
            )

        # Compute effective rank of L
        _, s, _ = np.linalg.svd(L, full_matrices=False)
        if s[0] > 0:
            effective_rank = np.sum(s) / s[0]
        else:
            effective_rank = 0.0

        # Bootstrap variance estimation
        effective_lambda = (lambda_time, lambda_unit, lambda_nn)

        se, bootstrap_dist = self._bootstrap_variance_global(
            data,
            outcome,
            treatment,
            unit,
            time,
            effective_lambda,
            treated_periods,
            survey_design=survey_design,
            unit_weight_arr=unit_weight_arr,
            resolved_survey=resolved_survey,
        )

        # Compute test statistics
        df_trop = max(1, n_valid_treated - 1)
        t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_trop)

        # Create results dictionaries
        unit_effects_dict = {idx_to_unit[i]: alpha[i] for i in range(n_units)}
        time_effects_dict = {idx_to_period[t]: beta[t] for t in range(n_periods)}

        self.results_ = TROPResults(
            att=float(att),
            se=float(se),
            t_stat=float(t_stat) if np.isfinite(t_stat) else t_stat,
            p_value=float(p_value) if np.isfinite(p_value) else p_value,
            conf_int=conf_int,
            n_obs=len(data),
            n_treated=len(treated_unit_idx),
            n_control=len(control_unit_idx),
            n_treated_obs=int(n_valid_treated),
            unit_effects=unit_effects_dict,
            time_effects=time_effects_dict,
            treatment_effects=treatment_effects,
            lambda_time=lambda_time,
            lambda_unit=lambda_unit,
            lambda_nn=original_lambda_nn,
            factor_matrix=L,
            effective_rank=effective_rank,
            loocv_score=best_score,
            alpha=self.alpha,
            n_pre_periods=n_pre_periods,
            n_post_periods=n_post_periods,
            n_bootstrap=self.n_bootstrap,
            bootstrap_distribution=bootstrap_dist if len(bootstrap_dist) > 0 else None,
            survey_metadata=survey_metadata,
        )

        self.is_fitted_ = True
        return self.results_

    def _bootstrap_variance_global(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        optimal_lambda: Tuple[float, float, float],
        treated_periods: int,
        survey_design=None,
        unit_weight_arr: Optional[np.ndarray] = None,
        resolved_survey=None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute bootstrap standard error for global method.

        Uses Rust backend when available for parallel bootstrap (5-15x speedup).
        When a full survey design (strata/PSU/FPC) is present, uses Rao-Wu
        rescaled bootstrap instead, which skips the Rust path.

        Parameters
        ----------
        data : pd.DataFrame
            Original data.
        outcome : str
            Outcome column name.
        treatment : str
            Treatment column name.
        unit : str
            Unit column name.
        time : str
            Time column name.
        optimal_lambda : tuple
            Optimal tuning parameters.
        treated_periods : int
            Number of post-treatment periods.
        survey_design : SurveyDesign, optional
            Survey design specification.
        unit_weight_arr : np.ndarray, optional
            Unit-level survey weights.
        resolved_survey : ResolvedSurveyDesign, optional
            Resolved survey design (observation-level).

        Returns
        -------
        Tuple[float, np.ndarray]
            (se, bootstrap_estimates).
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
            return self._bootstrap_rao_wu_global(
                data,
                outcome,
                treatment,
                unit,
                time,
                optimal_lambda,
                treated_periods,
                resolved_survey,
                survey_design,
            )

        # Try Rust backend for parallel bootstrap (5-15x speedup)
        # Only used for pweight-only designs (no strata/PSU/FPC)
        if HAS_RUST_BACKEND and _rust_bootstrap_trop_variance_global is not None:
            try:
                # Create matrices for Rust function
                all_units = sorted(data[unit].unique())
                all_periods = sorted(data[time].unique())

                Y = (
                    data.pivot(index=time, columns=unit, values=outcome)
                    .reindex(index=all_periods, columns=all_units)
                    .values
                )
                D = (
                    data.pivot(index=time, columns=unit, values=treatment)
                    .reindex(index=all_periods, columns=all_units)
                    .fillna(0)
                    .astype(np.float64)
                    .values
                )

                bootstrap_estimates, se = _rust_bootstrap_trop_variance_global(
                    Y,
                    D,
                    lambda_time,
                    lambda_unit,
                    lambda_nn,
                    self.n_bootstrap,
                    self.max_iter,
                    self.tol,
                    self.seed if self.seed is not None else 0,
                    unit_weight_arr,
                )

                if len(bootstrap_estimates) < 10:
                    warnings.warn(
                        f"Only {len(bootstrap_estimates)} bootstrap iterations succeeded.",
                        UserWarning,
                    )
                    if len(bootstrap_estimates) == 0:
                        return np.nan, np.array([])

                return float(se), np.array(bootstrap_estimates)

            except Exception as e:
                logger.debug("Rust bootstrap (global) failed, falling back to Python: %s", e)

        # Python fallback implementation
        rng = np.random.default_rng(self.seed)

        # Stratified bootstrap sampling
        unit_ever_treated = data.groupby(unit)[treatment].max()
        treated_units = np.array(unit_ever_treated[unit_ever_treated == 1].index.tolist())
        control_units = np.array(unit_ever_treated[unit_ever_treated == 0].index.tolist())

        n_treated_units = len(treated_units)
        n_control_units = len(control_units)

        bootstrap_estimates_list: List[float] = []

        for _ in range(self.n_bootstrap):
            # Stratified sampling
            if n_control_units > 0:
                sampled_control = rng.choice(control_units, size=n_control_units, replace=True)
            else:
                sampled_control = np.array([], dtype=object)

            if n_treated_units > 0:
                sampled_treated = rng.choice(treated_units, size=n_treated_units, replace=True)
            else:
                sampled_treated = np.array([], dtype=object)

            sampled_units = np.concatenate([sampled_control, sampled_treated])

            # Create bootstrap sample
            boot_data = pd.concat(
                [
                    data[data[unit] == u].assign(**{unit: f"{u}_{idx}"})
                    for idx, u in enumerate(sampled_units)
                ],
                ignore_index=True,
            )

            try:
                tau = self._fit_global_with_fixed_lambda(
                    boot_data,
                    outcome,
                    treatment,
                    unit,
                    time,
                    optimal_lambda,
                    treated_periods,
                    survey_design=survey_design,
                )
                if np.isfinite(tau):
                    bootstrap_estimates_list.append(tau)
            except (ValueError, np.linalg.LinAlgError, KeyError):
                continue

        bootstrap_estimates = np.array(bootstrap_estimates_list)

        if len(bootstrap_estimates) < 10:
            warnings.warn(
                f"Only {len(bootstrap_estimates)} bootstrap iterations succeeded.", UserWarning
            )
            if len(bootstrap_estimates) == 0:
                return np.nan, np.array([])

        se = np.std(bootstrap_estimates, ddof=1)
        return float(se), bootstrap_estimates

    def _bootstrap_rao_wu_global(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        optimal_lambda: Tuple[float, float, float],
        treated_periods: int,
        resolved_survey,
        survey_design,
    ) -> Tuple[float, np.ndarray]:
        """
        Rao-Wu rescaled bootstrap for global method with full survey design.

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
        treated_periods : int
            Number of post-treatment periods.
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
        from diff_diff.survey import ResolvedSurveyDesign

        lambda_time, lambda_unit, lambda_nn = optimal_lambda
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
        from diff_diff.linalg import _factorize_cluster_ids

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

        # Bootstrap loop with Rao-Wu rescaled weights
        all_periods = sorted(data[time].unique())
        n_periods = len(all_periods)

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

        bootstrap_estimates_list: List[float] = []

        for _ in range(self.n_bootstrap):
            try:
                # Generate Rao-Wu rescaled weights (unit-level)
                boot_weights = generate_rao_wu_weights(unit_resolved, rng)

                # Skip if all control or all treated weights are zero
                control_mask_units = treatment_group == 0
                treated_mask_units = treatment_group == 1
                if boot_weights[control_mask_units].sum() == 0:
                    continue
                if boot_weights[treated_mask_units].sum() == 0:
                    continue

                # Compute global weights and fit model
                delta = self._compute_global_weights(
                    Y, D, lambda_time, lambda_unit, treated_periods, n_units, n_periods
                )
                mu, alpha, beta, L = self._solve_global_model(Y, delta, lambda_nn)

                # Extract weighted ATT using Rao-Wu rescaled weights
                att, _, _ = self._extract_posthoc_tau(
                    Y, D, mu, alpha, beta, L, unit_weights=boot_weights
                )

                if np.isfinite(att):
                    bootstrap_estimates_list.append(att)
            except (ValueError, np.linalg.LinAlgError, KeyError):
                continue

        bootstrap_estimates = np.array(bootstrap_estimates_list)

        if len(bootstrap_estimates) < 10:
            warnings.warn(
                f"Only {len(bootstrap_estimates)} bootstrap iterations succeeded.",
                UserWarning,
            )
            if len(bootstrap_estimates) == 0:
                return np.nan, np.array([])

        se = np.std(bootstrap_estimates, ddof=1)
        return float(se), bootstrap_estimates

    def _fit_global_with_fixed_lambda(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        fixed_lambda: Tuple[float, float, float],
        treated_periods: int,
        survey_design=None,
    ) -> float:
        """
        Fit global model with fixed tuning parameters.

        Returns the ATT (mean of post-hoc per-observation treatment effects).
        """
        lambda_time, lambda_unit, lambda_nn = fixed_lambda

        all_units = sorted(data[unit].unique())
        all_periods = sorted(data[time].unique())

        # Extract per-unit survey weights for weighted ATT in bootstrap
        if survey_design is not None and survey_design.weights is not None:
            from diff_diff.survey import _extract_unit_survey_weights

            local_weight_arr = _extract_unit_survey_weights(data, unit, survey_design, all_units)
        else:
            local_weight_arr = None

        n_units = len(all_units)
        n_periods = len(all_periods)

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

        # Compute weights (includes (1-W) masking)
        delta = self._compute_global_weights(
            Y, D, lambda_time, lambda_unit, treated_periods, n_units, n_periods
        )

        # Fit model on control data and extract post-hoc tau
        mu, alpha, beta, L = self._solve_global_model(Y, delta, lambda_nn)
        att, _, _ = self._extract_posthoc_tau(
            Y, D, mu, alpha, beta, L, unit_weights=local_weight_arr
        )
        return att
