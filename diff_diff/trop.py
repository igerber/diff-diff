"""
Triply Robust Panel (TROP) estimator.

Implements the TROP estimator from Athey, Imbens, Qu & Viviano (2025).
TROP combines three robustness components:
1. Nuclear norm regularized factor model (interactive fixed effects)
2. Exponential distance-based unit weights
3. Exponential time decay weights

The estimator uses leave-one-out cross-validation for tuning parameter
selection and provides robust treatment effect estimates under factor
confounding.

References
----------
Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). Triply Robust Panel
Estimators. *Working Paper*. https://arxiv.org/abs/2508.21536
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_loocv_grid_search,
)
from diff_diff.trop_global import TROPGlobalMixin
from diff_diff.trop_local import TROPLocalMixin
from diff_diff.trop_results import (
    _LAMBDA_INF,
    _PrecomputedStructures,
    TROPResults,
)
from diff_diff.utils import safe_inference


class TROP(TROPLocalMixin, TROPGlobalMixin):
    """
    Triply Robust Panel (TROP) estimator.

    Implements the exact methodology from Athey, Imbens, Qu & Viviano (2025).
    TROP combines three robustness components:

    1. **Nuclear norm regularized factor model**: Estimates interactive fixed
       effects L_it via matrix completion with nuclear norm penalty ||L||_*

    2. **Exponential distance-based unit weights**: ω_j = exp(-λ_unit × d(j,i))
       where d(j,i) is the RMSE of outcome differences between units

    3. **Exponential time decay weights**: θ_s = exp(-λ_time × |s-t|)
       weighting pre-treatment periods by proximity to treatment

    Tuning parameters (λ_time, λ_unit, λ_nn) are selected via leave-one-out
    cross-validation on control observations.

    Parameters
    ----------
    method : str, default='local'
        Estimation method to use:

        - 'local': Per-observation model fitting following Algorithm 2 of
          Athey et al. (2025). Computes observation-specific weights and fits
          a model for each treated observation, averaging the individual
          treatment effects. More flexible but computationally intensive.

        - 'global': Computationally efficient adaptation using the (1-W)
          masking principle from Eq. 2. Fits a single model on control
          observations with global weights, then computes per-observation
          treatment effects as residuals:
          tau_it = Y_it - mu - alpha_i - beta_t - L_it for treated cells.
          ATT is the mean of these effects. For the paper's full
          per-treated-cell estimator, use ``method='local'``.

        - 'twostep': Deprecated alias for 'local'. Will be removed in v3.0.

        - 'joint': Deprecated alias for 'global'. Will be removed in v3.0.

    lambda_time_grid : list, optional
        Grid of time weight decay parameters. 0.0 = uniform weights (disabled).
        Must not contain inf. Default: [0, 0.1, 0.5, 1, 2, 5].
    lambda_unit_grid : list, optional
        Grid of unit weight decay parameters. 0.0 = uniform weights (disabled).
        Must not contain inf. Default: [0, 0.1, 0.5, 1, 2, 5].
    lambda_nn_grid : list, optional
        Grid of nuclear norm regularization parameters. inf = factor model
        disabled (L=0). Default: [0, 0.01, 0.1, 1].
    max_iter : int, default=100
        Maximum iterations for nuclear norm optimization.
    tol : float, default=1e-6
        Convergence tolerance for optimization.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default=200
        Number of bootstrap replications for variance estimation. Must be >= 2.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    results_ : TROPResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> from diff_diff import TROP
    >>> trop = TROP()
    >>> results = trop.fit(
    ...     data,
    ...     outcome='outcome',
    ...     treatment='treated',
    ...     unit='unit',
    ...     time='period',
    ... )
    >>> results.print_summary()

    References
    ----------
    Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). Triply Robust
    Panel Estimators. *Working Paper*. https://arxiv.org/abs/2508.21536
    """

    def __init__(
        self,
        method: str = "local",
        lambda_time_grid: Optional[List[float]] = None,
        lambda_unit_grid: Optional[List[float]] = None,
        lambda_nn_grid: Optional[List[float]] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 0.05,
        n_bootstrap: int = 200,
        seed: Optional[int] = None,
    ):
        # Validate method parameter
        # 'local'/'global' are preferred; 'twostep'/'joint' are deprecated aliases
        valid_methods = ("local", "twostep", "joint", "global")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        if method == "twostep":
            warnings.warn(
                "method='twostep' is deprecated and will be removed in v3.0. "
                "Use method='local' instead.",
                FutureWarning,
                stacklevel=2,
            )
            method = "local"
        if method == "joint":
            warnings.warn(
                "method='joint' is deprecated and will be removed in v3.0. "
                "Use method='global' instead.",
                FutureWarning,
                stacklevel=2,
            )
            method = "global"
        self.method = method

        # Default grids from paper
        self.lambda_time_grid = lambda_time_grid or [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_unit_grid = lambda_unit_grid or [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_nn_grid = lambda_nn_grid or [0.0, 0.01, 0.1, 1.0, 10.0]

        if n_bootstrap < 2:
            raise ValueError(
                "n_bootstrap must be >= 2 for TROP (bootstrap variance "
                "estimation is always used)"
            )

        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        # Validate that time/unit grids do not contain inf.
        # Per Athey et al. (2025) Eq. 3, λ_time=0 and λ_unit=0 give uniform
        # weights (exp(-0 × dist) = 1). Using inf is a misunderstanding of
        # the paper's convention. Only λ_nn=∞ is valid (disables factor model).
        for grid_name, grid_vals in [
            ("lambda_time_grid", self.lambda_time_grid),
            ("lambda_unit_grid", self.lambda_unit_grid),
        ]:
            if any(np.isinf(v) for v in grid_vals):
                raise ValueError(
                    f"{grid_name} must not contain inf. Use 0.0 for uniform "
                    f"weights (disabled) per Athey et al. (2025) Eq. 3: "
                    f"exp(-0 × dist) = 1 for all distances."
                )

        # Internal state
        self.results_: Optional[TROPResults] = None
        self.is_fitted_: bool = False
        self._optimal_lambda: Optional[Tuple[float, float, float]] = None

        # Pre-computed structures (set during fit)
        self._precomputed: Optional[_PrecomputedStructures] = None

    # =========================================================================
    # Parameter search (used by local method's fit() path)
    # =========================================================================

    def _univariate_loocv_search(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        control_mask: np.ndarray,
        control_unit_idx: np.ndarray,
        n_units: int,
        n_periods: int,
        param_name: str,
        grid: List[float],
        fixed_params: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Search over one parameter with others fixed.

        Following paper's footnote 2, this performs a univariate grid search
        for one tuning parameter while holding others fixed. The fixed_params
        use 0.0 for disabled time/unit weights and _LAMBDA_INF for disabled
        factor model:
        - lambda_nn = inf: Skip nuclear norm regularization (L=0)
        - lambda_time = 0.0: Uniform time weights (exp(-0×dist)=1)
        - lambda_unit = 0.0: Uniform unit weights (exp(-0×dist)=1)

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
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.
        param_name : str
            Name of parameter to search: 'lambda_time', 'lambda_unit', or 'lambda_nn'.
        grid : List[float]
            Grid of values to search over.
        fixed_params : Dict[str, float]
            Fixed values for other parameters. May include _LAMBDA_INF for lambda_nn.

        Returns
        -------
        Tuple[float, float]
            (best_value, best_score) for the searched parameter.
        """
        best_score = np.inf
        best_value = grid[0] if grid else 0.0

        for value in grid:
            params = {**fixed_params, param_name: value}

            lambda_time = params.get("lambda_time", 0.0)
            lambda_unit = params.get("lambda_unit", 0.0)
            lambda_nn = params.get("lambda_nn", 0.0)

            # Convert λ_nn=∞ → large finite value (factor model disabled, L≈0)
            # λ_time and λ_unit use 0.0 for uniform weights per Eq. 3 (no inf conversion needed)
            if np.isinf(lambda_nn):
                lambda_nn = 1e10

            try:
                score = self._loocv_score_obs_specific(
                    Y,
                    D,
                    control_mask,
                    control_unit_idx,
                    lambda_time,
                    lambda_unit,
                    lambda_nn,
                    n_units,
                    n_periods,
                )
                if score < best_score:
                    best_score = score
                    best_value = value
            except (np.linalg.LinAlgError, ValueError):
                continue

        return best_value, best_score

    def _cycling_parameter_search(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        control_mask: np.ndarray,
        control_unit_idx: np.ndarray,
        n_units: int,
        n_periods: int,
        initial_lambda: Tuple[float, float, float],
        max_cycles: int = 10,
    ) -> Tuple[float, float, float]:
        """
        Cycle through parameters until convergence (coordinate descent).

        Following paper's footnote 2 (Stage 2), this iteratively optimizes
        each tuning parameter while holding the others fixed, until convergence.

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
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.
        initial_lambda : Tuple[float, float, float]
            Initial values (lambda_time, lambda_unit, lambda_nn).
        max_cycles : int, default=10
            Maximum number of coordinate descent cycles.

        Returns
        -------
        Tuple[float, float, float]
            Optimized (lambda_time, lambda_unit, lambda_nn).
        """
        lambda_time, lambda_unit, lambda_nn = initial_lambda
        prev_score = np.inf

        for cycle in range(max_cycles):
            # Optimize λ_unit (fix λ_time, λ_nn)
            lambda_unit, _ = self._univariate_loocv_search(
                Y,
                D,
                control_mask,
                control_unit_idx,
                n_units,
                n_periods,
                "lambda_unit",
                self.lambda_unit_grid,
                {"lambda_time": lambda_time, "lambda_nn": lambda_nn},
            )

            # Optimize λ_time (fix λ_unit, λ_nn)
            lambda_time, _ = self._univariate_loocv_search(
                Y,
                D,
                control_mask,
                control_unit_idx,
                n_units,
                n_periods,
                "lambda_time",
                self.lambda_time_grid,
                {"lambda_unit": lambda_unit, "lambda_nn": lambda_nn},
            )

            # Optimize λ_nn (fix λ_unit, λ_time)
            lambda_nn, score = self._univariate_loocv_search(
                Y,
                D,
                control_mask,
                control_unit_idx,
                n_units,
                n_periods,
                "lambda_nn",
                self.lambda_nn_grid,
                {"lambda_unit": lambda_unit, "lambda_time": lambda_time},
            )

            # Check convergence
            if abs(score - prev_score) < 1e-6:
                logger.debug(
                    "Cycling search converged after %d cycles with score %.6f", cycle + 1, score
                )
                break
            prev_score = score

        return lambda_time, lambda_unit, lambda_nn

    # =========================================================================
    # Main fit method
    # =========================================================================

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        survey_design=None,
    ) -> TROPResults:
        """
        Fit the TROP model.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with observations for multiple units over multiple
            time periods.
        outcome : str
            Name of the outcome variable column.
        treatment : str
            Name of the treatment indicator column (0/1).

            IMPORTANT: This should be an ABSORBING STATE indicator, not a
            treatment timing indicator. For each unit, D=1 for ALL periods
            during and after treatment:

            - D[t, i] = 0 for all t < g_i (pre-treatment periods)
            - D[t, i] = 1 for all t >= g_i (treatment and post-treatment)

            where g_i is the treatment start time for unit i.

            For staggered adoption, different units can have different g_i.
            The ATT averages over ALL D=1 cells per Equation 1 of the paper.
        unit : str
            Name of the unit identifier column.
        time : str
            Name of the time period column.
        survey_design : SurveyDesign, optional
            Survey design specification. Supports pweight, strata, PSU, and
            FPC. Full-design surveys (strata/PSU/FPC) use Rao-Wu rescaled
            bootstrap; Rust backend is pweight-only (Python fallback for
            full design). Survey weights enter ATT aggregation only.

        Returns
        -------
        TROPResults
            Object containing the ATT estimate, standard error,
            factor estimates, and tuning parameters. The lambda_*
            attributes show the selected grid values. For lambda_time and
            lambda_unit, 0.0 means uniform weights; inf is not accepted.
            For lambda_nn, inf is converted to 1e10 (factor model disabled).

        Raises
        ------
        ValueError
            If required columns are missing or non-pweight survey design.
        """
        # Validate inputs
        required_cols = [outcome, treatment, unit, time]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Resolve survey design
        from diff_diff.survey import (
            _extract_unit_survey_weights,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, _survey_weights, _survey_wt, survey_metadata = _resolve_survey_for_fit(
            survey_design, data, "analytical"
        )
        # Reject replicate-weight designs — TROP uses Rao-Wu bootstrap
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "TROP does not yet support replicate-weight survey designs. "
                "Use a TSL-based survey design (strata/psu/fpc)."
            )
        # Validate weight_type is pweight (keep restriction), but allow
        # strata/PSU/FPC — those are handled via Rao-Wu rescaled bootstrap.
        if resolved_survey is not None and resolved_survey.weight_type != "pweight":
            raise ValueError(
                "TROP requires pweight survey weights. "
                f"Got weight_type='{resolved_survey.weight_type}'."
            )
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)

        # Dispatch based on estimation method
        if self.method == "global":
            return self._fit_global(
                data,
                outcome,
                treatment,
                unit,
                time,
                resolved_survey=resolved_survey,
                survey_metadata=survey_metadata,
                survey_design=survey_design,
            )

        # Below is the local method (default)
        # Get unique units and periods
        all_units = sorted(data[unit].unique())

        # Extract unit-level survey weights
        if resolved_survey is not None:
            unit_weight_arr = _extract_unit_survey_weights(data, unit, survey_design, all_units)
        else:
            unit_weight_arr = None
        all_periods = sorted(data[time].unique())

        n_units = len(all_units)
        n_periods = len(all_periods)

        # Create mappings
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        period_to_idx = {p: i for i, p in enumerate(all_periods)}
        idx_to_unit = {i: u for u, i in unit_to_idx.items()}
        idx_to_period = {i: p for p, i in period_to_idx.items()}

        # Create outcome matrix Y (n_periods x n_units) and treatment matrix D
        # Vectorized: use pivot for O(1) reshaping instead of O(n) iterrows loop
        Y = (
            data.pivot(index=time, columns=unit, values=outcome)
            .reindex(index=all_periods, columns=all_units)
            .values
        )

        # For D matrix, track missing values BEFORE fillna to support unbalanced panels
        # Issue 3 fix: Missing observations should not trigger spurious violations
        D_raw = data.pivot(index=time, columns=unit, values=treatment).reindex(
            index=all_periods, columns=all_units
        )
        missing_mask = pd.isna(D_raw).values  # True where originally missing
        D = D_raw.fillna(0).astype(int).values

        # Validate D is monotonic non-decreasing per unit (absorbing state)
        # D[t, i] must satisfy: once D=1, it must stay 1 for all subsequent periods
        # Issue 3 fix (round 10): Check each unit's OBSERVED D sequence for monotonicity
        # This catches 1->0 violations that span missing period gaps
        # Example: D[2]=1, missing [3,4], D[5]=0 is a real violation even though
        # adjacent period transitions don't show it (the gap hides the transition)
        violating_units = []
        for unit_idx in range(n_units):
            # Get observed D values for this unit (where not missing)
            observed_mask = ~missing_mask[:, unit_idx]
            observed_d = D[observed_mask, unit_idx]

            # Check if observed sequence is monotonically non-decreasing
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

        # Determine pre/post periods from treatment indicator D
        # D matrix is the sole input for treatment timing per the paper
        first_treat_period = None
        for t in range(n_periods):
            if np.any(D[t, :] == 1):
                first_treat_period = t
                break
        if first_treat_period is None:
            raise ValueError("Could not infer post-treatment periods from D matrix")

        n_pre_periods = first_treat_period
        # Count periods where D=1 is actually observed (matches docstring)
        # Per docstring: "Number of post-treatment periods (periods with D=1 observations)"
        n_post_periods = int(np.sum(np.any(D[first_treat_period:, :] == 1, axis=1)))

        if n_pre_periods < 2:
            raise ValueError("Need at least 2 pre-treatment periods")

        # Step 1: Grid search with LOOCV for tuning parameters
        best_lambda = None
        best_score = np.inf

        # Control observations mask (for LOOCV)
        control_mask = D == 0

        # Pre-compute structures that are reused across LOOCV iterations
        self._precomputed = self._precompute_structures(Y, D, control_unit_idx, n_units, n_periods)

        # Use Rust backend for parallel LOOCV grid search (10-50x speedup)
        if HAS_RUST_BACKEND and _rust_loocv_grid_search is not None:
            try:
                # Prepare inputs for Rust function
                control_mask_u8 = control_mask.astype(np.uint8)
                time_dist_matrix = self._precomputed["time_dist_matrix"].astype(np.int64)

                lambda_time_arr = np.array(self.lambda_time_grid, dtype=np.float64)
                lambda_unit_arr = np.array(self.lambda_unit_grid, dtype=np.float64)
                lambda_nn_arr = np.array(self.lambda_nn_grid, dtype=np.float64)

                result = _rust_loocv_grid_search(
                    Y,
                    D.astype(np.float64),
                    control_mask_u8,
                    time_dist_matrix,
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
                # else: best_lambda stays None, triggering defaults fallback
                # Emit warnings consistent with Python implementation
                if n_valid == 0:
                    # Include failed observation coordinates if available (Issue 2 fix)
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
                    # Include failed observation coordinates if available
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
                logger.debug("Rust LOOCV grid search failed, falling back to Python: %s", e)
                best_lambda = None
                best_score = np.inf

        # Fall back to Python implementation if Rust unavailable or failed
        # Uses two-stage approach per paper's footnote 2:
        # Stage 1: Univariate searches for initial values
        # Stage 2: Cycling (coordinate descent) until convergence
        if best_lambda is None:
            # Stage 1: Univariate searches with extreme fixed values
            # Following paper's footnote 2 for initial bounds

            # λ_time search: fix λ_unit=0, λ_nn=∞ (disabled - no factor adjustment)
            lambda_time_init, _ = self._univariate_loocv_search(
                Y,
                D,
                control_mask,
                control_unit_idx,
                n_units,
                n_periods,
                "lambda_time",
                self.lambda_time_grid,
                {"lambda_unit": 0.0, "lambda_nn": _LAMBDA_INF},
            )

            # λ_nn search: fix λ_time=0 (uniform time weights), λ_unit=0
            lambda_nn_init, _ = self._univariate_loocv_search(
                Y,
                D,
                control_mask,
                control_unit_idx,
                n_units,
                n_periods,
                "lambda_nn",
                self.lambda_nn_grid,
                {"lambda_time": 0.0, "lambda_unit": 0.0},
            )

            # λ_unit search: fix λ_nn=∞, λ_time=0
            lambda_unit_init, _ = self._univariate_loocv_search(
                Y,
                D,
                control_mask,
                control_unit_idx,
                n_units,
                n_periods,
                "lambda_unit",
                self.lambda_unit_grid,
                {"lambda_nn": _LAMBDA_INF, "lambda_time": 0.0},
            )

            # Stage 2: Cycling refinement (coordinate descent)
            lambda_time, lambda_unit, lambda_nn = self._cycling_parameter_search(
                Y,
                D,
                control_mask,
                control_unit_idx,
                n_units,
                n_periods,
                (lambda_time_init, lambda_unit_init, lambda_nn_init),
            )

            # Compute final score for the optimized parameters
            try:
                best_score = self._loocv_score_obs_specific(
                    Y,
                    D,
                    control_mask,
                    control_unit_idx,
                    lambda_time,
                    lambda_unit,
                    lambda_nn,
                    n_units,
                    n_periods,
                )
                # Only accept finite scores - infinite means all fits failed
                if np.isfinite(best_score):
                    best_lambda = (lambda_time, lambda_unit, lambda_nn)
                # else: best_lambda stays None, triggering defaults fallback
            except (np.linalg.LinAlgError, ValueError):
                # If even the optimized parameters fail, best_lambda stays None
                pass

        if best_lambda is None:
            warnings.warn("All tuning parameter combinations failed. Using defaults.", UserWarning)
            best_lambda = (1.0, 1.0, 0.1)
            best_score = np.nan

        self._optimal_lambda = best_lambda
        lambda_time, lambda_unit, lambda_nn = best_lambda

        # Store original λ_nn for results (only λ_nn needs original→effective conversion).
        # λ_time and λ_unit use 0.0 for uniform weights directly per Eq. 3.
        original_lambda_nn = lambda_nn

        # Convert λ_nn=∞ → large finite value (factor model disabled, L≈0)
        if np.isinf(lambda_nn):
            lambda_nn = 1e10

        # effective_lambda with converted λ_nn for ALL downstream computation
        # (variance estimation uses the same parameters as point estimation)
        effective_lambda = (lambda_time, lambda_unit, lambda_nn)

        # Step 2: Final estimation - per-observation model fitting following Algorithm 2
        # For each treated (i,t): compute observation-specific weights, fit model, compute tau_{it}
        treatment_effects = {}
        tau_values = []
        tau_weights = []  # parallel to tau_values for survey-weighted ATT
        alpha_estimates = []
        beta_estimates = []
        L_estimates = []

        # Use pre-computed treated observations
        treated_observations = self._precomputed["treated_observations"]

        for t, i in treated_observations:
            unit_id = idx_to_unit[i]
            time_id = idx_to_period[t]

            # Skip observations where outcome is missing -- record NaN but
            # don't fit the model or include in tau_values (avoids NaN poisoning)
            if not np.isfinite(Y[t, i]):
                treatment_effects[(unit_id, time_id)] = np.nan
                continue

            # Compute observation-specific weights for this (i, t)
            weight_matrix = self._compute_observation_weights(
                Y, D, i, t, lambda_time, lambda_unit, control_unit_idx, n_units, n_periods
            )

            # Fit model with these weights
            alpha_hat, beta_hat, L_hat = self._estimate_model(
                Y, control_mask, weight_matrix, lambda_nn, n_units, n_periods
            )

            # Compute treatment effect: tau_{it} = Y_{it} - alpha_i - beta_t - L_{it}
            tau_it = Y[t, i] - alpha_hat[i] - beta_hat[t] - L_hat[t, i]

            treatment_effects[(unit_id, time_id)] = tau_it
            tau_values.append(tau_it)
            if unit_weight_arr is not None:
                tau_weights.append(unit_weight_arr[i])

            # Store for averaging
            alpha_estimates.append(alpha_hat)
            beta_estimates.append(beta_hat)
            L_estimates.append(L_hat)

        # Count valid treated observations
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

        # Average ATT (survey-weighted when applicable)
        if unit_weight_arr is not None and tau_values:
            att = float(np.average(tau_values, weights=tau_weights))
        else:
            att = np.mean(tau_values) if tau_values else np.nan

        # Average parameter estimates for output (representative)
        alpha_hat = np.mean(alpha_estimates, axis=0) if alpha_estimates else np.zeros(n_units)
        beta_hat = np.mean(beta_estimates, axis=0) if beta_estimates else np.zeros(n_periods)
        L_hat = np.mean(L_estimates, axis=0) if L_estimates else np.zeros((n_periods, n_units))

        # Compute effective rank
        _, s, _ = np.linalg.svd(L_hat, full_matrices=False)
        if s[0] > 0:
            effective_rank = np.sum(s) / s[0]
        else:
            effective_rank = 0.0

        # Step 4: Variance estimation
        # Use effective_lambda (converted values) to ensure SE is computed with same
        # parameters as point estimation. This fixes the variance inconsistency issue.
        se, bootstrap_dist = self._bootstrap_variance(
            data,
            outcome,
            treatment,
            unit,
            time,
            effective_lambda,
            Y=Y,
            D=D,
            control_unit_idx=control_unit_idx,
            survey_design=survey_design,
            unit_weight_arr=unit_weight_arr,
            resolved_survey=resolved_survey,
        )

        # Compute test statistics
        df_trop = max(1, n_valid_treated - 1)
        t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_trop)

        # Create results dictionaries
        unit_effects_dict = {idx_to_unit[i]: alpha_hat[i] for i in range(n_units)}
        time_effects_dict = {idx_to_period[t]: beta_hat[t] for t in range(n_periods)}

        # Store results
        self.results_ = TROPResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
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
            factor_matrix=L_hat,
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

    # =========================================================================
    # sklearn-like API
    # =========================================================================

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters."""
        return {
            "method": self.method,
            "lambda_time_grid": self.lambda_time_grid,
            "lambda_unit_grid": self.lambda_unit_grid,
            "lambda_nn_grid": self.lambda_nn_grid,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "TROP":
        """Set estimator parameters."""
        for key, value in params.items():
            if key == "method" and value == "twostep":
                warnings.warn(
                    "method='twostep' is deprecated and will be removed in "
                    "v3.0. Use method='local' instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                value = "local"
            if key == "method" and value == "joint":
                warnings.warn(
                    "method='joint' is deprecated and will be removed in "
                    "v3.0. Use method='global' instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                value = "global"
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self


def trop(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit: str,
    time: str,
    survey_design=None,
    **kwargs,
) -> TROPResults:
    """
    Convenience function for TROP estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column name.
    treatment : str
        Treatment indicator column name (0/1).

        IMPORTANT: This should be an ABSORBING STATE indicator, not a treatment
        timing indicator. For each unit, D=1 for ALL periods during and after
        treatment (D[t,i]=0 for t < g_i, D[t,i]=1 for t >= g_i where g_i is
        the treatment start time for unit i).
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    survey_design : SurveyDesign, optional
        Survey design specification. Supports pweight, strata, PSU, and FPC.
    **kwargs
        Additional arguments passed to TROP constructor.

    Returns
    -------
    TROPResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import trop
    >>> results = trop(data, 'y', 'treated', 'unit', 'time')
    >>> print(f"ATT: {results.att:.3f}")
    """
    estimator = TROP(**kwargs)
    return estimator.fit(data, outcome, treatment, unit, time, survey_design=survey_design)
