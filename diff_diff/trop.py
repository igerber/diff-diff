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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_loocv_grid_search,
)
from diff_diff._base import BaseEstimator
from diff_diff.trop_global import TROPGlobalMixin
from diff_diff.trop_local import (
    TROPLocalMixin,
    _setup_trop_data,
    _treated_cell_is_estimable,
)
from diff_diff.trop_results import (
    _LAMBDA_INF,
    TROPResults,
    _PrecomputedStructures,
)
from diff_diff.utils import safe_inference, validate_n_bootstrap, warn_if_not_converged


class TROP(TROPLocalMixin, TROPGlobalMixin, BaseEstimator):
    """
    Triply Robust Panel (TROP) estimator.

    Implements the exact methodology from Athey, Imbens, Qu & Viviano (2025).
    TROP combines three robustness components:

    1. **Nuclear norm regularized factor model**: Estimates interactive fixed
       effects L_it via matrix completion with nuclear norm penalty ||L||_*

    2. **Exponential distance-based unit weights**: ω_j = exp(-λ_unit × d(j,i))
       where d(j,i) is the RMSE of outcome differences between units

    3. **Exponential time decay weights**: θ_s = exp(-λ_time × :math:`|s-t|`)
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
    non_absorbing : bool, default=False
        Treatment-assignment scope for the treatment indicator.

        - ``False`` (default): require an ABSORBING STATE indicator (once
          treated, always treated). A non-monotonic indicator raises
          ``ValueError``. This guards against the common mistake of encoding
          absorbing treatment as an event-style spike (a single D=1 period),
          which would silently bias the ATT.
        - ``True``: accept general (on/off) assignment patterns, where treatment
          may switch on and off, per Athey et al. (2025) Eq. 12 / Algorithm 2.
          Supported for ``method='local'`` only (``method='global'`` raises).
          Relies on the paper's no-dynamic-effects (no carryover) assumption; the
          triple-robustness guarantee (Theorem 5.1) is proven only under block
          assignment, so a ``UserWarning`` is emitted on fit. The estimand
          averages the per-cell effects over the **estimable** treated (D=1)
          cells (Eq. 1): a cell is non-estimable (NaN, excluded) when its unit/time
          fixed effect ``alpha_i + beta_t`` is unidentified by the control fit --
          i.e. the target unit and target period are not in the same connected
          component of the observed-control graph (an always-treated unit, a
          fully-treated period, or disconnected control support). This matches the
          library non-estimable->NaN convention (see REGISTRY ## TROP
          "non-absorbing non-estimable-cell trimming").

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
        non_absorbing: bool = False,
    ):
        # Validate method parameter
        valid_methods = ("local", "global")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        self.method = method

        # Validate non_absorbing flag (must be a plain bool, not a truthy value).
        # When False (default) TROP requires an absorbing-state treatment indicator;
        # when True it accepts general (on/off) assignment patterns per Athey et al.
        # (2025) Eq. 12 / Algorithm 2 -- local method only (see fit()).
        if not isinstance(non_absorbing, bool):
            raise ValueError(f"non_absorbing must be a bool, got {type(non_absorbing).__name__}")
        self.non_absorbing = non_absorbing

        # Default grids from paper
        self.lambda_time_grid = lambda_time_grid or [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_unit_grid = lambda_unit_grid or [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_nn_grid = lambda_nn_grid or [0.0, 0.01, 0.1, 1.0, 10.0]

        # Shared type guard first (rejects bool/float), then the
        # TROP-specific floor.
        validate_n_bootstrap(n_bootstrap)
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

            By default (``non_absorbing=False``) this must be an ABSORBING STATE
            indicator, not a treatment timing indicator. For each unit, D=1 for
            ALL periods during and after treatment:

            - D[t, i] = 0 for all t < g_i (pre-treatment periods)
            - D[t, i] = 1 for all t >= g_i (treatment and post-treatment)

            where g_i is the treatment start time for unit i.

            For staggered adoption, different units can have different g_i (this
            is still absorbing). Set ``non_absorbing=True`` to allow treatment to
            switch on and off (general assignment, ``method='local'`` only). The
            ATT averages over the **estimable** D=1 cells per Equation 1 (a cell
            whose unit/time fixed effect is unidentified by the control fit is
            NaN and excluded; see ``non_absorbing`` and ``TROPResults``).
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
        _ctx = _setup_trop_data(
            data,
            outcome,
            treatment,
            unit,
            time,
            resolved_survey,
            survey_design,
            non_absorbing=self.non_absorbing,
        )

        # Non-absorbing (general assignment) is a paper-supported point estimator
        # (Athey et al. 2025 Eq. 12 / Algorithm 2) but the formal triple-robustness
        # guarantee (Theorem 5.1) is proven only under block assignment, and the
        # bootstrap's validity (Algorithm 3) requires a growing number of treated
        # units. Surface that caveat once per fit so users do not over-read the SE.
        if self.non_absorbing:
            warnings.warn(
                "TROP(non_absorbing=True): treating the panel as a general "
                "(on/off) assignment pattern per Athey et al. (2025) Eq. 12 / "
                "Algorithm 2. This relies on the no-dynamic-effects (no carryover) "
                "assumption. The triple-robustness guarantee (Theorem 5.1) is "
                "proven only under block assignment, and bootstrap-SE validity "
                "requires a growing number of treated units -- interpret standard "
                "errors with care.",
                UserWarning,
                stacklevel=2,
            )

        n_units = _ctx["n_units"]
        n_periods = _ctx["n_periods"]
        idx_to_unit = _ctx["idx_to_unit"]
        idx_to_period = _ctx["idx_to_period"]
        unit_weight_arr = _ctx["unit_weight_arr"]
        Y = _ctx["Y"]
        D = _ctx["D"]
        missing_mask = _ctx["missing_mask"]
        n_treated_obs = _ctx["n_treated_obs"]
        treated_unit_idx = _ctx["treated_unit_idx"]
        control_unit_idx = _ctx["control_unit_idx"]
        n_pre_periods = _ctx["n_pre_periods"]
        n_post_periods = _ctx["n_post_periods"]

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
                warnings.warn(
                    f"Rust backend failed for LOOCV grid search; "
                    f"falling back to Python. Performance may be reduced. "
                    f"Error: {e}",
                    UserWarning,
                    stacklevel=2,
                )
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
        nonconverg_tracker: list = []
        n_fits_attempted = 0
        n_no_support = 0

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

            # Guard against a treated cell with no positively-weighted, observed
            # control support. Under non_absorbing with lambda_unit>0, a unit that
            # is never observed untreated has inf distance to every donor, so all
            # unit weights collapse to 0; the model then fits nothing and tau would
            # silently equal the raw outcome Y_it. Mark such cells non-estimable
            # (NaN) -- consistent with the missing-outcome NaN convention above --
            # rather than report a wrong effect. The cell-specific check also
            # covers lambda_unit=0 (uniform weights still leave an always-treated
            # unit's alpha_i unidentified) and a fully-treated period (beta_t
            # unidentified). It is a general correctness guard applied to every
            # local fit: a no-op when each treated cell's unit and period have an
            # observed control cell (always so on balanced panels, and in
            # absorbing mode unless an unbalanced panel leaves a unit's pre-period
            # controls or a period's controls entirely missing).
            if not _treated_cell_is_estimable(control_mask, Y, weight_matrix, i, t):
                treatment_effects[(unit_id, time_id)] = np.nan
                n_no_support += 1
                continue

            # Fit model with these weights
            n_fits_attempted += 1
            alpha_hat, beta_hat, L_hat = self._estimate_model(
                Y,
                control_mask,
                weight_matrix,
                lambda_nn,
                n_units,
                n_periods,
                _nonconvergence_tracker=nonconverg_tracker,
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

        if n_no_support > 0:
            warnings.warn(
                f"{n_no_support} of {n_treated_obs} treated cell(s) are not "
                f"estimable: the target unit and target period are not connected "
                f"in the observed-control graph, so the cell's unit/time fixed "
                f"effect (alpha_i + beta_t) is unidentified (e.g. an always-treated "
                f"unit, a period in which every unit is treated, or disconnected "
                f"control support). Their treatment effects are NaN and are "
                f"excluded from the ATT.",
                UserWarning,
                stacklevel=2,
            )

        if nonconverg_tracker:
            warn_if_not_converged(
                False,
                f"TROP local per-treated-observation fit: "
                f"{len(nonconverg_tracker)} of {n_fits_attempted} "
                f"fits did not converge",
                self.max_iter,
                self.tol,
            )

        # Count valid (estimable) treated observations. A cell is excluded when
        # its outcome is NaN/missing or it has no weighted control support (the
        # latter is additionally surfaced by the no-support warning above).
        n_valid_treated = len(tau_values)
        if n_valid_treated == 0:
            if n_no_support > 0:
                warnings.warn(
                    "No treated cells were estimable (for every treated cell the "
                    "target unit and target period are not connected in the "
                    "observed-control graph, leaving alpha_i + beta_t "
                    "unidentified). Cannot estimate ATT.",
                    UserWarning,
                )
            else:
                warnings.warn(
                    "All treated outcomes are NaN/missing. Cannot estimate ATT.",
                    UserWarning,
                )
        elif n_valid_treated < n_treated_obs:
            warnings.warn(
                f"Only {n_valid_treated} of {n_treated_obs} treated cells were "
                "estimable (finite outcome with weighted control support). "
                "df and n_treated_obs reflect estimable observations only.",
                UserWarning,
            )

        # Average ATT (survey-weighted when applicable). Guard the weighted path
        # against a zero total weight (e.g. the only estimable treated cells all
        # carry zero survey weight after non-estimable cells are excluded), which
        # would make np.average raise; fall back to NaN per the inference contract.
        if unit_weight_arr is not None and tau_values and float(np.sum(tau_weights)) > 0.0:
            att = float(np.average(tau_values, weights=tau_weights))
        elif tau_values and unit_weight_arr is None:
            att = float(np.mean(tau_values))
        else:
            att = np.nan

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
            # Force the guarded Python bootstrap (the Rust per-cell tau path lacks
            # the estimability guard) whenever a resample could need it: (a) the
            # point fit already trimmed a cell (n_no_support>0); or (b) the panel
            # is unbalanced (has missing cells) -- a bootstrap resample can then
            # lose a cell's only control support even if the original fit was
            # fully estimable, and Rust would contaminate that draw's SE. Balanced
            # panels keep the Rust happy path: the stratified resample always
            # re-draws the control stratum, so support is preserved.
            force_python=bool(n_no_support > 0 or np.any(missing_mask)),
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
            non_absorbing=self.non_absorbing,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # sklearn-like API
    # =========================================================================

    # get_params/set_params come from BaseEstimator.


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

    .. deprecated:: 3.9
        ``trop()`` is deprecated and will be removed in 4.0 (row M-073).
        Construct the estimator instead: ``TROP(...).fit(data, ...)``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column name.
    treatment : str
        Treatment indicator column name (0/1).

        By default (``non_absorbing=False``) this must be an ABSORBING STATE
        indicator, not a treatment timing indicator: for each unit, D=1 for ALL
        periods during and after treatment (D[t,i]=0 for t < g_i, D[t,i]=1 for
        t >= g_i where g_i is the treatment start time for unit i). Pass
        ``non_absorbing=True`` (via ``**kwargs``) to accept general on/off
        assignment patterns (``method='local'`` only); see ``TROP``.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    survey_design : SurveyDesign, optional
        Survey design specification. Supports pweight, strata, PSU, and FPC.
    **kwargs
        Additional arguments passed to TROP constructor (e.g. ``non_absorbing``).

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
    warnings.warn(
        "trop() is deprecated and will be removed in 4.0; "
        "construct the estimator instead: TROP(...).fit(data, ...).",
        FutureWarning,
        stacklevel=2,
    )
    estimator = TROP(**kwargs)
    return estimator.fit(data, outcome, treatment, unit, time, survey_design=survey_design)
